"""DPoP authentication for console.x.ai requests."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import orjson
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

from app.control.proxy.models import ProxyLease
from app.dataplane.proxy.adapters.headers import build_console_headers
from app.dataplane.proxy.adapters.profile import resolve_proxy_profile
from app.platform.errors import UpstreamError


_CACHE_LIMIT = 4096
_REFRESH_SKEW_S = 20
_MAX_TOKEN_LIFETIME_S = 3600


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


@dataclass(frozen=True)
class DPoPSession:
    access_token: str
    private_key: ec.EllipticCurvePrivateKey
    public_jwk: dict[str, str]
    expires_at: float


def public_jwk(private_key: ec.EllipticCurvePrivateKey) -> dict[str, str]:
    numbers = private_key.public_key().public_numbers()
    return {
        "kty": "EC",
        "crv": "P-256",
        "x": _b64url(numbers.x.to_bytes(32, "big")),
        "y": _b64url(numbers.y.to_bytes(32, "big")),
    }


def jwk_thumbprint(jwk: dict[str, str]) -> str:
    canonical = {
        "crv": jwk["crv"],
        "kty": jwk["kty"],
        "x": jwk["x"],
        "y": jwk["y"],
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return _b64url(hashlib.sha256(encoded).digest())


def parse_access_token(value: str) -> tuple[float, str]:
    parts = value.split(".")
    if len(parts) != 3:
        raise ValueError("Console DPoP access token format is invalid")
    try:
        claims = orjson.loads(_b64url_decode(parts[1]))
        expires_at = float(claims["exp"])
        thumbprint = str(claims["cnf"]["jkt"]).strip()
    except (KeyError, TypeError, ValueError, orjson.JSONDecodeError) as exc:
        raise ValueError("Console DPoP access token claims are invalid") from exc
    if expires_at <= 0 or not thumbprint:
        raise ValueError("Console DPoP access token claims are invalid")
    return expires_at, thumbprint


def dpop_htu(url: str) -> str:
    parsed = urlsplit(url)
    path = parsed.path or "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def create_dpop_proof(
    session: DPoPSession,
    *,
    method: str,
    url: str,
    now: int | None = None,
    jti: str | None = None,
) -> str:
    header = {
        "alg": "ES256",
        "typ": "dpop+jwt",
        "jwk": session.public_jwk,
    }
    claims = {
        "jti": jti or str(uuid.uuid4()),
        "htm": method.upper(),
        "htu": dpop_htu(url),
        "iat": int(time.time()) if now is None else int(now),
        "ath": _b64url(hashlib.sha256(session.access_token.encode()).digest()),
    }
    encoded_header = _b64url(orjson.dumps(header))
    encoded_claims = _b64url(orjson.dumps(claims))
    signing_input = f"{encoded_header}.{encoded_claims}".encode("ascii")
    der_signature = session.private_key.sign(signing_input, ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(der_signature)
    signature = _b64url(r.to_bytes(32, "big") + s.to_bytes(32, "big"))
    return f"{encoded_header}.{encoded_claims}.{signature}"


def apply_dpop_headers(
    headers: dict[str, str],
    session: DPoPSession,
    *,
    method: str,
    url: str,
) -> None:
    headers["Authorization"] = f"DPoP {session.access_token}"
    headers["DPoP"] = create_dpop_proof(session, method=method, url=url)


class DPoPSessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, DPoPSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def cache_key(self, sso_token: str, lease: ProxyLease | None) -> str:
        token = sso_token[4:] if sso_token.startswith("sso=") else sso_token
        profile = resolve_proxy_profile(lease)
        identity = "|".join((
            hashlib.sha256(token.encode()).hexdigest(),
            lease.proxy_url if lease and lease.proxy_url else "direct",
            profile.user_agent,
        ))
        return hashlib.sha256(identity.encode()).hexdigest()

    def cached(self, key: str, *, now: float | None = None) -> DPoPSession | None:
        current_time = time.time() if now is None else now
        session = self._sessions.get(key)
        if session is None:
            return None
        if session.expires_at <= current_time + _REFRESH_SKEW_S:
            self._sessions.pop(key, None)
            return None
        return session

    def invalidate(self, key: str, access_token: str = "") -> None:
        current = self._sessions.get(key)
        if current is None:
            return
        if access_token and current.access_token != access_token:
            return
        self._sessions.pop(key, None)

    async def get(
        self,
        *,
        sso_token: str,
        lease: ProxyLease | None,
        http_session: Any,
        token_endpoint: str,
        timeout_s: float,
        force: bool = False,
    ) -> tuple[DPoPSession, str]:
        key = self.cache_key(sso_token, lease)
        if force:
            self.invalidate(key)
        cached = self.cached(key)
        if cached is not None:
            return cached, key

        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = self.cached(key)
            if cached is not None:
                return cached, key
            session = await fetch_dpop_session(
                sso_token=sso_token,
                lease=lease,
                http_session=http_session,
                token_endpoint=token_endpoint,
                timeout_s=timeout_s,
            )
            self._store(key, session)
            return session, key

    def _store(self, key: str, session: DPoPSession) -> None:
        now = time.time()
        for cached_key, cached in list(self._sessions.items()):
            if cached.expires_at <= now + _REFRESH_SKEW_S:
                self._sessions.pop(cached_key, None)
        if key not in self._sessions and len(self._sessions) >= _CACHE_LIMIT:
            oldest_key = min(
                self._sessions,
                key=lambda item: self._sessions[item].expires_at,
            )
            self._sessions.pop(oldest_key, None)
        self._sessions[key] = session


async def fetch_dpop_session(
    *,
    sso_token: str,
    lease: ProxyLease | None,
    http_session: Any,
    token_endpoint: str,
    timeout_s: float,
) -> DPoPSession:
    private_key = ec.generate_private_key(ec.SECP256R1())
    jwk = public_jwk(private_key)
    headers = build_console_headers(
        sso_token,
        lease=lease,
        include_cluster=False,
    )
    response = await http_session.post(
        token_endpoint,
        headers=headers,
        data=orjson.dumps({"jwk": jwk}),
        timeout=timeout_s,
    )
    if response.status_code < 200 or response.status_code >= 300:
        try:
            body = response.content.decode("utf-8", "replace")[:400]
        except Exception:
            body = ""
        raise UpstreamError(
            f"Console DPoP token API returned {response.status_code}",
            status=response.status_code,
            body=body,
        )

    try:
        data = orjson.loads(response.content)
        access_token = str(data["access_token"]).strip()
        token_type = str(data["token_type"]).strip()
        expires_in = int(data["expires_in"])
    except (KeyError, TypeError, ValueError, orjson.JSONDecodeError) as exc:
        raise UpstreamError("Console DPoP token response is invalid", status=502) from exc

    if not access_token or token_type.lower() != "dpop":
        raise UpstreamError("Console DPoP token response is invalid", status=502)
    if expires_in <= 0 or expires_in > _MAX_TOKEN_LIFETIME_S:
        raise UpstreamError("Console DPoP token lifetime is invalid", status=502)

    try:
        token_expiry, token_thumbprint = parse_access_token(access_token)
    except ValueError as exc:
        raise UpstreamError(str(exc), status=502) from exc
    if token_thumbprint != jwk_thumbprint(jwk):
        raise UpstreamError("Console DPoP token key binding is invalid", status=502)

    now = time.time()
    expires_at = min(now + expires_in, token_expiry)
    if expires_at <= now + _REFRESH_SKEW_S:
        raise UpstreamError("Console DPoP token is expired", status=502)
    return DPoPSession(
        access_token=access_token,
        private_key=private_key,
        public_jwk=jwk,
        expires_at=expires_at,
    )


dpop_sessions = DPoPSessionManager()


__all__ = [
    "DPoPSession",
    "DPoPSessionManager",
    "apply_dpop_headers",
    "create_dpop_proof",
    "dpop_htu",
    "dpop_sessions",
    "fetch_dpop_session",
    "jwk_thumbprint",
    "parse_access_token",
    "public_jwk",
]
