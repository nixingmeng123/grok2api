import base64
import hashlib
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import orjson
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from app.dataplane.proxy.adapters.headers import build_console_headers
from app.dataplane.reverse.protocol.xai_console_dpop import (
    DPoPSession,
    create_dpop_proof,
    dpop_htu,
    fetch_dpop_session,
    jwk_thumbprint,
    parse_access_token,
    public_jwk,
)
from app.dataplane.reverse.protocol.xai_console_chat import stream_console_chat


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _decode_segment(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _access_token(*, expires_at: int, thumbprint: str) -> str:
    header = _b64url(b'{"alg":"ES256"}')
    claims = _b64url(orjson.dumps({
        "exp": expires_at,
        "cnf": {"jkt": thumbprint},
    }))
    return f"{header}.{claims}.signature"


class ConsoleDPoPTests(unittest.TestCase):
    def setUp(self):
        self.private_key = ec.generate_private_key(ec.SECP256R1())
        self.jwk = public_jwk(self.private_key)
        self.access_token = _access_token(
            expires_at=2_000_000_000,
            thumbprint=jwk_thumbprint(self.jwk),
        )
        self.session = DPoPSession(
            access_token=self.access_token,
            private_key=self.private_key,
            public_jwk=self.jwk,
            expires_at=2_000_000_000,
        )

    def test_public_jwk_and_access_token_binding(self):
        self.assertEqual(self.jwk["kty"], "EC")
        self.assertEqual(self.jwk["crv"], "P-256")
        self.assertEqual(len(_decode_segment(self.jwk["x"])), 32)
        self.assertEqual(len(_decode_segment(self.jwk["y"])), 32)

        expires_at, thumbprint = parse_access_token(self.access_token)
        self.assertEqual(expires_at, 2_000_000_000)
        self.assertEqual(thumbprint, jwk_thumbprint(self.jwk))

    def test_proof_claims_and_signature(self):
        proof = create_dpop_proof(
            self.session,
            method="post",
            url="https://console.x.ai/v1/responses?ignored=true",
            now=1_900_000_000,
            jti="fixed-jti",
        )
        encoded_header, encoded_claims, encoded_signature = proof.split(".")
        header = json.loads(_decode_segment(encoded_header))
        claims = json.loads(_decode_segment(encoded_claims))

        self.assertEqual(header["alg"], "ES256")
        self.assertEqual(header["typ"], "dpop+jwt")
        self.assertEqual(header["jwk"], self.jwk)
        self.assertEqual(claims["jti"], "fixed-jti")
        self.assertEqual(claims["htm"], "POST")
        self.assertEqual(claims["htu"], "https://console.x.ai/v1/responses")
        self.assertEqual(claims["iat"], 1_900_000_000)
        expected_ath = _b64url(hashlib.sha256(self.access_token.encode()).digest())
        self.assertEqual(claims["ath"], expected_ath)

        raw_signature = _decode_segment(encoded_signature)
        self.assertEqual(len(raw_signature), 64)
        r = int.from_bytes(raw_signature[:32], "big")
        s = int.from_bytes(raw_signature[32:], "big")
        der_signature = encode_dss_signature(r, s)
        signing_input = f"{encoded_header}.{encoded_claims}".encode("ascii")
        self.private_key.public_key().verify(
            der_signature,
            signing_input,
            ec.ECDSA(hashes.SHA256()),
        )

    def test_htu_excludes_query_and_fragment(self):
        self.assertEqual(
            dpop_htu("https://console.x.ai/v1/responses?a=1#fragment"),
            "https://console.x.ai/v1/responses",
        )

    def test_console_headers_no_longer_use_anonymous_bearer(self):
        profile = SimpleNamespace(browser="", user_agent="", cf_clearance="")
        with patch(
            "app.dataplane.proxy.adapters.headers._resolve_profile",
            return_value=profile,
        ):
            token_headers = build_console_headers("sso-value", include_cluster=False)
            api_headers = build_console_headers("sso-value")

        self.assertNotIn("Authorization", token_headers)
        self.assertNotIn("DPoP", token_headers)
        self.assertNotIn("x-cluster", token_headers)
        self.assertEqual(api_headers["x-cluster"], "https://us-east-1.api.x.ai")
        self.assertIn("sso=sso-value", token_headers["Cookie"])


class ConsoleDPoPFetchTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetches_and_validates_bound_access_token(self):
        private_key = ec.generate_private_key(ec.SECP256R1())
        jwk = public_jwk(private_key)
        expires_at = int(time.time()) + 300
        access_token = _access_token(
            expires_at=expires_at,
            thumbprint=jwk_thumbprint(jwk),
        )
        response = SimpleNamespace(
            status_code=200,
            content=orjson.dumps({
                "access_token": access_token,
                "token_type": "DPoP",
                "expires_in": 300,
            }),
        )
        http_session = SimpleNamespace(post=AsyncMock(return_value=response))

        with (
            patch(
                "app.dataplane.reverse.protocol.xai_console_dpop.ec.generate_private_key",
                return_value=private_key,
            ),
            patch(
                "app.dataplane.reverse.protocol.xai_console_dpop.build_console_headers",
                return_value={"Cookie": "sso=value"},
            ),
        ):
            session = await fetch_dpop_session(
                sso_token="value",
                lease=None,
                http_session=http_session,
                token_endpoint="https://console.x.ai/v1/dpop/token",
                timeout_s=30,
            )

        self.assertEqual(session.access_token, access_token)
        self.assertEqual(session.public_jwk, jwk)
        call = http_session.post.await_args
        self.assertEqual(call.args[0], "https://console.x.ai/v1/dpop/token")
        self.assertEqual(orjson.loads(call.kwargs["data"]), {"jwk": jwk})
        self.assertNotIn("x-cluster", call.kwargs["headers"])


class _FakeStreamResponse:
    def __init__(self, status_code: int, lines: list[str] | None = None):
        self.status_code = status_code
        self.content = b""
        self.lines = lines or []
        self.drained = False

    async def aiter_content(self):
        self.drained = True
        yield b"rejected"

    async def aiter_lines(self):
        for line in self.lines:
            yield line


class _FakeHTTPContext:
    def __init__(self, responses: list[_FakeStreamResponse]):
        self.post = AsyncMock(side_effect=responses)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None


class ConsoleDPoPStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_401_refreshes_dpop_session_once(self):
        private_key = ec.generate_private_key(ec.SECP256R1())
        jwk = public_jwk(private_key)
        session = DPoPSession(
            access_token="access-token",
            private_key=private_key,
            public_jwk=jwk,
            expires_at=time.time() + 300,
        )
        rejected = _FakeStreamResponse(401)
        accepted = _FakeStreamResponse(200, [
            "event: response.completed",
            'data: {"response":{"status":"completed"}}',
            "data: [DONE]",
        ])
        http_context = _FakeHTTPContext([rejected, accepted])
        proxy = SimpleNamespace(
            acquire=AsyncMock(return_value=None),
            feedback=AsyncMock(),
        )
        manager = SimpleNamespace(
            get=AsyncMock(return_value=(session, "cache-key")),
            invalidate=Mock(),
        )

        with (
            patch(
                "app.dataplane.proxy.get_proxy_runtime",
                AsyncMock(return_value=proxy),
            ),
            patch(
                "app.dataplane.proxy.adapters.session.build_session_kwargs",
                return_value={},
            ),
            patch(
                "app.dataplane.proxy.adapters.session.ResettableSession",
                return_value=http_context,
            ),
            patch(
                "app.dataplane.proxy.adapters.headers.build_console_headers",
                return_value={},
            ),
            patch(
                "app.dataplane.reverse.protocol.xai_console_dpop.dpop_sessions",
                manager,
            ),
        ):
            events = [
                event
                async for event in stream_console_chat("sso", {"input": []})
            ]

        self.assertTrue(rejected.drained)
        self.assertEqual(http_context.post.await_count, 2)
        self.assertEqual(manager.get.await_count, 2)
        self.assertFalse(manager.get.await_args_list[0].kwargs["force"])
        self.assertTrue(manager.get.await_args_list[1].kwargs["force"])
        manager.invalidate.assert_called_once_with("cache-key", "access-token")
        self.assertEqual(events[0][0], "response.completed")
        for call in http_context.post.await_args_list:
            self.assertEqual(call.kwargs["headers"]["Authorization"], "DPoP access-token")
            self.assertIn("DPoP", call.kwargs["headers"])


if __name__ == "__main__":
    unittest.main()
