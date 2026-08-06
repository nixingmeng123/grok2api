"""Console video generation over the DPoP-authenticated xAI API."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from urllib.parse import quote, urlparse

import orjson

from app.control.proxy.models import ProxyFeedback, ProxyFeedbackKind
from app.dataplane.proxy import get_proxy_runtime
from app.dataplane.proxy.adapters.headers import build_console_headers
from app.dataplane.proxy.adapters.session import ResettableSession, build_session_kwargs
from app.dataplane.reverse.protocol.xai_console_dpop import (
    apply_dpop_headers,
    dpop_sessions,
)
from app.dataplane.reverse.runtime.endpoint_table import (
    CONSOLE_DPOP_TOKEN,
    CONSOLE_VIDEO_GENERATIONS,
    CONSOLE_VIDEOS,
)
from app.platform.errors import UpstreamError, ValidationError


_POLL_INTERVAL_S = 2.0
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_TRUSTED_VIDEO_HOST = "vidgen.x.ai"


@dataclass(frozen=True, slots=True)
class ConsoleVideoResult:
    url: str


def build_console_video_payload(
    *,
    prompt: str,
    duration: int,
    aspect_ratio: str,
    resolution: str,
    image_url: str = "",
) -> dict[str, Any]:
    if duration < 1 or duration > 15:
        raise ValidationError("Console video duration must be between 1 and 15 seconds")
    if resolution not in {"480p", "720p"}:
        raise ValidationError("Console video resolution must be 480p or 720p")

    payload: dict[str, Any] = {
        "model": "grok-imagine-video",
        "duration": duration,
    }
    cleaned_prompt = prompt.strip()
    if cleaned_prompt:
        payload["prompt"] = cleaned_prompt
    if aspect_ratio.strip():
        payload["aspect_ratio"] = aspect_ratio.strip()
    if resolution.strip():
        payload["resolution"] = resolution.strip()
    if image_url:
        if not valid_console_image_url(image_url):
            raise ValidationError(
                "Console video reference must be an HTTPS URL or image data URL",
                param="input_reference.image_url",
            )
        payload["image"] = {"url": image_url}
    if "prompt" not in payload and "image" not in payload:
        raise ValidationError("Video generation requires a prompt or reference image")
    return payload


def valid_console_image_url(value: str) -> bool:
    normalized = value.strip()
    lowered = normalized.lower()
    if lowered.startswith("data:image/"):
        return ";base64," in lowered
    parsed = urlparse(normalized)
    return parsed.scheme == "https" and bool(parsed.netloc) and parsed.username is None


def parse_console_video_create(data: bytes | str | dict[str, Any]) -> str:
    payload = _decode_object(data, context="create")
    request_id = str(payload.get("request_id") or "").strip()
    if not request_id:
        raise UpstreamError("Console video create response has no request_id", status=502)
    return request_id


def parse_console_video_status(
    data: bytes | str | dict[str, Any],
) -> tuple[ConsoleVideoResult | None, bool, int]:
    payload = _decode_object(data, context="status")
    status = str(payload.get("status") or "").strip().lower()
    try:
        progress = max(0, min(99, int(payload.get("progress") or 0)))
    except (TypeError, ValueError):
        progress = 0

    if status in {"done", "completed", "succeeded", "success", "ready"}:
        video = payload.get("video")
        url = str(video.get("url") or "").strip() if isinstance(video, dict) else ""
        if not url:
            raise UpstreamError("Console video completed without a content URL", status=502)
        if not trusted_console_video_url(url):
            raise UpstreamError("Console video returned an untrusted content URL", status=502)
        return ConsoleVideoResult(url=url), True, 100

    if status in {"failed", "expired", "cancelled", "canceled", "error"}:
        message = _safe_error_message(payload.get("error")) or status
        raise UpstreamError(f"Console video generation failed: {message}", status=502)

    if status in {"pending", "processing", "in_progress", "queued"}:
        return None, False, progress
    raise UpstreamError(f"Console video returned invalid status: {status!r}", status=502)


async def generate_console_video(
    token: str,
    payload: dict[str, Any],
    *,
    timeout_s: float,
    progress_cb: Callable[[int], Awaitable[None]] | None = None,
) -> ConsoleVideoResult:
    deadline = asyncio.get_running_loop().time() + timeout_s
    created = await _request_json(
        token,
        method="POST",
        url=CONSOLE_VIDEO_GENERATIONS,
        payload=payload,
        timeout_s=_remaining_timeout(deadline),
    )
    request_id = parse_console_video_create(created)
    if progress_cb is not None:
        await progress_cb(1)

    status_url = f"{CONSOLE_VIDEOS}/{quote(request_id, safe='')}"
    try:
        while True:
            status_data = await _request_json(
                token,
                method="GET",
                url=status_url,
                payload=None,
                timeout_s=_remaining_timeout(deadline),
            )
            result, done, progress = parse_console_video_status(status_data)
            if progress_cb is not None and progress > 0:
                await progress_cb(progress)
            if done and result is not None:
                return result
            await asyncio.sleep(min(_POLL_INTERVAL_S, _remaining_timeout(deadline)))
    except UpstreamError as exc:
        # A request id means the upstream job already exists. Callers must not
        # retry this failure by submitting another billable generation.
        exc.details["video_request_id"] = request_id
        raise


async def download_console_video(url: str, *, timeout_s: float) -> tuple[bytes, str]:
    if not trusted_console_video_url(url):
        raise UpstreamError("Console video content URL is not trusted", status=502)
    proxy = await get_proxy_runtime()
    lease = await proxy.acquire()
    kwargs = build_session_kwargs(lease=lease)
    try:
        async with ResettableSession(**kwargs) as session:
            response = await session.get(
                url,
                headers={"Accept": "video/*,*/*;q=0.8"},
                timeout=timeout_s,
            )
    except Exception as exc:
        await proxy.feedback(
            lease,
            ProxyFeedback(kind=ProxyFeedbackKind.TRANSPORT_ERROR),
        )
        if isinstance(exc, UpstreamError):
            raise
        raise UpstreamError(f"Console video download failed: {exc}", status=502) from exc

    if response.status_code < 200 or response.status_code >= 300:
        await proxy.feedback(lease, _status_feedback(response.status_code))
        raise UpstreamError(
            f"Console video download returned {response.status_code}",
            status=response.status_code,
        )
    raw = bytes(response.content or b"")
    content_type = str(response.headers.get("content-type") or "video/mp4").split(";", 1)[0]
    if not raw or not content_type.lower().startswith("video/"):
        await proxy.feedback(
            lease,
            ProxyFeedback(kind=ProxyFeedbackKind.TRANSPORT_ERROR),
        )
        raise UpstreamError("Console video download returned invalid content", status=502)
    await proxy.feedback(
        lease,
        ProxyFeedback(kind=ProxyFeedbackKind.SUCCESS, status_code=200),
    )
    return raw, content_type


async def _request_json(
    token: str,
    *,
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    timeout_s: float,
) -> bytes:
    proxy = await get_proxy_runtime()
    lease = await proxy.acquire()
    kwargs = build_session_kwargs(lease=lease)
    body = orjson.dumps(payload) if payload is not None else None

    async with ResettableSession(**kwargs) as session:
        response = None
        for auth_attempt in range(2):
            try:
                dpop_session, cache_key = await dpop_sessions.get(
                    sso_token=token,
                    lease=lease,
                    http_session=session,
                    token_endpoint=CONSOLE_DPOP_TOKEN,
                    timeout_s=timeout_s,
                    force=auth_attempt > 0,
                )
                headers = build_console_headers(token, lease=lease)
                apply_dpop_headers(headers, dpop_session, method=method, url=url)
                request = session.post if method == "POST" else session.get
                request_kwargs: dict[str, Any] = {
                    "headers": headers,
                    "timeout": timeout_s,
                }
                if body is not None:
                    request_kwargs["data"] = body
                response = await request(url, **request_kwargs)
            except Exception as exc:
                await proxy.feedback(
                    lease,
                    ProxyFeedback(kind=ProxyFeedbackKind.TRANSPORT_ERROR),
                )
                if isinstance(exc, UpstreamError):
                    raise
                raise UpstreamError(f"Console video transport failed: {exc}", status=502) from exc

            if response.status_code != 401 or auth_attempt > 0:
                break
            dpop_sessions.invalidate(cache_key, dpop_session.access_token)

        if response is None:
            raise UpstreamError("Console video DPoP request failed", status=502)
        data = bytes(response.content or b"")
        if len(data) > _MAX_RESPONSE_BYTES:
            raise UpstreamError("Console video response exceeds 2 MiB", status=502)
        if response.status_code < 200 or response.status_code >= 300:
            await proxy.feedback(lease, _status_feedback(response.status_code))
            raise UpstreamError(
                _console_api_error_message(response.status_code, data),
                status=response.status_code,
                body=data.decode("utf-8", "replace")[:400],
            )
        await proxy.feedback(
            lease,
            ProxyFeedback(kind=ProxyFeedbackKind.SUCCESS, status_code=200),
        )
        return data


def trusted_console_video_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    host = (parsed.hostname or "").lower()
    return (
        parsed.scheme == "https"
        and parsed.username is None
        and (host == _TRUSTED_VIDEO_HOST or host.endswith(f".{_TRUSTED_VIDEO_HOST}"))
    )


def _decode_object(data: bytes | str | dict[str, Any], *, context: str) -> dict[str, Any]:
    if isinstance(data, dict):
        return data
    try:
        parsed = orjson.loads(data)
    except orjson.JSONDecodeError as exc:
        raise UpstreamError(f"Console video {context} response is invalid JSON", status=502) from exc
    if not isinstance(parsed, dict):
        raise UpstreamError(f"Console video {context} response is not an object", status=502)
    return parsed


def _safe_error_message(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("message", "code", "type"):
            if value.get(key):
                return str(value[key]).strip()[:160]
    if isinstance(value, str):
        return value.strip()[:160]
    return ""


def _console_api_error_message(status: int, data: bytes) -> str:
    try:
        payload = orjson.loads(data)
    except orjson.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        code = str(payload.get("code") or "").strip()
        message = _safe_error_message(payload.get("error"))
        if code == "imagine:content-moderated":
            return "Generated video rejected by content moderation"
        if message:
            return f"Console video API error: {message}"
    return f"Console video API returned {status}"


def _remaining_timeout(deadline: float) -> float:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise UpstreamError("Console video generation timed out", status=504)
    return remaining


def _status_feedback(status: int) -> ProxyFeedback:
    if status == 403:
        kind = ProxyFeedbackKind.CHALLENGE
    elif status == 429:
        kind = ProxyFeedbackKind.RATE_LIMITED
    elif status >= 500:
        kind = ProxyFeedbackKind.UPSTREAM_5XX
    else:
        kind = ProxyFeedbackKind.FORBIDDEN
    return ProxyFeedback(kind=kind, status_code=status)


__all__ = [
    "ConsoleVideoResult",
    "build_console_video_payload",
    "download_console_video",
    "generate_console_video",
    "parse_console_video_create",
    "parse_console_video_status",
    "trusted_console_video_url",
    "valid_console_image_url",
]
