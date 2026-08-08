import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from app.dataplane.reverse.protocol.xai_console_video import (
    build_console_video_edit_payload,
    build_console_video_payload,
    edit_console_video,
    generate_console_video,
    parse_console_video_create,
    parse_console_video_status,
    trusted_console_video_url,
)
from app.dataplane.reverse.protocol import xai_console_video as console_video_protocol
from app.platform.errors import UpstreamError, ValidationError
from app.products.openai import video as video_service


class _FakeResponse:
    def __init__(self, status_code: int, body: bytes = b"{}") -> None:
        self.status_code = status_code
        self.content = body
        self.headers = {"content-type": "application/json"}


class _FakeSession:
    def __init__(self, post_responses, get_responses) -> None:
        self.post = AsyncMock(side_effect=post_responses)
        self.get = AsyncMock(side_effect=get_responses)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None


class ConsoleVideoPayloadTests(unittest.TestCase):
    def test_text_to_video_payload(self):
        payload = build_console_video_payload(
            prompt="night street",
            duration=6,
            aspect_ratio="16:9",
            resolution="720p",
        )
        self.assertEqual(payload, {
            "model": "grok-imagine-video",
            "duration": 6,
            "prompt": "night street",
            "aspect_ratio": "16:9",
            "resolution": "720p",
        })

    def test_image_to_video_payload_accepts_data_url(self):
        payload = build_console_video_payload(
            prompt="move slowly",
            duration=10,
            aspect_ratio="9:16",
            resolution="480p",
            image_url="data:image/png;base64,AAAA",
        )
        self.assertEqual(
            payload["image"],
            {"url": "data:image/png;base64,AAAA"},
        )

    def test_video_edit_payload(self):
        payload = build_console_video_edit_payload(
            prompt="make the person stand up",
            video_url="https://vidgen.x.ai/source.mp4",
        )
        self.assertEqual(payload, {
            "model": "grok-imagine-video",
            "prompt": "make the person stand up",
            "video": {"url": "https://vidgen.x.ai/source.mp4"},
        })

    def test_rejects_unsupported_duration(self):
        with self.assertRaises(ValidationError):
            build_console_video_payload(
                prompt="x",
                duration=16,
                aspect_ratio="16:9",
                resolution="720p",
            )

    def test_parse_create_and_status(self):
        self.assertEqual(parse_console_video_create(b'{"request_id":"req-1"}'), "req-1")
        result, done, progress = parse_console_video_status(
            {"status": "processing", "progress": 42}
        )
        self.assertIsNone(result)
        self.assertFalse(done)
        self.assertEqual(progress, 42)

        result, done, progress = parse_console_video_status({
            "status": "completed",
            "progress": 100,
            "video": {"url": "https://files.vidgen.x.ai/out.mp4"},
        })
        self.assertTrue(done)
        self.assertEqual(progress, 100)
        self.assertEqual(result.url, "https://files.vidgen.x.ai/out.mp4")

    def test_rejects_untrusted_output_url(self):
        with self.assertRaises(UpstreamError):
            parse_console_video_status({
                "status": "completed",
                "video": {"url": "https://example.com/out.mp4"},
            })
        self.assertTrue(trusted_console_video_url("https://vidgen.x.ai/out.mp4"))
        self.assertFalse(trusted_console_video_url("http://vidgen.x.ai/out.mp4"))

    def test_formats_content_moderation_error(self):
        message = console_video_protocol._console_api_error_message(
            400,
            b'{"code":"imagine:content-moderated","error":"rejected"}',
        )
        self.assertEqual(
            message,
            "Generated video rejected by content moderation",
        )


class ConsoleVideoTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_refreshes_dpop_then_polls_to_completion(self):
        fake_session = _FakeSession(
            post_responses=[
                _FakeResponse(401),
                _FakeResponse(200, b'{"request_id":"req/1"}'),
            ],
            get_responses=[
                _FakeResponse(200, b'{"status":"processing","progress":50}'),
                _FakeResponse(
                    200,
                    b'{"status":"completed","progress":100,"video":{"url":"https://cdn.vidgen.x.ai/out.mp4"}}',
                ),
            ],
        )
        proxy = SimpleNamespace(
            acquire=AsyncMock(return_value=None),
            feedback=AsyncMock(),
        )
        dpop_session = SimpleNamespace(access_token="access-token")
        manager = SimpleNamespace(
            get=AsyncMock(return_value=(dpop_session, "cache-key")),
            invalidate=Mock(),
        )
        progress = []

        async def record_progress(value: int) -> None:
            progress.append(value)

        def add_auth(headers, _session, *, method, url):
            headers["Authorization"] = "DPoP access-token"
            headers["DPoP"] = f"{method}:{url}"

        with (
            patch(
                "app.dataplane.reverse.protocol.xai_console_video.get_proxy_runtime",
                AsyncMock(return_value=proxy),
            ),
            patch(
                "app.dataplane.reverse.protocol.xai_console_video.ResettableSession",
                return_value=fake_session,
            ),
            patch(
                "app.dataplane.reverse.protocol.xai_console_video.build_session_kwargs",
                return_value={},
            ),
            patch(
                "app.dataplane.reverse.protocol.xai_console_video.build_console_headers",
                return_value={},
            ),
            patch(
                "app.dataplane.reverse.protocol.xai_console_video.apply_dpop_headers",
                side_effect=add_auth,
            ),
            patch(
                "app.dataplane.reverse.protocol.xai_console_video.dpop_sessions",
                manager,
            ),
            patch(
                "app.dataplane.reverse.protocol.xai_console_video.asyncio.sleep",
                AsyncMock(),
            ),
        ):
            result = await generate_console_video(
                "sso-token",
                {"model": "grok-imagine-video", "duration": 6, "prompt": "hi"},
                timeout_s=30,
                progress_cb=record_progress,
            )

        self.assertEqual(result.url, "https://cdn.vidgen.x.ai/out.mp4")
        self.assertEqual(fake_session.post.await_count, 2)
        self.assertEqual(fake_session.get.await_count, 2)
        manager.invalidate.assert_called_once_with("cache-key", "access-token")
        self.assertEqual(progress, [1, 50, 100])
        self.assertTrue(manager.get.await_args_list[1].kwargs["force"])
        self.assertIn("req%2F1", fake_session.get.await_args_list[0].args[0])

    async def test_video_edit_uses_edit_endpoint(self):
        fake_session = _FakeSession(
            post_responses=[_FakeResponse(200, b'{"request_id":"edit-1"}')],
            get_responses=[
                _FakeResponse(
                    200,
                    b'{"status":"completed","progress":100,"video":{"url":"https://vidgen.x.ai/edited.mp4"}}',
                ),
            ],
        )
        proxy = SimpleNamespace(acquire=AsyncMock(return_value=None), feedback=AsyncMock())
        dpop_session = SimpleNamespace(access_token="access-token")
        manager = SimpleNamespace(
            get=AsyncMock(return_value=(dpop_session, "cache-key")),
            invalidate=Mock(),
        )

        with (
            patch(
                "app.dataplane.reverse.protocol.xai_console_video.get_proxy_runtime",
                AsyncMock(return_value=proxy),
            ),
            patch(
                "app.dataplane.reverse.protocol.xai_console_video.ResettableSession",
                return_value=fake_session,
            ),
            patch(
                "app.dataplane.reverse.protocol.xai_console_video.build_session_kwargs",
                return_value={},
            ),
            patch(
                "app.dataplane.reverse.protocol.xai_console_video.build_console_headers",
                return_value={},
            ),
            patch(
                "app.dataplane.reverse.protocol.xai_console_video.apply_dpop_headers",
            ),
            patch(
                "app.dataplane.reverse.protocol.xai_console_video.dpop_sessions",
                manager,
            ),
        ):
            result = await edit_console_video(
                "sso-token",
                build_console_video_edit_payload(
                    prompt="stand up",
                    video_url="https://vidgen.x.ai/source.mp4",
                ),
                timeout_s=30,
            )

        self.assertEqual(result.url, "https://vidgen.x.ai/edited.mp4")
        self.assertEqual(
            fake_session.post.await_args.args[0],
            "https://console.x.ai/v1/videos/edits",
        )


class ConsoleVideoRoutingTests(unittest.IsolatedAsyncioTestCase):
    def test_extracts_structured_video_reference(self):
        prompt, images, video_url = video_service._extract_video_prompt_and_reference([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "make the person stand up"},
                    {
                        "type": "video_url",
                        "video_url": {"url": "https://vidgen.x.ai/source.mp4"},
                    },
                ],
            }
        ])
        self.assertEqual(prompt, "make the person stand up")
        self.assertIsNone(images)
        self.assertEqual(video_url, "https://vidgen.x.ai/source.mp4")

    async def test_supported_request_uses_console_protocol(self):
        artifact = video_service._VideoArtifact("https://vidgen.x.ai/a.mp4", "", "", "")
        console = AsyncMock(return_value=artifact)
        legacy = AsyncMock()
        with (
            patch.object(video_service, "_run_console_video_with_account", console),
            patch.object(video_service, "_run_video_with_account", legacy),
        ):
            result = await video_service._run_video_generation(
                model="grok-imagine-video",
                prompt="hello",
                aspect_ratio="16:9",
                resolution_name="720p",
                seconds=6,
            )
        self.assertIs(result, artifact)
        console.assert_awaited_once()
        legacy.assert_not_awaited()

    async def test_video_edit_uses_console_protocol(self):
        artifact = video_service._VideoArtifact("https://vidgen.x.ai/edited.mp4", "", "", "")
        console = AsyncMock(return_value=artifact)
        legacy = AsyncMock()
        with (
            patch.object(video_service, "_run_console_video_with_account", console),
            patch.object(video_service, "_run_video_with_account", legacy),
        ):
            result = await video_service._run_video_generation(
                model="grok-imagine-video",
                prompt="stand up",
                aspect_ratio="16:9",
                resolution_name="720p",
                seconds=6,
                video_reference_url="https://vidgen.x.ai/source.mp4",
            )
        self.assertIs(result, artifact)
        console.assert_awaited_once()
        legacy.assert_not_awaited()

    async def test_long_request_keeps_legacy_fallback(self):
        artifact = video_service._VideoArtifact("https://assets.grok.com/a.mp4", "", "", "")
        console = AsyncMock()
        legacy = AsyncMock(return_value=artifact)
        with (
            patch.object(video_service, "_run_console_video_with_account", console),
            patch.object(video_service, "_run_video_with_account", legacy),
        ):
            result = await video_service._run_video_generation(
                model="grok-imagine-video",
                prompt="hello",
                aspect_ratio="16:9",
                resolution_name="720p",
                seconds=16,
            )
        self.assertIs(result, artifact)
        legacy.assert_awaited_once()
        console.assert_not_awaited()

    async def test_429_switches_to_another_account(self):
        first = SimpleNamespace(token="token-1")
        second = SimpleNamespace(token="token-2")
        directory = SimpleNamespace(
            reserve_any=AsyncMock(side_effect=[first, second]),
            release=AsyncMock(),
            feedback=AsyncMock(),
        )
        runner = AsyncMock(side_effect=[
            UpstreamError("limited", status=429),
            "done",
        ])
        video_service._CONSOLE_VIDEO_COOLDOWN_UNTIL.clear()
        with patch("app.dataplane.account._directory", directory):
            result = await video_service._run_console_video_with_account(
                model="grok-imagine-video",
                runner=runner,
            )
        self.assertEqual(result, "done")
        self.assertEqual(runner.await_count, 2)
        self.assertEqual(directory.release.await_count, 2)
        directory.feedback.assert_not_awaited()
        self.assertIn("token-1", video_service._CONSOLE_VIDEO_COOLDOWN_UNTIL)

    async def test_created_job_failure_is_not_resubmitted(self):
        account = SimpleNamespace(token="token-1")
        directory = SimpleNamespace(
            reserve_any=AsyncMock(return_value=account),
            release=AsyncMock(),
            feedback=AsyncMock(),
        )
        error = UpstreamError("poll timed out", status=504)
        error.details["video_request_id"] = "request-1"
        runner = AsyncMock(side_effect=error)

        with patch("app.dataplane.account._directory", directory):
            with self.assertRaises(UpstreamError):
                await video_service._run_console_video_with_account(
                    model="grok-imagine-video",
                    runner=runner,
                )

        runner.assert_awaited_once()
        directory.reserve_any.assert_awaited_once()
        directory.release.assert_awaited_once()

    async def test_stream_sends_heartbeat_while_progress_is_unchanged(self):
        artifact = video_service._VideoArtifact(
            "https://vidgen.x.ai/a.mp4", "token", "", ""
        )

        async def generate(**kwargs):
            await kwargs["progress_cb"](94)
            await asyncio.sleep(0.04)
            return artifact

        with (
            patch.object(video_service, "_run_video_generation", side_effect=generate),
            patch.object(
                video_service,
                "_resolve_video_output",
                AsyncMock(return_value="https://vidgen.x.ai/a.mp4"),
            ),
            patch.object(video_service, "_VIDEO_HEARTBEAT_INTERVAL_S", 0.01),
        ):
            stream = await video_service.completions(
                model="grok-imagine-video",
                messages=[{"role": "user", "content": "hello"}],
                stream=True,
                seconds=6,
            )
            chunks = [chunk async for chunk in stream]

        self.assertIn(": heartbeat\n\n", chunks)
        self.assertTrue(any('"media_reference"' in chunk for chunk in chunks))
        self.assertTrue(any('"finish_reason":"stop"' in chunk for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
