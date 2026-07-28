"""Console Messages API handler — /v1/messages for console.x.ai models.

将 console.x.ai 上游 SSE 转换为 Anthropic Messages API 格式输出。
"""

import asyncio
from typing import Any, AsyncGenerator

import orjson

from app.platform.logging.logger import logger
from app.platform.config.snapshot import get_config
from app.platform.errors import RateLimitError, UpstreamError
from app.platform.runtime.clock import now_s
from app.platform.tokens import estimate_prompt_tokens, estimate_tokens, estimate_tool_call_tokens
from app.control.account.enums import FeedbackKind
from app.control.account.invalid_credentials import feedback_kind_for_error
from app.control.account.runtime import get_refresh_service
from app.control.model.registry import resolve as resolve_model
from app.dataplane.account.selector import current_strategy
from app.dataplane.reverse.protocol.xai_console_chat import (
    build_console_payload,
    ConsoleStreamAdapter,
    stream_console_chat,
)
from app.dataplane.reverse.protocol.tool_parser import parse_tool_calls
from app.dataplane.reverse.protocol.tool_prompt import (
    build_tool_system_prompt,
    extract_tool_names,
    tool_calls_to_xml,
)
from app.products._account_selection import reserve_account, selection_max_retries
from app.products.openai.chat import _configured_retry_codes, _should_retry_upstream
from app.products.openai._tool_sieve import ToolSieve


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {orjson.dumps(data).decode()}\n\n"


def _log_task_exception(task: "asyncio.Task") -> None:
    exc = task.exception() if not task.cancelled() else None
    if exc:
        logger.warning("background task failed: task={} error={}", task.get_name(), exc)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text = block.get("text") or ""
                else:
                    text = str(block)
            else:
                text = str(block)
            if text:
                parts.append(text)
        return "\n".join(parts)
    return str(content)


def _prepare_console_messages(
    messages: list[dict],
    tools: list[dict] | None,
    tool_choice: Any,
) -> tuple[list[dict], list[str]]:
    tool_names = extract_tool_names(tools or []) if tools else []
    prepared: list[dict] = []

    if tools:
        prepared.append({
            "role": "system",
            "content": build_tool_system_prompt(tools, tool_choice),
        })

    for msg in messages:
        role = msg.get("role", "user")
        tool_calls = msg.get("tool_calls")

        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            label = f"[tool result for {tool_call_id}]" if tool_call_id else "[tool result]"
            text = _content_to_text(msg.get("content"))
            if text:
                prepared.append({"role": "user", "content": f"{label}:\n{text}"})
            continue

        if role == "assistant" and tool_calls:
            text = _content_to_text(msg.get("content")).strip()
            xml = tool_calls_to_xml(tool_calls)
            content = f"{text}\n{xml}" if text else xml
            prepared.append({"role": "assistant", "content": content})
            continue

        prepared.append(msg)

    return prepared, tool_names


def _tool_use_blocks(calls) -> list[dict]:
    content: list[dict] = []
    for call in calls:
        try:
            parsed_input = orjson.loads(call.arguments)
        except (orjson.JSONDecodeError, ValueError):
            parsed_input = {}
        content.append({
            "type": "tool_use",
            "id": call.call_id,
            "name": call.name,
            "input": parsed_input,
        })
    return content


async def _quota_sync(token: str, mode_id: int) -> None:
    """Fire-and-forget: 成功调用后持久化配额扣减和 usage_use_count。

    Console 配额(mode_id=5)为本地管理，不依赖上游 API，
    无论 random/quota 策略都需要执行扣减和窗口重置。
    """
    try:
        if current_strategy() != "quota" and mode_id != 5:
            return
        svc = get_refresh_service()
        if svc:
            await svc.refresh_call_async(token, mode_id)
    except Exception as exc:
        logger.warning(
            "console messages quota sync failed: token={}... mode_id={} error={}",
            token[:10],
            mode_id,
            exc,
        )


async def _fail_sync(token: str, mode_id: int, exc: BaseException | None = None) -> None:
    """Fire-and-forget: 失败后持久化失败计数。"""
    try:
        svc = get_refresh_service()
        if svc:
            await svc.record_failure_async(token, mode_id, exc)
    except Exception as e:
        logger.warning(
            "console messages fail sync error: token={}... mode_id={} error={}",
            token[:10],
            mode_id,
            e,
        )


async def create(
    *,
    model: str,
    messages: list[dict],
    stream: bool,
    emit_think: bool,
    temperature: float,
    top_p: float,
    msg_id: str,
    tools: list[dict] | None = None,
    tool_choice: Any = None,
) -> dict | AsyncGenerator[str, None]:
    """Console models /v1/messages handler (Anthropic format)."""

    cfg = get_config()
    spec = resolve_model(model)
    timeout_s = cfg.get_float("chat.timeout", 120.0)
    max_retries = selection_max_retries()
    retry_codes = _configured_retry_codes(cfg)
    effort = "low" if emit_think else "none"
    request_messages, tool_names = _prepare_console_messages(messages, tools, tool_choice)

    from app.dataplane.account import _directory as _acct_dir
    if _acct_dir is None:
        raise RateLimitError("Account directory not initialised")
    directory = _acct_dir

    # ── Streaming ─────────────────────────────────────────────────────────────
    if stream:
        async def _run_stream() -> AsyncGenerator[str, None]:
            excluded: list[str] = []
            for attempt in range(max_retries + 1):
                acct, selected_mode_id = await reserve_account(
                    directory, spec, now_s_override=now_s(),
                    exclude_tokens=excluded or None,
                )
                if acct is None:
                    raise RateLimitError("No available accounts for this model tier")

                token = acct.token
                success = False
                fail_exc: BaseException | None = None
                _retry = False
                adapter = ConsoleStreamAdapter()
                text_buf: list[str] = []
                sieve = ToolSieve(tool_names) if tool_names else None
                block_index = 0
                text_started = False
                tool_calls_emitted = False
                tool_output_tokens = 0

                try:
                    payload = build_console_payload(
                        messages=request_messages,
                        model=model,
                        temperature=temperature,
                        top_p=top_p,
                        reasoning_effort=effort,
                        stream=True,
                    )

                    try:
                        # message_start
                        yield _sse("message_start", {
                            "type": "message_start",
                            "message": {
                                "id": msg_id,
                                "type": "message",
                                "role": "assistant",
                                "model": model,
                                "content": [],
                                "stop_reason": None,
                                "usage": {"input_tokens": estimate_prompt_tokens(request_messages), "output_tokens": 0},
                            },
                        })
                        yield _sse("ping", {"type": "ping"})

                        yield ": heartbeat\n\n"
                        async for event_type, data in stream_console_chat(
                            token, payload, timeout_s=timeout_s
                        ):
                            tokens = adapter.feed(event_type, data)
                            for tok in tokens:
                                if sieve is not None:
                                    text_chunk, calls = sieve.feed(tok)
                                    if calls is not None:
                                        if text_started:
                                            yield _sse("content_block_stop", {
                                                "type": "content_block_stop",
                                                "index": block_index,
                                            })
                                            block_index += 1
                                            text_started = False
                                        for call in calls:
                                            yield _sse("content_block_start", {
                                                "type": "content_block_start",
                                                "index": block_index,
                                                "content_block": {
                                                    "type": "tool_use",
                                                    "id": call.call_id,
                                                    "name": call.name,
                                                    "input": {},
                                                },
                                            })
                                            yield _sse("content_block_delta", {
                                                "type": "content_block_delta",
                                                "index": block_index,
                                                "delta": {
                                                    "type": "input_json_delta",
                                                    "partial_json": call.arguments,
                                                },
                                            })
                                            yield _sse("content_block_stop", {
                                                "type": "content_block_stop",
                                                "index": block_index,
                                            })
                                            block_index += 1
                                        tool_output_tokens = estimate_tool_call_tokens(calls)
                                        tool_calls_emitted = True
                                        break
                                else:
                                    text_chunk = tok

                                if not text_chunk:
                                    continue
                                if not text_started:
                                    yield _sse("content_block_start", {
                                        "type": "content_block_start",
                                        "index": block_index,
                                        "content_block": {"type": "text", "text": ""},
                                    })
                                    text_started = True
                                text_buf.append(text_chunk)
                                yield _sse("content_block_delta", {
                                    "type": "content_block_delta",
                                    "index": block_index,
                                    "delta": {"type": "text_delta", "text": text_chunk},
                                })
                            if tool_calls_emitted:
                                break

                        if sieve is not None and not tool_calls_emitted:
                            calls = sieve.flush()
                            if calls:
                                if text_started:
                                    yield _sse("content_block_stop", {
                                        "type": "content_block_stop",
                                        "index": block_index,
                                    })
                                    block_index += 1
                                    text_started = False
                                for call in calls:
                                    yield _sse("content_block_start", {
                                        "type": "content_block_start",
                                        "index": block_index,
                                        "content_block": {
                                            "type": "tool_use",
                                            "id": call.call_id,
                                            "name": call.name,
                                            "input": {},
                                        },
                                    })
                                    yield _sse("content_block_delta", {
                                        "type": "content_block_delta",
                                        "index": block_index,
                                        "delta": {
                                            "type": "input_json_delta",
                                            "partial_json": call.arguments,
                                        },
                                    })
                                    yield _sse("content_block_stop", {
                                        "type": "content_block_stop",
                                        "index": block_index,
                                    })
                                    block_index += 1
                                tool_output_tokens = estimate_tool_call_tokens(calls)
                                tool_calls_emitted = True

                        full_text = "".join(text_buf)
                        if tool_calls_emitted:
                            yield _sse("message_delta", {
                                "type": "message_delta",
                                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                                "usage": {"output_tokens": tool_output_tokens},
                            })
                        else:
                            if not text_started:
                                yield _sse("content_block_start", {
                                    "type": "content_block_start",
                                    "index": block_index,
                                    "content_block": {"type": "text", "text": ""},
                                })
                                text_started = True
                            yield _sse("content_block_stop", {
                                "type": "content_block_stop",
                                "index": block_index,
                            })
                            output_tokens = (
                                adapter.usage.get("output_tokens", 0) if adapter.usage
                                else estimate_tokens(full_text)
                            )
                            yield _sse("message_delta", {
                                "type": "message_delta",
                                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                "usage": {"output_tokens": output_tokens},
                            })

                        yield _sse("message_stop", {"type": "message_stop"})
                        yield "data: [DONE]\n\n"
                        success = True
                        logger.info(
                            "console messages stream completed: model={} text_len={} attempt={}/{}",
                            model, len(full_text), attempt + 1, max_retries + 1,
                        )

                    except UpstreamError as exc:
                        fail_exc = exc
                        if _should_retry_upstream(exc, retry_codes) and attempt < max_retries:
                            _retry = True
                            logger.warning(
                                "console messages retry: attempt={}/{} status={}",
                                attempt + 1, max_retries, exc.status,
                            )
                        else:
                            raise

                finally:
                    await directory.release(acct)
                    kind = (
                        FeedbackKind.SUCCESS if success
                        else feedback_kind_for_error(fail_exc) if fail_exc
                        else FeedbackKind.SERVER_ERROR
                    )
                    await directory.feedback(token, kind, selected_mode_id, now_s_val=now_s())
                    if success:
                        asyncio.create_task(
                            _quota_sync(token, selected_mode_id)
                        ).add_done_callback(_log_task_exception)
                    else:
                        asyncio.create_task(
                            _fail_sync(token, selected_mode_id, fail_exc)
                        ).add_done_callback(_log_task_exception)

                if success or not _retry:
                    return
                excluded.append(token)

        return _run_stream()

    # ── Non-streaming ─────────────────────────────────────────────────────────
    excluded: list[str] = []
    for attempt in range(max_retries + 1):
        acct, selected_mode_id = await reserve_account(
            directory, spec, now_s_override=now_s(),
            exclude_tokens=excluded or None,
        )
        if acct is None:
            raise RateLimitError("No available accounts for this model tier")

        token = acct.token
        success = False
        fail_exc: BaseException | None = None
        adapter = ConsoleStreamAdapter()

        try:
            payload = build_console_payload(
                messages=request_messages,
                model=model,
                temperature=temperature,
                top_p=top_p,
                reasoning_effort=effort,
                stream=True,
            )

            try:
                async for event_type, data in stream_console_chat(
                    token, payload, timeout_s=timeout_s
                ):
                    adapter.feed(event_type, data)

                full_text = adapter.full_text
                usage_data = adapter.usage
                input_tokens = (
                    usage_data.get("input_tokens", 0) if usage_data
                    else estimate_prompt_tokens(request_messages)
                )
                output_tokens = (
                    usage_data.get("output_tokens", 0) if usage_data
                    else estimate_tokens(full_text)
                )

                if tool_names:
                    tc_result = parse_tool_calls(full_text, tool_names)
                    if tc_result.calls:
                        result = {
                            "id": msg_id,
                            "type": "message",
                            "role": "assistant",
                            "model": model,
                            "content": _tool_use_blocks(tc_result.calls),
                            "stop_reason": "tool_use",
                            "stop_sequence": None,
                            "usage": {
                                "input_tokens": input_tokens,
                                "output_tokens": estimate_tool_call_tokens(tc_result.calls),
                            },
                        }
                        success = True
                        logger.info(
                            "console messages non-stream tool_calls: model={} calls={}",
                            model, len(tc_result.calls),
                        )
                        return result

                result = {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [{"type": "text", "text": full_text}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                }
                success = True
                logger.info(
                    "console messages non-stream completed: model={} text_len={}",
                    model, len(full_text),
                )
                return result

            except UpstreamError as exc:
                fail_exc = exc
                if _should_retry_upstream(exc, retry_codes) and attempt < max_retries:
                    logger.warning(
                        "console messages non-stream retry: attempt={}/{} status={}",
                        attempt + 1, max_retries, exc.status,
                    )
                    excluded.append(token)
                    continue
                raise

        finally:
            await directory.release(acct)
            kind = (
                FeedbackKind.SUCCESS if success
                else feedback_kind_for_error(fail_exc) if fail_exc
                else FeedbackKind.SERVER_ERROR
            )
            await directory.feedback(token, kind, selected_mode_id, now_s_val=now_s())
            if success:
                asyncio.create_task(
                    _quota_sync(token, selected_mode_id)
                ).add_done_callback(_log_task_exception)
            else:
                asyncio.create_task(
                    _fail_sync(token, selected_mode_id, fail_exc)
                ).add_done_callback(_log_task_exception)

    raise RateLimitError("No available accounts after retries")


__all__ = ["create"]
