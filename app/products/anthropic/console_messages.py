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
    extract_tool_names,
)
from app.products._account_selection import reserve_account, selection_max_retries
from app.products.openai.chat import _configured_retry_codes, _should_retry_upstream
from app.products.openai._tool_sieve import ToolSieve


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {orjson.dumps(data).decode()}\n\n"


_HEARTBEAT_INTERVAL_S = 5.0


async def _with_heartbeats(iterator, *, interval_s: float | None = None):
    """Yield None while waiting for a quiet upstream iterator."""
    interval = interval_s or _HEARTBEAT_INTERVAL_S
    upstream = iterator.__aiter__()
    while True:
        next_item = asyncio.create_task(anext(upstream))
        try:
            while True:
                done, _ = await asyncio.wait({next_item}, timeout=interval)
                if next_item in done:
                    break
                yield None
            try:
                item = next_item.result()
            except StopAsyncIteration:
                return
        finally:
            if not next_item.done():
                next_item.cancel()
                await asyncio.gather(next_item, return_exceptions=True)
        yield item


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


def _normalize_gateway_base(value: str) -> str:
    base = (value or "").strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")
    return base


def _image_generation_prompt(
    tool_names: list[str],
    *,
    enabled: bool,
    gateway_base: str,
) -> str:
    if not enabled or not ({"PowerShell", "Bash"} & set(tool_names)):
        return ""

    base = _normalize_gateway_base(gateway_base)
    endpoint = (
        f"{base}/v1/images/generations"
        if base
        else "<ANTHROPIC_BASE_URL without a trailing /v1>/v1/images/generations"
    )
    return (
        "\nCLAUDE CODE IMAGE GENERATION:\n"
        "- Original bitmap image generation is available through this gateway. Use it "
        "proactively when the user's task genuinely needs new photographic, illustrative, "
        "texture, background, or other raster assets. Do not use it for icons, simple "
        "diagrams, or visuals that are better implemented with HTML/CSS/SVG.\n"
        "- Generate one asset at a time by using the real local PowerShell or Bash tool to "
        f"POST {endpoint}. Send JSON with model=grok-imagine-image-lite, prompt, n=1, "
        "size=1024x1024, and response_format=url.\n"
        "- Authenticate without exposing credentials. Read the token at runtime from "
        "ANTHROPIC_AUTH_TOKEN, falling back to ANTHROPIC_API_KEY, and send it as either "
        "Authorization: Bearer <token> or x-api-key. Never print, echo, or embed the token "
        "in generated files.\n"
        "- If the endpoint above is expressed with ANTHROPIC_BASE_URL, read that variable at "
        "runtime, trim trailing slashes and a trailing /v1, then append "
        "/v1/images/generations.\n"
        "- Read data[0].url from the JSON response, download it into an appropriate assets "
        "directory in the user's current project, and reference that local file from the "
        "page or application. Create the directory first when needed.\n"
        "- Continue the original task after the download succeeds. If generation fails, "
        "report the real error once and continue with a reasonable non-generated fallback; "
        "do not loop or claim that an image was saved when it was not.\n"
    )


def _build_console_tool_prompt(
    tools: list[dict],
    tool_choice: Any,
    *,
    auto_image_enabled: bool = False,
    image_gateway_base: str = "",
) -> str:
    tool_names = extract_tool_names(tools)
    choice_instruction = ""
    if tool_choice == "required" or (
        isinstance(tool_choice, dict) and tool_choice.get("type") == "required"
    ):
        choice_instruction = " You must call one of the provided tools for this turn."
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        function = tool_choice.get("function")
        forced_name = function.get("name") if isinstance(function, dict) else tool_choice.get("name")
        if forced_name:
            choice_instruction = f" You must call the {forced_name} tool for this turn."
    prompt = (
        "CLAUDE CODE LOCAL TOOL ROUTING:\n"
        f"- The client provided these real local tools: {', '.join(tool_names)}.\n"
        "- Use the provided native function tools whenever the user asks to read, create, "
        "write, edit, search local files, or run a local command.\n"
        "- Use Write to create files, Edit to modify files, Read to read files, and "
        "Bash or PowerShell for shell commands.\n"
        "- Do not merely say that you are calling a tool. Emit an actual function call.\n"
        "- Do not claim that local filesystem access is unavailable; the client executes "
        "the function call for you.\n"
        "- Internet search is available through upstream web search. For web queries, "
        "search upstream and answer directly; do not call Bash or PowerShell only to "
        "access the internet.\n"
        "- After a tool result reports success, continue from that result. Do not restart "
        "the original task or repeat Write/Edit on the same file. If the requested work "
        "is complete, return a final plain-text summary without another tool call.\n"
        f"- Tool names are case-sensitive.{choice_instruction}"
    )
    return prompt + _image_generation_prompt(
        tool_names,
        enabled=auto_image_enabled,
        gateway_base=image_gateway_base,
    )


_UPSTREAM_WEB_TOOL_NAMES = frozenset({"WebSearch", "WebFetch"})


def _console_local_tools(tools: list[dict] | None) -> list[dict]:
    local_tools: list[dict] = []
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if name not in _UPSTREAM_WEB_TOOL_NAMES:
            local_tools.append(tool)
    return local_tools


def _console_tool_choice(tool_choice: Any) -> Any:
    if not isinstance(tool_choice, dict) or tool_choice.get("type") != "function":
        return tool_choice
    function = tool_choice.get("function")
    name = function.get("name") if isinstance(function, dict) else tool_choice.get("name")
    return "auto" if name in _UPSTREAM_WEB_TOOL_NAMES else tool_choice


def _prepend_text_to_content(content: Any, text: str) -> Any:
    if isinstance(content, str):
        return f"{text}\n\n{content}"
    if isinstance(content, list):
        return [{"type": "text", "text": text}, *content]
    if content is None:
        return text
    return f"{text}\n\n{content}"


def _prepare_console_messages(
    messages: list[dict],
    tools: list[dict] | None,
    tool_choice: Any,
    *,
    auto_image_enabled: bool = False,
    image_gateway_base: str = "",
) -> tuple[list[dict], list[str]]:
    tool_names = extract_tool_names(tools or []) if tools else []
    prepared: list[dict] = []
    tool_prompt = (
        _build_console_tool_prompt(
            tools,
            tool_choice,
            auto_image_enabled=auto_image_enabled,
            image_gateway_base=image_gateway_base,
        )
        if tools
        else ""
    )
    prompt_injected = False

    if tool_prompt:
        prepared.append({
            "role": "system",
            "content": tool_prompt,
        })

    def append_user_like(content: Any) -> None:
        nonlocal prompt_injected
        if tool_prompt and not prompt_injected:
            content = _prepend_text_to_content(content, tool_prompt)
            prompt_injected = True
        prepared.append({"role": "user", "content": content})

    for msg in messages:
        role = msg.get("role", "user")
        tool_calls = msg.get("tool_calls")

        if role == "tool":
            prepared.append(msg)
            continue

        if role == "assistant" and tool_calls:
            prepared.append(msg)
            continue

        if role == "user":
            updated = dict(msg)
            if tool_prompt and not prompt_injected:
                updated["content"] = _prepend_text_to_content(updated.get("content"), tool_prompt)
                prompt_injected = True
            prepared.append(updated)
        else:
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


def _last_successful_tool_action(messages: list[dict]):
    calls_by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    latest = None
    error_markers = ("error", "failed", "failure", "denied", "错误", "失败", "拒绝")
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict):
                    continue
                call_id = str(call.get("id") or "")
                name = str(function.get("name") or "")
                arguments = function.get("arguments") or "{}"
                try:
                    parsed = orjson.loads(arguments) if isinstance(arguments, str) else arguments
                except (orjson.JSONDecodeError, ValueError):
                    parsed = {}
                if call_id and name and isinstance(parsed, dict):
                    calls_by_id[call_id] = (name, parsed)
        elif message.get("role") == "tool":
            result_text = _content_to_text(message.get("content")).lower()
            if any(marker in result_text for marker in error_markers):
                continue
            action = calls_by_id.get(str(message.get("tool_call_id") or ""))
            if action:
                latest = action
        elif message.get("role") == "user":
            # A new user turn may intentionally ask to modify the same file again.
            latest = None
    return latest


def _is_repeated_file_mutation(call, completed_action) -> bool:
    if not completed_action:
        return False
    completed_name, completed_arguments = completed_action
    if call.name != completed_name or call.name not in {"Write", "Edit", "NotebookEdit"}:
        return False
    try:
        arguments = orjson.loads(call.arguments)
    except (orjson.JSONDecodeError, ValueError):
        return False
    if not isinstance(arguments, dict):
        return False
    target_key = "notebook_path" if call.name == "NotebookEdit" else "file_path"
    target = arguments.get(target_key)
    return bool(target and target == completed_arguments.get(target_key))


def _native_tool_calls(
    adapter: ConsoleStreamAdapter,
    tool_names: list[str],
    completed_action=None,
):
    if not tool_names:
        return []
    allowed = set(tool_names)
    return [
        call
        for call in adapter.function_calls
        if call.name in allowed and not _is_repeated_file_mutation(call, completed_action)
    ]


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
    local_tools = _console_local_tools(tools)
    local_tool_choice = _console_tool_choice(tool_choice)
    request_messages, tool_names = _prepare_console_messages(
        messages,
        local_tools,
        local_tool_choice,
        auto_image_enabled=cfg.get_bool("features.claude_code_auto_image", True),
        image_gateway_base=cfg.get_str("app.app_url", ""),
    )
    completed_action = _last_successful_tool_action(messages)

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
                response_committed = False
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
                        tools=local_tools,
                        tool_choice=local_tool_choice,
                    )

                    try:
                        upstream = stream_console_chat(
                            token, payload, timeout_s=timeout_s
                        )
                        stream_started = False
                        async for upstream_event in _with_heartbeats(upstream):
                            if upstream_event is None:
                                response_committed = True
                                yield ": heartbeat\n\n"
                                continue

                            event_type, data = upstream_event
                            if not stream_started and event_type == "error":
                                adapter.feed(event_type, data)

                            if not stream_started:
                                stream_started = True
                                response_committed = True
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

                            tokens = adapter.feed(event_type, data)
                            if not tokens:
                                # Native function-call arguments can be very
                                # large. Keep Cloudflare and Claude Code from
                                # treating that otherwise-silent period as a
                                # dead connection.
                                yield ": heartbeat\n\n"
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

                        raw_native_calls = adapter.function_calls
                        native_calls = _native_tool_calls(
                            adapter,
                            tool_names,
                            completed_action,
                        )
                        repeat_suppressed = bool(raw_native_calls and not native_calls)
                        if native_calls and not tool_calls_emitted:
                            if text_started:
                                yield _sse("content_block_stop", {
                                    "type": "content_block_stop",
                                    "index": block_index,
                                })
                                block_index += 1
                                text_started = False
                            for call in native_calls:
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
                            tool_output_tokens = estimate_tool_call_tokens(native_calls)
                            tool_calls_emitted = True
                            logger.info(
                                "console messages native tool_calls: model={} calls={}",
                                model, len(native_calls),
                            )
                        elif repeat_suppressed:
                            note = "\n已完成；未重复执行已经成功的文件操作。"
                            if not text_started:
                                yield _sse("content_block_start", {
                                    "type": "content_block_start",
                                    "index": block_index,
                                    "content_block": {"type": "text", "text": ""},
                                })
                                text_started = True
                            text_buf.append(note)
                            yield _sse("content_block_delta", {
                                "type": "content_block_delta",
                                "index": block_index,
                                "delta": {"type": "text_delta", "text": note},
                            })
                            logger.warning(
                                "console messages suppressed repeated file mutation: model={} action={}",
                                model, completed_action[0] if completed_action else "?",
                            )

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

                        if not stream_started:
                            stream_started = True
                            response_committed = True
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
                        success = True
                        logger.info(
                            "console messages stream completed: model={} text_len={} attempt={}/{}",
                            model, len(full_text), attempt + 1, max_retries + 1,
                        )

                    except UpstreamError as exc:
                        fail_exc = exc
                        if (
                            not response_committed
                            and _should_retry_upstream(exc, retry_codes)
                            and attempt < max_retries
                        ):
                            _retry = True
                            logger.warning(
                                "console messages retry: attempt={}/{} status={}",
                                attempt + 1, max_retries, exc.status,
                            )
                        else:
                            if response_committed:
                                logger.warning(
                                    "console messages stream aborted after downstream commit: "
                                    "model={} status={}",
                                    model, exc.status,
                                )
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
                tools=local_tools,
                tool_choice=local_tool_choice,
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

                raw_native_calls = adapter.function_calls
                native_calls = _native_tool_calls(
                    adapter,
                    tool_names,
                    completed_action,
                )
                if native_calls:
                    result = {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": _tool_use_blocks(native_calls),
                        "stop_reason": "tool_use",
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": input_tokens,
                            "output_tokens": estimate_tool_call_tokens(native_calls),
                        },
                    }
                    success = True
                    logger.info(
                        "console messages non-stream native tool_calls: model={} calls={}",
                        model, len(native_calls),
                    )
                    return result
                if raw_native_calls and completed_action:
                    full_text += "\n已完成；未重复执行已经成功的文件操作。"
                    output_tokens = estimate_tokens(full_text)
                    logger.warning(
                        "console messages non-stream suppressed repeated file mutation: model={} action={}",
                        model, completed_action[0],
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
