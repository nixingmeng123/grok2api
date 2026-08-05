"""XAI console.x.ai chat protocol — payload builder and SSE stream adapter.

端点: POST https://console.x.ai/v1/responses
认证: short-lived DPoP access token + per-request ES256 proof

请求格式 (OpenAI Responses API):
{
    "model": "grok-4.3",
    "input": [{"role": "user", "content": [{"type": "input_text", "text": "..."}]}],
    "max_output_tokens": 1000000,
    "temperature": 0.7,
    "top_p": 0.95,
    "reasoning": {"effort": "low"},
    "store": false,
    "include": ["reasoning.encrypted_content"],
    "stream": true
}

响应 SSE 事件类型:
- response.created / response.in_progress  — 忽略
- response.output_item.added               — 忽略
- response.output_item.done                — reasoning item，含 encrypted_content（不可读）
- response.content_part.added             — 忽略
- response.output_text.delta              — 文本 token，delta 字段
- response.output_text.done              — 忽略
- response.content_part.done             — 忽略
- response.output_item.done (message)    — 忽略
- response.completed                      — 含 usage 统计
"""

from typing import Any, AsyncGenerator

import orjson

from app.dataplane.reverse.protocol.tool_parser import ParsedToolCall
from app.platform.errors import UpstreamError
from app.platform.logging.logger import logger


# ---------------------------------------------------------------------------
# 支持的模型名 → console.x.ai 实际 model 字段映射
# ---------------------------------------------------------------------------

# console.x.ai 上可用的模型（通过 grok.com SSO 免费访问）
# key = grok2api 对外暴露的模型名，value = console.x.ai 实际 model 字段
CONSOLE_MODELS: dict[str, str] = {
    "grok-4.5-console":                     "grok-4.5",
    "grok-4.5-low":                         "grok-4.5",
    "grok-4.5-medium":                      "grok-4.5",
    "grok-4.5-high":                        "grok-4.5",
    "grok-4.3-console":                     "grok-4.3",
    "grok-4.3-low":                         "grok-4.3",
    "grok-4.3-medium":                      "grok-4.3",
    "grok-4.3-high":                        "grok-4.3",
    "grok-4.20-0309-reasoning-console":     "grok-4.20-0309-reasoning",
    "grok-4.20-0309-console":               "grok-4.20-0309",
    "grok-4.20-0309-non-reasoning-console": "grok-4.20-0309-non-reasoning",
    "grok-4.20-multi-agent-console":        "grok-4.20-multi-agent-0309",
    "grok-4.20-multi-agent-low":            "grok-4.20-multi-agent-0309",
    "grok-4.20-multi-agent-medium":         "grok-4.20-multi-agent-0309",
    "grok-4.20-multi-agent-high":           "grok-4.20-multi-agent-0309",
    "grok-4.20-multi-agent-xhigh":          "grok-4.20-multi-agent-0309",
    "grok-build-console":                   "grok-build-0.1",
}

# 需要附带 reasoning 字段的模型。
_MODELS_WITH_REASONING_FIELD: frozenset[str] = frozenset({
    "grok-4.5",
    "grok-4.3",
    "grok-4.20-multi-agent-0309",
})

# 模型名后缀 → 固定 effort 值（优先级高于用户传入的 reasoning_effort）
_MODEL_FIXED_EFFORT: dict[str, str] = {
    "grok-4.5-low":    "low",
    "grok-4.5-medium": "medium",
    "grok-4.5-high":   "high",
    "grok-4.3-low":    "low",
    "grok-4.3-medium": "medium",
    "grok-4.3-high":   "high",
    "grok-4.20-multi-agent-low":    "low",
    "grok-4.20-multi-agent-medium": "medium",
    "grok-4.20-multi-agent-high":   "high",
    "grok-4.20-multi-agent-xhigh":  "xhigh",
}

# 特殊 max_output_tokens（默认 1_000_000）
_MODEL_MAX_OUTPUT_TOKENS: dict[str, int] = {
    "grok-4.20-multi-agent-0309": 2_000_000,
    "grok-build-0.1": 256_000,
}

# 支持 web_search / x_search 工具的模型
_MODELS_WITH_SEARCH_TOOLS: frozenset[str] = frozenset({
    "grok-4.20-multi-agent-0309",
    "grok-4.20-0309",
    "grok-4.20-0309-reasoning",
    "grok-4.20-0309-non-reasoning",
    "grok-4.5",
    "grok-4.3",
    "grok-build-0.1",
})

# reasoning effort 映射：OpenAI reasoning_effort → console API effort
_EFFORT_MAP: dict[str, str] = {
    "none":    "none",
    "minimal": "low",
    "low":     "low",
    "medium":  "medium",
    "high":    "high",
    "xhigh":   "xhigh",
}


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

def _convert_function_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Convert Chat Completions function tools to Responses API tools."""
    converted: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        item: dict[str, Any] = {
            "type": "function",
            "name": name,
            "parameters": function.get("parameters") or {
                "type": "object",
                "properties": {},
            },
        }
        description = function.get("description")
        if description:
            item["description"] = str(description)
        if "strict" in function:
            item["strict"] = bool(function["strict"])
        converted.append(item)
    return converted


def client_function_tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    """Return the names of valid client-declared function tools."""
    names: set[str] = set()
    for tool in tools or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        function = tool.get("function")
        source = function if isinstance(function, dict) else tool
        name = str(source.get("name") or "").strip()
        if name:
            names.add(name)
    return names


def _convert_tool_choice(tool_choice: Any) -> Any:
    """Convert Chat Completions tool_choice to Responses API format."""
    if tool_choice is None:
        return "auto"
    if isinstance(tool_choice, str):
        return tool_choice if tool_choice in {"auto", "none", "required"} else "auto"
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type in {"auto", "none", "required"}:
            return choice_type
        if choice_type == "function":
            function = tool_choice.get("function")
            name = function.get("name") if isinstance(function, dict) else tool_choice.get("name")
            if name:
                return {"type": "function", "name": str(name)}
    return "auto"


def build_console_payload(
    *,
    messages: list[dict[str, Any]],
    model: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    reasoning_effort: str | None = None,
    stream: bool = True,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> dict[str, Any]:
    """Build the JSON payload for POST console.x.ai/v1/responses.

    将 OpenAI messages 格式转换为 Responses API input 格式。
    """
    # 转换 messages → input 数组
    input_items: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls")

        if role == "tool":
            output = content
            if not isinstance(output, str):
                try:
                    output = orjson.dumps(output).decode()
                except (TypeError, ValueError):
                    output = str(output)
            input_items.append({
                "type": "function_call_output",
                "call_id": str(msg.get("tool_call_id") or ""),
                "output": output,
            })
            continue

        if role == "assistant" and tool_calls:
            if content:
                if isinstance(content, str):
                    content_blocks = [{"type": "input_text", "text": content}]
                else:
                    content_blocks = [{"type": "input_text", "text": str(content)}]
                input_items.append({"role": "assistant", "content": content_blocks})
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function") or {}
                name = str(function.get("name") or "").strip()
                if not name:
                    continue
                arguments = function.get("arguments") or "{}"
                if not isinstance(arguments, str):
                    try:
                        arguments = orjson.dumps(arguments).decode()
                    except (TypeError, ValueError):
                        arguments = "{}"
                input_items.append({
                    "type": "function_call",
                    "call_id": str(tool_call.get("id") or ""),
                    "name": name,
                    "arguments": arguments,
                })
            continue

        # 映射 role
        if role in ("system", "developer"):
            # system 消息作为 instructions 字段处理，这里先放入 input
            api_role = "system"
        elif role == "assistant":
            api_role = "assistant"
        else:
            api_role = "user"

        # 处理 content
        if isinstance(content, str):
            content_blocks = [{"type": "input_text", "text": content}]
        elif isinstance(content, list):
            content_blocks = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    content_blocks.append({"type": "input_text", "text": block.get("text", "")})
                elif btype == "image_url":
                    url = (block.get("image_url") or {}).get("url", "")
                    if url:
                        content_blocks.append({"type": "input_image", "image_url": url})
                else:
                    # 其他类型降级为文本
                    text = block.get("text") or str(block)
                    content_blocks.append({"type": "input_text", "text": text})
        else:
            content_blocks = [{"type": "input_text", "text": str(content)}]

        if content_blocks:
            input_items.append({"role": api_role, "content": content_blocks})

    # reasoning effort：模型名固定值优先，其次用户传入，最后默认 medium
    effort = _MODEL_FIXED_EFFORT.get(model) or _EFFORT_MAP.get(reasoning_effort or "medium", "medium")

    # 获取 console 实际模型名
    console_model = CONSOLE_MODELS.get(model, model)

    payload: dict[str, Any] = {
        "model": console_model,
        "input": input_items,
        "max_output_tokens": _MODEL_MAX_OUTPUT_TOKENS.get(console_model, 1_000_000),
        "temperature": temperature,
        "top_p": top_p,
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "stream": stream,
    }

    if console_model in _MODELS_WITH_REASONING_FIELD:
        payload["reasoning"] = {"effort": effort}

    function_tools = _convert_function_tools(tools)
    payload_tools: list[dict[str, Any]] = []
    if console_model in _MODELS_WITH_SEARCH_TOOLS:
        payload_tools.extend([
            {"type": "web_search", "enable_image_understanding": True},
            {"type": "x_search", "enable_video_understanding": True},
        ])
    payload_tools.extend(function_tools)
    if payload_tools:
        payload["tools"] = payload_tools
        payload["tool_choice"] = (
            _convert_tool_choice(tool_choice) if function_tools else "auto"
        )

    logger.debug(
        "console payload built: model={} console_model={} input_items={} has_reasoning={} function_tools={}",
        model, console_model, len(input_items), console_model in _MODELS_WITH_REASONING_FIELD,
        len(function_tools),
    )
    return payload


# ---------------------------------------------------------------------------
# SSE stream adapter
# ---------------------------------------------------------------------------

class ConsoleStreamAdapter:
    """Parse console.x.ai SSE events and yield text tokens.

    只关心 response.output_text.delta 事件，其余忽略。
    response.completed 事件用于提取 usage 统计。
    """

    __slots__ = (
        "text_buf",
        "usage",
        "_done",
        "_function_items",
        "_function_tool_names",
    )

    def __init__(
        self,
        function_tool_names: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.text_buf: list[str] = []
        self.usage: dict[str, Any] | None = None
        self._done = False
        self._function_items: dict[str, dict[str, Any]] = {}
        self._function_tool_names = (
            None
            if function_tool_names is None
            else {
                str(name).strip()
                for name in function_tool_names
                if str(name).strip()
            }
        )

    @staticmethod
    def _function_key(item: dict[str, Any], fallback: Any = None) -> str:
        return str(item.get("id") or item.get("call_id") or fallback or "function")

    def _capture_function_item(
        self,
        item: dict[str, Any],
        *,
        fallback: Any = None,
    ) -> None:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            return
        key = self._function_key(item, fallback)
        record = self._function_items.setdefault(
            key,
            {"id": key, "call_id": "", "name": "", "arguments": "", "chunks": []},
        )
        if item.get("id"):
            record["id"] = str(item["id"])
        if item.get("call_id"):
            record["call_id"] = str(item["call_id"])
        if item.get("name"):
            record["name"] = str(item["name"])
        if item.get("arguments") not in (None, ""):
            record["arguments"] = item["arguments"]

    def _function_record(self, obj: dict[str, Any]) -> dict[str, Any]:
        key = str(
            obj.get("item_id")
            or obj.get("call_id")
            or obj.get("output_index")
            or "function"
        )
        return self._function_items.setdefault(
            key,
            {"id": key, "call_id": "", "name": "", "arguments": "", "chunks": []},
        )

    def feed(self, event_type: str, data: str) -> list[str]:
        """解析一个 SSE 事件，返回文本 token 列表（通常 0 或 1 个）。"""
        if self._done:
            return []

        try:
            obj = orjson.loads(data)
        except (orjson.JSONDecodeError, ValueError):
            return []

        if event_type == "response.output_text.delta":
            delta = obj.get("delta", "")
            if delta:
                self.text_buf.append(delta)
                return [delta]

        elif event_type in {"response.output_item.added", "response.output_item.done"}:
            self._capture_function_item(
                obj.get("item") or {},
                fallback=obj.get("output_index"),
            )

        elif event_type == "response.function_call_arguments.delta":
            record = self._function_record(obj)
            delta = obj.get("delta", "")
            if delta:
                record["chunks"].append(str(delta))

        elif event_type == "response.function_call_arguments.done":
            record = self._function_record(obj)
            if obj.get("arguments") is not None:
                record["arguments"] = obj.get("arguments")

        elif event_type == "response.completed":
            resp = obj.get("response", {})
            self.usage = resp.get("usage")
            for index, item in enumerate(resp.get("output") or []):
                self._capture_function_item(item, fallback=index)
            self._done = True

        elif event_type == "error":
            msg = obj.get("message") or str(obj)
            raise UpstreamError(f"Console API error: {msg}", status=502)

        return []

    @property
    def full_text(self) -> str:
        return "".join(self.text_buf)

    @property
    def function_calls(self) -> list[ParsedToolCall]:
        calls: list[ParsedToolCall] = []
        for record in self._function_items.values():
            name = str(record.get("name") or "").strip()
            if not name:
                continue
            if (
                self._function_tool_names is not None
                and name not in self._function_tool_names
            ):
                continue
            arguments = record.get("arguments")
            if arguments in (None, ""):
                arguments = "".join(record.get("chunks") or [])
            if not isinstance(arguments, str):
                try:
                    arguments = orjson.dumps(arguments).decode()
                except (TypeError, ValueError):
                    arguments = "{}"
            if not arguments:
                arguments = "{}"
            calls.append(
                ParsedToolCall(
                    call_id=str(record.get("call_id") or record.get("id") or ""),
                    name=name,
                    arguments=arguments,
                )
            )
        return calls


def classify_console_line(line: str) -> tuple[str, str]:
    """Parse a raw SSE line into (event_type, data).

    console.x.ai 使用标准 SSE 格式:
        event: response.output_text.delta
        data: {...}
    """
    line = line.strip()
    if not line:
        return "skip", ""
    if line.startswith("event:"):
        return "event", line[6:].strip()
    if line.startswith("data:"):
        data = line[5:].strip()
        if data == "[DONE]":
            return "done", ""
        return "data", data
    return "skip", ""


async def stream_console_chat(
    token: str,
    payload: dict[str, Any],
    *,
    timeout_s: float = 120.0,
) -> AsyncGenerator[tuple[str, str], None]:
    """POST to console.x.ai/v1/responses and yield (event_type, data) pairs.

    走现有的 proxy lease + curl-cffi 体系，与 grok.com 共用 CF clearance。
    """
    from app.dataplane.proxy import get_proxy_runtime
    from app.dataplane.proxy.adapters.headers import build_console_headers
    from app.dataplane.proxy.adapters.session import ResettableSession, build_session_kwargs
    from app.dataplane.reverse.protocol.xai_console_dpop import (
        apply_dpop_headers,
        dpop_sessions,
    )
    from app.dataplane.reverse.runtime.endpoint_table import (
        CONSOLE_DPOP_TOKEN,
        CONSOLE_RESPONSES,
    )

    proxy = await get_proxy_runtime()
    lease = await proxy.acquire()

    payload_bytes = orjson.dumps(payload)
    session_kwargs = build_session_kwargs(lease=lease)

    async with ResettableSession(**session_kwargs) as session:
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
                apply_dpop_headers(
                    headers,
                    dpop_session,
                    method="POST",
                    url=CONSOLE_RESPONSES,
                )
                response = await session.post(
                    CONSOLE_RESPONSES,
                    headers=headers,
                    data=payload_bytes,
                    timeout=timeout_s,
                    stream=True,
                )
            except UpstreamError as exc:
                await proxy.feedback(lease, _status_feedback(exc.status))
                raise
            except Exception as exc:
                await proxy.feedback(lease, _transport_error_feedback())
                raise UpstreamError(f"Console transport failed: {exc}", status=502) from exc

            if response.status_code != 401 or auth_attempt > 0:
                break
            await _discard_response(response)
            dpop_sessions.invalidate(cache_key, dpop_session.access_token)

        if response is None:
            raise UpstreamError("Console DPoP request failed", status=502)

        if response.status_code != 200:
            try:
                body = response.content.decode("utf-8", "replace")[:400]
            except Exception:
                body = ""
            await proxy.feedback(lease, _status_feedback(response.status_code))
            raise UpstreamError(
                f"Console API returned {response.status_code}",
                status=response.status_code,
                body=body,
            )

        await proxy.feedback(lease, _success_feedback())

        current_event = ""
        try:
            async for raw_line in response.aiter_lines():
                # curl-cffi 的 aiter_lines 返回 bytes，先解码为 str
                if isinstance(raw_line, bytes):
                    try:
                        raw_line = raw_line.decode("utf-8")
                    except UnicodeDecodeError:
                        raw_line = raw_line.decode("utf-8", errors="replace")
                kind, value = classify_console_line(raw_line)
                if kind == "event":
                    current_event = value
                elif kind == "data":
                    yield current_event, value
                    current_event = ""
                elif kind == "done":
                    return
        except Exception as exc:
            raise UpstreamError(f"Console stream read failed: {exc}", status=502) from exc


async def _discard_response(response: Any) -> None:
    """Drain a rejected streamed response before reusing the HTTP session."""
    try:
        async for _ in response.aiter_content():
            pass
    except Exception:
        pass


def _success_feedback():
    from app.control.proxy.models import ProxyFeedback, ProxyFeedbackKind
    return ProxyFeedback(kind=ProxyFeedbackKind.SUCCESS, status_code=200)

def _transport_error_feedback():
    from app.control.proxy.models import ProxyFeedback, ProxyFeedbackKind
    return ProxyFeedback(kind=ProxyFeedbackKind.TRANSPORT_ERROR)

def _status_feedback(status: int):
    from app.control.proxy.models import ProxyFeedback, ProxyFeedbackKind
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
    "CONSOLE_MODELS",
    "build_console_payload",
    "client_function_tool_names",
    "ConsoleStreamAdapter",
    "classify_console_line",
    "stream_console_chat",
]
