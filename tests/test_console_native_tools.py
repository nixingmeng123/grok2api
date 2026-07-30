import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.dataplane.reverse.protocol.xai_console_chat import (
    ConsoleStreamAdapter,
    build_console_payload,
    client_function_tool_names,
)
from app.products.openai import console_responses, responses
from app.products.anthropic.console_messages import (
    _last_successful_tool_action,
    _native_tool_calls,
)


WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "Write",
        "description": "Write a file to the local filesystem.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
        },
    },
}


class ConsoleNativeToolTests(unittest.TestCase):
    def test_client_function_tool_names_filters_invalid_tools(self):
        tools = [
            WRITE_TOOL,
            {"type": "web_search"},
            {"type": "function", "function": {"name": "  Read  "}},
            {"type": "function", "function": {}},
            None,
        ]

        self.assertEqual(
            client_function_tool_names(tools),
            {"Write", "Read"},
        )

    def test_payload_forwards_client_function_tools(self):
        payload = build_console_payload(
            messages=[{"role": "user", "content": "Create hello.txt"}],
            model="grok-4.5-high",
            tools=[WRITE_TOOL],
            tool_choice={
                "type": "function",
                "function": {"name": "Write"},
            },
        )

        self.assertEqual(payload["tools"][-1], {
            "type": "function",
            "name": "Write",
            "description": "Write a file to the local filesystem.",
            "parameters": WRITE_TOOL["function"]["parameters"],
        })
        self.assertEqual(payload["tools"][0]["type"], "web_search")
        self.assertEqual(payload["tools"][1]["type"], "x_search")
        self.assertEqual(
            payload["tool_choice"],
            {"type": "function", "name": "Write"},
        )

    def test_adapter_collects_streamed_function_call(self):
        adapter = ConsoleStreamAdapter()
        adapter.feed(
            "response.output_item.added",
            json.dumps({
                "output_index": 0,
                "item": {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "Write",
                    "arguments": "",
                    "status": "in_progress",
                },
            }),
        )
        adapter.feed(
            "response.function_call_arguments.delta",
            json.dumps({
                "item_id": "fc_1",
                "output_index": 0,
                "delta": '{"file_path":"hello.txt",',
            }),
        )
        adapter.feed(
            "response.function_call_arguments.delta",
            json.dumps({
                "item_id": "fc_1",
                "output_index": 0,
                "delta": '"content":"hello"}',
            }),
        )
        adapter.feed(
            "response.function_call_arguments.done",
            json.dumps({
                "item_id": "fc_1",
                "output_index": 0,
                "arguments": '{"file_path":"hello.txt","content":"hello"}',
            }),
        )
        adapter.feed(
            "response.completed",
            json.dumps({
                "response": {
                    "output": [{
                        "id": "fc_1",
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "Write",
                        "arguments": '{"file_path":"hello.txt","content":"hello"}',
                        "status": "completed",
                    }],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            }),
        )

        self.assertEqual(len(adapter.function_calls), 1)
        call = adapter.function_calls[0]
        self.assertEqual(call.call_id, "call_1")
        self.assertEqual(call.name, "Write")
        self.assertEqual(
            json.loads(call.arguments),
            {"file_path": "hello.txt", "content": "hello"},
        )

    def test_payload_preserves_function_call_history(self):
        payload = build_console_payload(
            messages=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "Write",
                            "arguments": '{"file_path":"test.txt","content":"ok"}',
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "File written successfully",
                },
                {"role": "user", "content": "Now inspect the directory"},
            ],
            model="grok-4.5-high",
            tools=[WRITE_TOOL],
        )

        self.assertEqual(payload["input"][0], {
            "type": "function_call",
            "call_id": "call_1",
            "name": "Write",
            "arguments": '{"file_path":"test.txt","content":"ok"}',
        })
        self.assertEqual(payload["input"][1], {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "File written successfully",
        })

    def test_repeated_successful_write_is_suppressed(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "Write",
                        "arguments": '{"file_path":"result.txt","content":"first"}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "File written successfully",
            },
        ]
        adapter = ConsoleStreamAdapter()
        adapter.feed(
            "response.completed",
            json.dumps({
                "response": {
                    "output": [{
                        "id": "fc_2",
                        "type": "function_call",
                        "call_id": "call_2",
                        "name": "Write",
                        "arguments": '{"file_path":"result.txt","content":"second"}',
                        "status": "completed",
                    }],
                },
            }),
        )

        completed_action = _last_successful_tool_action(messages)
        calls = _native_tool_calls(adapter, ["Write"], completed_action)

        self.assertEqual(completed_action[0], "Write")
        self.assertEqual(calls, [])

    def test_write_to_different_file_is_allowed(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "Write",
                        "arguments": '{"file_path":"result.txt","content":"first"}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "File written successfully",
            },
        ]
        adapter = ConsoleStreamAdapter()
        adapter.feed(
            "response.completed",
            json.dumps({
                "response": {
                    "output": [{
                        "id": "fc_2",
                        "type": "function_call",
                        "call_id": "call_2",
                        "name": "Write",
                        "arguments": '{"file_path":"summary.txt","content":"second"}',
                        "status": "completed",
                    }],
                },
            }),
        )

        completed_action = _last_successful_tool_action(messages)
        calls = _native_tool_calls(adapter, ["Write"], completed_action)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "Write")

    def test_new_user_turn_allows_same_file_write(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "Write",
                        "arguments": '{"file_path":"result.txt","content":"first"}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "File written successfully",
            },
            {
                "role": "user",
                "content": "Rewrite result.txt with the corrected weather.",
            },
        ]

        self.assertIsNone(_last_successful_tool_action(messages))


class ConsoleResponsesToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_responses_route_forwards_flat_tools_to_console(self):
        flat_tool = {
            "type": "function",
            "name": "Write",
            "description": WRITE_TOOL["function"]["description"],
            "parameters": WRITE_TOOL["function"]["parameters"],
        }
        spec = SimpleNamespace(
            mode_id=5,
            is_console_chat=lambda: True,
        )

        with (
            patch.object(
                responses,
                "get_config",
                return_value=SimpleNamespace(
                    get=lambda key, default=None: default,
                    get_float=lambda key, default: default,
                ),
            ),
            patch.object(responses, "resolve_model", return_value=spec),
            patch("app.dataplane.account._directory", object()),
            patch.object(
                console_responses,
                "create",
                new_callable=AsyncMock,
                return_value={"ok": True},
            ) as console_create,
        ):
            result = await responses.create(
                model="grok-4.5-high",
                input_val="Create hello.txt",
                instructions=None,
                stream=True,
                emit_think=True,
                reasoning_effort="high",
                temperature=0.7,
                top_p=0.95,
                tools=[flat_tool],
                tool_choice="auto",
            )

        self.assertEqual(result, {"ok": True})
        forwarded = console_create.await_args.kwargs
        self.assertEqual(forwarded["tools"], [WRITE_TOOL])
        self.assertEqual(forwarded["tool_choice"], "auto")

    async def test_console_responses_emits_function_call_events(self):
        captured_payload = {}

        async def fake_stream_console_chat(token, payload, *, timeout_s):
            captured_payload.update(payload)
            yield "response.output_item.added", json.dumps({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "fc_upstream",
                    "type": "function_call",
                    "call_id": "call_write",
                    "name": "Write",
                    "arguments": "",
                    "status": "in_progress",
                },
            })
            yield "response.function_call_arguments.delta", json.dumps({
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_upstream",
                "output_index": 0,
                "delta": '{"file_path":"hello.txt","content":"hello"}',
            })
            yield "response.function_call_arguments.done", json.dumps({
                "type": "response.function_call_arguments.done",
                "item_id": "fc_upstream",
                "output_index": 0,
                "name": "Write",
                "arguments": '{"file_path":"hello.txt","content":"hello"}',
            })
            yield "response.output_item.done", json.dumps({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "fc_upstream",
                    "type": "function_call",
                    "call_id": "call_write",
                    "name": "Write",
                    "arguments": '{"file_path":"hello.txt","content":"hello"}',
                    "status": "completed",
                },
            })
            yield "response.completed", json.dumps({
                "type": "response.completed",
                "response": {
                    "output": [{
                        "id": "fc_upstream",
                        "type": "function_call",
                        "call_id": "call_write",
                        "name": "Write",
                        "arguments": '{"file_path":"hello.txt","content":"hello"}',
                        "status": "completed",
                    }],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            })

        directory = SimpleNamespace(
            release=AsyncMock(),
            feedback=AsyncMock(),
        )
        cfg = SimpleNamespace(get_float=lambda key, default: default)
        acct = SimpleNamespace(token="test-token")

        with (
            patch.object(console_responses, "get_config", return_value=cfg),
            patch.object(
                console_responses,
                "resolve_model",
                return_value=SimpleNamespace(),
            ),
            patch.object(console_responses, "selection_max_retries", return_value=0),
            patch.object(console_responses, "_configured_retry_codes", return_value=set()),
            patch.object(
                console_responses,
                "reserve_account",
                new_callable=AsyncMock,
                return_value=(acct, 5),
            ),
            patch.object(
                console_responses,
                "stream_console_chat",
                new=fake_stream_console_chat,
            ),
            patch.object(console_responses, "_quota_sync", new_callable=AsyncMock),
            patch("app.dataplane.account._directory", directory),
        ):
            stream = await console_responses.create(
                model="grok-4.5-high",
                messages=[{"role": "user", "content": "Create hello.txt"}],
                stream=True,
                emit_think=True,
                reasoning_effort="high",
                temperature=0.7,
                top_p=0.95,
                response_id="resp_test",
                reasoning_id="rs_test",
                message_id="msg_test",
                tools=[WRITE_TOOL],
                tool_choice="auto",
            )
            chunks = [chunk async for chunk in stream]

        events = []
        for chunk in chunks:
            if not chunk.startswith("event: "):
                continue
            header, payload = chunk.split("\n", 1)
            events.append((
                header.removeprefix("event: "),
                json.loads(payload.removeprefix("data: ").strip()),
            ))

        event_types = [event_type for event_type, _ in events]
        self.assertIn("response.function_call_arguments.delta", event_types)
        self.assertIn("response.function_call_arguments.done", event_types)

        argument_done = next(
            payload
            for event_type, payload in events
            if event_type == "response.function_call_arguments.done"
        )
        self.assertEqual(argument_done["name"], "Write")
        self.assertEqual(
            json.loads(argument_done["arguments"]),
            {"file_path": "hello.txt", "content": "hello"},
        )

        completed = next(
            payload["response"]
            for event_type, payload in events
            if event_type == "response.completed"
        )
        function_items = [
            item for item in completed["output"]
            if item["type"] == "function_call"
        ]
        self.assertEqual(len(function_items), 1)
        self.assertEqual(function_items[0]["call_id"], "call_write")
        self.assertEqual(function_items[0]["name"], "Write")

        self.assertEqual(captured_payload["tools"][-1]["name"], "Write")
        self.assertEqual(captured_payload["tool_choice"], "auto")


if __name__ == "__main__":
    unittest.main()
