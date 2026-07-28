import json
import unittest

from app.dataplane.reverse.protocol.xai_console_chat import (
    ConsoleStreamAdapter,
    build_console_payload,
)
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


if __name__ == "__main__":
    unittest.main()
