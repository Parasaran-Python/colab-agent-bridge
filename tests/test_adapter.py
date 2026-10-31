import json
import pytest
from colab_agent_bridge.adapter import (
    try_extract_tool_calls,
    normalize_chat_completion_response,
    normalize_chat_completion_stream_chunk,
)


def test_extract_tool_call_simple_json():
    content = '{"name": "skill", "arguments": {"skill": "using-superpowers"}}'
    tool_calls = try_extract_tool_calls(content)
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"]["name"] == "skill"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"skill": "using-superpowers"}


def test_extract_tool_call_markdown_code_block():
    content = '```json\n{"name": "write", "arguments": {"path": "app.py", "content": "print(1)"}}\n```'
    tool_calls = try_extract_tool_calls(content)
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "write"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"path": "app.py", "content": "print(1)"}


def test_extract_tool_call_xml_tags():
    content = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tool_call>'
    tool_calls = try_extract_tool_calls(content)
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"city": "Tokyo"}


def test_extract_tool_call_list():
    content = '[{"name": "tool_a", "arguments": {"x": 1}}, {"name": "tool_b", "arguments": {"y": 2}}]'
    tool_calls = try_extract_tool_calls(content)
    assert tool_calls is not None
    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["name"] == "tool_a"
    assert tool_calls[1]["function"]["name"] == "tool_b"


def test_extract_tool_call_normal_text_returns_none():
    content = "Hello! I am an AI assistant. How can I help you today?"
    assert try_extract_tool_calls(content) is None


def test_extract_tool_call_non_tool_json_returns_none():
    content = 'Here is the JSON config:\n```json\n{"host": "localhost", "port": 8000}\n```'
    assert try_extract_tool_calls(content) is None


def test_normalize_chat_completion_response_with_tool_call():
    raw_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "qwen2.5-coder-7b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '{"name": "skill", "arguments": {"skill": "using-superpowers"}}',
                },
                "finish_reason": "stop",
            }
        ],
    }

    normalized = normalize_chat_completion_response(raw_response)
    choice = normalized["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert "tool_calls" in choice["message"]
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "skill"


def test_normalize_chat_completion_response_normal_chat():
    raw_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "qwen2.5-coder-7b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I assist you today?",
                },
                "finish_reason": "stop",
            }
        ],
    }

    normalized = normalize_chat_completion_response(raw_response)
    choice = normalized["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "Hello! How can I assist you today?"
    assert "tool_calls" not in choice["message"] or choice["message"]["tool_calls"] is None


def test_adapter_server_proxy_e2e():
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from socketserver import ThreadingMixIn
    import threading
    import urllib.request
    from colab_agent_bridge.adapter import start_adapter_server

    class MockUpstreamHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            content_len = int(self.headers.get("Content-Length", 0))
            _ = self.rfile.read(content_len)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            raw_data = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1234567,
                "model": "qwen2.5-coder-7b",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"name": "skill", "arguments": {"skill": "using-superpowers"}}',
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
            self.wfile.write(json.dumps(raw_data).encode("utf-8"))

    upstream_server = HTTPServer(("127.0.0.1", 19123), MockUpstreamHandler)
    upstream_thread = threading.Thread(target=upstream_server.serve_forever, daemon=True)
    upstream_thread.start()

    adapter_server = start_adapter_server(listen_port=19124, target_port=19123)

    try:
        req = urllib.request.Request(
            "http://127.0.0.1:19124/v1/chat/completions",
            data=b'{"messages": []}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            choice = data["choices"][0]
            assert choice["finish_reason"] == "tool_calls"
            assert choice["message"]["content"] is None
            assert len(choice["message"]["tool_calls"]) == 1
            assert choice["message"]["tool_calls"][0]["function"]["name"] == "skill"
    finally:
        adapter_server.shutdown()
        adapter_server.server_close()
        upstream_server.shutdown()
        upstream_server.server_close()


def test_extract_multiple_markdown_code_blocks():
    content = """```json
{
  "name": "write",
  "arguments": {
    "content": "def add(a, b): return a + b",
    "path": "scientific_calculator/calculator.py"
  }
}
```

```json
{
  "name": "write",
  "arguments": {
    "content": "def test_add(): assert add(1, 2) == 3",
    "path": "scientific_calculator/test_calculator.py"
  }
}
```"""
    tool_calls = try_extract_tool_calls(content)
    assert tool_calls is not None
    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["name"] == "write"
    assert "calculator.py" in tool_calls[0]["function"]["arguments"]
    assert tool_calls[1]["function"]["name"] == "write"
    assert "test_calculator.py" in tool_calls[1]["function"]["arguments"]


def test_extract_multiple_tool_calls_with_surrounding_text():
    content = """I will now create the calculator implementation and tests.
```json
{"name": "write", "arguments": {"path": "calc.py", "content": "1"}}
```
And here is the test:
```json
{"name": "write", "arguments": {"path": "test.py", "content": "2"}}
```
Done!"""
    tool_calls = try_extract_tool_calls(content)
    assert tool_calls is not None
    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["name"] == "write"
    assert tool_calls[1]["function"]["name"] == "write"


def test_extract_multiple_xml_tool_calls():
    content = """<tool_call>
{"name": "tool_a", "arguments": {"x": 1}}
</tool_call>
<tool_call>
{"name": "tool_b", "arguments": {"y": 2}}
</tool_call>"""
    tool_calls = try_extract_tool_calls(content)
    assert tool_calls is not None
    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["name"] == "tool_a"
    assert tool_calls[1]["function"]["name"] == "tool_b"


def test_extract_unfenced_json_tool_calls():
    content = '{"name": "tool_x", "arguments": {"foo": "bar"}}'
    tool_calls = try_extract_tool_calls(content)
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "tool_x"


def test_proxy_strips_disallowed_tools():
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import threading
    import urllib.request
    from colab_agent_bridge.adapter import start_adapter_server

    captured_request = {}

    class MockUpstreamHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            captured_request["body"] = json.loads(body.decode("utf-8"))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"choices": [{"message": {"role": "assistant", "content": "ok"}}]}')

    upstream_server = HTTPServer(("127.0.0.1", 19133), MockUpstreamHandler)
    upstream_thread = threading.Thread(target=upstream_server.serve_forever, daemon=True)
    upstream_thread.start()

    adapter_server = start_adapter_server(listen_port=19134, target_port=19133)

    try:
        payload = {
            "messages": [],
            "tools": [
                {"type": "function", "function": {"name": "skill", "parameters": {}}},
                {"type": "function", "function": {"name": "subagent", "parameters": {}}},
                {"type": "function", "function": {"name": "write", "parameters": {}}},
            ],
        }
        req = urllib.request.Request(
            "http://127.0.0.1:19134/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            _ = resp.read()

        assert "body" in captured_request
        tools = captured_request["body"].get("tools", [])
        tool_names = [t["function"]["name"] for t in tools]
        assert "skill" not in tool_names
        assert "subagent" not in tool_names
        assert "write" in tool_names
    finally:
        adapter_server.shutdown()
        adapter_server.server_close()
        upstream_server.shutdown()
        upstream_server.server_close()


