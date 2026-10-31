import json
import re
import uuid
import select
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import urllib.request
import urllib.error
from typing import Optional, List, Dict, Any, Union


def _make_tool_call(name: str, arguments: Any) -> Dict[str, Any]:
    arg_str = json.dumps(arguments) if isinstance(arguments, (dict, list)) else str(arguments)
    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": arg_str,
        },
    }


def try_extract_tool_calls(content: str) -> Optional[List[Dict[str, Any]]]:
    """Inspect text content to check if it contains JSON tool calls.

    Handles:
    - Pure JSON dict: {"name": "...", "arguments": {...}}
    - XML wrapped: <tool_call>{"name": ...}</tool_call>
    - Markdown fenced code blocks: ```json {"name": ...} ```
    - List of tool calls: [{"name": ...}, ...]
    """
    if not content or not isinstance(content, str):
        return None

    cleaned = content.strip()

    # 1. Strip XML <tool_call> tags if present
    match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()

    # 2. Strip Markdown code blocks ```json ... ``` or ``` ... ```
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
            cleaned = "\n".join(lines[1:-1]).strip()

    # 3. Quick check if it looks like JSON object or array
    if not (cleaned.startswith("{") and cleaned.endswith("}")) and not (
        cleaned.startswith("[") and cleaned.endswith("]")
    ):
        return None

    try:
        parsed = json.loads(cleaned)
    except Exception:
        return None

    # Case A: Single dict with 'name' and ('arguments' or 'parameters')
    if isinstance(parsed, dict):
        name = parsed.get("name")
        args = parsed.get("arguments", parsed.get("parameters"))
        if name and isinstance(name, str) and args is not None:
            return [_make_tool_call(name, args)]

    # Case B: List of dicts with 'name' and ('arguments' or 'parameters')
    if isinstance(parsed, list) and len(parsed) > 0:
        calls = []
        for item in parsed:
            if isinstance(item, dict):
                name = item.get("name")
                args = item.get("arguments", item.get("parameters"))
                if name and isinstance(name, str) and args is not None:
                    calls.append(_make_tool_call(name, args))
                else:
                    return None
            else:
                return None
        if calls:
            return calls

    return None


def normalize_chat_completion_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Promotes JSON in message.content to real OpenAI tool_calls if detected."""
    choices = response.get("choices", [])
    if not choices or not isinstance(choices, list):
        return response

    for choice in choices:
        message = choice.get("message", {})
        content = message.get("content")
        tool_calls = message.get("tool_calls")

        # If already has tool_calls, nothing to normalize
        if tool_calls:
            continue

        if content and isinstance(content, str):
            extracted = try_extract_tool_calls(content)
            if extracted:
                message["tool_calls"] = extracted
                message["content"] = None
                choice["finish_reason"] = "tool_calls"

    return response


def normalize_chat_completion_stream_chunk(chunk_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Normalizes an individual SSE chunk if it contains a complete tool call."""
    choices = chunk_dict.get("choices", [])
    if not choices or not isinstance(choices, list):
        return chunk_dict

    for choice in choices:
        delta = choice.get("delta", {})
        content = delta.get("content")
        if content and isinstance(content, str):
            extracted = try_extract_tool_calls(content)
            if extracted:
                choice["delta"] = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": extracted[0]["id"],
                            "type": "function",
                            "function": {
                                "name": extracted[0]["function"]["name"],
                                "arguments": extracted[0]["function"]["arguments"],
                            },
                        }
                    ],
                }
                choice["finish_reason"] = "tool_calls"

    return chunk_dict


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def start_adapter_server(listen_port: int, target_port: int) -> ThreadedHTTPServer:
    """Spins up a background threaded HTTP proxy normalizing OpenAI tool calls."""
    import threading

    class BoundHandler(SmartAdapterHandler):
        pass

    BoundHandler.target_port = target_port

    server = ThreadedHTTPServer(("127.0.0.1", listen_port), BoundHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    return server


class SmartAdapterHandler(BaseHTTPRequestHandler):
    """Transparent proxy handler that forwards requests to upstream and normalizes responses."""

    target_port: int = 8000

    def log_message(self, format, *args):
        # Quiet down standard HTTP access logs
        pass

    def do_GET(self):
        self._proxy_request("GET")

    def do_POST(self):
        self._proxy_request("POST")

    def do_HEAD(self):
        self._proxy_request("HEAD")

    def _proxy_request(self, method: str):
        target_url = f"http://127.0.0.1:{self.target_port}{self.path}"
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else None

        req_headers = {k: v for k, v in self.headers.items() if k.lower() != "host"}
        req_headers["Host"] = f"127.0.0.1:{self.target_port}"

        req = urllib.request.Request(target_url, data=body, headers=req_headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                resp_headers = resp.info()
                content_type = resp_headers.get("Content-Type", "")

                # If non-streaming chat completions JSON, normalize tool calls
                if (
                    "application/json" in content_type
                    and "/chat/completions" in self.path
                    and method == "POST"
                ):
                    raw_body = resp.read()
                    try:
                        data = json.loads(raw_body.decode("utf-8"))
                        normalized = normalize_chat_completion_response(data)
                        output_bytes = json.dumps(normalized).encode("utf-8")
                    except Exception:
                        output_bytes = raw_body

                    self.send_response(resp.status)
                    for k, v in resp_headers.items():
                        if k.lower() not in ["content-length", "transfer-encoding"]:
                            self.send_header(k, v)
                    self.send_header("Content-Length", str(len(output_bytes)))
                    self.end_headers()
                    self.wfile.write(output_bytes)
                else:
                    # Stream or raw pass-through
                    self.send_response(resp.status)
                    for k, v in resp_headers.items():
                        if k.lower() != "transfer-encoding":
                            self.send_header(k, v)
                    self.end_headers()

                    while True:
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()

        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for k, v in e.headers.items():
                if k.lower() != "transfer-encoding":
                    self.send_header(k, v)
            self.end_headers()
            err_body = e.read()
            self.wfile.write(err_body)
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            err_msg = json.dumps({"error": f"Bridge Adapter error: {str(e)}"}).encode("utf-8")
            self.send_header("Content-Length", str(len(err_msg)))
            self.end_headers()
            self.wfile.write(err_msg)
