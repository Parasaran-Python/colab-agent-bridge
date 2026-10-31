import json
import re
import uuid
import select
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import urllib.request
import urllib.error
from typing import Optional, List, Dict, Any, Union


TOOL_ALIASES = {
    "create": "write",
    "create_file": "write",
    "new_file": "write",
    "write_file": "write",
    "new": "write",
    "touch": "write",
    "read_file": "read",
    "view": "read",
    "view_file": "read",
    "cat": "read",
    "open_file": "read",
    "edit_file": "edit",
    "modify_file": "edit",
    "update_file": "edit",
    "terminal": "bash",
    "cmd": "bash",
    "run_command": "bash",
    "execute_command": "bash",
    "exec_command": "bash",
}


def _normalize_tool_arguments(name: str, args: Any) -> Any:
    if not isinstance(args, dict):
        return args

    # For write / edit / read file paths
    if name in ("write", "edit", "read", "patch"):
        if "path" not in args:
            for alt in ("file_path", "filepath", "filename", "file", "target"):
                if alt in args:
                    args["path"] = args.pop(alt)
                    break
        if name in ("write", "edit") and "content" not in args:
            for alt in ("text", "data", "code", "body"):
                if alt in args:
                    args["content"] = args.pop(alt)
                    break

    # For bash / shell command
    if name in ("bash", "shell"):
        if "command" not in args:
            for alt in ("cmd", "script", "code"):
                if alt in args:
                    args["command"] = args.pop(alt)
                    break

    return args


def _make_tool_call(name: str, arguments: Any) -> Dict[str, Any]:
    norm_name = TOOL_ALIASES.get(name.lower().strip(), name)
    if isinstance(arguments, dict):
        norm_args = _normalize_tool_arguments(norm_name, arguments)
        arg_str = json.dumps(norm_args)
    elif isinstance(arguments, list):
        arg_str = json.dumps(arguments)
    elif isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            norm_args = _normalize_tool_arguments(norm_name, parsed)
            arg_str = json.dumps(norm_args)
        except Exception:
            arg_str = json.dumps(arguments)
    else:
        arg_str = json.dumps(arguments)

    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": norm_name,
            "arguments": arg_str,
        },
    }


DISALLOWED_TOOLS = {"skill", "subagent", "question"}


def _is_tool_dict(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if "name" in item and isinstance(item["name"], str) and ("arguments" in item or "parameters" in item):
        return True
    if item.get("type") == "function" and isinstance(item.get("function"), dict):
        fn = item["function"]
        if "name" in fn and isinstance(fn["name"], str):
            return True
    if "function" in item and isinstance(item["function"], dict):
        fn = item["function"]
        if "name" in fn and isinstance(fn["name"], str):
            return True
    return False


def _extract_from_dict(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not _is_tool_dict(item):
        return None
    if "name" in item:
        name = item["name"]
        args = item.get("arguments", item.get("parameters", {}))
        return _make_tool_call(name, args)
    if "function" in item and isinstance(item["function"], dict):
        fn = item["function"]
        name = fn.get("name")
        args = fn.get("arguments", fn.get("parameters", {}))
        return _make_tool_call(name, args)
    return None


def try_extract_tool_calls(content: str) -> Optional[List[Dict[str, Any]]]:
    """Inspect text content to check if it contains JSON tool calls.

    Handles:
    - Pure JSON dict: {"name": "...", "arguments": {...}}
    - XML wrapped: <tool_call>{"name": ...}</tool_call>
    - Single or multiple Markdown fenced code blocks: ```json {"name": ...} ```
    - List of tool calls: [{"name": ...}, ...]
    - Raw un-fenced JSON objects embedded in conversational text
    """
    if not content or not isinstance(content, str):
        return None

    cleaned = content.strip()
    calls: List[Dict[str, Any]] = []

    # Strategy 1: Check for XML <tool_call> tags
    xml_matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", cleaned, re.DOTALL)
    if xml_matches:
        for m in xml_matches:
            m_clean = m.strip()
            if m_clean.startswith("```"):
                m_inner = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", m_clean)
                if m_inner:
                    m_clean = m_inner.group(1).strip()
            try:
                parsed = json.loads(m_clean)
                if isinstance(parsed, list):
                    for it in parsed:
                        c = _extract_from_dict(it)
                        if c:
                            calls.append(c)
                elif isinstance(parsed, dict):
                    c = _extract_from_dict(parsed)
                    if c:
                        calls.append(c)
            except Exception:
                pass
        if calls:
            return calls

    # Strategy 2: Check for markdown code blocks ```json ... ``` or ``` ... ```
    md_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if md_matches:
        for m in md_matches:
            try:
                parsed = json.loads(m.strip())
                if isinstance(parsed, list):
                    for it in parsed:
                        c = _extract_from_dict(it)
                        if c:
                            calls.append(c)
                elif isinstance(parsed, dict):
                    c = _extract_from_dict(parsed)
                    if c:
                        calls.append(c)
            except Exception:
                pass
        if calls:
            return calls

    # Strategy 3: Check if the whole cleaned string is a JSON object or array
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            for it in parsed:
                c = _extract_from_dict(it)
                if c:
                    calls.append(c)
            if calls:
                return calls
        elif isinstance(parsed, dict):
            c = _extract_from_dict(parsed)
            if c:
                return [c]
    except Exception:
        pass

    # Strategy 4: Raw JSON decoder scan (for un-fenced JSON objects in conversational text)
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(cleaned):
        match = re.search(r"[\{\[]", cleaned[pos:])
        if not match:
            break
        idx = pos + match.start()
        try:
            parsed, end_idx = decoder.raw_decode(cleaned, idx)
            if isinstance(parsed, dict):
                c = _extract_from_dict(parsed)
                if c:
                    calls.append(c)
                    pos = end_idx
                    continue
            elif isinstance(parsed, list):
                has_calls = False
                for it in parsed:
                    c = _extract_from_dict(it)
                    if c:
                        calls.append(c)
                        has_calls = True
                if has_calls:
                    pos = end_idx
                    continue
        except Exception:
            pass
        pos = idx + 1

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

        # If already has tool_calls, normalize tool names (e.g. create -> write)
        if tool_calls and isinstance(tool_calls, list):
            for tc in tool_calls:
                fn = tc.get("function", {})
                tname = fn.get("name", "")
                if tname and tname.lower().strip() in TOOL_ALIASES:
                    norm = TOOL_ALIASES[tname.lower().strip()]
                    fn["name"] = norm
                    arg_raw = fn.get("arguments")
                    if isinstance(arg_raw, str):
                        try:
                            parsed_args = json.loads(arg_raw)
                            norm_args = _normalize_tool_arguments(norm, parsed_args)
                            fn["arguments"] = json.dumps(norm_args)
                        except Exception:
                            pass
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
                            "index": idx,
                            "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"],
                            },
                        }
                        for idx, tc in enumerate(extracted)
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
        import time

        target_url = f"http://127.0.0.1:{self.target_port}{self.path}"
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else None

        client_wants_stream = False
        forward_body = body

        if method == "POST" and "/chat/completions" in self.path and body:
            try:
                req_json = json.loads(body.decode("utf-8"))
                if req_json.get("stream") is True:
                    client_wants_stream = True
                    req_json["stream"] = False

                # Strip disorienting or prohibited tools so model never hallucinates calling them
                if "tools" in req_json and isinstance(req_json["tools"], list):
                    filtered_tools = []
                    for t in req_json["tools"]:
                        tname = ""
                        if isinstance(t, dict):
                            fn = t.get("function")
                            if isinstance(fn, dict):
                                tname = fn.get("name", "")
                            elif "name" in t:
                                tname = t.get("name", "")
                        if tname.lower() not in DISALLOWED_TOOLS:
                            filtered_tools.append(t)
                    req_json["tools"] = filtered_tools

                forward_body = json.dumps(req_json).encode("utf-8")
            except Exception:
                pass

        req_headers = {k: v for k, v in self.headers.items() if k.lower() != "host"}
        req_headers["Host"] = f"127.0.0.1:{self.target_port}"
        if forward_body is not None:
            req_headers["Content-Length"] = str(len(forward_body))

        req = urllib.request.Request(target_url, data=forward_body, headers=req_headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                resp_headers = resp.info()
                content_type = resp_headers.get("Content-Type", "")

                # Normalize chat completions
                if (
                    "application/json" in content_type
                    and "/chat/completions" in self.path
                    and method == "POST"
                ):
                    raw_body = resp.read()
                    try:
                        data = json.loads(raw_body.decode("utf-8"))
                        normalized = normalize_chat_completion_response(data)
                    except Exception:
                        normalized = None

                    if client_wants_stream and normalized:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "keep-alive")
                        self.end_headers()

                        choice = normalized.get("choices", [{}])[0]
                        message = choice.get("message", {})
                        tool_calls = message.get("tool_calls")
                        content = message.get("content")
                        finish_reason = choice.get("finish_reason", "stop")
                        msg_id = normalized.get("id", f"chatcmpl-{uuid.uuid4().hex[:8]}")
                        created = normalized.get("created", int(time.time()))
                        model = normalized.get("model", "")

                        # 1. Initial role chunk
                        c1 = {
                            "id": msg_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                        }
                        self.wfile.write(f"data: {json.dumps(c1)}\n\n".encode("utf-8"))

                        # 2. Tool calls or content
                        if tool_calls:
                            delta_tc = []
                            for idx, tc in enumerate(tool_calls):
                                args = tc["function"]["arguments"]
                                arg_str = args if isinstance(args, str) else json.dumps(args)
                                delta_tc.append({
                                    "index": idx,
                                    "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                                    "type": "function",
                                    "function": {
                                        "name": tc["function"]["name"],
                                        "arguments": arg_str,
                                    },
                                })
                            c2 = {
                                "id": msg_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{"index": 0, "delta": {"tool_calls": delta_tc}, "finish_reason": None}],
                            }
                            self.wfile.write(f"data: {json.dumps(c2)}\n\n".encode("utf-8"))
                        elif content is not None:
                            c2 = {
                                "id": msg_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                            }
                            self.wfile.write(f"data: {json.dumps(c2)}\n\n".encode("utf-8"))

                        # 3. Final chunk with finish_reason
                        c3 = {
                            "id": msg_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                        }
                        self.wfile.write(f"data: {json.dumps(c3)}\n\n".encode("utf-8"))
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        return

                    # Return standard non-streaming JSON
                    output_bytes = json.dumps(normalized).encode("utf-8") if normalized else raw_body
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
