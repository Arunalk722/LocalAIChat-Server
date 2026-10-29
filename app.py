import argparse
import json
import time
import re
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ============================================================
# Configuration
# ============================================================

MODEL_PATH = os.environ.get("LOCAL_GGUF_MODEL")
model = None
model_name = "local-gguf"
model_lock = threading.Lock()


def load_model(model_path):
    global model, model_name

    if not model_path:
        raise RuntimeError(
            "Pass the path to the main language-model .gguf file with --model-path. "
        
        )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"GGUF model file not found: {model_path}")
    if not model_path.lower().endswith(".gguf") or os.path.basename(model_path).lower().startswith("mmproj-"):
        raise ValueError("--model-path must point to the main language-model GGUF, not an mmproj file.")

    try:
        from llama_cpp import Llama
    except ImportError as error:
        raise RuntimeError(
            "Direct GGUF loading requires llama-cpp-python. Install it in this venv with: "
            "pip install llama-cpp-python"
        ) from error

    print("=" * 60)
    print("Local GGUF AI")
    print("=" * 60)
    print(f"Loading model: {model_path}")
    model_name = os.path.basename(model_path)
    model = Llama(model_path=model_path, n_ctx=4096, verbose=False)
    print("Model loaded successfully!")

# ============================================================
# Simple Conversation Layer
# ============================================================

def handle_simple_question(text):

    question = text.lower().strip()

    # Name
    if question in [
        "what is your name",
        "what's your name",
        "who are you",
        "tell me your name"
    ]:
        return "I'm a local AI assistant."

    # Greeting
    if question in [
        "hi",
        "hello",
        "hey",
        "hi there",
        "hello there"
    ]:
        return "Hello! How can I help you?"

    # How are you
    if (
        "how are you" in question
        or "how are you doing" in question
    ):
        return "I'm doing well. How can I help you?"

    # Thanks
    if question in [
        "thanks",
        "thank you",
        "thank you very much"
    ]:
        return "You're welcome!"

    # Goodbye
    if question in [
        "bye",
        "goodbye",
        "see you"
    ]:
        return "Goodbye!"

    return None


# ============================================================
# Detect Programming Request
# ============================================================

def is_programming_question(text):

    programming_keywords = [
        "write code",
        "write a code",
        "generate code",
        "create code",
        "python",
        "c#",
        "csharp",
        "java",
        "javascript",
        "typescript",
        "sql",
        "mysql",
        "mssql",
        "html",
        "css",
        "react",
        "flutter",
        "function",
        "class",
        "algorithm",
        "program",
        "debug",
        "bug",
        "error",
        "api",
        "database",
        "query",
        "code"
    ]

    text_lower = text.lower()

    return any(
        keyword in text_lower
        for keyword in programming_keywords
    )


# ============================================================
# Generate Code
# ============================================================

def generate_code(prompt):
    if model is None:
        raise RuntimeError("The model has not been loaded.")

    with model_lock:
        response = model.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful local assistant. Answer clearly and use fenced Markdown code blocks when providing code.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
    try:
        answer = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("The local GGUF model returned an unexpected response.") from error
    if not isinstance(answer, str):
        raise RuntimeError("The local GGUF model returned a non-text response.")
    return re.sub(r"\n{3,}", "\n\n", answer).strip()


def parse_response_segments(response, default_language="text"):
    segments = []
    code_fence = re.compile(r"```([\w+#.-]*)\s*\r?\n([\s\S]*?)(?:```|$)")
    cursor = 0

    for match in code_fence.finditer(response):
        message = response[cursor:match.start()].strip()
        if message:
            segments.append({"type": "message", "content": message})

        code = match.group(2).strip("\r\n")
        if code:
            segments.append({
                "type": "code",
                "language": match.group(1) or default_language,
                "content": code
            })
        cursor = match.end()

    trailing_message = response[cursor:].strip()
    if trailing_message:
        segments.append({"type": "message", "content": trailing_message})

    if not segments and response.strip():
        segments.append({"type": "message", "content": response.strip()})

    return segments


# ============================================================
# Shared chat behavior
# ============================================================

def get_response(user_input):
    return generate_code(user_input)


# ============================================================
# REST API
# ============================================================

class ChatRequestHandler(BaseHTTPRequestHandler):
    allowed_origins = {
        "http://localhost:4200",
        "http://127.0.0.1:4200"
    }

    def _set_cors_headers(self):
        origin = self.headers.get("Origin")
        if origin in self.allowed_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, MCP-Protocol-Version")

    def _send_json(self, status_code, payload):
        response = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(response)

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == "/v1/models":
            self._send_json(200, {
                "object": "list",
                "data": [{
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                }],
            })
            return

        if self.path != "/health":
            self._send_json(404, {"error": {"message": "Not found", "type": "invalid_request_error"}})
            return

        self._send_json(200, {
            "status": "ok",
            "model_loaded": model is not None
        })

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._handle_openai_chat_completion()
            return
        if self.path != "/mcp":
            self._send_json(404, {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Use the /mcp endpoint."}})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 1_000_000:
                self._send_json(400, {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Request body must be between 1 byte and 1 MB."}})
                return

            request_data = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Request body must be valid JSON."}})
            return

        if not isinstance(request_data, dict) or request_data.get("jsonrpc") != "2.0":
            self._send_json(400, {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid JSON-RPC request."}})
            return

        request_id = request_data.get("id")
        method = request_data.get("method")
        if request_id is None and isinstance(method, str) and method.startswith("notifications/"):
            self.send_response(202)
            self._set_cors_headers()
            self.end_headers()
            return

        if "id" not in request_data or not isinstance(method, str):
            self._send_json(400, {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32600, "message": "A request id and method are required."}})
            return

        params = request_data.get("params") or {}
        if method == "initialize":
            result = {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "local-gguf-chat", "version": "1.0.0"},
            }
        elif method == "tools/list":
            result = {
                "tools": [{
                    "name": "chat",
                    "description": "Send a message to the locally loaded GGUF model.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"message": {"type": "string", "description": "The user message."}},
                        "required": ["message"],
                    },
                }]
            }
        elif method == "tools/call":
            if params.get("name") != "chat":
                self._send_json(200, {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32602, "message": "Unknown tool."}})
                return
            arguments = params.get("arguments") or {}
            message = arguments.get("message") if isinstance(arguments, dict) else None
            if not isinstance(message, str) or not message.strip():
                self._send_json(200, {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32602, "message": "A non-empty message string is required."}})
                return
            try:
                answer = get_response(message.strip())
                result = {"content": [{"type": "text", "text": answer}], "isError": False}
            except Exception as error:
                self.log_error("MCP chat tool failed: %s", error)
                result = {"content": [{"type": "text", "text": "Local model generation failed."}], "isError": True}
        else:
            self._send_json(200, {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": "Method not found."}})
            return

        self._send_json(200, {"jsonrpc": "2.0", "id": request_id, "result": result})

    def _handle_openai_chat_completion(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 1_000_000:
                self._send_json(400, {"error": {"message": "Request body must be between 1 byte and 1 MB.", "type": "invalid_request_error"}})
                return
            request_data = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": {"message": "Request body must be valid JSON.", "type": "invalid_request_error"}})
            return

        messages = request_data.get("messages") if isinstance(request_data, dict) else None
        if not isinstance(messages, list) or not messages:
            self._send_json(400, {"error": {"message": "A non-empty messages array is required.", "type": "invalid_request_error"}})
            return
        if any(not isinstance(message, dict) or not isinstance(message.get("role"), str) for message in messages):
            self._send_json(400, {"error": {"message": "Each message must include a role.", "type": "invalid_request_error"}})
            return

        parameters = {
            "messages": messages,
            "temperature": request_data.get("temperature", 0.2),
            "max_tokens": request_data.get("max_tokens", request_data.get("max_completion_tokens", 1024)),
            "stream": bool(request_data.get("stream", False)),
        }
        for key in ("top_p", "stop", "presence_penalty", "frequency_penalty", "seed", "tools", "tool_choice"):
            if key in request_data:
                parameters[key] = request_data[key]

        try:
            if parameters["stream"]:
                response = None
            else:
                with model_lock:
                    response = model.create_chat_completion(**parameters)
        except Exception as error:
            self.log_error("OpenAI-compatible completion failed: %s", error)
            self._send_json(500, {"error": {"message": "Local model generation failed.", "type": "server_error"}})
            return

        if parameters["stream"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self._set_cors_headers()
            self.end_headers()
            try:
                with model_lock:
                    response = model.create_chat_completion(**parameters)
                    for chunk in response:
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                self.log_message("Streaming client disconnected")
            except Exception as error:
                self.log_error("OpenAI-compatible streaming completion failed: %s", error)
            return

        self._send_json(200, response)


def run_api(host, port):
    server = ThreadingHTTPServer((host, port), ChatRequestHandler)
    print(f"MCP server listening at http://{host}:{port}/mcp")
    print("Health endpoint: GET /health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping REST API.")
    finally:
        server.server_close()


# ============================================================
# CLI
# ============================================================

def run_cli():
    print("\n")
    print("=" * 60)
    print("AI READY")
    print("=" * 60)
    print("Type 'exit' to quit.")
    print("=" * 60)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("AI: Goodbye!")
            break
        print(f"AI: {get_response(user_input)}")


def main():
    parser = argparse.ArgumentParser(description="Run a local GGUF model with an MCP chat API.")
    parser.add_argument("--cli", action="store_true", help="Use the interactive console instead of the API.")
    parser.add_argument("--host", default="127.0.0.1", help="API bind address (default: localhost only).")
    parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000).")
    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        help="Path to the main language-model GGUF file (not an mmproj file)."
    )
    args = parser.parse_args()

    load_model(args.model_path)
    if args.cli:
        run_cli()
    else:
        run_api(args.host, args.port)


if __name__ == "__main__":
    main()