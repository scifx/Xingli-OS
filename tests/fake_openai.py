"""一个极小的 OpenAI 兼容服务器：用来验证真实端点路径（离线、确定性）。

它模仿 llama.cpp / vLLM 的行为：``GET /v1/models``、``POST /v1/chat/completions``。
测试里可以注入任意返回，用来验证「不支持 response_format」「返回围栏 JSON」
「只给 reasoning_content」「401」等真实世界里的毛病。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

Reply = tuple[int, dict[str, Any]]
Handler = Callable[[dict[str, Any], dict[str, str]], Reply | dict[str, Any]]


def chat_reply(content: str | None, *, reasoning: str | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "model": "fake-local",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


def packet(**overrides: Any) -> dict[str, Any]:
    data = {
        "inner_monologue": "先听清楚她在说什么，再决定要不要记下来。",
        "focus_update": None,
        "emotion_delta": {"warmth": 0.04},
        "memory_ops": [],
        "relation_ops": [],
        "say": "收到，我在这里。",
        "wake": {"continue_think": None, "schedule_wake": None},
        "code_action": None,
        "evolve_ops": [],
    }
    data.update(overrides)
    return data


class FakeOpenAI:
    """默认行为：正常返回一个认知包；``handler`` 可覆盖。"""

    def __init__(self, handler: Handler | None = None, model: str = "fake-local"):
        self.handler = handler
        self.model = model
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.paths: list[str] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._build())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    # -- lifecycle ------------------------------------------------------
    def start(self) -> "FakeOpenAI":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)

    def __enter__(self) -> "FakeOpenAI":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    @property
    def calls(self) -> int:
        return len(self.requests)

    def last_content(self) -> str:
        messages = self.requests[-1]["messages"]
        return str(messages[-1].get("content", ""))

    # -- HTTP -----------------------------------------------------------
    def _build(self) -> type[BaseHTTPRequestHandler]:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # 安静
                return

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                outer.paths.append(self.path)
                if self.path.rstrip("/").endswith("/models"):
                    self._send(
                        200,
                        {
                            "object": "list",
                            "data": [{"id": outer.model, "object": "model"}],
                        },
                    )
                else:
                    self._send(404, {"error": {"message": "not found"}})

            def do_POST(self) -> None:  # noqa: N802
                outer.paths.append(self.path)
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8") if length else "{}"
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    payload = {"_raw": raw}
                outer.requests.append(payload)
                outer.headers.append({k.lower(): v for k, v in self.headers.items()})
                reply: Reply | dict[str, Any]
                if outer.handler is None:
                    reply = chat_reply(
                        json.dumps(packet(), ensure_ascii=False),
                    )
                else:
                    reply = outer.handler(payload, outer.headers[-1])
                if isinstance(reply, tuple):
                    status, body = reply
                else:
                    status, body = 200, reply
                self._send(status, body)

        return Handler
