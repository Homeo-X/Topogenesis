from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .state import GameBridgeState


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


class BridgeHandler(BaseHTTPRequestHandler):
    state = GameBridgeState()

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/health":
            _json_response(self, 200, {"ok": True, "service": "topogenesis-bridge"})
            return
        if self.path == "/snapshot":
            _json_response(self, 200, self.state.snapshot())
            return
        _json_response(self, 404, {"error": "unknown endpoint"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            _json_response(self, 400, {"error": "invalid json"})
            return
        if self.path == "/step":
            _json_response(self, 200, self.state.step(payload if isinstance(payload, dict) else {}))
            return
        if self.path == "/restore":
            self.state.restore(payload if isinstance(payload, dict) else {})
            _json_response(self, 200, self.state.snapshot())
            return
        _json_response(self, 404, {"error": "unknown endpoint"})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Topogenesis Godot bridge server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), BridgeHandler)
    print(f"[topogenesis-bridge] listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
