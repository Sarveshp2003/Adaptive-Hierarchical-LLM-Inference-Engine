import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "build" / "Release" / "runtime_bridge_cli.exe"
if not CLI.exists():
    CLI = ROOT / "build" / "Release" / "runtime_bridge_cli"

class Handler(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok"})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            body = json.loads(data)
        except Exception as exc:
            self._send(400, {"error": f"invalid json: {exc}"})
            return

        action = body.get("action")
        if action == "start":
            subprocess.run([str(CLI), "start"], check=False, cwd=str(ROOT))
            self._send(200, {"ok": True, "action": "start"})
        elif action == "stop":
            subprocess.run([str(CLI), "stop"], check=False, cwd=str(ROOT))
            self._send(200, {"ok": True, "action": "stop"})
        elif action == "request":
            layer = int(body.get("layer_id", 0))
            subprocess.run([str(CLI), "request", str(layer)], check=False, cwd=str(ROOT))
            self._send(200, {"ok": True, "action": "request", "layer_id": layer})
        else:
            self._send(400, {"error": "unknown action"})

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"RPC bridge listening on http://127.0.0.1:{port}")
    server.serve_forever()
