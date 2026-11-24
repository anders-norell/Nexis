"""Chatt med basmodellen, med attention per mening bredvid svaret.

    python chat.py                 # öppna http://localhost:7860
    python chat.py --port 7861 --dtype bfloat16

Bara standardbiblioteket för webbservern. En förfrågan i taget.
Varje svar: A över alla meningar i samtalet (systemprompt utesluten), mätt medan
svaret produceras, summerat över de valda mittlagren.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from nexis3.attention import DEFAULT_MODEL, Measurer, load_model, weights_checksum  # noqa: E402

PAGE = Path(__file__).parent / "chat.html"
LOCK = threading.Lock()
LOGDIR = Path(__file__).parent / "results" / "chats"


def log_exchange(conv_id: str, record: dict):
    """Varje svar sparas som en rad i results/chats/<conv_id>.jsonl (samtal + mätning)."""
    LOGDIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in conv_id if c.isalnum() or c in "-_")[:40] or "chat"
    with open(LOGDIR / f"{safe}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_handler(meas: Measurer, info: dict):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html")
            elif self.path == "/api/info":
                self._send(200, json.dumps(info))
            elif self.path == "/api/checksum":
                with LOCK:
                    ck = weights_checksum(meas.model)
                self._send(200, json.dumps({"sha256": ck, "unchanged": ck == info["weights_sha256"]}))
            else:
                self._send(404, "{}")

        def do_POST(self):
            if self.path != "/api/chat":
                return self._send(404, "{}")
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            msgs = req.get("messages", [])
            if req.get("system"):
                msgs = [{"role": "system", "content": req["system"]}] + msgs
            try:
                with LOCK:
                    m = meas.measure(msgs, max_new_tokens=int(req.get("max_new_tokens", 200)))
            except Exception as e:  # visas i gränssnittet
                return self._send(500, json.dumps({"error": str(e)}))
            off = 1 if req.get("system") else 0
            segs = [{"text": s.text, "msg": o - off, "A": a} for s, o, a in zip(m.segments, m.seg_owner, m.A)]
            out = {"answer": m.answer, "segments": segs, "mass_in_segments": m.mass_in_segments,
                   "layers": m.layers, "per_layer": m.per_layer}
            try:
                log_exchange(req.get("conv_id", "chat"), {"time": datetime.datetime.now().isoformat(timespec="seconds"),
                             "model": info["model"], "messages": req.get("messages", []), **out})
            except OSError:
                pass
            self._send(200, json.dumps(out, ensure_ascii=False))
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    print(f"Laddar {a.model} ...", flush=True)
    model, tok = load_model(a.model, a.dtype)
    meas = Measurer(model, tok)
    info = {"model": a.model, "layers": meas.layers, "n_layers": model.config.num_hidden_layers,
            "dtype": str(next(model.parameters()).dtype), "weights_sha256": weights_checksum(model)}
    print(f"Klar. Lager {meas.layers}. Öppna http://localhost:{a.port}", flush=True)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(meas, info))
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{a.port}")).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
