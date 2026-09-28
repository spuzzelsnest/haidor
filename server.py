#!/usr/bin/env python3
"""
pi-agent web - a small web front end for the pi-agent orchestrator.

Serves a chat UI on http://<pi-ip>:8080. Agent trace events (delegations,
tool calls) are streamed to the browser live while the agent works.

Layout (server.py, agent.py and static/ live in the same folder):
    static/index.html   page structure
    static/style.css    styling
    static/app.js       browser logic
    static/bg.jpg       background image
Edit anything in static/ and just refresh the browser - no restart needed.
Everything is stdlib-only, no pip installs needed.

Usage:
    python3 server.py                     # then open http://<pi-ip>:8080
    PORT=9090 python3 server.py           # custom port
    HOST=127.0.0.1 python3 server.py      # local only (e.g. behind a proxy)
    WEB_PASSWORD=secret python3 server.py # require a password (any username)

Config comes from the same .env file agent.py uses.
"""

import base64
import hmac
import json
import mimetypes
import os
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import agent as pi_agent  # loads .env itself

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8080"))
WEB_PASSWORD = os.environ.get("WEB_PASSWORD", "")
MAX_BODY = 16 * 1024
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

HISTORY = []                    # shared conversation memory (last few exchanges)
AGENT_LOCK = threading.Lock()   # one agent run at a time



class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass  # keep the console quiet

    # ---------------------------------------------------------- helpers ---

    def _send(self, code, body=b"", ctype="application/json", headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode())

    def _authorized(self):
        """Optional HTTP Basic auth (any username, password = WEB_PASSWORD)."""
        if not WEB_PASSWORD:
            return True
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Basic "):
            try:
                _, _, pw = base64.b64decode(auth[6:]).decode().partition(":")
                return hmac.compare_digest(pw, WEB_PASSWORD)
            except Exception:
                pass
        self._send(401, b'{"error": "unauthorized"}',
                   headers={"WWW-Authenticate": 'Basic realm="pi-agent"'})
        return False

    # ------------------------------------------------------------ GET ---

    def _serve_static(self, rel):
        """Serve a file from STATIC_DIR (never anything outside it)."""
        root = os.path.realpath(STATIC_DIR)
        try:
            full = os.path.realpath(os.path.join(root, rel))
            inside = os.path.commonpath([full, root]) == root
            ok = inside and os.path.isfile(full)
        except ValueError:
            ok = False
        if not ok:
            self._json({"error": "not found"}, 404)
            return
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        with open(full, "rb") as f:
            data = f.read()
        # code files are never cached, so edits show up on a plain refresh
        cache = ("no-cache" if full.endswith((".html", ".css", ".js"))
                 else "max-age=86400")
        self._send(200, data, ctype, {"Cache-Control": cache})

    def do_GET(self):
        if not self._authorized():
            return
        path = urlparse(self.path).path
        if path == "/":
            self._serve_static("index.html")
        elif path.startswith("/static/"):
            self._serve_static(unquote(path[len("/static/"):]))
        elif path == "/api/status":
            self._json({"orchestrator": pi_agent.ORCHESTRATOR_MODEL,
                        "analyst": pi_agent.ANALYST_MODEL,
                        "ha": pi_agent.HA_URL})
        else:
            self._json({"error": "not found"}, 404)

    # ----------------------------------------------------------- POST ---

    def do_POST(self):
        if not self._authorized():
            return
        if urlparse(self.path).path != "/api/chat":
            self._json({"error": "not found"}, 404)
            return
        # Requiring JSON content-type forces a CORS preflight for cross-site
        # requests, which we never approve - blocks drive-by CSRF from other
        # web pages triggering your smart home.
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            self._json({"error": "content-type must be application/json"}, 415)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length > MAX_BODY:
            self._json({"error": "message too large"}, 413)
            return
        try:
            msg = str(json.loads(self.rfile.read(length) or b"{}")
                      .get("message", "")).strip()
        except Exception:
            msg = ""
        if not msg:
            self._json({"error": "empty message"}, 400)
            return

        # stream ndjson events to the browser as the agent works
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()

        def send(obj):
            try:
                self.wfile.write((json.dumps(obj) + "\n").encode())
                self.wfile.flush()
            except Exception:
                pass  # browser closed the tab

        events = queue.Queue()

        def worker():
            try:
                with AGENT_LOCK:
                    old_sink = pi_agent.EVENT_SINK
                    pi_agent.EVENT_SINK = events.put
                    try:
                        answer, _ = pi_agent.orchestrator(msg, history=HISTORY)
                        HISTORY.extend([{"role": "user", "content": msg},
                                        {"role": "assistant", "content": answer}])
                        del HISTORY[:-10]
                        events.put(("answer", answer))
                    finally:
                        pi_agent.EVENT_SINK = old_sink
            except Exception as e:
                events.put(("error", str(e)[:300]))
            finally:
                events.put(None)          # always terminate the stream

        threading.Thread(target=worker, daemon=True).start()

        while True:
            item = events.get()
            if item is None:
                break
            if isinstance(item, tuple):       # ("answer"|"error", text)
                send({"type": item[0], "text": item[1]})
            else:                              # trace line
                send({"type": "trace", "text": str(item)})


def main():
    if not os.path.isfile(os.path.join(STATIC_DIR, "index.html")):
        raise SystemExit("static/index.html not found in %s" % STATIC_DIR)
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print("pi-agent web listening on http://%s:%d" % (HOST, PORT))
    if not WEB_PASSWORD and HOST != "127.0.0.1":
        print("WARNING: no WEB_PASSWORD set - anyone on your network can "
              "control your home through this page.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
