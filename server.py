#!/usr/bin/env python3
"""
pi-agent web - a small web front end for the pi-agent orchestrator.

Serves a chat UI on http://<pi-ip>:8080. Agent trace events (delegations,
tool calls) are streamed to the browser live while the agent works.

Requirements: server.py sits in the SAME folder as agent.py. Both are
stdlib-only, no pip installs needed.

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
import os
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import agent as pi_agent  # loads .env itself

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8080"))
WEB_PASSWORD = os.environ.get("WEB_PASSWORD", "")
MAX_BODY = 16 * 1024
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

HISTORY = []                    # shared conversation memory (last few exchanges)
AGENT_LOCK = threading.Lock()   # one agent run at a time

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>pi-agent</title>
<style>
  :root { --bg:#0f1117; --panel:#171a23; --line:#262a38; --text:#e6e8ef;
          --dim:#8b91a5; --accent:#7aa2f7; --user:#2b3a5e; }
  * { box-sizing:border-box; margin:0; padding:0; }
  html, body { height:100%; }
  body::before { content:''; position:fixed; inset:0;
         background:rgba(15,17,23,.75); z-index:-1; }
  body { background: var(--bg) url('/static/bg.jpg') center/cover no-repeat fixed;
         color:var(--text); height:100dvh;
         font:15px/1.5 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
         display:flex; flex-direction:column; }
  header { padding:14px 20px; border-bottom:1px solid var(--line);
           display:flex; align-items:baseline; gap:12px; }
  header h1 { font-size:17px; font-weight:600; }
  header .sub { color:var(--dim); font-size:12px; }
  #chat { flex:1; min-height:0; overflow-y:auto; padding:20px; display:flex;
          flex-direction:column; gap:14px; max-width:820px; width:100%;
          margin:0 auto; }
  .msg { display:flex; }
  .msg.user { justify-content:flex-end; }
  .bubble { max-width:75%; padding:10px 14px; border-radius:12px;
            white-space:pre-wrap; word-wrap:break-word; }
  .msg.user .bubble { background:var(--user); border-bottom-right-radius:4px; }
  .msg.agent .bubble { background:var(--panel); border:1px solid var(--line);
                       border-bottom-left-radius:4px; min-width:120px; }
  .answer:empty { display:none; }
  .trace { font:12px/1.6 ui-monospace,Menlo,Consolas,monospace; color:var(--dim);
           margin-top:8px; padding-left:10px; border-left:2px solid var(--line); }
  .trace:empty { display:none; }
  .trace div { animation:fade .3s; }
  @keyframes fade { from{opacity:0} to{opacity:1} }
  .thinking { color:var(--accent); font-size:13px; margin-top:8px; }
  #dock { border-top:1px solid var(--line); padding:14px 20px 20px;
          max-width:820px; width:100%; margin:0 auto; }
  #chips { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px; }
  #chips button { background:none; border:1px solid var(--line); color:var(--dim);
          border-radius:999px; padding:4px 12px; font-size:12px; cursor:pointer; }
  #chips button:hover { color:var(--text); border-color:var(--accent); }
  #form { display:flex; gap:10px; }
  #input { flex:1; background:var(--panel); border:1px solid var(--line);
           color:var(--text); border-radius:10px; padding:11px 14px;
           font-size:15px; outline:none; }
  #input:focus { border-color:var(--accent); }
  #send { background:var(--accent); border:none; color:#0f1117; font-weight:600;
          border-radius:10px; padding:0 18px; cursor:pointer; font-size:15px; }
  #send:disabled { opacity:.4; cursor:default; }
</style>
</head>
<body>
<header>
  <h1>&#129302; pi-agent</h1>
  <span class="sub" id="status">connecting...</span>
</header>
<div id="chat"></div>
<div id="dock">
  <div id="chips">
    <button>What lights are on?</button>
    <button>Turn everything off downstairs</button>
    <button>How warm is the living room?</button>
    <button>How is the Pi doing?</button>
    <button>Energy usage summary</button>
  </div>
  <form id="form">
    <input id="input" placeholder="Ask your home anything..." autocomplete="off">
    <button id="send">Send</button>
  </form>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const statusEl = document.getElementById('status');
let busy = false;

fetch('/api/status').then(r => r.json()).then(s => {
  statusEl.textContent = s.orchestrator + ' \u2192 ' + s.ha;
}).catch(() => { statusEl.textContent = 'offline'; });

const scrollDown = () => { chat.scrollTop = chat.scrollHeight; };

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text) e.textContent = text;
  return e;
}

function addUser(text) {
  const m = el('div', 'msg user');
  m.appendChild(el('div', 'bubble', text));
  chat.appendChild(m); scrollDown();
}

// An agent bubble has three parts: answer text, live trace, thinking timer.
function addAgent() {
  const m = el('div', 'msg agent'), bubble = el('div', 'bubble');
  const parts = { answer: el('div', 'answer'), trace: el('div', 'trace'),
                  think: el('div', 'thinking', 'thinking... 0s') };
  bubble.append(parts.answer, parts.trace, parts.think);
  m.appendChild(bubble); chat.appendChild(m); scrollDown();
  return parts;
}

async function ask(text) {
  text = text.trim();
  if (busy || !text) return;
  busy = true; sendBtn.disabled = true;
  addUser(text);
  const ui = addAgent();
  let secs = 0, gotAnswer = false;
  const timer = setInterval(() => {
    ui.think.textContent = 'thinking... ' + (++secs) + 's';
  }, 1000);

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text})
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let i;
      while ((i = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
        if (!line) continue;
        const ev = JSON.parse(line);
        if (ev.type === 'trace') {
          ui.trace.appendChild(el('div', '', ev.text.replace(/^\s+/, '')));
        } else if (ev.type === 'answer') {
          ui.answer.textContent = ev.text || '(empty answer)'; gotAnswer = true;
        } else if (ev.type === 'error') {
          ui.answer.textContent = '\u26a0 ' + ev.text; gotAnswer = true;
        }
        scrollDown();
      }
    }
    if (!gotAnswer) ui.answer.textContent = '(no response)';
  } catch (e) {
    ui.answer.textContent = '\u26a0 request failed: ' + e.message;
  } finally {
    clearInterval(timer); ui.think.remove();
    busy = false; sendBtn.disabled = false; input.focus(); scrollDown();
  }
}

document.getElementById('form').addEventListener('submit', e => {
  e.preventDefault(); const t = input.value; input.value = ''; ask(t);
});
document.querySelectorAll('#chips button').forEach(b =>
  b.addEventListener('click', () => ask(b.textContent)));
input.focus();
</script>
</body>
</html>
"""


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

    def do_GET(self):
        if not self._authorized():
            return
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, HTML.encode(), "text/html; charset=utf-8")
        elif path == "/api/status":
            self._json({"orchestrator": pi_agent.ORCHESTRATOR_MODEL,
                        "analyst": pi_agent.ANALYST_MODEL,
                        "ha": pi_agent.HA_URL})
        elif path == "/static/bg.jpg":
            try:
                with open(os.path.join(STATIC_DIR, "bg.jpg"), "rb") as f:
                    data = f.read()
            except OSError:
                self._json({"error": "not found"}, 404)
                return
            self._send(200, data, "image/jpeg",
                       {"Cache-Control": "max-age=86400"})
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
