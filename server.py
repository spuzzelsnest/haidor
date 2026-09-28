#!/usr/bin/env python3
"""
pi-agent web — a small web front end for the pi-agent orchestrator.

Serves a chat UI on http://<pi-ip>:8080. Agent trace events (delegations,
tool calls) are streamed to the browser live while the agent works.

Requirements: server.py sits in the SAME folder as agent.py. Both are
stdlib-only, no pip installs needed.

Usage:
    python3 server.py            # then open http://<pi-ip>:8080
    PORT=9090 python3 server.py  # custom port

Config comes from the same .env file agent.py uses.
"""

import json
import os
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import agent as pi_agent  # noqa: E402  (loads .env itself)

PORT = int(os.environ.get("PORT", "8080"))
HISTORY = []              # shared conversation memory (last few exchanges)
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
  body::before { content:''; position:fixed; inset:0;
  background:rgba(15,17,23,.75); z-index:-1; }
  body { background:var(--bg); url('/static/bg.jpg') center/cover no-repeat fixed;
         font:15px/1.5 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
         display:flex; flex-direction:column; }
  header { padding:14px 20px; border-bottom:1px solid var(--line);
           display:flex; align-items:baseline; gap:12px; }
  header h1 { font-size:17px; font-weight:600; }
  header .sub { color:var(--dim); font-size:12px; }
  #chat { flex:1; overflow-y:auto; padding:20px; display:flex;
          flex-direction:column; gap:14px; max-width:820px; width:100%;
          margin:0 auto; }
  .msg { display:flex; }
  .msg.user { justify-content:flex-end; }
  .bubble { max-width:75%; padding:10px 14px; border-radius:12px;
            white-space:pre-wrap; word-wrap:break-word; }
  .msg.user .bubble { background:var(--user); border-bottom-right-radius:4px; }
  .msg.agent .bubble { background:var(--panel); border:1px solid var(--line);
                       border-bottom-left-radius:4px; min-width:120px; }
  .trace { font:12px/1.6 ui-monospace,Menlo,Consolas,monospace; color:var(--dim);
           margin-top:8px; padding-left:10px; border-left:2px solid var(--line); }
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
let busy = false, timer = null;

fetch('/api/status').then(r => r.json()).then(s => {
  statusEl.textContent = s.orchestrator + ' \u2192 ' + s.ha;
}).catch(() => { statusEl.textContent = 'offline'; });

function add(cls, text) {
  const m = document.createElement('div'); m.className = 'msg ' + cls;
  const b = document.createElement('div'); b.className = 'bubble';
  b.textContent = text; m.appendChild(b); chat.appendChild(m);
  chat.scrollTop = chat.scrollHeight; return b;
}

function addTrace(text) {
  const bubble = chat.querySelector('.msg.agent:last-child .bubble');
  if (!bubble) return;
  let t = bubble.querySelector('.trace');
  if (!t) { t = document.createElement('div'); t.className = 'trace';
            bubble.appendChild(t); }
  const line = document.createElement('div');
  line.textContent = text.replace(/^\s+/, '');
  t.appendChild(line); chat.scrollTop = chat.scrollHeight;
}

async function ask(text) {
  if (busy || !text.trim()) return;
  busy = true; sendBtn.disabled = true;
  add('user', text);
  const bubble = add('agent', '');
  const think = document.createElement('div'); think.className = 'thinking';
  let secs = 0; think.textContent = 'thinking... 0s';
  bubble.appendChild(think); chat.scrollTop = chat.scrollHeight;
  timer = setInterval(() => { think.textContent = 'thinking... ' + (++secs) + 's'; }, 1000);

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text})
    });
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
        if (ev.type === 'trace') addTrace(ev.text);
        else if (ev.type === 'answer') { bubble.firstChild.textContent = ev.text;
          if (!ev.text) bubble.textContent = '(empty answer)'; }
        else if (ev.type === 'error') bubble.firstChild.textContent =
          '\u26a0 ' + ev.text;
        chat.scrollTop = chat.scrollHeight;
      }
    }
  } catch (e) {
    bubble.firstChild.textContent = '\u26a0 request failed: ' + e.message;
  } finally {
    clearInterval(timer); think.remove();
    busy = false; sendBtn.disabled = false; input.focus();
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

    # ------------------------------------------------------------ GET ---

    def do_GET(self):
        if self.path == "/":
            data = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/api/status":
            self._json({"orchestrator": pi_agent.ORCHESTRATOR_MODEL,
                        "analyst": pi_agent.ANALYST_MODEL,
                        "ha": pi_agent.HA_URL})
        elif self.path == "/static/bg.jpg":
            import mimetypes
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "static", "bg.jpg")
            with open(path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self._json({"error": "not found"}, 404)

    # ----------------------------------------------------------- POST ---

    def do_POST(self):
        if self.path != "/api/chat":
            self._json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            msg = json.loads(self.rfile.read(length) or b"{}").get(
                "message", "").strip()
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
            with AGENT_LOCK:
                old_sink = pi_agent.EVENT_SINK
                pi_agent.EVENT_SINK = events.put
                try:
                    answer, _ = pi_agent.orchestrator(msg, history=HISTORY)
                    HISTORY.append({"role": "user", "content": msg})
                    HISTORY.append({"role": "assistant", "content": answer})
                    del HISTORY[:-10]
                    events.put(("answer", answer))
                except Exception as e:
                    events.put(("error", str(e)[:300]))
                finally:
                    pi_agent.EVENT_SINK = old_sink
                    events.put(None)

        threading.Thread(target=worker, daemon=True).start()

        while True:
            try:
                item = events.get(timeout=600)
            except queue.Empty:
                break
            if item is None:
                break
            if isinstance(item, tuple):       # ("answer"|"error", text)
                send({"type": item[0], "text": item[1]})
            else:                              # trace line
                send({"type": "trace", "text": str(item)})

    # ---------------------------------------------------------- helpers ---

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print("pi-agent web listening on http://0.0.0.0:%d" % PORT)
    print("(open http://<this pi's ip>:%d from any device on your network)"
          % PORT)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
