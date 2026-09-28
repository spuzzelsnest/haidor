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
