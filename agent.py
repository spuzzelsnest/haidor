#!/usr/bin/env python3
"""
pi-agent - a local AI orchestrator with specialist sub-agents, running on a Pi.

  Orchestrator (qwen3:4b)  ->  plans and delegates work to sub-agents:
     - home_assistant : Home Assistant REST API tools (list/get/call services)
     - analyst        : deepseek-r1:7b, reasoning-heavy analysis tasks
     - system         : strictly whitelisted read-only commands on this Pi

Config: environment variables, or a .env file next to this script:
    OLLAMA_URL=http://localhost:11434
    ORCHESTRATOR_MODEL=qwen3:4b
    ANALYST_MODEL=deepseek-r1:7b
    HA_URL=http://gatekeeper.local
    HA_TOKEN=<long-lived access token>
    HA_BLOCKED=shell_command,hassio,...   # optional, see below

Usage:
    python3 agent.py                       # interactive chat
    python3 agent.py --once "turn off the living room lights"
"""

import json
import os
import re
import shlex
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- config ---

def load_env():
    path = os.path.join(HERE, ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

load_env()

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
HA_URL = os.environ.get("HA_URL", "http://gatekeeper.local").rstrip("/")
HA_TOKEN = os.environ.get("HA_TOKEN", "")
ORCHESTRATOR_MODEL = os.environ.get("ORCHESTRATOR_MODEL", "qwen3:4b")
ANALYST_MODEL = os.environ.get("ANALYST_MODEL", "deepseek-r1:7b")
MAX_TURNS = int(os.environ.get("MAX_TURNS", "8"))
TOOL_RESULT_LIMIT = 4000   # chars of tool output fed back to a model

# Services the LLM may NOT call. Entries are a whole domain ("hassio") or an
# exact "domain.service". Set HA_BLOCKED="" to disable, or edit the list.
HA_BLOCKED = {
    s.strip() for s in os.environ.get(
        "HA_BLOCKED",
        "shell_command,python_script,hassio,alarm_control_panel,"
        "lock.unlock,lock.open,homeassistant.restart,homeassistant.stop",
    ).split(",") if s.strip()
}

if not HA_TOKEN:
    print("WARNING: HA_TOKEN is not set - Home Assistant calls will fail.",
          file=sys.stderr)

# Event hook for UIs (e.g. server.py): set agent.EVENT_SINK = my_callback
EVENT_SINK = None

def emit(text):
    """Print a trace event, and forward it to the UI sink if one is set."""
    print(text)
    if EVENT_SINK:
        try:
            EVENT_SINK(text)
        except Exception:
            pass

# ----------------------------------------------------------------- http ---

def http_json(url, method="GET", body=None, token=None, timeout=90):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        return {"error": "HTTP %d: %s" % (e.code, e.read().decode()[:300])}
    except Exception as e:
        return {"error": str(e)[:300]}

# Models that rejected the `think` option (e.g. some deepseek-r1 builds)
_NO_THINK = set()

def ollama_chat(model, messages, tools=None, retries=2):
    """One chat call. Returns the assistant message, raises on failure."""
    payload = {"model": model, "messages": messages, "stream": False,
               "keep_alive": "30m"}
    if model not in _NO_THINK:
        payload["think"] = False      # skip reasoning traces for speed
    if tools:
        payload["tools"] = tools

    last_err, attempt = None, 0
    while attempt < retries:
        out = http_json(OLLAMA_URL + "/api/chat", "POST", payload, timeout=600)
        err = out.get("error") if isinstance(out, dict) else "bad response"
        if not err:
            return out.get("message", {})
        err = str(err)
        if "think" in err.lower() and "think" in payload:
            payload.pop("think")      # retry without it; doesn't cost an attempt
            _NO_THINK.add(model)
            continue
        last_err = err
        attempt += 1
        if attempt < retries and "timed out" in err.lower():
            emit("  (ollama slow, retrying %d/%d...)" % (attempt, retries))
            continue
        break
    raise RuntimeError(last_err or "ollama chat failed")

# Strip <think>...</think> blocks (qwen3 / deepseek-r1), even if unterminated
THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.S)

def clean(text):
    return THINK_RE.sub("", text or "").strip()

# -------------------------------------------------- tool-calling agent loop -

def normalize_args(raw):
    """Tool arguments may arrive as a dict or a JSON string."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return {}
    return raw if isinstance(raw, dict) else {}

def tool_content(result, limit=TOOL_RESULT_LIMIT):
    text = json.dumps(result, default=str)
    return text if len(text) <= limit else text[:limit] + "... [truncated]"

def run_agent(model, system, tools, handlers, task, history=None,
              max_turns=None, label=None):
    """Run one agent: chat -> tool calls -> results -> ... -> final answer.

    If `label` is given, each tool call is emitted live as a trace event.
    """
    stamp = datetime.now().strftime("%A %Y-%m-%d %H:%M")
    messages = [{"role": "system",
                 "content": "%s\nCurrent date/time: %s" % (system, stamp)}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": task})
    trace = []
    for _ in range(max_turns or MAX_TURNS):
        msg = ollama_chat(model, messages, tools or None)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return clean(msg.get("content")), trace
        messages.append(msg)
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = normalize_args(fn.get("arguments"))
            if label:
                emit("     [%s] %s %s" % (label, name,
                                          json.dumps(args, default=str)[:60]))
            handler = handlers.get(name)
            if handler is None:
                result = {"error": "unknown tool: " + name}
            else:
                try:
                    result = handler(**args)
                except TypeError as e:
                    result = {"error": "bad arguments for %s: %s" % (name, e)}
                except Exception as e:
                    result = {"error": str(e)[:300]}
            trace.append((name, args, result))
            messages.append({"role": "tool", "tool_name": name,
                             "content": tool_content(result)})
    return "(reached the tool-call limit without a final answer)", trace

# --------------------------------------------------- Home Assistant tools ---

_ENTITY_RE = re.compile(r"^[a-z0-9_]+\.[a-z0-9_]+$")
_NAME_RE = re.compile(r"^[a-z0-9_]+$")

def ha_list_entities(domain=None, query=None):
    states = http_json(HA_URL + "/api/states", token=HA_TOKEN)
    if isinstance(states, dict):          # error response
        return states
    lines = []
    for s in states:
        eid = s.get("entity_id", "")
        if domain and not eid.startswith(domain + "."):
            continue
        name = (s.get("attributes") or {}).get("friendly_name", "")
        line = "%s = %s" % (eid, s.get("state"))
        if name:
            line += "  (%s)" % name
        if query and query.lower() not in line.lower():
            continue
        lines.append(line)
    return {"count": len(lines), "entities": lines[:200]}

def ha_get_state(entity_id):
    if not _ENTITY_RE.match(entity_id or ""):
        return {"error": "invalid entity_id: %r" % entity_id}
    return http_json(HA_URL + "/api/states/" + entity_id, token=HA_TOKEN)

def ha_call_service(domain, service, data=None):
    if not (_NAME_RE.match(domain or "") and _NAME_RE.match(service or "")):
        return {"error": "invalid domain/service"}
    if domain in HA_BLOCKED or "%s.%s" % (domain, service) in HA_BLOCKED:
        return {"error": "%s.%s is blocked by policy; tell the user to do it "
                         "manually" % (domain, service)}
    url = "%s/api/services/%s/%s" % (HA_URL, domain, service)
    return http_json(url, "POST", normalize_args(data), token=HA_TOKEN)

def ha_get_history(entity_id):
    if not _ENTITY_RE.match(entity_id or ""):
        return {"error": "invalid entity_id: %r" % entity_id}
    url = "%s/api/history/period?filter_entity_id=%s" % (
        HA_URL, urllib.parse.quote(entity_id))
    out = http_json(url, token=HA_TOKEN)
    if isinstance(out, list) and out:
        return [{"entity_id": s[0].get("entity_id") if s else entity_id,
                 "states": [(e.get("last_updated"), e.get("state"))
                            for e in s[:100]]}
                for s in out]
    return out

HA_TOOLS = [
    {"type": "function", "function": {
        "name": "list_entities",
        "description": "List Home Assistant entities, optionally filtered by "
                       "domain (light, switch, sensor, climate, media_player...) "
                       "and/or a free-text query.",
        "parameters": {"type": "object", "properties": {
            "domain": {"type": "string",
                       "description": "HA domain, e.g. 'light' or 'sensor'"},
            "query": {"type": "string",
                      "description": "substring to match in id or friendly name"},
        }}}},
    {"type": "function", "function": {
        "name": "get_state",
        "description": "Get the full current state and attributes of one entity.",
        "parameters": {"type": "object", "properties": {
            "entity_id": {"type": "string",
                          "description": "e.g. 'light.living_room'"}},
            "required": ["entity_id"]}}},
    {"type": "function", "function": {
        "name": "call_service",
        "description": "Call a Home Assistant service, e.g. "
                       "domain='light', service='turn_off', "
                       "data={'entity_id': 'light.living_room'}",
        "parameters": {"type": "object", "properties": {
            "domain": {"type": "string"},
            "service": {"type": "string"},
            "data": {"type": "object",
                     "description": "service payload, e.g. entity_id, brightness"},
        }, "required": ["domain", "service"]}}},
    {"type": "function", "function": {
        "name": "get_history",
        "description": "Get recent state history for one entity (last ~24h).",
        "parameters": {"type": "object", "properties": {
            "entity_id": {"type": "string"}},
            "required": ["entity_id"]}}},
]

HA_HANDLERS = {
    "list_entities": ha_list_entities,
    "get_state": ha_get_state,
    "call_service": ha_call_service,
    "get_history": ha_get_history,
}

HA_SYSTEM = (
    "You are the Home Assistant specialist agent. You control Jack's smart home "
    "through the provided tools. Always look up exact entity_ids with "
    "list_entities before acting on a device unless the id is already known. "
    "When asked to change something, call the right service and then report "
    "what you did. Be concise. Report errors honestly instead of pretending "
    "an action succeeded."
)

# -------------------------------------------------------- system sub-agent ---
#
# No shell is involved: the command is split with shlex and run directly, so
# `;`, `&&`, `|`, `$(...)` etc. are just inert arguments. Each allowed command
# also has an argument check so read-only tools can't be turned into writers
# (e.g. `systemctl restart`, `ollama rm`, `ip link set`, `cat .env`).

_READABLE_FILES = {
    "/proc/cpuinfo", "/proc/meminfo", "/proc/loadavg", "/proc/uptime",
    "/sys/class/thermal/thermal_zone0/temp", "/etc/os-release",
}
_SYSTEMCTL_SUBS = {"status", "is-active", "is-enabled", "is-failed",
                   "list-units", "list-timers", "show"}
_SYSTEMCTL_FLAGS = {"--no-pager", "-l", "--failed", "-a", "--all"}
_IP_OBJECTS = {"a", "addr", "address", "r", "route", "l", "link", "neigh"}
_IP_FLAGS = {"-4", "-6", "-br", "-brief", "-c", "-s", "-j", "-o", "-p"}

def _flags_only(args):
    return all(a.startswith("-") for a in args)

def _check_ss(args):
    return not any(a == "--kill" or (a.startswith("-") and not a.startswith("--")
                                     and "K" in a) for a in args)

def _check_ip(args):
    flags = [a for a in args if a.startswith("-")]
    words = [a for a in args if not a.startswith("-")]
    return (all(f in _IP_FLAGS for f in flags) and bool(words)
            and words[0] in _IP_OBJECTS
            and all(w == "show" for w in words[1:]))

def _check_systemctl(args):
    words = [a for a in args if not a.startswith("-")]
    flags = [a for a in args if a.startswith("-")]
    return bool(words) and words[0] in _SYSTEMCTL_SUBS \
        and all(f in _SYSTEMCTL_FLAGS for f in flags)

ALLOWED_CMDS = {
    "uptime": _flags_only,
    "free": _flags_only,
    "df": _flags_only,
    "uname": _flags_only,
    "lsblk": _flags_only,
    "ps": lambda a: True,
    "ss": _check_ss,
    "ip": _check_ip,
    "hostname": lambda a: all(x in {"-I", "-i", "-f", "-s", "-d"} for x in a),
    "timedatectl": lambda a: a in ([], ["status"], ["show"]),
    "systemctl": _check_systemctl,
    "ollama": lambda a: bool(a) and a[0] in {"list", "ps", "show"},
    "vcgencmd": lambda a: bool(a) and (a[0].startswith("measure_")
                                       or a[0] in {"get_throttled", "get_mem"}),
    "cat": lambda a: bool(a) and all(x in _READABLE_FILES for x in a),
}

def sys_shell(cmd):
    if not isinstance(cmd, str) or not cmd.strip():
        return {"error": "empty command"}
    try:
        argv = shlex.split(cmd)
    except ValueError as e:
        return {"error": "could not parse command: %s" % e}
    check = ALLOWED_CMDS.get(argv[0])
    if check is None:
        return {"error": "command '%s' is not on the allowed list (%s)"
                % (argv[0], ", ".join(sorted(ALLOWED_CMDS)))}
    if not check(argv[1:]):
        return {"error": "arguments not allowed for '%s'. Single read-only "
                         "commands only (no pipes or chaining)." % argv[0]}
    try:
        p = subprocess.run(argv, capture_output=True, timeout=30)
        out = (p.stdout + p.stderr).decode(errors="replace")[:3000]
        return {"exit_code": p.returncode, "output": out}
    except Exception as e:
        return {"error": str(e)[:200]}

SYSTEM_TOOLS = [
    {"type": "function", "function": {
        "name": "shell",
        "description": "Run ONE whitelisted read-only diagnostic command on this "
                       "Raspberry Pi, no pipes or chaining. Examples: uptime, "
                       "free -h, df -h, vcgencmd measure_temp, "
                       "vcgencmd get_throttled, systemctl status <unit>, "
                       "ss -tlnp, ps aux, ollama list, ollama ps.",
        "parameters": {"type": "object", "properties": {
            "cmd": {"type": "string"}},
            "required": ["cmd"]}}},
]

SYSTEM_HANDLERS = {"shell": sys_shell}

SYSTEM_SYSTEM = (
    "You are the system specialist for this Raspberry Pi. Use the shell tool "
    "with whitelisted read-only commands to answer questions about the "
    "machine (temperature, memory, disk, services, uptime, Ollama models). "
    "Never attempt to modify the system."
)

# ---------------------------------------------------------- analyst sub-agent -

ANALYST_SYSTEM = (
    "You are the analyst sub-agent. You receive data or questions that need "
    "careful reasoning (patterns, summaries, comparisons, decisions). Think "
    "step by step, then give a clear, concise answer."
)

# ---------------------------------------------------------------- registry ---

SUBAGENTS = {
    "home_assistant": {
        "model": ORCHESTRATOR_MODEL,
        "system": HA_SYSTEM,
        "tools": HA_TOOLS,
        "handlers": HA_HANDLERS,
        "blurb": "controls and queries the smart home (lights, sensors, "
                 "switches, scenes, any HA entity)",
    },
    "analyst": {
        "model": ANALYST_MODEL,
        "system": ANALYST_SYSTEM,
        "tools": [],
        "handlers": {},
        "blurb": "deep reasoning and analysis of data or questions",
    },
    "system": {
        "model": ORCHESTRATOR_MODEL,
        "system": SYSTEM_SYSTEM,
        "tools": SYSTEM_TOOLS,
        "handlers": SYSTEM_HANDLERS,
        "blurb": "diagnostics of this Raspberry Pi itself",
    },
}

# ------------------------------------------------------------ orchestrator ---

def delegate(agent, task):
    a = SUBAGENTS.get(agent)
    if not a:
        return {"error": "unknown agent '%s' (use: %s)"
                % (agent, ", ".join(SUBAGENTS))}
    emit("  -> delegating to %s: %s" % (agent, task[:70]))
    answer, _ = run_agent(a["model"], a["system"], a["tools"], a["handlers"],
                          task, label=agent)
    return {"agent": agent, "answer": answer[:TOOL_RESULT_LIMIT]}

DELEGATE_TOOL = [
    {"type": "function", "function": {
        "name": "delegate",
        "description": (
            "Delegate a task to a specialist sub-agent and get its result. "
            + " ".join("'%s': %s." % (k, v["blurb"]) for k, v in SUBAGENTS.items())
            + " Write the task as a complete, self-contained instruction."
        ),
        "parameters": {"type": "object", "properties": {
            "agent": {"type": "string",
                      "enum": list(SUBAGENTS),
                      "description": "which sub-agent to use"},
            "task": {"type": "string",
                     "description": "full, self-contained instruction for the agent"},
        }, "required": ["agent", "task"]}}},
]

ORCH_SYSTEM = (
    "You are the coordinator AI agent running on Jack's Raspberry Pi 5. "
    "You manage specialist sub-agents. You have NO tools of your own except "
    "'delegate'. For anything about the smart home, the Pi, or analysis, "
    "delegate to the right sub-agent with a complete, self-contained task. "
    "You may delegate multiple times to answer one request. Combine the "
    "sub-agents' answers into a short, helpful reply for Jack. "
    "Never invent device states or system facts - always delegate."
)

def orchestrator(task, history=None):
    return run_agent(ORCHESTRATOR_MODEL, ORCH_SYSTEM, DELEGATE_TOOL,
                     {"delegate": delegate}, task, history=history)

# -------------------------------------------------------------------- cli ---

def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "--once":
        task = " ".join(argv[1:])
        if not task:
            print("usage: agent.py --once <task>", file=sys.stderr)
            sys.exit(1)
        answer, _ = orchestrator(task)
        print(answer)
        return

    print("pi-agent ready. models: orchestrator=%s analyst=%s"
          % (ORCHESTRATOR_MODEL, ANALYST_MODEL))
    print("Home Assistant: %s  (type your request, Ctrl-D to quit)\n" % HA_URL)
    history = []
    while True:
        try:
            task = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not task:
            continue
        if task.lower() in {"exit", "quit"}:
            break
        try:
            answer, _ = orchestrator(task, history=history)
        except Exception as e:
            print("error: %s" % e)
            continue
        history += [{"role": "user", "content": task},
                    {"role": "assistant", "content": answer}]
        history = history[-10:]          # keep the context small
        print("\nagent> %s\n" % answer)

if __name__ == "__main__":
    main()
