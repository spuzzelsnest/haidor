#!/usr/bin/env python3
"""
pi-agent — a local AI orchestrator with specialist sub-agents, running on a Pi.

  Orchestrator (qwen3:4b)  ->  plans and delegates work to sub-agents:
     - home_assistant : full Home Assistant REST API tools (list/get/call services)
     - analyst        : deepseek-r1:7b, reasoning-heavy analysis tasks
     - system         : whitelisted shell commands on this Pi (temps, disk, uptime...)

Config: environment variables, or a .env file next to this script:
    OLLAMA_URL=http://localhost:11434
    ORCHESTRATOR_MODEL=qwen3:4b
    ANALYST_MODEL=deepseek-r1:7b
    HA_URL=http://gatekeeper.local
    HA_TOKEN=<long-lived access token>

Usage:
    python3 agent.py                       # interactive chat
    python3 agent.py --once "turn off the living room lights"
"""

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- config ---

def load_env():
    path = os.path.join(HERE, ".env")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v.strip().strip('"').strip("'"))

load_env()

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
HA_URL = os.environ.get("HA_URL", "http://gatekeeper.local").rstrip("/")
HA_TOKEN = os.environ.get("HA_TOKEN", "")
ORCHESTRATOR_MODEL = os.environ.get("ORCHESTRATOR_MODEL", "qwen3:4b")
ANALYST_MODEL = os.environ.get("ANALYST_MODEL", "deepseek-r1:7b")
MAX_TURNS = int(os.environ.get("MAX_TURNS", "8"))

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
    req = urllib.request.Request(url, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(body).encode() if body is not None else None
    try:
        with urllib.request.urlopen(req, data=data, timeout=timeout) as r:
            raw = r.read().decode()
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        return {"error": "HTTP %d: %s" % (e.code, e.read().decode()[:300])}
    except Exception as e:
        return {"error": str(e)[:300]}

def ollama_chat(model, messages, tools=None, think=False, retries=2):
    payload = {"model": model, "messages": messages, "stream": False,
               "think": think, "keep_alive": "30m"}
    if tools:
        payload["tools"] = tools
    last_err = None
    for attempt in range(retries):
        out = http_json(OLLAMA_URL + "/api/chat", "POST", payload, timeout=600)
        if isinstance(out, dict) and "error" in out:
            err = str(out["error"])
            # some models (e.g. deepseek-r1) reject think=false -> retry without it
            if "think" in err.lower() and "think" in payload:
                payload.pop("think", None)
                continue
            last_err = err
            if "timed out" in err.lower():
                emit("  (ollama slow, retrying %d/%d...)" % (attempt + 1, retries))
                continue
        return out.get("message", {}) if isinstance(out, dict) else {}
    raise RuntimeError(last_err or "ollama chat failed")

# Strip <think>...</think> blocks (qwen3 / deepseek-r1 reasoning traces)
THINK_RE = re.compile(r"<think>.*?</think>", re.S)

def clean(text):
    return THINK_RE.sub("", text or "").strip()

# -------------------------------------------------- tool-calling agent loop -

def normalize_args(raw):
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return raw or {}

def run_agent(model, system, tools, handlers, task, history=None, max_turns=None):
    """Run one agent: chat -> tool calls -> results -> ... -> final answer."""
    messages = [{"role": "system", "content": system}]
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
            if name in handlers:
                try:
                    result = handlers[name](**args)
                except TypeError as e:
                    result = {"error": "bad arguments for %s: %s" % (name, e)}
                except Exception as e:
                    result = {"error": str(e)[:300]}
            else:
                result = {"error": "unknown tool: " + name}
            trace.append((name, args, result))
            messages.append({
                "role": "tool",
                "content": json.dumps(result, default=str)[:4000],
            })
    return "(reached the tool-call limit without a final answer)", trace

# --------------------------------------------------- Home Assistant tools ---

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
    return http_json(HA_URL + "/api/states/" + entity_id, token=HA_TOKEN)

def ha_call_service(domain, service, data=None):
    url = "%s/api/services/%s/%s" % (HA_URL, domain, service)
    return http_json(url, "POST", data or {}, token=HA_TOKEN)

def ha_get_history(entity_id):
    url = "%s/api/history/period?filter_entity_id=%s" % (HA_URL, entity_id)
    out = http_json(url, token=HA_TOKEN)
    if isinstance(out, list) and out:
        return [{"entity_id": s.get("entity_id"),
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

ALLOWED_CMDS = {
    "uptime", "free", "df", "vcgencmd", "uname", "hostname", "ip", "ss",
    "ps", "lsblk", "systemctl", "timedatectl", "cat", "ollama",
}

def sys_shell(cmd):
    if not isinstance(cmd, str) or not cmd.strip():
        return {"error": "empty command"}
    first = cmd.strip().split()[0]
    if first not in ALLOWED_CMDS:
        return {"error": "command '%s' is not on the allowed list" % first}
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, timeout=30)
        out = (p.stdout + p.stderr).decode(errors="replace")[:3000]
        return {"exit_code": p.returncode, "output": out}
    except Exception as e:
        return {"error": str(e)[:200]}

SYSTEM_TOOLS = [
    {"type": "function", "function": {
        "name": "shell",
        "description": "Run a whitelisted diagnostic command on this Raspberry Pi "
                       "(uptime, free -h, df -h, vcgencmd measure_temp, "
                       "systemctl status <unit>, ss -tlnp, ps aux, ollama list...). "
                       "Read-only diagnostics only.",
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

ANALYST_TOOLS = []
ANALYST_HANDLERS = {}

# ---------------------------------------------------------------- registry ---

SUBAGENTS = {
    "home_assistant": {
        "model": ORCHESTRATOR_MODEL,
        "system": HA_SYSTEM,
        "tools": HA_TOOLS,
        "handlers": HA_HANDLERS,
    },
    "analyst": {
        "model": ANALYST_MODEL,
        "system": ANALYST_SYSTEM,
        "tools": ANALYST_TOOLS,
        "handlers": ANALYST_HANDLERS,
    },
    "system": {
        "model": ORCHESTRATOR_MODEL,
        "system": SYSTEM_SYSTEM,
        "tools": SYSTEM_TOOLS,
        "handlers": SYSTEM_HANDLERS,
    },
}

AGENT_BLURB = {
    "home_assistant": "controls and queries the smart home (lights, sensors, "
                      "switches, scenes, any HA entity)",
    "analyst": "deep reasoning and analysis of data or questions",
    "system": "diagnostics of this Raspberry Pi itself",
}

# ------------------------------------------------------------ orchestrator ---

def delegate(agent, task):
    a = SUBAGENTS.get(agent)
    if not a:
        return {"error": "unknown agent '%s' (use: %s)"
                % (agent, ", ".join(SUBAGENTS))}
    emit("  -> delegating to %s: %s" % (agent, task[:70]))
    answer, trace = run_agent(a["model"], a["system"], a["tools"],
                              a["handlers"], task)
    for name, args, _ in trace:
        emit("     [%s] %s %s" % (agent, name,
                                 json.dumps(args, default=str)[:60]))
    return {"agent": agent, "answer": answer[:4000]}

DELEGATE_TOOL = [
    {"type": "function", "function": {
        "name": "delegate",
        "description": (
            "Delegate a task to a specialist sub-agent and get its result. "
            + " ".join("'%s': %s." % (k, v) for k, v in AGENT_BLURB.items())
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

ORCH_HANDLERS = {"delegate": delegate}

def orchestrator(task, history=None):
    return run_agent(ORCHESTRATOR_MODEL, ORCH_SYSTEM, DELEGATE_TOOL,
                     ORCH_HANDLERS, task, history=history)

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
        history.append({"role": "user", "content": task})
        history.append({"role": "assistant", "content": answer})
        history = history[-10:]          # keep the context small
        print("\nagent> %s\n" % answer)

if __name__ == "__main__":
    main()
