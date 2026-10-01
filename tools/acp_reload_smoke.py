#!/usr/bin/env python3
# MCP server reload (SIGHUP) smoke test.
#
# Starts mlx-agent in ACP mode with two fake MCP servers, rewrites the --mcp-config file and sends
# SIGHUP, and checks what happened to the server processes and to the tools.
#
# Usage: python3 tools/acp_reload_smoke.py /path/to/mlx-agent
#
# It uses `--backend foundation` only to get a session without loading a model (session/new is
# what starts the MCP servers); no prompt is sent, so nothing is generated. On a Mac without
# Apple Intelligence, pass another backend after the binary:
#     python3 tools/acp_reload_smoke.py /path/to/mlx-agent --model /path/to/mlx-model
#
# Every check is on an OBSERVED effect: a server's pid before and after, the argument the running
# process was started with, and the agent's own log line.
#   1. A server whose arguments changed is replaced by a new process; the other keeps its pid.
#   2. A config that only changes gatedTools restarts nothing.
#   3. A config whose changed server cannot start leaves the old one running.
#   4. A file that is not JSON leaves every server running.
#   5. A server removed from the file is stopped; one added is started, and the tools are
#      reported as changed.
#   6. A second server of a name the file lists twice is stopped by a reload, not left running
#      outside the table.
#   7. A signal that arrives during a turn waits for the turn's end. The turn is real but nothing
#      generates: a fake OpenAI-compatible server (as in acp_permission_smoke.py) scripts one
#      call of a tool that sleeps, and the signal is sent while it sleeps.
#   8. An agent that leaves during a reload takes the servers the reload already started with it,
#      even ones that do not end when their stdin closes.

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acp_smoke import reader  # noqa: E402
import queue  # noqa: E402

# A fake MCP server: one tool named after its first argument, and a pid file so the test can tell
# which process serves. argv: <tool name> <pid file> <tag> [<behavior> <seconds>]
# The behaviors: "slow-call" sleeps in tools/call (after touching <pid file>.call), "slow-start"
# sleeps before answering initialize, "linger" stays alive after its stdin closed.
FAKE_SERVER = r'''
import json, os, sys, time
tool, pid_file, tag = sys.argv[1], sys.argv[2], sys.argv[3]
behavior = sys.argv[4] if len(sys.argv) > 5 else ""
seconds = float(sys.argv[5]) if len(sys.argv) > 5 else 0
with open(pid_file, "w") as handle:
    handle.write("%d %s\n" % (os.getpid(), tag))

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    method, rid = req.get("method"), req.get("id")
    if method == "initialize":
        if behavior == "slow-start":
            time.sleep(seconds)
        send({"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
            "serverInfo": {"name": tool, "version": "1.0"}}})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": rid, "result": {"tools": [{
            "name": tool, "description": "says " + tool,
            "inputSchema": {"type": "object", "properties": {}}}]}})
    elif method == "tools/call":
        if behavior == "slow-call":
            open(pid_file + ".call", "w").close()
            time.sleep(seconds)
        send({"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": tag}]}})
    elif rid is not None:
        send({"jsonrpc": "2.0", "id": rid, "result": {}})
if behavior == "linger":
    time.sleep(seconds)
'''

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail if not cond else ''}")
    if cond:
        PASS += 1
    else:
        FAIL += 1


class Agent:
    def __init__(self, argv):
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.q = queue.Queue()
        self.err_lines = []
        self._lock = threading.Lock()
        threading.Thread(target=reader, args=(self.p.stdout, self.q), daemon=True).start()
        threading.Thread(target=self._drain, daemon=True).start()
        self._id = 0

    def _drain(self):
        for raw in self.p.stderr:
            with self._lock:
                self.err_lines.append(raw.decode(errors="replace").rstrip("\n"))

    def log_count(self):
        with self._lock:
            return len(self.err_lines)

    def wait_log(self, needle, since, timeout=30):
        """The first log line after index `since` that contains `needle`, or None."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                for line in self.err_lines[since:]:
                    if needle in line:
                        return line
            time.sleep(0.1)
        return None

    def send(self, method, params):
        self._id += 1
        self.p.stdin.write((json.dumps({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}) + "\n").encode())
        self.p.stdin.flush()
        return self._id

    def call(self, method, params, timeout=120):
        return self.wait(self.send(method, params), timeout)

    def wait(self, want, timeout=120):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                msg = self.q.get(timeout=max(0.1, deadline - time.time()))
            except queue.Empty:
                break
            if msg.get("id") == want and ("result" in msg or "error" in msg):
                return msg
        return None


def served(pid_file):
    """(pid, tag) of the process that last started for this server, or (None, None)."""
    try:
        pid, tag = open(pid_file).read().split()
        return int(pid), tag
    except (OSError, ValueError):
        return None, None


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def gone(pid, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.1)
    return False


def main():
    if len(sys.argv) < 2:
        print(__doc__ or "usage: acp_reload_smoke.py /path/to/mlx-agent [backend options]")
        return 2
    agent_bin = sys.argv[1]
    backend = sys.argv[2:] or ["--backend", "foundation"]
    work = tempfile.mkdtemp(prefix="mlx-agent-reload-")
    server_py = os.path.join(work, "fake_server.py")
    with open(server_py, "w") as handle:
        handle.write(FAKE_SERVER)
    config_path = os.path.join(work, "mcp-config.json")
    pid_a, pid_b, pid_c = (os.path.join(work, name) for name in ("a.pid", "b.pid", "c.pid"))

    def server(name, pid_file, tag, gated=(), command=sys.executable, behavior=()):
        return {"name": name, "command": command, "args": [server_py, name, pid_file, tag, *behavior],
                "gatedTools": list(gated)}

    def write(servers):
        # Written whole and moved into place, as a host should: the agent may read at any moment.
        tmp = config_path + ".tmp"
        with open(tmp, "w") as handle:
            json.dump({"servers": servers}, handle)
        os.replace(tmp, config_path)

    write([server("alpha", pid_a, "one"), server("beta", pid_b, "one")])
    agent = Agent([agent_bin, "acp", *backend, "--mcp-config", config_path])
    try:
        agent.call("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
        new = agent.call("session/new", {"cwd": work, "mcpServers": []})
        if not new or "result" not in new:
            print("session/new failed, so no MCP server started:", new)
            return 2
        a1, _ = served(pid_a)
        b1, _ = served(pid_b)
        check("both servers started", a1 is not None and b1 is not None and alive(a1) and alive(b1))

        def reload(servers=None, raw=None):
            since = agent.log_count()
            if raw is not None:
                with open(config_path, "w") as handle:
                    handle.write(raw)
            else:
                write(servers)
            os.kill(agent.p.pid, signal.SIGHUP)
            return agent.wait_log("reload", since)

        # 1. alpha's arguments change; beta's do not.
        line = reload([server("alpha", pid_a, "two"), server("beta", pid_b, "one")])
        a2, tag = served(pid_a)
        check("the agent says it restarted alpha, tools unchanged",
              line is not None and "restarted alpha" in line and "the tools are unchanged" in line, str(line))
        check("alpha is a new process with the new argument", a2 != a1 and tag == "two" and alive(a2), f"{a1} -> {a2} {tag}")
        check("the old alpha is gone", gone(a1))
        check("beta kept its process", served(pid_b)[0] == b1 and alive(b1))

        # 2. Only gatedTools changes.
        line = reload([server("alpha", pid_a, "two", gated=["alpha"]), server("beta", pid_b, "one")])
        check("a change of gated tools restarts nothing",
              line is not None and "nothing to restart" in line and served(pid_a)[0] == a2 and alive(a2), str(line))

        # 3. alpha's new command cannot start.
        line = reload([server("alpha", pid_a, "three", command="/nonexistent/server"), server("beta", pid_b, "one")])
        check("a server that cannot start is reported",
              line is not None and "failed to start alpha" in line, str(line))
        check("  and the running alpha stays", served(pid_a)[0] == a2 and alive(a2))

        # 4. The file is not JSON.
        line = reload(raw="{ not json")
        check("an unreadable config leaves the servers as they are",
              line is not None and "servers left as they are" in line and alive(a2) and alive(b1), str(line))

        # 5. beta is removed, gamma is added.
        line = reload([server("alpha", pid_a, "two"), server("gamma", pid_c, "one")])
        c1, _ = served(pid_c)
        check("beta is removed and gamma added, and the tools changed",
              line is not None and "added gamma" in line and "removed beta" in line and "the tools changed" in line, str(line))
        check("beta's process is gone", gone(b1))
        check("gamma runs, alpha kept its process", c1 is not None and alive(c1) and served(pid_a)[0] == a2 and alive(a2))

        # Leaving: closing stdin ends the agent, which must take its servers with it.
        agent.p.stdin.close()
        agent.p.wait(timeout=20)
        check("the servers end with the agent", gone(a2) and gone(c1))
    finally:
        if agent.p.poll() is None:
            agent.p.kill()

    def sighup(agent):
        since = agent.log_count()
        os.kill(agent.p.pid, signal.SIGHUP)
        return since

    # 6. The file lists alpha twice. Only the first starts, and a reload leaves it alone.
    pid_a2 = os.path.join(work, "a2.pid")
    write([server("alpha", pid_a, "one"), server("alpha", pid_a2, "one")])
    agent = Agent([agent_bin, "acp", *backend, "--mcp-config", config_path])
    try:
        agent.call("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
        agent.call("session/new", {"cwd": work, "mcpServers": []})
        first, second = served(pid_a)[0], served(pid_a2)[0]
        skipped = agent.wait_log("skipping a second server named alpha", 0, timeout=5)
        check("a second server of the same name is not started, and the log says so",
              first is not None and second is None and skipped is not None, f"{first} {second} {skipped}")
        line = agent.wait_log("reload", sighup(agent))
        check("  and a reload keeps the first",
              line is not None and "nothing to restart" in line and alive(first), f"{first} {line}")
    finally:
        agent.p.kill()

    # 7. A signal during a turn. The fake model calls the tool `slow`, which sleeps 3 s.
    script = [
        [{"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_slow", "type": "function",
                                                 "function": {"name": "slow", "arguments": "{}"}}]},
                       "finish_reason": None}]},
         {"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
          "usage": {"prompt_tokens": 10, "completion_tokens": 5}}],
        [{"choices": [{"delta": {"content": "Done."}, "finish_reason": None}]},
         {"choices": [{"delta": {}, "finish_reason": "stop"}],
          "usage": {"prompt_tokens": 10, "completion_tokens": 3}}],
    ]

    class FakeModel(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            chunks = script.pop(0) if script else []
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode())
            self.wfile.flush()

    model = HTTPServer(("127.0.0.1", 0), FakeModel)
    threading.Thread(target=model.serve_forever, daemon=True).start()
    pid_s = os.path.join(work, "s.pid")
    write([server("slow", pid_s, "one", behavior=("slow-call", "3"))])
    agent = Agent([agent_bin, "acp", "--backend", "openai", "--base-url",
                   f"http://127.0.0.1:{model.server_address[1]}/v1", "--mcp-config", config_path])
    try:
        agent.call("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
        new = agent.call("session/new", {"cwd": work, "mcpServers": []})
        sid = ((new or {}).get("result") or {}).get("sessionId")
        s1 = served(pid_s)[0]
        prompt = agent.send("session/prompt", {"sessionId": sid, "prompt": [{"type": "text", "text": "go"}]})
        deadline = time.time() + 30
        while time.time() < deadline and not os.path.exists(pid_s + ".call"):
            time.sleep(0.05)
        in_call = os.path.exists(pid_s + ".call")
        write([server("slow", pid_s, "two", behavior=("slow-call", "3"))])
        since = sighup(agent)
        early = agent.wait_log("reload", since, timeout=1.5)
        check("a signal during a turn starts no reload while the tool call runs",
              sid is not None and in_call and early is None and served(pid_s)[0] == s1 and alive(s1),
              f"in_call={in_call} {early}")
        done = agent.wait(prompt, timeout=60)
        check("  the turn ends normally", done is not None and (done.get("result") or {}).get("stopReason") == "end_turn", str(done))
        line = agent.wait_log("reload", since)
        with agent._lock:
            after = agent.err_lines[since:]
        resolved = next((i for i, text in enumerate(after) if "turn resolved" in text), None)
        reloaded = next((i for i, text in enumerate(after) if "reload" in text), None)
        check("  and the reload runs after it",
              line is not None and "restarted slow" in line and resolved is not None and reloaded is not None
              and resolved < reloaded and gone(s1) and served(pid_s)[1] == "two", f"{line} {resolved} {reloaded}")
    finally:
        agent.p.kill()
        model.shutdown()

    # 8. alpha restarts at once; beta's new process is slow to answer, so the reload is still in
    # flight when stdin closes. The new alpha does not end on its own for 30 s.
    write([server("alpha", pid_a, "one"), server("beta", pid_b, "one")])
    agent = Agent([agent_bin, "acp", *backend, "--mcp-config", config_path])
    try:
        agent.call("initialize", {"protocolVersion": 1, "clientCapabilities": {}})
        agent.call("session/new", {"cwd": work, "mcpServers": []})
        a1, b1 = served(pid_a)[0], served(pid_b)[0]
        write([server("alpha", pid_a, "two", behavior=("linger", "30")),
               server("beta", pid_b, "two", behavior=("slow-start", "5"))])
        sighup(agent)
        deadline = time.time() + 20
        while time.time() < deadline and served(pid_b)[0] == b1:
            time.sleep(0.05)
        a2, b2 = served(pid_a)[0], served(pid_b)[0]
        agent.p.stdin.close()
        agent.p.wait(timeout=20)
        check("an agent leaving during a reload stops the server the reload had started",
              a2 not in (None, a1) and b2 != b1 and gone(a2, timeout=5), f"{a1} -> {a2}")
        # The server still in its handshake is not the agent's to stop yet; do not leave it behind.
        for pid in (a2, b2):
            if pid is not None and alive(pid):
                os.kill(pid, signal.SIGKILL)
    finally:
        if agent.p.poll() is None:
            agent.p.kill()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
