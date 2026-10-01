#!/usr/bin/env python3
# Sandbox (--sandbox-profile) smoke test.
#
# Runs mlx-agent under profiles and checks, from OBSERVED effects, that the process is confined
# and still works:
#   1. --sandbox-print shows a deny-by-default profile and runs nothing.
#   2. A profile with an unknown key ends the process before anything else runs.
#   3. With one local port allowed, the openai backend reaches a model server on that port.
#   4. Without the port, it cannot.
#   5. An MCP tool server started by the confined agent inherits the confinement: it reads a file
#      in a granted folder, cannot read a file outside one, cannot write outside a writable
#      folder, and cannot connect out. (The model server is a fake that scripts the tool call.)
#   6. With --foundation: Apple's on-device model answers under "foundation_models".
#   7. With --model <dir>: a real MLX model answers under "gpu", with no network.
#
# Usage: python3 tools/sandbox_smoke.py /path/to/mlx-agent [--foundation] [--model <mlx model dir>]
#
# 1-5 need no model. The tool server is this Python; the profile grants what it needs to start.

import json
import os
import subprocess
import sys
import sysconfig
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail if not cond else ''}")
    if cond:
        PASS += 1
    else:
        FAIL += 1


# The tool server: one tool, `probe`, that tries four things and writes what happened.
PROBE_SERVER = r'''
import json, os, socket, sys
allowed_file, outside_file, report, outside_write = sys.argv[1:5]

def attempt(action):
    try:
        action()
        return "ok"
    except OSError as error:
        return "refused: " + (error.strerror or str(error))

def connect():
    with socket.create_connection(("1.1.1.1", 53), timeout=3):
        pass

def write_outside():
    with open(outside_write, "w") as handle:
        handle.write("x")

def probe():
    result = {
        "read_allowed": attempt(lambda: open(allowed_file).read()),
        "read_outside": attempt(lambda: open(outside_file).read()),
        "write_outside": attempt(write_outside),
        "connect_out": attempt(connect),
    }
    with open(report, "w") as handle:
        json.dump(result, handle)
    return result

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
        send({"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
            "serverInfo": {"name": "probe", "version": "1.0"}}})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": rid, "result": {"tools": [{
            "name": "probe", "description": "tries what the sandbox should refuse",
            "inputSchema": {"type": "object", "properties": {}}}]}})
    elif method == "tools/call":
        send({"jsonrpc": "2.0", "id": rid, "result": {"content": [
            {"type": "text", "text": json.dumps(probe())}]}})
    elif rid is not None:
        send({"jsonrpc": "2.0", "id": rid, "result": {}})
'''


def fake_model(script):
    """An OpenAI-compatible server that answers each completion with the next scripted stream."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = b'{"status":"ok"}' if self.path == "/health" else b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            chunks = script.pop(0) if script else [
                {"choices": [{"delta": {"content": "ready"}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode())
            self.wfile.flush()

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def run(argv, timeout=180):
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def main():
    args = sys.argv[1:]
    if not args:
        print("usage: sandbox_smoke.py /path/to/mlx-agent [--foundation] [--model <mlx model dir>]")
        return 2
    agent = args[0]
    foundation = "--foundation" in args
    model_dir = args[args.index("--model") + 1] if "--model" in args else None

    # Work in the home folder, not the temporary one: the per-user temporary folder sits beside
    # the Metal caches, and the point is a folder no rule of the profile reaches.
    work = tempfile.mkdtemp(prefix=".mlx-agent-sandbox-smoke-", dir=os.path.expanduser("~"))
    granted = os.path.join(work, "granted")
    writable = os.path.join(work, "writable")
    outside = os.path.join(work, "outside")
    for folder in (granted, writable, outside):
        os.mkdir(folder)
    allowed_file = os.path.join(granted, "note.txt")
    outside_file = os.path.join(outside, "secret.txt")
    for path in (allowed_file, outside_file):
        with open(path, "w") as handle:
            handle.write("content\n")
    report = os.path.join(writable, "report.json")
    server_py = os.path.join(granted, "probe_server.py")
    with open(server_py, "w") as handle:
        handle.write(PROBE_SERVER)

    def profile(name, config):
        path = os.path.join(work, name)
        with open(path, "w") as handle:
            json.dump(config, handle)
        return path

    try:
        # 1. Print only.
        code, out, err = run([agent, "oneshot", "--sandbox-profile", profile("empty.json", {}),
                              "--sandbox-print", "--backend", "openai", "--base-url", "http://127.0.0.1:1/v1",
                              "--prompt", "x"])
        check("--sandbox-print shows a deny-by-default profile and exits 0",
              code == 0 and "(deny default)" in out and "(deny network*)" in out, f"{code} {err[-200:]}")

        # 2. A misspelled key.
        code, out, err = run([agent, "oneshot", "--sandbox-profile", profile("typo.json", {"allow_netwrok": True}),
                              "--backend", "openai", "--base-url", "http://127.0.0.1:1/v1", "--prompt", "x"])
        check("a profile with an unknown key ends the process", code == 2 and "unknown key" in err, f"{code} {err[-200:]}")

        # 3. One local port.
        model = fake_model([])
        port = model.server_address[1]
        base = f"http://127.0.0.1:{port}/v1"
        code, out, err = run([agent, "oneshot", "--sandbox-profile",
                              profile("port.json", {"network_connect": [f"localhost:{port}"]}),
                              "--backend", "openai", "--base-url", base, "--prompt", "say ready"])
        check("with its port allowed, the openai backend answers", code == 0 and "ready" in out, f"{code} {err[-300:]}")

        # 4. No port.
        code, out, err = run([agent, "oneshot", "--sandbox-profile", profile("noport.json", {}),
                              "--backend", "openai", "--base-url", base, "--prompt", "say ready"], timeout=120)
        check("without the port it cannot reach the server", "ready" not in out, f"{code} {out[-200:]}")

        # 5. A tool server under the inherited profile.
        script = [
            [{"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_probe", "type": "function",
                                                     "function": {"name": "probe", "arguments": "{}"}}]},
                           "finish_reason": None}]},
             {"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}],
        ]
        model = fake_model(script)
        port = model.server_address[1]
        python = os.path.realpath(sys.executable)
        python_home = [os.path.realpath(p) for p in {sys.base_prefix, sysconfig.get_paths()["stdlib"]}]
        config = profile("mcp.json", {"servers": [{
            "name": "probe", "command": python,
            "args": [server_py, allowed_file, outside_file, report, os.path.join(outside, "written.txt")]}]})
        code, out, err = run([agent, "oneshot", "--sandbox-profile", profile("tools.json", {
            "network_connect": [f"localhost:{port}"],
            "read_only": [granted] + python_home,
            "read_write": [writable],
            "read_only_files": [config],
            # Any program may be started: a framework Python starts a second executable of its
            # own, and this section is about what a child can read, write and reach.
            "allow_exec": True,
        }), "--backend", "openai", "--base-url", f"http://127.0.0.1:{port}/v1", "--mcp-config", config,
            "--auto-permission", "allow", "--prompt", "probe"])
        try:
            result = json.load(open(report))
        except (OSError, ValueError):
            result = {}
        check("the tool server started and ran its tool", bool(result), f"{code} {err[-400:]}")
        check("  it reads a file in a granted folder", result.get("read_allowed") == "ok", str(result))
        check("  it cannot read a file outside one", str(result.get("read_outside", "")).startswith("refused"), str(result))
        check("  it cannot write outside a writable folder",
              str(result.get("write_outside", "")).startswith("refused")
              and not os.path.exists(os.path.join(outside, "written.txt")), str(result))
        check("  it cannot connect out", str(result.get("connect_out", "")).startswith("refused"), str(result))

        # 6. Apple's on-device model.
        if foundation:
            code, out, err = run([agent, "oneshot", "--sandbox-profile",
                                  profile("fm.json", {"foundation_models": True}),
                                  "--backend", "foundation", "--prompt", "Reply with the single word: ready"])
            check("Apple's on-device model answers under foundation_models",
                  code == 0 and "ready" in out.lower(), f"{code} {err[-300:]}")
            code, out, err = run([agent, "oneshot", "--sandbox-profile", profile("fm-none.json", {}),
                                  "--backend", "foundation", "--prompt", "Reply with the single word: ready"])
            check("  and not without it", code != 0 or "ready" not in out.lower(), f"{code} {out[-100:]}")

        # 7. A real MLX model on the GPU.
        if model_dir:
            repo = os.path.realpath(model_dir)
            # A Hugging Face snapshot links into the repository's blobs folder, two levels up.
            parts = repo.split(os.sep)
            if "snapshots" in parts:
                repo = os.sep.join(parts[:parts.index("snapshots")])
            code, out, err = run([agent, "oneshot", "--sandbox-profile",
                                  profile("mlx.json", {"gpu": True, "read_only": [repo]}),
                                  "--model", model_dir, "--prompt", "Reply with the single word: ready"], timeout=600)
            check("an MLX model answers under gpu, with no network", code == 0 and "ready" in out.lower(),
                  f"{code} {err[-300:]}")
            code, out, err = run([agent, "oneshot", "--sandbox-profile", profile("mlx-nogpu.json", {"read_only": [repo]}),
                                  "--model", model_dir, "--prompt", "Reply with the single word: ready"], timeout=600)
            check("  and not without the gpu rules", code != 0, f"{code} {out[-100:]}")
    finally:
        subprocess.run(["/bin/rm", "-rf", work])
    print(f"\n{PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
