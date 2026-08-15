#!/usr/bin/env python3
# Permission-flow tests for `mlx-agent acp` - the session-scoped "always" decisions.
#
# Stands up a FAKE OpenAI-compatible server (no model, no llama-server, no GPU) that replies
# with scripted SSE tool calls, and drives a REAL MCP server (mcp_server_time) whose tools are
# force-gated via the config's gatedTools. That makes the properties deterministic, which a live
# model cannot be: the model must emit exactly the tool call each check needs, in order.
#
# The checks are about what does and does not reach the WIRE, because "always" is defined by the
# absence of a round-trip - asserting on agent-internal state would pass even if the client were
# still being interrupted:
#
#   1. session/request_permission advertises ACP's standard four options, with correct kinds.
#   2. allow_always suppresses every LATER request for that tool, and the tool still runs.
#   3. The grant is per TOOL: always-allowing one gated tool leaves the other still asking.
#      (write_file being bounded by the sandbox says nothing about a shell tool in the same
#      sandbox - so one grant must never imply the other.)
#   4. reject_always denies later calls with no round-trip, and the denial still reaches the
#      model as a tool result so it can adapt rather than hang.
#   5. session/prime AND session/new clear the grants: a new conversation is a new consent
#      context. prime is the one that matters - it is how the shipping client actually starts a
#      conversation (ChatView calls session/new once per spawn, in start(); New Chat and every
#      sidebar switch go through session/prime, and `prime []` is the documented reset).
#   6. An optionId we never advertised denies. Fail-closed, not fail-open.
#   7. A permission answer that arrives AFTER the session was replaced does not write a grant
#      into the new session.
#   8. Not a permission property, but this is the only harness that scripts a model turn against
#      real MCP tools: an identical repeat call in one turn is short-circuited, not run twice.
#
# EVERY check asserts the tool's OUTCOME, not just the number of permission requests. Counting
# requests alone cannot tell a standing ALLOW from a standing DENY: both are "no round-trip",
# and the scripted SSE is identical either way. An earlier version of this file counted only,
# and a mutant that turned Always Allow into a silent deny passed all 12 checks.
#
# Usage: python3 tools/acp_permission_smoke.py /path/to/mlx-agent
#        (run from the repo root - it imports tools/acp_smoke.py by relative path)
#        (needs an interpreter that can run `-m mcp_server_time` - see INTERPRETER below)
# Exit code = number of failed checks (0 = all good). "Could not run at all" also exits 2, so
# read stderr to tell that apart from two failures; either way it is not a pass.

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, "tools")
from acp_smoke import Agent

PORT = 8399  # deliberately not 8099: the applet's llama-server owns that one
script = []  # queued SSE responses, one per /chat/completions call


def sse(chunks):
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def tool_call_response(name, args):
    """An assistant turn emitting exactly one tool call, arguments as a JSON string."""
    return sse([
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": f"call_{name}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}]},
            "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    ])


def dup_tool_call_response(name, args):
    """One assistant turn emitting the SAME call twice - identical name and arguments."""
    return sse([
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": f"call_{name}_a", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}},
            {"index": 1, "id": f"call_{name}_b", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}]},
            "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    ])


def text_response(text="Done."):
    return sse([
        {"choices": [{"delta": {"content": text}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
    ])


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        payload = script.pop(0) if script else text_response()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            self.wfile.write(payload.encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class PermAgent(Agent):
    """Agent client that can answer session/request_permission."""

    def reply(self, req_id, result):
        self.p.stdin.write((json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}) + "\n").encode())
        self.p.stdin.flush()


TIME_ARGS = {"timezone": "America/New_York"}
CONVERT_ARGS = {"source_timezone": "America/New_York", "time": "12:00",
                "target_timezone": "Europe/Warsaw"}


def main():
    if len(sys.argv) < 2:
        print("usage: acp_permission_smoke.py /path/to/mlx-agent")
        sys.exit(2)
    agent_bin = sys.argv[1]

    # INTERPRETER: any python that can run `-m mcp_server_time` (pip install mcp-server-time).
    # This used to hardcode a path into a sibling app's bundled python, which was wrong twice
    # over: mlx-agent is a standalone repo and must not take a private peer project as a test
    # dependency, and the path baked in one machine's directory layout, so on any other machine
    # every check failed on "no such file" rather than on anything about permissions.
    #
    # MLX_AGENT_SMOKE_PYTHON points at a different interpreter - a venv, or an app-bundled one
    # that has the module. Set PYTHONPATH yourself if that interpreter needs it to find the
    # module. PYTHONPYCACHEPREFIX is defaulted here because CPython writes __pycache__ next to
    # each imported module's SOURCE, so a bundled interpreter's packages would collect bytecode
    # inside a signed app bundle.
    os.environ.setdefault(
        "PYTHONPYCACHEPREFIX", os.path.join(tempfile.gettempdir(), "mlx-agent-pyc"))
    # shutil.which, because the two launchers disagree: subprocess.run below searches PATH, but
    # the agent spawns this command through Process.executableURL (MCPClients.launch), which does
    # NOT. So `MLX_AGENT_SMOKE_PYTHON=python3` would pass the probe and then fail to launch, and
    # a launch failure is only logged - the agent runs on with no servers and every check fails
    # on "unknown tool". which() only SEARCHES when the value has no separator; anything with a
    # "/" it merely validates (exists, executable, not a directory) and returns unchanged, so
    # abspath follows it. The pair makes the probe test the exact string that gets spawned.
    want = os.environ.get("MLX_AGENT_SMOKE_PYTHON") or sys.executable
    found = shutil.which(want)
    py = os.path.abspath(found) if found else ""  # guarded: abspath("") is the cwd
    try:
        if not py:
            raise OSError(f"{want}: not an executable file, and not on PATH")
        # find_spec, not import: the check has to reach `__main__` (the server is launched as
        # `-m mcp_server_time`, so an importable package without one would pass a plain import
        # of the package and then fail to spawn) - but IMPORTING `__main__` RUNS it, since it
        # has no `if __name__` guard. The server then blocks on stdin for JSON-RPC that never
        # comes, and the probe reports "cannot run" on a machine where it is installed and
        # working. A broken dependency chain is still caught, though that is this package's
        # doing rather than find_spec's: resolving the submodule imports the parent, whose
        # __init__ imports .server at module scope, which pulls mcp and pydantic in with it.
        # stdin is pinned closed for the same reason: whether that hang happens
        # at all depends on what the caller's stdin is, which is how it stayed invisible here -
        # a shell pipeline gives EOF and passes, a terminal hangs for the full timeout.
        # sys.exit(str) prints to stderr and exits 1, so the "no __main__" case reports itself
        # rather than being inferred from a silent status - which cannot be told apart from any
        # other executable that exits non-zero without output (/usr/bin/false, a wrapper shim).
        probe = subprocess.run(
            [py, "-c", "import importlib.util as u, sys; "
                       "sys.exit(0 if u.find_spec('mcp_server_time.__main__') else "
                       "'importable, but it has no __main__, so `-m` cannot run it')"],
            capture_output=True, stdin=subprocess.DEVNULL, timeout=60)
        bad = probe.returncode
        why = (probe.stderr or b"").decode(errors="replace").strip().splitlines()
    except (OSError, subprocess.TimeoutExpired) as e:
        # A nonexistent path raises rather than returning non-zero, and that is the likeliest
        # way to misconfigure the override - so the instructions below must survive it.
        bad, why = 1, [str(e)]
    if bad:
        # Every probe failure above names itself on stderr, so silence here means the thing being
        # run is not the interpreter it claims to be: it exited non-zero, or died on a signal
        # (which leaves no stderr either - the "Killed: 9" line comes from a shell, and there is
        # none here; an app-bundled python killed for a bad signature is exactly the setup the
        # comment above invites). Report the status rather than guessing at a cause.
        if not why:
            why = [f"exited with status {bad} and wrote nothing to stderr"]
        # sys.exit, not return: main()'s return value is discarded at the bottom of this file,
        # so returning a code would still exit 0 - a harness would read "cannot run" as "all
        # checks passed", which is worse than the stale path this replaced.
        print(f"{py or want} cannot run mcp_server_time; "
              "pip install mcp-server-time, or set MLX_AGENT_SMOKE_PYTHON to an interpreter "
              "that has it" + (f"\n  {why[-1]}" if why else ""), file=sys.stderr)
        sys.exit(2)

    cfg = {"servers": [{
        "name": "time",
        "command": py,
        "args": ["-m", "mcp_server_time", "--local-timezone", "America/New_York"],
        # Force BOTH tools gated. mcp_server_time is read-only in reality; gatedTools is a
        # config decision, which is exactly what lets this test use a real MCP server.
        "gatedTools": ["get_current_time", "convert_time"],
    }]}
    cfg_path = os.path.join(tempfile.mkdtemp(prefix="acp-perm-"), "mcp-config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    server = HTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
        if not ok:
            fails += 1

    agent = PermAgent([agent_bin, "acp", "--backend", "openai",
                       "--base-url", f"http://127.0.0.1:{PORT}/v1",
                       "--mcp-config", cfg_path])
    rid = agent.send("initialize", {"protocolVersion": 1})
    agent.await_response(rid)

    def new_session():
        r = agent.send("session/new", {"cwd": "/tmp"})
        return agent.await_response(r)["result"]["sessionId"]

    sid = new_session()

    def prompt(text, tool, args, answer):
        """One prompt whose model turn calls `tool`. `answer` -> optionId, or None to not reply.

        Returns (permission requests seen, final tool_call_update). The tool result is what
        distinguishes a standing allow from a standing deny - the request count cannot.
        """
        script.clear()
        script.append(tool_call_response(tool, args))
        script.append(text_response())
        seen = []
        updates = []

        def on_update(msg):
            if msg.get("method") == "session/request_permission":
                seen.append(msg)
                opt = answer(msg) if callable(answer) else answer
                if opt is not None:
                    agent.reply(msg["id"], {"outcome": {"outcome": "selected", "optionId": opt}})
            elif msg.get("method") == "session/update":
                u = (msg.get("params") or {}).get("update") or {}
                if u.get("sessionUpdate") == "tool_call_update" and u.get("status") in ("completed", "failed"):
                    updates.append(u)

        r = agent.send("session/prompt",
                       {"sessionId": sid, "prompt": [{"type": "text", "text": text}]})
        agent.await_response(r, on_update=on_update, timeout=90)
        return seen, (updates[-1] if updates else None)

    def ran(update):
        """The gated tool actually executed: completed, with the server's real output."""
        return bool(update) and update.get("status") == "completed" \
            and "permission denied" not in (update.get("rawOutput") or "")

    def denied(update):
        """The gate refused, and the denial reached the model as a tool result."""
        return bool(update) and update.get("status") == "failed" \
            and "permission denied" in (update.get("rawOutput") or "")

    # 1. the four options, with correct kinds
    seen, upd = prompt("time?", "get_current_time", TIME_ARGS, "allow")
    check("a gated tool asks, and allow runs it", len(seen) == 1 and ran(upd),
          f"requests={len(seen)} status={upd and upd.get('status')}")
    opts = seen[0]["params"]["options"] if seen else []
    got = {o["optionId"]: o["kind"] for o in opts}
    want = {"allow": "allow_once", "allow_always": "allow_always",
            "reject": "reject_once", "reject_always": "reject_always"}
    check("advertises ACP's standard four", got == want, f"got={got}")
    seen, upd = prompt("time?", "get_current_time", TIME_ARGS, "reject")
    check("reject denies, and the denial reaches the model", len(seen) == 1 and denied(upd),
          f"raw={(upd or {}).get('rawOutput')!r}")

    # 2. allow_always suppresses later requests for that tool AND still runs it
    seen, upd = prompt("time?", "get_current_time", TIME_ARGS, "allow_always")
    check("allow_always is answered once and runs", len(seen) == 1 and ran(upd),
          f"requests={len(seen)}")
    seen, upd = prompt("time again?", "get_current_time", TIME_ARGS, "reject")
    check("allow_always: NO later request, and the tool RAN", len(seen) == 0 and ran(upd),
          f"requests={len(seen)} status={upd and upd.get('status')}")

    # 3. the grant is per tool - the other gated tool still asks
    seen, upd = prompt("convert?", "convert_time", CONVERT_ARGS, "reject")
    check("grant is per tool: the other tool still asks", len(seen) == 1 and denied(upd),
          f"requests={len(seen)}")

    # 4. reject_always: denies later calls with no round-trip
    seen, upd = prompt("convert?", "convert_time", CONVERT_ARGS, "reject_always")
    check("reject_always is answered once and denies", len(seen) == 1 and denied(upd),
          f"requests={len(seen)}")
    seen, upd = prompt("convert again?", "convert_time", CONVERT_ARGS, "allow")
    check("reject_always: NO later request, and the tool was DENIED",
          len(seen) == 0 and denied(upd), f"requests={len(seen)} status={upd and upd.get('status')}")
    # allow_always must be unaffected by the reject_always on the OTHER tool
    seen, upd = prompt("time?", "get_current_time", TIME_ARGS, "reject")
    check("the two sets are independent", len(seen) == 0 and ran(upd), f"requests={len(seen)}")

    # 5a. session/prime clears both sets. THIS is the path the shipping client uses for a new
    # conversation: ChatView calls session/new once per spawn, and New Chat / sidebar switches
    # send session/prime (prime [] being the documented reset).
    r = agent.send("session/prime", {"sessionId": sid, "messages": []})
    agent.await_response(r)
    seen, upd = prompt("time?", "get_current_time", TIME_ARGS, "allow")
    check("session/prime [] clears allow_always", len(seen) == 1 and ran(upd),
          f"requests={len(seen)}")
    seen, upd = prompt("convert?", "convert_time", CONVERT_ARGS, "reject_always")
    check("... and a fresh grant works after prime", len(seen) == 1 and denied(upd),
          f"requests={len(seen)}")
    r = agent.send("session/prime", {"sessionId": sid,
                                     "messages": [{"role": "user", "content": "earlier"}]})
    agent.await_response(r)
    seen, upd = prompt("convert?", "convert_time", CONVERT_ARGS, "allow")
    check("session/prime with history clears reject_always", len(seen) == 1 and ran(upd),
          f"requests={len(seen)}")

    # 5b. session/new clears too (belt and braces; not the client's path)
    sid = new_session()
    seen, upd = prompt("time?", "get_current_time", TIME_ARGS, "allow_always")
    check("session/new: a grant can be made", len(seen) == 1 and ran(upd), f"requests={len(seen)}")
    sid = new_session()
    seen, upd = prompt("time?", "get_current_time", TIME_ARGS, "allow")
    check("session/new clears allow_always", len(seen) == 1 and ran(upd), f"requests={len(seen)}")

    # 6. an optionId we never advertised denies (fail-closed)
    seen, upd = prompt("time?", "get_current_time", TIME_ARGS, "bogus_option")
    check("unknown optionId denies", len(seen) == 1 and denied(upd),
          f"raw={(upd or {}).get('rawOutput')!r}")
    seen, upd = prompt("time?", "get_current_time", TIME_ARGS, "allow")
    check("unknown optionId did not become a standing grant", len(seen) == 1 and ran(upd),
          f"requests={len(seen)}")

    # 6b. an error response, and an explicit cancel, deny and are not remembered
    def error_reply(msg):
        agent.p.stdin.write((json.dumps(
            {"jsonrpc": "2.0", "id": msg["id"],
             "error": {"code": -32000, "message": "client blew up"}}) + "\n").encode())
        agent.p.stdin.flush()
        return None

    seen, upd = prompt("time?", "get_current_time", TIME_ARGS, error_reply)
    check("an error response denies", len(seen) == 1 and denied(upd),
          f"raw={(upd or {}).get('rawOutput')!r}")
    seen, upd = prompt("time?", "get_current_time", TIME_ARGS, "allow")
    check("an error response is not remembered", len(seen) == 1 and ran(upd),
          f"requests={len(seen)}")

    # 7. an answer that lands AFTER the session was replaced must not grant into the new session.
    #
    # Single-threaded on purpose: Agent has ONE reader queue, so two concurrent await_response
    # calls steal each other's messages. Instead, answer from INSIDE the reader loop - when the
    # permission request arrives, send session/new (replacing the session under the parked
    # request) and only then answer allow_always.
    #
    # The SLEEP is load-bearing, not sloppiness. session/new is handled asynchronously (health
    # check, stack rebuild), so replying immediately makes the grant land BEFORE the clear - it
    # is then wiped by the clear and no leak occurs, which makes the check vacuous. Verified:
    # without the wait this check passes even with the epoch guard neutered. The answer has to
    # arrive AFTER the swap has completed, which is the interleaving being defended against.
    def swap_then_allow(msg):
        agent.send("session/new", {"cwd": "/tmp"})  # response is ignored by await_response
        time.sleep(2.0)  # let session/new complete its swap + clear before the answer lands
        return "allow_always"

    script.clear()
    script.append(tool_call_response("get_current_time", TIME_ARGS))
    script.append(text_response())
    seen, upd = prompt("time?", "get_current_time", TIME_ARGS, swap_then_allow)
    # The grant was answered against the session that session/new just replaced; it must not
    # apply to the new one.
    seen, upd = prompt("time?", "get_current_time", TIME_ARGS, "allow")
    check("a stale answer does not grant into the NEW session", len(seen) == 1 and ran(upd),
          f"requests={len(seen)}")

    # 8. The duplicate-call short-circuit. Not about permissions, but this is the only harness
    # that scripts a model turn against REAL MCP tools (openai_wire_smoke.py scripts turns too,
    # with no servers), and the guardrail is otherwise untested - it was rewritten
    # when the dup cache moved out of runTurn into Agent state (Foundation Models dispatches
    # tools concurrently, so a runTurn local could not hold it any more).
    script.clear()
    script.append(dup_tool_call_response("convert_time", CONVERT_ARGS))
    script.append(text_response())
    outputs = []

    def on_dup(msg):
        if msg.get("method") == "session/request_permission":
            agent.reply(msg["id"], {"outcome": {"outcome": "selected", "optionId": "allow"}})
        elif msg.get("method") == "session/update":
            u = (msg.get("params") or {}).get("update") or {}
            if u.get("sessionUpdate") == "tool_call_update" and u.get("status") == "completed":
                outputs.append(u.get("rawOutput") or "")

    r = agent.send("session/prompt",
                   {"sessionId": sid, "prompt": [{"type": "text", "text": "convert twice"}]})
    agent.await_response(r, on_update=on_dup, timeout=90)
    shorted = [o for o in outputs if "duplicate call short-circuited" in o]
    check("an identical repeat call is short-circuited, not run twice",
          len(outputs) == 2 and len(shorted) == 1,
          f"completed={len(outputs)} shortCircuited={len(shorted)}")

    agent.close()
    server.shutdown()
    print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAILED'}")
    sys.exit(fails)


if __name__ == "__main__":
    main()
