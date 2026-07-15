#!/usr/bin/env python3
# Wire-shape tests for `mlx-agent acp --backend openai`.
#
# Stands up a FAKE OpenAI-compatible server (no model, no llama-server) that records the
# exact request body of every /chat/completions call and replies with scripted SSE. That
# makes the properties a live model can only demonstrate by luck deterministic:
#
#   1. An unanswered tool-call announcement never reaches the wire. Agent.runTurn dispatches
#      tool calls but only feeds the results back on its NEXT backend.stream() call, and the
#      tool-iteration cap returns BEFORE making that call - so the backend's conversation is
#      left holding an assistant tool_calls announcement with no tool results after it.
#      Re-sending that verbatim leaves the chat template mid-tool-turn; llama-server renders
#      the template server-side on every turn, so a strict template would then hard-fail
#      every subsequent prompt in the session.
#   2. function.arguments goes out as a JSON STRING (the OpenAI spec), not the JSON object
#      the library's MessageGenerator bridge emits.
#   3. A cancelled turn still records the partial answer, so the conversation never sends
#      two user messages in a row.
#
# Usage: python3 tools/openai_wire_smoke.py /path/to/mlx-agent
# Exit code = number of failed checks (0 = all good).

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, "tools")
from acp_smoke import Agent, collect_updates

PORT = 8299
requests_seen = []          # one parsed request body per /chat/completions call
script = []                 # queued responses, one per call

def sse(chunks):
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"

def tool_call_response():
    """An assistant turn that emits one tool call, split across deltas like llama-server."""
    return sse([
        {"choices": [{"delta": {"content": "Let me check. "}, "finish_reason": None}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_abc", "type": "function",
             "function": {"name": "get_current_time", "arguments": ""}}]}, "finish_reason": None}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '{"timezone"'}}]}, "finish_reason": None}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": ': "America/New_York"}'}}]}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    ])

def text_response(text="Done."):
    return sse([
        {"choices": [{"delta": {"content": text}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
    ])

def slow_text_response():
    """Streams forever, a token at a time, so a cancel lands mid-answer."""
    def gen():
        yield "data: " + json.dumps(
            {"choices": [{"delta": {"content": "counting "}, "finish_reason": None}]}) + "\n\n"
        for i in range(2000):
            yield "data: " + json.dumps(
                {"choices": [{"delta": {"content": f"{i} "}, "finish_reason": None}]}) + "\n\n"
            time.sleep(0.02)
    return gen

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
        body = self.rfile.read(int(self.headers["Content-Length"]))
        requests_seen.append(json.loads(body))
        payload = script.pop(0) if script else text_response()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            if callable(payload):
                for frame in payload()():
                    self.wfile.write(frame.encode())
                    self.wfile.flush()
            else:
                self.wfile.write(payload.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass    # the client cancelled: exactly what the cancel check wants to cause

def main():
    if len(sys.argv) < 2:
        print("usage: openai_wire_smoke.py /path/to/mlx-agent")
        sys.exit(2)
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    fails = 0
    def check(name, ok, detail=""):
        nonlocal fails
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
        if not ok:
            fails += 1

    base = f"http://127.0.0.1:{PORT}/v1"
    # --max-tool-iters 1: the turn dispatches the tool call, then hits the cap before the
    # results can reach the backend. That is the state that strands the announcement.
    agent = Agent([sys.argv[1], "acp", "--backend", "openai", "--base-url", base,
                   "--max-tool-iters", "1"])
    try:
        agent.await_response(agent.send("initialize", {"protocolVersion": 1}), timeout=30)
        sid = agent.await_response(agent.send("session/new", {"cwd": "/tmp", "mcpServers": []}),
                                   timeout=30)["result"]["sessionId"]

        # Turn 1: the model announces a tool call; the cap ends the turn right after dispatch.
        script.append(tool_call_response())
        r = agent.await_response(agent.send("session/prompt", {
            "sessionId": sid, "prompt": [{"type": "text", "text": "what time is it?"}]}),
            on_update=collect_updates({}), timeout=60)
        check("tool cap ends the turn", r.get("result", {}).get("stopReason") == "max_turn_requests",
              f"stopReason={r.get('result', {}).get('stopReason')}")

        # Turn 2: whatever the backend now believes the conversation is, goes on the wire.
        script.append(text_response())
        agent.await_response(agent.send("session/prompt", {
            "sessionId": sid, "prompt": [{"type": "text", "text": "never mind, hello"}]}),
            on_update=collect_updates({}), timeout=60)

        check("a second request was made", len(requests_seen) >= 2, f"{len(requests_seen)} requests")
        messages = requests_seen[1]["messages"]
        dangling = []
        for i, m in enumerate(messages):
            if not m.get("tool_calls"):
                continue
            announced = {c.get("id") for c in m["tool_calls"]}
            answered = set()
            for later in messages[i + 1:]:
                if later.get("role") != "tool":
                    break
                answered.add(later.get("tool_call_id"))
            if not announced <= answered:
                dangling.append(i)
        check("no unanswered tool-call announcement reaches the wire", not dangling,
              f"messages={[m.get('role') for m in messages]}")
        check("consecutive user turns are not produced",
              not any(messages[i].get("role") == "user" and messages[i + 1].get("role") == "user"
                      for i in range(len(messages) - 1)),
              f"roles={[m.get('role') for m in messages]}")

        # A well-formed tool round-trip must still serialize arguments as a STRING.
        requests_seen.clear()
        script.append(tool_call_response())
        script.append(text_response("The time is 5pm."))
        agent2 = Agent([sys.argv[1], "acp", "--backend", "openai", "--base-url", base])
        try:
            agent2.await_response(agent2.send("initialize", {"protocolVersion": 1}), timeout=30)
            sid2 = agent2.await_response(agent2.send("session/new", {"cwd": "/tmp", "mcpServers": []}),
                                         timeout=30)["result"]["sessionId"]
            agent2.await_response(agent2.send("session/prompt", {
                "sessionId": sid2, "prompt": [{"type": "text", "text": "what time is it?"}]}),
                on_update=collect_updates({}), timeout=60)
            follow = requests_seen[1]["messages"]
            assistant = next((m for m in follow if m.get("tool_calls")), None)
            check("the answered announcement is preserved", assistant is not None,
                  f"roles={[m.get('role') for m in follow]}")
            if assistant:
                args = assistant["tool_calls"][0]["function"]["arguments"]
                check("function.arguments serializes as a JSON string", isinstance(args, str),
                      f"type={type(args).__name__}, value={args!r}")
                check("the tool result answers the announcement by id",
                      any(m.get("role") == "tool" and m.get("tool_call_id") == "call_abc"
                          for m in follow), "")
        finally:
            agent2.close()

        # A cancelled turn must still record what the user saw.
        requests_seen.clear()
        script.append(slow_text_response)
        agent3 = Agent([sys.argv[1], "acp", "--backend", "openai", "--base-url", base])
        try:
            agent3.await_response(agent3.send("initialize", {"protocolVersion": 1}), timeout=30)
            sid3 = agent3.await_response(agent3.send("session/new", {"cwd": "/tmp", "mcpServers": []}),
                                         timeout=30)["result"]["sessionId"]
            rid = agent3.send("session/prompt", {
                "sessionId": sid3, "prompt": [{"type": "text", "text": "count for me"}]})
            time.sleep(1.0)
            agent3.send("session/cancel", {"sessionId": sid3}, notify=True)
            r = agent3.await_response(rid, on_update=collect_updates({}), timeout=60)
            check("cancel resolves the turn", r.get("result", {}).get("stopReason") == "cancelled",
                  f"stopReason={r.get('result', {}).get('stopReason')}")
            script.append(text_response())
            agent3.await_response(agent3.send("session/prompt", {
                "sessionId": sid3, "prompt": [{"type": "text", "text": "stop"}]}),
                on_update=collect_updates({}), timeout=60)
            roles = [m.get("role") for m in requests_seen[-1]["messages"]]
            check("a cancelled turn leaves no two user turns in a row",
                  not any(roles[i] == "user" and roles[i + 1] == "user" for i in range(len(roles) - 1)),
                  f"roles={roles}")
        finally:
            agent3.close()

    except Exception as e:
        print(f"[FAIL] exception: {e!r}")
        fails += 1
    finally:
        agent.close()
        server.shutdown()

    print("-" * 60)
    print(f"{'ALL PASS' if fails == 0 else str(fails) + ' FAILED'}")
    sys.exit(fails)

if __name__ == "__main__":
    main()
