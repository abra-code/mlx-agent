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

def tool_call_response(preamble="Let me check. "):
    """An assistant turn that emits one tool call, split across deltas like llama-server.

    `preamble=""` is the realistic text-less announcement (models routinely call a tool with
    no preamble). It matters: with no text, an announcement stripped as unanswered loses its
    whole message, so the turn's user message ends up adjacent to the next one.
    """
    return sse([
        {"choices": [{"delta": {"content": preamble}, "finish_reason": None}]},
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

def error_response():
    """A 500 with a JSON body: what a llama-server being restarted under us looks like."""
    return ("HTTP_ERROR", 500, '{"error":{"message":"model is being reloaded"}}')

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
        if isinstance(payload, tuple) and payload[0] == "HTTP_ERROR":
            _, status, text = payload
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(text.encode())
            return
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

        # Turn 1: the model announces a tool call with NO text preamble, and the cap ends the
        # turn right after dispatch. Text-less is the harder case: the stripped announcement
        # keeps nothing, so the assistant message disappears and only the normalization below
        # stops this turn's user message from colliding with the next one.
        script.append(tool_call_response(preamble=""))
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

        # A turn that fails BEFORE producing a single token (the server 500s mid-session -
        # what an applet-driven llama-server restart looks like) leaves nothing to flush, so
        # only user-turn normalization keeps the next request well-formed.
        requests_seen.clear()
        script.append(error_response())
        agent4 = Agent([sys.argv[1], "acp", "--backend", "openai", "--base-url", base])
        try:
            agent4.await_response(agent4.send("initialize", {"protocolVersion": 1}), timeout=30)
            sid4 = agent4.await_response(agent4.send("session/new", {"cwd": "/tmp", "mcpServers": []}),
                                         timeout=30)["result"]["sessionId"]
            r = agent4.await_response(agent4.send("session/prompt", {
                "sessionId": sid4, "prompt": [{"type": "text", "text": "first try"}]}),
                on_update=collect_updates({}), timeout=60)
            check("a zero-token server error fails the turn cleanly",
                  "error" in r, f"got={list(r)}")
            script.append(text_response("Hello."))
            r = agent4.await_response(agent4.send("session/prompt", {
                "sessionId": sid4, "prompt": [{"type": "text", "text": "second try"}]}),
                on_update=collect_updates({}), timeout=60)
            check("the next prompt still works after a failed turn",
                  r.get("result", {}).get("stopReason") == "end_turn",
                  f"stopReason={r.get('result', {}).get('stopReason')}")
            roles = [m.get("role") for m in requests_seen[-1]["messages"]]
            check("a zero-token failure leaves no two user turns in a row",
                  not any(roles[i] == "user" and roles[i + 1] == "user"
                          for i in range(len(roles) - 1)),
                  f"roles={roles}")
            merged = requests_seen[-1]["messages"][-1]["content"]
            check("merging keeps both user messages", "first try" in merged and "second try" in merged,
                  f"content={merged!r}")
        finally:
            agent4.close()

        # session/prime accepts a PARTIALLY answered announcement (2 calls, 1 answered) and
        # passes it through mid-transcript. Stripping the array wholesale would strand the
        # answered tool message as an orphan - as malformed as the dangling announcement.
        requests_seen.clear()
        script.append(text_response("Sure."))
        agent5 = Agent([sys.argv[1], "acp", "--backend", "openai", "--base-url", base])
        try:
            agent5.await_response(agent5.send("initialize", {"protocolVersion": 1}), timeout=30)
            sid5 = agent5.await_response(agent5.send("session/new", {"cwd": "/tmp", "mcpServers": []}),
                                         timeout=30)["result"]["sessionId"]
            agent5.await_response(agent5.send("session/prime", {
                "sessionId": sid5,
                "messages": [
                    {"role": "user", "content": "do two things"},
                    {"role": "assistant", "content": "working",
                     "toolCalls": [{"id": "tc-a", "name": "read_file", "arguments": {"path": "/a"}},
                                   {"id": "tc-b", "name": "read_file", "arguments": {"path": "/b"}}]},
                    {"role": "tool", "content": "A", "toolCallId": "tc-a"},
                ]}), timeout=30)
            agent5.await_response(agent5.send("session/prompt", {
                "sessionId": sid5, "prompt": [{"type": "text", "text": "and now?"}]}),
                on_update=collect_updates({}), timeout=60)
            msgs = requests_seen[-1]["messages"]
            orphans = [i for i, m in enumerate(msgs) if m.get("role") == "tool"
                       and not (i > 0 and msgs[i - 1].get("tool_calls")
                                and any(c.get("id") == m.get("tool_call_id")
                                        for c in msgs[i - 1]["tool_calls"]))]
            check("a partially answered announcement leaves no orphan tool message",
                  not orphans, f"roles={[m.get('role') for m in msgs]}")
            assistant = next((m for m in msgs if m.get("tool_calls")), None)
            check("the answered call survives, the unanswered one is dropped",
                  assistant is not None
                  and [c["id"] for c in assistant["tool_calls"]] == ["tc-a"],
                  f"tool_calls={assistant.get('tool_calls') if assistant else None}")
        finally:
            agent5.close()

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
