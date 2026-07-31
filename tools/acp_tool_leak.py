#!/usr/bin/env python3
# ACP regression test: no raw model markers in the visible message, across a TOOL CALL.
#
# Guards two bugs that shipped together and produced the same symptom (marker text in the
# answer bubble), but had independent causes:
#
#   1. Backend.swift asked llama-server for `reasoning_format: "none"`, i.e. reasoning left
#      INLINE in delta.content as raw, per-family markers. gpt-oss emits harmony
#      (<|channel|>analysis<|message|>...<|end|>), which nothing downstream parsed, so it
#      rendered verbatim. Fixed by asking for "auto" and forwarding `reasoning_content`.
#
#   2. Agent.swift flushed ThinkSplitter only at end-of-TURN, not end-of-PASS. A pass that
#      ended in a tool call stranded the splitter's withheld tail (`keep` chars), which then
#      reappeared glued to the front of the next pass's answer: "<|channel|>" came out as
#      "<|ch" + ... + "annel|>".
#
#   3. ThinkSplitter knew only <think>/</think>, so Gemma 4's reasoning delimiters -
#      <|channel>thought\n...<channel|>, real special tokens - were classified as ordinary
#      message text and rendered in the answer bubble.
#
# A TOOL CALL IS REQUIRED to reproduce all three: (2) lives on the tool path, and Gemma 4's
# template only makes the model emit the channel itself when the previous message was a tool
# call or tool response (otherwise the template prefills a closed, empty block). That is why
# this exists alongside acp_smoke.py rather than inside it.
#
# (3) went unnoticed for a second, structural reason: this script used to hardcode
# `--backend openai`, so it only ever exercised llama-server - where reasoning_format "auto"
# makes the SERVER split reasoning into reasoning_content and the markers never reach
# ThinkSplitter. The native MLX path, which is the one the apps use for safetensors models,
# had no coverage at all. Hence --backend below: run it BOTH ways.
#
# Usage:
#   acp_tool_leak.py <mlx-agent> --backend openai --base-url <url> --mcp-config <path>
#   acp_tool_leak.py <mlx-agent> --backend mlx    --model <dir>    --mcp-config <path> \
#                    --no-expect-thought
#
# Needs an MCP server exporting a tool the prompt will trigger. Verified against
# llama-server 9553 + gpt-oss-20b-MXFP4, against gemma-4-12B-it-MLX-8bit on the MLX backend,
# and the bundled mcp_server_time. Exit code = number of failed checks (0 = all good).
#
# --no-expect-thought: Gemma 4 with thinking off emits an EMPTY thought channel, so "some
# reasoning was captured" is not a fair check there. It stays on by default because for
# gpt-oss an empty thought means reasoning was swallowed into the message, which is bug (1).

import argparse
import json
import queue
import re
import subprocess
import sys
import threading

# Full markers that must never survive into a visible assistant message.
HARMONY = ["<|channel|>", "<|message|>", "<|start|>", "<|end|>", "<|constrain|>", "<|return|>"]
THINK = ["<think>", "</think>"]
# Gemma 4's control tokens. The delimiter pairs put the pipe on ONE side only (the exception
# is the standalone "<|think|>"), so none of these is a substring of harmony's "<|channel|>"
# and checking for harmony alone never caught them.
GEMMA4 = ["<|channel>", "<channel|>", "<|turn>", "<turn|>", "<|think|>",
          "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>"]

# Fragments a TORN marker leaves behind. Checked only AFTER full markers are removed:
# "annel|>" is a substring of "<|channel|>", so searching for it naively just re-reports
# an intact-marker leak instead of the distinct tail-stranding bug.
TORN = ["annel|>", "<|ch", "hannel", "essage|>", "<|me", "ool_call|>", "urn|>"]


def reader(stream, q):
    for raw in stream:
        line = raw.decode("utf-8", "replace").rstrip("\n")
        if line:
            try:
                q.put(json.loads(line))
            except json.JSONDecodeError:
                pass


class Agent:
    def __init__(self, argv):
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.q = queue.Queue()
        threading.Thread(target=reader, args=(self.p.stdout, self.q), daemon=True).start()
        self._id = 0

    def call(self, method, params=None):
        self._id += 1
        msg = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            msg["params"] = params
        self.p.stdin.write((json.dumps(msg) + "\n").encode())
        self.p.stdin.flush()
        return self._id

    def reply(self, req_id, result):
        self.p.stdin.write(
            (json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}) + "\n").encode())
        self.p.stdin.flush()

    def pump(self, want_id, timeout=240):
        """Collect session/update notifications until the response to want_id arrives."""
        updates = []
        while True:
            msg = self.q.get(timeout=timeout)
            if msg.get("method") == "session/update":
                updates.append(msg["params"]["update"])
            elif msg.get("method") == "session/request_permission":
                # Auto-allow, so a gated tool cannot stall an unattended run.
                self.reply(msg["id"], {"outcome": {"outcome": "selected", "optionId": "allow"}})
            elif msg.get("id") == want_id:
                return msg, updates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent")
    ap.add_argument("--backend", choices=["openai", "mlx"], default="openai")
    ap.add_argument("--base-url", help="required by --backend openai")
    ap.add_argument("--model", help="model directory; required by --backend mlx")
    ap.add_argument("--mcp-config", required=True)
    ap.add_argument("--prompt", default="what time is it now in Warsaw?")
    ap.add_argument("--expect-thought", dest="expect_thought", action="store_true", default=True)
    ap.add_argument("--no-expect-thought", dest="expect_thought", action="store_false")
    args = ap.parse_args()

    if args.backend == "openai":
        if not args.base_url:
            ap.error("--backend openai needs --base-url")
        engine = ["--backend", "openai", "--base-url", args.base_url]
    else:
        if not args.model:
            ap.error("--backend mlx needs --model")
        engine = ["--backend", "mlx", "--model", args.model]

    agent = Agent([args.agent, "acp"] + engine + ["--mcp-config", args.mcp_config])
    agent.pump(agent.call("initialize", {"protocolVersion": 1}))
    resp, _ = agent.pump(agent.call("session/new", {"cwd": "/tmp", "mcpServers": []}))
    sid = resp["result"]["sessionId"]

    print(f"prompt: {args.prompt!r}\n")
    resp, updates = agent.pump(agent.call("session/prompt", {
        "sessionId": sid, "prompt": [{"type": "text", "text": args.prompt}]}))

    def joined(kind):
        return "".join(u["content"]["text"] for u in updates
                       if u.get("sessionUpdate") == kind)

    message, thought = joined("agent_message_chunk"), joined("agent_thought_chunk")
    tools = [u for u in updates if u.get("sessionUpdate") == "tool_call"]

    print(f"backend    : {args.backend}")
    print(f"stopReason : {resp.get('result', {}).get('stopReason')}")
    print(f"tool calls : {[t.get('title') for t in tools]}")
    print(f"thought    : {thought[:160]!r}")
    print(f"MESSAGE    : {message[:400]!r}\n")

    fails = 0

    def check(ok, label):
        nonlocal fails
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            fails += 1

    # Without a tool call the run says nothing about bug (2); fail loudly rather than
    # report a green that was never earned.
    check(bool(tools), "the model called a tool (REQUIRED - bug 2 lives on the tool path)")

    markers = HARMONY + THINK + GEMMA4
    leaked = [m for m in markers if m in message]
    check(not leaked, f"no raw markers in the visible message (found {leaked})")

    stripped = message
    for m in markers:
        stripped = stripped.replace(m, "")
    torn = [f for f in TORN if f in stripped]
    check(not torn, f"no TORN marker fragment in the message (found {torn})")

    check(bool(message.strip()), "the message is non-empty")
    check(not re.match(r"^\s*[a-z|>]{2,8}\|>", message),
          "the message does not open with a torn marker fragment")
    # Gemma 4's channel LABEL is metadata, never content - a bare "thought" opening the
    # answer means the label was routed instead of consumed.
    check(not re.match(r"^\s*(thought|content)\s*$", message.split("\n")[0]),
          "the message does not open with a bare channel label")
    if args.expect_thought:
        check(bool(thought.strip()),
              "reasoning was captured as thought (not swallowed into the message)")

    agent.p.terminate()
    print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' CHECK(S) FAILED'}")
    return fails


if __name__ == "__main__":
    sys.exit(main())
