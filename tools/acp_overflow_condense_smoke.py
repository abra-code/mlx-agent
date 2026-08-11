#!/usr/bin/env python3
# Prime a conversation that does not fit, then prompt - and expect an answer, not an error.
#
# The foundation backend has a 4096-token window, whole. Before summarize-and-retry, a restored
# conversation of any real length made every prompt fail with exceededContextWindowSize: not a
# degraded answer, a dead session. This drives that exact shape.
#
# The hard asserts are STRUCTURAL, and deliberately so - whether a 3B model's summary happens to
# retain any particular fact is nondeterministic, and asserting on it would be a flaky lie. What
# is deterministic: the agent survives, the turn completes, and the condensation is reported on
# stderr with counts. Fact recall is printed as an observation, not a check.
#
# Usage: python3 tools/acp_overflow_condense_smoke.py /path/to/mlx-agent [--backend foundation | --model <dir>]
# Exit code = number of failed checks (0 = all good).

import json
import queue
import re
import subprocess
import sys
import threading
import time

CODENAME = "QUETZAL-9"


class Agent:
    def __init__(self, argv):
        self.p = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.q = queue.Queue()
        self.err = []
        self.streamed = ""
        self.n = 0
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._errReader, daemon=True).start()

    def _reader(self):
        for raw in self.p.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            # The answer arrives as session/update notifications, NOT in the prompt response -
            # that carries only a stopReason. Accumulate it here or there is nothing to inspect.
            if msg.get("method") == "session/update":
                up = msg.get("params", {}).get("update", {})
                if up.get("sessionUpdate") in ("agent_message_chunk", "agent_thought_chunk"):
                    content = up.get("content") or {}
                    if isinstance(content, dict):
                        self.streamed += content.get("text", "")
            self.q.put(msg)

    def _errReader(self):
        for raw in self.p.stderr:
            text = raw.decode("utf-8", "replace").rstrip()
            self.err.append(text)
            print("  | " + text)

    def send(self, method, params):
        self.n += 1
        payload = {"jsonrpc": "2.0", "id": self.n, "method": method, "params": params}
        self.p.stdin.write((json.dumps(payload) + "\n").encode())
        self.p.stdin.flush()
        return self.n

    def await_response(self, rid, timeout=300):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                m = self.q.get(timeout=max(0.1, deadline - time.time()))
            except queue.Empty:
                break
            if m.get("id") == rid and ("result" in m or "error" in m):
                return m
        return {}

    def stderr_text(self):
        return "\n".join(self.err)

    def alive(self):
        return self.p.poll() is None


def oversized_history():
    """~19 KB of conversation - comfortably over a 4096-token window.

    Sized against the EXACT tokenizer, not bytes/4. English filler tokenizes at better than four
    bytes per token, so an 11 KB history that looks over budget by the byte estimate actually
    fits, and the agent (correctly) declines to condense it. It has to be genuinely too large or
    this test asserts nothing.

    The codename goes in the FIRST exchange so it lands in the summarized part rather than the
    verbatim tail; otherwise the test would pass without the digest doing anything.
    """
    filler = (
        "We walked through the packaging steps in detail, covering the installer layout, the "
        "component plist, the postinstall script ordering, and how the receipt is validated on "
        "the next launch. None of that changed the conclusion but it took a while to confirm. ")
    messages = [
        {"role": "user",
         "content": f"Before anything else: the project codename is {CODENAME}. Remember it."},
        {"role": "assistant",
         "content": f"Noted. The project codename is {CODENAME}."},
    ]
    for i in range(34):
        messages.append({"role": "user", "content": f"Step {i}: what about the installer? " + filler})
        messages.append({"role": "assistant", "content": f"For step {i}, the answer is the same. " + filler})
    return messages


def main():
    if len(sys.argv) < 2:
        print("usage: acp_overflow_condense_smoke.py /path/to/mlx-agent [backend args]")
        return 2

    failures = []

    def check(name, ok, detail=""):
        print(("[PASS] " if ok else "[FAIL] ") + name + ("" if ok else f"  {detail}"))
        if not ok:
            failures.append(name)

    agent = Agent([sys.argv[1], "acp"] + sys.argv[2:])
    agent.await_response(agent.send("initialize", {"protocolVersion": 1}))
    r = agent.await_response(agent.send("session/new", {"cwd": "/tmp"}))
    sid = (r.get("result") or {}).get("sessionId")
    if not sid:
        print(f"[FAIL] session/new: {r}")
        return 1

    history = oversized_history()
    wire_bytes = sum(len(m["content"]) for m in history)
    print(f"priming {len(history)} messages, {wire_bytes} bytes (~{wire_bytes // 4} tokens)")
    r = agent.await_response(agent.send("session/prime", {"sessionId": sid, "messages": history}))
    check("session/prime accepts an oversized history",
          (r.get("result") or {}).get("primed") == len(history),
          f"result={r.get('result') or r.get('error')}")

    # The turn that used to be impossible.
    started = time.time()
    rid = agent.send("session/prompt", {
        "sessionId": sid,
        "prompt": [{"type": "text", "text": "What is the project codename? Answer in one line."}],
    })
    resp = agent.await_response(rid, timeout=300)
    elapsed = time.time() - started

    check("the agent survived the overflow path", agent.alive(),
          f"exit {agent.p.returncode}")
    stop = (resp.get("result") or {}).get("stopReason")
    check("the prompt completed instead of failing on window size", stop == "end_turn",
          f"stopReason={stop} error={resp.get('error')}")

    err = agent.stderr_text()
    # Only the foundation backend condenses - the others have room. Run against MLX as a control
    # (does the same prime-then-prompt work at all?) and the condense-specific checks stand down
    # rather than reporting a failure that means nothing there.
    if "backend: foundation" in err:
        condensed = re.search(
            r"summarized (\d+) turns \((\d+) bytes\) into digest ([0-9a-f]{8})", err)
        check("the condensation was reported with counts", condensed is not None,
              "no 'summarized N turns' line on stderr")
        if condensed:
            turns, dropped_bytes, sha = condensed.groups()
            check("more than the verbatim tail was summarized", int(turns) > 6, f"turns={turns}")
            print(f"       digest {sha}: {turns} turns, {dropped_bytes} bytes replaced")
        check("no fallback-to-full-history line", "was left whole" not in err,
              "the summarizer bailed; see stderr above")
    else:
        print("[skip] condense checks: this backend has room for the history")

    answer = agent.streamed.strip()
    check("the answer was not empty", answer != "", "nothing streamed")
    # A retry after a partial answer would concatenate two replies. The prompt asks for one line,
    # so anything essay-sized here means the retry rule leaked.
    check("the answer is one reply, not two stitched together", len(answer) < 400,
          f"{len(answer)} chars streamed for a one-line question")

    # The session was rebuilt around the digest. Nothing else in the suite prompts a rebuilt
    # session, so without this a "the new session is unusable" regression is invisible.
    agent.streamed = ""
    rid = agent.send("session/prompt", {
        "sessionId": sid,
        "prompt": [{"type": "text", "text": "Say OK."}],
    })
    second = agent.await_response(rid, timeout=120)
    check("a second prompt works on the session rebuilt around the digest",
          agent.alive() and (second.get("result") or {}).get("stopReason") == "end_turn",
          f"stopReason={(second.get('result') or {}).get('stopReason')} alive={agent.alive()}")

    # Observation, not a check: a 3B summarizer keeping any particular fact is not deterministic,
    # and asserting on it would make this test lie about what it verifies.
    # `answer`, not agent.streamed - the second prompt above reset the accumulator.
    print(f"       turn took {elapsed:.1f} s")
    print(f"       answer: {answer[:200]}")
    print(f"       codename survived condensation: "
          f"{'YES' if CODENAME.lower() in answer.lower() else 'no'}")

    agent.p.kill()

    # The negative case, and the reason it exists: a bug that made the fit check fire on EVERY
    # prompt passed every assertion above. Condensing a conversation that fits costs seconds and
    # throws away fidelity for nothing, and it is completely silent.
    print("-" * 60)
    print("control: a short conversation must NOT be condensed")
    small = Agent([sys.argv[1], "acp"] + sys.argv[2:])
    small.await_response(small.send("initialize", {"protocolVersion": 1}))
    r = small.await_response(small.send("session/new", {"cwd": "/tmp"}))
    small_sid = (r.get("result") or {}).get("sessionId")
    small.await_response(small.send("session/prime", {
        "sessionId": small_sid,
        "messages": [
            {"role": "user", "content": "My favorite color is teal."},
            {"role": "assistant", "content": "Noted, teal."},
            {"role": "user", "content": "And my favorite number is 7."},
            {"role": "assistant", "content": "Noted, 7."},
        ],
    }))
    rid = small.send("session/prompt", {
        "sessionId": small_sid,
        "prompt": [{"type": "text", "text": "What is my favorite color?"}],
    })
    r = small.await_response(rid, timeout=120)
    check("the short conversation completed",
          (r.get("result") or {}).get("stopReason") == "end_turn",
          f"result={r.get('result') or r.get('error')}")
    check("the short conversation was not summarized",
          "summarized" not in small.stderr_text() and "was left whole" not in small.stderr_text(),
          "a conversation that fits was condensed anyway")
    small.p.kill()

    print("-" * 60)
    print("ALL PASS" if not failures else f"{len(failures)} FAILED: " + ", ".join(failures))
    return len(failures)


if __name__ == "__main__":
    sys.exit(main())
