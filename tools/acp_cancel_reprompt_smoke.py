#!/usr/bin/env python3
# Cancel, then prompt again, on the SAME session - repeatedly.
#
# This shape is deliberately absent from acp_smoke.py, and that absence hid a process-killing
# crash: its cancel check is immediately followed by session/prime, which rebuilds the backend and
# its session. A plain prompt after a cancel never happens there.
#
# On the foundation backend, reusing a LanguageModelSession whose previous streamResponse was
# cancelled traps inside FoundationModels (SIGTRAP), killing the whole agent - reproduced in 14 of
# 20 attempts before FoundationBackend started rebuilding the session on a non-clean pass. It is
# exactly what a user does: press Stop, then ask something else.
#
# Usage: python3 tools/acp_cancel_reprompt_smoke.py /path/to/mlx-agent [--backend foundation | --model <dir>] [rounds]
# Exit code = number of failed rounds (0 = all good).

import json
import queue
import subprocess
import sys
import threading
import time


class Agent:
    def __init__(self, argv):
        self.p = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None)
        self.q = queue.Queue()
        self.n = 0
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for raw in self.p.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if line:
                try:
                    self.q.put(json.loads(line))
                except json.JSONDecodeError:
                    pass

    def send(self, method, params):
        self.n += 1
        payload = {"jsonrpc": "2.0", "id": self.n, "method": method, "params": params}
        self.p.stdin.write((json.dumps(payload) + "\n").encode())
        self.p.stdin.flush()
        return self.n

    def notify(self, method, params):
        self.p.stdin.write(
            (json.dumps({"jsonrpc": "2.0", "method": method, "params": params}) + "\n").encode())
        self.p.stdin.flush()

    def await_response(self, rid, timeout=120):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                m = self.q.get(timeout=deadline - time.time())
            except queue.Empty:
                break
            if m.get("id") == rid and ("result" in m or "error" in m):
                return m
        return {}

    def alive(self):
        return self.p.poll() is None


def main():
    if len(sys.argv) < 2:
        print("usage: acp_cancel_reprompt_smoke.py /path/to/mlx-agent [backend args] [rounds]")
        return 2
    args = sys.argv[2:]
    rounds = 6
    if args and args[-1].isdigit():
        rounds = int(args[-1])
        args = args[:-1]

    agent = Agent([sys.argv[1], "acp"] + args)
    agent.await_response(agent.send("initialize", {"protocolVersion": 1}))
    r = agent.await_response(agent.send("session/new", {"cwd": "/tmp"}))
    sid = (r.get("result") or {}).get("sessionId")
    if not sid:
        print(f"[FAIL] session/new: {r}")
        return 1

    failures = 0
    for i in range(1, rounds + 1):
        # Start a long answer, let it stream briefly, then cancel it.
        agent.send("session/prompt", {
            "sessionId": sid,
            "prompt": [{"type": "text", "text": "Write a long detailed essay about the history of cartography."}],
        })
        time.sleep(1.0)
        agent.notify("session/cancel", {"sessionId": sid})
        time.sleep(0.3)

        if not agent.alive():
            print(f"[FAIL] round {i}: agent died during cancel (exit {agent.p.returncode})")
            return failures + 1

        # The step acp_smoke.py never takes: prompt again on the same session.
        rid = agent.send("session/prompt", {
            "sessionId": sid,
            "prompt": [{"type": "text", "text": "What is 2 plus 2? Answer with just the number."}],
        })
        resp = agent.await_response(rid, timeout=90)

        if not agent.alive():
            print(f"[FAIL] round {i}: AGENT CRASHED on the prompt after a cancel "
                  f"(exit {agent.p.returncode}) - this is the SIGTRAP")
            return failures + 1
        stop = (resp.get("result") or {}).get("stopReason")
        if stop != "end_turn":
            print(f"[FAIL] round {i}: prompt after cancel did not complete (stopReason={stop})")
            failures += 1
        else:
            print(f"[PASS] round {i}: cancel then prompt survived")

    agent.p.kill()
    print("-" * 60)
    print("ALL PASS" if failures == 0 else f"{failures} FAILED")
    return failures


if __name__ == "__main__":
    sys.exit(main())
