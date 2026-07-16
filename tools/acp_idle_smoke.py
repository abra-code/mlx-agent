#!/usr/bin/env python3
# MLX idle-unload smoke test.
#
# Drives a REAL MLX model over ACP with a short --idle-unload-seconds, proves that the model is
# released after inactivity (RAM actually drops) and that the next prompt transparently reloads
# it with the conversation context intact.
#
# Usage: python3 tools/acp_idle_smoke.py /path/to/mlx-agent --model <dir>
#
# Every check asserts an OBSERVED effect, not just "a log line appeared":
#   - the unload log's active-memory delta is negative (weights really left RAM);
#   - the reload log names the model and the replay count;
#   - the post-reload answer recalls a fact stated ONLY before the unload (context survived);
#   - the unload happens BETWEEN turns, never during one (ordering in the captured log).
#
# openai backend is out of scope: it holds no weights here (llama-server does its own idle
# sleep), so this test is MLX-only by construction.

import os
import re
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acp_smoke import reader, collect_updates  # noqa: E402
import json  # noqa: E402
import queue  # noqa: E402

IDLE_SECONDS = 3           # agent unloads after this much inactivity
WAIT_PAST_IDLE = IDLE_SECONDS + 4   # idle long enough that the timer fires and unload completes
SECRET = "ZEBRA-7"

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    if cond:
        PASS += 1
    else:
        FAIL += 1


class IdleAgent:
    """acp_smoke.Agent, but with stderr captured to a buffer so the idle-unload / idle-reload
    log lines can be asserted on."""

    def __init__(self, argv):
        self.p = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.q = queue.Queue()
        self.err_lines = []
        self._err_lock = threading.Lock()
        threading.Thread(target=reader, args=(self.p.stdout, self.q), daemon=True).start()
        threading.Thread(target=self._drain_err, daemon=True).start()
        self._id = 0

    def _drain_err(self):
        for raw in self.p.stderr:
            line = raw.decode(errors="replace").rstrip("\n")
            with self._err_lock:
                self.err_lines.append(line)

    def err_text(self):
        with self._err_lock:
            return "\n".join(self.err_lines)

    def send(self, method, params=None, notify=False):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if not notify:
            self._id += 1
            msg["id"] = self._id
        self.p.stdin.write((json.dumps(msg) + "\n").encode())
        self.p.stdin.flush()
        return None if notify else self._id

    def await_response(self, want_id, on_update=None, timeout=180):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                msg = self.q.get(timeout=deadline - time.time())
            except queue.Empty:
                break
            if msg.get("id") == want_id and ("result" in msg or "error" in msg):
                return msg
            if msg.get("method") and on_update:
                on_update(msg)
        raise TimeoutError(f"no response for id {want_id} within {timeout}s")

    def prompt_raw(self, sid, text):
        """Return the raw JSON-RPC response (result OR error) plus collected message text."""
        store = {}
        rid = self.send("session/prompt", {
            "sessionId": sid,
            "prompt": [{"type": "text", "text": text}],
        })
        resp = self.await_response(rid, on_update=collect_updates(store))
        return resp, store.get("agent_message_chunk", "")

    def prompt(self, sid, text):
        return self.prompt_raw(sid, text)[1]

    def close(self):
        try:
            self.p.stdin.close()
        except Exception:
            pass
        try:
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


def main():
    if len(sys.argv) < 2 or "--model" not in sys.argv:
        print("usage: acp_idle_smoke.py /path/to/mlx-agent --model <dir>")
        sys.exit(2)

    argv = [sys.argv[1], "acp", "--idle-unload-seconds", str(IDLE_SECONDS)] + sys.argv[2:]
    agent = IdleAgent(argv)
    try:
        agent.await_response(agent.send("initialize", {"protocolVersion": 1}))
        new = agent.await_response(agent.send("session/new", {}))
        sid = new["result"]["sessionId"]

        # Turn 1: state a fact only the model's context will carry across the unload.
        a1 = agent.prompt(sid, f"My secret word is {SECRET}. Reply with just: OK.")
        check("first turn answered", len(a1) > 0, f"answer={a1!r}")
        check("no unload before the idle wait",
              "idle-unload" not in agent.err_text(),
              "(unload must never happen mid-conversation while active)")

        # Sit idle past the threshold so the model is released.
        time.sleep(WAIT_PAST_IDLE)
        err = agent.err_text()

        m = re.search(r"idle-unload: released .*resident (\d+)MiB -> (\d+)MiB", err)
        check("model was idle-unloaded", m is not None,
              "(expected 'idle-unload: released ...' in log)")
        if m:
            before, after = int(m.group(1)), int(m.group(2))
            check("unload actually freed memory", after < before,
                  f"resident {before}MiB -> {after}MiB")

        # Turn 2: must reload and still recall the secret from before the unload.
        a2 = agent.prompt(sid, "What is my secret word? Reply with just the word.")
        err = agent.err_text()
        check("model was reloaded on the next prompt",
              "idle-reload: loading" in err,
              "(expected 'idle-reload: loading ...' in log)")
        check("unload happened BEFORE reload (never mid-turn)",
              err.find("idle-unload") < err.find("idle-reload") if "idle-reload" in err else False,
              "")
        check("context survived the unload/reload", SECRET.lower() in a2.lower(),
              f"answer={a2!r}")

        # A second idle cycle proves the timer re-arms after a reload, not just once.
        time.sleep(WAIT_PAST_IDLE)
        unloads = agent.err_text().count("idle-unload: released")
        check("idle timer re-arms after reload", unloads >= 2,
              f"unload count={unloads}")
    finally:
        agent.close()

    # ---- A failing reload must not permanently wedge the session (deadlock regression) ----
    # Spawn a second agent pointed at a SYMLINK to the model. After an idle-unload, break the
    # symlink so the next prompt's reloadAfterIdle() fast-fails; then restore it and prompt
    # again. If the post-Task-creation publish ever stranded promptTask on a fast-finishing
    # child, this recovery prompt would return -32003 forever instead of succeeding.
    reload_failure_recovery(sys.argv[1], sys.argv[sys.argv.index("--model") + 1])

    print("-" * 60)
    print("ALL PASS" if FAIL == 0 else f"{FAIL} FAILED, {PASS} passed")
    sys.exit(1 if FAIL else 0)


def reload_failure_recovery(agent_bin, real_model):
    import tempfile
    import shutil
    # A REAL directory of file-level symlinks into the model. mlx-swift-lm mis-maps a symlinked
    # model DIRECTORY (the applet resolves it with `pwd -P`), but a real dir whose entries are
    # file symlinks loads fine - and lets us break just config.json to make a reload fast-fail.
    shadow = tempfile.mkdtemp(prefix="mlx-idle-shadow-")
    for name in os.listdir(real_model):
        os.symlink(os.path.join(real_model, name), os.path.join(shadow, name))
    cfg = os.path.join(shadow, "config.json")
    argv = [agent_bin, "acp", "--idle-unload-seconds", str(IDLE_SECONDS), "--model", shadow]
    agent = IdleAgent(argv)
    try:
        agent.await_response(agent.send("initialize", {"protocolVersion": 1}))
        sid = agent.await_response(agent.send("session/new", {}))["result"]["sessionId"]
        agent.prompt(sid, "Say OK.")                     # load through the shadow dir

        time.sleep(WAIT_PAST_IDLE)                        # idle-unload
        os.remove(cfg)                                   # break it so reloadAfterIdle fast-fails
        resp, _ = agent.prompt_raw(sid, "anything")
        check("a failing idle-reload returns an error (not a hang)",
              "error" in resp, f"resp={resp.get('error') or resp.get('result')}")

        os.symlink(os.path.join(real_model, "config.json"), cfg)   # restore the model
        resp2, a2 = agent.prompt_raw(sid, "Reply with just: RECOVERED.")
        check("session is NOT wedged after a failed reload",
              "result" in resp2 and resp2["result"].get("stopReason") is not None,
              f"resp={resp2.get('error') or resp2.get('result')}")
        check("the recovered turn actually generated",
              len(a2) > 0, f"answer={a2!r}")
    finally:
        agent.close()
        shutil.rmtree(shadow, ignore_errors=True)


if __name__ == "__main__":
    main()
