#!/usr/bin/env python3
# ACP stdio smoke test for `mlx-agent acp` (Phase 1: chat, no tools).
#
# Spawns the agent, speaks newline-delimited JSON-RPC 2.0 over its stdin/stdout
# (the same framing ActionUIChat's ACPConnection uses), and exercises the Phase-1
# surface: initialize -> session/new -> session/prompt (with streamed
# session/update chunks) -> a second prompt (KV reuse) -> session/cancel.
#
# Usage: python3 tools/acp_smoke.py /path/to/mlx-agent [--model <dir>]
# Exit code = number of failed checks (0 = all good).

import json
import queue
import subprocess
import sys
import threading
import time

def reader(stream, q):
    for raw in stream:
        line = raw.decode("utf-8", "replace").rstrip("\n")
        if not line:
            continue
        try:
            q.put(json.loads(line))
        except json.JSONDecodeError:
            q.put({"_nonjson": line})

class Agent:
    def __init__(self, argv):
        self.p = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None)
        self.q = queue.Queue()
        threading.Thread(target=reader, args=(self.p.stdout, self.q), daemon=True).start()
        self._id = 0

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
        """Read messages until the response for want_id arrives; feed notifications to on_update."""
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

    def close(self):
        try:
            self.p.stdin.close()
        except Exception:
            pass
        try:
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()

def collect_updates(store):
    def handler(msg):
        if msg.get("method") != "session/update":
            return
        up = msg.get("params", {}).get("update", {})
        kind = up.get("sessionUpdate")
        text = (up.get("content") or {}).get("text", "")
        store.setdefault(kind, "")
        store[kind] += text
    return handler

def main():
    if len(sys.argv) < 2:
        print("usage: acp_smoke.py /path/to/mlx-agent [--model <dir>]")
        sys.exit(2)
    argv = [sys.argv[1], "acp"] + sys.argv[2:]
    agent = Agent(argv)
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
        if not ok:
            fails += 1

    try:
        # 1. initialize
        rid = agent.send("initialize", {
            "protocolVersion": 1,
            "clientCapabilities": {"fs": {"readTextFile": False, "writeTextFile": False}, "terminal": False},
            "clientInfo": {"name": "acp_smoke", "title": "smoke", "version": "1.0"},
        })
        r = agent.await_response(rid, timeout=30)
        pv = r.get("result", {}).get("protocolVersion")
        check("initialize", pv == 1, f"protocolVersion={pv}")

        # 2. session/new (loads the model)
        rid = agent.send("session/new", {"cwd": "/tmp", "mcpServers": []})
        r = agent.await_response(rid, timeout=180)
        res = r.get("result", {})
        sid = res.get("sessionId")
        opts = res.get("configOptions", [])
        model_opt = next((o for o in opts if o.get("id") == "model"), None)
        check("session/new", isinstance(sid, str) and bool(sid), f"sessionId={sid}")
        check("configOptions has model select",
              model_opt is not None and model_opt.get("type") == "select"
              and len(model_opt.get("options", [])) >= 1,
              f"current={model_opt.get('currentValue') if model_opt else None}")

        # 3. session/prompt - streamed
        store = {}
        rid = agent.send("session/prompt", {
            "sessionId": sid,
            "prompt": [{"type": "text", "text": "In one short sentence, what is the capital of France?"}],
        })
        r = agent.await_response(rid, on_update=collect_updates(store), timeout=180)
        stop = r.get("result", {}).get("stopReason")
        msg_text = store.get("agent_message_chunk", "")
        check("session/prompt streamed a message", len(msg_text.strip()) > 0,
              f"message={msg_text.strip()[:80]!r}")
        check("session/prompt stopReason", stop == "end_turn", f"stopReason={stop}")
        check("answer mentions Paris", "paris" in msg_text.lower(), "")
        if "agent_thought_chunk" in store:
            print(f"       (thought captured, {len(store['agent_thought_chunk'])} chars - folded reasoning path works)")

        # 4. second prompt (KV reuse across turns in the same session)
        store2 = {}
        t0 = time.time()
        rid = agent.send("session/prompt", {
            "sessionId": sid,
            "prompt": [{"type": "text", "text": "And its most famous landmark, in one sentence?"}],
        })
        r = agent.await_response(rid, on_update=collect_updates(store2), timeout=180)
        check("second prompt completes", r.get("result", {}).get("stopReason") == "end_turn",
              f"{time.time()-t0:.1f}s, message={store2.get('agent_message_chunk','').strip()[:80]!r}")

        # 5. cancel: start a guaranteed-long generation, cancel it early (mid-turn)
        store3 = {}
        t0 = time.time()
        rid = agent.send("session/prompt", {
            "sessionId": sid,
            "prompt": [{"type": "text",
                        "text": "Count slowly from 1 to 300. Put each number on its own line "
                                "with a one-word comment. Do not stop early."}],
        })
        time.sleep(1.0)  # generation is well under way but nowhere near done
        agent.send("session/cancel", {"sessionId": sid}, notify=True)
        r = agent.await_response(rid, on_update=collect_updates(store3), timeout=60)
        stop = r.get("result", {}).get("stopReason")
        check("session/cancel resolves the turn as cancelled", stop == "cancelled",
              f"stopReason={stop} after {time.time()-t0:.1f}s")

    except Exception as e:
        print(f"[FAIL] exception: {e!r}")
        fails += 1
    finally:
        agent.close()

    print("-" * 60)
    print(f"{'ALL PASS' if fails == 0 else str(fails) + ' FAILED'}")
    sys.exit(fails)

if __name__ == "__main__":
    main()
