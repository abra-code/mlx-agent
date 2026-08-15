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

def content_text(content):
    """Text out of an ACP content value: one ContentBlock dict, or an ARRAY of
    tool-call content entries ({"type":"content","content":{"type":"text",...}}) -
    tool_call_update carries the array shape."""
    if isinstance(content, dict):
        return content.get("text", "")
    if isinstance(content, list):
        out = ""
        for block in content:
            if not isinstance(block, dict):
                continue
            inner = block.get("content")
            if isinstance(inner, dict):
                out += inner.get("text", "")
            else:
                out += block.get("text", "")
        return out
    return ""

def collect_updates(store):
    def handler(msg):
        if msg.get("method") != "session/update":
            return
        up = msg.get("params", {}).get("update", {})
        kind = up.get("sessionUpdate")
        # usage_update carries token counters, not content: keep the whole update.
        if kind == "usage_update":
            store["_usage"] = up
            return
        store.setdefault(kind, "")
        store[kind] += content_text(up.get("content"))
    return handler

def main():
    if len(sys.argv) < 2:
        print("usage: acp_smoke.py /path/to/mlx-agent [--model <dir>]")
        print("       acp_smoke.py /path/to/mlx-agent --backend openai --base-url <url>")
        sys.exit(2)
    passthrough = sys.argv[2:]
    argv = [sys.argv[1], "acp"] + passthrough
    # Neither backend advertises anything in configOptions (see ACPServer.configOptionsJSON), so
    # that expectation no longer varies. This flag remains for the one place the backends really
    # do differ: the openai backend REFUSES a model switch outright (check 11), because there the
    # host app owns llama-server, whereas the MLX path can still switch on the wire.
    openai = "openai" in passthrough
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
        caps = r.get("result", {}).get("agentCapabilities", {})
        check("initialize advertises sessionPrime", caps.get("sessionPrime") is True,
              f"agentCapabilities={sorted(caps)}")

        # 2. session/new (loads the model)
        rid = agent.send("session/new", {"cwd": "/tmp", "mcpServers": []})
        r = agent.await_response(rid, timeout=180)
        res = r.get("result", {})
        sid = res.get("sessionId")
        opts = res.get("configOptions", [])
        check("session/new", isinstance(sid, str) and bool(sid), f"sessionId={sid}")
        # configOptions is empty on BOTH backends: nothing is advertised, so no client renders a
        # picker. Asserting the whole list is empty (not just "no model option") is deliberate -
        # it fails if anything is ever advertised again without a decision to do so.
        check("configOptions is empty (no client-rendered picker)",
              opts == [], f"ids={[o.get('id') for o in opts]}")

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
        usage = store.get("_usage") or {}
        check("session/prompt reports usage", isinstance(usage.get("used"), int)
              and usage["used"] > 0, f"usage={usage}")
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

        # 6. session/prime - resume: replay a fabricated transcript, then require recall.
        rid = agent.send("session/prime", {
            "sessionId": sid,
            "messages": [
                {"role": "user", "content": "My favorite color is teal. Please remember it."},
                {"role": "assistant", "content": "Understood: your favorite color is teal."},
            ],
        })
        r = agent.await_response(rid, timeout=30)
        check("session/prime accepts a transcript", r.get("result", {}).get("primed") == 2,
              f"result={r.get('result') or r.get('error')}")
        store4 = {}
        rid = agent.send("session/prompt", {
            "sessionId": sid,
            "prompt": [{"type": "text",
                        "text": "What is my favorite color? Answer in one short sentence."}],
        })
        r = agent.await_response(rid, on_update=collect_updates(store4), timeout=180)
        msg = store4.get("agent_message_chunk", "")
        check("primed context is recalled", "teal" in msg.lower(),
              f"message={msg.strip()[:80]!r}")

        # 7. session/prime - reset: empty messages must clear the context and the next
        # turn must still run cleanly. (No recall assertion - a model may guess or
        # refuse; the reset contract is "the call succeeds and the session works".)
        rid = agent.send("session/prime", {"sessionId": sid, "messages": []})
        r = agent.await_response(rid, timeout=30)
        check("session/prime empty resets", r.get("result", {}).get("primed") == 0,
              f"result={r.get('result') or r.get('error')}")
        store5 = {}
        rid = agent.send("session/prompt", {
            "sessionId": sid,
            "prompt": [{"type": "text", "text": "In one short sentence, who wrote Hamlet?"}],
        })
        r = agent.await_response(rid, on_update=collect_updates(store5), timeout=180)
        check("post-reset prompt completes", r.get("result", {}).get("stopReason") == "end_turn",
              f"message={store5.get('agent_message_chunk','').strip()[:60]!r}")

        # 8. session/prime - tool-turn fidelity: replay an assistant tool call + result
        # (and one orphan tool message that must be dropped), then ask about the result.
        rid = agent.send("session/prime", {
            "sessionId": sid,
            "messages": [
                {"role": "tool", "content": "orphan - must be dropped", "toolCallId": "tc-none"},
                {"role": "user", "content": "Read the file /tmp/status.txt and tell me what it says."},
                {"role": "assistant", "content": "",
                 "toolCalls": [{"id": "tc-1", "name": "read_file",
                                "arguments": {"path": "/tmp/status.txt"}}]},
                {"role": "tool", "content": "{\"content\": \"DEPLOY CODE: ZEBRA-7\"}",
                 "toolCallId": "tc-1"},
                {"role": "assistant",
                 "content": "The file contains the deploy code ZEBRA-7."},
            ],
        })
        r = agent.await_response(rid, timeout=30)
        check("session/prime tool turns accepted (orphan dropped)",
              r.get("result", {}).get("primed") == 4,
              f"result={r.get('result') or r.get('error')}")
        store6 = {}
        rid = agent.send("session/prompt", {
            "sessionId": sid,
            "prompt": [{"type": "text",
                        "text": "What deploy code did the file contain? Answer in one short sentence."}],
        })
        r = agent.await_response(rid, on_update=collect_updates(store6), timeout=180)
        msg = store6.get("agent_message_chunk", "")
        check("primed tool result is recalled", "zebra" in msg.lower(),
              f"message={msg.strip()[:80]!r}")

        # 8b. A TRAILING unanswered tool-call announcement is stripped (template guard):
        # only the user message survives.
        rid = agent.send("session/prime", {
            "sessionId": sid,
            "messages": [
                {"role": "user", "content": "Check the weather in Paris."},
                {"role": "assistant", "content": "",
                 "toolCalls": [{"id": "tc-9", "name": "get_weather", "arguments": {}}]},
            ],
        })
        r = agent.await_response(rid, timeout=30)
        check("trailing unanswered tool call is stripped", r.get("result", {}).get("primed") == 1,
              f"result={r.get('result') or r.get('error')}")

        # 9. session/prime error paths: wrong session id.
        rid = agent.send("session/prime", {"sessionId": "no-such-session", "messages": []})
        r = agent.await_response(rid, timeout=30)
        check("session/prime rejects unknown session", r.get("error", {}).get("code") == -32602,
              f"error={r.get('error')}")

        # 10. Priming UNDER a running turn must be refused rather than swapping the stack
        # out from under it. Both requests are dispatched from the same stdin reader, so
        # the prime provably arrives while the prompt task is live.
        prompt_id = agent.send("session/prompt", {
            "sessionId": sid,
            "prompt": [{"type": "text",
                        "text": "Count slowly from 1 to 300, one number per line."}],
        })
        prime_id = agent.send("session/prime", {
            "sessionId": sid,
            "messages": [{"role": "user", "content": "too late"}],
        })
        r = agent.await_response(prime_id, timeout=60)
        check("session/prime is refused during a prompt", r.get("error", {}).get("code") == -32003,
              f"error={r.get('error')}")
        agent.send("session/cancel", {"sessionId": sid}, notify=True)
        agent.await_response(prompt_id, timeout=60)  # drain the turn before moving on

        # 11. The model belongs to whoever runs llama-server on the openai backend, and it
        # changes by restarting the server - so a model switch here must be refused, not
        # silently accepted.
        if openai:
            rid = agent.send("session/set_config_option",
                             {"sessionId": sid, "configId": "model", "value": "anything"})
            r = agent.await_response(rid, timeout=30)
            check("model switch refused on the openai backend",
                  r.get("error", {}).get("code") == -32601, f"error={r.get('error')}")

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
