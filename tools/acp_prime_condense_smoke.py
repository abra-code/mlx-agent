#!/usr/bin/env python3
# session/prime with `condense`: the wire contract, asserted on structure only.
#
# Nothing here checks summary QUALITY - it is nondeterministic and a content assert would be a
# flaky lie. What is deterministic is the contract, and every clause of it is a promise a client
# will build on:
#
#   - a prime WITHOUT `condense` behaves exactly as before (no new keys, still near-instant)
#   - a prime WITH `condense` returns condensed/summarizer/digest/dropped, and `primed` counts the
#     REPLACEMENT history, not the input
#   - the digest parses as schemaVersion 1
#   - the arithmetic closes: primed == injected + (accepted - dropped.turns), where
#     `accepted` is the count AFTER primeHistory's well-formedness filtering - the client's own
#     message count does not close it on a transcript with orphan tool turns or empty content
#   - a summarizer that cannot run FALLS BACK to the full history - condensed:false plus a reason,
#     and `primed` equal to what was sent. This is the clause that matters most: the failure mode
#     it guards against is silent history loss.
#   - --digest-dir writes an artifact holding the digest AND the text it was made from
#
# WHICH model summarizes is a separate axis (--digest-backend), and the contract above is the same
# on all of them: either a valid digest, or the full history with a reason. So this script does not
# branch on the engine. `--expect` states which of the two outcomes the caller is testing for, and
# it is what turns "the session's own model can summarize" into an assertion rather than a hope.
#
# Usage: python3 tools/acp_prime_condense_smoke.py /path/to/mlx-agent \
#            [--expect condensed|declined|any] [--backend foundation | --model <dir>] [acp args]
#
#   python3 tools/acp_prime_condense_smoke.py "$BIN" --backend foundation --expect condensed
#   python3 tools/acp_prime_condense_smoke.py "$BIN" --model "$M" --digest-backend session --expect condensed
#   python3 tools/acp_prime_condense_smoke.py "$BIN" --model "$M" --digest-backend none --expect declined
#
# Exit code = number of failed checks (0 = all good).

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time

KEEP_RECENT = 6


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
            # The model's answer arrives as notifications, not in the prompt response.
            if msg.get("method") == "session/update":
                up = msg.get("params", {}).get("update", {})
                if up.get("sessionUpdate") in ("agent_message_chunk", "agent_thought_chunk"):
                    content = up.get("content") or {}
                    if isinstance(content, dict):
                        self.streamed += content.get("text", "")
            self.q.put(msg)

    def _errReader(self):
        for raw in self.p.stderr:
            self.err.append(raw.decode("utf-8", "replace").rstrip())

    def send(self, method, params):
        self.n += 1
        self.p.stdin.write(
            (json.dumps({"jsonrpc": "2.0", "id": self.n, "method": method,
                         "params": params}) + "\n").encode())
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

    def start_session(self):
        self.await_response(self.send("initialize", {"protocolVersion": 1}))
        r = self.await_response(self.send("session/new", {"cwd": "/tmp"}))
        return (r.get("result") or {}).get("sessionId")

    def alive(self):
        return self.p.poll() is None


def long_history(pairs=34):
    """Big enough that the older half is genuinely worth summarizing."""
    filler = (
        "We went through the installer layout, the component plist, the postinstall ordering, "
        "and how the receipt is validated on the next launch. ")
    msgs = [
        {"role": "user", "content": "The project codename is QUETZAL-9. Remember it."},
        {"role": "assistant", "content": "Noted, the codename is QUETZAL-9."},
    ]
    for i in range(pairs):
        msgs.append({"role": "user", "content": f"Step {i}: what about the installer? " + filler})
        msgs.append({"role": "assistant", "content": f"Step {i} is unchanged. " + filler})
    return msgs


def main():
    if len(sys.argv) < 2:
        print("usage: acp_prime_condense_smoke.py /path/to/mlx-agent [backend args]")
        return 2
    binary, backend_args = sys.argv[1], sys.argv[2:]
    # --expect is ours; everything else is passed through to `mlx-agent acp` untouched.
    expect = "any"
    if "--expect" in backend_args:
        i = backend_args.index("--expect")
        if i + 1 >= len(backend_args):
            print("--expect requires condensed|declined|any")
            return 2
        expect = backend_args[i + 1]
        del backend_args[i:i + 2]
        if expect not in ("condensed", "declined", "any"):
            print(f"unknown --expect {expect}")
            return 2
    failures = []

    def check(name, ok, detail=""):
        print(("[PASS] " if ok else "[FAIL] ") + name + ("" if ok else f"  {detail}"))
        if not ok:
            failures.append(name)

    archive = tempfile.mkdtemp(prefix="digest-smoke-")
    agent = Agent([binary, "acp", "--digest-dir", archive] + backend_args)
    sid = agent.start_session()
    if not sid:
        print("[FAIL] session/new")
        return 1
    want_session = "session" in backend_args and "--digest-backend" in backend_args

    # 1. A plain prime must be untouched by this feature.
    history = long_history()
    r = agent.await_response(agent.send(
        "session/prime", {"sessionId": sid, "messages": history}), timeout=60)
    plain = r.get("result") or {}
    check("a prime without condense returns only `primed`",
          plain.get("primed") == len(history) and "condensed" not in plain,
          f"result={plain}")

    # 2. The condensing prime.
    started = time.time()
    r = agent.await_response(agent.send("session/prime", {
        "sessionId": sid,
        "messages": history,
        "condense": {"keepRecentTurns": KEEP_RECENT, "maxDigestTokens": 700},
    }), timeout=300)
    res = r.get("result") or {}
    elapsed = time.time() - started
    print(f"       condensing prime took {elapsed:.1f} s")

    condensed = res.get("condensed") is True
    if expect != "any":
        check(f"the condensing prime {expect}", condensed == (expect == "condensed"),
              f"result={res}")

    if not condensed:
        # The fallback is the clause that matters most: a summarizer that cannot run must prime
        # EVERYTHING and say so. Silent truncation here would be the worst possible outcome.
        check("declining primes the FULL history", res.get("primed") == len(history),
              f"primed={res.get('primed')} of {len(history)}")
        check("declining says why", bool(res.get("reason")), f"result={res}")
        agent.p.kill()
        shutil.rmtree(archive, ignore_errors=True)
        print("-" * 60)
        print("ALL PASS" if not failures else f"{len(failures)} FAILED: " + ", ".join(failures))
        return len(failures)

    check("it reports which model summarized", bool(res.get("summarizer")), f"result={res}")
    if want_session:
        # --digest-backend session must not quietly resolve to the on-device model: the whole
        # point of the flag is to pin WHICH model reads the conversation.
        check("--digest-backend session summarized with the session's model",
              str(res.get("summarizer", "")).startswith("session:"),
              f"summarizer={res.get('summarizer')}")

    digest = res.get("digest") or {}
    check("the digest parses as schemaVersion 1", digest.get("schemaVersion") == 1,
          f"digest keys={sorted(digest.keys())}")
    check("the digest carries provenance",
          bool(digest.get("sourceSHA256")) and digest.get("sourceTurnCount", 0) > 0,
          f"sha={digest.get('sourceSHA256')} turns={digest.get('sourceTurnCount')}")

    dropped = res.get("dropped") or {}
    primed, turns = res.get("primed", 0), dropped.get("turns", 0)
    accepted = res.get("accepted")
    check("something was actually dropped", turns > KEEP_RECENT, f"dropped={dropped}")
    check("the response reports the accepted (post-filter) input count", accepted is not None,
          f"result keys={sorted(res.keys())}")
    # primed = the turns the planner injected + whatever it kept verbatim. Against `accepted`,
    # NOT against what the client sent: primeHistory filters, and the difference is exactly what
    # makes a client's own arithmetic go negative on a real transcript.
    injected = primed - ((accepted or 0) - turns)
    check("the counts close", 1 <= injected <= 3 and primed < (accepted or 0),
          f"primed={primed} accepted={accepted} dropped={turns} -> injected={injected}")

    # A transcript with the shapes primeHistory legitimately drops. The identity must still hold.
    dirty = long_history(pairs=34)
    dirty.insert(0, {"role": "tool", "content": "orphan", "toolCallId": "tc-none"})
    dirty.insert(0, {"role": "user", "content": ""})
    dirty.insert(0, {"role": "wizard", "content": "unknown role"})
    r2 = agent.await_response(agent.send("session/prime", {
        "sessionId": sid, "messages": dirty,
        "condense": {"keepRecentTurns": KEEP_RECENT},
    }), timeout=300)
    d = r2.get("result") or {}
    if d.get("condensed"):
        inj = d.get("primed", 0) - (d.get("accepted", 0) - (d.get("dropped") or {}).get("turns", 0))
        check("the identity holds on a transcript with filtered turns", 1 <= inj <= 3,
              f"primed={d.get('primed')} accepted={d.get('accepted')} "
              f"sent={len(dirty)} dropped={(d.get('dropped') or {}).get('turns')} -> injected={inj}")
        check("accepted is smaller than what was sent", d.get("accepted", 0) < len(dirty),
              f"accepted={d.get('accepted')} sent={len(dirty)}")

    # 3. The session must actually work afterward.
    rid = agent.send("session/prompt", {
        "sessionId": sid,
        "prompt": [{"type": "text", "text": "What is the project codename? One line."}],
    })
    p = agent.await_response(rid, timeout=180)
    check("a prompt works against the condensed context",
          agent.alive() and (p.get("result") or {}).get("stopReason") == "end_turn",
          f"stopReason={(p.get('result') or {}).get('stopReason')} alive={agent.alive()}")

    # 4. The artifact.
    files = sorted(os.listdir(archive))
    art = [f for f in files if f.startswith("digest-") and f.endswith(".json")]
    check("--digest-dir wrote an artifact", len(art) >= 1, f"dir contains {files}")
    check("--digest-dir wrote an index", "digests.jsonl" in files, f"dir contains {files}")
    if art:
        with open(os.path.join(archive, art[0])) as f:
            payload = json.load(f)
        check("the artifact holds the digest AND its source text",
              payload.get("digest", {}).get("schemaVersion") == 1
              and len(payload.get("source", "")) > 500,
              f"keys={sorted(payload.keys())} source={len(payload.get('source', ''))} bytes")
        check("the artifact's hash matches its source",
              payload.get("digest", {}).get("sourceSHA256") ==
              __import__("hashlib").sha256(payload.get("source", "").encode()).hexdigest(),
              "sourceSHA256 does not cover the recorded source")

    agent.p.kill()
    shutil.rmtree(archive, ignore_errors=True)

    # 5. Concurrency. A condensing prime takes SECONDS, which is a wide-open window the rest of
    #    this file never looks at - and a review found a real leak in it: a session/new during a
    #    condensation was overwritten when the condensation published, so a conversation the user
    #    had discarded came back and the model quoted from it.
    print("-" * 60)
    print("concurrency: requests sent DURING a condensation")
    concurrency_checks(binary, backend_args, check)


    print("-" * 60)
    print("ALL PASS" if not failures else f"{len(failures)} FAILED: " + ", ".join(failures))
    return len(failures)


SECRET = "ZEBRA-77"


def concurrency_checks(binary, backend_args, check):
    """Send a competing request mid-condensation and assert the discarded context stays discarded.

    Reached only when the earlier condensation actually ran - on a configuration that declines, the
    prime returns instantly and there is no window to race.
    """

    def secret_history():
        h = long_history(pairs=34)
        h.append({"role": "user", "content": f"The access token is {SECRET}. Keep it."})
        h.append({"role": "assistant", "content": f"Understood, the access token is {SECRET}."})
        return h

    for label, supersede in (
        ("session/new", lambda a, s: a.send("session/new", {"cwd": "/tmp"})),
        ("a plain reset prime",
         lambda a, s: a.send("session/prime", {"sessionId": s, "messages": []})),
    ):
        agent = Agent([binary, "acp"] + backend_args)
        sid = agent.start_session()
        rid = agent.send("session/prime", {
            "sessionId": sid,
            "messages": secret_history(),
            "condense": {"keepRecentTurns": KEEP_RECENT},
        })
        time.sleep(1.0)                      # let the condensation get under way
        other = supersede(agent, sid)
        agent.await_response(other, timeout=60)
        agent.await_response(rid, timeout=300)

        # The client asked for a fresh/empty context. The secret must be gone.
        pid = agent.send("session/prompt", {
            "sessionId": sid,
            "prompt": [{"type": "text", "text": "What is the access token? Answer in one line."}],
        })
        agent.await_response(pid, timeout=180)
        leaked = SECRET.lower() in agent.streamed.lower()
        check(f"{label} during a condensation is not overwritten by it", not leaked,
              f"the discarded conversation's secret {SECRET} came back: "
              f"{agent.streamed.strip()[:120]}")
        agent.p.kill()

    # A cancel must actually stop it, rather than the slot being held for minutes.
    agent = Agent([binary, "acp"] + backend_args)
    sid = agent.start_session()
    rid = agent.send("session/prime", {
        "sessionId": sid, "messages": long_history(pairs=34),
        "condense": {"keepRecentTurns": KEEP_RECENT},
    })
    time.sleep(1.0)
    agent.send("session/cancel", {"sessionId": sid})
    started = time.time()
    r = agent.await_response(rid, timeout=120)
    elapsed = time.time() - started
    res = r.get("result") or {}
    check("session/cancel reaches an in-flight condensation",
          r != {} and elapsed < 30 and res.get("condensed") is not True,
          f"took {elapsed:.1f} s after cancel, result={res or r.get('error')}")
    check("a canceled condensation still primes the full history",
          res.get("condensed") is False and res.get("primed", 0) > KEEP_RECENT,
          f"result={res}")
    agent.p.kill()


if __name__ == "__main__":
    sys.exit(main())
