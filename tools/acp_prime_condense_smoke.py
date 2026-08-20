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
# WHICH model summarizes is a separate axis, and the contract above is the same on all of them:
# either a valid digest, or the full history with a reason. So this script does not branch on the
# engine. `--expect` states which of the two outcomes the caller is testing for, and it is what
# turns "the session's own model can summarize" into an assertion rather than a hope.
#
# That axis has two sources - `--digest-backend` at launch and `condense.backend` per restore - and
# `backend_override_checks` below asserts the precedence between them. Those checks all resolve to
# refusals, so they cost process launches and no generation.
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
        # STILL RUN THESE. They build their own agents and assert refusals, so they neither need
        # nor care whether the prime above condensed - and this is the branch a machine that
        # cannot summarize always takes, which is exactly where a summarizer-selection bug would
        # go unnoticed. Returning here used to skip them and print ALL PASS.
        print("-" * 60)
        print("condense.backend: the request against the launch flag")
        backend_override_checks(binary, backend_args, check)
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

    # 5. Who summarizes, asked per restore rather than per process.
    print("-" * 60)
    print("condense.backend: the request against the launch flag")
    backend_override_checks(binary, backend_args, check)

    # 6. Concurrency. A condensing prime takes SECONDS, which is a wide-open window the rest of
    #    this file never looks at - and a review found a real leak in it: a session/new during a
    #    condensation was overwritten when the condensation published, so a conversation the user
    #    had discarded came back and the model quoted from it.
    print("-" * 60)
    print("concurrency: requests sent DURING a condensation")
    concurrency_checks(binary, backend_args, check)


    print("-" * 60)
    print("ALL PASS" if not failures else f"{len(failures)} FAILED: " + ", ".join(failures))
    return len(failures)


def without_digest_backend(args):
    """`args` with any `--digest-backend <x>` removed, so this file can state the flag itself."""
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a == "--digest-backend":
            skip = True
            continue
        out.append(a)
    return out


def backend_override_checks(binary, backend_args, check):
    """`condense.backend` names the summarizer for ONE restore; `--digest-backend` is the default.

    Every check here resolves BEFORE any model runs - each one is a refusal, and a refusal is
    decided by `chooseSummarizer` before the planner is called - so this section costs a few
    process launches and no generation whatever engine it is pointed at. Keep it that way: the
    obvious way to test an ACCEPTED value is to let one through, and on a machine with Apple
    Intelligence that quietly turns a structural assertion into a real summarization.

    What is being proved is precedence, and the reasons are asserted rather than just the outcome:
    a decline with the wrong reason means the request was read as something else, which is exactly
    the failure a client cannot see. The clause that matters most is `--digest-backend none`: it is
    the operator turning summarization off, and a request must not switch it back on.
    """
    base = without_digest_backend(backend_args)
    history = long_history(pairs=8)

    def prime(flag, requested):
        agent = Agent([binary, "acp", "--digest-backend", flag] + base)
        sid = agent.start_session()
        ask = {"keepRecentTurns": KEEP_RECENT}
        if requested is not None:
            ask["backend"] = requested
        r = agent.await_response(agent.send("session/prime", {
            "sessionId": sid, "messages": history, "condense": ask}), timeout=120)
        agent.p.kill()
        return r.get("result") or {}

    res = prime("auto", "none")
    reason = str(res.get("reason", ""))
    check("a request for `none` overrides a permissive launch flag",
          res.get("condensed") is False and "no summarizer" in reason, f"result={res}")
    check("and overriding still primes the FULL history",
          res.get("primed") == len(history), f"primed={res.get('primed')} of {len(history)}")

    res = prime("none", "session")
    reason = str(res.get("reason", ""))
    check("--digest-backend none is NOT overridable by a request",
          res.get("condensed") is False and "disabled" in reason, f"result={res}")
    check("and the refusal says the requested summarizer was not used",
          "session" in reason, f"reason={reason}")

    res = prime("auto", "wizard")
    reason = str(res.get("reason", ""))
    check("an unknown summarizer declines rather than falling back to the flag",
          res.get("condensed") is False and "wizard" in reason, f"result={res}")

    # Asked under `none` rather than under a permissive flag: the refusal is decided before any
    # model runs, and it distinguishes the two readings of an empty string. Absent gives the plain
    # "disabled" reason; parsed as a value it would give either "unknown summarizer" or the
    # "was not used" clause that names a request.
    res = prime("none", "")
    reason = str(res.get("reason", ""))
    check("an empty backend is absent, not a summarizer named \"\"",
          res.get("condensed") is False and "disabled" in reason
          and "unknown summarizer" not in reason and "was not used" not in reason,
          f"result={res}")

    # A value of the wrong TYPE is a client bug, not a preference. Reading it as "absent" would
    # summarize with the launch flag's model and report only that name - the same silent
    # substitution the string case refuses. Named by TYPE: a JSON boolean arrives as a number, so
    # echoing the value reports a summarizer called "1", which the client never sent.
    res = prime("auto", 42)
    reason = str(res.get("reason", ""))
    check("a backend that is not a string is refused, not ignored",
          res.get("condensed") is False and "must be a string" in reason and "a number" in reason,
          f"result={res}")

    res = prime("auto", True)
    check("  and a boolean is named as one rather than echoed as a value",
          "a boolean" in str(res.get("reason", "")), f"result={res}")

    # The value is echoed into `reason` and into the log, so it needs a bound - and a bound is
    # only worth having if it is asserted on the input that breaks it. A cap measured in Swift
    # `Character`s bounds GRAPHEMES: one base letter plus 20 000 combining marks is a single
    # Character, so an ASCII-only assertion here passes while 40 KB goes out on the wire.
    res = prime("auto", "a" + "\u0301" * 20000)
    reason = str(res.get("reason", ""))
    check("  and an overlong value is capped rather than echoed whole",
          res.get("condensed") is False and len(reason) < 200,
          f"reason is {len(reason)} chars")
    res = prime("auto", "x" * 500)
    check("  whatever it is made of",
          len(str(res.get("reason", ""))) < 200 and "..." in str(res.get("reason", "")),
          f"result={res}")

    # Control characters are not whitespace by Unicode's definition, so a value carrying terminal
    # escapes would reach a terminal reading the log intact.
    res = prime("auto", "\u001b[2J\u001b[1;31mPWNED\u001b[0m")
    check("  and terminal escapes do not survive into the reason",
          "\u001b" not in str(res.get("reason", "")), f"result={res}")

    # A value with nothing displayable in it is the same class of non-answer as whitespace, and is
    # read the same way. Asked under `none` so the reason distinguishes "absent" from "a value I
    # could not read", and so it costs no generation.
    res = prime("none", "\u001b\u001b")
    reason = str(res.get("reason", ""))
    check("  and a value made only of controls is absent, not a summarizer named \"\"",
          "unknown summarizer" not in reason and "was not used" not in reason and "disabled" in reason,
          f"result={res}")

    # The positive direction needs a request that reaches a DIFFERENT summarizer than the flag
    # would have, and identifies itself without generating. The foundation engine has exactly one:
    # `session` there is refused with a reason no other path produces, so seeing it proves the
    # request - not the flag - chose the summarizer.
    if "foundation" in base:
        res = prime("foundation", "session")
        reason = str(res.get("reason", ""))
        check("a request selects a different summarizer than the flag would have",
              res.get("condensed") is False and "session's model IS the on-device model" in reason,
              f"result={res}")


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
