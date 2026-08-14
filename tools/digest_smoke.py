#!/usr/bin/env python3
# `mlx-agent digest`: the batch contract, asserted on structure only.
#
# Nothing here checks summary QUALITY - it is nondeterministic and a content assert would be a
# flaky lie. What is deterministic is the contract, and this mode's contract is what a script or a
# Makefile builds on:
#
#   - a condensed run exits 0, writes ONE JSON document on stdout, and the digest is schemaVersion 1
#   - the tail-count arithmetic closes: primed == injected + (accepted - dropped.turns), where
#     `accepted` is the count AFTER primeHistory's filtering. Same identity as session/prime, and
#     the same trap: a caller's own message count does not close it on a real transcript
#   - --render prints the preamble that would actually be primed, not the JSON
#   - --out writes the artifact to a file and leaves stdout clean
#   - a transcript too short to be worth condensing exits 3 with a reason - a fallback, NOT a crash,
#     and the distinction is the whole reason 3 exists separately from 1
#   - unreadable input exits 2 and generates nothing
#   - RESULT_JSON goes to STDERR, so `digest | jq` works without filtering
#
# WHICH model summarizes does not change any of the above, so this script does not branch on the
# engine - pass whichever one you want tested.
#
# Usage: python3 tools/digest_smoke.py /path/to/mlx-agent [--backend foundation | --model <dir>]
#
#   python3 tools/digest_smoke.py "$BIN" --backend foundation
#   python3 tools/digest_smoke.py "$BIN" --model "$M"
#
# Exit code = number of failed checks (0 = all good).

import json
import os
import shutil
import subprocess
import sys
import tempfile

KEEP_RECENT = 6
TIMEOUT = 900  # a local model summarizing several slices; not a latency assertion

PREAMBLE_OPENING = "Context restored from an earlier conversation."


def long_history(pairs=18):
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
        print("usage: digest_smoke.py /path/to/mlx-agent [backend args]")
        return 2
    binary, backend_args = sys.argv[1], sys.argv[2:]
    failures = []

    def check(name, ok, detail=""):
        print(("[PASS] " if ok else "[FAIL] ") + name + ("" if ok else f"  {detail}"))
        if not ok:
            failures.append(name)

    work = tempfile.mkdtemp(prefix="digest-mode-smoke-")

    def run(args, stdin_bytes=None):
        return subprocess.run(
            [binary, "digest"] + backend_args + args,
            input=stdin_bytes, capture_output=True, timeout=TIMEOUT)

    def write(name, obj):
        path = os.path.join(work, name)
        with open(path, "w") as f:
            json.dump(obj, f)
        return path

    try:
        history = long_history()
        transcript = write("transcript.json", history)

        # 1. The ordinary run: JSON on stdout, exit 0.
        r = run(["--in", transcript, "--keep-recent", str(KEEP_RECENT)])
        stderr = r.stderr.decode("utf-8", "replace")
        if r.returncode != 0:
            check("a long transcript condenses (exit 0)", False,
                  f"rc={r.returncode}\n{stderr[-2000:]}")
            print("-" * 60)
            print(f"{len(failures)} FAILED - cannot continue without a digest")
            return len(failures)
        check("a long transcript condenses (exit 0)", True)

        try:
            res = json.loads(r.stdout.decode("utf-8", "replace"))
        except json.JSONDecodeError as e:
            check("stdout is one JSON document", False, f"{e}: {r.stdout[:400]}")
            print("-" * 60)
            return len(failures) + 1
        check("stdout is one JSON document", True)
        check("it reports condensed:true", res.get("condensed") is True, f"result={res}")
        check("it reports which model summarized", bool(res.get("summarizer")), f"result={res}")

        digest = res.get("digest") or {}
        check("the digest parses as schemaVersion 1", digest.get("schemaVersion") == 1,
              f"digest keys={sorted(digest.keys())}")
        check("the digest carries provenance",
              bool(digest.get("sourceSHA256")) and digest.get("sourceTurnCount", 0) > 0,
              f"sha={digest.get('sourceSHA256')} turns={digest.get('sourceTurnCount')}")
        check("the digest names its generator", digest.get("generator") == res.get("summarizer"),
              f"generator={digest.get('generator')} summarizer={res.get('summarizer')}")

        dropped = res.get("dropped") or {}
        primed, turns = res.get("primed", 0), dropped.get("turns", 0)
        accepted = res.get("accepted")
        check("something was actually dropped", turns > KEEP_RECENT, f"dropped={dropped}")
        check("bytes dropped are reported", dropped.get("bytes", 0) > 0, f"dropped={dropped}")
        # primed = the turns the planner injected + whatever it kept verbatim. Against `accepted`,
        # never against what the caller sent - see the header.
        #
        # 1 or 2 EXACTLY: the digest preamble, plus an acknowledgment only when the verbatim tail
        # would otherwise put two user turns back to back. Not 3 - a hoisted system turn would land
        # in `injected` without counting toward `dropped.turns`, and this wire shape has no system
        # role at all (primeHistory drops it as an unknown role), so that case cannot arise here.
        injected = primed - ((accepted or 0) - turns)
        check("the counts close", 1 <= injected <= 2 and primed < (accepted or 0),
              f"primed={primed} accepted={accepted} dropped={turns} -> injected={injected}")

        # RESULT_JSON is on stderr precisely so the parse above needs no filtering.
        result_lines = [ln for ln in stderr.splitlines() if ln.startswith("RESULT_JSON: ")]
        check("a RESULT_JSON line goes to stderr", len(result_lines) == 1,
              f"found {len(result_lines)}")
        if result_lines:
            summary = json.loads(result_lines[0][len("RESULT_JSON: "):])
            check("RESULT_JSON agrees with the artifact",
                  summary.get("condensed") is True
                  and summary.get("primed") == primed
                  and summary.get("accepted") == accepted,
                  f"RESULT_JSON={summary}")

        # 2. --render: the text that actually gets primed.
        r = run(["--in", transcript, "--keep-recent", str(KEEP_RECENT), "--render"])
        rendered = r.stdout.decode("utf-8", "replace")
        check("--render exits 0", r.returncode == 0,
              f"rc={r.returncode}\n{r.stderr.decode('utf-8', 'replace')[-1000:]}")
        check("--render prints the preamble", PREAMBLE_OPENING in rendered,
              f"stdout starts: {rendered[:200]!r}")
        check("--render prints text, not JSON", not rendered.lstrip().startswith("{"),
              f"stdout starts: {rendered[:80]!r}")
        check("--render states the verbatim tail", "unmodified" in rendered,
              f"stdout starts: {rendered[:200]!r}")

        # 3. --out: the artifact goes to a file, stdout stays clean.
        out = os.path.join(work, "digest.json")
        r = run(["--in", transcript, "--out", out])
        check("--out exits 0", r.returncode == 0,
              f"rc={r.returncode}\n{r.stderr.decode('utf-8', 'replace')[-1000:]}")
        check("--out leaves stdout empty", r.stdout.strip() == b"", f"stdout={r.stdout[:200]!r}")
        wrote = os.path.exists(out)
        check("--out wrote the file", wrote)
        if wrote:
            with open(out) as f:
                saved = json.load(f)
            check("the written artifact is a digest",
                  (saved.get("digest") or {}).get("schemaVersion") == 1,
                  f"keys={sorted(saved.keys())}")

        # 4. stdin is the default input, and the params object form is accepted.
        r = run([], stdin_bytes=json.dumps({"messages": history}).encode())
        check("stdin + the session/prime params object form works", r.returncode == 0,
              f"rc={r.returncode}\n{r.stderr.decode('utf-8', 'replace')[-1000:]}")

        # 5. A transcript with the shapes primeHistory legitimately drops. The identity must hold
        #    against `accepted`, which must itself be smaller than what was sent.
        dirty = long_history()
        dirty.insert(0, {"role": "tool", "content": "orphan", "toolCallId": "tc-none"})
        dirty.insert(0, {"role": "user", "content": ""})
        dirty.insert(0, {"role": "wizard", "content": "unknown role"})
        r = run(["--in", write("dirty.json", dirty), "--keep-recent", str(KEEP_RECENT)])
        if r.returncode == 0:
            d = json.loads(r.stdout.decode("utf-8", "replace"))
            inj = d.get("primed", 0) - (d.get("accepted", 0) - (d.get("dropped") or {}).get("turns", 0))
            check("the identity holds on a transcript with filtered turns", 1 <= inj <= 2,
                  f"primed={d.get('primed')} accepted={d.get('accepted')} "
                  f"sent={len(dirty)} -> injected={inj}")
            check("accepted is smaller than what was sent", d.get("accepted", 0) < len(dirty),
                  f"accepted={d.get('accepted')} sent={len(dirty)}")
        else:
            check("a transcript with filtered turns still condenses", False, f"rc={r.returncode}")

        # 6. The fallback. Nothing old enough to summarize is an ANSWER, not a failure: exit 3,
        #    a reason on stderr, and a parseable condensed:false document on stdout.
        short = write("short.json", history[:4])
        r = run(["--in", short, "--keep-recent", "16"])
        stderr = r.stderr.decode("utf-8", "replace")
        check("too short to condense exits 3", r.returncode == 3,
              f"rc={r.returncode}\n{stderr[-1000:]}")
        check("the fallback says why on stderr", "not condensed" in stderr, f"stderr={stderr[-400:]}")
        try:
            fell = json.loads(r.stdout.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            fell = {}
        check("the fallback still writes a parseable document",
              fell.get("condensed") is False and bool(fell.get("reason")), f"stdout={fell}")
        check("the fallback reports the full history as primed",
              fell.get("primed") == fell.get("accepted"), f"stdout={fell}")

        # --render has no preamble to print when nothing was condensed, so it prints NOTHING and
        # still exits 3. The reason is on stderr either way - that is the invariant, not stdout.
        r = run(["--in", short, "--keep-recent", "16", "--render"])
        check("--render on a fallback exits 3 with empty stdout",
              r.returncode == 3 and r.stdout.strip() == b"",
              f"rc={r.returncode} stdout={r.stdout[:200]!r}")
        check("--render on a fallback still says why on stderr",
              "not condensed" in r.stderr.decode("utf-8", "replace"))

        # 7. Unreadable input is a usage error, and produces no artifact.
        r = run([], stdin_bytes=b"this is not json")
        check("non-JSON input exits 2", r.returncode == 2, f"rc={r.returncode}")
        check("non-JSON input writes nothing to stdout", r.stdout.strip() == b"",
              f"stdout={r.stdout[:200]!r}")

        r = run([], stdin_bytes=json.dumps({"turns": history}).encode())
        check("a JSON object without `messages` exits 2", r.returncode == 2, f"rc={r.returncode}")

        # 8. A flag that cannot take effect must be refused, not ignored. `option()` hands back
        #    whatever the next argv element is, so every one of these was reachable: the artifact
        #    written to a file named "--render", a trailing --in silently reading stdin, and a
        #    typo'd number running the default while the operator believed otherwise.
        r = run(["--in", transcript, "--keep-recent", "later"])
        check("a non-numeric --keep-recent exits 2", r.returncode == 2, f"rc={r.returncode}")

        r = run(["--in", transcript, "--out", "--render"])
        check("--out swallowing the next flag exits 2", r.returncode == 2, f"rc={r.returncode}")
        check("--out swallowing the next flag wrote no file named --render",
              not os.path.exists("--render") and not os.path.exists(os.path.join(work, "--render")))

        r = run(["--keep-recent", str(KEEP_RECENT), "--in"], stdin_bytes=b"[]")
        check("a trailing --in exits 2 rather than reading stdin", r.returncode == 2,
              f"rc={r.returncode}")

        r = run(["--in", transcript, "--digest-window", "0"])
        check("--digest-window 0 exits 2", r.returncode == 2, f"rc={r.returncode}")

        # 9. A write that cannot land is a failure (1), not a fallback (3) and not a success.
        r = run(["--in", transcript, "--out", os.path.join(work, "no-such-dir", "d.json")])
        check("an unwritable --out exits 1", r.returncode == 1, f"rc={r.returncode}")
        check("an unwritable --out reports the path",
              "cannot write" in r.stderr.decode("utf-8", "replace"),
              f"stderr={r.stderr.decode('utf-8', 'replace')[-300:]}")
        check("an unwritable --out emits no RESULT_JSON",
              "RESULT_JSON" not in r.stderr.decode("utf-8", "replace"))

    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("-" * 60)
    print("ALL PASS" if not failures else f"{len(failures)} FAILED: " + ", ".join(failures))
    return len(failures)


if __name__ == "__main__":
    sys.exit(main())
