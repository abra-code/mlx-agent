#!/usr/bin/env python3
# Map-reduce smoke test: a stitch job with a "reduce" template, against a REAL llama-server.
#
# Drives `map --backend openai` (same setup as map_openai_smoke.py) and proves the reduce
# contract through job.json in, status.json/result.txt out:
#   - a text that fits in one chunk goes through the REDUCE template only (one step);
#   - a text that needs several chunks is mapped chunk by chunk, then reduced once: the result is
#     the reduce template's output alone, result.txt never shows a map output, and status counts
#     the map steps plus the final reduce;
#   - a reduce job is rejected when it asks for collect output, or its template has no {{chunk}};
#   - a stop file during a reduce job cancels it.
#
# The templates ask for marker words (MAPPED, REDUCED) so the checks can tell which template
# produced the result without depending on what a model writes beyond that. Use an instruct model
# that follows "reply with exactly ..." (any Gemma 4 or Qwen3 instruct does).
#
# Usage:
#   python3 tools/map_reduce_smoke.py /path/to/mlx-agent --base-url http://127.0.0.1:8188/v1

import json
import os
import subprocess
import sys
import threading
import time

POLL_S = 0.1

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    if cond:
        PASS += 1
    else:
        FAIL += 1


class Broker:
    def __init__(self, argv):
        self.p = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.err_lines = []
        self._lock = threading.Lock()
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        for raw in self.p.stderr:
            line = raw.decode("utf-8", "replace").rstrip()
            with self._lock:
                self.err_lines.append(line)

    def log_lines(self):
        with self._lock:
            return list(self.err_lines)

    def stop(self):
        self.p.terminate()
        try:
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


def read_state(spool):
    try:
        with open(os.path.join(spool, "status.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def read_result(spool):
    try:
        with open(os.path.join(spool, "result.txt")) as f:
            return f.read()
    except OSError:
        return ""


def wait_any(spool, wants, timeout, epoch, watch=None):
    # watch: called on every poll with (status, result.txt) - lets a check see the result file
    # while the job runs, not only at the end.
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = read_state(spool)
        if watch is not None and st.get("epoch") == epoch:
            watch(st, read_result(spool))
        if st.get("state") in wants and st.get("epoch") == epoch:
            return st
        time.sleep(POLL_S)
    return read_state(spool)


def submit(spool, job):
    tmp = os.path.join(spool, "job.json.tmp")
    with open(tmp, "w") as f:
        json.dump(job, f)
    os.replace(tmp, os.path.join(spool, "job.json"))


MAP = [{"role": "user", "content": "Reply with exactly the word MAPPED and nothing else.\n\n{{chunk}}"}]
REDUCE = {"messages": [{"role": "user",
                        "content": "Reply with exactly the word REDUCED and nothing else.\n\n{{chunk}}"}]}
# About 60 short paragraphs: far over a 128-token budget, so the text splits into several chunks,
# while the joined map outputs ("MAPPED" per chunk) fit in one.
LONG = "\n\n".join(f"Paragraph {i}: the river rose and the town moved its archive uphill." for i in range(60))


def main():
    if len(sys.argv) < 4 or sys.argv[2] != "--base-url":
        print("usage: map_reduce_smoke.py /path/to/mlx-agent --base-url <url>")
        return 2
    agent_bin, base_url = sys.argv[1], sys.argv[3]
    spool = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"map_reduce_smoke_{os.getpid()}")
    os.makedirs(spool, exist_ok=True)

    broker = Broker([
        agent_bin, "map", "--backend", "openai", "--base-url", base_url, "--spool", spool,
        "--temperature", "0", "--max-new-tokens", "64",
    ])
    try:
        deadline = time.time() + 60
        while time.time() < deadline and read_state(spool).get("state") != "ready":
            time.sleep(POLL_S)
        st = read_state(spool)
        check("broker becomes ready", st.get("state") == "ready", str(st))

        # Job 1: fits in one chunk -> the reduce template alone, one step.
        submit(spool, {"epoch": 1, "text": "A short note about the river.", "budget_tokens": 1024,
                       "output": "stitch", "messages": MAP, "reduce": REDUCE})
        st = wait_any(spool, {"done", "error", "cancelled"}, 120, 1)
        out = read_result(spool)
        check("one-chunk reduce job done", st.get("state") == "done", str(st))
        check("its result comes from the reduce template", "REDUCED" in out and "MAPPED" not in out, repr(out[:60]))
        check("one step in all", st.get("chunk") == 1 and st.get("total") == 1, str(st))

        # Job 2: several chunks -> map each, then reduce once.
        shown = set()
        totals = set()

        def watch(status, result):
            if result.strip():
                shown.add(result.strip()[:40])
            if status.get("state") == "mapping" and isinstance(status.get("total"), int):
                totals.add(status["total"])

        submit(spool, {"epoch": 2, "text": LONG, "budget_tokens": 128,
                       "output": "stitch", "messages": MAP, "reduce": REDUCE})
        st = wait_any(spool, {"done", "error", "cancelled"}, 300, 2, watch)
        out = read_result(spool)
        check("multi-chunk reduce job done", st.get("state") == "done", str(st))
        check("its result is the reduce output alone", "REDUCED" in out and "MAPPED" not in out, repr(out[:60]))
        check("result.txt never showed a map output", not any("MAPPED" in s for s in shown), str(shown))
        steps = st.get("total") or 0
        check("status counts the map steps plus the reduce", steps >= 3 and st.get("chunk") == steps, str(st))
        check("the running total included the reduce step", steps in totals, f"seen {sorted(totals)}")
        rounds = [l for l in broker.log_lines() if "job 2: reduce round" in l]
        check("one map round, then the reduce", len(rounds) == 2 and "map over" in rounds[0]
              and "reduce over 1 chunk" in rounds[1], str(rounds))

        # Job 3: reduce with collect output is refused.
        submit(spool, {"epoch": 3, "text": "x", "output": "collect", "messages": MAP, "reduce": REDUCE})
        st = wait_any(spool, {"done", "error"}, 30, 3)
        check("reduce with collect is an error", st.get("state") == "error"
              and "stitch" in (st.get("message") or ""), str(st))

        # Job 4: a reduce template without the placeholder is refused.
        submit(spool, {"epoch": 4, "text": "x", "output": "stitch", "messages": MAP,
                       "reduce": {"messages": [{"role": "user", "content": "Say REDUCED."}]}})
        st = wait_any(spool, {"done", "error"}, 30, 4)
        check("reduce without {{chunk}} is an error", st.get("state") == "error"
              and "reduce" in (st.get("message") or ""), str(st))

        # Job 5: stop during the map phase of a reduce job cancels it.
        submit(spool, {"epoch": 5, "text": LONG, "budget_tokens": 128,
                       "output": "stitch", "messages": MAP, "reduce": REDUCE})
        st = wait_any(spool, {"mapping"}, 30, 5)
        open(os.path.join(spool, "stop"), "w").close()
        st = wait_any(spool, {"cancelled", "done", "error"}, 60, 5)
        check("a stop cancels a reduce job", st.get("state") == "cancelled", str(st))
    finally:
        broker.stop()
        subprocess.run(["/bin/rm", "-rf", spool], check=False)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
