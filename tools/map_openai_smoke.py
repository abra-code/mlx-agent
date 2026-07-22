#!/usr/bin/env python3
# Map-mode openai-backend smoke test.
#
# Drives `map --backend openai` against a REAL, already-running llama-server and proves the new
# engine serves the SAME spool protocol the MLX path does (map_idle_smoke.py): the broker becomes
# ready, a messages-mode job stitches output, a prompt-mode job (raw completion + add_special_tokens,
# via /tokenize then /completion) collects a per-chunk record, and a stop file cancels a job
# mid-generation - all through job.json in, status.json/result.txt/results.jsonl out.
#
# The llama-server is assumed to be launched separately (the applet owns it in production); pass its
# OpenAI-compatible base-url. Use a SMALL model - the checks only assert protocol behavior, not
# translation quality.
#
# Usage:
#   python3 tools/map_openai_smoke.py /path/to/mlx-agent --base-url http://127.0.0.1:8188/v1
#
# Every check asserts an OBSERVED effect:
#   - the broker reaches "ready" with no model loaded here (weights live in llama-server);
#   - a messages-mode job reaches "done" and stitches non-empty result.txt;
#   - a prompt-mode collect job reaches "done" and appends one results.jsonl record whose
#     source/output/sep round-trip the input;
#   - a stop file dropped mid-generation lands the job in "cancelled" (not done/error) promptly.

import json
import os
import subprocess
import sys
import threading
import time

POLL_S = 0.2

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


def wait_state(spool, want, timeout, seen=None, epoch=None):
    # epoch matters: after job N is done, status.json still says "done" for epoch N while job N+1
    # is being picked up - waiting on state alone returns instantly with stale data.
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = read_state(spool)
        if seen is not None and st.get("state") and (epoch is None or st.get("epoch") == epoch):
            seen.add(st["state"])
        if st.get("state") == want and (epoch is None or st.get("epoch") == epoch):
            return st
        time.sleep(POLL_S)
    return read_state(spool)


def wait_any(spool, wants, timeout, epoch=None):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = read_state(spool)
        if st.get("state") in wants and (epoch is None or st.get("epoch") == epoch):
            return st
        time.sleep(POLL_S)
    return read_state(spool)


def submit(spool, job):
    tmp = os.path.join(spool, "job.json.tmp")
    with open(tmp, "w") as f:
        json.dump(job, f)
    os.replace(tmp, os.path.join(spool, "job.json"))


def main():
    if len(sys.argv) < 4 or sys.argv[2] != "--base-url":
        print("usage: map_openai_smoke.py /path/to/mlx-agent --base-url <url>")
        return 2
    agent_bin, base_url = sys.argv[1], sys.argv[3]
    spool = os.path.join(
        os.environ.get("TMPDIR", "/tmp"), f"map_openai_smoke_{os.getpid()}")
    os.makedirs(spool, exist_ok=True)

    broker = Broker([
        agent_bin, "map", "--backend", "openai", "--base-url", base_url, "--spool", spool,
        "--temperature", "0", "--max-new-tokens", "800",
    ])
    try:
        st = wait_state(spool, "ready", 60)
        check("broker becomes ready (no local model)", st.get("state") == "ready", str(st))

        # Job 1: messages mode -> stitch. The server owns templating (--jinja); we only substitute
        # {{chunk}} into the messages array and post it as-is.
        submit(spool, {
            "epoch": 1,
            "text": "The quick brown fox.\n\nJumps over the lazy dog.",
            "budget_tokens": 256,
            "output": "stitch",
            "messages": [{
                "role": "user",
                "content": "Repeat the following text exactly, nothing else:\n\n{{chunk}}",
            }],
        })
        st = wait_state(spool, "done", 120, epoch=1)
        check("messages job done", st.get("state") == "done" and st.get("epoch") == 1, str(st))
        try:
            with open(os.path.join(spool, "result.txt")) as f:
                out1 = f.read()
        except OSError:
            out1 = ""
        check("messages job stitched non-empty result.txt", bool(out1.strip()), out1[:80])

        # Job 2: prompt mode -> collect. Raw completion prompt with add_special_tokens, tokenized
        # via /tokenize then generated via /completion.
        submit(spool, {
            "epoch": 2,
            "text": "Bonjour le monde.",
            "budget_tokens": 256,
            "output": "collect",
            "add_special_tokens": True,
            "prompt": "Translate this from French to English:\nFrench: {{chunk}}\nEnglish:",
        })
        st = wait_state(spool, "done", 120, epoch=2)
        check("prompt job done", st.get("state") == "done" and st.get("epoch") == 2, str(st))
        records = []
        try:
            with open(os.path.join(spool, "results.jsonl")) as f:
                records = [json.loads(line) for line in f if line.strip()]
        except OSError:
            pass
        rec = records[0] if records else {}
        check("prompt job appended one collect record", len(records) == 1, str(rec)[:100])
        check(
            "collect record round-trips source + non-empty output",
            rec.get("source") == "Bonjour le monde." and bool((rec.get("output") or "").strip())
            and "sep" in rec,
            f"source={rec.get('source')!r} output[:40]={(rec.get('output') or '')[:40]!r}")

        # Job 3: cancellation. A long generation (counting keeps the model off EOS) gives the stop
        # file time to land mid-stream; the job must resolve "cancelled", not done/error.
        submit(spool, {
            "epoch": 3,
            "text": "go",
            "budget_tokens": 256,
            "output": "stitch",
            "messages": [{
                "role": "user",
                "content": "Count slowly from 1 to 400, one number per line. Do not stop early. "
                           "Start now: {{chunk}}",
            }],
        })
        st = wait_state(spool, "mapping", 30, epoch=3)
        check("cancel job reached mapping", st.get("state") == "mapping", str(st))
        time.sleep(0.5)  # let a few tokens stream so the stop lands mid-generation
        open(os.path.join(spool, "stop"), "w").close()
        drop_t = time.time()
        st = wait_any(spool, {"cancelled", "done", "error"}, 30, epoch=3)
        check("mid-generation stop cancels the job", st.get("state") == "cancelled", str(st))
        check("cancel was prompt (< 5s)", (time.time() - drop_t) < 5, f"{time.time() - drop_t:.2f}s")
    finally:
        broker.stop()
        subprocess.run(["/bin/rm", "-rf", spool], check=False)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
