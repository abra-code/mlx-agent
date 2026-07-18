#!/usr/bin/env python3
# Map-mode idle-unload smoke test.
#
# Drives a REAL MLX model in `map` mode with a short --idle-unload-seconds and proves the
# same policy the ACP path has (acp_idle_smoke.py): the model is released after inactivity
# (resident memory actually drops) and the next job transparently reloads it.
#
# Usage: python3 tools/map_idle_smoke.py /path/to/mlx-agent --model <dir>
#
# Every check asserts an OBSERVED effect:
#   - job 1 completes with non-empty stitched output (baseline behavior intact);
#   - the unload log's resident-memory delta is negative (weights really left RAM);
#   - the process RSS drops across the unload (OS-visible, not just MLX bookkeeping);
#   - job 2 AFTER the unload passes through "loading" and completes (transparent reload);
#   - the unload fires again after job 2 (the policy re-arms per job, not once).

import json
import os
import re
import subprocess
import sys
import threading
import time

IDLE_SECONDS = 5
WAIT_PAST_IDLE = IDLE_SECONDS + 6
POLL_S = 0.25

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    if cond:
        PASS += 1
    else:
        FAIL += 1


def rss_mib(pid):
    try:
        out = subprocess.check_output(["/bin/ps", "-o", "rss=", "-p", str(pid)], text=True)
        return int(out.strip()) // 1024
    except Exception:
        return 0


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
    # epoch matters: after job N is done, status.json still says "done" for epoch N while
    # job N+1 is being picked up - waiting on state alone returns instantly with stale data.
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = read_state(spool)
        if seen is not None and st.get("state") and (epoch is None or st.get("epoch") == epoch):
            seen.add(st["state"])
        if st.get("state") == want and (epoch is None or st.get("epoch") == epoch):
            return st
        time.sleep(POLL_S)
    return read_state(spool)


def submit_job(spool, epoch, text):
    job = {
        "epoch": epoch,
        "text": text,
        "budget_tokens": 512,
        "output": "stitch",
        "messages": [{"role": "user", "content": "Repeat exactly this text, nothing else: {{chunk}}"}],
    }
    tmp = os.path.join(spool, "job.json.tmp")
    with open(tmp, "w") as f:
        json.dump(job, f)
    os.replace(tmp, os.path.join(spool, "job.json"))


def unload_lines(broker):
    return [l for l in broker.log_lines() if "idle-unload: released" in l]


def main():
    if len(sys.argv) < 4 or sys.argv[2] != "--model":
        print("usage: map_idle_smoke.py /path/to/mlx-agent --model <dir>")
        return 2
    agent_bin, model_dir = sys.argv[1], sys.argv[3]
    spool = os.path.join(
        os.environ.get("TMPDIR", "/tmp"), f"map_idle_smoke_{os.getpid()}")
    os.makedirs(spool, exist_ok=True)

    broker = Broker([
        agent_bin, "map", "--model", model_dir, "--spool", spool,
        "--idle-unload-seconds", str(IDLE_SECONDS),
    ])
    try:
        st = wait_state(spool, "ready", 180)
        check("broker becomes ready", st.get("state") == "ready", str(st))

        # Job 1: baseline map behavior.
        submit_job(spool, 1, "The quick brown fox jumps over the lazy dog.")
        st = wait_state(spool, "done", 300, epoch=1)
        check("job 1 done", st.get("state") == "done" and st.get("epoch") == 1, str(st))
        try:
            with open(os.path.join(spool, "result.txt")) as f:
                out1 = f.read()
        except OSError:
            out1 = ""
        check("job 1 produced output", bool(out1.strip()), out1[:60])

        loaded_rss = rss_mib(broker.p.pid)

        # Idle long enough for the unload to fire.
        time.sleep(WAIT_PAST_IDLE)
        ul = unload_lines(broker)
        check("idle-unload fired after job 1", len(ul) >= 1, ul[-1] if ul else "(no line)")
        m = re.search(r"resident (\d+)MiB -> (\d+)MiB", ul[-1]) if ul else None
        check(
            "unload actually freed memory (log delta)",
            bool(m) and int(m.group(2)) < int(m.group(1)),
            m.group(0) if m else "(no resident delta in log)")
        unloaded_rss = rss_mib(broker.p.pid)
        # A substantial drop, not any drop: scheduler noise can shave a few MiB without the
        # weights leaving RAM. A real unload of a GiB-class model frees far more than 25%.
        check(
            "process RSS dropped substantially across unload",
            0 < unloaded_rss and (loaded_rss - unloaded_rss) > max(256, loaded_rss // 4),
            f"{loaded_rss}MiB -> {unloaded_rss}MiB")

        # Job 2: transparent reload. "loading" must be observable on the way to done.
        seen = set()
        submit_job(spool, 2, "Pack my box with five dozen liquor jugs.")
        st = wait_state(spool, "done", 300, seen=seen, epoch=2)
        check("job 2 done after reload", st.get("state") == "done" and st.get("epoch") == 2, str(st))
        check("reload passed through loading", "loading" in seen, f"states seen: {sorted(seen)}")
        reload_logs = [l for l in broker.log_lines() if "idle-reload: loading" in l]
        check("reload was logged", len(reload_logs) >= 1, reload_logs[-1] if reload_logs else "")
        try:
            with open(os.path.join(spool, "result.txt")) as f:
                out2 = f.read()
        except OSError:
            out2 = ""
        # Content check, not just non-empty: a stale job-1 result.txt must not false-pass.
        check("job 2 produced job 2's output", "liquor" in out2.lower(), out2[:60])

        # The policy re-arms: a second idle window unloads again.
        time.sleep(WAIT_PAST_IDLE)
        check("idle-unload fired again after job 2", len(unload_lines(broker)) >= 2)
    finally:
        broker.stop()
        subprocess.run(["/bin/rm", "-rf", spool], check=False)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
