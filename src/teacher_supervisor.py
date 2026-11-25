"""Watchdog for the teacher server: the second layer under the server's own application-level heartbeats.

    uv run python src/teacher_supervisor.py --port 5555 -- --port 5555 --compile_teacher false ...    # everything after -- is teacher_server.py's

It starts `teacher_server.py`, polls its status (`stream.query_status`, a tiny request the server answers from its own accept
thread) and restarts it, killing the whole process tree first, when it
  - exited (crashed, was killed, or a CUDA fault took the process down),
  - stops answering status queries for `--unreachable_s` (hung process: held GIL, wedged driver, dead socket thread; a fresh start
    gets `--grace_s` to load the model and cache the eval set before the first answer is expected),
  - is stuck in one teacher call for `--wedge_s` (a normal call takes ~1-2 s: a hung kernel looks like this), or
  - failed `--teacher_failures` teacher calls in a row (a corrupted CUDA context never recovers by itself).
A restart is invisible to the training except as a stall: the trainers' connections drop, they retry with backoff, and each resumes
at the exact batch it had not received (the stream is a pure function of seed and position). The supervisor itself never gives up;
restarts back off exponentially (up to 60 s) so a server that cannot start does not spin."""
import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import stream


@dataclass
class Watch:
    port: int = 5555
    poll_s: float = 5.0
    grace_s: float = 900.0
    unreachable_s: float = 60.0
    wedge_s: float = 90.0
    teacher_failures: int = 3


def judge(cfg, now, started, last_ok, status):
    """-> why the server should be restarted, or None. `status` is the last query's dict, or None if it did not answer.
    `last_ok` is when it last did (None: never since this start)."""
    if status is not None:
        if status.get("teacher_call_s", 0.0) > cfg.wedge_s:
            return f"stuck in one teacher call for {status['teacher_call_s']:.0f}s"
        if status.get("teacher_failures", 0) >= cfg.teacher_failures:
            return f"{status['teacher_failures']} teacher calls failed in a row"
        return None
    silent = now - (started if last_ok is None else last_ok)
    limit = cfg.grace_s if last_ok is None else cfg.unreachable_s
    if silent > limit:
        return f"no status answer for {silent:.0f}s" + (" since it started" if last_ok is None else "")
    return None


def kill_tree(proc):
    """Stop the server and anything it spawned (compile workers, dataloader processes)."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, check=False)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def supervise(cfg, spawn, query=None, clock=time.monotonic, sleep=time.sleep, log=print, stop=lambda: False):
    """Run `spawn()` (-> a Popen) under watch until `stop()`; returns the number of restarts. Injectable parts are for the tests."""
    query = query or (lambda: stream.query_status(f"127.0.0.1:{cfg.port}", timeout=10))
    restarts, failures = 0, 0
    while not stop():
        proc, started, last_ok = spawn(), clock(), None
        log(f"[supervisor] started teacher server (pid {proc.pid})")
        why = None
        try:
            while why is None and not stop():
                sleep(cfg.poll_s)
                if proc.poll() is not None:
                    why = f"exited with code {proc.returncode}"
                    break
                try:
                    status = query()
                except (OSError, ValueError) as e:
                    status = None
                    if last_ok is not None and clock() - last_ok > cfg.poll_s * 3:
                        log(f"[supervisor] status query failed: {e!r}")
                else:
                    last_ok = clock()
                why = judge(cfg, clock(), started, last_ok, status)
        except BaseException:  # Ctrl-C or a bug here must not leave the server running unsupervised
            kill_tree(proc)
            raise
        if stop():
            kill_tree(proc)
            break
        log(f"[supervisor] restarting the teacher server: {why}")
        kill_tree(proc)
        restarts += 1
        failures = 0 if last_ok is not None and clock() - started > 300 else failures + 1  # a server that ran 5 minutes was healthy
        wait = min(2 ** failures, 60)
        log(f"[supervisor] waiting {wait}s before restart #{restarts}")
        sleep(wait)
    return restarts


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, default in vars(Watch()).items():
        parser.add_argument(f"--{name}", type=type(default), default=default)
    parser.add_argument("--server_log", default="server.log", help="the server's stdout/stderr (appended)")
    parser.add_argument("--script", default=str(Path(__file__).with_name("teacher_server.py")))
    parser.add_argument("server_args", nargs="*", help="after --: teacher_server.py's own arguments")
    args = parser.parse_args(argv)
    cfg = Watch(**{k: getattr(args, k) for k in vars(Watch())})
    started = time.strftime("%Y-%m-%d %H:%M:%S")

    def spawn():
        kwargs = {"start_new_session": True} if sys.platform != "win32" else {}
        with open(args.server_log, "ab") as out:  # the child keeps its own duplicate of the handle
            out.write(f"\n===== supervisor {started}, server start {time.strftime('%H:%M:%S')} =====\n".encode())
            out.flush()
            return subprocess.Popen([sys.executable, "-u", args.script, *args.server_args], stdout=out, stderr=subprocess.STDOUT, **kwargs)

    log = lambda s: print(f"{time.strftime('%H:%M:%S')} {s}", flush=True)
    try:
        supervise(cfg, spawn, log=log)
    except KeyboardInterrupt:
        log("[supervisor] interrupted")


if __name__ == "__main__":
    main()
