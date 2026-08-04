"""
Run a job in a child process with a hard, per-phase deadline.

Threads cannot be killed in Python: a `future.result(timeout=...)` only stops
*waiting*, the runaway loop keeps burning CPU until it finishes. A child
process can be SIGKILLed, so that is what this module does.

The job is a function `target(queue, *args)` that reports progress by putting
messages on the queue:

    ("phase", <phase name>, <seconds>, <extra dict>)   phase finished
    ("result", <picklable object>)                     job finished
    ("error", <phase name>, <message>)                 job failed

The parent gives each phase its own timeout. When a phase overruns, the whole
process group of the child is killed, so anything the child spawned itself
(e.g. pm4py's own multiprocessing pools) dies with it.
"""

import multiprocessing as mp
import os
import signal
import time
from queue import Empty

# How often the parent wakes up to check the deadline / whether the child died.
POLL_INTERVAL = 0.25

_METHODS = mp.get_all_start_methods()
CTX = mp.get_context("fork" if "fork" in _METHODS else "spawn")


def _child_wrapper(target, queue, args):
    # Own process group -> the parent can kill the child *and* its descendants.
    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            pass
    target(queue, *args)


def kill_process_tree(proc):
    """SIGKILL the worker and everything it spawned. Safe to call twice."""
    if proc.pid is None:
        return
    killed = False
    if hasattr(os, "killpg"):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            killed = True
        except (ProcessLookupError, PermissionError, OSError):
            killed = False
    if not killed:
        try:
            if proc.is_alive():
                proc.terminate()
                proc.join(5)
            if proc.is_alive():
                proc.kill()
        except (AssertionError, OSError, ValueError):
            pass
    proc.join(10)


def run_phased(target, args=(), phases=(("work", None),)):
    """
    Run target(queue, *args) in a child process.

    phases: ordered [(phase_name, timeout_seconds_or_None), ...]. The first
    phase's clock starts when the child starts; each ("phase", ...) message
    starts the clock of the next one.

    Returns a dict:
        status       SUCCESS | TIMEOUT | FAILED | CRASHED
        failed_phase phase that timed out / failed, else None
        phase_times  {phase_name: seconds} for the phases that completed
                     (plus a partial entry for a phase that timed out)
        extras       {phase_name: extra dict} from the phase messages
        result       payload of the ("result", ...) message, else None
        error        message, else ""
    """
    phase_list = list(phases)
    queue = CTX.Queue()
    proc = CTX.Process(target=_child_wrapper, args=(target, queue, tuple(args)), daemon=True)

    out = {
        "status": "SUCCESS",
        "failed_phase": None,
        "phase_times": {},
        "extras": {},
        "result": None,
        "error": "",
    }

    proc.start()
    index = 0
    name, timeout = phase_list[0]
    phase_start = time.monotonic()
    deadline = None if timeout is None else phase_start + timeout

    try:
        while True:
            try:
                message = queue.get(timeout=POLL_INTERVAL)
            except Empty:
                if deadline is not None and time.monotonic() > deadline:
                    out["status"] = "TIMEOUT"
                    out["failed_phase"] = name
                    out["phase_times"][name] = time.monotonic() - phase_start
                    out["error"] = f"{name} exceeded {timeout} seconds (worker killed)"
                    break
                if not proc.is_alive():
                    # The child may have exited with a message still in flight.
                    try:
                        message = queue.get(timeout=1.0)
                    except Empty:
                        out["failed_phase"] = name
                        out["phase_times"][name] = time.monotonic() - phase_start

                        # Exit code -24 is SIGXCPU, -9 is SIGKILL
                        if proc.exitcode in (-24, -9):
                            out["status"] = "TIMEOUT"
                            out[
                                "error"] = f"{name} exceeded CPU time limit (OS killed worker with signal {-proc.exitcode})"
                        else:
                            out["status"] = "CRASHED"
                            out["error"] = (
                                f"worker died during {name} without reporting "
                                f"(exit code {proc.exitcode})"
                            )
                        break
                else:
                    continue
            # falls through with a message either from the poll or the last-chance get
            kind = message[0]
            if kind == "phase":
                _, done_name, elapsed, extra = message
                out["phase_times"][done_name] = elapsed
                out["extras"][done_name] = extra
                index += 1
                if index < len(phase_list):
                    name, timeout = phase_list[index]
                    phase_start = time.monotonic()
                    deadline = None if timeout is None else phase_start + timeout
                else:
                    deadline = None
            elif kind == "result":
                out["result"] = message[1]
                break
            elif kind == "error":
                out["status"] = "FAILED"
                out["failed_phase"] = message[1]
                out["error"] = message[2]
                break
    finally:
        kill_process_tree(proc)

    return out