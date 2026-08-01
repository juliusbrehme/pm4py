"""
Tiny stopwatch registry shared by the benchmark script and the discovery code.

Put this file somewhere importable from both (next to the benchmark script, or
inside your fork as e.g. pm4py/util/phase_timing.py) and wrap the phases you
care about inside discover_enhanced_process_tree:

    from phase_timing import phase

    with phase("preprocessing"):
        filtered_log = ...                  # your log preparation / refinement

    with phase("inductive_miner"):
        tree = inductive_miner.apply(filtered_log, ...)

    with phase("postprocessing"):
        tree = delete_taus(tree, ...)       # your tree rewriting

Timings accumulate per name (so a phase entered twice sums up) and live in the
process that runs the discovery. The benchmark resets the registry before every
run and snapshots it afterwards.

If nothing is instrumented yet, snapshot() just returns {} and the benchmark
leaves those CSV columns empty -- everything else keeps working.
"""

import time
from contextlib import contextmanager

_TIMINGS = {}


def reset():
    """Clear all recorded timings. Called once per benchmark run."""
    _TIMINGS.clear()


def snapshot():
    """Return a copy of the recorded timings as {phase_name: seconds}."""
    return dict(_TIMINGS)


def record(name, seconds):
    """Manually add seconds to a phase (if a context manager doesn't fit)."""
    _TIMINGS[name] = _TIMINGS.get(name, 0.0) + float(seconds)


@contextmanager
def phase(name):
    """Time the enclosed block and add it to the registry under `name`."""
    start = time.perf_counter()
    try:
        yield
    finally:
        record(name, time.perf_counter() - start)