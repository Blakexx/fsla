"""The child process a profiler attaches to: one probe arm, N launches, nothing else.

It is deliberately the smallest program that runs the kernel. Every launch a
profiler sees must be one it was asked for, so this takes no lock (the parent holds
it), settles no clock (ncu controls clocks itself, at `--clock-control base` by
default) and times nothing (a timing loop pollutes counters).

Printed on start is the exact command line that produced it, which is what makes a
counter capture replayable by a reader who has only the ledger row.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.cells import CELLS  # noqa: E402
from bench.pairing import _timed  # noqa: E402
from probe.registry import get  # noqa: E402
from rola.routing.types import LeafMassDecay  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--launches", type=int, default=8)
    a = ap.parse_args()

    probe = get(a.probe)
    norm, decay_temp = probe.arms[a.arm]
    cell = CELLS[probe.cell]
    fx = cell.build()
    decay = None if decay_temp is None else LeafMassDecay(dials=fx["dials"][decay_temp])
    g_read = None if norm == "global" else fx["g_read"]
    plan, sched = cell.plan(norm=norm, decay_on=decay is not None, prec="bf16", fx=fx)
    for _ in range(a.launches):
        _, sched = _timed(cell, fx, plan, decay, g_read, sched)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
