"""P39 acceptance: one cell's ARMTIME at a FORCED window, both trees.

The acceptance question is whether the prefetch buys latency independence, and
the instrument for that is the WIDE WINDOW -- an arm that trades CTA residency
for amortisation. `select_window` refuses it on this box, so it is pinned here
the same way `tests/tier1_semantics/test_consumer_vs_oracle.py` pins it: by
setting `CONSUMER_WINDOWS` to a one-tuple before the plan is built.

    python benchmarks/p39_window_probe.py --cell atscale-L2048 --window 64

Prints median/IQR/min; writes no ledger row (the window pin makes it a forced
configuration, not the cell's shipped identity).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.pairing import arm_time
from probe.discipline import disciplined


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="p39_window_probe")
    ap.add_argument("--cell", default="atscale-L2048")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args(argv)

    from rola.planner import schedule as schedule_mod
    schedule_mod.CONSUMER_WINDOWS = (a.window,)

    with disciplined(tier="landing", cmd=f"p39_window_probe --cell {a.cell} "
                                        f"--window {a.window}"):
        result = arm_time(a.cell, reps=a.reps, norm="global", decay_temp=None)
    print(f"ARMTIME {a.cell} window={a.window} reps={a.reps} [{result['source']}]")
    print(f"  median {result['median_ms']:.4f} ms   IQR {result['iqr']:.4f}   "
          f"min {result['min_ms']:.4f}")


if __name__ == "__main__":
    main()
