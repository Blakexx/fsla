"""Cross-build ARMTIME on ONE cell, ledgered. Default cell: `atscale-L2048`.

These cells have no registered `rola-probe` entry (the registry's probes are
paired-arm cells; this is the escape-hatch case `bench.pairing.arm_time`
documents -- two builds differing by a compile-time kernel constant, which cannot
share a process). P10's original measurement of the at-scale cell went to a log
file, never to `perf/ledger.jsonl` -- that is the P22 finding this script
remedies. It reuses the SAME discipline every other probe uses
(`probe.discipline.disciplined`: GPU lock, settled clock, stamped identity) and
the SAME stopwatch (`bench.pairing.arm_time`), and writes rows through the SAME
`bench.ledger.make_row` schema.

**THE CELL IS A PARAMETER, AND THE DEFAULT IS FROZEN.** With no arguments this
script is byte-for-byte the P22 measurement it has always been -- same cell, same
probe name, same reps -- so every row it has ever written stays comparable. A
`--cell` run takes a DIFFERENT probe name, because a number about a different
cell is a different instrument, not a variant reading of this one. The parameter
exists because `consumer_reg_budget` is the compile-time constant `arm_time`'s
docstring names, and pricing a change to it requires the wall of a cell whose
arms that key actually governs (the `BC = 16` arms), not just the flagship.

Invoke BARE -- this takes the GPU lock itself (see docs/testing.md).

    python benchmarks/atscale_probe.py
    python benchmarks/atscale_probe.py --cell topo-D4-w8x8x8x16-BC16
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
#: AND the repo root, AHEAD of it: `rola` is installed EDITABLE into the shared
#: venv, so without this a worktree run imports the MAIN tree's extension while
#: stamping this tree's sha (P11's discarded A/B). `probe.discipline` REFUSES if
#: the two still disagree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.ledger import LEDGER, append
from bench.pairing import arm_time
from probe.discipline import disciplined

DEFAULT_CELL = "atscale-L2048"
PROBE = "consumer_forward_atscale"
PROBE_VERSION = 1
REPS = 30


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="atscale_probe")
    ap.add_argument("--cell", default=DEFAULT_CELL,
                    help=f"cell to time (default {DEFAULT_CELL}, the frozen P22 identity)")
    a = ap.parse_args(argv)
    cell = a.cell
    #: A different cell is a different INSTRUMENT, so it takes its own probe name
    #: rather than writing rows that would join against the at-scale series.
    probe = PROBE if cell == DEFAULT_CELL else f"consumer_forward_cell_{cell}"

    cmd = (f"python benchmarks/atscale_probe.py --cell {cell}  # arm_time({cell}, "
           f"norm=global, reps={REPS})")
    with disciplined(tier="landing", cmd=cmd) as run:
        result = arm_time(cell, reps=REPS, norm="global", decay_temp=None)
        row = run.row(
            probe=probe,
            probe_version=PROBE_VERSION,
            arm=f"{cell}|global|none",
            source=result["source"],
            metric="duration",
            unit="ms",
            values=result["times"],
            paired_with=None,
            note=("P22 remedy (U08 headline): " if cell == DEFAULT_CELL else "")
                 + "cross-build armtime, escape hatch per bench.pairing arm_time "
                   "docstring; median/IQR/min printed above",
        )
    n = append([row])
    print(f"{n} row appended to {LEDGER}")
    print(f"  sha={row['sha'][:8]} run_id={row['run_id']} median={result['median_ms']:.4f} "
          f"min={result['min_ms']:.4f} iqr={result['iqr']:.4f}")


if __name__ == "__main__":
    main()
