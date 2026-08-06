"""P22-remedy evidence script: ledgers the atscale-L2048 cross-build ARMTIME number.

`atscale-L2048` has no registered `rola-probe` entry (the registry's probes are
paired-arm cells; this one is the escape-hatch case `bench.pairing.arm_time`
documents -- two builds differing by a compile-time kernel constant, which cannot
share a process). P10's original measurement of this cell went to a log file, never
to `perf/ledger.jsonl` -- that is the P22 finding this script remedies. It reuses
the SAME discipline every other probe uses (`probe.discipline.disciplined`: GPU
lock, settled clock, stamped identity) and the SAME stopwatch
(`bench.pairing.arm_time`), and writes rows through the SAME `bench.ledger.make_row`
schema, under probe name `consumer_forward_atscale` (new identity, since it is not
the registry's `consumer_forward` cell).

Invoke BARE -- this takes the GPU lock itself (see docs/testing.md).

    python benchmarks/atscale_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench.ledger import LEDGER, append
from bench.pairing import arm_time
from probe.discipline import disciplined

PROBE = "consumer_forward_atscale"
PROBE_VERSION = 1
REPS = 30


def main():
    cmd = ("python benchmarks/atscale_probe.py  # arm_time(atscale-L2048, "
           f"norm=global, reps={REPS})")
    with disciplined(tier="landing", cmd=cmd) as run:
        result = arm_time("atscale-L2048", reps=REPS, norm="global", decay_temp=None)
        row = run.row(
            probe=PROBE,
            probe_version=PROBE_VERSION,
            arm="atscale-L2048|global|none",
            source=result["source"],
            metric="duration",
            unit="ms",
            values=result["times"],
            paired_with=None,
            note="P22 remedy (U08 headline): cross-build armtime, escape hatch "
                 "per bench.pairing arm_time docstring; median/IQR/min printed above",
        )
    n = append([row])
    print(f"{n} row appended to {LEDGER}")
    print(f"  sha={row['sha'][:8]} run_id={row['run_id']} median={result['median_ms']:.4f} "
          f"min={result['min_ms']:.4f} iqr={result['iqr']:.4f}")


if __name__ == "__main__":
    main()
