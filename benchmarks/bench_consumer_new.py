"""The consumer benchmark CLI. **A DISPATCHER, AND NOTHING ELSE.**

The harness itself is :mod:`bench` (design §6.9): :mod:`bench.cells` is what is
measured, :mod:`bench.pairing` is THE methodology, :mod:`bench.instruments` is what
is measured about it, and :mod:`bench.drivers` is one thin function per question.
This file only parses arguments and calls one of them, so that adding a question
cannot add machinery here.

The prose that used to head the 1385-line original now lives with the thing it
describes: the cell rule in :mod:`bench.cells`, the interleaving and paired-ratio
rules in :mod:`bench.pairing`. What remains here is the mode table.

Modes:

  --mode oracle   correctness against the canonical fp64 oracle `naive_rola`,
                  at the declared per-arm tolerances. NEVER against a second
                  reference: validating against a freshly written one proves only
                  self-consistency (spec §2.1's prohibition; the shared-router bug
                  passed exactly that way).

  --mode atscale  DELETED by item 6b (2026-08-01): it was the OLD-consumer-vs-NEW
                  interleaved paired-ratio A/B, and there is one consumer now. Its
                  landed number (-24.7 % paired at `atscale-L2048`, IQR
                  0.094) is recorded in `rewrite.md` §2.20. What it used to say
                  about the two arms
                  ALTERNATE REP BY REP -- never arm-A-block then arm-B-block, which
                  lets a drift step land between the arms and masquerade as an
                  effect. Measured on this box, that bias was real: sequential
                  blocks read x3.27-3.30 where interleaving read x3.33-3.35, and
                  interleaving cut the run-to-run spread by ~7x.


  --mode skipshare  the read-only-`(owner, tile)` share per cell -- the fraction of
                  visits on which the fold GEMM and the decay panels are skipped.

  --mode armtime  ONE arm's absolute times on one cell. The only non-paired timing
                  mode, and it exists for the only comparison the paired doctrine
                  cannot express: two builds that differ by a COMPILE-TIME kernel
                  constant cannot share a process. The caller alternates PROCESSES
                  and pairs across them; see `arm_time`.

Usage (GPU work belongs under `flock /tmp/rola_gpu.lock`):

    python benchmarks/bench_consumer_new.py --mode oracle
    python benchmarks/bench_consumer_new.py --mode manifest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: The harness is a package NEXT TO this script, so a plain `python
#: benchmarks/bench_consumer_new.py` from anywhere finds it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench.cells import (  # noqa: E402  -- after the sys.path line, deliberately
    CELLS,
    GROUPS,
    MANIFEST,
    build_manifest,
    check_manifest,
    skip_share,
)
from bench.drivers import (  # noqa: E402
    decay_price,
    norm_price,
    run_oracle,
)
from bench.instruments import gather_stats  # noqa: E402
from bench.pairing import arm_time  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="oracle",
                    choices=["oracle", "skipshare", "manifest",
                             "decayprice", "normprice",
                             "gatherstats", "armtime"])
    ap.add_argument("--cell", default="atscale-L2048")
    ap.add_argument("--norm", default="global", choices=["global", "raw"])
    ap.add_argument("--decay", default=None, choices=[None, "warm", "hot"])
    ap.add_argument("--no-decay-axis", action="store_true",
                    help="oracle mode: skip the decay arms")
    ap.add_argument("--hot", action="store_true")
    ap.add_argument("--reps", type=int, default=11)
    ap.add_argument("--write", action="store_true", help="manifest mode: write it")
    a = ap.parse_args()

    if a.mode == "oracle":
        raise SystemExit(0 if run_oracle(decay=not a.no_decay_axis, hot=a.hot) else 1)
        return
    if a.mode == "armtime":
        arm_time(a.cell, reps=a.reps, norm=a.norm, decay_temp=a.decay)
        return
    if a.mode == "gatherstats":
        for name in (GROUPS["topo"] + GROUPS["wk1"] + GROUPS["atscale"]):
            st = gather_stats(CELLS[name])
            print(f"  {name:26s} BC={CELLS[name].BC:3d}  mean K {st['mean_K']:7.3f}"
                  f"  mean KI {st['mean_KI']:7.3f}  k-steps {st['mean_ksteps']:.3f}"
                  f"  m-tiles {st['mean_mtiles']:.3f}"
                  f"  ({st['owners_sampled']} owners, {st['candidate_pairs']} pairs)")
        return
    if a.mode == "decayprice":
        #: Three cells BRACKETING the write-live share, because the structural
        #: prediction is that decay's price scales with it: a dense-write cell, the
        #: sparse-routing at-scale cell, and a write-`k=1` cell whose read-only
        #: share is 77.9 %.
        for name in ("topo-D2-w64x64-BC64", "atscale-L2048", "wk1-D2-w64x64-BC64"):
            decay_price(name, reps=a.reps)
        return
    if a.mode == "normprice":
        for name in ("topo-D2-w64x64-BC64", "atscale-L2048", "wk1-D2-w64x64-BC64"):
            norm_price(name, reps=a.reps, decay_temp=a.decay)
        return
    if a.mode == "skipshare":
        for name in (GROUPS["topo"] + GROUPS["wk1"]):
            s = skip_share(CELLS[name])
            print(f"  {name:26s} owners {s['n_owners']:5d}  candidate {s['candidate_pairs']:7d}"
                  f"  read-only (FOLD SKIPPED) {s['read_only_pairs']:7d}"
                  f" = {100 * s['read_only_frac']:5.1f} %")
        return
    if a.write:
        MANIFEST.write_text(json.dumps(build_manifest(), indent=2) + "\n")
        print(f"wrote {MANIFEST}")
        return
    raise SystemExit(0 if check_manifest() else 1)


if __name__ == "__main__":
    main()
