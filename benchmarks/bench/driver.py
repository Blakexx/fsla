"""The process that runs the harness: argument parsing, the discipline, the ledger write.

Thin by design. Every driver under `benchmarks/unit/` and `benchmarks/integration/` is
this module plus the name of the subject it prices, so the discipline cannot be acquired
in nine slightly different ways.

An ARM SPEC is `subject@cell` -- both halves are names from the registries, so a command
line is citable and a row's `arm` field reads the same way. `--ab A:B` runs the two specs
interleaved rep by rep through `bench.pairing.interleaved_ab`; without it each spec is
measured alone, which is the BASELINE mode.

Usage (this harness is invoked WRAPPED; it checks that it was):

    flock /tmp/rola_gpu.lock python benchmarks/unit/bench_chunk_facts.py --tier landing
"""
from __future__ import annotations

import argparse
import json

from bench.chunk_cells import CELLS, GROUPS
from bench.discipline import disciplined
from bench.ledger import append
from bench.pairing import INSTRUMENTS, REPS_PER_ROUND, Arm, interleaved_ab, measure_arm
from bench.regression import CANONICAL_INSTRUMENT
from bench.stats import MIN_ROUNDS, paired_verdict
from bench.subjects import SUBJECTS, applicable

#: The probe version of THIS harness. It is inside every fingerprint, so a redefinition
#: of what a subject measures starts a new series rather than extending the old one.
PROBE_VERSION = 1

#: Rounds per tier. `landing` is `MIN_ROUNDS`, which is DERIVED from the paired test's
#: alpha: below it no outcome can be significant and the test is undefined, not weak.
TIER_ROUNDS = {"light": 3, "landing": MIN_ROUNDS}


def parse_spec(spec: str, subjects) -> tuple[str, str]:
    subject, _, cell = spec.partition("@")
    if not cell:
        raise SystemExit(f"arm spec {spec!r} is not `subject@cell`")
    if subject not in subjects:
        raise SystemExit(f"{subject!r} is not one of this driver's subjects: {sorted(subjects)}")
    if cell not in CELLS:
        raise SystemExit(f"{cell!r} is not a registered cell: {sorted(CELLS)}")
    if subject not in applicable(CELLS[cell]):
        raise SystemExit(f"{subject!r} does not apply to {cell!r} "
                         f"(applicable: {applicable(CELLS[cell])})")
    return subject, cell


def build_arm(subject: str, cell: str, cache: dict) -> Arm:
    """The fixture is built ONCE per cell per process: two subjects on one cell must see
    the same fixture, or the pairing is across two draws as well as two arms."""
    if cell not in cache:
        cache[cell] = CELLS[cell].build()
    arm = SUBJECTS[subject](cache[cell])
    return Arm(name=f"{subject}|{cell}", call=arm.call)


def _row(run, subject: str, cell: str, values, source: str, **extra) -> dict:
    return run.row(probe=f"chunk:{subject}", probe_version=PROBE_VERSION,
                   arm=f"{subject}|{cell}", source=source, metric="duration", unit="ms",
                   values=values, **extra)


def main(subjects, argv=None, *, description: str = "") -> int:
    """One driver's whole body. `subjects` is the roster this driver is allowed to run."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--cells", default="all",
                   help=f"comma-separated cell names, or a group: {sorted(GROUPS)}")
    p.add_argument("--subject", default=None,
                   help=f"one of {sorted(subjects)}; default is all of this driver's")
    p.add_argument("--ab", default=None, metavar="A:B",
                   help="two `subject@cell` specs, run interleaved rep by rep")
    p.add_argument("--tier", default="light", choices=sorted(TIER_ROUNDS))
    p.add_argument("--rounds", type=int, default=None, help="override the tier's rounds")
    p.add_argument("--reps", type=int, default=REPS_PER_ROUND, help="reps per round")
    p.add_argument("--instrument", default=CANONICAL_INSTRUMENT, choices=sorted(INSTRUMENTS))
    p.add_argument("--json", default=None, help="also write the results here")
    p.add_argument("--no-ledger", action="store_true",
                   help="do not append to perf/ledger.jsonl (a scratch run, not a record)")
    args = p.parse_args(argv)

    rounds = args.rounds or TIER_ROUNDS[args.tier]
    cmd = f"bench {args.ab or args.subject or 'all'} --tier {args.tier}"
    names = GROUPS[args.cells] if args.cells in GROUPS else args.cells.split(",")
    out = []

    with disciplined(tier=args.tier, cmd=cmd) as run:
        print(f"run {run.run_id}  sha {run.sha[:12]}{' DIRTY' if run.dirty else ''}  "
              f"clock {run.clocks['sm_mhz']:.0f} MHz  {run.env['gpu']}")
        cache: dict = {}
        rows = []
        if args.ab:
            a_spec, _, b_spec = args.ab.partition(":")
            a = build_arm(*parse_spec(a_spec, subjects), cache)
            b = build_arm(*parse_spec(b_spec, subjects), cache)
            result = interleaved_ab(a, b, rounds=rounds, reps_per_round=args.reps,
                                    instrument=args.instrument)
            summary = result.summary()
            summary["significance"] = paired_verdict(result.diffs())
            out.append(summary)
            print(json.dumps(summary, indent=1, sort_keys=True, default=str))
            for arm, times in ((a, result.a_times), (b, result.b_times)):
                subject, cell = arm.name.split("|")
                rows.append(_row(run, subject, cell, times, args.instrument,
                                 paired_with=(b if arm is a else a).name,
                                 rounds=rounds, reps_per_round=args.reps))
        else:
            wanted = [args.subject] if args.subject else sorted(subjects)
            for cell in names:
                for subject in wanted:
                    if subject not in applicable(CELLS[cell]):
                        continue
                    arm = build_arm(subject, cell, cache)
                    res = measure_arm(arm, rounds=rounds, reps_per_round=args.reps,
                                      instrument=args.instrument)
                    res["cell"], res["subject"] = cell, subject
                    out.append(res)
                    print(f"  {subject:<20} {cell:<22} "
                          f"{res['median_ms']:9.4f} ms  IQR {res['iqr_ms']:.4f}")
                    rows.append(_row(run, subject, cell, res["values"], args.instrument,
                                     rounds=rounds, reps_per_round=args.reps))
        if not args.no_ledger:
            print(f"ledger += {append(rows)} rows")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, sort_keys=True, default=str)
    return 0
