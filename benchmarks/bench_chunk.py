"""The chunk arm's harness front door: the registry, the manifest, the baseline, the gate.

Five subcommands. Everything that reaches the device is invoked WRAPPED, and the harness
CHECKS that it was (`bench.discipline.require_gpu_lock`); `list` is the one that touches
no device and needs no lock.

    python benchmarks/bench_chunk.py list
    flock /tmp/rola_gpu.lock python benchmarks/bench_chunk.py manifest [--write]
    flock /tmp/rola_gpu.lock python benchmarks/bench_chunk.py run --ab A:B --tier landing
    flock /tmp/rola_gpu.lock python benchmarks/bench_chunk.py baseline --write
    flock /tmp/rola_gpu.lock python benchmarks/bench_chunk.py gate

`baseline --write` is the ONE quiet session: it measures every applicable (subject, cell)
at landing tier and records the file whole. `gate` reads that file and never writes it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json

from bench import chunk_baseline
from bench.chunk_cells import CELLS, GROUPS, check_manifest, write_manifest
from bench.discipline import disciplined
from bench.driver import PROBE_VERSION, TIER_ROUNDS, build_arm
from bench.driver import main as drive
from bench.ledger import append
from bench.pairing import REPS_PER_ROUND, measure_arm
from bench.regression import CANONICAL_INSTRUMENT
from bench.subjects import INTEGRATION, SUBJECTS, UNIT, applicable


def _list() -> int:
    print("CELLS")
    for name, cell in CELLS.items():
        print(f"  {name:<22} widths={cell.widths} B={cell.B} T={cell.T} H={cell.H} "
              f"alpha={cell.alpha} gain={cell.logit_gain} seed={cell.seed}")
        print(f"    {cell.note}")
    print(f"GROUPS  {json.dumps(GROUPS)}")
    print(f"UNIT SUBJECTS         {sorted(UNIT)}")
    print(f"INTEGRATION SUBJECTS  {sorted(INTEGRATION)}")
    return 0


def _manifest(args) -> int:
    with disciplined(tier="light", cmd="bench_chunk manifest"):
        if args.write:
            write_manifest()
            print("wrote the committed manifest")
            return 0
        return 0 if check_manifest() else 1


def _sweep(run, *, rounds, reps, instrument):
    """Every applicable (subject, cell), measured alone. The baseline and the gate share
    this one path, so the two sides of the comparison are the same quantity."""
    cache, arms, rows = {}, {}, []
    for cell_name, cell in CELLS.items():
        for subject in applicable(cell):
            arm = build_arm(subject, cell_name, cache)
            res = measure_arm(arm, rounds=rounds, reps_per_round=reps,
                              instrument=instrument)
            arms[chunk_baseline.key(subject, cell_name)] = res
            rows.append(run.row(probe=f"chunk:{subject}", probe_version=PROBE_VERSION,
                                arm=f"{subject}|{cell_name}", source=instrument,
                                metric="duration", unit="ms", values=res["values"],
                                rounds=rounds, reps_per_round=reps))
            print(f"  {subject:<20} {cell_name:<22} {res['median_ms']:9.4f} ms  "
                  f"IQR {res['iqr_ms']:.4f}")
    return arms, rows


def _baseline(args) -> int:
    if not args.write:
        raise SystemExit("`baseline` records; pass --write to say so. `gate` is the read side.")
    from bench.regression import summarize

    with disciplined(tier="landing", cmd="bench_chunk baseline --write") as run:
        if not check_manifest():
            raise SystemExit("REFUSING to record a baseline over DRIFTED cells: the file "
                             "would name fixtures nobody can rebuild.")
        arms, rows = _sweep(run, rounds=TIER_ROUNDS["landing"], reps=args.reps,
                            instrument=args.instrument)
        recorded = {k: dict(summarize(v["blocks"], source=v["source"]),
                            rounds=v["rounds"], reps_per_round=v["reps_per_round"],
                            warmup=v["warmup"])
                    for k, v in arms.items()}
        chunk_baseline.write(recorded, dirty=run.dirty, sha=run.sha,
                             build_stamp=run.stamp)
        print(f"ledger += {append(rows)} rows")
        print(f"wrote {len(recorded)} baseline rows to {chunk_baseline.BASELINE}")
    return 0


def _gate(args) -> int:
    baseline = chunk_baseline.load()
    with disciplined(tier=args.tier, cmd="bench_chunk gate") as run:
        arms, rows = _sweep(run, rounds=TIER_ROUNDS[args.tier], reps=args.reps,
                            instrument=args.instrument)
        print(f"ledger += {append(rows)} rows")
    over = []
    for k, res in arms.items():
        subject, cell = k.split("|")
        verdict = chunk_baseline.compare(baseline, subject, cell, res)
        if verdict["verdict"] == "over_limit":
            over.append(verdict)
        print(f"  {k:<44} {verdict['verdict']:<13} {verdict['median_ms']:9.4f} ms  "
              f"limit {verdict['limit_ms']:.4f}")
    print(f"{len(over)} of {len(arms)} arms over their committed limit (EFFECT SIZE "
          f"only -- the significance and debounce gates decide a regression)")
    return 1 if over else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    #: `run` is the general driver over the WHOLE roster; its own flags are parsed by
    #: `bench.driver`, so they are deliberately not re-declared here.
    sub.add_parser("run")
    m = sub.add_parser("manifest")
    m.add_argument("--write", action="store_true")
    for name in ("baseline", "gate"):
        s = sub.add_parser(name)
        s.add_argument("--reps", type=int, default=REPS_PER_ROUND)
        s.add_argument("--instrument", default=CANONICAL_INSTRUMENT)
        if name == "baseline":
            s.add_argument("--write", action="store_true")
        else:
            s.add_argument("--tier", default="landing", choices=sorted(TIER_ROUNDS))
    args, rest = p.parse_known_args(argv)
    if args.command == "list":
        return _list()
    if args.command == "manifest":
        return _manifest(args)
    if args.command == "baseline":
        return _baseline(args)
    if args.command == "gate":
        return _gate(args)
    return drive(set(SUBJECTS), rest, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
