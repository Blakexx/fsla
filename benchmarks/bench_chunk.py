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
import subprocess

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


def _sweep_once(args) -> int:
    """ONE session's sweep, dumped as JSON. The unit the recorder pairs ACROSS."""
    with disciplined(tier="landing", cmd="bench_chunk sweep") as run:
        arms, rows = _sweep(run, rounds=args.rounds, reps=args.reps,
                            instrument=args.instrument)
        print(f"ledger += {append(rows)} rows")
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump({k: {"median_ms": v["median_ms"], "blocks": v["blocks"],
                       "source": v["source"], "rounds": v["rounds"],
                       "reps_per_round": v["reps_per_round"], "warmup": v["warmup"]}
                   for k, v in arms.items()}, fh)
    return 0


def _baseline(args) -> int:
    """Record the baseline across SESSIONS, because that is what the gate compares.

    **MEASURED, and the reason this is not a single sweep.** A first baseline was recorded
    from one session's block medians, and the very next gate run on the SAME BINARY put 17
    of 44 arms over their limit. The within-session block IQR is the spread of eleven reps
    taken back to back on one warm fixture -- often 0.0000 ms -- and the gate's comparison
    is across two PROCESSES, where the fixture is rebuilt, the allocator is cold and the
    launch order differs. Those are different spreads, and the threshold has to be derived
    from the one the comparison actually crosses.

    So the recorder runs `--sessions` CHILD PROCESSES and takes the median and IQR over
    their session medians. This is the retired recorder's own shape (35 reps x 5 blocks x
    7 processes) arrived at from the same evidence. The within-session IQR is recorded
    beside it as `within_session_iqr_ms`, because it is a real measurement of a different
    thing and dropping it would lose the contrast that motivated the change.

    The children inherit the wrapper's lock descriptor, so `require_gpu_lock` passes in
    each of them without any of them taking a second lock.
    """
    import statistics
    import tempfile

    if not args.write:
        raise SystemExit("`baseline` records; pass --write to say so. `gate` is the read side.")
    from bench.regression import CANONICAL_INSTRUMENT as _ci
    from bench.regression import iqr

    with disciplined(tier="light", cmd="bench_chunk baseline --write") as run:
        if not check_manifest():
            raise SystemExit("REFUSING to record a baseline over DRIFTED cells: the file "
                             "would name fixtures nobody can rebuild.")
        sha, dirty, stamp = run.sha, run.dirty, run.stamp

    sessions = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(args.sessions):
            out = Path(tmp) / f"session{i}.json"
            print(f"--- session {i + 1}/{args.sessions}")
            done = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "sweep",
                 "--json", str(out), "--rounds", str(args.rounds),
                 "--reps", str(args.reps), "--instrument", args.instrument],
                check=False)
            if done.returncode != 0:
                raise SystemExit(f"session {i} failed with {done.returncode}; a baseline is "
                                 f"recorded WHOLE or not at all")
            sessions.append(json.loads(out.read_text()))

    keys = set(sessions[0])
    for session in sessions[1:]:
        if set(session) != keys:
            raise SystemExit("the sessions measured different arms; a baseline whose rows "
                             "come from different rosters compares nothing")
    recorded = {}
    for k in sorted(keys):
        medians = [s[k]["median_ms"] for s in sessions]
        first = sessions[0][k]
        recorded[k] = {"median_ms": round(statistics.median(medians), 6),
                       "iqr_ms": round(iqr(medians), 6),
                       "within_session_iqr_ms": round(iqr(first["blocks"]), 6),
                       "session_medians": [round(m, 6) for m in medians],
                       "sessions": len(medians), "reps": first["rounds"] * first["reps_per_round"],
                       "rounds": first["rounds"], "reps_per_round": first["reps_per_round"],
                       "warmup": first["warmup"], "source": first["source"] or _ci}
        print(f"  {k:<44} {recorded[k]['median_ms']:9.4f} ms  "
              f"cross-session IQR {recorded[k]['iqr_ms']:.4f}  "
              f"(within-session {recorded[k]['within_session_iqr_ms']:.4f})")
    chunk_baseline.write(recorded, dirty=dirty, sha=sha, build_stamp=stamp,
                         sessions=args.sessions)
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
    for name in ("baseline", "gate", "sweep"):
        s = sub.add_parser(name)
        s.add_argument("--reps", type=int, default=REPS_PER_ROUND)
        s.add_argument("--instrument", default=CANONICAL_INSTRUMENT)
        if name == "baseline":
            s.add_argument("--write", action="store_true")
            #: SEVEN sessions, the retired recorder's own count. The threshold is derived
            #: from the spread ACROSS them because that is the gap the gate crosses.
            s.add_argument("--sessions", type=int, default=7)
            s.add_argument("--rounds", type=int, default=TIER_ROUNDS["light"])
        elif name == "sweep":
            s.add_argument("--json", required=True)
            s.add_argument("--rounds", type=int, default=TIER_ROUNDS["light"])
        else:
            s.add_argument("--tier", default="landing", choices=sorted(TIER_ROUNDS))
    args, rest = p.parse_known_args(argv)
    if args.command == "list":
        return _list()
    if args.command == "manifest":
        return _manifest(args)
    if args.command == "baseline":
        return _baseline(args)
    if args.command == "sweep":
        return _sweep_once(args)
    if args.command == "gate":
        return _gate(args)
    return drive(set(SUBJECTS), rest, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
