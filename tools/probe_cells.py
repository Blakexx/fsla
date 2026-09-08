#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CARRY FAMILY'S A/B ORCHESTRATOR -- two binaries, the registered benches, one
measurement definition.

THIS FILE DEFINES NO STEP, NO CELL AND NO SHAPE. What is measured is a REGISTERED BENCH
(`benchmarks/bench/subjects.py`) over a REGISTERED CELL (`benchmarks/cells`), which is
the same callable and the same draw the unit drivers and the receipts run. A bench that
cannot be A/B'd through here is the defect: it means a second definition of a launch
exists somewhere, and two definitions are how a harness comes to disagree with itself.

What this file owns is the CROSS-BINARY comparison, and only that:

  * TWO BINARIES, each run by its OWN worktree's venv python in its own process, so the
    only thing that differs between an "A" run and a "B" run is which compiled extension
    answers `import rola`. The measurement code is this file plus the registry plus the
    subject, all read from the worktree being measured.
  * INTERLEAVED ROUNDS WITH ALTERNATING ORDER, A,B then B,A then A,B -- never all of A
    then all of B, and never the same arm first in every round. Run-to-run drift on this
    host is of the same order as the effects being measured: a block layout lets a clock
    change land BETWEEN the arms, and a fixed order inside every round charges any drift
    slower than a round wholly to one arm -- this host has a two-state sustained clock
    ~17% apart with a minutes-long time constant, and the same binary in both arms once
    read 0.864 vs 0.739 ms under a fixed order. Alternating the order cancels first-
    position bias; the clock-fixed ncu duration beside every ratio is the check.
  * THE SYMBOL ASSERT. Before it times anything, each worker asks the binary for the
    family's DEVICE-SIDE build stamp -- a fact only a built kernel can produce -- and
    refuses if the binary cannot answer. A path check and a hash check both pass against
    a stale extension; only a device-side fact does not.
  * `--ncu`, whose kernel filter NAMES THE BODY (the subject's own `symbol`) rather than
    a family prefix, because the first kernel of a family is not the same kernel on two
    tips and a ratio taken across two different kernels means nothing.
  * THE RECORD. Every run writes `benchmarks/results/` rows through the same writer the
    benches use, so a measurement is not lost to a terminal scrollback.

Usage:

    python tools/probe_cells.py --bench carry_forward --cells flagship-alt-k4 \\
        --binary worktree:/path/a,venv:/path/venv-a,label:A \\
        --binary worktree:/path/b,venv:/path/venv-b,label:B

See `docs/KERNEL_STANDARDS.md` section 16 (the perf probe every stage gate takes) and its
addenda: instructions are necessary but not sufficient, so carry IPC and stall reasons
beside them.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import subprocess
import sys
from pathlib import Path

from gpu_lock import GPU_LOCK_DEFAULT, gpu_lock

THIS_FILE = Path(__file__).resolve()
REPO = THIS_FILE.parents[1]
NCU_BIN = "/usr/local/cuda/bin/ncu"

#: The registry and the bench package are imported from the WORKTREE BEING MEASURED, not
#: from this file's own tree: a worker runs with its own worktree as the working
#: directory, so these two entries resolve there.
sys.path.insert(0, str(REPO / "benchmarks"))
sys.path.insert(0, str(REPO))

#: THE STATE SHAPES A CALL CAN BIND. `fresh` binds one exit plane over a zero entry state -- the
#: like-for-like default. `continuation` passes the SAME plane in and out, which is the
#: shape a growing sequence actually runs and the one where the entry sweep's loads and
#: the exit sweep's stores address the same pages.
STATE_SHAPES = ("fresh", "continuation")


# ---------------------------------------------------------------------------
# WORKER MODE: runs IN the target binary's own venv python, in its own process.

def worker_main(args) -> None:
    import torch

    import rola
    from bench.driver import CELLS
    from bench.subjects import SUBJECTS, applicable
    from rola.ops._ext import extension

    subject = SUBJECTS[args.bench]
    #: THE SYMBOL ASSERT, made of a fact the DEVICE produces. A stale extension at the
    #: right path answers every question about paths and hashes correctly.
    stamp_entry = getattr(extension(), subject.stamp, None)
    refusal = None
    family_stamp = None
    if stamp_entry is None:
        refusal = (f"this binary carries no {subject.stamp} entry, so it cannot state "
                   f"that it has {subject.name}'s family at all")
    else:
        try:
            family_stamp = int(stamp_entry())
        except RuntimeError as ex:
            refusal = (f"{subject.stamp} refused: {subject.name}'s kernel is not built "
                       f"here, so there is nothing to measure -- {ex}")

    try:
        props = torch.cuda.get_device_properties(0)
        device_uuid = str(props.uuid)
    except Exception:  # noqa: BLE001
        props, device_uuid = torch.cuda.get_device_properties(0), None
    try:
        from rola._build_config import BUILD_CONFIG
        manifest_sha256, ptxas = BUILD_CONFIG["manifest_sha256"], BUILD_CONFIG["ptxas"]
    except Exception:  # noqa: BLE001
        manifest_sha256, ptxas = None, None
    stamp = {
        "rola_file": rola.__file__,
        "device_name": torch.cuda.get_device_name(0),
        "device_uuid": device_uuid,
        "manifest_sha256": manifest_sha256,
        "family_stamp": family_stamp,
        "sm": f"sm_{props.major}{props.minor}",
        "driver_cuda": torch.version.cuda,
        "torch": torch.__version__,
        "ptxas": ptxas,
    }

    rows = {}
    if refusal is not None:
        #: A REFUSAL IS A RESULT, reported in the same JSON every run emits. Raising here
        #: would hand the orchestrator a traceback where it expects a row, and a probe
        #: over an unbuilt kernel is an ordinary state on this line, not a crash.
        print(json.dumps({"stamp": stamp, "refused": refusal, "rows": {}}))
        sys.exit(1)
    for name in args.cells.split(","):
        try:
            if name not in CELLS:
                raise KeyError(f"{name!r} is not a registered cell")
            kind, spec = CELLS[name]
            if args.bench not in applicable(spec, kind):
                raise ValueError(
                    f"{args.bench} does not apply to {name} "
                    f"(applicable: {applicable(spec, kind)})")
            fx = ({"cell": spec} if kind == "carry"
                  else {**_layer_fixture(spec), "cell": spec})
            fx["state_arm"] = args.state
            fx["schedule"] = args.schedule
            arm = subject.build(fx)
            for _ in range(args.warmup):
                arm.call()
            torch.cuda.synchronize()
            #: THE CLOCK the reps run in, read off the device beside them (a binary from before
            #: the probe reports none). The governor moves it with power, so the number is
            #: comparable across runs only with the clock locked from the host
            #: (`docs/internals/common/sm_clock.md`); the read is what proves the lock held.
            from rola.ops import carry as carry_ops
            read_ghz = getattr(carry_ops, "sm_clock_ghz", None)
            ghz = read_ghz() if read_ghz else None
            if args.oneshot:
                arm.call()
                torch.cuda.synchronize()
                rows[name] = {"oneshot": True, "state_arm": args.state,
                              "schedule": args.schedule}
                continue
            times = []
            for _ in range(args.reps):
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                arm.call()
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            rows[name] = {"median_ms": statistics.median(times), "min_ms": min(times),
                          "max_ms": max(times), "reps": args.reps, "warmup": args.warmup,
                          "state_arm": args.state, "schedule": args.schedule,
                          "sm_ghz": ghz}
            del arm, fx
            torch.cuda.empty_cache()
        except Exception as ex:  # noqa: BLE001 -- reported per cell, never swallowed
            rows[name] = {"error": f"{type(ex).__name__}: {ex}"}
    print(json.dumps({"stamp": stamp, "rows": rows}))
    if any("error" in row for row in rows.values()):
        sys.exit(1)


def _layer_fixture(spec):
    from cells.layer import build

    return build(spec)


# ---------------------------------------------------------------------------
# ORCHESTRATOR

def parse_binary_spec(spec: str) -> dict:
    """``worktree:/path,venv:/path[,label:NAME][,schedule:NAME]`` -> dict.

    ``schedule`` is the carry family's per-arm runtime dial (`rola.ops.carry.CarrySchedule`):
    ``box`` (the default, and what a caller who names nothing gets: one stream a warp box),
    ``sparse`` (both sides at the reaping bound) or ``sparse-gN`` (both sides at a stream
    grain of N leaves). It is an ARM property, not a run property, so the two arms of a probe may be the
    SAME binary differing in exactly the dial under test -- which is what the reap is measured
    as. An arm whose binary predates the dial ignores it, because each worker runs its own
    worktree's copy of this file.
    """
    out = {}
    for part in spec.split(","):
        key, _, value = part.partition(":")
        out[key.strip()] = value.strip()
    if "worktree" not in out or "venv" not in out:
        raise argparse.ArgumentTypeError(
            f"--binary needs worktree:PATH,venv:PATH (got {spec!r})")
    out.setdefault("label", Path(out["worktree"]).name)
    out.setdefault("schedule", "box")
    return out


def _git(worktree: str, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", worktree, *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception as ex:  # noqa: BLE001
        return f"UNKNOWN ({ex})"


def git_sha(worktree: str) -> str:
    return _git(worktree, "rev-parse", "HEAD")


def git_branch(worktree: str) -> str:
    return _git(worktree, "rev-parse", "--abbrev-ref", "HEAD")


def git_dirty(worktree: str) -> bool:
    """Whether the MEASURED binary's own tree was clean. A dirty tree does not block the
    measurement but stamps the row advisory."""
    out = _git(worktree, "status", "--porcelain")
    return bool(out.strip()) if not out.startswith("UNKNOWN") else True


def tree_digest(worktree: str) -> str:
    """THE TREE'S OWN IDENTITY: sha256 of the diff against HEAD over the tracked files, empty
    for a clean tree. A dirty tree's record carries the base commit's sha, and two different
    uncommitted kernels measured over the same base share that sha -- the deferral of
    2026-09-05 measured 5.67 and 6.80 under one label and one sha (journal section 39). The
    digest tells them apart after the fact; the printed row says DIRTY at the time."""
    import hashlib
    diff = _git(worktree, "diff", "HEAD", "--", ".")
    return "" if not diff.strip() or diff.startswith("UNKNOWN") else hashlib.sha256(diff.encode()).hexdigest()


def venv_python(venv_path: str) -> str:
    python = Path(venv_path) / "bin" / "python"
    if not python.exists():
        raise FileNotFoundError(f"no python at {python}")
    return str(python)


def worker_argv(binary: dict, bench: str, cells: list[str], reps: int, warmup: int,
                state: str, oneshot: bool, schedule: str = "box") -> list[str]:
    """The worker command, built against the MEASURED worktree's own copy of this file.

    Both halves come from the tree being measured -- the harness and the registry -- so a
    tip that changed what a cell is measures its own definition and says so through its
    SHA, rather than being measured by this tree's definition of someone else's cell.
    """
    argv = [venv_python(binary["venv"]),
            str(Path(binary["worktree"]) / "tools" / "probe_cells.py"),
            "--worker", "--bench", bench, "--cells", ",".join(cells),
            "--reps", str(reps), "--warmup", str(warmup), "--state", state]
    #: PASSED ONLY WHEN IT IS NOT THE DEFAULT: the worker script is the MEASURED tree's, and
    #: a tip that predates the dial would refuse an argument it has never heard of. `box`
    #: IS what such a tip runs, so saying nothing says the right thing.
    if schedule != "box":
        argv += ["--schedule", schedule]
    if oneshot:
        argv.append("--oneshot")
    return argv


def run_worker(binary: dict, bench: str, cells: list[str], reps: int, warmup: int,
               lock_path: str, state: str, oneshot: bool = False,
               timeout: int = 600) -> dict:
    cmd = worker_argv(binary, bench, cells, reps, warmup, state, oneshot,
                      binary.get("schedule", "box"))
    with gpu_lock(lock_path, mode="exclusive"):  # measured work
        proc = subprocess.run(cmd, cwd=binary["worktree"], capture_output=True, text=True,
                              timeout=timeout)
    if proc.returncode not in (0, 1):  # 1 = a cell REFUSED, still emits partial JSON
        raise RuntimeError(f"worker for {binary['label']} crashed (rc={proc.returncode}):\n"
                           f"stdout={proc.stdout}\nstderr={proc.stderr}")
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as ex:  # noqa: BLE001
        raise RuntimeError(f"worker for {binary['label']} produced no JSON "
                           f"(rc={proc.returncode}): stdout={proc.stdout!r} "
                           f"stderr={proc.stderr!r}") from ex


#: The counters every stage gate reads: the wall time, the instruction and IPC pair the
#: standards' addendum requires beside it, the three stall reasons, the two memory legs,
#: the shared-memory leg -- a layout above its floor shows up as bank conflicts and MIO
#: pressure long before it shows up in an instruction count -- and the not-selected share,
#: which is the launch shape's own claim read back off the scheduler.
NCU_METRICS = (
    "gpu__time_duration.sum",
    "smsp__inst_executed.sum",
    "sm__inst_executed.avg.per_cycle_active",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "dram__bytes.sum",
    "lts__t_bytes.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "smsp__inst_executed_op_shared_ld.sum",
    "smsp__inst_executed_op_shared_st.sum",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    #: NOT_SELECTED is the one-CTA-per-SM claim, measured: at the design point there is
    #: no second CTA for the scheduler to pick instead, so a warp that was eligible and
    #: not chosen is a sign the launch shape is not the one the design believes it is.
    "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
)


def run_ncu_cell(binary: dict, bench: str, symbol: str, cell: str, lock_path: str,
                 out_csv_prefix: str, state: str, timeout: int = 900) -> dict:
    """One `ncu` oneshot launch: the named body's counters for ``cell`` on ``binary``."""
    cmd = [NCU_BIN, "--target-processes", "all", "-k", f"regex:{symbol}", "-c", "1",
           "--print-kernel-base", "mangled", "--metrics", ",".join(NCU_METRICS), "--csv",
           *worker_argv(binary, bench, [cell], 0, 0, state, True,
                        binary.get("schedule", "box"))]
    out_csv = f"{out_csv_prefix}_{binary['label']}_{bench}_{cell}.csv"
    with gpu_lock(lock_path, mode="exclusive"):  # measured work
        proc = subprocess.run(cmd, cwd=binary["worktree"], capture_output=True, text=True,
                              timeout=timeout)
    Path(out_csv).write_text(proc.stdout)
    row = {"csv": out_csv, "rc": proc.returncode}
    #: ncu's stdout interleaves its own progress lines and the worker's JSON stamp BEFORE
    #: the real CSV header; find the header by its known first column.
    lines = proc.stdout.splitlines()
    header = next((i for i, line in enumerate(lines) if line.startswith('"ID"')), None)
    if header is None:
        row["stderr"] = proc.stderr[-2000:]
        row["stdout_tail"] = "\n".join(lines[-10:])
        return row
    metrics = {}
    kernel_name = None
    for record in csv.DictReader(io.StringIO("\n".join(lines[header:]))):
        kernel_name = record.get("Kernel Name", kernel_name)
        if record.get("Metric Name"):
            metrics[record["Metric Name"]] = record.get("Metric Value")
    row["kernel_name"] = kernel_name
    row["metrics"] = metrics
    return row


def interleaved_rounds(binaries: list[dict], bench: str, cells: list[str], reps: int,
                       warmup: int, lock_path: str, rounds: int, state: str) -> dict:
    """A,B then B,A then A,B -- interleaved, and the first position alternates by round
    (this file's second rule), so a drift slower than a round is not charged to one arm."""
    per_binary = {b["label"]: [] for b in binaries}
    for index in range(rounds):
        order = binaries if index % 2 == 0 else list(reversed(binaries))
        for binary in order:
            payload = run_worker(binary, bench, cells, reps, warmup, lock_path, state)
            per_binary[binary["label"]].append({"round": index, **payload})
    return per_binary


def summarize(round_rows: list[dict], cell: str) -> dict | None:
    values = [rr["rows"][cell]["median_ms"] for rr in round_rows
              if rr["rows"].get(cell, {}).get("median_ms") is not None]
    if not values:
        return None
    #: the clock state each round ran in, off the device; the rounds' values, not one
    #: number, because the state changes between rounds too.
    ghz = [rr["rows"][cell].get("sm_ghz") for rr in round_rows
           if rr["rows"].get(cell, {}).get("median_ms") is not None]
    return {"median_of_round_medians_ms": statistics.median(values),
            "round_medians_ms": values, "n_rounds": len(values), "round_sm_ghz": ghz}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True,
                    help="a registered bench (benchmarks/bench/subjects.py)")
    ap.add_argument("--cells", required=True,
                    help="comma-separated names from benchmarks/cells")
    ap.add_argument("--binary", action="append", default=[], type=parse_binary_spec,
                    help="worktree:PATH,venv:PATH[,label:NAME]; repeatable")
    ap.add_argument("--reps", type=int, default=21)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--lock", default=GPU_LOCK_DEFAULT)
    ap.add_argument("--ncu", action="store_true")
    ap.add_argument("--schedule", default="box",
                    help="the carry family's per-arm runtime dial: union | sparse | "
                         "sparse-gN (N leaves). Worker-side; the orchestrator reads it off "
                         "each --binary spec.")
    ap.add_argument("--state", choices=STATE_SHAPES, default="fresh",
                    help="fresh binds an exit plane over a zero entry state; "
                         "continuation passes the SAME plane in and out")
    ap.add_argument("--out", default=None, help="path for one JSON row per (binary, cell)")
    #: Recording is ON by default -- a probe run IS a measurement, and the record's whole
    #: point is that nothing measured is lost to a terminal scrollback.
    ap.add_argument("--no-record", action="store_true",
                    help="do not write benchmarks/results/ rows (a scratch run)")
    ap.add_argument("--stage", default=None, help="the queue stage this probe belongs to")
    ap.add_argument("--agent", default=None, help="who ran it (a name, not an identity)")
    ap.add_argument("--purpose", default=None, help="free text: what this probe was for")
    # worker-mode-only flags (this file invoked as a subprocess of itself):
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--oneshot", action="store_true", help=argparse.SUPPRESS)
    return ap


def record(binaries, results, args) -> None:
    from bench import store

    for binary in binaries:
        entries = [r for r in results if r["binary"] == binary["label"]]
        if not entries:
            continue
        head = entries[0]
        env = {"gpu": head.get("device_name") or "unknown",
               "sm": head.get("sm") or "sm_unknown",
               "driver_cuda": head.get("driver_cuda"), "torch": head.get("torch"),
               "ptxas": head.get("ptxas")}
        try:
            run = store.build_run(
                harness_paths=[THIS_FILE], stage=args.stage, agent=args.agent,
                purpose=args.purpose, env=env, git_sha=binary["git_sha"],
                dirty=git_dirty(binary["worktree"]), branch=git_branch(binary["worktree"]),
                worktree_name=Path(binary["worktree"]).name,
                stamp_magic=head.get("manifest_sha256"))
        except ValueError as ex:
            print(f"record REFUSED for {binary['label']}: {ex}", file=sys.stderr)
            continue
        rows = []
        for entry in entries:
            if "error" in entry:
                continue
            #: The committed record never carries this box's absolute paths: the
            #: worktree BASENAME is already stamped by `build_run`.
            extra = dict(entry)
            extra["worktree"] = Path(entry["worktree"]).name
            extra["venv"] = Path(entry["venv"]).name
            if extra.get("rola_file"):
                extra["rola_file"] = str(Path(entry["rola_file"]).resolve().relative_to(
                    Path(entry["worktree"]).resolve()))
            rows.append(store.build_result(
                run, kind="probe", cell=entry["cell"], arm=args.bench,
                metric="duration_median_ms", value=entry["median_of_round_medians_ms"],
                unit="ms", extra=extra))
        print(f"record: {binary['label']} -> {store.write_run(run, rows)} "
              f"({len(rows)} result row(s))")


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    if args.worker:
        worker_main(args)
        return

    from bench.driver import CELLS
    from bench.subjects import SUBJECTS

    if args.bench not in SUBJECTS:
        ap.error(f"unknown bench {args.bench!r}; registered: {sorted(SUBJECTS)}")
    cells = args.cells.split(",")
    for cell in cells:
        if cell not in CELLS:
            ap.error(f"unknown cell {cell!r}; registered: {sorted(CELLS)}")
    if not args.binary:
        ap.error("need at least one --binary worktree:PATH,venv:PATH")

    subject = SUBJECTS[args.bench]
    print(f"BENCH {args.bench}: {subject.what}")
    print(f"CELLS: {cells}")
    for binary in args.binary:
        binary["git_sha"] = git_sha(binary["worktree"])
        print(f"BINARY {binary['label']}: worktree={binary['worktree']} "
              f"venv={binary['venv']} sha={binary['git_sha']} "
              f"schedule={binary.get('schedule', 'box')}")
    print(f"STATE ARM: {args.state}")

    round_rows = interleaved_rounds(args.binary, args.bench, cells, args.reps,
                                    args.warmup, args.lock, args.rounds, args.state)
    results = []
    print()
    print(f"{'binary':<16}{'cell':<26}{'ms (median-of-medians)':<26}{'stamp':<12}{'sha':<10}")
    for binary in args.binary:
        rounds = round_rows[binary["label"]]
        tree = tree_digest(binary["worktree"])
        stamp = rounds[0]["stamp"] if rounds else {}
        refused = rounds[0].get("refused") if rounds else None
        if refused:
            print(f"{binary['label']:<16}REFUSED: {refused}")
        for cell in cells:
            summary = summarize(rounds, cell)
            first = rounds[0]["rows"].get(cell, {}) if rounds else {}
            entry = {"binary": binary["label"], "worktree": binary["worktree"],
                     "venv": binary["venv"], "git_sha": binary["git_sha"], "tree_sha256": tree,
                     "cell": cell,
                     "bench": args.bench, "state_arm": args.state,
                     "schedule": binary.get("schedule", "box"),
                     "reps": args.reps, "warmup": args.warmup, "n_rounds": args.rounds,
                     **{k: stamp.get(k) for k in
                        ("device_name", "device_uuid", "rola_file", "manifest_sha256",
                         "sm", "driver_cuda", "torch", "ptxas", "family_stamp")}}
            if summary:
                entry.update(summary)
                shown = f"{summary['median_of_round_medians_ms']:.4f}"
            else:
                entry["error"] = first.get("error", refused or "no successful rounds")
                shown = "ERROR: " + str(entry["error"])[:20]
            results.append(entry)
            ghz = [g for g in (summary or {}).get("round_sm_ghz", []) if g]
            clock = f"{min(ghz):.2f}-{max(ghz):.2f}GHz" if ghz else "GHz n/a"
            print(f"{binary['label']:<16}{cell:<26}{shown:<26}"
                  f"{str(entry['family_stamp'])[:10]:<12}{binary['git_sha'][:8]:<10}"
                  f"{clock:<14}{' DIRTY ' + tree[:8] if tree else ''}")

    if args.ncu:
        print(f"\nNCU (one oneshot launch per binary x cell, filtered to {subject.symbol})")
        prefix = args.out.rsplit(".", 1)[0] if args.out else "/tmp/probe_cells_ncu"
        for binary in args.binary:
            for cell in cells:
                row = run_ncu_cell(binary, args.bench, subject.symbol, cell, args.lock,
                                   prefix, args.state)
                for result in results:
                    if result["binary"] == binary["label"] and result["cell"] == cell:
                        result["ncu"] = row
                print(f"  ncu {binary['label']}/{cell}: kernel={row.get('kernel_name')}")

    if args.out:
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in results))
        print(f"\nwrote {len(results)} rows to {args.out}")
    if not args.no_record:
        record(args.binary, results, args)


if __name__ == "__main__":
    main()
