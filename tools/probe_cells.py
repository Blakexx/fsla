# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""tools/probe_cells.py -- THE CANONICAL K35 CARRY-FAMILY MEASUREMENT HARNESS.

One script, one measurement definition, checked in, so every stage/bisect/ruling
number comes from the same code.  Built per `BRIEF_PROBE_harness.md`
(`rola-scratch/prototypes/k35_final_build/`) to close two recorded harness
discrepancies:

  * R-B vs K36's own harness: ~12% apart on an UNMOVED control.  Root cause
    (`## K36`, section "the reschedule costs nothing"): run-to-run drift on this
    host is ~12% at that cell, and the two numbers came from SEPARATE (non-
    interleaved) runs.  Fix here: A/B binaries are measured INTERLEAVED, three
    alternating rounds, never "all of A then all of B" (`interleaved_rounds`).
  * R-C vs R-B: the record cell (arm 3, W=512) read +42% high.  Root cause
    (`## R-C`, "A BASELINE CORRECTION"): K36's own TU split RESCHEDULED that
    arm's SASS (proven byte-for-byte in `## K36`'s "BYTE-IDENTITY" section) --
    a real codegen difference between git tips, not a harness bug.  This
    harness cannot paper over that; it reports the git SHA of the measured
    binary precisely so a future reader can tell the two apart instead of
    silently averaging them.

Design points forced by that history:

  * THE STEP is exactly what PROBE-K35 measured (`## PROBE-K35`): "the one
    `ext.carry_forward_inter` call = 1x `chunk_facts_tables` + 1x
    `carry::carry_kernel`, PACKING OUTSIDE."  Packing (`pack_side`, the
    gate-plane permute/reshape, the value cast) happens ONCE per cell before
    the timed loop; the CUDA-event span wraps ONLY the raw extension call.
    `rola.ops.carry.carry_inter` (the python wrapper) does its own packing
    EVERY call and must never be timed directly -- R-B/R-C's own probes did,
    which is a plausible second contributor to their spread this harness does
    not repeat.
  * NEVER default dispatch (brief step 2(1)).  A cell names a DECLARATION
    (topology, k, m, d_v, window, chunk, level_modes); which built CONTAINER
    (`carve`) serves it is resolved in PYTHON, before every call, by scanning
    this binary's OWN `rola.ops.carry.arms()` census for an EXACT match in
    `CARVE_PREFS` order and refusing if none exists.  This matters because the
    kernel-side "law 3" fallback that used to serve an unbuilt declaration's
    DENSE_BOTH body silently (`## BISECT (corrected)`'s nine-stage 5x miss) is
    only REMOVED on the current tip (`## R-B` "THE SEAM"); mainline and M1a
    are read-only history and still carry the old silent fallback in their
    C++, so this harness's own pre-flight arm resolution is the only thing
    standing between a mismatched request and a wrong-body measurement on
    those binaries.
  * GPU work is serialized under `flock` and measurement code is IDENTICAL
    across git tips: this file is invoked by each worktree's OWN venv python
    (which resolves `import rola` to that worktree's editable install), so the
    only thing that differs between a "mainline" run and a "k31-formb3" run of
    this same file is which compiled extension answers `rola.ops.carry`.

See `docs/KERNEL_STANDARDS.md` SS16 (perf probe every stage gate) and its three
addenda (instructions are necessary but not sufficient -- carry IPC and stall
reasons beside them; the compile ledger, not touched by this tool).
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import io
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
GPU_LOCK_DEFAULT = "/tmp/rola_gpu.lock"
NCU_BIN = "/usr/local/cuda/bin/ncu"

# ---------------------------------------------------------------------------
# THE FIXED CELL SET (brief step 2(1)).
#
# Topology is the flagship everywhere: D=2, B=256, K=4, M=8.  `window` is the
# carry kernel's `W` argument; `T` is the total token count drawn (a cell's
# `L`).  `level_modes` is the packed per-level declaration word
# (`rola.ops.intra.pack_modes`; DENSE_BOTH=0 on every level for "dense", the
# alternating WRITE_SPARSE/READ_SPARSE word (=6 at D=2) for every ALT cell --
# see `alt_cell` in `tests/k31/test_carry_vs_oracle.py`, mirrored below).
# Seeds match the R-C/R-B probes' own cell table (`rc/probe.py` in the session
# scratchpad) so this harness's draws are byte-identical to theirs where the
# same k_tok/cohort/seed triple was already in use.
CHUNK = 32
CELL_SPECS = {
    "dense": dict(widths=(256, 256), k=4, m=8, d_v=64, window=384, T=768,
                  k_tok=None, cohort=None, seed=5, level_modes=0),
    "k16_w384": dict(widths=(256, 256), k=4, m=8, d_v=64, window=384, T=768,
                      k_tok=16, cohort=None, seed=21, level_modes=6),
    "k16_w512_record": dict(widths=(256, 256), k=4, m=8, d_v=64, window=512, T=16384,
                              k_tok=16, cohort=None, seed=22, level_modes=6),
    "k4_w512_record": dict(widths=(256, 256), k=4, m=8, d_v=64, window=512, T=16384,
                             k_tok=4, cohort=None, seed=23, level_modes=6),
    # Off by default (brief step 2(1)): pass explicitly via --cells to include it.
    "k16_cohort64": dict(widths=(256, 256), k=4, m=8, d_v=64, window=384, T=768,
                           k_tok=16, cohort=64, seed=24, level_modes=6),
}
DEFAULT_CELLS = ["dense", "k16_w384", "k16_w512_record", "k4_w512_record"]

#: Container ("carve") resolution priority: try the shipped K35 1x8-wps2
#: container FIRST (the design this whole family exists to measure), then the
#: universal base container every era since before K31 builds (LEAF_SPLIT=1).
#: Numeric, not `rola.ops.carry.LEAF_SPLIT`/`CONTAINER_1X8_SPLIT`, because
#: those constants are `None` on mainline's build (`## R-C`'s probe confirms
#: the numbering is stable across every era measured here even where the
#: python module does not export names for it).
CARVE_PREFS = (2, 1)


# ---------------------------------------------------------------------------
# THE DRAW.  Inlined rather than imported from `tests/k31/fixtures.py` /
# `test_carry_vs_oracle.py` ON PURPOSE (cross-tip comparison item (5) of the
# brief): those test modules can and do drift per git tip, and importing
# whichever version a given worktree happens to carry would silently break
# "identical measurement code across git tips."  This is a frozen mirror of
# `Cell.__init__` (dense) and `alt_cell` (the alternating declaration) as of
# k31-formb3 `7f77caf`; kept in exact RNG-stream order so a cell's draw is
# reproducible from its name alone, matching the fixtures' own contract.
def simplex(shape, k_tok, gen, device):
    x = torch.rand(shape, device=device, dtype=torch.float64, generator=gen)
    if k_tok is not None and k_tok < shape[-1]:
        keep = torch.zeros(shape, device=device, dtype=torch.float64)
        idx = torch.argsort(torch.rand(shape, device=device, generator=gen),
                             dim=-1)[..., :k_tok]
        keep.scatter_(-1, idx, 1.0)
        x = x * keep
    return x / x.sum(-1, keepdim=True)


def clustered(shape, k_tok, cohort, gen, device):
    B, T, H, width = shape
    x = torch.rand(shape, device=device, dtype=torch.float64, generator=gen)
    starts = torch.randint(0, width, (B, T // cohort, 1, 1), device=device, generator=gen)
    idx = (starts.expand(B, T // cohort, cohort, 1).reshape(B, T, 1, 1)
           + torch.arange(k_tok, device=device).view(1, 1, 1, k_tok)) % width
    keep = torch.zeros(shape, device=device, dtype=torch.float64)
    keep.scatter_(-1, idx.expand(B, T, H, k_tok), 1.0)
    x = x * keep
    return x / x.sum(-1, keepdim=True)


def draw_cell(spec, device="cuda"):
    """``(read_bf, write_bf, g_bf, v_bf)`` -- the packable, un-packed operands.

    Dense cells (``k_tok is None``) draw ``read``/``write`` directly off one
    generator, exactly as ``Cell.__init__``.  ALT cells first draw-and-discard
    a dense pass (``Cell(k_tok=None)``'s own RNG consumption, which is what
    ``alt_cell`` does before it overwrites ``read``/``write`` off a FRESH
    same-seeded generator) so the ``g``/``v`` draws land on the RNG stream
    position ``alt_cell`` itself produces them at, then redraws the actual
    alternating support off the fresh stream.  Level ``l`` is READ-sparse when
    ``l`` is odd and WRITE-sparse when ``l`` is even (the ruling `alt_cell`
    documents: "EVEN levels write-sparse/read-dense, ODD levels
    read-sparse/write-dense").
    """
    widths, T, d_v = spec["widths"], spec["T"], spec["d_v"]
    seed, k_tok, cohort = spec["seed"], spec["k_tok"], spec["cohort"]
    shape = (1, T, 1)
    gen = torch.Generator(device=device).manual_seed(seed)
    read0 = tuple(simplex(shape + (w,), None, gen, device) for w in widths)
    write0 = tuple(simplex(shape + (w,), None, gen, device) for w in widths)
    g = torch.rand(*shape, device=device, dtype=torch.float64, generator=gen) + 0.5
    v = torch.randn(*shape, d_v, device=device, dtype=torch.float64, generator=gen)
    if k_tok is None:
        read, write = read0, write0
    else:
        gen2 = torch.Generator(device=device).manual_seed(seed)

        def draw_side(width, sparse):
            if not sparse:
                return simplex(shape + (width,), None, gen2, device)
            if cohort:
                return clustered(shape + (width,), k_tok, cohort, gen2, device)
            return simplex(shape + (width,), k_tok, gen2, device)

        read = tuple(draw_side(w, l % 2 == 1) for l, w in enumerate(widths))
        write = tuple(draw_side(w, l % 2 == 0) for l, w in enumerate(widths))
    return (tuple(t.to(torch.bfloat16) for t in read),
            tuple(t.to(torch.bfloat16) for t in write),
            g.to(torch.bfloat16), v.to(torch.bfloat16))


# ---------------------------------------------------------------------------
# ARM RESOLUTION -- never default dispatch (brief step 2(1)/(5)).
def _arm_schema(row):
    """Interpret one `arms()` row by its FIELD COUNT, which is stable within
    an era and has changed exactly twice on this ancestry (`## R-C`'s probe
    confirms all three shapes below by direct inspection of the three
    reference binaries this tool validates against):

      * 13 fields (current, k31-formb3 since S3b): D,B,K,M,DV,W,C,CV,MO,thr,cta,smem,xch
      * 12 fields (M1a-era, pre-container-dial): D,B,K,M,W,C,CV,MO,thr,cta,smem,xch (DV implicit=64)
      * 10 fields (mainline-era, pre-K35 declarations): D,B,K,M,W,C,CV,thr,cta,smem
        (no MO -- the kernel is data-adaptive, no declaration argument exists;
        no DV -- d_v was not yet part of the arm key)

    Returns ``(D, B, K, M, DV, W, C, CV, MO)`` with ``MO=None`` where the era
    has no declaration word.  Raises rather than guess on an unknown shape.
    """
    n = len(row)
    if n == 13:
        rd, rb, rk, rm, rdv, rw, rc, rcv, rmo = row[:9]
    elif n == 12:
        rd, rb, rk, rm, rw, rc, rcv, rmo = row[:8]
        rdv = 64
    elif n == 10:
        rd, rb, rk, rm, rw, rc, rcv = row[:7]
        rdv, rmo = 64, None
    else:
        raise RuntimeError(
            f"unrecognized carry arms() row shape ({n} fields): {row!r} -- "
            "teach _arm_schema this era before trusting it (never guess)")
    return rd, rb, rk, rm, rdv, rw, rc, rcv, rmo


def resolve_arm(carry_ops, widths, k, m, d_v, window, chunk, level_modes):
    """The declaration's EXACT built arm, or a refusal.

    Scans this binary's own device-reported `arms()` census (never a
    hardcoded arm index) for a row matching the requested topology/k/m/d_v/
    window/chunk exactly, trying `CARVE_PREFS` in order, and requiring an
    exact `level_modes` match wherever this era's schema carries a MO field
    at all.  Returns ``(carve, arm_index, arm_row)``.  Raises ``RuntimeError``
    (the brief's "REFUSE") if no candidate carve is built for this
    declaration on this binary -- it never falls through to whatever the
    kernel's own dispatch would otherwise pick, because mainline/M1a-era C++
    still contains the silent "unbuilt declaration served DENSE_BOTH" law-3
    fallback that `## R-B` removed only on the current tip.
    """
    rows = carry_ops.arms()
    D = len(widths)
    by_carve = {}
    for i, row in enumerate(rows):
        rd, rb, rk, rm, rdv, rw, rc, rcv, rmo = _arm_schema(row)
        if (rd, rb, rk, rm, rw, rc) != (D, widths[0], k, m, window, chunk):
            continue
        if rdv != d_v:
            continue
        if rmo is not None and rmo != level_modes:
            continue
        by_carve[rcv] = (i, row)
    for cv in CARVE_PREFS:
        if cv in by_carve:
            i, row = by_carve[cv]
            return cv, i, row
    raise RuntimeError(
        f"REFUSED: no built arm matches widths={widths} k={k} m={m} d_v={d_v} "
        f"window={window} chunk={chunk} level_modes={level_modes} at any carve "
        f"in CARVE_PREFS={CARVE_PREFS}. This binary's arms(): {rows}")


# ---------------------------------------------------------------------------
# THE CALL -- packing OUTSIDE the timed span; `level_modes` presence probed
# once (mainline's `carry_forward_inter` has no such parameter at all -- it
# predates per-level declarations, confirmed by inspecting all three
# reference binaries' signatures).
def make_call(ext, pr, pw, gw, v_bf, widths, k, m, window, chunk, carve, level_modes):
    args9 = (pr, pw, gw, v_bf, list(widths), int(k), int(m), int(window), int(chunk), int(carve))
    try:
        ext.carry_forward_inter(*args9, level_modes, None)
        takes_modes = True
    except TypeError:
        takes_modes = False
    if takes_modes:
        return (lambda: ext.carry_forward_inter(*args9, level_modes, None)), True
    return (lambda: ext.carry_forward_inter(*args9, None)), False


def pack_operands(carry_ops, read_bf, write_bf, g_bf, v_bf):
    pr = carry_ops.pack_side(read_bf)
    pw = carry_ops.pack_side(write_bf)
    gw = g_bf.permute(0, 2, 1).contiguous()
    gw = gw.reshape(gw.shape[0] * gw.shape[1], -1)
    v = v_bf.contiguous()
    return pr, pw, gw, v


# ---------------------------------------------------------------------------
# WORKER MODE: runs IN the target binary's own venv python, in its own
# process (brief step 2(5): "each binary in its OWN subprocess ... using that
# worktree's venv python").  Emits one JSON object to stdout.
def worker_main(args):
    global torch
    import torch  # noqa: PLC0415 -- deferred so --help works without a GPU env

    import rola
    from rola.ops import carry as carry_ops
    from rola.ops._ext import extension

    ext = extension()
    device_name = torch.cuda.get_device_name(0)
    try:
        device_uuid = str(torch.cuda.get_device_properties(0).uuid)
    except Exception:
        device_uuid = None
    stamp = {
        "rola_file": rola.__file__,
        "device_name": device_name,
        "device_uuid": device_uuid,
    }
    rows = {}
    cell_names = args.cells.split(",")
    for name in cell_names:
        spec = CELL_SPECS[name]
        try:
            carve, arm_index, arm_row = resolve_arm(
                carry_ops, spec["widths"], spec["k"], spec["m"], spec["d_v"],
                spec["window"], CHUNK, spec["level_modes"])
            read_bf, write_bf, g_bf, v_bf = draw_cell(spec)
            pr, pw, gw, v = pack_operands(carry_ops, read_bf, write_bf, g_bf, v_bf)
            call, took_modes = make_call(ext, pr, pw, gw, v, spec["widths"], spec["k"],
                                          spec["m"], spec["window"], CHUNK, carve,
                                          spec["level_modes"])
            for _ in range(args.warmup):
                call()
            torch.cuda.synchronize()
            if args.oneshot:
                call()
                torch.cuda.synchronize()
                rows[name] = {"oneshot": True, "resolved_carve": carve, "arm_index": arm_index,
                              "arm_row": list(arm_row), "level_modes_applied": took_modes}
                continue
            ts = []
            for _ in range(args.reps):
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                call()
                e1.record()
                torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1))
            rows[name] = {
                "median_ms": statistics.median(ts),
                "min_ms": min(ts), "max_ms": max(ts), "reps": args.reps,
                "warmup": args.warmup,
                "resolved_carve": carve, "arm_index": arm_index, "arm_row": list(arm_row),
                "level_modes_applied": took_modes,
            }
            del read_bf, write_bf, g_bf, v_bf, pr, pw, gw, v
            torch.cuda.empty_cache()
        except Exception as ex:  # noqa: BLE001 -- reported per-cell, never swallowed
            rows[name] = {"error": f"{type(ex).__name__}: {ex}"}
    print(json.dumps({"stamp": stamp, "rows": rows}))
    if any("error" in r for r in rows.values()):
        sys.exit(1)


# ---------------------------------------------------------------------------
# ORCHESTRATOR: cross-tip subprocess launch, GPU flock, interleaved A/B.
def parse_binary_spec(spec: str) -> dict:
    """``worktree:/path,venv:/path[,label:NAME]`` -> dict."""
    out = {}
    for part in spec.split(","):
        k, _, v = part.partition(":")
        out[k.strip()] = v.strip()
    if "worktree" not in out or "venv" not in out:
        raise argparse.ArgumentTypeError(
            f"--binary needs worktree:PATH,venv:PATH (got {spec!r})")
    out.setdefault("label", Path(out["worktree"]).name)
    return out


def git_sha(worktree: str) -> str:
    try:
        return subprocess.run(["git", "-C", worktree, "rev-parse", "HEAD"],
                               capture_output=True, text=True, check=True
                               ).stdout.strip()
    except Exception as ex:  # noqa: BLE001
        return f"UNKNOWN ({ex})"


def venv_python(venv_path: str) -> str:
    py = Path(venv_path) / "bin" / "python"
    if not py.exists():
        raise FileNotFoundError(f"no python at {py}")
    return str(py)


def run_worker(binary: dict, cells: list[str], reps: int, warmup: int, lock_path: str,
                oneshot: bool = False, timeout: int = 600) -> dict:
    py = venv_python(binary["venv"])
    cmd = [py, str(THIS_FILE), "--worker", "--cells", ",".join(cells),
           "--reps", str(reps), "--warmup", str(warmup)]
    if oneshot:
        cmd.append("--oneshot")
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        proc = subprocess.run(cmd, cwd=binary["worktree"], capture_output=True, text=True,
                               timeout=timeout)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    if proc.returncode not in (0, 1):  # 1 = a cell REFUSED, still emits partial JSON
        raise RuntimeError(f"worker for {binary['label']} crashed (rc={proc.returncode}):\n"
                            f"stdout={proc.stdout}\nstderr={proc.stderr}")
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as ex:  # noqa: BLE001
        raise RuntimeError(f"worker for {binary['label']} produced no JSON "
                            f"(rc={proc.returncode}): stdout={proc.stdout!r} "
                            f"stderr={proc.stderr!r}") from ex
    return payload


def run_ncu_cell(binary: dict, cell: str, lock_path: str, out_csv_prefix: str,
                  timeout: int = 900) -> dict:
    """One `ncu` oneshot launch: kernel name + metrics for ``cell`` on ``binary``.

    Mirrors the session scratchpad's `ledger/run_ledger.sh` pattern -- one
    `-c 1` capture on a regex-matched `carry.*kernel`, CSV out, under the
    SAME flock every wall-clock run uses.
    """
    py = venv_python(binary["venv"])
    metrics = ",".join([
        "gpu__time_duration.sum",
        "smsp__inst_executed.sum",
        "sm__inst_executed.avg.per_cycle_active",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
        "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
        "dram__bytes.sum",
        "lts__t_bytes.sum",
    ])
    out_csv = f"{out_csv_prefix}_{binary['label']}_{cell}.csv"
    cmd = [NCU_BIN, "--target-processes", "all", "-k", "regex:carry.*kernel", "-c", "1",
           "--print-kernel-base", "mangled", "--metrics", metrics, "--csv",
           py, str(THIS_FILE), "--worker", "--cells", cell, "--oneshot"]
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        proc = subprocess.run(cmd, cwd=binary["worktree"], capture_output=True, text=True,
                               timeout=timeout)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    Path(out_csv).write_text(proc.stdout)
    row = {"csv": out_csv, "rc": proc.returncode}
    # ncu's stdout interleaves "==PROF==" progress lines and this worker's own
    # one-line JSON stamp BEFORE the real CSV header; find the header by its
    # known first column rather than assuming line 0 is it.
    lines = proc.stdout.splitlines()
    header_i = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
    if header_i is not None:
        csv_text = "\n".join(lines[header_i:])
        reader = csv.DictReader(io.StringIO(csv_text))
        kernel_name = None
        metric_vals = {}
        for r in reader:
            kernel_name = r.get("Kernel Name", kernel_name)
            mname, mval = r.get("Metric Name"), r.get("Metric Value")
            if mname:
                metric_vals[mname] = mval
        row["kernel_name"] = kernel_name
        row["metrics"] = metric_vals
    else:
        row["stderr"] = proc.stderr[-2000:]
        row["stdout_tail"] = "\n".join(lines[-10:])
    return row


def interleaved_rounds(binaries: list[dict], cells: list[str], reps: int, warmup: int,
                        lock_path: str, rounds: int) -> dict:
    """A,B,A,B,A,B (never "all of A then all of B") -- the K36 lesson
    (`## K36`, "the reschedule costs nothing" section: separated runs read a
    1.12-1.37x false regression on this host; only interleaved same-cell A/B
    separates a real schedule change from thermal/contention drift)."""
    per_binary_round_rows = {b["label"]: [] for b in binaries}
    for r in range(rounds):
        for b in binaries:
            payload = run_worker(b, cells, reps, warmup, lock_path)
            per_binary_round_rows[b["label"]].append({"round": r, **payload})
    return per_binary_round_rows


def summarize(round_rows: list[dict], cell: str) -> dict | None:
    vals = []
    for rr in round_rows:
        row = rr["rows"].get(cell)
        if row and "median_ms" in row:
            vals.append(row["median_ms"])
    if not vals:
        return None
    return {"median_of_round_medians_ms": statistics.median(vals),
            "round_medians_ms": vals, "n_rounds": len(vals)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default=",".join(DEFAULT_CELLS),
                     help="comma-separated cell names, e.g. dense,k16_w384")
    ap.add_argument("--binary", action="append", default=[], type=parse_binary_spec,
                     help="worktree:/path,venv:/path[,label:NAME]; repeatable")
    ap.add_argument("--reps", type=int, default=21)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--lock", default=GPU_LOCK_DEFAULT)
    ap.add_argument("--ncu", action="store_true")
    ap.add_argument("--out", default=None, help="path for one JSON row per (binary, cell)")
    # worker-mode-only flags (invoked as a subprocess of itself):
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--oneshot", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        worker_main(args)
        return

    cells = args.cells.split(",")
    for c in cells:
        if c not in CELL_SPECS:
            ap.error(f"unknown cell {c!r}; known cells: {sorted(CELL_SPECS)}")
    if not args.binary:
        ap.error("need at least one --binary worktree:PATH,venv:PATH")

    print(f"CELLS: {cells}")
    for b in args.binary:
        b["git_sha"] = git_sha(b["worktree"])
        print(f"BINARY {b['label']}: worktree={b['worktree']} venv={b['venv']} "
              f"sha={b['git_sha']}")

    round_rows = interleaved_rounds(args.binary, cells, args.reps, args.warmup,
                                     args.lock, args.rounds)

    results = []
    print()
    print(f"{'binary':<16}{'cell':<20}{'ms (median-of-medians)':<26}{'carve':<7}"
          f"{'arm_idx':<9}{'sha':<10}")
    for b in args.binary:
        rr = round_rows[b["label"]]
        stamp = rr[0]["stamp"] if rr else {}
        for cell in cells:
            summ = summarize(rr, cell)
            row0 = rr[0]["rows"].get(cell, {}) if rr else {}
            entry = {
                "binary": b["label"], "worktree": b["worktree"], "venv": b["venv"],
                "git_sha": b["git_sha"], "cell": cell,
                "device_name": stamp.get("device_name"), "device_uuid": stamp.get("device_uuid"),
                "rola_file": stamp.get("rola_file"),
                "resolved_carve": row0.get("resolved_carve"),
                "arm_index": row0.get("arm_index"), "arm_row": row0.get("arm_row"),
                "level_modes_applied": row0.get("level_modes_applied"),
                "reps": args.reps, "warmup": args.warmup, "n_rounds": args.rounds,
            }
            if summ:
                entry.update(summ)
                ms = f"{summ['median_of_round_medians_ms']:.4f}"
            else:
                entry["error"] = row0.get("error", "no successful rounds")
                ms = "ERROR: " + str(entry["error"])[:20]
            results.append(entry)
            print(f"{b['label']:<16}{cell:<20}{ms:<26}{str(entry['resolved_carve']):<7}"
                  f"{str(entry['arm_index']):<9}{b['git_sha'][:8]:<10}")

    if args.ncu:
        print("\nNCU (one oneshot launch per binary x cell; may take a while)...")
        out_prefix = args.out.rsplit(".", 1)[0] if args.out else "/tmp/probe_cells_ncu"
        for b in args.binary:
            for cell in cells:
                ncu_row = run_ncu_cell(b, cell, args.lock, out_prefix)
                for r in results:
                    if r["binary"] == b["label"] and r["cell"] == cell:
                        r["ncu"] = ncu_row
                print(f"  ncu {b['label']}/{cell}: kernel={ncu_row.get('kernel_name')}")

    if args.out:
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nwrote {len(results)} rows to {args.out}")


if __name__ == "__main__":
    main()
