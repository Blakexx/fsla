# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""tools/probe_cells.py -- THE CANONICAL K35 CARRY-FAMILY MEASUREMENT HARNESS.

One script, one measurement definition, checked in, so every stage/bisect/ruling
number comes from the same code.  Built per the coordinator's K35-campaign probe-harness
brief to close two recorded harness discrepancies:

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
  * GPU work is serialized by `tools/gpu_lock.py`'s `gpu_lock()` (K46 --
    this file used to open `/tmp/rola_gpu.lock` with its own ad hoc
    `fcntl.flock`; it now shares the ONE primitive every GPU entry point
    takes) and measurement code is IDENTICAL across git tips: this file is
    invoked by each worktree's OWN venv python (which resolves `import
    rola` to that worktree's editable install), so the only thing that
    differs between a "mainline" run and a "k31-formb3" run of this same
    file is which compiled extension answers `rola.ops.carry`.

See `docs/KERNEL_STANDARDS.md` SS16 (perf probe every stage gate) and its three
addenda (instructions are necessary but not sufficient -- carry IPC and stall
reasons beside them; the compile ledger, not touched by this tool).
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
NCU_BIN = "/usr/local/cuda/bin/ncu"

#: K38: the writer epilogue (`benchmarks/bench/store.py`) is this harness's own
#: identity as data -- `import bench.store` needs `benchmarks/` on `sys.path`, which
#: `benchmarks/*.py`'s own drivers do the same way (`pyproject.toml`'s per-file
#: `E402` exemption for exactly this pattern).
sys.path.insert(0, str(THIS_FILE.parents[1] / "benchmarks"))

# ---------------------------------------------------------------------------
# THE FIXED CELL SET (brief step 2(1)).
#
# Topology is the flagship everywhere: D=2, B=256, K=4, M=8.  `window` is the
# carry kernel's `W` argument; `T` is the total token count drawn (a cell's
# `L`).  `level_modes` is the packed per-level declaration word
# (`rola.ops.intra.pack_modes`; DENSE_BOTH=0 on every level for "dense", the
# alternating WRITE_SPARSE/READ_SPARSE word (=6 at D=2) for every ALT cell --
# see `alt_cell` in `tests/oracle/test_carry_vs_oracle.py`, mirrored below).
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
    #: P80b's IDLE-RESIDENT cell, off by default. STRUCTURED support (each level
    #: truncated to its first `support` fraction of digits) rather than `k_tok`'s
    #: unstructured one, because unstructured sparsity at long `T` still reaches
    #: EVERY atom -- measured: `k16_w384` and `k4_w512_record` both come out
    #: 4096/4096 atoms read AND written, so they have nothing for the activity gate
    #: to skip and price only its overhead. A truncation to a quarter of each
    #: level's digits touches a 1/16 corner of the leaf space, so at full residency
    #: fifteen atoms in sixteen are RESIDENT AND IDLE -- the continuation case.
    "struct25_w384": dict(widths=(256, 256), k=4, m=8, d_v=64, window=384, T=768,
                            k_tok=None, cohort=None, seed=25, level_modes=6,
                            support=0.25),
    #: THE CARVED BODY ON DENSE DATA (K35 F1), off by default.  The ALT DECLARATION --
    #: so the dispatch resolves the CARVED arm and its two-sided carve and transit run
    #: -- over a FULLY DENSE draw (`k_tok=None`), so nothing is skipped and every
    #: per-window fixed cost is paid at its maximum against the largest possible MMA
    #: bill.  KERNEL_STANDARDS section R10 names this the worst case and the test:
    #: dense is not a regime to concede, and `## R-A` measured exactly this shape at
    #: +56% when the transit last landed above its floor.  `seed=26` follows the
    #: existing cells so every one of them keeps its byte-identical draw.
    "alt_dense_w384": dict(widths=(256, 256), k=4, m=8, d_v=64, window=384, T=768,
                             k_tok=None, cohort=None, seed=26, level_modes=6),
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
# THE DRAW.  Inlined rather than imported from `tests/oracle/fixtures.py` /
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
    support = spec.get("support")
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
    if support is not None:
        #: STRUCTURED support, applied to the drawn rows and renormalized -- the
        #: `live=` truncation `tests/oracle/fixtures.py` draws with, mirrored here in
        #: the same after-the-draw position so a cell WITHOUT it keeps its exact
        #: frozen RNG stream (this file's cross-tip contract).
        def truncate(t):
            live = max(1, int(t.shape[-1] * support))
            t = t.clone()
            t[..., live:] = 0.0
            return t / t.sum(-1, keepdim=True)
        read = tuple(truncate(t) for t in read)
        write = tuple(truncate(t) for t in write)
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

      * 11 fields, LAST FIELD == 2 (K50, the runtime-addressing era): the row is the
        arm's SHAPE and nothing else -- D,DV,W,C,warps,WPS,split,BC,smem,cta,schema.
        The topology and the mode word are DATA, so they are NOT in the key: one row
        serves every topology and every declaration of its depth, which is exactly what
        this tool must not mistake for "the arm did not change".  The schema version is
        a FIELD because this row's field count collides with the pre-K35 one below.
      * 13 fields (k31-formb3 since S3b): D,B,K,M,DV,W,C,CV,MO,thr,cta,smem,xch
      * 12 fields (M1a-era, pre-container-dial): D,B,K,M,W,C,CV,MO,thr,cta,smem,xch (DV implicit=64)
      * 10 fields (mainline-era, pre-K35 declarations): D,B,K,M,W,C,CV,thr,cta,smem
        (no MO -- the kernel is data-adaptive, no declaration argument exists;
        no DV -- d_v was not yet part of the arm key)

    Returns ``(D, B, K, M, DV, W, C, CV, MO)`` with ``MO=None`` where the era
    has no declaration word.  Raises rather than guess on an unknown shape.
    """
    n = len(row)
    if n == 11 and row[10] == 2:
        rd, rdv, rw, rc, _warps, _wps, rsplit = row[:7]
        #: `B`, `k` and `m` are not in this era's key at all.  `None` says so, and
        #: `resolve_arm` must not compare them; `CV` becomes the BODY flag.
        return rd, None, None, None, rdv, rw, rc, rsplit, None
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
    runtime_addressing = bool(rows) and len(rows[0]) == 11 and rows[0][10] == 2
    by_carve = {}
    for i, row in enumerate(rows):
        rd, rb, rk, rm, rdv, rw, rc, rcv, rmo = _arm_schema(row)
        if (rd, rw, rc) != (D, window, chunk) or rdv != d_v:
            continue
        if rb is not None and (rb, rk, rm) != (widths[0], k, m):
            continue
        if rmo is not None and rmo != level_modes:
            continue
        by_carve[rcv] = (i, row)
    if runtime_addressing:
        #: THE BODY IS THE CALLER'S NOW, not the arm list's.  Under the older schema a
        #: cell's `level_modes` chose the body THROUGH the arm list -- a DENSE_BOTH
        #: declaration had only a channel-split row built for it, a sparse one only a
        #: carved row.  K50 makes one binary serve every declaration, so the harness
        #: must name the body explicitly, and it names the one the cell's declaration
        #: reached on the parent: `level_modes == 0` is the channel-split control, and
        #: anything else is the carved body.  Getting this wrong would compare two
        #: different bodies across the tips and call the difference a regression.
        want = 0 if level_modes else 1
        if want in by_carve:
            i, row = by_carve[want]
            return want, i, row
        raise RuntimeError(
            f"REFUSED: this binary carries no {'carved' if want == 0 else 'channel-split'} "
            f"body at depth={D} d_v={d_v} window={window} chunk={chunk}. arms(): {rows}")
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
def make_call(ext, pr, pw, gw, v_bf, widths, k, m, window, chunk, carve, level_modes,
              s_out=None, page_tbl=None, atom_bits=None):
    """The timed closure, plus the two arity probes this harness has to make.

    `level_modes` is the older one (mainline predates per-level declarations). The
    SECOND is P80's: the state planes and the page table became THE CALLER'S, so a
    post-P80 binary takes `(s_in, s_out, page_tbl)` where an earlier one took `s_in`
    alone and allocated the exit plane itself. Both are probed rather than assumed,
    because this harness's whole job is comparing binaries across that seam -- and the
    comparison is only like-for-like if the pre-P80 arm's implicit state store is
    matched by a state plane on the post-P80 side, which is why `s_out` defaults to a
    plane the caller allocates rather than to `None` (which would measure the NULL-STATE
    variant against a stateful baseline).
    """
    args9 = (pr, pw, gw, v_bf, list(widths), int(k), int(m), int(window), int(chunk), int(carve))
    #: THE K50 SIGNATURE: `k` and `m` left the seam with the templates, and `carve`
    #: became the BODY flag.  It is probed by ARITY like every other era's -- 13
    #: positional arguments where the previous era took 15 -- so one invoking copy of
    #: this file drives both tips and the A/B stays like-for-like.
    args_rt = (pr, pw, gw, v_bf, list(widths), int(window), int(chunk), int(carve))

    #: EVERY PROBE THAT SUCCEEDS MUST LAUNCH THE CONFIGURATION THE TIMED CALL LAUNCHES.
    #: An arity probe is a real kernel launch, and since P80 the seam carries a single
    #: body whose entry/exit sweeps take a RUNTIME branch on the plane pointers (K41
    #: deleted the earlier `STATE` template-bool axis that made this two SEPARATE
    #: instantiations) -- so a probe made with both planes null runs the SAME kernel
    #: symbol down the null-state branch, and `ncu -c 1`, which captures the FIRST
    #: matching launch, reports that (shorter) branch's metrics for a measurement meant
    #: to be the stateful one. Caught exactly that way at P80, and still a live hazard
    #: post-K41 even though it is no longer a different symbol. A signature that does
    #: not exist raises `TypeError`
    #: out of pybind's overload resolution BEFORE any launch, so an unsuccessful probe
    #: costs nothing and only the successful one -- the richest signature the binary
    #: takes, at this call's own state arm -- ever reaches the device.
    def accepts(*a):
        try:
            ext.carry_forward_inter(*a)
            return True
        except TypeError:
            return False

    #: THE THIRD ARITY IS P80b's: the facts pass' per-atom ACTIVITY byte joined the
    #: seam, and a binary that takes it gates its state sweeps on it. It is probed
    #: like the other two and never assumed, so this harness still drives every
    #: earlier binary; an arm that REQUIRES it (`paged_idle`) refuses a binary that
    #: does not take it, rather than measuring an ungated sweep under that name.
    for tail_bits in (True, False):
        args = (*args_rt, level_modes, None, s_out, page_tbl) + ((atom_bits,) if tail_bits else ())
        if tail_bits and atom_bits is None:
            continue
        if accepts(*args):
            return ((lambda a=args: ext.carry_forward_inter(*a)), True, True, tail_bits)
    for takes_bits, takes_modes, takes_planes in (
            (True, True, True), (False, True, True), (False, False, True),
            (False, True, False), (False, False, False)):
        tail = (level_modes,) if takes_modes else ()
        args = (*args9, *tail, None) + ((s_out, page_tbl) if takes_planes else ())
        args = args + ((atom_bits,) if takes_bits else ())
        if not takes_planes and page_tbl is not None:
            continue
        if takes_bits and atom_bits is None:
            continue
        if accepts(*args):
            return ((lambda a=args: ext.carry_forward_inter(*a)), takes_modes,
                    takes_planes, takes_bits)
    if atom_bits is not None:
        raise RuntimeError(
            "REFUSED: this binary's carry_forward_inter takes no activity bitmap, so "
            "the requested state arm would measure an UNGATED sweep under a gated "
            "arm's name.")
    raise RuntimeError(
        "REFUSED: no accepted carry_forward_inter signature for this binary at state "
        f"arm page_tbl={'set' if page_tbl is not None else 'null'}.")


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
    props = torch.cuda.get_device_properties(0)
    #: K38: the record's provenance fields, read device-side (`BUILD_CONFIG`) rather
    #: than off a path -- a stale `.so` at the right path is the extension trap a
    #: path check cannot catch (`extension-trap-device-side-check`).
    try:
        from rola._build_config import BUILD_CONFIG
        manifest_sha256 = BUILD_CONFIG["manifest_sha256"]
        ptxas = BUILD_CONFIG["ptxas"]
    except Exception:
        manifest_sha256, ptxas = None, None
    stamp = {
        "rola_file": rola.__file__,
        "device_name": device_name,
        "device_uuid": device_uuid,
        "manifest_sha256": manifest_sha256,
        "sm": f"sm_{props.major}{props.minor}",
        "driver_cuda": torch.version.cuda,
        "torch": torch.__version__,
        "ptxas": ptxas,
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
            #: THE STATE ARM (P80). `dense` is the pre-P80 measurement reproduced -- one
            #: dense exit plane, stored -- and is the DEFAULT so a cross-tip A/B stays
            #: like-for-like. `paged` runs the same call at FULL COVERAGE (every atom
            #: resident), which is the arm that prices the slot resolution itself; the
            #: identity table is the WORST case for it, since a permuted one would only
            #: add locality noise on top of the same eight loads. `none` is the
            #: NULL-STATE variant and is comparable only against itself.
            #: `paged_idle` is P80b's arm: the SAME full residency `paged` runs, plus
            #: the facts pass' own activity byte, so every atom this cell's routing
            #: neither reads nor writes is RESIDENT AND IDLE -- the continuation case,
            #: priced against `paged` (identical launch, identical table) so the
            #: difference is the skipped state traffic and nothing else.
            s_out, page_tbl, atom_bits = None, None, None
            if args.state != "none":
                s_out = carry_ops.state_plane(spec["widths"], pr.shape[0], spec["d_v"])
                if args.state in ("paged", "paged_idle"):
                    n_atoms = s_out.shape[1]
                    page_tbl = torch.arange(pr.shape[0] * n_atoms, device="cuda",
                                            dtype=torch.int32).view(pr.shape[0], n_atoms)
                if args.state == "paged_idle":
                    from rola.engine.facts import planes
                    #: the facts entry names the leading axes `[B, H, L, sum]` because
                    #: its grid is per `(bh)`; `pack_side` lays the same bytes out as
                    #: `[BH, L, sum]` for the kernel seam. Every cell here draws `B = 1`
                    #: (`draw_cell`'s `shape`), so the view is `[1, BH, L, sum]`.
                    atom_bits = planes.atom_bits(pw.reshape(1, *pw.shape),
                                                 spec["widths"],
                                                 read_plane=pr.reshape(1, *pr.shape))
            call, took_modes, took_planes, took_bits = make_call(
                ext, pr, pw, gw, v, spec["widths"], spec["k"], spec["m"], spec["window"],
                CHUNK, carve, spec["level_modes"], s_out, page_tbl, atom_bits)
            for _ in range(args.warmup):
                call()
            torch.cuda.synchronize()
            if args.oneshot:
                call()
                torch.cuda.synchronize()
                rows[name] = {"oneshot": True, "resolved_carve": carve, "arm_index": arm_index,
                              "arm_row": list(arm_row), "level_modes_applied": took_modes,
                              "state_arm": args.state, "caller_owned_planes": took_planes,
                              "activity_gated": took_bits}
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
                "state_arm": args.state, "caller_owned_planes": took_planes,
                "activity_gated": took_bits,
            }
            del read_bf, write_bf, g_bf, v_bf, pr, pw, gw, v, s_out, page_tbl, atom_bits
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


def git_dirty(worktree: str) -> bool:
    """K38: whether the MEASURED binary's own tree was clean. A dirty tree does not
    block the measurement but stamps the row advisory, mirroring `bench.ledger`."""
    try:
        out = subprocess.run(["git", "-C", worktree, "status", "--porcelain"],
                             capture_output=True, text=True, check=True).stdout
        return bool(out.strip())
    except Exception:  # noqa: BLE001
        return True


def git_branch(worktree: str) -> str:
    try:
        return subprocess.run(["git", "-C", worktree, "rev-parse", "--abbrev-ref", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception as ex:  # noqa: BLE001
        return f"UNKNOWN ({ex})"


def venv_python(venv_path: str) -> str:
    py = Path(venv_path) / "bin" / "python"
    if not py.exists():
        raise FileNotFoundError(f"no python at {py}")
    return str(py)


def run_worker(binary: dict, cells: list[str], reps: int, warmup: int, lock_path: str,
                oneshot: bool = False, timeout: int = 600, state: str = "dense") -> dict:
    py = venv_python(binary["venv"])
    cmd = [py, str(THIS_FILE), "--worker", "--cells", ",".join(cells),
           "--reps", str(reps), "--warmup", str(warmup), "--state", state]
    if oneshot:
        cmd.append("--oneshot")
    with gpu_lock(lock_path):
        proc = subprocess.run(cmd, cwd=binary["worktree"], capture_output=True, text=True,
                               timeout=timeout)
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
                  timeout: int = 900, state: str = "dense") -> dict:
    """One `ncu` oneshot launch: kernel name + metrics for ``cell`` on ``binary``.

    Mirrors the session scratchpad's `ledger/run_ledger.sh` pattern -- one
    `-c 1` capture on a regex-matched `carry.*kernel`, CSV out, under the
    SAME `gpu_lock()` every wall-clock run uses.
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
        #: THE SHARED-MEMORY LEG (K35 F1).  The carry's on-chip mechanisms -- the panel
        #: ring, the V staging and the exchange's publish/pull -- are all SMEM traffic,
        #: and a layout above its floor shows up as bank conflicts and as MIO pressure
        #: long before it shows up in an instruction count (`## R-A`: the transit that
        #: cost 21% carried +131% bank conflicts on -5.6% instructions).  A stage that
        #: claims max-way-one has to be able to SHOW it on the built binary.
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
        "smsp__inst_executed_op_shared_ld.sum",
        "smsp__inst_executed_op_shared_st.sum",
        "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    ])
    out_csv = f"{out_csv_prefix}_{binary['label']}_{cell}.csv"
    cmd = [NCU_BIN, "--target-processes", "all", "-k", "regex:carry.*kernel", "-c", "1",
           "--print-kernel-base", "mangled", "--metrics", metrics, "--csv",
           py, str(THIS_FILE), "--worker", "--cells", cell, "--oneshot", "--state", state]
    with gpu_lock(lock_path):
        proc = subprocess.run(cmd, cwd=binary["worktree"], capture_output=True, text=True,
                               timeout=timeout)
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
                        lock_path: str, rounds: int, state: str = "dense") -> dict:
    """A,B,A,B,A,B (never "all of A then all of B") -- the K36 lesson
    (`## K36`, "the reschedule costs nothing" section: separated runs read a
    1.12-1.37x false regression on this host; only interleaved same-cell A/B
    separates a real schedule change from thermal/contention drift)."""
    per_binary_round_rows = {b["label"]: [] for b in binaries}
    for r in range(rounds):
        for b in binaries:
            payload = run_worker(b, cells, reps, warmup, lock_path, state=state)
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
    #: THE STATE ARM (P80). `dense` reproduces the pre-P80 measurement (one dense exit
    #: plane, stored) and is the default, so a cross-tip A/B is like-for-like; `paged`
    #: resolves every atom through a FULL-COVERAGE table, which is the arm that prices
    #: the slot resolution; `paged_idle` is that same table plus P80b's activity byte,
    #: so every atom the routing does not touch is resident and IDLE and its state I/O
    #: is skipped -- the continuation case, priced against `paged`; `none` is the
    #: null-state variant, comparable only to itself.
    ap.add_argument("--state", choices=("dense", "paged", "paged_idle", "none"),
                    default="dense")
    ap.add_argument("--out", default=None, help="path for one JSON row per (binary, cell)")
    #: K38: the writer epilogue. Recording is ON by default -- a probe run IS a
    #: measurement, and the record's whole point is that nothing measured is lost
    #: to a terminal scrollback; `--no-record` is for a genuine scratch/dry run.
    ap.add_argument("--no-record", action="store_true",
                     help="do not write benchmarks/results/ rows (a scratch run, not a record)")
    ap.add_argument("--stage", default=None, help="the queue stage this probe belongs to")
    ap.add_argument("--agent", default=None, help="who ran it (a name, not an identity)")
    ap.add_argument("--purpose", default=None, help="free-text: what this probe was for")
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

    print(f"STATE ARM: {args.state}")
    round_rows = interleaved_rounds(args.binary, cells, args.reps, args.warmup,
                                     args.lock, args.rounds, args.state)

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
                "manifest_sha256": stamp.get("manifest_sha256"), "sm": stamp.get("sm"),
                "driver_cuda": stamp.get("driver_cuda"), "torch": stamp.get("torch"),
                "ptxas": stamp.get("ptxas"),
                "resolved_carve": row0.get("resolved_carve"),
                "arm_index": row0.get("arm_index"), "arm_row": row0.get("arm_row"),
                "level_modes_applied": row0.get("level_modes_applied"),
                "state_arm": args.state,
                "caller_owned_planes": row0.get("caller_owned_planes"),
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
                ncu_row = run_ncu_cell(b, cell, args.lock, out_prefix, state=args.state)
                for r in results:
                    if r["binary"] == b["label"] and r["cell"] == cell:
                        r["ncu"] = ncu_row
                print(f"  ncu {b['label']}/{cell}: kernel={ncu_row.get('kernel_name')}")

    if args.out:
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nwrote {len(results)} rows to {args.out}")

    if not args.no_record:
        from bench import store  # noqa: PLC0415 -- deferred, see the sys.path insert above

        for b in args.binary:
            entries = [r for r in results if r["binary"] == b["label"]]
            if not entries:
                continue
            stamp0 = entries[0]
            env = {
                "gpu": stamp0.get("device_name") or "unknown",
                "sm": stamp0.get("sm") or "sm_unknown",
                "driver_cuda": stamp0.get("driver_cuda"),
                "torch": stamp0.get("torch"),
                "ptxas": stamp0.get("ptxas"),
            }
            try:
                run = store.build_run(
                    harness_paths=[THIS_FILE], stage=args.stage, agent=args.agent,
                    purpose=args.purpose, env=env, git_sha=b["git_sha"],
                    dirty=git_dirty(b["worktree"]), branch=git_branch(b["worktree"]),
                    worktree_name=Path(b["worktree"]).name,
                    stamp_magic=stamp0.get("manifest_sha256"))
            except ValueError as ex:
                print(f"K38 record REFUSED for {b['label']}: {ex}", file=sys.stderr)
                continue
            rows = []
            for entry in entries:
                if "error" in entry:
                    continue
                #: §17: the committed record never carries this box's absolute worktree/
                #: venv/extension paths -- `entry` keeps them (useful for a local --out
                #: dump) but `extra_json` gets only the worktree BASENAME (`build_run`
                #: already stamps that as `worktree_name`) and the extension path relative
                #: to its own worktree root.
                extra = dict(entry)
                extra["worktree"] = Path(entry["worktree"]).name
                extra["venv"] = Path(entry["venv"]).name
                if extra.get("rola_file"):
                    extra["rola_file"] = str(
                        Path(entry["rola_file"]).resolve().relative_to(
                            Path(entry["worktree"]).resolve()))
                rows.append(store.build_result(
                    run, kind="probe", cell=entry["cell"], arm=entry.get("resolved_carve"),
                    metric="duration_median_ms", value=entry["median_of_round_medians_ms"],
                    unit="ms", extra=extra))
            out_path = store.write_run(run, rows)
            print(f"K38 record: {b['label']} -> {out_path} ({len(rows)} result row(s))")


if __name__ == "__main__":
    main()
