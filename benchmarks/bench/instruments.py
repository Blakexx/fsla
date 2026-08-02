"""What is measured ABOUT a run, beyond wall time.

Wall time says an arm got faster; it never says why, and a harness that can only
report time invites the story to be supplied afterwards. These instruments compute
the quantities a claim about the kernel's behaviour has to cite:

* :func:`gather_stats` -- the gather's REALIZED extents (``mean_K``, ``mean_KI``,
  k-steps, m-tiles), re-derived from the routing INDEPENDENTLY of the kernel, so it
  corroborates a timing claim instead of restating it.
* :func:`_decay_preamble_ms` -- the decay arm's per-call preamble, isolated, so a
  decay price is not quietly a preamble price.
* :func:`settle_clocks` -- item 0b's clock-settling protocol: what a timing run does
  to the box BEFORE it measures anything, so the measurement is of the kernel and
  not of the clock still ramping.

(``ncu`` counter collection is driven externally and lands here when it is
committed; the design's §6.9 name for this module is "ncu counters, gatherstats".)
"""
from __future__ import annotations

import shutil
import statistics
import subprocess
import threading
import time

import torch

from bench.cells import Cell
from rola.planner.schedule import build_schedule, owner_geometry, unpack_bits
from rola.routing.types import LeafMassDecay

# ---------------------------------------------------------------------------
# CLOCK-SETTLING PROTOCOL (item 0b)
# ---------------------------------------------------------------------------

def _nvidia_smi_clock_mhz(idx: int, field: str) -> float:
    """One ``nvidia-smi`` clock field, in MHz, for device ``idx``.

    Raises rather than returning a sentinel on failure: a settle routine that
    cannot see the clock has to say so (`settle_clocks` catches this and reports
    it), not proceed as though the box were settled.
    """
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits",
         f"--id={idx}"],
        capture_output=True, text=True, timeout=5, check=True)
    return float(out.stdout.strip().splitlines()[0])


def settle_clocks(*, hold_polls=4, plateau_tol_mhz=30.0, poll_interval_s=0.15,
                   timeout_s=12.0, matmul_n=4096) -> dict:
    """Ramp this box to a steady SM clock before ANY latency in this process is
    timed -- the fix for the gate defect `extraction.md` "A GATE DEFECT FOUND ON THE
    WAY" recorded rather than worked around.

    **The defect.** The box idles at ~210-240 MHz of a ~2100 MHz rated max. The
    baseline RECORDER (`bench_regression.py`'s 35 reps * 5 blocks * 7 processes)
    ramps the clock as an incidental side effect of doing that much work; the
    per-commit GATE (`test_latency_regression.py`, 11 reps, one process) does not do
    enough work to ramp it the same way. The two sides of the comparison were
    therefore starting from different clock states, and the two sub-millisecond
    wave-quantized `topo-*` cells (`waves_per_multiprocessor = 0.53`, 128 CTAs over
    80 SMs) are narrow enough for that gap alone to read as a regression.

    **The target is a PLATEAU, not a fraction of the rated max.** The rated max
    clock (`clocks.max.sm`) is a boost ceiling this card does not sustain under
    continuous compute -- measured here at 100 % utilization it settles at
    1710-1740 MHz, ~82 % of the 2100 MHz rating, and holds there (power/thermal
    limited, not still ramping); `extraction.md`'s own "forcing 1740 MHz by hand"
    note independently measured the same ceiling. A threshold expressed as a
    fraction of `clocks.max.sm` is therefore unreachable and times out on every
    call. The correct target is: the clock has STOPPED CHANGING, wherever it stops
    -- so this polls `clocks.sm` while a continuous background load runs, and calls
    it settled once the last `hold_polls` samples all fall within
    `plateau_tol_mhz` of each other.

    **The fix.** A background thread keeps issuing matmuls on the SAME device the
    caller is about to time -- continuously, not start/stop per poll, because a
    stop/query/restart cycle lets the clock relax between bursts and never
    reflects the steady state the timed reps will actually run under. The main
    thread polls `nvidia-smi` (a separate process; it does not contend with the
    GIL the burn thread holds while launching kernels) until the plateau
    criterion holds or `timeout_s` elapses.

    **Reported, never silently skipped.** A box without `nvidia-smi`, or one whose
    query fails, gets ``settled: False`` and a ``reason`` in the returned dict
    instead of the routine pretending it settled a clock it never measured. Callers
    print this dict so a flaky run says whether settling even ran, rather than
    leaving the clock as an unstated variable the way the pre-existing gate did.
    """
    result: dict = {"settled": False}
    if not torch.cuda.is_available():
        result["reason"] = "no cuda device"
        return result
    idx = torch.cuda.current_device()
    if shutil.which("nvidia-smi") is None:
        result["reason"] = "nvidia-smi not found"
        return result
    try:
        max_mhz = _nvidia_smi_clock_mhz(idx, "clocks.max.sm")
    except (subprocess.SubprocessError, OSError, ValueError, IndexError) as exc:
        result["reason"] = f"could not query clocks.max.sm: {exc}"
        return result

    a = torch.randn(matmul_n, matmul_n, device="cuda", dtype=torch.float16)
    b = torch.randn(matmul_n, matmul_n, device="cuda", dtype=torch.float16)
    stop = threading.Event()

    def _burn() -> None:
        while not stop.is_set():
            a @ b  # noqa: B018 -- async launch is the point; the result is unused

    burner = threading.Thread(target=_burn, daemon=True)
    burner.start()
    try:
        samples: list[float] = []
        t0 = time.monotonic()
        cur = 0.0
        while time.monotonic() - t0 < timeout_s:
            time.sleep(poll_interval_s)
            try:
                cur = _nvidia_smi_clock_mhz(idx, "clocks.sm")
            except (subprocess.SubprocessError, OSError, ValueError, IndexError) as exc:
                result["reason"] = f"could not query clocks.sm: {exc}"
                return result
            samples.append(cur)
            window = samples[-hold_polls:]
            if len(window) >= hold_polls and max(window) - min(window) <= plateau_tol_mhz:
                result.update(settled=True, sm_mhz=cur, max_mhz=max_mhz,
                              plateau_mhz=round(max(window) - min(window), 1),
                              elapsed_s=round(time.monotonic() - t0, 3))
                return result
        result.update(reason="timeout", sm_mhz=cur, max_mhz=max_mhz,
                      elapsed_s=round(time.monotonic() - t0, 3))
        return result
    finally:
        stop.set()
        burner.join(timeout=2.0)
        torch.cuda.synchronize()


def gather_stats(cell: Cell, fx=None, max_owners=48):
    """The gather's REALIZED extents, computed independently of the kernel.

    Re-derives §3 step 2's per-leaf support words and the two-pass ballot's counts in
    Python from the schedule's TOKEN bitmaps -- the same object the kernel reads --
    and reports what compaction actually buys per candidate `(owner, tile)`: `K`
    (the readout's k-extent and the fold's m-extent), `KI` (the intra Gram's), and
    the two in the QUANTA the kernel actually moves in, MMA k-steps and warp m-tiles.

    **The quanta are the point.** A cell can have `mean_K = 0.97` of `BC = 64` and
    still cost a full k-step, because `ceil(K/16)` cannot go below 1 on a candidate
    tile. `mean_K` alone overstates the saving; `mean_ksteps` is what the kernel
    pays.

    Owners are SAMPLED (evenly spaced, up to `max_owners`) because the at-scale cell
    has 8180 of them; the sample size is reported with the number rather than
    hidden.
    """
    fx = cell.build() if fx is None else fx
    plan, sch = cell.plan(norm="global", decay_on=False, prec="bf16", fx=fx)
    if sch is None:
        sch = build_schedule(fx["read_f32"], fx["write_f32"], None,
                                fx["g_write_f32"], plan)
    geo = owner_geometry(tuple(cell.widths), cell.BC)
    j, nsl, nw = geo.first_spanned_level, geo.n_slices, sch.n_words
    #: `plan.BT`, NOT `cell.BT`. FIXED 2026-08-01 (item 6b), and it was a real defect
    #: rather than a tidy-up: the bitmaps below belong to `sch`, which is built at
    #: `plan.BT`, while the tile count was taken from the cell's NOMINAL `BT`.
    #: Whenever the planner selected the other tile size the two disagreed and
    #: `unpack_bits` read a tile axis the words do not have. It was reachable and it
    #: was reached: the committed `atscale-L2048-BC16` entry (749436 candidate pairs)
    #: was produced by a `BT = 32` schedule counted at `nt = 2048/16 = 128`. At
    #: `plan.BT == cell.BT` the two forms are identical, which is why every other
    #: cell's entry is unchanged.
    nt = (cell.T + plan.BT - 1) // plan.BT
    widths = tuple(cell.widths)
    m32 = (1 << 32) - 1

    def leaf_digits(level, slice_index):
        """`base + (i / stride) % modulus` for `i in [0, BC)` -- the kernel's map."""
        i = torch.arange(cell.BC, dtype=torch.int64)
        if level < j:
            return torch.zeros(cell.BC, dtype=torch.int64)   # base only; added below
        inner = 1
        for w in widths[level + 1:]:
            inner *= w
        modulus = geo.slice_digits if level == j else widths[level]
        base = slice_index * geo.slice_digits if level == j else 0
        return base + (i // inner) % modulus

    tok_r = sch.read_token_bits.to(torch.int64) & m32     # [BH, n_tiles, total_rows]
    tok_w = sch.write_token_bits.to(torch.int64) & m32
    ru = sch.read_slice_union.view(-1, nsl, nw)
    wu = sch.write_slice_union.view(-1, nsl, nw)
    owners = sch.owners.cpu()
    n_own = int(owners.shape[0])
    step = max(1, n_own // max_owners)
    picks = list(range(0, n_own, step))[:max_owners]
    valid = torch.tensor([min(cell.BT, max(0, cell.T - t * cell.BT)) for t in range(nt)],
                         dtype=torch.int64)
    valid = (1 << valid) - 1                                     # [n_tiles]
    tot_K = tot_KI = tot_ks = tot_mt = 0
    n_cand = 0
    for k in picks:
        bh, ow = int(owners[k, 0]), int(owners[k, 1])
        digits, slice_index = geo.owner_digits(ow)
        r, w = ru[bh, ow % nsl].clone(), wu[bh, ow % nsl].clone()
        for l in range(j):
            row = sch.level_offsets[l] + digits[l]
            r &= sch.read_tile_bits[bh, row]
            w &= sch.write_tile_bits[bh, row]
        cand = (unpack_bits(r, nt).bool() | unpack_bits(w, nt).bool()).cpu()
        # (a') the FIXED levels' AND, as token validity; (b) the per-leaf AND.
        fr = valid.clone()
        fw = valid.clone()
        for l in range(j):
            row = sch.level_offsets[l] + digits[l]
            fr = fr & tok_r[bh, :, row].cpu()
            fw = fw & tok_w[bh, :, row].cpu()
        rw = fr[:, None].repeat(1, cell.BC)
        ww = fw[:, None].repeat(1, cell.BC)
        for l in range(j, len(widths)):
            rows = sch.level_offsets[l] + leaf_digits(l, slice_index)
            rw &= tok_r[bh][:, rows].cpu()
            ww &= tok_w[bh][:, rows].cpu()
        active = (rw | ww) != 0
        first_write = ww & (-ww)
        in_intra = (rw & (-first_write)) != 0
        K = active.sum(1)[cand]
        KI = in_intra.sum(1)[cand]
        n_cand += int(K.numel())
        tot_K += int(K.sum())
        tot_KI += int(KI.sum())
        unit = 16          # bf16 `m16n8k16`
        tot_ks += int(((K + unit - 1) // unit).sum())
        tot_mt += int(((K + 15) // 16).sum())
    d = max(1, n_cand)
    return dict(owners_sampled=len(picks), candidate_pairs=n_cand,
                mean_K=tot_K / d, mean_KI=tot_KI / d,
                mean_ksteps=tot_ks / d, mean_mtiles=tot_mt / d)




def _decay_preamble_ms(cell, fx, decay_temp, reps=9):
    """Median cost of the decay arm's per-call preamble, on this fixture."""
    from rola.planner.schedule import build_clock_stream, leaf_rate_dials, panel_exponent_bound
    decay = LeafMassDecay(dials=fx["dials"][decay_temp])
    plan, _ = cell.plan(norm="global", decay_on=True, prec="bf16",
                        fx=fx)
    wl = fx["write_f32"]
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ca = build_clock_stream(wl, wl, plan)
        leaf_rate_dials(decay, plan.topology).to("cuda")
        panel_exponent_bound(ca, decay, plan.BT)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)

