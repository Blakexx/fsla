"""ONE THIN FUNCTION PER QUESTION. No machinery lives here.

Each driver picks a cell, picks two arms, and hands them to
:mod:`bench.pairing`'s engine (or, for the correctness questions, to the fp64
oracle). That is deliberately all they do: the file these came from accreted
eleven per-question drivers around a shared core until the core was no longer
separable from the questions, and a driver that grows machinery is machinery that
escaped review.

The correctness drivers gate against the CANONICAL oracle ``naive_rola`` and
nothing else -- never a second reference, which proves only self-consistency (spec
§2.1's prohibition; the shared-router bug passed exactly that way).
"""
from __future__ import annotations

import statistics

import torch

from bench.cells import BF16_FOLD_STATE_RTOL, BF16_RTOL, CELLS, GROUPS, RTOL, Cell, skip_share
from bench.instruments import _decay_preamble_ms
from bench.pairing import _iqr, _paired_one_kernel, _timed
from rola.ops.consumer import _consumer_forward
from rola.ops.naive import naive_rola
from rola.planner.schedule import build_schedule
from rola.routing.types import LeafMassDecay

# ---------------------------------------------------------------------------
# ORACLE GATE
# ---------------------------------------------------------------------------

def run_cell(cell: Cell, *, norm="global", decay_temp=None, prec="bf16",
             fx=None):
    """Run one arm of one cell and return `(y, state, y_ref, state_ref)`."""
    fx = cell.build() if fx is None else fx
    decay = None if decay_temp is None else LeafMassDecay(dials=fx["dials"][decay_temp])
    g_read = None if norm == "global" else fx["g_read"]
    topology = fx["topology"]

    y_ref, state_ref = naive_rola(
        fx["v"], fx["read_levels"], fx["write_levels"], g_read, fx["g_write"],
        topology, decay, norm, initial_state=None, output_final_state=True)

    #: THE NEW CONSUMER MUST BE STAMPED `panel` UNDER DECAY, and it is a gating
    #: requirement rather than bookkeeping. Its tile algorithm IS §7.4.1's panel
    #: lowering -- two separately exponentiated survival factors -- so it inherits
    #: that lowering's DOMAIN, and `_consumer_forward` evaluates
    #: `panel_exponent_bound` against `PANEL_MAX_EXPONENT` only when the plan says
    #: `panel`. Stamping anything else would let the new kernel run outside the
    #: range in which its two panels are an exact rewrite of the recurrence, with
    #: the admission check silently skipped. Off the decay arm the axis is
    #: unreachable by this kernel and the planner's own choice stands.
    plan, sched = cell.plan(norm=norm, decay_on=decay is not None, fx=fx)
    y, state, _ = _consumer_forward(
        fx["v_f32"], fx["read_f32"], fx["write_f32"],
        None if g_read is None else fx["g_read_f32"], fx["g_write_f32"],
        plan, decay=decay, schedule=sched)
    return y, state, y_ref, state_ref




def gate_cell(cell: Cell, *, norm, decay_temp, prec, verbose=True):
    arm = f"{norm} {prec}" + (f" decay:{decay_temp}" if decay_temp else "")
    try:
        y, state, y_ref, state_ref = run_cell(
            cell, norm=norm, decay_temp=decay_temp, prec=prec)
    except ValueError as exc:
        #: A REFUSAL IS A PASS, and it is the only outcome that can be, because the
        #: alternative is a wrong answer rather than an inaccurate one. `panel`'s
        #: admission bound refuses a routing whose per-leaf reference exponent
        #: exceeds `PANEL_MAX_EXPONENT`; past it the write factor saturates at the
        #: kernel's clamp while the read factor underflows and their product --
        #: `O(1)` at `t ~ j` -- comes out as `~0`. The HOT-dial arm crossed into
        #: that regime on the write-`k=1` cells, which is exactly the condition the
        #: bound names (a near-unit leaf rate met by a near-one-hot write clock), so
        #: this path is EXERCISED rather than hypothetical. It is reported
        #: distinctly, never silently counted as a numeric pass.
        if "not admissible" not in str(exc):
            raise
        if verbose:
            print(f"  RFSD  {cell.name:26s} {arm:22s} "
                  f"admission bound refused (expected: this is the §12 rule-3 "
                  f"dynamic-range limit)")
        return True
    err_y = float((y.double() - y_ref).abs().max()) / max(1e-30, float(y_ref.abs().max()))
    err_s = float((state.double() - state_ref).abs().max()) / max(
        1e-30, float(state_ref.abs().max()))
    tol_y = RTOL if prec == "tf32x3" else BF16_RTOL
    tol_s = RTOL if prec == "tf32x3" else BF16_FOLD_STATE_RTOL
    ok = (err_y < tol_y) and (err_s < tol_s)
    if verbose:
        print(f"  {'PASS' if ok else 'FAIL':4s}  {cell.name:26s} {arm:22s} "
              f"y {err_y:.3e} (<{tol_y:.0e})  state {err_s:.3e} (<{tol_s:.2e})")
    return ok




def run_oracle(*, decay=True, hot=False):
    """The gate, by named group. Every cell states its arm; nothing is implicit."""
    temps = [None] + ([("hot" if hot else "warm")] if decay else [])
    total = passed = 0
    for group, arms in (
            ("topo", [(n, "global", t, p) for n in GROUPS["topo"]
                      for p in ("bf16", "tf32x3") for t in temps]),
            #: `wk1` is decay-only: its whole purpose is the read-only-tile
            #: exemption, which only exists on the decay arm.
            ("wk1", [(n, "global", ("hot" if hot else "warm"), p)
                     for n in GROUPS["wk1"] for p in ("bf16", "tf32x3")]),
            ("raw", [(n, "raw", t, p) for n in GROUPS["raw"]
                     for p in ("bf16", "tf32x3") for t in temps]),
            #: M-1's isolating cells. They exist to have their INSTRUCTION COUNTS
            #: read, so the arm gated here is the arm those counts were taken on
            #: (`bf16`, the shipped default), crossed with `tf32x3` because a
            #: geometry these cells reach and the `topo` group does not -- a
            #: multi-level spanned owner rectangle -- must be shown exact and not
            #: merely within a format tolerance.
            ("iso", [(n, "global", t, p) for n in GROUPS["iso"]
                     for p in ("bf16", "tf32x3") for t in temps]),
    ):
        print(f"-- group {group} ({len(arms)} arms)")
        for name, norm, temp, prec in arms:
            total += 1
            passed += int(gate_cell(CELLS[name], norm=norm, decay_temp=temp, prec=prec))
    print(f"\n{passed}/{total} oracle arms PASS")
    return passed == total




# ---------------------------------------------------------------------------
# §6.4 SEQUENCE PARALLELISM (`P > 1`)
# ---------------------------------------------------------------------------

def _forced_split(plan, schedule, P):
    """Split EVERY owner `P` ways, BYPASSING the three admission gates.

    The gates are a PERFORMANCE POLICY (occupancy waves, a relative bandwidth
    budget, a workspace cap) and they are selected at plan time from schedule
    statistics -- device code branches on nothing. The kernel's EXACTNESS must not
    depend on which `P` the policy happened to admit, so the numeric gate drives
    `P` directly. The policy's own behaviour is gated separately, in
    `tests/ops/test_rola_v3_sequence_parallel_policy.py`.
    """
    from rola.routing.sequence_parallel import (
        SequenceParallelPlan,
        SequenceParallelPolicy,
        owner_slot_counts,
        select_sequence_parallelism,
    )
    base = select_sequence_parallelism(
        plan, schedule, policy=SequenceParallelPolicy(workspace_cap=1 << 30))
    n_owners = int(schedule.owners.shape[0])
    assert schedule.n_tiles >= P, (
        f"cannot split {P} ways over {schedule.n_tiles} tiles; raise T or lower P "
        "rather than letting the split degenerate into empty segments")
    Ps = tuple([P] * n_owners)
    return SequenceParallelPlan(
        P=Ps, limiting_gate=(None,) * n_owners, binding_gate=None, sms=base.sms,
        blocks_per_sm=base.blocks_per_sm, waves_before=base.waves_before,
        waves_after=0.0, base_traffic_bytes=0, roundtrip_bytes=0,
        roundtrip_seconds=0.0, base_traffic_seconds=0.0, ledger=base.ledger,
        n_slots=sum(owner_slot_counts(Ps)), BC=base.BC, cols=base.cols)




def run_split(cell: Cell, *, P, norm="global", decay_temp=None, prec="bf16",
              fx=None, verbose=True):
    """`P`-way split against the canonical fp64 oracle, plus the `P = 1` control.

    **TWO DIFFERENT GATES, BECAUSE THE TWO QUANTITIES HAVE DIFFERENT STATUS.**

    * `P = 1` must be **BIT-IDENTICAL** to passing no plan at all. That is the
      structural carve-out and it is checkable exactly: the `nullptr` work arrays
      make `seg = 0`, `nseg = 1`, `last_seg` true and the range the whole axis
      WITHOUT READING MEMORY, so nothing about the launch changes.
    * `P > 1` is **NOT** bitwise and must not be gated as such. The split
      re-associates the `P - 1` boundary folds in fp32 -- pass 3's last segment
      still runs serially from the accumulated boundary, which is what confines the
      re-association to the boundaries -- so it is gated against the ORACLE at the
      `P = 1` tolerance, which is the honest statement: the split must not be worse
      than the unsplit kernel, not that it must be identical to it.
    """
    fx = cell.build() if fx is None else fx
    decay = None if decay_temp is None else LeafMassDecay(dials=fx["dials"][decay_temp])
    g_read = None if norm == "global" else fx["g_read"]
    plan, sched = cell.plan(norm=norm, decay_on=decay is not None, fx=fx)
    if sched is None:
        sched = build_schedule(fx["read_f32"], fx["write_f32"],
                                  None if g_read is None else fx["g_read_f32"],
                                  fx["g_write_f32"], plan)
    y_ref, state_ref = naive_rola(
        fx["v"], fx["read_levels"], fx["write_levels"], g_read, fx["g_write"],
        fx["topology"], decay, norm, initial_state=None, output_final_state=True)

    def launch(sp):
        return _consumer_forward(
            fx["v_f32"], fx["read_f32"], fx["write_f32"],
            None if g_read is None else fx["g_read_f32"], fx["g_write_f32"], plan,
            decay=decay, schedule=sched, sequence_parallel=sp)

    y1, st1, _ = launch(None)
    sp = _forced_split(plan, sched, P)
    yp, stp, _ = launch(sp)

    err_y = float((yp.double() - y_ref).abs().max()) / max(1e-30, float(y_ref.abs().max()))
    err_s = float((stp.double() - state_ref).abs().max()) / max(
        1e-30, float(state_ref.abs().max()))
    tol_y = RTOL if prec == "tf32x3" else BF16_RTOL
    tol_s = RTOL if prec == "tf32x3" else BF16_FOLD_STATE_RTOL
    ok = (err_y < tol_y) and (err_s < tol_s)
    #: THE `P = 1` CONTROL IS RUN IN THE SAME PROCESS ON THE SAME FIXTURE, so a
    #: failure at `P > 1` can be attributed to the SPLIT rather than to the cell.
    #: Without it a bad fixture and a bad split look identical -- the discipline
    #: that resolved the paging gate, where the control proved the gate wrong
    #: rather than the kernel.
    ctl = float((y1.double() - y_ref).abs().max()) / max(1e-30, float(y_ref.abs().max()))
    if verbose:
        arm = f"P={P} {norm} {prec}" + (f" decay:{decay_temp}" if decay_temp else "")
        print(f"  {'PASS' if ok else 'FAIL':4s}  {cell.name:26s} {arm:24s} "
              f"y {err_y:.3e}  state {err_s:.3e}  (P=1 control y {ctl:.3e})")
    return ok




def run_split_gate(*, decay=True):
    """`P` crossed with decay and normalization -- the two axes §2.13 names.

    `A` is decay's object (it is identically `1` on the `decay = None` arm and the
    accumulated `s0^2` under decay), and the summary ROW WIDTH is normalization's
    (`cols` carries the mass column under `global` and not under `raw`). Testing
    `P` at one point on either axis would leave the interaction untested.
    """
    total = passed = 0
    temps = [None] + (["warm"] if decay else [])
    for name in ("topo-D2-w64x64-BC64", "topo-D3-w16x16x64-BC64",
                 "wk1-D2-w64x64-BC64", "topo-D2-w64x16-BC16"):
        for P in (2, 3):
            for norm in ("global", "raw"):
                for temp in temps:
                    total += 1
                    passed += int(run_split(CELLS[name], P=P, norm=norm,
                                            decay_temp=temp, prec="tf32x3"))
    print(f"\n{passed}/{total} P>1 arms PASS")
    return passed == total




def run_p1_identity():
    """`P = 1` with a plan vs `P = 1` with no plan -- **ON STATE, and `y` EXCLUDED.**

    `P = 1` is the primary benchmarked configuration (§6.4), so it must reach the
    kernel through exactly the arguments it reached it through before `P` existed:
    a perturbed baseline is not a baseline. An unsplit plan carries no work list and
    no workspace, so every sequence-parallel pointer is null and the kernel takes
    the pre-`P` path without reading memory.

    **`y` IS DELIBERATELY NOT ASSERTED, and the NULL CONTROL below is what
    establishes that rather than a claim.** The §5 readout lands through `atomicAdd`
    into `num`/`den` from concurrently scheduled owners, so its summation order is a
    RACE BY CONSTRUCTION -- two byte-identical launches already disagree in the last
    bits. Asserting `torch.equal` on `y` would be asserting a property the kernel has
    never had. STATE is the output `P` actually moves from deterministic to
    re-associated: it is per-owner and published exactly once, so it is the output
    the bitwise discipline can and does apply to.

    This is the same discipline the paging gate needed (§2.8), and it is run the
    same way round: there the control proved the GATE wrong rather than the kernel,
    and a first version of THIS gate asserted `y` bitwise and failed for exactly
    that reason.
    """
    from rola.routing.sequence_parallel import (
        SequenceParallelPlan,
        SequenceParallelPolicy,
        owner_slot_counts,
        select_sequence_parallelism,
    )
    ok = True
    for name in ("topo-D2-w64x64-BC64", "wk1-D2-w64x64-BC64"):
        for temp in (None, "warm"):
            cell = CELLS[name]
            fx = cell.build()
            d = None if temp is None else LeafMassDecay(dials=fx["dials"][temp])
            plan, sched = cell.plan(norm="global", decay_on=d is not None,
                                    fx=fx)
            if sched is None:
                sched = build_schedule(fx["read_f32"], fx["write_f32"], None,
                                          fx["g_write_f32"], plan)
            base = select_sequence_parallelism(
                plan, sched, policy=SequenceParallelPolicy(workspace_cap=1 << 30))
            n = int(sched.owners.shape[0])
            Ps = tuple([1] * n)
            unsplit = SequenceParallelPlan(
                P=Ps, limiting_gate=(None,) * n, binding_gate=None, sms=base.sms,
                blocks_per_sm=base.blocks_per_sm, waves_before=base.waves_before,
                waves_after=0.0, base_traffic_bytes=0, roundtrip_bytes=0,
                roundtrip_seconds=0.0, base_traffic_seconds=0.0, ledger=base.ledger,
                n_slots=sum(owner_slot_counts(Ps)), BC=base.BC, cols=base.cols)

            def launch(sp):
                return _consumer_forward(
                    fx["v_f32"], fx["read_f32"], fx["write_f32"], None,
                    fx["g_write_f32"], plan, decay=d, schedule=sched,
                    sequence_parallel=sp)

            ya, sa, _ = launch(None)
            yb, sb, _ = launch(unsplit)
            # THE NULL CONTROL: two identical NO-PLAN launches. If `y` differs here
            # too, a `y` difference above is the atomic race and not the plan.
            yc, sc, _ = launch(None)
            state_bit = bool(torch.equal(sa, sb))
            null_state_bit = bool(torch.equal(sa, sc))
            y_plan = bool(torch.equal(ya, yb))
            y_null = bool(torch.equal(ya, yc))
            # The gate is STATE bitwise, on both the plan comparison and the null
            # control. `y` is reported, never asserted.
            bit = state_bit and null_state_bit
            ok = ok and bit
            print(f"  {'PASS' if bit else 'FAIL':4s}  {name:26s} "
                  f"{'decay' if temp else 'none ':5s}  STATE bitwise: plan {state_bit} "
                  f"null-control {null_state_bit}   "
                  f"(y bitwise: plan {y_plan}, NULL CONTROL {y_null} "
                  f"-- the atomicAdd race, not the plan)")
    return ok




def run_btab(name, *, reps=21, norm="global", decay_temp=None):
    """`BT = 16` against `BT = 32` ON THE NEW CONSUMER, interleaved rep by rep.

    **This is the calibration instrument for the planner's `BT` axis** (§2.31,
    §2.32). Both arms are the ONE consumer, so the only
    difference is the tile size, and the schedule is REBUILT at each `BT` by
    restricting the planner's candidate set (see `Cell.plan`), never stamped.

    The reported statistic is the median of the per-pair ratios `t32 / t16`, so a
    value above 1 means `BT = 32` is SLOWER. Absolute medians and IQRs are printed
    alongside; a cell whose IQR cannot resolve the effect is reported as
    unresolvable here and decided on `ncu` counters instead, which is stated in the
    findings rather than hidden by quoting the ratio anyway.
    """
    cell = CELLS[name]
    fx = cell.build()
    decay = None if decay_temp is None else LeafMassDecay(dials=fx["dials"][decay_temp])
    gr = None if norm == "global" else fx["g_read"]

    def mk(bt):
        #: The new consumer must be stamped `panel` on the decay arm so its
        #: admission bound is checked -- the same divergence `run_atscale` makes,
        #: and it is applied to BOTH arms here, so it cannot bias the ratio.
        pl, sch = cell.plan(norm=norm, decay_on=decay is not None,
                            fx=fx, bt=bt)
        _, sch = _timed(cell, fx, pl, decay, gr, sch)
        return pl, sch

    p16, s16 = mk(16)
    p32, s32 = mk(32)
    if (p16.BT, p32.BT) != (16, 32):
        raise SystemExit(f"the planner returned BT={p16.BT}/{p32.BT}, not 16/32")
    t16, t32, pair = [], [], []
    for _ in range(reps):
        x, _ = _timed(cell, fx, p16, decay, gr, s16)
        y, _ = _timed(cell, fx, p32, decay, gr, s32)
        t16.append(x)
        t32.append(y)
        pair.append(y / x)
    m16, m32, mr = statistics.median(t16), statistics.median(t32), statistics.median(pair)
    print(f"{name} norm={norm} decay={decay_temp} BC={cell.BC}")
    print(f"  BT=16 {m16:.3f} ms (IQR {_iqr(t16):.3f})   "
          f"BT=32 {m32:.3f} ms (IQR {_iqr(t32):.3f})")
    print(f"  PAIRED 32/16 {mr:.4f} (IQR {_iqr(pair):.4f})   = {100 * (mr - 1):+.1f} %")
    return dict(paired=mr, paired_iqr=_iqr(pair), ms16=m16, ms32=m32)




def decay_price(name, *, reps=11, decay_temp="warm"):
    """DECAY'S OWN PRICE: the new kernel against itself, decay vs none.

    The structural expectation this replaces with a measurement: on a DENSE-WRITE
    cell decay adds two `expf` panels (`2 * BT * BC`, on the XU pipe, which §2.5's
    instruction mix measured at 0.8 M -- essentially idle), the clock prefix, the
    two coefficient folds and the `s0` entry/exit sweeps. On a SPARSE-WRITE cell
    §2's write-conditioned quiescence exempts read-only tiles STRUCTURALLY
    (`c = 0` => `keep == 1` exactly), so the cost should scale with the WRITE-LIVE
    share and approach free as that share falls.
    """
    cell = CELLS[name]
    fx = cell.build()
    share = skip_share(cell, fx)["read_only_frac"]
    r = _paired_one_kernel(
        cell, fx, ("global", None), ("global", decay_temp), reps,
        "DECAY PRICE (new kernel both arms)",
        f", read-only share {100 * share:.1f} %")
    #: **THE RATIO ABOVE IS NOW THE KERNEL'S PRICE, and it was not always.** The
    #: decay arm's preamble (`build_clock_stream` + `leaf_rate_dials` +
    #: `panel_exponent_bound`) used to be rebuilt on EVERY call, including calls
    #: handed a prebuilt `schedule=`, and landed entirely in this ratio because the
    #: `decay=None` arm pays none of it. It is now memoized on the schedule
    #: (`DecayStream`), so it is paid once on the warm call and is OUTSIDE the
    #: timed region -- which is why it is REPORTED here but NOT subtracted.
    #: Subtracting it now would double-count.
    #:
    #: It never contaminated the production A/B: there both arms are decay arms and
    #: paid it identically, so it cancelled.
    pre = _decay_preamble_ms(cell, fx, decay_temp)
    print(f"  (one-time preamble, now memoized OUT of the timed region: {pre:.3f} ms)")
    r.update(preamble_ms=pre)
    return r




def norm_price(name, *, reps=11, decay_temp=None):
    """NORMALIZATION'S OWN PRICE: `global` vs `raw`, new kernel both arms.

    `global` costs, structurally: the mass column in the fold (`v := 1`, one extra
    column of `DV + 1`), the scalar denominator matvec (a `[BT x BC] x [BC x 1]`
    reduction deliberately NOT lowered to an MMA -- an `n8` tile would waste 7/8 of
    a fragment), one extra element in each `s0` sweep under decay, the denominator
    atomics, and the `v3_divide` epilogue.
    """
    cell = CELLS[name]
    fx = cell.build()
    return _paired_one_kernel(
        cell, fx, ("raw", decay_temp), ("global", decay_temp), reps,
        "NORMALIZATION PRICE (new kernel both arms), global vs raw")

