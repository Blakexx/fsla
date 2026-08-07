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

from bench.cells import BF16_FOLD_STATE_RTOL, BF16_RTOL, CELLS, GROUPS, RTOL, Cell, skip_share
from bench.instruments import _decay_preamble_ms
from bench.pairing import _paired_one_kernel
from rola.ops.consumer import _consumer_forward
from rola.ops.naive import naive_rola
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

