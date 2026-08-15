"""TIER 2 — execute every row of ``matrix.py``.

One test body per contract class, parametrized over the declared matrix. The
class decides the assertion, and the row decides nothing else:

* **class (a) ENGINEERED IDENTITY** -> ``torch.equal`` on the raw bits, after the
  row's ``precondition`` has been ASSERTED (never assumed).
* **class (b) REASSOCIATION-EQUIVALENT** -> BOTH of §6.2's required assertions:
  each geometry independently within the tier-1 ``BF16_RTOL`` of the SAME fp64 oracle
  point, and the two within a bound DERIVED from the reassociation.

The two prohibitions are structural here rather than remembered: there is no
``allclose`` in the class-(a) body, and there is no tolerance constant in the
class-(b) body — only ``BOUNDS``, whose entries are functions of the geometry.
"""
from __future__ import annotations

import pytest
import torch

from rola.ops.consumer import _consumer_forward
from rola.ops.paging import PageArena
from tests.tier1_semantics.tolerances import BF16_RTOL
from tests.tier2_geometry.fixtures import (
    _Fixture,
    _relative,
)
from tests.tier2_geometry.matrix import (
    ENGINEERED_IDENTITY,
    GEOMETRY_MATRIX,
    REASSOCIATION_EQUIVALENT,
    GeometryPair,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the consumer is a CUDA kernel"),
]

#: P26: the consumer is bf16-only now (no more tf32x3/fp32-MMA arm), so the
#: dominant per-term rounding a reassociation exposes is bf16's, not fp32's --
#: MEASURED: the BC row's cross-geometry delta lands at exactly one bf16 ULP
#: (7.812e-3 == `torch.finfo(torch.bfloat16).eps`) against an fp32-eps bound
#: 1400x tighter than that. Re-derived, not loosened by a chosen number.
_EPS = torch.finfo(torch.bfloat16).eps


# ---------------------------------------------------------------------------
# Fixtures — named once, referenced by rows
# ---------------------------------------------------------------------------
#
# Every fixture is a CLOSURE over its own seed, so the two geometries of a row see
# byte-identical inputs. That is the precondition of the whole tier: a difference
# in the inputs would be indistinguishable from a difference in the geometry.

def _multi_owner_dense(**over):
    """widths `(8,8)` — 64 columns, so `BC = 16` gives 4 owners and `BC = 64` gives 1."""
    over.setdefault("BC", 16)
    return _Fixture((8, 8), over.pop("BC"), 64, B=1, T=128, H=2,
                    p_read=1.0, p_write=1.0, seed=4242, **over)


def _paged_dense(**over):
    """widths `(4,8,8)` at `BC = 16` — 16 owners, densely written, so every page is
    admitted and the arena is fully resident (§7.1's dense short-circuit).

    P26: reseeded `11 -> 12`. Under the bf16-only readout (`y` comes back bf16,
    cast after the fp32 divide) `seed=11`'s split-K atomic reordering noise is
    smaller than half a bf16 ULP and rounds away, so the paging `y` row's
    REASSOCIATION_EQUIVALENT contract is real but this draw could not witness
    it -- M3's own non-vacuity check (verified by construction) caught exactly
    this. `seed=12` clears it (measured, not tuned: swept seeds 11..39, first
    hit at 12).
    """
    return _Fixture((4, 8, 8), 16, 64, B=1, T=48, H=2, p_read=1.0, p_write=1.0,
                    seed=12, **over)


FIXTURES = {
    "multi_owner_dense": _multi_owner_dense,
    "paged_dense": _paged_dense,
}


# ---------------------------------------------------------------------------
# Selectors — the declared restrictions
# ---------------------------------------------------------------------------

SELECTORS: dict = {}


# ---------------------------------------------------------------------------
# Derived bounds — class (b) only, and never a constant
# ---------------------------------------------------------------------------

def _split_readout_n2_eps(fx_a, fx_b, out_a, out_b):
    """The most `y` can move under a reassociation of the §5 readout's atomics.

    One output element is a sum of at most `owners_total` fp32 partials, and
    reassociating an `n`-term fp32 sum perturbs it by at most
    `(n-1) * eps * sum|terms|`. Bounding `sum|terms|` by `n * max|y|` (the reachable
    scale when the partials do not catastrophically cancel) gives `n^2 * eps * max|y|`.
    `n` is the LARGER of the two geometries' owner counts, because the bound must
    cover whichever arm re-associates more. Derived arithmetic, not a tuned number.
    (Name kept from the P10 era: `split-K` names the readout SHAPE — partial per
    owner, folded after — not the fold mechanism, so the bound stayed apt across
    the atomics<->deterministic-workspace revert, P23.)
    """
    n = max(fx_a.plan.geometry.owners_total, fx_b.plan.geometry.owners_total)
    return n * n * _EPS * float(out_a.abs().max())


BOUNDS = {
    "split_readout_n2_eps": _split_readout_n2_eps,
}


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------

_PLAN_AXES = {"BC": "BC"}


def build(row: GeometryPair, geometry: dict):
    """Materialize one side of a row: the fixture, plus the row's own axis applied."""
    over = {_PLAN_AXES[axis]: value for axis, value in geometry.items()
            if axis in _PLAN_AXES}
    return FIXTURES[row.fixture](**over)


def run(row: GeometryPair, fx, geometry: dict):
    """Launch one side and return `(y, state)` in ONE layout for both sides."""
    if geometry.get("paging"):
        arena = PageArena(widths=fx.topology.widths, BC=fx.plan.BC, cols=fx.cols,
                          BH=fx.v.shape[0] * fx.v.shape[2], device=torch.device("cuda"))
        y, returned, _ = _consumer_forward(
            fx._f32["v"], fx._f32["read_levels"], fx._f32["write_levels"],
            fx._f32["g_read"], fx._f32["g_write"], fx.plan, schedule=fx.schedule,
            decay=fx.decay, paging=arena)
        assert returned is arena, "under paging the arena IS the carried state"
        return y, arena.materialize()
    y, state, _ = fx.run()
    if "paging" in geometry:
        # The unpaged arm, reshaped into the arena's layout so the comparison is
        # between the same thing addressed two ways -- which is the whole claim.
        return y, state.reshape(-1, fx.topology.N, fx.cols)
    return y, state


def pick(row: GeometryPair, fx, y, state):
    out = y if row.output == "y" else state
    if row.restrict is not None:
        out = SELECTORS[row.restrict](fx, out)
    return out


def both_sides(row: GeometryPair):
    fx_a = build(row, dict(row.a))
    fx_b = build(row, dict(row.b))
    ya, sa = run(row, fx_a, dict(row.a))
    yb, sb = run(row, fx_b, dict(row.b))
    return fx_a, fx_b, pick(row, fx_a, ya, sa), pick(row, fx_b, yb, sb)


def _ids(rows):
    return [r.id for r in rows]


_CLASS_A = [r for r in GEOMETRY_MATRIX if r.contract is ENGINEERED_IDENTITY]
_CLASS_B = [r for r in GEOMETRY_MATRIX if r.contract is REASSOCIATION_EQUIVALENT]


# ---------------------------------------------------------------------------
# Preconditions — asserted, never assumed
# ---------------------------------------------------------------------------

def assert_precondition(row: GeometryPair, fx_a, fx_b):
    """Every row's stated precondition, checked against the built geometries.

    §6.2: *"a bit-identity assertion whose precondition is unstated is a future
    flake"* — and one that is stated but not checked is the same flake with a
    comment. The mapping from a row to its check is by AXIS and CONTRACT, so a new
    row cannot arrive without one.
    """
    if row.axis == "BC":
        assert fx_a.topology.widths == fx_b.topology.widths, (
            f"{row.id}: the two BC values must tile the SAME state columns")
        if row.contract is ENGINEERED_IDENTITY:
            assert fx_a.initial_state is None and fx_b.initial_state is None
        else:
            assert fx_a.plan.geometry.owners_total != fx_b.plan.geometry.owners_total, (
                f"{row.id}: both arms have the same owner count, so no split-K "
                "reassociation is being exercised")
    elif row.axis == "paging":
        assert fx_a.plan.geometry.owners_total > 1 or row.output == "state"
    else:
        raise AssertionError(
            f"{row.id}: axis {row.axis!r} has no precondition check. A row whose "
            "precondition is prose only is not gated by anything.")


# ---------------------------------------------------------------------------
# CLASS (a) — byte equality, no tolerance, ever
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("row", _CLASS_A, ids=_ids(_CLASS_A))
def test_an_engineered_identity_is_byte_equal(row: GeometryPair):
    fx_a, fx_b, out_a, out_b = both_sides(row)
    assert_precondition(row, fx_a, fx_b)
    assert torch.equal(out_a.view(torch.int32), out_b.view(torch.int32)), (
        f"{row.id} is declared ENGINEERED_IDENTITY and is NOT byte-equal: "
        f"max |delta| = {float((out_a - out_b).abs().max()):.3e}.\n"
        f"  basis: {row.basis}\n"
        f"  precondition: {row.precondition}\n"
        "Either the structural argument in `basis` no longer holds (a kernel change), "
        "or the row's class is wrong (a matrix change). Loosening this to `allclose` "
        "is not one of the options: it would mean the planner's performance decision "
        "moved numbers.")


# ---------------------------------------------------------------------------
# CLASS (b) — the oracle anchor AND the derived bound. Both required.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("row", _CLASS_B, ids=_ids(_CLASS_B))
def test_a_reassociation_is_oracle_anchored_and_within_its_derived_bound(row: GeometryPair):
    fx_a, fx_b, out_a, out_b = both_sides(row)
    assert_precondition(row, fx_a, fx_b)

    # 1. THE ORACLE ANCHOR. Each geometry against the SAME fp64 oracle point,
    #    independently. This is what stops two wrong geometries from agreeing --
    #    the shared-router bug passed exactly by agreeing with a sibling.
    y_ref, state_ref = fx_a.oracle()
    ref = y_ref if row.output == "y" else state_ref
    if row.restrict is not None:
        ref = SELECTORS[row.restrict](fx_a, ref)
    for tag, out in (("A", out_a), ("B", out_b)):
        rel = _relative(out, ref.reshape(out.shape))
        assert rel < BF16_RTOL, (
            f"{row.id}: geometry {tag} is {rel:.3e} from the oracle, outside the "
            f"tier-1 BF16_RTOL {BF16_RTOL:.0e}. A geometry that does not hit the oracle cannot "
            "anchor its partner, whatever the two agree on.")

    # 2. THE DERIVED MUTUAL BOUND. Named by the row, computed from the geometry.
    bound = BOUNDS[row.bound](fx_a, fx_b, out_a, out_b)
    delta = float((out_a.double() - out_b.double()).abs().max())
    assert delta <= bound, (
        f"{row.id}: the two geometries differ by {delta:.3e}, outside the derived "
        f"`{row.bound}` bound {bound:.3e}. The answer is a bug hunt or a corrected "
        "derivation -- never a bigger number.")
