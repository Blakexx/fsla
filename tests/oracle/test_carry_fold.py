"""Gate for THE CARRY'S FOLD (csrc/rola/src/carry/fold.cuh): the state at a call's end, read through the probe, is the
fp64 reference of the fold's declared arithmetic (`tests/oracle/fold_reference.py`, `fold_state`) per slot, at both
orders of the even box's levels, and its den state the reference's mass per leaf (docs/internals/carry/fold.md#den).

Symbols: ``W`` the window (the device's), ``L`` tokens, ``B_l`` level ``l``'s width, ``d_v`` the value width.
"""
from __future__ import annotations

import pytest
import torch

from rola.engine.carve import plan_carve
from rola.ops import carry as carry_ops
from rola.ops import facts as facts_ops
from rola.ops.window import window_of
from tests.oracle import carry_plan
from tests.oracle.carry_fixtures import FORM_CELLS, alternating_bundle
from tests.oracle.fold_reference import fold_state


#: THE BOUND, relative to each slot's envelope (sum |A| |v|, absolute): a slot takes at most one fp32 accumulation a
#: 16-token step, ``L / 16`` of them, each off by at most 2^-23 (the MMA's accumulate truncates); the rescales exact.
def _rtol(L: int) -> float:
    return L / 16 * 2.0 ** -23


def _den_ratio(den: torch.Tensor, mass: torch.Tensor, L: int) -> float:
    """The den state's largest leaf distance from the reference's mass over its bound: the steps' fp32 accumulation as
    the state's and one fp32 add a window into the den state (every term positive: the mass is its own envelope)."""
    bound = mass * (_rtol(L) + (L / window_of() + 1) * 2.0 ** -23)
    return ((den.double() - mass).abs() / bound.clamp_min(1e-300)).max().item()


def _probed(routes, values, facts, carve, unit, forced=None):
    """``(probe, den)``: one launch's final state and den state through the probes."""
    widths = routes.topology.widths
    BH, N, d_v = routes.batch * routes.heads, widths[0] * widths[1], values.v.shape[-1]
    probe = torch.zeros((BH, N, d_v), dtype=torch.float32, device="cuda")
    den = torch.zeros((BH, N), dtype=torch.float32, device="cuda")
    carry_ops.probe_bind(probe, den)
    try:
        carry_ops.carry(routes, values, facts, carve, unit, forced=forced)
        torch.cuda.synchronize()
    finally:
        carry_ops.probe_bind(None)
    return probe, den


def _reference(routes, values, facts, level_s, BH, N, d_v):
    """``(state, env, mass)`` the reference's, in the probes' leaf order."""
    state, env, mass, _ = fold_state(routes, values, facts.list_exp, level_s, window_of())
    if level_s == 1:
        state, env, mass = state.permute(0, 2, 1, 3), env.permute(0, 2, 1, 3), mass.permute(0, 2, 1)
    return state.reshape(BH, N, d_v), env.reshape(BH, N, d_v), mass.reshape(BH, N)


def _fold_ratio(widths, L, write_sparse, growth, *, live=4, forced=None):
    """``(ratio, plans)``: the call's largest slot distance from the reference over its bound, and its boxes' plans."""
    routes, values = alternating_bundle(widths, L, write_sparse, live=live)
    at = torch.arange(L, device="cuda") // window_of()
    routes.g_write.mul_(torch.exp2(growth * at.float())[None, :, None])
    carve = plan_carve(tuple(routes.topology.padded_widths), *routes.sparse, values.v.shape[-1])
    assert carry_ops.arm_key(carve) == (32, 1, 32, 1, 32, 0)
    layout = facts_ops.call_layout(routes)
    facts = facts_ops.facts(routes, values, layout, *facts_ops.call_statics(routes, layout),
                            carry_ops.form(carry_ops.arm_key(carve)).read_row)
    BH, N, d_v = routes.batch * routes.heads, widths[0] * widths[1], values.v.shape[-1]
    unit = torch.zeros((BH, L), dtype=torch.int8, device="cuda")
    probe, den = _probed(routes, values, facts, carve, unit, forced)
    want, env, mass = _reference(routes, values, facts, 1 - write_sparse, BH, N, d_v)
    err = (probe.double() - want).abs()
    assert (env > 0).float().mean() > 0.5, "most slots unwritten: the cell is vacuous"
    ratio = max((err / (env * _rtol(L)).clamp_min(1e-300)).max().item(), _den_ratio(den, mass, L))
    return ratio, carry_plan.plans(routes, facts, carve, forced)


#: fresh cells whose write side has no list level (every level dense)
DENSE_CELLS = ("flagship-dense", "flat-small-multiwindow", "corner-window-flip")


@pytest.mark.parametrize("name", DENSE_CELLS)
def test_a_write_side_with_no_list_level_folds_to_the_reference(name):
    """A dense write side's final state per slot within the bound of the reference's: each token's factor F in its
    coefficient, every list at its window's bound (`dense_exponents`)."""
    from measure.cells import carry_cells
    from measure.subjects import carry_operands

    spec = next(c for c in carry_cells(tier="oracle") if c.name == name)
    routes, values, facts, carve, unit = carry_operands(spec)
    assert not carve.fold.lists, f"{name}'s write side lists"
    widths, L = routes.topology.widths, routes.tokens
    BH, N, d_v = routes.batch * routes.heads, widths[0] * widths[1], values.v.shape[-1]
    probe, den = _probed(routes, values, facts, carve, unit)
    want, env, mass = _reference(routes, values, facts, routes.level_s, BH, N, d_v)
    assert (env > 0).float().mean() > 0.5, "most slots unwritten: the cell is vacuous"
    ratio = ((probe.double() - want).abs() / (env * _rtol(L)).clamp_min(1e-300)).max().item()
    assert ratio <= 1.0, f"a slot is {ratio:.3g}x its bound"
    den_ratio = _den_ratio(den, mass, L)
    assert den_ratio <= 1.0, f"a leaf's den is {den_ratio:.3g}x its bound"


@pytest.mark.parametrize("write_sparse", [0, 1])
@pytest.mark.parametrize("growth", [0, 8, -8])
def test_the_fold_is_the_reference_per_slot(write_sparse, growth):
    """The even box (`[DS, SD]` and `[SD, DS]` at 64 x 64): every leaf and channel of the final state within the bound
    of the reference's over four windows -- the write gain steady, or scaled 2^growth a window, so the units climb
    and rescale the state, or are held under the bound's headroom."""
    ratio, _ = _fold_ratio((64, 64), 2048, write_sparse, growth)
    assert ratio <= 1.0, f"a slot is {ratio:.3g}x its bound"


@pytest.mark.parametrize(("widths", "live", "forced", "forms"), FORM_CELLS)
def test_every_plan_form_folds_to_the_reference(widths, live, forced, forms):
    """Each of the plan's forms (`tests/oracle/carry_fixtures.py` `FORM_CELLS`), its cell taking them by the plans'
    model: the final state within its bound per slot, the units climbing 2^8 a window."""
    ratio, plans = _fold_ratio(widths, 2048, 0, 8, live=live, forced=forced)
    assert forms <= carry_plan.forms(plans), f"the cell takes {sorted(carry_plan.forms(plans))}, not {sorted(forms)}"
    assert ratio <= 1.0, f"a slot is {ratio:.3g}x its bound"
