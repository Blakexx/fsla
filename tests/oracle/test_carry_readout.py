"""Gate for THE CARRY'S READOUT (csrc/rola/src/carry/parts.cuh): each token's packed partials in ``num`` are the fp64
reference of the readout's declared arithmetic (`tests/oracle/readout_reference.py`) per slot, at both orders of the
even box's levels, over four windows (docs/internals/carry/parts.md#readout).

Symbols: ``W`` the window (512 tokens), ``L`` tokens, ``B_l`` level ``l``'s width, ``d_v`` the value width.
"""
from __future__ import annotations

import pytest
import torch

from rola.engine.carve import plan_carve
from rola.ops import carry as carry_ops
from rola.ops import facts as facts_ops
from tests.oracle import carry_plan
from tests.oracle.carry_fixtures import FORM_CELLS, alternating_bundle
from tests.oracle.readout_reference import readout, units, unpack_num


def _readout_ratio(widths, L, write_sparse, growth, accumulate, *, live=4, forced=None):
    """``(ratio, plans)``: the call's largest slot distance from the reference over its bound, and its boxes' plans."""
    routes, values = alternating_bundle(widths, L, write_sparse, live=live)
    window = torch.arange(L, device="cuda") // carry_ops.WINDOW
    routes.g_write.mul_(torch.exp2(growth * window.float())[None, :, None])
    carve = plan_carve(tuple(routes.topology.padded_widths), *routes.sparse, values.v.shape[-1])
    key = carry_ops.arm_key(carve)
    assert key == (32, 1, 32, 1, 32, 0)
    layout = facts_ops.call_layout(routes)
    facts = facts_ops.facts(routes, values, layout, *facts_ops.call_statics(routes, layout),
                            carry_ops.form(key).read_row)
    level_s = 1 - write_sparse
    unit = units(routes, values, facts.list_exp, level_s, carry_ops.WINDOW)
    num, _ = carry_ops.carry(routes, values, facts, carve, unit, accumulate=accumulate, forced=forced)
    torch.cuda.synchronize()
    want, bound = readout(routes, values, facts.list_exp, level_s, carry_ops.WINDOW, unit, accumulate)
    got = unpack_num(num).double()
    later = slice(carry_ops.WINDOW, L)
    assert (want[:, later].abs() > 64).float().mean() > 0.5, "most later slots under 64 units: the cell is vacuous"
    assert torch.equal(got[:, :carry_ops.WINDOW], torch.zeros_like(got[:, :carry_ops.WINDOW])), (
        "the first window read a state before any write")
    return ((got - want).abs() / bound).max().item(), carry_plan.plans(routes, facts, carve, forced)


@pytest.mark.parametrize("accumulate", ["fp32", "fp16"])
@pytest.mark.parametrize("write_sparse", [0, 1])
@pytest.mark.parametrize("growth", [0, 8])
def test_the_readout_is_the_reference_per_slot(write_sparse, growth, accumulate):
    """The even box (`[DS, SD]` and `[SD, DS]` at 64 x 64), both accumulate forms: every token and channel of ``num``
    within the bound of the reference's partials at the token's unit -- the write gain steady, or scaled 2^growth a
    window, so the fold's lists climb through their units and the image puts them on one scale."""
    ratio, _ = _readout_ratio((64, 64), 2048, write_sparse, growth, accumulate)
    assert ratio <= 1.0, f"a slot is {ratio:.3g}x its bound"


@pytest.mark.parametrize(("widths", "live", "forced", "forms"), FORM_CELLS)
def test_every_plan_form_reads_out_to_the_reference(widths, live, forced, forms):
    """Each of the plan's forms (`tests/oracle/carry_fixtures.py` `FORM_CELLS`), its cell taking them by the plans'
    model: every slot of ``num`` within its bound, the units climbing 2^8 a window, the device's accumulate form."""
    ratio, plans = _readout_ratio(widths, 2048, 0, 8, carry_ops.readout_accumulate(), live=live, forced=forced)
    assert forms <= carry_plan.forms(plans), f"the cell takes {sorted(carry_plan.forms(plans))}, not {sorted(forms)}"
    assert ratio <= 1.0, f"a slot is {ratio:.3g}x its bound"
