"""Gate for THE CARRY'S READOUT (csrc/rola/src/carry/parts.cuh): each slot of ``num`` is the fp64 reference of the
readout's declared arithmetic (`tests/oracle/readout_reference.py`: a CTA's units from its leaves' maxima, its pieces
rounded at them, the CTAs summed in the call's reduce dtype) within its bound, at both orders of the even box's levels,
over four windows, at both reduce dtypes (docs/internals/carry/parts.md#readout).

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
from tests.oracle.readout_reference import num_absolute, readout, units


def _readout_ratio(widths, L, write_sparse, growth, accumulate, *, live=4, forced=None, reduce_dtype=torch.float16):
    """``(ratio, plans)``: the call's largest slot distance from the reference over its bound, and its boxes' plans."""
    routes, values = alternating_bundle(widths, L, write_sparse, live=live)
    at = torch.arange(L, device="cuda") // window_of()
    routes.g_write.mul_(torch.exp2(growth * at.float())[None, :, None])
    carve = plan_carve(tuple(routes.topology.padded_widths), *routes.sparse, values.v.shape[-1])
    key = carry_ops.arm_key(carve)
    assert key == (32, 1, 32, 1, 32, 0)
    layout = facts_ops.call_layout(routes)
    facts = facts_ops.facts(routes, values, layout, *facts_ops.call_statics(routes, layout),
                            carry_ops.form(key).read_row)
    level_s = 1 - write_sparse
    scale = units(routes, values, facts.list_exp, level_s, window_of())
    num, _ = carry_ops.carry(routes, values, facts, carve, scale, accumulate=accumulate, forced=forced,
                             reduce_dtype=reduce_dtype)
    torch.cuda.synchronize()
    want, bound = readout(routes, values, facts.list_exp, level_s, window_of(), scale, accumulate, reduce_dtype)
    got = num_absolute(num, scale)
    later = slice(window_of(), L)
    assert (want[:, later].abs() > 0).float().mean() > 0.5, "most later slots read nothing: the cell is vacuous"
    assert torch.equal(got[:, :window_of()], torch.zeros_like(got[:, :window_of()])), (
        "the first window read a state before any write")
    return ((got - want).abs() / bound).max().item(), carry_plan.plans(routes, facts, carve, forced)


@pytest.mark.parametrize("reduce_dtype", [torch.float16, torch.float32], ids=["f16", "fp32"])
@pytest.mark.parametrize("accumulate", ["fp32", "fp16"])
@pytest.mark.parametrize("write_sparse", [0, 1])
@pytest.mark.parametrize("growth", [0, 8])
def test_the_readout_is_the_reference_per_slot(write_sparse, growth, accumulate, reduce_dtype):
    """The even box (`[DS, SD]` and `[SD, DS]` at 64 x 64), both accumulate forms, both reduce dtypes: every token and
    channel of ``num`` within the bound of the reference's readout -- the write gain steady, or scaled 2^growth a
    window, so the fold's lists climb through their units and the image puts them on one scale."""
    ratio, _ = _readout_ratio((64, 64), 2048, write_sparse, growth, accumulate, reduce_dtype=reduce_dtype)
    assert ratio <= 1.0, f"a slot is {ratio:.3g}x its bound"


@pytest.mark.parametrize(("widths", "live", "forced", "forms"), FORM_CELLS)
def test_every_plan_form_reads_out_to_the_reference(widths, live, forced, forms):
    """Each of the plan's forms (`tests/oracle/carry_fixtures.py` `FORM_CELLS`), its cell taking them by the plans'
    model: every slot of ``num`` within its bound, the units climbing 2^8 a window, the device's accumulate form."""
    ratio, plans = _readout_ratio(widths, 2048, 0, 8, carry_ops.readout_accumulate(), live=live, forced=forced)
    assert forms <= carry_plan.forms(plans), f"the cell takes {sorted(carry_plan.forms(plans))}, not {sorted(forms)}"
    assert ratio <= 1.0, f"a slot is {ratio:.3g}x its bound"
