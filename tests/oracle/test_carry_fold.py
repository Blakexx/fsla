"""Gate for THE CARRY'S FOLD (csrc/rola/src/carry/fold.cuh): the state at a call's end, read through the probe, is the
fp64 reference of the fold's declared arithmetic (`tests/oracle/fold_reference.py`, `fold_state`) per slot, at both
orders of the even box's levels (docs/internals/carry/fold.md).

Symbols: ``W`` the window (512 tokens), ``L`` tokens, ``B_l`` level ``l``'s width, ``d_v`` the value width.
"""
from __future__ import annotations

import pytest
import torch

from rola.engine.carve import plan_carve
from rola.ops import carry as carry_ops
from rola.ops import facts as facts_ops
from tests.oracle.carry_fixtures import alternating_bundle
from tests.oracle.fold_reference import fold_state


#: THE BOUND, relative to each slot's envelope (sum |A| |v|, absolute): a slot takes at most one fp32 accumulation a
#: 16-token step, ``L / 16`` of them, each off by at most 2^-23 (the MMA's accumulate truncates); the rescales exact.
def _rtol(L: int) -> float:
    return L / 16 * 2.0 ** -23


@pytest.mark.parametrize("write_sparse", [0, 1])
@pytest.mark.parametrize("growth", [0, 8, -8])
def test_the_fold_is_the_reference_per_slot(write_sparse, growth):
    """The even box (`[DS, SD]` and `[SD, DS]` at 64 x 64): every leaf and channel of the final state within the bound
    of the reference's over four windows -- the write gain steady, or scaled 2^growth a window, so the units climb
    and rescale the state, or are held under the bound's headroom."""
    widths, L = (64, 64), 2048
    routes, values = alternating_bundle(widths, L, write_sparse)
    window = torch.arange(L, device="cuda") // carry_ops.WINDOW
    routes.g_write.mul_(torch.exp2(growth * window.float())[None, :, None])
    carve = plan_carve(tuple(routes.topology.padded_widths), *routes.sparse, values.v.shape[-1])
    assert carry_ops.arm_key(carve) == (32, 1, 32, 1, 32, 0)
    layout = facts_ops.call_layout(routes)
    facts = facts_ops.facts(routes, values, layout, *facts_ops.call_statics(routes, layout),
                            carry_ops.form(carry_ops.arm_key(carve)).read_row)
    BH, N, d_v = routes.batch * routes.heads, widths[0] * widths[1], values.v.shape[-1]
    probe = torch.zeros((BH, N, d_v), dtype=torch.float32, device="cuda")
    unit = torch.zeros((BH, L), dtype=torch.int8, device="cuda")
    carry_ops.probe_bind(probe)
    try:
        carry_ops.carry(routes, values, facts, carve, unit)
        torch.cuda.synchronize()
    finally:
        carry_ops.probe_bind(None)
    level_s = 1 - write_sparse
    state, env, _, _ = fold_state(routes, values, facts.list_exp, level_s, carry_ops.WINDOW)
    if level_s == 1:
        state, env = state.permute(0, 2, 1, 3), env.permute(0, 2, 1, 3)
    want, env = state.reshape(BH, N, d_v), env.reshape(BH, N, d_v)
    err = (probe.double() - want).abs()
    ratio = (err / (env * _rtol(L)).clamp_min(1e-300)).max().item()
    assert (env > 0).float().mean() > 0.5, "most slots unwritten: the cell is vacuous"
    assert ratio <= 1.0, f"a slot is {ratio:.3g}x its bound"
