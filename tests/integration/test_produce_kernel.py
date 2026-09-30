"""The fused producer's kernel (csrc/rola/src/produce/), at the op: every output it owes, written.

The component gates land beside this as the components do (docs/internals/produce/produce.md): the skeleton's claim
is the composition's SHAPE -- every element of every output is written, whatever the routing, and the arrival
counters are back at zero after the call.
"""
from __future__ import annotations

import pytest
import torch

import rola
from rola.ops.produce import ProducerPlan, _job_tensor, channel_major

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the producer is a CUDA kernel")

_ROUTINGS = {
    "union": [rola.union_routing(64), rola.union_routing(32)],
    "split": [rola.split_routing(64, sparse_duty="write"), rola.split_routing(64, sparse_duty="read")],
    "tied-deep": [rola.tied_routing(16), rola.dense_routing(32), rola.union_routing(16)],
}


def _operands(name: str, *, B=2, T=5, hidden=48, H=3, d_v=64):
    torch.manual_seed(0)
    plan = ProducerPlan(routing=rola.Routing(_ROUTINGS[name]), num_heads=H, hidden_size=hidden, d_v=d_v)
    route_W = torch.randn(H, hidden, plan.routing.columns, device="cuda").to(torch.bfloat16)
    gain_bias = torch.zeros(H, plan.routing.gain_columns, device="cuda")
    v_W = channel_major(torch.randn(H * d_v, hidden, device="cuda").to(torch.bfloat16))
    x = torch.randn(B, T, hidden, device="cuda")
    return plan, x, route_W, gain_bias, v_W


def _launch(plan, x, route_W, gain_bias, v_W, fill: int):
    """One launch into buffers poisoned with ``fill``; the outputs and the arrival counters after it."""
    from rola.ops._ext import extension

    B, T, hidden = x.shape
    H, D, d_v = plan.num_heads, plan.routing.topology.D, plan.d_v
    rows = plan.routing.resolved.packed_logit_width
    row_dtype = torch.uint8 if D <= 2 else torch.float16

    def poisoned(*shape, dtype=torch.float32):
        out = torch.empty(B, T, H, *shape, dtype=dtype, device="cuda")
        out.view(torch.uint8).fill_(fill)
        return out

    out = {"amp_read": poisoned(rows, dtype=row_dtype), "amp_write": poisoned(rows, dtype=row_dtype),
           "scale_read": poisoned(D), "scale_write": poisoned(D), "mass_read": poisoned(D), "mass_write": poisoned(D),
           "read_mass": poisoned(), "g_write": poisoned(), "v_hat": poisoned(d_v, dtype=torch.int8),
           "v_lo": poisoned(d_v, dtype=torch.uint8), "v_scale": poisoned()}
    jobs, per_head = _job_tensor(plan, "cuda:0")
    arrivals = torch.zeros(-(-(B * T) // 4) * H, dtype=torch.int32, device="cuda")
    extension().rola_produce_forward(
        x.reshape(B * T, hidden).to(torch.bfloat16).contiguous(), route_W, None, gain_bias, v_W, jobs,
        *out.values(), arrivals, D, per_head)
    torch.cuda.synchronize()
    return out, arrivals


@pytest.mark.parametrize("name", list(_ROUTINGS))
def test_every_output_is_written_and_the_arrivals_reset(name):
    operands = _operands(name)
    first, arrivals = _launch(*operands, fill=0x55)
    second, _ = _launch(*operands, fill=0xAA)
    for key in first:
        assert torch.equal(first[key].view(torch.uint8), second[key].view(torch.uint8)), (
            f"{name}: `{key}` holds a byte the kernel did not write")
    assert not arrivals.any(), f"{name}: an arrival counter was left standing"
