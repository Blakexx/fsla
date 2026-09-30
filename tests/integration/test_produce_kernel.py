"""The fused producer's kernel (csrc/rola/src/produce/), at the op, one gate a landed component.

The skeleton's claim is the composition's SHAPE -- every element of every output is written, whatever the routing,
and the arrival counters are back at zero after the call. The projection's is its logits: every job's columns for
every token, against the fp32 product of the same bf16 operands, per slot within the envelope
(docs/internals/produce/produce.md#accumulate).
"""
from __future__ import annotations

import pytest
import torch

import rola
from rola.ops.produce import _TOKEN_TILE, ProducerPlan, _job_tensor, hidden_contiguous, job_table

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the producer is a CUDA kernel")

_ROUTINGS = {
    "union": [rola.union_routing(64), rola.union_routing(32)],
    "split": [rola.split_routing(64, sparse_duty="write"), rola.split_routing(64, sparse_duty="read")],
    "tied-deep": [rola.tied_routing(16), rola.dense_routing(32), rola.union_routing(16)],
    "wide": [rola.union_routing(256), rola.split_routing(200, sparse_duty="read")],
}
_MAX_COLUMNS = 512


def _projection_rtol(hidden: int) -> float:
    """The projection's per-slot bound, relative to the envelope sum |x| |W| (+ |bias|): the bf16 products are exact
    in fp32, so the error is the fp32 accumulation's -- ``hidden + 1`` adds of at most one unit in the last place
    each, in any order, truncating or not. Measured: at most 2^-22.5 of the envelope at hidden 256, 2^-20.7 at 1024."""
    return (hidden + 1) * 2.0 ** -23


def _operands(name: str, *, B=2, T=5, hidden=48, H=3, d_v=64, bias=True):
    torch.manual_seed(0)
    plan = ProducerPlan(routing=rola.Routing(_ROUTINGS[name]), num_heads=H, hidden_size=hidden, d_v=d_v)
    C = plan.routing.columns
    route_W = hidden_contiguous(torch.randn(H, hidden, C, device="cuda").to(torch.bfloat16))
    route_bias = torch.randn(H, C, device="cuda") if bias else None
    gain_bias = torch.zeros(H, plan.routing.gain_columns, device="cuda")
    v_W = torch.randn(H * d_v, hidden, device="cuda").to(torch.bfloat16)
    x = torch.randn(B, T, hidden, device="cuda").to(torch.bfloat16)
    return plan, x, route_W, route_bias, gain_bias, v_W


def _launch(plan, x, route_W, route_bias, gain_bias, v_W, fill: int, probe: bool = False):
    """One launch into buffers poisoned with ``fill``; the outputs, the arrival counters and the probe after it."""
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
    arrivals = torch.zeros(-(-(B * T) // _TOKEN_TILE) * H, dtype=torch.int32, device="cuda")
    logits = torch.full((jobs.shape[0], B * T, _MAX_COLUMNS), float("nan"), device="cuda") if probe else None
    extension().rola_produce_forward(
        x.reshape(B * T, hidden).contiguous(), route_W, route_bias, gain_bias, v_W, jobs,
        *out.values(), arrivals, logits, D, per_head)
    torch.cuda.synchronize()
    return out, arrivals, logits


def _expected_logits(plan, x, route_W, route_bias, v_W):
    """Per job, ``(product, envelope)`` over its columns: fp64 sums of the bf16 operands, and of their magnitudes."""
    xs = x.reshape(-1, x.shape[-1]).double()
    out = []
    for kind, head, heads, _solve, width, in0, in1, *_ in job_table(plan)[0]:
        if kind == 1:
            W = v_W[head * plan.d_v:(head + heads) * plan.d_v].double().t()
            bias = torch.zeros(W.shape[1], dtype=torch.float64, device=W.device)
        else:
            cols = list(range(in0, in0 + width)) + (list(range(in1, in1 + width)) if in1 >= 0 else [])
            W = route_W[head][:, cols].double()
            bias = (route_bias[head, cols].double() if route_bias is not None
                    else torch.zeros(len(cols), dtype=torch.float64, device=W.device))
        out.append((xs @ W + bias, xs.abs() @ W.abs() + bias.abs()))
    return out


@pytest.mark.parametrize("name", list(_ROUTINGS))
def test_every_output_is_written_and_the_arrivals_reset(name):
    operands = _operands(name)
    first, arrivals, _ = _launch(*operands, fill=0x55)
    second, _, _ = _launch(*operands, fill=0xAA)
    for key in first:
        assert torch.equal(first[key].view(torch.uint8), second[key].view(torch.uint8)), (
            f"{name}: `{key}` holds a byte the kernel did not write")
    assert not arrivals.any(), f"{name}: an arrival counter was left standing"


@pytest.mark.parametrize(("name", "B", "T", "hidden", "bias"), [
    ("union", 2, 5, 48, True),      # a partial k-block, one live token group
    ("split", 1, 71, 256, True),    # three token tiles, the last with one live group
    ("tied-deep", 3, 7, 64, False), # sliced jobs (16 columns: eight warps across the hidden axis), no bias
    ("wide", 1, 1, 256, True),      # the decode shape: one token, jobs of 512 and 200 columns
    ("wide", 4, 40, 1024, True),    # a wide hidden axis: 32 k-blocks, two in flight
])
def test_the_projection_matches_the_fp32_product(name, B, T, hidden, bias):
    plan, x, route_W, route_bias, gain_bias, v_W = _operands(name, B=B, T=T, hidden=hidden, bias=bias)
    _, _, logits = _launch(plan, x, route_W, route_bias, gain_bias, v_W, fill=0, probe=True)
    for index, (product, envelope) in enumerate(_expected_logits(plan, x, route_W, route_bias, v_W)):
        got = logits[index, :, :product.shape[1]].double()
        excess = (got - product).abs() - _projection_rtol(hidden) * envelope
        assert excess.max() <= 0, (
            f"{name}: job {index}'s logit (token, column) {divmod(int(excess.argmax()), product.shape[1])} is off by "
            f"{float((got - product).abs().flatten()[excess.argmax()]):.3g} against an envelope of "
            f"{float(envelope.flatten()[excess.argmax()]):.3g}")
