"""The fused producer's kernel (csrc/rola/src/produce/), at the op, one gate a landed component.

The skeleton's claim is the composition's SHAPE -- every element of every output is written, whatever the routing,
and the arrival counters are back at zero after the call. The projection's is its logits: every job's columns for
every token, against the fp64 product of the same bf16 operands, per slot within the envelope
(docs/internals/produce/produce.md#accumulate). The solve's is its levels: every route job's sides against the fp64
reference solve on the kernel's own logits (docs/internals/produce/produce.md#solve). The bundle's is the routes the
fp64 composition makes from the same bf16 operands: every row's digits, every scale, the read mass and the write gain
(docs/internals/produce/produce.md#bundle). The values': every channel's f16 unit and every scale, against the same
(docs/internals/produce/produce.md#value-rows).
"""
from __future__ import annotations

import pytest
import torch

import rola
from rola.ops.produce import (
    _TOKEN_TILE,
    ProducerPlan,
    ProducerWeights,
    _job_tensor,
    hidden_contiguous,
    job_table,
    produce,
    row_layout,
)
from rola.routing.entmax.production import _torch_activation, _torch_union_split

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the producer is a CUDA kernel")

_ROUTINGS = {
    "union": [rola.union_routing(64), rola.union_routing(32)],
    "split": [rola.split_routing(64, sparse_duty="write"), rola.split_routing(64, sparse_duty="read")],
    "tied-deep": [rola.tied_routing(16), rola.dense_routing(32), rola.union_routing(16)],
    "wide": [rola.union_routing(256), rola.split_routing(200, sparse_duty="read")],
    "sparsemax": [rola.union_routing(64, 2.0), rola.tied_routing(32, rola.entmax(2.0))],
}
_MAX_COLUMNS = 512


def _projection_rtol(hidden: int) -> float:
    """The projection's per-slot bound, relative to the envelope sum |x| |W| (+ |bias|): the bf16 products are exact
    in fp32, so the error is the fp32 accumulation's -- ``hidden + 1`` adds of at most one unit in the last place
    each, in any order, truncating or not. Measured: at most 2^-22.5 of the envelope at hidden 256, 2^-20.7 at 1024."""
    return (hidden + 1) * 2.0 ** -23


def _operands(name: str, *, B=2, T=5, hidden=48, H=3, d_v=64, bias=True, gain=True):
    torch.manual_seed(0)
    plan = ProducerPlan(routing=rola.Routing(_ROUTINGS[name], gain=gain), num_heads=H, hidden_size=hidden, d_v=d_v)
    C = plan.routing.columns
    route_W = hidden_contiguous(torch.randn(H, hidden, C, device="cuda").to(torch.bfloat16))
    route_bias = torch.randn(H, C, device="cuda") if bias else None
    gain_bias = torch.zeros(H, plan.routing.gain_columns, device="cuda")
    v_W = torch.randn(H * d_v, hidden, device="cuda").to(torch.bfloat16)
    x = torch.randn(B, T, hidden, device="cuda").to(torch.bfloat16)
    return plan, x, route_W, route_bias, gain_bias, v_W


def _launch(plan, x, route_W, route_bias, gain_bias, v_W, fill: int, probe: bool = False):
    """One launch into buffers poisoned with ``fill``; the outputs, the arrival counters and the probes after it."""
    from rola.ops._ext import extension

    B, T, hidden = x.shape
    H, D, d_v = plan.num_heads, plan.routing.topology.D, plan.d_v
    rows = row_layout(plan)[1]

    def poisoned(*shape, dtype=torch.float32):
        out = torch.empty(B, H, T, *shape, dtype=dtype, device="cuda")
        out.view(torch.uint8).fill_(fill)
        return out

    out = {"amp_read": poisoned(rows, dtype=torch.uint8), "amp_write": poisoned(rows, dtype=torch.uint8),
           "scale_read": poisoned(D), "scale_write": poisoned(D), "mass_read": poisoned(D), "mass_write": poisoned(D),
           "read_mass": poisoned(), "g_write": poisoned(), "v": poisoned(d_v, dtype=torch.float16), "v_scale": poisoned()}
    jobs, shapes, per_head = _job_tensor(plan, "cuda:0")
    arrivals = torch.zeros(-(-(B * T) // _TOKEN_TILE) * H, dtype=torch.int32, device="cuda")
    probes = [torch.full((jobs.shape[0], B * T, _MAX_COLUMNS), float("nan"), device="cuda") if probe else None
              for _ in range(2)]
    extension().rola_produce_forward(
        x.reshape(B * T, hidden).contiguous(), route_W, route_bias, gain_bias, v_W, jobs,
        *out.values(), arrivals, *probes, shapes, D, per_head)
    torch.cuda.synchronize()
    return {key: None if t is None else t.transpose(1, 2) for key, t in out.items()}, arrivals, probes


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
    for key in (key for key in first if first[key] is not None):
        assert torch.equal(first[key].contiguous().view(torch.uint8), second[key].contiguous().view(torch.uint8)), (
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
    _, _, (logits, _) = _launch(plan, x, route_W, route_bias, gain_bias, v_W, fill=0, probe=True)
    for index, (product, envelope) in enumerate(_expected_logits(plan, x, route_W, route_bias, v_W)):
        got = logits[index, :, :product.shape[1]].double()
        excess = (got - product).abs() - _projection_rtol(hidden) * envelope
        assert excess.max() <= 0, (
            f"{name}: job {index}'s logit (token, column) {divmod(int(excess.argmax()), product.shape[1])} is off by "
            f"{float((got - product).abs().flatten()[excess.argmax()]):.3g} against an envelope of "
            f"{float(envelope.flatten()[excess.argmax()]):.3g}")


#: the job table's solve codes (produce.cuh `SolveKind`) as the reference's activations
_ACTIVATIONS = {0: rola.softmax(), 1: rola.entmax(1.5), 2: rola.entmax(2.0)}
_UNION_ALPHA = {3: 1.5, 4: 2.0}


def _solve_rtol(width: int) -> float:
    """The solve's per-slot bound on a level (every level is at most 1): the fp32 threshold's prefix sums over at most
    ``width`` sorted members of the shifted row, and the exponentials of softmax and the union's write side, each a few
    units in the last place a member. Measured below it by the factor the test prints on failure."""
    return (width + 16) * 2.0 ** -23


@pytest.mark.parametrize(("name", "B", "T"), [("union", 2, 5), ("split", 1, 40), ("tied-deep", 3, 7),
                                              ("wide", 1, 33), ("sparsemax", 2, 9)])
def test_the_solve_matches_the_fp64_reference_on_the_kernels_logits(name, B, T):
    plan, x, route_W, route_bias, gain_bias, v_W = _operands(name, B=B, T=T, hidden=64)
    _, _, (logits, levels) = _launch(plan, x, route_W, route_bias, gain_bias, v_W, fill=0, probe=True)
    for index, (kind, _head, _heads, solve, width, _in0, in1, out_read, out_write, *_rest) in enumerate(
            job_table(plan)[0]):
        if kind != 0:
            continue
        row = logits[index].double()
        if solve in _UNION_ALPHA:
            read, write = _torch_union_split(row[:, :width], row[:, width:2 * width], _UNION_ALPHA[solve])
        else:
            read = write = _torch_activation(row[:, :width], _ACTIVATIONS[solve])
        for side, expected, out in (("read", read, out_read), ("write", write, out_write)):
            if out < 0:
                continue
            got = levels[index, :, (0 if side == "read" else width):][:, :width].double()
            error = float((got - expected).abs().max())
            assert error <= _solve_rtol(width), (
                f"{name}: job {index} (solve {solve}, width {width}) {side} side is off by {error:.3g}, "
                f"{error / _solve_rtol(width):.2f} of the bound")


#: the bundle's per-slot bounds against the fp64 composition. A u8 digit is a level within the solve's bound of the
#: reference's, rounded: the same digit or its neighbour. A scale is a row's max (the write side's over its mass, about
#: 1): twice the solve's bound. The read mass and the write
#: gain: measured at most 4.8e-5 and 2.7e-6 relative over these cells.
_MASS_RTOL = 2.0 ** -12


@pytest.mark.parametrize(("name", "B", "T", "bias", "gain"), [
    ("union", 2, 5, True, True), ("split", 1, 40, True, True), ("tied-deep", 3, 7, False, True),
    ("wide", 1, 33, True, True), ("sparsemax", 2, 9, True, False),   # no gain column: the gain is 1
])
def test_the_bundle_matches_the_fp64_composition(name, B, T, bias, gain):
    plan, x, route_W, route_bias, gain_bias, v_W = _operands(name, B=B, T=T, hidden=64, bias=bias, gain=gain)
    gain_bias = torch.randn_like(gain_bias)
    out, _, _ = _launch(plan, x, route_W, route_bias, gain_bias, v_W, fill=0x55)
    reference = ProducerWeights(route_W=route_W.double(), gain_bias=gain_bias.double(), v_W=v_W.double(),
                                route_bias=None if route_bias is None else route_bias.double())
    routes, _ = produce(x.double(), plan, reference)
    offsets = row_layout(plan)[0]
    for level, width in enumerate(plan.routing.resolved.branches):
        for side, rows, scales in (("read", routes.amp_read, routes.scale_read),
                                   ("write", routes.amp_write, routes.scale_write)):
            got = out[f"amp_{side}"][..., offsets[level]:offsets[level] + width]
            digit = int((got.int() - rows[level].int()).abs().max())
            assert digit <= 1, f"{name}: level {level} {side} digits off by {digit}"
            scale = float((out[f"scale_{side}"][..., level].double() - scales[level].double()).abs().max())
            assert scale <= (width + 16) * 2.0 ** -22, f"{name}: level {level} {side} scale off by {scale:.3g}"
    read_mass = routes.read_mass if routes.read_mass is not None else torch.ones_like(routes.g_write)
    for key, expected in (("read_mass", read_mass), ("g_write", routes.g_write)):
        error = float(((out[key].double() - expected.double()).abs() / expected.double().abs()).max())
        assert error <= _MASS_RTOL, f"{name}: {key} off by {error:.3g} relative"


@pytest.mark.parametrize(("name", "B", "T"), [("union", 2, 5), ("wide", 1, 33), ("split", 1, 40)])
def test_the_values_match_the_fp64_composition(name, B, T):
    """Per slot, one f16 unit in the last place (at the larger of the two units) plus the projection's fp32 error,
    ``(hidden + 1) 2^-23`` of the envelope sum |x| |W|, the channel's and the scale's, over the token's scale."""
    hidden, H, d_v = 64, 3, 64
    plan, x, route_W, route_bias, gain_bias, v_W = _operands(name, B=B, T=T, hidden=hidden, H=H, d_v=d_v)
    out, _, _ = _launch(plan, x, route_W, route_bias, gain_bias, v_W, fill=0x55)
    reference = ProducerWeights(route_W=route_W.double(), gain_bias=gain_bias.double(), v_W=v_W.double(),
                                route_bias=None if route_bias is None else route_bias.double())
    _, values = produce(x.double(), plan, reference)

    envelope = (x.double().abs() @ v_W.double().abs().t()).reshape(B, T, H, d_v)
    error = (hidden + 1) * 2.0 ** -23 * envelope
    scale = values.scale.double()[..., None]
    got, want = out["v"].double(), values.v.double()
    top = torch.maximum(got.abs(), want.abs()).clamp_min(2.0 ** -14)
    ulp = torch.exp2(torch.floor(torch.log2(top)) - 10)
    excess = (got - want).abs() - (ulp + (error + error.amax(-1, keepdim=True)) / scale)
    assert excess.max() <= 0, f"{name}: a value off by {float(excess.max()):.3g} past its bound"
    scale_excess = (out["v_scale"].double() - values.scale.double()).abs() - error.amax(-1)
    assert scale_excess.max() <= 0, f"{name}: a value scale off by {float(scale_excess.max()):.3g} past its bound"


def test_a_bound_ledger_books_every_phase_and_changes_nothing():
    """The phase ledger (docs/internals/produce/produce.md#phase-ledger): every warp of every job books, a route job
    every route phase and a value job its own, and the outputs are the same bytes with it bound or not."""
    from rola.ops import produce as producer

    operands = _operands("union", B=2, T=40, hidden=64)
    plain, _, _ = _launch(*operands, fill=0x55)
    jobs = job_table(operands[0])[0]
    ledger = torch.zeros(len(jobs), producer.LEDGER_WARPS, len(producer.LEDGER_PHASES), dtype=torch.int64,
                         device="cuda")
    producer.ledger_bind(ledger)
    try:
        booked, _, _ = _launch(*operands, fill=0x55)
    finally:
        producer.ledger_bind(None)
    for key in (key for key in plain if plain[key] is not None):
        same = torch.equal(plain[key].contiguous().view(torch.uint8), booked[key].contiguous().view(torch.uint8))
        assert same, f"binding moved `{key}`"
    phases = dict(zip(producer.LEDGER_PHASES, range(len(producer.LEDGER_PHASES)), strict=True))
    route = [name for name in producer.LEDGER_PHASES if name != "values"]
    for index, job in enumerate(jobs):
        names = route if job[0] == 0 else ["job", "project", "barrier", "values"]
        for name in names:
            assert bool((ledger[index, :, phases[name]] > 0).all()), f"job {index}: a warp booked no `{name}`"
