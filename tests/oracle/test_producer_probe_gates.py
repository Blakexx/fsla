"""The fused producer's PROJECTION gate: the probe's logits against the fp64 projection.

PURPOSE 1 (FIDELITY), per P74's oracle-precision doctrine. The projection half of the
entmax seam split is gated NUMERICALLY, with an envelope DERIVED from the kernel's
declared arithmetic rather than borrowed from a tolerance table -- and derived is the
operative word, because the tile order is ours for the first time:

* operands are rounded to bf16 (unit roundoff ``u_b = 2**-8``);
* every accumulation is fp32 (``u = 2**-24``);
* the K axis is accumulated as ``hidden/16`` sequential ``mma.m16n8k16`` steps.

So the per-element floor is ``(2*u_b + (hidden/16)*u) * sum_k |x_k| |W_k|`` and the gate
reports RATIO-TO-FLOOR. Ratio <= 1 says the kernel is doing exactly the arithmetic it
declares; the teeth case below shows the ratio is not vacuously small.

WHY THE PROBE AND NOT A SECOND GEMM: ``producer_probe`` is the shipping fused kernel's
own tile-GEMM component with the fp32 tile emitted instead of solved, shipped and
ratified as one binary, so what this gate measures is what the shipped path computes --
a call-graph fact (docs/internals/entmax/producer.md#probe), not two implementations that
agree.
"""
from __future__ import annotations

import pytest
import torch

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused producer is CUDA")

#: `(B, T, hidden, H, columns)`. Ragged on every axis that can be ragged: a token count
#: that is not a multiple of the support word's 32, a column count that is not a multiple
#: of the 8-wide mma n-tile, and a `hidden` that is not a multiple of the 32-wide K block.
CELLS = [
    (2, 64, 512, 8, 257),
    (1, 32, 128, 2, 64),
    (1, 100, 256, 3, 130),
    (2, 33, 64, 1, 8),
    (1, 17, 48, 2, 33),
]


def _extension():
    from rola.ops._ext import extension

    return extension()


def _floor(x64: torch.Tensor, w64: torch.Tensor, hidden: int) -> torch.Tensor:
    magnitude = torch.einsum("bld,hdc->blhc", x64.abs(), w64.abs())
    return (2 * 2.0 ** -8 + (hidden / 16) * 2.0 ** -24) * magnitude


def _probe(x, W, bias=None):
    logits = torch.empty(x.shape[0], x.shape[1], W.shape[0], W.shape[2],
                         device=x.device, dtype=torch.float32)
    _extension().producer_probe(x, W, bias, logits)
    return logits


@_CUDA
@pytest.mark.parametrize("cell", CELLS, ids=lambda c: "x".join(str(v) for v in c))
def test_the_probe_projection_lands_inside_its_own_derived_floor(cell):
    B, T, hidden, H, columns = cell
    torch.manual_seed(1234)
    x = torch.randn(B, T, hidden, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(H, hidden, columns, device="cuda", dtype=torch.float32) / hidden ** 0.5
    got = _probe(x, W)
    reference = torch.einsum("bld,hdc->blhc", x.double(), W.double())
    ratio = ((got.double() - reference).abs() / _floor(x.double(), W.double(), hidden)).max()
    assert ratio <= 1.0, f"ratio-to-floor {ratio:.4f} at {cell}"


@_CUDA
def test_the_bias_rides_the_same_tile():
    torch.manual_seed(7)
    B, T, hidden, H, columns = 1, 64, 128, 2, 65
    x = torch.randn(B, T, hidden, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(H, hidden, columns, device="cuda", dtype=torch.float32) / hidden ** 0.5
    bias = torch.randn(H, columns, device="cuda", dtype=torch.float32)
    plain = _probe(x, W)
    biased = _probe(x, W, bias)
    want = bias[None, None, :, :].expand_as(plain)
    #: Not bit-identity: `(acc + bias) - acc` is an fp32 subtraction of two rounded sums,
    #: so the round trip costs an ulp of the SUM and not of the bias.
    assert torch.allclose(biased - plain, want, rtol=0.0,
                          atol=8 * torch.finfo(torch.float32).eps * biased.abs().max())


@_CUDA
def test_teeth_the_floor_rejects_a_coarser_operand_class():
    """The gate is not vacuous, and here is exactly how much teeth it has.

    The floor is a WORST-CASE bound -- it assumes no cancellation across the K axis --
    while a real random-signed dot product cancels, so the shipped kernel sits at ~0.1 of
    it. That headroom is the gate's resolution and is stated rather than hidden: it
    rejects an operand class coarser than the declared one by ~8x or more (five mantissa
    bits below bf16 here), and a one-column tile slip -- the failure an mma fragment
    layout error actually produces -- by ~700x its own margin.
    """
    torch.manual_seed(11)
    B, T, hidden, H, columns = 1, 64, 256, 2, 64
    x = torch.randn(B, T, hidden, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(H, hidden, columns, device="cuda", dtype=torch.float32) / hidden ** 0.5
    reference = torch.einsum("bld,hdc->blhc", x.double(), W.double())
    floor = _floor(x.double(), W.double(), hidden)

    shipped = ((_probe(x, W).double() - reference).abs() / floor).max()
    assert shipped <= 1.0

    coarse = torch.einsum("bld,hdc->blhc", _drop_mantissa_bits(x, 5).double(), W.double())
    assert ((coarse - reference).abs() / floor).max() > 1.0

    #: THE STRUCTURAL TEETH: one column of tile slip, which is the failure an mma fragment
    #: layout error actually produces. It misses by ~70x the floor, against the shipped
    #: kernel's 0.1 -- nearly three orders of magnitude of separation.
    slipped = reference.roll(1, dims=3)
    assert ((slipped - reference).abs() / floor).max() > 10.0


def _drop_mantissa_bits(x: torch.Tensor, bits: int) -> torch.Tensor:
    """`x` with `bits` more mantissa bits cleared than bf16 keeps -- a coarser operand."""
    raw = x.view(torch.int16).to(torch.int32) & 0xFFFF
    return (raw & ~((1 << bits) - 1)).to(torch.int16).view(torch.bfloat16)
