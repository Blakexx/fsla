# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""THE DECODE FOLD: the declared summation order, held to BIT-IDENTITY.

The step's operand pass is one kernel (``csrc/rola/src/decode/decode_fold.cu``). Two of
the three things it does are bit-exact for free -- the ``[B, 1, H, W] -> [B*H, W]`` fold
is the same bytes at ``T = 1``, and a widening cast rounds nothing. The third, the
per-level normalizing sum, has a summation ORDER, and an order nobody wrote down is an
order no gate can hold anything to.

So the order is DECLARED -- lane-strided ascending partials over 32 lanes, folded by the
five ``__shfl_down_sync`` steps -- and this file is that declaration's gate: the reference
below reimplements exactly it, in torch, and the kernel is held to ``torch.equal``.

Cell 4 is the protocol's teeth: on a row built so that summation order is observable, the
declared order and a different one DISAGREE, so the reference is answering a question
about order rather than about arithmetic in general.

Run: ``pytest tests/integration/test_decode_fold.py``
"""
from __future__ import annotations

import pytest
import torch

from rola.ops.decode import DecodeScratch, derive_decode_geometry, fold_levels, fold_operands
from tests.oracle.oracle_fixtures import _topology

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the fold is a CUDA kernel"),
]

_DTYPES = (torch.float32, torch.bfloat16)


def declared_row_sum(row: torch.Tensor) -> torch.Tensor:
    """``[BH, W] -> [BH]`` fp32 in the kernel's DECLARED order, and in no other."""
    BH, width = row.shape
    lanes = torch.zeros(BH, 32, dtype=torch.float32, device=row.device)
    for lane in range(32):
        acc = torch.zeros(BH, dtype=torch.float32, device=row.device)
        for k in range(lane, width, 32):
            acc = acc + row[:, k].float()
        lanes[:, lane] = acc
    off = 16
    while off:
        lanes[:, :off] = lanes[:, :off] + lanes[:, off:2 * off]
        off >>= 1
    return lanes[:, 0]


def reference_fold(levels, normalize):
    """The fold's declared answer for one side: widen, then divide the flagged levels."""
    out = []
    for level, needs in zip(fold_levels(levels), normalize, strict=True):
        out.append(level / declared_row_sum(level).unsqueeze(-1) if needs else level)
    return out


def _bundle(widths, *, B=3, H=2, d_v=64, dtype=torch.float32, seed=0, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    kw = dict(dtype=dtype, device=device, generator=gen)
    read = tuple(torch.rand(B, 1, H, w, **kw) for w in widths)
    write = tuple(torch.rand(B, 1, H, w, **kw) for w in widths)
    g_write = torch.rand(B, 1, H, **kw)
    v = torch.rand(B, 1, H, d_v, **kw)
    return read, write, g_write, v


def _fold(widths, read, write, g_write, v, normalize, *, d_v=64):
    topology = _topology(widths)
    config = derive_decode_geometry(topology, d_v=d_v, decay=False,
                                    BH=read[0].shape[0] * read[0].shape[2],
                                    device=read[0].device)
    scratch = DecodeScratch.build(config, read[0].shape[0] * read[0].shape[2],
                                  read[0].device)
    amp_read, amp_write, _clock, _dials, g, folded_v = fold_operands(
        read, write, g_write, v, None, config, scratch, normalize=normalize)
    return amp_read, amp_write, g, folded_v


# ---------------------------------------------------------------------------
# 1. the cast-only half: bit-exact against the torch fold, every dtype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("widths", [(16,), (16, 16), (64, 8), (256,), (4, 4, 4, 4)])
def test_unnormalized_fold_is_the_torch_fold(dtype, widths):
    read, write, g_write, v = _bundle(widths, dtype=dtype)
    normalize = (False,) * len(widths)
    amp_read, amp_write, g, folded_v = _fold(widths, read, write, g_write, v, normalize)
    for got, want in zip(amp_read, fold_levels(read), strict=True):
        assert torch.equal(got, want)
    for got, want in zip(amp_write, fold_levels(write), strict=True):
        assert torch.equal(got, want)
    B, _T, H = g_write.shape
    assert torch.equal(g, g_write.reshape(B * H).float())
    assert torch.equal(folded_v, v.reshape(B * H, v.shape[-1]).float())


# ---------------------------------------------------------------------------
# 2. the declared order, held to bit-identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("widths", [(16,), (16, 16), (64, 8), (256,), (33, 5)])
def test_normalized_read_matches_the_declared_order(dtype, widths):
    read, write, g_write, v = _bundle(widths, dtype=dtype, seed=7)
    normalize = (True,) * len(widths)
    amp_read, amp_write, _g, _v = _fold(widths, read, write, g_write, v, normalize)
    for got, want in zip(amp_read, reference_fold(read, normalize), strict=True):
        assert torch.equal(got, want)
    #: The WRITE side is never normalized, whatever the read side's flags say.
    for got, want in zip(amp_write, fold_levels(write), strict=True):
        assert torch.equal(got, want)


def test_per_level_flags_are_honoured_independently():
    widths = (16, 16, 16)
    normalize = (True, False, True)
    read, write, g_write, v = _bundle(widths, seed=11)
    amp_read, _w, _g, _v = _fold(widths, read, write, g_write, v, normalize)
    for got, want in zip(amp_read, reference_fold(read, normalize), strict=True):
        assert torch.equal(got, want)


# ---------------------------------------------------------------------------
# 3. the fold's refusals
# ---------------------------------------------------------------------------

def test_more_than_one_token_is_refused():
    widths = (16,)
    read, write, g_write, v = _bundle(widths)
    two = tuple(t.repeat(1, 2, 1, 1) for t in read)
    with pytest.raises(RuntimeError, match=r"\[B, 1, H, W\]"):
        _fold(widths, two, write, g_write, v, (False,))


def test_a_dtype_outside_the_instantiated_set_is_refused():
    widths = (16,)
    read, write, g_write, v = _bundle(widths, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="float32 or bfloat16"):
        _fold(widths, read, write, g_write, v, (False,))


@pytest.mark.parametrize("normalize", [(False,), (True,)])
def test_a_strided_view_folds_by_address(normalize):
    """A level is ordinarily a VIEW of one packed plane, and the fold indexes it there."""
    widths = (16,)
    read, write, g_write, v = _bundle((32,), seed=3)
    sliced = tuple(t[..., ::2] for t in read)
    dense = tuple(t.contiguous() for t in sliced)
    amp_read, _w, _g, _v = _fold(widths, sliced, tuple(t[..., :16] for t in write),
                                 g_write, v, normalize)
    for got, want in zip(amp_read, reference_fold(dense, normalize), strict=True):
        assert torch.equal(got, want)


# ---------------------------------------------------------------------------
# 4. the teeth: the reference is answering a question about ORDER
# ---------------------------------------------------------------------------

def test_the_declared_order_is_distinguishable_from_another_one():
    """A row where summation order is OBSERVABLE, so the gate cannot be vacuous."""
    #: Each addend is half an ulp of 1, so ascending accumulation loses every one of
    #: them; the declared order pairs them off in the lanes first and keeps them.
    tiny = 2.0 ** -24
    row = torch.tensor([[1.0] + [tiny] * 63], dtype=torch.float32, device="cuda")
    ascending = torch.zeros(1, dtype=torch.float32, device="cuda")
    for k in range(row.shape[1]):
        ascending = ascending + row[:, k]
    declared = declared_row_sum(row)
    assert not torch.equal(declared, ascending)
    assert declared.item() > ascending.item() == 1.0
