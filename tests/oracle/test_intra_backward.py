# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""R4's intra reverse pass against the fp64 WITHIN-WINDOW reference.

The reference is `tests/oracle/backward_reference.py`'s intra half, which
`test_prefill_backward_reference.py` has already pinned to ``torch.autograd`` through the whole
operator. The digit gradients carry the same support contract the carry's do -- a
level's own factor is dropped, so ``d_pr_l`` is live wherever the WRITE side has support
at that digit and the exact-entmax Jacobian is what makes the rest unreadable.

THE CONTRACT IS TWO-SIDED, and this file asserts both halves. On a side the mode word
declares sparse, that side's digit gradient is compared ON its own support and required
to be EXACTLY ZERO off it -- the producer's VJP never loads an off-support cotangent, so
computing one is work whose only consumer is this test. On a side the mode word leaves
dense there is no support to restrict to and the whole digit axis is compared, which is
what keeps a softmax duty's plane from being quietly truncated.
"""
from __future__ import annotations

import pytest
import torch

from rola.ops import carry as carry_ops
from rola.ops import intra as intra_ops
from rola.ops._ext import is_available
from tests.oracle.backward_reference import intra_backward_reference
from tests.oracle.fixtures import simplex

pytestmark = pytest.mark.skipif(not is_available() or not torch.cuda.is_available(),
                                reason="the reverse pass is a CUDA kernel")

WIDTHS = (intra_ops.LEVEL_WIDTH, intra_ops.LEVEL_WIDTH)
KAPPA = 8.0

DENSE = (intra_ops.DENSE_BOTH, intra_ops.DENSE_BOTH)
SHARED = (intra_ops.BOTH_SPARSE, intra_ops.BOTH_SPARSE)
#: THE GRADED DECLARATION, one entmax duty per level and on the OTHER side each time: it
#: is the production structure, and it is the only cell in which one plane of a level is
#: restricted while the other must stay whole.
ALT = (intra_ops.WRITE_SPARSE, intra_ops.READ_SPARSE)

CELLS = [
    pytest.param(intra_ops.WINDOW_SMALL, 128, None, DENSE, 1, id="W=64 dense"),
    pytest.param(intra_ops.WINDOW_SMALL, 192, None, DENSE, 2, id="W=64 3 windows"),
    pytest.param(intra_ops.WINDOW_SMALL, 128, 8, SHARED, 3, id="W=64 k_tok=8"),
    pytest.param(intra_ops.WINDOW_WIDE, 512, None, DENSE, 4, id="W=512 dense"),
    pytest.param(intra_ops.WINDOW_WIDE, 1024, 16, SHARED, 5, id="W=512 k_tok=16"),
    pytest.param(intra_ops.WINDOW_SMALL, 128, 8, ALT, 6, id="W=64 ALT k_tok=8"),
    pytest.param(intra_ops.WINDOW_WIDE, 1024, 16, ALT, 7, id="W=512 ALT k_tok=16"),
]


def _draw(T, k_tok, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    shape = (1, T, 1)
    read = tuple(simplex(shape + (w,), k_tok, gen, "cuda") for w in WIDTHS)
    write = tuple(simplex(shape + (w,), k_tok, gen, "cuda") for w in WIDTHS)
    gw = torch.rand(*shape, device="cuda", dtype=torch.float64, generator=gen) + 0.5
    v = torch.randn(*shape, 64, device="cuda", dtype=torch.float64, generator=gen)
    d_num = torch.randn(1, T, 1, 64, device="cuda", dtype=torch.float64, generator=gen)
    d_den = torch.randn(1, T, 1, device="cuda", dtype=torch.float64, generator=gen)
    return read, write, gw, v, d_num, d_den


@pytest.mark.parametrize("window,T,k_tok,modes,seed", [p.values for p in CELLS],
                         ids=[p.id for p in CELLS])
def test_the_intra_cotangents_match_the_reference(window, T, k_tok, modes, seed):
    assert intra_ops.backward_stamp() > 0
    read, write, gw, v, d_num, d_den = _draw(T, k_tok, seed)
    read_bf = tuple(x.to(torch.bfloat16) for x in read)
    write_bf = tuple(x.to(torch.bfloat16) for x in write)
    gw_bf, v_bf = gw.to(torch.bfloat16), v.to(torch.bfloat16)
    dnum_bf = d_num.to(torch.bfloat16)

    pr = carry_ops.pack_side(read_bf)
    pw = carry_ops.pack_side(write_bf)
    gwp = gw_bf.permute(0, 2, 1).contiguous().reshape(1, T)
    d_pr, d_pw, d_gw, d_v = intra_ops.intra_backward(
        pr, pw, gwp, v_bf.contiguous(), dnum_bf.reshape(1, T, 64).contiguous(),
        d_den.to(torch.float32).reshape(1, T).contiguous(),
        modes, window=window)

    want = intra_backward_reference(
        tuple(x.double() for x in read_bf), tuple(x.double() for x in write_bf),
        gw_bf.double(), v_bf.double(), WIDTHS, window, dnum_bf.double(),
        d_den.to(torch.float32).double())

    def flat(levels):
        return torch.cat([x.permute(0, 2, 1, 3).reshape(1, T, x.shape[-1]) for x in levels], -1)

    #: The kernel reads the STAGED bf16 operand, so the support the gate sees is the bf16
    #: factor's, not the fp64 draw's.
    def restriction(sides, bit):
        live = []
        for side, w, mode in zip(sides, WIDTHS, modes):
            plane = side.permute(0, 2, 1, 3).reshape(1, T, w)
            live.append((plane != 0) if (mode & bit) else torch.ones_like(plane, dtype=torch.bool))
        return torch.cat(live, -1)

    rows = {
        "d_pr": (d_pr, flat(want[0]), restriction(read_bf, intra_ops.READ_SPARSE)),
        "d_pw": (d_pw, flat(want[1]), restriction(write_bf, intra_ops.WRITE_SPARSE)),
        "d_gw": (d_gw, want[2].permute(0, 2, 1).reshape(1, T), None),
        "d_v": (d_v, want[3].permute(0, 2, 1, 3).reshape(1, T, 64), None),
    }
    for name, (got, ref, live) in rows.items():
        scale = max(1e-30, float(ref.abs().max()))
        if live is None:
            live = torch.ones_like(got, dtype=torch.bool)
        else:
            dead = int((got[~live] != 0).sum())
            assert dead == 0, f"{name}: {dead} nonzero cells off the declared support"
        err = float((got.double() - ref)[live].abs().max()) / scale
        assert err < KAPPA * 2 ** -9, f"{name}: {err:.3e}"
