# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The COMBINED operator's gradient, against the fp64 reference for the whole recurrence.

The mirror of `test_prefill_vs_oracle.py`: the rung's earlier reverse gates compared
each half against its own statement of the window split, and nothing then checked that
the three reverse kernels' grids were the SAME grid. Here one autograd node runs all
three and the whole cotangent set is compared against the reference the decomposition
gate already pinned to ``torch.autograd`` -- so a grid disagreement, a double-counted
diagonal or a dropped tile pair in the GRADIENT has nowhere to hide either.
"""
from __future__ import annotations

import zlib

import pytest
import torch

from rola.ops import intra as intra_ops
from rola.ops._ext import is_available
from rola.ops.prefill import prefill_autograd
from tests.k31.backward_reference import carry_backward_reference, intra_backward_reference
from tests.k31.fixtures import assert_fresh_binary, canonical_from_plane
from tests.k31.intra_reference import _simplex
from tests.k31.test_prefill_vs_oracle import ALT, DENSE, READ_ONE, WIDTHS, K, M, require_carry_arm

pytestmark = pytest.mark.skipif(not is_available() or not torch.cuda.is_available(),
                                reason="the reverse pass is a CUDA kernel")

N = WIDTHS[0] * WIDTHS[1]
#: ONE KAPPA PER OUTPUT, published: a ratio to the rounded-reference floor.
KAPPA = 6.0

CELLS = [
    ("dense-W512", 1024, 512, DENSE, None, False),
    ("dense-W512-chained", 1024, 512, DENSE, None, True),
    ("alt-k4-W512", 1024, 512, ALT, 4, False),
    ("read-one-k4-W512", 1024, 512, READ_ONE, 4, False),
]


def _cell(T, modes, k_tok, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    read, write = [], []
    for level, mode in enumerate(modes):
        width = WIDTHS[level]
        read_k = k_tok if mode in (intra_ops.READ_SPARSE, intra_ops.BOTH_SPARSE) else None
        write_k = k_tok if mode in (intra_ops.WRITE_SPARSE, intra_ops.BOTH_SPARSE) else None
        read.append(_simplex((1, T, width), read_k, gen, "cuda", False))
        write.append(_simplex((1, T, width), write_k, gen, "cuda", False))
    read = tuple(x.reshape(1, T, 1, WIDTHS[i]).to(torch.bfloat16).double()
                 for i, x in enumerate(read))
    write = tuple(x.reshape(1, T, 1, WIDTHS[i]).to(torch.bfloat16).double()
                  for i, x in enumerate(write))
    g = (torch.rand(1, T, 1, device="cuda", dtype=torch.float64, generator=gen) + 0.5)
    v = torch.randn(1, T, 1, 64, device="cuda", dtype=torch.float64, generator=gen)
    return read, write, g.to(torch.bfloat16).double(), v.to(torch.bfloat16).double(), gen


@pytest.mark.parametrize("name,T,window,modes,k_tok,chained", CELLS, ids=[c[0] for c in CELLS])
def test_every_cotangent_of_the_combined_operator(name, T, window, modes, k_tok, chained):
    #: THE fp64 REFERENCE MATERIALIZES `[T, N]` PER WINDOW; a card that cannot hold it
    #: skips with the reason and the class rather than narrowing the claim.
    free = torch.cuda.get_device_properties(0).total_memory
    if free < 10 * 2 ** 30:
        pytest.skip(f"N={N}, T={T}: the fp64 reverse reference exceeds this card; rental-tier")
    assert_fresh_binary()
    require_carry_arm(window, modes)
    seed = zlib.crc32(name.encode()) & 0xFFFF
    read, write, g, v, gen = _cell(T, modes, k_tok, seed)
    state_in = (0.02 * torch.randn(1, N // 16, 16, 65, device="cuda", dtype=torch.float32,
                                  generator=gen)) if chained else None

    leaves = [x.clone().requires_grad_(True) for x in read + write]
    gv = g.clone().requires_grad_(True)
    vv = v.clone().requires_grad_(True)
    num, den, state_out = prefill_autograd(tuple(leaves[:2]), tuple(leaves[2:]), gv, vv,
                                           WIDTHS, k=K, m=M, window=window, modes=modes,
                                           sread=intra_ops.pack_support(read),
                                           swrite=intra_ops.pack_support(write),
                                           state_in=state_in)
    d_num = torch.randn(num.shape, device="cuda", dtype=torch.float32, generator=gen)
    d_den = torch.randn(den.shape, device="cuda", dtype=torch.float32, generator=gen)
    d_state = torch.randn(state_out.shape, device="cuda", dtype=torch.float32, generator=gen)
    got = torch.autograd.grad((num, den, state_out), leaves + [gv, vv], (d_num, d_den, d_state))
    del num, den, state_out
    torch.cuda.empty_cache()

    dn_bth = d_num.double().reshape(1, 1, T, 64).permute(0, 2, 1, 3)
    dd_bth = d_den.double().reshape(1, 1, T).permute(0, 2, 1)
    sin64 = (canonical_from_plane(state_in.double()).reshape(1, 1, N, 65)
             if chained else None)
    dso64 = canonical_from_plane(d_state.double()).reshape(1, 1, N, 65)
    c_read, c_write, c_gw, c_v, _ = carry_backward_reference(
        read, write, g, v, WIDTHS, window, dn_bth, dd_bth, sin64, dso64)
    torch.cuda.empty_cache()
    i_read, i_write, i_gw, i_v = intra_backward_reference(read, write, g, v, WIDTHS, window,
                                                          dn_bth, dd_bth)
    torch.cuda.empty_cache()

    want = ([c_read[l] + i_read[l] for l in range(2)]
            + [c_write[l] + i_write[l] for l in range(2)] + [c_gw + i_gw, c_v + i_v])
    names = ["d_read0", "d_read1", "d_write0", "d_write1", "d_gw", "d_v"]
    #: the digit gradients are exact ON EACH LEVEL'S OWN SUPPORT and partial off it --
    #: the carry's schedule visits an owner only where every level's support meets it,
    #: and the exact-entmax Jacobian is zero exactly where the partial sums live.
    masks = [x != 0 for x in read] + [x != 0 for x in write] + [None, None]
    for label, a, b, mask in zip(names, got, want, masks):
        scale = max(1e-30, float(b.abs().max()))
        sel = (slice(None),) if mask is None else (mask,)
        err = float((a.double() - b)[sel].abs().max()) / scale
        assert err < KAPPA * 2 ** -9, f"{name}/{label}: {err:.3e}"
