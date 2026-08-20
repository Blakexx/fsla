# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CTX CONTRACT: what the reverse pass retains, asserted PER FIELD against a closed form.

R4's whole structure rests on retaining NO state: the state and the schedule are both
replayed, and the pairwise coefficient is recomputed. "Grows slowly" is what a
regression hides in, so the claim here is an exact byte count per retained tensor, and
the gate fails on a field that appears as much as on one that grows.
"""
from __future__ import annotations

import pytest
import torch

from rola.ops import intra as intra_ops
from rola.ops._ext import is_available
from rola.ops.prefill import prefill_autograd
from tests.k31.intra_reference import _simplex

pytestmark = pytest.mark.skipif(not is_available() or not torch.cuda.is_available(),
                                reason="the reverse pass is a CUDA kernel")

WIDTHS, K, M, WINDOW, D_V = (256, 256), 4, 8, 512, 64
DENSE = (intra_ops.DENSE_BOTH, intra_ops.DENSE_BOTH)


def _run(T, chained):
    gen = torch.Generator(device="cuda").manual_seed(5)
    read = tuple(_simplex((1, T, w), None, gen, "cuda", False).reshape(1, T, 1, w).to(
        torch.bfloat16) for w in WIDTHS)
    write = tuple(_simplex((1, T, w), None, gen, "cuda", False).reshape(1, T, 1, w).to(
        torch.bfloat16) for w in WIDTHS)
    g = (0.5 + torch.rand(1, T, 1, device="cuda", generator=gen)).to(torch.bfloat16)
    v = torch.randn(1, T, 1, D_V, device="cuda", generator=gen).to(torch.bfloat16)
    N = WIDTHS[0] * WIDTHS[1]
    s_in = torch.zeros(1, N // 16, 16, D_V + 1, device="cuda") if chained else None
    leaves = [x.clone().requires_grad_(True) for x in read + write]
    out = prefill_autograd(tuple(leaves[:2]), tuple(leaves[2:]),
                           g.clone().requires_grad_(True), v.clone().requires_grad_(True),
                           WIDTHS, k=K, m=M, window=WINDOW, modes=DENSE, state_in=s_in)
    #: the outputs are returned WITH the node, because a freed graph frees the ctx and
    #: the whole subject of this file is what the ctx holds.
    return out[0].grad_fn, out


@pytest.mark.parametrize("T", [1024, 2048])
def test_the_retained_set_is_exactly_the_operands(T):
    """Per field, and the closed form is `O(L)` with a per-topology constant."""
    fn, _alive = _run(T, chained=False)
    saved = [t for t in fn.saved_tensors if t is not None]
    want = {
        ("pread", (1, T, sum(WIDTHS)), torch.bfloat16),
        ("pwrite", (1, T, sum(WIDTHS)), torch.bfloat16),
        ("gwrite", (1, T), torch.bfloat16),
        ("value", (1, T, 1, D_V), torch.bfloat16),
    }
    got = {(tuple(t.shape), t.dtype) for t in saved}
    assert got == {(shape, dt) for _n, shape, dt in want}, (
        f"the retained set is {sorted(got)}; the contract is {sorted(want)}")
    #: NO STATE, NO LISTS, NO THRESHOLDS, NO PAIRWISE COEFFICIENT: nothing retained is
    #: sized in `N`, and nothing is sized in `T x T` or `N x T`.
    N = WIDTHS[0] * WIDTHS[1]
    for t in saved:
        assert t.numel() <= T * (2 * sum(WIDTHS)), f"a retained field is not O(L): {t.shape}"
        assert N not in t.shape, f"a retained field is sized in N: {t.shape}"
    closed = T * (2 * sum(WIDTHS) + 1 + D_V) * 2
    assert sum(t.numel() * t.element_size() for t in saved) == closed


def test_a_chained_call_retains_its_entry_state_and_nothing_more():
    """`state_in` is the caller's own tensor: retained by reference, never copied."""
    fn, _alive = _run(1024, chained=True)
    saved = [t for t in fn.saved_tensors if t is not None]
    N = WIDTHS[0] * WIDTHS[1]
    states = [t for t in saved if t.numel() == N * (D_V + 1)]
    assert len(states) == 1, "the entry state is retained exactly once"
    assert len(saved) == 5, f"the chained ctx is the four operands plus the entry state, got {len(saved)}"
