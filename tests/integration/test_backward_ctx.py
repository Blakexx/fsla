# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CTX CONTRACT: what the reverse pass retains, asserted PER FIELD against a closed form.

The reverse pass's whole structure rests on retaining NO state: the state and the
schedule are both replayed, and the pairwise coefficient is recomputed. "Grows slowly"
is what a regression hides in, so the claim here is an exact byte count per retained
tensor, and the gate fails on a field that appears as much as on one that grows.

The cells are the registry's (`benchmarks/cells`), filtered to what the differentiable
entry carries: depth two at the reverse families' level width, the shipped value width,
the dense backing. Nothing here names a window, a ``(k, m)`` box or a carve -- those
axes left the surface, and a test that still named one would be asserting against a
signature no kernel has.
"""
from __future__ import annotations

import pytest
import torch

from benchmarks.cells import carry_cells, realize
from rola.ops import intra as intra_ops
from rola.ops._ext import is_available
from rola.ops.carry import SHIPPED_DV, WINDOW
from rola.ops.prefill import prefill_autograd

pytestmark = pytest.mark.skipif(not is_available() or not torch.cuda.is_available(),
                                reason="the reverse pass is a CUDA kernel")

#: The differentiable entry is built at depth two, at the intra reverse family's level
#: width, and the value width must be one the descriptor ships. The registry is filtered
#: rather than duplicated: a second list of shapes here would be a second registry.
CELLS = tuple(c for c in carry_cells()
              if c.D == 2 and set(c.widths) == {intra_ops.LEVEL_WIDTH}
              and c.dv in SHIPPED_DV and c.backing == "dense"
              and c.tokens % WINDOW == 0)
IDS = [c.name for c in CELLS]


def modes_of(spec):
    """The cell's draw as the intra's per-level declaration."""
    if spec.draw == "dense":
        return (intra_ops.DENSE_BOTH,) * spec.D
    if spec.draw == "both":
        return tuple(intra_ops.BOTH_SPARSE if level == 0 else intra_ops.DENSE_BOTH
                     for level in range(spec.D))
    return tuple(intra_ops.READ_SPARSE if level % 2 else intra_ops.WRITE_SPARSE
                 for level in range(spec.D))


def _run(spec, chained):
    drawn = realize(spec)
    state_in = None
    if chained:
        from rola.ops.carry import state_plane
        state_in = state_plane(spec.descriptor(), 1)
    leaves = [x.clone().requires_grad_(True) for x in drawn.read + drawn.write]
    out = prefill_autograd(tuple(leaves[:spec.D]), tuple(leaves[spec.D:]),
                           drawn.gain.clone().requires_grad_(True),
                           drawn.v.clone().requires_grad_(True), spec.widths,
                           modes=modes_of(spec),
                           sread=intra_ops.pack_support(drawn.read),
                           swrite=intra_ops.pack_support(drawn.write),
                           state_in=state_in)
    #: the outputs are returned WITH the node, because a freed graph frees the ctx and
    #: the whole subject of this file is what the ctx holds.
    return out[0].grad_fn, out


@pytest.mark.parametrize("spec", CELLS, ids=IDS)
def test_the_retained_set_is_exactly_the_operands(spec):
    """Per field, and the closed form is `O(L)` with a per-topology constant."""
    fn, _alive = _run(spec, chained=False)
    saved = [t for t in fn.saved_tensors if t is not None]
    L, wtot = spec.tokens, sum(spec.widths)
    want = {
        ("pread", (1, L, wtot), torch.bfloat16),
        ("pwrite", (1, L, wtot), torch.bfloat16),
        ("gwrite", (1, L), torch.bfloat16),
        ("value", (1, L, 1, spec.dv), torch.bfloat16),
    }
    got = {(tuple(t.shape), t.dtype) for t in saved}
    assert got == {(shape, dt) for _n, shape, dt in want}, (
        f"the retained set is {sorted(got)}; the contract is {sorted(want)}")
    #: NO STATE, NO LISTS, NO THRESHOLDS, NO PAIRWISE COEFFICIENT: nothing retained is
    #: sized in `N`, and nothing is sized in `L x L` or `N x L`.
    for t in saved:
        assert t.numel() <= L * (2 * wtot), f"a retained field is not O(L): {t.shape}"
        assert spec.N not in t.shape, f"a retained field is sized in N: {t.shape}"
    closed = L * (2 * wtot + 1 + spec.dv) * 2
    assert sum(t.numel() * t.element_size() for t in saved) == closed


@pytest.mark.parametrize("spec", CELLS[:1], ids=IDS[:1])
def test_a_chained_call_retains_its_entry_state_and_nothing_more(spec):
    """`state_in` is the caller's own tensor: retained by reference, never copied."""
    fn, _alive = _run(spec, chained=True)
    saved = [t for t in fn.saved_tensors if t is not None]
    states = [t for t in saved if t.numel() == spec.N * (spec.dv + 1)]
    assert len(states) == 1, "the entry state is retained exactly once"
    assert len(saved) == 5, (
        f"the chained ctx is the four operands plus the entry state, got {len(saved)}")
