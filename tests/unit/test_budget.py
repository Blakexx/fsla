"""`rola.routing.budget`'s two general-purpose statistics primitives.

The module used to also carry architecture §4.5's budget-conditioning
arithmetic (target/slack/usage-count machinery) and `rola.ops.losses`'s
training-side Tier 1/Tier 2 losses that consumed it. That whole family --
routing-coherence residual losses plus injected routing factors -- is
DESCOPED (queue P29, 2026-08-06; `docs/internals/DELETIONS.md`). What
remains here is exact, gradient-free, and general: `RouteFactors.
support_counts()`/`touched_bits()` are the surviving consumers.
"""

from __future__ import annotations

import torch

from rola.routing.budget import exact_support_count, touched_bitset_update


def test_exact_support_count_counts_nonzero_entries():
    values = torch.tensor([[1.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    counts = exact_support_count(values)
    assert torch.equal(counts, torch.tensor([2.0, 0.0, 4.0]))


def test_touched_bitset_update_is_or_reduction_and_detached():
    touched = torch.zeros(1, 3, dtype=torch.bool)
    chunk = torch.zeros(1, 2, 3, requires_grad=True)
    with torch.no_grad():
        chunk[0, 0, 0] = 1.0
        chunk[0, 1, 1] = 1.0
    updated = touched_bitset_update(touched, chunk)
    assert torch.equal(updated, torch.tensor([[True, True, False]]))
    assert not updated.requires_grad
    # idempotent: touching the same digit again does not un-touch anything
    updated2 = touched_bitset_update(updated, chunk)
    assert torch.equal(updated2, updated)
