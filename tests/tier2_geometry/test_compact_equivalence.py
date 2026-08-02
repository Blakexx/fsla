"""TIER 2, class (a) -- the compact gather is an ENGINEERED IDENTITY.

Design §6.4's transfer table: this gate's home is tier 2, because it relates the
kernel to ITSELF at two geometries rather than to the oracle. It stays a
hand-written module rather than dissolving into `matrix.py` because it carries a
16-cell GRID (two single-owner rectangles x both `BT` x both normalizations x both
decay arms) over one matrix row's fixture -- the row declares the contract, this
file exercises its breadth. `matrix.py`'s `COMPACT` rows and this file assert the
same claim; M2 keeps the row honest and this keeps the coverage.

`_run_case` and `_decay_dials` are imported from the tier-1 module rather than
re-written, for the reason §6.1 gives about second references: a freshly written
copy proves self-consistency only.
"""
from __future__ import annotations

import pytest
import torch

from tests.tier1_semantics.test_consumer_vs_oracle import _decay_dials, _run_case

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the consumer is a CUDA kernel"),
]


@pytest.mark.parametrize("widths,BC", [((16,), 16), ((8, 8), 64)])
@pytest.mark.parametrize("BT", [16, 32])
@pytest.mark.parametrize("norm", ["raw", "global"])
@pytest.mark.parametrize("decay", [False, True])
def test_the_two_compact_arms_are_bit_identical_at_saturation(widths, BC, BT, norm, decay):
    """At `K == BC` the two instantiations must agree BIT for BIT, not to a tolerance.

    That is the whole argument for making `compact` a free plan-time choice: the
    compaction is ASCENDING, so at saturation `c_idx[k] == k` and the compacted
    path issues the same loads in the same order as the compiled-out one, with the
    same FMA order. Anything less than bit equality would mean the planner's
    performance decision moved numbers, and the scatter-fill identity argument
    (`0.0f` for an amplitude, `1.0f` for a survival factor) would be doing work it
    is not entitled to do.

    **SINGLE-OWNER RECTANGLES ONLY, and that restriction is load-bearing.** With
    more than one owner per head the split-K `atomicAdd` across concurrent owner
    CTAs makes the output nondeterministic run to run by construction (see
    `test_split_k_determinism_is_not_assumed`), so a bit comparison there would be
    testing the scheduler, not the kernel. `(16,)/BC=16` and `(8,8)/BC=64` are
    exactly the rectangles with `owners_total == 1`.

    The fixture is fully DENSE, which is what puts every tile at `K == BC`; the
    sparse case is not bit-identical and is not claimed to be -- it is a genuinely
    shorter contraction, gated against the oracle above."""
    from math import prod

    assert prod(widths) // BC == 1, "this gate is only meaningful on a single-owner rectangle"
    kwargs = dict(
        nonzero_m0=True, seed=3141, return_outputs=True,
        decay_dials=(_decay_dials(widths, 3, torch.Generator(device="cuda").manual_seed(3141),
                                  "cuda") if decay else None))
    y_c, state_c = _run_case(widths, BC, BT, 64, 2, 37, 3, norm, 1.0, 1.0,
                             compact=True, **kwargs)
    y_d, state_d = _run_case(widths, BC, BT, 64, 2, 37, 3, norm, 1.0, 1.0,
                             compact=False, **kwargs)
    for name, a, b in (("y", y_c, y_d), ("final state", state_c, state_d)):
        assert torch.equal(a.view(torch.int32), b.view(torch.int32)), (
            f"{name} is not BIT-identical between compact=True and compact=False at a "
            f"saturated (fully dense) fixture: max |delta| = {float((a - b).abs().max()):.3e}. "
            "At K == BC the gather returns the identity, so the two instantiations must "
            "issue the same loads in the same order.")
