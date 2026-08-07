# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""H10 (coverage-closure): `use_cache=False` must not mutate the caller's cache.

`decode_forward` updates its state argument IN PLACE (by design: one owning CTA
per row, no atomic, bitwise reproducible). The layer's T=1 decode fork hands it
`recurrent_state.to(torch.float32).contiguous()` -- and the cache's carried state
is ALREADY fp32 and contiguous, so both casts are IDENTITY and the kernel writes
straight through into the caller's `RoLAState`. Under `use_cache=True` that is
the intended in-place carry; under `use_cache=False` the layer then discards
`final_state` (`final_state = None`) while the cache it promised not to touch has
ALREADY been advanced one step -- a silent state corruption for any caller doing
a speculative / throwaway step against a kept cache.

The unpaged prefill seam has the pattern this gate holds decode to
(`rola/ops/consumer.py`: `initial_state...to(torch.float32).contiguous().clone()`
-- the clone is what severs the alias).
"""

from __future__ import annotations

import pytest
import torch

from rola.state import RoLAState
from tests.layers.test_decode_dispatch import _layer

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(),
                           reason="the decode fork is a CUDA kernel")


def _prime_fp32(layer, device, *, prefill=4):
    """A CUDA-path prefill, so the cache carries an FP32 CONTIGUOUS state.

    `test_decode_dispatch._prime` runs its prefill under grad, which dispatches to
    the fp64 oracle -- the cache then holds an fp64 state, `.to(torch.float32)`
    COPIES, and the aliasing this file gates never happens (measured: the first
    version of this gate passed vacuously exactly that way). The identity-cast
    precondition is asserted below so the fixture can never rot back."""
    cache = RoLAState()
    x = torch.randn(1, prefill, 128, device=device)
    with torch.no_grad():
        layer(x, past_key_values=cache, use_cache=True)
    rs = cache[0].recurrent_state
    cast = rs.to(torch.float32).contiguous()
    assert cast.data_ptr() == rs.data_ptr(), (
        "the fixture's cached state is not fp32-contiguous; the identity-cast "
        "alias under test is not reachable from it")
    return cache


@pytest.mark.cuda
@_CUDA
def test_a_use_cache_false_decode_step_leaves_the_cache_untouched():
    device = "cuda"
    layer = _layer(device).eval()
    cache = _prime_fp32(layer, device)

    carried = cache[0].recurrent_state
    assert carried is not None
    snapshot = carried.clone()

    x = torch.randn(1, 1, 128, device=device)
    with torch.no_grad():
        layer(x, past_key_values=cache, use_cache=False)
    assert layer.execution_backend_counts.get("decode", 0) >= 1, (
        "the step did not take the decode fork; this gate is about the decode "
        "seam and must run through it")

    assert torch.equal(cache[0].recurrent_state, snapshot), (
        "use_cache=False advanced the caller's cached state anyway: decode_forward "
        "writes in place and the layer's identity casts alias the cache tensor -- "
        "the clone at the seam is missing")


@pytest.mark.cuda
@_CUDA
def test_a_use_cache_true_decode_step_still_carries_in_place():
    """The other direction: the fix must not sever the intended in-place carry."""
    device = "cuda"
    layer = _layer(device).eval()
    cache = _prime_fp32(layer, device)
    snapshot = cache[0].recurrent_state.clone()

    x = torch.randn(1, 1, 128, device=device)
    with torch.no_grad():
        layer(x, past_key_values=cache, use_cache=True)
    assert layer.execution_backend_counts.get("decode", 0) >= 1
    assert not torch.equal(cache[0].recurrent_state, snapshot), (
        "a use_cache=True decode step did not advance the cache at all")
