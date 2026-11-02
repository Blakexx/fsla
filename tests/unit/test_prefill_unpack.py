"""The prefill's unpack of the carry's ``num`` (`rola.ops.prefill.unpack_num`): its one permuted multiply is the
oracle's channel map (`tests/oracle/readout_reference.unpack_num`, the definition: masks, casts and a scatter) times
the token's factor, BIT FOR BIT, for the f16-pair words and for an fp32 ``num``.
"""
from __future__ import annotations

import pytest
import torch

from rola.ops.prefill import unpack_num
from tests.oracle.readout_reference import unpack_num as reference_unpack


def _words(BH: int, L: int, boxes: int, seed: int) -> torch.Tensor:
    """Random finite f16 pairs as the drain's int32 words: each half a random f16 of either sign, never inf or NaN."""
    g = torch.Generator().manual_seed(seed)
    halves = (torch.randn(BH, L, boxes * 32, generator=g) * 1e3).to(torch.float16)
    return halves.view(torch.int32).reshape(BH, L, boxes * 16)


@pytest.mark.parametrize("boxes", [1, 2])
def test_the_pairs_unpack_scaled_as_the_reference_maps_them(boxes):
    num = _words(3, 40, boxes, seed=boxes)
    factor = torch.exp2(torch.randint(-8, 8, (3, 40)).float() + 1.0) * 0.73
    got = unpack_num(num, 32, factor)
    want = reference_unpack(num, 32) * factor.unsqueeze(-1)
    assert got.shape == want.shape and got.is_contiguous()
    assert torch.equal(got.view(torch.int32), want.view(torch.int32))


def test_an_fp32_num_is_scaled_in_place_order():
    num = torch.randn(2, 16, 64)
    assert torch.equal(unpack_num(num, 32, 0.5), num * 0.5)
    factor = torch.rand(2, 16)
    assert torch.equal(unpack_num(num, 32, factor), num * factor.unsqueeze(-1))


def test_a_box_of_other_than_32_channels_is_refused():
    with pytest.raises(ValueError, match="32-channel box"):
        unpack_num(_words(1, 4, 1, seed=0), 64, 1.0)
