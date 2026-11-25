# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE MASK-PREPARATION PASS AGAINST ITS HOST REFERENCE, exactly (csrc/fsla/src/mask_prepare/mask_prepare.cu, docs/internals/mask_prepare/mask_prepare.md): every group's words and each group's total of live
tokens of the window are integers, so the comparison is bit for bit (`tests/oracle/mask_prepare_reference.py`, written from the rule). The bytes of a block
the layout does not use at the level-side's group are unspecified and never compared; a level-side whose group is the level's size is whole (byte 255 always) and its region is left as it was.

DEVICE TIER: it launches the kernel, so it runs where a GPU is (`bazel test //tests:oracle`); the CPU tier holds the reference's own checks (`tests/unit/test_mask_prepare.py`).

A window is the card's (``window_of()`` tokens, ``window_of() // 32`` mask words): every case states its token counts in windows.

Symbols: ``T`` tokens, ``W`` the card's window in tokens, ``B`` batch, ``H`` a side's heads, ``width`` a level-side's digits, ``g`` its group in digits."""
from __future__ import annotations

import pytest
import torch

from fsla.ops.mask_prepare import prepare_masks
from fsla.ops.window import window_of
from fsla.routing.masks import PREFILL, Masks, pack_tokens
from tests.oracle import mask_prepare_reference as reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

SENTINEL = 0xA5
LEVELS = reference.LEVELS
RAGGED = 24  #: the tokens a ragged last window falls short of whole words by (its last word holds 8)


def window_words() -> int:
    """The card's window in mask words, 32 tokens a word."""
    return window_of() // 32


def live_mask(B: int, T: int, H: int, width: int, fraction: float, seed: int) -> torch.Tensor:
    """A PREFILL-layout level mask, int32 ``[B, H, ceil(T / 32), width]``, each (token, digit) bit set with probability ``fraction`` (all zero at 0, all ones at 1)."""
    gen = torch.Generator().manual_seed(seed)
    bits = torch.rand(B, T, H, width, generator=gen) < fraction if 0 < fraction < 1 else torch.full((B, T, H, width), fraction == 1)
    return pack_tokens(bits).cuda()


def byte_for(group: int, width: int) -> int:
    """A density byte whose group at ``width`` is ``group``."""
    return next(byte for byte in range(255) if reference.group_of(byte, width) == group)


def bundle(read: dict[int, torch.Tensor], write: dict[int, torch.Tensor], levels: int) -> Masks:
    """The bundle of ``levels`` levels whose present entries are the dicts' (level -> mask), the rest absent."""
    return Masks(PREFILL, tuple(read.get(level) for level in range(levels)), tuple(write.get(level) for level in range(levels)))


def density_of(bytes_: dict[int, int], levels: int) -> torch.Tensor:
    """The call's ``[2, D]`` uint8 density: byte 255 where ``bytes_`` (slot -> byte) names no slot."""
    out = torch.full((2, levels), 255, dtype=torch.uint8)
    for slot, byte in bytes_.items():
        out[slot // LEVELS, slot % LEVELS] = byte
    return out.cuda()


def launch(masks: Masks, density: torch.Tensor, fill: int | None = None) -> torch.Tensor:
    """The pass on the bundle into a workspace pre-filled with ``fill`` (the op's own allocation where ``None``)."""
    if fill is None:
        return prepare_masks(masks, density)
    from fsla.ops._ext import extension

    entries = reference.slots_of(masks)
    workspace = torch.full((reference.bytes_of(masks, window_words()),), fill, dtype=torch.uint8, device="cuda")
    extension().mask_prepare([m for _, m in entries], [s for s, _ in entries], density, workspace)
    return workspace


def check(masks: Masks, density: torch.Tensor, workspace: torch.Tensor) -> None:
    """Every present level-side that is not whole holds the reference's fields exactly."""
    torch.cuda.synchronize()
    want = reference.expected(masks, density, window_words())
    assert want, "the cell prepares no level-side"
    for slot, blocks in want.items():
        got = reference.read(workspace, masks, slot, blocks.group, window_words())
        assert torch.equal(got.totals, blocks.totals), f"slot {slot}: totals at group {blocks.group}"
        if blocks.words is not None:
            assert torch.equal(got.words, blocks.words), f"slot {slot}: words at group {blocks.group}"


def one_side(T: int, width: int, group: int, *, B: int = 1, H: int = 1, fraction: float = 0.5, seed: int = 0):
    masks = bundle({0: live_mask(B, T, H, width, fraction, seed)}, {}, 1)
    return masks, density_of({0: byte_for(group, width)}, 1)


@pytest.mark.parametrize("width", [2, 4, 8, 16, 32, 64, 256])
def test_each_group_smaller_than_the_level_a_width_allows(width):
    for group in (g for g in reference.GROUP_SIZES if g < width):
        masks, density = one_side(window_of(), width, group)
        check(masks, density, launch(masks, density))


@pytest.mark.parametrize("group", [g for g in reference.GROUP_SIZES if g < 256])
def test_each_of_the_eight_groups_smaller_than_256_digits_over_two_windows(group):
    masks, density = one_side(2 * window_of(), 256, group, seed=group)
    check(masks, density, launch(masks, density))


#: the call's token counts, each a function of the card's window `W`: one window, two windows, a ragged last window, a partial window of two words, a long call
LENGTHS = {"one window": lambda w: w, "two windows": lambda w: 2 * w, "a ragged last window": lambda w: w - RAGGED, "a partial window": lambda w: 33, "a long call": lambda w: 64 * w}


@pytest.mark.parametrize("length", list(LENGTHS))
@pytest.mark.parametrize(("width", "group"), [(16, 1), (16, 4), (256, 2), (256, 8), (256, 16), (256, 128)])
def test_each_length_whole_windows_ragged_words_a_partial_window_and_a_long_call(length, width, group):
    tokens = LENGTHS[length](window_of())
    masks, density = one_side(tokens, width, group, seed=tokens)
    check(masks, density, launch(masks, density))


@pytest.mark.parametrize(("batch", "read_heads", "write_heads"), [(1, 1, 1), (3, 4, 2), (2, 8, 1)])
def test_several_batch_heads_and_a_write_side_of_fewer_heads(batch, read_heads, write_heads):
    masks = bundle({0: live_mask(batch, window_of() - RAGGED, read_heads, 64, 0.4, 1)}, {0: live_mask(batch, window_of() - RAGGED, write_heads, 64, 0.4, 2)}, 1)
    density = density_of({0: byte_for(16, 64), 4: byte_for(4, 64)}, 1)
    check(masks, density, launch(masks, density))


def test_several_level_sides_in_one_launch_with_some_entries_absent():
    two = 2 * window_of()
    read = {0: live_mask(2, two, 2, 256, 0.5, 3), 2: live_mask(2, two, 2, 16, 0.2, 4)}
    write = {1: live_mask(2, two, 1, 64, 0.7, 5), 2: live_mask(2, two, 1, 4, 0.9, 6)}
    masks = bundle(read, write, 3)
    density = density_of({0: byte_for(32, 256), 2: byte_for(1, 16), 5: byte_for(64, 64), 6: byte_for(4, 4)}, 3)
    check(masks, density, launch(masks, density))


def test_a_present_mask_at_byte_255_leaves_its_region_as_it_was():
    read = {0: live_mask(1, window_of(), 1, 16, 0.5, 7), 1: live_mask(1, window_of(), 1, 16, 0.5, 8)}
    masks = bundle(read, {}, 2)
    density = density_of({0: byte_for(4, 16)}, 2)
    workspace = launch(masks, density, fill=SENTINEL)
    check(masks, density, workspace)
    dense_at = reference.layout([16] * reference.SIDES, [1, 1] + [0] * 6, 1, window_words())[0][1]
    assert (workspace[dense_at:dense_at + reference.block_bytes(16, window_words())] == SENTINEL).all().item()


@pytest.mark.parametrize("width", [2, 4, 8, 16, 256])
def test_a_level_side_whose_group_is_the_level_is_whole_below_byte_255_too_and_leaves_its_region_as_it_was(width):
    read = {0: live_mask(1, window_of(), 1, width, 0.5, 11), 1: live_mask(1, window_of(), 1, width, 0.9, 12)}
    masks = bundle(read, {}, 2)
    density = density_of({0: byte_for(1, width), 1: byte_for(width, width)}, 2)
    workspace = launch(masks, density, fill=SENTINEL)
    check(masks, density, workspace)
    whole_at = reference.layout([width] * reference.SIDES, [1, 1] + [0] * 6, 1, window_words())[0][1]
    assert (workspace[whole_at:whole_at + reference.block_bytes(width, window_words())] == SENTINEL).all().item()


@pytest.mark.parametrize("fraction", [0.0, 0.02, 0.5, 0.98, 1.0])
@pytest.mark.parametrize("group", [1, 2, 8, 16, 128])
def test_random_masks_at_several_live_fractions_including_none_and_all(fraction, group):
    masks, density = one_side(2 * window_of(), 256, group, fraction=fraction, seed=int(fraction * 100))
    check(masks, density, launch(masks, density))


def test_a_bundle_with_no_present_entry_returns_an_empty_tensor_and_launches_nothing():
    masks = bundle({}, {}, 2)
    out = prepare_masks(masks, density_of({}, 2))
    assert out.dtype == torch.uint8 and out.numel() == 0


def test_the_ops_byte_count_is_the_references_layout():
    from fsla.ops._ext import extension

    masks = bundle({0: live_mask(2, window_of() - RAGGED, 3, 64, 0.5, 9)}, {1: live_mask(2, window_of() - RAGGED, 2, 16, 0.5, 10)}, 2)
    workspace = launch(masks, density_of({0: 5, 5: 5}, 2))
    assert workspace.numel() == reference.bytes_of(masks, window_words())
    widths, heads = [2] * reference.SIDES, [0] * reference.SIDES
    widths[0], heads[0], widths[5], heads[5] = 64, 6, 16, 4
    assert extension().mask_prepare_bytes([w.bit_length() - 1 for w in widths], heads, window_words(), torch.cuda.current_device()) == reference.layout(widths, heads, 1, window_words())[1]


@pytest.mark.parametrize(
    ("spoil", "words"),
    [
        (lambda mask: mask.float(), "wrong dtype"),
        (lambda mask: mask.repeat(1, 1, 1, 2)[..., ::2], "is contiguous"),
        (lambda mask: mask.cpu(), "density's card"),
    ],
    ids=["a float mask", "a strided mask", "a host mask"],
)
def test_an_operand_the_kernel_cannot_read_is_refused_by_name(spoil, words):
    """The op reads a mask as a contiguous int32 array on the density's card and refuses any other, naming what is wrong."""
    from fsla.ops._ext import extension

    mask = live_mask(1, window_of(), 1, 16, 0.25, seed=0)
    density = density_of({0: byte_for(4, 16)}, 1)
    workspace = torch.zeros(reference.bytes_of(bundle({0: mask}, {}, 1), window_words()), dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match=words):
        extension().mask_prepare([spoil(mask)], [0], density, workspace)
