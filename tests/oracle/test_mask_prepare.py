# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE MASK-PREPARATION PASS AGAINST ITS HOST REFERENCE, exactly (csrc/fsla/src/mask_prepare/mask_prepare.cu, docs/internals/mask_prepare/mask_prepare.md): each side's leaf groups' lines and counts are
integers, so the comparison is bit for bit (`tests/oracle/mask_prepare_reference.py`, written from the rule), every line and every count, both sides. The bytes of a side's region past its blocks and of a block
between its counts and its lines are unspecified and never compared.

DEVICE TIER: it launches the kernel, so it runs where a GPU is (`bazel test //tests:oracle`); the CPU tier holds the reference's own checks (`tests/unit/test_mask_prepare.py`).

A window is the card's (``window_of()`` tokens, ``window_of() // 32`` mask words): every case states its token counts in windows.

Symbols: ``T`` tokens a head, ``W`` the card's window in tokens, ``B`` batch, ``H`` a side's heads, ``width`` a level's digits, ``k`` the digits a density byte ``k - 1`` makes live (``k`` 256ths), the
side's group at a level of 256 digits where the carve raises none."""
from __future__ import annotations

import pytest
import torch

from fsla.ops.mask_prepare import prepare_masks
from fsla.ops.window import window_of
from fsla.routing.masks import PREFILL, Masks, pack_tokens
from tests.oracle import mask_prepare_reference as reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

RAGGED = 24  #: the tokens a ragged last window falls short of whole words by (its last word holds 8)


def window_words() -> int:
    """The card's window in mask words, 32 tokens a word."""
    return window_of() // 32


def live_mask(B: int, T: int, H: int, width: int, fraction: float, seed: int, *, past: bool = False) -> torch.Tensor:
    """A PREFILL-layout level mask, int32 ``[B, H, ceil(T / 32), width]``, each (token, digit) bit set with probability ``fraction`` (all zero at 0, all ones at 1). With ``past``, every bit of the last word
    past token ``T - 1`` is set too (a caller's garbage: the lines must not hold it)."""
    gen = torch.Generator().manual_seed(seed)
    bits = torch.rand(B, T, H, width, generator=gen) < fraction if 0 < fraction < 1 else torch.full((B, T, H, width), fraction == 1)
    mask = pack_tokens(bits)
    if past and T % 32:
        mask[:, :, -1, :] |= torch.tensor(((0xFFFFFFFF << (T % 32)) & 0xFFFFFFFF) - (1 << 32), dtype=torch.int64).to(torch.int32)
    return mask.cuda()


def byte_for(live: int) -> int:
    """The density byte of ``live`` 256ths: a level of 256 digits has ``live`` live digits (a power of two from 1 to 128)."""
    return live - 1


def bundle(read: dict[int, torch.Tensor], write: dict[int, torch.Tensor], levels: int) -> Masks:
    """The bundle of ``levels`` levels whose present entries are the dicts' (level -> mask), the rest absent."""
    return Masks(PREFILL, tuple(read.get(level) for level in range(levels)), tuple(write.get(level) for level in range(levels)))


def density_of(read: dict[int, int], write: dict[int, int], levels: int) -> torch.Tensor:
    """The call's ``[2, D]`` uint8 density: byte 255 (the dense flag) where a dict (level -> byte) names no level, a level with no mask included."""
    out = torch.full((2, levels), 255, dtype=torch.uint8)
    for level, byte in read.items():
        out[0, level] = byte
    for level, byte in write.items():
        out[1, level] = byte
    return out.cuda()


def heads_of(masks: Masks, default: tuple[int, int] = (1, 1)) -> list[int]:
    """Each side's batch-heads: from the side's first present mask, else ``default``."""
    out = []
    for side, name in enumerate(("read", "write")):
        found = [m.shape[0] * m.shape[1] for m in getattr(masks, name) if m is not None]
        out.append(found[0] if found else default[side])
    return out


def check(masks: Masks, density: torch.Tensor, widths: list[int], heads: list[int], tokens: int, workspace: torch.Tensor | None = None) -> list[int]:
    """The pass on the bundle (the op's own workspace where ``workspace`` is None) holds the reference's counts and lines exactly, on both sides; returns each side's leaf group bits."""
    ww = window_words()
    if workspace is None:
        workspace = prepare_masks(masks, density, widths, heads, tokens)
    torch.cuda.synchronize()
    assert workspace.dtype == torch.uint8 and workspace.numel() == reference.bytes_of(masks, widths, heads, tokens, ww)
    want = reference.expected(masks, density.cpu(), widths, heads, tokens, ww)
    for side, name in enumerate(("read", "write")):
        got = reference.read(workspace, masks, density.cpu(), widths, heads, tokens, ww, side)
        assert got.bits == want[side].bits
        assert torch.equal(got.counts, want[side].counts), f"the {name} side's counts"
        assert torch.equal(got.lines, want[side].lines), f"the {name} side's lines"
    return [want[0].bits, want[1].bits]


def one_side(T: int, group: int, *, side: int = 0, B: int = 1, H: int = 1, fraction: float = 0.5, seed: int = 0, past: bool = False, width: int = 256):
    """A call of two levels of ``width`` and 256 digits whose one mask is ``side``'s level 0, at ``group`` live digits; the second level, whole and unmasked, holds a set at 16 leaves so that the carve raises nothing."""
    mask = live_mask(B, T, H, width, fraction, seed, past=past)
    masks = bundle({0: mask} if side == 0 else {}, {0: mask} if side == 1 else {}, 2)
    density = density_of({0: byte_for(group)} if side == 0 else {}, {0: byte_for(group)} if side == 1 else {}, 2)
    return masks, density, [width, 256], heads_of(masks)


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("group", [1, 2, 4, 16, 32, 128])
def test_one_listed_level_at_each_group_size_over_two_windows(group, side):
    """One level listed: 256 / group leaf groups (256 at a group of 1, 2 at 128), the other side with none (one leaf group)."""
    T = 2 * window_of()
    masks, density, widths, heads = one_side(T, group, side=side, seed=group)
    bits = check(masks, density, widths, heads, T)
    assert bits[side] == 8 - group.bit_length() + 1 and bits[1 - side] == 0


def test_two_listed_levels_on_one_side_4096_leaf_groups_at_256_by_256_with_both_at_about_4_live_digits():
    T = window_of()
    masks = bundle({0: live_mask(1, T, 1, 256, 0.5, 21), 1: live_mask(1, T, 1, 256, 0.5, 22)}, {}, 2)
    density, widths = density_of({0: byte_for(4), 1: byte_for(4)}, {}, 2), [256, 256]
    assert check(masks, density, widths, [1, 1], T) == [12, 0]


@pytest.mark.parametrize("fraction", [0.0, 0.02, 0.5, 0.98, 1.0])
def test_two_listed_levels_on_each_side_at_several_live_fractions(fraction):
    T = window_of()
    masks = bundle({0: live_mask(1, T, 1, 256, fraction, 23), 1: live_mask(1, T, 1, 256, fraction, 24)}, {0: live_mask(1, T, 1, 256, fraction, 25), 1: live_mask(1, T, 1, 256, fraction, 26)}, 2)
    density, widths = density_of({0: byte_for(4), 1: byte_for(4)}, {0: byte_for(4), 1: byte_for(4)}, 2), [256, 256]
    check(masks, density, widths, [1, 1], T)


@pytest.mark.parametrize("unmasked_byte", [0, 3, 63, 254])
@pytest.mark.parametrize("case", ["read side", "write side", "both sides", "read side's absent mask changes the write side"])
def test_a_level_side_with_no_mask_is_whole_whatever_its_density_byte_under_255(case, unmasked_byte):
    """Level 1 has NO mask on a side but its density byte there is under 255 (an absent mask is the dense flag: the level-side is walked whole, 256 256ths, in the carve of both sides). The pass must write what the
    reference gives with that level whole, and the workspace ``prepare_masks`` allocates (sized from the masks present) must hold it (``check`` compares its size with the reference's)."""
    T = window_of() + 40
    mask = lambda seed: live_mask(1, T, 1, 256, 0.5, seed)  # noqa: E731
    byte = byte_for(4)
    if case == "read side":
        masks, density = bundle({0: mask(51)}, {}, 2), density_of({0: byte, 1: unmasked_byte}, {}, 2)
    elif case == "write side":
        masks, density = bundle({}, {0: mask(52)}, 2), density_of({}, {0: byte, 1: unmasked_byte}, 2)
    elif case == "both sides":
        masks, density = bundle({0: mask(53)}, {0: mask(54)}, 2), density_of({0: byte, 1: unmasked_byte}, {0: byte, 1: unmasked_byte}, 2)
    else:
        #: the write side lists both levels; the read side's level 1 has no mask, so its byte (under 255) must still be taken as whole by the carve of the write side's leaf groups
        masks, density = bundle({0: mask(55)}, {0: mask(56), 1: mask(57)}, 2), density_of({0: byte, 1: unmasked_byte}, {0: byte, 1: byte}, 2)
    bits = check(masks, density, [256, 256], heads_of(masks), T)
    assert bits[0] <= reference.most([8, 8], reference.masked_of(masks)[0]) and bits[1] <= reference.most([8, 8], reference.masked_of(masks)[1])


def test_a_side_with_no_mask_at_all_is_one_leaf_group_the_calls_own_tokens_beside_a_side_with_one():
    T = 2 * window_of() - RAGGED
    masks, density, widths, heads = one_side(T, 4, side=0, seed=5)
    assert check(masks, density, widths, heads, T) == [6, 0]
    want = reference.expected(masks, density.cpu(), widths, heads, T, window_words())[1]
    assert want.counts.shape[1] == 1 and int(want.counts[0, 0]) == window_of()


@pytest.mark.parametrize("tokens", [lambda w: w, lambda w: w - RAGGED, lambda w: 33, lambda w: 1, lambda w: 3 * w + 40])
def test_a_call_with_no_mask_on_either_side_writes_both_sides_one_leaf_group_of_the_calls_own_tokens(tokens):
    T = tokens(window_of())
    masks = bundle({}, {}, 2)
    density, widths = density_of({}, {}, 2), [16, 256]
    assert check(masks, density, widths, [2, 1], T) == [0, 0]


@pytest.mark.parametrize("width", [2, 4, 8, 16, 32, 64])
def test_a_level_narrower_than_256(width):
    """Every group a width allows below the level, the density bytes that select them at the level's width (a level of 256 whole and unmasked beside it)."""
    T = window_of()
    for group in (g for g in reference.GROUP_SIZES if g < width):
        byte = next(b for b in range(255) if reference.group_of(b, width) == group)
        mask = live_mask(1, T, 1, width, 0.5, group)
        masks = bundle({0: mask}, {}, 2)
        check(masks, density_of({0: byte}, {}, 2), [width, 256], [1, 1], T)


def test_three_levels_listed_on_both_sides_at_different_widths_with_one_absent_on_each_side():
    T = 2 * window_of()
    read = {0: live_mask(1, T, 1, 256, 0.5, 31), 2: live_mask(1, T, 1, 16, 0.3, 32)}
    write = {1: live_mask(1, T, 1, 64, 0.7, 33), 2: live_mask(1, T, 1, 16, 0.9, 34)}
    masks = bundle(read, write, 3)
    density = density_of({0: byte_for(32), 2: 0}, {1: 63, 2: 3}, 3)
    check(masks, density, [256, 64, 16], [1, 1], T)


@pytest.mark.parametrize(("batch", "read_heads", "write_heads"), [(1, 1, 1), (3, 4, 2), (2, 8, 1)])
def test_batch_and_heads_above_one_and_a_write_side_of_fewer_heads(batch, read_heads, write_heads):
    T = window_of() + 40
    masks = bundle({0: live_mask(batch, T, read_heads, 64, 0.4, 1)}, {0: live_mask(batch, T, write_heads, 64, 0.4, 2)}, 1)
    density = density_of({0: 15}, {0: 3}, 1)
    check(masks, density, [64], [batch * read_heads, batch * write_heads], T)


@pytest.mark.parametrize("length", ["ragged last word", "ragged last window", "fewer than one window", "one token", "a long call"])
@pytest.mark.parametrize("group", [1, 4, 32])
def test_a_ragged_call_tokens_not_a_multiple_of_32_and_a_last_window_missing_whole_words(length, group):
    T = {"ragged last word": window_of() + 8, "ragged last window": 2 * window_of() - RAGGED - 32 * 3, "fewer than one window": 33,
         "one token": 1, "a long call": 8 * window_of() + 5}[length]
    masks, density, widths, heads = one_side(T, group, B=2, H=2, seed=T)
    check(masks, density, widths, heads, T)


@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("T", [lambda w: w - RAGGED, lambda w: 2 * w - 1, lambda w: 17])
def test_mask_bits_set_past_the_calls_last_token_are_not_in_the_lines(fraction, T):
    """Every mask's last word has its bits past ``T`` set; the reference's lines hold none of them, and so must the pass's (compared bit for bit, both sides)."""
    tokens = T(window_of())
    masks = bundle({0: live_mask(1, tokens, 1, 256, fraction, 41, past=True)}, {0: live_mask(1, tokens, 1, 256, fraction, 42, past=True)}, 2)
    density, widths = density_of({0: byte_for(8)}, {0: byte_for(2)}, 2), [256, 256]
    check(masks, density, widths, [1, 1], tokens)
    want = reference.expected(masks, density.cpu(), widths, [1, 1], tokens, window_words())
    for side in want:
        last_word = (tokens - 1) // 32 % window_words()
        assert int((side.lines[-1, :, last_word] >> (tokens % 32 or 32)).count_nonzero()) == 0


@pytest.mark.parametrize("fraction", [0.0, 0.02, 0.5, 0.98, 1.0])
@pytest.mark.parametrize("group", [1, 2, 8, 16, 128])
def test_random_masks_at_several_live_fractions_including_none_and_all(fraction, group):
    T = 2 * window_of()
    masks, density, widths, heads = one_side(T, group, fraction=fraction, seed=int(fraction * 100))
    check(masks, density, widths, heads, T)


def test_the_workspace_is_sized_at_each_sides_most_and_what_the_pass_does_not_write_is_not_compared():
    """A workspace pre-filled with a sentinel: the reference's fields hold wherever the pass writes, whatever lies between."""
    from fsla.ops._ext import extension

    T = window_of() - RAGGED
    masks, density, widths, heads = one_side(T, 4, B=2, H=2, seed=9)
    size = reference.bytes_of(masks, widths, heads, T, window_words())
    assert size == int(extension().mask_prepare_bytes(widths, reference.masked_of(masks), heads, T, torch.cuda.current_device()))
    workspace = torch.full((size,), 0xA5, dtype=torch.uint8, device="cuda")
    extension().mask_prepare([m for m in masks.read if m is not None], [0], widths, heads, T, density, workspace)
    check(masks, density, widths, heads, T, workspace)


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

    T = window_of()
    mask = live_mask(1, T, 1, 16, 0.25, seed=0)
    density = density_of({0: 3}, {}, 1)
    workspace = torch.zeros(reference.bytes_of(bundle({0: mask}, {}, 1), [16], [1, 1], T, window_words()), dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match=words):
        extension().mask_prepare([spoil(mask)], [0], [16], [1, 1], T, density, workspace)


@pytest.mark.parametrize(
    ("widths", "heads", "words"),
    [([16, 16], [1, 1], "the width of each level"), ([32], [1, 1], "its level's"), ([16], [2, 1], "its side's"), ([16], [1], "each side's batch-heads")],
    ids=["a width too many", "a mask of another width", "a mask of other batch-heads", "one side's heads"],
)
def test_arguments_that_are_not_the_calls_are_refused_by_name(widths, heads, words):
    from fsla.ops._ext import extension

    T = window_of()
    mask = live_mask(1, T, 1, 16, 0.25, seed=0)
    density = density_of({0: 3}, {}, 1)
    workspace = torch.zeros(reference.bytes_of(bundle({0: mask}, {}, 1), [16], [1, 1], T, window_words()), dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match=words):
        extension().mask_prepare([mask], [0], widths, heads, T, density, workspace)


def test_a_call_of_no_tokens_is_refused_by_name():
    from fsla.ops._ext import extension

    with pytest.raises(RuntimeError, match="a token or more"):
        extension().mask_prepare_bytes([16], [0, 0], [1, 1], 0, torch.cuda.current_device())
