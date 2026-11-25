# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE MASK-PREPARATION PASS'S HOST REFERENCE, written from the rule (docs/internals/common/plan/prefill_launch.md#groups,
docs/internals/common/plan/mask_prepare_layout.md) and not from the kernel: what each level-side's blocks hold, field by field.

Symbols: ``width`` a level-side's digits; ``g`` its group, in digits; ``groups = width / g``; ``window_words`` a window's mask words, 32 tokens a word (a window is the card's, ``fsla.ops.window.window_of() // 32``, and every function that depends on it takes it, with no default). The bytes
of a block that the layout does not use at the level-side's group are UNSPECIFIED: :func:`read` and :func:`expected` hold the fields the layout
names and nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from fsla.routing.masks import PREFILL, Masks

#: the level-sides of a call: `side * LEVELS + level`, side 0 the read side
LEVELS = 4
SIDES = 2 * LEVELS
#: the group sizes a level-side may take: every power of two from 1 to 256
GROUP_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def group_of(byte: int, width: int) -> int:
    """A level-side's group, in digits, from its density byte: with ``d = byte + 1`` (256ths), the allowed size no larger than ``width``
    nearest to ``d * width / 256`` digits, a tie to the smaller, compared exactly in integers (the distance of a size ``a`` is
    ``|a * 256 - d * width|``). At byte 255 the group is the level itself, whatever its width."""
    live = (byte + 1) * width
    return min((size for size in GROUP_SIZES if size <= width), key=lambda size: (abs(size * 256 - live), size))


def whole(byte: int, width: int) -> bool:
    """Whether a level-side is WHOLE: its group is the level's size. Its mask is never read and nothing is written for it."""
    return group_of(byte, width) == width


def words_at(groups: int) -> int:
    """In a block of ``groups`` groups: the totals at 0 (uint16 a group), then the words (uint32, word-major; none at a group of one digit): the words start at the totals' bytes, rounded up to 4."""
    return (2 * groups + 3) & ~3


def block_bytes(width: int, window_words: int) -> int:
    """A window's block of a level-side ``width`` digits wide, whatever its group, the window ``window_words`` mask words: the larger of ``words_at(width)`` (a total a digit) and,
    where ``width >= 4``, ``words_at(width // 2) + 4 * window_words * (width // 2)`` (groups of 2 digits: the totals, then a word a mask word a group)."""
    ones = words_at(width)
    if width < 4:
        return ones
    return max(ones, words_at(width // 2) + 4 * window_words * (width // 2))


def layout(widths: list[int], heads: list[int], windows: int, window_words: int) -> tuple[list[int], int]:
    """``(offset of each level-side's blocks, the bytes of them all)``: ``widths`` and ``heads`` (batch times the side's heads) of each of the
    8 level-sides, 0 heads where it has no mask, over ``windows`` windows of ``window_words`` mask words; level-sides in slot order, a block a (batch-head, window), blocks in (batch-head, window) order."""
    offsets, total = [], 0
    for width, h in zip(widths, heads, strict=True):
        offsets.append(total)
        total += h * windows * block_bytes(width, window_words) if h else 0
    return offsets, total


def slots_of(masks: Masks) -> list[tuple[int, torch.Tensor]]:
    """The bundle's present entries as ``(slot, mask)``, in slot order."""
    assert masks.consumer == PREFILL
    return [(side * LEVELS + level, mask)
            for side, name in enumerate(("read", "write"))
            for level, mask in enumerate(getattr(masks, name)) if mask is not None]


def _popcount(words: torch.Tensor) -> torch.Tensor:
    return sum(((words >> bit) & 1) for bit in range(32))


@dataclass(frozen=True)
class Blocks:
    """One level-side's blocks, every (batch-head, window): ``totals [N, groups]`` and ``words [N, window_words, groups]``
    (``None`` at a group of one digit: it keeps none), ``N = heads * windows`` in (batch-head, window) order, all int64."""

    group: int
    totals: torch.Tensor
    words: torch.Tensor | None


def blocks_of(mask: torch.Tensor, group: int, window_words: int) -> Blocks:
    """The fields the rule gives a level-side's mask ``[B, H, ceil(T / 32), width]`` int32 at ``group``, in windows of ``window_words`` mask words: a window past the mask's last word reads as zero
    words; ``words[j][g]`` the OR of the group's digits' words ``j``, ``totals[g]`` the group's live tokens of the window."""
    B, H, words, width = mask.shape
    windows = -(-words // window_words)
    x = (mask.to(torch.int64) & 0xFFFFFFFF).cpu()
    x = torch.cat([x, x.new_zeros(B, H, windows * window_words - words, width)], dim=2)
    x = x.reshape(B * H * windows, window_words, width // group, group)
    ored = torch.zeros_like(x[..., 0])
    for digit in range(group):
        ored |= x[..., digit]
    live = _popcount(ored)
    return Blocks(group, live.sum(dim=1), ored if group > 1 else None)


def expected(masks: Masks, density: torch.Tensor, window_words: int) -> dict[int, Blocks]:
    """What the pass writes, by slot: the blocks of every present entry that is not whole (``density`` is uint8 ``[2, D]``)."""
    out = {}
    for slot, mask in slots_of(masks):
        byte, width = int(density[slot // LEVELS, slot % LEVELS]), int(mask.shape[3])
        if not whole(byte, width):
            out[slot] = blocks_of(mask, group_of(byte, width), window_words)
    return out


def bytes_of(masks: Masks, window_words: int) -> int:
    """The workspace's bytes for the bundle: the layout over its present entries (0 for a bundle with none)."""
    entries = slots_of(masks)
    if not entries:
        return 0
    widths, heads = [2] * SIDES, [0] * SIDES
    for slot, mask in entries:
        widths[slot] = int(mask.shape[3])
        heads[slot] = int(mask.shape[0] * mask.shape[1])
    return layout(widths, heads, -(-int(entries[0][1].shape[2]) // window_words), window_words)[1]


def read(workspace: torch.Tensor, masks: Masks, slot: int, group: int, window_words: int) -> Blocks:
    """The fields of ``slot``'s blocks as a workspace holds them at ``group``: the layout's own reading of the bytes."""
    entries = slots_of(masks)
    widths, heads = [2] * SIDES, [0] * SIDES
    for s, mask in entries:
        widths[s] = int(mask.shape[3])
        heads[s] = int(mask.shape[0] * mask.shape[1])
    windows = -(-int(entries[0][1].shape[2]) // window_words)
    offsets, _ = layout(widths, heads, windows, window_words)
    width, groups = widths[slot], widths[slot] // group
    blocks = workspace.cpu()[offsets[slot]:offsets[slot] + heads[slot] * windows * block_bytes(width, window_words)].reshape(-1, block_bytes(width, window_words))
    totals = blocks[:, :2 * groups].clone(memory_format=torch.contiguous_format).view(torch.int16).to(torch.int64) & 0xFFFF
    words = None
    if group > 1:
        at = words_at(groups)
        words = (blocks[:, at:at + 4 * window_words * groups].clone(memory_format=torch.contiguous_format).view(torch.int32).to(torch.int64) & 0xFFFFFFFF).reshape(-1, window_words, groups)
    return Blocks(group, totals, words)
