# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE MASK-PREPARATION PASS'S HOST REFERENCE, written from the rule (docs/internals/common/plan/prefill_launch.md#groups,
docs/internals/common/plan/mask_prepare_layout.md) and not from the kernel: what each level-side's blocks hold, field by field.

Symbols: ``width`` a level-side's digits; ``g`` its group, in digits; ``groups = width / g``; a window is 1,024 tokens, 32 mask words. The bytes
of a block that the layout does not use at the level-side's group are UNSPECIFIED: :func:`read` and :func:`expected` hold the fields the layout
names and nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from fsla.routing.masks import PREFILL, Masks

#: a window's mask words, 32 tokens a word
WINDOW_WORDS = 32
#: the level-sides of a call: `side * LEVELS + level`, side 0 the read side
LEVELS = 4
SIDES = 2 * LEVELS
#: the group sizes a level-side may take, and the density whose byte is the dense flag
GROUP_SIZES = (1, 4, 16, 32, 64, 128, 256)
DENSE_BYTE = 255


def group_of(byte: int, width: int) -> int | None:
    """A level-side's group, in digits, from its density byte: with ``d = byte + 1`` (256ths), the allowed size no larger than ``width``
    nearest to ``d * width / 256`` digits, a tie to the smaller, compared exactly in integers (the distance of a size ``a`` is
    ``|a * 256 - d * width|``). ``None`` at byte 255: the level-side is dense and nothing is written for it."""
    if byte == DENSE_BYTE:
        return None
    live = (byte + 1) * width
    return min((size for size in GROUP_SIZES if size <= width), key=lambda size: (abs(size * 256 - live), size))


def block_bytes(width: int) -> int:
    """A window's block of a level-side ``width`` digits wide, whatever its group: a total and 32 counts a digit, 2 bytes each."""
    return 2 * (WINDOW_WORDS + 1) * width


def counts_at(groups: int) -> int:
    return 2 * groups


def words_at(groups: int) -> int:
    return (2 * (WINDOW_WORDS + 1) * groups + 3) & ~3


def layout(widths: list[int], heads: list[int], windows: int) -> tuple[list[int], int]:
    """``(offset of each level-side's blocks, the bytes of them all)``: ``widths`` and ``heads`` (batch times the side's heads) of each of the
    8 level-sides, 0 heads where it has no mask; level-sides in slot order, a block a (batch-head, window), blocks in (batch-head, window) order."""
    offsets, total = [], 0
    for width, h in zip(widths, heads, strict=True):
        offsets.append(total)
        total += h * windows * block_bytes(width) if h else 0
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
    """One level-side's blocks, every (batch-head, window): ``totals [N, groups]``, ``counts [N, 32, groups]`` and ``words [N, 32, groups]``
    (``None`` at a group of one digit: it keeps none), ``N = heads * windows`` in (batch-head, window) order, all int64."""

    group: int
    totals: torch.Tensor
    counts: torch.Tensor
    words: torch.Tensor | None


def blocks_of(mask: torch.Tensor, group: int) -> Blocks:
    """The fields the rule gives a level-side's mask ``[B, H, ceil(T / 32), width]`` int32 at ``group``: a window past the mask's last word reads as zero
    words; ``words[j][g]`` the OR of the group's digits' words ``j``, ``counts[j][g]`` the group's live tokens of the window before word ``j``, ``totals[g]`` the window's."""
    B, H, words, width = mask.shape
    windows = -(-words // WINDOW_WORDS)
    x = (mask.to(torch.int64) & 0xFFFFFFFF).cpu()
    x = torch.cat([x, x.new_zeros(B, H, windows * WINDOW_WORDS - words, width)], dim=2)
    x = x.reshape(B * H * windows, WINDOW_WORDS, width // group, group)
    ored = torch.zeros_like(x[..., 0])
    for digit in range(group):
        ored |= x[..., digit]
    live = _popcount(ored)
    counts = live.cumsum(dim=1) - live
    return Blocks(group, live.sum(dim=1), counts, ored if group > 1 else None)


def expected(masks: Masks, density: torch.Tensor) -> dict[int, Blocks]:
    """What the pass writes, by slot: the blocks of every present entry whose density byte is not 255 (``density`` is uint8 ``[2, D]``)."""
    out = {}
    for slot, mask in slots_of(masks):
        group = group_of(int(density[slot // LEVELS, slot % LEVELS]), mask.shape[3])
        if group is not None:
            out[slot] = blocks_of(mask, group)
    return out


def bytes_of(masks: Masks) -> int:
    """The workspace's bytes for the bundle: the layout over its present entries (0 for a bundle with none)."""
    entries = slots_of(masks)
    if not entries:
        return 0
    widths, heads = [2] * SIDES, [0] * SIDES
    for slot, mask in entries:
        widths[slot] = int(mask.shape[3])
        heads[slot] = int(mask.shape[0] * mask.shape[1])
    return layout(widths, heads, -(-int(entries[0][1].shape[2]) // WINDOW_WORDS))[1]


def read(workspace: torch.Tensor, masks: Masks, slot: int, group: int) -> Blocks:
    """The fields of ``slot``'s blocks as a workspace holds them at ``group``: the layout's own reading of the bytes."""
    entries = slots_of(masks)
    widths, heads = [2] * SIDES, [0] * SIDES
    for s, mask in entries:
        widths[s] = int(mask.shape[3])
        heads[s] = int(mask.shape[0] * mask.shape[1])
    windows = -(-int(entries[0][1].shape[2]) // WINDOW_WORDS)
    offsets, _ = layout(widths, heads, windows)
    width, groups = widths[slot], widths[slot] // group
    blocks = workspace.cpu()[offsets[slot]:offsets[slot] + heads[slot] * windows * block_bytes(width)].reshape(-1, block_bytes(width))
    totals = blocks[:, :2 * groups].clone(memory_format=torch.contiguous_format).view(torch.int16).to(torch.int64) & 0xFFFF
    at = counts_at(groups)
    counts = blocks[:, at:at + 2 * WINDOW_WORDS * groups].clone(memory_format=torch.contiguous_format).view(torch.int16).to(torch.int64) & 0xFFFF
    words = None
    if group > 1:
        at = words_at(groups)
        words = (blocks[:, at:at + 4 * WINDOW_WORDS * groups].clone(memory_format=torch.contiguous_format).view(torch.int32).to(torch.int64) & 0xFFFFFFFF).reshape(-1, WINDOW_WORDS, groups)
    return Blocks(group, totals, counts.reshape(-1, WINDOW_WORDS, groups), words)
