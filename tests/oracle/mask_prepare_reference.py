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
    """A level-side's SIZE, in digits, from its density byte (its group where the carve raises none): with ``d = byte + 1`` (256ths), the allowed size no larger than ``width``
    nearest to ``d * width / 256`` digits, a tie to the smaller, compared exactly in integers (the distance of a size ``a`` is
    ``|a * 256 - d * width|``). At byte 255 the group is the level itself, whatever its width."""
    live = (byte + 1) * width
    return min((size for size in GROUP_SIZES if size <= width), key=lambda size: (abs(size * 256 - live), size))


#: a set's least leaves and the doublings a CTA's carve shares among the levels, as log2
LEAST_SET_BITS = 4
CTA_LEAF_BITS = 9


def carve_groups(width_bits: list[int], read: list[int], write: list[int]) -> tuple[list[int], list[int]]:
    """``(read group bits, write group bits)`` a level, from the levels' width bits and each side's densities in 256ths, written from the rule
    (docs/internals/common/plan/prefill_launch.md) and not from the planner: a level's size; the CTA's carve, nine doublings each to the level of
    the largest summed density over its holdings plus one, then the larger read density, then the lower level; the folds' split; and each side's
    groups raised, the densest level with room in the box first, until a set holds 16 leaves."""
    levels = len(width_bits)

    def size(density: int, bits: int) -> int:
        return min(range(bits + 1), key=lambda b: (abs((256 << b) - (density << bits)), b))

    sizes_r = [size(read[l], width_bits[l]) for l in range(levels)]
    sizes_w = [size(write[l], width_bits[l]) for l in range(levels)]
    carve = [0] * levels
    for _ in range(CTA_LEAF_BITS):
        best = None
        for l in range(levels):
            if carve[l] + 1 > width_bits[l]:
                continue
            if best is None:
                best = l
                continue
            lhs, rhs = (read[l] + write[l]) * (carve[best] + 1), (read[best] + write[best]) * (carve[l] + 1)
            if lhs > rhs or (lhs == rhs and read[l] > read[best]):
                best = l
        if best is not None:
            carve[best] += 1
    held = [l for l in range(levels) if carve[l] >= 1]
    split = next((l for l in held if sizes_w[l] < width_bits[l]), held[0] if held else -1)
    fold = [carve[l] - (1 if l == split else 0) for l in range(levels)]

    def raised(sizes: list[int], density: list[int], box: list[int]) -> list[int]:
        order = sorted(range(levels), key=lambda l: (density[l], l))
        groups = list(sizes)
        for _ in range(LEAST_SET_BITS):
            room = [l for l in order if groups[l] < box[l]]
            if sum(min(groups[l], box[l]) for l in range(levels)) >= LEAST_SET_BITS or not room:
                break
            groups[room[-1]] += 1
        return groups

    return raised(sizes_r, read, carve), raised(sizes_w, write, fold)


def groups_of(density: torch.Tensor, widths: list[int]) -> dict[int, int]:
    """Every level-side's group, in digits, by slot: the call's carve's, from its ``[2, D]`` density bytes (a byte ``b`` is ``b + 1`` 256ths) and its levels' widths."""
    bits = [int(w).bit_length() - 1 for w in widths]
    read = [int(b) + 1 for b in density[0].tolist()]
    write = [int(b) + 1 for b in density[1].tolist()]
    groups = carve_groups(bits, read, write)
    return {side * LEVELS + level: 1 << g for side in range(2) for level, g in enumerate(groups[side])}


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


def expected(masks: Masks, density: torch.Tensor, widths: list[int], window_words: int) -> dict[int, Blocks]:
    """What the pass writes, by slot: the blocks of every present entry whose group, the carve's, is not the level's size (``density`` is uint8 ``[2, D]``, ``widths`` the levels' digits)."""
    out = {}
    groups = groups_of(density, widths)
    for slot, mask in slots_of(masks):
        if groups[slot] != int(mask.shape[3]):
            out[slot] = blocks_of(mask, groups[slot], window_words)
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
