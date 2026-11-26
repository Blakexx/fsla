# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE MASK-PREPARATION PASS'S HOST REFERENCE, written from the rule (docs/internals/common/plan/leaf_groups.md, docs/internals/common/plan/mask_prepare_layout.md,
docs/internals/mask_prepare/mask_prepare.md) and not from the kernel or the planner's C++: what the pass writes for each SIDE'S LEAF GROUPS, field by field.

Symbols: a SIDE is 0 (read) or 1 (write); ``width`` a level's digits, ``width_bits`` its log2; ``group_bits[l]`` a side's group of a level, log2 of its digits; a LEAF GROUP is one
group on each level of a side, ``bits`` of them as log2 (the sum over levels of ``width_bits - group_bits``); ``window_words`` a window's mask words, 32 tokens a word (a window is the
card's, ``fsla.ops.window.window_of() // 32``, and every function that depends on it takes it, with no default); ``N`` a side's blocks, its batch-heads times the call's windows, in
(batch-head, window) order, batch-heads ``B * H`` with ``H`` inner.

What the pass writes, for each side and each block: ``counts [G]`` (a leaf group's live tokens of the window) and ``lines [G, window_words]`` (its mask words), ``G = 1 << bits``.

UNSPECIFIED, and so never produced by :func:`expected` nor compared: the bytes of a side's region past its blocks (a region is sized at the side's ``most``, its blocks fill the first
``N * leaf_block_bytes(bits)``), and the bytes of a block between its counts and its lines (the counts' round-up to a line).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from fsla.routing.masks import PREFILL, Masks

#: the levels a call holds at most, and its sides
LEVELS = 4
SIDES = 2
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


def carve_groups(width_bits: list[int], read: list[int], write: list[int]) -> Carve:
    """The :class:`Carve`: each side's group bits a level, the boxes, and each side's loop order, from the levels' width bits and each side's densities in 256ths, written from the rule
    (docs/internals/common/plan/prefill_launch.md) and not from the planner: a level's size; the CTA's carve, nine doublings each to the level that saves
    the most fetches (half of each side's left -- the product of size over share where the size is the larger -- for the sides that still need the level), then the smaller
    share, then a level the read side needs, then the lower level; the folds' split; and each side's
    groups raised, the densest level with room in the box first, until a set holds 16 leaves. A side's LOOP ORDER is its levels from the sparsest (outermost) to the densest (innermost),
    equal densities keeping the lower level outside."""
    levels = len(width_bits)

    def size(density: int, bits: int) -> int:
        return min(range(bits + 1), key=lambda b: (abs((256 << b) - (density << bits)), b))

    sizes_r = [size(read[l], width_bits[l]) for l in range(levels)]
    sizes_w = [size(write[l], width_bits[l]) for l in range(levels)]
    carve = [0] * levels
    for _ in range(CTA_LEAF_BITS):
        write_short = sum(max(0, sizes_w[l] - carve[l]) for l in range(levels))
        read_short = sum(max(0, sizes_r[l] - carve[l]) for l in range(levels))

        def key(l: int) -> tuple[int, int, int, int, int]:
            writes, reads = int(sizes_w[l] > carve[l]), int(sizes_r[l] > carve[l])
            if write_short > read_short:
                first, second = writes, reads
            elif read_short > write_short:
                first, second = reads, writes
            else:
                first, second = writes + reads, 0
            return (first, second, -carve[l], reads, -l)

        open_levels = [l for l in range(levels) if carve[l] + 1 <= width_bits[l]]
        if open_levels:
            carve[max(open_levels, key=key)] += 1
    held = [l for l in range(levels) if carve[l] >= 1]
    split = next((l for l in held if sizes_w[l] < width_bits[l]), held[0] if held else -1)
    fold = [carve[l] - (1 if l == split else 0) for l in range(levels)]

    def order_of(density: list[int]) -> list[int]:
        return sorted(range(levels), key=lambda l: (density[l], l))

    def raised(sizes: list[int], density: list[int], box: list[int]) -> list[int]:
        order = order_of(density)
        groups = list(sizes)
        for _ in range(LEAST_SET_BITS):
            room = [l for l in order if groups[l] < box[l]]
            if sum(min(groups[l], box[l]) for l in range(levels)) >= LEAST_SET_BITS or not room:
                break
            groups[room[-1]] += 1
        return groups

    return Carve(raised(sizes_r, read, carve), raised(sizes_w, write, fold), carve, fold, order_of(read), order_of(write))


@dataclass(frozen=True)
class Carve:
    """What the carve gives a call, per level: each side's ``group bits``, the CTA's ``carve`` bits (the read side's box) and the folds' ``fold`` bits (the write side's box), and each
    side's loop order, the levels from the sparsest (outermost) to the densest (innermost)."""

    read_groups: list[int]
    write_groups: list[int]
    carve: list[int]
    fold: list[int]
    read_order: list[int]
    write_order: list[int]

    def side(self, side: int) -> tuple[list[int], list[int], list[int]]:
        """``(group bits, box bits, loop order)`` of ``side``."""
        return (self.read_groups, self.carve, self.read_order) if side == 0 else (self.write_groups, self.fold, self.write_order)


@dataclass(frozen=True)
class Order:
    """A side's leaf groups' order, from leaf_groups.md#order: per level the group index's ``inner`` low bits (placed from bit ``inner_at``, the densest level lowest) and ``outer`` rest
    (placed from bit ``outer_at``, above every inner field, the lowest level lowest), ``bits`` the leaf groups as log2, ``listed`` the levels whose group is smaller than the level."""

    bits: int
    inner: list[int]
    inner_at: list[int]
    outer: list[int]
    outer_at: list[int]
    listed: list[int]
    group_bits: list[int]

    def leaf_of(self, groups: list[int]) -> int:
        """A leaf group's index from one group index a level."""
        leaf = 0
        for l, g in enumerate(groups):
            leaf |= (g & ((1 << self.inner[l]) - 1)) << self.inner_at[l]
            leaf |= (g >> self.inner[l]) << self.outer_at[l]
        return leaf

    def groups_of(self, leaf: int) -> list[int]:
        """One group index a level from a leaf group's index: the inverse of :meth:`leaf_of`."""
        return [((leaf >> self.inner_at[l]) & ((1 << self.inner[l]) - 1)) | (((leaf >> self.outer_at[l]) & ((1 << self.outer[l]) - 1)) << self.inner[l]) for l in range(len(self.inner))]


def order_of(width_bits: list[int], carve: Carve, side: int) -> Order:
    """The side's leaf groups' order, formed from the rule: a level's group index has ``width_bits - group_bits`` bits; its inner field is the box's bits above the group (the box holds
    ``1 << (box - group)`` groups of it, at least none), the rest outer. Inner fields are laid from bit 0 densest level first; outer fields above them, the lowest level first."""
    group_bits, box, order = carve.side(side)
    levels = len(width_bits)
    along = [width_bits[l] - group_bits[l] for l in range(levels)]
    inner = [max(0, box[l] - group_bits[l]) for l in range(levels)]
    outer = [along[l] - inner[l] for l in range(levels)]
    inner_at, at = [0] * levels, 0
    for l in reversed(order):
        inner_at[l], at = at, at + inner[l]
    outer_at = [0] * levels
    for l in range(levels):
        outer_at[l], at = at, at + outer[l]
    return Order(sum(along), inner, inner_at, outer, outer_at, [l for l in range(levels) if along[l] > 0], list(group_bits))


def most(width_bits: list[int], masked: int) -> int:
    """The most leaf groups a side can have, as log2, from the levels' width bits and the levels it has a mask on (``masked``, a bit a level), from leaf_groups.md#most: the smaller of the
    digits of the masked levels (a group a digit, in bits) and the leaves over 16 (the call's leaves in bits, less 4 where it has 32 leaves or more; a smaller call is bounded by its leaves).
    MY READING: the page says a fold holds 16 leaves 'in every call of 32 leaves or more' and 'a smaller call is bounded by its leaves'; for fewer than 32 leaves (under 5 bits) the bound is taken as the
    leaves' bits themselves (a question for the page: the exact bound is not stated)."""
    digits = sum(width_bits[l] for l in range(len(width_bits)) if (masked >> l) & 1)
    leaves = sum(width_bits)
    return min(digits, leaves - 4 if leaves >= 5 else leaves)


def line_bytes(words: int) -> int:
    """A leaf group's line, in bytes: 4 a mask word."""
    return 4 * words


def lines_at(bits: int, words: int) -> int:
    """In a block of ``1 << bits`` leaf groups, the byte where the lines start: the counts' bytes (2 a leaf group) rounded up to a whole number of lines."""
    line = line_bytes(words)
    return -(-(2 << bits) // line) * line


def block_bytes(bits: int, words: int) -> int:
    """A (batch-head, window) block's bytes at ``1 << bits`` leaf groups: the counts' region, then a line a leaf group."""
    return lines_at(bits, words) + (line_bytes(words) << bits)


def layout(width_bits: list[int], masked: list[int], heads: list[int], windows: int, window_words: int) -> tuple[list[int], int]:
    """``(offset of each side's region, the total bytes)``: the sides' regions in side order, each ``heads[side] * windows`` blocks sized at the side's :func:`most` (not its leaf
    groups), from the levels' width bits, each side's ``masked`` levels and ``heads`` (batch times the side's heads)."""
    offsets, total = [], 0
    for side in range(SIDES):
        offsets.append(total)
        total += heads[side] * windows * block_bytes(most(width_bits, masked[side]), window_words)
    return offsets, total


def masked_of(masks: Masks) -> list[int]:
    """Each side's masked levels as a bit a level."""
    assert masks.consumer == PREFILL
    return [sum(1 << l for l, m in enumerate(getattr(masks, name)) if m is not None) for name in ("read", "write")]


def windows_of(tokens: int, window_words: int) -> int:
    """The call's windows."""
    return -(-((tokens + 31) // 32) // window_words)


def _own(tokens: int, windows: int, window_words: int) -> torch.Tensor:
    """The call's own tokens as mask words of every window, int64 ``[windows, window_words]``: every bit below the call's last word, the low ``tokens % 32`` of it, none past it."""
    words = torch.zeros(windows * window_words, dtype=torch.int64)
    whole, rest = divmod(tokens, 32)
    words[:whole] = 0xFFFFFFFF
    if rest:
        words[whole] = (1 << rest) - 1
    return words.reshape(windows, window_words)


def _popcount(words: torch.Tensor) -> torch.Tensor:
    return sum(((words >> bit) & 1) for bit in range(32))


@dataclass(frozen=True)
class Leaves:
    """One side's leaf groups' fields, every block: ``counts [N, G]`` and ``lines [N, G, window_words]``, int64, ``G = 1 << bits``, ``N`` the side's batch-heads times the windows."""

    bits: int
    counts: torch.Tensor
    lines: torch.Tensor


def effective_density(masks: Masks, density: torch.Tensor) -> torch.Tensor:
    """The density the carve is formed from: a level-side with NO mask is walked whole whatever its byte (mask_prepare_layout.md), so its byte is 255 (256 256ths); both sides' densities enter the carve,
    so a missing read mask changes the write side's leaf groups too."""
    assert masks.consumer == PREFILL
    out = density.clone()
    for side, name in enumerate(("read", "write")):
        for level, mask in enumerate(getattr(masks, name)):
            if mask is None:
                out[side, level] = 255
    return out


def groups_of(density: torch.Tensor, widths: list[int]) -> Carve:
    """The call's carve, from its ``[2, D]`` density bytes (a byte ``b`` is ``b + 1`` 256ths) and its levels' widths in digits."""
    bits = [int(w).bit_length() - 1 for w in widths]
    return carve_groups(bits, [int(b) + 1 for b in density[0].tolist()], [int(b) + 1 for b in density[1].tolist()])


def expected(masks: Masks, density: torch.Tensor, widths: list[int], heads: list[int], tokens: int, window_words: int) -> tuple[Leaves, Leaves]:
    """What the pass writes, ``(read side, write side)``, from the rule: for every block, each leaf group's line (the call's own tokens, ANDed on every listed level the side has a
    mask on with the OR of the leaf group's group's digits of that level) and its count (the line's set bits). A level-side with no mask is whole (its density is taken as 255 for the carve), so a listed level always has a mask; a side with no listed level has one leaf group,
    the call's own tokens. ``heads`` is each side's batch-heads, ``density`` uint8 ``[2, D]``, ``widths`` the levels' digits."""
    width_bits = [int(w).bit_length() - 1 for w in widths]
    carve = groups_of(effective_density(masks, density), widths)
    windows = windows_of(tokens, window_words)
    own = _own(tokens, windows, window_words)
    out = []
    for side, name in enumerate(("read", "write")):
        order = order_of(width_bits, carve, side)
        blocks = heads[side] * windows
        leaves = torch.arange(1 << order.bits)
        groups = [torch.zeros_like(leaves) for _ in widths]
        for leaf in range(1 << order.bits):
            for l, g in enumerate(order.groups_of(leaf)):
                groups[l][leaf] = g
        lines = own.unsqueeze(0).expand(heads[side], windows, window_words).reshape(blocks, 1, window_words).repeat(1, 1 << order.bits, 1)
        for l in order.listed:
            mask = getattr(masks, name)[l]
            assert mask is not None, "a listed level always has a mask: a level-side with no mask is whole"
            B, H, words, width = mask.shape
            assert heads[side] == B * H and widths[l] == width, f"the {name} side's level {l} mask is [{B}, {H}, {words}, {width}], the call's {heads[side]} batch-heads and {widths[l]} digits"
            x = (mask.to(torch.int64) & 0xFFFFFFFF).cpu()
            x = torch.cat([x, x.new_zeros(B, H, windows * window_words - words, width)], dim=2)
            size = 1 << carve.side(side)[0][l]
            x = x.reshape(B * H * windows, window_words, width // size, size)
            ored = torch.zeros_like(x[..., 0])
            for digit in range(size):
                ored |= x[..., digit]
            lines &= ored[:, :, groups[l]].permute(0, 2, 1)
        out.append(Leaves(order.bits, _popcount(lines).sum(dim=2), lines))
    return out[0], out[1]


def bytes_of(masks: Masks, widths: list[int], heads: list[int], tokens: int, window_words: int) -> int:
    """The workspace's bytes: both sides' regions at their ``most`` (the layout's size, whatever the densities)."""
    return layout([int(w).bit_length() - 1 for w in widths], masked_of(masks), heads, windows_of(tokens, window_words), window_words)[1]


def read(workspace: torch.Tensor, masks: Masks, density: torch.Tensor, widths: list[int], heads: list[int], tokens: int, window_words: int, side: int) -> Leaves:
    """``side``'s counts and lines as a workspace holds them: the layout's own reading of the bytes (counts uint16 at the block's start, lines uint32 from :func:`lines_at`, a leaf group
    after another, blocks one after another at the side's own leaf groups)."""
    width_bits = [int(w).bit_length() - 1 for w in widths]
    masked = masked_of(masks)
    windows = windows_of(tokens, window_words)
    offset = layout(width_bits, masked, heads, windows, window_words)[0][side]
    bits = order_of(width_bits, groups_of(effective_density(masks, density), widths), side).bits
    size, blocks = block_bytes(bits, window_words), heads[side] * windows
    raw = workspace.cpu()[offset:offset + blocks * size].reshape(blocks, size)
    at = lines_at(bits, window_words)
    counts = raw[:, :2 << bits].clone(memory_format=torch.contiguous_format).view(torch.int16).to(torch.int64) & 0xFFFF
    lines = raw[:, at:at + (line_bytes(window_words) << bits)].clone(memory_format=torch.contiguous_format).view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return Leaves(bits, counts, lines.reshape(blocks, 1 << bits, window_words))
