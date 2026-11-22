# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The liveness words' CONTRACT, and the two class-2 folds.

No facts kernel is built on this line (:mod:`fsla.ops.facts`). What lives here is the contract the pass that reads the
masks and every consumer share -- the word layout (:class:`LivenessLayout`), a reader that turns those words back into
per-token bits, and a pure-torch MODEL of the words (the supplied masks laid out as the pass reads them) and of the
folds, the definition a Hopper schedule is graded against (docs/internals/facts/liveness_contract.md). Liveness is
the MASKS' (:mod:`fsla.routing.masks`): no row is read to find it.

Vocabulary, defined here and used unqualified below: ``D`` = routing depth;
``B_l`` = the PADDED width of level ``l`` (a power of two at or above 16, the
descriptor's); ``b_l`` = its LOGICAL width, which never leaves the host; ``L`` = tokens
in the call; ``BH`` = batch x heads; ``N = prod_l B_l`` = leaves; atom = 16 consecutive
leaves in canonical order, which is the page granule; a BOX = one warp sub-box, an
aligned rectangle in digit space.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from fsla.engine.facts.planes import ATOM_READ, ATOM_WRITTEN
from fsla.routing.masks import PREFILL

#: Tokens are the table's minor axis, 32 to a word: one warp ballot per row.
TOK_BITS = 32

#: A TILE UNION's grain: 64 tokens (two words) of a 32-digit group of one level.
TILE_TOKENS = 64
GROUP_DIGITS = 32

#: The two sides, in table order.
READ, WRITE = 0, 1
SIDES = 2

#: The stored word type. ``uint32_t`` on the device; torch carries it as the
#: same 32 bits signed, exactly as the union table already did.
WORD_DTYPE = torch.int32

#: The page granule, in leaves.
ATOM_LEAVES = 16


def token_words(L: int) -> int:
    return (L + TOK_BITS - 1) // TOK_BITS


@dataclass(frozen=True, slots=True)
class LivenessLayout:
    """Where the bit for (side, level, digit, token) lives.

    One row per DIGIT in canonical order: row ``row_base(l) + d`` is level ``l``'s digit
    ``d``, the column a packed plane of the levels would hold it at.
    """

    D: int
    B: tuple[int, ...]
    L: int

    def __post_init__(self) -> None:
        if len(self.B) != self.D:
            raise ValueError(f"a layout names D={self.D} levels and {len(self.B)} widths")
        if self.L < 1:
            raise ValueError(f"a call carries at least one token; got L={self.L}")

    @classmethod
    def of(cls, fmt, L: int) -> LivenessLayout:
        """The layout a state's FORMAT descriptor implies for a call of ``L`` tokens."""
        return cls(D=fmt.D, B=tuple(fmt.B), L=int(L))

    def row_base(self, level: int) -> int:
        return sum(self.B[:level])

    @property
    def rows(self) -> int:
        return sum(self.B)

    def row_of(self, level: int, digit: int) -> int:
        if not 0 <= digit < self.B[level]:
            raise ValueError(f"level {level} carries {self.B[level]} digits; got {digit}")
        return self.row_base(level) + digit

    @property
    def words(self) -> int:
        return token_words(self.L)

    @property
    def side_words(self) -> int:
        return self.rows * self.words

    @property
    def side_bytes(self) -> int:
        return self.side_words * 4

    @property
    def call_words(self) -> int:
        return SIDES * self.side_words

    def word_index(self, bh: int, side: int, row: int, token: int) -> int:
        return (bh * self.call_words + side * self.side_words + row * self.words
                + token // TOK_BITS)

    def token_bit(self, token: int) -> int:
        return 1 << (token % TOK_BITS)

    def group_base(self, level: int) -> int:
        return sum(-(-width // GROUP_DIGITS) for width in self.B[:level])

    @property
    def groups(self) -> int:
        """A side's 32-digit groups: each level's digits in groups of 32, a narrower level one group."""
        return self.group_base(self.D)

    @property
    def tiles(self) -> int:
        return -(-self.L // TILE_TOKENS)

    def union_index(self, bh: int, side: int, group: int, tile: int) -> int:
        return ((bh * SIDES + side) * self.groups + group) * self.tiles + tile

    @property
    def N(self) -> int:
        n = 1
        for width in self.B:
            n *= width
        return n

    @property
    def atoms(self) -> int:
        return self.N // ATOM_LEAVES

    @property
    def radix_strides(self) -> tuple[int, ...]:
        strides, suffix = [], 1
        for width in reversed(self.B):
            strides.append(suffix)
            suffix *= width
        return tuple(reversed(strides))


def gather_words(masks, layout: LivenessLayout) -> torch.Tensor:
    """THE MODEL OF THE PASS'S TABLE: `[B H_w, g + 1, rows, words]` words laid out from the supplied PREFILL-layout masks
    (:class:`~fsla.routing.masks.Masks`: token-packed words ``[B, H, ceil(L / 32), w_l]`` a level), ``g = H / H_w``;
    batch-major, slots ``0 .. g - 1`` a write head's query heads' read side, slot ``g`` its write side. A level's digits
    at or past its width ``w_l`` are the pad and hold zero bits; no value is tested, every bit is the mask's."""
    if masks.consumer != PREFILL:
        raise ValueError(f"the pass reads the {PREFILL!r}-layout masks, got {masks.consumer!r}")
    sides = []
    for level_masks in (masks.read, masks.write):
        if len(level_masks) != layout.D:
            raise ValueError(f"a side holds {layout.D} levels' masks, got {len(level_masks)}")
        rows = []
        for level, mask in enumerate(level_masks):
            B, H, words, width = mask.shape
            if words != layout.words or width > layout.B[level]:
                raise ValueError(f"level {level}'s mask is {tuple(mask.shape)}; [B, H, {layout.words}, <= {layout.B[level]}]")
            rows.append(torch.nn.functional.pad(mask.permute(0, 1, 3, 2), (0, 0, 0, layout.B[level] - width)))
        sides.append(torch.cat(rows, dim=2).reshape(B * H, layout.rows, layout.words))
    write = sides[1]
    read = sides[0].reshape(write.shape[0], -1, *write.shape[1:])
    return torch.cat((read, write[:, None]), dim=1)


def tile_unions(words: torch.Tensor, layout: LivenessLayout) -> torch.Tensor:
    """THE MODEL OF THE TILE UNIONS: `[BH, slots, rows, words]` words -> `[BH, slots, groups, tiles]` int32, bit ``i`` of
    (group ``g`` of level ``l``, tile ``t``) set when digit ``32 g + i`` of level ``l`` is live at a token in
    ``[64 t, 64 t + 64)``: the OR of the tile's two words, a row a bit."""
    BH, slots = words.shape[:2]
    pad = 2 * layout.tiles - layout.words
    if pad:
        words = torch.cat([words, words.new_zeros((*words.shape[:-1], pad))], dim=-1)
    live = (words.reshape(BH, slots, layout.rows, layout.tiles, 2) != 0).any(-1)
    groups = []
    for level, width in enumerate(layout.B):
        base = layout.row_base(level)
        rows = live[:, :, base:base + width]
        rows = torch.cat([rows, rows.new_zeros((BH, slots, -width % GROUP_DIGITS, layout.tiles))], dim=2)
        groups.append(rows.reshape(BH, slots, -1, GROUP_DIGITS, layout.tiles))
    bits = torch.cat(groups, dim=2).to(torch.int64)
    weights = (1 << torch.arange(GROUP_DIGITS, device=words.device, dtype=torch.int64))[:, None]
    unions = (bits * weights).sum(dim=3)
    return (unions - (unions >= 1 << 31).to(torch.int64) * (1 << 32)).to(WORD_DTYPE)


def unpack(words: torch.Tensor, layout: LivenessLayout) -> torch.Tensor:
    """THE READER: `[BH, 2, rows, words]` words -> `[BH, 2, rows, L]` bool."""
    shifts = torch.arange(TOK_BITS, device=words.device, dtype=torch.int32)
    bits = (words.unsqueeze(-1) >> shifts) & 1
    flat = bits.reshape(*words.shape[:-1], layout.words * TOK_BITS)
    return flat[..., :layout.L].bool()


@dataclass(frozen=True, slots=True)
class Box:
    """One warp sub-box: an aligned rectangle in digit space, per level a run.

    The geometry block derives these; the folds take them as data, so nothing here is
    a second derivation of the carve.
    """

    start: tuple[int, ...]
    span: tuple[int, ...]

    def check(self, layout: LivenessLayout) -> None:
        if len(self.start) != layout.D or len(self.span) != layout.D:
            raise ValueError(f"a box names {layout.D} runs; got {len(self.start)}/{len(self.span)}")
        for level, (start, span) in enumerate(zip(self.start, self.span)):
            width = layout.B[level]
            if span < 1 or span > width or width % span or start % span or start + span > width:
                raise ValueError(f"level {level}: the run [{start}, {start + span}) is not an "
                                 f"aligned divisor run of B_l={width}")

    def leaves(self, layout: LivenessLayout) -> torch.Tensor:
        """The canonical leaf ids this box owns, as a flat tensor."""
        self.check(layout)
        strides = layout.radix_strides
        leaves = torch.zeros(1, dtype=torch.int64)
        for level in range(layout.D):
            digits = torch.arange(self.start[level], self.start[level] + self.span[level],
                                  dtype=torch.int64)
            leaves = (leaves[:, None] + (digits * strides[level])[None, :]).reshape(-1)
        return leaves

    def pages(self, layout: LivenessLayout) -> torch.Tensor:
        """THE BOX -> PAGE MAP: the atoms this box's leaves fall in, deduplicated."""
        return torch.unique(self.leaves(layout) // ATOM_LEAVES)


def box_activity(words: torch.Tensor, layout: LivenessLayout, boxes, side: int) -> torch.Tensor:
    """CLASS 2, THE PRIMARY BITS: `[BH, len(boxes)]` bool, per box over the WHOLE CALL.

    A box is live for a token when EVERY level has a live digit inside that level's
    run -- a token's live leaf set is the product of its per-level live digit sets, so
    the AND over levels is exact and never a relaxation -- and the call-level bit is
    the OR of that over the call's tokens. Nothing is kept per window: a window with
    nothing live for a box is a schedule fact, read off these rows while walking. Every
    level's rows are its mask's words.
    """
    table = words[:, side]
    out = []
    for box in boxes:
        box.check(layout)
        clause = None
        for level in range(layout.D):
            base = layout.row_base(level) + box.start[level]
            run = _or_rows(table, base, box.span[level])
            clause = run if clause is None else clause & run
        out.append((clause != 0).any(dim=-1))
    if not out:
        return torch.zeros((words.shape[0], 0), dtype=torch.bool, device=words.device)
    return torch.stack(out, dim=1)


def _or_rows(table: torch.Tensor, base: int, span: int) -> torch.Tensor:
    run = table[:, base, :]
    for row in range(base + 1, base + span):
        run = run | table[:, row, :]
    return run


def page_activity(layout: LivenessLayout, read_boxes, read_bits, write_boxes,
                  write_bits) -> torch.Tensor:
    """CLASS 2, THE DERIVED BITS: `[BH, N/16]` uint8, the HOST's OR over boxes.

    One READ and one WRITTEN bit per page, in the byte
    ``fsla.engine.facts.planes``'s two constants already name: the page is READ if any
    read-side box that intersects it is read, WRITTEN if any write-side box that
    intersects it is written. ``plan_exact`` allocates on the WRITTEN half, residency
    checks read the READ half, and a decode step consumes this set because its unit is
    the page.
    """
    BH = read_bits.shape[0]
    bits = torch.zeros((BH, layout.atoms), dtype=torch.uint8, device=read_bits.device)
    for boxes, live, flag in ((read_boxes, read_bits, ATOM_READ),
                              (write_boxes, write_bits, ATOM_WRITTEN)):
        for index, box in enumerate(boxes):
            pages = box.pages(layout).to(bits.device)
            hit = live[:, index, None].expand(-1, pages.numel())
            bits[:, pages] |= hit.to(torch.uint8) * flag
    return bits


def atom_spans(layout: LivenessLayout) -> tuple[int, ...]:
    """THE ATOM AS A BOX: the per-level spans of the trailing 16 canonical leaves.

    A page atom is 16 consecutive leaves in canonical order, and canonical order is the
    radix over the levels with the last one minor, so an atom IS an aligned rectangle:
    the last levels whole until their product reaches 16, and an aligned run in the
    level that completes it. That is what makes the exact per-page fold a special case
    of :func:`page_activity` rather than a second derivation.
    """
    spans, remaining = [1] * layout.D, ATOM_LEAVES
    for level in reversed(range(layout.D)):
        span = min(layout.B[level], remaining)
        if remaining % span or layout.B[level] % span:
            raise ValueError(f"level {level}'s width {layout.B[level]} does not tile the "
                             f"{remaining} leaves the page atom still needs")
        spans[level], remaining = span, remaining // span
        if remaining == 1:
            break
    if remaining != 1:
        raise ValueError(f"the topology holds {layout.N} leaves, fewer than one page atom")
    return tuple(spans)


def atom_activity(words: torch.Tensor, layout: LivenessLayout, side: int) -> torch.Tensor:
    """`[BH, N/16]` bool: :func:`box_activity` at the ATOM box set, every box at once, of the words' slot ``side``
    (``READ``, ``WRITE`` ungrouped; a query head's or the write side's grouped).

    The same law -- the OR over tokens of the AND over levels of the OR over the box's
    digit run -- folded as one outer product over the levels, which the atom set admits
    because it is the whole aligned partition and its canonical order is exactly the
    order that product builds.
    """
    spans = atom_spans(layout)
    bits = unpack(words, layout)[:, side]
    BH = bits.shape[0]
    clause = None
    for level in range(layout.D):
        base, width, span = layout.row_base(level), layout.B[level], spans[level]
        run = bits[:, base:base + width, :].reshape(BH, width // span, span, layout.L).any(dim=2)
        clause = run if clause is None else (
            clause[:, :, None, :] & run[:, None, :, :]).reshape(BH, -1, layout.L)
    return clause.any(dim=-1)


def page_bits(words: torch.Tensor, layout: LivenessLayout) -> torch.Tensor:
    """`[BH, N/16]` uint8: the EXACT activity byte, `ATOM_READ | ATOM_WRITTEN`, a write head's (its state's): read
    by any of its query heads (slots ``0 .. g - 1``), written by its write side (slot ``g``).

    :func:`page_activity` is this fold at an arbitrary carve, where a box straddling a
    page makes the answer a sound over-report; at the atom box set the map is a
    bijection and the byte is exact, which is what `plan_exact` allocates on and what a
    residency check reads.
    """
    heads = words.shape[1] - 1
    read = atom_activity(words, layout, READ)
    for slot in range(1, heads):
        read = read | atom_activity(words, layout, slot)
    write = atom_activity(words, layout, heads).to(torch.uint8) * ATOM_WRITTEN
    return read.to(torch.uint8) * ATOM_READ | write
