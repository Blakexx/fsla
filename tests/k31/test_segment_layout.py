# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""K35 S2 -- the per-stream SEGMENT's named layouts, gated by a round trip.

A named SMEM layout is derived from BOTH of its access patterns and gated by a round
trip (KERNEL_STANDARDS section 3).  This file is that gate for the three layouts the
cooperative fetch introduces:

  ``SegValueRows``   ``[stream][row][channel]`` at row stride ``LdV = d_v + 8``.
                     WRITER: the fetcher's ``cp.async``, one sixteen-byte quantum per
                     lane, eight lanes to a row.  READER: the consumer's
                     ``ldmatrix.trans`` at ``rT/cT``.
  ``SegCoeffRows``   ``[stream][row][slot]``    at row stride ``LdS = kLeafW + 8``.
                     WRITER: the expansion's ``st.shared.u32``, one packed pair per
                     lane, ``kLeafW/2`` lanes to a row, TWO rows at a time.  READER:
                     the consumer's ``ldmatrix.trans`` at ``rF/cF``.
  ``StagedFactorRow````[row][amplitude]`` at stride ``BP::kStageW``: the owner's digit
                     run at each level, level by level, each run aligned to its own
                     asynchronous quantum.  WRITER: ``cp.async`` of 4, 8 or 16 bytes.
                     READER: the expansion's scalar ``ld.shared.u16``/``u32``.  ONE layout
                     serves BOTH sides -- the fold stages from ``pwrite``, the readout
                     from ``pread`` -- so this gate covers the read staging too.
  ``ReaderSegRows``  ``[reader][row][slot]`` at row stride ``LdR = kLeafE + 8`` (K35 S4b).
                     WRITER: the READ expansion's ``st.shared.u32``.  READER: the
                     readout's ``load_frag``, whose fragment map puts the segment's ROWS
                     on the MMA's ``n`` axis, because on the readout the rows are TOKENS.
  ``RingRows``       ``[slot][channel]`` fp32 at stride ``LdRing = d_v + 8`` (K35 S4b).
                     WRITER: the scatter's ``red.shared.add.f32``, four ring rows at one
                     eight-channel granule per issue.  READER: the flush, thirty-two
                     lanes over thirty-two consecutive channels.
  ``ParkedLo``       ``[word][lane]`` over the warp's OWN exchange segment (K35 S5):
                     ``kParkWords`` opaque packed-pair words per lane, written and read
                     by that warp alone.  WRITER and READER are the same burst of
                     ``st.shared.u32``/``ld.shared.u32``, 32 consecutive words each --
                     the one layout here that is NOT the layout of the bytes it takes.

THE DEVICE LEG proves the two address maps AGREE: the probe writes a pattern whose
sixteen bits NAME the cell it was stored at and hands back what each fragment
recovered, at the fragment's own thread-value map.  THE MODEL LEG below derives the
bank behaviour of each access from the same address functions, because a round trip
cannot see a conflict -- it only sees the answer.
"""
from __future__ import annotations

import pytest
import torch

from rola.ops import carry as carry_ops
from tests.k31.fixtures import assert_fresh_binary

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

#: (widths, k, m, d_v, W, declaration, modes) for every CARVED arm -- the bodies that
#: own a segment.  The two 1x8 container arms are the same cell as `flagship-ALT` under a
#: different EXECUTOR MAP, and the round trip is what says the column split's two warps
#: recover disjoint halves of one segment row.  THE LIST IS PARAMETERIZED OVER `d_v`
#: (K35 S3b): the wide plane changes `LdV` (72 -> 136) and the value row's quantum count
#: (8 -> 16), so both legs -- the device round trip and the bank model -- run at both
#: widths and neither derivation is allowed to be a coincidence of one of them.
D_V = carry_ops.D_V
WIDE = carry_ops.D_V_WIDE
ARMS = [
    pytest.param(((256, 256), 4, 8, D_V, carry_ops.WINDOW, carry_ops.LEAF_SPLIT, 6),
                 id="flagship-ALT"),
    pytest.param(((256, 256), 4, 8, D_V, carry_ops.WINDOW_WIDE, carry_ops.LEAF_SPLIT, 6),
                 id="flagship+-ALT"),
    pytest.param(((64, 64), 4, 8, D_V, carry_ops.WINDOW_SMALL, carry_ops.LEAF_SPLIT, 6),
                 id="small-ALT"),
    pytest.param(((8, 8, 8, 8), 2, 8, D_V, carry_ops.WINDOW_SMALL, carry_ops.LEAF_SPLIT, 102),
                 id="deep4-ALT"),
    pytest.param(((4, 4, 4, 4), 4, 1, D_V, carry_ops.WINDOW_SMALL, carry_ops.LEAF_SPLIT, 102),
                 id="carve4-ALT"),
    pytest.param(((256, 256), 4, 8, D_V, carry_ops.WINDOW, carry_ops.CONTAINER_1X8_SPLIT, 6),
                 id="1x8-wps2"),
    pytest.param(((256, 256), 4, 8, D_V, carry_ops.WINDOW, carry_ops.CONTAINER_1X8_STREAMS, 6),
                 id="1x8-streams8"),
    pytest.param(((256, 256), 4, 8, WIDE, carry_ops.WINDOW, carry_ops.CONTAINER_1X8_SPLIT, 6),
                 id="1x8-wps2-dv128"),
    pytest.param(((256, 256), 4, 8, WIDE, carry_ops.WINDOW, carry_ops.CONTAINER_1X8_STREAMS, 6),
                 id="1x8-streams8-dv128"),
    pytest.param(((256, 256), 4, carry_ops.CAP_BC64, WIDE, carry_ops.WINDOW,
                  carry_ops.LEAF_SPLIT, 6), id="2x4-bc64-dv128"),
]

LD_PAD = 8      # `kPadE`: the bf16 row padding both segment sub-tiles carry
SEG_ROWS = 16   # `kSegRows == CP == kMmaK`: the batch's union tokens
#: THE LSU'S GROUP FOR A SIXTEEN-BYTE ACCESS.  A 128-bit access is serviced eight lanes
#: at a time -- eight lanes x 16 B is exactly the 32 banks x 4 B the memory delivers in
#: one cycle -- so a conflict is a property of an EIGHT-LANE group and never of the whole
#: lockstep unit.  At `d_v = 64` a value row IS one such group (8 quanta); at 128 it is
#: two.  Modelling the whole warp would call the wide row "two-way" and be reading the
#: access WIDTH as a conflict.
WIDE_LANES = 8


@pytest.mark.parametrize("arm", ARMS)
def test_the_segment_round_trips_through_the_fragments(arm):
    """Everything the fetcher wrote is what the consumer's fragments recovered."""
    assert_fresh_binary()
    widths, k, m, d_v, W, declaration, modes = arm
    v, c, _, _, _ = carry_ops.layout_probe(widths, k, m, W, 32, declaration, modes, d_v=d_v)
    assert v.shape[-1] == d_v, "the value sub-tile is the arm's own width"
    for name, got in (("SegValueRows", v), ("SegCoeffRows", c)):
        streams, rows, cols = got.shape
        assert rows == SEG_ROWS, f"{name}: a segment slot is one contraction step"
        want = (torch.arange(rows, device=got.device).reshape(1, -1, 1) * 128
                + torch.arange(cols, device=got.device).reshape(1, 1, -1)).expand_as(got)
        bad = (got != want).nonzero()
        assert bad.numel() == 0, (
            f"{name}: {bad.shape[0]} of {got.numel()} cells came back from a different "
            f"(row, column) than they were stored at; first is {bad[0].tolist()} "
            f"(got {int(got[tuple(bad[0].tolist())])}, want {int(want[tuple(bad[0].tolist())])})")


@pytest.mark.parametrize("arm", ARMS)
def test_the_staged_factor_row_lands_run_by_run(arm):
    """Each level's owner run reaches its own offset, whole and in order.

    The staged row is written by asynchronous copies of 4, 8 or 16 bytes; a
    misalignment is a FAULT rather than a wrong answer, so the leg has to be executed
    and not modelled.  The pattern names the SOURCE column, so a run that landed at the
    wrong offset still says where it came from.
    """
    assert_fresh_binary()
    widths, k, m, d_v, W, declaration, modes = arm
    _, _, staged, _, _ = carry_ops.layout_probe(widths, k, m, W, 32, declaration, modes,
                                                d_v=d_v)
    _, s, _, _, _ = carry_ops.box_shape(widths, k, m)
    D, B = len(widths), widths[0]
    rows, stage_w = staged.shape
    assert rows == SEG_ROWS
    assert stage_w >= sum(s), "the staged row holds every level's owner run"
    end = -1
    for level in range(D):
        want = torch.tensor([level * B + i for i in range(int(s[level]))],
                            device=staged.device, dtype=staged.dtype)
        row0 = staged[0]
        hits = [o for o in range(stage_w - len(want) + 1)
                if bool(torch.equal(row0[o:o + len(want)], want))]
        assert hits, f"level {level}'s run ({want.tolist()}) is nowhere in the staged row"
        off = hits[0]
        assert off > end, "the levels' runs must not overlap"
        end = off + len(want) - 1
        for r in range(rows):
            assert torch.equal(staged[r, off:off + len(want)], want + r * 1024), (
                f"row {r}, level {level}: the staged run is not that row's token's")


# --------------------------------------------------------- the model leg

def _banks(words):
    """max-way over 32 four-byte banks for one access's word addresses."""
    seen = {}
    for w in words:
        seen.setdefault(w % 32, set()).add(w)
    return max(len(v) for v in seen.values())


@pytest.mark.parametrize("arm", ARMS)
def test_the_segment_access_patterns_are_derived_not_guessed(arm):
    """The bank behaviour of every leg, from the same address functions.

    VALUE ROWS, both legs, are CONFLICT-FREE at any placement: a row's eight sixteen-
    byte quanta cover all thirty-two banks exactly once, whatever word the row starts
    at, and `ldmatrix`'s eight row addresses per 8x8 tile do the same because `LdV = 72`
    puts consecutive rows four banks apart.

    COEFFICIENT ROWS: the consumer's leg is conflict-free for the same reason
    (`LdS = kLeafW + 8` is 20 words at `kLeafW = 32`, and 20*lr mod 32 is a permutation
    of the eight bank quads).  The fetcher's leg is TWO-WAY at worst and that is the
    FLOOR, not a defect: a `kLeafW/2`-word row occupies `kLeafW/2` banks BY DEFINITION,
    so two rows written at once either share their sixteen banks or do not, and the
    alternative -- one row per instruction with half the lanes idle -- costs the same
    two cycles unconditionally.  The pair mapping is therefore never worse and usually
    better; what the model pins is that it is never WORSE than two.
    """
    widths, k, m, d_v, W, declaration, modes = arm
    _, s, _, bc, _ = carry_ops.box_shape(widths, k, m)
    #: the arm's WRITE STREAM count and its stream's leaf count, from the census the
    #: device reports.  The segment is per STREAM, not per executor, so this is the
    #: census's stream field and never `threads // 32` -- under a column split the two
    #: differ by `WPS`.
    stamp = carry_ops.build_stamp(widths, k, m, W, 32, declaration, modes, d_v=d_v)
    assert stamp[15] == d_v, "the census must report the arm's own value width"
    streams = stamp[14]
    leaf = int(bc) // streams
    ld_v, ld_s = d_v + LD_PAD, leaf + LD_PAD

    def seg_word(stride, stream, row):
        return ((stream * SEG_ROWS + row) * stride * 2) // 4

    #: FETCHER, value: `d_v / 8` lanes to a row, one sixteen-byte quantum each, serviced
    #: in EIGHT-LANE groups.  A group's eight quanta cover all thirty-two banks exactly
    #: once whatever word the row starts at, so the leg is conflict-free at BOTH widths;
    #: the wide row is simply two groups rather than one.
    for stream in range(streams):
        for row in range(SEG_ROWS):
            base = seg_word(ld_v, stream, row)
            for grp in range(d_v // 8 // WIDE_LANES):
                words = [base + (grp * WIDE_LANES + c) * 4 + w
                         for c in range(WIDE_LANES) for w in range(4)]
                assert _banks(words) == 1, "a value quantum group must cover the banks once"
    #: CONSUMER, value: one 8x8 tile's eight row addresses, sixteen bytes each.
    for stream in range(streams):
        for tl in range(4):
            words = [seg_word(ld_v, stream, (tl >> 1) * 8 + lr) + (tl & 1) * 4 + w
                     for lr in range(8) for w in range(4)]
            assert _banks(words) == 1, "the value pull must be conflict-free"
    #: CONSUMER, coefficient: the same, at the coefficient stride.
    for stream in range(streams):
        for nj in range(leaf // 16):
            for tl in range(4):
                words = [seg_word(ld_s, stream, (tl & 1) * 8 + lr) + (tl >> 1) * 4 + nj * 8 + w
                         for lr in range(8) for w in range(4)]
                assert _banks(words) == 1, "the coefficient pull must be conflict-free"
    #: FETCHER, coefficient: two rows at once, `leaf/2` lanes each.
    worst = 1
    lanes_row = min(32, leaf // 2)
    if lanes_row * 2 == 32:
        for s0 in range(streams):
            for s1 in range(streams):
                for r0 in range(SEG_ROWS):
                    for r1 in range(SEG_ROWS):
                        words = ([seg_word(ld_s, s0, r0) + j for j in range(lanes_row)]
                                 + [seg_word(ld_s, s1, r1) + j for j in range(lanes_row)])
                        worst = max(worst, _banks(words))
    assert worst <= 2, f"the expansion's store is {worst}-way, above the two-row floor"


# ------------------------------------------ the EXCHANGE's segments, both changes

#: THE ARM'S TWO CARVES, mirroring the `CARRY_ARM_*` rows of `csrc/rola/src/carry/carry.cu`
#: -- the per-level digit span one STREAM owns on the write side and on the read side.
#: The kernel derives everything below from these two vectors and nothing else, so the
#: model needs them and nothing else either.
CARVES = {
    #: keyed by `(widths, k, m, d_v, declaration)` -- the arm's key less the window,
    #: which no carve depends on.
    ((256, 256), 4, 8, D_V, carry_ops.LEAF_SPLIT): ((2, 16), (8, 4), 1),
    ((64, 64), 4, 8, D_V, carry_ops.LEAF_SPLIT): ((2, 16), (8, 4), 1),
    ((8, 8, 8, 8), 2, 8, D_V, carry_ops.LEAF_SPLIT): ((2, 2, 1, 8), (2, 2, 4, 2), 1),
    ((4, 4, 4, 4), 4, 1, D_V, carry_ops.LEAF_SPLIT): ((1, 4, 2, 4), (4, 1, 4, 2), 1),
    ((256, 256), 4, 8, D_V, carry_ops.CONTAINER_1X8_SPLIT): ((2, 16), (8, 2), 2),
    ((256, 256), 4, 8, D_V, carry_ops.CONTAINER_1X8_STREAMS): ((1, 16), (8, 2), 1),
    ((256, 256), 4, 8, WIDE, carry_ops.CONTAINER_1X8_SPLIT): ((2, 16), (8, 2), 2),
    ((256, 256), 4, 8, WIDE, carry_ops.CONTAINER_1X8_STREAMS): ((1, 16), (8, 2), 1),
    ((256, 256), 4, carry_ops.CAP_BC64, WIDE, carry_ops.LEAF_SPLIT): ((1, 16), (4, 4), 1),
}


def _ilog2(x):
    return x.bit_length() - 1


class _Side:
    """`StreamSpans`: a side's stream grid and its `SpanMap` private order."""

    def __init__(self, owner_spans, spans):
        self.s = list(spans)
        self.g = [o // v for o, v in zip(owner_spans, spans)]
        self.carves = [v < o for o, v in zip(owner_spans, spans)]
        self.D = len(spans)
        self.streams = 1
        self.leaf = 1
        for x in self.g:
            self.streams *= x
        for x in self.s:
            self.leaf *= x
        self.grid_div = [_prod(self.g[l + 1:]) for l in range(self.D)]
        #: THE PRIVATE-ORDER RULE with NO LEVEL PINNED (K35 S4a): carved levels
        #: outermost in lattice order, whole-spanned levels inside them in lattice order.
        def before(a, b):
            if self.carves[a] != self.carves[b]:
                return not self.carves[a]
            return a > b
        self.rank = [sum(1 for i in range(self.D) if before(i, l)) for l in range(self.D)]
        bits = [_ilog2(v) for v in self.s]
        self.shift = [sum(bits[i] for i in range(self.D) if self.rank[i] < self.rank[l])
                      for l in range(self.D)]
        pair = [l for l in range(self.D) if self.shift[l] == 0 and self.s[l] >= 2]
        assert len(pair) == 1, f"{spans}: exactly one level lies lowest with a span above one"
        self.pair_level = pair[0]

    def leaf_of(self, slot, off, run_shift):
        """the LATTICE leaf of a private slot, given the stream's digit offsets."""
        return sum((((slot >> self.shift[l]) & (self.s[l] - 1)) + off[l]) << run_shift[l]
                   for l in range(self.D))

    def off(self, stream):
        return [((stream // self.grid_div[l]) % self.g[l]) * self.s[l] for l in range(self.D)]


def _prod(xs):
    out = 1
    for x in xs:
        out *= x
    return out


def _xch(owner_spans, sw, sr, wps):
    """`(rows_read, rows_write, leaves_read, leaves_write)` -- the two row maps."""
    D = len(owner_spans)
    run_shift = [sum(_ilog2(owner_spans[i]) for i in range(l + 1, D)) for l in range(D)]
    W, R = _Side(owner_spans, sw), _Side(owner_spans, sr)
    warps, leaf_e, leaf_w = R.streams, R.leaf, W.leaf
    assert W.streams * wps == warps and warps * leaf_e == _prod(owner_spans)

    def rows_read(warp):
        """a warp's `(write slot -> consumer row, lattice leaf)` under the WRITE carve."""
        wstrm, col = warp >> _ilog2(wps), warp & (wps - 1)
        off = W.off(wstrm)
        out = []
        for slot in range(leaf_e):
            ps = slot + col * leaf_e
            row = 0
            for l in range(D):
                dig = ((ps >> W.shift[l]) & (W.s[l] - 1)) + off[l]
                row += (dig // R.s[l]) * (R.grid_div[l] * leaf_e) + ((dig % R.s[l]) << R.shift[l])
            out.append((row, W.leaf_of(ps, off, run_shift)))
        return out

    def rows_write(warp):
        """a warp's `(read slot -> consumer row, lattice leaf)` under the READ carve."""
        off = R.off(warp)
        out = []
        for slot in range(leaf_e):
            row = 0
            for l in range(D):
                dig = ((slot >> R.shift[l]) & (R.s[l] - 1)) + off[l]
                row += (dig // W.s[l]) * (W.grid_div[l] * leaf_w) + ((dig % W.s[l]) << W.shift[l])
            out.append((row, R.leaf_of(slot, off, run_shift)))
        return out

    return W, R, rows_read, rows_write


def _swz(row):
    """``xch_swz``: the row's granule swizzle word."""
    return ((row >> 3) ^ (row >> 4) ^ (row >> 6)) & 7


def _mass_word(row):
    """``xch_mass``: the mass's element offset inside the row's ONE pad granule."""
    return 2 * (((row >> 3) ^ (row >> 5)) & 3)


def _elem(row, chan, d_v):
    return d_v + _mass_word(row) if chan == d_v else ((chan >> 3) ^ _swz(row)) * 8 + (chan & 7)


@pytest.mark.parametrize("arm", ARMS)
def test_the_exchange_relays_the_state_leaf_for_leaf(arm):
    """THE ROUND TRIP: every leaf reaches the consumer's row that holds that same leaf.

    The exchange is writer-side placement: a publisher owning a slot in ITS order names
    the CONSUMER's row directly, and the consumer then pulls rows it addresses as its OWN
    private slots.  The two derivations meet nowhere else, so what has to be proven is
    that the composite is the IDENTITY ON LATTICE LEAVES -- for both changes of a window,
    write order to read order and back.  A stride, an order or a stream-grid term that
    disagreed between the two sides would show up here as a leaf landing in the wrong row.
    """
    widths, k, m, d_v, W, declaration, modes = arm
    _, s, _, bc, _ = carry_ops.box_shape(widths, k, m)
    sw, sr, wps = CARVES[(tuple(widths), k, m, d_v, declaration)]
    owner = [int(x) for x in s]
    Wside, Rside, rows_read, rows_write = _xch(owner, sw, sr, wps)
    warps = Rside.streams
    for name, rows_of, consumer in (("fold->readout", rows_read, Rside),
                                    ("readout->fold", rows_write, Wside)):
        seen = {}
        for warp in range(warps):
            for row, leaf in rows_of(warp):
                assert row not in seen, f"{name}: two publishers claim row {row}"
                seen[row] = leaf
        assert len(seen) == int(bc), f"{name}: the publishers cover {len(seen)} of {bc} rows"
        #: and the consumer reads its OWN rows as its OWN private slots, so the row's
        #: leaf must be the leaf that consumer's slot names.
        leaf_e = consumer.leaf if consumer is Rside else Rside.leaf
        run_shift = [sum(_ilog2(owner[i]) for i in range(l + 1, len(owner)))
                     for l in range(len(owner))]
        for warp in range(warps):
            if consumer is Rside:
                off, slots = Rside.off(warp), range(leaf_e)
                want = [Rside.leaf_of(q, off, run_shift) for q in slots]
            else:
                wstrm, col = warp >> _ilog2(wps), warp & (wps - 1)
                off = Wside.off(wstrm)
                want = [Wside.leaf_of(p + col * leaf_e, off, run_shift) for p in range(leaf_e)]
            for q, leaf in enumerate(want):
                got = seen[warp * leaf_e + q]
                assert got == leaf, (
                    f"{name}: consumer {warp} slot {q} pulls leaf {got}, wants {leaf}")


@pytest.mark.parametrize("arm", ARMS)
def test_the_exchange_row_map_is_affine_in_the_slots_bits(arm):
    """``XchRow``: the kernel pays the map ONCE per warp, so it had better be affine.

    A slot's level-``l`` digit is ``offset_l | field_l`` (the stream's offset is a
    multiple of the span, so it sits above the field); the consumer's group and private
    digit are a BIT SPLIT of that digit; the group terms sum into the consumer index in
    mixed radix and the private terms into disjoint fields of the consumer's slot.  Every
    sum is an OR, so ``row(slot) = row(0) + sum over the slot's SET BITS of a per-bit
    STEP`` and every step is compile-time.  The kernel folds the base and the lane's slot
    pair into ONE per-warp value and adds a constant per fragment; this pins the property
    that makes that legal, and pins that the steps do not depend on WHICH stream publishes.
    """
    widths, k, m, d_v, W, declaration, modes = arm
    _, s, _, bc, _ = carry_ops.box_shape(widths, k, m)
    sw, sr, wps = CARVES[(tuple(widths), k, m, d_v, declaration)]
    owner = [int(x) for x in s]
    Wside, Rside, rows_read, rows_write = _xch(owner, sw, sr, wps)
    leaf_e = Rside.leaf
    for name, rows_of, slots in (("fold->readout", rows_read, Wside.leaf // wps),
                                 ("readout->fold", rows_write, Rside.leaf)):
        assert slots == leaf_e
        steps = None
        for warp in range(Rside.streams):
            table = [row for row, _ in rows_of(warp)]
            here = [table[1 << b] - table[0] for b in range(_ilog2(leaf_e))]
            if steps is None:
                steps = here
            assert here == steps, f"{name}: warp {warp}'s per-bit steps are not the arm's"
            for slot in range(leaf_e):
                want = table[0] + sum(steps[b] for b in range(_ilog2(leaf_e))
                                      if (slot >> b) & 1)
                assert table[slot] == want, (
                    f"{name}: warp {warp} slot {slot} is {table[slot]}, not the affine {want}")


@pytest.mark.parametrize("arm", ARMS)
def test_the_exchange_swizzle_is_a_bank_permutation(arm):
    """``xch_swz`` / ``xch_mass``, derived from BOTH access patterns.

    `LdX / 2` is 4 (mod 32) at both widths, so consecutive rows sit ONE four-bank block
    apart and a bank is ``(4*row + 4*granule + word) mod 32``.

    THE PULL is `ldmatrix.trans`, serviced eight lanes at a time: a phase is EIGHT
    CONSECUTIVE rows at ONE granule and already walks eight distinct blocks, so it asks
    only that the swizzle be CONSTANT over an aligned group of eight rows -- which
    ``(row>>3)``-only terms are.

    THE PUBLISH is `st.shared.b16`: one instruction covers the lane quad's FOUR
    consumer rows x EIGHT CONSECUTIVE channels, and two lanes sharing a 32-bit word are
    a BROADCAST and not a conflict, so it is four rows x four words inside ONE granule
    and it needs the four rows in four DIFFERENT blocks.  Those rows are the consumer-
    side images of four leaves and the carves scatter them, so the row index must reach
    the granule; every row bit gets a distinct nonzero image under ``xch_swz``.  The
    MASS is one element and must spread over its pad granule's four words on its own.
    """
    widths, k, m, d_v, W, declaration, modes = arm
    _, s, _, bc, _ = carry_ops.box_shape(widths, k, m)
    sw, sr, wps = CARVES[(tuple(widths), k, m, d_v, declaration)]
    owner = [int(x) for x in s]
    Wside, Rside, rows_read, rows_write = _xch(owner, sw, sr, wps)
    ld_x, n_g, mch = d_v + LD_PAD, d_v // 8, d_v // 16
    leaf_e = Rside.leaf
    assert (ld_x // 2) % 32 == 4, "a segment row must step one four-bank block"

    def words(rows, chans):
        return [(rows[lane & 3] * ld_x + _elem(rows[lane & 3], chans(lane), d_v)) // 2
                for lane in range(32)]

    #: THE PUBLISH, both changes.  The lane quad's four slots are four rows; the eight
    #: channel lanes are one granule.  `fold->readout` publishes the accumulator's
    #: `(channel, slot)`, `readout->fold` the A fragment's, and the two differ only in
    #: which of the lane's two channel halves rides which register.
    for name, rows_of in (("fold->readout", rows_read), ("readout->fold", rows_write)):
        for warp in range(Rside.streams):
            table = [row for row, _ in rows_of(warp)]
            for j in range(leaf_e // 8):
                for el in range(2):
                    rows = [table[j * 8 + 2 * a + el] for a in range(4)]
                    for mt in range(mch):
                        for h in range(2):
                            got = _banks(words(rows, lambda l, mt=mt, h=h:
                                               mt * 16 + (l >> 2) + 8 * h))
                            assert got == 1, f"{name}: the publish is {got}-way"
                    got = _banks(words(rows, lambda l: d_v)[:4])
                    assert got == 1, f"{name}: the mass publish is {got}-way"
    #: THE PULL, both changes: eight consecutive rows of the consumer's own segment at
    #: one granule, and the mass pair's four rows.
    for base in range(0, int(bc), 8):
        for g in range(n_g):
            wl = []
            for lr in range(8):
                row = base + lr
                w0 = (row * ld_x + ((g ^ _swz(row)) * 8)) // 2
                wl += [w0 + i for i in range(4)]
            assert _banks(wl) == 1, "the ldmatrix.trans phase must be conflict-free"
        for hf in range(2):
            rows = [base + 2 * a + hf for a in range(4)]
            assert _banks(words(rows, lambda l: d_v)) == 1, "the mass pull must be conflict-free"


# ------------------------------- the READOUT's segment and the fan-in ring (K35 S4b)

RING_C = 32     # `kRingC`: the ring's declared depth, in rows
RING_PAD = 8    # `LdRing - d_v`, in fp32 words


@pytest.mark.parametrize("arm", ARMS)
def test_the_reader_segment_round_trips_through_the_readout_fragments(arm):
    """Everything the READ expansion wrote is what the readout's fragments recovered.

    The readout's B operand is ``R~^T [leaves x tokens]``, so the segment's ROWS are the
    MMA's ``n`` axis and its columns are the contraction: the writer addresses a row by
    the token's RANK in that reader's list and the consumer's fragment map puts the row
    on ``n``.  The two derivations meet nowhere else.
    """
    assert_fresh_binary()
    widths, k, m, d_v, W, declaration, modes = arm
    *_, rows, _ = carry_ops.layout_probe(widths, k, m, W, 32, declaration, modes, d_v=d_v)
    readers, nrows, leaf = rows.shape
    assert nrows == SEG_ROWS, "a reader's segment slot is one batch of the union walk"
    want = (torch.arange(nrows, device=rows.device).reshape(1, -1, 1) * 128
            + torch.arange(leaf, device=rows.device).reshape(1, 1, -1)).expand_as(rows)
    bad = (rows != want).nonzero()
    assert bad.numel() == 0, (
        f"ReaderSegRows: {bad.shape[0]} of {rows.numel()} cells came back from a different "
        f"(row, slot) than they were stored at; first is {bad[0].tolist()}")


@pytest.mark.parametrize("arm", ARMS)
def test_the_ring_round_trips_from_the_scatter_to_the_flush(arm):
    """The scatter's cell address and the flush's are the same map.

    The scatter writes four ring rows at ONE eight-channel granule per issue (the
    accumulator's four lane quads are four tokens); the flush reads ONE row with
    thirty-two lanes over thirty-two consecutive channels.  The granule swizzle is a
    function of the row, so the two legs agree or the row comes back permuted.
    """
    assert_fresh_binary()
    widths, k, m, d_v, W, declaration, modes = arm
    *_, ring = carry_ops.layout_probe(widths, k, m, W, 32, declaration, modes, d_v=d_v)
    slots, cols = ring.shape
    assert slots == RING_C and cols == d_v + 1, "a ring row is the channels and the mass"
    want = (torch.arange(slots, device=ring.device).reshape(-1, 1) * 1024
            + torch.arange(cols, device=ring.device).reshape(1, -1)).expand_as(ring)
    bad = (ring != want).nonzero()
    assert bad.numel() == 0, (
        f"RingRows: {bad.shape[0]} of {ring.numel()} cells came back from a different "
        f"(slot, channel) than they were stored at; first is {bad[0].tolist()}")


@pytest.mark.parametrize("d_v", [D_V, WIDE], ids=["dv64", "dv128"])
def test_the_rings_access_patterns_are_derived_not_guessed(d_v):
    """The ring's bank behaviour, from the same address functions the kernel uses.

    THE FLUSH is conflict-free unconditionally: thirty-two lanes read four granules of
    eight consecutive words and the granule swizzle is a permutation inside the row.

    THE SCATTER is DATA-ADDRESSED and that is the whole point of modelling it: the four
    rows of one issue are four UNION RANKS, so no compile-time pattern names them.  With
    ``LdRing % 32 == 8`` a row's granule occupies a whole eight-bank block and the block
    is ``(row + granule) % 4``, so two of the four rows collide exactly when their ranks
    agree mod four -- and FOUR-WAY is the floor for ANY row permutation, four blocks
    being all there are.  What the swizzle buys is the DENSE case, which is the common
    one and the only one with a shape: a reader holding the whole batch scatters ranks
    ``r, r+2, r+4, r+6``, whose blocks are ``(r + 2q + swz) % 4``: two-way ALWAYS with
    no swizzle, and ``const + 3q`` -- a permutation of the four blocks -- under
    ``swz(row) = (row >> 1) & 3``.  Both facts are pinned here.
    """
    ld_ring = d_v + RING_PAD
    assert ld_ring % 32 == 8, "a ring row must step one eight-bank block"

    def cell(slot, granule, sub):
        return slot * ld_ring + ((granule ^ ((slot >> 1) & 3)) << 3) + sub

    #: THE FLUSH, one row: lanes 0..31 over 32 consecutive channels.
    for slot in range(RING_C):
        for c in range(d_v // 32):
            words = [cell(slot, c * 4 + (lane >> 3), lane & 7) for lane in range(32)]
            assert _banks(words) == 1, "the ring's flush must be conflict-free"
    #: THE SCATTER, the DENSE case: four ranks two apart, at every granule and every
    #: alignment of the batch inside the ring.
    for r0 in range(RING_C):
        for g in range(d_v // 8):
            words = [cell((r0 + 2 * q) % RING_C, g, sub) for q in range(4) for sub in range(8)]
            assert _banks(words) == 1, (
                f"the dense scatter is {_banks(words)}-way at rank {r0}, granule {g}")
    #: THE SCATTER, the general case: four INCREASING ranks of one batch.  Four-way is
    #: the floor and the model pins that nothing exceeds it.
    worst = 1
    for r0 in range(RING_C):
        for d1 in range(1, SEG_ROWS - 2):
            for d2 in range(d1 + 1, SEG_ROWS - 1):
                for d3 in range(d2 + 1, SEG_ROWS):
                    rows = [r0, r0 + d1, r0 + d2, r0 + d3]
                    words = [cell(x % RING_C, 0, sub) for x in rows for sub in range(8)]
                    worst = max(worst, _banks(words))
    assert worst <= 4, f"the scatter is {worst}-way, above the four-block floor"
    #: THE MASS rides the row's pad granule and spreads over its eight words by row.
    for r0 in range(RING_C):
        words = [(r0 + 2 * q) % RING_C * ld_ring + d_v + (((r0 + 2 * q) % RING_C) & 7)
                 for q in range(4)]
        assert _banks(words) <= 2, "the mass column must not be worse than two-way"


@pytest.mark.parametrize("d_v", [D_V, WIDE], ids=["dv64", "dv128"])
@pytest.mark.parametrize("leaves", [32, 16], ids=["leaf32", "leaf16"])
def test_the_parks_layout_is_the_segments_bytes_under_its_own_map(d_v, leaves):
    """``ParkedLo`` (K35 S5): a layout of its OWN over the exchange segment's bytes.

    The park is the one tenant that is NOT laid out the way the bytes it occupies are.
    Parked ``lo`` is not a leaf-by-channel matrix in either private order -- it is
    ``kParkWords`` OPAQUE packed-pair words per lane, written and read by ONE warp and by
    nothing else -- so the exchange's ``[leaf][channel]`` map and its row swizzle have no
    meaning for it.  What is derived here is the two facts that let it take those bytes:

    THE FIT IS AN IDENTITY.  A lane keeps ``2*MCH*NL + NL`` words, so a warp keeps
    ``32 * 4 * NL * (2*MCH + 1)`` bytes -- exactly ``kLeafE * LdX * 2``, its own segment,
    at every width and every box size.  Neither side of that has slack to hide an error.

    THE MAP IS CONFLICT-FREE BY CONSTRUCTION.  Word ``w`` of lane ``i`` sits at word
    ``32*w + i`` of the segment, so one ``st.shared.u32``/``ld.shared.u32`` covers 32
    CONSECUTIVE words: bank ``i`` for lane ``i``, MAX-WAY ONE, with no swizzle to derive.
    """
    mch, nl = d_v // 16, leaves // 8
    park_words = nl * (2 * mch + 1)
    ld_x = d_v + LD_PAD
    assert park_words * 32 * 4 == leaves * ld_x * 2, (
        "a warp's parked `lo` must be exactly its own exchange segment")
    for w in range(park_words):
        assert _banks([32 * w + lane for lane in range(32)]) == 1, (
            "a park burst must cover the banks exactly once")
