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
                     READER: the expansion's scalar ``ld.shared.u16``/``u32``.

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

#: (widths, k, m, W, declaration, modes) for every CARVED arm -- the bodies that own a
#: segment.  The two 1x8 container arms are the same cell as `flagship-ALT` under a
#: different EXECUTOR MAP, and the round trip is what says the column split's two warps
#: recover disjoint halves of one segment row.
ARMS = [
    pytest.param(((256, 256), 4, 8, carry_ops.WINDOW, carry_ops.LEAF_SPLIT, 6), id="flagship-ALT"),
    pytest.param(((256, 256), 4, 8, carry_ops.WINDOW_WIDE, carry_ops.LEAF_SPLIT, 6),
                 id="flagship+-ALT"),
    pytest.param(((64, 64), 4, 8, carry_ops.WINDOW_SMALL, carry_ops.LEAF_SPLIT, 6), id="small-ALT"),
    pytest.param(((8, 8, 8, 8), 2, 8, carry_ops.WINDOW_SMALL, carry_ops.LEAF_SPLIT, 102),
                 id="deep4-ALT"),
    pytest.param(((4, 4, 4, 4), 4, 1, carry_ops.WINDOW_SMALL, carry_ops.LEAF_SPLIT, 102),
                 id="carve4-ALT"),
    pytest.param(((256, 256), 4, 8, carry_ops.WINDOW, carry_ops.CONTAINER_1X8_SPLIT, 6),
                 id="1x8-wps2"),
    pytest.param(((256, 256), 4, 8, carry_ops.WINDOW, carry_ops.CONTAINER_1X8_STREAMS, 6),
                 id="1x8-streams8"),
]

LD_PAD = 8      # `kPadE`: the bf16 row padding both segment sub-tiles carry
SEG_ROWS = 16   # `kSegRows == CP == kMmaK`: the batch's union tokens


@pytest.mark.parametrize("arm", ARMS)
def test_the_segment_round_trips_through_the_fragments(arm):
    """Everything the fetcher wrote is what the consumer's fragments recovered."""
    assert_fresh_binary()
    widths, k, m, W, declaration, modes = arm
    v, c, _ = carry_ops.layout_probe(widths, k, m, W, 32, declaration, modes)
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
    widths, k, m, W, declaration, modes = arm
    _, _, staged = carry_ops.layout_probe(widths, k, m, W, 32, declaration, modes)
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
    widths, k, m, W, declaration, modes = arm
    _, s, _, bc, _ = carry_ops.box_shape(widths, k, m)
    #: the arm's WRITE STREAM count and its stream's leaf count, from the census the
    #: device reports.  The segment is per STREAM, not per executor, so this is the
    #: census's stream field and never `threads // 32` -- under a column split the two
    #: differ by `WPS`.
    stamp = carry_ops.build_stamp(widths, k, m, W, 32, declaration, modes)
    streams = stamp[14]
    leaf = int(bc) // streams
    ld_v, ld_s = 64 + LD_PAD, leaf + LD_PAD

    def seg_word(stride, stream, row):
        return ((stream * SEG_ROWS + row) * stride * 2) // 4

    #: FETCHER, value: eight lanes to a row, one sixteen-byte quantum each.
    for stream in range(streams):
        for row in range(SEG_ROWS):
            base = seg_word(ld_v, stream, row)
            words = [base + c * 4 + w for c in range(8) for w in range(4)]
            assert _banks(words) == 1, "a value row must cover the banks exactly once"
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
