# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""A SIDE'S LEAF GROUPS (`csrc/fsla/src/common/plan/leaf_groups.h`, docs/internals/common/plan/leaf_groups.md): `plan_tool --cta-carve` prints, beside a call's carve, each
side's leaf groups -- their count, the fields of a leaf group's index, and the most the call's shape allows -- with no GPU. The hand-worked cases are worked in their comments;
the properties run over the calls of depth 1 to 4.

Run: ``bazel test //tests:cpu --test_filter=test_leaf_groups``."""
from __future__ import annotations

import itertools
import json
import random
import subprocess
import sys
from pathlib import Path

from conftest import one_read_head

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TOOL = ROOT / "csrc" / "fsla" / "plan_tool"
WHOLE = 256  # a side walks the level whole: it is one group
SIDES = (("write", "fold_bits", "fold_sets"), ("read", "carve_bits", "read_sets"))


def _run(calls: list[tuple[list[int], list[int], list[int]]]) -> list[dict]:
    """The tool's answers to calls of (width bits, write densities, read densities) a level, one process for each thousand of them (an argument has a length limit)."""
    if len(calls) > 1000:
        return [out for at in range(0, len(calls), 1000) for out in _run(calls[at:at + 1000])]
    arg = lambda pick: ";".join(",".join(map(str, pick(c))) for c in calls)  # noqa: E731
    run = subprocess.run([str(TOOL), "--cta-carve", arg(lambda c: c[0]), "--write-density", arg(lambda c: c[1]), "--read-density", arg(lambda c: c[2])],
                         capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr
    out = [one_read_head(json.loads(line)) for line in run.stdout.splitlines()]
    assert len(out) == len(calls)
    return out


def _one(widths: list[int], write: list[int], read: list[int]) -> dict:
    return _run([(widths, write, read)])[0]


def test_the_cell_has_64_leaf_groups_a_side_and_a_box_holds_4_of_them():
    # two levels of 256 digits; the write side on 4 digits of level 0 and whole on level 1, the read side the reverse.
    # A group is 4 digits of the sparse level by the whole level: 256 / 4 = 64 leaf groups a side. A fold holds 16 digits
    # of level 0, 4 groups, so 2 inner bits; the CTA holds 16 digits of level 1, 4 groups, so 2 inner bits for the read side
    out = _one([8, 8], [4, WHOLE], [WHOLE, 4])
    assert (out["write_leaf_bits"], out["write_leaf_inner_bits"]) == (6, 2)
    assert (out["read_leaf_bits"], out["read_leaf_inner_bits"]) == (6, 2)
    assert out["write_leaf_inner"] == [2, 0] and out["read_leaf_inner"] == [0, 2]
    # one level of each side has a mask: at most 256 leaf groups, the masked level at groups of one digit
    assert out["write_leaf_bits_most"] == 8 and out["read_leaf_bits_most"] == 8


def test_two_levels_at_4_digits_are_4096_leaf_groups_of_16_leaves_the_most_the_shape_allows():
    # both sides on 4 digits of both levels: groups of 4 digits a level, 64 x 64 leaf groups of 16 leaves; 65,536 / 16 is the most
    out = _one([8, 8], [4, 4], [4, 4])
    for side in ("write", "read"):
        assert out[f"{side}_leaf_bits"] == 12 and out[f"{side}_leaf_bits_most"] == 12
    # the side's densest level's inner field is the lowest: equal densities keep the lower level outside, so level 1's is at 0
    assert out["write_leaf_inner_at"] == [2, 0] and out["write_leaf_inner"] == [2, 2]


def test_a_call_with_no_mask_is_one_leaf_group_and_can_have_no_more():
    out = _one([8, 8], [WHOLE, WHOLE], [WHOLE, WHOLE])
    for side in ("write", "read"):
        assert out[f"{side}_leaf_bits"] == 0 and out[f"{side}_leaf_bits_most"] == 0


def _calls(depth: int, widths=(4, 6, 8)):
    for ws in itertools.product(widths, repeat=depth):
        for write in itertools.product((1, 5, 40, WHOLE), repeat=depth):
            for read in itertools.product((1, 5, WHOLE), repeat=depth):
                yield list(ws), list(write), list(read)


def _check(call, out) -> None:
    widths, _, _ = call
    for side, box, sets in SIDES:
        groups = out[f"{side}_group_bits"]
        inner, outer = out[f"{side}_leaf_inner"], out[f"{side}_leaf_outer"]
        inner_at, outer_at = out[f"{side}_leaf_inner_at"], out[f"{side}_leaf_outer_at"]
        bits = out[f"{side}_leaf_bits"]
        # a leaf group is one group a level
        assert bits == sum(w - g for w, g in zip(widths, groups, strict=True)), call
        assert [i + o for i, o in zip(inner, outer, strict=True)] == [w - g for w, g in zip(widths, groups, strict=True)], call
        # a level's inner bits are its groups inside a box, and a box's sets are exactly the inner fields
        assert inner == [max(0, b - g) for b, g in zip(out[box], groups, strict=True)], call
        assert 1 << out[f"{side}_leaf_inner_bits"] == out[sets], call
        # the fields tile the index: no two share a bit, and none is past the count
        taken = 0
        for at, width in list(zip(inner_at, inner, strict=True)) + list(zip(outer_at, outer, strict=True)):
            field = ((1 << width) - 1) << at
            assert taken & field == 0, call
            taken |= field
        assert taken == (1 << bits) - 1, call
        assert all(at + width <= out[f"{side}_leaf_inner_bits"] for at, width in zip(inner_at, inner, strict=True)), call
        assert out[f"{side}_leaf_one_to_one"] is True, call
        # whatever the densities, no more than the shape allows
        assert bits <= out[f"{side}_leaf_bits_most"], call


def test_the_properties_over_every_call_of_depth_1_and_2_and_a_fixed_sample_of_depth_3_and_4():
    calls = [c for depth in (1, 2) for c in _calls(depth)]
    rng = random.Random(20261125)
    calls += rng.sample(list(_calls(3)), 6000)
    calls += rng.sample(list(_calls(4)), 6000)
    for call, out in zip(calls, _run(calls), strict=True):
        _check(call, out)


def test_the_most_is_reached_and_a_level_with_no_mask_divides_it_by_its_width():
    # every level masked and sparse: the most is the leaves over 16
    for widths in ([8, 8], [8, 8, 4], [6, 6, 6, 6]):
        out = _one(widths, [1] * len(widths), [1] * len(widths))
        assert out["write_leaf_bits"] == out["write_leaf_bits_most"] == sum(widths) - 4, widths
    # one level whole: the other levels' digits, a group a digit
    out = _one([8, 8, 4], [1, WHOLE, 1], [1, 1, 1])
    assert out["write_leaf_bits"] == out["write_leaf_bits_most"] == 12
