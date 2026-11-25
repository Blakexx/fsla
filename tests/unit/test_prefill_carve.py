# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PREFILL'S CARVE (`csrc/fsla/src/common/plan/cta_carve.h`, docs/internals/common/plan/cta_carve.md): `plan_tool --cta-carve` prints the carve of a call's levels from their width bits and each side's
densities alone, with no descriptor and no device. Each case below is worked by the rules by hand; the tool's answer is the case's. The carve sizes alone, over every density pattern, are `tests/unit/test_cta_carve.py`'s; the
launch's host half is `tests/unit/test_prefill_launch.py`'s."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TOOL = ROOT / "csrc" / "fsla" / "plan_tool"


def _carve(widths: list[int], write: list[int], read: list[int]):
    """The tool's run on levels of the given width bits and each side's densities a level."""
    join = lambda v: ",".join(map(str, v))  # noqa: E731
    return subprocess.run([str(TOOL), "--cta-carve", join(widths), "--write-density", join(write), "--read-density", join(read)],
                          capture_output=True, text=True, timeout=60)


def _out(run) -> dict:
    assert run.returncode == 0, run.stderr
    out = json.loads(run.stdout)
    assert out["fold_leaves"] * out["folds"] == out["cta_leaves"]
    assert out["fold_sets"] * out["fold_set_leaves"] == out["fold_leaves"] and out["read_sets"] * out["read_set_leaves"] == out["cta_leaves"]
    assert out["units_per_head"] * out["unit_ctas"] == out["ctas_per_head"]
    return out


def _fields(out: dict) -> dict:
    return {"carve": [1 << b for b in out["carve_bits"]], "write_groups": [1 << b for b in out["write_group_bits"]], "read_groups": [1 << b for b in out["read_group_bits"]],
            "split_level": out["split_level"], "fold": [1 << b for b in out["fold_bits"]],
            "fold_set_leaves": out["fold_set_leaves"], "fold_sets": out["fold_sets"], "read_set_leaves": out["read_set_leaves"], "read_sets": out["read_sets"],
            "ctas_per_head": out["ctas_per_head"], "unit_ctas": out["unit_ctas"], "units_per_head": out["units_per_head"]}


def _case(widths, write, read, carve, write_groups, read_groups, split, fold, fold_sets, read_sets, counts):
    """`fold_sets` and `read_sets` are (leaves of a set, sets); `counts` is (CTAs a head, unit CTAs, units a head)."""
    return (widths, write, read, dict(carve=carve, write_groups=write_groups, read_groups=read_groups, split_level=split, fold=fold,
                                      fold_set_leaves=fold_sets[0], fold_sets=fold_sets[1], read_set_leaves=read_sets[0], read_sets=read_sets[1],
                                      ctas_per_head=counts[0], unit_ctas=counts[1], units_per_head=counts[2]))


# Each case is worked by hand in the page (docs/internals/common/plan/cta_carve.md). A level is write density / read density; widths in bits.
CASES = [
    _case([8, 8], [5, 256], [256, 1], [32, 16], [4, 256], [256, 1], 0, [16, 16], (64, 4), (32, 16), (128, 16, 8)),
    _case([8, 8], [1, 256], [256, 1], [32, 16], [1, 256], [256, 1], 0, [16, 16], (16, 16), (32, 16), (128, 16, 8)),
    _case([8, 8], [5, 256], [256, 5], [32, 16], [4, 256], [256, 4], 0, [16, 16], (64, 4), (128, 4), (128, 16, 8)),
    _case([8, 8], [256, 1], [1, 256], [16, 32], [256, 1], [1, 256], 1, [16, 16], (16, 16), (32, 16), (128, 16, 8)),
    _case([8, 8], [256, 256], [256, 256], [32, 16], [256, 256], [256, 256], 0, [16, 16], (256, 1), (512, 1), (128, 128, 1)),
    _case([8, 8, 8], [5, 256, 256], [256, 1, 256], [4, 4, 32], [4, 256, 256], [256, 1, 256], 0, [2, 4, 32], (256, 1), (128, 4), (32768, 512, 64)),
    _case([8, 8, 8, 8], [5, 256, 5, 256], [256, 1, 256, 1], [8, 4, 4, 4], [4, 256, 4, 256], [256, 1, 256, 1], 0, [4, 4, 4, 4], (256, 1), (32, 16),
          (8388608, 4096, 2048)),
    _case([8, 8], [128, 256], [256, 1], [32, 16], [128, 256], [256, 1], 0, [16, 16], (256, 1), (32, 16), (128, 64, 2)),
    #: one level: the write side's size is 4 digits and a set of 4 leaves is under 16, so its group is raised to 16
    _case([8], [5], [256], [256], [16], [256], 0, [128], (16, 8), (256, 1), (1, 1, 1)),
    _case([4, 4], [64, 256], [256, 16], [16, 16], [4, 16], [16, 1], 0, [8, 16], (64, 2), (16, 16), (1, 1, 1)),
    #: one level of 16 digits: the fold holds 8, so the group is raised to the fold's 8 and no further
    _case([4], [64], [256], [16], [8], [16], 0, [8], (8, 1), (16, 1), (1, 1, 1)),
]


@pytest.mark.parametrize("widths, write, read, want", CASES, ids=[f"case{n}" for n in range(len(CASES))])
def test_the_carve_of_the_owners_cases(widths, write, read, want):
    # The groups are each side's allowed size nearest to density x width / 256 digits (a tie to the smaller); the apportionment gives each level at least one digit and
    # shares the nine doublings by weight (the sum of its two densities over its holdings plus one), a tie to the larger read density, then the lower level; the folds split along the first level not written
    # whole that holds two digits; a set is the product of the side's group clipped to the fold (write) or the CTA (read); the counts follow from the carve and the write groups.
    got = _fields(_out(_carve(widths, write, read)))
    assert got == want


@pytest.mark.parametrize("density, group", [(1, 1), (2, 2), (3, 2), (4, 4), (6, 4), (7, 8), (12, 8), (13, 16), (24, 16), (25, 32), (48, 32), (49, 64), (96, 64), (97, 128), (192, 128), (193, 256), (256, 256)])
def test_a_density_is_the_nearest_allowed_group_on_a_level_of_256_digits_a_tie_to_the_smaller(density, group):
    out = _out(_carve([8], [density], [density]))
    assert out["write_size_bits"] == out["read_size_bits"] == [group.bit_length() - 1]


def test_a_set_under_16_leaves_raises_the_group_of_the_densest_level_with_room_in_the_fold():
    #: widths 256 and 256; the write side 5 and 41 of 256 (sizes 4 and 32 digits), the read side whole and 5. The carve is 256 x 2 and the folds
    #: split level 0, a fold 128 x 2: a set of 4 x 2 = 8 leaves. Level 1 is the denser and has no room (its group is past the fold's 2 digits), so level 0's
    #: group is doubled once: 8 x 2 = 16 leaves, 16 sets a fold
    out = _out(_carve([8, 8], [5, 41], [256, 5]))
    assert out["carve_bits"] == [8, 1] and out["fold_bits"] == [7, 1]
    assert out["write_size_bits"] == [2, 5] and out["write_group_bits"] == [3, 5]
    assert (out["fold_set_leaves"], out["fold_sets"]) == (16, 16)
    #: the read side's set is the whole of level 0 by one digit of level 1, 256 leaves: nothing is raised
    assert out["read_size_bits"] == out["read_group_bits"] == [8, 2]


def test_one_digit_groups_on_both_levels_are_raised_to_16_leaves_the_denser_level_first():
    #: both sides one live digit of 256 on both levels: a tie in density keeps the lower level first in the order, so level 1 is the densest
    out = _out(_carve([8, 8], [1, 1], [1, 1]))
    assert out["write_size_bits"] == out["read_size_bits"] == [0, 0]
    assert out["fold_set_leaves"] == 16 and out["read_set_leaves"] == 16
    assert out["write_group_bits"][1] >= out["write_group_bits"][0] and sum(out["write_group_bits"]) == 4


def test_a_group_is_never_larger_than_its_level():
    #: a level of 2 digits: at half density its group is one digit, at the whole it is the level's two and no more
    out = _out(_carve([4, 1], [256, 256], [255, 128]))
    assert out["write_group_bits"] == [4, 1] and out["read_group_bits"] == [4, 0]


def test_one_level_of_256_digits_is_a_cta_of_256_leaves_and_two_folds_of_128():
    out = _out(_carve([8], [256], [256]))
    assert out["carve_bits"] == [8] and out["cta_leaves"] == 256 and out["ctas_per_head"] == 1
    # the write group is the whole level, so no level qualifies first: the split is the first level holding 2 digits, level 0
    assert (out["split_level"], out["folds"], out["fold_bits"], out["fold_leaves"]) == (0, 2, [7], 128)
    assert (out["fold_set_leaves"], out["fold_sets"], out["read_set_leaves"], out["read_sets"]) == (128, 1, 256, 1)


def test_a_cta_of_one_leaf_has_one_fold_and_no_split_level():
    out = _out(_carve([0], [1], [1]))
    assert (out["cta_leaves"], out["split_level"], out["folds"], out["fold_leaves"]) == (1, -1, 1, 1)


@pytest.mark.parametrize("densities, clamped", [([0, 0], [1, 1]), ([300, 300], [256, 256]), ([0, 300], [1, 256]), ([-7, 1000], [1, 256])])
def test_the_carve_is_total_a_density_outside_1_to_256_is_clamped_into_it(densities, clamped):
    # case 0's levels with a density of 0 or 300 give the result of 1 and 256: the carve has no failure path
    for side in range(2):
        write, read = ([*densities], [256, 1]) if side == 0 else ([5, 256], [*densities])
        want = ([*clamped], [256, 1]) if side == 0 else ([5, 256], [*clamped])
        got, ref = _out(_carve([8, 8], write, read)), _out(_carve([8, 8], *want))
        assert got == ref
        assert got["write_density"] == want[0] and got["read_density"] == want[1]
