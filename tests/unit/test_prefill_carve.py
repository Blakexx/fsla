# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PREFILL LAUNCH FUNCTION'S CARVE (`csrc/fsla/src/common/plan/prefill_launch.h`, docs/internals/common/plan/prefill_launch.md): `plan_tool` prints a
prefill call's carve as JSON from its call facts, with no GPU. Each cell below is worked by the rules by hand; the tool's answer is the header's."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "csrc" / "fsla" / "plan_tool"

HOPPER = dict(cc=900, regs_per_sm=65536, smem_per_sm=233472, smem_per_cta_optin=232448, max_threads_per_sm=2048, sm_count=132)


def _carve(widths: list[int], write: list[int], read: list[int], tokens: int = 65536, **override):
    """The tool's run on a prefill call of the given levels: width bits, write flags and read flags a level, every other fact lawful."""
    levels = len(widths)
    facts = {"prefill": 1, "d_v": 64, "decay": 0, "tokens": tokens, "batch_heads": 1, "stateful": 0, "levels": levels}
    for k in range(1, 5):
        facts[f"width_bits_{k}"] = widths[k - 1] if k <= levels else 0
        facts[f"write_listed_{k}"] = write[k - 1] if k <= levels else 0
        facts[f"read_listed_{k}"] = read[k - 1] if k <= levels else 0
    facts.update(override)
    return subprocess.run([str(TOOL), "--device", ",".join(f"{k}={v}" for k, v in HOPPER.items()),
                           "--call", ",".join(f"{k}={v}" for k, v in facts.items()), "--carve", "json"],
                          capture_output=True, text=True, timeout=60)


def _out(run) -> dict:
    assert run.returncode == 0, run.stderr
    out = json.loads(run.stdout)
    if out["carved"]:
        # a fold is 256 leaves with a pair, 128 without
        assert out["fold_lists"] * out["fold_columns"] * out["fold_outer"] == (256 if out["pair_level"] >= 0 else 128)
    return out


KEYS = ("pair_level", "split_level", "fold_lists", "fold_columns", "boxes_per_head", "ctas_per_head", "units_per_head", "unit_ctas")


@pytest.mark.parametrize("write, read, box_bits, wl, rl, cell", [
    ([1, 0], [0, 1], [4, 4], 0, 1, (0, 0, 16, 16, 256, 128, 8, 16)),
    ([0, 1], [1, 0], [4, 4], 1, 0, (1, 1, 16, 16, 256, 128, 8, 16)),
    ([1, 0], [1, 0], [4, 4], 0, 0, (1, 0, 8, 32, 256, 128, 16, 8)),
    ([1, 1], [1, 1], [4, 4], 0, 0, (1, 0, 8, 32, 256, 128, 128, 1)),
    ([0, 0], [0, 0], [4, 4], 0, 0, (1, 1, 16, 16, 256, 128, 1, 128)),
])
def test_two_levels_of_256_digits_carve_as_the_designs_table(write, read, box_bits, wl, rl, cell):
    out = _out(_carve([8, 8], write, read))
    assert out["carved"] and out["box_bits"] == box_bits
    assert (out["write_list_level"], out["read_list_level"]) == (wl, rl)
    assert tuple(out[k] for k in KEYS) == cell
    assert out["window"] == 1024 and out["windows"] == 64


def test_the_calls_kernels_are_owed_and_the_carve_is_printed_all_the_same():
    out = _out(_carve([8, 8], [1, 0], [0, 1]))
    assert out["kernels"] == "owed" and out["owed"] == ["prefill"]


@pytest.mark.parametrize("write, cell", [
    ([1], (0, 0, 16, 16, 256, 128, 128, 1)),
    ([0], (0, 0, 16, 16, 256, 128, 1, 128)),
])
def test_one_level_of_65536_digits_is_256_boxes_paired_along_the_level(write, cell):
    out = _out(_carve([16], write, [0]))
    assert out["box_bits"] == [8] and out["read_list_bits"] == [4] and out["write_list_bits"] == [4]
    assert tuple(out[k] for k in KEYS) == cell


def test_one_level_of_256_digits_is_one_box_and_one_cta_with_no_pair():
    out = _out(_carve([8], [0], [0]))
    assert (out["pair_level"], out["boxes_per_cta"], out["boxes_per_head"], out["ctas_per_head"]) == (-1, 1, 1, 1)
    assert (out["fold_lists"], out["fold_columns"]) == (8, 16)


def test_one_level_of_128_digits_is_not_carved():
    assert _out(_carve([7], [0], [0])) == {"carved": False}


def test_two_levels_of_16_digits_are_one_box_one_cta_and_split_along_the_write_lists():
    out = _out(_carve([4, 4], [1, 0], [0, 0]))
    assert (out["boxes_per_head"], out["ctas_per_head"], out["pair_level"], out["split_level"]) == (1, 1, -1, 0)
    assert (out["fold_lists"], out["fold_columns"]) == (8, 16)


def test_the_pair_falls_to_the_read_list_level_when_its_column_level_has_one_box():
    # 16 x 64 digits, write 0,1 / read 0,1. A = level 1 (first written by lists), B = level 0: box bits 4,4. The read side lists level 1 only, so its list level
    # is 1 (64 wide: 4 boxes) and its column level 0 (16 wide: 1 box): the pair falls to level 1. The write list level is 1, walked by lists: split 1, fold 16 x 16.
    # Boxes 1 x 4 = 4, CTAs 2, units: level 1's 4 boxes halved = 2, unit CTAs 1.
    out = _out(_carve([4, 6], [0, 1], [0, 1]))
    assert (out["read_list_level"], out["pair_level"], out["split_level"]) == (1, 1, 1)
    assert tuple(out[k] for k in KEYS[2:]) == (16, 16, 4, 2, 2, 1)


def test_a_depth_3_cell_of_64_x_64_x_16():
    # write 0,1,0: A = 1 (first listed), B = 0 (first whole): box levels 0 and 1, box bits 4,4,0. Write: only level 1 listed, so its list level is 1, column 0.
    # Read 1,1,0: both box levels listed, so list level 0, column level 1 (listed): rule 1 is refused. Rule 2 FIRES: level 2 is outside the box, read whole, with
    # 2^(4-0) = 16 boxes: pair 2. The write side lists level 1, so the split is level 1, not the pair: fold 8 x 16, outer 2 (8 x 16 x 2 = 256 leaves).
    # Boxes 4 x 4 x 16 = 256, CTAs 128. Units: level 1's 4 boxes (the pair is not write-listed) = 4, unit CTAs 32.
    out = _out(_carve([6, 6, 4], [0, 1, 0], [1, 1, 0]))
    assert out["box_bits"] == [4, 4, 0] and (out["write_list_level"], out["read_list_level"]) == (1, 0)
    assert tuple(out[k] for k in KEYS) == (2, 1, 8, 16, 256, 128, 4, 32) and out["fold_outer"] == 2


def test_a_depth_4_cell_of_16_x_16_x_16_x_16():
    # write 0,0,1,1: A = 2, B = 0: box levels 0 and 2, box bits 4,0,4,0. Write lists level 2 only: its list level 2, column 0. Read 1,0,0,0: level 0 listed, level 2
    # whole: read list level 0, column level 2, which has 1 box (2^(4-4)): rule 1 is refused. Rule 2 FIRES: level 1 is outside the box, read whole, 16 boxes: pair 1.
    # Split: the write list level 2, walked by lists: fold 8 x 16, outer 2. Boxes 1 x 16 x 1 x 16 = 256, CTAs 128. Units: write-listed levels 2 (1 box) and 3 (16 boxes)
    # = 16, unit CTAs 8.
    out = _out(_carve([4, 4, 4, 4], [0, 0, 1, 1], [1, 0, 0, 0]))
    assert out["box_bits"] == [4, 0, 4, 0] and (out["write_list_level"], out["read_list_level"]) == (2, 0)
    assert tuple(out[k] for k in KEYS) == (1, 2, 8, 16, 256, 128, 16, 8) and out["fold_outer"] == 2


def test_the_pair_is_the_read_list_level_when_it_is_whole_and_its_column_level_has_one_box():
    # 256 x 16 digits, write 1,0 / read 0,0: box levels 0 and 1, box bits 4,4. Read both whole: list level 0 (256 wide: 16 boxes), column level 1 (16 wide: 1 box).
    # Rule 1 is refused (one box); rule 2 FIRES on the read list level, whole: pair 0, a box level, so fold_outer is 1. Write lists level 0, walked: split 0; the pair is
    # the write list level: 32 / 2 = 16 lists x 16 columns. Boxes 16, CTAs 8, units: level 0's 16 boxes halved = 8, unit CTAs 1.
    out = _out(_carve([8, 4], [1, 0], [0, 0]))
    assert tuple(out[k] for k in KEYS) == (0, 0, 16, 16, 16, 8, 8, 1) and out["fold_outer"] == 1


def test_the_pair_is_the_column_level_when_every_level_is_read_listed_and_the_column_has_boxes():
    # 256 x 256, write 0,0 / read 1,1: box bits 4,4. Read both listed: list level 0, column level 1. No level is read whole: rules 1 and 2 are refused. Rule 3 FIRES
    # on its first clause: the column level 1 has 16 boxes: pair 1. Write both whole: list level 0, not walked, so the split is the pair, 1: fold 16 x 16 (the column
    # level halves 32). Boxes 256, CTAs 128, units 1 (no write-listed level), unit CTAs 128.
    out = _out(_carve([8, 8], [0, 0], [1, 1]))
    assert tuple(out[k] for k in KEYS) == (1, 1, 16, 16, 256, 128, 1, 128) and out["fold_outer"] == 1


def test_the_pair_is_a_level_outside_the_box_when_every_level_is_read_listed_and_the_box_levels_have_one_box():
    # 16 x 16 x 64 digits, write 1,0,0 / read 1,1,1: box levels 0 and 1, box bits 4,4,0, one box each. Read all listed: list level 0, column level 1; no whole level
    # (rules 1 and 2 refused), the column level and the list level have one box (rule 3's first two clauses refused). Rule 3 FIRES on its last clause: the lowest other
    # level with 2 or more boxes, level 2 (2^(6-0) = 64 boxes): pair 2. Write lists level 0, walked: split 0, fold 8 x 16, outer 2. Boxes 64, CTAs 32, units:
    # level 0's 1 box = 1, unit CTAs 32.
    out = _out(_carve([4, 4, 6], [1, 0, 0], [1, 1, 1]))
    assert out["box_bits"] == [4, 4, 0]
    assert tuple(out[k] for k in KEYS) == (2, 0, 8, 16, 64, 32, 1, 32) and out["fold_outer"] == 2


def test_a_window_is_the_hopper_builds_and_a_call_is_cut_into_its_windows():
    build = json.loads((ROOT / "csrc" / "fsla" / "src" / "common" / "hardware.json").read_text())["arch"]["sm_90a"]["build"]
    for tokens, windows in ((1, 1), (1024, 1), (1025, 2), (65536, 64), (65537, 65)):
        out = _out(_carve([8, 8], [1, 0], [0, 1], tokens))
        assert out["window"] == build["window"] and out["windows"] == windows


@pytest.mark.parametrize("override", [dict(tokens=0), dict(levels=5), dict(write_listed_1=2)])
def test_a_fact_out_of_its_range_aborts(override):
    run = _carve([8, 8], [1, 0], [0, 1], **override)
    assert run.returncode != 0 and run.stdout == ""
