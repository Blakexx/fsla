# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The hopper carry's floor model (tools/carry_floor.py): its counts are the expectations they claim, and a cluster
moves the terms it is said to move. No GPU: the rates are read from the H100 labs' committed records."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import carry_floor  # noqa: E402

ALT = dict(widths=(256, 256), read_sparse=(0, 1), write_sparse=(1, 0), k=4, box=(16, 16), dv=64, tokens=65536)


def test_a_sides_counts_are_the_expected_ones():
    #: 4 uniform digits of 16 ranges fall in 16 (1 - (15/16)^4) of them
    ranges = 16 * (1 - (15 / 16) ** 4)
    read = carry_floor.side((256, 256), (0, 1), 4, (16, 16), (16, 1))
    assert read.leaves == 1024
    assert read.boxes == pytest.approx(16 * ranges)
    assert read.clusters == pytest.approx(ranges)
    dense = carry_floor.side((256, 256), (0, 0), 4, (16, 16), (4, 4))
    assert (dense.leaves, dense.boxes, dense.clusters, dense.lists) == (65536, 256, 16, 16)


def test_a_clusters_sum_moves_the_drains_and_nothing_else():
    none = carry_floor.floor(cluster=(1, 1), **ALT)["terms"]
    rows = carry_floor.floor(cluster=(16, 1), form="slots", **ALT)["terms"]
    columns = carry_floor.floor(cluster=(1, 16), form="slots", **ALT)["terms"]
    for name in ("tensor, readout", "tensor, fold", "readout operand, expanded"):
        assert none[name] == rows[name] == columns[name]
    drains, loads = "sum: drains to global memory", "fold operands"
    assert rows[drains]["cycles"][0] < columns[drains]["cycles"][0] == pytest.approx(none[drains]["cycles"][0] * 16 / 58.25, rel=0.01)
    #: a cluster shares no load
    assert columns[loads] == rows[loads] == none[loads]
    assert none["sum: partials into the cluster"]["cycles"] == [0.0, 0.0]


def test_the_cluster_forms_read_the_labs_line():
    hop, rounds = "sum: partials into the cluster", "sum: the rounds' barriers"
    slots = carry_floor.floor(cluster=(16, 1), form="slots", **ALT)["terms"]
    s64 = carry_floor.floor(cluster=(16, 1), form="s64", **ALT)["terms"]
    longer = carry_floor.floor(cluster=(16, 1), form="slots", round_partials=512, **ALT)["terms"]
    assert slots[hop]["rate"] == "hop_slots" and s64[hop]["cycles"][0] > slots[hop]["cycles"][0] > 0.0
    assert longer[hop] == slots[hop]
    assert longer[rounds]["cycles"][0] == pytest.approx(slots[rounds]["cycles"][0] / 8)


def test_a_cluster_with_no_sum_is_refused():
    with pytest.raises(SystemExit, match="sums nothing"):
        carry_floor.floor(cluster=(1, 16), **ALT)


def test_a_missing_record_row_is_refused():
    with pytest.raises(SystemExit, match="has no row"):
        carry_floor.median("wgmma_rate", "a row no record holds")


def test_a_box_every_token_writes_pays_its_steps_and_the_whole_tile():
    even = carry_floor.floor(cluster=(1, 1), **ALT)["terms"]
    strip = carry_floor.floor(cluster=(1, 1), **{**ALT, "box": (256, 1)})["terms"]
    #: the readout's leaves of a list fill whole steps of sixteen in both shapes
    assert even["tensor, readout"]["cycles"] == strip["tensor, readout"]["cycles"]
    #: the fold is its steps' time, a sixteenth of one a (token, list, box): 64 of those square, 1,024 in strips
    rates = carry_floor.measured()
    assert strip["tensor, fold"]["cycles"][0] == pytest.approx(256 * 4 * rates["step_8"][0] / 16)
    assert even["tensor, fold"]["cycles"][0] == pytest.approx(1024 / 16 * rates["step_16"][0] / 16, rel=0.01)
    #: square boxes list their entries: 64 a token, each at the measured entry's cost
    assert even["fold operands"]["rate"].startswith("LISTED")
    assert even["fold operands"]["cycles"][0] == pytest.approx(1024 / 16 * rates["entry_low"][0], rel=0.01)
    #: every token writes every strip, 4 lists each: 1,024 entries a token
    assert strip["fold operands"]["cycles"][0] == pytest.approx(
        min(1024 * rates["entry_low"][0], 256 * rates["load_packed"][0] + 1024 * 2 / rates["shared"][0]
            + 1024 * 64 * 4 / rates["shared"][0]))


def test_a_pair_drains_one_line_for_two_boxes_and_stores_its_image_a_window():
    rates = carry_floor.measured()
    alone = carry_floor.floor(cluster=(1, 1), **ALT)["terms"]
    pair = carry_floor.floor(cluster=(1, 1), pair=True, **ALT)["terms"]
    assert pair["tensor, readout"]["cycles"] == pair["sum: drains to global memory"]["cycles"] == [0.0, 0.0]
    assert pair["readout and sum: a pair's f16 line"]["cycles"][0] == pytest.approx(58.25 / 2 * rates["line_low"][0], rel=0.01)
    #: the carve is divergent: 128 CTAs store their image once in 1,024 tokens
    assert pair["state image, a window"]["cycles"][0] == pytest.approx(128 * rates["image"][0] / 1024)
    assert pair["fold operands"] == alone["fold operands"] and pair["tensor, fold"] == alone["tensor, fold"]
    #: sparse on the same level, there is no image
    same = carry_floor.floor(cluster=(1, 1), pair=True, **{**ALT, "write_sparse": (0, 1), "box": (16, 16)})["terms"]
    assert same["state image, a window"]["cycles"] == [0.0, 0.0]
    #: a box the whole read-dense level has no neighbour to pair with
    with pytest.raises(SystemExit, match="a pair is two boxes"):
        carry_floor.floor(cluster=(1, 1), pair=True, **{**ALT, "box": (256, 1)})
