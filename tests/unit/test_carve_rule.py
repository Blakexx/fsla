# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The carve rule: producer v1's boxes for the layouts its rules were chosen on, each side's warp tile and publish
route, the pinned plan's check, and the refusals (docs/internals/engine/rules.md#carve)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rola.engine import Carve, PlanOverrides
from rola.engine.carve import check_carve, plan_carve
from rola.engine.rules.carve import carve as carve_rule


def _flags(level_tags):
    return tuple(t[0] == "S" for t in level_tags), tuple(t[1] == "S" for t in level_tags)


@pytest.mark.parametrize("widths, level_tags, extents, channels", [
    ((256, 256), ("SD", "DS"), (32, 32), 32),
    ((32, 32, 32, 32), ("DS", "SD", "DS", "SD"), (32, 32, 1, 1), 32),
    ((32, 32, 32, 32), ("SS", "SS", "DD", "DD"), (1, 1, 32, 32), 32),
    ((32, 32, 32, 32), ("SD", "DS", "DD", "SS"), (8, 4, 32, 1), 32),
    ((256, 256), ("SS", "DD"), (32, 32), 32),
    ((64, 64, 256), ("SD", "DS", "DD"), (8, 4, 32), 32),
    ((16, 256), ("SD", "DS"), (16, 32), 64),
    ((256,), ("DD",), (256,), 64),
])
def test_producer_v1_boxes(widths, level_tags, extents, channels):
    read, write = _flags(level_tags)
    c = plan_carve(widths, read, write, 64)
    assert (c.extents, c.channels) == (extents, channels)
    assert len(widths) == 1 or max(c.extents) <= 32
    check_carve(c, widths, read, write, 64)


def test_even_box_tiles_and_route():
    c = plan_carve((256, 256), (False, True), (True, False), 64)
    assert (c.fold.lists, c.fold.operand, c.fold.tile_lists, c.fold.tile_rows) == ((0,), (1,), 4, 32)
    assert (c.readout.lists, c.readout.operand, c.readout.tile_lists, c.readout.tile_rows) == ((1,), (0,), 4, 32)
    assert not c.publish_in_registers
    assert c.ctas(64) == 8 * 8 * 2


def test_dense_dense_box_publishes_in_registers():
    c = plan_carve((32, 32, 32, 32), (True, True, False, False), (True, True, False, False), 64)
    assert (c.fold.tile_lists, c.fold.tile_rows) == (1, 128)
    assert (c.readout.tile_lists, c.readout.tile_rows) == (1, 128)
    assert c.publish_in_registers


def test_refusals():
    with pytest.raises(ValueError, match="every level sparse"):
        plan_carve((256, 256), (True, True), (True, False), 64)
    with pytest.raises(ValueError, match="power of two"):
        plan_carve((256, 200), (False, True), (True, False), 64)
    good = plan_carve((256, 256), (False, True), (True, False), 64)
    with pytest.raises(ValueError, match="plan"):
        check_carve(Carve(good.widths, good.extents, good.channels, good.readout, good.fold, False),
                    (256, 256), (False, True), (True, False), 64)
    with pytest.raises(ValueError, match="divid"):
        check_carve(Carve(good.widths, (32, 48), 32, good.fold, good.readout, False),
                    (256, 256), (False, True), (True, False), 64)


def test_rule_reads_shapes_and_takes_a_checked_pin():
    routes = SimpleNamespace(amp_read=(torch.empty(1, 1, 1, 256), torch.empty(1, 1, 1, 256)),
                             sparse=((False, True), (True, False)))
    assert carve_rule(routes, 64, None) == plan_carve((256, 256), (False, True), (True, False), 64)
    pinned = plan_carve((256, 256), (False, True), (True, False), 64)
    assert carve_rule(routes, 64, PlanOverrides(carve=pinned)) is pinned
    bad = Carve(pinned.widths, pinned.extents, pinned.channels, pinned.fold, pinned.readout, True)
    with pytest.raises(ValueError, match="publish"):
        carve_rule(routes, 64, PlanOverrides(carve=bad))
