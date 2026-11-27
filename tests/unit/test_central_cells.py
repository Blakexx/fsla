# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""fsla's reading of the central cells (`measure/cells`): the window the central cells state their tail and flip
against is this kernel's, and every central layer cell's routing is an FSLA declaration the layer accepts. No GPU:
nothing here draws a cell."""
from __future__ import annotations

import pytest
import torch
from fsla_devtools.cells import build, central
from fsla_devtools.cells.carry import WINDOW

from fsla.ops.window import window_of
from measure.cells import carry_cells
from measure.cells.layer import routing


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the device's window is the binary's table at its card")
def test_the_device_window_is_a_whole_number_of_the_central_window():
    """The central cells state their tail and flip against a 512-token window; every card's window is a whole number
    of it (at W = 1,024 a flip cell's two halves share one kernel window)."""
    assert window_of() % WINDOW == 0


def test_every_layer_cells_routing_is_a_declaration_the_layer_accepts():
    """A layer cell states its routing in `fsla.Routing`'s terms; the declaration it names constructs, and lays out at
    the cell's group (the read heads the layer reads it by)."""
    layers = [c for c in map(build, central().cells.values()) if c.kind == "layer"]
    assert layers
    for cell in layers:
        declared = routing(cell)
        assert declared.topology.widths == cell.routing.topology, cell.name
        declared.layout(cell.group)


def test_the_oracle_tier_is_a_selection_of_the_central_carry_cells():
    oracle = carry_cells(tier="oracle")
    assert oracle and all(c.tier in ("oracle", "both") for c in oracle)
    assert set(oracle) <= set(carry_cells())
