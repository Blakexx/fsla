# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""FSLA'S READING OF THE CENTRAL CELLS: the cells are fsla-devtools' (`fsla_devtools.cells`): every package's tests and
measurements read one set of inputs with one draw, every cell in one form (`fsla_devtools.cells.form.Cell`) and named
by its own fields, and a cell names nothing a package runs it with.

This module is fsla's side of that line: a central cell's record by name (`by_name`), the carry cells and their tiers
(`carry_cells`), and a carry cell's draw (`realize`). What the carry kernel is given for a cell is built where it is
timed (`measure.subjects.bundle_of`). `layer.py` is the other half: the FSLA layer a layer cell's routing and gain
build, and the fixture its arms read.
"""
from __future__ import annotations

from fsla_devtools.cells import build, central
from fsla_devtools.cells.carry import realize
from fsla_devtools.cells.form import Cell


def by_name(name: str) -> Cell:
    """A central cell's record, of any kind."""
    return build(central().cell(name))


def carry_cells(*, tier: str | None = None) -> tuple[Cell, ...]:  #: reserved: the oracle tests' tier
    """Every central carry cell; ``tier`` keeps the cells sized for that reader (`oracle`, `probe`; `both` is either).
    ``tier`` is reserved: the oracle tier's tests select by it."""
    cells = tuple(c for c in map(build, central().cells.values()) if c.kind == "carry")
    return cells if tier is None else tuple(c for c in cells if c.tier in (tier, "both"))


__all__ = ["Cell", "by_name", "carry_cells", "realize"]
