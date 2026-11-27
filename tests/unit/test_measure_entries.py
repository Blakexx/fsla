# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""This tree's timing entries and diff sides (`measure/entries.bzl`), evaluated here over the central cells' fields as
Bazel evaluates them (the macro file is Starlark that reads as Python once its `load` lines are dropped): every subject
of the roster on every cell of its kind it takes, each a label of its arm and cell; each diff surface on the cells of its
kind and tier, the backward's under its retained-state budget. No GPU: nothing is built."""
from __future__ import annotations

from pathlib import Path

from fsla_devtools.cells import cell, facts

from measure.subjects import SUBJECTS
from tests.oracle import sides

MACROS = Path(__file__).resolve().parents[2] / "measure" / "entries.bzl"


def evaluate() -> tuple[dict, list[dict], list[dict]]:
    """The macro file's namespace, and what its two macros declare (each call's arguments)."""
    entries, diff_sides = [], []
    namespace = {"CELLS": facts(), "timing_entry": lambda **kw: entries.append(kw),
                 "diff_side": lambda **kw: diff_sides.append(kw)}
    source = "\n".join(line for line in MACROS.read_text().splitlines() if not line.startswith("load("))
    exec(compile(source, str(MACROS), "exec"), namespace)  # noqa: S102 -- the tree's own macro file
    namespace["timing_entries"]()
    namespace["diff_sides"]()
    return namespace, entries, diff_sides


def test_the_subjects_table_is_the_roster():
    namespace, _entries, _sides = evaluate()
    assert {name: kind for name, (kind, _decodes) in namespace["SUBJECTS"].items()} == \
        {name: s.kind for name, s in SUBJECTS.items()}


def test_every_subject_is_one_entry_on_every_cell_of_its_kind_it_takes():
    _namespace, entries, _sides = evaluate()
    declared = {(e["params"]["arm"], e["cells"][0].removeprefix("@fsla_cells//:")) for e in entries}
    for name, subject in SUBJECTS.items():
        wanted = {c for c, f in facts().items() if f["kind"] == subject.kind
                  and (name != "decode_step" or f["decode_steps"] > 0)}
        assert {c for arm, c in declared if arm == name} == wanted, name
    assert all(e["name"] == f"{e['params']['arm']}_{e['cells'][0].removeprefix('@fsla_cells//:')}" for e in entries)
    assert all(e["executor"] == "measure.provider:timed" for e in entries)


def test_each_surface_is_a_side_of_an_existing_executor_and_the_backward_admits_only_its_budget():
    namespace, _entries, declared = evaluate()
    by = {s["name"]: s for s in declared}
    assert sorted(by) == sorted(f"side_{surface}" for surface in namespace["SURFACES"])
    for surface, (executor, _kind, _tier, device) in namespace["SURFACES"].items():
        module, _, function = executor.partition(":")
        assert module == "tests.oracle.sides" and callable(getattr(sides, function)), executor
        assert by[f"side_{surface}"]["device"] is device
    oracle = {c.removeprefix("@fsla_cells//:") for c in by["side_oracle"]["cells"]}
    tier = {c for c, f in facts().items() if f["kind"] == "carry" and f["tier"] == "oracle"}
    retained = {c: 2 * cell(c).tokens * cell(c).routing.N * (cell(c).dv + 1) * 8 for c in tier}
    assert oracle == {c for c in tier if retained[c] < namespace["RETAINED_STATE_BUDGET"]}
    assert oracle
