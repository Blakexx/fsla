# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE FLOOR GATE (`bazel/floor.bzl`, docs/internals/tools/floor.md): `kernel_floor`'s arguments to warpline's bound action and test,
evaluated here as Bazel evaluates them (the macro file is Starlark that reads as Python once its `load` lines are dropped), over the
central cells' fields; and the gate's verdicts on its own test's kernel (`measure/floors/`), through warpline's floor and checkers: a
run inside the line passes, a run over it and a run under the floor are refused. No GPU: nothing is built."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fsla_devtools.cells import facts

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import dev_config  # noqa: E402

MACROS = ROOT / "bazel" / "floor.bzl"
WIRING = ROOT / "measure" / "floors"


class Failed(Exception):
    """Starlark's `fail`."""


def evaluate() -> tuple[dict, list[dict], list[dict]]:
    """The macro file's namespace and what `kernel_floor` declares of the test's kernel (`measure/floors/BUILD.bazel`): each call
    of warpline's bound and test, with its arguments."""
    bounds, tests = [], []

    def fail(message):
        raise Failed(message)

    #: Starlark's `type` names a value's type as a string
    namespace = {"CELLS": facts(), "fail": fail, "type": lambda v: type(v).__name__,
                 "warpline_floor_bound": lambda **kw: bounds.append(kw), "warpline_compare_test": lambda **kw: tests.append(kw)}
    source = "\n".join(line for line in MACROS.read_text().splitlines() if not line.startswith("load("))
    exec(compile(source, str(MACROS), "exec"), namespace)  # noqa: S102 -- the tree's own macro file
    build = "\n".join(line for line in (WIRING / "BUILD.bazel").read_text().splitlines() if not line.startswith("load("))
    exec(compile(build, str(WIRING / "BUILD.bazel"), "exec"),  # noqa: S102 -- the tree's own BUILD file
         {"kernel_floor": namespace["kernel_floor"], "test_suite": lambda **_: None, "tree_part": lambda **_: None})
    return namespace, bounds, tests


@pytest.fixture(scope="module")
def warpline():
    root = dev_config.warpline_python()
    if root is None:
        pytest.fail("warpline is handed in by the graph (bazel test //tests:cpu)")
    if root not in sys.path:
        sys.path.insert(0, root)
    from warpline import checkers, declare, floor, manifest
    return checkers, declare, floor, manifest


def test_the_gate_is_the_bound_and_the_test_of_one_kernel_on_one_cell():
    namespace, (bound,), (test,) = evaluate()
    cell = facts()["carry-alt-k4-16x16-T64"]
    assert bound["name"] == "wiring_bound" and bound["factor"] == "1.15" and bound["figure"] == "launch/grid_ns"
    assert bound["params"] == {k: str(v) for k, v in cell.items() if type(v) in (int, float)}
    assert bound["params"]["tokens"] == "64"
    assert (test["name"], test["declare"], test["manifests"]) == ("wiring_floor", ":wiring_bound", ["wiring.run.manifest.json"])
    with pytest.raises(Failed, match="no central cell"):
        namespace["cell_parameters"]("no-such-cell")


def test_a_run_over_the_line_or_under_the_floor_is_refused(warpline):
    checkers, declare, floor, manifest = warpline
    _namespace, (bound,), _tests = evaluate()
    work = floor.load_work((WIRING / "wiring.work.json").read_text())
    result = floor.floor(work, floor.load_rates((WIRING / "wiring.rates.json").read_text()),
                         {k: float(v) for k, v in bound["params"].items()})
    assert result.ns == pytest.approx(64 * 1048576 / 10)
    text = json.dumps(floor.declarations(result, work.kernel, float(bound["factor"]), bound["figure"]))

    def refusals(run: dict) -> list[str]:
        m = manifest.loads(json.dumps(run))
        out = declare.compare([m], declare.load_declarations(text, []), None, checkers.load(text))
        return [line for r in out.refusals for line in r.lines()]

    inside = json.loads((WIRING / "wiring.run.manifest.json").read_text())
    assert refusals(inside) == []
    over = json.loads((WIRING / "wiring.over.run.manifest.json").read_text())
    assert any("launch/grid_ns is 8000000, at most" in line for line in refusals(over))
    inside["places"][0]["figures"]["launch"]["grid_ns"] = 6000000
    assert any("at least" in line for line in refusals(inside))
