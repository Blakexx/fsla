"""The runtime units listed in `csrc/fsla/units.bzl` are the ones `tools/kernels.py` derives from the source: the hand
units under `src/common/` that define no kernel. A unit that gains a kernel, or a new runtime unit, is a list to
change, and this test is what says so."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import kernels  # noqa: E402


def _listed() -> list[str]:
    for node in ast.parse((ROOT / "csrc" / "fsla" / "units.bzl").read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "RUNTIME_UNITS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("units.bzl lists no RUNTIME_UNITS")


def test_the_listed_runtime_units_are_the_derived_ones():
    derived = [p.relative_to(kernels.SRC.parent).as_posix() for p in kernels.runtime_units()]
    assert sorted(_listed()) == derived
