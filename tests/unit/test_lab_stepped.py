# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""KERNEL_STANDARDS #stepped's mechanism (tools/lab.py's `untraced`): an experiment of a COMPOSED kernel -- one that
compiles a library header whose kernel has a budget file, or that defines more than one role -- records nothing without a
traced row; a traced one records, and a kernel of one role and no budget is not held to it. A FIXTURE: a synthetic
library and budget tree, no device."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import lab  # noqa: E402


@pytest.fixture
def tree(tmp_path, monkeypatch):
    library, budgets = tmp_path / "src", tmp_path / "budgets"
    (library / "k").mkdir(parents=True)
    budgets.mkdir()
    (library / "k" / "budgeted.cuh").write_text("__device__ void body() {}\n")
    (budgets / "budgeted.json").write_text("{}\n")
    (library / "k" / "roles.cuh").write_text("__device__ void load_role() {}\n__device__ void math_role() {}\n")
    (library / "k" / "plain.cuh").write_text("__device__ void load_role() {}\n")
    monkeypatch.setattr(lab, "LIBRARY", library)
    monkeypatch.setattr(lab, "BUDGETS", budgets)
    return tmp_path


TRACED = {"_traces": [{"lab": "trace", "variant": "a row, every warp's phases stamped", "record": {}}]}


@pytest.mark.parametrize("header", ["k/budgeted.cuh", "k/roles.cuh"])
def test_an_untraced_composed_row_is_refused(tree, header):
    why = lab.untraced("x", f'#include "lab.cuh"\n#include "{header}"\n', {})
    assert why and header in why[0]


@pytest.mark.parametrize("header", ["k/budgeted.cuh", "k/roles.cuh"])
def test_a_traced_composed_row_records(tree, header):
    assert lab.untraced("x", f'#include "{header}"\n', TRACED) == []


def test_a_kernel_of_one_role_and_no_budget_is_not_composed(tree):
    assert lab.untraced("x", '#include "k/plain.cuh"\n#include "lab.cuh"\n', {}) == []
