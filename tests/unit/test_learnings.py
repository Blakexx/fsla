# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The learnings' gate (tools/lint/learnings.py): a row is kept by a mechanism the tree holds."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "lint"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import learnings  # noqa: E402

HEAD = "| the lesson | the incident | the rule | enforced by |\n|---|---|---|---|\n"


def test_the_trees_ledger_passes():
    assert learnings.findings(learnings.LEDGER.read_text()) == []


def test_a_row_without_a_mechanism_is_refused(tmp_path):
    (tmp_path / "tool.py").write_text("CHECK = 1\n")
    kept = HEAD + "| a | b | c | `tool.py::CHECK` |\n"
    assert learnings.findings(kept, tmp_path) == []
    assert "names no mechanism" in learnings.findings(HEAD + "| a | b | c | be careful |\n", tmp_path)[0]
    assert "not in the tree" in learnings.findings(HEAD + "| a | b | c | `gone.py::CHECK` |\n", tmp_path)[0]
    assert "no longer holds" in learnings.findings(HEAD + "| a | b | c | `tool.py::OTHER` |\n", tmp_path)[0]
    assert learnings.findings(HEAD, tmp_path) == ["the ledger holds no row"]


def test_an_experiments_timeline_is_read_from_its_header(tmp_path):
    import lab

    with_one, without = tmp_path / "a.cu", tmp_path / "b.cu"
    with_one.write_text("// a.cu -- QUESTION: q METHOD: m\n// TIMELINE: load, then use\n#include \"lab.cuh\"\n")
    without.write_text("// b.cu -- QUESTION: q METHOD: m\n#include \"lab.cuh\"\n// TIMELINE: too late\n")
    assert lab.timeline(with_one) and not lab.timeline(without)
