# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""A COMMITTED FILE THE GRAPH WRITES (docs/build-graph.md#committed-files): the check fails when the committed file
differs from what the graph's action writes, and names the run target that writes it; the derivation's record is a
function of the declared cases, the sources they name and the harness's output alone."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import committed_file  # noqa: E402
import derivation  # noqa: E402


def test_a_committed_file_that_is_the_graph_s_passes_and_a_differing_one_names_its_update(tmp_path):
    committed, graph = tmp_path / "f.json", tmp_path / "graph.json"
    committed.write_text("a\nb\n")
    graph.write_text("a\nb\n")
    assert committed_file.verdict(committed, graph, "//:f_update") == []
    graph.write_text("a\nc\n")
    found = committed_file.verdict(committed, graph, "//:f_update")
    assert "bazel run //:f_update" in found[0]
    assert "-b" in found and "+c" in found


CASES = {"schema": 1, "sources": ["common/geom.cuh"],
         "geometry": [{"name": "g"}], "sub_boxes": [{"name": "s"}]}
OUTPUT = "g\tok\t2\t0a0b\ns\trefused\tno box\n"


def _src(tmp_path, text="x"):
    (tmp_path / "common").mkdir(exist_ok=True)
    (tmp_path / "common" / "geom.cuh").write_text(text)
    return tmp_path


def test_the_record_is_a_function_of_the_cases_the_sources_and_the_harness_s_output(tmp_path):
    src = _src(tmp_path)
    first = derivation.record(derivation.declared(CASES, src), src, OUTPUT)
    assert derivation.record(derivation.declared(CASES, src), src, OUTPUT) == first
    written = json.loads(first)
    assert written["golden"]["s"] == "refused: no box" and written["golden"]["g"].startswith("ok: 2 bytes")
    assert "date" not in written
    _src(tmp_path, "y")
    assert derivation.record(derivation.declared(CASES, src), src, OUTPUT) != first
    assert derivation.record(derivation.declared(CASES, src), src, OUTPUT.replace("0a0b", "0a0c")) != first


def test_a_record_that_drops_a_case_or_names_no_source_is_refused(tmp_path):
    src = _src(tmp_path)
    with pytest.raises(SystemExit, match="printed no line"):
        derivation.record(CASES, src, "g\tok\t2\t0a0b\n")
    with pytest.raises(SystemExit, match="does not exist"):
        derivation.declared({**CASES, "sources": ["common/none.cuh"]}, src)
    with pytest.raises(SystemExit, match="no `sources`"):
        derivation.declared({**CASES, "sources": []}, src)
