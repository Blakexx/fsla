# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""A COMMITTED FILE THE GRAPH WRITES (docs/build-graph.md#committed-files): the check fails when the committed file
differs from what the graph's action writes, and names the run target that writes it."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import committed_file  # noqa: E402


def test_a_committed_file_that_is_the_graph_s_passes_and_a_differing_one_names_its_update(tmp_path):
    committed, graph = tmp_path / "f.json", tmp_path / "graph.json"
    committed.write_text("a\nb\n")
    graph.write_text("a\nb\n")
    assert committed_file.verdict(committed, graph, "//:f_update") == []
    graph.write_text("a\nc\n")
    found = committed_file.verdict(committed, graph, "//:f_update")
    assert "bazel run //:f_update" in found[0]
    assert "-b" in found and "+c" in found
