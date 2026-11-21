# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""A COMMITTED FILE THE GRAPH WRITES, HELD TO IT: the check of the pattern `bazel/defs.bzl`'s `committed_file` makes
(docs/build-graph.md#committed-files). The committed file is byte for byte what the graph's action writes, or the check
fails with the difference and the one command that writes it. The standard library only.

    committed_file.py --committed README.md --graph <the action's output> --update //:readme_update
"""
from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

#: the lines of the difference a failure shows
SHOWN = 40


def verdict(committed: Path, graph: Path, update: str) -> list[str]:
    """The failure's lines: empty when the committed file is what the graph writes."""
    have, want = committed.read_text(), graph.read_text()
    if have == want:
        return []
    diff = list(difflib.unified_diff(have.splitlines(), want.splitlines(), "committed", "the graph's", lineterm="", n=1))
    shown = diff[:SHOWN] + ([f"... {len(diff) - SHOWN} more lines"] if len(diff) > SHOWN else [])
    return [f"{committed.name} is not what the graph writes; `bazel run {update}` writes it", *shown]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--committed", type=Path, required=True, help="the committed file")
    ap.add_argument("--graph", type=Path, required=True, help="the file the graph's action wrote")
    ap.add_argument("--update", required=True, help="the run target that writes the committed file")
    a = ap.parse_args()
    lines = verdict(a.committed, a.graph, a.update)
    print("\n".join(lines) if lines else f"{a.committed.name}: what the graph writes")
    return 1 if lines else 0


if __name__ == "__main__":
    sys.exit(main())
