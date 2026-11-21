#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE SYNTHETIC SELECTION HEADER of tests/unit/test_arm_switch.py's probe: the one renderer the graph's header action uses
(`tools/gen_shards.py`'s `carry_selection_header`) over a three-row table, of which a build carries the arms it is given.
Run by the genrules of tests/BUILD.bazel.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: unresolved: in the sandbox the script is a link, and its siblings are the action's declared inputs
ROOT = Path(__file__).absolute().parents[3]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT)]
import gen_shards  # noqa: E402

#: THE SYNTHETIC TABLE: three rows on the real key, of which a build carries the first and the last. The middle row is what
#: makes "declared but not built" a distinct case from "not declared".
TABLE = ((32, 1, 32, 1, 32, 0), (32, 1, 32, 1, 32, 1), (128, 2, 128, 2, 64, 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="", help="the arms the build carries, comma-separated; empty is none")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    gen_shards.CARRY_ARMS = TABLE
    a.out.write_text(gen_shards.carry_selection_header([int(x) for x in a.arms.split(",") if x]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
