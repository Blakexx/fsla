#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE BUILD'S CARRIED BUILDS HEADER for one build's architectures (`measure/hardware.py`'s `builds_header`). Run by the
graph's header action (`bazel/defs.bzl`'s `rola_headers`); the arm families' headers are `bazel/arms.bzl`'s.
Docs: docs/build-graph.md#compile.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: unresolved: in the sandbox the script is a link, and its siblings are the action's declared inputs beside it
ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))

from measure import hardware  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--archs", required=True, help="the build's architectures, comma-separated (`90a`)")
    a = ap.parse_args()
    (a.out_dir / "builds.inc").write_text(hardware.builds_header([x for x in a.archs.split(",") if x]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
