#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE REGISTRY'S OUTPUTS, written for the graph's actions: `plan/facts.h` (the `Facts` struct, from the schema), the contract as the matchers'
tool reads it (`--contexts`) and as the run time holds it (`--contract-header`), and the README's supported-configurations rows (`--supported`, `tools/supported.py`).
Docs: docs/internals/hardware/registry.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from hardware import registry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=registry.ROOT, help="the registry's directory")
    ap.add_argument("--facts-header", type=Path)
    ap.add_argument("--contexts", type=Path)
    ap.add_argument("--contract-header", type=Path)
    ap.add_argument("--contract", default=registry.FULL, help="the contract: `full`, or a narrowed contract of descriptors/dev (the build's `--//:contract`)")
    a = ap.parse_args()
    registry.check(a.root)
    if a.facts_header:
        a.facts_header.write_text(registry.facts_header(registry.schema(a.root)))
    if a.contexts:
        a.contexts.write_text(registry.contexts_text(registry.contexts(a.root, a.contract), a.contract))
    if a.contract_header:
        a.contract_header.write_text(registry.contract_header(registry.contexts(a.root, a.contract), registry.schema(a.root), a.contract))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
