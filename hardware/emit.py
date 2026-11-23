#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE REGISTRY'S OUTPUTS, written for the graph's actions. For a KERNEL (`--kernel`): `plan/<kernel>/facts.h` (the kernel's `Facts` class and reader, from the shared
device table and the kernel's schema), `plan/<kernel>/contract.h` (its contract, its owed contexts and its `Kernel` definition, as the run time holds them) and the
contract as the matchers' tool reads it (`--contexts`). And `common/hardware.cuh`, the architectures the device parts state as the kernels read them (`--arch-header`).
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
    ap.add_argument("--kernel", help="the kernel the facts, contexts and contract outputs are of")
    ap.add_argument("--facts-header", type=Path)
    ap.add_argument("--contexts", type=Path)
    ap.add_argument("--contract-header", type=Path)
    ap.add_argument("--arch-header", type=Path)
    ap.add_argument("--contract", default=registry.FULL, help="the contract: `full`, or a narrowed contract of one kernel's descriptors/dev (the build's `--//:contract`)")
    a = ap.parse_args()
    registry.check(a.root)
    if (a.facts_header or a.contexts or a.contract_header) and not a.kernel:
        ap.error("--facts-header, --contexts and --contract-header are a kernel's: name it with --kernel")
    if a.facts_header:
        a.facts_header.write_text(registry.facts_header(a.kernel, registry.facts(a.kernel, a.root)))
    if a.contexts:
        applied = registry.applied(a.kernel, a.contract, a.root)
        a.contexts.write_text(registry.contexts_text(a.kernel, registry.contexts(a.kernel, a.root, a.contract), applied))
    if a.contract_header:
        applied = registry.applied(a.kernel, a.contract, a.root)
        a.contract_header.write_text(registry.contract_header(a.kernel, registry.contexts(a.kernel, a.root, a.contract), registry.owed(a.kernel, a.root),
                                                              registry.facts(a.kernel, a.root), applied))
    if a.arch_header:
        a.arch_header.write_text(registry.arch_header(registry.architectures(a.root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
