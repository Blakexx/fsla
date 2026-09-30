#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE COVERAGE GATE: every kernel the source defines meets every obligation, or says why not. Gated by
`tools/lint/ratchet.py coverage`.

`tools/kernels.py` derives the kernels; `tools/coverage.json` declares, per kernel and obligation, the artifact that
meets it, an exemption with its reason, or an owed entry. This prints one finding per gap, and the ratchet holds
today's gaps as its backlog, so:

- a kernel the declaration does not name is a NEW finding: a kernel lands gated, never silently uncovered;
- a pointer to nothing (a deleted test, a renamed sanitizer row, a `file::test` whose test is gone) is a new finding;
- an owed entry is backlog, and paying it (a pointer in its place) shrinks the baseline;
- RATIFIED is read, not declared: a kernel no `tools/manifests/*/sm_*.json` records is a finding per arch.

Usage:

    python tools/lint/coverage.py            # the inventory: every gap, then a per-obligation count
Docs: docs/internals/tools/coverage.md
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import kernels  # noqa: E402 -- path insert must precede this import

DECLARATION = ROOT / "tools" / "coverage.json"
SANITIZER = ROOT / "tools" / "sanitize_oracle.py"
REL = DECLARATION.relative_to(ROOT)


def sanitizer_rows(path: Path = SANITIZER) -> dict[str, tuple[str, bool]]:
    """`{row: (test node, present)}` off `_FAMILIES`, read by AST so the lint imports nothing of the tool's."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "_FAMILIES" for t in node.targets):
            rows = {}
            for key, call in zip(node.value.keys, node.value.values):
                kw = {k.arg: ast.literal_eval(k.value) for k in call.keywords if k.arg in ("test", "present")}
                rows[ast.literal_eval(key)] = (kw["test"], kw["present"])
            return rows
    raise SystemExit(f"coverage: {path.relative_to(ROOT)} has no `_FAMILIES` table to read")


def test_node_exists(node: str) -> bool:
    """`path` or `path::name[params]` names an existing file and, when given, a `def name` in it."""
    path, _, name = node.partition("::")
    target = ROOT / path
    if not target.is_file():
        return False
    return not name or re.search(rf"^\s*(?:async\s+)?def {re.escape(name.split('[')[0])}\(", target.read_text(),
                                 re.M) is not None


def line_of(text: str, kernel: str) -> int:
    for i, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith(f'"{kernel}"'):
            return i
    return 1


def findings() -> list[str]:
    text = DECLARATION.read_text()
    blob = json.loads(text)
    obligations, declared = blob["obligations"], blob["kernels"]
    rows = sanitizer_rows()
    source = {k.name: k for k in kernels.kernels()}
    out = []
    for name, k in sorted(source.items()):
        if name not in declared:
            out.append(f"{k.where()}: `{name}` ({k.family}) is not declared in {REL} -- declare how it meets "
                       f"{sorted(obligations)}")
    for name, entry in sorted(declared.items()):
        at = f"{REL}:{line_of(text, name)}"
        if name not in source:
            out.append(f"{at}: `{name}` is declared but no source defines it -- stale")
            continue
        for obligation in sorted(obligations):
            value = entry.get(obligation)
            if value is None:
                out.append(f"{at}: `{name}` declares nothing for `{obligation}`")
            elif isinstance(value, dict) and "owed" in value:
                out.append(f"{at}: `{name}` owes `{obligation}` -- {value['owed']}")
            elif isinstance(value, dict) and value.get("exempt"):
                continue
            elif obligation == "sanitizer":
                row = rows.get(value)
                if row is None:
                    out.append(f"{at}: `{name}`'s sanitizer row `{value}` is not in tools/sanitize_oracle.py")
                elif not row[1] or not test_node_exists(row[0]):
                    out.append(f"{at}: `{name}`'s sanitizer row `{value}` launches nothing (present={row[1]}, "
                               f"test {row[0]!r})")
            elif not isinstance(value, str) or not test_node_exists(value):
                out.append(f"{at}: `{name}`'s `{obligation}` points at {value!r}, which does not exist")
        for extra in sorted(set(entry) - set(obligations)):
            out.append(f"{at}: `{name}` declares `{extra}`, which is not an obligation")
    for manifest in sorted((ROOT / "tools" / "manifests").glob("*/sm_*.json")):
        arch = manifest.stem.split("_", 1)[1]
        out += kernels.check_manifest(arch)
    return out


def main() -> int:
    found = findings()
    for line in found:
        print(f"  {line}")
    print(f"coverage: {len(kernels.kernels())} kernel(s), {len(found)} gap(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
