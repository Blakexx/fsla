#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE COVERAGE GATE: every kernel the source defines meets every obligation, or says why not. Gated by
`tools/lint/ratchet.py coverage`.

`tools/kernels.py` derives the kernels; `tools/coverage.json` declares, per kernel and obligation, the artifact that
meets it, an exemption with its reason, or an owed entry with its reason. The FINDINGS (the ratchet gates them) are
the gaps nobody declared:

- a kernel the declaration does not name: a kernel lands declared, never silently uncovered;
- a pointer to nothing (a deleted test, a renamed sanitizer row, a `file::test` whose test is gone);
- an obligation left blank, a declared kernel the source no longer defines.

OWED entries are the backlog, printed with their reasons and counted, never findings: a new kernel necessarily lands
owing (no budget yet, no sanitizer row), and each owed line is a reviewed line
of the declaration.

Usage:

    bazel run //tools/lint:coverage          # the inventory: every gap, then a per-obligation count
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


#: THE FLOOR GATES' PACKAGE: every `kernel_floor` is here, where `//measure/floors:floor` (a member of `//:gate`) runs each
FLOORS = "measure/floors"


def floor_test_exists(label: str) -> bool:
    """`//measure/floors:<name>_floor` names the test of a `kernel_floor(name = "<name>"` call in the floor gates' BUILD.bazel."""
    found = re.fullmatch(rf"//{FLOORS}:([\w.-]+)_floor", label)
    build = ROOT / FLOORS / "BUILD.bazel"
    return bool(found) and build.is_file() and \
        re.search(rf'kernel_floor\(\s*name = "{re.escape(found.group(1))}"', build.read_text()) is not None


def line_of(text: str, kernel: str) -> int:
    for i, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith(f'"{kernel}"'):
            return i
    return 1


def findings() -> list[str]:
    text = DECLARATION.read_text()
    blob = json.loads(text)
    obligations, declared = blob["obligations"], blob["kernels"]
    source = {k.name: k for k in kernels.kernels()}
    out = [f"{k.where()}: `{name}` ({k.family}) is not declared in {REL} -- declare how it meets {sorted(obligations)}"
           for name, k in sorted(source.items()) if name not in declared]

    rows = {"sanitizer": sanitizer_rows()}
    for name, entry in sorted(declared.items()):
        at = f"{REL}:{line_of(text, name)}"
        if name not in source:
            out.append(f"{at}: `{name}` is declared but no source defines it -- stale")
            continue
        for obligation in sorted(obligations):
            out += _check(name, obligation, entry.get(obligation), at, rows)
        out += [f"{at}: `{name}` declares `{extra}`, which is not an obligation"
                for extra in sorted(set(entry) - set(obligations))]
    return out


def _check(name: str, obligation: str, value, at: str, rows) -> list[str]:
    """The findings of one kernel's one obligation: none when it is owed, exempt, or a pointer that resolves."""
    if value is None:
        return [f"{at}: `{name}` declares nothing for `{obligation}`"]
    if isinstance(value, dict) and ("owed" in value or value.get("exempt")):
        return []
    if obligation == "floor":
        #: THE FLOOR OBLIGATION'S POINTER IS THE GATE'S TEST (KERNEL_STANDARDS section 28): `//measure/floors:<name>_floor`, the test
        #: a `kernel_floor(name = "<name>", ...)` call of bazel/floor.bzl in the floor gates' package declares
        if isinstance(value, str) and floor_test_exists(value):
            return []
        return [f"{at}: `{name}`'s floor points at {value!r}, which is no `kernel_floor` test (`//{FLOORS}:<name>_floor`, a "
                f"`kernel_floor(name = \"<name>\"` call in {FLOORS}/BUILD.bazel: bazel/floor.bzl)"]
    if obligation == "sanitizer":
        row = rows["sanitizer"].get(value)
        if row is None:
            return [f"{at}: `{name}`'s sanitizer row `{value}` is not in tools/sanitize_oracle.py"]
        if not row[1] or not test_node_exists(row[0]):
            return [f"{at}: `{name}`'s sanitizer row `{value}` launches nothing (present={row[1]}, test {row[0]!r})"]
        return []
    if not isinstance(value, str) or not test_node_exists(value):
        return [f"{at}: `{name}`'s `{obligation}` points at {value!r}, which does not exist"]
    return []


def owed() -> list[str]:
    """Every owed obligation, with its reason: the backlog."""
    blob = json.loads(DECLARATION.read_text())
    return [f"{name}: {obligation} -- {value['owed']}" for name, entry in sorted(blob["kernels"].items())
            for obligation, value in sorted(entry.items()) if isinstance(value, dict) and "owed" in value]


def main() -> int:
    found = findings()
    backlog = owed()
    for line in found:
        print(f"  {line}")
    for line in backlog:
        print(f"  owed: {line}")
    print(f"coverage: {len(kernels.kernels())} kernel(s), {len(found)} undeclared gap(s), {len(backlog)} owed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
