#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE LEARNINGS' GATE: every row of `docs/internals/LEARNINGS.md` names a mechanism that exists.

    python3 tools/lint/learnings.py

A row's last cell holds at least one `path::text`: a file of the tree and a string the file contains. A row that
names none, a path that is gone, or a string its file no longer holds fails the commit -- a lesson is kept by what
enforces it (docs/internals/LEARNINGS.md), so a lesson whose mechanism left the tree is a finding.

Docs: docs/internals/tools/lint.md.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "docs" / "internals" / "LEARNINGS.md"
MECHANISM = re.compile(r"`([^`\s:]+)::([^`]+)`")


def findings(text: str, root: Path = ROOT) -> list[str]:
    """Each row's faults, by its lesson's first words."""
    found = []
    rows = [line for line in text.splitlines() if line.startswith("|")][2:]
    if not rows:
        found.append("the ledger holds no row")
    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        lesson = re.sub(r"<[^>]+>", "", cells[0])[:60]
        named = MECHANISM.findall(cells[-1])
        if not named:
            found.append(f"{lesson!r}: names no mechanism (`path::text`)")
        for path, needle in named:
            target = root / path
            if not target.is_file():
                found.append(f"{lesson!r}: its mechanism's file `{path}` is not in the tree")
            elif needle not in target.read_text():
                found.append(f"{lesson!r}: `{path}` no longer holds `{needle}`")
    return found


def main() -> int:
    found = findings(LEDGER.read_text())
    for line in found:
        print(f"learnings: {line}")
    print(f"learnings: {'FAIL' if found else 'ok'} -- {len(found)} finding(s)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
