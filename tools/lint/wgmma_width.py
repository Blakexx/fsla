#!/usr/bin/env python3
"""THE TENSOR STEP'S WIDTH LINT (gating, CONSTITUTION R25): no library kernel issues a `wgmma.mma_async` step wider than 64
columns. A GROUP is a layout and its steps are shaped to it, narrow up to 64 columns and never wider: a layout wider than a
step is several 64-column steps (a step's own time is 32, 64 and 128 cycles at 64, 128 and 256 columns, `docs/lab/wgmma_width.md`,
and ptxas refuses a 256-column step in a kernel that enters at 128 registers a thread, `docs/lab/reg_trade.md`).

Reads every device and host source of `csrc/fsla` (the library's own; `csrc/third_party` is vendored and `measure/lab/` is exempt:
`wgmma_width` measures the wide steps on purpose) and fails on a `wgmma.mma_async` mnemonic of shape `m<M>n<N>k<K>` with N over 64.
Comments are text and pass. The mnemonic is read as written in the source: a shape spelled by a template argument or built from
string pieces is not seen (`docs/rules.json`'s row says so).

`--test-fixtures` (the graph's `//tools/lint:wgmma_width_fixtures`): the wide fixture must fire on each of its wide steps and the
narrow fixture on none.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSRC = ROOT / "csrc" / "fsla"
FIXDIR = ROOT / "tools" / "lint" / "fixtures"
SUFFIXES = (".cu", ".cuh", ".cpp", ".h", ".hpp")

#: the widest step a library kernel issues, in columns
MAX_COLUMNS = 64

STEP = re.compile(r"wgmma\.mma_async[\w.]*?\.m(\d+)n(\d+)k(\d+)")
COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)


def blank_comments(text: str) -> str:
    """The source with each comment's characters replaced by spaces (newlines kept), so a finding's line is the source's."""
    return COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def findings(path: Path, text: str) -> list[str]:
    """Each `wgmma.mma_async` step of `text` wider than `MAX_COLUMNS`, as `path:line: ...`."""
    code = blank_comments(text)
    out = []
    for m in STEP.finditer(code):
        columns = int(m.group(2))
        if columns > MAX_COLUMNS:
            line = code.count("\n", 0, m.start()) + 1
            out.append(f"{path}:{line}: `{m.group(0)}` is a step of {columns} columns; a library kernel's step is at most "
                       f"{MAX_COLUMNS} (CONSTITUTION R25): lay the group out as {columns // MAX_COLUMNS} steps of {MAX_COLUMNS} columns "
                       "on one A operand")
    return out


def sources() -> list[Path]:
    return sorted(p for suffix in SUFFIXES for p in CSRC.rglob(f"*{suffix}"))


def self_test() -> int:
    wide = FIXDIR / "wgmma_width_wide.txt"
    narrow = FIXDIR / "wgmma_width_narrow.txt"
    got = findings(wide, wide.read_text())
    ok = True
    if [f.split(":")[1] for f in got] != ["4", "7", "10"]:
        print(f"SELF-TEST FAILED: the wide fixture must fire at lines 4, 7 and 10, got {got}", file=sys.stderr)
        ok = False
    stray = findings(narrow, narrow.read_text())
    if stray:
        print(f"SELF-TEST FAILED: the narrow fixture fired: {stray}", file=sys.stderr)
        ok = False
    if not ok:
        return 1
    print("wgmma_width.py --test-fixtures: PASS")
    return 0


def main() -> int:
    if "--test-fixtures" in sys.argv:
        return self_test()
    files = sources()
    found = [f.replace(str(p), str(p.relative_to(ROOT))) for p in files for f in findings(p, p.read_text())]
    print(f"wgmma_width: {len(files)} source(s) of csrc/fsla read, {len(found)} step(s) wider than {MAX_COLUMNS} columns")
    for f in found:
        print(f"  {f}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
