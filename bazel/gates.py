#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE ARM GATE FOR THE GRAPH'S LIBRARY: the built arm table against the declared arms, and the coverage sweep
(`bazel/arms.py`'s `check_built_arm_table` and `sweep_binary`) -- the binary asked
what it carries, loaded by the environment's interpreter against the toolkit's driver stub, so the answer is the
binary's on any machine, GPU or none. Run by `bazel/defs.bzl`'s `fsla_extension`; writes its verdict to `--out`.
Docs: docs/build-graph.md#gates.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

#: unresolved: in the sandbox the script is a link, and its siblings are the action's declared inputs beside it
ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "bazel"))
import arms  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--so", type=Path, required=True, help="the built extension")
    ap.add_argument("--header", action="append", default=[], metavar="FAMILY=PATH",
                    help="a family's selection header the build compiled against")
    ap.add_argument("--table", action="append", default=[], metavar="FAMILY=PATH", help="a family's generated table")
    ap.add_argument("--python", required=True, help="the environment's interpreter, torch beside it")
    ap.add_argument("--driver-stub", type=Path, required=True, help="the toolkit's libcuda.so stub")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    headers, tables = arms.by_family(a.header, "--header"), arms.by_family(a.table, "--table")
    report = []
    with tempfile.TemporaryDirectory(prefix="fsla_arm_gate_") as tmp:
        #: the stub under its soname, the one the binary's NEEDED names
        (Path(tmp) / "libcuda.so.1").symlink_to(a.driver_stub.absolute())
        for family in sorted(tables):
            table = arms.Table(Path(tables[family]))
            want = list(range(len(table.arms)))
            bad = arms.check_built_arm_table(Path(headers[family]).read_text(), want, table, family)
            if bad:
                raise SystemExit(f"the built {family} arm table:\n" + "\n".join(bad))
            swept = arms.sweep_binary(str(Path(a.python).absolute()), a.so.absolute(), family, want, table, cwd=Path(tmp),
                                      env={"LD_LIBRARY_PATH": tmp, "PATH": "/usr/bin:/bin"})
            report.append(f"arm table OK  {family}: {len(want)} arms the build's contract reaches; {swept} arm(s) swept\n")
    a.out.write_text("".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
