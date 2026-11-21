"""EVERY HEADER COMPILES ALONE -- the gate for a name a header uses but does not include.

**WHAT THIS CATCHES.** A header that references a name it never included compiles
anyway, silently, whenever some *other* file in the translation unit happened to
include that name's home first. Nothing in the build says so, the layering it claims
in its own file header is false, and the break surfaces later -- when a new caller
includes the header first, or when the include that was carrying it is removed for an
unrelated reason. It is the standard failure mode of splitting one file into layers,
and this project met it the first time it did: an extracted layout header used a
constant from its caller's header and compiled only because that caller had
included the constant's home four lines earlier.

**HOW.** One translation unit per header, containing that header twice and nothing
else (`csrc/rola/BUILD.bazel`'s `header_selfcheck_sources`). Twice, because the second
inclusion is what proves the include guard: a header without `#pragma once` passes the
single-inclusion form and fails here. The graph's CUDA rule compiles each unit with the
SAME flag list and include roots the extension is built with (`header_selfcheck_units`), so
a header that needs a define the build supplies is not reported as broken and a header
that needs one the build does NOT supply is; a unit that does not compile fails the build
this test depends on, naming the header. This tool is the test over what the graph built:
every header has its unit, and its object is there.

    bazel test //csrc/rola:header_selfcheck --test_output=all

ONE BUILD, the library's: this asks a question about NAMES, not about codegen. Neither
does `--ptxas-options=-v` matter: there are no entry points in these translation units.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def unit_stem(header: Path) -> str:
    """The unit's stem for a header (`csrc/rola/src/common/x.cuh` -> `common__x_selfcheck`), as the BUILD file names it."""
    parts = header.parts[header.parts.index("src") + 1:]
    return "__".join(parts).removesuffix(".cuh").replace("/", "__") + "_selfcheck"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headers", nargs="+", type=Path, required=True, help="every header under csrc/rola/src")
    ap.add_argument("--objects", nargs="+", type=Path, required=True, help="the units' objects, the graph's compile of them")
    args = ap.parse_args()

    built = {obj.name.removesuffix(".pic.o").removesuffix(".o") for obj in args.objects}
    missing = [h for h in args.headers if unit_stem(h) not in built]
    for header in args.headers:
        print(f"{'FAIL' if header in missing else 'ok  '}  {header}")
    if missing:
        print(f"\nFAIL  {len(missing)} of {len(args.headers)} header(s) have no compiled unit")
        return 1
    print(f"\nPASS  all {len(args.headers)} headers compile standalone (and twice over)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
