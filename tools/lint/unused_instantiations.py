#!/usr/bin/env python3
"""UNUSED-INSTANTIATION CHECK (gating).

Every `__global__` FUNCTION TEMPLATE declared anywhere under `csrc/rola/` must be instantiated in at least one unit the
graph compiled: its mangled name is among the entry functions `ptxas -v` reported for the build's units. The census
(`bazel/defs.bzl`'s `ratify_census`, read through `tools/ratify.py`) is the source of truth, the build's own compiles --
a template no unit instantiates is the closed world leaking, and that is the gating finding.

SCOPE IS `__global__` ONLY, NOT `__device__` -- MEASURED, not assumed. `ptxas -v` reports ENTRY FUNCTIONS ONLY. A
`__device__` helper (`ops.cuh`'s `stage_run`, `geom.cuh`'s `geom_read_index`, ...) is normally `__forceinline__` and
inlined into every kernel that calls it; it has NO standalone symbol in a report regardless of whether it is used.
Checking `__device__` templates the same way as `__global__` ones was tried first and produced 65 findings out of 81
candidates, essentially all of them live, frequently-called helpers this codebase already declares as "wrapped once, call
sites use the wrapper" (KERNEL_STANDARDS §3) -- a false-positive rate that would have made a deletion pass remove
load-bearing code. `__device__` templates are reported SEPARATELY below, explicitly marked NOT CHECKED, rather than
silently flagged as unused. A real `__device__` liveness check needs a symbol dump off the actual `.cubin`
(`cuobjdump --dump-elf`, filtered to STT_FUNC entries, which DOES preserve non-inlined `__device__` symbols)
cross-referenced against which are actually called -- out of scope for this check; named here as the gap.

METHOD for `__global__` (heuristic, not a real C++ parser -- see LIMITS below):
  1. Find every `__global__` qualifier in a `.cu`/`.cuh` under csrc/rola/ (excluding csrc/third_party), scan forward for
     the next `void <name>(` or `<type> <name>(` token to get the function's UNQUALIFIED name, and check the ~15 lines
     above it for a `template <` line -- only templates are in scope (a non-template kernel is either always present or
     is a one-off entry point this check does not classify).
  2. Read the census's entry functions, run `c++filt` once over the whole batch, and check whether the demangled
     signature contains `<name>` as a whole identifier (namespace nesting is not reconstructed; a name collision across
     two `rola::` sub-namespaces would under-report, which is why this is a LINT a reviewer rules on, not a silent
     deletion).
  3. A template name with ZERO hits across the census is a finding.

LIMITS: (a) no macro expansion -- a kernel name built by token-pasting would not match its own textual declaration; none
exist in this codebase today. (b) demangled-substring matching can false-negative on a name that is also a common
English/C substring of an unrelated symbol (none observed: kernel names here are long and specific). (c) the census is
the build's: an iteration build (`--//:<family>_arms`, `--//:archs`) compiles a subset, and a template only the other arms
instantiate reads as unused -- run the check on the default build.

Usage: `bazel test //csrc/rola:unused_instantiations`, which runs it on the graph's census.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSRC = ROOT / "csrc" / "rola"

GLOBAL_RE = re.compile(r"__global__")
DEVICE_RE = re.compile(r"__device__")
TEMPLATE_RE = re.compile(r"^\s*template\s*<")
NAME_RE = re.compile(r"\b(?:void|int|float|uint32_t|int64_t|bool)\s+(\w+)\s*\(")
IDENT_BOUNDARY = re.compile(r"[A-Za-z0-9_]")


def _scan(src_root: Path, qualifier_re: re.Pattern) -> list[tuple[Path, int, str]]:
    """[(file, line_no, name), ...] for every template fn matching qualifier_re."""
    found = []
    for suf in (".cu", ".cuh"):
        for p in src_root.rglob(f"*{suf}"):
            if "third_party" in p.parts:
                continue
            lines = p.read_text().splitlines()
            for i, line in enumerate(lines):
                if not qualifier_re.search(line):
                    continue
                # Is this a template? Scan up to 15 lines back for `template <`.
                is_template = any(
                    TEMPLATE_RE.match(lines[j]) for j in range(max(0, i - 15), i)
                )
                if not is_template:
                    continue
                window = "\n".join(lines[i : i + 12])
                m = NAME_RE.search(window)
                if m:
                    found.append((p, i + 1, m.group(1)))
    return found


def declared_global_templates(src_root: Path) -> list[tuple[Path, int, str]]:
    return _scan(src_root, GLOBAL_RE)


def declared_device_templates(src_root: Path) -> list[tuple[Path, int, str]]:
    return _scan(src_root, DEVICE_RE)


def demangled(mangled: list[str]) -> list[str]:
    """The demangled signature of each mangled entry-function name."""
    if not mangled:
        return []
    proc = subprocess.run(["c++filt"], input="\n".join(mangled), capture_output=True, text=True)
    return proc.stdout.splitlines()


def name_present(name: str, demangled: list[str]) -> bool:
    for sig in demangled:
        idx = sig.find(name)
        while idx != -1:
            before_ok = idx == 0 or not IDENT_BOUNDARY.match(sig[idx - 1])
            after = idx + len(name)
            after_ok = after >= len(sig) or not IDENT_BOUNDARY.match(sig[after])
            if before_ok and after_ok:
                return True
            idx = sig.find(name, idx + 1)
    return False


def self_test() -> int:
    """The scan finds both fixture kernels, and the name match finds the instantiated one and not the other."""
    fixdir = ROOT / "tools" / "lint" / "fixtures"
    entries = json.loads((fixdir / "unused_instantiation_entries.json").read_text())["entries"]
    symbols = demangled([key.split(":", 1)[1] for key in entries])
    templates = _scan(fixdir, GLOBAL_RE)
    names = {name for _, _, name in templates if name in ("present_kernel", "absent_kernel")}
    if "present_kernel" not in names or "absent_kernel" not in names:
        print(f"SELF-TEST FAILED: fixture scan did not find both kernel names: {names}",
              file=sys.stderr)
        return 1
    if not name_present("present_kernel", symbols):
        print("SELF-TEST FAILED: 'present_kernel' should be found in the fixture's entries",
              file=sys.stderr)
        return 1
    if name_present("absent_kernel", symbols):
        print("SELF-TEST FAILED: 'absent_kernel' should NOT be found in the fixture's entries",
              file=sys.stderr)
        return 1
    print("unused_instantiations.py --test-fixtures: PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-fixtures", action="store_true", help="prove the scan and the name match on the fixtures")
    ap.add_argument("--graph", type=Path, help="the graph's census record (bazel/defs.bzl's `ratify_census`)")
    ap.add_argument("--runfiles", type=Path, help="the directory the census record's paths are relative to")
    a = ap.parse_args()
    if a.test_fixtures:
        return self_test()
    if not (a.graph and a.runfiles):
        raise SystemExit("this check reads the graph's census: `bazel test //csrc/rola:unused_instantiations`")

    sys.path.insert(0, str(ROOT / "tools"))
    import ratify

    ratify.use_graph(a.graph, a.runfiles)
    symbols = demangled(ratify.compiled_entries())
    templates = declared_global_templates(CSRC)
    findings = []
    for path, lineno, name in templates:
        if not name_present(name, symbols):
            findings.append(f"{path.relative_to(ROOT)}:{lineno}: template kernel '{name}' is instantiated by no unit "
                            f"the graph compiled -- delete it, or declare the arm that instantiates it")

    print(f"unused_instantiations: {len(templates)} template __global__ function(s) "
          f"checked against {len(symbols)} compiled entry function(s)")
    for f in findings:
        print(f"  {f}")
    print(f"unused_instantiations: {len(findings)} finding(s)")

    device_templates = declared_device_templates(CSRC)
    print(f"unused_instantiations: {len(device_templates)} template __device__ "
          f"function(s) found -- NOT CHECKED (see module docstring: `ptxas -v` only records __global__ entry "
          f"points; a __device__ helper is normally inlined and has no standalone symbol to look up)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
