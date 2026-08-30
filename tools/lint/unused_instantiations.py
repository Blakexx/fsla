#!/usr/bin/env python3
"""UNUSED-INSTANTIATION CHECK (LINT2 item 3, G_FOUNDATION G5, REPORT-ONLY until
G5, when it starts DELETING the ones it finds).

Every `__global__` FUNCTION TEMPLATE declared anywhere under `csrc/rola/` must
be present, mangled, in at least one binary's symbol table -- "at least one"
because the SHIPPED set and the TEST-only arm set (`gen_shards.CARRY_SHIPPED_ARMS`
vs `CARRY_TEST_ARMS`) are different binaries, and a kernel instantiated only by
a conformance-only arm is still live. The CHECKED-IN manifests
(`tools/manifests/sm_80.json`, `sm_86.json` -- closed-world codegen policy:
these ARE what a real build produced, ratified) are the source of truth here
rather than a fresh `cuobjdump` on whatever happens to be built in this
worktree right now, which would silently narrow the check to this worktree's
own arm subset (K36b: an arm-set change is a FILE subset, so an iteration
build's `.so` never carries every declared kernel).

SCOPE IS `__global__` ONLY, NOT `__device__` -- MEASURED, not assumed. The
manifest's `entries` are keyed from `ptxas -v`'s "Compiling entry function
'...' for 'sm_XX'" line (`tools/ratify.py`'s `ENTRY` regex) -- `ptxas -v`
reports ENTRY FUNCTIONS ONLY. A `__device__` helper (`ops.cuh`'s `stage_run`,
`box.cuh`'s `canon_leaf_acc`, ...) is normally `__forceinline__` and inlined
into every kernel that calls it; it has NO standalone symbol in this project's
manifest format regardless of whether it is used. Checking `__device__`
templates the same way as `__global__` ones was tried first and produced 65
findings out of 81 candidates, essentially all of them live, frequently-called
helpers this codebase already declares as "wrapped once, call sites use the
wrapper" (KERNEL_STANDARDS §3) -- a false-positive rate that would have made
G5 delete load-bearing code. `__device__` templates are reported SEPARATELY
below, explicitly marked NOT CHECKED, rather than silently flagged as unused.
A real `__device__` liveness check needs a symbol dump off the actual `.cubin`
(`cuobjdump --dump-elf`, filtered to STT_FUNC entries, which DOES preserve
non-inlined `__device__` symbols) cross-referenced against which are actually
called -- out of scope for this report-only stage; named here as the gap.

METHOD for `__global__` (heuristic, not a real C++ parser -- see LIMITS below):
  1. Find every `__global__` qualifier in a `.cu`/`.cuh` under csrc/rola/
     (excluding csrc/third_party), scan forward for the next `void <name>(`
     or `<type> <name>(` token to get the function's UNQUALIFIED name, and
     check the ~15 lines above it for a `template <` line -- only templates
     are in scope (a non-template kernel, e.g. the build's
     `stamp_kernel`/`bwd_stamp_kernel` census probes, is either always
     present or is a one-off entry point this check does not classify).
  2. Read every manifest's `entries` mapping (`"sm_XX:<mangled>": {...}`), run
     `c++filt` once over the whole batch, and check whether the demangled
     signature contains the qualified name `rola::<...>::<name>` -- actually
     just `<name>` as a whole identifier (namespace nesting is not
     reconstructed; a name collision across two `rola::` sub-namespaces would
     under-report, which is why this is a LINT the coordinator rules on, not a
     silent deletion).
  3. A template name with ZERO hits across every manifest is a finding.

LIMITS: (a) no macro expansion -- a kernel name built by token-pasting would
not match its own textual declaration; none exist in this codebase today.
(b) demangled-substring matching can false-negative on a name that is also a
common English/C substring of an unrelated symbol (none observed: kernel names
here are long and specific, e.g. `carry_kernel`, `chunk_facts_tables`).
(c) manifests are committed at the LAST ratify, not this instant -- a kernel
added since the last milestone and not yet ratified reads as unused; that is
exactly the "G5 deletes it" backstop working as designed on stale state, and
is reported as such rather than silently trusted.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSRC = ROOT / "csrc" / "rola"
MANIFEST_DIR = ROOT / "tools" / "manifests"

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


def manifest_demangled(manifest_dir: Path) -> list[str]:
    mangled = []
    #: THE PER-ARCH RATIFICATION FILES, by name. `tools/manifests/` also holds the
    #: DECLARATIONS the build reads (the shipped set, the derivation's cases and its
    #: ratified goldens), which carry no `entries` block; a bare `*.json` glob reads
    #: those as manifests and fails on the missing key.
    for f in sorted(manifest_dir.glob("sm_*.json")):
        entries = json.loads(f.read_text()).get("entries", {})
        for key in entries:
            # key is "sm_XX:<mangled>"
            mangled.append(key.split(":", 1)[1])
    if not mangled:
        return []
    proc = subprocess.run(["c++filt"], input="\n".join(mangled),
                           capture_output=True, text=True)
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
    """Excluded from the commit gate (K46 convention)."""
    fixdir = ROOT / "tools" / "lint" / "fixtures"
    manifest = json.loads((fixdir / "unused_instantiation_manifest.json").read_text())
    demangled = manifest_demangled_from_entries(manifest["entries"])
    templates = _scan(fixdir, GLOBAL_RE)
    names = {name for _, _, name in templates if name in ("present_kernel", "absent_kernel")}
    if "present_kernel" not in names or "absent_kernel" not in names:
        print(f"SELF-TEST FAILED: fixture scan did not find both kernel names: {names}",
              file=sys.stderr)
        return 1
    if not name_present("present_kernel", demangled):
        print("SELF-TEST FAILED: 'present_kernel' should be found in the fixture manifest",
              file=sys.stderr)
        return 1
    if name_present("absent_kernel", demangled):
        print("SELF-TEST FAILED: 'absent_kernel' should NOT be found in the fixture manifest",
              file=sys.stderr)
        return 1
    print("unused_instantiations.py --test-fixtures: PASS")
    return 0


def manifest_demangled_from_entries(entries: dict) -> list[str]:
    mangled = [key.split(":", 1)[1] for key in entries]
    proc = subprocess.run(["c++filt"], input="\n".join(mangled), capture_output=True, text=True)
    return proc.stdout.splitlines()


def main() -> int:
    if "--test-fixtures" in sys.argv:
        return self_test()

    if not MANIFEST_DIR.is_dir() or not list(MANIFEST_DIR.glob("sm_*.json")):
        print("unused_instantiations: SKIPPED -- no tools/manifests/sm_*.json "
              "(ratify at least once: tools/ratify.py --arch 80 --arch 86)")
        return 0

    templates = declared_global_templates(CSRC)
    demangled = manifest_demangled(MANIFEST_DIR)

    findings = []
    for path, lineno, name in templates:
        if not name_present(name, demangled):
            findings.append(f"{path.relative_to(ROOT)}:{lineno}: template kernel "
                             f"'{name}' not found (mangled) in any manifest under "
                             f"tools/manifests/ -- G5 deletes it if this holds at "
                             f"the next ratify")

    print(f"unused_instantiations: {len(templates)} template __global__ function(s) "
          f"checked against {len(demangled)} manifest symbol(s)")
    for f in findings:
        print(f"  {f}")
    print(f"unused_instantiations: {len(findings)} finding(s)")

    device_templates = declared_device_templates(CSRC)
    print(f"unused_instantiations: {len(device_templates)} template __device__ "
          f"function(s) found -- NOT CHECKED (see module docstring: ptxas -v / the "
          f"manifest format only records __global__ entry points; a __device__ "
          f"helper is normally inlined and has no standalone symbol to look up)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
