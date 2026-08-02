"""SPILL-MANIFEST GATE for the consumer kernel's launch bounds.

**WHAT THIS CONVERTS, AND WHY.** the consumer rewrite's §2.21 ruling landed a
three-term residency formula and ACCEPTED one 4-byte spill on one of 128
instantiations as the measured optimum -- the alternatives spill worse (the unbounded
status quo: 20 st / 56 ld; the thread-slot cap: 70 st / 284 ld). That acceptance was a
judgment recorded in a document, and it lived nowhere else: **nothing failed if a
later codegen change grew it.** This file makes it an ASSERTED INVARIANT, the same
doctrine as the dense-degeneracy bit-identity gate -- a property the batch believes is
a property the batch checks.

FOUR OUTCOMES, and the difference between them is deliberate:

* **FAIL -- toolchain drift.** The `ptxas` this gate invokes reports a version string
  different from the one the manifest was ratified under. **CLOSED-WORLD CODEGEN POLICY
  RULE 1 (user directive, 2026-08-01.)** A spill/register manifest is a property of the
  ASSEMBLER as much as of the source: every number in this file was produced by one
  `ptxas`, and a different one may legitimately reach different allocations. Checking a
  new assembler's output against an old assembler's ratification is wrong in BOTH
  directions -- it can fail a regression nobody introduced, and it can pass one the new
  allocator did introduce while the stale numbers happened to be looser. So the version
  is RECORDED in the manifest and CHECKED, and a mismatch is a hard failure demanding
  re-ratification (`--write`), never a warning: a warning is read once and thereafter
  ignored, which is the same as not checking.
* **FAIL -- spill growth.** Any instantiation whose spill bytes EXCEED its manifest
  entry. Spills are the thing being bounded; growth is a regression whatever caused it.
* **FAIL -- an unhonored bound.** Any `Value of threads per SM ... is out of range`
  warning. That entry compiled with NO launch bound, so `ptxas` was free to spend
  registers as it liked -- correction (d)'s failure mode, and the defect §2.21 exists
  to have fixed. It is a hard failure even at zero spills.
* **REPORT -- register drift.** A changed `Used N registers` with no spill growth is
  printed, not failed. The register count is an ALLOCATOR OUTCOME that legitimately
  moves when the kernel changes; what must not move is the spilling.

**RE-RATIFICATION IS A MEASUREMENT, NOT AN EDIT.** The manifest is regenerated only by
`--write`, which re-runs this same compile and records what it reports. Hand-editing an
entry to make the gate pass would be asserting a spill nobody measured, which is
exactly the failure this file replaces. When a kernel change legitimately moves the
numbers: re-run `--write`, and put the new numbers in the findings document with the
reason they moved.

Usage (no GPU required -- this is a COMPILE-time gate, so it does not take the lock):

    python tools/ratify.py
    python tools/ratify.py --arch 80 --arch 86
    python tools/ratify.py --arch 89 --write        # bringing up a new architecture

`--self-test` requires the gate to FAIL on four things it must reject: a deliberately
tightened manifest, a deliberately perturbed `ptxas` version, an arch with no committed
manifest, and a per-arch split that does not rejoin to one digest. That is what makes a
green run mean something: a gate that cannot fail is not a gate (`rewrite.md`'s vacuity
discipline, applied to itself).

MANIFEST FORMAT: `tools/manifests/sm_XX.json`, ONE FILE PER ARCH (§3.7), each
`{"toolchain": {"ptxas": <version string>}, "entries": {<mangled name>: {...}}}`. The
toolchain block is a sibling of the entries rather than a reserved key inside them, so
an instantiation can never be confused for metadata by a name that happens to collide.

**WHY PER ARCH.** *"Does arch X have a ratified manifest?"* becomes a file-existence
question, which is what `setup.py`'s rule-3 gate needs to ask before it compiles
anything; a new architecture's bring-up is then one reviewable file in a PR instead of a
1024-line diff; and an `--arch 89` run cannot restate measurements it did not take. The
SPLIT IS FILING ONLY, and that is checked rather than asserted: the digest
`_build_config.py` records is taken over the merged canonical record, so N files hash to
exactly what one file hashed to (`--self-test`, last arm).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import sysconfig
import time
from pathlib import Path


def cuda_bin(tool: str) -> str:
    """ONE toolchain location for both binaries, resolved the way the BUILD resolves it.

    `nvcc` drives the compile and `ptxas` is the assembler it invokes and whose
    report this gate parses, so reading the version from a DIFFERENT `ptxas` than
    the one that produced the numbers would defeat the check it exists to make.
    Deriving both from one function is what makes that impossible rather than merely
    unlikely -- and the resolution order is `setup.py`'s, because the manifest must
    describe the assembler the EXTENSION will be built with: torch's `CUDA_HOME`
    first, then `$CUDA_HOME`, then the conventional prefix as a last resort. The
    hard-coded `/usr/local/cuda` this replaces would silently ratify a different
    assembler than the build uses on any host with a versioned or relocated toolkit.
    """
    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except ImportError:
        CUDA_HOME = None
    home = CUDA_HOME or os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    return str(Path(home) / "bin" / tool)


PROBE = Path(__file__).with_name("probe_launch_bounds.cu")
MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"
SRC = Path(__file__).resolve().parents[1] / "csrc" / "rola" / "src"

#: The extension's own `-gencode` targets (`setup.py`'s `ROLA_CUDA_ARCHS`, default
#: `80;86`), overridable per run with `--arch`. BOTH are probed by default, because
#: both are in the shipped fat binary and the two allocators reach different
#: answers: sm_80 admits more CTAs by shared memory, so its bound is a different
#: number and its spill profile is a different measurement.
DEFAULT_ARCHES = ("80", "86")


def manifest_path(arch: str) -> Path:
    """ONE FILE PER ARCH (§3.7).

    *"Does arch X have a ratified manifest?"* is then a FILE-EXISTENCE question --
    which is what `setup.py`'s rule-3 gate needs to ask before it compiles anything,
    and what makes a new-architecture bring-up a single reviewable file in a PR
    rather than a 1024-line diff nobody can read. The union of the files is the
    ratification; `load_manifest` is the only thing that forms it.
    """
    return MANIFEST_DIR / f"sm_{arch}.json"


def load_manifest(arches) -> dict:
    """The per-arch files, read as ONE ratification record.

    Every file must pin the SAME assembler. A manifest whose halves were ratified
    under different `ptxas` builds is not one ratification, and silently merging
    them would produce a record that describes no binary that was ever built.
    """
    toolchain = None
    entries: dict[str, dict] = {}
    for arch in arches:
        path = manifest_path(arch)
        if not path.exists():
            raise SystemExit(
                f"no ratified manifest for sm_{arch} at {path}; run\n"
                f"    python tools/ratify.py --arch {arch} --write\n"
                f"and put the numbers in the findings document")
        blob = json.loads(path.read_text())
        if toolchain is None:
            toolchain = blob["toolchain"]
        elif blob["toolchain"] != toolchain:
            raise SystemExit(
                f"{path.name} pins a different assembler than the other manifests:\n"
                f"    {blob['toolchain']['ptxas']}\n"
                f"    {toolchain['ptxas']}\n"
                "The per-arch SPLIT is a filing decision; it may not become a split "
                "TOOLCHAIN. Re-ratify every arch under one assembler.")
        for key, value in blob["entries"].items():
            if key in entries:
                raise SystemExit(f"{key} is ratified in two files")
            entries[key] = value
    return {"toolchain": toolchain, "entries": entries}


def manifest_digest(manifest: dict) -> str:
    """The digest `setup.py` stamps into `_build_config.py`.

    Canonical JSON of `{toolchain, entries}` with sorted keys -- NOT the bytes of
    any file -- so the digest is a property of the RATIFICATION and not of its
    filing. That is what makes §3.7's split provably non-semantic: the same 1024
    measurements in two files hash to what they hashed to in one.
    """
    blob = json.dumps({"toolchain": manifest["toolchain"], "entries": manifest["entries"]},
                      sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()

ENTRY = re.compile(r"Compiling entry function '([^']+)' for 'sm_(\d+)'")
SPILL = re.compile(r"(\d+) bytes spill stores, (\d+) bytes spill loads")
REGS = re.compile(r"Used (\d+) registers")
OUT_OF_RANGE = re.compile(r"Value of threads per SM for entry (\S+) is out of range")


def _nvcc_command(arches) -> list[str]:
    """The extension's own compile flags, taken from torch rather than restated.

    Restating them would let this gate compile the probe under different flags than
    the shipped kernel is built with, and a register/spill profile is a property of
    the flags as much as of the source.
    """
    import torch.utils.cpp_extension as ext

    cmd = [cuda_bin("nvcc"), "-c", str(PROBE), "-o", "/dev/null",
           f"-I{SRC}"]
    for inc in ext.include_paths(device_type="cuda"):
        cmd += ["-isystem", inc]
    cmd += ["-isystem", sysconfig.get_paths()["include"]]
    cmd += ["-D_GLIBCXX_USE_CXX11_ABI=0", "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__", "--expt-relaxed-constexpr"]
    for a in arches:
        cmd.append(f"-gencode=arch=compute_{a},code=sm_{a}")
    cmd += ["-O3", "-std=c++17", "-lineinfo", "--threads=2", "-Xptxas", "-v"]
    return cmd


def ptxas_version() -> str:
    """`ptxas --version`, whitespace-normalized to one line.

    The WHOLE output, not a parsed release number: the build string
    (`cuda_12.4.r12.4/compiler.33961263_0`) distinguishes two shipments of the same
    nominal release, and it is exactly those that can move an allocation while the
    release number says nothing changed. Normalizing whitespace makes the comparison
    insensitive to line endings and nothing else.
    """
    ptxas = cuda_bin("ptxas")
    proc = subprocess.run([ptxas, "--version"], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"`{ptxas} --version` FAILED (rc={proc.returncode}); the gate "
                         f"cannot ratify a manifest against an assembler it cannot ask")
    return " ".join(proc.stdout.split())


def check_toolchain(got: str, want: str) -> bool:
    """Rule 1: the recorded assembler and the invoked assembler must be the same one."""
    if got == want:
        print(f"toolchain OK  ptxas: {got}")
        return True
    print(f"\nFAIL  TOOLCHAIN DRIFT. This manifest's spill and register numbers were "
          f"ratified under a different assembler:\n"
          f"        manifest  {want}\n"
          f"        invoked   {got}\n"
          f"      A launch-bound manifest is a property of the assembler as much as of "
          f"the source, so these numbers say nothing about this `ptxas`. RE-RATIFICATION "
          f"IS REQUIRED: re-run with --write and record the new numbers, and the version "
          f"they were taken under, in the findings document.")
    return False


def compile_and_parse(arches) -> tuple[dict, list[str]]:
    """Compile the probe and read `ptxas`'s report. Returns (entries, unhonored)."""
    cmd = _nvcc_command(arches)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"probe compile FAILED (rc={proc.returncode})")
    print(f"probe compiled in {wall:.1f} s over {len(arches)} targets "
          f"({', '.join('sm_' + a for a in arches)})")

    entries: dict[str, dict] = {}
    unhonored = sorted(set(OUT_OF_RANGE.findall(proc.stderr)))
    key = None
    for line in proc.stderr.splitlines():
        m = ENTRY.search(line)
        if m:
            #: The MANGLED name is the key. It is exact, stable, and encodes every
            #: template argument, so an entry cannot be silently re-pointed at a
            #: different instantiation by a rename or a reordering.
            key = f"sm_{m.group(2)}:{m.group(1)}"
            entries[key] = dict(regs=0, spill_stores=0, spill_loads=0)
            continue
        if key is None:
            continue
        m = SPILL.search(line)
        if m:
            entries[key]["spill_stores"] = int(m.group(1))
            entries[key]["spill_loads"] = int(m.group(2))
        m = REGS.search(line)
        if m:
            entries[key]["regs"] = int(m.group(1))
    return entries, unhonored


def check(entries: dict, unhonored: list[str], want: dict) -> bool:
    ok = True
    if unhonored:
        ok = False
        print(f"\nFAIL  {len(unhonored)} instantiation(s) compiled with the launch "
              f"bound DISCARDED ('out of range'). `ptxas` was unconstrained there:")
        for name in unhonored[:8]:
            print(f"        {name}")
        if len(unhonored) > 8:
            print(f"        ... and {len(unhonored) - 8} more")

    missing = sorted(set(want) - set(entries))
    added = sorted(set(entries) - set(want))
    if missing or added:
        ok = False
        print(f"\nFAIL  the instantiation set moved: {len(missing)} manifest entries "
              f"absent from the build, {len(added)} new entries not in the manifest. "
              f"Re-ratify with --write once the matrix change is intended.")
        for name in (missing[:4] + added[:4]):
            print(f"        {name}")

    grew, drifted = [], []
    for name, got in sorted(entries.items()):
        exp = want.get(name)
        if exp is None:
            continue
        if (got["spill_stores"] > exp["spill_stores"]
                or got["spill_loads"] > exp["spill_loads"]):
            grew.append((name, exp, got))
        elif got["regs"] != exp["regs"]:
            drifted.append((name, exp, got))
    if grew:
        ok = False
        print(f"\nFAIL  {len(grew)} instantiation(s) SPILL MORE than the manifest:")
        for name, exp, got in grew:
            print(f"        {name}\n"
                  f"          manifest {exp['spill_stores']} st / {exp['spill_loads']} ld"
                  f"   realized {got['spill_stores']} st / {got['spill_loads']} ld")
    if drifted:
        print(f"\nREPORT  {len(drifted)} instantiation(s) changed register count with "
              f"NO spill growth (an allocator outcome, not a regression):")
        for name, exp, got in drifted[:10]:
            print(f"        {exp['regs']} -> {got['regs']}  {name}")
        if len(drifted) > 10:
            print(f"        ... and {len(drifted) - 10} more")

    spillers = [(n, e) for n, e in sorted(entries.items()) if e["spill_stores"]
                or e["spill_loads"]]
    print(f"\n{len(entries)} instantiations probed, {len(unhonored)} with an unhonored "
          f"bound, {len(spillers)} that spill at all:")
    for name, e in spillers:
        print(f"    {e['spill_stores']:4d} st / {e['spill_loads']:4d} ld  "
              f"{e['regs']:3d} regs  {name}")
    print(f"\n{'PASS' if ok else 'FAIL'}  launch-bound spill manifest")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="RE-RATIFY: record what this compile reports as the manifest")
    ap.add_argument("--self-test", action="store_true",
                    help="prove the gate can fail, by re-checking against a "
                         "deliberately tightened manifest")
    ap.add_argument("--arch", action="append", metavar="XX",
                    help="a compute capability to probe and ratify, repeatable "
                         f"(default: {' '.join(DEFAULT_ARCHES)}). One manifest file "
                         "per arch, so bringing up a new one touches no other arch's "
                         "measurements.")
    a = ap.parse_args()

    arches = tuple(a.arch) if a.arch else DEFAULT_ARCHES

    version = ptxas_version()
    entries, unhonored = compile_and_parse(arches)
    if a.write:
        #: WRITTEN PER ARCH, from the one compile. Splitting here rather than at
        #: read time is what keeps `sm_86.json` untouched by an `--arch 80` run:
        #: a bring-up must not restate measurements it did not take.
        for arch in arches:
            prefix = f"sm_{arch}:"
            mine = {k: v for k, v in sorted(entries.items()) if k.startswith(prefix)}
            if not mine:
                raise SystemExit(
                    f"the probe reported NO entries for sm_{arch}; refusing to write "
                    f"an empty ratification. A manifest that exists but measures "
                    f"nothing would pass rule 3 by file existence alone.")
            path = manifest_path(arch)
            path.write_text(json.dumps(
                {"toolchain": {"ptxas": version}, "entries": mine}, indent=1) + "\n")
            print(f"wrote {path} ({len(mine)} instantiations) under ptxas: {version}")
        print(f"{len(unhonored)} unhonored bounds across {len(arches)} arch(es)")
        return 0
    manifest = load_manifest(arches)
    want = manifest["entries"]
    print(f"manifest sha256 {manifest_digest(manifest)}  "
          f"({len(want)} entries over {', '.join('sm_' + x for x in arches)})")
    #: THE TOOLCHAIN CHECK RUNS FIRST AND ITS RESULT IS ANDed IN, rather than
    #: short-circuiting: on a version bump the entry-level report is still the most
    #: useful thing to print (it shows WHAT the new assembler did), and printing it
    #: costs nothing once the compile has already happened. What it must not do is
    #: turn a green entry comparison into a green gate.
    ok = check_toolchain(version, manifest["toolchain"]["ptxas"])
    ok = check(entries, unhonored, want) and ok

    if a.self_test:
        #: NON-VACUITY. Tighten one entry that actually spills (or, if nothing
        #: spills, any entry) and require the gate to FAIL on it. A gate whose green
        #: run is unfalsifiable says nothing, and this batch has already recorded two
        #: gates that failed themselves as vacuous before they were fixed.
        target = next((n for n, e in sorted(want.items())
                       if e["spill_stores"] or e["spill_loads"]), None)
        target = target or sorted(want)[0]
        tightened = json.loads(json.dumps(want))
        tightened[target]["spill_stores"] = -1
        tightened[target]["spill_loads"] = -1
        print(f"\n--- SELF-TEST: the same result against a manifest tightened at\n"
              f"    {target}\n    (this run MUST fail) ---")
        if check(entries, unhonored, tightened):
            print("\nSELF-TEST FAILED: the gate PASSED a manifest it must reject")
            return 1
        print("\nSELF-TEST PASSED: the gate rejects a spill it was not ratified for")

        #: NON-VACUITY OF RULE 1, the same way. The version string is perturbed rather
        #: than a second `ptxas` installed -- the property under test is the COMPARISON,
        #: and a comparison that cannot fail on a string it was not given is exactly the
        #: vacuous gate this discipline exists to catch.
        perturbed = manifest["toolchain"]["ptxas"] + " (perturbed by --self-test)"
        print("\n--- SELF-TEST: the real ptxas version against a perturbed manifest\n"
              "    entry (this run MUST fail) ---")
        if check_toolchain(version, perturbed):
            print("\nSELF-TEST FAILED: the gate ACCEPTED a toolchain it was not "
                  "ratified under")
            return 1
        print("\nSELF-TEST PASSED: the gate rejects an assembler it was not ratified "
              "under")

        #: NON-VACUITY OF THE PER-ARCH SPLIT (§3.7). Rule 3 is now a FILE-EXISTENCE
        #: question, and a question whose answer is always yes is not a gate: an arch
        #: nobody ratified must RAISE, not merge into an empty entry set that then
        #: passes every comparison below it vacuously.
        print("\n--- SELF-TEST: an arch with no committed manifest\n"
              "    (this lookup MUST fail) ---")
        try:
            load_manifest(("999",))
        except SystemExit as exc:
            print(f"SELF-TEST PASSED: {str(exc).splitlines()[0]}")
        else:
            print("\nSELF-TEST FAILED: an UNRATIFIED arch loaded as though it were "
                  "ratified; rule 3 would pass on a manifest that does not exist")
            return 1

        #: NON-VACUITY OF THE SPLIT'S CLAIM TO BE NON-SEMANTIC. The digest is over
        #: the merged canonical record, so two files must hash to what one file hashed
        #: to. If the filing could move the digest, `_build_config.py`'s
        #: `manifest_sha256` would stop identifying the ratification and start
        #: identifying a directory listing.
        halves = [load_manifest((x,)) for x in arches]
        rejoined = {"toolchain": halves[0]["toolchain"],
                    "entries": {k: v for h in halves for k, v in h["entries"].items()}}
        if manifest_digest(rejoined) != manifest_digest(manifest):
            print("\nSELF-TEST FAILED: the per-arch files do not rejoin to the same "
                  "record; the split is not filing-only")
            return 1
        print(f"SELF-TEST PASSED: {len(arches)} per-arch files rejoin to one digest "
              f"{manifest_digest(rejoined)[:16]}...")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
