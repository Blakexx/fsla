"""SPILL-MANIFEST GATE for the consumer kernel's launch bounds.

**WHAT THIS CONVERTS, AND WHY.** the consumer rewrite's §2.21 ruling landed a
three-term residency formula and ACCEPTED one 4-byte spill on one of 128
instantiations as the measured optimum -- the alternatives spill worse (the unbounded
status quo: 20 st / 56 ld; the thread-slot cap: 70 st / 284 ld). That acceptance was a
judgment recorded in a document, and it lived nowhere else: **nothing failed if a
later codegen change grew it.** This file makes it an ASSERTED INVARIANT, the same
doctrine as the dense-degeneracy bit-identity gate -- a property the batch believes is
a property the batch checks.

SIX OUTCOMES, and the difference between them is deliberate:

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
* **FAIL -- source drift (session 9).** The tree's `csrc/rola/src` digest differs
  from the one the manifest was ratified against. The manifest's numbers -- and the
  `consumer_reg_budget` derivation they justify -- describe a kernel that no longer
  exists; re-ratify with `--write`. This is the structural fix for the
  stale-constant trap (the budget's justification table went stale by 10x across
  K0-K4 and nothing failed).
* **FAIL -- hot-loop growth (session 9).** Any instantiation executing MORE
  `LDL`/`STL` inside a loop (backward-`BRA` interval in the SASS) than its ratified
  `hot_locals` count. In-loop spill is a memory round-trip per iteration and is
  disqualifying whatever the byte count; the count is a RATCHET because the decay
  arms carry structural dynamically-indexed local-array traffic in loops that no
  budget removes (measured at 255 registers).
* **REPORT -- register drift.** A changed `Used N registers` with no spill growth is
  printed, not failed. The register count is an ALLOCATOR OUTCOME that legitimately
  moves when the kernel changes; what must not move is the spilling.

THE FORCING CENSUS (`--census [--force-ctas N] [--json PATH]`) is the measurement
mode the register rung is derived from: one probe compile over the whole 512-entry
dispatch matrix at a candidate launch bound, reported per structural key. It never
touches a manifest, and `--force-ctas` outside `--census` is refused -- a forced
bound must never be ratified.

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

`--self-test` requires the gate to FAIL on the things it must reject: a deliberately
tightened manifest, a perturbed `ptxas` version, an arch with no committed manifest, a
per-arch split that does not rejoin to one digest, a perturbed source digest, and
in-loop local-access growth past a ratified count (plus the hot-loop parser proving it
counts an in-loop `LDL` and clears a cold one on synthetic SASS). That is what makes a
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
#: The vendored CuTe/CUTLASS headers, in the SAME order `setup.py` passes them
#: (repository headers first). This gate's whole premise is that it compiles the
#: probe under the shipped kernel's flags; an include path the extension has and the
#: probe does not is that premise failing silently.
CUTLASS_INC = Path(__file__).resolve().parents[1] / "csrc" / "third_party" / "cutlass" / "include"

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
    sources: dict[str, dict | None] = {}
    for arch in arches:
        path = manifest_path(arch)
        if not path.exists():
            raise SystemExit(
                f"no ratified manifest for sm_{arch} at {path}; run\n"
                f"    python tools/ratify.py --arch {arch} --write\n"
                f"and put the numbers in the findings document")
        blob = json.loads(path.read_text())
        sources[arch] = blob.get("source")
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
    return {"toolchain": toolchain, "entries": entries, "sources": sources}


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

#: The consumer kernel's mangled template arguments, in declaration order:
#: `<int BT, int BC, int DV, int D, MmaPrec FOLD, MmaPrec READ, bool DECAY,
#:  bool GLOBAL, bool COMPACT>`. The mangled name is the manifest key, so this is
#: the ONE place a key is turned back into a shape; the census's per-key grouping
#: and the hot-loop gate's reporting both go through it.
CONSUMER_TPL = re.compile(
    r"consumer_forward_kernelILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E"
    r"LNS_7MmaPrecE(\d)ELS\d?_(\d)ELb(\d)ELb(\d)ELb(\d)E")


def parse_shape(mangled: str) -> dict | None:
    """Shape axes off a consumer instantiation's mangled name, or None."""
    m = CONSUMER_TPL.search(mangled)
    if not m:
        return None
    #: `readout_mma.cuh`: `enum class MmaPrec { kTf32x3 = 0, kBf16 = 1 }`.
    prec = {"0": "tf32x3", "1": "bf16"}
    return dict(BT=int(m.group(1)), BC=int(m.group(2)), DV=int(m.group(3)),
                D=int(m.group(4)), fold=prec[m.group(5)], read=prec[m.group(6)],
                decay=m.group(7) == "1", glob=m.group(8) == "1",
                compact=m.group(9) == "1")


#: The calibration macros a shipped binary compiles under (coverage-closure audit
#: item 10). Neither `setup.py` nor this probe passes a `-D` for them, so the
#: effective value is the header default; recording it in the manifest's toolchain
#: blob is what makes a silent default flip (or a build that DID pass a `-D`) a
#: checkable drift instead of an unverifiable claim, and it is what the ring-depth
#: agreement gate (`tests/unit/test_ring_depth_agreement.py`) reads as the
#: compiled truth. The values are read off the SAME digested source the manifest
#: describes -- the one place they are defined.
RATIFIED_DEFINES = ("ROLA_C_RING_DEPTH", "ROLA_C_WT_WINDOW")


def shipped_defines() -> dict[str, int]:
    """The `#define` defaults of :data:`RATIFIED_DEFINES` in the digested source."""
    src = (SRC / "consumer_kernel.cuh").read_text()
    out = {}
    for name in RATIFIED_DEFINES:
        m = re.search(rf"^#define {name} (\d+)$", src, re.MULTILINE)
        if m is None:
            raise SystemExit(f"consumer_kernel.cuh no longer defines a default for "
                             f"{name}; the provenance record cannot be produced")
        out[name] = int(m.group(1))
    return out


def check_defines(manifest: dict) -> bool:
    """The recorded calibration macros and the tree's defaults must be the same.

    A missing block is a failure too: a manifest ratified before audit item 10
    landed cannot answer what the binary compiled, which is the gap the record
    exists to close -- `--write` on the current tree produces it.
    """
    want = shipped_defines()
    got = manifest["toolchain"].get("defines")
    if got == want:
        print(f"defines OK  {want}")
        return True
    print(f"\nFAIL  DEFINE DRIFT. The manifest records calibration macros {got} but "
          f"the tree's shipped defaults are {want}. The ratified numbers were "
          f"measured under macros the tree no longer compiles (or were never "
          f"recorded at all); re-ratify with --write.")
    return False


def csrc_digest() -> str:
    """A digest of the kernel sources the manifest's numbers were measured against.

    THE FRESHNESS TRIGGER (session 9, R1). `consumer_reg_budget` is a measured
    constant; its justification EXPIRES when `csrc/` moves (the shipped 96 was
    derived pre-K3/K4 and went stale by 10x -- D8-Q1). Recording the source digest
    in the manifest and CHECKING it makes that trap structural: a ratification
    whose sources have materially changed is a hard failure demanding `--write`,
    which is a fresh measurement, never an edit.
    """
    h = hashlib.sha256()
    for path in sorted(SRC.rglob("*")):
        if path.is_file():
            h.update(str(path.relative_to(SRC)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def _nvcc_command(arches) -> list[str]:
    """The extension's own compile flags, taken from torch rather than restated.

    Restating them would let this gate compile the probe under different flags than
    the shipped kernel is built with, and a register/spill profile is a property of
    the flags as much as of the source.
    """
    import torch.utils.cpp_extension as ext

    cmd = [cuda_bin("nvcc"), "-c", str(PROBE),
           f"-I{SRC}", f"-I{CUTLASS_INC}"]
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


def check_freshness(manifest: dict, digest: str | None = None) -> bool:
    """THE FRESHNESS TRIGGER (session 9, R1). The manifest's numbers -- and the
    register-budget derivation they justify -- are dated against a source tree;
    when `csrc/rola/src` moves materially, the ratification is a description of a
    kernel that no longer exists and must be RE-MEASURED (`--write`), never
    trusted. This is the structural fix for the stale-constant trap that blocked
    K5: `consumer_reg_budget`'s justification cannot silently outlive the working
    set it was measured on, because the gate that every build runs re-checks the
    tree the measurement was taken from."""
    digest = digest if digest is not None else csrc_digest()
    ok = True
    for arch, src in sorted(manifest["sources"].items()):
        if src is None:
            print(f"\nFAIL  sm_{arch}'s manifest predates the source-freshness "
                  f"trigger (no `source` block). Re-ratify: "
                  f"python tools/ratify.py --arch {arch} --write")
            ok = False
        elif src["csrc_sha256"] != digest:
            print(f"\nFAIL  SOURCE DRIFT. sm_{arch}'s manifest was ratified against "
                  f"csrc {src['csrc_sha256'][:16]}... on {src.get('date', '?')}; "
                  f"the tree is now {digest[:16]}... . The register/spill numbers "
                  f"and the budget derivation they justify describe a kernel that "
                  f"no longer exists. Re-ratify with --write and record the new "
                  f"numbers in the findings document.")
            ok = False
    if ok:
        print(f"source freshness OK  csrc sha256 {digest[:16]}... "
              f"({len(manifest['sources'])} arch manifest(s))")
    return ok


def compile_and_parse(arches, obj_path: str = "/dev/null",
                      defines: tuple[str, ...] = ()) -> tuple[dict, list[str]]:
    """Compile the probe and read `ptxas`'s report. Returns (entries, unhonored).

    `obj_path` keeps the object file (the hot-loop gate disassembles the SASS out
    of the SAME compile the numbers came from, so the two can never describe
    different binaries); `defines` is the census mode's `-DROLA_FORCE_CTAS=N`.
    """
    cmd = _nvcc_command(arches) + ["-o", obj_path] + [f"-D{d}" for d in defines]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"probe compile FAILED (rc={proc.returncode})")
    print(f"probe compiled in {wall:.1f} s over {len(arches)} targets "
          f"({', '.join('sm_' + a for a in arches)})"
          + (f" with {' '.join(defines)}" if defines else ""))

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


#: --------------------------------------------------------------------------
#: THE HOT-LOOP LOCAL-ACCESS GATE (session 9, R1 enforcement).
#:
#: A spill's COST is where it lands, not how much of it there is: the `base[D]`
#: prologue array spilling once per CTA is free, the same bytes inside the
#: 2,000-3,000-instruction tile loop are a memory round-trip per iteration. The
#: register-rung scoping (REGISTER_RUNG_SCOPE.md §2.3) mechanized that as "no
#: `LDL`/`STL` at loop depth >= 1", and this gate is that test as a BUILD FAILURE.
#:
#: **IT IS A RATCHET, NOT AN ABSOLUTE, and the difference was MEASURED before the
#: gate landed (session 9).** The decay arms execute dynamically-indexed LOCAL
#: ARRAYS (`LDL.64 R28, [R26]` -- computed addresses, `ptxas` spill bytes ZERO)
#: inside their loops; that is §2.23's "local ARRAYS, not pressure" class,
#: structural to the arm's indexing and unfixable by any register budget (three
#: keys still carry it at 255 registers). An absolute in-loop ban would therefore
#: fail every baseline forever on traffic that is not the defect. At SASS level
#: array traffic and pressure spill are not separable by address form (both reach
#: `[R1+imm]`), so the gate pins the MEASURED per-instantiation in-loop count in
#: the manifest and fails on GROWTH: a budget change that pushes pressure spill
#: into a loop adds in-loop `LDL`/`STL` and is refused; the structural array
#: traffic is ratified as what it is. (R1's own key changes are held to the
#: stricter reading -- no increase at all on a changed key -- by the same number.)
#:
#: Loop membership is computed from the SASS itself: an instruction is "in a loop"
#: iff it lies inside the interval of a BACKWARD branch (a `BRA` whose target
#: precedes it in the function). That is the same reading the scoping session used.

_SASS_FUNC = re.compile(r"^\s*Function : (\S+)")
_SASS_ARCH = re.compile(r"^\s*arch = sm_(\d+)\s*$")
_SASS_INST = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?);")
_SASS_LABEL_DEF = re.compile(r"^\s*(\.L_[\w.]+):")
_SASS_LOCAL = re.compile(r"\b(LDL|STL)\b")
_SASS_BRA = re.compile(r"\bBRA\b.*?(?:`?\((\.L_[\w.]+)\)|0x([0-9a-f]+))")


def _hot_locals_in_function(items: list) -> list:
    """`items` = [(pos, kind, payload)] for one function, in order; kinds are
    'label' (payload: name), 'inst' (payload: (addr, text)). Returns the hot
    local accesses: [(addr, opcode_text)] that sit inside a backward-BRA interval."""
    labels: dict[str, int] = {}
    addr_pos: dict[int, int] = {}
    for pos, kind, payload in items:
        if kind == "label":
            labels[payload] = pos
        else:
            addr_pos.setdefault(payload[0], pos)
    intervals = []
    for pos, kind, payload in items:
        if kind != "inst":
            continue
        addr, text = payload
        m = _SASS_BRA.search(text)
        if not m:
            continue
        target = labels.get(m.group(1)) if m.group(1) else \
            addr_pos.get(int(m.group(2), 16))
        if target is not None and target < pos:
            intervals.append((target, pos))
    if not intervals:
        return []
    hot = []
    for pos, kind, payload in items:
        if kind != "inst":
            continue
        addr, text = payload
        if _SASS_LOCAL.search(text) and any(a <= pos <= b for a, b in intervals):
            hot.append((addr, text.strip()))
    return hot


def hot_loop_report(sass_lines, wanted: set) -> dict:
    """Stream `cuobjdump -sass` output; return {key: [(addr, inst), ...]} for every
    `wanted` instantiation, listing its local accesses inside loops (possibly [])."""
    report: dict[str, list] = {}
    arch, key, items, pos = None, None, [], 0

    def flush():
        if key in wanted and items:
            report[key] = _hot_locals_in_function(items)

    for line in sass_lines:
        m = _SASS_ARCH.match(line)
        if m:
            flush()
            arch, key, items = f"sm_{m.group(1)}", None, []
            continue
        m = _SASS_FUNC.match(line)
        if m:
            flush()
            key, items, pos = f"{arch}:{m.group(1)}", [], 0
            continue
        if key is None:
            continue
        m = _SASS_LABEL_DEF.match(line)
        if m:
            items.append((pos, "label", m.group(1)))
            pos += 1
            continue
        m = _SASS_INST.search(line)
        if m:
            items.append((pos, "inst", (int(m.group(1), 16), m.group(2))))
            pos += 1
    flush()
    return report


def measure_hot_locals(obj_path: str, entries: dict) -> dict[str, int]:
    """Disassemble the SAME object the register numbers came from and return the
    per-instantiation count of local accesses inside loops."""
    proc = subprocess.Popen([cuda_bin("cuobjdump"), "-sass", obj_path],
                            stdout=subprocess.PIPE, text=True)
    report = hot_loop_report(proc.stdout, set(entries))
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"cuobjdump -sass FAILED (rc={rc}); the hot-loop gate "
                         f"cannot pass a binary it cannot read")
    return {name: len(report.get(name, [])) for name in entries}


def check_hot_loop(hot: dict[str, int], want: dict) -> bool:
    """The gate proper: any instantiation whose in-loop local-access count GREW
    past its ratified count is a build failure (the ratchet)."""
    grew = [(n, want[n].get("hot_locals", 0), c) for n, c in sorted(hot.items())
            if n in want and c > want[n].get("hot_locals", 0)]
    if grew:
        print(f"\nFAIL  HOT-LOOP GROWTH: {len(grew)} instantiation(s) execute MORE "
              f"local accesses INSIDE a loop than were ratified. In-loop spill is "
              f"a memory round-trip per iteration and is disqualifying whatever "
              f"the byte count; lower the pressure or raise that key's budget:")
        for name, exp, got in grew[:10]:
            print(f"        ratified {exp} -> realized {got}  {name}")
        if len(grew) > 10:
            print(f"        ... and {len(grew) - 10} more")
        return False
    total = sum(hot.values())
    print(f"hot-loop gate: in-loop local accesses within ratified counts on all "
          f"{len(hot)} entries (tree total {total}); PASS")
    return True


#: --------------------------------------------------------------------------
#: THE FORCING CENSUS (session 9, R0) -- the permanent instrument the register
#: rung is derived from. `--census` compiles the whole 512-entry probe at a
#: candidate bound (`--force-ctas N` -> `-DROLA_FORCE_CTAS=N`; omitted -> the
#: shipped derived bound) and prints the per-key register/spill table. It never
#: touches a manifest: a census is a measurement of a CANDIDATE, ratification is
#: the recording of the SHIPPED bound, and conflating them is how a stale budget
#: constant survived three kernel rungs.

def _regs_ctas(regs: int) -> int:
    """Register-file-side CTAs/SM at 256 threads after sm_86's 8-register
    allocation granularity. The REGISTER term only -- the driver
    (`rola_consumer_occupancy`) stays the sole residency authority."""
    alloc = 256 * ((regs + 7) // 8) * 8
    return min(6, 65536 // alloc) if alloc else 6


def census_report(entries: dict, unhonored: list[str],
                  hot: dict[str, int] | None = None) -> None:
    hot = hot or {}
    by_key: dict[tuple, list] = {}
    for name, e in sorted(entries.items()):
        arch = name.split(":", 1)[0]
        shape = parse_shape(name)
        if shape is None:
            continue
        k = (arch, shape["BT"], shape["BC"], shape["DV"], shape["read"],
             shape["decay"])
        by_key.setdefault(k, []).append((e, hot.get(name, 0)))
    print(f"\n{'arch':6} {'BT':>3} {'BC':>3} {'DV':>3} {'prec':7} {'decay':5} "
          f"{'n':>3} {'regs':>9} {'ctas(reg)':>9} {'st':>6} {'ld':>6} "
          f"{'#spill':>6} {'worst st':>8} {'hot':>5}")
    for k, es in sorted(by_key.items()):
        regs = sorted(e["regs"] for e, _ in es)
        ctas = sorted(_regs_ctas(r) for r in regs)
        st = sum(e["spill_stores"] for e, _ in es)
        ld = sum(e["spill_loads"] for e, _ in es)
        nsp = sum(1 for e, _ in es if e["spill_stores"] or e["spill_loads"])
        worst = max(e["spill_stores"] for e, _ in es)
        nhot = sum(h for _, h in es)
        rg = f"{regs[0]}-{regs[-1]}" if regs[0] != regs[-1] else f"{regs[0]}"
        cg = f"{ctas[0]}-{ctas[-1]}" if ctas[0] != ctas[-1] else f"{ctas[0]}"
        print(f"{k[0]:6} {k[1]:>3} {k[2]:>3} {k[3]:>3} {k[4]:7} "
              f"{str(k[5]).lower():5} {len(es):>3} {rg:>9} {cg:>9} {st:>6} "
              f"{ld:>6} {nsp:>6} {worst:>8} {nhot:>5}")
    total_st = sum(e["spill_stores"] for e in entries.values())
    total_ld = sum(e["spill_loads"] for e in entries.values())
    zero = sum(1 for e in entries.values()
               if not e["spill_stores"] and not e["spill_loads"])
    print(f"\nTOTAL {len(entries)} entries: {total_st} spill st / {total_ld} "
          f"spill ld, {zero} zero-spill, register sum "
          f"{sum(e['regs'] for e in entries.values())}, "
          f"{sum(hot.values())} in-loop local accesses, "
          f"{len(unhonored)} unhonored bound(s)")


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
    ap.add_argument("--census", action="store_true",
                    help="FORCING CENSUS: compile the whole probe matrix, print the "
                         "per-key register/spill table, touch no manifest")
    ap.add_argument("--force-ctas", type=int, metavar="N",
                    help="census only: compile with -DROLA_FORCE_CTAS=N, i.e. force "
                         "every instantiation's launch bound to N CTAs/SM")
    ap.add_argument("--json", metavar="PATH",
                    help="census only: dump the raw per-instantiation record for "
                         "arm-vs-arm diffing")
    a = ap.parse_args()

    arches = tuple(a.arch) if a.arch else DEFAULT_ARCHES
    if a.force_ctas is not None and not a.census:
        raise SystemExit("--force-ctas is a CENSUS instrument; a forced bound must "
                         "never be ratified or checked against a manifest "
                         "(use --census)")

    version = ptxas_version()
    import tempfile
    with tempfile.TemporaryDirectory(prefix="rola_ratify_") as tmp:
        obj = str(Path(tmp) / "probe.o")
        defines = (f"ROLA_FORCE_CTAS={a.force_ctas}",) if a.force_ctas is not None \
            else ()
        entries, unhonored = compile_and_parse(arches, obj_path=obj,
                                               defines=defines)
        hot = measure_hot_locals(obj, entries)
        for name, count in hot.items():
            entries[name]["hot_locals"] = count
        if a.census:
            census_report(entries, unhonored, hot)
            if a.json:
                Path(a.json).write_text(json.dumps(
                    dict(ptxas=version, force_ctas=a.force_ctas,
                         csrc_sha256=csrc_digest(), entries=entries),
                    indent=1, sort_keys=True) + "\n")
                print(f"census written to {a.json}")
            return 0
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
            #: The SOURCE DIGEST is recorded beside the toolchain: a manifest is a
            #: measurement OF a source tree UNDER an assembler, and both halves of
            #: that sentence must be checkable (the freshness trigger, session 9).
            path.write_text(json.dumps(
                {"toolchain": {"ptxas": version, "defines": shipped_defines()},
                 "source": {"csrc_sha256": csrc_digest(),
                            "date": time.strftime("%Y-%m-%d")},
                 "entries": mine}, indent=1) + "\n")
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
    ok = check_defines(manifest) and ok
    ok = check_freshness(manifest) and ok
    ok = check(entries, unhonored, want) and ok
    ok = check_hot_loop(hot, want) and ok

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

        #: NON-VACUITY OF THE FRESHNESS TRIGGER. A perturbed source digest must
        #: fail: a freshness gate that passes a tree it was not measured on is the
        #: stale-constant trap re-armed.
        print("\n--- SELF-TEST: the manifest against a perturbed source digest\n"
              "    (this run MUST fail) ---")
        if check_freshness(manifest, digest="0" * 64):
            print("\nSELF-TEST FAILED: the gate ACCEPTED a source tree the "
                  "manifest was not measured on")
            return 1
        print("SELF-TEST PASSED: the gate rejects a moved source tree")

        #: NON-VACUITY OF THE HOT-LOOP GATE, against synthetic SASS: one LDL
        #: inside a backward-branch interval MUST be flagged, and the same LDL
        #: outside every loop MUST NOT be.
        hot_sass = [
            "arch = sm_86",
            "                Function : k_hot",
            "        /*0000*/ MOV R1, c[0x0][0x28] ;",
            ".L_x_0:",
            "        /*0010*/ LDL R2, [R1] ;",
            "        /*0020*/ @P0 BRA `(.L_x_0) ;",
            "                Function : k_cold",
            "        /*0000*/ LDL R2, [R1] ;",
            ".L_x_1:",
            "        /*0010*/ IADD3 R3, R3, 1, RZ ;",
            "        /*0020*/ @P0 BRA `(.L_x_1) ;",
        ]
        rep = hot_loop_report(iter(hot_sass), {"sm_86:k_hot", "sm_86:k_cold"})
        print("\n--- SELF-TEST: the hot-loop parser on synthetic SASS ---")
        counts = {k: len(v) for k, v in rep.items()}
        if counts != {"sm_86:k_hot": 1, "sm_86:k_cold": 0}:
            print(f"\nSELF-TEST FAILED: hot-loop parser reported {counts}; it "
                  f"must count exactly the in-loop LDL (k_hot=1, k_cold=0)")
            return 1
        print("SELF-TEST PASSED: the parser counts the in-loop LDL and clears the "
              "cold one")
        #: ... and the RATCHET must fail on growth past a ratified count.
        print("\n--- SELF-TEST: hot-loop growth against a ratified count of 0\n"
              "    (this run MUST fail) ---")
        if check_hot_loop({"sm_86:k_hot": 1}, {"sm_86:k_hot": {"hot_locals": 0}}):
            print("\nSELF-TEST FAILED: the ratchet PASSED an in-loop count it was "
                  "not ratified for")
            return 1
        print("SELF-TEST PASSED: the ratchet rejects in-loop growth")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
