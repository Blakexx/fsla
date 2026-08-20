"""SPILL-MANIFEST GATE for the extension's launch bounds.

**WHAT THIS CONVERTS, AND WHY.** A measured spill profile accepted as an optimum is
a judgment recorded in a document, and a judgment that lives nowhere else means
**nothing fails if a later codegen change grows it.** This file makes it an ASSERTED
INVARIANT, on the same doctrine every other gate in this tree runs on -- a property
the batch believes is a property the batch checks. The profile is a RATCHET per
instantiation, not a spill ban: the gate holds each entry AT its ratified bytes.

The gate measures the shipped translation units together: the chunk consumer's
instantiation shards, the DECODE GEMV and the two UNION-ENTMAX kernels, so every
`__launch_bounds__` in the extension is read off `ptxas -v` instead of asserted in a
comment.

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
  registers as it liked. It is a hard failure even at zero spills.
* **FAIL -- source drift (session 9).** The tree's `csrc/rola/src` digest differs
  from the one the manifest was ratified against. The manifest's numbers -- and the
  register-budget derivation they justify -- describe a kernel that no longer
  exists; re-ratify with `--write`. This is the structural fix for the
  stale-constant trap: a measured constant's justification must not outlive the
  working set it was measured on.
* **FAIL -- hot-loop growth (session 9).** Any instantiation executing MORE
  `LDL`/`STL` inside a loop (backward-`BRA` interval in the SASS) than its ratified
  `hot_locals` count. In-loop spill is a memory round-trip per iteration and is
  disqualifying whatever the byte count; the count is a RATCHET rather than a ban so
  that structural dynamically-indexed local-array traffic no budget removes is held
  at its measured level instead of forbidden.
* **REPORT -- register drift.** A changed `Used N registers` with no spill growth is
  printed, not failed. The register count is an ALLOCATOR OUTCOME that legitimately
  moves when the kernel changes; what must not move is the spilling.

THE CENSUS (`--census [--json PATH]`) is the measurement mode the register rung is
derived from: a compile of every built arm at the SHIPPED derived bound, reported
per structural key and per SHARD. It never touches a manifest. Measuring a
CANDIDATE bound is a scratch-tree edit of the bound itself
(`docs/measurement.md`), not a flag here -- a forced bound must never be
ratifiable, and the cheapest way to guarantee that is for the tool not to have
one.

**RE-RATIFICATION IS A MEASUREMENT, NOT AN EDIT.** The manifest is regenerated only by
`--write`, which re-runs this same compile and records what it reports. Hand-editing an
entry to make the gate pass would be asserting a spill nobody measured, which is
exactly the failure this file replaces. When a kernel change legitimately moves the
numbers: re-run `--write`, and put the new numbers in the findings document with the
reason they moved.

Usage (no GPU required -- this is a COMPILE-time gate, so it does not take the lock):

    python tools/ratify.py
    python tools/ratify.py --arch 80 --arch 86
    python tools/ratify.py --arch 89 --write               # bringing up a new arch
    python tools/ratify.py --shard chunk_s1                # triage one shard's slice

`--shard` is a MEASUREMENT scope and never a ratification scope: a ratchet refusal is
investigated one shard's slice at a time instead of by re-deriving every entry, but
`--write` refuses it, because a manifest written from a subset of the translation
units would silently drop every instantiation it did not compile.

`--self-test` requires the gate to FAIL on the things it must reject: a deliberately
tightened manifest, a perturbed `ptxas` version, an arch with no committed manifest, a
per-arch split that does not rejoin to one digest, per-arch files that disagree about
shard membership, a shard block that is decorative or that is a property of the
filing, a perturbed source digest, and in-loop local-access growth past a ratified
count (plus the hot-loop parser proving it counts an in-loop `LDL` and clears a cold
one on synthetic SASS). That is what makes a green run mean something: a gate that
cannot fail is not a gate (`rewrite.md`'s vacuity discipline, applied to itself).

MANIFEST FORMAT: `tools/manifests/sm_XX.json`, ONE FILE PER ARCH, each
`{"toolchain": {ptxas, defines, nvcc_flags, sass_hash_version, sccache}, "source":
{...}, "shards": {...},
"entries": {<mangled name>:
{...}}}`. The metadata blocks are siblings of the entries rather than reserved keys
inside them, so an instantiation can never be confused for metadata by a name that
happens to collide. `shards` is `{shard name: {members_sha256, count}}` -- MEMBERSHIP
ONLY, no paths and no ordering -- and it enters the digest, which is what makes "only
shards X and Y were re-measured" a provable statement rather than a claim. Every arch
file must agree on it: the per-arch split is a filing decision and may not become a
split MATRIX.

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
import shlex
import subprocess
import sys
import sysconfig
import time
from concurrent.futures import ThreadPoolExecutor
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


sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_flags  # noqa: E402
import gen_shards  # noqa: E402
import sccache_toolchain  # noqa: E402

#: Set once by `gate_sccache()`, read by `_nvcc_command`. `None` means "compile
#: with the real `nvcc` directly" -- the default until `gate_sccache()` runs, and
#: the permanent state if no pinned cache is available.
_NVCC_BIN: str | None = None
_SCCACHE_INFO: dict | None = None
_SCCACHE_GATED = False


def gate_sccache() -> dict | None:
    """Wire the pinned sccache wrapper in for THIS process's `nvcc` invocations.

    Same resolution `setup.py`'s `_gate_sccache` uses (`tools/sccache_toolchain.py`)
    so the two land in the same cache namespace: this is what turns the post-build
    recompile (`_post_build_manifest_check` shells out to this file) into a cache
    HIT against the build's own ninja compile, rather than a second cold compile of
    the same ten translation units. Idempotent -- `main()` calls it once; nothing
    else needs to.
    """
    global _NVCC_BIN, _SCCACHE_INFO, _SCCACHE_GATED
    if _SCCACHE_GATED:
        return _SCCACHE_INFO
    _SCCACHE_GATED = True
    info = sccache_toolchain.gate()
    if info is not None:
        real_nvcc = cuda_bin("nvcc")
        wrapper, env = sccache_toolchain.wrap(real_nvcc)
        os.environ.update(env)
        _NVCC_BIN = wrapper
        print(f"sccache: wired for ratify.py's compile ({wrapper} -> {real_nvcc})")
    _SCCACHE_INFO = info
    return info

MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"
SRC = Path(__file__).resolve().parents[1] / "csrc" / "rola" / "src"

#: THE SAME FIXED DIRECTORY `setup.py`'s `RoLABuildExtension.finalize_options`
#: pins `build_temp` to. Not a filing convenience: sccache's cache key for an
#: `nvcc` invocation is sensitive to the invoking process's CWD even when every
#: path in the command is already absolute (measured, `docs/build.md#sccache`),
#: so this gate's compiles must run from the SAME directory the build's ninja
#: compile just ran from, or the post-build recompile can never hit the build's
#: own cache entry -- and a plain `python tools/ratify.py` run by a developer
#: must use the same fixed directory too, so re-ratifying twice in a row (or
#: after a build) hits by construction rather than by the accident of an
#: unchanged shell CWD.
COMPILE_CWD = Path(__file__).resolve().parents[1] / "build" / "temp"

#: THE TRANSLATION UNITS THIS GATE COMPILES -- THE SHIPPED ONES.
#:
#: It used to compile a probe: one file under `tools/` that re-enumerated the
#: instantiation matrix and `#include`d `decode.cu` and `entmax.cu` to reach their
#: dispatch closures. That probe was a SECOND enumeration of a set the dispatch
#: already defined (the drift shape this project has been bitten by), it re-compiled
#: the whole matrix on every build cycle on top of the build's own compile of it,
#: and -- the part that was never stated -- its decode/entmax keys named symbols
#: that existed IN NO SHIPPED BINARY, because nvcc derives an anonymous-namespace
#: kernel's mangled discriminator from the file that instantiates it, and that file
#: was the probe.
#:
#: Now the gate compiles the shipped sources themselves: the generated instantiation
#: shard, plus `decode.cu` and `entmax.cu` as the translation units they are in the
#: build. The facts keys are stable by construction -- the kernels live in a NAMED
#: namespace, so their mangled names do not depend on which translation unit
#: expands them -- and the decode/entmax keys name the symbols that are actually in
#: the `.so`, which `tools/fatbin_entries.py` checks rather than asserts.
STANDALONE_TUS = ("decode/decode.cu", "entmax/entmax.cu",
                  "entmax/factor.cu",
                  #: K31 R1: the carry family is shipped codegen the moment it is in
                  #: SOURCES, so it is measured like every other shipped TU.
                  "carry/carry.cu", "intra/intra.cu")


def shard_paths() -> list[Path]:
    names = sorted(gen_shards.chunk_shards())
    return [SRC / "instantiations" / f"{name}.cu" for name in names]


def tu_paths(shards: tuple[str, ...] | None = None) -> list[Path]:
    """The translation units to compile. `shards` scopes to a subset by stem.

    Scoping is a MEASUREMENT scope, never a ratification scope: `--write` still
    refuses to file a manifest that does not carry the whole matrix, because a
    partial manifest would silently drop every arm it did not compile.
    """
    paths = shard_paths()
    if shards is not None:
        by_stem = {p.stem: p for p in paths}
        unknown = sorted(set(shards) - set(by_stem))
        if unknown:
            raise SystemExit(f"no such shard: {', '.join(unknown)}\n"
                             f"    known: {', '.join(sorted(by_stem))}")
        paths = [by_stem[s] for s in sorted(shards)]
        return paths
    return paths + [SRC / name for name in STANDALONE_TUS]
#: The vendored CuTe/CUTLASS headers, in the SAME order `setup.py` passes them
#: (repository headers first). This gate's whole premise is that it compiles the
#: shipped translation units under the shipped kernel's flags; an include path the
#: extension has and this gate does not is that premise failing silently.
CUTLASS_INC = Path(__file__).resolve().parents[1] / "csrc" / "third_party" / "cutlass" / "include"

#: The extension's own `-gencode` targets (`setup.py`'s `ROLA_CUDA_ARCHS`, default
#: `80;86`), overridable per run with `--arch`. BOTH are measured by default, because
#: both are in the shipped fat binary and the two allocators reach different
#: answers: sm_80 admits more CTAs by shared memory, so its bound is a different
#: number and its spill profile is a different measurement.
DEFAULT_ARCHES = ("80", "86")


def manifest_path(arch: str) -> Path:
    """ONE FILE PER ARCH.

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
    shards = None
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
        #: THE SHARD MEMBERSHIP is a property of the ratification, not of one arch:
        #: the same source files were compiled for every target. Two files that
        #: disagree about which instantiations exist are not one ratification, and
        #: merging them would produce a record describing no binary ever built --
        #: the same failure the split-toolchain check above refuses.
        if shards is None:
            shards = blob.get("shards")
        elif blob.get("shards") != shards:
            raise SystemExit(
                f"{path.name} records a different shard membership than the other "
                f"manifests. The per-arch SPLIT is a filing decision; it may not "
                f"become a split MATRIX. Re-ratify every arch off one tree.")
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
    return {"toolchain": toolchain, "entries": entries, "shards": shards,
            "sources": sources}


def manifest_digest(manifest: dict) -> str:
    """The digest `setup.py` stamps into `_build_config.py`.

    Canonical JSON of `{toolchain, entries, shards}` with sorted keys -- NOT the
    bytes of any file -- so the digest is a property of the RATIFICATION and not of
    its filing. That is what makes the per-arch split provably non-semantic: the same 1216
    measurements in two files hash to what they hashed to in one.

    THE SHARD BLOCK PARTICIPATES because "only shards X and Y were re-measured" is
    otherwise an unprovable claim: it is membership (`{name: {members_sha256,
    count}}`) and nothing else -- no paths, no ordering -- so it records WHAT was
    ratified without recording how it was filed.
    """
    blob = json.dumps({"toolchain": manifest["toolchain"], "entries": manifest["entries"],
                       "shards": manifest["shards"]}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()

ENTRY = re.compile(r"Compiling entry function '([^']+)' for 'sm_(\d+)'")
SPILL = re.compile(r"(\d+) bytes spill stores, (\d+) bytes spill loads")
REGS = re.compile(r"Used (\d+) registers")
OUT_OF_RANGE = re.compile(r"Value of threads per SM for entry (\S+) is out of range")

#: The FACTS kernels' mangled template arguments, in DECLARATION order.  The
#: union-table pass is templated on the PLAN TYPE since K31-i-d, so its key
#: carries the plan's `(D, B, s_0..s_3)` -- the SPANS, which is what the pass is
#: actually parameterized by -- rather than the `BC` a lex arm is named by;
#: `atom_bitmap_kernel<int D, int B>` and `block_bitmap_kernel<int D, int B, int BC>`
#: are lex-only reductions and keep their integer keys.  Decoded in ONE place: the
#: manifest key is what the assembler saw, and every consumer of it -- the census's
#: per-key grouping, the hot-loop gate's reporting, the fatbin gate's ownership
#: check -- goes through here.
FACTS_TPL = re.compile(r"(atom_bitmap_kernel|block_bitmap_kernel)"
                       r"ILi(\d+)ELi(\d+)(?:ELi(\d+))?E")
PLAN_TPL = re.compile(r"(chunk_facts_tables)I[A-Za-z0-9_]*?8SpanPlan"
                      r"ILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)EEELi(\d+)E")


def parse_shape(mangled: str) -> dict | None:
    """Shape axes off a facts instantiation's mangled name, or None."""
    m = PLAN_TPL.search(mangled)
    if m:
        D, B = int(m.group(2)), int(m.group(3))
        spans = tuple(int(m.group(4 + i)) for i in range(4))
        bc = 1
        for l in range(D):
            bc *= spans[l]
        return dict(kernel=m.group(1), D=D, B=B, spans=spans, mask=int(m.group(8)), BC=bc)
    m = FACTS_TPL.search(mangled)
    if not m:
        return None
    return dict(kernel=m.group(1), D=int(m.group(2)), B=int(m.group(3)),
                BC=int(m.group(4)) if m.group(4) else None, spans=None, mask=None)


def shard_of_key(mangled: str) -> str:
    """The translation unit that owns a ratified instantiation.

    An arm's shard is a STRUCTURAL function of its tuple, so it is recomputed from
    the mangled name rather than recorded beside it -- the name is what the assembler
    saw and cannot disagree with the binary. The decode and entmax kernels are single
    translation units and carry their file's stem, so every ratified entry answers
    the same question in the same schema.
    """
    if parse_shape(mangled) is not None:
        return next(iter(sorted(gen_shards.chunk_shards())))
    for tu in STANDALONE_TUS:
        if f"_{Path(tu).stem}_cu_" in mangled:
            return Path(tu).stem
    return "?"


def shard_field(entries: dict) -> dict[str, str]:
    """`{entry key -> shard}`, the census's per-row shard column."""
    return {name: shard_of_key(name.split(":", 1)[1]) for name in entries}


#: The calibration macros a shipped binary compiles under (coverage-closure audit
#: item 10). EMPTY since the endgame executor: the ring
#: (`ROLA_C_RING_DEPTH`) and the write-panel window (`ROLA_C_WT_WINDOW`) died
#: with the aligned walk -- the kernel has no compile-time calibration macros
#: left. The mechanism stays so the NEXT macro is a one-line addition, and a
#: manifest still records the (empty) block, keeping "what did this compile
#: under" a checkable statement rather than an unverifiable claim.
RATIFIED_DEFINES: tuple[str, ...] = ()


def shipped_defines() -> dict[str, int]:
    """The `#define` defaults of :data:`RATIFIED_DEFINES` in the digested source."""
    src = (SRC / "facts" / "facts_kernel.cuh").read_text()
    out = {}
    for name in RATIFIED_DEFINES:
        m = re.search(rf"^#define {name} (\d+)$", src, re.MULTILINE)
        if m is None:
            raise SystemExit(f"facts_kernel.cuh no longer defines a default for "
                             f"{name}; the provenance record cannot be produced")
        out[name] = int(m.group(1))
    return out


#: Flags that say HOW THE COMPILE IS RUN rather than WHAT CODE IT PRODUCES, and are
#: therefore excluded from the ratified list. `-gencode` names the targets, which the
#: per-arch FILING already records -- keeping it would make one arch's manifest
#: disagree with another's on a block every arch file must share. `--threads` is a
#: parallelism knob with an env override (`ROLA_NVCC_THREADS`), so including it would
#: make a build that spends more cores on itself read as a codegen change.
_NON_CODEGEN_FLAG_PREFIXES = ("-gencode", "--threads=")


def ratified_flags() -> list[str]:
    """The codegen flags a manifest is measured under, ARCH- AND JOB-INDEPENDENT.

    Everything `tools/build_flags.py` emits except :data:`_NON_CODEGEN_FLAG_PREFIXES`:
    the optimization level, the language standard, `-lineinfo`, the torch defines,
    `--ptxas-options=-v` and the fatbin compression. Each of those either changes what
    `ptxas` assembles or changes the report this gate parses.
    """
    flags = build_flags.torch_defines() + build_flags.nvcc_flags(())
    return [f for f in flags if not f.startswith(_NON_CODEGEN_FLAG_PREFIXES)]


def check_flags(manifest: dict) -> bool:
    """The recorded flag list and the tree's flag list must be the same.

    THE GAP THIS CLOSES, recorded when `-Xfatbin -compress-all` landed: the manifest
    pinned the assembler and the calibration macros but not the flags, so a
    codegen-affecting flag change left no fingerprint in anything a gate reads. It was
    caught then only because both compilers import one flag list -- a property of the
    code, not a gate. A missing block is a failure for the same reason a missing
    `defines` block is: a manifest that cannot say what it compiled under cannot
    certify what shipped.
    """
    want = ratified_flags()
    got = manifest["toolchain"].get("nvcc_flags")
    if got == want:
        print(f"flags OK  {len(want)} codegen flags")
        return True
    missing = [f for f in want if f not in (got or [])]
    extra = [f for f in (got or []) if f not in want]
    if got is not None and not missing and not extra:
        #: SAME FLAGS, DIFFERENT ORDER. Order is compared deliberately -- `nvcc` reads
        #: its flag list left to right and a later flag can override an earlier one, so
        #: a reordering is not provably a no-op -- but a refusal whose two difference
        #: lists are both empty says nothing, and a reader would take it for a bug in
        #: the gate rather than a difference in the manifest.
        print(f"\nFAIL  FLAG DRIFT (ORDER). The manifest records the same {len(want)} "
              f"flags in a different order.\n  tree:     {want}\n  manifest: {got}\n"
              f"Order is compared because `nvcc` reads the list left to right and a "
              f"later flag can override an earlier one; re-ratify with --write.")
        return False
    print(f"\nFAIL  FLAG DRIFT. The manifest was measured under a different flag list "
          f"than the tree compiles with.\n  in the tree, not the manifest: {missing}\n"
          f"  in the manifest, not the tree: {extra}\n"
          f"The ratified register and spill numbers describe a compile nobody runs; "
          f"re-ratify with --write.")
    return False


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

    THE FRESHNESS TRIGGER (session 9, R1). A register budget is a MEASURED
    constant; its justification EXPIRES when `csrc/` moves. Recording the source digest
    in the manifest and CHECKING it makes that trap structural: a ratification
    whose sources have materially changed is a hard failure demanding `--write`,
    which is a fresh measurement, never an edit.
    """
    h = hashlib.sha256()
    for path in sorted(SRC.rglob("*")):
        #: MARKDOWN IS NOT A CODEGEN INPUT. `instantiations/README.md` lives inside
        #: the digested tree because it documents the generated files beside it;
        #: hashing it would make an editorial change read as SOURCE DRIFT and demand
        #: a full re-measurement, which is how people are trained to re-ratify
        #: without reading. Paths participate as well as contents, so a `.cu`
        #: renamed to `.md` still moves the digest -- by disappearing from it.
        if path.is_file() and path.suffix != ".md":
            h.update(str(path.relative_to(SRC)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()


#: The build's own record of exactly what it ran, at the fixed directory
#: `setup.py` and this file's `COMPILE_CWD` share.
NINJA_FILE = COMPILE_CWD / "build.ninja"


def _ninja_cuda_template() -> tuple[str, str, str] | None:
    """`(nvcc, cuda_cflags, cuda_post_cflags)` read VERBATIM out of the build's
    own `build.ninja`, or `None` if no build has ever run from `COMPILE_CWD`.

    See `_nvcc_command`'s docstring for why this exists. Three variables, read
    by name rather than by line position, so a torch/ninja version that
    reorders the file (but keeps the same variable names) still parses.
    """
    if not NINJA_FILE.exists():
        return None
    text = NINJA_FILE.read_text()

    def var(name: str) -> str | None:
        m = re.search(rf"^{name} = (.*)$", text, re.M)
        return m.group(1) if m else None

    nvcc, cuda_cflags, cuda_post_cflags = var("nvcc"), var("cuda_cflags"), var("cuda_post_cflags")
    if nvcc is None or cuda_cflags is None or cuda_post_cflags is None:
        return None
    return nvcc, cuda_cflags, cuda_post_cflags


def _without_gencode(flags: list[str]) -> list[str]:
    """`flags` with every `-gencode` dropped, in both of nvcc's spellings
    (`-gencode=...` and a separated `-gencode ...`)."""
    out: list[str] = []
    drop_next = False
    for f in flags:
        if drop_next:
            drop_next = False
            continue
        if f in ("-gencode", "--generate-code"):
            drop_next = True
            continue
        if f.startswith(("-gencode=", "--generate-code=")):
            continue
        out.append(f)
    return out


def _nvcc_command(arches, source: Path, obj_path: str) -> list[str]:
    """The extension's own compile flags, IMPORTED rather than restated.

    This gate compiles the same translation units the build does, so a flag list
    maintained separately from `setup.py`'s would produce ratified numbers that
    describe a binary nobody ships. `tools/build_flags.py` is the one source for
    the CODEGEN-relevant flags (`ratified_flags()`, checked against the manifest
    by `check_flags()` -- untouched by anything below).

    **THE ARGV ITSELF**, though, is reconstructed here two different ways, and
    which one runs is a CACHE-KEY FIDELITY question, not a codegen one:

    * If `COMPILE_CWD/build.ninja` exists (a build has run from the fixed
      directory `setup.py`'s `RoLABuildExtension.finalize_options` pins), reuse
      its `cuda_cflags`/`cuda_post_cflags` VERBATIM. This replaced a hand
      reconstruction that was independently wrong in three ways, each found only
      by a full build measuring 0% sccache hits and re-deriving the real argv
      from a captured `build.ninja` (`docs/build.md#sccache`): missing the
      torch-injected pybind/`TORCH_EXTENSION_NAME` defines, missing
      `--compiler-options '-fPIC'`, and using `-isystem` where `setup.py`'s
      actual `CUDAExtension(include_dirs=...)` path emits plain `-I` for every
      include including torch's own (a `distutils`/`setuptools` behavior
      distinct from `torch.utils.cpp_extension`'s JIT-`load()` path, which DOES
      use `-isystem` -- the two are not the same code path and this file
      compiled against the wrong one's convention). Reconstructing by hand is an
      unbounded surface against future torch/setuptools versions; reusing the
      build's own record is correct BY CONSTRUCTION instead of by transcription.
    * Otherwise (no build has ever run from `COMPILE_CWD` -- a fresh checkout
      before the first `pip install -e .`), fall back to the independent
      reconstruction below. This path has no cache to collide with yet, so its
      only job is to compile correctly, which `ratified_flags()`'s own
      byte-for-byte record covers.

    **THE ONE FLAG THAT IS NOT INHERITED IS `-gencode`.** The targets are this
    RUN's (`--arch`, defaulting to :data:`DEFAULT_ARCHES`), never the last
    build's. Inheriting them made a single-arch build (`ROLA_CUDA_ARCHS=86`)
    silently narrow the next ratification to sm_86 and then refuse with "the
    compile reported NO entries for sm_80" -- a message that reads as source
    drift and is not one. That is the extension-trap family: an environment fact
    the tool could not see, surfacing as a claim about the tree. `-gencode` is
    already excluded from :data:`_NON_CODEGEN_FLAG_PREFIXES`' ratified list for
    the same reason it is substituted here: it names targets, and the per-arch
    manifest filing is what records them.
    """
    tpl = _ninja_cuda_template()
    if tpl is not None:
        nvcc, cuda_cflags, cuda_post_cflags = tpl
        cmd = [_NVCC_BIN or nvcc, "--generate-dependencies-with-compile",
               "--dependency-output", f"{obj_path}.d"]
        cmd += _without_gencode(shlex.split(cuda_cflags))
        cmd += ["-c", str(source), "-o", obj_path]
        cmd += _without_gencode(shlex.split(cuda_post_cflags))
        return cmd + build_flags.gencodes(arches)

    import torch.utils.cpp_extension as ext

    cmd = [_NVCC_BIN or cuda_bin("nvcc"), "-c", str(source), f"-I{SRC}", f"-I{CUTLASS_INC}"]
    for inc in ext.include_paths(device_type="cuda"):
        cmd += ["-I", inc]
    cmd += ["-I", sysconfig.get_paths()["include"]]
    cmd += build_flags.torch_defines()
    cmd += build_flags.nvcc_flags(arches)
    #: CACHE-KEY FIDELITY, not codegen fidelity -- see
    #: `build_flags.torch_extension_cache_key_flags`'s docstring. Deliberately NOT
    #: part of `ratified_flags()`/the manifest's `nvcc_flags` block: these flags
    #: are inert in every TU this gate compiles, so they must never gate a ratchet
    #: or trigger a re-ratification on their own.
    cmd += build_flags.torch_extension_cache_key_flags()
    return cmd + ["-o", obj_path]


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
    K5: a register budget's justification cannot silently outlive the working
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


def _parse_report(stderr: str) -> tuple[dict, list[str]]:
    entries: dict[str, dict] = {}
    unhonored = sorted(set(OUT_OF_RANGE.findall(stderr)))
    key = None
    for line in stderr.splitlines():
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


def compile_and_parse(arches, objdir: str, sources: list[Path],
                      jobs: int | None = None) -> tuple[dict, list[str], dict[str, str]]:
    """Compile the shipped translation units and read `ptxas`'s report.

    Returns (entries, unhonored, objects) where `objects` maps each entry key to the
    object file it came from -- the hot-loop gate disassembles the SASS out of the
    SAME compile the register numbers came from, so the two can never describe
    different binaries, and with more than one translation unit that association has
    to be carried rather than assumed.

    The translation units are compiled CONCURRENTLY, at the same memory-derived job
    count the build uses: this gate compiles the same matrix the build does, and
    running it serially would put the second full-matrix compile back into every
    build cycle, which is the cost the shard exists to remove.
    """
    jobs = jobs or build_flags.default_max_jobs(2)
    entries: dict[str, dict] = {}
    unhonored: set[str] = set()
    objects: dict[str, str] = {}
    t0 = time.perf_counter()

    COMPILE_CWD.mkdir(parents=True, exist_ok=True)

    def run(source: Path) -> tuple[Path, str, str]:
        obj = str(Path(objdir) / f"{source.stem}.o")
        cmd = _nvcc_command(arches, source, obj)
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(COMPILE_CWD))
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            raise SystemExit(f"{source.name} FAILED to compile (rc={proc.returncode})")
        return source, obj, proc.stderr

    with ThreadPoolExecutor(max_workers=min(jobs, len(sources))) as pool:
        for source, obj, stderr in pool.map(run, sources):
            got, bad = _parse_report(stderr)
            for key, value in got.items():
                if key in entries:
                    #: TWO TRANSLATION UNITS EMITTING ONE INSTANTIATION is the exact
                    #: state the shard exists to prevent, and merging their reports
                    #: would hide it behind numbers that look fine.
                    raise SystemExit(
                        f"{key} was compiled by more than one translation unit "
                        f"({source.name} is the second). Nothing is sharded; run "
                        f"tools/gen_shards.py --check and tools/fatbin_entries.py")
                entries[key] = value
                objects[key] = obj
            unhonored |= set(bad)

    wall = time.perf_counter() - t0
    print(f"{len(sources)} translation unit(s) compiled in {wall:.1f} s at "
          f"{min(jobs, len(sources))} jobs over {len(arches)} targets "
          f"({', '.join('sm_' + a for a in arches)})")
    return entries, sorted(unhonored), objects


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
    print(f"\n{len(entries)} instantiations measured, {len(unhonored)} with an unhonored "
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
#: register-rung scoping mechanized that as "no
#: `LDL`/`STL` at loop depth >= 1", and this gate is that test as a BUILD FAILURE.
#:
#: **IT IS A RATCHET, NOT AN ABSOLUTE, and the difference was MEASURED before the
#: gate landed (session 9).** The decay arms execute dynamically-indexed LOCAL
#: ARRAYS (`LDL.64 R28, [R26]` -- computed addresses, `ptxas` spill bytes ZERO)
#: inside their loops; that is the "local ARRAYS, not pressure" class,
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

#: THE BODY-HASH LINE RULES, and they are deliberately WIDER than `_SASS_INST`'s.
#: The hot-loop parser wants an instruction it can reason about, so it requires a
#: terminating `;`; the body hash wants EVERY line that carries codegen, which
#: includes the continuation line holding an instruction's second encoding word. Two
#: instructions can print identically and encode differently -- a predicate, a
#: scheduling control field, a reuse flag -- so the encoding words are in the hash
#: and must be. Not in it: `Function :` headers, `.headerflags`, lineinfo, and the
#: padding that aligns the encoding column, which is a property of the LISTING and
#: moves with the longest instruction in it.
#: THE SCANNER'S OWN VERSION. What reaches the digest is a property of
#: :func:`sass_scan`, not of the binary: widening the capture once moved every digest
#: in an unchanged binary, which reads exactly like a codegen change and is not one.
#: Every artifact carrying one of these hashes is stamped with it -- the maps
#: `tools/sass_bodies.py` writes and the per-instantiation `sass_sha256` in every
#: manifest -- and each refuses a comparison across two versions. BUMP IT in the same
#: commit that changes which lines `sass_scan` feeds the digest.
#: 1 = instruction text only (the pre-2026-08-05 capture, which dropped the encoding
#: words); 2 = every offset-marked and encoding line, whitespace-normalised.
SASS_HASH_VERSION = 2

_SASS_BODY_INST = re.compile(r"/\*[0-9a-f]{4,}\*/")
_SASS_BODY_ENC = re.compile(r"^\s*/\*\s*0x[0-9a-f]+\s*\*/\s*$")


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


def sass_scan(sass_lines, wanted: set | None = None) -> dict:
    """ONE pass over `cuobjdump -sass`, answering every question asked of the SASS.

    `{key -> {"body_sha256", "hot_locals": [(addr, inst)], "local_ops": int}}`:

    * `body_sha256` is the assembled body's identity, and is what "the same code"
      means in this repository -- names are the weaker check twice over, since a
      mangling can move while the code is identical (a translation-unit restructure
      does exactly that to anonymous-namespace kernels) and the code can move while
      every name is identical.
    * `hot_locals` is the in-loop local traffic the ratchet bounds.
    * `local_ops` is EVERY `LDL`/`STL`, in a loop or not -- a seconds-cheap presence
      count that catches local traffic the byte-level spill report does not (a
      dynamically indexed local array is not a spill) and that the in-loop count does
      not see outside a backward-branch interval.

    `wanted` filters; `None` scans every entry point, which is what a whole-binary
    comparison needs. A name appearing twice within one arch with DIFFERENT bodies
    is refused rather than resolved last-writer-wins: it means the kernel is
    duplicated across objects, which is `tools/fatbin_entries.py`'s failure, and
    hashing one of the two copies would hide it behind a green comparison.
    """
    report: dict[str, dict] = {}
    arch, key, items, pos, digest, locals_seen = None, None, [], 0, None, 0

    def flush():
        if key is None or (wanted is not None and key not in wanted):
            return
        entry = {"body_sha256": digest.hexdigest(),
                 "hot_locals": _hot_locals_in_function(items),
                 "local_ops": locals_seen}
        if key in report and report[key]["body_sha256"] != entry["body_sha256"]:
            raise SystemExit(f"{key} is disassembled twice with DIFFERENT bodies; "
                             f"run tools/fatbin_entries.py")
        report[key] = entry

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
            digest, locals_seen = hashlib.sha256(), 0
            continue
        if key is None:
            continue
        if _SASS_BODY_INST.search(line) or _SASS_BODY_ENC.match(line):
            digest.update(" ".join(line.split()).encode())
            digest.update(b"\n")
        m = _SASS_LABEL_DEF.match(line)
        if m:
            items.append((pos, "label", m.group(1)))
            pos += 1
            continue
        m = _SASS_INST.search(line)
        if m:
            items.append((pos, "inst", (int(m.group(1), 16), m.group(2))))
            pos += 1
            if _SASS_LOCAL.search(m.group(2)):
                locals_seen += 1
    flush()
    return report


def measure_sass(objects: dict[str, str], entries: dict) -> dict[str, dict]:
    """Disassemble the SAME objects the register numbers came from.

    One `cuobjdump` pass per object serves all three SASS-derived facts, so arming
    the body hash and the local-presence ratchet costs no additional disassembly.
    """
    def one(obj: str) -> dict[str, dict]:
        wanted = {k for k, v in objects.items() if v == obj}
        proc = subprocess.Popen([cuda_bin("cuobjdump"), "-sass", obj],
                                stdout=subprocess.PIPE, text=True)
        got = sass_scan(proc.stdout, wanted)
        rc = proc.wait()
        if rc != 0:
            raise SystemExit(f"cuobjdump -sass FAILED (rc={rc}) on {obj}; the "
                             f"SASS gates cannot pass a binary they cannot read")
        return got

    #: DISASSEMBLED IN PARALLEL, for the same reason the compiles are: this runs on
    #: every build, and a serial pass over ten objects is the shard win being handed
    #: back at the far end of the gate. The objects are disjoint by construction --
    #: each instantiation is emitted by exactly one of them -- so there is nothing to
    #: reconcile between the workers.
    report: dict[str, dict] = {}
    objs = sorted(set(objects.values()))
    with ThreadPoolExecutor(max_workers=min(len(objs), (os.cpu_count() or 1))) as pool:
        for got in pool.map(one, objs):
            report.update(got)
    return {name: report.get(name, {"body_sha256": None, "hot_locals": [],
                                    "local_ops": 0})
            for name in entries}


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


def check_local_presence(local_ops: dict[str, int], want: dict) -> bool:
    """The CHEAP RATCHET: `LDL`/`STL` PRESENCE, in a loop or not.

    A complement to the two expensive gates rather than a replacement for either.
    `ptxas`'s spill bytes miss local traffic that is not a spill -- a dynamically
    indexed local array is the case that matters here -- and the in-loop count only
    sees what sits inside a backward-branch interval. Counting every local access is
    a single integer per instantiation, read off the disassembly the other gates
    already produced, and it moves whenever either of them would.

    A ratchet, not a ban: the decay arms carry structural local traffic that no
    register budget removes, so each entry is held AT its ratified count.
    """
    missing = sorted(n for n in local_ops if n in want and "local_ops" not in want[n])
    if missing:
        print(f"\nFAIL  {len(missing)} manifest entr(ies) predate the local-presence "
              f"ratchet (no `local_ops`). An unarmed ratchet is not a looser gate, "
              f"it is no gate; re-ratify with --write, e.g.\n"
              f"        {missing[0]}")
        return False
    grew = [(n, want[n]["local_ops"], c) for n, c in sorted(local_ops.items())
            if n in want and c > want[n]["local_ops"]]
    if grew:
        print(f"\nFAIL  LOCAL-MEMORY GROWTH: {len(grew)} instantiation(s) execute "
              f"MORE local accesses than were ratified. Local traffic is a memory "
              f"round-trip the register file was supposed to absorb:")
        for name, exp, got in grew[:10]:
            print(f"        ratified {exp} -> realized {got}  {name}")
        if len(grew) > 10:
            print(f"        ... and {len(grew) - 10} more")
        return False
    print(f"local-presence gate: LDL/STL counts within ratified values on all "
          f"{len(local_ops)} entries (tree total {sum(local_ops.values())}); PASS")
    return True


def check_sass_bodies(bodies: dict[str, str], want: dict, fresh: bool,
                      stamped_version) -> bool:
    """THE INCREMENTAL RATCHET: per-instantiation codegen identity.

    The source digest is a WHOLESALE trigger -- any edit under `csrc/rola/src` moves
    it and invalidates every one of the ratified entries at once -- and that is
    correct as a trigger and useless as a diagnosis. The body hash is the per-entry
    answer to the question the digest can only ask globally: which instantiations'
    CODE actually moved.

    So the check reads two ways, and both are hard:

    * `fresh` (the digest matches): every ratified body must still hash the same. The
      digest says the inputs did not move; this says the OUTPUT did not either, which
      is an independent statement and the one the spill numbers depend on.
    * not `fresh` (the digest moved): the numbers must be re-measured, but the entries
      whose bodies are unchanged are provably still described by their ratified
      numbers. Naming the moved ones turns "re-ratify" into a scoped, checkable act
      instead of a 1,216-entry re-derivation.
    """
    if stamped_version != SASS_HASH_VERSION:
        print(f"\nFAIL  SASS HASH VERSION. The ratified bodies were hashed by scanner "
              f"version {stamped_version!r}; this tool is version "
              f"{SASS_HASH_VERSION}. Two versions hash DIFFERENT text, so their "
              f"digests disagree on every entry while describing identical code -- "
              f"the same false verdict the path trap produces. Re-ratify with --write.")
        return False
    missing = sorted(n for n in bodies if n in want and "sass_sha256" not in want[n])
    if missing:
        print(f"\nFAIL  {len(missing)} manifest entr(ies) predate the SASS-body "
              f"ratchet (no `sass_sha256`), so codegen identity is unratified and "
              f"unfalsifiable. Re-ratify with --write, e.g.\n        {missing[0]}")
        return False
    moved = sorted(n for n, h in bodies.items()
                   if n in want and h != want[n]["sass_sha256"])
    if not moved:
        if fresh:
            print(f"SASS-body gate: all {len(bodies)} ratified bodies unchanged; PASS")
            return True
        print("\nFAIL  the source digest moved but NO ratified body did. The "
              "manifest still describes this codegen exactly; re-ratify with "
              "--write to record the new source digest.")
        return False
    print(f"\nFAIL  SASS BODY DRIFT: {len(moved)} of {len(bodies)} ratified "
          f"instantiation(s) assemble to different code than was ratified"
          + ("" if fresh else " (the source digest moved too, so this is the SCOPE "
                             "of that move)") + ":")
    for name in moved[:10]:
        print(f"        {name}")
    if len(moved) > 10:
        print(f"        ... and {len(moved) - 10} more")
    if fresh:
        print("      The source digest did NOT move, so this is codegen changing "
              "under a source tree that did not: a toolchain or environment "
              "difference the digest cannot see.")
    return False


#: --------------------------------------------------------------------------
#: THE CENSUS (session 9, R0) -- the permanent instrument the register rung is
#: derived from. `--census` compiles every built arm at the shipped derived bound
#: and prints the per-key register/spill table. It never touches a manifest:
#: ratification is the recording of the SHIPPED bound, and conflating it with a
#: measurement is how a stale budget constant survived three kernel rungs.

def _regs_ctas(regs: int) -> int:
    """Register-file-side CTAs/SM at 256 threads after sm_86's 8-register
    allocation granularity. The REGISTER term only -- the driver's own occupancy
    query stays the sole residency authority."""
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
        #: THE SHARD IS A CENSUS COLUMN. Every row of the register/spill table says
        #: which translation unit would have to be recompiled to move it, which is
        #: what turns "re-ratify only these shards" from a plan into a checkable
        #: claim: csrc/rola/src/instantiations/README.md
        #: the SPANS name a plan; `BC` alone no longer does (two plans of one
        #: depth can share a `BC` and differ in how it is partitioned).
        span = ("x".join(str(x) for x in shape["spans"][:shape["D"]])
                if shape["spans"] else str(shape["BC"] or 0))
        k = (arch, shard_of_key(name.split(":", 1)[1]), shape["kernel"][:5],
             span, f"{shape['D']}x{shape['B']}")
        by_key.setdefault(k, []).append((e, hot.get(name, 0)))
    print(f"\n{'arch':6} {'shard':10} {'kern':>5} {'spans':>9} "
          f"{'DxB':>7} {'n':>3} {'regs':>9} {'ctas(reg)':>9} {'st':>6} {'ld':>6} "
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
        print(f"{k[0]:6} {k[1]:10} {k[2]:>5} {k[3]:>9} "
              f"{k[4]:>7} {len(es):>3} {rg:>9} {cg:>9} {st:>6} "
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
                    help="a compute capability to measure and ratify, repeatable "
                         f"(default: {' '.join(DEFAULT_ARCHES)}). One manifest file "
                         "per arch, so bringing up a new one touches no other arch's "
                         "measurements.")
    ap.add_argument("--census", action="store_true",
                    help="FORCING CENSUS: compile the whole matrix, print the "
                         "per-key register/spill table, touch no manifest")
    ap.add_argument("--json", metavar="PATH",
                    help="census only: dump the raw per-instantiation record for "
                         "arm-vs-arm diffing")
    ap.add_argument("--shard", action="append", metavar="NAME",
                    help="MEASUREMENT SCOPE: compile only these instantiation shards "
                         "(repeatable). A ratchet refusal is investigated one shard "
                         "at a time; --write still refuses a partial matrix.")
    a = ap.parse_args()

    arches = tuple(a.arch) if a.arch else DEFAULT_ARCHES
    if a.shard and a.write:
        raise SystemExit("--shard is a MEASUREMENT scope, not a ratification scope: "
                         "a manifest written from a subset of the translation units "
                         "would silently drop every instantiation it did not compile")
    sources = tu_paths(tuple(a.shard) if a.shard else None)

    sccache_info = gate_sccache()
    version = ptxas_version()
    import tempfile
    with tempfile.TemporaryDirectory(prefix="rola_ratify_") as tmp:
        entries, unhonored, objects = compile_and_parse(arches, tmp, sources)
        scan = measure_sass(objects, entries)
        hot = {name: len(s["hot_locals"]) for name, s in scan.items()}
        bodies = {name: s["body_sha256"] for name, s in scan.items()}
        local_ops = {name: s["local_ops"] for name, s in scan.items()}
        for name, s in scan.items():
            entries[name]["hot_locals"] = len(s["hot_locals"])
            entries[name]["local_ops"] = s["local_ops"]
            entries[name]["sass_sha256"] = s["body_sha256"]
        if a.census:
            census_report(entries, unhonored, hot)
            if a.json:
                Path(a.json).write_text(json.dumps(
                    dict(ptxas=version,
                         csrc_sha256=csrc_digest(), entries=entries,
                         shards=shard_field(entries)),
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
                    f"the compile reported NO entries for sm_{arch}; refusing to "
                    f"write an empty ratification. A manifest that exists but "
                    f"measures nothing would pass rule 3 by file existence alone.")
            path = manifest_path(arch)
            #: The SOURCE DIGEST is recorded beside the toolchain: a manifest is a
            #: measurement OF a source tree UNDER an assembler, and both halves of
            #: that sentence must be checkable (the freshness trigger, session 9).
            path.write_text(json.dumps(
                {"toolchain": {"ptxas": version, "defines": shipped_defines(),
                                #: THE CODEGEN FLAG LIST, arch-independent part.
                                #: A ratification is a measurement OF sources UNDER
                                #: an assembler AND its flags; without this the
                                #: third of those was unrecorded and a
                                #: codegen-affecting flag could land without moving
                                #: anything a gate reads.
                                "nvcc_flags": ratified_flags(),
                                #: The scanner version behind every entry's
                                #: `sass_sha256`. Same argument as the flag list:
                                #: a record that cannot say what produced its
                                #: hashes cannot certify them.
                                "sass_hash_version": SASS_HASH_VERSION,
                                #: THE COMPILER CACHE, per the input-closure rule
                                #: (tools/sccache_toolchain.py). `None` if this
                                #: ratification ran uncached. Recorded, not
                                #: ratchet-gated: sccache is a measured-transparent
                                #: pass-through (docs/build.md#sccache) so its
                                #: version cannot move the register/spill numbers
                                #: the way a `ptxas` swap can -- drift is REPORTED
                                #: (below), not FAILED, the same treatment register
                                #: drift gets.
                                "sccache": {"version": sccache_info["version"],
                                            "sha256": sccache_info["sha256"]}
                                           if sccache_info else None},
                 "source": {"csrc_sha256": csrc_digest(),
                            "date": time.strftime("%Y-%m-%d")},
                 #: WHICH INSTANTIATIONS THIS RATIFICATION COVERS, by membership
                 #: hash rather than by file list. It enters the digest, so a
                 #: re-partition is a visible re-ratification and a scoped
                 #: re-measurement is a provable statement.
                 "shards": gen_shards.shard_manifest_block(),
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
    ok = check_flags(manifest) and ok
    #: REPORT, not FAIL (see the `--write` block's comment on why): sccache cannot
    #: move a ratified number, so a version drift here is provenance, not a ratchet.
    manifest_sccache = manifest["toolchain"].get("sccache")
    this_sccache = {"version": sccache_info["version"], "sha256": sccache_info["sha256"]} \
        if sccache_info else None
    if manifest_sccache != this_sccache:
        print(f"sccache drift (REPORT only)  manifest: {manifest_sccache}  "
              f"this run: {this_sccache}")
    #: The freshness verdict is REUSED rather than recomputed: the body gate reads
    #: differently on each side of it, and two independent computations of "is the
    #: source digest current" could disagree.
    fresh = check_freshness(manifest)
    ok = fresh and ok
    ok = check(entries, unhonored, want) and ok
    ok = check_hot_loop(hot, want) and ok
    ok = check_local_presence(local_ops, want) and ok
    ok = check_sass_bodies(bodies, want, fresh,
                           manifest["toolchain"].get("sass_hash_version")) and ok

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

        #: NON-VACUITY OF THE PER-ARCH SPLIT. Rule 3 is now a FILE-EXISTENCE
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
        rejoined = {"toolchain": halves[0]["toolchain"], "shards": halves[0]["shards"],
                    "entries": {k: v for h in halves for k, v in h["entries"].items()}}
        if manifest_digest(rejoined) != manifest_digest(manifest):
            print("\nSELF-TEST FAILED: the per-arch files do not rejoin to the same "
                  "record; the split is not filing-only")
            return 1
        print(f"SELF-TEST PASSED: {len(arches)} per-arch files rejoin to one digest "
              f"{manifest_digest(rejoined)[:16]}...")

        #: ... AND THE SHARD BLOCK DOES NOT MAKE THE DIGEST A FILING PROPERTY. It is
        #: membership only, so re-filing the same measurements -- reordering the
        #: block's keys, or splitting the arches differently -- must move nothing. A
        #: block that carried paths or an ordering would fail exactly here, which is
        #: why it carries neither.
        refiled = {"toolchain": manifest["toolchain"],
                   "shards": dict(reversed(list(manifest["shards"].items()))),
                   "entries": manifest["entries"]}
        if manifest_digest(refiled) != manifest_digest(manifest):
            print("\nSELF-TEST FAILED: re-filing the shard block moved the manifest "
                  "digest; the block is recording HOW the ratification is filed, not "
                  "WHAT it covers")
            return 1
        #: ... and it must still be LOAD-BEARING: a changed membership must move it.
        moved = {"toolchain": manifest["toolchain"], "entries": manifest["entries"],
                 "shards": {k: dict(v, count=v["count"] + 1)
                            for k, v in manifest["shards"].items()}}
        if manifest_digest(moved) == manifest_digest(manifest):
            print("\nSELF-TEST FAILED: a changed shard membership did not move the "
                  "manifest digest; the block is decorative")
            return 1
        print("SELF-TEST PASSED: the shard block is membership-only (re-filing is "
              "digest-neutral) and load-bearing (a membership change is not)")

        #: NON-VACUITY OF THE CROSS-ARCH AGREEMENT. Two arch files that disagree
        #: about which instantiations exist are not one ratification.
        print("\n--- SELF-TEST: per-arch manifests disagreeing on shard membership\n"
              "    (this load MUST fail) ---")
        real = manifest_path(arches[-1])
        saved = real.read_text()
        blob = json.loads(saved)
        blob["shards"] = dict(blob.get("shards") or {}, __self_test__={"count": 1})
        try:
            real.write_text(json.dumps(blob, indent=1) + "\n")
            load_manifest(arches)
        except SystemExit as exc:
            print(f"SELF-TEST PASSED: {str(exc).splitlines()[0]}")
        else:
            print("\nSELF-TEST FAILED: two manifests with different shard "
                  "memberships loaded as one ratification")
            return 1
        finally:
            real.write_text(saved)

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
        rep = sass_scan(iter(hot_sass), {"sm_86:k_hot", "sm_86:k_cold"})
        print("\n--- SELF-TEST: the hot-loop parser on synthetic SASS ---")
        counts = {k: len(v["hot_locals"]) for k, v in rep.items()}
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

        #: NON-VACUITY OF THE LOCAL-PRESENCE RATCHET, on the same synthetic SASS.
        #: `k_cold`'s `LDL` sits outside every loop, so the hot-loop gate clears it
        #: and this one must NOT -- which is the whole reason the cheap count is
        #: kept beside the expensive one.
        print("\n--- SELF-TEST: local-presence growth on an out-of-loop LDL\n"
              "    (this run MUST fail) ---")
        assert rep["sm_86:k_cold"]["local_ops"] == 1
        if check_local_presence({"sm_86:k_cold": 1},
                                {"sm_86:k_cold": {"local_ops": 0}}):
            print("\nSELF-TEST FAILED: the presence ratchet PASSED local traffic "
                  "it was not ratified for")
            return 1
        print("SELF-TEST PASSED: the presence ratchet catches a local access the "
              "in-loop gate is right to ignore")

        #: NON-VACUITY OF THE BODY HASH, both directions. Two functions whose
        #: instruction text differs must hash differently, or the comparison is
        #: decorative; the SAME text laid out with different column padding must hash
        #: identically, or every rebuild reads as a codegen change and the gate gets
        #: turned off.
        print("\n--- SELF-TEST: the SASS body hash on synthetic SASS ---")
        moved_sass = [
            "arch = sm_86",
            "                Function : k_a",
            "        /*0000*/ IADD3 R3, R3, 1, RZ ;",
            "                                     /* 0x000fe200078e0203 */",
            "                Function : k_b",
            "        /*0000*/ IADD3 R3, R3, 2, RZ ;",
            "                                     /* 0x000fe200078e0203 */",
            "                Function : k_a_repadded",
            "   /*0000*/     IADD3   R3,  R3, 1, RZ ;",
            "     /* 0x000fe200078e0203 */",
        ]
        got = sass_scan(iter(moved_sass))
        a, b = got["sm_86:k_a"]["body_sha256"], got["sm_86:k_b"]["body_sha256"]
        repad = got["sm_86:k_a_repadded"]["body_sha256"]
        if a == b:
            print("\nSELF-TEST FAILED: two different instruction texts hashed the "
                  "same; the body hash cannot detect a codegen change")
            return 1
        if a != repad:
            print("\nSELF-TEST FAILED: re-padding the listing moved the body hash; "
                  "the hash is over the LISTING's column alignment, not the code")
            return 1
        print("SELF-TEST PASSED: the body hash separates changed code and ignores "
              "listing padding")

        #: ... and the GATE built on it must fail on a perturbed ratified hash.
        print("\n--- SELF-TEST: a ratified body hash that no longer matches\n"
              "    (this run MUST fail) ---")
        if check_sass_bodies({"sm_86:k_a": a},
                             {"sm_86:k_a": {"sass_sha256": "0" * 64}}, True,
                             SASS_HASH_VERSION):
            print("\nSELF-TEST FAILED: the gate ACCEPTED an instantiation whose "
                  "assembled body is not the one that was ratified")
            return 1
        print("SELF-TEST PASSED: the gate rejects codegen it did not ratify")

        #: ... and an UNARMED entry is a failure, not a pass. A ratchet with no
        #: recorded value is the vacuous gate this discipline exists to catch.
        print("\n--- SELF-TEST: a manifest entry with no ratified body hash\n"
              "    (this run MUST fail) ---")
        if check_sass_bodies({"sm_86:k_a": a}, {"sm_86:k_a": {}}, True,
                             SASS_HASH_VERSION):
            print("\nSELF-TEST FAILED: an unratified body hash passed as though it "
                  "had been checked")
            return 1
        print("SELF-TEST PASSED: an unarmed body ratchet fails rather than passes")

        #: ... and a body hash produced by a DIFFERENT scanner version is refused
        #: whatever it says. This is P06's map-stamp argument applied to the
        #: manifest: two versions hash different text, so an unstamped or
        #: differently-stamped ratification disagrees on every entry while
        #: describing identical code.
        print("\n--- SELF-TEST: ratified bodies hashed by another scanner version\n"
              "    (both of these MUST fail) ---")
        for stamped in (SASS_HASH_VERSION - 1, None):
            if check_sass_bodies({"sm_86:k_a": a},
                                 {"sm_86:k_a": {"sass_sha256": a}}, True, stamped):
                print(f"\nSELF-TEST FAILED: bodies stamped {stamped!r} were compared "
                      f"against version {SASS_HASH_VERSION} digests")
                return 1
        print("SELF-TEST PASSED: the gate rejects hashes it did not produce")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
