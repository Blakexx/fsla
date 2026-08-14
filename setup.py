"""THE BUILD -- ahead-of-time, arch-gated, and refused when the evidence is missing.

**WHAT THIS REPLACES.** In the fork the extension was built by
``torch.utils.cpp_extension.load(...)`` on the first call into the kernel: a JIT
compile at runtime, in the user's process, under whatever toolchain the user's
machine happened to have. The closed-world codegen policy forbids that (rule 2), and
the ratification manifest -- a per-instantiation register/spill measurement taken
under one named ``ptxas`` -- is meaningless against a binary somebody else's compiler
produced. So the build moved here, and the gates moved with it.

**THE FOUR RULES, and where each one lives in this file.**

===== ==================================================================== ==========
rule   what it forbids                                                      here
===== ==================================================================== ==========
R1     a build under an assembler the manifest was not ratified under       ``_gate_toolchain``
R2     ``code=compute_XX`` (PTX) anywhere in the fatbin                     ``gencodes`` + its assertion
R3     an architecture in the fatbin with no ratified manifest              ``_gate_ratified``
R4     a *launch* on a device whose arch was never measured                 ``csrc/rola/src/arch_table.cuh`` (C++)
R5     a vendored codegen INPUT that does not match its recorded pin        ``_gate_vendored``
===== ==================================================================== ==========

**R5 AND WHY A HEADER-ONLY LIBRARY IS A CODEGEN INPUT, NOT A PACKAGING DETAIL.**
``csrc/third_party/cutlass`` vendors CUTLASS/CuTe headers. Header-only or not, every
line of it that the kernel instantiates is code ``ptxas`` assembles, so a silent edit
to a vendored header moves the register allocation the manifest ratified exactly as an
edit to ``consumer_kernel.cuh`` would. ``_gate_vendored`` therefore hashes the whole
vendored subtree and compares it against ``PIN.json``'s recorded digest BEFORE the
compile, in the same pass as rules 1 and 3, and the pin (upstream, tag, commit, digest)
is stamped into ``rola/_build_config.py`` so an installed wheel can answer "which
CUTLASS is inside this ``.so``?" with no checkout. The vendored subtree is
byte-identical to the upstream tag, which is what makes the digest checkable against
something other than itself.

R1 and R3 run BEFORE ``build_ext`` -- they are cheap, and failing after a multi-hour
compile helps nobody. The manifest gate runs AFTER, because the pre-build check
proves the *inputs* were ratified and only the post-build run proves the *shipped
codegen* still matches what was measured. A source install therefore re-runs the
ratification gates by construction; it is not an optional ``make`` target.

**FLAG FIDELITY (extraction W1).** Every codegen-affecting flag here is the flag the
fork's JIT build passed, and the list is not a redesign. ``bindings.py`` passed
``-O3 -std=c++17 -lineinfo --threads=2`` plus torch's own ``COMMON_NVCC_FLAGS``
(``-D__CUDA_NO_HALF_OPERATORS__``, ``-D__CUDA_NO_HALF_CONVERSIONS__``,
``-D__CUDA_NO_BFLOAT16_CONVERSIONS__``, ``-D__CUDA_NO_HALF2_OPERATORS__``,
``--expt-relaxed-constexpr``) and ``-D_GLIBCXX_USE_CXX11_ABI=<torch's>``; ``CUDAExtension``
adds exactly that same set, so it is reproduced rather than restated. TWO deliberate
deviations, both non-codegen-affecting, both stated so a reader does not have to
diff:

* ``--ptxas-options=-v`` is ADDED. It changes what the assembler PRINTS, not what it
  emits, and ``tools/ratify.py`` parses exactly that output.
* the design's ``-U__CUDA_NO_HALF_OPERATORS__`` / ``-U__CUDA_NO_HALF_CONVERSIONS__``
  / ``-U__CUDA_NO_BFLOAT16_CONVERSIONS__`` / ``--expt-extended-lambda`` are NOT
  adopted. Those would UNDO torch's own defines and enable a language feature the
  measured build did not have. The manifest was ratified without them; adopting them
  would be a codegen change smuggled in as a packaging change.

``--use_fast_math`` is likewise refused: Tier-1's tolerances are derived as fp32
accumulation tolerances against an fp64 oracle, and fast-math silently changes the
numeric contract those tolerances were computed under.
"""
from __future__ import annotations

import hashlib
import json
import os
import pprint
import re
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc" / "rola"
MANIFEST_DIR = ROOT / "tools" / "manifests"
CUTLASS_DIR = ROOT / "csrc" / "third_party" / "cutlass"

#: `tools/` is not an installed package -- it is the repository's instruments -- but
#: the build and the ratification gate must not maintain two copies of the codegen
#: flag list or two copies of the instantiation matrix. Both are imported from there.
sys.path.insert(0, str(ROOT / "tools"))
import build_flags  # noqa: E402
import gen_shards  # noqa: E402
import ratify  # noqa: E402
import sccache_toolchain  # noqa: E402

#: THE RATIFIED SET. Overridable only downward in practice: adding an arch here
#: without a manifest fails the build at ``_gate_ratified`` (rule 3), which is the
#: point -- the supported matrix IS the ratified set, not a wish list.
ROLA_CUDA_ARCHS = [a for a in os.getenv("ROLA_CUDA_ARCHS", "80;86").split(";") if a]

#: THE EXTENSION IS NAMESPACED INSIDE THE PACKAGE, not top-level. A top-level
#: ``rola_cuda`` is the flash-attention / mamba precedent, and it has one concrete
#: cost: an in-place (``pip install -e`` / ``build_ext --inplace``) build drops the
#: ``.so`` at the REPO ROOT, so a developer's import path is not the installed
#: layout. A dotted name makes setuptools place the binary inside ``rola/`` in both
#: cases, so there is ONE layout, the repo root stays clean, and the module cannot be
#: imported by accident from a working directory that merely happens to contain it.
#:
#: This is a PACKAGING name only. ``PYBIND11_MODULE(TORCH_EXTENSION_NAME, ...)`` takes
#: it from this constant, so nothing in ``csrc/`` names it; the C++ namespaces
#: (``rola``, ``rola::mma``, ``rola::paging``) and every kernel symbol are untouched --
#: which is why the per-arch manifests diff at ZERO value changes across this move.
#: The manifests key on MANGLED INSTANTIATION NAMES of ``rola::consumer::
#: consumer_forward_kernel``, and the module-init symbol they do not contain is the
#: only thing this rename can move.
MODULE_NAME = "rola._C"


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------
def read_version() -> str:
    """From ``rola/__init__.py``. No setuptools_scm: the tag must not be able to
    disagree with the shipped ``__version__``, which is also stamped next to the
    manifest hashes in ``_build_config.py``."""
    text = (ROOT / "rola" / "__init__.py").read_text()
    match = re.search(r'^__version__\s*=\s*([\'"])([^\'"]+)\1', text, re.M)
    if not match:
        raise RuntimeError("no __version__ in rola/__init__.py")
    return match.group(2)


# ---------------------------------------------------------------------------
# RULE 2 -- gencode
# ---------------------------------------------------------------------------
def assert_no_ptx(flags: list[str]) -> None:
    """Not decoration: this is what makes rule 2 survive a future editor who adds a
    'just for the next architecture' PTX line. The build fails rather than silently
    shipping unmeasured codegen."""
    bad = [f for f in flags if "code=compute_" in f]
    if bad:
        raise RuntimeError(
            "CLOSED-WORLD RULE 2: PTX JIT fallback must never be emitted. Offending "
            f"gencode flag(s): {bad}. A `code=compute_XX` entry lets the driver "
            "compile this kernel for an architecture nobody measured; ratify the "
            "architecture instead (docs/bringup.md)."
        )


# ---------------------------------------------------------------------------
# RULES 1 and 3 -- the pre-build gates
# ---------------------------------------------------------------------------
def cuda_bin(tool: str) -> str:
    home = CUDA_HOME or os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    return str(Path(home) / "bin" / tool)


def ptxas_version() -> str:
    """The WHOLE ``ptxas --version`` output, whitespace-normalized -- the same
    normalization ``tools/ratify.py`` records, so the two strings are comparable
    byte for byte. The build string distinguishes two shipments of one nominal
    release, and it is exactly those that can move an allocation."""
    proc = subprocess.run([cuda_bin("ptxas"), "--version"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"cannot ask `{cuda_bin('ptxas')} --version`; a closed-world "
                           f"build cannot proceed without identifying its assembler")
    return " ".join(proc.stdout.split())


def manifest_path(arch: str) -> Path:
    """ONE FILE PER ARCH (§3.7), so rule 3 is a FILE-EXISTENCE question.

    That matters because rule 3 runs BEFORE the compile: the cheapest possible check
    is the one that must never be tempting to skip. It also makes a new
    architecture's bring-up one reviewable file in a PR rather than a diff inside a
    thousand-line document nobody reads to the end.
    """
    return MANIFEST_DIR / f"sm_{arch}.json"


def _gate_ratified(archs: list[str]) -> dict:
    """RULE 3: every arch in the fatbin has a ratified manifest, and they agree.

    Returns the MERGED record -- the union of exactly the archs being built, which is
    what ``manifest_sha256`` must then describe. A wheel built for one arch is gated
    against one arch's measurements, and its recorded digest says so.
    """
    toolchain = None
    shards = None
    entries: dict = {}
    for a in archs:
        path = manifest_path(a)
        if not path.exists():
            raise RuntimeError(
                f"CLOSED-WORLD RULE 3: sm_{a} is in ROLA_CUDA_ARCHS but there is no "
                f"ratified manifest at {path}. Ratify it first:\n"
                f"    python tools/ratify.py --arch {a} --write\n"
                f"and record why the numbers are what they are (docs/bringup.md). "
                f"Building an architecture nobody measured is the failure this rule "
                f"exists to prevent."
            )
        blob = json.loads(path.read_text())
        if not blob["entries"]:
            raise RuntimeError(
                f"CLOSED-WORLD RULE 3: {path} exists but ratifies NOTHING. File "
                f"existence is the cheap form of the question, not a weaker one."
            )
        if toolchain is None:
            toolchain, shards = blob["toolchain"], blob["shards"]
        elif blob["toolchain"] != toolchain:
            raise RuntimeError(
                f"CLOSED-WORLD RULE 1: {path.name} pins a different assembler than the "
                f"other manifests. The per-arch split is a FILING decision; one fatbin "
                f"may not be gated against two ratifications."
            )
        elif blob["shards"] != shards:
            raise RuntimeError(
                f"CLOSED-WORLD RULE 3: {path.name} records a different instantiation "
                f"membership than the other manifests. The per-arch split is a FILING "
                f"decision; one fatbin may not be gated against two matrices."
            )
        entries.update(blob["entries"])
        print(f"rule 3 OK  sm_{a}: {len(blob['entries'])} ratified instantiations "
              f"({path.name})")
    return {"toolchain": toolchain, "shards": shards, "entries": entries}


def vendored_tree_sha256(root: Path) -> tuple[str, int]:
    """A digest of a DIRECTORY, defined so it cannot be argued with.

    sha256 over the sorted relative POSIX paths, each followed by a NUL byte and by
    the sha256 DIGEST of that file's bytes; ``PIN.json`` itself is excluded because it
    carries the answer. Paths participate as well as contents, so a RENAME -- which
    changes which header an ``#include`` resolves to, and therefore what is compiled --
    moves the digest exactly as a content edit does.
    """
    digest = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "PIN.json")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest(), len(files)


def _gate_vendored() -> dict:
    """RULE 5: the vendored codegen inputs are the ones that were pinned.

    Runs BEFORE the compile, with rules 1 and 3, for the same reason they do: it is
    cheap, and a vendored header that drifted is not something to discover from a
    manifest diff after a multi-hour build.
    """
    pin_path = CUTLASS_DIR / "PIN.json"
    if not pin_path.exists():
        raise RuntimeError(
            f"CLOSED-WORLD RULE 5: {pin_path} is missing. A vendored library with no "
            f"pin is an unmeasured codegen input; restore the pin or remove the "
            f"vendored tree."
        )
    pin = json.loads(pin_path.read_text())
    here, count = vendored_tree_sha256(CUTLASS_DIR)
    if here != pin["tree_sha256"]:
        raise RuntimeError(
            "CLOSED-WORLD RULE 5: the vendored CUTLASS subtree does not match its "
            f"pin.\n  PIN.json: {pin['tree_sha256']} ({pin['file_count']} files)\n"
            f"  on disk : {here} ({count} files)\n"
            "A vendored header is code `ptxas` assembles, so this is a codegen change "
            "and not a packaging one. Re-vendor the pinned tag, or move the pin "
            "DELIBERATELY and re-ratify (`python tools/ratify.py --write`)."
        )
    print(f"rule 5 OK  cutlass {pin['tag']} ({pin['commit'][:12]}): {count} files, "
          f"sha256 {here[:16]}...")
    return pin


def _gate_toolchain(manifest: dict) -> str:
    pinned = manifest["toolchain"]["ptxas"]
    here = ptxas_version()
    if pinned != here:
        raise RuntimeError(
            "CLOSED-WORLD RULE 1: this manifest was ratified under a different "
            f"assembler.\n  manifest: {pinned}\n  this host: {here}\n"
            "Re-ratify (`python tools/ratify.py --write`) and record the new numbers "
            "-- do not hand-edit the manifest."
        )
    print(f"rule 1 OK  ptxas: {here}")
    return here


def _gate_sccache() -> dict | None:
    """Wire ``tools/sccache_nvcc.py`` in as ``PYTORCH_NVCC``, if the pinned cache is
    available (closed-world rule 1's extension, ``tools/sccache_toolchain.py``).

    Runs AFTER ``_gate_toolchain``, and points the wrapper at exactly the ``nvcc``
    that gate's ``ptxas --version`` check already resolved and pinned -- the wrapper
    itself never re-resolves it (see that module's docstring for why letting it do
    so would reopen the hole rule 1 exists to close).
    """
    info = sccache_toolchain.gate()
    if info is None:
        return None
    real_nvcc = cuda_bin("nvcc")
    wrapper, env = sccache_toolchain.wrap(real_nvcc)
    os.environ.update(env)
    os.environ["PYTORCH_NVCC"] = wrapper
    print(f"sccache: wired as PYTORCH_NVCC ({wrapper} -> {real_nvcc})")
    return info


# ---------------------------------------------------------------------------
# _build_config.py -- the self-describing binary
# ---------------------------------------------------------------------------
def write_build_config(archs: list[str], ptxas: str, manifest: dict, pin: dict,
                        sccache: dict | None) -> None:
    import torch

    config = {
        "version": read_version(),
        "archs": [f"sm_{a}" for a in archs],
        "ptxas": ptxas,
        "torch": torch.__version__,
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        #: TAKEN FROM `tools/ratify.py`, NEVER RESTATED. This digest identifies the
        #: ratification a binary was gated against, and the tier-2 census recomputes
        #: it independently and requires a match -- so a third spelling of "what goes
        #: into the hash" is a spelling that can disagree with both.
        "manifest_sha256": ratify.manifest_digest(manifest),
        "manifest_entries": {f"sm_{a}": sum(1 for k in manifest["entries"]
                                            if k.startswith(f"sm_{a}:"))
                             for a in archs},
        #: THE RATIFIED SHARD MEMBERSHIP, `{shard: members_sha256}`, taken from the
        #: manifest rather than recomputed. It is what the dispatch table's rows are
        #: joined against at runtime: a row whose shard membership is not the one that
        #: was MEASURED describes an arm nobody has register or spill numbers for, and
        #: `rola.planner.dispatch_table` filters it rather than planning against it.
        #: An installed wheel carries no `tools/manifests/`, so without this the join
        #: could only be made from a repository checkout -- which is the half of the
        #: closed world a user does not have.
        "shards": {name: block["members_sha256"]
                   for name, block in sorted(manifest["shards"].items())},
        "ptx_jit_fallback": False,
        #: THE VENDORED CODEGEN INPUTS, one entry per pinned library. Recorded next to
        #: `ptxas` because it answers the same question about the same binary: what
        #: assembled this, and from what sources.
        "vendored": {
            "cutlass": {"tag": pin["tag"], "commit": pin["commit"],
                        "tree_sha256": pin["tree_sha256"]},
        },
        #: THE COMPILER CACHE, recorded per the input-closure rule (closed-world
        #: rule 1's extension, `tools/sccache_toolchain.py`): any cache config that
        #: CAN affect output is part of the fingerprint, even though this one is
        #: measured transparent (docs/build.md#sccache) -- `None` when the build ran
        #: uncached, never omitted, so "was this binary's compile cached at all" is
        #: answerable from the wheel with no checkout.
        "sccache": sccache,
    }
    body = (
        '"""GENERATED by setup.py -- do not edit.\n\n'
        "Answers 'which measured evidence backs this exact .so?' from an installed\n"
        "wheel, with no repository checkout: `manifest_sha256` is the hash of the\n"
        "ratification this binary was gated against.\n"
        '"""\n\n'
        #: `pprint`, not `json.dumps`: this is a PYTHON module, and JSON spells the
        #: booleans `true`/`false`, which do not parse here.
        f"BUILD_CONFIG = {pprint.pformat(config, indent=4, sort_dicts=False, width=88)}\n\n\n"
        "def show():\n"
        '    """A copy, so a caller cannot mutate the record."""\n'
        "    return dict(BUILD_CONFIG)\n"
    )
    (ROOT / "rola" / "_build_config.py").write_text(body)
    print(f"wrote rola/_build_config.py (manifest sha256 "
          f"{config['manifest_sha256'][:16]}...)")


# ---------------------------------------------------------------------------
# the extension
# ---------------------------------------------------------------------------
#: THE HAND-WRITTEN TRANSLATION UNITS. The consumer's 512 kernel instantiations are
#: NOT here: they live in the generated shards under `csrc/rola/src/instantiations/`,
#: which `shard_sources()` appends. `consumer.cu` is now the dispatch and the torch
#: entry only, and compiles like the host file it is.
SOURCES = [
    "csrc/rola/rola_api.cpp",
    "csrc/rola/src/consumer.cu",
    # TRACK D (thunder §6b): the T=1 decode path. Two TUs, purely additive -- the tiled
    # consumer's sources are untouched and the decode kernels share no header with it.
    "csrc/rola/src/decode_expand.cu",
    "csrc/rola/src/decode.cu",
    "csrc/rola/src/entmax.cu",
    "csrc/rola/src/summaries.cu",
    "csrc/rola/src/paging/vmm_owner.cu",
    # P69: the CHUNK consumer (the ladder operator, adopted).  chunk.cu is
    # dispatch+entries only; its 24 instantiations live in the chunk_s* shards
    # (hand-carried until the generator learns the second matrix, stage D).
    "csrc/rola/src/chunk/chunk.cu",
    "csrc/rola/src/chunk/chunk_s0.cu",
    "csrc/rola/src/chunk/chunk_s1.cu",
    "csrc/rola/src/chunk/chunk_s2.cu",
]


def shard_sources() -> list[str]:
    """The generated instantiation shards, TAKEN FROM THE GENERATOR.

    Not globbed off the disk: a shard file that the matrix definition does not
    produce must not be able to enter the build by being dropped in the directory,
    and one it does produce must not be able to leave the build by being deleted.
    The `--check` gate below is what makes the two agree.
    """
    return [f"csrc/rola/src/instantiations/{name}.cu"
            for name in sorted(gen_shards.shards())]


def _gate_shards() -> None:
    """The checked-in shards are the ones the matrix definition produces.

    Runs BEFORE the compile, with the other cheap gates. Generated sources are
    committed so that what is reviewed is what compiles (closed-world); that promise
    is only worth anything if a stale generated file fails the build instead of
    silently shipping a matrix nobody wrote.
    """
    proc = subprocess.run([sys.executable, str(ROOT / "tools" / "gen_shards.py"), "--check"],
                          cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError(
            "the checked-in instantiation shards do not match the matrix definition "
            "in tools/gen_shards.py. Regenerate them (`python tools/gen_shards.py "
            "--write`) and review the diff -- a generated file that disagrees with "
            "its generator is a matrix nobody wrote."
        )


def build_extension():
    nvcc_threads = int(os.getenv("ROLA_NVCC_THREADS", "2"))
    os.environ.setdefault("MAX_JOBS", str(build_flags.default_max_jobs(nvcc_threads)))

    nvcc_flags = build_flags.nvcc_flags(ROLA_CUDA_ARCHS, nvcc_threads) \
        + list(build_flags.DEPFILE_FLAGS)
    assert_no_ptx(nvcc_flags)

    ldflags = ["-lcuda"]
    #: WSL puts the driver stub outside the toolkit's lib dir. A build knob, so it
    #: belongs to the build; the runtime (`rola/ops/_ext.py`) has none of this.
    wsl = Path("/usr/lib/wsl/lib")
    if (wsl / "libcuda.so").exists():
        ldflags.insert(0, f"-L{wsl}")

    return CUDAExtension(
        name=MODULE_NAME,
        sources=SOURCES + shard_sources(),
        #: The vendored CuTe/CUTLASS headers come LAST, so a repository header always
        #: wins a name collision and the vendored tree can never shadow ours.
        include_dirs=[str(CSRC / "src"), str(CUTLASS_DIR / "include")],
        extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": nvcc_flags},
        extra_link_args=ldflags,
    )


class RoLABuildExtension(BuildExtension):
    """``BuildExtension`` plus the closed-world gates, before and after."""

    def finalize_options(self):
        """Pin ``build_temp`` to a FIXED, repo-relative directory, overriding
        whatever ``pip``'s ``build_editable`` hook chose.

        ``pip install -e . --no-build-isolation`` runs the PEP 660 editable-build
        hook in a FRESH ``tempfile.mkdtemp()`` directory every single invocation
        -- verified: two consecutive installs of an UNCHANGED tree produced
        ``/tmp/tmpXXXXXXXX.build-temp`` with different random suffixes each time.
        Ninja's build graph (and therefore its OWN incremental-rebuild skip, and
        sccache's cache key -- both measured to include the invoking ``nvcc``
        process's CWD, ``docs/build.md#sccache``) lives inside that directory, so
        a random CWD every build defeats BOTH: ninja treats an unchanged source
        as new work because there is no ``build.ninja`` from last time to compare
        against, and sccache never hits across builds because the two ``nvcc``
        invocations of the identical TU ran from two different working
        directories. Pinning ``build_temp`` here fixes the object-file layer for
        the same reason multiple worktree builds already need SOURCE-content-only
        mangled names (P19, ``docs/build.md``'s path-trap section): a build
        artifact's identity must be a function of the tree's CONTENT, not of
        which ephemeral directory pip happened to allocate this time.
        """
        super().finalize_options()
        self.build_temp = str(ROOT / "build" / "temp")

    def run(self):
        manifest = _gate_ratified(ROLA_CUDA_ARCHS)
        ptxas = _gate_toolchain(manifest)
        pin = _gate_vendored()
        _gate_shards()
        sccache = _gate_sccache()
        super().run()
        self._post_build_fatbin_check()
        self._post_build_manifest_check()
        write_build_config(ROLA_CUDA_ARCHS, ptxas, manifest, pin, sccache)

    def _post_build_fatbin_check(self):
        """THE SHARDING ITSELF, PROVEN FROM THE BINARY.

        The two ways a translation-unit split silently fails to split anything --
        the launcher's definition leaking into the dispatch TU, or a query ladder
        still taking kernel addresses there -- both leave the build green, the
        census correct and the manifest byte-identical, while every kernel gets
        compiled twice and the compile time simply does not fall. `nm` cannot see
        it either: a `__global__` template's host stub is weak and the linker keeps
        one. The fatbin can: it carries one device entry per instantiating object.
        `tools/fatbin_entries.py` counts them, per arch, and checks each one against
        the shard the matrix says owns it.
        """
        so = next(Path(self.build_lib).rglob("_C*.so"), None) or \
            next((ROOT / "rola").glob("_C*.so"), None)
        if so is None:
            raise RuntimeError("built extension not found; the fatbin gate cannot "
                               "prove a binary it cannot read")
        cmd = [sys.executable, str(ROOT / "tools" / "fatbin_entries.py"), "--check", str(so)]
        if subprocess.run(cmd, cwd=str(ROOT)).returncode != 0:
            raise RuntimeError(
                "the built fatbin does not carry exactly one device entry per "
                "instantiation in the shard the matrix assigns it to. The binary is "
                "NOT shippable: either the shard partition and the dispatch have "
                "drifted apart, or the instantiations are being duplicated across "
                "translation units and nothing is actually sharded."
            )

    def _post_build_manifest_check(self):
        """The pre-build gates prove the INPUTS were ratified. This proves the
        codegen still matches what was measured: `tools/ratify.py` re-compiles the
        SHIPPED translation units -- the eight instantiation shards plus
        `decode.cu` and `entmax.cu`, under the same flag list this build uses
        (`tools/build_flags.py`) -- and fails on any spill growth or unhonored
        launch bound."""
        if os.getenv("ROLA_SKIP_POST_BUILD_RATIFY") == "1":
            print("post-build ratification SKIPPED (ROLA_SKIP_POST_BUILD_RATIFY=1)")
            return
        print("post-build: re-running the ratification gate ...")
        #: EXACTLY the archs being built, passed explicitly. The gate has its own
        #: default arch list, and a default that silently differed from
        #: ``ROLA_CUDA_ARCHS`` would ratify a fatbin nobody produced.
        cmd = [sys.executable, str(ROOT / "tools" / "ratify.py")]
        for a in ROLA_CUDA_ARCHS:
            cmd += ["--arch", a]
        proc = subprocess.run(cmd, cwd=str(ROOT))
        if proc.returncode != 0:
            raise RuntimeError(
                "post-build ratification FAILED: the built sources no longer match "
                "the committed manifest. The binary is NOT shippable. Re-ratify "
                "(`python tools/ratify.py --write`) only once the change is "
                "understood and the new numbers are recorded."
            )


setup(
    name="rola",
    version=read_version(),
    packages=find_packages(include=["rola", "rola.*"]),
    ext_modules=[build_extension()],
    cmdclass={"build_ext": RoLABuildExtension},
    zip_safe=False,
)
