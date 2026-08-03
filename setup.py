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
===== ==================================================================== ==========

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
def gencodes(archs: list[str]) -> list[str]:
    """``code=sm_XX`` ONLY.

    A ``code=compute_XX`` entry would embed PTX in the fatbin and let the DRIVER
    just-in-time compile this kernel for an architecture whose register and spill
    behaviour was never measured -- precisely the failure the manifest exists to
    exclude. There is no environment variable that turns this on.
    """
    flags: list[str] = []
    for a in archs:
        flags.append(f"-gencode=arch=compute_{a},code=sm_{a}")
    return flags


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
            toolchain = blob["toolchain"]
        elif blob["toolchain"] != toolchain:
            raise RuntimeError(
                f"CLOSED-WORLD RULE 1: {path.name} pins a different assembler than the "
                f"other manifests. The per-arch split is a FILING decision; one fatbin "
                f"may not be gated against two ratifications."
            )
        entries.update(blob["entries"])
        print(f"rule 3 OK  sm_{a}: {len(blob['entries'])} ratified instantiations "
              f"({path.name})")
    return {"toolchain": toolchain, "entries": entries}


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


# ---------------------------------------------------------------------------
# MAX_JOBS -- a MEASURED constant, not a guess
# ---------------------------------------------------------------------------
def default_max_jobs(nvcc_threads: int) -> int:
    """RAM, not cores, is the binding resource here, and that is a measurement:
    ``MAX_JOBS=8`` was OOM-KILLED on this 23 GB / 16-core host after 8m32 (``ninja``
    reported ``Killed``; ``sys`` time 28m47 -- the machine was thrashing, not
    compiling). So the memory term participates rather than the core count alone."""
    import psutil

    cpus = os.cpu_count() or 1
    free_gb = psutil.virtual_memory().available / (1024 ** 3)
    return max(1, int(min(cpus // 2, free_gb / (5 * nvcc_threads))))


# ---------------------------------------------------------------------------
# _build_config.py -- the self-describing binary
# ---------------------------------------------------------------------------
def write_build_config(archs: list[str], ptxas: str, manifest: dict) -> None:
    import torch

    blob = json.dumps({"toolchain": manifest["toolchain"],
                       "entries": manifest["entries"]}, sort_keys=True).encode()
    config = {
        "version": read_version(),
        "archs": [f"sm_{a}" for a in archs],
        "ptxas": ptxas,
        "torch": torch.__version__,
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "manifest_sha256": hashlib.sha256(blob).hexdigest(),
        "manifest_entries": {f"sm_{a}": sum(1 for k in manifest["entries"]
                                            if k.startswith(f"sm_{a}:"))
                             for a in archs},
        "ptx_jit_fallback": False,
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
SOURCES = [
    "csrc/rola/rola_api.cpp",
    "csrc/rola/src/consumer.cu",
    "csrc/rola/src/entmax.cu",
    "csrc/rola/src/summaries.cu",
    "csrc/rola/src/paging/page_plan.cu",
    "csrc/rola/src/paging/page_plan_host.cu",
    "csrc/rola/src/paging/vmm_owner.cu",
]


def build_extension():
    nvcc_threads = int(os.getenv("ROLA_NVCC_THREADS", "2"))
    os.environ.setdefault("MAX_JOBS", str(default_max_jobs(nvcc_threads)))

    arch_flags = gencodes(ROLA_CUDA_ARCHS)
    assert_no_ptx(arch_flags)

    nvcc_flags = [
        "-O3", "-std=c++17", "-lineinfo", f"--threads={nvcc_threads}",
        #: MANDATORY, not optional: `tools/ratify.py` parses exactly this report.
        "--ptxas-options=-v",
    ] + arch_flags

    ldflags = ["-lcuda"]
    #: WSL puts the driver stub outside the toolkit's lib dir. A build knob, so it
    #: belongs to the build; the runtime (`rola/ops/_ext.py`) has none of this.
    wsl = Path("/usr/lib/wsl/lib")
    if (wsl / "libcuda.so").exists():
        ldflags.insert(0, f"-L{wsl}")

    return CUDAExtension(
        name=MODULE_NAME,
        sources=SOURCES,
        include_dirs=[str(CSRC / "src")],
        extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": nvcc_flags},
        extra_link_args=ldflags,
    )


class RoLABuildExtension(BuildExtension):
    """``BuildExtension`` plus the closed-world gates, before and after."""

    def run(self):
        manifest = _gate_ratified(ROLA_CUDA_ARCHS)
        ptxas = _gate_toolchain(manifest)
        super().run()
        self._post_build_manifest_check()
        write_build_config(ROLA_CUDA_ARCHS, ptxas, manifest)

    def _post_build_manifest_check(self):
        """The pre-build gates prove the INPUTS were ratified. This proves the
        codegen still matches what was measured: `tools/ratify.py` re-compiles the
        probe -- the whole 512-instantiation dispatch closure, under the same flags
        -- and fails on any spill growth or unhonored launch bound."""
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
