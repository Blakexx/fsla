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
R4     a *launch* on a device whose arch was never measured                 ``csrc/rola/src/chunk/arch_caps.cuh`` (C++)
R5     a vendored codegen INPUT that does not match its recorded pin        ``_gate_vendored``
===== ==================================================================== ==========

**R5 AND WHY A HEADER-ONLY LIBRARY IS A CODEGEN INPUT, NOT A PACKAGING DETAIL.**
``csrc/third_party/cutlass`` vendors CUTLASS/CuTe headers. Header-only or not, every
line of it that the kernel instantiates is code ``ptxas`` assembles, so a silent edit
to a vendored header moves the register allocation the manifest ratified exactly as an
edit to ``chunk/chunk_kernel.cuh`` would. ``_gate_vendored`` therefore hashes the whole
vendored subtree and compares it against ``PIN.json``'s recorded digest BEFORE the
compile, in the same pass as rules 1 and 3, and the pin (upstream, tag, commit, digest)
is stamped into ``rola_cu13/_build_config.py`` so an installed wheel can answer "which
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

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc" / "rola"
CUTLASS_DIR = ROOT / "csrc" / "third_party" / "cutlass"

#: `tools/` is not an installed package -- it is the repository's instruments -- but
#: the build and the ratification gate must not maintain two copies of the codegen
#: flag list or two copies of the instantiation matrix. Both are imported from there.
sys.path.insert(0, str(ROOT / "tools"))
# These follow the sys.path insert above and cannot be hoisted into the import block
# at the top of the file.
# isort: off
import dev_config  # noqa: E402

#: torch.utils.cpp_extension resolves its toolkit when it is imported, so the dev config's is handed to it first.
if dev_config.get("toolchain.cuda_home"):
    os.environ["CUDA_HOME"] = dev_config.get("toolchain.cuda_home")
#: torch >= 2.10 prefixes any `ccache`/`sccache` on PATH to every compiler it runs; the only cache is the pinned one.
os.environ["TORCH_NO_COMPILER_WRAPPER"] = "1"
from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # noqa: E402
import build_flags  # noqa: E402
import build_record  # noqa: E402
import build_lock  # noqa: E402
import gen_shards  # noqa: E402
import kernels  # noqa: E402
import mold_toolchain  # noqa: E402
import ratify  # noqa: E402
import sccache_toolchain  # noqa: E402
import toolchains  # noqa: E402


# isort: on

#: THE RATIFIED SET. Overridable only downward in practice: adding an arch here
#: without a manifest fails the build at ``_gate_ratified`` (rule 3), which is the
#: point -- the supported matrix IS the ratified set, not a wish list.
def _declared_archs() -> list[str]:
    """The architectures this build compiles, from the environment or the manifests.

    ``ROLA_CUDA_ARCHS`` names them explicitly. Unset, they are read from the ratified
    manifests of this build's toolchain -- the same files rule 3 then gates each
    arch against, so the fatbin's arch list and the ratification's have ONE source.
    An unset variable used to default to a hard-coded pair, which is a second
    declaration of the supported matrix and the reason a build asked for one arch's
    worth of slots and compiled two arches' worth of ``cicc``.
    """
    env = os.getenv("ROLA_CUDA_ARCHS")
    if env is not None:
        return [a for a in env.split(";") if a]
    named = TOOLCHAIN.archs()
    if not named:
        raise RuntimeError(
            f"ROLA_CUDA_ARCHS is unset and toolchain {TOOLCHAIN.name} ratifies no "
            f"architecture ({TOOLCHAIN.manifest_dir}), so this build has no declared arch "
            "list. Name the arches (ROLA_CUDA_ARCHS=86) or ratify one first "
            "(tools/ratify.py --arch 86 --write)."
        )
    return named


#: THE TOOLCHAIN: the declared record naming the configured toolkit's assembler (tools/toolchains.py). A Python-only
#: install compiles nothing and resolves none.
TOOLCHAIN = None if os.getenv("ROLA_NO_EXTENSION") == "1" else toolchains.for_ptxas(ratify.ptxas_version())
ROLA_CUDA_ARCHS = [] if TOOLCHAIN is None else _declared_archs()

#: THE MILESTONE/CI SWITCH (K43, KERNEL_STANDARDS §16). Manifests go stale between
#: milestones BY DESIGN -- ratification is a per-milestone act, not a per-stage one --
#: so the POST-BUILD gate that compares the freshly built binary against the
#: checked-in ``tools/manifests/<toolchain>/sm_XX.json`` (``_post_build_manifest_check``'s
#: ``ratify.py`` re-run) REPORTS a mismatch instead of failing the install by default.
#: ``ROLA_STRICT_MANIFEST=1`` restores the pre-K43 hard failure: the milestone/CI gate
#: build that immediately precedes ``tools/ratify.py --write`` sets it (docs/build.md),
#: because THAT build must fail loudly if the tree it is about to ratify is not
#: actually shippable. This never weakens ``tools/ratify.py`` itself -- ``--write``/
#: ``--check`` invoked directly are untouched, as are every PRE-build gate below
#: (``_gate_ratified``/``_gate_toolchain``/``_gate_vendored``/``_gate_shards``: those
#: are prerequisites, not staleness) -- only whether THIS INSTALL treats a
#: stale-manifest finding as fatal.
ROLA_STRICT_MANIFEST = os.getenv("ROLA_STRICT_MANIFEST") == "1"


#: THE ARM SUBSET IS A FILE SUBSET (KERNEL_STANDARDS section 7, K36). Instantiation
#: cost is the dominant build resource, so an arm the build does not want is an arm
#: whose TRANSLATION UNIT is not in the build at all -- not a macro that still costs a
#: parse. ``ROLA_CARRY_ARMS=11`` (or ``11,12``, or ``all``) names rows of the carry
#: family's arm list by index (`tools/gen_shards.py`'s ``CARRY_ARMS``); each selected
#: row contributes its generated ``carry_arm_<i>.cu``, and the SELECTION -- which of
#: those rows the dispatch's ``CARRY_ARMS_X`` expands to -- is a GENERATED HEADER
#: (K36b, ``_write_carry_selection``), not a compiler flag: it used to be a
#: ``-DROLA_CARRY_BUILD_<i>`` that ``CUDAExtension`` applied to EVERY source's argv,
#: which is why an arm-set change forced ninja to rebuild every translation unit even
#: though only the per-arm TUs and the carry dispatch TU read those macros.
#:
#: THE DEFAULT IS EVERY DECLARED ROW (`tools/manifests/shipped_set.json`): each is built and shipped. A subset is an
#: iteration build. A binary is SELF-IDENTIFYING either way: ``rola.ops.carry.arms()`` lists exactly what was built and
#: every entry point refuses an arm the binary does not carry.
#: docs/build.md.
def _carry_arms() -> list[int]:
    return gen_shards.family_arms(gen_shards.CARRY, os.getenv("ROLA_CARRY_ARMS", ""), "ROLA_CARRY_ARMS")


def _decode_arms() -> list[int]:
    return gen_shards.family_arms(gen_shards.DECODE, os.getenv("ROLA_DECODE_ARMS", ""), "ROLA_DECODE_ARMS")


def _write_carry_selection() -> None:
    """Write `build/generated/carry_selection.inc` for THIS build's arm scope (K36b).

    Runs BEFORE the compile, next to `_gate_shards()` -- cheap, pre-compile,
    source-of-truth-for-what-gets-built. Only the per-arm TUs and the carry
    dispatch TU ever include the generated header, so
    writing it here (rather than passing `-D` on `nvcc_flags`) is what keeps an
    arm-set change from perturbing every other translation unit's argv.
    """
    gen_shards.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    text = gen_shards.carry_selection_header(_carry_arms())
    gen_shards.write_if_changed(gen_shards.GENERATED_DIR / gen_shards.CARRY_SELECTION_INC, text)


def _built_extension(build_lib: str) -> Path | None:
    """The `_C` extension this build produced, found under its own output tree."""
    found = sorted(Path(build_lib).glob(f"{PLUGIN}/_C.*so"))
    return found[0] if found else None


def _carry_parts_spec() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The carry kernel's real and trivial parts for this build (`gen_shards.carry_parts_spec`): every part unless
    `ROLA_CARRY_PARTS` names a subset."""
    return gen_shards.carry_parts_spec(os.getenv("ROLA_CARRY_PARTS", "all"), "ROLA_CARRY_PARTS")


def _carry_parts() -> tuple[str, ...]:
    return _carry_parts_spec()[0]


def _write_carry_parts() -> None:
    """Write `build/generated/carry_parts.inc` on EVERY build, so a stubbed mask never
    outlives the build that asked for it."""
    gen_shards.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    gen_shards.write_if_changed(gen_shards.GENERATED_DIR / gen_shards.CARRY_PARTS_INC,
                                gen_shards.carry_parts_header(*_carry_parts_spec()))


def _write_csrc_stamp() -> None:
    """Write `build/generated/csrc_stamp.inc` for THIS tree (the device stamp's key).

    Beside `_write_carry_selection()`, pre-compile, and from `tools/ratify.py`'s own
    function so the stamp this build compiles is the stamp that gate assembles. It is
    a generated HEADER rather than a `-D` for the same reason the arm selection is:
    the define would reach every translation unit's argv, so every `csrc/` edit would
    rebuild the whole extension instead of the stamp sites.
    """
    ratify.write_csrc_stamp()


def _carry_sources() -> list[str]:
    """The per-arm translation units this build compiles, TAKEN FROM THE GENERATOR.

    Two rows that are ONE kernel share a file (the generator's `carry_kernel_key`), so
    this is a set and not a list comprehension over the indices.
    """
    return [str(p.relative_to(ROOT)) for p in kernels.arm_units(_carry_arms())]


def _is_iteration_build() -> bool:
    """A build whose arm set is not the SHIPPED set is an ITERATION build.

    The post-build gate proves SHIPPABILITY -- that the fatbin carries every ratified
    instantiation. A binary built with a different arm set carries a different set BY
    CONSTRUCTION, so that gate would fail for the one reason that is not a defect. It
    is skipped here and nowhere else, behind a banner, and the binary stays
    self-identifying.

    NOTE the asymmetry, and that it is deliberate: a build carrying MORE than the
    shipped rows (the battery's `all`) is an iteration build too. The manifest ratifies
    the shipped set; a binary carrying arms it does not ratify is not the thing that
    ships, whichever direction it differs in.

    A build whose ARCHITECTURES are not the toolchain's ratified set is one too (the
    single-arch iteration build, `ROLA_CUDA_ARCHS=86`): it cannot ship either, so it
    takes no exclusive budget draw and re-runs no ratify, whose only possible outcome
    there is a warning (2026-11-01, docs/build.md#iteration).
    """
    return gen_shards.is_iteration(_carry_arms(), _decode_arms(), *_carry_parts_spec(), ROLA_CUDA_ARCHS,
                                   TOOLCHAIN.archs())


def _arch_subset_active() -> bool:
    """This build's architectures differ from the toolchain's ratified set."""
    return set(ROLA_CUDA_ARCHS) != set(TOOLCHAIN.archs())


#: THE ITERATION ARM SUBSET FOR DECODE (KERNEL_STANDARDS section 7, D1b). The
#: mechanism of ``_carry_arms``, one family over: ``ROLA_DECODE_ARMS=5`` (or
#: ``5,13``) names rows of the decode family's declared arm list by index
#: (``tools/manifests/decode_set.json``, the ``d_v x D x decay`` matrix in that order),
#: written to its generated selection header (``_write_decode_selection``). The DEFAULT IS EVERY DECLARED ARM, so a plain build -- and
#: every gate build -- is the closed-world one. A subset binary is SELF-IDENTIFYING:
#: ``rola.ops.decode.arms()`` lists exactly what was built and the launch refuses an
#: arm the binary does not carry. Iteration builds pair it with
#: ``ROLA_CUDA_ARCHS=86``; docs/build.md.
def _decode_arm_subset_active() -> bool:
    return set(_decode_arms()) != set(range(len(gen_shards.DECODE_ARMS)))


def _write_decode_selection() -> None:
    """Write `build/generated/decode_selection.inc` for THIS build's decode arm scope, as the carry's
    (`_write_carry_selection`): included by `decode.cu` alone, never a flag on every unit's argv."""
    gen_shards.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    text = gen_shards.decode_selection_header(_decode_arms())
    gen_shards.write_if_changed(gen_shards.GENERATED_DIR / gen_shards.DECODE.header, text)


#: THE EXTENSION LIVES IN THE TOOLCHAIN'S BINARY PLUGIN, ``rola_cu13`` (wheel naming D,
#: docs/build.md#wheels): ``rola`` is pure Python, and the binary, its build record and its
#: manifests ship as ``rola-cu13``. A dotted name makes setuptools place the binary inside
#: the plugin's package in an in-place build and in a wheel alike, so a checkout imports it
#: by the path an install does, and the repo root stays clean.
#:
#: This is a PACKAGING name: the registration TU's ``PyInit__C`` is the one place csrc
#: spells its last component, and the operators it registers live in ``torch.ops.rola``;
#: the C++ namespaces (``rola``, ``rola::paging``) and every kernel symbol are untouched.
#: The manifests key on MANGLED KERNEL NAMES, and the module-init symbol they do not
#: contain is the only thing this name can move.
PLUGIN, EXTRA = "rola_cu13", "cu13"
if TOOLCHAIN is not None and TOOLCHAIN.name != EXTRA:
    raise RuntimeError(f"toolchain {TOOLCHAIN.name} has no binary plugin; this tree ships {PLUGIN} alone")
MODULE_NAME = f"{PLUGIN}._C"

#: THE STABLE ABI TARGET (`tools/build_flags.py`'s `TORCH_MIN`, docs/build.md#stable-abi): every compile targets it, and
#: `CUDAExtension(py_limited_api=True)` builds the module object against Python's limited API, defining the flag itself.
STABLE_DEFINES = list(build_flags.STABLE_DEFINES)


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------
def read_version() -> str:
    """From ``rola/__init__.py`` (`tools/build_record.py`'s `version`): the tag must not be able to disagree with the
    shipped ``__version__``, which is also stamped next to the manifest hashes in ``_build_config.py``."""
    return build_record.version(ROOT / "rola" / "__init__.py")


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
cuda_bin = dev_config.cuda_bin


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
    return TOOLCHAIN.manifest_path(arch)


def _gate_ratified(archs: list[str]) -> dict:
    """RULE 3 (`tools/build_record.py`'s `ratified_manifest`): every arch in the fatbin has a ratified manifest, and
    they agree; the merged record is what ``manifest_sha256`` describes."""
    return build_record.ratified_manifest(archs, TOOLCHAIN.manifest_dir)


def _gate_vendored() -> dict:
    """RULE 5 (`tools/build_record.py`'s `vendored_pin`): the vendored codegen inputs are the ones that were pinned.
    Runs BEFORE the compile, with rules 1 and 3: it is cheap, and a vendored header that drifted is not something to
    discover from a manifest diff after a long build."""
    return build_record.vendored_pin(CUTLASS_DIR)


def _gate_toolchain(manifest: dict) -> str:
    import torch

    pinned = toolchains.normalize(manifest["toolchain"]["ptxas"])
    here = ptxas_version()
    if pinned != here or here != TOOLCHAIN.ptxas:
        raise RuntimeError(
            f"CLOSED-WORLD RULE 1: toolchain {TOOLCHAIN.name}'s manifests were ratified under a "
            f"different assembler.\n  manifest: {pinned}\n  this host: {here}\n"
            f"Re-ratify (`python tools/ratify.py --write`) and record the new numbers "
            "-- do not hand-edit the manifest."
        )
    toolchains.torch_pairing(TOOLCHAIN, torch.version.cuda)
    print(f"rule 1 OK  toolchain {TOOLCHAIN.name}, torch CUDA {torch.version.cuda}, ptxas: {here}")
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
    wrapper, env = sccache_toolchain.wrap(real_nvcc, info["path"])
    os.environ.update(env)
    os.environ["PYTORCH_NVCC"] = wrapper
    print(f"sccache: wired as PYTORCH_NVCC ({wrapper} -> {real_nvcc})")
    return info


# ---------------------------------------------------------------------------
# _build_config.py -- the self-describing binary
# ---------------------------------------------------------------------------
def write_build_config(archs: list[str], ptxas: str, manifest: dict, pin: dict,
                        sccache: dict | None) -> Path:
    """The build record (`tools/build_record.py`): the self-describing binary."""
    import torch

    record = ROOT / PLUGIN / "_build_config.py"
    record.write_text(build_record.render(build_record.config(
        version=read_version(), archs=archs, iteration=_is_iteration_build(), toolchain=TOOLCHAIN.name, ptxas=ptxas,
        torch=torch.__version__, cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI), manifest=manifest, cutlass_pin=pin,
        sccache=sccache)))
    print(f"wrote {PLUGIN}/_build_config.py")
    return record


# ---------------------------------------------------------------------------
# the extension
# ---------------------------------------------------------------------------
#: THE TRANSLATION UNITS, DERIVED: the host binding, then every hand-written `.cu` under `csrc/rola/src`
#: (`tools/kernels.py`, the one list the build, the ratification and the gates share). The per-arm carry units
#: are appended by the arm selection.
SOURCES = ["csrc/rola/rola_api.cpp"] + [str(p.relative_to(ROOT)) for p in kernels.hand_units()]


def _gate_shards() -> None:
    """The checked-in shards are the ones the arm list produces.

    Runs BEFORE the compile, with the other cheap gates. Generated sources are
    committed so that what is reviewed is what compiles (closed-world); that promise
    is only worth anything if a stale generated file fails the build instead of
    silently shipping an arm list nobody wrote.
    """
    proc = subprocess.run([sys.executable, str(ROOT / "tools" / "gen_shards.py"), "--check"],
                          cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError(
            "the checked-in instantiation shards do not match the arm list "
            "in tools/gen_shards.py. Regenerate them (`python tools/gen_shards.py "
            "--write`) and review the diff -- a generated file that disagrees with "
            "its generator is an arm list nobody wrote."
        )


def _ninja_jobs(held_slots: int) -> int:
    """Ninja's job count, so the concurrent ``cicc`` count is the slots held.

    A slot of the host budget (``rola_devtools.locks.host``) is one ``cicc``, and one ``nvcc`` runs one
    ``cicc`` PER ARCHITECTURE, up to ``--threads``: the two parallelism axes multiply,
    so the job count is the budget divided by that per-process factor. Setting ninja's
    ``-j`` to the slot count directly oversubscribes the budget by exactly the number
    of architectures compiled.
    """
    per_process = min(dev_config.get("host.nvcc_threads"), max(1, len(ROLA_CUDA_ARCHS)))
    return max(1, held_slots // per_process)


def _link_flags() -> list[str]:
    ldflags = ["-lcuda"]
    #: WSL puts the driver stub outside the toolkit's lib dir. A build knob, so it
    #: belongs to the build; the runtime (`rola/ops/_ext.py`) has none of this.
    wsl = Path("/usr/lib/wsl/lib")
    if (wsl / "libcuda.so").exists():
        ldflags.insert(0, f"-L{wsl}")
    #: mold is THE linker, unconditionally (Blake ruling, 2026-08-28): raises if
    #: the pin (tools/mold_pin.json) is not satisfied -- no fallback to
    #: ld.bfd/ld.gold. See tools/mold_toolchain.py's module docstring.
    ldflags += mold_toolchain.link_flags()
    return ldflags


#: THE PART HARNESS'S DRIVER (`measure/harness/bench_carry_parts.py`), built AHEAD OF TIME
#: beside the arm under `ROLA_BUILD_PARTS=1` -- the same pass, the same lock, the same cache
#: -- where a torch JIT `load()` of the same three sources ran for minutes and was killed
#: by the host's memory watchdog (KERNEL_STANDARDS §22: a gating instrument that cannot
#: run is a defect). Never shipped: an iteration-build instrument, `-lineinfo` for the
#: ledgers. docs/build.md#parts
PARTS_MODULE_NAME = f"{PLUGIN}._C_parts"
PARTS_SOURCES = ["measure/harness/carry_parts/carry_parts.cu",
                 "csrc/rola/src/common/arch_runtime.cu", "csrc/rola/src/common/geom.cu"]


def build_parts_extension():
    nvcc_threads = dev_config.get("host.nvcc_threads")
    nvcc_flags = build_flags.nvcc_flags(ROLA_CUDA_ARCHS, nvcc_threads) \
        + list(build_flags.DEPFILE_FLAGS) + ["-lineinfo"]
    assert_no_ptx(nvcc_flags)
    return CUDAExtension(
        name=PARTS_MODULE_NAME,
        sources=PARTS_SOURCES,
        include_dirs=[str(CSRC / "src"), str(CUTLASS_DIR / "include"), str(ROOT / "build" / "generated")],
        extra_compile_args={"cxx": list(build_flags.HOST_CXX_FLAGS), "nvcc": nvcc_flags + STABLE_DEFINES},
        extra_link_args=_link_flags(),
        py_limited_api=True,
    )


def desired_jobs() -> int:
    """The ninja jobs a build asks the host budget for: `host.build_jobs`, else derived from the `cicc` a job runs --
    one an architecture up to `host.nvcc_threads` (`_ninja_jobs`' per-process factor), so a single-arch build asks for
    the jobs a single-arch compile can run."""
    per_process = min(dev_config.get("host.nvcc_threads"), max(1, len(ROLA_CUDA_ARCHS)))
    return dev_config.get("host.build_jobs") or build_flags.default_max_jobs(per_process)


def build_extension():
    nvcc_threads = dev_config.get("host.nvcc_threads")

    nvcc_flags = build_flags.nvcc_flags(ROLA_CUDA_ARCHS, nvcc_threads) \
        + list(build_flags.DEPFILE_FLAGS)
    assert_no_ptx(nvcc_flags)

    ldflags = _link_flags()

    #: The generated selection header's directory (K36b). A FIXED path, added to
    #: every build regardless of arm selection -- it does not itself vary per-arm-set
    #: and so does not perturb other TUs' argv/cache-key any more than the CUTLASS
    #: include dir already does. Must exist before `CUDAExtension(...)` is
    #: constructed: nvcc/gcc get handed `-I<dir>` at configure time even on a
    #: completely fresh checkout, before `_write_carry_selection` ever runs.
    generated_dir = ROOT / "build" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    return CUDAExtension(
        name=MODULE_NAME,
        sources=SOURCES + _carry_sources(),
        #: The vendored CuTe/CUTLASS headers come LAST, so a repository header always
        #: wins a name collision and the vendored tree can never shadow ours.
        include_dirs=[str(CSRC / "src"), str(CUTLASS_DIR / "include"), str(generated_dir)],
        extra_compile_args={"cxx": list(build_flags.HOST_CXX_FLAGS), "nvcc": nvcc_flags + STABLE_DEFINES},
        extra_link_args=ldflags,
        py_limited_api=True,
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
        _write_carry_selection()
        _write_decode_selection()
        _write_carry_parts()
        gen_shards.write_builds(ROLA_CUDA_ARCHS)
        _write_csrc_stamp()
        sccache = _gate_sccache()
        #: THE MACHINE-WIDE CICC BUDGET (KERNEL_STANDARDS §14 addendum 2, Blake
        #: ruling 2026-08-28-night): TOOL-ENFORCED here so no brief, no
        #: worktree and no agent can forget it -- see tools/build_lock.py's
        #: module docstring for why this cannot stay a documentation-only
        #: convention. A gate build (the shipped arm set) takes the WHOLE
        #: budget; an iteration build takes whatever share is free right now
        #: and MAX_JOBS is clamped to that share, never to what was asked.
        #: `_is_iteration_build` covers BOTH families (ROLA_CARRY_ARMS,
        #: ROLA_DECODE_ARMS) -- one mechanism, not two (§18).
        with build_lock.acquire(gate=not _is_iteration_build(),
                                 desired=desired_jobs()) as held_slots:
            os.environ["MAX_JOBS"] = str(_ninja_jobs(held_slots))
            super().run()
        self._post_build_arm_table_check()
        self._post_build_sass_gate()
        self._post_build_manifest_check()
        record = write_build_config(ROLA_CUDA_ARCHS, ptxas, manifest, pin, sccache)
        #: A wheel's build tree took the package's files before this record was written; it ships this build's.
        if not self.inplace:
            shutil.copy(record, Path(self.build_lib) / PLUGIN / record.name)

    def _iteration_banner(self) -> bool:
        if not _is_iteration_build():
            return False
        print("=" * 78)
        if set(_carry_arms()) != set(range(len(gen_shards.CARRY_ARMS))):
            print("ITERATION BUILD -- NOT SHIPPABLE.  This build carries carry arms "
                  f"{_carry_arms()}, not every declared arm "
                  f"{list(range(len(gen_shards.CARRY_ARMS)))}, so the shippability gate "
                  "(the manifest) is skipped. Build with "
                  "ROLA_CARRY_ARMS unset to get a gateable binary.")
        if set(_carry_parts()) != set(gen_shards.CARRY_PARTS):
            print("ITERATION BUILD -- NOT SHIPPABLE.  ROLA_CARRY_PARTS stubs "
                  f"{sorted(set(gen_shards.CARRY_PARTS) - set(_carry_parts()))} of the carry kernel "
                  "(a composer measurement build: the stubbed parts keep their HMMAs and drop "
                  "their other work). Build with ROLA_CARRY_PARTS unset to get the kernel.")
        if _arch_subset_active():
            print("ITERATION BUILD -- NOT SHIPPABLE.  This build compiles architectures "
                  f"{sorted(ROLA_CUDA_ARCHS)}, not the toolchain's ratified set "
                  f"{sorted(TOOLCHAIN.archs())}, so the post-build ratify is skipped (the SASS "
                  "gate, the arm table and the device stamp still run). Build with "
                  "ROLA_CUDA_ARCHS unset to get a gateable binary.")
        if _decode_arm_subset_active():
            print("ITERATION BUILD -- NOT SHIPPABLE.  ROLA_DECODE_ARMS is set, so this "
                  "fatbin carries a SUBSET of the decode arms, so the shippability "
                  "gate (the manifest) is skipped. "
                  "rola.ops.decode.arms() lists what it does carry. Build with "
                  "ROLA_DECODE_ARMS unset to get a gateable binary.")
        print("=" * 78)
        return True

    def _post_build_arm_table_check(self):
        """The built arm table against the DECLARED SHIPPED SET, and the coverage sweep (`gen_shards`'
        `check_built_arm_table` and `sweep_binary`). FATAL WHATEVER `ROLA_STRICT_MANIFEST` SAYS: that switch is about
        manifest staleness, and none of these is a measurement -- a binary whose arm set disagrees with the
        declaration is not stale, it is wrong."""
        want = _carry_arms()
        header = gen_shards.GENERATED_DIR / gen_shards.CARRY_SELECTION_INC
        bad = gen_shards.check_built_arm_table(header.read_text() if header.exists() else None, want)
        if bad:
            raise RuntimeError("the built arm table: " + "\n".join(bad))
        so = _built_extension(self.build_lib)
        if so is None:
            raise RuntimeError(f"no {PLUGIN}/_C.*so under {self.build_lib} after the build; the arm coverage sweep has "
                               "no artifact to ask")
        #: ABSOLUTE and outside the checkout: a wheel's `build_lib` is relative to it, and `import rola` there would
        #: resolve the source tree instead of the build
        swept = gen_shards.sweep_binary(sys.executable, so.resolve(), want, cwd=ROOT.parent)
        print(f"arm table OK  {len(want)} built of {len(gen_shards.CARRY_ARMS)} "
              f"declared, keyed {list(gen_shards.ARM_KEY)}; {swept} arm(s) swept")

    def _post_build_sass_gate(self):
        """KERNEL_STANDARDS §20 on the binary this build produced: `tools/sass_gate.py` reads every function of every
        cubin (local memory, stack frames, convergence subroutines, predicated HMMA) and reds any kernel the source
        defines that no cubin carries. It compares against no manifest, so it runs on EVERY build, iteration builds
        included, and a red is fatal: a binary with a spill is not a binary to measure."""
        so = _built_extension(self.build_lib)
        if so is None:
            raise SystemExit("post-build SASS gate: no `_C` extension under the build tree to read")
        proc = subprocess.run([sys.executable, str(ROOT / "tools" / "sass_gate.py"), str(so)], cwd=str(ROOT),
                              capture_output=True, text=True)
        lines = proc.stdout.splitlines()
        reds = [line for line in lines if line.startswith("RED")]
        read = sum(1 for line in lines if line.startswith(("ok", "RED ")))
        print(f"post-build: the SASS gate read {read} function(s), {len(reds)} red")
        if proc.returncode != 0:
            print("\n".join(reds) or proc.stdout + proc.stderr)
            raise SystemExit("post-build SASS gate FAILED (KERNEL_STANDARDS §20): the reds are printed above")

    def _post_build_manifest_check(self):
        """The pre-build gates prove the INPUTS were ratified. This proves the
        codegen still matches what was measured: `tools/ratify.py` re-compiles the
        SHIPPED translation units -- every unit the build compiles (`tools/kernels.py`),
        under the same flag list this build uses (`tools/build_flags.py`) -- and fails
        on any spill growth or unhonored launch bound.

        EVERY one of `ratify.py`'s checks (entry set, SASS-body hashes, the
        hot-loop/local-presence ratchets) is a comparison against the COMMITTED
        `tools/manifests/<toolchain>/sm_XX.json`, and that manifest is allowed to be stale
        between milestones by design (KERNEL_STANDARDS §16) -- unlike the fatbin
        gate above, there is no sub-check here that is independent of the manifest,
        so the K43 split is a single blanket one: FAIL is fatal only under
        `ROLA_STRICT_MANIFEST=1` (set by the milestone/CI gate build that precedes
        `ratify.py --write`, docs/build.md); otherwise it is a REPORT and the
        install proceeds. This never touches `ratify.py` itself -- `--write`/
        `--check` invoked directly still behave exactly as before.
        """
        #: THE SKIP ANNOUNCES ITSELF. An iteration build is the one that skips this gate, so it is the one build
        #: whose operator has to be told the binary is not shippable -- the banner was written for exactly this
        #: return and nothing called it, so the skip was silent (found by the dead-code lint, 2026-09-15).
        if self._iteration_banner():
            return
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
            message = (
                "post-build ratification FAILED (see the entry/body counts printed "
                "above): the built sources no longer match the committed manifest. "
                "Re-ratify (`python tools/ratify.py --write`) only once the change is "
                "understood and the new numbers are recorded."
            )
            if ROLA_STRICT_MANIFEST:
                raise RuntimeError(message + " The binary is NOT shippable.")
            print("\n" + "=" * 78)
            print("WARNING -- post-build ratification FAILED (counts printed above "
                  "this banner). Install proceeding because ROLA_STRICT_MANIFEST is "
                  "not set: manifests go stale between milestones by design "
                  "(KERNEL_STANDARDS §16). This binary is NOT certified shippable -- "
                  "re-run with ROLA_STRICT_MANIFEST=1 (the milestone/CI gate build "
                  "always does) before trusting it, or `python tools/ratify.py "
                  "--write` if the drift is understood and intentional.")
            print("=" * 78 + "\n")


#: THE EXTRAS, here and not in pyproject.toml because `cu13` names this version: `pip install "rola[cu13]"` installs the
#: binary plugin built at exactly this `rola` (docs/build.md#wheels), and `rola.ops._ext` refuses any other.
EXTRAS = {
    EXTRA: [f"{PLUGIN.replace('_', '-')}=={read_version()}"],
    #: `pytest-benchmark` was REMOVED, measured: it drove the latency gate's repetition, and its round schedule -- not
    #: the kernel -- set the number the gate read once the committed spread narrowed to a device-event IQR.
    #: `rola_results` holds the record now (docs/measurement.md). THE ATTENTION REFERENCE every reported carry number
    #: sits beside is rola-bench's arm (torch's own flash backend, docs/measurement.md), so it needs no package here.
    "bench": ["matplotlib", "pandas"],
    #: `entmax` (DeepSPIN) is the fp64 REFERENCE the routing gates are anchored to -- the paper authors' own
    #: implementation, pure python with torch as its only dependency. A TEST dependency and never a runtime one:
    #: `rola/routing/entmax/production.py` is RoLA's own hand-CUDA producer and imports nothing from it, which
    #: `tests/integration/test_production_levels.py` asserts rather than assumes.
    #:
    #: PINNED to `==1.3`: `rola/routing/entmax/PIN.json` records the version AND a digest of the installed package, and
    #: `rola/routing/entmax/reference.py` refuses to import (`pin.gate()`) an install that drifted from either. This `==`
    #: is the coarse half of that guard -- the resolver refuses the bump before pip finishes -- and the pin is the fine
    #: half, because the resolver cannot see an editable or re-published install that keeps the version string. See
    #: `PIN.json`'s `why_this_is_the_only_guard`: the dual-run protocol imports the same installed package on both its
    #: sides and cannot see this drift itself.
    "test": ["pytest", "pytest-xdist", "entmax==1.3"],
    "dev": ["pytest", "ruff==0.14.10", "pre-commit==4.6.2", "vulture==2.16", "clang-format==19.1.7",
            "ast-grep-cli==0.45.2"],
}


setup(
    name="rola",
    version=read_version(),
    packages=find_packages(include=["rola", "rola.*", PLUGIN]),
    #: entmax's pin record, read beside `rola/routing/entmax/pin.py`: without it an installed reference cannot gate itself.
    package_data={"rola.routing.entmax": ["PIN.json"]},
    extras_require=EXTRAS,
    #: ROLA_NO_EXTENSION=1 installs the Python package with NO extension build -- for a
    #: worktree that receives a built _C from the integration checkout at the same
    #: SHA (`python tools/dev.py worktree`); the device stamp still gates freshness.
    ext_modules=[] if os.getenv("ROLA_NO_EXTENSION") == "1" else (
        [build_extension()] + ([build_parts_extension()] if os.getenv("ROLA_BUILD_PARTS") == "1" else [])),
    cmdclass={"build_ext": RoLABuildExtension},
    zip_safe=False,
)
