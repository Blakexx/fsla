"""THE ONE SOURCE OF THE nvcc FLAG LIST.

**WHY THIS FILE EXISTS.** Two programs compile the same translation units. The graph
compiles them into the shipped library (`bazel/flags.py` writes its flags from this file);
`tools/ratify.py`'s compile path compiles them to read `ptxas`'s register and spill report
for an architecture's first manifest. A register/spill profile
is a property of the FLAGS as much as of the source, so if those two flag lists were
maintained independently -- as they were, one restating the other in a comment that
said it must not -- a divergence would produce ratified numbers that describe a binary
nobody ships. That is not a style problem; it is the closed-world codegen policy's own
failure mode. The lists live here, once, and both importers take them.

**WHAT IS AND IS NOT HERE.** The CODEGEN-AFFECTING flags: optimization level,
language standard, the `-gencode` targets, the preprocessor defines torch's
`CUDAExtension` adds, and `--threads`. Not here: link flags, the C++ host compiler's
own flags, and include directories that only one of the two callers can know (torch's
own header locations are resolved by `include_paths` at the call site).

`--ptxas-options=-v` is included and is MANDATORY: `tools/ratify.py` parses exactly
that report. It changes what the assembler PRINTS, not what it emits.

REFUSED, each a codegen change that would arrive as a packaging change: the
`-U__CUDA_NO_HALF_OPERATORS__` / `-U__CUDA_NO_HALF_CONVERSIONS__` /
`-U__CUDA_NO_BFLOAT16_CONVERSIONS__` undefines and `--expt-extended-lambda`, which would
undo torch's own defines and enable a language feature the measured build did not have;
and `--use_fast_math`, since the tolerances are fp32 accumulation tolerances against an
fp64 oracle, and fast-math silently changes the numeric contract they were derived under.
"""
from __future__ import annotations

import os

#: Torch's `COMMON_NVCC_FLAGS`, restated ONCE. `CUDAExtension` adds exactly this set
#: to the build, so a standalone compile of the same file must pass it too or it is
#: compiling a different program.
COMMON_NVCC_FLAGS = (
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
)


def torch_defines() -> list[str]:
    """The preprocessor defines torch's `CUDAExtension` adds, ABI included.

    The ABI define is READ FROM TORCH rather than written down. It is the one define
    in the set whose value is a property of the INSTALLED torch and not of this
    project: a hardcoded `=0` compiled against a `=1` build would produce a
    ratification measured on a program the extension does not link, and it would do
    it silently, because the two agree about every other flag.
    """
    import torch

    abi = int(bool(torch._C._GLIBCXX_USE_CXX11_ABI))
    return [f"-D_GLIBCXX_USE_CXX11_ABI={abi}", *COMMON_NVCC_FLAGS]


def extension_defines() -> list[str]:
    """The defines torch's `CUDAExtension` gives every source of a `py_limited_api` extension, the extension `_C`, and
    this project's stable-ABI target: what the graph's compile passes in its place (docs/build-graph.md#compile)."""
    from torch.utils.cpp_extension import min_supported_cpython
    return ["-DTORCH_EXTENSION_NAME=_C", "-DTORCH_API_INCLUDE_EXTENSION_H",
            f"-DPy_LIMITED_API={min_supported_cpython}", *STABLE_DEFINES]


#: THE STABLE ABI TARGET (docs/build.md#stable-abi): the extension is built against torch's stable C shim and
#: header-only types only, pinned to the oldest torch whose stable surface carries everything the host code uses (the
#: native handle of the current CUDA stream arrived in 2.13), so one binary loads under every torch from there on. The
#: graph passes these to every compile, and so does ratify's compile path.
TORCH_MIN = (2, 13)
STABLE_DEFINES = (f"-DTORCH_TARGET_VERSION=0x{(TORCH_MIN[0] << 56) | (TORCH_MIN[1] << 48):016x}",)

#: THE HOST C++ COMPILE'S OWN FLAGS (the registration unit, `csrc/rola/rola_api.cpp`): host code only, never a codegen
#: input of a kernel
HOST_CXX_FLAGS = ("-O3", "-std=c++20", "-DNDEBUG", *STABLE_DEFINES)


def gencodes(archs) -> list[str]:
    """`code=sm_XX` ONLY -- never `code=compute_XX`.

    A PTX entry would let the driver just-in-time compile this kernel for an
    architecture whose register and spill behaviour nobody measured, which is exactly
    what the ratification manifest exists to exclude (closed-world rule 2).
    """
    return [f"-gencode=arch=compute_{a},code=sm_{a}" for a in archs]


def nvcc_flags(archs, nvcc_threads: int = 2) -> list[str]:
    """The codegen flag list both compilers of these sources must pass.

    `--threads` is the ARCH COUNT by default and not a core count: it parallelizes
    nvcc across `-gencode` targets, so beyond the number of targets it buys nothing
    while multiplying the peak resident set (jobs x `--threads` cicc processes).
    """
    return ["-O3", "-std=c++20", "-lineinfo", f"--threads={nvcc_threads}",
            #: MANDATORY, not optional: `tools/ratify.py` parses exactly this report.
            "--ptxas-options=-v",
            #: COMPRESS EVERY IMAGE IN THE FATBIN, not just the PTX and debug ones
            #: `fatbinary --compress` covers by default. Without it the packer applies
            #: a SIZE HEURISTIC and leaves small cubins stored raw, which makes the
            #: shipped artifact a function of the SHARD PARTITION: at 32 arms per
            #: shard twelve of sixteen consumer cubins fall under the threshold
            #: (measured: every stored-raw cubin <= 9,076,960 B, every compressed one
            #: >= 10,075,232 B) and the `.so` is 141,919,496 B, against 73,743,608 B
            #: for the same 256 arms in four shards. With this flag it is
            #: 61,322,504 B at either partition, so shipped size stops being a
            #: function of the shard count. Codegen is untouched (716 SASS bodies,
            #: 0 differing) and the cost is +3.7 ms of decompression at the first
            #: launch of a process, measured.
            #: docs/build.md#fatbin-compression
            "-Xfatbin", "-compress-all"] + gencodes(archs)


#: GIGABYTES OF RESIDENT SET PER ``cicc``, MEASURED -- the constant that turns free
#: RAM into a job count, and the reason this is a formula and not a hardcoded
#: ``MAX_JOBS``.
#:
#: RE-DERIVED, AND THE OLD VALUE WAS WRONG BY 2x. At 1.5 GB the formula
#: returned ``MAX_JOBS=7`` on this 23 GB host and a cold-cache build was OOM-KILLED --
#: three translation units died with ``rc=255`` and the ninja build reported
#: ``Killed``, twice, before the number was re-measured. The old table was taken when
#: the heavy units were the instantiation shards; it did not survive the carry family
#: joining ``SOURCES``.
#:
#: WHAT SETS IT NOW IS THE TENSOR LIBRARY'S HEADER, not any kernel. Measured per
#: ``cicc`` (sm_86, ``--threads=1``):
#:
#: ===================================== ========== ==========
#: translation unit                        wall      peak/cicc
#: ===================================== ========== ==========
#: a bare CUDA TU (one trivial kernel)      0.9 s      0.19 GB
#: the same, plus ``torch/extension.h``    33.7 s      2.86 GB
#: one carry arm (torch-free ABI)           4.7 s      0.33 GB
#: carry/carry.cu (dispatch only)          36.2 s      2.88 GB
#: ===================================== ========== ==========
#:
#: and per ``cicc`` over a full both-arch build (sampled): ``factor.cu`` 2.92 GB,
#: ``entmax.cu`` 2.89, ``decode.cu`` 2.81, ``carry/carry.cu`` 2.75 -- every unit that
#: parses the extension header lands in one narrow band, and the 22 generated carry arm
#: translation units do not appear in the top twelve at all. The carry
#: family's single TU peaked at **6.14 GB** and was the binding term by more than 2x.
#:
#: So the constant is the HEADER's footprint, and it stops being a function of the arm
#: count -- which is the property that makes it stable as arms are added. RE-MEASURE it
#: when a translation unit joins the hand units (``tools/kernels.py``), or when torch's headers move.
GB_PER_NVCC_THREAD = 2.9


def default_max_jobs(nvcc_threads: int) -> int:
    """RAM, not cores, is the binding resource here, and that is a measurement:
    ``MAX_JOBS=8`` was OOM-KILLED on this 23 GB / 16-core host, twice -- once after
    8m32 while the constant said 1.5 GB, and again with a cold compiler cache,
    which is what forced the re-measurement above. So the memory term participates
    rather than the core count alone.

    At this value it returns 3 for a both-arch build (``--threads=2``) and 7 for a
    single-arch iteration build on a 22 GB-free host, which are the settings the
    measured builds used.

    The two parallelism axes MULTIPLY -- the jobs' concurrent ``nvcc`` processes,
    each running ``--threads`` concurrent ``cicc`` processes -- so the divisor is
    per-thread and not per-job."""
    import psutil

    cpus = os.cpu_count() or 1
    free_gb = psutil.virtual_memory().available / (1024 ** 3)
    return max(1, int(min(cpus // 2, free_gb / (GB_PER_NVCC_THREAD * nvcc_threads))))
