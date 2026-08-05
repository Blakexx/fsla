"""THE ONE SOURCE OF THE nvcc FLAG LIST.

**WHY THIS FILE EXISTS.** Two programs compile the same translation units. `setup.py`
compiles them into the shipped `.so`; `tools/ratify.py` compiles them to read
`ptxas`'s register and spill report and certify that `.so`. A register/spill profile
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
that report. `--use_fast_math` and the `-U__CUDA_NO_*` undefines are refused; the
reasoning is in `setup.py`'s module docstring, which is the build's own contract.
"""
from __future__ import annotations

import os

#: Torch's `COMMON_NVCC_FLAGS` plus the ABI define, restated ONCE. `CUDAExtension`
#: adds exactly this set to the build; a standalone compile of the same file must
#: therefore pass it too, or it is compiling a different program.
TORCH_DEFINES = (
    "-D_GLIBCXX_USE_CXX11_ABI=0",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
)

#: THE `-MP` INSURANCE. It adds a phony make target per header to the dependency
#: file, so a depfile consumer that meets a header which no longer exists treats the
#: rule as dirty instead of erroring. On the pinned toolchain (nvcc 12.4) with
#: ninja 1.13 the failure it insures against DOES NOT REPRODUCE -- deleting a header
#: rebuilds cleanly with and without it, measured four ways
#: (`rola-scratch/workflows/refactor/p02_build_measure.md`, M1) -- so this is not the
#: fix for a live defect. It is a zero-risk build-graph flag with no codegen effect
#: whatever, kept because the graph it protects is also read by older ninja and by
#: non-ninja consumers, and because the cost of being wrong about that is a wiped
#: build directory.
DEPFILE_FLAGS = ("-MP",)


def gencodes(archs) -> list[str]:
    """`code=sm_XX` ONLY -- never `code=compute_XX`.

    A PTX entry would let the driver just-in-time compile this kernel for an
    architecture whose register and spill behaviour nobody measured, which is exactly
    what the ratification manifest exists to exclude (closed-world rule 2, asserted in
    `setup.py:assert_no_ptx`).
    """
    return [f"-gencode=arch=compute_{a},code=sm_{a}" for a in archs]


def nvcc_flags(archs, nvcc_threads: int = 2) -> list[str]:
    """The codegen flag list both compilers of these sources must pass.

    `--threads` is the ARCH COUNT by default and not a core count: it parallelizes
    nvcc across `-gencode` targets, so beyond the number of targets it buys nothing
    while multiplying the peak resident set (`MAX_JOBS` x `--threads` cicc processes).
    """
    return ["-O3", "-std=c++17", "-lineinfo", f"--threads={nvcc_threads}",
            #: MANDATORY, not optional: `tools/ratify.py` parses exactly this report.
            "--ptxas-options=-v"] + gencodes(archs)


#: GIGABYTES OF RESIDENT SET PER ``cicc``, MEASURED -- the constant that turns free
#: RAM into a job count, and the reason this is a formula and not a hardcoded
#: ``MAX_JOBS``: the same source gives 7 jobs on this 23 GB host and 4 on a 16 GB-class
#: one, with nobody editing anything.
#:
#: The measurement (``/usr/bin/time -v`` around one compile, peak RSS of the largest
#: single ``cicc``; the two axes MULTIPLY, ``MAX_JOBS`` x ``--threads`` of these at
#: once):
#:
#: ===================================== ==========
#: translation unit                       peak/cicc
#: ===================================== ==========
#: consumer.cu, pre-shard (512 arms)        4.4 GB
#: one instantiation shard (64 arms)        0.61 GB
#: entmax.cu                                1.54 GB
#: decode.cu                                2.89 GB
#: ===================================== ==========
#:
#: 1.5 is not the largest of those and is not meant to be: only two translation units
#: in the graph exceed it, so the worst concurrent footprint the formula admits here
#: is 2x2.89 + 2x1.54 + 5x2x0.61 = 15 GB against 21 GB free. Sizing on ``decode.cu``
#: instead would cost three compile waves for the eight shards to protect a case that
#: cannot happen more than twice. RE-MEASURE THIS whenever the shard size changes.
GB_PER_NVCC_THREAD = 1.5


def default_max_jobs(nvcc_threads: int) -> int:
    """RAM, not cores, is the binding resource here, and that is a measurement:
    ``MAX_JOBS=8`` was OOM-KILLED on this 23 GB / 16-core host after 8m32 (``ninja``
    reported ``Killed``; ``sys`` time 28m47 -- the machine was thrashing, not
    compiling). So the memory term participates rather than the core count alone.

    The two parallelism axes MULTIPLY -- ``MAX_JOBS`` concurrent ``nvcc`` processes,
    each running ``--threads`` concurrent ``cicc`` processes -- so the divisor is
    per-thread and not per-job."""
    import psutil

    cpus = os.cpu_count() or 1
    free_gb = psutil.virtual_memory().available / (1024 ** 3)
    return max(1, int(min(cpus // 2, free_gb / (GB_PER_NVCC_THREAD * nvcc_threads))))
