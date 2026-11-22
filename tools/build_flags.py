"""THE ONE SOURCE OF THE nvcc FLAG LIST.

**WHY THIS FILE EXISTS.** The graph compiles the extension's units into the shipped library, and `bazel/flags.py` writes
its flags from this file. A register/spill profile is a property of the FLAGS as much as of the source, so the list lives
here, once, and the graph takes it: the census (`tools/ratify.py`) reads `ptxas`'s report from those same compiles, never
from a compile with a flag list of its own.

**WHAT IS AND IS NOT HERE.** The CODEGEN-AFFECTING flags: optimization level,
language standard, the `-gencode` targets, the preprocessor defines torch's
`CUDAExtension` adds, and `--threads`. Not here: link flags, the C++ host compiler's
own flags, and include directories that only one of the two callers can know (torch's
own header locations are resolved by `include_paths` at the call site).

`--ptxas-options=-v` is included and is MANDATORY: `tools/ratify.py`'s census parses exactly
that report. It changes what the assembler PRINTS, not what it emits (as `--warn-on-spills` does).

REFUSED, each a codegen change that would arrive as a packaging change: the
`-U__CUDA_NO_HALF_OPERATORS__` / `-U__CUDA_NO_HALF_CONVERSIONS__` /
`-U__CUDA_NO_BFLOAT16_CONVERSIONS__` undefines and `--expt-extended-lambda`, which would
undo torch's own defines and enable a language feature the measured build did not have;
and `--use_fast_math`, since the tolerances are fp32 accumulation tolerances against an
fp64 oracle, and fast-math silently changes the numeric contract they were derived under.
"""
from __future__ import annotations

#: Torch's `COMMON_NVCC_FLAGS`, restated ONCE. `CUDAExtension` adds exactly this set to a build, so the graph's compile
#: of the same file must pass it too or it is compiling a different program.
COMMON_NVCC_FLAGS = (
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
)


def extension_defines() -> list[str]:
    """The defines torch's `CUDAExtension` gives every source of a `py_limited_api` extension, the extension `_C`, and
    this project's stable-ABI target: what the graph's compile passes in its place (docs/build-graph.md#compile)."""
    from torch.utils.cpp_extension import min_supported_cpython
    return ["-DTORCH_EXTENSION_NAME=_C", "-DTORCH_API_INCLUDE_EXTENSION_H",
            f"-DPy_LIMITED_API={min_supported_cpython}", *STABLE_DEFINES]


#: THE STABLE ABI TARGET (docs/build.md#stable-abi): the extension is built against torch's stable C shim and
#: header-only types only, pinned to the oldest torch whose stable surface carries everything the host code uses (the
#: native handle of the current CUDA stream arrived in 2.13), so one binary loads under every torch from there on. The
#: graph passes these to every compile.
TORCH_MIN = (2, 13)
STABLE_DEFINES = (f"-DTORCH_TARGET_VERSION=0x{(TORCH_MIN[0] << 56) | (TORCH_MIN[1] << 48):016x}",)

#: THE HOST C++ COMPILE'S OWN FLAGS (the registration unit, `csrc/rola/rola_api.cpp`): host code only, never a codegen
#: input of a kernel
HOST_CXX_FLAGS = ("-O3", "-std=c++20", "-DNDEBUG", *STABLE_DEFINES)


def gencodes(archs) -> list[str]:
    """`code=sm_XX` ONLY -- never `code=compute_XX`.

    A PTX entry would let the driver just-in-time compile this kernel for an
    architecture whose register and spill behaviour nobody measured, which is exactly
    what the closed world exists to exclude (closed-world rule 2).
    """
    return [f"-gencode=arch=compute_{a},code=sm_{a}" for a in archs]


def nvcc_flags(archs, nvcc_threads: int) -> list[str]:
    """The codegen flag list the compile of these sources passes.

    `--threads` is the machine's `host.nvcc_threads`, the architecture count and not a core count: it parallelizes
    nvcc across `-gencode` targets, so beyond the number of targets it buys nothing while multiplying the peak
    resident set (jobs x `--threads` cicc processes).
    """
    return ["-O3", "-std=c++20", "-lineinfo", f"--threads={nvcc_threads}",
            #: MANDATORY, not optional: `tools/ratify.py`'s census parses exactly this report.
            "--ptxas-options=-v",
            #: WARN WHERE A FUNCTION SPILLS REGISTERS: ptxas prints a notice naming the function and its bytes, which the gate reads
            #: against the ledger (`register_spill`) like every notice of the build; it changes what the assembler PRINTS only
            "--ptxas-options=--warn-on-spills",
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
