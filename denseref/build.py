"""Self-contained build of the clean-room dense chunked-LA extension.

Nothing here touches the RoLA build system. One source file, one extension, one
build directory, and a DEVICE-SIDE identity check that a stale binary cannot
pass (a magic written by a launched kernel plus the ptxas register/shared-memory
census of the very entry the measurement runs).
"""
from __future__ import annotations

import os
import pathlib

from torch.utils.cpp_extension import load

HERE = pathlib.Path(__file__).resolve().parent
BUILD = HERE / "_build"
BUILD.mkdir(exist_ok=True)

#: must equal kBuildMagic in dense_la.cu
BUILD_MAGIC = 580102

#: the compute capability this extension is compiled for.  One arch, closed
#: world: `arch_row` will answer for any tabulated cc, but only this one is
#: BUILT, and `assert_build_identity` refuses a binary that is not it.
TARGET_CC = 860

#: mirrors arch_caps.cuh `Caps`, in the order `arch_row` returns.
ARCH_ROW_FIELDS = (
    "cc",
    "async_copy", "tile_load", "mma_bf16_f32", "mma_a_from_regs",
    "mma_b_from_regs", "bf16_cvt_packed",
    "bf16_simd_arith", "async_bulk_copy", "warpgroup_mma",
    "supported",
    "mma_unit_threads", "smem_per_sm", "smem_per_cta_max", "regs_per_sm",
    "reg_alloc_unit", "max_threads_per_sm", "max_ctas_per_sm",
    "driver_smem_reserve", "tensor_f32_accum_pct",
    "digest",
)
REQUIRED = ("async_copy", "tile_load", "mma_bf16_f32", "mma_a_from_regs",
            "bf16_cvt_packed")
SELECTING = ("bf16_simd_arith", "async_bulk_copy", "warpgroup_mma")
#: not a baseline clause: which SIDE an accumulator-as-operand lands on is a
#: property of a kernel, and each kernel asserts its own at its own site.
KERNEL_LOCAL = ("mma_b_from_regs",)

#: every compute capability with a row, whether or not it is supported
TABULATED_CC = (800, 860, 870, 890, 900)

_mod = None


def module():
    global _mod
    if _mod is None:
        os.environ.setdefault("MAX_JOBS", "4")
        _mod = load(
            name="denseref_cuda",
            sources=[str(HERE / "csrc" / "dense_la.cu")],
            build_directory=str(BUILD),
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "-lineinfo",
                "-gencode", "arch=compute_86,code=sm_86",
            ],
            extra_cflags=["-O3", "-std=c++17"],
            verbose=False,
        )
    return _mod


def arms():
    """The (C, STAGES, SWZ, smem_bytes) set the binary actually contains."""
    return [(c, s, bool(w), b) for c, s, w, b in module().arms()]


def arch_row(cc: int = TARGET_CC) -> dict:
    """The capability table row for any TABULATED cc, evaluated by the table's own
    constexpr code. Answering for an architecture this box does not own is the
    whole point of tabulating per-SM resources."""
    return dict(zip(ARCH_ROW_FIELDS, module().arch_row(cc)))


def residency_plan(smem_bytes: int, regs_per_lane: int, cc: int = TARGET_CC,
                   threads: int = 128) -> dict:
    """Derived, never a literal: the three residency ceilings, the binding one,
    and the per-lane register budget that sustains 1..4 CTAs on `cc`. `threads`
    defaults to this kernel's CTA but is free, so the table can plan for a CTA
    shape this binary does not contain."""
    v = module().residency_plan(smem_bytes, regs_per_lane, cc, threads)
    return dict(by_smem=v[0], by_regs=v[1], by_threads=v[2], by_cta_cap=v[3],
                ctas=v[4], reg_budget={n: v[4 + n] for n in (1, 2, 3, 4)})


def assert_build_identity(chunk: int, stages: int, swizzle: bool = False) -> dict:
    """Device-side, not path-side. Raises unless the loaded binary IS this source
    AND was compiled against this capability table row."""
    m = module()
    (magic, dstamp, caps_digest, regs, static_smem, binver, dyn_smem,
     maxthreads, local_bytes) = m.build_stamp(chunk, stages, swizzle)
    if magic != BUILD_MAGIC:
        raise RuntimeError(f"stale binary: device magic {magic} != source {BUILD_MAGIC}")
    if dstamp != 4 * 1000 + 64:
        raise RuntimeError(f"stale binary: warp/d stamp {dstamp}")
    if binver < 86:
        raise RuntimeError(f"not an sm_86 binary: binaryVersion {binver}")
    want = arch_row(TARGET_CC)["digest"]
    if caps_digest != want:
        raise RuntimeError(f"the device code was compiled against capability row "
                           f"digest {caps_digest}, not {TARGET_CC}'s {want}")
    if local_bytes:
        raise RuntimeError(f"REGISTER SPILL: {local_bytes} B local for C={chunk} "
                           f"S={stages} SWZ={swizzle}")
    return dict(magic=magic, warps=dstamp // 1000, d=dstamp % 1000,
                caps_digest=caps_digest, regs_per_thread=regs,
                static_smem=static_smem, binary_version=binver, dynamic_smem=dyn_smem,
                max_threads=maxthreads, local_bytes=local_bytes, so=m.__file__)


def forward(q, k, v, chunk: int, stages: int = None, swizzle: bool = False):
    if stages is None:
        stages = {16: 4, 32: 4, 64: 3}[chunk]
    return module().forward(q, k, v, chunk, stages, swizzle)
