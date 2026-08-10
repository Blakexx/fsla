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
BUILD_MAGIC = 570203

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


def assert_build_identity(chunk: int, stages: int, swizzle: bool = False) -> dict:
    """Device-side, not path-side. Raises unless the loaded binary IS this source."""
    m = module()
    (magic, dstamp, regs, static_smem, binver, dyn_smem,
     maxthreads, local_bytes) = m.build_stamp(chunk, stages, swizzle)
    if magic != BUILD_MAGIC:
        raise RuntimeError(f"stale binary: device magic {magic} != source {BUILD_MAGIC}")
    if dstamp != 4 * 1000 + 64:
        raise RuntimeError(f"stale binary: warp/d stamp {dstamp}")
    if binver < 86:
        raise RuntimeError(f"not an sm_86 binary: binaryVersion {binver}")
    if local_bytes:
        raise RuntimeError(f"REGISTER SPILL: {local_bytes} B local for C={chunk} "
                           f"S={stages} SWZ={swizzle}")
    return dict(magic=magic, warps=dstamp // 1000, d=dstamp % 1000,
                regs_per_thread=regs, static_smem=static_smem,
                binary_version=binver, dynamic_smem=dyn_smem,
                max_threads=maxthreads, local_bytes=local_bytes, so=m.__file__)


def forward(q, k, v, chunk: int, stages: int = None, swizzle: bool = False):
    if stages is None:
        stages = {16: 4, 32: 4, 64: 3}[chunk]
    return module().forward(q, k, v, chunk, stages, swizzle)
