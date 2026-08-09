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
BUILD_MAGIC = 570106

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


def assert_build_identity(chunk: int) -> dict:
    """Device-side, not path-side. Raises unless the loaded binary IS this source."""
    m = module()
    magic, warpstamp, regs, static_smem, binver, dyn_smem = m.build_stamp(chunk)
    if magic != BUILD_MAGIC:
        raise RuntimeError(f"stale binary: device magic {magic} != source {BUILD_MAGIC}")
    if warpstamp != 4 * 1000 + 72:
        raise RuntimeError(f"stale binary: warp/ld stamp {warpstamp}")
    if binver < 86:
        raise RuntimeError(f"not an sm_86 binary: binaryVersion {binver}")
    return dict(magic=magic, warps=warpstamp // 1000, smem_ld=warpstamp % 1000,
                regs_per_thread=regs, static_smem=static_smem,
                binary_version=binver, dynamic_smem=dyn_smem, so=m.__file__)


def forward(q, k, v, chunk: int):
    return module().forward(q, k, v, chunk)
