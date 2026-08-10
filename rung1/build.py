"""Self-contained build of the ladder RUNG 1 extension.

Same discipline as the base's `denseref/build.py`, and the same capability-layer
identity: one source, one build directory, and a DEVICE-SIDE check a stale binary
cannot pass -- a magic written by a launched kernel, the geometry stamp, the
instantiated level widths, the digest of the capability row the DEVICE CODE was
compiled against, and the ptxas register/shared-memory census (including local
bytes, so a spill is a build failure rather than a footnote) of the very entry
the measurement runs.

The rung compiles against the base's `arch_caps.cuh` / `ops.cuh` by include path;
there is exactly one capability table in the tree.
"""
from __future__ import annotations

import os
import pathlib

from torch.utils.cpp_extension import load

HERE = pathlib.Path(__file__).resolve().parent
BASE_CSRC = HERE.parent / "denseref" / "csrc"
BUILD = HERE / "_build"
BUILD.mkdir(exist_ok=True)

#: must equal kBuildMagic in routed_la.cu
BUILD_MAGIC = 590301
#: must equal (kW0, kW1) in routed_la.cu
WIDTHS = (8, 8)
#: the one compute capability this extension is BUILT for
TARGET_CC = 860

_mod = None


def module():
    global _mod
    if _mod is None:
        os.environ.setdefault("MAX_JOBS", "4")
        _mod = load(
            name="rung1_cuda",
            sources=[str(HERE / "csrc" / "routed_la.cu")],
            build_directory=str(BUILD),
            extra_include_paths=[str(BASE_CSRC)],
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
    """The (C, STAGES, SWZ, PACKED, smem_bytes) set the binary actually contains."""
    return [(c, s, bool(w), bool(p), b) for c, s, w, p, b in module().arms()]


def assert_build_identity(chunk: int, stages: int, swizzle: bool = False,
                          packed: bool = False) -> dict:
    """Device-side, not path-side. Raises unless the loaded binary IS this source
    AND was compiled against the base's capability table row for TARGET_CC."""
    from denseref import build as dbuild

    m = module()
    (magic, dstamp, caps_digest, widths, regs, static_smem, binver, dyn_smem,
     maxthreads, local_bytes) = m.build_stamp(chunk, stages, swizzle, packed)
    if magic != BUILD_MAGIC:
        raise RuntimeError(f"stale binary: device magic {magic} != source {BUILD_MAGIC}")
    if dstamp != 4 * 1000 + 64:
        raise RuntimeError(f"stale binary: warp/N stamp {dstamp}")
    if widths != WIDTHS[0] * 1000 + WIDTHS[1]:
        raise RuntimeError(f"stale binary: level widths {widths} != {WIDTHS}")
    if binver < 86:
        raise RuntimeError(f"not an sm_86 binary: binaryVersion {binver}")
    want = dbuild.arch_row(TARGET_CC)["digest"]
    if caps_digest != want:
        raise RuntimeError(f"the device code was compiled against capability row "
                           f"digest {caps_digest}, not {TARGET_CC}'s {want}")
    if local_bytes:
        raise RuntimeError(f"REGISTER SPILL: {local_bytes} B local for C={chunk} "
                           f"S={stages} SWZ={swizzle} PACKED={packed}")
    return dict(magic=magic, warps=dstamp // 1000, n=dstamp % 1000,
                caps_digest=caps_digest, widths=(widths // 1000, widths % 1000),
                regs_per_thread=regs, static_smem=static_smem,
                binary_version=binver, dynamic_smem=dyn_smem,
                max_threads=maxthreads, local_bytes=local_bytes, so=m.__file__)


#: the base's default arm per chunk, so `forward(.., c)` means the same thing on
#: both sides of a delta
DEFAULT_STAGES = {16: 4, 32: 4, 64: 3}


#: THE DEFAULT PRODUCTION ARM.  `packed=True` states the coefficient product as
#: `mul_bf16x2` on two packed pairs -- the capability table's SELECTING entry.
#: `packed=False` (WIDEN) hard-codes the fp32 route into the algorithm.  On sm_86
#: the two are within 1% on every measure and identical in total instructions, so
#: the tie is broken by the only asymmetry left: PACKED is the form an
#: architecture with bf16 SIMD arithmetic can serve in one instruction per two
#: coefficients, and WIDEN is not.  Both are built, both are gated, and they are
#: bit-identical.
DEFAULT_PACKED = True


def forward(pr, pw, v, chunk: int, stages: int = None, swizzle: bool = False,
            packed: bool = DEFAULT_PACKED):
    if stages is None:
        stages = DEFAULT_STAGES[chunk]
    return module().forward(pr, pw, v, chunk, stages, swizzle, packed)
