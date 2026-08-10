"""Self-contained build of the ladder RUNG 5B extension.

Same discipline as every rung below it: one source, one build directory, and a
DEVICE-SIDE check a stale binary cannot pass -- a magic written by a launched
kernel, the geometry stamp, the digest of the capability row the DEVICE CODE was
compiled against, and the ptxas register/shared-memory census (including local
bytes, so a spill is a build failure rather than a footnote) of the very entry
the measurement runs.

An arm is `(C, SWZ, BC, W0, W1)` -- rung 5A's arm set, arm for arm, so every
delta this rung reports is a same-arm delta.

THE TABLE IS THE SAME TABLE AND IS STILL A SEPARATE LAUNCH. `union_table` builds
it, `forward_raw` consumes it per TOKEN where rung 5A reduced it per GROUP, and
`forward` is the two plus the divide pass.
"""
from __future__ import annotations

import os
import pathlib
import sys

from torch.utils.cpp_extension import load

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
BASE_CSRC = HERE.parent / "denseref" / "csrc"
BUILD = HERE / "_build"
BUILD.mkdir(exist_ok=True)

#: must equal kBuildMagic in perside_compaction_la.cu
BUILD_MAGIC = 5200203
#: the one compute capability this extension is BUILT for
TARGET_CC = 860

#: the three width configurations in the binary, by role
CTRL = (8, 8)      # N = 64, one block: the same-work control against rung 1
FLAG = (64, 64)    # N = 4096, the flagship
SMALL = (16, 16)   # N = 256, a real fan-in an fp64 reference can gate at length

_mod = None


def module():
    global _mod
    if _mod is None:
        os.environ.setdefault("MAX_JOBS", "4")
        _mod = load(
            name="rung5b_cuda",
            sources=[str(HERE / "csrc" / "perside_compaction_la.cu")],
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
    """[(C, SWZ, BC, W0, W1, blocks, lead_span, tail_span, S, smem, ctas, entries)]."""
    return [(c, bool(w), bc, w0, w1, nb, ls, ts, s, b, t, e)
            for c, w, bc, w0, w1, nb, ls, ts, s, b, t, e in module().arms()]


def assert_build_identity(chunk, swizzle, bc, w0, w1) -> dict:
    """Device-side, not path-side."""
    from denseref import build as dbuild

    m = module()
    (magic, dstamp, caps_digest, regs, static_smem, binver, dyn_smem, maxthreads,
     local_bytes, stages, blocks, entries) = m.build_stamp(chunk, swizzle, bc, w0, w1)
    if magic != BUILD_MAGIC:
        raise RuntimeError(f"stale binary: device magic {magic} != source {BUILD_MAGIC}")
    if dstamp != 4 * 1000 + 64:
        raise RuntimeError(f"stale binary: warp/d_v stamp {dstamp}")
    if binver < 86:
        raise RuntimeError(f"not an sm_86 binary: binaryVersion {binver}")
    want = dbuild.arch_row(TARGET_CC)["digest"]
    if caps_digest != want:
        raise RuntimeError(f"the device code was compiled against capability row "
                           f"digest {caps_digest}, not {TARGET_CC}'s {want}")
    if local_bytes:
        raise RuntimeError(f"REGISTER SPILL: {local_bytes} B local for C={chunk} "
                           f"SWZ={swizzle} BC={bc} widths=({w0},{w1})")
    if blocks * bc != w0 * w1:
        raise RuntimeError(f"block plan does not tile N: {blocks}*{bc} != {w0*w1}")
    return dict(magic=magic, warps=dstamp // 1000, d_v=dstamp % 1000,
                caps_digest=caps_digest, regs_per_thread=regs,
                static_smem=static_smem, binary_version=binver,
                dynamic_smem=dyn_smem, max_threads=maxthreads,
                local_bytes=local_bytes, stages=stages, blocks=blocks,
                entries=entries, so=m.__file__)


#: THE DEFINITION'S CONSTANT, from the definition's file.
from rola.ops.constants import READOUT_EPS  # noqa: E402


def union_table(pr, pw, v, chunk, bc, w0, w1):
    """[B*H, 2, entries, L/32] int32: the membership clauses, packed by token."""
    return module().union_table(pr, pw, v, chunk, bc, w0, w1)


def forward_raw(pr, pw, v, tbl, chunk, swizzle, bc, w0, w1):
    """(num [B,H,L,d_v], den [B,H,L]) -- the readout's d_v+1 columns, undivided."""
    return module().forward_raw(pr, pw, v, tbl, chunk, swizzle, bc, w0, w1)


def forward(pr, pw, v, chunk, swizzle, bc, w0, w1, eps=READOUT_EPS):
    """y = num / (den + eps): table pass, kernel, divide pass."""
    return module().forward(pr, pw, v, chunk, swizzle, bc, w0, w1, float(eps))


def divide_pass(num, den, eps=READOUT_EPS):
    return module().divide_pass(num, den, float(eps))
