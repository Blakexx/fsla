"""Self-contained build of the ladder RUNG 6F extension (the level plan, G1).

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

#: must equal kBuildMagic in formed_schedule_la.cu
BUILD_MAGIC = 6600101
#: THE SHIPPED FORMED-RING DEPTH, in words per side; must equal the source's
#: `RUNG5D_FORM_WORDS` default. A build at any other depth is a SEPARATE module
#: with a separate build directory, and the device stamp carries the depth so a
#: number can never be attributed to the wrong one.
FORM_WORDS = 128
#: the one compute capability this extension is BUILT for
TARGET_CC = 860

#: the three width configurations in the binary, by role
CTRL = (8, 8)      # N = 64, one block: the same-work control against rung 1
FLAG = (64, 64)    # N = 4096, the flagship
SMALL = (16, 16)   # N = 256, a real fan-in an fp64 reference can gate at length

_mods = {}


def module(form_words: int = FORM_WORDS):
    m = _mods.get(form_words)
    if m is None:
        os.environ.setdefault("MAX_JOBS", "4")
        bdir = BUILD if form_words == FORM_WORDS else BUILD / f"f{form_words}"
        bdir.mkdir(exist_ok=True)
        m = load(
            name=f"rung6g_cuda_f{form_words}",
            sources=[str(HERE / "csrc" / "level_plan_la.cu")],
            build_directory=str(bdir),
            extra_include_paths=[str(BASE_CSRC)],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "-lineinfo",
                f"-DRUNG6_FORM_WORDS={form_words}",
                "-gencode", "arch=compute_86,code=sm_86",
            ],
            extra_cflags=["-O3", "-std=c++17"],
            verbose=False,
        )
        _mods[form_words] = m
    return m


def arms(form_words: int = FORM_WORDS):
    """[(C, SWZ, BC, depth, width, blocks, outer_span, inner_span, S, smem, ctas, entries)].

    An arm's topology is (depth, width): `depth` uniform power-of-two levels of
    `width` branches, N = width**depth.  Callers wanting a widths tuple take
    `(width,) * depth`.
    """
    return [(c, bool(w), bc, d, wd, nb, os_, is_, s, b, t, e)
            for c, w, bc, d, wd, nb, os_, is_, s, b, t, e in module(form_words).arms()]


def assert_build_identity(chunk, swizzle, bc, widths, form_words=FORM_WORDS) -> dict:
    """Device-side, not path-side -- and the formed ring's depth is one of the facts."""
    from denseref import build as dbuild

    m = module(form_words)
    widths = list(widths)
    (magic, dstamp, caps_digest, regs, static_smem, binver, dyn_smem, maxthreads,
     local_bytes, stages, blocks, entries) = m.build_stamp(chunk, swizzle, bc, widths)
    if magic != BUILD_MAGIC + form_words:
        raise RuntimeError(f"stale binary: device magic {magic} != source "
                           f"{BUILD_MAGIC} + form words {form_words}")
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
                           f"SWZ={swizzle} BC={bc} widths={tuple(widths)}")
    n_slots = 1
    for w in widths:
        n_slots *= w
    if blocks * bc != n_slots:
        raise RuntimeError(f"block plan does not tile N: {blocks}*{bc} != {n_slots}")
    return dict(magic=magic, warps=dstamp // 1000, d_v=dstamp % 1000,
                caps_digest=caps_digest, regs_per_thread=regs,
                static_smem=static_smem, binary_version=binver,
                dynamic_smem=dyn_smem, max_threads=maxthreads,
                local_bytes=local_bytes, stages=stages, blocks=blocks,
                entries=entries, form_words=form_words, so=m.__file__)


#: THE DEFINITION'S CONSTANT, from the definition's file.
from rola.ops.constants import READOUT_EPS  # noqa: E402


def union_table(pr, pw, v, chunk, bc, widths, form_words=FORM_WORDS):
    """[B*H, 2, entries, L/32] int32: the membership clauses, packed by token."""
    return module(form_words).union_table(pr, pw, v, chunk, bc, list(widths))


def forward_raw(pr, pw, gw, v, tbl, chunk, swizzle, bc, widths,
                form_words=FORM_WORDS):
    """(num [B,H,L,d_v], den [B,H,L]) -- the readout's d_v+1 columns, undivided.

    ``gw`` is the write gain: per-token positive scalar, bf16 [B,H,L].  The
    amplitudes stay pure simplexes; the kernel gains the deposit.
    """
    return module(form_words).forward_raw(pr, pw, gw, v, tbl, chunk, swizzle, bc,
                                          list(widths))


def forward(pr, pw, gw, v, chunk, swizzle, bc, widths, eps=READOUT_EPS,
            form_words=FORM_WORDS):
    """y = num / (den + eps): table pass, kernel, divide pass."""
    return module(form_words).forward(pr, pw, gw, v, chunk, swizzle, bc,
                                      list(widths), float(eps))


def divide_pass(num, den, eps=READOUT_EPS):
    return module().divide_pass(num, den, float(eps))
