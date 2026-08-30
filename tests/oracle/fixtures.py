# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""K31 R1's cells: the routing draws, the plane crossings, and the build stamp gate.

``k_tok`` (a token's nonzero digits per level, the routing sparsity) is a property
of the DRAW here and of nothing in the kernel -- which is the design's whole claim,
so the fixtures state it explicitly and the kernel is never told.
"""
from __future__ import annotations

import math

import torch

from rola.ops import carry as carry_ops
from rola.ops.paging import from_split_planes, to_split_planes


def simplex(shape, k_tok, gen, device, live=None):
    """A row-normalized draw with exactly ``k_tok`` nonzeros per row, or dense.

    ``live`` truncates the row to its FIRST ``live`` digits -- STRUCTURED support, as
    against ``k_tok``'s unstructured one. The two are different questions: unstructured
    sparsity at long ``T`` still reaches every atom, so only a truncation can produce an
    atom the routing never touches (P80b's idle-resident cell). It is applied after both
    draws so a cell's RNG stream does not depend on it.
    """
    x = torch.rand(shape, device=device, dtype=torch.float64, generator=gen)
    if k_tok is not None and k_tok < shape[-1]:
        keep = torch.zeros(shape, device=device, dtype=torch.float64)
        idx = torch.argsort(torch.rand(shape, device=device, generator=gen), dim=-1)[..., :k_tok]
        keep.scatter_(-1, idx, 1.0)
        x = x * keep
    if live is not None and live < shape[-1]:
        x[..., live:] = 0.0
    return x / x.sum(-1, keepdim=True)


class Cell:
    """One realized cell. ``widths`` is the topology, ``k_tok`` the routing draw."""

    def __init__(self, widths, T, k_tok=None, seed=0, d_v=64, B=1, H=1,
                 device="cuda", state_in=False, support=1.0):
        gen = torch.Generator(device=device).manual_seed(seed)
        self.widths, self.T, self.d_v, self.B, self.H = tuple(widths), T, d_v, B, H
        self.N = math.prod(widths)
        shape = (B, T, H)
        self.support = support
        live = tuple(max(1, int(w * support)) for w in widths)
        self.read = tuple(simplex(shape + (w,), k_tok, gen, device, live=lv)
                          for w, lv in zip(widths, live))
        self.write = tuple(simplex(shape + (w,), k_tok, gen, device, live=lv)
                           for w, lv in zip(widths, live))
        self.g_write = torch.rand(*shape, device=device, dtype=torch.float64,
                                  generator=gen) + 0.5
        self.v = torch.randn(*shape, d_v, device=device, dtype=torch.float64, generator=gen)
        #: THE OPERAND PLANES ARE bf16 AND THE REFERENCE READS THE SAME BYTES.  The
        #: gate is a conformance band, not a rounding coincidence: comparing an fp64
        #: reference of the fp64 draws against a kernel fed their bf16 images would
        #: measure the CAST, which is not what is under test.
        self.read_bf = tuple(t.to(torch.bfloat16) for t in self.read)
        self.write_bf = tuple(t.to(torch.bfloat16) for t in self.write)
        self.g_bf = self.g_write.to(torch.bfloat16)
        self.v_bf = self.v.to(torch.bfloat16)
        self.state_in = None
        if state_in:
            self.state_in = 0.1 * torch.randn(B * H, self.N, d_v + 1, device=device,
                                              dtype=torch.float64, generator=gen)

    # -- the reference reads the ROUNDED operands ---------------------------
    def ref_levels(self):
        return (tuple(t.double() for t in self.read_bf),
                tuple(t.double() for t in self.write_bf),
                self.g_bf.double(), self.v_bf.double())

    def state_in_plane(self):
        """``state_in`` as the kernel's state plane, ``[BH, N/16, 16, cols]``.

        A RE-BLOCKING AND THE SPLIT-PLANE PACK since K50: the consumer addresses its state by
        CANONICAL atom id in both backings, so the plane is the atom-major view of
        ``[BH, N, cols]`` that `docs/internals/state.md` section 3 defines, and the leaf
        permutation this method used to apply is gone with the lattice keying.
        """
        if self.state_in is None:
            return None
        return to_split_planes(self.state_in.to(torch.float32).reshape(
            self.B * self.H, self.N // 16, 16, self.d_v + 1).contiguous())


def canonical_from_plane(plane):
    """A ``[BH, N/16, 16, stored_cols]`` state plane as ``[BH, N, cols]`` CANONICAL.

    The kernel's plane IS the atom-major view of the canonical one (P80) in the STORED
    split-plane form (K50), so this is the hi||lo gather and a reshape. It stays a NAMED step because the assertion it feeds is the family's
    final-state gate, and a leaf-order error must surface at a named seam rather than as
    noise inside a comparison. The gather is also where the page stride's pad leaves:
    the plane is `stored_cols` wide and the canonical state is `cols` wide.
    """
    gathered = from_split_planes(plane)
    return gathered.reshape(plane.shape[0], -1, gathered.shape[-1])


#: THE DECLARATION THE FLAGSHIP SHIPS UNDER: write-sparse on the even levels and
#: read-sparse on the odd ones (`sum_l mode_l << 2l` = 6 at D = 2).  Spelled here
#: because the freshness stamp has to be read off an arm the binary actually carries,
#: and since K36 a default build carries the SHIPPED rows -- of which the DENSE_BOTH
#: flagship (arm 0) is not one.
ALT = 6


def built_arms() -> set:
    """`{(D, B, k, m, d_v, W, C, carve, level_modes)}` -- what THIS binary carries.

    `arms()` is the binary's own answer (`rola.ops.carry.arms()`), so this is a fact
    about the `.so` under test and never about the source tree that produced it.
    """
    return {tuple(row[:9]) for row in carry_ops.arms()}


def declared_arms() -> set:
    """The rows the ARM LIST declares, in the same key `built_arms` reports.

    Read from `tools/gen_shards.py`, which owns the list and generates both the
    dispatch's macro and the per-arm translation units.
    """
    import sys
    from pathlib import Path as _Path

    tools = _Path(__file__).resolve().parents[2] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    import gen_shards

    return {tuple(row[:9]) for row in gen_shards.CARRY_ARMS}


def require_arm(widths, k, m, window, chunk, carve, level_modes, d_v=carry_ops.D_V):
    """SKIP unless this binary carries the arm, and NAME the one it is missing.

    Since K36 a default build compiles the SHIPPED rows only; the conformance cells
    (`N = 4096`, `W = 64`) and the two DENSE_BOTH controls are TEST rows, built for the
    battery and not shipped. A test that needs one says so here rather than failing
    inside the dispatch with "unbuilt carry arm", which is the same fact reported as a
    defect.

    AN UNDECLARED ARM IS NOT A SKIP. If the key is in no row of the list at all, the
    binary is not missing it -- nothing builds it, in any configuration -- and turning
    that into a skip would report an absent arm as satisfied coverage. Only a DECLARED
    row that this build did not compile is skippable; anything else falls through to the
    dispatch, which refuses it and names the whole key.
    """
    import pytest

    key = (len(widths), widths[0], k, m, d_v, window, chunk, carve, level_modes)
    if key in built_arms() or key not in declared_arms():
        return
    pytest.skip(
        f"this binary does not carry the carry arm (widths={tuple(widths)}, k={k}, "
        f"m={m}, d_v={d_v}, W={window}, C={chunk}, carve={carve}, "
        f"level_modes={level_modes}). It is a declared TEST row; build the battery's "
        f"set: ROLA_CARRY_ARMS=all ROLA_CUDA_ARCHS=86 pip install -e . "
        f"--no-build-isolation")


def assert_fresh_binary():
    """DEVICE-SIDE, because a path or hash check cannot catch an extension trap.

    Read off a SHIPPED arm (the flagship at `ALT`): the stamp itself is arm-independent,
    but the entry point that returns it is not, and a default build does not carry the
    DENSE_BOTH flagship this used to ask for.
    """
    import rola
    stamp = carry_ops.build_stamp((256, 256), 4, 8, carry_ops.WINDOW, 32,
                                  carry_ops.LEAF_SPLIT, ALT)
    assert stamp[0] == carry_ops.BUILD_MAGIC, (
        f"stale binary: the carry kernel's device stamp is {stamp[0]}, not "
        f"{carry_ops.BUILD_MAGIC}. rola is {rola.__file__}")
    return stamp


def relative(actual, reference):
    scale = max(1e-30, float(reference.abs().max()))
    return float((actual.double() - reference.double()).abs().max()) / scale
