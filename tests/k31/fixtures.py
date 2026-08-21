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


def simplex(shape, k_tok, gen, device):
    """A row-normalized draw with exactly ``k_tok`` nonzeros per row, or dense."""
    x = torch.rand(shape, device=device, dtype=torch.float64, generator=gen)
    if k_tok is not None and k_tok < shape[-1]:
        keep = torch.zeros(shape, device=device, dtype=torch.float64)
        idx = torch.argsort(torch.rand(shape, device=device, generator=gen), dim=-1)[..., :k_tok]
        keep.scatter_(-1, idx, 1.0)
        x = x * keep
    return x / x.sum(-1, keepdim=True)


class Cell:
    """One realized cell. ``widths`` is the topology, ``k_tok`` the routing draw."""

    def __init__(self, widths, T, k_tok=None, seed=0, d_v=64, B=1, H=1,
                 device="cuda", state_in=False):
        gen = torch.Generator(device=device).manual_seed(seed)
        self.widths, self.T, self.d_v, self.B, self.H = tuple(widths), T, d_v, B, H
        self.N = math.prod(widths)
        shape = (B, T, H)
        self.read = tuple(simplex(shape + (w,), k_tok, gen, device) for w in widths)
        self.write = tuple(simplex(shape + (w,), k_tok, gen, device) for w in widths)
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

    def state_in_lattice(self, k, m):
        """``state_in`` re-viewed in LATTICE leaf order, as the kernel's plane."""
        if self.state_in is None:
            return None
        pi = carry_ops.lattice_of_canonical(self.widths, k, m, device=self.state_in.device)
        out = torch.zeros_like(self.state_in)
        out.index_copy_(1, pi, self.state_in)
        return out.to(torch.float32).reshape(
            self.B * self.H, self.N // 16, 16, self.d_v + 1).contiguous()


def canonical_from_lattice(plane, widths, k, m):
    """A ``[BH, N/16, 16, cols]`` LATTICE plane as ``[BH, N, cols]`` CANONICAL.

    A leaf-order error surfaces HERE as a permutation rather than as noise -- the
    shipped family's final-state gate property, kept.
    """
    N = math.prod(widths)
    pi = carry_ops.lattice_of_canonical(widths, k, m, device=plane.device)
    flat = plane.reshape(plane.shape[0], N, -1)
    return flat.index_select(1, pi)


def assert_fresh_binary():
    """DEVICE-SIDE, because a path or hash check cannot catch an extension trap."""
    import rola
    stamp = carry_ops.build_stamp((256, 256), 4, 8, carry_ops.WINDOW, 32,
                                  carry_ops.LEAF_SPLIT)
    assert stamp[0] == carry_ops.BUILD_MAGIC, (
        f"stale binary: the carry kernel's device stamp is {stamp[0]}, not "
        f"{carry_ops.BUILD_MAGIC}. rola is {rola.__file__}")
    return stamp


def relative(actual, reference):
    scale = max(1e-30, float(reference.abs().max()))
    return float((actual.double() - reference.double()).abs().max()) / scale
