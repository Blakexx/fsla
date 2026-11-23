# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE LEAF ORDER, as the oracle and test seam reorders leaves by it. The lattice itself -- the ``(k, m)`` box of the level widths, its search and its
constants -- is C++'s (``csrc/fsla/src/common/plan/lattice.h``, docs/internals/common/plan/lattice.md); this module reads the box from the extension
(``fsla_decode_lattice_box``) and holds only the permutation that box defines.

Symbols, each at first use: ``widths`` = the per-level digit counts (``B_l`` each), ``D = len(widths)``, ``k`` = the uniform sub-box span per level,
``m`` = the capacity multiplier, ``s_l`` = an owner's span at level ``l``, ``g_l = B_l / s_l`` = the owner-grid extent, ``BC = prod_l s_l`` = the
leaves one owner holds, ``N = prod_l B_l``, ``ell`` = a leaf's canonical MSB-first mixed-radix index, ``lambda`` = its LATTICE index.

THE INTRA-BOX ORDER IS MIXED RADIX over the owner's run offsets, most significant level first:
``lambda = O*BC + sum_l r_l * prod_{l' > l} s_l'``, ``O = sum_l (d_l // s_l) * prod_{l' > l} g_l'``. ``pi`` is a fixed BIT PERMUTATION of the leaf
index, so :func:`permutation` is a convenience for the ORACLE and TEST seams and never a table any kernel consults.
"""
from __future__ import annotations

import math

import torch

from fsla.ops._ext import extension

__all__ = ["permutation", "to_canonical", "to_lattice"]


def permutation(widths, k: int, m: int, device="cuda") -> torch.Tensor:
    """``pi``: an ``[N]`` int64 tensor with ``pi[ell] = lambda``, for the ``(k, m)`` box of ``widths`` (the extension refuses a box none fits).

    THE ORACLE AND TEST SEAM'S VIEW, and nothing else. The oracle's state is canonical forever; a plane crosses into lattice order by
    ``lat.index_copy_(1, pi, canonical)`` and back by ``canonical = lat.index_select(1, pi)``.
    """
    widths = tuple(int(w) for w in widths)
    D = len(widths)
    words = [int(x) for x in extension().fsla_decode_lattice_box(list(widths), k, m)]
    bc, s, g = words[2], words[4:4 + D], words[4 + D:4 + 2 * D]
    ell = torch.arange(math.prod(widths), device=device, dtype=torch.int64)
    o_term = torch.zeros_like(ell)
    local = torch.zeros_like(ell)
    for l in range(D):
        d = (ell // math.prod(widths[l + 1:])) % widths[l]
        o_term += (d // s[l]) * math.prod(g[l + 1:])
        local += (d % s[l]) * math.prod(s[l + 1:])
    return o_term * bc + local


def to_lattice(plane: torch.Tensor, widths, k: int, m: int) -> torch.Tensor:
    """A ``[..., N, cols]`` plane in CANONICAL leaf order, re-read in LATTICE order.

    THE ORACLE/TEST SEAM, and the only place a conversion is legal: no shipped path ever converts a plane, because every kernel is born in lattice
    order (docs/internals/decode/decode_lattice.md#the-seam).
    """
    n = math.prod(int(w) for w in widths)
    pi = permutation(widths, k, m, device=plane.device)
    flat = plane.reshape(-1, n, plane.shape[-1])
    out = torch.empty_like(flat)
    out.index_copy_(1, pi, flat)
    return out.reshape(plane.shape)


def to_canonical(plane: torch.Tensor, widths, k: int, m: int) -> torch.Tensor:
    """The inverse of :func:`to_lattice` -- a lattice plane as the oracle's view."""
    n = math.prod(int(w) for w in widths)
    pi = permutation(widths, k, m, device=plane.device)
    flat = plane.reshape(-1, n, plane.shape[-1])
    return flat.index_select(1, pi).reshape(plane.shape)
