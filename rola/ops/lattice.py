# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE LEAF ORDER: the ``(k, m)`` box lattice, as host arithmetic.

Symbols, each at first use: ``widths`` = the per-level digit counts (``B_l`` each),
``D = len(widths)``, ``k`` = the uniform sub-box span per level, ``m`` = the capacity
multiplier, ``m_l`` its balanced per-level share, ``s_l = k * m_l`` = an owner's span at
level ``l``, ``g_l = B_l / s_l`` = the owner-grid extent, ``BC = prod_l s_l`` = the leaves
one owner holds, ``N = prod_l B_l``, ``ell`` = a leaf's canonical MSB-first mixed-radix
index and ``lambda`` = its LATTICE index.

The formulas are the R0 spec's LATTICE-1 (sections 2.1-2.5); this module is where the
shipped path gets them. ``pi`` is a fixed BIT PERMUTATION of the leaf index, so
:func:`permutation` is a convenience for the ORACLE and TEST seams and never a table any
kernel consults.
"""
from __future__ import annotations

import math

import torch

__all__ = ["LATTICE_CAP_M", "box_shape", "derive_lattice", "permutation", "to_canonical",
           "to_lattice"]

#: The capacity multiplier the flagship carry arm is solved at, and the ceiling this
#: solver searches down from. It is the ONE number here that is a choice rather than a
#: consequence, and it is the carry's: the two families write and read one plane.
LATTICE_CAP_M = 8

#: The owner span and the sub-box the device path reads as single 32-bit slices, mirrored
#: from `csrc/rola/src/decode/decode.cuh`. A candidate the device would refuse is refused
#: here instead, so a topology's lattice is decided once and never negotiated at a launch.
_MAX_SPAN = 32
_MAX_SUB_BOX = 64


def _admissible_spans(k: int, m: int, widths) -> tuple[int, ...] | None:
    """``(s_l)`` if this ``(k, m)`` sits inside every level, else ``None``."""
    D = len(widths)
    j = int(math.log2(m))
    spans = []
    for l in range(D):
        a = j // D + (1 if l < (j % D) else 0)
        s = k * (1 << a)
        if s > _MAX_SPAN or widths[l] % s != 0:
            return None
        spans.append(s)
    return tuple(spans)


def derive_lattice(widths) -> tuple[int, int]:
    """``(k, m)`` for a topology: the ONE authority, host-side and static.

    ``k`` is the SMALLEST span whose sub-box holds a whole 16-leaf atom, because that is
    the one that leaves the most owners for the grid to partition; ``log2(k)`` must divide
    4 so an atom is an aligned digit rectangle rather than a half-cut field (R0 spec
    section 3.3, K3/K4). A topology no such ``k`` divides takes ``k = 1``: every leaf is
    its own owner, ``pi`` is the identity, and the paged backing -- the only consumer of
    the atom rectangle -- refuses it.

    ``m`` is the largest admissible capacity at or below :data:`LATTICE_CAP_M`, which is
    what makes the flagship rows agree with the carry family's built arms.
    """
    widths = tuple(int(w) for w in widths)
    D = len(widths)
    k = 1
    for kappa in (1, 2, 4):
        cand = 1 << kappa
        if cand**D < 16 or cand**D > _MAX_SUB_BOX:
            continue
        if all(w % cand == 0 for w in widths):
            k = cand
            break
    m = 1
    cap = LATTICE_CAP_M
    while cap >= 1:
        if _admissible_spans(k, cap, widths) is not None:
            m = cap
            break
        cap >>= 1
    return k, m


def box_shape(widths, k: int, m: int):
    """``(m_l, s_l, g_l, BC, owners)`` for the ``(k, m)`` box on ``widths``.

    The balanced round-robin distribution of ``m`` across the levels is R0 spec section
    2.5 -- level 0 grows first and the exponents differ by at most one.
    """
    widths = tuple(int(w) for w in widths)
    D = len(widths)
    j = int(math.log2(m))
    if 2**j != m:
        raise ValueError(f"the lattice capacity m must be a power of two, got {m}")
    m_l = [1 << (j // D + (1 if l < (j % D) else 0)) for l in range(D)]
    s = [k * x for x in m_l]
    for l, (w, t) in enumerate(zip(widths, s, strict=True)):
        if w % t != 0:
            raise ValueError(
                f"level {l} of width {w} is not a whole number of owner spans k*m_l = {t}")
    g = [w // t for w, t in zip(widths, s, strict=True)]
    return m_l, s, g, math.prod(s), math.prod(g)


def permutation(widths, k: int, m: int, device="cuda") -> torch.Tensor:
    """``pi``: an ``[N]`` int64 tensor with ``pi[ell] = lambda`` (R0 spec section 2.3).

    THE ORACLE AND TEST SEAM'S VIEW, and nothing else. The oracle's state is canonical
    forever; a plane crosses into lattice order by ``lat.index_copy_(1, pi, canonical)``
    and back by ``canonical = lat.index_select(1, pi)``.
    """
    widths = tuple(int(w) for w in widths)
    D = len(widths)
    m_l, s, g, bc, _owners = box_shape(widths, k, m)
    ell = torch.arange(math.prod(widths), device=device, dtype=torch.int64)
    o_term = torch.zeros_like(ell)
    u_term = torch.zeros_like(ell)
    c_term = torch.zeros_like(ell)
    for l in range(D):
        d = (ell // math.prod(widths[l + 1:])) % widths[l]
        o_term += (d // s[l]) * math.prod(g[l + 1:])
        u_term += ((d % s[l]) // k) * math.prod(m_l[l + 1:])
        c_term += (d % k) * (k ** (D - 1 - l))
    return o_term * bc + u_term * (k**D) + c_term


def to_lattice(plane: torch.Tensor, widths, k: int, m: int) -> torch.Tensor:
    """A ``[..., N, cols]`` plane in CANONICAL leaf order, re-read in LATTICE order.

    THE ORACLE/TEST SEAM, and the only place a conversion is legal: no shipped path ever
    converts a plane, because every kernel is born in lattice order
    (docs/internals/decode/decode_lattice.md#the-seam).
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
