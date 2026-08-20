# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE LEAF ORDER: the ``(k, m)`` box lattice, as host arithmetic, and the ONE host
authority over it -- ``rola.ops.carry`` reads its box from here.

Symbols, each at first use: ``widths`` = the per-level digit counts (``B_l`` each),
``D = len(widths)``, ``k`` = the uniform sub-box span per level, ``m`` = the capacity
multiplier, ``m_l`` its per-level share, ``s_l = k * m_l`` = an owner's span at level
``l``, ``g_l = B_l / s_l`` = the owner-grid extent, ``BC = prod_l s_l`` = the leaves one
owner holds, ``N = prod_l B_l``, ``ell`` = a leaf's canonical MSB-first mixed-radix index,
``lambda`` = its LATTICE index, ``r_l`` = a leaf's run offset inside its owner at level
``l``, and ``atom`` = the ``MMA_K_QUANTUM``-leaf page granule.

THE INTRA-BOX ORDER IS MIXED RADIX over the owner's run offsets, most significant level
first: ``lambda = O*BC + sum_l r_l * prod_{l' > l} s_l'``. That is the device's order
(``csrc/rola/src/carry/box.cuh``), it is the ONLY order (the one-order ruling), and it is
what makes each level's digits CONTIGUOUS in the owner-local index -- the innermost
level's run is contiguous and, under K13, atom-aligned.

``pi`` is a fixed BIT PERMUTATION of the leaf index, so :func:`permutation` is a
convenience for the ORACLE and TEST seams and never a table any kernel consults.
"""
from __future__ import annotations

import math

import torch

from rola.engine.types import MMA_K_QUANTUM

__all__ = ["LATTICE_CAP_M", "box_shape", "covers_an_atom", "derive_lattice", "permutation",
           "to_canonical", "to_lattice"]

#: The capacity multiplier the flagship carry arm is solved at, and the ceiling this
#: solver searches down from. It is the ONE number here that is a choice rather than a
#: consequence, and it is the carry's: the two families write and read one plane.
LATTICE_CAP_M = 8

#: The owner span the device path reads as a single 32-bit slice, mirrored from
#: `csrc/rola/src/decode/decode.cuh`. A candidate the device would refuse is refused here
#: instead, so a topology's lattice is decided once and never negotiated at a launch.
_MAX_SPAN = 32

#: The uniform sub-box span is the carry family's codegen unit and never a free variable;
#: the search runs over the powers of two at or below it, largest first.
_MAX_SPAN_K = 4


def box_shape(widths, k: int, m: int):
    """``(m_l, s_l, g_l, BC, owners)`` for the ``(k, m)`` box on ``widths``.

    The distribution of ``m`` across the levels is R0 spec section 2.5 as amended by K13
    (wave-2b): the INNERMOST level takes whatever K13 (``s_{D-1} >= MMA_K_QUANTUM``)
    requires and never less than its balanced share, and the rest of ``j = log2 m`` is
    spread balanced over the outer levels, innermost first, so those exponents differ by
    at most one.

    K13 IS A FLOOR HERE, NOT AN ASSERTION: a topology whose widths cannot reach it (an odd
    level, a level narrower than an atom) still has a lattice -- it is the PAGED backing
    alone that needs one unit's leaves to be a whole number of atoms, and that backing
    refuses such a topology at its entry (:func:`covers_an_atom`). The floor is taken as
    far as ``j`` allows and no further.
    """
    widths = tuple(int(w) for w in widths)
    D = len(widths)
    j = int(math.log2(m))
    if 2 ** j != m:
        raise ValueError(f"the lattice capacity m must be a power of two, got {m}")
    kappa = int(math.log2(k))
    if 2 ** kappa != k:
        raise ValueError(f"the lattice span k must be a power of two, got {k}")
    floor_inner = max(0, int(math.log2(MMA_K_QUANTUM)) - kappa)
    a_inner = min(max(j // D + (1 if j % D else 0), floor_inner), j)
    j_outer = j - a_inner
    a = [a_inner if l == D - 1 else
         j_outer // (D - 1) + (1 if l >= (D - 1) - (j_outer % (D - 1)) else 0)
         for l in range(D)]
    m_l = [1 << x for x in a]
    s = [k * x for x in m_l]
    for l, (w, t) in enumerate(zip(widths, s, strict=True)):
        if w % t != 0:
            raise ValueError(
                f"level {l} of width {w} is not a whole number of owner spans k*m_l = {t} "
                "(R0 spec K6: an owner sits inside a level)")
    g = [w // t for w, t in zip(widths, s, strict=True)]
    return m_l, s, g, math.prod(s), math.prod(g)


def covers_an_atom(spans) -> bool:
    """K13: does the innermost level's span cover a whole number of atoms?

    THE PAGED BACKING'S ONE LATTICE CONDITION. Under the mixed-radix order a walk unit is
    ``(owner, r_0..r_{D-2})`` and its leaves are the innermost level's whole run, so this
    predicate is exactly "one unit's leaves are a whole number of atoms" -- which is what
    makes the candidate-atom enumeration a PARTITION. The two clauses the interleaved
    order needed (``k^D >= 16`` and ``4 % log2 k == 0``) are discharged by the order
    itself (`box.cuh`, K3/K4).
    """
    return int(spans[-1]) % MMA_K_QUANTUM == 0


def _admissible(k: int, m: int, widths) -> tuple[int, ...] | None:
    """``(s_l)`` if this ``(k, m)`` sits inside every level, else ``None``."""
    try:
        _, s, _, _, _ = box_shape(widths, k, m)
    except ValueError:
        return None
    return tuple(s) if all(x <= _MAX_SPAN for x in s) else None


def derive_lattice(widths) -> tuple[int, int]:
    """``(k, m)`` for a topology: the ONE authority, host-side and static.

    ``k`` is the LARGEST sub-box span at or below the carry family's codegen unit that
    divides every level, and ``m`` the largest admissible capacity at or below
    :data:`LATTICE_CAP_M` -- which is what makes the flagship rows agree with the carry
    family's built arms, ``(64, 64)`` and ``(256, 256)`` both at ``(4, 8)``.

    A topology no span divides takes ``k = 1``, and one no capacity fits takes ``m = 1``:
    every leaf is then its own owner, ``pi`` is the identity, and the paged backing --
    the only consumer of the atom rectangle -- refuses it (:func:`covers_an_atom`).
    """
    widths = tuple(int(w) for w in widths)
    k = 1
    cand = _MAX_SPAN_K
    while cand > 1:
        if all(w % cand == 0 for w in widths):
            k = cand
            break
        cand >>= 1
    m = 1
    cap = LATTICE_CAP_M
    while cap >= 1:
        if _admissible(k, cap, widths) is not None:
            m = cap
            break
        cap >>= 1
    return k, m


def permutation(widths, k: int, m: int, device="cuda") -> torch.Tensor:
    """``pi``: an ``[N]`` int64 tensor with ``pi[ell] = lambda`` (R0 spec section 2.3).

    THE ORACLE AND TEST SEAM'S VIEW, and nothing else. The oracle's state is canonical
    forever; a plane crosses into lattice order by ``lat.index_copy_(1, pi, canonical)``
    and back by ``canonical = lat.index_select(1, pi)``.
    """
    widths = tuple(int(w) for w in widths)
    D = len(widths)
    _, s, g, bc, _owners = box_shape(widths, k, m)
    ell = torch.arange(math.prod(widths), device=device, dtype=torch.int64)
    o_term = torch.zeros_like(ell)
    local = torch.zeros_like(ell)
    for l in range(D):
        d = (ell // math.prod(widths[l + 1:])) % widths[l]
        o_term += (d // s[l]) * math.prod(g[l + 1:])
        #: THE INTRA-BOX ORDER: mixed radix over the owner's run offsets, most
        #: significant level first -- `box.cuh`'s `run_of`.
        local += (d % s[l]) * math.prod(s[l + 1:])
    return o_term * bc + local


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
