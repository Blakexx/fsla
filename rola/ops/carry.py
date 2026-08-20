# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""K31 R1 -- the box-native CARRY family's python seam.

A BRING-UP SURFACE, not a shipped path. Nothing in ``rola.engine`` or
``rola.layer`` reaches this module; the shipped chunk consumer stays the baseline
until this family beats it cell for cell (the study's rung law, section 6).

Symbols, each at first use: ``widths`` = the per-level digit counts (``B`` each),
``k`` = the uniform sub-box span per level, ``m`` = the capacity multiplier,
``W`` = the global window length in tokens, ``C`` = the private batch's token
tile, ``BC = prod_l k*m_l`` = the leaves one owner holds, ``lambda`` = a leaf's
LATTICE index (the R0 spec's LATTICE-1), ``ell`` = its canonical MSB-first
mixed-radix index.
"""
from __future__ import annotations

import math

import torch

from rola.ops._ext import extension

#: the carve, ruled: channel-split is the sm_80 carve (R1 measurement,
#: 2026-08-20); the state-split variants are deleted -- receipts in
#: workflows/k31-r1-carry/FINDINGS.md, revival = git.
CHANNEL_SPLIT = 1

MMA_K_QUANTUM = 16


def box_shape(widths, k: int, m: int):
    """``(m_l, s_l, g_l, BC, owners)`` for the ``(k, m)`` box on ``widths``.

    The balanced round-robin distribution of ``m`` across the levels is R0 spec
    section 2.5 -- level 0 grows first and the exponents differ by at most one.
    Derived here rather than imported from the device header for the same reason
    the oracle derives its own strides: two derivations disagree when one is wrong.
    """
    D = len(widths)
    j = int(math.log2(m))
    assert 2 ** j == m, "m is a power of two"
    a = [j // D + (1 if l < (j % D) else 0) for l in range(D)]
    m_l = [1 << x for x in a]
    s = [k * x for x in m_l]
    g = [w // t for w, t in zip(widths, s)]
    assert all(w % t == 0 for w, t in zip(widths, s)), "K6: an owner sits inside a level"
    bc = math.prod(s)
    return m_l, s, g, bc, math.prod(g)


def lattice_of_canonical(widths, k: int, m: int, device="cuda") -> torch.Tensor:
    """``pi``: an ``[N]`` int64 tensor with ``pi[ell] = lambda`` (R0 spec section 2.3).

    A permutation, and only a permutation -- `pi` is a fixed BIT PERMUTATION of the
    leaf index (spec section 2.4), so this tensor is a convenience for the host
    seam and never a table the device consults.
    """
    D = len(widths)
    m_l, s, g, bc, _ = box_shape(widths, k, m)
    N = math.prod(widths)
    ell = torch.arange(N, device=device, dtype=torch.int64)
    lam = torch.zeros_like(ell)
    o_term = torch.zeros_like(ell)
    u_term = torch.zeros_like(ell)
    c_term = torch.zeros_like(ell)
    for l in range(D):
        stride = math.prod(widths[l + 1:])
        d = (ell // stride) % widths[l]
        o_term += (d // s[l]) * math.prod(g[l + 1:])
        u_term += ((d % s[l]) // k) * math.prod(m_l[l + 1:])
        c_term += (d % k) * (k ** (D - 1 - l))
    lam = o_term * bc + u_term * (k ** D) + c_term
    return lam


def pack_side(levels) -> torch.Tensor:
    """The per-level ``[B,T,H,width_l]`` tuple as one ``[BH, T, sum_l width_l]``
    bf16 plane -- the layout both kernel families read."""
    Bn, T, H, _ = levels[0].shape
    out = torch.empty((Bn, H, T, sum(x.shape[-1] for x in levels)),
                      dtype=torch.bfloat16, device=levels[0].device)
    off = 0
    for level in levels:
        w = level.shape[-1]
        out[:, :, :, off:off + w].copy_(level.permute(0, 2, 1, 3))
        off += w
    return out.reshape(Bn * H, T, -1)


def carry_inter(read_levels, write_levels, g_write, v, widths, *, k: int, m: int,
                window: int, chunk: int = 32, carve: int = CHANNEL_SPLIT,
                state_in: torch.Tensor | None = None):
    """The INTER term, undivided: ``(num, den, state_out)``.

    ``num`` is ``[BH, T, d_v]`` fp32, ``den`` is ``[BH, T]`` fp32, and
    ``state_out`` is ``[BH, N/16, 16, d_v+1]`` fp32 in LATTICE leaf order.
    ``state_in``, when given, is the same layout and order.
    """
    pr = pack_side(read_levels)
    pw = pack_side(write_levels)
    gw = g_write.permute(0, 2, 1).contiguous().to(torch.bfloat16)
    gw = gw.reshape(gw.shape[0] * gw.shape[1], -1)
    return extension().carry_forward_inter(
        pr, pw, gw, v.to(torch.bfloat16).contiguous(), list(widths), int(k), int(m),
        int(window), int(chunk), int(carve), state_in)


def build_stamp(widths, k, m, window, chunk, carve):
    return extension().carry_build_stamp(list(widths), int(k), int(m), int(window),
                                         int(chunk), int(carve))


def arms():
    return extension().carry_arms()
