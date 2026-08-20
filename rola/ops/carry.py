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

import torch

from rola.ops import lattice
from rola.ops._ext import extension

#: the carve, ruled: channel-split is the sm_80 carve (R1 measurement,
#: 2026-08-20); the state-split variants are deleted -- receipts in
#: workflows/k31-r1-carry/FINDINGS.md, revival = git.
CHANNEL_SPLIT = 1

#: the flagship window, and the conformance cell's -- the architecture doc's
#: two-constraint sizing (integration rung I1(b)); `rola.ops.intra` names the
#: SAME two, because the two kernels run one window grid.
WINDOW = 384
WINDOW_WIDE = 512
WINDOW_SMALL = 64

#: the carry kernel's device build magic, mirrored from `carry_kernel.cuh`. It
#: moves whenever that source does, which is what makes reading it back off the
#: DEVICE an extension-trap check rather than a path check.
BUILD_MAGIC = 31010013

MMA_K_QUANTUM = lattice.MMA_K_QUANTUM


def box_shape(widths, k: int, m: int):
    """``(m_l, s_l, g_l, BC, owners)`` for the ``(k, m)`` box, with K13 ASSERTED.

    The derivation is `rola.ops.lattice`'s -- the ONE host authority over the leaf
    order, shared with the decode family, because the two families write and read one
    plane and two derivations disagree exactly when one is wrong. What this wrapper
    adds is the carry family's own gate: K13 is a floor for the general solver and a
    REQUIREMENT here, since a carry arm whose innermost span is thinner than an atom
    would fold a bucket across two of them.
    """
    out = lattice.box_shape(widths, k, m)
    assert lattice.covers_an_atom(out[1]), (
        "K13: the innermost span must cover an atom; this (k, m) cannot reach it")
    return out


def lattice_of_canonical(widths, k: int, m: int, device="cuda") -> torch.Tensor:
    """``pi``: an ``[N]`` int64 tensor with ``pi[ell] = lambda`` (R0 spec section 2.3).

    A permutation, and only a permutation -- `pi` is a fixed BIT PERMUTATION of the
    leaf index, so this tensor is a convenience for the host seam and never a table
    the device consults.  ``lambda = O*BC + local``, with ``O`` the owner index in
    mixed radix over ``g_l`` and ``local`` the owner's run offsets in mixed radix
    over ``s_l``.  `rola.ops.lattice.permutation` is the derivation; this entry adds
    the carry family's K13 gate.
    """
    box_shape(widths, k, m)
    return lattice.permutation(widths, k, m, device=device)


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


def owner_rows(read_levels, write_levels, widths, *, k: int, m: int):
    """The owner BALLOT ROWS the liveness pass consumes: ``[BH, 2, entries, L/32]``.

    Row ``(side, prefix_entries(l) + o_l)`` bit ``t`` answers "token ``t``'s
    level-``l`` support on this side intersects the owner group ``o_l``'s digit run
    ``[o_l*s_l, (o_l+1)*s_l)``" -- the owner rectangle test, one word per 32 tokens.
    ``entries = sum_l B/s_l`` is SIGMA-shaped: the table does not grow with ``N``.
    """
    return extension().carry_owner_rows(pack_side(read_levels), pack_side(write_levels),
                                        list(widths), int(k), int(m))


def build_stamp(widths, k, m, window, chunk, carve):
    return extension().carry_build_stamp(list(widths), int(k), int(m), int(window),
                                         int(chunk), int(carve))


def arms():
    return extension().carry_arms()
