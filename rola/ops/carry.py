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

#: THE SEAM'S DECLARATION SCALAR: which built (carve, EXECUTOR MAP) an arm is.  A
#: stream owns a box, an executor (a warp) runs streams, and the two counts are
#: independent dials, so an arm is not named by its carve alone.
#:
#: ``LEAF_SPLIT`` is the ruled carve (K33 clean-sheet tier A, measured 2026-08-21) --
#: a stream owns an affine rectangle of the owner's digits and all ``d_v + 1``
#: channels -- run at ONE warp per stream, which is the ONE executor map the
#: transplanted M1a body builds (K35 M1P).  The two CONTAINER values, the PARK bit and
#: the wide value plane named arms whose bodies M1P deleted; each returns as its own
#: stage, with its constant restored beside its arm row.  A caller asking for one now
#: reaches the seam's refusal, which names the whole key.
LEAF_SPLIT = 1

#: THE VALUE WIDTH IS PART OF AN ARM'S KEY (K35 S3b) and is a template parameter of
#: both carry bodies.  M1P builds ONE width: the ``d_v`` dial returns as its own stage.
D_V = 64

#: the ``(k, m)`` capacity that reaches the built box size on the flagship topology,
#: ``BC = 128``.
CAP_BC128 = 8

#: the flagship window, and the conformance cell's -- the architecture doc's
#: two-constraint sizing (integration rung I1(b)); `rola.ops.intra` names the
#: SAME two, because the two kernels run one window grid.
WINDOW = 384
WINDOW_WIDE = 512
WINDOW_SMALL = 64

#: the carry kernel's device build magic, mirrored from `carry_kernel.cuh`. It
#: moves whenever that source does, which is what makes reading it back off the
#: DEVICE an extension-trap check rather than a path check.
BUILD_MAGIC = 31010026

#: the reverse family's own device magic, mirrored from `backward_kernel.cuh`.
BWD_BUILD_MAGIC = 31040001

MMA_K_QUANTUM = lattice.MMA_K_QUANTUM


def box_shape(widths, k: int, m: int):
    """``(m_l, s_l, g_l, BC, owners)`` for the ``(k, m)`` box, with K2 ASSERTED.

    The derivation is `rola.ops.lattice`'s -- the ONE host authority over the leaf
    order, shared with the decode family, because the two families write and read one
    plane and two derivations disagree exactly when one is wrong. What this wrapper
    adds is the carry family's own gate: an owner's box is a whole number of atoms, so
    a fold bucket never straddles two of them and the state plane's
    ``[BH, N/16, 16, cols]`` shape is a re-blocking of the owner grid.
    """
    out = lattice.box_shape(widths, k, m)
    assert out[3] % MMA_K_QUANTUM == 0, (
        f"K2: an owner box of {out[3]} leaves must be a whole number of "
        f"{MMA_K_QUANTUM}-leaf atoms")
    return out


def lattice_of_canonical(widths, k: int, m: int, device="cuda") -> torch.Tensor:
    """``pi``: an ``[N]`` int64 tensor with ``pi[ell] = lambda`` (R0 spec section 2.3).

    A permutation, and only a permutation -- `pi` is a fixed BIT PERMUTATION of the
    leaf index, so this tensor is a convenience for the host seam and never a table
    the device consults.  ``lambda = O*BC + local``, with ``O`` the owner index in
    mixed radix over ``g_l`` and ``local`` the owner's run offsets in mixed radix
    over ``s_l``.  `rola.ops.lattice.permutation` is the derivation; this entry adds
    the carry family's K2 gate.
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
                window: int, chunk: int = 32, carve: int = LEAF_SPLIT,
                level_modes: int = 0, state_in: torch.Tensor | None = None):
    """The INTER term, undivided: ``(num, den, state_out)``.

    ``num`` is ``[BH, T, d_v]`` fp32, ``den`` is ``[BH, T]`` fp32, and
    ``state_out`` is ``[BH, N/16, 16, d_v+1]`` fp32 in LATTICE leaf order.
    ``state_in``, when given, is the same layout and order.  ``d_v`` is READ OFF ``v``
    and is part of the arm key, so a width no arm declares is refused, not degraded.
    """
    pr = pack_side(read_levels)
    pw = pack_side(write_levels)
    gw = g_write.permute(0, 2, 1).contiguous().to(torch.bfloat16)
    gw = gw.reshape(gw.shape[0] * gw.shape[1], -1)
    return extension().carry_forward_inter(
        pr, pw, gw, v.to(torch.bfloat16).contiguous(), list(widths), int(k), int(m),
        int(window), int(chunk), int(carve), int(level_modes), state_in)


def owner_rows(read_levels, write_levels, widths, *, k: int, m: int, carve: bool = False):
    """The BALLOT ROWS the liveness pass consumes: ``[BH, entries_read + entries_write, L/32]``.

    Row ``prefix_entries(l) + o_l`` of a side's block, bit ``t``, answers "token ``t``'s
    level-``l`` support on this side intersects the group ``o_l``'s digit run" -- one
    word per 32 tokens. ``entries = sum_l B/s_l`` is SIGMA-shaped: the table does not
    grow with ``N``. THE TWO SIDES CARRY THEIR OWN PARTITION (K35 S1b): a bitlist is
    voted from one side's own factors, so the read block is at the read carve's runs and
    the write block at the write carve's, and a level a side spans whole emits no rows.
    This diagnostic asks for ONE grain on both sides, so its two blocks coincide:
    ``carve`` picks the split of the innermost level's owner run into four groups, whose
    owner-grain word is then the OR of a group's four rows.
    """
    return extension().carry_owner_rows(pack_side(read_levels), pack_side(write_levels),
                                        list(widths), int(k), int(m), bool(carve))


def build_stamp(widths, k, m, window, chunk, carve, level_modes=0, d_v=D_V):
    """The device magic, the caps digest and the arm's census.

    Fields 3.. are the census: registers, SMEM (attributes), binary version, SMEM
    (geometry), max threads, local bytes, ``BC``, owners, ring depth, CTA target,
    the read-carve level, the WRITE STREAM count and the arm's ``d_v`` -- so the
    THREE dials (executors ``stamp[7] // 32``, streams ``stamp[14]``, width
    ``stamp[15]``) are all readable off the device.

    Fields 16..25 are THE DERIVATION READ BACK (K35 S1b): the four write spans
    ``stamp[16:20]``, the four read spans ``stamp[20:24]``, the exchange's TRANSIT in
    bytes ``stamp[24]``, and the exchange's STATE ``stamp[25]`` -- ``0`` elided, ``1``
    full, ``2`` full only because case (c)'s register reorder is not built (so the row
    OVERPAYS and can be tabled), ``-1`` an arm with no exchange at all.  The spans are no longer declared -- they are what the
    order-by-mode rule cut out of ``(level_modes, stream count)`` -- so reading them off
    the device is the only way to see what an arm actually holds.  The transit is ZERO on
    an arm whose two carves came out identical: the permutation is then the identity and
    the exchange is elided at compile time.
    """
    return extension().carry_build_stamp(list(widths), int(k), int(m), int(window),
                                         int(chunk), int(carve), int(level_modes), int(d_v))


def arms():
    return extension().carry_arms()


#: THE REVERSE PASS'S PRE-PACKED PLANE WIDTH.  The value and seed rows are handed to
#: the two scans `cols`-wide and zero-padded to whole MMA n-tiles, so one staging body
#: serves both and the mass channel is an ordinary column rather than a carve.
BWD_COLS_PAD = 72
BWD_LD = 80


def pack_cols(main: torch.Tensor, last) -> torch.Tensor:
    """``[BH, L, d_v]`` plus a last column as one ``[BH, L, BWD_LD]`` bf16 plane.

    ``last`` is either a ``[BH, L]`` tensor (the seed's ``d_den``) or a scalar (the
    value's ones column). Everything past ``d_v + 1`` is zero, which is what lets the
    kernels read the final n-tile with no partial-tile path.
    """
    bh, length, d_v = main.shape
    out = torch.zeros((bh, length, BWD_LD), dtype=torch.bfloat16, device=main.device)
    out[:, :, :d_v] = main.to(torch.bfloat16)
    if torch.is_tensor(last):
        out[:, :, d_v] = last.to(torch.bfloat16)
    else:
        out[:, :, d_v] = float(last)
    return out


def carry_backward(read_levels, write_levels, g_write, v, d_num, d_den, widths, *, k: int,
                   m: int, window: int, chunk: int = 32, state_in=None, d_state_out=None,
                   want_replay: bool = False):
    """The CROSS-WINDOW reverse pass: ``(d_pr, d_pw, d_gw, d_v, d_state_in, replay)``.

    ``d_num``/``d_den`` are the seeds the op seam forms from the division's backward,
    shaped like the forward's ``num``/``den``. ``d_pr``/``d_pw`` are
    ``[BH, T, sum_l width_l]`` fp32 on the packed amplitude layout, ``d_v`` is
    ``[BH, T, d_v]``, ``d_gw`` is ``[BH, T]``, and ``d_state_in``/``replay`` are
    ``[BH, N/16, 16, d_v+1]`` fp32 in LATTICE leaf order.
    """
    pr = pack_side(read_levels)
    pw = pack_side(write_levels)
    gw = g_write.permute(0, 2, 1).contiguous().to(torch.bfloat16)
    gw = gw.reshape(gw.shape[0] * gw.shape[1], -1)
    bn, t, h, d_v = v.shape
    vt = pack_cols(v.permute(0, 2, 1, 3).reshape(bn * h, t, d_v), 1.0)
    dn = pack_cols(d_num, d_den)
    return extension().carry_backward_inter(
        pr, pw, gw, vt, dn, list(widths), int(k), int(m), int(window), int(chunk),
        state_in, d_state_out, bool(want_replay))


def backward_stamp(widths, k, m, window, chunk):
    return extension().carry_backward_stamp(list(widths), int(k), int(m), int(window), int(chunk))
