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

#: THE BODY, AND IT IS THE ONLY DECLARATION LEFT AT THIS SEAM.  K50 turned the carve
#: into a DERIVATION -- a stream owns an affine rectangle of the owner's digits, cut
#: out of ``(level_modes, stream count)`` at the prologue -- so there is no carve to
#: declare.  What is still a choice is which BODY runs: ``CARVED`` is the design's
#: (one warp per stream, the transit, the two-sided order), ``CHANNEL_SPLIT`` is the
#: dense control that keeps the whole box in every warp and splits the channels.  The
#: split body is deleted at G4; until then a caller names it here.
CARVED = False
CHANNEL_SPLIT = True

#: THE VALUE WIDTH IS PART OF AN ARM'S KEY (K35 S3b) and is a template parameter of
#: both carry bodies.  M1P builds ONE width: the ``d_v`` dial returns as its own stage.
D_V = 64

#: THE OWNER'S BOX, in leaves.  It is a REGISTER SHAPE (K50's register law: leaves per
#: warp times the CTA's warps), not a routing declaration, and no caller passes it: the
#: seam reads it off the arm and derives the spans that fill it.
BC = 128

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


def state_plane(widths, BH: int, d_v: int = D_V, device="cuda") -> torch.Tensor:
    """A DENSE state plane, ``[BH, N/16, 16, d_v+1]`` fp32 in CANONICAL leaf order.

    The dense BACKING, byte for byte the atom-major view of ``[BH, N, cols]``
    (`docs/internals/state.md` section 3). It is the caller's to allocate because the
    planes are the caller's: a continuation hands the SAME tensor in and out and
    advances it in place, which is what keeps a stateful chain to ONE plane per
    sequence. A PAGED call passes the arena's plane and its table instead, and the
    kernel is the same kernel -- one base translation, not a second path.
    """
    n = 1
    for w in widths:
        n *= int(w)
    return torch.zeros((BH, n // MMA_K_QUANTUM, MMA_K_QUANTUM, d_v + 1),
                       dtype=torch.float32, device=device)


def carry_inter(read_levels, write_levels, g_write, v, widths, *,
                window: int, chunk: int = 32, split: bool = False,
                level_modes: int = 0, state_in: torch.Tensor | None = None,
                state_out: torch.Tensor | None = None,
                page_table: torch.Tensor | None = None,
                atom_bits: torch.Tensor | None = None):
    """The INTER term, undivided: ``(num, den, state_out)``.

    ``num`` is ``[BH, T, d_v]`` fp32 and ``den`` is ``[BH, T]`` fp32. ``d_v`` is READ
    OFF ``v`` and is part of the arm key, so a width no arm declares is refused, not
    degraded.

    THE DIGIT WIDTHS AND THE MODE WORD ARE DATA (K50). An arm is a SHAPE --
    ``(D, d_v, window, chunk, warps, WPS, body)`` -- and ``widths``/``level_modes``
    are derived into the addressing block at the seam, which refuses a topology the
    shape cannot hold rather than asking for an arm nobody compiled. ``split`` picks
    the CHANNEL-SPLIT body (the dense control, deleted at G4) over the carved one;
    it is a body, not a geometry.

    THE STATE PLANES ARE THE CALLER'S (`docs/internals/state.md` section 4).
    ``state_in``/``state_out`` are ``[BH, N/16, 16, d_v+1]`` fp32 in CANONICAL leaf order
    -- `state_plane` builds a dense one, an arena hands over ``arena.state`` -- and a
    CONTINUATION passes the SAME tensor as both, advancing it in place. ``page_table``
    is the ``[BH, N/16]`` int32 slot table; ``None`` is the dense backing, where the
    slot IS the atom, so the two backings are one kernel, one ABI and one accumulation
    order. Passing NEITHER plane is the NULL-STATE call: it reaches an arm variant with
    no state I/O compiled into it and returns ``state_out`` as ``None``.

    ``atom_bits`` is the facts pass' ``[BH, N/16]`` uint8 ACTIVITY bitmap -- bit 0
    WRITTEN, bit 1 READ, one byte per atom (`rola.engine.facts.planes.atom_bits`). The
    kernel loads an atom iff it is resident AND (read or written) and stores it iff
    written, identically under both backings: residency says WHERE an atom is, activity
    says WHETHER this call touches it. ``None`` asserts nothing and every resident atom
    is swept, which is the pre-P80b behaviour and what a caller with no facts pass gets.
    """
    pr = pack_side(read_levels)
    pw = pack_side(write_levels)
    gw = g_write.permute(0, 2, 1).contiguous().to(torch.bfloat16)
    gw = gw.reshape(gw.shape[0] * gw.shape[1], -1)
    num, den = extension().carry_forward_inter(
        pr, pw, gw, v.to(torch.bfloat16).contiguous(), list(widths),
        int(window), int(chunk), int(bool(split)), int(level_modes), state_in, state_out,
        page_table, atom_bits)
    return num, den, state_out


def owner_rows(read_levels, write_levels, widths, *, level_modes: int = 0,
               carve: bool = False):
    """The BALLOT ROWS the liveness pass consumes: ``[BH, entries_read + entries_write, L/32]``.

    Row ``prefix_entries(l) + o_l`` of a side's block, bit ``t``, answers "token ``t``'s
    level-``l`` support on this side intersects the group ``o_l``'s digit run" -- one
    word per 32 tokens. ``entries = sum_l B/s_l`` is SIGMA-shaped: the table does not
    grow with ``N``. THE TWO SIDES CARRY THEIR OWN PARTITION (K35 S1b): a bitlist is
    voted from one side's own factors, so the read block is at the read carve's runs and
    the write block at the write carve's, and a level a side spans whole emits no rows.
    ``carve`` picks the GRAIN: ``False`` is the owner's own box (one stream a side),
    ``True`` is the carved body's own (four streams a side, ordered by ``level_modes``).
    Both come from the same derivation the kernel runs.
    """
    return extension().carry_owner_rows(pack_side(read_levels), pack_side(write_levels),
                                        list(widths), int(level_modes), bool(carve))


def build_stamp(widths, window, chunk, split=False, level_modes=0, d_v=D_V):
    """The device magic, the caps digest and the arm's census.

    Fields 3.. are the census: registers, SMEM (attributes), binary version, SMEM
    (geometry), max threads, local bytes, ``BC``, owners, ring depth, CTA target,
    the read-carve level, the WRITE STREAM count and the arm's ``d_v`` -- so the
    THREE dials (executors ``stamp[7] // 32``, streams ``stamp[14]``, width
    ``stamp[15]``) are all readable off the device.

    Fields 16..25 are THE DERIVATION READ BACK, and after K50 it is the RUNTIME
    derivation: the four write spans ``stamp[16:20]``, the four read spans
    ``stamp[20:24]``, the transit's bytes ``stamp[24]`` (ZERO only on the channel-split
    body, which has no transit -- the carved body always runs one), and ``stamp[25]``,
    which is ``1`` when the transit's ROW MAP IS THE IDENTITY. That last field is the
    former ``kTied``: K50 deleted the elided body ("one transit, always"), so a tie is
    now a reported FACT that makes a wasted transit visible in the ledger, never a
    second code path.
    """
    return extension().carry_build_stamp(list(widths), int(window), int(chunk),
                                         int(bool(split)), int(level_modes), int(d_v))


def geometry(widths, level_modes=0, bc=128, nsr=4, nsw=4, d_v=64):
    """THE ADDRESSING BLOCK the dispatch derives, flattened -- the model test's subject.

    This is the SHIPPED derivation, reached through the extension, not a Python mirror
    of it: a mirror can only prove that two derivations agree with each other
    (`matching-the-naive-antipattern`).
    """
    return extension().carry_geometry(list(widths), int(level_modes), int(bc), int(nsr),
                                      int(nsw), int(d_v))


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
                   page_table=None, want_replay: bool = False):
    """The CROSS-WINDOW reverse pass: ``(d_pr, d_pw, d_gw, d_v, d_state_in, replay)``.

    ``d_num``/``d_den`` are the seeds the op seam forms from the division's backward,
    shaped like the forward's ``num``/``den``. ``d_pr``/``d_pw`` are
    ``[BH, T, sum_l width_l]`` fp32 on the packed amplitude layout, ``d_v`` is
    ``[BH, T, d_v]``, ``d_gw`` is ``[BH, T]``, and ``d_state_in``/``replay`` are
    ``[BH, N/16, 16, d_v+1]`` fp32 in CANONICAL leaf order.

    ``page_table`` resolves the two ENTRY slabs the caller owns (``state_in`` and
    ``d_state_out``) through the same slot table the forward uses. The two EXIT slabs
    -- ``d_state_in`` and ``replay`` -- are this dispatch's own dense allocations and
    are addressed densely; paging a gradient plane needs a plan of its own and is the
    reverse family's stage, not this one's.
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
        state_in, d_state_out, page_table, bool(want_replay))


def backward_stamp(widths, k, m, window, chunk):
    return extension().carry_backward_stamp(list(widths), int(k), int(m), int(window), int(chunk))
