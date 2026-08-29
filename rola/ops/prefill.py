# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""K31 — the box-native PREFILL: the carry and the intra as one call.

THE TWO KERNELS ARE ONE OPERATOR, and this module is the seam that says so. The
window grid is GLOBAL and FIXED, identical for both: the carry computes every
pair whose write fell in a STRICTLY EARLIER window than its read, and the intra
computes every pair INSIDE one window. Together they are the whole recurrence,
with no pair counted twice and none dropped — the study's exactness clause, which
`tests/oracle/test_carry_vs_oracle.py::test_m2_decomposition_is_exact` states
independently in fp64.

Symbols, each at first use: ``widths`` the per-level digit counts, ``D`` levels,
``k`` the uniform sub-box span, ``m`` the capacity multiplier, ``W`` the window
length in tokens, ``C`` the carry's private batch tile, ``BC`` the leaves one
owner holds, ``d_v = 64`` value channels, ``modes`` the per-level support-mode
declaration the intra's zero certificate reads.

WHY THE INTRA GOES SECOND AND REDUCES.  The intra kernel's write-back is a RED
(``fan_in_add``), not a store, so it ADDS into whatever ``o``/``den`` already
hold. The carry allocates and fills them; the intra then folds its term into the
same buffers on the same stream. Nothing is summed on the host, and there is no
second output pair to reconcile — which is the whole reason the intra was built
as a reducer rather than as a returner.

WHAT THIS OPERATOR REFUSES, and why each refusal is structural rather than a
missing feature:

* ``L % W != 0`` — the intra's tile grid is a COMPILE-TIME count per arm, so a
  partial trailing window has no kernel. (The carry alone tolerates one; the
  combined operator cannot, and a host-side patch for the tail would be a second
  code path for the case the window grid exists to remove.)
* any topology but ``(256, 256)`` — the intra's level width is 256 by
  instantiation. The carry's ``(64, 64)`` arm therefore has no intra partner and
  is an INTER-ONLY conformance cell; see `docs/internals/intra/intra_kernel.md`.
"""
from __future__ import annotations

import torch

from rola.ops import carry as carry_ops
from rola.ops import intra as intra_ops
from rola.ops._ext import extension

#: the built windows, restated from the two families so a caller has one import.
WINDOW = carry_ops.WINDOW
WINDOW_WIDE = carry_ops.WINDOW_WIDE

#: the level widths the intra kernel is instantiated at, and therefore the ones the
#: combined operator has.  A topology is admitted when BOTH families carry an arm at
#: its ``(D, B, W)``; the widths are uniform per topology, as everywhere else.
LEVEL_WIDTH = intra_ops.LEVEL_WIDTH
LEVEL_WIDTHS = (intra_ops.LEVEL_WIDTH, intra_ops.LEVEL_WIDTH_DEEP3, intra_ops.LEVEL_WIDTH_DEEP4)


def prefill(read_levels, write_levels, g_write, v, widths, *, k: int, m: int,
            window: int, modes, sread, swrite, chunk: int = 32,
            state_in: torch.Tensor | None = None,
            state_out: torch.Tensor | None = None,
            page_table: torch.Tensor | None = None,
            atom_bits: torch.Tensor | None = None):
    """The whole forward term, undivided: ``(num, den, state_out)``.

    ``num`` is ``[BH, T, d_v]`` fp32, ``den`` is ``[BH, T]`` fp32 and
    ``state_out`` is ``[BH, N/16, 16, d_v+1]`` fp32 in CANONICAL leaf order.
    The readout a caller wants is ``num / (den + READOUT_EPS)``; this operator
    does not divide, because the state and the mass are what a chained call
    needs and a ratio is not.

    THE STATE PLANES ARE THE CALLER'S (`docs/internals/state.md` section 4).
    ``state_out`` is the plane this call advances -- pass the SAME tensor as
    ``state_in`` to continue a sequence in place, or the arena's plane together with
    its ``page_table`` to run the paged backing; a call that passes none is binding a
    fresh dense backing and gets a plane of its own. ``page_table`` is the
    ``[BH, N/16]`` int32 slot table, and ``None`` is the dense backing, where the slot
    IS the atom. ``atom_bits`` is the facts pass' per-atom ACTIVITY byte (bit 0 WRITTEN,
    bit 1 READ), the one truth both backings gate their state sweeps on
    (:func:`rola.ops.carry.carry_inter`).

    ``modes`` is one :mod:`rola.ops.intra` support mode per level. It is a
    DECLARATION about the routing, not a switch: the intra's slab gate reads
    all-ones for an undeclared side and therefore cannot move a number, so
    ``DENSE_BOTH`` everywhere is always correct and only ever slower.

    ``sread``/``swrite`` are the frozen support words of the two sides
    (``[BH, L/32, sum(widths)]`` int32). THEY ARE AN INPUT, NOT A DERIVATION: the
    routing producer emits exactly this word from the solve that produced the
    amplitudes (``entmax/factor.cu:219-241``), so deriving them here would be a
    second pass over the planes to answer a question already answered. A caller
    that drew its planes instead of solving them has
    :func:`rola.ops.intra.pack_support`.
    """
    D = len(widths)
    if len(modes) != D:
        raise ValueError(f"{D} levels want {D} support modes; got {len(modes)}")
    if len(set(widths)) != 1 or widths[0] not in LEVEL_WIDTHS:
        raise ValueError(
            f"the combined prefill is built at one uniform level width out of "
            f"{LEVEL_WIDTHS} (the intra kernel's instantiations); got widths "
            f"{tuple(widths)}. A topology neither family carries is an inter-only "
            f"cell -- call rola.ops.carry.carry_inter directly.")
    length = read_levels[0].shape[1]
    if length % window:
        raise ValueError(
            f"L = {length} is not a whole number of {window}-token windows; the "
            f"intra kernel's tile grid is a compile-time count per arm.")

    #: THE OPERANDS ARE PACKED ONCE.  Both kernels read the SAME planes, so
    #: routing them through `carry_inter` (which packs its own) would pack them
    #: twice and hand the intra a second, equal-valued copy -- two images of one
    #: operand is how a rounding difference becomes a conformance mystery.
    pread = carry_ops.pack_side(read_levels)
    pwrite = carry_ops.pack_side(write_levels)
    gwrite = g_write.permute(0, 2, 1).contiguous().to(torch.bfloat16)
    gwrite = gwrite.reshape(gwrite.shape[0] * gwrite.shape[1], -1)
    value = v.to(torch.bfloat16).contiguous()

    ext = extension()
    #: THE STATE PLANE IS THE CALLER'S WHEN THE CALLER HAS ONE (state.md section 4): a
    #: continuation hands its own plane -- the arena's, or the one it is advancing --
    #: and it is written in place, so a stateful chain allocates ONE plane for the
    #: SEQUENCE. A call that supplies none is binding a fresh dense backing and gets it.
    if state_out is None:
        state_out = carry_ops.state_plane(widths, pread.shape[0], value.shape[-1],
                                          device=pread.device)
    num, den = ext.carry_forward_inter(
        pread, pwrite, gwrite, value, list(widths), int(k), int(m), int(window),
        int(chunk), carry_ops.LEAF_SPLIT, intra_ops.pack_modes(tuple(modes)), state_in,
        state_out, page_table, atom_bits)
    #: SAME STREAM, SAME BUFFERS.  The intra REDs into `num`/`den`; the stream
    #: order is the only synchronization either kernel needs, and there is no
    #: host round trip between them.
    ext.intra_forward(pread, pwrite, gwrite, value, sread, swrite, num, den, D,
                      int(widths[0]), intra_ops.pack_modes(tuple(modes)), int(window))
    return num, den, state_out


__all__ = ["LEVEL_WIDTH", "LEVEL_WIDTHS", "WINDOW", "WINDOW_WIDE", "prefill",
           "prefill_autograd"]


class _Prefill(torch.autograd.Function):
    """The whole operator as ONE autograd node: three kernels forward, three reverse.

    The node is over the UNDIVIDED ``(num, den, state_out)``, so the division's
    backward is ordinary autograd at whatever forms ``num / (den + eps)`` -- there is
    no seam formula to keep in step with the kernels. The ctx retains the op's own
    inputs and NOTHING ELSE: no state, no lists, no thresholds, no pairwise coefficient.
    """

    @staticmethod
    def forward(ctx, D, widths, k, m, window, modes, chunk, sread, swrite, state_in, g_write, v,
                *levels):
        pread = carry_ops.pack_side(levels[:D])
        pwrite = carry_ops.pack_side(levels[D:])
        gwrite = g_write.permute(0, 2, 1).contiguous().to(torch.bfloat16)
        gwrite = gwrite.reshape(gwrite.shape[0] * gwrite.shape[1], -1)
        value = v.to(torch.bfloat16).contiguous()

        ext = extension()
        state_out = carry_ops.state_plane(widths, pread.shape[0], value.shape[-1],
                                          device=pread.device)
        num, den = ext.carry_forward_inter(
            pread, pwrite, gwrite, value, list(widths), int(k), int(m), int(window),
            int(chunk), carry_ops.LEAF_SPLIT, intra_ops.pack_modes(tuple(modes)), state_in,
            state_out, None)
        ext.intra_forward(pread, pwrite, gwrite, value, sread, swrite, num, den, D,
                          int(widths[0]), intra_ops.pack_modes(tuple(modes)), int(window))

        ctx.save_for_backward(pread, pwrite, gwrite, value, state_in)
        ctx.shape = (v.shape, g_write.shape, tuple(x.shape[-1] for x in levels))
        ctx.dtypes = tuple(x.dtype for x in (g_write, v) + levels)
        ctx.plan = (D, tuple(widths), int(k), int(m), int(window), tuple(modes), int(chunk))
        return num, den, state_out

    @staticmethod
    def backward(ctx, d_num, d_den, d_state_out):
        pread, pwrite, gwrite, value, state_in = ctx.saved_tensors
        D, widths, k, m, window, modes, chunk = ctx.plan
        (Bn, T, H, dv), _, level_widths = ctx.shape
        bh = Bn * H
        if d_num is None:
            d_num = torch.zeros((bh, T, dv), dtype=torch.float32, device=pread.device)
        if d_den is None:
            d_den = torch.zeros((bh, T), dtype=torch.float32, device=pread.device)
        d_num = d_num.contiguous()
        d_den = d_den.contiguous()

        vt = carry_ops.pack_cols(value.permute(0, 2, 1, 3).reshape(bh, T, dv), 1.0)
        dn = carry_ops.pack_cols(d_num, d_den)
        d_pr, d_pw, d_gw, d_v, d_state_in, _ = extension().carry_backward_inter(
            pread, pwrite, gwrite, vt, dn, list(widths), int(k), int(m), int(window),
            int(chunk), state_in, d_state_out, None, False)
        #: SAME STREAM, SAME PLANES.  The intra REDs its within-window half into the
        #: buffers the two scans just wrote, exactly as the forward intra REDs into
        #: `num`/`den`; nothing is summed on the host.
        extension().intra_backward(
            pread, pwrite, gwrite, value, d_num.to(torch.bfloat16).contiguous(),
            d_den.to(torch.float32).contiguous(), d_pr, d_pw, d_gw, d_v, D, int(widths[0]),
            intra_ops.pack_modes(tuple(modes)), int(window))

        def unflatten(plane, base_dtype):
            out, off = [], 0
            for width in level_widths[:D]:
                part = plane[:, :, off:off + width].reshape(Bn, H, T, width)
                out.append(part.permute(0, 2, 1, 3).to(base_dtype))
                off += width
            return out

        g_read = unflatten(d_pr, ctx.dtypes[2])
        g_write_levels = unflatten(d_pw, ctx.dtypes[2 + D])
        g_gain = d_gw.reshape(Bn, H, T).permute(0, 2, 1).to(ctx.dtypes[0])
        g_value = d_v.reshape(Bn, H, T, dv).permute(0, 2, 1, 3).to(ctx.dtypes[1])
        g_state = d_state_in if ctx.needs_input_grad[9] else None
        return (None, None, None, None, None, None, None, None, None, g_state, g_gain, g_value,
                *g_read, *g_write_levels)


def prefill_autograd(read_levels, write_levels, g_write, v, widths, *, k: int, m: int,
                     window: int, modes, sread, swrite, chunk: int = 32,
                     state_in: torch.Tensor | None = None):
    """:func:`prefill`, differentiable -- the same three kernels with R4's reverse pass.

    Returns the same undivided ``(num, den, state_out)``. ``state_in`` and every
    routing operand carry gradients; ``widths``, ``k``, ``m``, ``window``, ``modes``
    and the two support words are configuration and carry none -- a support word is a
    STATEMENT ABOUT where the amplitudes are zero, and the amplitudes it describes are
    what the gradient flows through.
    """
    D = len(widths)
    if len(modes) != D:
        raise ValueError(f"{D} levels want {D} support modes; got {len(modes)}")
    #: THE REVERSE PASS IS BUILT AT DEPTH TWO, WIDTH 256 ONLY.  Both reverse
    #: families refuse a third level -- the digit reduction's pass geometry is a
    #: per-arm derivation -- so the differentiable entry is narrower than the
    #: forward one by exactly that gap, and says so rather than half-serving it.
    if any(w != LEVEL_WIDTH for w in widths) or D != 2:
        raise ValueError(
            f"the differentiable prefill is built at D = 2, level width {LEVEL_WIDTH} "
            f"only (the reverse families' arms); got {D} levels of widths "
            f"{tuple(widths)}. The forward-only `prefill` carries the deep arms.")
    length = read_levels[0].shape[1]
    if length % window:
        raise ValueError(
            f"L = {length} is not a whole number of {window}-token windows; the "
            f"intra kernel's tile grid is a compile-time count per arm.")
    return _Prefill.apply(D, tuple(widths), k, m, window, tuple(modes), chunk, sread, swrite,
                          state_in, g_write, v, *read_levels, *write_levels)
