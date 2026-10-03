# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The box-native PREFILL: the carry and the intra as one call.

THE TWO KERNELS ARE ONE OPERATOR, and this module is the seam that says so. The
window grid is GLOBAL and FIXED, identical for both: the carry computes every
pair whose write fell in a STRICTLY EARLIER window than its read, and the intra
computes every pair INSIDE one window. Together they are the whole recurrence,
with no pair counted twice and none dropped — the study's exactness clause, which
`tests/oracle/test_carry_vs_oracle.py::test_m2_decomposition_is_exact` states
independently in fp64.

THE INTRA READS THE BUNDLE (`rola.ops.intra`): its rows and values as the producer wrote
them, and the facts pass's words for its gate. THE CARRY LEG IS `rola.ops.carry`, still on
floating planes until its int8 port: this module dequantizes the bundle for it once a call
-- the read side at its unit rows, since the reader's per-token factor cancels in
``num / den`` and the intra drops it too (the output contract) -- builds that surface's
descriptor, geometry block and launch shape, takes the liveness words and activity bits
from its caller, and calls `carry_forward`. The window is a kernel constant
(`carry_ops.WINDOW`), not a parameter of this operator.

Symbols, each at first use: ``D`` levels, ``W`` the window length in tokens (the carry's
constant), ``d_v`` value channels, ``modes`` the per-level support-mode declaration the
carry's carve order reads.

WHY THE INTRA GOES SECOND AND REDUCES.  The intra kernel's write-back is a RED
(``fan_in_add``), not a store, so it ADDS into whatever ``num``/``den`` already
hold. The carry allocates and fills them; the intra then folds its term into the
same buffers on the same stream.

WHAT THIS OPERATOR REFUSES, and why each refusal is structural rather than a
missing feature:

* ``L % W != 0`` — the intra's windows are whole, its grid a window a column.
* level widths or a value width the intra does not read (`rola.ops.intra.admits`).
"""
from __future__ import annotations

import torch

from rola._state import CANONICAL_ORDER, PAGE_RECTANGLE_BITS, SPLIT_PLANE_DTYPE, StateFormat
from rola.ops import carry as carry_ops
from rola.ops import intra as intra_ops
from rola.ops import padding as padding_ops
from rola.routing import quant

#: THE WINDOW, restated so a caller has one import. It is the CARRY KERNEL's constant
#: (the axis law), and the intra family must carry an arm at the same number, because
#: the two kernels run ONE window grid.
WINDOW = carry_ops.WINDOW


def _padded_shape(widths, dv: int) -> padding_ops.PaddedShape:
    """This call's `rola.ops.padding.PaddedShape` -- every level widened to
    `padded_level_width` and `dv` widened to the smallest of `carry_ops.SHIPPED_DV`
    that admits it (KERNEL_STANDARDS §the "two contracts": the carry surface's
    kernel boundary refuses anything but its own strict shape, so this seam's
    descriptor and operand packing pad BEFORE either helper builds anything the
    boundary would see). A caller already at a shipped shape pays nothing here --
    `PaddedShape.of` is a no-op past an already-padded width.
    """
    return padding_ops.PaddedShape.of(tuple(widths), int(dv), shipped_dv=carry_ops.SHIPPED_DV)


def _descriptor(widths, dv: int, bh: int) -> StateFormat:
    """The call's own state descriptor -- the same fields `rola._state._format_of`
    derives for a routed call, built here from what this operator already has.
    `widths`/`dv` are padded first (:func:`_padded_shape`), so the descriptor always
    names a shape the carry surface's kernel boundary accepts -- a caller's logical
    shape never reaches `StateFormat` directly."""
    shape = _padded_shape(widths, dv)
    n = 1
    for width in shape.padded_widths:
        n *= width
    return StateFormat(D=len(shape.padded_widths), B=shape.padded_widths, order=CANONICAL_ORDER,
                       page_bits=PAGE_RECTANGLE_BITS, DV=shape.DV, dtype=SPLIT_PLANE_DTYPE,
                       ids=int(bh) * (n >> PAGE_RECTANGLE_BITS))


def _conservative_activity(descriptor: StateFormat, bh: int, device) -> torch.Tensor:
    """Every page READ and WRITTEN: the conservative statement this seam makes when
    no facts-pass output is available for the call. It asserts nothing about the
    draw and therefore skips nothing (`measure.cells.conservative_activity`
    states the same fact for a cell)."""
    pages = descriptor.N // carry_ops.PAGE_LEAVES
    return torch.full((bh, pages), carry_ops.ACTIVITY_READ | carry_ops.ACTIVITY_WRITTEN,
                      dtype=torch.uint8, device=device)


def _pack_operands(read_levels, write_levels, g_write, v):
    """The carry's planes, packed ONCE: `(pread, pwrite, gwrite, v_bh)`, ``v_bh`` the carry
    surface's `[BH, L, DV]`.

    THE LEVELS AND ``v`` ARE PADDED FIRST (:func:`_padded_shape`, `rola.ops.padding`):
    `pad_routes` widens each level's trailing dimension to its `PaddedShape.
    padded_widths` entry with exact-zero columns before `carry_ops.pack_side` packs
    it, and `pad_v` widens `v`'s value axis to `PaddedShape.DV` the same way -- a
    logical `(31, 45)`/`d_v=48` call is packed at the padded shape the carry surface's
    kernel boundary accepts, and an already-shipped call pays nothing (`pad_routes`/
    `pad_v` are no-ops past an already-padded width).
    """
    widths = tuple(level.shape[-1] for level in read_levels)
    shape = _padded_shape(widths, v.shape[-1])
    pread = carry_ops.pack_side(padding_ops.pad_routes(read_levels, shape.padded_widths))
    pwrite = carry_ops.pack_side(padding_ops.pad_routes(write_levels, shape.padded_widths))
    gwrite = g_write.permute(0, 2, 1).contiguous().to(torch.bfloat16)
    gwrite = gwrite.reshape(gwrite.shape[0] * gwrite.shape[1], -1)
    v_bthd = padding_ops.pad_v(v.to(torch.bfloat16).contiguous(), shape.DV)
    Bn, T, H, dv = v_bthd.shape
    v_bh = v_bthd.permute(0, 2, 1, 3).contiguous().reshape(Bn * H, T, dv)
    return pread, pwrite, gwrite, v_bh


def intra_admits(widths, d_v: int) -> bool:
    """Whether the intra reads these level widths and value width (`rola.ops.intra.admits`)."""
    return intra_ops.admits(widths, d_v)


def _carry_levels(routes, values):
    """The carry leg's floating operands from the bundle, dequantized once: the read side at its unit rows (the
    reader's factor cancels in ``num / den``, and the intra drops it), the write side and the values whole."""
    read = tuple(quant.dequantize_rows(rows, torch.ones_like(scale) * (scale > 0), not sparse)
                 for rows, scale, sparse in zip(routes.amp_read, routes.scale_read, routes.sparse[0], strict=True))
    write = tuple(quant.dequantize_rows(rows, scale, not sparse)
                  for rows, scale, sparse in zip(routes.amp_write, routes.scale_write, routes.sparse[1], strict=True))
    return read, write, values.dequantized()


def reader_factor(routes) -> torch.Tensor:
    """``[BH, T]``: each token's read scales multiplied -- the per-token factor the kernels leave out of ``num`` and
    ``den``."""
    r = routes.scale_read[0]
    for scale in routes.scale_read[1:]:
        r = r * scale
    return r.permute(0, 2, 1).reshape(routes.batch * routes.heads, routes.tokens)


def readout(num: torch.Tensor, den: torch.Tensor, routes) -> torch.Tensor:
    """THE READOUT, ``y = num / (den + eps)`` of the full sums, ``[B, T, H, d_v]`` fp32. The kernels' sums leave out
    each token's reader factor ``r`` (it cancels in the ratio), so the floor divides by it: ``num' / (den' + eps / r)``
    is the defined readout exactly, and a token that reads nothing (``r = 0``) reads zero."""
    from rola.ops.constants import READOUT_EPS

    y = num / (den + READOUT_EPS / reader_factor(routes)).unsqueeze(-1)
    return y.reshape(routes.batch, routes.heads, routes.tokens, -1).permute(0, 2, 1, 3)


def prefill(routes, values, *, modes, liveness: carry_ops.LivenessWords, unions: torch.Tensor, layout, read_statics,
            write_statics,
            weight_digits: int = intra_ops.DEFAULT_WEIGHT_DIGITS,
            state_in: torch.Tensor | None = None,
            state_out: torch.Tensor | None = None,
            page_table: torch.Tensor | None = None,
            atom_bits: torch.Tensor | None = None):
    """The whole forward term, undivided: ``(num, den, state_out)``.

    ``num`` is ``[BH, T, d_v]`` fp32, ``den`` is ``[BH, T]`` fp32 and
    ``state_out`` is ``[BH, N/16, 16, d_v+1]`` fp32 in CANONICAL leaf order -- the
    caller's own ``state_out`` plane, or ``None`` where the call stores no state.
    The readout a caller wants is ``num / (den + READOUT_EPS)``; this operator
    does not divide, because the state and the mass are what a chained call
    needs and a ratio is not.

    THE STATE PLANES ARE THE CALLER'S (`docs/internals/state.md` section 4).
    THE FOUR SHAPES, the kernel's own, and this operator allocates no plane: neither
    plane is the NULL-STATE call (nothing in, nothing out); ``state_in`` alone is
    READ-ONLY (the readout against a frozen state, the folds never stored);
    ``state_out`` alone is a FRESH sequence (a zero state in, no entry sweep, stored
    out); both, the SAME tensor, ADVANCE it in place. A plane is the arena's, with its
    ``page_table`` for the paged backing, or a dense one from
    :func:`rola.ops.carry.state_plane`. ``page_table`` is the
    ``[BH, N/16]`` int32 slot table, and ``None`` is the dense backing, where the slot
    IS the atom. ``atom_bits`` is the per-atom ACTIVITY byte (bit 0 WRITTEN, bit 1
    READ); a call that passes none states the conservative fact
    (:func:`_conservative_activity`) rather than a derivation this seam does not own.

    ``modes`` is one :mod:`rola.ops.intra` support mode per level. It is a
    DECLARATION about the routing, not a switch: the intra's slab gate reads
    all-ones for an undeclared side and therefore cannot move a number, so
    ``DENSE_BOTH`` everywhere is always correct and only ever slower.

    ``liveness`` is both sides' class-1 words in ``layout`` and ``unions`` their tile
    unions, the facts pass's (:func:`rola.ops.facts.facts`) over this bundle with
    ``read_statics`` and ``write_statics``. ``weight_digits`` is the intra value
    product's declared digit count.
    """
    widths = routes.topology.widths
    D, d_v, length = len(widths), values.d_v, routes.tokens
    if len(modes) != D:
        raise ValueError(f"{D} levels want {D} support modes; got {len(modes)}")
    if not intra_admits(widths, d_v):
        raise ValueError(
            f"the intra does not read level widths {tuple(widths)} at d_v {d_v} (one or two levels of "
            f"{intra_ops.WIDTH_GRANULE}-digit granules up to {intra_ops.MAX_LEVEL_WIDTH}, d_v "
            f"{intra_ops.VALUE_WIDTH}). A topology only the carry carries is an inter-only "
            f"cell -- call rola.ops.carry.carry_forward directly.")
    if length % WINDOW:
        raise ValueError(f"L = {length} is not a whole number of {WINDOW}-token windows")

    read_levels, write_levels, v = _carry_levels(routes, values)
    pread, pwrite, gwrite, v_bh = _pack_operands(read_levels, write_levels, routes.g_write, v)
    bh, dv = v_bh.shape[0], v_bh.shape[2]

    descriptor = _descriptor(widths, dv, bh)
    launch = carry_ops.LaunchShape()
    geometry = carry_ops.geometry_block(descriptor, launch,
                                        intra_ops.pack_modes(tuple(modes)))
    activity = (atom_bits if atom_bits is not None
               else _conservative_activity(descriptor, bh, v_bh.device))
    #: THE STATE PLANES ARE THE CALLER'S, both of them, and this operator allocates none:
    #: the kernel's four shapes pass through as the caller names them (state.md section 4).

    planes = carry_ops.RoutePlanes(read=pread, write=pwrite, gain=gwrite)
    num, den = carry_ops.carry_forward(
        planes, v_bh, descriptor=descriptor, geometry=geometry, liveness=liveness,
        activity=activity, launch=launch, state_in=state_in, state_out=state_out,
        page_table=page_table)
    #: SAME STREAM, SAME BUFFERS.  The intra REDs into `num`/`den`; the stream
    #: order is the only synchronization either kernel needs.
    intra_ops.intra_forward(routes, values, unions, layout, read_statics, write_statics, num, den,
                            weight_digits=weight_digits)
    return num, den, state_out


__all__ = ["WINDOW", "intra_admits", "prefill", "reader_factor", "readout"]

