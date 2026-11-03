# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PREFILL COMPOSITION: one segment's kernels over the bundle and the epilogue that joins them.

THE KERNELS ARE ONE OPERATOR on one window grid (`rola.ops.window.window_of`, the device's): the carry computes every
pair whose write fell in a strictly earlier window than its read -- its ``num`` and, as the v+1 channel, its ``den`` --
and the intra every pair inside one window. Together they are the whole recurrence, no pair counted twice and none
dropped (`tests/oracle/test_carry_vs_oracle.py`'s decomposition test states it in fp64). The scale pass gives the
carry's f16 sum across CTAs each token's scale byte (`rola.ops.scale`) before the carry runs.

THE UNITS. Every kernel leaves each token's READER FACTOR ``r`` (its read levels' scales) out of its sums, and the
readout divides by it (:func:`readout`): ``y = num / (den + eps / r)`` is the defined readout exactly. The intra's
``num`` and ``den`` are already in those units; the carry's take its state's unit and its read rows'
(:func:`carry_factor`), ``num`` times the f16 sum's scale ``2^(s + 1)`` where it is summed as f16 pairs, ``s`` each
token's scale byte -- every CTA's partials meet at it.

Symbols, each at first use: ``W`` the window, ``L`` tokens, ``BH`` batch x heads, ``d_v`` value channels, level ``S``
the carry's (`rola.routing.factors.carry_level_s`) and ``R`` the other.
"""
from __future__ import annotations

import torch

from rola.ops import carry as carry_ops
from rola.ops import intra as intra_ops
from rola.ops import scale as scale_ops
from rola.routing import quant

__all__ = ["UNIT_HEADROOM", "UNIT_LEVELS", "UNIT_NONE", "carry_factor", "prefill", "reader_factor", "readout",
           "unpack_num"]

#: THE f16 SUM's SCALE BYTE, the kernels' `unit_byte` rule (`csrc/rola/src/carry/carry.cuh`): the smallest ``s`` with
#: ``1.02 P <= 2^(s + 15)`` for the token's bound ``P``, held to int8, `UNIT_NONE` where the token reads nothing
UNIT_LEVELS, UNIT_HEADROOM, UNIT_NONE = 15, 1.02, -100


def _u8(rows: torch.Tensor) -> bool:
    return rows.dtype == quant.ROW_DTYPE


def carry_factor(routes) -> float:
    """What the carry's partials at a token's unit are multiplied by to be the recurrence's numerator over ``r``: the
    state's byte scale (level S's write rows a byte over 256 or an f16, over 255) and the read rows' storage -- level
    S's coefficient a byte over 256 (``256 / 255`` of its unit value), level R's row a byte (``1 / 255``)."""
    ls = routes.level_s
    lr = 1 - ls
    kappa = (256.0 / quant.AMP_LEVELS if _u8(routes.amp_write[ls]) else 1.0) / quant.AMP_LEVELS
    k_s = (quant.AMP_LEVELS + 1.0) / quant.AMP_LEVELS if _u8(routes.amp_read[ls]) else 1.0
    k_r = 1.0 / quant.AMP_LEVELS if _u8(routes.amp_read[lr]) else 1.0
    return kappa * k_s * k_r


def unpack_num(num: torch.Tensor, channels: int, factor: torch.Tensor | float = 1.0) -> torch.Tensor:
    """``[BH, L, d_v]`` fp32 in channel order, times ``factor`` (``[BH, L]`` a token's, or one float): the carry's
    ``num`` as the drain wrote it -- fp32, or int32 words of f16 pairs, a box's word ``8 m + g`` holding channels ``(c,
    c + 2)``, ``c = 8 (g >> 1) + 4 m + (g & 1)`` of its ``channels`` (`docs/internals/carry/parts.md#image`). The pairs
    are a fixed bit pattern -- word ``8 m + 2 a + b``'s halves ``h`` are channel ``8 a + 4 m + 2 h + b`` -- so the words
    read as f16 elements ``[m][a][b][h]`` a box and ONE permuted multiply writes them scaled in channel order
    ``[a][m][h][b]``: one pass over the planes."""
    tokenwise = isinstance(factor, torch.Tensor)
    if num.dtype == torch.float32:
        return num * (factor.unsqueeze(-1) if tokenwise else factor)
    if channels != 32:
        raise ValueError(f"the drain's f16 pairs are a 32-channel box's, got a box of {channels} channels")
    BH, L, words = num.shape
    boxes = words // 16
    pairs = num.view(torch.float16).view(BH, L, boxes, 2, 4, 2, 2)
    out = torch.empty(BH, L, boxes, 4, 2, 2, 2, dtype=torch.float32, device=num.device)
    scale = factor.view(BH, L, 1, 1, 1, 1, 1) if tokenwise else factor
    torch.mul(pairs.permute(0, 1, 2, 4, 3, 6, 5), scale, out=out)
    return out.view(BH, L, 2 * words)


def prefill(routes, values, facts, carve, layout, read_statics, write_statics, *, accumulate: str | None = None,
            reduce_dtype: torch.dtype | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """``(num, den)`` fp32 ``[BH, L, d_v]`` and ``[BH, L]`` over ``r`` (:func:`readout` divides by it): the scale pass,
    the carry at its bytes (its ``num`` and its ``den``), and the intra reduced into the same planes. ``facts`` is the
    facts pass's output over this bundle in ``layout`` with the carve's read row (`rola.ops.carry.form`) and these
    statics. ``accumulate`` and ``reduce_dtype`` are reserved: the gates' choice of the MMAs' accumulation and the
    carry's cross-CTA dtype, each the device's or the carve's rule by default."""
    unit = carry_factor(routes)
    scale = scale_ops.scale(routes, facts, unit)
    num_c, den_c, _ = carry_ops.carry(routes, values, facts, carve, scale, accumulate=accumulate,
                                      reduce_dtype=reduce_dtype)
    #: the f16 sum's tokens at ``2^(s + 1)`` times the unit, an fp32 sum's at the unit: one pass with the unpack
    factor = torch.exp2(scale.float() + 1.0) * unit if num_c.dtype != torch.float32 else unit
    num = unpack_num(num_c, carve.channels, factor)
    den = (den_c * unit).contiguous()
    intra_ops.intra_forward(routes, values, facts, layout, read_statics, write_statics, num, den,
                            accumulate=accumulate)
    return num, den


def reader_factor(routes) -> torch.Tensor:
    """``[BH, L]``: each token's read scales multiplied -- the per-token factor the kernels leave out of ``num`` and
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
