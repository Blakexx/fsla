# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PREFILL COMPOSITION: one segment's kernels over the bundle and the epilogue that joins them.

THE KERNELS ARE ONE OPERATOR on one window grid (`rola.ops.window.window_of`, the device's): the carry computes every
pair whose write fell in a strictly earlier window than its read, the intra every pair inside one window, and the den
pass each token's denominator against the earlier windows' write mass. Together they are the whole recurrence, no pair
counted twice and none dropped (`tests/oracle/test_carry_vs_oracle.py`'s decomposition test states it in fp64).

THE UNITS. Every kernel leaves each token's READER FACTOR ``r`` (its read levels' scales) out of its sums, and the
readout divides by it (:func:`readout`): ``y = num / (den + eps / r)`` is the defined readout exactly. The intra's
``num`` and ``den`` are already in those units; the den pass's take its rows' storage factor (:func:`den_factor`),
and the carry's ``num`` its state's unit and its read rows' (:func:`carry_factor`) times the f16 sum's scale ``2^(s +
1)``, ``s`` each token's scale byte from the den pass's bound (:func:`scale_bytes`) -- every CTA's partials meet at it.

Symbols, each at first use: ``W`` the window, ``L`` tokens, ``BH`` batch x heads, ``d_v`` value channels, level ``S``
the carry's (`rola.routing.factors.carry_level_s`) and ``R`` the other.
"""
from __future__ import annotations

import torch

from rola.ops import carry as carry_ops
from rola.ops import den as den_ops
from rola.ops import intra as intra_ops
from rola.routing import quant

__all__ = ["UNIT_HEADROOM", "UNIT_LEVELS", "UNIT_NONE", "carry_factor", "den_factor", "prefill", "reader_factor",
           "readout", "scale_bytes", "unpack_num"]

#: THE f16 SUM's SCALE BYTE, the kernel's `unit_byte` rule (`csrc/rola/src/carry/parts.cuh`): the smallest ``s`` with
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


def den_factor(routes) -> float:
    """What the den pass's sums are multiplied by to be the recurrence's over ``r``: level S's read coefficient a byte
    over 256 and level R's write mass on the byte scale."""
    ls = routes.level_s
    lr = 1 - ls
    k = 256.0 / quant.AMP_LEVELS if _u8(routes.amp_read[ls]) else 1.0
    return k / quant.AMP_LEVELS if _u8(routes.amp_write[lr]) else k


def scale_bytes(routes, bound: torch.Tensor) -> torch.Tensor:
    """``[BH, L]`` int8: each token's scale byte from the den pass's ``bound`` (its read product against the mass of
    ``m s``, which bounds every channel of the carry's ``num``), in the carry's partials' units."""
    p = bound * (den_factor(routes) / carry_factor(routes))
    s = torch.ceil(torch.log2(UNIT_HEADROOM * p.clamp_min(1e-30))) - UNIT_LEVELS
    return torch.where(p > 0, s, torch.full_like(s, UNIT_NONE)).clamp(-128, 127).to(torch.int8)


def unpack_num(num: torch.Tensor, channels: int) -> torch.Tensor:
    """``[BH, L, d_v]`` fp32 in channel order: the carry's ``num`` as the drain wrote it -- fp32, or int32 words of f16
    pairs, a box's word ``8 m + g`` holding channels ``(c, c + 2)``, ``c = 8 (g >> 1) + 4 m + (g & 1)`` of its
    ``channels`` (`docs/internals/carry/parts.md#image`)."""
    if num.dtype == torch.float32:
        return num
    BH, L, words = num.shape
    lo = (num & 0xFFFF).to(torch.int16).view(torch.float16).float()
    hi = ((num >> 16) & 0xFFFF).to(torch.int16).view(torch.float16).float()
    k = torch.arange(words, device=num.device)
    box, kk = k // (channels // 2), k % (channels // 2)
    m, g = kk // 8, kk % 8
    channel = channels * box + 8 * (g >> 1) + 4 * m + (g & 1)
    out = torch.empty(BH, L, 2 * words, dtype=torch.float32, device=num.device)
    out[..., channel] = lo
    out[..., channel + 2] = hi
    return out


def prefill(routes, values, facts, carve, layout, read_statics, write_statics, *, accumulate: str | None = None,
            reduce_dtype: torch.dtype | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """``(num, den)`` fp32 ``[BH, L, d_v]`` and ``[BH, L]`` over ``r`` (:func:`readout` divides by it): the den pass,
    the carry at the den pass's scale bytes, and the intra reduced into the same planes. ``facts`` is the facts pass's
    output over this bundle in ``layout`` with the carve's read row (`rola.ops.carry.form`) and these statics.
    ``accumulate`` and ``reduce_dtype`` are reserved: the gates' choice of the MMAs' accumulation and the carry's
    cross-CTA dtype, each the device's or the carve's rule by default."""
    den_inter, bound = den_ops.den(routes, facts)
    scale = scale_bytes(routes, bound)
    num_c, _ = carry_ops.carry(routes, values, facts, carve, scale, accumulate=accumulate, reduce_dtype=reduce_dtype)
    num = unpack_num(num_c, carve.channels)
    unit = carry_factor(routes)
    if num_c.dtype != torch.float32:
        num = num * (torch.exp2(scale.float() + 1.0) * unit).unsqueeze(-1)
    else:
        num = num * unit
    num = num.contiguous()
    den = (den_inter * den_factor(routes)).contiguous()
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
