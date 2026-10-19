# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The fp64 reference of the carry's READOUT (docs/internals/carry/parts.md#readout): each token's partials at its
unit, from the fold's declared state before its window (`fold_reference.fold_state`).

THE OUTPUT: ``P(t, c) x unit(t) = sum_{d, r} x(t, d) a(t, r) S(d, r, c) / kappa`` -- ``S`` the absolute state before
``t``'s window (level S's digit ``d``, level R's list ``r``), ``x`` level S's read coefficient as stored (a sparse
level's byte / 256, a dense level's f16), ``a`` level R's read row as stored (f16), ``kappa`` the state's byte scale
(``unit_scale``). The read side's per-token scales and ``kappa`` are common to ``num`` and ``den`` and cancel in the
epilogue. ``unit(t) = 2^u(t)``, ``u`` the token's unit byte (the den pass's; here :func:`units` from the same sum's
bound).

THE BOUND, a slot's distance from the kernel in units (:func:`readout`): each add rounds once (half a unit; a token
takes one add a live digit of level S a box along level R), the image holds 11 bits a value and its subnormals 2^-25 at
the digit's power of two, the fold's state is within ``(L / 16) 2^-23`` of its envelope, and the MMA accumulates in
fp32 or (GeForce) fp16.

Symbols: ``W`` the window, ``L`` tokens, ``B_l`` level ``l``'s padded width, ``C`` the value channels.
"""
from __future__ import annotations

import numpy as np
import torch

from rola.routing import quant
from tests.oracle.fold_reference import fold_state, unit_scale

#: THE IMAGE'S TOP (`parts.cuh` `kImageTop`) and c's (`fold.cuh` `kCoefTop`)
IMAGE_TOP = 10
COEF_TOP = 4
#: a unit holds a final under 2^15 with 2% headroom; a token with nothing to read takes the least unit byte
UNIT_LEVELS = 15
UNIT_HEADROOM = 1.02
UNIT_NONE = -100
#: the boxes along level R: 32 lists a CTA
BOX_LISTS = 32
#: the MMA's accumulation, relative to the sum of its products' magnitudes
ACCUMULATE = {"fp32": 2.0 ** -18, "fp16": 2.0 ** -9}


def _read_rows(routes, level_s: int):
    """``(x, a)``: level S's coefficients ``[B, T, H, B_S]`` and level R's rows ``[B, T, H, B_R]``, fp64, as stored."""
    level_r = 1 - level_s
    xs = routes.amp_read[level_s]
    x = xs.double() if xs.dtype == torch.float16 else (xs.double() / 256).half().double()
    return x, routes.amp_read[level_r].double()


def _windows(routes, values, list_exp, level_s: int, window: int):
    """Each window's absolute state over kappa and the fold's envelope, ``[B, H, B_S, B_R, C]`` fp64 each, before its
    writes."""
    kappa = unit_scale(routes, level_s)
    out = []
    fold_state(routes, values, list_exp, level_s, window,
               at_window=lambda w, state, env: out.append((state / kappa, env / kappa)))
    return out


def units(routes, values, list_exp, level_s: int, window: int) -> torch.Tensor:
    """``[BH, T]`` int8: each token's unit byte from the bound ``num(t) = sum_{d, r} x a max_c |S| / kappa``,
    ``ceil(log2(1.02 num)) - 15`` (``UNIT_NONE`` where ``num`` is 0)."""
    x, a = _read_rows(routes, level_s)
    B, T, H, _ = x.shape
    num = torch.zeros(B, T, H, dtype=torch.float64, device=x.device)
    for w, (state, _env) in enumerate(_windows(routes, values, list_exp, level_s, window)):
        sl = slice(w * window, (w + 1) * window)
        top = state.abs().amax(-1)                                                          # [B, H, B_S, B_R]
        num[:, sl] = torch.einsum("bthd,bthr,bhdr->bth", x[:, sl], a[:, sl], top)
    u = torch.where(num > 0, torch.ceil(torch.log2(UNIT_HEADROOM * num.clamp_min(1e-300))) - UNIT_LEVELS,
                    torch.full_like(num, UNIT_NONE))
    return u.clamp(-128, 127).to(torch.int8).permute(0, 2, 1).reshape(B * H, T).contiguous()


def readout(routes, values, list_exp, level_s: int, window: int, unit: torch.Tensor, accumulate: str):
    """``(P, bound)``: ``[BH, T, C]`` fp64 each token's partials at its unit (``unit`` ``[BH, T]`` the unit bytes) and
    each slot's bound in units, for an MMA accumulating in ``accumulate`` (``"fp32"`` or ``"fp16"``)."""
    x, a = _read_rows(routes, level_s)
    B, T, H, _ = x.shape
    B_R = a.shape[-1]
    L = T
    scale = torch.exp2(-unit.double()).reshape(B, H, T).permute(0, 2, 1)                   # [B, T, H]
    C = values.v.shape[-1]
    P = torch.zeros(B, T, H, C, dtype=torch.float64, device=x.device)
    bound = torch.zeros_like(P)
    live = (x != 0).double()
    adds = live.sum(-1) * (B_R // BOX_LISTS)                                                 # [B, T, H]
    for w, (state, env) in enumerate(_windows(routes, values, list_exp, level_s, window)):
        sl = slice(w * window, (w + 1) * window)
        xs, as_ = x[:, sl], a[:, sl]
        P[:, sl] = torch.einsum("bthd,bthr,bhdrc->bthc", xs, as_, state)
        mag = torch.einsum("bthd,bthr,bhdrc->bthc", xs.abs(), as_.abs(), state.abs())
        fold = torch.einsum("bthd,bthr,bhdrc->bthc", xs.abs(), as_.abs(), env) * (L / 16 * 2.0 ** -23)
        #: the subnormal floor at each digit's power of two: its largest |S| over every list and channel, doubled
        top = state.abs().amax(dim=(-1, -2))                                                 # [B, H, B_S]
        floor = torch.einsum("bthd,bthr,bhd->bth", xs.abs(), as_.abs(), 2 * top) * 2.0 ** (-25 - IMAGE_TOP)
        bound[:, sl] = (2.0 ** -11 + ACCUMULATE[accumulate]) * mag + fold + floor[..., None]
    P = P * scale[..., None]
    bound = bound * scale[..., None] + 0.5 * adds[..., None]
    return P.permute(0, 2, 1, 3).reshape(B * H, T, C), bound.permute(0, 2, 1, 3).reshape(B * H, T, C)


def unpack_num(num: torch.Tensor, channels: int = 32) -> torch.Tensor:
    """``[BH, T, C]`` int64: the carry's ``num`` (``[BH, T, C / 2]`` int32, ``hi x 2^16 + lo``) as each channel's
    integer. A box's word ``8 m + g`` pairs channels ``(c, c + 2)``, ``c = 8 (g >> 1) + 4 m + (g & 1)``: the image's
    16-byte unit holds a writer lane's channels (``docs/internals/carry/parts.md#image``)."""
    BH, T, words = num.shape
    v = num.long()
    lo = ((v & 0xFFFF) ^ 0x8000) - 0x8000
    hi = (v - lo) >> 16
    k = np.arange(words)
    box, kk = k // (channels // 2), k % (channels // 2)
    m, g = kk // 8, kk % 8
    channel = channels * box + 8 * (g >> 1) + 4 * m + (g & 1)
    out = torch.zeros(BH, T, 2 * words, dtype=torch.int64, device=num.device)
    out[..., torch.as_tensor(channel, device=num.device)] = lo
    out[..., torch.as_tensor(channel + 2, device=num.device)] = hi
    return out


def truth_scale(routes, unit: torch.Tensor, level_s: int) -> torch.Tensor:
    """``[B, T, H]`` fp64: what a token's partials at its unit are multiplied by to be the recurrence's numerator, the
    per-token factor the epilogue cancels against ``den``: ``2^u(t)``, ``kappa`` and each read level's row scale, a u8
    level's stored value scaled to its digit over 255 (level S's coefficient is the byte over 256, level R's row the
    byte), an f16 level's the value itself -- a read side with no list level's two levels both (level S its
    coefficient level, level R its row level: `carry_level_s`)."""
    if level_s < 0:
        raise NotImplementedError("the carry's level S is defined at depth two (`carry_level_s`)")
    level_r = 1 - level_s
    stored = {level_s: quant.AMP_LEVELS + 1, level_r: 1.0}
    factor = unit_scale(routes, level_s)
    for level in (level_s, level_r):
        rows = routes.amp_read[level]
        k = 1.0 if rows.dtype == torch.float16 else stored[level] / quant.AMP_LEVELS
        factor = factor * k * routes.scale_read[level].double()
    B, T, H = factor.shape
    return factor * torch.exp2(unit[:, :T].double().reshape(B, H, T).permute(0, 2, 1))


def num_in_truth(num: torch.Tensor, routes, unit: torch.Tensor) -> torch.Tensor:
    """``[B, T, H, C]`` fp64: the carry's ``num`` (:func:`unpack_num`) times :func:`truth_scale` -- the inter term's
    numerator, comparable with `tests.oracle.reference.inter_reference`'s."""
    B, T, H = routes.batch, routes.tokens, routes.heads
    got = unpack_num(num[:, :T]).double().reshape(B, H, T, -1).permute(0, 2, 1, 3)
    return got * truth_scale(routes, unit, routes.level_s)[..., None]
