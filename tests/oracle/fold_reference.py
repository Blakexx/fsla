# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The fp64 reference of the fp16 fold's DECLARED arithmetic, beginning with its scales.

THE LIST EXPONENTS (the facts pass writes them; docs/internals/facts/facts.md#lists): per window ``w`` and digit ``r``
of the list level, ``K[w, r, col] = ceil(log2(max over t in w of b(t, r) x col(t)))``, floored at -126 -- ``b`` the
token's write byte at ``r`` (a dense level's zero, in a live row, at ``255 x DENSE_FLOOR``), ``col`` either ``m(t)``
(the write gain times every write level's scale, column 0) or ``m(t) s(t)`` (times the value scale, column 1). The fold
puts each list's largest coefficient at a fixed power of two by them.

Symbols: ``W`` the window, ``L`` tokens, ``B_l`` level ``l``'s padded width.
"""
from __future__ import annotations

import torch

from rola.routing import quant

#: the exponents' floor: the smallest normal fp32's
EXPONENT_FLOOR = -126


def list_columns(routes, values) -> torch.Tensor:
    """``[B, T, H, 2]`` fp64: each token's ``m`` and ``m s``."""
    m = routes.g_write.double()
    for scale in routes.scale_write:
        m = m * scale.double()
    return torch.stack((m, m * values.scale.double()), -1)


def list_maxima(routes, values, level: int, window: int) -> torch.Tensor:
    """``[B, H, windows, B_level, 2]`` fp64: per window and digit of ``level``, the largest ``b x col`` over its
    tokens, ``b`` the write byte (a dense level's zero in a live row at the floor), zero past the rows' width."""
    rows = routes.amp_write[level].double()
    if not routes.sparse[1][level]:
        live = (routes.scale_write[level] > 0).unsqueeze(-1)
        rows = torch.where(live, rows.clamp_min(quant.AMP_LEVELS * quant.DENSE_FLOOR), rows)
    B, T, H, width = rows.shape
    cols = list_columns(routes, values)
    windows = -(-T // window)
    pad = windows * window - T
    prod = rows.unsqueeze(-1) * cols.unsqueeze(-2)                                     # [B, T, H, width, 2]
    prod = torch.cat((prod, prod.new_zeros(B, pad, H, width, 2)), 1)
    return prod.reshape(B, windows, window, H, width, 2).amax(2).permute(0, 2, 1, 3, 4)


def _exponents(top: torch.Tensor) -> torch.Tensor:
    return torch.ceil(torch.log2(top.clamp_min(2.0 ** EXPONENT_FLOOR))).to(torch.int64)


def _padded_maxima(routes, values, level: int, window: int, padded_width: int) -> torch.Tensor:
    top = list_maxima(routes, values, level, window)
    return torch.nn.functional.pad(top, (0, 0, 0, padded_width - top.shape[-2]))


def list_exponents(routes, values, level: int, window: int, padded_width: int) -> torch.Tensor:
    """``[B, H, windows, padded_width, 2]`` int: :func:`list_maxima`'s ceil-log2, floored at -126."""
    return _exponents(_padded_maxima(routes, values, level, window, padded_width))


def exponent_bounds(routes, values, level: int, window: int, padded_width: int, rtol: float):
    """``(lo, hi)``: the exponents of the maxima moved by ``rtol`` either way -- a kernel computing the products in
    fp32 lands in between, the exponent changing only where a maximum sits within ``rtol`` of a power of two."""
    top = _padded_maxima(routes, values, level, window, padded_width)
    return _exponents(top * (1 - rtol)), _exponents(top * (1 + rtol))
