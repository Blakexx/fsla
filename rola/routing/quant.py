# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE INTERNAL FORMAT'S QUANTIZERS: unit-max coefficient rows, per-token values, and the straight-through view.

One declaration of the arithmetic the op's oracle takes as its input
(``tests/oracle/int_reference.py`` is its fp64 statement). A coefficient row is stored against its own max --
u8 with the max digit at 255, or f16 with the max at 1 -- and the row's max is its fp32 scale. A token's values
are stored as f16 against their largest channel (at 1), the largest its fp32 scale. ``straight_through`` hands a
consumer the DEQUANTIZED values with an identity gradient, which is how the format trains (docs/api.md
section 2.1).
"""
from __future__ import annotations

import torch

__all__ = [
    "AMP_LEVELS",
    "ROW_DTYPES",
    "VALUE_DTYPE",
    "dequantize_rows",
    "dequantize_values",
    "quantize_unit_max_rows",
    "quantize_values",
    "straight_through",
]

#: u8 coefficient levels: the row max at 255.
AMP_LEVELS = 255.0
#: the stored coefficient forms: u8 (depth <= 2) and f16 (any depth).
ROW_DTYPES = (torch.uint8, torch.float16)
#: the stored value form: f16 against the token's largest channel.
VALUE_DTYPE = torch.float16
_TINY = 1e-30


def quantize_unit_max_rows(level: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """``(rows [..., w], scale fp32 [...])``: each row against its own max; an all-zero row stays zero."""
    exact = level.detach().float()
    scale = exact.abs().amax(dim=-1)
    unit = exact / scale.clamp_min(_TINY).unsqueeze(-1)
    if dtype is torch.uint8:
        return torch.round(unit * AMP_LEVELS).to(torch.uint8), scale
    if dtype is torch.float16:
        return unit.to(torch.float16), scale
    raise ValueError(f"a coefficient row is stored as one of {ROW_DTYPES}, got {dtype}")


def dequantize_rows(rows: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    unit = rows.float() / AMP_LEVELS if rows.dtype is torch.uint8 else rows.float()
    return unit * scale.unsqueeze(-1)


def quantize_values(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(unit f16 [..., c], scale fp32 [...])``: the token's values against its largest channel, rounded once to
    f16; an all-zero token stays zero."""
    exact = v.detach().float()
    scale = exact.abs().amax(dim=-1)
    return (exact / scale.clamp_min(_TINY).unsqueeze(-1)).to(VALUE_DTYPE), scale


def dequantize_values(unit: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return unit.float() * scale.unsqueeze(-1)


def straight_through(exact: torch.Tensor, dequantized: torch.Tensor) -> torch.Tensor:
    """The quantized VALUES, fp32, with ``exact``'s gradient: the rounding is stepped through as identity."""
    wide = exact.float()
    return wide + (dequantized.float() - wide).detach()
