# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE INTERNAL FORMAT'S QUANTIZERS: unit-max rows, per-token values, and the straight-through view.

One declaration of the arithmetic the op's oracle takes as its input
(``tests/oracle/int_reference.py`` is its fp64 statement): an amplitude row is stored u8 with its
max digit at 255 and the row's max as its scale; a token's values are stored s8 with the largest
channel at +-127 and that magnitude as the scale. ``straight_through`` hands a consumer the
DEQUANTIZED values with an identity gradient, which is how the format trains
(docs/api.md section 2.1).
"""
from __future__ import annotations

import torch

__all__ = [
    "AMP_LEVELS",
    "VALUE_LEVELS",
    "dequantize_rows",
    "dequantize_values",
    "quantize_unit_max_rows",
    "quantize_values",
    "straight_through",
]

#: u8 amplitude levels (the row max at 255) and s8 value levels (the largest channel at 127).
AMP_LEVELS = 255.0
VALUE_LEVELS = 127.0
_TINY = 1e-30


def quantize_unit_max_rows(level: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(rows u8 [..., w], scale [...])``: each row against its own max; an all-zero row stays zero."""
    scale = level.detach().abs().amax(dim=-1)
    rows = torch.round(level.detach() / scale.clamp_min(_TINY).unsqueeze(-1) * AMP_LEVELS).to(torch.uint8)
    return rows, scale


def dequantize_rows(rows: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return rows.to(scale.dtype) * (scale / AMP_LEVELS).unsqueeze(-1)


def quantize_values(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(v_hat s8 [..., c], scale [...])``: the token's largest channel at +-127."""
    scale = v.detach().abs().amax(dim=-1)
    v_hat = torch.round(v.detach() / scale.clamp_min(_TINY).unsqueeze(-1) * VALUE_LEVELS).to(torch.int8)
    return v_hat, scale


def dequantize_values(v_hat: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return v_hat.to(scale.dtype) * (scale / VALUE_LEVELS).unsqueeze(-1)


def straight_through(exact: torch.Tensor, dequantized: torch.Tensor) -> torch.Tensor:
    """The quantized VALUES with ``exact``'s gradient: the rounding is stepped through as identity."""
    return exact + (dequantized.to(exact.dtype) - exact).detach()
