# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE INTERNAL FORMAT'S QUANTIZERS: unit-max coefficient rows, per-token values, and the straight-through view.

One declaration of the arithmetic the op's oracle takes as its input
(``tests/oracle/int_reference.py`` is its fp64 statement). A coefficient row is stored against its own max -- u8 with
the max digit at 255, except a DENSE WRITE level's, f16 at 1 -- and the row's max is its fp32 scale. A dense level is
never zero: a dense write row is stored with ``DENSE_FLOOR`` of the row max already taken, and a dense read level's u8
zero digit, in a live row, reads as that floor (docs/internals/facts/liveness_contract.md#floor). A token's values
are stored as f16 against their largest channel (at 1), the largest its fp32 scale. ``straight_through`` hands a
consumer the DEQUANTIZED values with an identity gradient, which is how the format trains (docs/api.md
section 2.1).
"""
from __future__ import annotations

import torch

__all__ = [
    "AMP_LEVELS",
    "DENSE_DTYPE",
    "DENSE_FLOOR",
    "ROW_ALIGN",
    "ROW_DTYPE",
    "VALUE_DTYPE",
    "aligned_rows",
    "dequantize_rows",
    "dequantize_values",
    "quantize_unit_max_rows",
    "quantize_values",
    "row_slot",
    "straight_through",
]

#: u8 coefficient levels: the row max at 255.
AMP_LEVELS = 255.0
#: the stored coefficient form: u8 at every depth.
ROW_DTYPE = torch.uint8
#: a dense write level's rows: f16 against the row max, the floor taken.
DENSE_DTYPE = torch.float16
#: a dense level's least coefficient, in a live row, as a fraction of the row max: the numerically safe minimum (f16's
#: least normal is 2^-14), inside a u8 zero digit's own [0, 1/510).
DENSE_FLOOR = 2.0 ** -13
#: THE ROW ALIGNMENT LAW: every level's rows start on, and step batch, token and head by, multiples of this many bytes
#: (docs/internals/facts/liveness_contract.md#alignment).
ROW_ALIGN = 32
#: the stored value form: f16 against the token's largest channel.
VALUE_DTYPE = torch.float16
_TINY = 1e-30


def row_slot(width: int) -> int:
    """A level's bytes in a packed row: its width in bytes rounded up to the alignment law's grain."""
    return -(-width // ROW_ALIGN) * ROW_ALIGN


def aligned_rows(B: int, T: int, H: int, width: int, device, dtype: torch.dtype = ROW_DTYPE) -> torch.Tensor:
    """``[B, T, H, width]`` zeros of ``dtype`` under the alignment law: a view of storage whose rows are
    ``row_slot`` of their bytes."""
    size = torch.empty((), dtype=dtype).element_size()
    return torch.zeros((B, T, H, row_slot(width * size) // size), dtype=dtype, device=device)[..., :width]


def quantize_unit_max_rows(level: torch.Tensor, dtype: torch.dtype = ROW_DTYPE,
                           dense: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """``(rows [..., w], scale fp32 [...])``: each row against its own max, in storage under the alignment law -- u8
    digits at 255, or (``DENSE_DTYPE``) f16 at 1 with a ``dense`` level's floor taken in a live row; an all-zero row
    stays zero."""
    exact = level.detach().float()
    scale = exact.abs().amax(dim=-1)
    unit = exact / scale.clamp_min(_TINY).unsqueeze(-1)
    B, T, H, width = exact.shape
    rows = aligned_rows(B, T, H, width, exact.device, dtype)
    if dtype is DENSE_DTYPE:
        if dense:
            unit = torch.where((scale > 0).unsqueeze(-1), unit.clamp_min(DENSE_FLOOR), unit)
        rows.copy_(unit.to(DENSE_DTYPE))
    else:
        rows.copy_(torch.round(unit * AMP_LEVELS).to(ROW_DTYPE))
    return rows, scale


def dequantize_rows(rows: torch.Tensor, scale: torch.Tensor, dense: bool) -> torch.Tensor:
    """The rows' amplitudes, fp32: each u8 digit over 255 (or each f16 coefficient) times its row's scale, a dense
    level at least the floor (a dead row, its scale 0, stays zero)."""
    unit = rows.float() if rows.dtype is DENSE_DTYPE else rows.float() / AMP_LEVELS
    if dense:
        unit = unit.clamp_min(DENSE_FLOOR)
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
