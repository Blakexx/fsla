# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE VALUE STREAM IN THE INTERNAL FORMAT: a token's values against its largest channel, and the token's scale.

The op's value operand is the same kind of thing its routing operand is -- a record of
tensors in the one format the kernels read (ruled 2026-09-26, docs/api.md section 4.1).
At one digit ``v_hat`` holds each token's values with its largest channel at +-127; at
two (the default) the token's values are the s16 ``q = round(32767 x)`` stored as the
s8 high byte ``v_hat`` and the u8 low byte ``v_lo``. ``scale`` is the fp32 magnitude.
:meth:`Values.dequantized` is the floating view a consumer still computing on floats
takes at its seam. The format is written in one place, :meth:`Values.from_projection`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from rola.routing import quant

__all__ = ["DEFAULT_VALUE_DIGITS", "Values"]

#: the value stream's digit count unless a layer declares one (ruled 2026-09-29).
DEFAULT_VALUE_DIGITS = 2


@dataclass(frozen=True, slots=True)
class Values:
    """The value stream: ``v_hat`` s8 ``[B, T, H, d_v]``; ``v_lo`` its u8 low byte at two digits or ``None`` at
    one; ``scale`` fp32 ``[B, T, H]``; ``source`` the projection's floating values, held ONLY when they require a
    gradient (the straight-through anchor for :meth:`dequantized`), ``None`` otherwise."""

    v_hat: torch.Tensor
    v_lo: torch.Tensor | None
    scale: torch.Tensor
    source: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.v_hat, torch.Tensor) or self.v_hat.dtype is not torch.int8 or self.v_hat.ndim != 4:
            raise ValueError(
                f"Values.v_hat is the s8 [B, T, H, d_v] plane, got "
                f"{getattr(self.v_hat, 'dtype', type(self.v_hat).__name__)} "
                f"{getattr(self.v_hat, 'shape', '')}")
        if self.v_lo is not None and (not isinstance(self.v_lo, torch.Tensor) or self.v_lo.dtype is not torch.uint8
                                      or tuple(self.v_lo.shape) != tuple(self.v_hat.shape)):
            raise ValueError(
                f"Values.v_lo is the u8 low-byte plane, {tuple(self.v_hat.shape)}, got "
                f"{getattr(self.v_lo, 'dtype', type(self.v_lo).__name__)} {getattr(self.v_lo, 'shape', '')}")
        if (not isinstance(self.scale, torch.Tensor) or self.scale.dtype is not torch.float32
                or tuple(self.scale.shape) != tuple(self.v_hat.shape[:3])):
            raise ValueError(
                f"Values.scale is the fp32 [B, T, H]={tuple(self.v_hat.shape[:3])} token scale, got "
                f"{getattr(self.scale, 'dtype', type(self.scale).__name__)} "
                f"{getattr(self.scale, 'shape', '')}")
        if self.source is not None and tuple(self.source.shape) != tuple(self.v_hat.shape):
            raise ValueError(f"Values.source is the planes' floating original, {tuple(self.v_hat.shape)}")
        for name in ("v_lo", "scale", "source"):
            t = getattr(self, name)
            if t is not None and t.device != self.v_hat.device:
                raise ValueError(f"Values.{name} is on {t.device} but v_hat is on {self.v_hat.device}")

    @classmethod
    def from_projection(cls, v: torch.Tensor, digits: int = DEFAULT_VALUE_DIGITS) -> Values:
        """THE FRAMEWORK'S EPILOGUE for values: a floating ``[B, T, H, d_v]`` in, the format out at ``digits``
        (`rola.routing.quant`)."""
        if not isinstance(v, torch.Tensor) or v.ndim != 4 or not v.dtype.is_floating_point:
            raise ValueError(
                f"the value projection is a floating [B, T, H, d_v] tensor, got "
                f"{getattr(v, 'dtype', type(v).__name__)} {getattr(v, 'shape', '')}")
        v_hat, v_lo, scale = quant.quantize_values(v, digits)
        source = v if torch.is_grad_enabled() and v.requires_grad else None
        return cls(v_hat=v_hat, v_lo=v_lo, scale=scale, source=source)

    def dequantized(self) -> torch.Tensor:
        """The values as fp32, computed at the call; with a ``source`` the projection's gradient steps straight
        through the rounding. A consumer still computing on floating values calls this at its seam."""
        v = quant.dequantize_values(self.v_hat, self.v_lo, self.scale)
        return v if self.source is None else quant.straight_through(self.source, v)

    @property
    def digits(self) -> int:
        return 1 if self.v_lo is None else 2

    @property
    def batch(self) -> int:
        return self.v_hat.shape[0]

    @property
    def tokens(self) -> int:
        return self.v_hat.shape[1]

    @property
    def heads(self) -> int:
        return self.v_hat.shape[2]

    @property
    def d_v(self) -> int:
        return self.v_hat.shape[3]

    @property
    def device(self) -> torch.device:
        return self.v_hat.device

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Every tensor the OPERATION consumes, for the gradient-reachability predicate."""
        return tuple(t for t in (self.v_hat, self.v_lo, self.scale, self.source) if t is not None)
