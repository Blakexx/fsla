# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE VALUE STREAM IN THE INTERNAL FORMAT: a token's values as s8 with the token's scale.

The op's value operand is the same kind of thing its routing operand is -- a record of
tensors in the one format the kernels read (ruled 2026-09-26, docs/api.md section 4.1):
``v_hat`` holds each token's values with its largest channel at +-127 and ``scale`` is
that magnitude, so ``v = v_hat / 127 * scale`` per token. ``v`` is that dequantized value
with the projection's gradient stepped straight through the rounding; it is what every
consumer still computing on floating values reads, and it is deleted with the last
kernel port. The format is written in one place, :meth:`Values.from_projection`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from rola.routing import quant

__all__ = ["Values"]


@dataclass(frozen=True, slots=True)
class Values:
    """The value stream: ``v_hat`` s8 ``[B, T, H, d_v]``, ``scale`` ``[B, T, H]`` in the projection's floating
    dtype, and ``v`` the dequantized straight-through view, ``[B, T, H, d_v]`` in that dtype."""

    v_hat: torch.Tensor
    scale: torch.Tensor
    v: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.v_hat, torch.Tensor) or self.v_hat.dtype is not torch.int8 or self.v_hat.ndim != 4:
            raise ValueError(
                f"Values.v_hat is the s8 [B, T, H, d_v] plane, got "
                f"{getattr(self.v_hat, 'dtype', type(self.v_hat).__name__)} "
                f"{getattr(self.v_hat, 'shape', '')}")
        if (not isinstance(self.scale, torch.Tensor) or not self.scale.dtype.is_floating_point
                or tuple(self.scale.shape) != tuple(self.v_hat.shape[:3])):
            raise ValueError(
                f"Values.scale is the floating [B, T, H]={tuple(self.v_hat.shape[:3])} token scale, got "
                f"{getattr(self.scale, 'dtype', type(self.scale).__name__)} "
                f"{getattr(self.scale, 'shape', '')}")
        if (not isinstance(self.v, torch.Tensor) or tuple(self.v.shape) != tuple(self.v_hat.shape)
                or self.v.dtype is not self.scale.dtype):
            raise ValueError(
                f"Values.v is the dequantized view, {tuple(self.v_hat.shape)} in {self.scale.dtype}, got "
                f"{getattr(self.v, 'dtype', type(self.v).__name__)} {getattr(self.v, 'shape', '')}")
        for name, t in (("scale", self.scale), ("v", self.v)):
            if t.device != self.v_hat.device:
                raise ValueError(f"Values.{name} is on {t.device} but v_hat is on {self.v_hat.device}")

    @classmethod
    def from_projection(cls, v: torch.Tensor) -> Values:
        """THE FRAMEWORK'S EPILOGUE for values: a floating ``[B, T, H, d_v]`` in, the format out, with the
        input's gradient stepped straight through the rounding (`rola.routing.quant`)."""
        if not isinstance(v, torch.Tensor) or v.ndim != 4 or not v.dtype.is_floating_point:
            raise ValueError(
                f"the value projection is a floating [B, T, H, d_v] tensor, got "
                f"{getattr(v, 'dtype', type(v).__name__)} {getattr(v, 'shape', '')}")
        v_hat, scale = quant.quantize_values(v)
        return cls(v_hat=v_hat, scale=scale, v=quant.straight_through(v, quant.dequantize_values(v_hat, scale)))

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
