# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE VALUE STREAM IN THE INTERNAL FORMAT: a token's values against its largest channel, and the token's scale.

The op's value operand is the same kind of thing its routing operand is -- a record of
tensors in the one format the kernels read (ruled 2026-09-26, docs/api.md section 4.1).
``v`` holds each token's values in f16 with its largest channel at 1; ``scale`` is that
largest magnitude in fp32. :meth:`Values.dequantized` is the floating view a consumer
still computing on floats takes at its seam. The format is written in one place,
:meth:`Values.from_projection`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from rola.routing import quant

__all__ = ["Values"]


@dataclass(frozen=True, slots=True)
class Values:
    """The value stream: ``v`` f16 ``[B, T, H, d_v]`` against each token's largest channel; ``scale`` fp32
    ``[B, T, H]``; ``source`` the projection's floating values, held ONLY when they require a gradient (the
    straight-through anchor for :meth:`dequantized`), ``None`` otherwise."""

    v: torch.Tensor
    scale: torch.Tensor
    source: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.v, torch.Tensor) or self.v.dtype is not quant.VALUE_DTYPE or self.v.ndim != 4:
            raise ValueError(
                f"Values.v is the f16 [B, T, H, d_v] plane, got "
                f"{getattr(self.v, 'dtype', type(self.v).__name__)} {getattr(self.v, 'shape', '')}")
        if (not isinstance(self.scale, torch.Tensor) or self.scale.dtype is not torch.float32
                or tuple(self.scale.shape) != tuple(self.v.shape[:3])):
            raise ValueError(
                f"Values.scale is the fp32 [B, T, H]={tuple(self.v.shape[:3])} token scale, got "
                f"{getattr(self.scale, 'dtype', type(self.scale).__name__)} "
                f"{getattr(self.scale, 'shape', '')}")
        if self.source is not None and tuple(self.source.shape) != tuple(self.v.shape):
            raise ValueError(f"Values.source is the plane's floating original, {tuple(self.v.shape)}")
        for name in ("scale", "source"):
            t = getattr(self, name)
            if t is not None and t.device != self.v.device:
                raise ValueError(f"Values.{name} is on {t.device} but v is on {self.v.device}")

    @classmethod
    def from_projection(cls, v: torch.Tensor) -> Values:
        """THE FRAMEWORK'S EPILOGUE for values: a floating ``[B, T, H, d_v]`` in, the format out
        (`rola.routing.quant`)."""
        if not isinstance(v, torch.Tensor) or v.ndim != 4 or not v.dtype.is_floating_point:
            raise ValueError(
                f"the value projection is a floating [B, T, H, d_v] tensor, got "
                f"{getattr(v, 'dtype', type(v).__name__)} {getattr(v, 'shape', '')}")
        unit, scale = quant.quantize_values(v)
        source = v if torch.is_grad_enabled() and v.requires_grad else None
        return cls(v=unit, scale=scale, source=source)

    def dequantized(self) -> torch.Tensor:
        """The values as fp32, computed at the call; with a ``source`` the projection's gradient steps straight
        through the rounding. A consumer still computing on floating values calls this at its seam."""
        v = quant.dequantize_values(self.v, self.scale)
        return v if self.source is None else quant.straight_through(self.source, v)

    @property
    def batch(self) -> int:
        return self.v.shape[0]

    @property
    def tokens(self) -> int:
        return self.v.shape[1]

    @property
    def heads(self) -> int:
        return self.v.shape[2]

    @property
    def d_v(self) -> int:
        return self.v.shape[3]

    @property
    def device(self) -> torch.device:
        return self.v.device

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Every tensor the OPERATION consumes, for the gradient-reachability predicate."""
        return tuple(t for t in (self.v, self.scale, self.source) if t is not None)
