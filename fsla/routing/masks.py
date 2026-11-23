# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE MASKS: the op's DATA operand beside the rows, like attention's mask -- a bit per side, per level, per digit, per
token, true where the digit is live. The op computes with ``r * m`` (the rows times their bits); a bit is data and is
never tested against the rows (docs/api.md section 2.6).

Two layouts, each the CONSUMER's native one, chosen by the producer called
(:func:`fsla.ops.produce.produce_prefill`, :func:`fsla.ops.produce.produce_decode`), with no conversion between the producer and the op:

* ``PREFILL``: token-packed, the entmax kernels' ``support_words`` as they are. A level's mask is int32
  ``[B, H, ceil(T / 32), width_l]`` (``H`` the side's heads: ``H_w`` on the write side): bit ``t % 32`` of word
  ``t // 32`` at digit ``d`` = token ``t`` live at digit ``d``. The solve's stream axis is ``B * H`` with ``H`` inner,
  so the words' ``[B * H, ...]`` is this shape as a free view. Bits past ``T`` are zero.
* ``DECODE``: digit-packed, one token. A level's mask is int32 ``[B, H, ceil(width_l / 32)]``: bit ``d % 32`` of word
  ``d // 32`` = digit ``d`` live. Bits past ``width_l`` are zero.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["DECODE", "PREFILL", "Masks", "pack_digits", "pack_tokens", "unpack_digits", "unpack_tokens"]

PREFILL, DECODE = "prefill", "decode"
WORD_BITS = 32
WORD_DTYPE = torch.int32


def _words(bits: torch.Tensor) -> torch.Tensor:
    """``[..., 32]`` bool -> ``[...]`` int32, lane ``i`` at bit ``i``."""
    lanes = bits.to(torch.int64) * (1 << torch.arange(WORD_BITS, device=bits.device, dtype=torch.int64))
    words = lanes.sum(dim=-1)
    return (words - (words >= 1 << 31).to(torch.int64) * (1 << 32)).to(WORD_DTYPE)


def _lanes(words: torch.Tensor) -> torch.Tensor:
    """``[...]`` int32 -> ``[..., 32]`` bool."""
    shifts = torch.arange(WORD_BITS, device=words.device, dtype=torch.int32)
    return ((words.unsqueeze(-1) >> shifts) & 1).bool()


def pack_tokens(live: torch.Tensor) -> torch.Tensor:
    """The PREFILL layout of one level: bool ``[B, T, H, width]`` -> int32 ``[B, H, ceil(T / 32), width]``."""
    B, T, H, W = live.shape
    pad = -T % WORD_BITS
    if pad:
        live = torch.cat([live, live.new_zeros((B, pad, H, W))], dim=1)
    lanes = live.permute(0, 2, 3, 1).reshape(B, H, W, -1, WORD_BITS)
    return _words(lanes).permute(0, 1, 3, 2).contiguous()


def unpack_tokens(words: torch.Tensor, tokens: int) -> torch.Tensor:
    """The inverse of :func:`pack_tokens`: int32 ``[B, H, ceil(T / 32), width]`` -> bool ``[B, T, H, width]``."""
    B, H, _, W = words.shape
    bits = _lanes(words.permute(0, 1, 3, 2)).reshape(B, H, W, -1)[..., :tokens]
    return bits.permute(0, 3, 1, 2)


def pack_digits(live: torch.Tensor) -> torch.Tensor:
    """The DECODE layout of one level: bool ``[B, 1, H, width]`` -> int32 ``[B, H, ceil(width / 32)]``."""
    B, T, H, W = live.shape
    if T != 1:
        raise ValueError(f"the decode layout holds one token, got T={T}")
    pad = -W % WORD_BITS
    bits = live[:, 0]
    if pad:
        bits = torch.cat([bits, bits.new_zeros((B, H, pad))], dim=-1)
    return _words(bits.reshape(B, H, -1, WORD_BITS))


def unpack_digits(words: torch.Tensor, width: int) -> torch.Tensor:
    """The inverse of :func:`pack_digits`: int32 ``[B, H, ceil(width / 32)]`` -> bool ``[B, 1, H, width]``."""
    B, H, _ = words.shape
    return _lanes(words).reshape(B, H, -1)[..., :width].unsqueeze(1)


@dataclass(frozen=True, slots=True)
class Masks:
    """One bit per side, level, digit and token, in the layout of the consumer named by ``consumer``: ``read`` and
    ``write`` are each one int32 tensor a level (:data:`PREFILL`, :data:`DECODE` above). Tensors and a static name,
    nothing else. They are DATA: validated for shape, dtype and device (:meth:`check`), never for their values."""

    consumer: str
    read: tuple[torch.Tensor, ...]
    write: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if self.consumer not in (PREFILL, DECODE):
            raise ValueError(f"Masks.consumer is {PREFILL!r} or {DECODE!r}, got {self.consumer!r}")
        object.__setattr__(self, "read", tuple(self.read))
        object.__setattr__(self, "write", tuple(self.write))

    def level(self, side: str, level: int) -> torch.Tensor:
        return getattr(self, side)[level]

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (*self.read, *self.write)

    def expected_shape(self, routes, side: str, level: int) -> tuple[int, ...]:
        """The shape ``side``'s level mask has against ``routes``' rows."""
        rows = routes.rows(level, side)
        B, T, H, W = rows.shape
        if self.consumer == PREFILL:
            return (B, H, -(-T // WORD_BITS), W)
        return (B, H, -(-W // WORD_BITS))

    def check(self, routes) -> None:
        """Shape, dtype and device against the bundle's rows, and nothing about the values."""
        D = routes.topology.D
        for side in ("read", "write"):
            masks = getattr(self, side)
            if len(masks) != D:
                raise ValueError(f"Masks.{side} holds one tensor per level (D={D}), got {len(masks)}")
            for level, mask in enumerate(masks):
                want = self.expected_shape(routes, side, level)
                if not isinstance(mask, torch.Tensor) or mask.dtype is not WORD_DTYPE or tuple(mask.shape) != want:
                    raise ValueError(
                        f"Masks.{side}[{level}] ({self.consumer} layout) is int32 {want}; got "
                        f"{getattr(mask, 'dtype', type(mask).__name__)} {tuple(getattr(mask, 'shape', ()))}")
                if mask.device != routes.device:
                    raise ValueError(f"Masks.{side}[{level}] is on {mask.device}, the rows on {routes.device}")

    def bits(self, side: str, level: int, tokens: int, width: int) -> torch.Tensor:
        """``side``'s level as bool ``[B, T, H, width]``: the bits unpacked, for the references."""
        mask = self.level(side, level)
        return unpack_tokens(mask, tokens) if self.consumer == PREFILL else unpack_digits(mask, width)
