# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE MASKS: the op's DATA operand beside the rows, like attention's mask -- a bit per side, per level, per digit, per
token. For the PREFILL a bit of 0 is the caller's GUARANTEE that the entry's row is zero and a bit of 1 guarantees nothing; for
DECODE the op computes with ``r * m`` (the rows times their bits). A bit is data and is never tested against the rows
(docs/api.md section 2.6). The prefill's DENSITY, ``[2, D]``, is its own input beside the masks: :func:`masks_and_density` builds both from a
bundle's rows and :func:`density_of_masks` the density of masks a caller already has.

THE DENSE FLAG IS THE DENSITY BYTE 255: a level-side at 255 is computed with every digit live and its mask is never read, so in the
PREFILL layout its entry in a bundle may be ABSENT, ``None``. An absent entry's byte is 255: its supplier's duty, which the call does not
check. The library's producer and helpers leave a mask absent, and state the 255, exactly where the level-side's declared activation
cannot make exact zeros (:func:`fsla.routing.types.activation_walks`: softmax), from the declaration and never from the data; a
level-side that can make zeros (entmax, an opaque level) always gets its mask, however dense its data turns out.
DECODE has no absent entry: its masks are made for every level-side.

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
from typing import TYPE_CHECKING

import torch

from fsla.routing.types import activation_walks

if TYPE_CHECKING:
    from fsla.routing.factors import RouteFactors

__all__ = ["DECODE", "PREFILL", "Masks", "density_of_masks", "masks_and_density", "pack_digits", "pack_tokens",
           "unpack_digits", "unpack_tokens"]

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
    ``write`` are each one int32 tensor a level (:data:`PREFILL`, :data:`DECODE` above), or in the PREFILL layout ``None``: ABSENT
    MEANS DENSE, every digit of every token live on that level-side. Tensors and a static name, nothing else. They are DATA:
    validated for shape, dtype and device (:meth:`check`), never for their values."""

    consumer: str
    read: tuple[torch.Tensor | None, ...]
    write: tuple[torch.Tensor | None, ...]

    def __post_init__(self) -> None:
        if self.consumer not in (PREFILL, DECODE):
            raise ValueError(f"Masks.consumer is {PREFILL!r} or {DECODE!r}, got {self.consumer!r}")
        object.__setattr__(self, "read", tuple(self.read))
        object.__setattr__(self, "write", tuple(self.write))
        if self.consumer == DECODE:
            for side in ("read", "write"):
                for level, mask in enumerate(getattr(self, side)):
                    if mask is None:
                        raise ValueError(f"Masks.{side}[{level}] is absent in a {DECODE!r}-layout bundle: only the {PREFILL!r} layout "
                                         f"may leave a level-side's mask absent (dense); the decode step's masks are made for every level-side")

    def level(self, side: str, level: int) -> torch.Tensor | None:
        """``side``'s level mask, or ``None`` where it is absent (dense)."""
        return getattr(self, side)[level]

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """The mask tensors that are present."""
        return tuple(mask for mask in (*self.read, *self.write) if mask is not None)

    def expected_shape(self, routes, side: str, level: int) -> tuple[int, ...]:
        """The shape ``side``'s level mask has against ``routes``' rows."""
        rows = routes.rows(level, side)
        B, T, H, W = rows.shape
        if self.consumer == PREFILL:
            return (B, H, -(-T // WORD_BITS), W)
        return (B, H, -(-W // WORD_BITS))

    def check(self, routes) -> None:
        """Shape, dtype and device against the bundle's rows, and nothing about the values; an absent (``None``) entry is dense and holds nothing to check."""
        D = routes.topology.D
        for side in ("read", "write"):
            masks = getattr(self, side)
            if len(masks) != D:
                raise ValueError(f"Masks.{side} holds one tensor per level (D={D}), got {len(masks)}")
            for level, mask in enumerate(masks):
                if mask is None:
                    continue
                want = self.expected_shape(routes, side, level)
                if not isinstance(mask, torch.Tensor) or mask.dtype is not WORD_DTYPE or tuple(mask.shape) != want:
                    raise ValueError(
                        f"Masks.{side}[{level}] ({self.consumer} layout) is int32 {want}; got "
                        f"{getattr(mask, 'dtype', type(mask).__name__)} {tuple(getattr(mask, 'shape', ()))}")
                if mask.device != routes.device:
                    raise ValueError(f"Masks.{side}[{level}] is on {mask.device}, the rows on {routes.device}")

    def bits(self, side: str, level: int, tokens: int, width: int) -> torch.Tensor:
        """``side``'s level as bool ``[B, T, H, width]``: the bits unpacked, for the references. An absent entry holds no
        tensor to unpack: it is refused here by name and :meth:`live` gives its bits."""
        mask = self.level(side, level)
        if mask is None:
            raise ValueError(f"Masks.{side}[{level}] is absent (dense) and holds no bits to unpack; Masks.live(routes, {side!r}, {level}) "
                             f"gives its bits, all live")
        return unpack_tokens(mask, tokens) if self.consumer == PREFILL else unpack_digits(mask, width)

    def live(self, routes, side: str, level: int) -> torch.Tensor:
        """``side``'s level as bool ``[B, T, H, width]`` against ``routes``' rows: :meth:`bits` of a present entry, and for an absent
        (dense) one all True on the rows' shape and device. No host read."""
        rows = routes.rows(level, side)
        if self.level(side, level) is None:
            return torch.ones(rows.shape, dtype=torch.bool, device=rows.device)
        return self.bits(side, level, rows.shape[1], rows.shape[3])


def _live_byte(live: torch.Tensor) -> torch.Tensor:
    """The share of a level's ``(token, digit)`` pairs that are live as a byte: bool ``[B, T, H, width]`` -> int64 scalar
    ``(2 * 255 * live + total) // (2 * total)``, the share times 255 rounded to the nearest, a half up, in integers."""
    total = live.numel()
    return (2 * 255 * live.sum(dtype=torch.int64) + total) // (2 * total)


def _density(read_live, write_live, device: torch.device) -> torch.Tensor:
    """``[2, D]`` uint8 on ``device``: row 0 the read side's bytes, row 1 the write side's, an entry a level. A side's entry is ``None``
    for a level-side with no mask to count (dense), and its byte is the constant 255, made on ``device`` with no read."""
    dense = torch.full((), 255, dtype=torch.int64, device=device)
    return torch.stack([torch.stack([dense if live is None else _live_byte(live) for live in side])
                        for side in (read_live, write_live)]).to(torch.uint8)


def masks_and_density(routes: RouteFactors) -> tuple[Masks, torch.Tensor]:
    """THE PREFILL'S MASKS AND DENSITY from a bundle's rows: a digit is live where its row is not zero. Returns the
    :data:`PREFILL`-layout :class:`Masks` and ``density`` ``[2, D]`` uint8 on the rows' device -- row 0 the read side, row 1 the
    write side, an entry a level: a byte ``b`` for the share ``b / 255`` of the level's ``(token, digit)`` pairs, over the batch and
    the side's heads, that are live (255 where every digit is, 0 where none is). With ``live`` those pairs and ``total`` all of
    them, ``b = (2 * 255 * live + total) // (2 * total)``: the nearest byte, a half up, in integer arithmetic. The masks built this way keep the guarantee a mask of the
    prefill gives (docs/api.md section 2.6). A level-side whose declared activation cannot make exact zeros (softmax) gets NO mask
    (its entry is ``None``: dense) and the byte 255, decided from the declaration with no read of the rows; every other level-side
    gets its mask, however dense its rows are. Torch operations on the rows' device; no host read."""
    can_zero = activation_walks(routes.topology)
    live = tuple(tuple(routes.rows(level, side) != 0 if can_zero[s][level] else None for level in range(routes.topology.D))
                 for s, side in enumerate(("read", "write")))
    masks = Masks(PREFILL, *(tuple(None if level is None else pack_tokens(level) for level in side) for side in live))
    return masks, _density(*live, routes.device)


def density_of_masks(routes: RouteFactors, masks: Masks) -> torch.Tensor:
    """THE DENSITY of :data:`PREFILL` masks a caller already has: ``[2, D]`` uint8 on the masks' device, row 0 the read side, row 1
    the write side, an entry a level -- a byte ``b`` for the share ``b / 255`` of the level's ``(token, digit)`` pairs, over the batch
    and the side's heads, whose bit is set (255 where every bit is, 0 where none is), rounded as :func:`masks_and_density` rounds it:
    ``b = (2 * 255 * live + total) // (2 * total)``. An absent entry (dense) is 255, with no bits to count. It takes the BUNDLE, not a token count: the bundle carries the
    token count and the shapes the masks are checked against (:meth:`Masks.check` runs first), so a count that disagrees with
    the masks cannot be handed over. Torch operations on the masks' device; no host read."""
    if masks.consumer != PREFILL:
        raise ValueError(f"the density is the prefill's: the masks are {masks.consumer!r}-layout, not {PREFILL!r}")
    masks.check(routes)
    live = tuple(tuple(None if masks.level(side, level) is None else masks.live(routes, side, level)
                       for level in range(routes.topology.D)) for side in ("read", "write"))
    return _density(*live, routes.device)
