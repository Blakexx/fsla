# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""A SPARSE LEVEL IN THE INTERNAL FORMAT: one side of one routing level as its bitlist and its live coefficients
packed in bit order.

A level whose activation has exact zeros is stored this way on that side; every other level stays unit-max rows
(:class:`~rola.routing.factors.RouteFactors`). The coefficients are the rows' own digits (u8, or f16 above depth two);
only their placement differs. Two forms, by the call's shape -- docs/internals/routing/sparse.md:

THE DIGIT FORM (``T > 1``): ``words`` int32 ``[B, H, width, nw]``, bit ``t`` of word ``i`` set when token ``32 i + t``
is live on the digit (``nw`` is ``T / 32`` rounded up to a whole word, then to four; the bits past ``T`` are zero), and
``terms`` ``[B, H, width, windows, slot]``: each (digit, window) slot holds the window's live coefficients from its
front in token order, so the ``k``-th set bit of the digit's window words is the slot's ``k``-th entry. ``slot`` is the
window, or ``T`` rounded up to a word below it. Past a slot's live count its entries are unspecified.

THE TOKEN FORM (``T == 1``, the decode step): ``words`` int32 ``[B, H, ceil(width / 32)]``, the token's mask over the
digits, and ``terms`` ``[B, H, width]``, its live coefficients in digit order from the front.

ANY (words, terms) PAIR IS A VALID LEVEL: a set bit over a zero term costs a slot and contributes nothing, and a term
no bit reaches is never read. So the structure is the only invariant checked.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from rola.engine.constants import WINDOW
from rola.routing import quant

__all__ = ["WORD_TOKENS", "SparseLevel", "slot_of", "word_count"]

#: Tokens a word covers.
WORD_TOKENS = 32


def word_count(tokens: int) -> int:
    """A digit's words for ``tokens`` tokens: whole words, rounded up to four (a 16-byte row)."""
    words = -(-tokens // WORD_TOKENS)
    return -(-words // 4) * 4


def slot_of(tokens: int) -> int:
    """A term slot's entries: the window, or the call's tokens rounded to a word below it."""
    return min(WINDOW, -(-tokens // WORD_TOKENS) * WORD_TOKENS)


def _pack_bits(live: torch.Tensor) -> torch.Tensor:
    """``live [..., n * 32]`` bool -> int32 ``[..., n]``, bit ``t`` of word ``i`` for element ``32 i + t``."""
    shifts = torch.arange(WORD_TOKENS, device=live.device, dtype=torch.int64)
    words = (live.reshape(*live.shape[:-1], -1, WORD_TOKENS).to(torch.int64) << shifts).sum(-1)
    return torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)


def _unpack_bits(words: torch.Tensor, count: int) -> torch.Tensor:
    """The inverse of :func:`_pack_bits`, the first ``count`` elements."""
    shifts = torch.arange(WORD_TOKENS, device=words.device, dtype=torch.int64)
    bits = ((words.to(torch.int64) & 0xFFFFFFFF).unsqueeze(-1) >> shifts) & 1
    return bits.reshape(*words.shape[:-1], -1)[..., :count].bool()


def _front_pack(values: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
    """Each row's live entries moved to its front in order, the rest zero."""
    order = torch.sort((~live).to(torch.int8), dim=-1, stable=True).indices
    packed = torch.gather(values, -1, order)
    return packed.masked_fill(~torch.gather(live, -1, order), 0)


def _unpack_front(terms: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
    """The inverse of :func:`_front_pack`: entry ``k`` of a row lands at its ``k``-th live position."""
    rank = (live.to(torch.int64).cumsum(-1) - 1).clamp_min(0)
    return torch.gather(terms, -1, rank).masked_fill(~live, 0)


@dataclass(frozen=True, slots=True)
class SparseLevel:
    """One side of one routing level: ``words`` (the bitlist) and ``terms`` (its live coefficients in bit order)."""

    words: torch.Tensor
    terms: torch.Tensor

    def __post_init__(self) -> None:
        """Shapes, devices and dtypes only -- never a device reduction."""
        words, terms = self.words, self.terms
        if not isinstance(words, torch.Tensor) or words.dtype is not torch.int32:
            raise TypeError(f"SparseLevel.words is int32, got {getattr(words, 'dtype', type(words).__name__)}")
        if not isinstance(terms, torch.Tensor) or terms.dtype not in quant.ROW_DTYPES:
            raise TypeError(f"SparseLevel.terms is one of {quant.ROW_DTYPES}, got "
                            f"{getattr(terms, 'dtype', type(terms).__name__)}")
        if words.device != terms.device:
            raise ValueError(f"SparseLevel's words and terms live on one device: {words.device}, {terms.device}")
        if words.ndim == 3 and terms.ndim == 3:
            if words.shape[:2] != terms.shape[:2] or words.shape[2] != -(-terms.shape[2] // WORD_TOKENS):
                raise ValueError(f"a token-form SparseLevel's mask is [B, H, ceil(width / 32)] over terms [B, H, width]; "
                                 f"got {tuple(words.shape)} and {tuple(terms.shape)}")
        elif words.ndim == 4 and terms.ndim == 5:
            if words.shape[:3] != terms.shape[:3] or words.shape[3] % 4 or words.shape[3] * WORD_TOKENS < (
                    terms.shape[3] - 1) * terms.shape[4]:
                raise ValueError(f"a digit-form SparseLevel is words [B, H, width, nw] and terms "
                                 f"[B, H, width, windows, slot]; got {tuple(words.shape)} and {tuple(terms.shape)}")
        else:
            raise ValueError(f"a SparseLevel is the digit form (words 4-d, terms 5-d) or the token form (both 3-d); "
                             f"got {words.ndim}-d words and {terms.ndim}-d terms")

    @property
    def token_form(self) -> bool:
        return self.words.ndim == 3

    @property
    def width(self) -> int:
        return self.terms.shape[2]

    @property
    def dtype(self) -> torch.dtype:
        return self.terms.dtype

    @classmethod
    def from_rows(cls, rows: torch.Tensor) -> SparseLevel:
        """THE REFERENCE PACKING: unit-max rows ``[B, T, H, width]`` in, the level out (a zero digit is dead)."""
        B, T, H, width = rows.shape
        if T == 1:
            x = rows[:, 0]
            live = x != 0
            pad = -width % WORD_TOKENS
            words = _pack_bits(torch.nn.functional.pad(live, (0, pad)))
            return cls(words=words.contiguous(), terms=_front_pack(x, live).contiguous())
        x = rows.permute(0, 2, 3, 1)
        live = x != 0
        whole = -(-T // WORD_TOKENS) * WORD_TOKENS
        words = _pack_bits(torch.nn.functional.pad(live, (0, whole - T)))
        words = torch.nn.functional.pad(words, (0, word_count(T) - whole // WORD_TOKENS))
        slot = slot_of(T)
        windows = -(-T // slot)
        span = windows * slot - T
        xs = torch.nn.functional.pad(x, (0, span)).reshape(B, H, width, windows, slot)
        ls = torch.nn.functional.pad(live, (0, span)).reshape(B, H, width, windows, slot)
        return cls(words=words.contiguous(), terms=_front_pack(xs, ls).contiguous())

    def dense(self, tokens: int) -> torch.Tensor:
        """The rows ``[B, T, H, width]`` this level holds -- for the oracle and the references, never a kernel path."""
        if self.token_form:
            live = _unpack_bits(self.words, self.width)
            return _unpack_front(self.terms, live).unsqueeze(1)
        B, H, width, windows, slot = self.terms.shape
        covered = min(self.words.shape[-1] * WORD_TOKENS, windows * slot)
        live = torch.nn.functional.pad(_unpack_bits(self.words, covered), (0, windows * slot - covered))
        live = live.reshape(B, H, width, windows, slot)
        rows = _unpack_front(self.terms, live).reshape(B, H, width, windows * slot)[..., :tokens]
        return rows.permute(0, 1, 3, 2).contiguous().transpose(1, 2)
