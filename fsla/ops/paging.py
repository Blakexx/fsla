"""The state page's FORMAT and its codec: byte geometry, encode and decode (docs/internals/common/state_page.md).

Vocabulary: ``N`` = leaf count; ``cols`` = ``d_v`` plus one mass column (the ratio readout's denominator);
**page** = one ``MMA_K_QUANTUM``-leaf block in canonical leaf order, stored as s24 planes at a group exponent.
The host codec is the oracle and test seam: a measured path never rejoins a page on the host.
"""
from __future__ import annotations

from typing import NamedTuple

import torch

from fsla.engine.types import MMA_K_QUANTUM


class PageLayout(NamedTuple):
    """THE STATE PAGE'S BYTE GEOMETRY, the mirror of `csrc/fsla/src/common/state_page.cuh`'s `Blocks<DV>`: 16 leaves as
    ``[16 x DV hi s16][16 x DV lo u8][16 x DV/32 exponent s8][16 mass fp32]``, the mass 16-byte aligned
    (docs/internals/common/state_page.md#geometry)."""

    hi: int
    lo: int
    exp: int
    mass: int
    bytes: int
    groups: int


def page_layout(d_v: int) -> PageLayout:
    if d_v % 32:
        raise ValueError(f"a channel group is 32 channels, so d_v is a multiple of 32; got {d_v}")
    groups, rows = d_v // 32, MMA_K_QUANTUM
    lo = rows * d_v * 2
    exp = lo + rows * d_v
    mass = (exp + rows * groups + 15) & ~15
    return PageLayout(hi=0, lo=lo, exp=exp, mass=mass, bytes=mass + rows * 4, groups=groups)


#: The s24 mantissa's bound and the smallest group exponent, the kernel's own (`state_page.cuh`).
_MANTISSA_MAX = (1 << 23) - 1
_EXP_MIN = -126


def encode_pages(logical: torch.Tensor) -> torch.Tensor:
    """A logical ``[..., 16, d_v + 1]`` fp32 page block in the STORED int8 state format, ``[..., page_bytes]`` uint8.

    Per leaf and 32-channel group: the exponent ``e = max(k - 23, -126)`` from the group's largest magnitude
    ``f * 2^k``, the value rounded once to the s24 mantissa ``m = round(v / 2^e)`` held inside s24, stored as its
    high s16 (``m >> 8``) and low u8 (``m & 255``) bytes; the mass as fp32. The kernel's arithmetic, bit for bit
    (docs/internals/common/state_page.md#codec).
    """
    _refuse_ragged(logical)
    d_v = logical.shape[-1] - 1
    layout = page_layout(d_v)
    pages = logical.reshape(-1, MMA_K_QUANTUM, d_v + 1).float()
    values = pages[..., :d_v].reshape(-1, MMA_K_QUANTUM, layout.groups, 32)
    amax = values.abs().amax(-1)
    _, k = torch.frexp(amax)
    #: an all-zero group stores exponent zero, so a zero page is all zero bytes.
    e = torch.where(amax == 0, torch.zeros_like(k), (k - 23).clamp_min(_EXP_MIN))
    m = torch.round(torch.ldexp(values, -e.unsqueeze(-1))).clamp(-_MANTISSA_MAX, _MANTISSA_MAX)
    m = m.to(torch.int32).reshape(-1, MMA_K_QUANTUM, d_v)
    out = torch.zeros((pages.shape[0], layout.bytes), dtype=torch.uint8, device=pages.device)
    out[:, layout.hi:layout.lo] = (m >> 8).to(torch.int16).reshape(pages.shape[0], -1).view(torch.uint8)
    out[:, layout.lo:layout.exp] = (m & 0xFF).to(torch.uint8).reshape(pages.shape[0], -1)
    out[:, layout.exp:layout.exp + MMA_K_QUANTUM * layout.groups] = (
        e.to(torch.int8).reshape(pages.shape[0], -1).view(torch.uint8))
    out[:, layout.mass:] = pages[..., d_v].contiguous().view(torch.uint8).reshape(pages.shape[0], -1)
    return out.reshape(*logical.shape[:-2], layout.bytes)


def decode_pages(stored: torch.Tensor, d_v: int, *, hi_only: bool = False) -> torch.Tensor:  #: reserved: hi_only is the gates' read-only reconstruction (tests/integration/test_decode_fold.py)
    """The inverse view: stored ``[..., page_bytes]`` uint8 pages as the logical ``[..., 16, d_v + 1]`` fp32 -- every
    leaf-channel ``(hi << 8 | lo) * 2^e`` and the mass; ``hi_only`` reads what a READ-ONLY load reconstructs, the hi
    plane alone (``lo`` taken as zero). ORACLE AND BRIDGE ONLY: a measured path never rejoins a page on the host
    (docs/internals/common/state_page.md#codec)."""
    layout = page_layout(d_v)
    if stored.dtype is not torch.uint8 or stored.shape[-1] != layout.bytes:
        raise ValueError(f"a d_v={d_v} state page is {layout.bytes} uint8 bytes; got {stored.dtype} "
                         f"{tuple(stored.shape)}")
    pages = stored.reshape(-1, layout.bytes)
    count = pages.shape[0]
    hi = pages[:, layout.hi:layout.lo].contiguous().view(torch.int16).to(torch.int32)
    lo = pages[:, layout.lo:layout.exp].to(torch.int32) * (0 if hi_only else 1)
    e = pages[:, layout.exp:layout.exp + MMA_K_QUANTUM * layout.groups].contiguous().view(torch.int8)
    #: an s24 mantissa and a power of two are both exact in fp32, and so is their product above the subnormals.
    m = ((hi << 8) | lo).reshape(count, MMA_K_QUANTUM, layout.groups, 32).float()
    values = torch.ldexp(m, e.to(torch.int32).reshape(count, MMA_K_QUANTUM, layout.groups, 1))
    out = torch.empty((count, MMA_K_QUANTUM, d_v + 1), dtype=torch.float32, device=stored.device)
    out[..., :d_v] = values.reshape(count, MMA_K_QUANTUM, d_v)
    out[..., d_v] = pages[:, layout.mass:].contiguous().view(torch.float32)
    return out.reshape(*stored.shape[:-1], MMA_K_QUANTUM, d_v + 1)


def to_split_planes(logical: torch.Tensor) -> torch.Tensor:
    """A logical ``[..., 16, cols]`` fp32 page block in the STORED split-plane form.

    THE PAGE IS ``[16 x DV hi][16 x DV lo][16 mass hi][16 mass lo]``:
    the same fp32 bits, re-laid out so a value row's hi plane is one 128 B line at
    ``DV = 64`` and an atom a call only READS moves half the bytes. The container stays
    fp32 because the LOGICAL element is fp32 -- ``hi || lo`` is the word exactly -- and
    the state's descriptor is what declares the layout (docs/internals/state.md#format).
    """
    _refuse_ragged(logical)
    _refuse_dtype(logical)
    pages = logical.reshape(-1, MMA_K_QUANTUM, logical.shape[-1]).contiguous()
    d_v = logical.shape[-1] - 1
    words = pages.view(torch.int32)
    hi, lo = (words >> 16).to(torch.int16), words.to(torch.int16)
    out = torch.empty_like(pages)
    stored = out.view(torch.int16).reshape(pages.shape[0], -1)
    rows = MMA_K_QUANTUM * d_v
    stored[:, :rows] = hi[:, :, :d_v].reshape(-1, rows)
    stored[:, rows:2 * rows] = lo[:, :, :d_v].reshape(-1, rows)
    stored[:, 2 * rows:2 * rows + MMA_K_QUANTUM] = hi[:, :, d_v]
    stored[:, 2 * rows + MMA_K_QUANTUM:] = lo[:, :, d_v]
    return out.reshape(logical.shape)


def _refuse_dtype(state: torch.Tensor) -> None:
    """The stored form is 32-bit words; a wider or narrower element is not a page."""
    if state.dtype is not torch.float32:
        raise ValueError(
            f"a state page is fp32 words split into bf16 planes, so the pack and the "
            f"gather run on float32; got {state.dtype}. Convert AFTER the gather, never "
            f"before it (docs/internals/common/state_page.md#geometry)")


def _refuse_ragged(state: torch.Tensor) -> None:
    """Refuse a state whose leaf axis is not a whole number of pages.

    THERE IS NO RAGGED STATE (Blake, 2026-08-29). A lawful state has
    ``N = prod_l B_l`` with every ``B_l`` a power of two at or above 16, so ``N`` is
    page-aligned by construction; a leaf axis that is not is a state no descriptor can
    name, and it is refused here rather than padded, split across a tail page, or
    served by a second layout (docs/internals/state.md#format).
    """
    n = state.shape[-2] if state.dim() >= 2 else 0
    if n % MMA_K_QUANTUM:
        raise ValueError(
            f"a state's leaf axis is a whole number of {MMA_K_QUANTUM}-leaf pages; got "
            f"{n}. N = prod_l B_l with each B_l a power of two at or above 16, so a "
            f"lawful state is page-aligned by construction (docs/internals/state.md#format)")


def bytes_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Byte identity of two STORED page blocks -- the split format's own comparison.

    ``torch.equal`` is a FLOAT comparison, and a stored page is not a float: its 32-bit
    words are two neighbouring elements' halves, so a page pair that is byte-identical
    can contain a word whose float reading is NaN and compare UNEQUAL. A claim about stored pages is about bytes, so it is asserted on the integer view -- which is
    also strictly stronger than the fp32 ``torch.equal`` it replaces.
    """
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int32),
                                              b.contiguous().view(torch.int32))


def from_split_planes(stored: torch.Tensor) -> torch.Tensor:
    """The hi||lo GATHER: stored pages back as the logical ``[..., 16, cols]`` fp32.

    The inverse of :func:`to_split_planes`, and exactly what ``materialize()`` is.
    """
    _refuse_dtype(stored)
    pages = stored.reshape(-1, MMA_K_QUANTUM, stored.shape[-1]).contiguous()
    d_v = stored.shape[-1] - 1
    halves = pages.view(torch.int16).reshape(pages.shape[0], -1).to(torch.int32) & 0xFFFF
    rows = MMA_K_QUANTUM * d_v
    words = torch.empty((pages.shape[0], MMA_K_QUANTUM, d_v + 1), dtype=torch.int32,
                        device=pages.device)
    words[:, :, :d_v] = ((halves[:, :rows] << 16)
                         | halves[:, rows:2 * rows]).reshape(-1, MMA_K_QUANTUM, d_v)
    words[:, :, d_v] = ((halves[:, 2 * rows:2 * rows + MMA_K_QUANTUM] << 16)
                        | halves[:, 2 * rows + MMA_K_QUANTUM:])
    return words.view(torch.float32).reshape(stored.shape)


__all__ = [
    "MMA_K_QUANTUM",
    "PageLayout",
    "bytes_equal",
    "decode_pages",
    "encode_pages",
    "from_split_planes",
    "page_layout",
    "to_split_planes",
]
