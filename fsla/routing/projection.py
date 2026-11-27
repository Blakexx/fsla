"""THE ROUTING PROJECTION: the packed router layout of a topology at a group, and the one GEMM that projects it.

``route_W`` is ``[H_w, hidden, columns]``, one slice a WRITE head: its write levels' columns first, in level order, then
each of its ``G`` read heads' untied levels' columns (head 0's levels in order, then head 1's, ...), then the gain's
span. A tied read entry has no columns: that head's level is the write side's rows. Every column is consumed, so the
projection is ONE GEMM over the whole parameter and a side's logits are column spans of its output, never a gather.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import torch

from fsla.routing.types import Tied, Topology

__all__ = ["RouterLayout", "router_logits"]


def _offsets(widths) -> tuple[int, ...]:
    offsets = [0]
    for width in widths:
        offsets.append(offsets[-1] + width)
    return tuple(offsets)


@dataclass(frozen=True)
class RouterLayout:
    """The packed layout of ``topology`` read by ``group`` read heads a write head: the projection's columns, a side's
    level offsets in its solve plane, and the solve's support-word columns. Every number is derived from the two."""

    topology: Topology
    group: int

    def __post_init__(self) -> None:
        self.topology.read_heads(self.group)

    @cached_property
    def heads(self) -> tuple[tuple, ...]:
        """The ``group`` read heads' tuples."""
        return self.topology.read_heads(self.group)

    @property
    def D(self) -> int:
        return self.topology.D

    @property
    def widths(self) -> tuple[int, ...]:
        return self.topology.widths

    @cached_property
    def level_offsets(self) -> tuple[int, ...]:
        """``D + 1`` cumulative offsets of a side's levels laid end to end: the write region of ``route_W`` and the
        columns of each side's solve plane."""
        return _offsets(self.widths)

    @property
    def side_width(self) -> int:
        """Columns of one side's packed levels: the write region's, and a read head's solve plane."""
        return self.level_offsets[-1]

    @cached_property
    def _read_starts(self) -> dict[tuple[int, int], int]:
        starts, column = {}, self.side_width
        for head, entries in enumerate(self.heads):
            for level, entry in enumerate(entries):
                if type(entry) is not Tied:
                    starts[(head, level)] = column
                    column += self.widths[level]
        return starts

    @cached_property
    def packed_router_width(self) -> int:
        """``route_W``'s routing columns a write head: the write region and every untied read entry's span."""
        return self.side_width + sum(self.widths[level] for _, level in self._read_starts)

    def write_columns(self, level: int) -> slice:
        """The column span of write level ``level`` in ``route_W`` and in the projection's output."""
        return slice(self.level_offsets[level], self.level_offsets[level + 1])

    def read_columns(self, head: int, level: int) -> slice | None:
        """The column span of read head ``head``'s level ``level``, ``None`` where the entry is tied."""
        start = self._read_starts.get((head, level))
        return None if start is None else slice(start, start + self.widths[level])

    def tied_heads(self, level: int) -> tuple[int, ...]:
        """The read heads whose entry on ``level`` is tied: their rows there are the write side's."""
        return tuple(head for head, entries in enumerate(self.heads) if type(entries[level]) is Tied)

    @cached_property
    def write_support_offsets(self) -> tuple[int, ...]:
        """Each level's first column in the write solve's support words, or -1 where the write activation makes no
        exact zeros (it writes none)."""
        return self._support(lambda level: self.topology.can_zero("write", level))

    @cached_property
    def read_support_offsets(self) -> tuple[int, ...]:
        """Each level's first column in the read solves' support words -- one plane over every read head, each head's
        rows its streams -- or -1 where no read head's untied entry makes exact zeros."""
        return self._support(lambda level: any(
            type(entries[level]) is not Tied and entries[level].can_zero for entries in self.heads))

    def _support(self, has) -> tuple[int, ...]:
        offsets, column = [], 0
        for level, width in enumerate(self.widths):
            offsets.append(column if has(level) else -1)
            column += width if has(level) else 0
        return tuple(offsets)

    @property
    def write_support_width(self) -> int:
        return sum(w for w, o in zip(self.widths, self.write_support_offsets) if o >= 0)

    @property
    def read_support_width(self) -> int:
        return sum(w for w, o in zip(self.widths, self.read_support_offsets) if o >= 0)


def router_logits(
    h: torch.Tensor,
    route_W: torch.Tensor,
    layout: RouterLayout,
    *,
    route_bias: torch.Tensor | None = None,
    projection_dtype: torch.dtype,
    token_major: bool = False,
    gain_columns: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``(logits, gain_logits)``: the routing columns of ``h @ route_W`` in ``projection_dtype``, and the gain's.

    ``logits`` is ``[B * H_w, T, layout.packed_router_width]`` (stream ``b * H_w + w``), or with ``token_major`` the
    same numbers as ``[B, T, H_w, packed_router_width]`` WITHOUT the transpose flattening ``(B, H_w)`` needs: the
    sgemm's own output is already that layout, so the flat form costs a pass over the plane that it is the only reason
    for; the solve reads either through its stride tuple. A level's logits are a column span of it
    (:meth:`RouterLayout.write_columns`, :meth:`RouterLayout.read_columns`). BIT-IDENTICAL in either layout: a column
    of the output is a dot product over the full hidden dim, independent of where it is written -- gated by
    ``test_the_two_projection_layouts_are_one_gemm_and_agree_at_the_bit``.

    ``gain_logits`` is ``None`` when ``gain_columns`` is 0 and ``[B, T, H_w, gain_columns]`` otherwise: the gain rides
    the same GEMM as its trailing columns, so it never becomes a launch that would not have happened anyway.
    """
    if type(layout) is not RouterLayout:
        raise TypeError(f"layout must be a RouterLayout, got {type(layout).__name__}")
    if type(gain_columns) is not int or gain_columns < 0:
        raise ValueError(f"gain_columns must be a nonnegative exact int, got {gain_columns!r}")
    if not isinstance(route_W, torch.Tensor) or route_W.ndim != 3:
        raise ValueError(f"route_W must be [H_w, hidden_size, columns], got {getattr(route_W, 'shape', route_W)!r}")
    routed = layout.packed_router_width
    heads = route_W.shape[0]
    if heads < 1 or route_W.shape[2] != routed + gain_columns:
        raise ValueError(f"route_W must be [H_w >= 1, hidden_size, {routed} routing + {gain_columns} gain columns], "
                         f"got {tuple(route_W.shape)}")
    if route_bias is not None and tuple(route_bias.shape) != (heads, routed + gain_columns):
        raise ValueError(f"route_bias must have shape {(heads, routed + gain_columns)}, got {tuple(route_bias.shape)}")
    if not isinstance(h, torch.Tensor) or h.ndim not in (3, 4) or h.shape[-1] != route_W.shape[1]:
        raise ValueError(f"h must be [B, T, hidden] or [B, T, H_w, hidden] with hidden = route_W's {route_W.shape[1]}, "
                         f"got {getattr(h, 'shape', None)}")
    if h.ndim == 4 and h.shape[2] != heads:
        raise ValueError(f"h's rank-4 head count must match route_W's {heads} write heads, got {h.shape[2]}")

    #: `blhc` is the sgemm's OWN output order, so the einsum returns a view of it and the head axis costs nothing;
    #: `bhlc` is that plus a full-plane transpose, which exists only to let `(B, H_w)` flatten into one stream axis.
    order = "blhc" if token_major else "bhlc"
    equation = f"bld,hdc->{order}" if h.ndim == 3 else f"blhd,hdc->{order}"
    out = torch.einsum(equation, h.to(projection_dtype), route_W.to(projection_dtype))
    if route_bias is not None:
        bias = route_bias.to(projection_dtype)
        out = out + (bias[None, None, :, :] if token_major else bias[None, :, None, :])
    gain = None
    if gain_columns:
        gain = out[..., routed:]
        if not token_major:
            gain = gain.permute(0, 2, 1, 3)
    logits = out[..., :routed]
    if token_major:
        return logits, gain
    return logits.reshape(h.shape[0] * heads, h.shape[1], routed), gain
