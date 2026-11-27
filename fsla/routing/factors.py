# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The routing bundle: the op's first operand, and the package's ONE duck boundary.

Linear attention's op takes ``(q, k, v)``. FSLA's takes ``(routes, v)``, where ``routes``
is the ``(q, k)`` pair AFTER the feature map -- the read-side and write-side levels, each a
scaled block (:class:`fsla.routing.quant.ScaledBlock`). That bundle is :class:`RouteFactors`,
and it is the whole producer contract: **any callable that returns a well-formed one is a
producer.** There is no producer base class to subclass, no per-level tier and no assembly
gateway, because none of them was a boundary anything checked -- the bundle's own invariants
are, and they are structural rather than provenance-based.

The bundle carries blocks and nothing else, which is why it is constructible rather
than sealed. What its invariants are, and why unit-sum-ness is not among them:
``docs/api.md`` section 2.1. The layer's producer (:func:`fsla.ops.produce.produce_prefill`) builds it from a declared
:class:`~fsla.routing.producer.Routing`; :meth:`RouteFactors.from_levels` builds it from any levels, equally checked.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fsla.routing import quant
from fsla.routing.quant import ScaledBlock
from fsla.routing.types import Topology

__all__ = ["RouteFactors"]

SIDES = ("read", "write")


@dataclass(frozen=True, slots=True)
class RouteFactors:
    """The feature-mapped routing sides in THE INTERNAL FORMAT: q and k as scaled blocks, one a level-side.

    ``read``/``write``: one :class:`~fsla.routing.quant.ScaledBlock` in the ``ROWS`` form a level, ``[B, T, H, width_l]``
    read and ``[B, T, H_w, width_l]`` write (``H_w`` write heads, each read by ``H / H_w`` consecutive query heads), each
    row UNIT-MAX -- u8 with its largest digit at 255, at every level, a zero a zero -- and its scale the row's max, fp32
    at its side's prefix. An all-zero row is a dead token at that level. Whatever magnitude a level carries, a
    producer's gain included, is in its scale; no level need sum to 1 (docs/CONSTITUTION.md#r23). A side's shared scale
    is the product of its levels'. A block's ``source`` is the producer's floating level, held ONLY when it requires a
    gradient -- the straight-through gradient's anchor for :meth:`dequantized` -- so an inference bundle holds the rows
    and nothing else.

    The format is written by the framework's producers and by nothing else (ruled 2026-09-26): a new
    production is a producer class here. Direct construction is legal for a fixture; ``__post_init__``
    holds it to the format's invariants (``docs/api.md`` section 2.1).
    """

    topology: Topology
    read: tuple[ScaledBlock, ...]
    write: tuple[ScaledBlock, ...]

    def __post_init__(self) -> None:
        """Shapes, devices, forms and ``None``-ness only -- never a device reduction."""
        object.__setattr__(self, "read", tuple(self.read))
        object.__setattr__(self, "write", tuple(self.write))
        if type(self.topology) is not Topology:
            raise TypeError(
                f"RouteFactors.topology must be a Topology, got {type(self.topology).__name__}")
        D, widths = self.topology.D, self.topology.widths
        prefixes, first = {}, None
        for side in SIDES:
            blocks, prefix = getattr(self, side), None
            if len(blocks) != D:
                raise ValueError(f"RouteFactors.{side} must hold one block per routing level (D={D}), got {len(blocks)}")
            for level, block in enumerate(blocks):
                if type(block) is not ScaledBlock or block.form is not quant.ROWS:
                    raise TypeError(f"RouteFactors.{side}[{level}] is a ScaledBlock in the {quant.ROWS.name} form, got "
                                    f"{getattr(block, 'form', type(block).__name__)!r}")
                if block.width != widths[level]:
                    raise ValueError(f"RouteFactors.{side}[{level}] is {block.width} wide but the topology's "
                                     f"level {level} is {widths[level]}")
                if prefix is None:
                    prefix = tuple(block.data.shape[:3])
                elif tuple(block.data.shape[:3]) != prefix:
                    raise ValueError(f"a side's routing levels share one prefix; {side}[{level}] is "
                                     f"{tuple(block.data.shape[:3])} against {prefix}")
                first = block if first is None else first
                if block.device != first.device:
                    raise ValueError(f"every routing level lives on one device; {side}[{level}] is on "
                                     f"{block.device} against {first.device}")
            prefixes[side] = prefix
        (b, t, h), (bw, tw, hw) = prefixes["read"], prefixes["write"]
        if (b, t) != (bw, tw) or hw < 1 or h % hw:
            raise ValueError(f"the write side's [B, T, H_w]={prefixes['write']} must share the read side's B and T "
                             f"and its H_w divide H ({prefixes['read']}): each write head is read by H / H_w "
                             "consecutive query heads (docs/api.md section 2.1)")

    @classmethod
    def from_levels(cls, topology: Topology, read, write) -> RouteFactors:
        """THE FACTORY: any nonnegative, finite floating levels in, the format out -- each level quantized into the
        rows' form against its own row max (`fsla.routing.quant.ROWS`: u8), refusing an entry the form does not hold,
        with one host read for them all. Whatever magnitude a level carries lands in its scale; there is nothing else
        to pass."""
        read, write = tuple(read), tuple(write)
        quant.refuse_unheld((quant.ROWS, level) for level in read + write)

        def blocks(levels):
            return tuple(ScaledBlock.quantize(level, quant.ROWS, checked=True) for level in levels)

        return cls(topology=topology, read=blocks(read), write=blocks(write))

    @property
    def batch(self) -> int:
        return self.read[0].batch

    @property
    def tokens(self) -> int:
        return self.read[0].tokens

    @property
    def heads(self) -> int:
        """The query heads ``H``: the read side's."""
        return self.read[0].heads

    @property
    def write_heads(self) -> int:
        """The write heads ``H_w``: the write side's, the values' and the state's."""
        return self.write[0].heads

    @property
    def group(self) -> int:
        """The query heads reading each write head, ``H / H_w`` (1 for an ungrouped layer)."""
        return self.heads // self.write_heads

    def refuse_grouped(self, consumer: str) -> None:
        """THE SEAM'S REFUSAL for a consumer that reads one head count: a grouped bundle (``group`` > 1) is refused by
        name rather than read at the wrong heads (docs/api.md section 2.1)."""
        if self.group != 1:
            raise NotImplementedError(f"{consumer} reads one head count, and this bundle groups {self.heads} query "
                                      f"heads over {self.write_heads} write heads (docs/api.md section 2.1)")

    @property
    def device(self) -> torch.device:
        return self.read[0].device

    def side(self, side: str) -> tuple[ScaledBlock, ...]:
        """``side``'s blocks, one a level."""
        if side not in SIDES:
            raise ValueError(f"side must be 'read' or 'write', got {side!r}")
        return getattr(self, side)

    def rows(self, level: int, side: str) -> torch.Tensor:
        """``level``'s rows ``[B, T, H, width_l]`` on ``side``."""
        return self.side(side)[level].data

    def scales(self, side: str) -> tuple[torch.Tensor, ...]:
        """``side``'s scales, fp32 ``[B, T, H]`` a level."""
        return tuple(block.scale for block in self.side(side))

    @property
    def read_rows(self) -> tuple[torch.Tensor, ...]:
        """Every read level as rows (:meth:`rows`), for the references."""
        return tuple(block.data for block in self.read)

    @property
    def write_rows(self) -> tuple[torch.Tensor, ...]:
        """Every write level as rows (:meth:`rows`), for the references."""
        return tuple(block.data for block in self.write)

    def dequantized(self) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """``(read, write)``: the levels as fp32 values, ``rows x scale``, computed at the call
        (:meth:`~fsla.routing.quant.ScaledBlock.dequantized`). With a ``source`` the producer's gradient steps
        straight through the rounding. A consumer still computing on floating levels calls this at its seam, and the
        call goes when that consumer reads the rows."""
        return (tuple(block.dequantized() for block in self.read),
                tuple(block.dequantized() for block in self.write))

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Every tensor the OPERATION consumes, for the gradient-reachability predicate."""
        return tuple(t for block in (*self.read, *self.write) for t in block.tensors())
