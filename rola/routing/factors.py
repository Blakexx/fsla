# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The routing bundle: the op's first operand, and the package's ONE duck boundary.

Linear attention's op takes ``(q, k, v)``. RoLA's takes ``(routes, v)``, where ``routes``
is the ``(q, k)`` pair AFTER the feature map -- the read-side and write-side simplices
and their side gains. That bundle is :class:`RouteFactors`, and it is the whole producer
contract: **any callable that returns a well-formed one is a producer.** There is no
producer base class to subclass, no per-level tier and no assembly gateway, because none
of them was a boundary anything checked -- the bundle's own invariants are, and they are
structural rather than provenance-based.

The bundle carries tensors and nothing else, which is why it is constructible rather
than sealed. What its invariants are, and why unit-sum-ness is not among them:
``docs/api.md`` section 2.1. :class:`~rola.routing.producer.RouteProducer` is the one
this package ships; hand construction is equally legal and equally checked.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rola.routing import quant
from rola.routing.types import Topology

__all__ = ["RouteFactors"]


def _quantize_side(levels):
    rows, scales, dequantized = [], [], []
    for level in levels:
        row, scale = quant.quantize_unit_max_rows(level)
        rows.append(row)
        scales.append(scale)
        dequantized.append(quant.straight_through(level, quant.dequantize_rows(row, scale)))
    return tuple(rows), tuple(scales), tuple(dequantized)


@dataclass(frozen=True, slots=True)
class RouteFactors:
    """The feature-mapped routing sides in THE INTERNAL FORMAT, and their side gain. Tensors, and nothing else.

    ``amp_read``/``amp_write``: one u8 ``[B, T, H, width_l]`` row set per level, each row UNIT-MAX (its
    largest digit at 255; an all-zero row is a dead token at that level); ``scale_read``/``scale_write``:
    the rows' maxima, ``[B, T, H]`` per level, in the producer's floating dtype -- a read level's scale
    carries the read mass, a write level's the normalized simplex's max (its mass rides ``g_write``).
    ``read``/``write``: the DEQUANTIZED levels, ``rows / 255 * scale``, as the values every consumer
    computes on, with the producer's gradient stepped straight through the rounding; they are a
    transitional carry until the intra, decode and carry kernels read the rows themselves.
    ``g_write``: ``[B, T, H]`` write-side gain (there is no read gain: the readout is a ratio).
    ``read_mass``: the ``[B, T, H]`` per-token read scale the read levels carry, or ``None``.
    ``digits``: the amplitude planes' digit count (one).

    The format is written by the framework's producers and by nothing else (ruled 2026-09-26): a new
    production is a producer class here. Direct construction is legal for a fixture; ``__post_init__``
    holds it to the format's invariants (``docs/api.md`` section 2.1).
    """

    topology: Topology
    read: tuple[torch.Tensor, ...]
    write: tuple[torch.Tensor, ...]
    g_write: torch.Tensor
    amp_read: tuple[torch.Tensor, ...]
    amp_write: tuple[torch.Tensor, ...]
    scale_read: tuple[torch.Tensor, ...]
    scale_write: tuple[torch.Tensor, ...]
    read_mass: torch.Tensor | None = None
    digits: int = 1  #: reserved: the amplitude planes' digit count, one until a two-digit plane is declared

    def __post_init__(self) -> None:
        """Shapes, devices, dtypes and ``None``-ness only -- never a device reduction."""
        object.__setattr__(self, "read", tuple(self.read))
        object.__setattr__(self, "write", tuple(self.write))
        if type(self.topology) is not Topology:
            raise TypeError(
                f"RouteFactors.topology must be a Topology, got {type(self.topology).__name__}")
        D, widths = self.topology.D, self.topology.widths

        for name, levels in (("read", self.read), ("write", self.write)):
            if len(levels) != D:
                raise ValueError(
                    f"RouteFactors.{name} must hold one tensor per routing level "
                    f"(D={D}), got {len(levels)}")

        prefix = None
        for name, levels in (("read", self.read), ("write", self.write)):
            for level, tensor in enumerate(levels):
                if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
                    shape = getattr(tensor, "shape", type(tensor).__name__)
                    raise TypeError(
                        f"RouteFactors.{name}[{level}] must be a [B, T, H, width_l] "
                        f"tensor, got {shape}")
                if tensor.shape[-1] != widths[level]:
                    raise ValueError(
                        f"RouteFactors.{name}[{level}] is {tensor.shape[-1]} wide but the "
                        f"topology's level {level} is {widths[level]}")
                if prefix is None:
                    prefix = tuple(tensor.shape[:3])
                elif tuple(tensor.shape[:3]) != prefix:
                    raise ValueError(
                        f"every routing level shares one [B, T, H] prefix; "
                        f"{name}[{level}] is {tuple(tensor.shape[:3])} against {prefix}")
                if tensor.device != self.read[0].device:
                    raise ValueError(
                        f"every routing level lives on one device; {name}[{level}] is on "
                        f"{tensor.device} against {self.read[0].device}")
                if tensor.dtype != self.read[0].dtype:
                    raise ValueError(
                        f"every routing level shares one dtype; {name}[{level}] is "
                        f"{tensor.dtype} against {self.read[0].dtype}")
        object.__setattr__(self, "amp_read", tuple(self.amp_read))
        object.__setattr__(self, "amp_write", tuple(self.amp_write))
        object.__setattr__(self, "scale_read", tuple(self.scale_read))
        object.__setattr__(self, "scale_write", tuple(self.scale_write))
        if self.digits != 1:
            raise ValueError(f"RouteFactors.digits is {self.digits}; the amplitude planes are one u8 digit")
        for name, rows, scales, levels in (("read", self.amp_read, self.scale_read, self.read),
                                           ("write", self.amp_write, self.scale_write, self.write)):
            if len(rows) != D or len(scales) != D:
                raise ValueError(f"RouteFactors.amp_{name}/scale_{name} hold one tensor per level (D={D}), "
                                 f"got {len(rows)}/{len(scales)}")
            for level, (row, scale, dq) in enumerate(zip(rows, scales, levels)):
                if not isinstance(row, torch.Tensor) or row.dtype is not torch.uint8 or tuple(row.shape) != tuple(dq.shape):
                    raise TypeError(f"RouteFactors.amp_{name}[{level}] is the level's u8 rows, {tuple(dq.shape)}; got "
                                    f"{getattr(row, 'dtype', type(row).__name__)} {tuple(getattr(row, 'shape', ()))}")
                if not isinstance(scale, torch.Tensor) or tuple(scale.shape) != prefix:
                    raise ValueError(f"RouteFactors.scale_{name}[{level}] is the rows' maxima, [B, T, H]={prefix}; got "
                                     f"{tuple(getattr(scale, 'shape', ()))}")
        for name, column in (("g_write", self.g_write),
                             ("read_mass", self.read_mass)):
            if column is None:
                continue
            if not isinstance(column, torch.Tensor) or tuple(column.shape) != prefix:
                shape = getattr(column, "shape", type(column).__name__)
                raise ValueError(
                    f"RouteFactors.{name} must be a [B, T, H]={prefix} tensor, got {shape}")

    @classmethod
    def from_levels(cls, topology: Topology, read, write, g_write, read_mass=None) -> RouteFactors:
        """THE FRAMEWORK'S EPILOGUE as a function: floating levels in, the format out. Each level's rows are
        stored u8 unit-max with their maxima, and the levels a consumer computes on are the dequantized values
        with the input's gradient stepped straight through the rounding (`rola.routing.quant`)."""
        amp_read, scale_read, dq_read = _quantize_side(read)
        amp_write, scale_write, dq_write = _quantize_side(write)
        return cls(topology=topology, read=dq_read, write=dq_write, g_write=g_write,
                   amp_read=amp_read, amp_write=amp_write, scale_read=scale_read, scale_write=scale_write,
                   read_mass=read_mass)

    @property
    def batch(self) -> int:
        return self.read[0].shape[0]

    @property
    def tokens(self) -> int:
        return self.read[0].shape[1]

    @property
    def heads(self) -> int:
        return self.read[0].shape[2]

    @property
    def device(self) -> torch.device:
        return self.read[0].device

    def read_needs_normalization(self) -> tuple[bool, ...]:
        """Per read level: is it still off the simplex? THE PREDICATE, stated once.

        :meth:`read_simplex` divides exactly these levels and the decode fold normalizes
        exactly these levels inside its kernel; two sites folding one question from the
        same facts is one divergence waiting to happen.
        """
        if self.read_mass is None:
            return (False,) * len(self.read)
        return tuple(not self.topology.duty_simplex_output(index, "read")
                     for index in range(len(self.read)))

    def read_simplex(self) -> tuple[torch.Tensor, ...]:
        """The read levels on the simplex -- ``read`` with :attr:`read_mass` divided out.

        The consumer a bundle is built for reads ``(read, read_mass)`` unscaled; this is
        for every consumer that is defined on the simplex instead (the fp64 oracle and
        its contract, the decode arm, the reference recurrence).
        """
        if self.read_mass is None:
            return self.read
        #: PER LEVEL, not by the product: `read_mass` is the product the readout's ratio
        #: takes, but a simplex is a per-level claim, and the levels the topology already
        #: declares on it are returned untouched so this is exactly the bundle a
        #: mass-carrying one would have been.
        #:
        #: THE QUOTIENT IS AT LEAST fp32: a bf16 quotient of a bf16 sum of bf16 addends
        #: is three roundings where one is forced, and every consumer of this widens
        #: anyway. A level already wider than fp32 keeps its own width.

        def divided(level: torch.Tensor) -> torch.Tensor:
            wide = torch.promote_types(level.dtype, torch.float32)
            return level.to(wide) / level.sum(dim=-1, keepdim=True, dtype=wide)

        return tuple(divided(level) if needs else level
                     for level, needs in zip(self.read, self.read_needs_normalization(),
                                             strict=True))

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Every tensor the OPERATION consumes, for the gradient-reachability predicate."""
        mass = () if self.read_mass is None else (self.read_mass,)
        return (*self.read, *self.write, self.g_write, *mass)
