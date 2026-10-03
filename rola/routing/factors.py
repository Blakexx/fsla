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
``docs/api.md`` section 2.1. The layer's producer (:func:`rola.ops.produce.produce`) builds it from a declared
:class:`~rola.routing.producer.Routing`; :meth:`RouteFactors.from_levels` builds it from any levels, equally checked.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rola.routing import quant
from rola.routing.types import Topology

__all__ = ["RouteFactors"]

SIDES = ("read", "write")


def _requires_grad(levels) -> bool:
    return torch.is_grad_enabled() and any(level.requires_grad for level in levels)


@dataclass(frozen=True, slots=True)
class RouteFactors:
    """The feature-mapped routing sides in THE INTERNAL FORMAT, and their side gain. Tensors, and nothing else.

    ``amp_read``/``amp_write``: one ``[B, T, H, width_l]`` row set per level, each row UNIT-MAX -- u8 with its
    largest digit at 255, at every depth. An all-zero row is a dead token at that level. ``sparse``: ``(read, write)``, a bool a level -- the
    levels whose support the facts pass words (they may hold exact zeros worth skipping); set at production, read
    by every consumer, never derived from the activations. All dense by default. ``scale_read``/``scale_write``: the rows' maxima,
    fp32 ``[B, T, H]`` per level -- a read level's scale carries the read mass, a write level's the normalized
    simplex's max (its mass rides ``g_write``). ``g_write``: ``[B, T, H]`` write-side gain (there is no read
    gain: the readout is a ratio). ``read_mass``: the ``[B, T, H]`` per-token read scale the read levels carry,
    or ``None``. ``source``: the producer's floating ``(read, write)`` levels, held ONLY when they require a
    gradient -- the straight-through gradient's anchor for :meth:`dequantized`; ``None`` otherwise, so an
    inference bundle holds the rows and nothing else.

    The format is written by the framework's producers and by nothing else (ruled 2026-09-26): a new
    production is a producer class here. Direct construction is legal for a fixture; ``__post_init__``
    holds it to the format's invariants (``docs/api.md`` section 2.1).
    """

    topology: Topology
    amp_read: tuple[torch.Tensor, ...]
    amp_write: tuple[torch.Tensor, ...]
    scale_read: tuple[torch.Tensor, ...]
    scale_write: tuple[torch.Tensor, ...]
    g_write: torch.Tensor
    read_mass: torch.Tensor | None = None
    source: tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]] | None = None
    sparse: tuple[tuple[bool, ...], tuple[bool, ...]] | None = None

    def __post_init__(self) -> None:
        """Shapes, devices, dtypes and ``None``-ness only -- never a device reduction."""
        for name in ("amp_read", "amp_write", "scale_read", "scale_write"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if type(self.topology) is not Topology:
            raise TypeError(
                f"RouteFactors.topology must be a Topology, got {type(self.topology).__name__}")
        D = self.topology.D
        flags = ((False,) * D, (False,) * D) if self.sparse is None else tuple(tuple(map(bool, f)) for f in self.sparse)
        if len(flags) != 2 or any(len(f) != D for f in flags):
            raise ValueError(f"RouteFactors.sparse is (read, write), a bool a level (D={D}); got {self.sparse!r}")
        object.__setattr__(self, "sparse", flags)
        prefix, first = self._check_rows()
        if first.dtype is not quant.ROW_DTYPE:
            raise TypeError(f"RouteFactors rows are stored as {quant.ROW_DTYPE}, got {first.dtype}")
        for name in ("scale_read", "scale_write"):
            scales = getattr(self, name)
            if len(scales) != D:
                raise ValueError(f"RouteFactors.{name} holds one tensor per level (D={D}), got {len(scales)}")
            for level, scale in enumerate(scales):
                if (not isinstance(scale, torch.Tensor) or scale.dtype is not torch.float32
                        or tuple(scale.shape) != prefix):
                    raise ValueError(f"RouteFactors.{name}[{level}] is the rows' maxima, fp32 [B, T, H]={prefix}; "
                                     f"got {getattr(scale, 'dtype', type(scale).__name__)} "
                                     f"{tuple(getattr(scale, 'shape', ()))}")
        for name, column in (("g_write", self.g_write), ("read_mass", self.read_mass)):
            if column is None and name == "read_mass":
                continue
            if not isinstance(column, torch.Tensor) or tuple(column.shape) != prefix:
                shape = getattr(column, "shape", type(column).__name__)
                raise ValueError(f"RouteFactors.{name} must be a [B, T, H]={prefix} tensor, got {shape}")
        if self.source is not None:
            for side, levels, rows in (("read", self.source[0], self.amp_read),
                                       ("write", self.source[1], self.amp_write)):
                if len(levels) != D or any(tuple(a.shape) != tuple(r.shape) for a, r in zip(levels, rows)):
                    raise ValueError(f"RouteFactors.source's {side} levels are the rows' floating originals")

    def _check_rows(self) -> tuple[tuple[int, ...], torch.Tensor]:
        """Every row set: one tensor a level, ``[B, T, H, width_l]``, one prefix, device and dtype."""
        D, widths = self.topology.D, self.topology.widths
        prefix, first = None, None
        for name in ("amp_read", "amp_write"):
            rows = getattr(self, name)
            if len(rows) != D:
                raise ValueError(f"RouteFactors.{name} must hold one tensor per routing level (D={D}), got {len(rows)}")
            for level, tensor in enumerate(rows):
                if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
                    shape = getattr(tensor, "shape", type(tensor).__name__)
                    raise TypeError(f"RouteFactors.{name}[{level}] must be a [B, T, H, width_l] tensor, got {shape}")
                if tensor.shape[-1] != widths[level]:
                    raise ValueError(f"RouteFactors.{name}[{level}] is {tensor.shape[-1]} wide but the topology's "
                                     f"level {level} is {widths[level]}")
                if first is None:
                    prefix, first = tuple(tensor.shape[:3]), tensor
                elif tuple(tensor.shape[:3]) != prefix:
                    raise ValueError(f"every routing level shares one [B, T, H] prefix; {name}[{level}] is "
                                     f"{tuple(tensor.shape[:3])} against {prefix}")
                if tensor.device != first.device:
                    raise ValueError(f"every routing level lives on one device; {name}[{level}] is on "
                                     f"{tensor.device} against {first.device}")
                if tensor.dtype != first.dtype:
                    raise ValueError(f"every routing level shares one row dtype; {name}[{level}] is {tensor.dtype} "
                                     f"against {first.dtype}")
        return prefix, first

    @classmethod
    def from_levels(cls, topology: Topology, read, write, g_write, read_mass=None, *, sparse=None) -> RouteFactors:
        """THE FRAMEWORK'S EPILOGUE as a function: floating levels in, the format out -- u8 digits, each row
        against its own max (`rola.routing.quant`), and the ``sparse`` flags (:meth:`from_rows`)."""
        read, write = tuple(read), tuple(write)
        amp_read, scale_read = zip(*(quant.quantize_unit_max_rows(level) for level in read))
        amp_write, scale_write = zip(*(quant.quantize_unit_max_rows(level) for level in write))
        source = (read, write) if _requires_grad(read + write) else None
        return cls.from_rows(topology, amp_read, amp_write, scale_read, scale_write, g_write, read_mass=read_mass,
                             source=source, sparse=sparse)

    @classmethod
    def from_rows(cls, topology: Topology, amp_read, amp_write, scale_read, scale_write, g_write, read_mass=None,
                  source=None, *, sparse=None) -> RouteFactors:
        """The bundle over rows already in the format (unit-max u8, ``[B, T, H, width_l]`` a level a side).
        ``sparse`` is ``(read, write)``, a bool a level; by default the production's own rule, a side whose
        activation has exact zeros (an opaque level, which may have zeros, included)."""
        if sparse is None:
            sparse = tuple(tuple(topology.activation_properties(level, side).exact_zeros
                                 for level in range(topology.D)) for side in SIDES)
        return cls(topology=topology, amp_read=amp_read, amp_write=amp_write, scale_read=scale_read,
                   scale_write=scale_write, g_write=g_write, read_mass=read_mass, source=source, sparse=sparse)

    @property
    def batch(self) -> int:
        return self.amp_read[0].shape[0]

    @property
    def tokens(self) -> int:
        return self.amp_read[0].shape[1]

    @property
    def heads(self) -> int:
        return self.amp_read[0].shape[2]

    @property
    def device(self) -> torch.device:
        return self.amp_read[0].device

    @property
    def read_rows(self) -> tuple[torch.Tensor, ...]:
        """Every read level as rows (:meth:`rows`), for the references."""
        return tuple(self.rows(level, "read") for level in range(self.topology.D))

    @property
    def write_rows(self) -> tuple[torch.Tensor, ...]:
        """Every write level as rows (:meth:`rows`), for the references."""
        return tuple(self.rows(level, "write") for level in range(self.topology.D))

    def rows(self, level: int, side: str) -> torch.Tensor:
        """``level``'s rows ``[B, T, H, width_l]`` on ``side``."""
        return getattr(self, f"amp_{side}")[level]

    def dequantized(self) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """``(read, write)``: the levels as fp32 values, ``rows x scale``, computed at the call. With a ``source``
        the producer's gradient steps straight through the rounding. A consumer still computing on floating
        levels calls this at its seam, and the call goes when that consumer reads the rows."""
        sides = []
        for side, name in enumerate(SIDES):
            scales = getattr(self, f"scale_{name}")
            levels = tuple(quant.dequantize_rows(self.rows(level, name), scale) for level, scale in enumerate(scales))
            if self.source is not None:
                levels = tuple(quant.straight_through(exact, level) for exact, level in zip(self.source[side], levels))
            sides.append(levels)
        return sides[0], sides[1]

    def read_needs_normalization(self) -> tuple[bool, ...]:
        """Per read level: is it still off the simplex? THE PREDICATE, stated once.

        :meth:`read_simplex` divides exactly these levels and the decode fold normalizes
        exactly these levels inside its kernel; two sites folding one question from the
        same facts is one divergence waiting to happen.
        """
        if self.read_mass is None:
            return (False,) * self.topology.D
        return tuple(not self.topology.duty_simplex_output(index, "read")
                     for index in range(self.topology.D))

    def read_simplex(self) -> tuple[torch.Tensor, ...]:
        """The dequantized read levels on the simplex -- :attr:`read_mass` divided out PER LEVEL (a simplex is a
        per-level claim; the levels the topology already declares on it are returned untouched)."""
        read, _ = self.dequantized()
        if self.read_mass is None:
            return read
        return tuple(level / level.sum(dim=-1, keepdim=True) if needs else level
                     for level, needs in zip(read, self.read_needs_normalization(), strict=True))

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Every tensor the OPERATION consumes, for the gradient-reachability predicate."""
        mass = () if self.read_mass is None else (self.read_mass,)
        source = () if self.source is None else (*self.source[0], *self.source[1])
        return (*self.amp_read, *self.amp_write, *self.scale_read, *self.scale_write, self.g_write, *mass, *source)
