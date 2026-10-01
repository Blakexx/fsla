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
from rola.routing.sparse import SparseLevel
from rola.routing.types import Topology

__all__ = ["U8_MAX_DEPTH", "RouteFactors"]

SIDES = ("read", "write")

#: u8 coefficient rows serve depth <= 2; a deeper topology's rows are f16 (ruled 2026-09-29).
U8_MAX_DEPTH = 2


def _requires_grad(levels) -> bool:
    return torch.is_grad_enabled() and any(level.requires_grad for level in levels)


@dataclass(frozen=True, slots=True)
class RouteFactors:
    """The feature-mapped routing sides in THE INTERNAL FORMAT, and their side gain. Tensors, and nothing else.

    ``amp_read``/``amp_write``: one entry per level, each row UNIT-MAX -- u8 with its largest digit at 255
    (depth <= 2), or f16 with its largest entry at 1 (any depth); one dtype for the whole bundle. An entry is
    ``[B, T, H, width_l]`` rows (dense: every digit is read) or a :class:`~rola.routing.sparse.SparseLevel` (its
    bitlist and its live digits in bit order): the entry's type is the whole fact, chosen at production. A
    one-token bundle is all rows. A zero digit is dead. ``scale_read``/``scale_write``: the rows' maxima,
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

    def __post_init__(self) -> None:
        """Shapes, devices, dtypes and ``None``-ness only -- never a device reduction."""
        for name in ("amp_read", "amp_write", "scale_read", "scale_write"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if type(self.topology) is not Topology:
            raise TypeError(
                f"RouteFactors.topology must be a Topology, got {type(self.topology).__name__}")
        D = self.topology.D
        prefix = self._check_scales()
        dtype = self._check_rows(prefix)
        if dtype not in quant.ROW_DTYPES:
            raise TypeError(f"RouteFactors rows are stored as one of {quant.ROW_DTYPES}, got {dtype}")
        if dtype is torch.uint8 and D > U8_MAX_DEPTH:
            raise ValueError(f"u8 rows serve depth <= {U8_MAX_DEPTH}; a depth-{D} topology's rows are f16")
        for name, column in (("g_write", self.g_write), ("read_mass", self.read_mass)):
            if column is None and name == "read_mass":
                continue
            if not isinstance(column, torch.Tensor) or tuple(column.shape) != prefix:
                shape = getattr(column, "shape", type(column).__name__)
                raise ValueError(f"RouteFactors.{name} must be a [B, T, H]={prefix} tensor, got {shape}")
        if self.source is not None:
            for side, levels in zip(SIDES, self.source):
                if len(levels) != D or any(tuple(a.shape) != prefix + (w,)
                                           for a, w in zip(levels, self.topology.widths)):
                    raise ValueError(f"RouteFactors.source's {side} levels are the rows' floating originals")

    def sparse(self, level: int, side: str) -> bool:
        """Is ``level`` stored as a :class:`SparseLevel` on ``side``? The entry says so."""
        return isinstance(getattr(self, f"amp_{side}")[level], SparseLevel)

    def _check_scales(self) -> tuple[int, ...]:
        """The scales: fp32 ``[B, T, H]`` a level a side, one prefix and device -- the bundle's ``[B, T, H]``."""
        D = self.topology.D
        prefix, device = None, None
        for name in ("scale_read", "scale_write"):
            scales = getattr(self, name)
            if len(scales) != D:
                raise ValueError(f"RouteFactors.{name} holds one tensor per level (D={D}), got {len(scales)}")
            for level, scale in enumerate(scales):
                if not isinstance(scale, torch.Tensor) or scale.dtype is not torch.float32 or scale.ndim != 3:
                    raise ValueError(f"RouteFactors.{name}[{level}] is the rows' maxima, fp32 [B, T, H]; got "
                                     f"{getattr(scale, 'dtype', type(scale).__name__)} "
                                     f"{tuple(getattr(scale, 'shape', ()))}")
                if prefix is None:
                    prefix, device = tuple(scale.shape), scale.device
                elif tuple(scale.shape) != prefix or scale.device != device:
                    raise ValueError(f"every scale shares one [B, T, H] prefix and device; {name}[{level}] is "
                                     f"{tuple(scale.shape)} on {scale.device} against {prefix} on {device}")
        return prefix

    def _check_rows(self, prefix: tuple[int, ...]) -> torch.dtype:
        """Every level's entry: rows ``[B, T, H, width_l]`` or, over more than one token, a :class:`SparseLevel`; the
        bundle's prefix, device and one dtype. Returns the dtype."""
        D, widths = self.topology.D, self.topology.widths
        B, T, H = prefix
        device = self.scale_read[0].device
        dtype = None
        for side in SIDES:
            name = f"amp_{side}"
            entries = getattr(self, name)
            if len(entries) != D:
                raise ValueError(f"RouteFactors.{name} must hold one entry per routing level (D={D}), got {len(entries)}")
            for level, entry in enumerate(entries):
                where = f"RouteFactors.{name}[{level}]"
                if isinstance(entry, SparseLevel):
                    if T == 1:
                        raise ValueError(f"{where}: a one-token bundle holds every level as rows")
                    if entry.terms.shape[:2] != (B, H) or entry.width != widths[level]:
                        raise ValueError(f"{where} is [B, H, width]={(B, H, widths[level])}; got words "
                                         f"{tuple(entry.words.shape)}, terms {tuple(entry.terms.shape)}")
                    if entry.words.shape[3] * 32 < T:
                        raise ValueError(f"{where}'s words cover {entry.words.shape[3] * 32} tokens, not T = {T}")
                    tensor = entry.terms
                else:
                    if not isinstance(entry, torch.Tensor) or tuple(entry.shape) != prefix + (widths[level],):
                        shape = getattr(entry, "shape", type(entry).__name__)
                        raise TypeError(f"{where} is rows [B, T, H, width_l]={prefix + (widths[level],)}, got {shape}")
                    tensor = entry
                if tensor.device != device:
                    raise ValueError(f"every routing level lives on one device; {where} is on {tensor.device} "
                                     f"against {device}")
                if dtype is None:
                    dtype = tensor.dtype
                elif tensor.dtype != dtype:
                    raise ValueError(f"every routing level shares one row dtype; {where} is {tensor.dtype} "
                                     f"against {dtype}")
        return dtype

    @classmethod
    def from_levels(cls, topology: Topology, read, write, g_write, read_mass=None, *, sparse=None) -> RouteFactors:
        """THE FRAMEWORK'S EPILOGUE as a function: floating levels in, the format out -- u8 digits at depth <= 2 and
        f16 deeper, each row against its own max (`rola.routing.quant`), each side ``sparse`` names packed
        (:meth:`from_rows`)."""
        dtype = torch.uint8 if topology.D <= U8_MAX_DEPTH else torch.float16
        read, write = tuple(read), tuple(write)
        amp_read, scale_read = zip(*(quant.quantize_unit_max_rows(level, dtype) for level in read))
        amp_write, scale_write = zip(*(quant.quantize_unit_max_rows(level, dtype) for level in write))
        source = (read, write) if _requires_grad(read + write) else None
        return cls.from_rows(topology, amp_read, amp_write, scale_read, scale_write, g_write, read_mass=read_mass,
                             source=source, sparse=sparse)

    @classmethod
    def from_rows(cls, topology: Topology, amp_read, amp_write, scale_read, scale_write, g_write, read_mass=None,
                  source=None, *, sparse=None) -> RouteFactors:
        """The bundle over rows already in the format (unit-max u8 or f16, ``[B, T, H, width_l]`` a level a side), each
        side ``sparse`` names packed (:meth:`SparseLevel.from_rows`), every other kept as rows. ``sparse`` is
        ``(read, write)``, a bool a level each; by default a side whose activation has exact zeros. A one-token
        bundle is all rows whatever ``sparse`` says."""
        if sparse is None:
            sparse = tuple(tuple(topology.activation_properties(level, side).exact_zeros
                                 for level in range(topology.D)) for side in SIDES)
        one_token = scale_read[0].shape[1] == 1
        amp_read, amp_write = (
            tuple(SparseLevel.from_rows(rows) if flag and not one_token else rows
                  for rows, flag in zip(entries, flags, strict=True))
            for entries, flags in zip((amp_read, amp_write), sparse))
        return cls(topology=topology, amp_read=amp_read, amp_write=amp_write, scale_read=scale_read,
                   scale_write=scale_write, g_write=g_write, read_mass=read_mass, source=source)

    @property
    def batch(self) -> int:
        return self.scale_read[0].shape[0]

    @property
    def tokens(self) -> int:
        return self.scale_read[0].shape[1]

    @property
    def heads(self) -> int:
        return self.scale_read[0].shape[2]

    @property
    def device(self) -> torch.device:
        return self.scale_read[0].device

    def refuse_sparse(self, reader: str, port: str) -> None:
        """RAISE if a level is a :class:`SparseLevel`: ``reader`` does not read the sparse format until ``port``
        lands (interface first: an unported reader refuses by name, never guesses at rows)."""
        if any(isinstance(entry, SparseLevel) for entry in (*self.amp_read, *self.amp_write)):
            raise NotImplementedError(f"{reader} does not read a sparse level's native format until {port} "
                                      f"(docs/internals/routing/sparse.md)")

    @property
    def read_rows(self) -> tuple[torch.Tensor, ...]:
        """Every read level as rows (:meth:`rows`), for the references."""
        return tuple(self.rows(level, "read") for level in range(self.topology.D))

    @property
    def write_rows(self) -> tuple[torch.Tensor, ...]:
        """Every write level as rows (:meth:`rows`), for the references."""
        return tuple(self.rows(level, "write") for level in range(self.topology.D))

    def rows(self, level: int, side: str) -> torch.Tensor:
        """``level``'s rows ``[B, T, H, width_l]`` on ``side``: a dense level's own, a sparse level's
        :meth:`SparseLevel.dense` -- for the oracle and the references, never a kernel path."""
        entry = getattr(self, f"amp_{side}")[level]
        return entry.dense(self.tokens) if isinstance(entry, SparseLevel) else entry

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
        stored = tuple(t for entry in (*self.amp_read, *self.amp_write)
                       for t in ((entry.words, entry.terms) if isinstance(entry, SparseLevel) else (entry,)))
        return (*stored, *self.scale_read, *self.scale_write, self.g_write, *mass, *source)
