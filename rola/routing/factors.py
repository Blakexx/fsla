# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The route gateway: producers in, one validated :class:`RouteFactors` out.

Linear attention's op takes ``(q, k, v)``. RoLA's takes ``(routes, v)``, where
``routes`` is the ``(q, k)`` pair AFTER the feature map -- the read-side and write-side
simplices and their side gains. That bundle is :class:`RouteFactors`;
:class:`RouteGateway` is the module that assembles one out of producers.

The bundle carries tensors and nothing else, which is why it is constructible rather
than sealed, and why its invariants are structural rather than provenance. What they
are and why unit-sum-ness is not among them: ``docs/api.md`` §2.1.

**One call per producer, and the gateway owns packing.** Each producer is called once
with every level it covers and returns amplitudes in the plain documented layout; the
gateway places them in the slots the list assigned and derives everything downstream --
the fold into the kernel's view and the side gains. Batching across levels is a
PRODUCER's business: :class:`~rola.routing.producer.SimplexProducer` covers as many
levels as it is given and runs them through one projection and one solve. Batching
across two DIFFERENT producers is not built; the prize is launch overhead at ``D <= 4``
and nobody has measured it to matter.

**The mass fold, and the one modelling consequence it carries.** A producer's levels
need not be unit-sum. The gateway factors each level into ``(mass, simplex)``, stores
the simplex, and folds the mass product into that side's gain: multiplicative per-level
masses collapse into one side-wide product. The identity is
loss-free for the readout. It is not neutral for the recurrence: the decay clock is
defined on the STORED (normalized) level, so a producer whose write-side mass differs
from 1 has moved magnitude out of the clock and into ``g_write``. That is the defined
semantics, not a defect, and it is stated here because a producer author choosing an
unnormalized write map is choosing it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from rola.routing.producer import RouteSource
from rola.routing.side_gain import GainConfig, SideGain
from rola.routing.types import (
    LevelConfig,
    OpaqueRouting,
    Topology,
    validate_normalization,
)

__all__ = ["RouteFactors", "RouteGateway"]

@dataclass(frozen=True, slots=True)
class RouteFactors:
    """The feature-mapped routing sides and their side gains. Tensors, and nothing else.

    ``read``/``write``: one ``[B, T, H, width_l]`` simplex per topology level.
    ``g_read``/``g_write``: ``[B, T, H]`` side gains; ``g_read`` is ``None`` under
    ``normalization='global'``, where the readout's ratio form fixes it to 1.

    Direct construction is legal; ``__post_init__`` holds it to the same invariants a
    derived bundle satisfies (``docs/api.md`` §2.1).
    """

    topology: Topology
    normalization: str
    read: tuple[torch.Tensor, ...]
    write: tuple[torch.Tensor, ...]
    g_read: torch.Tensor | None
    g_write: torch.Tensor

    def __post_init__(self) -> None:
        """Shapes, devices, dtypes and ``None``-ness only -- never a device reduction."""
        object.__setattr__(self, "normalization", validate_normalization(self.normalization))
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
        if self.g_read is not None:
            raise ValueError(
                "RouteFactors.g_read must be None: the readout is a ratio, so g_read := 1 "
                "under the self-normalizing readout, so there is no read-gain column "
                "and a supplied one would be silently ignored")
        for name, gain in (("g_read", self.g_read), ("g_write", self.g_write)):
            if gain is None:
                continue
            if not isinstance(gain, torch.Tensor) or tuple(gain.shape) != prefix:
                shape = getattr(gain, "shape", type(gain).__name__)
                raise ValueError(
                    f"RouteFactors.{name} must be a [B, T, H]={prefix} tensor, got {shape}")

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

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Every tensor the OPERATION consumes, for the gradient-reachability predicate."""
        gains = (self.g_write,) if self.g_read is None else (self.g_read, self.g_write)
        return (*self.read, *self.write, *gains)


class RouteGateway(nn.Module):
    """The gateway from producers to :class:`RouteFactors`.

    Args:
        producers: one :class:`~rola.routing.producer.LevelProducer` per routing
            level, outermost level first. A producer covering several levels occupies
            that many consecutive slots.
        normalization: ``'global'`` or ``'raw'`` -- the recurrence's normalization arm.
        gain: whether the side gains are learned projections. ``False`` removes the
            columns rather than pinning them to 1, so no dead parameter is optimized.
        gain_bias_init: initialize the gain bias so an all-zero-logit gain starts at
            this value. ``None`` leaves it at zero (unit gain at zero logits).
        gain_config: the two-boolean research surface behind ``gain``. It is reachable
            from :mod:`rola.expert` and is deliberately not part of the contract.

    Construction validates the whole assembly -- widths in range, a buildable topology,
    a gain layout consistent with the normalization arm -- so an unbuildable model
    fails here and never inside a forward.
    """

    def __init__(
        self,
        producers,
        *,
        num_heads: int,
        hidden_size: int,
        normalization: str = "global",
        gain: bool = True,
        gain_bias_init: float | None = None,
        gain_config: GainConfig | None = None,
    ) -> None:
        super().__init__()
        producers = list(producers)
        if not producers:
            raise ValueError("RouteGateway needs at least one producer -- one per routing level")
        for index, producer in enumerate(producers):
            widths = getattr(producer, "widths", None)
            if not isinstance(widths, tuple) or not widths or any(
                    type(width) is not int for width in widths):
                raise TypeError(
                    f"producers[{index}] ({type(producer).__name__}) must declare "
                    "`widths`, a nonempty tuple of exact ints -- one branch width per "
                    f"level it covers. Got {widths!r}.")
            if not callable(producer):
                raise TypeError(f"producers[{index}] ({type(producer).__name__}) is not callable")

        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.normalization = validate_normalization(normalization)
        self.producers = nn.ModuleList(producers)
        #: ``slots[p][i]`` is the routing level producer ``p``'s level ``i`` fills.
        self.slots = _slots(producers)
        widths = _scattered_widths(producers, self.slots)
        self.topology = Topology(
            levels=tuple(LevelConfig(width, routing)
                         for width, routing in zip(widths, _scattered_tags(producers, self.slots))))
        self.side_gain = SideGain(
            num_heads=self.num_heads, hidden_size=self.hidden_size,
            config=(gain_config if gain_config is not None
                    else GainConfig.from_public(gain)),
            bias_init=gain_bias_init)

    @property
    def widths(self) -> tuple[int, ...]:
        return self.topology.widths

    def forward(
        self,
        read_stream: torch.Tensor,
        write_stream: torch.Tensor | None = None,
    ) -> RouteFactors:
        """Run every producer once and derive the bundle.

        ``read_stream``/``write_stream`` are ``[B, T, hidden_size]``; passing only one
        means both duties read the same stream, and producers can see that (the two
        fields are then the same object).
        """
        if not isinstance(read_stream, torch.Tensor) or read_stream.ndim != 3:
            raise ValueError(
                "read_stream must be [B, T, hidden_size], got "
                f"{getattr(read_stream, 'shape', type(read_stream).__name__)}")
        if read_stream.shape[-1] != self.hidden_size:
            raise ValueError(
                f"read_stream is {read_stream.shape[-1]} wide but this RouteGateway was built "
                f"for hidden_size={self.hidden_size}")
        if write_stream is None:
            write_stream = read_stream
        elif write_stream.shape != read_stream.shape:
            raise ValueError("write_stream must have the same shape as read_stream")
        source = RouteSource(read=read_stream, write=write_stream)

        D = self.topology.D
        read: list[torch.Tensor | None] = [None] * D
        write: list[torch.Tensor | None] = [None] * D
        for producer, slots in zip(self.producers, self.slots):
            try:
                produced = producer(source)
            except Exception as error:  # noqa: BLE001 - re-raised, never swallowed
                # ATTRIBUTION: a producer covering several levels fails as one unit, and
                # a message naming only that unit leaves the reader to work out which
                # level's data was involved. The gateway holds the level map, so the
                # gateway is what can say.
                raise type(error)(
                    f"{type(producer).__name__} failed while producing routing level(s) "
                    f"{list(slots)}: {error}") from error
            _validate_produced(producer, produced, slots, self.topology, source, self.num_heads)
            for local, level in enumerate(slots):
                read[level] = produced.read[local]
                write[level] = produced.write[local]

        B, H = source.batch, self.num_heads
        #: THE READ MASS IS DROPPED, and that is the self-normalizing readout's own
        #: statement: `R` enters as num/den, so a per-token scale common to every leaf
        #: cancels exactly. The WRITE mass rides the write channel,
        #: where it scales the deposit.
        read_levels, _read_mass = _fold(tuple(read), B=B, H=H)
        write_levels, write_mass = _fold(tuple(write), B=B, H=H)
        gains = self.side_gain(read_stream)

        return RouteFactors(
            topology=self.topology, normalization=self.normalization,
            read=read_levels, write=write_levels, g_read=None,
            g_write=gains["g_write"] * write_mass)


# --- construction-time assembly ---------------------------------------------


def _slots(producers) -> list[tuple[int, ...]]:
    """The routing levels each producer fills, assigned from the ORDER of the list.

    A producer covering ``k`` levels takes the next ``k`` slots. That is the whole
    mapping: the list IS the topology's level order, outermost first.
    """
    slots: list[tuple[int, ...]] = []
    cursor = 0
    for producer in producers:
        slots.append(tuple(range(cursor, cursor + len(producer.widths))))
        cursor += len(producer.widths)
    return slots


def _scattered_widths(producers, slots) -> tuple[int, ...]:
    widths: dict[int, int] = {}
    for group, span in zip(producers, slots):
        for local, level in enumerate(span):
            widths[level] = group.widths[local]
    return tuple(widths[level] for level in sorted(widths))


def _scattered_tags(producers, slots) -> tuple:
    """The routing tag each level is produced under, or an opaque one.

    A producer that names activations from the closed registry may expose them as
    ``level_routing``; the gateway reads them so the topology can answer the
    activation questions the dispatch asks (``unratified_activations``). A producer
    that does not is opaque, which costs it nothing but that answer.
    """
    tags: dict[int, object] = {}
    for group, span in zip(producers, slots):
        declared = getattr(group, "level_routing", None) or getattr(group, "levels", None)
        for local, level in enumerate(span):
            tags[level] = (declared[local] if isinstance(declared, tuple) and local < len(declared)
                           else OpaqueRouting())
    return tuple(tags[level] for level in sorted(tags))


def _fold_bth(tensor: torch.Tensor) -> torch.Tensor:
    """``[B, T, H, *] -> [B*H, T, *]`` (the convention every routing module uses).

    Held here rather than imported: the routing side must not import a consumer, and
    three lines of permute arithmetic with a gate on each side is the cheaper of the
    two couplings.
    """
    return tensor.permute(0, 2, 1, 3).reshape(
        tensor.shape[0] * tensor.shape[2], tensor.shape[1], *tensor.shape[3:])


# --- per-call derivation ------------------------------------------------------


def _validate_produced(group, produced, slots, topology, source, heads) -> None:
    """Hard-fail with the caller's vocabulary if a producer's output is not well formed."""
    name = type(group).__name__
    read = getattr(produced, "read", None)
    write = getattr(produced, "write", None)
    if not isinstance(read, tuple) or not isinstance(write, tuple):
        raise TypeError(
            f"{name}.forward must return a ProducedRoutes whose `read` and `write` are tuples "
            f"of one tensor per covered level; got {type(produced).__name__}")
    if len(read) != len(slots) or len(write) != len(slots):
        raise ValueError(
            f"{name} covers {len(slots)} routing level(s) but returned {len(read)} read and "
            f"{len(write)} write tensors")
    expected_batch, expected_tokens = source.batch, source.tokens
    for local, level in enumerate(slots):
        width = topology.widths[level]
        for side, tensor in (("read", read[local]), ("write", write[local])):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name}'s {side} factor for level {level} is not a tensor")
            shape = (expected_batch, expected_tokens, heads, width)
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"{name}'s {side} factor for level {level} must be [B, T, H, width_l]="
                    f"{shape}, got {tuple(tensor.shape)}")
            if tensor.device != source.read.device:
                raise ValueError(
                    f"{name}'s {side} factor for level {level} is on {tensor.device} but the "
                    f"source stream is on {source.read.device}")
            if tensor.dtype != read[0].dtype:
                raise ValueError(
                    f"{name}'s factors must share one dtype; level {level}'s {side} side is "
                    f"{tensor.dtype} against {read[0].dtype}")


def _fold(levels: tuple[torch.Tensor, ...], *, B: int, H: int):
    """Factor every level into ``(simplex, mass)`` and return the mass product.

    The stored level is the simplex; the magnitude leaves for the side gain. See the
    module docstring for the one consequence this carries for the decay clock.

    **PACKING LIVES HERE, and the normalization is done in the packed domain.** The
    kernel's view is ``[BH, T, width]``; the public layout is ``[B, T, H, width]``. Those
    differ by a permute, so exactly one of them can be the materialized one, and it must
    be the kernel's -- normalizing in the public layout would produce a tensor the fold
    then has to copy, which is the inverse-copy pair this deliberately does not create.
    So: fold (free, the producer returned a view), divide (one pass, materializing the
    packed tensor), and hand back a permuted view for the public layout.
    """
    normalized: list[torch.Tensor] = []
    mass_product: torch.Tensor | None = None
    for level in levels:
        folded = _fold_bth(level)
        mass = folded.sum(dim=-1)
        packed = folded / mass.unsqueeze(-1)
        width = packed.shape[-1]
        normalized.append(packed.reshape(B, H, -1, width).permute(0, 2, 1, 3))
        mass_product = mass if mass_product is None else mass_product * mass
    return tuple(normalized), mass_product.reshape(B, H, -1).permute(0, 2, 1)
