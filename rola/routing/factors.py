# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The route gateway: producers in, one validated :class:`RouteFactors` out.

Linear attention's op takes ``(q, k, v)``. RoLA's takes ``(routes, v)``, where
``routes`` is the ``(q, k)`` pair AFTER the feature map -- the read-side and write-side
simplices, their side gains, and the summary statistics the planner prices with. That
bundle is :class:`RouteFactors`, and the only way to obtain one is to call
:class:`RouteProducer`.

**Why the gateway is mandatory.** Every field of a ``RouteFactors`` is computed by the
derivation code in this module: the mass fold, the side gains, the support statistics,
the touched bits, the declared density. A caller therefore cannot hand the op a bundle
whose statistics disagree with its tensors, because a caller never authors the bundle
-- they choose producers, and producers return tensors. That is what makes the whole
surface trustless rather than merely documented: correctness is derived, never
declared. Construction validates shapes, layout and dtype and fails hard with the
public argument names in the message; :func:`rola.rola_op` type-gates on the result, so
no well-formed bundle means no call.

**One call per producer, and the gateway owns packing.** Each producer is called once
with every level it covers and returns amplitudes in the plain documented layout; the
gateway places them in the slots the list assigned and derives everything downstream --
the fold into the kernel's view, the masks, the statistics. Batching across levels is a
PRODUCER's business: :class:`~rola.routing.producer.SimplexProducer` covers as many
levels as it is given and runs them through one projection and one solve. Batching
across two DIFFERENT producers is not built; the prize is launch overhead at ``D <= 4``
and nobody has measured it to matter.

**The mass fold, and the one modelling consequence it carries.** A producer's levels
need not be unit-sum. The gateway factors each level into ``(mass, simplex)``, stores
the simplex, and folds the mass product into that side's gain -- architecture §4.2's
"multiplicative per-level masses collapse into one side-wide product". The identity is
loss-free for the readout. It is not neutral for the recurrence: the decay clock is
defined on the STORED (normalized) level, so a producer whose write-side mass differs
from 1 has moved magnitude out of the clock and into ``g_write``. That is the defined
semantics, not a defect, and it is stated here because a producer author choosing an
unnormalized write map is choosing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from rola.planner.schedule import StaticDensity, _fold_bth
from rola.routing.budget import exact_support_count, touched_bitset_update
from rola.routing.producer import RouteSource
from rola.routing.side_gain import GainConfig, init_side_gain_bias, project_side_gains
from rola.routing.types import (
    LevelConfig,
    OpaqueRouting,
    Topology,
    validate_normalization,
)

__all__ = ["RouteFactors", "RouteProducer"]

#: Handed to :class:`RouteFactors` by the derivation code below and by nothing else.
#: The bundle is not a struct users fill in; it is the record of what our code derived.
_DERIVED = object()


@dataclass(frozen=True, slots=True)
class RouteFactors:
    """The feature-mapped routing sides, and everything derived from them.

    ``read``/``write``: one ``[B, T, H, width_l]`` unit-sum simplex per level.
    ``g_read``/``g_write``: ``[B, T, H]`` side gains; ``g_read`` is ``None`` under
    ``normalization='global'``, where architecture §4.2 fixes it to 1.
    ``support_counts()``/``touched_bits()``: the auxiliary statistics, DERIVED ON
    DEMAND. Nothing in the library reads them -- they exist for the budget losses and
    for diagnostics -- so computing them eagerly spent two launches per level per step
    on numbers no consumer asked for (measured at ``L = 4096``: 3.0% of the step). A
    producer that already knows them supplies them and no launch happens at all.
    ``slack_history``: ``{(level, side): [B, H, chunks]}``, budgeted pairs only. It is
    a field rather than a method because it is a byproduct of the conditioning
    recurrence and cannot be recovered afterwards.
    ``density``: which cells are full-support by construction, for PRICING only --
    the schedule the kernel executes is built from the realized masks and never reads
    it (:class:`~rola.planner.schedule.StaticDensity`).

    Obtain one from :class:`RouteProducer`. Constructing it directly is refused: a
    bundle whose fields did not come from the derivation code is a bundle whose
    statistics can disagree with its tensors, and every trustless property on this
    surface rests on that not being possible.
    """

    topology: Topology
    normalization: str
    read: tuple[torch.Tensor, ...]
    write: tuple[torch.Tensor, ...]
    g_read: torch.Tensor | None
    g_write: torch.Tensor
    density: StaticDensity
    exposed_counts: dict[tuple[int, str], torch.Tensor] = field(default_factory=dict)
    exposed_bits: dict[int, torch.Tensor] = field(default_factory=dict)
    slack_history: dict[tuple[int, str], torch.Tensor] = field(default_factory=dict)
    origin: object = None
    #: Derivations computed on demand and reused within the bundle's lifetime. A view
    #: lives inside one forward and cannot go stale.
    resident: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.origin is not _DERIVED:
            raise TypeError(
                "RouteFactors is built by RouteProducer, not by a caller. Its fields are "
                "derivations of the producers' tensors -- support bits, statistics and the "
                "mass fold -- and a hand-built bundle could carry statistics that do not "
                "describe its own tensors. Call `RouteProducer(...)(x)` to get one.")

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

    def support_counts(self) -> dict[tuple[int, str], torch.Tensor]:
        """``{(level, side): [B, T, H]}`` exact per-token nonzero counts.

        Producer-supplied where a producer knew them as a byproduct of its own solve,
        measured from the stored tensors otherwise. Both are exact counts of the same
        values: which one a cell takes is a placement decision, never a numerical one.
        """
        counts = self.resident.get("support_counts")
        if counts is None:
            counts = {}
            for level in range(len(self.read)):
                for side, levels in (("read", self.read), ("write", self.write)):
                    supplied = self.exposed_counts.get((level, side))
                    counts[(level, side)] = (supplied if supplied is not None
                                             else exact_support_count(levels[level]))
            self.resident["support_counts"] = counts
        return counts

    def touched_bits(self) -> dict[int, torch.Tensor]:
        """``{level: [B, H, width_l]}`` digits any write touched, over the sequence."""
        bits = self.resident.get("touched_bits")
        if bits is None:
            B, H = self.batch, self.heads
            bits = {
                level: self.exposed_bits.get(level) if level in self.exposed_bits else
                touched_bitset_update(
                    torch.zeros(B, H, self.topology.widths[level], dtype=torch.bool,
                                device=self.device),
                    self.write[level].permute(0, 2, 1, 3))
                for level in range(len(self.write))}
            self.resident["touched_bits"] = bits
        return bits

    def views(self):
        """The folded ``[BH, T, width]`` view and its exact-nonzero masks.

        Computed ONCE and shared by the summary pass and the schedule build, so the
        fold copy and the masks are paid once per bundle rather than once per consumer.
        """
        views = self.resident.get("views")
        if views is None:
            from rola.planner.schedule import routing_views

            views = routing_views(self.read, self.write, self.topology)
            self.resident["views"] = views
        return views

    def summaries(self, *, BC: int, BT: int):
        """The realized routing's summary statistics, at one stated granularity.

        This is the pass over the amplitude tensors -- the one bandwidth term that
        scales with tokens -- and it rides PRODUCTION, on the bundle, rather than the
        planner: what the planner then does is host-side arithmetic over numbers that
        are already resident.

        The granularity is stated rather than hidden. ``BT`` is the FINEST instantiated
        tile: ``candidate_frac`` and the quanta histogram are statistics OF A TILING and
        are not ``BT``-invariant, so a finer tile's value under-prices a coarser one,
        and §4.2 freezes the struct at one granularity instead of pretending otherwise.
        ``BC`` is the finest feasible owner rectangle, by the same symmetry -- the pass
        needs one owner geometry while selection reuses the histogram across the whole
        feasible grid.
        """
        key = ("summaries", BC, BT)
        summaries = self.resident.get(key)
        if summaries is None:
            from rola.planner.schedule import owner_geometry
            from rola.routing.summaries import summarize_routing

            views = self.views()
            summaries = summarize_routing(
                views.read_folded, views.write_folded,
                owner_geometry(tuple(self.topology.widths), BC), BT,
                # Every side the declaration marks full-support is asserted realized
                # under the debug env flag -- a producer-author courtesy, never a guard.
                density=self.density)
            self.resident[key] = summaries
        return summaries


class RouteProducer(nn.Module):
    """The mandatory gateway from producers to :class:`RouteFactors`.

    Args:
        producers: one :class:`~rola.routing.producer.LevelProducer` per routing
            level, outermost level first. A producer covering several levels occupies
            that many consecutive slots.
        d_v: per-head value width. It is a feature-map fact (the width of the state
            each leaf holds), so it lives with the topology rather than on the layer.
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
        d_v: int,
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
            raise ValueError("RouteProducer needs at least one producer -- one per routing level")
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
                         for width, routing in zip(widths, _scattered_tags(producers, self.slots))),
            d_v=int(d_v))
        self.density = _declared_density(self.topology, producers, self.slots)

        self.gain_config = (gain_config.validate(self.normalization) if gain_config is not None
                            else GainConfig.from_public(gain, self.normalization).validate(
                                self.normalization))
        self.gain_sides = self.gain_config.layout
        sides = len(self.gain_sides)
        self.gain_W = nn.Parameter(torch.empty(self.num_heads, sides, self.hidden_size))
        if sides:
            nn.init.kaiming_uniform_(
                self.gain_W.view(self.num_heads * sides, self.hidden_size), a=5 ** 0.5)
        bias_value = 0.0 if gain_bias_init is None else init_side_gain_bias(gain_bias_init)
        self.gain_bias = nn.Parameter(torch.full((self.num_heads, sides), bias_value))

    @property
    def d_v(self) -> int:
        return self.topology.d_v

    @property
    def widths(self) -> tuple[int, ...]:
        return self.topology.widths

    def forward(
        self,
        read_stream: torch.Tensor,
        write_stream: torch.Tensor | None = None,
        *,
        budget: torch.Tensor | None = None,
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
                f"read_stream is {read_stream.shape[-1]} wide but this RouteProducer was built "
                f"for hidden_size={self.hidden_size}")
        if write_stream is None:
            write_stream = read_stream
        elif write_stream.shape != read_stream.shape:
            raise ValueError("write_stream must have the same shape as read_stream")
        source = RouteSource(read=read_stream, write=write_stream, budget=budget)

        D = self.topology.D
        read: list[torch.Tensor | None] = [None] * D
        write: list[torch.Tensor | None] = [None] * D
        exposed: dict[tuple[int, str], torch.Tensor] = {}
        exposed_bits: dict[int, torch.Tensor] = {}
        slack: dict[tuple[int, str], torch.Tensor] = {}
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
            for (local, side), value in getattr(produced, "support_counts", {}).items():
                exposed[(slots[local], side)] = value
            for local, value in getattr(produced, "touched_bits", {}).items():
                exposed_bits[slots[local]] = value
            for (local, side), value in getattr(produced, "slack_history", {}).items():
                slack[(slots[local], side)] = value

        B, H = source.batch, self.num_heads
        read_levels, read_mass = _fold(tuple(read), B=B, H=H)
        write_levels, write_mass = _fold(tuple(write), B=B, H=H)
        gains = project_side_gains(
            read_stream, self.gain_W, self.gain_bias, normalization=self.normalization,
            write_gain=self.gain_config.write_gain, read_gain=self.gain_config.read_gain,
            projection_dtype=(torch.float64 if read_stream.dtype == torch.float64
                              else torch.float32))
        # The write mass always rides the write channel: W scales the deposit under both
        # normalizations. The read mass matters only under `raw` -- under `global`, R
        # enters as num/den and a per-token scale common to every leaf cancels exactly.
        g_write = gains["g_write"] * write_mass
        g_read = gains["g_read"]
        if self.normalization == "raw":
            g_read = g_read * read_mass

        return RouteFactors(
            topology=self.topology, normalization=self.normalization,
            read=read_levels, write=write_levels, g_read=g_read, g_write=g_write,
            density=self.density, exposed_counts=exposed, exposed_bits=exposed_bits,
            slack_history=slack, origin=_DERIVED)


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
    ``level_routing``; the gateway reads them for the PRICING columns only (declared
    density, kernel ratification). A producer that does not is opaque, which costs it
    the closed-form density substitution and nothing else -- its levels are measured
    like any other unknown-provenance tensor.
    """
    tags: dict[int, object] = {}
    for group, span in zip(producers, slots):
        declared = getattr(group, "level_routing", None) or getattr(group, "levels", None)
        for local, level in enumerate(span):
            tags[level] = (declared[local] if isinstance(declared, tuple) and local < len(declared)
                           else OpaqueRouting())
    return tuple(tags[level] for level in sorted(tags))


def _declared_density(topology: Topology, producers, slots) -> StaticDensity:
    """Registry-derived density, with a producer's own declaration where it has one.

    An opaque level defaults to MEASURED, bit for bit. A producer may override its own
    levels with ``dense_sides`` -- ``(read, write)`` per covered level -- which is
    trusted outright: over-approximating support is inert in the kernel (dead routes
    are exact zeros and zeros are inert in every accumulator, in the mass channel and
    in the clock), so a false declaration costs its author a plan's price and can never
    cost anyone a result.
    """
    read = list(StaticDensity.from_topology(topology).read)
    write = list(StaticDensity.from_topology(topology).write)
    for group, span in zip(producers, slots):
        declared = getattr(group, "dense_sides", None)
        if declared is None:
            continue
        for local, level in enumerate(span):
            cell = declared[local]
            read[level], write[level] = bool(cell[0]), bool(cell[1])
    return StaticDensity(read=tuple(read), write=tuple(write))


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
