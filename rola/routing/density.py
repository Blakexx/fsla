# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The producer's declared support density, per ``(level, side)``.

A fact about the CONFIGURATION -- which activation each routing cell was built
with -- and therefore a routing-side object rather than a launch dial: it is read
from ``ACTIVATION_REGISTRY``'s ``exact_zeros`` column through
:meth:`~rola.routing.types.Topology.activation_properties`, and its readers are
:class:`~rola.routing.factors.RouteFactors` and the summary pass.
"""

from __future__ import annotations

from dataclasses import dataclass

from rola.routing.types import Topology


@dataclass(frozen=True, slots=True)
class StaticDensity:
    """Which ``(level, side)`` cells are FULL-SUPPORT BY CONSTRUCTION (§2.7 ruling).

    One boolean per level per side: ``True`` means the configured ACTIVATION for
    that cell cannot produce an exact structural zero, so every digit of that level
    is live at every token in every realization. That is a fact about the
    configuration, available at model build, and it is read from the ONE place
    activation properties live -- ``ACTIVATION_REGISTRY``'s ``exact_zeros`` column,
    via :meth:`Topology.activation_properties` -- never from a name list and never
    from a measurement of the data (`rola/routing/activations.py`: *"No case list
    over activations may exist anywhere outside this module"*).

    **Why this is an INPUT rather than something derived from the ``topology`` it
    already holds.** The topology states what the model's producer will emit; a
    measurement is handed tensors. Those agree in production (the producer applies
    exactly the configured activation), and they deliberately do NOT agree in
    synthetic fixtures, where a support constructor imposes a density the declared
    activation does not describe. Deriving the closed form from the tag alone would therefore price a
    density those callers do not have. Passing it makes the claim the CALLER's --
    :class:`~rola.layers.RoLA` builds one from its own topology once, at
    construction -- and leaves every caller that does not pass one on the fully
    data-dependent pass, bit for bit.

    **What a wrong declaration costs, stated exactly.** Only the COST MODEL: what
    the kernel executes is derived from the realized masks and never reads this
    object, so a side declared dense whose realization has zeros can only mis-price
    a measurement, never mis-execute a launch. The
    one realistic way that happens is float underflow -- a softmax logit far
    enough below the row max that ``exp`` rounds to exactly ``0`` -- which shrinks
    the realized support without changing the activation's mathematics; the closed
    form then over-counts ``K`` by the underflowed fraction.

    **Measured, not hypothetical (Gate 2, 2026-08-01, item 3e):** a parity-shaped
    ``RoLA`` layer (``hidden=256``, ``H=4``, ``d_v=64``, ``branches=(64,64)``,
    ``BC=64``, ``read="dense"``) at ``L=2048``, ``seed=11``, with ``route_W``
    scaled by a ``logit_gain`` lever, stays true through ``logit_gain in {1.0,
    8.0, 16.0}`` (0.0 realized zeros on the declared-dense read side at every
    level -- the repo's own bench default is ``logit_gain=8.0``, so it is safe)
    and goes false at ``logit_gain=32.0``: realized read-side zero fractions of
    ``0.002056`` and ``0.001883`` across the two levels. At that cell the
    substituted (declared) vs. measured (data-dependent) :class:`TileUnionStats`
    differ by ~2e-4 relative on ``mean_KI_candidate`` (0.4067078 vs 0.4066162)
    and ``mean_kisteps_candidate`` (0.2754822 vs 0.2754517); ``cost_per_token``
    was UNCHANGED at this cell, i.e. the mispricing moved no decision here,
    consistent with "cost model only, never mis-execute" above. There is no known
    upper bound on how large a gain would be needed to move one; this is one
    measured point, not
    a proof the mispricing is always inert. A caller that scales router logits
    by a large factor (an aggressive temperature/gain schedule, an
    ill-conditioned ``route_W`` init) should re-check this assumption for its
    own topology rather than assume the bench's ``logit_gain=8.0`` headroom
    transfers.
    """

    read: tuple[bool, ...]
    write: tuple[bool, ...]

    @classmethod
    def from_topology(cls, topology: Topology) -> StaticDensity:
        """Read the per-``(level, duty)`` ``exact_zeros`` column of the registry."""
        return cls(*(tuple(not topology.activation_properties(level, duty).exact_zeros
                           for level in range(topology.D))
                     for duty in ("read", "write")))

    def side_dense(self, duty: str) -> bool:
        """Is the WHOLE side full-support -- the condition the closed form needs?

        A leaf is live at a token iff EVERY level's digit is live there
        (:func:`tile_union_stats`), so one dense level among sparse ones does not
        make the side's support the full rectangle and the closed form below does
        not apply. The per-LEVEL resolution is kept because §2.7's other planes
        (the producer's support machinery, the consumer's feeder specialization)
        are per level; the tile statistics are the one consumer of this object for
        which only the conjunction is actionable.
        """
        if duty == "read":
            return all(self.read)
        if duty == "write":
            return all(self.write)
        raise ValueError(f"duty must be 'read' or 'write', got {duty!r}")


__all__ = ["StaticDensity"]
