# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""One call's facts, read off the routing bundle and the value stream.

The rules fold VALUES -- that is what makes them testable with no device present --
so something has to read those values off the objects a caller actually holds. This
is that adapter and nothing else: it finds facts and applies one rule each, and it
decides nothing of its own (docs/internals/engine/engine.md).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rola.engine import rules
from rola.engine.facts import manifest

if TYPE_CHECKING:
    from rola.engine import Arm


def widths_of(routes) -> tuple[int, ...]:
    """The per-level widths of this bundle's topology, in level order."""
    return tuple(level.width for level in routes.topology.levels)


def arm_of(routes) -> Arm | None:
    """`(C, BC)` -- the built arm this routing runs on, or None if it has none.

    The facade needs `BC` before the launch, because the page arena's extent policy
    is stated in owners; it is the one launch parameter that leaves the plan.
    """
    return rules.arm(widths_of(routes), manifest.chunk_arms())


def envelope_clauses(routes, decay: Any = None, *, d_v: int) -> tuple:
    """The facts :func:`rola.engine.rules.envelope.arm_envelope` folds, for this call.

    `d_v` is the value stream's per-head width, which is a fact about the operand and
    not about the routing -- the caller reads it off `v` and states it here. The arm
    is the ArmRule's own verdict: the envelope refuses the ABSENCE of an arm and
    never re-derives one.
    """
    widths = widths_of(routes)
    return (widths, d_v, routes.normalization, decay, routes.tokens,
            rules.arm(widths, manifest.chunk_arms()))


def arm_refusal(routes, decay: Any = None, *, d_v: int) -> str | None:
    """Why the chunk arm refuses this call, or None if it runs."""
    return rules.arm_envelope(*envelope_clauses(routes, decay, d_v=d_v))


def backward_refusal(routes, decay: Any = None, *, d_v: int) -> str | None:
    """Why a GRAD-BEARING call has no arm, or None if it trains."""
    clauses = envelope_clauses(routes, decay, d_v=d_v)
    return rules.backward_envelope(rules.arm_envelope(*clauses), clauses[0],
                                   clauses[-1], manifest.scan_arms())
