"""Ops-internal RoLA routing implementation.

This package is intentionally not a root ``rola.ops`` export surface.
The canonical routing-plan types are re-exported here for routing internals.

The V2 prefix-run schedule ABI (``routing/schedule.py``) that used to be
re-exported here was deleted in the ``rola-v3`` purge: it was reachable only from
the V2 kernels, the V2 state cache and the V2 half of ``cuda/bindings.py``. V3's
plan lives in the planner package (``rola/planner/``).
"""

from rola.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    LevelRouting,
    ResolvedRouting,
    RouteDescriptor,
    RoutingActivation,
    SoftmaxActivation,
    UnionRouting,
)

__all__ = [
    "EntmaxActivation",
    "IndependentRouting",
    "LevelRouting",
    "ResolvedRouting",
    "RouteDescriptor",
    "RoutingActivation",
    "SoftmaxActivation",
    "UnionRouting",
]
