"""Ops-internal FSLA routing implementation.

This package is intentionally not a root ``fsla.ops`` export surface.
The canonical routing types are re-exported here for routing internals.
"""

from fsla.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    LevelRouting,
    ResolvedRouting,
    RoutingActivation,
    SoftmaxActivation,
    TiedRouting,
)

__all__ = [
    "EntmaxActivation",
    "IndependentRouting",
    "LevelRouting",
    "ResolvedRouting",
    "RoutingActivation",
    "SoftmaxActivation",
    "TiedRouting",
]
