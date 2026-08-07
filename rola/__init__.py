# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""RoLA — Routed Linear Attention.

RoLA is linear attention under a structured feature map. A head's recurrent state is
expanded into many sub-states that share the head's projections, and a learned routing
decides which sub-states each token reads and writes. That reduction IS this API: a
user who knows linear attention meets exactly ONE new concept here, the feature map,
and everything else is linear attention's own contract.

    rola_op(routes, v, decay)

``routes`` is the ``(q, k)`` pair after the map -- a ``RouteFactors`` bundle -- ``v`` is
the value stream, ``decay`` is the recurrence dial. Build the bundle through the
gateway, whose list of producers is the feature map:

    producer = SimplexProducer((64, 64), sparse_routing(1.5),
                               hidden_size=1024, num_heads=8)
    routes = RouteProducer([producer], d_v=64, num_heads=8, hidden_size=1024)
    layer = RoLA(1024, 8, routes, decay=LearnedDecay(widths=routes.widths, num_heads=8))

**No public name says `v3`.** This repository IS that generation; encoding it in the
API would force a break at the next one for a reason that is not an API reason.

**What is deliberately NOT here.** Plans, schedules, cost models, tilings, numeric arms,
allocator caps -- every one of them is how the op RUNS rather than what the model IS,
and the op decides how it runs. They live in :mod:`rola.expert`, which is a marked door
and not part of the contract.

| Group | Names | Stability |
|---|---|---|
| op | `rola_op` | STABLE, semver-major to break |
| feature map | `RouteProducer`, `RouteFactors`, `LevelProducer`, `SimplexProducer`, `dense_routing`, `sparse_routing` | STABLE |
| recurrence dials | `ConstantDecay`, `LearnedDecay`, `DecaySource`, `LeafMassDecay` | STABLE |
| layer | `RoLA` | STABLE |
| state | `RoLAState`, `LayerRecurrentState` | STABLE |
| configuration | `Topology`, `LevelConfig`, `softmax()`, `entmax(alpha)`, `IndependentRouting`, `UnionRouting` | STABLE |
| oracle | `naive_rola` | STABLE signature, EXACT semantics — it is the definition |
"""

__version__ = "0.1.0.dev0"

from rola.interface import rola_op
from rola.layers.rola import RoLA
from rola.ops.naive import naive_rola
from rola.routing.decay import ConstantDecay, DecaySource, LearnedDecay
from rola.routing.factors import RouteFactors, RouteProducer
from rola.routing.producer import (
    LevelProducer,
    ProducedRoutes,
    RouteSource,
    SimplexProducer,
    dense_routing,
    sparse_routing,
)
from rola.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    LeafMassDecay,
    LevelConfig,
    SoftmaxActivation,
    Topology,
    UnionRouting,
    entmax,
    softmax,
)
from rola.state import LayerRecurrentState, RoLAState

__all__ = [
    # the op — the single autograd boundary (rola/interface.py)
    "rola_op",
    # the feature map: the gateway, its bundle, and the producers that fill it
    "RouteProducer",
    "RouteFactors",
    "LevelProducer",
    "ProducedRoutes",
    "RouteSource",
    "SimplexProducer",
    "dense_routing",
    "sparse_routing",
    # the decay-source slot — where the recurrence's rates come from
    "DecaySource",
    "ConstantDecay",
    "LearnedDecay",
    "LeafMassDecay",
    # layer
    "RoLA",
    # the stateful surface — a fixed-size recurrent state, NOT a KV cache
    "RoLAState",
    "LayerRecurrentState",
    # oracle — ships in the wheel; it is the definition (docs/testing.md)
    "naive_rola",
    # configuration
    "Topology",
    "LevelConfig",
    "IndependentRouting",
    "UnionRouting",
    "SoftmaxActivation",
    "EntmaxActivation",
    # tagged activation constructors — users write these, never property metadata
    "softmax",
    "entmax",
    "__version__",
]
