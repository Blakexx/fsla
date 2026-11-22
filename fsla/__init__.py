# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""FSLA — Factored Sparse Linear Attention.

FSLA is linear attention under a structured feature map. A head's recurrent state is
expanded into many sub-states that share the head's projections, and a learned routing
decides which sub-states each token reads and writes. That reduction IS this API: a
user who knows linear attention meets exactly ONE new concept here, the feature map,
and everything else is linear attention's own contract.

    fsla_op(routes, values, masks, decay, plan=plan)

``routes`` is the ``(q, k)`` pair after the map -- a ``RouteFactors`` bundle -- ``values``
is the value stream in the internal format -- a ``Values`` bundle, ``Values.from_projection(v)``
-- ``masks`` is the DATA beside the rows (a bit per side, level, digit and token; the op computes with ``r * m``),
``decay`` is the recurrence dial and ``plan`` is the static what-to-do (``fsla.expert``). The layer DECLARES its routing -- the levels, each carrying its own
width -- and computes the bundles itself, in one producer call; a caller of the op with its own levels builds the
bundles with the same format utilities, ``RouteFactors.from_levels`` and ``Values.from_projection``:

    routing = Routing(uniform(2, union_routing(64, alpha=1.5)))
    layer = FSLA(routing, hidden_size=1024, num_heads=8, d_v=64,
                 decay=LearnedDecay(widths=routing.widths, num_heads=8))

**No public name says `v3`.** This repository IS that generation; encoding it in the
API would force a break at the next one for a reason that is not an API reason.

**What is deliberately NOT here.** Plans, schedules, launch geometry, numeric arms,
allocator caps -- every one of them is how the op RUNS rather than what the model IS,
and the op decides how it runs. They live in :mod:`fsla.expert`, which is a marked door
and not part of the contract.

| Group | Names | Stability |
|---|---|---|
| op | `fsla_op` | STABLE, semver-major to break |
| planner arguments | `descriptor`, `Descriptor`, `workspace_bytes` | STABLE |
| feature map | `RouteFactors`, `Masks`, `Routing`, `dense_routing`, `split_routing`, `tied_routing`, `union_routing`, `uniform` | STABLE |
| value stream | `Values` | STABLE |
| recurrence dials | `ConstantDecay`, `LearnedDecay`, `DecaySource`, `LeafMassDecay` | STABLE |
| layer | `FSLA`, `LayerContinuation` | STABLE |
| state | `state()`, `FSLAState` | STABLE |
| configuration | `Topology`, `softmax()`, `entmax(alpha)`, `IndependentRouting`, `TiedRouting`, `UnionRouting` | STABLE |
| oracle | `naive_fsla` | STABLE signature, EXACT semantics — it is the definition |
"""

__version__ = "0.1.0.dev0"

#: `fsla.state` is the CONSTRUCTOR and nothing else. The class and the function live
#: in the private `fsla._state` (renamed from `fsla/state.py`, whose public
#: name the function was shadowing): docs/internals/state.md
from fsla._state import FSLAState, state
from fsla.interface import fsla_op
from fsla.layer import FSLA, LayerContinuation
from fsla.ops.descriptor import Descriptor, descriptor, workspace_bytes
from fsla.ops.naive import naive_fsla
from fsla.routing.decay import ConstantDecay, DecaySource, LearnedDecay
from fsla.routing.factors import RouteFactors
from fsla.routing.masks import Masks
from fsla.routing.producer import (
    Routing,
    dense_routing,
    split_routing,
    tied_routing,
    uniform,
    union_routing,
)
from fsla.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    LeafMassDecay,
    SoftmaxActivation,
    TiedRouting,
    Topology,
    UnionRouting,
    entmax,
    softmax,
)
from fsla.values import Values

__all__ = [
    # the op — the single autograd boundary (fsla/interface.py)
    "fsla_op",
    # the op's planner arguments: the validated context of a call, and the scratch it needs (fsla/ops/descriptor.py)
    "Descriptor",
    "descriptor",
    "workspace_bytes",
    # the feature map: the bundle (the whole producer contract) and the shipped producer
    "RouteFactors",
    "Masks",
    "Values",
    "Routing",
    "dense_routing",
    "split_routing",
    "tied_routing",
    "uniform",
    "union_routing",
    # the decay-source slot — where the recurrence's rates come from
    "DecaySource",
    "ConstantDecay",
    "LearnedDecay",
    "LeafMassDecay",
    # layer
    "FSLA",
    "LayerContinuation",
    # the stateful surface — a fixed-size recurrent state, NOT a KV cache
    "state",
    "FSLAState",
    # oracle — ships in the wheel; it is the definition (docs/testing.md)
    "naive_fsla",
    # configuration
    "Topology",
    "IndependentRouting",
    "TiedRouting",
    "UnionRouting",
    "SoftmaxActivation",
    "EntmaxActivation",
    # tagged activation constructors — users write these, never property metadata
    "softmax",
    "entmax",
    "__version__",
]
