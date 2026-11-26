# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""FSLA — Factored Sparse Linear Attention.

FSLA is linear attention under a structured feature map. A head's recurrent state is
expanded into many sub-states that share the head's projections, and a learned routing
decides which sub-states each token reads and writes. That reduction IS this API: a
user who knows linear attention meets exactly ONE new concept here, the feature map,
and everything else is linear attention's own contract.

    y, state = fsla.prefill(routes, values, masks, density, state=state, decay=decay)

``routes`` is the ``(q, k)`` pair after the map -- a ``RouteFactors`` bundle -- ``values``
is the value stream in the internal format -- a ``Values`` bundle, ``Values.from_projection(v)``
-- ``masks`` is the DATA beside the rows (a bit per side, level, digit and token; a bit of 0 is the caller's guarantee that the row is zero; a level-side's mask may be absent, ``None``, which means dense),
``density`` is how full each level is (a uint8 ``[2, D]``, 255 for every digit live, built by ``fsla.masks_and_density`` or ``fsla.density_of_masks``; it chooses how the work is laid out and changes no result),
the ``state`` is its own argument, in and out, and ``decay`` is the recurrence dial. The call carries only its inputs: it builds its own descriptor
and workspace inside. The one-token decode step is ``fsla.decode_step_gated(routes, values, masks, state, decay=decay)``. The layer DECLARES its routing -- the levels, each carrying its own
width -- and computes the bundles itself, in one producer call; a caller of the op with its own levels builds the
bundles with the same format utilities, ``RouteFactors.from_levels`` and ``Values.from_projection``:

    routing = Routing(uniform(2, split_routing(64, alpha=1.5)))
    layer = FSLA(routing, hidden_size=1024, num_heads=8, d_v=64,
                 decay=LearnedDecay(widths=routing.widths, num_heads=8))

**No public name says `v3`.** This repository IS that generation; encoding it in the
API would force a break at the next one for a reason that is not an API reason.

**What is deliberately NOT here.** Plans, schedules, launch geometry, numeric arms,
allocator caps -- every one of them is how the op RUNS rather than what the model IS,
and the op decides how it runs, inside the call.

| Group | Names | Stability |
|---|---|---|
| op | `prefill`, `decode_step_gated` | STABLE, semver-major to break |
| feature map | `RouteFactors`, `Masks`, `masks_and_density`, `density_of_masks`, `Routing`, `dense_routing`, `split_routing`, `tied_routing`, `uniform` | STABLE |
| value stream | `Values` | STABLE |
| recurrence dials | `ConstantDecay`, `LearnedDecay`, `DecaySource`, `LeafMassDecay` | STABLE |
| layer | `FSLA`, `LayerContinuation` | STABLE |
| state | `dense_state()`, `FSLAState` | STABLE |
| segmentation | `max_tokens(n)`, `max_memory(bytes)`, `segment_plan(length, maximum)` | STABLE |
| configuration | `Topology`, `softmax()`, `entmax(alpha)`, `IndependentRouting`, `TiedRouting` | STABLE |
| oracle | `naive_fsla` | STABLE signature, EXACT semantics — it is the definition |
"""

__version__ = "0.1.0.dev0"

#: `fsla.dense_state` is the CONSTRUCTOR and nothing else. The interface and the function live
#: in the private `fsla._state` (docs/internals/state.md)
from fsla._state import FSLAState, dense_state
from fsla.engine.decode_step import decode_step_gated
from fsla.interface import prefill
from fsla.layer import FSLA, LayerContinuation
from fsla.ops.naive import naive_fsla
from fsla.routing.decay import ConstantDecay, DecaySource, LearnedDecay
from fsla.routing.factors import RouteFactors
from fsla.routing.masks import Masks, density_of_masks, masks_and_density
from fsla.routing.producer import (
    Routing,
    dense_routing,
    split_routing,
    tied_routing,
    uniform,
)
from fsla.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    LeafMassDecay,
    SoftmaxActivation,
    TiedRouting,
    Topology,
    entmax,
    softmax,
)
from fsla.segmentation import max_memory, max_tokens, segment_plan
from fsla.values import Values

__all__ = [
    # the ops — the prefill, the single autograd boundary (fsla/interface.py), and the decode step
    # (fsla/engine/decode_step.py)
    "prefill",
    "decode_step_gated",
    # the feature map: the bundle (the whole producer contract) and the shipped producer
    "RouteFactors",
    "Masks",
    "masks_and_density",
    "density_of_masks",
    "Values",
    "Routing",
    "dense_routing",
    "split_routing",
    "tied_routing",
    "uniform",
    # the decay-source slot — where the recurrence's rates come from
    "DecaySource",
    "ConstantDecay",
    "LearnedDecay",
    "LeafMassDecay",
    # layer
    "FSLA",
    "LayerContinuation",
    # the stateful surface — a fixed-size recurrent state, NOT a KV cache
    "dense_state",
    "FSLAState",
    # segmentation: the handles a layer is given and the one function that plans a call (fsla/segmentation.py)
    "max_tokens",
    "max_memory",
    "segment_plan",
    # oracle — ships in the wheel; it is the definition (docs/testing.md)
    "naive_fsla",
    # configuration
    "Topology",
    "IndependentRouting",
    "TiedRouting",
    "SoftmaxActivation",
    "EntmaxActivation",
    # tagged activation constructors — users write these, never property metadata
    "softmax",
    "entmax",
    "__version__",
]
