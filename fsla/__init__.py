# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""FSLA — Factored Sparse Linear Attention.

FSLA is linear attention under a structured feature map. A head's recurrent state is
expanded into many sub-states that share the head's projections, and a learned routing
decides which sub-states each token reads and writes. That reduction IS this API: a
user who knows linear attention meets exactly ONE new concept here, the feature map,
and everything else is linear attention's own contract.

    y, state = fsla.prefill(routes, values, masks, density, state=state, decay=decay)

``routes`` is the ``(q, k)`` pair after the map -- a ``RouteFactors`` bundle, a scaled block a level-side -- ``values``
is the value stream in the internal format -- a ``ScaledBlock`` in the ``VALUES`` form, ``ScaledBlock.quantize(v, VALUES)``
-- ``masks`` is the DATA beside the rows (a bit per side, level, digit and token; a bit of 0 is the caller's guarantee that the row is zero; a level-side's mask may be absent, ``None``, which means dense),
``density`` is how full each level is (a uint8 ``[2, D]``, 255 for every digit live, built by ``fsla.masks_and_density`` or ``fsla.density_of_masks``; it chooses how the work is laid out and changes no result),
the ``state`` is its own argument, in and out, and ``decay`` is the recurrence dial. The call carries only its inputs: it builds its own descriptor
and workspace inside. The one-token decode step is ``fsla.decode_step_gated(routes, values, masks, state, decay=decay)``. The layer DECLARES its routing -- the
topology once, a flat nonnegative activation a write level, and the read heads' tuples -- and computes the bundles itself, in one producer call; a
caller of the op with its own levels builds the bundles with the same format utilities, ``RouteFactors.from_levels`` and ``ScaledBlock.quantize``:

    routing = Routing(topology=(64, 64), write=(entmax(1.5), softmax()), read=((softmax(), entmax(1.5)),))
    layer = FSLA(routing, hidden_size=1024, num_heads=8, group=1, d_v=64,
                 decay=LearnedDecay(widths=routing.widths, num_heads=8))

**No public name says `v3`.** This repository IS that generation; encoding it in the
API would force a break at the next one for a reason that is not an API reason.

**What is deliberately NOT here.** Plans, schedules, launch geometry, numeric arms,
allocator caps -- every one of them is how the op RUNS rather than what the model IS,
and the op decides how it runs, inside the call.

| Group | Names | Stability |
|---|---|---|
| op | `prefill`, `decode_step_gated` | STABLE, semver-major to break |
| feature map | `RouteFactors`, `Masks`, `masks_and_density`, `density_of_masks`, `Routing` | STABLE |
| scaled blocks | `ScaledBlock` (a routing level-side's rows, the values) | STABLE |
| recurrence dials | `ConstantDecay`, `LearnedDecay`, `DecaySource`, `LeafMassDecay` | STABLE |
| layer | `FSLA`, `LayerContinuation` | STABLE |
| state | `dense_state()`, `FSLAState` | STABLE |
| segmentation | `SegmentPolicy(max_tokens, max_heads)`, `max_tokens(n)`, `max_memory(bytes)`, `segment_plan(length, policy, *, batch, heads)` | STABLE |
| configuration | `Topology`, `softmax()`, `entmax(alpha)`, `tied()` | STABLE |
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
from fsla.routing.producer import Routing
from fsla.routing.quant import ScaledBlock
from fsla.routing.types import (
    EntmaxActivation,
    LeafMassDecay,
    SoftmaxActivation,
    Topology,
    entmax,
    softmax,
    tied,
)
from fsla.segmentation import SegmentPolicy, max_memory, max_tokens, segment_plan

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
    "Routing",
    # the scaled block: a routing level-side's rows and the values (fsla/routing/quant.py)
    "ScaledBlock",
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
    # segmentation: the policy a layer is given, its makers, and the one function that plans a call (fsla/segmentation.py)
    "SegmentPolicy",
    "max_tokens",
    "max_memory",
    "segment_plan",
    # oracle — ships in the wheel; it is the definition (docs/testing.md)
    "naive_fsla",
    # configuration
    "Topology",
    "SoftmaxActivation",
    "EntmaxActivation",
    # tagged activation constructors and the tied marker — users write these, never property metadata
    "softmax",
    "entmax",
    "tied",
    "__version__",
]
