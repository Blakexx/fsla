# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""RoLA — Routed Linear Attention.

A linear-attention head's recurrent state is expanded into many sub-states that
share the head's projections, with learned routing over those states. See
docs/spec.md for the normative contract and docs/evidence.md for how it is gated.

NOTE ON NAMES (extraction, in progress). Several exported names still carry the
`v3` generation marker they had inside the fork — `naive_rola_v3`,
`select_v3_plan`, `plan_and_schedule_v3`, `v3_consumer_forward`. Ratified design
decision O-3 is that **no public name says `v3`**: this repository *is* that
generation, and encoding it in the API would force a break at the next one for a
reason that is not an API reason. The rename lands with `rola/interface.py` and
the three-layer op stack (design §4.3), which is the commit that also gives the
op a single autograd boundary. Until then this module exports the names as they
are rather than inventing aliases that would leave two spellings of one function
in the wild.
"""

__version__ = "0.1.0.dev0"

from rola.layers.rola import LevelRoutingConfig, RoLA, resolve_level_routing_configs
from rola.ops.consumer import v3_consumer_forward
from rola.ops.naive import naive_rola_v3
from rola.planner.schedule import MMA_PRECISIONS, PlanAxes, plan_and_schedule_v3, select_v3_plan
from rola.routing.types import (
    EntmaxActivation,
    LeafMassDecay,
    LevelConfig,
    SoftmaxActivation,
    Topology,
)
from rola.state import LayerRecurrentState, RoLAState

__all__ = [
    # layer
    "RoLA",
    "LevelRoutingConfig",
    "resolve_level_routing_configs",
    # the stateful surface — a fixed-size recurrent state, NOT a KV cache
    "RoLAState",
    "LayerRecurrentState",
    # op
    "v3_consumer_forward",
    # oracle — ships in the wheel; it is the definition (docs/evidence.md)
    "naive_rola_v3",
    # planner
    "select_v3_plan",
    "plan_and_schedule_v3",
    "PlanAxes",
    "MMA_PRECISIONS",
    # configuration — the semantic surface docs/spec.md describes
    "Topology",
    "LevelConfig",
    "SoftmaxActivation",
    "EntmaxActivation",
    "LeafMassDecay",
    "__version__",
]
