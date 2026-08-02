# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""RoLA — Routed Linear Attention.

A linear-attention head's recurrent state is expanded into many sub-states that
share the head's projections, with learned routing over those states. See
docs/spec.md for the normative contract and docs/evidence.md for how it is gated.

**No public name says `v3`** (ratified decision O-3). This repository *is* that
generation; encoding it in the API would force a break at the next one for a reason
that is not an API reason. The names below are the only spellings — the `v3`-marked
ones the fork used were renamed in a single commit, so no function has ever had two.

Five groups, and the grouping is the contract (extraction design §4.2):

| Group | Stability |
|---|---|
| functional op — `rola_func`, `rola_forward` | STABLE, semver-major to break |
| layer — `RoLA`, `LevelRoutingConfig` | STABLE |
| state — `RoLAState`, `LayerRecurrentState` | STABLE |
| configuration — `Topology`, `LevelConfig`, `softmax()`/`entmax(alpha)`, decay types | STABLE; the semantic surface `docs/spec.md` normatively describes |
| planner — `Plan`, `PlanAxes`, `select_plan`, `plan_and_schedule` | STABLE construction, **ADVISORY contents** (§4.4: a `Plan`'s fields move with every recalibration; construct plans and pass them, do not assert on them) |
| oracle — `naive_rola` | STABLE signature, EXACT semantics — it is the definition |
"""

__version__ = "0.1.0.dev0"

from rola.interface import requires_backward, rola_forward, rola_func
from rola.layers.rola import LevelRoutingConfig, RoLA, resolve_level_routing_configs
from rola.ops.naive import naive_rola
from rola.planner.schedule import (
    MMA_PRECISIONS,
    Plan,
    PlanAxes,
    StaticDensity,
    plan_and_schedule,
    select_plan,
)
from rola.routing.types import (
    EntmaxActivation,
    LeafMassDecay,
    LevelConfig,
    SoftmaxActivation,
    Topology,
    entmax,
    softmax,
)
from rola.state import LayerRecurrentState, RoLAState

__all__ = [
    # functional op — the single autograd boundary (rola/interface.py)
    "rola_func",
    "rola_forward",
    "requires_backward",
    # layer
    "RoLA",
    "LevelRoutingConfig",
    "resolve_level_routing_configs",
    # the stateful surface — a fixed-size recurrent state, NOT a KV cache
    "RoLAState",
    "LayerRecurrentState",
    # oracle — ships in the wheel; it is the definition (docs/evidence.md)
    "naive_rola",
    # planner — STABLE construction, ADVISORY contents (§4.4)
    "Plan",
    "PlanAxes",
    "StaticDensity",
    "select_plan",
    "plan_and_schedule",
    "MMA_PRECISIONS",
    # configuration — the semantic surface docs/spec.md describes
    "Topology",
    "LevelConfig",
    "SoftmaxActivation",
    "EntmaxActivation",
    # tagged activation constructors — users write these, never property metadata
    "softmax",
    "entmax",
    "LeafMassDecay",
    "__version__",
]
