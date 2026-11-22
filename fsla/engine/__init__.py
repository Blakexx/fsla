# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The engine: ONE structure carries every kernel family from raw inputs to a launch.

    Context -> DAG -> Plan -> execute(plan)

This ROOT holds the shared CORE and nothing family-specific: the DAG framework
(:mod:`fsla.engine.runner`), the value types (:mod:`fsla.engine.types`) and the
per-family plans (:mod:`fsla.engine.plan`). The DOMAINS are its subpackages --
:mod:`fsla.engine.facts` (the fact primitives and their bodies),
:mod:`fsla.engine.rules` (the band-2 folds) and :mod:`fsla.engine.dags` (the
per-family compositions).

The core is re-exported here because that is the layering statement: a domain reaches
it THROUGH this module and never through a sibling domain, so `rule` is a property of
the engine rather than of the fact-finder that happened to define it
(docs/internals/engine/engine.md).
"""

from fsla.engine.plan import (
    DecodeLaunch,
    DecodeOperands,
    DecodePlan,
    DecodeWorkspace,
    PrefillLaunch,
    PrefillOperands,
    PrefillPlan,
    PrefillWorkspace,
)
from fsla.engine.runner import (
    Node,
    node,
    rule,
    run,
    validate_capturable,
    validate_dag,
)
from fsla.engine.types import (
    MMA_K_QUANTUM,
    SUPPLIABLE,
    BackwardNotImplemented,
    CallClass,
    Carve,
    SidePlan,
    Variant,
)

__all__ = [
    "MMA_K_QUANTUM",
    "SUPPLIABLE",
    "BackwardNotImplemented",
    "CallClass",
    "Carve",
    "DecodeLaunch",
    "DecodeOperands",
    "DecodePlan",
    "DecodeWorkspace",
    "Node",
    "PrefillLaunch",
    "PrefillOperands",
    "PrefillPlan",
    "PrefillWorkspace",
    "SidePlan",
    "Variant",
    "node",
    "rule",
    "run",
    "validate_capturable",
    "validate_dag",
]
