# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The facts: what is TRUE about one call, and the framework that finds it.

A fact is an immutable value produced by exactly one declared node. This package
holds the primitives and their BODIES -- the packing and the write-atom bitmap
(:mod:`rola.facts.planes`), the build's arm censuses (:mod:`rola.facts.manifest`),
the operand list (:mod:`rola.facts.operands`), the bundle adapters
(:mod:`rola.facts.call`) -- beside the value types, the plan objects and the runner.

Only the DEVICE-FREE core is re-exported here, and that is a layering statement
rather than a convenience one: :mod:`rola.rules` reads this package and must not
reach the extension through it. The primitives that do are imported by module
(docs/internals/facts/facts.md).
"""

from rola.facts.plan import BackwardPlan, ChunkPlan
from rola.facts.runner import Node, node, rule, run, validate_dag
from rola.facts.types import MMA_K_QUANTUM, SUPPLIABLE, Arm, CallClass, Mask, Seeds

__all__ = [
    "MMA_K_QUANTUM",
    "SUPPLIABLE",
    "Arm",
    "BackwardPlan",
    "CallClass",
    "ChunkPlan",
    "Mask",
    "Node",
    "Seeds",
    "node",
    "rule",
    "run",
    "validate_dag",
]
