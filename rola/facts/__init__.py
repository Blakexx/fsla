# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The facts engine: one structure from a call's raw inputs to its launch arguments.

A DAG is an ordered tuple of declared nodes; the runner walks it once; the terminal
node folds what the walk produced into a PLAN, and the kernels read only the plan
(docs/internals/facts/facts.md).
"""

from rola.facts.backward_dag import BACKWARD_DAG, run_backward
from rola.facts.chunk_dag import (
    CHUNK_DAG,
    ChunkContext,
    build_chunk_plan,
    validate_context,
)
from rola.facts.plan import BackwardPlan, ChunkPlan
from rola.facts.runner import Node, node, run, validate_dag
from rola.facts.types import SUPPLIABLE, Arm, CallClass, Mask, Seeds

__all__ = [
    "BACKWARD_DAG",
    "CHUNK_DAG",
    "SUPPLIABLE",
    "Arm",
    "BackwardPlan",
    "CallClass",
    "ChunkContext",
    "ChunkPlan",
    "Mask",
    "Node",
    "Seeds",
    "build_chunk_plan",
    "node",
    "run",
    "run_backward",
    "validate_context",
    "validate_dag",
]
