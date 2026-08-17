# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The per-family compositions: one DAG each, read top to bottom.

A DAG is a literal ordered tuple of band-1 fact nodes (:mod:`rola.facts.nodes`) and
band-2 rules (:mod:`rola.rules`), walked once by :func:`rola.facts.runner.run`; its
terminal node folds what the walk produced into the family's PLAN, and the kernels
read only the plan (docs/internals/facts/facts.md).
"""

from rola.dags.backward_dag import BACKWARD_DAG, run_backward
from rola.dags.chunk_dag import (
    CHUNK_DAG,
    ChunkContext,
    build_chunk_plan,
    validate_context,
)

__all__ = [
    "BACKWARD_DAG",
    "CHUNK_DAG",
    "ChunkContext",
    "build_chunk_plan",
    "run_backward",
    "validate_context",
]
