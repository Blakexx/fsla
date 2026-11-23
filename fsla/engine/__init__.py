# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The engine: the host side of a call, from raw inputs to a launch.

    prefill_call / decode_step -> Launch -> execute(launch)

This ROOT holds the shared CORE and nothing family-specific: the value types (:mod:`fsla.engine.types`), the launch bundles
(:mod:`fsla.engine.plan`) and the profiler range (:mod:`fsla.engine.trace`). The two calls are plain functions beside them
(:mod:`fsla.engine.prefill_call`, :mod:`fsla.engine.decode_step`); their primitives are the subpackages
:mod:`fsla.engine.facts` (the packing, the operand list, the gradient clause) and :mod:`fsla.engine.rules` (the pure host folds)
(docs/internals/engine/engine.md).
"""

from fsla.engine.plan import (
    DecodeLaunch,
    DecodeOperands,
    DecodeWorkspace,
    PrefillLaunch,
    PrefillOperands,
    PrefillWorkspace,
)
from fsla.engine.types import MMA_K_QUANTUM, BackwardNotImplemented

__all__ = [
    "MMA_K_QUANTUM",
    "BackwardNotImplemented",
    "DecodeLaunch",
    "DecodeOperands",
    "DecodeWorkspace",
    "PrefillLaunch",
    "PrefillOperands",
    "PrefillWorkspace",
]
