# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""RetentionRule -- whether this call keeps a reverse-pass ctx."""
from __future__ import annotations

from rola.facts.runner import rule


@rule(gives=("retained",), needs=("grad",))
def retention(grad: bool) -> bool:
    """Whether the forward retains its plan for a reverse pass.

    RETENTION IS THE WHOLE DIFFERENCE between a grad-bearing call and an inference
    one: what is kept is a REFERENCE to the plan that ran plus the readout's
    denominator, so an inference call keeps nothing and pays no two-pass activation
    cost (docs/internals/chunk/backward.md). Grad is a retention decision and never a
    mask -- it selects no subgraph and reaches no kernel.
    """
    return grad
