# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""MaskRule -- which specialized subgraph this call class runs."""
from __future__ import annotations

from rola.facts.runner import rule
from rola.facts.types import Mask


@rule(gives=("mask",), needs=("call_class",))
def mask(call_class) -> Mask:
    """The DAG band this call runs: M2 the moment any state bit is engaged.

    Specialization is per CALL CLASS and never per fact: a bool per fact would be
    `2^n` built subgraphs, and intra-mask variation is a runtime-null output skip
    instead (docs/internals/facts/rules.md).
    """
    return (Mask.M2 if (call_class.state_in or call_class.state_out or call_class.paged)
            else Mask.M1)
