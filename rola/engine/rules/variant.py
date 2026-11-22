# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""VariantRule -- which specialized subgraph this call class runs, per kernel family."""
from __future__ import annotations

from rola.engine import Variant, rule


@rule(gives=("variant",), needs=("call_class",))
def variant(call_class) -> Variant:
    """The DAG band this call runs: the stateful band the moment any state bit is engaged.

    Specialization is per CALL CLASS and never per fact: a bool per fact would be
    `2^n` built subgraphs, and intra-variant variation is a runtime-null output skip
    instead (docs/internals/engine/rules.md).
    """
    return (Variant.PREFILL_STATEFUL if (call_class.state_in or call_class.state_out or call_class.paged)
            else Variant.PREFILL_STATELESS)


@rule(gives=("variant",), needs=("paged",), variants=(Variant.DECODE_DENSE, Variant.DECODE_PAGED))
def decode_variant(paged) -> Variant:
    """The decode band this step runs: the BACKING is the whole of the distinction.

    A dense-backed step addresses the leaf space directly and has nothing to admit; a
    paged one asks the device whether its write atoms are resident. That is the only
    difference in the node set, and a growth step selects no different subgraph -- it is
    DECODE_PAGED walked twice around an admission (docs/internals/engine/decode_dag.md).
    """
    return Variant.DECODE_PAGED if paged else Variant.DECODE_DENSE
