# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The rule layer: band 2, the pure host folds from facts to execution decisions.

Rule engines are PLURAL over ONE shared fact set. None of them owns a fact, none of
them recomputes one, and none of them may issue a launch or an allocation. Each is
declared with :func:`rola.facts.runner.rule`, which is what makes this directory's
listing the layer's inventory (docs/internals/facts/rules.md).
"""

from rola.rules.arm import CHUNK_TOKENS, arm
from rola.rules.envelope import (
    arm_envelope,
    backward_envelope,
    envelope,
    training_envelope,
)
from rola.rules.mask import decode_mask, mask
from rola.rules.paging import admission, paging_preference
from rola.rules.retention import retention
from rola.rules.run_table import run_table

__all__ = [
    "CHUNK_TOKENS",
    "admission",
    "arm",
    "arm_envelope",
    "backward_envelope",
    "decode_mask",
    "envelope",
    "mask",
    "paging_preference",
    "retention",
    "run_table",
    "training_envelope",
]
