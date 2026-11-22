# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The rule layer: band 2, the pure host folds from facts to execution decisions.

Rule engines are PLURAL over ONE shared fact set. None of them owns a fact, none of
them recomputes one, and none of them may issue a launch or an allocation. Each is
declared with :func:`fsla.engine.runner.rule`, which is what makes this directory's
listing the layer's inventory (docs/internals/engine/rules.md).
"""

from fsla.engine.rules.envelope import envelope, prefill_envelope
from fsla.engine.rules.paging import admission, paging_preference
from fsla.engine.rules.run_table import run_table
from fsla.engine.rules.variant import decode_variant, variant

__all__ = [
    "admission",
    "decode_variant",
    "envelope",
    "paging_preference",
    "prefill_envelope",
    "run_table",
    "variant",
]
