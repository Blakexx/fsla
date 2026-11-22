# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The advanced surface. Nothing here is part of the contract.

The public surface is the reduction theorem: linear attention, plus the one slot that
makes it FSLA. Everything in this module is BELOW that line -- the envelope's own
answer, the pins over axes the op otherwise decides, the fine-grained gain layout
behind the layer's one public flag. They are exported because a serious consumer
legitimately needs them (a caller that must ask whether a configuration has an arm
before building it, a deployment that must pin the state's backing) and reaching past
a package into its privates is worse than a marked door.

**The contract for everything here: STABLE CONSTRUCTION, ADVISORY CONTENTS.** Build
these objects and pass them; do not assert on their fields. What
:func:`~fsla.engine.facts.call.envelope_refusal` reports moves with the built matrix.

Importing this module is a statement that you accept that. The op never requires it.
"""

from __future__ import annotations

from fsla.engine import PrefillPlan
from fsla.engine.facts.call import envelope_refusal, require_envelope
from fsla.interface import requires_backward
from fsla.ops.produce import decode_descriptor, prefill_plan
from fsla.routing.side_gain import GainConfig

__all__ = [
    "GainConfig",
    "PrefillPlan",
    "decode_descriptor",
    "envelope_refusal",
    "prefill_plan",
    "require_envelope",
    "requires_backward",
]
