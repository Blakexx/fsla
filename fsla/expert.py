# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The advanced surface. Nothing here is part of the contract.

The public surface is the reduction theorem: linear attention, plus the one slot that
makes it FSLA. Everything in this module is BELOW that line -- the producer's descriptor rules, the pins over axes the op otherwise decides, the fine-grained gain layout
behind the layer's one public flag. They are exported because a serious consumer
legitimately needs them (a caller that must make the descriptor of a call before making the call) and reaching past
a package into its privates is worse than a marked door.

**The contract for everything here: STABLE CONSTRUCTION, ADVISORY CONTENTS.** Build
these objects and pass them; do not assert on their fields. What
a descriptor's refusal says moves with the built matrix.

Importing this module is a statement that you accept that. The op never requires it.
"""

from __future__ import annotations

from fsla.interface import requires_backward
from fsla.ops.produce import decode_descriptor, prefill_descriptor
from fsla.routing.side_gain import GainConfig

__all__ = [
    "GainConfig",
    "decode_descriptor",
    "prefill_descriptor",
    "requires_backward",
]
