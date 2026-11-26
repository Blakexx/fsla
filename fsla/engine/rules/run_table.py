# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""RunTableRule -- the block-skip truth table over the liveness byte's three bits.

The rule is written once, in readable form, and the mask is DERIVED by evaluating it
over all eight bit values; the kernel's whole predicate is then one shift and one AND
(docs/internals/engine/rules.md#run-table). This fold is the ONE derivation and the
`uint32` is a launch argument: the table a call PLANS is the table it RUNS, which a
second derivation behind the extension boundary could not promise.

The exit sweep stores the atoms this call WRITES and no others, so a block with no write has nothing to
store, and a fresh dense plane is zeroed at bind (`fsla/engine/prefill_call.py`'s backing).

The two inputs are PRESENCE facts -- the entry-state plane and the exit-state plane, each a tensor or None --
and never the call class's engagement bits: it is the launch argument that decides what a skipped block would
have cost.
"""
from __future__ import annotations

from typing import Any


def run_table(state_in: Any, state_out: Any) -> int:
    """The `uint32` whose bit `v` says a block with liveness byte `v` must run."""
    carries_in, stores_out = state_in is not None, state_out is not None
    table = 0
    for v in range(8):
        R, W = bool(v & 1), bool(v & 2)
        #: `y` needs a read entry AND something for it to read: this call's own write, or a carried
        #: state, which a dense plane always holds.
        run = R and (W or carries_in)
        #: the state epilogue stores this block's atoms. An unwritten block must end holding what it
        #: carried in, which is what skipping leaves: a continuation is updated in place, and a FRESH
        #: dense plane is ZEROED at bind rather than written whole.
        if stores_out:
            run = run or W
        if run:
            table |= 1 << v
    return table
