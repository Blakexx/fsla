# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""RunTableRule -- the block-skip truth table over the liveness byte's three bits.

The rule is written once, in readable form, and the mask is DERIVED by evaluating it
over all eight bit values; the kernel's whole predicate is then one shift and one AND
(docs/internals/chunk/chunk.md#run-table). The production derivation is still
`csrc/rola/src/chunk/chunk.cu`'s; this fold is gated against the same five-row rule
and moves under the ABI in P75 stage 6 (design.md §H.2).

The three inputs are PRESENCE facts -- the entry-state plane, the exit-state plane and
the page table, each a tensor or None -- and never the call class's engagement bits.
The two differ: a fresh paged sequence engages a state and reaches the launch with no
entry plane at all, and it is the launch argument that decides what a skipped block
would have cost.
"""
from __future__ import annotations

from typing import Any

from rola.engine import rule


@rule(gives=("run_table",), needs=("state_in", "state_out", "page_table"))
def run_table(state_in: Any, state_out: Any, page_table: Any) -> int:
    """The `uint32` whose bit `v` says a block with liveness byte `v` must run."""
    carries_in, stores_out, paged = (state_in is not None, state_out is not None,
                                     page_table is not None)
    table = 0
    for v in range(8):
        R, W, S = bool(v & 1), bool(v & 2), bool(v & 4)
        #: `y` needs a read entry AND something for it to read: this call's own
        #: write, or a carried state the block may hold. Without a residency map the
        #: dense backing cannot witness the latter and must assume it.
        run = R and (W or (carries_in and (S if paged else True)))
        #: the state epilogue stores this block's atoms. Paged, an unwritten block
        #: must leave its slot UNCHANGED, which is what skipping does; dense, the
        #: plane is uninitialized and is written WHOLE, so nothing may be skipped.
        if stores_out:
            run = run or (W if paged else True)
        if run:
            table |= 1 << v
    return table
