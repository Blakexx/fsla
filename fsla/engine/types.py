# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The facts engine's value types: the call class, the variant it selects, and the refusal every
no-backward clause raises.

Every fact is an immutable value. Tensor-valued facts are held by REFERENCE and
nothing here allocates (docs/internals/engine/engine.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

#: the MMA k-extent, and the page granule. The Python mirror of
#: the MMA k-extent (16, the ampere atom's `m16n8k16`, `common/atoms/mma.cuh`), which a C++
#: ``constexpr`` cannot supply here; :meth:`fsla.ops.paging.PageArena.__init__` checks
#: it against the atom geometry it addresses.
MMA_K_QUANTUM = 16


class BackwardNotImplemented(NotImplementedError):
    """Raised when a gradient is demanded of the forward-only kernel surface."""


class Variant(IntEnum):
    """The call classes a DAG specializes over -- never a per-fact bool.

    Specialization is per CALL CLASS so the built set stays fixed and small; a bool
    per fact would be `2^n` of them, and intra-variant variation is a runtime-null
    output skip instead (`state_in is None`).
    """

    PREFILL_STATELESS = 1
    """PREFILL_STATELESS -- `state_in=0 state_out=0 paged=0`."""
    PREFILL_STATEFUL = 2
    """PREFILL_STATEFUL -- `state_out=1`, `state_in` and `paged` free."""
    DECODE_DENSE = 3
    """DECODE_DENSE -- one token onto a dense plane: no probe, no gate, no admission."""
    DECODE_PAGED = 4
    """DECODE_PAGED -- one token onto an arena: the probe, the gate and the admission.

    A GROWTH STEP IS NOT A VARIANT. It is DECODE_PAGED run twice around an admission, which is why
    the growth path selects no different subgraph (docs/internals/engine/decode_dag.md).
    """


@dataclass(frozen=True, slots=True)
class CallClass:
    """What KIND of call this is, in the bits every downstream decision folds.

    These are ENGAGEMENT bits: `state_in` says this call was handed a state to
    continue, which is a different question from whether an entry-state TENSOR reaches
    the launch. The latter is `residency`'s answer, and it is what the skip predicate
    folds (`fsla/engine/rules/run_table.py`).

    The variant these bits select is :func:`fsla.engine.rules.variant.variant` -- a rule, not a
    property here: a decision folded from a fact belongs to band 2 wherever it is
    written, and one written as a method on the fact is a rule that no listing of
    `fsla/engine/rules/` would show.
    """

    state_in: bool
    state_out: bool
    paged: bool
    grad: bool


#: THE SUPPLIABLE SET IS CLOSED. A caller may substitute production -- the routing
#: bundle and the packed amplitude planes it stands for -- and nothing below it.
#: Everything downstream of that boundary is one canonical, un-overridable
#: implementation, gated once, so supplying one of those facts RAISES rather than
#: forking fact production.
SUPPLIABLE = frozenset({"routes", "read_plane", "write_plane"})
