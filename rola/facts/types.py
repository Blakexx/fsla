# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The facts engine's value types: the call class, the mask it selects, the arm.

Every fact is an immutable value. Tensor-valued facts are held by REFERENCE and
nothing here allocates (docs/internals/facts/facts.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple


class Mask(IntEnum):
    """The call classes a DAG specializes over -- never a per-fact bool.

    Specialization is per CALL CLASS so the built set stays fixed and small; a bool
    per fact would be `2^n` of them, and intra-mask variation is a runtime-null
    output skip instead (`read_mass is None`, `state_in is None`).
    """

    M1 = 1
    """PREFILL_STATELESS -- `state_in=0 state_out=0 paged=0`."""
    M2 = 2
    """PREFILL_STATEFUL -- `state_out=1`, `state_in` and `paged` free."""


class Arm(NamedTuple):
    """`(C, BC)` -- the built manifest arm a call runs on."""

    c: int
    bc: int


@dataclass(frozen=True, slots=True)
class CallClass:
    """What KIND of call this is, in the bits every downstream decision folds.

    These four bits are also the `run_table` truth table's input triple plus the
    retention bit, which is why the plan carries the class rather than the byte
    (`csrc/rola/src/chunk/chunk.cu`, docs/internals/chunk/chunk.md#the-run-table).
    """

    state_in: bool
    state_out: bool
    paged: bool
    grad: bool

    @property
    def mask(self) -> Mask:
        return Mask.M2 if (self.state_in or self.state_out or self.paged) else Mask.M1


#: THE SUPPLIABLE SET IS CLOSED. A caller may substitute production -- the routing
#: bundle and the packed amplitude planes it stands for -- and nothing below it.
#: Everything downstream of that boundary is one canonical, un-overridable
#: implementation, gated once, so supplying one of those facts RAISES rather than
#: forking fact production.
SUPPLIABLE = frozenset({"routes", "read_plane", "write_plane", "g_write", "read_mass"})
