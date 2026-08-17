# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The facts engine's value types: the call class, the mask it selects, the arm.

Every fact is an immutable value. Tensor-valued facts are held by REFERENCE and
nothing here allocates (docs/internals/facts/facts.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import torch


#: the MMA k-extent, and the page granule. The Python mirror of
#: the MMA k-extent (`chunk/arch_caps.cuh`'s `kMmaK` world), which a C++
#: ``constexpr`` cannot supply here; :meth:`rola.ops.paging.PageArena.__init__` checks
#: it against the atom geometry it addresses.
MMA_K_QUANTUM = 16


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


class Seeds(NamedTuple):
    """The reverse pass's SEED cotangents -- what the readout's divide hands the scans.

    `dm` is the read mass's own gradient and is `None` exactly when the bundle carries
    no mass: a runtime-null output skip, not a mask. Nothing here is RETAINED -- the
    seeds are built from `dy` inside the backward and die with it, which is why they
    are a fact of the walk and not a field of anything the forward keeps.
    """

    dnum: torch.Tensor
    dden: torch.Tensor
    dm: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class CallClass:
    """What KIND of call this is, in the bits every downstream decision folds.

    These are ENGAGEMENT bits: `state_in` says this call was handed a state to
    continue, which is a different question from whether an entry-state TENSOR reaches
    the launch. The latter is `residency`'s answer, and it is what the skip predicate
    folds (`rola/rules/run_table.py`).

    The mask these bits select is :func:`rola.rules.mask.mask` -- a rule, not a
    property here: a decision folded from a fact belongs to band 2 wherever it is
    written, and one written as a method on the fact is a rule that no listing of
    `rola/rules/` would show.
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
SUPPLIABLE = frozenset({"routes", "read_plane", "write_plane", "g_write", "read_mass"})
