# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The advanced surface. Nothing here is part of the contract.

The public surface is the reduction theorem: linear attention, plus the one slot that
makes it RoLA. Everything in this module is BELOW that line -- launch geometry, the
derivation that fixes it, the plan objects, the seam the op is split at. They are
exported because a serious consumer legitimately needs them (a gate that must ask for
the exact numeric arm, a conformance test that must run a stale plan, a deployment that
must cap an allocator) and reaching past a package into its privates is worse than a
marked door.

**The contract for everything here: STABLE CONSTRUCTION, ADVISORY CONTENTS.** Build
these objects and pass them; do not assert on their fields. A ``Plan``'s geometry is
the answer to "how should this launch run", and that answer moves with the machine it
is derived for. The model facts a plan carries (``topology``, ``normalization``) are
the exception and are stable, because they are not decisions.

Importing this module is a statement that you accept that. The op never requires it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rola.interface import Dispatch, dispatch_for, requires_backward, select_backend
from rola.interface import _derive_plan as derive_plan
from rola.interface import _execute as execute
from rola.planner.derive import MachineFacts, derive_bc, feasible_bc, largest_feasible_bc
from rola.planner.schedule import (
    Plan,
    Schedule,
    StaticDensity,
    build_plan,
    build_schedule,
    plan_and_schedule,
)
from rola.routing.side_gain import GainConfig

__all__ = [
    "Dispatch",
    "GainConfig",
    "MachineFacts",
    "Plan",
    "PlanOverrides",
    "Schedule",
    "StaticDensity",
    "build_plan",
    "build_schedule",
    "derive_bc",
    "derive_plan",
    "dispatch_for",
    "execute",
    "feasible_bc",
    "largest_feasible_bc",
    "plan_and_schedule",
    "requires_backward",
    "select_backend",
]


@dataclass(frozen=True, slots=True)
class PlanOverrides:
    """Pins over axes the planner would otherwise derive.

    ``state_block`` (``BC``): the number of leafs one owner block owns -- the CTA's
    resident state tile, and THE EXPERT PIN. An explicit value REPLACES
    :func:`~rola.planner.derive.derive_bc`'s three ceilings rather than biasing them,
    and it is still alignment-validated like any other member, which is what makes it a
    pin over a derived axis and not an escape from one. ``None`` derives it.

    ``mma_precision``: DEAD AXIS, retained as a declared restriction: ``"bf16"`` is
    the only value (bf16 storage, fp32 accumulation, fp64 reference as oracle); any
    other value is a hard refusal. A gate needing more precision runs the fp64
    reference path.

    ``paging``: commit state pages on first write to an owner, so the resident state
    scales with the owners the routing REALIZES rather than with the topology's
    provisioned capacity. One page is one MMA atom, pages are admitted by the write-side
    plan, and an atom with no page reads as zeros -- so paging is BITWISE INVISIBLE, and
    it is here rather than on the model because it changes what is allocated and nothing
    about what is computed. A cap implies it.

    ``max_physical_pages``: the paging allocator's hard cap, in atoms. ``None``
    provisions the dense limit, at which the ceiling can never bind. A finite value is a
    POLICY limit: a launch whose realized write footprint exceeds it is REFUSED,
    host-side, naming the demand and the arena size. It is never degraded.
    """

    state_block: int | None = None
    mma_precision: str = "bf16"
    paging: bool = False
    max_physical_pages: int | None = None

    def __post_init__(self) -> None:
        if self.state_block is not None and type(self.state_block) is not int:
            raise TypeError(f"state_block must be None or an exact int, got {self.state_block!r}")
        if self.mma_precision != "bf16":
            raise ValueError(
                "mma_precision is a dead axis: 'bf16' is the only value (the fp64 "
                f"reference path is the precision escape), got {self.mma_precision!r}")
        if type(self.paging) is not bool:
            raise TypeError(f"paging must be bool, got {self.paging!r}")
        if self.max_physical_pages is not None and type(self.max_physical_pages) is not int:
            raise TypeError(
                f"max_physical_pages must be None or an exact int, got {self.max_physical_pages!r}")

    @classmethod
    def of(cls, value: Any) -> PlanOverrides:
        """The overrides, or a hard failure naming what was passed instead."""
        if type(value) is cls:
            return value
        raise TypeError(
            "the `expert` argument takes a rola.expert.PlanOverrides -- the marked advanced "
            f"surface -- got {type(value).__name__}. Its axes are ones the planner derives; "
            "pinning one is a deliberate act with a name.")
