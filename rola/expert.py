# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The advanced surface. Nothing here is part of the contract.

The public surface is the reduction theorem: linear attention, plus the one slot that
makes it RoLA. Everything in this module is BELOW that line -- launch geometry, cost
models, the plan objects, the seam the op is split at. They are exported because a
serious consumer legitimately needs them (a gate that must ask for the exact numeric
arm, a conformance test that must run a stale plan, a deployment that must cap an
allocator) and reaching past a package into its privates is worse than a marked door.

**The contract for everything here: STABLE CONSTRUCTION, ADVISORY CONTENTS.** Build
these objects and pass them; do not assert on their fields. A ``Plan``'s geometry moves
with every recalibration, by design -- it is the answer to "how should this launch
run", and that answer is allowed to change when the cost model learns something. The
model facts a plan carries (``topology``, ``normalization``) are the exception and are
stable, because they are not decisions.

Importing this module is a statement that you accept that. The op never requires it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rola.interface import Dispatch, dispatch_for, requires_backward, select_backend
from rola.interface import _derive_plan as derive_plan
from rola.interface import _execute as execute
from rola.planner.schedule import (
    MMA_PRECISIONS,
    Plan,
    PlanAxes,
    Schedule,
    StaticDensity,
    build_schedule,
    plan_and_schedule,
    select_plan,
)
from rola.planner.select import (
    ConfigChoice,
    ConfigEvidence,
    CostModelSelector,
    FixedChoiceSelector,
    HardwareFacts,
    Selector,
    StaticPlanInputs,
    StatisticsBundle,
    feasible_bc,
    largest_feasible_bc,
    select_with,
)
from rola.routing.side_gain import GainConfig

__all__ = [
    "MMA_PRECISIONS",
    "ConfigChoice",
    "ConfigEvidence",
    "CostModelSelector",
    "Dispatch",
    "FixedChoiceSelector",
    "GainConfig",
    "HardwareFacts",
    "Plan",
    "PlanAxes",
    "PlanOverrides",
    "Schedule",
    "Selector",
    "StaticDensity",
    "StaticPlanInputs",
    "StatisticsBundle",
    "build_schedule",
    "derive_plan",
    "dispatch_for",
    "execute",
    "feasible_bc",
    "largest_feasible_bc",
    "plan_and_schedule",
    "requires_backward",
    "select_backend",
    "select_plan",
    "select_with",
]


@dataclass(frozen=True, slots=True)
class PlanOverrides:
    """Pins over axes the planner would otherwise choose.

    ``state_block`` (``BC``): the number of leafs one owner block owns -- the CTA's
    resident state tile. An explicit value SHRINKS THE FEASIBLE SET TO ITSELF, so the
    planner has nothing left to choose; it is still alignment-validated like any other
    member, which is what makes it a pin over a selected axis rather than a bypass of
    one. ``None`` takes the largest feasible value.

    ``mma_precision``: the readout MMA's numeric format. ``bf16`` is production; a gate
    comparing against the fp64 oracle asks for the exact arm here.

    ``max_physical_pages``: the paging allocator's hard cap, in atoms. ``None``
    provisions the dense limit, at which the ceiling can never bind. A finite value is a
    POLICY limit: a launch whose realized write footprint exceeds it is REFUSED,
    host-side, naming the demand and the arena size. It is never degraded.

    ``selector``: the cost model that turns statistics into a choice. It is one
    implementation behind an interface, and the shipped one reports ``INTERIM`` because
    its constants are knowingly unaudited. Swapping it is an assignment; that it is an
    assignment is the property this field exists to keep true.
    """

    state_block: int | None = None
    mma_precision: str = "bf16"
    max_physical_pages: int | None = None
    selector: Any = None

    def __post_init__(self) -> None:
        if self.state_block is not None and type(self.state_block) is not int:
            raise TypeError(f"state_block must be None or an exact int, got {self.state_block!r}")
        if self.mma_precision not in MMA_PRECISIONS:
            raise ValueError(
                f"mma_precision must be one of {MMA_PRECISIONS}, got {self.mma_precision!r}")
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
            f"surface -- got {type(value).__name__}. Its axes are ones the planner selects; "
            "pinning one is a deliberate act with a name.")
