# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The advanced surface. Nothing here is part of the contract.

The public surface is the reduction theorem: linear attention, plus the one slot that
makes it RoLA. Everything in this module is BELOW that line -- the dispatch's own
answer, the pins over axes the op otherwise decides, the producer-side statistics a
measurement reads. They are exported because a serious consumer legitimately needs
them (a gate that must ask for the exact numeric arm, a deployment that must pin the
state's backing) and reaching past a package into its privates is worse than a marked
door.

**The contract for everything here: STABLE CONSTRUCTION, ADVISORY CONTENTS.** Build
these objects and pass them; do not assert on their fields. What a :class:`Dispatch`
reports is the answer to "how will this call run", and that answer moves with the
machine and with the built matrix. The model facts it carries are the exception and
are stable, because they are not decisions.

Importing this module is a statement that you accept that. The op never requires it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rola.interface import Dispatch, dispatch_for, requires_backward, select_backend
from rola.routing.density import StaticDensity
from rola.routing.side_gain import GainConfig

__all__ = [
    "Dispatch",
    "GainConfig",
    "PlanOverrides",
    "StaticDensity",
    "dispatch_for",
    "requires_backward",
    "select_backend",
]


@dataclass(frozen=True, slots=True)
class PlanOverrides:
    """Pins over axes the op would otherwise decide.

    ``state_block`` (``BC``): the number of leafs one owner block owns -- a CTA's
    resident state tile. The chunk consumer's ``BC`` belongs to the built arm the
    manifest carries for a topology and is NOT host-selectable, so any value other than
    ``None`` is refused by name at the call rather than accepted and ignored: a knob
    that silently does nothing is how a caller comes to believe a launch was shaped a
    way it never was.

    ``mma_precision``: DEAD AXIS, retained as a declared restriction: ``"bf16"`` is
    the only value (bf16 storage, fp32 accumulation, fp64 reference as oracle); any
    other value is a hard refusal. A gate needing more precision runs the fp64
    reference path.

    ``paging``: the state's BACKING, and the expert overrides the MAPPING ONLY.
    A stateful call pages by default -- one page is one MMA atom, pages are admitted
    by the write-side plan's exact bitmap, and an atom with no page reads as zeros, so
    paging is BITWISE INVISIBLE and there is nothing for a caller to turn on.
    ``None`` (the default) is that default; ``False`` pins the DENSE plane, which is
    what the bit-identity gate compares against; ``True`` is REFUSED, because a knob
    whose only value is the default is a knob that lies about being a decision.
    """

    state_block: int | None = None
    mma_precision: str = "bf16"
    paging: bool | None = None

    def __post_init__(self) -> None:
        if self.state_block is not None and type(self.state_block) is not int:
            raise TypeError(f"state_block must be None or an exact int, got {self.state_block!r}")
        if self.mma_precision != "bf16":
            raise ValueError(
                "mma_precision is a dead axis: 'bf16' is the only value (the fp64 "
                f"reference path is the precision escape), got {self.mma_precision!r}")
        if self.paging is not None and type(self.paging) is not bool:
            raise TypeError(f"paging must be None or a bool, got {self.paging!r}")
        if self.paging is True:
            raise ValueError(
                "paging is ON by default for every stateful call, so there is nothing "
                "to turn on: pass paging=None (or omit it) for the paged backing, or "
                "paging=False to pin the dense plane. The expert overrides the MAPPING "
                "only (docs/internals/state.md).")

    @classmethod
    def of(cls, value: Any) -> PlanOverrides:
        """The overrides, or a hard failure naming what was passed instead."""
        if type(value) is cls:
            return value
        raise TypeError(
            "the `expert` argument takes a rola.expert.PlanOverrides -- the marked advanced "
            f"surface -- got {type(value).__name__}. Its axes are ones the op decides; "
            "pinning one is a deliberate act with a name.")
