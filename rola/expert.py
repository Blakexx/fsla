# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The advanced surface. Nothing here is part of the contract.

The public surface is the reduction theorem: linear attention, plus the one slot that
makes it RoLA. Everything in this module is BELOW that line -- the envelope's own
answer, the pins over axes the op otherwise decides, the fine-grained gain layout
behind the layer's one public flag. They are exported because a serious consumer
legitimately needs them (a caller that must ask whether a configuration has an arm
before building it, a deployment that must pin the state's backing) and reaching past
a package into its privates is worse than a marked door.

**The contract for everything here: STABLE CONSTRUCTION, ADVISORY CONTENTS.** Build
these objects and pass them; do not assert on their fields. What
:func:`~rola.interface.envelope_refusal` reports moves with the built matrix.

Importing this module is a statement that you accept that. The op never requires it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rola.interface import envelope_refusal, require_envelope, requires_backward
from rola.routing.side_gain import GainConfig

__all__ = [
    "GainConfig",
    "PlanOverrides",
    "envelope_refusal",
    "require_envelope",
    "requires_backward",
]


@dataclass(frozen=True, slots=True)
class PlanOverrides:
    """Pins over axes the op would otherwise decide.

    resident state tile. The chunk consumer's ``BC`` belongs to the built arm the
    manifest carries for a topology and is NOT host-selectable, so any value other than
    ``None`` is refused by name at the call rather than accepted and ignored: a knob
    that silently does nothing is how a caller comes to believe a launch was shaped a
    way it never was.


    ``paging``: the state's BACKING, and the expert overrides the MAPPING ONLY.
    A stateful call pages by default -- one page is one MMA atom, pages are admitted
    by the write-side plan's exact bitmap, and an atom with no page reads as zeros, so
    paging is BITWISE INVISIBLE and there is nothing for a caller to turn on.
    ``None`` (the default) is that default; ``False`` pins the DENSE plane, which is
    what the bit-identity gate compares against; ``True`` is REFUSED, because a knob
    whose only value is the default is a knob that lies about being a decision.
    """

    paging: bool | None = None

    def __post_init__(self) -> None:
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
