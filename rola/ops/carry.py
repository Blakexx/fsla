# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CARRY, as a caller reaches it: one launch over a call's windows, from the bundle's rows and values, the facts
pass's lists, the den pass's units and the carve (docs/internals/carry/carry_api.md).

Symbols, each at first use: ``B`` batch, ``T`` = ``L`` tokens, ``H`` heads, ``BH`` = ``B H``, ``d_v`` the value
width, ``W`` the window (:data:`WINDOW`). The carve (:class:`rola.engine.types.Carve`) names the box -- an extent a
level and the channel extent -- and each side's list levels and operand levels; its arm key
(:func:`arm_key`) selects the compiled body, and the box's extents are runtime addressing. The readout adds packed
s16 pairs at each token's unit (``unit``, the den pass's exponent bytes) into ``num`` ``[BH, L, d_v / 2]`` int32.
"""
from __future__ import annotations

import torch

from rola.engine.facts import lists as list_model
from rola.engine.types import Carve
from rola.ops import facts as facts_ops
from rola.ops._ext import extension

__all__ = ["LEDGER_PHASES", "LEDGER_WARPS", "WINDOW", "arm_key", "arms", "build_stamp", "carry", "census",
           "ledger_bind", "sm_clock_ghz"]

#: THE KERNEL'S WINDOW, in tokens: a kernel constant the intra and the facts pass share, never an argument.
WINDOW = list_model.WINDOW

#: THE PHASE LEDGER'S SHAPE: ``[ctas][warps][phases]``, the kernel's own `CarryPhase` order.
LEDGER_PHASES = ("lists", "units", "publish", "readout", "drain", "fold")
LEDGER_WARPS = 8


def arm_key(carve: Carve) -> tuple[int, ...]:
    """The compiled body a carve selects: ``(fold_rows, fold_factors, read_rows, read_factors, channels,
    publish_regs)`` -- each side's warp tile of dense rows and its operand's level count, the channel extent and
    the publish route (``tools/gen_shards.py``'s ``ARM_KEY``)."""
    return (carve.fold.tile_rows, len(carve.fold.operand), carve.readout.tile_rows, len(carve.readout.operand),
            carve.channels, int(carve.publish_in_registers))


def _list_level(plan) -> int:
    return plan.lists[0] if plan.lists else -1


def _write_lists(lists: list_model.ListSet) -> list[torch.Tensor]:
    """The write side's lists as the kernel reads them: the 8-byte records (slot, 0, c_v, c_m) the slots are column 0
    of, the pass starts, the union tokens and counts."""
    slot = lists.slot
    records = slot.as_strided((*slot.shape, 4), (*slot.stride(), 1))
    return [records, lists.at, lists.union, lists.count]


def carry(routes, values, facts, carve: Carve, unit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """THE LAUNCH: ``(num, check)`` -- ``num`` the readout's packed partials at each token's unit, ``check`` the
    skeleton's fp32 sum of the final state (docs/internals/carry/carry_api.md#carry)."""
    BH = routes.batch * routes.heads
    d_v = values.v.shape[-1]
    num = torch.zeros((BH, routes.tokens, d_v // 2), dtype=torch.int32, device=routes.device)
    check = torch.zeros(1, dtype=torch.float32, device=routes.device)
    read = facts_ops.side_lists(facts, "read")
    extension().carry_forward(
        list(routes.amp_read), list(routes.amp_write), values.v, _write_lists(facts_ops.side_lists(facts, "write")),
        [read.slot, read.at, read.union, read.count], unit, num, check, list(carve.widths), list(carve.extents),
        carve.channels, _list_level(carve.fold), list(carve.fold.operand), _list_level(carve.readout),
        list(carve.readout.operand), list(arm_key(carve)))
    return num, check


def arms() -> list[tuple[int, ...]]:
    """The arms THIS binary carries, each its :func:`arm_key`: the binary's own answer, never the source tree's."""
    return [tuple(row) for row in extension().carry_arms()]


def census() -> list[tuple[int, ...]]:
    """Each built arm's device census: its index and key, then registers, local bytes, shared bytes, the block's
    thread limit and the binary version."""
    return [tuple(row) for row in extension().carry_census()]


def build_stamp() -> int:
    """The carry family's DEVICE build stamp, a fact only a built kernel produces (the window, the capability digest,
    the state's values and the pass's slots, read back off a kernel this binary compiled)."""
    return extension().carry_build_stamp()


def ledger_bind(ledger: torch.Tensor | None) -> None:  #: reserved: the measurement instrument (docs/internals/carry/carry_kernel.md#phases)
    """Bind the phase ledger the next launches book into, ``[ctas, LEDGER_WARPS, len(LEDGER_PHASES)]`` int64 cycles,
    or ``None``."""
    extension().carry_ledger_bind(ledger)


def sm_clock_ghz() -> float | None:
    """The SM's effective clock under load, read off the device beside a measurement; None from a binary without the
    probe."""
    fn = getattr(extension(), "sm_clock_ghz", None)
    return None if fn is None else float(fn())
