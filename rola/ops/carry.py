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

from typing import NamedTuple

import torch

from rola.engine.facts import lists as list_model
from rola.engine.types import Carve
from rola.ops import facts as facts_ops
from rola.ops._ext import extension

__all__ = [
    "LEDGER_PHASES",
    "LEDGER_WARPS",
    "WINDOW",
    "Form",
    "arm_key",
    "arms",
    "build_stamp",
    "carry",
    "census",
    "form",
    "ledger_bind",
    "probe_bind",
    "readout_accumulate",
    "sm_clock_ghz",
]

#: THE KERNEL'S WINDOW, in tokens: a kernel constant the intra and the facts pass share, never an argument.
WINDOW = list_model.WINDOW

#: THE PHASE LEDGER'S SHAPE: ``[ctas][warps][phases]``, the kernel's own `CarryPhase` order.
LEDGER_PHASES = ("landed", "maxima", "store", "readout", "drain", "fill", "fold")
LEDGER_WARPS = 8


def arm_key(carve: Carve) -> tuple[int, ...]:
    """The compiled body a carve selects: ``(fold_rows, fold_factors, read_rows, read_factors, channels,
    publish_regs)`` -- each side's warp tile of dense rows and its operand's level count, the channel extent and
    the publish route (``tools/gen_shards.py``'s ``ARM_KEY``)."""
    return (carve.fold.tile_rows, len(carve.fold.operand), carve.readout.tile_rows, len(carve.readout.operand),
            carve.channels, int(carve.publish_in_registers))


def _list_level(plan) -> int:
    return plan.lists[0] if plan.lists else -1


def carry(routes, values, facts, carve: Carve, unit: torch.Tensor,
          accumulate: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """THE LAUNCH: ``(num, check)`` -- ``num`` the readout's packed partials at each token's unit, ``check`` the
    skeleton's fp32 sum of the final state. ``facts`` is the facts pass's over this bundle in this arm's :func:`form`
    on the device's card: its lists' passes are the arm's readout pass and fold unit, and the read entries' offsets its
    read pool row's. The readout accumulates in the device's form (:func:`readout_accumulate`); ``accumulate``
    (``"fp32"`` or ``"fp16"``) is reserved: the gates' choice of form, so both forms are gated on any card
    (``tests/oracle/test_carry_readout.py``; docs/internals/carry/carry_api.md#carry)."""
    BH = routes.batch * routes.heads
    d_v = values.v.shape[-1]
    num = torch.zeros((BH, routes.tokens, d_v // 2), dtype=torch.int32, device=routes.device)
    check = torch.zeros(1, dtype=torch.float32, device=routes.device)
    extension().carry_forward(
        list(routes.amp_read), list(routes.amp_write), values.v, list(facts_ops.side_lists(facts, "write")),
        facts.list_exp, list(facts_ops.side_lists(facts, "read")), unit, num, check, list(carve.widths),
        list(carve.extents), carve.channels, _list_level(carve.fold), list(carve.fold.operand),
        _list_level(carve.readout), list(carve.readout.operand), list(arm_key(carve)),
        _ACCUMULATE[accumulate or readout_accumulate(routes.device)])
    return num, check


_ACCUMULATE = {"fp32": 0, "fp16": 1}


def readout_accumulate(device: torch.device | None = None) -> str:
    """The readout's MMA accumulation on ``device`` (the current one by default): ``"fp16"`` where its fp32-accumulate
    MMA runs slower than its fp16-accumulate one by its rates (a GeForce card), else ``"fp32"``
    (docs/internals/carry/carry_api.md#carry-readout-accumulate)."""
    index = torch.cuda.current_device() if device is None or device.index is None else device.index
    return "fp16" if extension().carry_readout_accumulate(index) else "fp32"


def arms() -> list[tuple[int, ...]]:
    """The arms THIS binary carries, each its :func:`arm_key`: the binary's own answer, never the source tree's."""
    return [tuple(row) for row in extension().carry_arms()]


def census() -> list[tuple[int, ...]]:
    """Each built arm's device census: its index and key, then registers, local bytes, shared bytes, the block's
    thread limit, the binary version, and its form on this card (:class:`Form`'s fields after the shared bytes)."""
    return [tuple(row) for row in extension().carry_census()]


class Form(NamedTuple):
    """An arm's form on the current card (docs/internals/carry/plan.md#forms): the image's parts, a readout pass's and
    the fold's pass's slots (``unit_slots``: a fold unit's, or the whole window where the units are resident,
    plan.md#resident), a read and a write pool row's bytes -- what the facts pass sizes its passes and offsets by."""

    parts: int
    pass_slots: int
    unit_slots: int
    read_row: int
    write_row: int


def form(key: tuple[int, ...]) -> Form:
    """The form of the built arm ``key`` (:func:`arm_key`) on the current card, refusing a key no built arm carries."""
    for row in census():
        if tuple(row[1:7]) == tuple(key):
            return Form(*row[12:17])
    raise ValueError(f"no built carry arm has the key {tuple(key)}; this binary carries {arms()}")


def build_stamp() -> int:
    """The carry family's DEVICE build stamp, a fact only a built kernel produces (the window, the capability digest,
    the state's values and a list's staged entries, read back off a kernel this binary compiled)."""
    return extension().carry_build_stamp()


def ledger_bind(ledger: torch.Tensor | None) -> None:  #: reserved: the measurement instrument (docs/internals/carry/carry_kernel.md#phases)
    """Bind the phase ledger the next launches book into, ``[ctas, LEDGER_WARPS, len(LEDGER_PHASES)]`` int64 cycles,
    or ``None``."""
    extension().carry_ledger_bind(ledger)


def probe_bind(probe: torch.Tensor | None) -> None:  #: reserved: the fold's oracle instrument (docs/internals/carry/fold.md#probe)
    """Bind the probe the next launches write their final state into, absolute fp32 ``[BH, N, d_v]`` by global leaf
    in level order, or ``None``."""
    extension().carry_probe_bind(probe)


def sm_clock_ghz() -> float | None:
    """The SM's effective clock under load, read off the device beside a measurement; None from a binary without the
    probe."""
    fn = getattr(extension(), "sm_clock_ghz", None)
    return None if fn is None else float(fn())
