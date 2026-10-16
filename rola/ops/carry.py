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
    "capture_bind",
    "capture_bytes",
    "carry",
    "census",
    "forced_word",
    "form",
    "ledger_bind",
    "plan_words",
    "plans_out",
    "probe_bind",
    "readout_accumulate",
    "sm_clock_ghz",
]

#: THE KERNEL'S WINDOW, in tokens: a kernel constant the intra and the facts pass share, never an argument.
WINDOW = list_model.WINDOW

#: THE PHASE LEDGER'S SHAPE: ``[ctas][warps][phases]``, the kernel's own `CarryPhase` order.
LEDGER_PHASES = ("landed", "maxima", "store", "readout", "drain", "fill", "fold", "wait")
LEDGER_WARPS = 8


def arm_key(carve: Carve) -> tuple[int, ...]:
    """The compiled body a carve selects: ``(fold_rows, fold_factors, read_rows, read_factors, channels,
    publish_regs)`` -- each side's warp tile of dense rows and its operand's level count, the channel extent and
    the publish route (``tools/gen_shards.py``'s ``ARM_KEY``)."""
    return (carve.fold.tile_rows, len(carve.fold.operand), carve.readout.tile_rows, len(carve.readout.operand),
            carve.channels, int(carve.publish_in_registers))


def _list_level(plan) -> int:
    return plan.lists[0] if plan.lists else -1


def carry(routes, values, facts, carve: Carve, unit: torch.Tensor, accumulate: str | None = None,
          forced: str | None = None, plans: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """THE LAUNCH: ``(num, check)`` -- ``num`` the readout's packed partials at each token's unit, ``check`` the
    skeleton's fp32 sum of the final state. ``facts`` is the facts pass's over this bundle, its read entries' offsets
    in this arm's read pool rows (:func:`form`). The readout accumulates in the device's form
    (:func:`readout_accumulate`); ``accumulate`` (``"fp32"`` or ``"fp16"``) is reserved: the gates' choice of form, so
    both forms are gated on any card (``tests/oracle/test_carry_readout.py``). Each window's plan is the planner rule's
    (docs/internals/carry/schedule.md#planner); ``forced`` (:func:`forced_word`) is reserved: the gates' and the
    measurements' choice of the plan a window takes where every chunk of it fits; ``plans`` (:func:`plans_out`) is
    reserved: the plan pass's output for a gate to read, else the call's own
    (docs/internals/carry/carry_api.md#carry). A grouped bundle's CTAs are its write heads', each reading its query
    heads' rows; ``num`` and ``unit`` are a query head's."""
    BH = routes.batch * routes.heads
    d_v = values.v.shape[-1]
    num = torch.zeros((BH, routes.tokens, d_v // 2), dtype=torch.int32, device=routes.device)
    check = torch.zeros(1, dtype=torch.float32, device=routes.device)
    extension().carry_forward(
        list(routes.amp_read), list(routes.amp_write), values.v, list(facts_ops.side_lists(facts, "write")),
        facts.list_exp, list(facts_ops.side_lists(facts, "read")), unit, num, check, list(carve.widths),
        list(carve.extents), carve.channels, _list_level(carve.fold), list(carve.fold.operand),
        _list_level(carve.readout), list(carve.readout.operand), list(arm_key(carve)),
        _ACCUMULATE[accumulate or readout_accumulate(routes.device)], forced_word(forced), plans)
    return num, check


def plan_words(heads: int = 1) -> int:
    """A window's plan record's 32-bit words at ``heads`` query heads a write head (`carry.cuh` `plan_words`): its plan
    word, then a 16-bit step word a step -- its read chunks (8 a head), its fold chunks (8), the retiring step -- in
    whole 16-byte copies."""
    steps = list_model.LIST_AT * heads + list_model.LIST_AT + 1
    return (1 + (steps + 1) // 2 + 3) // 4 * 4


def plans_out(routes, carve: Carve) -> torch.Tensor:
    """An uninitialized ``[BH, windows, box classes, plan_words(g)]`` int32 tensor for :func:`carry`'s plan records: a
    box class a pair of the two list levels' 32-digit groups (docs/internals/carry/schedule.md#pass)."""
    groups = 1
    for plan in (carve.readout, carve.fold):
        groups *= max(1, -(-carve.widths[plan.lists[0]] // 32)) if plan.lists else 1
    windows = -(-routes.tokens // WINDOW)
    return torch.empty((routes.batch * routes.write_heads, windows, groups, plan_words(routes.group)),
                       dtype=torch.int32, device=routes.device)


_ACCUMULATE = {"fp32": 0, "fp16": 1}
_MODES = {"rule": 0, "whole": 1, "halves": 2}


def forced_word(forced: str | None) -> int:
    """A FORCED PLAN FORM as the plan pass takes it (docs/internals/carry/schedule.md#planner): ``"mode/read/fold"``,
    the mode ``rule``, ``whole`` or ``halves`` and each chunk ``rule``, ``union`` or a multiple of 64 slots -- a window
    takes it where every chunk of it fits, else the rule's plan; ``None`` the rule's everywhere."""
    if forced is None:
        return -1
    mode, read, fold = forced.split("/")

    def chunk(name: str, union: int) -> int:
        if name in ("rule", "union"):
            return 0 if name == "rule" else union
        if int(name) % 64 or not 0 < int(name) // 64 < union:
            raise ValueError(f"a forced chunk is rule, union or a multiple of 64 slots below {64 * union}: {name}")
        return int(name) // 64

    return _MODES[mode] | chunk(read, 0xFF) << 2 | chunk(fold, 0xF) << 10


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
    thread limit, the binary version, and its form on this card (the area its plans place buffers in, then its pool
    rows: :class:`Form`)."""
    return [tuple(row) for row in extension().carry_census()]


class Form(NamedTuple):
    """An arm's pool rows on the current card (docs/internals/carry/plan.md#plan): a read and a write row's bytes --
    what the facts pass addresses the read entries' rows in."""

    read_row: int
    write_row: int


def form(key: tuple[int, ...]) -> Form:
    """The pool rows of the built arm ``key`` (:func:`arm_key`) on the current card, refusing a key no built arm
    carries."""
    for row in census():
        if tuple(row[1:7]) == tuple(key):
            return Form(*row[13:15])
    raise ValueError(f"no built carry arm has the key {tuple(key)}; this binary carries {arms()}")


def build_stamp() -> int:
    """The carry family's DEVICE build stamp, a fact only a built kernel produces (the window, the capability digest,
    the state's values and a list's boundary granularity, read back off a kernel this binary compiled)."""
    return extension().carry_build_stamp()


def ledger_bind(ledger: torch.Tensor | None) -> None:  #: reserved: the measurement instrument (docs/internals/carry/carry_kernel.md#phases)
    """Bind the phase ledger the next launches book into, ``[ctas, LEDGER_WARPS, len(LEDGER_PHASES)]`` int64 cycles,
    or ``None``."""
    extension().carry_ledger_bind(ledger)


def capture_bind(out: torch.Tensor | None) -> None:  #: reserved: the lab's window on a real call (docs/internals/carry/carry_api.md#carry-capture-bind)
    """Bind the host u8 tensor (:func:`capture_bytes` long) the next launch copies its operand block and grid into, or
    ``None``."""
    extension().carry_capture_bind(out)


def capture_bytes() -> int:
    """A capture's bytes in this binary: its byte count, boxes and batch-heads (int64 each), then the operand
    block."""
    return int(extension().carry_capture_bytes())


def probe_bind(probe: torch.Tensor | None) -> None:  #: reserved: the fold's oracle instrument (docs/internals/carry/fold.md#probe)
    """Bind the probe the next launches write their final state into, absolute fp32 ``[BH, N, d_v]`` by global leaf
    in level order, or ``None``."""
    extension().carry_probe_bind(probe)


def sm_clock_ghz() -> float | None:
    """The SM's effective clock under load, read off the device beside a measurement; None from a binary without the
    probe."""
    fn = getattr(extension(), "sm_clock_ghz", None)
    return None if fn is None else float(fn())
