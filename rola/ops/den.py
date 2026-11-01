# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE DEN PASS, as a caller reaches it: one launch over a call's windows beside the carry, from the bundle's rows and
the facts pass's lists, each token's window-aligned denominator and readout bound (docs/internals/den/den_api.md).

Symbols, each at first use: ``B`` batch, ``L`` tokens, ``H_w`` write heads, ``g`` query heads a write head,
``BH = B H_w``, ``W`` the window (the device's, :func:`rola.ops.window.window_of`), level ``S`` the read side's list level
(:func:`rola.routing.factors.carry_level_s`), level ``R`` the other. ``den`` ``[BH g, L]`` fp32 is each query head's
token's read leaf product against the write mass of the windows before -- the inter term's denominator in the kernels'
units (the reader's per-token factor left out, as the carry's ``num`` leaves it; `tests/oracle/test_den_vs_oracle.py`
states the factor) -- and ``num`` the same product against the mass of ``F = m s``, so ``num / den`` bounds every
readout channel.
"""
from __future__ import annotations

import torch

from rola.ops import facts as facts_ops
from rola.ops._ext import extension
from rola.ops.window import window_of

__all__ = ["LEDGER_PHASES", "LEDGER_WARPS", "MAX_HEADS", "ROWS", "SPAN", "den", "ledger_bind", "serves", "tiles"]

#: THE PHASE LEDGER'S SHAPE: ``[ctas][warps][phases]``, the kernel's own `DenPhase` order.
LEDGER_PHASES = ("stage", "publish", "den", "drain", "writes", "wait")
LEDGER_WARPS = 16
#: A CTA'S TILE: level-S rows x level-R digits; the query heads a write head the accumulator holds.
ROWS, SPAN, MAX_HEADS = 64, 64, 4


def serves(routes) -> str | None:
    """Why the den pass refuses this bundle, or ``None``: depth two, level S and R at multiples of the tile, and either
    the read side listing S with the write side listing R, or both sides dense."""
    widths = routes.topology.padded_widths
    if len(widths) != 2:
        return f"depth {len(widths)}: the den pass is built for depth two"
    level_s = routes.level_s
    read_sparse, write_sparse = routes.sparse
    if level_s < 0 and any(read_sparse):
        return "the read side lists at no single level"
    ls = level_s if level_s >= 0 else 1
    lr = 1 - ls
    if widths[ls] % ROWS or widths[lr] % SPAN:
        return f"widths {tuple(widths)}: level S a multiple of {ROWS} digits and level R of {SPAN}"
    dense = not any(read_sparse) and not any(write_sparse)
    alt = read_sparse == tuple(l == ls for l in range(2)) and write_sparse == tuple(l == lr for l in range(2))
    if not (dense or alt):
        return f"sparse {routes.sparse}: the read side listing level S with the write side listing level R, or both dense"
    if routes.group > MAX_HEADS:
        return f"{routes.group} query heads a write head: the accumulator holds {MAX_HEADS}"
    if routes.group * window_of(routes.device) > 1 << PLACE_BITS:
        return (f"{routes.group} query heads a write head at a window of {window_of(routes.device)}: a slot's place "
                f"takes {PLACE_BITS} bits")
    return None


def tiles(routes) -> int:
    """CTAs a write head: the tile grid over (level S rows, level R digits)."""
    widths = routes.topology.padded_widths
    ls = routes.level_s if routes.level_s >= 0 else 1
    return (widths[ls] // ROWS) * (widths[1 - ls] // SPAN)


def den(routes, facts: facts_ops.Facts) -> tuple[torch.Tensor, torch.Tensor]:
    """``(den, num)`` ``[BH g, L]`` fp32 over the whole call, zeroed here and added into by the pass's CTAs."""
    why = serves(routes)
    if why is not None:
        raise NotImplementedError(f"the den pass does not serve this bundle: {why}")
    widths = list(routes.topology.padded_widths)
    ls = routes.level_s if routes.level_s >= 0 else 1
    dense = not any(routes.sparse[0]) and not any(routes.sparse[1])
    BH, g = routes.batch * routes.write_heads, routes.group
    L = routes.tokens
    device = routes.device
    out = torch.zeros((BH * g, L), dtype=torch.float32, device=device)
    bound = torch.zeros_like(out)
    empty = [torch.empty(0, device=device)] * 5
    read_lists = empty if dense else list(facts_ops.side_lists(facts, "read"))
    write_lists = empty if dense else list(facts_ops.side_lists(facts, "write"))
    extension().den_forward(list(routes.amp_read), list(routes.amp_write), read_lists, write_lists, facts.list_exp,
                            facts.gains16, facts.gains_exp.reshape(BH, -1), out, bound, widths, ls,
                            -1 if dense else ls, -1 if dense else 1 - ls, g)
    return out, bound


def ledger_bind(ledger: torch.Tensor | None) -> None:
    """Bind the phase ledger the next launches add into, ``[ctas, LEDGER_WARPS, len(LEDGER_PHASES)]`` int64 cycles,
    or ``None``."""
    extension().den_ledger_bind(ledger)


#: a slot's place (query head x token) is the den pass's 14-bit field: at W = 1,024 four query heads fit
PLACE_BITS = 14
