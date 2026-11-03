# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CARVE PLANNER -- producer v1 and the check every carve passes: the carry's box and each side's operation
plan, from the declaration.

The planner is a pure fold over the declaration (the shipped widths, each level's two sparse flags, ``d_v``), so a
better planner replaces this one without touching a kernel: every kernel that depends on the geometry takes the
:class:`~rola.engine.types.Carve` it returns and refuses one it was not built for (docs/internals/engine/carve.md).
"""
from __future__ import annotations

from itertools import product

from rola.engine.types import Carve, SidePlan

#: THE BOX'S STATE, the architecture's register budget for it: 8 warps x 32 lanes x 128 fp32 registers a lane. Its
#: leaves are the state over the channel extent, and no level's extent exceeds the square root of 1,024 leaves.
WARPS = 8
BOX_VALUES = WARPS * 32 * 128
CHANNELS = 32
CAP = 32
#: A SIDE'S DENSE EXTENT is an MMA tile of rows at least.
MIN_DENSE = 16

SIDES = ("read", "write")


def _log2(n: int) -> int:
    return n.bit_length() - 1


def tags(read_sparse, write_sparse) -> tuple[str, ...]:
    """Each level's tag, its read flag then its write flag: ``S`` sparse, ``D`` dense."""
    return tuple(("S" if r else "D") + ("S" if w else "D") for r, w in zip(read_sparse, write_sparse))


def _allocate(widths, level_tags, total: int, cap: int) -> list[int] | None:
    """Producer v1's extents as exponents, or None when the box cannot be filled under the cap: DD levels as much as
    fits, spread evenly; SD and DS by turns, SD first, each concentrated on its widest level; SS last."""
    room = [min(_log2(w), _log2(cap)) for w in widths]
    bits = [0] * len(widths)
    left = total

    def give(level: int) -> bool:
        nonlocal left
        if left == 0 or bits[level] == room[level]:
            return False
        bits[level] += 1
        left -= 1
        return True

    def widest(tag: str) -> int | None:
        open_ = [lv for lv, t in enumerate(level_tags) if t == tag and bits[lv] < room[lv]]
        return max(open_, key=lambda lv: (room[lv], -lv)) if open_ else None

    dd = [lv for lv, t in enumerate(level_tags) if t == "DD"]
    while left and any(give(lv) for lv in sorted(dd, key=lambda lv: bits[lv])):
        pass
    while left:
        moved = False
        for tag in ("SD", "DS"):
            lv = widest(tag)
            moved |= lv is not None and give(lv)
        if not moved:
            break
    while left and (lv := widest("SS")) is not None:
        give(lv)
    return bits if left == 0 else None


def _side(lists, operand, extents, leaves: int, coef=None) -> SidePlan:
    """A side's plan. A side with a sparse level lists it and forms its rows from the dense ones; a side with none
    takes one dense level as its rows and the other as its coefficients, named in that order -- its coefficients the
    other side's list level where that side lists one of its levels (so the two sides' warps split alike), else its
    first dense level (both sides of a dense-dense declaration alike) (docs/internals/engine/carve.md#sides)."""
    if not lists and len(operand) == 2:
        c = coef if coef in operand else operand[0]
        rows = operand[1] if c == operand[0] else operand[0]
        tile_rows = extents[rows]
        return SidePlan(lists=(), operand=(rows, c), tile_lists=leaves // WARPS // tile_rows, tile_rows=tile_rows)
    rows = 1
    for lv in operand:
        rows *= extents[lv]
    tile_rows = min(rows, leaves // WARPS)
    return SidePlan(lists=tuple(lists), operand=tuple(operand), tile_lists=leaves // WARPS // tile_rows,
                    tile_rows=tile_rows)


def _warps_of(plan: SidePlan, extents) -> frozenset:
    """The box's leaves as each warp of this side owns them, one frozen set a warp: a warp's groups (its list levels'
    values, or a side with no list level's coefficient values) by its rows."""
    groups, row_levels = (plan.lists, plan.operand) if plan.lists else (plan.operand[1:], plan.operand[:1])
    rows = 1
    for lv in row_levels:
        rows *= extents[lv]
    owned: dict[int, list] = {}
    for leaf in product(*(range(e) for e in extents)):
        li = ri = 0
        for lv in groups:
            li = li * extents[lv] + leaf[lv]
        for lv in row_levels:
            ri = ri * extents[lv] + leaf[lv]
        warp = (li // plan.tile_lists) * (rows // plan.tile_rows) + ri // plan.tile_rows
        owned.setdefault(warp, []).append(leaf)
    return frozenset(frozenset(v) for v in owned.values())


def _build(widths, read_sparse, write_sparse, extents, channels: int) -> Carve:
    D = len(widths)
    leaves = 1
    for e in extents:
        leaves *= e
    write_lists = [lv for lv in range(D) if write_sparse[lv]]
    read_lists = [lv for lv in range(D) if read_sparse[lv]]
    fold = _side(write_lists, [lv for lv in range(D) if not write_sparse[lv]], extents, leaves,
                 coef=read_lists[0] if len(read_lists) == 1 else None)
    readout = _side(read_lists, [lv for lv in range(D) if not read_sparse[lv]], extents, leaves,
                    coef=write_lists[0] if len(write_lists) == 1 else None)
    return Carve(widths=tuple(widths), extents=tuple(extents), channels=channels, fold=fold, readout=readout,
                 publish_in_registers=_warps_of(fold, extents) == _warps_of(readout, extents))


def _declared(widths, read_sparse, write_sparse, d_v: int) -> None:
    if len(read_sparse) != len(widths) or len(write_sparse) != len(widths):
        raise ValueError(f"a carve needs both sides' flags a level: widths {widths}, read {read_sparse}, "
                         f"write {write_sparse}")
    for w in widths:
        if w < MIN_DENSE or w & (w - 1):
            raise ValueError(f"a shipped level width is a power of two at least {MIN_DENSE}, got {w}")
    for side, flags in zip(SIDES, (read_sparse, write_sparse)):
        if all(flags):
            raise ValueError(f"the {side} side declares every level sparse; every side has a dense level")
    if d_v < CHANNELS or d_v % CHANNELS:
        raise ValueError(f"d_v is a multiple of the box's {CHANNELS} channels, got {d_v}")


def plan_carve(widths, read_sparse, write_sparse, d_v: int) -> Carve:
    """Producer v1: the box from the tags under the per-level cap; when the box cannot be filled under it, the box's
    leaves halve and its channel extent doubles (the same state) before the cap lifts to the widths."""
    _declared(widths, read_sparse, write_sparse, d_v)
    level_tags = tags(read_sparse, write_sparse)
    channels = CHANNELS
    while channels <= d_v and d_v % channels == 0:
        bits = _allocate(widths, level_tags, _log2(BOX_VALUES // channels), CAP)
        if bits is not None:
            carve_ = _build(widths, read_sparse, write_sparse, [1 << b for b in bits], channels)
            check_carve(carve_, widths, read_sparse, write_sparse, d_v)
            return carve_
        channels *= 2
    channels //= 2
    bits = [_log2(w) for w in widths]
    while sum(bits) > _log2(BOX_VALUES // channels):
        bits[max(range(len(bits)), key=lambda lv: (bits[lv], lv))] -= 1
    carve_ = _build(widths, read_sparse, write_sparse, [1 << b for b in bits], channels)
    check_carve(carve_, widths, read_sparse, write_sparse, d_v)
    return carve_


def check_carve(carve: Carve, widths, read_sparse, write_sparse, d_v: int) -> None:
    """Refuse a carve the declaration does not admit, naming what is wrong (the kernels refuse the same)."""
    _declared(widths, read_sparse, write_sparse, d_v)
    if carve.widths != tuple(widths) or len(carve.extents) != len(widths):
        raise ValueError(f"the carve is for widths {carve.widths}, the declaration's are {tuple(widths)}")
    for w, e in zip(widths, carve.extents):
        if e < 1 or e & (e - 1) or w % e:
            raise ValueError(f"an extent is a power of two dividing its level's width: {carve.extents} of {widths}")
    if carve.channels < CHANNELS or carve.channels & (carve.channels - 1) or d_v % carve.channels:
        raise ValueError(f"the channel extent is a power of two dividing d_v {d_v}, got {carve.channels}")
    if carve.leaves * carve.channels > BOX_VALUES:
        raise ValueError(f"the box holds {carve.leaves} x {carve.channels} values, over the state's {BOX_VALUES}")
    want = _build(widths, read_sparse, write_sparse, carve.extents, carve.channels)
    for side, plan in (("fold", carve.fold), ("readout", carve.readout)):
        if plan != getattr(want, side):
            raise ValueError(f"the {side}'s plan {plan} is not the box's {getattr(want, side)}")
        if plan.tile_rows < MIN_DENSE:
            raise ValueError(f"the {side}'s warps own {plan.tile_rows} dense rows, under an MMA tile of {MIN_DENSE}")
    if carve.publish_in_registers != want.publish_in_registers:
        raise ValueError("the carve's publish route disagrees with its warps' ownership")
