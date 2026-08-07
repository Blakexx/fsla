# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""``BC`` IS DERIVED. The planner is one closed form over launch-static inputs.

    ``BC = min(subtree_ceiling, box_ceiling, fill_ceiling)``, clamped down to a
    compiled row.

``BC`` is the number of leaves one owner block owns -- the CTA's resident state tile,
and the only launch geometry left to decide. There is nothing to choose it AMONG: the
consumer instantiates one executor, the token tile is the fixed
:data:`GROUP_TOKENS` quantum, and every remaining axis is a model fact. So the planner
does not score candidates against a cost model and does not read a routing statistic.
It evaluates three ceilings, each a property of the topology or the machine, and takes
the largest compiled block that clears all three.

Each ceiling answers one question, and none of them is a threshold on a measured
distribution:

``subtree_ceiling``
    How large can a block be before it OWNS LEAVES FROM INDEPENDENT SUBTREES?
``box_ceiling``
    How large a block does the shared-memory carve fit, once the rings are reserved?
``fill_ceiling``
    How large a block still leaves enough CTAs to fill the machine?

The clamp is DOWNWARD onto the dispatch table's own ``BC`` axis: the formula picks
among rows the binary carries, and a derived value with no row is a structural error
rather than an approximation. What each term is derived from -- and what would falsify
it -- is stated at its own definition.

Symbols: ``widths`` the level widths ``b_0..b_{D-1}``, MSB first; ``D = len(widths)``
the level count; ``N = prod(widths)`` the leaf capacity; ``BH = batch * heads`` the
cohort multiplier; ``d_v`` the value width; ``BC`` leaves per owner block.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import prod

#: The hardware ``mma.sync`` k-extent on the sm80-class bf16 path (``kMmaKQuantum``).
#: A SHAPE of the instruction the kernel issues, never a threshold.
MMA_K_QUANTUM = 16

#: The executor's work unit: one MMA m-quantum of gathered tokens. The token-tile axis
#: is dead -- groups are always full and there are ``ceil(gathered / 16)`` of them.
GROUP_TOKENS = MMA_K_QUANTUM

#: The ``BC`` ladder the consumer instantiates, for callers planning with NO extension
#: present (CPU-only construction-time admissibility). A MIRROR of the dispatch table,
#: and a mirror needs a gate: ``tests/unit/test_dispatch_table.py`` pins this tuple to
#: the table's own axis values, read off the built binary. The runtime derivation reads
#: the table itself and never this tuple.
SUPPORTED_BC = (16, 64)

#: Shared memory reserved per CTA for the schedule and cook RINGS, in bytes, before the
#: state slab and the staging carve are priced against the budget. The endgame executor
#: on this tree stages at the group top and holds no ring, so the reservation is ZERO
#: and the box term prices the carve alone. It is named rather than absent because the
#: ring era's budget plugs in HERE and nowhere else: rings are what replace co-residency
#: as the latency-hiding mechanism, so their depth is bought with exactly the shared
#: memory a larger ``BC`` would otherwise spend on the state slab, and this constant is
#: that trade written down.
RING_RESERVE_BYTES = 0

#: How many waves of CTAs the grid must supply before a larger block is preferred.
#:
#: A launch of ``w`` waves wastes at most one partially-filled wave, i.e. at most
#: ``1/ceil(w)`` of the machine; the marginal wave buys ``1/(w(w+1))``, so the return
#: falls off as ``1/w^2`` while every doubling of ``BC`` halves the ``N/BC``-way split-K
#: reduction and doubles the leaves the per-tile fixed work amortizes over. Two waves is
#: where those meet: it caps the tail loss at a quarter of the launch and costs one
#: doubling of ``BC`` at most. It is a target, not a floor -- a topology that cannot
#: reach it is planned at the smallest block rather than refused, because when no block
#: fills the machine the smallest one fills it best.
WAVES_TARGET = 2


# ---------------------------------------------------------------------------
# The machine
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MachineFacts:
    """Tier H: what the derivation is allowed to know about the machine.

    Measured once per ``(device, install)`` and carried, never queried ambiently from
    inside the formula -- which is what keeps :func:`derive_bc` a pure function of
    recorded inputs, so the same inputs give the same plan and a CPU-only gate can
    supply a synthetic machine instead of a mock.

    ``arm_smem_bytes`` is the dispatch table's own ``{(BC, d_v, D, decay): bytes}``
    map: the ``constexpr`` carve of each compiled arm, read off the built binary after
    the staleness join. It is the box term's input, and taking it from the table is
    what makes the box arithmetic the KERNEL's arithmetic rather than a model of it.
    """

    sm_count: int
    max_dynamic_smem_bytes: int
    arm_smem_bytes: Mapping[tuple[int, int, int, bool], int]

    @classmethod
    def for_device(cls, device) -> MachineFacts:
        """The facts for one CUDA device, from the driver and the built extension."""
        import torch

        from rola._build_config import BUILD_CONFIG
        from rola.ops._ext import extension
        from rola.planner.dispatch_table import DispatchTable

        ext = extension()
        table = DispatchTable.from_extension(ext=ext, ratified_keys=BUILD_CONFIG.get("shards"))
        return cls(
            sm_count=torch.cuda.get_device_properties(torch.device(device)).multi_processor_count,
            max_dynamic_smem_bytes=int(ext.rola_max_dynamic_smem_bytes()),
            arm_smem_bytes={row.key: int(row.smem_bytes) for row in table.rows})

    def compiled_bc(self, d_v: int, D: int, decay: bool) -> tuple[int, ...]:
        """Every ``BC`` this binary carries an arm for at this ``(d_v, D, decay)``."""
        return tuple(sorted(
            BC for (BC, dv, levels, arm) in self.arm_smem_bytes
            if dv == d_v and levels == D and arm == bool(decay)))


# ---------------------------------------------------------------------------
# The three ceilings
# ---------------------------------------------------------------------------


def subtree_ceiling(widths: tuple[int, ...]) -> int:
    """``b_{D-1}`` -- the widest block that stays inside ONE subtree.

    Leaves are the MSB-first mixed-radix enumeration of the level digits, so an
    aligned block of ``BC <= b_{D-1}`` leaves shares every digit above the innermost
    level: it is a slice of ONE parent's children. A block wider than ``b_{D-1}``
    necessarily spans whole innermost levels under two or more distinct digit tuples
    above -- it unions leaves of INDEPENDENT SUBTREES.

    That union is what makes the block wrong rather than merely large. Routing selects
    subtrees; two independent ones have unrelated supports, so the union's candidate
    bitmap is their OR and the block is visited whenever ANY of them is live, paying
    the whole ``BC`` k-extent for whichever subset was actually routed. Widening past
    the tree's own branching point therefore buys amortization with visits, and buys
    it at a rate the topology -- not the workload -- fixes.

    A property of the tree, evaluated on ``widths`` alone. FALSIFIED by a measurement
    showing a block spanning independent subtrees retiring no more work than the
    subtree-aligned blocks it replaces at the same realized support.
    """
    return int(widths[-1])


def box_admitted(machine: MachineFacts, d_v: int, D: int, decay: bool) -> tuple[int, ...]:
    """The compiled ``BC`` values whose carve fits shared memory, with the rings paid.

    ONE inequality, and it is the launch's own::

        arm_smem_bytes[(BC, d_v, D, decay)] + RING_RESERVE_BYTES
            <= max_dynamic_smem_bytes

    the ``cudaFuncAttributeMaxDynamicSharedMemorySize`` admission
    (:meth:`~rola.planner.dispatch_table.DispatchRow.admitted_by`) with the ring
    reservation added to the demand. A ``BC`` this refuses is one whose launch the
    driver would refuse, so the box is a FEASIBILITY ceiling and not a preference:
    there is no candidate above it to prefer.

    **``BC`` is what the budget is spent on.** The state slab is ``BC * (d_v + 4)``
    fp32 words and the digit, ``WT``, fragment and decay-panel buffers all scale with
    ``BC`` too, so the carve is affine in ``BC`` with a large slope -- which is why the
    per-CTA budget, and not co-residency, is the term that bounds the block. Residency
    does not appear: the endgame executor hides latency INSIDE one CTA through the item
    stream, so a second resident CTA is not what the shared memory is for. When the
    rings land, :data:`RING_RESERVE_BYTES` moves off zero and this is the inequality
    that trades their depth against the slab.

    Empty is a REFUSAL, not a fallback: a machine that will not grant a block enough
    shared memory for any compiled arm has no launch, so it has no plan.
    """
    budget = int(machine.max_dynamic_smem_bytes) - RING_RESERVE_BYTES
    admitted = tuple(BC for BC in machine.compiled_bc(d_v, D, decay)
                     if machine.arm_smem_bytes[(BC, d_v, D, bool(decay))] <= budget)
    if not admitted:
        carved = {BC: machine.arm_smem_bytes[(BC, d_v, D, bool(decay))]
                  for BC in machine.compiled_bc(d_v, D, decay)}
        raise ValueError(
            f"no compiled arm fits shared memory at d_v={d_v}, D={D}, decay={bool(decay)}: "
            f"the carves are {carved} bytes, the rings reserve {RING_RESERVE_BYTES}, and "
            f"this device grants a block {machine.max_dynamic_smem_bytes}. The launch "
            "would be refused by the driver, so there is no plan for this geometry on "
            "this machine.")
    return admitted


def fill_ceiling(N: int, BH: int, sm_count: int) -> int:
    """``BH * N / (sm_count * WAVES_TARGET)`` -- the block a full grid still admits.

    The launch is one CTA per ``(batch, head, owner)``, i.e. ``BH * N / BC`` CTAs, and
    ``BC`` is the only term that shrinks it. Requiring :data:`WAVES_TARGET` waves of
    ``sm_count`` CTAs rearranges to this bound exactly.

    It is a TARGET and it saturates. A topology whose whole grid is smaller than the
    machine cannot be made to fill it by any ``BC``, and refusing to plan there would
    refuse every small cell the gates run on; the derivation therefore falls back to
    the smallest compiled block, which is the one that supplies the most CTAs. The
    guard's job is to stop a large block from starving a grid that WOULD otherwise
    have filled the machine, and that is exactly what it does at every size.

    ``BH`` is launch-static -- it is the batch and head count, not a routing statistic
    -- so this term cannot drift within a launch and cannot be stale.
    """
    if sm_count < 1:
        raise ValueError(f"sm_count must be >= 1, got {sm_count}")
    return max(1, (int(BH) * int(N)) // (int(sm_count) * WAVES_TARGET))


# ---------------------------------------------------------------------------
# Alignment, and the ladder
# ---------------------------------------------------------------------------


def feasible_bc(widths: tuple[int, ...], *,
                instantiated: tuple[int, ...] = SUPPORTED_BC,
                pinned: int | None = None) -> tuple[int, ...]:
    """``{BC in instantiated : owner_geometry(widths, BC) admits}``, ascending.

    ALIGNMENT is the filter, and it is not a formality: an owner block must be exactly
    one rectangle -- fixed digits, an aligned sub-range of the boundary level, a full
    span below it -- and a topology whose narrow trailing levels cannot compose one at a
    given ``BC`` has no owner geometry there at all. ``widths = (4, 6)`` admits neither
    instantiated block, because 16 does not divide its 24 leaves.

    ``pinned`` is ``state_block``, the EXPERT OVERRIDE: it shrinks the set to itself and
    is validated against alignment exactly like any other member, so an inadmissible pin
    is an error on the line that wrote it rather than a plan-time surprise. It is NOT
    filtered against ``instantiated``: a caller pinning a geometry off the ladder is
    taking responsibility for supplying a kernel for it.
    """
    from rola.planner.schedule import owner_geometry

    candidates = (int(pinned),) if pinned is not None else tuple(sorted(instantiated))
    out = []
    for BC in candidates:
        try:
            owner_geometry(tuple(widths), BC)
        except ValueError:
            continue
        out.append(BC)
    if not out:
        raise ValueError(
            f"no feasible BC for widths={tuple(widths)}: none of {candidates} names a "
            "BC-aligned owner rectangle (the block must be exactly one rectangle -- "
            "fixed digits x an aligned sub-range of the boundary level x a full span "
            "below it). This is a TOPOLOGY defect, not a runtime condition: permute a "
            "wider level innermost, or instantiate a BC the topology admits.")
    return tuple(out)


def largest_feasible_bc(widths: tuple[int, ...], **kwargs) -> int:
    """The largest aligned ``BC`` on the instantiated ladder -- a TOPOLOGY fact.

    What a caller building an explicit plan asks for when it wants the widest block the
    geometry admits. It is not the derivation: :func:`derive_bc` applies three ceilings
    this function knows nothing about.
    """
    return feasible_bc(widths, **kwargs)[-1]


def quanta_per_owner(BC: int) -> int:
    """``QPO = ceil(BC / MMA_K_QUANTUM)`` -- the dense k-quantum count of one block.

    Every quanta-shaped quantity goes through this function, so a ``BC``-derived literal
    cannot exist. The ``ceil`` form is kept although the geometry validator guarantees
    ``MMA_K_QUANTUM | BC``, because restating that guarantee as a division would make
    this expression depend on it.
    """
    return -(-int(BC) // MMA_K_QUANTUM)


# ---------------------------------------------------------------------------
# The derivation
# ---------------------------------------------------------------------------


def derive_bc(topology, machine: MachineFacts, *, BH: int, decay: bool,
              pinned: int | None = None) -> int:
    """THE ``BC`` of one launch, and the whole of the planner's decision.

    ``topology`` supplies ``widths`` and ``d_v`` as ONE model fact rather than as a
    destructured set that can be assembled inconsistently. ``machine`` is tier H.
    ``BH`` and ``decay`` are launch-static. Nothing here reads a tensor, issues a
    reduction, allocates, or touches the routing -- which is why a plan cannot go stale
    within a launch and why there is nothing left to calibrate.

    ``pinned`` is the expert override (``PlanOverrides.state_block``). It REPLACES the
    derivation rather than biasing it, and it is still alignment-validated, which is
    what makes it a pin over a derived axis and not an escape from one.

    The three ceilings are independent and their ``min`` is not a tie-break: each one is
    a bound that holds on its own, so the largest admissible block clearing all three is
    the answer, and the clamp is downward onto the compiled ladder because a ``BC`` with
    no row cannot launch.

    **TWO OF THE FOUR CONSTRAINTS REFUSE AND TWO SATURATE, and the split is what each
    one MEANS.** :func:`box_admitted` and :func:`feasible_bc` are FEASIBILITY: an arm
    that does not fit shared memory and a ``BC`` that names no owner rectangle cannot be
    launched at all, so an empty set is a loud error and there is nothing below it to
    fall back to. :func:`subtree_ceiling` and :func:`fill_ceiling` are EFFICIENCY: the
    kernel runs correctly beneath them, they bound how wide a block should be rather
    than whether one exists, and the smallest compiled block is the closest the ladder
    comes to satisfying them. So when both ceilings fall below the ladder's floor -- a
    topology whose innermost level is narrower than one MMA quantum, a grid too small to
    fill any machine -- the answer is that floor, not a refusal to plan.
    """
    widths = tuple(topology.widths)
    d_v = int(topology.d_v)
    D = len(widths)
    admitted = box_admitted(machine, d_v, D, bool(decay))
    if pinned is not None:
        return feasible_bc(widths, instantiated=admitted, pinned=int(pinned))[-1]
    ladder = feasible_bc(widths, instantiated=admitted)
    ceiling = min(subtree_ceiling(widths),
                  fill_ceiling(prod(widths), BH, machine.sm_count))
    for BC in reversed(ladder):
        if ceiling >= BC:
            return BC
    return ladder[0]


__all__ = [
    "GROUP_TOKENS",
    "MMA_K_QUANTUM",
    "RING_RESERVE_BYTES",
    "SUPPORTED_BC",
    "WAVES_TARGET",
    "MachineFacts",
    "box_admitted",
    "derive_bc",
    "feasible_bc",
    "fill_ceiling",
    "largest_feasible_bc",
    "quanta_per_owner",
    "subtree_ceiling",
]
