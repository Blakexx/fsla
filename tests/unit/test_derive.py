# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""THE ``BC`` DERIVATION, term by term.

``derive_bc`` is the whole of the planner's decision, so each of its three ceilings is
gated on its own -- with the other two held wide open, so a passing term cannot be
carried by a neighbour -- and then the composition and the clamp are gated together.

The machine is INJECTED. :class:`MachineFacts` is a plain record, so a term can be
asked about a machine with an arbitrary SM count and an arbitrary arm carve without a
device and without a mock; the one test that must see the REAL numbers reads them off
the built binary's dispatch table and is marked ``cuda`` for it.

Symbols: ``widths`` the level widths MSB first; ``D = len(widths)``; ``N`` the leaf
capacity; ``BH = batch * heads``; ``d_v`` the value width; ``BC`` leaves per owner.
"""
from __future__ import annotations

import pytest

from rola.planner.derive import (
    MMA_K_QUANTUM,
    WAVES_TARGET,
    MachineFacts,
    box_admitted,
    derive_bc,
    feasible_bc,
    fill_ceiling,
    largest_feasible_bc,
    quanta_per_owner,
    subtree_ceiling,
)
from rola.routing.types import (
    IndependentRouting,
    LevelConfig,
    SoftmaxActivation,
    Topology,
)

#: A machine that binds NOTHING: enough shared memory for any carve below, and few
#: enough SMs that the fill ceiling is above every candidate. Terms under test are
#: reintroduced one at a time against it.
_OPEN_SMEM = 1 << 30


def _machine(*, sm_count: int = 1, max_dynamic_smem_bytes: int = _OPEN_SMEM,
             carve: dict | None = None, bcs=(16, 64), d_v: int = 64, D: int = 2,
             decay: bool = False) -> MachineFacts:
    """A synthetic machine carrying one arm per ``BC`` at a stated footprint."""
    #: The arm key carries the WINDOW as its last coordinate. `derive_bc` prices the
    #: GROUP-QUANTUM arm (the window is chosen later, from the realized routing), so
    #: a synthetic machine states its carve at that window.
    arms = (carve if carve is not None
            else {(BC, d_v, D, decay, MMA_K_QUANTUM): 1024 for BC in bcs})
    return MachineFacts(sm_count=sm_count, max_dynamic_smem_bytes=max_dynamic_smem_bytes,
                        arm_smem_bytes=arms)


def _topology(widths, d_v=64) -> Topology:
    routing = IndependentRouting(
        tied=False, read_activation=SoftmaxActivation(), write_activation=SoftmaxActivation())
    return Topology(
        levels=tuple(LevelConfig(width=w, routing=routing) for w in widths), d_v=d_v)


# ---------------------------------------------------------------------------
# Term 1: the structural ceiling
# ---------------------------------------------------------------------------


def test_the_subtree_ceiling_is_the_innermost_level_width():
    """A block wider than ``b_{D-1}`` reaches into a second parent's children."""
    assert subtree_ceiling((64, 64)) == 64
    assert subtree_ceiling((64, 64, 16)) == 16
    assert subtree_ceiling((16,)) == 16


def test_the_subtree_ceiling_binds_the_derivation_against_an_otherwise_open_machine():
    """``(64, 64, 16)`` admits ``BC = 64`` by ALIGNMENT and is still derived at 16.

    ``owner_geometry`` accepts ``BC = 64`` there -- 4 digits of the width-64 middle
    level times the full width-16 innermost one -- so alignment is not what refuses it.
    The refusal is structural: those 4 digits are 4 INDEPENDENT SUBTREES.
    """
    assert 64 in feasible_bc((64, 64, 16))
    machine = _machine(sm_count=1, D=3)
    assert derive_bc(_topology((64, 64, 16)), machine, BH=1024, decay=False) == 16
    # The same widths with the wide level innermost derive the wide block.
    assert derive_bc(_topology((64, 16, 64)), _machine(sm_count=1, D=3),
                     BH=1024, decay=False) == 64


def test_the_subtree_ceiling_saturates_at_the_smallest_block_below_the_ladder():
    """``(4, 8, 8)``'s innermost level is narrower than one MMA quantum.

    Every compiled block spans independent subtrees there, so the ceiling admits none
    of them. The kernel still runs correctly at the smallest one, and it is the closest
    the ladder comes to the bound -- an efficiency ceiling bounds how wide a block
    should be, never whether one exists.
    """
    assert subtree_ceiling((4, 8, 8)) < 16
    assert derive_bc(_topology((4, 8, 8)), _machine(sm_count=1, D=3),
                     BH=1024, decay=False) == 16


# ---------------------------------------------------------------------------
# Term 2: the box
# ---------------------------------------------------------------------------


def test_the_box_admits_exactly_the_arms_whose_carve_plus_rings_fit():
    carve = {(16, 64, 2, False, MMA_K_QUANTUM): 12000, (64, 64, 2, False, MMA_K_QUANTUM): 32000}
    assert box_admitted(_machine(carve=carve, max_dynamic_smem_bytes=32000),
                        64, 2, False) == (16, 64)
    assert box_admitted(_machine(carve=carve, max_dynamic_smem_bytes=31999),
                        64, 2, False) == (16,)


def test_the_box_binds_the_derivation_when_the_wide_arm_does_not_fit():
    carve = {(16, 64, 2, False, MMA_K_QUANTUM): 12000, (64, 64, 2, False, MMA_K_QUANTUM): 32000}
    machine = _machine(carve=carve, max_dynamic_smem_bytes=20000, sm_count=1)
    assert derive_bc(_topology((64, 64)), machine, BH=1024, decay=False) == 16


def test_a_machine_that_fits_no_arm_is_a_refusal_and_not_a_smaller_block():
    machine = _machine(carve={(16, 64, 2, False, MMA_K_QUANTUM): 12000}, max_dynamic_smem_bytes=8000)
    with pytest.raises(ValueError, match="no compiled arm fits shared memory"):
        box_admitted(machine, 64, 2, False)
    with pytest.raises(ValueError, match="no compiled arm fits shared memory"):
        derive_bc(_topology((64, 64)), machine, BH=1024, decay=False)


def test_the_box_keys_on_the_arm_and_not_only_on_bc():
    """A decay arm's carve is its own; asking for the wrong one is not a near miss."""
    carve = {(16, 64, 2, False, MMA_K_QUANTUM): 12000, (64, 64, 2, False, MMA_K_QUANTUM): 12000,
             (16, 64, 2, True, MMA_K_QUANTUM): 12000}
    machine = _machine(carve=carve, max_dynamic_smem_bytes=12000)
    assert box_admitted(machine, 64, 2, False) == (16, 64)
    assert box_admitted(machine, 64, 2, True) == (16,)


@pytest.mark.cuda
def test_the_box_arithmetic_holds_against_the_binarys_own_carve():
    """The real numbers: every admitted arm is one the launch would also admit.

    ``MachineFacts.for_device`` reads the ``constexpr`` footprints out of the built
    binary -- carves that already include the cooked schedule ring -- so this is the
    KERNEL's arithmetic rather than a model of it. The claim is the enable predicate's
    own: admitted means ``smem_bytes <= the block limit``, refused means it does not
    fit, and the two partition the compiled set.
    """
    import torch

    from rola.planner.dispatch_table import DispatchTable

    machine = MachineFacts.for_device(torch.device("cuda"))
    table = DispatchTable.from_extension()
    assert machine.arm_smem_bytes, "the binary carries no arms to price"
    for row in table.rows:
        assert machine.arm_smem_bytes[row.key] == row.smem_bytes
        #: THE BOX PRICES THE GROUP-QUANTUM ARM. `derive_bc` runs before a schedule
        #: exists, so it cannot know the window -- the window is chosen afterwards,
        #: from the realized routing, and `select_window` applies the SAME admission
        #: to the wider arm before selecting it. So the box's claim is asked of the
        #: `GT == 16` rows, and every row (wide included) is still checked against
        #: its own per-row predicate, which is the inequality the launch enforces.
        if row.GT == MMA_K_QUANTUM:
            admitted = row.BC in box_admitted(machine, row.DV, row.D, row.decay)
            assert admitted == (row.smem_bytes <= machine.max_dynamic_smem_bytes)
        assert (row.admitted_by(machine.max_dynamic_smem_bytes)
                == (row.smem_bytes <= machine.max_dynamic_smem_bytes))


# ---------------------------------------------------------------------------
# Term 3: the fill guard
# ---------------------------------------------------------------------------


def test_the_fill_ceiling_is_the_block_that_still_supplies_the_target_waves():
    """``BH * N / BC >= sm_count * WAVES_TARGET`` rearranged, exactly."""
    N, BH, sms = 4096, 8, 80
    ceiling = fill_ceiling(N, BH, sms)
    assert ceiling == (BH * N) // (sms * WAVES_TARGET)
    assert (BH * N) / ceiling >= sms * WAVES_TARGET
    assert (BH * N) / (ceiling + 1) < sms * WAVES_TARGET


def test_the_fill_ceiling_never_falls_below_one_however_small_the_grid():
    assert fill_ceiling(N=16, BH=1, sm_count=80) == 1


def test_the_fill_guard_binds_a_large_block_on_a_grid_that_would_starve():
    """Same topology, same machine, two cohort counts -- and only ``BH`` moves."""
    machine = _machine(sm_count=80)
    topology = _topology((64, 64))          # N = 4096, subtree ceiling 64
    assert derive_bc(topology, machine, BH=1, decay=False) == 16
    assert derive_bc(topology, machine, BH=64, decay=False) == 64


def test_the_fill_guard_saturates_at_the_smallest_block_rather_than_refusing():
    """A grid no block can fill is planned at the block that fills it best.

    ``N * BH`` here is 64 CTAs at ``BC = 16`` against a 80-SM machine wanting 160, so
    every candidate is under target and the guard has no admissible ``BC`` beneath it.
    The answer is the smallest compiled block, not an error: refusing would refuse
    every small cell the gates run on, and no smaller alternative exists.
    """
    machine = _machine(sm_count=80)
    assert fill_ceiling(1024, 1, 80) < 16
    assert derive_bc(_topology((16, 64)), machine, BH=1, decay=False) == 16


# ---------------------------------------------------------------------------
# The composition, and the clamp onto the compiled ladder
# ---------------------------------------------------------------------------


def test_the_derivation_takes_the_minimum_of_the_three_ceilings():
    """Each term binds in turn while the other two are open."""
    topology = _topology((64, 64))
    open_machine = _machine(sm_count=1)
    assert derive_bc(topology, open_machine, BH=1024, decay=False) == 64

    narrow = _topology((64, 64, 16))                      # subtree ceiling 16
    assert derive_bc(narrow, _machine(sm_count=1, D=3), BH=1024, decay=False) == 16
    boxed = _machine(sm_count=1, carve={(16, 64, 2, False, MMA_K_QUANTUM): 1, (64, 64, 2, False, MMA_K_QUANTUM): 1 << 20},
                     max_dynamic_smem_bytes=1024)
    assert derive_bc(topology, boxed, BH=1024, decay=False) == 16
    assert derive_bc(topology, _machine(sm_count=80), BH=1, decay=False) == 16


def test_a_ceiling_between_two_rows_clamps_down_to_the_lower_one():
    """The formula picks AMONG compiled rows; 63 is not a plan, 16 is."""
    machine = _machine(sm_count=1)
    topology = _topology((32, 32))                        # subtree ceiling 32
    assert 32 not in machine.compiled_bc(64, 2, False)
    assert derive_bc(topology, machine, BH=1024, decay=False) == 16


def test_a_topology_the_ladder_cannot_align_is_a_loud_topology_defect():
    """``(2, 3, 5)`` has 30 leaves: no instantiated block partitions them."""
    with pytest.raises(ValueError, match="no feasible BC"):
        derive_bc(_topology((2, 3, 5)), _machine(sm_count=1, D=3), BH=1024, decay=False)


def test_the_expert_pin_replaces_the_derivation_and_is_still_alignment_validated():
    topology = _topology((64, 64, 16))                    # derives 16
    assert derive_bc(topology, _machine(sm_count=1, D=3), BH=1024, decay=False) == 16
    assert derive_bc(topology, _machine(sm_count=1, D=3), BH=1024, decay=False,
                     pinned=64) == 64
    with pytest.raises(ValueError, match="no feasible BC"):
        derive_bc(_topology((2, 4, 16)), _machine(sm_count=1, D=3), BH=1024, decay=False,
                  pinned=48)


def test_the_derivation_reads_no_tensor_and_no_routing_statistic():
    """Same launch-static inputs, same answer -- there is nothing to go stale.

    The signature is the argument: a routing realization cannot reach the formula,
    so a plan cannot be invalidated by one.
    """
    machine = _machine(sm_count=80)
    topology = _topology((64, 64))
    first = derive_bc(topology, machine, BH=64, decay=False)
    assert all(derive_bc(topology, machine, BH=64, decay=False) == first
               for _ in range(4))


# ---------------------------------------------------------------------------
# The ladder helpers
# ---------------------------------------------------------------------------


def test_the_feasible_ladder_is_the_alignment_admissible_set():
    assert feasible_bc((64, 64)) == (16, 64)
    assert feasible_bc((2, 4, 16)) == (16, 64)
    with pytest.raises(ValueError, match="no feasible BC"):
        feasible_bc((4, 6))
    assert largest_feasible_bc((64, 64)) == 64


def test_quanta_per_owner_is_the_dense_k_quantum_count():
    assert quanta_per_owner(16) == 1
    assert quanta_per_owner(64) == 4
