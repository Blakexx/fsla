# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""The producer/decode capacity relation, pinned across the language boundary.

``csrc/rola/src/decode.cuh`` carries a ``static_assert`` that the decode path's
level-width capacity is at least the producer's ``MAX_BRANCH_WIDTH``. That assertion is
only half the gate: **a C++ ``static_assert`` cannot read a Python constant**, so the
C++ side compares against its own mirrored copy of the number, and a change to
``rola/routing/topology.py`` would move the producer's real limit while leaving the
mirror — and therefore the assertion — silently satisfied.

This file is the other half: it compares the SHIPPED BINARY's capacity against the
SHIPPED PYTHON's constant. Together the two make the relation checkable rather than
merely written down, and a topology the producer can emit but the decode path would
over-read on becomes a red test instead of an out-of-bounds read.
"""

from __future__ import annotations

import pytest
import torch

from rola.routing.topology import MAX_BRANCH_WIDTH

pytestmark = pytest.mark.cuda


def _topology(widths):
    """A softmax topology of the given widths -- the shape only; no cell here prices
    a routing, they compare two packings of one declaration."""
    from rola.routing.types import (
        IndependentRouting,
        LevelConfig,
        SoftmaxActivation,
        Topology,
    )

    routing = IndependentRouting(tied=False, read_activation=SoftmaxActivation(),
                                 write_activation=SoftmaxActivation())
    return Topology(levels=tuple(LevelConfig(width=w, routing=routing) for w in widths),
                    d_v=16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reads the compiled extension")
def test_decode_capacity_covers_the_producer_max_branch_width():
    from rola.ops._ext import extension

    capacity = extension().rola_decode_capacity()
    assert capacity >= MAX_BRANCH_WIDTH, (
        f"the decode path's level-width capacity is {capacity} but the producer emits "
        f"levels up to MAX_BRANCH_WIDTH={MAX_BRANCH_WIDTH}. A topology the producer can "
        "build would over-read the decode kernel's staged amplitude row. Raise "
        "kDecodeMaxLevelWidth in csrc/rola/src/decode.cuh (and re-check the GEMV's SMEM "
        "ledger, which scales with sum_l width_l).")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reads the compiled extension")
def test_the_cxx_mirror_of_max_branch_width_has_not_drifted():
    """The mirror is EQUAL, not merely bounded.

    The ``static_assert`` would still pass if the producer's limit dropped and the
    mirror stayed high, which is harmless, and if the producer's limit ROSE while the
    mirror stayed put, which is not. Pinning equality catches the second case at the
    only place both numbers are visible at once.
    """
    from rola.ops._ext import extension

    assert extension().rola_decode_producer_width_mirror() == MAX_BRANCH_WIDTH, (
        "csrc/rola/src/decode.cuh's kProducerMaxBranchWidth mirror has drifted from "
        f"rola/routing/topology.py's MAX_BRANCH_WIDTH={MAX_BRANCH_WIDTH}; update the "
        "mirror in the same commit that moves the producer's limit")


# ---------------------------------------------------------------------------
# §8b items 4 and 6 -- THE CROSS-TRACK CARRIER RECONCILIATION
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("widths", [(8, 16), (4,), (2, 3, 4), (16, 8, 8), (4, 4, 4, 4)])
def test_the_two_feeder_flag_packings_are_the_same_word_bit_for_bit(widths):
    """P's ``FeederFlags.packed()`` IS Track D's ``static_density`` bit layout.

    The two tracks built one 32-bit word twice, independently and concurrently: §4.4
    fixes bit ``2l`` = ``dense_read[l]`` and bit ``2l+1`` = ``dense_write[l]``, Track P
    implemented it as :meth:`FeederFlags.packed` and Track D open-coded the same shifts
    in :func:`freeze_decode_config`'s ``static_density`` branch. The merge made P's
    object canonical and had D consume it, and this is what makes that consumption
    CHECKED rather than asserted: if either layout is ever edited, the two words diverge
    here instead of the kernel decoding a word built under the other convention.

    Exhaustive over both sides of every level, so it pins the LAYOUT and not one sample
    of it.
    """
    import itertools

    from rola.ops.decode import freeze_decode_config
    from rola.planner.feeder import FeederFlags
    from rola.planner.select import ConfigChoice, FrozenConfig

    D = len(widths)
    topology = _topology(widths)

    for bits in itertools.product((False, True), repeat=2 * D):
        read = tuple(bits[2 * l] for l in range(D))
        write = tuple(bits[2 * l + 1] for l in range(D))
        density = tuple((read[l], write[l]) for l in range(D))

        via_density = freeze_decode_config(
            topology, "raw", decay=False, BH=1, device="cpu",
            static_density=density, n_split=1, mma_precision="bf16").feeder_flags

        flags = FeederFlags(read=read, write=write)
        frozen = FrozenConfig(choice=ConfigChoice(BC=16, BT=16, packing=False,
                                                  feeder_flags=flags))
        via_canonical = freeze_decode_config(
            topology, "raw", decay=False, BH=1, device="cpu",
            frozen=frozen, n_split=1, mma_precision="bf16").feeder_flags

        assert via_density == via_canonical == flags.packed(), (
            f"the feeder-flag word disagrees at read={read} write={write}: "
            f"static_density -> {via_density:#0{2 * D + 2}b}, "
            f"FrozenConfig -> {via_canonical:#0{2 * D + 2}b}, "
            f"FeederFlags.packed() -> {flags.packed():#0{2 * D + 2}b}")


def test_supplying_both_flag_sources_is_refused_rather_than_ordered():
    """Two answers to one question is a raise, not a precedence rule."""
    from rola.ops.decode import freeze_decode_config
    from rola.planner.feeder import FeederFlags
    from rola.planner.select import ConfigChoice, FrozenConfig

    topology = _topology((8, 16))
    frozen = FrozenConfig(choice=ConfigChoice(
        BC=16, BT=16, packing=False, feeder_flags=FeederFlags.none(2)))
    with pytest.raises(ValueError, match="never both"):
        freeze_decode_config(topology, "raw", decay=False, BH=1, device="cpu",
                             frozen=frozen, static_density=((False, False),) * 2,
                             n_split=1, mma_precision="bf16")
