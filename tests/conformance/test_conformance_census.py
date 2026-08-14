"""CENSUS JOIN — (instantiation x data-regime), extending the M1 census.

Tier 2's M1 census proves every INSTANTIATION axis the shipped binary carries
is declared and rowed. The conformance design extends that census by ONE
DIMENSION: the join of the instantiation axes with the regime axes, where a
cell is covered when some family realizes both coordinates, and an empty cell
is LISTED, never silent. This file is that extension — it imports matrix.py's
axis declarations as its left-hand side rather than keeping a parallel copy,
so a new template parameter or axis value fails HERE too (demanding families
or expected-empty rows) exactly as it fails M1.

**The join is PAIRWISE, and that is a stated granularity, not an oversight.**
A family covers the pair (instantiation value, regime value) for every pair it
realizes jointly; the full outer product of all axes at once (~10^5 cells)
would be a coverage claim nobody funds. Pairwise is the granularity at which
the shared-router class of bug lives (one axis's handling silently specialized
to another axis's default), and it is the granularity the expected-empty list
makes reviewable.

**EXPECTED_EMPTY is a committed rule with named exceptions.** The rule: regime
values are swept at the DEFAULT instantiation, instantiation values at the
MID-BOX regime, so the (off-mid instantiation x non-mid regime) cross terms
are deliberately uncovered — except where a corner or producer family pins
both (those are the exceptions), plus two singletons the producer families'
one gateway configuration leaves open. The test asserts EXACT equality in both
directions: a newly-empty cell fails (coverage silently lost), and a
newly-covered cell listed as empty also fails (a stale list is the same
defect — the instruments-batch review's stale-table lesson).

Pure CPU: declarations only, no torch, no device.
"""
from __future__ import annotations

import pytest

from tests.conformance.families import _MID, FAMILIES, REGIME_AXES
from tests.tier2_geometry.matrix import AXIS_DECLARATIONS


def _covered() -> set[tuple]:
    cells = set()
    for family in FAMILIES:
        #: decay retired from the operator: decay families gate the ORACLE
        #: side and the refusal, never a kernel cell -- they leave the
        #: kernel-coverage join (docs/internals/chunk/chunk.md).
        if family.config.get("decay"):
            continue
        for ia, ivalue in family.instantiation_values().items():
            for ra, rvalue in family.regime.items():
                cells.add((ia, ivalue, ra, rvalue))
    return cells


def _grid() -> set[tuple]:
    return {(axis.name, value, ra, rvalue)
            for axis in AXIS_DECLARATIONS for value in axis.values
            for ra, rvalues in REGIME_AXES.items() for rvalue in rvalues}


def test_every_family_realizes_declared_axis_values_only():
    """The join's left-hand side IS matrix.py's: every family's instantiation
    coordinates must name declared axes and declared values, in both
    directions — a family pinning an undeclared value is a census escape, and
    a declared axis no family states is a hole the join cannot see."""
    declared = {axis.name: set(axis.values) for axis in AXIS_DECLARATIONS}
    #: decay families are oracle-only (decay retired from the operator) and
    #: do not name kernel cells -- same exclusion as _covered().
    declared["DECAY"].add(1)  # tolerated in DECLARATIONS ONLY for these rows
    for family in FAMILIES:
        values = family.instantiation_values()
        assert set(values) == set(declared), (
            f"{family.name} states axes {sorted(values)} but matrix.py declares "
            f"{sorted(declared)}; a new template parameter must be joined here "
            "the moment it is declared there")
        for axis, value in values.items():
            assert value in declared[axis], (
                f"{family.name} realizes {axis} = {value!r}, which is not a "
                f"declared value {sorted(map(str, declared[axis]))}")


# ---------------------------------------------------------------------------
# The committed expected-empty list
# ---------------------------------------------------------------------------

#: The instantiation values NOT part of the default cell the named regime
#: families run at. (BC 16, DV 64, D 2, both DECAY, unpaged is the default;
#: DECAY is fully arm-swept by variants.) P26: `BT`/`FOLD_PREC`/`READ_PREC`
#: are gone from `AXIS_DECLARATIONS` entirely (no `BT` template parameter;
#: `mma_precision` is single-valued) rather than pinned mid-cells, so they no
#: longer appear as instantiation coordinates at all.
_OFF_MID_INSTANTIATION: frozenset[tuple] = frozenset({
    ("BC", 64), ("DV", 32), ("D", 1), ("D", 3), ("D", 4),
    ("paging", True),
})

#: Every non-mid regime value (the mid-box cell is what the instantiation
#: sweep carries across the grid).
_NON_MID_REGIME: frozenset[tuple] = frozenset(
    (axis, value)
    for axis, values in REGIME_AXES.items() for value in values
    if _MID[axis] != value)

#: Cross terms that ARE covered, because a family pins both coordinates.
_EXCEPTIONS: frozenset[tuple] = frozenset({
    # producer/swept-gain runs the gateway at widths (16,) and d_v = 32;
    # producer/swept-gain-raw-d2 at (8, 8) and d_v = 32. Their sweeps carry
    # `swept` density and `coherent` routing onto D = 1 and DV = 32.
    ("D", 1, "density", "swept"), ("D", 1, "coherence", "coherent"),
    ("DV", 32, "density", "swept"), ("DV", 32, "coherence", "coherent"),
})

#: Mid-instantiation cells the producer families' ONE gateway configuration
#: leaves open: the sweep runs undecayed at d_v = 32, so `swept` never meets
#: DECAY = 1 or DV = 64. Deliberate — the sweep's job is the density knob, and
#: the decay/value-width axes meet every OTHER regime value; extending the
#: gateway sweep across them is priced, not forgotten.
_EXTRAS: frozenset[tuple] = frozenset()

EXPECTED_EMPTY: frozenset[tuple] = frozenset(
    (ia, ivalue, ra, rvalue)
    for (ia, ivalue) in _OFF_MID_INSTANTIATION
    for (ra, rvalue) in _NON_MID_REGIME
    if (ia, ivalue, ra, rvalue) not in _EXCEPTIONS
) | _EXTRAS


def test_the_census_join_lists_every_empty_cell():
    empty = _grid() - _covered()
    newly_empty = empty - EXPECTED_EMPTY
    stale = EXPECTED_EMPTY - empty
    assert not newly_empty, (
        f"{len(newly_empty)} join cell(s) went empty without being listed — "
        "coverage was lost silently. Either restore the family that carried "
        f"them or list them here WITH a reason: {sorted(newly_empty, key=str)}")
    assert not stale, (
        f"{len(stale)} listed-empty cell(s) are now covered — the list is "
        "stale, which is the same defect as a silent hole (a reviewer reading "
        f"it is being lied to): {sorted(stale, key=str)}")


def test_the_join_has_the_size_its_rule_implies():
    """The rule's arithmetic, pinned so an edit to the rule pieces cannot
    silently reshape the list: |cross| - |exceptions| + |extras|."""
    # 57 -> 56 at D2a: decay retired from the operator, so the lone decay
    # _EXTRAS row left the list (decay families are oracle-only now).
    assert len(_OFF_MID_INSTANTIATION) * len(_NON_MID_REGIME) == 60
    assert len(EXPECTED_EMPTY) == 60 - len(_EXCEPTIONS) + len(_EXTRAS) == 56


@pytest.mark.parametrize("cell", sorted(_EXCEPTIONS | _EXTRAS, key=str), ids=str)
def test_every_exception_and_extra_is_still_meaningful(cell):
    """An exception must be COVERED (it names a family pin) and an extra must
    be EMPTY (it names a real hole); one that is neither is dead weight."""
    covered = _covered()
    if cell in _EXTRAS:
        assert cell not in covered, f"extra {cell} is covered now; delete it"
    else:
        assert cell in covered, f"exception {cell} lost its covering family"
