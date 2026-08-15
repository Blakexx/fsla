"""CENSUS JOIN — (instantiation x data-regime).

The join is (instantiation axis value) x (regime axis value): a cell is covered
when some family realizes both coordinates, and an empty cell is LISTED, never
silent. The left-hand side is `tests.conformance.instantiation`'s axis
declarations, and this file asserts THOSE are the projections of the built arm
list `tools/gen_shards.py::CHUNK_ARMS` — so a widened chunk matrix fails HERE,
demanding families or expected-empty rows. That is the M1 property the tier-2
geometry matrix used to carry for the tiled consumer's template parameters.

P67 D2-b moved the left-hand side. It was `tests.tier2_geometry.matrix`'s
`(BC, DV, D, DECAY, paging)`; four of those five axes retire with the tiled
consumer (`instantiation.py` states why each one went), and `(D, B, paging)` —
the axes a caller can actually move on the chunk arm — took their place.

**The join is PAIRWISE, and that is a stated granularity, not an oversight.**
A family covers the pair (instantiation value, regime value) for every pair it
realizes jointly; the full outer product of all axes at once would be a
coverage claim nobody funds. Pairwise is the granularity at which the
shared-router class of bug lives (one axis's handling silently specialized to
another axis's default), and it is the granularity the expected-empty list
makes reviewable.

**Only families the KERNEL ARM RUNS are coverage.** A family outside the built
chunk matrix keeps a live claim — the refusal is named, and the semantics are
graded against the oracle the test calls — but it is not a kernel-coverage cell.
D2-a excluded the
decay families by name; post-D2 decay is one refusing clause among six, and
singling it out would have counted a `d_v = 32` or ragged-`T` family as kernel
coverage it is not, so the exclusion is now the envelope itself.

**EXPECTED_EMPTY is a committed rule with named exceptions.** The rule: regime
values are swept at the DEFAULT instantiation, instantiation values at the
MID-BOX regime, so the (off-mid instantiation x non-mid regime) cross terms are
deliberately uncovered — except where a producer family pins both (those are
the exceptions), plus the cells `_EXTRAS` names one by one with a reason. The
test asserts EXACT equality in both directions: a newly-empty cell fails
(coverage silently lost), and a newly-covered cell listed as empty also fails
(a stale list is the same defect — a reviewer reading it is being lied to).

Pure CPU: declarations only, no torch, no device.
"""
from __future__ import annotations

import pytest

from tests.conformance.families import _MID, FAMILIES, REGIME_AXES
from tests.conformance.instantiation import (
    AXIS_DECLARATIONS,
    BUILT_TOPOLOGIES,
    AxisDeclaration,
)


def _covered() -> set[tuple]:
    cells = set()
    for family in FAMILIES:
        if not family.runs_on_kernel:
            continue
        for ia, ivalue in family.instantiation_values().items():
            for ra, rvalue in family.regime.items():
                cells.add((ia, ivalue, ra, rvalue))
    return cells


def _grid() -> set[tuple]:
    return {(axis.name, value, ra, rvalue)
            for axis in AXIS_DECLARATIONS for value in axis.values
            for ra, rvalues in REGIME_AXES.items() for rvalue in rvalues}


def test_the_declared_axes_are_the_built_arm_list_projected():
    """M1, on the axes that survived: DECLARED must equal SHIPPED.

    `CHUNK_ARMS` is the enumeration the generated instantiation TUs come from,
    so it is what the assembler saw; the declarations are what the census joins
    over. An arm added to the matrix without a declaration here would widen the
    shipped product and leave the census reporting full coverage of a smaller
    one.
    """
    projections = {
        "D": {D for (D, _B) in BUILT_TOPOLOGIES},
        "B": {B for (_D, B) in BUILT_TOPOLOGIES},
    }
    manifest = {a.name for a in AXIS_DECLARATIONS if a.source == "manifest"}
    assert manifest == set(projections), (
        f"manifest axes {sorted(manifest)} do not match the arm list's "
        f"projections {sorted(projections)}")
    for axis in AXIS_DECLARATIONS:
        if axis.source != "manifest":
            continue
        assert set(axis.values) == projections[axis.name], (
            f"axis {axis.name} declares {sorted(axis.values)} but the built arm "
            f"list projects {sorted(projections[axis.name])}")


def test_a_runtime_axis_is_in_the_join_only_because_it_is_declared():
    """The other half of M1's completeness claim, stated rather than assumed: a
    runtime axis leaves NO symbol behind — paging is an allocation of the state,
    not an instantiation — so the arm list cannot discover it and only the
    declaration puts it in the join at all."""
    runtime = [a for a in AXIS_DECLARATIONS if a.source == "runtime"]
    assert [a.name for a in runtime] == ["paging"]
    assert isinstance(runtime[0], AxisDeclaration)
    assert runtime[0].values == (False, True)


def test_every_family_realizes_declared_axis_values_only():
    """Every family's instantiation coordinates must name declared axes and
    declared values, in both directions — a family pinning an undeclared value
    is a census escape, and a declared axis no family states is a hole the join
    cannot see."""
    declared = {axis.name: set(axis.values) for axis in AXIS_DECLARATIONS}
    for family in FAMILIES:
        values = family.instantiation_values()
        assert set(values) == set(declared), (
            f"{family.name} states axes {sorted(values)} but the census declares "
            f"{sorted(declared)}; a new axis must be joined here the moment it "
            "is declared there")
        for axis, value in values.items():
            assert value in declared[axis], (
                f"{family.name} realizes {axis} = {value!r}, which is not a "
                f"declared value {sorted(map(str, declared[axis]))}")


# ---------------------------------------------------------------------------
# The committed expected-empty list
# ---------------------------------------------------------------------------

#: The instantiation values NOT part of the default cell the named regime
#: families run at. The default is the two-level, width-8, unpaged topology
#: (`D = 2, B = 8, paging = False`) — `families._cfg`'s own defaults.
_OFF_MID_INSTANTIATION: frozenset[tuple] = frozenset({
    ("D", 1), ("D", 3), ("D", 4),
    ("B", 16), ("B", 64), ("B", 256),
    ("paging", True),
})

#: Every non-mid regime value (the mid-box cell is what the instantiation
#: sweep carries across the grid).
_NON_MID_REGIME: frozenset[tuple] = frozenset(
    (axis, value)
    for axis, values in REGIME_AXES.items() for value in values
    if _MID[axis] != value)

#: Cross terms that ARE covered, because a family pins both coordinates. All
#: four come from the producer sweeps, the only families that carry a non-mid
#: regime onto an off-mid instantiation: `producer/swept-gain` runs the real
#: gateway at the flat-routing arm (`D = 1, B = 64`) and
#: `producer/swept-gain-d64` at `(D = 2, B = 64)`, and both carry `swept`
#: density and `coherent` routing there.
_EXCEPTIONS: frozenset[tuple] = frozenset({
    ("D", 1, "density", "swept"), ("D", 1, "coherence", "coherent"),
    ("B", 64, "density", "swept"), ("B", 64, "coherence", "coherent"),
})

#: MID-instantiation cells that are nonetheless empty, each with its reason.
#:
#: `tail = ragged` is empty at EVERY instantiation, and that is an envelope
#: fact rather than a coverage gap: the chunk arm's token quantum is
#: `rola.ops.chunk.CHUNK_TOKENS`, and a `T` that is not a multiple of it has no
#: kernel at all — the call is REFUSED. So the ragged regime is oracle-graded by
#: construction, never kernel-covered. `stratified/ragged` and
#: `stratified/ragged-raw-decay` still run: they hold the ragged SEMANTICS
#: against the oracle and pin the named refusal, which is what remains to hold.
#: The off-mid ragged cells are already covered by the cross-term rule; these
#: three are the mid-instantiation ones the rule does not reach.
_EXTRAS: frozenset[tuple] = frozenset({
    ("D", 2, "tail", "ragged"),
    ("B", 8, "tail", "ragged"),
    ("paging", False, "tail", "ragged"),
})

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
    silently reshape the list: |cross| - |exceptions| + |extras|.

    P67 D2-b re-pinned both numbers. The cross was 6 off-mid instantiation
    values x 10 non-mid regime values = 60 under the tiled axes
    `(BC, DV, D, paging)`; it is 7 x 10 = 70 under `(D, B, paging)` — `BC` and
    `DV` left, `B` arrived with four values of which three are off-mid, and
    `D` kept its three off-mid values.
    """
    assert len(_OFF_MID_INSTANTIATION) * len(_NON_MID_REGIME) == 70
    assert len(EXPECTED_EMPTY) == 70 - len(_EXCEPTIONS) + len(_EXTRAS) == 69


@pytest.mark.parametrize("cell", sorted(_EXCEPTIONS | _EXTRAS, key=str), ids=str)
def test_every_exception_and_extra_is_still_meaningful(cell):
    """An exception must be COVERED (it names a family pin) and an extra must
    be EMPTY (it names a real hole); one that is neither is dead weight."""
    covered = _covered()
    if cell in _EXTRAS:
        assert cell not in covered, f"extra {cell} is covered now; delete it"
    else:
        assert cell in covered, f"exception {cell} lost its covering family"
