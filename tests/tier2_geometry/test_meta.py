"""TIER 2 — the three meta-tests that make ``matrix.py`` self-policing.

Design §6.3. Without these the matrix is a list of tests with a nicer shape; with
them it is a CLAIM about completeness that the suite itself enforces.

* **M1 CENSUS** — runs at IMPORT, so an undeclared geometry fails at COLLECTION
  rather than as one red test among many. CPU-only: it reads the ratification
  manifest, not a device.
* **M2 NON-VACUITY** — a class-(a) row on a fixture where a broken kernel could
  not differ is a green test that proves nothing.
* **M3 CLASS DRIFT** — a class-(b) row that starts producing byte-identical output
  is a silent kernel change (a split-K path disabled, an atomic turned serial, a
  `P > 1` launch quietly degraded to `P = 1`) OR a fixture that stopped exercising
  the reassociation. **A tolerance-based test cannot see either**, because
  "identical" passes any bound.

M2 and M3 are themselves verified BY CONSTRUCTION at the bottom of this file: each
is pointed at an input where it MUST fail, and the failure is required. A
meta-test that cannot fail is not a meta-test.
"""
from __future__ import annotations

import pytest
import torch

from rola._build_config import BUILD_CONFIG
from tests.tier2_geometry import matrix as _matrix
from tests.tier2_geometry.matrix import (
    AXIS_BY_NAME,
    AXIS_DECLARATIONS,
    ENGINEERED_IDENTITY,
    GEOMETRY_MATRIX,
    REASSOCIATION_EQUIVALENT,
    AxisKind,
    GeometryPair,
    manifest_sha256,
)

# ===========================================================================
# M1 -- CENSUS
# ===========================================================================

def census() -> None:
    """Every geometry the shipped product carries is DECLARED, and declared once.

    Five checks, in the order a new axis would trip them:

    1. The manifest this reads IS the one the shipped `.so` was gated against
       (`BUILD_CONFIG["manifest_sha256"]`). A census over some other manifest is a
       census of a binary nobody built.
    2. The template axis NAMES decoded from the mangled instantiation names equal
       the declared template axes, in both directions.
    3. Each template axis's VALUE SET equals its declaration. A new `BT = 64`
       instantiation lands here.
    4. Every declared GEOMETRY axis has a matrix row for every pair its census rule
       requires; non-geometry axes have none.
    5. No two rows are the same (axis, values, output, restrict, precondition).
    """
    # 1 -----------------------------------------------------------------
    got, want = manifest_sha256(), BUILD_CONFIG["manifest_sha256"]
    assert got == want, (
        f"the geometry census is reading a manifest ({got}) that is NOT the "
        f"ratification the shipped binary was gated against ({want}). The census "
        "would then be enumerating a product that was never built.")

    # 2, 3 --------------------------------------------------------------
    #: Through the MODULE, not a from-import: M1's own by-construction test
    #: substitutes a product with one extra value, and a bound name would
    #: leave the real decoder in place and prove nothing.
    shipped = _matrix.shipped_axis_values()
    declared_template = {a.name: a for a in AXIS_DECLARATIONS if a.source == "template"}
    assert set(shipped) == set(declared_template), (
        f"the shipped instantiations carry template axes {sorted(set(shipped))}, but "
        f"matrix.py declares {sorted(set(declared_template))}. A template parameter "
        "cannot enter the product without a declared axis KIND: it is either a "
        "GEOMETRY (and needs rows), a PROBLEM_SHAPE, or a SEMANTIC arm gated in tier 1.")
    for name, values in sorted(shipped.items()):
        declared = set(declared_template[name].values)
        assert values == declared, (
            f"axis {name!r} ships values {sorted(values)} but matrix.py declares "
            f"{sorted(declared)}. A NEW VALUE ON AN EXISTING AXIS is exactly the case "
            "§6.3 is about: adding it to the kernel without declaring its contract "
            "class must fail here, before any geometry test runs.")

    # 4 -----------------------------------------------------------------
    by_axis: dict[str, list[GeometryPair]] = {}
    for row in GEOMETRY_MATRIX:
        axis = AXIS_BY_NAME.get(row.axis)
        assert axis is not None, f"row {row.id} names an undeclared axis {row.axis!r}"
        assert axis.kind is AxisKind.GEOMETRY, (
            f"row {row.id} relates two values of {row.axis!r}, which is declared "
            f"{axis.kind.value}: {axis.basis}")
        for v in row.values:
            assert v in axis.values, (
                f"row {row.id} uses {row.axis} = {v!r}, which the product does not ship")
        by_axis.setdefault(row.axis, []).append(row)

    for axis in AXIS_DECLARATIONS:
        if axis.kind is not AxisKind.GEOMETRY:
            assert axis.name not in by_axis
            continue
        rows = by_axis.get(axis.name, [])
        covered = {frozenset(r.values) for r in rows}
        for pair in axis.required_pairs():
            assert frozenset(pair) in covered, (
                f"geometry axis {axis.name!r} ships the pair {pair} with NO row in "
                f"GEOMETRY_MATRIX. Every geometry the planner may choose must carry a "
                f"declared contract class; an undeclared one is a geometry nobody "
                f"decided about. ({axis.basis})")

    # 5 -----------------------------------------------------------------
    keys = [(r.axis, r.values, r.output, r.restrict, r.precondition) for r in GEOMETRY_MATRIX]
    dupes = {k for k in keys if keys.count(k) > 1}
    assert not dupes, (
        f"duplicated rows {dupes}: two rows that differ in nothing but their class "
        "would make the matrix say two things at once.")


#: RUN AT IMPORT. §6.3's consequence -- *"adding `BT = 64` to the kernel without
#: declaring its contract class fails at collection"* -- is only true if the census
#: is a module-level statement rather than a test body.
census()


def test_M1_census():
    """The census, again, as a named reportable test (it already ran at import)."""
    census()


def test_the_census_notices_a_new_undeclared_geometry(monkeypatch):
    """M1, verified by construction: pretend the kernel gained `BT = 64`."""
    real = _matrix.shipped_axis_values
    monkeypatch.setattr(
        _matrix, "shipped_axis_values",
        lambda *a, **k: {**real(), "BT": real()["BT"] | {64}})
    with pytest.raises(AssertionError, match="NEW VALUE ON AN EXISTING AXIS"):
        census()


# ===========================================================================
# M2 / M3 -- these need the device
# ===========================================================================

_CLASS_A = [r for r in GEOMETRY_MATRIX if r.contract is ENGINEERED_IDENTITY]
_CLASS_B = [r for r in GEOMETRY_MATRIX if r.contract is REASSOCIATION_EQUIVALENT]

#: The `cuda` MARKER as well as the skipif, so `-m "not cuda"` deselects these at
#: COLLECTION -- before torch initializes a context (§6.5's VRAM-lock discipline).
#: M1 above carries neither: it is a file census and must run in the CPU job.
cuda_only = pytest.mark.cuda


def _rows():
    from tests.tier2_geometry import test_matrix_rows as R
    return R


def _perturb_one_ulp(row: GeometryPair, fx) -> None:
    """Nudge ONE element of the row's declared probe input by one fp32 ULP.

    fp32, not fp64: the kernel consumes the fp32 cast, so an fp64 ULP would vanish
    on the way in and the control would silently measure nothing.
    """
    if row.nonvacuity_probe == "v":
        t = fx._f32["v"]
        idx = (0, 0, 0, 0)
    else:
        assert fx.initial_state is not None, f"{row.id}: no initial state to probe"
        widths = fx.topology.widths
        leaf = torch.arange(fx.topology.N, device=fx.initial_state.device)
        never = (leaf // widths[1]) >= 2
        idx = (0, 0, int(never.nonzero()[0]), 0)
        t = fx.initial_state
    before = t[idx].to(torch.float32)
    after = torch.nextafter(before, before + 1.0)
    assert float(after) != float(before), "nextafter did not move the probe"
    t[idx] = after.to(t.dtype)


@cuda_only
@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="the consumer is a CUDA kernel")
@pytest.mark.parametrize("row", _CLASS_A, ids=[r.id for r in _CLASS_A])
def test_M2_an_engineered_identity_row_is_not_vacuous(row: GeometryPair):
    """A byte-equality that a BROKEN kernel would also satisfy proves nothing."""
    R = _rows()
    fx = R.build(row, dict(row.a))
    y, state = R.run(row, fx, dict(row.a))
    out = R.pick(row, fx, y, state)

    assert out.numel() > 0, f"{row.id}: the row's output selection is EMPTY"
    assert int(out.unique().numel()) > 1, (
        f"{row.id}: the compared output is constant, so any two geometries agree on "
        "it whatever they compute")

    control = R.build(row, dict(row.a))
    _perturb_one_ulp(row, control)
    y2, state2 = R.run(row, control, dict(row.a))
    out2 = R.pick(row, control, y2, state2)
    assert not torch.equal(out, out2), (
        f"{row.id}: a one-ULP change to `{row.nonvacuity_probe}` did NOT move the "
        "compared output, so this fixture cannot distinguish a working kernel from a "
        "broken one and the row's byte-equality is vacuous. Fix the FIXTURE (or the "
        "declared probe), never the assertion.")


def class_drift_check(row: GeometryPair, out_a, out_b) -> None:
    """M3's core assertion, factored out so it can be pointed at a known failure."""
    assert not torch.equal(out_a.view(torch.int32), out_b.view(torch.int32)), (
        f"{row.id} is declared REASSOCIATION_EQUIVALENT but the two geometries are "
        "BYTE-IDENTICAL. Exactly one of two things happened, and the correct response "
        "differs:\n"
        "  (1) THE KERNEL CHANGED -- the reassociation stopped happening. A split-K "
        "path was disabled, an atomic became a serial reduction, or a `P > 1` launch "
        "quietly degraded to `P = 1`. This is a BUG.\n"
        "  (2) THE FIXTURE CHANGED -- it no longer reaches a re-associated boundary. "
        "This is a RE-DERIVATION of the row: fix the fixture, or move the row to "
        "ENGINEERED_IDENTITY with a structural basis for why it is now exact.\n"
        f"  the row's basis was: {row.basis}")


@cuda_only
@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="the consumer is a CUDA kernel")
@pytest.mark.parametrize("row", _CLASS_B, ids=[r.id for r in _CLASS_B])
def test_M3_a_reassociation_row_still_reassociates(row: GeometryPair):
    R = _rows()
    _, _, out_a, out_b = R.both_sides(row)
    class_drift_check(row, out_a, out_b)


# ---------------------------------------------------------------------------
# M3, VERIFIED BY CONSTRUCTION -- two directions
# ---------------------------------------------------------------------------

@cuda_only
@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="the consumer is a CUDA kernel")
def test_M3_fails_when_a_split_silently_degrades_to_P_equals_one():
    """Force `P > 1` to launch as `P = 1` and require M3 to catch it.

    This is the exact failure §6.3 names: a geometry silently collapsing into
    another, with every tolerance-based test still green because "identical" passes
    any bound.
    """
    R = _rows()
    row = next(r for r in _CLASS_B if r.axis == "P" and r.output == "state")
    fx_a = R.build(row, dict(row.a))
    fx_b = R.build(row, dict(row.b))
    # THE DEGRADATION: geometry B asks for P = 2..4 and gets the unsplit launch.
    ya, sa = R.run(row, fx_a, dict(row.a))
    yb, sb = R.run(row, fx_b, {"P": 1})
    out_a = R.pick(row, fx_a, ya, sa)
    out_b = R.pick(row, fx_b, yb, sb)
    with pytest.raises(AssertionError, match="BYTE-IDENTICAL"):
        class_drift_check(row, out_a, out_b)


@cuda_only
@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="the consumer is a CUDA kernel")
def test_M3_fails_when_an_engineered_identity_is_mislabelled_as_a_reassociation():
    """The other direction of class drift: a class-(a) pair declared class (b).

    Pointed at a row that IS byte-equal by construction (the compact gather at
    saturation). If M3 passed here, it would be passing on a pair that never
    re-associates -- which is the same blindness in reverse.
    """
    R = _rows()
    row = next(r for r in _CLASS_A if r.axis == "COMPACT" and r.output == "state")
    _, _, out_a, out_b = R.both_sides(row)
    with pytest.raises(AssertionError, match="BYTE-IDENTICAL"):
        class_drift_check(row, out_a, out_b)


@cuda_only
@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="the consumer is a CUDA kernel")
def test_M2_fails_on_a_fixture_a_broken_kernel_could_not_move():
    """M2, verified by construction: point it at a genuinely inert probe.

    A never-written leaf's state is insensitive to `v` BY CONSTRUCTION, so a row
    that probed `v` there would be vacuous -- and M2 must say so rather than pass.
    """
    from dataclasses import replace
    row = next(r for r in _CLASS_A if r.restrict == "never_written_leaves")
    vacuous = replace(row, nonvacuity_probe="v")
    with pytest.raises(AssertionError, match="vacuous"):
        test_M2_an_engineered_identity_row_is_not_vacuous(vacuous)
