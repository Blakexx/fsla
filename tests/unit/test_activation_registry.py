# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The activation registry (design §4.6, as amended 2026-08-01).

The point of the registry is that activation properties live in ONE table and every
rule reads columns from it. So most of what is gated here is not "does softmax have
the right flags" -- it is the *structural* claims: that the table is the only source,
that an unknown tag cannot run, that rules are stated against columns rather than
against activation names, and that the docs' matrix is generated rather than typed.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import torch

import rola
from rola.routing.activations import (
    ACTIVATION_COLUMNS,
    ACTIVATION_REGISTRY,
    ActivationProperties,
    UnknownActivation,
    capability_table,
    properties_for,
    ratified_entmax_alphas,
)
from rola.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    LeafMassDecay,
    LevelConfig,
    SoftmaxActivation,
    Topology,
    UnionRouting,
    entmax,
    softmax,
)

# --- the table itself --------------------------------------------------------

def test_v0_1_ratifies_exactly_the_two_declared_activation_families():
    """§4.6: v0.1 ships exactly two ratified families, plus the opaque row.

    A fourth row appearing without a version bump and a docs change is the thing this
    catches. ``("opaque", None)`` is the row a level filled by a FOREIGN producer reads,
    and it is a row rather than a special case precisely so that every rule quantifying
    over cells covers such a level without knowing it is one.
    """
    assert set(ACTIVATION_REGISTRY) == {
        ("softmax", None), ("entmax", 1.5), ("entmax", 2.0), ("opaque", None)}
    assert all(row.kerneled for row in ACTIVATION_REGISTRY.values())


def test_every_row_declares_every_column():
    for tag, row in ACTIVATION_REGISTRY.items():
        assert isinstance(row, ActivationProperties), tag
        for column in ACTIVATION_COLUMNS:
            assert isinstance(getattr(row, column), bool), f"{tag}.{column}"


def test_the_declared_properties_are_the_ones_the_maps_actually_have():
    """The table is only worth reading if it is true. Checked against the reference
    implementations, not against the docstrings that describe them."""
    logits = torch.tensor([[3.0, 0.5, -2.0, -4.0]], dtype=torch.float64)
    from rola.routing.entmax.reference import entmax as entmax_reference

    dense = torch.softmax(logits, dim=-1)
    assert torch.allclose(dense.sum(-1), torch.ones(1, dtype=torch.float64))
    assert (dense > 0).all(), "softmax declares exact_zeros=False"
    assert ACTIVATION_REGISTRY[("softmax", None)].normalized
    assert not ACTIVATION_REGISTRY[("softmax", None)].exact_zeros

    for alpha in ratified_entmax_alphas():
        p = entmax_reference(logits, alpha=alpha)
        assert torch.allclose(p.sum(-1), torch.ones(1, dtype=torch.float64), atol=1e-9)
        assert (p == 0).any(), f"entmax({alpha}) declares exact_zeros=True"
        assert ACTIVATION_REGISTRY[("entmax", alpha)].exact_zeros
        assert ACTIVATION_REGISTRY[("entmax", alpha)].normalized


# --- closed world ------------------------------------------------------------

def test_an_unknown_tag_is_a_hard_panic_at_config_construction():
    """Closed-world lookup. A variant either has a row -- and therefore participates
    in every rule immediately -- or nothing runs. There is no default row."""
    with pytest.raises(UnknownActivation, match="CLOSED-WORLD"):
        properties_for(("entmax", 1.25), owner="test")
    with pytest.raises(UnknownActivation, match="CLOSED-WORLD"):
        properties_for(("sigmoid_top_k", 8), owner="test")
    with pytest.raises(UnknownActivation):
        EntmaxActivation(1.25)
    with pytest.raises(UnknownActivation):
        UnionRouting(alpha=3.0)


def test_the_admissible_alphas_are_read_off_the_registry_not_listed_twice():
    assert ratified_entmax_alphas() == (1.5, 2.0)
    assert EntmaxActivation(2).alpha == 2.0
    assert entmax(1.5).tag == ("entmax", 1.5)
    assert softmax().tag == ("softmax", None)


def test_a_registry_row_added_at_runtime_is_immediately_constructible():
    """The extension story, exercised rather than asserted: one row, and the variant
    exists everywhere. If this needed a second edit somewhere, the registry would not
    be the single source of truth."""
    tag = ("entmax", 1.75)
    ACTIVATION_REGISTRY[tag] = ActivationProperties(
        normalized=True, exact_zeros=True, support_differentiable=False, kerneled=False)
    try:
        activation = EntmaxActivation(1.75)
        assert activation.properties.exact_zeros
        assert not activation.properties.kerneled
        assert 1.75 in ratified_entmax_alphas()
    finally:
        del ACTIVATION_REGISTRY[tag]


# --- users never write property metadata -------------------------------------

def test_the_public_constructors_take_no_property_metadata():
    """§4.6 (1): plain tagged constructors only. A caller that could supply
    `normalized=` could supply it wrong, and every validator downstream would then be
    checking a claim instead of a fact."""
    assert [f.name for f in SoftmaxActivation.__dataclass_fields__.values()] == []
    assert [f.name for f in EntmaxActivation.__dataclass_fields__.values()] == ["alpha"]
    for column in ACTIVATION_COLUMNS:
        with pytest.raises(TypeError):
            EntmaxActivation(1.5, **{column: True})


# --- rules read columns, never names -----------------------------------------

def _topology(read, write, *, width=4, tied=True):
    return Topology(
        levels=(LevelConfig(width=width, routing=IndependentRouting(
            tied=tied, read_activation=read, write_activation=write)),))


def test_per_level_and_per_duty_configuration_is_addressable_as_a_grid():
    """§4.6: per-level AND per-duty configuration is what makes level-split pure
    CONFIGURATION (M-3) rather than a code path."""
    topology = _topology(softmax(), entmax(1.5), tied=False)
    assert topology.duty_cells() == ((0, "read"), (0, "write"))
    assert topology.activation(0, "read").tag == ("softmax", None)
    assert topology.activation(0, "write").tag == ("entmax", 1.5)
    assert topology.activation_properties(0, "write").exact_zeros
    with pytest.raises(ValueError, match="duty must be"):
        topology.activation(0, "both")


def test_mass_decay_demands_a_normalized_write_side_loudly():
    """THE property-violation test. `normalized=False` on the write side is refused
    with an error naming the COLUMN that failed and why it matters -- never silently
    accepted, and never phrased as "softmax or entmax is required"."""
    tag = ("entmax", 1.75)
    ACTIVATION_REGISTRY[tag] = ActivationProperties(
        normalized=False, exact_zeros=False, support_differentiable=True, kerneled=False)
    try:
        topology = _topology(softmax(), EntmaxActivation(1.75), tied=False)
        decay = LeafMassDecay(dials=(torch.full((1, 4), 0.5),))
        with pytest.raises(ValueError, match="normalized"):
            decay.validate_against(topology)
    finally:
        del ACTIVATION_REGISTRY[tag]


def test_an_activation_with_a_row_is_configurable_without_widening_an_isinstance_list():
    """Admission is a registry lookup, so a variant with a row is usable at once.
    A duck-typed activation stands in for a future family the shipped classes cannot
    express -- if this needed a second edit in `types.py`, the registry would not be
    the single source of truth it claims to be."""

    class _Future:
        tag = ("future_family", None)

    ACTIVATION_REGISTRY[("future_family", None)] = ActivationProperties(
        normalized=True, exact_zeros=False, support_differentiable=True, kerneled=False)
    try:
        routing = IndependentRouting(
            tied=False, read_activation=_Future(), write_activation=_Future())
        assert routing.activation("read").tag == ("future_family", None)
    finally:
        del ACTIVATION_REGISTRY[("future_family", None)]

    with pytest.raises(UnknownActivation):
        IndependentRouting(tied=False, read_activation=_Future(), write_activation=_Future())


def test_mass_decay_accepts_every_ratified_activation():
    """M2 non-vacuity for the rule above: it must not reject the shipped world."""
    for activation in (softmax(), entmax(1.5), entmax(2.0)):
        topology = _topology(activation, activation)
        LeafMassDecay(dials=(torch.full((1, 4), 0.5),)).validate_against(topology)


def test_union_routing_demands_exact_zeros_for_the_intersection_guarantee():
    tag = ("entmax", 1.75)
    ACTIVATION_REGISTRY[tag] = ActivationProperties(
        normalized=True, exact_zeros=False, support_differentiable=True, kerneled=False)
    try:
        with pytest.raises(ValueError, match="exact_zeros"):
            UnionRouting(alpha=1.75)
    finally:
        del ACTIVATION_REGISTRY[tag]


def test_two_exact_zero_sides_still_require_one_shared_solve():
    with pytest.raises(ValueError, match="exact_zeros"):
        IndependentRouting(tied=False, read_activation=entmax(1.5),
                           write_activation=entmax(1.5))


# --- the not-kerneled surface ------------------------------------------------

class _StubPlan:
    def __init__(self, topology):
        self.topology = topology
        self.normalization = "raw"


def test_an_unratified_activation_dispatches_to_the_reference_with_an_explicit_flag():
    """§4.6: a LABELED RESEARCH SURFACE, not a kernel fallback. The flag is a
    property of the configuration, so it is answerable before the call and without a
    device -- and `dispatch_for` honours it even when everything else says 'cuda'."""
    tag = ("entmax", 1.75)
    ACTIVATION_REGISTRY[tag] = ActivationProperties(
        normalized=True, exact_zeros=True, support_differentiable=False, kerneled=False)
    try:
        topology = _topology(entmax(1.75), entmax(1.75))
        assert not topology.kerneled
        assert topology.unratified_activations == (
            (0, "read", ("entmax", 1.75)), (0, "write", ("entmax", 1.75)))

        dispatch = rola.interface.dispatch_for(
            _StubPlan(topology), torch.zeros(1, 1, 1, 64))
        assert dispatch.backend == "reference"
        assert dispatch.not_kerneled == topology.unratified_activations
    finally:
        del ACTIVATION_REGISTRY[tag]


#: RETIRED at P67 D2-b: `test_the_private_kernel_path_refuses_an_unratified_
#: activation_rather_than_running_it`. Its subject was the retired tiled arm's own
#: activation check -- a refusal at its private entry, so that a caller who
#: bypassed the dispatch could not launch something nobody measured. The chunk arm
#: has no such entry to bypass (`chunk.run` has exactly one caller and it is inside
#: the autograd node, asserted structurally in `tests/unit/test_interface.py`), and
#: the path the check guarded is closed at BOTH ends by rows that already exist:
#: the GATEWAY refuses to produce an unratified activation's routing at all
#: (`rola/routing/entmax/production.py` raises -- MEASURED here while re-anchoring:
#: registering `("entmax", 1.75)` does not make the producer able to emit it), and
#: `rola_op` refuses any bundle the gateway did not derive
#: (`tests/unit/test_api_contract.py`), so no bundle carrying an unratified
#: activation can reach any arm. The DISPATCH half -- an unratified activation goes
#: to the reference arm and is named in `Dispatch.not_kerneled` -- is the row
#: directly above this one.


def test_every_shipped_configuration_is_kerneled():
    for activation in (softmax(), entmax(1.5), entmax(2.0)):
        assert _topology(activation, activation).kerneled


# --- no case list may exist outside the registry -----------------------------

_ALLOWED_TO_NAME_ACTIVATIONS = {
    # The registry itself, and the two tagged constructors that key into it.
    "routing/activations.py",
    "routing/types.py",
    # The producer and the reference lower a tag to a kernel/solve; that dispatch IS
    # a mapping from tag to implementation and cannot be a column.
    "routing/producer.py",
    "routing/reference.py",
    "routing/entmax/production.py",
    "routing/entmax/reference.py",
    # The layer translates the PUBLIC 'dense'/'sparse' vocabulary into activations.
    "layers/rola.py",
    # The calibration AUDITOR constructs fixture topologies to price cells with. It is
    # not a runtime rule -- nothing under `rola/` imports it at module scope and it is
    # reached only by the bench and the planner gates -- and it names ONE activation
    # for every level of every fixture because the pricing is invariant in the routing
    # tag (its own docstring says so). It moved under `rola/` for PACKAGING (an
    # installed wheel has no `benchmarks/` sibling), not because it became runtime.
    "planner/calibration/auditor.py",
}


def test_no_module_outside_the_registry_branches_on_an_activation_type():
    """§4.6 (3): rules state preconditions against COLUMNS, never against tag lists.
    A module that does `isinstance(a, EntmaxActivation)` to decide a POLICY is a
    second, invisible copy of the table.

    The exemptions are the modules that lower a tag to an implementation -- that is a
    dispatch, not a rule, and it is exactly what a registry column cannot express.
    """
    package = pathlib.Path(rola.__path__[0])
    names = {"SoftmaxActivation", "EntmaxActivation"}
    offenders = []
    for path in sorted(package.rglob("*.py")):
        relative = str(path.relative_to(package))
        if relative in _ALLOWED_TO_NAME_ACTIVATIONS:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in names:
                offenders.append(f"{relative}:{node.lineno}:{node.id}")
    assert not offenders, (
        "these modules name an activation class outside the registry: " + str(offenders))


# --- the docs' matrix is generated -------------------------------------------

def test_the_capability_matrix_is_generated_from_the_registry():
    table = capability_table()
    assert table.splitlines()[0] == "| activation | " + " | ".join(ACTIVATION_COLUMNS) + " |"
    assert "| entmax(alpha=1.5) | yes | yes | no | yes |" in table
    assert "| softmax | yes | no | yes | yes |" in table
    assert len(table.splitlines()) == len(ACTIVATION_REGISTRY) + 2


def test_a_new_column_reaches_the_matrix_without_being_typed_anywhere():
    assert tuple(ActivationProperties.__annotations__) == ACTIVATION_COLUMNS
