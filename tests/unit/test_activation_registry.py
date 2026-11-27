# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The activation registry.

The point of the registry is that activation properties live in ONE table and every
rule reads columns from it. So most of what is gated here is not "does softmax have
the right flags" -- it is the *structural* claims: that the table is the only source,
that an unknown tag cannot run, that rules are stated against columns rather than
against activation names.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import torch

import fsla
from fsla.routing.activations import (
    ACTIVATION_COLUMNS,
    ACTIVATION_REGISTRY,
    ActivationProperties,
    UnknownActivation,
    properties_for,
    ratified_entmax_alphas,
)
from fsla.routing.types import (
    EntmaxActivation,
    LeafMassDecay,
    SoftmaxActivation,
    Topology,
    entmax,
    softmax,
    tied,
)

# --- the table itself --------------------------------------------------------


def test_v0_1_ratifies_exactly_the_two_declared_activation_families():
    """v0.1 ships exactly two ratified families, plus the opaque row.

    A fourth row appearing without a version bump and a docs change is the thing this
    catches. ``("opaque", None)`` is the row a level filled by a FOREIGN producer reads,
    and it is a row rather than a special case precisely so that every rule quantifying
    over cells covers such a level without knowing it is one.
    """
    assert set(ACTIVATION_REGISTRY) == {
        ("softmax", None), ("entmax", 1.5), ("entmax", 2.0), ("opaque", None)}


def test_every_row_declares_every_column():
    for tag, row in ACTIVATION_REGISTRY.items():
        assert isinstance(row, ActivationProperties), tag
        for column in ACTIVATION_COLUMNS:
            assert isinstance(getattr(row, column), bool), f"{tag}.{column}"


def test_the_declared_properties_are_the_ones_the_maps_actually_have():
    """The table is only worth reading if it is true. Checked against the reference
    implementations, not against the docstrings that describe them."""
    logits = torch.tensor([[3.0, 0.5, -2.0, -4.0]], dtype=torch.float64)
    from fsla.routing.entmax.reference import entmax as entmax_reference

    dense = torch.softmax(logits, dim=-1)
    assert (dense > 0).all(), "softmax declares exact_zeros=False"
    assert not ACTIVATION_REGISTRY[("softmax", None)].exact_zeros

    assert (dense >= 0).all() and ACTIVATION_REGISTRY[("softmax", None)].nonnegative

    for alpha in ratified_entmax_alphas():
        p = entmax_reference(logits, alpha=alpha)
        assert (p == 0).any(), f"entmax({alpha}) declares exact_zeros=True"
        assert ACTIVATION_REGISTRY[("entmax", alpha)].exact_zeros
        assert (p >= 0).all() and ACTIVATION_REGISTRY[("entmax", alpha)].nonnegative


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
    ACTIVATION_REGISTRY[tag] = ActivationProperties(exact_zeros=True, nonnegative=True)
    try:
        activation = EntmaxActivation(1.75)
        assert activation.properties.exact_zeros
        assert 1.75 in ratified_entmax_alphas()
    finally:
        del ACTIVATION_REGISTRY[tag]


# --- users never write property metadata -------------------------------------

def test_the_public_constructors_take_no_property_metadata():
    """Plain tagged constructors only. A caller that could supply
    `exact_zeros=` could supply it wrong, and every validator downstream would then be
    checking a claim instead of a fact."""
    assert [f.name for f in SoftmaxActivation.__dataclass_fields__.values()] == []
    assert [f.name for f in EntmaxActivation.__dataclass_fields__.values()] == ["alpha"]
    for column in ACTIVATION_COLUMNS:
        with pytest.raises(TypeError):
            EntmaxActivation(1.5, **{column: True})


# --- rules read columns, never names -----------------------------------------

def _topology(read, write, *, width=4):
    """One level: ``write`` its write activation, ``read`` its one read head's entry (an activation or ``tied()``)."""
    return Topology(widths=(width,), write=(write,), read=((read,),))


def test_per_level_and_per_side_configuration_is_addressable_as_a_grid():
    """Per-level AND per-side configuration -- and per read head -- is what makes level-split pure
    CONFIGURATION (M-3) rather than a code path."""
    topology = Topology(widths=(4, 4), write=(entmax(1.5), softmax()),
                        read=((softmax(), entmax(2.0)), (tied(), softmax())))
    assert topology.read_activation(0, 0).tag == ("softmax", None)
    assert topology.read_activation(1, 0).tag == ("entmax", 1.5)
    assert topology.write[0].tag == ("entmax", 1.5)
    assert [topology.can_zero(side, level) for side in ("read", "write") for level in range(2)] == [
        True, True, True, False]
    with pytest.raises(ValueError, match="side must be"):
        topology.can_zero("both", 0)


class _Future:
    tag = ("future_family", None)


def test_an_activation_with_a_row_is_configurable_without_widening_an_isinstance_list():
    """Admission to the topology is a registry lookup, so a variant with a row is usable at once, on either side and
    under a tied read. A duck-typed activation stands in for a future family the shipped classes cannot express -- if
    this needed a second edit in `types.py`, the registry would not be the single source of truth it claims to be."""
    ACTIVATION_REGISTRY[("future_family", None)] = ActivationProperties(exact_zeros=False, nonnegative=True)
    try:
        topology = _topology(_Future(), _Future())
        assert topology.read_activation(0, 0).tag == ("future_family", None)
        assert _topology(tied(), _Future()).read_activation(0, 0).tag == ("future_family", None)
    finally:
        del ACTIVATION_REGISTRY[("future_family", None)]

    with pytest.raises(UnknownActivation):
        _topology(_Future(), softmax())


def test_mass_decay_accepts_every_ratified_activation():
    """The decay asks nothing of a level's activation: its clock is the write rows' own shares
    (docs/CONSTITUTION.md#r23)."""
    for activation in (softmax(), entmax(1.5), entmax(2.0)):
        topology = _topology(tied(), activation)
        LeafMassDecay(dials=(torch.full((1, 4), 0.5),)).validate_against(topology)


def test_two_exact_zero_sides_of_a_level_are_two_solves_untied_and_one_solve_tied():
    """Stated against the `exact_zeros` column: an untied read that makes zeros over a write that does is its own solve
    with its own support words and its own columns of the projection; a tied one has neither."""
    untied = fsla.Routing(topology=(4,), write=(entmax(1.5),), read=((entmax(1.5),),)).layout(1)
    assert (untied.write_support_offsets, untied.read_support_offsets) == ((0,), (0,))
    assert untied.read_columns(0, 0) == slice(4, 8) and untied.packed_router_width == 8
    shared = fsla.Routing(topology=(4,), write=(entmax(1.5),), read=((tied(),),)).layout(1)
    assert (shared.write_support_offsets, shared.read_support_offsets) == ((0,), (-1,))
    assert shared.read_columns(0, 0) is None and shared.packed_router_width == 4


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
    # The layer names activations where it validates the decay source against them.
    "layer.py",
}


def test_no_module_outside_the_registry_branches_on_an_activation_type():
    """Rules state preconditions against COLUMNS, never against tag lists.
    A module that does `isinstance(a, EntmaxActivation)` to decide a POLICY is a
    second, invisible copy of the table.

    The exemptions are the modules that lower a tag to an implementation -- that is a
    dispatch, not a rule, and it is exactly what a registry column cannot express.
    """
    package = pathlib.Path(fsla.__path__[0])
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
