"""The routing vocabulary's own gate.

The vocabulary is ONE declaration: ``Routing(topology=(widths...), write=(an activation a level), read=(a tuple a
read head, or one for every head))``, each entry an activation OBJECT (``softmax()``, ``entmax(alpha)``) or, on the
read side, the ``tied()`` marker (``fsla.routing.types``). Every row here holds one refusal that vocabulary owes a
caller: a value that is not an activation, a width that cannot be a width, a depth past the built bound.
"""

from __future__ import annotations

import pytest

from fsla import Routing
from fsla.routing.types import entmax, softmax, tied


def test_the_routing_declares_its_topology_activations_and_gain_and_nothing_else():
    routing = Routing(topology=(4, 8), write=(softmax(), entmax(1.5)), read=((softmax(), tied()),))
    assert routing.widths == (4, 8)
    assert routing.topology.write == (softmax(), entmax(1.5)) and routing.topology.read == ((softmax(), tied()),)
    with pytest.raises(TypeError):
        Routing(topology=(4,), write=(softmax(),), read=((softmax(),),), probability="softmax")


@pytest.mark.parametrize("side", ["read", "write"])
@pytest.mark.parametrize("value", ["softmax", "entmax", "SPARSE", 1, None])
def test_the_routing_rejects_a_non_activation_entry(side, value):
    """An entry is an activation OBJECT (`softmax()`/`entmax(alpha)`) -- a bare string, int or `None` has no `.tag` to
    look up in the registry, so the failure is `TypeError` ("not an activation"), not `ValueError` ("not a recognized
    string")."""
    entries = {"write": (softmax(),), "read": ((softmax(),),)}
    entries[side] = (value,) if side == "write" else ((value,),)
    with pytest.raises(TypeError):
        Routing(topology=(4,), **entries)


@pytest.mark.parametrize("value", [True, 1, 1.8, float("inf"), float("nan")])
def test_entmax_validates_its_alpha(value):
    error = TypeError if value is True else ValueError
    with pytest.raises(error):
        entmax(value)


@pytest.mark.parametrize("widths,match", [((0,), ">= 1"), ((257,), "<= 256"), ((4,) * 5, "D must be <= 4")])
def test_the_routing_rejects_a_width_or_depth_past_the_built_bounds(widths, match):
    with pytest.raises(ValueError, match=match):
        Routing(topology=widths, write=(softmax(),) * len(widths), read=((softmax(),) * len(widths),))
