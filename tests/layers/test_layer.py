"""The routing vocabulary's own gate (`ROLA_V3_SPEC.md` §1; architecture §1.2, §8, §9.4).

**MIGRATED (API-contract break, 2026-08-05).** The subject this file gated used to be
one public dataclass, `rola.layers.LevelRoutingConfig`, resolved by a free function
`resolve_level_routing_configs`. The break DELETES both: "the routing vocabulary is
the tag vocabulary directly" -- `IndependentRouting(tied=..., read_activation=...,
write_activation=...)` and `UnionRouting(alpha)` (`rola.routing.types`), with
`rola.dense_routing()`/`rola.union_routing(alpha)` as the two convenience
constructors. There is no longer one dataclass to gate; there are two, and the
broadcast/length-validation the free function used to do moved onto
`SimplexProducer`, which is where a routing tag actually meets a topology now. Every
assertion below is the OLD one, re-pointed at its new owner -- none weakened. Where
the new validation raises a DIFFERENT exception type than the old string-keyed
config did (`TypeError` instead of `ValueError` for a non-activation `read`/`write`
value), that is noted at the row: the new vocabulary takes activation OBJECTS
(`softmax()`/`entmax(alpha)`), not strings, so a bad value fails "not an activation"
(`TypeError`) rather than "not a recognized string" (`ValueError`) -- a stricter
failure mode, not a looser one.
"""

from __future__ import annotations

import inspect

import pytest

from rola.routing.producer import SimplexProducer
from rola.routing.types import IndependentRouting, UnionRouting, entmax, softmax

# ---------------------------------------------------------------------------
# `IndependentRouting`/`UnionRouting` validation -- the tag vocabulary's own gate.
# ---------------------------------------------------------------------------


def test_independent_routing_has_only_public_fields():
    assert set(inspect.signature(IndependentRouting).parameters) == {
        'tied', 'read_activation', 'write_activation'}
    with pytest.raises(TypeError):
        IndependentRouting(probability='softmax')


def test_union_routing_has_only_public_fields():
    assert set(inspect.signature(UnionRouting).parameters) == {'alpha'}
    with pytest.raises(TypeError):
        UnionRouting(probability='softmax')


@pytest.mark.parametrize('value', [0, 1, 'true', None])
def test_independent_routing_requires_an_exact_boolean_tied(value):
    with pytest.raises(TypeError):
        IndependentRouting(tied=value, read_activation=softmax(), write_activation=softmax())


@pytest.mark.parametrize('side', ['read_activation', 'write_activation'])
@pytest.mark.parametrize('value', ['softmax', 'entmax', 'SPARSE', 1, None])
def test_independent_routing_rejects_non_activation_side_values(side, value):
    """The old config took `read='sparse'`/`write='dense'` strings and rejected an
    unrecognized one with `ValueError`. The new vocabulary takes activation OBJECTS
    (`softmax()`/`entmax(alpha)`) -- a bare string, int or `None` has no `.tag` to
    look up in the registry, so the failure is `TypeError` ("not an activation"),
    not `ValueError` ("not a recognized string")."""
    kwargs = dict(tied=False, read_activation=softmax(), write_activation=softmax())
    kwargs[side] = value
    with pytest.raises(TypeError):
        IndependentRouting(**kwargs)


def test_a_producer_broadcasts_one_routing_tag_and_validates_a_lists_length():
    """The old free function `resolve_level_routing_configs(config_or_list, D)`
    broadcast one tag across every level and validated a per-level list's length.
    That job now belongs to `SimplexProducer`, the only place a routing tag meets a
    concrete level count."""
    tag = IndependentRouting(True, softmax(), entmax(1.5))
    broadcast = SimplexProducer((4, 4, 4), tag, hidden_size=8, num_heads=1)
    assert broadcast.levels == (tag, tag, tag)

    per_level = [IndependentRouting(False, softmax(), softmax()), UnionRouting(1.5)]
    matched = SimplexProducer((4, 4), per_level, hidden_size=8, num_heads=1)
    assert matched.levels == tuple(per_level)

    with pytest.raises(ValueError, match=r"1 routing tags for 2 widths"):
        SimplexProducer((4, 4), [tag], hidden_size=8, num_heads=1)


@pytest.mark.parametrize('value', [True, 1, 1.8, float('inf'), float('nan')])
def test_union_routing_validates_alpha_like_entmax(value):
    error = TypeError if value is True else ValueError
    with pytest.raises(error):
        UnionRouting(alpha=value)
    with pytest.raises(error):
        entmax(value)
