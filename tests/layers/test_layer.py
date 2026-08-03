"""V3 Stage 2 layer tests (`ROLA_V3_SPEC.md` §10 Stage 2; architecture §1.2, §8, §9.4).

Covers the level-routing configuration surface: the public dataclass, its export
identity, and `resolve_level_routing_configs`.

**SCOPE NARROWED (post-merge, 2026-08-02).** Nine cells built a `states_per_level=[4]`
(or `[3, 5]`, `[6]`) LAYER -- the deletion inventory, the fp64 oracle
forward/backward grid, the budget-actuator gradient walk, the state-size and
temperature-init cells, and the routing side-channel cell. `widths = (4,)` / `(3, 5)`
/ `(6,)` name no BC-aligned owner rectangle for any instantiated `BC` (16, 64), so
`THUNDER_RUN_SPEC.md` §6 P6's construction-time feasibility validation refuses them at
`__init__`. That is P6's stated intent -- "construction-time validation gives
construction-time errors, never plan-time surprises" -- and an explicit `state_block=`
pin does not rescue them, because a pin is alignment-validated against the same
ladder. The cells only ever ran the fp64 ORACLE path, which needs no `BC` at all, so
re-pointing them onto an instantiated topology (`(16,)`, `(16, 16)`) would have changed
BOTH what they cover and their cost. Per the ruling they are DELETED; the oracle
forward/backward path is gated on instantiated topologies under `tier1_semantics` and
the dispatch cells in `test_layer_dispatch.py`.
"""

from __future__ import annotations

import inspect

import pytest

from rola.layers import LevelRoutingConfig as ExportedLevelRoutingConfig
from rola.layers.rola import LevelRoutingConfig, resolve_level_routing_configs

# ---------------------------------------------------------------------------
# `LevelRoutingConfig` validation, MOVED VERBATIM from the deleted
# `tests/layers/test_rola_level_routing_config.py` (rola-v3 purge, findings §2.7).
#
# That file mixed two subjects: the public config dataclass, which is live V3
# code (`rola.layers.LevelRoutingConfig`, resolved into the plan that
# `producer_v3` consumes), and Q/K-era layer construction
# (`RoLA(head_k_dim=...)`), which spec §1.2 deleted. Only the second subject
# died with the purge. These five functions never construct a layer -- they are
# the config's own gate and nothing else -- so they are carried across with
# their assertions unchanged rather than re-derived: a rewritten check would
# only prove the rewrite self-consistent.
# ---------------------------------------------------------------------------

def test_level_routing_config_has_only_public_fields():
    assert ExportedLevelRoutingConfig is LevelRoutingConfig
    assert set(inspect.signature(LevelRoutingConfig).parameters) == {'read', 'write', 'tied', 'alpha'}
    with pytest.raises(TypeError):
        LevelRoutingConfig(probability='softmax')


@pytest.mark.parametrize('value', [0, 1, 'true', None])
def test_level_routing_config_requires_an_exact_boolean_tied(value):
    with pytest.raises(TypeError):
        LevelRoutingConfig(tied=value)


@pytest.mark.parametrize('side', ['read', 'write'])
@pytest.mark.parametrize('value', ['softmax', 'entmax', 'SPARSE', 1, None])
def test_level_routing_config_rejects_non_public_side_values(side, value):
    kwargs = {side: value}
    with pytest.raises(ValueError):
        LevelRoutingConfig(**kwargs)


def test_resolve_level_routing_configs_broadcasts_and_validates_length():
    config = LevelRoutingConfig(read='dense', write='sparse', tied=True)
    assert resolve_level_routing_configs(config, D=3) == [config, config, config]
    configs = [LevelRoutingConfig(), LevelRoutingConfig(tied=True)]
    assert resolve_level_routing_configs(configs, D=2) == configs
    with pytest.raises(ValueError, match='exactly D=2'):
        resolve_level_routing_configs([config], D=2)


@pytest.mark.parametrize('value', [True, 1, 1.8, float('inf'), float('nan')])
def test_level_routing_config_validates_alpha_like_entmax(value):
    error = TypeError if value is True else ValueError
    with pytest.raises(error):
        LevelRoutingConfig(alpha=value)
