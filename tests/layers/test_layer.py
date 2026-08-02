"""V3 Stage 2 layer tests (`ROLA_V3_SPEC.md` §10 Stage 2; architecture §1.2, §8, §9.4).

Covers: the deletion inventory (no Q/K symbol importable from the layer's
public surface), the layer's forward/backward path at fp64 through the
canonical Stage 1 oracle (`naive_rola`), and gradient flow to every
actuator parameter (side gains, budget temperature/usage-bias heads).
"""

from __future__ import annotations

import inspect

import pytest
import torch

from rola.layers import LevelRoutingConfig as ExportedLevelRoutingConfig
from rola.layers.rola import LevelRoutingConfig, RoLA, resolve_level_routing_configs
from rola.routing.producer import BudgetConditioning

_DELETED_Q_K_SYMBOLS = (
    "head_k_dim", "qkv_proj", "qk_norm", "q_norm", "k_norm",
    "q_conv1d", "k_conv1d", "_feature_map", "proj_qk", "feat_dim",
)


def test_no_q_k_symbol_is_importable_from_the_layer_public_surface():
    """Spec §1.2 deletion inventory: none of these may survive the layer, its
    constructor signature, or its instance attributes after construction."""
    source = inspect.getsource(RoLA)
    for symbol in _DELETED_Q_K_SYMBOLS:
        assert symbol not in source, f"deleted Q/K symbol {symbol!r} still present in RoLA source"

    layer = RoLA(hidden_size=8, num_heads=2, head_v_dim=4, levels_per_head=1, states_per_level=4)
    for symbol in _DELETED_Q_K_SYMBOLS:
        assert not hasattr(layer, symbol), f"RoLA instance still exposes deleted attribute {symbol!r}"

    signature_params = set(inspect.signature(RoLA.__init__).parameters)
    for symbol in _DELETED_Q_K_SYMBOLS:
        assert symbol not in signature_params


def test_retained_symbols_still_exist():
    """Spec §1.2: `v_conv1d`/route convolutions are explicitly RETAINED, not deleted by association."""
    layer = RoLA(
        hidden_size=8, num_heads=2, head_v_dim=4, levels_per_head=1, states_per_level=[6],
        level_routing_configs=LevelRoutingConfig(read='sparse', write='sparse'),
        use_short_conv=True,
    )
    assert hasattr(layer, "v_conv1d")
    assert hasattr(layer, "write_route_conv1d")
    assert hasattr(layer, "read_route_conv1d")  # untied sparse/sparse level -> union -> untied


@pytest.mark.parametrize("normalization", ["raw", "global"])
@pytest.mark.parametrize("decay", ["none", "leaf_mass"])
def test_layer_forward_backward_at_fp64_through_the_canonical_oracle(normalization, decay):
    """Item 4: run the layer's forward (which dispatches to `naive_rola`, the only
    V3 execution backend at Stage 2 -- see the layer module docstring) at fp64 and
    check gradients reach input, router, and gain parameters."""
    torch.manual_seed(0)
    layer = RoLA(
        hidden_size=10, num_heads=2, head_v_dim=3, levels_per_head=2, states_per_level=[3, 5],
        state_norm=normalization, decay=decay,
    ).double()
    x = torch.randn(2, 7, 10, dtype=torch.float64, requires_grad=True)
    o, _, _ = layer(x)
    assert o.dtype == torch.float64
    assert torch.isfinite(o).all()
    o.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in layer.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} gradient is non-finite"


def test_layer_forward_with_budget_conditioning_reaches_all_actuator_parameters():
    torch.manual_seed(0)
    budget = BudgetConditioning(budgeted_write_levels=(1,), budgeted_read_levels=(1,), usage_bias_enabled=True)
    layer = RoLA(
        hidden_size=10, num_heads=2, head_v_dim=3, levels_per_head=2, states_per_level=[3, 5],
        level_routing_configs=LevelRoutingConfig(read='sparse', write='sparse'),
        state_norm="global", budget=budget,
    ).double()
    x = torch.randn(2, 20, 10, dtype=torch.float64, requires_grad=True)
    r = torch.full((2,), 0.5, dtype=torch.float64)
    o, _, _ = layer(x, budget_r=r)
    o.sum().backward()
    for name, param in layer.named_parameters():
        assert param.grad is not None, f"{name} received no gradient under active budget conditioning"


def test_budget_r_without_budget_config_raises():
    layer = RoLA(hidden_size=8, num_heads=2, head_v_dim=4, levels_per_head=1, states_per_level=4)
    x = torch.randn(1, 5, 8)
    with pytest.raises(ValueError, match="BudgetConditioning"):
        layer(x, budget_r=torch.full((1,), 0.5))


def test_layer_state_size_matches_topology_capacity():
    layer = RoLA(hidden_size=8, num_heads=2, head_v_dim=4, levels_per_head=2, states_per_level=[3, 5])
    stats = layer.get_stats()
    assert stats['num_chunks'] == 15  # 3 * 5
    assert stats['n_heads'] == 2
    assert stats['d_v'] == 4


def test_zero_config_temperature_is_exactly_one_at_init():
    """D5 regression: a freshly budgeted level must not itself perturb an otherwise-baseline
    model before any training signal exists. Before the fix, `temp_b` initialized to raw
    zero, giving `c = softplus(0) + 1e-2 = 0.7031` at an all-zero conditioning input --
    merely configuring a budget changed the routing distribution. With the default
    `temp_bias_init=1.0`, an all-zero `(r, slack)` input (`temp_W` is always zero-init) must
    give exactly `c=1`."""
    from rola.routing.producer import _TEMPERATURE_FLOOR

    budget = BudgetConditioning(budgeted_write_levels=(0,))
    layer = RoLA(
        hidden_size=8, num_heads=2, head_v_dim=4, levels_per_head=1, states_per_level=4,
        level_routing_configs=LevelRoutingConfig(read='sparse', write='sparse'),
        budget=budget,
    )
    assert torch.all(layer.temp_W == 0.0)
    c = torch.nn.functional.softplus(layer.temp_b) + _TEMPERATURE_FLOOR
    assert torch.allclose(c, torch.ones_like(c), atol=1e-6)


def test_double_cast_temperature_is_exactly_one_not_fp32_rounded():
    """D5r regression: `temp_b` is materialized at the default (fp32) dtype, so a naive
    `.double()` cast used to carry the fp32-rounded `init_temp_bias` value forward and land
    on `c = 1 - 1.62e-8`, not exactly 1. After the `_apply` fix, `.double()` must recompute
    the exact fp64 target rather than merely widen the stale fp32 constant, so a budgeted
    layer cast to fp64 (the oracle-equivalence regime) matches an unbudgeted fp64 baseline
    bit-for-bit at init."""
    from rola.routing.producer import _TEMPERATURE_FLOOR

    budget = BudgetConditioning(budgeted_write_levels=(0,))
    layer = RoLA(
        hidden_size=8, num_heads=2, head_v_dim=4, levels_per_head=1, states_per_level=4,
        level_routing_configs=LevelRoutingConfig(read='sparse', write='sparse'),
        budget=budget,
    ).double()
    assert layer.temp_b.dtype == torch.float64
    c = torch.nn.functional.softplus(layer.temp_b) + _TEMPERATURE_FLOOR
    assert torch.equal(c, torch.ones_like(c))


def test_no_dead_routing_output_side_channel_after_forward():
    """D6 regression: the layer must not retain the full routing output (an autograd-graph-
    attached side channel with no reader) on the module instance across forward calls."""
    torch.manual_seed(0)
    layer = RoLA(hidden_size=8, num_heads=2, head_v_dim=4, levels_per_head=1, states_per_level=4)
    x = torch.randn(1, 5, 8)
    layer(x)
    assert not hasattr(layer, "_last_routing_output")


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
