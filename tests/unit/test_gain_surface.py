# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The gain surface (user ruling, 2026-08-01): one public flag, no dead knobs.

The design claim being gated is not "the gains compute the right numbers" -- the
tier-1 gates do that. It is that **every reachable setting of the configuration is
connected to real behaviour**: a disabled gain is a missing projection column rather
than a pinned parameter, and the one combination that would optimize a parameter with
an identically zero gradient is refused at construction.

**SCOPE NARROWED (post-merge, 2026-08-02).** Six cells built a `states_per_level=[4]`
LAYER to read the surface off a real module. `widths = (4,)` names no BC-aligned owner
rectangle for any instantiated `BC` (16, 64), so the gain-surface contract's
construction-time feasibility validation refuses it -- correctly, and by design: those
cells only ever ran on the fp64 oracle path, which needs no `BC` at all. They are
deleted rather than re-pointed onto an instantiated topology, per the ruling: the
CONFIGURATION surface they gated is exercised here directly against `GainConfig` and
`project_side_gains`, which is where it lives, and re-pointing would have bought a
second, slower copy of that.
"""

from __future__ import annotations

import pytest
import torch

from rola.layers.rola import RoLA
from rola.routing.side_gain import GainConfig, SideGain, project_side_gains

_H, _DV = 2, 32
_HIDDEN = _H * _DV

# --- the public flag ---------------------------------------------------------

def test_the_public_flag_maps_onto_the_one_internal_boolean():
    """`gain` enables the WRITE gain, and there is no second side to enable:
    architecture §4.2 fixes `g_read := 1` under the self-normalizing readout."""
    assert GainConfig.from_public(True) == GainConfig(write_gain=True)
    assert GainConfig.from_public(False) == GainConfig(write_gain=False)


def test_the_public_flag_is_the_only_gain_parameter_on_the_route_producer():
    """A tri-state, or a pair of booleans, would put a research surface in the public
    signature. The fine-grained combinations stay reachable via `gain_config`.

    MIGRATED (API-contract break, 2026-08-05): `gain` moved off `RoLA.__init__`
    onto `RouteProducer.__init__` -- it is a fact about the feature map's side
    gains, not about the layer wrapping it (`RoLA`'s own signature carries no gain
    knob at all any more)."""
    import inspect

    from rola.routing.factors import RouteProducer

    layer_parameters = inspect.signature(RoLA.__init__).parameters
    assert not any(name in layer_parameters for name in ("gain", "write_gain", "read_gain"))

    parameters = inspect.signature(RouteProducer.__init__).parameters
    assert "gain" in parameters and parameters["gain"].default is True
    assert not any(name in parameters for name in ("write_gain", "read_gain"))


def test_the_public_flag_rejects_a_non_bool_rather_than_coercing_it():
    with pytest.raises(TypeError, match="gain must be a bool"):
        GainConfig.from_public("auto")


# --- what the engine actually receives ---------------------------------------

@pytest.mark.parametrize("config,write_is_one", [
    (GainConfig(True), False),
    (GainConfig(False), True),
])
def test_a_missing_column_becomes_the_constant_one_and_there_is_never_a_read_gain(
        config, write_is_one):
    """The engine's contract does not change shape when the gain is ablated: an absent
    write gain is the CONSTANT 1, and there is no read gain in either case -- which is
    what fixes `g_read := 1` in the ratio."""
    torch.manual_seed(5)
    h = torch.randn(2, 6, _HIDDEN, dtype=torch.float64)
    gain_W = torch.randn(3, len(config.layout), _HIDDEN, dtype=torch.float64)
    gains = project_side_gains(h, gain_W, None, write_gain=config.write_gain,
                               projection_dtype=torch.float64)

    assert set(gains) == {"g_write"}
    assert gains["g_write"].shape == (2, 6, 3)
    assert torch.equal(gains["g_write"], torch.ones_like(gains["g_write"])) is write_is_one


# --- the named producer ------------------------------------------------------

def test_the_side_gain_module_owns_the_layout_and_the_projection():
    """`SideGain` is the gain half of the feature map as a MODULE, and it must be the
    same function as the free projection it wraps -- same layout, same parameters,
    same numbers -- so extracting it from the gateway's body moved code and not
    semantics."""
    torch.manual_seed(7)
    module = SideGain(num_heads=3, hidden_size=_HIDDEN, config=GainConfig.from_public(True))
    assert module.layout == ("write",)
    assert module.gain_W.shape == (3, 1, _HIDDEN)
    assert module.gain_bias.shape == (3, 1)

    h = torch.randn(2, 6, _HIDDEN, dtype=torch.float64)
    gains = module(h)
    direct = project_side_gains(
        h, module.gain_W, module.gain_bias, write_gain=module.config.write_gain,
        projection_dtype=torch.float64)
    assert torch.equal(gains["g_write"], direct["g_write"])
    assert "g_read" not in gains


def test_the_side_gain_bias_init_places_the_zero_logit_gain():
    """`bias_init` is the flagged model-level init decision, inverted through
    `softplus` so an all-zero-logit gain starts where the caller asked."""
    module = SideGain(num_heads=2, hidden_size=_HIDDEN,
                      config=GainConfig.from_public(True), bias_init=1.0)
    with torch.no_grad():
        module.gain_W.zero_()
    gains = module(torch.zeros(1, 3, _HIDDEN))
    torch.testing.assert_close(gains["g_write"], torch.ones(1, 3, 2), rtol=1e-6, atol=1e-6)
