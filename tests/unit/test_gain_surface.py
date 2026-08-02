# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The gain surface (user ruling, 2026-08-01): one public flag, no dead knobs.

The design claim being gated is not "the gains compute the right numbers" -- the
tier-1 gates do that. It is that **every reachable setting of the configuration is
connected to real behaviour**: a disabled gain is a missing projection column rather
than a pinned parameter, and the one combination that would optimize a parameter with
an identically zero gradient is refused at construction.
"""

from __future__ import annotations

import pytest
import torch

from rola.layers.rola import RoLA
from rola.routing.side_gain import GainConfig, gain_layout, project_side_gains

_H, _DV = 2, 32
_HIDDEN = _H * _DV


def _layer(**kwargs):
    return RoLA(hidden_size=_HIDDEN, num_heads=_H, head_v_dim=_DV,
                levels_per_head=1, states_per_level=[4], layer_idx=0, **kwargs)


# --- the public flag ---------------------------------------------------------

def test_the_public_flag_maps_onto_the_two_internal_booleans():
    """`gain=True` enables the write gain under both normalizations and the read gain
    only under `raw`. That is the opinionated mapping, and it is the whole public
    surface."""
    assert GainConfig.from_public(True, "raw") == GainConfig(write_gain=True, read_gain=True)
    assert GainConfig.from_public(True, "global") == GainConfig(write_gain=True, read_gain=False)
    assert GainConfig.from_public(False, "raw") == GainConfig(write_gain=False, read_gain=False)
    assert GainConfig.from_public(False, "global") == GainConfig(write_gain=False, read_gain=False)


def test_the_public_flag_is_the_only_gain_parameter_on_the_layer():
    """A tri-state, or a pair of booleans, would put a research surface in the public
    signature. The fine-grained combinations stay reachable via `gain_config`."""
    import inspect

    parameters = inspect.signature(RoLA.__init__).parameters
    assert "gain" in parameters and parameters["gain"].default is True
    assert not any(name in parameters for name in ("write_gain", "read_gain"))


@pytest.mark.parametrize("norm,expected", [("raw", ("write", "read")), ("global", ("write",))])
def test_the_default_layer_projects_the_sides_the_normalization_admits(norm, expected):
    layer = _layer(state_norm=norm)
    assert layer.gain_sides == expected
    assert layer.gain_W.shape == (_H, len(expected), _HIDDEN)
    assert layer.has_read_gain is (norm == "raw")


@pytest.mark.parametrize("norm", ["raw", "global"])
def test_disabling_the_gain_removes_the_columns_rather_than_pinning_them(norm):
    """The no-dead-knobs rule, made structural: an ablated gain has NO parameter, so
    there is nothing for an optimizer to move and nothing for a gradient to be zero
    on. A pinned-to-1 parameter would look identical in the forward and waste a
    gradient forever."""
    layer = _layer(state_norm=norm, gain=False)
    assert layer.gain_sides == ()
    assert layer.gain_W.numel() == 0 and layer.gain_bias.numel() == 0


# --- the one refusal ---------------------------------------------------------

def test_a_read_gain_under_global_is_refused_at_construction():
    """The identically-zero-gradient combination. Refused where it is configured, not
    discovered later as a parameter that never moves."""
    with pytest.raises(ValueError, match="identically zero gradient"):
        GainConfig(write_gain=True, read_gain=True).validate("global")
    with pytest.raises(ValueError, match="identically zero gradient"):
        _layer(state_norm="global", gain_config=GainConfig(write_gain=True, read_gain=True))
    with pytest.raises(ValueError, match="identically zero gradient"):
        gain_layout("global", read_gain=True)


def test_the_public_flag_can_never_reach_the_refused_combination():
    """The mapping's job: the public surface has no way to ask for the dead knob."""
    for gain in (True, False):
        for norm in ("raw", "global"):
            GainConfig.from_public(gain, norm).validate(norm)
            _layer(state_norm=norm, gain=gain)


def test_the_public_flag_rejects_a_non_bool_rather_than_coercing_it():
    with pytest.raises(TypeError, match="gain must be a bool"):
        GainConfig.from_public("auto", "raw")


# --- the research surface ----------------------------------------------------

def test_the_internal_config_reaches_the_combinations_the_public_flag_does_not():
    """`raw` with a write gain and no read gain is a legitimate ablation and stays
    constructible -- as an internal config object, not as a public parameter."""
    layer = _layer(state_norm="raw", gain_config=GainConfig(write_gain=True, read_gain=False))
    assert layer.gain_sides == ("write",)
    assert layer.has_read_gain is False

    layer = _layer(state_norm="raw", gain_config=GainConfig(write_gain=False, read_gain=True))
    assert layer.gain_sides == ("read",)


# --- what the engine actually receives ---------------------------------------

@pytest.mark.parametrize("config,norm,write_is_one,read_shape", [
    (GainConfig(True, True), "raw", False, "tensor"),
    (GainConfig(False, True), "raw", True, "tensor"),
    (GainConfig(True, False), "raw", False, "tensor"),
    (GainConfig(False, False), "raw", True, "tensor"),
    (GainConfig(True, False), "global", False, "none"),
    (GainConfig(False, False), "global", True, "none"),
])
def test_a_missing_column_becomes_the_constant_one_and_global_still_has_no_read_gain(
        config, norm, write_is_one, read_shape):
    """The engine's contract does not change shape when a gain is ablated. Under
    `raw` the read channel still exists, so an absent read gain is the CONSTANT 1
    (the oracle requires an explicit `g_read` there); under `global` there is no read
    gain at all and it stays `None`, which is what fixes `g_read := 1` in the ratio."""
    torch.manual_seed(5)
    h = torch.randn(2, 6, _HIDDEN, dtype=torch.float64)
    layout = config.validate(norm).layout
    gain_W = torch.randn(3, len(layout), _HIDDEN, dtype=torch.float64)
    gains = project_side_gains(
        h, gain_W, None, normalization=norm, write_gain=config.write_gain,
        read_gain=config.read_gain, projection_dtype=torch.float64)

    assert gains["g_write"].shape == (2, 6, 3)
    assert torch.equal(gains["g_write"], torch.ones_like(gains["g_write"])) is write_is_one
    if read_shape == "none":
        assert gains["g_read"] is None
    else:
        assert gains["g_read"].shape == (2, 6, 3)
        assert (gains["g_read"] > 0).all()


@pytest.mark.parametrize("norm", ["raw", "global"])
def test_an_ablated_write_gain_leaves_the_level_mass_riding_the_mass_channel(norm):
    """`write_gain=False` removes the LEARNED magnitude, not the algebra: the level
    masses still ride the write channel, so `g_write` is the mass product and the
    ablation really is 'salience through routing only'."""
    layer = _layer(state_norm=norm, gain_config=GainConfig(write_gain=False, read_gain=False))
    layer.eval()
    x = torch.randn(1, 16, _HIDDEN)
    with torch.no_grad():
        out, _, _ = layer(x)
    assert torch.isfinite(out).all()
    assert not any(p.numel() for p in (layer.gain_W, layer.gain_bias))
