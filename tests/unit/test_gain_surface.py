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
rectangle for any instantiated `BC` (16, 64), so `THUNDER_RUN_SPEC.md` §6 P6's
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
from rola.routing.side_gain import GainConfig, project_side_gains

_H, _DV = 2, 32
_HIDDEN = _H * _DV

# --- the public flag ---------------------------------------------------------

def test_the_public_flag_maps_onto_the_two_internal_booleans():
    """`gain=True` enables the write gain under both normalizations and the read gain
    only under `raw`. That is the opinionated mapping, and it is the whole public
    surface."""
    assert GainConfig.from_public(True, "raw") == GainConfig(write_gain=True, read_gain=True)
    assert GainConfig.from_public(True, "global") == GainConfig(write_gain=True, read_gain=False)
    assert GainConfig.from_public(False, "raw") == GainConfig(write_gain=False, read_gain=False)
    assert GainConfig.from_public(False, "global") == GainConfig(write_gain=False, read_gain=False)


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
        GainConfig.from_public("auto", "raw")


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

