"""THE PRODUCER'S DENSITY SWEEP: the logit gain is the knob that moves realized routing's density, as it does in
training, so a cell set that claims to cover the distribution has to show the knob moving it.

The sweep is four central layer cells (`SWEEP`): one entmax layer at four logit gains on one input, the cells alike in
every field but the gain. Each is built by `measure.cells.layer.build`, and its REALIZED write density -- the
exact-nonzero fraction of the write side's amplitudes -- must fall monotonically as the gain rises, across at least a
factor of four.
"""
from __future__ import annotations

import pytest
import torch

from fsla.routing.types import activation_walks
from measure.cells import by_name
from measure.cells.layer import build

#: THE SWEEP, by rising gain
SWEEP = ("layer-alt-entmax1.5-64x64-T128-h4-g0.25-fp32", "layer-alt-entmax1.5-64x64-T128-h4-fp32",
         "layer-alt-entmax1.5-64x64-T128-h4-g4-fp32", "layer-alt-entmax1.5-64x64-T128-h4-g16-fp32")

pytestmark = [pytest.mark.cuda,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="the producer is built on the GPU")]


def _write_density(routes) -> float:
    #: only the write levels the producer walks by lists (exact zeros): a dense level's zeros are values that rounded to zero
    walked = activation_walks(routes.topology)[1]
    rows = [routes.rows(level, "write") for level in range(routes.topology.D) if walked[level]]
    return sum(int((r != 0).sum()) for r in rows) / sum(r.numel() for r in rows)


@pytest.mark.requires_kernel("produce")
def test_the_logit_gain_sweeps_the_realized_density():
    import dataclasses

    sweep = [by_name(name) for name in SWEEP]
    assert [c.gain for c in sweep] == sorted(c.gain for c in sweep), "the sweep is listed by rising gain"
    #: a cell's note is prose about it, and each of these names its gain: the input is every other field
    assert len({dataclasses.replace(c, gain=1.0, note="") for c in sweep}) == 1, "the sweep moves the gain alone, on one input"
    densities = [_write_density(build(c)["routes"]) for c in sweep]
    assert all(a >= b for a, b in zip(densities, densities[1:])), (
        f"realized write densities {densities} do not fall as the gain rises; the gain is not the density knob")
    assert densities[0] >= 4.0 * max(densities[-1], 1e-9), (
        f"realized write densities {densities} span less than 4x; the sweep does not move the distribution")
