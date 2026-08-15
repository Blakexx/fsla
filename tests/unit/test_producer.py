"""V3 Stage 2 producer tests (`ROLA_V3_SPEC.md` §10 Stage 2).

Validates `rola.routing.producer` against the ONE canonical
ground truth this branch authorizes: the Stage 1 fp64 oracle
(`rola.ops.naive.naive_rola`). Per the "matching-the-naive
anti-pattern" standing rule, nothing here is checked against a second,
freshly written reference -- every assertion either (a) feeds the producer's
output through the oracle and checks numerical properties the oracle itself
guarantees (finite output, differentiability), or (b) checks an algebraic
identity of the producer's own contract (mass-fold correctness, simplex
normalization) that is checkable without an independent implementation.

The architecture §4.5 conditioning surface this file used to also cover
(a temperature actuator, a budget embedding, a usage-conditioned bias) is
DESCOPED (queue P29, 2026-08-06) -- see `docs/internals/DELETIONS.md`.
"""

from __future__ import annotations

import pytest
import torch
from conftest import produce_routing, resolved_routing_from_topology

from rola.ops.naive import naive_rola
from rola.routing.types import (
    IndependentRouting,
    LevelConfig,
    SoftmaxActivation,
    Topology,
    UnionRouting,
)


def _topology(mixed: bool = False) -> Topology:
    levels = [
        LevelConfig(4, IndependentRouting(False, SoftmaxActivation(), SoftmaxActivation())),
        LevelConfig(6, UnionRouting(1.5)),
    ]
    if mixed:
        levels.append(LevelConfig(3, IndependentRouting(True, SoftmaxActivation(), SoftmaxActivation())))
    return Topology(levels=tuple(levels), d_v=5)


def _params(topology: Topology, *, B, L, H, dm, dtype, seed=0, gain_sides=2):
    torch.manual_seed(seed)
    resolved = resolved_routing_from_topology(topology)
    x = torch.randn(B, L, dm, dtype=dtype, requires_grad=True)
    route_W = torch.randn(H, dm, resolved.packed_router_width, dtype=dtype, requires_grad=True)
    gain_W = torch.randn(H, gain_sides, dm, dtype=dtype, requires_grad=True)
    gain_bias = torch.zeros(H, gain_sides, dtype=dtype, requires_grad=True)
    v = torch.randn(B, L, H, topology.d_v, dtype=dtype)
    return x, route_W, gain_W, gain_bias, v


@pytest.mark.parametrize("normalization", ["global"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_producer_output_satisfies_oracle_contract_and_differentiates(normalization, dtype):
    """Item 4: run the Stage 1 oracle end-to-end on the producer's output, fp64
    and fp32, and check gradients reach every producer input (value stream,
    router weights, side gains)."""
    topology = _topology(mixed=True)
    B, L, H, dm = 2, 9, 2, 7
    sides = 1 if normalization == "global" else 2
    x, route_W, gain_W, gain_bias, v = _params(
        topology, B=B, L=L, H=H, dm=dm, dtype=dtype, gain_sides=sides)

    out = produce_routing(x, None, v, route_W, None, gain_W, gain_bias, topology, normalization)
    assert len(out.read_levels) == topology.D
    assert len(out.write_levels) == topology.D
    for level, width in enumerate(topology.widths):
        assert out.read_levels[level].shape == (B, L, H, width)
        assert out.write_levels[level].shape == (B, L, H, width)
        assert torch.allclose(
            out.read_levels[level].sum(dim=-1), torch.ones(B, L, H, dtype=dtype), atol=1e-4)
        assert torch.allclose(
            out.write_levels[level].sum(dim=-1), torch.ones(B, L, H, dtype=dtype), atol=1e-4)

    g_read = out.g_read
    if normalization == "global":
        assert g_read is None
    else:
        assert g_read.shape == (B, L, H)
        assert torch.all(g_read > 0)
    assert out.g_write.shape == (B, L, H)
    assert torch.all(out.g_write > 0)

    y, _ = naive_rola(v, out.read_levels, out.write_levels, g_read, out.g_write, topology, None, normalization)
    assert torch.isfinite(y).all()
    y.sum().backward()
    for name, tensor in [("x", x), ("route_W", route_W), ("gain_W", gain_W), ("gain_bias", gain_bias)]:
        assert tensor.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(tensor.grad).all(), f"{name} gradient is non-finite"


def test_softmax_level_mass_is_identically_one():
    """`IndependentRouting`/softmax levels are unit-sum by construction (spec §4.1);
    the producer's mass-fold (`_split_level_mass`) must be a no-op for them, so
    an all-softmax topology's g_write must equal its raw projected gain exactly
    (up to floating point), not scaled by any level mass."""
    levels = (LevelConfig(4, IndependentRouting(False, SoftmaxActivation(), SoftmaxActivation())),)
    topology = Topology(levels=levels, d_v=3)
    B, L, H, dm = 1, 4, 1, 5
    x, route_W, gain_W, gain_bias, v = _params(topology, B=B, L=L, H=H, dm=dm, dtype=torch.float64, gain_sides=1)
    out = produce_routing(x, None, v, route_W, None, gain_W, gain_bias, topology, "global")
    from rola.routing.side_gain import project_side_gains
    gains = project_side_gains(x, gain_W, gain_bias, normalization="global", projection_dtype=torch.float64)
    torch.testing.assert_close(out.g_write, gains["g_write"], rtol=1e-10, atol=1e-10)


def test_union_mass_fold_is_an_exact_identity_at_fp64():
    """The fold must MOVE the per-level mass, not approximate it (architecture §4.2:
    "multiplicative per-level masses collapse into one side-wide product").

    Required identity for every leaf `s`, on the write side -- the side that carries
    a gain under the shipped `global` arm:
        g_write_raw * prod_l level_l_raw[d_l(s)]  ==  g_write_out * prod_l p_l_out[d_l(s)]
    where `level_l_raw` is `routing_factor_levels`' own (possibly unnormalized, on
    UnionRouting's side) output and `p_l_out` is the producer's simplex.
    Checked against the machinery the producer wraps -- NOT against a second
    implementation of routing (the "matching-the-naive" standing rule).
    """
    from rola.routing.projection import _packed_router_logits
    from rola.routing.reference import routing_factor_levels
    from rola.routing.side_gain import project_side_gains

    topology = Topology(
        levels=(
            LevelConfig(4, UnionRouting(1.5)),
            LevelConfig(5, IndependentRouting(False, SoftmaxActivation(), SoftmaxActivation())),
        ),
        d_v=3,
    )
    B, L, H, dm = 2, 6, 3, 7
    x, route_W, gain_W, gain_bias, v = _params(
        topology, B=B, L=L, H=H, dm=dm, dtype=torch.float64, gain_sides=1)
    resolved = resolved_routing_from_topology(topology)

    out = produce_routing(x, None, v, route_W, None, gain_W, gain_bias, topology, "global")

    read_logits, write_logits = _packed_router_logits(
        x, route_W, resolved, route_bias=None, h_w=None, projection_dtype=torch.float64)
    reads_raw, writes_raw = routing_factor_levels(read_logits, write_logits, resolved)
    gains_raw = project_side_gains(
        x, gain_W, gain_bias, normalization="global", projection_dtype=torch.float64)

    def unfold(tensor):
        T, width = tensor.shape[1], tensor.shape[2]
        return tensor.reshape(B, H, T, width).permute(0, 2, 1, 3)

    # The union side must be genuinely unnormalized here, or the identity is vacuous.
    read_mass = unfold(reads_raw[0]).sum(-1)
    assert float((read_mass - 1).abs().max()) > 1e-3

    for side, raw_levels, raw_gain, out_levels, out_gain in (
        ("write", writes_raw, gains_raw["g_write"], out.write_levels, out.g_write),
    ):
        leaf_raw = raw_gain[..., None, None] * (
            unfold(raw_levels[0])[..., :, None] * unfold(raw_levels[1])[..., None, :])
        leaf_out = out_gain[..., None, None] * (
            out_levels[0][..., :, None] * out_levels[1][..., None, :])
        torch.testing.assert_close(
            leaf_out, leaf_raw, rtol=1e-13, atol=1e-13,
            msg=f"{side}-side mass fold is not an identity")
