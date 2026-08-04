"""V3 Stage 2 producer tests (`ROLA_V3_SPEC.md` §10 Stage 2; architecture §4.5).

Validates `rola.routing.producer` against the ONE canonical
ground truth this branch authorizes: the Stage 1 fp64 oracle
(`rola.ops.naive.naive_rola`). Per the "matching-the-naive
anti-pattern" standing rule, nothing here is checked against a second,
freshly written reference -- every assertion either (a) feeds the producer's
output through the oracle and checks numerical properties the oracle itself
guarantees (finite output, differentiability), or (b) checks an algebraic
identity of the producer's own contract (mass-fold correctness, simplex
normalization) that is checkable without an independent implementation.
"""

from __future__ import annotations

import pytest
import torch

from rola.ops.naive import naive_rola
from rola.routing.budget import chunk_boundaries
from rola.routing.producer import (
    BudgetConditioning,
    _project_logits,
    embed_budget,
    produce_routing,
    resolved_routing_from_topology,
)
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


@pytest.mark.parametrize("normalization", ["raw", "global"])
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


def test_union_level_mass_is_folded_not_dropped():
    """UnionRouting's read side is not individually unit-sum (reference.py's own
    `_union_split_with_support`); the producer must still return a unit-sum
    read level (spec §4.1) by folding the leftover mass into g_read, not by
    silently renormalizing away information. Check the identity
    `mass * normalized == raw_union_read` end to end via the oracle's raw-mode
    output being sensitive to that mass (a design smoke check: multiplying the
    write-side projection weights should NOT commute with dropping the mass)."""
    levels = (LevelConfig(6, UnionRouting(1.5)),)
    topology = Topology(levels=levels, d_v=3)
    B, L, H, dm = 1, 4, 1, 5
    x, route_W, gain_W, gain_bias, v = _params(topology, B=B, L=L, H=H, dm=dm, dtype=torch.float64, gain_sides=2)
    out = produce_routing(x, None, v, route_W, None, gain_W, gain_bias, topology, "raw")
    assert torch.allclose(out.read_levels[0].sum(-1), torch.ones(B, L, H, dtype=torch.float64), atol=1e-8)
    assert torch.allclose(out.write_levels[0].sum(-1), torch.ones(B, L, H, dtype=torch.float64), atol=1e-8)
    # g_read must not be a pure function of gain_W alone (it must carry the folded read mass) --
    # i.e. it must differ from the raw projected gain whenever the union split's read mass != 1.
    from rola.routing.side_gain import project_side_gains
    raw_gains = project_side_gains(x, gain_W, gain_bias, normalization="raw", projection_dtype=torch.float64)
    assert not torch.allclose(out.g_read, raw_gains["g_read"])


def test_union_mass_fold_is_an_exact_identity_at_fp64():
    """The fold must MOVE the per-level mass, not approximate it (architecture §4.2:
    "multiplicative per-level masses collapse into one side-wide product").

    Required identity, per side, for every leaf `s`:
        g_side_raw * prod_l level_l_raw[d_l(s)]  ==  g_side_out * prod_l p_l_out[d_l(s)]
    where `level_l_raw` is `routing_factor_levels`' own (possibly unnormalized, on
    UnionRouting's read side) output and `p_l_out` is the producer's simplex.
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
        topology, B=B, L=L, H=H, dm=dm, dtype=torch.float64, gain_sides=2)
    resolved = resolved_routing_from_topology(topology)

    out = produce_routing(x, None, v, route_W, None, gain_W, gain_bias, topology, "raw")

    read_logits, write_logits = _packed_router_logits(
        x, route_W, resolved, route_bias=None, h_w=None, projection_dtype=torch.float64)
    reads_raw, writes_raw = routing_factor_levels(read_logits, write_logits, resolved)
    gains_raw = project_side_gains(
        x, gain_W, gain_bias, normalization="raw", projection_dtype=torch.float64)

    def unfold(tensor):
        T, width = tensor.shape[1], tensor.shape[2]
        return tensor.reshape(B, H, T, width).permute(0, 2, 1, 3)

    # The union read side must be genuinely unnormalized here, or the identity is vacuous.
    read_mass = unfold(reads_raw[0]).sum(-1)
    assert float((read_mass - 1).abs().max()) > 1e-3

    for side, raw_levels, raw_gain, out_levels, out_gain in (
        ("read", reads_raw, gains_raw["g_read"], out.read_levels, out.g_read),
        ("write", writes_raw, gains_raw["g_write"], out.write_levels, out.g_write),
    ):
        leaf_raw = raw_gain[..., None, None] * (
            unfold(raw_levels[0])[..., :, None] * unfold(raw_levels[1])[..., None, :])
        leaf_out = out_gain[..., None, None] * (
            out_levels[0][..., :, None] * out_levels[1][..., None, :])
        torch.testing.assert_close(
            leaf_out, leaf_raw, rtol=1e-13, atol=1e-13,
            msg=f"{side}-side mass fold is not an identity")


@pytest.mark.parametrize("normalization", ["raw", "global"])
def test_producer_exact_support_counts_match_a_manual_nonzero_count(normalization):
    """Tier 1's exact measurement must be exact (architecture §4.5: the nonzero count
    per token, "O(L*b_l) and free") -- bit-identical to a manual count, on every level
    and side, including the zero-capable union level."""
    topology = Topology(
        levels=(
            LevelConfig(4, UnionRouting(1.5)),
            LevelConfig(5, IndependentRouting(False, SoftmaxActivation(), SoftmaxActivation())),
        ),
        d_v=3,
    )
    B, L, H, dm = 2, 6, 3, 7
    sides = 1 if normalization == "global" else 2
    x, route_W, gain_W, gain_bias, v = _params(
        topology, B=B, L=L, H=H, dm=dm, dtype=torch.float64, gain_sides=sides)
    out = produce_routing(x, None, v, route_W, None, gain_W, gain_bias, topology, normalization)

    saw_sparse = False
    for level, width in enumerate(topology.widths):
        for side, levels in (("read", out.read_levels), ("write", out.write_levels)):
            manual = (levels[level] != 0).sum(dim=-1).to(levels[level].dtype)
            assert torch.equal(out.exact_support_counts[(level, side)], manual), (
                f"exact_support_counts[({level},{side!r})] is not the exact nonzero count")
            assert float(manual.max()) <= width
            saw_sparse |= bool((manual < width).any())
    # a genuinely sparse (zero-capable) level must be present, or the test is vacuous
    assert saw_sparse


def test_embed_budget_off_is_identity():
    x = torch.randn(2, 3, 5)
    embed_W = torch.randn(5)
    assert torch.equal(embed_budget(x, None, embed_W), x)


def test_embed_budget_adds_per_sequence_constant():
    x = torch.zeros(2, 3, 4)
    embed_W = torch.ones(4)
    r = torch.tensor([0.5, 1.0])
    out = embed_budget(x, r, embed_W)
    assert torch.allclose(out[0], torch.full((3, 4), 0.5))
    assert torch.allclose(out[1], torch.full((3, 4), 1.0))


def test_fully_inert_budget_conditioning_warns_loudly():
    """D8 regression: a `BudgetConditioning` with no budgeted pairs and
    `usage_bias_enabled=False` does nothing (`produce_routing` always takes the
    unconditioned baseline path for it) and must not be accepted silently -- unlike the
    softmax-budget case (a loud `ValueError`), this is a loud warning, not an error,
    since an inert conditioning object is not itself invalid, just useless."""
    with pytest.warns(UserWarning, match="inert"):
        BudgetConditioning()


def test_budget_conditioning_rejects_softmax_level():
    """Spec §4.5: "a budget on a softmax side is a loud config validation error, not a no-op"."""
    topology = Topology(
        levels=(LevelConfig(4, IndependentRouting(False, SoftmaxActivation(), SoftmaxActivation())),), d_v=2)
    budget = BudgetConditioning(budgeted_write_levels=(0,))
    with pytest.raises(ValueError, match="zero-capable"):
        budget.validate_against(topology)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_conditioned_path_matches_oracle_contract_and_differentiates_all_actuators(dtype):
    """Item 4: the FULL §4.5 conditioning surface (temperature actuator, budget
    embedding, usage-bias head g) wired end to end, checked against the same
    oracle contract, with gradients required to reach every actuator parameter."""
    topology = _topology(mixed=True)
    B, L, H, dm = 2, 17, 2, 6
    x, route_W, gain_W, gain_bias, v = _params(topology, B=B, L=L, H=H, dm=dm, dtype=dtype, gain_sides=1)
    budget = BudgetConditioning(
        budgeted_write_levels=(1,), budgeted_read_levels=(1,), usage_bias_enabled=True, chunk_size=5)
    n_pairs = len(budget.budgeted_pairs)
    temp_W = (torch.randn(H, n_pairs, 2, dtype=dtype) * 0.1).requires_grad_()
    temp_b = torch.zeros(H, n_pairs, dtype=dtype, requires_grad=True)
    usage_bias_W = (torch.randn(H, topology.D, 4, dtype=dtype) * 0.1).requires_grad_()
    usage_bias_b = torch.zeros(H, topology.D, dtype=dtype, requires_grad=True)
    budget_embed_W = (torch.randn(dm, dtype=dtype) * 0.1).requires_grad_()
    r = torch.full((B,), 0.5, dtype=dtype)

    out = produce_routing(
        x, None, v, route_W, None, gain_W, gain_bias, topology, "global",
        budget=budget, r=r, budget_embed_W=budget_embed_W,
        temp_W=temp_W, temp_b=temp_b, usage_bias_W=usage_bias_W, usage_bias_b=usage_bias_b,
    )
    for level, width in enumerate(topology.widths):
        assert torch.allclose(
            out.read_levels[level].sum(-1), torch.ones(B, L, H, dtype=dtype), atol=1e-4)
    y, _ = naive_rola(v, out.read_levels, out.write_levels, None, out.g_write, topology, None, "global")
    y.sum().backward()
    for name, tensor in [
        ("x", x), ("route_W", route_W), ("gain_W", gain_W),
        ("temp_W", temp_W), ("temp_b", temp_b),
        ("usage_bias_W", usage_bias_W), ("usage_bias_b", usage_bias_b),
        ("budget_embed_W", budget_embed_W),
    ]:
        assert tensor.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(tensor.grad).all(), f"{name} gradient is non-finite"

    assert set(out.slack_history.keys()) == set(budget.budgeted_pairs)
    for pair, hist in out.slack_history.items():
        assert hist.shape[:2] == (B, H)
    assert set(out.touched_bits.keys()) == set(range(topology.D))
    for level, width in enumerate(topology.widths):
        assert out.touched_bits[level].shape == (B, H, width)
    for level in range(topology.D):
        assert out.exact_support_counts[(level, "read")].shape == (B, L, H)
        assert out.exact_support_counts[(level, "write")].shape == (B, L, H)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_project_logits_is_chunk_invariant(dtype):
    """V6 regression: `_conditioned_pass` hoists `_project_logits` out of the
    64-step chunk loop on the grounds that it is a pure per-token function of
    `x`/`route_W`/`route_bias` -- it reads no conditioning state, unlike the
    temperature/usage-bias actuators that genuinely depend on the previous
    chunk's output. This pins that claim directly and bitwise: projecting once
    over the full sequence and slicing per chunk must equal projecting
    per-chunk on the same slices, with no tolerance, for every chunk boundary
    the conditioned path actually uses."""
    topology = _topology(mixed=True)
    resolved = resolved_routing_from_topology(topology)
    B, L, H, dm, chunk_size = 2, 17, 2, 6, 5
    torch.manual_seed(0)
    x = torch.randn(B, L, dm, dtype=dtype)
    route_W = torch.randn(H, dm, resolved.packed_router_width, dtype=dtype)
    route_bias = torch.randn(H, resolved.packed_router_width, dtype=dtype)

    read_full, write_full = _project_logits(x, route_W, resolved, route_bias=route_bias, h_w=None)

    for start, end in chunk_boundaries(L, chunk_size):
        read_chunk, write_chunk = _project_logits(
            x[:, start:end], route_W, resolved, route_bias=route_bias, h_w=None)
        assert torch.equal(read_chunk, read_full[:, start:end]), f"read logits diverge at [{start}:{end}]"
        assert torch.equal(write_chunk, write_full[:, start:end]), f"write logits diverge at [{start}:{end}]"


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_conditioned_path_output_unchanged_by_logit_projection_hoist(dtype):
    """V6 regression, end to end: `produce_routing`'s conditioned path (temperature
    actuator + usage-bias head, both live) reduces to a call sequence that composes
    `_project_logits` once over the full sequence with the same per-chunk conditioning
    arithmetic that reads its slices. This exercises that composed path (not just the
    projection in isolation) against the oracle contract with a chunk size that does not
    evenly divide `L`, so the slicing at both chunk-loop boundaries and the final ragged
    chunk is covered."""
    topology = _topology(mixed=True)
    B, L, H, dm = 2, 17, 2, 6
    x, route_W, gain_W, gain_bias, v = _params(topology, B=B, L=L, H=H, dm=dm, dtype=dtype, gain_sides=1)
    budget = BudgetConditioning(
        budgeted_write_levels=(1,), budgeted_read_levels=(1,), usage_bias_enabled=True, chunk_size=5)
    n_pairs = len(budget.budgeted_pairs)
    torch.manual_seed(2)
    temp_W = (torch.randn(H, n_pairs, 2, dtype=dtype) * 0.1)
    temp_b = torch.zeros(H, n_pairs, dtype=dtype)
    usage_bias_W = (torch.randn(H, topology.D, 4, dtype=dtype) * 0.1)
    usage_bias_b = torch.zeros(H, topology.D, dtype=dtype)
    budget_embed_W = (torch.randn(dm, dtype=dtype) * 0.1)
    r = torch.full((B,), 0.5, dtype=dtype)

    with torch.no_grad():
        out = produce_routing(
            x, None, v, route_W, None, gain_W, gain_bias, topology, "global",
            budget=budget, r=r, budget_embed_W=budget_embed_W,
            temp_W=temp_W, temp_b=temp_b, usage_bias_W=usage_bias_W, usage_bias_b=usage_bias_b,
        )
    # The oracle contract itself must still hold post-hoist (a divergence in the projection
    # would generically break simplex normalization downstream).
    for level, width in enumerate(topology.widths):
        assert torch.allclose(
            out.read_levels[level].sum(-1), torch.ones(B, L, H, dtype=dtype), atol=1e-4)


def test_usage_bias_fires_even_when_no_tier1_budget_is_supplied():
    """D3 regression: usage conditioning (`g`) is a separate mechanism from Tier 1 with its
    own ablation gate (architecture §4.5) and must not be silently disabled just because
    `r` is None (the "sandwich sampling includes budget=off" branch). Before the fix,
    `active_conditioning` required `r is not None`, so a `BudgetConditioning` with
    `usage_bias_enabled=True` and no budgeted pairs produced output bit-identical to the
    fully unconditioned path whenever `r=None` -- this test pins the opposite: a nonzero
    `usage_bias_W`/`usage_bias_b` must perturb the routing output relative to the
    unconditioned baseline even with `r=None`."""
    topology = _topology(mixed=False)
    B, L, H, dm = 2, 9, 2, 6
    x, route_W, gain_W, gain_bias, v = _params(topology, B=B, L=L, H=H, dm=dm, dtype=torch.float64, gain_sides=1)
    budget = BudgetConditioning(usage_bias_enabled=True, chunk_size=4)

    torch.manual_seed(1)
    usage_bias_W = torch.randn(H, topology.D, 4, dtype=torch.float64) * 0.5
    usage_bias_b = torch.randn(H, topology.D, dtype=torch.float64) * 0.5

    conditioned = produce_routing(
        x, None, v, route_W, None, gain_W, gain_bias, topology, "global",
        budget=budget, r=None, budget_embed_W=None,
        temp_W=None, temp_b=None, usage_bias_W=usage_bias_W, usage_bias_b=usage_bias_b,
    )
    unconditioned = produce_routing(
        x, None, v, route_W, None, gain_W, gain_bias, topology, "global",
        budget=None, r=None,
    )

    differs = False
    for level in range(topology.D):
        if not torch.allclose(conditioned.write_levels[level], unconditioned.write_levels[level]):
            differs = True
    assert differs, "usage-bias head g had no effect on the write routing when r=None"


@pytest.mark.parametrize("perturbation", [1e-12, 1e-7])
def test_conditioned_r_input_preserves_fp64_precision(perturbation):
    """D4 regression: `r` must not be hard-cast to fp32 inside `_conditioned_pass` (the
    same class of defect the builder already fixed for the packed-router logit
    projection). Before the fix, `r_bh = r.to(..., dtype=torch.float32)` quantized the
    actuator input regardless of caller dtype, so an fp64 `r` perturbed by 1e-12 (below
    fp32 ULP at this magnitude) produced a bit-identical conditioned write allocation. At
    fp64 throughout, even a 1e-12 perturbation must propagate to a nonzero (if tiny)
    change; a 1e-7 perturbation must change the output by more than fp32 noise (~1e-6)."""
    topology = _topology(mixed=False)
    B, L, H, dm = 1, 6, 1, 4
    x, route_W, gain_W, gain_bias, v = _params(topology, B=B, L=L, H=H, dm=dm, dtype=torch.float64, gain_sides=1)
    budget = BudgetConditioning(budgeted_write_levels=(1,), chunk_size=3)
    n_pairs = len(budget.budgeted_pairs)

    torch.manual_seed(2)
    temp_W = torch.randn(H, n_pairs, 2, dtype=torch.float64) * 0.5
    temp_b = torch.zeros(H, n_pairs, dtype=torch.float64)
    r0 = torch.tensor([0.5], dtype=torch.float64)
    r1 = r0 + perturbation

    def _write(r):
        out = produce_routing(
            x, None, v, route_W, None, gain_W, gain_bias, topology, "global",
            budget=budget, r=r, budget_embed_W=None,
            temp_W=temp_W, temp_b=temp_b, usage_bias_W=None, usage_bias_b=None,
        )
        return out.write_levels[1]

    w0 = _write(r0)
    w1 = _write(r1)
    diff = (w1 - w0).abs().max().item()
    assert diff != 0.0, f"perturbing fp64 r by {perturbation} produced no change (fp32 quantization)"
