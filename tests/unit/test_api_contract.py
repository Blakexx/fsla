# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CONTRACT: one op, one slot, and a bundle nobody but us can build.

The reduction theorem says the public surface is linear attention's contract plus ONE
new concept. That is a claim about a SHAPE, and a shape claim rots quietly -- a knob
creeps back onto a constructor, a plan type reappears at the top level, a struct users
were never meant to author acquires a friendly constructor -- with nothing failing.

This is what fails. Each row holds one property of the contract and is
FAIL-DEMONSTRABLE in the way this tree requires: plant the violation, observe the
refusal, restore. Where a row asserts that something is REFUSED, the refusal itself is
the observation.

Everything here runs on the CPU except the three rows marked ``cuda``: the surface is a
surface, and a row that needed a device would be testing the kernel. The three that do
are the ones about the SEAM -- the summary pass a plan is derived from is a CUDA kernel
with no CPU path by design, and executing a synthetic plan means executing.
"""
from __future__ import annotations

import pytest
import torch

import rola
from rola.planner.schedule import routing_views
from rola.routing.factors import RouteFactors
from rola.routing.producer import ProducedRoutes


def _routes(*, widths=(4, 4), hidden=32, heads=2, d_v=8, routing=None, **kwargs):
    producer = rola.SimplexProducer(
        widths, rola.dense_routing() if routing is None else routing,
        hidden_size=hidden, num_heads=heads, **kwargs)
    return rola.RouteProducer([producer], d_v=d_v, num_heads=heads, hidden_size=hidden)


def _x(gateway, *, B=2, T=6, dtype=torch.float64):
    return torch.randn(B, T, gateway.hidden_size, dtype=dtype)


# ---------------------------------------------------------------------------
# The bundle is OURS
# ---------------------------------------------------------------------------


def test_a_hand_built_routing_bundle_is_refused():
    """The gateway's monopoly on construction is the whole trustless argument.

    Every field of a bundle is a DERIVATION of the producers' tensors. If a caller could
    author one, they could pair statistics with tensors those statistics do not
    describe -- and the op would believe them, because believing them is the point of
    passing them.

    PLANTED VIOLATION: assemble a bundle by hand from perfectly good tensors.
    """
    gateway = _routes()
    factors = gateway(_x(gateway))
    with pytest.raises(TypeError, match="RouteProducer"):
        RouteFactors(
            topology=factors.topology, normalization=factors.normalization,
            read=factors.read, write=factors.write, g_read=factors.g_read,
            g_write=factors.g_write, density=factors.density)


def test_the_op_refuses_anything_that_is_not_a_bundle():
    """PLANTED VIOLATION: hand the op the tensors directly, as the old signature did."""
    gateway = _routes()
    factors = gateway(_x(gateway))
    v = torch.randn(2, 6, 2, 8, dtype=torch.float64)
    with pytest.raises(TypeError, match="RouteFactors"):
        rola.rola_op(factors.read, v)
    #: The well-formed call is the control: the refusal above must be about the TYPE,
    #: not about this fixture being unrunnable.
    y, _ = rola.rola_op(factors, v)
    assert y.shape == v.shape


def test_the_bundle_carries_v_shape_disagreement_to_the_caller_not_to_the_kernel():
    gateway = _routes()
    factors = gateway(_x(gateway))
    with pytest.raises(ValueError, match=r"v must be \[B, T, H, d_v\]"):
        rola.rola_op(factors, torch.randn(2, 6, 2, 9, dtype=torch.float64))


# ---------------------------------------------------------------------------
# The one slot
# ---------------------------------------------------------------------------


def test_a_foreign_producer_runs_with_no_registration_and_no_tier():
    """U02, mechanically: the consumer eats tensors and cannot tell what produced them.

    The duck below shares no base class, no tag and no metadata with the shipped
    producer -- it is a plain module returning simplices. It must produce a bundle the
    op accepts, and its levels must be OPAQUE: measured density rather than a
    substituted closed form, which is the only thing being foreign costs it.
    """
    class Uniform(torch.nn.Module):
        widths = (4,)

        def forward(self, source):
            B, T = source.batch, source.tokens
            level = torch.full((B, T, 2, 4), 0.25, dtype=source.read.dtype)
            return ProducedRoutes(read=(level,), write=(level.clone(),))

    gateway = rola.RouteProducer(
        [Uniform()], d_v=8, num_heads=2, hidden_size=32)
    factors = gateway(_x(gateway))
    assert factors.density.read == (False,) and factors.density.write == (False,), (
        "an opaque level must fall to the measured pass, not to a declared density")
    y, _ = rola.rola_op(factors, torch.randn(2, 6, 2, 8, dtype=torch.float64))
    assert torch.isfinite(y).all()


def test_a_producer_whose_output_is_malformed_is_named_with_its_level():
    """A merged or multi-level producer fails as one unit; the gateway holds the level
    map, so the gateway is what can say which level's data was involved.

    PLANTED VIOLATION: a duck that returns the wrong width.
    """
    class Wrong(torch.nn.Module):
        widths = (4, 4)

        def forward(self, source):
            B, T = source.batch, source.tokens
            good = torch.full((B, T, 2, 4), 0.25, dtype=source.read.dtype)
            bad = torch.full((B, T, 2, 3), 1 / 3, dtype=source.read.dtype)
            return ProducedRoutes(read=(good, bad), write=(good, good))

    gateway = rola.RouteProducer([Wrong()], d_v=8, num_heads=2, hidden_size=32)
    with pytest.raises(ValueError, match=r"level 1"):
        gateway(_x(gateway))


def test_a_producer_that_declares_no_widths_is_refused_at_construction():
    """Construction-time validation, so an unbuildable model fails on the line that
    built it rather than inside a forward several frames away."""
    with pytest.raises(TypeError, match="widths"):
        rola.RouteProducer([torch.nn.Identity()], d_v=8, num_heads=2, hidden_size=32)


def test_one_producer_covers_many_levels_and_the_gateway_keeps_their_order():
    """Batching across levels is a PRODUCER's business: one instance, one solve, and the
    level order is the order the widths were declared in."""
    gateway = _routes(widths=(4, 6))
    assert gateway.topology.widths == (4, 6)
    factors = gateway(_x(gateway))
    assert [tuple(level.shape[-1:]) for level in factors.read] == [(4,), (6,)]


# ---------------------------------------------------------------------------
# Statistics: placement, never numbers
# ---------------------------------------------------------------------------


def test_producer_supplied_statistics_equal_the_measured_ones_exactly():
    """The fused and separate arms are a PLACEMENT decision, so they must not be a
    numerical one. Bit-identical, not close: both are exact nonzero counts of the same
    stored values.
    """
    torch.manual_seed(0)
    plain = _routes(routing=rola.sparse_routing(1.5))
    torch.manual_seed(0)
    fused = _routes(routing=rola.sparse_routing(1.5), expose_stats=True)
    fused.producers[0].route_W.data.copy_(plain.producers[0].route_W.data)

    x = _x(plain)
    measured = plain(x).support_counts()
    supplied = fused(x).support_counts()
    assert set(measured) == set(supplied)
    for cell, counts in measured.items():
        assert torch.equal(counts, supplied[cell]), f"{cell} disagrees across the arms"


def test_the_statistics_are_derived_on_demand_and_not_before():
    """Nothing in the library reads them; a step that does not ask must not pay.

    Observed structurally rather than by timing: the bundle's derivation cache is empty
    until the accessor is called, and the accessor is what fills it.
    """
    gateway = _routes()
    factors = gateway(_x(gateway))
    assert "support_counts" not in factors.resident
    assert "touched_bits" not in factors.resident
    factors.support_counts()
    assert "support_counts" in factors.resident
    assert "touched_bits" not in factors.resident, "one accessor must not force the other"


# ---------------------------------------------------------------------------
# Packing belongs to the gateway
# ---------------------------------------------------------------------------


def test_the_public_layout_is_a_view_of_the_packed_one():
    """The solve emits the kernel's layout, so exactly one of the two can be
    materialized -- and it is the kernel's. The public tensors are permuted views, so
    the fold the planner takes is free.

    Asserted on STORAGE, which is the fact; a shape assertion would pass over a copy.
    """
    gateway = _routes(widths=(4, 6))
    factors = gateway(_x(gateway))
    views = routing_views(factors.read, factors.write, factors.topology)
    for level, (public, folded) in enumerate(zip(factors.read, views.read_folded)):
        assert not public.is_contiguous(), f"level {level}'s public tensor was materialized"
        assert folded.is_contiguous()
        assert folded.data_ptr() == public.data_ptr(), (
            f"level {level}'s fold copied; the inverse-copy pair is back")


def test_the_stored_levels_are_simplices_and_the_mass_went_to_the_gain():
    """The mass fold is the gateway's, and it is loss-free for the readout: the stored
    level sums to one and the magnitude it lost is on the side gain."""
    class Doubled(torch.nn.Module):
        widths = (4,)

        def forward(self, source):
            B, T = source.batch, source.tokens
            level = torch.full((B, T, 2, 4), 0.5, dtype=source.read.dtype)  # sums to 2
            return ProducedRoutes(read=(level,), write=(level.clone(),))

    gateway = rola.RouteProducer([Doubled()], d_v=8, num_heads=2, hidden_size=32,
                                 gain=False)
    factors = gateway(_x(gateway))
    ones = torch.ones(2, 6, 2, dtype=torch.float64)
    assert torch.allclose(factors.write[0].sum(-1), ones)
    assert torch.allclose(factors.g_write, 2 * ones), (
        "the write mass must ride the write channel, not vanish")


# ---------------------------------------------------------------------------
# The decay-source slot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scope", ["state", "global"])
def test_every_learned_decay_scope_emits_the_same_dial_shape(scope):
    """The scopes differ in degrees of freedom, never in what the recurrence reads."""
    source = rola.LearnedDecay(2 ** -8, widths=(4, 6), num_heads=2, scope=scope)
    dials = source().dials
    assert [tuple(d.shape) for d in dials] == [(2, 4), (2, 6)]
    assert all(((d >= 0) & (d < 1)).all() for d in dials)


def test_a_constant_decay_source_has_no_parameters_and_reaches_the_target_rate():
    source = rola.ConstantDecay(2 ** -8, widths=(4, 4), num_heads=2)
    assert list(source.parameters()) == [], "a user-fixed rate must not be optimized"
    product = source().dials[0][0, 0] * source().dials[1][0, 0]
    assert torch.isclose(product, torch.tensor(2.0 ** -8))


def test_a_decay_source_sized_for_another_topology_is_refused_at_construction():
    """PLANTED VIOLATION: the right source, the wrong widths."""
    gateway = _routes(widths=(4, 4))
    with pytest.raises(ValueError, match="widths"):
        rola.RoLA(32, 2, gateway,
                  decay=rola.LearnedDecay(widths=(4, 6), num_heads=2))
    rola.RoLA(32, 2, gateway, decay=rola.LearnedDecay(widths=(4, 4), num_heads=2))


# ---------------------------------------------------------------------------
# The quarantine
# ---------------------------------------------------------------------------


def test_no_launch_geometry_is_reachable_from_the_top_level():
    """The plan family is HOW the op runs, not what the model is. It was exported at
    the top level, one shelf below the model types, and the tier label on it was a
    warning sign at eye level. It now lives behind a marked door.
    """
    quarantined = {"Plan", "Schedule", "StaticDensity",
                   "plan_and_schedule",
                   "PlanOverrides", "derive_plan", "execute", "requires_backward",
                   "dispatch_for"}
    import rola.expert as expert

    assert quarantined.isdisjoint(rola.__all__)
    assert quarantined.issubset(set(expert.__all__))


def test_the_layer_takes_the_slot_and_not_the_slot_spelled_out():
    """The constructor's shape IS the claim. A knob creeping back is what this catches."""
    import inspect

    signature = inspect.signature(rola.RoLA.__init__)
    assert list(signature.parameters) == [
        "self", "hidden_size", "num_heads", "routes",
        "decay", "value_conv", "route_conv", "layer_idx", "expert"]
    for name in ("decay", "value_conv", "route_conv", "layer_idx", "expert"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_an_expert_pin_shrinks_the_feasible_set_and_an_inadmissible_one_is_refused():
    from rola.expert import PlanOverrides

    producer = rola.SimplexProducer((4, 256), rola.dense_routing(),
                                    hidden_size=64, num_heads=2)
    routes = rola.RouteProducer([producer], d_v=32, num_heads=2, hidden_size=64)
    assert rola.RoLA(64, 2, routes).feasible_BC == (16, 64)
    pinned = rola.RoLA(64, 2, routes, expert=PlanOverrides(state_block=16))
    assert pinned.feasible_BC == (16,)
    with pytest.raises(ValueError):
        rola.RoLA(64, 2, routes, expert=PlanOverrides(state_block=24))


def test_the_expert_argument_refuses_a_lookalike():
    """PLANTED VIOLATION: a dict that happens to have the right keys."""
    gateway = _routes()
    with pytest.raises(TypeError, match="PlanOverrides"):
        rola.RoLA(32, 2, gateway, expert={"state_block": 16})


# ---------------------------------------------------------------------------
# The layer's one condition, and the arms
# ---------------------------------------------------------------------------


def test_training_mode_alone_selects_the_differentiable_arm():
    """The condition the op cannot own: reentrant checkpointing runs its first pass
    under `no_grad()` with the module still training, so a predicate keyed only on
    gradient reachability would take the kernel there while backward recomputes the same
    step on the reference -- two implementations agreeing only to a tolerance.
    """
    layer = rola.RoLA(32, 2, _routes()).double().train()
    with torch.no_grad():
        layer(torch.randn(2, 6, 32, dtype=torch.float64))
    assert layer.execution_backend_counts["oracle"] == 1
    assert layer.last_execution_backend == "oracle"


def test_an_unsupported_forward_keyword_raises_rather_than_being_swallowed():
    """PLANTED VIOLATION: the HF-compat keyword the layer used to accept and ignore."""
    layer = rola.RoLA(32, 2, _routes()).double().eval()
    x = torch.randn(2, 6, 32, dtype=torch.float64)
    with pytest.raises(TypeError, match="output_attentions"):
        layer(x, output_attentions=True)
    with pytest.raises(TypeError, match="cu_seqlens"):
        layer(x, cu_seqlens=torch.tensor([0, 6]))


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------


@pytest.mark.cuda
def test_a_plan_carries_its_model_facts_from_tier_zero_and_not_from_a_second_path():
    """The `decay` flag used to be computed twice, by two different expressions, on two
    different paths out of the layer. One path now: the plan reads its model fields from
    the tier-0 record the planner was given.
    """
    from rola.expert import derive_plan

    gateway = _routes(widths=(16, 16), d_v=64).cuda()
    factors = gateway(_x(gateway, dtype=torch.float32).cuda())
    source = rola.LearnedDecay(widths=gateway.widths, num_heads=2)

    assert derive_plan(factors, device=factors.device).decay is False
    #: decay retired from the operator: no arm exists for decay geometry, so
    #: the planner cannot derive a plan for it -- the refusal IS the contract
    #: (docs/internals/chunk/chunk.md); the flag's one-path property is
    #: covered by the False case above.
    with pytest.raises(NotImplementedError, match="decay"):
        derive_plan(factors, decay=source(), device=factors.device)
    assert derive_plan(factors, device=factors.device).topology is factors.topology


@pytest.mark.cuda
def test_execute_obeys_a_synthetic_plan_that_no_planner_would_have_chosen():
    """The seam's whole point: a plan is DATA, so a plan nobody derived is runnable.

    This is the handle the conformance families need. A pipeline that derived its plan
    and immediately used it could only ever be tested at the plan its own cost model
    picked -- which is exactly the plan least likely to expose a boundary bug. Here the
    tiling is chosen to disagree with the planner's, and the result must still be the
    one the definition gives.
    """
    from rola.expert import Plan, derive_plan, execute

    torch.manual_seed(0)
    gateway = _routes(widths=(16, 16), d_v=64).cuda()
    x = _x(gateway, B=1, T=64, dtype=torch.float32).cuda()
    factors = gateway(x)
    v = torch.randn(1, 64, 2, 64, device="cuda")

    chosen = derive_plan(factors, device=v.device)
    other_BC = 64 if chosen.BC == 16 else 16
    synthetic = Plan(topology=factors.topology, normalization=factors.normalization,
                     BC=other_BC, warp_tile=chosen.warp_tile, decay=False)
    with torch.no_grad():
        y_synthetic, _state, _schedule = execute(synthetic, factors, v)
        reference, _ = rola.naive_rola(
            v.double(), tuple(t.double() for t in factors.read),
            tuple(t.double() for t in factors.write), None,
            factors.g_write.double(), factors.topology, None, factors.normalization)
    # bf16 is the only storage format now (no tf32x3 arm), so the tolerance is the
    # derived bf16 family bound rather than the old near-fp32 rtol.
    assert torch.allclose(y_synthetic.double(), reference, rtol=1e-2, atol=1e-2), (
        "a plan the planner did not pick must still compute the definition")
