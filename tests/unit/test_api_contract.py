# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CONTRACT: one op, one slot, and a bundle that is tensors all the way down.

The reduction theorem says the public surface is linear attention's contract plus ONE
new concept. That is a claim about a SHAPE, and a shape claim rots quietly -- a knob
creeps back onto a constructor, a plan type reappears at the top level, a declaration
users could disagree with reappears on a record of tensors -- with nothing failing.

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
from rola.routing.factors import RouteFactors
from rola.routing.producer import ProducedRoutes


def _routes(*, widths=(4, 4), hidden=32, heads=2, routing=None, **kwargs):
    producer = rola.SimplexProducer(
        widths, rola.dense_routing() if routing is None else routing,
        hidden_size=hidden, num_heads=heads, **kwargs)
    return rola.RouteProducer([producer], num_heads=heads, hidden_size=hidden)


def _x(gateway, *, B=2, T=6, dtype=torch.float64):
    return torch.randn(B, T, gateway.hidden_size, dtype=dtype)


# ---------------------------------------------------------------------------
# The bundle is a record of tensors, and its invariants are structural
# ---------------------------------------------------------------------------


def test_a_well_formed_hand_built_bundle_is_legal():
    """The bundle carries no DECLARATION, so there is nothing to protect from a caller.

    It once carried statistics and a declared density, which a hand-built bundle could
    have paired with tensors they did not describe -- and the op would have believed
    them, because believing them was the point of passing them. Those are gone: what is
    left is the tensors and the topology they are shaped by, so authoring one is a
    legitimate thing to do and a fixture that plants a specific routing pattern does
    exactly that.
    """
    gateway = _routes()
    factors = gateway(_x(gateway))
    hand_built = RouteFactors(
        topology=factors.topology, normalization=factors.normalization,
        read=factors.read, write=factors.write, g_read=factors.g_read,
        g_write=factors.g_write)
    y, _ = rola.rola_op(hand_built, torch.randn(2, 6, 2, 8, dtype=torch.float64))
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("break_it,message", [
    ("width", "wide but the topology"),
    ("levels", "one tensor per routing level"),
    ("prefix", r"one \[B, T, H\] prefix"),
    ("dtype", "one dtype"),
    ("gain_shape", r"g_write must be a \[B, T, H\]"),
    ("read_gain", "g_read is None under"),
])
def test_a_malformed_hand_built_bundle_is_refused_by_its_structure(break_it, message):
    """Unsealing removed the ORIGIN check, not the invariants. Each row plants one
    structural violation the derivation code could never have produced, and the
    refusal is the observation.

    Deliberately NOT among them: unit-sum-ness. It is a device reduction per level per
    call, it is the gateway's own output contract, and conformance fixtures plant
    non-canonical amplitudes on purpose.
    """
    gateway = _routes()
    f = gateway(_x(gateway))
    fields = dict(topology=f.topology, normalization=f.normalization,
                  read=list(f.read), write=list(f.write), g_read=f.g_read,
                  g_write=f.g_write)
    if break_it == "width":
        fields["read"][0] = fields["read"][0][..., :2]
    elif break_it == "levels":
        fields["read"] = fields["read"][:1]
    elif break_it == "prefix":
        fields["write"][1] = fields["write"][1][:1]
    elif break_it == "dtype":
        fields["write"][0] = fields["write"][0].to(torch.float32)
    elif break_it == "gain_shape":
        fields["g_write"] = fields["g_write"][..., :1]
    elif break_it == "read_gain":
        fields["g_read"] = fields["g_write"].clone()
    with pytest.raises((ValueError, TypeError), match=message):
        RouteFactors(**{k: (tuple(v) if isinstance(v, list) else v)
                        for k, v in fields.items()})


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
    """The TOKEN ADDRESSING must agree; the value width is not a disagreement at all.

    `d_v` is `v`'s own width -- the routing carries no value geometry -- so a
    different one is a different model, not a mismatch, and the op reads it. What
    the bundle does pin is `[B, T, H]`, and a `v` that disagrees there is refused
    with the expected prefix in the message rather than handed to a kernel.
    """
    gateway = _routes()
    factors = gateway(_x(gateway))
    with pytest.raises(ValueError, match=r"\[B, T, H\]=\(2, 6, 2\)"):
        rola.rola_op(factors, torch.randn(2, 5, 2, 8, dtype=torch.float64))
    #: the control: the same call at a width the bundle never declared runs fine.
    y, _ = rola.rola_op(factors, torch.randn(2, 6, 2, 9, dtype=torch.float64))
    assert y.shape == (2, 6, 2, 9)


# ---------------------------------------------------------------------------
# The one slot
# ---------------------------------------------------------------------------


def test_a_foreign_producer_runs_with_no_registration_and_no_tier():
    """U02, mechanically: the consumer eats tensors and cannot tell what produced them.

    The duck below shares no base class, no tag and no metadata with the shipped
    producer -- it is a plain module returning simplices. It must produce a bundle the
    op accepts, and its levels must be OPAQUE: the topology reads the opaque registry
    row for them, which is the only thing being foreign costs it.
    """
    class Uniform(torch.nn.Module):
        widths = (4,)

        def forward(self, source):
            B, T = source.batch, source.tokens
            level = torch.full((B, T, 2, 4), 0.25, dtype=source.read.dtype)
            return ProducedRoutes(read=(level,), write=(level.clone(),))

    gateway = rola.RouteProducer(
        [Uniform()], num_heads=2, hidden_size=32)
    factors = gateway(_x(gateway))
    assert type(gateway.topology.levels[0].routing).__name__ == "OpaqueRouting", (
        "a producer that names no activation must take the opaque registry row")
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

    gateway = rola.RouteProducer([Wrong()], num_heads=2, hidden_size=32)
    with pytest.raises(ValueError, match=r"level 1"):
        gateway(_x(gateway))


def test_a_producer_that_declares_no_widths_is_refused_at_construction():
    """Construction-time validation, so an unbuildable model fails on the line that
    built it rather than inside a forward several frames away."""
    with pytest.raises(TypeError, match="widths"):
        rola.RouteProducer([torch.nn.Identity()], num_heads=2, hidden_size=32)


def test_one_producer_covers_many_levels_and_the_gateway_keeps_their_order():
    """Batching across levels is a PRODUCER's business: one instance, one solve, and the
    level order is the order the widths were declared in."""
    gateway = _routes(widths=(4, 6))
    assert gateway.topology.widths == (4, 6)
    factors = gateway(_x(gateway))
    assert [tuple(level.shape[-1:]) for level in factors.read] == [(4,), (6,)]


# ---------------------------------------------------------------------------
# Packing belongs to the gateway
# ---------------------------------------------------------------------------


def test_the_public_layout_is_a_view_and_not_a_materialization():
    """The solve emits ONE layout and the public tensors are permuted views of it,
    so nothing on this surface costs a copy.

    Asserted on STORAGE, which is the fact; a shape assertion would pass over a copy.

    P67 D2-b NARROWED this row, per ruling R-6. It used to also assert WHICH of the
    two layouts is the materialized one, by comparing `data_ptr()` against
    `routing_views`' folded view -- and `routing_views` is the TILED consumer's
    packing, deleted with it. The chunk arm builds its own packed operands per call
    (`rola/ops/chunk.py` says so, and says they are scheduled for deletion into the
    producer epilogue), so there is no second standing layout left to be the one
    that is materialized. What survives, and is what a caller of this surface is
    owed, is that the public tensors are views.
    """
    gateway = _routes(widths=(4, 6))
    factors = gateway(_x(gateway))
    for level, public in enumerate(factors.read):
        assert not public.is_contiguous(), f"level {level}'s public tensor was materialized"
        assert public._base is not None, (
            f"level {level}'s public tensor owns its storage; the solve's layout was "
            "copied out instead of being viewed")


def test_the_stored_levels_are_simplices_and_the_mass_went_to_the_gain():
    """The mass fold is the gateway's, and it is loss-free for the readout: the stored
    level sums to one and the magnitude it lost is on the side gain."""
    class Doubled(torch.nn.Module):
        widths = (4,)

        def forward(self, source):
            B, T = source.batch, source.tokens
            level = torch.full((B, T, 2, 4), 0.5, dtype=source.read.dtype)  # sums to 2
            return ProducedRoutes(read=(level,), write=(level.clone(),))

    gateway = rola.RouteProducer([Doubled()], num_heads=2, hidden_size=32,
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
    #: The set names what is BEHIND the door today; the RULE is what this row states
    #: and it does not move when the set does: launch geometry is behind the marked
    #: door, never at the top level.
    quarantined = {"PlanOverrides", "requires_backward", "dispatch_for"}
    import rola.expert as expert

    assert quarantined.isdisjoint(rola.__all__)
    assert quarantined.issubset(set(expert.__all__))


def test_the_layer_takes_the_slot_and_not_the_slot_spelled_out():
    """The constructor's shape IS the claim. A knob creeping back is what this catches."""
    import inspect

    signature = inspect.signature(rola.RoLA.__init__)
    assert list(signature.parameters) == [
        "self", "hidden_size", "num_heads", "routes",
        "d_v", "decay", "value_conv", "route_conv", "layer_idx", "expert"]
    for name in ("d_v", "decay", "value_conv", "route_conv", "layer_idx", "expert"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


#: RETIRED at P67 D2-b (ruling R-2): `test_an_expert_pin_shrinks_the_feasible_set_
#: and_an_inadmissible_one_is_refused`. Its subject is `RoLA.feasible_BC` and the
#: tiled ladder `derive.SUPPORTED_BC` behind it, both deleted in D2-c: the chunk
#: arm's `BC` belongs to the built arm the manifest carries for a topology and is
#: not host-selectable, so there is no feasible SET for a pin to shrink. The pin
#: itself did not become silently inert -- D2-a refused
#: `PlanOverrides(state_block=...)` on the kernel arm BY NAME, and
#: `tests/layers/test_layer_dispatch.py` holds that refusal.


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


#: RETIRED at P67 D2-b, both rows of "The seam":
#: `test_a_plan_carries_its_model_facts_from_tier_zero_and_not_from_a_second_path`
#: and `test_execute_obeys_a_synthetic_plan_that_no_planner_would_have_chosen`.
#: Their subject is `rola.expert.{Plan, derive_plan, execute}` -- the plan-as-DATA
#: seam -- which D2-c deletes whole. The second is a real loss and is recorded as
#: one: it was the adversarial handle a derive-then-immediately-use pipeline does
#: not have, and the conformance family that used the same handle
#: (`producer/stale-plan`) retired with it. What replaces the seam is that there is
#: no plan: the chunk arm's geometry is the manifest's, and `rola_op` refuses a pin
#: over it rather than accepting one it cannot honor.


# ---------------------------------------------------------------------------
# The installed layout
# ---------------------------------------------------------------------------


def test_the_prefill_path_imports_from_the_INSTALLED_package_layout():
    """The release blocker this gate exists to close, gated FOREVER.

    **THE DEFECT.** A module under `rola/` reached a package under `benchmarks/`
    by inserting that directory onto `sys.path` RELATIVE TO ITS OWN FILE. That
    resolves from a source checkout and from nowhere else: `setup.py` packages
    `find_packages(include=["rola", "rola.*"])`, so an installed wheel has no
    `benchmarks/` sibling and every relocated name raised `ModuleNotFoundError`
    there.

    **WHY A SUBPROCESS OVER A SYNTHESIZED LAYOUT.** `tests/conftest.py` puts
    `benchmarks/` on `sys.path` for the bench packages, so an in-process import
    proves nothing about a wheel. This builds the INSTALLED layout -- a directory
    whose only entry is the `rola` package -- runs a fresh interpreter with exactly
    that on the path, and imports the whole prefill path there. A regression
    re-introducing a `benchmarks/` reach fails here with the import error itself,
    and one that merely IMPORTS a benchmarks module fails on the second assertion.

    The module list is the prefill path itself -- the facade, the chunk arm, the
    state, decode, paging -- and nothing else: a module the prefill path does not
    resolve does not belong in a probe about what the prefill path resolves. The
    general `benchmarks/` clause is the release blocker.
    """
    import os
    import pathlib
    import subprocess
    import sys
    import tempfile

    root = pathlib.Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        site = pathlib.Path(tmp)
        # The wheel's shape: `rola/` and nothing beside it. A symlink rather than a
        # copy so the built extension is the one under test and not a stale duplicate.
        (site / "rola").symlink_to(root / "rola", target_is_directory=True)
        assert not (site / "benchmarks").exists()

        probe = (
            "import sys;"
            # Everything the CUDA prefill path resolves, imported by name.
            "import rola;"
            "from rola.layers.rola import RoLA;"
            "from rola.interface import rola_op, dispatch_for;"
            "from rola.ops.chunk import CHUNK_TOKENS, run, stats, unsupported;"
            "from rola.ops.decode import freeze_decode_config;"
            "from rola.ops.paging import PageArena, MMA_K_QUANTUM;"
            # Nothing may have come from a `benchmarks/` or `tests/` tree.
            "bad = [m.__name__ for m in list(sys.modules.values())"
            " if getattr(m, '__file__', None)"
            " and ('benchmarks' in str(m.__file__) or '/tests/' in str(m.__file__))];"
            "print(('BAD' + repr(bad)) if bad else 'CLEAN')")

        env = dict(os.environ)
        env["PYTHONPATH"] = str(site)
        # `-P` (3.11+) keeps the script's directory off `sys.path`, so the checkout
        # cannot leak in through the cwd.
        out = subprocess.run([sys.executable, "-P", "-c", probe],
                             capture_output=True, text=True, cwd=tmp, env=env)

    assert out.returncode == 0, (
        "the prefill path does not import from the installed package layout -- this is "
        f"the ModuleNotFoundError class this gate exists to close.\n{out.stderr}")
    assert out.stdout.strip().endswith("CLEAN"), (
        f"the prefill path pulled in a `benchmarks/` or `tests/` module, which it "
        f"must not: {out.stdout!r}")
