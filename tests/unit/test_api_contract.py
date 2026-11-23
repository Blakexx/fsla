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

import fsla
from fsla.routing.factors import RouteFactors
from fsla.routing.masks import PREFILL
from fsla.values import Values


def _routes(*, widths=(4, 4), hidden=32, heads=2, routing=None, **kwargs):
    """A layer over a declared routing: the producer is the layer's."""
    levels = [fsla.dense_routing(w) if routing is None else routing.at(w) for w in widths]
    return fsla.FSLA(fsla.Routing(levels, **kwargs), hidden_size=hidden, num_heads=heads, d_v=32)


def _x(layer, *, B=2, T=6, dtype=torch.float64):
    return torch.randn(B, T, layer.hidden_size, dtype=dtype)


def _factors(layer, **kwargs) -> RouteFactors:
    """The routes of a stretch of tokens, from the layer's producer."""
    return layer._produce(_x(layer, **kwargs))[0]


def _validate(routes, v) -> None:
    """One walk of the engine's fast-fail over `(routes, v)`: the prefill call's `validate_context`, which is where
    the bundle contract is gated, before any fact is found."""
    from fsla.engine.prefill_call import validate_context

    values = Values.from_projection(v) if isinstance(v, torch.Tensor) else v
    validate_context(_context(routes, values))


def _context(routes, values):
    """The call's context: the bundle's own bits as the PREFILL-layout masks and the prefill descriptor of the call, made by hand (validation reads the
    masks' shapes and the descriptor's facts) (a non-bundle is the type refusal's, which speaks before either is read)."""
    from conftest import hand_descriptor

    from fsla.engine.prefill_call import PrefillContext
    from tests.oracle.fixtures import masks_of

    if not isinstance(routes, RouteFactors) or not isinstance(values, Values):
        return PrefillContext(routes=routes, values=values, masks=None, descriptor=None)
    return PrefillContext(routes=routes, values=values, masks=masks_of(routes, PREFILL),
                          descriptor=hand_descriptor(routes, values))


def _admitted(routes, v) -> None:
    """The engine's validation ACCEPTED this call: everything about the bundle passed and `validate_context` returned (a prefill is refused by name
    where its descriptor is made, not here)."""
    with torch.no_grad():
        _validate(routes, v)


def _oracle(routes, v):
    from fsla.ops.naive import naive_fsla

    _, write = routes.dequantized()
    y, _ = naive_fsla(v, routes.dequantized()[0], write, routes.topology, None)
    return y


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
    producer = _routes()
    factors = _factors(producer)
    hand_built = RouteFactors.from_levels(factors.topology, *factors.dequantized())
    v = torch.randn(2, 6, 2, 8, dtype=torch.float64)
    _admitted(hand_built, v)
    assert torch.isfinite(_oracle(hand_built, v)).all()


@pytest.mark.parametrize("break_it,message", [
    ("width", "wide but the topology"),
    ("levels", "one tensor per routing level"),
    ("prefix", "a side's routing levels share one prefix"),
    ("dtype", "is stored as"),
    ("scale_shape", r"scale_write\[0\] is the rows' maxima"),
])
def test_a_malformed_hand_built_bundle_is_refused_by_its_structure(break_it, message):
    """Unsealing removed the ORIGIN check, not the invariants. Each row plants one
    structural violation the derivation code could never have produced, and the
    refusal is the observation.

    Deliberately NOT among them: unit-sum-ness. No level is required to sum to one
    (docs/CONSTITUTION.md#r23), and conformance fixtures plant amplitudes of any
    magnitude on purpose.
    """
    producer = _routes()
    f = _factors(producer)
    fields = dict(topology=f.topology, amp_read=list(f.amp_read), amp_write=list(f.amp_write),
                  scale_read=f.scale_read, scale_write=list(f.scale_write))
    if break_it == "width":
        fields["amp_read"][0] = fields["amp_read"][0][..., :2]
    elif break_it == "levels":
        fields["amp_read"] = fields["amp_read"][:1]
    elif break_it == "prefix":
        fields["amp_write"][1] = fields["amp_write"][1][:1]
    elif break_it == "dtype":
        #: every level's rows have one stored type, u8: plant another
        fields["amp_write"][0] = fields["amp_write"][0].to(torch.float16)
    elif break_it == "scale_shape":
        fields["scale_write"][0] = fields["scale_write"][0][..., :1]
    with pytest.raises((ValueError, TypeError), match=message):
        RouteFactors(**fields)


def _grouped(heads, write_heads, *, B=2, T=6):
    """A hand-built bundle of ``heads`` query heads over ``write_heads`` write heads, the layer's topology."""
    topology, gen = _factors(_routes()).topology, torch.Generator().manual_seed(11)

    def side(h):
        return tuple(torch.rand((B, T, h, w), generator=gen).softmax(-1) for w in topology.widths)

    return RouteFactors.from_levels(topology, side(heads), side(write_heads))


def test_a_grouped_bundle_names_its_query_heads_write_heads_and_group():
    """GROUPED QUERIES (docs/api.md section 2.1): the read side at H query heads, the write side at H_w write heads,
    each write head read by H / H_w consecutive query heads."""
    routes = _grouped(4, 2)
    assert (routes.heads, routes.write_heads, routes.group) == (4, 2, 2)
    assert routes.scale_write[0].shape == (2, 6, 2) and routes.scale_read[0].shape == (2, 6, 4)


def test_the_write_heads_must_divide_the_query_heads():
    with pytest.raises(ValueError, match="its H_w divide H"):
        _grouped(3, 2)


def test_the_prefill_reads_groups_and_a_consumer_of_one_head_count_refuses_them_by_name():
    """The prefill's kernels read groups -- the values a write head's, the bundle's rows its query heads' -- so its
    fast-fail admits a grouped bundle; a consumer that reads one head count (the decode step, the carried state)
    refuses one at its seam."""
    _admitted(_grouped(4, 2), torch.randn(2, 6, 2, 8, dtype=torch.float64))
    with pytest.raises(NotImplementedError, match="a consumer reads one head count"):
        _grouped(4, 2).refuse_grouped("a consumer")
    _grouped(2, 2).refuse_grouped("a consumer")


def test_the_decode_digit_mirrors_the_quantizer():
    """`decode_fold.cuh` and `fsla.routing.quant` state one format: the row max's digit."""
    from pathlib import Path

    from fsla.routing import quant

    text = (Path(__file__).resolve().parents[2] / "csrc/fsla/src/decode/decode_fold.cuh").read_text()
    assert "constexpr float kRowMaxDigit = 255.0f;" in text and quant.AMP_LEVELS == 255.0


def test_the_factory_refuses_an_entry_its_form_does_not_hold():
    """docs/CONSTITUTION.md#r23: a routing form is nonnegative and every form finite, and the factory refuses what its
    form does not hold -- a negative or non-finite routing level, a non-finite value -- where a signed value is held."""
    from fsla.routing import quant

    topology = _factors(_routes()).topology
    good = tuple(torch.rand(1, 2, 2, w, dtype=torch.float64) for w in topology.widths)
    for planted in (-1e-3, float("nan"), float("inf")):
        bad = (good[0].clone(),) + good[1:]
        bad[0][0, 0, 0, 0] = planted
        with pytest.raises(ValueError, match="form holds nonnegative, finite entries"):
            RouteFactors.from_levels(topology, good, bad)
    v = torch.randn(1, 2, 2, 8, dtype=torch.float64)
    assert torch.equal(Values.from_projection(v).v.float() < 0, v < 0), "the signed form holds negative values"
    v[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="form holds finite entries"):
        Values.from_projection(v)
    with pytest.raises(ValueError, match="bf16 values"):
        quant.VALUES.quantize(v)


def test_a_zero_digit_is_a_zero_at_every_level():
    """docs/CONSTITUTION.md#r23: a level's rows are u8 with no floor under them -- a zero digit dequantizes to an exact
    zero in a live row, and a dead row (scale 0) stays zero."""
    from fsla.routing import quant

    scale = torch.tensor([2.0, 0.0])
    rows = quant.ROWS.dequantize(torch.tensor([[255, 0, 3], [0, 0, 0]], dtype=torch.uint8), scale)
    assert rows[0, 1] == 0.0 and rows[0, 0] == 2.0 and torch.equal(rows[1], torch.zeros(3))


def test_rows_are_u8_at_every_depth():
    """The producer writes u8 rows at every depth, each against its own max (the bundle carries no flag); a row of any other type is refused,
    and a deep bundle dequantizes to its levels within half a digit a row."""
    from fsla.routing.types import OpaqueRouting, Topology

    deep = Topology(levels=tuple(OpaqueRouting(width=4) for _ in range(3)))
    level = torch.rand(1, 2, 1, 4, dtype=torch.float64) + 0.1
    bundle = RouteFactors.from_levels(deep, (level,) * 3, (level,) * 3)
    assert all(row.dtype is torch.uint8 for row in bundle.amp_read + bundle.amp_write)
    for got in bundle.dequantized()[0]:
        top = level.amax(-1, keepdim=True)
        assert ((got.double() - level).abs() <= top / 510 * (1 + 2 ** -20)).all()
    wrong = list(bundle.amp_read)
    wrong[1] = wrong[1].to(torch.float16)
    with pytest.raises(TypeError, match="stored as torch.uint8"):
        RouteFactors(topology=deep, amp_read=wrong, amp_write=bundle.amp_write, scale_read=bundle.scale_read,
                     scale_write=bundle.scale_write)


def test_the_op_refuses_anything_that_is_not_a_bundle():
    """PLANTED VIOLATION: hand the op the tensors directly, as the old signature did."""
    producer = _routes()
    factors = _factors(producer)
    v = torch.randn(2, 6, 2, 8, dtype=torch.float64)
    with pytest.raises(TypeError, match="RouteFactors"):
        _validate(factors.amp_read, v)
    #: The well-formed call is the control: the refusal above must be about the TYPE,
    #: not about this fixture being unrunnable.
    _admitted(factors, v)


def test_the_op_refuses_a_bare_value_tensor_and_names_the_bundle():
    """PLANTED VIOLATION: hand the op the value tensor itself. The value stream is a
    `Values` bundle -- the format the kernels read -- and the refusal says so by name."""
    from fsla.engine.prefill_call import validate_context

    producer = _routes()
    factors = _factors(producer)
    v = torch.randn(2, 6, 2, 8, dtype=torch.float64)
    with pytest.raises(TypeError, match="Values"):
        validate_context(_context(factors, v))
    _admitted(factors, v)


def test_the_value_bundle_is_the_format_and_its_view_is_the_rounded_projection():
    """`Values.from_projection` writes the value format against the token's largest channel: bf16 with that channel
    at 1; the scale is its magnitude in fp32, and `dequantized` is the rounded projection carrying its gradient
    (docs/api.md section 4.1)."""
    v = torch.randn(2, 6, 2, 8, dtype=torch.float64, requires_grad=True)
    values = Values.from_projection(v)
    assert values.v.dtype is torch.bfloat16 and tuple(values.v.shape) == (2, 6, 2, 8)
    assert values.scale.dtype is torch.float32
    torch.testing.assert_close(values.scale, v.detach().abs().amax(-1).float())
    assert values.v.abs().amax(-1).eq(1).all()
    torch.testing.assert_close(values.dequantized().double(), v.detach(), rtol=2.0 ** -8, atol=0)
    values.dequantized().sum().backward()
    assert torch.equal(v.grad, torch.ones_like(v))
    with pytest.raises(ValueError, match="bf16"):
        Values(v=values.v.float(), scale=values.scale)


def test_the_bundle_carries_v_shape_disagreement_to_the_caller_not_to_the_kernel():
    """The TOKEN ADDRESSING must agree; the value width is not a disagreement at all.

    `d_v` is `v`'s own width -- the routing carries no value geometry -- so a
    different one is a different model, not a mismatch, and the op reads it. What
    the bundle does pin is `[B, T, H_w]` (the values are a write head's), and a `v` that
    disagrees there is refused with the expected prefix in the message rather than handed to a kernel.
    """
    producer = _routes()
    factors = _factors(producer)
    with pytest.raises(ValueError, match=r"\[B, T, H_w\]=\(2, 6, 2\)"):
        _validate(factors, torch.randn(2, 5, 2, 8, dtype=torch.float64))
    #: the control: the same call at a width the bundle never declared is admitted.
    _admitted(factors, torch.randn(2, 6, 2, 9, dtype=torch.float64))


# ---------------------------------------------------------------------------
# The one slot
# ---------------------------------------------------------------------------


def test_a_producer_outside_the_framework_still_emits_the_format_or_nothing():
    """RULED 2026-09-26: production lives in our framework and emits ONE internal format. A module that returns a
    bundle can only do so through the format's epilogue (`RouteFactors.from_levels`), so what the op admits is
    always u8 unit-max rows with their scales; a bundle assembled from bare floating levels is refused by its
    structure. Such levels reach the op through the format's utilities; the LAYER takes a declared routing, and a
    callable in its place is refused (ruled 2026-09-29: the producer is a declared configuration).
    """
    from fsla.routing.types import OpaqueRouting, Topology

    class Uniform(torch.nn.Module):
        hidden_size, num_heads = 32, 2
        topology = Topology(levels=(OpaqueRouting(width=4),))

        def forward(self, stream):
            B, T = stream.shape[0], stream.shape[1]
            level = torch.full((B, T, 2, 4), 0.25, dtype=stream.dtype)
            return RouteFactors.from_levels(self.topology, (level,), (level.clone(),))

    producer = Uniform()
    factors = producer(_x(producer))
    assert factors.amp_read[0].dtype is torch.uint8 and factors.amp_write[0].dtype is torch.uint8
    assert torch.equal(factors.amp_read[0], torch.full_like(factors.amp_read[0], 255)), "a flat dense row is all 255"
    v = torch.randn(2, 6, 2, 8, dtype=torch.float64)
    _admitted(factors, v)
    assert torch.isfinite(_oracle(factors, v)).all()
    with pytest.raises(TypeError, match="declared routing"):
        fsla.FSLA(producer, hidden_size=32, num_heads=2, d_v=8)
    with pytest.raises(TypeError):
        RouteFactors(topology=producer.topology, amp_read=factors.amp_read,
                     amp_write=factors.amp_write)  # the format's fields are not optional


def test_a_layer_without_a_declared_routing_is_refused_at_construction():
    """Construction-time validation, so an unbuildable model fails on the line that
    built it rather than inside a forward several frames away.

    PLANTED VIOLATION: a module where the declaration goes.
    """
    with pytest.raises(TypeError, match="declared routing"):
        fsla.FSLA(torch.nn.Identity(), hidden_size=32, num_heads=2)


def test_one_producer_covers_many_levels_in_the_order_they_were_declared():
    """One solve group, one instance, and the level order is the order the widths
    were declared in."""
    producer = _routes(widths=(4, 6))
    assert producer.topology.widths == (4, 6)
    factors = _factors(producer)
    assert [tuple(level.shape[-1:]) for level in factors.amp_read] == [(4,), (6,)]


# ---------------------------------------------------------------------------
# Packing belongs to the producer
# ---------------------------------------------------------------------------


def test_the_public_levels_are_the_formats_rows_dequantized():
    """The solve emits ONE layout and the public tensors are permuted views of it,
    so nothing on this surface costs a copy.

    Asserted on STORAGE, which is the fact; a shape assertion would pass over a copy.

    P67 D2-b NARROWED this row, per ruling R-6. It used to also assert WHICH of the
    two layouts is the materialized one, by comparing `data_ptr()` against
    `routing_views`' folded view -- and `routing_views` is the TILED consumer's
    packing, deleted with it. The engine builds its own packed operands per call
    (`fsla/engine/facts/planes.py`), so there is no second standing layout left
    to be the one that is materialized. What survives, and is what a caller of this surface is
    owed, is that the public tensors are views.
    """
    producer = _routes(widths=(4, 6))
    factors = _factors(producer)
    #: RULED 2026-09-26: the rows (u8) and their scales ARE the stored layout; the dequantized
    #: levels are computed at a consumer's seam, for consumers that still compute on floating levels.
    read, _ = factors.dequantized()
    for level, (rows, scale, public) in enumerate(zip(factors.amp_read, factors.scale_read, read)):
        assert rows.dtype is torch.uint8 and tuple(rows.shape) == tuple(public.shape)
        assert torch.equal(rows.amax(dim=-1), torch.full_like(rows.amax(dim=-1), 255)), (
            f"level {level}'s rows are not unit-max")
        assert torch.allclose((rows.float() / 255 * scale.unsqueeze(-1)).to(public.dtype), public.detach()), (
            f"level {level}'s public level is not its rows dequantized")


# ---------------------------------------------------------------------------
# The decay-source slot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scope", ["state", "global"])
def test_every_learned_decay_scope_emits_the_same_dial_shape(scope):
    """The scopes differ in degrees of freedom, never in what the recurrence reads."""
    source = fsla.LearnedDecay(2 ** -8, widths=(4, 6), num_heads=2, scope=scope)
    dials = source().dials
    assert [tuple(d.shape) for d in dials] == [(2, 4), (2, 6)]
    assert all(((d >= 0) & (d < 1)).all() for d in dials)


def test_a_constant_decay_source_has_no_parameters_and_reaches_the_target_rate():
    source = fsla.ConstantDecay(2 ** -8, widths=(4, 4), num_heads=2)
    assert list(source.parameters()) == [], "a user-fixed rate must not be optimized"
    product = source().dials[0][0, 0] * source().dials[1][0, 0]
    assert torch.isclose(product, torch.tensor(2.0 ** -8))


def test_a_decay_source_sized_for_another_topology_is_refused_at_construction():
    """PLANTED VIOLATION: the right source, the wrong widths."""
    producer = _routes(widths=(4, 4))
    with pytest.raises(ValueError, match="widths"):
        fsla.FSLA(producer.routing, hidden_size=32, num_heads=2,
                  decay=fsla.LearnedDecay(widths=(4, 6), num_heads=2))
    fsla.FSLA(producer.routing, hidden_size=32, num_heads=2, decay=fsla.LearnedDecay(widths=(4, 4), num_heads=2))


# ---------------------------------------------------------------------------
# The quarantine
# ---------------------------------------------------------------------------


def test_no_launch_geometry_is_reachable_from_the_top_level():
    """The planner's own answers are HOW the op runs, not what the model is: the producer's descriptor rules and the envelope's
    sentence live behind a marked door, never one shelf below the model types.
    """
    #: The set names what is BEHIND the door today; the RULE is what this row states
    #: and it does not move when the set does: launch geometry is behind the marked
    #: door, never at the top level.
    quarantined = {"decode_descriptor", "prefill_descriptor", "requires_backward"}
    import fsla.expert as expert

    assert quarantined.isdisjoint(fsla.__all__)
    assert quarantined.issubset(set(expert.__all__))


def test_the_layer_takes_the_declaration_and_its_widths_and_nothing_spelled_out():
    """The constructor's shape IS the claim. A knob creeping back is what this catches."""
    import inspect

    signature = inspect.signature(fsla.FSLA.__init__)
    assert list(signature.parameters) == [
        "self", "routing", "hidden_size", "num_heads",
        "d_v", "decay", "layer_idx", "memory_budget"]
    for name in ("hidden_size", "num_heads", "d_v", "decay", "layer_idx", "memory_budget"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


#: RETIRED at P67 D2-b (ruling R-2): `test_an_expert_pin_shrinks_the_feasible_set_
#: and_an_inadmissible_one_is_refused`. Its subject is `FSLA.feasible_BC` and the
#: tiled ladder `derive.SUPPORTED_BC` behind it, both deleted with the tiled consumer: the chunk
#: arm's `BC` belongs to the built arm the manifest carries for a topology and is
#: not host-selectable, so there is no feasible SET for a pin to shrink. The pin
#: itself did not become silently inert -- the unsupported-topology arm refused
#: a `state_block` pin on the kernel arm BY NAME, and
#: `tests/integration/test_layer_dispatch.py` holds that refusal.


def test_the_op_and_its_planner_arguments_are_a_descriptor_and_a_workspace_and_no_plan_object_exists():
    """An operator is `op(data..., descriptor, workspace=None)`: the pinned parameter lists of the op, the descriptor's producer rule and the workspace's
    builder are the surface, and no plan object is reachable from the layer, the expert door or the engine."""
    import inspect

    import fsla.engine as engine
    import fsla.expert as expert
    from fsla.ops.produce import prefill_descriptor

    op = inspect.signature(fsla.fsla_op).parameters
    assert list(op) == ["routes", "values", "masks", "descriptor", "workspace", "state", "decay"]
    assert all(op[name].kind is inspect.Parameter.KEYWORD_ONLY for name in ("workspace", "state", "decay"))
    assert op["workspace"].default is None and op["state"].default is None and op["decay"].default is None
    assert list(inspect.signature(fsla.build_workspace).parameters) == ["routes", "values", "masks", "descriptor", "workspace"]
    assert list(inspect.signature(prefill_descriptor).parameters) == [
        "topology", "d_v", "decay", "tokens", "batch_heads", "stateful", "device"]
    assert fsla.workspace_bytes is not None
    for gone in ("PrefillPlan", "prefill_plan", "Carve", "SidePlan"):
        assert not hasattr(expert, gone) and not hasattr(engine, gone) and gone not in fsla.__all__
    for gone in ("prefill_plan", "segment_tokens", "state_bytes"):
        assert not hasattr(fsla.FSLA, gone)


# ---------------------------------------------------------------------------
# The layer's one condition, and the arms
# ---------------------------------------------------------------------------


def test_the_layer_prefill_path_refuses_before_any_kernel_runs():
    """The layer's prefill dispatch lands on `fsla_op`, whose fast-fail speaks before any kernel could run: in
    training mode the gradient clause (the prefill has no reverse pass yet), out of it the envelope's device
    clause for these CPU fp64 fixtures."""
    from fsla.engine.types import BackwardNotImplemented

    layer = _routes().double().train()
    with pytest.raises(BackwardNotImplemented, match="has no reverse pass"):
        layer(torch.randn(2, 6, 32, dtype=torch.float64))
    with torch.no_grad(), pytest.raises(NotImplementedError, match="CUDA kernel"):
        layer(torch.randn(2, 6, 32, dtype=torch.float64))
    assert layer.last_execution_backend is None, "the refusal let a kernel run first"


def test_an_unsupported_forward_keyword_raises_rather_than_being_swallowed():
    """PLANTED VIOLATION: the HF-compat keyword the layer used to accept and ignore."""
    layer = _routes().double().eval()
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
#: Their subject is `fsla.expert.{Plan, derive_plan, execute}` -- the plan-as-DATA
#: seam -- which the tiled consumer's deletion takes whole. The second is a real loss and is recorded as
#: one: it was the adversarial handle a derive-then-immediately-use pipeline does
#: not have, and the conformance family that used the same handle
#: (`producer/stale-plan`) retired with it. What replaces the seam is that there is
#: no plan: the chunk arm's geometry is the manifest's, and `fsla_op` refuses a pin
#: over it rather than accepting one it cannot honor.


# ---------------------------------------------------------------------------
# The installed layout
# ---------------------------------------------------------------------------


def test_the_prefill_path_imports_from_the_INSTALLED_package_layout():
    """The release blocker this gate exists to close, gated FOREVER.

    **THE DEFECT.** A module under `fsla/` reached a package under `measure/`
    by inserting that directory onto `sys.path` RELATIVE TO ITS OWN FILE. That
    resolves from a source checkout and from nowhere else: the `fsla` wheel packages
    the `fsla/` package alone (`tools/wheels.py`), so an installed wheel has no
    `measure/` sibling and every relocated name raised `ModuleNotFoundError`
    there.

    **WHY A SUBPROCESS OVER A SYNTHESIZED LAYOUT.** `tests/conftest.py` puts
    `measure/` on `sys.path` for the bench packages, so an in-process import
    proves nothing about a wheel. This builds the INSTALLED layout -- a directory
    whose only entry is the `fsla` package -- runs a fresh interpreter with exactly
    that on the path, and imports the whole prefill path there. A regression
    re-introducing a `measure/` reach fails here with the import error itself,
    and one that merely IMPORTS a measure module fails on the second assertion.

    The module list is the prefill path itself -- the facade, the prefill call, the
    state, decode, paging -- and nothing else: a module the prefill path does not
    resolve does not belong in a probe about what the prefill path resolves. The
    general `measure/` clause is the release blocker.
    """
    import os
    import pathlib
    import subprocess
    import sys
    import tempfile

    root = pathlib.Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        site = pathlib.Path(tmp)
        # The wheel's shape: `fsla/` and nothing beside it. A symlink rather than a
        # copy so the built extension is the one under test and not a stale duplicate.
        (site / "fsla").symlink_to(root / "fsla", target_is_directory=True)
        assert not (site / "measure").exists()

        probe = (
            "import sys;"
            # Everything the CUDA prefill path resolves, imported by name.
            "import fsla;"
            "from fsla.layer import FSLA;"
            "from fsla.interface import fsla_op;"
            "from fsla.ops.decode import step_plane;"
            "from fsla.engine.facts.call import refuse_backward;"
            "from fsla.engine.rules.envelope import envelope;"
            "from fsla.engine.prefill_call import PrefillContext, run_prefill;"
            "from fsla.ops.prefill import prefill;"
            "from fsla.ops.decode import decode_launch;"
            "from fsla.ops.paging import PageArena, MMA_K_QUANTUM;"
            # Nothing may have come from a `measure/` or `tests/` tree.
            "bad = [m.__name__ for m in list(sys.modules.values())"
            " if getattr(m, '__file__', None)"
            " and ('measure' in str(m.__file__) or '/tests/' in str(m.__file__))];"
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
        f"the prefill path pulled in a `measure/` or `tests/` module, which it "
        f"must not: {out.stdout!r}")
