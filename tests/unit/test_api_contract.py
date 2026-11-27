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

import importlib

import pytest
import torch

import fsla
from fsla.routing import quant
from fsla.routing.factors import RouteFactors
from fsla.routing.quant import ScaledBlock


def _routes(*, widths=(4, 4), hidden=32, heads=2, group=1, read=None, **kwargs):
    """A layer over a declared routing, softmax on the write side and by default on the read: the producer is the
    layer's."""
    D = len(widths)
    read = ((fsla.softmax(),) * D,) if read is None else read
    routing = fsla.Routing(topology=tuple(widths), write=(fsla.softmax(),) * D, read=read, **kwargs)
    return fsla.FSLA(routing, hidden_size=hidden, num_heads=heads, group=group, d_v=32)


def _values(v):
    return ScaledBlock.quantize(v, quant.VALUES)


def _x(layer, *, B=2, T=6, dtype=torch.float64):
    return torch.randn(B, T, layer.hidden_size, dtype=dtype)


def _factors(layer, **kwargs) -> RouteFactors:
    """The routes of a stretch of tokens, from the layer's producer."""
    return layer._produce(_x(layer, **kwargs))[0]


def _validate(routes, v) -> None:
    """One walk of the engine's fast-fail over `(routes, v)`: the prefill call's `validate_context`, which is where
    the bundle contract is gated, before any fact is found."""
    from fsla.engine.prefill_call import validate_context

    values = _values(v) if isinstance(v, torch.Tensor) else v
    validate_context(_context(routes, values))


def _context(routes, values):
    """The call's context: the bundle's own bits as the PREFILL-layout masks (validation reads the masks' shapes) (a non-bundle is the type
    refusal's, which speaks before the masks are read)."""
    from fsla.engine.prefill_call import PrefillContext

    if not isinstance(routes, RouteFactors) or not isinstance(values, ScaledBlock):
        return PrefillContext(routes=routes, values=values, masks=None, density=None)
    masks, density = fsla.masks_and_density(routes)
    return PrefillContext(routes=routes, values=values, masks=masks, density=density)


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
    ("levels", "one block per routing level"),
    ("prefix", "a side's routing levels share one prefix"),
    ("form", "in the u8 rows form"),
    ("dtype", "form's torch.uint8"),
    ("scale_shape", "is the rows' maxima"),
])
def test_a_malformed_hand_built_bundle_is_refused_by_its_structure(break_it, message):
    """Unsealing removed the ORIGIN check, not the invariants. Each row plants one
    structural violation the derivation code could never have produced, and the
    refusal is the observation.

    Deliberately NOT among them: unit-sum-ness. No level is required to sum to one
    (docs/CONSTITUTION.md#r23), and conformance fixtures plant amplitudes of any
    magnitude on purpose.
    """
    import dataclasses

    producer = _routes()
    f = _factors(producer)
    read, write = list(f.read), list(f.write)

    def rebuilt(block, **changes):
        return ScaledBlock(**{**{name: getattr(block, name) for name in ("data", "scale", "form")}, **changes})

    with pytest.raises((ValueError, TypeError), match=message):
        if break_it == "width":
            read[0] = rebuilt(read[0], data=read[0].data[..., :2])
        elif break_it == "levels":
            read = read[:1]
        elif break_it == "prefix":
            write[1] = rebuilt(write[1], data=write[1].data[:1], scale=write[1].scale[:1])
        elif break_it == "form":
            #: every level's rows are the ROWS form: plant the values' form over the same rows
            write[0] = dataclasses.replace(write[0], form=quant.VALUES, data=write[0].data.to(torch.bfloat16),
                                           source=None)
        elif break_it == "dtype":
            #: every level's rows have one stored type, u8: plant another
            write[0] = rebuilt(write[0], data=write[0].data.to(torch.float16))
        elif break_it == "scale_shape":
            write[0] = rebuilt(write[0], scale=write[0].scale[..., :1])
        RouteFactors(topology=f.topology, read=read, write=write)


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
    assert routes.write[0].scale.shape == (2, 6, 2) and routes.read[0].scale.shape == (2, 6, 4)


def test_a_grouped_layer_produces_its_read_heads_each_by_its_own_tuple():
    """THE ROUTING'S G READ HEADS (docs/api.md section 2.3): the layer's ``group`` query heads read each write head,
    head ``g`` by the routing's ``g``-th read tuple -- a tied entry its write head's rows on that level -- and the
    producer's bundle is the grouped one the prefill admits."""
    layer = _routes(group=2, read=((fsla.softmax(), fsla.softmax()), (fsla.tied(), fsla.softmax()))).double()
    assert layer.query_heads == 4 and layer.o_proj.in_features == 4 * 32
    routes, values, masks, density = layer._produce(_x(layer))
    assert (routes.heads, routes.write_heads, routes.group) == (4, 2, 2)
    read, write = routes.dequantized()
    #: query head w * 2 + 1 is write head w's tied read head: its level 0 is the write side's, but for the gain
    #: folded into write level 0's scale
    torch.testing.assert_close(read[0][:, :, 1::2] / read[0][:, :, 1::2].amax(-1, keepdim=True),
                               write[0] / write[0].amax(-1, keepdim=True), rtol=0, atol=1 / 255)
    assert not torch.equal(read[1][:, :, 1::2], write[1]), "the untied head's level is its own solve"
    _admitted(routes, values.dequantized().double())
    assert tuple(density.shape) == (2, 2)


def test_a_read_count_that_is_not_the_layers_group_is_refused_naming_both():
    two = ((fsla.softmax(), fsla.softmax()), (fsla.tied(), fsla.tied()))
    with pytest.raises(ValueError, match="2 read heads' tuples .* G=3"):
        _routes(group=3, read=two)
    with pytest.raises(ValueError, match="2 read heads' tuples .* G=1"):
        _routes(group=1, read=two)
    assert _routes(group=3).query_heads == 6, "one tuple reads every head of any group"


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
    assert torch.equal(_values(v).data.float() < 0, v < 0), "the signed form holds negative values"
    v[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="form holds finite entries"):
        _values(v)
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
    from fsla.routing.types import Topology

    deep = Topology.opaque((4, 4, 4))
    level = torch.rand(1, 2, 1, 4, dtype=torch.float64) + 0.1
    bundle = RouteFactors.from_levels(deep, (level,) * 3, (level,) * 3)
    assert all(row.dtype is torch.uint8 for row in bundle.read_rows + bundle.write_rows)
    for got in bundle.dequantized()[0]:
        top = level.amax(-1, keepdim=True)
        assert ((got.double() - level).abs() <= top / 510 * (1 + 2 ** -20)).all()
    with pytest.raises(ValueError, match="torch.uint8"):
        ScaledBlock(data=bundle.read[1].data.to(torch.float16), scale=bundle.read[1].scale, form=quant.ROWS)


def test_the_op_refuses_anything_that_is_not_a_bundle():
    """PLANTED VIOLATION: hand the op the tensors directly, as the old signature did."""
    producer = _routes()
    factors = _factors(producer)
    v = torch.randn(2, 6, 2, 8, dtype=torch.float64)
    with pytest.raises(TypeError, match="RouteFactors"):
        _validate(factors.read_rows, v)
    #: The well-formed call is the control: the refusal above must be about the TYPE,
    #: not about this fixture being unrunnable.
    _admitted(factors, v)


def test_the_op_refuses_a_bare_value_tensor_and_names_the_bundle():
    """PLANTED VIOLATION: hand the op the value tensor itself, then a routing level's block in its place. The value
    stream is a scaled block in the VALUES form -- the format the kernels read -- and the refusal says so by name."""
    from fsla.engine.prefill_call import validate_context

    producer = _routes()
    factors = _factors(producer)
    v = torch.randn(2, 6, 2, 8, dtype=torch.float64)
    with pytest.raises(TypeError, match="VALUES form"):
        validate_context(_context(factors, v))
    with pytest.raises(TypeError, match="VALUES form"):
        validate_context(_context(factors, factors.write[0]))
    _admitted(factors, v)


def test_the_value_block_is_the_format_and_its_view_is_the_rounded_projection():
    """`ScaledBlock.quantize(v, VALUES)` writes the value format against the token's largest channel: bf16 with that
    channel at 1; the scale is its magnitude in fp32, and `dequantized` is the rounded projection carrying its gradient
    (docs/api.md section 4.1)."""
    v = torch.randn(2, 6, 2, 8, dtype=torch.float64, requires_grad=True)
    values = _values(v)
    assert values.data.dtype is torch.bfloat16 and tuple(values.data.shape) == (2, 6, 2, 8)
    assert values.form is quant.VALUES and values.scale.dtype is torch.float32
    torch.testing.assert_close(values.scale, v.detach().abs().amax(-1).float())
    assert values.data.abs().amax(-1).eq(1).all()
    torch.testing.assert_close(values.dequantized().double(), v.detach(), rtol=2.0 ** -8, atol=0)
    values.dequantized().sum().backward()
    assert torch.equal(v.grad, torch.ones_like(v))
    with pytest.raises(ValueError, match="bf16"):
        ScaledBlock(data=values.data.float(), scale=values.scale, form=quant.VALUES)


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
    from fsla.routing.types import Topology

    class Uniform(torch.nn.Module):
        hidden_size, num_heads = 32, 2
        topology = Topology.opaque((4,))

        def forward(self, stream):
            B, T = stream.shape[0], stream.shape[1]
            level = torch.full((B, T, 2, 4), 0.25, dtype=stream.dtype)
            return RouteFactors.from_levels(self.topology, (level,), (level.clone(),))

    producer = Uniform()
    factors = producer(_x(producer))
    assert factors.read_rows[0].dtype is torch.uint8 and factors.write_rows[0].dtype is torch.uint8
    assert torch.equal(factors.read_rows[0], torch.full_like(factors.read_rows[0], 255)), "a flat dense row is all 255"
    v = torch.randn(2, 6, 2, 8, dtype=torch.float64)
    _admitted(factors, v)
    assert torch.isfinite(_oracle(factors, v)).all()
    with pytest.raises(TypeError, match="declared routing"):
        fsla.FSLA(producer, hidden_size=32, num_heads=2, d_v=8)
    with pytest.raises(TypeError):
        RouteFactors(topology=producer.topology, read=factors.read_rows,
                     write=factors.write_rows)  # a level-side is a scaled block, never bare rows


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
    assert [block.width for block in factors.read] == [4, 6]


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
    for level, (rows, scale, public) in enumerate(zip(factors.read_rows, factors.scales("read"), read)):
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


def test_no_descriptor_builder_is_exported_and_there_is_no_expert_door():
    """The planner's own answers are HOW the op runs, not what the model is: the producer's descriptor rules and the envelope's
    gradient predicate are the library's own, never one shelf below the model types, and no module of the package re-exports them."""
    #: The set names what the top level does not export today; the RULE is what this row states
    #: and it does not move when the set does: launch geometry is never at the top level.
    internal = {"decode_descriptor", "prefill_descriptor", "requires_backward"}
    assert internal.isdisjoint(fsla.__all__)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("fsla.expert")


def test_the_layer_takes_the_declaration_and_its_widths_and_nothing_spelled_out():
    """The constructor's shape IS the claim. A knob creeping back is what this catches."""
    import inspect

    signature = inspect.signature(fsla.FSLA.__init__)
    assert list(signature.parameters) == [
        "self", "routing", "hidden_size", "num_heads", "group",
        "d_v", "decay", "layer_idx", "segments"]
    for name in ("hidden_size", "num_heads", "group", "d_v", "decay", "layer_idx", "segments"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


#: RETIRED at P67 D2-b (ruling R-2): `test_an_expert_pin_shrinks_the_feasible_set_
#: and_an_inadmissible_one_is_refused`. Its subject is `FSLA.feasible_BC` and the
#: tiled ladder `derive.SUPPORTED_BC` behind it, both deleted with the tiled consumer: the chunk
#: arm's `BC` belongs to the built arm the manifest carries for a topology and is
#: not host-selectable, so there is no feasible SET for a pin to shrink. The pin
#: itself did not become silently inert -- the unsupported-topology arm refused
#: a `state_block` pin on the kernel arm BY NAME, and
#: `tests/integration/test_layer_dispatch.py` holds that refusal.


def test_the_calls_carry_only_their_inputs_and_no_plan_object_exists():
    """A call is `op(inputs...)`: the pinned parameter lists of the prefill and the decode step are the surface, the descriptor and the workspace are
    not exported, and no plan object is reachable from the layer or the engine."""
    import inspect

    import fsla.engine as engine

    op = inspect.signature(fsla.prefill).parameters
    assert list(op) == ["routes", "values", "masks", "density", "state", "decay"]
    assert op["density"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and op["density"].default is inspect.Parameter.empty
    assert all(op[name].kind is inspect.Parameter.KEYWORD_ONLY for name in ("state", "decay"))
    assert op["state"].default is None and op["decay"].default is None
    step = inspect.signature(fsla.decode_step_gated).parameters
    assert list(step) == ["routes", "values", "masks", "state", "decay"]
    assert step["decay"].kind is inspect.Parameter.KEYWORD_ONLY and step["decay"].default is None
    for gone in ("Descriptor", "descriptor", "workspace_bytes", "build_workspace", "prefill_descriptor", "decode_descriptor", "fsla_op"):
        assert not hasattr(fsla, gone) and gone not in fsla.__all__
    for gone in ("growth_pending", "admit_growth", "pool_headroom", "refill_pool", "state"):
        assert not hasattr(fsla, gone) and gone not in fsla.__all__
    for kept in ("dense_state", "FSLAState"):
        assert kept in fsla.__all__
    for gone in ("PrefillPlan", "prefill_plan", "Carve", "SidePlan"):
        assert not hasattr(engine, gone) and gone not in fsla.__all__
    for gone in ("prefill_plan", "segment_tokens", "state_bytes"):
        assert not hasattr(fsla.FSLA, gone)


# ---------------------------------------------------------------------------
# The layer's one condition, and the arms
# ---------------------------------------------------------------------------


def test_the_layer_prefill_path_refuses_before_any_kernel_runs():
    """The layer's prefill dispatch lands on `prefill`, whose fast-fail speaks before any kernel could run: in
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
    state, decode, the page codec -- and nothing else: a module the prefill path does not
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
            "from fsla.interface import prefill;"
            "from fsla.ops.decode import step_plane;"
            "from fsla.engine.facts.call import refuse_backward;"
            "from fsla.engine.rules.envelope import envelope;"
            "from fsla.engine.prefill_call import PrefillContext, run_prefill;"
            "from fsla.ops.prefill import prefill;"
            "from fsla.ops.decode import decode_launch;"
            "from fsla.ops.paging import MMA_K_QUANTUM, decode_pages, encode_pages;"
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
