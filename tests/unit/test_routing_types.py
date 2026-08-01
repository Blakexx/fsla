"""Focused tests for the compact RouteDescriptor ABI."""

from dataclasses import FrozenInstanceError, fields, replace

import pytest
import torch

from rola.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    READ_SUPPORT,
    RouteDescriptor,
    ResolvedRouting,
    ROUTE_DESCRIPTOR_VERSION,
    SoftmaxActivation,
    UnionRouting,
    WRITE_SUPPORT,
)


def _descriptor(plan, *, tokens=33):
    tiles = (tokens + 31) // 32
    values = torch.empty(2, tiles, plan.level_offsets[-1], 32, dtype=torch.bfloat16)
    support = torch.empty(
        2, tiles,
        sum(width for width, offset in zip(plan.branches, plan.support_offsets) if offset >= 0),
        dtype=torch.int32,
    )
    return RouteDescriptor(
        values, values, support,
        plan.level_offsets, plan.support_offsets, plan.branches,
        plan.radix_strides, plan.support_side_flags, 17, tokens, plan.topology_stamp,
    )


def test_union_routing_has_only_validated_alpha_and_fixed_split_semantics():
    union = UnionRouting()
    assert fields(UnionRouting)[0].name == "alpha"
    assert union.alpha == 1.5
    assert UnionRouting(2.0) == UnionRouting(alpha=2.0)
    assert UnionRouting(2.0) != union
    assert not hasattr(union, "tied")
    assert not hasattr(union, "read_activation")
    assert not hasattr(union, "write_activation")


def test_resolved_routing_derives_compact_descriptor_metadata():
    plan = ResolvedRouting(
        (
            IndependentRouting(True, SoftmaxActivation(), SoftmaxActivation()),
            IndependentRouting(False, EntmaxActivation(2.0), SoftmaxActivation()),
            UnionRouting(),
            IndependentRouting(True, EntmaxActivation(1.5), SoftmaxActivation()),
        ),
        (2, 3, 5, 4),
    )
    assert plan.level_offsets == (0, 2, 5, 10, 14)
    assert plan.radix_strides == (60, 20, 4, 1)
    assert plan.support_offsets == (-1, 0, 3, 8)
    assert plan.support_side_flags == (0, READ_SUPPORT, READ_SUPPORT | WRITE_SUPPORT, READ_SUPPORT)
    assert plan.topology_stamp == (
        ROUTE_DESCRIPTOR_VERSION, 4, 2, 3, 5, 4, 60, 20, 4, 1,
        0, READ_SUPPORT, READ_SUPPORT | WRITE_SUPPORT, READ_SUPPORT,
    )
    assert not hasattr(plan, "kernel_width")
    assert not hasattr(plan, "support_stream_map")
    assert not hasattr(plan, "support_stream_count")
    with pytest.raises(FrozenInstanceError):
        plan.branches = (2, 2, 2, 2)


def test_route_descriptor_has_one_compact_tile_major_and_support_arena():
    plan = ResolvedRouting(
        (IndependentRouting(False, EntmaxActivation(1.5), SoftmaxActivation()), UnionRouting(2.0)),
        (3, 5),
    )
    descriptor = _descriptor(plan)
    assert descriptor.read_values.shape == descriptor.write_values.shape == (2, 2, 8, 32)
    assert descriptor.read_values.dtype == descriptor.write_values.dtype == torch.bfloat16
    assert descriptor.support_words.shape == (2, 2, 8)
    assert descriptor.support_words.dtype == torch.int32
    assert descriptor.read_values is descriptor.write_values
    assert descriptor.token_origin == 17
    assert descriptor.token_count == 33
    assert not hasattr(descriptor, "read")
    assert not hasattr(descriptor, "write")
    assert not hasattr(descriptor, "digits")
    assert not hasattr(descriptor, "support_stream_map")
    with pytest.raises(FrozenInstanceError):
        descriptor.token_count = 1


def test_route_descriptor_rejects_noncompact_or_inconsistent_metadata():
    plan = ResolvedRouting((UnionRouting(),), (4,))
    values = torch.empty(1, 1, 4, 32, dtype=torch.bfloat16)
    support = torch.empty(1, 1, 4, dtype=torch.int32)
    with pytest.raises(ValueError, match="topology_stamp"):
        RouteDescriptor(
            values, values, support, (0, 4), (0,), (4,), (1,), (READ_SUPPORT | WRITE_SUPPORT,),
            0, 1, (ROUTE_DESCRIPTOR_VERSION, 1, 4, 1, READ_SUPPORT),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("level_offsets", (0, True)),
        ("support_offsets", (0.0,)),
        ("level_widths", (4.0,)),
        ("radix_strides", (1.0,)),
        ("support_side_flags", (3.0,)),
        ("topology_stamp", (float(ROUTE_DESCRIPTOR_VERSION), 1, 4, 1, 3)),
    ],
)
def test_route_descriptor_requires_exact_integer_metadata(field, value):
    plan = ResolvedRouting((UnionRouting(),), (4,))
    descriptor = _descriptor(plan, tokens=1)
    with pytest.raises(TypeError, match="exact ints"):
        replace(descriptor, **{field: value})


@pytest.mark.parametrize("field", ["token_origin", "token_count"])
def test_route_descriptor_rejects_non_exact_integer_token_metadata(field):
    plan = ResolvedRouting((UnionRouting(),), (4,))
    descriptor = _descriptor(plan, tokens=1)
    with pytest.raises(ValueError, match="exact int"):
        replace(descriptor, **{field: True})


@pytest.mark.parametrize(
    "factory,error",
    [
        (lambda: EntmaxActivation(True), TypeError),
        (lambda: EntmaxActivation(1), ValueError),
        (lambda: EntmaxActivation(1.8), ValueError),
        (lambda: UnionRouting(True), TypeError),
        (lambda: UnionRouting(float("nan")), ValueError),
        (lambda: UnionRouting(1), ValueError),
        (lambda: UnionRouting(1.8), ValueError),
        (lambda: IndependentRouting(1, SoftmaxActivation(), SoftmaxActivation()), TypeError),
        (lambda: IndependentRouting(False, "softmax", SoftmaxActivation()), TypeError),
        (lambda: IndependentRouting(False, EntmaxActivation(1.5), EntmaxActivation(1.5)), ValueError),
        (lambda: IndependentRouting(True, EntmaxActivation(1.5), EntmaxActivation(2.0)), ValueError),
        (lambda: ResolvedRouting((), ()), ValueError),
        (lambda: ResolvedRouting((UnionRouting(),), ()), ValueError),
        (lambda: ResolvedRouting((UnionRouting(),), (True,)), TypeError),
        (lambda: ResolvedRouting((UnionRouting(),), (1,)), ValueError),
    ],
)
def test_invalid_routing_values_are_rejected(factory, error):
    with pytest.raises(error):
        factory()


def test_tied_asymmetric_routing_does_not_share_a_solve():
    asymmetric = IndependentRouting(True, SoftmaxActivation(), EntmaxActivation(1.5))
    equal = IndependentRouting(True, EntmaxActivation(1.5), EntmaxActivation(1.5))
    assert not asymmetric.shares_solve
    assert equal.shares_solve
    assert ResolvedRouting((asymmetric, equal), (2, 2)).shared_solve_levels == (1,)


@pytest.mark.parametrize(
    "tied,read_alpha,write_alpha",
    [
        (False, 1.5, 1.5),
        (True, 1.5, 2.0),
    ],
)
def test_independent_routing_rejects_distinct_sparse_support_solves(tied, read_alpha, write_alpha):
    with pytest.raises(ValueError, match="UnionRouting"):
        IndependentRouting(
            tied,
            EntmaxActivation(read_alpha),
            EntmaxActivation(write_alpha),
        )


def test_sparse_branch_width_is_bounded_without_limiting_total_capacity():
    sparse = IndependentRouting(False, EntmaxActivation(1.5), SoftmaxActivation())
    with pytest.raises(ValueError, match="branch width must be <= 256"):
        ResolvedRouting((sparse,), (257,))
    assert ResolvedRouting((sparse, sparse, sparse), (256, 256, 256)).state_count == 256 ** 3


def test_packed_router_shape_validation_derives_heads():
    plan = ResolvedRouting(
        (IndependentRouting(False, SoftmaxActivation(), SoftmaxActivation()), UnionRouting()),
        (2, 3),
    )
    route_W = torch.randn(2, 4, 5, 3)
    route_bias = torch.randn(2, 4, 3)
    assert plan.validate_packed_router(route_W, route_bias) == 2
    with pytest.raises(ValueError, match="4"):
        plan.validate_packed_router(torch.randn(2, 3, 5, 3))
    with pytest.raises(ValueError, match="3"):
        plan.validate_packed_router(torch.randn(2, 4, 5, 2))
    with pytest.raises(ValueError, match="route_bias"):
        plan.validate_packed_router(route_W, torch.randn(2, 4, 2))
