"""Focused CUDA tests for compact RouteDescriptor production."""

import pytest
import torch

from rola.routing.entmax.production import (
    production_factor,
    production_routing_factors,
)
from rola.routing.entmax.reference import entmax
from rola.routing.reference import routing_factors as reference_routing_factors
from rola.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    ResolvedRouting,
    SoftmaxActivation,
    UnionRouting,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="production entmax requires CUDA")

_PRODUCTION_RTOL = 2e-5
_PRODUCTION_ATOL = 2e-6
_PRODUCTION_GRAD_RTOL = 5e-5
_PRODUCTION_GRAD_ATOL = 5e-6
_STORED_ROUTE_RTOL = 4e-3
_STORED_ROUTE_ATOL = 2e-3


def _assert_matches_oracle(actual, expected, *, gradient=False):
    rtol = _PRODUCTION_GRAD_RTOL if gradient else _PRODUCTION_RTOL
    atol = _PRODUCTION_GRAD_ATOL if gradient else _PRODUCTION_ATOL
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, check_dtype=False)


def _unpack_token_support(words, length):
    tokens = torch.arange(length, device=words.device, dtype=torch.long)
    selected = words.index_select(-2, torch.div(tokens, 32, rounding_mode="floor"))
    shifts = tokens.remainder(32).view(*((1,) * (selected.ndim - 2)), length, 1)
    return ((selected.to(torch.int64) >> shifts) & 1).bool()


def _union_midpoint(read_logits, write_logits):
    return 0.5 * read_logits + 0.5 * write_logits


def _untile_level(arena, descriptor, level):
    offset = descriptor.level_offsets[level]
    width = descriptor.level_widths[level]
    values = arena[:, :, offset:offset + width, :]
    return values.permute(0, 1, 3, 2).contiguous().reshape(
        arena.shape[0], -1, width)[:, :descriptor.token_count]


def _assert_support_matches_stored_descriptor(descriptor):
    for level, (offset, flag) in enumerate(
        zip(descriptor.support_offsets, descriptor.support_side_flags)):
        if offset < 0:
            continue
        width = descriptor.level_widths[level]
        expected = torch.zeros(
            descriptor.read_values.shape[0], descriptor.token_count, width,
            device=descriptor.read_values.device, dtype=torch.bool)
        if flag & 1:
            expected |= _untile_level(descriptor.read_values, descriptor, level) != 0
        if flag & 2:
            expected |= _untile_level(descriptor.write_values, descriptor, level) != 0
        packed = descriptor.support_words[:, :, offset:offset + width]
        assert torch.equal(_unpack_token_support(packed, descriptor.token_count), expected)


def _assert_descriptor_values_match_reference(actual, expected):
    torch.testing.assert_close(
        actual.read_values.float(), expected.read_values.float(),
        rtol=_STORED_ROUTE_RTOL, atol=_STORED_ROUTE_ATOL, check_dtype=False)
    torch.testing.assert_close(
        actual.write_values.float(), expected.write_values.float(),
        rtol=_STORED_ROUTE_RTOL, atol=_STORED_ROUTE_ATOL, check_dtype=False)


def _descriptor_level_support(descriptor, level):
    offset = descriptor.support_offsets[level]
    if offset < 0:
        return None
    width = descriptor.level_widths[level]
    return _unpack_token_support(
        descriptor.support_words[:, :, offset:offset + width], descriptor.token_count)


def _stored_activation_vjp(activation, values, gradient, support=None):
    """Reference the production VJP whose probability authority is stored bf16."""
    values = values.float()
    gradient = gradient.float()
    if isinstance(activation, SoftmaxActivation):
        return values * (gradient - (values * gradient).sum(dim=-1, keepdim=True))
    assert isinstance(activation, EntmaxActivation) and support is not None
    if activation.alpha == 2.0:
        sensitivity = support.float()
    else:
        sensitivity = torch.where(support, values.sqrt(), 0.0)
    denominator = sensitivity.sum(dim=-1, keepdim=True).clamp_min(1.0)
    centered = (sensitivity * gradient).sum(dim=-1, keepdim=True) / denominator
    return sensitivity * (gradient - centered)


def _overflow_safe_union_delta(read_logits, write_logits):
    fp32_max = torch.finfo(torch.float32).max
    read_negative = read_logits < 0
    write_negative = write_logits < 0
    opposite_sign = read_negative != write_negative
    read_magnitude = read_logits.abs()
    write_magnitude = write_logits.abs()
    high = torch.maximum(read_magnitude, write_magnitude)
    low = torch.minimum(read_magnitude, write_magnitude)
    room = fp32_max - high
    opposite_magnitude = high + torch.minimum(low, room)
    opposite_delta = torch.where(read_negative, -opposite_magnitude, opposite_magnitude)
    delta = torch.where(
        opposite_sign,
        opposite_delta,
        torch.where(opposite_sign, torch.zeros_like(read_logits), read_logits)
        - torch.where(opposite_sign, torch.zeros_like(write_logits), write_logits),
    )
    saturated = opposite_sign & (high > fp32_max * 0.5) & (low > room)
    return delta, saturated


def _stored_descriptor_vjp(read_logits, write_logits, routing, descriptor, d_read, d_write):
    """Evaluate the exact analytic VJP selected by the compact production descriptor."""
    d_read_logits = torch.zeros_like(read_logits)
    d_write_logits = torch.zeros_like(write_logits)
    for index, level in enumerate(routing.levels):
        width = routing.branches[index]
        span = routing.level_logit_slice(index)
        read_values = _untile_level(descriptor.read_values, descriptor, index).float()
        write_values = _untile_level(descriptor.write_values, descriptor, index).float()
        # Autograd casts cotangents back through the public ``.float()`` view
        # before invoking the custom backward, so its stored-route boundary is
        # bf16 for both probabilities and incoming route gradients.
        read_gradient = _untile_level(d_read, descriptor, index).to(
            descriptor.read_values.dtype).float()
        write_gradient = _untile_level(d_write, descriptor, index).to(
            descriptor.write_values.dtype).float()
        support = _descriptor_level_support(descriptor, index)

        if isinstance(level, UnionRouting):
            midpoint_logits = 0.5 * read_logits[:, :, span] + 0.5 * write_logits[:, :, span]
            midpoint_values, _ = production_factor(
                midpoint_logits.detach(), "entmax", alpha=level.alpha)
            delta, saturated = _overflow_safe_union_delta(
                read_logits[:, :, span], write_logits[:, :, span])
            read_gate = torch.sigmoid(delta)
            write_gate = torch.sigmoid(-delta)
            d_log_weight = write_values * (
                write_gradient - (write_values * write_gradient).sum(dim=-1, keepdim=True))
            safe_midpoint = torch.where(
                support & (midpoint_values != 0), midpoint_values, torch.ones_like(midpoint_values))
            d_midpoint_values = read_gradient * read_gate + torch.where(
                support, d_log_weight / safe_midpoint, 0.0)
            d_midpoint_logits = _stored_activation_vjp(
                EntmaxActivation(level.alpha), midpoint_values, d_midpoint_values, support)
            d_delta = (
                read_gradient * read_values * write_gate - d_log_weight * read_gate
            ) * (~saturated).float()
            d_read_logits[:, :, span] = 0.5 * d_midpoint_logits + d_delta
            d_write_logits[:, :, span] = 0.5 * d_midpoint_logits - d_delta
            continue

        assert isinstance(level, IndependentRouting)
        if level.shares_solve:
            d_write_logits[:, :, span] = _stored_activation_vjp(
                level.write_activation, read_values,
                read_gradient + write_gradient, support)
        elif level.tied:
            d_write_logits[:, :, span] = (
                _stored_activation_vjp(
                    level.read_activation, read_values, read_gradient, support)
                + _stored_activation_vjp(
                    level.write_activation, write_values, write_gradient, support)
            )
        else:
            d_read_logits[:, :, span] = _stored_activation_vjp(
                level.read_activation, read_values, read_gradient, support)
            d_write_logits[:, :, span] = _stored_activation_vjp(
                level.write_activation, write_values, write_gradient, support)
    return d_read_logits, d_write_logits


def _logits(routing, tokens=33, seed=31):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (2, tokens, routing.packed_logit_width)
    return torch.randn(shape, device="cuda", dtype=torch.float32, generator=generator)


@pytest.mark.parametrize("alpha", [1.5, 2.0])
def test_production_factor_mask_ties_support_values_and_vjp(alpha):
    z = torch.tensor([[4.0, 4.0, 0.25, -8.0, 11.0]], device="cuda", dtype=torch.float32)
    mask = torch.tensor([[True, True, True, False, True]], device="cuda")
    coeff = torch.tensor([[0.4, -0.7, 1.2, 0.0, -0.2]], device="cuda", dtype=torch.float32)
    zp = z.detach().clone().requires_grad_()
    p, support_words = production_factor(zp, "entmax", alpha=alpha, mask=mask)
    zr = z.detach().clone().to(torch.float64).requires_grad_()
    ref = entmax(zr, alpha, dim=-1, n_iter=80, mask=mask)
    support = _unpack_token_support(support_words, z.shape[-2])
    assert support_words.dtype == torch.int32
    assert support_words.shape == (1, z.shape[-1])
    assert torch.equal(support, p != 0)
    assert p.dtype == torch.float32
    _assert_matches_oracle(p, ref)
    (p * coeff).sum().backward()
    (ref * coeff).sum().backward()
    _assert_matches_oracle(zp.grad, zr.grad, gradient=True)


@pytest.mark.parametrize("alpha", [1.5, 2.0])
@pytest.mark.parametrize("length", [1, 31, 32, 33, 65])
def test_production_factor_packs_bit31_and_zeroes_tail_words(alpha, length):
    z = torch.full((2, length, 3), -10.0, device="cuda", dtype=torch.float32)
    z[..., 0] = 5.0
    p, support_words = production_factor(z, "entmax", alpha=alpha)

    assert torch.equal(_unpack_token_support(support_words, length), p != 0)
    assert torch.all(p[..., 0] != 0)
    raw_words = support_words.to(torch.int64) & 0xFFFFFFFF
    if length >= 32:
        assert torch.all((raw_words[:, 0, 0] & (1 << 31)) != 0)
    remainder = length % 32
    if remainder:
        tail_bits = 0xFFFFFFFF ^ ((1 << remainder) - 1)
        assert torch.count_nonzero(raw_words[:, -1, :] & tail_bits) == 0
    _assert_matches_oracle(p, entmax(z.to(torch.float64), alpha, dim=-1, n_iter=80))


def test_production_factor_packs_masked_arbitrary_leading_dimensions():
    gen = torch.Generator(device="cuda").manual_seed(68)
    z = torch.randn(2, 3, 33, 4, device="cuda", dtype=torch.float32, generator=gen)
    batch = torch.arange(2, device="cuda")[:, None, None, None]
    token = torch.arange(33, device="cuda")[None, None, :, None]
    branch = torch.arange(4, device="cuda")[None, None, None, :]
    mask = ((batch + token + branch) % 3 != 0).expand(2, 1, 33, 4).clone()
    mask[..., 0] = True

    p, support_words = production_factor(z, "entmax", alpha=1.5, mask=mask)
    ref = entmax(z.to(torch.float64), 1.5, dim=-1, n_iter=80, mask=mask)

    assert support_words.shape == (2, 3, 2, 4)
    assert torch.equal(_unpack_token_support(support_words, z.shape[-2]), p != 0)
    _assert_matches_oracle(p, ref)


def test_production_factor_empty_rows_preserve_autograd_contract():
    z = torch.empty(0, 4, device="cuda", dtype=torch.float32, requires_grad=True)
    p, support_words = production_factor(z, "entmax", alpha=1.5)
    assert p.shape == z.shape
    assert support_words.shape == (0, z.shape[-1])
    assert support_words.dtype == torch.int32
    p.sum().backward()
    assert z.grad is not None and z.grad.shape == z.shape

    routing = ResolvedRouting(
        (IndependentRouting(False, EntmaxActivation(1.5), SoftmaxActivation()),) * 2,
        (4, 4),
    )
    read_logits = torch.empty(
        1, 0, routing.packed_logit_width, device="cuda", dtype=torch.float32, requires_grad=True)
    write_logits = torch.empty_like(read_logits, requires_grad=True)
    descriptor = production_routing_factors(read_logits, write_logits, routing)
    assert descriptor.read_values.shape == descriptor.write_values.shape == (1, 0, 8, 32)
    assert descriptor.read_values.dtype == descriptor.write_values.dtype == torch.bfloat16
    assert descriptor.support_words.shape == (1, 0, 8)
    assert descriptor.support_words.dtype == torch.int32
    assert descriptor.token_count == 0
    _assert_support_matches_stored_descriptor(descriptor)
    (descriptor.read_values.float().sum() + descriptor.write_values.float().sum()).backward()
    assert read_logits.grad is not None and write_logits.grad is not None

    union_routing = ResolvedRouting((UnionRouting(2.0),), (4,))
    read_logits = torch.empty(1, 0, 4, device="cuda", dtype=torch.float32, requires_grad=True)
    write_logits = torch.empty_like(read_logits, requires_grad=True)
    union = production_routing_factors(read_logits, write_logits, union_routing)
    assert union.read_values.shape == union.write_values.shape == (1, 0, 4, 32)
    assert union.read_values.dtype == union.write_values.dtype == torch.bfloat16
    assert union.support_words.shape == (1, 0, 4)
    assert union.support_words.dtype == torch.int32
    assert union.token_count == 0
    _assert_support_matches_stored_descriptor(union)
    (union.read_values.float().sum() + union.write_values.float().sum()).backward()
    assert read_logits.grad is not None and write_logits.grad is not None


def test_production_routing_width_limit_rejects_before_jit():
    boundary = torch.empty(0, 256, device="cuda", dtype=torch.float32)
    values, support_words = production_factor(boundary, "entmax", alpha=1.5)
    assert values.shape == boundary.shape
    assert support_words.shape == (0, 256)

    too_wide = torch.empty(0, 257, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="branch width must be <= 256"):
        production_factor(too_wide, "entmax", alpha=1.5)

    with pytest.raises(ValueError, match="branch width must be <= 256"):
        ResolvedRouting(
            (IndependentRouting(False, SoftmaxActivation(), SoftmaxActivation()),),
            (257,),
        )


@pytest.mark.parametrize("alpha", [1.25, 1.8])
def test_production_factor_rejects_unsupported_alpha(alpha):
    logits = torch.zeros(1, 4, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="exactly alpha=1.5.*alpha=2.0"):
        production_factor(logits, "entmax", alpha=alpha)


@pytest.mark.parametrize("alpha", [1.5, 2.0])
def test_production_writes_compact_tile_major_bf16_and_one_support_arena(alpha):
    routing = ResolvedRouting(
        (
            IndependentRouting(False, EntmaxActivation(alpha), SoftmaxActivation()),
            UnionRouting(alpha),
            IndependentRouting(False, SoftmaxActivation(), SoftmaxActivation()),
        ),
        (3, 5, 2),
    )
    read_logits = _logits(routing, seed=31)
    write_logits = _logits(routing, seed=32)
    descriptor = production_routing_factors(read_logits, write_logits, routing, token_origin=64)

    assert descriptor.read_values.shape == descriptor.write_values.shape == (2, 2, 10, 32)
    assert descriptor.read_values.dtype == descriptor.write_values.dtype == torch.bfloat16
    assert descriptor.support_words.shape == (2, 2, 8)
    assert descriptor.support_words.dtype == torch.int32
    assert descriptor.level_offsets == (0, 3, 8, 10)
    assert descriptor.level_widths == (3, 5, 2)
    assert descriptor.support_offsets == (0, 3, -1)
    assert descriptor.support_side_flags == (1, 3, 0)
    assert descriptor.radix_strides == (10, 2, 1)
    assert descriptor.topology_stamp == (2, 3, 3, 5, 2, 10, 2, 1, 1, 3, 0)
    assert descriptor.token_origin == 64
    assert descriptor.token_count == 33
    assert descriptor.read_values.stride() == (640, 320, 32, 1)
    assert descriptor.write_values.stride() == (640, 320, 32, 1)
    _assert_support_matches_stored_descriptor(descriptor)


@pytest.mark.parametrize("alpha", [1.5, 2.0])
def test_production_independent_forward_matches_reference_and_vjp_matches_stored_contract(alpha):
    routing = ResolvedRouting(
        (
            IndependentRouting(False, EntmaxActivation(alpha), SoftmaxActivation()),
            IndependentRouting(False, SoftmaxActivation(), EntmaxActivation(alpha)),
        ),
        (4, 4),
    )
    read_base = _logits(routing, tokens=33, seed=44)
    write_base = _logits(routing, tokens=33, seed=45)
    read_production = read_base.detach().clone().requires_grad_()
    write_production = write_base.detach().clone().requires_grad_()
    read_reference = read_production.detach().clone().to(torch.float64).requires_grad_()
    write_reference = write_production.detach().clone().to(torch.float64).requires_grad_()

    production = production_routing_factors(read_production, write_production, routing)
    reference = reference_routing_factors(read_reference, write_reference, routing)
    assert production.read_values.shape == production.write_values.shape == (2, 2, 8, 32)
    assert read_production.dtype == write_production.dtype == torch.float32
    assert read_reference.dtype == write_reference.dtype == torch.float64
    assert production.read_values.dtype == production.write_values.dtype == torch.bfloat16
    _assert_descriptor_values_match_reference(production, reference)
    _assert_support_matches_stored_descriptor(production)

    generator = torch.Generator(device="cuda").manual_seed(46)
    d_read = torch.randn(
        production.read_values.shape, device="cuda", dtype=torch.float32, generator=generator)
    d_write = torch.randn(
        production.write_values.shape, device="cuda", dtype=torch.float32, generator=generator)
    production_vjp = torch.autograd.grad(
        (production.read_values.float() * d_read).sum()
        + (production.write_values.float() * d_write).sum(),
        (read_production, write_production),
    )
    expected_vjp = _stored_descriptor_vjp(
        read_production.detach(), write_production.detach(), routing, production,
        d_read, d_write,
    )
    _assert_matches_oracle(production_vjp[0], expected_vjp[0], gradient=True)
    _assert_matches_oracle(production_vjp[1], expected_vjp[1], gradient=True)


def test_production_tied_values_alias_only_for_a_whole_shared_plan():
    shared_routing = ResolvedRouting(
        (IndependentRouting(True, EntmaxActivation(1.5), EntmaxActivation(1.5)),) * 2,
        (4, 4),
    )
    logits = _logits(shared_routing, tokens=7).requires_grad_()
    descriptor = production_routing_factors(logits, logits, shared_routing)
    reference_logits = logits.detach().clone().to(torch.float64).requires_grad_()
    reference = reference_routing_factors(reference_logits, reference_logits, shared_routing)
    assert (
        descriptor.read_values.untyped_storage().data_ptr()
        == descriptor.write_values.untyped_storage().data_ptr()
    )
    assert descriptor.support_words.shape == (2, 1, 8)
    _assert_descriptor_values_match_reference(descriptor, reference)
    _assert_support_matches_stored_descriptor(descriptor)
    descriptor.read_values.float().square().sum().backward()
    assert logits.grad is not None
    assert logits.grad.dtype == torch.float32

    mixed_routing = ResolvedRouting(
        (
            IndependentRouting(True, EntmaxActivation(1.5), EntmaxActivation(1.5)),
            IndependentRouting(False, SoftmaxActivation(), SoftmaxActivation()),
        ),
        (4, 4),
    )
    mixed_logits = _logits(mixed_routing, tokens=7)
    mixed = production_routing_factors(mixed_logits, mixed_logits, mixed_routing)
    assert (
        mixed.read_values.untyped_storage().data_ptr()
        != mixed.write_values.untyped_storage().data_ptr()
    )


@pytest.mark.parametrize("alpha", [1.5, 2.0])
def test_production_union_forward_matches_reference_and_vjp_matches_stored_contract(alpha):
    gen = torch.Generator(device="cuda").manual_seed(82)
    routing = ResolvedRouting((UnionRouting(alpha),), (4,))
    read_base = torch.randn(
        2, 33, 4, device="cuda", dtype=torch.float32, generator=gen)
    write_base = torch.randn(
        read_base.shape, device="cuda", dtype=torch.float32, generator=gen)
    read_production = read_base.detach().clone().requires_grad_()
    write_production = write_base.detach().clone().requires_grad_()
    read_reference = read_production.detach().clone().to(torch.float64).requires_grad_()
    write_reference = write_production.detach().clone().to(torch.float64).requires_grad_()

    production = production_routing_factors(read_production, write_production, routing)
    reference = reference_routing_factors(read_reference, write_reference, routing)
    assert production.read_values.shape == production.write_values.shape == (2, 2, 4, 32)
    assert read_production.dtype == write_production.dtype == torch.float32
    assert read_reference.dtype == write_reference.dtype == torch.float64
    assert production.read_values.dtype == production.write_values.dtype == torch.bfloat16
    _assert_descriptor_values_match_reference(production, reference)
    _assert_support_matches_stored_descriptor(production)

    raw_words = production.support_words.to(torch.int64) & 0xFFFFFFFF
    stored_support = _unpack_token_support(production.support_words, 33)
    assert torch.equal(((raw_words[:, 0] >> 31) & 1).bool(), stored_support[:, 31])
    assert torch.count_nonzero(raw_words[:, -1] & (0xFFFFFFFF ^ 1)) == 0

    d_read = torch.randn(
        production.read_values.shape, device="cuda", dtype=torch.float32, generator=gen)
    d_write = torch.randn(
        production.write_values.shape, device="cuda", dtype=torch.float32, generator=gen)
    production_vjp = torch.autograd.grad(
        (production.read_values.float() * d_read).sum()
        + (production.write_values.float() * d_write).sum(),
        (read_production, write_production),
    )
    expected_vjp = _stored_descriptor_vjp(
        read_production.detach(), write_production.detach(), routing, production,
        d_read, d_write,
    )
    _assert_matches_oracle(production_vjp[0], expected_vjp[0], gradient=True)
    _assert_matches_oracle(production_vjp[1], expected_vjp[1], gradient=True)


@pytest.mark.parametrize("alpha", [1.5, 2.0])
@pytest.mark.parametrize("relationship", ["equal", "opposite"])
def test_production_union_handles_near_max_logits_and_vjp(alpha, relationship):
    # Opposite finite extrema overflow a raw fp32 subtraction while their midpoint
    # remains exactly representable. Equal logits use a less degenerate solve fixture.
    near_max = torch.finfo(torch.float32).max if relationship == "opposite" else 1.0e8
    read_base = torch.tensor(
        [[[near_max, -near_max, near_max, -near_max]]],
        device="cuda", dtype=torch.float32)
    write_base = read_base.detach().clone()
    if relationship == "opposite":
        write_base = -write_base
    routing = ResolvedRouting((UnionRouting(alpha),), (4,))
    read_production = read_base.detach().clone().requires_grad_()
    write_production = write_base.detach().clone().requires_grad_()
    read_reference = read_production.detach().clone().to(torch.float64).requires_grad_()
    write_reference = write_production.detach().clone().to(torch.float64).requires_grad_()
    midpoint = _union_midpoint(read_base, write_base)
    assert torch.isfinite(midpoint).all()
    if relationship == "opposite":
        assert torch.count_nonzero(midpoint) == 0
    production = production_routing_factors(read_production, write_production, routing)
    reference = reference_routing_factors(read_reference, write_reference, routing)
    assert read_production.dtype == write_production.dtype == torch.float32
    assert read_reference.dtype == write_reference.dtype == torch.float64
    d_read = torch.zeros_like(production.read_values, dtype=torch.float32)
    d_write = torch.zeros_like(production.write_values, dtype=torch.float32)
    d_read[0, 0, :, 0] = torch.tensor(
        [0.3, -0.8, 1.1, 0.4], device="cuda", dtype=torch.float32)
    d_write[0, 0, :, 0] = torch.tensor(
        [-0.2, 0.7, 0.5, -1.3], device="cuda", dtype=torch.float32)
    production_vjp = torch.autograd.grad(
        (production.read_values.float() * d_read).sum()
        + (production.write_values.float() * d_write).sum(),
        (read_production, write_production),
    )
    expected_vjp = _stored_descriptor_vjp(
        read_production.detach(), write_production.detach(), routing, production,
        d_read, d_write,
    )

    assert torch.isfinite(production.read_values).all() and torch.isfinite(production.write_values).all()
    assert torch.isfinite(production_vjp[0]).all() and torch.isfinite(production_vjp[1]).all()
    _assert_descriptor_values_match_reference(production, reference)
    _assert_matches_oracle(production_vjp[0], expected_vjp[0], gradient=True)
    _assert_matches_oracle(production_vjp[1], expected_vjp[1], gradient=True)
    if relationship == "opposite":
        assert _unpack_token_support(production.support_words, 1).all()
        torch.testing.assert_close(production_vjp[0], production_vjp[1], rtol=0, atol=0)
        torch.testing.assert_close(expected_vjp[0], expected_vjp[1], rtol=0, atol=0)


def test_production_compact_mixed_radix_support_has_tail_and_stored_bf16_authority():
    gen = torch.Generator(device="cuda").manual_seed(97)
    routing = ResolvedRouting(
        (
            IndependentRouting(False, SoftmaxActivation(), EntmaxActivation(2.0)),
            IndependentRouting(False, EntmaxActivation(1.5), SoftmaxActivation()),
            IndependentRouting(True, EntmaxActivation(1.5), EntmaxActivation(1.5)),
            UnionRouting(2.0),
        ),
        (2, 3, 5, 4),
    )
    read_logits = torch.randn(
        1, 33, routing.packed_logit_width, device="cuda", dtype=torch.float32, generator=gen)
    write_logits = torch.randn(
        read_logits.shape, device="cuda", dtype=torch.float32, generator=gen)
    descriptor = production_routing_factors(read_logits, write_logits, routing)

    assert descriptor.read_values.shape == descriptor.write_values.shape == (1, 2, 14, 32)
    assert descriptor.support_words.shape == (1, 2, 14)
    assert descriptor.level_offsets == (0, 2, 5, 10, 14)
    assert descriptor.support_offsets == (0, 2, 5, 10)
    assert descriptor.support_side_flags == (2, 1, 3, 3)
    assert descriptor.radix_strides == (60, 20, 4, 1)
    assert torch.count_nonzero(descriptor.read_values[:, 1, :, 1:]) == 0
    assert torch.count_nonzero(descriptor.write_values[:, 1, :, 1:]) == 0
    raw_tail_words = descriptor.support_words[:, -1].to(torch.int64) & 0xFFFFFFFF
    assert torch.count_nonzero(raw_tail_words & (0xFFFFFFFF ^ 1)) == 0
    _assert_support_matches_stored_descriptor(descriptor)

    narrow_routing = ResolvedRouting(
        (IndependentRouting(True, EntmaxActivation(1.5), EntmaxActivation(1.5)),),
        (2,),
    )
    narrow_logits = torch.tensor(
        [[[0.0, 1.99998]]], device="cuda", dtype=torch.float32)
    narrow = production_routing_factors(narrow_logits, narrow_logits, narrow_routing)
    stored = _untile_level(narrow.read_values, narrow, 0)
    assert stored[0, 0, 0] != 0
    assert stored.to(torch.float16)[0, 0, 0] == 0
    assert _unpack_token_support(narrow.support_words, 1)[0, 0, 0]
    _assert_support_matches_stored_descriptor(narrow)
