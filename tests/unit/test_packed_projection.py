import pytest
import torch

from rola.routing.projection import packed_router_logits
from rola.routing.types import IndependentRouting, ResolvedRouting, SoftmaxActivation, UnionRouting


def _routing_for_untied(levels, untied_levels, branch):
    return ResolvedRouting(
        tuple(
            IndependentRouting(level not in untied_levels, SoftmaxActivation(), SoftmaxActivation())
            for level in range(levels)
        ),
        (branch,) * levels,
    )


def _full_read(route_W, route_bias, routing):
    indices = list(routing.packed_read_slot_lookup)
    return route_W[:, indices], None if route_bias is None else route_bias[:, indices]


def _reference(hidden, weight, bias, heads):
    equation = "bld,hsdc->bhlsc" if hidden.dim() == 3 else "blhd,hsdc->bhlsc"
    logits = torch.einsum(equation, hidden.float(), weight.float())
    if bias is not None:
        logits = logits + bias.float()[None, :, None]
    # The production layout is PACKED (`ResolvedRouting.packed_logit_width`): one side's
    # levels laid end to end on `level_offsets`, with no per-level padding to
    # `real_width`. These fixtures use uniform widths, so the padded slot axis flattens
    # exactly onto those offsets and the reference stays a plain reshape.
    return logits.reshape(
        hidden.shape[0] * heads, hidden.shape[1], weight.shape[1] * weight.shape[3])


def _expected_read(hidden, write_hidden, route_W, route_bias, routing, expected_write, heads):
    if write_hidden is hidden:
        read_W, read_bias = _full_read(route_W, route_bias, routing)
        return _reference(hidden, read_W, read_bias, heads)
    if not routing.untied_levels:
        return expected_write
    read_bias = None if route_bias is None else route_bias[:, routing.D:]
    untied = _reference(hidden, route_W[:, routing.D:], read_bias, heads)
    slots = {level: slot for slot, level in enumerate(routing.untied_levels)}

    def block(source, position):
        width = routing.branches[0]
        return source[:, :, position * width:(position + 1) * width]

    return torch.cat([
        block(untied, slots[level]) if level in slots else block(expected_write, level)
        for level in range(routing.D)
    ], dim=2)


@pytest.mark.parametrize("untied_levels", [(), (0, 1, 2), (0, 2)])
@pytest.mark.parametrize("distinct_write_stream", [False, True])
def test_packed_router_matches_separate_projection_and_gradients(untied_levels, distinct_write_stream):
    torch.manual_seed(31)
    batch, length, heads, levels, hidden, branch = 2, 5, 2, 3, 7, 4
    routing = _routing_for_untied(levels, untied_levels, branch)
    h = torch.randn(batch, length, hidden, dtype=torch.float32, requires_grad=True)
    h_w = torch.randn_like(h, requires_grad=True) if distinct_write_stream else None
    route_W = torch.randn(heads, routing.packed_slot_count, hidden, branch, requires_grad=True)
    route_bias = torch.randn(heads, routing.packed_slot_count, branch, requires_grad=True)

    read_logits, write_logits = packed_router_logits(
        h, route_W, routing, route_bias=route_bias, h_w=h_w)
    write_hidden = h if h_w is None else h_w
    expected_write = _reference(write_hidden, route_W[:, :levels], route_bias[:, :levels], heads)
    expected_read = _expected_read(
        h, write_hidden, route_W, route_bias, routing, expected_write, heads)
    torch.testing.assert_close(read_logits, expected_read, rtol=0, atol=0)
    torch.testing.assert_close(write_logits, expected_write, rtol=0, atol=0)

    loss = read_logits.square().mean() + 0.7 * write_logits.square().mean()
    gradient_inputs = (h, route_W, route_bias) + (() if h_w is None else (h_w,))
    gradients = torch.autograd.grad(loss, gradient_inputs, allow_unused=True)
    expected_loss = expected_read.square().mean() + 0.7 * expected_write.square().mean()
    expected_gradients = torch.autograd.grad(
        expected_loss,
        gradient_inputs,
        allow_unused=True,
    )
    for actual, expected in zip(gradients, expected_gradients):
        if actual is None or expected is None:
            assert actual is expected
        else:
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_packed_router_rejects_plan_shape_mismatches():
    routing = ResolvedRouting(
        (IndependentRouting(False, SoftmaxActivation(), SoftmaxActivation()), UnionRouting()),
        (2, 3),
    )
    h = torch.randn(1, 2, 3)
    with pytest.raises(ValueError, match="4"):
        packed_router_logits(h, torch.randn(1, 3, 3, 3), routing)
    with pytest.raises(ValueError, match="3"):
        packed_router_logits(h, torch.randn(1, 4, 3, 2), routing)


def test_packed_router_rejects_hidden_width_mismatch():
    routing = _routing_for_untied(2, (), 2)
    route_W = torch.randn(2, routing.packed_slot_count, 3, 2)

    with pytest.raises(ValueError, match="hidden width"):
        packed_router_logits(torch.randn(1, 2, 4), route_W, routing)


def test_packed_router_rejects_rank4_head_count_mismatch():
    routing = _routing_for_untied(2, (), 2)
    route_W = torch.randn(2, routing.packed_slot_count, 3, 2)

    with pytest.raises(ValueError, match="head count"):
        packed_router_logits(torch.randn(1, 2, 1, 3), route_W, routing)


def test_packed_router_rejects_mismatched_write_hidden():
    routing = _routing_for_untied(2, (), 2)
    route_W = torch.randn(2, routing.packed_slot_count, 3, 2)
    h = torch.randn(1, 2, 3)

    with pytest.raises(ValueError, match="shape and device"):
        packed_router_logits(h, route_W, routing, h_w=torch.randn(1, 3, 3))


@pytest.mark.parametrize("untied_levels", [(), (0, 1, 2)])
def test_uniform_tie_state_returns_read_write_views_of_one_projection(untied_levels):
    routing = _routing_for_untied(3, untied_levels, 2)
    h = torch.randn(1, 2, 3)
    route_W = torch.randn(1, routing.packed_slot_count, 3, 2)
    read_logits, write_logits = packed_router_logits(h, route_W, routing)
    if not untied_levels:
        assert read_logits.untyped_storage().data_ptr() == write_logits.untyped_storage().data_ptr()
