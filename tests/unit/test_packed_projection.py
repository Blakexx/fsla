import pytest
import torch

from fsla.routing.projection import RouterLayout, router_logits
from fsla.routing.types import Topology, entmax, softmax, tied


def _layout(levels, untied_levels, branch, group=1):
    """``levels`` softmax levels of width ``branch``; the read entry of a level not in ``untied_levels`` is tied."""
    read = tuple(softmax() if level in untied_levels else tied() for level in range(levels))
    return RouterLayout(Topology(widths=(branch,) * levels, write=(softmax(),) * levels, read=(read,)), group)


def _reference(hidden, route_W, bias):
    """The projection written out: ``h @ route_W[head]`` per head, flat ``[B * H_w, T, columns]``."""
    equation = "bld,hdc->bhlc" if hidden.dim() == 3 else "blhd,hdc->bhlc"
    logits = torch.einsum(equation, hidden.float(), route_W.float())
    if bias is not None:
        logits = logits + bias.float()[None, :, None]
    return logits.reshape(hidden.shape[0] * route_W.shape[0], hidden.shape[1], route_W.shape[2])


@pytest.mark.parametrize("untied_levels", [(), (0, 1, 2), (0, 2)])
def test_the_packed_router_is_the_projection_and_its_gradients(untied_levels):
    torch.manual_seed(31)
    batch, length, heads, levels, hidden, branch = 2, 5, 2, 3, 7, 4
    layout = _layout(levels, untied_levels, branch)
    assert layout.packed_router_width == (levels + len(untied_levels)) * branch
    h = torch.randn(batch, length, hidden, dtype=torch.float32, requires_grad=True)
    route_W = torch.randn(heads, hidden, layout.packed_router_width, requires_grad=True)
    route_bias = torch.randn(heads, layout.packed_router_width, requires_grad=True)

    logits, gain = router_logits(h, route_W, layout, route_bias=route_bias, projection_dtype=torch.float32)
    assert gain is None
    expected = _reference(h, route_W, route_bias)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)

    loss = logits.square().mean()
    gradient_inputs = (h, route_W, route_bias)
    gradients = torch.autograd.grad(loss, gradient_inputs)
    expected_gradients = torch.autograd.grad(expected.square().mean(), gradient_inputs)
    for actual, want in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, want, rtol=1e-6, atol=1e-7)


def test_a_tied_read_entry_has_no_columns_of_its_own():
    """A tied entry reads the write side's routing on its level: the projection carries no column for it, and its
    level's rows are the write solve's."""
    layout = RouterLayout(Topology(widths=(4, 3), write=(entmax(1.5), softmax()), read=((tied(), softmax()),)), 1)
    #: write-0 (w=4), write-1 (w=3), read-1 (untied, w=3) -> 10 columns
    assert layout.packed_router_width == 10
    assert layout.read_columns(0, 0) is None and layout.read_columns(0, 1) == slice(7, 10)
    assert layout.tied_heads(0) == (0,) and layout.tied_heads(1) == ()


def test_the_packed_router_rejects_shape_mismatches():
    layout = _layout(2, (0, 1), 2)
    h = torch.randn(1, 2, 3)
    assert layout.packed_router_width == 8
    with pytest.raises(ValueError, match="route_W must be"):
        router_logits(h, torch.randn(1, 4, 3, 3), layout, projection_dtype=torch.float32)
    with pytest.raises(ValueError, match="8 routing"):
        router_logits(h, torch.randn(1, 3, 12), layout, projection_dtype=torch.float32)
    with pytest.raises(ValueError, match="hidden = route_W's 3"):
        router_logits(torch.randn(1, 2, 4), torch.randn(2, 3, 8), layout, projection_dtype=torch.float32)
    with pytest.raises(ValueError, match="head count"):
        router_logits(torch.randn(1, 2, 1, 3), torch.randn(2, 3, 8), layout, projection_dtype=torch.float32)


@pytest.mark.parametrize("untied_levels", [(), (0,), (0, 1, 2)])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("group", [1, 2])
def test_the_two_projection_layouts_are_one_gemm_and_agree_at_the_bit(untied_levels, with_bias, group):
    """`token_major=True` returns `[B, T, H_w, W]` instead of `[B*H_w, T, W]`.

    TWO claims, and they are separate. **The numbers are the same bytes**: a column of
    the output is a dot product over the full hidden dim, so it cannot depend on where
    the column is written -- `torch.equal`, not a tolerance, because a tolerance here
    would hide a real arithmetic change behind an "acceptable" one and this seam's whole
    licence is that it moves no number.

    **And the token-major form is a VIEW of one GEMM**: with no gain columns it is contiguous, where the flat
    form needs a transpose of the whole plane to make `(B, H_w)` one axis. That is the
    0.085 ms this layout exists to delete, and asserting contiguity is what keeps a
    future edit from silently reintroducing the copy while still returning the right
    numbers. Swept over the tie states and the read heads, which decide the columns, and the bias (a different
    broadcast per layout).
    """
    torch.manual_seed(23)
    heads, batch, tokens, hidden = 3, 2, 5, 4
    layout = _layout(3, untied_levels, 2, group)
    h = torch.randn(batch, tokens, hidden)
    route_W = torch.randn(heads, hidden, layout.packed_router_width)
    route_bias = torch.randn(heads, layout.packed_router_width) if with_bias else None

    def project(token_major):
        return router_logits(h, route_W, layout, route_bias=route_bias, projection_dtype=torch.float32,
                             token_major=token_major)[0]

    flat, major = project(False), project(True)
    width = layout.packed_router_width
    assert major.shape == (batch, tokens, heads, width), f"token-major must be [B, T, H_w, W], got {tuple(major.shape)}"
    folded = major.permute(0, 2, 1, 3).reshape(batch * heads, tokens, width)
    assert torch.equal(folded, flat), (
        f"the two layouts disagree in {int((folded != flat).sum())} of {flat.numel()} entries. They are "
        "the same dot products written to different addresses; any difference is a "
        "real arithmetic change, and the entmax support boundary is a threshold on "
        "these numbers.")
    assert major.is_contiguous(), (
        "the token-major projection must BE the GEMM's output; a non-contiguous one means the copy this layout "
        "exists to delete is back")
