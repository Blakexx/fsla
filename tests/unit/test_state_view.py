# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""A STATE'S VIEW OVER A RANGE OF BATCH-HEADS, and binding from a call's shape alone, on the host (docs/internals/state.md#view).

The layer runs a call's head blocks on views of the caller's state (docs/api.md section 4.2): a block is whole sequences
or a range of one sequence's write heads, so its rows of the batch-major plane are one contiguous run, and a view shares
them. Pinned here with no kernel: the bound shape and the zeroed plane of ``_bind_plane``, a view's rows, shape and format,
that it shares the plane, and its refusals.
"""
from __future__ import annotations

import pytest
import torch

import fsla
from fsla._state import _format_of_widths
from fsla.ops.paging import bytes_equal, encode_pages

WIDTHS, DV, B, H = (16, 32), 64, 3, 4
N = WIDTHS[0] * WIDTHS[1]


def _bound() -> fsla.FSLAState:
    state = fsla.dense_state()
    state._bind_plane(batch=B, heads=H, widths=WIDTHS, d_v=DV, device=torch.device("cpu"))
    return state


def _carrying() -> fsla.FSLAState:
    """A bound state whose plane holds a random logical state."""
    state = _bound()
    torch.manual_seed(0)
    logical = torch.randn(B * H, N // 16, 16, DV + 1)
    state._plane.copy_(encode_pages(logical))
    return state


def test_binding_from_the_shape_alone_holds_a_zeroed_plane_of_the_whole_call():
    state = _bound()
    assert state.bound and state.carries and state.shape == (B, H, N, DV + 1)
    assert state.format == _format_of_widths(WIDTHS, DV, B * H)
    assert int(state._plane.count_nonzero()) == 0 and tuple(state._plane.shape)[0] == B * H


def test_binding_a_carrying_state_keeps_its_plane_and_checks_its_format():
    state = _carrying()
    plane, before = state._plane, state._plane.clone()
    state._bind_plane(batch=B, heads=H, widths=WIDTHS, d_v=DV, device=torch.device("cpu"))
    assert state._plane is plane and bytes_equal(plane, before)
    with pytest.raises(ValueError, match="bound with DV=64"):
        state._bind_plane(batch=B, heads=H, widths=WIDTHS, d_v=2 * DV, device=torch.device("cpu"))


@pytest.mark.parametrize("batch,heads", [((0, B), (0, H)), ((1, 3), (0, H)), ((2, 3), (0, H)), ((1, 2), (1, 3)),
                                         ((0, 1), (3, 4))])
def test_a_view_is_its_rows_of_the_plane_with_its_shape_and_format_restated(batch, heads):
    state = _carrying()
    view = state._view(batch, heads)
    (b0, b1), (h0, h1) = batch, heads
    rows = (b1 - b0) * (h1 - h0)
    assert view.shape == (b1 - b0, h1 - h0, N, DV + 1) and view.device == state.device
    assert view.format == _format_of_widths(WIDTHS, DV, rows)
    torch.testing.assert_close(view.materialize(), state.materialize()[b0:b1, h0:h1], rtol=0, atol=0)
    first = b0 * H + h0
    assert view._plane.data_ptr() == state._plane[first].data_ptr() and view._plane.shape[0] == rows


def test_a_write_through_a_view_lands_in_the_states_plane_in_place():
    state = _carrying()
    before = state._plane.clone()
    view = state._view((1, 2), (1, 3))
    view._plane.fill_(7)
    changed = (state._plane != before).flatten(1).any(1).nonzero().flatten().tolist()
    assert changed == [1 * H + 1, 1 * H + 2] and bool((state._plane[1 * H + 1:1 * H + 3] == 7).all())


@pytest.mark.parametrize("batch,heads", [((0, 2), (0, 2)), ((0, B + 1), (0, H)), ((1, 1), (0, H)), ((0, 1), (2, H + 1))])
def test_a_view_whose_rows_are_not_one_run_of_the_plane_is_refused(batch, heads):
    with pytest.raises(ValueError, match="one contiguous run"):
        _carrying()._view(batch, heads)


def test_a_view_of_a_state_that_carries_nothing_is_refused():
    with pytest.raises(RuntimeError, match="carries nothing"):
        fsla.dense_state()._view((0, 1), (0, 1))
