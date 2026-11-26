# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""SEGMENTATION ON THE HOST (docs/api.md section 4.2): the plan of a call from its length and a maximum, and the two handles.

A handle states the most tokens a step may hold and the plan takes the largest whole number of windows under it. A sequence no
longer than the maximum is one step, whatever the window; one that must be split under a maximum below one window is refused.
"""
from __future__ import annotations

import pytest
import torch

import fsla
from fsla.segmentation import SegmentContext, SegmentPlan

W = 1024


def _context(**over) -> SegmentContext:
    return SegmentContext(**{"widths": (256, 256), "masked": (3, 3), "heads": (4, 4), "device": torch.device("cuda", 0), "window": W,
                             **over})


@pytest.mark.parametrize("length,maximum,steps,tokens,last", [
    (300, None, 1, 300, 300),          # no limit: one step, a partial window
    (300, 500, 1, 300, 300),           # under the maximum, below a window: one step, nothing padded
    (10_000, 4_000, 4, 3_072, 784),    # three whole windows a step, the last shorter
    (4_096, 1_024, 4, 1_024, 1_024),   # a window a step
    (4_096, 4_096, 1, 4_096, 4_096),   # exactly the maximum: one step
    (2_049, 2_048, 2, 2_048, 1),       # one token past: a last step of one token
])
def test_the_plan_is_the_largest_whole_windows_under_the_maximum(length, maximum, steps, tokens, last):
    plan = fsla.segment_plan(length, maximum, window=W)
    assert plan == SegmentPlan(steps=steps, tokens=tokens, length=length) and plan.last == last
    bounds = plan.bounds()
    assert len(bounds) == steps and bounds[0][0] == 0 and bounds[-1][1] == length
    assert all(a_end == b_start for (_, a_end), (b_start, _) in zip(bounds, bounds[1:], strict=False))
    assert all(start % W == 0 for start, _ in bounds), "every step starts on a window boundary"
    assert all(end - start <= (maximum or length) for start, end in bounds)


@pytest.mark.parametrize("length,maximum", [(10_000, 500), (800, 500), (1_025, 1_023)])
def test_a_sequence_that_must_be_split_under_a_maximum_below_one_window_is_refused_naming_the_window(length, maximum):
    with pytest.raises(ValueError, match=rf"the window is {W} tokens") as refused:
        fsla.segment_plan(length, maximum, window=W)
    assert f"maximum of {maximum}" in str(refused.value)


@pytest.mark.parametrize("call", [lambda: fsla.segment_plan(0, 8, window=W), lambda: fsla.segment_plan(8, 0, window=W),
                                  lambda: fsla.segment_plan(8.0, 8, window=W), lambda: fsla.max_tokens(0), lambda: fsla.max_tokens(True),
                                  lambda: fsla.max_memory(-1)])
def test_a_length_a_maximum_and_a_limit_are_positive_exact_ints(call):
    with pytest.raises(ValueError, match="positive exact int"):
        call()


def test_the_token_handle_states_its_maximum_whatever_the_context():
    handle = fsla.max_tokens(5_000)
    assert handle(_context()) == 5_000 == handle(_context(window=512, widths=(16,)))


def test_the_layer_takes_a_handle_or_none():
    with pytest.raises(TypeError, match="segment handle"):
        fsla.FSLA(fsla.Routing([fsla.dense_routing(256), fsla.dense_routing(256)]), hidden_size=64, num_heads=2, segments=4096)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the workspace's bytes are the extension's planner function, on a card")
def test_the_memory_handle_allows_the_most_whole_windows_whose_worst_case_workspace_fits_and_refuses_where_one_does_not():
    from fsla.ops.window import window_of
    from fsla.segmentation import workspace_bytes

    context = _context(window=window_of())
    one = workspace_bytes(context, context.window)
    assert one > 0 and workspace_bytes(context, 3 * context.window) > one
    for windows in (1, 3, 7):
        limit = workspace_bytes(context, windows * context.window)
        allowed = fsla.max_memory(limit)(context)
        assert allowed % context.window == 0 and workspace_bytes(context, allowed) <= limit
        assert workspace_bytes(context, allowed + context.window) > limit, "the next window would not fit"
    with pytest.raises(ValueError, match="no segmentation serves a memory limit"):
        fsla.max_memory(one - 1)(context)
    #: a level-side that can hold no mask has no leaf groups of its own: the worst case, and so the bytes, are smaller
    assert workspace_bytes(_context(window=window_of(), masked=(0, 0)), context.window) < one
