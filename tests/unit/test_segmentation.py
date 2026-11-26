# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""SEGMENTATION (docs/api.md section 4.2, docs/internals/common/plan/segments.md): the plan of a call from its length, a
maximum and the card's window, and the two handles.

The plan is the planner's (`csrc/fsla/src/common/plan/segments.h`): on the host `plan_tool --segments` prints it for a
window given by hand, no GPU; on a card `fsla.segment_plan` and `fsla.max_memory` ask the library, which reads the card. A
handle states the most tokens a step may hold and the plan takes the largest whole number of windows under it. A sequence
no longer than the maximum is one step, whatever the window; one that must be split under a maximum below one window has no
plan.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import fsla  # noqa: E402
from fsla.segmentation import SegmentContext, SegmentPlan  # noqa: E402

TOOL = ROOT / "csrc" / "fsla" / "plan_tool"
W = 1024
_ON_A_CARD = pytest.mark.skipif(not torch.cuda.is_available(), reason="the library's planner reads the card")


def _planned(calls: list[tuple[int, int | None]], window: int = W) -> list[SegmentPlan]:
    """The planner's plans of ``(length, maximum)`` calls at ``window``; ``steps`` 0 where a call has none."""
    run = subprocess.run([str(TOOL), "--segments", ";".join(f"{length},{maximum or 0},{window}" for length, maximum in calls)],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    out = [json.loads(line) for line in run.stdout.splitlines()]
    assert len(out) == len(calls)
    return [SegmentPlan(steps=o["steps"], tokens=o["tokens"], length=length) for o, (length, _) in zip(out, calls, strict=True)]


def _context(**over) -> SegmentContext:
    return SegmentContext(**{"widths": (256, 256), "masked": (3, 3), "heads": (4, 4), "value_width": 64, "device": torch.device("cuda", 0),
                             **over})


CASES = [
    (300, None, 1, 300, 300),          # no limit: one step, a partial window
    (300, 500, 1, 300, 300),           # under the maximum, below a window: one step, nothing padded
    (10_000, 4_000, 4, 3_072, 784),    # three whole windows a step, the last shorter
    (4_096, 1_024, 4, 1_024, 1_024),   # a window a step
    (4_096, 4_096, 1, 4_096, 4_096),   # exactly the maximum: one step
    (2_049, 2_048, 2, 2_048, 1),       # one token past: a last step of one token
]
REFUSED = [(10_000, 500), (800, 500), (1_025, 1_023)]


@pytest.mark.parametrize("length,maximum,steps,tokens,last", CASES)
def test_the_plan_is_the_largest_whole_windows_under_the_maximum(length, maximum, steps, tokens, last):
    (plan,) = _planned([(length, maximum)])
    assert plan == SegmentPlan(steps=steps, tokens=tokens, length=length) and plan.last == last
    bounds = plan.bounds()
    assert len(bounds) == steps and bounds[0][0] == 0 and bounds[-1][1] == length
    assert all(a_end == b_start for (_, a_end), (b_start, _) in zip(bounds, bounds[1:], strict=False))
    assert all(start % W == 0 for start, _ in bounds), "every step starts on a window boundary"
    assert all(end - start <= (maximum or length) for start, end in bounds)


def test_a_sequence_that_must_be_split_under_a_maximum_below_one_window_has_no_plan():
    assert [plan.steps for plan in _planned(REFUSED)] == [0] * len(REFUSED)


def test_every_plan_of_a_sweep_covers_the_call_in_steps_no_longer_than_the_maximum():
    calls = [(length, maximum) for length in (1, 31, 1_023, 1_024, 1_025, 5_000, 65_536, 100_001)
             for maximum in (None, 1, 1_023, 1_024, 1_025, 3_000, 65_536, 1 << 20)]
    for (length, maximum), plan in zip(calls, _planned(calls), strict=True):
        if plan.steps == 0:
            assert maximum is not None and maximum < W and length > maximum
            continue
        assert plan.bounds()[-1][1] == length and 0 < plan.last <= plan.tokens <= (maximum or length)
        assert plan.steps == 1 or plan.tokens % W == 0 and plan.tokens + W > maximum, "the most whole windows under the maximum"


def test_no_limit_is_one_step_and_asks_no_card():
    assert fsla.segment_plan(300, None) == SegmentPlan(steps=1, tokens=300, length=300)


@pytest.mark.parametrize("call", [lambda: fsla.segment_plan(0, 8), lambda: fsla.segment_plan(8, 0), lambda: fsla.segment_plan(8.0, 8),
                                  lambda: fsla.max_tokens(0), lambda: fsla.max_tokens(True), lambda: fsla.max_memory(-1)])
def test_a_length_a_maximum_and_a_limit_are_positive_exact_ints(call):
    with pytest.raises(ValueError, match="positive exact int"):
        call()


def test_the_token_handle_states_its_maximum_whatever_the_context():
    handle = fsla.max_tokens(5_000)
    assert handle(_context()) == 5_000 == handle(_context(widths=(16,)))


def test_the_layer_takes_a_handle_or_none():
    with pytest.raises(TypeError, match="segment handle"):
        fsla.FSLA(fsla.Routing([fsla.dense_routing(256), fsla.dense_routing(256)]), hidden_size=64, num_heads=2, segments=4096)


@pytest.mark.cuda
@_ON_A_CARD
def test_the_librarys_plan_on_the_card_is_the_cases_at_the_cards_window_and_its_refusal_names_the_window():
    from fsla.ops.window import window_of

    window = window_of()
    scale = lambda n: None if n is None else n * window // W  # noqa: E731
    for length, maximum, steps, tokens, _ in CASES:
        plan = fsla.segment_plan(scale(length), scale(maximum))
        assert plan == SegmentPlan(steps=steps, tokens=scale(tokens), length=scale(length))
    with pytest.raises(RuntimeError, match=rf"the window is {window} tokens") as refused:
        fsla.segment_plan(3 * window, window - 1)
    assert f"maximum of {window - 1}" in str(refused.value)


def _step(context: SegmentContext, windows: int, window: int = W) -> dict:
    """The planner's bytes of a step of ``windows`` windows of this shape: ``workspace`` and ``operands``."""
    text = "/".join([",".join(map(str, context.widths)), ",".join(map(str, context.masked)), ",".join(map(str, context.heads)),
                     str(context.value_width), str(windows), str(window)])
    run = subprocess.run([str(TOOL), "--step", text], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def _operands(context: SegmentContext, windows: int, window: int) -> int:
    """A step's projected inputs from the formats, written out: a row digit a byte, a scale an fp32, a mask word an int32
    of 32 tokens of a digit, a value channel a bf16 (the values and their scale on the write side)."""
    total = 0
    for side, (masked, heads) in enumerate(zip(context.masked, context.heads, strict=True)):
        token = sum(w + 4 for w in context.widths) + (context.value_width * 2 + 4 if side == 1 else 0)
        word = sum(4 * w for level, w in enumerate(context.widths) if masked >> level & 1)
        total += heads * windows * (token * window + word * (window // 32))
    return total


def _routing(sparse: tuple[str | None, str | None]) -> fsla.Routing:
    return fsla.Routing([fsla.dense_routing(256) if duty is None else fsla.split_routing(256, sparse_duty=duty) for duty in sparse])


@pytest.mark.parametrize("sparse", [("write", "read"), (None, "read"), (None, None)])
def test_a_steps_operand_bytes_are_the_bytes_of_the_tensors_a_producer_makes(sparse):
    """THE TIE TO WHAT IS ALLOCATED, on the host: PyTorch allocates a step's inputs from the producer's formats, so the
    planner's figure for them is checked against a real bundle's tensors -- rows, scales, masks, values -- and the context
    the layer hands a handle against that bundle (a mask is present exactly where the context says a side can hold one)."""
    from conftest import reference_produce

    torch.manual_seed(0)
    seen: list[SegmentContext] = []
    layer = fsla.FSLA(_routing(sparse), hidden_size=64, num_heads=2, d_v=64, segments=lambda context: seen.append(context)).eval()
    x = torch.randn(3, 2 * W, 64)
    assert layer._segment_plan(x).steps == 1
    (context,) = seen
    with torch.no_grad():
        routes, values, masks = reference_produce(layer, x)
    assert context.masked == tuple(sum(1 << level for level, mask in enumerate(side) if mask is not None)
                                   for side in (masks.read, masks.write))
    tensors = [*routes.amp_read, *routes.amp_write, *routes.scale_read, *routes.scale_write, values.v, values.scale, *masks.read, *masks.write]
    assert _step(context, 2)["operands"] == sum(t.numel() * t.element_size() for t in tensors if t is not None) == _operands(context, 2, W)


@pytest.mark.cuda
@_ON_A_CARD
def test_a_steps_workspace_on_the_card_is_the_size_its_allocation_is_made_by():
    from fsla.ops._ext import extension
    from fsla.ops.window import window_of
    from fsla.segmentation import step_bytes

    window, context = window_of(), _context(masked=(3, 1), heads=(4, 6))
    for tokens in (1, window, 3 * window + 1):
        workspace, operands = step_bytes(context, tokens)
        assert workspace == int(extension().mask_prepare_bytes(list(context.widths), list(context.masked), list(context.heads), tokens, 0))
        assert operands == _operands(context, -(-tokens // window), window)


@pytest.mark.cuda
@_ON_A_CARD
def test_the_memory_handle_allows_the_most_whole_windows_whose_bytes_fit_and_refuses_where_one_does_not():
    from fsla.ops.window import window_of
    from fsla.segmentation import step_bytes

    window, context = window_of(), _context()
    one = sum(step_bytes(context, window))
    assert sum(step_bytes(context, 3 * window)) == 3 * one and step_bytes(context, 1) == step_bytes(context, window)
    for windows in (1, 3, 7):
        for spare in (0, one - 1):
            assert fsla.max_memory(windows * one + spare)(context) == windows * window
    with pytest.raises(RuntimeError, match="no segmentation serves a memory limit") as refused:
        fsla.max_memory(one - 1)(context)
    assert f"needs {one} bytes" in str(refused.value)
    #: a level-side that can hold no mask has no leaf groups of its own and no mask words: both parts are smaller
    bare = step_bytes(_context(masked=(0, 0)), window)
    assert bare[0] < step_bytes(context, window)[0] and bare[1] < step_bytes(context, window)[1]
