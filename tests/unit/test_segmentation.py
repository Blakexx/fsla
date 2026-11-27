# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""SEGMENTATION (docs/api.md section 4.2, docs/internals/common/plan/segments.md): the policy, the plan of a call over its
head blocks and token steps, the memory handle's policy, and the layer's loop over the plan.

The plan is the planner's (`csrc/fsla/src/common/plan/segments.h`): on the host `plan_tool --segments` prints it and
`plan_tool --policy` a step's policy under a byte limit, for a window given by hand, no GPU; on a card
`fsla.segment_plan` and `fsla.max_memory` ask the library, which reads the card. A policy states the most tokens and the
most write heads a step may hold; the plan takes the largest whole number of windows under the first, and whole
sequences, or a range of one sequence's write heads, under the second. The layer's loop is pinned on the host with a
stand-in for the op (the prefill's kernels are the card's): the producer's per-head weights sliced against a whole-heads
produce in fp64, the output assembled block by block, and each block's state a view that lands in the caller's plane.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import fsla  # noqa: E402
from fsla.segmentation import HeadBlock, SegmentContext, SegmentPlan, SegmentPolicy  # noqa: E402

TOOL = ROOT / "csrc" / "fsla" / "plan_tool"
W = 1024
_ON_A_CARD = pytest.mark.skipif(not torch.cuda.is_available(), reason="the library's planner reads the card")


def _planned(calls: list[tuple], window: int = W) -> list[SegmentPlan]:
    """The planner's plans of ``(length, max_tokens, max_heads, batch, heads)`` calls at ``window`` (a maximum ``None``: no
    limit); ``steps`` 0 where a call has none."""
    text = ";".join(f"{length},{tokens or 0},{most or 0},{batch},{heads},{window}" for length, tokens, most, batch, heads in calls)
    run = subprocess.run([str(TOOL), "--segments", text], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    out = [json.loads(line) for line in run.stdout.splitlines()]
    assert len(out) == len(calls)
    return [SegmentPlan(length=length, batch=batch, heads=heads, steps=o["steps"], tokens=o["tokens"], block_sequences=o["sequences"],
                        block_heads=o["heads"]) for o, (length, _, _, batch, heads) in zip(out, calls, strict=True)]


def _context(**over) -> SegmentContext:
    return SegmentContext(**{"widths": (256, 256), "masked": (3, 3), "batch": 2, "write_heads": 2, "query_heads": 2, "value_width": 64,
                             "device": torch.device("cuda", 0), **over})


# --- the token axis -------------------------------------------------------------

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
def test_the_plan_is_the_largest_whole_windows_under_the_token_maximum(length, maximum, steps, tokens, last):
    (plan,) = _planned([(length, maximum, None, 2, 4)])
    assert (plan.steps, plan.tokens, plan.last) == (steps, tokens, last)
    assert plan.blocks() == [HeadBlock(batch=(0, 2), heads=(0, 4))], "no head maximum: one block of every head"
    bounds = plan.bounds()
    assert len(bounds) == steps and bounds[0][0] == 0 and bounds[-1][1] == length
    assert all(a_end == b_start for (_, a_end), (b_start, _) in zip(bounds, bounds[1:], strict=False))
    assert all(start % W == 0 for start, _ in bounds), "every step starts on a window boundary"
    assert all(end - start <= (maximum or length) for start, end in bounds)


def test_a_sequence_that_must_be_split_under_a_maximum_below_one_window_has_no_plan_whatever_the_heads():
    calls = [(length, maximum, most, 2, 4) for length, maximum in REFUSED for most in (None, 1, 3)]
    assert [plan.steps for plan in _planned(calls)] == [0] * len(calls)


# --- the head axis --------------------------------------------------------------

HEAD_CASES = [
    # batch, heads, max_heads: the blocks' sequences and write heads
    (2, 4, None, [((0, 2), (0, 4))]),                                  # no limit: one block
    (2, 4, 8, [((0, 2), (0, 4))]),                                     # every write head fits: one block
    (3, 4, 9, [((0, 2), (0, 4)), ((2, 3), (0, 4))]),                   # whole sequences: two a block, the last the rest
    (3, 4, 4, [((0, 1), (0, 4)), ((1, 2), (0, 4)), ((2, 3), (0, 4))]),  # a sequence a block
    (2, 4, 3, [((0, 1), (0, 3)), ((0, 1), (3, 4)), ((1, 2), (0, 3)), ((1, 2), (3, 4))]),  # a range of one sequence's heads
    (1, 4, 1, [((0, 1), (h, h + 1)) for h in range(4)]),               # a write head a block
]


@pytest.mark.parametrize("batch,heads,most,blocks", HEAD_CASES)
def test_a_head_block_is_whole_sequences_or_a_range_of_one_sequences_write_heads(batch, heads, most, blocks):
    (plan,) = _planned([(300, None, most, batch, heads)])
    assert plan.steps == 1 and [(b.batch, b.heads) for b in plan.blocks()] == blocks


def test_every_plan_of_a_sweep_covers_each_write_head_once_in_contiguous_blocks_under_both_maxima():
    calls = [(length, tokens, most, batch, heads) for length in (1, 1_025, 5_000) for tokens in (None, 1_024, 3_000)
             for most in (None, 1, 2, 3, 5, 8, 13, 64) for batch in (1, 3) for heads in (1, 4, 6)]
    for (length, tokens, most, batch, heads), plan in zip(calls, _planned(calls), strict=True):
        covered = []
        for block in plan.blocks():
            (b0, b1), (h0, h1) = block.batch, block.heads
            assert b1 - b0 == 1 or (h0, h1) == (0, heads), "a block of several sequences holds all their write heads"
            assert (b1 - b0) * (h1 - h0) <= (most or batch * heads), "no block passes the head maximum"
            covered += [(b, h) for b in range(b0, b1) for h in range(h0, h1)]
        assert covered == [(b, h) for b in range(batch) for h in range(heads)], "each write head once, batch-major"
        if most is not None and most < batch * heads:
            whole = most // heads if most >= heads else 0
            assert (plan.block_sequences, plan.block_heads) == ((whole, heads) if whole else (1, most)), "the most a block"
        assert plan.bounds()[-1][1] == length and 0 < plan.last <= plan.tokens <= (tokens or length)


# --- the policy and the plan's Python face ----------------------------------------

def test_a_call_no_maximum_cuts_is_one_op_call_and_asks_no_card():
    for policy in (SegmentPolicy(), SegmentPolicy(max_tokens=500, max_heads=8), fsla.max_tokens(300)):
        assert fsla.segment_plan(300, policy, batch=2, heads=4) == SegmentPlan.whole(300, 2, 4)


@pytest.mark.parametrize("call", [lambda: SegmentPolicy(max_tokens=0), lambda: SegmentPolicy(max_heads=True),
                                  lambda: SegmentPolicy(max_heads=2.0), lambda: fsla.max_tokens(0), lambda: fsla.max_memory(-1),
                                  lambda: fsla.segment_plan(0, SegmentPolicy(), batch=1, heads=1),
                                  lambda: fsla.segment_plan(8, SegmentPolicy(), batch=0, heads=1),
                                  lambda: fsla.segment_plan(8, SegmentPolicy(), batch=1, heads=1.0)])
def test_a_maximum_a_length_a_batch_heads_and_a_limit_are_positive_exact_ints(call):
    with pytest.raises(ValueError, match="positive exact int"):
        call()


def test_the_policy_is_frozen_and_the_token_maker_is_a_policy():
    policy = fsla.max_tokens(5_000)
    assert policy == SegmentPolicy(max_tokens=5_000) and policy.max_heads is None and "SegmentPolicy" in fsla.__all__
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.max_tokens = 1  # type: ignore[misc]
    with pytest.raises(TypeError, match="SegmentPolicy"):
        fsla.segment_plan(8, 8, batch=1, heads=1)  # type: ignore[arg-type]


def _routing(sparse: tuple[str | None, str | None] = ("write", "read"), widths: tuple[int, int] = (256, 256)) -> fsla.Routing:
    return fsla.Routing([fsla.dense_routing(w) if duty is None else fsla.split_routing(w, sparse_duty=duty)
                         for duty, w in zip(sparse, widths, strict=True)])


def test_the_layer_takes_a_policy_a_handle_or_none_and_its_default_is_64_windows_of_every_head():
    assert fsla.FSLA(_routing(), hidden_size=64, num_heads=2).segments == SegmentPolicy(max_tokens=65_536)
    with pytest.raises(TypeError, match="SegmentPolicy"):
        fsla.FSLA(_routing(), hidden_size=64, num_heads=2, segments=4096)
    layer = fsla.FSLA(_routing(), hidden_size=64, num_heads=2, segments=lambda context: 4096)
    with pytest.raises(TypeError, match="returns a fsla.SegmentPolicy or None"):
        layer._segment_plan(torch.randn(1, 8, 64))


# --- a step's bytes and the memory handle's policy ----------------------------------

def _step_text(context: SegmentContext, a: int, window: int) -> str:
    heads = (context.batch * context.query_heads, context.batch * context.write_heads)
    return "/".join([",".join(map(str, context.widths)), ",".join(map(str, context.masked)), ",".join(map(str, heads)),
                     str(context.value_width), str(a), str(window)])


def _tool(flag: str, text: str) -> dict:
    run = subprocess.run([str(TOOL), flag, text], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def _step(context: SegmentContext, windows: int, window: int = W) -> dict:
    """The planner's bytes of a step of ``windows`` windows of every head of this shape: ``workspace`` and ``operands``."""
    return _tool("--step", _step_text(context, windows, window))


def _operands(context: SegmentContext, windows: int, window: int) -> int:
    """A step's projected inputs from the formats, written out: a row digit a byte, a scale an fp32, a mask word an int32
    of 32 tokens of a digit, a value channel a bf16 (the values and their scale on the write side)."""
    total = 0
    sides = (context.batch * context.query_heads, context.batch * context.write_heads)
    for side, (masked, heads) in enumerate(zip(context.masked, sides, strict=True)):
        token = sum(w + 4 for w in context.widths) + (context.value_width * 2 + 4 if side == 1 else 0)
        word = sum(4 * w for level, w in enumerate(context.widths) if masked >> level & 1)
        total += heads * windows * (token * window + word * (window // 32))
    return total


@pytest.mark.parametrize("text", ["256,256/3,1/0,6/64/1/1024", "256,256/3,1/6,6/0/1/1024", "250/1,1/6,6/64/1/1024",
                                  "2,2,2,2,2/0,0/1,1/64/1/1024", "256,256/3,1/4,6/64/1/1024"])
def test_a_shape_outside_the_ops_is_refused_by_the_one_function_the_library_calls(text):
    """`plan::step_shape` makes a step's shape for the tool and for the library's entries alike, so what the tool refuses the
    library refuses: no batch-head, no value channel, a width that is no power of two, more levels than the op has, query
    heads that are no whole number of write heads."""
    for flag in ("--step", "--policy"):
        run = subprocess.run([str(TOOL), flag, text], capture_output=True, text=True, timeout=60)
        assert run.returncode != 0 and "outside the op's" in run.stderr


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
    assert layer._segment_plan(x) == SegmentPlan.whole(2 * W, 3, 2)
    (context,) = seen
    assert (context.batch, context.write_heads, context.query_heads) == (3, 2, 2)
    with torch.no_grad():
        routes, values, masks = reference_produce(layer, x)
    assert context.masked == tuple(sum(1 << level for level, mask in enumerate(side) if mask is not None)
                                   for side in (masks.read, masks.write))
    tensors = [*routes.amp_read, *routes.amp_write, *routes.scale_read, *routes.scale_write, values.v, values.scale, *masks.read, *masks.write]
    assert _step(context, 2)["operands"] == sum(t.numel() * t.element_size() for t in tensors if t is not None) == _operands(context, 2, W)


#: a grouped shape, two query heads a write head, read side masked on both levels and the write side on the first
_GROUPED = {"masked": (3, 1), "batch": 2, "write_heads": 3, "query_heads": 6}


def test_a_steps_bytes_are_a_write_heads_bytes_a_write_head_and_the_policy_takes_tokens_first():
    """THE MEMORY POLICY on the host (`plan::step_policy_within`, the function `fsla_step_policy` calls): the most whole
    windows of every head where one window of every head fits; only where it does not, one window and the most write
    heads, each with every query head reading it; none where one window of one write head does not fit."""
    context = _context(**_GROUPED)
    one = sum(_step(context, 1).values())
    head = sum(_step(_context(**{**_GROUPED, "batch": 1, "write_heads": 1, "query_heads": 2}), 1).values())
    assert one == context.batch * context.write_heads * head
    policy = lambda limit: _tool("--policy", _step_text(context, limit, W))  # noqa: E731
    for windows in (1, 3, 7):
        for spare in (0, one - 1):
            assert policy(windows * one + spare) == {"tokens": windows * W, "heads": 0}
    for heads in (1, 4, 5):
        for spare in (0, head - 1):
            assert policy(heads * head + spare) == {"tokens": W, "heads": heads}
    assert policy(head - 1) == {"tokens": 0, "heads": 0}


# --- the layer's loop, on the host with a stand-in for the op --------------------------

_HOST_WIDTHS = (16, 32)


def _host_layer(heads: int = 4) -> fsla.FSLA:
    torch.manual_seed(0)
    return fsla.FSLA(_routing(widths=_HOST_WIDTHS), hidden_size=32, num_heads=heads, d_v=64).eval()


def test_a_blocks_weights_produce_exactly_the_whole_produces_rows_of_its_heads_in_fp64():
    """THE WEIGHTS' SLICING: a head block's producer call, on its sequences with its heads' rows of ``route_W``,
    ``route_bias``, ``gain_bias`` and ``v_W``, is the whole-heads produce's block, routes, values and masks alike (the
    reference producer, in fp64)."""
    from fsla.ops.produce import _composition
    from fsla.routing.masks import PREFILL

    layer = fsla.FSLA(fsla.Routing([fsla.split_routing(16, sparse_duty="write"), fsla.split_routing(32, sparse_duty="read")],
                                   bias=True), hidden_size=32, num_heads=4, d_v=64).double().eval()
    with torch.no_grad():
        layer.route_bias.normal_()
        layer.gain_bias.normal_()
    x = torch.randn(3, 70, 32, dtype=torch.float64)
    with torch.no_grad():
        weights = layer._weights(x)
        whole = _composition(x, layer.routing, weights, PREFILL)
        for block in (HeadBlock(batch=(1, 2), heads=(1, 3)), HeadBlock(batch=(0, 1), heads=(3, 4)), HeadBlock(batch=(1, 3), heads=(0, 4))):
            b, h = slice(*block.batch), slice(*block.heads)
            part = _composition(x[b], layer.routing, layer._block_weights(weights, block), PREFILL)
            for (mine, axis), (theirs, _) in zip(_leaves(part), _leaves(whole), strict=True):
                assert (mine is None) == (theirs is None)
                if mine is not None:
                    torch.testing.assert_close(mine, theirs[b].narrow(axis, h.start, h.stop - h.start), rtol=0, atol=0)


def _leaves(bundle) -> list[tuple[torch.Tensor | None, int]]:
    """Every tensor of a ``(routes, values, masks)`` bundle with its head axis: the rows, scales and values are
    ``[B, T, H, ...]``, the masks ``[B, H, words, width]``."""
    routes, values, masks = bundle
    tokens = [*routes.amp_read, *routes.amp_write, *routes.scale_read, *routes.scale_write, values.v, values.scale]
    return [(t, 2) for t in tokens] + [(m, 1) for m in (*masks.read, *masks.write)]


def _stand_in(calls: list):
    """A stand-in for the op on the host: it records each call's sequences, heads and tokens, returns the values as the
    readout, and for a state binds it (or checks it) through the seam and adds one to every byte of the plane it runs on."""
    from fsla.engine.facts.planes import state_plane

    def prefill(routes, values, masks, density, *, state=None, decay=None):
        calls.append((routes.batch, routes.heads, routes.tokens))
        y = values.dequantized().float()
        if state is None:
            return y, None
        plane = state._kernel_entry(routes, d_v=values.d_v)
        if plane is None:
            plane = state_plane(routes, d_v=values.d_v, zeros=True)
        plane += 1
        state._kernel_commit(plane)
        return y, state

    return prefill


_HOST_PLANS = [
    # length, max_tokens, max_heads, batch, window: the plan the planner makes, at a window small enough for the host
    (200, 128, 3, 2, 64),     # four blocks of a range of one sequence's heads, two token steps each
    (200, None, 8, 3, 64),    # two blocks of whole sequences, one step each
    (200, 64, None, 2, 64),   # one block, four token steps
]


@pytest.mark.parametrize("length,tokens,most,batch,window", _HOST_PLANS)
def test_the_layers_loop_runs_each_block_and_step_and_assembles_the_whole_calls_output(monkeypatch, length, tokens, most, batch, window):
    calls: list = []
    monkeypatch.setattr("fsla.layer.prefill", _stand_in(calls))
    layer = _host_layer()
    x = torch.randn(batch, length, 32)
    with torch.no_grad():
        whole, _ = layer._prefill(x, None, None)
        (plan,) = _planned([(length, tokens, most, batch, layer.num_heads)], window)
        monkeypatch.setattr(layer, "_segment_plan", lambda _x: plan)
        calls.clear()
        cut, state = layer._prefill(x, None, None)
    assert state is None, "a stateless call hands no state back, however it is cut"
    assert calls == [(b1 - b0, h1 - h0, end - start) for block in plan.blocks() for (b0, b1), (h0, h1) in [(block.batch, block.heads)]
                     for start, end in plan.bounds()], "each head block, then each token range in it"
    torch.testing.assert_close(cut, whole, rtol=2 ** -8, atol=0)


@pytest.mark.parametrize("length,tokens,most,batch,window", _HOST_PLANS)
@pytest.mark.parametrize("carrying", [False, True])
def test_a_stateful_loop_lands_every_blocks_steps_in_the_callers_plane_in_place(monkeypatch, length, tokens, most, batch, window, carrying):
    """The caller's state is bound to the whole call (a zeroed plane where it carries nothing) and each block runs on a view
    of its rows: every row of the caller's plane is advanced once a token step, and no row twice."""
    calls: list = []
    monkeypatch.setattr("fsla.layer.prefill", _stand_in(calls))
    layer = _host_layer()
    x = torch.randn(batch, length, 32)
    state = fsla.dense_state()
    if carrying:
        state._bind_plane(batch=batch, heads=layer.num_heads, widths=_HOST_WIDTHS, d_v=layer.d_v, device=x.device)
        state._plane.fill_(5)
    plane = state._plane
    (plan,) = _planned([(length, tokens, most, batch, layer.num_heads)], window)
    monkeypatch.setattr(layer, "_segment_plan", lambda _x: plan)
    with torch.no_grad():
        _, out_state = layer._prefill(x, state, None)
    assert out_state is state and state.shape == (batch, layer.num_heads, _HOST_WIDTHS[0] * _HOST_WIDTHS[1], layer.d_v + 1)
    assert plane is None or state._plane is plane, "a carried plane is advanced in place, never replaced"
    assert bool((state._plane == (5 if carrying else 0) + plan.steps).all())


# --- on a card --------------------------------------------------------------------

@pytest.mark.cuda
@_ON_A_CARD
def test_the_librarys_plan_on_the_card_is_the_cases_at_the_cards_window_and_its_refusal_names_the_window():
    from fsla.ops.window import window_of

    window = window_of()
    scale = lambda n: None if n is None else n * window // W  # noqa: E731
    for length, maximum, steps, tokens, _ in CASES:
        plan = fsla.segment_plan(scale(length), SegmentPolicy(max_tokens=scale(maximum), max_heads=3), batch=2, heads=4)
        assert (plan.steps, plan.tokens, plan.block_sequences, plan.block_heads) == (steps, scale(tokens), 1, 3)
    for batch, heads, most, blocks in HEAD_CASES:
        plan = fsla.segment_plan(300, SegmentPolicy(max_heads=most), batch=batch, heads=heads)
        assert [(b.batch, b.heads) for b in plan.blocks()] == blocks
    with pytest.raises(RuntimeError, match=rf"the window is {window} tokens") as refused:
        fsla.segment_plan(3 * window, SegmentPolicy(max_tokens=window - 1), batch=1, heads=1)
    assert f"maximum of {window - 1}" in str(refused.value)


@pytest.mark.cuda
@_ON_A_CARD
def test_a_steps_workspace_on_the_card_is_the_size_its_allocation_is_made_by():
    from fsla.ops._ext import extension
    from fsla.ops.window import window_of
    from fsla.segmentation import _shape, step_bytes

    window, context = window_of(), _context(**_GROUPED)
    widths, masked, heads, _ = _shape(context)
    for tokens in (1, window, 3 * window + 1):
        workspace, operands = step_bytes(context, tokens)
        assert workspace == int(extension().mask_prepare_bytes(widths, masked, heads, tokens, 0))
        assert operands == _operands(context, -(-tokens // window), window)


@pytest.mark.cuda
@_ON_A_CARD
def test_the_memory_handle_takes_whole_windows_of_every_head_first_then_write_heads_and_refuses_where_one_does_not_fit():
    from fsla.ops.window import window_of
    from fsla.segmentation import step_bytes

    window, context = window_of(), _context(**_GROUPED)
    one = sum(step_bytes(context, window))
    head = sum(step_bytes(_context(**{**_GROUPED, "batch": 1, "write_heads": 1, "query_heads": 2}), window))
    assert one == 6 * head and sum(step_bytes(context, 3 * window)) == 3 * one and step_bytes(context, 1) == step_bytes(context, window)
    for windows in (1, 3, 7):
        for spare in (0, one - 1):
            assert fsla.max_memory(windows * one + spare)(context) == SegmentPolicy(max_tokens=windows * window)
    for heads in (1, 4, 5):
        assert fsla.max_memory(heads * head + head - 1)(context) == SegmentPolicy(max_tokens=window, max_heads=heads)
    with pytest.raises(RuntimeError, match="no segmentation serves a memory limit") as refused:
        fsla.max_memory(head - 1)(context)
    assert f"needs {head} bytes" in str(refused.value) and "one write head" in str(refused.value)
    #: a level-side that can hold no mask has no leaf groups of its own and no mask words: both parts are smaller
    bare = step_bytes(_context(masked=(0, 0)), window)
    assert bare[0] < step_bytes(_context(), window)[0] and bare[1] < step_bytes(_context(), window)[1]
