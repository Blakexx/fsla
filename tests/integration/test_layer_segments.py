"""THE OP ON ONE SEGMENT, AND THE LAYER'S SEGMENTS.

`rola_op` runs the prefill DAG over the segment it is handed and continues the state it is handed; the layer
splits a sequence into segments by a declared memory budget, so its projections are bounded with the op.
Graded here: the op against the fp64 oracle per slot, a segmented call against the whole call (the op's own
readout to fp32 ULPs, the layer's through its bf16 output projection), the budget's derivation and refusal, and
the per-token working-set bound against the measured peak (docs/api.md section 4.2).
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

import rola
from rola.ops.carry import WINDOW
from tests.oracle.fixtures import assert_slots_close, oracle_run
from tests.oracle.tolerances import PREFILL_READOUT

pytestmark = [pytest.mark.cuda,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="the prefill kernels are CUDA")]

WIDTHS = (256, 256)

#: RED BY PLAN (ruled 2026-09-29: no temporary patches): the unported carry reads and writes its own split pages and
#: refuses the int8 state format by name, so a STATEFUL prefill cannot run until the carry's int8 port.
_STATEFUL_PREFILL = pytest.mark.xfail(strict=True, raises=Exception,
                                      reason="the carry refuses the int8 state format until its port")


def _producer(hidden: int = 64, heads: int = 2) -> rola.RouteProducer:
    torch.manual_seed(0)
    return rola.RouteProducer([rola.split_routing(WIDTHS[0], sparse_duty="write"),
                               rola.split_routing(WIDTHS[1], sparse_duty="read")],
                              hidden_size=hidden, num_heads=heads)


def _bundles(tokens: int, heads: int = 2):
    producer = _producer(heads=heads).cuda()
    with torch.no_grad():
        routes = producer(torch.randn(1, tokens, 64, device="cuda"))
        values = rola.Values.from_projection(torch.randn(1, tokens, heads, 64, device="cuda"))
    return routes, values


def _cut(bundle, span: slice):
    """One stretch of tokens of a bundle: every ``[B, T, ...]`` field sliced on its token axis."""
    kw = {}
    for field in dataclasses.fields(bundle):
        value = getattr(bundle, field.name)
        if isinstance(value, torch.Tensor):
            value = value[:, span]
        elif isinstance(value, tuple) and value and isinstance(value[0], torch.Tensor):
            value = tuple(t[:, span] for t in value)
        kw[field.name] = value
    return type(bundle)(**kw)


def test_the_op_matches_the_fp64_oracle_per_slot():
    routes, values = _bundles(2 * WINDOW)
    with torch.no_grad():
        y, state = rola.rola_op(routes, values)
    assert state is None and tuple(y.shape) == (1, 2 * WINDOW, 2, 64)
    _, write = routes.dequantized()
    ref = oracle_run(values.dequantized().double(), tuple(r.double() for r in routes.read_simplex()),
                     tuple(w.double() for w in write), routes.g_write.double(), routes.topology)
    assert_slots_close(y.double(), ref.y, output=PREFILL_READOUT, envelope=ref.y_envelope, what="rola_op readout")


@pytest.mark.parametrize("paging", [True, False])
@_STATEFUL_PREFILL
def test_segments_carried_through_a_state_are_the_whole_call(paging):
    """The window grid is global, so a segment boundary on a window boundary moves no pair: two segments through
    one state are the whole call to fp32 ULPs, in either backing."""
    from rola.engine import PlanOverrides

    routes, values = _bundles(4 * WINDOW)
    expert = None if paging else PlanOverrides(paging=False)
    with torch.no_grad():
        whole, _ = rola.rola_op(routes, values)
        state = rola.state()
        parts = []
        for span in (slice(0, WINDOW), slice(WINDOW, 4 * WINDOW)):
            y, state = rola.rola_op(_cut(routes, span), _cut(values, span), state=state, expert=expert)
            parts.append(y)
    assert state.paged is paging
    torch.testing.assert_close(torch.cat(parts, dim=1), whole, rtol=0, atol=float(whole.abs().max()) * 2 ** -20)


@_STATEFUL_PREFILL
def test_a_budgeted_layer_segments_and_matches_the_whole_call():
    layer = rola.RoLA(_producer(), d_v=64).cuda().eval()
    x = torch.randn(1, 4 * WINDOW, 64, device="cuda")
    with torch.no_grad():
        whole, whole_cont = layer(x)
    assert whole_cont.state is None
    layer.memory_budget = layer.state_bytes(1) + 2 * WINDOW * layer.segment_bytes_per_token()
    assert layer.segment_tokens(1, 4 * WINDOW) == 2 * WINDOW
    with torch.no_grad():
        segmented, cont = layer(x)
    assert cont.state is None, "a stateless call hands no state back, however many segments it ran"
    #: the op's readouts agree to fp32 ULPs; the layer's output passes a bf16 projection, one ULP of it
    torch.testing.assert_close(segmented, whole, rtol=2 ** -7, atol=float(whole.abs().max()) * 2 ** -7)


@_STATEFUL_PREFILL
def test_a_stateful_segmented_call_leaves_the_whole_calls_state():
    layer = rola.RoLA(_producer(), d_v=64).cuda().eval()
    x = torch.randn(1, 4 * WINDOW, 64, device="cuda")
    with torch.no_grad():
        _, whole = layer(x, continuation=rola.LayerContinuation(state=rola.state()))
        layer.memory_budget = layer.state_bytes(1) + WINDOW * layer.segment_bytes_per_token()
        _, segmented = layer(x, continuation=rola.LayerContinuation(state=rola.state()))
    a, b = whole.state.materialize(), segmented.state.materialize()
    torch.testing.assert_close(b, a, rtol=0, atol=float(a.abs().max()) * 2 ** -20)


def test_the_budget_derives_the_segment_and_refuses_below_one_window():
    layer = rola.RoLA(_producer(), d_v=64)
    assert layer.segment_tokens(2, 3 * WINDOW) == 3 * WINDOW, "no budget: one segment"
    per_window = WINDOW * 2 * layer.segment_bytes_per_token()
    layer.memory_budget = layer.state_bytes(2) + 2 * per_window + per_window // 2
    assert layer.segment_tokens(2, 8 * WINDOW) == 2 * WINDOW
    assert layer.segment_tokens(2, WINDOW) == WINDOW, "a segment never exceeds the call"
    layer.memory_budget = layer.state_bytes(2) + per_window - 1
    with pytest.raises(ValueError, match="cannot hold one"):
        layer.segment_tokens(2, 8 * WINDOW)
    with pytest.raises(ValueError, match="memory_budget"):
        rola.RoLA(_producer(), memory_budget=0)


@pytest.mark.parametrize("heads,hidden", [(2, 64), (4, 256)])
def test_the_working_set_bound_holds_against_the_measured_peak(heads, hidden):
    """`segment_bytes_per_token` is an UPPER BOUND on the layer's working set: the measured peak's slope over the
    sequence length must sit under it, and within 1.5x of it (a bound that loose would starve the budget)."""
    producer = rola.RouteProducer([rola.split_routing(WIDTHS[0], sparse_duty="write"),
                                   rola.split_routing(WIDTHS[1], sparse_duty="read")],
                                  hidden_size=hidden, num_heads=heads)
    layer = rola.RoLA(producer, d_v=64).cuda().eval()
    peaks = {}
    for tokens in (WINDOW, 8 * WINDOW):
        x = torch.randn(1, tokens, hidden, device="cuda")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        with torch.no_grad():
            out, _ = layer(x)
        torch.cuda.synchronize()
        peaks[tokens] = torch.cuda.max_memory_allocated() - base
        del out
    measured = (peaks[8 * WINDOW] - peaks[WINDOW]) / (7 * WINDOW)
    bound = layer.segment_bytes_per_token()
    assert measured <= bound <= 1.5 * measured, f"measured {measured:.0f} B/token against the bound {bound}"
