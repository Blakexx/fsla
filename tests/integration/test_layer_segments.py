"""THE OP ON ONE SEGMENT, AND THE LAYER'S SEGMENTS.

`fsla_op` runs the prefill DAG over the segment it is handed, under the descriptor and the workspace its caller built, and continues
the state it is handed; the layer splits a sequence into segments by a declared memory budget, so its projections are bounded with the op.
Graded here: the op against the fp64 oracle per slot, a segmented call against the whole call (the op's own
readout to fp32 ULPs, the layer's through its bf16 output projection), and the budget's refusal by name while the prefill launch function
that derives the segment length is not built (docs/api.md section 4.2).
"""
from __future__ import annotations

import dataclasses

import pytest
import torch
from conftest import reference_produce

import fsla
from fsla.ops.window import window_of
from tests.oracle.fixtures import assert_slots_close, oracle_run
from tests.oracle.tolerances import PREFILL_READOUT

pytestmark = [pytest.mark.cuda,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="the prefill kernels are CUDA")]

WIDTHS = (256, 256)

#: OWED (`requires_kernel`, `tests/conftest.py`): a call of the layer needs the producer and the prefill, a call of the op the
#: prefill alone; each is owed while its kernel is not built
_LAYER_PREFILL = pytest.mark.requires_kernel("produce", "prefill")
_OP_PREFILL = pytest.mark.requires_kernel("prefill")


def _routing() -> fsla.Routing:
    return fsla.Routing([fsla.split_routing(WIDTHS[0], sparse_duty="write"),
                         fsla.split_routing(WIDTHS[1], sparse_duty="read")])


def _layer(hidden: int = 64, heads: int = 2, **kwargs) -> fsla.FSLA:
    torch.manual_seed(0)
    return fsla.FSLA(_routing(), hidden_size=hidden, num_heads=heads, d_v=64, **kwargs)


def _bundles(tokens: int, heads: int = 2):
    layer = _layer(heads=heads).cuda().eval()
    with torch.no_grad():
        routes, _values, masks = reference_produce(layer, torch.randn(1, tokens, 64, device="cuda"))
        values = fsla.Values.from_projection(torch.randn(1, tokens, heads, 64, device="cuda"))
    return routes, values, masks


def _descriptor(routes, values, *, stateful=False):
    """The prefill's descriptor of a bundle, by the producer's rule: owed while the prefill kernel is not built, so it is refused here, by name."""
    from fsla.ops.produce import prefill_descriptor

    return prefill_descriptor(routes.topology, d_v=values.d_v, decay=False, tokens=routes.tokens, batch_heads=routes.batch * routes.heads,
                              stateful=stateful, device=routes.device)


def _op(routes, values, masks, descriptor, **kwargs):
    """`fsla_op` over a call whose workspace its caller built (`fsla.build_workspace`)."""
    return fsla.fsla_op(routes, values, masks, descriptor, workspace=fsla.build_workspace(routes, values, masks, descriptor), **kwargs)


def _cut_masks(masks, span: slice):
    """One stretch of a call's masks: the token-packed words of the stretch, whole 32-token words (a window is)."""
    first, last = span.start // 32, -(-span.stop // 32)
    return fsla.Masks(masks.consumer, tuple(m[:, :, first:last] for m in masks.read),
                      tuple(m[:, :, first:last] for m in masks.write))


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


@_OP_PREFILL
def test_the_op_matches_the_fp64_oracle_per_slot():
    routes, values, masks = _bundles(2 * window_of())
    with torch.no_grad():
        y, state = _op(routes, values, masks, _descriptor(routes, values))
    assert state is None and tuple(y.shape) == (1, 2 * window_of(), 2, 64)
    _, write = routes.dequantized()
    ref = oracle_run(values.dequantized().double(), tuple(r.double() for r in routes.dequantized()[0]), tuple(w.double() for w in write), routes.topology)
    assert_slots_close(y.double(), ref.y, output=PREFILL_READOUT, envelope=ref.y_envelope, what="fsla_op readout")


@pytest.mark.parametrize("paging", [True, False])
@_OP_PREFILL
def test_segments_carried_through_a_state_are_the_whole_call(paging):
    """The window grid is global, so a segment boundary on a window boundary moves no pair: two segments through
    one state are the whole call to fp32 ULPs, in either backing."""
    routes, values, masks = _bundles(4 * window_of())
    with torch.no_grad():
        whole, _ = _op(routes, values, masks, _descriptor(routes, values))
        state = fsla.state(backing="paged" if paging else "dense")
        parts = []
        for span in (slice(0, window_of()), slice(window_of(), 4 * window_of())):
            cut_routes, cut_values, cut_masks = _cut(routes, span), _cut(values, span), _cut_masks(masks, span)
            y, state = _op(cut_routes, cut_values, cut_masks, _descriptor(cut_routes, cut_values, stateful=True),
                           state=state)
            parts.append(y)
    assert state.paged is paging
    torch.testing.assert_close(torch.cat(parts, dim=1), whole, rtol=0, atol=float(whole.abs().max()) * 2 ** -20)


@_LAYER_PREFILL
def test_a_budgeted_layer_segments_and_matches_the_whole_call():
    layer = _layer().cuda().eval()
    x = torch.randn(1, 4 * window_of(), 64, device="cuda")
    with torch.no_grad():
        whole, whole_cont = layer(x)
    assert whole_cont.state is None
    layer.memory_budget = 2 * window_of() * 4096
    with torch.no_grad():
        segmented, cont = layer(x)
    assert cont.state is None, "a stateless call hands no state back, however many segments it ran"
    #: the op's readouts agree to fp32 ULPs; the layer's output passes a bf16 projection, one ULP of it
    torch.testing.assert_close(segmented, whole, rtol=2 ** -7, atol=float(whole.abs().max()) * 2 ** -7)


@_LAYER_PREFILL
def test_a_stateful_segmented_call_leaves_the_whole_calls_state():
    layer = _layer().cuda().eval()
    x = torch.randn(1, 4 * window_of(), 64, device="cuda")
    with torch.no_grad():
        _, whole = layer(x, continuation=fsla.LayerContinuation(state=fsla.state()))
        layer.memory_budget = window_of() * 4096
        _, segmented = layer(x, continuation=fsla.LayerContinuation(state=fsla.state()))
    a, b = whole.state.materialize(), segmented.state.materialize()
    torch.testing.assert_close(b, a, rtol=0, atol=float(a.abs().max()) * 2 ** -20)


def test_a_budgeted_prefill_is_refused_by_name_before_anything_runs():
    """A budget is the LAYER's: it would ask the planner how many tokens fit beside the state and loop over segments, and the planner's answer
    is the prefill launch function's, which is not built. So a budgeted prefill is refused as `KernelNotBuilt` naming the prefill kernel at the
    one place the layer would have segmented -- before the producer, the descriptor or any kernel -- and never as a call outside the contract."""
    from fsla.ops._ext import KernelNotBuilt

    layer = _layer(memory_budget=10 ** 9).cuda().eval()
    with torch.no_grad(), pytest.raises(KernelNotBuilt, match=r"\bprefill kernel\b") as refused:
        layer(torch.randn(2, 3 * window_of(), 64, device="cuda"))
    assert "outside FSLA's contract" not in str(refused.value)
