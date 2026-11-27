"""THE DECODE PHASE LEDGER: the instrument the int8 port's budget line is measured with books every warp of every
launched CTA into its four phases (fold, factor, walk, combine), and binding it moves no number (docs/internals/decode/decode.md#phase-ledger)."""
from __future__ import annotations

import pytest
import torch
from conftest import reference_produce

import fsla
from fsla.ops import decode as decode_ops
from tests.oracle.fixtures import assert_reordered_close, state_pages

pytestmark = [pytest.mark.cuda,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="the decode kernel is CUDA")]


def _carrying_state(layer, x) -> fsla.FSLAState:
    """A dense state bound by the layer's own routing and holding a drawn entry state, so a one-token call is a
    decode step without a prefill."""
    with torch.no_grad():
        routes, _values, _masks = reference_produce(layer, x)
        state = fsla.dense_state()
        state._kernel_entry(routes, d_v=layer.d_v)
        entry = 0.1 * torch.randn(routes.batch * routes.heads, layer.topology.N, layer.d_v + 1, device="cuda")
        entry[..., layer.d_v] = entry[..., layer.d_v].abs()
        state._kernel_commit(state_pages(entry))
    return state


@pytest.mark.requires_kernel("produce")
def test_a_bound_ledger_books_every_phase_and_changes_nothing():
    torch.manual_seed(0)
    routing = fsla.Routing(topology=(64, 64), write=(fsla.entmax(1.5), fsla.softmax()),
                           read=((fsla.softmax(), fsla.entmax(1.5)),))
    layer = fsla.FSLA(routing, hidden_size=64, num_heads=2, d_v=64).cuda().eval()
    x = torch.randn(1, 1, 64, device="cuda")
    plain_state = _carrying_state(layer, x)
    booked_state = plain_state.clone()
    with torch.no_grad():
        y_plain, _ = layer(x, continuation=fsla.LayerContinuation(state=plain_state))
        ledger = torch.zeros(2 * 1024, decode_ops.ledger_warps(), len(decode_ops.LEDGER_PHASES),
                             dtype=torch.int64, device="cuda")
        decode_ops.ledger_bind(ledger)
        try:
            y_booked, _ = layer(x, continuation=fsla.LayerContinuation(state=booked_state))
        finally:
            decode_ops.ledger_bind(None)
    torch.cuda.synchronize()
    assert_reordered_close(y_booked, y_plain, "binding the ledger")
    live = ledger.sum((1, 2)) > 0
    assert int(live.sum()) >= 2, "no CTA booked"
    booked = ledger[live]
    phases = dict(zip(decode_ops.LEDGER_PHASES, booked.sum((0, 1)).tolist()))
    assert all(phases[name] > 0 for name in decode_ops.LEDGER_PHASES), phases
    assert bool((booked.sum(2) > 0).all()), "a warp of a launched CTA booked nothing"
