"""THE DECODE PHASE LEDGER: the instrument the int8 port's budget line is measured with books every warp of every
launched CTA into its five phases, and an unbound ledger costs a step nothing it can observe
(docs/internals/decode/decode.md#phase-ledger)."""
from __future__ import annotations

import pytest
import torch

import rola
from rola.ops import decode as decode_ops

pytestmark = [pytest.mark.cuda,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="the decode kernel is CUDA")]


def test_a_bound_ledger_books_every_phase_and_an_unbound_one_changes_nothing():
    torch.manual_seed(0)
    producer = rola.RouteProducer([rola.split_routing(256, sparse_duty="write"),
                                   rola.split_routing(256, sparse_duty="read")], hidden_size=64, num_heads=2)
    layer = rola.RoLA(producer, d_v=64).cuda().eval()
    x = torch.randn(1, 513, 64, device="cuda")
    with torch.no_grad():
        _, cont = layer(x[:, :512], continuation=rola.LayerContinuation(state=rola.state()))
        reference = cont.state.clone()
        y_plain, _ = layer(x[:, 512:], continuation=rola.LayerContinuation(state=reference))
        ledger = torch.zeros(2 * 1024, decode_ops.LEDGER_WARPS, len(decode_ops.LEDGER_PHASES),
                             dtype=torch.int64, device="cuda")
        decode_ops.ledger_bind(ledger)
        try:
            y_booked, _ = layer(x[:, 512:], continuation=cont)
        finally:
            decode_ops.ledger_bind(None)
    torch.cuda.synchronize()
    assert torch.equal(y_booked, y_plain), "binding the ledger moved a number"
    live = ledger.sum((1, 2)) > 0
    assert int(live.sum()) >= 2, "no CTA booked"
    booked = ledger[live]
    phases = dict(zip(decode_ops.LEDGER_PHASES, booked.sum((0, 1)).tolist()))
    assert all(phases[name] > 0 for name in ("fold", "factor", "admit", "walk", "combine")), phases
    assert bool((booked.sum(2) > 0).all()), "a warp of a launched CTA booked nothing"
