# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The facade's stateful half: the SURFACE that survives the K31 deletion.

Until the K31 batch (baseline = tag `baseline/pre-k31`) this file walked the
whole stateful surface through the prefill op -- bind-then-continue, chained
halves vs one whole call, clone independence, dense-vs-paged agreement, the BC
flip over a live arena. Those claims are claims about the PREFILL BINDING PATH,
which is deleted, and they return with it; what remains gated
here is the state object's own contract -- construction refusals, the engine's
state-type refusal, and the decode continuation's carrying requirement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import alternating_routing, build_layer, reference_produce  # noqa: E402

import fsla  # noqa: E402

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the decode arm is a CUDA kernel"),
]

_HIDDEN, _H, _DV, _WIDTHS, _T = 64, 2, 64, (16, 16), 128


def _producer(seed=0, widths=_WIDTHS):
    """The routes of a stretch of tokens, from a layer's producer."""
    torch.manual_seed(seed)
    layer = build_layer(hidden_size=_HIDDEN, num_heads=_H, widths=widths, d_v=_DV,
                        routing=alternating_routing(len(widths))).cuda().eval()
    return lambda x: reference_produce(layer, x)[0]


def _inputs(seed=0, T=_T, B=2):
    torch.manual_seed(seed + 1000)
    return (torch.randn(B, T, _HIDDEN, device="cuda"),
            torch.randn(B, T, _H, _DV, device="cuda"))


def test_a_raw_tensor_is_not_a_state():
    """The engine's fast-fail refuses a raw tensor where a FSLAState belongs.

    Gated at `validate_context` (the seam the rebuilt kernel re-fronts): `fsla_op` itself is
    the deletion refusal and speaks before it reads its arguments.
    """
    from conftest import hand_descriptor

    from fsla.engine.prefill_call import PrefillContext, validate_context
    from fsla.routing.masks import PREFILL
    from fsla.values import Values
    from tests.oracle.fixtures import masks_of

    producer, (x, v) = _producer(), _inputs()
    routes = producer(x)
    values = Values.from_projection(v)
    with torch.no_grad(), pytest.raises(TypeError, match="fsla.FSLAState"):
        validate_context(PrefillContext(routes=routes, values=values, masks=masks_of(routes, PREFILL),
                                        descriptor=hand_descriptor(routes, values, stateful=True),
                                        state=torch.zeros(1), want_state=True))


def test_an_unbuilt_strategy_is_refused_by_name():
    with pytest.raises(NotImplementedError, match="cow"):
        fsla.state("cow")
    with pytest.raises(ValueError, match="strategy must be"):
        fsla.state("mutable-ish")


def test_a_fresh_state_carries_nothing_and_decode_refuses_it():
    """`carries` is contents, not binding, and a decode step CONTINUES: with the
    prefill arm deleted there is nothing in production to charge a fresh state,
    and the decode entry's refusal must still say so by name."""
    state = fsla.state()
    assert not state.bound and not state.carries
    with pytest.raises(RuntimeError, match="carries nothing"):
        state._require_carrying()
