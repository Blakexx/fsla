# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CONV RINGS ARE PART OF THE RECURRENCE — the gate that says so.

`value_conv` / `route_conv` are kept for campaign parity against FLA baselines, and
until now they had two contract lines and no functional test: nothing in the tree
carried a ring across a call boundary and checked the answer. That is exactly the
shape of defect that ships — a continuation that silently restarted its convolution
from zeros would compute a different function for the first `kernel - 1` tokens of
EVERY step, and every gate in the suite would still pass.

The convolution here is written in this file, in ten lines, so the gate depends on no
`fla` install and so the ring convention it asserts is visible rather than imported.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from conftest import build_routes
from einops import rearrange

import rola
from rola.ops.naive import naive_rola

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the arm is a CUDA kernel"),
]

_HIDDEN, _H, _DV, _WIDTHS = 128, 2, 64, (16, 16)
_KERNEL = 4


class CausalConv(nn.Module):
    """A depthwise causal convolution with an EXPLICIT ring, in the layer's shape.

    The contract the layer calls: ``conv(x, cache=ring, output_final_state=bool)``
    returns ``(y, ring)``, where the ring is the last ``kernel - 1`` tokens -- the
    only history a causal depthwise kernel needs to continue exactly.
    """

    def __init__(self, width: int, kernel: int = _KERNEL) -> None:
        super().__init__()
        self.kernel = kernel
        self.weight = nn.Parameter(torch.randn(kernel, width) / kernel)

    def forward(self, x, cache=None, output_final_state=False):
        B, T, D = x.shape
        history = (torch.zeros(B, self.kernel - 1, D, dtype=x.dtype, device=x.device)
                   if cache is None else cache)
        seq = torch.cat((history, x), dim=1)
        y = sum(seq[:, i:i + T] * self.weight[self.kernel - 1 - i] for i in range(self.kernel))
        ring = seq[:, seq.shape[1] - (self.kernel - 1):] if output_final_state else None
        return y, ring


def _layer(seed: int = 0):
    torch.manual_seed(seed)
    routes = build_routes(hidden_size=_HIDDEN, num_heads=_H, widths=_WIDTHS,
                          normalization="global")
    layer = rola.RoLA(routes, d_v=_DV, layer_idx=0,
                      value_conv=CausalConv(_H * _DV),
                      route_conv=CausalConv(_HIDDEN))
    return layer.cuda().float().eval()


def _oracle_last_token(layer, x):
    """`y[:, -1]` from the fp64 oracle over the WHOLE sequence, convolutions and all.

    An independent reference for the seam: the whole sequence is not a chunk multiple
    here, so no kernel arm could produce it even if one were asked."""
    with torch.no_grad():
        v, _ = layer.value_conv(layer.v_proj(x))
        source, _ = layer.route_conv(x)
        v = rearrange(v, "b l (h d) -> b l h d", d=_DV)
        routes = layer.routes(source)
        y, _ = naive_rola(v, routes.read_simplex(), routes.write, routes.g_read, routes.g_write,
                          layer.topology, None, "global")
        y = layer.o_proj(y.to(x.dtype).reshape(x.shape[0], x.shape[1], -1))
    return y[:, -1]


def test_a_conv_configured_continuation_is_the_whole_sequence():
    """THE GATE: full == chained halves, WITH convolutions configured.

    Both halves carry the pair `(state, conv_rings)`. A dropped ring would restart the
    convolution at the cut and move the second half's `v` and routing outright -- far
    outside any reassociation envelope -- which is what the control below measures.
    """
    layer = _layer(seed=3)
    T = 64
    x = torch.randn(2, T, _HIDDEN, device="cuda")
    half = T // 2

    with torch.no_grad():
        whole, _ = layer(x, state=(rola.state(), None))
        continuation = (rola.state(), None)
        y1, continuation = layer(x[:, :half], state=continuation)
        y2, continuation = layer(x[:, half:], state=continuation)
        #: the CONTROL: the same second half with the rings dropped, which is what a
        #: layer that carried only the op's state would compute.
        dropped, _ = layer(x[:, half:], state=(continuation[0].clone(), None))

    rings = continuation[1]
    assert rings is not None and len(rings) == 2 and all(r is not None for r in rings), (
        "a conv-configured layer returned no rings to continue with")
    assert rings[0].shape == (2, _KERNEL - 1, _H * _DV)

    rel1 = float((y1 - whole[:, :half]).norm() / whole[:, :half].norm())
    rel2 = float((y2 - whole[:, half:]).norm() / whole[:, half:].norm())
    assert rel1 < 1e-2, f"the first half moved: rel={rel1:.3e}"
    assert rel2 < 1e-2, f"the chained conv continuation left the band: rel={rel2:.3e}"
    control = float((dropped - whole[:, half:]).norm() / whole[:, half:].norm())
    assert control > 10 * max(rel2, 1e-12), (
        "dropping the rings did not move the answer, so this cell cannot see the "
        f"defect it exists for (kept={rel2:.3e}, dropped={control:.3e})")


def test_a_decode_step_carries_the_conv_ring_too():
    """The `T = 1` cell: the rings continue across the prefill/decode boundary.

    Graded against the oracle over all `T + 1` tokens -- the convolutions included --
    because the decode step's whole claim is that the token it produces is the token
    the whole sequence would have produced there.
    """
    layer = _layer(seed=5)
    T = 32
    x = torch.randn(2, T + 1, _HIDDEN, device="cuda")

    with torch.no_grad():
        continuation = (rola.state(), None)
        _y, continuation = layer(x[:, :T], state=continuation)
        prefill_rings = tuple(r.clone() for r in continuation[1])
        y_step, continuation = layer(x[:, T:], state=continuation)
    assert layer.last_execution_backend == "decode", "the step did not take decode"

    rings = continuation[1]
    assert not torch.equal(rings[0], prefill_rings[0]), (
        "the decode step did not advance the value ring")
    reference = _oracle_last_token(layer, x)
    rel = float((y_step[:, 0] - reference).norm() / reference.norm())
    assert rel < 1e-2, (
        f"the conv-configured decode step left the bf16 family band: rel={rel:.3e}")
