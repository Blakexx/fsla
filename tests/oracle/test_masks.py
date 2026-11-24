# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE MASKS ARE DATA (ruled 2026-11-22): the layouts as the consumers read them, the producer's output in each, and the
oracle computing ``r * m`` -- planted nonzero rows under zero bits change nothing but the bits, and a masked
coefficient's gradient is zero. No device: the oracle is fp64 torch and the producer's composition is the host's.

Symbols: ``B`` batch, ``T`` tokens, ``H`` heads, ``w`` a level's width, ``r`` a coefficient row, ``m`` its bit.
"""
from __future__ import annotations

import pytest
import torch
from conftest import build_layer

from fsla.ops.naive import naive_fsla
from fsla.ops.produce import activation_walks, produce_decode, produce_prefill
from fsla.routing.masks import DECODE, PREFILL, Masks, pack_digits, pack_tokens, unpack_digits, unpack_tokens
from fsla.routing.types import EntmaxActivation, IndependentRouting, SoftmaxActivation, TiedRouting, Topology

_TOPOLOGY = Topology(levels=tuple(
    IndependentRouting(width=w, read=SoftmaxActivation(), write=SoftmaxActivation()) for w in (4, 6)))


# --------------------------------------------------------------- the layouts, bit by bit


def test_the_prefill_layout_is_token_packed_bit_t_mod_32_of_word_t_div_32():
    """Verified against the definition, not against the packer's inverse: ``[B, H, ceil(T / 32), width]``, token ``t`` at
    bit ``t % 32`` of word ``t // 32``, the digit the last axis."""
    T, H, W = 70, 3, 5
    live = torch.zeros(2, T, H, W, dtype=torch.bool)
    live[1, 33, 2, 4] = True
    live[0, 0, 0, 0] = True
    live[0, 31, 1, 2] = True
    words = pack_tokens(live)
    assert words.dtype is torch.int32 and tuple(words.shape) == (2, H, 3, W)
    assert int(words[1, 2, 1, 4]) == 1 << 1
    assert int(words[0, 0, 0, 0]) == 1
    assert int(words[0, 1, 0, 2]) == -(1 << 31), "bit 31 is the sign bit of the int32 word"
    assert int((words != 0).sum()) == 3
    assert torch.equal(unpack_tokens(words, T), live)


def test_the_decode_layout_is_digit_packed_bit_d_mod_32_of_word_d_div_32():
    W, H = 70, 3
    live = torch.zeros(2, 1, H, W, dtype=torch.bool)
    live[1, 0, 2, 33] = True
    live[0, 0, 0, 0] = True
    live[0, 0, 1, 31] = True
    words = pack_digits(live)
    assert words.dtype is torch.int32 and tuple(words.shape) == (2, H, 3)
    assert int(words[1, 2, 1]) == 1 << 1
    assert int(words[0, 0, 0]) == 1
    assert int(words[0, 1, 0]) == -(1 << 31)
    assert torch.equal(unpack_digits(words, W), live)


def test_the_solve_stream_view_is_free():
    """The entmax kernels' stream axis is ``B * H`` with ``H`` inner, so their ``[B * H, ceil(T / 32), cols]`` words are
    the public ``[B, H, ceil(T / 32), cols]`` as a view."""
    B, H, T, W = 2, 3, 40, 4
    live = torch.rand(B, T, H, W) < 0.5
    words = pack_tokens(live)
    stream = words.reshape(B * H, words.shape[2], W)
    for b in range(B):
        for h in range(H):
            assert torch.equal(stream[b * H + h], words[b, h])
    assert stream.data_ptr() == words.data_ptr()


# --------------------------------------------------------------- the producer emits the consumer's layout


def _layer_and_x(T, routing=None):
    layer = build_layer(hidden_size=32, num_heads=2, d_v=32, widths=(16, 16), routing=routing)
    return layer, torch.randn(2, T, 32, dtype=torch.float64)


@pytest.mark.parametrize("consumer", (PREFILL, DECODE))
def test_each_producer_returns_the_masks_in_its_consumers_layout(consumer):
    """The softmax layer's prefill masks are absent on every level-side (dense); its decode masks are made for every one."""
    T = 1 if consumer == DECODE else 70
    layer, x = _layer_and_x(T)
    produce = produce_decode if consumer == DECODE else produce_prefill
    routes, _values, masks, *_ = produce(x, layer.routing, layer._weights(x))
    assert masks.consumer == consumer
    masks.check(routes)
    assert (masks.tensors() == ()) == (consumer == PREFILL)
    for side in ("read", "write"):
        for level in range(routes.topology.D):
            rows = routes.rows(level, side)
            bits = masks.live(routes, side, level)
            assert bool((bits >= (rows != 0)).all()), "a mask covers every nonzero row digit"


def test_the_producers_read_their_geometry_off_the_arguments_and_refuse_by_name():
    """`num_heads` is `route_W`'s leading extent, `hidden` is `x`'s last and `route_W`'s middle, `d_v` is `v_W`'s rows over the heads."""
    layer, x = _layer_and_x(4)
    weights = layer._weights(x)
    routes, values, *_ = produce_prefill(x, layer.routing, weights)
    assert (routes.heads, values.d_v) == (2, 32)
    with pytest.raises(ValueError, match=r"x must be \[B, T, hidden\] with hidden = route_W's 32"):
        produce_prefill(x[..., :16], layer.routing, weights)
    with pytest.raises(ValueError, match=r"v_W must be \[H \* d_v, hidden\]"):
        produce_prefill(x, layer.routing, type(weights)(weights.route_W, weights.route_bias, weights.gain_bias, weights.v_W[:3]))
    with pytest.raises(ValueError, match=r"route_W must be \[H, hidden, columns\]"):
        produce_decode(x[:, :1], layer.routing, type(weights)(weights.route_W[0], weights.route_bias, weights.gain_bias, weights.v_W))


def test_masks_are_validated_for_shape_and_dtype_and_never_for_values():
    layer, x = _layer_and_x(40, routing=TiedRouting(width=16, op=EntmaxActivation(1.5)))
    routes, _values, masks, _density = layer._produce(x)
    everything_dead = Masks(PREFILL, tuple(torch.zeros_like(m) for m in masks.read), masks.write)
    everything_dead.check(routes)
    short = Masks(PREFILL, (masks.read[0][:, :, :1], *masks.read[1:]), masks.write)
    with pytest.raises(ValueError, match=r"Masks.read\[0\] \(prefill layout\)"):
        short.check(routes)
    wrong_dtype = Masks(PREFILL, (masks.read[0].to(torch.int64), *masks.read[1:]), masks.write)
    with pytest.raises(ValueError, match="int32"):
        wrong_dtype.check(routes)


def test_the_walk_rule_reads_the_declaration_alone():
    """The producer-owned rule chooses from the DECLARATION: the levels each side walks by lists, which the descriptor states as its
    ``write_listed_k`` facts (decode's); no operand is read, and the declaration is hashable (the rule's cache key)."""
    layer, _x = _layer_and_x(1)
    assert activation_walks(layer.topology) == ((False, False), (False, False))
    topology = Topology(levels=(IndependentRouting(width=256, read=SoftmaxActivation(), write=SoftmaxActivation()),) * 2)
    assert activation_walks(topology) == ((False, False), (False, False))
    assert hash(topology) == hash(topology)


# --------------------------------------------------------------- the oracle: r * m, planted


def _draw(T, consumer):
    """Nonzero rows everywhere (so every bit is a real choice), random bits, the masks in ``consumer``'s layout."""
    gen = torch.Generator().manual_seed(11)
    B, H, d_v = 1, 2, 3
    read = [torch.rand(B, T, H, w, generator=gen, dtype=torch.float64) + 0.1 for w in (4, 6)]
    write = [torch.rand(B, T, H, w, generator=gen, dtype=torch.float64) + 0.1 for w in (4, 6)]
    v = torch.randn(B, T, H, d_v, generator=gen, dtype=torch.float64)
    bits = {"read": [], "write": []}
    for side in ("read", "write"):
        for w in (4, 6):
            live = torch.rand(B, T, H, w, generator=gen) < 0.6
            live[..., 0] = True
            bits[side].append(live)
    pack = pack_tokens if consumer == PREFILL else pack_digits
    masks = Masks(consumer, tuple(pack(b) for b in bits["read"]), tuple(pack(b) for b in bits["write"]))
    return v, read, write, bits, masks


@pytest.mark.parametrize(("consumer", "T"), [(DECODE, 1)])
def test_the_oracle_computes_r_times_m_and_a_masked_coefficients_gradient_is_zero(consumer, T):
    """PLANTED: nonzero rows under zero bits. The forward and every gradient equal the oracle run on the rows pre-multiplied
    by their bits (an independent bool expansion, not the masks' own reader); a masked coefficient's gradient is exactly
    zero; and the unmasked oracle differs, so the planting is not vacuous."""
    v, read, write, bits, masks = _draw(T, consumer)
    assert any(not b.all() for b in bits["read"] + bits["write"])
    leaves = [[t.clone().requires_grad_() for t in side] for side in (read, write)]
    twin = [[t.clone().requires_grad_() for t in side] for side in (read, write)]
    y, _ = naive_fsla(v, tuple(leaves[0]), tuple(leaves[1]), _TOPOLOGY, None, masks=masks)
    y.sum().backward()
    premasked = [tuple(t * b.double() for t, b in zip(twin[i], bits[side])) for i, side in enumerate(("read", "write"))]
    want, _ = naive_fsla(v, premasked[0], premasked[1], _TOPOLOGY, None)
    want.sum().backward()
    torch.testing.assert_close(y, want, rtol=0, atol=0)
    for i, side in enumerate(("read", "write")):
        for level, live in enumerate(bits[side]):
            torch.testing.assert_close(leaves[i][level].grad, twin[i][level].grad, rtol=0, atol=0)
            assert leaves[i][level].grad[~live].abs().max() == 0, "a masked coefficient has no gradient"
            assert leaves[i][level].grad[live].abs().max() > 0
    unmasked, _ = naive_fsla(v, tuple(read), tuple(write), _TOPOLOGY, None)
    assert (unmasked - y.detach()).abs().max() > 1e-6


def test_the_oracle_without_masks_is_every_bit_set():
    v, read, write, _bits, _masks = _draw(37, PREFILL)
    ones = Masks(PREFILL, tuple(pack_tokens(torch.ones_like(t, dtype=torch.bool)) for t in read),
                 tuple(pack_tokens(torch.ones_like(t, dtype=torch.bool)) for t in write))
    a, _ = naive_fsla(v, tuple(read), tuple(write), _TOPOLOGY, None)
    b, _ = naive_fsla(v, tuple(read), tuple(write), _TOPOLOGY, None, masks=ones)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
