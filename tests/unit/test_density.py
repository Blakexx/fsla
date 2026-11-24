# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PREFILL'S DENSITY (docs/api.md section 2.6): the helpers that build it against a brute-force count, the call's refusals by their words, and
that it changes no defined result. No device.

Symbols: ``B`` batch, ``T`` tokens, ``H`` heads, ``D`` levels, ``w`` a level's width; the density is ``[2, D]``, row 0 the read side and row 1 the write side.
"""
from __future__ import annotations

import pytest
import torch

import fsla
from fsla.ops.naive import naive_fsla
from fsla.routing.factors import RouteFactors
from fsla.routing.masks import PREFILL, Masks, density_of_masks, masks_and_density, pack_tokens
from fsla.routing.types import IndependentRouting, SoftmaxActivation, Topology
from fsla.values import Values


def _topology(widths):
    return Topology(levels=tuple(IndependentRouting(width=w, read=SoftmaxActivation(), write=SoftmaxActivation()) for w in widths))


def _levels(gen, B, T, H, widths, keep):
    """Nonnegative levels of shape ``[B, T, H, w]`` a level, each digit zero with probability ``1 - keep``."""
    out = []
    for w in widths:
        level = torch.rand(B, T, H, w, generator=gen) + 0.5
        out.append(level * (torch.rand(B, T, H, w, generator=gen) < keep))
    return out


def _bundle(widths, B, T, H, H_w, keep, seed=5):
    gen = torch.Generator().manual_seed(seed)
    routes = RouteFactors.from_levels(_topology(widths), _levels(gen, B, T, H, widths, keep), _levels(gen, B, T, H_w, widths, keep))
    return routes, gen


def _brute_density(routes):
    """The share of live ``(token, digit)`` pairs of each level, counted one entry at a time over the rows."""
    out = []
    for side in ("read", "write"):
        row = []
        for level in range(routes.topology.D):
            rows = routes.rows(level, side)
            live = sum(1 for entry in rows.reshape(-1).tolist() if entry != 0)
            row.append(live / rows.numel())
        out.append(row)
    return torch.tensor(out, dtype=torch.float32)


CASES = [((8,), 1, 1, 1, 1), ((4, 8), 2, 31, 2, 2), ((4, 4, 8), 1, 33, 3, 3), ((2, 4, 4, 8), 2, 70, 1, 1), ((8, 4), 1, 65, 4, 2)]


@pytest.mark.parametrize(("widths", "B", "T", "H", "H_w"), CASES)
def test_the_helpers_equal_a_brute_force_count_and_the_masks_are_the_rows_nonzero(widths, B, T, H, H_w):
    routes, _ = _bundle(widths, B, T, H, H_w, keep=0.4)
    masks, density = masks_and_density(routes)
    want = _brute_density(routes)
    assert density.dtype is torch.float32 and tuple(density.shape) == (2, len(widths)) and density.device == routes.device
    torch.testing.assert_close(density, want, rtol=0, atol=0)
    torch.testing.assert_close(density_of_masks(routes, masks), want, rtol=0, atol=0)
    masks.check(routes)
    assert masks.consumer == PREFILL
    for side in ("read", "write"):
        for level in range(len(widths)):
            rows, words = routes.rows(level, side), masks.level(side, level)
            for b, t, h, d in [(0, 0, 0, 0), (B - 1, T - 1, rows.shape[2] - 1, rows.shape[3] - 1), (0, T // 2, 0, rows.shape[3] // 2)]:
                bit = (int(words[b, h, t // 32, d]) >> (t % 32)) & 1
                assert bit == int(rows[b, t, h, d] != 0), "a bit is 1 exactly where the row is not zero"


def test_a_level_all_live_is_exactly_one_and_a_level_all_dead_exactly_zero():
    routes, _ = _bundle((4, 8), 2, 45, 2, 2, keep=1.0)
    _, density = masks_and_density(routes)
    assert torch.equal(density, torch.ones(2, 2))
    dead, _ = _bundle((4, 8), 2, 45, 2, 2, keep=0.0)
    _, density = masks_and_density(dead)
    assert torch.equal(density, torch.zeros(2, 2))
    #: bits past T are not tokens: a word set whole counts only the T tokens it holds
    whole = Masks(PREFILL, tuple(torch.full_like(m, -1) for m in masks_and_density(routes)[0].read),
                  tuple(torch.zeros_like(m) for m in masks_and_density(routes)[0].write))
    assert torch.equal(density_of_masks(routes, whole), torch.tensor([[1.0, 1.0], [0.0, 0.0]]))


def test_the_density_of_masks_a_caller_has_is_the_set_bits_not_the_rows():
    """A bit of 1 over a zero row is allowed (it guarantees nothing) and counts: the density is of the masks."""
    routes, _ = _bundle((4,), 1, 40, 1, 1, keep=0.25)
    widened = Masks(PREFILL, (torch.full_like(masks_and_density(routes)[0].read[0], -1),), masks_and_density(routes)[0].write)
    assert float(density_of_masks(routes, widened)[0, 0]) == 1.0
    assert float(masks_and_density(routes)[1][0, 0]) < 1.0


def test_the_density_of_masks_refuses_masks_that_do_not_fit_the_bundle():
    routes, _ = _bundle((4, 8), 1, 40, 1, 1, keep=0.5)
    masks, _ = masks_and_density(routes)
    short = Masks(PREFILL, (masks.read[0][:, :, :1], *masks.read[1:]), masks.write)
    with pytest.raises(ValueError, match=r"Masks.read\[0\] \(prefill layout\)"):
        density_of_masks(routes, short)
    decode = Masks("decode", masks.read, masks.write)
    with pytest.raises(ValueError, match="the density is the prefill's"):
        density_of_masks(routes, decode)


# --------------------------------------------------------------- the call's refusals, by their words


def _call_operands(T=40):
    routes, gen = _bundle((4, 8), 1, T, 2, 2, keep=0.5)
    values = Values.from_projection(torch.randn(1, T, 2, 8, generator=gen))
    masks, density = masks_and_density(routes)
    return routes, values, masks, density


def test_a_missing_density_is_refused():
    routes, values, masks, _ = _call_operands()
    with pytest.raises(TypeError, match="density"):
        fsla.prefill(routes, values, masks)


@pytest.mark.parametrize(("make", "error", "words"), [
    (lambda D: [[0.5] * D, [0.5] * D], TypeError, r"`density` is a floating torch.Tensor \[2, D\], got list"),
    (lambda D: torch.ones(3, D), ValueError, r"`density` is \[2, D\] with D=2 .*got \(3, 2\)"),
    (lambda D: torch.ones(2, D + 1), ValueError, r"`density` is \[2, D\] with D=2 .*got \(2, 3\)"),
    (lambda D: torch.ones(D), ValueError, r"`density` is \[2, D\] with D=2 .*got \(2,\)"),
    (lambda D: torch.ones(2, D, dtype=torch.int64), ValueError, r"`density` is a floating tensor, got torch.int64"),
    (lambda D: torch.ones(2, D, dtype=torch.bool), ValueError, r"`density` is a floating tensor, got torch.bool"),
    (lambda D: torch.ones(2, D, device="meta"), ValueError, r"`density` is on meta but the rows are on cpu"),
])
def test_the_calls_refusals_name_the_density_clause(make, error, words):
    routes, values, masks, _ = _call_operands()
    with torch.no_grad(), pytest.raises(error, match=words):
        fsla.prefill(routes, values, masks, make(routes.topology.D))


def test_no_value_of_the_density_is_refused_and_none_changes_the_refusal_the_call_gives():
    """Nothing about the values is checked: a negative, a huge, a NaN and an infinite entry pass the density's clause, and two very
    different densities meet the same refusal, the built envelope's (this machine has no device)."""
    routes, values, masks, density = _call_operands()
    crazy = torch.tensor([[-3.0, 1e9], [float("nan"), float("inf")]])
    messages = []
    for given in (density, torch.zeros_like(density), torch.ones_like(density), crazy):
        with torch.no_grad(), pytest.raises(NotImplementedError, match="CUDA kernel") as raised:
            fsla.prefill(routes, values, masks, given)
        messages.append(str(raised.value))
    assert len(set(messages)) == 1


def test_the_reference_result_does_not_change_with_the_density():
    """The oracle takes no density. Masks that keep the guarantee (a zero bit over a zero row) give the rows' result, and widening
    bits to 1 guarantees nothing and changes nothing: the two mask sets have very different densities and the same result."""
    routes, gen = _bundle((4, 8), 1, 37, 2, 2, keep=0.3)
    v = torch.randn(1, 37, 2, 3, generator=gen, dtype=torch.float64)
    sparse, sparse_density = masks_and_density(routes)
    full = Masks(PREFILL, tuple(pack_tokens(torch.ones_like(routes.rows(level, "read"), dtype=torch.bool)) for level in range(2)),
                 tuple(pack_tokens(torch.ones_like(routes.rows(level, "write"), dtype=torch.bool)) for level in range(2)))
    full_density = density_of_masks(routes, full)
    assert float(sparse_density.max()) < 0.5 and torch.equal(full_density, torch.ones(2, 2))
    read, write = routes.dequantized()
    results = [naive_fsla(v, read, write, routes.topology, None, masks=masks)[0] for masks in (sparse, full, None)]
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
    torch.testing.assert_close(results[0], results[2], rtol=0, atol=0)
