# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE INT8 CARRY'S DECLARED ARITHMETIC AGAINST THE fp64 OP, and the report beside it.

Three claims. (1) The Hadamard rotation is exact for the op: it is orthogonal and the op is linear in the values, so
un-rotating the op's output on rotated values IS the op on the values. (2) The declared recurrence is the op's: with
every rounding turned off, `int_reference` reproduces `inter_reference` to fp64 on every cell it is run on -- the
structure is judged before any rounding is. (3) THE REPORT: on the oracle cells, the declared arithmetic's error against
the unquantized op, per slot over the slot's envelope and per token in norm, at one and two digits, in the rotated and
the real basis. It is PRINTED and never gated (`docs/internals/...`: the truth oracle reports, the fidelity gate is the
op gate on the same quantized inputs, quality is the layer oracle's). The one property asserted is monotone: two digits
never lose to one on the same draw.
"""
from __future__ import annotations

import pytest
import torch

from measure.cells import by_name, realize
from rola.ops import carry as carry_ops
from tests.oracle import int_reference as iref
from tests.oracle import reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

STRUCTURE_CELLS = ["flat-small-alt-k4", "one-box-short", "deep3-alt-k4"]
REPORT_CELLS = ["flat-small-dense", "flat-small-alt-k4", "flat-small-alt-k16", "one-box-alt-k4",
                "flagship-alt-k4", "flagship-cohort-k16", "deep3-alt-k4"]


def _sparsity(cell):
    return cell.declared_sparsity()


def test_the_hadamard_is_orthogonal_and_commutes_with_the_op():
    dev = "cuda"
    h = iref.hadamard(64, dev)
    eye = torch.eye(64, device=dev, dtype=torch.float64)
    assert torch.allclose(h @ h.T, eye, atol=1e-13)
    cell = by_name("flat-small-alt-k4")
    drawn = realize(cell, device=dev)
    read, write, gain, v = drawn.doubles()
    num, den, state = reference.inter_reference(read, write, gain, v, cell.widths, carry_ops.WINDOW)
    num_r, den_r, state_r = reference.inter_reference(read, write, gain, iref.rotate(v, h), cell.widths,
                                                      carry_ops.WINDOW)
    assert torch.allclose(iref.unrotate(num_r, h), num, rtol=1e-12, atol=1e-12 * num.abs().max())
    assert torch.allclose(den_r, den)
    assert torch.allclose(iref.unrotate(state_r[..., :-1], h), state[..., :-1], rtol=1e-12,
                          atol=1e-12 * state.abs().max())


@pytest.mark.parametrize("name", STRUCTURE_CELLS)
def test_the_declared_recurrence_is_the_op_when_nothing_is_rounded(name):
    cell = by_name(name)
    drawn = realize(cell, device="cuda")
    read, write, gain, v = drawn.doubles()
    want = reference.inter_reference(read, write, gain, v, cell.widths, carry_ops.WINDOW)
    got = iref.int_reference(read, write, gain, v, cell.widths, _sparsity(cell), carry_ops.WINDOW, exact=True)
    for a, b, what in zip(got, want, ("num", "den", "state")):
        scale = b.abs().max().clamp_min(1e-300)
        assert torch.allclose(a, b, rtol=1e-11, atol=1e-11 * float(scale)), f"{name}: {what} is not the op's"


@pytest.mark.parametrize("name", REPORT_CELLS)
def test_the_quantization_report(name, capsys):
    """REPORTED, NEVER GATED: the declared int arithmetic's error against the unquantized op."""
    cell = by_name(name)
    drawn = realize(cell, device="cuda")
    lines = [f"{name}: widths={cell.widths} tokens={cell.tokens} draw={cell.draw} k={cell.k_tok}"]
    for basis, rot in (("rotated-basis-read", False), ("real-basis-after-unrotate", True)):
        rows = iref.quantization_report(drawn, _sparsity(cell), carry_ops.WINDOW, rotate_values=rot)
        for r in rows:
            lines.append(f"  {basis:26s} digits fold={r['fold_digits']} readout={r['readout_digits']}: "
                         f"num per-slot err / (2^-8 env): max {r['slot_max']:8.2f}, frac>1 {100*r['slot_frac_over']:5.1f}% | "
                         f"per-token norm err/env: max {r['token_max']:.2e} mean {r['token_mean']:.2e} | "
                         f"state per-slot max {r['state_slot_max']:8.2f} frac>1 {100*r['state_slot_frac_over']:5.1f}% | "
                         f"den rel {r['den_max_rel']:.1e}")
        one, two = rows[0], rows[-1]
        assert two["token_mean"] <= one["token_mean"] * 1.0001, "two digits lost to one on the same draw"
    with capsys.disabled():
        print("\n" + "\n".join(lines))
