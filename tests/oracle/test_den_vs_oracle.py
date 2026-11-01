# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE DEN PASS AGAINST THE fp64 RECURRENCE: each token's window-aligned denominator and readout bound over a whole
call, per slot, on every central oracle cell.
``den`` is the inter term's denominator (`tests/oracle/reference.py`'s ``inter_reference`` on the bundle's own
dequantized operands, the state before each window read by the window's tokens); ``num`` the same product against the
mass of ``F = m s`` -- the recurrence run with each writing token's value scale as its one value channel. The kernel's
sums are in its operands' units: level S's coefficient is the byte over 256 where the level is u8 (the reference's byte
over 255), the mass carries the write bytes where level R is u8 (the reference's bytes over 255), and the reader's
per-token scales are left out (the carry's contract); :func:`in_truth` applies the factor.
RED BY PLAN: a cell the pass does not serve (`rola.ops.den.serves`) or whose entry state it cannot take fails by name.
Symbols: ``W`` the window, ``L`` tokens, ``g`` query heads a write head.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from measure.cells import carry_cells, realize
from measure.subjects import bundle_of, read_row_of
from rola.ops import den as den_ops
from rola.ops import facts as facts_ops
from rola.ops.window import window_of
from tests.oracle import reference
from tests.oracle.fixtures import assert_planted_errors_fail, assert_slots_close
from tests.oracle.tolerances import CARRY_DEN

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

ORACLE_CELLS = carry_cells(tier="oracle")
IDS = [cell.name for cell in ORACLE_CELLS]


def run(spec):
    """``(routes, values, den, num)``: one den launch over the cell."""
    routes, values, layout, statics = bundle_of(spec, realize(spec))
    if spec.state != "fresh":
        raise NotImplementedError(f"{spec.name}: the den pass takes no entry state yet")
    facts = facts_ops.facts(routes, values, layout, *statics, read_row_of(spec))
    den, num = den_ops.den(routes, facts)
    torch.cuda.synchronize()
    return routes, values, den, num


def in_truth(x: torch.Tensor, routes) -> torch.Tensor:
    """``[B, T, H]`` fp64: the kernel's ``[BH g, L]`` sums in the recurrence's units."""
    B, T, Hw, g = routes.batch, routes.tokens, routes.write_heads, routes.group
    y = x.double().reshape(B, Hw * g, T).permute(0, 2, 1)
    ls = routes.level_s
    lr = 1 - ls
    f = routes.scale_read[ls].double() * routes.scale_read[lr].double()
    if routes.sparse[0][ls]:
        f = f * (256.0 / 255.0)
    if routes.sparse[1][lr]:
        f = f / 255.0
    return y * f


def refs(routes, values) -> tuple[torch.Tensor, torch.Tensor]:
    """``(den, num)`` ``[B, T, H]`` fp64: the inter term's denominator and the same recurrence with each token's value
    scale as its one value channel."""
    read, write = routes.dequantized()
    num, den, _ = reference.inter_reference(tuple(x.double() for x in read), tuple(x.double() for x in write),
                                            routes.g_write.double(), values.scale.double().unsqueeze(-1),
                                            routes.topology.widths, window_of())
    return den, num[..., 0]


@pytest.mark.parametrize("spec", ORACLE_CELLS, ids=IDS)
def test_the_denominator_is_the_fp64_reference(spec):
    """Every token's ``den`` over the whole call, in the recurrence's units, against the inter term's denominator."""
    routes, values, den, _num = run(spec)
    den_ref, _ = refs(routes, values)
    assert_slots_close(in_truth(den, routes), den_ref, output=CARRY_DEN, envelope=den_ref, what=f"{spec.name} den")


@pytest.mark.parametrize("spec", ORACLE_CELLS, ids=IDS)
def test_the_bound_is_the_fp64_reference(spec):
    """Every token's ``num`` -- the read product against the mass of ``F`` -- against the recurrence on the value
    scales."""
    routes, values, _den, num = run(spec)
    _, num_ref = refs(routes, values)
    assert_slots_close(in_truth(num, routes), num_ref, output=CARRY_DEN, envelope=num_ref, what=f"{spec.name} num")


def test_the_rule_fails_planted_errors_on_the_kernels_own_output():
    """THE RULE HAS TEETH on the den: on a real cell the kernel's output passes, and a wiped median slot and the
    smallest slots moved past their allowance fail."""
    #: two windows at least, at the card's W: a one-window call's inter term is identically zero, nothing to plant
    base = next(c for c in ORACLE_CELLS if c.name == "flagship-alt-k4")
    spec = dataclasses.replace(base, tokens=max(base.tokens, 2 * window_of()))
    routes, values, den, _num = run(spec)
    den_ref, _ = refs(routes, values)
    assert_planted_errors_fail(in_truth(den, routes), den_ref, output=CARRY_DEN, envelope=den_ref,
                               what=f"{spec.name} den")


def test_a_bound_ledger_books_every_phase_and_changes_nothing():
    """The phase ledger, bound, books cycles in every phase of each warp's role, none in the other role's, and leaves
    the outputs as they were."""
    spec = next(c for c in ORACLE_CELLS if c.name == "flagship-alt-k4")
    routes, values, den, num = run(spec)
    ctas = den_ops.tiles(routes) * routes.batch * routes.write_heads
    ledger = torch.zeros((ctas, den_ops.LEDGER_WARPS, len(den_ops.LEDGER_PHASES)), dtype=torch.int64, device="cuda")
    den_ops.ledger_bind(ledger)
    try:
        _, _, den2, num2 = run(spec)
    finally:
        den_ops.ledger_bind(None)
    torch.testing.assert_close(den2, den, rtol=1e-5, atol=0.0)
    torch.testing.assert_close(num2, num, rtol=1e-5, atol=0.0)
    for warps, phases in den_ops.LEDGER_ROLES.values():
        for k, phase in enumerate(den_ops.LEDGER_PHASES):
            booked = ledger[:, warps.start:warps.stop, k]
            assert ((booked > 0) if phase in phases else (booked == 0)).all(), (phase, booked)
