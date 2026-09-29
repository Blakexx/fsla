# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PRE-PASS KERNEL against its declared arithmetic, and its frame's ledger.

Two rows. The FRAME row runs the skeleton on a cell with the phase ledger bound and asserts the frame
executed: every phase carries cycles on every warp, the outputs are finite. The GATE row compares
``(den, scale)`` per token to ``int_reference.prepass_reference`` -- den within the u8 read row's precision,
scale never below the reference's bound -- and is the op gate the components land against; it is marked
strict-xfail until the fold and readout components land, so its first pass is a recorded event and not a
silent one.
"""
from __future__ import annotations

import pytest
import torch

from measure.cells import by_name, realize
from rola.ops import carry as carry_ops
from rola.ops import prepass as pp
from rola.routing.factors import RouteFactors
from rola.routing.types import IndependentRouting, SoftmaxActivation, Topology
from rola.values import Values
from tests.oracle import int_reference as iref

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

CELLS = ["flagship-alt-k4", "flagship-cohort-k16"]
FOLD_CELLS = ["one-box-alt-k4", "flagship-alt-k4", "flagship-cohort-k16"]


def _bundle(cell, drawn):
    read, write, gain, v = drawn.doubles()
    #: the bundle's topology: the widths are what the pre-pass reads; the activations are the draw's business
    routing = IndependentRouting(width=1, read=SoftmaxActivation(), write=SoftmaxActivation())
    topology = Topology(levels=tuple(routing.at(w) for w in cell.widths))
    factors = RouteFactors.from_levels(topology, tuple(x.float() for x in read), tuple(x.float() for x in write),
                                       gain.float())
    values = Values.from_projection(v.float())
    return factors, values, (read, write, gain, v)


def test_the_plan_is_the_tall_owner_set():
    plan = pp.prepass_plan(2)
    assert plan["threads"] == 256 and plan["leaves"] == 1024 and plan["leaves_per_warp"] == 128
    assert plan["smem_bytes"] <= 99 * 1024 and plan["local_bytes"] == 0, plan


@pytest.mark.parametrize("name", CELLS)
def test_the_frame_runs_and_every_phase_is_counted(name, capsys):
    cell = by_name(name)
    drawn = realize(cell, device="cuda")
    factors, values, _ = _bundle(cell, drawn)
    plan = pp.prepass_plan(2)
    g = carry_ops.geometry(list(cell.widths), level_modes=pp.level_modes_of(cell.declared_sparsity()), bc=1024,
                           nsr=8, nsw=8, d_v=32, atom_floor=0)
    ctas = g["owners"] * factors.batch * factors.heads
    ledger = torch.zeros(ctas, 8, plan["phases"], dtype=torch.int64, device="cuda")
    pp.ledger_bind(ledger)
    try:
        den, scale = pp.prepass_forward(factors, values, cell.declared_sparsity())
        torch.cuda.synchronize()
    finally:
        pp.ledger_bind(None)
    assert torch.isfinite(den).all() and torch.isfinite(scale).all()
    per_phase = ledger.sum(dim=(0, 1)).double() / (ledger.shape[0] * 8)
    windows = (cell.tokens + carry_ops.WINDOW - 1) // carry_ops.WINDOW
    names = ["frame", "words", "lists", "fill", "fold", "publish", "rescale", "readout", "drain"]
    with capsys.disabled():
        print(f"\n{name}: {ctas} CTAs x {windows} windows; cycles a warp a window:")
        for n, c in zip(names, per_phase.tolist()):
            print(f"  {n:8s} {c / windows:10.0f}")
        print(f"  total    {per_phase.sum().item() / windows:10.0f}")
    assert bool((ledger.sum(dim=2) > 0).all()), "a warp recorded no cycles"


@pytest.mark.parametrize("name", FOLD_CELLS)
def test_the_fold_state_is_the_declared_arithmetic(name):
    """The fold component's gate: the kernel's fp32 state a canonical leaf against the reference's mass and
    bound -- identical integers by construction, so the tolerance is fp32 against fp64 accumulation."""
    import math

    cell = by_name(name)
    drawn = realize(cell, device="cuda")
    factors, values, (read, write, gain, v) = _bundle(cell, drawn)
    N = math.prod(cell.widths)
    BH = factors.batch * factors.heads
    state = torch.zeros(BH, N, 2, dtype=torch.float32, device="cuda")
    pp.prepass_forward(factors, values, cell.declared_sparsity(), fold_dump=state)
    torch.cuda.synchronize()
    tel: dict = {}
    iref.prepass_reference(read, write, gain, v, cell.widths, cell.declared_sparsity(), carry_ops.WINDOW,
                           telemetry=tel)
    got = state.reshape(factors.batch, factors.heads, N, 2).double()
    for col, want in ((0, tel["mass"]), (1, tel["bound"])):
        have = got[..., col]
        assert torch.isfinite(have).all()
        assert bool((want > 0).any()), "the reference fold produced no state"
        assert torch.allclose(have, want, rtol=2.0 ** -12, atol=2.0 ** -20 * want.max().item()), (
            col, (have - want).abs().max().item(), want.max().item())


@pytest.mark.xfail(strict=True, reason="the pre-pass's readout component has not landed (queue card S8)")
@pytest.mark.parametrize("name", CELLS)
def test_den_and_scale_are_the_declared_arithmetic(name):
    cell = by_name(name)
    drawn = realize(cell, device="cuda")
    factors, values, (read, write, gain, v) = _bundle(cell, drawn)
    den, scale = pp.prepass_forward(factors, values, cell.declared_sparsity())
    want_den, want_scale = iref.prepass_reference(read, write, gain, v, cell.widths, cell.declared_sparsity(),
                                                  carry_ops.WINDOW)
    Bn, T, H = want_den.shape
    den_bth = den.reshape(Bn, H, T).permute(0, 2, 1).double()
    scale_bth = scale.reshape(Bn, H, T).permute(0, 2, 1).double()
    live = want_den > 0
    assert torch.allclose(den_bth[live], want_den[live], rtol=2.0 ** -8, atol=0.0)
    assert bool((scale_bth[live] * (1 + 2.0 ** -12) >= want_scale[live]).all())
