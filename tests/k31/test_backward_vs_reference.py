# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""R4's two reverse scans against the fp64 CROSS-WINDOW reference, cell by cell.

The reference is `tests/k31/backward_reference.py`'s carry half, which
`test_backward_reference.py` has already pinned to ``torch.autograd`` through the whole
operator. So a disagreement here is the KERNEL's -- and the floor it is measured
against is the ROUNDED REFERENCE, the same gradient taken with the operands rounded the
way the kernels hold them, so what the ratio isolates is the algorithmic error.

THE DIGIT GRADIENTS ARE EXACT ON EACH LEVEL'S OWN SUPPORT, and that is the whole of
what a consumer can read. The schedule visits an owner only where EVERY level's support
meets that owner's run, while ``d_p_l`` drops level ``l``'s own factor -- so off level
``l``'s support the kernel holds a partial sum. It is never read: the exact-entmax
Jacobian is zero off the support in both indices, and a softmax level has no off-support
entry at all. `test_the_off_support_digit_gradients_are_never_read` states that as an
algebraic gate rather than as a comment.
"""
from __future__ import annotations

import pytest
import torch

from rola.ops import carry as carry_ops
from rola.ops._ext import is_available
from tests.k31.backward_reference import carry_backward_reference
from tests.k31.fixtures import Cell, canonical_from_lattice

pytestmark = pytest.mark.skipif(not is_available() or not torch.cuda.is_available(),
                                reason="the reverse pass is a CUDA kernel")

#: the (64, 64) arm is the conformance cell: the fp64 reference materializes ``[T, N]``,
#: so ``N = 4096`` is where a whole cell is checkable.
WIDTHS, K, M, WINDOW = (64, 64), 4, 8, carry_ops.WINDOW_SMALL

#: ONE KAPPA PER OUTPUT, published, never widened per cell. It is a ratio to the
#: rounded-reference floor, so 1 means "no worse than holding the operands in bf16".
KAPPA = {"d_pr": 4.0, "d_pw": 4.0, "d_gw": 4.0, "d_v": 4.0, "d_state_in": 4.0}

CELLS = [
    pytest.param(128, None, 1.0, 1, id="T=2W dense unit-gain"),
    pytest.param(128, None, None, 2, id="T=2W dense gain!=1"),
    pytest.param(256, None, None, 3, id="T=4W dense gain!=1"),
    pytest.param(128, 4, None, 4, id="T=2W k_tok=4"),
    pytest.param(192, 8, None, 5, id="T=3W k_tok=8"),
]


def assert_fresh_backward_binary():
    """DEVICE-SIDE, because a path or hash check cannot catch an extension trap."""
    stamp = carry_ops.backward_stamp(WIDTHS, K, M, WINDOW, 32)
    assert stamp[0] == carry_ops.BWD_BUILD_MAGIC, (
        f"stale binary: the reverse scans' device stamp is {stamp[0]}, not "
        f"{carry_ops.BWD_BUILD_MAGIC}")
    return stamp


def _bth(x, cell, trailing=()):
    """``[BH, T, ...]`` as the reference's ``[B, T, H, ...]``."""
    return x.reshape(cell.B, cell.H, cell.T, *trailing).permute(
        0, 2, 1, *range(3, 3 + len(trailing)))


def _flat_levels(levels, cell):
    """A per-level tuple of ``[B,T,H,w]`` as one ``[BH, T, sum_l w]`` plane."""
    return torch.cat(
        [x.permute(0, 2, 1, 3).reshape(cell.B * cell.H, cell.T, x.shape[-1]) for x in levels],
        dim=-1)


def _run(cell, seed, chained):
    gen = torch.Generator(device="cuda").manual_seed(seed + 100)
    d_num = torch.randn(cell.B * cell.H, cell.T, cell.d_v, device="cuda", dtype=torch.float64,
                        generator=gen)
    d_den = torch.randn(cell.B * cell.H, cell.T, device="cuda", dtype=torch.float64,
                        generator=gen)
    ds_out = (0.05 * torch.randn(cell.B * cell.H, cell.N, cell.d_v + 1, device="cuda",
                                 dtype=torch.float64, generator=gen)) if chained else None
    ds_lat = None
    if ds_out is not None:
        pi = carry_ops.lattice_of_canonical(cell.widths, K, M, device="cuda")
        tmp = torch.zeros_like(ds_out)
        tmp.index_copy_(1, pi, ds_out)
        ds_lat = tmp.to(torch.float32).reshape(cell.B * cell.H, cell.N // 16, 16,
                                               cell.d_v + 1).contiguous()
    got = carry_ops.carry_backward(
        cell.read_bf, cell.write_bf, cell.g_bf.double(), cell.v_bf.double(), d_num, d_den,
        cell.widths, k=K, m=M, window=WINDOW, state_in=cell.state_in_lattice(K, M),
        d_state_out=ds_lat, want_replay=True)
    return got, d_num, d_den, ds_out


def _reference(cell, d_num, d_den, ds_out, rounded):
    levels = cell.ref_levels() if rounded else (cell.read, cell.write, cell.g_write, cell.v)
    seeds = ((d_num.to(torch.bfloat16).double(), d_den.to(torch.bfloat16).double()) if rounded
             else (d_num, d_den))
    sin64 = None if cell.state_in is None else cell.state_in.reshape(cell.B, cell.H, cell.N,
                                                                     cell.d_v + 1)
    dso64 = None if ds_out is None else ds_out.reshape(cell.B, cell.H, cell.N, cell.d_v + 1)
    return carry_backward_reference(levels[0], levels[1], levels[2], levels[3], cell.widths,
                                    WINDOW, _bth(seeds[0], cell, (cell.d_v,)),
                                    _bth(seeds[1], cell), sin64, dso64)


@pytest.mark.parametrize("T,k_tok,gain,seed", [p.values for p in CELLS], ids=[p.id for p in CELLS])
@pytest.mark.parametrize("chained", [False, True])
def test_every_cotangent_matches_the_reference(T, k_tok, gain, seed, chained):
    assert_fresh_backward_binary()
    cell = Cell(WIDTHS, T, k_tok=k_tok, seed=seed, state_in=chained)
    if gain is not None:
        cell.g_write = torch.full_like(cell.g_write, gain)
        cell.g_bf = cell.g_write.to(torch.bfloat16)
    (d_pr, d_pw, d_gw, d_v, d_sin, replay), d_num, d_den, ds_out = _run(cell, seed, chained)
    want = _reference(cell, d_num, d_den, ds_out, rounded=False)
    floor = _reference(cell, d_num, d_den, ds_out, rounded=True)

    supp_r = _flat_levels(tuple(x != 0 for x in cell.read_bf), cell)
    supp_w = _flat_levels(tuple(x != 0 for x in cell.write_bf), cell)
    rows = {
        "d_pr": (d_pr, _flat_levels(want[0], cell), _flat_levels(floor[0], cell), supp_r),
        "d_pw": (d_pw, _flat_levels(want[1], cell), _flat_levels(floor[1], cell), supp_w),
        "d_gw": (d_gw, want[2].permute(0, 2, 1).reshape(cell.B * cell.H, T),
                 floor[2].permute(0, 2, 1).reshape(cell.B * cell.H, T), None),
        "d_v": (d_v, want[3].permute(0, 2, 1, 3).reshape(cell.B * cell.H, T, cell.d_v),
                floor[3].permute(0, 2, 1, 3).reshape(cell.B * cell.H, T, cell.d_v), None),
        "d_state_in": (canonical_from_lattice(d_sin, cell.widths, K, M),
                       want[4].reshape(cell.B * cell.H, cell.N, cell.d_v + 1),
                       floor[4].reshape(cell.B * cell.H, cell.N, cell.d_v + 1), None),
    }
    for name, (got, ref, fl, mask) in rows.items():
        scale = max(1e-30, float(ref.abs().max()))
        sel = (slice(None),) if mask is None else (mask,)
        err = float((got.double() - ref)[sel].abs().max()) / scale
        bar = max(float((fl - ref)[sel].abs().max()) / scale, 1e-7)
        assert err <= KAPPA[name] * bar, f"{name}: {err:.3e} > {KAPPA[name]} x floor {bar:.3e}"

    #: THE REPLAY GATE, the cheapest strong gate in the rung: Scan A's slab at the end
    #: of its walk is the forward's own final state, same grid and same batch order.
    fwd = carry_ops.carry_inter(cell.read_bf, cell.write_bf, cell.g_bf.double(),
                                cell.v_bf.double(), cell.widths, k=K, m=M, window=WINDOW,
                                state_in=cell.state_in_lattice(K, M))
    assert torch.equal(replay, fwd[2]), (
        "the replay is not bit-identical to the forward's state_out: rel "
        f"{float((replay - fwd[2]).abs().max()) / max(1e-30, float(fwd[2].abs().max())):.3e}")


def _entmax_vjp(p, dp, alpha=1.5):
    """The alpha-entmax VJP, from the support pattern and ``p`` alone.

    ``gppr = p ** (2 - alpha)`` on the support and 0 off it, ``dz = gppr * (dp - q)``
    with ``q = <gppr, dp> / <gppr, 1>`` -- so an off-support ``dp`` enters neither term.
    """
    gppr = torch.where(p > 0, p.clamp_min(0) ** (2 - alpha), torch.zeros_like(p))
    dx = dp * gppr
    q = dx.sum(-1, keepdim=True) / gppr.sum(-1, keepdim=True)
    return dx - q * gppr


@pytest.mark.parametrize("k_tok", [4, 8])
def test_the_off_support_digit_gradients_are_never_read(k_tok):
    """The contract the schedule's coverage rests on, as algebra rather than as prose.

    The kernel's ``d_p`` is a PARTIAL sum off a level's own support -- the schedule
    visits an owner only where every level's support meets it. Pushing the kernel's
    plane and the complete fp64 plane through the same exact-entmax VJP gives the same
    logit gradient, which is the whole of what the producer reads.
    """
    assert_fresh_backward_binary()
    cell = Cell(WIDTHS, 128, k_tok=k_tok, seed=11)
    (d_pr, d_pw, *_), d_num, d_den, ds_out = _run(cell, 11, False)
    want = _reference(cell, d_num, d_den, ds_out, rounded=True)
    for got, ref, levels in ((d_pr, want[0], cell.read_bf), (d_pw, want[1], cell.write_bf)):
        off = 0
        for level, whole in zip(levels, ref):
            width = level.shape[-1]
            p = level.double().permute(0, 2, 1, 3).reshape(cell.B * cell.H, cell.T, width)
            mine = _entmax_vjp(p, got[:, :, off:off + width].double())
            full = _entmax_vjp(p, whole.permute(0, 2, 1, 3).reshape(cell.B * cell.H, cell.T,
                                                                    width))
            scale = max(1e-30, float(full.abs().max()))
            assert float((mine - full).abs().max()) / scale < 1e-2
            off += width


@pytest.mark.parametrize("window", [carry_ops.WINDOW, carry_ops.WINDOW_WIDE])
def test_the_replay_is_bit_identical_at_the_flagship_arm(window):
    """THE CHEAPEST STRONG GATE IN THE RUNG, at the arm the flagship actually runs.

    Scan A's slab at the end of its walk is the forward's own ``state_out``: same window
    grid, same lists, same batch order, same k-loop order. A drifting replay is exactly
    the failure a loose band hides, so the claim is EQUALITY and not a band -- and it
    needs no fp64 reference, which is what lets it run at ``N = 65536``.
    """
    from tests.k31.intra_reference import _simplex
    widths, T = (256, 256), 4 * window
    gen = torch.Generator(device="cuda").manual_seed(19)
    read = tuple(_simplex((1, T, w), 4, gen, "cuda", False).reshape(1, T, 1, w).to(
        torch.bfloat16) for w in widths)
    write = tuple(_simplex((1, T, w), None, gen, "cuda", False).reshape(1, T, 1, w).to(
        torch.bfloat16) for w in widths)
    g = (0.5 + torch.rand(1, T, 1, device="cuda", generator=gen)).to(torch.bfloat16)
    v = torch.randn(1, T, 1, 64, device="cuda", generator=gen).to(torch.bfloat16)
    d_num = torch.randn(1, T, 64, device="cuda", generator=gen)
    d_den = torch.randn(1, T, device="cuda", generator=gen)
    replay = carry_ops.carry_backward(read, write, g.double(), v.double(), d_num, d_den,
                                      widths, k=K, m=M, window=window, want_replay=True)[5]
    state_out = carry_ops.carry_inter(read, write, g.double(), v.double(), widths, k=K, m=M,
                                      window=window)[2]
    assert torch.equal(replay, state_out)
