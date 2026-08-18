"""THE BACKWARD LADDER, RUNG 0 — the dense two-pass backward, against the oracle.

Vocabulary, defined here and used unqualified below: `C` = chunk tokens; `BC` =
slots per CTA block, which at this rung IS the slot space (`N = BC`, one block);
`T` = tokens; `d_v = 64`; `R`/`Wc` = the per-slot read/write coefficients; `S` = the
forward state `[N, d_v]`; `dS` its adjoint; `dnum` = the readout's value-channel
cotangent seed. Rung 0 carries no mass column, no write gain and no digits, so
`dden` and the ones column do not appear: they arrive at rung b1.

**WHAT IS GATED, AND AGAINST WHAT.** Two references, and both are needed:

* the **fp64 autograd oracle** — an eleven-line textbook recurrence in float64,
  differentiated by autograd. It knows nothing about chunks, so it is the statement
  of WHAT the kernel computes and cannot agree with a kernel that is merely
  self-consistent;
* the **rounded reference** — the same algebra chunked the way the kernel chunks it,
  rounding every operand where the kernel rounds it (the published state and both
  Grams to bf16), which is what makes a TIGHT derived budget possible. What is left
  to separate the two is summation ORDER plus each stored operand's one rounding.

The teeth are explicit: the budget must REJECT a reference with the intra staircase
dropped and one with the staircase made exclusive.
"""
from __future__ import annotations

import pytest
import torch

from rola.ops._ext import extension, is_available

pytestmark = pytest.mark.skipif(
    not is_available() or not torch.cuda.is_available(),
    reason="the ladder's rung 0 is a CUDA kernel")

DV = 64
EPS32 = 2.0 ** -24
ULP_BF16_PAIR = 2.0 ** -7

#: `(C, BC)` -- the built rung-0 arms. `(32, 64)` is the head-to-head arm: it is
#: where a scan arm's `N` equals its `BC`, so rung 0 and the shipped scans run the
#: same dense cell.
ARMS = [(32, 64), (16, 64)]

#: The TOPOLOGIES the dense coefficients are drawn from, both reaching `N = 64`:
#: the square `(8, 8)` and the flat `(64,)`. Both are scan arms, which is what makes
#: the head-to-head comparison a comparison and not an analogy.
TOPOLOGIES = [(8, 8), (64,)]


def _rb(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float32)


def _kron(widths, T, gen, dev) -> torch.Tensor:
    """One duty's DENSE per-slot coefficients: a softmax per level, multiplied over
    the levels exactly as the routed Kronecker product does. Softmax on both duties
    is the dense constructors' own shape, so every slot is on support."""
    out = torch.ones(T, 1, device=dev, dtype=torch.float32)
    for w in widths:
        amp = torch.softmax(torch.randn(T, w, device=dev, generator=gen), dim=-1)
        out = (out.unsqueeze(2) * amp.unsqueeze(1)).reshape(T, -1)
    return out


def _inputs(widths, T, seed, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(seed)
    r = _kron(widths, T, g, dev)
    w = _kron(widths, T, g, dev)
    v = torch.randn(1, T, 1, DV, device=dev, generator=g)
    dnum = torch.randn(1, T, DV, device=dev, generator=g)
    return tuple(t.to(torch.bfloat16).contiguous() for t in (r.unsqueeze(0), w.unsqueeze(0),
                                                             v, dnum))


def _run(arm, inputs):
    c, bc = arm
    r, w, v, dnum = inputs
    m = extension()
    (dr,) = m.chunk_rung0_pass1(w, v, dnum, c, bc)
    dv, dw = m.chunk_rung0_pass2(r, w, v, dnum, c, bc)
    return dr[0], dv[0], dw[0]


# ---------------------------------------------------------------- the oracle
def _fp64_oracle(r, w, v, dnum):
    """THE ELEVEN-LINE RECURRENCE, in float64, differentiated by autograd.

    `S` accumulates the write side's outer products and a read at `t` sees its own
    write, which is the inclusive read-after-write the forward performs.
    """
    R = r[0].to(torch.float64).requires_grad_(True)
    Wc = w[0].to(torch.float64).requires_grad_(True)
    V = v[0, :, 0].to(torch.float64).requires_grad_(True)
    dn = dnum[0].to(torch.float64)
    T, N = R.shape
    S = torch.zeros(N, DV, device=R.device, dtype=torch.float64)
    out = []
    for t in range(T):
        S = S + Wc[t].unsqueeze(1) * V[t].unsqueeze(0)
        out.append(R[t] @ S)
    loss = (torch.stack(out) * dn).sum()
    dR, dW, dV = torch.autograd.grad(loss, (R, Wc, V))
    return dR, dV, dW


# ------------------------------------------------------- the rounded reference
def _rounded(r, w, v, dnum, C, *, intra=True, exclusive=False):
    """The kernel's own association, with every stored operand rounded where the
    kernel rounds it, plus the magnitude accumulation the budget is derived from."""
    dev = r.device
    R, Wc = _rb(r)[0], _rb(w)[0]
    V, dn = _rb(v)[0, :, 0], _rb(dnum)[0]
    T, N = R.shape
    tok = torch.arange(C, device=dev)
    #: rows are READ tokens and columns WRITE tokens, and a write is kept at or
    #: before the read; the exclusive form is the planted fault, not a variant.
    stair = ((tok.unsqueeze(1) > tok.unsqueeze(0)) if exclusive
             else (tok.unsqueeze(1) >= tok.unsqueeze(0))).float()

    dR = torch.zeros(T, N, device=dev)
    aR = torch.zeros(T, N, device=dev)
    S = torch.zeros(N, DV, device=dev)
    aS = torch.zeros(N, DV, device=dev)
    for c0 in range(0, T, C):
        idx = slice(c0, c0 + C)
        Sb = _rb(S)
        Gd = _rb(dn[idx] @ V[idx].T) * stair
        inter = dn[idx] @ Sb.T
        aI = dn[idx].abs() @ aS.T
        dR[idx] = inter + (Gd @ Wc[idx] if intra else 0.0)
        aR[idx] = aI + Gd.abs() @ Wc[idx].abs()
        S = S + Wc[idx].T @ V[idx]
        aS = aS + Wc[idx].abs().T @ V[idx].abs()

    dv = torch.zeros(T, DV, device=dev)
    adv = torch.zeros(T, DV, device=dev)
    dW = torch.zeros(T, N, device=dev)
    adW = torch.zeros(T, N, device=dev)
    dS = torch.zeros(N, DV, device=dev)
    adS = torch.zeros(N, DV, device=dev)
    for c0 in range(T - C, -1, -C):
        idx = slice(c0, c0 + C)
        dSb = _rb(dS)
        Gu = _rb(R[idx] @ Wc[idx].T) * stair
        Gd = _rb(dn[idx] @ V[idx].T) * stair
        dv[idx] = Wc[idx] @ dSb + (Gu.T @ dn[idx] if intra else 0.0)
        adv[idx] = Wc[idx].abs() @ adS + Gu.abs().T @ dn[idx].abs()
        dW[idx] = V[idx] @ dSb.T + (Gd.T @ R[idx] if intra else 0.0)
        adW[idx] = V[idx].abs() @ adS.T + Gd.abs().T @ R[idx].abs()
        dS = dS + R[idx].T @ dn[idx]
        adS = adS + R[idx].abs().T @ dn[idx].abs()
    return {"dR": dR, "dv": dv, "dW": dW, "aR": aR, "adv": adv, "adW": adW}


def _budget(ref, abs_key, T, N):
    """The derived per-element budget: one bf16 rounding on each stored operand,
    plus fp32 accumulation over the terms the sum actually runs."""
    return (ULP_BF16_PAIR + (T + N + DV) * EPS32) * ref[abs_key]


CELLS = [
    #: ONE CHUNK: the entry state and entry adjoint are exactly zero, so every
    #: gradient here is ENTIRELY the intra staircase. Nothing else can carry it.
    pytest.param(1, id="T=C (pure intra)"),
    pytest.param(4, id="T=4C"),
    pytest.param(10, id="T=10C"),
]


@pytest.mark.parametrize("arm", ARMS, ids=str)
@pytest.mark.parametrize("widths", TOPOLOGIES, ids=str)
@pytest.mark.parametrize("chunks", [p.values[0] for p in CELLS], ids=[p.id for p in CELLS])
def test_rung0_matches_the_rounded_reference(arm, widths, chunks):
    C, BC = arm
    T = C * chunks
    inputs = _inputs(widths, T, seed=11)
    dr, dv, dw = _run(arm, inputs)
    ref = _rounded(*inputs, C)
    for name, got, key, abs_key in (("dR", dr, "dR", "aR"), ("dv", dv, "dv", "adv"),
                                    ("dW", dw, "dW", "adW")):
        bound = _budget(ref, abs_key, T, BC)
        err = (got - ref[key]).abs()
        worst = (err / bound.clamp_min(torch.finfo(torch.float32).tiny)).max().item()
        assert worst <= 1.0, (
            f"rung 0's {name} left its derived budget at arm {arm} widths={widths} "
            f"T={T}: worst |err| / bound = {worst:.3f}; max |err| = {err.max():.3e} "
            f"against a bound of {bound.max():.3e}. The reference rounds every operand "
            f"where the kernel rounds it, so only summation ORDER may separate them.")


@pytest.mark.parametrize("arm", ARMS, ids=str)
@pytest.mark.parametrize("widths", TOPOLOGIES, ids=str)
def test_rung0_matches_the_fp64_autograd_oracle(arm, widths):
    """The kernel against the RECURRENCE, not against the kernel's own association.

    The oracle carries no chunking and no rounding, so what separates it from the
    kernel is bf16's own resolution -- which is why this gate is relative and the
    rounded reference's is not. A kernel that agrees with its own restatement and not
    with this has restated itself.
    """
    C, _ = arm
    T = C * 4
    inputs = _inputs(widths, T, seed=13)
    dr, dv, dw = _run(arm, inputs)
    oR, oV, oW = _fp64_oracle(*inputs)
    for name, got, want in (("dR", dr, oR), ("dv", dv, oV), ("dW", dw, oW)):
        got64, want64 = got.to(torch.float64), want
        rel = (got64 - want64).abs().max() / want64.abs().max()
        assert rel < 3e-2, (
            f"rung 0's {name} disagrees with the fp64 autograd oracle at arm {arm} "
            f"widths={widths}: relative error {rel:.3e}. bf16 operands carry ~4e-3 of "
            f"relative resolution, so a term that is wrong rather than rounded lands "
            f"orders of magnitude outside this.")


@pytest.mark.parametrize("fault", ["no-intra", "exclusive-staircase"])
def test_the_budget_rejects_a_planted_fault(fault):
    """A budget that cannot separate the kernel from a WRONG reference is not gating
    anything. Both faults are the intra staircase's: dropping it, and making it
    exclusive so a token stops seeing its own write."""
    arm, widths = (32, 64), (8, 8)
    C, BC = arm
    T = C * 4
    inputs = _inputs(widths, T, seed=17)
    ref = _rounded(*inputs, C)
    bad = _rounded(*inputs, C, intra=(fault != "no-intra"),
                   exclusive=(fault == "exclusive-staircase"))
    for key, abs_key in (("dR", "aR"), ("dv", "adv"), ("dW", "adW")):
        bound = _budget(ref, abs_key, T, BC)
        assert ((bad[key] - ref[key]).abs() > bound).any(), (
            f"the {key} budget accepts the {fault} reference")


def test_rung0_addresses_every_batch_head():
    """One CTA per (bh), and the value plane is the operator's own `[B, T, H, d_v]`.

    A single-head cell cannot see a head stride at all: every plane's base would be
    correct by accident. Here `B = 2, H = 3` and each (bh) carries its own draw, so a
    base formed from the wrong half of the index lands on another head's data.
    """
    arm, widths, C = (32, 64), (8, 8), 32
    B, H, T = 2, 3, 4 * C
    rows = [_inputs(widths, T, seed=31 + i) for i in range(B * H)]
    r = torch.cat([x[0] for x in rows]).contiguous()
    w = torch.cat([x[1] for x in rows]).contiguous()
    dnum = torch.cat([x[3] for x in rows]).contiguous()
    v = torch.stack([torch.stack([rows[b * H + h][2][0, :, 0] for h in range(H)], dim=1)
                     for b in range(B)]).contiguous()
    m = extension()
    (dr,) = m.chunk_rung0_pass1(w, v, dnum, *arm)
    dv, dw = m.chunk_rung0_pass2(r, w, v, dnum, *arm)
    for bh in range(B * H):
        ref = _rounded(*rows[bh], C)
        for name, got, key, abs_key in (("dR", dr[bh], "dR", "aR"), ("dv", dv[bh], "dv", "adv"),
                                        ("dW", dw[bh], "dW", "adW")):
            bound = _budget(ref, abs_key, T, arm[1])
            worst = ((got - ref[key]).abs()
                     / bound.clamp_min(torch.finfo(torch.float32).tiny)).max().item()
            assert worst <= 1.0, (
                f"rung 0's {name} left its budget at batch-head {bh} of {B * H}: "
                f"worst |err| / bound = {worst:.3f}. Every plane's base is formed from "
                f"the CTA's own (bh), and the value plane's head rides a displacement "
                f"inside a token's row.")


def test_rung0_census_is_spill_free():
    """The rung declares one register floor for both passes; the census is where the
    claim is falsified, per arm and per pass."""
    rows = extension().chunk_rung0_census()
    assert rows, "rung 0 reports no built arm"
    for row in rows:
        arm, pas = tuple(row[:2]), row[2]
        assert row[4] == 0, f"arm {arm} pass {pas} spills {row[4]} bytes"
        assert row[3] <= row[10], (
            f"arm {arm} pass {pas} holds {row[3]} registers against a budget of {row[10]}")
        assert row[6] >= 2, f"arm {arm} pass {pas} staged at depth {row[6]}"
        assert row[7] == 2, (
            f"arm {arm} pass {pas} reaches {row[7]} CTAs: rung 0's residency is the "
            f"forward's and moving it is a carve decision, not a side effect")
