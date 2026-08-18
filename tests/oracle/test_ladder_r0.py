"""THE BACKWARD LADDER, RUNGS 0-1 — the dense two-pass backward, against the oracle.

Vocabulary, defined here and used unqualified below: `C` = chunk tokens; `BC` =
slots per CTA block, which at this rung IS the slot space (`N = BC`, one block);
`T` = tokens; `d_v = 64`; `R`/`Wc` = the per-slot read/write coefficients; `S` = the
forward state `[N, d_v]` and `z` its mass column `[N]`; `dS`/`dz` their adjoints;
`dnum`/`dden` = the readout's two cotangent seeds, one per value channel and one for
the mass. The readout is SELF-NORMALIZING: `y[t] = (R[t] @ S) / (R[t] @ z)`, and
`dden` is what carries the denominator's half of that. Rung b1 carries no write gain
and no digits: those are later rungs.

**WHAT IS GATED, AND AGAINST WHAT.** Three references, and all three are needed:

* the **fp64 autograd oracle** — an eleven-line textbook recurrence in float64,
  differentiated by autograd against the seeded linear form. It knows nothing about
  chunks, so it is the statement of WHAT the kernel computes and cannot agree with a
  kernel that is merely self-consistent;
* the **fp64 RATIO oracle** — the same recurrence read out as `num / den` and
  differentiated end to end from `dy`, with the seeds derived from it and handed to
  the kernel. It is what makes the pair a backward for the self-normalizing readout
  rather than for two unrelated linear functionals;
* the **rounded reference** — the same algebra chunked the way the kernel chunks it,
  rounding every operand where the kernel rounds it (the published state and both
  Grams to bf16, the ones column INSIDE the Gram's one rounding), which is what makes
  a TIGHT derived budget possible.

The teeth are explicit: the budget must REJECT a reference with the intra staircase
dropped, one with the staircase made exclusive, one with the R-b ones column absent,
one with the entry mass absent, and one whose mass column is flattened to its mean.
The last of those is what makes the mass adjoint NON-VACUOUS: a per-slot quantity that
never varies over slots is indistinguishable from a scalar, and the cells here are
drawn so that it varies.
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

#: THE MASS REGIMES. `dden = 0` is the b0 configuration and every mass-bearing term
#: is then identically zero, so it is the regression cell and not a gate on b1; the
#: seeded regime is where the mass adjoint carries load.
MASS = [pytest.param(False, id="read_mass absent"), pytest.param(True, id="read_mass present")]


def _rb(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float32)


def _kron(widths, T, gen, dev) -> torch.Tensor:
    """One duty's DENSE per-slot coefficients: a softmax per level, multiplied over
    the levels exactly as the routed Kronecker product does. Softmax on both duties
    is the dense constructors' own shape, so every slot is on support -- and, because
    the logits are drawn independently per token, the write side's column sums (the
    mass column) differ from slot to slot, which is what the mass adjoint needs to be
    distinguishable from a scalar."""
    out = torch.ones(T, 1, device=dev, dtype=torch.float32)
    for w in widths:
        amp = torch.softmax(torch.randn(T, w, device=dev, generator=gen), dim=-1)
        out = (out.unsqueeze(2) * amp.unsqueeze(1)).reshape(T, -1)
    return out


def _inputs(widths, T, seed, dev="cuda", mass=True):
    g = torch.Generator(device=dev).manual_seed(seed)
    r = _kron(widths, T, g, dev)
    w = _kron(widths, T, g, dev)
    v = torch.randn(1, T, 1, DV, device=dev, generator=g)
    dnum = torch.randn(1, T, DV, device=dev, generator=g)
    dden = (torch.randn(1, T, device=dev, generator=g) if mass
            else torch.zeros(1, T, device=dev))
    return tuple(t.to(torch.bfloat16).contiguous() for t in (r.unsqueeze(0), w.unsqueeze(0),
                                                             v, dnum, dden))


def _run(arm, inputs):
    c, bc = arm
    r, w, v, dnum, dden = inputs
    m = extension()
    (dr,) = m.chunk_rung0_pass1(w, v, dnum, dden, c, bc)
    dv, dw = m.chunk_rung0_pass2(r, w, v, dnum, dden, c, bc)
    return dr[0], dv[0], dw[0]


# ---------------------------------------------------------------- the oracle
def _recurrence(R, Wc, V):
    """THE ELEVEN-LINE RECURRENCE. `S` and its mass column `z` accumulate the write
    side, and a read at `t` sees its own write -- the inclusive read-after-write the
    forward performs."""
    T, N = R.shape
    S = torch.zeros(N, DV, device=R.device, dtype=R.dtype)
    z = torch.zeros(N, device=R.device, dtype=R.dtype)
    num, den = [], []
    for t in range(T):
        S = S + Wc[t].unsqueeze(1) * V[t].unsqueeze(0)
        z = z + Wc[t]
        num.append(R[t] @ S)
        den.append(R[t] @ z)
    return torch.stack(num), torch.stack(den)


def _fp64_oracle(r, w, v, dnum, dden):
    """The recurrence against the SEEDED LINEAR FORM the kernel is handed."""
    R = r[0].to(torch.float64).requires_grad_(True)
    Wc = w[0].to(torch.float64).requires_grad_(True)
    V = v[0, :, 0].to(torch.float64).requires_grad_(True)
    num, den = _recurrence(R, Wc, V)
    loss = (num * dnum[0].to(torch.float64)).sum() + (den * dden[0].to(torch.float64)).sum()
    return torch.autograd.grad(loss, (R, Wc, V))


def _fp64_ratio_oracle(r, w, v, dy):
    """THE SELF-NORMALIZING READOUT, differentiated end to end.

    The kernel is not handed `dy`: it is handed the two seeds the readout's own
    backward produces. So this returns both the gradients AND the seeds, and the gate
    is that feeding those seeds to the kernel reproduces these gradients. Nothing here
    knows the chunked form, and nothing in the kernel knows the ratio.
    """
    R = r[0].to(torch.float64).requires_grad_(True)
    Wc = w[0].to(torch.float64).requires_grad_(True)
    V = v[0, :, 0].to(torch.float64).requires_grad_(True)
    num, den = _recurrence(R, Wc, V)
    y = num / den.unsqueeze(1)
    loss = (y * dy[0].to(torch.float64)).sum()
    grads = torch.autograd.grad(loss, (R, Wc, V))
    dnum = dy[0].to(torch.float64) / den.detach().unsqueeze(1)
    dden = -(dy[0].to(torch.float64) * y.detach()).sum(1) / den.detach()
    return grads, dnum, dden


# ------------------------------------------------------- the rounded reference
def _rounded(r, w, v, dnum, dden, C, *, intra=True, exclusive=False, ones=True,
             entry_mass=True, flat_mass=False):
    """The kernel's own association, with every stored operand rounded where the
    kernel rounds it, plus the magnitude accumulation the budget is derived from.

    `ones`, `entry_mass` and `flat_mass` are the mass column's planted faults: the R-b
    ones column removed from the transposed staircase, the entry mass dropped from the
    two per-slot terms, and the mass column replaced by its own mean over the slots.
    """
    dev = r.device
    R, Wc = _rb(r)[0], _rb(w)[0]
    V, dn = _rb(v)[0, :, 0], _rb(dnum)[0]
    dd = _rb(dden)[0]
    T, N = R.shape
    tok = torch.arange(C, device=dev)
    #: rows are READ tokens and columns WRITE tokens, and a write is kept at or
    #: before the read; the exclusive form is the planted fault, not a variant.
    stair = ((tok.unsqueeze(1) > tok.unsqueeze(0)) if exclusive
             else (tok.unsqueeze(1) >= tok.unsqueeze(0))).float()

    def flat(col):
        return col.mean().expand_as(col) if flat_mass else col

    dR = torch.zeros(T, N, device=dev)
    aR = torch.zeros(T, N, device=dev)
    S = torch.zeros(N, DV, device=dev)
    aS = torch.zeros(N, DV, device=dev)
    z = torch.zeros(N, device=dev)
    az = torch.zeros(N, device=dev)
    for c0 in range(0, T, C):
        idx = slice(c0, c0 + C)
        Sb = _rb(S)
        #: the ones column rides INSIDE the Gram's single rounding, because the
        #: kernel adds it to the accumulator before the store converts.
        col = dd[idx].unsqueeze(1) if ones else 0.0
        Gd = _rb(dn[idx] @ V[idx].T + col) * stair
        za = flat(z)
        mass = dd[idx].unsqueeze(1) * za.unsqueeze(0) if entry_mass else 0.0
        amass = dd[idx].abs().unsqueeze(1) * flat(az).unsqueeze(0)
        dR[idx] = dn[idx] @ Sb.T + mass + (Gd @ Wc[idx] if intra else 0.0)
        aR[idx] = dn[idx].abs() @ aS.T + amass + Gd.abs() @ Wc[idx].abs()
        S = S + Wc[idx].T @ V[idx]
        aS = aS + Wc[idx].abs().T @ V[idx].abs()
        z = z + Wc[idx].sum(0)
        az = az + Wc[idx].abs().sum(0)

    dv = torch.zeros(T, DV, device=dev)
    adv = torch.zeros(T, DV, device=dev)
    dW = torch.zeros(T, N, device=dev)
    adW = torch.zeros(T, N, device=dev)
    dS = torch.zeros(N, DV, device=dev)
    adS = torch.zeros(N, DV, device=dev)
    dz = torch.zeros(N, device=dev)
    adz = torch.zeros(N, device=dev)
    for c0 in range(T - C, -1, -C):
        idx = slice(c0, c0 + C)
        dSb = _rb(dS)
        Gu = _rb(R[idx] @ Wc[idx].T) * stair
        col = dd[idx].unsqueeze(1) if ones else 0.0
        Gd = _rb(dn[idx] @ V[idx].T + col) * stair
        #: `dv` is a value-channel quantity and the mass column does not enter it.
        dv[idx] = Wc[idx] @ dSb + (Gu.T @ dn[idx] if intra else 0.0)
        adv[idx] = Wc[idx].abs() @ adS + Gu.abs().T @ dn[idx].abs()
        dza = flat(dz).unsqueeze(0) if entry_mass else 0.0
        dW[idx] = V[idx] @ dSb.T + dza + (Gd.T @ R[idx] if intra else 0.0)
        adW[idx] = V[idx].abs() @ adS.T + flat(adz).unsqueeze(0) + Gd.abs().T @ R[idx].abs()
        dS = dS + R[idx].T @ dn[idx]
        adS = adS + R[idx].abs().T @ dn[idx].abs()
        dz = dz + R[idx].T @ dd[idx]
        adz = adz + R[idx].abs().T @ dd[idx].abs()
    return {"dR": dR, "dv": dv, "dW": dW, "aR": aR, "adv": adv, "adW": adW,
            "z": z, "dz": dz}


def _budget(ref, abs_key, T, N):
    """The derived per-element budget: one bf16 rounding on each stored operand,
    plus fp32 accumulation over the terms the sum actually runs."""
    return (ULP_BF16_PAIR + (T + N + DV) * EPS32) * ref[abs_key]


def _check(got_all, ref, T, BC, where):
    for name, got, key, abs_key in (("dR", got_all[0], "dR", "aR"),
                                    ("dv", got_all[1], "dv", "adv"),
                                    ("dW", got_all[2], "dW", "adW")):
        bound = _budget(ref, abs_key, T, BC)
        err = (got - ref[key]).abs()
        worst = (err / bound.clamp_min(torch.finfo(torch.float32).tiny)).max().item()
        assert worst <= 1.0, (
            f"the ladder's {name} left its derived budget at {where}: worst "
            f"|err| / bound = {worst:.3f}; max |err| = {err.max():.3e} against a bound of "
            f"{bound.max():.3e}. The reference rounds every operand where the kernel "
            f"rounds it, so only summation ORDER may separate them.")


CELLS = [
    #: ONE CHUNK: the entry state, the entry mass and both adjoints are exactly zero,
    #: so every gradient here is ENTIRELY the intra staircase and the R-b ones column.
    #: The ENTRY-mass terms are vacuous here by construction, which is why the fault
    #: plants below run at four chunks.
    pytest.param(1, id="T=C (pure intra)"),
    pytest.param(4, id="T=4C"),
    pytest.param(10, id="T=10C"),
]


@pytest.mark.parametrize("arm", ARMS, ids=str)
@pytest.mark.parametrize("widths", TOPOLOGIES, ids=str)
@pytest.mark.parametrize("mass", MASS)
@pytest.mark.parametrize("chunks", [p.values[0] for p in CELLS], ids=[p.id for p in CELLS])
def test_the_ladder_matches_the_rounded_reference(arm, widths, mass, chunks):
    C, BC = arm
    T = C * chunks
    inputs = _inputs(widths, T, seed=11, mass=mass)
    ref = _rounded(*inputs, C)
    _check(_run(arm, inputs), ref, T, BC,
           f"arm {arm} widths={widths} mass={mass} T={T}")


@pytest.mark.parametrize("arm", ARMS, ids=str)
@pytest.mark.parametrize("widths", TOPOLOGIES, ids=str)
@pytest.mark.parametrize("mass", MASS)
def test_the_ladder_matches_the_fp64_autograd_oracle(arm, widths, mass):
    """The kernel against the RECURRENCE, not against the kernel's own association.

    The oracle carries no chunking and no rounding, so what separates it from the
    kernel is bf16's own resolution -- which is why this gate is relative and the
    rounded reference's is not. A kernel that agrees with its own restatement and not
    with this has restated itself.
    """
    C, _ = arm
    T = C * 4
    inputs = _inputs(widths, T, seed=13, mass=mass)
    dr, dv, dw = _run(arm, inputs)
    oR, oW, oV = _fp64_oracle(*inputs)
    for name, got, want in (("dR", dr, oR), ("dv", dv, oV), ("dW", dw, oW)):
        rel = (got.to(torch.float64) - want).abs().max() / want.abs().max()
        assert rel < 3e-2, (
            f"the ladder's {name} disagrees with the fp64 autograd oracle at arm {arm} "
            f"widths={widths} mass={mass}: relative error {rel:.3e}. bf16 operands carry "
            f"~4e-3 of relative resolution, so a term that is wrong rather than rounded "
            f"lands orders of magnitude outside this.")


@pytest.mark.parametrize("arm", ARMS, ids=str)
@pytest.mark.parametrize("widths", TOPOLOGIES, ids=str)
def test_the_ladder_is_the_self_normalizing_readouts_backward(arm, widths):
    """THE RATIO, END TO END. `y = num / den` is differentiated by autograd in fp64,
    the two seeds are derived from that same readout, and the kernel is handed only
    the seeds. Nothing but the mass column and the ones column can carry `den`'s half
    of the gradient, so this is the gate that fails if either is absent.
    """
    C, _ = arm
    T = C * 4
    r, w, v, _, _ = _inputs(widths, T, seed=19)
    g = torch.Generator(device="cuda").manual_seed(919)
    dy = torch.randn(1, T, DV, device="cuda", generator=g)
    (oR, oW, oV), dnum, dden = _fp64_ratio_oracle(r, w, v, dy)
    seeds = (dnum.unsqueeze(0).to(torch.bfloat16).contiguous(),
             dden.unsqueeze(0).to(torch.bfloat16).contiguous())
    dr, dv, dw = _run(arm, (r, w, v, *seeds))
    for name, got, want in (("dR", dr, oR), ("dv", dv, oV), ("dW", dw, oW)):
        rel = (got.to(torch.float64) - want).abs().max() / want.abs().max()
        assert rel < 5e-2, (
            f"the ladder's {name} is not the self-normalizing readout's backward at arm "
            f"{arm} widths={widths}: relative error {rel:.3e}. The seeds are the readout's "
            f"own, so a missing mass term shows up as a whole term rather than as rounding.")


def test_the_mass_column_varies_over_the_slots():
    """NON-VACUITY, stated as a property of the cell rather than assumed.

    Both mass adjoints are PER-SLOT. On a cell whose mass column happened to be flat
    they would be indistinguishable from a scalar, and every gate below would pass
    against a kernel that broadcast one number. The dense Kronecker draw makes them
    vary; this is where that is checked, and `test_the_budget_rejects_a_planted_fault`
    is where it is made to bite.
    """
    ref = _rounded(*_inputs((8, 8), 128, seed=17), 32)
    for key in ("z", "dz"):
        col = ref[key]
        assert (col.std() / col.abs().mean()).item() > 0.05, (
            f"the cell's {key} is effectively flat over the slots, so the mass adjoint "
            f"it gates is vacuous")


FAULTS = ["no-intra", "exclusive-staircase", "no-ones-column", "no-entry-mass", "flat-mass"]


@pytest.mark.parametrize("fault", FAULTS)
def test_the_budget_rejects_a_planted_fault(fault):
    """A budget that cannot separate the kernel from a WRONG reference is not gating
    anything. Two faults are the intra staircase's; three are the mass column's -- the
    R-b ones column dropped from the transposed staircase (15 of 18 entries wrong
    without it), the entry mass dropped from both per-slot terms, and the per-slot mass
    replaced by its mean, which is the fault a flat cell could never see.

    FOUR chunks, not one: at a single chunk every ENTRY quantity is identically zero
    and two of these faults are vacuous.
    """
    arm, widths = (32, 64), (8, 8)
    C, BC = arm
    T = C * 4
    inputs = _inputs(widths, T, seed=17)
    ref = _rounded(*inputs, C)
    bad = _rounded(*inputs, C, intra=(fault != "no-intra"),
                   exclusive=(fault == "exclusive-staircase"),
                   ones=(fault != "no-ones-column"),
                   entry_mass=(fault != "no-entry-mass"),
                   flat_mass=(fault == "flat-mass"))
    #: `dv` carries no mass term at all, so the mass faults must NOT move it -- which
    #: is itself the statement that the mass column stays out of the value channels.
    keys = (("dR", "aR"), ("dv", "adv"), ("dW", "adW"))
    for key, abs_key in keys:
        bound = _budget(ref, abs_key, T, BC)
        moved = ((bad[key] - ref[key]).abs() > bound).any().item()
        if key == "dv" and fault in ("no-ones-column", "no-entry-mass", "flat-mass"):
            assert not moved, f"the {fault} reference moved dv, which carries no mass term"
        else:
            assert moved, f"the {key} budget accepts the {fault} reference"


def test_the_ladder_addresses_every_batch_head():
    """One CTA per (bh), and the value plane is the operator's own `[B, T, H, d_v]`.

    A single-head cell cannot see a head stride at all: every plane's base would be
    correct by accident. Here `B = 2, H = 3` and each (bh) carries its own draw, so a
    base formed from the wrong half of the index lands on another head's data. The
    mass seed is a `[BH, L]` plane and rides the same test.
    """
    arm, widths, C = (32, 64), (8, 8), 32
    B, H, T = 2, 3, 4 * C
    rows = [_inputs(widths, T, seed=31 + i) for i in range(B * H)]
    r = torch.cat([x[0] for x in rows]).contiguous()
    w = torch.cat([x[1] for x in rows]).contiguous()
    dnum = torch.cat([x[3] for x in rows]).contiguous()
    dden = torch.cat([x[4] for x in rows]).contiguous()
    v = torch.stack([torch.stack([rows[b * H + h][2][0, :, 0] for h in range(H)], dim=1)
                     for b in range(B)]).contiguous()
    m = extension()
    (dr,) = m.chunk_rung0_pass1(w, v, dnum, dden, *arm)
    dv, dw = m.chunk_rung0_pass2(r, w, v, dnum, dden, *arm)
    for bh in range(B * H):
        _check((dr[bh], dv[bh], dw[bh]), _rounded(*rows[bh], C), T, arm[1],
               f"batch-head {bh} of {B * H}")


def test_the_ladder_census_is_spill_free():
    """The rung declares one register floor for both passes; the census is where the
    claim is falsified, per arm and per pass."""
    rows = extension().chunk_rung0_census()
    assert rows, "the ladder reports no built arm"
    for row in rows:
        arm, pas = tuple(row[:2]), row[2]
        assert row[4] == 0, f"arm {arm} pass {pas} spills {row[4]} bytes"
        assert row[3] <= row[10], (
            f"arm {arm} pass {pas} holds {row[3]} registers against a budget of {row[10]}")
        assert row[6] >= 2, f"arm {arm} pass {pas} staged at depth {row[6]}"
        assert row[7] == 2, (
            f"arm {arm} pass {pas} reaches {row[7]} CTAs: the rung's residency is the "
            f"forward's and moving it is a carve decision, not a side effect")
