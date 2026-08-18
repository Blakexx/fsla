"""THE BACKWARD LADDER, RUNGS 0-3 — the ROUTED two-pass backward, against the oracle.

Vocabulary, defined here and used unqualified below: `C` = chunk tokens; `BC` =
slots per CTA block; `D` = routing levels and `Bw` one uniform level width, so
`N = Bw**D` is the slot space and the block lattice `N / BC` covers it; `pr`/`pw` =
the PER-LEVEL read/write amplitudes, `[T, D*Bw]`, one packed block of `Bw` columns
per level; `R[i,s] = prod_l pr[i, l*Bw + d_l(s)]` / `Wc` = the per-slot coefficients
those factor into; `d_pr`/`d_pw` = the PER-LEVEL digit gradients the kernel emits;
`T` = tokens; `d_v = 64`; `S` = the forward state `[N, d_v]` and `z` its mass column
`[N]`; `dS`/`dz` their adjoints; `dnum`/`dden` = the readout's two cotangent seeds,
one per value channel and one for the mass. The readout is SELF-NORMALIZING:
`y[t] = (R[t] @ S) / (R[t] @ z)`, and `dden` is what carries the denominator's half
of that. The ladder carries no write gain and no compaction: those are later rungs.

RUNG b3 IS WHERE THE OPERAND CHANGED, and with it what "the same answer" means. The
kernel no longer sees a slot: it consumes the per-level factors and emits the per-
level digit gradients, so a bit-identity control against rung b2's binary is not
available even where the schedule did not move -- the operand FORMAT moved, and a
different summation order over the same rationals is the honest expectation. The
control here is therefore the ORACLE, at every cell, and not `torch.equal`.

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

RUNG b2 ADDED THE LATTICE and b3 puts TWO more outputs on it. Every cell here runs at
its topology's whole `N`, so most run MANY blocks; `dv` contracts over the slot axis
and both digit gradients over the slots that share a digit, so all three are fan-ins
across the lattice. Their non-vacuity rows assert that every block's partial is a
material share, because a gate taken on a cell where one block carried the sum would
pass against a kernel that stored instead of adding.
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

#: THE BUILT LADDER ARMS, `(C, BC, D, Bw)`. Rung b3 made the topology a COMPILED
#: axis -- the kernel's digit geometry is the arm's -- so an arm names one slot space
#: and one block lattice, and the list is `tools/gen_shards.py:LADDER_R0_ARMS`. The
#: six that are also `SCAN_ARMS` are the head-to-head set: the ladder and the shipped
#: reverse pass run the same geometry, so their walls compare. `(16, 64, 2, 16)` is
#: the second point on the CHUNK axis.
ARMS = [(32, 128, 2, 64), (32, 128, 2, 256), (32, 64, 2, 8), (32, 64, 2, 16),
        (32, 64, 1, 64), (32, 64, 1, 256), (16, 64, 2, 16)]

#: THE ONE-BLOCK ARMS, named rather than derived, because they are the control the
#: kernel-side lattice fault is read against: `N == BC` is where a block base of zero
#: is right by accident.
ONE_BLOCK = [a for a in ARMS if a[3] ** a[2] == a[1]]

#: THE MASS REGIMES. `dden = 0` is the b0 configuration and every mass-bearing term
#: is then identically zero, so it is the regression cell and not a gate on b1; the
#: seeded regime is where the mass adjoint carries load.
MASS = [pytest.param(False, id="read_mass absent"), pytest.param(True, id="read_mass present")]


def _rb(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float32)


def _digits(D, Bw, dev):
    """A slot's digit at each level, as the leaf-order law spells it: level `l` sits
    at place value `Bw**(D-1-l)`."""
    slot = torch.arange(Bw ** D, device=dev)
    return [(slot // (Bw ** (D - 1 - l))) % Bw for l in range(D)]


def _kron(packed, D, Bw):
    """The per-slot coefficient the kernel expands in kernel: one amplitude per level
    at that slot's digit, multiplied over the levels."""
    out = torch.ones(packed.shape[0], Bw ** D, device=packed.device, dtype=packed.dtype)
    for l, dig in enumerate(_digits(D, Bw, packed.device)):
        out = out * packed[:, l * Bw + dig]
    return out


def _reduce_digits(dX, packed, D, Bw):
    """(A.9)/(A.10): the per-level digit gradient. Level `l`'s digit `d` collects the
    per-slot gradient of every slot whose `l`-digit is `d`, weighted by the LEAVE-ONE-
    OUT product over the OTHER levels -- never by `X / p_l`, which is `0/0` off the
    support with no epsilon that does not invent a gradient."""
    digs = _digits(D, Bw, dX.device)
    out = torch.zeros(dX.shape[0], D * Bw, device=dX.device, dtype=dX.dtype)
    for l in range(D):
        loo = torch.ones_like(dX)
        for l2 in range(D):
            if l2 != l:
                loo = loo * packed[:, l2 * Bw + digs[l2]]
        out[:, l * Bw:(l + 1) * Bw] = torch.zeros_like(out[:, :Bw]).index_add_(
            1, digs[l], dX * loo)
    return out


def _amps(D, Bw, T, gen, dev) -> torch.Tensor:
    """One duty's PACKED amplitudes: a softmax per level, laid out level-major exactly
    as the kernel's `[T, D*Bw]` row is. Softmax on both duties is the dense
    constructors' own shape, so every slot is on support -- and, because the logits are
    drawn independently per token, the write side's column sums (the mass column)
    differ from slot to slot, which is what the mass adjoint needs to be
    distinguishable from a scalar."""
    return torch.cat([torch.softmax(torch.randn(T, Bw, device=dev, generator=gen), dim=-1)
                      for _ in range(D)], dim=1)


def _inputs(arm, T, seed, dev="cuda", mass=True):
    _, _, D, Bw = arm
    g = torch.Generator(device=dev).manual_seed(seed)
    pr = _amps(D, Bw, T, g, dev)
    pw = _amps(D, Bw, T, g, dev)
    v = torch.randn(1, T, 1, DV, device=dev, generator=g)
    dnum = torch.randn(1, T, DV, device=dev, generator=g)
    dden = (torch.randn(1, T, device=dev, generator=g) if mass
            else torch.zeros(1, T, device=dev))
    return tuple(t.to(torch.bfloat16).contiguous() for t in (pr.unsqueeze(0), pw.unsqueeze(0),
                                                             v, dnum, dden))


def _run(arm, inputs):
    pr, pw, v, dnum, dden = inputs
    m = extension()
    (dpr,) = m.chunk_rung0_pass1(pr, pw, v, dnum, dden, *arm)
    dv, dpw = m.chunk_rung0_pass2(pr, pw, v, dnum, dden, *arm)
    return dpr[0], dv[0], dpw[0]

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


def _fp64_oracle(arm, pr, pw, v, dnum, dden):
    """The recurrence against the SEEDED LINEAR FORM the kernel is handed, DIFFERENTIATED
    THROUGH THE FACTORIZED FORM.

    This is the gate rung b3 exists for. The autograd variables are the PER-LEVEL
    amplitudes, and the Kronecker product is inside the graph, so what comes back is the
    digit gradient itself -- (A.9)/(A.10) obtained by differentiation rather than by
    restating the kernel's own reduction. A leave-one-out product written as `X / p_l`,
    a missing block offset in the fan-in column, or a digit collected at the wrong level
    all land here as whole terms.
    """
    _, _, D, Bw = arm
    PR = pr[0].to(torch.float64).requires_grad_(True)
    PW = pw[0].to(torch.float64).requires_grad_(True)
    V = v[0, :, 0].to(torch.float64).requires_grad_(True)
    num, den = _recurrence(_kron(PR, D, Bw), _kron(PW, D, Bw), V)
    loss = (num * dnum[0].to(torch.float64)).sum() + (den * dden[0].to(torch.float64)).sum()
    return torch.autograd.grad(loss, (PR, PW, V))


def _fp64_ratio_oracle(arm, pr, pw, v, dy):
    """THE SELF-NORMALIZING READOUT, differentiated end to end, through the factorized
    form as well.

    The kernel is not handed `dy`: it is handed the two seeds the readout's own backward
    produces. So this returns both the digit gradients AND the seeds, and the gate is
    that feeding those seeds to the kernel reproduces these gradients. Nothing here knows
    the chunked form or the block lattice, and nothing in the kernel knows the ratio.
    """
    _, _, D, Bw = arm
    PR = pr[0].to(torch.float64).requires_grad_(True)
    PW = pw[0].to(torch.float64).requires_grad_(True)
    V = v[0, :, 0].to(torch.float64).requires_grad_(True)
    num, den = _recurrence(_kron(PR, D, Bw), _kron(PW, D, Bw), V)
    y = num / den.unsqueeze(1)
    loss = (y * dy[0].to(torch.float64)).sum()
    grads = torch.autograd.grad(loss, (PR, PW, V))
    dnum = dy[0].to(torch.float64) / den.detach().unsqueeze(1)
    dden = -(dy[0].to(torch.float64) * y.detach()).sum(1) / den.detach()
    return grads, dnum, dden


# ------------------------------------------------------- the rounded reference
def _rounded(arm, pr, pw, v, dnum, dden, *, intra=True, exclusive=False, ones=True,
             entry_mass=True, flat_mass=False, loo_level=None, slot_mask=None):
    """The kernel's own association, with every stored operand rounded where the kernel
    rounds it, plus the magnitude accumulation the budget is derived from.

    The per-slot gradients are formed exactly as rung b2's were -- the algebra did not
    move -- and then REDUCED to digits with the same bf16 amplitudes the kernel reads,
    which is the only place b3 enters. `ones`, `entry_mass` and `flat_mass` are the mass
    column's planted faults; `loo_level` is b3's own, and it is the leave-one-out product
    taken from the WRONG level, which is what the digit reduction can get wrong while
    every per-slot term stays right.
    """
    C, BC, D, Bw = arm
    dev = pr.device
    PR, PW = _rb(pr)[0], _rb(pw)[0]
    R, Wc = _kron(PR, D, Bw), _kron(PW, D, Bw)
    if slot_mask is not None:
        R, Wc = R * slot_mask, Wc * slot_mask
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

    #: THE DIGIT REDUCTION, and the FAULT that swaps which level is left out. At depth
    #: two the leave-one-out factor IS the other level's amplitude, so the swap is the
    #: exact defect (A.10)'s own gate exchanges.
    swap = PR if loo_level == "read" else PW if loo_level == "write" else None
    return {"d_pr": _reduce_digits(dR, swap if swap is not None else PR, D, Bw),
            "dv": dv,
            "d_pw": _reduce_digits(dW, swap if swap is not None else PW, D, Bw),
            "a_pr": _reduce_digits(aR, PR.abs(), D, Bw), "adv": adv,
            "a_pw": _reduce_digits(adW, PW.abs(), D, Bw),
            #: the PER-SLOT gradients survive the return so the fan-in's non-vacuity row
            #: can restrict them to one block and reduce that; the kernel never stores
            #: them, and neither does any gate that is not about the lattice.
            "slot_pr": dR, "slot_pw": dW,
            "z": z, "dz": dz}


def _budget(ref, abs_key, T, N):
    """The derived per-element budget: one bf16 rounding on each stored operand, plus
    fp32 accumulation over the terms the sum actually runs -- which at b3 includes the
    digit reduction's own fan-in over the `N / Bw` slots that share a digit, already
    inside the `N` term."""
    return (ULP_BF16_PAIR + (T + N + DV) * EPS32) * ref[abs_key]


def _check(got_all, ref, T, N, where):
    for name, got, key, abs_key in (("d_pr", got_all[0], "d_pr", "a_pr"),
                                    ("dv", got_all[1], "dv", "adv"),
                                    ("d_pw", got_all[2], "d_pw", "a_pw")):
        bound = _budget(ref, abs_key, T, N)
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
@pytest.mark.parametrize("mass", MASS)
@pytest.mark.parametrize("chunks", [p.values[0] for p in CELLS], ids=[p.id for p in CELLS])
def test_the_ladder_matches_the_rounded_reference(arm, mass, chunks):
    C, BC, D, Bw = arm
    T, N = C * chunks, Bw ** D
    inputs = _inputs(arm, T, seed=11, mass=mass)
    ref = _rounded(arm, *inputs)
    _check(_run(arm, inputs), ref, T, N, f"arm {arm} blocks={N // BC} mass={mass} T={T}")


@pytest.mark.parametrize("arm", ARMS, ids=str)
@pytest.mark.parametrize("mass", MASS)
def test_the_ladder_matches_the_fp64_autograd_oracle(arm, mass):
    """The kernel against the RECURRENCE differentiated through the factorized form, not
    against the kernel's own restatement of its reduction.

    The oracle carries no chunking, no lattice and no rounding, so what separates it from
    the kernel is bf16's own resolution -- which is why this gate is relative and the
    rounded reference's is not. A kernel that agrees with its own restatement and not
    with this has restated itself.
    """
    T = arm[0] * 4
    inputs = _inputs(arm, T, seed=13, mass=mass)
    dpr, dv, dpw = _run(arm, inputs)
    oPR, oPW, oV = _fp64_oracle(arm, *inputs)
    for name, got, want in (("d_pr", dpr, oPR), ("dv", dv, oV), ("d_pw", dpw, oPW)):
        rel = (got.to(torch.float64) - want).abs().max() / want.abs().max()
        assert rel < 3e-2, (
            f"the ladder's {name} disagrees with the fp64 autograd oracle at arm {arm} "
            f"mass={mass}: relative error {rel:.3e}. bf16 operands carry ~4e-3 of relative "
            f"resolution, so a term that is wrong rather than rounded lands orders of "
            f"magnitude outside this.")


@pytest.mark.parametrize("arm", ARMS, ids=str)
def test_the_ladder_is_the_self_normalizing_readouts_backward(arm):
    """THE RATIO, END TO END. `y = num / den` is differentiated by autograd in fp64
    through the Kronecker product, the two seeds are derived from that same readout, and
    the kernel is handed only the seeds. Nothing but the mass column and the ones column
    can carry `den`'s half of the gradient, so this is the gate that fails if either is
    absent.
    """
    T = arm[0] * 4
    pr, pw, v, _, _ = _inputs(arm, T, seed=19)
    g = torch.Generator(device="cuda").manual_seed(919)
    dy = torch.randn(1, T, DV, device="cuda", generator=g)
    (oPR, oPW, oV), dnum, dden = _fp64_ratio_oracle(arm, pr, pw, v, dy)
    seeds = (dnum.unsqueeze(0).to(torch.bfloat16).contiguous(),
             dden.unsqueeze(0).to(torch.bfloat16).contiguous())
    dpr, dv, dpw = _run(arm, (pr, pw, v, *seeds))
    for name, got, want in (("d_pr", dpr, oPR), ("dv", dv, oV), ("d_pw", dpw, oPW)):
        rel = (got.to(torch.float64) - want).abs().max() / want.abs().max()
        assert rel < 5e-2, (
            f"the ladder's {name} is not the self-normalizing readout's backward at arm "
            f"{arm}: relative error {rel:.3e}. The seeds are the readout's own, so a "
            f"missing mass term shows up as a whole term rather than as rounding.")


def test_the_mass_column_varies_over_the_slots():
    """NON-VACUITY, stated as a property of the cell rather than assumed.

    Both mass adjoints are PER-SLOT. On a cell whose mass column happened to be flat
    they would be indistinguishable from a scalar, and every gate below would pass
    against a kernel that broadcast one number. The Kronecker draw makes them vary;
    this is where that is checked, and `test_the_budget_rejects_a_planted_fault` is
    where it is made to bite.
    """
    arm = (32, 64, 2, 8)
    ref = _rounded(arm, *_inputs(arm, 128, seed=17))
    for key in ("z", "dz"):
        col = ref[key]
        assert (col.std() / col.abs().mean()).item() > 0.05, (
            f"the cell's {key} is effectively flat over the slots, so the mass adjoint "
            f"it gates is vacuous")


@pytest.mark.parametrize("arm", [a for a in ARMS if a[3] ** a[2] > a[1]], ids=str)
def test_the_fan_in_has_more_than_one_live_contributor(arm):
    """NON-VACUITY OF THE SPLIT-K FAN-IN, as a property of the cell.

    Rung b2 had ONE fan-in target (`dv`); b3 has three, because a DIGIT is shared across
    the block lattice where a SLOT was not. A gate run on a cell where one block carried
    a sum would pass against a kernel that STORED instead of adding, so this asserts of
    the cell itself that every block's partial is a material share of all three totals --
    and `test_the_ladder_matches_the_rounded_reference` at the same arm is then the
    statement that they were in fact summed.
    """
    C, BC, D, Bw = arm
    T, N = C * 4, Bw ** D
    nblk = N // BC
    assert nblk > 1, "this cell is not multi-block and cannot gate a fan-in"
    inputs = _inputs(arm, T, seed=23)
    ref = _rounded(arm, *inputs)
    #: a block owns a CONTIGUOUS `BC`-run of the slot space, and both the digit reduction
    #: and `dv`'s slot contraction are LINEAR in the per-slot gradient, so a block's
    #: partial is that reduction restricted to its own slots.
    PR = _rb(inputs[0])[0]
    for b in range(nblk):
        mask = torch.zeros(N, device=PR.device)
        mask[b * BC:(b + 1) * BC] = 1.0
        part = _rounded(arm, *inputs, slot_mask=mask)
        for key in ("d_pr", "dv", "d_pw"):
            share = (part[key].abs().sum() / ref[key].abs().sum()).item()
            #: THE BAR IS THE EVEN SHARE, not an absolute one: at 512 blocks an even
            #: contributor carries 0.2 % of the total, and an absolute 2 % floor would
            #: declare a perfectly healthy lattice vacuous. What makes the fan-in real is
            #: that no block is NEGLIGIBLE against its own share of the sum.
            assert share > 0.3 / nblk, (
                f"block {b} of {nblk} contributes {share:.5f} of {key} at arm {arm}, "
                f"against an even share of {1 / nblk:.5f}: the fan-in is vacuous here")


FAULTS = ["no-intra", "exclusive-staircase", "no-ones-column", "no-entry-mass", "flat-mass",
          "swapped-leave-one-out"]


@pytest.mark.parametrize("fault", FAULTS)
def test_the_budget_rejects_a_planted_fault(fault):
    """A budget that cannot separate the kernel from a WRONG reference is not gating
    anything. Two faults are the intra staircase's; three are the mass column's -- the
    R-b ones column dropped from the transposed staircase (15 of 18 entries wrong
    without it), the entry mass dropped from both per-slot terms, and the per-slot mass
    replaced by its mean, which is the fault a flat cell could never see. THE SIXTH IS
    RUNG b3'S OWN: the leave-one-out product taken from the wrong duty's amplitudes,
    which leaves every per-slot term exactly right and only the digit reduction wrong.

    FOUR chunks, not one: at a single chunk every ENTRY quantity is identically zero
    and two of these faults are vacuous.
    """
    arm = (32, 64, 2, 8)
    C, BC, D, Bw = arm
    T = C * 4
    inputs = _inputs(arm, T, seed=17)
    ref = _rounded(arm, *inputs)
    bad = _rounded(arm, *inputs, intra=(fault != "no-intra"),
                   exclusive=(fault == "exclusive-staircase"),
                   ones=(fault != "no-ones-column"),
                   entry_mass=(fault != "no-entry-mass"),
                   flat_mass=(fault == "flat-mass"),
                   loo_level=("read" if fault == "swapped-leave-one-out" else None))
    #: `dv` carries no mass term and no digit reduction at all, so those faults must NOT
    #: move it -- which is itself the statement that neither leaks into the value channels.
    keys = (("d_pr", "a_pr"), ("dv", "adv"), ("d_pw", "a_pw"))
    for key, abs_key in keys:
        bound = _budget(ref, abs_key, T, Bw ** D)
        moved = ((bad[key] - ref[key]).abs() > bound).any().item()
        if key == "dv" and fault in ("no-ones-column", "no-entry-mass", "flat-mass",
                                     "swapped-leave-one-out"):
            assert not moved, f"the {fault} reference moved dv, which carries neither term"
        elif key == "d_pr" and fault == "swapped-leave-one-out":
            #: the swap hands the READ reduction its own amplitudes, so only the write
            #: side moves; the row is kept so the asymmetry is asserted rather than assumed.
            assert not moved, "the read side's leave-one-out was not the one swapped"
        else:
            assert moved, f"the {key} budget accepts the {fault} reference"


@pytest.mark.parametrize("arm", [(32, 64, 2, 16), (32, 128, 2, 64), (16, 64, 2, 16)], ids=str)
def test_the_ladder_addresses_every_batch_head(arm):
    """The grid is `(nblk, BH)`, and BOTH extents are gated here, together.

    A single-head cell cannot see a head stride at all: every plane's base would be
    correct by accident. Here `B = 2, H = 3` and each (bh) carries its own draw, so a
    base formed from the wrong half of the index lands on another head's data. The
    mass seed is a `[BH, L]` plane and rides the same test.

    The MULTI-BLOCK rows are why this is parametrized. `blockIdx.x` selects the slot
    block and `blockIdx.y` the batch-head, and the two are combined into one linear
    plane offset -- a swap, or a stride taken against `BC` where it should be `D*Bw`,
    is invisible while either extent is one. Every other cell in this file runs at
    `BH = 1`, so without these rows the lattice and the head stride would never have
    been wrong at the same time.
    """
    C, BC, D, Bw = arm
    B, H, T = 2, 3, 4 * C
    rows = [_inputs(arm, T, seed=31 + i) for i in range(B * H)]
    pr = torch.cat([x[0] for x in rows]).contiguous()
    pw = torch.cat([x[1] for x in rows]).contiguous()
    dnum = torch.cat([x[3] for x in rows]).contiguous()
    dden = torch.cat([x[4] for x in rows]).contiguous()
    v = torch.stack([torch.stack([rows[b * H + h][2][0, :, 0] for h in range(H)], dim=1)
                     for b in range(B)]).contiguous()
    m = extension()
    (dpr,) = m.chunk_rung0_pass1(pr, pw, v, dnum, dden, *arm)
    dv, dpw = m.chunk_rung0_pass2(pr, pw, v, dnum, dden, *arm)
    for bh in range(B * H):
        _check((dpr[bh], dv[bh], dpw[bh]), _rounded(arm, *rows[bh]), T, Bw ** D,
               f"batch-head {bh} of {B * H} at arm {arm} blocks={Bw ** D // BC}")


def test_the_ladder_census_is_spill_free():
    """The rung declares one register floor for both passes; the census is where the
    claim is falsified, per arm and per pass.

    THE ARM IS `(C, BC, D, B)` AT RUNG b3, because the topology became a compiled axis:
    two rows that agree on the block width and differ on the digit geometry are
    different kernels, and a census keyed on `(C, BC)` alone would collapse them.

    THE RESIDENCY STATEMENT IS WARPS PER SM, not CTAs. It became so at rung b2, where
    the block gained a second slot group: `BC = 128` runs ONE CTA of eight warps where
    `BC = 64` runs two of four, and the quantity the carve holds constant across that
    split -- the one the forward is at, and the one the ideal regime is stated in -- is
    the eight. A gate written on the CTA count alone would have called the split a
    regression and a gate written on nothing would have missed a real one.
    """
    rows = extension().chunk_rung0_census()
    assert rows, "the ladder reports no built arm"
    assert {tuple(r[:4]) for r in rows} == set(ARMS), (
        "the census and this file disagree on which arms are built")
    for row in rows:
        arm, pas = tuple(row[:4]), row[4]
        regs, spill, stages, cta, floor, budget, threads = (row[5], row[6], row[8], row[9],
                                                            row[10], row[11], row[13])
        assert spill == 0, f"arm {arm} pass {pas} spills {spill} bytes"
        assert regs <= floor, (
            f"arm {arm} pass {pas} holds {regs} registers against a DECLARED live set of "
            f"{floor}: the floor is the number `cta_target` is derived from, so a peak "
            f"above it is a residency claim that was never measured")
        assert regs <= budget, (
            f"arm {arm} pass {pas} holds {regs} registers against a budget of {budget}")
        assert stages >= 2, f"arm {arm} pass {pas} staged at depth {stages}"
        assert cta * threads // 32 == 8, (
            f"arm {arm} pass {pas} reaches {cta} CTAs of {threads} threads = "
            f"{cta * threads // 32} warps/SM: the rung's residency is the forward's "
            f"eight and moving it is a carve decision, not a side effect")
