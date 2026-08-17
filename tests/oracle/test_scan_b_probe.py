"""SCAN B, stage one: the register-budget probe and the arithmetic it measures.

Vocabulary, defined here and used unqualified below: `C` = chunk tokens (32);
`BC` = slots per CTA block (128); `D` = level count; `W` = one uniform level
width; `N = W ** D` = leaves/slots; `T` = tokens; `d_v = 64` = value channels;
`dnum`/`dden` = the readout's two cotangent seeds; `dS` = the adjoint state
`[N, d_v]`; `dz` = its mass column `[N]`; `R`/`Wc` = the read/write Kronecker
coefficients; a chunk's ENTRY adjoint `dS_a` is the one arriving from the future,
i.e. from every token ABOVE the chunk.

**WHAT SCAN B CARRIES AT THIS STAGE**, and the file gates exactly that and
nothing it does not carry: the descending walk, both operand panels, the fold
`dS += dnum^T . R` with its mass column, the SLOT GRAM `G = staircase(R . Wc^T)`,
`dv` COMPLETE (`Wc . dS_a` plus the intra term `G^T . dnum` the staircase
selects), and the CHANNEL CONTRACTION `<dS_a[s,:], v[j,:]>` -- the one arithmetic
primitive the backward adds, whose `k` axis is `d_v` rather than slots or tokens.

The second output is `sum_s dW[j,s]`, INTER-chunk part only, and it is **not
`d gw`** -- (A.11) weights that slot sum by the ungained level product `U[j,s]`
and this does not, so the two agree only at `U = 1`. It is named `dwsum` for that
reason. Completing it needs the ungained write panel and the `(d_v+1)`-th column
identity (`docs/internals/chunk/backward.md#dw-slotsum`); `d pr` and `d pw` are
not emitted at all.

The `T = C` cell below is the sharpest gate on the new term and is why it is
first: with one chunk the entry adjoint is exactly zero, so `dv` there is
ENTIRELY the intra staircase and nothing else can carry it.

**WHY THE REFERENCE CAN REPLAY THE PARTITION.** Scan B walks its own descending
partition (the chunk boundary is an association of a sum, not a semantic object),
so a reference generally cannot know where its chunks fall. It can in the cells
here: a token is made either fully live or fully dead on a side, so every block's
union-table row is the same and every block's walk closes at the same tokens. The
walk's closing rule is then a five-line simulation. Gating the walk's partition in
the GENERAL case is the next stage's device-side test, not a tolerance here.

**THE BOUND IS DERIVED, not read off a failure.** The reference rounds every
operand exactly where the kernel rounds it -- the level chain to bf16 at each
multiply, the write gain folded after the chain, and `dS` to bf16 at the moment it
becomes an mma operand -- so nothing separates the two runs except the ORDER of
two fp32 sums: the contraction over `N` slots inside a block, and the `N / BC`
block partials the fan-in adds with `atomicAdd`. The bound is that reassociation,
`n_terms * eps_fp32 * sum |terms|` over the three sums that reassociate -- the
fold's `T` terms, the write-side contraction's `N`, and the `N / BC` block
partials the fan-in adds -- with the term sums accumulated in ABSOLUTE value
beside the signed ones, because `|dS|` does not bound the error in a `dS` whose
own sum cancelled.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from rola.facts.manifest import scan_arms
from rola.ops._ext import extension, is_available

pytestmark = pytest.mark.skipif(
    not is_available() or not torch.cuda.is_available(),
    reason="the chunk consumer's Scan B is a CUDA kernel")

C = 32
DV = 64
EPS32 = 2.0 ** -24
#: ONE bf16 ULP, RELATIVE. bf16 carries 8 significand bits, so the unit roundoff is
#: `2 ** -8` and two correctly-rounded values either side of a boundary differ by
#: `2 ** -7`. This is the Gram's floor, and it is a ROUNDING floor, not a
#: reassociation one -- see `_reference`.
ULP_BF16_PAIR = 2.0 ** -7
#: An index-list entry is 16 bits, so a chunk must lie inside one K-window and the
#: walk is forced to cut at every window line (`chunk_kernel.cuh`). Descending, the
#: line is a floor rather than a ceiling, and that is NEW code.
K_WINDOW = 65536


#: THE ARM AXIS. Read off the census, so a widened scan matrix widens this
#: battery by collection rather than by anybody remembering to add a row.
@dataclass(frozen=True)
class Arm:
    C: int
    BC: int
    D: int
    W: int

    @property
    def N(self) -> int:
        return self.W ** self.D

    def digits(self, dev):
        """slot -> its digit at each level, most-significant level first."""
        s = torch.arange(self.N, device=dev)
        return [(s // (self.W ** (self.D - 1 - l))) % self.W for l in range(self.D)]

    def __str__(self) -> str:
        return f"C{self.C}BC{self.BC}D{self.D}W{self.W}"


ARMS = [Arm(*a) for a in sorted(scan_arms())]
PINNED = Arm(32, 128, 2, 64)


def _table_entries(a: Arm) -> int:
    """The level plan's table rows per side at this arm, READ OFF THE CENSUS."""
    for row in extension().chunk_scan_b_census():
        if tuple(row[:4]) == (a.C, a.BC, a.D, a.W):
            return row[11]
    raise AssertionError(f"no scan census row for arm {a}")


def _walk_liveness(a: Arm, live_r, live_w):
    """The liveness THE WALK SEES, which is the union table's and not the test's.

    A block whose plan has ZERO table entries spans the whole slot space, so its
    clause is empty and every token is a member of it: the walk gathers all of
    them and the partition is the dense one however sparse the amplitudes are.
    Where there are entries, this bundle's rows are dense across every digit, so
    a token is a member exactly when it is live.
    """
    if _table_entries(a):
        return live_r, live_w
    ones = torch.ones_like(live_r)
    return ones, ones.clone()


def _rb(x: torch.Tensor) -> torch.Tensor:
    """The operand as the kernel holds it: bf16, read back in fp32."""
    return x.to(torch.bfloat16).to(torch.float32)


def _chain(p, idx, digs, W):
    """The level chain in the kernel's own order: bf16 after every multiply."""
    out = _rb(p[idx][:, digs[0]])
    for l in range(1, len(digs)):
        out = _rb(out * p[idx][:, l * W + digs[l]])
    return out


def _leave_one_out(p, idx, digs, W, swap: bool):
    """(A.10)'s per-level factor: the product of the OTHER levels' amplitudes --
    exact at the zeros entmax puts there, which `W / p_l` would not be. `swap`
    returns the level's OWN amplitude instead, the alternative implementation the
    budget has to be able to reject."""
    amps = [p[idx][:, l * W + digs[l]] for l in range(len(digs))]
    out = []
    for l in range(len(digs)):
        if swap:
            out.append(amps[l])
            continue
        f = torch.ones_like(amps[l])
        for k in range(len(digs)):
            if k != l:
                f = f * amps[k]
        out.append(f)
    return out


def _inputs(a: Arm, T: int, seed: int, live_r: torch.Tensor, live_w: torch.Tensor):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    pr = torch.rand(1, 1, T, a.D * a.W, device=dev, generator=g) + 0.25
    pw = torch.rand(1, 1, T, a.D * a.W, device=dev, generator=g) + 0.25
    pr[0, 0][~live_r] = 0.0
    pw[0, 0][~live_w] = 0.0
    gw = torch.rand(1, 1, T, device=dev, generator=g) + 0.5
    v = torch.randn(1, T, 1, DV, device=dev, generator=g)
    dnum = torch.randn(1, 1, T, DV, device=dev, generator=g)
    dden = torch.randn(1, 1, T, device=dev, generator=g)
    return tuple(t.to(torch.bfloat16).contiguous() for t in (pr, pw, gw, v, dnum, dden))


def _partition(live_r, live_w, T) -> list[list[int]]:
    """The descending walk's chunks, each ASCENDING in token order.

    The kernel's closing rule, verbatim: walking down, a token is excluded from
    the open chunk when it is live on a side that has already gathered `C`
    entries above it.
    """
    chunks, cur, nr, nw = [], [], 0, 0
    for t in range(T - 1, -1, -1):
        #: a chunk holding entries closes AT the line; an empty one slides its
        #: window down, exactly as it crosses words
        crossed = cur and (t // K_WINDOW) != (cur[-1] // K_WINDOW)
        if (live_r[t] and nr == C) or (live_w[t] and nw == C) or (crossed and (nr or nw)):
            chunks.append(sorted(cur))
            cur, nr, nw = [], 0, 0
        elif crossed:
            cur = []
        cur.append(t)
        nr += int(live_r[t])
        nw += int(live_w[t])
    if cur:
        chunks.append(sorted(cur))
    return chunks


def _reference(a, pr, pw, gw, v, dnum, dden, chunks, T, live_w, swap_loo=False):
    """`(dv, dwsum)` and each one's BOUND -- the rounded reference and its budget."""
    dev = "cuda"
    N, W, D, BC = a.N, a.W, a.D, a.BC
    digs = a.digits(dev)
    prf, pwf, gwf = _rb(pr)[0, 0], _rb(pw)[0, 0], _rb(gw)[0, 0]
    vf, dnf, ddf = _rb(v)[0, :, 0], _rb(dnum)[0, 0], _rb(dden)[0, 0]

    #: the level chain and the gain in the kernel's own order -- the gain folds
    #: into the LEADING amplitude, after the level chain and before the run --
    #: and built PER CHUNK, because the whole `[T, N]` coefficient plane is a
    #: gigabyte once `T` crosses a K-window.
    def coeffs(idx):
        #: THE WRITE PANEL IS UNGAINED, as the kernel produces it; the gain is a
        #: per-token scale applied once at the readout's store, so it multiplies
        #: `dv` and never the level product `d gw` contracts against.
        return _chain(prf, idx, digs, W), _chain(pwf, idx, digs, W)

    dS = torch.zeros(N, DV, device=dev)
    dz = torch.zeros(N, device=dev)
    #: THE FOLD'S OWN REASSOCIATION rides into every write-side term through `dS`,
    #: and `|dS|` does not bound it once the sum cancels. These carry the sum of
    #: ABSOLUTE contributions, which does.
    adS = torch.zeros(N, DV, device=dev)
    adz = torch.zeros(N, device=dev)
    out = {k: torch.zeros(sh, device=dev)
           for k, sh in (("dv", (T, DV)), ("dwsum", (T,)), ("dgw", (T,)), ("dpw", (T, D * W)),
                         ("bdv", (T, DV)), ("bdwsum", (T,)), ("bdgw", (T,)),
                         ("bdpw", (T, D * W)))}
    #: THE INTER PATH'S REASSOCIATION: the `T` terms the fold accumulates into
    #: `dS`, the `N` the write-side contraction sums inside a block, and the
    #: `N / BC` block partials the fan-in adds in an order the scheduler picks.
    n_inter = T + N + N // BC
    for ch in chunks:
        idx = torch.tensor(ch, device=dev)
        Rc, Wc = coeffs(idx)
        dSb = _rb(dS)
        wl = live_w[idx].float()
        #: THE SLOT GRAM, and it is a NEW ROUNDING SITE: the kernel contracts it
        #: over slots in fp32 and stores it as bf16, so the reference stores it as
        #: bf16 too. Masking commutes with rounding (the mask is 0/1), so the
        #: order the two do it in does not matter.
        tok = idx.unsqueeze(0)
        #: the staircase in TOKEN terms: a write entry is kept when its token is
        #: at or before the read entry's, which is what `J` counts. Dead tokens
        #: carry exactly-zero coefficients, so the chunk's whole token span is the
        #: same statement as its compacted lists.
        stair = (tok <= tok.T).to(Rc.dtype)
        Gm = _rb(Rc @ Wc.T) * stair
        #: `Wc` is UNGAINED, so this is the ungained readout and the gain lands on
        #: the store -- exactly the kernel's own sequence.
        ungained = Wc @ dSb + Gm.T @ dnf[idx]
        out["dv"][idx] = ungained * gwf[idx, None]
        #: (A.11) d gw[j] = sum_s dW[j,s] * U[j,s], and the three terms it splits
        #: into: the value channels off the ungained readout, the entry mass
        #: column, and the in-chunk mass the staircase selects.
        out["dgw"][idx] = ((ungained * vf[idx]).sum(1)
                           + Wc @ dz
                           + Gm.T @ ddf[idx])
        #: TWO INDEPENDENT FLOORS, and the intra one is NOT a reassociation. Two
        #: correctly-rounded bf16 values of fp32 sums that differ at all can
        #: straddle a rounding boundary, so the kernel's Gram and the reference's
        #: may differ by a whole bf16 ulp -- `2 ** -7` relative, twice the unit
        #: roundoff -- which dominates the intra sum's own `C`-term reassociation.
        out["bdv"][idx] = gwf[idx, None].abs() * (
            n_inter * EPS32 * (Wc.abs() @ adS)
            + (ULP_BF16_PAIR + len(ch) * EPS32) * (Gm.abs().T @ dnf[idx].abs()))
        #: d gw inherits `dv`'s floor through the value contraction (a further `DV`
        #: terms), plus its own two slot/read reductions and the Gram's rounding.
        out["bdgw"][idx] = (
            (n_inter + DV) * EPS32 * ((Wc.abs() @ adS) * vf[idx].abs()).sum(1)
            + (ULP_BF16_PAIR + (len(ch) + DV) * EPS32)
            * ((Gm.abs().T @ dnf[idx].abs()) * vf[idx].abs()).sum(1)
            + (n_inter + N) * EPS32 * (Wc.abs() @ adz)
            + (ULP_BF16_PAIR + len(ch) * EPS32) * (Gm.abs().T @ ddf[idx].abs()))
        #: THE SLOT SUM OF dW, unweighted -- NOT `d gw`, which weights it by the
        #: ungained level product (A.11). Off the write support the level chain is
        #: exactly zero, so the emitted reduction is zero too.
        out["dwsum"][idx] = wl * ((vf[idx] @ dSb.T).sum(1) + dz.sum())
        out["bdwsum"][idx] = n_inter * EPS32 * wl * ((vf[idx].abs() @ adS.T).sum(1) + adz.sum())
        #: THE PER-SLOT dW, which `d pw` reduces over digit classes. Its intra half
        #: carries the R-b ONES COLUMN: `dW` differentiates both uses of `W`, so the
        #: adjoint entering write entry `j` is the state AND the mass, and one
        #: staircase covers them together.
        Gdp = _rb(dnf[idx] @ vf[idx].T + ddf[idx][:, None]) * stair
        dWc = (vf[idx] @ dSb.T) + dz[None, :] + Gdp.T @ Rc
        aWc = (vf[idx].abs() @ adS.T) + adz[None, :] + Gdp.abs().T @ Rc.abs()
        #: (A.10): the leave-one-out product is the OTHER level's amplitude, never
        #: `W / pw_l` -- exact at the zeros entmax puts there.
        gram_abs = Gdp.abs().T @ Rc.abs()
        loos = _leave_one_out(pwf, idx, digs, W, swap_loo)
        for lv, (loo, dig) in enumerate(zip(loos, digs)):
            gain = gwf[idx, None]
            #: OFF THE WRITE SUPPORT THE KERNEL EMITS NOTHING, because a token with no
            #: support is not a write entry and is never gathered. At two levels or
            #: more the leave-one-out factor is already exactly zero there and this
            #: changes nothing; at ONE level it is the empty product, and the mask is
            #: what states the contract instead of asserting a term the operator's
            #: producer discards anyway (an entmax Jacobian is supported on its own
            #: support) -- docs/internals/chunk/backward.md#pass-law
            wgt = dWc * loo * gain * wl[:, None]
            #: A DIGIT CLASS IS A SUM, so the scatter must ACCUMULATE: `N / W` slots
            #: land on each column and an advanced-index assignment would keep the
            #: last one.
            col = (lv * W + dig).unsqueeze(0).expand(len(ch), -1)
            acc_w = torch.zeros(len(ch), D * W, device=dev, dtype=wgt.dtype)
            acc_w.scatter_add_(1, col, wgt)
            out["dpw"].index_add_(0, idx, acc_w)
            #: the class's `N / W` slots reassociate, the `N / BC` block partials the
            #: fan-in adds reassociate, and the Gram's bf16 store is the same
            #: rounding floor `dv` carries.
            bnd = (((n_inter + N // W + N // BC) * EPS32 * (aWc * loo.abs() * gain.abs())
                    + ULP_BF16_PAIR * gram_abs * loo.abs() * gain.abs()) * wl[:, None])
            acc_b = torch.zeros(len(ch), D * W, device=dev, dtype=bnd.dtype)
            acc_b.scatter_add_(1, col, bnd)
            out["bdpw"].index_add_(0, idx, acc_b)
        dS += Rc.T @ dnf[idx]
        dz += Rc.T @ ddf[idx]
        adS += Rc.abs().T @ dnf[idx].abs()
        adz += Rc.abs().T @ ddf[idx].abs()
    return out


def _run(a: Arm, inputs):
    m = extension()
    pr, pw, gw, v, dnum, dden = inputs
    widths = [a.W] * a.D
    tbl = m.chunk_union_table(pr, pw, v, a.C, a.BC, widths)
    return m.chunk_scan_b_probe(pr, pw, gw, v, dnum, dden, tbl, a.C, a.BC, widths)


def _liveness(T: int, frac: float | None, seed: int):
    if frac is None:
        ones = torch.ones(T, dtype=torch.bool, device="cuda")
        return ones, ones.clone()
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.rand(T, device="cuda", generator=g) < frac,
            torch.rand(T, device="cuda", generator=g) < frac)


CELLS = [
    #: one chunk exactly, so the panel's padding rows must be exactly zero
    pytest.param(C, None, id="T=C dense"),
    pytest.param(96, None, id="T=3C dense"),
    pytest.param(128, None, id="T=4C dense"),
    #: compacted: chunks span far more tokens than they gather, and the
    #: descending walk's cut, its buffer tail and its per-side counts all move
    pytest.param(192, 0.5, id="token-sparse 1/2"),
    pytest.param(256, 0.25, id="token-sparse 1/4"),
    pytest.param(320, 0.1, id="token-sparse 1/10"),
    #: ACROSS A K-WINDOW LINE. An index-list entry is 16 bits, so a chunk may not
    #: span two windows and the walk is forced to cut at every line. Descending,
    #: that cut is a FLOOR the walk falls through rather than a ceiling it reaches,
    #: and it is the one piece of the reverse walk with no forward counterpart to
    #: inherit. Without this row it would ship unexecuted.
    pytest.param(K_WINDOW + 2 * C, None, id="across a K-window line"),
]

#: THE WINDOW-LINE CELL IS THE WALK'S, and the walk is an arm-blind component
#: (`Descent`, shared with the partition probe), so it runs at one arm rather
#: than paying its 65,536 tokens once per arm.
CELLS_PER_ARM = CELLS[:-1]


@pytest.mark.parametrize("arm", ARMS, ids=str)
@pytest.mark.parametrize("T,frac", [(p.values[0], p.values[1]) for p in CELLS_PER_ARM],
                         ids=[p.id for p in CELLS_PER_ARM])
def test_scan_b_matches_the_rounded_reference(arm, T, frac):
    _scan_b_cell(arm, T, frac)


@pytest.mark.parametrize("T,frac", [(p.values[0], p.values[1]) for p in CELLS[-1:]],
                         ids=[p.id for p in CELLS[-1:]])
def test_the_descending_walk_crosses_a_window_line(T, frac):
    _scan_b_cell(PINNED, T, frac)


def _scan_b_cell(a: Arm, T, frac):
    live_r, live_w = _liveness(T, frac, seed=7)
    inputs = _inputs(a, T, 3, live_r, live_w)
    wr, ww = _walk_liveness(a, live_r, live_w)
    chunks = _partition(wr.tolist(), ww.tolist(), T)
    dv_k, dws_k, dgw_k, dpw_k = _run(a, inputs)
    #: the WALK's write liveness, not the bundle's: the two unweighted terms
    #: (`sum_s dW` and, at one level, the digit gradient) are emitted for every
    #: GATHERED entry, and which tokens those are is the union table's answer.
    ref = _reference(a, *inputs, chunks, T, ww)

    for name, got, want, bound in (("dv", dv_k[0, 0], ref["dv"], ref["bdv"]),
                                   ("sum_s dW", dws_k[0, 0], ref["dwsum"], ref["bdwsum"]),
                                   ("d gw", dgw_k[0, 0], ref["dgw"], ref["bdgw"]),
                                   ("d pw", dpw_k[0, 0], ref["dpw"], ref["bdpw"])):
        err = (got - want).abs()
        worst = (err / bound.clamp_min(torch.finfo(torch.float32).tiny)).max().item()
        assert worst <= 1.0, (
            f"Scan B's {name} left the reassociation budget at arm {a} T={T} frac={frac}: "
            f"worst |err| / bound = {worst:.3f}; max |err| = {err.max():.3e} against a "
            f"bound of {bound.max():.3e}. The reference rounds every operand where the "
            f"kernel rounds it, so nothing but summation ORDER is allowed to separate "
            f"them -- a term that is missing, transposed or taken from the wrong chunk "
            f"lands orders of magnitude outside this.")


def test_d_pw_rejects_the_swapped_leave_one_out():
    """(A.10)'s leave-one-out factor is the OTHER level's amplitude, and which one
    is not a detail: at depth two the two digit gradients differ only in that
    choice. A budget that could not tell them apart would pass a kernel that
    weighted each level by its own amplitude.

    Recomputed here rather than perturbed, so the rejected answer is a REAL
    alternative implementation and not noise.
    """
    T = 96
    live_r, live_w = _liveness(T, None, seed=7)
    inputs = _inputs(PINNED, T, 3, live_r, live_w)
    chunks = _partition(live_r.tolist(), live_w.tolist(), T)
    ref = _reference(PINNED, *inputs, chunks, T, live_w)
    swapped = _reference(PINNED, *inputs, chunks, T, live_w, swap_loo=True)["dpw"]
    assert ((swapped - ref["dpw"]).abs() > ref["bdpw"]).any(), (
        "the d pw budget accepts the two levels' leave-one-out factors exchanged, so "
        "it is not gating (A.10)'s product at all")


def test_d_gw_rejects_the_unweighted_slot_sum():
    """THE NON-VACUITY ROW THAT MATTERS, because it is the defect that was there.

    `d gw` weights the slot sum by the UNGAINED level product (A.11); the plane
    beside it does not. The two coincide at `U = 1`, which is every ladder gate ever
    run, so the budget has to be able to tell them apart on a cell where `U` is not
    one. If this ever passes, `d gw` is being gated against something that is not
    `d gw`.
    """
    T = 96
    live_r, live_w = _liveness(T, None, seed=7)
    inputs = _inputs(PINNED, T, 3, live_r, live_w)
    chunks = _partition(live_r.tolist(), live_w.tolist(), T)
    ref = _reference(PINNED, *inputs, chunks, T, live_w)
    assert ((ref["dwsum"] - ref["dgw"]).abs() > ref["bdgw"]).any(), (
        "the d gw budget accepts the UNWEIGHTED slot sum of dW, so it cannot see the "
        "one term (A.11) adds and is not gating a gain gradient at all")


def test_the_budget_is_not_vacuous():
    """A gate that cannot fail is not a gate: perturb one operand by one bf16 ulp
    of the state and the same bound must reject it."""
    T = 96
    live_r, live_w = _liveness(T, None, seed=7)
    inputs = _inputs(PINNED, T, 3, live_r, live_w)
    chunks = _partition(live_r.tolist(), live_w.tolist(), T)
    ref = _reference(PINNED, *inputs, chunks, T, live_w)
    bound = ref["bdv"]
    #: the smallest STRUCTURAL wrong answer available: one chunk's entry adjoint
    #: taken from the wrong side of its own fold.
    wrong = ref["dv"].roll(C, dims=0)
    err = (wrong - ref["dv"]).abs()
    assert (err > bound).any(), (
        "the reassociation budget accepts a chunk-shifted answer, so it is not "
        "measuring the partition at all")


def test_the_probe_census_is_spill_free_and_leaves_the_arm_where_it_was():
    """R-A's exit criterion, asserted rather than reported.

    The adjoint state and the gradient accumulator have to fit BESIDE each other
    with no spill, and Scan B's own shared-memory carve has to reach the same CTA
    target the forward's does -- a backward resident at half its forward's
    occupancy would move the arm without anybody deciding to.
    """
    m = extension()
    rows = m.chunk_scan_b_census()
    assert rows, "Scan B reports no built arm"
    for c, bc, d, b, regs, local, smem, stages, ctas, floor, budget, entries in rows:
        assert regs > 0, f"the census could not read arm ({c},{bc},{d},{b})"
        assert local == 0, (
            f"Scan B spills {local} bytes at arm ({c},{bc},{d},{b}): the adjoint state and "
            f"the gradient accumulator do not fit beside each other, so the carve has to "
            f"change (shared residency, or a pass split over d_v) rather than the budget")
        #: THE FLOOR MODEL, MADE FALSIFIABLE PER ARM: the declared floor picks the CTA
        #: target, the target fixes the budget, and the measured live set has to fit
        #: inside it. A row that spills is caught above; a row that fits only because
        #: this part is generous is caught here.
        assert regs <= budget, (
            f"arm ({c},{bc},{d},{b}) holds {regs} registers against a budget of {budget} "
            f"at {ctas} CTAs -- the declared floor {floor} bought a residency the live "
            f"set cannot pay for")
        forward = [a for a in m.chunk_arms() if tuple(a[:4]) == (c, bc, d, b)]
        assert forward, f"Scan B is built for an arm the forward is not: ({c},{bc},{d},{b})"
        #: THE SCANS MAY RUN BELOW THE FORWARD'S OCCUPANCY, NEVER ABOVE IT: the reverse
        #: pass' live set does not fall with `BC` the way the forward's does, so the
        #: two floors are separate declarations and the PAIR is what the review reads.
        assert ctas <= forward[0][9], (
            f"Scan B reaches {ctas} CTAs where its forward reaches {forward[0][9]} at arm "
            f"({c},{bc},{d},{b})")
        assert stages >= 2, "the wait/issue order needs two staging groups"
        #: `entries == 0` is the legal answer for a block that spans the whole slot
        #: space -- there is no group to test membership in, so the plan builds no
        #: table row for it.
        assert regs <= 255 and floor > 0 and entries >= 0
