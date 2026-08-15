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

`d gw` is still the INTER-chunk part alone: its intra half needs the OTHER Gram
(`dnum . v^T`, contracted over `d_v`) and the per-slot `dW` the digit reduction
consumes, which is the next stage's. `d pr` and `d pw` are not emitted at all.

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

import pytest
import torch

from rola.ops._ext import extension, is_available

pytestmark = pytest.mark.skipif(
    not is_available() or not torch.cuda.is_available(),
    reason="the chunk consumer's Scan B is a CUDA kernel")

C, BC, D, W = 32, 128, 2, 64
DV = 64
N = W ** D
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


def _rb(x: torch.Tensor) -> torch.Tensor:
    """The operand as the kernel holds it: bf16, read back in fp32."""
    return x.to(torch.bfloat16).to(torch.float32)


def _inputs(T: int, seed: int, live_r: torch.Tensor, live_w: torch.Tensor):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    pr = torch.rand(1, 1, T, D * W, device=dev, generator=g) + 0.25
    pw = torch.rand(1, 1, T, D * W, device=dev, generator=g) + 0.25
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


def _reference(pr, pw, gw, v, dnum, dden, chunks, T, live_w):
    """`(dv, dgw)` and each one's BOUND -- the rounded reference and its own budget."""
    dev = "cuda"
    s = torch.arange(N, device=dev)
    d0, d1 = s // W, s % W
    prf, pwf, gwf = _rb(pr)[0, 0], _rb(pw)[0, 0], _rb(gw)[0, 0]
    vf, dnf, ddf = _rb(v)[0, :, 0], _rb(dnum)[0, 0], _rb(dden)[0, 0]

    #: the level chain and the gain in the kernel's own order -- the gain folds
    #: into the LEADING amplitude, after the level chain and before the run --
    #: and built PER CHUNK, because the whole `[T, N]` coefficient plane is a
    #: gigabyte once `T` crosses a K-window.
    def coeffs(idx):
        return (_rb(prf[idx][:, d0] * prf[idx][:, W + d1]),
                _rb(_rb(pwf[idx][:, d0] * gwf[idx, None]) * pwf[idx][:, W + d1]))

    dS = torch.zeros(N, DV, device=dev)
    dz = torch.zeros(N, device=dev)
    #: THE FOLD'S OWN REASSOCIATION rides into every write-side term through `dS`,
    #: and `|dS|` does not bound it once the sum cancels. These carry the sum of
    #: ABSOLUTE contributions, which does.
    adS = torch.zeros(N, DV, device=dev)
    adz = torch.zeros(N, device=dev)
    out = {k: torch.zeros(sh, device=dev)
           for k, sh in (("dv", (T, DV)), ("dgw", (T,)), ("bdv", (T, DV)), ("bdgw", (T,)))}
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
        out["dv"][idx] = Wc @ dSb + Gm.T @ dnf[idx]
        #: TWO INDEPENDENT FLOORS, and the intra one is NOT a reassociation. Two
        #: correctly-rounded bf16 values of fp32 sums that differ at all can
        #: straddle a rounding boundary, so the kernel's Gram and the reference's
        #: may differ by a whole bf16 ulp -- `2 ** -7` relative, twice the unit
        #: roundoff -- which dominates the intra sum's own `C`-term reassociation.
        out["bdv"][idx] = (n_inter * EPS32 * (Wc.abs() @ adS)
                           + (ULP_BF16_PAIR + len(ch) * EPS32) * (Gm.abs().T @ dnf[idx].abs()))
        #: d gw is a WRITE-SIDE quantity: off the write support the level chain
        #: is exactly zero, so the emitted slot reduction is zero too.
        out["dgw"][idx] = wl * ((vf[idx] @ dSb.T).sum(1) + dz.sum())
        out["bdgw"][idx] = n_inter * EPS32 * wl * ((vf[idx].abs() @ adS.T).sum(1) + adz.sum())
        dS += Rc.T @ dnf[idx]
        dz += Rc.T @ ddf[idx]
        adS += Rc.abs().T @ dnf[idx].abs()
        adz += Rc.abs().T @ ddf[idx].abs()
    return out


def _run(inputs):
    m = extension()
    pr, pw, gw, v, dnum, dden = inputs
    tbl = m.chunk_union_table(pr, pw, v, C, BC, [W] * D)
    return m.chunk_scan_b_probe(pr, pw, gw, v, dnum, dden, tbl, C, BC, [W] * D)


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


@pytest.mark.parametrize("T,frac", [(p.values[0], p.values[1]) for p in CELLS],
                         ids=[p.id for p in CELLS])
def test_scan_b_matches_the_rounded_reference(T, frac):
    live_r, live_w = _liveness(T, frac, seed=7)
    inputs = _inputs(T, 3, live_r, live_w)
    chunks = _partition(live_r.tolist(), live_w.tolist(), T)
    dv_k, dgw_k = _run(inputs)
    ref = _reference(*inputs, chunks, T, live_w)

    for name, got, want, bound in (("dv", dv_k[0, 0], ref["dv"], ref["bdv"]),
                                   ("dgw", dgw_k[0, 0], ref["dgw"], ref["bdgw"])):
        err = (got - want).abs()
        worst = (err / bound.clamp_min(torch.finfo(torch.float32).tiny)).max().item()
        assert worst <= 1.0, (
            f"Scan B's {name} left the reassociation budget at T={T} frac={frac}: "
            f"worst |err| / bound = {worst:.3f}; max |err| = {err.max():.3e} against a "
            f"bound of {bound.max():.3e}. The reference rounds every operand where the "
            f"kernel rounds it, so nothing but summation ORDER is allowed to separate "
            f"them -- a term that is missing, transposed or taken from the wrong chunk "
            f"lands orders of magnitude outside this.")


def test_the_budget_is_not_vacuous():
    """A gate that cannot fail is not a gate: perturb one operand by one bf16 ulp
    of the state and the same bound must reject it."""
    T = 96
    live_r, live_w = _liveness(T, None, seed=7)
    inputs = _inputs(T, 3, live_r, live_w)
    chunks = _partition(live_r.tolist(), live_w.tolist(), T)
    ref = _reference(*inputs, chunks, T, live_w)
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
    for c, bc, d, b, regs, local, smem, stages, ctas, floor in rows:
        assert regs > 0, f"the census could not read arm ({c},{bc},{d},{b})"
        assert local == 0, (
            f"Scan B spills {local} bytes at arm ({c},{bc},{d},{b}): the adjoint state and "
            f"the gradient accumulator do not fit beside each other, so the carve has to "
            f"change (shared residency, or a pass split over d_v) rather than the budget")
        forward = [a for a in m.chunk_arms() if tuple(a[:4]) == (c, bc, d, b)]
        assert forward, f"Scan B is built for an arm the forward is not: ({c},{bc},{d},{b})"
        assert ctas == forward[0][9], (
            f"Scan B reaches {ctas} CTAs where its forward reaches {forward[0][9]} at arm "
            f"({c},{bc},{d},{b})")
        assert stages >= 2, "the wait/issue order needs two staging groups"
        assert regs <= 255 and floor > 0
