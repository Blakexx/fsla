"""SCAN A — the forward rebuild, and the read side's digit gradient `d pr`.

Vocabulary, defined here and used unqualified below: `C` = chunk tokens (32);
`BC` = slots per CTA block (128); `D` = level count; `W` = one uniform level
width; `N = W ** D` = leaves/slots; `T` = tokens; `d_v = 64`; `R`/`Wc` = the
read/write coefficients (`Wc` GAINED, as this scan's fold deposits it); `S`/`z` =
the forward state and its mass column; `dnum`/`dden` = the readout's two cotangent
seeds; a chunk's ENTRY state `S_a`/`z_a` is the one arriving from every token
BELOW the chunk.

**WHAT SCAN A IS.** Scan B walks down carrying the adjoint `dS`; Scan A walks UP
carrying the forward state `S`, rebuilt from zero over the SAVED union table, and
emits the one gradient that needs `S` rather than `dS`:

    dR[i,s] = <dnum[i,:], S[i,s,:]> + dden[i] * z[i,s]                          (A.8)
    d pr_l[t,d] = sum_{s: d_l(s)=d} dR[t,s] * prod_{l' != l} pr_{l'}[t, d_{l'}(s)]  (A.9)

with NO gain anywhere: `g_read` does not exist, so (A.9) has no gain factor and
there is no `d g_read` output.

**THE GRAM IS THE SAME ONE SCAN B FORMS.** Expanding the INCLUSIVE `S`/`z` over the
chunk gives

    dR[i,s] = <dnum[i,:], S_a[s,:]> + dden[i]*z_a[s]
            + sum_j [j <= i] * Wc[j,s] * ( <dnum[i,:], v[j,:]> + dden[i] )

and the bracket is `Gd'`, ones column and all — the design's (A.14) and (A.15)
collapsed into ONE masked contraction. The reference below therefore builds `Gd'`
exactly as `test_scan_b_probe.py` does, because the kernel does.

**THE BOUND IS DERIVED**, and the reference rounds every operand where the kernel
rounds it: the level chains to bf16 at each multiply, the gain folded into the
write panel, `S` to bf16 where it is transposed into shared to become an mma
operand, and the Gram to bf16 where it is stored. What is left to separate them is
summation ORDER, plus the Gram's one worst-case rounding flip.
"""
from __future__ import annotations

import pytest
import torch

from rola.ops._ext import extension, is_available
from tests.oracle.test_scan_b_probe import ARMS, PINNED, Arm, _chain, _leave_one_out, _walk_liveness

pytestmark = pytest.mark.skipif(
    not is_available() or not torch.cuda.is_available(),
    reason="Scan A is a CUDA kernel")

C = 32
DV = 64
EPS32 = 2.0 ** -24
ULP_BF16_PAIR = 2.0 ** -7
K_WINDOW = 65536


def _rb(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float32)


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
    """The ASCENDING walk's chunks — the forward's own closing rule, which Scan A
    drives through the same `Ascent` component the forward does."""
    chunks, cur, nr, nw = [], [], 0, 0
    for t in range(T):
        crossed = bool(cur) and (t // K_WINDOW) != (cur[-1] // K_WINDOW)
        if (live_r[t] and nr == C) or (live_w[t] and nw == C) or (crossed and (nr or nw)):
            chunks.append(cur)
            cur, nr, nw = [], 0, 0
        elif crossed:
            cur = []
        cur.append(t)
        nr += int(live_r[t])
        nw += int(live_w[t])
    if cur:
        chunks.append(cur)
    return chunks


def _reference(a, pr, pw, gw, v, dnum, dden, chunks, T, live_r):
    """`d pr` and its bound, from the rounded reference."""
    dev = "cuda"
    N, W, D, BC = a.N, a.W, a.D, a.BC
    digs = a.digits(dev)
    prf, pwf, gwf = _rb(pr)[0, 0], _rb(pw)[0, 0], _rb(gw)[0, 0]
    vf, dnf, ddf = _rb(v)[0, :, 0], _rb(dnum)[0, 0], _rb(dden)[0, 0]

    def coeffs(idx):
        #: the write panel is GAINED in this scan: its fold deposits `Wc` itself
        #: the WRITE panel is GAINED in this scan, and the gain folds into the
        #: LEADING amplitude -- after the first level, before the rest of the chain.
        w = gwf[idx, None]
        for l in range(D):
            w = _rb(w * pwf[idx][:, l * W + digs[l]])
        return _chain(prf, idx, digs, W), w

    S = torch.zeros(N, DV, device=dev)
    z = torch.zeros(N, device=dev)
    aS = torch.zeros(N, DV, device=dev)
    az = torch.zeros(N, device=dev)
    out = {k: torch.zeros(T, D * W, device=dev) for k in ("dpr", "bdpr")}
    n_inter = T + N + N // BC
    for ch in chunks:
        idx = torch.tensor(ch, device=dev)
        Rc, Wc = coeffs(idx)
        #: `S` is rounded where the kernel rounds it: at the transpose into shared,
        #: which is what makes it an mma operand.
        Sb = _rb(S)
        tok = idx.unsqueeze(0)
        #: the staircase: a write entry is kept when its token is at or before the
        #: read entry's, which is the inclusive read-after-write the forward does.
        stair = (tok <= tok.T).to(Rc.dtype)
        Gdp = _rb(dnf[idx] @ vf[idx].T + ddf[idx][:, None]) * stair
        dRc = (dnf[idx] @ Sb.T) + ddf[idx][:, None] * z[None, :] + Gdp @ Wc
        aRc = (dnf[idx].abs() @ aS.T) + ddf[idx].abs()[:, None] * az[None, :] + Gdp.abs() @ Wc.abs()
        gram_abs = Gdp.abs() @ Wc.abs()
        #: (A.9)'s leave-one-out product is the product of the OTHER levels'
        #: amplitudes, and there is NO gain on this side.  The liveness factor is
        #: the mirror of Scan B's: off the READ support nothing is gathered, so
        #: nothing is emitted -- vacuous above one level, the statement itself at one.
        rl = live_r[idx].float()[:, None]
        for lv, (loo, dig) in enumerate(zip(_leave_one_out(prf, idx, digs, W, False), digs)):
            col = (lv * W + dig).unsqueeze(0).expand(len(ch), -1)
            acc = torch.zeros(len(ch), D * W, device=dev)
            acc.scatter_add_(1, col, dRc * loo * rl)
            out["dpr"].index_add_(0, idx, acc)
            bnd = (((n_inter + N // W + N // BC) * EPS32 * (aRc * loo.abs())
                    + ULP_BF16_PAIR * gram_abs * loo.abs()) * rl)
            accb = torch.zeros(len(ch), D * W, device=dev)
            accb.scatter_add_(1, col, bnd)
            out["bdpr"].index_add_(0, idx, accb)
        S += Wc.T @ vf[idx]
        z += Wc.sum(0)
        aS += Wc.abs().T @ vf[idx].abs()
        az += Wc.abs().sum(0)
    return out


def _run(a: Arm, inputs):
    m = extension()
    pr, pw, gw, v, dnum, dden = inputs
    widths = [a.W] * a.D
    tbl = m.chunk_union_table(pr, pw, v, a.C, a.BC, widths)
    return m.chunk_scan_a_probe(pr, pw, gw, v, dnum, dden, tbl, a.C, a.BC, widths)


def _liveness(T: int, frac: float | None, seed: int):
    if frac is None:
        ones = torch.ones(T, dtype=torch.bool, device="cuda")
        return ones, ones.clone()
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.rand(T, device="cuda", generator=g) < frac,
            torch.rand(T, device="cuda", generator=g) < frac)


CELLS = [
    #: ONE CHUNK: the entry state and entry mass are exactly zero, so `d pr` here is
    #: ENTIRELY the intra staircase including its ones column. Nothing else can
    #: carry it, which makes this the sharpest cell for the Gd' collapse.
    pytest.param(C, None, id="T=C (pure intra)"),
    pytest.param(96, None, id="T=3C dense"),
    pytest.param(128, None, id="T=4C dense"),
    pytest.param(192, 0.5, id="token-sparse 1/2"),
    pytest.param(256, 0.25, id="token-sparse 1/4"),
    pytest.param(320, 0.1, id="token-sparse 1/10"),
    #: the ascending walk's window-line cut, which is the forward's own and is
    #: exercised here through the shared component
    pytest.param(K_WINDOW + 2 * C, None, id="across a K-window line"),
]


#: the window-line cell is the walk's, and `Ascent` is arm-blind: it runs at one arm.
CELLS_PER_ARM = CELLS[:-1]


@pytest.mark.parametrize("arm", ARMS, ids=str)
@pytest.mark.parametrize("T,frac", [p.values for p in CELLS_PER_ARM],
                         ids=[p.id for p in CELLS_PER_ARM])
def test_scan_a_matches_the_rounded_reference(arm, T, frac):
    _scan_a_cell(arm, T, frac)


@pytest.mark.parametrize("T,frac", [p.values for p in CELLS[-1:]], ids=[p.id for p in CELLS[-1:]])
def test_the_ascending_walk_crosses_a_window_line(T, frac):
    _scan_a_cell(PINNED, T, frac)


def _scan_a_cell(a: Arm, T, frac):
    live_r, live_w = _liveness(T, frac, seed=7)
    inputs = _inputs(a, T, 3, live_r, live_w)
    wr, ww = _walk_liveness(a, live_r, live_w)
    chunks = _partition(wr.tolist(), ww.tolist(), T)
    (dpr_k,) = _run(a, inputs)
    ref = _reference(a, *inputs, chunks, T, wr)
    err = (dpr_k[0, 0] - ref["dpr"]).abs()
    worst = (err / ref["bdpr"].clamp_min(torch.finfo(torch.float32).tiny)).max().item()
    assert worst <= 1.0, (
        f"Scan A's d pr left its derived budget at arm {a} T={T} frac={frac}: worst "
        f"|err| / bound = {worst:.3f}; max |err| = {err.max():.3e} against a bound of "
        f"{ref['bdpr'].max():.3e}. The reference rounds every operand where the kernel "
        f"rounds it, so only summation ORDER and the Gram's one rounding flip may "
        f"separate them -- a missing term lands orders of magnitude outside this.")


def test_d_pr_rejects_the_swapped_leave_one_out():
    """(A.9)'s leave-one-out factor is the OTHER level's amplitude. At depth two the
    two digit gradients differ ONLY in that choice, so a budget that cannot separate
    them is not gating the product at all."""
    T = 96
    live_r, live_w = _liveness(T, None, seed=7)
    inputs = _inputs(PINNED, T, 3, live_r, live_w)
    chunks = _partition(live_r.tolist(), live_w.tolist(), T)
    ref = _reference(PINNED, *inputs, chunks, T, live_r)
    swapped = _swapped_reference(*inputs, chunks, T)
    assert ((swapped - ref["dpr"]).abs() > ref["bdpr"]).any(), (
        "the d pr budget accepts the two levels' leave-one-out factors exchanged")


def _swapped_reference(pr, pw, gw, v, dnum, dden, chunks, T):
    """The PINNED arm's alternative implementation: the two levels' leave-one-out
    factors exchanged. Depth two is where the two differ only in that choice, which
    is what makes it the rejection the budget has to be able to make."""
    dev = "cuda"
    N, W, D = PINNED.N, PINNED.W, PINNED.D
    s = torch.arange(N, device=dev)
    d0, d1 = s // W, s % W
    prf, pwf, gwf = _rb(pr)[0, 0], _rb(pw)[0, 0], _rb(gw)[0, 0]
    vf, dnf, ddf = _rb(v)[0, :, 0], _rb(dnum)[0, 0], _rb(dden)[0, 0]
    S = torch.zeros(N, DV, device=dev)
    z = torch.zeros(N, device=dev)
    out = torch.zeros(T, D * W, device=dev)
    for ch in chunks:
        idx = torch.tensor(ch, device=dev)
        Rc = _rb(prf[idx][:, d0] * prf[idx][:, W + d1])
        Wc = _rb(_rb(pwf[idx][:, d0] * gwf[idx, None]) * pwf[idx][:, W + d1])
        tok = idx.unsqueeze(0)
        stair = (tok <= tok.T).to(Rc.dtype)
        Gdp = _rb(dnf[idx] @ vf[idx].T + ddf[idx][:, None]) * stair
        dRc = (dnf[idx] @ _rb(S).T) + ddf[idx][:, None] * z[None, :] + Gdp @ Wc
        pr0, pr1 = prf[idx][:, d0], prf[idx][:, W + d1]
        for lv, loo, dig in ((0, pr0, d0), (1, pr1, d1)):   # EXCHANGED
            col = (lv * W + dig).unsqueeze(0).expand(len(ch), -1)
            acc = torch.zeros(len(ch), D * W, device=dev)
            acc.scatter_add_(1, col, dRc * loo)
            out.index_add_(0, idx, acc)
        S += Wc.T @ vf[idx]
        z += Wc.sum(0)
    return out


def test_scan_a_census_is_spill_free():
    """Scan A shares Scan B's geometry, so it must share its residency: a scan that
    ran at a different occupancy would make the two halves of one training step
    disagree about the arm."""
    m = extension()
    rows = m.chunk_scan_b_census()
    assert rows, "the scans report no built arm"
    for row in rows:
        assert row[5] == 0, (
            f"arm {tuple(row[:4])} spills {row[5]} bytes: the two scans share one "
            f"geometry, so a spill in either is a carve that has to change")
        assert row[4] <= row[10], (
            f"arm {tuple(row[:4])} holds {row[4]} registers against a budget of "
            f"{row[10]}")
