"""THE BACKWARD'S SEEDS — the readout differentiated, and the fifth output.

Vocabulary, defined here and used unqualified below: `B` batch, `T` tokens, `H`
heads, `d_v = 64` value channels; `num`/`den` the forward's numerator and
denominator; `m` the per-token READ MASS (`packed.rm`), which the producer leaves
on the read levels instead of dividing out; `eps` the denominator floor; `y` the
op's output; `dy` its cotangent; `Dt = den + eps * m` the exact quantity the
divide forms.

The op's readout is `y = num / Dt` with an **ADDITIVE** floor — `naive.py:223`
says `readout[..., d_v:] + eps` and `chunk.cu:203` says `den + eps * mass`, and
neither is a `max`. So there is no kink, no subgradient and no `den`-dependent
branch anywhere in the backward, and the three seeds are exact:

    dnum[t,:] = dy[t,:] / Dt
    dden[t]   = -<dy[t,:], y[t,:]> / Dt
    dm[t]     = eps * dden[t]

`dm` is a FIFTH output nobody had listed. The m-carry is forward-exact but **not
gradient-neutral**: `m` reaches `y` only through `Dt`, so its cotangent contracts
to `eps * dden`. A backward that dropped it would silently zero a real gradient of
every producer that does not declare a simplex. It exists only when a read mass
does — there is no `m` to differentiate in a declared-simplex bundle, and
returning a zero for it would be a claim rather than an absence.

**THE BUDGET IS DERIVED FROM THE ARITHMETIC, term by term, and never tuned.**
The kernel computes in fp32 and rounds the two mma operands to bf16, so for each
output the floor is the sum of exactly the roundings it performs — nothing is
allowed for "noise", and a term the model does not name is a term the gate would
have to be widened to accept, which is the deviation this file's failure message
asks for.
"""
from __future__ import annotations

import pytest
import torch

from rola.ops._ext import extension, is_available

pytestmark = pytest.mark.skipif(
    not is_available() or not torch.cuda.is_available(),
    reason="the seed pass is a CUDA kernel")

DV = 64
#: ROUND-TO-NEAREST UNIT ROUNDOFF, `2 ** -p` for a format of `p` significand bits.
#: bf16 stores 7 and carries one implicit, so `p = 8`; fp32 stores 23, so `p = 24`.
#: (This file first carried `2 ** -9` for bf16 -- an off-by-one in the DERIVATION,
#: caught because every dnum row landed at a ratio of 1.99 rather than scattering.
#: The constant was corrected; the gate was not widened.)
ULP_BF16 = 2.0 ** -8
ULP_F32 = 2.0 ** -24
#: the warp reduction's depth over the `d_v` channels: two sequential adds per
#: lane, then five shuffle levels
REDUCE_DEPTH = 2 + 5


def _fixture(B: int, T: int, H: int, seed: int, dtype, den_scale: float, with_mass: bool):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    dy = torch.randn(B, T, H, DV, device=dev, generator=g).to(dtype)
    y = torch.randn(B, T, H, DV, device=dev, generator=g).to(dtype)
    den = (torch.rand(B, H, T, device=dev, generator=g) * den_scale)
    mass = (torch.rand(B, H, T, device=dev, generator=g) + 0.5) if with_mass else None
    return dy, y, den, mass


def _reference(dy, y, den, mass, eps: float):
    """The seeds in fp64, from the operands ROUNDED THE WAY THE KERNEL HAS THEM.

    `dy` and `y` are already whatever dtype the caller passed; casting them up
    changes nothing. The kernel's own roundings are the two bf16 stores and the
    fp32 arithmetic, and those are what the budget below charges — so the
    reference performs none of them.
    """
    dy64, y64 = dy.double(), y.double()
    den64 = den.double()
    m64 = mass.double() if mass is not None else torch.ones_like(den64)
    dt = den64 + eps * m64
    #: `[B, H, T]` -> `[B, T, H, 1]`, which broadcasts over dy's operand layout;
    #: the seeds then transpose to `[B, H, T, d_v]`, the fold's own
    dtp = dt.permute(0, 2, 1).unsqueeze(-1)
    dnum = (dy64 / dtp).permute(0, 2, 1, 3).contiguous()
    dot = (dy64 * y64).sum(-1).permute(0, 2, 1).contiguous()   # [B, H, T]
    dden = -dot / dt
    return dnum, dden, eps * dden, (dy64.abs() * y64.abs()).sum(-1).permute(0, 2, 1) / dt


CELLS = [
    #: the ordinary cell
    pytest.param(2, 64, 3, 1.0, True, torch.float32, id="fp32 dy, mass"),
    pytest.param(2, 64, 3, 1.0, False, torch.float32, id="fp32 dy, declared simplex"),
    #: the cotangent arrives in the readout's dtype, which is bf16 in production
    pytest.param(1, 128, 4, 1.0, True, torch.bfloat16, id="bf16 dy, mass"),
    #: THE EXPLODING CELL: read support and write support disjoint, so den -> 0,
    #: Dt = eps * m, and dnum = dy / eps is the largest number in the design
    pytest.param(1, 64, 2, 0.0, True, torch.float32, id="den = 0 (the eps floor alone)"),
    #: THE VANISHING CELL for dm: den >> eps, so dm / dden = eps is tiny and must
    #: still be right
    pytest.param(1, 64, 2, 1.0e4, True, torch.float32, id="den >> eps (dm vanishing)"),
    #: a whole batch worth of tokens, so the block tail is exercised
    pytest.param(3, 256, 5, 1.0, True, torch.float32, id="T = 256, five heads"),
]


@pytest.mark.parametrize("B,T,H,den_scale,with_mass,dtype", [p.values for p in CELLS],
                         ids=[p.id for p in CELLS])
@pytest.mark.parametrize("eps", [1e-6, 1e-3], ids=["eps=1e-6", "eps=1e-3"])
def test_the_seeds_are_the_readouts_exact_gradient(B, T, H, den_scale, with_mass, dtype, eps):
    m = extension()
    dy, y, den, mass = _fixture(B, T, H, 5, dtype, den_scale, with_mass)
    got = m.chunk_backward_seeds(dy, y, den, mass, eps)
    assert len(got) == (3 if with_mass else 2), (
        "the fifth output must be present exactly when a read mass is, and absent "
        "otherwise -- a zero for a mass that does not exist would be a claim")
    want_num, want_den, want_m, absdot = _reference(dy, y, den, mass, eps)

    #: dnum: one fp32 divide, then the bf16 store the fold's A operand requires
    bound = (ULP_BF16 + ULP_F32) * want_num.abs()
    err = (got[0].double() - want_num).abs()
    _assert(err, bound, "dnum", "the divide's fp32 rounding and the bf16 operand store")

    #: dden: the same divide, a reduction over d_v whose CANCELLATION is bounded
    #: by the sum of |dy_i * y_i| rather than by the result, and the bf16 store
    bound = ULP_BF16 * want_den.abs() + (REDUCE_DEPTH + 1) * ULP_F32 * absdot
    err = (got[1].double() - want_den).abs()
    _assert(err, bound, "dden", "the channel reduction's fp32 reassociation and the "
                                "bf16 operand store")

    if with_mass:
        #: dm is taken off the fp32 dden, NOT off the bf16 one, so it carries the
        #: reduction's error scaled by eps and no bf16 term at all. If this row
        #: needs the bf16 term, the kernel rounded before it multiplied.
        bound = eps * ((REDUCE_DEPTH + 1) * ULP_F32 * absdot + ULP_F32 * want_den.abs())
        err = (got[2].double() - want_m).abs()
        _assert(err, bound, "dm", "the reduction's fp32 reassociation, scaled by eps")


def _assert(err, bound, name, model):
    tiny = torch.finfo(torch.float64).tiny
    ratio = (err / bound.clamp_min(tiny)).max().item()
    assert ratio <= 1.0, (
        f"the seed {name} left its derived budget: worst |err| / bound = {ratio:.3f} "
        f"(max |err| = {err.max():.3e} against a bound of {bound.max():.3e}). The "
        f"budget charges exactly {model} and nothing else. Do not widen it -- record "
        f"the row, the deviation, and which term of the floor model under-charges.")


def test_dm_is_eps_times_dden_and_is_not_zero():
    """SURPRISE 2, asserted rather than described.

    A backward that dropped `dm` would pass every gate above except this one, and
    would silently zero a real gradient of every producer that does not declare a
    simplex (`factors.py:23-37` -- every foreign producer).
    """
    m = extension()
    eps = 1e-3
    dy, y, den, mass = _fixture(2, 64, 3, 5, torch.float32, 1.0, True)
    dnum, dden, dm = m.chunk_backward_seeds(dy, y, den, mass, eps)
    assert dm.abs().max().item() > 0.0, (
        "dm is identically zero: the m-carry is forward-exact but NOT "
        "gradient-neutral, and this is the term that says so")
    #: the exact algebra, to the fp32 rounding of one multiply
    assert torch.allclose(dm.double(), eps * dden.double(), rtol=ULP_BF16 * 2, atol=0.0), (
        "dm is not eps * dden: the floor is additive, so m reaches y only through "
        "Dt and its cotangent contracts to exactly this")


def test_the_budget_is_not_vacuous():
    """A gate that cannot fail is not a gate: the smallest structural error
    available is the floor left off the denominator, and the budget must reject
    it at the cell where the floor is what the denominator IS."""
    m = extension()
    eps = 1e-3
    dy, y, den, mass = _fixture(1, 64, 2, 5, torch.float32, 0.0, True)
    dnum, dden, dm = m.chunk_backward_seeds(dy, y, den, mass, eps)
    want_num, _, _, _ = _reference(dy, y, den, mass, eps)
    #: the same seeds computed with an unscaled floor -- `eps` instead of `eps * m`,
    #: which is the m-carry being forgotten rather than differentiated
    unscaled, _, _, _ = _reference(dy, y, den, None, eps)
    bound = (ULP_BF16 + ULP_F32) * want_num.abs()
    assert ((unscaled - want_num).abs() > bound).any(), (
        "the budget accepts a denominator floored at eps rather than at eps * m, so "
        "it is not measuring the m-carry at all")
