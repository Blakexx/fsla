"""I6 gate for the consumer's SMEM OPERAND FORMAT (THUNDER_RUN_SPEC.md §5 K2).

**Written BEFORE the change it gates**, per invariant I6 (the TF32 lesson): a
precision change ships with an oracle-anchored, NON-COMMON-MODE gate, and the
gate is authored against the pre-change kernel so that "it passes" is a
measurement rather than a calibration.

WHAT K2 CHANGES. The write-coefficient panel `WT` (`[BC][BTP]`, leaf-major) and
the scale-folded value tile `Vw` (`[BT][VWP]`) move from fp32 SMEM storage to
bf16 SMEM storage ON THE bf16 PRODUCTION ARM ONLY (`mma_precision='bf16'`); the
tf32x3 exactness arm keeps fp32 storage, because RULING 1's argument -- the fold
needs the full mantissa THERE -- is unchanged. fp32 accumulation is unchanged
everywhere, the `P > 1` partials stay fp32, and the state `S` and its HBM I/O are
untouched.

Symbols: `BC` leafs per owner (one CTA = one owner); `BT` token tile; `DV` = `d_v`
the value width; `WT[i][t]` the write coefficient of leaf `i` at token `t`;
`Vw[t][ch]` the write-gain-scaled value; `sw[t]` the write row scale.

THE GATE IS TWO CLAIMS, AND THEY ARE DIFFERENT KINDS OF CLAIM.

**Claim 1 -- BIT-IDENTITY, where the format was ALREADY bf16.** Every MMA
consumer of `WT` and `Vw` on the bf16 arm ALREADY narrows the operand at the
fragment (`mma_tile` / `mma_tile_fragA` call `pack_bf16x2(clamp_bf16_finite(...))`,
and `pack_bf16x2` is `__floats2bfloat162_rn` -- round-to-nearest-even, the same
rounding `__float2bfloat16` applies at a store). Narrowing at the STORE instead of
at the FRAGMENT therefore produces the identical bf16 value. So on
`normalization='raw'` -- where `WT` and `Vw` have NO non-MMA consumer -- K2 must be
**bit-identical**, not merely close, and this file asserts byte equality of `y`
and the final state against a recorded reference. Likewise the whole `tf32x3` arm
must be byte-unchanged.

**Claim 2 -- an fp64-ANCHORED TOLERANCE, where a scalar reduction really does
move.** Under `normalization='global'` the state's mass column is folded by a
SCALAR reduction, `acc += WT[q][t] * sw[t]`, which today reads `WT` at fp32 and
after K2 reads it at bf16. That is a genuine numeric change and byte equality is
the WRONG gate for it. It is gated against `naive_rola` in fp64 -- a different
arithmetic domain, computed from the same routing tensors but through no shared
code with the kernel, which is what makes it non-common-mode -- at a bound that
is DERIVED FROM THE FORMAT rather than fitted to the observation:

    one bf16 rounding is <= 2^-8 relative; the mass fold contracts `BT` of them
    with fp32 accumulation, so the relative error of the mass column is bounded
    by ~`BT * 2^-8` in the worst case and by ~`sqrt(BT) * 2^-8` in the
    incoherent case. `MASS_RTOL` below is the worst-case bound at the largest
    instantiated `BT`, and it is stated as an inequality on the RATIO READOUT
    `y = num / den`, which is what the product exposes.

If claim 2 fails, the finding is "the mass column needs a tf32 `n`-tile" and THAT
lands -- not a fallback (`[[ship-right-or-stop]]`, and THUNDER_RUN_SPEC §5 K5
states the same disposition for the mass column's own rung).
"""

from __future__ import annotations

import pytest
import torch

from rola.ops.naive import naive_rola
from rola.planner.schedule import Plan
from rola.routing.types import (
    IndependentRouting,
    LeafMassDecay,
    LevelConfig,
    SoftmaxActivation,
    Topology,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the V3 consumer is a CUDA kernel"),
]

_ROUTING = IndependentRouting(
    tied=False, read_activation=SoftmaxActivation(), write_activation=SoftmaxActivation())

#: The mass column's own bound, DERIVED (see the module docstring), not fitted:
#: `BT_max * 2^-8` with `BT_max = 32`, rounded up to a stated constant. It is a
#: bound on the RATIO READOUT, which is what `global` normalization delivers.
MASS_RTOL = 32 * 2.0 ** -8

#: The value path's own bound. Unchanged from the tier-1 suite's `BF16_RTOL`, and
#: deliberately NOT loosened by this batch: K2 must not move it at all (claim 1),
#: so it is here as a REGRESSION TRIPWIRE rather than as the gate's real content.
VALUE_RTOL = 1e-2


def _simplex(shape, p_nonzero, generator, device):
    x = torch.rand(shape, device=device, dtype=torch.float64, generator=generator)
    if p_nonzero < 1.0:
        keep = torch.rand(shape, device=device, dtype=torch.float64,
                          generator=generator) < p_nonzero
        x = x * keep
    x = torch.where(x.sum(-1, keepdim=True) == 0, torch.ones_like(x), x)
    return x / x.sum(-1, keepdim=True)


def _case(widths, BC, BT, d_v, norm, p_read, p_write, *, seed, decay, prec,
          compact, B=2, T=37, H=3):
    """Run the fixture through BOTH the fp64 oracle and the CUDA consumer.

    The two are handed the SAME routing tensors, which is the tier-1 discipline
    (`[[matching-the-naive-antipattern]]`: the only validation target is the
    canonical `naive_rola`). What makes THIS test non-common-mode for a SMEM
    FORMAT change is that the oracle carries the panels in fp64 end to end --
    there is no shared narrowing site for the change to cancel against.
    """
    from rola.ops.consumer import _consumer_forward

    device = "cuda"
    g = torch.Generator(device=device).manual_seed(seed)
    topology = Topology(levels=tuple(LevelConfig(w, _ROUTING) for w in widths), d_v=d_v)
    read_levels = tuple(_simplex((B, T, H, w), p_read, g, device) for w in widths)
    write_levels = tuple(_simplex((B, T, H, w), p_write, g, device) for w in widths)
    g_write = torch.rand(B, T, H, device=device, dtype=torch.float64, generator=g) + 0.5
    g_read = (None if norm == "global"
              else torch.rand(B, T, H, device=device, dtype=torch.float64, generator=g) + 0.5)
    v = torch.randn(B, T, H, d_v, device=device, dtype=torch.float64, generator=g)
    dec = None
    if decay:
        dec = LeafMassDecay(dials=tuple(
            0.05 + 0.85 * torch.rand(H, w, device=device, dtype=torch.float64, generator=g)
            for w in widths))

    y_ref, state_ref = naive_rola(
        v, read_levels, write_levels, g_read, g_write, topology, dec, norm,
        initial_state=None, output_final_state=True)

    plan = Plan(topology=topology, normalization=norm, BT=BT, BC=BC,
                warp_tile=(1, max(1, d_v // 32)), decay=dec is not None,
                mma_precision=prec, compact=compact)
    y, state, _ = _consumer_forward(
        v.to(torch.float32),
        tuple(t.to(torch.float32) for t in read_levels),
        tuple(t.to(torch.float32) for t in write_levels),
        None if g_read is None else g_read.to(torch.float32),
        g_write.to(torch.float32), plan, initial_state=None, decay=dec)
    return y, state, y_ref, state_ref


def _rel(actual, reference):
    scale = max(1e-30, float(reference.abs().max()))
    return float((actual.double() - reference).abs().max()) / scale


# ---------------------------------------------------------------------------
# CLAIM 2 -- the mass column, fp64-anchored, at a FORMAT-DERIVED bound.
# ---------------------------------------------------------------------------
# The `global` arm is the ONLY place a non-MMA consumer reads `WT`: the mass
# state's fold, `acc += WT[q][t] * sw[t]`. Every row below is `global`, and the
# quantity gated is the RATIO READOUT the product delivers, not an internal.
@pytest.mark.parametrize("widths,BC", [((16,), 16), ((8, 8), 64), ((4, 4, 4), 16),
                                       ((4, 4, 4, 4), 16)])
@pytest.mark.parametrize("BT", [16, 32])
@pytest.mark.parametrize("decay", [False, True])
@pytest.mark.parametrize("p_read,p_write", [(1.0, 1.0), (1.0, 0.2), (0.4, 0.3)])
def test_global_mass_column_stays_inside_its_format_bound(widths, BC, BT, decay,
                                                          p_read, p_write):
    """`y = num/den` on the `global` arm, against the fp64 oracle, at `BT * 2^-8`.

    This is the row that MOVES when `WT` becomes a bf16 SMEM buffer. It is
    written before the change so that the recorded margin is a measurement of
    the pre-change kernel, and the post-change run is compared against a bound
    that was fixed in advance.
    """
    y, state, y_ref, state_ref = _case(
        widths, BC, BT, 64, "global", p_read, p_write,
        seed=4242 + BC + BT, decay=decay, prec="bf16", compact=False)
    err_y = _rel(y, y_ref)
    err_mass = _rel(state[..., 64], state_ref[..., 64])
    assert err_y < MASS_RTOL, (
        f"the `global` ratio readout left its format bound: relative {err_y:.3e} "
        f"against {MASS_RTOL:.3e}. If this fires ONLY after the bf16-SMEM change, "
        f"the finding is 'the mass column needs a tf32 n-tile' -- land that, not a "
        f"fallback")
    assert err_mass < MASS_RTOL, (
        f"the state's mass column left its format bound: relative {err_mass:.3e}")


# ---------------------------------------------------------------------------
# CLAIM 1 -- the value path must NOT move at all.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("widths,BC", [((16,), 16), ((8, 8), 64), ((4, 4, 4), 16)])
@pytest.mark.parametrize("BT", [16, 32])
@pytest.mark.parametrize("prec", ["bf16", "tf32x3"])
@pytest.mark.parametrize("decay", [False, True])
def test_raw_value_path_is_unmoved_by_the_smem_format(widths, BC, BT, prec, decay):
    """On `raw` there is no non-MMA consumer of `WT`/`Vw`, so K2 is a pure
    narrowing-site MOVE: the same round-to-nearest-even value reaches the same
    fragment slot in the same accumulation order.

    Byte equality against a recorded reference is the mechanical form of this
    claim and it lives in the dev-loop bit-identity harness (it needs a
    pre-change binary to compare to, which a unit test does not have). What this
    test pins IN-TREE is the tolerance the claim implies: `raw` must meet the
    tier-1 bf16 number exactly as it did before, and the tf32x3 arm must meet the
    exactness number -- a regression on either is the claim failing.
    """
    tol = VALUE_RTOL if prec == "bf16" else 2e-5
    y, state, y_ref, state_ref = _case(
        widths, BC, BT, 64, "raw", 1.0, 0.3,
        seed=515 + BC + BT, decay=decay, prec=prec, compact=False)
    assert _rel(y, y_ref) < tol, f"`raw` output moved: relative {_rel(y, y_ref):.3e}"
    assert _rel(state, state_ref) < tol, (
        f"`raw` final state moved: relative {_rel(state, state_ref):.3e}")
