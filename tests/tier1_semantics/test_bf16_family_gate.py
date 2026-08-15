"""GATE I6-B -- the bf16 MEMORY-FAMILY gate.

The kernel arm stores its routing amplitudes and its value tensor in bf16 and
accumulates in fp32. Everything else -- state, mass, the num/den accumulators --
stays fp32, and `y` comes back in the operand dtype with the fp32 accumulator's
precision behind it. The comparison basis is the fp64 oracle, and the derived
budget is `tolerances.BF16_RTOL`.

**The file's job is to prove that budget can SEE a defect of the family's own
magnitude.** A tolerance nobody has tried to break is a number, not a bound. So
the claim carries a MUTANT -- a value-level quantization injected on the KERNEL
side only, with the fp64 oracle keeping the exact input -- and the mutant
carries an unmutated control, because a mutant whose control does not pass
proves nothing.

P67 D2-b RE-ANCHORED this file onto the chunk arm, and the re-anchor SIMPLIFIED
the staging in a way worth stating, because it changes what a pass means:

* the narrowing used to be STAGED, with quantizers applied to the tiled
  consumer's fp32 inputs. `rola.ops.chunk.run` casts the routing sides and `v`
  to bf16 itself, so on this arm the narrowing IS what the kernel does and the
  budget row MEASURES it rather than emulating it. `q_amps`/`q_v` at bf16 would
  be no-ops and are gone. What remains staged is `q_y` -- the kernel hands back
  the operand dtype, so a caller who STORES `y` in bf16 adds a rounding the
  kernel did not -- and M3's coarser-than-bf16 amplitude mutant, which is a real
  perturbation precisely because it is applied before the kernel's own cast.
* the tiled arm distinguished FIXED from SPANNED levels: only the spanned ones
  were stored bf16, and the mutant scope followed
  `Plan.geometry.first_spanned_level`. The chunk arm concatenates every level
  and casts the whole thing, so the distinction does not exist and the mutant
  applies to ALL levels. That is a wider mutant scope, not a narrower one.

RETIRED WITH THEIR SUBJECTS, both already inert before this commit:

* **M2**, the decay-clock mutant (`write_levels_stored` at bf16 -- the §4.7
  exponent-amplification argument that makes "the clock stays fp32" a checked
  property). Decay is retired from the operator: there is no clock in the kernel
  arm to pin, and `write_levels_stored` was an argument of the retired tiled
  entry. It
  was `pytest.skip`-ed at the decay retirement, so its removal changes nothing
  that ran.
* **the decay arm** of the family-budget and T-scaling rows, for the same
  reason. The T-scaling row's UNDECAYED arm was skipped along with it and only
  the decay one had cause, so this commit turns a skip back into a running
  measurement.

RETIRED AS A CLAIM, stated rather than dropped: **support invariance under the
amplitude cast.** The tiled gate read a SCHEDULE's `read_token_bits` at a
planted sub-bf16-subnormal amplitude and asserted the liveness bit was set while
the stored coefficient was a hard zero -- support decided strictly above the
cast. The chunk arm builds no read-side support bitmap at all (it multiplies
through the cast amplitudes; residency is a WRITE-side fact from `chunk_stats`,
gated in `tests/tier2_geometry/test_atom_bitmap.py`), so there is no support
decision for the cast to sit above. What survives, and is kept below, is the
half that is still checkable and is why the claim mattered: a live leaf whose
stored amplitude flushed to zero is INERT.

The two arms `PARITY_BATCH_DESIGN.md` §2.B3/§2.B4 demand are both here:

* **the T-scaling arm** -- the narrowing at `T in {512, 2048, 8192}`, asserting
  the deviation is FLAT in `T` (fitted log-log slope well below the `sqrt(T)`
  walk's `+0.5`). This is the batch-1 lesson: bf16 RESIDENT STATE died on a
  `sqrt(T)` re-quantization walk that a short fixture passes; the amplitudes,
  the value tensor and `y` are read-only -- no accumulator, no walk -- and that
  claim is MEASURED here, not asserted.
* **the flush arm** -- a planted amplitude below bf16's smallest subnormal is
  inert, on a fixture where the flush is asserted to actually happen.
"""

from __future__ import annotations

import math

import pytest
import torch

from rola.ops import chunk as _chunk
from rola.ops.naive import naive_rola
from rola.routing.factors import RouteFactors
from tests.tier1_semantics.oracle_fixtures import _output_charge, _simplex, _topology
from tests.tier1_semantics.tolerances import BF16_RTOL

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the chunk consumer is a CUDA kernel"),
]


def q_bf16(t: torch.Tensor) -> torch.Tensor:
    """One bf16 rounding, in the caller's dtype: the family's own perturbation."""
    return t.to(torch.bfloat16).to(t.dtype)


def q_mant4(t: torch.Tensor) -> torch.Tensor:
    """Round to 4 mantissa bits (fp8-ish): the anti-slack format, two rungs below
    bf16. Valid for the nonnegative amplitude domain this file applies it to."""
    f = t.to(torch.float32)
    keep = 23 - 4
    bits = f.view(torch.int32)
    rounded = (bits + (1 << (keep - 1))) & ~((1 << keep) - 1)
    return rounded.view(torch.float32).to(t.dtype)


def _family_run(widths, d_v, B, T, H, p_read, p_write, *, seed,
                q_amps=None, q_y=None, read_levels=None):
    """`(y, state, y_ref, state_ref)` -- the chunk arm and the fp64 oracle.

    `q_amps` is applied to BOTH routing sides at every level BEFORE the call, so
    the kernel's own bf16 cast happens on top of it and only a quantizer COARSER
    than bf16 perturbs anything. `q_y` is applied to the returned output, which
    the kernel hands back at the operand dtype. `normalization` is always
    `"global"`: it is the only value the chunk arm carries.
    """
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed)
    topology = _topology(widths, d_v)
    if read_levels is None:
        read_levels = tuple(_simplex((B, T, H, w), p_read, gen, device) for w in widths)
    write_levels = tuple(_simplex((B, T, H, w), p_write, gen, device) for w in widths)
    g_write = torch.rand(B, T, H, device=device, dtype=torch.float64, generator=gen) + 0.5
    v = torch.randn(B, T, H, d_v, device=device, dtype=torch.float64, generator=gen)
    cols = d_v + 1
    initial_state = 0.1 * torch.randn(
        B, H, topology.N, cols, device=device, dtype=torch.float64, generator=gen)
    initial_state[..., d_v] = initial_state[..., d_v].abs()

    y_ref, state_ref = naive_rola(
        v, tuple(read_levels), write_levels, None, g_write, topology, None, "global",
        initial_state=initial_state, output_final_state=True)

    def quantized(levels):
        if q_amps is None:
            return tuple(levels)
        out = []
        for t in levels:
            qt = q_amps(t)
            # The mutant must perturb VALUES, never SUPPORT -- otherwise it tests
            # a bitmap difference, which is a different (and gated) property.
            assert bool(((qt == 0) == (t == 0)).all()), (
                "the quantizer changed the support set; this mutant is malformed")
            out.append(qt)
        return tuple(out)

    routes = RouteFactors(
        topology=topology, normalization="global",
        read=tuple(t.to(torch.float32) for t in quantized(read_levels)),
        write=tuple(t.to(torch.float32) for t in quantized(write_levels)),
        g_read=None, g_write=g_write.to(torch.float32))
    refusal = _chunk.unsupported(routes, None)
    assert refusal is None, f"the fixture left the built chunk matrix: {refusal}"
    y, state = _chunk.run(
        routes, v.to(torch.float32), want_state=True,
        state_in=initial_state.to(torch.float32).reshape(
            B * H, topology.N // 16, 16, cols))
    if q_y is not None:
        y = q_y(y)

    return y, state.reshape(B, H, topology.N, cols), y_ref, state_ref


def _charge(y, state, y_ref, state_ref) -> tuple[float, float]:
    return (_output_charge(y, y_ref),
            float((state.double() - state_ref).abs().max())
            / max(1e-30, float(state_ref.abs().max())))


def _family_errors(*a, **k) -> tuple[float, float]:
    """`(e_out, e_state)` of the chunk arm against the canonical fp64 oracle."""
    return _charge(*_family_run(*a, **k))


#: The topology rows the family gate runs, all BUILT arms of the chunk manifest:
#: the flagship-shaped `D = 2` cell, a `D = 3` cell, and the §4.4 WORST CASE
#: `D = 4` (four routed factors multiplied into every leaf coefficient, with no
#: margin argued -- measured here). Pre-D2 these were `(8,8)`, `(4,4,4)` and
#: `(2,2,2,2)`: the DEPTH axis is what the row set is for, and it is preserved;
#: the widths moved to the ones the manifest builds.
_FAMILY_CELLS = [(8, 8), (16, 16, 16), (8, 8, 8, 8)]

#: `T` must be a multiple of the operator's chunk (`rola.ops.chunk.CHUNK_TOKENS`).
#: `37` -- the tiled file's deliberately ragged length -- has no kernel arm now.
_T = 64


# --- the production arm must HOLD ITS BUDGET under the full family ----------


@pytest.mark.parametrize("widths", _FAMILY_CELLS, ids=lambda w: "x".join(map(str, w)))
def test_family_narrowing_holds_bf16_budget(widths):
    """bf16 amplitudes + bf16 `v` + a bf16-stored `y` TOGETHER stay under
    `BF16_RTOL`.

    The first two are the kernel's own casts; the third is the caller's, staged
    with `q_y`. If the stacked roundings blow the budget on any topology -- the
    `D = 4` row has no argued margin -- the family halts here.
    """
    e_out, e_state = _family_errors(
        widths=widths, d_v=64, B=2, T=_T, H=3,
        p_read=0.4, p_write=0.3, seed=90210, q_y=q_bf16)
    assert e_out < BF16_RTOL, f"family narrowing blows the output budget: {e_out:.3e}"
    assert e_state < BF16_RTOL, f"family narrowing blows the state budget: {e_state:.3e}"


# --- M3: the anti-slack mutant ---------------------------------------------


@pytest.mark.parametrize("widths", [(8, 8), (8, 8, 8, 8)], ids=lambda w: "x".join(map(str, w)))
def test_M3_four_mantissa_bit_amps_exceed_BF16_RTOL(widths):
    """A format two rungs below bf16 must BREACH the budget.

    Otherwise `BF16_RTOL` is slack enough to swallow the change under test and
    gates nothing. The control is the same fixture unmutated and must pass.
    """
    kw = dict(widths=widths, d_v=64, B=2, T=_T, H=3,
              p_read=0.4, p_write=0.3, seed=90210)
    clean_out, clean_state = _family_errors(**kw)
    assert clean_out < BF16_RTOL and clean_state < BF16_RTOL, (
        f"M3's control does not pass (out {clean_out:.3e}, state {clean_state:.3e})")
    e_out, e_state = _family_errors(q_amps=q_mant4, **kw)
    assert max(e_out, e_state) > BF16_RTOL, (
        f"4-mantissa-bit amplitudes moved the gate only to out {e_out:.3e} / state "
        f"{e_state:.3e}: BF16_RTOL is slack enough to swallow a format two rungs below "
        "the change under test, so it gates nothing")


# --- the T-scaling arm (§2.B3): flat in T, or the family halts --------------


_T_ARM = (512, 2048, 8192)


def test_family_deviation_is_flat_in_T():
    """The read-only/no-walk claim, measured. KILL CONDITION: a monotone T-trend.

    A resident-state walk grows as `sqrt(T)` (measured exponent +0.49, J1-B); a
    read-only rounding is flat. The fitted log-log slope over the three lengths
    must sit far below the walk's, and every point must clear the budget.
    """
    errs = []
    for T in _T_ARM:
        e_out, e_state = _family_errors(
            widths=(8, 8), d_v=64, B=2, T=T, H=3,
            p_read=0.4, p_write=0.3, seed=90210, q_y=q_bf16)
        assert e_out < BF16_RTOL, f"T={T}: family narrowing blows the budget: {e_out:.3e}"
        assert e_state < BF16_RTOL, f"T={T}: family narrowing blows the budget: {e_state:.3e}"
        errs.append(e_out)
    xs = [math.log(t) for t in _T_ARM]
    ys = [math.log(max(e, 1e-30)) for e in errs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
             / sum((x - mx) ** 2 for x in xs))
    detail = ", ".join(f"T={t}: {e:.3e}" for t, e in zip(_T_ARM, errs))
    assert slope < 0.25, (
        f"the family deviation TRENDS with T (fitted log-log slope {slope:+.3f}; {detail}). "
        "A trend that clears 1e-2 at T=8192 does not clear it at nc~65K, which is the "
        "standing scale target -- the family halts pending re-derivation (batch-1 rule)")


# --- the flush arm (§2.B4): a flushed live amplitude is INERT ---------------


def test_a_sub_subnormal_amplitude_flushes_and_is_inert():
    """A live amplitude below bf16's smallest subnormal stores as a hard zero,
    and the kernel run on that fixture still matches the oracle in budget.

    The flush is ASSERTED rather than assumed -- `1e-41` is an fp32 subnormal and
    a bf16 zero, checked here at the planted value after renormalization -- so
    the cell cannot go green by the plant quietly surviving the cast. A leaf
    whose stored coefficient is zero contributes exactly zero to every product it
    enters, and `1e-41` of amplitude moves nothing at `1e-2`; that is what inert
    means, and the oracle comparison is what measures it.
    """
    device = "cuda"
    widths, d_v, B, T, H = (8, 8), 64, 2, _T, 3
    gen = torch.Generator(device=device).manual_seed(2718)
    # Dense on the planted level so every digit starts live -- the plant then has
    # company by construction rather than by chance.
    read_levels = [_simplex((B, T, H, w), 1.0 if i == len(widths) - 1 else 0.4, gen, device)
                   for i, w in enumerate(widths)]
    level_idx, digit, token = len(widths) - 1, 3, 5
    tiny = read_levels[level_idx].clone()
    tiny[0, token, 0, digit] = 1e-41
    tiny[0, token, 0] = tiny[0, token, 0] / tiny[0, token, 0].sum()
    read_levels[level_idx] = tiny

    planted = tiny[0, token, 0, digit]
    assert float(planted) > 0.0, "the plant did not survive renormalization in fp64"
    assert float(planted.to(torch.float32).to(torch.bfloat16)) == 0.0, (
        f"the planted amplitude {float(planted)!r} does NOT flush under bf16 storage: "
        "the fixture no longer tests the flush direction and must be re-derived, "
        "not deleted")

    e_out, e_state = _family_errors(
        widths=widths, d_v=d_v, B=B, T=T, H=H, p_read=0.4, p_write=0.3,
        seed=2718, read_levels=tuple(read_levels))
    assert e_out < BF16_RTOL and e_state < BF16_RTOL, (
        f"the planted-flush fixture is out of budget (out {e_out:.3e}, state "
        f"{e_state:.3e}): a flushed-to-zero live leaf is NOT inert")


# --- the CHARGE itself, and the mass column's own localization --------------


def test_output_charge_sees_a_drift_the_global_max_cannot():
    """The metric the budget rows above are stated in has to be the right one.

    The readout is a ratio whose denominator is accumulated write mass, so at
    `t = 0` -- nothing accumulated -- `|y|` is orders of magnitude above the rest
    of the sequence. A single global normalizer is therefore set by token 0, and
    both the numerator's max and the normalizer's max land on the token whose
    state has been updated zero times: that metric is structurally incapable of
    seeing an error that ACCUMULATES. A `sqrt(t)` drift ten times the budget is
    planted, and the two forms must disagree about it -- the old one passing it
    is the premise, the new one catching it is the claim.
    """
    T = 2048
    y, _state, y_ref, _state_ref = _family_run(
        widths=(8, 8), d_v=64, B=2, T=T, H=3, p_read=0.4, p_write=0.3, seed=90210)

    def global_charge(a):
        return float((a.double() - y_ref).abs().max()) / max(1e-30, float(y_ref.abs().max()))

    assert global_charge(y) < BF16_RTOL and _output_charge(y, y_ref) < BF16_RTOL, (
        "the unperturbed control must pass under BOTH forms, or the mutant proves "
        f"nothing (global {global_charge(y):.3e}, per-segment {_output_charge(y, y_ref):.3e})")

    p = 10.0 * BF16_RTOL
    ramp = (torch.arange(T, device=y.device, dtype=torch.float64) / (T - 1)).sqrt()
    mutant = y.double() * (1.0 + p * ramp[None, :, None, None])

    assert global_charge(mutant) < BF16_RTOL, (
        f"a {p:.1e} sqrt-t drift moved the OLD global-max charge to "
        f"{global_charge(mutant):.3e}, i.e. past {BF16_RTOL:.0e}: the premise of this "
        "claim is that the old form passes this mutant, and it no longer does -- "
        "re-derive the mutant's size rather than deleting the claim")
    assert _output_charge(mutant, y_ref) > BF16_RTOL, (
        f"a {p:.1e} sqrt-t drift moved the per-segment charge only to "
        f"{_output_charge(mutant, y_ref):.3e}: the replacement metric is as blind as "
        "the one it replaced")


def test_the_mass_column_is_charged_separately_from_the_value_block():
    """A mass-column defect must not be able to hide inside a value-dominated max.

    The state is `[..., d_v + 1]`: `d_v` value channels and ONE mass column, and
    the two are on different scales whenever the write gains are. `e_state` above
    is a single max over both, so a mass error smaller than the value block's
    largest deviation is invisible to it. This row charges the two blocks
    separately, and PROVES the split is load-bearing by perturbing the mass
    column alone and requiring exactly the mass charge to move.

    (The tiled predecessor made the same point through a `_mass_errors` helper
    that injected into the kernel's own mass accumulator. There is no such
    injection point on the chunk arm, so the perturbation is applied to the
    returned state -- which tests the CHARGE's localization, which is what the
    predecessor's third assertion was actually about.)
    """
    d_v = 64
    _y, state, _y_ref, state_ref = _family_run(
        widths=(8, 8), d_v=d_v, B=2, T=_T, H=3, p_read=1.0, p_write=1.0, seed=31337)

    def charges(s):
        val = (float((s[..., :d_v].double() - state_ref[..., :d_v]).abs().max())
               / max(1e-30, float(state_ref[..., :d_v].abs().max())))
        mass = (float((s[..., d_v].double() - state_ref[..., d_v]).abs().max())
                / max(1e-30, float(state_ref[..., d_v].abs().max())))
        return val, mass

    clean_val, clean_mass = charges(state)
    assert clean_val < BF16_RTOL and clean_mass < BF16_RTOL, (
        "the unperturbed control must pass on BOTH blocks, or the perturbation "
        f"below proves nothing (value {clean_val:.3e}, mass {clean_mass:.3e})")

    perturbed = state.double().clone()
    perturbed[..., d_v] *= 1.0 + 100.0 * BF16_RTOL
    bad_val, bad_mass = charges(perturbed)
    assert bad_mass > BF16_RTOL, (
        f"a 100x-budget mass perturbation moved the mass charge only to "
        f"{bad_mass:.3e}: the split is not gating the mass column")
    assert bad_val < BF16_RTOL, (
        f"perturbing the mass column moved the VALUE charge to {bad_val:.3e}: the "
        "two charges are not independent and the localization is unsound")
