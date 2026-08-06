"""GATE I6-B, EXTENDED — the bf16 MEMORY-FAMILY gate, written before the family.

The phase-2 family stores five surfaces in bf16, keyed to ``mma_precision``
(the ruling of 2026-08-04): the SPANNED-level amplitudes, the value tensor, the
v/o projection GEMM operands, and the output ``y`` cast after the fp32 divide,
with the host boundary relaxed to let those dtypes through. Everything else --
state, mass, num/den accumulators, the decay clock, the FIXED-level amplitudes
(the per-token row scales), and the producer/support tensors -- stays fp32.

**This file is the gate, and it exists before the code it gates.** The mutants
below are VALUE-LEVEL quantizations injected on the KERNEL side only (the fp64
oracle keeps the exact input), so every one of them runs on the pre-change tree:
they prove the two tolerances can SEE a defect of the family's own magnitude
before the family is allowed to introduce that magnitude on the arm where it is
legal. Each mutant carries an unmutated control, because a mutant whose control
does not pass proves nothing.

The named mutants (`PARITY_BATCH_DESIGN.md` §2.B2, extended to the family):

* **M1** amplitudes at bf16 values under ``tf32x3``  -> must exceed ``RTOL``.
* **M2** clock values at bf16 on a decay arm         -> must exceed ``BF16_RTOL``
  (the exponent amplification argument, §4.7 -- this is what makes "the clock
  stays fp32" a checked property rather than a comment).
* **M3** amplitudes at 4 mantissa bits on the bf16 arm -> must exceed
  ``BF16_RTOL`` (the anti-slack mutant: the tolerance is not wide enough to
  swallow a format two rungs below the one under test).
* **M4** value tensor at bf16 values under ``tf32x3`` -> must exceed ``RTOL``.
* **M5** ``y`` at bf16 values under ``tf32x3``        -> must exceed ``RTOL``.

M1/M4/M5 are the SCOPE gate: the tf32x3 arm holding at ``RTOL`` is what catches
an accidental global cast, so it must demonstrably fail when handed exactly the
narrowing the production arm is about to adopt.

Plus the two arms §2.B3/§2.B4 demand:

* **the T-scaling arm** -- the full family's narrowing at ``T in {512, 2048,
  8192}``, asserting the deviation is FLAT in ``T`` (fitted log-log slope well
  below the ``sqrt(T)`` walk's ``+0.5``). This is the batch-1 lesson: bf16
  RESIDENT STATE died on a ``sqrt(T)`` re-quantization walk that a short fixture
  passes; the amplitudes/value/y are read-only -- no accumulator, no walk -- and
  that claim is MEASURED here, not asserted.
* **support invariance** -- the amplitude cast happens strictly BELOW support
  selection, so the schedule's bitmaps/owners are byte-identical between
  ``amp_dtype`` fp32 and bf16, INCLUDING on a fixture where the cast provably
  flushes values to zero (the flush count is asserted nonzero so the claim is
  tested rather than vacuously true; the flushed leaf stays bitmap-live and is
  inert by zero-route annihilation, spec §12 rule 4).
"""

from __future__ import annotations

import math

import pytest
import torch

from rola.ops.naive import naive_rola
from rola.planner.schedule import Plan, build_schedule
from rola.routing.types import LeafMassDecay
from tests.tier1_semantics.test_consumer_vs_oracle import (
    BF16_RTOL,
    RTOL,
    _decay_dials,
    _output_charge,
    _simplex,
    _topology,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="V3 consumer is a CUDA kernel"),
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


def _family_errors(widths, BC, BT, d_v, B, T, H, norm, p_read, p_write, *, seed,
                   mma_precision, decay=False, q_amps=None, q_clock=None, q_v=None,
                   q_y=None, write_levels=None, dials=None, g_write=None):
    """`(e_out, e_state)` of the kernel vs the canonical fp64 oracle, with the
    named value-level quantizers applied to the KERNEL's inputs/output only.

    ``q_amps`` applies to the SPANNED levels (``l >= j``) of both routing sides --
    the family's Option B scope; fixed levels and the row scales they feed stay
    exact. ``q_clock`` applies to the kernel's ``write_levels_stored`` (the decay
    clock's source) while the oracle keeps the exact clock. ``q_v``/``q_y`` apply
    to the value tensor and the output.
    """
    from rola.ops.consumer import _consumer_forward

    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed)
    topology = _topology(widths, d_v)
    read_levels = tuple(_simplex((B, T, H, w), p_read, gen, device) for w in widths)
    if write_levels is None:
        write_levels = tuple(_simplex((B, T, H, w), p_write, gen, device) for w in widths)
    if g_write is None:
        g_write = torch.rand(B, T, H, device=device, dtype=torch.float64, generator=gen) + 0.5
    g_read = (None if norm == "global"
              else torch.rand(B, T, H, device=device, dtype=torch.float64, generator=gen) + 0.5)
    v = torch.randn(B, T, H, d_v, device=device, dtype=torch.float64, generator=gen)
    cols = d_v + (1 if norm == "global" else 0)
    initial_state = 0.1 * torch.randn(
        B, H, topology.N, cols, device=device, dtype=torch.float64, generator=gen)
    if norm == "global":
        initial_state[..., d_v] = initial_state[..., d_v].abs()

    if decay and dials is None:
        dials = _decay_dials(widths, H, gen, device)
    decay_obj = None if dials is None else LeafMassDecay(dials=dials)

    y_ref, state_ref = naive_rola(
        v, read_levels, write_levels, g_read, g_write, topology, decay_obj, norm,
        initial_state=initial_state, output_final_state=True)

    plan = Plan(topology=topology, normalization=norm, BT=BT, BC=BC,
                warp_tile=(1, max(1, d_v // 32)), decay=decay,
                mma_precision=mma_precision)
    j = plan.geometry.first_spanned_level

    def spanned(levels, q):
        if q is None:
            return levels
        out = []
        for level, t in enumerate(levels):
            if level < j:
                out.append(t)
                continue
            qt = q(t)
            # The mutant must perturb VALUES, never SUPPORT -- otherwise it tests
            # a bitmap difference, which is a different (and gated) property.
            assert bool(((qt == 0) == (t == 0)).all()), (
                "the quantizer changed the support set; this mutant is malformed")
            out.append(qt)
        return out

    k_read = tuple(spanned(read_levels, q_amps))
    k_write = tuple(spanned(write_levels, q_amps))
    k_stored = None
    if q_clock is not None:
        assert decay, "the clock mutant is a decay-arm mutant"
        k_stored = tuple(q_clock(t) for t in write_levels)
    k_v = v if q_v is None else q_v(v)

    y, state, _schedule = _consumer_forward(
        k_v.to(torch.float32),
        tuple(t.to(torch.float32) for t in k_read),
        tuple(t.to(torch.float32) for t in k_write),
        None if g_read is None else g_read.to(torch.float32),
        g_write.to(torch.float32),
        plan,
        initial_state=initial_state.to(torch.float32),
        decay=decay_obj,
        write_levels_stored=None if k_stored is None else tuple(
            t.to(torch.float32) for t in k_stored))
    if q_y is not None:
        y = q_y(y)

    e_out = _output_charge(y, y_ref)
    e_state = (float((state.double() - state_ref).abs().max())
               / max(1e-30, float(state_ref.abs().max())))
    return e_out, e_state


#: The topology rows the family gate runs: the flagship-shaped one-spanned-level
#: cell, a fixed-level cell (`j = 1`), and the §4.4 WORST CASE `D = 4, j = 0`
#: (four spanned factors, no margin argued -- measured here).
_FAMILY_CELLS = [((8, 8), 64), ((4, 4, 4), 16), ((2, 2, 2, 2), 16)]


def _family_quantizers():
    """The production arm's full narrowing, as value-level quantizers."""
    return dict(q_amps=q_bf16, q_v=q_bf16, q_y=q_bf16)


# --- the production arm must HOLD ITS BUDGET under the full family ----------


@pytest.mark.parametrize("widths,BC", _FAMILY_CELLS)
@pytest.mark.parametrize("norm", ["raw", "global"])
def test_family_narrowing_holds_bf16_budget(widths, BC, norm):
    """bf16 spanned amps + bf16 v + bf16 y TOGETHER stay under `BF16_RTOL`.

    This is the family's feasibility measured before it is built: if the stacked
    roundings blow the budget on any topology (the `D=4, j=0` row has no argued
    margin), the family halts here, before any kernel change.
    """
    e_out, e_state = _family_errors(
        widths=widths, BC=BC, BT=16, d_v=64, B=2, T=37, H=3, norm=norm,
        p_read=0.4, p_write=0.3, seed=90210, mma_precision="bf16",
        **_family_quantizers())
    assert e_out < BF16_RTOL, f"family narrowing blows the output budget: {e_out:.3e}"
    assert e_state < BF16_RTOL, f"family narrowing blows the state budget: {e_state:.3e}"


@pytest.mark.parametrize("widths,BC", [((8, 8), 64), ((2, 2, 2, 2), 16)])
def test_family_narrowing_holds_budget_under_decay(widths, BC):
    """Same claim on the decay arm -- with the clock EXACT, as the family pins it."""
    e_out, e_state = _family_errors(
        widths=widths, BC=BC, BT=16, d_v=64, B=2, T=37, H=3, norm="global",
        p_read=0.4, p_write=0.3, seed=555, mma_precision="bf16", decay=True,
        **_family_quantizers())
    assert e_out < BF16_RTOL, f"decay-arm family narrowing blows the budget: {e_out:.3e}"
    assert e_state < BF16_RTOL, f"decay-arm family narrowing blows the budget: {e_state:.3e}"


# --- the mutants: each must FAIL its arm's tolerance ------------------------


@pytest.mark.parametrize("widths,BC", [((8, 8), 64), ((2, 2, 2, 2), 16)])
def test_M1_bf16_amplitudes_under_tf32x3_exceed_RTOL(widths, BC):
    kw = dict(widths=widths, BC=BC, BT=16, d_v=64, B=2, T=37, H=3, norm="global",
              p_read=0.4, p_write=0.3, seed=90210, mma_precision="tf32x3")
    clean_out, clean_state = _family_errors(**kw)
    assert clean_out < RTOL and clean_state < RTOL, (
        f"M1's control does not pass (out {clean_out:.3e}, state {clean_state:.3e})")
    e_out, e_state = _family_errors(q_amps=q_bf16, **kw)
    assert max(e_out, e_state) > RTOL, (
        f"bf16-valued amplitudes under tf32x3 moved the gate only to out {e_out:.3e} / "
        f"state {e_state:.3e}: RTOL cannot see the narrowing it must forbid on this arm")


def test_M2_bf16_clock_values_exceed_BF16_RTOL():
    """The clock's fp32 pin, enforced by a mutant with a VISIBLE amplified exponent.

    The clock error enters survival as `exp(eps * E) - 1` with `E` the elapsed
    decay exponent (§4.7; clamp at 88), but a fixture with continuous writes hides
    it: every token's readout is dominated by its FRESH deposits, whose elapsed
    exponent -- and therefore whose amplified error -- is near zero. So the
    fixture makes the elapsed exponent the only thing left to read:

    * COHERENT concentrated writes (one dominant digit per (batch, head), value
      0.502 -- maximally far from a bf16 grid point at [0.5, 1)'s 2^-8 ulp) make
      the clock prefix accumulate the SAME signed rounding every token;
    * the write gain drops to ~0 after the first 32 tokens, so every later token
      reads the DECAYED early state, whose survival carries the full elapsed
      exponent (~35 nats by the last segment, inside fp32's range);
    * `norm='raw'` (the ratio readout cancels a common survival error between
      num and den) and the PER-SEGMENT output charge (each segment normalizes by
      its own decayed scale) keep the amplified error visible.

    Predicted deviation at the tail: `exp(~7.6e-3 * 35) - 1 ~ 0.3`, against a
    bf16-MMA control floor of ~5e-3 -- an order past `BF16_RTOL` either way.
    """
    device = "cuda"
    widths, B, T, H = (8, 8), 2, 512, 3
    gen = torch.Generator(device=device).manual_seed(777)
    concentrated = []
    for w in widths:
        digit = torch.randint(w, (B, 1, H, 1), device=device, generator=gen).expand(B, T, H, 1)
        level = torch.full((B, T, H, w), 0.498 / (w - 1), device=device, dtype=torch.float64)
        level = level.scatter(-1, digit, 0.502)
        concentrated.append(level)
    dials = tuple(torch.full((H, w), 0.86, device=device, dtype=torch.float64)
                  for w in widths)
    g_write = torch.rand(B, T, H, device=device, dtype=torch.float64,
                         generator=gen) + 0.5
    g_write[:, 32:, :] = 1e-12
    kw = dict(widths=widths, BC=64, BT=16, d_v=64, B=B, T=T, H=H, norm="raw",
              p_read=1.0, p_write=0.6, seed=555, mma_precision="bf16", decay=True,
              write_levels=tuple(concentrated), dials=dials, g_write=g_write)
    clean_out, clean_state = _family_errors(**kw)
    assert clean_out < BF16_RTOL and clean_state < BF16_RTOL, (
        f"M2's control does not pass (out {clean_out:.3e}, state {clean_state:.3e})")
    e_out, e_state = _family_errors(q_clock=q_bf16, **kw)
    assert max(e_out, e_state) > BF16_RTOL, (
        f"a bf16-valued clock moved the gate only to out {e_out:.3e} / state {e_state:.3e}: "
        "the clock's fp32 pin is not load-bearing on this fixture, so the pin is unenforced")


@pytest.mark.parametrize("widths,BC", [((8, 8), 64), ((2, 2, 2, 2), 16)])
def test_M3_four_mantissa_bit_amps_exceed_BF16_RTOL(widths, BC):
    kw = dict(widths=widths, BC=BC, BT=16, d_v=64, B=2, T=37, H=3, norm="global",
              p_read=0.4, p_write=0.3, seed=90210, mma_precision="bf16")
    clean_out, clean_state = _family_errors(**kw)
    assert clean_out < BF16_RTOL and clean_state < BF16_RTOL, (
        f"M3's control does not pass (out {clean_out:.3e}, state {clean_state:.3e})")
    e_out, e_state = _family_errors(q_amps=q_mant4, **kw)
    assert max(e_out, e_state) > BF16_RTOL, (
        f"4-mantissa-bit amplitudes moved the gate only to out {e_out:.3e} / state "
        f"{e_state:.3e}: BF16_RTOL is slack enough to swallow a format two rungs below "
        "the change under test, so it gates nothing")


def test_M4_bf16_value_tensor_under_tf32x3_exceeds_RTOL():
    kw = dict(widths=(8, 8), BC=64, BT=16, d_v=64, B=2, T=37, H=3, norm="global",
              p_read=0.4, p_write=0.3, seed=90210, mma_precision="tf32x3")
    clean_out, clean_state = _family_errors(**kw)
    assert clean_out < RTOL and clean_state < RTOL, (
        f"M4's control does not pass (out {clean_out:.3e}, state {clean_state:.3e})")
    e_out, e_state = _family_errors(q_v=q_bf16, **kw)
    assert max(e_out, e_state) > RTOL, (
        f"a bf16-valued value tensor under tf32x3 moved the gate only to out {e_out:.3e} / "
        f"state {e_state:.3e}: the exactness arm cannot see the family's v narrowing")


def test_M5_bf16_y_under_tf32x3_exceeds_RTOL():
    kw = dict(widths=(8, 8), BC=64, BT=16, d_v=64, B=2, T=37, H=3, norm="global",
              p_read=0.4, p_write=0.3, seed=90210, mma_precision="tf32x3")
    clean_out, _ = _family_errors(**kw)
    assert clean_out < RTOL, f"M5's control does not pass (out {clean_out:.3e})"
    e_out, _ = _family_errors(q_y=q_bf16, **kw)
    assert e_out > RTOL, (
        f"a bf16 y cast under tf32x3 moved the output charge only to {e_out:.3e}: "
        "the exactness arm cannot see the epilogue cast it must forbid")


# --- the T-scaling arm (§2.B3): flat in T, or the family halts --------------


_T_ARM = (512, 2048, 8192)


@pytest.mark.parametrize("decay", [False, True])
def test_family_deviation_is_flat_in_T(decay):
    """The read-only/no-walk claim, measured. KILL CONDITION: a monotone T-trend.

    A resident-state walk grows as `sqrt(T)` (measured exponent +0.49, J1-B); a
    read-only rounding is flat. The fitted log-log slope over the three lengths
    must sit far below the walk's, and every point must clear the budget.
    """
    errs = []
    for T in _T_ARM:
        e_out, e_state = _family_errors(
            widths=(8, 8), BC=64, BT=16, d_v=64, B=2, T=T, H=3, norm="global",
            p_read=0.4, p_write=0.3, seed=90210, mma_precision="bf16", decay=decay,
            **_family_quantizers())
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


# --- support invariance (§2.B4): the cast is BELOW support selection --------


def test_support_is_invariant_under_amp_dtype_and_the_flush_is_inert():
    """Bitmaps/owners are byte-identical between fp32 and bf16 `amp_dtype`, on a
    fixture where the cast PROVABLY flushes amplitudes to zero.

    The flush count is asserted nonzero: bf16 shares fp32's exponent, so the only
    value-class change is a positive value below bf16's smallest subnormal
    (~9.2e-41) flushing to 0.0 -- without planting one, this test would pass
    vacuously on any fixture and the §2.B4 claim would be about nothing. The
    flushed leaf stays bitmap-LIVE (support is decided on the exact fp32 folded
    tensors, upstream of the cast) and contributes exactly zero (§12 rule 4), and
    the kernel run on the flushed values still matches the oracle in budget.
    """
    device = "cuda"
    widths, BC, BT, d_v, B, T, H = (8, 8), 64, 16, 64, 2, 37, 3
    gen = torch.Generator(device=device).manual_seed(2718)
    topology = _topology(widths, d_v)
    read_levels = [_simplex((B, T, H, w), 0.4, gen, device) for w in widths]
    write_levels = [_simplex((B, T, H, w), 0.3, gen, device) for w in widths]
    # Plant sub-bf16-subnormal amplitudes on live entries of the SPANNED level.
    # 1e-41 is nonzero in fp32 (subnormal) and flushes to 0.0 in bf16.
    tiny = read_levels[-1].clone()
    rows = tiny[0, :, 0, :]                                  # [T, width]
    live = rows > 0
    multi = live.sum(-1) >= 2                                # rows that keep O(1) mass
    assert bool(multi.any()), "the fixture has no multi-live rows to plant a flush in"
    first_live = torch.argmax(live.to(torch.int64), dim=-1)  # one live digit per row
    plant = torch.zeros_like(rows, dtype=torch.bool)
    plant[torch.arange(rows.shape[0], device=device), first_live] = True
    plant &= multi.unsqueeze(-1)
    rows = torch.where(plant, torch.full_like(rows, 1e-41), rows)
    # Renormalize so every row stays a simplex; the planted entries remain ~1e-41
    # (the rest of the row carries O(1) mass), far below bf16's smallest subnormal.
    tiny[0, :, 0, :] = rows / rows.sum(-1, keepdim=True)
    read_levels[-1] = tiny
    g_write = torch.rand(B, T, H, device=device, dtype=torch.float64, generator=gen) + 0.5
    v = torch.randn(B, T, H, d_v, device=device, dtype=torch.float64, generator=gen)

    def schedule_for(amp_dtype):
        plan = Plan(topology=topology, normalization="global", BT=BT, BC=BC,
                    warp_tile=(1, d_v // 32), mma_precision="bf16", amp_dtype=amp_dtype)
        return build_schedule(
            tuple(t.to(torch.float32) for t in read_levels),
            tuple(t.to(torch.float32) for t in write_levels),
            None, g_write.to(torch.float32), plan)

    s32 = schedule_for(torch.float32)
    s16 = schedule_for(torch.bfloat16)

    flushes = 0
    for a32, a16 in zip(list(s32.read_amps) + list(s32.write_amps),
                        list(s16.read_amps) + list(s16.write_amps)):
        # The two dtypes pad to different pitches (the pitch is cp.async-derived
        # from the itemsize); the level's real columns are the comparable region,
        # and the padding is structural zeros on both sides by construction.
        w = min(a32.shape[-1], a16.shape[-1])
        a32, a16 = a32[..., :w], a16[..., :w]
        f32_live = a32 != 0
        b16_live = a16.to(torch.float32) != 0
        # One direction only: rounding cannot manufacture a nonzero from a zero.
        assert bool((~f32_live & b16_live).sum() == 0), (
            "a bf16 amplitude is nonzero where the fp32 one is zero -- impossible by "
            "rounding; the stream builder is broken")
        flushes += int((f32_live & ~b16_live).sum())
    assert flushes > 0, (
        "the planted sub-subnormal amplitudes did not flush under bf16: the fixture no "
        "longer tests the flush direction and must be re-derived, not deleted")

    for name in ("read_tile_bits", "write_tile_bits", "read_token_bits",
                 "write_token_bits", "owners", "owner_costs"):
        t32, t16 = getattr(s32, name), getattr(s16, name)
        assert torch.equal(t32, t16), (
            f"schedule.{name} differs between amp_dtype fp32 and bf16: the cast has "
            "moved above support selection, which the ruling forbids")

    # The flushed configuration is INERT: the kernel runs on THIS planted fixture
    # (post-family, the spanned stream is stored bf16 and the planted entries are
    # real flushes; pre-family this is the fixture's exact control) against the
    # exact-fp64 oracle, in budget -- a bitmap-live leaf with a zero coefficient
    # contributes exactly zero, and 1e-41 of amplitude moves nothing at 1e-2.
    from rola.ops.consumer import _consumer_forward
    y_ref, state_ref = naive_rola(
        v, tuple(read_levels), tuple(write_levels), None, g_write, topology, None,
        "global", output_final_state=True)
    plan = Plan(topology=topology, normalization="global", BT=BT, BC=BC,
                warp_tile=(1, d_v // 32), mma_precision="bf16")
    y, state, _ = _consumer_forward(
        v.to(torch.float32),
        tuple(t.to(torch.float32) for t in read_levels),
        tuple(t.to(torch.float32) for t in write_levels),
        None, g_write.to(torch.float32), plan)
    e_out = _output_charge(y, y_ref)
    e_state = (float((state.double() - state_ref).abs().max())
               / max(1e-30, float(state_ref.abs().max())))
    assert e_out < BF16_RTOL and e_state < BF16_RTOL, (
        f"the planted-flush fixture is out of budget (out {e_out:.3e}, state "
        f"{e_state:.3e}): a flushed-to-zero live leaf is NOT inert")
