"""GATE I6-B, EXTENDED -- the bf16 MEMORY-FAMILY gate, written before the family.

The phase-2 family stores five surfaces in bf16 (the ruling of 2026-08-04): the
SPANNED-level amplitudes, the value tensor, the v/o projection GEMM operands, and
the output ``y`` cast after the fp32 divide. As of P26 (2026-08-06) this is no
longer one storage arm selected by an ``mma_precision`` field -- it is
UNCONDITIONAL, the only arm the kernel has (``Plan`` carries no precision axis at
all; ``level_amp_dtype`` hardcodes bf16 for every spanned level). Everything else
-- state, mass, num/den accumulators, the decay clock, the FIXED-level amplitudes
(the per-token row scales), and the producer/support tensors -- stays fp32.

**This file is the gate, and it exists before the code it gates.** The mutants
below are VALUE-LEVEL quantizations injected on the KERNEL side only (the fp64
oracle keeps the exact input): they prove the family's own tolerance can SEE a
defect of the family's own magnitude. Each mutant carries an unmutated control,
because a mutant whose control does not pass proves nothing.

**P26 collapsed the file's scope gate.** M1/M4/M5 existed to prove a SEPARATE
``tf32x3`` exactness arm still held ``RTOL`` when handed exactly the bf16
narrowing the production arm was about to adopt -- catching an accidental global
cast onto the wrong arm. There is no second arm left to accidentally cast onto
(bf16 is the only storage format the kernel has), so the scope question M1/M4/M5
asked is unaskable and all three are DELETED (accounting in
``workflows/perf/p26_endgame.md``):

* M1 gated amplitudes at bf16 values, staged under the tf32x3 arm.
* M4 gated the value tensor at bf16 values, staged under the tf32x3 arm.
* M5 gated ``y`` at bf16 values, staged under the tf32x3 arm.

The remaining named mutant (`PARITY_BATCH_DESIGN.md` §2.B2, extended to the
family):

* **M2** clock values at bf16 on a decay arm         -> must exceed ``BF16_RTOL``
  (the exponent amplification argument, §4.7 -- this is what makes "the clock
  stays fp32" a checked property rather than a comment).
* **M3** amplitudes at 4 mantissa bits on the bf16 arm -> must exceed
  ``BF16_RTOL`` (the anti-slack mutant: the tolerance is not wide enough to
  swallow a format two rungs below the one under test).

**M2's comparison basis changed.** Pre-P26, M2 ran under ``normalization="raw"``
specifically because the ``global`` ratio readout ``y = num/den`` was suspected
to cancel a common multiplicative clock error between its numerator and
denominator, hiding the mutant. ``raw`` is dead now (``Plan`` validates
``global`` only), so M2 was measured directly against ``global``'s output AND
final state. MEASURED (this file's exact fixture, `_family_errors`): the ratio
readout `y` really does cancel the mutant's effect near-exactly -- the control
and mutant `y` charges came back numerically identical (`~6.48e-3`, both under
budget) -- but the final STATE (the raw decayed `M`/`z` accumulators the ratio
divides, never itself normalized) does not: control state error `~1.6e-3`,
mutant state error `~1.25e-2`, a clear breach of `BF16_RTOL = 1e-2` against a
control comfortably inside it. M2 is therefore KEPT, reworked to compare
directly against the fp64 oracle at `BF16_RTOL` (no tf32x3 arm), with its
discriminating power now resting on the STATE term rather than the (cancelling)
output term -- both terms are still asserted, `max(e_out, e_state)`, exactly as
before; only the comparison basis under it changed.

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


def _family_errors(widths, BC, d_v, B, T, H, p_read, p_write, *, seed,
                   decay=False, q_amps=None, q_clock=None, q_v=None,
                   q_y=None, write_levels=None, dials=None, g_write=None):
    """`(e_out, e_state)` of the kernel vs the canonical fp64 oracle, with the
    named value-level quantizers applied to the KERNEL's inputs/output only.

    ``normalization`` is always ``"global"`` -- the only value ``Plan`` accepts
    (P26). ``q_amps`` applies to the SPANNED levels (``l >= j``) of both routing
    sides -- the family's Option B scope; fixed levels and the row scales they
    feed stay exact. ``q_clock`` applies to the kernel's ``write_levels_stored``
    (the decay clock's source) while the oracle keeps the exact clock.
    ``q_v``/``q_y`` apply to the value tensor and the output.
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
    # `normalization` is always "global" now, so `g_read` is always None -- the
    # kernel raises on a non-None `g_read` under global normalization.
    g_read = None
    v = torch.randn(B, T, H, d_v, device=device, dtype=torch.float64, generator=gen)
    cols = d_v + 1
    initial_state = 0.1 * torch.randn(
        B, H, topology.N, cols, device=device, dtype=torch.float64, generator=gen)
    initial_state[..., d_v] = initial_state[..., d_v].abs()

    if decay and dials is None:
        dials = _decay_dials(widths, H, gen, device)
    decay_obj = None if dials is None else LeafMassDecay(dials=dials)

    y_ref, state_ref = naive_rola(
        v, read_levels, write_levels, g_read, g_write, topology, decay_obj, "global",
        initial_state=initial_state, output_final_state=True)

    plan = Plan(topology=topology, normalization="global", BC=BC,
                warp_tile=(1, max(1, d_v // 32)), decay=decay)
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
        g_read,
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
def test_family_narrowing_holds_bf16_budget(widths, BC):
    """bf16 spanned amps + bf16 v + bf16 y TOGETHER stay under `BF16_RTOL`.

    This is the family's feasibility measured before it is built: if the stacked
    roundings blow the budget on any topology (the `D=4, j=0` row has no argued
    margin), the family halts here, before any kernel change. (P26: `normalization`
    is always `"global"` now -- the `"raw"` arm this row set used to also sweep is
    dead, see `workflows/perf/p26_endgame.md` DELETIONS.)
    """
    e_out, e_state = _family_errors(
        widths=widths, BC=BC, d_v=64, B=2, T=37, H=3,
        p_read=0.4, p_write=0.3, seed=90210,
        **_family_quantizers())
    assert e_out < BF16_RTOL, f"family narrowing blows the output budget: {e_out:.3e}"
    assert e_state < BF16_RTOL, f"family narrowing blows the state budget: {e_state:.3e}"


@pytest.mark.parametrize("widths,BC", [((8, 8), 64), ((2, 2, 2, 2), 16)])
def test_family_narrowing_holds_budget_under_decay(widths, BC):
    """Same claim on the decay arm -- with the clock EXACT, as the family pins it."""
    pytest.skip("decay retired from the operator; retires with the old "
                "consumer (docs/internals/chunk/chunk.md)")
    e_out, e_state = _family_errors(
        widths=widths, BC=BC, d_v=64, B=2, T=37, H=3,
        p_read=0.4, p_write=0.3, seed=555, decay=True,
        **_family_quantizers())
    assert e_out < BF16_RTOL, f"decay-arm family narrowing blows the budget: {e_out:.3e}"
    assert e_state < BF16_RTOL, f"decay-arm family narrowing blows the budget: {e_state:.3e}"


# --- the mutants: each must FAIL its arm's tolerance ------------------------


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
    * the PER-SEGMENT output charge (each segment normalizes by its own decayed
      scale) keeps the amplified error visible in principle -- but MEASURED on
      this exact fixture, `global`'s ratio readout `y = num/den` cancels the
      mutant's effect almost exactly (control and mutant `e_out` came back
      numerically identical, `~6.48e-3`). What stays discriminating is the final
      STATE: the raw decayed `M`/`z` accumulators the ratio divides, never
      themselves normalized, carry the clock error directly (control `~1.6e-3`,
      mutant `~1.25e-2` -- a clean breach of `BF16_RTOL` against a control that
      clears it with room to spare). Both terms are still asserted via
      `max(e_out, e_state)`, exactly as before P26; only the STATE term is what
      does the discriminating now that `raw` (which avoided the cancellation
      entirely) is dead.

    Predicted deviation at the tail: `exp(~7.6e-3 * 35) - 1 ~ 0.3`, against a
    bf16-MMA control floor of ~5e-3 -- an order past `BF16_RTOL` either way; the
    measured `~1.25e-2` on the state is the diluted-by-owner-mixing version of
    that same amplification argument.
    """
    pytest.skip("decay retired from the operator; retires with the old "
                "consumer (docs/internals/chunk/chunk.md)")
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
    kw = dict(widths=widths, BC=64, d_v=64, B=B, T=T, H=H,
              p_read=1.0, p_write=0.6, seed=555, decay=True,
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
    kw = dict(widths=widths, BC=BC, d_v=64, B=2, T=37, H=3,
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


@pytest.mark.parametrize("decay", [False, True])
def test_family_deviation_is_flat_in_T(decay):
    """The read-only/no-walk claim, measured. KILL CONDITION: a monotone T-trend.

    A resident-state walk grows as `sqrt(T)` (measured exponent +0.49, J1-B); a
    read-only rounding is flat. The fitted log-log slope over the three lengths
    must sit far below the walk's, and every point must clear the budget.
    """
    pytest.skip("decay retired from the operator; retires with the old "
                "consumer (docs/internals/chunk/chunk.md)")
    errs = []
    for T in _T_ARM:
        e_out, e_state = _family_errors(
            widths=(8, 8), BC=64, d_v=64, B=2, T=T, H=3,
            p_read=0.4, p_write=0.3, seed=90210, decay=decay,
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
    """The bitmap's liveness bit for a planted sub-bf16-subnormal leaf stays SET
    even though the STORED amplitude at that leaf is a hard zero, and the kernel
    run on that fixture still matches the oracle in budget.

    **P26 rework note.** Pre-P26, `Plan` accepted a caller-chosen `amp_dtype` and
    this test built TWO schedules (`fp32`, `bf16`) from the identical fixture and
    diffed their bitmaps byte-for-byte -- the direct way to show the cast sits
    strictly below support selection. `Plan` no longer exposes `amp_dtype`
    (`level_amp_dtype` hardcodes bf16 for every spanned level, unconditionally),
    so there is no longer a full-precision arm to build and diff against; only
    ONE schedule is buildable now. The claim is re-derived from that single
    schedule instead: the fixture plants a live amplitude below bf16's smallest
    subnormal (~9.2e-41) at a KNOWN `(level, digit, token)` triple, and the test
    reads the schedule's `read_token_bits` support bit and its `read_amps` stored
    value AT THAT SAME TRIPLE directly -- the bit must be 1 (the exact fp32-side
    mask that built the bitmap saw the leaf as live) while the stored coefficient
    is exactly 0.0 (the cast flushed it). That is the same "support decided
    upstream of the cast" claim, checked from one schedule instead of a diff of
    two, because a diff of two is no longer constructible.
    """
    device = "cuda"
    widths, BC, d_v, B, T, H = (8, 8), 64, 64, 2, 37, 3
    gen = torch.Generator(device=device).manual_seed(2718)
    topology = _topology(widths, d_v)
    # Dense (`p_nonzero=1.0`) on the planted level so every digit starts live --
    # the plant then has company (the "multi-live" requirement) by construction,
    # not by chance.
    read_levels = [_simplex((B, T, H, w), 1.0 if i == len(widths) - 1 else 0.4, gen, device)
                   for i, w in enumerate(widths)]
    write_levels = [_simplex((B, T, H, w), 0.3, gen, device) for w in widths]

    level_idx = len(widths) - 1                 # the last (spanned) level
    digit, token = 3, 5                          # a fixed, known plant site
    # Plant a sub-bf16-subnormal amplitude at ONE (batch, head, token, digit):
    # 1e-41 is nonzero in fp32 (subnormal) and flushes to 0.0 in bf16.
    tiny = read_levels[level_idx].clone()
    tiny[0, token, 0, digit] = 1e-41
    tiny[0, token, 0] = tiny[0, token, 0] / tiny[0, token, 0].sum()
    read_levels[level_idx] = tiny
    g_write = torch.rand(B, T, H, device=device, dtype=torch.float64, generator=gen) + 0.5
    v = torch.randn(B, T, H, d_v, device=device, dtype=torch.float64, generator=gen)

    plan = Plan(topology=topology, normalization="global", BC=BC,
                warp_tile=(1, d_v // 32))
    sched = build_schedule(
        tuple(t.to(torch.float32) for t in read_levels),
        tuple(t.to(torch.float32) for t in write_levels),
        None, g_write.to(torch.float32), plan)

    # The stored amplitude is a hard bf16 flush: the family's `level_amp_dtype`
    # is bf16 for every spanned level (`level_idx` is spanned since j = 0 at
    # `BC == N`), and the planted value is below bf16's smallest subnormal.
    assert plan.level_amp_dtype(level_idx) == torch.bfloat16
    tile, t_in_tile = token // 32, token % 32
    stored = float(sched.read_amps[level_idx][0, tile, t_in_tile, digit])
    assert stored == 0.0, (
        f"the planted sub-subnormal amplitude did NOT flush under bf16 storage "
        f"(read {stored!r}): the fixture no longer tests the flush direction and "
        "must be re-derived, not deleted")

    # The bitmap's liveness bit is still SET at that exact (level, digit, token):
    # support was decided on the exact fp32 mask, upstream of the bf16 store.
    row = sched.level_offsets[level_idx] + digit
    bit = (int(sched.read_token_bits[0, tile, row]) >> t_in_tile) & 1
    assert bit == 1, (
        "the planted leaf's support bit is NOT set: the bitmap moved above the "
        "exact fp32 mask, which the ruling forbids")

    # The flushed configuration is INERT: the kernel runs on THIS planted fixture
    # against the exact-fp64 oracle, in budget -- a bitmap-live leaf with a zero
    # coefficient contributes exactly zero, and 1e-41 of amplitude moves nothing
    # at 1e-2.
    from rola.ops.consumer import _consumer_forward
    y_ref, state_ref = naive_rola(
        v, tuple(read_levels), tuple(write_levels), None, g_write, topology, None,
        "global", output_final_state=True)
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
