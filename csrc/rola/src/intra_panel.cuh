#pragma once

// RoLA V3 consumer -- the `panel` intra lowering (spec §7.4.1; the DENSE-LIMIT
// lowering under `LeafMassDecay`), and as of item 6b (2026-08-01) the DECAY ARM'S
// ONLY lowering: the `intra_lowering` axis and its `realized_leaf` token-sequential
// alternative went with the old consumer, so the consumer includes this header
// unconditionally. Its sibling `intra_gram.cuh` -- the same rung on the
// `decay = None` arm -- went with it too.
//
// WHAT IT REPLACES AND WHY. Under LeafMassDecay a MIXED tile cannot use the
// "carry against a fixed tile-entry state plus a causal Gram" split, because the
// state evolves inside the tile. The kernel's standing answer is the
// TOKEN-SEQUENTIAL WALK (`realized_leaf` on this arm): walk the tile's tokens in
// order, `M *= keep; M += W*v; y = R*M`. That is exact and `O(BT * BC)` per tile
// -- but every one of those `BT * BC` FMAs is in the thread->channel map, the
// walk carries a loop-carried dependence over `t`, and NOTHING in it reaches the
// tensor cores: neither the carry (which the `decay = None` arm does with the §5
// MMA readout) nor the intra term. Spec §7.4.1: `LeafMassDecay -> decay-aware
// panels: survival is a per-leaf scalar, so fold cumulative survival into the R/W
// factors`.
//
// THE FACTORIZATION (exact in exact arithmetic; the fp32 statement is below).
// All indices are tile-local: `i` is the owner's leaf, `t`/`j` are tokens of one
// tile, and the intra term needs `j <= t` (§5's inclusive read-after-write).
//
//   keep_i     = 1 - rate_i                    log_keep[i] = log1pf(-rate_i) <= 0
//   P[t,i]     = sum_{u <= t} c[u,i]           (INCLUSIVE prefix of the clock)
//   survival(j -> t) = prod_{u=j+1..t} keep[u,i] = keep_i^{P[t,i] - P[j,i]}
//
// The exponent splits across the two operands of a Gram because it is a SUM in
// `t` and `j` separately -- which is the whole content of "survival is a per-leaf
// scalar". With a per-leaf reference exponent `P_ref[i]` (see rule 1 below):
//
//   S0[i]    = exp(P_ref[i] * log_keep[i])                          (<= 1)
//   R~[t,i]  = R[t,i] * exp( (P[t,i] - P_ref[i]) * log_keep[i])
//   W~[j,i]  = W[j,i] * exp(-(P[j,i] - P_ref[i]) * log_keep[i])
//   R~[t,i] * W~[j,i] = R[t,i] * W[j,i] * keep_i^{P[t,i] - P[j,i]}
//
// so one mixed tile becomes, verbatim, the `decay = None` tile:
//
//   carry[t]  = sum_i R~[t,i] * (S0[i] * M0[i])            -- the §5 MMA readout
//   intra     = causal_mask(R~ @ W~^T) @ (v * scale_write) -- the §5 causal Gram
//   M_end[i]  = S0[i] * ( S0[i]*M0[i] + sum_j W~[j,i]*gw[j]*v[j] )
//
// THE LOWERING THEREFORE DELETES A PATH RATHER THAN ADDING ONE. `panel` is not a
// third tile algorithm: it is the `decay = None` tile algorithm, reached under
// LeafMassDecay by pre-scaling the entry state into a per-leaf reference frame,
// folding the two exponentials into the two panels, and post-scaling out. That is
// why it composes with GLOBAL, COMPACT, BT, BC, `mma_precision` and §6.4's `P`
// with no new interaction: every one of those axes is already exercised by the
// path it joins. The one genuinely new object is the pair of exponentials, and
// they are where `GLA_MASS_DECAY_SPEC.md` §12 rules 1 and 3 go live (spec §7.4's
// per-lowering audit; `readout_mma.cuh` records that rule 1 becomes live with
// exactly this construct and nowhere earlier).
//
// ---------------------------------------------------------------------------
// §12 RULE 1 -- mid-tile clock normalization. KEPT VERBATIM, and it is what
// `P_ref` is.
//
// `P_ref[i] := P[BT-1, i] / 2` is the MINIMAX reference: the largest exponent
// magnitude either panel forms is `max_t |P[t,i] - P_ref[i]| * |log_keep[i]|`,
// and over a non-decreasing `P` that is minimized, at exactly half the tile's
// total, by the midpoint of the range. No other single per-leaf reference does
// better, and a per-`(t,j)` reference does not exist -- a Gram contracts `i`, so
// the reference must be constant along the contracted axis. Rule 1's documented
// "square-root mitigation" is therefore not an approximation here, it is the
// optimal choice of a free parameter, and both scales come out equal:
// `P[BT-1] - P_ref == P_ref`, so the tile's entry scale and exit scale are the
// SAME `S0`.
//
// §12 RULE 2 -- no epsilon floor. There is none, and the asymmetry of the clamp
// below is what keeps it that way: `exp` UNDERFLOWING to zero is the correct
// limit of `keep^{large}` and is left alone. Flooring it would alter the
// retention contract, which rule 2 forbids.
//
// §12 RULE 3 -- finite clamp. `W~[j,i] = W[j,i] * keep_i^{-(P[j,i] - P_ref[i])}`
// is a DIVISION by cumulative survival and grows without bound as `P[j,i] -> 0`
// (late `P_ref`, early `j`); §7.4's audit table names this as rule 3's reason to
// exist. But it is NOT the only quantity that grows: `R~[t,i] = R[t,i] *
// keep_i^{+(P[t,i] - P_ref[i])}` grows equally without bound as `P[t,i] -> 0`
// (early `t`), by the SAME magnitude -- both panels' exponents range over
// `[-E, +E]` with `E := max_i P_ref[i] * |log_keep[i]|`, and `panel-review.md` §1
// confirms `max |read exp| == max |write exp| == E` numerically. `panel_read_factor`
// carries the identical `fminf(., kPanelMaxExponent)` clamp for exactly this
// reason; it is not a defensive copy of the write side's guard, it is load-bearing
// on its own. `kPanelMaxExponent` clamps the exponent ARGUMENT, not the result, at
// `88` -- `expf(88) = 1.65e38`, finite in fp32 AND in bf16 (bf16 shares fp32's
// 8-bit exponent, so `clamp_bf16_finite` in the readout is a no-op on an
// already-clamped panel entry). The clamp is one-sided by construction (see rule 2).
//
// **THE CLAMP IS A FINITENESS GUARANTEE, NOT AN ACCURACY ONE, AND THAT IS WHY
// THE LOWERING HAS A PLAN-TIME ADMISSION BOUND.** Rules 1 and 3 are "a pair, not
// alternatives" (spec §7.4) and the pair is still not sufficient at unbounded
// dials. Past `E := max_i P_ref[i] * |log_keep[i]| ~ 88` two things break, in
// either order depending on the cell: `W~` clamps DOWN while `R~` underflows
// toward zero, so their product -- `O(1)` at `t ~ j` -- is computed as `~0`; and
// the deposit sums `BT` terms of size `exp(88) = 1.65e38`, which overflows fp32
// to `inf` and then meets the exit scale `S0 ~ 0` as `inf * 0 = NaN`.
//
// MEASURED (panel.md §2, RTX 3080 Ti, `widths=(4,4) BC=16 BT=32 T=64`, one-hot
// write clock so the bound is EXACT, dials swept to move `E` alone):
//
//     E      20     40     60     80     84       88       120      200
//     panel  1.4e-7 1.4e-7 1.2e-7 1.7e-7 1.4e-7   NaN      NaN      NaN
//     walk   6.1e-8 6.3e-8 6.0e-8 7.2e-8 6.5e-8   6.4e-8   6.8e-8   5.9e-8
//
// (relative to the fp64 oracle). The transition is AT the clamp and it is sharp:
// exact to `E = 84`, gone at `E = 88`. `PANEL_MAX_EXPONENT = 40` is therefore a
// 2.1x margin on a measured cliff, not a guess. The walk is flat across the whole
// sweep, which is spec §7.4's per-lowering verdict made visible: survival as a
// running product has no range to lose.
//
// So the domain is CHECKED rather than documented.
// `routing/schedule_v3.panel_exponent_bound()` computes a cheap
// EXACT UPPER BOUND on `max_{i,tile} P_ref[i] * |log_keep[i]|` from the dials and
// the clock stream (no `N`-shaped materialization: `prod_l c_l[t, d_l(i)] <=
// prod_l max_d c_l[t,d]` and `g_i <= -log1p(-prod_l max_d delta_l[d])`), and the
// driver REFUSES a `panel` launch above `PANEL_MAX_EXPONENT = 40`. Inside that
// domain the clamp provably never fires; outside it the launch is rejected
// loudly. The clamp remains, because "provably unreachable" is a property of the
// admission check and a kernel should not be one edit away from producing `inf`.
//
// WHY THERE IS NO `-inf * 0` NaN CORNER HERE. Architecture §6.2.1's design note
// describes a LOG-SPACE variant of this lowering (log the per-level factors,
// select digits with a binary selector MMA, exponentiate in the epilogue) whose
// one numeric corner is `-inf * 0`. That variant is NOT what this file
// implements -- the decision, with its arithmetic, is recorded in
// `workflows/20260730-wiring/panel.md` §1. In the direct staged-factor form
// implemented here the amplitude is never logged: a structurally zero factor
// stays an exact `0.0f`, is multiplied by a FINITE (clamped) exponential, and
// yields an exact `0.0f`. Sparse entmax routing therefore reaches the oracle's
// exact zero bit for bit, and §12 rule 4's branchless write-side annihilation
// (`a == 0.0f`) still fires on a structurally dead route, which it would NOT do
// if a `-inf` had been clamped to `-88` and exponentiated to `6e-39`.
//
// ---------------------------------------------------------------------------
// MEASURED STATUS -- READ BEFORE SELECTING THIS LOWERING (2026-07-30, the
// increment that landed it). **`panel` is SLOWER than the token-sequential walk
// at every cell benchmarked -- 16 of 16, by +12% to +106%.** It is implemented,
// oracle-gated and explicitly reachable, and it is NOT plan-selected;
// `select_intra_lowering`'s structural rule keeps `realized_leaf` on this arm.
// That is the same position `gram` held while it measured slower, and it comes
// off the same way: a measured cell where the lowering wins.
//
// RTX 3080 Ti, `B=1 H=8 L=2048 d_v=64 branches=[8,16]`, real V3 producer routing,
// 30 reps, median, `compact=false` (the plan-selected instantiation everywhere).
// Consumer ms, `realized_leaf` (the walk) against `panel`:
//
//   router     BC  BT norm     walk    panel   delta
//   softmax    16  16 raw     2.3122   4.1585  +79.9%
//   softmax    16  32 raw     2.2280   4.5946 +106.2%
//   softmax    16  16 global  2.9266   4.2486  +45.2%
//   softmax    16  32 global  2.9338   4.5240  +54.2%
//   softmax    64  16 raw     5.9668   7.2294  +21.2%
//   softmax    64  32 raw     5.9523   7.6420  +28.4%
//   softmax    64  16 global  6.5085   7.3073  +12.3%
//   softmax    64  32 global  6.4184   7.8223  +21.9%
//   entmax g8  (8 further cells, same direction, +14.5% to +82.6%)
//
// THE CAUSE IS NAMED, not left as "it is slower", because it is a property of
// what is still MISSING rather than of the lowering:
//
//   1. **A mixed tile emits TWO atomics per (token, channel) on this path and the
//      walk emits one.** The walk fuses carry+intra into one accumulator per
//      token before its single `atomicAdd`; `panel` takes the `decay = None`
//      path, whose readout and intra term land as separate split-K atomics (see
//      the "still scalar" note at the intra site in `consumer.cu`). Under
//      LeafMassDecay with any write density most tiles are MIXED, so this
//      doubles the epilogue traffic on almost every tile. Fusing the intra term
//      into the readout epilogue is the standing follow-up for BOTH lowerings and
//      it is what has to land before `panel` is re-measured.
//   2. `panel` pays TWO `expf` panels (`2 * BT * BC`) where the walk pays one,
//      plus the clock PREFIX on top of the clock product the walk also computes.
//      MUFU-bound work, strictly added.
//   3. What it saves is real but smaller at these shapes: the carry moves to the
//      tensor cores, the intra drops from `BT * BC` per THREAD to
//      `BT(BT+1)/2 * KI / DV`, and the serial `t` dependence goes.
//
// The `BC = 16` rows are the largest deltas and the least trustworthy in
// magnitude (a byte-identical kernel was measured moving 12-23% between builds at
// `BC = 16`); the DIRECTION is what carries here, and it is
// unanimous across 16 cells, two routers, two norms and two `BT`.
//
// COST, per mixed tile, against the walk it replaces (the numbers are counts, not
// a measurement; the measured A/B is above and in `panel.md` §4):
//
//   walk    : BT*BC decay FMAs + BT*BC deposit FMAs + BT*BC read FMAs, ALL per
//             thread (the thread owns a channel and must touch every leaf), plus
//             BT*BC `expf` for the survival panel, plus a serial dependence on t.
//   panel   : BT*BC deposit FMAs per thread (unchanged), the carry on the tensor
//             cores, the intra as BT(BT+1)/2 * KI Gram entries spread over DV
//             threads, and 2*BT*BC `expf` for the two panels (against the walk's
//             BT*BC -- the survival exponential is paid either way, once per
//             (token, leaf); `panel` pays it twice because it forms two panels).
//
// So the trade is: one extra `expf` panel and a `BT(BT+1)/2 * KI / DV` Gram, for
// the removal of `2 * BT * BC` per-thread FMAs and of the serial `t` chain.

#include "consumer.cuh"

namespace rola {

//: §12 rule 3's clamp point, as an EXPONENT bound rather than a value bound.
//: `expf(88) = 1.65e38 < FLT_MAX` and `< kBf16MaxFinite`, so a clamped panel
//: entry is finite in every format the readout narrows to. One-sided: see rule 2.
constexpr float kPanelMaxExponent = 88.0f;

//: `keep_i^{+(P[t,i] - P_ref[i])}`, the READ side's survival factor. Amplifies to
//: `e^{+E}` at `P[t,i] = 0` (early `t`); the clamp below is load-bearing here too
//: (rule 3 is symmetric across both panels -- see the comment above `kPanelMaxExponent`).
__device__ __forceinline__ float panel_read_factor(float P, float p_ref, float log_keep) {
  return expf(fminf((P - p_ref) * log_keep, kPanelMaxExponent));
}

//: `keep_i^{-(P[j,i] - P_ref[i])}`, the WRITE side's. This is the division by
//: cumulative survival that §7.4's audit table names as rule 3's motivating case,
//: but the read side above amplifies by the identical amount at the opposite end
//: of the tile and needs the same clamp.
__device__ __forceinline__ float panel_write_factor(float P, float p_ref, float log_keep) {
  return expf(fminf((p_ref - P) * log_keep, kPanelMaxExponent));
}

//: Per-leaf INCLUSIVE prefix of the clock over one tile, plus rule 1's reference
//: exponent and the entry/exit scale it implies.
//:
//: `clk[t*BC + i] = P[t,i]`, `p_ref[i] = P[BT-1,i]/2`, `s0[i] = exp(p_ref *
//: log_keep)`. One thread per leaf, serial over the tile's `BT` tokens, because
//: the prefix IS a serial dependence in `t` and `BT <= 32`; parallelizing it
//: would cost more barriers than it saves steps.
//:
//: The clock product runs over ALL levels, unlike the read/write panels: a fixed
//: level's factor cannot hoist into the §3 step 3 row scale, because the clock is
//: an EXPONENT. The deleted `gram` lowering's level loop started at
//: `first_spanned_level` for that same reason; this one does not.
//:
//: Scatter-fill identity, exactly as the walk's survival panel: an inert leaf
//: (dropped by §3 step 2's gather) contributes `0` to the prefix, and that is not
//: a convention -- `build_clock_stream` enforces `stored != 0 => live != 0` per
//: (level, digit), so a leaf the gather dropped has `c == 0` at every token of
//: this tile and its true prefix IS zero. `keep^0 == 1` then makes every factor
//: this file computes exactly `1`, which is §2's write-conditioned quiescence.
template <int BT, int BC, int DV, bool COMPACT>
__device__ __forceinline__ void panel_clock_prefix(
    const ConsumerParams& p, const int* base, int64_t tile_row, int tid,
    const int* __restrict__ c_slot, const float* __restrict__ log_keep, float* __restrict__ clk,
    float* __restrict__ p_ref, float* __restrict__ s0) {
  for (int i = tid; i < BC; i += DV) {
    const bool live = (!COMPACT || c_slot[i] >= 0);
    float acc = 0.0f;
    for (int t = 0; t < BT; ++t) {
      float cl = 0.0f;
      if (live) {
        cl = 1.0f;
        for (int l = 0; l < p.D; ++l) {
          cl *= p.amp_clock[l]
                           [(tile_row + t) * p.level_pitch_clock[l] + leaf_digit(p, l, base[l], i)];
        }
      }
      // fp32 accumulation is a requirement, not a default (spec §7.4's V3-specific
      // rule): this prefix multiplies `log_keep` and is then exponentiated, so a
      // lost low bit here is an exponent error, not a mantissa one.
      acc += cl;
      clk[t * BC + i] = acc;
    }
    const float pr = 0.5f * acc;  // §12 rule 1: the minimax mid-tile reference.
    p_ref[i] = pr;
    // `pr >= 0` and `log_keep <= 0`, so `s0 <= 1` and no clamp is reachable on
    // this one. Underflow to zero is the correct limit (a leaf whose tile-total
    // survival is below fp32 has genuinely lost its entry state) and is rule 2's
    // sanctioned underflow, not a floor.
    s0[i] = expf(pr * log_keep[i]);
  }
}

}  // namespace rola
