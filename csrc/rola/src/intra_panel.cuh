#pragma once

// RoLA V3 consumer -- the `panel` intra lowering: the DENSE-LIMIT lowering under
// `LeafMassDecay` (spec §7.4.1). Since item 6b (2026-08-01) the consumer includes
// this header unconditionally; the `intra_lowering` axis and its `realized_leaf`
// token-sequential alternative went with the old consumer, as did the
// `decay = None` sibling `intra_gram.cuh`.
//
// A mixed tile under `LeafMassDecay` cannot use the "fixed entry state plus causal
// Gram" split, because the state evolves inside the tile. This file pre-scales the
// entry state into a per-leaf reference frame, folds cumulative survival into the
// read and write amplitude panels as two exponentials, and post-scales out -- which
// turns a mixed tile into the `decay = None` tile VERBATIM. It therefore deletes a
// path rather than adding one: no new interaction with GLOBAL, COMPACT, BT, BC,
// `mma_precision` or §6.4's `P`.
//
// The two exponentials are where `GLA_MASS_DECAY_SPEC.md` §12 rules 1-3 go live.
// The lowering has a PLAN-TIME ADMISSION BOUND (`PANEL_MAX_EXPONENT = 40`) because
// its clamp is a finiteness guarantee, not an accuracy one, and it is IMPLEMENTED
// BUT NOT PLAN-SELECTED: measured slower than the token-sequential walk at 16 of
// 16 cells.
//
// The factorization, the three rules, the measured cliff at the clamp, the
// admission bound and the A/B against the walk: docs/internals/intra_panel.md

#include "consumer.cuh"

namespace rola {

//: §12 rule 3's clamp point, as an EXPONENT bound rather than a value bound.
//: `expf(88) = 1.65e38` is finite in every format the readout narrows to.
//: One-sided by construction: docs/internals/intra_panel.md#rule-3
constexpr float kPanelMaxExponent = 88.0f;

//: `keep_i^{+(P[t,i] - P_ref[i])}`, the READ side's survival factor. Its clamp is
//: load-bearing, not a defensive copy of the write side's: rule 3 is symmetric
//: across both panels -- docs/internals/intra_panel.md#rule-3
__device__ __forceinline__ float panel_read_factor(float P, float p_ref, float log_keep) {
  return expf(fminf((P - p_ref) * log_keep, kPanelMaxExponent));
}

//: `keep_i^{-(P[j,i] - P_ref[i])}`, the WRITE side's -- the division by cumulative
//: survival that §7.4's audit table names as rule 3's motivating case.
__device__ __forceinline__ float panel_write_factor(float P, float p_ref, float log_keep) {
  return expf(fminf((p_ref - P) * log_keep, kPanelMaxExponent));
}

//: Per-leaf INCLUSIVE prefix of the clock over one tile, plus rule 1's reference
//: exponent and the entry/exit scale it implies: `clk[t*BC + i] = P[t,i]`,
//: `p_ref[i] = P[BT-1,i]/2`, `s0[i] = exp(p_ref * log_keep)`. One thread per leaf,
//: serial over the tile's `BT` tokens. The clock product runs over ALL levels,
//: unlike the read/write panels.
//: Serialization, the all-levels product and the scatter-fill identity that makes
//: an inert leaf's factors exactly 1: docs/internals/intra_panel.md#clock-prefix
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
      //: HAZARD fp32-prefix -- docs/internals/intra_panel.md#fp32-prefix
      acc += cl;
      clk[t * BC + i] = acc;
    }
    const float pr = 0.5f * acc;  // §12 rule 1: the minimax mid-tile reference.
    p_ref[i] = pr;
    s0[i] = expf(pr * log_keep[i]);
  }
}

}  // namespace rola
