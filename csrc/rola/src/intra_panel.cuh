#pragma once

// RoLA V3 consumer -- the `panel` intra lowering under `LeafMassDecay`
// (spec §7.4.1). THIS IS THE DECAY ARM'S LOWERING: there is no `intra_lowering`
// template axis and no alternative path, so the factors below are applied on every
// decay tile the kernel executes.
//
// A mixed tile under `LeafMassDecay` cannot use the "fixed entry state plus causal
// Gram" split, because the state evolves inside the tile. This file pre-scales the
// entry state into a per-leaf reference frame, folds cumulative survival into the
// read and write amplitude panels as two exponentials, and post-scales out -- which
// turns a mixed tile into the `decay = None` tile VERBATIM. It therefore deletes a
// path rather than adding one: no new interaction with GLOBAL, BT, BC,
// `mma_precision` or §6.4's `P`.
//
// The two exponentials are where `GLA_MASS_DECAY_SPEC.md` §12 rules 1-3 go live.
// Because their clamp is a FINITENESS guarantee and not an accuracy one, the decay
// arm carries a PLAN-TIME ADMISSION BOUND (`PANEL_MAX_EXPONENT = 40`): routing that
// would exceed it is REFUSED at plan time rather than clamped into silence.
//
// The factorization, the three rules, the measured cliff at the clamp and the
// admission bound: docs/internals/intra_panel.md

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

}  // namespace rola
