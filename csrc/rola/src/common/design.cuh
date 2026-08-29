#pragma once

// LAYER 2a -- THE DERIVED SELECTIONS: what the design LOOKS LIKE on this
// architecture.  Every entry is COMPUTED from `arch_caps.cuh`'s facts, never
// looked up per architecture; the kernel reads THIS layer for its design and
// calls `ops.cuh` to execute it.  What is derived (state residency, staging
// depth, the CTA target), what deliberately stays a measured free axis, and
// the announce-the-gap obligation live in docs/internals/common/design.md.

#include "facts/arch_caps.cuh"

namespace denseref {
namespace design {

// ------------------------------------------------ the compile-time announcement
//: A compile-time diagnostic that does NOT fail the build. Instantiating
//: `Announce<true>` calls a deprecated constructor, which every supported
//: toolchain reports with the message attached.
// see docs/internals/common/design.md#announce
template <bool Suboptimal>
struct Announce {
  __host__ __device__ constexpr Announce() {}
};
template <>
struct Announce<true> {
  [[deprecated(
      "KNOWINGLY SUBOPTIMAL SELECTION: this architecture's capability row "
      "offers a better implementation of a baseline operation than the one "
      "this baseline builds. The build is correct and the gap is stated; see "
      "the announcement site in ops.cuh for which operation.")]] __host__
  __device__ constexpr Announce() {}
};

// --------------------------------------------------------- STATE RESIDENCY
//: WHERE THE STATE LIVES. The algorithm's requirement is not an instruction: it
//: is that THE FOLD'S ACCUMULATOR IS REUSABLE AS THE READOUT'S OPERAND WITHOUT
//: A MEMORY ROUND TRIP.
// see docs/internals/common/design.md#note-l36
enum class StateResidency { REGISTERS, TMEM, SHARED };

constexpr unsigned residency_bit(StateResidency r) { return 1u << (unsigned)r; }

//: The residencies that MEET THE REQUIREMENT on this architecture, as a mask.
//: SHARED never appears: a state that has to go through shared memory HAS taken
//: the round trip, which is a different algorithm.
constexpr unsigned satisfiable_residencies(const arch::Caps& c) {
  return (c.accum_in_tmem && c.mma_a_from_tmem ? residency_bit(StateResidency::TMEM) : 0u)
         | (!c.accum_in_tmem && c.mma_a_from_regs ? residency_bit(StateResidency::REGISTERS) : 0u);
}

//: PREFERENCE ORDER, most preferred first.  TMEM is preferred where it exists
//: because the accumulator is already there and the register file stays free.
constexpr StateResidency kResidencyOrder[2] = {StateResidency::TMEM, StateResidency::REGISTERS};

//: The best residency that BOTH meets the requirement and has an implementation.
//: Returns SHARED to mean "none", which is the value that can never be selected.
constexpr StateResidency select_residency(unsigned satisfiable, unsigned built) {
  return (satisfiable & built & residency_bit(kResidencyOrder[0]))   ? kResidencyOrder[0]
         : (satisfiable & built & residency_bit(kResidencyOrder[1])) ? kResidencyOrder[1]
                                                                     : StateResidency::SHARED;
}

//: True when a residency BETTER than the selected one meets the requirement on
//: this part and simply is not built.  This is the announcement's condition and
//: not an error: the built path is correct, it is merely not the best available.
constexpr bool residency_suboptimal(unsigned satisfiable, unsigned built) {
  return (satisfiable & residency_bit(kResidencyOrder[0]))
         && select_residency(satisfiable, built) == kResidencyOrder[1];
}

constexpr unsigned kSatisfiableResidencies = satisfiable_residencies(arch::kCaps);

// The requirement itself, which is structural.  A part where no residency meets
// it cannot run this design at all, and that is an UNSUPPORTED architecture --
// distinct from one whose satisfying residency merely has no implementation
// yet, which `ops.cuh` reports as NOT BUILT.
static_assert(kSatisfiableResidencies != 0u,
              "UNSUPPORTED ARCHITECTURE: on this part the MMA's accumulator "
              "cannot be reused as an operand without a memory round trip, in "
              "any storage class. The fold accumulator IS the readout's "
              "operand; a design that stores and reloads it is a different "
              "algorithm, not a slower arm of this one.");

//: THE RESIDENCIES THIS BASELINE BUILDS.
// see docs/internals/common/design.md#kbuiltresidencies
constexpr unsigned kBuiltResidencies = residency_bit(StateResidency::REGISTERS);

constexpr StateResidency kStateResidency =
    select_residency(kSatisfiableResidencies, kBuiltResidencies);

static_assert(kStateResidency != StateResidency::SHARED,
              "NOT BUILT FOR THIS ARCHITECTURE: the residency this part meets "
              "the accumulator-as-operand requirement in has no implementation "
              "here (a `tcgen05` accumulator needs the TMEM path, which is not "
              "written). Every structural clause holds; this is a capability "
              "gap, not a refused structure.");

//: Announced, not refused: the built path is correct on such a part, it is
//: merely not the best available there.
constexpr bool kResidencySuboptimal =
    residency_suboptimal(kSatisfiableResidencies, kBuiltResidencies);
inline constexpr Announce<kResidencySuboptimal> kResidencyAnnouncement{};

// ------------------------------------------------- WARP SPECIALIZATION
//: WHETHER A UNIT MAY BE DEDICATED TO PRODUCTION. Derived, and the derivation
//: is about REGISTERS rather than about warps.
// see docs/internals/common/design.md#kspecializationpays
constexpr bool specialization_pays(const arch::Caps& c) { return c.dyn_reg_realloc; }

constexpr bool kSpecializationPays = specialization_pays(arch::kCaps);

//: NOT BUILT, and announced rather than refused.
// see docs/internals/common/design.md#kbuiltspecialization
constexpr bool kBuiltSpecialization = false;
inline constexpr Announce<kSpecializationPays && !kBuiltSpecialization>
    kSpecializationAnnouncement{};

// ------------------------------------------------------------------ STAGES
//: THE RING'S DEPTH, DERIVED. One clause: TAKE THE SHALLOWEST RING THAT REACHES
//: THIS ARM'S MAXIMUM RESIDENCY.
// see docs/internals/common/design.md#kstagesmin
constexpr int kStagesMin = 2;  // the wait/issue order needs two groups
constexpr int kStagesMax = 8;  // the search bound, not a design constant

//: The largest per-lane register count is a table quantity; this is its inverse
//: -- the most CTAs whose budget still holds a live set of `reg_floor`.
constexpr int ctas_by_reg_floor(const arch::Caps& c, int threads, int reg_floor, int n) {
  return n <= 1 ? 1
                : (arch::reg_budget(c, threads, n) >= reg_floor
                       ? n
                       : ctas_by_reg_floor(c, threads, reg_floor, n - 1));
}

//: The residency ceiling that is NOT a function of this arm's shared memory:
//: registers, threads and the hard CTA cap.
constexpr int ceiling_without_smem(const arch::Caps& c, int threads, int reg_floor) {
  return ctas_by_reg_floor(c, threads, reg_floor,
                           arch::ctas_by_threads(c, threads) < c.max_ctas_per_sm
                               ? arch::ctas_by_threads(c, threads)
                               : c.max_ctas_per_sm);
}

//: THE SECOND LAUNCH BOUND: the CTA count this arm can actually reach.  Shared
//: memory, registers, threads and the CTA cap, whichever binds first.
constexpr int cta_target(const arch::Caps& c, int smem_bytes, int threads, int reg_floor) {
  return arch::ctas_by_smem(c, smem_bytes) < ceiling_without_smem(c, threads, reg_floor)
             ? arch::ctas_by_smem(c, smem_bytes)
             : ceiling_without_smem(c, threads, reg_floor);
}

//: A ring of `s` slots on top of `fixed` bytes of unringed buffers.
constexpr int ring_bytes(int slot_bytes, int fixed_bytes, int s) {
  return s * slot_bytes + fixed_bytes;
}

constexpr bool stages_fit(const arch::Caps& c, int slot_bytes, int fixed_bytes, int s) {
  return ring_bytes(slot_bytes, fixed_bytes, s) <= c.smem_per_cta_max;
}

//: The residency reachable at ANY depth. The ring can only shrink it, so this
//: is the shallowest LEGAL ring's residency; an arm whose shallowest ring does
//: not fit at all is refused by `smem_fits` at its own site rather than ...
// see docs/internals/common/design.md#note-l217
constexpr int best_residency(const arch::Caps& c, int slot_bytes, int fixed_bytes, int threads,
                             int reg_floor, int s) {
  return s >= kStagesMax || stages_fit(c, slot_bytes, fixed_bytes, s)
             ? cta_target(c, ring_bytes(slot_bytes, fixed_bytes, s), threads, reg_floor)
             : best_residency(c, slot_bytes, fixed_bytes, threads, reg_floor, s + 1);
}

//: The SHALLOWEST ring that reaches `want` CTAs, searched upward from the
//: minimum: the point past which shared memory buys nothing and still costs L1.
constexpr int stages_by_residency(const arch::Caps& c, int slot_bytes, int fixed_bytes, int threads,
                                  int reg_floor, int want, int s) {
  return s >= kStagesMax
             ? s
             : (stages_fit(c, slot_bytes, fixed_bytes, s)
                        && cta_target(c, ring_bytes(slot_bytes, fixed_bytes, s), threads, reg_floor)
                               >= want
                    ? s
                    : stages_by_residency(c, slot_bytes, fixed_bytes, threads, reg_floor, want,
                                          s + 1));
}

constexpr int stages_for(const arch::Caps& c, int slot_bytes, int fixed_bytes, int threads,
                         int reg_floor) {
  return stages_by_residency(
      c, slot_bytes, fixed_bytes, threads, reg_floor,
      best_residency(c, slot_bytes, fixed_bytes, threads, reg_floor, kStagesMin), kStagesMin);
}

//: The compiling target's answers, so a kernel writes a geometry and never a
//: capability row.
constexpr int stages_for(int slot_bytes, int fixed_bytes, int threads, int reg_floor) {
  return stages_for(arch::kCaps, slot_bytes, fixed_bytes, threads, reg_floor);
}
constexpr int cta_target(int smem_bytes, int threads, int reg_floor) {
  return cta_target(arch::kCaps, smem_bytes, threads, reg_floor);
}

//: THE ONE HARD RESOURCE REFUSAL, from the table rather than from a literal: an
//: arm either fits this architecture's per-CTA shared-memory maximum or it is
//: refused.  No arm is silently shrunk to fit.
constexpr bool smem_fits(size_t bytes) { return bytes <= (size_t)arch::kCaps.smem_per_cta_max; }

//: The digest of the capability row THE DEVICE CODE WAS COMPILED AGAINST,
//: carried here so a kernel's build stamp names no layer-1 symbol.
constexpr int kCapsDigest = arch::caps_digest(arch::kCaps);

}  // namespace design
}  // namespace denseref
