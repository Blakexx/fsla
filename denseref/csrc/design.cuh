#pragma once

// LAYER 2a -- THE DERIVED SELECTIONS.
//
// WHAT THE DESIGN LOOKS LIKE ON THIS ARCHITECTURE.  Every entry here is
// COMPUTED from `arch_caps.cuh`'s facts; none is looked up per architecture,
// because a per-arch table of selections is an architecture branch in a costume
// and the standing rule is to branch on the flag, never on the number.
//
// The kernel reads THIS layer for its design and calls `ops.cuh` to execute it.
// It never reads a capability fact, and it never sees an architecture number.
//
// WHAT IS HERE AND WHAT IS NOT
// ----------------------------
//   STATE_RESIDENCY   where the fold's accumulator lives while it is also the
//                     readout's operand.  DERIVED from the residency facts.
//   STAGES            the ring's depth.  DERIVED from the work one round trip
//                     must be covered with and from the shared-memory budget,
//                     capped by this kernel's live register set.  It was a free
//                     axis; what ended it is that the optimum turned out not to
//                     move in DEPTH but in TOKENS IN FLIGHT, which is a
//                     derivation from the chunk size rather than a search.
//   the CTA target    the second launch bound: the residency this arm can
//                     actually reach, shared-memory ceiling CAPPED BY THE
//                     REGISTER CEILING.  The cap is not optional -- without it
//                     the shared-memory ceiling ran past the register file and
//                     three arms spilled (p59b).
//
// NOT here, deliberately: the SHARED-MEMORY LAYOUT (pad-8 against an XOR
// swizzle) and the CHUNK SIZE.  Both are legal on every tabulated architecture
// and both have a measured optimum that MOVES -- the swizzle saved the dense
// base 0.545 inst/mma at C=16 and cost the routed rung 1.260 at the same arm, a
// genuine sign flip.  They stay FREE AXES, chosen by measurement, and a
// selection here would make every future measurement of them unfalsifiable.
//
// THE ONE OBLIGATION.  A selection that is knowingly suboptimal -- one where the
// table says a better implementation exists on this part and the baseline does
// not build it -- must ANNOUNCE ITSELF at compile time.  A documented gap a
// reader sees is a stated engineering decision; a SILENT slower path is the
// forbidden thing.  `Announce` below is that mechanism, and `ops.cuh` wires the
// sites, because whether an implementation is built is a fact about operations.

#include "arch_caps.cuh"

namespace denseref {
namespace design {

// ------------------------------------------------ the compile-time announcement
//: A compile-time diagnostic that does NOT fail the build.  Instantiating
//: `Announce<true>` calls a deprecated constructor, which every supported
//: toolchain reports with the message attached.  Used for a gap that is a
//: stated decision rather than a defect; a gap that would produce WRONG or
//: UNBUILDABLE code is a `static_assert` instead, and reads NOT BUILT.
template <bool Suboptimal> struct Announce {
  __host__ __device__ constexpr Announce() {}
};
template <> struct Announce<true> {
  [[deprecated(
      "KNOWINGLY SUBOPTIMAL SELECTION: this architecture's capability row "
      "offers a better implementation of a baseline operation than the one "
      "this baseline builds. The build is correct and the gap is stated; see "
      "the announcement site in ops.cuh for which operation.")]] __host__
  __device__ constexpr Announce() {}
};

// --------------------------------------------------------- STATE RESIDENCY
//: WHERE THE STATE LIVES.  The algorithm's requirement is not an instruction:
//: it is that THE FOLD'S ACCUMULATOR IS REUSABLE AS THE READOUT'S OPERAND
//: WITHOUT A MEMORY ROUND TRIP.  Ampere and Hopper satisfy it in the register
//: file (Hopper via the `_RS` atom family); a `tcgen05` part satisfies it in
//: tensor memory, with no register-resident state at all.  Naming the
//: requirement after the Ampere instruction would REFUSE that part for having
//: the wrong storage class while meeting the actual requirement.
enum class StateResidency { REGISTERS, TMEM, SHARED };

constexpr unsigned residency_bit(StateResidency r) {
  return 1u << (unsigned)r;
}

//: The residencies that MEET THE REQUIREMENT on this architecture, as a mask.
//: SHARED never appears: a state that has to go through shared memory HAS taken
//: the round trip, which is a different algorithm.
constexpr unsigned satisfiable_residencies(const arch::Caps &c) {
  return (c.accum_in_tmem && c.mma_a_from_tmem ? residency_bit(StateResidency::TMEM)
                                               : 0u) |
         (!c.accum_in_tmem && c.mma_a_from_regs
              ? residency_bit(StateResidency::REGISTERS)
              : 0u);
}

//: PREFERENCE ORDER, most preferred first.  TMEM is preferred where it exists
//: because the accumulator is already there and the register file stays free.
constexpr StateResidency kResidencyOrder[2] = {StateResidency::TMEM,
                                               StateResidency::REGISTERS};

//: The best residency that BOTH meets the requirement and has an implementation.
//: Returns SHARED to mean "none", which is the value that can never be selected.
constexpr StateResidency select_residency(unsigned satisfiable, unsigned built) {
  return (satisfiable & built & residency_bit(kResidencyOrder[0]))
             ? kResidencyOrder[0]
         : (satisfiable & built & residency_bit(kResidencyOrder[1]))
             ? kResidencyOrder[1]
             : StateResidency::SHARED;
}

//: True when a residency BETTER than the selected one meets the requirement on
//: this part and simply is not built.  This is the announcement's condition and
//: not an error: the built path is correct, it is merely not the best available.
constexpr bool residency_suboptimal(unsigned satisfiable, unsigned built) {
  return (satisfiable & residency_bit(kResidencyOrder[0])) &&
         select_residency(satisfiable, built) == kResidencyOrder[1];
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

//: THE RESIDENCIES THIS BASELINE BUILDS.  The one statement layer 2a needs from
//: layer 2b, and it is a statement about OPERATIONS rather than about hardware:
//: `ops.cuh`'s consistency section asserts that every residency named here is
//: served by an operation it actually implements, so the two cannot drift.
//:
//: REGISTERS ONLY, deliberately.  A TMEM path is not built and a SHARED path
//: never will be (it is the round trip this design exists to avoid).  Selecting
//: an unbuilt residency is a compile error reading NOT BUILT -- the architecture
//: is admissible in structure and the path is simply unwritten -- which is the
//: same distinction the 32-thread MMA unit already draws.  Building a
//: speculative path for hardware we do not have would be an operation invented
//: before it has a caller.
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

// ------------------------------------------------------------------ STAGES
//: THE RING'S DEPTH, DERIVED.  One clause: TAKE THE SHALLOWEST RING THAT REACHES
//: THIS ARM'S MAXIMUM RESIDENCY.  Depth is BOUGHT, never taken.
//:
//: THE SWEEP IS WHAT REMOVED THE AXIS.  Measured warm, one source, four builds
//: differing only in the depth each arm is given, on both cells: every arm of
//: every chunk size lands within 1.2% across depths 2..6, and the SHALLOWEST is
//: best or tied on eleven of the twelve (arm, cell) pairs -- worst case +0.4%,
//: at C=16 padded on the 16-CTA cell.  The optimum does not move, so the depth
//: is not an axis.
//:
//: WHY MINIMISING RATHER THAN MAXIMISING.  Shared memory is not free even at
//: fixed residency: it is carved out of the unified L1, so a ring deeper than
//: residency pays for reads the L1 would otherwise absorb.  Measured at C=64
//: padded, where every depth is one resident CTA: the deepest affordable ring
//: (S=4, 73,728 B) cost +3.2% at the 16-CTA cell and +5.8% at the filled cell
//: against S=2 (55,296 B).  In the other direction the clause PAYS: at C=64
//: swizzled, S=2 fits TWO CTAs where the hand-fixed S=3 fits one, worth -18.3%
//: at the filled cell.
//:
//: A NOTE ON THE INSTRUMENT, because it nearly wrote a different rule here.  The
//: first arm timed in a cold process bears the clock ramp from 210 MHz and reads
//: ~14% slow.  That artifact looked exactly like a coverage requirement -- deep
//: rings good at small chunks -- and a coverage clause was written, measured
//: against a warm instrument, and deleted.  Every number above is warm, behind a
//: discarded warm-up pass.
//:
//: On every arm of every tabulated architecture this evaluates to `kStagesMin`,
//: because the residency ceiling can only fall as the ring deepens.  The search
//: is written in the general form anyway: it states the RULE rather than its
//: current answer, and it is what would notice an arm whose fixed buffers leave
//: the shallowest ring unaffordable.
constexpr int kStagesMin = 2;   // the wait/issue order needs two groups
constexpr int kStagesMax = 8;   // the search bound, not a design constant

//: The largest per-lane register count is a table quantity; this is its inverse
//: -- the most CTAs whose budget still holds a live set of `reg_floor`.
constexpr int ctas_by_reg_floor(const arch::Caps &c, int threads, int reg_floor,
                                int n) {
  return n <= 1 ? 1
                : (arch::reg_budget(c, threads, n) >= reg_floor
                       ? n
                       : ctas_by_reg_floor(c, threads, reg_floor, n - 1));
}

//: The residency ceiling that is NOT a function of this arm's shared memory:
//: registers, threads and the hard CTA cap.
constexpr int ceiling_without_smem(const arch::Caps &c, int threads, int reg_floor) {
  return ctas_by_reg_floor(c, threads, reg_floor,
                           arch::ctas_by_threads(c, threads) < c.max_ctas_per_sm
                               ? arch::ctas_by_threads(c, threads)
                               : c.max_ctas_per_sm);
}

//: THE SECOND LAUNCH BOUND: the CTA count this arm can actually reach.  Shared
//: memory, registers, threads and the CTA cap, whichever binds first.
constexpr int cta_target(const arch::Caps &c, int smem_bytes, int threads,
                         int reg_floor) {
  return arch::ctas_by_smem(c, smem_bytes) < ceiling_without_smem(c, threads, reg_floor)
             ? arch::ctas_by_smem(c, smem_bytes)
             : ceiling_without_smem(c, threads, reg_floor);
}

//: A ring of `s` slots on top of `fixed` bytes of unringed buffers.
constexpr int ring_bytes(int slot_bytes, int fixed_bytes, int s) {
  return s * slot_bytes + fixed_bytes;
}

constexpr bool stages_fit(const arch::Caps &c, int slot_bytes, int fixed_bytes,
                          int s) {
  return ring_bytes(slot_bytes, fixed_bytes, s) <= c.smem_per_cta_max;
}

//: The residency reachable at ANY depth.  The ring can only shrink it, so this
//: is the shallowest LEGAL ring's residency; an arm whose shallowest ring does
//: not fit at all is refused by `smem_fits` at its own site rather than silently
//: deepened here.
constexpr int best_residency(const arch::Caps &c, int slot_bytes, int fixed_bytes,
                             int threads, int reg_floor, int s) {
  return s >= kStagesMax || stages_fit(c, slot_bytes, fixed_bytes, s)
             ? cta_target(c, ring_bytes(slot_bytes, fixed_bytes, s), threads, reg_floor)
             : best_residency(c, slot_bytes, fixed_bytes, threads, reg_floor, s + 1);
}

//: The SHALLOWEST ring that reaches `want` CTAs, searched upward from the
//: minimum: the point past which shared memory buys nothing and still costs L1.
constexpr int stages_by_residency(const arch::Caps &c, int slot_bytes,
                                  int fixed_bytes, int threads, int reg_floor,
                                  int want, int s) {
  return s >= kStagesMax
             ? s
             : (stages_fit(c, slot_bytes, fixed_bytes, s) &&
                        cta_target(c, ring_bytes(slot_bytes, fixed_bytes, s), threads,
                                   reg_floor) >= want
                    ? s
                    : stages_by_residency(c, slot_bytes, fixed_bytes, threads,
                                          reg_floor, want, s + 1));
}

constexpr int stages_for(const arch::Caps &c, int slot_bytes, int fixed_bytes,
                         int threads, int reg_floor) {
  return stages_by_residency(
      c, slot_bytes, fixed_bytes, threads, reg_floor,
      best_residency(c, slot_bytes, fixed_bytes, threads, reg_floor, kStagesMin),
      kStagesMin);
}

//: The compiling target's answers, so a kernel writes a geometry and never a
//: capability row.
constexpr int stages_for(int slot_bytes, int fixed_bytes, int threads,
                         int reg_floor) {
  return stages_for(arch::kCaps, slot_bytes, fixed_bytes, threads, reg_floor);
}
constexpr int cta_target(int smem_bytes, int threads, int reg_floor) {
  return cta_target(arch::kCaps, smem_bytes, threads, reg_floor);
}

//: THE ONE HARD RESOURCE REFUSAL, from the table rather than from a literal: an
//: arm either fits this architecture's per-CTA shared-memory maximum or it is
//: refused.  No arm is silently shrunk to fit.
constexpr bool smem_fits(size_t bytes) {
  return bytes <= (size_t)arch::kCaps.smem_per_cta_max;
}

//: The digest of the capability row THE DEVICE CODE WAS COMPILED AGAINST,
//: carried here so a kernel's build stamp names no layer-1 symbol.
constexpr int kCapsDigest = arch::caps_digest(arch::kCaps);

}  // namespace design
}  // namespace denseref
