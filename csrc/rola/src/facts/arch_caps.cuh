#pragma once

// LAYER 1 -- THE ARCHITECTURE CAPABILITY FACTS: the only file that knows an
// architecture exists.  Objective per-arch facts, no judgment; the consumers
// are the facts dispatch's closed-world runtime check (facts.cu) and, when it
// lands, the K31 kernel's own selection layers.  The entry kinds (REQUIRED /
// SELECTING / RESIDENCY / FACT) and the no-fallback-chain rule live in
// docs/internals/facts/arch_caps.md.

#include <cuda_runtime.h>

namespace denseref {
namespace arch {

struct Caps {
  int cc;  // compute capability x10, e.g. 860

  // -------------------------------------------------------------- REQUIRED
  // Each of these is a clause of the baseline structure.  `supported()` is
  // their conjunction; a false clause is an UNSUPPORTED architecture, not a
  // slower path.
  bool async_copy;         // global->shared staging whose completion is a signal
  bool tile_load;          // 8x8 b16 quads landed directly in MMA register order
  bool mma_bf16_f32;       // tensor MAC, bf16 operands, fp32 accumulator
  bool bf16_cvt_packed;    // two fp32 -> one b32 of two bf16, one instruction
  bool reg_transpose_b16;  // an 8x8 b16 tile spread across the MMA unit's own
                           // registers is transposed IN PLACE.  A clause of the
                           // baseline because the adjoint's contraction runs
                           // over the accumulator's OWN m axis, which puts the
                           // accumulator on the k side, and the accumulator map
                           // and the k-side map differ by exactly this
                           // transpose (measured on the retired consumer's
                           // catalogue; baseline = tag baseline/pre-k31).

  // ------------------------------------------------------------- RESIDENCY
  // Instruction facts about WHERE an accumulator lives and where an operand may
  // be sourced from.  `design.cuh` reads these and nothing else does: the
  // algorithm's requirement is that the fold's accumulator be reusable as the
  // readout's operand WITHOUT A MEMORY ROUND TRIP, which is a statement about
  // residency and not about any one instruction.
  bool mma_a_from_regs;  // the A operand may be supplied from the issuing
                         // unit's OWN registers.  Warpgroup MMAs keep a
                         // register-source A (the `_RS` atom family), so this
                         // holds on every tabulated architecture.
  bool mma_b_from_regs;  // the same for B.  A warpgroup MMA sources B ONLY
                         // from a shared-memory descriptor.  Which SIDE a
                         // given kernel's accumulator lands on is a clause of
                         // THAT kernel, asserted against the built operation
                         // at its own site.
  bool accum_in_tmem;    // the MMA's accumulator lives in a dedicated tensor
                         // memory rather than in the unit's register file
                         // (`tcgen05`).  False on every row below; a Blackwell
                         // row would set it, and the residency DERIVATION --
                         // not a new refusal -- is what would then have to
                         // hold.
  bool mma_a_from_tmem;  // the A operand may be sourced from that same tensor
                         // memory, which is what lets the invariant survive
                         // with no register-resident state at all.

  // ------------------------------------------------------------- SELECTING
  bool bf16_simd_arith;  // elementwise bf16 arithmetic on a packed pair
                         // (`mul.rn.bf16x2`).  MEASURED 2026-08-09: ptxas
                         // rejects it below sm_90 -- "requires .target sm_90
                         // or higher".  Ampere has native bf16 CONVERSION and
                         // native bf16 MMA but no bf16 SIMD ARITHMETIC, so
                         // the same product costs 1 FMUL + 1/2 pack per
                         // element there against ~1/2 instruction on sm_90.
                         // Same baseline operation, different implementation,
                         // no structural change: the archetypal flag.
  bool async_bulk_copy;  // TMA.  Would implement `stage_tile`'s contract with
                         // one descriptor issue instead of one per 16 B.
  bool warpgroup_mma;    // `wgmma`.  Present on sm_90 and the reason
                         // `mma_b_from_regs` is false there.
  bool dyn_reg_realloc;  // `setmaxnreg`: a warpgroup may GIVE UP or CLAIM
                         // registers at run time, so two groups of the same
                         // CTA can hold different per-lane allocations.
                         // Without it, allocation is uniform per thread across
                         // the whole kernel and sized by the MAXIMUM live set
                         // over the paths a thread takes -- which is why a
                         // thread that produces and then consumes pays
                         // `max(producer, consumer)` and reuses the registers,
                         // and why DEDICATING a unit to production cannot be
                         // paid for by shrinking it.

  // ------------------------------------------------------------------ FACT
  int mma_unit_threads;  // threads that jointly issue one MMA and jointly own
                         // its accumulator.  32 (warp) or 128 (warpgroup).
                         // The carve is expressed in THESE units, not warps.
  int smem_per_sm;       // `cudaDevAttrMaxSharedMemoryPerMultiprocessor`
  int smem_per_cta_max;  // opt-in dynamic maximum, one CTA
  int smem_granularity;  // shared-memory allocation quantum, bytes
  int regs_per_sm;
  int regs_per_lane_max;
  int reg_alloc_unit;  // registers allocated per warp, in units of this
  int max_threads_per_sm;
  int max_ctas_per_sm;
  int driver_smem_reserve;   // per-CTA reserve the driver takes off the top; the
                             // residency condition is on smem_per_cta + this
  int tensor_f32_accum_pct;  // fp32-accumulate tensor rate as a percentage of
                             // the fp16-accumulate rate.  50 on consumer
                             // Ampere/Ada, 100 on GA100.  Not a branch: it is
                             // why a tensor-bound baseline (FA) is flattered by
                             // a consumer part and ratios must be quoted from
                             // the target.
};

//: The RATIFIED SET.  A compute capability that is not a row here fails the
//: build; it is never approximated by a neighbouring row.
constexpr bool tabulated(int cc) {
  return cc == 800 || cc == 860 || cc == 870 || cc == 890 || cc == 900;
}

// Field order, for every row below:
//   cc |  REQUIRED: async_copy, tile_load, mma_bf16_f32, bf16_cvt_packed,
//                   reg_transpose_b16
//      | RESIDENCY: mma_a_from_regs, mma_b_from_regs, accum_in_tmem,
//                   mma_a_from_tmem
//      | SELECTING: bf16_simd_arith, async_bulk_copy, warpgroup_mma,
//                   dyn_reg_realloc
//      |      FACT: mma_unit_threads, smem_per_sm, smem_per_cta_max,
//                   smem_granularity, regs_per_sm, regs_per_lane_max,
//                   reg_alloc_unit, max_threads_per_sm, max_ctas_per_sm,
//                   driver_smem_reserve, tensor_f32_accum_pct
constexpr Caps caps_of(int cc) {
  return cc == 800   ? Caps{800,   true,  true,  true,  true,  true, true,   true,   false,
                          false, false, false, false, false, 32,   167936, 166912, 128,
                          65536, 255,   256,   2048,  32,    1024, 100}
         : cc == 870 ? Caps{870,   true,  true,  true,  true,  true, true,   true,   false,
                            false, false, false, false, false, 32,   167936, 166912, 128,
                            65536, 255,   256,   1536,  16,    1024, 50}
         : cc == 890 ? Caps{890,   true,  true,  true,  true,  true, true,   true,   false,
                            false, false, false, false, false, 32,   102400, 101376, 128,
                            65536, 255,   256,   1536,  24,    1024, 50}
         : cc == 900
             // sm_90 is TABULATED AND NOT BUILT.  Every REQUIRED clause holds and
             // the residency derivation succeeds -- `wgmma` keeps a
             // register-source A operand (the `_RS` atom family), so the state
             // can be an operand without a round trip there too.  What is
             // missing is an IMPLEMENTATION: the built `mma`, `load_frag` and
             // carve are written for a 32-thread unit and a warpgroup unit is
             // 128.  A capability GAP, not a structural refusal.
             ? Caps{900,   true, true, true, true, true, true,   false,  false,
                    false, true, true, true, true, 128,  233472, 232448, 128,
                    65536, 255,  256,  2048, 32,   1024, 100}
             // sm_86: the local development part (GA102).
             : Caps{860,   true,  true,  true,  true,  true, true,   true,   false,
                    false, false, false, false, false, 32,   102400, 101376, 128,
                    65536, 255,   256,   1536,  16,    1024, 50};
}

//: The conjunction of the REQUIRED clauses.  There is no partial support.
//: `mma_a_from_regs` is NOT among them and must not be: whether the state can be
//: an operand without a memory round trip is a RESIDENCY question, answered by
//: `design.cuh` from the residency facts, and an architecture that answers it a
//: different way is a derivation rather than a refusal.
constexpr bool supported(const Caps& c) {
  return c.async_copy && c.tile_load && c.mma_bf16_f32 && c.bf16_cvt_packed && c.reg_transpose_b16;
}

// ------------------------------------------------------------- DERIVATION
// The quantitative half.  These are the only consumers of the FACT rows, and
// none of them is a branch on an architecture.

constexpr int round_up(int x, int q) { return (x + q - 1) / q * q; }

//: Registers a CTA actually reserves.  Allocation is per warp in `reg_alloc_unit`
//: quanta, so a lane asking for 129 costs the same as one asking for 136.
constexpr int regs_per_cta(const Caps& c, int threads, int regs_per_lane) {
  return threads / 32 * round_up(regs_per_lane * 32, c.reg_alloc_unit);
}

constexpr int ctas_by_smem(const Caps& c, int smem_per_cta) {
  return c.smem_per_sm / (round_up(smem_per_cta, c.smem_granularity) + c.driver_smem_reserve);
}
constexpr int ctas_by_regs(const Caps& c, int threads, int regs_per_lane) {
  return c.regs_per_sm / regs_per_cta(c, threads, regs_per_lane);
}
constexpr int ctas_by_threads(const Caps& c, int threads) { return c.max_threads_per_sm / threads; }

//: Resident CTAs per SM: the binding constraint among shared memory, registers,
//: threads and the hard CTA cap.
constexpr int min4(int a, int b, int d, int e) {
  return a < b ? (a < d ? (a < e ? a : e) : (d < e ? d : e))
               : (b < d ? (b < e ? b : e) : (d < e ? d : e));
}
constexpr int residency(const Caps& c, int smem_per_cta, int threads, int regs_per_lane) {
  return min4(ctas_by_smem(c, smem_per_cta), ctas_by_regs(c, threads, regs_per_lane),
              ctas_by_threads(c, threads), c.max_ctas_per_sm);
}

//: THE PER-ARCH REGISTER BUDGET.  The largest per-lane register count that
//: still admits `target_ctas` CTAs of `threads` threads, honouring the
//: allocation quantum.  This is the number that must NEVER be a global
//: literal: at 256 threads and 3 CTAs it is 80 on every part with 65536
//: registers per SM, and the arithmetic that produced it is the table's, not a
//: measurement session's.
constexpr int reg_budget(const Caps& c, int threads, int target_ctas) {
  return (c.regs_per_sm / target_ctas / (threads / 32) / c.reg_alloc_unit * c.reg_alloc_unit) / 32
                 < c.regs_per_lane_max
             ? (c.regs_per_sm / target_ctas / (threads / 32) / c.reg_alloc_unit * c.reg_alloc_unit)
                   / 32
             : c.regs_per_lane_max;
}

// ---------------------------------------------------- the compiling target
// Inert in the host pass, where it is pinned to the SMALLEST tabulated
// configuration so that if it ever stopped being inert it would under-promise.

#ifdef __CUDA_ARCH__
constexpr int kArch = __CUDA_ARCH__;
#else
constexpr int kArch = 860;
#endif

static_assert(tabulated(kArch),
              "untabulated compute capability: add its row to caps_of() rather "
              "than letting an operation guess which implementation it has");

constexpr Caps kCaps = caps_of(kArch);

static_assert(supported(kCaps),
              "UNSUPPORTED ARCHITECTURE: a required capability "
              "clause is false; see the Caps row");

// `mma_unit_threads` is NOT asserted here.  That the carve is written for a
// 32-thread unit is a fact about the OPERATION THIS BASELINE BUILDS, not about
// any architecture, so it is a 2a/2b consistency clause and lives in `ops.cuh`.

//: A digest of the whole active row, written by a kernel and checked on the
//: host.  A binary compiled against a different table cannot masquerade as this
//: one -- no path or file hash can establish that, only a device-side fact.
constexpr int caps_digest(const Caps& c) {
  return c.cc * 1000 + (int)c.async_copy * 1 + (int)c.tile_load * 2 + (int)c.mma_bf16_f32 * 4
         + (int)c.mma_b_from_regs * 8 + (int)c.bf16_cvt_packed * 16 + (int)c.bf16_simd_arith * 32
         + (int)c.async_bulk_copy * 64 + (int)c.warpgroup_mma * 128 + (int)c.mma_a_from_regs * 256
         + (int)c.accum_in_tmem * 512 + (int)c.mma_a_from_tmem * 1024
         + (int)c.reg_transpose_b16 * 2048 + c.mma_unit_threads * 2;
}

}  // namespace arch
}  // namespace denseref
