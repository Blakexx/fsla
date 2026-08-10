#pragma once

// THE ARCHITECTURE CAPABILITY TABLE.
//
// One BASELINE STRUCTURE, stated in `ops.cuh` as operations with contracts.
// This file says, per architecture, WHICH IMPLEMENTATION OF EACH OPERATION IS
// AVAILABLE, and it says it in exactly two kinds of entry:
//
//   REQUIRED   a capability whose absence would change the STRUCTURE, not the
//              implementation.  Never a branch: `kSupported` goes false and the
//              build stops with the clause that failed.
//   SELECTING  a capability that implements a baseline operation more
//              efficiently.  A compile-time bool.  Code branches on the FLAG,
//              never on an architecture number.
//
// and one kind of non-capability entry:
//
//   FACT       a quantitative resource (bytes, registers, threads).  NOT a
//              branch under any circumstances.  It is an INPUT TO DERIVATION:
//              residency, staging depth, the register budget.
//
// THERE IS NO THIRD KIND, and in particular there are no fallback chains.  A
// capability must never silently select a slower route that then ships
// unnoticed; that is the difference between this table and a fallback chain.
//
// PER-SM FACTS LIVE HERE; PER-DEVICE FACTS DO NOT.  SM count is not an
// architectural constant -- GA102 ships at 80 and 82 SMs, GA100 at 108 -- so it
// is a runtime `cudaDeviceProp` query, never a table row.  Everything in
// `Caps` is a property of the compute capability itself.

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
  bool mma_b_from_regs;    // the B operand may be supplied from the unit's OWN
                           // registers.  THE PIVOT: the fold accumulator is the
                           // readout's B operand, so the state never reaches
                           // shared memory.  An MMA that can only take B from a
                           // shared-memory descriptor implements a DIFFERENT
                           // algorithm, not a slower one.
  bool bf16_cvt_packed;    // two fp32 -> one b32 of two bf16, one instruction

  // ------------------------------------------------------------- SELECTING
  bool bf16_simd_arith;    // elementwise bf16 arithmetic on a packed pair
                           // (`mul.rn.bf16x2`).  MEASURED 2026-08-09: ptxas
                           // rejects it below sm_90 -- "requires .target sm_90
                           // or higher".  Ampere has native bf16 CONVERSION and
                           // native bf16 MMA but no bf16 SIMD ARITHMETIC, so
                           // the same product costs 1 FMUL + 1/2 pack per
                           // element there against ~1/2 instruction on sm_90.
                           // Same baseline operation, different implementation,
                           // no structural change: the archetypal flag.
  bool async_bulk_copy;    // TMA.  Would implement `stage_tile`'s contract with
                           // one descriptor issue instead of one per 16 B.
  bool warpgroup_mma;      // `wgmma`.  Present on sm_90 and the reason
                           // `mma_b_from_regs` is false there.

  // ------------------------------------------------------------------ FACT
  int mma_unit_threads;    // threads that jointly issue one MMA and jointly own
                           // its accumulator.  32 (warp) or 128 (warpgroup).
                           // The carve is expressed in THESE units, not warps.
  int smem_per_sm;         // `cudaDevAttrMaxSharedMemoryPerMultiprocessor`
  int smem_per_cta_max;    // opt-in dynamic maximum, one CTA
  int smem_granularity;    // shared-memory allocation quantum, bytes
  int regs_per_sm;
  int regs_per_lane_max;
  int reg_alloc_unit;      // registers allocated per warp, in units of this
  int max_threads_per_sm;
  int max_ctas_per_sm;
  int driver_smem_reserve; // per-CTA reserve the driver takes off the top; the
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

constexpr Caps caps_of(int cc) {
  return cc == 800
             ? Caps{800, true, true, true, true, true,
                    false, false, false,
                    32, 167936, 166912, 128, 65536, 255, 256, 2048, 32, 1024, 100}
         : cc == 870
             ? Caps{870, true, true, true, true, true,
                    false, false, false,
                    32, 167936, 166912, 128, 65536, 255, 256, 1536, 16, 1024, 50}
         : cc == 890
             ? Caps{890, true, true, true, true, true,
                    false, false, false,
                    32, 102400, 101376, 128, 65536, 255, 256, 1536, 24, 1024, 50}
         : cc == 900
             // sm_90 is TABULATED AND REFUSED.  Every clause but one holds; the
             // table exists so the refusal names its reason and so a future
             // pass can see exactly what has to change.  `wgmma` sources B from
             // a shared-memory descriptor, which is why ThunderKittens'
             // Hopper linear-attention kernels round-trip their state through
             // shared memory.  That is a different algorithm.  Selecting
             // warp-scoped `mma.sync` on a part that has `wgmma` would be a
             // fallback chain, which this table does not have.
             ? Caps{900, true, true, true, false, true,
                    true, true, true,
                    128, 233472, 232448, 128, 65536, 255, 256, 2048, 32, 1024, 100}
             // sm_86: the local development part (GA102).
             : Caps{860, true, true, true, true, true,
                    false, false, false,
                    32, 102400, 101376, 128, 65536, 255, 256, 1536, 16, 1024, 50};
}

//: The conjunction of the REQUIRED clauses.  There is no partial support.
constexpr bool supported(const Caps &c) {
  return c.async_copy && c.tile_load && c.mma_bf16_f32 && c.mma_b_from_regs &&
         c.bf16_cvt_packed;
}

// ------------------------------------------------------------- DERIVATION
// The quantitative half.  These are the only consumers of the FACT rows, and
// none of them is a branch on an architecture.

constexpr int round_up(int x, int q) { return (x + q - 1) / q * q; }

//: Registers a CTA actually reserves.  Allocation is per warp in `reg_alloc_unit`
//: quanta, so a lane asking for 129 costs the same as one asking for 136.
constexpr int regs_per_cta(const Caps &c, int threads, int regs_per_lane) {
  return threads / 32 * round_up(regs_per_lane * 32, c.reg_alloc_unit);
}

constexpr int ctas_by_smem(const Caps &c, int smem_per_cta) {
  return c.smem_per_sm /
         (round_up(smem_per_cta, c.smem_granularity) + c.driver_smem_reserve);
}
constexpr int ctas_by_regs(const Caps &c, int threads, int regs_per_lane) {
  return c.regs_per_sm / regs_per_cta(c, threads, regs_per_lane);
}
constexpr int ctas_by_threads(const Caps &c, int threads) {
  return c.max_threads_per_sm / threads;
}

//: Resident CTAs per SM: the binding constraint among shared memory, registers,
//: threads and the hard CTA cap.
constexpr int min4(int a, int b, int d, int e) {
  return a < b ? (a < d ? (a < e ? a : e) : (d < e ? d : e))
               : (b < d ? (b < e ? b : e) : (d < e ? d : e));
}
constexpr int residency(const Caps &c, int smem_per_cta, int threads,
                        int regs_per_lane) {
  return min4(ctas_by_smem(c, smem_per_cta), ctas_by_regs(c, threads, regs_per_lane),
              ctas_by_threads(c, threads), c.max_ctas_per_sm);
}

//: THE PER-ARCH REGISTER BUDGET.  The largest per-lane register count that
//: still admits `target_ctas` CTAs of `threads` threads, honouring the
//: allocation quantum.  This is the number that must NEVER be a global
//: literal: at 256 threads and 3 CTAs it is 80 on every part with 65536
//: registers per SM, and the arithmetic that produced it is the table's, not a
//: measurement session's.
constexpr int reg_budget(const Caps &c, int threads, int target_ctas) {
  return (c.regs_per_sm / target_ctas / (threads / 32) / c.reg_alloc_unit *
          c.reg_alloc_unit) /
                     32 <
                 c.regs_per_lane_max
             ? (c.regs_per_sm / target_ctas / (threads / 32) / c.reg_alloc_unit *
                c.reg_alloc_unit) /
                   32
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

static_assert(kCaps.mma_b_from_regs,
              "UNSUPPORTED ARCHITECTURE: this kernel's structure requires an MMA "
              "whose B operand can come from the issuing unit's own registers, "
              "because the fold accumulator IS the readout's B operand and the "
              "state never enters shared memory. A warpgroup MMA that sources B "
              "from a shared-memory descriptor implements a different algorithm. "
              "Supporting it is work, not a flag.");
static_assert(supported(kCaps), "UNSUPPORTED ARCHITECTURE: a required capability "
                                "clause is false; see the Caps row");
static_assert(kCaps.mma_unit_threads == 32,
              "UNSUPPORTED ARCHITECTURE: the d_v carve gives one MMA unit a "
              "private 16-row slice of the state; at 128 threads per unit the "
              "carve and the CTA shape are both different");

//: A digest of the whole active row, written by a kernel and checked on the
//: host.  A binary compiled against a different table cannot masquerade as this
//: one -- no path or file hash can establish that, only a device-side fact.
constexpr int caps_digest(const Caps &c) {
  return c.cc * 1000 + (int)c.async_copy * 1 + (int)c.tile_load * 2 +
         (int)c.mma_bf16_f32 * 4 + (int)c.mma_b_from_regs * 8 +
         (int)c.bf16_cvt_packed * 16 + (int)c.bf16_simd_arith * 32 +
         (int)c.async_bulk_copy * 64 + (int)c.warpgroup_mma * 128 +
         c.mma_unit_threads * 2;
}

}  // namespace arch
}  // namespace denseref
