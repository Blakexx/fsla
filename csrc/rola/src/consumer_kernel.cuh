#pragma once

// THE RoLA V3 CONSUMER KERNEL -- the §10i.12 synthesis. It is the ONLY consumer.
//
// ORIGIN, kept because the numbers below are its evidence and a rename does not
// re-derive them: this began as the stage-1 SHAPE PROBE of
// `workflows/20260801-rewrite`, built to answer one question with a measurement --
// does the §10i.12/§10i.13 restructure (state SMEM-resident instead of
// register-resident, per-chunk work lowered to §10i.8's coefficient-domain GEMMs,
// and a WIDE CTA freed by the first two) move the kernel decisively off the
// latency-bound regime ncu attributed in harvest.md §30.3 (tensor pipe 0.60 % of
// peak, warps active 8.3 %, 0.28 inst/cycle)? It did (x2.32 at `L = 2048`); the
// register-resident consumer it was A/B'd against is DELETED (item 6b); this is
// the production path. Item 6c retires the probe framing and the `spike` naming
// with it. What survives the rename is the measurement.
//
// WHAT IT KEEPS FROM THE DELETED CONSUMER (deliberately, so the A/B isolated the
// restructure and nothing else):
//   * the same `ConsumerParams`, the same schedule, the same candidate walk
//     (`read_union`/`write_union` AND the fixed levels' tile bitmaps, §4),
//   * the same owner-stationary run-to-completion dataflow -- ONE CTA PER OWNER,
//     `blockIdx.x` indexing the same work list. §10i.12 keeps this ("it was
//     traffic-optimal all along") and so does this kernel.
//   * the same split-K atomic `num`/`den` accumulation and the same `divide_pass`.
//
// WHAT IT CHANGES (the three moves that ARE §10i.12, and nothing else):
//   1. STATE LIVES IN SMEM (`S[BC][SP]`), not in `M[BC]` per thread. §10i.13: the
//      register file is wrong for this object because it is thread-PRIVATE (so the
//      state layout dictates the thread structure), NOT ADDRESSABLE, and it locks
//      updates out of the tensor cores. Note the register-resident consumer ALREADY
//      paid this SMEM (`state_pub`, the MMA readout's B operand) -- BOTH copies. So
//      the move costs zero net shared memory and returns 64 registers per thread.
//   2. EVERY PER-TILE TERM IS A GEMM (§10i.8's coefficient domain, chunk-scoped):
//        readout   y_tile  = R . S                 [BT x BC] x [BC x DV]
//        intra     G       = R . W^T               [BT x BC] x [BC x BT], causal
//                  y_tile += G . Vw                [BT x BT] x [BT x DV]
//        fold      S      += W^T . Vw              [BC x BT] x [BT x DV]
//      The fold is §10i.8's "S += W_chunk^T V_chunk, one batched tensor-core GEMM
//      per live owner", REPLACING that consumer's per-visit scattered d_v-wide
//      scalar accumulates -- the 0.28 inst/cycle serial chains §10i.5 measured.
//   3. THE CTA IS 8 WARPS (256 threads), not `DV` = 64 threads / 2 warps. This is
//      the payoff of (1): once the state is shared rather than private, the thread
//      count decouples from the state layout (§10i.13).
//
// THE ONE-COPY INVARIANT (§10i.10 Q3) IS STRUCTURAL HERE, and it is worth saying
// exactly why, because it turns out to need NO wave machinery at all: the grid is
// one CTA per owner, run to completion. A CTA reads its owner's `[BC x cols]` state
// block ONCE at entry and writes it ONCE at exit. §10i.12's "waves" are therefore
// just the CUDA scheduler's own CTA rotation when `gridDim` exceeds the resident
// capacity -- the cost of a wave is exactly the token re-read §10i.12 prices, and
// there is no explicit wave scheduling, no per-chunk state history, and no second
// bound state in HBM at any instant. The invariant is a property of the LAUNCH
// SHAPE, not of a policy the kernel has to implement.
//
// THE SUPPORTED SURFACE IS DECLARED AT `launch_consumer` (`consumer.cu`) AND
// NOT RESTATED HERE. The probe's five stated restrictions are all LIFTED as of
// items 6a/6b: `decay`, `normalization`, `P > 1`/paging and `MmaPrec` are template
// axes with every value instantiated, and `BT x BC x d_v` spans the planner's own
// `_SUPPORTED_*` sets exactly. The instantiation matrix, its arity and the reason
// every axis is a template rather than a runtime test live at that dispatch, which
// is the one place a shape can be refused -- restating them here is the second
// description of one kernel this file's own rule (`consumer.cu`'s banner) bars.
//
// ONE SCOPE NOTE THAT IS AN ALGORITHM FACT RATHER THAN A DISPATCH FACT: under
// `normalization = "global"` the mass column rides as column `DV` of the state,
// and is handled by scalar reductions rather than as an extra MMA `n` (one extra
// `n` would waste 7/8 of an `n8` fragment -- see `frag_a_get` below).

#include <cuda/std/type_traits>
#include <cuda_bf16.h>

#include "arch_table.cuh"
#include "intra_panel.cuh"
#include "consumer.cuh"
#include "mma.cuh"
#include "readout_mma.cuh"
#include "segment_mask.cuh"

namespace rola {
namespace consumer {

using rola::mma::mma_m16n8k16;
using rola::mma::pack_bf16x2;
using rola::clamp_bf16_finite;
//: §12 rules 1 and 3's survival factors, taken from `intra_panel.cuh` rather than
//: reimplemented: it is the one clamp point and the one exponent sign convention,
//: and a second copy of `expf(fminf(x, 88.0f))` is a second thing that can drift.
using rola::panel_read_factor;
using rola::panel_write_factor;
//: §6.4's per-segment tile-range mask, from `segment_mask.cuh` -- one named
//: definition, because the host-side segment partition at `P > 1` is pinned against
//: exactly this arithmetic.
using rola::detail::segment_word_mask;

constexpr int kConsumerThreads = 256;   // 8 warps -- the §10i.13 CTA-width unlock


//: **A MEASURED KERNEL FACT, NOT AN ARCHITECTURE FACT AND NOT A THRESHOLD.**
//:
//: This is the register working set `ptxas` reports for THIS kernel's GEMM inner loop
//: when it is given slack -- per `mma.m16n8k16`, 4 A fragments + 2 B fragments + 4
//: accumulators times the fold's warp tiles, plus the operand-pack temporaries and
//: loop state. It is keyed on `BC` AND on `DECAY`, the two plan axes the working set
//: actually varies with (the `BC = 64` arms carry four fold m-tiles where `BC = 16`
//: carries one; the decay arms carry two survival panels, the clock prefix and the
//: two coefficient folds on top of the same GEMM), and every number is READ OFF
//: `-Xptxas -v`, never chosen:
//:
//:   `BC = 64`             : 66 registers (`decay = None`), **80** (decay + compact)
//:   `BC = 16, decay=false`: 60-64 registers, ZERO spills on all 32 entries
//:   `BC = 16, decay=true` : 80 registers, and at a 64-register budget it SPILLS
//:                           (worst 66 st / 212 ld, §2.22)
//:
//: The MAXIMUM per key is taken, because this constant divides: a budget that is too
//: LARGE only understates the residency the register file admits, while one that is
//: too small over-promises and the allocator buys the difference by SPILLING. The
//: conservative direction is therefore well defined, and it is the one taken.
//:
//: **WHY `DECAY` IS A KEY AND NOT AN AVERAGE -- BOTH SIDES OF THE TRADE ARE
//: MEASURED (§2.23).** A single `BC = 16` budget loses on one arm whichever value it
//: takes, because the two arms want opposite things and the loser is not small.
//: One `ncu` launch per arm at `topo-D4-w8x8x8x16-BC16` (1024 CTAs, ~3 waves),
//: `--clock-control base`, which is what makes cross-BUILD durations comparable on a
//: contended box:
//:
//:   arm            budget  regs  CTAs/SM  warps_active  duration   local ld sectors
//:   -------------- ------  ----  -------  ------------  ---------  ----------------
//:   decay = None      64     64     4        63.9 %     278-293 us      11,262
//:   decay = None      80     69     3        49.5 %     324-336 us      11,262
//:   decay = warm      64     64     4        65.7 %     626-635 us   3,345,396
//:   decay = warm      80     80     3        49.7 %       588 us        142,330
//:
//: On the `decay = None` arm the budget of 80 buys NOTHING (the local traffic is
//: identical -- it is local arrays, not spills) and costs a CTA: **+12 to +16 %**.
//: On the decay arm it removes 23x the local LOAD traffic and is **-6.5 %** even
//: though it gives up a quarter of the occupancy. Keying the constant is therefore
//: not a tuning knob, it is the two measurements written down.
//:
//: `BC = 64` is NOT keyed, and that is also a measurement rather than an omission:
//: shared memory binds there at 3 CTAs/SM on every arm, so the register term is
//: slack and both 66 and 80 produce the same bound.
//:
//: Reproduce the register/spill side with `tools/probe_launch_bounds.cu`
//: (`tools/ratify.py` compiles it over all 256 entries) and
//: the residency side with `rola_consumer_occupancy`; if the kernel's inner loop
//: changes, re-run both and update these numbers rather than the formula.
//: **THE ROW-BLOCK TERM (item 6a, 2026-08-01) -- MEASURED OVER FIVE CANDIDATE
//: BUDGETS, AND IT IS KEYED ON `NM` ALONE.** A `BT = 32` tile is `NM = 2` `m16` row
//: blocks, so the readout and the intra term carry a second live y-fragment quad per
//: warp wherever `DV = 64`, and a second A-fragment block everywhere. `d_v` was
//: expected to matter as well and the measurement says it does NOT -- so it is not a
//: key, because a key nobody can point at a number for is an extrapolation.
//:
//: The instrument is `probe_launch_bounds.cu` over all 1024 instantiations, once
//: per candidate budget, with the ratified `BT = 16, DV = 64` matrix held at its
//: landed value as an IN-RUN CONTROL (it came back byte-identical at all five sweep
//: points, so the numbers below are the new cells' own and nothing else's). Spill
//: bytes summed over the 64 instantiations of each key:
//:
//:   (BT, DV, BC, decay)   b=64    b=80    b=96   b=128   b=168
//:   -------------------  ------  ------  ------  ------  ------
//:   (16, 32, 16, false)     300       0       0       0       0
//:   (16, 32, 16, true )    3938     352     872     872       0
//:   (16, 32, 64, false)     772    1056     728     728       0
//:   (16, 32, 64, true )    4480    1884    1732    1732       0
//:   (32, 32, 16, false)    2528    1272       0       0       0
//:   (32, 32, 16, true )    5614    2532    1612    1612    1624
//:   (32, 32, 64, false)    2666    1388    1276    1276       0
//:   (32, 32, 64, true )    3488    2388    1860    1860    1680
//:   (32, 64, 16, false)    1860    1188      96      96       0
//:   (32, 64, 16, true )    5566    2316     344     344     384
//:   (32, 64, 64, false)     568     568     416     416       0
//:   (32, 64, 64, true )    1768    1768     992     992     352
//:
//: THREE THINGS THE TABLE DECIDES, AND ALL THREE ARE READINGS RATHER THAN CHOICES:
//:
//: 1. **`b = 128` IS `b = 96`.** `by_regs = 65536 / (256 * b)` is 2 at both, so the
//:    two budgets state the SAME launch bound and ptxas emits the same code -- every
//:    row above is identical between the two columns. 128 is therefore not a
//:    candidate, it is a spelling of 96.
//: 2. **`b = 168` IS NOT A CANDIDATE EITHER**: it states ONE CTA/SM, which is the
//:    unconstrained allocator, and it is in the table only to show what the working
//:    set costs when nothing bounds it. Note that three keys STILL spill there, at
//:    255 registers -- those are local ARRAYS, not pressure, exactly as §2.23 found.
//: 3. **`d_v = 32` NEEDS NO NEW KEY.** At `BT = 16` the ratified `(BC, decay)`
//:    formula is already the best shipping value on three of the four `DV = 32` keys
//:    (80 at `BC = 16, decay`; 80 at both `BC = 64` arms is within 300 bytes of the
//:    best and 96 costs a CTA there). The fourth, `(16, 32, 16, false)`, spills 300
//:    bytes over 7 of its 64 instantiations at 64 where 80 spills none -- and §2.23
//:    MEASURED that trade on the same arm one `d_v` over: on a no-decay `BC = 16`
//:    arm the extra budget buys a CTA's worth of nothing and costs +12 to +16 %,
//:    against a spill smaller than the 4 st / 8 ld §2.22 accepted as the optimum.
//:    The occupancy keeps it.
//:
//: What is left is `NM`, and it is not close: at `NM = 2` the ratified formula's 64
//: and 80 leave 1860-5614 spill bytes on the `BC = 16` keys where 96 leaves 0-344,
//: and 96 is best-or-equal on ALL EIGHT `NM = 2` keys. `NM` is the same tile-count
//: predicate the warp partition forks on, so the budget and the codegen state the
//: same structural fact once.
constexpr int consumer_reg_budget(int BT, int BC, bool decay) {
  const int base = (BC >= 64) ? 80 : (decay ? 80 : 64);
  //: `NM` -- the tile's `m16` ROW BLOCKS. Two of them is a second A-fragment block
  //: and (at `DV = 64`) a second y quad live across all three readout GEMMs.
  return (BT / 16 >= 2) ? 96 : base;
}

//: CTAs/SM to STATE IN THE LAUNCH BOUND -- the smallest of what the three hardware
//: resources admit. STRUCTURAL, computed per instantiation rather than fixed: the
//: decay arm carries ~4.9 KiB of panel machinery the `decay=None` arm does not, so a
//: single constant would either over-promise on one arm or under-use the other.
//:
//: **TWO OF THE THREE TERMS WERE MISSING, AND THE GAP SILENTLY DISABLED THE BOUND ON
//: EVERY `BC = 16` INSTANTIATION -- half the matrix (64 of 128).** At `BC = 16` the
//: footprint is ~13 KiB, so the shared-memory term ALONE returns 7, and
//: `7 x 256 = 1792 > 1536` thread slots. `ptxas` does not CLAMP an out-of-range
//: bound, it DISCARDS it (`Value of threads per SM ... is out of range.
//: .minnctapersm will be ignored`) -- precisely the unconstrained-allocator state
//: correction (d) exists to prevent, reintroduced through an arithmetic gap rather
//: than through an omitted attribute.
//:
//: **AND THE THREAD TERM ALONE IS NOT THE FIX**, which is why the register term is
//: here. Measured per candidate bound at `BC = 16` (§2.21):
//:
//:   bound   | regs | entries that SPILL | worst spill
//:   --------|------|--------------------|---------------
//:   absent  | 57-64| 1 of 6             | 20 st / 56 ld
//:   6       | 40   | **4 of 6**         | **70 st / 284 ld**
//:   5       | 51   | 2 of 6             | 66 st
//:   **4**   | 64   | 1 of 6             | **4 st / 8 ld**  <- the measured optimum
//:
//: Asking for the 6 CTAs the thread slots admit caps the allocator at
//: `65536 / (6 * 256) = 42` registers and it buys those CTAs by SPILLING. The
//: register term returns 4 at `BC = 16, decay = false`, which is both the register
//: file's own answer at that working set AND what the allocator converges to when
//: left alone -- and ncu confirms the residency is 4 CTAs/SM either way, so on that
//: arm the bound converts a silent hazard into a stated invariant at zero measured
//: occupancy cost.
//:
//: **AND ON THE `BC = 16` DECAY ARM THE TERM RETURNS 3, BY MEASUREMENT** -- see
//: `consumer_reg_budget`, which carries both arms' `ncu` numbers. That arm's working
//: set does not fit in 64 registers, so a budget of 64 there does not buy the fourth
//: CTA, it buys 66 st / 212 ld of SPILL; 3 CTAs at 80 registers is 6.5 % FASTER than
//: 4 CTAs at 64 with the spill. The table above is the `decay = false` arm's.
constexpr int ctas_for_residency(size_t smem, int reg_budget) {
  const int by_smem =
      static_cast<int>(kSm.smem_per_sm / (smem + kDriverSmemReservePerCta));
  const int by_threads = kSm.max_threads_per_sm / kConsumerThreads;
  const int by_regs = kSm.regs_per_sm / (kConsumerThreads * reg_budget);
  const int m = by_smem < by_threads ? by_smem : by_threads;
  return m < by_regs ? m : by_regs;
}

//: WHY THIS IS STAMPED INTO `__launch_bounds__` AND NOT LEFT TO THE COMPILER:
//: meeting the shared-memory budget does NOT get you the occupancy. At
//: `__launch_bounds__(256, 1)` ptxas is free to spend up to 255 registers/thread --
//: it spent 96 -- and 96 x 256 caps residency at 2 CTAs even where SMEM admits 3.
//: Told the real target it re-schedules to 66 registers with ZERO spills, so the
//: constraint costs nothing and its absence costs a third of the occupancy. This is
//: coordinator correction (d): the launch bound is a structural PLAN quantity carried
//: through the instantiation matrix.

//: ---------------------------------------------------------------------------
//: MMA PRECISION IS ONE AXIS WITH TWO NAMED POINTS (RULING 1, REVISED 2026-08-01).
//:
//:   PRODUCTION  = bf16 at BOTH sites   (`Plan.mma_precision == "bf16"`)
//:   EXACTNESS   = tf32x3 at BOTH sites (`Plan.mma_precision == "tf32x3"`)
//:
//: The split is about WHAT EACH ARM IS FOR, not about which is numerically nicer:
//: the exactness arm exists so an oracle gate can see a STRUCTURAL bug through the
//: numerics (that needs mantissa), and the production arm exists to ship (that needs
//: speed). Gating the shipped arm on the exact arm's tolerance would be measuring a
//: kernel nobody runs -- the failure `de8567a0` already had to fix once.
//:
//: WHY THE FOLD IS bf16 IN PRODUCTION, having briefly defaulted to tf32x3:
//:  * the accumulation hazard lives in the ACCUMULATOR, and the accumulator is fp32
//:    SMEM state -- unchanged by the operand format;
//:  * the state deviation this costs was MEASURED FLAT IN L (3.447e-03 at L=128,
//:    3.441e-03 at L=256), i.e. per-deposit quantization with no growth pathway;
//:  * `v` is bf16-quantized at the layer boundary already, so the fold's `V` operand
//:    carries that error whatever the MMA does with it;
//:  * flash-attention's `P.V` and FLA's chunk kernels ship exactly this contract;
//:  * and tf32x3's "near-free" justification was measured AT 3.55 % TENSOR PIPE. It
//:    expires at the compute-bound end state this redesign is aiming for, where 3x
//:    the fold's passes is ~1.7x the kernel's total MMA work.
constexpr MmaPrec kFoldPrecDefault = MmaPrec::kBf16;
constexpr MmaPrec kReadPrecDefault = MmaPrec::kBf16;

//: **THE BANK-CONFLICT PROPERTY, AS A COMPILE-TIME PROOF RATHER THAN A COMMENT.**
//:
//: Every MMA operand load has 32 lanes addressing SMEM as an affine function of the
//: fragment's two lane coordinates, `g = lane >> 2` and `r = lane & 3`. The access is
//: conflict-free exactly when those 32 addresses land on 32 DISTINCT banks. This
//: enumerates that condition instead of asserting a modular formula, so each operand
//: proves its own pitch at compile time and a future change to `BT`/`BC`/`DV` fails
//: the build rather than silently costing a 4x wavefront amplification (which is what
//: the unpadded build did measure -- §2.2).
//:
//: `stride_g` / `stride_r` are the address deltas in 4-BYTE BANKS per unit `g` and
//: `r`, which is why a bf16 operand read as 32-bit pairs contributes `pitch / 2`.
constexpr bool banks_distinct(int stride_g, int stride_r) {
  unsigned seen = 0;
  for (int g = 0; g < 8; ++g) {
    for (int r = 0; r < 4; ++r) {
      const int b = ((g * stride_g + r * stride_r) % 32 + 32) % 32;
      if (seen & (1u << b)) return false;
      seen |= 1u << b;
    }
  }
  return true;
}

//: `MmaPrec` (`readout_mma.cuh`) is a PER-SITE template axis, but the two shipped
//: POINTS on it set both sites together -- see the contract above `kFoldPrecDefault`.
//: The per-site parameterization is kept because it is what makes the contract
//: STATEABLE and testable at the call sites, not because a mixed arm ships.
//:
//: The `STATE` deviation the production (bf16) arm carries is DERIVED, not observed:
//: each deposit rounds two bf16 operands, each to within `2^-8` relative, and the
//: accumulation is fp32, so the per-deposit relative error is bounded by `2^-7` and
//: DOES NOT GROW WITH THE NUMBER OF DEPOSITS. That is the bound the suite declares,
//: and the flat-in-L measurement is its confirmation rather than its source.

//: ---------------------------------------------------------------------------
//: FRAGMENT-ORDER OPERAND SYNTHESIS.
//:
//: The generic `mma_tile` below fetches each operand element with its own indexed
//: SMEM read, so a `[16 x 8]` output tile costs ~6 address computations and ~8 loads
//: PER K-STEP, and the A operand is re-fetched by every warp that shares it (all 8 in
//: the readout). The instruction mix says that arithmetic -- not the loads and not the
//: MMA -- is the kernel's leading cost.
//:
//: We SYNTHESISE the read coefficients in SMEM rather than receiving them, so we can
//: store them in the order the MMA's A fragment wants: word `w` of lane `lane` at
//: k-step `ks` at `[ks * 128 + lane * 4 + w]`. A warp then loads a whole fragment with
//: ONE 128-bit `uint4` -- 32 lanes x 16 B = 512 B, four full transactions, no bank
//: conflict and no address arithmetic at all.
//:
//: BOTH precisions are laid out, because `READ_PREC` is a RULING 1 axis and a layout
//: that only served bf16 would silently un-support the tf32x3 readout. They share the
//: storage SHAPE (`[ksteps][32][4]` 32-bit words) and differ only in the compile-time
//: forward map below, so there is one path, parameterized -- not a fast path with a
//: fallback.
//:   bf16 m16n8k16 : a0..a3 = (g, 2r|2r+1), (g+8, 2r|2r+1), (g, 2r+8|..), (g+8, ..)
//:                   -> m = g + 8*(w&1),  k = 16*ks + 8*(w>>1) + 2r   (a PAIR)
//:   tf32 m16n8k8  : a0..a3 = (g, r), (g+8, r), (g, r+4), (g+8, r+4)
//:                   -> m = g + 8*(w&1),  k =  8*ks + 4*(w>>1) +  r   (a single)
//: The `m` map is identical; only the `k` step and the element count differ.

//: 32-bit words in a fragment-ordered A operand contracting over `K`, FOR ONE `m16`
//: ROW BLOCK. A `BT`-token tile holds `NM = BT / 16` of them, laid out block-major at
//: a FIXED stride of `frag_a_words(BC)` -- fixed rather than packed to the compacted
//: `Kpad`, so a block's base address is a compile-time multiple and the MMA's `ks`
//: offset stays exactly what it was before `BT` had a second block.
template <MmaPrec PREC>
constexpr int frag_a_words(int K) {
  return (PREC == MmaPrec::kBf16 ? K / 16 : K / 8) * 128;
}

//: The (m, k) that fragment slot `s` holds, and how many elements it packs.
template <MmaPrec PREC>
__device__ __forceinline__ void frag_a_slot(int s, int& m, int& k) {
  const int ks = s >> 7;
  const int rem = s & 127;
  const int lane = rem >> 2, w = rem & 3;
  m = (lane >> 2) + ((w & 1) << 3);
  const int r = lane & 3;
  k = (PREC == MmaPrec::kBf16) ? (16 * ks + ((w >> 1) << 3) + 2 * r)
                               : (8 * ks + ((w >> 1) << 2) + r);
}

//: The per-CTA shared-memory footprint, `constexpr` so the launch bound below can be
//: DERIVED from it rather than restated next to it -- two numbers that must agree are
//: one number too many.
//: THE `Vw` / `clk` UNION, AND WHY IT IS A `max` RATHER THAN A CONDITIONAL BUFFER.
//:
//: `clk` (the tile's per-leaf inclusive clock prefix, `[BT][BC + 4]`) is DEAD before
//: `Vw` (`[BT][DV + 8]`) is live, so it has always aliased `Vw`'s storage. The alias
//: was licensed by `BC + 4 <= DV + 8`, which is true at `DV = 64` and FALSE at
//: `DV = 32, BC = 64` -- the shape the `d_v` axis introduces.
//:
//: The resolution is to size the region as the MAX OVER ITS LIVE TENANTS, which is
//: the same rule that always governed it, now stated instead of assumed. `clk` is a
//: tenant only under `DECAY`, so every non-decay arm is byte-for-byte unchanged, and
//: so is every `DV = 64` arm (there `Vw` is the larger tenant on its own). No branch
//: dodges a shape: one expression covers all four `(BT, DV)` cells.
template <int BT, int BC, int DV, bool DECAY>
constexpr int vw_union_pitch() {
  return (DECAY && (BC + 4) > (DV + 8)) ? (BC + 4) : (DV + 8);
}

template <int BT, int BC, int DV, int D, MmaPrec READ_PREC, bool DECAY, bool GLOBAL,
          bool COMPACT>
constexpr size_t consumer_smem_bytes() {
  size_t b = static_cast<size_t>(BC) * (DV + 4) * sizeof(float);          // S
  //: `den` exists only under `global`; the `raw` arm has no denominator stream at
  //: all, so allocating its row would be dead storage rather than a rounding.
  b += (GLOBAL ? 3 : 2) * static_cast<size_t>(BT) * sizeof(float);        // sr, sw, den
  b += 2 * static_cast<size_t>(kMaxWords) * sizeof(uint32_t);             // candidates
  b += static_cast<size_t>(D) * BC * sizeof(int);                         // dig
  // Row pitches carry the bank-conflict padding; see the kernel's ledger.
  b += static_cast<size_t>(BC) * (BT + 4) * sizeof(float);                // WT (leaf-major)
  //: `Vw` and its `clk` tenant -- see `vw_union_pitch`.
  b += static_cast<size_t>(BT) * vw_union_pitch<BT, BC, DV, DECAY>() * sizeof(float);
  //: `Rf` carries ONE fragment-ordered A operand PER `m16` ROW BLOCK, `BT / 16` of
  //: them. At `BT = 16` this is the single block it always was.
  b += static_cast<size_t>(BT / 16) * frag_a_words<READ_PREC>(BC) * sizeof(unsigned);  // Rf
  // `Gm` follows `READ_PREC` in both type and pitch (see the kernel).
  b += (READ_PREC == MmaPrec::kBf16)
           ? static_cast<size_t>(BT) * (BT + 8) * sizeof(__nv_bfloat16)
           : static_cast<size_t>(BT) * (BT + 4) * sizeof(float);          // Gm
  //: DECAY's OWN CARVE, AND IT IS THREE BUFFERS RATHER THAN FOUR.
  //:
  //: The naive ledger is `log_keep[BC]` + `P_ref[BC]` + `s0[BC]` + `clk[BT][BC]`,
  //: and at the at-scale cell `clk` alone is 4096 B against 1429 B of headroom --
  //: 36544 B/CTA, i.e. `3 * (36544 + 1024) = 112704 > 102400`, which would cost a
  //: CTA and a third of this redesign's entire occupancy result.
  //:
  //: It does not have to be paid, and the argument is a LIFETIME rather than a
  //: saving. `clk` exists only to build `Rf` and `WT`; it is DEAD before `Vw` is
  //: live (the tile's barrier order below makes that explicit), and `Vw` is the
  //: larger buffer. So `clk` ALIASES `Vw`'s storage and only the three `[BC]`
  //: vectors are new -- 768 B, leaving `3 * (32448 + 1024) = 100416 <= 102400`.
  //: **Decay ships at 3 CTAs/SM, full residency.** The production consumer makes
  //: the identical argument for `clk_panel` aliasing `gram`.
  if (DECAY) b += 3 * static_cast<size_t>(BC) * sizeof(float);            // log_keep, p_ref, s0
  //: COMPACT's CARVE IS `c_idx` + `c_slot` + `c_count` AND NOTHING ELSE -- 520 B at
  //: `BC = 64`, which is what leaves the decay+compact arm inside 3 CTAs/SM with
  //: ~140 B to spare (`3 * (32968 + 1024) = 101976 <= 102400`). READ OFF THE DRIVER
  //: on every build, never assumed: this is the tightest residency margin in the
  //: kernel.
  //:
  //: The DESIGN (§2.18(e)) budgeted a fourth buffer, `supp_leaf_r/w` (`2 * BC` u32),
  //: aliased onto `WT` by a lifetime argument. It does not exist here, and the
  //: reason is that the support words are consumed by ONE WARP in the same
  //: instruction stream that produces them (see the gather below), so they live in
  //: REGISTERS -- which also removes the barrier that would have published them and
  //: the aliasing argument that would have licensed the storage. Same 520 B, one
  //: fewer barrier, no alias to keep sound.
  if (COMPACT) b += (2 * static_cast<size_t>(BC) + 2) * sizeof(int);      // c_idx, c_slot, c_count
  return b;
}

//: `clk`'s row pitch is `BC + 4`, token-major (`clk[t * CKP + i]`), on the same
//: bank-decorrelation argument every other pitch here uses. The alias onto `Vw` is
//: sound because `vw_union_pitch` SIZES the region for whichever tenant is larger;
//: this predicate is what the kernel `static_assert`s to prove that, rather than
//: assuming the sizing and the use agree.
template <int BT, int BC, int DV, bool DECAY>
constexpr bool clk_fits_in_vw() {
  return BC + 4 <= vw_union_pitch<BT, BC, DV, DECAY>();
}


//: One `[16 x 8]` tile against a fragment-ordered A. `d += Af * B`.
template <int KMAX, typename TB, bool B_T, bool B_GATHER, MmaPrec PREC>
__device__ __forceinline__ void mma_tile_fragA(
    const unsigned* __restrict__ Af, const TB* __restrict__ B, int ldb,
    const int* __restrict__ b_row, int K, int n0, int lane,
    float& d0, float& d1, float& d2, float& d3);

__device__ __forceinline__ float mma_src(float x) { return x; }
//: Store a computed fp32 value into a buffer whose element type follows a precision
//: axis. Overloaded rather than `static_cast`, because the torch build defines
//: `__CUDA_NO_BFLOAT16_CONVERSIONS__` -- the narrowing must be the explicit intrinsic.
__device__ __forceinline__ void gm_put(float& d, float x) { d = x; }
__device__ __forceinline__ void gm_put(__nv_bfloat16& d, float x) {
  d = __float2bfloat16(x);
}
__device__ __forceinline__ float mma_src(__nv_bfloat16 x) { return __bfloat162float(x); }

//: `D[16x8] += A[16 x K] * B[K x 8]`, fp32 accumulate, one warp.
//:
//: `TA`/`TB` are the operands' SMEM storage types and are independent of `PREC`:
//: `Wp` and `Vw` are stored fp32 because the fold needs their full mantissa, and the
//: same buffers feed the bf16 intra GEMMs by narrowing at the fragment, which costs
//: nothing extra (the pack was already a cast). Storing them bf16 to save 4 KiB
//: would make the fold's precision unrecoverable, which is the whole of RULING 1.
//:
//: `A_T`/`B_T` select the operand's storage orientation, because the four GEMMs of
//: a tile need three different ones and a transposed COPY of a `[BT x BC]` panel
//: would be pure SMEM traffic:
//:   `A_T == false` : A element `(m, k)` at `A[m * lda + k]`
//:   `A_T == true`  : A element `(m, k)` at `A[k * lda + m]`
//: and likewise for B with `(k, n)` and `ldb`. Fragment index maps are the PTX
//: `mma.sync.aligned` layouts, taken verbatim from `readout_mma.cuh` (`g = lane
//: >> 2` is `groupID`, `r = lane & 3` is `threadID_in_group`) rather than
//: re-derived -- that header's maps are the ones the shipped oracle gates pass on.
//:
//: §12 rule 3's clamp applies on the bf16 arm only, for the same reason
//: `readout_mma.cuh` gives: bf16 narrows the EXPONENT range, so the gap between
//: bf16's max finite and fp32's rounds to `inf`; tf32 carries fp32's 8-bit exponent
//: and a clamp there would be dead code.
template <typename TA, bool A_T, typename TB, bool B_T, MmaPrec PREC>
__device__ __forceinline__ void mma_tile(
    const TA* __restrict__ A, int lda, const TB* __restrict__ B, int ldb,
    int K, int m0, int n0, int lane,
    float& d0, float& d1, float& d2, float& d3) {
  const int g = lane >> 2;
  const int r = lane & 3;
  auto a_at = [&](int m, int k) { return mma_src(A_T ? A[k * lda + m] : A[m * lda + k]); };
  auto b_at = [&](int k, int n) { return mma_src(B_T ? B[n * ldb + k] : B[k * ldb + n]); };
  const int ra = m0 + g;
  const int nb = n0 + g;
  if (PREC == MmaPrec::kTf32x3) {
    // m16n8k8. A: a0..a3 -> (g, r), (g+8, r), (g, r+4), (g+8, r+4).
    //          B: b0, b1 -> (row r, col g), (row r+4, col g).
    for (int k0 = 0; k0 < K; k0 += 8) {
      mma::mma_m16n8k8_tf32x3(
          d0, d1, d2, d3,
          a_at(ra, k0 + r), a_at(ra + 8, k0 + r),
          a_at(ra, k0 + r + 4), a_at(ra + 8, k0 + r + 4),
          b_at(k0 + r, nb), b_at(k0 + r + 4, nb),
          d0, d1, d2, d3);
    }
  } else {
    // m16n8k16. A: a0..a3 pack (g, 2r|2r+1), (g+8, 2r|2r+1),
    //                          (g, 2r+8|2r+9), (g+8, 2r+8|2r+9).
    //           B: b0, b1 pack (row 2r|2r+1, col g), (row 2r+8|2r+9, col g).
    auto cl = clamp_bf16_finite;
    for (int k0 = 0; k0 < K; k0 += 16) {
      const int ca = k0 + r * 2;
      const unsigned a0 = pack_bf16x2(cl(a_at(ra, ca)), cl(a_at(ra, ca + 1)));
      const unsigned a1 = pack_bf16x2(cl(a_at(ra + 8, ca)), cl(a_at(ra + 8, ca + 1)));
      const unsigned a2 = pack_bf16x2(cl(a_at(ra, ca + 8)), cl(a_at(ra, ca + 9)));
      const unsigned a3 = pack_bf16x2(cl(a_at(ra + 8, ca + 8)), cl(a_at(ra + 8, ca + 9)));
      const unsigned b0 = pack_bf16x2(cl(b_at(ca, nb)), cl(b_at(ca + 1, nb)));
      const unsigned b1 = pack_bf16x2(cl(b_at(ca + 8, nb)), cl(b_at(ca + 9, nb)));
      mma_m16n8k16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
    }
  }
}

//: The `[16 x 8]` D/C fragment's element map, shared by every writeback below.
//:   `d0, d1 -> (row g,     col 2r, 2r+1)`
//:   `d2, d3 -> (row g + 8, col 2r, 2r+1)`
//: `d += Af . B` with A FRAGMENT-ORDERED. The A operand costs one `uint4` per
//: k-step and zero address arithmetic; B keeps the generic indexed path, because B is
//: `n`-indexed per warp and each warp wants a different slice of it.
//:
//: `B_GATHER` is COMPACTION's readout indirection: the A operand's `k` is a position
//: in the compacted leaf list while `B` (the state) stays LEAF-indexed and persists
//: across tiles, so row `k` of B is `b_row[k]`. It is a template flag rather than a
//: `b_row != nullptr` test because the non-compacted arm must not carry the load.
//:
//: **THE K-LOOP KEEPS A COMPILE-TIME TRIP COUNT AND AN EARLY EXIT, never a runtime
//: bound.** `KMAX` is the operand's allocated extent (`BC`, or `BT` for the intra
//: term) and `K` is the realized one; a runtime loop bound around an MMA sequence is
//: un-unrollable and costs more than the k-steps it skips (the production kernel
//: records the same measurement for its own ragged-`K` loop). With `COMPACT = false`
//: the caller passes `K = KMAX`, the compare folds away and the emitted code is the
//: pre-compaction loop exactly.
template <int KMAX, typename TB, bool B_T, bool B_GATHER, MmaPrec PREC>
__device__ __forceinline__ void mma_tile_fragA(
    const unsigned* __restrict__ Af, const TB* __restrict__ B, int ldb,
    const int* __restrict__ b_row, int K, int n0, int lane,
    float& d0, float& d1, float& d2, float& d3) {
  const int g = lane >> 2;
  const int r = lane & 3;
  const int nb = n0 + g;
  auto b_at = [&](int k, int n) {
    return mma_src(B_GATHER ? B[b_row[k] * ldb + n]
                            : (B_T ? B[n * ldb + k] : B[k * ldb + n]));
  };
  constexpr int kStep = (PREC == MmaPrec::kBf16) ? 16 : 8;
#pragma unroll
  for (int ks = 0; ks < KMAX / kStep; ++ks) {
    if (ks * kStep >= K) break;
    const uint4 a = *reinterpret_cast<const uint4*>(Af + ks * 128 + lane * 4);
    if (PREC == MmaPrec::kTf32x3) {
      mma::mma_m16n8k8_tf32x3(
          d0, d1, d2, d3,
          __uint_as_float(a.x), __uint_as_float(a.y),
          __uint_as_float(a.z), __uint_as_float(a.w),
          b_at(ks * 8 + r, nb), b_at(ks * 8 + r + 4, nb), d0, d1, d2, d3);
    } else {
      const int ca = ks * 16 + r * 2;
      const unsigned b0 = pack_bf16x2(clamp_bf16_finite(b_at(ca, nb)),
                                      clamp_bf16_finite(b_at(ca + 1, nb)));
      const unsigned b1 = pack_bf16x2(clamp_bf16_finite(b_at(ca + 8, nb)),
                                      clamp_bf16_finite(b_at(ca + 9, nb)));
      mma_m16n8k16(d0, d1, d2, d3, a.x, a.y, a.z, a.w, b0, b1, d0, d1, d2, d3);
    }
  }
}

//: Read element `(m, k)` back OUT of a fragment-ordered A operand -- the inverse of
//: `frag_a_slot`. The mass column's denominator is a `[BT x BC] x [BC x 1]` matvec,
//: one column out of `DV + 1`, and giving it its own `n8` MMA tile would waste 7/8 of
//: a fragment.
//: So it stays a scalar reduction and pays this inverse instead: ~10 integer ops on
//: `BT * BC` elements spread over `BT` threads, once per tile.
template <MmaPrec PREC>
__device__ __forceinline__ float frag_a_get(const unsigned* Af, int m, int k) {
  const int g = m & 7;
  const int w_lo = (m >> 3) & 1;
  if (PREC == MmaPrec::kBf16) {
    const int ks = k >> 4, kk = k & 15;
    const int lane = (g << 2) | ((kk & 7) >> 1);
    const int w = ((kk >= 8) << 1) | w_lo;
    const unsigned word = Af[ks * 128 + lane * 4 + w];
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&word);
    return __bfloat162float(h[kk & 1]);
  }
  const int ks = k >> 3, kk = k & 7;
  const int lane = (g << 2) | (kk & 3);
  const int w = ((kk >= 4) << 1) | w_lo;
  return __uint_as_float(Af[ks * 128 + lane * 4 + w]);
}

#define CONSUMER_D_ROWS(lane) const int _g = (lane) >> 2, _r = (lane) & 3

//: `D` is a COMPILE-TIME axis, on the same argument as `BT`/`BC`: the level count is
//: a STRUCTURAL invariant of the topology, fixed at plan time, with four legal values
//: (spec §1: `1 <= D <= 4`). It is a topology fact, not a threshold, so branching on
//: it is the standing "branch on structure, not thresholds" rule rather than an
//: exception to it. There is NO runtime-`D` path: the host dispatches `D` exactly as
//: it dispatches `BT` and `BC`, and one comprehensive path ships.
//:
//: What it buys beyond the dispatch is that every per-level loop gets a COMPILE-TIME
//: trip count, so `#pragma unroll` turns level-indexed local arrays into registers
//: instead of local memory -- which is precisely the cause §2.4's base-pointer
//: rejection identified.
//: `DECAY` is the LAST structural axis of the tile algorithm, and it is a template
//: parameter for the same reason `D`, `BC` and the precision points are: it is a
//: plan fact, known before launch, and the arm that does not use it must not carry
//: its instructions, its barriers or its 768 B.
//:
//: THE LOWERING IS §7.4.1's DECAY-AWARE PANELS, and it does not add a fifth GEMM
//: or change the shape of any of the four. `keep[t,s] = (1-rate[s])^c[t,s]`, so
//: with `P[t,s] = sum_{u <= t} c[u,s]` the survival from `j` to `t` is
//: `keep^{P[t] - P[j]}`, and THAT EXPONENT SPLITS ACROSS THE TWO OPERANDS OF A
//: GRAM because it is a difference of a per-`t` term and a per-`j` term. With
//: §12 rule 1's per-leaf reference `P_ref = P[BT-1]/2`:
//:
//:   fr[t,s] = exp((P[t,s] - P_ref[s]) * log_keep[s])      folded into `Rf`
//:   fw[j,s] = exp((P_ref[s] - P[j,s]) * log_keep[s])      folded into `WT`
//:   s0[s]   = exp(P_ref[s] * log_keep[s])                 scales the STATE
//:   fr[t,s] * fw[j,s] = survival(j -> t)                  (the identity)
//:
//: so every term of the tile keeps its existing shape and the ONLY new arithmetic
//: is the clock prefix and two exponentials per (token, leaf):
//:
//:   readout    Sum_s (R[t,s] fr[t,s]) * (s0[s] M_entry[s])   `Rf` gains `fr`
//:   intra Gram G = R~ . W~^T, causal                          unchanged
//:   intra y    y += G . Vw                                    unchanged
//:   fold       M_exit = s0 * (s0 M_entry + Sum_j W~[j,s] v[j])  `WT` gains `fw`
//:
//: **ONE WRITE PANEL SERVES BOTH THE GRAM AND THE FOLD**, which is what rule 1's
//: half-mass shift buys and why decay does not need a second `[BT][BC]` buffer the
//: SMEM budget cannot afford. The mass column takes the identical treatment with
//: `v := 1`, because it is the same recurrence on a `v` of ones.
//:
//: A READ-ONLY TILE IS EXEMPT STRUCTURALLY, NOT AS AN OPTIMIZATION: no write in
//: the tile means `c[t,s] = 0` on every leaf the owner holds, hence `keep == 1`
//: exactly and `fr = fw = s0 = 1`. That is §2's write-conditioned quiescence, and
//: it is why every decay block below is guarded by `is_write` -- a block-uniform
//: SCHEDULE bit, not a density test.
//: `COMPACT` is spec §6's sparsity quantum (iii), and in THIS kernel's shape it is a
//: compute cut rather than a trip-count cut: the gathered leaf list shrinks the
//: READOUT's k-extent, the intra Gram's k-extent (through §5's causal sub-list) and
//: the FOLD's m-extent. The old cost model priced compaction against the old
//: kernel's economics (a scalar walk's trip count, a dense-panel-to-compacted COPY,
//: a staging-ring probe, a third barrier); none of those terms exists here, which is
//: why RULING 2 required the axis to be re-derived rather than inherited.
template <int BT, int BC, int DV, int D, MmaPrec FOLD_PREC, MmaPrec READ_PREC,
          bool DECAY, bool GLOBAL, bool COMPACT>
__global__ __launch_bounds__(
    kConsumerThreads,
    ctas_for_residency(consumer_smem_bytes<BT, BC, DV, D, READ_PREC, DECAY, GLOBAL,
                                        COMPACT>(),
                       consumer_reg_budget(BT, BC, DECAY)))
void consumer_forward_kernel(ConsumerParams p) {
  // ---- THE STATE'S ROW PITCH, AND WHY IT IS `DV + 4` ------------------------
  // **THE CONDITIONAL `state_pitch` MANDATE DISCHARGES HERE** (item 6b, 2026-08-01):
  // it required the divergence from the old consumer's `state_pitch<DV>() = DV + 1`
  // to stay documented for as long as two kernels disagreed on the pitch. The old
  // consumer is deleted, `state_pitch` is deleted with it, and this is now the one
  // pitch. The DERIVATION is kept, because it is what makes the value checkable:
  //
  // `DV + 1` was derived from the tf32 MMA B map, where `k = k0 + r` steps by one --
  // at `SP == DV` (a multiple of 32) every lane in a group of four lands in the same
  // bank and the `+1` decorrelates them. The bf16 map steps `k = k0 + 2r`, so the
  // bank index becomes `(2r + g) % 32`, which COLLIDES up to four ways (`2r + g = 6`
  // is hit by `(r,g) = (0,6), (1,4), (2,2), (3,0)`). Since bf16 is the production
  // default (`de8567a0`), `DV + 1` would read the state through the conflicted map.
  //
  // `DV + 4 = 68` makes the bf16 map conflict-free: `2r * 68 = 136r`, so the bank is
  // `(8r + g) % 32`, and `r in [0,4)`, `g in [0,8)` covers all 32 banks exactly once.
  constexpr int SP = DV + 4;
  constexpr int NT_DV = DV / 8;                  // n8 column tiles of a DV-wide output
  //: THE TILE'S OUTPUT IS `NM x NT_DV` WARP TILES, AND THAT PRODUCT -- not `DV` alone
  //: -- IS WHAT THE 8 WARPS PARTITION. `BT = 16, DV = 64` gives `1 x 8`, one tile per
  //: warp, which is the shape every earlier revision hard-coded as `warp < NT_DV`.
  //: The three other cells are `2 x 8`, `1 x 4` and `2 x 4`; the partition below is
  //: the same strided warp loop the FOLD has always used, so no cell is a special
  //: case and none is a fallback.
  constexpr int NM = BT / 16;                    // m16 row blocks of a BT-token tile
  extern __shared__ char smem_raw[];

  // ---- SMEM ledger (the §10i.14 carve, at the at-scale cell's real sizes) ----
  //   S      [BC][SP]  fp32   16640 B   the STATE, resident for the owner's whole run
  //   WT     [BC][BTP] fp32    5120 B   write coefficients, LEAF-MAJOR (RULING 1:
  //                                     fp32, because the FOLD reads it at tf32x3)
  //   Vw     [BT][VWP] fp32    4608 B   the scale-folded value tile (likewise)
  //   sr,sw  [BT]      fp32     128 B   the §3 step-3 row scales
  //   den    [BT]      fp32      64 B
  //   cand   2*kMaxWords u32   1024 B
  //   dig    [4][BC]   int     1024 B   per-owner leaf digits (hoisted, see below)
  //   Rf     frag-order       2048 B   read coefficients, MMA-FRAGMENT ORDERED
  //   Gm     [BT][BT]  bf16     512 B   the causal effective-score block
  //                        = 32704 B = 31.94 KiB -> 3 CTAs/SM x 8 warps = 50 % occ.
  //
  // RULING 1's 4 KiB: `Wp`/`Vw` fp32 instead of bf16 costs 4096 B and the residency
  // is UNCHANGED (3 x 31.94 = 95.8 KiB against the SM's 100 KiB) -- checked against
  // the driver, not assumed, because 3 CTAs/SM is the whole occupancy result.
  // ---- OPERAND ROW PITCHES ARE BANK-CONFLICT PADDING, and they are load-bearing --
  // Measured on the unpadded build at the at-scale cell: 65.6 M shared-load
  // instructions retired 266.1 M wavefronts -- **4.06x amplification, i.e. 75 % of
  // all SMEM load traffic was bank conflict** -- which is most of the 71.87 % memory
  // -pipe wall the stage-1 ncu attribution found.
  //
  // The cause is the MMA fragment map, not the buffers: a fragment's lanes read
  // DOWN a column (`g = lane >> 2` selects the row for the A operand, the column for
  // B), so an unpadded row stride that is a multiple of 32 banks maps every one of
  // those lanes onto the same bank. Concretely, at `Wp[k * BC + m]` with `BC = 64`
  // floats the bank is `m % 32` for EVERY `k`, so the four threads of a group
  // collide four ways; at `Rp[ra * BC + ca]` in bf16 the 32-bit pair address is
  // `g * 32 + ...`, so all EIGHT `g` values collide.
  //
  // One extra element per row decorrelates them, which is exactly the argument
  // `SP = DV + 4` above already records for the published state. The bf16
  // buffers take `+2` elements rather than `+1` because they are read as 32-bit
  // PAIRS, so the stride must move by a whole 4-byte bank. Cost is 256 B per CTA
  // total and the residency is unchanged.
  // Pitches are DERIVED from the fragment maps and PROVEN by `banks_distinct`
  // below, not chosen. `+1` (the first-cut padding) merely decorrelates; it does not
  // make the 32 lanes distinct, and the residual was 2.71 wavefronts per load.
  constexpr int BTP = BT + 4;   // fp32 write coefficients, LEAF-MAJOR (see WT)
  constexpr int VWP = DV + 8;   // fp32 value tile -- `Vw`'s OWN pitch
  //: The allocated pitch of the region `Vw` shares with `clk`; see `vw_union_pitch`.
  //: `Vw` is still addressed at `VWP` -- only the RESERVATION is the max.
  constexpr int VWU = vw_union_pitch<BT, BC, DV, DECAY>();
  // The effective-score block's STORAGE TYPE FOLLOWS `READ_PREC`, and that is a
  // correctness requirement rather than a nicety: `Gm` is an INTERMEDIATE of the `y`
  // path (`G = R.W^T`, then `y += G.Vw`), so storing it bf16 on the tf32x3 arm
  // quantizes a value the exactness arm exists to keep exact. The oracle gate caught
  // exactly that -- state 1.8e-07 (exact) alongside `y` 1.0e-03 (not) -- which
  // localizes the defect to the one object on the `y` path that was still narrow.
  using GmT = typename cuda::std::conditional<READ_PREC == MmaPrec::kBf16,
                                              __nv_bfloat16, float>::type;
  // The pitch differs with the type because the FRAGMENT MAP does: the bf16 map reads
  // 32-bit pairs (`stride_g = GMP/2`), the tf32 map single floats (`stride_g = GMP`).
  // Both are proven by `banks_distinct` below.
  constexpr int GMP = (READ_PREC == MmaPrec::kBf16) ? BT + 8 : BT + 4;
  float* S = reinterpret_cast<float*>(smem_raw);
  float* WT = S + BC * SP;      // [BC][BTP] -- LEAF-major, token-contiguous
  float* Vw = WT + BC * BTP;
  float* sr = Vw + BT * VWU;
  float* sw = sr + BT;
  float* den = sw + BT;
  uint32_t* cand_read = reinterpret_cast<uint32_t*>(den + (GLOBAL ? BT : 0));
  uint32_t* cand_write = cand_read + kMaxWords;
  // §3's leaf-digit decomposition, HOISTED TO THE OWNER. `leaf_digit` is
  // `base + (i / stride) % modulus` with `stride` and `modulus` RUNTIME values, so
  // it is a runtime integer division AND modulo -- and NVIDIA hardware has no
  // integer divide, so each is a ~20-instruction sequence. It was being evaluated
  // once per (tile, token, leaf, level) in both panel builds, and it depends on
  // NONE of the tile, the token or the side: only on (level, owner, leaf). Computed
  // once per CTA it costs `D * BC` evaluations for the owner's whole run instead of
  // `n_tiles * BT * BC * (D - j) * 2`.
  int* dig = reinterpret_cast<int*>(cand_write + kMaxWords);   // [D][BC]
  // The read coefficients are FRAGMENT-ORDERED (see `frag_a_slot`), so they carry no
  // row pitch at all -- the layout is conflict-free by construction rather than by
  // padding, and `RPP` is gone with it.
  //: The A-fragment map covers one `m16` row block, so a `BT`-token tile holds `NM`
  //: of them at a FIXED block stride. `kRfBlock` is that stride; `kRfWords` the total.
  constexpr int kRfBlock = frag_a_words<READ_PREC>(BC);
  constexpr int kRfWords = NM * kRfBlock;
  unsigned* Rf = reinterpret_cast<unsigned*>(dig + D * BC);
  GmT* Gm = reinterpret_cast<GmT*>(Rf + kRfWords);
  // ---- DECAY's three per-leaf vectors, and the clock's ALIAS onto `Vw` --------
  // `log_keep` is per OWNER (computed once at CTA entry); `p_ref` and `s0` are per
  // owner-and-TILE. All three are `[BC]`, which is the 768 B the residency ledger
  // in `consumer_smem_bytes` accounts for.
  //
  // `clk` is the tile's per-leaf INCLUSIVE clock prefix and is a TRANSIENT: it is
  // written before the two panel builds and dead after them, strictly before `Vw`
  // is written. So it lives in `Vw`'s bytes. The barrier that separates the two
  // lifetimes is marked `<- clk is now DEAD` at the site; this `static_assert` is
  // the other half of the argument, because a lifetime claim about a buffer that
  // does not FIT is not a lifetime claim.
  float* log_keep = reinterpret_cast<float*>(Gm + (READ_PREC == MmaPrec::kBf16
                                                       ? BT * (BT + 8)
                                                       : BT * (BT + 4)));
  float* p_ref = log_keep + (DECAY ? BC : 0);
  float* s0 = p_ref + (DECAY ? BC : 0);
  constexpr int CKP = BC + 4;              // clk's row pitch, token-major
  float* clk = Vw;                         // ALIASED -- see the lifetime note above
  // ---- COMPACT's gather state (§3 step 2) ----------------------------------
  //   `c_idx[0..K)`  the gathered leaf list, ASCENDING. `c_idx[q] = 0` for `q >= K`
  //                  keeps the readout's pad rows ADDRESSABLE (their A-operand
  //                  column is zero, which is what makes the product zero).
  //   `c_slot[i]`    the inverse, `-1` when leaf `i` is inert for the whole tile.
  //                  Read ONLY by the decay clock prefix, which is where the
  //                  scatter-fill identity is realized.
  //   `c_count`      `{K, KI}` -- the list length and its §5 causal PREFIX length.
  int* c_idx = reinterpret_cast<int*>(s0 + (DECAY ? BC : 0));
  int* c_slot = c_idx + (COMPACT ? BC : 0);
  int* c_count = c_slot + (COMPACT ? BC : 0);
  static_assert(!DECAY || clk_fits_in_vw<BT, BC, DV, DECAY>(),
                "clk aliases the Vw region and must fit: BC + 4 <= vw_union_pitch");
  // ---- §6.4 PASS 1's OWN STATE, ALIASED ONTO `Rf` --------------------------
  //   `a_state[i]`  the per-leaf survival product `A` over the segment. Identically
  //                 `1` under `decay = None`; under decay it accumulates `s0^2` per
  //                 write tile (see the fold block, and the derivation note there).
  //   `sum_slot[i]` the written-leaf FLAG, rewritten in place by the compaction to
  //                 the leaf's POSITION in the emitted list (`-1` if not in it).
  //   `sum_count`   the list length.
  //
  // The alias costs ZERO shared memory, which is the point: pass 1 forces
  // `is_read = false` on EVERY tile (the candidate walk zeroes `cand_read`), so the
  // read coefficients are neither written nor read in that pass -- while `P = 1`
  // never runs pass 1 at all. A dedicated allocation would have moved the `P = 1`
  // launch's SMEM footprint, and `P = 1` is the primary benchmarked configuration
  // (§6.4): a perturbed baseline is not a baseline.
  float* a_state = reinterpret_cast<float*>(Rf);               // [BC]
  int* sum_slot = reinterpret_cast<int*>(Rf) + BC;             // [BC]
  int* sum_count = reinterpret_cast<int*>(Rf) + 2 * BC;        // [1]
  static_assert(frag_a_words<READ_PREC>(BC) >= 2 * BC + 1,
                "pass 1's summary state must fit inside the read-coefficient buffer");

  // Each operand's pitch PROVES its own conflict-freedom. `stride_g` / `stride_r`
  // are read straight off the fragment maps in `mma_tile`:
  //   A, bf16 pair-read : addr banks = g * (pitch/2) + r          (k steps by 2)
  //   B, bf16           : addr banks = g * 1        + r * 2*pitch
  //   A, tf32           : addr banks = g * pitch    + r * 1       (k steps by 1)
  //   B, tf32           : addr banks = g * 1        + r * pitch
  static_assert(banks_distinct(1, 2 * SP), "S: GEMM1 B operand conflicts");
  static_assert(banks_distinct(1, 2 * BTP), "WT: GEMM2 B operand conflicts");
  static_assert(banks_distinct(READ_PREC == MmaPrec::kBf16 ? GMP / 2 : GMP, 1),
                "Gm: GEMM3 A operand conflicts");
  static_assert(banks_distinct(BTP, 1), "WT: GEMM4 A operand conflicts");
  static_assert(banks_distinct(1, VWP), "Vw: GEMM4 B operand conflicts");
  // THE ONE OPERAND THAT IS NOT CONFLICT-FREE, AND THE TRADE IS EXPLICIT: `Vw` is
  // read by the fold under the tf32 map (`k` steps by 1) and by the intra GEMM under
  // the bf16 map (`k` steps by 2), and NO single pitch satisfies both -- the two
  // conditions are `VWP = 8 (mod 32)` and `VWP = 4 (mod 16)`, which are disjoint.
  // The fold is chosen because it carries ~8x the operand traffic (4 warp-tiles x 2
  // k-steps against the intra term's single k-step), so `Vw` is sized for it and the
  // intra term keeps a 2-way conflict on one of its two operands. Stated here rather
  // than left for someone to rediscover with a profiler.
  static_assert(!banks_distinct(1, 2 * VWP), "Vw/GEMM3: update this note if it changes");

  const int work = blockIdx.x;
  const int bh = p.owners[2 * work];
  const int owner = p.owners[2 * work + 1];
  // ---- §6.4's (owner, segment) work item -----------------------------------
  // `nullptr` is `P = 1`, and it is `P = 1` WITHOUT READING MEMORY: `seg = 0`,
  // `nseg = 1`, `last_seg` true, and the tile range is the whole axis. That is what
  // keeps `P = 1` BIT-IDENTICAL rather than merely equivalent -- the primary
  // benchmarked configuration must not acquire a load it did not have.
  const int seg = (p.work_seg != nullptr) ? p.work_seg[work] : 0;
  const int nseg = (p.work_nseg != nullptr) ? p.work_nseg[work] : 1;
  const int owner_slot0 = (p.work_slot != nullptr) ? p.work_slot[work] : 0;
  const bool summary = (p.pass_mode == kPassSummary);
  const bool last_seg = (seg == nseg - 1);
  const int tile_lo = static_cast<int>((static_cast<int64_t>(seg) * p.n_tiles) / nseg);
  const int tile_hi = static_cast<int>((static_cast<int64_t>(seg + 1) * p.n_tiles) / nseg);
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  constexpr int kWarps = kConsumerThreads / 32;

  // ---- owner decomposition (§3's rectangle), verbatim from the shipped kernel --
  const int j = p.first_spanned_level;
  const int slice = owner % p.n_slices;
  const int rest = owner / p.n_slices;
  int base[D];
#pragma unroll
  for (int l = 0; l < D; ++l) {
    if (l < j) {
      base[l] = (rest / p.owner_stride[l]) % p.level_width[l];
    } else if (l == j) {
      base[l] = slice * p.slice_digits;
    } else {
      base[l] = 0;
    }
  }

  // ---- §7.1 admit bit: one bit per owner, read once at CTA init --------------
  bool admitted = true;
  if (p.admit != nullptr) {
    const int words_per_bh = (p.owners_total + 31) / 32;
    const uint32_t word = p.admit[static_cast<int64_t>(bh) * words_per_bh + (owner >> 5)];
    admitted = ((word >> (owner & 31)) & 1u) != 0u;
  }

  // ---- candidate tiles: the §4 word-AND chain, unchanged ---------------------
  for (int w = tid; w < p.n_words; w += kConsumerThreads) {
    const int64_t union_row = static_cast<int64_t>(bh) * p.n_slices + slice;
    uint32_t r = p.read_union[union_row * p.n_words + w];
    uint32_t wr = p.write_union[union_row * p.n_words + w];
    for (int l = 0; l < j; ++l) {
      const int64_t row =
          static_cast<int64_t>(bh) * p.total_rows + p.level_row_offset[l] + base[l];
      r &= p.read_bits[row * p.n_words + w];
      wr &= p.write_bits[row * p.n_words + w];
    }
    // A denied page's write factor is annihilated -- the same rule as a structurally
    // zero write, applied once per owner rather than per leaf (§7.1).
    if (!admitted) wr = 0u;
    // §6.4: restrict the candidate walk to THIS SEGMENT's tile range.
    const uint32_t range = segment_word_mask(w, tile_lo, tile_hi);
    r &= range;
    wr &= range;
    // Pass 1 walks WRITE tiles only. A read-only tile cannot move the state (under
    // LeafMassDecay its survival is `keep == 1` EXACTLY -- §2's write-conditioned
    // quiescence, the same fact the fold's skip rests on), so it contributes
    // nothing to the summary; and a read emitted here would be counted a SECOND
    // time by pass 3, which re-runs the whole segment from the accumulated
    // boundary.
    cand_read[w] = summary ? 0u : r;
    cand_write[w] = wr;
  }

  // The owner's leaf digits, once. `D * BC <= 4 * 64 = 256 = kConsumerThreads`, so
  // this is at most one iteration per thread and the `idx / BC` it costs is paid
  // once rather than `n_tiles * BT` times.
  for (int idx = tid; idx < D * BC; idx += kConsumerThreads) {
    const int l = idx / BC;
    dig[idx] = leaf_digit(p, l, base[l], idx - l * BC);
  }

  // ---- §12 RULE 2: per-leaf log-retention, ONCE PER OWNER, IN LOG SPACE -------
  // `rate[s] = prod_l dial_l[d_l(s)]` (spec §2) accumulated in fp32, then
  // `log_keep = log1pf(-rate)`. Rule 2 is two prohibitions and both are structural
  // here rather than checked:
  //   * the retention is never formed as a rounded linear-domain `1 - rate` and
  //     then logged -- `log1pf` IS the composition, and it is what recovers the
  //     bits a near-unit `rate` cancels away;
  //   * there is NO EPSILON FLOOR. `delta < 1` is strict in the fp32 dial this
  //     kernel reads (re-checked at the cast boundary by
  //     `schedule_v3.leaf_rate_dials`, because an fp64 dial within ~3e-8 of 1 is
  //     strictly `< 1` in fp64 and still rounds to exactly `1.0f`), so `rate < 1`
  //     and `keep > 0` structurally, and any later underflow of `expf` is ordinary
  //     final-power underflow -- the correct limit, never fed back as a base.
  //
  // It re-evaluates `leaf_digit` rather than reading the `dig` table two lines
  // above, and that is deliberate: `dig` is written by a DIFFERENT thread than the
  // one that would read it here (`idx = l*BC + i` vs `i`), so using it would need
  // a barrier. `D * BC` divisions once per owner is exactly what the digit hoist
  // itself costs; a barrier per CTA to avoid it would be the worse trade.
  if (DECAY) {
    const int h = bh % p.H;
    for (int i = tid; i < BC; i += kConsumerThreads) {
      float rate = 1.0f;
#pragma unroll
      for (int l = 0; l < D; ++l) {
        rate *= p.dials[static_cast<int64_t>(h) * p.total_rows + p.level_row_offset[l]
                        + leaf_digit(p, l, base[l], i)];
      }
      log_keep[i] = log1pf(-rate);
    }
  }

  // ---- the owner's state block, read from HBM ONCE (the one-copy invariant) ---
  // ---- §7.1 page indirection --------------------------------------------------
  // One page is one owner (`page_states := BC`), so an owner's rectangle is `BC`
  // contiguous rows in BOTH layouts and the only thing paging changes is which base
  // those rows start at. `resident == false` is an owner with no committed page: its
  // state is identically zero (nothing has ever written it) and it is never stored
  // back, which is exactly the dense path's arithmetic on a never-written owner --
  // hence paging is BITWISE INVISIBLE rather than approximately equivalent.
  //: `GLOBAL` is the NORMALIZATION arm, and it is a template axis rather than a
  //: runtime `p.den != nullptr` test because it changes the STATE'S SHAPE. Under
  //: `global` the state carries a `+1` mass column (spec §4.2's `y = num / den`
  //: with `g_read := 1`); under `raw` there is no mass column, no denominator
  //: stream and an explicit `g_read`. A runtime branch would make `cols` a runtime
  //: value in every state-traffic address, and would keep the mass reduction's
  //: registers alive on an arm that has no mass.
  constexpr int cols = GLOBAL ? DV + 1 : DV;
  bool resident = true;
  int64_t state_base;
  if (p.page_table != nullptr) {
    const int physical = p.page_table[static_cast<int64_t>(bh) * p.owners_total + owner];
    resident = physical >= 0;
    state_base = static_cast<int64_t>(resident ? physical : 0) * BC;
  } else {
    state_base = static_cast<int64_t>(bh) * p.N + static_cast<int64_t>(owner) * BC;
  }
  // ---- WHERE THIS SEGMENT'S ENTRY STATE COMES FROM (§6.4) -------------------
  //   pass 1                : ZERO. The summary is `(A, B)` with `M_0 := 0`, which
  //                           is the definition of `B`.
  //   pass 3, SPLIT owner   : workspace slot `owner_slot0 + seg`, ALWAYS -- the
  //                           scan's entry stash at `seg = 0`, the accumulated
  //                           boundary otherwise.
  //   pass 3, UNSPLIT owner : the owner's own state, paged or dense, as at `P = 1`.
  //
  // **A SPLIT OWNER'S PASS-3 CTA NEVER READS `p.state`, AND THAT IS A CORRECTNESS
  // REQUIREMENT, NOT A SAVING (fix pass R1).** The owner's LAST segment WRITES
  // those rows below, from a different CTA of the SAME launch, and CUDA orders
  // nothing between CTAs of one launch -- so a segment-0 read of them would be a
  // race whose only protection was the work list happening to be ordered. It was
  // demonstrable: a permutation of the pass-3 work list that the contract PERMITS
  // (every owner's last segment before its first) produced `y` with relative error
  // 1.0. Pass 2 stashes `M_0` into `owner_slot0` in a DIFFERENT LAUNCH, hence
  // ordered, so every pass-3 read is disjoint from every pass-3 write BY
  // CONSTRUCTION and the work list's order is free again. That is what costs one
  // extra slot per split owner (`P`, not `P - 1`).
  const bool from_ws = (!summary) && (nseg > 1);
  const float* ws_entry =
      from_ws ? (p.ws_state + static_cast<int64_t>(owner_slot0 + seg) * BC * cols)
              : nullptr;
  for (int idx = tid; idx < BC * cols; idx += kConsumerThreads) {
    const int i = idx / cols;
    const int c = idx - i * cols;
    S[i * SP + c] = summary
                        ? 0.0f
                        : (from_ws ? ws_entry[i * cols + c]
                                   : (resident ? p.state[(state_base + i) * cols + c]
                                               : 0.0f));
  }
  if (summary) {
    for (int i = tid; i < BC; i += kConsumerThreads) {
      a_state[i] = 1.0f;   // `A`'s identity; `decay = None` never multiplies it
      sum_slot[i] = 0;     // the written-leaf FLAG until the compaction rewrites it
    }
  }
  __syncthreads();

  // The candidate words are already masked to `[tile_lo, tile_hi)`, so restricting
  // the loop is a SPEED change and not a semantic one -- every tile it drops has
  // no bit set and would have hit the `continue` below. At `P = 1` the bounds are
  // `0` and `p.n_tiles`, i.e. the pre-`P` loop exactly.
  for (int tile = tile_lo; tile < tile_hi; ++tile) {
    const bool is_read = ((cand_read[tile >> 5] >> (tile & 31)) & 1u) != 0u;
    const bool is_write = ((cand_write[tile >> 5] >> (tile & 31)) & 1u) != 0u;
    if (!is_read && !is_write) continue;

    const int t0 = tile * BT;
    // ---- 64-BIT ADDRESS ARITHMETIC, PAID ONCE PER TILE INSTEAD OF PER ELEMENT ----
    // The amplitude streams are `[BH, n_tiles, BT, pitch]`, so the row base needs 64
    // bits, but the offset WITHIN a tile does not -- it is `t * pitch + digit`,
    // bounded by `BT * pitch`. Folding the 64-bit part into a base pointer once per
    // (tile, level) leaves the inner loop in 32-bit index arithmetic.
    //
    // THIS EXACT CHANGE WAS MEASURED AND REVERTED ONCE (§2.4), and the reason it now
    // stands is the reason it failed then: with a RUNTIME `p.D` the level loop could
    // not unroll, so these arrays were dynamically indexed and ptxas put them in
    // LOCAL memory -- stack frame 16 -> 80 bytes, exactly the two 4-pointer arrays,
    // and the kernel got SLOWER. `D` is compile-time here, every level loop below is
    // `#pragma unroll`ed to a compile-time trip count, and the arrays are therefore
    // statically indexed and live in registers.
    const int64_t tile_row = (static_cast<int64_t>(bh) * p.n_tiles + tile) * BT;
    const float* amp_r[D];
    const float* amp_w[D];
#pragma unroll
    for (int l = 0; l < D; ++l) {
      amp_r[l] = p.amp_read[l] + tile_row * p.level_pitch_read[l];
      amp_w[l] = p.amp_write[l] + tile_row * p.level_pitch_write[l];
    }

    // ---- COMPACT: §3 step 2's SUPPORT-DRIVEN GATHER -------------------------
    // Three steps, none of them a branch on a density:
    //   (a') the FIXED levels' AND, folded into TOKEN VALIDITY. A fixed level
    //        cannot distinguish two leafs of this owner -- but a zero there kills
    //        the whole token's row scale, so ANDing it into `valid_tokens` shrinks
    //        every leaf's support word identically and is EXACT. Without it a
    //        `branches=[8,16], BC=16` owner's whole width-8 level is invisible to
    //        the compaction.
    //   (b)  the per-leaf AND over the SPANNED levels -> `R_s`, `W_s`.
    //   (c)  the two-pass ballot -> the ASCENDING list `c_idx[0..K)`, with §5's
    //        causal sub-list as its PREFIX `[0, KI)`.
    //
    // **THE SUPPORT WORDS NEVER REACH SHARED MEMORY.** `warp 0` computes them into
    // REGISTERS (`kRounds` per lane, a compile-time count) and consumes them in the
    // ballot two instructions later. The production kernel stages them
    // (`supp_leaf_r/w`) because a 64-thread CTA computes them across the whole
    // block; here one warp does, so the design's `supp_leaf_* aliases WT` lifetime
    // argument is not needed -- and neither is the barrier that would have
    // published them. Same 520 B of SMEM, one fewer barrier per candidate tile.
    //
    // Cost is `(D - j) * BC` support-word loads on ONE warp per tile, against the
    // panel builds' `BT * BC * (D - j)` amplitude loads on all eight -- ~1/16 of the
    // build's traffic even before compaction shrinks the build itself.
    if (COMPACT && warp == 0) {
      const int64_t slab =
          (static_cast<int64_t>(bh) * p.n_tiles + tile) * p.total_rows;
      const int tile_tokens = p.T - t0;
      const uint32_t valid_tokens =
          (tile_tokens >= BT) ? ((BT == 32) ? 0xffffffffu : ((1u << BT) - 1u))
                              : (tile_tokens <= 0 ? 0u : ((1u << tile_tokens) - 1u));
      uint32_t fr = is_read ? valid_tokens : 0u;
      uint32_t fw = is_write ? valid_tokens : 0u;
#pragma unroll
      for (int l = 0; l < D; ++l) {
        if (l >= j) break;
        const int64_t row = slab + p.level_row_offset[l] + base[l];
        if (is_read) fr &= p.read_tok_bits[row];
        if (is_write) fw &= p.write_tok_bits[row];
      }
      // `kRounds` is compile-time (`BC <= 64`), so these are registers, statically
      // indexed -- the same requirement §2.4's rejected base-pointer hoist violated.
      constexpr int kRounds = (BC + 31) / 32;
      uint32_t rwv[kRounds], wwv[kRounds];
#pragma unroll
      for (int b = 0; b < kRounds; ++b) {
        const int i = b * 32 + lane;
        uint32_t rw = (i < BC && is_read) ? fr : 0u;
        uint32_t ww = (i < BC && is_write) ? fw : 0u;
        if (i < BC) {
#pragma unroll
          for (int l = 0; l < D; ++l) {
            if (l < j) continue;   // the fixed levels are already in `fr`/`fw`
            const int64_t row = slab + p.level_row_offset[l] + dig[l * BC + i];
            if (is_read) rw &= p.read_tok_bits[row];
            if (is_write) ww &= p.write_tok_bits[row];
          }
        }
        rwv[b] = rw;
        wwv[b] = ww;
      }
      int k = 0;
#pragma unroll
      for (int pass = 0; pass < 2; ++pass) {
#pragma unroll
        for (int b = 0; b < kRounds; ++b) {
          const int i = b * 32 + lane;
          const bool in_range = i < BC;
          const uint32_t rw = rwv[b];
          const uint32_t ww = wwv[b];
          // MEMBERSHIP IS THE UNION `(R_s | W_s) != 0`, NEVER `W_s` ALONE. The
          // backward iterates `support(W) UNION support(R)` per chunk, because an
          // untouched-but-READ leaf has a zero forward summary and a NONZERO
          // backward covector -- a write-only compaction would break the backward
          // while passing every forward gate on a zero initial state.
          const bool active = (rw | ww) != 0u;
          // §5 (NORMATIVE): a leaf enters the intra gather only if it is read AT OR
          // AFTER its first in-tile write. `ww & -ww` is the first write token;
          // negating it sets every bit at or above it. The test is exact in ONE
          // DIRECTION: over-inclusion is a pure performance loss (a leaf with no
          // in-tile write has a zero write panel; one written only after `t` is
          // masked out by the Gram's `u <= t`), while UNDER-inclusion drops state.
          // That asymmetry is also what licenses the intra GEMM to run over
          // `ceil(KI/kUnit) * kUnit >= KI` k-steps: the over-run reaches leaves in
          // `[KI, K)`, every one of which contributes exactly zero.
          const uint32_t first_write = ww & (0u - ww);
          const bool in_intra = (rw & (0u - first_write)) != 0u;
          const bool take = in_range && ((pass == 0) ? in_intra : (active && !in_intra));
          const unsigned mask = __ballot_sync(0xffffffffu, take);
          const int slot = k + __popc(mask & ((1u << lane) - 1u));
          if (take) {
            c_idx[slot] = i;
            c_slot[i] = slot;
          } else if (pass == 0 && in_range && !active) {
            c_slot[i] = -1;
          }
          k += __popc(mask);
        }
        if (pass == 0 && lane == 0) c_count[1] = k;   // KI
      }
      // The pad: `c_idx[q >= K] = 0` keeps the gathered B-operand load ADDRESSABLE.
      // The A operand's columns `[K, Kpad)` are ZEROED at the panel build, and that
      // -- not the pad row -- is what makes the padded product zero.
      for (int q = k + lane; q < BC; q += 32) c_idx[q] = 0;
      if (lane == 0) c_count[0] = k;
    }
    if (COMPACT) __syncthreads();
    // Compiled out, the gather's answer is its `K == BC` degeneracy, known at
    // compile time: every leaf is active, §5's causal sub-list is the whole
    // ascending list, and `c_idx[k] == k`. Every site below resolves to the
    // identity WITHOUT reading SMEM, which is what makes `COMPACT = false`
    // bit-identical to the pre-compaction kernel rather than merely equal to it --
    // and, in the other direction, what makes `COMPACT = true` bit-identical to
    // `COMPACT = false` on a tile whose gather IS dense.
    const int K = COMPACT ? c_count[0] : BC;
    const int KI = COMPACT ? c_count[1] : BC;
    // `k` per MMA issue is the SHAPE's, compile-time from the format axis --
    // `m16n8k8` on tf32x3, `m16n8k16` on bf16. Never a size, never a density.
    constexpr int kUnit = (READ_PREC == MmaPrec::kTf32x3) ? 8 : 16;
    const int Kpad = ((K + kUnit - 1) / kUnit) * kUnit;      // readout k-extent
    const int KIpad = ((KI + kUnit - 1) / kUnit) * kUnit;    // intra Gram k-extent
    const int KpadM = ((K + 15) / 16) * 16;                  // fold m-extent
    // `KIpad <= Kpad <= KpadM <= BC`, so zeroing `WT`'s rows `[K, KpadM)` covers
    // every pad row any of the three extents can reach.

    // ---- DECAY, step (a): the clock prefix, `P_ref`, `s0` -- INTO `Vw` --------
    // Guarded by `is_write` exactly as the production `panel` lowering is: a
    // read-only tile has `c = 0` on every leaf, so every factor this block would
    // compute is the identity and the plain path is EXACT, not approximate.
    //
    // The clock product runs over ALL `D` levels, including the FIXED ones
    // (`l < j`), where the read and write panels' products start at `j`. That is
    // not an inconsistency: a fixed level's read/write factor is constant along
    // the owner's leafs and hoists into the §3 step-3 ROW SCALE, but the clock is
    // an EXPONENT and a row scale cannot carry it.
    if (DECAY && is_write) {
      const float* amp_c[D];
#pragma unroll
      for (int l = 0; l < D; ++l) {
        amp_c[l] = p.amp_clock[l] + tile_row * p.level_pitch_clock[l];
      }
      // One thread per leaf, SERIAL over the tile's `BT` tokens, because the
      // prefix IS a serial dependence in `t` and `BT = 16`; a parallel scan would
      // cost more barriers than it saves steps. The store `clk[t * CKP + i]` is
      // `i`-contiguous across consecutive threads, so it is conflict-free.
      // COMPACT walks the GATHERED LIST, so a dropped leaf's clock is never formed
      // and its `clk`/`p_ref`/`s0` are never written -- nothing reads them, because
      // every consumer of those three is itself compacted (both panels, the entry
      // sweep, the fold's epilogue, the summary's test and `a_state`).
      //
      // SCATTER-FILL IDENTITY, a THEOREM rather than a convention: a leaf the gather
      // dropped has `c == 0` at every token of this tile, because
      // `build_clock_stream` enforces `stored != 0 => live != 0` per (level, digit)
      // -- so a leaf with no live support on EITHER side has a zero clock product at
      // every token, its true prefix IS zero, and `keep^0 == 1` makes every factor
      // this block would have computed exactly `1`. **That is what makes the
      // compacted entry/exit scaling exact: the leaves the fold never touches are
      // exactly the leaves whose `s0` is exactly 1.**
      //
      // **IT IS A LIST RESTRICTION AND NOT A PREDICATE, AND THE GATE IS WHY.** The
      // first version kept the `[0, BC)` walk and predicated the clock product on
      // `c_slot[i] >= 0`. That is the same arithmetic but NOT the same EXPRESSION:
      // putting the product inside a branch blocks the `acc += a0 * a1` FMA
      // contraction the unpredicated form gets, so the compacted arm computed a
      // different (equally valid) `P[t,i]`. The dense-degeneracy gate caught it as a
      // ~7e-09 state difference that appeared on the `D >= 2` decay cells and NOT on
      // `D = 1` -- exactly the signature, since at `D = 1` there is no product to
      // contract. Written this way the inner expression is textually identical on
      // both arms and the degeneracy is bit-identical.
      for (int q = tid; q < (COMPACT ? K : BC); q += kConsumerThreads) {
        const int i = COMPACT ? c_idx[q] : q;
        float acc = 0.0f;
        for (int t = 0; t < BT; ++t) {
          float cl = 1.0f;
#pragma unroll
          for (int l = 0; l < D; ++l) {
            cl *= amp_c[l][t * p.level_pitch_clock[l] + dig[l * BC + i]];
          }
          // fp32 accumulation is a REQUIREMENT here, not a default (spec §7.4's
          // V3-specific rule): this prefix multiplies `log_keep` and is then
          // exponentiated, so a lost low bit is an EXPONENT error, not a mantissa
          // one.
          acc += cl;
          clk[t * CKP + i] = acc;
        }
        // §12 RULE 1 -- mid-tile clock normalization, and `P_ref = P[BT-1]/2` IS
        // that rule rather than an approximation of it. The largest exponent
        // either panel forms is `max_t |P[t] - P_ref| * |log_keep|`; over a
        // non-decreasing `P` that is minimized at the midpoint of the range, and a
        // per-`(t,j)` reference cannot exist because a Gram contracts `i`, so the
        // reference must be constant along the contracted axis. It is the optimal
        // choice of a free parameter. The half-mass shift also makes the tile's
        // ENTRY and EXIT scales the same object (`P[BT-1] - P_ref == P_ref`),
        // which is what lets ONE write panel serve both the Gram and the fold.
        const float pr = 0.5f * acc;
        p_ref[i] = pr;
        // `pr >= 0` and `log_keep <= 0`, so `s0 <= 1` and no clamp is reachable on
        // this one. Underflow to zero is rule 2's sanctioned underflow (a leaf
        // whose tile-total survival is below fp32 has genuinely lost its entry
        // state), not a floor.
        s0[i] = expf(pr * log_keep[i]);
      }
      __syncthreads();
      // ---- DECAY, step (c): the entry state enters the reference frame -------
      // The carry term is `R~[t,i] * (s0[i] * M_entry[i])`, so `s0` multiplies the
      // STATE, not the read panel. The mass column (`c == DV`) rides in the same
      // sweep because it is the same recurrence with `v := 1`.
      //
      // Row-per-warp rather than a flat `idx / cols` walk: under `global`
      // `cols = DV + 1 = 65` is not a power of two, so the flat form would pay an
      // integer division per element -- ~17 of them per thread per tile, on the
      // hardware whose lack of an integer divide §2.4 already measured at 20
      // instructions a time. The mass column rides this sweep when it exists,
      // because it is the same recurrence with `v := 1`.
      // Under COMPACT the sweep runs over the GATHERED rows only, and that is exact
      // rather than approximate: a dropped leaf has `s0 == 1` EXACTLY (the
      // scatter-fill theorem above), so the rows it skips are the rows whose scale
      // is the identity. It is also what keeps the entry and EXIT scalings on the
      // same row set -- the exit scale rides the fold, which is compacted.
      for (int q = warp; q < (COMPACT ? K : BC); q += kWarps) {
        const int i = COMPACT ? c_idx[q] : q;
        const float s = s0[i];
        for (int c = lane; c < cols; c += 32) S[i * SP + c] *= s;
      }
    }

    // ---- §3 STEP 3's ROW SCALES, HOISTED ABOVE THE PANEL BUILDS (r1 rung 0) --
    // `sr[t]` and `sw[t]` are the tile's per-token row scales: the read/write gain
    // times the FIXED levels' amplitudes at the owner's own digit. They are constant
    // along the leaf axis, which is why they are held out of the coefficient blocks
    // at all (`readout_mma.cuh`'s argument), and -- the point of the hoist -- they
    // depend on NOTHING the tile computes: not the gather, not the clock prefix, not
    // either panel. Only on `(tile, token)` and the owner's `base[]`.
    //
    // WHAT THE HOIST BUYS, stated as the phase count rather than as a barrier count,
    // because the two differ per arm. Before the hoist the tile's pre-GEMM section
    // was THREE phases -- [gather] [panels + row scales] [Vw] -- and the middle
    // barrier existed for one reason: to publish `sw[t]` to the `Vw` writers, which
    // are different threads (`sw`'s loop is `t = tid`, `Vw`'s is `t = tid / DV`).
    // With the row scales computed and published FIRST, `Vw` no longer waits on the
    // panels and the section is TWO phases -- [gather + row scales] [panels + Vw].
    // On the `COMPACT` arms that is a barrier deleted outright; on the plain arm it
    // is barrier-neutral and buys the `Vw` staging's global loads an overlap with
    // the panel builds' amplitude loads instead of a rendezvous between them.
    //
    // THE ONE ARM THAT KEEPS THE MIDDLE BARRIER IS `DECAY`, and it is a LIFETIME
    // constraint rather than a publication one: `clk` ALIASES `Vw`'s bytes (see
    // `vw_union_pitch`), both panel builds READ `clk`, and the `Vw` writers would
    // clobber it. That barrier is now WRITTEN AT ITS REASON below instead of being
    // an incidental side effect of `sw`'s publication -- which is the whole hazard
    // this hoist would otherwise have introduced silently.
    if (is_read) {
      for (int t = tid; t < BT; t += kConsumerThreads) {
        const int token = t0 + t;
        float s = (token < p.T)
                      ? (p.g_read != nullptr
                             ? p.g_read[static_cast<int64_t>(bh) * p.T + token]
                             : 1.0f)
                      : 0.0f;
#pragma unroll
        for (int l = 0; l < D; ++l) {
          if (l >= j) break;
          s *= amp_r[l][t * p.level_pitch_read[l] + base[l]];
        }
        sr[t] = s;
      }
    }
    if (is_write) {
      for (int t = tid; t < BT; t += kConsumerThreads) {
        const int token = t0 + t;
        float s = (token < p.T) ? p.g_write[static_cast<int64_t>(bh) * p.T + token] : 0.0f;
#pragma unroll
        for (int l = 0; l < D; ++l) {
          if (l >= j) break;
          s *= amp_w[l][t * p.level_pitch_write[l] + base[l]];
        }
        sw[t] = s;
      }
    }
    // On `DECAY = false` this rendezvous publishes `sw` to the `Vw` writers, which is
    // what lets the staging run inside the SAME phase as the two panel builds. On
    // `DECAY` it is ABSENT: the `clk` -> `Vw` lifetime barrier below already stands
    // between the panels and the staging and publishes `sw` on its way past, so a
    // second rendezvous here would order nothing.
    if (!DECAY && is_write) __syncthreads();

    // ---- the two coefficient blocks and the row scales (§3 steps 2-3) --------
    // `R[t, i] = scale_read[t] * prod_{l >= j} amp_read[l][t, digit_l(i)]`, with the
    // row scale held OUT of the block: it is constant along both `k` and `n`, so it
    // applies after the accumulation (`readout_mma.cuh`'s own argument) and
    // keeping it out of a bf16 operand keeps its precision.
    if (is_read) {
      // The loop walks FRAGMENT SLOTS, not `(token, leaf)` pairs: `frag_a_slot` is
      // the FORWARD map, so the store is `Rf[s]` with `s` contiguous in `tid` --
      // conflict-free, one 32-bit store, no scatter. Walking `(t, i)` and inverting
      // would have been the same arithmetic backwards and a scattered store.
      auto amp_read_at = [&](int t, int k) {
        // COMPACT: the panel's column `k` is a POSITION in the gathered list, so the
        // leaf it holds is `c_idx[k]` -- and columns `[K, Kpad)` are the RAGGED-K
        // pad, ZEROED here. The A-side zero is what makes the padded product zero;
        // the B side only has to be addressable, which `c_idx[q >= K] = 0` gives it.
        // **This is the §2.6 dividend: we SYNTHESISE the coefficients, so we build
        // them compacted in the first place.** The production kernel has to build a
        // dense `[BT x BC]` panel and COPY it into a compacted view -- 8 KiB and an
        // extra barrier, both of which this shape simply does not have.
        if (COMPACT && k >= K) return 0.0f;
        const int i = COMPACT ? c_idx[k] : k;
        float a = 1.0f;
#pragma unroll
        for (int l = 0; l < D; ++l) {
          if (l < j) continue;   // fixed levels hoist into the row scale (§3 step 3)
          a *= amp_r[l][t * p.level_pitch_read[l] + dig[l * BC + i]];
        }
        // DECAY, step (d): `fr[t,i] = keep_i^{+(P[t,i] - P_ref[i])}`, folded into
        // the READ coefficient. §12 RULE 4 (zero-route annihilation) is satisfied
        // BRANCHLESSLY and structurally by this ordering: `a` is a product of
        // amplitudes, so a structurally dead route is an exact `0.0f`; the factor
        // it multiplies is FINITE by rule 3's clamp; and `0 * finite == 0` exactly.
        // No test on `a`, and no `-inf` anywhere -- the amplitude is never logged,
        // which is the whole reason this lowering takes the direct staged-factor
        // form rather than the log-space one.
        //
        // The clamp is on the read side too, and it is LOAD-BEARING rather than a
        // defensive copy of the write side's: both panels' exponents range over
        // `[-E, +E]` with `E = max_i P_ref[i] * |log_keep[i]|`, so `fr` amplifies
        // to `e^{+E}` at `P[t,i] = 0` (early `t`) by exactly the amount `fw`
        // amplifies at late `j`.
        if (DECAY && is_write) {
          a *= panel_read_factor(clk[t * CKP + i], p_ref[i], log_keep[i]);
        }
        return a;
      };
      // COMPACT builds `frag_a_words(Kpad)` slots instead of `frag_a_words(BC)` --
      // the k-extent cut, realized at the BUILD as well as at the MMA. The bound is
      // a runtime value, which is fine HERE (it is a plain loop, not an MMA
      // sequence); the MMA's own trip count stays compile-time with an early exit.
      const int rf_words = COMPACT ? frag_a_words<READ_PREC>(Kpad) : kRfBlock;
      // The `m16` row blocks are an OUTER, COMPILE-TIME loop, so at `NM == 1` this is
      // the single-block loop verbatim -- no division by a runtime word count, and no
      // per-slot block arithmetic on the cell the baseline was measured at.
#pragma unroll
      for (int mb = 0; mb < NM; ++mb) {
        unsigned* Rb = Rf + mb * kRfBlock;
        for (int sl = tid; sl < rf_words; sl += kConsumerThreads) {
          int tl, k;
          frag_a_slot<READ_PREC>(sl, tl, k);
          const int t = mb * 16 + tl;
          if (READ_PREC == MmaPrec::kBf16) {
            // One slot is an ADJACENT PAIR on `k`, which is why the bf16 map packs two
            // leaves per word and the loop runs half as many iterations.
            Rb[sl] = pack_bf16x2(clamp_bf16_finite(amp_read_at(t, k)),
                                 clamp_bf16_finite(amp_read_at(t, k + 1)));
          } else {
            Rb[sl] = __float_as_uint(amp_read_at(t, k));
          }
        }
      }
    }
    // The write side folds `scale_write[t]` into the VALUE tile instead, because
    // there `t` is the CONTRACTION index and the scale cannot hoist out of the sum.
    // `Vw[t, ch] = scale_write[t] * v[t0 + t, ch]` then serves BOTH the intra term
    // and the boundary fold -- one staged object, two GEMMs.
    if (is_write) {
      // The loop's fastest axis is `t`, NOT `i`, because `WT` is leaf-major: with
      // `i` fastest the consecutive lanes would store at stride `BTP`, which is a
      // 4-way STORE conflict (`gcd(BTP, 32) = 4`). Walking `t` fastest makes the
      // store contiguous. The amplitude reads are the same set either way.
      // COMPACT builds only `KpadM` rows -- `K` real ones plus the fold m-tile's
      // ragged pad, ZEROED. `WT` needs NO indirection at either consumer once it is
      // built this way: GEMM 2 reads it as `[k][n]` with `k` the compacted position,
      // GEMM 4 as `[m][k]` with `m` the compacted position. That is the second half
      // of the §2.6 dividend -- the production kernel needs a separate compacted
      // COPY (`panel_write_c`) precisely because it receives its panel dense.
      const int wt_rows = COMPACT ? KpadM : BC;
      for (int idx = tid; idx < BT * wt_rows; idx += kConsumerThreads) {
        const int q = idx / BT;
        const int t = idx - q * BT;
        if (COMPACT && q >= K) { WT[q * BTP + t] = 0.0f; continue; }
        const int i = COMPACT ? c_idx[q] : q;
        float a = 1.0f;
#pragma unroll
        for (int l = 0; l < D; ++l) {
          if (l < j) continue;
          a *= amp_w[l][t * p.level_pitch_write[l] + dig[l * BC + i]];
        }
        // DECAY, step (d): `fw[t,i] = keep_i^{-(P[t,i] - P_ref[i])}`, folded into
        // the WRITE coefficient. THE SAME `WT` THEN SERVES BOTH GEMM 2 (the causal
        // Gram, where it meets `Rf`'s `fr` and the two reconstruct
        // `survival(j -> t)` exactly) AND GEMM 4 (the fold, where it meets the
        // state's `s0`). One panel, two consumers -- rule 1's half-mass shift is
        // what makes that possible and it is why decay needs no second `[BT][BC]`
        // buffer.
        //
        // This is the DIVISION by cumulative survival that §12 rule 3 exists for,
        // and with bf16 now the production fold (RULING 1 revised) rule 3's second
        // site -- `mma_tile`'s `clamp_bf16_finite` at the narrowing point -- goes
        // live on the WRITE side of the SHIPPED arm, which is rule 3's own
        // motivating case. Both sites are required and both are live: the exponent
        // clamp here bounds the value at `e^88 = 1.65e38` (finite in fp32 AND in
        // bf16, which shares fp32's 8-bit exponent), and the format guard at the
        // MMA catches anything the accumulated product does downstream.
        if (DECAY) a *= panel_write_factor(clk[t * CKP + i], p_ref[i], log_keep[i]);
        WT[q * BTP + t] = a;
      }
    }
    // ---- THE `clk` -> `Vw` LIFETIME BARRIER, AND IT IS DECAY'S ALONE ---------
    // `clk` (the tile's per-leaf inclusive clock prefix) ALIASES `Vw`'s bytes -- see
    // `vw_union_pitch` and the ledger above it. Both panel builds READ `clk` (the
    // `panel_read_factor` / `panel_write_factor` calls), and the `Vw` staging below
    // WRITES those same bytes, so the two lifetimes need a rendezvous between them.
    // Under `DECAY = false` there is no `clk` tenant, `Vw` is disjoint from `WT` and
    // `Rf`, and the staging joins the panel builds' phase: one fewer rendezvous and,
    // more to the point for r1's ranking, the staging's `BT * DV` fp32 GLOBAL loads
    // now issue alongside the panel builds' amplitude loads instead of behind them.
    //
    // `is_write` is CTA-uniform (the same per-tile schedule bit the blocks above are
    // guarded on), and a read-only tile writes neither `clk` nor `Vw`.
    if (DECAY && is_write) __syncthreads();   // <- clk is now DEAD
    if (is_write) {
      for (int idx = tid; idx < BT * DV; idx += kConsumerThreads) {
        const int t = idx / DV;
        const int ch = idx - t * DV;
        const int token = t0 + t;
        const float vv =
            (token < p.T) ? p.v[(static_cast<int64_t>(bh) * p.T + token) * DV + ch] : 0.0f;
        Vw[t * VWP + ch] = sw[t] * vv;
      }
    }
    __syncthreads();

    // ---- GEMM 1: the readout, y_tile = R . S -------------------------------
    // `[BT x BC] x [BC x DV]`. One `m16` row block (BT = 16), `DV/8 = 8` `n8`
    // column blocks -- exactly one per warp, which is why the CTA is 8 warps.
    //
    // The state is fp32 in SMEM and the MMA wants bf16, so the B operand is packed
    // on the fly out of `S`. Kept as its own loop rather than a third orientation
    // flag on `mma_tile_bf16`, because this is the ONE operand whose source dtype
    // differs from its fragment dtype and hiding that in a template would bury it.
    // STATE AND ACCUMULATION STAY fp32 (the authorized bf16 contract); only the
    // MMA's own operand narrows, exactly as `readout_mma.cuh`'s bf16 arm does.
    // THE y-TILE ACCUMULATOR IS REGISTERS, NOT SMEM, and that is structural rather
    // than a saving: warp `w` owns output columns `[8w, 8w+8)` for BOTH the readout
    // and the intra term, so the two GEMMs write the SAME `m16n8` fragment of the
    // SAME warp. No other warp ever reads it, so an SMEM y-tile would be a private
    // object routed through shared memory -- 4 KiB and a round trip for nothing. The
    // fragment survives the `__syncthreads()` between the GEMMs because a barrier
    // orders memory, not registers.
    // `kYTiles` output fragments per warp, a COMPILE-TIME count: warp `w` owns tiles
    // `w, w + kWarps, ...` of the `NM * NT_DV` grid. At `(BT, DV) = (16, 64)` that is
    // exactly one tile per warp and the loop has trip count 1, which is the shape the
    // baseline was measured at. `DV = 32, BT = 16` leaves warps 4-7 with zero tiles --
    // the same strided-loop answer the fold already gives, not a special case.
    constexpr int kYTiles = (NM * NT_DV + kWarps - 1) / kWarps;
    float y[kYTiles][4];
#pragma unroll
    for (int q = 0; q < kYTiles; ++q) { y[q][0] = 0.0f; y[q][1] = 0.0f;
                                        y[q][2] = 0.0f; y[q][3] = 0.0f; }
    // ---- THE WARP PARTITION FORKS ON THE ROW-BLOCK COUNT `NM` --------------
    // `NM == 1` -- the tile is ONE `m16` row block -- means warp `w` owns column
    // tile `w` outright and every m-offset is identically zero. The general strided
    // form below COMPUTES that same partition, but computing it is not free: the
    // collapse is exact at the source level and NOT at the register allocator, and
    // the 256-entry ratified manifest is a property of the allocator. Measured: with
    // the general form alone, 143 of the 256 shipped instantiations move and 24
    // spill worse; with this fork, 0 move. Both arms are LIVE -- `NM == 1` is
    // `BT = 16`, `NM == 2` is `BT = 32` -- and the predicate is a TILE COUNT, the
    // same class as the `NT_DV` guard it restores.
    //
    // COMPACT cuts the K EXTENT on both arms -- `ceil(K/kUnit)` k-steps instead of
    // `BC/kUnit` -- and gathers the B operand's rows through `c_idx`, because the
    // state is LEAF-indexed and persists across tiles while the panel's `k` is a
    // per-tile list position.
    if constexpr (NM == 1) {
      if (is_read && warp < NT_DV) {
        mma_tile_fragA<BC, float, false, COMPACT, READ_PREC>(
            Rf, S, SP, c_idx, Kpad, warp * 8, lane,
            y[0][0], y[0][1], y[0][2], y[0][3]);
      }
    } else if (is_read) {
#pragma unroll
      for (int q = 0; q < kYTiles; ++q) {
        const int wt = warp + q * kWarps;
        if (wt >= NM * NT_DV) break;
        const int mb = wt / NT_DV, nt = wt - mb * NT_DV;
        mma_tile_fragA<BC, float, false, COMPACT, READ_PREC>(
            Rf + mb * kRfBlock, S, SP, c_idx, Kpad, nt * 8, lane,
            y[q][0], y[q][1], y[q][2], y[q][3]);
      }
    }

    // ---- GEMM 2: the causal effective-score block, G = R . W^T --------------
    // `[BT x BC] x [BC x BT]`, then the §5 INCLUSIVE causal mask (`u <= t`): a
    // deposit at `t` is visible to the read at `t`, which is what makes the tile's
    // read-after-write inclusive.
    const bool mixed = is_read && is_write;
    // `G` is `[BT x BT]`: `NM` m-blocks x `BT/8` n-blocks of warp tiles, strided over
    // the warps exactly as the readout and the fold are. At `BT = 16` the grid is
    // `1 x 2` and warps 0-1 take one tile each -- the old `warp < BT / 8`.
    // Forked on `NM` for the same measured reason as the readout above.
    if constexpr (NM == 1) {
    if (mixed && warp < BT / 8) {
      CONSUMER_D_ROWS(lane);
      float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
      mma_tile_fragA<BC, float, false, false, READ_PREC>(
          Rf, WT, BTP, nullptr, KIpad, warp * 8, lane, d0, d1, d2, d3);
      const int n0 = warp * 8;
      const int c0 = n0 + 2 * _r, c1 = n0 + 2 * _r + 1;
      gm_put(Gm[_g * GMP + c0], (c0 <= _g) ? d0 : 0.0f);
      gm_put(Gm[_g * GMP + c1], (c1 <= _g) ? d1 : 0.0f);
      gm_put(Gm[(_g + 8) * GMP + c0], (c0 <= _g + 8) ? d2 : 0.0f);
      gm_put(Gm[(_g + 8) * GMP + c1], (c1 <= _g + 8) ? d3 : 0.0f);
    }
    } else
    for (int gt = warp; mixed && gt < NM * (BT / 8); gt += kWarps) {
      const int gmb = gt / (BT / 8), gnt = gt - gmb * (BT / 8);
      CONSUMER_D_ROWS(lane);
      float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
      // `B_T = false`: with `WT` leaf-major, `G`'s B operand element `(k = leaf,
      // n = token)` IS `WT[leaf * BTP + token]`, i.e. already `[k][n]`. The
      // transpose that `W^T` names is now in the STORAGE, so the fragment does not
      // perform one -- which is the whole point of building `WT` that way.
      // COMPACT cuts this k-extent to §5's CAUSAL PREFIX, `ceil(KI/kUnit)` k-steps.
      // No gather: `WT`'s rows are already the compacted positions. The pad rows
      // `[KI, KIpad)` are leaves in `[KI, K)` (real, but non-causal, so exactly zero
      // under the `u <= t` mask) or the zeroed rows `[K, KpadM)` -- over-inclusion,
      // which is the direction §5's test is exact in.
      mma_tile_fragA<BC, float, false, false, READ_PREC>(
          Rf + gmb * kRfBlock, WT, BTP, nullptr, KIpad, gnt * 8, lane, d0, d1, d2, d3);
      const int n0 = gnt * 8;
      // The causal test is on the TILE-GLOBAL token indices, so the m-block's row
      // offset enters both sides of `u <= t`. At `NM == 1` the offset is zero and
      // this is the mask it always was.
      const int m0 = gmb * 16;
      const int r0 = m0 + _g, r1 = m0 + _g + 8;
      const int c0 = n0 + 2 * _r, c1 = n0 + 2 * _r + 1;
      gm_put(Gm[r0 * GMP + c0], (c0 <= r0) ? d0 : 0.0f);
      gm_put(Gm[r0 * GMP + c1], (c1 <= r0) ? d1 : 0.0f);
      gm_put(Gm[r1 * GMP + c0], (c0 <= r1) ? d2 : 0.0f);
      gm_put(Gm[r1 * GMP + c1], (c1 <= r1) ? d3 : 0.0f);
    }
    __syncthreads();

    // ---- GEMM 3: the intra term, y_tile += G . Vw ---------------------------
    // The SAME tile decomposition the readout used -- including the same `NM` fork
    // -- so the two GEMMs accumulate into the same warp's same fragment, which is
    // what keeps the y-tile in registers rather than in shared memory.
    if constexpr (NM == 1) {
      if (mixed && warp < NT_DV) {
        mma_tile<GmT, false, float, false, READ_PREC>(
            Gm, GMP, Vw, VWP, BT, 0, warp * 8, lane,
            y[0][0], y[0][1], y[0][2], y[0][3]);
      }
    } else if (mixed) {
#pragma unroll
      for (int q = 0; q < kYTiles; ++q) {
        const int wt = warp + q * kWarps;
        if (wt >= NM * NT_DV) break;
        const int mb = wt / NT_DV, nt = wt - mb * NT_DV;
        mma_tile<GmT, false, float, false, READ_PREC>(
            Gm, GMP, Vw, VWP, BT, mb * 16, nt * 8, lane,
            y[q][0], y[q][1], y[q][2], y[q][3]);
      }
    }

    // ---- the mass column, by scalar reduction -------------------------------
    // `den[t] = scale_read[t] * ( sum_i R[t,i] * S[i][DV] + sum_{u <= t} G[t,u] *
    // scale_write[u] )` -- the same split the value term takes, with `v := 1`. It is
    // ONE column out of `DV + 1`, so it is ~1.5 % of the tile's arithmetic; making
    // it an MMA `n` would waste 7/8 of an `n8` fragment -- the same argument
    // `frag_a_get` records.
    if (GLOBAL && is_read) {
      for (int t = tid; t < BT; t += kConsumerThreads) {
        float acc = 0.0f;
        // COMPACT: `K` terms instead of `BC`, gathering the mass column through
        // `c_idx` exactly as the readout's B operand does.
        // `Rf`'s block for token `t`; `NM == 1` folds the arithmetic away entirely.
        const unsigned* Rb = Rf + (NM == 1 ? 0 : (t >> 4) * kRfBlock);
        const int tl = (NM == 1) ? t : (t & 15);
        for (int k = 0; k < K; ++k) {
          acc += frag_a_get<READ_PREC>(Rb, tl, k) * S[(COMPACT ? c_idx[k] : k) * SP + DV];
        }
        if (mixed) {
          for (int u = 0; u <= t; ++u) acc += mma_src(Gm[t * GMP + u]) * sw[u];
        }
        den[t] = acc;
      }
    }
    // ---- NO BARRIER HERE, AND THE DISCHARGE IS AN INTERVAL ARGUMENT (r1 rung 0) --
    // A `__syncthreads()` stood between the mass reduction and the emission below.
    // It is REMOVED, and what licenses that is the SMEM read/write sets of the
    // interval it separated, enumerated rather than eyeballed:
    //   * the emission READS `y[q][*]` (REGISTERS, so no barrier can order them and
    //     none is needed -- they are the same warp's own accumulators), `sr[t]`
    //     (published three barriers back, at the end of the panel phase) and `den[t]`;
    //   * the emission WRITES only GLOBAL memory, through `atomicAdd`;
    //   * `den[t]` is written at `den[t] = acc` above and read at the denominator
    //     emit below BY THE SAME THREAD -- both loops are `t = tid; t < BT;
    //     t += kConsumerThreads`, so the dependence is program order, not a
    //     cross-thread one.
    // So the interval contains no SMEM object written by one thread and read by
    // another, in EITHER direction: nothing the emission writes is read by a lagging
    // thread's mass reduction (it writes no SMEM at all), and nothing it reads is
    // written by one (`sr` and `den` are both already-ordered). The next barrier
    // (before the fold) still separates the readout's consumption of `S` from the
    // fold's advance of it, which is the tile's real read-then-write ordering.

    // ---- split-K emission of this tile's partial (existing licensed semantics) --
    if (is_read) {
      // Straight out of the D fragment: `d0,d1 -> (row g, col 2r, 2r+1)` and
      // `d2,d3 -> (row g+8, ...)`, with the §3 step-3 row scale applied here because
      // it is constant along both `k` and `n`.
      // The emits stay WRITTEN OUT per fragment slot rather than looped over an array
      // of pointers to the accumulators: taking the address of a register-resident
      // value is how an accumulator ends up in local memory, which is the one thing
      // this fragment must not do. `y[q][*]` is indexed by an unrolled compile-time
      // `q`, which is a register selection rather than an address.
      // Forked on `NM` for the same measured reason as the three GEMMs.
      if constexpr (NM == 1) {
        if (warp < NT_DV) {
          CONSUMER_D_ROWS(lane);
          const int n0 = warp * 8;
          // The output side DOES keep its hoist: one base pointer, no array, so no
          // dynamic indexing and no local memory.
          float* num_t0 = p.num + (static_cast<int64_t>(bh) * p.T + t0) * DV;
          auto emit = [&](int t, int ch, float val) {
            if (t0 + t >= p.T) return;
            const float out = sr[t] * val;
            if (out != 0.0f) atomicAdd(&num_t0[t * DV + ch], out);
          };
          const int c0 = n0 + 2 * _r, c1 = c0 + 1;
          emit(_g, c0, y[0][0]);
          emit(_g, c1, y[0][1]);
          emit(_g + 8, c0, y[0][2]);
          emit(_g + 8, c1, y[0][3]);
        }
      } else {
        CONSUMER_D_ROWS(lane);
        float* num_t0 = p.num + (static_cast<int64_t>(bh) * p.T + t0) * DV;
        auto emit = [&](int t, int ch, float val) {
          if (t0 + t >= p.T) return;
          const float out = sr[t] * val;
          if (out != 0.0f) atomicAdd(&num_t0[t * DV + ch], out);
        };
#pragma unroll
        for (int q = 0; q < kYTiles; ++q) {
          const int wt = warp + q * kWarps;
          if (wt >= NM * NT_DV) break;
          const int mb = wt / NT_DV, nt = wt - mb * NT_DV;
          const int n0 = nt * 8, m0 = mb * 16;
          const int c0 = n0 + 2 * _r, c1 = c0 + 1;
          emit(m0 + _g, c0, y[q][0]);
          emit(m0 + _g, c1, y[q][1]);
          emit(m0 + _g + 8, c0, y[q][2]);
          emit(m0 + _g + 8, c1, y[q][3]);
        }
      }
      if (GLOBAL) {
        for (int t = tid; t < BT; t += kConsumerThreads) {
          const int token = t0 + t;
          if (token >= p.T) continue;
          const float d = sr[t] * den[t];
          if (d != 0.0f) atomicAdd(&p.den[static_cast<int64_t>(bh) * p.T + token], d);
        }
      }
    }
    // The readout above consumed the state at TILE ENTRY; the fold below advances
    // it. The barrier between them is the whole read-then-write ordering of the
    // tile, and it is one barrier, not a per-visit chain.
    __syncthreads();

    // ---- GEMM 4: the boundary fold, S += W^T . Vw ---------------------------
    // `W^T` is not a transpose at the fragment any more -- `WT` IS the transpose,
    // built that way. We SYNTHESISE the write coefficients rather than receive them
    // in someone else's layout, so storing them leaf-major costs exactly nothing at
    // the build (one index expression) and buys the fold a conflict-free,
    // token-contiguous A operand. The same buffer serves GEMM 2 as a `[k][n]` B
    // operand with no orientation flag at all.
    // `[BC x BT] x [BT x DV]` -- §10i.8 (2), the move that REPLACES the shipped
    // kernel's per-visit scattered `d_v`-wide scalar accumulates. `BC/16 = 4` row
    // blocks x `DV/8 = 8` column blocks = 32 warp tiles over 8 warps.
    if (is_write) {
      // COMPACT cuts the M EXTENT: `ceil(K/16)` warp m-tiles instead of `BC/16`.
      // This is the third of the three extents compaction shrinks, and on the
      // flagship cell's density (`mean_K_candidate = 0.9665` of `BC = 64`) it is a
      // 4 -> 1 cut on the kernel's most expensive GEMM.
      for (int wt = warp; wt < (COMPACT ? KpadM / 16 : BC / 16) * NT_DV; wt += kWarps) {
        CONSUMER_D_ROWS(lane);
        const int mt = wt / NT_DV;
        const int nt = wt - mt * NT_DV;
        float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        mma_tile<float, false, float, false, FOLD_PREC>(
            WT, BTP, Vw, VWP, BT, mt * 16, nt * 8, lane, d0, d1, d2, d3);
        const int r0 = mt * 16 + _g, n0 = nt * 8 + 2 * _r;
        // ---- DECAY's EXIT SCALE, RIDING THE FOLD'S EXISTING READ-MODIFY-WRITE --
        // `M_exit = s0 * (s0 * M_entry + Sum_j W~[j] v[j])`, and the inner `s0`
        // was applied at entry, so what is left here is `S <- s0 * (S + d)`. The
        // fold ALREADY reads `S`, adds to it and writes it back; folding the scale
        // into that epilogue costs ONE EXTRA MULTIPLY on a value already in a
        // register and ZERO extra shared-memory traffic.
        //
        // It replaces a separate `[BC x cols]` sweep, and the counters are why:
        // decay's two `s0` sweeps were 2 x 4160 element read-modify-writes per
        // WRITE TILE, and at the at-scale cell they accounted for the bulk of
        // decay's +87.7 M shared-store instructions (+219.6 % over the
        // `decay = None` arm) -- 16x the clock prefix's traffic, which is the term
        // I had wrongly blamed. The warp tiles cover `BC x DV` exactly once, so
        // every element is scaled exactly once and none is missed.
        // ---- COMPACT's OUTPUT SCATTER, and its RAGGED-M GUARD -----------------
        // The fold's `m` is a compacted position while the state is leaf-indexed
        // and persists across tiles, so the store scatters through `c_idx`. The
        // guard is NOT redundant with the zeroed pad rows: a pad row's DELTA is
        // zero, but `c_idx[q >= K] = 0` aims it at LEAF 0, and under decay the
        // epilogue's `s0` multiply would then be applied to leaf 0 a second time.
        // Predicating the STORE is what keeps the pad inert; zeroing the A operand
        // alone is not enough.
        const int i0 = COMPACT ? c_idx[r0] : r0;
        const int i1 = COMPACT ? c_idx[r0 + 8] : r0 + 8;
        const float e0 = DECAY ? s0[i0] : 1.0f;
        const float e1 = DECAY ? s0[i1] : 1.0f;
        if (!COMPACT || r0 < K) {
          S[i0 * SP + n0] = e0 * (S[i0 * SP + n0] + d0);
          S[i0 * SP + n0 + 1] = e0 * (S[i0 * SP + n0 + 1] + d1);
        }
        if (!COMPACT || r0 + 8 < K) {
          S[i1 * SP + n0] = e1 * (S[i1 * SP + n0] + d2);
          S[i1 * SP + n0 + 1] = e1 * (S[i1 * SP + n0 + 1] + d3);
        }
      }
      // The mass state's own recurrence, `mass[i] += sum_t W[t, i] * scale_write[t]`
      // -- the `v := 1` column of the same fold. It exists only under `global`.
      if (GLOBAL) {
        // COMPACT: the reduction runs over the gathered rows and the row it writes
        // is scattered through `c_idx` -- the mass column IS the fold's `v := 1`
        // column, so it takes the same compaction and the same scatter.
        for (int q = tid; q < (COMPACT ? K : BC); q += kConsumerThreads) {
          const int i = COMPACT ? c_idx[q] : q;
          float acc = 0.0f;
          for (int t = 0; t < BT; ++t) acc += WT[q * BTP + t] * sw[t];
          // The mass column takes the identical epilogue -- it is the same
          // recurrence with `v := 1`, so it takes the same exit scale in the same
          // read-modify-write, and the two together cover `BC x (DV + 1)` exactly.
          S[i * SP + DV] = (DECAY ? s0[i] : 1.0f) * (S[i * SP + DV] + acc);
        }
      }
      if (summary) {
        // ---- §6.4 pass 1: the written-leaf list, accumulated across the segment
        // A leaf enters the list iff it can LEAVE `(A, B) = (1, 0)`, and the test
        // is deliberately the WEAKER of the two available ones. OVER-inclusion is
        // free -- the leaf is emitted at its exact `(1, 0)` and the scan's fold
        // returns the incoming value unchanged -- while UNDER-inclusion silently
        // drops state. So the test is over-inclusion-safe by construction, which
        // is the same direction every `!= 0.0f` skip in this kernel takes.
        //
        // `WT` is read here while it still holds THIS tile's coefficients, which
        // is why the test lives in the fold's block rather than in the epilogue.
        // COMPACT: a leaf OUTSIDE the gathered list cannot enter the summary, and
        // that is exact on both terms of the test -- it has no live write in this
        // tile (so its write coefficients are zero) and its `s0` is exactly `1` (the
        // scatter-fill theorem), so it cannot leave `(A, B) = (1, 0)`.
        for (int q = tid; q < (COMPACT ? K : BC); q += kConsumerThreads) {
          const int i = COMPACT ? c_idx[q] : q;
          bool any = false;
          for (int t = 0; t < BT; ++t) any |= (WT[q * BTP + t] != 0.0f);
          // Under decay `WT` carries `W * fw`, and "a nonzero write forces a
          // nonzero coefficient" is an argument about `W`, not about `W * fw`:
          // `fw` can UNDERFLOW to zero on the early half of a tile at an extreme
          // survival range. That range is outside what `panel_exponent_bound`
          // admits, but the summary's inclusion test should not be the thing that
          // depends on the admission check. `s0 != 1` is the DIRECT statement that
          // this leaf's clock moved in this tile, which is exactly the condition
          // under which `A` leaves `1`.
          if (DECAY && s0[i] != 1.0f) any = true;
          if (any) sum_slot[i] = 1;
        }
        if (DECAY) {
          // ---- `A`, the segment's per-leaf survival product (§2.13) -----------
          // One write tile's map is `M_out = s0^2 * M_in + s0 * (W~^T Vw)`, so the
          // tile's `A` factor is `s0 * s0` and the segment's is their product.
          // Rule 1's half-mass reference is what makes this expressible as `s0`
          // squared rather than as a second exponential to form and store:
          // `s0^2 = exp(2 * P_ref * log_keep) = keep^(tile's TOTAL clock)`, i.e.
          // the genuine survival across the tile. `A` therefore needs no separate
          // reduction and no separate buffer.
          //
          // It is a `[BC]` loop, not a `[BC x cols]` one -- `A` is a per-leaf
          // SCALAR, which is the whole reason a segment summary is cheap.
          // COMPACT: the gathered rows only. A dropped leaf's `s0` is exactly `1`
          // (the scatter-fill theorem), so the factor it would contribute is the
          // identity -- and under COMPACT its `s0` is not even written, which is why
          // this loop MUST be restricted rather than merely being free to be.
          for (int q = tid; q < (COMPACT ? K : BC); q += kConsumerThreads) {
            const int i = COMPACT ? c_idx[q] : q;
            a_state[i] *= s0[i] * s0[i];
          }
        }
      }
    }
    // DECAY's exit scale is NOT a separate pass: it rides the fold's epilogue
    // above (see the note there). An earlier build did make it a separate
    // `[BC x cols]` sweep with its own barrier pair -- the fold writes `S` under
    // the `(m16, n8)` warp-tile map while a row-per-warp sweep reads it back under
    // a different one, so the two could not share a barrier. Folding the scale
    // into the fold's own read-modify-write removes the sweep, its traffic and its
    // barrier together.
    __syncthreads();
  }

  if (summary) {
    // ---- §6.4 PASS 1's PUBLICATION: the SPARSE written-leaf summary ----------
    // §6.4 makes the summary NORMATIVELY sparse: its traffic must scale with
    // REALIZED writes, not with `BC`.
    //
    // Slot `owner_slot0` is the scan's entry-state stash, so segment `seg`'s
    // summary lands one past it. Pass 3's segment `q` then reads `owner_slot0 + q`
    // UNIFORMLY: the stash at `q = 0`, and the boundary the scan accumulated over
    // this summary at `q = seg + 1`.
    const int64_t slot = static_cast<int64_t>(owner_slot0 + seg + 1);
    if (warp == 0) {
      // Warp 0 compacts the flags to an ASCENDING leaf list -- the same ballot
      // compaction §3 step 2's gather uses, for the same reason (an ascending list
      // is what makes the dense degeneracy the identity). `sum_slot[i]` is
      // rewritten in place from "is written" to "position in the list"; only lane
      // `i` ever touches entry `i`, and it reads before it writes, so the rewrite
      // is race-free.
      int k = 0;
      for (int b0 = 0; b0 < BC; b0 += 32) {
        const int i = b0 + lane;
        const bool in_range = i < BC;
        const bool take = in_range && (sum_slot[i] != 0);
        const unsigned below = (1u << lane) - 1u;
        const unsigned mask = __ballot_sync(0xffffffffu, take);
        const int q = k + __popc(mask & below);
        if (in_range) sum_slot[i] = take ? q : -1;
        if (take) p.ws_leaf[slot * BC + q] = i;
        k += __popc(mask);
      }
      if (lane == 0) sum_count[0] = k;
    }
    __syncthreads();
    if (tid == 0) p.ws_count[slot] = sum_count[0];
    for (int i = tid; i < BC; i += kConsumerThreads) {
      const int q = sum_slot[i];
      // `A` is identically `1` under `decay = None` (nothing multiplies
      // `a_state` there), so the scan's multiply is EXACT on that arm rather than
      // skipped -- one code path, no arm-specific fold.
      if (q >= 0) p.ws_scale[slot * BC + q] = a_state[i];
    }
    // `B`. **THE RESTRUCTURE SIMPLIFIES THIS, and the simplification is taken.**
    // The production kernel must emit `B` through a compile-time `#pragma unroll`
    // over leaves, because its state is REGISTER-resident and indexing `M[i]` by a
    // runtime list position would spill the whole state to local memory and
    // destroy §5's register invariant. This kernel's state is SMEM-resident and
    // ADDRESSABLE, so a runtime `i` is just an index.
    for (int idx = tid; idx < BC * cols; idx += kConsumerThreads) {
      const int i = idx / cols;
      const int c = idx - i * cols;
      const int q = sum_slot[i];
      if (q >= 0) p.ws_state[(slot * BC + q) * cols + c] = S[i * SP + c];
    }
    return;
  }
  // ---- the owner's state block, written to HBM ONCE --------------------------
  // A non-resident owner is never stored back: it has no page, and its state was
  // read as zeros above. `last_seg` is §6.4's publication rule and is `true` at
  // `P = 1`: the final state is published by pass 3's LAST segment, running
  // serially from the accumulated boundary, NOT by the scan. That is a deliberate
  // numeric choice -- it leaves the last segment's own arithmetic in serial order
  // and confines the fold's re-association to the `P - 1` boundaries.
  if (!last_seg || !resident) return;
  for (int idx = tid; idx < BC * cols; idx += kConsumerThreads) {
    const int i = idx / cols;
    const int c = idx - i * cols;
    p.state[(state_base + i) * cols + c] = S[i * SP + c];
  }
}

#undef CONSUMER_D_ROWS

}  // namespace consumer
}  // namespace rola
