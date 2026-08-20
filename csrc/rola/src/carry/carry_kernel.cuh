// csrc/rola/src/carry/carry_kernel.cuh -- K31 RUNG R1: THE BRING-UP CARRY KERNEL.
//
// A NEW KERNEL FAMILY, BESIDE THE SHIPPED ONE.  Nothing here is reachable from
// `rola::chunk`'s dispatch and nothing here edits it; the shipped arms remain the
// baseline until this family beats them cell for cell (study section 6).
//
// WHAT IT COMPUTES.  The INTER (carried-state) term only:
//
//     O^T[:, t] += S^T[:, box] . R~^T[box, t]      readout, vs state-at-window-start
//     S^T[:, box] += V~^T[:, t] . W~[t, box]       fold, of the window's own writes
//
// where `R~`/`W~` are the read/write leaf coefficients restricted to the owner's
// box, `V~` is the value row with the ones column appended (the mass channel), and
// the window grid is GLOBAL and FIXED at `W` tokens -- identical for every owner,
// which is what removes the shipped kernel's boundary drift (study section 2.1).
// The within-window (INTRA) pairs are NOT computed here; rung R2 builds that kernel
// and R1's conformance adds the contribution on the host from the fp64 oracle.
//
// SYMBOLS, each at first use: `D` levels, `B` digits per level, `K` the uniform
// sub-box span, `M` the capacity multiplier, `BC` the leaves one owner holds,
// `W` the window length in tokens, `C` the private batch's token tile, `d_v = 64`
// value channels.  `k_tok` (a token's nonzero digits per level) appears NOWHERE in
// this file, and that is the point: the schedule is computed, never branched on.
//
// THE ONE ORDERING LAW is window-grain read-before-write on the owner's own slots.
// A batch therefore NEVER crosses a window line: a batch that did would owe the
// cross-window pairs inside it to neither the readout (which is against
// state-at-batch-start) nor the intra (which is within-window).  See
// docs/internals/carry/carry_kernel.md.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <type_traits>

#include "box.cuh"
#include "facts/arch_caps.cuh"
#include "common/design.cuh"
#include "common/ops.cuh"

namespace rola::carry {

using namespace denseref::ops;
namespace design = denseref::design;

//: THE BUILD'S IDENTITY, device-side.  A green battery against a stale binary is
//: this project's recurring failure; a stamp read OFF THE DEVICE is the only check
//: that catches it, so every gate in `tests/k31/` asserts this first.
static constexpr int kCarryBuildMagic = 31010013;

static constexpr int kDv = 64;         // d_v
static constexpr int kCols = kDv + 1;  // the mass column rides the state
static constexpr int kPadE = 8;        // bf16 row padding, elements (16 B)
static constexpr int kPadF = 4;        // fp32 row padding, elements (16 B)
static constexpr int kWarps = 4;
static constexpr int kThreads = kWarps * 32;
static constexpr int kAtomLeaves = kMmaK;  // 16, the page granule

//: A COMPILE-TIME LOOP.  Every per-level quantity here is a template argument or a
//: register-array index, so the level index must be a constant expression; a
//: `for (int l = 0; l < D; ++l)` would make `load_run<n_l(l)>` ill-formed and would
//: turn `lv[l][...]` into a dynamic index, which is a stack frame.
template <int N, class F>
__device__ __forceinline__ void static_for(F&& f) {
  if constexpr (N > 0) {
    static_for<N - 1>(f);
    f(std::integral_constant<int, N - 1>{});
  }
}

//: THE CARVE IS SETTLED AND IS THEREFORE NOT AN AXIS (Blake ruling 2026-08-20,
//: commit 46a3d3e; the integration rung removes the machinery the ruling left
//: standing).  Each warp owns `kMmaM` of the `d_v` CHANNELS across ALL `BC`
//: leaves, so the four warps' readout rows are DISJOINT: nothing in the owner
//: loop reduces across warps, and the kernel's only fan-in is the RED at the top.
//: The state-split alternative -- warps owning leaf slices and meeting in an
//: intra-CTA shared-memory tree -- was R1's measured loser on sm_80 and lives in
//: git, not in a dead branch.

//: a contiguous run of `n` bf16, at the widest access that covers it.  The runs are
//: naturally aligned by construction (an owner's run starts at a multiple of `s_l`
//: and a bucket's key digit run at a multiple of its own width), so this is one,
//: two or four accesses.
template <int n>
__device__ __forceinline__ void load_run(unsigned short (&dst)[n], const void* p) {
  static_assert(n == 1 || n == 2 || n == 4 || (n >= 8 && n % 8 == 0),
                "a run is one access of 2, 4, 8 or 16 bytes, or a whole number of them");
  if constexpr (n == 1) {
    dst[0] = *reinterpret_cast<const unsigned short*>(p);
  } else if constexpr (n == 2) {
    *reinterpret_cast<uint32_t*>(dst) = *reinterpret_cast<const uint32_t*>(p);
  } else if constexpr (n == 4) {
    *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(p);
  } else {
#pragma unroll
    for (int i = 0; i < n / 8; ++i)
      *reinterpret_cast<uint4*>(dst + i * 8) =
          load_vec16(reinterpret_cast<const char*>(p) + i * 16);
  }
}

//: THE K33 GEOMETRY -- the factorized bucket contraction's shapes, all derived.
//: `RK` / `WK` are the READ and WRITE side's KEY LEVEL: the level whose digit
//: indexes the buckets, or -1 for a side with no sparse level.  They are a
//: PLAN property (the per-level support-mode declaration, law 3), never a
//: computed statistic -- the kernel is told the STRUCTURE and never the density.
//:
//: `NB` buckets of `P` leaves each, `NB * P = BC` exactly: the bucket key
//: factors the box's leaf axis, it does not sample it.  A side with no key is
//: ONE bucket of `BC` leaves, which is the pre-K33 kernel written in the new
//: vocabulary -- no branch, no second path.
template <int D, int B, int K, int M, int KEY>
struct BucketShape {
  using BP = BoxPlan<D, B, K, M>;
  static constexpr int kSpan = (KEY < 0) ? 1 : BP::s(KEY);  // buckets
  static constexpr int kWidth = BP::kBC / kSpan;            // leaves per bucket
  //: the leaves BELOW the key level, i.e. how far apart a bucket's leaves sit.
  //: `>= BC` means "no key", and then the image map below is the identity.
  static constexpr int kLo = (KEY < 0) ? BP::kBC : BP::prod_s(KEY + 1);
  //: bucket `d`'s `j`-th leaf, in the owner's LATTICE-local order.
  static constexpr int leaf_of(int d, int j) {
    return (KEY < 0) ? j : (j / kLo) * (kLo * kSpan) + d * kLo + (j % kLo);
  }
  //: and the inverse: where `leaf` sits once the leaves are re-blocked
  //: BUCKET-MAJOR.  This is the READOUT's state image map (the ratified SMEM
  //: substitute for the card's constexpr slice-pack): under it a bucket's
  //: `kWidth` leaves are CONTIGUOUS, so its `A` operand is one `ldmatrix`.
  static constexpr int img_of(int leaf) {
    return (KEY < 0)
               ? leaf
               : ((leaf / kLo) % kSpan) * kWidth + (leaf / (kLo * kSpan)) * kLo + (leaf % kLo);
  }
};

//: THE PANEL PRODUCE'S THREAD MAP, one statement for both sides.  A panel row is
//: one bucket ENTRY and a panel column is the Kronecker product of the DENSE
//: levels -- every level except the key -- so `SKIP` drops exactly one factor and
//: `SKIP = -1` is the whole box, which is the pre-K33 produce.
//:
//: The column index `c` enumerates the non-key levels in the SAME mixed radix the
//: leaf order uses, with the key level's place value divided out; a level's run
//: offset is therefore the bit field at `shift(l)`, and a thread's `SPT`
//: consecutive columns vary exactly the bottom `ilog2(SPT)` bits of it.  So the
//: expansion is `sum_l n_l` amplitude loads followed by `SPT` products of register
//: values at CONSTANT indices -- no dynamic indexing, no spill, no materialized
//: gate plane.
template <int D, int B, int K, int M, int ROWS, int SKIP, int THR = kThreads>
struct PanelMap {
  using BP = BoxPlan<D, B, K, M>;
  static constexpr int kSkipSpan = (SKIP < 0) ? 1 : BP::s(SKIP);
  static constexpr int kWidth = BP::kBC / kSkipSpan;
  //: `THR` is the thread population that expands the panel -- the CTA for the
  //: fold's shared panel, ONE WARP for the readout's private one.
  static constexpr int kRowThr = THR / ROWS;
  static constexpr int SPT = kWidth / kRowThr;
  static_assert(kRowThr * SPT == kWidth, "the row's threads cover the panel exactly");
  static constexpr int place(int l) {
    return (SKIP >= 0 && l < SKIP) ? BP::prod_s(l + 1) / kSkipSpan : BP::prod_s(l + 1);
  }
  static constexpr int shift(int l) { return ilog2c(place(l)); }
  static constexpr int vary(int l) { return SPT >> shift(l); }
  static constexpr int n_l(int l) {
    return l == SKIP ? 0 : (vary(l) < 1 ? 1 : (vary(l) > BP::s(l) ? BP::s(l) : vary(l)));
  }
  static constexpr int max_n(int l) {
    return l == D ? 1 : (n_l(l) > max_n(l + 1) ? n_l(l) : max_n(l + 1));
  }
  static constexpr int kMaxN = max_n(0);
  //: element of level `l`'s loaded run that unrolled column `j` wants.
  static constexpr int idx_of(int l, int j) { return (j >> shift(l)) & (n_l(l) - 1); }
  //: the level's run offset at column `c` -- the loaded run's aligned base.
  static constexpr int base_of(int l, int c) { return (c >> shift(l)) & (BP::s(l) - 1); }
  //: THE INNERMOST NON-KEY LEVEL supplies the ADJACENT PAIR (its place value is
  //: one, so its digits are the fastest-varying columns).
  static constexpr int kInner = (SKIP == D - 1) ? D - 2 : D - 1;
  static_assert(kInner >= 0, "a panel needs at least one dense level");
};

template <int D, int B, int K, int M, int W, int C, int RK, int WK>
struct CarryGeom {
  using BP = BoxPlan<D, B, K, M>;
  using RS = BucketShape<D, B, K, M, RK>;
  using WS = BucketShape<D, B, K, M, WK>;
  static constexpr int BC = BP::kBC;
  static constexpr int LdV = kDv + kPadE;
  static constexpr int LdO = kDv + kPadF;

  //: THE FOLD'S CONSUMER SET, and the ONE rule that covers both structures.  A
  //: fold bucket's leaves are a contiguous run; the warps that own leaves in it are
  //: its CONSUMERS, and there are `kWarps` of them when the bucket is the whole box
  //: (a side with no key) and exactly ONE when it is narrower than a warp's run.
  //: The bucket's panel and its `V` rows are produced BY ITS CONSUMERS, SPLIT BY
  //: ROW, and waited on a NAMED barrier over exactly those threads -- so a keyed
  //: fold never crosses a warp and a dense fold gathers `V` exactly once, which the
  //: obvious alternative (a private ring per warp) would have turned into `kWarps`
  //: times the DRAM traffic, a movement-hierarchy violation and not a tuning
  //: choice.  `kFoldGroups` such sets run CONCURRENTLY, so the ring is per set.
  static constexpr int kFoldCons = (WS::kWidth >= BP::kBC / kWarps) ? kWarps : 1;
  static constexpr int kFoldThr = kFoldCons * 32;
  static constexpr int kFoldGroups = kWarps / kFoldCons;
  //: THE FOLD'S DRAIN GROUP, SIZED SO THE CTA'S IN-FLIGHT FOLD FOOTPRINT DOES NOT
  //: DEPEND ON THE STRUCTURE.  Ownership makes `kFoldGroups` consumer sets drain
  //: CONCURRENTLY, each with its own panel ring and its own value ring, so the
  //: group scales inversely with that count and `kGrp * kFoldGroups` is the same
  //: number for every key pair.  The floor is `kMmaK`, because a group shorter than
  //: the mma's contraction extent would pad the fold's own `k` axis.
  //:
  //: K33 doubled this for a measured reason -- "every group carries a produce, TWO
  //: BARRIERS and a fan-in pass" -- and that reason EXPIRES with the barriers where
  //: ownership deletes them: a private group's produce, staging and fan-in are the
  //: warp's own and cost a `__syncwarp`.  Where ownership does NOT delete them (a
  //: side with no key, whose one bucket every warp consumes) the group stays at
  //: K33's value, which is what the formula says.
  static constexpr int kGrpMin = 2 * kMmaK / kFoldGroups;
  static constexpr int kGrp = kGrpMin < kMmaK ? kMmaK : kGrpMin;
  //: THE SCATTER CHUNK is the CTA's width, walked by ONE warp in lockstep units: an
  //: entry's rank inside its bucket is then `popc(vote & below)` against a base only
  //: lane zero advances, so the append needs no cross-warp scan, no atomic and NO
  //: BARRIER -- and the entry ORDER stays a function of the token list alone, which
  //: is what keeps a chained state bit-identical run to run.  The chunk is `kThreads`
  //: and not one unit because the DRAIN that follows it must see several groups to
  //: run its produce one group ahead; at one unit per drain there is never a next
  //: group to produce, and the amplitude gather goes back in front of the mma block.
  static constexpr int kChunk = kThreads;
  //: THE BUCKET'S CAPACITY: one scatter chunk plus the residual a drain leaves.
  //: Entries persist ACROSS windows (the gated cross-window batching), so the
  //: residual is a standing tenant and not a per-window remainder.
  static constexpr int kCap = ((kChunk + kGrp - 1) + 7) / 8 * 8;

  static constexpr int PRP = (RS::kWidth + kMmaK - 1) / kMmaK * kMmaK;
  //: AN `ldmatrix` ROW STRIDE, and the ONE arithmetic condition that keeps it off
  //: a bank conflict.  The fragment reads eight rows of sixteen bytes, so the eight
  //: rows must land in eight DIFFERENT 16-byte bank groups: the stride measured in
  //: 16-byte groups has to be ODD.  `stride16` rounds a required width up to the
  //: nearest such stride.
  //:
  //: This is not a micro-optimization.  At `BC = 128` the state image's natural
  //: row stride is 128 bf16 = 16 groups -- EVEN, so every row of the image starts
  //: on the same banks and the operand that carries the entire readout was issuing
  //: at THIRTY-TWO-WAY conflict (ncu, dense cell: `LDSM.16.M88.4` at n-way 32,
  //: 473M shared-load conflicts against master's 5M).
  static constexpr int stride16(int width) { return ((width + 15) / 16) * 16 + 8; }
  //: THE STATE IMAGE'S ROW STRIDE.  At least the slack the LAST bucket's `A` operand
  //: reads past the box when a bucket is narrower than one `k` tile.  Those columns
  //: meet a panel column the produce zeroed, so what they hold never reaches an
  //: output -- they are zeroed anyway, because a shared-memory word that happens to
  //: encode a NaN would survive the multiplication.
  static constexpr int LdImg = stride16(BC + (PRP - RS::kWidth));
  static constexpr int kImgBytes = kWarps * kMmaM * LdImg * (int)sizeof(__nv_bfloat16);
  static constexpr int kMassBytes = LdImg * (int)sizeof(__nv_bfloat16);

  //: THE READOUT'S UNIT, and it is `kMmaK` for every key pair.  Its entries are the
  //: accumulator's `n` axis and a warp spans every CHANNEL m-tile, so the unit sizes
  //: the register cost directly (`MDV * (kGrpR / kMmaN) * 4` floats) AND the private
  //: panel's shared footprint (`kWarps * kGrpR * LdRP`), and both are binding at the
  //: widest panel -- a DENSE read side, whose bucket is the whole box.  Sixteen
  //: entries is 32 accumulator registers and two whole n-tiles, so one `ldmatrix` of
  //: the panel still feeds both mma of a k-step.
  static constexpr int kGrpR = kMmaK;
  static constexpr int LdRP = stride16(PRP);
  static constexpr int LdWP = stride16(WS::kWidth);
  //: ONE READOUT PANEL SLOT PER WARP, not two per CTA: the ring bought latency the
  //: shared form could not otherwise hide, and the private form does not need it
  //: (micro-probe 3: producing one bucket ahead is neutral-to-worse once the warps
  //: are free-running).  Same bytes, four private producers.
  static constexpr int rp_bytes = kWarps * kGrpR * LdRP * (int)sizeof(__nv_bfloat16);
  static constexpr int wp_bytes = 2 * kFoldGroups * kGrp * LdWP * (int)sizeof(__nv_bfloat16);
  //: THE VALUE RING: two free-on-consume slots of one drain group's rows, PER
  //: CONSUMER SET.  `V` rides the WRITE buckets and is gathered once per write
  //: entry, at its own bucket's stage (single-touch stream law); the gather for the
  //: next group is issued before the current group's mma block and waited after it.
  static constexpr int kStages = 2;
  static constexpr int v_bytes = kStages * kFoldGroups * kGrp * LdV * (int)sizeof(__nv_bfloat16);
  //: A BUCKET ENTRY: the GLOBAL token index (an entry outlives its window) and the
  //: key digit's coefficient.
  static constexpr int entry_bytes(int nb) {
    return nb * kCap * (int)(sizeof(uint32_t) + sizeof(__nv_bfloat16));
  }
  //: THE PHASE UNION.  The state image, the mass row, the read panel ring and the
  //: READ BUCKETS are PHASE-1 tenants; the value ring, the write panel ring and the
  //: WRITE BUCKETS are PHASE-2 tenants.  They are disjoint IN TIME on both counts: a
  //: fold invalidates the image whose bytes it takes (the next readout rebuilds it
  //: either way), and a NONEMPTY read bucket never coexists with a fold, because
  //: each phase empties the other side's buckets before filling its own -- so at
  //: most one side is ever pending.  That was a scheduling invariant (the
  //: window-grain read-before-write law); here it is also a memory-safety one.
  //: This is the architecture doc's V/staging alias, widened.
  static constexpr int phase1_bytes = kImgBytes + kMassBytes + rp_bytes + entry_bytes(RS::kSpan);
  static constexpr int phase2_bytes = v_bytes + wp_bytes + entry_bytes(WS::kSpan);
  static constexpr int phase_bytes = phase1_bytes > phase2_bytes ? phase1_bytes : phase2_bytes;

  //: THE OUTPUT STAGING IS ONE N-TILE PER WARP.  A private accumulator drains
  //: eight entries at a time and fans them in immediately, so the buffer stays at
  //: the shared carve's footprint.  `sden` is GONE: the mass row is the warp's own
  //: contraction now and rides column `kDv` of its own slice.
  static constexpr int out_bytes = kWarps * kMmaN * LdO * (int)sizeof(float);
  static constexpr int den_bytes = 0;
  static constexpr int kListStride = ((W + 7) / 8) * 8;
  static constexpr int list_bytes = 2 * kListStride * (int)sizeof(uint16_t);
  //: 16 scan words, a live count per bucket, and a drained count per bucket -- the
  //: last so the residual COMPACTION of every bucket can happen behind ONE pair of
  //: barriers after the bucket loop instead of a pair inside each iteration.  At the
  //: graded cell a bucket drains one group, so that pair WAS the drain's dominant
  //: cost: barrier is this kernel's top stall at 2.2 warps per issue-active.
  static constexpr int scan_bytes = (16 + 2 * (RS::kSpan + WS::kSpan)) * (int)sizeof(int);

  //: `kRegFloor` is the DECLARED floor for this family -- the measured census of
  //: the heaviest built arm, recorded in the findings doc and re-measured at every
  //: milestone.  K33 raised it: the state slab, a bucket's A operand and the two
  //: readout accumulators are all resident at once, and the heaviest key pair
  //: censuses 254.  A floor left at the pre-K33 168 does not merely under-state the
  //: need -- it hands the DENSE pair (whose smaller shared footprint admits three
  //: CTAs) a 170-register launch bound, and that arm SPILLS.  The ideal-regime
  //: ruling settles which way to move: few warps with filled registers is the
  //: target, so the floor follows the census and the residency follows the floor.
  static constexpr int kRegFloor = 254;
  static constexpr size_t bytes =
      (size_t)(phase_bytes + out_bytes + den_bytes + list_bytes + scan_bytes);
  static constexpr int cta_target = design::cta_target((int)bytes, kThreads, kRegFloor);
  //: THE RESIDENCY THIS FAMILY IS SIZED FOR, asserted rather than hoped: the ideal
  //: regime is few warps with filled registers, and one CTA per SM is not it.  A key
  //: pair whose shared footprint would drop below it is a REFUSAL, not a silently
  //: shrunk arm.
  static_assert(cta_target >= 2, "this key pair's shared footprint admits one CTA per SM");
};

//: THE SECOND LAUNCH BOUND: the CTA count this arm can actually reach, taken from
//: the same geometry the ring depth is.  A ring sized for a residency the launch
//: bound does not ask for is a ring sized for nothing.
template <int D, int B, int K, int M, int W, int C, int RK, int WK>
constexpr int kCarryCtaTarget = CarryGeom<D, B, K, M, W, C, RK, WK>::cta_target;

// ------------------------------------------------------------- the produce
//: ONE PANEL ROW, expanded from factors.  The row is one bucket ENTRY and its
//: columns are the Kronecker product of the DENSE levels -- `SKIP` is the key
//: level and drops out, `SKIP = -1` keeps them all.
//:
//: THE PANEL IS UNGAINED AND UNSCALED, on both sides (R4 section 4.2/4.3).  The
//: bucket scalar and the write gain are applied LATE, on the entry axis: the
//: readout scales its ACCUMULATOR COLUMN (so the accumulator literally is the
//: backward's unscaled `q^(d)` at the moment before scaling), and the fold scales
//: its `A` operand's TOKEN axis.  `gate` is therefore only the dead-row zeroing: a
//: padding row of a partial group has no entry, and a zero coefficient contributes
//: exactly nothing to either MMA.
template <class PM, int D, int SKIP>
__device__ __forceinline__ void produce_row(const __nv_bfloat16* base, __nv_bfloat16* row,
                                            const int (&pcol)[D], uint32_t tokOff, uint32_t gate) {
  unsigned short lv[D][PM::kMaxN];
  static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    if constexpr (l != SKIP) {
      unsigned short run[PM::n_l(l)];
      load_run<PM::n_l(l)>(run, gather_ptr(base, tokOff + (uint32_t)(pcol[l] * 2)));
#pragma unroll
      for (int i = 0; i < PM::n_l(l); ++i) lv[l][i] = run[i];
    }
  });
  constexpr int IN = PM::kInner;
  if constexpr (PM::SPT >= 2) {
#pragma unroll
    for (int j = 0; j < PM::SPT; j += 2) {
      //: the innermost DENSE level supplies the ADJACENT PAIR (its place value is
      //: one), every level above it a per-pair scalar -- the producer's chain.
      uint32_t acc =
          ((uint32_t)lv[IN][PM::idx_of(IN, j + 1)] << 16) | (uint32_t)lv[IN][PM::idx_of(IN, j)];
      static_for<D>([&](auto Lc) {
        constexpr int l = decltype(Lc)::value;
        if constexpr (l != SKIP && l != IN)
          acc = mul_bf16x2(
              acc, splat_bf16x2(*reinterpret_cast<const __nv_bfloat16*>(&lv[l][PM::idx_of(l, j)])));
      });
      store_bf16x2(row + j, mul_bf16x2(acc, gate));
    }
  } else {
    //: a one-column thread has no pair to form, so the chain runs on a splat and
    //: only its low half is stored.
    uint32_t acc = (uint32_t)lv[IN][0] | ((uint32_t)lv[IN][0] << 16);
    static_for<D>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      if constexpr (l != SKIP && l != IN)
        acc = mul_bf16x2(acc, splat_bf16x2(*reinterpret_cast<const __nv_bfloat16*>(&lv[l][0])));
    });
    *reinterpret_cast<unsigned short*>(row) = (unsigned short)(mul_bf16x2(acc, gate) & 0xFFFFu);
  }
}

// ---------------------------------------------------------------- the kernel

template <int D, int B, int K, int M, int W, int C, int RK, int WK, bool STATE>
__global__ __launch_bounds__(kThreads, (kCarryCtaTarget<D, B, K, M, W, C, RK, WK>)) void
carry_kernel(const __nv_bfloat16* __restrict__ pread, const __nv_bfloat16* __restrict__ pwrite,
             const __nv_bfloat16* __restrict__ gwrite, const __nv_bfloat16* __restrict__ v,
             const uint32_t* __restrict__ rows, float* __restrict__ o, float* __restrict__ den,
             int L, int tok_words, int v_row_bytes, const float* __restrict__ s_in,
             float* __restrict__ s_out) {
  using BP = BoxPlan<D, B, K, M>;
  using G = CarryGeom<D, B, K, M, W, C, RK, WK>;
  using RS = typename G::RS;
  using WS = typename G::WS;
  using RPM = PanelMap<D, B, K, M, G::kGrpR, RK, 32>;
  using WPM = PanelMap<D, B, K, M, G::kGrp, WK, G::kFoldThr>;

  constexpr int BC = BP::kBC;
  constexpr int NDV = BC / kMmaN;  // the box's n-tiles (leaves)
  //: THE OWNERSHIP GRAIN (wave 2).  A warp owns a contiguous LATTICE-LOCAL run of
  //: `kBCW` leaves and ALL `kDv` channels -- the carve that makes a fold bucket
  //: warp-private, since the fold keys level zero and a level-zero bucket is a
  //: contiguous run of whole state tiles.  The slab does not grow: the carve
  //: ROTATES it (16 channels x 128 leaves and 64 channels x 32 leaves are the same
  //: 2048 accumulator floats), and the readout is untouched by the rotation because
  //: its `A` operand transits the SMEM state image, not the registers.
  constexpr int kGrp = G::kGrp, kGrpR = G::kGrpR, kCap = G::kCap;
  constexpr int kFoldCons = G::kFoldCons, kFoldThr = G::kFoldThr;
  constexpr int kFoldGroups = G::kFoldGroups;
  constexpr int NBR = RS::kSpan, PR = RS::kWidth, PRP = G::PRP;
  constexpr int NBW = WS::kSpan, PW = WS::kWidth;
  constexpr int LdImg = G::LdImg, LdRP = G::LdRP, LdWP = G::LdWP;
  constexpr int LdV = G::LdV, LdO = G::LdO;
  constexpr int kBCW = BC / kWarps;   // the warp's leaves
  constexpr int NDVW = kBCW / kMmaN;  // its state n-tiles
  constexpr int MDV = kDv / kMmaM;    // the channel m-tiles it now spans
  constexpr int kMassTiles = NDVW;
  constexpr int kListStride = G::kListStride;

  //: THE FOLD KEYS THE OUTERMOST LEVEL, AND ONLY IT.  A bucket's leaves are a
  //: contiguous run of whole state tiles exactly when no level ABOVE the key has a
  //: span, and the fold's accumulator IS the state -- it cannot be gathered.  A
  //: write side sparse only at an inner level therefore runs the trivial one-bucket
  //: fold with a partially zero panel: exact, uniform, one code path.
  static_assert(WK < 0 || WK == 0, "the fold keys the outermost level only");
  static_assert(PW % kMmaN == 0, "a fold bucket is a whole number of state tiles");
  static_assert(BC % kMmaN == 0 && NDV % kWarps == 0, "the box tiles the mass carve");
  static_assert(kBCW % kMmaN == 0 && kDv % kMmaM == 0, "the warp owns whole state tiles");
  static_assert(kThreads % kGrp == 0, "the drain group's threads divide");
  static_assert(BC % kAtomLeaves == 0, "K2: an owner holds whole page atoms");
  static_assert(design::smem_fits(G::bytes), "this arm does not fit the SMEM maximum");
  static_assert(LdImg % 8 == 0 && LdRP % 8 == 0 && LdWP % 8 == 0 && LdV % 8 == 0,
                "an ldmatrix row address must be 16-byte aligned");
  static_assert(BP::span_bits(D - 1) >= 1, "the innermost level supplies the adjacent pair");
  static_assert(W % kBallotLanes == 0 && kThreads % kBallotLanes == 0,
                "a warp's tokens must be exactly one ballot word of the owner's row");
  //: THE IMAGE'S STORE WIDTH: two accumulator elements that are adjacent in the
  //: IMAGE, at the widest access that covers them.  Either the leaf pair of one
  //: state tile is adjacent (the key's bucket run is contiguous) or the same
  //: position in two adjacent tiles is (the key's run is strided by exactly one
  //: tile); the admitted span vectors give one or the other.
  constexpr bool kLeafPair =
      RS::img_of(1) == RS::img_of(0) + 1 && RS::img_of(kMmaN + 1) == RS::img_of(kMmaN) + 1;
  constexpr bool kTilePair =
      RS::img_of(kMmaN) == RS::img_of(0) + 1 && RS::img_of(kMmaN + 1) == RS::img_of(1) + 1;
  static_assert(kLeafPair || kTilePair,
                "no two accumulator elements are adjacent in the bucket-major image, "
                "so the state pack would be one store per element");

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int owner = (int)blockIdx.x;
  const int bh = (int)blockIdx.y;

  extern __shared__ __align__(16) char carry_smem[];
  //: THE PHASE UNION at a fixed offset: phase 1 is the state image, the mass row and
  //: the read panel ring; phase 2 is the value ring and the write panel ring.
  __nv_bfloat16* const simg = reinterpret_cast<__nv_bfloat16*>(carry_smem);
  __nv_bfloat16* const smass = simg + kWarps * kMmaM * LdImg;
  __nv_bfloat16* const srp = smass + LdImg;
  uint32_t* const btok = reinterpret_cast<uint32_t*>(srp + kWarps * kGrpR * LdRP);
  __nv_bfloat16* const bval = reinterpret_cast<__nv_bfloat16*>(btok + NBR * kCap);
  __nv_bfloat16* const sv = reinterpret_cast<__nv_bfloat16*>(carry_smem);
  __nv_bfloat16* const swp = sv + G::kStages * kFoldGroups * kGrp * LdV;
  uint32_t* const wtok = reinterpret_cast<uint32_t*>(swp + 2 * kFoldGroups * kGrp * LdWP);
  __nv_bfloat16* const wval = reinterpret_cast<__nv_bfloat16*>(wtok + NBW * kCap);
  float* const so = reinterpret_cast<float*>(carry_smem + G::phase_bytes);
  uint16_t* const slist = reinterpret_cast<uint16_t*>(so + kWarps * kMmaN * LdO);
  int* const sscan = reinterpret_cast<int*>(slist + 2 * kListStride);
  int* const scnt = sscan + 16;    // [NBR] live read entries
  int* const wcnt = scnt + NBR;    // [NBW] live write entries
  int* const sdone = wcnt + NBW;   // [NBR] read entries this drain consumed
  int* const wdone = sdone + NBR;  // [NBW] write entries this drain consumed

  //: ONE 64-BIT BASE PER PLANE, and a 32-BIT BYTE DISPLACEMENT after it.  Bucket
  //: entries carry the GLOBAL token index, because they outlive their window.
  const int vRowElems = v_row_bytes / (int)sizeof(__nv_bfloat16);
  const int heads = vRowElems / kDv;
  const __nv_bfloat16* const vPlane =
      pin_address(v + (long)(bh / heads) * L * vRowElems + (long)(bh % heads) * kDv);
  const __nv_bfloat16* const pR = pin_address(pread + (long)bh * L * BP::kWtot);
  const __nv_bfloat16* const pW = pin_address(pwrite + (long)bh * L * BP::kWtot);
  const __nv_bfloat16* const gW = pin_address(gwrite + (long)bh * L);
  float* const oPlane = pin_address(o + (long)bh * L * kDv);
  float* const denPlane = pin_address(den + (long)bh * L);

  // ---- the owner's digit runs and its ballot rows, one per level -----------
  int runCol[D];
  uint32_t rowOff[D];
  const uint32_t rowBytes = (uint32_t)tok_words * (uint32_t)sizeof(uint32_t);
  static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    const int o_l = (owner / BP::owner_div(l)) % BP::g(l);
    runCol[l] = BP::col_base(l) + o_l * BP::s(l);
    rowOff[l] = (uint32_t)(BP::row_prefix(l) + o_l) * rowBytes;
  });
  const uint32_t* const rowsR =
      pin_address(rows + (long)(2 * bh + 0) * (long)BP::kRows * (long)tok_words);
  const uint32_t* const rowsW =
      pin_address(rows + (long)(2 * bh + 1) * (long)BP::kRows * (long)tok_words);

  // ---- lane-invariant operand addressing ----------------------------------
  const int lr = lane & 7, tl = lane >> 3;
  //: `A` from an `[m][k]` image (readout state), `B` from an `[n][k]` panel
  //: (readout), `B` from a `[k][n]` panel transposed (fold), `A` from a `[k][m]`
  //: tile transposed (fold's V).  Four maps, each a property of one operand's
  //: memory orientation and of nothing else.
  const int rA = lr + 8 * (tl & 1), cA = 8 * (tl >> 1);
  const int rT = (tl >> 1) * 8 + lr, cT = tl & 1;
  const int rF = (tl & 1) * 8 + lr, cF = tl >> 1;
  //: the warp's LEAF origin.  Its channel origin is zero: it holds every channel.
  const int leaf0 = warp * kBCW;
  //: THE FOLD'S CONSUMER SET: which set this warp is in, and its rank inside it.
  //: At `kFoldCons == 1` the set IS the warp and `fgtid` is the lane; at
  //: `kFoldCons == kWarps` the set is the CTA and `fgtid` is the thread id.  Named
  //: barrier zero is the CTA's, so the sets take ids from one.
  const int fgrp = warp / kFoldCons;
  const int fgtid = (warp % kFoldCons) * 32 + lane;
  const int fbar = fgrp + 1;
  const int chan0 = warp * kMmaM;  // the READOUT's channel tile (carve-independent)

  //: the readout's `A` base carries NO channel origin: the warp walks every
  //: channel m-tile of the image, and the tile's row offset is added at the load.
  const uint32_t aImg = smem_addr(simg) + (uint32_t)(rA * LdImg + cA) * 2;
  const uint32_t aMass = smem_addr(smass) + (uint32_t)cA * 2;
  const uint32_t bRP = smem_addr(srp) + (uint32_t)((warp * kGrpR + rT) * LdRP + cT * 8) * 2;
  const uint32_t bWP = smem_addr(swp) + (uint32_t)((fgrp * 2 * kGrp + rF) * LdWP + cF * 8) * 2;
  const uint32_t bWP2 = smem_addr(swp) + (uint32_t)((fgrp * 2 * kGrp + (lane & 15)) * LdWP) * 2;
  //: the fold's `A` operand walks the CHANNEL m-tiles now, so its base carries no
  //: channel origin and the tile's offset is added at the load.
  const uint32_t aV =
      smem_addr(sv) + (uint32_t)((fgrp * G::kStages * kGrp + rT) * LdV + cT * 8) * 2;
  constexpr uint32_t kWpSlot = (uint32_t)(kGrp * LdWP) * 2;
  constexpr uint32_t kVSlot = (uint32_t)(kGrp * LdV) * 2;
  constexpr int kMTok = kGrp / kMmaM;    // entry m-tiles of a FOLD group
  constexpr int kNTokR = kGrpR / kMmaN;  // entry n-tiles of a READOUT unit
  static_assert(kGrpR % kMmaM == 0, "a readout unit is a whole mma m-tile of entries");
#define ADDR_RP(b_, R_, c_) ((b_) + (uint32_t)(((R_) * LdRP + (c_)) * 2))
#define ADDR_WP(b_, R_, c_) ((b_) + (uint32_t)(((R_) * LdWP + (c_)) * 2))
#define ADDR_V(b_, R_) ((b_) + (uint32_t)((R_) * LdV * 2))

  // ---- the state, transposed: m = channels, n = this warp's leaves ---------
  float st[MDV][NDVW][4];
#pragma unroll
  for (int m = 0; m < MDV; ++m)
#pragma unroll
    for (int j = 0; j < NDVW; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) st[m][j][e] = 0.f;
  //: THE MASS CHANNEL RIDES THE WARP'S OWN TILES.  Under the ownership carve the
  //: leaves are already the carved axis, so the mass -- the one channel that cannot
  //: be split along channels -- needs no separate distribution and no predicate:
  //: a warp accumulates the mass of exactly the tiles it owns.
  float mass_st[kMassTiles][4];
#pragma unroll
  for (int j = 0; j < kMassTiles; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) mass_st[j][e] = 0.f;

  constexpr uint32_t kOneSplat = 0x3F803F80u;

  // ---- the state plane's page sweep, in LATTICE leaf order -----------------
  constexpr int kAtoms = BC / kAtomLeaves;
  constexpr int kPageElems = kAtomLeaves * kCols;
  constexpr int kPageCarry = kThreads - kCols;
  static_assert(kThreads > kCols && kThreads < 2 * kCols,
                "the page sweep's step must cross one row boundary, so its row "
                "cursor advances by one or two and never by a division");
  static_assert(kWarps * kMmaN * LdO >= kPageElems, "the staging buffer must hold one page");
  auto page_sweep = [&](uint32_t pageOff, auto&& body) {
    int e = tid;
    int row = tid < kCols ? 0 : 1;
    int col = tid < kCols ? tid : tid - kCols;
#pragma unroll 1
    while (e < kPageElems) {
      body(row * LdO + col, pageOff + (uint32_t)(e * (int)sizeof(float)));
      e += kThreads;
      col += kPageCarry;
      const bool wrap = col >= kCols;
      row += wrap ? 2 : 1;
      col -= wrap ? kCols : 0;
    }
  };
  const uint32_t atomBase = (uint32_t)(bh * (int)(BP::kLeaves / kAtomLeaves) + owner * kAtoms);
  auto page_off = [](uint32_t atom) { return atom * (uint32_t)(kPageElems * (int)sizeof(float)); };
  auto page_value = [&](int m, int p, int k) {
    return (p * 8 + (lane & 3) * 2 + (k & 1)) * LdO + m * kMmaM + (lane >> 2) + 8 * (k >> 1);
  };
  auto page_mass = [&](int p, int k) { return (p * 8 + lane * 2 + k) * LdO + kDv; };

  //: ---- THE ENTRY STATE.  The fragments are SET, never accumulated: they are
  //: provably zero here, and with no decay the recurrence is affine in the state
  //: with an inclusive-after-write readout, so this IS the oracle's seeding.
  if constexpr (STATE) {
    if (s_in != nullptr) {
      const float* const plane = pin_address(s_in);
#pragma unroll
      for (int a = 0; a < kAtoms; ++a) {
        page_sweep(page_off(atomBase + (uint32_t)a), [&](int sidx, uint32_t off) {
          so[sidx] = *reinterpret_cast<const float*>(gather_ptr(plane, off));
        });
        rendezvous();
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          //: the page is swept by the whole CTA and read by the tile's OWNER.  `a`
          //: and `p` are unrolled, so the owning warp and the LOCAL tile index are
          //: both constant expressions -- which is what keeps `st` a register array.
          if ((2 * a + p) / NDVW == warp) {
#pragma unroll
            for (int m = 0; m < MDV; ++m)
#pragma unroll
              for (int k = 0; k < 4; ++k) st[m][(2 * a + p) % NDVW][k] = so[page_value(m, p, k)];
            if (lane < 4) {
#pragma unroll
              for (int k = 0; k < 2; ++k) mass_st[(2 * a + p) % NDVW][k] = so[page_mass(p, k)];
            }
          }
        }
        rendezvous();
      }
    }
  }

  // ---- the two panels' per-thread maps ------------------------------------
  //: the readout panel is expanded by ONE WARP into its own slot, so its row map
  //: is a LANE map; the fold's panel is still the CTA's.
  const int rprow = lane / RPM::kRowThr, rslot0 = (lane % RPM::kRowThr) * RPM::SPT;
  const int wprow = fgtid / WPM::kRowThr, wslot0 = (fgtid % WPM::kRowThr) * WPM::SPT;
  int rcol[D], wcol[D];
  static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    rcol[l] = (l == RK) ? 0 : runCol[l] + RPM::base_of(l, rslot0);
    wcol[l] = (l == WK) ? 0 : runCol[l] + WPM::base_of(l, wslot0);
  });

  // ---- the fan-in's own addressing ----------------------------------------
  //: the readout's fan-in is warp-private now: thirty-two lanes are thirty-two
  //: consecutive channels of ONE token, and the entry axis is a loop, so the CTA's
  //: row/pass accounting has nothing left to describe.
  static_assert(kDv == 2 * 32, "the fan-in covers d_v in two lane-wide halves");
  const uint32_t fanColB = (uint32_t)(lane * (int)sizeof(float));
  const uint32_t below = (1u << lane) - 1u;
  constexpr uint32_t kWtotB = (uint32_t)(BP::kWtot * 2);

  // ---- V staging: one drain group's rows, asynchronous ---------------------
  constexpr int kChunkV = kDv / 8;
  constexpr int kVRows = kFoldThr / kChunkV;
  constexpr int kVPasses = kGrp / kVRows;
  static_assert(kVPasses * kVRows == kGrp, "the staging passes cover the group exactly");
  const int ldRow = fgtid / kChunkV, ldCol = (fgtid % kChunkV) * 8;
  const uint32_t svL =
      smem_addr(sv) + (uint32_t)((fgrp * G::kStages * kGrp + ldRow) * LdV + ldCol) * 2;

  // ---- THE STATE IMAGE: the readout's A operand, bucket-major -------------
  //: THE RATIFIED SUBSTITUTE for the card's constexpr slice-pack.  A bucket's
  //: leaves are a STRIDED subset of the state's tiles, and the mma's A operand
  //: takes its `k` axis from the lane's low bits exactly as the accumulator takes
  //: its `n` axis -- so a register pack can only form an A fragment out of WHOLE
  //: in-order n-tiles, and a strided slice's values sit in the wrong LANES.  The
  //: pack therefore lands in shared memory, bucket-major, once per readout phase
  //: (not once per batch, which is what the pre-K33 kernel paid), and each bucket's
  //: A operand is one `ldmatrix`.  On sm_90 this image is exactly the SMEM-resident
  //: A operand `wgmma` requires, so the substitute is also the portable form.
  auto build_image = [&]() {
    //: UNDER THE OWNERSHIP CARVE THE IMAGE IS WRITTEN COLUMN-WISE.  A warp holds
    //: every channel of `kBCW` leaves, so it fills all `kDv` image ROWS at its own
    //: `kBCW` columns -- where the channel carve filled its own 16 rows at every
    //: column.  Same bytes, same map `img_of`, transposed writer.
    if constexpr (kLeafPair) {
#pragma unroll
      for (int m = 0; m < MDV; ++m)
#pragma unroll
        for (int t = 0; t < NDVW; ++t) {
          const int c0 = RS::img_of(leaf0 + t * kMmaN + (lane & 3) * 2);
          *reinterpret_cast<uint32_t*>(&simg[(m * kMmaM + (lane >> 2)) * LdImg + c0]) =
              pack_bf16x2(st[m][t][0], st[m][t][1]);
          *reinterpret_cast<uint32_t*>(&simg[(m * kMmaM + (lane >> 2) + 8) * LdImg + c0]) =
              pack_bf16x2(st[m][t][2], st[m][t][3]);
        }
    } else {
      static_assert(!kTilePair || NDVW >= 2,
                    "the tile-pair image pack needs two of the warp's tiles");
#pragma unroll
      for (int m = 0; m < MDV; ++m)
#pragma unroll
        for (int i = 0; i < NDVW / 2; ++i)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int c0 = RS::img_of(leaf0 + 2 * i * kMmaN + (lane & 3) * 2 + (e & 1));
            *reinterpret_cast<uint32_t*>(
                &simg[(m * kMmaM + (lane >> 2) + 8 * (e >> 1)) * LdImg + c0]) =
                pack_bf16x2(st[m][2 * i][e], st[m][2 * i + 1][e]);
          }
    }
    //: THE SLACK COLUMNS ARE ZEROED, not left as they lie: the last bucket's A
    //: operand reads a whole `k` tile past its own width, and a shared-memory word
    //: that happens to encode a NaN would survive multiplication by the produce's
    //: zeroed panel column.
    if constexpr (LdImg > BC) {
      for (int e = tid; e < kWarps * kMmaM * (LdImg - BC); e += kThreads) {
        const int r = e / (LdImg - BC);
        simg[r * LdImg + BC + e - r * (LdImg - BC)] = __float2bfloat16(0.f);
      }
      if (tid < LdImg - BC) smass[BC + tid] = __float2bfloat16(0.f);
    }
    if (lane < 4) {
#pragma unroll
      for (int q = 0; q < kMassTiles; ++q)
#pragma unroll
        for (int k = 0; k < 2; ++k)
          smass[RS::img_of(leaf0 + q * kMmaN + lane * 2 + k)] = __float2bfloat16(mass_st[q][k]);
    }
  };
  bool img_valid = false;
  auto ensure_image = [&]() {
    if (!img_valid) {
      rendezvous();
      build_image();
      rendezvous();
      img_valid = true;
    }
  };

  // ---- THE READOUT DRAIN: WARP-PRIVATE (bucket, group) UNITS ---------------
  //: THE OWNERSHIP GRAIN, on the side that owns no leaves.  The readout's `A`
  //: operand is the SMEM state image, which is read-only and carve-independent,
  //: so the readout is free to be carved on the axis that costs nothing: its work
  //: list is the (bucket, entry-group) pairs and the warps take them in a stripe.
  //: A warp then owns its unit's PANEL, its ACCUMULATOR and its OUTPUT STAGING --
  //: the three things a shared drain has to synchronize on -- and the entire
  //: bucket loop runs on `__syncwarp` alone.
  //:
  //: THE STRIPE IS ONE FORM FOR BOTH STRUCTURES.  A DENSE read side is one bucket
  //: with many groups, so the stripe is an ENTRY split with full n-tile fill; the
  //: ALT read side is `s_1` buckets with about one group apiece, so the same
  //: stripe is a BUCKET split.  No branch on a structure, no second path, and the
  //: trivial bucket needs no shared panel to degenerate well.
  //:
  //: Nothing waits on the stripe's balance, because nothing waits at all: a warp
  //: that draws a light unit goes on to its next one.
  //:
  //: `flush` executes a partial group; otherwise only WHOLE groups run and the
  //: residual stays for the next window (the gated cross-window batching).
  auto produce_read = [&](int b, int off, int cnt) {
    const bool alive = rprow < cnt;
    produce_row<RPM, D, RK>(pR, srp + warp * (kGrpR * LdRP) + rprow * LdRP + rslot0, rcol,
                            btok[b * kCap + off + (alive ? rprow : 0)] * kWtotB,
                            alive ? kOneSplat : 0u);
    if constexpr (PRP > PR) {
      __nv_bfloat16* const pad = srp + warp * (kGrpR * LdRP);
      for (int e = lane; e < kGrpR * (PRP - PR); e += 32) {
        const int r = e / (PRP - PR);
        pad[r * LdRP + PR + e - r * (PRP - PR)] = __float2bfloat16(0.f);
      }
    }
  };
  auto drain_read = [&](bool flush) {
    //: the unit counter advances on EVERY warp for every unit, drained or not --
    //: `scnt` is shared, so the four warps agree on the numbering without talking.
    int unit = 0;
#pragma unroll 1
    for (int b = 0; b < NBR; ++b) {
      const int n = scnt[b];
      const int groups = flush ? (n + kGrpR - 1) / kGrpR : n / kGrpR;
      if (tid == 0) sdone[b] = groups * kGrpR < n ? groups * kGrpR : n;
#pragma unroll 1
      for (int g = 0; g < groups; ++g, ++unit) {
        if ((unit & (kWarps - 1)) != warp) continue;
        const int off = g * kGrpR;
        const int cnt = (n - off) < kGrpR ? (n - off) : kGrpR;
        produce_read(b, off, cnt);
        __syncwarp();
        //: the warp spans EVERY channel of its entries, so the accumulator walks
        //: the image's channel m-tiles and the two sides' partials never meet:
        //: two warps hold partials of different ENTRIES, never of one output.
        //: That is what deletes the readout's cross-warp reduction -- the R1
        //: objection to state-split, which priced a leaf carve, not this one.
        float acc[MDV][kNTokR][4];
        float macc[kNTokR][4];
#pragma unroll
        for (int j = 0; j < kNTokR; ++j) {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            macc[j][e] = 0.f;
#pragma unroll
            for (int m = 0; m < MDV; ++m) acc[m][j][e] = 0.f;
          }
        }
#pragma unroll
        for (int kk = 0; kk < PRP / kMmaK; ++kk) {
          uint32_t rb[4];
          load_frag(rb, ADDR_RP(bRP, 0, kk * kMmaK));
          static_for<MDV>([&](auto Mc) {
            constexpr int mt = decltype(Mc)::value;
            uint32_t af[4];
            load_frag(af, aImg + (uint32_t)(mt * kMmaM * LdImg + b * PR + kk * kMmaK) * 2);
            mma(acc[mt][0], af, &rb[0]);
            mma(acc[mt][1], af, &rb[2]);
          });
          //: THE MASS ROW IS THE WARP'S OWN NOW.  It was carved across the warps by
          //: k-step and met in `sden` because under the channel carve no single warp
          //: held the whole contraction; under unit ownership one warp holds all of
          //: it, so the rotation, the `sden` buffer and its cross-warp sum all go.
          uint32_t mf[4];
          load_frag(mf, aMass + (uint32_t)(b * PR + kk * kMmaK) * 2);
          mma(macc[0], mf, &rb[0]);
          mma(macc[1], mf, &rb[2]);
        }
        //: R4: THE BUCKET SCALAR IS APPLIED HERE AND NOT IN THE PRODUCE.  The entry
        //: axis is the accumulator's `n` axis, so `acc` literally IS the backward's
        //: unscaled `q^(d)` at this instruction; scaling it per COLUMN is what keeps
        //: that object expressible.  A padding column's scalar reads zero, for the
        //: fold's reason: `0 * NaN` is NaN.
        const __nv_bfloat16* const rv = bval + b * kCap + off;
        auto rs = [&](int e) { return e < cnt ? __bfloat162float(rv[e]) : 0.f; };
        //: THE STAGING IS ONE N-TILE DEEP.  A warp's slice of `so` holds eight
        //: entries at a time and the fan-in follows immediately, which keeps the
        //: staging buffer at the shared footprint the channel carve had while the
        //: accumulator itself stays resident across the whole unit.
        float* const sow = so + warp * kMmaN * LdO;
#pragma unroll
        for (int j = 0; j < kNTokR; ++j) {
          const float r0 = rs(j * 8 + (lane & 3) * 2);
          const float r1 = rs(j * 8 + (lane & 3) * 2 + 1);
          static_for<MDV>([&](auto Mc) {
            constexpr int mt = decltype(Mc)::value;
#pragma unroll
            for (int k = 0; k < 4; ++k)
              store_f32(
                  &sow[((lane & 3) * 2 + (k & 1)) * LdO + mt * kMmaM + (lane >> 2) + 8 * (k >> 1)],
                  acc[mt][j][k] * ((k & 1) ? r1 : r0));
          });
          if (lane < 4) {
#pragma unroll
            for (int k = 0; k < 2; ++k)
              store_f32(&sow[(lane * 2 + k) * LdO + kDv], macc[j][k] * rs(j * 8 + lane * 2 + k));
          }
          __syncwarp();
          //: THE RED FAN-IN, WARP-PRIVATE.  Thirty-two lanes are thirty-two
          //: consecutive channels of one token, twice; the entry axis is the loop.
#pragma unroll
          for (int e = 0; e < kMmaN; ++e) {
            const int ent = j * kMmaN + e;
            if (ent < cnt) {
              float* dp = reinterpret_cast<float*>(
                  gather_ptr(oPlane, btok[b * kCap + off + ent] * (uint32_t)(kDv * 4) + fanColB));
              fan_in_add(dp, sow[e * LdO + lane]);
              fan_in_add(dp + 32, sow[e * LdO + lane + 32]);
            }
          }
          if (lane < kMmaN && j * kMmaN + lane < cnt)
            fan_in_add(reinterpret_cast<float*>(
                           gather_ptr(denPlane, btok[b * kCap + off + j * kMmaN + lane] * 4u)),
                       sow[lane * LdO + kDv]);
          __syncwarp();
        }
      }
    }
    //: THE RESIDUALS COMPACT TOGETHER, behind one pair of barriers for every bucket
    //: rather than a pair apiece -- and this pair is now the ONLY rendezvous the
    //: readout drain contains.
    rendezvous();
#pragma unroll 1
    for (int b = 0; b < NBR; ++b) {
      const int done = sdone[b], keep = scnt[b] - done;
      if (done > 0) {
        if (tid < kGrpR && tid < keep) {
          btok[b * kCap + tid] = btok[b * kCap + done + tid];
          bval[b * kCap + tid] = bval[b * kCap + done + tid];
        }
        if (tid == 0) scnt[b] = keep;
      }
    }
    rendezvous();
  };

  // ---- THE FOLD DRAIN: a bucket's entries fold into whole state tiles ------
  auto stage_v = [&](int d, int off, int cnt, int slot) {
#pragma unroll
    for (int p = 0; p < kVPasses; ++p) {
      const int r = ldRow + p * kVRows;
      //: a dead row is ZEROED and not gathered, for the produce's reason and one
      //: more: its A column is scaled by a zero scalar, and `0 * NaN` is NaN.
      if (r < cnt)
        stage_tile(svL + (uint32_t)slot * kVSlot + (uint32_t)(p * kVRows * LdV) * 2,
                   gather_ptr(vPlane, wtok[d * kCap + off + r] * (uint32_t)v_row_bytes
                                          + (uint32_t)(ldCol * 2)));
      else
        store_vec16(&sv[(fgrp * G::kStages + slot) * kGrp * LdV + r * LdV + ldCol],
                    uint4{0u, 0u, 0u, 0u});
    }
    stage_commit();
  };
  auto produce_write = [&](int d, int off, int cnt, int slot) {
    const bool alive = wprow < cnt;
    produce_row<WPM, D, WK>(pW, swp + (fgrp * 2 + slot) * (kGrp * LdWP) + wprow * LdWP + wslot0,
                            wcol, wtok[d * kCap + off + (alive ? wprow : 0)] * kWtotB,
                            alive ? kOneSplat : 0u);
  };
  auto drain_write = [&](bool flush) {
    //: THE IMAGE DIES BEFORE THE FOLD TOUCHES ITS BYTES, not after: the value ring
    //: and the write panel are the phase union's other tenant, so the flag has to
    //: fall before the first write and not on the last mma.  It falls HERE, on
    //: every thread, because the warps no longer walk the same buckets and
    //: `ensure_image`'s rendezvous is CTA-wide: a flag some warps set and others
    //: did not would be a deadlock, not a stale image.
    img_valid = false;
#pragma unroll 1
    for (int d = 0; d < NBW; ++d) {
      //: A WARP DRAINS THE BUCKETS IT OWNS LEAVES IN.  When a bucket is narrower
      //: than a warp's run that is exactly one warp, and the four warps drain four
      //: different buckets CONCURRENTLY with nothing shared; when the bucket is the
      //: whole box every warp is a consumer and the set is the CTA.
      if constexpr (PW < kBCW) {
        if ((d * PW) / kBCW != warp) continue;
      }
      const int n = wcnt[d];
      const int groups = flush ? (n + kGrp - 1) / kGrp : n / kGrp;
      if (fgtid == 0) wdone[d] = 0;
      if (groups == 0) continue;
      //: V RIDES THE WRITE BUCKETS and is gathered once per write entry, at its own
      //: bucket's stage.  The gather for the next group is issued before this
      //: group's mma block and waited after it.
      stage_v(d, 0, n < kGrp ? n : kGrp, 0);
      produce_write(d, 0, n < kGrp ? n : kGrp, 0);
#pragma unroll 1
      for (int g = 0; g < groups; ++g) {
        const int off = g * kGrp;
        const int cnt = (n - off) < kGrp ? (n - off) : kGrp;
        const uint32_t vsl = aV + (uint32_t)(g & 1) * kVSlot;
        const uint32_t wp = bWP + (uint32_t)(g & 1) * kWpSlot;
        const uint32_t wp2 = bWP2 + (uint32_t)(g & 1) * kWpSlot;
        stage_wait<0>();
        rendezvous_group(fbar, kFoldThr);
        if (g + 1 < groups) {
          const int noff = off + kGrp;
          stage_v(d, noff, (n - noff) < kGrp ? (n - noff) : kGrp, (g + 1) & 1);
        }
#pragma unroll
        for (int kk = 0; kk < kMTok; ++kk) {
          //: R4: THE BUCKET SCALAR RIDES THE TOKEN AXIS of the A operand.  The panel
          //: stays the UNGAINED dense-run Kronecker and `w[d] * gw[t]` multiplies the
          //: entry's V column -- four packed multiplies per channel tile, not `d_v`
          //: of them.
          //:
          //: A PADDING ENTRY'S SCALAR IS READ AS ZERO, and the zero PANEL row it also
          //: has is not enough: an mma sums `A[m,k] * B[k,n]` over `k`, so a NaN in
          //: `A` meets `B = 0` and poisons the accumulator rather than vanishing from
          //: it.  The slot past the live count is unwritten shared memory -- the phase
          //: union hands it whatever the other phase left -- so the bound is read
          //: here, where the operand is formed.
          const int e0 = kk * kMmaM + (lane & 3) * 2;
          const unsigned short* const bv =
              reinterpret_cast<const unsigned short*>(wval + d * kCap + off);
          auto sc = [&](int e) { return e < cnt ? (uint32_t)bv[e] : 0u; };
          const uint32_t s01 = (sc(e0 + 1) << 16) | sc(e0);
          const uint32_t s89 = (sc(e0 + 9) << 16) | sc(e0 + 8);
          //: THE WARP HOLDS EVERY CHANNEL OF ITS OWN LEAVES, so the fold's `A` walks
          //: the channel m-tiles.  Loaded and scaled ONCE per k-step and reused by
          //: every leaf tile the bucket gives this warp -- which is why the extra
          //: fragments are not extra traffic: the warp's LEAF extent fell by the same
          //: factor its CHANNEL extent grew.
          uint32_t av[MDV][4];
          static_for<MDV>([&](auto Mc) {
            constexpr int mt = decltype(Mc)::value;
            load_frag_t(av[mt], ADDR_V(vsl, kk * kMmaM) + (uint32_t)(mt * kMmaM) * 2);
            av[mt][0] = mul_bf16x2(av[mt][0], s01);
            av[mt][1] = mul_bf16x2(av[mt][1], s01);
            av[mt][2] = mul_bf16x2(av[mt][2], s89);
            av[mt][3] = mul_bf16x2(av[mt][3], s89);
          });
          //: THE MASS OPERAND IS THE SAME SCALAR WITHOUT THE VALUE.  The mass column
          //: is the ones column of `V`, so its A fragment is the entry scalar in row
          //: zero and nothing elsewhere -- what was a literal `1` back when the PANEL
          //: carried the gain.  Moving the scalar off the panel moves it here too, or
          //: the mass would be the one channel that lost it.
          uint32_t mass_a[4];
          mass_a[0] = (lane >> 2) == 0 ? s01 : 0u;
          mass_a[1] = 0u;
          mass_a[2] = (lane >> 2) == 0 ? s89 : 0u;
          mass_a[3] = 0u;
          //: THE TARGET TILES ARE NAMED SCALARS.  A state fragment is a register
          //: array, so a bucket's tile index must be a constant expression: the
          //: dispatch is a case over the built bucket count, never a dynamic index.
          static_for<NBW>([&](auto Db) {
            constexpr int dd = decltype(Db)::value;
            if (dd != d) return;
            //: THE BUCKET'S OWNING WARP, and the ONE STATEMENT that covers both
            //: structures.  A fold bucket is a contiguous run of `PW` leaves.  When
            //: that run is NARROWER than a warp's `kBCW` it lies inside exactly one
            //: warp, which folds it alone; when it is the WHOLE BOX (a side with no
            //: key) every warp folds its own quarter of it.  Both read: *a warp folds
            //: the leaves it owns, for every bucket that intersects them.*  No branch
            //: on a structure, no second path -- `kOwn`, `kNT` and `kT0` are
            //: constant expressions of the bucket map alone.
            constexpr int kFirst = dd * PW;                       // the bucket's first leaf
            constexpr bool kWhole = PW >= kBCW;                   // it spans every warp
            constexpr int kOwn = kWhole ? -1 : kFirst / kBCW;     // its warp, or all of them
            constexpr int kNT = (PW < kBCW ? PW : kBCW) / kMmaN;  // tiles THIS warp folds
            constexpr int kT0 = kWhole ? 0 : (kFirst % kBCW) / kMmaN;
            static_assert(kWhole || PW % kBCW == 0 || kBCW % PW == 0,
                          "a fold bucket must nest in a warp's run or contain it whole");
            if (kOwn >= 0 && warp != kOwn) return;
            //: the panel column the warp's first tile reads: its own leaf origin when
            //: the panel spans the box, zero when the panel IS the bucket.
            const int pcol0 = kWhole ? leaf0 : 0;
            if constexpr (kNT == 1) {
              uint32_t b2[2];
              load_frag_pair_t(b2, ADDR_WP(wp2, kk * kMmaM, pcol0));
              static_for<MDV>([&](auto Mc) {
                constexpr int mt = decltype(Mc)::value;
                mma(st[mt][kT0], av[mt], b2);
              });
              //: the mass rides the warp's own tile UNCONDITIONALLY now -- the
              //: `t0 % kWarps == warp` predicate was the channel carve's way of
              //: naming a mass owner, and the ownership carve has already named one.
              mma(mass_st[kT0], mass_a, b2);
            } else {
              static_for<kNT / 2>([&](auto Qc) {
                constexpr int q = decltype(Qc)::value;
                uint32_t bb[4];
                load_frag_t(bb, ADDR_WP(wp, kk * kMmaM, pcol0 + q * 2 * kMmaN));
                static_for<MDV>([&](auto Mc) {
                  constexpr int mt = decltype(Mc)::value;
                  mma(st[mt][kT0 + 2 * q], av[mt], &bb[0]);
                  mma(st[mt][kT0 + 2 * q + 1], av[mt], &bb[2]);
                });
                mma(mass_st[kT0 + 2 * q], mass_a, &bb[0]);
                mma(mass_st[kT0 + 2 * q + 1], mass_a, &bb[2]);
              });
            }
          });
        }
        if (g + 1 < groups) {
          const int noff = off + kGrp;
          produce_write(d, noff, (n - noff) < kGrp ? (n - noff) : kGrp, (g + 1) & 1);
        }
        rendezvous_group(fbar, kFoldThr);
      }
      if (fgtid == 0) wdone[d] = groups * kGrp < n ? groups * kGrp : n;
    }
    //: the compaction is the ONE place the whole CTA meets again, and it must:
    //: every bucket's residual is compacted here, whichever warp owns it.
    rendezvous();
#pragma unroll 1
    for (int d = 0; d < NBW; ++d) {
      const int done = wdone[d], keep = wcnt[d] - done;
      if (done > 0) {
        if (tid < kGrp && tid < keep) {
          wtok[d * kCap + tid] = wtok[d * kCap + done + tid];
          wval[d * kCap + tid] = wval[d * kCap + done + tid];
        }
        if (tid == 0) wcnt[d] = keep;
      }
    }
    rendezvous();
  };

  // ---- THE PER-DIGIT SCATTER ----------------------------------------------
  //: The walk already reads the key level's amplitude run at the liveness scan, so
  //: the bucket lists cost that read and a ballot per digit.  The rank is a TWO
  //: LEVEL SCAN (lane ballot, then one thread over the warps) rather than an atomic
  //: counter, so a bucket's entry order is a function of the token list alone and a
  //: chained state stays bit-identical run to run.
  int tok0 = 0;
  auto scatter_side = [&](auto KeyC, auto NbC, const __nv_bfloat16* amp, const uint16_t* list,
                          int c0, int n, int* cnt, uint32_t* tokArr, __nv_bfloat16* valArr,
                          bool gained) {
    constexpr int KL = decltype(KeyC)::value;
    constexpr int NB = decltype(NbC)::value;
    constexpr int KS = (KL < 0) ? 1 : BP::s(KL);
    //: ONE WARP APPENDS.  Its lanes hold one lockstep unit of the token list, so a
    //: digit's ballot gives both the count and every entry's RANK inside it, against
    //: a base only lane zero advances -- no cross-warp scan, no atomic, no barrier
    //: inside, and an order that is a function of the token list alone.
    if (warp == 0) {
#pragma unroll 1
      for (int u = 0; u < G::kChunk; u += kBallotLanes) {
        const int idx = c0 + u + lane;
        const bool has = idx < n;
        const uint32_t gtok = has ? (uint32_t)(tok0 + (int)list[idx]) : 0u;
        unsigned short krun[KS];
#pragma unroll
        for (int i = 0; i < KS; ++i) krun[i] = 0;
        if constexpr (KL >= 0) {
          if (has) load_run<KS>(krun, gather_ptr(amp, gtok * kWtotB + (uint32_t)(runCol[KL] * 2)));
        }
        uint32_t gain = kOneSplat;
        if (gained && has)
          gain = splat_bf16x2(*reinterpret_cast<const __nv_bfloat16*>(gather_ptr(gW, gtok * 2u)));
        static_for<NB>([&](auto Db) {
          constexpr int d = decltype(Db)::value;
          const bool live = has && ((KL < 0) || ((krun[d] & 0x7FFFu) != 0u));
          const uint32_t vote = lane_ballot(live);
          const int base = cnt[d];
          if (live) {
            const int pos = base + bit_count(vote & below);
            tokArr[d * kCap + pos] = gtok;
            //: the stored scalar is the KEY DIGIT's coefficient, times the write gain
            //: on the write side.  The panel never sees either.
            const uint32_t kv =
                (KL < 0) ? kOneSplat : (((uint32_t)krun[d] << 16) | (uint32_t)krun[d]);
            *reinterpret_cast<unsigned short*>(&valArr[d * kCap + pos]) =
                (unsigned short)((gained ? mul_bf16x2(kv, gain) : kv) & 0xFFFFu);
          }
          if (lane == 0) cnt[d] = base + bit_count(vote);
        });
      }
    }
    rendezvous();
  };

  auto pending = [&](const int* cnt, int nb) {
    int p = 0;
    for (int i = 0; i < nb; ++i) p += cnt[i];
    return p;
  };
  auto flush_read = [&]() {
    if (pending(scnt, NBR) == 0) return;
    ensure_image();
    drain_read(true);
  };
  auto flush_write = [&]() {
    if (pending(wcnt, NBW) == 0) return;
    drain_write(true);
  };

  // ---- the GLOBAL window grid ---------------------------------------------
  const int nWindows = (L + W - 1) / W;
  for (int e = tid; e < 2 * (NBR + NBW); e += kThreads) scnt[e] = 0;
  if (tid < 2) sscan[8 + tid] = 0;
  rendezvous();

#pragma unroll 1
  for (int win = 0; win < nWindows; ++win) {
    tok0 = win * W;
    const int wLen = (L - tok0) < W ? (L - tok0) : W;

    // ---- the liveness pass: TWO lists, one per SIDE -----------------------
    //: THE SCHEDULE IS TWO-SIDED.  Per level at most ONE side is sparse, so there
    //: is one stored row set per level -- but the two sides' CLAUSE SETS are
    //: different level subsets, and a dense side's clause is absent-means-full.
    //: ANDing each side's OWN rows IS that conjunction over exactly the levels
    //: where that side is sparse: a dense level's ballot word is all ones for
    //: every token and drops out on its own, with no structure fact carried into
    //: the kernel.  A warp's thirty-two tokens are one word of the owner's row.
    if (tid < 2) sscan[8 + tid] = 0;
    rendezvous();
#pragma unroll 1
    for (int p0 = 0; p0 < W; p0 += kThreads) {
      const int loc = p0 + tid;
      bool inR = loc < wLen, inW = loc < wLen;
      if (loc < wLen) {
        const uint32_t wordB = (uint32_t)(((tok0 + p0) >> 5) + warp) * (uint32_t)sizeof(uint32_t);
        uint32_t mR = 0xFFFFFFFFu, mW = 0xFFFFFFFFu;
        static_for<D>([&](auto Lc) {
          constexpr int l = decltype(Lc)::value;
          if constexpr (BP::row_g(l) > 0) {
            mR &= *reinterpret_cast<const uint32_t*>(gather_ptr(rowsR, rowOff[l] + wordB));
            mW &= *reinterpret_cast<const uint32_t*>(gather_ptr(rowsW, rowOff[l] + wordB));
          }
        });
        inR = ((mR >> lane) & 1u) != 0u;
        inW = ((mW >> lane) & 1u) != 0u;
      }
      const uint32_t voteR = lane_ballot(inR), voteW = lane_ballot(inW);
      const int mineR = bit_count(voteR & below), mineW = bit_count(voteW & below);
      if (lane == 0) {
        sscan[warp] = bit_count(voteR);
        sscan[kWarps + warp] = bit_count(voteW);
      }
      rendezvous();
      if (tid == 0) {
        int runR = sscan[8], runW = sscan[9];
#pragma unroll
        for (int u = 0; u < kWarps; ++u) {
          const int cr = sscan[u], cw = sscan[kWarps + u];
          sscan[u] = runR;
          sscan[kWarps + u] = runW;
          runR += cr;
          runW += cw;
        }
        sscan[8] = runR;
        sscan[9] = runW;
      }
      rendezvous();
      if (inR) slist[sscan[warp] + mineR] = (uint16_t)loc;
      if (inW) slist[kListStride + sscan[kWarps + warp] + mineW] = (uint16_t)loc;
      rendezvous();
    }
    const int nr = sscan[8], nw = sscan[9];
    if (nr == 0 && nw == 0) continue;

    //: THE WINDOW-GRAIN READ-BEFORE-WRITE LAW, as a flush schedule.  Reads of this
    //: window must see the state as of its START, so every pending FOLD lands
    //: first; and pending READS were taken against a state those folds would
    //: advance, so they land before the folds.  Each phase empties the other side's
    //: buckets before filling its own, which makes "at most one side is pending"
    //: an invariant -- and cross-window batching is then exactly what happens
    //: across a run of windows that touch one side only.
    if (nr > 0) {
      flush_read();
      flush_write();
      ensure_image();
#pragma unroll 1
      for (int c0 = 0; c0 < nr; c0 += G::kChunk) {
        scatter_side(std::integral_constant<int, RK>{}, std::integral_constant<int, NBR>{}, pR,
                     slist, c0, nr, scnt, btok, bval, false);
        drain_read(false);
      }
    }
    if (nw > 0) {
      flush_read();
#pragma unroll 1
      for (int c0 = 0; c0 < nw; c0 += G::kChunk) {
        scatter_side(std::integral_constant<int, WK>{}, std::integral_constant<int, NBW>{}, pW,
                     slist + kListStride, c0, nw, wcnt, wtok, wval, true);
        drain_write(false);
      }
    }
  }
  //: THE TAIL: the residuals the batching held back, reads before folds.
  flush_read();
  flush_write();

  // ---- the FINAL STATE, in LATTICE leaf order ------------------------------
  if constexpr (STATE) {
    if (s_out != nullptr) {
      float* const plane = pin_address(s_out);
      rendezvous();
#pragma unroll
      for (int a = 0; a < kAtoms; ++a) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          if ((2 * a + p) / NDVW == warp) {
#pragma unroll
            for (int m = 0; m < MDV; ++m)
#pragma unroll
              for (int k = 0; k < 4; ++k)
                store_f32(&so[page_value(m, p, k)], st[m][(2 * a + p) % NDVW][k]);
            if (lane < 4) {
#pragma unroll
              for (int k = 0; k < 2; ++k)
                store_f32(&so[page_mass(p, k)], mass_st[(2 * a + p) % NDVW][k]);
            }
          }
        }
        rendezvous();
        page_sweep(page_off(atomBase + (uint32_t)a), [&](int sidx, uint32_t offb) {
          store_f32(reinterpret_cast<float*>(gather_ptr(plane, offb)), so[sidx]);
        });
        rendezvous();
      }
    }
  }
#undef ADDR_RP
#undef ADDR_WP
#undef ADDR_V
}

}  // namespace rola::carry
