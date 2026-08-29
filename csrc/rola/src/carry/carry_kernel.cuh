// csrc/rola/src/carry/carry_kernel.cuh -- K31 RUNG R1: THE CARRY KERNEL'S CARVED BODY.
//
// WHAT IT COMPUTES.  The INTER (carried-state) term only:
//
//     O^T[:, t] += S^T[:, box] . R~^T[box, t]      readout, vs state-at-window-start
//     S^T[:, box] += V~^T[:, t] . W~[t, box]       fold, of the window's own writes
//
// where `R~`/`W~` are the read/write leaf coefficients restricted to the owner's
// box, `V~` is the value row with the ones column appended (the mass channel), and
// the window grid is GLOBAL and FIXED at `W` tokens -- identical for every owner.
// The within-window (INTRA) pairs are NOT computed here.
//
// SYMBOLS, each at first use: `D` levels, `B` digits per level, `K` the uniform
// sub-box span, `M` the capacity multiplier, `BC` the leaves one owner holds,
// `W` the window length in tokens, `C` the private batch's token tile, `DV` (`d_v`)
// the value channel count, `MO` the per-level mode declaration, `NSR`/`NSW` the two
// sides' stream counts, `WPS` warps per write stream, `PK` the parked-`lo`
// declaration.  `k_tok` (a token's nonzero digits per level) appears NOWHERE in this
// file, and that is the point: the schedule is computed, never branched on.
//
// THE FORM IS M1a's (K35 M1P, Blake ruled 2026-08-28).  Leaf streams in ONE order --
// the READ side's, derived by `StreamSpans` from `MO` and `NSR` -- walked by both
// phases; a per-warp walk-my-own-bits schedule; per-warp staging into a two-slot
// panel ring with in-place expansion; and the readout's fan-in issued as native
// global RED out of a per-warp staging panel.  There is no two-sided carve, no
// exchange or transit, no cooperative fill, no fan-in ring and no flow counter: each
// of those is a later stage's single addition, and the arm list carries only the
// declarations this shape serves (the dispatch refuses the rest by name).
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

static constexpr int kPadE = 8;  // bf16 row padding, elements (16 B)
static constexpr int kPadF = 4;  // fp32 row padding, elements (16 B)
//: `d_v` IS AN ARM PARAMETER AND NOT A FILE CONSTANT (K35 S3b).  Both FORWARD bodies
//: take it as `DV` -- `CarryGeom`/`SplitGeom` name it `kDv` and derive `kCols = DV + 1`,
//: the mass column that rides the state.  THE REVERSE FAMILY IS OUT OF SCOPE and keeps
//: its own `kDv = 64` (`backward_kernel.cuh`).  The CHANNEL-SPLIT body's warp count and
//: the family's build stamp live in `box.cuh` with the rest of the arm-independent
//: geometry (K36), so the DISPATCH TU can read them without parsing this header.
static_assert(kAtomLeaves == kMmaK, "the page granule is one MMA k-step");

//: A COMPILE-TIME LOOP.  Every per-level quantity here is a template argument or a
//: register-array index, so the level index must be a constant expression; a
//: `for (int l = 0; l < D; ++l)` would make `load_run<n_l(l)>` ill-formed and would
//: turn every per-level array index dynamic, which is a stack frame.
template <int N, class F>
__device__ __forceinline__ void static_for(F&& f) {
  if constexpr (N > 0) {
    static_for<N - 1>(f);
    f(std::integral_constant<int, N - 1>{});
  }
}

//: THE CARVE.  A STREAM owns an affine RECTANGLE of the owner's digits -- a per-level
//: span vector, `prod_l S_l` leaves and ALL `kCols` channels -- and tiles those leaves
//: in a private mixed-radix order.  That tiling is private to the stream: the lattice's
//: `local` order, the state plane, decode and the facts geometry never see it, so the
//: readout needs no state image, no `ldmatrix` for A and no CTA barrier.
//:
//: ONE ORDER, THE READ SIDE'S (K35 M1P).  M1a's body carves the box ONCE and both
//: phases walk that carve, so `SVW` IS `SVR` here: the write side inherits the read
//: side's vector, the leaf -> (stream, slot) map is the same map in both phases, and
//: there is nothing to publish between them.  The two-sided carve, its transit and its
//: exchange are a later stage's addition; an arm that declares a second stream count or
//: a column split is REFUSED at compile time below rather than served this shape.
//: ONE WARP PER STREAM: the executor map's `WPS` dial returns with the containers.

template <int D, int B, int K, int M, int DV, int W, int C, int WPS, int NSW, int NSR, int PK,
          uint32_t MO>
struct CarryGeom {
  using BP = BoxPlan<D, B, K, M>;
  using SVR = StreamSpans<D, B, K, M, MO, kSideRead, NSR>;
  using SVW = SVR;
  static_assert(WPS == 1,
                "M1P builds ONE warp per stream; the column split (`WPS > 1`) returns with "
                "the containers, as its own stage");
  static_assert(NSW == NSR,
                "M1P carves the box ONCE, in the read side's order, and the write side walks "
                "that carve; a second stream count has no body here");
  static_assert(PK == 0, "the parked `lo` is a tenant of the exchange, which this body has not");
  static constexpr int BC = BP::kBC;
  //: THE VALUE WIDTH, AS A DECLARATION.  `kDv` channels plus the MASS column, which
  //: rides the state as one more row of every plane the state touches.
  static constexpr int kDv = DV;
  static constexpr int kCols = DV + 1;
  static_assert(DV == 64,
                "M1a's readout stages exactly `kDv` channels into the batch's `C` panel rows; "
                "the `d_v` dial returns as its own stage and is never a runtime test");

  //: THE EXECUTOR MAP: one warp per stream, and the batch tile IS the executor count
  //: times an accumulator n-tile -- which is what makes the readout's staging block one
  //: token tile of the batch AND a whole state page.
  static constexpr int kStreams = SVR::kStreams;
  static constexpr int kWarps = kStreams;
  static constexpr int kThreads = kWarps * 32;
  //: the stream's leaves, and the private panel row that indexes them.
  static constexpr int kLeafW = SVR::kLeaf;
  static constexpr int LdV = kDv + kPadE;
  static constexpr int LdS = kLeafW + kPadE;
  static constexpr int LdO = kDv + kPadF;

  //: THE TWO PANELS ARE A TWO-SLOT RING, AND THEY COST NOTHING EXTRA.  A window's
  //: readouts all precede its folds, so the write panel is dead through phase 1 and
  //: the read panel is dead through phase 2: each phase double-buffers in the
  //: other's buffer and produces one batch AHEAD, which is what puts the produce's
  //: global-load latency underneath the previous batch's MMA block.
  static constexpr int panel_bytes = 2 * kWarps * C * LdS * (int)sizeof(__nv_bfloat16);
  //: the readout's staging block, one `kMmaN`-token region per warp -- and, at
  //: entry and exit, the state plane's page.
  static constexpr int out_bytes = C * LdO * (int)sizeof(float);
  static_assert(kWarps * kMmaN == C && C * LdO >= kAtomLeaves * LdO,
                "the staging block is one token tile of the batch and holds a whole page");
  //: `kWarps` READ lists, one per stream, then ONE write list: a read entry belongs to
  //: the streams its carved digits fall in, a write entry to all of them.
  static constexpr int kLists = kWarps + 1;
  static constexpr int kListStride = ((W + 7) / 8) * 8;
  static constexpr int list_bytes = kLists * kListStride * (int)sizeof(uint16_t);
  //: THE WINDOW'S BALLOT WORDS, one per lane of the walk -- so a window may be at
  //: most a lockstep unit's worth of words wide.
  static constexpr int kLW = W / kBallotLanes;
  //: THE V RING'S DEPTH IS DERIVED FROM THE SMEM BUDGET AND THE REGISTER FLOOR: the
  //: shallowest ring that reaches the best residency this arm can reach at any depth.
  //: `kRegFloor` is the DECLARED floor for this family -- the measured census of the
  //: heaviest built arm, re-measured at every milestone.  A CTA target the live set
  //: cannot honour is a spill: the carve holds `kLeafW` leaves x all `kCols` channels
  //: per warp, and this is what that costs.
  static constexpr int kRegFloor = 220;
  static constexpr int slot_bytes = C * LdV * (int)sizeof(__nv_bfloat16);
  static constexpr int fixed_bytes = panel_bytes + out_bytes + list_bytes;
  static constexpr int kStages = design::stages_for(slot_bytes, fixed_bytes, kThreads, kRegFloor);
  static constexpr size_t bytes = (size_t)fixed_bytes + (size_t)kStages * slot_bytes;
  static constexpr int cta_target = design::cta_target((int)bytes, kThreads, kRegFloor);
  static_assert(design::smem_fits(bytes), "this arm does not fit the SMEM maximum");
};

//: THE OWNER-GRAIN PRODUCE MAP, which the channel-split body and the reverse family's
//: panel are expanded against (`carry_split_kernel.cuh`, `backward_kernel.cuh`): a row's
//: `kRowThr` threads split the box's `BC` coefficients into `SPT` contiguous LATTICE
//: slots each.  Because the lattice index is a bit permutation (R0 spec section 2.4), a
//: thread's slot run is a sub-lattice: per level the digits it touches are a CONTIGUOUS
//: run of the owner's own run at a runtime base, and WHICH element of that run each of
//: its slots wants is a compile-time function of the unrolled slot index.
template <int D, int B, int K, int M, int C>
struct ProduceMap {
  using BP = BoxPlan<D, B, K, M>;
  static constexpr int kRowThr = kThreads / C;
  static constexpr int SPT = BP::kBC / kRowThr;
  static constexpr int kJBits = ilog2c(SPT);
  static constexpr int vary(int l) { return SPT >> BP::run_shift(l); }
  static constexpr int n_l(int l) {
    return vary(l) < 1 ? 1 : (vary(l) > BP::s(l) ? BP::s(l) : vary(l));
  }
  static constexpr int max_n(int l) {
    return l == D ? 1 : (n_l(l) > max_n(l + 1) ? n_l(l) : max_n(l + 1));
  }
  static constexpr int kMaxN = max_n(0);
  static constexpr int idx_of(int l, int j) { return (j >> BP::run_shift(l)) & (n_l(l) - 1); }
};

//: THE PRIVATE LEAF ORDER.  A stream's leaves are numbered by a MIXED RADIX over its
//: own per-level spans -- `p = sum_l r_l << shift(l)`, with `r_l` the digit's offset
//: inside the stream's range at level `l` -- so `p` is a bit permutation of the levels'
//: digit offsets exactly as `local` is, and the expansion is `sum_l S_l` amplitude loads
//: followed by products of register values at CONSTANT indices: no dynamic indexing, no
//: spill, no materialized gate plane.  A span-1 level contributes a constant and drops
//: out of the order on its own.
//:
//: THE PRIVATE-ORDER RULE IS THE ORDER BY MODE (P76, ruled 2026-08-26; K35 S1b), fixed
//: by the DECLARATION alone -- no span is consulted: DENSE innermost, then SPARSE, then
//: BOTH_SPARSE, lattice order inside a class, and NO LEVEL IS PINNED.  `ModeOrder`
//: (box.cuh) states the rule and `StreamSpans` derives the carve from it, which is why
//: nothing here can drift from what was cut.
template <int D, int B, int K, int M, class SV>
struct SpanMap {
  using BP = BoxPlan<D, B, K, M>;
  static constexpr int kLeaf = SV::kLeaf;
  static constexpr int span(int l) { return SV::s(l); }
  static constexpr int bits(int l) { return ilog2c(span(l)); }
  static constexpr int rank(int l) { return SV::Ord::rank(l); }
  static constexpr int shift_from(int l, int i) {
    return i >= D ? 0 : (rank(i) < rank(l) ? bits(i) : 0) + shift_from(l, i + 1);
  }
  static constexpr int shift(int l) { return shift_from(l, 0); }
  //: THE PACKED PAIR IS SIMPLY THE INNERMOST LEVEL OF THE ORDER.  A lane mints two
  //: adjacent private slots at once, and they differ in the digit of whichever level
  //: lies lowest in `p` -- so that level's two amplitudes are adjacent in the run and
  //: one four-byte access covers the pair.  Levels of span one occupy no bits and their
  //: position is immaterial, so the level wanted is the lowest one with a span of at
  //: least two; a box holds at least two leaves, so exactly one level qualifies.
  static constexpr int pair_from(int l) {
    return l >= D ? -1 : ((shift(l) == 0 && span(l) >= 2) ? l : pair_from(l + 1));
  }
  static constexpr int kPairLevel = pair_from(0);
  //: THE DEAD ROW'S LEVEL.  A padding row of a partial batch has no token, and a zero
  //: coefficient contributes exactly nothing to either MMA (the produce's dead-row
  //: zeroing law) -- so zeroing ONE level's run zeroes the whole row, because every
  //: private slot multiplies exactly one element of every level's run.  The narrowest
  //: run is the cheapest place to do it, and it is never the pair level, whose run is
  //: consumed by the packed seed before any mask could apply.
  static constexpr int narrow_from(int l, int best) {
    return l >= D ? best
                  : narrow_from(l + 1,
                                (l != kPairLevel && (best < 0 || span(l) < span(best))) ? l : best);
  }
  static constexpr int kZeroLevel = narrow_from(0, -1);
  static_assert(kLeaf >= 2 && kPairLevel >= 0 && span(kPairLevel) >= 2,
                "a box has at least two leaves, so exactly one level lies lowest in the "
                "private order with a span of at least two and supplies the packed pair");
  static_assert(D >= 2 && kZeroLevel >= 0 && kZeroLevel != kPairLevel,
                "the dead-row mask needs a level other than the pair level to zero, which "
                "is a level count of at least two");
};

//: a contiguous run of `n` bf16, at the widest access that covers it.  The runs are
//: naturally aligned by construction (a level's column base is a multiple of `B`, an
//: owner's run starts at a multiple of `s_l` and a stream's sub-run at a multiple of
//: its own width), so this is one, two or four accesses.
template <int n>
__device__ __forceinline__ void load_run(unsigned short (&dst)[n], const void* p) {
  static_assert(n == 1 || n == 2 || n == 4 || n == 8 || n == 16,
                "an admissible (k, m) run is 1, 2, 4, 8 or 16 wide");
  if constexpr (n == 1) {
    dst[0] = *reinterpret_cast<const unsigned short*>(p);
  } else if constexpr (n == 2) {
    *reinterpret_cast<unsigned int*>(dst) = *reinterpret_cast<const unsigned int*>(p);
  } else if constexpr (n == 4) {
    *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(p);
  } else {
#pragma unroll
    for (int i = 0; i < n / 8; ++i)
      *reinterpret_cast<uint4*>(dst + i * 8) =
          load_vec16(reinterpret_cast<const char*>(p) + i * 16);
  }
}

//: THE SECOND LAUNCH BOUND: the CTA count this arm can actually reach, taken from the
//: same geometry the ring depth is.  A ring sized for a residency the launch bound does
//: not ask for is a ring sized for nothing.
template <int D, int B, int K, int M, int DV, int W, int C, int WPS, int NSW, int NSR, int PK,
          uint32_t MO>
constexpr int kCarryCtaTarget = CarryGeom<D, B, K, M, DV, W, C, WPS, NSW, NSR, PK, MO>::cta_target;

// ---------------------------------------------------------------- the kernel

template <int D, int B, int K, int M, int DV, int W, int C, int WPS, int NSW, int NSR, int PK,
          uint32_t MO, bool STATE>
__global__ __launch_bounds__((CarryGeom<D, B, K, M, DV, W, C, WPS, NSW, NSR, PK, MO>::kThreads),
                             (kCarryCtaTarget<D, B, K, M, DV, W, C, WPS, NSW, NSR, PK, MO>)) void
carry_kernel(const __nv_bfloat16* __restrict__ pread, const __nv_bfloat16* __restrict__ pwrite,
             const __nv_bfloat16* __restrict__ gwrite, const __nv_bfloat16* __restrict__ v,
             const uint32_t* __restrict__ rows, float* __restrict__ o, float* __restrict__ den,
             int L, int tok_words, int v_row_bytes, const float* __restrict__ s_in,
             float* __restrict__ s_out, const int32_t* __restrict__ page_tbl) {
  using BP = BoxPlan<D, B, K, M>;
  using G = CarryGeom<D, B, K, M, DV, W, C, WPS, NSW, NSR, PK, MO>;
  constexpr int kDv = DV;
  constexpr int kCols = DV + 1;
  //: THE ONE SPAN VECTOR, and the private order and the ballot rows it induces.  Both
  //: phases read these: the carve is the read side's and the write side walks it.
  using SV = typename G::SVR;
  using PM = SpanMap<D, B, K, M, SV>;
  using CR = CarveRows<SV>;
  constexpr int kWarps = G::kWarps;
  constexpr int kThreads = G::kThreads;

  constexpr int BC = BP::kBC;
  constexpr int LdV = G::LdV, LdS = G::LdS, LdO = G::LdO;
  constexpr int MTOK = C / kMmaM;  // token m-tiles of a batch
  //: the stream's own leaves: `NL` accumulator n-tiles, `KDw` readout k-steps.
  constexpr int kLeafW = G::kLeafW;
  constexpr int NL = kLeafW / kMmaN;
  constexpr int KDw = kLeafW / kMmaK;
  constexpr int MCH = kDv / kMmaM;  // the channel m-tiles, ALL of them per warp

  static_assert(C % kMmaM == 0 && kThreads % C == 0, "the batch tile must divide");
  static_assert(kLeafW % kMmaK == 0, "a stream's leaves are a whole number of k-tiles");
  static_assert(BC % kAtomLeaves == 0, "K2: an owner holds whole page atoms");
  static_assert(LdS % 8 == 0 && LdV % 8 == 0, "an ldmatrix row address must be 16-byte aligned");
  static_assert(kBallotLanes % kWarps == 0,
                "the write list gives each writer a fixed slice of every ballot word");
  static_assert(W % kBallotLanes == 0 && kThreads % kBallotLanes == 0,
                "a warp's tokens must be exactly one ballot word of the owner's row");

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int owner = (int)blockIdx.x;
  const int bh = (int)blockIdx.y;

  extern __shared__ __align__(16) char carry_smem[];
  constexpr int STAGES = G::kStages;
  static_assert(STAGES >= 2, "the wait/issue order needs at least two stages");
  __nv_bfloat16* const sv = reinterpret_cast<__nv_bfloat16*>(carry_smem);
  __nv_bfloat16* const sr = sv + STAGES * C * LdV;
  __nv_bfloat16* const sw = sr + kWarps * C * LdS;
  float* const so = reinterpret_cast<float*>(sw + kWarps * C * LdS);
  constexpr int kListStride = G::kListStride;
  uint16_t* const slist = reinterpret_cast<uint16_t*>(so + C * LdO);

  //: ONE 64-BIT BASE PER PLANE, and a 32-BIT BYTE DISPLACEMENT after it -- the
  //: shipped addressing discipline (docs/internals/carry/carry_kernel.md#state-addressing).
  const int vRowElems = v_row_bytes / (int)sizeof(__nv_bfloat16);
  const int heads = vRowElems / kDv;
  const __nv_bfloat16* const vPlane =
      pin_address(v + (long)(bh / heads) * L * vRowElems + (long)(bh % heads) * kDv);
  const __nv_bfloat16* const pR = pin_address(pread + (long)bh * L * BP::kWtot);
  const __nv_bfloat16* const pW = pin_address(pwrite + (long)bh * L * BP::kWtot);
  const __nv_bfloat16* const gW = pin_address(gwrite + (long)bh * L);
  float* const oPlane = pin_address(o + (long)bh * L * kDv);
  float* const denPlane = pin_address(den + (long)bh * L);

  //: THE STREAM ROTATION.  Two CTAs co-resident on an SM are one grid wave apart and
  //: the wave is a multiple of `kWarps`, so `owner % kWarps` would rotate both
  //: identically; the wave-stride form does not (K33 1.45).
  const int strm = (warp + ((owner + (owner >> 5)) & (kWarps - 1))) & (kWarps - 1);

  // ---- the owner's digit runs, one per level -------------------------------
  //: `o_l` is a bit-field extract of the owner index: the lattice's owner
  //: coordinates are the top `sum_l log2 g_l` bits of `lambda` (R0 spec section 2.4).
  int runCol[D];
  //: AND THE STREAM'S OWN DIGIT OFFSET INSIDE THAT RUN, per level: the stream index
  //: decomposes in MIXED RADIX over the per-level stream grids, so a multi-level carve
  //: is `grp_l * S_l` at EVERY carved level and nothing changes at the levels the carve
  //: spans whole.  Computed ONCE per warp, out of shifts and masks over compile-time
  //: powers of two.
  int off[D];
  //: AND THE BALLOT ROW BASE, per level: the SAME `o_l`, addressing the FIRST row the
  //: facts pass voted for this owner's run at this level (K31-i-d) at the CARVE's grain,
  //: so a stream's own range is ONE row per carved level and the owner's word is the OR
  //: over the streams.  The liveness pass reads these words instead of re-reading the
  //: amplitude runs itself.
  uint32_t rowOff[D];
  const uint32_t rowBytes = (uint32_t)tok_words * (uint32_t)sizeof(uint32_t);
  static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    const int o_l = (owner / BP::owner_div(l)) % BP::g(l);
    runCol[l] = BP::col_base(l) + o_l * BP::s(l);
    off[l] = ((strm >> ilog2c(SV::grid_div(l))) & (SV::g(l) - 1)) * SV::s(l);
    rowOff[l] = (uint32_t)(CR::row_prefix(l) + o_l * CR::rows_of(BP::s(l), l)) * rowBytes;
  });
  //: ONE 64-BIT BASE PER SIDE, and 32-bit byte displacements after it: the READ block
  //: and then the WRITE block, each `kRows * tok_words * 4` bytes, which is 32-bit at
  //: every admitted length.  Both sides are voted at the SAME partition here, because
  //: the body carves once (`SVW` is `SVR`).
  const uint32_t* const rowsR =
      pin_address(rows + (long)bh * (long)(2 * CR::kRows) * (long)tok_words);
  const uint32_t* const rowsW =
      pin_address(rows + ((long)bh * (long)(2 * CR::kRows) + (long)CR::kRows) * (long)tok_words);

  // ---- lane-invariant operand addressing (the shipped idiom) ---------------
  const int lr = lane & 7, tl = lane >> 3;
  const int rF = (tl & 1) * 8 + lr, cF = tl >> 1;
  const int rT = (tl >> 1) * 8 + lr, cT = tl & 1;
  //: THE PANEL RING'S TWO BASES, AS NAMED SCALARS.  A two-element array indexed by
  //: the batch parity would be a runtime index into a register array, which is a
  //: stack frame; a select is one instruction and no memory.
  __nv_bfloat16* const panR = sr + warp * C * LdS;
  __nv_bfloat16* const panW = sw + warp * C * LdS;
  const uint32_t brT0 = smem_addr(panR) + (uint32_t)((rT * LdS + cT * 8) * 2);
  const uint32_t brT1 = smem_addr(panW) + (uint32_t)((rT * LdS + cT * 8) * 2);
  const uint32_t bwF0 = smem_addr(panW) + (uint32_t)((rF * LdS + cF * 8) * 2);
  const uint32_t bwF1 = smem_addr(panR) + (uint32_t)((rF * LdS + cF * 8) * 2);
  const uint32_t svB = smem_addr(sv) + (uint32_t)((rT * LdV + cT * 8) * 2);
  constexpr uint32_t kVSlot = (uint32_t)(C * LdV) * 2;
#define ADDR_S(b_, R_, g_) ((b_) + (uint32_t)(((R_) * LdS + (g_) * 8) * 2))
#define ADDR_V(b_, R_, dv_) ((b_) + (uint32_t)(((R_) * LdV + (dv_)) * 2))

  //: the warp's own staging region of the readout block.
  float* const sow = so + warp * kMmaN * LdO;

  // ---- the state, transposed: m = channels, n = this warp's leaves ---------
  float st[MCH][NL][4];
#pragma unroll
  for (int m = 0; m < MCH; ++m)
#pragma unroll
    for (int j = 0; j < NL; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) st[m][j][e] = 0.f;
  float mass_st[NL][4];
#pragma unroll
  for (int j = 0; j < NL; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) mass_st[j][e] = 0.f;

  //: THE ONES OPERAND, AN IMMEDIATE: the mass channel's value is literally 1, so
  //: the fold's A fragment for it is the constant whose row 0 is ones.
  constexpr uint32_t kOnesPair = 0x3F803F80u;
  uint32_t ones_a[4];
  ones_a[0] = (lane >> 2) == 0 ? kOnesPair : 0u;
  ones_a[1] = 0u;
  ones_a[2] = ones_a[0];
  ones_a[3] = 0u;

  //: THE LATTICE LEAF OF A PRIVATE SLOT.  The private order's own `shift`/`span` give
  //: each level's digit inside `P`; the lattice's `run_shift` puts it back, and every
  //: carved level carries the stream's own digit offset.  It is the ONE place the two
  //: orders meet, and the state plane's page sweep is what asks.
  auto leaf_of = [&](int P) {
    int lf = 0;
    static_for<D>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      const int dig = ((P >> PM::shift(l)) & (PM::span(l) - 1)) + off[l];
      lf += dig << BP::run_shift(l);
    });
    return lf;
  };

  // ---- the state plane's page sweep, at CANONICAL atom ids -----------------
  constexpr int kAtoms = BC / kAtomLeaves;
  constexpr int kPageElems = kAtomLeaves * kCols;
  //: THE SWEEP'S STEP, AS A CARRY.  A step of `kThreads` elements advances the row
  //: cursor by `kRowStep` or by one more, so the walk is a compile-time count of
  //: predicated subtractions and never a division.
  constexpr int kRowStep = kThreads / kCols;
  constexpr int kPageCarry = kThreads - kRowStep * kCols;
  static_assert(kPageCarry >= 0 && kPageCarry < kCols,
                "a sweep step's carry is the thread count modulo the row, so the cursor "
                "crosses at most one row boundary per step");
  auto page_sweep = [&](uint32_t pageOff, auto&& body) {
    int e = tid;
    int row = 0, col = tid;
    static_for<kRowStep>([&](auto) {
      const bool over = col >= kCols;
      row += over ? 1 : 0;
      col -= over ? kCols : 0;
    });
#pragma unroll 1
    while (e < kPageElems) {
      body(row * LdO + col, pageOff + (uint32_t)(e * (int)sizeof(float)));
      e += kThreads;
      col += kPageCarry;
      const bool wrap = col >= kCols;
      row += wrap ? kRowStep + 1 : kRowStep;
      col -= wrap ? kCols : 0;
    }
  };
  //: THE PLANE IS ADDRESSED BY CANONICAL ATOM ID (P80).  `atomBH` is this `bh`'s base
  //: and `BP::canon_atom` re-keys the owner's own `kAtoms` lattice atoms to the
  //: canonical numbering the page plan uses -- ids only, at the prologue; the row
  //: INSIDE an atom is the same in both orders, so no data is permuted.
  const uint32_t atomBH = (uint32_t)bh * (uint32_t)(BP::kLeaves / kAtomLeaves);
  //: THE SLOT, NOT THE ATOM, ADDRESSES THE PLANE (`ops.cuh`'s `page_slot`,
  //: docs/internals/paging/paging.md).  The atom id is the logical name and the slot is
  //: the physical page the plan committed for it; a null table makes the two the same
  //: number, which is the dense backing as this same kernel rather than a second path.
  constexpr uint32_t kPageBytes = (uint32_t)(kPageElems * (int)sizeof(float));
  auto page_base = [](int32_t slot) { return (uint32_t)slot * kPageBytes; };
  //: THE PAGE'S SLICE OF A STREAM, PER FRAGMENT.  Register-resident state exists only as
  //: fragments (KERNEL_STANDARDS section 1): the accumulator is `NL` n-tiles x `MCH`
  //: m-tiles of the m16n8 shape, and a lane's fragment holds TWO ADJACENT private slots
  //: at two channel rows.  So this nest iterates FRAGMENT MODES -- n-tile, the pair's
  //: half, m-tile, the row half -- and the leaf map is folded into the ADDRESS.
  //: BODY COUNT: `NL * 2 * (2*MCH + 1)` per atom (72 at the 32-leaf `d_v = 64` arms),
  //: times `kAtoms` per call.  It is O(fragments) in both modes.
  auto page_regs = [&](auto Ac, auto Ld) {
    constexpr int a = decltype(Ac)::value;
    constexpr bool LOAD = decltype(Ld)::value;
    static_for<NL>([&](auto Jc) {
      constexpr int j = decltype(Jc)::value;
      static_for<2>([&](auto Lc) {
        constexpr int el = decltype(Lc)::value;
        const int lf = leaf_of(j * kMmaN + 2 * (lane & 3) + el);
        if ((lf >> ilog2c(kAtomLeaves)) != a) return;
        const int row = lf & (kAtomLeaves - 1);
        static_for<MCH>([&](auto Mc) {
          constexpr int m = decltype(Mc)::value;
          static_for<2>([&](auto Hc) {
            constexpr int h = decltype(Hc)::value;
            float* const cell = &so[row * LdO + m * kMmaM + (lane >> 2) + 8 * h];
            if constexpr (LOAD)
              st[m][j][el + 2 * h] = *cell;
            else
              store_f32(cell, st[m][j][el + 2 * h]);
          });
        });
        //: the mass rides row 0 of its own accumulator, so only the lane quad holding
        //: that row carries it.
        if ((lane >> 2) == 0) {
          float* const cell = &so[row * LdO + kDv];
          if constexpr (LOAD)
            mass_st[j][el] = *cell;
          else
            store_f32(cell, mass_st[j][el]);
        }
      });
    });
  };

  //: ---- THE ENTRY STATE.  The fragments are SET, never accumulated: they are
  //: provably zero here, and with no decay the recurrence is affine in the state
  //: with an inclusive-after-write readout, so this IS the oracle's seeding.
  if constexpr (STATE) {
    if (s_in != nullptr) {
      const float* const plane = pin_address(s_in);
      static_for<kAtoms>([&](auto Ac) {
        //: AN ABSENT ATOM IS SKIPPED, AND WITH IT THE TWO BARRIERS: the plan committed
        //: nothing here, the fragments keep the zero the prologue set, and reading a
        //: never-written state is inert under the self-normalizing readout.
        const int32_t slot =
            page_slot(page_tbl, atomBH + BP::canon_atom(owner, decltype(Ac)::value));
        if (slot < 0) return;
        page_sweep(page_base(slot), [&](int sidx, uint32_t off_) {
          so[sidx] = *reinterpret_cast<const float*>(gather_ptr(plane, off_));
        });
        rendezvous();
        page_regs(Ac, std::true_type{});
        rendezvous();
      });
    }
  }

  // ---- the produce's per-thread map ---------------------------------------
  //: ONE LANE MINTS ONE TOKEN'S WHOLE ROW: the batch tile is the warp width, so a
  //: thread's slot run is the stream's entire box and the expansion is `sum_l S_l`
  //: amplitude loads folded into `kLeafW / 2` packed accumulators.
  constexpr int SPT = kLeafW;
  static_assert(C == kBallotLanes,
                "a warp's panel row is ONE lane's: the batch tile is the warp width, so a "
                "thread's slot run is the stream's whole box");
  const int prow = lane;  // this lane's token row of the batch
  int lvCol[D];
  static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    lvCol[l] = runCol[l] + off[l];
  });

  //: A DEAD ROW IS ZEROED AT THE NARROWEST FACTOR IT HAS (`PM::kZeroLevel`).
  //:
  //: THE READ SIDE HAS NO GAIN.  Its scalar is identically 1 and a trailing
  //: `mul_bf16x2(acc, 1)` was 44% of the read produce -- so the side is a compile-time
  //: argument and the read body stores its accumulators directly.  The WRITE side's
  //: dead row is zeroed by its GAIN instead, which is zero where the row has no token.
  auto produce = [&](auto Wt, const __nv_bfloat16* base, __nv_bfloat16* dst, uint32_t tokOff,
                     uint32_t gain, bool have) {
    constexpr bool WRITE = decltype(Wt)::value;
    //: THE SLOT ACCUMULATORS ARE THE ONLY ARRAY THE PRODUCE CARRIES.  A level's run
    //: is loaded, folded into every slot and dropped inside one `static_for` body, so
    //: no `[D][n]` array is ever live.
    uint32_t acc[SPT / 2];
    {
      unsigned short run[PM::span(PM::kPairLevel)];
      load_run<PM::span(PM::kPairLevel)>(
          run, gather_ptr(base, tokOff + (uint32_t)(lvCol[PM::kPairLevel] * 2)));
      static_for<SPT / 2>([&](auto Jc) {
        constexpr int j = 2 * decltype(Jc)::value;
        acc[j / 2] = ((uint32_t)run[(j + 1) & (PM::span(PM::kPairLevel) - 1)] << 16)
                     | (uint32_t)run[j & (PM::span(PM::kPairLevel) - 1)];
      });
    }
    static_for<D>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      if constexpr (l != PM::kPairLevel && l != PM::kZeroLevel) {
        unsigned short run[PM::span(l)];
        load_run<PM::span(l)>(run, gather_ptr(base, tokOff + (uint32_t)(lvCol[l] * 2)));
        static_for<SPT / 2>([&](auto Jc) {
          constexpr int j = 2 * decltype(Jc)::value;
          acc[j / 2] = mul_bf16x2(acc[j / 2],
                                  splat_bf16x2_bits(run[(j >> PM::shift(l)) & (PM::span(l) - 1)]));
        });
      }
    });
    {
      constexpr int z = PM::kZeroLevel;
      unsigned short run[PM::span(z)];
      load_run<PM::span(z)>(run, gather_ptr(base, tokOff + (uint32_t)(lvCol[z] * 2)));
      if constexpr (!WRITE) {
        static_for<PM::span(z)>([&](auto Ic) {
          constexpr int i = decltype(Ic)::value;
          run[i] = have ? run[i] : (unsigned short)0;
        });
      }
      static_for<SPT / 2>([&](auto Jc) {
        constexpr int j = 2 * decltype(Jc)::value;
        acc[j / 2] =
            mul_bf16x2(acc[j / 2], splat_bf16x2_bits(run[(j >> PM::shift(z)) & (PM::span(z) - 1)]));
      });
    }
    __nv_bfloat16* const row = dst + prow * LdS;
    if constexpr (WRITE) {
      static_for<SPT / 2>([&](auto Jc) {
        constexpr int j = 2 * decltype(Jc)::value;
        store_bf16x2(row + j, mul_bf16x2(acc[j / 2], gain));
      });
    } else {
      static_for<SPT / 2>([&](auto Jc) {
        constexpr int j = 2 * decltype(Jc)::value;
        store_bf16x2(row + j, acc[j / 2]);
      });
    }
  };

  // ---- V staging -----------------------------------------------------------
  constexpr int kChunkV = kDv / 8;
  constexpr int kRowsPerPass = kThreads / kChunkV;
  constexpr int kPasses = C / kRowsPerPass;
  static_assert(kPasses >= 1, "a batch must be at least kThreads/8 rows");
  const int ldRow = tid / kChunkV, ldCol = (tid % kChunkV) * 8;
  const uint32_t svL = smem_addr(sv) + (uint32_t)((ldRow * LdV + ldCol) * 2);

  // ---- the GLOBAL window grid ---------------------------------------------
  const int nWindows = (L + W - 1) / W;

#pragma unroll 1
  for (int win = 0; win < nWindows; ++win) {
    const int tok0 = win * W;
    const int wLen = (L - tok0) < W ? (L - tok0) : W;
    const uint32_t winP = (uint32_t)tok0 * (uint32_t)(BP::kWtot * 2);
    const uint32_t winG = (uint32_t)tok0 * 2u;
    const uint32_t winV = (uint32_t)tok0 * (uint32_t)v_row_bytes;
    const uint32_t winO = (uint32_t)tok0 * (uint32_t)(kDv * (int)sizeof(float));
    const uint32_t winD = (uint32_t)tok0 * 4u;
    //: ONE BATCH'S PANEL, expanded from factors.  The list, the row's token and the
    //: gain are all resolved here so the two phases call ONE body; `have` false
    //: zeroes the narrowest factor, and a zero coefficient contributes exactly nothing
    //: to either MMA (the produce's dead-row zeroing law).
    auto produce_batch = [&](auto Wt, const __nv_bfloat16* base, __nv_bfloat16* dst,
                             const uint16_t* list, int o0, int ntot) {
      const int nb = (ntot - o0) < C ? (ntot - o0) : C;
      const bool have = prow < nb;
      const int tok = list[o0 + (have ? prow : nb - 1)];
      uint32_t gain = 0u;
      if constexpr (decltype(Wt)::value)
        gain = have ? splat_bf16x2(*reinterpret_cast<const __nv_bfloat16*>(
                          gather_ptr(gW, winG + (uint32_t)tok * 2u)))
                    : 0u;
      produce(Wt, base, dst, winP + (uint32_t)tok * (uint32_t)(BP::kWtot * 2), gain, have);
    };

    // ---- the liveness pass: one READ list per warp, ONE write list --------
    //: THE SCHEDULE IS TWO-SIDED (Blake ruling 2026-08-19).  Per level at most ONE
    //: side is sparse, so there is one stored row set per level -- but the two
    //: sides' CLAUSE SETS are different level subsets, and a dense side's clause is
    //: absent-means-full.  ANDing each side's OWN rows IS that conjunction over
    //: exactly the levels where that side is sparse, with no structure fact carried
    //: into the kernel.  The read side's carved clause is the STREAM'S OWN rows; the
    //: write side wants the owner's run, which is the OR over the streams and is one
    //: warp's work because every warp folds every write entry.
    //:
    //: ONE BALLOT WORD PER LANE.  A warp's whole window is `LW` words, so a row is
    //: ONE `LDG.32` and every warp can afford to hold BOTH sides' owner-grain
    //: ballots.  That is what makes the schedule barrier-free: the lists, the
    //: counts and the whole-window skip are all warp-private computations of
    //: CTA-identical inputs, so the warps agree by construction and reach the
    //: window line without a barrier of the schedule's own.
    constexpr int LW = G::kLW;
    static_assert(W <= 0xFFFF, "a token's window-local index must fit the list's element");
    static_assert(LW <= kBallotLanes,
                  "the walk gives one ballot word to each lane, so a window is at most a "
                  "lockstep unit's worth of words");
    const int room = wLen - lane * kBallotLanes;
    const bool inW = lane < LW;
    const uint32_t valid =
        (!inW || room <= 0) ? 0u : (room >= kBallotLanes ? 0xFFFFFFFFu : ((1u << room) - 1u));
    //: a lane past the window's last word holds no word; it reads word zero, which
    //: is in range, and `valid` empties its ballot.
    const int wIdx = (tok0 >> 5) + (inW ? lane : 0);
    const uint32_t wordB = (uint32_t)(wIdx < tok_words ? wIdx : tok_words - 1) * 4u;

    //: the trip count is the window's LIVENESS, not its width: one `bit_count`
    //: where a per-word form spends `LW`, one lane prefix where it spends `LW`
    //: adds, and one scatter of the lane's own set bits.
    //: `bits` is the writer's slice of each word, `below` the bits under it: a list
    //: with several writers is built by giving each one a bit range and letting all
    //: of them derive the same prefix from the same ballot.
    auto walk = [&](uint32_t m, uint16_t* dst, uint32_t bits, uint32_t below) {
      const int cnt = bit_count(m);
      const int pfx = lane_prefix_incl<LW>(cnt);
      int k = (pfx - cnt) + bit_count(m & below);
      const uint16_t tb = (uint16_t)(lane * kBallotLanes);
      uint32_t mm = m & bits;
      while (mm) {
        dst[k++] = tb + (uint16_t)first_set(mm);
        mm &= mm - 1;
      }
      return lane_broadcast(pfx, LW - 1);
    };
    //: THE CLAUSE ABOVE THE CARVE: the levels the stream spans WHOLE, whose row is the
    //: owner's own and is therefore shared by every stream.  A level the owner spans
    //: whole emits no row at all and drops out of the conjunction on its own.
    auto clause_of = [&](const uint32_t* rowsS) {
      uint32_t m = valid;
      static_for<D>([&](auto Lc) {
        constexpr int l = decltype(Lc)::value;
        if constexpr (!SV::carves(l) && CR::row_g(l) > 0)
          m &= *reinterpret_cast<const uint32_t*>(gather_ptr(rowsS, rowOff[l] + wordB));
      });
      return m;
    };
    //: AND THE CARVED CLAUSE OF ONE STREAM: the AND over the carved levels of that
    //: stream's own row.  A single-level carve is one load; a multi-level carve is one
    //: per carved level, which is the exact conjunction the box asks for.
    auto stream_row = [&](const uint32_t* rowsS, int s) {
      uint32_t m = 0xFFFFFFFFu;
      static_for<D>([&](auto Lc) {
        constexpr int l = decltype(Lc)::value;
        if constexpr (SV::carves(l)) {
          const uint32_t c = (uint32_t)((s >> ilog2c(SV::grid_div(l))) & (SV::g(l) - 1));
          m &= *reinterpret_cast<const uint32_t*>(
              gather_ptr(rowsS, rowOff[l] + c * rowBytes + wordB));
        }
      });
      return m;
    };

    const uint32_t clauseR = clause_of(rowsR), clauseW = clause_of(rowsW);
    uint32_t orR = 0u, orW = 0u, mineR = 0u;
#pragma unroll
    for (int s = 0; s < kWarps; ++s) {
      const uint32_t r = stream_row(rowsR, s);
      orR |= r;
      mineR = (s == strm) ? r : mineR;
      orW |= stream_row(rowsW, s);
    }
    const uint32_t liveW = clauseW & orW;
    //: THE WHOLE-WINDOW SKIP, at (owner, window) grain, per side -- the block-bitmap
    //: mechanism's harvest (study section 2.1).  It is CTA-uniform WITHOUT a
    //: barrier: the two owner-grain ballots are the same words in every warp.
    if (lane_ballot(((clauseR & orR) | liveW) != 0u) == 0u) continue;

    uint16_t* const rlist = slist + warp * kListStride;
    uint16_t* const wlist = slist + kWarps * kListStride;
    const int nr = walk(clauseR & mineR, rlist, 0xFFFFFFFFu, 0u);
    __syncwarp();
    //: THE WRITE LIST IS ONE LIST WITH `kWarps` WRITERS.  Every warp folds every
    //: write entry, so the list is shared -- and because every warp holds the same
    //: ballot, each can take a fixed slice of every word's bits and derive its own
    //: offsets.  The list is therefore built BALANCED at any distribution, with no
    //: CTA prefix, no barrier of its own, and no warp carrying it alone.  The
    //: window line is what publishes it.
    constexpr uint32_t kBitsW = (uint32_t)(kBallotLanes / kWarps);
    const int nw =
        walk(liveW, wlist, ((1u << kBitsW) - 1u) << (warp * kBitsW), (1u << (warp * kBitsW)) - 1u);

    // ---- PHASE 1: the READOUT of every read batch, against the window's ENTRY
    //      state.  ALL of the window's readouts precede ANY of its folds, because a
    //      within-window pair is the INTRA term's and this kernel must not carry it.
    //      The batch line is therefore the window line on both sides.
    //: THE READ PHASE SHARES NOTHING: the list, the panel region, the staging block
    //: and the fan-in are the warp's own, so the warps run to the window line
    //: without a single CTA barrier.  Its `__syncwarp` is the panel's, between a
    //: lane's `STS` and the `ldmatrix` that reads another lane's row.
    const int nbr = (nr + C - 1) / C;
    //: THE PRODUCE RUNS ONE BATCH AHEAD, into the other panel of the two-slot ring,
    //: UNCONDITIONALLY and AFTER the batch's MMA block.  Unconditionally because a
    //: guard is a basic block and ptxas schedules within one; after, because the
    //: produce ends in panel `STS` and the MMA begins in panel `LDSM`, and the two
    //: shared ranges are not provably disjoint -- in the other order that aliasing
    //: edge pins every produce instruction ahead of the HMMAs.  The trailing batch is
    //: entirely dead rows by the produce's zeroing law and lands in a panel no MMA
    //: reads.
    if (nbr > 0) produce_batch(std::false_type{}, pR, panR, rlist, 0, nr);
    __syncwarp();
#pragma unroll 1
    for (int i = 0; i < nbr; ++i) {
      const int off_ = i * C;
      const int n = (nr - off_) < C ? (nr - off_) : C;
      const uint32_t brT = (i & 1) ? brT1 : brT0;

      //: THE READ DRAIN'S STATE SOURCE, AND ITS ONE PLACE.  K35: the fragment is the
      //: accumulator's TOP HALVES taken by `prmt` -- one instruction per pair, and the
      //: same bytes a hi-plane hybrid would publish, so that hybrid is a placement
      //: change and never an arithmetic one.  It TRUNCATES where an image would have
      //: ROUNDED; declared, and mirrored in the oracle.  Here the accumulator IS the
      //: operand (the warp's own n-tiles, packed in place, no `ldmatrix`), so the drain
      //: is the ONLY thing that touches the state's residency and it is written HERE
      //: rather than behind a lambda: a generic lambda taking an array reference whose
      //: bound is a template-dependent constant (`MCH`, from the `DV` axis) crashes the
      //: front end, and the fragment stack has exactly one consumer anyway.
      //: BODY COUNT: `KDw * (MCH + 1)` -- O(fragments) in the arm.
      uint32_t sa[KDw][MCH][4], ma[KDw][4];
      static_for<KDw>([&](auto Kc) {
        constexpr int kk = decltype(Kc)::value;
        static_for<MCH>([&](auto Mc) {
          constexpr int m = decltype(Mc)::value;
          sa[kk][m][0] = hi_bf16x2(st[m][2 * kk][0], st[m][2 * kk][1]);
          sa[kk][m][1] = hi_bf16x2(st[m][2 * kk][2], st[m][2 * kk][3]);
          sa[kk][m][2] = hi_bf16x2(st[m][2 * kk + 1][0], st[m][2 * kk + 1][1]);
          sa[kk][m][3] = hi_bf16x2(st[m][2 * kk + 1][2], st[m][2 * kk + 1][3]);
        });
        ma[kk][0] = hi_bf16x2(mass_st[2 * kk][0], mass_st[2 * kk][1]);
        ma[kk][1] = hi_bf16x2(mass_st[2 * kk][2], mass_st[2 * kk][3]);
        ma[kk][2] = hi_bf16x2(mass_st[2 * kk + 1][0], mass_st[2 * kk + 1][1]);
        ma[kk][3] = hi_bf16x2(mass_st[2 * kk + 1][2], mass_st[2 * kk + 1][3]);
      });
#pragma unroll
      for (int mb = 0; mb < MTOK; ++mb) {
        float acc[2][MCH][4], macc[2][4];
#pragma unroll
        for (int p = 0; p < 2; ++p) {
#pragma unroll
          for (int m = 0; m < MCH; ++m)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[p][m][e] = 0.f;
#pragma unroll
          for (int e = 0; e < 4; ++e) macc[p][e] = 0.f;
        }
        uint32_t rb[KDw][4];
#pragma unroll
        for (int kk = 0; kk < KDw; ++kk) load_frag(rb[kk], ADDR_S(brT, mb * 16, 2 * kk));
#pragma unroll
        for (int kk = 0; kk < KDw; ++kk)
#pragma unroll
          for (int p = 0; p < 2; ++p) {
#pragma unroll
            for (int m = 0; m < MCH; ++m) mma(acc[p][m], sa[kk][m], &rb[kk][2 * p]);
            mma(macc[p], ma[kk], &rb[kk][2 * p]);
          }
        //: THE FAN-IN IS WARP-PRIVATE AND DIRECT (K33 1.4): one writer per read
        //: entry at the sparse operating point, and RED is associative where an
        //: entry's digits reach two streams, so no meeting is instantiated.  The
        //: staging pass is what makes the RED lane-per-CHANNEL and therefore one
        //: coalesced segment per issue -- the form R-A and R-B both measured against
        //: the register-sourced ones and found ahead.
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          const int j = 2 * mb + p;
#pragma unroll
          for (int m = 0; m < MCH; ++m)
#pragma unroll
            for (int e = 0; e < 4; ++e)
              store_f32(
                  &sow[((lane & 3) * 2 + (e & 1)) * LdO + m * kMmaM + (lane >> 2) + 8 * (e >> 1)],
                  acc[p][m][e]);
          if (lane < 4) {
            store_f32(&sow[(lane * 2 + 0) * LdO + kDv], macc[p][0]);
            store_f32(&sow[(lane * 2 + 1) * LdO + kDv], macc[p][1]);
          }
          __syncwarp();
#pragma unroll
          for (int r = 0; r < kMmaN; ++r) {
            const int idx = j * kMmaN + r;
            if (idx < n) {
              float* const dp = reinterpret_cast<float*>(
                  gather_ptr(oPlane, (uint32_t)rlist[off_ + idx] * (uint32_t)(kDv * 4) + winO
                                         + (uint32_t)(lane * (int)sizeof(float))));
              static_for<kDv / kBallotLanes>([&](auto Cc) {
                constexpr int cb = decltype(Cc)::value * kBallotLanes;
                fan_in_add(dp + cb, sow[r * LdO + lane + cb]);
              });
            }
          }
          if (lane < kMmaN && j * kMmaN + lane < n)
            fan_in_add(reinterpret_cast<float*>(gather_ptr(
                           denPlane, (uint32_t)rlist[off_ + j * kMmaN + lane] * 4u + winD)),
                       sow[lane * LdO + kDv]);
          __syncwarp();
        }
      }
      produce_batch(std::false_type{}, pR, ((i + 1) & 1) ? panW : panR, rlist, (i + 1) * C, nr);
      __syncwarp();
    }

    //: THE WINDOW LINE IS THE ONE BARRIER LEFT ON THE READ SIDE, and it is there by
    //: the ordering law: no warp may fold this window's writes while another is
    //: still reading the window's entry state.
    rendezvous();

    // ---- PHASE 2: the FOLD of every write batch ---------------------------
    //: V IS A WRITE-SIDE QUANTITY and is gathered by the WRITE list; the read list
    //: never touches it.  The value ring is `STAGES` deep and asynchronous: issue
    //: order is FIFO, one implicit group per batch, and the wait leaves `STAGES-2`
    //: groups in flight so the gather for batch `i + STAGES - 1` is airborne while
    //: batch `i` folds.
    const int nbw = (nw + C - 1) / C;
    auto stage_batch = [&](int b, int slot) {
      const int o0 = b * C;
      const int nb = (nw - o0) < C ? (nw - o0) : C;
#pragma unroll
      for (int p = 0; p < kPasses; ++p) {
        const int r = ldRow + p * kRowsPerPass;
        const int t = wlist[o0 + (r < nb ? r : nb - 1)];
        stage_tile(
            svL + (uint32_t)slot * kVSlot + (uint32_t)(p * kRowsPerPass * LdV * 2),
            gather_ptr(vPlane, (uint32_t)t * (uint32_t)v_row_bytes + winV + (uint32_t)(ldCol * 2)));
      }
    };
#pragma unroll 1
    for (int sst = 0; sst < STAGES - 1; ++sst) {
      if (sst < nbw) stage_batch(sst, sst % STAGES);
      stage_commit();
    }
    if (nbw > 0) produce_batch(std::true_type{}, pW, panW, wlist, 0, nw);
#pragma unroll 1
    for (int i = 0; i < nbw; ++i) {
      const uint32_t bwF = (i & 1) ? bwF1 : bwF0;
      const uint32_t vsl = svB + (uint32_t)(i % STAGES) * kVSlot;
      stage_wait<STAGES - 2>();
      rendezvous();
      //: the ring slot is advanced on EVERY batch, live or not, so the wait/commit
      //: accounting never depends on the data.
      const int f = i + STAGES - 1;
      if (f < nbw) stage_batch(f, f % STAGES);
      stage_commit();

#pragma unroll
      for (int kk = 0; kk < MTOK; ++kk) {
        uint32_t a[MCH][4];
#pragma unroll
        for (int m = 0; m < MCH; ++m) load_frag_t(a[m], ADDR_V(vsl, kk * 16, m * kMmaM));
#pragma unroll
        for (int nj = 0; nj < NL / 2; ++nj) {
          uint32_t b[4];
          load_frag_t(b, ADDR_S(bwF, kk * 16, 2 * nj));
#pragma unroll
          for (int m = 0; m < MCH; ++m) {
            mma(st[m][2 * nj], a[m], &b[0]);
            mma(st[m][2 * nj + 1], a[m], &b[2]);
          }
          mma(mass_st[2 * nj], ones_a, &b[0]);
          mma(mass_st[2 * nj + 1], ones_a, &b[2]);
        }
      }
      produce_batch(std::true_type{}, pW, ((i + 1) & 1) ? panR : panW, wlist, (i + 1) * C, nw);
    }
    //: the ring's unconsumed groups are retired before the next window reuses the
    //: value slots.
    stage_wait<0>();
  }

  // ---- the FINAL STATE, at CANONICAL atom ids ------------------------------
  //: the mirror of the entry load, after the last fan-in has consumed the staging
  //: buffer -- which is what the one extra rendezvous below is for.
  if constexpr (STATE) {
    if (s_out != nullptr) {
      float* const plane = pin_address(s_out);
      rendezvous();
      static_for<kAtoms>([&](auto Ac) {
        //: THE EXIT STORES ONLY PLANNED ATOMS.  An unplanned one is not written at all,
        //: which is what leaves a skipped slot carrying exactly what it held.
        const int32_t slot =
            page_slot(page_tbl, atomBH + BP::canon_atom(owner, decltype(Ac)::value));
        if (slot < 0) return;
        page_regs(Ac, std::false_type{});
        rendezvous();
        page_sweep(page_base(slot), [&](int sidx, uint32_t offb) {
          store_f32(reinterpret_cast<float*>(gather_ptr(plane, offb)), so[sidx]);
        });
        rendezvous();
      });
    }
  }
#undef ADDR_S
#undef ADDR_V
}

}  // namespace rola::carry
