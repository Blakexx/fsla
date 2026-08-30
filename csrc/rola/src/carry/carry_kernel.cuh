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
#include "common/state_page.cuh"

namespace rola::carry {

using namespace denseref::ops;
namespace design = denseref::design;

static constexpr int kPadE = 8;  // bf16 row padding, elements (16 B)
static constexpr int kPadF = 4;  // fp32 row padding, elements (16 B)
//: `d_v` IS AN ARM PARAMETER AND NOT A FILE CONSTANT (K35 S3b). -- see docs/internals/carry/carry_kernel.md#static-assert
static_assert(kAtomLeaves == kMmaK, "the page granule is one MMA k-step");

//: A COMPILE-TIME LOOP. -- see docs/internals/carry/carry_kernel.md#static-for
template <int N, class F>
__device__ __forceinline__ void static_for(F&& f) {
  if constexpr (N > 0) {
    static_for<N - 1>(f);
    f(std::integral_constant<int, N - 1>{});
  }
}

//: THE CARVE. A STREAM owns an affine RECTANGLE of the owner's digits -- a per-
//: level span vector, `prod_l S_l` leaves and ALL `kCols` channels -- and tiles
//: those leaves in a private mixed-radix order. -- see docs/internals/carry/carry_kernel.md#carrygeom

template <int D, int B, int K, int M, int DV, int W, int C, int WPS, int NSW, int NSR, int PK,
          uint32_t MO>
struct CarryGeom {
  using BP = BoxPlan<D, B, K, M>;
  //: THE TWO CARVES, DERIVED INDEPENDENTLY (K35 F1): each side's level ORDER comes -- see docs/internals/carry/carry_kernel.md#svr
  using SVR = StreamSpans<D, B, K, M, MO, kSideRead, NSR>;
  using SVW = StreamSpans<D, B, K, M, MO, kSideWrite, NSW>;
  static_assert(WPS == 1,
                "M1P builds ONE warp per stream; the column split (`WPS > 1`) returns with "
                "the containers, as its own stage");
  static_assert(NSW == NSR,
                "one warp is BOTH a read stream and a write stream, so the two sides carve "
                "the box into equally many boxes; unequal counts arrive with the write "
                "side's own schedule");
  static_assert(PK == 0,
                "the parked `lo` is a tenant of the exchange's drained slots, and "
                "the lo park is a later stage's mechanism");
  static constexpr int BC = BP::kBC;
  //: THE VALUE WIDTH, AS A DECLARATION.  `kDv` channels plus the MASS column, which -- see docs/internals/carry/carry_kernel.md#kdv
  static constexpr int kDv = DV;
  static constexpr int kCols = DV + 1;
  static_assert(DV == 64,
                "M1a's readout stages exactly `kDv` channels into the batch's `C` panel rows; "
                "the `d_v` dial returns as its own stage and is never a runtime test");

  //: THE EXECUTOR MAP: one warp per stream, and the batch tile IS the executor count -- see docs/internals/carry/carry_kernel.md#kstreams
  static constexpr int kStreams = SVR::kStreams;
  static constexpr int kWarps = kStreams;
  static constexpr int kThreads = kWarps * 32;
  //: the stream's leaves, and the private panel row that indexes them.  The two -- see docs/internals/carry/carry_kernel.md#kleafw
  static constexpr int kLeafW = SVW::kLeaf;
  static constexpr int kLeafR = SVR::kLeaf;
  static_assert(kLeafW == kLeafR,
                "at one stream count per side both carves partition the same box into the "
                "same number of leaves, so one panel stride serves both phases");
  static constexpr int LdV = kDv + kPadE;
  static constexpr int LdS = kLeafW + kPadE;
  static constexpr int LdO = kDv + kPadF;
  //: THE EXCHANGE'S ROW: one leaf's channels and its MASS, padded to the sixteen-byte -- see docs/internals/carry/carry_kernel.md#ldx
  static constexpr int LdX = kDv + kPadE;

  //: THE TIE, AND THE THREE STATES OF THE EXCHANGE (K35 S1b's `kXchState`). -- see docs/internals/carry/carry_kernel.md#kxchstate
  static constexpr bool same_vec(int l) {
    return l >= D || (SVW::s(l) == SVR::s(l) && same_vec(l + 1));
  }
  static constexpr bool same_ord(int l) {
    return l >= D || (SVW::Ord::rank(l) == SVR::Ord::rank(l) && same_ord(l + 1));
  }
  static constexpr bool kTied = same_vec(0) && same_ord(0);
  static constexpr int kXchZero = 0, kXchFull = 1, kXchFullReorder = 2;
  static constexpr int kXchState = kTied ? kXchZero : (same_vec(0) ? kXchFullReorder : kXchFull);
  static constexpr int kXchBytes = kTied ? 0 : BC * LdX * (int)sizeof(__nv_bfloat16);

  //: THE TWO PANELS ARE A TWO-SLOT RING, AND THEY COST NOTHING EXTRA. -- see docs/internals/carry/carry_kernel.md#panel-bytes
  static constexpr int panel_bytes = 2 * kWarps * C * LdS * (int)sizeof(__nv_bfloat16);
  //: the readout's staging block, one `kMmaN`-token region per warp -- and, at -- see docs/internals/carry/carry_kernel.md#outbytes
  static constexpr int out_bytes = C * LdO * (int)sizeof(float);
  static_assert(kWarps * kMmaN == C && C * LdO >= kAtomLeaves * LdO,
                "the staging block is one token tile of the batch and holds a whole page");
  //: `kWarps` READ lists, one per stream, then ONE write list: a read entry belongs to -- see docs/internals/carry/carry_kernel.md#klists
  static constexpr int kLists = kWarps + 1;
  static constexpr int kListStride = ((W + 7) / 8) * 8;
  static constexpr int list_bytes = kLists * kListStride * (int)sizeof(uint16_t);
  //: THE WINDOW'S BALLOT WORDS, one per lane of the walk -- so a window may be at -- see docs/internals/carry/carry_kernel.md#klw
  static constexpr int kLW = W / kBallotLanes;
  //: THE V RING'S DEPTH IS DERIVED FROM THE SMEM BUDGET AND THE REGISTER FLOOR: -- see docs/internals/carry/carry_kernel.md#kregfloor
  static constexpr int kRegFloor = 220;
  static constexpr int slot_bytes = C * LdV * (int)sizeof(__nv_bfloat16);
  static constexpr int fixed_bytes = panel_bytes + out_bytes + list_bytes;
  static constexpr int kStages = design::stages_for(slot_bytes, fixed_bytes, kThreads, kRegFloor);
  static constexpr size_t bytes = (size_t)fixed_bytes + (size_t)kStages * slot_bytes;
  static constexpr int cta_target = design::cta_target((int)bytes, kThreads, kRegFloor);
  static_assert(design::smem_fits(bytes), "this arm does not fit the SMEM maximum");
  //: THE TRANSIT COSTS NO ALLOCATION: it STRADDLES the tenants that are dead at the -- see docs/internals/carry/carry_kernel.md#straddle-bytes
  static constexpr int straddle_bytes = panel_bytes / 2 + out_bytes;
  static_assert(kXchBytes <= straddle_bytes,
                "the transit's region is the SECOND panel and the readout staging block: the "
                "two tenants dead at BOTH of its moments, and disjoint from the V ring and "
                "the first panel, which is what leaves the work that covers each leg free to "
                "run against it; an arm whose transit outgrew them would have to allocate, "
                "and none does");
};

//: THE OWNER-GRAIN PRODUCE MAP, which the channel-split body and the reverse -- see docs/internals/carry/carry_kernel.md#producemap
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

//: THE PRIVATE LEAF ORDER. -- see docs/internals/carry/carry_kernel.md#spanmap
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
  //: THE PACKED PAIR IS SIMPLY THE INNERMOST LEVEL OF THE ORDER. -- see docs/internals/carry/carry_kernel.md#pair-from
  static constexpr int pair_from(int l) {
    return l >= D ? -1 : ((shift(l) == 0 && span(l) >= 2) ? l : pair_from(l + 1));
  }
  static constexpr int kPairLevel = pair_from(0);
  //: THE DEAD ROW'S LEVEL. -- see docs/internals/carry/carry_kernel.md#narrow-from
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

//: THE EXCHANGE'S ROW MAP (K35 F1).  The publisher's write slot `P` names the -- see docs/internals/carry/carry_kernel.md#xchmap
template <int D, int NSW, class SVW, class SVR, class PMW, class PMR>
struct XchMap {
  static constexpr int kLeafR = SVR::kLeaf;
  static constexpr int kBitsP = ilog2c(SVW::kLeaf);
  static constexpr int digit(int l, int w, int P) {
    return ((P >> PMW::shift(l)) & (SVW::s(l) - 1))
           + ((w >> ilog2c(SVW::grid_div(l))) & (SVW::g(l) - 1)) * SVW::s(l);
  }
  static constexpr int reader(int l, int w, int P) {
    return l >= D ? 0 : (digit(l, w, P) / SVR::s(l)) * SVR::grid_div(l) + reader(l + 1, w, P);
  }
  static constexpr int slot(int l, int w, int P) {
    return l >= D ? 0 : ((digit(l, w, P) % SVR::s(l)) << PMR::shift(l)) + slot(l + 1, w, P);
  }
  static constexpr int row(int w, int P) { return reader(0, w, P) * kLeafR + slot(0, w, P); }
  //: the per-bit STEP.  `row(0, 0)` is zero (stream zero's offsets are), so the step
  //: IS the row, and the derivation makes it independent of the publisher.
  static constexpr int delta(int P) { return row(0, P); }

  //: the two swizzles, GF(2)-LINEAR in the row so `swz(base ^ delta)` splits into a -- see docs/internals/carry/carry_kernel.md#swz
  static constexpr int swz(int r) { return ((r >> 3) ^ (r >> 4) ^ (r >> 6)) & 7; }
  static constexpr int msw(int r) { return ((r >> 3) ^ (r >> 5)) & 3; }

  //: THE DISJOINTNESS THE ADDRESS ARITHMETIC RESTS ON is proved by the model test, not -- see docs/internals/carry/carry_kernel.md#xchmap
  //: by a `static_assert`: the check exceeds nvcc's constexpr budget at `D = 4`.
};

//: a contiguous run of `n` bf16, at the widest access that covers it. -- see docs/internals/carry/carry_kernel.md#load-run
template <int N>
__device__ __forceinline__ void load_run(unsigned short (&dst)[N], const void* p) {
  static_assert(N == 1 || N == 2 || N == 4 || N == 8 || N == 16,
                "an admissible (k, m) run is 1, 2, 4, 8 or 16 wide");
  if constexpr (N == 1) {
    dst[0] = *reinterpret_cast<const unsigned short*>(p);
  } else if constexpr (N == 2) {
    *reinterpret_cast<unsigned int*>(dst) = *reinterpret_cast<const unsigned int*>(p);
  } else if constexpr (N == 4) {
    *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(p);
  } else {
    //: §6: `N` is this function's template parameter (ALL_CAPS per project convention, -- see docs/internals/carry/carry_kernel.md#koctets
    constexpr int kOctets = N / 8;
#pragma unroll
    for (int i = 0; i < kOctets; ++i)
      *reinterpret_cast<uint4*>(dst + i * 8) =
          load_vec16(reinterpret_cast<const char*>(p) + i * 16);
  }
}

//: THE SECOND LAUNCH BOUND: the CTA count this arm can actually reach, taken from the -- see docs/internals/carry/carry_kernel.md#near-line-215
template <int D, int B, int K, int M, int DV, int W, int C, int WPS, int NSW, int NSR, int PK,
          uint32_t MO>
constexpr int kCarryCtaTarget = CarryGeom<D, B, K, M, DV, W, C, WPS, NSW, NSR, PK, MO>::cta_target;

// ---------------------------------------------------------------- the kernel

template <int D, int B, int K, int M, int DV, int W, int C, int WPS, int NSW, int NSR, int PK,
          uint32_t MO>
__global__ __launch_bounds__((CarryGeom<D, B, K, M, DV, W, C, WPS, NSW, NSR, PK, MO>::kThreads),
                             (kCarryCtaTarget<D, B, K, M, DV, W, C, WPS, NSW, NSR, PK, MO>)) void
carry_kernel(const __nv_bfloat16* __restrict__ pread, const __nv_bfloat16* __restrict__ pwrite,
             const __nv_bfloat16* __restrict__ gwrite, const __nv_bfloat16* __restrict__ v,
             const uint32_t* __restrict__ rows, float* __restrict__ o, float* __restrict__ den,
             int L, int tok_words, int v_row_bytes, const float* __restrict__ s_in,
             float* __restrict__ s_out, const int32_t* __restrict__ page_tbl,
             const uint8_t* __restrict__ atom_bits) {
  using BP = BoxPlan<D, B, K, M>;
  using G = CarryGeom<D, B, K, M, DV, W, C, WPS, NSW, NSR, PK, MO>;
  constexpr int kDv = DV;
  constexpr int kCols = DV + 1;
  //: THE TWO SPAN VECTORS, each with the private order and the ballot rows it -- see docs/internals/carry/carry_kernel.md#sv
  using SVR = typename G::SVR;
  using SVW = typename G::SVW;
  using PMR = SpanMap<D, B, K, M, SVR>;
  using PMW = SpanMap<D, B, K, M, SVW>;
  using CRR = CarveRows<SVR>;
  using CRW = CarveRows<SVW>;
  using XM = XchMap<D, NSW, SVW, SVR, PMW, PMR>;
  constexpr int kWarps = G::kWarps;
  constexpr int kThreads = G::kThreads;

  constexpr int BC = BP::kBC;
  constexpr int LdV = G::LdV, LdS = G::LdS, LdO = G::LdO;
  constexpr int MTOK = C / kMmaM;  // token m-tiles of a batch
  //: the stream's own leaves: `kNL` accumulator n-tiles, `kDw` readout k-steps.
  constexpr int kLeafW = G::kLeafW;
  constexpr int kLeafR = G::kLeafR;
  constexpr int LdX = G::LdX;
  //: the FOLD accumulator's n-tiles are the WRITE carve's leaves; the READOUT's -- see docs/internals/carry/carry_kernel.md#knl
  constexpr int kNL = kLeafW / kMmaN;
  constexpr int kDw = kLeafR / kMmaK;
  constexpr int MCH = kDv / kMmaM;  // the channel m-tiles, ALL of them per warp

  static_assert(C % kMmaM == 0 && kThreads % C == 0, "the batch tile must divide");
  static_assert(kLeafW % kMmaN == 0 && kLeafR % kMmaK == 0,
                "each side's leaves are a whole number of that phase's tiles");
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

  //: ONE 64-BIT BASE PER PLANE, and a 32-BIT BYTE DISPLACEMENT after it -- the -- see docs/internals/carry/carry_kernel.md#vrowelems
  const int vRowElems = v_row_bytes / (int)sizeof(__nv_bfloat16);
  const int heads = vRowElems / kDv;
  const __nv_bfloat16* const vPlane =
      pin_address(v + (long)(bh / heads) * L * vRowElems + (long)(bh % heads) * kDv);
  const __nv_bfloat16* const pR = pin_address(pread + (long)bh * L * BP::kWtot);
  const __nv_bfloat16* const pW = pin_address(pwrite + (long)bh * L * BP::kWtot);
  const __nv_bfloat16* const gW = pin_address(gwrite + (long)bh * L);
  float* const oPlane = pin_address(o + (long)bh * L * kDv);
  float* const denPlane = pin_address(den + (long)bh * L);

  //: THE STREAM ROTATION.  Two CTAs co-resident on an SM are one grid wave apart and -- see docs/internals/carry/carry_kernel.md#strm
  const int strm = (warp + ((owner + (owner >> 5)) & (kWarps - 1))) & (kWarps - 1);

  // ---- the owner's digit runs, one per level -------------------------------
  //: `o_l` is a bit-field extract of the owner index: the lattice's owner -- see docs/internals/carry/carry_kernel.md#near-line-296
  int runCol[D];
  //: AND THE STREAM'S OWN DIGIT OFFSET INSIDE THAT RUN, per level AND PER SIDE: the -- see docs/internals/carry/carry_kernel.md#note-l344
  int offR[D], offW[D];
  //: AND THE BALLOT ROW BASE, per level and per side: the SAME `o_l`, addressing the -- see docs/internals/carry/carry_kernel.md#rowbytes
  uint32_t rowOffR[D], rowOffW[D];
  const uint32_t rowBytes = (uint32_t)tok_words * (uint32_t)sizeof(uint32_t);
  static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    const int o_l = (owner / BP::owner_div(l)) % BP::g(l);
    runCol[l] = BP::col_base(l) + o_l * BP::s(l);
    offR[l] = ((strm >> ilog2c(SVR::grid_div(l))) & (SVR::g(l) - 1)) * SVR::s(l);
    offW[l] = ((strm >> ilog2c(SVW::grid_div(l))) & (SVW::g(l) - 1)) * SVW::s(l);
    rowOffR[l] = (uint32_t)(CRR::row_prefix(l) + o_l * CRR::rows_of(BP::s(l), l)) * rowBytes;
    rowOffW[l] = (uint32_t)(CRW::row_prefix(l) + o_l * CRW::rows_of(BP::s(l), l)) * rowBytes;
  });
  //: ONE 64-BIT BASE PER SIDE, and 32-bit byte displacements after it: the READ -- see docs/internals/carry/carry_kernel.md#pin-address
  constexpr int kRowsBoth = CRR::kRows + CRW::kRows;
  const uint32_t* const rowsR = pin_address(rows + (long)bh * (long)kRowsBoth * (long)tok_words);
  const uint32_t* const rowsW =
      pin_address(rows + ((long)bh * (long)kRowsBoth + (long)CRR::kRows) * (long)tok_words);

  // ---- lane-invariant operand addressing (the shipped idiom) ---------------
  const int lr = lane & 7, tl = lane >> 3;
  const int rF = (tl & 1) * 8 + lr, cF = tl >> 1;
  const int rT = (tl >> 1) * 8 + lr, cT = tl & 1;
  //: THE PANEL RING'S TWO BASES, AS NAMED SCALARS.  A two-element array indexed by -- see docs/internals/carry/carry_kernel.md#panr
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

  //: THE TRANSIT'S ADDRESSING IS LOOP-INVARIANT, so it is derived ONCE. -- see docs/internals/carry/carry_kernel.md#transit
  __nv_bfloat16* const xch = sw;
  int xchR = 0, xchQ = 0;
  static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    xchR += (offW[l] / SVR::s(l)) * SVR::grid_div(l);
    xchQ += (offW[l] % SVR::s(l)) << PMR::shift(l);
  });
  const int xchBase = xchR * kLeafR + xchQ;
  const int xchLaneRow =
      xchBase ^ ((lane & 1) ? XM::delta(2) : 0) ^ ((lane & 2) ? XM::delta(4) : 0);
  const int swzLane = XM::swz(xchLaneRow), mswLane = XM::msw(xchLaneRow);
  const uint32_t pubC = smem_addr(xch) + (uint32_t)(xchLaneRow * LdX * 2 + (lane >> 2) * 2);
  const uint32_t pubM = smem_addr(xch) + (uint32_t)(xchLaneRow * LdX * 2 + kDv * 2);
  //: THE RETURN LEG'S ROW IS THE PUBLISH'S, SUPPLIED BY A DIFFERENT LANE. -- see docs/internals/carry/carry_kernel.md#return-leg
  const int retQ = lane & 7;
  const int retRow = xchBase ^ ((retQ & 1) ? XM::delta(1) : 0) ^ ((retQ & 2) ? XM::delta(2) : 0)
                     ^ ((retQ & 4) ? XM::delta(4) : 0);
  const uint32_t retC = smem_addr(xch) + (uint32_t)(retRow * LdX * 2);
  const int swzRet = XM::swz(retRow), retG = lane >> 3;
  //: the READER's segment addresses the same rows for the pull and the store-back.
  const int pullRow = strm * kLeafR + rT;
  const int swzPull = XM::swz(pullRow);
  const uint32_t pullC = smem_addr(xch) + (uint32_t)(pullRow * LdX * 2);
  const int massRow = strm * kLeafR + 2 * (lane & 3);
  const int mswPull = XM::msw(massRow), swzSB = XM::swz(strm * kLeafR);
  const uint32_t pullM = smem_addr(xch) + (uint32_t)(massRow * LdX * 2 + kDv * 2);
  const uint32_t sbC = smem_addr(xch) + (uint32_t)(massRow * LdX * 2 + (lane >> 2) * 2);

  // ---- the state, transposed: m = channels, n = this warp's leaves ---------
  float st[MCH][kNL][4];
#pragma unroll
  for (int m = 0; m < MCH; ++m)
#pragma unroll
    for (int j = 0; j < kNL; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) st[m][j][e] = 0.f;
  float mass_st[kNL][4];
#pragma unroll
  for (int j = 0; j < kNL; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) mass_st[j][e] = 0.f;

  //: THE ONES OPERAND, AN IMMEDIATE: the mass channel's value is literally 1, so -- see docs/internals/carry/carry_kernel.md#konespair
  constexpr uint32_t kOnesPair = 0x3F803F80u;
  uint32_t ones_a[4];
  ones_a[0] = (lane >> 2) == 0 ? kOnesPair : 0u;
  ones_a[1] = 0u;
  ones_a[2] = ones_a[0];
  ones_a[3] = 0u;

  //: THE LATTICE LEAF OF A PRIVATE SLOT. The private order's own `shift`/`span` -- see docs/internals/carry/carry_kernel.md#l
  auto leaf_of = [&](int P) {
    int lf = 0;
    static_for<D>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      const int dig = ((P >> PMW::shift(l)) & (PMW::span(l) - 1)) + offW[l];
      lf += dig << BP::run_shift(l);
    });
    return lf;
  };

  // ---- the state plane's page sweep, at CANONICAL atom ids -----------------
  constexpr int kAtoms = BC / kAtomLeaves;
  //: THE PAGE HOLDS SPLIT bf16 PLANES (K50): a READ-ONLY atom moves its hi plane -- see docs/internals/carry/carry_kernel.md#split-planes
  using Page = rola::state_page::Blocks<kDv>;
  //: THE PLANE IS ADDRESSED BY CANONICAL ATOM ID (P80). -- see docs/internals/carry/carry_kernel.md#atombh
  const uint32_t atomBH = (uint32_t)bh * (uint32_t)(BP::kLeaves / kAtomLeaves);
  //: THE ACTIVITY OF THIS OWNER'S ATOMS, READ ONCE AT THE PROLOGUE (P80b): one -- see docs/internals/carry/carry_kernel.md#activity
  static_assert(kAtoms <= 32, "one bit per atom per mask: an owner holds at most 32 atoms");
  uint32_t actRead = 0u, actWrite = 0u;
  static_for<kAtoms>([&](auto Ac) {
    constexpr int a = decltype(Ac)::value;
    const uint8_t bits = atom_activity(atom_bits, atomBH + BP::canon_atom(owner, a));
    actRead |= (uint32_t)((bits & kAtomRead) != 0u) << a;
    actWrite |= (uint32_t)((bits & kAtomWritten) != 0u) << a;
  });
  //: THE SLOT, NOT THE ATOM, ADDRESSES THE PLANE (`ops.cuh`'s `page_slot`, -- see docs/internals/carry/carry_kernel.md#kpagebytes
  auto page_base = [](int32_t slot) { return (uint32_t)slot * Page::kBytes; };
  //: THE PAGE'S SLICE OF A STREAM, PER FRAGMENT. -- see docs/internals/carry/carry_kernel.md#a
  auto page_regs = [&](auto Ac, auto Ld) {
    constexpr int a = decltype(Ac)::value;
    constexpr bool LOAD = decltype(Ld)::value;
    static_for<kNL>([&](auto Jc) {
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
        //: the mass rides row 0 of its own accumulator, so only the lane quad holding -- see docs/internals/carry/carry_kernel.md#near-line-443
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

  //: ---- THE ENTRY STATE. The fragments are SET, never accumulated: they are -- see docs/internals/carry/carry_kernel.md#slot
  if (s_in != nullptr) {
    const float* const plane = pin_address(s_in);
    static_for<kAtoms>([&](auto Ac) {
      //: AN ABSENT OR IDLE ATOM IS SKIPPED, AND WITH IT THE TWO BARRIERS: the load -- see docs/internals/carry/carry_kernel.md#decltype
      constexpr int a = decltype(Ac)::value;
      const int32_t slot = page_slot(page_tbl, atomBH + BP::canon_atom(owner, a));
      if (slot < 0 || (((actRead | actWrite) >> a) & 1u) == 0u) return;
      rola::state_page::load<kDv, LdO, kThreads>(so, plane, page_base(slot), tid,
                                                 ((actWrite >> a) & 1u) != 0u);
      rendezvous();
      page_regs(Ac, std::true_type{});
      rendezvous();
    });
  }

  // ---- the produce's per-thread map ---------------------------------------
  //: ONE LANE MINTS ONE TOKEN'S WHOLE ROW: the batch tile is the warp width, so a -- see docs/internals/carry/carry_kernel.md#spt
  constexpr int SPT = kLeafW;
  static_assert(C == kBallotLanes,
                "a warp's panel row is ONE lane's: the batch tile is the warp width, so a "
                "thread's slot run is the stream's whole box");
  const int prow = lane;  // this lane's token row of the batch
  int lvColR[D], lvColW[D];
  static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    lvColR[l] = runCol[l] + offR[l];
    lvColW[l] = runCol[l] + offW[l];
  });

  //: A DEAD ROW IS ZEROED AT THE NARROWEST FACTOR IT HAS (`PM::kZeroLevel`). -- see docs/internals/carry/carry_kernel.md#write
  auto produce = [&](auto Wt, const __nv_bfloat16* base, __nv_bfloat16* dst, uint32_t tokOff,
                     uint32_t gain, bool have) {
    constexpr bool WRITE = decltype(Wt)::value;
    //: EACH PHASE MINTS IN ITS OWN CARVE'S ORDER, from its own digit offsets: the -- see docs/internals/carry/carry_kernel.md#pm
    using PM = std::conditional_t<WRITE, PMW, PMR>;
    const int* const lvCol = WRITE ? lvColW : lvColR;
    //: THE SLOT ACCUMULATORS ARE THE ONLY ARRAY THE PRODUCE CARRIES.  A level's run -- see docs/internals/carry/carry_kernel.md#near-line-497
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
    //: ONE BATCH'S PANEL, expanded from factors. -- see docs/internals/carry/carry_kernel.md#nb
    auto produce_batch = [&](auto Wt, const __nv_bfloat16* base, __nv_bfloat16* dst,
                             const uint16_t* list, int o0, int ntot) {
      const int nb = (ntot - o0) < C ? (ntot - o0) : C;
      const bool have = prow < nb;
      const int tok = have ? list[o0 + prow] : 0;
      uint32_t gain = 0u;
      if constexpr (decltype(Wt)::value)
        gain = have ? splat_bf16x2(*reinterpret_cast<const __nv_bfloat16*>(
                          gather_ptr(gW, winG + (uint32_t)tok * 2u)))
                    : 0u;
      produce(Wt, base, dst, winP + (uint32_t)tok * (uint32_t)(BP::kWtot * 2), gain, have);
    };

    // ---- the liveness pass: one READ list per warp, ONE write list --------
    //: THE SCHEDULE IS TWO-SIDED (Blake ruling 2026-08-19). Per level at most ONE -- see docs/internals/carry/carry_kernel.md#lw
    constexpr int LW = G::kLW;
    static_assert(W <= 0xFFFF, "a token's window-local index must fit the list's element");
    static_assert(LW <= kBallotLanes,
                  "the walk gives one ballot word to each lane, so a window is at most a "
                  "lockstep unit's worth of words");
    const int room = wLen - lane * kBallotLanes;
    const bool inW = lane < LW;
    const uint32_t valid =
        (!inW || room <= 0) ? 0u : (room >= kBallotLanes ? 0xFFFFFFFFu : ((1u << room) - 1u));
    //: a lane past the window's last word holds no word; it reads word zero, which -- see docs/internals/carry/carry_kernel.md#widx
    const int wIdx = (tok0 >> 5) + (inW ? lane : 0);
    const uint32_t wordB = (uint32_t)(wIdx < tok_words ? wIdx : tok_words - 1) * 4u;

    //: the trip count is the window's LIVENESS, not its width: one `bit_count` -- see docs/internals/carry/carry_kernel.md#cnt
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
    //: THE CLAUSE ABOVE THE CARVE: the levels the stream spans WHOLE, whose row is the -- see docs/internals/carry/carry_kernel.md#clauseof
    auto clause_of = [&](auto Wt, const uint32_t* rowsS, const uint32_t* rowOffS) {
      using SVS = std::conditional_t<decltype(Wt)::value, SVW, SVR>;
      using CRS = std::conditional_t<decltype(Wt)::value, CRW, CRR>;
      uint32_t m = valid;
      static_for<D>([&](auto Lc) {
        constexpr int l = decltype(Lc)::value;
        if constexpr (!SVS::carves(l) && CRS::row_g(l) > 0)
          m &= *reinterpret_cast<const uint32_t*>(gather_ptr(rowsS, rowOffS[l] + wordB));
      });
      return m;
    };
    //: AND THE CARVED CLAUSE OF ONE STREAM, on that side's OWN carve: the AND over -- see docs/internals/carry/carry_kernel.md#streamrow
    auto stream_row = [&](auto Wt, const uint32_t* rowsS, const uint32_t* rowOffS, int s) {
      using SVS = std::conditional_t<decltype(Wt)::value, SVW, SVR>;
      uint32_t m = 0xFFFFFFFFu;
      static_for<D>([&](auto Lc) {
        constexpr int l = decltype(Lc)::value;
        if constexpr (SVS::carves(l)) {
          const uint32_t c = (uint32_t)((s >> ilog2c(SVS::grid_div(l))) & (SVS::g(l) - 1));
          m &= *reinterpret_cast<const uint32_t*>(
              gather_ptr(rowsS, rowOffS[l] + c * rowBytes + wordB));
        }
      });
      return m;
    };

    const uint32_t clauseR = clause_of(std::false_type{}, rowsR, rowOffR);
    const uint32_t clauseW = clause_of(std::true_type{}, rowsW, rowOffW);
    uint32_t orR = 0u, orW = 0u, mineR = 0u;
#pragma unroll
    for (int s = 0; s < kWarps; ++s) {
      const uint32_t r = stream_row(std::false_type{}, rowsR, rowOffR, s);
      orR |= r;
      mineR = (s == strm) ? r : mineR;
      orW |= stream_row(std::true_type{}, rowsW, rowOffW, s);
    }
    const uint32_t liveW = clauseW & orW;
    //: THE WHOLE-WINDOW SKIP, at (owner, window) grain, per side -- the block-bitmap -- see docs/internals/carry/carry_kernel.md#continue
    if (lane_ballot(((clauseR & orR) | liveW) != 0u) == 0u) continue;

    uint16_t* const rlist = slist + warp * kListStride;
    uint16_t* const wlist = slist + kWarps * kListStride;
    const int nr = walk(clauseR & mineR, rlist, 0xFFFFFFFFu, 0u);
    __syncwarp();
    //: THE WRITE LIST IS ONE LIST WITH `kWarps` WRITERS. Every warp folds every -- see docs/internals/carry/carry_kernel.md#kbitsw
    constexpr uint32_t kBitsW = (uint32_t)(kBallotLanes / kWarps);
    const int nw =
        walk(liveW, wlist, ((1u << kBitsW) - 1u) << (warp * kBitsW), (1u << (warp * kBitsW)) - 1u);

    //: THE TRANSIT MOVES THE HI PLANE, IT DOES NOT COPY IT (K35 F1b). -- see docs/internals/carry/carry_kernel.md#the-exchange
    uint32_t sa[kDw][MCH][4], ma[kDw][4];
    uint32_t st_lo[MCH][kNL][2], mass_lo[kNL];
    //: THE READ PHASE SHARES NOTHING: the list, the panel region, the staging block -- see docs/internals/carry/carry_kernel.md#nbr
    const int nbr = (nr + C - 1) / C;
    static_for<MCH>([&](auto Mc) {
      constexpr int m = decltype(Mc)::value;
      static_for<kNL>([&](auto Jc) {
        constexpr int j = decltype(Jc)::value;
        st_lo[m][j][0] = pack_lo(st[m][j][0], st[m][j][1]);
        st_lo[m][j][1] = pack_lo(st[m][j][2], st[m][j][3]);
      });
    });
    static_for<kNL>([&](auto Jc) {
      constexpr int j = decltype(Jc)::value;
      mass_lo[j] = pack_lo(mass_st[j][0], mass_st[j][1]);
    });
    if constexpr (G::kTied) {
      //: THE TIE MOVES THE HI PLANE BY RENAMING IT, and pays no region, no barrier -- see docs/internals/carry/carry_kernel.md#tied-hybrid
      static_for<kDw>([&](auto Kc) {
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
      //: THE PRODUCE RUNS ONE BATCH AHEAD, into the other panel of the two-slot ring, -- see docs/internals/carry/carry_kernel.md#syncwarp
      if (nbr > 0) produce_batch(std::false_type{}, pR, panR, rlist, 0, nr);
      __syncwarp();
    } else {
      //: THE WAR EDGE against the PREVIOUS window's fold, which still reads the bytes -- see docs/internals/carry/carry_kernel.md#war
      rendezvous();
      //: THE OUT LEG'S COVER: read batch zero's factor gathers touch neither `st` nor -- see docs/internals/carry/carry_kernel.md#cover
      if (nbr > 0) produce_batch(std::false_type{}, pR, panR, rlist, 0, nr);
      static_for<kNL>([&](auto Jc) {
        constexpr int j = decltype(Jc)::value;
        static_for<4>([&](auto Ec) {
          constexpr int e = decltype(Ec)::value;
          constexpr int dl = XM::delta(j * kMmaN + (e & 1));
          static_for<MCH>([&](auto Mc) {
            constexpr int m = decltype(Mc)::value;
            constexpr int gr = (2 * m + (e >> 1)) ^ XM::swz(dl);
            store_shared_u16(pubC + (uint32_t)(dl * LdX * 2) + (uint32_t)((gr ^ swzLane) * 16),
                             hi_bf16(st[m][j][e]));
          });
        });
        if ((lane >> 2) == 0) {
          static_for<2>([&](auto Ec) {
            constexpr int e = decltype(Ec)::value;
            constexpr int dl = XM::delta(j * kMmaN + e);
            store_shared_u16(
                pubM + (uint32_t)(dl * LdX * 2) + (uint32_t)((XM::msw(dl) ^ mswLane) * 4),
                hi_bf16(mass_st[j][e]));
          });
        }
      });
      rendezvous();

      static_for<kDw>([&](auto Kc) {
        constexpr int kk = decltype(Kc)::value;
        constexpr int rk = kk * kMmaK;
        const int swzK = swzPull ^ XM::swz(rk);
        static_for<MCH>([&](auto Mc) {
          constexpr int m = decltype(Mc)::value;
          load_frag_t(sa[kk][m],
                      pullC + (uint32_t)(rk * LdX * 2) + (uint32_t)((((2 * m) | cT) ^ swzK) * 16));
        });
        //: the mass's two pairs.  `msw` ignores the row's low three bits, so a pair's -- see docs/internals/carry/carry_kernel.md#msw
        ma[kk][0] = 0u;
        ma[kk][1] = 0u;
        ma[kk][2] = 0u;
        ma[kk][3] = 0u;
        if ((lane >> 2) == 0) {
          const uint32_t bLo =
              pullM + (uint32_t)(rk * LdX * 2) + (uint32_t)((XM::msw(rk) ^ mswPull) * 4);
          const uint32_t bHi =
              pullM + (uint32_t)((rk + 8) * LdX * 2) + (uint32_t)((XM::msw(rk + 8) ^ mswPull) * 4);
          ma[kk][0] = (uint32_t)load_shared_u16(bLo)
                      | ((uint32_t)load_shared_u16(bLo + (uint32_t)(LdX * 2)) << 16);
          ma[kk][2] = (uint32_t)load_shared_u16(bHi)
                      | ((uint32_t)load_shared_u16(bHi + (uint32_t)(LdX * 2)) << 16);
        }
      });
      rendezvous();
    }

    // ---- PHASE 1: the READOUT of every read batch, against the window's ENTRY
    //      state.  ALL of the window's readouts precede ANY of its folds, because a
    //      within-window pair is the INTRA term's and this kernel must not carry it.
    //      The batch line is therefore the window line on both sides.

#pragma unroll 1
    for (int i = 0; i < nbr; ++i) {
      const int off_ = i * C;
      const int n = (nr - off_) < C ? (nr - off_) : C;
      const uint32_t brT = (i & 1) ? brT1 : brT0;

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
        uint32_t rb[kDw][4];
#pragma unroll
        for (int kk = 0; kk < kDw; ++kk) load_frag(rb[kk], ADDR_S(brT, mb * 16, 2 * kk));
#pragma unroll
        for (int kk = 0; kk < kDw; ++kk)
#pragma unroll
          for (int p = 0; p < 2; ++p) {
#pragma unroll
            for (int m = 0; m < MCH; ++m) mma(acc[p][m], sa[kk][m], &rb[kk][2 * p]);
            mma(macc[p], ma[kk], &rb[kk][2 * p]);
          }
        //: THE FAN-IN IS WARP-PRIVATE AND DIRECT (K33 1.4): one writer per read entry -- see docs/internals/carry/carry_kernel.md#j
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

    //: THE WINDOW LINE IS THE ONE BARRIER LEFT ON THE READ SIDE, and it is there by -- see docs/internals/carry/carry_kernel.md#rendezvous-2
    rendezvous();

    // ---- PHASE 2: the FOLD of every write batch ---------------------------
    //: V IS A WRITE-SIDE QUANTITY and is gathered by the WRITE list; the read list -- see docs/internals/carry/carry_kernel.md#nbw
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
    if (nbw > 0) produce_batch(std::true_type{}, pW, panR, wlist, 0, nw);

    //: THE RETURN LEG.  The readout never modified the state, so the hi plane comes -- see docs/internals/carry/carry_kernel.md#return-leg
    uint32_t hi[kNL][2 * MCH], mhi[kNL];
    static_assert(MCH % 2 == 0 && (2 * MCH) % 4 == 0,
                  "the writer pulls its hi plane four granules at a time, so the channel "
                  "m-tiles must pair into whole `ldmatrix.x4` groups");
    if constexpr (G::kTied) {
      static_for<kNL>([&](auto Jc) {
        constexpr int j = decltype(Jc)::value;
        static_for<MCH>([&](auto Mc) {
          constexpr int m = decltype(Mc)::value;
          hi[j][2 * m] = sa[j / 2][m][2 * (j & 1)];
          hi[j][2 * m + 1] = sa[j / 2][m][2 * (j & 1) + 1];
        });
        mhi[j] = ma[j / 2][2 * (j & 1)];
      });
    } else {
      //: THE STORE-BACK IS THE PULL RUN BACKWARDS: the reader writes each A fragment -- see docs/internals/carry/carry_kernel.md#store-back
      static_for<kDw>([&](auto Kc) {
        constexpr int kk = decltype(Kc)::value;
        static_for<4>([&](auto Ic) {
          constexpr int i = decltype(Ic)::value;
          constexpr int rr = kk * kMmaK + (i >> 1) * 8;
          const uint32_t a0 = sbC + (uint32_t)(rr * LdX * 2);
          static_for<MCH>([&](auto Mc) {
            constexpr int m = decltype(Mc)::value;
            constexpr int gr = (2 * m + (i & 1)) ^ XM::swz(rr);
            const uint32_t a = a0 + (uint32_t)((gr ^ swzSB) * 16);
            store_shared_u16(a, (uint16_t)(sa[kk][m][i] & 0xFFFFu));
            store_shared_u16(a + (uint32_t)(LdX * 2), (uint16_t)(sa[kk][m][i] >> 16));
          });
        });
        if ((lane >> 2) == 0) {
          constexpr int rk = kk * kMmaK;
          const uint32_t bLo =
              pullM + (uint32_t)(rk * LdX * 2) + (uint32_t)((XM::msw(rk) ^ mswPull) * 4);
          const uint32_t bHi =
              pullM + (uint32_t)((rk + 8) * LdX * 2) + (uint32_t)((XM::msw(rk + 8) ^ mswPull) * 4);
          store_shared_u16(bLo, (uint16_t)(ma[kk][0] & 0xFFFFu));
          store_shared_u16(bLo + (uint32_t)(LdX * 2), (uint16_t)(ma[kk][0] >> 16));
          store_shared_u16(bHi, (uint16_t)(ma[kk][2] & 0xFFFFu));
          store_shared_u16(bHi + (uint32_t)(LdX * 2), (uint16_t)(ma[kk][2] >> 16));
        }
      });
      rendezvous();
      //: THE WRITER PULLS IN ITS OWN LAYOUT.  Its eight slots of one n-tile are eight -- see docs/internals/carry/carry_kernel.md#writer-pull
      static_for<kNL>([&](auto Jc) {
        constexpr int j = decltype(Jc)::value;
        constexpr int dj = XM::delta(j * kMmaN);
        const uint32_t a0 = retC + (uint32_t)(dj * LdX * 2);
        static_for<MCH / 2>([&](auto Tc) {
          constexpr int t = decltype(Tc)::value;
          uint32_t r4[4];
          load_frag_t(r4, a0 + (uint32_t)((((4 * t + retG) ^ swzRet ^ XM::swz(dj)) * 16)));
          static_for<4>([&](auto Rc) {
            constexpr int r = decltype(Rc)::value;
            hi[j][4 * t + r] = r4[r];
          });
        });
        mhi[j] = 0u;
        if ((lane >> 2) == 0) {
          constexpr int dl0 = XM::delta(j * kMmaN), dl1 = XM::delta(j * kMmaN + 1);
          const uint32_t m0 = (uint32_t)load_shared_u16(pubM + (uint32_t)(dl0 * LdX * 2)
                                                        + (uint32_t)((XM::msw(dl0) ^ mswLane) * 4));
          const uint32_t m1 = (uint32_t)load_shared_u16(pubM + (uint32_t)(dl1 * LdX * 2)
                                                        + (uint32_t)((XM::msw(dl1) ^ mswLane) * 4));
          mhi[j] = m0 | (m1 << 16);
        }
      });
    }
    //: THE REJOIN IS EXACT AND IT IS NOT ARITHMETIC. -- see docs/internals/carry/carry_kernel.md#rejoin
    static_for<MCH>([&](auto Mc) {
      constexpr int m = decltype(Mc)::value;
      static_for<kNL>([&](auto Jc) {
        constexpr int j = decltype(Jc)::value;
        static_for<2>([&](auto Hc) {
          constexpr int h = decltype(Hc)::value;
          st[m][j][2 * h] = join_hi_lo(hi[j][2 * m + h], st_lo[m][j][h], 0);
          st[m][j][2 * h + 1] = join_hi_lo(hi[j][2 * m + h], st_lo[m][j][h], 1);
        });
      });
    });
    //: THE MASS RIDES ROW ZERO of its own accumulator, so its other rows are DEAD -- see docs/internals/carry/carry_kernel.md#rejoin-mass
    static_for<kNL>([&](auto Jc) {
      constexpr int j = decltype(Jc)::value;
      const bool row0 = (lane >> 2) == 0;
      mass_st[j][0] = row0 ? join_hi_lo(mhi[j], mass_lo[j], 0) : 0.f;
      mass_st[j][1] = row0 ? join_hi_lo(mhi[j], mass_lo[j], 1) : 0.f;
      mass_st[j][2] = 0.f;
      mass_st[j][3] = 0.f;
    });
#pragma unroll 1
    for (int i = 0; i < nbw; ++i) {
      const uint32_t bwF = (i & 1) ? bwF0 : bwF1;
      const uint32_t vsl = svB + (uint32_t)(i % STAGES) * kVSlot;
      stage_wait<STAGES - 2>();
      rendezvous();
      //: the ring slot is advanced on EVERY batch, live or not, so the wait/commit -- see docs/internals/carry/carry_kernel.md#f
      const int f = i + STAGES - 1;
      if (f < nbw) stage_batch(f, f % STAGES);
      stage_commit();

      //: §6: kNL/2 is compile-time (kNL is a constexpr local above) but the heuristic -- see docs/internals/carry/carry_kernel.md#knlhalf
      constexpr int kNLHalf = kNL / 2;
#pragma unroll
      for (int kk = 0; kk < MTOK; ++kk) {
        uint32_t a[MCH][4];
#pragma unroll
        for (int m = 0; m < MCH; ++m) load_frag_t(a[m], ADDR_V(vsl, kk * 16, m * kMmaM));
#pragma unroll
        for (int nj = 0; nj < kNLHalf; ++nj) {
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
      produce_batch(std::true_type{}, pW, ((i + 1) & 1) ? panW : panR, wlist, (i + 1) * C, nw);
    }
    //: the ring's unconsumed groups are retired before the next window reuses the -- see docs/internals/carry/carry_kernel.md#near-line-843
    stage_wait<0>();
  }

  // ---- the FINAL STATE, at CANONICAL atom ids ------------------------------
  //: the mirror of the entry load, after the last fan-in has consumed the staging -- see docs/internals/carry/carry_kernel.md#rendezvous
  if (s_out != nullptr) {
    float* const plane = pin_address(s_out);
    rendezvous();
    static_for<kAtoms>([&](auto Ac) {
      //: THE EXIT STORES ONLY WRITTEN, PLANNED ATOMS.  An atom this call does not -- see docs/internals/carry/carry_kernel.md#decltype-2
      constexpr int a = decltype(Ac)::value;
      const int32_t slot = page_slot(page_tbl, atomBH + BP::canon_atom(owner, a));
      if (slot < 0 || ((actWrite >> a) & 1u) == 0u) return;
      page_regs(Ac, std::false_type{});
      rendezvous();
      rola::state_page::store<kDv, LdO, kThreads>(so, plane, page_base(slot), tid);
      rendezvous();
    });
  }
#undef ADDR_S
#undef ADDR_V
}

}  // namespace rola::carry
