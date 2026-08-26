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
static constexpr int kCarryBuildMagic = 31010020;

static constexpr int kDv = 64;         // d_v
static constexpr int kCols = kDv + 1;  // the mass column rides the state
static constexpr int kPadE = 8;        // bf16 row padding, elements (16 B)
static constexpr int kPadF = 4;        // fp32 row padding, elements (16 B)
//: THE CHANNEL-SPLIT BODY'S SHAPE, and the reverse family's.  Those bodies split the
//: `kDv` CHANNELS over a fixed four warps and hold the whole box in each; the CARVED
//: body's executor count is the arm's STREAM count instead (`CarryGeom::kWarps`) and
//: is not a constant of this file.
static constexpr int kWarps = 4;
static constexpr int kThreads = kWarps * 32;
static_assert(kAtomLeaves == kMmaK, "the page granule is one MMA k-step");

//: A COMPILE-TIME LOOP.  Every per-level quantity here is a template argument or a
//: register-array index, so the level index must be a constant expression; a
//: `for (int l = 0; l < D; ++l)` would make `load_run<n_l(l)>` ill-formed and would
//: turn every per-level array index dynamic, which is a stack frame.
//: a type carried through a generic lambda's argument, which is how a body that is
//: instantiated per SIDE names that side's span vector without a second template.
template <class T>
struct type_tag {
  using type = T;
};

template <int N, class F>
__device__ __forceinline__ void static_for(F&& f) {
  if constexpr (N > 0) {
    static_for<N - 1>(f);
    f(std::integral_constant<int, N - 1>{});
  }
}

//: THE CARVE (K35 multi-level).  A STREAM owns an affine RECTANGLE of the owner's
//: digits -- a per-level span vector, `prod_l S_l` leaves and ALL `kCols` channels
//: -- and tiles those leaves in a private mixed-radix order.  That tiling is private
//: to the stream: the lattice's `local` order, the state plane, decode and the facts
//: geometry never see it, so the readout needs no state image, no `ldmatrix` for A
//: and no CTA barrier.  The two sides carve the SAME box along different axes, which
//: is why the state evacuates registers at each phase change (the transit plane).
//:
//: THE EXECUTOR MAP (K35 S3a), under the terminology law: a STREAM owns a box, an
//: EXECUTOR (a warp on sm_80) runs streams, and the two counts are independent dials
//: -- streams are the in-CTA sparsity dial, warps the execution budget.  `WPS` is the
//: arm's declared WARPS PER WRITE STREAM: `kWarps = SVW::kStreams * WPS`, a warp's
//: write stream is `strm >> log2 WPS` and its COLUMN HALF is `strm & (WPS-1)`.  The
//: READ carve is declared at ONE STREAM PER WARP, so the readout is untouched by the
//: split and the accumulator's leaf count is the read box's at every arm.

template <int D, int B, int K, int M, int W, int C, int WPS, int W0, int W1, int W2, int W3, int R0,
          int R1, int R2, int R3>
struct CarryGeom {
  using BP = BoxPlan<D, B, K, M>;
  using SVW = StreamSpans<D, B, K, M, W0, W1, W2, W3>;
  using SVR = StreamSpans<D, B, K, M, R0, R1, R2, R3>;
  static constexpr int BC = BP::kBC;
  //: THE TWO DIALS: `kStreams` is the WRITE side's ownership grain, `kWarps` the issue
  //: budget spent on it.
  static constexpr int kWPS = WPS;
  static constexpr int kStreams = SVW::kStreams;
  static constexpr int kWarps = kStreams * WPS;
  static constexpr int kThreads = kWarps * 32;
  static_assert(WPS >= 1 && (WPS & (WPS - 1)) == 0, "warps per write stream is a power of two");
  static_assert(SVR::kStreams == kWarps,
                "the READ carve declares one stream per WARP, so its stream count IS the "
                "executor count and the readout carries no column split");
  //: the WRITE STREAM's leaves -- the segment row's width -- and the EXECUTOR's own
  //: column half of them, which is the accumulator's width and the read box's size.
  static constexpr int kLeafW = SVW::kLeaf;
  static constexpr int kLeafE = kLeafW / WPS;
  static_assert(kLeafE * WPS == kLeafW && kLeafE == SVR::kLeaf,
                "a column half of the write stream's box is exactly one read stream's box");
  static constexpr int LdV = kDv + kPadE;
  static constexpr int LdS = kLeafW + kPadE;
  static constexpr int LdR = kLeafE + kPadE;
  static constexpr int LdO = kDv + kPadF;
  //: THE STRIP IS THE BATCH'S TOKEN TILE, and it is the MMA's whole `k` step: a batch
  //: of the union walk is `CP` tokens, a stream's rank inside it is below `CP` by
  //: construction, so ONE segment slot is exactly ONE contraction step.
  static constexpr int CP = kMmaK;
  static_assert(2 * CP == C, "a half holds exactly two strips");

  // ---- THE FOLD'S TENANTS (K35 S2, the cooperative fetch) ------------------
  //: THE PER-STREAM SEGMENT, SINGLE SLOT.  `SegValueRows` is `[stream][row][channel]`
  //: at row stride `LdV`, `SegCoeffRows` is `[stream][row][slot]` at row stride `LdS`;
  //: a ROW is one (token, stream) pair, placed by the FETCHER at the token's rank
  //: inside the batch.  The two sub-tiles are adjacent halves of one segment and are
  //: separate only because `ldmatrix` reads each at its own row stride.
  static constexpr int kSegRows = CP;
  static constexpr int seg_v_bytes = kStreams * kSegRows * LdV * (int)sizeof(__nv_bfloat16);
  static constexpr int seg_c_bytes = kStreams * kSegRows * LdS * (int)sizeof(__nv_bfloat16);
  //: THE WINDOW'S COEFFICIENT STAGING, as a ROLLING BLOCK: the fetcher expands from
  //: SMEM, and what it needs at a moment is the compressed factor rows of ONE batch's
  //: union tokens (`BP::kStageW` amplitudes each -- the owner's per-level runs, not
  //: the topology-wide packed row).  Two blocks, so the block for batch `b + 2` is in
  //: flight while `b + 1` is expanded: the same two-stages-in-flight law the segments
  //: obey, at a cost that does not grow with `W`.
  static constexpr int kStgSlots = 2;
  static constexpr int stg_bytes = kStgSlots * kSegRows * BP::kStageW * (int)sizeof(__nv_bfloat16);
  static constexpr int gain_bytes = align16c(kStgSlots * kSegRows * (int)sizeof(__nv_bfloat16));
  //: the state plane's page, at entry and exit only -- dead for the whole window loop,
  //: which is why the transit straddles it too.
  static constexpr int out_bytes = kAtomLeaves * LdO * (int)sizeof(float);

  //: ---- K35-E: THE TRANSIT PLANE -------------------------------------------
  //: The state EVACUATES registers at each phase change, because the readout and the
  //: fold carve the same leaves along DIFFERENT lattice axes.  The plane is
  //: `[channel][leaf]`, which is A row-major for the readout; its rows are the `kDv`
  //: value channels plus ONE mass row -- the mass accumulator's rows 1..15 are
  //: identically zero by construction of `ones_a`, so moving them would move zeros.
  static constexpr int kPlaneRows = kDv + 1;
  //: `LD` AND ITS SWIZZLE ARE A DERIVATION, NOT A GUESS (findings, derivation 2).
  //: `word ^= (chan & 7)` at `LD = BC + 16` measures max-way ONE on both legs.
  static constexpr int kHiLD = BC + 16;
  static constexpr int kHiSwz = 7;
  static constexpr int plane_bytes = kPlaneRows * kHiLD * (int)sizeof(__nv_bfloat16);
  //: THE TRANSIT STRADDLES THE FOLD'S DEAD TENANTS AND PAYS ONLY THE SHORTFALL.  The
  //: segments, the coefficient staging and the page buffer are dead for the whole
  //: readout, which is exactly the plane's lifetime; where they do not cover it the
  //: allocation carries a TAIL, and that tail is the plane's only private cost.
  static constexpr int fold_dead_bytes =
      seg_v_bytes + seg_c_bytes + stg_bytes + gain_bytes + out_bytes;
  static constexpr int tail_bytes = plane_bytes > fold_dead_bytes
                                        ? align16c(plane_bytes - fold_dead_bytes)
                                        : 0;

  // ---- THE READOUT'S TENANTS -----------------------------------------------
  //: THE READ PANELS ARE A TWO-SLOT RING.  A window's readouts all precede its folds,
  //: so the read side produces one batch AHEAD inside its own half, which is what puts
  //: the produce's global-load latency underneath the previous batch's MMA block.
  static constexpr int read_bytes = 2 * kWarps * CP * LdR * (int)sizeof(__nv_bfloat16);
  //: ONE READ LIST PER STREAM, plus the CTA's two shared ones.  The write side keeps
  //: NO per-stream list any more: the UNION list is the whole schedule and a stream's
  //: membership is its published ballot word.
  static constexpr int kLists = kWarps + 2;
  static constexpr int kListStride = ((W + 7) / 8) * 8;
  static constexpr int list_bytes = align16c(kLists * kListStride * (int)sizeof(uint16_t));
  //: THE WINDOW'S BALLOT WORDS, one per lane of the walk -- so a window may be at
  //: most a lockstep unit's worth of words wide.
  static constexpr int kLW = W / kBallotLanes;
  //: THE PUBLISHED MEMBERSHIP AND ITS PREFIX COUNTS.  A fetcher places rows into
  //: segments it does not consume, so it needs EVERY stream's write-side ballot and
  //: the per-word prefix popcount that turns a token into a RANK.  Each executor
  //: publishes its own once per window; `kStreams * kLW` words is the whole index.
  static constexpr int mem_bytes = align16c(kStreams * kLW * (int)sizeof(uint32_t));
  static constexpr int pfx_bytes = align16c(kStreams * kLW * (int)sizeof(uint16_t));
  //: THE PER-TILE LIVENESS MASK, ONE WORD PER EXECUTOR: bit `nj` says the batch wrote a
  //: nonzero coefficient somewhere in the write stream's n-tile pair `nj`, so skipping
  //: that pair's `ldmatrix` and MMAs is EXACT -- a tile with no live leaf contributes
  //: nothing.  The word is per EXECUTOR and not per stream so that a warp RESETS ONLY
  //: WHAT IT READS: the reset sits between a warp's own read of the mask and the
  //: barrier that lets the fetchers OR into it, and a word shared with the stream's
  //: other warp would be reset out from under that warp's read.  The `WPS` words of one
  //: stream carry the same bits; each warp tests its own column half's field of them.
  static constexpr int mask_bytes = align16c(kWarps * (int)sizeof(uint32_t));

  //: `kRegFloor` is the DECLARED register floor for this family -- the measured census
  //: of the heaviest built arm, recorded in the findings doc.  The cooperative fetch
  //: raised it: the consumer now holds a batch's value AND coefficient fragments across
  //: the fetch phase (F4's second stage IS the register file), which is `MCH*4 + NL/2*4`
  //: registers more than the produce it replaces.  The residency it derives is
  //: unchanged at both thread counts -- 64K/(128*235) and 64K/(256*235) round to the
  //: same CTA ceilings 220 gave -- so this is a truer statement of the same geometry.
  static constexpr int kRegFloor = 235;
  //: ONE slot per stream, because the SECOND STAGE IS REGISTERS (F4).
  static constexpr int kStages = 1;

  // ---- the allocation, as offsets -----------------------------------------
  static constexpr int off_seg_v = 0;
  static constexpr int off_seg_c = off_seg_v + seg_v_bytes;
  static constexpr int off_stg = off_seg_c + seg_c_bytes;
  static constexpr int off_gain = off_stg + stg_bytes;
  static constexpr int off_out = off_gain + gain_bytes;
  static constexpr int off_read = off_out + out_bytes + tail_bytes;
  static constexpr int off_list = off_read + read_bytes;
  static constexpr int off_mem = off_list + list_bytes;
  static constexpr int off_pfx = off_mem + mem_bytes;
  static constexpr int off_mask = off_pfx + pfx_bytes;
  static constexpr size_t bytes = (size_t)(off_mask + mask_bytes);
  static constexpr int cta_target = design::cta_target((int)bytes, kThreads, kRegFloor);
  static_assert(plane_bytes <= fold_dead_bytes + tail_bytes,
                "the transit must fit the fold's dead region plus its declared tail");
  static_assert(design::smem_fits(bytes), "this arm does not fit the SMEM maximum");
};

//: THE OWNER-GRAIN PRODUCE MAP, which the reverse family's panel is expanded
//: against (`backward_kernel.cuh`): a row's `kRowThr` threads split the box's `BC`
//: coefficients into `SPT` contiguous LATTICE slots each.  Because the lattice index
//: is a bit permutation (R0 spec section 2.4), a thread's slot run is a sub-lattice:
//: per level the digits it touches are a CONTIGUOUS run of the owner's own run at a
//: runtime base, and WHICH element of that run each of its slots wants is a
//: compile-time function of the unrolled slot index.
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

//: THE PRIVATE LEAF ORDER AND THE PRODUCE'S THREAD MAP.  A stream's leaves are
//: numbered by a MIXED RADIX over its own per-level spans -- `p = sum_l r_l <<
//: shift(l)`, with `r_l` the digit's offset inside the stream's range at level `l`
//: -- so `p` is a bit permutation of the levels' digit offsets exactly as `local`
//: is, and the expansion is `sum_l n_l` amplitude loads followed by products of
//: register values at CONSTANT indices: no dynamic indexing, no spill, no
//: materialized gate plane.  A span-1 level contributes a constant and drops out of
//: the order on its own.
//:
//: THE PRIVATE-ORDER RULE, which fixes the order from the span vector alone:
//:   1. level `D - 1` is INNERMOST.  The transit plane pairs `bf16x2` on
//:      LATTICE-ADJACENT leaves, so the level lying lowest in `p` must be the level
//:      lying lowest in `local`, and its span must be at least two.  THAT CLAUSE IS
//:      THE CURRENT EXCHANGE'S LIMITATION, NOT THE DESIGN'S: it comes from the plane
//:      publishing pairs, and the rebuilt exchange (writer-side placement into
//:      per-reader `[leaf][channel]` segments) admits any span vector.
//:   2. the remaining levels keep the LATTICE order, except the levels this side
//:      CARVES, which move OUTERMOST keeping their lattice order among themselves.
//: A side carves where it is declared sparse, so rule 2 is "sparse outermost, dense
//: innermost": an eight-slot n-tile spans dense digits and each sparse level indexes
//: WHOLE tiles, where the digit skip is exact.
template <int D, int B, int K, int M, int C, class SV>
struct SpanMap {
  using BP = BoxPlan<D, B, K, M>;
  static constexpr int kLeaf = SV::kLeaf;
  static constexpr int kRowThr = 32 / C;
  static constexpr int SPT = kLeaf / kRowThr;
  //: `kRowThr` lanes share a token's panel row, each minting `SPT` contiguous private
  //: slots.  At the strip shape that is TWO (32 lanes, 16 tokens), and the produce
  //: then needs the lane's slot base folded back into each level's run column --
  //: `adj` in `produce` (the 2.1 paper derivation for this shape).
  static_assert(kRowThr * SPT == kLeaf && (kRowThr & (kRowThr - 1)) == 0,
                "a token's panel row splits over a power-of-two lane group");
  static constexpr int span(int l) { return SV::s(l); }
  static constexpr int bits(int l) { return ilog2c(span(l)); }
  //: rule 2 as a strict order on the levels beneath `D - 1`, and `rank` as the
  //: position it induces -- `0` innermost.  Both are counts, so neither tabulates.
  static constexpr bool before(int a, int b) {
    return SV::carves(a) != SV::carves(b) ? !SV::carves(a) : a > b;
  }
  static constexpr int count_before(int l, int i) {
    return i >= D - 1 ? 0 : (before(i, l) ? 1 : 0) + count_before(l, i + 1);
  }
  static constexpr int rank(int l) { return l == D - 1 ? 0 : 1 + count_before(l, 0); }
  static constexpr int shift_from(int l, int i) {
    return i >= D ? 0 : (rank(i) < rank(l) ? bits(i) : 0) + shift_from(l, i + 1);
  }
  static constexpr int shift(int l) { return shift_from(l, 0); }
  static constexpr int vary(int l) { return SPT >> shift(l); }
  static constexpr int n_l(int l) {
    return vary(l) < 1 ? 1 : (vary(l) > span(l) ? span(l) : vary(l));
  }
  static constexpr int idx_of(int l, int j) { return (j >> shift(l)) & (n_l(l) - 1); }
  //: the ADJACENT PAIR comes from the level lying LOWEST in `p`, which rule 1 makes
  //: the lattice's innermost level: a lane's two adjacent slots are two adjacent
  //: digits of that level, hence two adjacent leaves.
  static constexpr int kPairLevel = D - 1;
  static_assert(shift(kPairLevel) == 0 && n_l(kPairLevel) >= 2 && span(D - 1) >= 2,
                "the lattice's innermost level lies lowest in the private order and supplies "
                "the pair the transit plane publishes (rule 1: the CURRENT exchange's "
                "requirement, which the rebuilt one lifts)");
};

//: a contiguous run of `n` bf16, at the widest access that covers it.  The runs are
//: naturally aligned by construction (a level's column base is a multiple of `B`, an
//: owner's run starts at a multiple of `s_l` and a group's sub-run at a multiple of
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

//: THE PER-LEVEL MODE DECLARATION, as the kernel reads it (the seam's own encoding):
//: two bits per level, level 0 in the low pair, bit 0 = the READ side is value-sparse
//: there, bit 1 = the WRITE side.  The kernel consumes it to PRUNE its clause: a side
//: reads a level's row only where that side is declared sparse.
constexpr uint32_t kSideRead = 1u, kSideWrite = 2u;
constexpr bool mode_declares(uint32_t modes, int l, uint32_t side) {
  return ((modes >> (2 * l)) & side) != 0u;
}

//: THE SECOND LAUNCH BOUND: the CTA count this arm can actually reach, taken from
//: the same geometry the ring depth is.  A ring sized for a residency the launch
//: bound does not ask for is a ring sized for nothing.
template <int D, int B, int K, int M, int W, int C, int WPS, int W0, int W1, int W2, int W3, int R0,
          int R1, int R2, int R3>
constexpr int kCarryCtaTarget =
    CarryGeom<D, B, K, M, W, C, WPS, W0, W1, W2, W3, R0, R1, R2, R3>::cta_target;

// ---------------------------------------------------------------- the kernel

template <int D, int B, int K, int M, int W, int C, int WPS, int W0, int W1, int W2, int W3, int R0,
          int R1, int R2, int R3, uint32_t MO, bool STATE>
__global__
__launch_bounds__((CarryGeom<D, B, K, M, W, C, WPS, W0, W1, W2, W3, R0, R1, R2, R3>::kThreads),
                  (kCarryCtaTarget<D, B, K, M, W, C, WPS, W0, W1, W2, W3, R0, R1, R2, R3>)) void
carry_kernel(const __nv_bfloat16* __restrict__ pread, const __nv_bfloat16* __restrict__ pwrite,
             const __nv_bfloat16* __restrict__ gwrite, const __nv_bfloat16* __restrict__ v,
             const uint32_t* __restrict__ rows, float* __restrict__ o, float* __restrict__ den,
             int L, int tok_words, int v_row_bytes, const float* __restrict__ s_in,
             float* __restrict__ s_out) {
  using BP = BoxPlan<D, B, K, M>;
  using G = CarryGeom<D, B, K, M, W, C, WPS, W0, W1, W2, W3, R0, R1, R2, R3>;
  //: THE TWO SPAN VECTORS.  The READ side's carve and the WRITE side's carve of the
  //: same box, each with its own private order under the rule above.
  using SVR = StreamSpans<D, B, K, M, R0, R1, R2, R3>;
  using SVW = StreamSpans<D, B, K, M, W0, W1, W2, W3>;
  using PM = SpanMap<D, B, K, M, G::CP, SVR>;
  using PMW = SpanMap<D, B, K, M, G::CP, SVW>;
  //: THE ROW TABLE, refined at every level EITHER side carves, so each stream reads
  //: its OWN range's rows on either side and a side wanting a coarser range ORs the
  //: finer rows it already holds.
  using CR = CarveRows<SVW, SVR>;
  constexpr int kWarps = G::kWarps;
  constexpr int kThreads = G::kThreads;

  constexpr int BC = BP::kBC;
  constexpr int LdV = G::LdV, LdS = G::LdS, LdR = G::LdR, LdO = G::LdO;
  constexpr int CP = G::CP;
  constexpr int MTOK = CP / kMmaM;  // token m-tiles of a strip
  //: THE WRITE STREAM'S leaves (`kLeafW`, the segment row) and THE EXECUTOR'S OWN
  //: COLUMN HALF of them (`kLeafE`), which is what the accumulator holds: `NL` n-tiles
  //: and `KDw` readout k-steps.  `NLW` is the stream's whole n-tile count, which the
  //: fetcher's expansion and the liveness mask are stated in.
  constexpr int kLeafW = G::kLeafW, kLeafE = G::kLeafE;
  constexpr int NL = kLeafE / kMmaN;
  constexpr int NLW = kLeafW / kMmaN;
  constexpr int KDw = kLeafE / kMmaK;
  constexpr int MCH = kDv / kMmaM;  // the channel m-tiles, ALL of them per warp
  static_assert(C % kMmaM == 0 && kThreads % C == 0, "the batch tile must divide");
  static_assert(kLeafE % kMmaK == 0, "an executor's leaves are a whole number of k-tiles");
  static_assert(BC % kAtomLeaves == 0, "K2: an owner holds whole page atoms");
  static_assert(design::smem_fits(G::bytes), "this arm does not fit the SMEM maximum");
  static_assert(LdS % 8 == 0 && LdR % 8 == 0 && LdV % 8 == 0,
                "an ldmatrix row address must be 16-byte aligned");
  static_assert(kWarps <= kBallotLanes, "one lane of the walk slices each executor's share");
  static_assert(SVW::kStreams * WPS == SVR::kStreams,
                "the executor map: `WPS` warps per WRITE stream, one READ stream each");
  static_assert(W % kBallotLanes == 0 && kThreads % kBallotLanes == 0,
                "a warp's tokens must be exactly one ballot word of the owner's row");

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int owner = (int)blockIdx.x;
  const int bh = (int)blockIdx.y;

  extern __shared__ __align__(16) char carry_smem[];
  //: ONE SLOT PER STREAM, and the SECOND STAGE IS REGISTERS: `ldmatrix` drains the
  //: slot, the `__syncwarp` after it IS the drained fact, and the slot is refilled for
  //: the next batch while this one's MMAs run out of fragments (F4, form (b)).
  constexpr int STAGES = G::kStages;
  static_assert(STAGES == 1, "the cooperative fetch carries one slot per stream");
  constexpr int kStreams = G::kStreams;
  constexpr int kSegRows = G::kSegRows;
  //: THE FOLD'S TENANTS COME FIRST AND ADJACENT -- the per-stream segments, the
  //: rolling coefficient staging and the page buffer -- because the retained plane
  //: straddles exactly those, and they are the set that is dead for the whole readout.
  __nv_bfloat16* const segV = reinterpret_cast<__nv_bfloat16*>(carry_smem + G::off_seg_v);
  __nv_bfloat16* const segC = reinterpret_cast<__nv_bfloat16*>(carry_smem + G::off_seg_c);
  __nv_bfloat16* const wstg = reinterpret_cast<__nv_bfloat16*>(carry_smem + G::off_stg);
  __nv_bfloat16* const gstg = reinterpret_cast<__nv_bfloat16*>(carry_smem + G::off_gain);
  float* const so = reinterpret_cast<float*>(carry_smem + G::off_out);
  __nv_bfloat16* const sr = reinterpret_cast<__nv_bfloat16*>(carry_smem + G::off_read);
  constexpr int kListStride = G::kListStride;
  uint16_t* const slist = reinterpret_cast<uint16_t*>(carry_smem + G::off_list);
  const SmemAddr memB = smem_addr(carry_smem + G::off_mem);
  const SmemAddr pfxB = smem_addr(carry_smem + G::off_pfx);
  const SmemAddr mskB = smem_addr(carry_smem + G::off_mask);
  const SmemAddr segVB = smem_addr(segV), segCB = smem_addr(segC);
  const SmemAddr stgB = smem_addr(wstg), gnB = smem_addr(gstg);
  //: THE RETAINED PLANE (2.2), straddled over the FOLD's tenants.  It is published
  //: once and stays valid until the fold reclaims it, which is what the readout's pull
  //: needs from it and the writers reload out of it with no republish leg.
  const SmemAddr hiB = smem_addr(carry_smem);

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
  //: THE EXECUTOR MAP, read off the rotated executor index: `strm` IS this warp's read
  //: stream (one per warp by declaration), `wstrm` its write stream and `col` its
  //: COLUMN HALF of that stream's leaves.  Both are shifts and masks of compile-time
  //: powers of two, and at `WPS == 1` the mask is zero, so `col` folds to the constant
  //: and every column term below disappears.
  constexpr int kWpsBits = ilog2c(WPS);
  const int wstrm = strm >> kWpsBits;
  const int col = strm & (WPS - 1);
  const int colBase = col * kLeafE;  // the warp's first private slot of the stream's box

  // ---- the owner's digit runs, one per level -------------------------------
  //: `o_l` is a bit-field extract of the owner index: the lattice's owner
  //: coordinates are the top `sum_l log2 g_l` bits of `lambda` (R0 spec section 2.4).
  int runCol[D];
  //: AND THE STREAM'S OWN DIGIT OFFSET INSIDE THAT RUN, per level and per side: the
  //: stream index decomposes in MIXED RADIX over the per-level stream grids, so a
  //: multi-level carve is `grp_l * S_l` at EVERY carved level and nothing changes at
  //: the levels the side spans whole.  Computed ONCE per warp, before the window
  //: loop, out of shifts and masks over compile-time powers of two.
  int offR[D], offW[D];
  //: AND THE BALLOT ROW BASE, per level: the SAME `o_l`, addressing the FIRST row the
  //: facts pass voted for this owner's run at this level (K31-i-d) -- at the table's
  //: grain, which is the finer of the two sides' spans, so a side reads its own
  //: range's rows and the owner's word is the OR of the run's.  The liveness pass
  //: reads these words instead of re-reading the amplitude runs itself.
  uint32_t rowOff[D];
  const uint32_t rowBytes = (uint32_t)tok_words * (uint32_t)sizeof(uint32_t);
  static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    const int o_l = (owner / BP::owner_div(l)) % BP::g(l);
    runCol[l] = BP::col_base(l) + o_l * BP::s(l);
    offR[l] = ((strm >> ilog2c(SVR::grid_div(l))) & (SVR::g(l) - 1)) * SVR::s(l);
    offW[l] = ((wstrm >> ilog2c(SVW::grid_div(l))) & (SVW::g(l) - 1)) * SVW::s(l);
    rowOff[l] = (uint32_t)(CR::row_prefix(l) + o_l * CR::rows_of(BP::s(l), l)) * rowBytes;
  });
  //: ONE 64-BIT BASE PER SIDE, and 32-bit byte displacements after it: the side's
  //: whole row block is `kRows * tok_words * 4` bytes, which is 32-bit at every
  //: admitted length.
  const uint32_t* const rowsR =
      pin_address(rows + (long)(2 * bh + 0) * (long)CR::kRows * (long)tok_words);
  const uint32_t* const rowsW =
      pin_address(rows + (long)(2 * bh + 1) * (long)CR::kRows * (long)tok_words);

  // ---- lane-invariant operand addressing (the shipped idiom) ---------------
  const int lr = lane & 7, tl = lane >> 3;
  const int rF = (tl & 1) * 8 + lr, cF = tl >> 1;
  const int rT = (tl >> 1) * 8 + lr, cT = tl & 1;
  //: THE PANEL RING'S TWO BASES, AS NAMED SCALARS.  A two-element array indexed by
  //: the batch parity would be a runtime index into a register array, which is a
  //: stack frame; a select is one instruction and no memory.
  //: each phase's two strips live in its OWN half: the read phase alternates inside
  //: `sr`, the fold inside `sw`, so neither touches the other's bytes.
  __nv_bfloat16* const panR0 = sr + warp * CP * LdR;
  __nv_bfloat16* const panR1 = sr + (kWarps + warp) * CP * LdR;
  const uint32_t brT0 = smem_addr(panR0) + (uint32_t)((rT * LdR + cT * 8) * 2);
  const uint32_t brT1 = smem_addr(panR1) + (uint32_t)((rT * LdR + cT * 8) * 2);
  //: THE CONSUMER'S TWO BASES INTO ITS STREAM'S SEGMENT.  The value sub-tile is pulled
  //: transposed (`k` = token, `m` = channel) and the coefficient sub-tile likewise
  //: (`k` = token, `n` = slot), so both are `ldmatrix.trans` at the segment's own row
  //: stride.  Both warps of a write stream pull the SAME value rows; the coefficient
  //: base carries the warp's COLUMN HALF, so the two read disjoint slot runs of one row.
  const uint32_t myV = segVB + (uint32_t)(((wstrm * kSegRows + rT) * LdV + cT * 8) * 2);
  const uint32_t myC = segCB + (uint32_t)(((wstrm * kSegRows + rF) * LdS + colBase + cF * 8) * 2);
#define ADDR_S(b_, R_, g_) ((b_) + (uint32_t)(((R_) * LdS + (g_) * 8) * 2))
#define ADDR_R(b_, R_, g_) ((b_) + (uint32_t)(((R_) * LdR + (g_) * 8) * 2))

  //: the warp's own staging region of the readout block.

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

  //: ---- K35-E: THE TRANSIT PLANE ------------------------------------------
  //: `leaf_of` is the LATTICE leaf of a private index `P`.  The private order's own
  //: `shift`/`span` give each level's digit inside `P`; the lattice's `run_shift`
  //: puts it back.  Every carved level additionally carries the stream's own digit
  //: offset.  This is the ONLY place the two orders meet, and it is a pure index map.
  auto leaf_of = [&](int P) {
    int lf = 0;
    static_for<D>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      const int dig = ((P >> PM::shift(l)) & (PM::span(l) - 1)) + offR[l];
      lf += dig << BP::run_shift(l);
    });
    return lf;
  };
  //: the plane is `[channel][leaf]` -- A row-major for the readout -- at stride
  //: `kHiLD` with the derived XOR swizzle on the two-leaf WORD index.  No linear pad
  //: is conflict-free on both the push and the pull; this is (findings, derivation 2).
  //: the same map under the WRITE carve, which is the order the accumulator -- and
  //: therefore the state plane and the transit's publish -- is held in.
  //: THE COLUMN HALF IS FOLDED IN HERE AND NOWHERE ELSE: `P` is the EXECUTOR's private
  //: slot in `[0, kLeafE)` and `colBase` lifts it to the STREAM's private slot in
  //: `[0, kLeafW)`, so `page_regs`, the publish, the reclaim and the fold's own
  //: accumulator all address the right leaves with no extra nest and no per-slot test.
  auto leafW_of = [&](int P) {
    int lf = 0;
    const int Ps = P + colBase;
    static_for<D>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      const int dig = ((Ps >> PMW::shift(l)) & (PMW::span(l) - 1)) + offW[l];
      lf += dig << BP::run_shift(l);
    });
    return lf;
  };
  auto plane_w = [&](int chan, int lf) {
    return hiB + (uint32_t)(chan * G::kHiLD * 2 + (((lf >> 1) ^ (chan & G::kHiSwz)) * 4));
  };
  //: THE PUBLISH.  The accumulator's `(row, col) = (channel, leaf)` map is the same
  //: one the A fragment wants, so a `bf16x2` word is one lane's two ADJACENT leaves
  //: at one channel -- the pair is lattice-adjacent under both carves' orders.
  //: The mass moves as ONE row: its accumulator's rows 1..15 are identically zero by
  //: construction of `ones_a`, so moving them would move zeros.
  auto publish_hi = [&]() {
    static_for<NL>([&](auto Jc) {
      constexpr int j = decltype(Jc)::value;
      const int lf = leafW_of(j * kMmaN + 2 * (lane & 3));
      static_for<MCH>([&](auto Mc) {
        constexpr int m = decltype(Mc)::value;
        const int ch = m * kMmaM + (lane >> 2);
        store_shared_u32(plane_w(ch, lf), hi_bf16x2(st[m][j][0], st[m][j][1]));
        store_shared_u32(plane_w(ch + 8, lf), hi_bf16x2(st[m][j][2], st[m][j][3]));
      });
      if ((lane >> 2) == 0)
        store_shared_u32(plane_w(kDv, lf), hi_bf16x2(mass_st[j][0], mass_st[j][1]));
    });
  };
  //: THE RECLAIM LEG of the retained-plane cycle: the inverse of `publish_hi`, at the
  //: same addresses, by the same warp.  Nothing else wrote them, so this needs no
  //: fence of its own -- the plane's single fence was the readers' wait.
  auto reload_hi = [&](uint32_t(&lo)[MCH][NL][2], uint32_t(&mlo)[NL][2]) {
    static_for<NL>([&](auto Jc) {
      constexpr int j = decltype(Jc)::value;
      const int lf = leafW_of(j * kMmaN + 2 * (lane & 3));
      static_for<MCH>([&](auto Mc) {
        constexpr int m = decltype(Mc)::value;
        const int ch = m * kMmaM + (lane >> 2);
        const uint32_t h0 = load_shared_u32(plane_w(ch, lf));
        const uint32_t h1 = load_shared_u32(plane_w(ch + 8, lf));
        st[m][j][0] = join_hi_lo(h0, lo[m][j][0], 0);
        st[m][j][1] = join_hi_lo(h0, lo[m][j][0], 1);
        st[m][j][2] = join_hi_lo(h1, lo[m][j][1], 0);
        st[m][j][3] = join_hi_lo(h1, lo[m][j][1], 1);
      });
      //: the mass rides row 0, so only the lane quad that published it reads back;
      //: every other row is identically zero and rejoins as zero.
      const uint32_t mh = ((lane >> 2) == 0) ? load_shared_u32(plane_w(kDv, lf)) : 0u;
      mass_st[j][0] = join_hi_lo(mh, mlo[j][0], 0);
      mass_st[j][1] = join_hi_lo(mh, mlo[j][0], 1);
      mass_st[j][2] = join_hi_lo(0u, mlo[j][1], 0);
      mass_st[j][3] = join_hi_lo(0u, mlo[j][1], 1);
    });
  };

  //: THE PULL, straight into the readout's A fragments.  `h` selects the n-tile, and
  //: within it register `2h` is channel row `lane/4` and `2h+1` is row `lane/4 + 8`
  //: -- the register order the readout's drain wants.
  auto pull_operand = [&](auto Kc, uint32_t(&sa)[MCH][4], uint32_t(&ma)[4]) {
    constexpr int kk = decltype(Kc)::value;
    static_for<2>([&](auto Hc) {
      constexpr int h = decltype(Hc)::value;
      const int lf = leaf_of((2 * kk + h) * kMmaN + 2 * (lane & 3));
      static_for<MCH>([&](auto Mc) {
        constexpr int m = decltype(Mc)::value;
        const int ch = m * kMmaM + (lane >> 2);
        sa[m][2 * h + 0] = load_shared_u32(plane_w(ch, lf));
        sa[m][2 * h + 1] = load_shared_u32(plane_w(ch + 8, lf));
      });
      ma[2 * h + 0] = ((lane >> 2) == 0) ? load_shared_u32(plane_w(kDv, lf)) : 0u;
      ma[2 * h + 1] = 0u;
    });
  };

  //: THE ONES OPERAND, AN IMMEDIATE: the mass channel's value is literally 1, so
  //: the fold's A fragment for it is the constant whose row 0 is ones.
  constexpr uint32_t kOnesPair = 0x3F803F80u;
  uint32_t ones_a[4];
  ones_a[0] = (lane >> 2) == 0 ? kOnesPair : 0u;
  ones_a[1] = 0u;
  ones_a[2] = ones_a[0];
  ones_a[3] = 0u;

  // ---- the state plane's page sweep, in LATTICE leaf order -----------------
  constexpr int kAtoms = BC / kAtomLeaves;
  constexpr int kPageElems = kAtomLeaves * kCols;
  //: THE SWEEP'S STEP, AS A CARRY.  A step of `kThreads` elements advances the row
  //: cursor by `kRowStep` or by one more, so the walk is a compile-time count of
  //: predicated subtractions and never a division -- at any executor count.
  constexpr int kRowStep = kThreads / kCols;
  constexpr int kPageCarry = kThreads - kRowStep * kCols;
  static_assert(kRowStep >= 1, "a sweep step must advance the row cursor");
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
  //: `lambda = owner*BC + local` (LATTICE-1), so the owner's atoms are
  //: `owner*(BC/16) + i`.  What the shipped kernel proves with `base_aligned` is a
  //: DEFINITION here (R0 spec section 2.6).
  const uint32_t atomBase = (uint32_t)(bh * (int)(BP::kLeaves / kAtomLeaves) + owner * kAtoms);
  auto page_off = [](uint32_t atom) { return atom * (uint32_t)(kPageElems * (int)sizeof(float)); };
  //: THE PAGE'S SLICE OF A STREAM, PER FRAGMENT.  Register-resident state exists only
  //: as fragments (KERNEL_STANDARDS section 1): the accumulator is `NL` n-tiles x
  //: `MCH` m-tiles of the m16n8 shape, and a lane's fragment holds TWO ADJACENT
  //: private slots at two channel rows.  So this nest iterates FRAGMENT MODES --
  //: n-tile, the pair's half, m-tile, the row half -- and the leaf map is folded into
  //: the ADDRESS: the private slot is a lane expression, its atom and its row inside
  //: the page come from the map, and every accumulator index is a constant.
  //: BODY COUNT: `NL * 2 * (2*MCH + 1)` per atom = 72 at every built arm, times
  //: `kAtoms` per call (8 at BC=128, 16 at BC=256).  The per-LEAF form this replaces
  //: was `kLeafW * (2*MCH + 1)` per atom with three quarters of the lanes predicated
  //: off -- 4,096 bodies at the eight-warp arm, and 50 minutes of cicc.
  auto page_regs = [&](auto Ac, auto Ld) {
    constexpr int a = decltype(Ac)::value;
    constexpr bool LOAD = decltype(Ld)::value;
    //: THE STATE PLANE IS WALKED UNDER THE WRITE CARVE, because that is the order the
    //: accumulator is held in.
    static_for<NL>([&](auto Jc) {
      constexpr int j = decltype(Jc)::value;
      static_for<2>([&](auto Lc) {
        constexpr int el = decltype(Lc)::value;
        const int lf = leafW_of(j * kMmaN + 2 * (lane & 3) + el);
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
#pragma unroll
      static_for<kAtoms>([&](auto Ac) {
        page_sweep(page_off(atomBase + (uint32_t)decltype(Ac)::value), [&](int sidx, uint32_t off) {
          so[sidx] = *reinterpret_cast<const float*>(gather_ptr(plane, off));
        });
        rendezvous();
        page_regs(Ac, std::true_type{});
        rendezvous();
      });
    }
  }

  // ---- the produce's per-thread map ---------------------------------------
  constexpr int SPT = PM::SPT;
  const int prow = lane / PM::kRowThr;  // this lane's token row of the strip
  const int psub = lane % PM::kRowThr;  // which half of that row's slots it mints
  int lvCol[D];
  static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    lvCol[l] = runCol[l] + offR[l];
  });

  //: A DEAD ROW IS ZEROED AT THE PAIR LEVEL'S FACTOR.  A padding row of a partial
  //: batch has no token, and a zero coefficient contributes exactly nothing to the
  //: MMA (the produce's dead-row zeroing law).  The pair level's run is the factor
  //: every slot of the row carries, so zeroing it covers the whole row for one or two
  //: selects and costs no arithmetic.
  //:
  //: THIS IS THE READ SIDE'S PRODUCE AND ONLY THE READ SIDE'S (K35 S2).  The write
  //: side's expansion moved to the FETCHER (`expand` below), which mints each stream's
  //: row once into that stream's segment; what stays here is the read panel's, until
  //: the readout takes the same treatment.  The read side HAS NO GAIN: its scalar is
  //: identically 1 and the trailing `mul_bf16x2(acc, 1)` was 44% of the read produce.
  auto produce = [&](const __nv_bfloat16* base, __nv_bfloat16* dst, uint32_t tokOff, bool have) {
    //: THE SLOT ACCUMULATORS ARE THE ONLY ARRAY THE PRODUCE CARRIES.  A level's run
    //: is loaded, folded into every slot and dropped inside one `static_for` body, so
    //: no `[D][n]` array is ever live.
    uint32_t acc[SPT / 2];
    {
      //: THE LANE'S SLOT BASE IS `psub * SPT`, and each level reads its digit out of
      //: `p` at that level's own shift -- so the base contributes a CONSTANT digit
      //: offset per level, which `idx_of` (indexing the LOCAL slot) cannot see.  Fold
      //: it into the run's column.
      constexpr int PL = PM::kPairLevel;
      const int adjPL = ((psub * SPT) >> PM::shift(PL)) & (PM::span(PL) - 1);
      unsigned short run[PM::n_l(PL)];
      load_run<PM::n_l(PL)>(run, gather_ptr(base, tokOff + (uint32_t)((lvCol[PL] + adjPL) * 2)));
      static_for<PM::n_l(PL)>([&](auto Ic) {
        constexpr int i = decltype(Ic)::value;
        run[i] = have ? run[i] : (unsigned short)0;
      });
      static_for<SPT / 2>([&](auto Jc) {
        constexpr int j = 2 * decltype(Jc)::value;
        acc[j / 2] =
            ((uint32_t)run[PM::idx_of(PL, j + 1)] << 16) | (uint32_t)run[PM::idx_of(PL, j)];
      });
    }
    static_for<D>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      if constexpr (l != PM::kPairLevel) {
        const int adj = ((psub * SPT) >> PM::shift(l)) & (PM::span(l) - 1);
        unsigned short run[PM::n_l(l)];
        load_run<PM::n_l(l)>(run, gather_ptr(base, tokOff + (uint32_t)((lvCol[l] + adj) * 2)));
        static_for<SPT / 2>([&](auto Jc) {
          constexpr int j = 2 * decltype(Jc)::value;
          acc[j / 2] = mul_bf16x2(acc[j / 2], splat_bf16x2_bits(run[PM::idx_of(l, j)]));
        });
      }
    });
    __nv_bfloat16* const row = dst + prow * LdR + psub * SPT;
    static_for<SPT / 2>([&](auto Jc) {
      constexpr int j = 2 * decltype(Jc)::value;
      store_bf16x2(row + j, acc[j / 2]);
    });
  };
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
    //: zeroes the leading factor, and a zero coefficient contributes exactly nothing
    //: to either MMA (the produce's dead-row zeroing law).
    auto produce_batch = [&](const __nv_bfloat16* base, __nv_bfloat16* dst, const uint16_t* list,
                             int o0, int ntot) {
      const int nb = (ntot - o0) < C ? (ntot - o0) : C;
      const bool have = prow < nb;
      const int tok = list[o0 + (have ? prow : nb - 1)];
      produce(base, dst, winP + (uint32_t)tok * (uint32_t)(BP::kWtot * 2), have);
    };

    // ---- the liveness pass: one READ list per warp, ONE write list --------
    //: THE SCHEDULE IS TWO-SIDED (Blake ruling 2026-08-19).  Per level at most ONE
    //: side is sparse, so there is one stored row set per level -- but the two
    //: sides' CLAUSE SETS are different level subsets, and a dense side's clause is
    //: absent-means-full.  ANDing each side's OWN rows IS that conjunction over
    //: exactly the levels where that side is sparse, with no structure fact carried
    //: into the kernel.  At a level a side CARVES, that side's clause is its own
    //: STREAM's range -- the OR of the table rows the range covers -- and the owner's
    //: run is the OR over the whole run, which the shared derivations want.
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

    //: The scatter is PER-LANE and divergent by choice, and its store is ADDRESSED
    //: rather than indexed; both are measured, and the word-major alternative is
    //: refused with numbers -- see docs/internals/carry/carry_kernel.md, "The
    //: schedule scatter".  `bits` is the writer's slice of each word and `below` the
    //: bits under it: one list with several writers, each deriving the same prefix.
    auto walk = [&](uint32_t m, uint16_t* dst, uint32_t bits, uint32_t below) {
      const int cnt = bit_count(m);
      const int pfx = lane_prefix_incl<LW>(cnt);
      const int k = (pfx - cnt) + bit_count(m & below);
      const uint16_t tb = (uint16_t)(lane * kBallotLanes);
      SmemAddr a = smem_addr(dst) + (SmemAddr)(k * (int)sizeof(uint16_t));
      uint32_t mm = m & bits;
      while (mm) {
        store_shared_u16(a, (uint16_t)(tb + (uint16_t)first_set(mm)));
        a += (SmemAddr)sizeof(uint16_t);
        mm &= mm - 1;
      }
      return lane_broadcast(pfx, LW - 1);
    };
    //: THE CONJUNCTION, ONE STATEMENT FOR EVERY LEVEL -- and EXACT at any carve,
    //: because a rectangle of digits holds a live leaf iff every level's range holds
    //: a live digit.  A digit RANGE's word is the OR of the table rows it covers
    //: (`RANGE`, in table rows), and the levels are ANDed.  Two ranges are asked for:
    //: the STREAM's own (`MINE`, its ownership test) and the OWNER's whole run (the
    //: shared derivation and the whole-window skip).  It is PRUNED BY THE
    //: DECLARATION: a side conjoins a level only where that side is declared sparse,
    //: because a dense side's row is all-ones and ANDing it is a load for nothing.
    auto ballot_of = [&](const uint32_t* rowsS, auto Side, auto Vec, auto Mine, const int* off) {
      constexpr uint32_t SIDE = decltype(Side)::value;
      using SV = typename decltype(Vec)::type;
      constexpr bool MINE = decltype(Mine)::value;
      uint32_t m = valid;
      static_for<D>([&](auto Lc) {
        constexpr int l = decltype(Lc)::value;
        if constexpr (CR::row_g(l) > 0 && mode_declares(MO, l, SIDE)) {
          constexpr int RANGE = CR::rows_of(MINE ? SV::s(l) : BP::s(l), l);
          const uint32_t base =
              rowOff[l] + (MINE ? (uint32_t)(off[l] >> ilog2c(CR::s(l))) * rowBytes : 0u);
          uint32_t r = 0u;
          static_for<RANGE>([&](auto Ic) {
            r |= *reinterpret_cast<const uint32_t*>(
                gather_ptr(rowsS, base + (uint32_t)decltype(Ic)::value * rowBytes + wordB));
          });
          m &= r;
        }
      });
      return m;
    };
    constexpr std::integral_constant<uint32_t, kSideRead> kRead{};
    constexpr std::integral_constant<uint32_t, kSideWrite> kWrite{};
    //: the OWNER-GRAIN ballots, which the cooperative walks are derived from, and the
    //: STREAM-GRAIN ballots, which each executor's pull tests its ids against.
    const uint32_t wideR = ballot_of(rowsR, kRead, type_tag<SVR>{}, std::false_type{}, offR);
    const uint32_t wideW = ballot_of(rowsW, kWrite, type_tag<SVW>{}, std::false_type{}, offW);
    const uint32_t mineR = ballot_of(rowsR, kRead, type_tag<SVR>{}, std::true_type{}, offR);
    const uint32_t mineW = ballot_of(rowsW, kWrite, type_tag<SVW>{}, std::true_type{}, offW);
    //: THE WHOLE-WINDOW SKIP, at (owner, window) grain, per side -- the block-bitmap
    //: mechanism's harvest (study section 2.1).  It is CTA-uniform WITHOUT a
    //: barrier: the two owner-grain ballots are the same words in every warp.
    if (lane_ballot((wideR | wideW) != 0u) == 0u) continue;

    //: THE READ SIDE GETS THE SAME SHAPE AS THE WRITE SIDE.  Every stream walking the
    //: whole bitmap under its own ballot is one pass per stream over one window;
    //: instead ONE cooperative walk derives the OWNER's read ids and each stream PULLS
    //: its own subsequence.  It is a per-cell TRADE, not a win -- where a token reads
    //: leaves in several boxes the owner's union is nearly the sum of the per-stream
    //: lists, so the shared walk saves nothing and the pull is added work.
    constexpr int kSliceR = kBallotLanes / kWarps;
    uint16_t* const rlistS = slist + (kWarps + 1) * kListStride;
    const int nrO = walk(wideR, rlistS, (uint32_t)(((1ull << kSliceR) - 1ull) << (warp * kSliceR)),
                         (uint32_t)((1ull << (warp * kSliceR)) - 1ull));
    rendezvous();
    uint16_t* const rlist = slist + warp * kListStride;
    int nr = 0;
    {
      const uint32_t belowR = (1u << lane) - 1u;
      const SmemAddr rbase = smem_addr(rlist);
#pragma unroll 1
      for (int s0 = 0; s0 < nrO; s0 += kBallotLanes) {
        const int idx = s0 + lane;
        const bool inR = idx < nrO;
        const int t = inR ? (int)rlistS[idx] : 0;
        const uint32_t w = (uint32_t)lane_broadcast((int)mineR, t >> 5);
        const bool has = inR && (((w >> (t & 31)) & 1u) != 0u);
        const uint32_t bal = lane_ballot(has);
        if (has)
          store_shared_u16(rbase + (SmemAddr)((nr + bit_count(bal & belowR)) * 2), (uint16_t)t);
        nr += bit_count(bal);
      }
    }
    __syncwarp();
    //: THE OWNER'S WRITE IDS, DERIVED ONCE FOR THE CTA.  `walk` is already the cheap
    //: form -- one lane per ballot word, `ffs` per live bit -- and its `bits`/`below`
    //: parameters exist for exactly this: several writers deriving the SAME global
    //: prefix and filling disjoint slices of every word, so the list comes out in the
    //: one ascending order every consumer expects.  The CTA wall is here and nowhere
    //: else in the fold.
    constexpr int kSliceBits = kBallotLanes / kWarps;
    uint16_t* const wlistS = slist + kWarps * kListStride;
    const int nw =
        walk(wideW, wlistS, (uint32_t)(((1ull << kSliceBits) - 1ull) << (warp * kSliceBits)),
             (uint32_t)((1ull << (warp * kSliceBits)) - 1ull));
    rendezvous();
    //: THE WRITE SIDE'S SCHEDULE IS THE SHARED UNION LIST built above, and it is the
    //: ONLY write-side list there is: under the cooperative fetch a warp does not walk
    //: its own tokens, it takes a fixed share of the UNION's and places rows into the
    //: segments of whichever streams want them.  So each executor PUBLISHES its own
    //: membership -- one ballot word per lane -- together with the per-word PREFIX
    //: POPCOUNT, and those two turn a (token, stream) pair into the RANK the placement
    //: addresses.  `mineW` is already `valid`-masked, so a lane past the window's last
    //: word contributes a zero count and the prefix is exact at any `LW`.
    //: PUBLISHED ONCE PER WRITE STREAM, by its HALF-0 warp: the `WPS` warps of a stream
    //: hold the SAME `mineW` (they differ only in which leaf columns they fold), so the
    //: other halves would restate the same words.
    if (inW && col == 0) store_shared_u32(memB + (uint32_t)((wstrm * LW + lane) * 4), mineW);
    {
      const int cnt = bit_count(mineW);
      const int pfx = lane_prefix_incl<LW>(cnt) - cnt;
      if (inW && col == 0)
        store_shared_u16(pfxB + (uint32_t)((wstrm * LW + lane) * 2), (uint16_t)pfx);
    }

    //: THE PHASE CHANGE.  Every warp publishes its own write-carve hi ONCE; the one
    //: fence is the readers' wait.  `lo` stays in registers -- packed, so the state
    //: costs half its fp32 footprint across the readout -- and the plane stays valid
    //: underneath, so the fold reloads its OWN hi from it with no republish.
    publish_hi();
    uint32_t lo[MCH][NL][2], mlo[NL][2];
    static_for<MCH>([&](auto Mc) {
      constexpr int m = decltype(Mc)::value;
      static_for<NL>([&](auto Jc) {
        constexpr int j = decltype(Jc)::value;
        lo[m][j][0] = pack_lo(st[m][j][0], st[m][j][1]);
        lo[m][j][1] = pack_lo(st[m][j][2], st[m][j][3]);
      });
    });
    static_for<NL>([&](auto Jc) {
      constexpr int j = decltype(Jc)::value;
      mlo[j][0] = pack_lo(mass_st[j][0], mass_st[j][1]);
      mlo[j][1] = pack_lo(mass_st[j][2], mass_st[j][3]);
    });
    rendezvous();
    uint32_t sa[KDw][MCH][4], ma[KDw][4];
    static_for<KDw>(
        [&](auto Kc) { pull_operand(Kc, sa[decltype(Kc)::value], ma[decltype(Kc)::value]); });

    // ---- PHASE 1: the READOUT of every read batch, against the window's ENTRY
    //      state.  ALL of the window's readouts precede ANY of its folds, because a
    //      within-window pair is the INTRA term's and this kernel must not carry it.
    //      The batch line is therefore the window line on both sides.
    //: THE READ PHASE SHARES NOTHING: the list, the panel region, the staging block
    //: and the fan-in are the stream's own, so the warps run to the window line
    //: without a single CTA barrier.  Its `__syncwarp` is the panel's, between a
    //: lane's `STS` and the `ldmatrix` that reads another lane's row.
    const int nbr = (nr + CP - 1) / CP;
    //: THE PRODUCE RUNS ONE BATCH AHEAD, into the other panel of the two-slot ring,
    //: UNCONDITIONALLY and AFTER the batch's MMA block.  Unconditionally because a
    //: guard is a basic block and ptxas schedules within one; after, because the
    //: produce ends in panel `STS` and the MMA begins in panel `LDSM`, and the two
    //: shared ranges are not provably disjoint -- in the other order that aliasing
    //: edge pins every produce instruction ahead of the HMMAs.  The trailing batch is
    //: entirely dead rows by the produce's zeroing law and lands in a panel no MMA
    //: reads.
    if (nbr > 0) produce_batch(pR, panR0, rlist, 0, nr);
    __syncwarp();
#pragma unroll 1
    for (int i = 0; i < nbr; ++i) {
      const int off = i * CP;
      const int n = (nr - off) < CP ? (nr - off) : CP;
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
        uint32_t rb[KDw][4];
#pragma unroll
        for (int kk = 0; kk < KDw; ++kk) load_frag(rb[kk], ADDR_R(brT, mb * 16, 2 * kk));
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
        //: entry's digits reach two groups, so no meeting is instantiated.
        //:
        //: THE PERMUTED OUTPUT ABI (2.6): the RED issues STRAIGHT FROM THE
        //: ACCUMULATOR, in the fragment's own order, and the staging pass is gone.
        //: The accumulator element `(m, e, lane)` carries
        //:     token   = j*kMmaN + 2*(lane&3) + (e&1)
        //:     channel = m*kMmaM + (lane>>2) + 8*(e>>1)
        //: so one issue covers FOUR tokens x EIGHT consecutive channels -- four 32-B
        //: segments, against the staged form's one 128-B segment.  Both move 2,048 B
        //: per token tile in 64 sectors from 16 issues; the sector count and the issue
        //: count are IDENTICAL, and what the staging bought was only the shape of the
        //: segments.  Deleting it removes 32 `STS` and two `__syncwarp`s per tile and
        //: gives the fan-in buffer's bytes back.
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          const int j = 2 * mb + p;
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int idx = j * kMmaN + 2 * (lane & 3) + (e & 1);
            if (idx < n) {
              const uint32_t rowO = (uint32_t)rlist[off + idx] * (uint32_t)(kDv * 4) + winO;
#pragma unroll
              for (int m = 0; m < MCH; ++m)
                fan_in_add(
                    reinterpret_cast<float*>(gather_ptr(
                        oPlane, rowO + (uint32_t)((m * kMmaM + (lane >> 2) + 8 * (e >> 1)) * 4))),
                    acc[p][m][e]);
              //: the mass rides row 0 of its own accumulator, so only the quad at
              //: `lane>>2 == 0` and the halves with `e>>1 == 0` carry it.
              if ((lane >> 2) == 0 && (e >> 1) == 0)
                fan_in_add(reinterpret_cast<float*>(
                               gather_ptr(denPlane, (uint32_t)rlist[off + idx] * 4u + winD)),
                           macc[p][e]);
            }
          }
        }
      }
      produce_batch(pR, ((i + 1) & 1) ? panR1 : panR0, rlist, (i + 1) * CP, nr);
      __syncwarp();
    }

    //: THE WINDOW LINE IS THE ONE BARRIER LEFT ON THE READ SIDE, and it is there by
    //: the ordering law: no warp may fold this window's writes while another is
    //: still reading the window's entry state.
    rendezvous();

    //: THE RECLAIM.  Each warp reads back its OWN slice -- same warp, same addresses
    //: as its publish -- and rejoins it with the `lo` it kept, exactly.  Every warp
    //: has finished reading the plane before the window edge below, after which the
    //: fold's strips may overwrite it.
    reload_hi(lo, mlo);
    rendezvous();

    // ---- PHASE 2: THE FOLD -- ONE UNION WALK, WRITER-SIDE PLACEMENT ---------
    //: F2/F4.  A BATCH is `kSegRows` consecutive tokens of the CTA's union list, and
    //: the CTA walks that list ONCE.  Executor `warp` takes the fixed share
    //: `i % kWarps` of the batch's union tokens -- no dynamic assignment -- and for
    //: each token it handles and each STREAM that wants it, it places into that
    //: stream's segment at `rank(t, s)`:
    //:   (a) the value row, by `cp.async`, and
    //:   (b) the FULLY EXPANDED coefficient row for that stream's box -- `kLeafW`
    //:       bf16 in the stream's private order, gain applied.
    //: The consumer does NO produce: tile skip -> `ldmatrix` -> HMMA.  The sub-products
    //: over the levels a stream does NOT carve are shared across the streams of one
    //: token and are minted once as `memo`; the memo is INTERNAL to the fetcher and its
    //: shape is the span vector's, at compile time.
    //:
    //: WHY THE RANK IS THE ADDRESS AND WHY THE BOUND IS STATIC.  The union list is
    //: ascending in token, so a stream's compacted list is a subsequence of it and
    //: `rank(t, s)` is the prefix popcount of that stream's ballot below `t`.  Inside a
    //: batch of `kSegRows` union tokens a stream can hold at most `kSegRows` of them,
    //: so the batch-local rank is below `kSegRows` BY CONSTRUCTION -- the segment needs
    //: no truncation path and the bound is the type's, not a runtime test.
    //:
    //: COVERAGE IS BY CONSTRUCTION, NOT BY ACCOUNTING (the salvage's law: `0 * NaN` is
    //: `NaN`, so a slot row the MMA reads must never be arbitrary SMEM).  THE DEAD-ROW
    //: LAW: a dead row is a FINITE (stale) value row with a ZERO coefficient row --
    //: finite operand, zero weight, exactly nothing contributed -- and coverage is the
    //: ONCE-PER-WINDOW clear plus the PER-BATCH dead-coefficient zeroing.  The window's
    //: first clear makes every row of the segment finite (the published plane straddles
    //: these very bytes, so the window inherits arbitrary bits and must not); thereafter
    //: a value row is either this batch's placement or a previous batch's, both finite,
    //: and the owning executor need only zero the coefficient rows no placement of THIS
    //: batch covers.  A full batch writes nothing.  No row's liveness is computed twice.
    constexpr int TPW = kSegRows / kWarps;  // union tokens one executor handles per batch
    static_assert(TPW * kWarps == kSegRows,
                  "the batch's union tokens split evenly over the executors");
    constexpr int kStrBits = ilog2c(kStreams);
    //: THE EXECUTOR'S CANDIDATE PAIRS: its `TPW` union tokens times the WRITE streams,
    //: one per lane.  At `WPS == 1` this is the whole batch (`kStreams == kWarps`);
    //: under a column split the executors are more numerous and each holds fewer.
    constexpr int kCand = TPW * kStreams;
    static_assert(kCand <= kBallotLanes, "one lane resolves each candidate pair");
    //: THE EXPANSION'S LANE MAP.  A coefficient row is `kWordsRow` packed pairs; the
    //: warp covers `kRowsPass` rows at once, one per lane group, and the rows of a pass
    //: are consecutive (token, stream) pairs -- SAME token, adjacent streams -- which
    //: is what lets one memo serve the whole pass.
    constexpr int kWordsRow = kLeafW / 2;
    constexpr int kLanesRow = kWordsRow < kBallotLanes ? kWordsRow : kBallotLanes;
    constexpr int kRowsPass = kBallotLanes / kLanesRow;
    constexpr int kWordsLane = kWordsRow / kLanesRow;
    constexpr int kStreamPass = kStreams / kRowsPass;
    static_assert(kStreams % kRowsPass == 0 && kStreamPass >= 1,
                  "a pass's rows are streams of ONE token, so the stream count carries them");
    static_assert(kLanesRow >= kMmaN, "an n-tile pair's liveness is a byte of the row's ballot");
    //: n-tile PAIRS -- one `ldmatrix` and one skip bit each -- of the WRITE STREAM's row
    //: (`kNJW`, which the expansion's ballot names) and of THIS EXECUTOR's column half
    //: (`kNJ`, which the fold issues).  The mask word carries the stream's bits; a
    //: warp's field of it starts at `col * kNJ`.
    constexpr int kNJW = NLW / 2;
    constexpr int kNJ = NL / 2;
    static_assert(kNJ * WPS == kNJW, "the column halves partition the stream's n-tile pairs");
    static_assert(kWordsLane * (kLanesRow / kMmaN) == kNJW,
                  "the mask carries exactly one bit per n-tile pair of the stream's row");
    static_assert(kLanesRow % WPS == 0,
                  "the `WPS` lanes that publish a row's mask are one per column half");
    //: the value row's copy map: `kChunkV` sixteen-byte chunks, `kValRows` rows a pass.
    constexpr int kChunkV = kDv / 8;
    constexpr int kValRows = kBallotLanes / kChunkV;
    constexpr int kValPass = kCand / kValRows;
    static_assert(kValPass * kValRows == kCand, "the candidate pairs split evenly");
    //: the staging's map: one thread per (union token, run chunk), the chunk group
    //: rounded to a power of two so the split is a shift and a mask.
    constexpr int kStgCh = ceil_pow2c(BP::kStageChunks);
    constexpr int kStgStep = kThreads / kStgCh;
    constexpr int kStgPass = (kSegRows + kStgStep - 1) / kStgStep;

    const int exH = lane >> ilog2c(kLanesRow);  // which row of the pass this lane serves
    const int exJ = lane & (kLanesRow - 1);     // its packed pair inside that row
    const int vlC = lane & (kChunkV - 1);       // its sixteen-byte chunk of a value row
    const int vlR = lane >> ilog2c(kChunkV);    // which row of the value pass
    //: THE MASK'S TWO PER-WARP CONSTANTS: the field of the stream's mask word this
    //: executor's column half owns, and the bit its own n-tile pair `nj` sits at.  Both
    //: are computed once per warp, so the column split costs the fold NOTHING per batch
    //: beyond the tests it already ran.
    const uint32_t mBit = 1u << (col * kNJ);
    const uint32_t mFld = (uint32_t)((((1u << kNJ) - 1u) << (col * kNJ)));
    //: and the lanes that PUBLISH a row's mask: one per column half, so the `WPS` words
    //: of a stream are written by the same single instruction the one word was.
    constexpr int kMskLanes = kLanesRow / WPS;
    const int exM = exJ >> ilog2c(kMskLanes);
    const bool exPub = (exJ & (kMskLanes - 1)) == 0;

    //: `rank_live(s, t)` -- the token's position in stream `s`'s compacted list, and
    //: whether the stream holds it at all, out of the published ballot and prefix.
    auto rank_live = [&](int s, int t) {
      const int w = t >> 5;
      const uint32_t word = load_shared_u32(memB + (uint32_t)((s * LW + w) * 4));
      const int pre = (int)load_shared_u16(pfxB + (uint32_t)((s * LW + w) * 2));
      const uint32_t below = (1u << (t & 31)) - 1u;
      return ((pre + bit_count(word & below)) << 1) | (int)((word >> (t & 31)) & 1u);
    };
    //: THE BATCH'S CANDIDATE PAIRS, RESOLVED ONCE.  Lane `q < kSegRows` owns candidate
    //: `q = (i, s)`, computes the placement and every other lane reads it by shuffle --
    //: so the four shared loads of a rank are paid once per pair and never per lane of
    //: the row that pair fills.  `-1` is "this stream does not want this token".
    auto batch_place = [&](int u0, int nb) {
      int e = -1;
      if (lane < kCand) {
        const int i = lane >> kStrBits, s = lane & (kStreams - 1);
        const int ii = i * kWarps + warp;
        if (ii < nb) {
          const int t = (int)wlistS[u0 + ii];
          const int rl = rank_live(s, t);
          if (rl & 1) {
            const int t0 = (int)wlistS[u0];
            e = (((rl >> 1) - (rank_live(s, t0) >> 1)) << 16) | t;
          }
        }
      }
      return e;
    };
    //: THE BATCH'S ROW COUNT FOR THE EXECUTOR'S OWN STREAM, out of the same published
    //: ballot the placement reads.  `rank_live` counts a stream's tokens BELOW a token,
    //: and the union list contains every stream's tokens in ascending order, so every
    //: token of stream `wstrm` between the batch's first and last union token IS in the
    //: batch: the count is the rank at the last token (plus its own liveness) less the
    //: rank at the first.  One lane pays the two lookups the placement already pays for
    //: its candidates; the warp reads it by shuffle.
    auto batch_rows = [&](int u0, int nb) {
      int n = 0;
      if (lane == 0 && nb > 0) {
        const int rl = rank_live(wstrm, (int)wlistS[u0 + nb - 1]);
        n = (rl >> 1) + (rl & 1) - (rank_live(wstrm, (int)wlistS[u0]) >> 1);
      }
      return lane_broadcast(n, 0);
    };
    //: THE VALUE LEG.  `cp.async` never touches registers, and the fetcher issues it
    //: into a segment it does not consume -- the fan-out pipeline's first owner.
    //: BODY COUNT: `kValPass` = `kCand * kChunkV / 32` -- 4 where the executors are the
    //: streams, 2 at the four-stream column-split arm -- each body one predicated issue.
    auto fetch_values = [&](int e0) {
      static_for<kValPass>([&](auto Pc) {
        constexpr int pp = decltype(Pc)::value;
        const int q = pp * kValRows + vlR;
        const int e = lane_broadcast(e0, q);
        if (e >= 0) {
          const int s = q & (kStreams - 1);
          stage_tile(segVB + (uint32_t)((((s * kSegRows + (e >> 16)) * LdV) + vlC * 8) * 2),
                     gather_ptr(vPlane, (uint32_t)(e & 0xFFFF) * (uint32_t)v_row_bytes + winV
                                            + (uint32_t)(vlC * 8 * 2)));
        }
      });
    };
    //: ONE AMPLITUDE OUT OF THE STAGED ROW.  The level lying LOWEST in the private
    //: order (level `D - 1`, by rule 1) supplies BOTH slots of a packed pair with one
    //: four-byte access, because its two digits are adjacent; every other level's digit
    //: is constant across the pair and is splatted.
    auto factor = [&](uint32_t rowB, auto Lc, int p0, int off) {
      constexpr int l = decltype(Lc)::value;
      const int d = ((p0 >> PMW::shift(l)) & (PMW::span(l) - 1)) + off;
      if constexpr (l == D - 1)
        return load_shared_u32(rowB + (uint32_t)((BP::stage_off(l) + d) * 2));
      else
        return splat_bf16x2_bits(load_shared_u16(rowB + (uint32_t)((BP::stage_off(l) + d) * 2)));
    };
    //: the stream's digit offset at a level, for a stream this executor does NOT run:
    //: the stream index decomposes in mixed radix over the per-level stream grids, so
    //: it is a shift and a mask over compile-time powers of two.
    auto offW_of = [&](int s, auto Lc) {
      constexpr int l = decltype(Lc)::value;
      return ((s >> ilog2c(SVW::grid_div(l))) & (SVW::g(l) - 1)) * SVW::s(l);
    };
    //: THE EXPANSION.  DECLARED ARITHMETIC ORDER, per packed slot pair:
    //:     w = gain, then x f_l for every level the side SPANS WHOLE in ascending `l`,
    //:                then x f_l for every level the side CARVES in ascending `l`,
    //: each product rounded to nearest even in bf16 (`mul_bf16x2`).  The prefix through
    //: the whole-spanned levels is the MEMO -- it does not depend on the stream, so it
    //: is minted once per (token, lane) and reused by every stream of that token.  This
    //: order is NOT the deleted produce's (which multiplied the pair level first and
    //: the gain last), so the fold's coefficients are not bit-identical to it; the
    //: total multiply count is unchanged and the oracle is the gate.
    //: BODY COUNT per call: `TPW * (1 + kWordsLane * U)` for the memo and
    //: `TPW * kStreamPass * kWordsLane * (Cv + 2)` for the rows, with `U`/`Cv` the
    //: whole-spanned and carved level counts -- 4*(1+1) + 4*2*1*3 = 32 at the D=2 ALT
    //: arms, 2*(1+3) + 2*4*1*4 = 40 at the D=4 multi-level carve.
    auto expand = [&](int slot, int e0) {
      const uint32_t blkB = stgB + (uint32_t)(slot * kSegRows * BP::kStageW * 2);
      const uint32_t gblkB = gnB + (uint32_t)(slot * kSegRows * 2);
      static_for<TPW>([&](auto Ic) {
        constexpr int ic = decltype(Ic)::value;
        const int ii = ic * kWarps + warp;
        const uint32_t rowB = blkB + (uint32_t)(ii * BP::kStageW * 2);
        const uint32_t gain = splat_bf16x2_bits(load_shared_u16(gblkB + (uint32_t)(ii * 2)));
        uint32_t memo[kWordsLane];
        static_for<kWordsLane>([&](auto Wc) {
          constexpr int wl = decltype(Wc)::value;
          const int p0 = 2 * (exJ + wl * kLanesRow);
          uint32_t a = gain;
          static_for<D>([&](auto Lc) {
            constexpr int l = decltype(Lc)::value;
            if constexpr (!SVW::carves(l)) a = mul_bf16x2(a, factor(rowB, Lc, p0, 0));
          });
          memo[wl] = a;
        });
        static_for<kStreamPass>([&](auto Pc) {
          constexpr int pp = decltype(Pc)::value;
          const int s = pp * kRowsPass + exH;
          const int e = lane_broadcast(e0, ic * kStreams + s);
          const uint32_t rowW = segCB + (uint32_t)(((s * kSegRows + (e >> 16)) * LdS) * 2);
          uint32_t msk = 0u;
          static_for<kWordsLane>([&](auto Wc) {
            constexpr int wl = decltype(Wc)::value;
            const int jw = exJ + wl * kLanesRow;
            uint32_t a = memo[wl];
            static_for<D>([&](auto Lc) {
              constexpr int l = decltype(Lc)::value;
              if constexpr (SVW::carves(l))
                a = mul_bf16x2(a, factor(rowB, Lc, 2 * jw, offW_of(s, Lc)));
            });
            if (e >= 0) store_shared_u32(rowW + (uint32_t)(jw * 4), a);
            //: THE PER-ROW TILE-LIVE MASK, produced BY the expansion because only the
            //: expansion knows which slots it wrote nonzero.  One ballot names every
            //: live packed pair of the pass's rows; a byte of it is an n-tile pair.
            const uint32_t bal = lane_ballot(e >= 0 && (a & kBf16MagnitudeMask) != 0u);
            static_for<kLanesRow / kMmaN>([&](auto Gc) {
              constexpr int g = decltype(Gc)::value;
              if (((bal >> (exH * kLanesRow + kMmaN * g)) & 0xFFu) != 0u)
                msk |= 1u << (wl * (kLanesRow / kMmaN) + g);
            });
          });
          //: ONE LANE PER COLUMN HALF publishes the row's mask into that half's word.
          //: Every word of the stream carries the WHOLE row's bits and each consumer
          //: reads only its own field, so this is the single instruction the one-word
          //: form was, with more lanes live rather than more issues.
          if (exPub && msk != 0u) or_shared_u32(mskB + (uint32_t)((s * WPS + exM) * 4), msk);
        });
      });
    };
    //: THE ROLLING COEFFICIENT STAGING.  The compressed factor rows of a batch's union
    //: tokens, `cp.async`ed one batch AHEAD of the expansion that reads them, so the
    //: expansion never carries a global latency and no per-token global factor load
    //: ever appears in the fetch loop.
    //: BODY COUNT: `kStgPass * D` = `D` at every built arm (`kThreads / kStgCh` is 32
    //: or 64, both at least `kSegRows`), each body one predicated asynchronous run.
    auto stage_factors = [&](int slot, int u0, int nb) {
      const int c = tid & (kStgCh - 1);
      const uint32_t blkB = stgB + (uint32_t)(slot * kSegRows * BP::kStageW * 2);
      static_for<kStgPass>([&](auto Pc) {
        const int r = (tid >> ilog2c(kStgCh)) + decltype(Pc)::value * kStgStep;
        if (r < nb) {
          const uint32_t tokB = winP + (uint32_t)wlistS[u0 + r] * (uint32_t)(BP::kWtot * 2);
          static_for<D>([&](auto Lc) {
            constexpr int l = decltype(Lc)::value;
            if (c >= BP::stage_chunk_base(l) && c < BP::stage_chunk_base(l) + BP::stage_chunks(l)) {
              const int sub = (c - BP::stage_chunk_base(l)) * BP::stage_el(l);
              stage_run<BP::stage_el(l) * 2>(
                  blkB + (uint32_t)((r * BP::kStageW + BP::stage_off(l) + sub) * 2),
                  gather_ptr(pW, tokB + (uint32_t)((runCol[l] + sub) * 2)));
            }
          });
        }
      });
    };
    //: THE GAIN rides the same block.  Its load is issued at the TOP of a fetch phase
    //: and its store at the BOTTOM, so the whole expansion covers the global latency,
    //: and the CTA barrier at the next batch's head is what publishes it.
    auto gain_load = [&](int u0, int nb) {
      unsigned short g = 0;
      if (tid < kSegRows && tid < nb)
        g = *reinterpret_cast<const unsigned short*>(
            gather_ptr(gW, winG + (uint32_t)wlistS[u0 + tid] * 2u));
      return g;
    };
    auto gain_store = [&](int slot, unsigned short g) {
      if (tid < kSegRows) store_shared_u16(gnB + (uint32_t)((slot * kSegRows + tid) * 2), g);
    };
    //: THE WINDOW'S CLEAR, paid ONCE before the window's first batch: it makes every row
    //: of the stream's segment finite, which is what lets every later batch leave its
    //: stale value rows in place.  IT IS SPLIT BY ROWS over the stream's `WPS` warps --
    //: the whole segment, value and coefficient, is covered before the rendezvous that
    //: follows it, so any partition of the rows is legal here and a row split keeps both
    //: legs contiguous.
    constexpr int kClrRows = kSegRows / WPS;  // segment rows one executor clears
    static_assert(kClrRows * WPS == kSegRows, "the segment's rows split over the stream's warps");
    constexpr int kClrV = kClrRows * LdV * 2 / 16, kClrC = kClrRows * LdS * 2 / 16;
    static_assert(kClrV * 16 == kClrRows * LdV * 2 && kClrC * 16 == kClrRows * LdS * 2,
                  "a segment row block is a whole number of sixteen-byte quanta");
    //: BODY COUNT: `ceil(kClrRows*LdV*2/512) + ceil(kClrRows*LdS*2/512)` = 5 + 3 at the
    //: uncolumned arms (`LdV = 72`, `LdS = 40`, `kSegRows = 16`), 3 + 2 at `WPS = 2`.
    auto clear_segment = [&]() {
      const uint4 z = make_uint4(0u, 0u, 0u, 0u);
      const int r0 = wstrm * kSegRows + col * kClrRows;
      const uint32_t vB = segVB + (uint32_t)(r0 * LdV * 2 + lane * 16);
      const uint32_t cB = segCB + (uint32_t)(r0 * LdS * 2 + lane * 16);
      static_for<(kClrV + kBallotLanes - 1) / kBallotLanes>([&](auto Pc) {
        constexpr int pp = decltype(Pc)::value;
        if (pp * kBallotLanes + 32 <= kClrV || pp * kBallotLanes + lane < kClrV)
          store_shared_vec16(vB + (uint32_t)(pp * kBallotLanes * 16), z);
      });
      static_for<(kClrC + kBallotLanes - 1) / kBallotLanes>([&](auto Pc) {
        constexpr int pp = decltype(Pc)::value;
        if (pp * kBallotLanes + 32 <= kClrC || pp * kBallotLanes + lane < kClrC)
          store_shared_vec16(cB + (uint32_t)(pp * kBallotLanes * 16), z);
      });
      if (lane == 0) store_shared_u32(mskB + (uint32_t)(strm * 4), 0u);
    };
    //: THE BATCH'S DEAD ROWS.  The union list ascends, so a stream's own tokens inside a
    //: batch are a PREFIX of its segment -- rows `[0, n)` for `n` placements -- and the
    //: dead rows are `[n, kSegRows)`.  THIS CLEAR IS SPLIT BY COLUMNS, not by rows, and
    //: that is a correctness requirement rather than a preference: it runs BETWEEN a
    //: batch's `ldmatrix` drain and the barrier that releases the writers, so a warp may
    //: only zero bytes IT ITSELF reads -- the stream's other warp is still draining rows
    //: that a row split would have this warp overwrite.  A column half of a row is
    //: `kLeafE` slots, contiguous and sixteen-byte aligned, so the run is the same
    //: predicated `STS.128` nest with a row stride carried per quantum.
    constexpr int kClrHalf = kLeafE * 2 / 16;  // quanta in ONE row's column half
    static_assert(kClrHalf * 16 == kLeafE * 2 && (kClrHalf & (kClrHalf - 1)) == 0,
                  "a column half of a coefficient row is a power-of-two number of quanta");
    constexpr int kDeadQ = kSegRows * kClrHalf;  // the whole segment's worth, the bound
    //: BODY COUNT: `ceil(kSegRows*kLeafE*2/512)` -- 2 at 32 leaves an executor, 1 at 16
    //: -- each one predicated store, and the stores actually issued are
    //: `(kSegRows - n) * kClrHalf`: zero on a full batch.
    auto clear_dead = [&](int n) {
      const uint4 z = make_uint4(0u, 0u, 0u, 0u);
      const int q = (kSegRows - n) * kClrHalf;
      const uint32_t cB = segCB + (uint32_t)(((wstrm * kSegRows + n) * LdS + colBase) * 2);
      static_for<(kDeadQ + kBallotLanes - 1) / kBallotLanes>([&](auto Pc) {
        constexpr int pp = decltype(Pc)::value;
        const int qi = pp * kBallotLanes + lane;
        if (qi < q)
          store_shared_vec16(
              cB + (uint32_t)((qi >> ilog2c(kClrHalf)) * LdS * 2 + (qi & (kClrHalf - 1)) * 16), z);
      });
      if (lane == 0) store_shared_u32(mskB + (uint32_t)(strm * 4), 0u);
    };

    const int nBatch = (nw + kSegRows - 1) / kSegRows;
    auto batch_n = [&](int b) {
      const int r = nw - b * kSegRows;
      return r <= 0 ? 0 : (r < kSegRows ? r : kSegRows);
    };
    if (nBatch > 0) {
      clear_segment();
      stage_factors(0, 0, batch_n(0));
      const unsigned short g0 = gain_load(0, batch_n(0));
      stage_commit();
      stage_wait<0>();
      gain_store(0, g0);
      rendezvous();
      {
        const int e0 = batch_place(0, batch_n(0));
        fetch_values(e0);
        const unsigned short g1 = gain_load(kSegRows, batch_n(1));
        stage_factors(1, kSegRows, batch_n(1));
        stage_commit();
        expand(0, e0);
        gain_store(1, g1);
      }
#pragma unroll 1
      for (int b = 0; b < nBatch; ++b) {
        //: LANDED.  Every fetcher's copies for batch `b` have arrived and the barrier
        //: makes every stream's rows visible to the stream that consumes them.
        stage_wait<0>();
        rendezvous();
        //: THE DRAIN, AND IT IS THE WHOLE OF THE CONSUMER.  BODY COUNT: `MCH + kNJ`
        //: `ldmatrix` (4 + 2 at every built arm) and `kNJ * (2*MCH + 2)` MMAs (2 * 10),
        //: both O(fragments) and both gated by the batch's tile-live mask.
        const uint32_t tm = load_shared_u32(mskB + (uint32_t)(strm * 4));
        uint32_t av[MCH][4], bc[kNJ][4];
        if (tm & mFld) {
          static_for<MCH>([&](auto Mc) {
            constexpr int m = decltype(Mc)::value;
            load_frag_t(av[m], myV + (uint32_t)(m * kMmaM * 2));
          });
          static_for<kNJ>([&](auto Nc) {
            constexpr int nj = decltype(Nc)::value;
            if (tm & (mBit << nj)) load_frag_t(bc[nj], ADDR_S(myC, 0, 2 * nj));
          });
        }
        //: the next batch's placement and its own row count come out of the window's
        //: published index, which no batch touches, so both are resolvable HERE -- and
        //: the count is what makes the clear below cost only the dead rows.
        const int n1 = batch_n(b + 1);
        const int e1 = batch_place((b + 1) * kSegRows, n1);
        clear_dead(batch_rows((b + 1) * kSegRows, n1));
        //: DRAINED.  The `ldmatrix` above has read the slot -- this is its completion
        //: point -- and the dead coefficient rows of the next batch are zero, so the
        //: writers may refill.
        __syncwarp();
        rendezvous();
        fetch_values(e1);
        const unsigned short gn = gain_load((b + 2) * kSegRows, batch_n(b + 2));
        stage_factors(b & 1, (b + 2) * kSegRows, batch_n(b + 2));
        stage_commit();
        expand((b + 1) & 1, e1);
        gain_store(b & 1, gn);
        //: AND THE MMAs RUN FROM FRAGMENTS, underneath the copies just issued.  That
        //: is where the overlap comes from in form (b): the drain-then-refill order,
        //: with registers as the second stage.
        static_for<kNJ>([&](auto Nc) {
          constexpr int nj = decltype(Nc)::value;
          if (tm & (mBit << nj)) {
            static_for<MCH>([&](auto Mc) {
              constexpr int m = decltype(Mc)::value;
              mma(st[m][2 * nj], av[m], &bc[nj][0]);
              mma(st[m][2 * nj + 1], av[m], &bc[nj][2]);
            });
            mma(mass_st[2 * nj], ones_a, &bc[nj][0]);
            mma(mass_st[2 * nj + 1], ones_a, &bc[nj][2]);
          }
        });
      }
      //: THE WINDOW EDGE.  Drain FIRST: the plane the next window publishes lives in
      //: these very segments, so an in-flight `cp.async` that landed after the barrier
      //: would write into the published plane.
      stage_wait<0>();
    }
    rendezvous();
  }

  // ---- the FINAL STATE, in LATTICE leaf order ------------------------------
  //: the mirror of the entry load, after the last fan-in has consumed the staging
  //: buffer -- which is what the one extra rendezvous below is for.
  if constexpr (STATE) {
    if (s_out != nullptr) {
      float* const plane = pin_address(s_out);
      rendezvous();
#pragma unroll
      static_for<kAtoms>([&](auto Ac) {
        page_regs(Ac, std::false_type{});
        rendezvous();
        page_sweep(page_off(atomBase + (uint32_t)decltype(Ac)::value),
                   [&](int sidx, uint32_t offb) {
                     store_f32(reinterpret_cast<float*>(gather_ptr(plane, offb)), so[sidx]);
                   });
        rendezvous();
      });
    }
  }
#undef ADDR_S
#undef ADDR_R
}

}  // namespace rola::carry
