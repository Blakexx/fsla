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
static constexpr int kCarryBuildMagic = 31010017;

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
//: This stage maps one stream to one EXECUTOR -- a warp on sm_80 -- so the CTA's
//: warp count is the stream count and never a constant of the file.

template <int D, int B, int K, int M, int W, int C, int S0, int S1, int S2, int S3>
struct CarryGeom {
  using BP = BoxPlan<D, B, K, M>;
  using SV = StreamSpans<D, B, K, M, S0, S1, S2, S3>;
  static constexpr int BC = BP::kBC;
  //: THE EXECUTOR MAP, 1:1 at this stage: one warp runs one stream.
  static constexpr int kWarps = SV::kStreams;
  static constexpr int kThreads = kWarps * 32;
  //: the stream's leaves, and the private panel row they index.
  static constexpr int kLeafW = SV::kLeaf;
  static constexpr int LdV = kDv + kPadE;
  static constexpr int LdS = kLeafW + kPadE;
  static constexpr int LdO = kDv + kPadF;

  //: THE TWO PANELS ARE A TWO-SLOT RING, AND THEY COST NOTHING EXTRA.  A window's
  //: readouts all precede its folds, so the write panel is dead through phase 1 and
  //: the read panel is dead through phase 2: each phase double-buffers in the
  //: other's buffer and produces one batch AHEAD, which is what puts the produce's
  //: global-load latency underneath the previous batch's MMA block.
  //: THE STRIP IS THE BATCH (2.4's two-tier panel).  Each PHASE double-buffers
  //: between two `CP`-token strips INSIDE ITS OWN HALF, so the other half is dead for
  //: the whole of that phase -- which is what lets the retained plane straddle it.
  //: The byte count is unchanged: two halves x two strips x `CP` == two x `C`.
  static constexpr int CP = kMmaK;
  static_assert(2 * CP == C, "a half holds exactly two strips");
  static constexpr int panel_bytes = 2 * 2 * kWarps * CP * LdS * (int)sizeof(__nv_bfloat16);
  //: the readout's staging block, one `kMmaN`-token region per warp -- and, at
  //: entry and exit, the state plane's page.
  //: THE FAN-IN STAGING IS GONE (2.6): the readout's RED issues straight from the
  //: accumulator, so this buffer's only remaining tenant is the state plane's PAGE at
  //: entry and exit -- one atom's leaves by every channel.
  //: the fan-in staging is deleted (2.6), so this buffer's only tenant is the state
  //: plane's PAGE at entry and exit.
  static constexpr int out_bytes = kAtomLeaves * LdO * (int)sizeof(float);
  //: ONE READ LIST PER STREAM, plus the CTA's two shared ones.  A read entry belongs
  //: to the streams whose boxes its digits reach; the write side keeps no list of its
  //: own -- its schedule IS the tier-2 bitmap, and its ids are transient: a per-stream
  //: cursor decompresses one ballot word at a time into a small ring, in the stage's
  //: shadow, so nothing window-sized is ever materialized on that side.
  static constexpr int kLists = kWarps + 2;  // + the CTA's shared read and write lists
  static constexpr int kIdRing = 64;
  static constexpr int id_bytes = kWarps * kIdRing * (int)sizeof(uint16_t);
  static constexpr int kListStride = ((W + 7) / 8) * 8;
  static constexpr int list_bytes = kLists * kListStride * (int)sizeof(uint16_t) + id_bytes;
  //: THE WINDOW'S BALLOT WORDS, one per lane of the walk -- so a window may be at
  //: most a lockstep unit's worth of words wide.
  static constexpr int kLW = W / kBallotLanes;
  //: THE V RING'S DEPTH IS DERIVED FROM THE SMEM BUDGET AND THE REGISTER FLOOR,
  //: exactly as the shipped `Geom::stages` is: the shallowest ring that reaches the
  //: best residency this arm can reach at any depth.  `kRegFloor` is the DECLARED
  //: floor for this family -- the measured census of the heaviest built arm,
  //: recorded in the findings doc and re-measured at every milestone.  A CTA target
  //: the live set cannot honour is a spill: a stream holds `prod_l S_l` leaves x all
  //: `kCols` channels, and 219 registers is what that costs at 32 leaves.
  static constexpr int kRegFloor = 220;
  //: ---- K35-E: THE TRANSIT PLANE -------------------------------------------
  //: The state EVACUATES registers at each phase change, because the readout and the
  //: fold carve the same leaves along DIFFERENT lattice axes.  The plane is
  //: `[channel][leaf]`, which is A row-major for the readout; its rows are the `kDv`
  //: value channels plus ONE mass row -- the mass accumulator's rows 1..15 are
  //: identically zero by construction of `ones_a`, so moving them would move zeros.
  static constexpr int kPlaneRows = kDv + 1;
  //: `LD` AND ITS SWIZZLE ARE A DERIVATION, NOT A GUESS (findings, derivation 2).
  //: The push varies a lane's word index in the LOW bits and the pull varies it in
  //: bit 3 and above, so NO linear stride is conflict-free on both legs -- searched
  //: over `LD`, best is 2-way.  `word ^= (chan & 7)` at `LD = BC + 16` measures
  //: max-way ONE on both legs, over every `(m, g, w, d0l)` slice: what the search
  //: found is a row stride of `8 mod 32` u32 words, which `BC + 16` holds at every
  //: box a page-aligned `BC` admits (144 at `BC = 128`, 272 at `BC = 256`).
  static constexpr int kHiLD = BC + 16;
  static constexpr int kHiSwz = 7;
  static constexpr int plane_bytes = kPlaneRows * kHiLD * (int)sizeof(__nv_bfloat16);
  //: THE VALUE STREAM IS PER-WARP AND PRIVATE (2.4): one `CP`-token tile per warp.
  //: A single buffer suffices -- `ldmatrix` drains the tile to registers before its
  //: MMAs, so the WAR window is the drain only and the next fill is issued into it
  //: immediately after.
  //: TWO slots per warp.  One is not enough: `cp.async` is asynchronous, so a fill
  //: issued after the drain can land BEFORE the `ldmatrix` has read the buffer -- a
  //: WAR hazard whose incidence grows with the tile count (measured: the scattered
  //: cells failed and the cohort cells, with a quarter the tiles, passed).  The
  //: second slot is the card's insurance and restores the fill-under-MMA overlap.
  static constexpr int kVSlots = 2;
  static constexpr int slot_bytes = kWarps * kVSlots * CP * LdV * (int)sizeof(__nv_bfloat16);
  //: THE TRANSIT ALIASES THE PANEL.  It is live ONLY at the phase change, when the
  //: panels are dead, so it costs no allocation of its own -- the straddling ledger.
  static_assert(plane_bytes <= panel_bytes, "the transit must fit inside the dead panel region");
  static constexpr int fixed_bytes = panel_bytes + out_bytes + list_bytes;
  //: ONE slot per warp, not a ring: async DEPTH is the warp's own, and the fill for
  //: the next tile is in flight across this tile's MMAs.
  static constexpr int kStages = 1;
  static constexpr size_t bytes = (size_t)fixed_bytes + (size_t)slot_bytes;
  static constexpr int cta_target = design::cta_target((int)bytes, kThreads, kRegFloor);
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
template <int D, int B, int K, int M, int W, int C, int S0, int S1, int S2, int S3>
constexpr int kCarryCtaTarget = CarryGeom<D, B, K, M, W, C, S0, S1, S2, S3>::cta_target;

// ---------------------------------------------------------------- the kernel

template <int D, int B, int K, int M, int W, int C, int W0, int W1, int W2, int W3, int R0, int R1,
          int R2, int R3, uint32_t MO, bool STATE>
__global__ __launch_bounds__((CarryGeom<D, B, K, M, W, C, R0, R1, R2, R3>::kThreads),
                             (kCarryCtaTarget<D, B, K, M, W, C, R0, R1, R2, R3>)) void
carry_kernel(const __nv_bfloat16* __restrict__ pread, const __nv_bfloat16* __restrict__ pwrite,
             const __nv_bfloat16* __restrict__ gwrite, const __nv_bfloat16* __restrict__ v,
             const uint32_t* __restrict__ rows, float* __restrict__ o, float* __restrict__ den,
             int L, int tok_words, int v_row_bytes, const float* __restrict__ s_in,
             float* __restrict__ s_out) {
  using BP = BoxPlan<D, B, K, M>;
  using G = CarryGeom<D, B, K, M, W, C, R0, R1, R2, R3>;
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
  constexpr int LdV = G::LdV, LdS = G::LdS, LdO = G::LdO;
  constexpr int CP = G::CP;
  constexpr int MTOK = CP / kMmaM;  // token m-tiles of a strip
  //: the warp's own leaves: `NL` accumulator n-tiles, `KDw` readout k-steps.
  constexpr int kLeafW = G::kLeafW;
  constexpr int NL = kLeafW / kMmaN;
  constexpr int KDw = kLeafW / kMmaK;
  constexpr int MCH = kDv / kMmaM;  // the channel m-tiles, ALL of them per warp
  static_assert(C % kMmaM == 0 && kThreads % C == 0, "the batch tile must divide");
  static_assert(kLeafW % kMmaK == 0, "a stream's leaves are a whole number of k-tiles");
  static_assert(BC % kAtomLeaves == 0, "K2: an owner holds whole page atoms");
  static_assert(design::smem_fits(G::bytes), "this arm does not fit the SMEM maximum");
  static_assert(LdS % 8 == 0 && LdV % 8 == 0, "an ldmatrix row address must be 16-byte aligned");
  static_assert(kWarps <= kBallotLanes, "one lane of the walk slices each executor's share");
  static_assert(SVW::kStreams == SVR::kStreams,
                "this stage maps one stream to one executor, so the two sides' carves must "
                "cut the box into the same number of streams");
  static_assert(W % kBallotLanes == 0 && kThreads % kBallotLanes == 0,
                "a warp's tokens must be exactly one ballot word of the owner's row");

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int owner = (int)blockIdx.x;
  const int bh = (int)blockIdx.y;

  extern __shared__ __align__(16) char carry_smem[];
  //: ONE slot per warp is enough because the drain precedes the issue: `ldmatrix`
  //: lifts the tile into registers, and only then is the next fill launched into the
  //: same buffer, so the WAR window is the drain itself (2.4).
  constexpr int STAGES = G::kStages;
  static_assert(STAGES == 1, "the private stream carries one slot per warp");
  __nv_bfloat16* const sv = reinterpret_cast<__nv_bfloat16*>(carry_smem);
  //: THE FOLD'S TENANTS COME FIRST AND ADJACENT -- the private value slots and the
  //: write strips -- because the retained plane straddles exactly those two, and
  //: they are the pair that is dead for the whole readout.
  __nv_bfloat16* const sw = sv + kWarps * G::kVSlots * CP * LdV;
  __nv_bfloat16* const sr = sw + 2 * kWarps * CP * LdS;
  float* const so = reinterpret_cast<float*>(sr + 2 * kWarps * CP * LdS);
  constexpr int kListStride = G::kListStride;
  uint16_t* const slist = reinterpret_cast<uint16_t*>(so + kAtomLeaves * LdO);
  //: THE RETAINED PLANE (2.2), straddled over the FOLD's tenants -- the write strips
  //: and the page buffer, both dead for the whole readout.  It is published once and
  //: stays valid until the fold reclaims it, which is what the readout's pull needs
  //: from it and the writers reload out of it with no republish leg.
  static_assert(
      G::plane_bytes <= kWarps * G::kVSlots * G::CP * LdV * 2 + 2 * kWarps * G::CP * LdS * 2,
      "the retained plane must fit the fold's dead region");
  const SmemAddr hiB = smem_addr(sv);

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
    offW[l] = ((strm >> ilog2c(SVW::grid_div(l))) & (SVW::g(l) - 1)) * SVW::s(l);
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
  __nv_bfloat16* const panR0 = sr + warp * CP * LdS;
  __nv_bfloat16* const panR1 = sr + (kWarps + warp) * CP * LdS;
  __nv_bfloat16* const panW0 = sw + warp * CP * LdS;
  __nv_bfloat16* const panW1 = sw + (kWarps + warp) * CP * LdS;
  const uint32_t brT0 = smem_addr(panR0) + (uint32_t)((rT * LdS + cT * 8) * 2);
  const uint32_t brT1 = smem_addr(panR1) + (uint32_t)((rT * LdS + cT * 8) * 2);
  const uint32_t bwF0 = smem_addr(panW0) + (uint32_t)((rF * LdS + cF * 8) * 2);
  const uint32_t bwF1 = smem_addr(panW1) + (uint32_t)((rF * LdS + cF * 8) * 2);
  //: the same base with the ROW left out: S-write's fold supplies the row per lane.
  const uint32_t svVc = smem_addr(sv) + (uint32_t)(cT * 8 * 2);
#define ADDR_S(b_, R_, g_) ((b_) + (uint32_t)(((R_) * LdS + (g_) * 8) * 2))
#define ADDR_V(b_, R_, dv_) ((b_) + (uint32_t)(((R_) * LdV + (dv_)) * 2))

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

  //: THE READ DRAIN'S STATE SOURCE, AS ONE SEAM.  K35: the fragment is the
  //: accumulator's TOP HALVES taken by `prmt` -- one instruction per pair, and the
  //: same bytes the hi-plane hybrid will publish, so that hybrid is a placement
  //: change and never an arithmetic one.  It TRUNCATES where the image ROUNDED;
  //: declared, and mirrored in the oracle.
  //:  It yields the readout's A
  //: fragments for k-step `kk` -- the `MCH` value m-tiles and the mass row -- and it
  //: is the ONLY place the drain touches the state's residency.  Here the
  //: accumulator IS the operand (the warp's own n-tiles, packed in place, no
  //: `ldmatrix`); a state held elsewhere substitutes its own load here and nothing
  //: else in the drain moves.
  auto state_operand = [&](auto Kc, uint32_t(&sa)[MCH][4], uint32_t(&ma)[4]) {
    constexpr int kk = decltype(Kc)::value;
    static_for<MCH>([&](auto Mc) {
      constexpr int m = decltype(Mc)::value;
      sa[m][0] = hi_bf16x2(st[m][2 * kk][0], st[m][2 * kk][1]);
      sa[m][1] = hi_bf16x2(st[m][2 * kk][2], st[m][2 * kk][3]);
      sa[m][2] = hi_bf16x2(st[m][2 * kk + 1][0], st[m][2 * kk + 1][1]);
      sa[m][3] = hi_bf16x2(st[m][2 * kk + 1][2], st[m][2 * kk + 1][3]);
    });
    ma[0] = hi_bf16x2(mass_st[2 * kk][0], mass_st[2 * kk][1]);
    ma[1] = hi_bf16x2(mass_st[2 * kk][2], mass_st[2 * kk][3]);
    ma[2] = hi_bf16x2(mass_st[2 * kk + 1][0], mass_st[2 * kk + 1][1]);
    ma[3] = hi_bf16x2(mass_st[2 * kk + 1][2], mass_st[2 * kk + 1][3]);
  };

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
  auto leafW_of = [&](int P) {
    int lf = 0;
    static_for<D>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      const int dig = ((P >> PMW::shift(l)) & (PMW::span(l) - 1)) + offW[l];
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
  //: -- exactly `state_operand`'s register order, so the drain below is unchanged.
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
  static_assert(PMW::SPT == SPT, "both carves split a row the same way");
  const int prow = lane / PM::kRowThr;  // this lane's token row of the strip
  const int psub = lane % PM::kRowThr;  // which half of that row's slots it mints
  int lvCol[D], lvColW[D];
  static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    lvCol[l] = runCol[l] + offR[l];
    lvColW[l] = runCol[l] + offW[l];
  });

  //: A DEAD ROW IS ZEROED AT THE PAIR LEVEL'S FACTOR.  A padding row of a partial
  //: batch has no token, and a zero coefficient contributes exactly nothing to either
  //: MMA (the produce's dead-row zeroing law).  The pair level's run is the factor
  //: every slot of the row carries, so zeroing it covers the whole row for one or two
  //: selects and costs no arithmetic.
  //:
  //: THE READ SIDE HAS NO GAIN.  Its scalar is identically 1 and the trailing
  //: `mul_bf16x2(acc, 1)` was 44% of the read produce -- 32 FMUL, 32 PRMT and 16 F2FP
  //: for `rn(a*1) = a` -- so the side is a compile-time argument and the read body
  //: stores its accumulators directly.
  auto produce = [&](auto Wt, const __nv_bfloat16* base, __nv_bfloat16* dst, uint32_t tokOff,
                     uint32_t gain, bool have) {
    constexpr bool WRITE = decltype(Wt)::value;
    using MapT = std::conditional_t<WRITE, PMW, PM>;
    const int* const cols = WRITE ? lvColW : lvCol;
    //: THE SLOT ACCUMULATORS ARE THE ONLY ARRAY THE PRODUCE CARRIES.  A level's run
    //: is loaded, folded into every slot and dropped inside one `static_for` body, so
    //: no `[D][n]` array is ever live.
    uint32_t acc[SPT / 2];
    {
      //: EACH SIDE EXPANDS UNDER ITS OWN SPAN MAP, and the pair comes from the level
      //: lying LOWEST in that map's private order -- the lattice's innermost level on
      //: both sides, by rule 1 -- so ONE run both supplies the adjacent pair and
      //: carries the dead-row zeroing.
      //: THE LANE'S SLOT BASE IS `psub * SPT`, and each level reads its digit out of
      //: `p` at that level's own shift -- so the base contributes a CONSTANT digit
      //: offset per level, which `idx_of` (indexing the LOCAL slot) cannot see.  Fold
      //: it into the run's column.  At the write carve level 0 this is exactly the
      //: lane half `psub`; without it every odd lane mints level 0's digit zero.
      constexpr int PL = MapT::kPairLevel;
      const int adjPL = ((psub * MapT::SPT) >> MapT::shift(PL)) & (MapT::span(PL) - 1);
      unsigned short run[MapT::n_l(PL)];
      load_run<MapT::n_l(PL)>(run, gather_ptr(base, tokOff + (uint32_t)((cols[PL] + adjPL) * 2)));
      {
        static_for<MapT::n_l(PL)>([&](auto Ic) {
          constexpr int i = decltype(Ic)::value;
          run[i] = have ? run[i] : (unsigned short)0;
        });
      }
      static_for<SPT / 2>([&](auto Jc) {
        constexpr int j = 2 * decltype(Jc)::value;
        acc[j / 2] =
            ((uint32_t)run[MapT::idx_of(PL, j + 1)] << 16) | (uint32_t)run[MapT::idx_of(PL, j)];
      });
    }
    static_for<D>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      if constexpr (l != MapT::kPairLevel) {
        const int adj = ((psub * MapT::SPT) >> MapT::shift(l)) & (MapT::span(l) - 1);
        unsigned short run[MapT::n_l(l)];
        load_run<MapT::n_l(l)>(run, gather_ptr(base, tokOff + (uint32_t)((cols[l] + adj) * 2)));
        static_for<SPT / 2>([&](auto Jc) {
          constexpr int j = 2 * decltype(Jc)::value;
          acc[j / 2] = mul_bf16x2(acc[j / 2], splat_bf16x2_bits(run[MapT::idx_of(l, j)]));
        });
      }
    });
    __nv_bfloat16* const row = dst + prow * LdS + psub * SPT;
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
    //: THE WRITE SIDE'S SCHEDULE IS THE SHARED LIST built above, and each warp's own
    //: share of it is its tier-2 digit row `mineW` -- tested per id by the pull rather
    //: than decompressed per ballot word.  The COMPACTION (`mineW` and not `wideW`) is
    //: what makes a warp fold only the tokens that reach its leaves; it changes the
    //: fold's k-order, so it carries the declared re-baseline and Scan A's shared
    //: order function with it.
    uint16_t* const idring =
        reinterpret_cast<uint16_t*>(slist + G::kLists * G::kListStride) + warp * G::kIdRing;

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
    if (nbr > 0) produce_batch(std::false_type{}, pR, panR0, rlist, 0, nr);
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
      produce_batch(std::false_type{}, pR, ((i + 1) & 1) ? panR1 : panR0, rlist, (i + 1) * CP, nr);
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

    // ---- PHASE 2: the FOLD, over the warp's OWN private value stream --------
    //: NO BATCHES, NO SHARED RING, NO RENDEZVOUS (2.4).  Each warp streams ITS OWN
    //: tokens into its own tile buffer and folds them; the only synchronization left
    //: in the window is the phase change above.  Ownership partitions the write
    //: tokens at the write carve, so the duplication this costs is the declared
    //: write-multiplicity bound (1.04-2.96 at the ALT arms), absorbed by L2.
    constexpr int kIdRing = G::kIdRing;
    constexpr int kChunkV = kDv / 8;  // 16-B chunks across a value row
    constexpr int kRowsW = 32 / kChunkV;
    constexpr int kPassW = CP / kRowsW;
    static_assert(kPassW * kRowsW == CP && (kIdRing & (kIdRing - 1)) == 0, "tile / ring shape");
    const int ldRowW = lane / kChunkV, ldColW = (lane % kChunkV) * 8;
    constexpr uint32_t kVSlotB = (uint32_t)(CP * LdV) * 2;
    __nv_bfloat16* const svW = sv + warp * G::kVSlots * CP * LdV;
    const uint32_t svLw = smem_addr(svW) + (uint32_t)((ldRowW * LdV + ldColW) * 2);
    const uint32_t svAw = smem_addr(svW) + (uint32_t)((rT * LdV + cT * 8) * 2);

    //: THE CURSOR IS THE DECOMPRESSOR (doctrine 7).  One ballot word a step:
    //: broadcast it, ballot the lane's own bit, rank it inside the word, append.  No
    //: cross-warp prefix and nothing window-sized -- the ids live only in a ring of
    //: `kIdRing`, which is the pipeline's depth and not the window's length.  The
    //: earlier ballot-cursor refutation was SHARED-stream-scoped: it failed because
    //: rank-indexed consumers needed a global rank.  Here the warp's stream order IS
    //: the pull's order and every consumer is affine in it.
    int head = 0, tail = 0, si = 0;
    //: `below` is lane-invariant AND loop-invariant, so it is built once here rather
    //: than rebuilt on every visit as it was.
    const uint32_t below = (1u << lane) - 1u;
    //: THE PULL.  A step tests 32 of the SHARED list's ids against this warp's own
    //: digit row, so the walk is over LIVE OWNER IDS rather than over the window's
    //: 512 token slots -- which is the whole difference at a sparse cell.  Token `t`'s
    //: membership bit lives in lane `t >> 5`'s word of `mineW`, and `__shfl` takes a
    //: PER-LANE source, so the gather of 32 different words is one instruction.
    auto refill = [&]() {
      while (tail - head <= kIdRing - kBallotLanes && si < nw) {
        const int idx = si + lane;
        const bool inR = idx < nw;
        const int t = inR ? (int)wlistS[idx] : 0;
        const uint32_t w = (uint32_t)lane_broadcast((int)mineW, t >> 5);
        const bool has = inR && (((w >> (t & 31)) & 1u) != 0u);
        const uint32_t bal = lane_ballot(has);
        if (has) idring[(tail + bit_count(bal & below)) & (kIdRing - 1)] = (uint16_t)t;
        tail += bit_count(bal);
        si += kBallotLanes;
      }
      __syncwarp();
    };
    auto stage_tile_w = [&](int h, int cnt, int slot) {
#pragma unroll
      for (int p = 0; p < kPassW; ++p) {
        const int r = ldRowW + p * kRowsW;
        const int t = idring[(h + (r < cnt ? r : (cnt > 0 ? cnt - 1 : 0))) & (kIdRing - 1)];
        stage_tile(svLw + (uint32_t)slot * kVSlotB + (uint32_t)(p * kRowsW * LdV * 2),
                   gather_ptr(vPlane,
                              (uint32_t)t * (uint32_t)v_row_bytes + winV + (uint32_t)(ldColW * 2)));
      }
    };
    auto produce_tile = [&](__nv_bfloat16* dst, int h, int cnt) {
      const bool have = prow < cnt;
      const int tok = idring[(h + (have ? prow : 0)) & (kIdRing - 1)];
      const uint32_t gain = have ? splat_bf16x2(*reinterpret_cast<const __nv_bfloat16*>(
                                       gather_ptr(gW, winG + (uint32_t)tok * 2u)))
                                 : 0u;
      produce(std::true_type{}, pW, dst, winP + (uint32_t)tok * (uint32_t)(BP::kWtot * 2), gain,
              have);
    };

    refill();
    int cnt = tail - head < CP ? tail - head : CP;
    if (cnt > 0) {
      stage_tile_w(head, cnt, 0);
      stage_commit();
      produce_tile(panW0, head, cnt);
      __syncwarp();
    }
    //: THE LOOP'S STRUCTURAL BOUND.  A warp can never own more than the window's `W`
    //: tokens, so it can never need more than `ceil(W / CP)` tiles; the cursor must
    //: not be trusted to terminate the loop on its own.  Measured: without this the
    //: sparsest cell (`alt-k4-W512`) inflates its state ~38x, which is spurious tiles
    //: folding past the end of the schedule.
    const int kTileCap = W / CP + 2;
#pragma unroll 1
    for (int t = 0; cnt > 0 && t < kTileCap; ++t) {
      stage_wait<0>();
      //: DRAIN FIRST: the tile is in registers before the next fill is issued into
      //: the same buffer, which is the whole of the single-slot WAR window.
      uint32_t a[MCH][4];
#pragma unroll
      for (int m = 0; m < MCH; ++m)
        load_frag_t(a[m], svAw + (uint32_t)(t & 1) * kVSlotB + (uint32_t)(m * kMmaM * 2));
      head += cnt;
      refill();
      const int nxt = tail - head < CP ? tail - head : CP;
      //: the next fill goes into the OTHER slot, so it cannot overtake this tile's
      //: drain, and it flies underneath this tile's MMAs.
      if (nxt > 0) {
        stage_tile_w(head, nxt, (t + 1) & 1);
        stage_commit();
      }
      const uint32_t bwF = (t & 1) ? bwF1 : bwF0;
#pragma unroll
      for (int nj = 0; nj < NL / 2; ++nj) {
        uint32_t b[4];
        load_frag_t(b, ADDR_S(bwF, 0, 2 * nj));
#pragma unroll
        for (int m = 0; m < MCH; ++m) {
          mma(st[m][2 * nj], a[m], &b[0]);
          mma(st[m][2 * nj + 1], a[m], &b[2]);
        }
        mma(mass_st[2 * nj], ones_a, &b[0]);
        mma(mass_st[2 * nj + 1], ones_a, &b[2]);
      }
      if (nxt > 0) {
        produce_tile((t & 1) ? panW0 : panW1, head, nxt);
        __syncwarp();
      }
      cnt = nxt;
    }
    //: THE WINDOW EDGE, AND THE ORDER MATTERS.  Drain FIRST: the plane the next
    //: window publishes lives in these very value slots, so an in-flight `cp.async`
    //: that lands after the barrier would write into the published plane.  Then
    //: fence, because private streams end the warps' lockstep and a warp whose
    //: cursor ran dry early must not publish over a slower warp's strips.  These are
    //: the window's only CTA syncs besides the phase change.
    stage_wait<0>();
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
#undef ADDR_V
}

}  // namespace rola::carry
