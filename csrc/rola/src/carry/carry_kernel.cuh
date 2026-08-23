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
static constexpr int kCarryBuildMagic = 31010016;

static constexpr int kDv = 64;         // d_v
static constexpr int kCols = kDv + 1;  // the mass column rides the state
static constexpr int kPadE = 8;        // bf16 row padding, elements (16 B)
static constexpr int kPadF = 4;        // fp32 row padding, elements (16 B)
static constexpr int kWarps = 4;
static constexpr int kThreads = kWarps * 32;
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

//: THE CARVE (K33 clean-sheet 1.2A, tier A).  A warp owns one GROUP of the
//: innermost level's digits across every outer digit -- `BC / kWarps` leaves and
//: ALL `kCols` channels -- and tiles those leaves so that a group's digit is a
//: whole accumulator n-tile.  That tiling is private to the warp: the lattice's
//: `local` order, the state plane, decode and the facts geometry never see it, so
//: the readout needs no state image, no `ldmatrix` for A and no CTA barrier.  The
//: read phase therefore shares NOTHING (per-warp lists, per-warp panel region,
//: warp-private staging, direct RED); the write phase keeps the shared value ring
//: and its per-batch rendezvous, because a write entry is live at every one of the
//: innermost level's digits and so belongs to all four warps.

template <int D, int B, int K, int M, int W, int C, int CL>
struct CarryGeom {
  using BP = BoxPlan<D, B, K, M>;
  static constexpr int BC = BP::kBC;
  //: the warp's leaves, and the private panel row they index.
  static constexpr int kLeafW = BC / kWarps;
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
  static constexpr int out_bytes = C * LdO * (int)sizeof(float);
  //: `kWarps` READ lists, one per digit group, then ONE write list: a read entry
  //: belongs to the groups its innermost digits fall in, a write entry to all.
  //: `kWarps` READ lists and NOTHING ELSE (2.4/2.7).  The write side keeps no list at
  //: all: its schedule IS the tier-2 bitmap, and its ids are transient -- a per-warp
  //: cursor decompresses one ballot word at a time into a small ring, in the stage's
  //: shadow, so nothing window-sized is ever materialized on that side.
  static constexpr int kLists = kWarps;
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
  //: the live set cannot honour is a spill: the carve holds `BC/kWarps` leaves x all
  //: `kCols` channels per warp, and 219 registers is what that costs.
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
  //: over `LD`, best is 2-way.  `word ^= (chan & 7)` at `LD = 144` measures max-way
  //: ONE on both legs, over every `(m, g, w, d0l)` slice.
  static constexpr int kHiLD = 144;
  static constexpr int kHiSwz = 7;
  static constexpr int plane_bytes = kPlaneRows * kHiLD * (int)sizeof(__nv_bfloat16);
  //: THE VALUE STREAM IS PER-WARP AND PRIVATE (2.4): one `CP`-token tile per warp.
  //: A single buffer suffices -- `ldmatrix` drains the tile to registers before its
  //: MMAs, so the WAR window is the drain only and the next fill is issued into it
  //: immediately after.
  static constexpr int slot_bytes = kWarps * CP * LdV * (int)sizeof(__nv_bfloat16);
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

//: THE PRIVATE LEAF ORDER AND THE PRODUCE'S THREAD MAP.  A warp's leaves are
//: `p = (r_CL - grp*kGS) * kOut + outer`: the carve level SLOWEST, so its digit
//: is a contiguous run of `kOut` leaves, and the outer levels in the lattice's own
//: mixed radix beneath it.  Because `p` is a bit permutation of the levels' digit
//: offsets exactly as `local` is, the expansion is `sum_l n_l` amplitude loads
//: followed by products of register values at CONSTANT indices -- no dynamic
//: indexing, no spill, no materialized gate plane.
//: `QMAJOR` NAMES WHICH LEVEL LIES LOWEST IN THE PRIVATE INDEX `p`, and the two
//: carves want opposite answers.  Under q-major the CARVE level is lowest, so a
//: lane's two adjacent slots are two adjacent digits of its own group.  Under
//: CL-major the carve level is HIGHEST and the lattice's innermost level is lowest,
//: so a lane's two adjacent slots are LATTICE-ADJACENT LEAVES -- which is what the
//: transit's `bf16x2` publish requires of the write carve.
template <int D, int B, int K, int M, int C, int CL, bool QMAJOR = true>
struct CarveMap {
  using BP = BoxPlan<D, B, K, M>;
  static constexpr int kLevel = CL;
  static constexpr int kGS = BP::s(kLevel) / kWarps;  // the group's digits
  static constexpr int kOut = BP::kBC / BP::s(kLevel);
  static constexpr int kLeaf = kGS * kOut;
  static constexpr int kRowThr = 32 / C;
  static constexpr int SPT = kLeaf / kRowThr;
  //: `kRowThr` lanes share a token's panel row, each minting `SPT` contiguous private
  //: slots.  At the strip shape that is TWO (32 lanes, 16 tokens), and the produce
  //: then needs the lane's slot base folded back into each level's run column --
  //: `colAdj` in `produce` (the 2.1 paper derivation for this shape).
  static_assert(kRowThr * SPT == kLeaf && (kRowThr & (kRowThr - 1)) == 0,
                "a token's panel row splits over a power-of-two lane group");
  //: THE PRIVATE ORDER IS q-MAJOR (K35 M1b): `p = outer * kGS + (r_CL - grpBase)`,
  //: so a level-0 digit owns `kGS` ADJACENT leaves and an `mma` n-tile of eight
  //: holds `8/kGS` whole digits.  That is what makes the write side's per-k-tile
  //: DIGIT UNION reapable as an n-tile skip: under the previous CL-major order one
  //: n-tile spanned ALL of the outer level's digits and there was nothing to skip.
  //: The relabelling is a PERMUTATION of the accumulator's n axis and nothing else
  //: -- the k order, and therefore the state plane, is bit-identical under it.
  //: the bits of the NON-CARVE levels lying inside `l`.  `run_shift` counts every
  //: level inside `l`, so the carve level's own span comes back out when it is one
  //: of them -- which is what lets the carve sit at an OUTER level at all.
  static constexpr int inner_bits(int l) {
    return BP::run_shift(l) - (kLevel > l ? BP::span_bits(kLevel) : 0);
  }
  static constexpr int shift(int l) {
    return QMAJOR ? (l == kLevel ? 0 : ilog2c(kGS) + inner_bits(l))
                  : (l == kLevel ? ilog2c(kOut) : inner_bits(l));
  }
  static constexpr int span(int l) { return l == kLevel ? kGS : BP::s(l); }
  static constexpr int vary(int l) { return SPT >> shift(l); }
  static constexpr int n_l(int l) {
    return vary(l) < 1 ? 1 : (vary(l) > span(l) ? span(l) : vary(l));
  }
  static constexpr int idx_of(int l, int j) { return (j >> shift(l)) & (n_l(l) - 1); }
  //: the ADJACENT PAIR comes from the level lying LOWEST in `p`, which under the
  //: q-major order is the CARVE LEVEL itself: a lane's two adjacent slots are two
  //: adjacent digits of its own group.
  static constexpr int kPairLevel = QMAJOR ? kLevel : (kLevel == D - 1 ? D - 2 : D - 1);
  static_assert(shift(kPairLevel) == 0 && n_l(kPairLevel) >= 2,
                "the carve level lies lowest in the private order and supplies the pair");
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
template <int D, int B, int K, int M, int W, int C, int CL>
constexpr int kCarryCtaTarget = CarryGeom<D, B, K, M, W, C, CL>::cta_target;

// ---------------------------------------------------------------- the kernel

template <int D, int B, int K, int M, int W, int C, int CL, int CW, uint32_t MO, bool STATE>
__global__ __launch_bounds__(kThreads, (kCarryCtaTarget<D, B, K, M, W, C, CL>)) void carry_kernel(
    const __nv_bfloat16* __restrict__ pread, const __nv_bfloat16* __restrict__ pwrite,
    const __nv_bfloat16* __restrict__ gwrite, const __nv_bfloat16* __restrict__ v,
    const uint32_t* __restrict__ rows, float* __restrict__ o, float* __restrict__ den, int L,
    int tok_words, int v_row_bytes, const float* __restrict__ s_in, float* __restrict__ s_out) {
  using BP = BoxPlan<D, B, K, M>;
  using G = CarryGeom<D, B, K, M, W, C, CL>;
  using PM = CarveMap<D, B, K, M, G::CP, CL>;
  //: THE WRITE CARVE'S MAP.  CL-major, so a lane's two adjacent private slots are
  //: LATTICE-ADJACENT leaves -- what the transit's `bf16x2` publish requires.  A
  //: single-carve body reuses the read map and nothing below changes.
  using PMW =
      std::conditional_t<(CW >= 0), CarveMap<D, B, K, M, G::CP, (CW >= 0 ? CW : 0), false>, PM>;
  //: THE ROW TABLE.  A two-carve body votes its rows under `TwoCarveRows` -- the same
  //: table refined at BOTH carved levels -- so each warp reads its OWN group's row on
  //: either side.  A single-carve body keeps `CarveRows` and the write list stays
  //: shared, because there every warp owns every write entry.
  static constexpr bool kTwo = CW >= 0;
  using CR = std::conditional_t<kTwo, TwoCarveRows<D, B, K, M, kWarps, (kTwo ? CW : 0), CL>,
                                CarveRows<D, B, K, M, kWarps, CL>>;

  constexpr int BC = BP::kBC;
  constexpr int LdV = G::LdV, LdS = G::LdS, LdO = G::LdO;
  constexpr int CP = G::CP;
  constexpr int MTOK = CP / kMmaM;  // token m-tiles of a strip
  //: the warp's own leaves: `NL` accumulator n-tiles, `KDw` readout k-steps.
  constexpr int kLeafW = G::kLeafW;
  constexpr int NL = kLeafW / kMmaN;
  constexpr int KDw = kLeafW / kMmaK;
  constexpr int MCH = kDv / kMmaM;  // the channel m-tiles, ALL of them per warp
  //: the carve level's group geometry, and the page's rectangle in it.
  constexpr int kGS = PM::kGS, kOut = PM::kOut;
  constexpr int kAtomOut = kAtomLeaves / BP::s(CL);
  static_assert(kAtomOut >= 1 && kAtomOut * BP::s(CL) == kAtomLeaves,
                "K3: the carve level's span divides the page granule, so a page is a whole "
                "number of outer runs times every one of that level's digits");

  static_assert(C % kMmaM == 0 && kThreads % C == 0, "the batch tile must divide");
  static_assert(kLeafW % kMmaK == 0, "the warp's leaves are a whole number of k-tiles");
  static_assert(BC % kAtomLeaves == 0, "K2: an owner holds whole page atoms");
  static_assert(design::smem_fits(G::bytes), "this arm does not fit the SMEM maximum");
  static_assert(LdS % 8 == 0 && LdV % 8 == 0, "an ldmatrix row address must be 16-byte aligned");
  static_assert(BP::span_bits(CL) >= 1, "the carve level supplies the adjacent pair");
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
  __nv_bfloat16* const sr = sv + kWarps * CP * LdV;
  __nv_bfloat16* const sw = sr + 2 * kWarps * CP * LdS;
  float* const so = reinterpret_cast<float*>(sw + 2 * kWarps * CP * LdS);
  constexpr int kListStride = G::kListStride;
  uint16_t* const slist = reinterpret_cast<uint16_t*>(so + C * LdO);
  //: THE RETAINED PLANE (2.2), straddled over the FOLD's tenants -- the write strips
  //: and the page buffer, both dead for the whole readout.  It is published once and
  //: stays valid until the fold reclaims it, which is what lets the readout steal
  //: from it and the writers reload out of it with no republish leg.
  static_assert(G::plane_bytes <= 2 * kWarps * G::CP * LdS * 2 + G::out_bytes,
                "the retained plane must fit the fold's dead region");
  const SmemAddr hiB = smem_addr(sw);

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

  //: THE GROUP ROTATION.  Two CTAs co-resident on an SM are one grid wave apart and
  //: the wave is a multiple of `kWarps`, so `owner % kWarps` would rotate both
  //: identically; the wave-stride form does not (K33 1.45).
  const int grp = (warp + ((owner + (owner >> 5)) & (kWarps - 1))) & (kWarps - 1);

  // ---- the owner's digit runs, one per level -------------------------------
  //: `o_l` is a bit-field extract of the owner index: the lattice's owner
  //: coordinates are the top `sum_l log2 g_l` bits of `lambda` (R0 spec section 2.4).
  int runCol[D];
  //: AND THE BALLOT ROW, per level: the SAME `o_l`, addressing the row the facts
  //: pass voted for this owner group at this level (K31-i-d) -- at the CARVE grain,
  //: so the innermost level's row is the WARP's group and the owner's own word is
  //: the OR of the `kWarps` of them.  The liveness pass reads these words instead of
  //: re-reading the amplitude runs itself.
  uint32_t rowOff[D];
  const uint32_t rowBytes = (uint32_t)tok_words * (uint32_t)sizeof(uint32_t);
  static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    const int o_l = (owner / BP::owner_div(l)) % BP::g(l);
    runCol[l] = BP::col_base(l) + o_l * BP::s(l);
    rowOff[l] = (uint32_t)(CR::row_prefix(l) + o_l * ((l == CL || (kTwo && l == CW)) ? kWarps : 1))
                * rowBytes;
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
  //: puts it back.  The carve level additionally carries the warp's group base.
  //: This is the ONLY place the two orders meet, and it is a pure index map.
  auto leaf_of = [&](int P) {
    int lf = 0;
    static_for<D>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      const int dig = ((P >> PM::shift(l)) & (PM::span(l) - 1)) + (l == CL ? grp * PM::kGS : 0);
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
      const int dig =
          ((P >> PMW::shift(l)) & (PMW::span(l) - 1)) + (l == PMW::kLevel ? grp * PMW::kGS : 0);
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
  constexpr int kPageCarry = kThreads - kCols;
  static_assert(kThreads > kCols && kThreads < 2 * kCols,
                "the page sweep's step must cross one row boundary, so its row "
                "cursor advances by one or two and never by a division");
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
  //: `lambda = owner*BC + local` (LATTICE-1), so the owner's atoms are
  //: `owner*(BC/16) + i`.  What the shipped kernel proves with `base_aligned` is a
  //: DEFINITION here (R0 spec section 2.6).
  const uint32_t atomBase = (uint32_t)(bh * (int)(BP::kLeaves / kAtomLeaves) + owner * kAtoms);
  auto page_off = [](uint32_t atom) { return atom * (uint32_t)(kPageElems * (int)sizeof(float)); };
  //: A PAGE HOLDS `kAtomOut` OUTER RUNS x EVERY DIGIT OF THE CARVE LEVEL, so the
  //: warp owns `kGS * kAtomOut` of its leaves in each page.  `local - 16a` is
  //: `oi*s_CL + grpBase + t`, and the accumulator element carrying it -- tile,
  //: column and the two `k` halves -- is a compile-time function of `(a, t, oi)`.
  const int grpBase = grp * kGS;
  auto page_regs = [&](auto Ac, auto Ld) {
    constexpr int a = decltype(Ac)::value;
    constexpr bool LOAD = decltype(Ld)::value;
    //: THE STATE PLANE IS WALKED UNDER THE WRITE CARVE, because that is the order the
    //: accumulator is held in.  `pp` is compile-time, so the tile and the slot within
    //: it are too; the page and the row inside it come from the leaf map and are
    //: runtime, which is exactly the freedom an OUTER carve level needs.
    static_for<kLeafW>([&](auto Pc) {
      constexpr int pp = decltype(Pc)::value;
      constexpr int tile = pp / kMmaN, nix = pp % kMmaN;
      if ((lane & 3) != nix / 2) return;
      const int lf = leafW_of(pp);
      if ((lf >> ilog2c(kAtomLeaves)) != a) return;
      const int row = lf & (kAtomLeaves - 1);
      static_for<MCH>([&](auto Mc) {
        constexpr int m = decltype(Mc)::value;
        static_for<2>([&](auto Kc) {
          constexpr int k = (nix & 1) + 2 * decltype(Kc)::value;
          float* const cell = &so[row * LdO + m * kMmaM + (lane >> 2) + 8 * (k >> 1)];
          if constexpr (LOAD)
            st[m][tile][k] = *cell;
          else
            store_f32(cell, st[m][tile][k]);
        });
      });
      if (lane == nix / 2) {
        float* const cell = &so[row * LdO + kDv];
        if constexpr (LOAD)
          mass_st[tile][nix & 1] = *cell;
        else
          store_f32(cell, mass_st[tile][nix & 1]);
      }
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
    lvCol[l] = runCol[l] + (l == CL ? grp * kGS : 0);
    lvColW[l] = runCol[l] + (l == PMW::kLevel ? grp * PMW::kGS : 0);
  });

  //: A DEAD ROW IS ZEROED AT THE NARROWEST FACTOR IT HAS.  A padding row of a partial
  //: batch has no token, and a zero coefficient contributes exactly nothing to either
  //: MMA (the produce's dead-row zeroing law).  The carve level's group run is the
  //: narrowest factor the private order has, so zeroing it covers the whole row for
  //: one or two selects and costs no arithmetic.
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
      //: THE PAIR LEVEL IS THE CARVE LEVEL under the q-major order, so ONE run --
      //: the warp's own group of `kGS` digits -- both supplies the adjacent pair and
      //: carries the dead-row zeroing, which is still the narrowest factor the
      //: private order has.
      //: EACH SIDE EXPANDS UNDER ITS OWN CARVE MAP.  The pair comes from the level
      //: lying LOWEST in that map's private order -- the carve level under q-major
      //: (the read side), the innermost non-carve level under CL-major (the write
      //: side) -- and it carries the dead-row zeroing either way.
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
    //: into the kernel.  The read side's innermost clause is the WARP'S OWN group
    //: row; the write side wants the owner's run, which is the OR of the `kWarps`
    //: of them and is one warp's work because every warp folds every write entry.
    //:
    //: ONE BALLOT WORD PER LANE.  A warp's whole window is `LW` words, so a row is
    //: ONE `LDG.32` and every warp can afford to hold BOTH sides' owner-grain
    //: ballots.  That is what makes the schedule barrier-free: the lists, the
    //: counts and the whole-window skip are all warp-private computations of
    //: CTA-identical inputs, so the four warps agree by construction and reach the
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
    //: TIER 3, THE PER-CTA PER-LEVEL CLAUSE -- and it is PRUNED BY THE DECLARATION.
    //: A side conjoins a level's row only where that side is declared sparse there;
    //: a dense side's row is all-ones and ANDing it is a load for nothing.  The
    //: side's OWN carve level is excluded because its tier-2 group row carries it.
    //: At ALT `D = 2` that empties both clauses outright: the read side is sparse
    //: only at `CL` and the write side only at `CW`.
    auto clause_of = [&](const uint32_t* rowsS, auto Side) {
      constexpr uint32_t SIDE = decltype(Side)::value;
      constexpr int own = (SIDE == kSideRead) ? CL : (kTwo ? CW : CL);
      uint32_t m = valid;
      static_for<D>([&](auto Lc) {
        constexpr int l = decltype(Lc)::value;
        if constexpr (l != own && CR::row_g(l) > 0 && mode_declares(MO, l, SIDE))
          m &= *reinterpret_cast<const uint32_t*>(gather_ptr(rowsS, rowOff[l] + wordB));
      });
      return m;
    };
    //: TIER 2, THE PER-CONSUMER DOWNSAMPLE: the warp's own multi-digit OR row at a
    //: carve level, one gather.  `LV` is the side's carve level -- `CL` for reads,
    //: `CW` for writes once the table is refined at both.
    auto group_row = [&](const uint32_t* rowsS, int lv, int g) {
      return *reinterpret_cast<const uint32_t*>(
          gather_ptr(rowsS, rowOff[lv] + (uint32_t)g * rowBytes + wordB));
    };

    const uint32_t clauseR = clause_of(rowsR, std::integral_constant<uint32_t, kSideRead>{});
    const uint32_t clauseW = clause_of(rowsW, std::integral_constant<uint32_t, kSideWrite>{});
    //: the read side wants its OWN group row (tier 2) and the owner's run (the OR of
    //: the four, for the whole-window skip).  The write side wants the owner's run
    //: for the SHARED VALUE STREAM, and -- once the table is refined at `CW` -- its
    //: own group row too, which is what makes an S-write list possible.
    uint32_t orR = 0u, orW = 0u, mineR = 0u, mineW = 0u;
#pragma unroll
    for (int g = 0; g < kWarps; ++g) {
      const uint32_t r = group_row(rowsR, CL, g);
      orR |= r;
      mineR = (g == grp) ? r : mineR;
      const uint32_t w = group_row(rowsW, kTwo ? CW : CL, g);
      orW |= w;
      mineW = (g == grp) ? w : mineW;
    }
    if constexpr (!kTwo) mineW = orW;
    const uint32_t liveW = clauseW & orW;
    //: THE WHOLE-WINDOW SKIP, at (owner, window) grain, per side -- the block-bitmap
    //: mechanism's harvest (study section 2.1).  It is CTA-uniform WITHOUT a
    //: barrier: the two owner-grain ballots are the same words in every warp.
    if (lane_ballot(((clauseR & orR) | (clauseW & orW)) != 0u) == 0u) continue;

    uint16_t* const rlist = slist + warp * kListStride;
    const int nr = walk(clauseR & mineR, rlist, 0xFFFFFFFFu, 0u);
    __syncwarp();
    //: THE WRITE SIDE KEEPS NO LIST.  Its schedule is this mask -- the warp's own
    //: tier-2 downsample met with the CTA clause -- and the fold's cursor
    //: decompresses it one ballot word at a time into a small ring of ids.
    const uint32_t mineWm = clauseW & mineW;
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
    //: and the fan-in are the warp's own, so the four warps run to the window line
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
    __nv_bfloat16* const svW = sv + warp * CP * LdV;
    const uint32_t svLw = smem_addr(svW) + (uint32_t)((ldRowW * LdV + ldColW) * 2);
    const uint32_t svAw = smem_addr(svW) + (uint32_t)((rT * LdV + cT * 8) * 2);

    //: THE CURSOR IS THE DECOMPRESSOR (doctrine 7).  One ballot word a step:
    //: broadcast it, ballot the lane's own bit, rank it inside the word, append.  No
    //: cross-warp prefix and nothing window-sized -- the ids live only in a ring of
    //: `kIdRing`, which is the pipeline's depth and not the window's length.  The
    //: earlier ballot-cursor refutation was SHARED-stream-scoped: it failed because
    //: rank-indexed consumers needed a global rank.  Here the warp's stream order IS
    //: the cursor's order and every consumer is affine in it.
    int head = 0, tail = 0, wi = 0;
    auto refill = [&]() {
      while (tail - head <= kIdRing - kBallotLanes && wi < LW) {
        const uint32_t w = lane_broadcast(mineWm, wi);
        const bool has = ((w >> lane) & 1u) != 0u;
        const uint32_t bal = lane_ballot(has);
        if (has)
          idring[(tail + bit_count(bal & ((1u << lane) - 1u))) & (kIdRing - 1)] =
              (uint16_t)(wi * kBallotLanes + lane);
        tail += bit_count(bal);
        ++wi;
      }
      __syncwarp();
    };
    auto stage_tile_w = [&](int h, int cnt) {
#pragma unroll
      for (int p = 0; p < kPassW; ++p) {
        const int r = ldRowW + p * kRowsW;
        const int t = idring[(h + (r < cnt ? r : (cnt > 0 ? cnt - 1 : 0))) & (kIdRing - 1)];
        stage_tile(svLw + (uint32_t)(p * kRowsW * LdV * 2),
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
      stage_tile_w(head, cnt);
      stage_commit();
      produce_tile(panW0, head, cnt);
      __syncwarp();
    }
#pragma unroll 1
    for (int t = 0; cnt > 0; ++t) {
      stage_wait<0>();
      //: DRAIN FIRST: the tile is in registers before the next fill is issued into
      //: the same buffer, which is the whole of the single-slot WAR window.
      uint32_t a[MCH][4];
#pragma unroll
      for (int m = 0; m < MCH; ++m) load_frag_t(a[m], svAw + (uint32_t)(m * kMmaM * 2));
      head += cnt;
      refill();
      const int nxt = tail - head < CP ? tail - head : CP;
      if (nxt > 0) {
        stage_tile_w(head, nxt);
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
    stage_wait<0>();
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
