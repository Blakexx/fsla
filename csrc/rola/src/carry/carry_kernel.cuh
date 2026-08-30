// csrc/rola/src/carry/carry_kernel.cuh -- K31 RUNG R1: THE CARRY KERNEL'S CARVED BODY.
//
// WHAT IT COMPUTES, the INTER (carried-state) term only:
//     O^T[:, t] += S^T[:, box] . R~^T[box, t]      readout, vs state-at-window-start
//     S^T[:, box] += V~^T[:, t] . W~[t, box]       fold, of the window's own writes
//
// `k_tok` (a token's nonzero digits per level) appears NOWHERE in this file, and that
// is the point: the schedule is computed, never branched on.  Since K50 the mode word
// and the digit widths are likewise absent -- they arrive as `CarryGeomRT`.
// The form is M1a's (K35 M1P); the ONE ORDERING LAW is window-grain read-before-write.
// See docs/internals/carry/carry_kernel.md
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <type_traits>

#include "box.cuh"
#include "geom.cuh"
#include "facts/arch_caps.cuh"
#include "common/design.cuh"
#include "common/ops.cuh"
#include "common/structure_switch.cuh"

namespace rola::carry {

using namespace denseref::ops;
namespace design = denseref::design;

static constexpr int kPadE = 8;  // bf16 row padding, elements (16 B)
static constexpr int kPadF = 4;  // fp32 row padding, elements (16 B)
//: `d_v` IS AN ARM PARAMETER AND NOT A FILE CONSTANT (K35 S3b). -- see docs/internals/carry/carry_kernel.md#static-assert
static_assert(kAtomLeaves == kMmaK, "the page granule is one MMA k-step");

//: THE CARVE. A STREAM owns an affine RECTANGLE of the owner's digits -- a per-
//: level span vector, `prod_l S_l` leaves and ALL `kCols` channels -- and tiles
//: those leaves in a private mixed-radix order. -- see docs/internals/carry/carry_kernel.md#carrygeom

template <int D, int DV, int W, int C, int KWARPS, int WPS>
struct CarryGeom {
  //: THE SHAPE IS THE REGISTER PLAN AND NOTHING ELSE (K50's axis law).  `BC` is the
  //: leaves one owner holds and it is DERIVED from the warp count by the register
  //: law; the addressing that used to reach this struct through `B`, `K`, `M` and
  //: `MO` now arrives as `CarryGeomRT`, once per launch. -- see docs/internals/carry/carry_kernel.md#carrygeom
  static_assert(WPS == 1,
                "M1P builds ONE warp per stream; the column split (`WPS > 1`) returns with "
                "the containers, as its own stage");
  static constexpr int kWarps = KWARPS;
  static constexpr int kStreams = KWARPS;
  static constexpr int kThreads = kWarps * 32;
  static constexpr int BC = leaves_per_warp(DV) * KWARPS;
  //: THE VALUE WIDTH, AS A DECLARATION.  `kDv` channels plus the MASS column. -- see docs/internals/carry/carry_kernel.md#kdv
  static constexpr int kDv = DV;
  static constexpr int kCols = DV + 1;
  static_assert(DV == 64,
                "M1a's readout stages exactly `kDv` channels into the batch's `C` panel rows; "
                "the `d_v` dial returns as its own stage and is never a runtime test");
  //: the stream's leaves, and the private panel row that indexes them. -- see docs/internals/carry/carry_kernel.md#kleafw
  static constexpr int kLeafR = BC / kStreams;
  static constexpr int kLeafW = BC / (kStreams / WPS);
  static_assert(kLeafW == kLeafR && kLeafR == leaves_per_warp(DV),
                "at one stream count per side both carves partition the same box into the "
                "same number of leaves, so one panel stride serves both phases");
  static constexpr int LdV = kDv + kPadE;
  static constexpr int LdS = kLeafW + kPadE;
  static constexpr int LdO = kDv + kPadF;
  //: THE EXCHANGE'S ROW: one leaf's channels and its MASS. -- see docs/internals/carry/carry_kernel.md#ldx
  //: THE GENERATED SET OF SUB-BOX SHAPES, and R9's layers 1 and 2 over it. -- see docs/internals/carry/carry_kernel.md#warp-box-set
  static constexpr int kNA = sub_boxes(D, BC, kStreams).count;
  static_assert(kNA >= 1 && kNA <= kGeomMaxSubBoxes,
                "the carve's shape set is generated from the box and the warp count; an "
                "arm whose set overflows the block's bound does not build");
  static constexpr bool sb_sound() {
    const SubBoxSet set = sub_boxes(D, BC, kStreams);
    if (set.count < 1) return false;
    for (int a = 0; a < set.count; ++a) {
      int tot = 0;
      for (int l = 0; l < D; ++l) {
        if (set.bits[a][l] < 0 || set.bits[a][l] > owner_span_bits(D, BC, l)) return false;
        tot += set.bits[a][l];
      }
      if ((1 << tot) != kLeafW || set.bits[a][D - 1] < 1) return false;
    }
    return true;
  }
  static_assert(sb_sound(),
                "every generated warp sub-box holds leaves-per-warp leaves inside the owner "
                "box, with the canonical innermost level non-empty");
  //: THE ARCH LAW, ENFORCED WHERE IT IS EVALUABLE. The shape set is a property of the
  //: SM box and `warps_per_sm`; a CTA that IS the SM (R15) derives exactly that set, and
  //: this asserts it rather than trusting the two expressions to agree. Below that warp
  //: count the two sides' contiguous subsets are DIFFERENT leaf sets, so the CTA box is
  //: the box the law is run on. -- see docs/internals/carry/geom.md#warp-box-set
  static constexpr bool sb_is_sm_grain = (KWARPS == warps_per_sm);
  static constexpr bool sb_matches_arch_law() {
    if constexpr (!sb_is_sm_grain) return true;
    const SubBoxSet arch = sub_boxes(D, leaves_per_sm(DV), warps_per_sm);
    const SubBoxSet mine = sub_boxes(D, BC, kStreams);
    if (arch.count != mine.count) return false;
    for (int a = 0; a < arch.count; ++a)
      for (int l = 0; l < D; ++l)
        if (arch.bits[a][l] != mine.bits[a][l]) return false;
    return true;
  }
  static_assert(sb_matches_arch_law(),
                "at one CTA per SM the arm's shape set must BE the arch law's set "
                "f(D, leaves_per_sm, warps_per_sm); kWarps is a launch subset, not a shape");
  static constexpr int LdX = kDv + kPadE;
  //: THE TRANSIT IS UNCONDITIONAL (K50, "kTied IS DELETED WITH THE MO MOVE"): one
  //: transit, always; a tie is a transit whose row map is the identity, reported as
  //: a census FACT and never as a second body.
  static constexpr int kXchBytes = BC * LdX * (int)sizeof(__nv_bfloat16);

  //: THE TWO PANELS ARE A TWO-SLOT RING, AND THEY COST NOTHING EXTRA. -- see docs/internals/carry/carry_kernel.md#panel-bytes
  static constexpr int panel_bytes = 2 * kWarps * C * LdS * (int)sizeof(__nv_bfloat16);
  //: the readout's staging block, one `kMmaN`-token region per warp. -- see docs/internals/carry/carry_kernel.md#outbytes
  static constexpr int out_bytes = C * LdO * (int)sizeof(float);
  static_assert(kWarps * kMmaN == C && C * LdO >= kAtomLeaves * LdO,
                "the staging block is one token tile of the batch and holds a whole page");
  //: `kWarps` READ lists, one per stream, then ONE write list. -- see docs/internals/carry/carry_kernel.md#klists
  static constexpr int kLists = kWarps + 1;
  static constexpr int kListStride = ((W + 7) / 8) * 8;
  static constexpr int list_bytes = kLists * kListStride * (int)sizeof(uint16_t);
  //: THE WINDOW'S BALLOT WORDS, one per lane of the walk. -- see docs/internals/carry/carry_kernel.md#klw
  static constexpr int kLW = W / kBallotLanes;
  //: THE V RING'S DEPTH IS DERIVED FROM THE SMEM BUDGET AND THE REGISTER FLOOR. -- see docs/internals/carry/carry_kernel.md#kregfloor
  static constexpr int kRegFloor = 220;
  static constexpr int slot_bytes = C * LdV * (int)sizeof(__nv_bfloat16);
  static constexpr int fixed_bytes = panel_bytes + out_bytes + list_bytes;
  static constexpr int kStages = design::stages_for(slot_bytes, fixed_bytes, kThreads, kRegFloor);
  static constexpr size_t bytes = (size_t)fixed_bytes + (size_t)kStages * slot_bytes;
  static constexpr int cta_target = design::cta_target((int)bytes, kThreads, kRegFloor);
  static_assert(design::smem_fits(bytes), "this arm does not fit the SMEM maximum");
  //: THE TRANSIT COSTS NO ALLOCATION: it STRADDLES the tenants dead at both of its
  //: moments.  NO SMEM TENANT IS SPAN-DEPENDENT -- every one of them is a function of
  //: `BC`, `C`, `DV` and `W` alone -- so the launch-sized layout K50 asks for is the
  //: identity on this kernel and the host passes these same offsets. -- see docs/internals/carry/carry_kernel.md#straddle-bytes
  static constexpr int straddle_bytes = panel_bytes / 2 + out_bytes;
  static_assert(kXchBytes <= straddle_bytes,
                "the transit's region is the SECOND panel and the readout staging block: the "
                "two tenants dead at BOTH of its moments, and disjoint from the V ring and "
                "the first panel, which is what leaves the work that covers each leg free to "
                "run against it; an arm whose transit outgrew them would have to allocate, "
                "and none does");
};

//: THE OWNER-GRAIN PRODUCE MAP, which the REVERSE pass takes (`backward_kernel.cuh`).
//: The backward family is a separate arm list with no mode word, so its box stays the
//: compile-time `BoxPlan`; K50's runtime addressing reaches it with its own card. -- see docs/internals/carry/carry_kernel.md#producemap
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

//: THE CHANNEL-SPLIT BODY'S SLOT SHAPE: how many slots one thread mints, from the
//: box size alone.  Its per-level digit fields come from the addressing block at
//: runtime, exactly as the carved body's do. -- see docs/internals/carry/carry_kernel.md#splitslots
template <int BC_, int C>
struct SplitSlots {
  static constexpr int kRowThr = kThreads / C;
  static constexpr int SPT = BC_ / kRowThr;
  static constexpr int kJBits = ilog2c(SPT);
};

//: a contiguous run of `n` bf16, at the widest access that covers it. -- see docs/internals/carry/carry_kernel.md#load-run
template <int N>
__device__ __forceinline__ void load_run(unsigned short (&dst)[N], const void* p) {
  static_assert(N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32,
                "an admissible level run is 1, 2, 4, 8, 16 or 32 wide");
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

//: THE RUN THAT NEEDS NO RE-TILING: when a level's stream span IS the whole slot
//: run, the private order's map is the identity and the token's amplitudes land in
//: the accumulator pairs directly -- one wide load, no second array, and therefore
//: no register above the narrower cases. -- see docs/internals/carry/carry_kernel.md#load-run-u32
template <int N>
__device__ __forceinline__ void load_run_u32(uint32_t (&dst)[N], const void* p) {
  static_assert(N % 4 == 0, "the identity run is a whole number of sixteen-byte granules");
  constexpr int kGranules = N / 4;
#pragma unroll
  for (int i = 0; i < kGranules; ++i)
    *reinterpret_cast<uint4*>(dst + i * 4) = load_vec16(reinterpret_cast<const char*>(p) + i * 16);
}

//: THE ACCUMULATOR'S SEED: the order's INNERMOST level, whose shift is zero by
//: definition of rank.  A run as wide as the whole slot run needs no re-tiling -- the
//: private order's map is the identity there and the amplitudes land in the accumulator
//: pairs directly, one wide load and no second array, so the widest case costs no
//: register above the narrower ones. -- see docs/internals/carry/carry_kernel.md#field-seed
template <int SPT, bool WRITE, int K>
__device__ __forceinline__ void field_seed(uint32_t (&acc)[SPT / 2], const void* src, bool have) {
  constexpr int SV = 1 << K;
  if constexpr (SV >= SPT) {
    load_run_u32<SPT / 2>(acc, src);
    if constexpr (!WRITE)
      static_for<SPT / 2>([&](auto Jc) {
        constexpr int pp = decltype(Jc)::value;
        acc[pp] = have ? acc[pp] : 0u;
      });
  } else {
    unsigned short run[SV];
    load_run<SV>(run, src);
    if constexpr (!WRITE)
      static_for<SV>([&](auto Ic) {
        constexpr int i = decltype(Ic)::value;
        run[i] = have ? run[i] : (unsigned short)0;
      });
    static_for<SPT / 2>([&](auto Jc) {
      constexpr int j = 2 * decltype(Jc)::value;
      acc[j / 2] = ((uint32_t)run[(j + 1) & (SV - 1)] << 16) | (uint32_t)run[j & (SV - 1)];
    });
  }
}

//: ONE MORE LEVEL'S FACTOR, folded into the accumulator at its own field.
template <int SPT, int K, int SH>
__device__ __forceinline__ void field_fold(uint32_t (&acc)[SPT / 2], const void* src) {
  constexpr int SV = 1 << K;
  static_assert(SH >= 1,
                "only the canonical INNERMOST level sits at bit offset zero, and that level "
                "SEEDS the accumulator; a folded level's slot pair always shares one digit");
  unsigned short run[SV];
  load_run<SV>(run, src);
  static_for<SPT / 2>([&](auto Jc) {
    constexpr int pp = decltype(Jc)::value;
    acc[pp] = mul_bf16x2(acc[pp], splat_bf16x2_bits(run[(pp >> (SH - 1)) & (SV - 1)]));
  });
}

//: THE SECOND LAUNCH BOUND: the CTA count this arm can actually reach, taken from the -- see docs/internals/carry/carry_kernel.md#near-line-215
template <int D, int DV, int W, int C, int KWARPS, int WPS>
constexpr int kCarryCtaTarget = CarryGeom<D, DV, W, C, KWARPS, WPS>::cta_target;

// ---------------------------------------------------------------- the kernel

template <int D, int DV, int W, int C, int KWARPS, int WPS>
__global__ __launch_bounds__((CarryGeom<D, DV, W, C, KWARPS, WPS>::kThreads),
                             (kCarryCtaTarget<D, DV, W, C, KWARPS, WPS>)) void
carry_kernel(const __nv_bfloat16* __restrict__ pread, const __nv_bfloat16* __restrict__ pwrite,
             const __nv_bfloat16* __restrict__ gwrite, const __nv_bfloat16* __restrict__ v,
             const uint32_t* __restrict__ rows, float* __restrict__ o, float* __restrict__ den,
             int L, int tok_words, int v_row_bytes, const float* __restrict__ s_in,
             float* __restrict__ s_out, const int32_t* __restrict__ page_tbl,
             const uint8_t* __restrict__ atom_bits, const __grid_constant__ CarryGeomRT geo) {
  using G = CarryGeom<D, DV, W, C, KWARPS, WPS>;
  constexpr int kDv = DV;
  constexpr int kCols = DV + 1;
  constexpr int kWarps = G::kWarps;
  constexpr int kThreads = G::kThreads;

  constexpr int BC = G::BC;
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
  const __nv_bfloat16* const pR = pin_address(pread + (long)bh * L * geo.wtot);
  const __nv_bfloat16* const pW = pin_address(pwrite + (long)bh * L * geo.wtot);
  const __nv_bfloat16* const gW = pin_address(gwrite + (long)bh * L);
  float* const oPlane = pin_address(o + (long)bh * L * kDv);
  float* const denPlane = pin_address(den + (long)bh * L);

  //: THE STREAM ROTATION.  Two CTAs co-resident on an SM are one grid wave apart and -- see docs/internals/carry/carry_kernel.md#strm
  const int strm = (warp + ((owner + (owner >> 5)) & (kWarps - 1))) & (kWarps - 1);

  // ---- the owner's digit runs, one per level -------------------------------
  //: `o_l` is a bit-field extract of the owner index: the lattice's owner -- see docs/internals/carry/carry_kernel.md#near-line-296
  int runCol[D];
  //: AND THE STREAM'S OWN DIGIT OFFSET INSIDE THAT RUN, per level AND PER SIDE: the -- see docs/internals/carry/carry_kernel.md#note-l344
  int offW[D], offR[D];
  //: AND THE BALLOT ROW BASE, per level and per side: the SAME `o_l`, addressing the -- see docs/internals/carry/carry_kernel.md#rowbytes
  uint32_t rowOffR[D], rowOffW[D];
  const uint32_t rowBytes = (uint32_t)tok_words * (uint32_t)sizeof(uint32_t);
  //: THE TWO BASE WORDS, one per side. -- see docs/internals/carry/carry_kernel.md#transit
  int baseW = 0, baseR = 0;
  static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    const int o_l = (owner >> geo.owner_div_bits[l]) & (geo.grid[l] - 1);
    runCol[l] = geo.col_base[l] + (o_l << owner_span_bits(D, BC, l));
    offW[l] = ((strm >> geo.w.grid_div_bits[l]) & (geo.w.grid[l] - 1)) << geo.w.sub_bits[l];
    offR[l] = ((strm >> geo.r.grid_div_bits[l]) & (geo.r.grid[l] - 1)) << geo.r.sub_bits[l];
    baseW += offW[l] << owner_run_shift(D, BC, l);
    baseR += offR[l] << owner_run_shift(D, BC, l);
    rowOffR[l] = (uint32_t)(geo.r.row_prefix[l] + (o_l << geo.r.grid_bits[l])) * rowBytes;
    rowOffW[l] = (uint32_t)(geo.w.row_prefix[l] + (o_l << geo.w.grid_bits[l])) * rowBytes;
  });
  //: ONE 64-BIT BASE PER SIDE, and 32-bit byte displacements after it: the READ -- see docs/internals/carry/carry_kernel.md#pin-address
  const int rowsBoth = geo.r.rows + geo.w.rows;
  const uint32_t* const rowsR = pin_address(rows + (long)bh * (long)rowsBoth * (long)tok_words);
  const uint32_t* const rowsW =
      pin_address(rows + ((long)bh * (long)rowsBoth + (long)geo.r.rows) * (long)tok_words);

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

  //: THE TRANSIT'S ADDRESSING IS LOOP-INVARIANT, so it is derived ONCE.  Under the
  //: canonical slot layout a leaf keeps its SLOT and changes only its WARP, so the row
  //: map is this ONE base word and the identity on the slot bits, and the region needs
  //: no swizzle: `LdX / 2 = 4 (mod 8) x odd`, so eight rows an aligned run apart sit in
  //: eight different four-bank blocks. -- see docs/internals/carry/carry_kernel.md#transit
  __nv_bfloat16* const xch = sw;
  static_assert(((LdX / 2) & 3) == 0 && ((LdX / 2) >> 2) % 2 == 1,
                "the transit region separates its rows by the ROW STRIDE: half a row must "
                "be four times an odd number of words, or an aligned eight of rows shares "
                "a four-bank block and the pull conflicts");
  const int retG = lane >> 3;

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
      const int dig = ((P >> geo.w.sub_shift[l]) & ((1 << geo.w.sub_bits[l]) - 1)) + offW[l];
      lf += dig << owner_run_shift(D, BC, l);
    });
    return lf;
  };

  // ---- the state plane's page sweep, at CANONICAL atom ids -----------------
  constexpr int kAtoms = BC / kAtomLeaves;
  constexpr int kPageElems = kAtomLeaves * kCols;
  //: THE SWEEP'S STEP, AS A CARRY.  A step of `kThreads` elements advances the row -- see docs/internals/carry/carry_kernel.md#krowstep
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
  //: THE PLANE IS ADDRESSED BY CANONICAL ATOM ID (P80). -- see docs/internals/carry/carry_kernel.md#atombh
  const uint32_t atomBH = (uint32_t)bh * (uint32_t)(geo.leaves / kAtomLeaves);
  //: THE CANONICAL ATOM of this owner's atom `a`, at the block's own radices. -- see docs/internals/carry/carry_kernel.md#canonatom
  auto canon_atom = [&](int a) {
    return (uint32_t)geom_canon_leaf_t<D, BC>(geo, owner, a * kAtomLeaves) >> ilog2c(kAtomLeaves);
  };
  //: THE ACTIVITY OF THIS OWNER'S ATOMS, READ ONCE AT THE PROLOGUE (P80b): one -- see docs/internals/carry/carry_kernel.md#activity
  static_assert(kAtoms <= 32, "one bit per atom per mask: an owner holds at most 32 atoms");
  uint32_t actRead = 0u, actWrite = 0u;
  static_for<kAtoms>([&](auto Ac) {
    constexpr int a = decltype(Ac)::value;
    const uint8_t bits = atom_activity(atom_bits, atomBH + canon_atom(a));
    actRead |= (uint32_t)((bits & kAtomRead) != 0u) << a;
    actWrite |= (uint32_t)((bits & kAtomWritten) != 0u) << a;
  });
  //: THE SLOT, NOT THE ATOM, ADDRESSES THE PLANE (`ops.cuh`'s `page_slot`, -- see docs/internals/carry/carry_kernel.md#kpagebytes
  constexpr uint32_t kPageBytes = (uint32_t)(kPageElems * (int)sizeof(float));
  auto page_base = [](int32_t slot) { return (uint32_t)slot * kPageBytes; };
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
      const int32_t slot = page_slot(page_tbl, atomBH + canon_atom(a));
      if (slot < 0 || (((actRead | actWrite) >> a) & 1u) == 0u) return;
      page_sweep(page_base(slot), [&](int sidx, uint32_t off_) {
        so[sidx] = *reinterpret_cast<const float*>(gather_ptr(plane, off_));
      });
      rendezvous();
      page_regs(Ac, std::true_type{});
      rendezvous();
    });
  }

  // ---- the produce's per-thread map ---------------------------------------
  //: ONE LANE MINTS ONE TOKEN'S WHOLE ROW: the batch tile is the warp width, so a -- see docs/internals/carry/carry_kernel.md#spt
  constexpr int SPT = kLeafW;
  static_assert(SPT == leaves_per_warp(DV),
                "the warp's slot run IS the register shape: the sub-box's spans and the slot "
                "layout are compile-time f(D, DV), and a different leaves-per-warp is a "
                "different sub-box, not a wider case");
  static_assert(C == kBallotLanes,
                "a warp's panel row is ONE lane's: the batch tile is the warp width, so a "
                "thread's slot run is the stream's whole box");
  const int prow = lane;  // this lane's token row of the batch
  //: THE PRODUCE'S VIEW IS BY LEVEL, IN THE CANONICAL SLOT LAYOUT: the sub-box's span
  //: and shift are IMMEDIATES on both sides, and the side's order reaches the produce
  //: only as the stream's column offset. -- see docs/internals/carry/carry_kernel.md#byrank
  auto produce = [&](auto Wt, const __nv_bfloat16* base, __nv_bfloat16* dst, uint32_t tokOff,
                     uint32_t gain, bool have) {
    constexpr bool WRITE = decltype(Wt)::value;
    //: EACH PHASE MINTS FROM ITS OWN CARVE'S OFFSETS, in the ONE canonical slot layout. -- see docs/internals/carry/carry_kernel.md#pm
    const SideGeom& sg = WRITE ? geo.w : geo.r;
    //: THE SIDE'S OWN MEMBER, SELECTED AT LAUNCH FROM THE MODE WORD. -- see docs/internals/carry/carry_kernel.md#per-side-produce
    rola::uniform_switch<G::kNA>(sg.assign, [&](auto Ac) {
      constexpr int A = decltype(Ac)::value;
      //: THE RUN'S COLUMN, PER LEVEL. -- see docs/internals/carry/carry_kernel.md#byrank
      auto col_at = [&](int l, int sb) {
        return runCol[l] + (((strm >> sg.grid_div_bits[l]) & (sg.grid[l] - 1)) << sb);
      };
      //: THE SLOT ACCUMULATORS ARE THE ONLY ARRAY THE PRODUCE CARRIES. -- see docs/internals/carry/carry_kernel.md#near-line-497
      uint32_t acc[SPT / 2];
      {
        //: the canonical INNERMOST level seeds the accumulator: its shift is zero.
        constexpr int l = D - 1, K = sb_span_bits(D, BC, G::kStreams, A, l);
        const void* const src = gather_ptr(base, tokOff + (uint32_t)(col_at(l, K) * 2));
        field_seed<SPT, WRITE, K>(acc, src, have);
      }
      static_for<D - 1>([&](auto Lc) {
        constexpr int l = decltype(Lc)::value;
        constexpr int K = sb_span_bits(D, BC, G::kStreams, A, l),
                      SH = sb_shift_of(D, BC, G::kStreams, A, l);
        const void* const src = gather_ptr(base, tokOff + (uint32_t)(col_at(l, K) * 2));
        field_fold<SPT, K, SH>(acc, src);
      });
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
    });
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
    const uint32_t winP = (uint32_t)tok0 * (uint32_t)(geo.wtot * 2);
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
      produce(Wt, base, dst, winP + (uint32_t)tok * (uint32_t)(geo.wtot * 2), gain, have);
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
      const SideGeom& sg = decltype(Wt)::value ? geo.w : geo.r;
      uint32_t m = valid;
      static_for<D>([&](auto Lc) {
        constexpr int l = decltype(Lc)::value;
        //: A CTA-UNIFORM BRANCH ON STRUCTURE: whether this side carves the level, and
        //: whether the level votes rows at all.  Both are geometry, never a threshold.
        if (((sg.carves >> l) & 1) == 0 && sg.row_g[l] > 0)
          m &= *reinterpret_cast<const uint32_t*>(gather_ptr(rowsS, rowOffS[l] + wordB));
      });
      return m;
    };
    //: AND THE CARVED CLAUSE OF ONE STREAM, on that side's OWN carve: the AND over -- see docs/internals/carry/carry_kernel.md#streamrow
    auto stream_row = [&](auto Wt, const uint32_t* rowsS, const uint32_t* rowOffS, int s) {
      const SideGeom& sg = decltype(Wt)::value ? geo.w : geo.r;
      uint32_t m = 0xFFFFFFFFu;
      static_for<D>([&](auto Lc) {
        constexpr int l = decltype(Lc)::value;
        if (((sg.carves >> l) & 1) != 0) {
          const uint32_t c = (uint32_t)((s >> sg.grid_div_bits[l]) & (sg.grid[l] - 1));
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
    //: THE WAR EDGE against the PREVIOUS window's fold, which still reads the bytes -- see docs/internals/carry/carry_kernel.md#war
    rendezvous();
    //: THE OUT LEG'S COVER: read batch zero's factor gathers touch neither `st` nor -- see docs/internals/carry/carry_kernel.md#cover
    if (nbr > 0) produce_batch(std::false_type{}, pR, panR, rlist, 0, nr);
    rola::uniform_switch<G::kNA>(geo.w.assign, [&](auto Ac) {
      constexpr int A = decltype(Ac)::value;
      //: THE SWIZZLE IS LINEAR, so the lane's base takes it ONCE and every fragment's row
      //: term stays a compile-time constant. -- see docs/internals/carry/geom.md#row-swizzle
      const int lrow = sb_swizzle(baseW ^ sb_scatter_of(D, BC, G::kStreams, A, 2 * (lane & 3)));
      const uint32_t xchB = smem_addr(xch);
      const uint32_t pubCol = (uint32_t)((lane >> 2) * 2);
      static_for<kNL>([&](auto Jc) {
        constexpr int j = decltype(Jc)::value;
        static_for<4>([&](auto Ec) {
          constexpr int e = decltype(Ec)::value;
          constexpr int dl = sb_swizzle(sb_scatter_of(D, BC, G::kStreams, A, j * kMmaN + (e & 1)));
          static_for<MCH>([&](auto Mc) {
            constexpr int m = decltype(Mc)::value;
            constexpr int gr = 2 * m + (e >> 1);
            store_shared_u16(
                xchB + (uint32_t)((lrow ^ dl) * LdX * 2) + pubCol + (uint32_t)(gr * 16),
                hi_bf16(st[m][j][e]));
          });
        });
        if ((lane >> 2) == 0) {
          static_for<2>([&](auto Ec) {
            constexpr int e = decltype(Ec)::value;
            constexpr int dl = sb_swizzle(sb_scatter_of(D, BC, G::kStreams, A, j * kMmaN + e));
            store_shared_u16(xchB + (uint32_t)((lrow ^ dl) * LdX * 2 + kDv * 2),
                             hi_bf16(mass_st[j][e]));
          });
        }
      });
    });
    rendezvous();

    rola::uniform_switch<G::kNA>(geo.r.assign, [&](auto Ac) {
      constexpr int A = decltype(Ac)::value;
      constexpr int nx = sb_swizzle(sb_scatter_of(D, BC, G::kStreams, A, 1));
      const uint32_t xchB = smem_addr(xch);
      const int pullRow = sb_swizzle(baseR ^ sb_scatter_of(D, BC, G::kStreams, A, rT));
      const int massRow = sb_swizzle(baseR ^ sb_scatter_of(D, BC, G::kStreams, A, 2 * (lane & 3)));
      static_for<kDw>([&](auto Kc) {
        constexpr int kk = decltype(Kc)::value;
        constexpr int rk = sb_swizzle(sb_scatter_of(D, BC, G::kStreams, A, kk * kMmaK));
        static_for<MCH>([&](auto Mc) {
          constexpr int m = decltype(Mc)::value;
          load_frag_t(sa[kk][m], xchB + (uint32_t)((pullRow ^ rk) * LdX * 2)
                                     + (uint32_t)(((2 * m) | cT) * 16));
        });
        //: the mass's two pairs, a row apart, at the reader's own segment. -- see docs/internals/carry/carry_kernel.md#msw
        ma[kk][0] = 0u;
        ma[kk][1] = 0u;
        ma[kk][2] = 0u;
        ma[kk][3] = 0u;
        if ((lane >> 2) == 0) {
          constexpr int rh = sb_swizzle(sb_scatter_of(D, BC, G::kStreams, A, kk * kMmaK + 8));
          const uint32_t mc = (uint32_t)(kDv * 2);
          ma[kk][0] =
              (uint32_t)load_shared_u16(xchB + (uint32_t)((massRow ^ rk) * LdX * 2) + mc)
              | ((uint32_t)load_shared_u16(xchB + (uint32_t)((massRow ^ rk ^ nx) * LdX * 2) + mc)
                 << 16);
          ma[kk][2] =
              (uint32_t)load_shared_u16(xchB + (uint32_t)((massRow ^ rh) * LdX * 2) + mc)
              | ((uint32_t)load_shared_u16(xchB + (uint32_t)((massRow ^ rh ^ nx) * LdX * 2) + mc)
                 << 16);
        }
      });
    });
    rendezvous();

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
    //: THE STORE-BACK IS THE PULL RUN BACKWARDS: the reader writes each A fragment -- see docs/internals/carry/carry_kernel.md#store-back
    rola::uniform_switch<G::kNA>(geo.r.assign, [&](auto Ac) {
      constexpr int A = decltype(Ac)::value;
      constexpr int nx = sb_swizzle(sb_scatter_of(D, BC, G::kStreams, A, 1));
      const int mrow = sb_swizzle(baseR ^ sb_scatter_of(D, BC, G::kStreams, A, 2 * (lane & 3)));
      const uint32_t xchB = smem_addr(xch);
      const uint32_t sbCol = (uint32_t)((lane >> 2) * 2);
      static_for<kDw>([&](auto Kc) {
        constexpr int kk = decltype(Kc)::value;
        static_for<4>([&](auto Ic) {
          constexpr int i = decltype(Ic)::value;
          constexpr int rr =
              sb_swizzle(sb_scatter_of(D, BC, G::kStreams, A, kk * kMmaK + (i >> 1) * 8));
          const uint32_t a0 = xchB + (uint32_t)((mrow ^ rr) * LdX * 2) + sbCol;
          static_for<MCH>([&](auto Mc) {
            constexpr int m = decltype(Mc)::value;
            constexpr int gr = 2 * m + (i & 1);
            const uint32_t a = a0 + (uint32_t)(gr * 16);
            store_shared_u16(a, (uint16_t)(sa[kk][m][i] & 0xFFFFu));
            store_shared_u16(
                xchB + (uint32_t)((mrow ^ rr ^ nx) * LdX * 2) + sbCol + (uint32_t)(gr * 16),
                (uint16_t)(sa[kk][m][i] >> 16));
          });
        });
        if ((lane >> 2) == 0) {
          constexpr int rk = sb_swizzle(sb_scatter_of(D, BC, G::kStreams, A, kk * kMmaK));
          constexpr int rh = sb_swizzle(sb_scatter_of(D, BC, G::kStreams, A, kk * kMmaK + 8));
          const uint32_t mc = (uint32_t)(kDv * 2);
          store_shared_u16(xchB + (uint32_t)((mrow ^ rk) * LdX * 2) + mc,
                           (uint16_t)(ma[kk][0] & 0xFFFFu));
          store_shared_u16(xchB + (uint32_t)((mrow ^ rk ^ nx) * LdX * 2) + mc,
                           (uint16_t)(ma[kk][0] >> 16));
          store_shared_u16(xchB + (uint32_t)((mrow ^ rh) * LdX * 2) + mc,
                           (uint16_t)(ma[kk][2] & 0xFFFFu));
          store_shared_u16(xchB + (uint32_t)((mrow ^ rh ^ nx) * LdX * 2) + mc,
                           (uint16_t)(ma[kk][2] >> 16));
        }
      });
    });
    rendezvous();
    //: THE WRITER PULLS IN ITS OWN LAYOUT.  Its eight slots of one n-tile are eight -- see docs/internals/carry/carry_kernel.md#writer-pull
    rola::uniform_switch<G::kNA>(geo.w.assign, [&](auto Ac) {
      constexpr int A = decltype(Ac)::value;
      const uint32_t xchB = smem_addr(xch);
      const int retRow = sb_swizzle(baseW ^ sb_scatter_of(D, BC, G::kStreams, A, lane & 7));
      const int pubRow = sb_swizzle(baseW ^ sb_scatter_of(D, BC, G::kStreams, A, 2 * (lane & 3)));
      static_for<kNL>([&](auto Jc) {
        constexpr int j = decltype(Jc)::value;
        constexpr int dj = sb_swizzle(sb_scatter_of(D, BC, G::kStreams, A, j * kMmaN));
        const uint32_t a0 = xchB + (uint32_t)((retRow ^ dj) * LdX * 2);
        static_for<MCH / 2>([&](auto Tc) {
          constexpr int t = decltype(Tc)::value;
          uint32_t r4[4];
          load_frag_t(r4, a0 + (uint32_t)((4 * t + retG) * 16));
          static_for<4>([&](auto Rc) {
            constexpr int r = decltype(Rc)::value;
            hi[j][4 * t + r] = r4[r];
          });
        });
        mhi[j] = 0u;
        if ((lane >> 2) == 0) {
          constexpr int dl0 = sb_swizzle(sb_scatter_of(D, BC, G::kStreams, A, j * kMmaN));
          constexpr int dl1 = sb_swizzle(sb_scatter_of(D, BC, G::kStreams, A, j * kMmaN + 1));
          const uint32_t m0 =
              (uint32_t)load_shared_u16(xchB + (uint32_t)((pubRow ^ dl0) * LdX * 2 + kDv * 2));
          const uint32_t m1 =
              (uint32_t)load_shared_u16(xchB + (uint32_t)((pubRow ^ dl1) * LdX * 2 + kDv * 2));
          mhi[j] = m0 | (m1 << 16);
        }
      });
    });
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
      const int32_t slot = page_slot(page_tbl, atomBH + canon_atom(a));
      if (slot < 0 || ((actWrite >> a) & 1u) == 0u) return;
      page_regs(Ac, std::false_type{});
      rendezvous();
      page_sweep(page_base(slot), [&](int sidx, uint32_t offb) {
        store_f32(reinterpret_cast<float*>(gather_ptr(plane, offb)), so[sidx]);
      });
      rendezvous();
    });
  }
#undef ADDR_S
#undef ADDR_V
}

}  // namespace rola::carry
