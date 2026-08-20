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
static constexpr int kCarryBuildMagic = 31010011;

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

template <int D, int B, int K, int M, int W, int C>
struct CarryGeom {
  using BP = BoxPlan<D, B, K, M>;
  static constexpr int BC = BP::kBC;
  static constexpr int LdV = kDv + kPadE;
  static constexpr int LdS = BC + kPadE;
  static constexpr int LdO = kDv + kPadF;

  //: THE TWO PANELS ARE A TWO-SLOT RING, AND THEY COST NOTHING EXTRA.  A window's
  //: readouts all precede its folds, so the write panel is dead through phase 1 and
  //: the read panel is dead through phase 2: each phase double-buffers in the
  //: other's buffer and produces one batch AHEAD, which is what puts the produce's
  //: global-load latency underneath the previous batch's MMA block.
  static constexpr int panel_bytes = 2 * C * LdS * (int)sizeof(__nv_bfloat16);
  static constexpr int out_bytes = C * LdO * (int)sizeof(float);
  static constexpr int den_bytes = kWarps * C * (int)sizeof(float);
  //: TWO packed lists per window, one per SIDE (study section 2.0, two-sided
  //: schedule): `nr` read entries then `nw` write entries, `kListStride` apart.
  static constexpr int kListStride = ((W + 7) / 8) * 8;
  static constexpr int list_bytes = 2 * kListStride * (int)sizeof(uint16_t);
  static constexpr int scan_bytes = 16 * (int)sizeof(int);
  //: THE V RING'S DEPTH IS DERIVED FROM THE SMEM BUDGET AND THE REGISTER FLOOR,
  //: exactly as the shipped `Geom::stages` is: the shallowest ring that reaches the
  //: best residency this arm can reach at any depth.  `kRegFloor` is the DECLARED
  //: floor for this family -- the measured census of the heaviest built arm,
  //: recorded in the findings doc and re-measured at every milestone.
  static constexpr int kRegFloor = 168;
  static constexpr int slot_bytes = C * LdV * (int)sizeof(__nv_bfloat16);
  static constexpr int fixed_bytes = panel_bytes + out_bytes + den_bytes + list_bytes + scan_bytes;
  static constexpr int kStages = design::stages_for(slot_bytes, fixed_bytes, kThreads, kRegFloor);
  static constexpr size_t bytes = (size_t)fixed_bytes + (size_t)kStages * slot_bytes;
  static constexpr int cta_target = design::cta_target((int)bytes, kThreads, kRegFloor);
};

//: THE PRODUCE'S THREAD MAP, derived.  A row's `kRowThr` threads split the box's
//: `BC` coefficients into `SPT` contiguous LATTICE slots each.  Because the lattice
//: index is a bit permutation (R0 spec section 2.4), a thread's slot run is a
//: sub-lattice: per level the digits it touches are a CONTIGUOUS run of the owner's
//: own run at a runtime base, and WHICH element of that run each of its slots wants
//: is a compile-time function of the unrolled slot index.  So the expansion is
//: `sum_l n_l` amplitude loads followed by `SPT` products of register values at
//: CONSTANT indices -- no dynamic indexing, no spill, no materialized gate plane.
template <int D, int B, int K, int M, int C>
struct ProduceMap {
  using BP = BoxPlan<D, B, K, M>;
  static constexpr int kRowThr = kThreads / C;
  static constexpr int SPT = BP::kBC / kRowThr;
  static constexpr int kJBits = ilog2c(SPT);
  //: level `l`'s sub-box multiplicity field lies entirely inside the thread's own
  //: slot index, or entirely above it.  Straddling would break the contiguity.
  static constexpr bool in_j(int l) { return BP::u_shift(l) + BP::a(l) <= kJBits; }
  static constexpr bool above_j(int l) { return BP::u_shift(l) >= kJBits; }
  static constexpr int n_l(int l) { return in_j(l) ? BP::s(l) : K; }
  static constexpr int max_n(int l) {
    return l == D ? 1 : (n_l(l) > max_n(l + 1) ? n_l(l) : max_n(l + 1));
  }
  static constexpr int kMaxN = max_n(0);
  //: element of level `l`'s loaded run that unrolled slot `j` wants.
  static constexpr int idx_of(int l, int j) { return BP::u_of(l, j) * K + BP::c_of(l, j); }
};

//: a contiguous run of `n` bf16, at the widest access that covers it.  The runs are
//: naturally aligned by construction (an owner's run starts at a multiple of `s_l`
//: and a sub-box's at a multiple of `K`), so this is one, two or four accesses.
template <int n>
__device__ __forceinline__ void load_run(unsigned short (&dst)[n], const void* p) {
  static_assert(n == 4 || n == 8 || n == 16, "an admissible (k, m) run is 4, 8 or 16 wide");
  if constexpr (n == 4) {
    *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(p);
  } else {
#pragma unroll
    for (int i = 0; i < n / 8; ++i)
      *reinterpret_cast<uint4*>(dst + i * 8) =
          load_vec16(reinterpret_cast<const char*>(p) + i * 16);
  }
}

//: THE SECOND LAUNCH BOUND: the CTA count this arm can actually reach, taken from
//: the same geometry the ring depth is.  A ring sized for a residency the launch
//: bound does not ask for is a ring sized for nothing.
template <int D, int B, int K, int M, int W, int C>
constexpr int kCarryCtaTarget = CarryGeom<D, B, K, M, W, C>::cta_target;

// ---------------------------------------------------------------- the kernel

template <int D, int B, int K, int M, int W, int C, bool STATE>
__global__ __launch_bounds__(kThreads, (kCarryCtaTarget<D, B, K, M, W, C>)) void carry_kernel(
    const __nv_bfloat16* __restrict__ pread, const __nv_bfloat16* __restrict__ pwrite,
    const __nv_bfloat16* __restrict__ gwrite, const __nv_bfloat16* __restrict__ v,
    const uint32_t* __restrict__ rows, float* __restrict__ o, float* __restrict__ den, int L,
    int tok_words, int v_row_bytes, const float* __restrict__ s_in, float* __restrict__ s_out) {
  using BP = BoxPlan<D, B, K, M>;
  using G = CarryGeom<D, B, K, M, W, C>;
  using PM = ProduceMap<D, B, K, M, C>;

  constexpr int BC = BP::kBC;
  constexpr int LdV = G::LdV, LdS = G::LdS, LdO = G::LdO;
  constexpr int MTOK = C / kMmaM;  // token m-tiles of a batch
  constexpr int NTOK = C / kMmaN;  // token n-tiles of a batch
  //: every warp holds the WHOLE box and a slice of the channels, so the state's
  //: leaf tiling is the box's and is not carved.
  constexpr int NDV = BC / kMmaN;  // the state's n-tiles (leaves)
  constexpr int KDw = BC / kMmaK;  // its k-tiles
  //: THE MASS CHANNEL'S CARVE.  One channel cannot be split along the channel
  //: axis, so it is split along the axis the mma leaves free -- the leaves.
  constexpr int kMassUnits = KDw < kWarps ? KDw : kWarps;
  constexpr int kMassKD = KDw / kMassUnits;
  constexpr int kMassTiles = 2 * kMassKD;

  static_assert(C % kMmaM == 0 && kThreads % C == 0, "the batch tile must divide");
  static_assert(BC % kMmaK == 0, "the box is a whole number of k-tiles");
  static_assert(BC % kAtomLeaves == 0, "K2: an owner holds whole page atoms");
  static_assert(design::smem_fits(G::bytes), "this arm does not fit the SMEM maximum");
  static_assert(PM::kRowThr * PM::SPT == BC, "the row's threads cover the box exactly");
  static_assert(PM::SPT % 2 == 0, "a thread publishes whole bf16 pairs");
  static_assert(LdS % 8 == 0 && LdV % 8 == 0, "an ldmatrix row address must be 16-byte aligned");
  static_assert(BP::kKappa >= 1, "the innermost level supplies the adjacent pair");
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
  __nv_bfloat16* const sw = sr + C * LdS;
  float* const so = reinterpret_cast<float*>(sw + C * LdS);
  float* const sden = so + C * LdO;
  constexpr int kListStride = G::kListStride;
  uint16_t* const slist = reinterpret_cast<uint16_t*>(sden + kWarps * C);
  int* const sscan = reinterpret_cast<int*>(slist + 2 * kListStride);

  //: ONE 64-BIT BASE PER PLANE, and a 32-BIT BYTE DISPLACEMENT after it -- the
  //: shipped addressing discipline (docs/internals/chunk/chunk_kernel.md#state-addressing).
  const int vRowElems = v_row_bytes / (int)sizeof(__nv_bfloat16);
  const int heads = vRowElems / kDv;
  const __nv_bfloat16* const vPlane =
      pin_address(v + (long)(bh / heads) * L * vRowElems + (long)(bh % heads) * kDv);
  const __nv_bfloat16* const pR = pin_address(pread + (long)bh * L * BP::kWtot);
  const __nv_bfloat16* const pW = pin_address(pwrite + (long)bh * L * BP::kWtot);
  const __nv_bfloat16* const gW = pin_address(gwrite + (long)bh * L);
  float* const oPlane = pin_address(o + (long)bh * L * kDv);
  float* const denPlane = pin_address(den + (long)bh * L);

  // ---- the owner's digit runs, one per level -------------------------------
  //: `o_l` is a bit-field extract of the owner index: the lattice's owner
  //: coordinates are the top `sum_l log2 g_l` bits of `lambda` (R0 spec section 2.4).
  int runCol[D];
  //: AND THE OWNER'S BALLOT ROW, per level: the SAME `o_l`, addressing the row the
  //: facts pass voted for this owner group at this level (K31-i-d).  The liveness
  //: pass reads these words instead of re-reading the amplitude runs itself -- once
  //: per (owner, window, level, side) 32 tokens, against `2*D*s_l` bf16 per TOKEN.
  uint32_t rowOff[D];
  const uint32_t rowBytes = (uint32_t)tok_words * (uint32_t)sizeof(uint32_t);
  static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    const int o_l = (owner / BP::owner_div(l)) % BP::g(l);
    runCol[l] = BP::col_base(l) + o_l * BP::s(l);
    rowOff[l] = (uint32_t)(BP::row_prefix(l) + o_l) * rowBytes;
  });
  //: ONE 64-BIT BASE PER SIDE, and 32-bit byte displacements after it: the side's
  //: whole row block is `kRows * tok_words * 4` bytes, which is 32-bit at every
  //: admitted length.
  const uint32_t* const rowsR =
      pin_address(rows + (long)(2 * bh + 0) * (long)BP::kRows * (long)tok_words);
  const uint32_t* const rowsW =
      pin_address(rows + (long)(2 * bh + 1) * (long)BP::kRows * (long)tok_words);

  // ---- lane-invariant operand addressing (the shipped idiom) ---------------
  const int lr = lane & 7, tl = lane >> 3;
  const int rF = (tl & 1) * 8 + lr, cF = tl >> 1;
  const int rT = (tl >> 1) * 8 + lr, cT = tl & 1;
  //: THE PANEL RING'S TWO BASES, AS NAMED SCALARS.  A two-element array indexed by
  //: the batch parity would be a runtime index into a register array, which is a
  //: stack frame; a select is one instruction and no memory.
  const uint32_t brT0 = smem_addr(sr) + (uint32_t)((rT * LdS + cT * 8) * 2);
  const uint32_t brT1 = smem_addr(sw) + (uint32_t)((rT * LdS + cT * 8) * 2);
  const uint32_t bwF0 = smem_addr(sw) + (uint32_t)((rF * LdS + cF * 8) * 2);
  const uint32_t bwF1 = smem_addr(sr) + (uint32_t)((rF * LdS + cF * 8) * 2);
  const uint32_t svB = smem_addr(sv) + (uint32_t)((rT * LdV + cT * 8) * 2);
  constexpr uint32_t kVSlot = (uint32_t)(C * LdV) * 2;
#define ADDR_S(b_, R_, g_) ((b_) + (uint32_t)(((R_) * LdS + (g_) * 8) * 2))
#define ADDR_V(b_, R_, dv_) ((b_) + (uint32_t)(((R_) * LdV + (dv_)) * 2))

  const int chan0 = warp * kMmaM;  // the warp's channel origin
  const uint32_t massOff = (uint32_t)(warp * kMmaK * 2);

  // ---- the state, transposed: m = channels, n = this warp's leaves ---------
  float st[NDV][4];
#pragma unroll
  for (int j = 0; j < NDV; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) st[j][e] = 0.f;
  float mass_st[kMassTiles][4];
#pragma unroll
  for (int j = 0; j < kMassTiles; ++j)
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
  //: element `k` of an accumulator tile sits at row `(lane>>2) + 8*(k>>1)` and
  //: column `(lane&3)*2 + (k&1)`; here rows are channels and columns are leaves.
  auto page_value = [&](int p, int k) {
    return (p * 8 + (lane & 3) * 2 + (k & 1)) * LdO + chan0 + (lane >> 2) + 8 * (k >> 1);
  };
  auto page_mass = [&](int p, int k) { return (p * 8 + lane * 2 + k) * LdO + kDv; };
  //: THE ATOM WALK'S SHAPE IS A REGISTER-INDEXING FACT, NOT A CONVENIENCE.  A
  //: state fragment is a register array, and a runtime index into one is a stack
  //: frame -- measured, on this kernel's first build: `kAtoms` walked at runtime put
  //: the WHOLE state (320 B/thread) in local memory.  So the walk is FULLY
  //: UNROLLED over the atoms, and every warp owns every atom (its tile index is a
  //: compile-time `2*a`) because the carve is along the channels.

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
        for (int p = 0; p < 2; ++p)
#pragma unroll
          for (int k = 0; k < 4; ++k) st[2 * a + p][k] = so[page_value(p, k)];
        if (warp == a % kMassUnits && lane < 4) {
#pragma unroll
          for (int p = 0; p < 2; ++p)
#pragma unroll
            for (int k = 0; k < 2; ++k) mass_st[2 * (a / kMassUnits) + p][k] = so[page_mass(p, k)];
        }
        rendezvous();
      }
    }
  }

  // ---- the produce's per-thread map ---------------------------------------
  constexpr int kRowThr = PM::kRowThr, SPT = PM::SPT;
  const int prow = tid / kRowThr;  // this thread's token row of the batch
  const int rIn = tid % kRowThr;   // its slot run within the box
  const int slot0 = rIn * SPT;
  int lvCol[D];
  static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    lvCol[l] = runCol[l] + BP::u_of(l, slot0) * K;
  });

  //: `gain == 0` zeroes the row -- a padding row of a partial batch has no token,
  //: and a zero coefficient contributes exactly nothing to either MMA (the produce's
  //: dead-row zeroing law).  It is one select on the LEADING factor, so the whole
  //: run is covered once.
  auto produce = [&](const __nv_bfloat16* base, __nv_bfloat16* dst, uint32_t tokOff,
                     uint32_t gain) {
    unsigned short lv[D][PM::kMaxN];
    static_for<D>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      unsigned short run[PM::n_l(l)];
      load_run<PM::n_l(l)>(run, gather_ptr(base, tokOff + (uint32_t)(lvCol[l] * 2)));
#pragma unroll
      for (int i = 0; i < PM::n_l(l); ++i) lv[l][i] = run[i];
    });
    __nv_bfloat16* const row = dst + prow * LdS + slot0;
#pragma unroll
    for (int j = 0; j < SPT; j += 2) {
      //: the innermost level supplies the ADJACENT PAIR (its `c` field owns bit 0),
      //: every level above it a per-pair scalar -- the shipped producer's chain.
      uint32_t acc = ((uint32_t)lv[D - 1][PM::idx_of(D - 1, j + 1)] << 16)
                     | (uint32_t)lv[D - 1][PM::idx_of(D - 1, j)];
      static_for<D - 1>([&](auto Lc) {
        constexpr int l = decltype(Lc)::value;
        acc = mul_bf16x2(
            acc, splat_bf16x2(*reinterpret_cast<const __nv_bfloat16*>(&lv[l][PM::idx_of(l, j)])));
      });
      store_bf16x2(row + j, mul_bf16x2(acc, gain));
    }
  };

  //: 1.0 as a bf16 pair -- the read side's gain.  Folding it through the same
  //: multiply keeps ONE produce body, and `rn(a*1) = a` leaves the read panel
  //: untouched: the multiply is paid, the value is not.
  constexpr uint32_t kOneSplat = 0x3F803F80u;

  // ---- V staging -----------------------------------------------------------
  constexpr int kChunkV = kDv / 8;
  constexpr int kRowsPerPass = kThreads / kChunkV;
  constexpr int kPasses = C / kRowsPerPass;
  static_assert(kPasses >= 1, "a batch must be at least kThreads/8 rows");
  const int ldRow = tid / kChunkV, ldCol = (tid % kChunkV) * 8;
  const uint32_t svL = smem_addr(sv) + (uint32_t)((ldRow * LdV + ldCol) * 2);

  // ---- the fan-in's own addressing ----------------------------------------
  constexpr int kFanLanes = 32;
  constexpr int kFanPerThr = kDv / kFanLanes;
  constexpr int kFanRows = kThreads / kFanLanes;
  constexpr int kFanPasses = C / kFanRows;
  static_assert(kFanPasses * kFanRows == C, "the fan-in's passes cover the batch exactly");
  const int fanRow = tid / kFanLanes, fanCol = tid % kFanLanes;
  const uint32_t fanColB = (uint32_t)(fanCol * (int)sizeof(float));
  const int fanE = fanRow * LdO + fanCol;
  const int oq = (lane & 3) * 2, om = lane >> 2;

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
                             int o0, int ntot, bool write_side) {
      const int nb = (ntot - o0) < C ? (ntot - o0) : C;
      const bool have = prow < nb;
      const int tok = list[o0 + (have ? prow : nb - 1)];
      uint32_t gain = kOneSplat;
      if (write_side)
        gain = splat_bf16x2(
            *reinterpret_cast<const __nv_bfloat16*>(gather_ptr(gW, winG + (uint32_t)tok * 2u)));
      produce(base, dst, winP + (uint32_t)tok * (uint32_t)(BP::kWtot * 2), have ? gain : 0u);
    };

    // ---- the liveness pass: TWO lists, one per SIDE -----------------------
    //: THE SCHEDULE IS TWO-SIDED (Blake ruling 2026-08-19, study section 2.0 as
    //: rewritten).  Per level at most ONE side is sparse, so there is one stored row
    //: set per level -- but the two sides' CLAUSE SETS are different level subsets,
    //: and a dense side's clause is absent-means-full.  ANDing each side's OWN rows
    //: IS that conjunction over exactly the levels where that side is sparse: a dense
    //: level's ballot word is all ones for every token and drops out on its own, with
    //: no structure fact carried into the kernel.  Union routing (one support serving
    //: both sides) is then the special case where the two row sets coincide, and it
    //: needs no separate code.
    //:
    //: Per-side lists make activity AIRTIGHT: a side's simplex weights are strictly
    //: positive on its own support, so a listed entry is live for that side by
    //: construction.  There is no union-superset padding anywhere, by design.
    if (tid < 2) sscan[8 + tid] = 0;
    rendezvous();
    //: THE SCAN IS A TABLE READ (K31-i-d).  A warp's thirty-two tokens are one
    //: BALLOT WORD of the owner's row, so the conjunction over levels is `D` word
    //: ANDs per side for the whole warp, and the lane reads its own bit out of the
    //: result -- where the amplitude scan it replaces re-read `2 * sum_l s_l`
    //: bf16 PER TOKEN, per owner, at owner-specific columns that coalesced with
    //: nothing.  A level the owner spans WHOLE emits no row: its clause is true for
    //: every token and drops out of the conjunction with no structure fact carried
    //: into the kernel, which is exactly what the amplitude form's `or_run` did.
#pragma unroll 1
    for (int p0 = 0; p0 < W; p0 += kThreads) {
      const int loc = p0 + tid;
      bool inR = loc < wLen, inW = loc < wLen;
      if (loc < wLen) {
        //: the word is the WARP's, not the lane's: `tok0` and `p0` are both whole
        //: numbers of words, so warp `u` owns word `(tok0 + p0)/32 + u`.
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
      const uint32_t below = (1u << lane) - 1u;
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
    //: THE WHOLE-WINDOW SKIP, at (owner, window) grain, per side -- the block-bitmap
    //: mechanism's harvest, kept (study section 2.1).
    if (nr == 0 && nw == 0) continue;

    // ---- PHASE 1: the READOUT of every read batch, against the window's ENTRY
    //      state.  ALL of the window's readouts precede ANY of its folds, because a
    //      within-window pair is the INTRA term's and this kernel must not carry it.
    //      The batch line is therefore the window line on both sides.
    const int nbr = (nr + C - 1) / C;
    //: THE PRODUCE RUNS ONE BATCH AHEAD, into the other panel of the two-slot ring.
    //: Its global loads are issued before the current batch's MMA block and consumed
    //: after the next rendezvous, so the amplitude-gather latency sits underneath
    //: arithmetic instead of in front of it.
    if (nbr > 0) produce_batch(pR, sr, slist, 0, nr, false);
#pragma unroll 1
    for (int i = 0; i < nbr; ++i) {
      const int off = i * C;
      const int n = (nr - off) < C ? (nr - off) : C;
      const uint32_t brT = (i & 1) ? brT1 : brT0;
      rendezvous();
      if (i + 1 < nbr) produce_batch(pR, ((i + 1) & 1) ? sw : sr, slist, (i + 1) * C, nr, false);

      float acc[NTOK][4];
#pragma unroll
      for (int j = 0; j < NTOK; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.f;
      uint32_t sa[KDw][4];
#pragma unroll
      for (int kk = 0; kk < KDw; ++kk) {
        sa[kk][0] = pack_bf16x2(st[2 * kk][0], st[2 * kk][1]);
        sa[kk][1] = pack_bf16x2(st[2 * kk][2], st[2 * kk][3]);
        sa[kk][2] = pack_bf16x2(st[2 * kk + 1][0], st[2 * kk + 1][1]);
        sa[kk][3] = pack_bf16x2(st[2 * kk + 1][2], st[2 * kk + 1][3]);
      }
#pragma unroll
      for (int m = 0; m < MTOK; ++m) {
        uint32_t rb[KDw][4];
#pragma unroll
        for (int kk = 0; kk < KDw; ++kk) load_frag(rb[kk], ADDR_S(brT, m * 16, 2 * kk));
#pragma unroll
        for (int kk = 0; kk < KDw; ++kk)
#pragma unroll
          for (int p = 0; p < 2; ++p) mma(acc[2 * m + p], sa[kk], &rb[kk][2 * p]);
      }
#pragma unroll
      for (int j = 0; j < NTOK; ++j)
#pragma unroll
        for (int k = 0; k < 4; ++k)
          store_f32(&so[(j * 8 + oq + (k & 1)) * LdO + chan0 + om + 8 * (k >> 1)], acc[j][k]);

      // ---- the mass row of the same readout ---------------------------------
      if (warp < kMassUnits) {
        uint32_t ma[kMassKD][4];
#pragma unroll
        for (int q = 0; q < kMassKD; ++q) {
          ma[q][0] = pack_bf16x2(mass_st[2 * q][0], mass_st[2 * q][1]);
          ma[q][1] = pack_bf16x2(mass_st[2 * q][2], mass_st[2 * q][3]);
          ma[q][2] = pack_bf16x2(mass_st[2 * q + 1][0], mass_st[2 * q + 1][1]);
          ma[q][3] = pack_bf16x2(mass_st[2 * q + 1][2], mass_st[2 * q + 1][3]);
        }
        float macc[NTOK][4];
#pragma unroll
        for (int j = 0; j < NTOK; ++j)
#pragma unroll
          for (int e = 0; e < 4; ++e) macc[j][e] = 0.f;
#pragma unroll
        for (int m = 0; m < MTOK; ++m) {
          uint32_t mb[kMassKD][4];
#pragma unroll
          for (int q = 0; q < kMassKD; ++q)
            load_frag(mb[q], ADDR_S(brT, m * 16, 2 * q * kMassUnits) + massOff);
#pragma unroll
          for (int q = 0; q < kMassKD; ++q)
#pragma unroll
            for (int p = 0; p < 2; ++p) mma(macc[2 * m + p], ma[q], &mb[q][2 * p]);
        }
        if (lane < 4) {
#pragma unroll
          for (int j = 0; j < NTOK; ++j) {
            store_f32(&sden[warp * C + j * 8 + lane * 2], macc[j][0]);
            store_f32(&sden[warp * C + j * 8 + lane * 2 + 1], macc[j][1]);
          }
        }
      } else if (lane < 4) {
#pragma unroll
        for (int j = 0; j < NTOK; ++j) {
          store_f32(&sden[warp * C + j * 8 + lane * 2], 0.f);
          store_f32(&sden[warp * C + j * 8 + lane * 2 + 1], 0.f);
        }
      }
      rendezvous();

      // ---- the RED fan-in of the readout partials ---------------------------
#pragma unroll
      for (int p = 0; p < kFanPasses; ++p) {
        const int row = fanRow + p * kFanRows;
        if (row < n) {
          float* dp = reinterpret_cast<float*>(gather_ptr(
              oPlane, (uint32_t)slist[off + row] * (uint32_t)(kDv * 4) + winO + fanColB));
          const int e0 = fanE + p * kFanRows * LdO;
#pragma unroll
          for (int e = 0; e < kFanPerThr; ++e)
            fan_in_add(dp + e * kFanLanes, so[e0 + e * kFanLanes]);
        }
      }
      if (tid < C && tid < n) {
        float sden_sum = 0.f;
#pragma unroll
        for (int u = 0; u < kWarps; ++u) sden_sum += sden[u * C + tid];
        fan_in_add(
            reinterpret_cast<float*>(gather_ptr(denPlane, (uint32_t)slist[off + tid] * 4u + winD)),
            sden_sum);
      }
      rendezvous();
    }

    // ---- PHASE 2: the FOLD of every write batch ---------------------------
    //: V IS A WRITE-SIDE QUANTITY and is gathered by the WRITE list; the read list
    //: never touches it.  The value ring is `STAGES` deep and asynchronous: issue
    //: order is FIFO, one implicit group per batch, and the wait leaves `STAGES-2`
    //: groups in flight so the gather for batch `i + STAGES - 1` is airborne while
    //: batch `i` folds.
    const int nbw = (nw + C - 1) / C;
    const uint16_t* const wlist = slist + kListStride;
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
    if (nbw > 0) produce_batch(pW, sw, wlist, 0, nw, true);
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
      if (i + 1 < nbw) produce_batch(pW, ((i + 1) & 1) ? sr : sw, wlist, (i + 1) * C, nw, true);

#pragma unroll
      for (int kk = 0; kk < MTOK; ++kk) {
        uint32_t a[4];
        load_frag_t(a, ADDR_V(vsl, kk * 16, chan0));
#pragma unroll
        for (int nj = 0; nj < KDw; ++nj) {
          uint32_t b[4];
          load_frag_t(b, ADDR_S(bwF, kk * 16, 2 * nj));
          mma(st[2 * nj], a, &b[0]);
          mma(st[2 * nj + 1], a, &b[2]);
        }
        if (warp < kMassUnits) {
#pragma unroll
          for (int q = 0; q < kMassKD; ++q) {
            uint32_t mb[4];
            load_frag_t(mb, ADDR_S(bwF, kk * 16, 2 * q * kMassUnits) + massOff);
            mma(mass_st[2 * q], ones_a, &mb[0]);
            mma(mass_st[2 * q + 1], ones_a, &mb[2]);
          }
        }
      }
      rendezvous();
    }
    //: the ring's unconsumed groups are retired before the next window reuses the
    //: value slots.
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
      for (int a = 0; a < kAtoms; ++a) {
#pragma unroll
        for (int p = 0; p < 2; ++p)
#pragma unroll
          for (int k = 0; k < 4; ++k) store_f32(&so[page_value(p, k)], st[2 * a + p][k]);
        if (warp == a % kMassUnits && lane < 4) {
#pragma unroll
          for (int p = 0; p < 2; ++p)
#pragma unroll
            for (int k = 0; k < 2; ++k)
              store_f32(&so[page_mass(p, k)], mass_st[2 * (a / kMassUnits) + p][k]);
        }
        rendezvous();
        page_sweep(page_off(atomBase + (uint32_t)a), [&](int sidx, uint32_t offb) {
          store_f32(reinterpret_cast<float*>(gather_ptr(plane, offb)), so[sidx]);
        });
        rendezvous();
      }
    }
  }
#undef ADDR_S
#undef ADDR_V
}

}  // namespace rola::carry
