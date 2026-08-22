// csrc/rola/src/carry/carry_split_kernel.cuh -- THE CHANNEL-SPLIT CARRY BODY.
//
// THE CARRY FAMILY HAS TWO BODIES AND THE PER-LEVEL MODE DECLARATION CHOOSES
// BETWEEN THEM (K33 law 3, K34 section 5).  This one is the body for a
// declaration with NO sparse level on the carve axis: every warp owns `kMmaM` of
// the `d_v` channels across ALL `BC` leaves, so the four warps' readout rows are
// disjoint and there is nothing for the leaf carve to exploit.  `carry_kernel.cuh`
// holds the other body, which carves the leaves at the level the declaration
// marks sparse.  Neither is a fallback for the other: a declaration selects one of
// them at compile time and the axis never appears at run time.
//
// Symbols are `carry_kernel.cuh`'s.  See docs/internals/carry/carry_kernel.md.
#pragma once

#include "carry_kernel.cuh"

namespace rola::carry {

template <int D, int B, int K, int M, int W, int C>
struct SplitGeom {
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
  //: the liveness scan's per-(iteration, warp) live counts, BOTH SIDES IN ONE WORD.
  static constexpr int kScanIters = (W + kThreads - 1) / kThreads;
  static constexpr int scan_bytes = kWarps * kScanIters * (int)sizeof(int);
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

//: THE SECOND LAUNCH BOUND: the CTA count this arm can actually reach, taken from
//: the same geometry the ring depth is.  A ring sized for a residency the launch
//: bound does not ask for is a ring sized for nothing.
template <int D, int B, int K, int M, int W, int C>
constexpr int kSplitCtaTarget = SplitGeom<D, B, K, M, W, C>::cta_target;

// ---------------------------------------------------------------- the kernel

template <int D, int B, int K, int M, int W, int C, bool STATE>
__global__ __launch_bounds__(kThreads, (kSplitCtaTarget<D, B, K, M, W, C>)) void carry_split_kernel(
    const __nv_bfloat16* __restrict__ pread, const __nv_bfloat16* __restrict__ pwrite,
    const __nv_bfloat16* __restrict__ gwrite, const __nv_bfloat16* __restrict__ v,
    const uint32_t* __restrict__ rows, float* __restrict__ o, float* __restrict__ den, int L,
    int tok_words, int v_row_bytes, const float* __restrict__ s_in, float* __restrict__ s_out) {
  using BP = BoxPlan<D, B, K, M>;
  using G = SplitGeom<D, B, K, M, W, C>;
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
  //: when the split reaches every warp the mass guard is a TAUTOLOGY, and it must be
  //: spelled as one: `__launch_bounds__` does not let the compiler prove
  //: `threadIdx.x < kThreads`, so the guard would otherwise cut the readout and the
  //: fold into basic blocks and pin the lookahead produce out of their HMMA shadows.
  constexpr bool kMassAllWarps = kMassUnits == kWarps;

  static_assert(C % kMmaM == 0 && kThreads % C == 0, "the batch tile must divide");
  static_assert(BC % kMmaK == 0, "the box is a whole number of k-tiles");
  static_assert(BC % kAtomLeaves == 0, "K2: an owner holds whole page atoms");
  static_assert(design::smem_fits(G::bytes), "this arm does not fit the SMEM maximum");
  static_assert(PM::kRowThr * PM::SPT == BC, "the row's threads cover the box exactly");
  static_assert(PM::SPT % 2 == 0, "a thread publishes whole bf16 pairs");
  static_assert(LdS % 8 == 0 && LdV % 8 == 0, "an ldmatrix row address must be 16-byte aligned");
  static_assert(BP::span_bits(D - 1) >= 1, "the innermost level supplies the adjacent pair");
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
    lvCol[l] = runCol[l] + BP::run_of(l, slot0);
  });

  //: A DEAD ROW IS ZEROED AT THE NARROWEST FACTOR IT HAS.  A padding row of a partial
  //: batch has no token, and a zero coefficient contributes exactly nothing to either
  //: MMA (the produce's dead-row zeroing law).  Level 0's run is the outermost one and
  //: therefore the narrowest at every built arm, so zeroing it covers the whole slot
  //: run for one or two selects and costs no arithmetic.
  //:
  //: THE READ SIDE HAS NO GAIN.  Its scalar is identically 1 and the trailing
  //: `mul_bf16x2(acc, 1)` was 44% of the read produce -- 32 FMUL, 32 PRMT and 16 F2FP
  //: for `rn(a*1) = a` -- so the side is a compile-time argument and the read body
  //: stores its accumulators directly.
  auto produce = [&](auto Wt, const __nv_bfloat16* base, __nv_bfloat16* dst, uint32_t tokOff,
                     uint32_t gain, bool have) {
    constexpr bool WRITE = decltype(Wt)::value;
    //: THE SLOT ACCUMULATORS ARE THE ONLY ARRAY THE PRODUCE CARRIES.  A level's run
    //: is loaded, folded into every slot and dropped inside one `static_for` body, so
    //: no `[D][n]` array is ever live: `SPT/2` packed pairs plus one run, against the
    //: `sum_l n_l` shorts a per-level array would hold at once.
    //: the innermost level supplies the ADJACENT PAIR (its run offset owns bit 0),
    //: every level above it a per-pair scalar -- the shipped producer's chain.
    uint32_t acc[SPT / 2];
    {
      unsigned short run[PM::n_l(D - 1)];
      load_run<PM::n_l(D - 1)>(run, gather_ptr(base, tokOff + (uint32_t)(lvCol[D - 1] * 2)));
#pragma unroll
      for (int j = 0; j < SPT; j += 2)
        acc[j / 2] =
            ((uint32_t)run[PM::idx_of(D - 1, j + 1)] << 16) | (uint32_t)run[PM::idx_of(D - 1, j)];
    }
    static_for<D - 1>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      unsigned short run[PM::n_l(l)];
      load_run<PM::n_l(l)>(run, gather_ptr(base, tokOff + (uint32_t)(lvCol[l] * 2)));
      if constexpr (!WRITE && l == 0) {
#pragma unroll
        for (int i = 0; i < PM::n_l(0); ++i) run[i] = have ? run[i] : (unsigned short)0;
      }
#pragma unroll
      for (int j = 0; j < SPT; j += 2)
        acc[j / 2] = mul_bf16x2(acc[j / 2], splat_bf16x2_bits(run[PM::idx_of(l, j)]));
    });
    __nv_bfloat16* const row = dst + prow * LdS + slot0;
    if constexpr (WRITE) {
#pragma unroll
      for (int j = 0; j < SPT; j += 2) store_bf16x2(row + j, mul_bf16x2(acc[j / 2], gain));
    } else {
#pragma unroll
      for (int j = 0; j < SPT; j += 2) store_bf16x2(row + j, acc[j / 2]);
    }
  };

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
    //: THE SCAN IS A TABLE READ (K31-i-d).  A warp's thirty-two tokens are one
    //: BALLOT WORD of the owner's row, so the conjunction over levels is `D` word
    //: ANDs per side for the whole warp, and the lane reads its own bit out of the
    //: result -- where the amplitude scan it replaces re-read `2 * sum_l s_l`
    //: bf16 PER TOKEN, per owner, at owner-specific columns that coalesced with
    //: nothing.  A level the owner spans WHOLE emits no row: its clause is true for
    //: every token and drops out of the conjunction with no structure fact carried
    //: into the kernel, which is exactly what the amplitude form's `or_run` did.
    //:
    //: THE COMPACTION IS TWO RENDEZVOUS, both load-bearing.  Every warp holds its
    //: whole window's ballots in registers and publishes its `kScanIters x 2` live
    //: counts ONCE; the first barrier publishes those counts, and after it EVERY
    //: thread re-derives the CTA-wide exclusive prefix from them in registers -- the
    //: running totals are CTA-uniform and never need a shared round trip, so neither
    //: the per-iteration barrier pair nor the `tid == 0` serial walk over the warps
    //: exists.  The second barrier publishes the LISTS, which the phases read from the
    //: first batch onward, and it doubles as the separation between this window's
    //: count reads and the next window's count writes -- which is why it precedes the
    //: whole-window skip rather than following it.
    constexpr int kScanIters = G::kScanIters;
    //: the two sides ride ONE count word, low half read and high half write.  A side's
    //: window total is at most `W`, so no low-half sum ever carries into the high half
    //: and one packed add carries both prefixes.
    static_assert(W <= 0xFFFF, "a side's window count must fit the packed word's half");
    uint32_t voteR[kScanIters], voteW[kScanIters];
#pragma unroll
    for (int t = 0; t < kScanIters; ++t) {
      const int loc = t * kThreads + tid;
      bool inR = false, inW = false;
      if (loc < wLen) {
        //: the word is the WARP's, not the lane's: `tok0` and `t * kThreads` are both
        //: whole numbers of words, so warp `u` owns word `(tok0 + t*kThreads)/32 + u`.
        const uint32_t wordB =
            (uint32_t)(((tok0 + t * kThreads) >> 5) + warp) * (uint32_t)sizeof(uint32_t);
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
      voteR[t] = lane_ballot(inR);
      voteW[t] = lane_ballot(inW);
      if (lane == 0) sscan[t * kWarps + warp] = bit_count(voteR[t]) | (bit_count(voteW[t]) << 16);
    }
    rendezvous();
    const uint32_t below = (1u << lane) - 1u;
    int run = 0;
#pragma unroll
    for (int t = 0; t < kScanIters; ++t) {
      int inc[kWarps];
      inc[0] = 0;
#pragma unroll
      for (int u = 1; u < kWarps; ++u) inc[u] = inc[u - 1] + sscan[t * kWarps + u - 1];
      int pre = 0;
#pragma unroll
      for (int u = 1; u < kWarps; ++u) pre = warp == u ? inc[u] : pre;
      const int base = run + pre;
      const uint16_t loc = (uint16_t)(t * kThreads + tid);
      if ((voteR[t] >> lane) & 1u) slist[(base & 0xFFFF) + bit_count(voteR[t] & below)] = loc;
      if ((voteW[t] >> lane) & 1u)
        slist[kListStride + (base >> 16) + bit_count(voteW[t] & below)] = loc;
      run += inc[kWarps - 1] + sscan[t * kWarps + kWarps - 1];
    }
    rendezvous();
    const int nr = run & 0xFFFF, nw = run >> 16;
    //: THE WHOLE-WINDOW SKIP, at (owner, window) grain, per side -- the block-bitmap
    //: mechanism's harvest, kept (study section 2.1).
    if (nr == 0 && nw == 0) continue;

    // ---- PHASE 1: the READOUT of every read batch, against the window's ENTRY
    //      state.  ALL of the window's readouts precede ANY of its folds, because a
    //      within-window pair is the INTRA term's and this kernel must not carry it.
    //      The batch line is therefore the window line on both sides.
    const int nbr = (nr + C - 1) / C;
    //: THE PRODUCE RUNS ONE BATCH AHEAD, into the other panel of the two-slot ring,
    //: UNCONDITIONALLY and AFTER the batch's MMA block.  Unconditionally because a
    //: guard is a basic block and ptxas schedules within one; after, because the
    //: produce ends in panel `STS` and the MMA begins in panel `LDSM`, and the two
    //: shared ranges are not provably disjoint -- in the other order that aliasing
    //: edge pins every produce instruction ahead of the HMMAs.  The trailing batch is
    //: entirely dead rows by the produce's zeroing law and lands in a panel no MMA
    //: reads.
    //: THE PRE-LOOP PRODUCE IS THE ONLY THING THE HEAD RENDEZVOUS EVER PUBLISHED, so
    //: it is taken OUT of the loop: for every later batch the publish is already
    //: carried by the readout's own two rendezvous (docs/internals/carry/carry_kernel.md,
    //: "The rendezvous the ring needs").
    if (nbr > 0) produce_batch(std::false_type{}, pR, sr, slist, 0, nr);
    rendezvous();
#pragma unroll 1
    for (int i = 0; i < nbr; ++i) {
      const int off = i * C;
      const int n = (nr - off) < C ? (nr - off) : C;
      const uint32_t brT = (i & 1) ? brT1 : brT0;

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
      if (kMassAllWarps || warp < kMassUnits) {
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
      } else if (!kMassAllWarps && lane < 4) {
#pragma unroll
        for (int j = 0; j < NTOK; ++j) {
          store_f32(&sden[warp * C + j * 8 + lane * 2], 0.f);
          store_f32(&sden[warp * C + j * 8 + lane * 2 + 1], 0.f);
        }
      }
      produce_batch(std::false_type{}, pR, ((i + 1) & 1) ? sw : sr, slist, (i + 1) * C, nr);
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
    if (nbw > 0) produce_batch(std::true_type{}, pW, sw, wlist, 0, nw);
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
        uint32_t a[4];
        load_frag_t(a, ADDR_V(vsl, kk * 16, chan0));
#pragma unroll
        for (int nj = 0; nj < KDw; ++nj) {
          uint32_t b[4];
          load_frag_t(b, ADDR_S(bwF, kk * 16, 2 * nj));
          mma(st[2 * nj], a, &b[0]);
          mma(st[2 * nj + 1], a, &b[2]);
        }
        if (kMassAllWarps || warp < kMassUnits) {
#pragma unroll
          for (int q = 0; q < kMassKD; ++q) {
            uint32_t mb[4];
            load_frag_t(mb, ADDR_S(bwF, kk * 16, 2 * q * kMassUnits) + massOff);
            mma(mass_st[2 * q], ones_a, &mb[0]);
            mma(mass_st[2 * q + 1], ones_a, &mb[2]);
          }
        }
      }
      produce_batch(std::true_type{}, pW, ((i + 1) & 1) ? sr : sw, wlist, (i + 1) * C, nw);
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
