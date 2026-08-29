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

template <int D, int B, int K, int M, int DV, int W, int C>
struct SplitGeom {
  using BP = BoxPlan<D, B, K, M>;
  static constexpr int BC = BP::kBC;
  //: THE VALUE WIDTH IS THIS ARM'S (K35 S3b), exactly as the carved body's is. -- see docs/internals/carry/carry_split_kernel.md#kdv
  static constexpr int kDv = DV;
  static constexpr int kCols = DV + 1;
  static_assert(kDv == kWarps * kMmaM,
                "THE CHANNEL SPLIT'S WIDTH IS ITS WARP COUNT: each of the fixed four "
                "warps holds exactly ONE MMA m-tile of channels, so this body's `d_v` is "
                "determined by its executor count and is not a free dial. A wider value "
                "plane is the CARVED body's arm (16 leaves per warp), never a branch "
                "inside this one.");
  static constexpr int LdV = kDv + kPadE;
  static constexpr int LdS = BC + kPadE;
  static constexpr int LdO = kDv + kPadF;

  //: THE TWO PANELS ARE A TWO-SLOT RING, AND THEY COST NOTHING EXTRA. -- see docs/internals/carry/carry_split_kernel.md#panel-bytes
  static constexpr int panel_bytes = 2 * C * LdS * (int)sizeof(__nv_bfloat16);
  static constexpr int out_bytes = C * LdO * (int)sizeof(float);
  static constexpr int den_bytes = kWarps * C * (int)sizeof(float);
  //: TWO packed lists per window, one per SIDE (study section 2.0, two-sided -- see docs/internals/carry/carry_split_kernel.md#kliststride
  static constexpr int kListStride = ((W + 7) / 8) * 8;
  static constexpr int list_bytes = 2 * kListStride * (int)sizeof(uint16_t);
  //: the liveness scan's per-(iteration, warp) live counts, BOTH SIDES IN ONE WORD.
  static constexpr int kScanIters = (W + kThreads - 1) / kThreads;
  static constexpr int scan_bytes = kWarps * kScanIters * (int)sizeof(int);
  //: THE V RING'S DEPTH IS DERIVED FROM THE SMEM BUDGET AND THE REGISTER FLOOR, -- see docs/internals/carry/carry_split_kernel.md#kregfloor
  static constexpr int kRegFloor = 168;
  static constexpr int slot_bytes = C * LdV * (int)sizeof(__nv_bfloat16);
  static constexpr int fixed_bytes = panel_bytes + out_bytes + den_bytes + list_bytes + scan_bytes;
  static constexpr int kStages = design::stages_for(slot_bytes, fixed_bytes, kThreads, kRegFloor);
  static constexpr size_t bytes = (size_t)fixed_bytes + (size_t)kStages * slot_bytes;
  static constexpr int cta_target = design::cta_target((int)bytes, kThreads, kRegFloor);
};

//: THE SECOND LAUNCH BOUND: the CTA count this arm can actually reach, taken from -- see docs/internals/carry/carry_split_kernel.md#near-line-61
template <int D, int B, int K, int M, int DV, int W, int C>
constexpr int kSplitCtaTarget = SplitGeom<D, B, K, M, DV, W, C>::cta_target;

// ---------------------------------------------------------------- the kernel

template <int D, int B, int K, int M, int DV, int W, int C>
__global__ __launch_bounds__(kThreads, (kSplitCtaTarget<D, B, K, M, DV, W, C>)) void
carry_split_kernel(const __nv_bfloat16* __restrict__ pread,
                   const __nv_bfloat16* __restrict__ pwrite,
                   const __nv_bfloat16* __restrict__ gwrite, const __nv_bfloat16* __restrict__ v,
                   const uint32_t* __restrict__ rows, float* __restrict__ o,
                   float* __restrict__ den, int L, int tok_words, int v_row_bytes,
                   const float* __restrict__ s_in, float* __restrict__ s_out,
                   const int32_t* __restrict__ page_tbl, const uint8_t* __restrict__ atom_bits) {
  using BP = BoxPlan<D, B, K, M>;
  using G = SplitGeom<D, B, K, M, DV, W, C>;
  using PM = ProduceMap<D, B, K, M, C>;
  //: THE ARM'S VALUE WIDTH, named once so the body reads a geometry, not a constant.
  constexpr int kDv = G::kDv, kCols = G::kCols;

  constexpr int BC = BP::kBC;
  constexpr int LdV = G::LdV, LdS = G::LdS, LdO = G::LdO;
  constexpr int MTOK = C / kMmaM;  // token m-tiles of a batch
  constexpr int NTOK = C / kMmaN;  // token n-tiles of a batch
  //: every warp holds the WHOLE box and a slice of the channels, so the state's -- see docs/internals/carry/carry_split_kernel.md#ndv
  constexpr int NDV = BC / kMmaN;  // the state's n-tiles (leaves)
  constexpr int kDw = BC / kMmaK;  // its k-tiles
  //: THE MASS CHANNEL'S CARVE.  One channel cannot be split along the channel -- see docs/internals/carry/carry_split_kernel.md#kmassunits
  constexpr int kMassUnits = kDw < kWarps ? kDw : kWarps;
  constexpr int kMassKD = kDw / kMassUnits;
  constexpr int kMassTiles = 2 * kMassKD;
  //: when the split reaches every warp the mass guard is a TAUTOLOGY, and it must -- see docs/internals/carry/carry_split_kernel.md#kmassallwarps
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

  //: ONE 64-BIT BASE PER PLANE, and a 32-BIT BYTE DISPLACEMENT after it -- the -- see docs/internals/carry/carry_split_kernel.md#vrowelems
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
  //: `o_l` is a bit-field extract of the owner index: the lattice's owner -- see docs/internals/carry/carry_split_kernel.md#near-line-144
  int runCol[D];
  //: AND THE OWNER'S BALLOT ROW, per level: the SAME `o_l`, addressing the row -- see docs/internals/carry/carry_split_kernel.md#rowbytes
  uint32_t rowOff[D];
  const uint32_t rowBytes = (uint32_t)tok_words * (uint32_t)sizeof(uint32_t);
  static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    const int o_l = (owner / BP::owner_div(l)) % BP::g(l);
    runCol[l] = BP::col_base(l) + o_l * BP::s(l);
    rowOff[l] = (uint32_t)(BP::row_prefix(l) + o_l) * rowBytes;
  });
  //: ONE 64-BIT BASE PER SIDE, and 32-bit byte displacements after it: the side's -- see docs/internals/carry/carry_split_kernel.md#rowsr
  const uint32_t* const rowsR =
      pin_address(rows + (long)(2 * bh + 0) * (long)BP::kRows * (long)tok_words);
  const uint32_t* const rowsW =
      pin_address(rows + (long)(2 * bh + 1) * (long)BP::kRows * (long)tok_words);

  // ---- lane-invariant operand addressing (the shipped idiom) ---------------
  const int lr = lane & 7, tl = lane >> 3;
  const int rF = (tl & 1) * 8 + lr, cF = tl >> 1;
  const int rT = (tl >> 1) * 8 + lr, cT = tl & 1;
  //: THE PANEL RING'S TWO BASES, AS NAMED SCALARS.  A two-element array indexed by -- see docs/internals/carry/carry_split_kernel.md#brt0
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

  //: THE ONES OPERAND, AN IMMEDIATE: the mass channel's value is literally 1, so -- see docs/internals/carry/carry_split_kernel.md#konespair
  constexpr uint32_t kOnesPair = 0x3F803F80u;
  uint32_t ones_a[4];
  ones_a[0] = (lane >> 2) == 0 ? kOnesPair : 0u;
  ones_a[1] = 0u;
  ones_a[2] = ones_a[0];
  ones_a[3] = 0u;

  // ---- the state plane's page sweep, at CANONICAL atom ids -----------------
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
  //: THE PLANE IS ADDRESSED BY CANONICAL ATOM ID (P80) -- the same re-key the carved -- see docs/internals/carry/carry_split_kernel.md#atombh
  const uint32_t atomBH = (uint32_t)bh * (uint32_t)(BP::kLeaves / kAtomLeaves);
  //: THE ACTIVITY OF THIS OWNER'S ATOMS, READ ONCE AT THE PROLOGUE (P80b): the -- see docs/internals/carry/carry_split_kernel.md#activity
  static_assert(kAtoms <= 32, "one bit per atom per mask: an owner holds at most 32 atoms");
  uint32_t actRead = 0u, actWrite = 0u;
#pragma unroll
  for (int a = 0; a < kAtoms; ++a) {
    const uint8_t bits = atom_activity(atom_bits, atomBH + BP::canon_atom(owner, a));
    actRead |= (uint32_t)((bits & kAtomRead) != 0u) << a;
    actWrite |= (uint32_t)((bits & kAtomWritten) != 0u) << a;
  }
  //: THE SLOT, NOT THE ATOM, ADDRESSES THE PLANE (`ops.cuh`'s `page_slot`, -- see docs/internals/carry/carry_split_kernel.md#kpagebytes
  constexpr uint32_t kPageBytes = (uint32_t)(kPageElems * (int)sizeof(float));
  auto page_base = [](int32_t slot) { return (uint32_t)slot * kPageBytes; };
  //: element `k` of an accumulator tile sits at row `(lane>>2) + 8*(k>>1)` and -- see docs/internals/carry/carry_split_kernel.md#pagevalue
  auto page_value = [&](int p, int k) {
    return (p * 8 + (lane & 3) * 2 + (k & 1)) * LdO + chan0 + (lane >> 2) + 8 * (k >> 1);
  };
  auto page_mass = [&](int p, int k) { return (p * 8 + lane * 2 + k) * LdO + kDv; };
  //: THE ATOM WALK'S SHAPE IS A REGISTER-INDEXING FACT, NOT A CONVENIENCE. -- see docs/internals/carry/carry_split_kernel.md#note-l250

  //: ---- THE ENTRY STATE. The fragments are SET, never accumulated: they are -- see docs/internals/carry/carry_split_kernel.md#note-l257
  if (s_in != nullptr) {
    const float* const plane = pin_address(s_in);
#pragma unroll
    for (int a = 0; a < kAtoms; ++a) {
      //: AN ABSENT OR IDLE ATOM IS SKIPPED, AND WITH IT THE TWO BARRIERS: the load -- see docs/internals/carry/carry_split_kernel.md#slot
      const int32_t slot = page_slot(page_tbl, atomBH + BP::canon_atom(owner, a));
      if (slot < 0 || (((actRead | actWrite) >> a) & 1u) == 0u) continue;
      page_sweep(page_base(slot), [&](int sidx, uint32_t off) {
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

  //: A DEAD ROW IS ZEROED AT THE NARROWEST FACTOR IT HAS. A padding row of a -- see docs/internals/carry/carry_split_kernel.md#write
  auto produce = [&](auto Wt, const __nv_bfloat16* base, __nv_bfloat16* dst, uint32_t tokOff,
                     uint32_t gain, bool have) {
    constexpr bool WRITE = decltype(Wt)::value;
    //: THE SLOT ACCUMULATORS ARE THE ONLY ARRAY THE PRODUCE CARRIES. -- see docs/internals/carry/carry_split_kernel.md#note-l317
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
        //: §6: PM::n_l(0) is compile-time (a constexpr static member function of a -- see docs/internals/carry/carry_split_kernel.md#knl0
        constexpr int kNL0 = PM::n_l(0);
#pragma unroll
        for (int i = 0; i < kNL0; ++i) run[i] = have ? run[i] : (unsigned short)0;
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
    //: ONE BATCH'S PANEL, expanded from factors. -- see docs/internals/carry/carry_split_kernel.md#nb
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
    //: THE SCHEDULE IS TWO-SIDED (Blake ruling 2026-08-19, study section 2.0 as -- see docs/internals/carry/carry_split_kernel.md#kscaniters
    constexpr int kScanIters = G::kScanIters;
    //: the two sides ride ONE count word, low half read and high half write.  A side's -- see docs/internals/carry/carry_split_kernel.md#near-line-381
    static_assert(W <= 0xFFFF, "a side's window count must fit the packed word's half");
    uint32_t voteR[kScanIters], voteW[kScanIters];
#pragma unroll
    for (int t = 0; t < kScanIters; ++t) {
      const int loc = t * kThreads + tid;
      bool inR = false, inW = false;
      if (loc < wLen) {
        //: the word is the WARP's, not the lane's: `tok0` and `t * kThreads` are both -- see docs/internals/carry/carry_split_kernel.md#wordb
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
    //: THE WHOLE-WINDOW SKIP, at (owner, window) grain, per side -- the block-bitmap -- see docs/internals/carry/carry_split_kernel.md#nr
    if (nr == 0 && nw == 0) continue;

    // ---- PHASE 1: the READOUT of every read batch, against the window's ENTRY
    //      state.  ALL of the window's readouts precede ANY of its folds, because a
    //      within-window pair is the INTRA term's and this kernel must not carry it.
    //      The batch line is therefore the window line on both sides.
    const int nbr = (nr + C - 1) / C;
    //: THE PRODUCE RUNS ONE BATCH AHEAD, into the other panel of the two-slot ring, -- see docs/internals/carry/carry_split_kernel.md#rendezvous
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
      uint32_t sa[kDw][4];
#pragma unroll
      for (int kk = 0; kk < kDw; ++kk) {
        sa[kk][0] = pack_bf16x2(st[2 * kk][0], st[2 * kk][1]);
        sa[kk][1] = pack_bf16x2(st[2 * kk][2], st[2 * kk][3]);
        sa[kk][2] = pack_bf16x2(st[2 * kk + 1][0], st[2 * kk + 1][1]);
        sa[kk][3] = pack_bf16x2(st[2 * kk + 1][2], st[2 * kk + 1][3]);
      }
#pragma unroll
      for (int m = 0; m < MTOK; ++m) {
        uint32_t rb[kDw][4];
#pragma unroll
        for (int kk = 0; kk < kDw; ++kk) load_frag(rb[kk], ADDR_S(brT, m * 16, 2 * kk));
#pragma unroll
        for (int kk = 0; kk < kDw; ++kk)
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
    //: V IS A WRITE-SIDE QUANTITY and is gathered by the WRITE list; the read list -- see docs/internals/carry/carry_split_kernel.md#nbw
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
      //: the ring slot is advanced on EVERY batch, live or not, so the wait/commit -- see docs/internals/carry/carry_split_kernel.md#f
      const int f = i + STAGES - 1;
      if (f < nbw) stage_batch(f, f % STAGES);
      stage_commit();

#pragma unroll
      for (int kk = 0; kk < MTOK; ++kk) {
        uint32_t a[4];
        load_frag_t(a, ADDR_V(vsl, kk * 16, chan0));
#pragma unroll
        for (int nj = 0; nj < kDw; ++nj) {
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
    //: the ring's unconsumed groups are retired before the next window reuses the -- see docs/internals/carry/carry_split_kernel.md#near-line-606
    stage_wait<0>();
  }

  // ---- the FINAL STATE, at CANONICAL atom ids ------------------------------
  //: the mirror of the entry load, after the last fan-in has consumed the staging -- see docs/internals/carry/carry_split_kernel.md#rendezvous-2
  if (s_out != nullptr) {
    float* const plane = pin_address(s_out);
    rendezvous();
#pragma unroll
    for (int a = 0; a < kAtoms; ++a) {
      //: THE EXIT STORES ONLY WRITTEN, PLANNED ATOMS.  An atom this call does not -- see docs/internals/carry/carry_split_kernel.md#slot-2
      const int32_t slot = page_slot(page_tbl, atomBH + BP::canon_atom(owner, a));
      if (slot < 0 || ((actWrite >> a) & 1u) == 0u) continue;
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
      page_sweep(page_base(slot), [&](int sidx, uint32_t offb) {
        store_f32(reinterpret_cast<float*>(gather_ptr(plane, offb)), so[sidx]);
      });
      rendezvous();
    }
  }
#undef ADDR_S
#undef ADDR_V
}

}  // namespace rola::carry
