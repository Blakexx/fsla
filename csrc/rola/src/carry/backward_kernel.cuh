// csrc/rola/src/carry/backward_kernel.cuh -- K31 RUNG R4: THE CARRY'S TWO REVERSE SCANS.
//
// ZERO STATE SNAPSHOTS.  With no decay the recurrence is affine in the state, so the
// state at any window boundary is the SUM of the deposits below it and the reverse
// pass REPLAYS it rather than storing it.  TWO KERNELS, ONE RESIDENT SLAB EACH: a
// fused form needs two fp32 slabs in one thread's registers and fails the 255-register
// cap.  The scans are data-independent given the seeds, so both slabs are resident in
// the MACHINE at once and neither thread holds two.
//
//     Scan A: windows ASCENDING, slab = the replayed state, emits d_pr
//     Scan B: windows DESCENDING, slab = the adjoint, emits d_v, d_gw, d_pw, d_state_in
//
// THE SLAB IS LEAF-MAJOR IN BOTH, the transpose of the forward's: both readouts
// contract over CHANNELS, so `m = leaf` and the accumulator is its own readout's A
// operand without a round trip.  The carve follows the `m` axis as it always has, so
// warps own LEAF slices here where the forward's own channel slices.
//
// SYMBOLS, each at first use: `D` levels, `B` digits per level, `K` the sub-box span,
// `M` the capacity multiplier, `BC` the leaves one owner holds, `W` the window length
// in tokens, `C` the private batch's token tile, `d_v = 64` value channels,
// `cols = d_v + 1`, `r_l` a leaf's run offset inside its owner at level `l`.
//
// THE REVERSE ORDERING LAW is the mirror of the forward's: writes of window `w`
// execute before `w`'s reads fold.  See docs/internals/carry/backward_kernel.md.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <type_traits>

#include "box.cuh"
#include "carry_kernel.cuh"
#include "common/design.cuh"
#include "common/ops.cuh"

namespace rola::carry {

//: THE BACKWARD FAMILY'S OWN DEVICE STAMP.  `cudaFuncGetAttributes` is a property of
//: a KERNEL, and so is a stale binary; the forward's magic says nothing about these.
static constexpr int kBwdBuildMagic = 31040001;

//: THE SEED AND VALUE PLANES ARE PRE-PACKED `cols`-WIDE AND PADDED TO WHOLE MMA
//: n-tiles, so ONE staging body serves both and the mass channel is an ordinary
//: column rather than a carve.  Columns past `kCols` are zero at the seam, which is
//: what lets the last n-tile be read with no partial-tile path.
static constexpr int kColsPad = 72;
static constexpr int kSlabN = kColsPad / kMmaN;
static constexpr int kColKTiles = kDv / kMmaK;

//: the gain's place in a produce: nowhere, folded into the panel (the forward's own
//: form, which Scan A replays), or staged for the accumulator's TOKEN axis -- which
//: is what (G6) needs, since `sum_s dW` is not `d_gw` unless the panel is ungained.
enum class GainSite { kNone, kPanel, kTokenAxis };

//: level `l`'s `s_l` digit-gradient rows start at `off(l)`.
template <class BP, int D>
struct DigitPlan {
  static constexpr int off(int l) { return l == 0 ? 0 : off(l - 1) + BP::s(l - 1); }
  static constexpr int kRows = off(D - 1) + BP::s(D - 1);
};

template <int D, int B, int K, int M, int W, int C>
struct BwdGeom {
  using BP = BoxPlan<D, B, K, M>;
  using DP = DigitPlan<BP, D>;
  static constexpr int BC = BP::kBC;
  static constexpr int LdS = BC + kPadE;
  static constexpr int LdD = kColsPad + kPadE;
  static constexpr int LdO = kDv + kPadF;

  static constexpr int panel_bytes = 2 * C * LdS * (int)sizeof(__nv_bfloat16);
  static constexpr int out_bytes = C * LdO * (int)sizeof(float);
  static constexpr int digit_bytes = 3 * DP::kRows * C * (int)sizeof(float);
  static constexpr int gain_bytes = 2 * C * (int)sizeof(float);
  static constexpr int col_bytes = DP::kRows * (int)sizeof(int);
  static constexpr int kListStride = ((W + 7) / 8) * 8;
  static constexpr int list_bytes = 2 * kListStride * (int)sizeof(uint16_t);
  static constexpr int scan_bytes = 16 * (int)sizeof(int);
  static constexpr int slot_bytes = C * LdD * (int)sizeof(__nv_bfloat16);
  static constexpr int fixed_bytes =
      panel_bytes + out_bytes + digit_bytes + gain_bytes + col_bytes + list_bytes + scan_bytes;

  //: TWO FLOORS, TWO CENSUSES.  One floor for two kernels is how a spill goes green:
  //: the build honors a bound the heavier scan cannot meet and only the census
  //: notices.  Both are DECLARED -- the measured live set of the heaviest built arm.
  static constexpr int kRegFloorA = 232;
  static constexpr int kRegFloorB = 250;
  static constexpr int kStagesA = design::stages_for(slot_bytes, fixed_bytes, kThreads, kRegFloorA);
  static constexpr int kStagesB = design::stages_for(slot_bytes, fixed_bytes, kThreads, kRegFloorB);
  static constexpr size_t bytes_a = (size_t)fixed_bytes + (size_t)kStagesA * slot_bytes;
  static constexpr size_t bytes_b = (size_t)fixed_bytes + (size_t)kStagesB * slot_bytes;
  static constexpr int cta_target_a = design::cta_target((int)bytes_a, kThreads, kRegFloorA);
  static constexpr int cta_target_b = design::cta_target((int)bytes_b, kThreads, kRegFloorB);
};

template <int D, int B, int K, int M, int W, int C>
constexpr int kScanACtaTarget = BwdGeom<D, B, K, M, W, C>::cta_target_a;
template <int D, int B, int K, int M, int W, int C>
constexpr int kScanBCtaTarget = BwdGeom<D, B, K, M, W, C>::cta_target_b;

//: THE OPERAND PACK IS SHARED so the two scans cannot drift on what a plane means;
//: each ignores the outputs it does not own.  `vt` and `dn` are `[BH, L, LdD]` bf16,
//: `(v, 1, 0...)` and `(d_num, d_den, 0...)`.
struct BwdParams {
  const __nv_bfloat16* pread;
  const __nv_bfloat16* pwrite;
  const __nv_bfloat16* gwrite;
  const __nv_bfloat16* vt;
  const __nv_bfloat16* dn;
  const uint32_t* rows;
  float* d_pr;
  float* d_pw;
  float* d_gw;
  float* d_v;
  const float* slab_in;
  float* slab_out;
  int L;
  int tok_words;
};

//: ONE OWNER'S ADDRESSING AND THE BODIES BOTH SCANS SHARE.  What differs -- the walk
//: direction, which list drives which phase, which slab is held -- lives in the
//: kernels below and nowhere else.
template <int D, int B, int K, int M, int W, int C, int STAGES>
struct BwdOwner {
  using BP = BoxPlan<D, B, K, M>;
  using G = BwdGeom<D, B, K, M, W, C>;
  using PM = ProduceMap<D, B, K, M, C>;
  using DP = DigitPlan<BP, D>;

  static constexpr int BC = BP::kBC;
  static constexpr int LdS = G::LdS, LdD = G::LdD, LdO = G::LdO;
  static constexpr int MTOK = C / kMmaM;
  static constexpr int NTOK = C / kMmaN;
  static constexpr int kWarpLeaves = BC / kWarps;
  static constexpr int kSlabM = kWarpLeaves / kMmaM;
  static constexpr int kAtoms = BC / kAtomLeaves;
  static constexpr int kPageElems = kAtomLeaves * kCols;
  static constexpr int kPageCarry = kThreads - kCols;
  static constexpr int kListStride = G::kListStride;
  static constexpr int kSlotElems = C * LdD;
  static constexpr uint32_t kSlotBytes = (uint32_t)kSlotElems * 2u;
  static constexpr int kChunks = LdD / 8;
  static constexpr int kDigitRows = DP::kRows;
  static constexpr int kFanRows = kThreads / kBallotLanes;
  static constexpr int kFanPasses = C / kFanRows;
  static constexpr int kFanPerThr = kDv / kBallotLanes;
  static constexpr int kGwRows = C / kWarps;
  static constexpr int kLeafGrp = kWarpLeaves / kMmaN;

  //: THE DIGIT REDUCTION'S ROW MAP, stated as the two conditions it rests on rather
  //: than as a depth.  A warp owns `kWarpLeaves` consecutive owner-local indices, so
  //: a level's digit field reduces WITHOUT a cross-warp meeting exactly when the
  //: field lies inside the warp's own bit range, and an accumulator ROW is one digit
  //: exactly when the innermost span is one m-tile.  Depth two at `s = (8, 16)`
  //: satisfies both, and only depth two can: the innermost level owns the four row
  //: bits and ONE other level must own everything above them, including the whole
  //: warp id, so its digits PARTITION across warps.  `s = (2, 4, 16)` splits those
  //: upper bits between two levels and level 1's field then straddles the warp-id
  //: boundary -- two warps hold each of its digits, so the staging acquires a
  //: per-warp digit base and a second cross-warp meeting.  `s = (2, 2, 4, 8)` breaks
  //: the row map as well (`s_3 = 8`, half an m-tile, so one accumulator row carries
  //: two levels' digits).  Each is a per-arm derivation with a wrong-gradient risk,
  //: and a named gap beats one.
  static_assert(kSlabM == 2 && BP::s(D - 1) == kMmaM,
                "the digit reduction's row map: a warp's leaf slice is exactly two m-tiles "
                "and the inner level's span is exactly one of them");
  static_assert(BP::span_bits(0) == BP::kLocalBits - ilog2c(kMmaM),
                "the digit reduction's warp map: the bits above the innermost level's field "
                "must be ONE level's, so that level's digits partition across warps and no "
                "level straddles the warp-id boundary");
  static_assert(kThreads == 4 * C, "the seed/value staging gives each token row four threads");
  static_assert(kThreads > kCols && kThreads < 2 * kCols,
                "the page sweep's row cursor advances by one or two, never by a division");
  static_assert(LdS % 8 == 0 && LdD % 8 == 0, "an ldmatrix row address is 16-byte aligned");
  static_assert(W % kBallotLanes == 0, "a warp's tokens are one ballot word of the owner's row");
  static_assert(kGwRows * kWarps == C && 2 * kBallotLanes == kDv,
                "the gain reduction gives each warp a token block and each lane two channels");
  static_assert(design::smem_fits(G::bytes_a) && design::smem_fits(G::bytes_b),
                "this arm does not fit the SMEM maximum");
  static_assert(STAGES >= 2, "the wait/issue order needs at least two stages");

  const int tid, warp, lane, owner, bh, L;
  __nv_bfloat16* sd;
  __nv_bfloat16 *sp0, *sp1;
  float* so;
  float* sdg;
  float* samp;
  float* sgain;
  int* sdcol;
  uint16_t* slist;
  int* sscan;
  const __nv_bfloat16 *pR, *pW, *gW, *vtP, *dnP;
  const uint32_t *rowsR, *rowsW;
  uint32_t rowOff[D];
  int runCol[D], lvCol[D];
  //: THE PANEL RING'S BASES, AS NAMED SCALARS.  A two-element array indexed by the
  //: batch parity is a runtime index into a register array, which is a stack frame;
  //: a select is one instruction and no memory.
  uint32_t bpT0, bpT1, bpF0, bpF1, bdT, bdF;
  uint32_t atomBase;
  int prow, rIn, slot0;
  int nr, nw;

  __device__ __forceinline__ BwdOwner(const BwdParams& p, char* smem)
      : tid((int)threadIdx.x),
        warp((int)threadIdx.x >> 5),
        lane((int)threadIdx.x & 31),
        owner((int)blockIdx.x),
        bh((int)blockIdx.y),
        L(p.L) {
    sd = reinterpret_cast<__nv_bfloat16*>(smem);
    sp0 = sd + STAGES * kSlotElems;
    sp1 = sp0 + C * LdS;
    so = reinterpret_cast<float*>(sp1 + C * LdS);
    sdg = so + C * LdO;
    samp = sdg + kDigitRows * C;
    sgain = samp + 2 * kDigitRows * C;
    sdcol = reinterpret_cast<int*>(sgain + 2 * C);
    slist = reinterpret_cast<uint16_t*>(sdcol + kDigitRows);
    sscan = reinterpret_cast<int*>(slist + 2 * kListStride);

    pR = pin_address(p.pread + (long)bh * L * BP::kWtot);
    pW = pin_address(p.pwrite + (long)bh * L * BP::kWtot);
    gW = pin_address(p.gwrite + (long)bh * L);
    vtP = pin_address(p.vt + (long)bh * L * LdD);
    dnP = pin_address(p.dn + (long)bh * L * LdD);
    rowsR = pin_address(p.rows + (long)(2 * bh + 0) * (long)BP::kRows * (long)p.tok_words);
    rowsW = pin_address(p.rows + (long)(2 * bh + 1) * (long)BP::kRows * (long)p.tok_words);

    const uint32_t rowBytes = (uint32_t)p.tok_words * (uint32_t)sizeof(uint32_t);
    static_for<D>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      const int o_l = (owner / BP::owner_div(l)) % BP::g(l);
      runCol[l] = BP::col_base(l) + o_l * BP::s(l);
      rowOff[l] = (uint32_t)(BP::row_prefix(l) + o_l) * rowBytes;
    });

    const int lr = lane & 7, tl = lane >> 3;
    const int rF = (tl & 1) * 8 + lr, cF = tl >> 1;
    const int rT = (tl >> 1) * 8 + lr, cT = tl & 1;
    bpT0 = smem_addr(sp0) + (uint32_t)((rT * LdS + cT * 8) * 2);
    bpT1 = smem_addr(sp1) + (uint32_t)((rT * LdS + cT * 8) * 2);
    bpF0 = smem_addr(sp0) + (uint32_t)((rF * LdS + cF * 8) * 2);
    bpF1 = smem_addr(sp1) + (uint32_t)((rF * LdS + cF * 8) * 2);
    bdT = smem_addr(sd) + (uint32_t)((rT * LdD + cT * 8) * 2);
    bdF = smem_addr(sd) + (uint32_t)((rF * LdD + cF * 8) * 2);

    prow = tid / PM::kRowThr;
    rIn = tid % PM::kRowThr;
    slot0 = rIn * PM::SPT;
    static_for<D>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      lvCol[l] = runCol[l] + BP::run_of(l, slot0);
    });
    atomBase = (uint32_t)(bh * (int)(BP::kLeaves / kAtomLeaves) + owner * kAtoms);

    if (tid < kDigitRows) {
      int col = 0;
      static_for<D>([&](auto Lc) {
        constexpr int l = decltype(Lc)::value;
        if (tid >= DP::off(l) && tid < DP::off(l) + BP::s(l)) col = runCol[l] + (tid - DP::off(l));
      });
      sdcol[tid] = col;
    }
    nr = 0;
    nw = 0;
  }

  //: THE LIVENESS PASS, RECOMPUTED rather than saved -- the R4 constraint applied to
  //: the SCHEDULE as well as to the state.  Both scans rebuild the SAME two packed
  //: lists from the same table in the same order, so no list crosses the boundary.
  __device__ __forceinline__ void liveness(int tok0, int wlen) {
    if (tid < 2) sscan[8 + tid] = 0;
    rendezvous();
#pragma unroll 1
    for (int p0 = 0; p0 < W; p0 += kThreads) {
      const int loc = p0 + tid;
      bool inR = loc < wlen, inW = loc < wlen;
      if (loc < wlen) {
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
    nr = sscan[8];
    nw = sscan[9];
  }

  //: ONE BATCH'S PANEL, expanded from factors.  `SAMP` also publishes the per-level
  //: amplitude RUNS: they are the digit reduction's leave-one-out weights, and they
  //: are the same values this body already holds -- never a second gather.
  template <bool SAMP>
  __device__ __forceinline__ void produce(const __nv_bfloat16* base, __nv_bfloat16* dst,
                                          uint32_t tokOff, uint32_t gain, int par) {
    unsigned short lv[D][PM::kMaxN];
    static_for<D>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      unsigned short run[PM::n_l(l)];
      load_run<PM::n_l(l)>(run, gather_ptr(base, tokOff + (uint32_t)(lvCol[l] * 2)));
#pragma unroll
      for (int i = 0; i < PM::n_l(l); ++i) lv[l][i] = run[i];
      if constexpr (SAMP) {
#pragma unroll
        for (int i = 0; i < PM::n_l(l); ++i)
          samp[(par * C + prow) * kDigitRows + DP::off(l) + BP::run_of(l, slot0) + i] =
              bf16_bits_to_float(lv[l][i]);
      }
    });
    __nv_bfloat16* const row = dst + prow * LdS + slot0;
#pragma unroll
    for (int j = 0; j < PM::SPT; j += 2) {
      uint32_t acc = ((uint32_t)lv[D - 1][PM::idx_of(D - 1, j + 1)] << 16)
                     | (uint32_t)lv[D - 1][PM::idx_of(D - 1, j)];
      static_for<D - 1>([&](auto Lc) {
        constexpr int l = decltype(Lc)::value;
        acc = mul_bf16x2(acc, splat_bf16x2_bits(lv[l][PM::idx_of(l, j)]));
      });
      store_bf16x2(row + j, mul_bf16x2(acc, gain));
    }
  }

  //: `gain == 0` zeroes a padding row, so a short batch needs no separate path.
  template <bool SAMP, GainSite GAIN>
  __device__ __forceinline__ void produce_batch(const __nv_bfloat16* base, __nv_bfloat16* dst,
                                                const uint16_t* list, int o0, int ntot,
                                                uint32_t winP, uint32_t winG, int par) {
    const int nb = (ntot - o0) < C ? (ntot - o0) : C;
    const bool have = prow < nb;
    const int tok = list[o0 + (have ? prow : nb - 1)];
    uint32_t gain = 0x3F803F80u;
    if constexpr (GAIN == GainSite::kPanel)
      gain = splat_bf16x2(
          *reinterpret_cast<const __nv_bfloat16*>(gather_ptr(gW, winG + (uint32_t)tok * 2u)));
    if constexpr (GAIN == GainSite::kTokenAxis) {
      if (rIn == 0)
        sgain[par * C + prow] = __bfloat162float(
            *reinterpret_cast<const __nv_bfloat16*>(gather_ptr(gW, winG + (uint32_t)tok * 2u)));
    }
    produce<SAMP>(base, dst, winP + (uint32_t)tok * (uint32_t)(BP::kWtot * 2), have ? gain : 0u,
                  par);
  }

  //: ONE `cols`-WIDE SLOT OF THE RING, gathered by the driving side's list.  The two
  //: planes have one layout, so one body stages either.
  __device__ __forceinline__ void stage_slot(const __nv_bfloat16* plane, const uint16_t* list,
                                             int b, int ntot, int slot, uint32_t winOff) {
    const int o0 = b * C;
    const int nb = (ntot - o0) < C ? (ntot - o0) : C;
    const int row = tid >> 2, cb = tid & 3;
    const int tok = list[o0 + (row < nb ? row : nb - 1)];
#pragma unroll
    for (int c = cb; c < kChunks; c += 4)
      stage_tile(
          smem_addr(sd + slot * kSlotElems + row * LdD + c * 8),
          gather_ptr(plane, (uint32_t)tok * (uint32_t)(LdD * 2) + winOff + (uint32_t)(c * 16)));
  }

  static __device__ __forceinline__ uint32_t page_off(uint32_t atom) {
    return atom * (uint32_t)(kPageElems * (int)sizeof(float));
  }

  __device__ __forceinline__ int slab_col(int q, int e) const {
    return kMmaN * q + (lane & 3) * 2 + (e & 1);
  }
  __device__ __forceinline__ int acc_row(int e) const { return (lane >> 2) + 8 * (e >> 1); }

  __device__ __forceinline__ void slab_zero(float (&slab)[kSlabM][kSlabN][4]) const {
#pragma unroll
    for (int pm = 0; pm < kSlabM; ++pm)
#pragma unroll
      for (int q = 0; q < kSlabN; ++q)
#pragma unroll
        for (int e = 0; e < 4; ++e) slab[pm][q][e] = 0.f;
  }

  //: THE LATTICE PAGE SWEEP, in the leaf-major orientation: atom `a` is warp `a>>1`'s
  //: m-tile `a&1`, because the carve slices the local index at a bit boundary.
  __device__ __forceinline__ void slab_load(float (&slab)[kSlabM][kSlabN][4], const float* src) {
    slab_zero(slab);
    if (src == nullptr) return;
    const float* const plane = pin_address(src);
#pragma unroll
    for (int a = 0; a < kAtoms; ++a) {
      const uint32_t base = page_off(atomBase + (uint32_t)a);
      int e = tid;
      int row = tid < kCols ? 0 : 1;
      int col = tid < kCols ? tid : tid - kCols;
#pragma unroll 1
      while (e < kPageElems) {
        so[row * LdO + col] = *reinterpret_cast<const float*>(
            gather_ptr(plane, base + (uint32_t)(e * (int)sizeof(float))));
        e += kThreads;
        col += kPageCarry;
        const bool wrap = col >= kCols;
        row += wrap ? 2 : 1;
        col -= wrap ? kCols : 0;
      }
      rendezvous();
      if (warp == (a >> 1)) {
        const int pm = a & 1;
#pragma unroll
        for (int q = 0; q < kSlabN; ++q)
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int c = slab_col(q, i);
            if (c < kCols) slab[pm][q][i] = so[acc_row(i) * LdO + c];
          }
      }
      rendezvous();
    }
  }

  __device__ __forceinline__ void slab_store(const float (&slab)[kSlabM][kSlabN][4], float* dst) {
    float* const plane = pin_address(dst);
#pragma unroll
    for (int a = 0; a < kAtoms; ++a) {
      rendezvous();
      if (warp == (a >> 1)) {
        const int pm = a & 1;
#pragma unroll
        for (int q = 0; q < kSlabN; ++q)
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int c = slab_col(q, i);
            if (c < kCols) store_f32(&so[acc_row(i) * LdO + c], slab[pm][q][i]);
          }
      }
      rendezvous();
      const uint32_t base = page_off(atomBase + (uint32_t)a);
      int e = tid;
      int row = tid < kCols ? 0 : 1;
      int col = tid < kCols ? tid : tid - kCols;
#pragma unroll 1
      while (e < kPageElems) {
        store_f32(
            reinterpret_cast<float*>(gather_ptr(plane, base + (uint32_t)(e * (int)sizeof(float)))),
            so[row * LdO + col]);
        e += kThreads;
        col += kPageCarry;
        const bool wrap = col >= kCols;
        row += wrap ? 2 : 1;
        col -= wrap ? kCols : 0;
      }
    }
  }

  //: THE GRADIENT TILE: the slab against the batch's `cols`-wide panel, contracted
  //: over CHANNELS -- `dR` in Scan A, `dW` in Scan B, one body.  `[leaf x token]`.
  __device__ __forceinline__ void channel_readout(const float (&slab)[kSlabM][kSlabN][4],
                                                  uint32_t bd, const __nv_bfloat16* panel,
                                                  float (&tile)[kSlabM][NTOK][4]) {
#pragma unroll
    for (int pm = 0; pm < kSlabM; ++pm) {
#pragma unroll
      for (int nt = 0; nt < NTOK; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) tile[pm][nt][e] = 0.f;
#pragma unroll
      for (int kc = 0; kc < kColKTiles; ++kc) {
        uint32_t a[4];
        a[0] = pack_bf16x2(slab[pm][2 * kc][0], slab[pm][2 * kc][1]);
        a[1] = pack_bf16x2(slab[pm][2 * kc][2], slab[pm][2 * kc][3]);
        a[2] = pack_bf16x2(slab[pm][2 * kc + 1][0], slab[pm][2 * kc + 1][1]);
        a[3] = pack_bf16x2(slab[pm][2 * kc + 1][2], slab[pm][2 * kc + 1][3]);
#pragma unroll
        for (int mt = 0; mt < MTOK; ++mt) {
          uint32_t b[4];
          load_frag(b, bd + (uint32_t)((mt * kMmaM * LdD + kc * 16) * 2));
#pragma unroll
          for (int pp = 0; pp < 2; ++pp) mma(tile[pm][2 * mt + pp], a, &b[2 * pp]);
        }
      }
      //: THE MASS CHANNEL RIDES THE SAME READOUT, and its PANEL entry is not 1: it is
      //: `d_den` on the seed side and 1 on the value side.  `cols` is not a whole number
      //: of k-tiles, so the term is a broadcast product rather than a fifth k-step.
      const float m0 = __shfl_sync(0xFFFFFFFFu, slab[pm][kSlabN - 1][0], lane & ~3);
      const float m1 = __shfl_sync(0xFFFFFFFFu, slab[pm][kSlabN - 1][2], lane & ~3);
#pragma unroll
      for (int nt = 0; nt < NTOK; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e)
          tile[pm][nt][e] +=
              ((e >> 1) ? m1 : m0)
              * __bfloat162float(panel[(kMmaN * nt + (lane & 3) * 2 + (e & 1)) * LdD + kCols - 1]);
    }
  }

  //: THE DIGIT PLANE IS CLEARED AT THE HEAD OF THE BATCH, so the leading rendezvous
  //: publishes it and the reduction below needs no barrier of its own.
  __device__ __forceinline__ void digit_zero() {
#pragma unroll
    for (int i = tid; i < kDigitRows * C; i += kThreads) sdg[i] = 0.f;
  }

  //: THE DIGIT REDUCTION.  A level's digit gradient is the tile's weighted marginal
  //: along that level's index axis, the weight being the OTHER levels' amplitude
  //: product -- one staged amplitude at `D = 2`, and never a division.  The staging is
  //: TOKEN-MAJOR because the drain below writes one contiguous digit run per token, and
  //: a digit-major buffer would make every one of those stores its own sector.
  __device__ __forceinline__ void digit_reduce(const float (&tile)[kSlabM][NTOK][4], int par) {
    constexpr int off0 = DP::off(0), off1 = DP::off(1);
    const float* const amp = samp + par * C * kDigitRows;
    //: inner level: one accumulator ROW is one digit, so the segment is an add.  All
    //: four warps hold every row, and the meeting place is a shared-memory RED rather
    //: than a round of barriers -- C6 accepts RED-order nondeterminism in gradients.
#pragma unroll
    for (int nt = 0; nt < NTOK; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int tokl = kMmaN * nt + (lane & 3) * 2 + (e & 1);
        float s = 0.f;
#pragma unroll
        for (int pm = 0; pm < kSlabM; ++pm)
          s += tile[pm][nt][e] * amp[tokl * kDigitRows + off0 + kSlabM * warp + pm];
        fan_in_add(&sdg[tokl * kDigitRows + off1 + acc_row(e)], s);
      }
    //: outer level: a whole m-tile is one digit and a warp owns its two exclusively, so
    //: the segment is three lane shuffles and no cross-warp term at all.
#pragma unroll
    for (int pm = 0; pm < kSlabM; ++pm)
#pragma unroll
      for (int nt = 0; nt < NTOK; ++nt)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          float acc = 0.f;
#pragma unroll
          for (int h = 0; h < 2; ++h)
            acc +=
                tile[pm][nt][2 * h + j]
                * amp[(kMmaN * nt + (lane & 3) * 2 + j) * kDigitRows + off1 + (lane >> 2) + 8 * h];
#pragma unroll
          for (int s = 4; s < 32; s <<= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, s);
          if (lane < 4)
            sdg[(kMmaN * nt + lane * 2 + j) * kDigitRows + off0 + kSlabM * warp + pm] += acc;
        }
  }

  template <bool GAINED>
  __device__ __forceinline__ void digit_drain(float* plane, const uint16_t* list, int o0, int n,
                                              uint32_t winP, int par) {
#pragma unroll
    for (int i = tid; i < kDigitRows * C; i += kThreads) {
      const int tokl = i / kDigitRows;
      if (tokl >= n) continue;
      float g = sdg[i];
      if constexpr (GAINED) g *= sgain[par * C + tokl];
      fan_in_add(reinterpret_cast<float*>(
                     gather_ptr(plane, winP + (uint32_t)list[o0 + tokl] * (uint32_t)(D * B * 4)
                                           + (uint32_t)(sdcol[i - tokl * kDigitRows] * 4))),
                 g);
    }
  }
};

// ------------------------------------------------------------------ Scan B

//: THE DESCENDING ADJOINT.  Per window: the WRITE side against the adjoint at window
//: EXIT, then the fold of that window's own reads.  The reverse ordering law is the
//: mirror of the forward's, and it is what keeps the within-window pairs out.
template <int D, int B, int K, int M, int W, int C>
__global__ __launch_bounds__(kThreads, (kScanBCtaTarget<D, B, K, M, W, C>)) void scan_b_kernel(
    BwdParams p) {
  using G = BwdGeom<D, B, K, M, W, C>;
  constexpr int STAGES = G::kStagesB;
  using OW = BwdOwner<D, B, K, M, W, C, STAGES>;
  using BP = typename OW::BP;
  constexpr int LdS = OW::LdS, LdD = OW::LdD, LdO = OW::LdO;
  constexpr int kSlabM = OW::kSlabM, NTOK = OW::NTOK, MTOK = OW::MTOK;

  extern __shared__ __align__(16) char bwd_smem[];
  OW ow(p, bwd_smem);

  float lam[kSlabM][kSlabN][4];
  ow.slab_load(lam, p.slab_in);
  rendezvous();

  float* const dpwP = pin_address(p.d_pw + (long)ow.bh * p.L * (D * B));
  float* const dgwP = pin_address(p.d_gw + (long)ow.bh * p.L);
  float* const dvP = pin_address(p.d_v + (long)ow.bh * p.L * kDv);
  const int nWindows = (p.L + W - 1) / W;

#pragma unroll 1
  for (int win = nWindows - 1; win >= 0; --win) {
    const int tok0 = win * W;
    const int wLen = (p.L - tok0) < W ? (p.L - tok0) : W;
    const uint32_t winP = (uint32_t)tok0 * (uint32_t)(BP::kWtot * 2);
    const uint32_t winG = (uint32_t)tok0 * 2u;
    const uint32_t winD = (uint32_t)tok0 * (uint32_t)(LdD * 2);
    const uint32_t winA = (uint32_t)tok0 * (uint32_t)(D * B * 4);
    const uint32_t winV = (uint32_t)tok0 * (uint32_t)(kDv * 4);
    const uint32_t winW = (uint32_t)tok0 * 4u;
    ow.liveness(tok0, wLen);
    if (ow.nr == 0 && ow.nw == 0) continue;

    // ---- PHASE 1': the write side, against the window's EXIT adjoint ---------
    const uint16_t* const wlist = ow.slist + OW::kListStride;
    const int nbw = (ow.nw + C - 1) / C;
#pragma unroll 1
    for (int sst = 0; sst < STAGES - 1; ++sst) {
      if (sst < nbw) ow.stage_slot(ow.vtP, wlist, sst, ow.nw, sst % STAGES, winD);
      stage_commit();
    }
    if (nbw > 0)
      ow.template produce_batch<true, GainSite::kTokenAxis>(ow.pW, ow.sp0, wlist, 0, ow.nw, winP,
                                                            winG, 0);
#pragma unroll 1
    for (int i = 0; i < nbw; ++i) {
      const int off = i * C;
      const int n = (ow.nw - off) < C ? (ow.nw - off) : C;
      const int par = i & 1;
      const uint32_t bp = par ? ow.bpF1 : ow.bpF0;
      const uint32_t bd = ow.bdT + (uint32_t)(i % STAGES) * OW::kSlotBytes;
      const int vslot = i % STAGES;
      ow.digit_zero();
      stage_wait<STAGES - 2>();
      rendezvous();
      const int f = i + STAGES - 1;
      if (f < nbw) ow.stage_slot(ow.vtP, wlist, f, ow.nw, f % STAGES, winD);
      stage_commit();
      if (i + 1 < nbw)
        ow.template produce_batch<true, GainSite::kTokenAxis>(
            ow.pW, (par ? ow.sp0 : ow.sp1), wlist, (i + 1) * C, ow.nw, winP, winG, par ^ 1);

      //: (G4) then (G7): one tile, one weighted marginal per level.
      float tile[kSlabM][NTOK][4];
      ow.channel_readout(lam, bd, ow.sd + vslot * OW::kSlotElems, tile);
      ow.digit_reduce(tile, par);
      rendezvous();
      ow.template digit_drain<true>(dpwP, wlist, off, n, winA, par);

      //: (G5) and (G6) off ONE accumulator: the panel is UNGAINED, so the value
      //: columns times `gw` are `d_v` and the same accumulator dotted with the ones-
      //: column-bearing value row is `d_gw`.  `sum_s dW` is not `d_gw` and is never
      //: formed.  `k = leaf` is the carved axis, so this is the backward's one
      //: cross-warp term -- four rounds through the output staging buffer.
      float ung[MTOK][kSlabN][4];
#pragma unroll
      for (int mt = 0; mt < MTOK; ++mt)
#pragma unroll
        for (int q = 0; q < kSlabN; ++q)
#pragma unroll
          for (int e = 0; e < 4; ++e) ung[mt][q][e] = 0.f;
#pragma unroll
      for (int kk = 0; kk < kSlabM; ++kk) {
        uint32_t a[MTOK][4];
#pragma unroll
        for (int mt = 0; mt < MTOK; ++mt)
          load_frag(
              a[mt],
              bp + (uint32_t)((mt * kMmaM * LdS + (OW::kLeafGrp * ow.warp + 2 * kk) * 8) * 2));
#pragma unroll
        for (int q = 0; q < kSlabN; ++q) {
          const uint32_t b2[2] = {transpose_frag_b16(pack_bf16x2(lam[kk][q][0], lam[kk][q][1])),
                                  transpose_frag_b16(pack_bf16x2(lam[kk][q][2], lam[kk][q][3]))};
#pragma unroll
          for (int mt = 0; mt < MTOK; ++mt) mma(ung[mt][q], a[mt], b2);
        }
      }
      //: `k = leaf` is the carved axis, so this is the backward's one cross-warp term:
      //: four warp partials of `[C x cols]` meeting in the output staging buffer.  The
      //: meeting is BARRIER-SEPARATED ROUNDS rather than a shared-memory RED: every
      //: warp writes the same cells, so a RED here is a four-way conflict on all
      //: `cols` of every token and measures worse than the three extra barriers.
#pragma unroll 1
      for (int round = 0; round < kWarps; ++round) {
        if (ow.warp == round) {
#pragma unroll
          for (int mt = 0; mt < MTOK; ++mt)
#pragma unroll
            for (int q = 0; q < kSlabN; ++q)
#pragma unroll
              for (int e = 0; e < 4; ++e) {
                const int c = ow.slab_col(q, e);
                if (c < kCols) {
                  float* const cell = &ow.so[(mt * kMmaM + ow.acc_row(e)) * LdO + c];
                  if (round == 0)
                    store_f32(cell, ung[mt][q][e]);
                  else
                    *cell += ung[mt][q][e];
                }
              }
        }
        rendezvous();
      }
      const int fanRow = ow.tid / kBallotLanes, fanCol = ow.tid % kBallotLanes;
#pragma unroll
      for (int pp = 0; pp < OW::kFanPasses; ++pp) {
        const int row = fanRow + pp * OW::kFanRows;
        if (row < n) {
          float* const dp = reinterpret_cast<float*>(gather_ptr(
              dvP,
              winV + (uint32_t)wlist[off + row] * (uint32_t)(kDv * 4) + (uint32_t)(fanCol * 4)));
          const float g = ow.sgain[par * C + row];
#pragma unroll
          for (int e = 0; e < OW::kFanPerThr; ++e)
            fan_in_add(dp + e * kBallotLanes, g * ow.so[row * LdO + fanCol + e * kBallotLanes]);
        }
      }
#pragma unroll
      for (int r = 0; r < OW::kGwRows; ++r) {
        const int row = ow.warp * OW::kGwRows + r;
        const __nv_bfloat16* const vrow = ow.sd + vslot * OW::kSlotElems + row * LdD;
        float s = 0.f;
        if (row < n)
          s = ow.so[row * LdO + 2 * ow.lane] * __bfloat162float(vrow[2 * ow.lane])
              + ow.so[row * LdO + 2 * ow.lane + 1] * __bfloat162float(vrow[2 * ow.lane + 1]);
#pragma unroll
        for (int q = 1; q < 32; q <<= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, q);
        if (ow.lane == 0 && row < n)
          fan_in_add(
              reinterpret_cast<float*>(gather_ptr(dgwP, winW + (uint32_t)wlist[off + row] * 4u)),
              s + ow.so[row * LdO + kDv]);
      }
      rendezvous();
    }
    stage_wait<0>();

    // ---- PHASE 2': the fold of the adjoint, gathered by the READ list --------
    const int nbr = (ow.nr + C - 1) / C;
#pragma unroll 1
    for (int sst = 0; sst < STAGES - 1; ++sst) {
      if (sst < nbr) ow.stage_slot(ow.dnP, ow.slist, sst, ow.nr, sst % STAGES, winD);
      stage_commit();
    }
    if (nbr > 0)
      ow.template produce_batch<false, GainSite::kNone>(ow.pR, ow.sp0, ow.slist, 0, ow.nr, winP,
                                                        winG, 0);
#pragma unroll 1
    for (int i = 0; i < nbr; ++i) {
      const int par = i & 1;
      const uint32_t bp = par ? ow.bpT1 : ow.bpT0;
      const uint32_t bd = ow.bdF + (uint32_t)(i % STAGES) * OW::kSlotBytes;
      stage_wait<STAGES - 2>();
      rendezvous();
      const int f = i + STAGES - 1;
      if (f < nbr) ow.stage_slot(ow.dnP, ow.slist, f, ow.nr, f % STAGES, winD);
      stage_commit();
      if (i + 1 < nbr)
        ow.template produce_batch<false, GainSite::kNone>(ow.pR, (par ? ow.sp0 : ow.sp1), ow.slist,
                                                          (i + 1) * C, ow.nr, winP, winG, par ^ 1);
#pragma unroll
      for (int kt = 0; kt < MTOK; ++kt)
#pragma unroll
        for (int pm = 0; pm < kSlabM; ++pm) {
          uint32_t a[4];
          load_frag_t(
              a, bp + (uint32_t)((kt * kMmaM * LdS + (OW::kLeafGrp * ow.warp + 2 * pm) * 8) * 2));
#pragma unroll
          for (int qb = 0; qb < (kSlabN + 1) / 2; ++qb) {
            uint32_t b[4];
            load_frag_t(b, bd + (uint32_t)((kt * kMmaM * LdD + qb * 16) * 2));
            mma(lam[pm][2 * qb], a, &b[0]);
            if (2 * qb + 1 < kSlabN) mma(lam[pm][2 * qb + 1], a, &b[2]);
          }
        }
      rendezvous();
    }
    stage_wait<0>();
  }

  //: (G8): the slab at the end of the descending walk IS `d_state_in`.
  if (p.slab_out != nullptr) ow.slab_store(lam, p.slab_out);
}

// ------------------------------------------------------------------ Scan A

//: THE ASCENDING REPLAY.  Per window: the read side against the window's ENTRY state,
//: then the forward's own fold of that window's writes -- so the slab at the end of
//: the walk is the forward's `state_out`, which is the rung's cheapest strong gate.
template <int D, int B, int K, int M, int W, int C>
__global__ __launch_bounds__(kThreads, (kScanACtaTarget<D, B, K, M, W, C>)) void scan_a_kernel(
    BwdParams p) {
  using G = BwdGeom<D, B, K, M, W, C>;
  constexpr int STAGES = G::kStagesA;
  using OW = BwdOwner<D, B, K, M, W, C, STAGES>;
  using BP = typename OW::BP;
  constexpr int LdS = OW::LdS, LdD = OW::LdD;
  constexpr int kSlabM = OW::kSlabM, NTOK = OW::NTOK, MTOK = OW::MTOK;

  extern __shared__ __align__(16) char bwd_smem[];
  OW ow(p, bwd_smem);

  float st[kSlabM][kSlabN][4];
  ow.slab_load(st, p.slab_in);
  rendezvous();

  float* const dprP = pin_address(p.d_pr + (long)ow.bh * p.L * (D * B));
  const int nWindows = (p.L + W - 1) / W;

#pragma unroll 1
  for (int win = 0; win < nWindows; ++win) {
    const int tok0 = win * W;
    const int wLen = (p.L - tok0) < W ? (p.L - tok0) : W;
    const uint32_t winP = (uint32_t)tok0 * (uint32_t)(BP::kWtot * 2);
    const uint32_t winG = (uint32_t)tok0 * 2u;
    const uint32_t winD = (uint32_t)tok0 * (uint32_t)(LdD * 2);
    const uint32_t winA = (uint32_t)tok0 * (uint32_t)(D * B * 4);
    ow.liveness(tok0, wLen);
    if (ow.nr == 0 && ow.nw == 0) continue;

    // ---- PHASE 1: the read side, against the window's ENTRY state ------------
    const int nbr = (ow.nr + C - 1) / C;
#pragma unroll 1
    for (int sst = 0; sst < STAGES - 1; ++sst) {
      if (sst < nbr) ow.stage_slot(ow.dnP, ow.slist, sst, ow.nr, sst % STAGES, winD);
      stage_commit();
    }
    if (nbr > 0)
      ow.template produce_batch<true, GainSite::kNone>(ow.pR, ow.sp0, ow.slist, 0, ow.nr, winP,
                                                       winG, 0);
#pragma unroll 1
    for (int i = 0; i < nbr; ++i) {
      const int off = i * C;
      const int n = (ow.nr - off) < C ? (ow.nr - off) : C;
      const int par = i & 1;
      const uint32_t bd = ow.bdT + (uint32_t)(i % STAGES) * OW::kSlotBytes;
      ow.digit_zero();
      stage_wait<STAGES - 2>();
      rendezvous();
      const int f = i + STAGES - 1;
      if (f < nbr) ow.stage_slot(ow.dnP, ow.slist, f, ow.nr, f % STAGES, winD);
      stage_commit();
      if (i + 1 < nbr)
        ow.template produce_batch<true, GainSite::kNone>(ow.pR, (par ? ow.sp0 : ow.sp1), ow.slist,
                                                         (i + 1) * C, ow.nr, winP, winG, par ^ 1);
      float tile[kSlabM][NTOK][4];
      ow.channel_readout(st, bd, ow.sd + (i % STAGES) * OW::kSlotElems, tile);
      ow.digit_reduce(tile, par);
      rendezvous();
      ow.template digit_drain<false>(dprP, ow.slist, off, n, winA, par);
      rendezvous();
    }
    stage_wait<0>();

    // ---- PHASE 2: the forward's own fold, replayed --------------------------
    const uint16_t* const wlist = ow.slist + OW::kListStride;
    const int nbw = (ow.nw + C - 1) / C;
#pragma unroll 1
    for (int sst = 0; sst < STAGES - 1; ++sst) {
      if (sst < nbw) ow.stage_slot(ow.vtP, wlist, sst, ow.nw, sst % STAGES, winD);
      stage_commit();
    }
    if (nbw > 0)
      ow.template produce_batch<false, GainSite::kPanel>(ow.pW, ow.sp0, wlist, 0, ow.nw, winP, winG,
                                                         0);
#pragma unroll 1
    for (int i = 0; i < nbw; ++i) {
      const int par = i & 1;
      const uint32_t bp = par ? ow.bpT1 : ow.bpT0;
      const uint32_t bd = ow.bdF + (uint32_t)(i % STAGES) * OW::kSlotBytes;
      stage_wait<STAGES - 2>();
      rendezvous();
      const int f = i + STAGES - 1;
      if (f < nbw) ow.stage_slot(ow.vtP, wlist, f, ow.nw, f % STAGES, winD);
      stage_commit();
      if (i + 1 < nbw)
        ow.template produce_batch<false, GainSite::kPanel>(ow.pW, (par ? ow.sp0 : ow.sp1), wlist,
                                                           (i + 1) * C, ow.nw, winP, winG, par ^ 1);
#pragma unroll
      for (int kt = 0; kt < MTOK; ++kt)
#pragma unroll
        for (int pm = 0; pm < kSlabM; ++pm) {
          uint32_t a[4];
          load_frag_t(
              a, bp + (uint32_t)((kt * kMmaM * LdS + (OW::kLeafGrp * ow.warp + 2 * pm) * 8) * 2));
#pragma unroll
          for (int qb = 0; qb < (kSlabN + 1) / 2; ++qb) {
            uint32_t b[4];
            load_frag_t(b, bd + (uint32_t)((kt * kMmaM * LdD + qb * 16) * 2));
            mma(st[pm][2 * qb], a, &b[0]);
            if (2 * qb + 1 < kSlabN) mma(st[pm][2 * qb + 1], a, &b[2]);
          }
        }
      rendezvous();
    }
    stage_wait<0>();
  }

  //: THE REPLAY GATE'S OUTPUT: the same state the forward's own walk ends with.
  if (p.slab_out != nullptr) ow.slab_store(st, p.slab_out);
}

}  // namespace rola::carry
