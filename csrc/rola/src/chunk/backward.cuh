// csrc/rola/src/chunk/backward.cuh -- SCAN B, the reverse adjoint of the chunk
// consumer (P65).  This is the STAGE-1 SKELETON: it carries the two adjoint
// accumulators, the descending walk, both operand panels and the one new
// arithmetic primitive (a contraction whose k axis is d_v), and it deliberately
// does NOT yet carry the Gram, the transposed staircase or the digit reduction.
// What it computes, what it omits, why the walk needs no reversed table, and the
// register/shared arithmetic it was built to measure: docs/internals/chunk/backward.md
#pragma once

#include "chunk_kernel.cuh"

namespace rola::chunk {

//: Scan B's launch arguments.  `dnum`/`dden` are the readout seeds, already bf16
//: because both are mma operands; `dnum` is `[BH, L, d_v]` and its row stride is
//: therefore d_v, unlike V's token-major one.
struct ScanBParams {
  const __nv_bfloat16 *pr, *pw, *gw, *v, *dnum, *dden;
  const uint32_t* tbl;
  float *dv, *dgw;
  int BH, L;
  int v_row_bytes;
};

//: SCAN B'S GEOMETRY.  It is NOT `Geom`: the ring carries TWO value-shaped
//: panels (`dnum` gathered by the read list, V by the write list), there is no
//: Gram, and there is a cross-unit window the forward has no term for.
template <int C, int BC>
struct GeomB {
  static constexpr int GRP = groups_for(BC);
  static constexpr int BCg = BC / GRP;
  static constexpr int LdV = kD + kPadE;
  static constexpr int LdS = BCg + kPadE;
  static constexpr int LdO = kD + kPadF;
  static constexpr int hdr = 16;
  static constexpr int idx_slot = hdr + ((5 * C + 15) / 16) * 16;
  static constexpr int slot = 2 * C * LdV * (int)sizeof(__nv_bfloat16) + idx_slot;
  static constexpr int form_bytes =
      2 * kFormWords * (int)sizeof(uint32_t) + kCookSlots * (int)sizeof(int);
  //: THE CHANNEL CONTRACTION'S k AXIS IS d_v, WHICH IS THE CARVE'S OWN AXIS, so
  //: every unit holds a PARTIAL and the sum across units is structural.  One row
  //: per unit, one column per token, is the whole of it.
  static constexpr int grad_bytes = kUnits * C * (int)sizeof(float);
  static constexpr int fixed = GRP * (2 * C * LdS) * (int)sizeof(__nv_bfloat16)
                               + GRP * C * LdO * (int)sizeof(float) + grad_bytes
                               + (1 + kAheadExtra) * idx_slot + form_bytes;
  static constexpr int stages = design::stages_for(slot, fixed, threads_for(BC), reg_floor(C, BC));
  static constexpr size_t bytes = (size_t)stages * slot + fixed;
};

//: THE CTA TARGET SCAN B IS BOUND AT, and it is checked against the forward's at
//: the instantiation site: a backward that resident-fewer CTAs than its forward
//: would move the arm's occupancy without anybody deciding to.
template <int C, int BC>
constexpr int kScanBCtaTarget =
    design::cta_target((int)GeomB<C, BC>::bytes, threads_for(BC), reg_floor(C, BC));

//: THE FORM, DESCENDING.  `form_impl`'s body with the token axis walked the
//: other way: `formLow` is the LOWEST word formed, so the ring holds
//: `[formLow, formLow + kFormWords)` and refilling means lowering it.
template <int kClauses, int kFormWordsT, int kThreadsT>
__device__ __forceinline__ void form_desc_impl(int& formLow, int limit, int tid,
                                               const uint32_t* rowR, const uint32_t* rowW,
                                               uint32_t cA, uint32_t cB, uint32_t cC, uint32_t cD,
                                               uint32_t tailMask, int lastWord, uint32_t* sfR,
                                               uint32_t* sfW) {
  for (int w = formLow - 1 - tid; w >= limit; w -= kThreadsT) {
    const uint32_t off = (uint32_t)w * (uint32_t)sizeof(uint32_t);
    uint32_t rm, wm;
    if constexpr (kClauses == 0) {
      rm = 0xFFFFFFFFu;
      wm = 0xFFFFFFFFu;
    } else if constexpr (kClauses == 1) {
      rm = *reinterpret_cast<const uint32_t*>(gather_ptr(rowR, cA + off));
      wm = *reinterpret_cast<const uint32_t*>(gather_ptr(rowW, cA + off));
    } else if constexpr (kClauses == 2) {
      rm = *reinterpret_cast<const uint32_t*>(gather_ptr(rowR, cA + off))
           & *reinterpret_cast<const uint32_t*>(gather_ptr(rowR, cB + off));
      wm = *reinterpret_cast<const uint32_t*>(gather_ptr(rowW, cA + off))
           & *reinterpret_cast<const uint32_t*>(gather_ptr(rowW, cB + off));
    } else {
      rm = *reinterpret_cast<const uint32_t*>(gather_ptr(rowR, cA + off))
           & *reinterpret_cast<const uint32_t*>(gather_ptr(rowR, cB + off))
           & *reinterpret_cast<const uint32_t*>(gather_ptr(rowR, cC + off));
      wm = *reinterpret_cast<const uint32_t*>(gather_ptr(rowW, cA + off))
           & *reinterpret_cast<const uint32_t*>(gather_ptr(rowW, cB + off))
           & *reinterpret_cast<const uint32_t*>(gather_ptr(rowW, cC + off));
      if constexpr (kClauses > 3) {
        rm &= *reinterpret_cast<const uint32_t*>(gather_ptr(rowR, cD + off));
        wm &= *reinterpret_cast<const uint32_t*>(gather_ptr(rowW, cD + off));
      }
    }
    if (w == lastWord) {
      rm &= tailMask;
      wm &= tailMask;
    }
    const int slot = w & (kFormWordsT - 1);
    sfR[slot] = rm;
    sfW[slot] = wm;
  }
  formLow = limit;
}

//: SCAN B.  One CTA per (bh, block), full grid: the sparse grid and the carried
//: state are not axes of the skeleton.
template <int C, int D, int B, int BC>
__global__ __launch_bounds__(threads_for(BC), (kScanBCtaTarget<C, BC>)) void scan_b_kernel(
    const __nv_bfloat16* __restrict__ pread, const __nv_bfloat16* __restrict__ pwrite,
    const __nv_bfloat16* __restrict__ gwrite, const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ dnum, const __nv_bfloat16* __restrict__ dden,
    const uint32_t* __restrict__ tbl, float* __restrict__ dv, float* __restrict__ dgw, int L,
    int v_row_bytes) {
  using LP = LevelPlan<D, B, BC>;
  using G = GeomB<C, BC>;
  constexpr int GROUPS = groups_for(BC);
  constexpr int kThreads = threads_for(BC);
  constexpr int kGroupWarps = kGroupThreads / 32;
  constexpr int BCg = BC / GROUPS;
  static_assert(GROUPS == 1, "the S1 skeleton is written for the one-group carve");
  static_assert(C % kMmaM == 0, "chunk must be a multiple of the mma m quantum");
  static_assert(BCg % kMmaK == 0, "a group's slice must be a whole number of mma k-tiles");
  static_assert(C <= 255, "a causal threshold is one byte, chunk-relative");
  static_assert(design::smem_fits(G::bytes),
                "this arm does not fit this architecture's per-CTA shared-memory maximum");
  static_assert(kScanBCtaTarget<C, BC> == kCtaTarget<C, BC>,
                "SCAN B MOVED THE ARM'S OCCUPANCY: its shared-memory carve reaches a "
                "different CTA target than the forward's, so the two halves of one "
                "training step would run at different residencies. Re-carve rather "
                "than accept it.");
  constexpr int STAGES = G::stages;
  constexpr int AHEAD = STAGES - 1 + kAheadExtra;
  constexpr int IRING = AHEAD + 2;
  constexpr int LdV = G::LdV, LdS = G::LdS, LdO = G::LdO;
  constexpr int MTOK = C / kMmaM;
  constexpr int NTOK = C / kMmaN;
  constexpr int KD = BCg / kMmaK;
  constexpr int NDV = BCg / kMmaN;
  constexpr int kChunkV = kD / 8;

  constexpr int kMassUnits = KD < kUnits ? KD : kUnits;
  constexpr int kMassKD = KD / kMassUnits;
  constexpr int kMassTiles = 2 * kMassKD;
  static_assert(KD % kMassUnits == 0 && NDV == kMassUnits * kMassTiles,
                "the mass channel's slots must be carved across the units exactly");

  // ---- coefficient production geometry, the forward's -----------------------
  constexpr int kWtot = LP::col_base(LP::depth);
  constexpr int kDepth = LP::depth;
  constexpr int kInnerSpan = LP::span_at(kDepth - 1);
  constexpr int kOuterSpan = BC / kInnerSpan;
  constexpr int kLeadSpanG = kOuterSpan / GROUPS;
  constexpr int kRowThr = kGroupThreads / C;
  constexpr int kThrPerLead = kRowThr >= kLeadSpanG ? kRowThr / kLeadSpanG : 1;
  constexpr int kLeadPerThr = kLeadSpanG >= kRowThr ? kLeadSpanG / kRowThr : 1;
  constexpr int kTailPerThr = kInnerSpan / kThrPerLead;
  constexpr int kTailVec = (kTailPerThr * 2) / 16;
  constexpr int kFastSpan = kDepth >= 2 ? LP::span_at(kDepth - 2) : 1;
  static_assert(kLeadPerThr * kTailPerThr * kRowThr == BCg,
                "the row's threads must cover the slice exactly once");
  static_assert(kTailPerThr % 8 == 0,
                "a thread's trailing run is published in 16-byte vectors, so it "
                "must be a whole number of eight coefficients");
  static_assert(kLeadPerThr == 1 || kFastSpan % kLeadPerThr == 0,
                "a thread's outer run crosses a fast-level boundary");

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int gwarp = warp % kGroupWarps;
  const int gtid = tid % kGroupThreads;
  const int blk = (int)blockIdx.x;

  extern __shared__ __align__(16) char smem_raw[];

  const long base = (long)blockIdx.y * L * kD;
  const long pbase = (long)blockIdx.y * L * kWtot;
  const int tokWords = tok_words(L);
  const int vRowElems = v_row_bytes / (int)sizeof(__nv_bfloat16);
  const int heads = vRowElems / kD;
  const __nv_bfloat16* const vPlane =
      pin_address(v + (long)(blockIdx.y / heads) * L * vRowElems + (long)(blockIdx.y % heads) * kD);
  const __nv_bfloat16* const dnumPlane = pin_address(dnum + base);
  const __nv_bfloat16* const ddenPlane = pin_address(dden + (long)blockIdx.y * L);
  float* const dvPlane = pin_address(dv + base);
  float* const dgwPlane = pin_address(dgw + (long)blockIdx.y * L);
  const __nv_bfloat16* const pRead = pin_address(pread + pbase);
  const __nv_bfloat16* const pWrite = pin_address(pwrite + pbase);
  const __nv_bfloat16* const gWrite = pin_address(gwrite + (long)blockIdx.y * L);
  const uint32_t* const rowR =
      pin_address(tbl + ((long)(2 * blockIdx.y + 0) * LP::entries) * tokWords);
  const uint32_t* const rowW =
      pin_address(tbl + ((long)(2 * blockIdx.y + 1) * LP::entries) * tokWords);
  const uint32_t clauseStride = (uint32_t)tokWords * (uint32_t)sizeof(uint32_t);

  __nv_bfloat16* sv = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* sdn = sv + STAGES * C * LdV;
  __nv_bfloat16* sr = sdn + STAGES * C * LdV;
  __nv_bfloat16* sw = sr + C * LdS;
  float* so = reinterpret_cast<float*>(sw + C * LdS);
  float* sgrad = so + C * LdO;
  char* sidx = reinterpret_cast<char*>(sgrad + kUnits * C);
  uint32_t* sfR = reinterpret_cast<uint32_t*>(sidx + IRING * G::idx_slot);
  uint32_t* sfW = sfR + kFormWords;
  int* scook = reinterpret_cast<int*>(sfW + kFormWords);
  static_assert((2 * C * LdS) % 8 == 0, "the fp32 region must start 16-byte aligned");
  static_assert((C * LdO + kUnits * C) % 4 == 0, "the index ring must start 16-byte aligned");
  static_assert(G::idx_slot % 16 == 0,
                "the formed schedule follows the index ring and must stay aligned");

  auto ihdr = [&](int d) { return reinterpret_cast<int*>(sidx + d * G::idx_slot); };
  auto iread = [&](int d) { return reinterpret_cast<uint16_t*>(sidx + d * G::idx_slot + G::hdr); };
  auto iwrite = [&](int d) { return iread(d) + C; };
  auto ithr = [&](int d) {
    return reinterpret_cast<uint8_t*>(sidx + d * G::idx_slot + G::hdr + 4 * C);
  };

  // ---- lane-invariant operand addressing, the forward's ---------------------
  const int lr = lane & 7, tl = lane >> 3;
  const int rF = (tl & 1) * 8 + lr, cF = tl >> 1;
  const int rT = (tl >> 1) * 8 + lr, cT = tl & 1;
  const uint32_t rowFs = (uint32_t)((rF * LdS + cF * 8) * 2);
  const uint32_t rowTs = (uint32_t)((rT * LdS + cT * 8) * 2);
  const uint32_t rowFv = (uint32_t)((rF * LdV + cF * 8) * 2);
  const uint32_t rowTv = (uint32_t)((rT * LdV + cT * 8) * 2);
  //: A PANEL ADDRESS IS A BASE PLUS A DISPLACEMENT, and which of the two
  //: fragment maps it serves is carried by the BASE (`rowFs`/`rowTs`), never by
  //: the displacement -- so one displacement form covers both.
#define ADDR_S(b_, R_, g_) ((b_) + (uint32_t)(((R_) * LdS + (g_) * 8) * 2))
  const uint32_t massOff = (uint32_t)(gwarp * kMmaK * (int)sizeof(__nv_bfloat16));
#define ADDR_S_M(b_, R_, q_) \
  ((b_) + (uint32_t)(((R_) * LdS + (2 * (q_) * kMassUnits) * 8) * 2) + massOff)

  const int dv0 = gwarp * 16;
#define ADDR_V(b_, R_) ((b_) + (uint32_t)(((R_) * LdV + dv0) * 2))

  //: THE ADJOINT STATE, transposed exactly as the forward's state is:
  //: m = d_v (this unit's 16 rows), n = this block's slots.
  float dst[NDV][4];
#pragma unroll
  for (int j = 0; j < NDV; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) dst[j][e] = 0.f;

  float dz_st[kMassTiles][4];
#pragma unroll
  for (int j = 0; j < kMassTiles; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) dz_st[j][e] = 0.f;

  // ---- global -> shared staging: V by the WRITE list, dnum by the READ list --
  constexpr int kRowsPerPass = kThreads / kChunkV;
  constexpr int kPasses = C / kRowsPerPass;
  static_assert(kPasses >= 1, "a chunk must be at least kThreads/8 rows");
  const int ldRow = tid / kChunkV, ldCol = (tid % kChunkV) * 8;
  const uint32_t ldColB = (uint32_t)(ldCol * 2);
  const uint32_t kVRowB = (uint32_t)v_row_bytes;
  constexpr uint32_t kDnRowB = (uint32_t)(kD * (int)sizeof(__nv_bfloat16));
  const uint32_t stgOff = (uint32_t)((ldRow * LdV + ldCol) * 2);

  constexpr uint32_t kSlot = (uint32_t)(C * LdV) * 2;
  constexpr uint32_t kRing = STAGES * kSlot;
  const uint32_t svL = smem_addr(sv) + stgOff;
  const uint32_t sdnL = smem_addr(sdn) + stgOff;
  const uint32_t svB = smem_addr(sv);
  const uint32_t sdnB = smem_addr(sdn);
  uint32_t stgSlot = 0, cmpSlot = 0;

  const uint32_t brF = smem_addr(sr) + rowFs;
  const uint32_t bwT = smem_addr(sw) + rowTs;

  // ---- the fan-in's addressing, the forward's ------------------------------
  constexpr int kFanLanes = 32;
  constexpr int kFanPerThr = kD / kFanLanes;
  constexpr int kFanRows = kGroupThreads / kFanLanes;
  constexpr int kFanPasses = C / kFanRows;
  static_assert(kFanPasses * kFanRows == C && kFanPasses >= 1,
                "the fan-in's passes must cover the chunk exactly");
  const int fanRow = gtid / kFanLanes, fanCol = gtid % kFanLanes;
  const uint32_t fanColB = (uint32_t)(fanCol * (int)sizeof(float));
  constexpr uint32_t kORowB = (uint32_t)(kD * (int)sizeof(float));
  const int fanE = fanRow * LdO + fanCol;

  const int oq = (lane & 3) * 2, om = lane >> 2;
  int oE[2][2];
#pragma unroll
  for (int e = 0; e < 2; ++e)
#pragma unroll
    for (int f = 0; f < 2; ++f) oE[e][f] = (oq + e) * LdO + dv0 + om + f * 8;

  //: THE STAGE IS WHAT IS SKIPPED; THE RING SLOT IS NOT.  Both panels ride one
  //: slot cursor, so the wait/commit accounting stays the forward's.
  auto stage_chunk = [&](int d, bool goW, bool goR) {
    const uint32_t win = (uint32_t)ihdr(d)[3];
    if (goW) {
      const uint16_t* lw = iwrite(d);
      const uint32_t winCol = ldColB + win * (kWindow * kVRowB);
#pragma unroll
      for (int p = 0; p < kPasses; ++p) {
        const uint32_t off = (uint32_t)lw[ldRow + p * kRowsPerPass] * kVRowB + winCol;
        stage_tile(svL + stgSlot + (uint32_t)(p * kRowsPerPass * LdV * 2), gather_ptr(vPlane, off));
      }
    }
    if (goR) {
      const uint16_t* lrd = iread(d);
      const uint32_t winCol = ldColB + win * (kWindow * kDnRowB);
#pragma unroll
      for (int p = 0; p < kPasses; ++p) {
        const uint32_t off = (uint32_t)lrd[ldRow + p * kRowsPerPass] * kDnRowB + winCol;
        stage_tile(sdnL + stgSlot + (uint32_t)(p * kRowsPerPass * LdV * 2),
                   gather_ptr(dnumPlane, off));
      }
    }
    stgSlot += kSlot;
    if (stgSlot == kRing) stgSlot = 0;
  };

  //: dv's FAN-IN, gathered by the WRITE list: the partial staged for write entry
  //: `j` belongs to token `w[j]`, and entries past `nw` have no token.
  auto drain_dv = [&](int d, int nw) {
    const uint16_t* lw = iwrite(d);
    const uint32_t winFan = fanColB + (uint32_t)ihdr(d)[3] * (uint32_t)(kWindow * kORowB);
#pragma unroll
    for (int p = 0; p < kFanPasses; ++p) {
      const int row = fanRow + p * kFanRows;
      if (row < nw) {
        float* dp =
            reinterpret_cast<float*>(gather_ptr(dvPlane, (uint32_t)lw[row] * kORowB + winFan));
        const int e0 = fanE + p * kFanRows * LdO;
#pragma unroll
        for (int e = 0; e < kFanPerThr; ++e) fan_in_add(dp + e * kFanLanes, so[e0 + e * kFanLanes]);
      }
    }
  };

  //: THE UNITS' PARTIAL CONTRACTIONS MEET HERE, which is the same shape the
  //: forward's denominator drain has and for the same reason: the contraction's
  //: k axis is the carve's axis, so no unit holds the whole sum.
  auto drain_dgw = [&](int d, int nw) {
    if (gtid < C && gtid < nw) {
      float s = 0.f;
#pragma unroll
      for (int u = 0; u < kUnits; ++u) s += sgrad[u * C + gtid];
      fan_in_add(
          reinterpret_cast<float*>(gather_ptr(
              dgwPlane, (uint32_t)iwrite(d)[gtid] * 4u
                            + (uint32_t)ihdr(d)[3] * (uint32_t)(kWindow * (int)sizeof(float)))),
          s);
    }
  };

  // ---- routed Kronecker coefficient production, the forward's --------------
  const int prow = gtid / kRowThr;
  const int rIn = gtid % kRowThr;
  const int leadLocal = (rIn / kThrPerLead) * kLeadPerThr;
  const int tailLocal = (rIn % kThrPerLead) * kTailPerThr;
  constexpr uint32_t kPRowB = (uint32_t)(kWtot * (int)sizeof(__nv_bfloat16));
  constexpr uint32_t kOneSplat = 0x3F803F80u;
  const int oBase = leadLocal;
  [[maybe_unused]] uint32_t colFast = 0;
  if constexpr (kDepth >= 2)
    colFast =
        (uint32_t)((LP::col_base(kDepth - 2) + LP::off_at(kDepth - 2, blk) + oBase % kFastSpan)
                   * 2);
  const uint32_t colInner =
      (uint32_t)((LP::col_base(kDepth - 1) + LP::off_at(kDepth - 1, blk) + tailLocal) * 2);
  constexpr int kUppers = kDepth - 2;
  [[maybe_unused]] uint32_t colUpA = 0, colUpB = 0;
  if constexpr (kUppers >= 1) {
    constexpr int lA = kDepth - 3;
    colUpA =
        (uint32_t)((LP::col_base(lA) + LP::off_at(lA, blk) + (oBase / kFastSpan) % LP::span_at(lA))
                   * 2);
  }
  if constexpr (kUppers >= 2) {
    constexpr int lB = kDepth - 4;
    colUpB = (uint32_t)((LP::col_base(lB) + LP::off_at(lB, blk)
                         + (oBase / (kFastSpan * LP::span_at(kDepth - 3))) % LP::span_at(lB))
                        * 2);
  }
  const int slot0 = leadLocal * kInnerSpan + tailLocal;
  char* dstR = reinterpret_cast<char*>(sr) + prow * (int)(LdS * 2) + slot0 * 2;
  char* dstW = reinterpret_cast<char*>(sw) + prow * (int)(LdS * 2) + slot0 * 2;
  int pOff[kLeadPerThr][kTailVec];
#pragma unroll
  for (int j = 0; j < kLeadPerThr; ++j)
#pragma unroll
    for (int i = 0; i < kTailVec; ++i) pOff[j][i] = (j * kInnerSpan + i * 8) * 2;

  //: A panel row past the gathered count must be EXACTLY zero, and it is zeroed
  //: at the LEADING amplitude rather than at the product.
  auto produce = [&](const __nv_bfloat16* bs, char* dst, uint32_t tokOff, bool have, uint32_t gsp) {
    [[maybe_unused]] const __nv_bfloat16* p0 =
        reinterpret_cast<const __nv_bfloat16*>(gather_ptr(bs, tokOff + colFast));
    const __nv_bfloat16* p1 =
        reinterpret_cast<const __nv_bfloat16*>(gather_ptr(bs, tokOff + colInner));
    __align__(8) unsigned short amp0[kLeadPerThr];
    if constexpr (kDepth == 1) {
      amp0[0] = 0x3F80;
    } else if constexpr (kLeadPerThr == 1) {
      amp0[0] = *reinterpret_cast<const unsigned short*>(p0);
    } else if constexpr (kLeadPerThr == 2) {
      *reinterpret_cast<uint32_t*>(amp0) = *reinterpret_cast<const uint32_t*>(p0);
    } else {
      *reinterpret_cast<uint2*>(amp0) = *reinterpret_cast<const uint2*>(p0);
    }
    if constexpr (kUppers >= 1) {
      uint32_t up =
          splat_bf16x2(*reinterpret_cast<const __nv_bfloat16*>(gather_ptr(bs, tokOff + colUpA)));
      if constexpr (kUppers >= 2)
        up = mul_bf16x2(
            up,
            splat_bf16x2(*reinterpret_cast<const __nv_bfloat16*>(gather_ptr(bs, tokOff + colUpB))));
#pragma unroll
      for (int j = 0; j < kLeadPerThr; ++j) {
        const uint32_t m =
            mul_bf16x2(splat_bf16x2(*reinterpret_cast<const __nv_bfloat16*>(&amp0[j])), up);
        amp0[j] = (unsigned short)(m & 0xffffu);
      }
    }
#pragma unroll
    for (int j = 0; j < kLeadPerThr; ++j) {
      const uint32_t m =
          mul_bf16x2(splat_bf16x2(*reinterpret_cast<const __nv_bfloat16*>(&amp0[j])), gsp);
      amp0[j] = (unsigned short)(m & 0xffffu);
    }
#pragma unroll
    for (int i = 0; i < kTailVec; ++i) {
      __align__(16) uint32_t a1[4];
      *reinterpret_cast<uint4*>(a1) = load_vec16(p1 + i * 8);
#pragma unroll
      for (int j = 0; j < kLeadPerThr; ++j) {
        __align__(16) uint32_t prod[4];
        const uint32_t a2 =
            have ? splat_bf16x2(*reinterpret_cast<const __nv_bfloat16*>(&amp0[j])) : 0u;
#pragma unroll
        for (int e = 0; e < 4; ++e) prod[e] = mul_bf16x2(a2, a1[e]);
        store_vec16(dst + pOff[j][i], *reinterpret_cast<const uint4*>(prod));
      }
    }
  };

  // ---- the membership bits, the forward's clause rows ----------------------
  constexpr int kClauses = LP::clauses;
  uint32_t cA = 0, cB = 0, cC = 0, cD = 0;
#define ROLA_CLAUSE(CL_, C_)                   \
  if constexpr ((CL_) < kClauses) {            \
    constexpr int kDiv = LP::suffix_groups((CL_) + 1); \
    constexpr int kGrp = LP::groups_at(CL_);   \
    constexpr int kPre = LP::prefix_entries(CL_); \
    C_ = (uint32_t)(kPre + (blk / kDiv) % kGrp) * clauseStride; \
  }
  ROLA_CLAUSE(0, cA)
  ROLA_CLAUSE(1, cB)
  ROLA_CLAUSE(2, cC)
  ROLA_CLAUSE(3, cD)
#undef ROLA_CLAUSE

  const uint32_t tailMask =
      (L & (kTokBits - 1)) ? ((1u << (L & (kTokBits - 1))) - 1u) : 0xFFFFFFFFu;
  const int lastWord = tokWords - 1;

  int formLow = tokWords;
  auto form = [&]() {
    const int tailW = scook[kCookWord] - kFormWords + 1;
    const int limit = tailW > 0 ? tailW : 0;
    form_desc_impl<kClauses, kFormWords, kThreads>(formLow, limit, tid, rowR, rowW, cA, cB, cC, cD,
                                                   tailMask, lastWord, sfR, sfW);
  };
  auto form_if_low = [&]() {
    if (formLow > 0 && scook[kCookWord] - formLow < kFormWords / 2) form();
  };

  //: THE DESCENDING WALK.  Scan B associates the sum its own way -- the chunk
  //: partition is an association and not a semantic object -- so it walks the
  //: token axis downward with the forward's closing rule and needs no reversed
  //: table.  The LISTS IT PUBLISHES ARE ASCENDING, which is what keeps every
  //: consumer below the forward's code.
  int cookWord = tokWords - 1, cookNr = 0, cookNw = 0;
  bool cookOpen = false;
  constexpr int kWinWords = kWindow / kTokBits;
  int chunkWin = 0, winBaseW = 0;
  constexpr int kListPasses = (C + kTokBits - 1) / kTokBits;
  auto walk = [&](int d) {
    uint16_t* lrd = iread(d);
    uint16_t* lwr = iwrite(d);
    uint8_t* ljt = ithr(d);
    const bool exists = cookOpen || cookWord >= 0;
    int nr = cookOpen ? cookNr : 0;
    int nw = cookOpen ? cookNw : 0;
    if (!cookOpen && cookWord >= 0) {
      chunkWin = cookWord / kWinWords;
      winBaseW = chunkWin * kWinWords;
    }
    bool closed = cookWord < 0;
    const uint32_t gt = lane == 31 ? 0u : ~((1u << (lane + 1)) - 1u);
#pragma unroll 1
    while (cookWord >= formLow) {
      if (cookWord < winBaseW) {
        if (nr != 0 || nw != 0) {
          closed = true;
          break;
        }
        --chunkWin;
        winBaseW -= kWinWords;
      }
      const int slot = cookWord & (kFormWords - 1);
      const uint32_t rm = sfR[slot];
      const uint32_t wm = sfW[slot];
      const int prx = bit_count(rm & gt);
      const int pwx = bit_count(wm & gt);
      const bool lvR = ((rm >> lane) & 1u) != 0u;
      const bool lvW = ((wm >> lane) & 1u) != 0u;
      const int cut = last_set(lane_ballot((lvR && nr + prx >= C) || (lvW && nw + pwx >= C)));
      const uint32_t keep = cut < 0 ? 0xFFFFFFFFu : (cut == 31 ? 0u : ~((1u << (cut + 1)) - 1u));
      const int t = (cookWord - winBaseW) * kTokBits + lane;
      if (lane > cut) {
        if (lvR) {
          lrd[C - 1 - (nr + prx)] = (uint16_t)t;
          ljt[C - 1 - (nr + prx)] = (uint8_t)(nw + pwx);
        }
        if (lvW) lwr[C - 1 - (nw + pwx)] = (uint16_t)t;
      }
      nr += bit_count(rm & keep);
      nw += bit_count(wm & keep);
      if (cut >= 0) {
        if (lane == 0) {
          sfR[slot] = rm & ~keep;
          sfW[slot] = wm & ~keep;
        }
        closed = true;
        break;
      }
      --cookWord;
      if (cookWord < 0) {
        closed = true;
        break;
      }
    }
    if (!closed) {
      cookNr = nr;
      cookNw = nw;
      cookOpen = true;
      if (lane == 0) {
        scook[kCookWord] = cookWord;
        scook[kCookOpen] = 1;
      }
      return;
    }
    //: THE DESCENDING FILL LEAVES THE LISTS AT THE BUFFER'S TOP, and the
    //: threshold it recorded is the count of write entries ABOVE the read
    //: entry -- both are resolved here, where the chunk's totals are known.
    uint16_t rv[kListPasses], wv[kListPasses];
    uint8_t jv[kListPasses];
#pragma unroll
    for (int k = 0; k < kListPasses; ++k) {
      const int i = lane + k * kTokBits;
      rv[k] = (i < nr) ? lrd[i + C - nr] : (uint16_t)0;
      wv[k] = (i < nw) ? lwr[i + C - nw] : (uint16_t)0;
      jv[k] = (i < nr) ? (uint8_t)(nw - ljt[i + C - nr]) : (uint8_t)0;
    }
    __syncwarp();
#pragma unroll
    for (int k = 0; k < kListPasses; ++k) {
      const int i = lane + k * kTokBits;
      if (i < C) {
        lrd[i] = rv[k];
        lwr[i] = wv[k];
        ljt[i] = jv[k];
      }
    }
    const bool wasOpen = cookOpen;
    cookOpen = false;
    if (lane == 0) {
      int* h = ihdr(d);
      h[0] = nr;
      h[1] = nw;
      h[2] = exists ? 1 : 0;
      h[3] = chunkWin;
      scook[kCookWord] = cookWord;
      if (wasOpen) scook[kCookOpen] = 0;
    }
  };

  auto walk_wait = [&](int d) {
#pragma unroll 1
    while (scook[kCookOpen]) {
      form();
      rendezvous();
      if (warp == 0) walk(d);
      rendezvous();
    }
  };

  if (tid == 0) {
    scook[kCookWord] = tokWords - 1;
    scook[kCookOpen] = 0;
  }
  rendezvous();
  form();
  rendezvous();
#pragma unroll 1
  for (int s = 0; s < AHEAD; ++s) {
    if (warp == 0) walk(s);
    rendezvous();
    walk_wait(s);
    form_if_low();
    rendezvous();
  }
#pragma unroll 1
  for (int s = 0; s < STAGES - 1; ++s) {
    stage_chunk(s, ihdr(s)[1] != 0, ihdr(s)[0] != 0);
    stage_commit();
  }

  int dc = 0, ds = STAGES - 1, dn = AHEAD;
#pragma unroll 1
  for (;;) {
    const int* h = ihdr(dc);
    const int nr = h[0], nw = h[1];
    if (!h[2]) break;
    const bool liveR = nr != 0, liveW = nw != 0;
    const bool live = liveR || liveW;

    if (warp == 0) walk(dn);

    if (live) {
      const uint32_t winP = (uint32_t)ihdr(dc)[3] * (uint32_t)(kWindow * kWtot * 2);
      const uint32_t rOff = (uint32_t)iread(dc)[prow] * kPRowB + winP;
      const uint32_t wOff = (uint32_t)iwrite(dc)[prow] * kPRowB + winP;
      const uint32_t winG = (uint32_t)ihdr(dc)[3] * (uint32_t)(kWindow * 2);
      const uint32_t gsp = splat_bf16x2(*reinterpret_cast<const __nv_bfloat16*>(
          gather_ptr(gWrite, (uint32_t)iwrite(dc)[prow] * 2u + winG)));
      produce(pRead, dstR, rOff, prow < nr, kOneSplat);
      produce(pWrite, dstW, wOff, prow < nw, gsp);
    }

    stage_wait<STAGES - 2>();
    rendezvous();  // (1) panels and both staged tiles visible; the previous
                   //     chunk's `so`/`sgrad` are dead; the walk has published

    walk_wait(dn);
    form_if_low();
    stage_chunk(ds, ihdr(ds)[1] != 0, ihdr(ds)[0] != 0);
    stage_commit();

    const uint32_t atv = svB + cmpSlot + rowFv;
    const uint32_t adn = sdnB + cmpSlot + rowTv;
    cmpSlot += kSlot;
    if (cmpSlot == kRing) cmpSlot = 0;

    //: BOTH WRITE-SIDE TERMS READ THE STATE THIS CHUNK ENTERS WITH -- the
    //: adjoint arriving from the FUTURE -- so both run before the fold.
    if (liveW) {
      uint32_t da[KD][4];
#pragma unroll
      for (int kk = 0; kk < KD; ++kk) {
        da[kk][0] = pack_bf16x2(dst[2 * kk][0], dst[2 * kk][1]);
        da[kk][1] = pack_bf16x2(dst[2 * kk][2], dst[2 * kk][3]);
        da[kk][2] = pack_bf16x2(dst[2 * kk + 1][0], dst[2 * kk + 1][1]);
        da[kk][3] = pack_bf16x2(dst[2 * kk + 1][2], dst[2 * kk + 1][3]);
      }
      // ---- (A.17) dv^T = dS^T . W^T, the forward's readout with the write panel
      float acc[NTOK][4];
#pragma unroll
      for (int j = 0; j < NTOK; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.f;
#pragma unroll
      for (int m = 0; m < MTOK; ++m) {
        uint32_t wb[KD][4];
#pragma unroll
        for (int kk = 0; kk < KD; ++kk) load_frag(wb[kk], ADDR_S(bwT, m * 16, 2 * kk));
#pragma unroll
        for (int kk = 0; kk < KD; ++kk)
#pragma unroll
          for (int p = 0; p < 2; ++p) mma(acc[2 * m + p], da[kk], &wb[kk][2 * p]);
      }
#pragma unroll
      for (int j = 0; j < NTOK; ++j) {
        const int jo = j * 8 * LdO;
        store_f32(&so[oE[0][0] + jo], acc[j][0]);
        store_f32(&so[oE[1][0] + jo], acc[j][1]);
        store_f32(&so[oE[0][1] + jo], acc[j][2]);
        store_f32(&so[oE[1][1] + jo], acc[j][3]);
      }

      // ---- (A.18) the CHANNEL CONTRACTION: dW[j,s] = <dS[s,:], v[j,:]> + dz[s],
      //      reduced over this block's slots, which is d gw's own shape.
      uint32_t va[MTOK][4];
#pragma unroll
      for (int m = 0; m < MTOK; ++m) load_frag(va[m], ADDR_V(atv, m * 16));
      float tok[MTOK][2];
#pragma unroll
      for (int m = 0; m < MTOK; ++m) {
        tok[m][0] = 0.f;
        tok[m][1] = 0.f;
      }
      //: TILE BY TILE, REDUCED IN PLACE.  Holding the whole C x BCg gradient
      //: accumulator is the measured alternative and it does not fit -- neither
      //: in the register budget nor beside the ring (docs/internals/chunk/backward.md).
#pragma unroll
      for (int n = 0; n < NDV; ++n) {
        uint32_t db[2];
        db[0] = transpose_frag_b16(pack_bf16x2(dst[n][0], dst[n][1]));
        db[1] = transpose_frag_b16(pack_bf16x2(dst[n][2], dst[n][3]));
        float g[MTOK][4];
#pragma unroll
        for (int m = 0; m < MTOK; ++m) {
#pragma unroll
          for (int e = 0; e < 4; ++e) g[m][e] = 0.f;
          mma(g[m], va[m], db);
          tok[m][0] += g[m][0] + g[m][1];
          tok[m][1] += g[m][2] + g[m][3];
        }
      }
      //: The mass column's share is slot-only, so it is ONE number per unit and
      //: it lands on every token of the chunk exactly once through the sum below.
      //: Its slots are spread over the whole unit -- the padding rows are
      //: identically zero, so the reduction needs no lane predicate, only every
      //: lane.
      float dzs = 0.f;
#pragma unroll
      for (int j = 0; j < kMassTiles; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) dzs += dz_st[j][e];
#pragma unroll
      for (int b2 = 1; b2 < 32; b2 <<= 1) dzs += __shfl_xor_sync(0xFFFFFFFFu, dzs, b2);
#pragma unroll
      for (int m = 0; m < MTOK; ++m) {
#pragma unroll
        for (int b2 = 1; b2 < 4; b2 <<= 1) {
          tok[m][0] += __shfl_xor_sync(0xFFFFFFFFu, tok[m][0], b2);
          tok[m][1] += __shfl_xor_sync(0xFFFFFFFFu, tok[m][1], b2);
        }
      }
      if ((lane & 3) == 0) {
#pragma unroll
        for (int m = 0; m < MTOK; ++m) {
          store_f32(&sgrad[gwarp * C + m * 16 + (lane >> 2)], tok[m][0] + dzs);
          store_f32(&sgrad[gwarp * C + m * 16 + (lane >> 2) + 8], tok[m][1] + dzs);
        }
      }
    }

    rendezvous();  // (2) `so` and `sgrad` are visible to every unit

    if (liveW) {
      drain_dv(dc, nw);
      drain_dgw(dc, nw);
    }

    if (liveR) {
      // ---- (A.16) THE FOLD: dS^T += dnum^T . R, and dz^T += dden^T . R -------
#pragma unroll
      for (int kk = 0; kk < MTOK; ++kk) {
        uint32_t a[4];
        load_frag_t(a, ADDR_V(adn, kk * 16));
        uint32_t b[KD][4];
#pragma unroll
        for (int nj = 0; nj < KD; ++nj) load_frag_t(b[nj], ADDR_S(brF, kk * 16, 2 * nj));
#pragma unroll
        for (int nj = 0; nj < KD; ++nj) {
          mma(dst[2 * nj], a, &b[nj][0]);
          mma(dst[2 * nj + 1], a, &b[nj][2]);
        }
        //: THE MASS COLUMN'S A OPERAND IS dden WHERE THE FORWARD'S IS THE
        //: LITERAL ONE: row 0 carries the seed, rows 1..15 are the m quantum's
        //: padding and stay identically zero.
        if (gwarp < kMassUnits) {
          uint32_t dda[4];
          {
            const int k0 = kk * 16 + (lane & 3) * 2;
            const uint16_t* lrd = iread(dc);
            const uint32_t winD = (uint32_t)ihdr(dc)[3] * (uint32_t)(kWindow * 2);
            const bool row0 = (lane >> 2) == 0;
            auto seed = [&](int k) {
              return row0 ? (uint32_t)*reinterpret_cast<const unsigned short*>(
                                gather_ptr(ddenPlane, (uint32_t)lrd[k] * 2u + winD))
                          : 0u;
            };
            dda[0] = seed(k0) | (seed(k0 + 1) << 16);
            dda[1] = 0u;
            dda[2] = seed(k0 + 8) | (seed(k0 + 9) << 16);
            dda[3] = 0u;
          }
#pragma unroll
          for (int q = 0; q < kMassKD; ++q) {
            uint32_t mb[4];
            load_frag_t(mb, ADDR_S_M(brF, kk * 16, q));
            mma(dz_st[2 * q], dda, &mb[0]);
            mma(dz_st[2 * q + 1], dda, &mb[2]);
          }
        }
      }
    }

    rendezvous();  // (3) `so`/`sgrad` are free and the panels are dead

    if (++dc == IRING) dc = 0;
    if (++ds == IRING) ds = 0;
    if (++dn == IRING) dn = 0;
  }
#undef ADDR_S
#undef ADDR_S_M
#undef ADDR_V
}

}  // namespace rola::chunk
