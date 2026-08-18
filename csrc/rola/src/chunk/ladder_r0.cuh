// csrc/rola/src/chunk/ladder_r0.cuh -- THE BACKWARD LADDER, RUNGS 0-1: the dense
// chunked linear-attention backward, written in RoLA's own quantities as a
// two-pass pair.  Pass 1 ascends, rebuilding the state and its mass column from
// zero and emitting the read side's per-slot cotangent; pass 2 descends, carrying
// both adjoints and emitting the value cotangent and the write side's per-slot
// one.  Dense routing, contiguous chunks, no write gain, no digits -- those are
// later rungs.  docs/internals/chunk/ladder_r0.md
#pragma once

#include "chunk_kernel.cuh"

namespace rola::chunk {

//: Rung 0's launch arguments.  The coefficients are PER SLOT: at this rung the
//: slot panel is the identity, so there is no digit arithmetic and no producer.
struct Rung0Params {
  const __nv_bfloat16 *r, *w, *v, *dnum, *dden;
  float *dr, *dv, *dw;
  int BH, L, v_row_bytes;
};

//: THE RUNG'S LIVE REGISTER SET, one measured DECLARATION for both passes and
//: falsified per arm by the census' spill and budget columns.  It is a single number
//: rather than a `BC` row because the rung is built at one block width; the pass that
//: peaks is the descending one, which carries the adjoint state through every block.
constexpr int reg_floor_r0() { return 192; }

//: RUNG 0'S GEOMETRY.  Every stream the kernel consumes over time rides the ring:
//: the two value-shaped panels always, the write coefficients always, the read
//: coefficients in the pass that carries the adjoint, and the readout's mass seed
//: inside the `dnum` panel's own row pad, which costs no bytes at all.  The
//: published state's rows are UNPADDED -- it is read four times per unit per visit,
//: where the pad would buy nothing and cost the CTA target.  Pass 2 sits EXACTLY on
//: the two-CTA shared-memory ceiling, which is why the mass column may not claim a
//: buffer -- docs/internals/chunk/ladder_r0.md
template <int C, int BC, bool PASS2>
struct GeomR0 {
  static constexpr int LdV = kD + kPadE;
  static constexpr int LdS = BC + kPadE;
  static constexpr int LdG = C + kPadE;
  static constexpr int LdDs = BC;
  static constexpr int grams = PASS2 ? 2 : 1;
  static constexpr int coef_panels = PASS2 ? 2 : 1;
  static constexpr int slot = (2 * C * LdV + coef_panels * C * LdS) * (int)sizeof(__nv_bfloat16);
  static constexpr int gram_bytes = grams * C * LdG * (int)sizeof(__nv_bfloat16);
  static constexpr int ds_bytes = kD * LdDs * (int)sizeof(__nv_bfloat16);
  static constexpr int fixed = gram_bytes + ds_bytes;
  static constexpr int stages = design::stages_for(slot, fixed, threads_for(BC), reg_floor_r0());
  static constexpr size_t bytes = (size_t)stages * slot + fixed;
};

template <int C, int BC, bool PASS2>
constexpr int kR0CtaTarget =
    design::cta_target((int)GeomR0<C, BC, PASS2>::bytes, threads_for(BC), reg_floor_r0());

//: A [C, COLS] tile of a contiguous plane, staged asynchronously.  Rung 0's chunks
//: are contiguous token runs, so a tile's rows are the plane's own and the schedule
//: contributes no gather.
template <int C, int COLS, int LD, int THREADS>
struct TileStage {
  static constexpr int gran = COLS / 8;
  static constexpr int rows = THREADS / gran;
  static constexpr int passes = C / rows;
  static_assert(COLS % 8 == 0 && THREADS % gran == 0 && C % rows == 0,
                "a staging pass must cover the tile's rows exactly");
  int row;
  uint32_t colBytes;
  uint32_t dstOff;

  __device__ __forceinline__ void init(int tid) {
    row = tid / gran;
    const int col = (tid % gran) * 8;
    colBytes = (uint32_t)(col * (int)sizeof(__nv_bfloat16));
    dstOff = (uint32_t)((row * LD + col) * (int)sizeof(__nv_bfloat16));
  }

  __device__ __forceinline__ void issue(uint32_t dst, const void* plane, uint32_t tok0,
                                        uint32_t rowBytes) const {
#pragma unroll
    for (int p = 0; p < passes; ++p) {
      const uint32_t off = (tok0 + (uint32_t)(row + p * rows)) * rowBytes + colBytes;
      stage_tile(dst + dstOff + (uint32_t)(p * rows * LD * (int)sizeof(__nv_bfloat16)),
                 gather_ptr(plane, off));
    }
  }
};

#define R0_ADDR(b_, ld_, R_, g_) ((b_) + (uint32_t)(((R_) * (ld_) + (g_) * 8) * 2))

//: THE MASS SEED'S SEAT, in the `dnum` panel's row pad: entry `i` at row `i / 8`,
//: column `d_v + i % 8`.  Eight-wide groups are what makes `(i, i + 1)` adjacent for
//: every even `i`, which is the whole reason for the shape -- the fold reads the seed
//: as an mma A fragment, whose lanes want exactly those pairs.
#define R0_DDEN(ld_, i_) (((i_) >> 3) * (ld_) + kD + ((i_) & 7))

//: THE SEED'S SEATS, RESOLVED ONCE.  Every consumer's entry index is a fixed
//: function of the lane, so the seat arithmetic is loop-invariant and only the ring
//: slot's base moves; carrying the offsets is what keeps the visit's added work to
//: the arithmetic the mass terms actually are.
template <int LD, int ROWS>
__device__ __forceinline__ void r0_dden_seats(int (&off)[ROWS][2], int row0, int stride) {
#pragma unroll
  for (int i = 0; i < ROWS; ++i) {
    off[i][0] = R0_DDEN(LD, row0 + i * stride);
    off[i][1] = R0_DDEN(LD, row0 + i * stride + 8);
  }
}

//: The mass folds contract a token axis against a coefficient panel with an A
//: operand that is EXACTLY representable in bf16 -- ones on the write side, the
//: bf16 seed plane itself on the read side -- so the fold rides the tensor pipe
//: without a compensated split.  `kOnesFrag` is bf16 `1.0` in both halves.
static constexpr uint32_t kOnesFrag = 0x3F803F80u;

//: THE STAIRCASE, DENSE: chunk-local token indices, and a write entry is kept when
//: its token is at or before the read entry's.
#define R0_STAIR(v_, tc_, tr_) ((tc_) <= (tr_) ? (v_) : 0.f)

//: PASS 1 -- the ascending rebuild.  It carries `S` and its mass column `z`, and
//: emits `dR[i,s] = <dnum_i, S_a[s,:]> + dden_i z_a[s] + sum_{j<=i} W[j,s] Gd'[i,j]`
//: with `Gd'[i,j] = <dnum_i, v_j> + dden_i`.
template <int C, int BC>
__global__ __launch_bounds__(threads_for(BC), (kR0CtaTarget<C, BC, false>)) void rung0_pass1_kernel(
    const __nv_bfloat16* __restrict__ wcoef, const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ dnum, const __nv_bfloat16* __restrict__ dden,
    float* __restrict__ dr, int L, int v_row_bytes) {
  using G = GeomR0<C, BC, false>;
  constexpr int kThreads = threads_for(BC);
  constexpr int kWarps = kThreads / 32;
  constexpr int STAGES = G::stages;
  constexpr int LdV = G::LdV, LdS = G::LdS, LdG = G::LdG, LdDs = G::LdDs;
  constexpr int MTOK = C / kMmaM, NTOK = C / kMmaN;
  constexpr int KD = BC / kMmaK, NDV = BC / kMmaN, KV = kD / kMmaK;
  //: the per-slot output splits over the SLOT axis: every unit runs the whole
  //: channel contraction out of the published state, so the four outputs are
  //: disjoint and no cross-unit sum exists.
  constexpr int NSLOT = NDV / kUnits;
  constexpr int NPAIR = NSLOT / 2;
  static_assert(kWarps == kUnits, "rung 0 is the one-group carve");
  static_assert(NDV == kUnits * NSLOT && NSLOT == 2 * NPAIR,
                "a unit's slot run must be a whole number of ldmatrix pairs");
  static_assert(design::smem_fits(G::bytes),
                "this arm does not fit this architecture's per-CTA shared-memory maximum");
  static_assert(STAGES >= 2, "the wait/issue order needs at least two stages");
  static_assert(C % 8 == 0 && kPadE >= 8 && C / 8 <= C,
                "the mass seed seats in the value panel's row pad, eight entries to a row");
  constexpr int MSPLIT = MTOK >= kWarps ? kWarps : MTOK;
  constexpr int NSPLIT_RAW = kWarps / MSPLIT;
  constexpr int NSPLIT = NSPLIT_RAW < NTOK / 2 ? NSPLIT_RAW : NTOK / 2;
  constexpr int MPW = MTOK / MSPLIT;
  constexpr int NPW = NTOK / NSPLIT;
  constexpr int GWARPS = MSPLIT * NSPLIT;
  constexpr int GPAIR = NPW / 2;

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;

  extern __shared__ __align__(16) char smem_raw[];
  __nv_bfloat16* sv = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* sdn = sv + STAGES * C * LdV;
  __nv_bfloat16* sw = sdn + STAGES * C * LdV;
  __nv_bfloat16* sg = sw + STAGES * C * LdS;
  __nv_bfloat16* sds = sg + C * LdG;

  const int vRowElems = v_row_bytes / (int)sizeof(__nv_bfloat16);
  const int heads = vRowElems / kD;
  const __nv_bfloat16* const vPlane =
      pin_address(v + (long)(blockIdx.x / heads) * L * vRowElems + (long)(blockIdx.x % heads) * kD);
  const __nv_bfloat16* const dnPlane = pin_address(dnum + (long)blockIdx.x * L * kD);
  const __nv_bfloat16* const ddPlane = pin_address(dden + (long)blockIdx.x * L);
  const __nv_bfloat16* const wPlane = pin_address(wcoef + (long)blockIdx.x * L * BC);
  float* const drPlane = pin_address(dr + (long)blockIdx.x * L * BC);

  const int lr = lane & 7, tl = lane >> 3;
  const int rF = (tl & 1) * 8 + lr, cF = tl >> 1;
  const int rT = (tl >> 1) * 8 + lr, cT = tl & 1;
  const uint32_t offFv = (uint32_t)((rF * LdV + cF * 8) * 2);
  const uint32_t offTv = (uint32_t)((rT * LdV + cT * 8) * 2);
  const uint32_t offFs = (uint32_t)((rF * LdS + cF * 8) * 2);
  const uint32_t offFg = (uint32_t)((rF * LdG + cF * 8) * 2);
  const uint32_t offFd = (uint32_t)((rF * LdDs + cF * 8) * 2);
  const int gm0 = (warp / NSPLIT) * MPW, gn0 = (warp % NSPLIT) * NPW;
  int ddG[MPW][2], ddM[MTOK][2];
  r0_dden_seats<LdV, MPW>(ddG, gm0 * 16 + (lane >> 2), 16);
  r0_dden_seats<LdV, MTOK>(ddM, lane >> 2, 16);

  const uint32_t svB = smem_addr(sv), sdnB = smem_addr(sdn), swB = smem_addr(sw);
  const uint32_t sgF = smem_addr(sg) + offFg;
  const uint32_t sdsF = smem_addr(sds) + offFd;

  //: the transposed state: m = d_v (this unit's 16 rows), n = this block's slots.
  float st[NDV][4];
#pragma unroll
  for (int j = 0; j < NDV; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) st[j][e] = 0.f;

  //: THE MASS COLUMN, in the accumulator layout its one consumer already reads.
  //: `A` is all ones, so every `m` row of the tile carries the same slot total and
  //: elements 0/1 are the pair the entry-mass term wants; no reduction exists.
  float zac[NSLOT][4];
#pragma unroll
  for (int n = 0; n < NSLOT; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) zac[n][e] = 0.f;
  const uint32_t ones[4] = {kOnesFrag, kOnesFrag, kOnesFrag, kOnesFrag};

  TileStage<C, kD, LdV, kThreads> stV;
  TileStage<C, BC, LdS, kThreads> stS;
  stV.init(tid);
  stS.init(tid);
  constexpr uint32_t kVSlotB = (uint32_t)(C * LdV * 2);
  constexpr uint32_t kSSlotB = (uint32_t)(C * LdS * 2);
  constexpr uint32_t kVRingB = (uint32_t)STAGES * kVSlotB;
  constexpr uint32_t kSRingB = (uint32_t)STAGES * kSSlotB;
  const uint32_t vRowB = (uint32_t)v_row_bytes;
  constexpr uint32_t kDnRowB = (uint32_t)(kD * (int)sizeof(__nv_bfloat16));
  constexpr uint32_t kSRowB = (uint32_t)(BC * (int)sizeof(__nv_bfloat16));
  constexpr uint32_t kDrRowB = (uint32_t)(BC * (int)sizeof(float));
  uint32_t stgV = 0, cmpV = 0, stgS = 0, cmpS = 0;

  const int NC = L / C;
  auto stage_chunk = [&](int c) {
    if (c < NC) {
      const uint32_t tok0 = (uint32_t)(c * C);
      stV.issue(svB + stgV, vPlane, tok0, vRowB);
      stV.issue(sdnB + stgV, dnPlane, tok0, kDnRowB);
      stS.issue(swB + stgS, wPlane, tok0, kSRowB);
      if (tid < C) sdn[(stgV >> 1) + R0_DDEN(LdV, tid)] = ddPlane[(int)tok0 + tid];
    }
    stgV += kVSlotB;
    if (stgV == kVRingB) stgV = 0;
    stgS += kSSlotB;
    if (stgS == kSRingB) stgS = 0;
  };

#pragma unroll 1
  for (int s = 0; s < STAGES - 1; ++s) {
    stage_chunk(s);
    stage_commit();
  }

#pragma unroll 1
  for (int c = 0; c < NC; ++c) {
    stage_wait<STAGES - 2>();
    rendezvous();  // (1) this chunk's tiles are visible; the last visit's Gram and
                   //     published state are dead
    stage_chunk(c + STAGES - 1);
    stage_commit();

    const uint32_t aVT = svB + cmpV + offTv;
    const uint32_t aDF = sdnB + cmpV + offFv;
    const uint32_t aWF = swB + cmpS + offFs;
    const __nv_bfloat16* const dd = sdn + (cmpV >> 1);
    const uint32_t tokBase = (uint32_t)(c * C);
    cmpV += kVSlotB;
    if (cmpV == kVRingB) cmpV = 0;
    cmpS += kSSlotB;
    if (cmpS == kSRingB) cmpS = 0;

    //: THE ENTRY STATE, PUBLISHED ONCE PER VISIT.  A unit owns sixteen channels of
    //: every slot, so the four publications are disjoint and what comes back is the
    //: whole channel axis -- which is what lets the per-slot term finish in one unit.
#pragma unroll
    for (int nj = 0; nj < NDV; ++nj) {
      const int s0 = nj * kMmaN + (lane & 3) * 2;
      const int c0 = warp * kMmaM + (lane >> 2);
      store_bf16x2(&sds[c0 * LdDs + s0], pack_bf16x2(st[nj][0], st[nj][1]));
      store_bf16x2(&sds[(c0 + 8) * LdDs + s0], pack_bf16x2(st[nj][2], st[nj][3]));
    }

    // ---- the d_v Gram: Gd' = staircase(dnum . v^T + dden (x) 1) --------------
    if (warp < GWARPS) {
#pragma unroll
      for (int m = 0; m < MPW; ++m) {
        float g[GPAIR][2][4];
#pragma unroll
        for (int u = 0; u < GPAIR; ++u)
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int e = 0; e < 4; ++e) g[u][h][e] = 0.f;
#pragma unroll
        for (int kk = 0; kk < KV; ++kk) {
          uint32_t qa[4];
          load_frag(qa, R0_ADDR(aDF, LdV, (gm0 + m) * 16, 2 * kk));
#pragma unroll
          for (int u = 0; u < GPAIR; ++u) {
            uint32_t bf[4];
            load_frag(bf, R0_ADDR(aVT, LdV, (gn0 + u * 2) * 8, 2 * kk));
            mma(g[u][0], qa, &bf[0]);
            mma(g[u][1], qa, &bf[2]);
          }
        }
        const int tr = (gm0 + m) * 16 + (lane >> 2);
        const float m0 = __bfloat162float(dd[ddG[m][0]]), m1 = __bfloat162float(dd[ddG[m][1]]);
#pragma unroll
        for (int u = 0; u < GPAIR; ++u)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int tc = (gn0 + u * 2 + h) * 8 + (lane & 3) * 2;
            store_bf16x2(&sg[tr * LdG + tc], pack_bf16x2(R0_STAIR(g[u][h][0] + m0, tc, tr),
                                                         R0_STAIR(g[u][h][1] + m0, tc + 1, tr)));
            store_bf16x2(&sg[(tr + 8) * LdG + tc],
                         pack_bf16x2(R0_STAIR(g[u][h][2] + m1, tc, tr + 8),
                                     R0_STAIR(g[u][h][3] + m1, tc + 1, tr + 8)));
          }
      }
    }

    rendezvous();  // (2) the published state and the Gram are visible

    // ---- dR, over this unit's slot run ---------------------------------------
    {
      float acc[MTOK][NSLOT][4];
#pragma unroll
      for (int m = 0; m < MTOK; ++m)
#pragma unroll
        for (int n = 0; n < NSLOT; ++n)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[m][n][e] = 0.f;
      const int n0 = warp * NSLOT;
      //: the INTER term: the whole channel axis, out of the published state.
#pragma unroll
      for (int kk = 0; kk < KV; ++kk)
#pragma unroll
        for (int np = 0; np < NPAIR; ++np) {
          uint32_t sb[4];
          load_frag_t(sb, R0_ADDR(sdsF, LdDs, kk * 16, n0 + np * 2));
#pragma unroll
          for (int m = 0; m < MTOK; ++m) {
            uint32_t da[4];
            load_frag(da, R0_ADDR(aDF, LdV, m * 16, 2 * kk));
            mma(acc[m][2 * np], da, &sb[0]);
            mma(acc[m][2 * np + 1], da, &sb[2]);
          }
        }
      //: the INTRA term: the staircase already lives in the Gram, so this is one
      //: contraction over the chunk's write tokens with the Gram as its A operand.
#pragma unroll
      for (int kk = 0; kk < MTOK; ++kk)
#pragma unroll
        for (int np = 0; np < NPAIR; ++np) {
          uint32_t wb[4];
          load_frag_t(wb, R0_ADDR(aWF, LdS, kk * 16, n0 + np * 2));
#pragma unroll
          for (int m = 0; m < MTOK; ++m) {
            uint32_t ga[4];
            load_frag(ga, R0_ADDR(sgF, LdG, m * 16, 2 * kk));
            mma(acc[m][2 * np], ga, &wb[0]);
            mma(acc[m][2 * np + 1], ga, &wb[2]);
          }
        }
      //: the ENTRY MASS, `dden_i z_a[s]`: an OUTER product of a per-read-token
      //: value and a per-slot one, so it is a rank-1 add and not a broadcast.
#pragma unroll
      for (int m = 0; m < MTOK; ++m) {
        const float d0 = __bfloat162float(dd[ddM[m][0]]);
        const float d1 = __bfloat162float(dd[ddM[m][1]]);
#pragma unroll
        for (int n = 0; n < NSLOT; ++n) {
          acc[m][n][0] += d0 * zac[n][0];
          acc[m][n][1] += d0 * zac[n][1];
          acc[m][n][2] += d1 * zac[n][0];
          acc[m][n][3] += d1 * zac[n][1];
        }
      }
#pragma unroll
      for (int m = 0; m < MTOK; ++m)
#pragma unroll
        for (int n = 0; n < NSLOT; ++n) {
          const uint32_t colB =
              (uint32_t)(((n0 + n) * kMmaN + (lane & 3) * 2) * (int)sizeof(float));
          const int t0 = m * 16 + (lane >> 2);
          float* p0 = reinterpret_cast<float*>(
              gather_ptr(drPlane, (tokBase + (uint32_t)t0) * kDrRowB + colB));
          float* p1 = reinterpret_cast<float*>(
              gather_ptr(drPlane, (tokBase + (uint32_t)(t0 + 8)) * kDrRowB + colB));
          store_f32(p0 + 0, acc[m][n][0]);
          store_f32(p0 + 1, acc[m][n][1]);
          store_f32(p1 + 0, acc[m][n][2]);
          store_f32(p1 + 1, acc[m][n][3]);
        }
    }

    // ---- the fold: S^T += V^T . W, and z += 1^T . W --------------------------
#pragma unroll
    for (int kk = 0; kk < MTOK; ++kk) {
      uint32_t a[4];
      load_frag_t(a, R0_ADDR(aVT, LdV, kk * 16, 2 * warp));
#pragma unroll
      for (int nj = 0; nj < KD; ++nj) {
        uint32_t b[4];
        load_frag_t(b, R0_ADDR(aWF, LdS, kk * 16, 2 * nj));
        mma(st[2 * nj], a, &b[0]);
        mma(st[2 * nj + 1], a, &b[2]);
      }
    }
    //: the mass fold takes the SAME operand from the SAME tiles the per-slot term
    //: reads, so it re-reads rather than outliving the fold's own k-run.
    {
      const int n0 = warp * NSLOT;
#pragma unroll
      for (int kk = 0; kk < MTOK; ++kk)
#pragma unroll
        for (int np = 0; np < NPAIR; ++np) {
          uint32_t wb[4];
          load_frag_t(wb, R0_ADDR(aWF, LdS, kk * 16, n0 + np * 2));
          mma(zac[2 * np], ones, &wb[0]);
          mma(zac[2 * np + 1], ones, &wb[2]);
        }
    }
  }
}

//: PASS 2 -- the descending adjoint.  It carries `dS` and its mass column `dz`, and
//: emits `dv[j,:] = sum_s W[j,s] dS_after[j,s,:]` and
//: `dW[j,s] = <dS_a[s,:], v_j> + dz_a[s] + sum_{i>=j} R[i,s] Gd'[i,j]`.  `dv` is a
//: value-channel quantity and the mass column does not enter it.
template <int C, int BC>
__global__ __launch_bounds__(threads_for(BC), (kR0CtaTarget<C, BC, true>)) void rung0_pass2_kernel(
    const __nv_bfloat16* __restrict__ rcoef, const __nv_bfloat16* __restrict__ wcoef,
    const __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ dnum,
    const __nv_bfloat16* __restrict__ dden, float* __restrict__ dv, float* __restrict__ dw, int L,
    int v_row_bytes) {
  using G = GeomR0<C, BC, true>;
  constexpr int kThreads = threads_for(BC);
  constexpr int kWarps = kThreads / 32;
  constexpr int STAGES = G::stages;
  constexpr int LdV = G::LdV, LdS = G::LdS, LdG = G::LdG, LdDs = G::LdDs;
  constexpr int MTOK = C / kMmaM, NTOK = C / kMmaN;
  constexpr int KD = BC / kMmaK, NDV = BC / kMmaN, KV = kD / kMmaK;
  constexpr int NSLOT = NDV / kUnits;
  constexpr int NPAIR = NSLOT / 2;
  static_assert(kWarps == kUnits, "rung 0 is the one-group carve");
  static_assert(NDV == kUnits * NSLOT && NSLOT == 2 * NPAIR,
                "a unit's slot run must be a whole number of ldmatrix pairs");
  static_assert(design::smem_fits(G::bytes),
                "this arm does not fit this architecture's per-CTA shared-memory maximum");
  static_assert(STAGES >= 2, "the wait/issue order needs at least two stages");
  static_assert(C % 8 == 0 && kPadE >= 8 && C / 8 <= C,
                "the mass seed seats in the value panel's row pad, eight entries to a row");
  constexpr int MSPLIT = MTOK >= kWarps ? kWarps : MTOK;
  constexpr int NSPLIT_RAW = kWarps / MSPLIT;
  constexpr int NSPLIT = NSPLIT_RAW < NTOK / 2 ? NSPLIT_RAW : NTOK / 2;
  constexpr int MPW = MTOK / MSPLIT;
  constexpr int NPW = NTOK / NSPLIT;
  constexpr int GWARPS = MSPLIT * NSPLIT;
  constexpr int GPAIR = NPW / 2;

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;

  extern __shared__ __align__(16) char smem_raw[];
  __nv_bfloat16* sv = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* sdn = sv + STAGES * C * LdV;
  __nv_bfloat16* sr = sdn + STAGES * C * LdV;
  __nv_bfloat16* sw = sr + STAGES * C * LdS;
  __nv_bfloat16* sgu = sw + STAGES * C * LdS;
  __nv_bfloat16* sgd = sgu + C * LdG;
  __nv_bfloat16* sds = sgd + C * LdG;

  const int vRowElems = v_row_bytes / (int)sizeof(__nv_bfloat16);
  const int heads = vRowElems / kD;
  const __nv_bfloat16* const vPlane =
      pin_address(v + (long)(blockIdx.x / heads) * L * vRowElems + (long)(blockIdx.x % heads) * kD);
  const __nv_bfloat16* const dnPlane = pin_address(dnum + (long)blockIdx.x * L * kD);
  const __nv_bfloat16* const ddPlane = pin_address(dden + (long)blockIdx.x * L);
  const __nv_bfloat16* const rPlane = pin_address(rcoef + (long)blockIdx.x * L * BC);
  const __nv_bfloat16* const wPlane = pin_address(wcoef + (long)blockIdx.x * L * BC);
  float* const dvPlane = pin_address(dv + (long)blockIdx.x * L * kD);
  float* const dwPlane = pin_address(dw + (long)blockIdx.x * L * BC);

  const int lr = lane & 7, tl = lane >> 3;
  const int rF = (tl & 1) * 8 + lr, cF = tl >> 1;
  const int rT = (tl >> 1) * 8 + lr, cT = tl & 1;
  const uint32_t offFv = (uint32_t)((rF * LdV + cF * 8) * 2);
  const uint32_t offTv = (uint32_t)((rT * LdV + cT * 8) * 2);
  const uint32_t offFs = (uint32_t)((rF * LdS + cF * 8) * 2);
  const uint32_t offTs = (uint32_t)((rT * LdS + cT * 8) * 2);
  const uint32_t offFg = (uint32_t)((rF * LdG + cF * 8) * 2);
  const uint32_t offTg = (uint32_t)((rT * LdG + cT * 8) * 2);
  const uint32_t offFd = (uint32_t)((rF * LdDs + cF * 8) * 2);
  const int gm0 = (warp / NSPLIT) * MPW, gn0 = (warp % NSPLIT) * NPW;
  int ddG[MPW][2], ddK[MTOK][2];
  r0_dden_seats<LdV, MPW>(ddG, gm0 * 16 + (lane >> 2), 16);
  r0_dden_seats<LdV, MTOK>(ddK, (lane & 3) * 2, 16);

  const uint32_t svB = smem_addr(sv), sdnB = smem_addr(sdn);
  const uint32_t srB = smem_addr(sr), swB = smem_addr(sw);
  const uint32_t sguF = smem_addr(sgu) + offFg;
  const uint32_t sgdT = smem_addr(sgd) + offTg;
  const uint32_t sdsF = smem_addr(sds) + offFd;

  //: the adjoint state, transposed exactly as the forward's state is.
  float dst[NDV][4];
#pragma unroll
  for (int j = 0; j < NDV; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) dst[j][e] = 0.f;

  //: THE ADJOINT MASS COLUMN, in the accumulator layout the write-side term reads.
  //: Its `A` operand is the staged seed broadcast down `m`, which the seed plane's
  //: own bf16 makes exact.
  float dzc[NSLOT][4];
#pragma unroll
  for (int n = 0; n < NSLOT; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) dzc[n][e] = 0.f;

  TileStage<C, kD, LdV, kThreads> stV;
  TileStage<C, BC, LdS, kThreads> stS;
  stV.init(tid);
  stS.init(tid);
  constexpr uint32_t kVSlotB = (uint32_t)(C * LdV * 2);
  constexpr uint32_t kSSlotB = (uint32_t)(C * LdS * 2);
  constexpr uint32_t kVRingB = (uint32_t)STAGES * kVSlotB;
  constexpr uint32_t kSRingB = (uint32_t)STAGES * kSSlotB;
  const uint32_t vRowB = (uint32_t)v_row_bytes;
  constexpr uint32_t kDnRowB = (uint32_t)(kD * (int)sizeof(__nv_bfloat16));
  constexpr uint32_t kSRowB = (uint32_t)(BC * (int)sizeof(__nv_bfloat16));
  constexpr uint32_t kDvRowB = (uint32_t)(kD * (int)sizeof(float));
  constexpr uint32_t kDwRowB = (uint32_t)(BC * (int)sizeof(float));
  uint32_t stgV = 0, cmpV = 0, stgS = 0, cmpS = 0;

  const int NC = L / C;
  auto stage_chunk = [&](int c) {
    if (c >= 0) {
      const uint32_t tok0 = (uint32_t)(c * C);
      stV.issue(svB + stgV, vPlane, tok0, vRowB);
      stV.issue(sdnB + stgV, dnPlane, tok0, kDnRowB);
      stS.issue(srB + stgS, rPlane, tok0, kSRowB);
      stS.issue(swB + stgS, wPlane, tok0, kSRowB);
      if (tid < C) sdn[(stgV >> 1) + R0_DDEN(LdV, tid)] = ddPlane[(int)tok0 + tid];
    }
    stgV += kVSlotB;
    if (stgV == kVRingB) stgV = 0;
    stgS += kSSlotB;
    if (stgS == kSRingB) stgS = 0;
  };

#pragma unroll 1
  for (int s = 0; s < STAGES - 1; ++s) {
    stage_chunk(NC - 1 - s);
    stage_commit();
  }

#pragma unroll 1
  for (int c = NC - 1; c >= 0; --c) {
    stage_wait<STAGES - 2>();
    rendezvous();  // (1) this chunk's tiles are visible; the last visit's Grams and
                   //     published state are dead
    stage_chunk(c - (STAGES - 1));
    stage_commit();

    const uint32_t aVF = svB + cmpV + offFv, aVT = svB + cmpV + offTv;
    const uint32_t aDF = sdnB + cmpV + offFv, aDT = sdnB + cmpV + offTv;
    const uint32_t aRF = srB + cmpS + offFs;
    const uint32_t aWF = swB + cmpS + offFs, aWT = swB + cmpS + offTs;
    const __nv_bfloat16* const dd = sdn + (cmpV >> 1);
    const uint32_t tokBase = (uint32_t)(c * C);
    cmpV += kVSlotB;
    if (cmpV == kVRingB) cmpV = 0;
    cmpS += kSSlotB;
    if (cmpS == kSRingB) cmpS = 0;

    //: THE ENTRY ADJOINT, PUBLISHED ONCE PER VISIT -- the whole channel axis, which
    //: is what lets the per-slot write-side term finish inside one unit.
#pragma unroll
    for (int nj = 0; nj < NDV; ++nj) {
      const int s0 = nj * kMmaN + (lane & 3) * 2;
      const int c0 = warp * kMmaM + (lane >> 2);
      store_bf16x2(&sds[c0 * LdDs + s0], pack_bf16x2(dst[nj][0], dst[nj][1]));
      store_bf16x2(&sds[(c0 + 8) * LdDs + s0], pack_bf16x2(dst[nj][2], dst[nj][3]));
    }

    // ---- both Grams: Gu = staircase(R . W^T) over slots, Gd over channels -----
    if (warp < GWARPS) {
#pragma unroll
      for (int m = 0; m < MPW; ++m) {
        float gu[GPAIR][2][4];
        float gd[GPAIR][2][4];
#pragma unroll
        for (int u = 0; u < GPAIR; ++u)
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              gu[u][h][e] = 0.f;
              gd[u][h][e] = 0.f;
            }
#pragma unroll
        for (int kk = 0; kk < KD; ++kk) {
          uint32_t qa[4];
          load_frag(qa, R0_ADDR(aRF, LdS, (gm0 + m) * 16, 2 * kk));
#pragma unroll
          for (int u = 0; u < GPAIR; ++u) {
            uint32_t bf[4];
            load_frag(bf, R0_ADDR(aWT, LdS, (gn0 + u * 2) * 8, 2 * kk));
            mma(gu[u][0], qa, &bf[0]);
            mma(gu[u][1], qa, &bf[2]);
          }
        }
#pragma unroll
        for (int kk = 0; kk < KV; ++kk) {
          uint32_t qa[4];
          load_frag(qa, R0_ADDR(aDF, LdV, (gm0 + m) * 16, 2 * kk));
#pragma unroll
          for (int u = 0; u < GPAIR; ++u) {
            uint32_t bf[4];
            load_frag(bf, R0_ADDR(aVT, LdV, (gn0 + u * 2) * 8, 2 * kk));
            mma(gd[u][0], qa, &bf[0]);
            mma(gd[u][1], qa, &bf[2]);
          }
        }
        const int tr = (gm0 + m) * 16 + (lane >> 2);
        const float m0 = __bfloat162float(dd[ddG[m][0]]), m1 = __bfloat162float(dd[ddG[m][1]]);
#pragma unroll
        for (int u = 0; u < GPAIR; ++u)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int tc = (gn0 + u * 2 + h) * 8 + (lane & 3) * 2;
            store_bf16x2(&sgu[tr * LdG + tc], pack_bf16x2(R0_STAIR(gu[u][h][0], tc, tr),
                                                          R0_STAIR(gu[u][h][1], tc + 1, tr)));
            store_bf16x2(&sgu[(tr + 8) * LdG + tc],
                         pack_bf16x2(R0_STAIR(gu[u][h][2], tc, tr + 8),
                                     R0_STAIR(gu[u][h][3], tc + 1, tr + 8)));
            store_bf16x2(&sgd[tr * LdG + tc], pack_bf16x2(R0_STAIR(gd[u][h][0] + m0, tc, tr),
                                                          R0_STAIR(gd[u][h][1] + m0, tc + 1, tr)));
            store_bf16x2(&sgd[(tr + 8) * LdG + tc],
                         pack_bf16x2(R0_STAIR(gd[u][h][2] + m1, tc, tr + 8),
                                     R0_STAIR(gd[u][h][3] + m1, tc + 1, tr + 8)));
          }
      }
    }

    rendezvous();  // (2) the published adjoint and both Grams are visible

    // ---- dv: the readout with the write panel, plus the slot Gram's staircase --
    {
      float acc[NTOK][4];
#pragma unroll
      for (int j = 0; j < NTOK; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.f;
#pragma unroll
      for (int kk = 0; kk < KD; ++kk) {
        uint32_t da[4];
        da[0] = pack_bf16x2(dst[2 * kk][0], dst[2 * kk][1]);
        da[1] = pack_bf16x2(dst[2 * kk][2], dst[2 * kk][3]);
        da[2] = pack_bf16x2(dst[2 * kk + 1][0], dst[2 * kk + 1][1]);
        da[3] = pack_bf16x2(dst[2 * kk + 1][2], dst[2 * kk + 1][3]);
#pragma unroll
        for (int np = 0; np < NTOK / 2; ++np) {
          uint32_t wb[4];
          load_frag(wb, R0_ADDR(aWT, LdS, np * 16, 2 * kk));
          mma(acc[2 * np], da, &wb[0]);
          mma(acc[2 * np + 1], da, &wb[2]);
        }
      }
#pragma unroll
      for (int kk = 0; kk < MTOK; ++kk) {
        uint32_t a[4];
        load_frag_t(a, R0_ADDR(aDT, LdV, kk * 16, 2 * warp));
#pragma unroll
        for (int np = 0; np < NTOK / 2; ++np) {
          uint32_t b[4];
          load_frag_t(b, R0_ADDR(sguF, LdG, kk * 16, 2 * np));
          mma(acc[2 * np], a, &b[0]);
          mma(acc[2 * np + 1], a, &b[2]);
        }
      }
      const int c0 = warp * kMmaM + (lane >> 2);
#pragma unroll
      for (int j = 0; j < NTOK; ++j) {
        const int t0 = j * kMmaN + (lane & 3) * 2;
        float* p0 = reinterpret_cast<float*>(gather_ptr(
            dvPlane, (tokBase + (uint32_t)t0) * kDvRowB + (uint32_t)(c0 * (int)sizeof(float))));
        float* p1 = reinterpret_cast<float*>(gather_ptr(
            dvPlane,
            (tokBase + (uint32_t)(t0 + 1)) * kDvRowB + (uint32_t)(c0 * (int)sizeof(float))));
        store_f32(p0, acc[j][0]);
        store_f32(p1, acc[j][1]);
        store_f32(p0 + 8, acc[j][2]);
        store_f32(p1 + 8, acc[j][3]);
      }
    }

    // ---- dW, over this unit's slot run ---------------------------------------
    {
      float acc[MTOK][NSLOT][4];
#pragma unroll
      for (int m = 0; m < MTOK; ++m)
#pragma unroll
        for (int n = 0; n < NSLOT; ++n)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[m][n][e] = 0.f;
      const int n0 = warp * NSLOT;
      //: the INTER term: the whole channel axis, out of the published adjoint.
#pragma unroll
      for (int kk = 0; kk < KV; ++kk)
#pragma unroll
        for (int np = 0; np < NPAIR; ++np) {
          uint32_t sb[4];
          load_frag_t(sb, R0_ADDR(sdsF, LdDs, kk * 16, n0 + np * 2));
#pragma unroll
          for (int m = 0; m < MTOK; ++m) {
            uint32_t av[4];
            load_frag(av, R0_ADDR(aVF, LdV, m * 16, 2 * kk));
            mma(acc[m][2 * np], av, &sb[0]);
            mma(acc[m][2 * np + 1], av, &sb[2]);
          }
        }
      //: the INTRA term: the TRANSPOSED staircase against the read panel.
#pragma unroll
      for (int kk = 0; kk < MTOK; ++kk)
#pragma unroll
        for (int np = 0; np < NPAIR; ++np) {
          uint32_t rb[4];
          load_frag_t(rb, R0_ADDR(aRF, LdS, kk * 16, n0 + np * 2));
#pragma unroll
          for (int m = 0; m < MTOK; ++m) {
            uint32_t ga[4];
            load_frag_t(ga, R0_ADDR(sgdT, LdG, kk * 16, 2 * m));
            mma(acc[m][2 * np], ga, &rb[0]);
            mma(acc[m][2 * np + 1], ga, &rb[2]);
          }
        }
      //: the ENTRY ADJOINT MASS, `dz_a[s]`: per-slot and constant down the write
      //: tokens, so it is a broadcast along the accumulator's `m` axis.
#pragma unroll
      for (int m = 0; m < MTOK; ++m)
#pragma unroll
        for (int n = 0; n < NSLOT; ++n) {
          acc[m][n][0] += dzc[n][0];
          acc[m][n][1] += dzc[n][1];
          acc[m][n][2] += dzc[n][0];
          acc[m][n][3] += dzc[n][1];
        }
#pragma unroll
      for (int m = 0; m < MTOK; ++m)
#pragma unroll
        for (int n = 0; n < NSLOT; ++n) {
          const uint32_t colB =
              (uint32_t)(((n0 + n) * kMmaN + (lane & 3) * 2) * (int)sizeof(float));
          const int t0 = m * 16 + (lane >> 2);
          float* p0 = reinterpret_cast<float*>(
              gather_ptr(dwPlane, (tokBase + (uint32_t)t0) * kDwRowB + colB));
          float* p1 = reinterpret_cast<float*>(
              gather_ptr(dwPlane, (tokBase + (uint32_t)(t0 + 8)) * kDwRowB + colB));
          store_f32(p0 + 0, acc[m][n][0]);
          store_f32(p0 + 1, acc[m][n][1]);
          store_f32(p1 + 0, acc[m][n][2]);
          store_f32(p1 + 1, acc[m][n][3]);
        }
    }

    // ---- the fold: dS^T += dnum^T . R, and dz += dden^T . R ------------------
#pragma unroll
    for (int kk = 0; kk < MTOK; ++kk) {
      uint32_t a[4];
      load_frag_t(a, R0_ADDR(aDT, LdV, kk * 16, 2 * warp));
#pragma unroll
      for (int nj = 0; nj < KD; ++nj) {
        uint32_t b[4];
        load_frag_t(b, R0_ADDR(aRF, LdS, kk * 16, 2 * nj));
        mma(dst[2 * nj], a, &b[0]);
        mma(dst[2 * nj + 1], a, &b[2]);
      }
    }
    {
      const int n0 = warp * NSLOT;
#pragma unroll
      for (int kk = 0; kk < MTOK; ++kk) {
        //: the seed broadcast down `m`: a lane's A fragment is one k-pair repeated,
        //: and the pad's eight-wide groups are what make that pair adjacent.
        uint32_t da[4];
        da[0] = da[1] = *reinterpret_cast<const uint32_t*>(&dd[ddK[kk][0]]);
        da[2] = da[3] = *reinterpret_cast<const uint32_t*>(&dd[ddK[kk][1]]);
#pragma unroll
        for (int np = 0; np < NPAIR; ++np) {
          uint32_t rb[4];
          load_frag_t(rb, R0_ADDR(aRF, LdS, kk * 16, n0 + np * 2));
          mma(dzc[2 * np], da, &rb[0]);
          mma(dzc[2 * np + 1], da, &rb[2]);
        }
      }
    }
  }
}

#undef R0_ADDR
#undef R0_STAIR
#undef R0_DDEN

// THE RUNG'S ARM SET, GENERATED: tools/gen_shards.py owns the one list.
#include "../instantiations/ladder_r0_arms.inc"

}  // namespace rola::chunk
