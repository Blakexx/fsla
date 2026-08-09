// Clean-room dense chunked linear attention, sm_80/sm_86, bf16 in / fp32 accumulate.
//
// SEMANTICS (the whole of it):
//     O[t] = sum_{s <= t} (Q[t] . K[s]) * V[s]
// realised chunk-wise with a carried state S = sum_s K[s] (x) V[s]:
//     O_c  = Q_c . S                    readout against the ENTRY state
//     O_c += tril(Q_c . K_c^T) . V_c    within-chunk causal correction
//     S   += K_c^T . V_c                fold, at the chunk boundary
// No routing, no decay, no normalisation, no masks beyond causality.
//
// THE ONE STRUCTURAL DECISION EVERYTHING FOLLOWS FROM
// ---------------------------------------------------
// The state is held TRANSPOSED, S^T[d_v][d_k], as the fold's accumulator with
// m = d_v and n = d_k.  For mma.m16n8k16 the fp32 accumulator gives lane l the
// elements (m = l/4, l/4+8) x (n = 2*(l%4), +1); the readout needs S as a B
// operand, i.e. (n = d_v = l/4) x (k = d_k = 2*(l%4), +1, +8, +9).  With the
// transposed state those are the SAME registers: b0 = pack(acc[2j].c0, .c1),
// b1 = pack(acc[2j+1].c0, .c1) for d_v = l/4, and the .c2/.c3 halves for
// d_v = l/4+8.  The state therefore never leaves registers -- no shared-memory
// round trip, no shuffle, no transpose pass, one cvt.rn.bf16x2.f32 per b32.
//
// The same identity is used a second time for the Gram block: the fp32
// accumulator of Q.K^T is, after masking and conversion, exactly the A operand
// of G.V.  (It is not exploited here only because G is exchanged between warps;
// see the partition note.)
//
// PARTITION (4 warps, d_v = 64)
// -----------------------------
// Warp w owns d_v in [16w, 16w+16).  That slice of the state is PRIVATE to the
// warp for the whole sequence: the fold writes only it, the readout reads only
// it, and the output column block O[:, 16w:16w+16] is produced only by it.
// There is no cross-warp reduction anywhere and no state in shared memory.
// The one thing every warp needs in full is the Gram block G = tril(Q.K^T),
// whose m axis is tokens; it is computed once (token-split across warps) and
// exchanged through shared memory in bf16.
//
// FRAGMENT / ldmatrix CONVENTIONS
// ------------------------------
// ldmatrix.x4 delivers four 8x8 tiles, tile i addressed by lanes 8i..8i+7.
// Non-trans, one tile puts (row = l/4, col = 2*(l%4), +1) in a lane.
//   A operand from a row-major [m][k] source      : tile order (r,c),(r+8,c),(r,c+8),(r+8,c+8)   -> kRowFast
//   two B operands from a row-major [n][k] source : tile order (r,c),(r+8,c... ) see below       -> kColFast
// .trans returns the transposed tile, (col = l/4, row = 2*(l%4), +1).
//   A^T operand from a row-major [k][m] source    : kColFast
//   two B operands from a row-major [k][n] source : kRowFast
// so exactly two lane-offset forms are needed, selected by `ColFast`.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

namespace denseref {

// Bumped by hand whenever the source changes; read back device-side so a stale
// binary cannot masquerade as this source.  See build_stamp().
static constexpr int kBuildMagic = 570106;

static constexpr int kD = 64;         // d_k = d_v
static constexpr int kPad = 8;        // shared-memory row padding, in elements
static constexpr int kLd = kD + kPad; // 72 elements = 144 B: conflict-free ldmatrix
static constexpr int kWarps = 4;
static constexpr int kThreads = 32 * kWarps;

// ---------------------------------------------------------------- primitives

__device__ __forceinline__ uint32_t smem_addr(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldm4(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

__device__ __forceinline__ void ldm4t(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4],
                                         const uint32_t *b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void *src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Lane byte offset inside an ldmatrix.x4 group.  `ColFast` selects which of the
// two source axes the tile pair index advances along; see the header note.
template <int Ld, bool ColFast> __device__ __forceinline__ int ldm_lane_off(int lane) {
  const int tile = lane >> 3, row = lane & 7;
  const int r = ColFast ? (tile >> 1) * 8 + row : (tile & 1) * 8 + row;
  const int c = ColFast ? (tile & 1) * 8 : (tile >> 1) * 8;
  return (r * Ld + c) * 2;
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  const __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t *>(&p);
}

// ------------------------------------------------------------------- kernel

template <int C, int STAGES>
__global__ __launch_bounds__(kThreads) void dense_la_kernel(
    const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v, __nv_bfloat16 *__restrict__ o, int L) {
  static_assert(C % 16 == 0, "chunk must be a multiple of the mma m quantum");
  constexpr int MTOK = C / 16;   // token m-tiles in a chunk
  constexpr int NTOK = C / 8;    // token n-tiles in a chunk
  constexpr int KD = kD / 16;    // 4 -- d_k k-tiles
  constexpr int NDV = kD / 8;    // 8 -- d_v n-tiles (state n-tiles are d_k, also 8)
  constexpr int CLd = C + kPad;

  // Gram tile partition: split m first, spill the remainder onto n.
  constexpr int MSPLIT = MTOK >= kWarps ? kWarps : MTOK;
  constexpr int NSPLIT_RAW = kWarps / MSPLIT;
  constexpr int NSPLIT = NSPLIT_RAW < NTOK / 2 ? NSPLIT_RAW : NTOK / 2;
  constexpr int MPW = MTOK / MSPLIT;
  constexpr int NPW = NTOK / NSPLIT;
  constexpr int GWARPS = MSPLIT * NSPLIT;
  static_assert(NPW % 2 == 0, "n-tiles are consumed in ldmatrix pairs");

  extern __shared__ __align__(16) char smem_raw[];
  __nv_bfloat16 *sq = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
  __nv_bfloat16 *sk = sq + STAGES * C * kLd;
  __nv_bfloat16 *sv = sk + STAGES * C * kLd;
  __nv_bfloat16 *sg = sv + STAGES * C * kLd;
  __nv_bfloat16 *so = sg + C * CLd;

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const long base = (long)blockIdx.x * L * kD;
  const int NC = L / C;

  // Lane-invariant ldmatrix bases (one IADD3 each, then immediate offsets).
  const int offA = ldm_lane_off<kLd, false>(lane);   // A from [m][k]
  const int offB = ldm_lane_off<kLd, true>(lane);    // B pair from [n][k]
  const int offAT = ldm_lane_off<kLd, true>(lane);   // A^T from [k][m], .trans
  const int offBT = ldm_lane_off<kLd, false>(lane);  // B pair from [k][n], .trans
  const int offAg = ldm_lane_off<CLd, false>(lane);  // A from the Gram buffer

  const int gm0 = (warp / NSPLIT) * MPW;  // this warp's Gram m-tile base
  const int gn0 = (warp % NSPLIT) * NPW;  // ... and n-tile base
  const int dv0 = warp * 16;              // this warp's private d_v slice

  // The state, transposed: m = d_v (this warp's 16 rows), n = d_k (8 tiles).
  float st[NDV][4];
#pragma unroll
  for (int j = 0; j < NDV; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) st[j][e] = 0.f;

  // ---- global -> shared staging: one row is 128 B = 8 lanes x 16 B ---------
  // ADDRESSING DISCIPLINE. Every 64-bit address is formed ONCE, here, and then
  // only advanced by a constant stride; the ring slots are induction variables
  // in 32 bits with a compare-and-reset wrap, so there is no `% STAGES` and no
  // IMAD.WIDE anywhere inside the chunk loop. Everything below a chunk is a
  // compile-time immediate off one of these bases.
  constexpr int kRowsPerPass = kThreads / 8;          // 16
  constexpr int kPasses = C / kRowsPerPass;
  const int ldRow = tid >> 3, ldCol = (tid & 7) * 8;

  constexpr long kChunkElems = (long)C * kD;
  const __nv_bfloat16 *pq = q + base + (long)ldRow * kD + ldCol;
  const __nv_bfloat16 *pk = k + base + (long)ldRow * kD + ldCol;
  const __nv_bfloat16 *pv = v + base + (long)ldRow * kD + ldCol;
  __nv_bfloat16 *po = o + base + (long)ldRow * kD + ldCol;

  constexpr uint32_t kSlot = (uint32_t)(C * kLd) * 2;   // bytes of one ring slot
  constexpr uint32_t kRing = STAGES * kSlot;
  const uint32_t sqL = smem_addr(sq) + (uint32_t)((ldRow * kLd + ldCol) * 2);
  const uint32_t skL = smem_addr(sk) + (uint32_t)((ldRow * kLd + ldCol) * 2);
  const uint32_t svL = smem_addr(sv) + (uint32_t)((ldRow * kLd + ldCol) * 2);
  const uint32_t sqB = smem_addr(sq), skB = smem_addr(sk), svB = smem_addr(sv);
  const uint32_t sgA = smem_addr(sg) + (uint32_t)offAg;
  uint32_t stgSlot = 0, cmpSlot = 0;

  auto stage_chunk = [&]() {
#pragma unroll
    for (int p = 0; p < kPasses; ++p) {
      constexpr int dummy = 0; (void)dummy;
      const uint32_t so_ = stgSlot + (uint32_t)(p * kRowsPerPass * kLd * 2);
      const long go = (long)(p * kRowsPerPass * kD);
      cp_async16(sqL + so_, pq + go);
      cp_async16(skL + so_, pk + go);
      cp_async16(svL + so_, pv + go);
    }
    pq += kChunkElems; pk += kChunkElems; pv += kChunkElems;
    stgSlot += kSlot;
    if (stgSlot == kRing) stgSlot = 0;
  };

#pragma unroll 1
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < NC) stage_chunk();
    cp_commit();
  }

#pragma unroll 1
  for (int c = 0; c < NC; ++c) {
    if (c + STAGES - 1 < NC) stage_chunk();
    cp_commit();
    cp_wait<STAGES - 1>();
    __syncthreads();

    const uint32_t aq = sqB + cmpSlot + (uint32_t)offA;
    const uint32_t bk = skB + cmpSlot + (uint32_t)offB;
    const uint32_t btk = skB + cmpSlot + (uint32_t)offBT;
    const uint32_t atv = svB + cmpSlot + (uint32_t)offAT;
    const uint32_t btv = svB + cmpSlot + (uint32_t)offBT;
    cmpSlot += kSlot;
    if (cmpSlot == kRing) cmpSlot = 0;

    // ---- 1. Gram: G = tril(Q.K^T), this warp's tiles, straight into shared --
    if (warp < GWARPS) {
      uint32_t qa[MPW][KD][4];
#pragma unroll
      for (int m = 0; m < MPW; ++m)
#pragma unroll
        for (int kk = 0; kk < KD; ++kk)
          ldm4(reinterpret_cast<uint32_t(&)[4]>(qa[m][kk]),
               aq + ((gm0 + m) * 16 * kLd + kk * 16) * 2);

#pragma unroll
      for (int np = 0; np < NPW / 2; ++np) {
        const int n0 = gn0 + np * 2;
        uint32_t bf[KD][4];
#pragma unroll
        for (int kk = 0; kk < KD; ++kk)
          ldm4(reinterpret_cast<uint32_t(&)[4]>(bf[kk]),
               bk + (n0 * 8 * kLd + kk * 16) * 2);
#pragma unroll
        for (int m = 0; m < MPW; ++m) {
          float g0[4] = {0.f, 0.f, 0.f, 0.f}, g1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int kk = 0; kk < KD; ++kk) {
            mma16816(g0, reinterpret_cast<const uint32_t(&)[4]>(qa[m][kk]), &bf[kk][0]);
            mma16816(g1, reinterpret_cast<const uint32_t(&)[4]>(qa[m][kk]), &bf[kk][2]);
          }
          const int tr = (gm0 + m) * 16 + (lane >> 2);
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            float *g = half ? g1 : g0;
            const int tc = (n0 + half) * 8 + (lane & 3) * 2;
            const float m0 = (tc + 0 <= tr) ? g[0] : 0.f;
            const float m1 = (tc + 1 <= tr) ? g[1] : 0.f;
            const float m2 = (tc + 0 <= tr + 8) ? g[2] : 0.f;
            const float m3 = (tc + 1 <= tr + 8) ? g[3] : 0.f;
            *reinterpret_cast<uint32_t *>(&sg[tr * CLd + tc]) = pack_bf16(m0, m1);
            *reinterpret_cast<uint32_t *>(&sg[(tr + 8) * CLd + tc]) = pack_bf16(m2, m3);
          }
        }
      }
    }
    __syncthreads();

    // ---- 2. readout: O = Q . S, B operand straight out of the state regs ----
    float acc[MTOK][2][4];
    {
      uint32_t sb[2][KD][2];
#pragma unroll
      for (int p = 0; p < 2; ++p)
#pragma unroll
        for (int kk = 0; kk < KD; ++kk) {
          sb[p][kk][0] = p ? pack_bf16(st[2 * kk][2], st[2 * kk][3])
                           : pack_bf16(st[2 * kk][0], st[2 * kk][1]);
          sb[p][kk][1] = p ? pack_bf16(st[2 * kk + 1][2], st[2 * kk + 1][3])
                           : pack_bf16(st[2 * kk + 1][0], st[2 * kk + 1][1]);
        }
#pragma unroll
      for (int m = 0; m < MTOK; ++m) {
#pragma unroll
        for (int p = 0; p < 2; ++p)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[m][p][e] = 0.f;
#pragma unroll
        for (int kk = 0; kk < KD; ++kk) {
          uint32_t a[4];
          ldm4(a, aq + (m * 16 * kLd + kk * 16) * 2);
#pragma unroll
          for (int p = 0; p < 2; ++p) mma16816(acc[m][p], a, sb[p][kk]);
        }
      }
    }

    // ---- 3. intra: O += G . V ----------------------------------------------
    {
      uint32_t vb[MTOK][4];
#pragma unroll
      for (int kk = 0; kk < MTOK; ++kk)
        ldm4t(reinterpret_cast<uint32_t(&)[4]>(vb[kk]), btv + (kk * 16 * kLd + dv0) * 2);
#pragma unroll
      for (int m = 0; m < MTOK; ++m)
#pragma unroll
        for (int kk = 0; kk < MTOK; ++kk) {
          uint32_t a[4];
          ldm4(a, sgA + (uint32_t)((m * 16 * CLd + kk * 16) * 2));
#pragma unroll
          for (int p = 0; p < 2; ++p) mma16816(acc[m][p], a, &vb[kk][2 * p]);
        }
    }

    // ---- 4. fold: S^T += V^T . K -------------------------------------------
#pragma unroll
    for (int kk = 0; kk < MTOK; ++kk) {
      uint32_t a[4];
      ldm4t(a, atv + (kk * 16 * kLd + dv0) * 2);
#pragma unroll
      for (int nj = 0; nj < KD; ++nj) {
        uint32_t b[4];
        ldm4t(b, btk + (kk * 16 * kLd + nj * 16) * 2);
        mma16816(st[2 * nj], a, &b[0]);
        mma16816(st[2 * nj + 1], a, &b[2]);
      }
    }

    // ---- 5. output: stage through shared, then 128-bit coalesced stores -----
#pragma unroll
    for (int m = 0; m < MTOK; ++m) {
      const int tr = m * 16 + (lane >> 2);
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        const int dc = dv0 + p * 8 + (lane & 3) * 2;
        *reinterpret_cast<uint32_t *>(&so[tr * kLd + dc]) =
            pack_bf16(acc[m][p][0], acc[m][p][1]);
        *reinterpret_cast<uint32_t *>(&so[(tr + 8) * kLd + dc]) =
            pack_bf16(acc[m][p][2], acc[m][p][3]);
      }
    }
    __syncthreads();
#pragma unroll
    for (int p = 0; p < kPasses; ++p) {
      *reinterpret_cast<uint4 *>(po + (long)(p * kRowsPerPass * kD)) =
          *reinterpret_cast<const uint4 *>(&so[(ldRow + p * kRowsPerPass) * kLd + ldCol]);
    }
    po += kChunkElems;
  }
}


// ------------------------------------------------------- fragment convention probe
// One m16n8k16 pair per mode, so the ldmatrix/mma conventions asserted in the
// header are TESTED rather than assumed.  16x16 output = two n-tiles.
//   mode 0: A from a row-major [m][k] source (ldm4, offA)
//           B from a row-major [n][k] source (ldm4, offB)   -> D = A . Bsrc^T
//   mode 1: A from a row-major [k][m] source (ldm4t, offAT)
//           B from a row-major [k][n] source (ldm4t, offBT) -> D = Asrc^T . Bsrc
__global__ void probe_kernel(const __nv_bfloat16 *a_src, const __nv_bfloat16 *b_src,
                             float *d_out, int mode) {
  __shared__ __nv_bfloat16 sa[16 * kLd], sb[16 * kLd];
  const int tid = threadIdx.x, lane = tid & 31;
  for (int i = tid; i < 16 * 16; i += 32) {
    sa[(i / 16) * kLd + (i % 16)] = a_src[i];
    sb[(i / 16) * kLd + (i % 16)] = b_src[i];
  }
  __syncwarp();
  uint32_t af[4], bf[4];
  if (mode == 0) {
    ldm4(af, smem_addr(sa) + ldm_lane_off<kLd, false>(lane));
    ldm4(bf, smem_addr(sb) + ldm_lane_off<kLd, true>(lane));
  } else {
    ldm4t(af, smem_addr(sa) + ldm_lane_off<kLd, true>(lane));
    ldm4t(bf, smem_addr(sb) + ldm_lane_off<kLd, false>(lane));
  }
  float d0[4] = {0, 0, 0, 0}, d1[4] = {0, 0, 0, 0};
  mma16816(d0, af, &bf[0]);
  mma16816(d1, af, &bf[2]);
  const int r = lane >> 2, c = (lane & 3) * 2;
  d_out[r * 16 + c] = d0[0];
  d_out[r * 16 + c + 1] = d0[1];
  d_out[(r + 8) * 16 + c] = d0[2];
  d_out[(r + 8) * 16 + c + 1] = d0[3];
  d_out[r * 16 + c + 8] = d1[0];
  d_out[r * 16 + c + 9] = d1[1];
  d_out[(r + 8) * 16 + c + 8] = d1[2];
  d_out[(r + 8) * 16 + c + 9] = d1[3];
}

torch::Tensor probe(torch::Tensor a, torch::Tensor b, int64_t mode) {
  auto d = torch::zeros({16, 16}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  probe_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16 *>(a.data_ptr()),
      reinterpret_cast<const __nv_bfloat16 *>(b.data_ptr()), d.data_ptr<float>(), (int)mode);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return d;
}

// --------------------------------------------------------------- host side

static size_t smem_bytes(int C, int S) {
  return (size_t)(3 * S * C * kLd + C * (C + kPad) + C * kLd) * sizeof(__nv_bfloat16);
}

template <int C, int S>
static void launch(const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &v,
                   torch::Tensor &o, int BH, int L) {
  const size_t sm = smem_bytes(C, S);
  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(dense_la_kernel<C, S>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)sm));
    configured = true;
  }
  dense_la_kernel<C, S><<<BH, kThreads, sm, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16 *>(q.data_ptr()),
      reinterpret_cast<const __nv_bfloat16 *>(k.data_ptr()),
      reinterpret_cast<const __nv_bfloat16 *>(v.data_ptr()),
      reinterpret_cast<__nv_bfloat16 *>(o.data_ptr()), L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, int64_t chunk) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 && q.is_contiguous());
  TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes());
  TORCH_CHECK(q.dim() == 4 && q.size(3) == kD, "expect [B,H,L,64]");
  const int B = q.size(0), H = q.size(1), L = q.size(2);
  TORCH_CHECK(L % chunk == 0, "L must be a multiple of the chunk");
  auto o = torch::empty_like(q);
  const int BH = B * H;
  switch (chunk) {
    case 16: launch<16, 4>(q, k, v, o, BH, L); break;
    case 32: launch<32, 4>(q, k, v, o, BH, L); break;
    case 64: launch<64, 3>(q, k, v, o, BH, L); break;
    default: TORCH_CHECK(false, "unsupported chunk");
  }
  return o;
}

// Device-side build identity.  Two independent facts, neither of which a host
// path or a file hash can supply: (a) a magic constant WRITTEN BY A KERNEL that
// only this source compiles, and (b) the register/shared-memory census that
// ptxas produced for the very kernel entry the measurement will run.
__global__ void stamp_kernel(int *out) {
  out[0] = kBuildMagic;
  out[1] = kWarps * 1000 + kLd;
}

template <int C, int S> static void func_census(std::vector<int64_t> &r) {
  cudaFuncAttributes a{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, dense_la_kernel<C, S>));
  r.push_back(a.numRegs);
  r.push_back((int64_t)a.sharedSizeBytes);
  r.push_back(a.binaryVersion);
  r.push_back((int64_t)smem_bytes(C, S));
}

std::vector<int64_t> build_stamp(int64_t chunk) {
  auto t = torch::zeros({2}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  stamp_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(t.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto h = t.cpu();
  std::vector<int64_t> r{h.data_ptr<int>()[0], h.data_ptr<int>()[1]};
  switch (chunk) {
    case 16: func_census<16, 4>(r); break;
    case 32: func_census<32, 4>(r); break;
    case 64: func_census<64, 3>(r); break;
    default: TORCH_CHECK(false, "unsupported chunk"); break;
  }
  return r;
}

}  // namespace denseref

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &denseref::forward, "dense chunked linear attention");
  m.def("probe", &denseref::probe, "fragment convention probe");
  m.def("build_stamp", &denseref::build_stamp, "build identity");
}
