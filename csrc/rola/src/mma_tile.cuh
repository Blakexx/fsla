#pragma once

// THE CONSUMER'S MMA TILE LAYER -- the fragment machinery and the four GEMMs' inner
// sequences, between the bare instruction wrappers of `mma.cuh` and the tile walk of
// `consumer_kernel.cuh`.
//
// What is here: the fragment-ordered A operand's element maps (`frag_a_slot` forward,
// `frag_a_get` inverse), the storage-format stores (`gm_put`, `stage_put`), the
// generic `[16 x 8]` tile, its fragment-ordered and `ldmatrix`-fed specializations,
// and the compile-time cross-checks that keep CuTe -- not this file -- the authority
// on what an `ldmatrix` and an `mma.sync` fragment actually contain.
//
// What is NOT here: where an operand LIVES. Every address these routines take comes
// from `smem_layout.cuh` (`vw_index`, `wt_index`, `stage_offset`), which is what makes
// it impossible for an atom and a scalar walk over the same buffer to disagree.
//
// The fragment index maps, the `ldmatrix` atoms and their `(m, k)` readings, and the
// bit-identity argument for each specialization: docs/internals/mma_tile.md

#include <cuda_bf16.h>

//: THE VENDORED CuTe PIN (`csrc/third_party/cutlass`, v3.5.1, closed-world rule 5),
//: a CODEGEN input rather than a packaging one. This layer uses it for the `ldmatrix`
//: COPY ATOMS and for the fragment maps the `static_assert`s below enumerate; the
//: `mma.sync` calls stay hand-rolled PTX and the correspondence is CHECKED against the
//: library's own maps rather than assumed. docs/internals/mma_tile.md#cute-pin
#include <cute/atom/copy_traits_sm75.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/tensor.hpp>

#include "mma.cuh"
#include "readout_mma.cuh"
#include "smem_layout.cuh"

namespace rola {
namespace consumer {

using rola::mma::mma_m16n8k16;
using rola::mma::pack_bf16x2;
//: §12 rule 3's format guard, taken from `readout_mma.cuh` rather than reimplemented:
//: it is the ONE clamp point on the bf16 arm.
using rola::clamp_bf16_finite;

//: The (m, k) that fragment slot `s` holds, and how many elements it packs.
template <MmaPrec PREC>
__device__ __forceinline__ void frag_a_slot(int s, int& m, int& k) {
  const int ks = s >> 7;
  const int rem = s & 127;
  const int lane = rem >> 2, w = rem & 3;
  m = (lane >> 2) + ((w & 1) << 3);
  const int r = lane & 3;
  k = (PREC == MmaPrec::kBf16) ? (16 * ks + ((w >> 1) << 3) + 2 * r)
                               : (8 * ks + ((w >> 1) << 2) + r);
}

//: One `[16 x 8]` tile against a fragment-ordered A. `d += Af * B`.
template <int KMAX, typename TB, bool B_T, MmaPrec PREC>
__device__ __forceinline__ void mma_tile_fragA(const unsigned* __restrict__ Af,
                                               const TB* __restrict__ B, int ldb, int K, int n0,
                                               int lane, float& d0, float& d1, float& d2,
                                               float& d3);

__device__ __forceinline__ float mma_src(float x) { return x; }
//: Store a computed fp32 value into a buffer whose element type follows a precision
//: axis. Overloaded rather than `static_cast`, because the torch build defines
//: `__CUDA_NO_BFLOAT16_CONVERSIONS__` -- the narrowing must be the explicit intrinsic.
//: docs/internals/smem_layout.md#stage-format
__device__ __forceinline__ void gm_put(float& d, float x) { d = x; }
__device__ __forceinline__ void gm_put(__nv_bfloat16& d, float x) { d = __float2bfloat16(x); }
__device__ __forceinline__ float mma_src(__nv_bfloat16 x) { return __bfloat162float(x); }
//: Store a STAGED OPERAND. On the fp32 arm this is the assignment it always was; on the
//: narrow arm it is where §12 rule 3's format guard now lives, because it is where the
//: fp32 value last exists -- a MOVE of the clamp, not a second one, and what makes the
//: `ldmatrix` fetch bit-identical to the generic path.
__device__ __forceinline__ void stage_put(float& d, float x) { d = x; }
__device__ __forceinline__ void stage_put(__nv_bfloat16& d, float x) {
  d = __float2bfloat16(clamp_bf16_finite(x));
}

//: `D[16x8] += A[16 x K] * B[K x 8]`, fp32 accumulate, one warp. `TA`/`TB` are the
//: operands' SMEM STORAGE types and are independent of `PREC`; `A_T`/`B_T` select
//: their storage ORIENTATION, because the four GEMMs of a tile need three different
//: ones and a transposed COPY of a `[BT x BC]` panel would be pure SMEM traffic.
//: Fragment index maps are the PTX `mma.sync.aligned` layouts taken VERBATIM from
//: `readout_mma.cuh` rather than re-derived. §12 rule 3's clamp applies on the bf16 arm
//: only (tf32 carries fp32's 8-bit exponent, so a clamp there is dead code).
//: docs/internals/mma_tile.md#mma-tile
template <typename TA, bool A_T, typename TB, bool B_T, MmaPrec PREC>
__device__ __forceinline__ void mma_tile(const TA* __restrict__ A, int lda,
                                         const TB* __restrict__ B, int ldb, int K, int m0, int n0,
                                         int lane, float& d0, float& d1, float& d2, float& d3) {
  const int g = lane >> 2;
  const int r = lane & 3;
  auto a_at = [&](int m, int k) { return mma_src(A_T ? A[k * lda + m] : A[m * lda + k]); };
  auto b_at = [&](int k, int n) { return mma_src(B_T ? B[n * ldb + k] : B[k * ldb + n]); };
  const int ra = m0 + g;
  const int nb = n0 + g;
  if (PREC == MmaPrec::kTf32x3) {
    // m16n8k8. A: a0..a3 -> (g, r), (g+8, r), (g, r+4), (g+8, r+4).
    //          B: b0, b1 -> (row r, col g), (row r+4, col g).
    for (int k0 = 0; k0 < K; k0 += 8) {
      mma::mma_m16n8k8_tf32x3(d0, d1, d2, d3, a_at(ra, k0 + r), a_at(ra + 8, k0 + r),
                              a_at(ra, k0 + r + 4), a_at(ra + 8, k0 + r + 4), b_at(k0 + r, nb),
                              b_at(k0 + r + 4, nb), d0, d1, d2, d3);
    }
  } else {
    // m16n8k16. A: a0..a3 pack (g, 2r|2r+1), (g+8, 2r|2r+1),
    //                          (g, 2r+8|2r+9), (g+8, 2r+8|2r+9).
    //           B: b0, b1 pack (row 2r|2r+1, col g), (row 2r+8|2r+9, col g).
    auto cl = clamp_bf16_finite;
    for (int k0 = 0; k0 < K; k0 += 16) {
      const int ca = k0 + r * 2;
      const unsigned a0 = pack_bf16x2(cl(a_at(ra, ca)), cl(a_at(ra, ca + 1)));
      const unsigned a1 = pack_bf16x2(cl(a_at(ra + 8, ca)), cl(a_at(ra + 8, ca + 1)));
      const unsigned a2 = pack_bf16x2(cl(a_at(ra, ca + 8)), cl(a_at(ra, ca + 9)));
      const unsigned a3 = pack_bf16x2(cl(a_at(ra + 8, ca + 8)), cl(a_at(ra + 8, ca + 9)));
      const unsigned b0 = pack_bf16x2(cl(b_at(ca, nb)), cl(b_at(ca + 1, nb)));
      const unsigned b1 = pack_bf16x2(cl(b_at(ca + 8, nb)), cl(b_at(ca + 9, nb)));
      mma_m16n8k16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
    }
  }
}

//: THE SAME `[16 x 8]` TILE WITH `Vw` AS B, FETCHED BY ONE `ldmatrix`.
//: `Vw` is B for BOTH of its consumers (GEMM 3 and GEMM 4) and both want the same
//: 16(k) x 8(n) block per k-step; `.trans` is right because `Vw` is stored `k`-major
//: while the fragment wants `k`-adjacent pairs at a fixed `n`. It is BIT-IDENTICAL to
//: the generic path, not merely equivalent -- `ldmatrix` delivers the STORED patterns
//: and `stage_put`'s clamp already guarantees they are finite bf16, so the composition
//: is the identity and no summation is reordered. The fragment correspondence is
//: `static_assert`ed against CuTe's own maps below, never assumed.
//: docs/internals/mma_tile.md#ldsm-atoms
template <int BT, int DV>
__device__ __forceinline__ void vw_frag_b(const __nv_bfloat16* __restrict__ Vw, int k0, int n0,
                                          int lane, unsigned& b0, unsigned& b1) {
  const __nv_bfloat16* src = Vw + vw_index<BT, DV, true>(k0 + (lane & 15), n0);
  cute::SM75_U16x4_LDSM_T::copy(*reinterpret_cast<const cute::uint128_t*>(src), b0, b1);
}

//: `WT` AS GEMM 2's B. The SAME atom against the SAME algebra: `WT` is
//: stored leaf-major and the fragment wants `k`-adjacent pairs at a fixed `n`, which is
//: `.trans` again. The only difference from `Vw` is which axis is called what.
template <int ROWS, int BT>
__device__ __forceinline__ void wt_frag_b(const __nv_bfloat16* __restrict__ WT, int k0, int n0,
                                          int lane, unsigned& b0, unsigned& b1) {
  const __nv_bfloat16* src = WT + wt_index<BT, true>(k0 + (lane & 15), n0);
  cute::SM75_U16x4_LDSM_T::copy(*reinterpret_cast<const cute::uint128_t*>(src), b0, b1);
}

//: `WT` AS GEMM 4's A, the atom's OTHER form: here `WT` is `k`-ADJACENT,
//: which is what `ldmatrix` reads WITHOUT transposing, so the fold's A operand is one
//: `ldmatrix.x4` for the whole `[16 m x 16 k]` block. THE ADDRESSING IS THE ATOM'S OWN
//: -- `SM75_U32x4_LDSM_N`'s source tile is FOUR 8x8 MATRICES STACKED, not a 16x16 block
//: -- and the `(m, k)` reading of those four is the CALLER's, checked over all 256
//: slots by `ldsm_n_dst_map_matches`: docs/internals/mma_tile.md#ldsm-atoms
template <int ROWS, int BT>
__device__ __forceinline__ void wt_frag_a(const __nv_bfloat16* __restrict__ WT, int m0, int k0,
                                          int lane, unsigned& a0, unsigned& a1, unsigned& a2,
                                          unsigned& a3) {
  const int row = m0 + 8 * ((lane >> 3) & 1) + (lane & 7);
  const int col = k0 + 8 * (lane >> 4);
  const __nv_bfloat16* src = WT + wt_index<BT, true>(row, col);
  cute::SM75_U32x4_LDSM_N::copy(*reinterpret_cast<const cute::uint128_t*>(src), a0, a1, a2, a3);
}

//: GEMM 4, THE FOLD, WITH BOTH OPERANDS FETCHED BY `ldmatrix`:
//: `S += WT^T . Vw`, A = `WT` (`.N`), B = `Vw` (`.trans`). NARROW ARM ONLY.
template <int ROWS, int BT, int DV>
__device__ __forceinline__ void mma_fold_tile(const __nv_bfloat16* __restrict__ WT,
                                              const __nv_bfloat16* __restrict__ Vw, int m0, int n0,
                                              int lane, float& d0, float& d1, float& d2,
                                              float& d3) {
#pragma unroll
  for (int k0 = 0; k0 < BT; k0 += 16) {
    unsigned a0, a1, a2, a3, b0, b1;
    wt_frag_a<ROWS, BT>(WT, m0, k0, lane, a0, a1, a2, a3);
    vw_frag_b<BT, DV>(Vw, k0, n0, lane, b0, b1);
    mma_m16n8k16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
  }
}

//: `d += A . Vw` for a generic (indexed) A and `Vw` as B -- GEMM 3. NARROW ARM ONLY:
//: the 16-bit atoms have no fp32 form, and `kNarrowStage` is precisely the arm on which
//: `Vw` is 16-bit. `K` is `BT` at both call sites, trip count compile-time.
template <typename TA, int BT, int DV>
__device__ __forceinline__ void mma_tile_vwB(const TA* __restrict__ A, int lda,
                                             const __nv_bfloat16* __restrict__ Vw, int m0, int n0,
                                             int lane, float& d0, float& d1, float& d2, float& d3) {
  const int g = lane >> 2;
  const int r = lane & 3;
  const int ra = m0 + g;
  auto a_at = [&](int m, int k) { return mma_src(A[m * lda + k]); };
  auto cl = clamp_bf16_finite;
#pragma unroll
  for (int k0 = 0; k0 < BT; k0 += 16) {
    const int ca = k0 + r * 2;
    const unsigned a0 = pack_bf16x2(cl(a_at(ra, ca)), cl(a_at(ra, ca + 1)));
    const unsigned a1 = pack_bf16x2(cl(a_at(ra + 8, ca)), cl(a_at(ra + 8, ca + 1)));
    const unsigned a2 = pack_bf16x2(cl(a_at(ra, ca + 8)), cl(a_at(ra, ca + 9)));
    const unsigned a3 = pack_bf16x2(cl(a_at(ra + 8, ca + 8)), cl(a_at(ra + 8, ca + 9)));
    unsigned b0, b1;
    vw_frag_b<BT, DV>(Vw, k0, n0, lane, b0, b1);
    mma_m16n8k16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
  }
}

//: THE TWO LIBRARY MAPS THIS FILE DEPENDS ON, ENUMERATED AT COMPILE TIME. Both must
//: equal `k = 2r + (v & 1) + 8 * (v >> 1)`, `n = g` -- which is `mma_tile`'s `b0`/`b1`,
//: letter for letter, so a library bump that moved either fails the BUILD rather than
//: the oracle: docs/internals/mma_tile.md#cute-crosschecks
constexpr bool mma_b_map_matches() {
  using B = typename cute::MMA_Traits<cute::SM80_16x8x16_F32BF16BF16F32_TN>::BLayout;
  for (int lane = 0; lane < 32; ++lane) {
    for (int v = 0; v < 4; ++v) {
      const int off = B{}(lane, v);
      if (off % 8 != (lane >> 2)) return false;  // n = g
      if (off / 8 != 2 * (lane & 3) + (v & 1) + 8 * (v >> 1)) return false;
    }
  }
  return true;
}

constexpr bool ldsm_t_dst_map_matches() {
  using Dst = typename cute::Copy_Traits<cute::SM75_U16x4_LDSM_T>::DstLayout;
  for (int lane = 0; lane < 32; ++lane) {
    for (int v = 0; v < 4; ++v) {
      const int e = Dst{}(lane, v * 16) / 16;  // bit -> half
      if (e % 8 != (lane >> 2)) return false;  // n = g
      if (e / 8 != 2 * (lane & 3) + (v & 1) + 8 * (v >> 1)) return false;
    }
  }
  return true;
}

constexpr bool ldsm_t_src_map_matches() {
  using Src = typename cute::Copy_Traits<cute::SM75_U16x4_LDSM_T>::SrcLayout;
  for (int lane = 0; lane < 32; ++lane) {
    if (Src{}(lane, 0) / 128 != (lane & 15)) return false;  // row = lane & 15
  }
  return true;
}

//: THE SAME QUESTION FOR THE `.N` ATOM AND THE A FRAGMENT: under the
//: `(m, k)` reading `wt_frag_a` implements, the destination map must equal `mma_tile`'s
//: `a0..a3` and the source map must be `row = lane`.
constexpr bool ldsm_n_dst_map_matches() {
  using Dst = typename cute::Copy_Traits<cute::SM75_U32x4_LDSM_N>::DstLayout;
  for (int lane = 0; lane < 32; ++lane) {
    for (int v = 0; v < 8; ++v) {  // 8 halves
      const int e = Dst{}(lane, v * 16) / 16;
      const int srow = e / 8, scol = e % 8, i = srow / 8;
      const int m_cute = 8 * (i & 1) + srow % 8, k_cute = 8 * (i >> 1) + scol;
      if (m_cute != (lane >> 2) + 8 * ((v / 2) & 1)) return false;
      if (k_cute != 2 * (lane & 3) + v % 2 + 8 * ((v / 2) >> 1)) return false;
    }
  }
  return true;
}

constexpr bool ldsm_n_src_map_matches() {
  using Src = typename cute::Copy_Traits<cute::SM75_U32x4_LDSM_N>::SrcLayout;
  for (int lane = 0; lane < 32; ++lane) {
    if (Src{}(lane, 0) / 128 != lane) return false;  // row = lane
  }
  return true;
}

static_assert(mma_b_map_matches(), "CuTe's B fragment map is not mma_tile's");
static_assert(ldsm_t_dst_map_matches(), "ldmatrix.x2.trans does not deliver b0/b1");
static_assert(ldsm_t_src_map_matches(), "ldmatrix.x2.trans row addresses moved");
static_assert(ldsm_n_dst_map_matches(), "ldmatrix.x4 does not deliver a0..a3");
static_assert(ldsm_n_src_map_matches(), "ldmatrix.x4 row addresses moved");

//: The `[16 x 8]` D/C fragment's element map, shared by every writeback below:
//:   `d0, d1 -> (row g,     col 2r, 2r+1)`
//:   `d2, d3 -> (row g + 8, col 2r, 2r+1)`
//: `d += Af . B` with A FRAGMENT-ORDERED (one `uint4` per k-step, zero address
//: arithmetic); B keeps the generic indexed path because each warp wants a different
//: `n`-slice.
//: **THE K-LOOP KEEPS A COMPILE-TIME TRIP COUNT AND AN EARLY EXIT, never a runtime
//: bound**: docs/internals/mma_tile.md#mma-tile-fraga
template <int KMAX, typename TB, bool B_T, MmaPrec PREC>
__device__ __forceinline__ void mma_tile_fragA(const unsigned* __restrict__ Af,
                                               const TB* __restrict__ B, int ldb, int K, int n0,
                                               int lane, float& d0, float& d1, float& d2,
                                               float& d3) {
  const int g = lane >> 2;
  const int r = lane & 3;
  const int nb = n0 + g;
  auto b_at = [&](int k, int n) { return mma_src(B_T ? B[n * ldb + k] : B[k * ldb + n]); };
  constexpr int kStep = (PREC == MmaPrec::kBf16) ? 16 : 8;
#pragma unroll
  for (int ks = 0; ks < KMAX / kStep; ++ks) {
    if (ks * kStep >= K) break;
    const uint4 a = *reinterpret_cast<const uint4*>(Af + ks * 128 + lane * 4);
    if (PREC == MmaPrec::kTf32x3) {
      mma::mma_m16n8k8_tf32x3(d0, d1, d2, d3, __uint_as_float(a.x), __uint_as_float(a.y),
                              __uint_as_float(a.z), __uint_as_float(a.w), b_at(ks * 8 + r, nb),
                              b_at(ks * 8 + r + 4, nb), d0, d1, d2, d3);
    } else {
      const int ca = ks * 16 + r * 2;
      const unsigned b0 =
          pack_bf16x2(clamp_bf16_finite(b_at(ca, nb)), clamp_bf16_finite(b_at(ca + 1, nb)));
      const unsigned b1 =
          pack_bf16x2(clamp_bf16_finite(b_at(ca + 8, nb)), clamp_bf16_finite(b_at(ca + 9, nb)));
      mma_m16n8k16(d0, d1, d2, d3, a.x, a.y, a.z, a.w, b0, b1, d0, d1, d2, d3);
    }
  }
}

//: GEMM 2 WITH A FRAGMENT-ORDERED A AND `WT` FETCHED BY `ldmatrix`.
//: `KMAX` is `WT`'s WINDOW row count and `K` the realized causal prefix,
//: so the early exit, its compile-time trip count and the k-order are the generic
//: path's, unchanged. NARROW ARM ONLY, for `vw_frag_b`'s reason. `KMAX % 16 == 0` is
//: what makes the fetch legal -- `ldmatrix` reads a WHOLE 16-row block.
template <int KMAX, int BT>
__device__ __forceinline__ void mma_tile_fragA_wtB(const unsigned* __restrict__ Af,
                                                   const __nv_bfloat16* __restrict__ WT, int K,
                                                   int n0, int lane, float& d0, float& d1,
                                                   float& d2, float& d3) {
  static_assert(KMAX % 16 == 0, "ldmatrix reads whole 16-row blocks of WT");
#pragma unroll
  for (int ks = 0; ks < KMAX / 16; ++ks) {
    if (ks * 16 >= K) break;
    const uint4 a = *reinterpret_cast<const uint4*>(Af + ks * 128 + lane * 4);
    unsigned b0, b1;
    wt_frag_b<KMAX, BT>(WT, ks * 16, n0, lane, b0, b1);
    mma_m16n8k16(d0, d1, d2, d3, a.x, a.y, a.z, a.w, b0, b1, d0, d1, d2, d3);
  }
}

//: Read element `(m, k)` back OUT of a fragment-ordered A operand. The mass column's
//: denominator is a `[BT x BC] x [BC x 1]` matvec, one column out of `DV + 1`, and an
//: `n8` MMA tile would waste 7/8 of a fragment -- so it stays a scalar reduction and
//: pays this inverse instead, ~10 integer ops per element once per tile.
template <MmaPrec PREC>
__device__ __forceinline__ float frag_a_get(const unsigned* Af, int m, int k) {
  //: HAZARD frag-a-get -- docs/internals/mma_tile.md#frag-a-get
  const int g = m & 7;
  const int w_lo = (m >> 3) & 1;
  if (PREC == MmaPrec::kBf16) {
    const int ks = k >> 4, kk = k & 15;
    const int lane = (g << 2) | ((kk & 7) >> 1);
    const int w = ((kk >= 8) << 1) | w_lo;
    const unsigned word = Af[ks * 128 + lane * 4 + w];
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&word);
    return __bfloat162float(h[kk & 1]);
  }
  const int ks = k >> 3, kk = k & 7;
  const int lane = (g << 2) | (kk & 3);
  const int w = ((kk >= 4) << 1) | w_lo;
  return __uint_as_float(Af[ks * 128 + lane * 4 + w]);
}

}  // namespace consumer
}  // namespace rola
