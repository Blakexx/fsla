#pragma once

// THE CONSUMER'S MMA TILE LAYER -- the fragment machinery and the four GEMMs' inner
// sequences, between the bare instruction wrappers of `mma.cuh` and the group walk of
// `consumer_kernel.cuh`.
//
// What is here: the fragment-ordered A operand's element maps (`frag_a_slot` forward,
// `frag_a_get` inverse), the storage-format stores (`gm_put`, `stage_put`), the
// `[16 x 8]` tile against a fragment-ordered A, its `ldmatrix`-fed forms, and the
// compile-time cross-checks that keep CuTe -- not this file -- the authority on what
// an `ldmatrix` and an `mma.sync` fragment actually contain.
//
// bf16 IS THE ONLY OPERAND FORMAT (the one precision story: bf16 storage, fp32
// accumulation, fp64 reference as oracle); the tf32x3 paths died with the PREC axis.
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
//: it is the ONE clamp point.
using rola::clamp_bf16_finite;

//: The (m, k) that fragment slot `s` holds. One k-step = one ATOM (`kMmaKQuantum`
//: leaves), 128 words per step.
__device__ __forceinline__ void frag_a_slot(int s, int& m, int& k) {
  const int ks = s >> 7;
  const int rem = s & 127;
  const int lane = rem >> 2, w = rem & 3;
  m = (lane >> 2) + ((w & 1) << 3);
  const int r = lane & 3;
  k = 16 * ks + ((w >> 1) << 3) + 2 * r;
}

__device__ __forceinline__ float mma_src(__nv_bfloat16 x) { return __bfloat162float(x); }
//: Store a computed fp32 value into a bf16 buffer. The intrinsic rather than
//: `static_cast`, because the torch build defines `__CUDA_NO_BFLOAT16_CONVERSIONS__`.
__device__ __forceinline__ void gm_put(__nv_bfloat16& d, float x) { d = __float2bfloat16(x); }
//: Store a STAGED OPERAND -- where §12 rule 3's format guard lives, because it is
//: where the fp32 value last exists: a MOVE of the clamp, not a second one, and what
//: makes the `ldmatrix` fetch bit-identical to the indexed path.
__device__ __forceinline__ void stage_put(__nv_bfloat16& d, float x) {
  d = __float2bfloat16(clamp_bf16_finite(x));
}

//: THE `[16 x 8]` D/C fragment's element map, shared by every writeback:
//:   `d0, d1 -> (row g,     col 2r, 2r+1)`
//:   `d2, d3 -> (row g + 8, col 2r, 2r+1)`
//: `d += Af . B` with A FRAGMENT-ORDERED (one `uint4` per k-step, zero address
//: arithmetic); B keeps the generic indexed path because each warp wants a different
//: `n`-slice (`TB` is the B buffer's storage type -- `float` when B is the fp32
//: state, whose narrowing to the MMA operand happens HERE, at the fragment).
//: **THE K-LOOP KEEPS A COMPILE-TIME TRIP COUNT AND AN EARLY EXIT, never a runtime
//: bound**: docs/internals/mma_tile.md#mma-tile-fraga
template <int KMAX, typename TB, bool B_T>
__device__ __forceinline__ void mma_tile_fragA(const unsigned* __restrict__ Af,
                                               const TB* __restrict__ B, int ldb, int K, int n0,
                                               int lane, float& d0, float& d1, float& d2,
                                               float& d3) {
  const int g = lane >> 2;
  const int r = lane & 3;
  const int nb = n0 + g;
  auto b_val = [&](int k, int n) -> float {
    if constexpr (B_T) {
      return B[n * ldb + k];
    } else {
      return B[k * ldb + n];
    }
  };
#pragma unroll
  for (int ks = 0; ks < KMAX / kMmaKQuantum; ++ks) {
    if (ks * kMmaKQuantum >= K) break;
    const uint4 a = *reinterpret_cast<const uint4*>(Af + ks * 128 + lane * 4);
    const int ca = ks * 16 + r * 2;
    const unsigned b0 =
        pack_bf16x2(clamp_bf16_finite(b_val(ca, nb)), clamp_bf16_finite(b_val(ca + 1, nb)));
    const unsigned b1 =
        pack_bf16x2(clamp_bf16_finite(b_val(ca + 8, nb)), clamp_bf16_finite(b_val(ca + 9, nb)));
    mma_m16n8k16(d0, d1, d2, d3, a.x, a.y, a.z, a.w, b0, b1, d0, d1, d2, d3);
  }
}

//: `Vw` AS B, FETCHED BY ONE `ldmatrix`. `Vw` is B for BOTH of its consumers
//: (GEMM 3 and GEMM 4) and both want the same 16(k) x 8(n) block per k-step;
//: `.trans` is right because `Vw` is stored `k`-major while the fragment wants
//: `k`-adjacent pairs at a fixed `n`. BIT-IDENTICAL to the indexed path, not merely
//: equivalent -- `ldmatrix` delivers the STORED patterns and `stage_put`'s clamp
//: already guarantees they are finite bf16. The fragment correspondence is
//: `static_assert`ed against CuTe's own maps below, never assumed.
//: docs/internals/mma_tile.md#ldsm-atoms
template <int DV>
__device__ __forceinline__ void vw_frag_b(const __nv_bfloat16* __restrict__ Vw, int k0, int n0,
                                          int lane, unsigned& b0, unsigned& b1) {
  const __nv_bfloat16* src = Vw + vw_index<DV>(k0 + (lane & 15), n0);
  cute::SM75_U16x4_LDSM_T::copy(*reinterpret_cast<const cute::uint128_t*>(src), b0, b1);
}

//: `WT` AS GEMM 2's B. The SAME atom against the SAME algebra: `WT` is stored
//: leaf-major and the fragment wants `k`-adjacent pairs at a fixed `n`, which is
//: `.trans` again. `WT16` points at the 16-ROW BLOCK (one atom) being contracted.
__device__ __forceinline__ void wt_frag_b(const __nv_bfloat16* __restrict__ WT16, int n0, int lane,
                                          unsigned& b0, unsigned& b1) {
  const __nv_bfloat16* src = WT16 + wt_index(lane & 15, n0);
  cute::SM75_U16x4_LDSM_T::copy(*reinterpret_cast<const cute::uint128_t*>(src), b0, b1);
}

//: `WT` AS GEMM 4's A, the atom's OTHER form: here `WT` is `k`-ADJACENT, which is
//: what `ldmatrix` reads WITHOUT transposing, so the fold's A operand is one
//: `ldmatrix.x4` for the whole `[16 m x 16 k]` block. THE ADDRESSING IS THE ATOM'S
//: OWN -- `SM75_U32x4_LDSM_N`'s source tile is FOUR 8x8 MATRICES STACKED, not a
//: 16x16 block -- and the `(m, k)` reading of those four is the CALLER's, checked
//: over all 256 slots by `ldsm_n_dst_map_matches`:
//: docs/internals/mma_tile.md#ldsm-atoms
__device__ __forceinline__ void wt_frag_a(const __nv_bfloat16* __restrict__ WT16, int lane,
                                          unsigned& a0, unsigned& a1, unsigned& a2, unsigned& a3) {
  const int row = 8 * ((lane >> 3) & 1) + (lane & 7);
  const int col = 8 * (lane >> 4);
  const __nv_bfloat16* src = WT16 + wt_index(row, col);
  cute::SM75_U32x4_LDSM_N::copy(*reinterpret_cast<const cute::uint128_t*>(src), a0, a1, a2, a3);
}

//: GEMM 4, THE FOLD, WITH BOTH OPERANDS FETCHED BY `ldmatrix`:
//: `S += WT^T . Vw` for ONE ATOM's 16-row block of `WT` (the contraction axis is the
//: group's `kGroupTokens == 16` tokens: exactly one k-step).
template <int DV>
__device__ __forceinline__ void mma_fold_tile(const __nv_bfloat16* __restrict__ WT16,
                                              const __nv_bfloat16* __restrict__ Vw, int n0,
                                              int lane, float& d0, float& d1, float& d2,
                                              float& d3) {
  unsigned a0, a1, a2, a3, b0, b1;
  wt_frag_a(WT16, lane, a0, a1, a2, a3);
  vw_frag_b<DV>(Vw, 0, n0, lane, b0, b1);
  mma_m16n8k16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
}

//: `d += A . Vw` for a generic (indexed) A and `Vw` as B -- GEMM 3. The contraction
//: is the group's 16 tokens: one k-step.
template <typename TA, int DV>
__device__ __forceinline__ void mma_tile_vwB(const TA* __restrict__ A, int lda,
                                             const __nv_bfloat16* __restrict__ Vw, int m0, int n0,
                                             int lane, float& d0, float& d1, float& d2, float& d3) {
  const int g = lane >> 2;
  const int r = lane & 3;
  const int ra = m0 + g;
  auto a_at = [&](int m, int k) { return mma_src(A[m * lda + k]); };
  auto cl = clamp_bf16_finite;
  const int ca = r * 2;
  const unsigned a0 = pack_bf16x2(cl(a_at(ra, ca)), cl(a_at(ra, ca + 1)));
  const unsigned a1 = pack_bf16x2(cl(a_at(ra + 8, ca)), cl(a_at(ra + 8, ca + 1)));
  const unsigned a2 = pack_bf16x2(cl(a_at(ra, ca + 8)), cl(a_at(ra, ca + 9)));
  const unsigned a3 = pack_bf16x2(cl(a_at(ra + 8, ca + 8)), cl(a_at(ra + 8, ca + 9)));
  unsigned b0, b1;
  vw_frag_b<DV>(Vw, 0, n0, lane, b0, b1);
  mma_m16n8k16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
}

//: GEMM 2 WITH A FRAGMENT-ORDERED A AND `WT` FETCHED BY `ldmatrix`, for ONE ATOM:
//: `Af16` is the atom's 128-word fragment block, `WT16` its 16-row `WT` block.
__device__ __forceinline__ void mma_tile_fragA_wtB(const unsigned* __restrict__ Af16,
                                                   const __nv_bfloat16* __restrict__ WT16, int n0,
                                                   int lane, float& d0, float& d1, float& d2,
                                                   float& d3) {
  const uint4 a = *reinterpret_cast<const uint4*>(Af16 + lane * 4);
  unsigned b0, b1;
  wt_frag_b(WT16, n0, lane, b0, b1);
  mma_m16n8k16(d0, d1, d2, d3, a.x, a.y, a.z, a.w, b0, b1, d0, d1, d2, d3);
}

//: THE LIBRARY MAPS THIS FILE DEPENDS ON, ENUMERATED AT COMPILE TIME. Both `.trans`
//: checks must equal `k = 2r + (v & 1) + 8 * (v >> 1)`, `n = g` -- which is the
//: hand-rolled `b0`/`b1`, letter for letter, so a library bump that moved either
//: fails the BUILD rather than the oracle: docs/internals/mma_tile.md#cute-crosschecks
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

//: THE SAME QUESTION FOR THE `.N` ATOM AND THE A FRAGMENT: under the `(m, k)`
//: reading `wt_frag_a` implements, the destination map must equal the `a0..a3`
//: packing and the source map must be `row = lane`.
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

static_assert(mma_b_map_matches(), "CuTe's B fragment map is not the hand-rolled one");
static_assert(ldsm_t_dst_map_matches(), "ldmatrix.x2.trans does not deliver b0/b1");
static_assert(ldsm_t_src_map_matches(), "ldmatrix.x2.trans row addresses moved");
static_assert(ldsm_n_dst_map_matches(), "ldmatrix.x4 does not deliver a0..a3");
static_assert(ldsm_n_src_map_matches(), "ldmatrix.x4 row addresses moved");

//: Read element `(m, k)` back OUT of a fragment-ordered A operand. The mass column's
//: denominator is a `[16 x BC] x [BC x 1]` matvec, one column out of `DV + 1`, and an
//: `n8` MMA tile would waste 7/8 of a fragment -- so it stays a scalar reduction and
//: pays this inverse instead, ~10 integer ops per element once per visit.
__device__ __forceinline__ float frag_a_get(const unsigned* Af, int m, int k) {
  //: HAZARD frag-a-get -- docs/internals/mma_tile.md#frag-a-get
  const int g = m & 7;
  const int w_lo = (m >> 3) & 1;
  const int ks = k >> 4, kk = k & 15;
  const int lane = (g << 2) | ((kk & 7) >> 1);
  const int w = ((kk >= 8) << 1) | w_lo;
  const unsigned word = Af[ks * 128 + lane * 4 + w];
  const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&word);
  return __bfloat162float(h[kk & 1]);
}

}  // namespace consumer
}  // namespace rola
