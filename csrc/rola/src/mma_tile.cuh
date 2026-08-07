#pragma once

// THE CONSUMER'S MMA TILE LAYER -- the fragment machinery and the four GEMMs' inner
// sequences, between the bare instruction wrappers of `mma.cuh` and the group walk of
// `consumer_kernel.cuh`.
//
// What is here: the fragment-ordered A operand's element map (`frag_a_slot`), the
// storage-format store (`stage_put`), the `[16 x 8]` tile against a fragment-ordered A,
// its `ldmatrix`-fed forms, the MASS COLUMN's broadcast-B forms, and the compile-time
// cross-checks that keep CuTe -- not this file -- the authority on what an `ldmatrix`
// and an `mma.sync` fragment actually contain.
//
// THE MASS COLUMN IS A MATRIX COLUMN. Its denominator terms are matvecs against a
// single column, and an `n8` tile carries a matvec by BROADCASTING that column across
// all eight of its `n` slots: every output column then holds the same value and the
// tile is read at `n = 0`. Seven eighths of the fragment is redundant and that is the
// whole price -- one `mma.sync` against the scalar reduction it replaces.
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
//: Store a bf16 OPERAND BUFFER's element -- where §12 rule 3's format guard lives,
//: because it is where the fp32 value last exists. THE CLAMP BELONGS AT THE WRITE:
//: every bf16 operand buffer is written once per element per visit and read by an
//: `ldmatrix` many times, so a read-side clamp would be a per-consumer repetition of
//: a per-producer fact -- and an `ldmatrix` has no place to put one. The intrinsic
//: rather than `static_cast`, because the torch build defines
//: `__CUDA_NO_BFLOAT16_CONVERSIONS__`.
__device__ __forceinline__ void stage_put(__nv_bfloat16& d, float x) {
  d = __float2bfloat16(clamp_bf16_finite(x));
}

//: THE `[16 x 8]` D/C fragment's element map, shared by every writeback:
//:   `d0, d1 -> (row g,     col 2r, 2r+1)`
//:   `d2, d3 -> (row g + 8, col 2r, 2r+1)`
//: `d += Af . B` with A FRAGMENT-ORDERED (one `uint4` per k-step, zero address
//: arithmetic); B keeps the generic indexed path because each warp wants a different
//: `n`-slice (`TB` is the B buffer's storage type -- `float` when B is the fp32
//: state, whose narrowing to the MMA operand happens HERE, at the fragment, because
//: the fp32 state is REWRITTEN WHOLE by every visit's fold and so has no read that
//: outlives its write).
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

//: A fp32 PAIR DELIVERED AS TWO bf16 WORDS -- the rounded value and its residual.
//: Two `mma.sync` against the same A fragment then carry ~16 mantissa bits of the B
//: operand instead of 8. §12 rule 3's guard applies to BOTH words: the residual of a
//: saturated value is itself out of range and must be narrowed the same way.
//: docs/internals/mma_tile.md#split-bf16x2
__device__ __forceinline__ void split_bf16x2(float x0, float x1, unsigned& hi, unsigned& lo) {
  const float c0 = clamp_bf16_finite(x0);
  const float c1 = clamp_bf16_finite(x1);
  const __nv_bfloat16 h0 = __float2bfloat16(c0);
  const __nv_bfloat16 h1 = __float2bfloat16(c1);
  hi = pack_bf16x2(__bfloat162float(h0), __bfloat162float(h1));
  lo = pack_bf16x2(clamp_bf16_finite(c0 - __bfloat162float(h0)),
                   clamp_bf16_finite(c1 - __bfloat162float(h1)));
}

//: THE MASS COLUMN's READOUT TILE, `d += Af . B[.][n0]`, WITH THE COLUMN COMPENSATED.
//: The B operand is a column of the fp32 STATE and it is the DENOMINATOR OF A RATIO:
//: unlike the numerator's columns it has nothing to cancel its rounding against, and
//: a single bf16 narrowing of it was MEASURED to put the bf16 family gate 12% over
//: budget at `T = 2048` on the decay arm. So the column rides `split_bf16x2` and the
//: tile is two `mma.sync` -- both broadcast across the tile's eight `n` slots, so
//: output column 0 carries the matvec.
//: docs/internals/mma_tile.md#mass-column
template <int KMAX>
__device__ __forceinline__ void mma_tile_fragA_massB(const unsigned* __restrict__ Af,
                                                     const float* __restrict__ B, int ldb, int K,
                                                     int n0, int lane, float& d0, float& d1,
                                                     float& d2, float& d3) {
  const int r = lane & 3;
#pragma unroll
  for (int ks = 0; ks < KMAX / kMmaKQuantum; ++ks) {
    if (ks * kMmaKQuantum >= K) break;
    const uint4 a = *reinterpret_cast<const uint4*>(Af + ks * 128 + lane * 4);
    const int ca = ks * 16 + r * 2;
    unsigned h0, l0, h1, l1;
    split_bf16x2(B[ca * ldb + n0], B[(ca + 1) * ldb + n0], h0, l0);
    split_bf16x2(B[(ca + 8) * ldb + n0], B[(ca + 9) * ldb + n0], h1, l1);
    mma_m16n8k16(d0, d1, d2, d3, a.x, a.y, a.z, a.w, h0, h1, d0, d1, d2, d3);
    mma_m16n8k16(d0, d1, d2, d3, a.x, a.y, a.z, a.w, l0, l1, d0, d1, d2, d3);
  }
}

//: THE MASS COLUMN's B, WHEN THE COLUMN IS A REGISTER-SIDE VECTOR: `v[k]` broadcast
//: across the tile's eight `n` slots. The B fragment's `k` map is
//: `2r + (w & 1) + 8 * (w >> 1)` and its `n` map is `g`, so a broadcast B is
//: `g`-INDEPENDENT -- four scalar loads and two packs for the whole fragment,
//: whatever the tile's `n0`. §12 rule 3's guard applies here because this is where
//: the fp32 row scale is narrowed. docs/internals/mma_tile.md#bcast-frag-b
__device__ __forceinline__ void bcast_frag_b(const float* __restrict__ v, int lane, unsigned& b0,
                                             unsigned& b1) {
  const int k0 = 2 * (lane & 3);
  b0 = pack_bf16x2(clamp_bf16_finite(v[k0]), clamp_bf16_finite(v[k0 + 1]));
  b1 = pack_bf16x2(clamp_bf16_finite(v[k0 + 8]), clamp_bf16_finite(v[k0 + 9]));
}

//: A `[16 m x 16 k]` bf16 A OPERAND FETCHED BY ONE `ldmatrix.x4` out of a PITCHED,
//: `k`-adjacent buffer -- the form `ldmatrix` reads WITHOUT transposing. THE
//: ADDRESSING IS THE ATOM'S OWN -- `SM75_U32x4_LDSM_N`'s source tile is FOUR 8x8
//: MATRICES STACKED, not a 16x16 block -- and the `(m, k)` reading of those four is
//: the CALLER's, checked over all 256 slots by `ldsm_n_dst_map_matches`. Both A
//: operands in this file are this shape: `WT`'s 16-row atom block (GEMM 4) and the
//: Gram (GEMM 3). docs/internals/mma_tile.md#ldsm-atoms
__device__ __forceinline__ void ldsm_frag_a(const __nv_bfloat16* __restrict__ A, int lda, int lane,
                                            unsigned& a0, unsigned& a1, unsigned& a2,
                                            unsigned& a3) {
  const int row = 8 * ((lane >> 3) & 1) + (lane & 7);
  const int col = 8 * (lane >> 4);
  cute::SM75_U32x4_LDSM_N::copy(*reinterpret_cast<const cute::uint128_t*>(A + row * lda + col), a0,
                                a1, a2, a3);
}

__device__ __forceinline__ void wt_frag_a(const __nv_bfloat16* __restrict__ WT16, int lane,
                                          unsigned& a0, unsigned& a1, unsigned& a2, unsigned& a3) {
  ldsm_frag_a(WT16, kWtPitch, lane, a0, a1, a2, a3);
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

//: THE MASS COLUMN's TILE, `d += A . v`, FOR AN `ldmatrix`-FETCHED A. ONE routine
//: serves both mass terms that contract over the group's tokens -- the fold's
//: (`A = WT`'s atom block, `v = sw`) and the intra term's (`A = Gm`, `v = sw`) --
//: because they are the same matvec against different pitched blocks. Output column
//: `0` (lanes with `r == 0`) carries rows `g` and `g + 8`.
__device__ __forceinline__ void mma_tile_bcastB(const __nv_bfloat16* __restrict__ A, int lda,
                                                const float* __restrict__ v, int lane, float& d0,
                                                float& d1, float& d2, float& d3) {
  unsigned a0, a1, a2, a3, b0, b1;
  ldsm_frag_a(A, lda, lane, a0, a1, a2, a3);
  bcast_frag_b(v, lane, b0, b1);
  mma_m16n8k16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
}

//: GEMM 3, `y += G . Vw`, with the Gram fetched by `ldmatrix` exactly as the fold
//: fetches `WT`: both are pitched, `k`-adjacent bf16 blocks, and
//: `pitched_ldsm_banks_ok` licenses the pitch. The Gram's §12 rule 3 guard is at its
//: STORE (`stage_put`), which is what makes an `ldmatrix` fetch legal here at all.
template <int DV>
__device__ __forceinline__ void mma_tile_vwB(const __nv_bfloat16* __restrict__ A, int lda,
                                             const __nv_bfloat16* __restrict__ Vw, int n0, int lane,
                                             float& d0, float& d1, float& d2, float& d3) {
  unsigned a0, a1, a2, a3, b0, b1;
  ldsm_frag_a(A, lda, lane, a0, a1, a2, a3);
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

}  // namespace consumer
}  // namespace rola
