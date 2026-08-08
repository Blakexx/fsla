#pragma once

// THE CONSUMER'S MMA TILE LAYER -- the fragment machinery and the four GEMMs' inner
// sequences, between the bare instruction wrappers of `mma.cuh` and the group walk of
// `consumer_kernel.cuh`.
//
// What is here: the two `ldmatrix` operand fetchers every staged panel is read
// through, the storage-format store (`stage_put`), the STATE'S D-FRAGMENTS-AS-AN-A-
// OPERAND narrowing (`state_frag_a`), the four GEMMs' tile forms, the MASS COLUMN's
// broadcast forms, and the compile-time cross-checks that keep CuTe -- not this file
// -- the authority on what an `ldmatrix` and an `mma.sync` fragment actually contain.
//
// THE MASS COLUMN IS A MATRIX COLUMN. Its denominator terms are matvecs against a
// single column, and a tile carries a matvec by BROADCASTING that column across the
// A operand's sixteen `m` rows: every output row then holds the same value and the
// tile is read at one row. The redundancy is the whole price -- one `mma.sync`
// against the scalar reduction it replaces.
//
// bf16 IS THE ONLY OPERAND FORMAT (the one precision story: bf16 storage, fp32
// accumulation, fp64 reference as oracle); the tf32x3 paths died with the PREC axis.
//
// What is NOT here: where an operand LIVES. Every address these routines take comes
// from `smem_layout.cuh` (`rp_index`, `wp_index`, `vp_index` and the two pitches),
// which is what makes it impossible for an atom and a scalar walk over the same
// buffer to disagree. The STATE lives nowhere here at all: it is the caller's
// registers.
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

//: **THE TWO OPERAND FETCHERS, AND THERE ARE ONLY TWO.** Every staged buffer in this
//: kernel is a PITCHED ROW-MAJOR bf16 block, so an operand is either read `m`-major
//: by the non-transposing atom (A: `m` = row, `k` = column) or `k`-major by the
//: transposing one (B: `k` = row, `n` = column). `pitched_ldsm_banks_ok` at the
//: buffer's pitch is what licenses both; the fragment correspondence is
//: `static_assert`ed against CuTe's own maps below, never assumed.
//: docs/internals/mma_tile.md#ldsm-atoms
__device__ __forceinline__ void pitched_frag_a(const __nv_bfloat16* __restrict__ P, int lda, int m0,
                                               int k0, int lane, unsigned& a0, unsigned& a1,
                                               unsigned& a2, unsigned& a3) {
  const int row = m0 + 8 * ((lane >> 3) & 1) + (lane & 7);
  const int col = k0 + 8 * (lane >> 4);
  cute::SM75_U32x4_LDSM_N::copy(*reinterpret_cast<const cute::uint128_t*>(P + row * lda + col), a0,
                                a1, a2, a3);
}

__device__ __forceinline__ void pitched_frag_b(const __nv_bfloat16* __restrict__ P, int ldb, int k0,
                                               int n0, int lane, unsigned& b0, unsigned& b1) {
  const __nv_bfloat16* src = P + (k0 + (lane & 15)) * ldb + n0;
  cute::SM75_U16x4_LDSM_T::copy(*reinterpret_cast<const cute::uint128_t*>(src), b0, b1);
}

//: A fp32 PAIR DELIVERED AS TWO bf16 WORDS -- the rounded value and its residual.
//: Two `mma.sync` against the same operand fragment then carry ~16 mantissa bits
//: instead of 8. §12 rule 3's guard applies to BOTH words: the residual of a
//: saturated value is itself out of range and must be narrowed the same way. Its one
//: consumer is the MASS COLUMN, the denominator of a ratio, which has nothing to
//: cancel its rounding against. docs/internals/mma_tile.md#split-bf16x2
__device__ __forceinline__ void split_bf16x2(float x0, float x1, unsigned& hi, unsigned& lo) {
  const float c0 = clamp_bf16_finite(x0);
  const float c1 = clamp_bf16_finite(x1);
  const __nv_bfloat16 h0 = __float2bfloat16(c0);
  const __nv_bfloat16 h1 = __float2bfloat16(c1);
  hi = pack_bf16x2(__bfloat162float(h0), __bfloat162float(h1));
  lo = pack_bf16x2(clamp_bf16_finite(c0 - __bfloat162float(h0)),
                   clamp_bf16_finite(c1 - __bfloat162float(h1)));
}

//: **THE STATE'S D FRAGMENTS, READ AS AN A OPERAND. THIS IS WHY THE STATE IS HELD
//: TRANSPOSED.** `mma.sync`'s D map and its A map are the SAME map: D's
//: `(m = g | g + 8, n = 2r, 2r + 1)` over the tiles at `n0 = 0` and `n0 = 8` is
//: exactly A's `a0..a3` over `k = 0..15`. So the fold accumulates the state IN PLACE
//: in its own D fragments and the readout consumes them with NO LOAD AT ALL -- the
//: one format event left in the state's whole life is this narrowing, `4` words per
//: 16-leaf k-step, where the shared-memory state paid a merge and a split at every
//: element of every visit. `st` is one atom's eight floats: the 8-leaf tile at
//: `k = 0..7` then the one at `k = 8..15`.
//: docs/internals/mma_tile.md#state-frag-a
__device__ __forceinline__ void state_frag_a(const float* __restrict__ st, unsigned& a0,
                                             unsigned& a1, unsigned& a2, unsigned& a3) {
  a0 = pack_bf16x2(clamp_bf16_finite(st[0]), clamp_bf16_finite(st[1]));
  a1 = pack_bf16x2(clamp_bf16_finite(st[2]), clamp_bf16_finite(st[3]));
  a2 = pack_bf16x2(clamp_bf16_finite(st[4]), clamp_bf16_finite(st[5]));
  a3 = pack_bf16x2(clamp_bf16_finite(st[6]), clamp_bf16_finite(st[7]));
}

//: A REGISTER-SIDE VECTOR BROADCAST ACROSS A FRAGMENT. The B fragment's `k` map is
//: `2r + (v & 1) + 8 * (v >> 1)` and its `n` map is `g`, so a `k`-only vector is
//: `g`-INDEPENDENT: four scalar loads and two packs for the whole fragment. THE SAME
//: TWO WORDS SERVE AS AN A OPERAND -- A's `k` map is the same and its `m` map is `g`,
//: so the broadcast A fragment is `(w0, w0, w1, w1)`. §12 rule 3's guard applies
//: here because this is where the fp32 row scale is narrowed.
//: docs/internals/mma_tile.md#bcast-frag
__device__ __forceinline__ void bcast_frag(const float* __restrict__ v, int lane, unsigned& w0,
                                           unsigned& w1) {
  const int k0 = 2 * (lane & 3);
  w0 = pack_bf16x2(clamp_bf16_finite(v[k0]), clamp_bf16_finite(v[k0 + 1]));
  w1 = pack_bf16x2(clamp_bf16_finite(v[k0 + 8]), clamp_bf16_finite(v[k0 + 9]));
}

//: THE SAME BROADCAST, COMPENSATED, for the MASS COLUMN: it is the denominator of a
//: ratio and a single bf16 narrowing of it was MEASURED to put the bf16 family gate
//: 12% over budget at `T = 2048` on the decay arm, so it rides `split_bf16x2` at the
//: fragment and its tile is two `mma.sync`. docs/internals/mma_tile.md#mass-column
__device__ __forceinline__ void bcast_frag_split(const float* __restrict__ v, int lane,
                                                 unsigned& h0, unsigned& h1, unsigned& l0,
                                                 unsigned& l1) {
  const int k0 = 2 * (lane & 3);
  split_bf16x2(v[k0], v[k0 + 1], h0, l0);
  split_bf16x2(v[k0 + 8], v[k0 + 9], h1, l1);
}

//: **GEMM 1, THE READOUT: `y^T += S^T . Rp`.** A is the warp's own state registers, so
//: the k-loop's ONLY memory traffic is the B fragment -- and it is a TWO-STAGE
//: SOFTWARE PIPELINE: k-step `ks + 1`'s `ldmatrix`s are issued before k-step `ks`'s
//: `mma.sync`, so the shared-memory latency of a fetch is covered by the multiply of
//: its predecessor rather than by a sibling warp. One k-step is one ATOM.
//: **BOTH TOKEN TILES RIDE ONE A OPERAND, AND THAT IS THE ILP POINT**: the group's
//: sixteen tokens are two `n8` tiles over the SAME sixteen state channels, so the
//: narrowing happens once and the two `mma.sync` are INDEPENDENT -- back-to-back
//: issue from one warp, with no dependency gap for the scheduler to cover by
//: switching to another. docs/internals/mma_tile.md#gemm-readout
template <int KS>
__device__ __forceinline__ void mma_readout_tiles(const float* __restrict__ st,
                                                  const __nv_bfloat16* __restrict__ Rp, int leaf0,
                                                  int lane, float (&d)[2][4]) {
  unsigned b00, b01, b10, b11;
  auto fetch = [&](int ks) {
    pitched_frag_b(Rp, kTokPitch, leaf0 + ks * kMmaKQuantum, 0, lane, b00, b01);
    pitched_frag_b(Rp, kTokPitch, leaf0 + ks * kMmaKQuantum, 8, lane, b10, b11);
  };
  fetch(0);
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    const unsigned c00 = b00, c01 = b01, c10 = b10, c11 = b11;
    unsigned a0, a1, a2, a3;
    state_frag_a(st + ks * 8, a0, a1, a2, a3);
    if (ks + 1 < KS) fetch(ks + 1);
    mma_m16n8k16(d[0][0], d[0][1], d[0][2], d[0][3], a0, a1, a2, a3, c00, c01, d[0][0], d[0][1],
                 d[0][2], d[0][3]);
    mma_m16n8k16(d[1][0], d[1][1], d[1][2], d[1][3], a0, a1, a2, a3, c10, c11, d[1][0], d[1][1],
                 d[1][2], d[1][3]);
  }
}

//: **A TILE AGAINST AN A OPERAND THE CALLER ALREADY HOLDS.** GEMM 4's A is the group's
//: VALUE TILE at the warp's channel rows -- ONE fragment for EVERY state tile the warp
//: owns -- so the fold fetches it once per staged group and issues `2 * kAPW`
//: INDEPENDENT `mma.sync` against it, into `2 * kAPW` independent accumulators. That
//: is the deep unrolled form: one `ldmatrix.x4` where a per-tile routine paid one
//: each, and no dependency chain between consecutive multiplies.
//: docs/internals/mma_tile.md#gemm-fold
__device__ __forceinline__ void mma_fragA_pitchedB(unsigned a0, unsigned a1, unsigned a2,
                                                   unsigned a3, const __nv_bfloat16* __restrict__ B,
                                                   int ldb, int k0, int n0, int lane, float& d0,
                                                   float& d1, float& d2, float& d3) {
  unsigned b0, b1;
  pitched_frag_b(B, ldb, k0, n0, lane, b0, b1);
  mma_m16n8k16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
}

//: A `[16 x 8]` TILE FROM TWO PITCHED BUFFERS -- the shape three of the four GEMMs
//: reduce to now that every operand is pitched row-major: GEMM 2's Gram sub-block
//: (`A = Wp`, `B = Rp`), GEMM 3's intra term (`A = Vp`, `B = Gm`) and GEMM 4's fold
//: (`A = Vp`, `B = Wp`).
__device__ __forceinline__ void mma_pitched_tile(const __nv_bfloat16* __restrict__ A, int lda,
                                                 int m0, int k0,
                                                 const __nv_bfloat16* __restrict__ B, int ldb,
                                                 int kb0, int n0, int lane, float& d0, float& d1,
                                                 float& d2, float& d3) {
  unsigned a0, a1, a2, a3;
  pitched_frag_a(A, lda, m0, k0, lane, a0, a1, a2, a3);
  mma_fragA_pitchedB(a0, a1, a2, a3, B, ldb, kb0, n0, lane, d0, d1, d2, d3);
}

//: **THE SMEM -> REGISTER LINK, AS A CHAIN RATHER THAN A TILE.** Every accumulation in
//: this kernel that runs over a compile-time sequence of tiles is expressed here, and
//: every one of them is TWO-STAGE: item `i + 1`'s `ldmatrix` is issued BEFORE item
//: `i`'s `mma.sync`, so a fetch's shared-memory latency is covered by its
//: predecessor's multiply instead of by a sibling warp. That property is the
//: kernel's fundamental goal applied to link 2, and it is the reason a caller must
//: never loop over `mma_pitched_tile` -- a per-tile routine can only fetch-then-use.
//:
//: The sequence is described by four strides rather than by a lambda, because both
//: axis kinds occur: an operand advances by a SLOT (a pointer stride, `a_stride` /
//: `b_stride`) or along its own `k` axis inside one buffer (`kb_step`). GEMM 3's
//: intra term walks `Vp` by slot and `Gm` by `k`; GEMM 4's fold walks `Wp` by `n`
//: against ONE `Vp` fragment (`mma_fold_tiles` below).
//: docs/internals/mma_tile.md#mma-chains
//: `NB` is the sequence's COMPILE-TIME BOUND and `nb` its actual length -- the intra
//: term's length is the causal triangle's row, which is a compile-time constant at one
//: call site and a descriptor-derived value at the other. Both fold to the same code
//: when `nb` is known.
template <int NB>
__device__ __forceinline__ void mma_pitched_chain(const __nv_bfloat16* __restrict__ A, int lda,
                                                  int a_stride, int m0,
                                                  const __nv_bfloat16* __restrict__ B, int ldb,
                                                  int b_stride, int kb0, int kb_step, int n0,
                                                  int nb, int lane, float& d0, float& d1, float& d2,
                                                  float& d3) {
  unsigned a0, a1, a2, a3, b0, b1;
  auto fetch = [&](int i) {
    pitched_frag_a(A + i * a_stride, lda, m0, 0, lane, a0, a1, a2, a3);
    pitched_frag_b(B + i * b_stride, ldb, kb0 + i * kb_step, n0, lane, b0, b1);
  };
  fetch(0);
#pragma unroll
  for (int i = 0; i < NB; ++i) {
    if (i >= nb) break;
    const unsigned x0 = a0, x1 = a1, x2 = a2, x3 = a3, c0 = b0, c1 = b1;
    if (i + 1 < nb) fetch(i + 1);
    mma_m16n8k16(d0, d1, d2, d3, x0, x1, x2, x3, c0, c1, d0, d1, d2, d3);
  }
}

//: THE SAME CHAIN WITH A BROADCAST A -- the two denominator terms that contract over
//: TOKENS. The vector advances by `v_stride` floats per item (`sw` of the next staged
//: group); the B operand takes the same two strides as above.
template <int NB>
__device__ __forceinline__ void mma_bcastA_chain(const float* __restrict__ v, int v_stride,
                                                 const __nv_bfloat16* __restrict__ B, int ldb,
                                                 int b_stride, int kb0, int kb_step, int n0, int nb,
                                                 int lane, float& d0, float& d1, float& d2,
                                                 float& d3) {
  unsigned w0, w1, b0, b1;
  auto fetch = [&](int i) {
    bcast_frag(v + i * v_stride, lane, w0, w1);
    pitched_frag_b(B + i * b_stride, ldb, kb0 + i * kb_step, n0, lane, b0, b1);
  };
  fetch(0);
#pragma unroll
  for (int i = 0; i < NB; ++i) {
    if (i >= nb) break;
    const unsigned x0 = w0, x1 = w1, c0 = b0, c1 = b1;
    if (i + 1 < nb) fetch(i + 1);
    mma_m16n8k16(d0, d1, d2, d3, x0, x0, x1, x1, c0, c1, d0, d1, d2, d3);
  }
}

//: **GEMM 4's FOLD SEQUENCE FOR ONE STAGED GROUP.** The `2 * APW` 8-leaf tiles the warp
//: owns share ONE A operand (the value tile at the warp's channel rows, which the
//: caller already holds), so the sequence is a pure B walk along `Wp`'s leaf axis --
//: and it is the two-stage chain, which is what the register-resident state's
//: independent accumulators exist to make possible. The ATOM SKIP rides the pipeline:
//: a dead atom's fetch is not issued either, so the mask still costs its own tiles
//: nothing. docs/internals/mma_tile.md#gemm-fold
template <int APW>
__device__ __forceinline__ void mma_fold_tiles(unsigned va0, unsigned va1, unsigned va2,
                                               unsigned va3, const __nv_bfloat16* __restrict__ Wp,
                                               int ldw, int atom0, unsigned live, int lane,
                                               float* __restrict__ st) {
  unsigned b0 = 0u, b1 = 0u;
  auto on = [&](int s) { return ((live >> (atom0 + (s >> 1))) & 1u) != 0u; };
  auto fetch = [&](int s) {
    pitched_frag_b(Wp, ldw, 0, (atom0 + (s >> 1)) * kMmaKQuantum + 8 * (s & 1), lane, b0, b1);
  };
  if (on(0)) fetch(0);
#pragma unroll
  for (int s = 0; s < 2 * APW; ++s) {
    const unsigned c0 = b0, c1 = b1;
    if (s + 1 < 2 * APW && on(s + 1)) fetch(s + 1);
    if (!on(s)) continue;
    float* const d = st + (s >> 1) * 8 + (s & 1) * 4;
    mma_m16n8k16(d[0], d[1], d[2], d[3], va0, va1, va2, va3, c0, c1, d[0], d[1], d[2], d[3]);
  }
}

//: GEMM 2's DENSE ARM, AS A TWO-STAGE PIPELINE: every atom is live, so the whole atom
//: sequence is known at compile time and atom `q + 1`'s fragments are issued before
//: atom `q`'s multiply. This is the arm `mask == kFullMask` selects, and it is where
//: `BC = 64`'s four atoms are four chained `ldmatrix` -> `mma.sync` dependencies
//: otherwise.
template <int QPO>
__device__ __forceinline__ void mma_gram_atoms(const __nv_bfloat16* __restrict__ Wp, int ldw,
                                               const __nv_bfloat16* __restrict__ Rp, int n0,
                                               int lane, float& d0, float& d1, float& d2,
                                               float& d3) {
  unsigned a0, a1, a2, a3, b0, b1;
  auto fetch = [&](int q) {
    pitched_frag_a(Wp, ldw, 0, q * kMmaKQuantum, lane, a0, a1, a2, a3);
    pitched_frag_b(Rp, kTokPitch, q * kMmaKQuantum, n0, lane, b0, b1);
  };
  fetch(0);
#pragma unroll
  for (int q = 0; q < QPO; ++q) {
    const unsigned x0 = a0, x1 = a1, x2 = a2, x3 = a3, c0 = b0, c1 = b1;
    if (q + 1 < QPO) fetch(q + 1);
    mma_m16n8k16(d0, d1, d2, d3, x0, x1, x2, x3, c0, c1, d0, d1, d2, d3);
  }
}

//: THE MASS COLUMN'S READOUT TILE, `d += broadcast(Smass) . Rp`, WITH THE COLUMN
//: COMPENSATED: two `mma.sync` per k-step against the hi and lo halves of the same
//: column. Its k-loop is the SAME two-stage pipeline the numerator's is.
template <int KS>
__device__ __forceinline__ void mma_mass_readout(const float* __restrict__ m,
                                                 const __nv_bfloat16* __restrict__ Rp, int leaf0,
                                                 int n0, int lane, float& d0, float& d1, float& d2,
                                                 float& d3) {
  unsigned h0, h1, l0, l1, b0, b1;
  auto fetch = [&](int ks) {
    bcast_frag_split(m + leaf0 + ks * kMmaKQuantum, lane, h0, h1, l0, l1);
    pitched_frag_b(Rp, kTokPitch, leaf0 + ks * kMmaKQuantum, n0, lane, b0, b1);
  };
  fetch(0);
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    const unsigned x0 = h0, x1 = h1, y0 = l0, y1 = l1, c0 = b0, c1 = b1;
    if (ks + 1 < KS) fetch(ks + 1);
    mma_m16n8k16(d0, d1, d2, d3, x0, x0, x1, x1, c0, c1, d0, d1, d2, d3);
    mma_m16n8k16(d0, d1, d2, d3, y0, y0, y1, y1, c0, c1, d0, d1, d2, d3);
  }
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
//: reading `pitched_frag_a` implements, the destination map must equal the `a0..a3`
//: packing and the source map must be `row = lane`. It is also the map
//: `state_frag_a` packs INTO, which is what makes the fold's D fragments a legal A
//: operand.
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
