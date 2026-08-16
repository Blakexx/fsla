// THE PER-ROW EXACT SOLVE COMPONENTS -- one token's row on one logical warp.
// Design of record: docs/internals/entmax/solve_row.md.
//
// These are the arithmetic of `factor.cu` and `entmax.cu` with the OPERAND FETCH lifted
// out: a solve takes a LOGIT SOURCE (any object with `float operator()(int col) const`,
// evaluated by the lane owning `col`) instead of a global pointer and a stride tuple.
// That is what lets the fused producer kernel run the identical solve against an fp32
// tile it just computed in shared memory, while the standalone kernels keep running it
// against a materialized plane.
//
// NO CANDIDATE MASK: the producer plan never passes one, so `valid_count == width` here
// and the empty-row branch of `factor.cu` cannot be taken. The masked surface
// (`production_factor`) keeps its own kernel.
#pragma once

#include <cub/warp/warp_merge_sort.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>

#include "entmax.cuh"

namespace rola {
namespace entmax {
namespace detail {

//: The overflow-free read-write difference: two opposite-sign finite extrema overflow a
//: raw subtraction while their midpoint stays representable, so the MAGNITUDE is clamped
//: instead -- docs/internals/entmax/entmax.md#overflow-free-delta
__device__ __forceinline__ float row_overflow_free_delta(float r, float w) {
  const float kMax = 3.4028234663852886e38f;
  bool rneg = r < 0.0f;
  bool opposite = rneg != (w < 0.0f);
  float rmag = fabsf(r), wmag = fabsf(w);
  float hi = fmaxf(rmag, wmag), lo = fminf(rmag, wmag);
  float mag = hi + fminf(lo, kMax - hi);
  float opp = rneg ? -mag : mag;
  float same = (opposite ? 0.0f : r) - (opposite ? 0.0f : w);
  return opposite ? opp : same;
}

//: log(sigmoid(-delta)) == -softplus(delta), finite for every finite input.
__device__ __forceinline__ float row_log_write_gate(float delta) {
  return -(fmaxf(delta, 0.0f) + log1pf(expf(-fabsf(delta))));
}

//: The fused kernel's source: one row of the CTA's fp32 logit tile, in shared memory.
struct TileLogits {
  const float* __restrict__ row;
  __device__ __forceinline__ float operator()(int col) const { return row[col]; }
};

//: The standalone source: a strided global plane, addressed exactly as `factor.cu` does.
struct PlaneLogits {
  const float* __restrict__ base;
  Strides s;
  int64_t sb, sh, token, col_base;
  __device__ __forceinline__ float operator()(int col) const {
    return base[addr_split(s, sb, sh, token, col_base + col)];
  }
};

//: THE THRESHOLD-AFTER-SORT SOLVE for one row. Returns the support mask over this lane's
//: `IPT` columns and leaves the amplitudes in `p`; `tau` is the row threshold. The
//: packed-key reduction picks the greatest valid k and carries its tau in one pass --
//: docs/internals/entmax/entmax.md#forward-phases
template <int LW, int IPT, bool ALPHA15, class Src, class SortStore>
__device__ __forceinline__ uint32_t solve_entmax_row(const Src& src, int width, int lane,
                                                     unsigned lmask, SortStore& sort_store,
                                                     float (&p)[IPT]) {
  using WarpScanP = cub::WarpScan<Pair, LW>;
  using WarpRedF = cub::WarpReduce<float, LW>;
  using WarpRedU = cub::WarpReduce<unsigned long long, LW>;
  using WarpSort = cub::WarpMergeSort<float, IPT, LW>;
  static_assert(sizeof(typename WarpScanP::TempStorage) <= 4, "WarpScan must be shfl-only");
  static_assert(sizeof(typename WarpRedF::TempStorage) <= 4, "WarpReduce must be shfl-only");
  static_assert(sizeof(typename WarpRedU::TempStorage) <= 4, "WarpReduce must be shfl-only");
  typename WarpScanP::TempStorage scan_store;
  typename WarpRedF::TempStorage red_store;
  typename WarpRedU::TempStorage redu_store;

  float z[IPT];
  uint32_t valid_mask = 0u;
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    const int col = lane + i * LW;
    const bool valid = col < width;
    if (valid) valid_mask |= (1u << i);
    z[i] = valid ? src(col) : 0.0f;
  }

  float local_max = -INFINITY, local_min = INFINITY;
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    if ((valid_mask >> i) & 1u) {
      local_max = fmaxf(local_max, z[i]);
      local_min = fminf(local_min, z[i]);
    }
  }
  float row_max = WarpRedF(red_store).Reduce(local_max, cub::Max());
  row_max = __shfl_sync(lmask, row_max, 0, LW);
  float row_min = WarpRedF(red_store).Reduce(local_min, cub::Min());
  row_min = __shfl_sync(lmask, row_min, 0, LW);

  //: A finite row-relative sentinel keeps padded lanes strictly below every real value
  //: without an infinity in the scan. `row_min` is pre-shift, so shift it too.
  const float sentinel = (row_min - row_max) - 1.0f;
  float keys[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    const bool valid = (valid_mask >> i) & 1u;
    z[i] = valid ? z[i] - row_max : 0.0f;
    keys[i] = valid ? z[i] : sentinel;
  }
  WarpSort(sort_store).Sort(keys, Descending());

  Pair local_sum{0.0f, 0.0f};
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    local_sum.p += keys[i];
    local_sum.q += keys[i] * keys[i];
  }
  Pair carry;
  WarpScanP(scan_store).ExclusiveScan(local_sum, carry, Pair{0.0f, 0.0f}, PairAdd());

  int best_col = -1;
  float best_tau = 0.0f;
  float run_p = carry.p, run_q = carry.q;
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    run_p += keys[i];
    run_q += keys[i] * keys[i];
    const int rank = lane * IPT + i;
    const float k = (float)(rank + 1);
    float tau_c;
    bool cand;
    if (ALPHA15) {
      const float disc = run_p * run_p - k * (run_q - 4.0f);
      tau_c = (run_p - sqrtf(fmaxf(disc, 0.0f))) / k;
      cand = (rank < width) && (disc >= 0.0f) && (keys[i] > tau_c);
    } else {
      tau_c = (run_p - 1.0f) / k;
      cand = (rank < width) && (keys[i] > tau_c);
    }
    if (cand) {
      best_col = rank;
      best_tau = tau_c;
    }
  }
  unsigned long long packed_key = ((unsigned long long)(uint32_t)(best_col + 1) << 32)
                                  | (unsigned long long)__float_as_uint(best_tau);
  packed_key = WarpRedU(redu_store).Reduce(packed_key, cub::Max());
  packed_key = __shfl_sync(lmask, packed_key, 0, LW);
  const int chosen = (int)(uint32_t)(packed_key >> 32) - 1;
  const float tau = chosen >= 0 ? __uint_as_float((uint32_t)(packed_key & 0xffffffffu)) : 0.0f;

  uint32_t support = 0u;
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    const bool sup = ((valid_mask >> i) & 1u) && (z[i] > tau);
    if (sup) support |= (1u << i);
    float base = (ALPHA15 ? 0.5f : 1.0f) * (z[i] - tau);
    base = sup ? fmaxf(base, 0.0f) : 0.0f;
    p[i] = ALPHA15 ? base * base : base;
  }
  return support;
}

//: THE FULL-SUPPORT SOLVE for one row: shifted exponentials over one warp-sum.
template <int LW, int IPT, class Src>
__device__ __forceinline__ void solve_softmax_row(const Src& src, int width, int lane,
                                                  unsigned lmask, float (&p)[IPT]) {
  using WarpRedF = cub::WarpReduce<float, LW>;
  static_assert(sizeof(typename WarpRedF::TempStorage) <= 4, "WarpReduce must be shfl-only");
  typename WarpRedF::TempStorage red_store;

  float local_max = -INFINITY;
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    const int col = lane + i * LW;
    p[i] = (col < width) ? src(col) : -INFINITY;
    local_max = fmaxf(local_max, p[i]);
  }
  float row_max = WarpRedF(red_store).Reduce(local_max, cub::Max());
  row_max = __shfl_sync(lmask, row_max, 0, LW);

  float local_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    p[i] = (lane + i * LW < width) ? expf(p[i] - row_max) : 0.0f;
    local_sum += p[i];
  }
  float den = WarpRedF(red_store).Reduce(local_sum, cub::Sum());
  den = __shfl_sync(lmask, den, 0, LW);
#pragma unroll
  for (int i = 0; i < IPT; ++i) p[i] = p[i] / den;
}

//: THE UNION SPLIT for one row: ONE entmax solve on the midpoint determines a shared
//: support, then the sigmoid split sends `read` and the log-domain renormalization sends
//: `write`. `mid` is retained for the backward -- docs/internals/entmax/entmax.md
template <int LW, int IPT, bool ALPHA15, class SrcR, class SrcW, class SortStore>
__device__ __forceinline__ uint32_t solve_union_row(const SrcR& src_r, const SrcW& src_w, int width,
                                                    int lane, unsigned lmask,
                                                    SortStore& sort_store, float (&mid)[IPT],
                                                    float (&read)[IPT], float (&write)[IPT]) {
  using WarpRedF = cub::WarpReduce<float, LW>;
  typename WarpRedF::TempStorage red_store;

  float delta[IPT];
  struct Midpoint {
    float v[IPT];
    int lane_;
    __device__ __forceinline__ float operator()(int col) const { return v[(col - lane_) / LW]; }
  } mp;
  mp.lane_ = lane;
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    const int col = lane + i * LW;
    const bool valid = col < width;
    const float r = valid ? src_r(col) : 0.0f;
    const float w = valid ? src_w(col) : 0.0f;
    mp.v[i] = 0.5f * r + 0.5f * w;
    delta[i] = row_overflow_free_delta(r, w);
  }

  const uint32_t support = solve_entmax_row<LW, IPT, ALPHA15>(mp, width, lane, lmask, sort_store,
                                                              mid);

  float local_lw = -INFINITY;
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    const bool sup = (support >> i) & 1u;
    read[i] = mid[i] * (1.0f / (1.0f + expf(-delta[i])));
    delta[i] = sup ? logf(mid[i]) + row_log_write_gate(delta[i]) : -INFINITY;
    if (sup) local_lw = fmaxf(local_lw, delta[i]);
  }
  float max_lw = WarpRedF(red_store).Reduce(local_lw, cub::Max());
  max_lw = __shfl_sync(lmask, max_lw, 0, LW);
  const bool has_mass = max_lw > -INFINITY;
  const float safe_max = has_mass ? max_lw : 0.0f;

  float local_den = 0.0f;
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    const bool sup = (support >> i) & 1u;
    write[i] = (sup && has_mass) ? expf(delta[i] - safe_max) : 0.0f;
    local_den += write[i];
  }
  float den = WarpRedF(red_store).Reduce(local_den, cub::Sum());
  den = __shfl_sync(lmask, den, 0, LW);
  const bool has_weight = den > 0.0f;
  const float safe_den = has_weight ? den : 1.0f;
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    const bool sup = (support >> i) & 1u;
    write[i] = (sup && has_weight) ? write[i] / safe_den : 0.0f;
  }
  return support;
}

}  // namespace detail
}  // namespace entmax
}  // namespace rola
