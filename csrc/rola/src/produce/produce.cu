// THE PRODUCER'S TRANSLATION UNIT -- the kernel, composed of its components, and the host entry that validates the
// operands and launches it. SKELETON: each component has its real signature, tenants and edges and does the cheapest
// thing that keeps the composition's shape (zero digits, unit gains); the components land one at a time into it.
// see docs/internals/produce/produce.md

#include "common/torch_seam.cuh"

#include <cstring>

#include "common/arch_runtime.cuh"
#include "common/burst.cuh"
#include "common/ops.cuh"
#include "common/phase_clock.cuh"
#include "common/static_for.cuh"
#include "common/structure_switch.cuh"
#include "entmax/entmax.cuh"
#include "produce/produce.cuh"
#include "produce/produce_api.cuh"

namespace rola {
namespace produce {
namespace detail {

namespace ops = denseref::ops;
namespace row_solve = rola::entmax::detail;
using Clock = PhaseClock<kProducePhases>;

//: THE BLOCK'S JOB: its control from the parameters' shape byte (uniform, so a branch on it is a warp-uniform branch
//: ptxas can prove), its addressing from the table.
__device__ __forceinline__ Job load_job(const ProduceParams& p, int index) {
  const int shape = p.shapes[index];
  const int32_t* row = p.jobs + static_cast<int64_t>(index) * kJobFields;
  return Job{(shape >> 5) & 1, shape & 7,           (shape >> 3) & 3,    (shape >> 6) & 1,
             shape >> 7,       row[kFieldHead],     row[kFieldHeads],    row[kFieldWidth],
             row[kFieldIn0],   row[kFieldIn1],      row[kFieldOutRead],  row[kFieldOutWrite],
             row[kFieldLevel], row[kFieldFoldRead], row[kFieldFoldWrite]};
}

//: A job's projection columns: a route job's row (both rows of a union's), or a value job's heads' channels.
__device__ __forceinline__ int job_columns(const ProduceParams& p, const Job& job) {
  if (job.kind == kJobValue) return job.heads * p.d_v;
  return job.in1 >= 0 ? 2 * job.width : job.width;
}

__device__ __forceinline__ int route_column(const Job& job, int j) {
  return j < job.width ? job.in0 + j : job.in1 + (j - job.width);
}

//: THE WEIGHT ROW of job column `j`: its hidden run, contiguous.
__device__ __forceinline__ const __nv_bfloat16* weight_row(const ProduceParams& p, const Job& job,
                                                           int j) {
  if (job.kind == kJobValue) return p.v_W + static_cast<int64_t>(job.head * p.d_v + j) * p.v_row;
  return p.route_W + job.head * p.w_head + static_cast<int64_t>(route_column(job, j)) * p.w_col;
}

//: The job's bias row, or null: a value job has none, and a route job only when the routing declares one.
__device__ __forceinline__ const float* bias_row(const ProduceParams& p, const Job& job) {
  return job.kind == kJobRoute && p.route_bias != nullptr ? p.route_bias + job.head * p.C : nullptr;
}

//: Column `j`'s bias: a predicated load, never a branch around one.
__device__ __forceinline__ float column_bias(const float* bias, const Job& job, int j, bool live) {
  return bias != nullptr && live ? __ldg(bias + route_column(job, j)) : 0.0f;
}

//: THE WARPS' SHARE of a job's columns: `per_warp` MMA column tiles a warp, `groups` warps across the columns, and
//: `slices` warps across the hidden axis where the columns alone would leave warps idle. -- see docs/internals/produce/produce.md#tiling
struct Tiling {
  int per_warp, groups, slices;
};

__device__ __forceinline__ Tiling tiling(int columns) {
  const int tiles = (columns + 15) / 16;
  const int per_warp = tiles <= kProduceWarps ? 1 : tiles <= 2 * kProduceWarps ? 2 : 4;
  const int groups = (tiles + per_warp - 1) / per_warp;
  const int slices = groups == 1 ? 8 : groups == 2 ? 4 : groups <= 4 ? 2 : 1;
  return {per_warp, groups, slices};
}

//: One lane's 16-byte chunk of a hidden run, or zeros past the run's end (and no load).
__device__ __forceinline__ uint4 chunk(const __nv_bfloat16* run, int k, bool live) {
  return live ? __ldg(reinterpret_cast<const uint4*>(run + k)) : make_uint4(0u, 0u, 0u, 0u);
}

//: ONE K-BLOCK's operands: a lane's chunks of its tiles' column runs and of its token groups' rows.
template <int PerWarp, int Groups>
struct KBlock {
  uint4 a[PerWarp][2];
  uint4 b[Groups];
};

//: THE K-LOOP of one warp's share: `PerWarp` column tiles by `Groups` token groups over its slice's k-blocks, a
//: burst. Both operands' fragments are the lanes' 16-byte chunks: the MMA's k order is permuted to the chunk's, the
//: same permutation on both sides. -- see docs/internals/produce/produce.md#accumulate
//: @burst
template <int PerWarp, int Groups>
__device__ __forceinline__ void project(const ProduceParams& p, const Job& job, int token0,
                                        int columns, int tile0, int slice, int slices,
                                        float (&acc)[PerWarp][Groups][4]) {
  const int lane = static_cast<int>(threadIdx.x) & 31, g = lane >> 2, q = lane & 3;
  const __nv_bfloat16* w_run[PerWarp][2];
  bool w_live[PerWarp][2];
  rola::static_for<PerWarp>([&](auto I) {
    constexpr int i = decltype(I)::value;
    rola::static_for<2>([&](auto R) {
      constexpr int r = decltype(R)::value;
      const int j = (tile0 + i) * 16 + g + 8 * r;
      w_live[i][r] = j < columns;
      w_run[i][r] = (w_live[i][r] ? weight_row(p, job, j) : p.x) + 8 * q;
    });
  });
  const __nv_bfloat16* x_run[Groups];
  bool x_live[Groups];
  rola::static_for<Groups>([&](auto N) {
    constexpr int n = decltype(N)::value;
    const int token = token0 + n * 8 + g;
    x_live[n] = token < p.BT;
    x_run[n] = p.x + static_cast<int64_t>(x_live[n] ? token : 0) * p.hidden + 8 * q;
  });
  const int blocks = (p.hidden + kKBlock - 1) / kKBlock;
  const int count = slice < blocks ? (blocks - slice + slices - 1) / slices : 0;

  using Ops = KBlock<PerWarp, Groups>;
  auto gather = [&](int unit, Ops& ops) {
    const int k = (slice + unit * slices) * kKBlock;
    const bool k_live = k + 8 * q < p.hidden;
    rola::static_for<PerWarp>([&](auto I) {
      constexpr int i = decltype(I)::value;
      ops.a[i][0] = chunk(w_run[i][0], k, k_live && w_live[i][0]);
      ops.a[i][1] = chunk(w_run[i][1], k, k_live && w_live[i][1]);
    });
    rola::static_for<Groups>([&](auto N) {
      constexpr int n = decltype(N)::value;
      ops.b[n] = chunk(x_run[n], k, k_live && x_live[n]);
    });
  };
  auto issue = [&](const Ops& ops) {
    rola::static_for<Groups>([&](auto N) {
      constexpr int n = decltype(N)::value;
      const uint32_t b0[2] = {ops.b[n].x, ops.b[n].y};
      const uint32_t b1[2] = {ops.b[n].z, ops.b[n].w};
      rola::static_for<PerWarp>([&](auto I) {
        constexpr int i = decltype(I)::value;
        const uint4& lo = ops.a[i][0];
        const uint4& hi = ops.a[i][1];
        const uint32_t a0[4] = {lo.x, hi.x, lo.y, hi.y};
        const uint32_t a1[4] = {lo.z, hi.z, lo.w, hi.w};
        ops::mma(acc[i][n], a0, b0);
        ops::mma(acc[i][n], a1, b1);
      });
    });
  };
  rola::burst::Burst<Ops>::run(
      0, count, gather,
      [&](Ops& cur, Ops& next, int unit) {
        gather(unit, next);
        issue(cur);
      },
      [&](Ops& cur, Ops& /*unused*/) { issue(cur); });
}

//: One warp's share, projected and stored: the logits (+ bias), or a slice's partials. -- see docs/internals/produce/produce.md#accumulate
template <int PerWarp, int Groups>
__device__ __forceinline__ void project_share(const ProduceParams& p, const Job& job, int token0,
                                              int columns, const Tiling& t, float* logits) {
  const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  const int group = warp / t.slices, slice = warp % t.slices;
  if (group >= t.groups) return;
  float acc[PerWarp][Groups][4] = {};
  project<PerWarp, Groups>(p, job, token0, columns, group * PerWarp, slice, t.slices, acc);
  const int g = lane >> 2, q = lane & 3;
  const int base = t.slices == 1 ? 0 : kPartialBase + slice * t.groups * 16;
  const float* bias = t.slices == 1 ? bias_row(p, job) : nullptr;
  //: a lane's accumulators hold two columns a tile, each for two tokens a group: the column facts once, the stores
  //: predicated
  rola::static_for<PerWarp>([&](auto I) {
    constexpr int i = decltype(I)::value;
    rola::static_for<2>([&](auto H) {
      constexpr int h = decltype(H)::value;
      const int j = (group * PerWarp + i) * 16 + g + 8 * h;
      const bool live = j < columns;
      const float b = column_bias(bias, job, j, live);
      float* column = logits + base + j;
      rola::static_for<Groups>([&](auto N) {
        constexpr int n = decltype(N)::value;
        rola::static_for<2>([&](auto T) {
          constexpr int tt = decltype(T)::value;
          if (live) column[(n * 8 + 2 * q + tt) * kLogitStride] = acc[i][n][2 * h + tt] + b;
        });
      });
    });
  });
}

//: The slices' partials summed in slice order (+ bias): the same logits on every run.
__device__ void reduce_slices(const ProduceParams& p, const Job& job, int columns, const Tiling& t,
                              float* logits) {
#pragma unroll 1
  for (int i = static_cast<int>(threadIdx.x); i < kTokenTile * columns; i += kProduceThreads) {
    float* row = logits + (i / columns) * kLogitStride;
    const int j = i % columns;
    float sum = 0.0f;
#pragma unroll 1
    for (int s = 0; s < t.slices; ++s) sum += row[kPartialBase + s * t.groups * 16 + j];
    row[j] = sum + column_bias(bias_row(p, job), job, j, true);
  }
}

//: THE PROJECTION: the job's columns for the tile's tokens, fp32 (+ bias), into the logit rows. The warps' share and
//: the tile's live token groups (three round up to four) choose the burst once. -- see docs/internals/produce/produce.md#accumulate
__device__ void accumulate(const ProduceParams& p, const Job& job, int token0, float* logits) {
  const int columns = job_columns(p, job);
  const Tiling t = tiling(columns);
  const int live_groups = min(kTokenGroups, (p.BT - token0 + 7) / 8);
  const int share = t.per_warp == 1 ? 0 : t.per_warp == 2 ? 1 : 2;
  const int groups = live_groups == 1 ? 0 : live_groups == 2 ? 1 : 2;
  rola::uniform_switch<3>(share, [&](auto S) {
    rola::uniform_switch<3>(groups, [&](auto G) {
      project_share<1 << decltype(S)::value, 1 << decltype(G)::value>(p, job, token0, columns, t,
                                                                      logits);
    });
  });
  if (t.slices > 1) {
    __syncthreads();
    reduce_slices(p, job, columns, t, logits);
  }
}

//: The job's logits as computed, for its gate: written only when the probe is asked for.
__device__ void probe_logits(const ProduceParams& p, int job_index, int token0, int columns,
                             const float* logits) {
  float* out = p.logit_probe + static_cast<int64_t>(job_index) * p.BT * kMaxColumns;
#pragma unroll 1
  for (int i = static_cast<int>(threadIdx.x); i < kTokenTile * columns; i += kProduceThreads) {
    const int token = token0 + i / columns;
    if (token < p.BT) {
      out[static_cast<int64_t>(token) * kMaxColumns + i % columns] =
          logits[(i / columns) * kLogitStride + i % columns];
    }
  }
}

//: One digit of a row plane, in the plane's form: a u8 byte, or an f16 of the unit value.
__device__ __forceinline__ void write_digit(const ProduceParams& p, uint8_t* plane, int64_t index,
                                            float unit) {
  if (p.row_form == kRowsF16) {
    reinterpret_cast<__half*>(plane)[index] = __float2half_rn(unit);
  } else {
    plane[index] = static_cast<uint8_t>(__float2uint_rn(unit * 255.0f));
  }
}

//: A value the whole warp holds, reduced by a butterfly: every lane ends with the same bits.
template <class Op>
__device__ __forceinline__ float warp_all(float v, Op op) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    v = op(v, __shfl_xor_sync(0xffffffffu, v, offset));
  return v;
}

//: SOFTMAX over a row's members.
template <int IPT>
__device__ __forceinline__ void softmax_row(const float (&z)[IPT], uint32_t members,
                                            float (&p)[IPT]) {
  float m = -INFINITY;
#pragma unroll
  for (int i = 0; i < IPT; ++i) m = (members >> i) & 1u ? fmaxf(m, z[i]) : m;
  m = warp_all(m, [](float a, float b) { return fmaxf(a, b); });
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    p[i] = (members >> i) & 1u ? __expf(z[i] - m) : 0.0f;
    sum += p[i];
  }
  sum = warp_all(sum, [](float a, float b) { return a + b; });
#pragma unroll
  for (int i = 0; i < IPT; ++i) p[i] = __fdividef(p[i], sum);
}

//: THE ALPHA-ENTMAX THRESHOLD BY NEWTON FROM THE LEFT (alpha 1.5 or 2): `z` shifted by the row max in place.
//: `f(tau) = sum g(z - tau) - 1` is convex and decreasing, so Newton from `tau = -2` (alpha 2: `-1`), where `f >= 0`,
//: never passes the root; the closed form on the support `{z > lo}` is the root once no member lies below it (the
//: formula the sorted solve evaluates at its chosen rank). A vote ends the loop. -- see docs/internals/produce/produce.md#threshold
template <int IPT, bool ALPHA15>
__device__ __forceinline__ float newton_threshold(float (&z)[IPT], uint32_t members) {
  float m = -INFINITY;
#pragma unroll
  for (int i = 0; i < IPT; ++i) m = (members >> i) & 1u ? fmaxf(m, z[i]) : m;
  m = warp_all(m, [](float a, float b) { return fmaxf(a, b); });
#pragma unroll
  for (int i = 0; i < IPT; ++i) z[i] = (members >> i) & 1u ? z[i] - m : 0.0f;
  float lo = ALPHA15 ? -2.0f : -1.0f, tau = 0.0f;
#pragma unroll 1
  for (int step = 0; step < kNewtonSteps; ++step) {
    float k = 0.0f, P = 0.0f, Q = 0.0f;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const bool in = ((members >> i) & 1u) && z[i] > lo;
      k += in ? 1.0f : 0.0f;
      P += in ? z[i] : 0.0f;
      Q += in ? z[i] * z[i] : 0.0f;
    }
    k = warp_all(k, [](float a, float b) { return a + b; });
    P = warp_all(P, [](float a, float b) { return a + b; });
    Q = warp_all(Q, [](float a, float b) { return a + b; });
    tau = ALPHA15 ? __fdividef(P - sqrtf(fmaxf(P * P - k * (Q - 4.0f), 0.0f)), k)
                  : __fdividef(P - 1.0f, k);
    bool settled = true;
#pragma unroll
    for (int i = 0; i < IPT; ++i)
      settled = settled && !(((members >> i) & 1u) && z[i] > lo && z[i] < tau);
    if (__all_sync(0xffffffffu, settled)) break;
    lo =
        ALPHA15
            ? lo + __fdividef((Q - 2.0f * P * lo + k * lo * lo) * 0.25f - 1.0f, 0.5f * (P - k * lo))
            : tau;
  }
  return tau;
}

//: THE SOLVE of one token's row, element `lane + 32 i` a lane's `i`th: the job's levels, fp32 (a one-sided solve fills
//: both sides with its one level); a row past the live tokens solves zeros. Compact code on the special-function
//: unit: the fp32 levels feed u8 or f16 digits. -- see docs/internals/produce/produce.md#solve
template <int IPT>
__device__ __forceinline__ void solve_row(const Job& job, float* row, bool live, int lane,
                                          float (&read)[IPT], float (&write)[IPT], Clock& clock) {
  const int width = job.width;
  const bool is_union = job.solve >= kUnion15;
  float z[IPT], delta[IPT];
  uint32_t members = 0u;
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    const int col = lane + 32 * i;
    const bool member = col < width;
    members |= (member ? 1u : 0u) << i;
    const float r = member && live ? row[col] : 0.0f;
    const float w = member && live && is_union ? row[width + col] : 0.0f;
    z[i] = is_union ? 0.5f * r + 0.5f * w : r;
    delta[i] = is_union ? row_solve::overflow_free_delta(r, w, nullptr) : 0.0f;
  }
  __syncwarp();
  clock.lap(kPhaseRowLoad);

  if (job.solve == kSoftmax) {
    softmax_row<IPT>(z, members, read);
#pragma unroll
    for (int i = 0; i < IPT; ++i) write[i] = read[i];
    clock.lap(kPhaseLevels);
    return;
  }
  const bool alpha15 = job.solve == kEntmax15 || job.solve == kUnion15;
  const float tau =
      alpha15 ? newton_threshold<IPT, true>(z, members) : newton_threshold<IPT, false>(z, members);
  clock.lap(kPhaseThreshold);

  float values[IPT];
  uint32_t support = 0u;
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    const bool sup = ((members >> i) & 1u) && z[i] > tau;
    support |= (sup ? 1u : 0u) << i;
    float base = (alpha15 ? 0.5f : 1.0f) * (z[i] - tau);
    base = sup ? fmaxf(base, 0.0f) : 0.0f;
    values[i] = alpha15 ? base * base : base;
    read[i] = is_union ? __fdividef(values[i], 1.0f + __expf(-delta[i])) : values[i];
  }
  if (!is_union) {
#pragma unroll
    for (int i = 0; i < IPT; ++i) write[i] = values[i];
    clock.lap(kPhaseLevels);
    return;
  }

  row_solve::union_write<32, IPT, true>(values, delta, support, 0xffffffffu);
#pragma unroll
  for (int i = 0; i < IPT; ++i) write[i] = delta[i];
  clock.lap(kPhaseLevels);
}

//: ONE SIDE OF A ROW IN THE BUNDLE'S FORM: its digits against its max, the max as its scale, and the level's mass
//: for the finisher where the fold applies (1 elsewhere). The write side's fold is APPLIED -- its scale is over its
//: mass, the stored level on the simplex -- and the read side's is carried. -- see docs/internals/produce/produce.md#bundle
template <int IPT>
__device__ __forceinline__ void write_side(const ProduceParams& p, const float (&level)[IPT],
                                           int width, bool live, int lane, uint8_t* plane,
                                           int64_t first, float* scale, float* mass, int64_t at,
                                           bool fold, bool applied) {
  float m = 0.0f, sum = 0.0f;
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    const bool member = lane + 32 * i < width;
    m = fmaxf(m, member ? level[i] : 0.0f);
    sum += member ? level[i] : 0.0f;
  }
  m = warp_all(m, [](float a, float b) { return fmaxf(a, b); });
  sum = warp_all(sum, [](float a, float b) { return a + b; });
  const float unit = fmaxf(m, 1e-30f);
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    const int col = lane + 32 * i;
    if (live && col < width) write_digit(p, plane, first + col, __fdividef(level[i], unit));
  }
  if (live && lane == 0) {
    scale[at] = applied && fold && sum > 0.0f ? m / sum : m;
    mass[at] = fold ? sum : 1.0f;
  }
}

//: ONE TOKEN'S ROW, solved and written as the bundle holds it: each side the job writes, its digits, scale and
//: mass. A row past the live tokens reduces zeros and stores nothing. -- see docs/internals/produce/produce.md#route-row
template <int IPT>
__device__ void route_row(const ProduceParams& p, const Job& job, int job_index, int token,
                          bool live, float* row, int lane, Clock& clock) {
  float read[IPT], write[IPT];
  solve_row<IPT>(job, row, live, lane, read, write, clock);
  if (p.level_probe != nullptr) {
    float* out = p.level_probe + (static_cast<int64_t>(job_index) * p.BT + token) * kMaxColumns;
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
      const int col = lane + 32 * i;
      if (live && col < job.width) {
        out[col] = read[i];
        out[job.width + col] = write[i];
      }
    }
  }
  const int64_t first = (static_cast<int64_t>(token) * p.H + job.head) * p.rows;
  const int64_t at = (static_cast<int64_t>(token) * p.H + job.head) * p.D + job.level;
  if (job.reads) {
    write_side<IPT>(p, read, job.width, live, lane, p.amp_read, first + job.out_read, p.scale_read,
                    p.mass_read, at, job.fold_read != 0, false);
  }
  if (job.writes) {
    write_side<IPT>(p, write, job.width, live, lane, p.amp_write, first + job.out_write,
                    p.scale_write, p.mass_write, at, job.fold_write != 0, true);
  }
}

//: THE WRITE GAIN'S LOGIT of one token: the gain column's dot product with the token's row, a warp's lanes over the
//: hidden axis in 16-byte chunks, plus both biases.
__device__ __forceinline__ float gain_logit(const ProduceParams& p, const Job& job, int token,
                                            bool live, int lane) {
  const int column = p.C - p.G;
  const __nv_bfloat16* w = p.route_W + job.head * p.w_head + static_cast<int64_t>(column) * p.w_col;
  const __nv_bfloat16* xr = p.x + static_cast<int64_t>(live ? token : 0) * p.hidden;
  float dot = 0.0f;
#pragma unroll 1
  for (int k0 = 0; k0 < p.hidden; k0 += 256) {
    const int k = k0 + 8 * lane;
    const uint4 wc = chunk(w, k, live && k < p.hidden);
    const uint4 xc = chunk(xr, k, live && k < p.hidden);
    const uint32_t wv[4] = {wc.x, wc.y, wc.z, wc.w};
    const uint32_t xv[4] = {xc.x, xc.y, xc.z, xc.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      dot += ops::pair_lo(wv[j]) * ops::pair_lo(xv[j]) + ops::pair_hi(wv[j]) * ops::pair_hi(xv[j]);
    }
  }
  dot = warp_all(dot, [](float a, float b) { return a + b; });
  const float bias = p.route_bias != nullptr ? p.route_bias[job.head * p.C + column] : 0.0f;
  return dot + bias + p.gain_bias[job.head * p.G];
}

//: THE HEAD'S FINISH, by the last of its route jobs in the tile to arrive: `read_mass`, the product of the levels'
//: read masses, and `g_write`, the gain's softplus times the write masses -- the other jobs' masses read past the
//: fence, coherently. -- see docs/internals/produce/produce.md#finish
__device__ void finish_head(const ProduceParams& p, const Job& job, int tile, int token0) {
  __shared__ int s_last;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    s_last = atomicAdd(&p.arrivals[tile * p.H + job.head], 1) + 1 == p.route_jobs_per_head ? 1 : 0;
  }
  __syncthreads();
  if (!__any_sync(0xffffffffu, s_last != 0)) return;
  __threadfence();
  const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  const int live = min(kTokenTile, p.BT - token0);
#pragma unroll 1
  for (int pass = 0; pass < (live + kProduceWarps - 1) / kProduceWarps; ++pass) {
    const int t = pass * kProduceWarps + warp;
    const bool here = t < live;
    const float gain = p.G > 0 ? gain_logit(p, job, token0 + t, here, lane) : 0.0f;
    if (here && lane == 0) {
      const int64_t at = static_cast<int64_t>(token0 + t) * p.H + job.head;
      float read_mass = 1.0f, write_mass = 1.0f;
#pragma unroll 1
      for (int l = 0; l < p.D; ++l) {
        read_mass *= __ldcg(p.mass_read + at * p.D + l);
        write_mass *= __ldcg(p.mass_write + at * p.D + l);
      }
      const float softplus = gain > 20.0f ? gain : log1pf(expf(gain));
      p.read_mass[at] = read_mass;
      p.g_write[at] = (p.G > 0 ? softplus : 1.0f) * write_mass;
    }
  }
  if (threadIdx.x == 0) p.arrivals[tile * p.H + job.head] = 0;
}

//: THE VALUE DIGITS of the job's heads for the tile's tokens: a warp a (token, head), its lanes over the channels --
//: the channel max |v| as the scale, and each channel's digits against it at the digit count's full scale, s8 alone
//: or the s16's s8 high byte and u8 low byte. -- see docs/internals/produce/produce.md#value-rows
__device__ void value_rows(const ProduceParams& p, const Job& job, int token0,
                           const float* logits) {
  const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  const int pairs = min(kTokenTile, p.BT - token0) * job.heads;
  const float full = p.v_lo != nullptr ? 32767.0f : 127.0f;
#pragma unroll 1
  for (int pass = 0; pass < (pairs + kProduceWarps - 1) / kProduceWarps; ++pass) {
    const int pair = pass * kProduceWarps + warp;
    const bool here = pair < pairs;
    const int t = here ? pair / job.heads : 0;
    const int h = here ? pair - t * job.heads : 0;
    const float* row = logits + t * kLogitStride + h * p.d_v;
    float m = 0.0f;
#pragma unroll 1
    for (int c = lane; c < p.d_v; c += 32) m = fmaxf(m, fabsf(row[c]));
    m = warp_all(m, [](float a, float b) { return fmaxf(a, b); });
    const float unit = fmaxf(m, 1e-30f);
    const int64_t at = (static_cast<int64_t>(token0 + t) * p.H + job.head + h) * p.d_v;
#pragma unroll 1
    for (int c = lane; c < p.d_v; c += 32) {
      const int q = __float2int_rn(__fdividef(row[c], unit) * full);
      if (here) {
        p.v_hat[at + c] = static_cast<int8_t>(p.v_lo != nullptr ? q >> 8 : q);
        if (p.v_lo != nullptr) p.v_lo[at + c] = static_cast<uint8_t>(q & 0xFF);
      }
    }
    if (here && lane == 0) p.v_scale[(static_cast<int64_t>(token0 + t) * p.H + job.head + h)] = m;
  }
}

//: THE GRID: token tiles fastest, so the blocks resident together share one job's weights in L2.
__global__ __launch_bounds__(kProduceThreads, 1) void produce_kernel(ProduceParams p) {
  const int job_index = static_cast<int>(blockIdx.y), tile = static_cast<int>(blockIdx.x);
  const Job job = load_job(p, job_index);
  const int token0 = tile * kTokenTile;
  extern __shared__ float logits[];
  __shared__ long long s_clock[kProduceWarps][Clock::kRow];
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  Clock clock(&s_clock[warp][0], static_cast<int>(threadIdx.x) & 31);
  clock.lap(kPhaseJob);
  accumulate(p, job, token0, logits);
  clock.lap(kPhaseProject);
  __syncthreads();
  clock.lap(kPhaseBarrier);
  if (p.logit_probe != nullptr) probe_logits(p, job_index, token0, job_columns(p, job), logits);
  if (job.kind == kJobRoute) {
    //: every warp makes the same passes, a count of the tile's live tokens (uniform), so the solve's collectives sit
    //: under no branch ptxas cannot prove warp-uniform (KERNEL_STANDARDS §20)
    const int live = min(kTokenTile, p.BT - token0);
    const int passes = (live + kProduceWarps - 1) / kProduceWarps;
    rola::uniform_switch<4>(job.rows, [&](auto I) {
#pragma unroll 1
      for (int pass = 0; pass < passes; ++pass) {
        const int row = pass * kProduceWarps + warp;
        route_row<1 << decltype(I)::value>(p, job, job_index, token0 + row, row < live,
                                           logits + row * kLogitStride, threadIdx.x & 31, clock);
        clock.lap(kPhaseBundle);
      }
    });
    finish_head(p, job, tile, token0);
    clock.lap(kPhaseFinish);
  } else {
    value_rows(p, job, token0, logits);
    clock.lap(kPhaseValues);
  }
  clock.publish(p.ledger, job_index, kProduceWarps, warp);
}

void check_bf16(const Tensor& t, const char* name) {
  STD_TORCH_CHECK(t.is_cuda() && t.scalar_type() == Dtype::BFloat16, name,
                  " is a CUDA bf16 tensor, got ", t.scalar_type());
}

void check_f32(const Tensor& t, const char* name, int64_t numel) {
  STD_TORCH_CHECK(
      t.is_cuda() && t.scalar_type() == Dtype::Float && t.is_contiguous() && t.numel() == numel,
      name, " is a contiguous CUDA fp32 tensor of ", numel, " elements, got ", t.numel());
}

//: THE BOUND PHASE LEDGER, or null: the next launches book into it.
long long* g_ledger = nullptr;
int64_t g_ledger_jobs = 0;

}  // namespace detail

//: -- see docs/internals/produce/produce_api.md#rola-produce-ledger-bind
void rola_produce_ledger_bind(std::optional<Tensor> ledger) {
  if (!ledger.has_value()) {
    detail::g_ledger = nullptr;
    detail::g_ledger_jobs = 0;
    return;
  }
  STD_TORCH_CHECK(ledger->is_cuda() && ledger->scalar_type() == Dtype::Long && ledger->dim() == 3
                      && ledger->size(1) == kProduceWarps && ledger->size(2) == kProducePhases
                      && ledger->is_contiguous(),
                  "the producer's phase ledger is a contiguous CUDA int64 tensor [jobs][",
                  kProduceWarps, "][", kProducePhases, "]");
  detail::g_ledger = reinterpret_cast<long long*>(ledger->mutable_data_ptr<int64_t>());
  detail::g_ledger_jobs = ledger->size(0);
}

//: -- see docs/internals/produce/produce_api.md#rola-produce-forward
void rola_produce_forward(Tensor x, Tensor route_W, std::optional<Tensor> route_bias,
                          Tensor gain_bias, Tensor v_W, Tensor jobs, Tensor amp_read,
                          Tensor amp_write, Tensor scale_read, Tensor scale_write, Tensor mass_read,
                          Tensor mass_write, Tensor read_mass, Tensor g_write, Tensor v_hat,
                          std::optional<Tensor> v_lo, Tensor v_scale, Tensor arrivals,
                          std::optional<Tensor> logit_probe, std::optional<Tensor> level_probe,
                          Tensor shapes, int64_t levels, int64_t route_jobs_per_head) {
  check_arch_table();
  detail::check_bf16(x, "x");
  detail::check_bf16(route_W, "route_W");
  detail::check_bf16(v_W, "v_W");
  STD_TORCH_CHECK(x.dim() == 2 && x.is_contiguous(), "x is [BT, hidden], contiguous, got ",
                  shape(x.sizes()));
  STD_TORCH_CHECK(route_W.dim() == 3 && route_W.stride(1) == 1,
                  "route_W is [H, hidden, C] with its hidden axis contiguous, got ",
                  shape(route_W.sizes()));
  STD_TORCH_CHECK(v_W.dim() == 2 && v_W.stride(1) == 1,
                  "v_W is [H * d_v, hidden] with its hidden axis contiguous, got ",
                  shape(v_W.sizes()));
  const int64_t BT = x.size(0), hidden = x.size(1), H = route_W.size(0), C = route_W.size(2);
  //: every hidden run is read in 16-byte chunks: eight elements, aligned
  STD_TORCH_CHECK(hidden % 8 == 0 && route_W.stride(0) % 8 == 0 && route_W.stride(2) % 8 == 0
                      && v_W.stride(0) % 8 == 0,
                  "the hidden size and the weights' strides are multiples of 8, got hidden ",
                  hidden);
  for (const Tensor* t : {&x, &route_W, &v_W}) {
    STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(t->const_data_ptr()) % 16 == 0,
                    "x and the weights start 16-byte aligned");
  }
  STD_TORCH_CHECK(route_W.size(1) == hidden && v_W.size(1) == hidden, "route_W and v_W read x's ",
                  hidden, " hidden channels");
  STD_TORCH_CHECK(v_W.size(0) % H == 0, "v_W's ", v_W.size(0), " channels are not ", H,
                  " heads' worth");
  const int64_t d_v = v_W.size(0) / H;
  STD_TORCH_CHECK(jobs.is_cuda() && jobs.scalar_type() == Dtype::Int && jobs.is_contiguous()
                      && jobs.dim() == 2 && jobs.size(1) == kJobFields,
                  "jobs is the plan's [n, ", kJobFields, "] int32 job table on the device");
  STD_TORCH_CHECK(levels >= 1 && levels <= 4, "the routing depth is one to four levels, got ",
                  levels);
  const bool f16 = amp_read.scalar_type() == Dtype::Half;
  STD_TORCH_CHECK((f16 || amp_read.scalar_type() == Dtype::Byte)
                      && amp_write.scalar_type() == amp_read.scalar_type()
                      && amp_read.is_contiguous() && amp_write.is_contiguous()
                      && amp_read.dim() >= 2 && amp_read.sizes() == amp_write.sizes(),
                  "the rows are [..., H, rows] u8 or f16, contiguous, both sides alike");
  const int64_t rows = amp_read.size(amp_read.dim() - 1);
  STD_TORCH_CHECK(amp_read.numel() == BT * H * rows, "the rows hold ", amp_read.numel(),
                  " digits; ", BT, " tokens of ", H, " heads of ", rows, " need ", BT * H * rows);
  for (auto [t, name] : {std::pair{&scale_read, "scale_read"},
                         {&scale_write, "scale_write"},
                         {&mass_read, "mass_read"},
                         {&mass_write, "mass_write"}}) {
    detail::check_f32(*t, name, BT * H * levels);
  }
  detail::check_f32(read_mass, "read_mass", BT * H);
  detail::check_f32(g_write, "g_write", BT * H);
  detail::check_f32(v_scale, "v_scale", BT * H);
  STD_TORCH_CHECK(
      v_hat.scalar_type() == Dtype::Char && v_hat.is_contiguous() && v_hat.numel() == BT * H * d_v,
      "v_hat is [BT, H, d_v] int8");
  const int64_t tiles = (BT + kTokenTile - 1) / kTokenTile;
  STD_TORCH_CHECK(arrivals.scalar_type() == Dtype::Int && arrivals.is_contiguous()
                      && arrivals.numel() >= tiles * H,
                  "arrivals holds a counter a (token tile, head): ", tiles * H);
  STD_TORCH_CHECK(jobs.size(0) <= kMaxJobs, "a launch runs at most ", kMaxJobs, " jobs, got ",
                  jobs.size(0));
  STD_TORCH_CHECK(!shapes.is_cuda() && shapes.scalar_type() == Dtype::Byte && shapes.is_contiguous()
                      && shapes.numel() == jobs.size(0),
                  "shapes is the plan's host u8 shape byte a job");
  for (auto [t, name] : {std::pair{&logit_probe, "logit_probe"}, {&level_probe, "level_probe"}}) {
    if (t->has_value()) detail::check_f32(**t, name, jobs.size(0) * BT * kMaxColumns);
  }

  ProduceParams p{};
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr());
  p.route_W = reinterpret_cast<const __nv_bfloat16*>(route_W.const_data_ptr());
  p.w_head = route_W.stride(0);
  p.w_col = route_W.stride(2);
  p.route_bias = route_bias.has_value() ? route_bias->const_data_ptr<float>() : nullptr;
  p.gain_bias = gain_bias.const_data_ptr<float>();
  p.v_W = reinterpret_cast<const __nv_bfloat16*>(v_W.const_data_ptr());
  p.v_row = v_W.stride(0);
  p.jobs = jobs.const_data_ptr<int32_t>();
  p.amp_read = reinterpret_cast<uint8_t*>(amp_read.mutable_data_ptr());
  p.amp_write = reinterpret_cast<uint8_t*>(amp_write.mutable_data_ptr());
  p.scale_read = scale_read.mutable_data_ptr<float>();
  p.scale_write = scale_write.mutable_data_ptr<float>();
  p.mass_read = mass_read.mutable_data_ptr<float>();
  p.mass_write = mass_write.mutable_data_ptr<float>();
  p.read_mass = read_mass.mutable_data_ptr<float>();
  p.g_write = g_write.mutable_data_ptr<float>();
  p.v_hat = v_hat.mutable_data_ptr<int8_t>();
  p.v_lo = v_lo.has_value() ? v_lo->mutable_data_ptr<uint8_t>() : nullptr;
  p.v_scale = v_scale.mutable_data_ptr<float>();
  p.arrivals = arrivals.mutable_data_ptr<int32_t>();
  p.logit_probe = logit_probe.has_value() ? logit_probe->mutable_data_ptr<float>() : nullptr;
  p.level_probe = level_probe.has_value() ? level_probe->mutable_data_ptr<float>() : nullptr;
  std::memcpy(p.shapes, shapes.const_data_ptr(), static_cast<size_t>(shapes.numel()));
  STD_TORCH_CHECK(detail::g_ledger == nullptr || detail::g_ledger_jobs >= jobs.size(0),
                  "the bound phase ledger holds ", detail::g_ledger_jobs,
                  " jobs; this launch runs ", jobs.size(0));
  p.ledger = detail::g_ledger;
  p.BT = static_cast<int>(BT);
  p.hidden = static_cast<int>(hidden);
  p.H = static_cast<int>(H);
  p.C = static_cast<int>(C);
  p.G = static_cast<int>(gain_bias.numel() / H);
  p.rows = static_cast<int>(rows);
  p.D = static_cast<int>(levels);
  p.d_v = static_cast<int>(d_v);
  p.row_form = f16 ? kRowsF16 : kRowsU8;
  p.route_jobs_per_head = static_cast<int>(route_jobs_per_head);

  const DeviceGuard guard(x.get_device_index());
  //: ONCE: the attribute call is host time on the launch path
  static bool configured = false;
  if (!configured) {
    const int optin = current_device_properties().sharedMemPerBlockOptin;
    STD_TORCH_CHECK(kLogitBytes <= optin, "the producer's ", kLogitBytes,
                    "-byte logit rows exceed this device's ", optin, "-byte opt-in limit");
    ROLA_CUDA_CHECK(cudaFuncSetAttribute(detail::produce_kernel,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, kLogitBytes));
    configured = true;
  }
  detail::produce_kernel<<<dim3(static_cast<unsigned>(tiles), static_cast<unsigned>(jobs.size(0))),
                           kProduceThreads, kLogitBytes, current_stream()>>>(p);
  ROLA_CUDA_LAUNCH_CHECK();
}

}  // namespace produce
}  // namespace rola
