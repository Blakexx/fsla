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
#include "facts/liveness_contract.cuh"
#include "produce/produce.cuh"
#include "produce/produce_api.cuh"

namespace rola {
namespace produce {
namespace detail {

namespace ops = denseref::ops;
namespace row_solve = rola::entmax::detail;
using Clock = PhaseClock<kProducePhases>;

//: A (token, head)'s slot in the head-major outputs `[B, H, T]`: the flat token split into its batch and step.
__device__ __forceinline__ int64_t slot_of(const ProduceParams& p, int token, int head) {
  const int b = token / p.T;
  return (static_cast<int64_t>(b) * p.H + head) * p.T + (token - b * p.T);
}

//: THE BLOCK'S JOB: its control from its shape word (the parameters', read at the block's index: uniform to ptxas), its
//: addressing from the table. -- see docs/internals/produce/produce.md#jobs
__device__ __forceinline__ Job load_job(const ProduceParams& p, int index, int signed_shape) {
  const int shape = signed_shape & 0xffff;
  const int32_t* row = p.jobs + static_cast<int64_t>(index) * kJobFields;
  return Job{(shape >> 5) & 1,  shape & 7,         (shape >> 3) & 3,   (shape >> 6) & 1,
             (shape >> 7) & 1,  row[kFieldHead],   row[kFieldHeads],   row[kFieldWidth],
             row[kFieldIn0],    row[kFieldIn1],    row[kFieldOutRead], row[kFieldOutWrite],
             row[kFieldLevel],  (shape >> 8) & 1,  (shape >> 9) & 1,   (shape >> 10) & 7,
             (shape >> 13) & 1, (shape >> 14) & 1, (shape >> 15) & 1};
}

//: A job's projection columns: a route job's row (both rows of a union's), or a value job's heads' channels.
__device__ __forceinline__ int job_columns(const ProduceParams& p, const Job& job) {
  if (job.kind == kJobValue) return job.heads * p.d_v;
  return job.in1 >= 0 ? 2 * job.width : job.width;
}

//: A ONE-ROW JOB'S ACCUMULATOR COLUMN `a` AS ITS JOB COLUMN, or -1 (`columns` the job's, `job_columns`): a narrow
//: union's write row starts at its width class's tile offset (`32 << rows`), so a column's two logits share a lane.
//: It reads no parameter: a lambda holding the parameters' address would copy them to local memory.
//: -- see docs/internals/produce/produce.md#form
__device__ __forceinline__ int job_column(const Job& job, int columns, int a) {
  if (job.kind == kJobRoute && job.in1 >= 0) {
    const int offset = 32 << job.rows;
    return a < job.width                           ? a
           : a >= offset && a < offset + job.width ? job.width + (a - offset)
                                                   : -1;
  }
  return a < columns ? a : -1;
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

//: A STAGE ROW'S 16-BYTE CHUNK `c`: a row is `S` k-steps (32 S bytes), its chunk at `c` XOR the row's swizzle --
//: eight rows at one chunk in eight bank groups at every depth. -- see docs/internals/produce/produce.md#stage
template <int S>
__device__ __forceinline__ uint32_t chunk_at(int row, int c) {
  const int swizzle = S == 1 ? (row >> 2) & 1 : S == 2 ? (row >> 1) & 3 : row & 7;
  return static_cast<uint32_t>(row * 32 * S + ((c ^ swizzle) << 4));
}

//: A JOB'S FORM: its columns in one warp's accumulators (a one-row job, a narrow union's both rows, a value job's
//: heads), or a wide union's halves across a warp pair. -- see docs/internals/produce/produce.md#form
__device__ __forceinline__ bool pair_form(const Job& job) { return job.pair != 0; }

//: THE JOB'S WEIGHT BASE, uniform a block: its head's routing columns or its heads' value channels.
__device__ __forceinline__ const __nv_bfloat16* weight_base(const ProduceParams& p,
                                                            const Job& job) {
  return job.kind == kJobValue ? p.v_W + static_cast<int64_t>(job.head) * p.d_v * p.v_row
                               : p.route_W + job.head * p.w_head;
}

//: A THREAD'S COPY SOURCES, formed once a job as 32-bit element offsets (from the job's weight base and from x; the
//: host bounds both): its `QW` weight chunks' and `QX` x chunks' runs, the k offset added a stage, and one mask of
//: what issues. A column past the job's is never issued (only dead accumulators read it).
template <int QW, int QX>
struct Sources {
  uint32_t w[QW], x[QX];
  //: bit q < QW: weight chunk q issues; bit QW + q: x chunk q issues; bit QW + QX + q: its token is live
  uint32_t mask;
};

template <int S, int QW, int QX, class Column>
__device__ __forceinline__ Sources<QW, QX> sources(const ProduceParams& p, const Job& job,
                                                   Column column, int token0, int x_rows) {
  constexpr int kChunks = 2 * S;
  const int t = static_cast<int>(threadIdx.x);
  Sources<QW, QX> s;
  s.mask = 0u;
  rola::static_for<QW>([&](auto I) {
    constexpr int q = decltype(I)::value;
    const int u = t + q * kProduceThreads, col = column(u / kChunks);
    const bool issue = col >= 0;
    const int64_t run = !issue ? 0
                        : job.kind == kJobValue
                            ? static_cast<int64_t>(col) * p.v_row
                            : static_cast<int64_t>(route_column(job, col)) * p.w_col;
    s.w[q] = static_cast<uint32_t>(run) + 8u * static_cast<uint32_t>(u % kChunks);
    s.mask |= (issue ? 1u : 0u) << q;
  });
  rola::static_for<QX>([&](auto I) {
    constexpr int q = decltype(I)::value;
    const int u = t + q * kProduceThreads, row = u / kChunks, token = token0 + row;
    const bool issue = row < x_rows, live = issue && token < p.BT;
    s.x[q] = static_cast<uint32_t>(live ? token : 0) * static_cast<uint32_t>(p.hidden)
             + 8u * static_cast<uint32_t>(u % kChunks);
    s.mask |= (issue ? 1u : 0u) << (QW + q) | (live ? 1u : 0u) << (QW + QX + q);
  });
  return s;
}

//: ONE STAGE OF `S` K-STEPS: the job's columns' 16 S hidden and the stage's x rows', one copy group of every thread; a
//: chunk past the hidden size zero-fills. `issue` false copies nothing (the loop's last stage).
template <int S, int XOff, int QW, int QX>
__device__ __forceinline__ void stage_step(const ProduceParams& p, const __nv_bfloat16* base,
                                           const Sources<QW, QX>& s, int k, uint32_t stage,
                                           bool issue) {
  constexpr int kChunks = 2 * S;
  const int t = static_cast<int>(threadIdx.x);
  rola::static_for<QW>([&](auto I) {
    constexpr int q = decltype(I)::value;
    const int u = t + q * kProduceThreads, c = u % kChunks;
    const bool live = k + 8 * c < p.hidden;
    ops::stage_tile_when(stage + chunk_at<S>(u / kChunks, c),
                         base + (s.w[q] + (live ? static_cast<uint32_t>(k) : 0u)),
                         issue && ((s.mask >> q) & 1u), live);
  });
  rola::static_for<QX>([&](auto I) {
    constexpr int q = decltype(I)::value;
    const int u = t + q * kProduceThreads, c = u % kChunks;
    const bool live = k + 8 * c < p.hidden && ((s.mask >> (QW + QX + q)) & 1u);
    ops::stage_tile_when(stage + XOff + chunk_at<S>(u / kChunks, c),
                         p.x + (s.x[q] + (live ? static_cast<uint32_t>(k) : 0u)),
                         issue && ((s.mask >> (QW + q)) & 1u), live);
  });
  ops::stage_commit();
}

//: ONE STAGE'S MMAs of a warp, each of its `S` k-steps: B its 16 tokens' 16 hidden (`ldmatrix.x4`, both n-tiles' two
//: k halves), then each of its `M` column tiles' A (`ldmatrix.x4`) into both n-tiles' MMAs, the tile's stage rows from
//: `row0(i)`. -- see docs/internals/produce/produce.md#accumulate
//: @burst
template <int M, int S, int XOff, class Row0>
__device__ __forceinline__ void mma_stage(uint32_t stage, int token_row, int lane, Row0 row0,
                                          float (&acc)[kAccTiles][kTokenTiles][4]) {
  //: the stage's (k-step, column tile) pairs in order, each one's fragments loaded a pair ahead of its MMAs
  constexpr int kPairs = S * M;
  auto b_at = [&](int ks) {
    return stage + XOff
           + chunk_at<S>(token_row + (lane & 7) + 8 * (lane >> 4), 2 * ks + ((lane >> 3) & 1));
  };
  auto a_at = [&](int j) {
    return stage
           + chunk_at<S>(row0(j % M) + (lane & 7) + 8 * ((lane >> 3) & 1),
                         2 * (j / M) + (lane >> 4));
  };
  uint32_t a[2][4], b[2][4];
  ops::load_frag(b[0], b_at(0));
  ops::load_frag(a[0], a_at(0));
  rola::static_for<kPairs>([&](auto J) {
    constexpr int j = decltype(J)::value, ks = j / M, i = j % M;
    if constexpr (j + 1 < kPairs) {
      if constexpr ((j + 1) % M == 0) ops::load_frag(b[((j + 1) / M) & 1], b_at((j + 1) / M));
      ops::load_frag(a[(j + 1) & 1], a_at(j + 1));
    }
    ops::mma(acc[i][0], a[j & 1], b[ks & 1]);
    ops::mma(acc[i][1], a[j & 1], b[ks & 1] + 2);
  });
}

//: THE K-LOOP: two stages, one copy group and one CTA barrier a stage (KERNEL_STANDARDS 4's sm_80 form) -- the stage
//: landed and the other released at the barrier, its refill issued then, a stage ahead of its MMAs.
//: -- see docs/internals/produce/produce.md#accumulate
template <int M, int S, int XOff, int QW, int QX, class Row0>
__device__ __forceinline__ void k_loop(const ProduceParams& p, const __nv_bfloat16* base,
                                       const Sources<QW, QX>& s, uint32_t stages, int token_row,
                                       Row0 row0, float (&acc)[kAccTiles][kTokenTiles][4]) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int steps = (p.hidden + kKStep * S - 1) / (kKStep * S);
  stage_step<S, XOff>(p, base, s, 0, stages, true);
#pragma unroll 1
  for (int step = 0; step < steps; ++step) {
    ops::stage_wait<0>();
    __syncthreads();
    const uint32_t here = stages + static_cast<uint32_t>((step & 1) * kStageSlot);
    stage_step<S, XOff>(p, base, s, (step + 1) * kKStep * S,
                        stages + static_cast<uint32_t>(((step + 1) & 1) * kStageSlot),
                        step + 1 < steps);
    mma_stage<M, S, XOff>(here, token_row, lane, row0, acc);
  }
}

//: THE ACCUMULATORS SEEDED with the routing bias, a lane's columns `col(i, h)` of tile `i`'s rows `g` and `g + 8` (0
//: where none or dead): a predicated load a column, before the k-loop, where no accumulator is live yet.
template <class Col>
__device__ __forceinline__ void seed_bias(const ProduceParams& p, const Job& job, int columns,
                                          Col col, float (&acc)[kAccTiles][kTokenTiles][4]) {
  const float* bias = job.kind == kJobRoute ? bias_row(p, job) : nullptr;
  rola::static_for<kAccTiles>([&](auto I) {
    constexpr int i = decltype(I)::value;
    rola::static_for<2>([&](auto H) {
      constexpr int h = decltype(H)::value;
      const int j = col(i, h);
      const float b = column_bias(bias, job, j, j >= 0 && j < columns);
      rola::static_for<kTokenTiles>([&](auto N) {
        constexpr int n = decltype(N)::value;
        acc[i][n][2 * h] = b;
        acc[i][n][2 * h + 1] = b;
      });
    });
  });
}

//: THE JOB'S LOGITS from the accumulators, for its gate: a lane's tile `i` rows at columns `col(i, h)`, tokens
//: `token(n, t)` -- written only when the probe is asked for.
template <class Col, class Token>
__device__ __forceinline__ void probe_logits(const ProduceParams& p, int job_index, int columns,
                                             Col col, Token token,
                                             const float (&acc)[kAccTiles][kTokenTiles][4]) {
  if (p.logit_probe == nullptr) return;
  float* out = p.logit_probe + static_cast<int64_t>(job_index) * p.BT * kMaxColumns;
  rola::static_for<kAccTiles>([&](auto I) {
    constexpr int i = decltype(I)::value;
    rola::static_for<kTokenTiles * 4>([&](auto E) {
      constexpr int n = decltype(E)::value / 4, e = decltype(E)::value % 4;
      const int j = col(i, e >> 1), tk = token(n, e & 1);
      if (j >= 0 && j < columns && tk < p.BT)
        out[static_cast<int64_t>(tk) * kMaxColumns + j] = acc[i][n][e];
    });
  });
}

//: A value the whole warp holds, reduced by a butterfly: every lane ends with the same bits.
template <class Op>
__device__ __forceinline__ float warp_all(float v, Op op) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    v = op(v, __shfl_xor_sync(0xffffffffu, v, offset));
  return v;
}

//: A ROW'S VALUE REDUCED over the 8 lanes holding it (those sharing `lane & 3`): three shuffles, every lane the same
//: bits. -- see docs/internals/produce/produce.md#solve
template <class Op>
__device__ __forceinline__ float row_all(float v, Op op) {
#pragma unroll
  for (int offset = 4; offset < 32; offset <<= 1)
    v = op(v, __shfl_xor_sync(0xffffffffu, v, offset));
  return v;
}

//: A LANE'S TWO TOKEN ROWS of one n-tile: slot `S` its token `2 t + S`, value `V` the column tile `V >> 1`'s row half
//: `V & 1` -- an accumulator, named. -- see docs/internals/produce/produce.md#solve
template <int S, int V>
__device__ __forceinline__ float& row_value(float (&acc)[kAccTiles][4]) {
  return acc[V >> 1][2 * (V & 1) + S];
}

template <int S, int V>
__device__ __forceinline__ float row_value(const float (&acc)[kAccTiles][4]) {
  return acc[V >> 1][2 * (V & 1) + S];
}

//: A lane's member columns, bit `v` its value `v`'s column `8 v + g` under `width`: the same for all its rows.
constexpr int kSlots = 2;

__device__ __forceinline__ uint32_t row_members(int width, int lane) {
  const int n = min(max((width - (lane >> 2) + 7) / 8, 0), 2 * kAccTiles);
  return n == 32 ? 0xffffffffu : (1u << n) - 1u;
}

//: SOFTMAX on a lane's two rows of `M` column tiles, in place; `top` each row's largest level, its own quotient.
template <int M>
__device__ __forceinline__ void softmax_rows(float (&acc)[kAccTiles][4], uint32_t members,
                                             float (&top)[kSlots]) {
  rola::static_for<kSlots>([&](auto Sl) {
    constexpr int s = decltype(Sl)::value;
    float m = -INFINITY;
    rola::static_for<2 * M>([&](auto V) {
      constexpr int v = decltype(V)::value;
      m = (members >> v) & 1u ? fmaxf(m, row_value<s, v>(acc)) : m;
    });
    m = row_all(m, [](float a, float b) { return fmaxf(a, b); });
    float sum = 0.0f;
    rola::static_for<2 * M>([&](auto V) {
      constexpr int v = decltype(V)::value;
      const float e = (members >> v) & 1u ? __expf(row_value<s, v>(acc) - m) : 0.0f;
      row_value<s, v>(acc) = e;
      sum += e;
    });
    sum = row_all(sum, [](float a, float b) { return a + b; });
    rola::static_for<2 * M>([&](auto V) {
      constexpr int v = decltype(V)::value;
      row_value<s, v>(acc) = __fdividef(row_value<s, v>(acc), sum);
    });
    top[s] = __fdividef(1.0f, sum);
  });
}

//: THE ALPHA-ENTMAX LEVELS (alpha 1.5 where `a15`, else 2: uniform, the shape word's) on a lane's two rows, in place, by Newton from the left: each row
//: shifted by its max; `f(tau) = sum g(z - tau) - 1` convex and decreasing from `tau = -2` (alpha 2: `-1`), so a step
//: never passes the root and the closed form on the support `{z > lo}` is the root once no member lies below it. One
//: vote over the warp's 8 rows ends the loop (a settled row's further steps keep its threshold). `top` each row's
//: largest level. -- see docs/internals/produce/produce.md#threshold
template <int M>
__device__ __forceinline__ void entmax_rows(float (&acc)[kAccTiles][4], uint32_t members, bool a15,
                                            float (&top)[kSlots], uint32_t (&support)[kSlots],
                                            Clock& clock) {
  float lo[kSlots], tau[kSlots];
  rola::static_for<kSlots>([&](auto Sl) {
    constexpr int s = decltype(Sl)::value;
    float m = -INFINITY;
    rola::static_for<2 * M>([&](auto V) {
      constexpr int v = decltype(V)::value;
      m = (members >> v) & 1u ? fmaxf(m, row_value<s, v>(acc)) : m;
    });
    m = row_all(m, [](float a, float b) { return fmaxf(a, b); });
    rola::static_for<2 * M>([&](auto V) {
      constexpr int v = decltype(V)::value;
      row_value<s, v>(acc) = (members >> v) & 1u ? row_value<s, v>(acc) - m : 0.0f;
    });
    lo[s] = a15 ? -2.0f : -1.0f;
    tau[s] = 0.0f;
  });

#pragma unroll 1
  for (int step = 0; step < kNewtonSteps; ++step) {
    //: settled: no member in (lo, tau) -- where tau > lo, the members at or past tau are all the support counted
    bool settled = true;
    rola::static_for<kSlots>([&](auto Sl) {
      constexpr int s = decltype(Sl)::value;
      float k = 0.0f, P = 0.0f, Q = 0.0f;
      rola::static_for<2 * M>([&](auto V) {
        constexpr int v = decltype(V)::value;
        const float z = row_value<s, v>(acc);
        //: the square through the step's own member value, so no product of the step-invariant `z` is hoisted
        const bool in = ((members >> v) & 1u) && z > lo[s];
        const float zi = in ? z : 0.0f;
        k += in ? 1.0f : 0.0f;
        P += zi;
        Q = fmaf(zi, z, Q);
      });
      const float here = k;
      k = row_all(k, [](float a, float b) { return a + b; });
      P = row_all(P, [](float a, float b) { return a + b; });
      Q = row_all(Q, [](float a, float b) { return a + b; });
      tau[s] = a15 ? __fdividef(P - ops::sqrt_approx(fmaxf(P * P - k * (Q - 4.0f), 0.0f)), k)
                   : __fdividef(P - 1.0f, k);
      float past = 0.0f;
      rola::static_for<2 * M>([&](auto V) {
        constexpr int v = decltype(V)::value;
        past += ((members >> v) & 1u) && row_value<s, v>(acc) >= tau[s] ? 1.0f : 0.0f;
      });
      settled = settled && (tau[s] <= lo[s] || past == here);
      lo[s] = a15 ? lo[s]
                        + __fdividef((Q - 2.0f * P * lo[s] + k * lo[s] * lo[s]) * 0.25f - 1.0f,
                                     0.5f * (P - k * lo[s]))
                  : tau[s];
    });
    if (__all_sync(0xffffffffu, settled)) break;
  }
  clock.lap(kPhaseThreshold);

  rola::static_for<kSlots>([&](auto Sl) {
    constexpr int s = decltype(Sl)::value;
    support[s] = 0u;
    rola::static_for<2 * M>([&](auto V) {
      constexpr int v = decltype(V)::value;
      const float z = row_value<s, v>(acc);
      const bool sup = ((members >> v) & 1u) && z > tau[s];
      support[s] |= (sup ? 1u : 0u) << v;
      const float base = sup ? fmaxf((a15 ? 0.5f : 1.0f) * (z - tau[s]), 0.0f) : 0.0f;
      row_value<s, v>(acc) = a15 ? base * base : base;
    });
    const float peak = fmaxf((a15 ? 0.5f : 1.0f) * (0.0f - tau[s]), 0.0f);
    top[s] = a15 ? peak * peak : peak;
  });
}

//: THE JOB'S SOLVED LEVELS from the accumulators, for its gate: the read side at `[0, w)` from values `v`, the write
//: side at `[w, 2w)` from values `v + W` (a one-activation row's `W` 0, its one level both sides') -- written only
//: when the probe is asked for.
template <int W, class Token>
__device__ __forceinline__ void probe_levels(const ProduceParams& p, int job_index, int width,
                                             int lane, Token token,
                                             const float (&acc)[kAccTiles][4]) {
  if (p.level_probe == nullptr) return;
  float* out = p.level_probe + static_cast<int64_t>(job_index) * p.BT * kMaxColumns;
  rola::static_for<kSlots>([&](auto Sl) {
    constexpr int s = decltype(Sl)::value;
    const int tk = token(s);
    rola::static_for<2 * kAccTiles - W>([&](auto V) {
      constexpr int v = decltype(V)::value;
      const int j = 8 * v + (lane >> 2);
      if (j < width && tk < p.BT) {
        out[static_cast<int64_t>(tk) * kMaxColumns + j] = row_value<s, v>(acc);
        out[static_cast<int64_t>(tk) * kMaxColumns + width + j] = row_value<s, v + W>(acc);
      }
    });
  });
}

//: A COEFFICIENT IN ITS STORED FORM against the row's max, by its reciprocal `inv` (one a row): a u8 digit at 255, or a
//: dense level's f16 at 1 with `floor` taken (0 for a dead row) -- within an fp32 unit of the quotient, never the
//: IEEE divide's slow-path call (a call a coefficient, cold, whose convention costs the body its registers).
__device__ __forceinline__ void store_coef(uint8_t* at, float level, float inv, float) {
  *at = static_cast<uint8_t>(__float2uint_rn(level * inv * 255.0f));
}

__device__ __forceinline__ void store_coef(__half* at, float level, float inv, float floor) {
  *at = __float2half_rn(fmaxf(level * inv, floor));
}

__device__ __forceinline__ void store_zero(uint8_t* at) { *at = uint8_t{0}; }

__device__ __forceinline__ void store_zero(__half* at) { *at = __float2half_rn(0.0f); }

//: A WARP'S STAGED ROWS: one token slot's 4 rows of one side (a lane quad's tokens), a row the widest slot plus 16
//: bytes (four rows' bytes in distinct banks), past the parked n-tiles in the stages the k-loop released.
//: -- see docs/internals/produce/produce.md#bundle
constexpr int kStagedRowBytes = kMaxWidth * 2 + 16;
constexpr int kStagedWarpBytes = 4 * kStagedRowBytes;
//: A WARP'S PARKED N-TILE: its second n-tile's accumulators, 16 bytes a lane a column tile, lane-major.
constexpr int kParkedWarpBytes = kAccTiles * 32 * 16;
static_assert(kProduceWarps * (kParkedWarpBytes + kStagedWarpBytes) <= kProduceSmem,
              "the parked n-tiles and the staged rows fit the stages");

//: ONE SIDE OF A WARP'S ROWS IN THE BUNDLE'S FORM, one token slot at a time (`Out`: u8, or f16 for a dense level, its
//: floor taken in a live row): each row's coefficients against its max `top` and its slot's tail zeros staged, the max
//: as the side's scale (over the mass where the write fold applies) and the level's mass for the finisher, then the
//: rows' slots stored 16 bytes a lane. `tile0` the n-tile's first token. -- see docs/internals/produce/produce.md#bundle
template <int M, class Out>
__device__ __forceinline__ void bundle_side(const ProduceParams& p, const Job& job, int tile0,
                                            int lane, const float (&acc)[kAccTiles][4],
                                            const float (&top)[kSlots], uint32_t members,
                                            Out* plane, int plane_row, int out_col, float* scale,
                                            float* mass, bool fold, bool applied, bool floored,
                                            uint8_t* staged) {
  constexpr int kGrain = 32 / static_cast<int>(sizeof(Out));
  constexpr int kRowElems = kStagedRowBytes / static_cast<int>(sizeof(Out));
  const int width = job.width, slot = (width + kGrain - 1) / kGrain * kGrain, g = lane >> 2;
  const int chunks = slot * static_cast<int>(sizeof(Out)) / 16;
  Out* rows = reinterpret_cast<Out*>(staged) + (lane & 3) * kRowElems;
  rola::static_for<kSlots>([&](auto Sl) {
    constexpr int s = decltype(Sl)::value;
    float sum = 1.0f;
    if (fold) {
      sum = 0.0f;
      rola::static_for<2 * M>([&](auto V) {
        constexpr int v = decltype(V)::value;
        sum += (members >> v) & 1u ? row_value<s, v>(acc) : 0.0f;
      });
      sum = row_all(sum, [](float a, float b) { return a + b; });
    }
    const float m = top[s], inv = __fdividef(1.0f, fmaxf(m, 1e-30f));
    const float floor = floored && m > 0.0f ? facts::kDenseFloor : 0.0f;
    rola::static_for<2 * M>([&](auto V) {
      constexpr int v = decltype(V)::value;
      const int c = 8 * v + g;
      if (c < width)
        store_coef(rows + c, row_value<s, v>(acc), inv, floor);
      else if (c < slot)
        store_zero(rows + c);
    });
    const int token = tile0 + 2 * (lane & 3) + s;
    if (g == 0 && token < p.BT) {
      const int64_t at = slot_of(p, token, job.head) * p.D + job.level;
      scale[at] = applied && fold && sum > 0.0f ? __fdividef(m, sum) : m;
      mass[at] = sum;
    }
    __syncwarp();
#pragma unroll 1
    for (int i = lane; i < 4 * chunks; i += 32) {
      const int r = i / chunks, chunk = i - r * chunks, tk = tile0 + 2 * r + s;
      if (tk < p.BT) {
        uint8_t* row = reinterpret_cast<uint8_t*>(
            plane + slot_of(p, tk, job.head) * static_cast<int64_t>(plane_row) + out_col);
        *reinterpret_cast<uint4*>(row + 16 * chunk) =
            *reinterpret_cast<const uint4*>(staged + r * kStagedRowBytes + 16 * chunk);
      }
    }
    //: the next slot's staging overwrites these rows
    __syncwarp();
  });
}

//: A ONE-ROW JOB'S ROWS IN THE BUNDLE: each side it writes, its plane the side's dense f16 or sparse u8 (the shape
//: word's bits). The read side's fold is carried (its scale the max); the write side's applied.
template <int M>
__device__ __forceinline__ void bundle_rows(const ProduceParams& p, const Job& job, int tile0,
                                            int lane, const float (&acc)[kAccTiles][4],
                                            const float (&top)[kSlots], uint32_t members,
                                            uint8_t* staged) {
  if (job.reads) {
    if (job.dense_read)
      bundle_side<M>(p, job, tile0, lane, acc, top, members, p.dense[0], p.dense_rows[0],
                     job.out_read, p.scale_read, p.mass_read, job.fold_read != 0, false, true,
                     staged);
    else
      bundle_side<M>(p, job, tile0, lane, acc, top, members, p.amp_read, p.rows_read, job.out_read,
                     p.scale_read, p.mass_read, job.fold_read != 0, false, false, staged);
  }
  if (job.writes) {
    if (job.dense_write)
      bundle_side<M>(p, job, tile0, lane, acc, top, members, p.dense[1], p.dense_rows[1],
                     job.out_write, p.scale_write, p.mass_write, job.fold_write != 0, true, true,
                     staged);
    else
      bundle_side<M>(p, job, tile0, lane, acc, top, members, p.amp_write, p.rows_write,
                     job.out_write, p.scale_write, p.mass_write, job.fold_write != 0, true, false,
                     staged);
  }
}

//: THE SOLVE of one n-tile of a one-row route job's rows on the accumulators, and their bundle, in its row class of
//: `M` column tiles: its activation, the level probe, the bundle. -- see docs/internals/produce/produce.md#solve
template <int M>
__device__ __forceinline__ void solve_class(const ProduceParams& p, const Job& job, int job_index,
                                            int tile0, int lane, float (&acc)[kAccTiles][4],
                                            uint32_t members, uint8_t* staged, Clock& clock) {
  float top[kSlots];
  uint32_t support[kSlots];
  if (job.solve == kSoftmax)
    softmax_rows<M>(acc, members, top);
  else
    entmax_rows<M>(acc, members, job.solve == kEntmax15, top, support, clock);
  clock.lap(kPhaseLevels);
  probe_levels<0>(
      p, job_index, job.width, lane, [&](int s) { return tile0 + 2 * (lane & 3) + s; }, acc);
  bundle_rows<M>(p, job, tile0, lane, acc, top, members, staged);
  clock.lap(kPhaseBundle);
}

//: THE SOLVE of one n-tile, its row class the shape word's (warp-uniform to ptxas). The class's body is a
//: force-inlined function, the switch's lambda one call to it: a lambda's own body past the inliner's threshold is
//: outlined, and an outlined body proves no shuffle converged. -- see docs/internals/produce/produce.md#solve
__device__ __forceinline__ void solve_tile(const ProduceParams& p, const Job& job, int job_index,
                                           int tile0, int lane, float (&acc)[kAccTiles][4],
                                           uint8_t* staged, Clock& clock) {
  const uint32_t members = row_members(job.width, lane);
  rola::uniform_switch<4>(job.rows, [&](auto R) __attribute__((always_inline)) {
    solve_class<2 << decltype(R)::value>(p, job, job_index, tile0, lane, acc, members, staged,
                                         clock);
  });
}

//: ONE N-TILE OF A VALUE JOB'S ROWS, its heads `C = d_v / 8` of a lane's values each: per (token, head) the channel max
//: |v| as the scale and each channel against it rounded once to f16. -- see docs/internals/produce/produce.md#value-rows
template <int M, int C>
__device__ __forceinline__ void value_class(const ProduceParams& p, const Job& job, int tile0,
                                            int lane, const float (&acc)[kAccTiles][4]) {
  const int g = lane >> 2;
  rola::static_for<kSlots>([&](auto Sl) {
    constexpr int s = decltype(Sl)::value;
    const int token = tile0 + 2 * (lane & 3) + s;
    rola::static_for<2 * M / C>([&](auto Hh) {
      constexpr int h = decltype(Hh)::value;
      float m = 0.0f;
      rola::static_for<C>(
          [&](auto K) { m = fmaxf(m, fabsf(row_value<s, h * C + decltype(K)::value>(acc))); });
      m = row_all(m, [](float a, float b) { return fmaxf(a, b); });
      const float inv = __fdividef(1.0f, fmaxf(m, 1e-30f));
      if (h < job.heads && token < p.BT) {
        const int64_t slot = slot_of(p, token, job.head + h);
        rola::static_for<C>([&](auto K) {
          constexpr int k = decltype(K)::value;
          p.v[slot * p.d_v + 8 * k + g] = __float2half_rn(row_value<s, h * C + k>(acc) * inv);
        });
        if (g == 0) p.v_scale[slot] = m;
      }
    });
  });
}

//: ONE N-TILE OF A VALUE JOB'S ROWS, its column class and head size the shape word's (warp-uniform to ptxas).
__device__ __forceinline__ void value_tile(const ProduceParams& p, const Job& job, int tile0,
                                           int lane, const float (&acc)[kAccTiles][4]) {
  rola::uniform_switch<3>(job.tiles - 2, [&](auto F) __attribute__((always_inline)) {
    constexpr int M = 4 << decltype(F)::value;
    rola::uniform_switch<4>(job.rows, [&](auto R) __attribute__((always_inline)) {
      constexpr int C = 4 << decltype(R)::value;
      if constexpr (2 * M >= C) value_class<M, C>(p, job, tile0, lane, acc);
    });
  });
}

//: One lane's 16-byte chunk of a hidden run, or zeros past the run's end (and no load).
__device__ __forceinline__ uint4 chunk(const __nv_bfloat16* run, int k, bool live) {
  return live ? __ldg(reinterpret_cast<const uint4*>(run + k)) : make_uint4(0u, 0u, 0u, 0u);
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
      const int64_t at = slot_of(p, token0 + t, job.head);
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

//: THE SKELETON'S ROWS (R3 lands the solve, the split and the bundle on the accumulators): each side the job writes,
//: its row slot zero, its scale zero and its mass one, for the tile's live tokens.
template <class Out>
__device__ __forceinline__ void zero_side(Out* plane, int64_t first, int width, int lane) {
  constexpr int kGrain = 32 / static_cast<int>(sizeof(Out));
  const int slot = (width + kGrain - 1) / kGrain * kGrain;
#pragma unroll 1
  for (int col = lane; col < slot; col += 32) store_zero(plane + first + col);
}

__device__ void stub_rows(const ProduceParams& p, const Job& job, int token0, int live, int warp,
                          int lane) {
#pragma unroll 1
  for (int r = warp; r < live; r += kProduceWarps) {
    const int64_t slot = slot_of(p, token0 + r, job.head);
    const int64_t at = slot * p.D + job.level;
    if (job.reads) {
      if ((p.dense_mask[0] >> job.level) & 1)
        zero_side(p.dense[0], slot * p.dense_rows[0] + job.out_read, job.width, lane);
      else
        zero_side(p.amp_read, slot * p.rows_read + job.out_read, job.width, lane);
      if (lane == 0) p.scale_read[at] = 0.0f, p.mass_read[at] = 1.0f;
    }
    if (job.writes) {
      if ((p.dense_mask[1] >> job.level) & 1)
        zero_side(p.dense[1], slot * p.dense_rows[1] + job.out_write, job.width, lane);
      else
        zero_side(p.amp_write, slot * p.rows_write + job.out_write, job.width, lane);
      if (lane == 0) p.scale_write[at] = 0.0f, p.mass_write[at] = 1.0f;
    }
  }
}

//: A ONE-ROW JOB'S PROJECTION: every warp the job's columns for its 16 tokens, in the job's class of `M` column tiles
//: and `S` k-steps a stage (the stage's bytes held near level). -- see docs/internals/produce/produce.md#form
__device__ __forceinline__ void project_one_row(const ProduceParams& p, int job_index, int shape,
                                                int token0, uint32_t stages, int warp, int lane,
                                                float (&acc)[kAccTiles][kTokenTiles][4]) {
  const int g = lane >> 2;
  {
    const Job job = load_job(p, job_index, shape);
    const int columns = job_columns(p, job);
    seed_bias(
        p, job, columns, [&](int i, int h) { return job_column(job, columns, 16 * i + g + 8 * h); },
        acc);
    const __nv_bfloat16* base = weight_base(p, job);
    rola::uniform_switch<3>(job.tiles - 2, [&](auto F) {
      //: the class's stage: `16 / M` k-steps, so its weights are 8 KB at every class and its x rows follow them
      constexpr int M = 4 << decltype(F)::value, S = kAccTiles / M, kXOff = 16 * M * 32 * S;
      constexpr int QW = 16 * M * 2 * S / kProduceThreads,
                    QX = kTokenTile * 2 * S / kProduceThreads;
      const Sources<QW, QX> s = sources<S, QW, QX>(
          p, job, [&](int a) { return job_column(job, columns, a); }, token0, kTokenTile);
      k_loop<M, S, kXOff>(
          p, base, s, stages, kWarpTokens * warp, [](int i) { return 16 * i; }, acc);
    });
  }
  //: the job's fields read again past the k-loop, never held across it
  const Job job = load_job(p, job_index, shape);
  const int columns = job_columns(p, job);
  auto token = [&](int n, int e) {
    return token0 + kWarpTokens * warp + 8 * n + 2 * (lane & 3) + e;
  };
  probe_logits(
      p, job_index, columns,
      [&](int i, int h) { return job_column(job, columns, 16 * i + g + 8 * h); }, token, acc);
}

//: A WIDE UNION'S PROJECTION, one 32-token pass: a warp pair a token group, warp `2j + h` columns `128 h ..` of
//: both rows, eight tiles a row, one k-step a stage (its 512 columns fill it). -- see docs/internals/produce/produce.md#form
__device__ __forceinline__ void project_pair(const ProduceParams& p, int job_index, int shape,
                                             int token0, uint32_t stages, int warp, int lane,
                                             float (&acc)[kAccTiles][kTokenTiles][4]) {
  const int pair = warp >> 1, half = warp & 1, g = lane >> 2;
  auto col = [&](int i, int h, int width) {
    const int c = 128 * half + 16 * (i & 7) + g + 8 * h;
    return c < width ? (i < 8 ? c : width + c) : -1;
  };
  {
    const Job job = load_job(p, job_index, shape);
    const int columns = job_columns(p, job), width = job.width;
    seed_bias(p, job, columns, [&](int i, int h) { return col(i, h, width); }, acc);
    const __nv_bfloat16* base = weight_base(p, job);
    constexpr int QW = kMaxColumns * 2 / kProduceThreads, QX = 1;
    const Sources<QW, QX> s = sources<1, QW, QX>(
        p, job, [&](int c) { return c < columns ? c : -1; }, token0, kTokenTile / 2);
    k_loop<kAccTiles, 1, kMaxColumns * 32>(
        p, base, s, stages, kWarpTokens * pair,
        [&](int i) { return (i < 8 ? 0 : width) + 128 * half + 16 * (i & 7); }, acc);
  }
  //: the job's fields read again past the k-loop, never held across it
  const Job job = load_job(p, job_index, shape);
  const int columns = job_columns(p, job), width = job.width;
  auto token = [&](int n, int e) {
    return token0 + kWarpTokens * pair + 8 * n + 2 * (lane & 3) + e;
  };
  probe_logits(p, job_index, columns, [&](int i, int h) { return col(i, h, width); }, token, acc);
}

//: THE GRID: token tiles fastest, so the blocks resident together share one job's weights in L2; two blocks an SM, so
//: one's solve runs beside the other's MMAs. The parameters are `__grid_constant__`: a body's lambdas hold their
//: address, which otherwise copies them whole to local memory. -- see docs/internals/produce/produce.md#grid
__global__ __launch_bounds__(kProduceThreads, 2) void produce_kernel(
    const __grid_constant__ ProduceParams p) {
  const int job_index = static_cast<int>(blockIdx.y), tile = static_cast<int>(blockIdx.x);
  const int shape = p.shapes[job_index];
  const int token0 = tile * kTokenTile;
  extern __shared__ __align__(128) uint8_t stage_bytes[];
  const uint32_t stages = ops::smem_addr(stage_bytes);
  __shared__ long long s_clock[kProduceWarps][Clock::kRow];
  //: the warp's index as ptxas knows it uniform (`uniform_warp`): a uniform value held across the k-loop at the register
  //: cap can live outside the vector registers
  const int warp = ops::uniform_warp<kProduceWarps>(), lane = static_cast<int>(threadIdx.x) & 31;
  Clock clock(&s_clock[warp][0], lane, p.ledger != nullptr);
  clock.lap(kPhaseJob);
  const int live = min(kTokenTile, p.BT - token0);

  if (pair_form(load_job(p, job_index, shape))) {
    //: a pass's 32 tokens a k-loop; the barrier between passes releases the stages the last step read
    const int passes = (live + kTokenTile / 2 - 1) / (kTokenTile / 2);
#pragma unroll 1
    for (int pass = 0; pass < passes; ++pass) {
      float acc[kAccTiles][kTokenTiles][4];
      project_pair(p, job_index, shape, token0 + pass * (kTokenTile / 2), stages, warp, lane, acc);
      __syncthreads();
    }
    clock.lap(kPhaseProject);
  } else {
    float acc[kAccTiles][kTokenTiles][4];
    project_one_row(p, job_index, shape, token0, stages, warp, lane, acc);
    //: every warp past its last stage before any parks or stages there
    __syncthreads();
    clock.lap(kPhaseProject);
    const Job job = load_job(p, job_index, shape);

    if (job.kind == kJobValue || job.solve < kUnion15) {
      //: the second n-tile parked, so the solve or the values hold one n-tile's 64 accumulators
      uint8_t* parked = stage_bytes + warp * kParkedWarpBytes;
      uint8_t* staged = stage_bytes + kProduceWarps * kParkedWarpBytes + warp * kStagedWarpBytes;
      float tile[kAccTiles][4];
      rola::static_for<kAccTiles>([&](auto I) {
        constexpr int i = decltype(I)::value;
        *reinterpret_cast<float4*>(parked + (i * 32 + lane) * 16) =
            make_float4(acc[i][1][0], acc[i][1][1], acc[i][1][2], acc[i][1][3]);
        rola::static_for<4>(
            [&](auto E) { tile[i][decltype(E)::value] = acc[i][0][decltype(E)::value]; });
      });
      if (job.kind == kJobRoute)
        solve_tile(p, job, job_index, token0 + kWarpTokens * warp, lane, tile, staged, clock);
      else
        value_tile(p, job, token0 + kWarpTokens * warp, lane, tile);
      rola::static_for<kAccTiles>([&](auto I) {
        constexpr int i = decltype(I)::value;
        const float4 v = *reinterpret_cast<const float4*>(parked + (i * 32 + lane) * 16);
        tile[i][0] = v.x, tile[i][1] = v.y, tile[i][2] = v.z, tile[i][3] = v.w;
      });
      if (job.kind == kJobRoute)
        solve_tile(p, job, job_index, token0 + kWarpTokens * warp + 8, lane, tile, staged, clock);
      else
        value_tile(p, job, token0 + kWarpTokens * warp + 8, lane, tile);
    }
  }

  const Job job = load_job(p, job_index, shape);
  if (job.kind == kJobRoute) {
    //: a union's rows are its split's step (the stub's meanwhile); every other route job bundled in its solve
    if (job.solve >= kUnion15) {
      stub_rows(p, job, token0, live, warp, lane);
      clock.lap(kPhaseBundle);
    }
    finish_head(p, job, tile, token0);
    clock.lap(kPhaseFinish);
  } else {
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
                          Tensor amp_write, std::optional<Tensor> dense_read,
                          std::optional<Tensor> dense_write, Tensor scale_read, Tensor scale_write,
                          Tensor mass_read, Tensor mass_write, Tensor read_mass, Tensor g_write,
                          Tensor v, Tensor v_scale, Tensor arrivals,
                          std::optional<Tensor> logit_probe, std::optional<Tensor> level_probe,
                          Tensor shapes, int64_t levels, int64_t route_jobs_per_head,
                          int64_t dense_read_mask, int64_t dense_write_mask) {
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
  if (route_bias.has_value()) detail::check_f32(*route_bias, "route_bias", H * C);
  STD_TORCH_CHECK(gain_bias.is_cuda() && gain_bias.scalar_type() == Dtype::Float
                      && gain_bias.is_contiguous() && gain_bias.numel() % H == 0,
                  "gain_bias is a contiguous CUDA fp32 [H, G]");
  //: every hidden run is read in 16-byte chunks: eight elements, aligned
  STD_TORCH_CHECK(hidden % 8 == 0 && route_W.stride(0) % 8 == 0 && route_W.stride(2) % 8 == 0
                      && v_W.stride(0) % 8 == 0,
                  "the hidden size and the weights' strides are multiples of 8, got hidden ",
                  hidden);
  for (const Tensor* t : {&x, &route_W, &v_W}) {
    STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(t->const_data_ptr()) % 16 == 0,
                    "x and the weights start 16-byte aligned");
  }
  //: the copy sources are 32-bit element offsets (produce.md#stage): x's tokens, a head's routing columns, a value
  //: job's channels
  STD_TORCH_CHECK(
      x.numel() < (int64_t{1} << 32) && route_W.size(2) * route_W.stride(2) < (int64_t{1} << 32)
          && v_W.numel() < (int64_t{1} << 32),
      "x, a head's routing columns and the value weights each hold under 2^32 elements");
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
  for (auto [t, name] : {std::pair{&amp_read, "amp_read"}, {&amp_write, "amp_write"}}) {
    STD_TORCH_CHECK(t->scalar_type() == Dtype::Byte && t->is_contiguous() && t->dim() == 4
                        && t->size(1) == H && t->size(0) * t->size(2) == BT,
                    name, " is head-major [B, H, T, rows] u8, contiguous");
  }
  STD_TORCH_CHECK(amp_read.size(2) == amp_write.size(2), "both sides' rows hold the same tokens");
  //: A SIDE'S DENSE PLANE: its dense levels' f16 rows, present exactly when its mask names a level
  const std::optional<Tensor>* dense[2] = {&dense_read, &dense_write};
  const int64_t masks[2] = {dense_read_mask, dense_write_mask};
  for (int side = 0; side < 2; ++side) {
    STD_TORCH_CHECK(masks[side] >= 0 && masks[side] < (int64_t{1} << levels)
                        && dense[side]->has_value() == (masks[side] != 0),
                    "side ", side, "'s dense levels' f16 rows come with their mask (", masks[side],
                    ")");
    if (dense[side]->has_value()) {
      const Tensor& t = **dense[side];
      STD_TORCH_CHECK(t.scalar_type() == Dtype::Half && t.is_contiguous() && t.dim() == 4
                          && t.sizes().slice(0, 3) == amp_read.sizes().slice(0, 3),
                      "side ", side,
                      "'s dense plane is head-major [B, H, T, rows] f16, contiguous, the rows' "
                      "tokens");
    }
  }
  for (auto [t, name] : {std::pair{&scale_read, "scale_read"},
                         {&scale_write, "scale_write"},
                         {&mass_read, "mass_read"},
                         {&mass_write, "mass_write"}}) {
    detail::check_f32(*t, name, BT * H * levels);
  }
  detail::check_f32(read_mass, "read_mass", BT * H);
  detail::check_f32(g_write, "g_write", BT * H);
  detail::check_f32(v_scale, "v_scale", BT * H);
  STD_TORCH_CHECK(v.scalar_type() == Dtype::Half && v.is_contiguous() && v.numel() == BT * H * d_v,
                  "v is [BT, H, d_v] f16");
  const int64_t tiles = (BT + kTokenTile - 1) / kTokenTile;
  STD_TORCH_CHECK(arrivals.scalar_type() == Dtype::Int && arrivals.is_contiguous()
                      && arrivals.numel() >= tiles * H,
                  "arrivals holds a counter a (token tile, head): ", tiles * H);
  STD_TORCH_CHECK(!shapes.is_cuda() && shapes.scalar_type() == Dtype::Short
                      && shapes.is_contiguous() && shapes.numel() == jobs.size(0),
                  "shapes is the plan's host int16 shape word a job");
  STD_TORCH_CHECK(jobs.size(0) <= kMaxJobs, "a launch runs at most ", kMaxJobs, " jobs, got ",
                  jobs.size(0));
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
  for (int side = 0; side < 2; ++side) {
    p.dense[side] = dense[side]->has_value()
                        ? reinterpret_cast<__half*>((*dense[side])->mutable_data_ptr())
                        : nullptr;
    p.dense_rows[side] = dense[side]->has_value() ? static_cast<int>((*dense[side])->size(3)) : 0;
    p.dense_mask[side] = static_cast<int>(masks[side]);
  }
  p.scale_read = scale_read.mutable_data_ptr<float>();
  p.scale_write = scale_write.mutable_data_ptr<float>();
  p.mass_read = mass_read.mutable_data_ptr<float>();
  p.mass_write = mass_write.mutable_data_ptr<float>();
  p.read_mass = read_mass.mutable_data_ptr<float>();
  p.g_write = g_write.mutable_data_ptr<float>();
  p.v = reinterpret_cast<__half*>(v.mutable_data_ptr());
  p.v_scale = v_scale.mutable_data_ptr<float>();
  p.arrivals = arrivals.mutable_data_ptr<int32_t>();
  p.logit_probe = logit_probe.has_value() ? logit_probe->mutable_data_ptr<float>() : nullptr;
  p.level_probe = level_probe.has_value() ? level_probe->mutable_data_ptr<float>() : nullptr;
  STD_TORCH_CHECK(detail::g_ledger == nullptr || detail::g_ledger_jobs >= jobs.size(0),
                  "the bound phase ledger holds ", detail::g_ledger_jobs,
                  " jobs; this launch runs ", jobs.size(0));
  std::memcpy(p.shapes, shapes.const_data_ptr(),
              static_cast<size_t>(shapes.numel()) * sizeof(int16_t));
  p.ledger = detail::g_ledger;
  p.BT = static_cast<int>(BT);
  p.T = static_cast<int>(amp_read.size(2));
  p.hidden = static_cast<int>(hidden);
  p.H = static_cast<int>(H);
  p.C = static_cast<int>(C);
  p.G = static_cast<int>(gain_bias.numel() / H);
  p.rows_read = static_cast<int>(amp_read.size(3));
  p.rows_write = static_cast<int>(amp_write.size(3));
  p.D = static_cast<int>(levels);
  p.d_v = static_cast<int>(d_v);
  p.route_jobs_per_head = static_cast<int>(route_jobs_per_head);

  const DeviceGuard guard(x.get_device_index());
  //: ONCE: the attribute call is host time on the launch path; the carveout to shared, so two blocks' stages fit an SM
  static bool configured = false;
  if (!configured) {
    const int optin = current_device_properties().sharedMemPerBlockOptin;
    STD_TORCH_CHECK(kProduceSmem <= optin, "the producer's ", kProduceSmem,
                    "-byte stages exceed this device's ", optin, "-byte opt-in limit");
    ROLA_CUDA_CHECK(cudaFuncSetAttribute(
        detail::produce_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kProduceSmem));
    ROLA_CUDA_CHECK(cudaFuncSetAttribute(detail::produce_kernel,
                                         cudaFuncAttributePreferredSharedMemoryCarveout,
                                         cudaSharedmemCarveoutMaxShared));
    configured = true;
  }
  detail::produce_kernel<<<dim3(static_cast<unsigned>(tiles), static_cast<unsigned>(jobs.size(0))),
                           kProduceThreads, kProduceSmem, current_stream()>>>(p);
  ROLA_CUDA_LAUNCH_CHECK();
}

}  // namespace produce
}  // namespace rola
