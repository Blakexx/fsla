// THE PRODUCER'S TRANSLATION UNIT -- the kernel, composed of its components, and the host entry that validates the
// operands and launches it. SKELETON: each component has its real signature, tenants and edges and does the cheapest
// thing that keeps the composition's shape (zero digits, unit gains); the components land one at a time into it.
// see docs/internals/produce/produce.md

#include "common/torch_seam.cuh"

#include "common/arch_runtime.cuh"
#include "common/burst.cuh"
#include "common/ops.cuh"
#include "common/static_for.cuh"
#include "common/structure_switch.cuh"
#include "produce/produce.cuh"
#include "produce/produce_api.cuh"

namespace rola {
namespace produce {
namespace detail {

namespace ops = denseref::ops;

__device__ __forceinline__ Job load_job(const int32_t* row) {
  return Job{row[kFieldKind],     row[kFieldHead],  row[kFieldHeads],    row[kFieldSolve],
             row[kFieldWidth],    row[kFieldIn0],   row[kFieldIn1],      row[kFieldOutRead],
             row[kFieldOutWrite], row[kFieldLevel], row[kFieldFoldRead], row[kFieldFoldWrite]};
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

//: ONE TOKEN'S ROW, solved and written as the bundle holds it: digits, scale, mass. STUB: zeros. -- see docs/internals/produce/produce.md#route-row
__device__ void route_row(const ProduceParams& p, const Job& job, int token, const float* /*row0*/,
                          const float* /*row1*/, int lane) {
  const int64_t base = (static_cast<int64_t>(token) * p.H + job.head) * p.rows;
#pragma unroll 1
  for (int col = lane; col < job.width; col += 32) {
    if (job.out_read >= 0) write_digit(p, p.amp_read, base + job.out_read + col, 0.0f);
    if (job.out_write >= 0) write_digit(p, p.amp_write, base + job.out_write + col, 0.0f);
  }
  if (lane == 0) {
    const int64_t at = (static_cast<int64_t>(token) * p.H + job.head) * p.D + job.level;
    if (job.out_read >= 0) {
      p.scale_read[at] = 0.0f;
      p.mass_read[at] = 0.0f;
    }
    if (job.out_write >= 0) {
      p.scale_write[at] = 0.0f;
      p.mass_write[at] = 0.0f;
    }
  }
}

//: THE HEAD'S FINISH, by the last of its route jobs in the tile: read_mass and g_write. STUB: ones. -- see docs/internals/produce/produce.md#finish
__device__ void finish_head(const ProduceParams& p, const Job& job, int tile, int token0) {
  __shared__ int s_last;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    s_last = atomicAdd(&p.arrivals[tile * p.H + job.head], 1) + 1 == p.route_jobs_per_head ? 1 : 0;
  }
  __syncthreads();
  if (s_last == 0) return;
  __threadfence();
  if (threadIdx.x < kTokenTile && token0 + static_cast<int>(threadIdx.x) < p.BT) {
    const int64_t at = static_cast<int64_t>(token0 + threadIdx.x) * p.H + job.head;
    p.read_mass[at] = 1.0f;
    p.g_write[at] = 1.0f;
  }
  if (threadIdx.x == 0) p.arrivals[tile * p.H + job.head] = 0;
}

//: THE VALUE DIGITS of the job's heads for the tile's tokens. STUB: zeros. -- see docs/internals/produce/produce.md#value-rows
__device__ void value_rows(const ProduceParams& p, const Job& job, int token0,
                           const float* /*logits*/) {
  const int channels = job.heads * p.d_v;
#pragma unroll 1
  for (int i = threadIdx.x; i < kTokenTile * channels; i += kProduceThreads) {
    const int token = token0 + i / channels;
    if (token >= p.BT) continue;
    const int64_t at = static_cast<int64_t>(token) * p.H * p.d_v + job.head * p.d_v + i % channels;
    p.v_hat[at] = 0;
    if (p.v_lo != nullptr) p.v_lo[at] = 0;
  }
  if (threadIdx.x < kTokenTile * job.heads) {
    const int token = token0 + threadIdx.x / job.heads;
    if (token < p.BT)
      p.v_scale[static_cast<int64_t>(token) * p.H + job.head + threadIdx.x % job.heads] = 0.0f;
  }
}

//: THE GRID: token tiles fastest, so the blocks resident together share one job's weights in L2.
__global__ __launch_bounds__(kProduceThreads, 1) void produce_kernel(ProduceParams p) {
  const int job_index = static_cast<int>(blockIdx.y), tile = static_cast<int>(blockIdx.x);
  const Job job = load_job(p.jobs + static_cast<int64_t>(job_index) * kJobFields);
  const int token0 = tile * kTokenTile;
  extern __shared__ float logits[];
  accumulate(p, job, token0, logits);
  __syncthreads();
  if (p.logit_probe != nullptr) probe_logits(p, job_index, token0, job_columns(p, job), logits);
  if (job.kind == kJobRoute) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll 1
    for (int row = warp; row < kTokenTile && token0 + row < p.BT; row += kProduceWarps) {
      const float* logit_row = logits + row * kLogitStride;
      route_row(p, job, token0 + row, logit_row, logit_row + job.width, threadIdx.x & 31);
    }
    finish_head(p, job, tile, token0);
  } else {
    value_rows(p, job, token0, logits);
  }
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

}  // namespace detail

//: -- see docs/internals/produce/produce_api.md#rola-produce-forward
void rola_produce_forward(Tensor x, Tensor route_W, std::optional<Tensor> route_bias,
                          Tensor gain_bias, Tensor v_W, Tensor jobs, Tensor amp_read,
                          Tensor amp_write, Tensor scale_read, Tensor scale_write, Tensor mass_read,
                          Tensor mass_write, Tensor read_mass, Tensor g_write, Tensor v_hat,
                          std::optional<Tensor> v_lo, Tensor v_scale, Tensor arrivals,
                          std::optional<Tensor> logit_probe, int64_t levels,
                          int64_t route_jobs_per_head) {
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
  STD_TORCH_CHECK(jobs.size(0) <= 65535, "a launch runs at most 65535 jobs, got ", jobs.size(0));
  if (logit_probe.has_value()) {
    detail::check_f32(*logit_probe, "logit_probe", jobs.size(0) * BT * kMaxColumns);
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
