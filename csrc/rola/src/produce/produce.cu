// THE PRODUCER'S TRANSLATION UNIT -- the kernel, composed of its components, and the host entry that validates the
// operands and launches it. SKELETON: each component has its real signature, tenants and edges and does the cheapest
// thing that keeps the composition's shape (zero digits, unit gains); the components land one at a time into it.
// see docs/internals/produce/produce.md

#include "common/torch_seam.cuh"

#include "common/arch_runtime.cuh"
#include "produce/produce.cuh"
#include "produce/produce_api.cuh"

namespace rola {
namespace produce {
namespace detail {

__device__ __forceinline__ Job load_job(const int32_t* row) {
  return Job{row[kFieldKind],     row[kFieldHead],  row[kFieldHeads],    row[kFieldSolve],
             row[kFieldWidth],    row[kFieldIn0],   row[kFieldIn1],      row[kFieldOutRead],
             row[kFieldOutWrite], row[kFieldLevel], row[kFieldFoldRead], row[kFieldFoldWrite]};
}

//: THE PROJECTION: the job's columns for the tile's tokens, fp32, into `logits`. STUB: zeros. -- see docs/internals/produce/produce.md#accumulate
__device__ void accumulate(const ProduceParams& /*p*/, const Job& /*job*/, int /*token0*/,
                           float (*logits)[kMaxColumns]) {
#pragma unroll 1
  for (int i = threadIdx.x; i < kTokenTile * kMaxColumns; i += kProduceThreads) {
    (&logits[0][0])[i] = 0.0f;
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
                           float (* /*logits*/)[kMaxColumns]) {
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

__global__ __launch_bounds__(kProduceThreads, 1) void produce_kernel(ProduceParams p) {
  const Job job = load_job(p.jobs + static_cast<int64_t>(blockIdx.x) * kJobFields);
  const int token0 = static_cast<int>(blockIdx.y) * kTokenTile;
  __shared__ float logits[kTokenTile][kMaxColumns];
  accumulate(p, job, token0, logits);
  __syncthreads();
  if (job.kind == kJobRoute) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int token = token0 + warp;
    if (warp < kTokenTile && token < p.BT) {
      route_row(p, job, token, logits[warp], logits[warp] + job.width, threadIdx.x & 31);
    }
    finish_head(p, job, static_cast<int>(blockIdx.y), token0);
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
                          int64_t levels, int64_t route_jobs_per_head) {
  check_arch_table();
  detail::check_bf16(x, "x");
  detail::check_bf16(route_W, "route_W");
  detail::check_bf16(v_W, "v_W");
  STD_TORCH_CHECK(x.dim() == 2 && x.is_contiguous(), "x is [BT, hidden], contiguous, got ",
                  shape(x.sizes()));
  STD_TORCH_CHECK(route_W.dim() == 3 && route_W.stride(2) == 1,
                  "route_W is [H, hidden, C] with its columns contiguous, got ",
                  shape(route_W.sizes()));
  STD_TORCH_CHECK(v_W.dim() == 2 && v_W.stride(0) == 1,
                  "v_W is [H * d_v, hidden] with its channels contiguous, got ",
                  shape(v_W.sizes()));
  const int64_t BT = x.size(0), hidden = x.size(1), H = route_W.size(0), C = route_W.size(2);
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

  ProduceParams p{};
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr());
  p.route_W = reinterpret_cast<const __nv_bfloat16*>(route_W.const_data_ptr());
  p.w_head = route_W.stride(0);
  p.w_k = route_W.stride(1);
  p.route_bias = route_bias.has_value() ? route_bias->const_data_ptr<float>() : nullptr;
  p.gain_bias = gain_bias.const_data_ptr<float>();
  p.v_W = reinterpret_cast<const __nv_bfloat16*>(v_W.const_data_ptr());
  p.v_k = v_W.stride(1);
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
  detail::produce_kernel<<<dim3(static_cast<unsigned>(jobs.size(0)), static_cast<unsigned>(tiles)),
                           kProduceThreads, 0, current_stream()>>>(p);
  ROLA_CUDA_LAUNCH_CHECK();
}

}  // namespace produce
}  // namespace rola
