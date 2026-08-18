// THE DECODE FOLD -- every per-token operand widened, normalized and published in ONE
// launch, into buffers the sequence already owns.
//
// Three facts a reader must not miss:
//   1. THE FOLD IS A GATHER, NOT A COPY PASS. Its inputs are whatever VIEW the producer
//      handed out -- a level is ordinarily a slice of one packed plane -- so the kernel
//      indexes by stride and the `[B, 1, H, W] -> [B*H, W]` fold is the address it
//      computes, never a materialized intermediate.
//   2. THE SUM'S ORDER IS DECLARED, not incidental: lane-strided ascending partials
//      folded by a 5-step `__shfl_down_sync` tree. A reduction whose order is a
//      property of whatever the library did that release is a reduction no gate can
//      hold to bit-identity.
//   3. NOTHING HERE ALLOCATES. Every destination is a `DecodeScratch` buffer the
//      sequence allocated once, which is what keeps the whole step capturable.
//
// The launch ledger, the declared order and the cast argument:
// docs/internals/decode/decode_fold.md

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include "decode.cuh"
#include "decode_api.cuh"

namespace rola {
namespace decode {
namespace detail {

__device__ __forceinline__ float widen(float x) { return x; }
__device__ __forceinline__ float widen(const __nv_bfloat16& x) { return __bfloat162float(x); }

//: One operand's ADDRESS, as the producer left it: the base and the three strides the
//: fold walks. `sw` is 0 for the per-token scalars, which have no last axis.
struct Span {
  const void* base;
  int64_t sb;
  int64_t sh;
  int64_t sw;
};

struct FoldParams {
  Span read[kMaxLevels];
  Span write[kMaxLevels];
  Span g_write;
  Span v;
  float* out_read[kMaxLevels];
  float* out_write[kMaxLevels];
  float* out_g;
  float* out_v;
  int level_width[kMaxLevels];
  int normalize[kMaxLevels];
  int D;
  int BH;
  int H;
  int d_v;
};

template <typename T>
__device__ __forceinline__ const T* row_of(const Span& s, int b, int h) {
  return static_cast<const T*>(s.base) + b * s.sb + h * s.sh;
}

//: THE DECLARED SUM. Lane `j` accumulates `row[j], row[j + 32], ...` in ascending index
//: order into an fp32 register, then the five `__shfl_down_sync` steps fold lane `j`
//: into lane `j - off`. Lanes past the row's width contribute an exact `0.0f`.
//: docs/internals/decode/decode_fold.md#declared-order
template <typename T>
__device__ __forceinline__ float row_sum(const T* row, int64_t sw, int width, int lane) {
  float acc = 0.0f;
  for (int k = lane; k < width; k += 32) acc += widen(row[k * sw]);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  return acc;
}

//: One CTA per batch-head. Warp `l < D` owns level `l`'s normalizing sum; the whole CTA
//: then publishes every row, the value and the gain.
template <typename TL, typename TG, typename TV>
__global__ __launch_bounds__(kDecodeThreads) void decode_fold_kernel(FoldParams p) {
  const int bh = blockIdx.x;
  const int b = bh / p.H;
  const int h = bh - b * p.H;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  __shared__ float s_sum[kMaxLevels];
  if (warp < p.D && p.normalize[warp] != 0) {
    const Span& span = p.read[warp];
    const float total = row_sum(row_of<TL>(span, b, h), span.sw, p.level_width[warp], lane);
    if (lane == 0) s_sum[warp] = total;
  }
  __syncthreads();

  for (int l = 0; l < p.D; ++l) {
    const int width = p.level_width[l];
    const TL* r_in = row_of<TL>(p.read[l], b, h);
    const TL* w_in = row_of<TL>(p.write[l], b, h);
    const int64_t r_sw = p.read[l].sw;
    const int64_t w_sw = p.write[l].sw;
    float* r_out = p.out_read[l] + static_cast<int64_t>(bh) * width;
    float* w_out = p.out_write[l] + static_cast<int64_t>(bh) * width;
    if (p.normalize[l] != 0) {
      const float total = s_sum[l];
      for (int g = tid; g < width; g += kDecodeThreads) r_out[g] = widen(r_in[g * r_sw]) / total;
    } else {
      for (int g = tid; g < width; g += kDecodeThreads) r_out[g] = widen(r_in[g * r_sw]);
    }
    for (int g = tid; g < width; g += kDecodeThreads) w_out[g] = widen(w_in[g * w_sw]);
  }

  const TV* v_in = row_of<TV>(p.v, b, h);
  const int64_t v_sw = p.v.sw;
  for (int c = tid; c < p.d_v; c += kDecodeThreads) {
    p.out_v[static_cast<int64_t>(bh) * p.d_v + c] = widen(v_in[c * v_sw]);
  }
  if (tid == 0) p.out_g[bh] = widen(*row_of<TG>(p.g_write, b, h));
}

//: ONE FLOATING AXIS OVER THE CLOSED SET THE PRODUCER AND THE VALUE PROJECTION EMIT.
//: `body` is called with a tag of the storage type, so `decltype(T)` is a template
//: argument at the call site -- the type mirror of `rola::int_switch`, and an
//: unmatched dtype is a REFUSAL rather than a silent widening on the host.
template <typename Body>
inline void float_switch(at::ScalarType dtype, const char* what, Body&& body) {
  if (dtype == at::kFloat) {
    body(float{});
  } else if (dtype == at::kBFloat16) {
    body(__nv_bfloat16{});
  } else {
    TORCH_CHECK(false, what, " must be float32 or bfloat16, got ", dtype);
  }
}

void check_folded(const at::Tensor& t, int64_t rows, int64_t cols, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kFloat && t.is_contiguous(), name,
              " must be a contiguous fp32 CUDA tensor, got ", t.scalar_type());
  TORCH_CHECK(t.dim() == 2 && t.size(0) == rows && t.size(1) == cols, name, " must be [", rows,
              ", ", cols, "], got ", t.sizes());
}

//: The `[B, 1, H, W] -> [B * H, W]` fold as an ADDRESS. `T != 1` is refused -- the fold
//: is defined on one token and a second one would be silently dropped:
//: docs/internals/decode/decode_fold.md#the-identity
Span span_of(const at::Tensor& t, int64_t BH, int64_t width, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.dim() == 4 && t.size(1) == 1, name, " must be [B, 1, H, W], got ", t.sizes());
  TORCH_CHECK(t.size(0) * t.size(2) == BH, name, " folds to ", t.size(0) * t.size(2),
              " batch-heads against ", BH);
  TORCH_CHECK(width == 0 || t.size(3) == width, name, " is ", t.size(3), " wide against ", width);
  return Span{t.data_ptr(), t.stride(0), t.stride(2), t.stride(3)};
}

Span scalar_span_of(const at::Tensor& t, int64_t BH, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.dim() == 3 && t.size(1) == 1, name, " must be [B, 1, H], got ", t.sizes());
  TORCH_CHECK(t.size(0) * t.size(2) == BH, name, " folds to ", t.size(0) * t.size(2),
              " batch-heads against ", BH);
  return Span{t.data_ptr(), t.stride(0), t.stride(2), 0};
}

}  // namespace detail

void rola_decode_fold(std::vector<at::Tensor> read, std::vector<at::Tensor> write,
                      at::Tensor g_write, at::Tensor v, std::vector<at::Tensor> amp_read,
                      std::vector<at::Tensor> amp_write, at::Tensor g_out, at::Tensor v_out,
                      std::vector<int64_t> normalize) {
  const int D = static_cast<int>(read.size());
  TORCH_CHECK(D >= 1 && D <= kMaxLevels, "D must be in [1, 4], got ", D);
  TORCH_CHECK(static_cast<int>(write.size()) == D && static_cast<int>(amp_read.size()) == D
                  && static_cast<int>(amp_write.size()) == D
                  && static_cast<int>(normalize.size()) == D,
              "read, write, amp_read, amp_write and normalize must each have D entries");

  TORCH_CHECK(v.dim() == 4 && v.size(1) == 1, "v must be [B, 1, H, d_v], got ", v.sizes());
  const int64_t BH = v.size(0) * v.size(2);
  const int d_v = static_cast<int>(v.size(3));

  detail::FoldParams p{};
  p.D = D;
  p.BH = static_cast<int>(BH);
  p.H = static_cast<int>(v.size(2));
  p.d_v = d_v;
  p.v = detail::span_of(v, BH, d_v, "v");
  p.g_write = detail::scalar_span_of(g_write, BH, "g_write");
  detail::check_folded(v_out, BH, d_v, "v_out");
  TORCH_CHECK(g_out.is_cuda() && g_out.scalar_type() == at::kFloat && g_out.is_contiguous()
                  && g_out.dim() == 1 && g_out.size(0) == BH,
              "g_out must be a contiguous [BH] fp32 CUDA tensor");
  p.out_g = g_out.data_ptr<float>();
  p.out_v = v_out.data_ptr<float>();

  const at::ScalarType level_dtype = read[0].scalar_type();
  for (int l = 0; l < D; ++l) {
    p.read[l] = detail::span_of(read[l], BH, 0, "read[l]");
    const int64_t width = read[l].size(3);
    TORCH_CHECK(width >= 1 && width <= kDecodeMaxLevelWidth, "level ", l, " width ", width,
                " exceeds the decode path's capacity ", kDecodeMaxLevelWidth);
    p.write[l] = detail::span_of(write[l], BH, width, "write[l]");
    TORCH_CHECK(read[l].scalar_type() == level_dtype && write[l].scalar_type() == level_dtype,
                "every routing level shares one dtype; level ", l, " disagrees");
    detail::check_folded(amp_read[l], BH, width, "amp_read[l]");
    detail::check_folded(amp_write[l], BH, width, "amp_write[l]");
    p.out_read[l] = amp_read[l].data_ptr<float>();
    p.out_write[l] = amp_write[l].data_ptr<float>();
    p.level_width[l] = static_cast<int>(width);
    p.normalize[l] = normalize[l] != 0 ? 1 : 0;
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  detail::float_switch(level_dtype, "the routing levels", [&](auto TL) {
    detail::float_switch(g_write.scalar_type(), "g_write", [&](auto TG) {
      detail::float_switch(v.scalar_type(), "v", [&](auto TV) {
        detail::decode_fold_kernel<decltype(TL), decltype(TG), decltype(TV)>
            <<<p.BH, kDecodeThreads, 0, stream>>>(p);
      });
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace decode
}  // namespace rola
