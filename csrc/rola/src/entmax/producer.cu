// THE FUSED PRODUCER KERNEL -- x in, the finished packed-bf16 bundle out.
// Design of record: docs/internals/entmax/producer.md.
//
// The producer plane used to be five kernels and three DRAM round trips: a cuBLAS fp32
// router GEMM, a materialized fp32 logit plane, one solve launch per (operator, operand
// set), a fold reduction and a gain projection. Here it is ONE launch: a CTA computes its
// 32-token logit tile with bf16 operands on the tensor cores, keeps it in shared memory,
// runs the exact per-row solves over it level by level, and writes the packed bf16
// bundle once. The fp32 logits never reach DRAM at all.
//
// THE PROBE is the same tile-GEMM component with the tile emitted instead of solved, so
// "the gated logits are the shipped logits" is a call-graph fact and not a claim about
// two implementations -- docs/internals/entmax/producer.md#probe

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "dispatch_switch.cuh"
#include "entmax.cuh"
#include "producer.cuh"
#include "solve_row.cuh"
#include "tile_gemm.cuh"

namespace rola {
namespace entmax {
namespace detail {

//: The two staging regions ALIAS: `xs`/`ws` are dead once the last `mma` retires, and the
//: fp32 tile is written after the barrier that ends the K loop.
struct TileSmem {
  int columns, ws_stride, tile_stride;
  __device__ __forceinline__ __nv_bfloat16* xs(char* p) const { return (__nv_bfloat16*)p; }
  __device__ __forceinline__ __nv_bfloat16* ws(char* p) const {
    return (__nv_bfloat16*)(p + WORD_TOKENS * XS_STRIDE * (int)sizeof(__nv_bfloat16));
  }
  __device__ __forceinline__ float* tile(char* p) const { return (float*)p; }
};

inline int ws_stride_for(int columns) { return (columns + 7) & ~7; }

inline int64_t tile_smem_bytes(int columns) {
  const int64_t staging = (int64_t)WORD_TOKENS * XS_STRIDE * 2 + (int64_t)TILE_K
                                                                     * ws_stride_for(columns) * 2;
  const int64_t tile = (int64_t)WORD_TOKENS * columns * 4;
  return staging > tile ? staging : tile;
}

template <int NT>
__global__ __launch_bounds__(GEMM_THREADS, 2) void producer_probe_kernel(
    const void* __restrict__ x, bool x_bf16, const void* __restrict__ w, bool w_bf16,
    const float* __restrict__ bias, float* __restrict__ logits, int tokens, int hidden,
    int columns, int heads, int64_t x_batch_stride, int64_t x_token_stride,
    int64_t w_head_stride, TileSmem layout, Strides sout) {
  extern __shared__ __align__(16) char smem[];
  const int stream32 = (int)blockIdx.y;
  const int64_t sb = stream32 / heads;
  const int64_t sh = stream32 - (int)sb * heads;
  const int token0 = blockIdx.x * WORD_TOKENS;
  const int rows = min(WORD_TOKENS, tokens - token0);

  tile_gemm<NT>(x, x_bf16, x_token_stride, sb * x_batch_stride + (int64_t)token0 * x_token_stride,
                w, w_bf16, sh * w_head_stride, bias == nullptr ? nullptr : bias + sh * columns,
                hidden, columns, rows, layout.xs(smem), layout.ws(smem), layout.ws_stride,
                layout.tile(smem), layout.tile_stride);
  __syncthreads();

  const float* tile = layout.tile(smem);
  for (int e = threadIdx.x; e < rows * columns; e += GEMM_THREADS) {
    const int row = e / columns, col = e - row * columns;
    logits[addr_split(sout, sb, sh, token0 + row, col)] = tile[row * layout.tile_stride + col];
  }
}

}  // namespace detail

void producer_probe(const torch::Tensor& x, const torch::Tensor& route_W,
                    const c10::optional<torch::Tensor>& route_bias, torch::Tensor& logits) {
  TORCH_CHECK(x.is_cuda() && route_W.is_cuda() && logits.is_cuda(), "the producer probe is CUDA");
  TORCH_CHECK(x.dim() == 3, "x must be [B, T, hidden]");
  TORCH_CHECK(route_W.dim() == 3, "route_W must be [H, hidden, columns]");
  TORCH_CHECK(logits.dim() == 4 && logits.scalar_type() == torch::kFloat32,
              "the probe emits fp32 [B, T, H, columns] logits");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kFloat32,
              "the producer's source stream is bf16 or fp32");
  TORCH_CHECK(route_W.scalar_type() == torch::kBFloat16 || route_W.scalar_type() == torch::kFloat32,
              "route_W is bf16 or fp32");
  TORCH_CHECK(x.stride(2) == 1 && route_W.stride(2) == 1 && route_W.stride(1) == route_W.size(2),
              "the tile GEMM reads x and route_W as contiguous rows");

  const int heads = (int)route_W.size(0), hidden = (int)route_W.size(1);
  const int columns = (int)route_W.size(2);
  const int batch = (int)x.size(0), tokens = (int)x.size(1);
  TORCH_CHECK(x.size(2) == hidden, "x and route_W disagree about hidden");
  TORCH_CHECK(logits.size(0) == batch && logits.size(1) == tokens && logits.size(2) == heads
                  && logits.size(3) == columns,
              "the probe's logit buffer must be [B, T, H, columns]");
  TORCH_CHECK(columns <= detail::MAX_PACKED_COLUMNS, "the fused producer holds at most ",
              detail::MAX_PACKED_COLUMNS, " packed router columns per head, got ", columns);
  const float* bias = nullptr;
  if (route_bias.has_value()) {
    TORCH_CHECK(route_bias->scalar_type() == torch::kFloat32 && route_bias->dim() == 2
                    && route_bias->size(0) == heads && route_bias->size(1) == columns,
                "route_bias must be fp32 [H, columns]");
    bias = route_bias->data_ptr<float>();
  }

  detail::TileSmem layout{columns, detail::ws_stride_for(columns), columns};
  const int64_t smem = detail::tile_smem_bytes(columns);
  const int n_tiles = (columns + detail::MMA_N - 1) / detail::MMA_N;
  const int nt = (n_tiles + detail::GEMM_WARPS - 1) / detail::GEMM_WARPS;
  dim3 grid((tokens + detail::WORD_TOKENS - 1) / detail::WORD_TOKENS, batch * heads);
  auto stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::CUDAGuard guard(x.device());
  const detail::Strides sout{logits.stride(0), logits.stride(2), logits.stride(1), logits.stride(3),
                             0, 0};
  int_switch<1, 2, 3, 4, 5, 6, 7, 8>(nt, "fused producer n-tiles per warp", [&](auto NT) {
    auto kernel = detail::producer_probe_kernel<decltype(NT)::value>;
    if (smem > 48 * 1024)
      C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          (int)smem));
    kernel<<<grid, detail::GEMM_THREADS, smem, stream>>>(
        x.data_ptr(), x.scalar_type() == torch::kBFloat16, route_W.data_ptr(),
        route_W.scalar_type() == torch::kBFloat16, bias, logits.data_ptr<float>(), tokens, hidden,
        columns, heads, x.stride(0), x.stride(1), route_W.stride(0), layout, sout);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace entmax
}  // namespace rola
