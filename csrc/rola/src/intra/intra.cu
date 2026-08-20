// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "intra/intra_kernel.cuh"

namespace rola {
namespace intra {

namespace {

//: Recompiled with the translation unit, so a test that reads it back is
//: reading a fact only the loaded binary can produce.
__device__ int64_t g_stamp_probe;

__global__ void stamp_kernel(int64_t* out) {
  *out = (int64_t)(sizeof(detail::IntraSmem<2>)) * 1000003 + (int64_t)detail::kThreads;
  g_stamp_probe = *out;
}

void check_plane(const at::Tensor& t, const char* name, int64_t bh, int64_t L, int64_t width) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kBFloat16, name,
              " must be a contiguous CUDA bf16 tensor");
  TORCH_CHECK(t.dim() == 3 && t.size(0) == bh && t.size(1) == L && t.size(2) == width, name,
              " must be [BH, L, sum-of-widths]");
}

template <int D>
void launch(const detail::IntraParams& p, int64_t bh, int64_t windows, cudaStream_t stream) {
  const dim3 grid((unsigned)windows, (unsigned)bh, 1);
  detail::intra_kernel<D><<<grid, detail::kThreads, 0, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void intra_forward(const at::Tensor& pread, const at::Tensor& pwrite, const at::Tensor& gwrite,
                   const at::Tensor& v, at::Tensor& o, at::Tensor& den, int64_t levels,
                   int64_t level_modes) {
  TORCH_CHECK(levels == 2, "the intra kernel is built for D = 2");
  const int64_t bh = pread.size(0);
  const int64_t L = pread.size(1);
  const int64_t width = levels * kLevelWidth;
  check_plane(pread, "pread", bh, L, width);
  check_plane(pwrite, "pwrite", bh, L, width);
  TORCH_CHECK(gwrite.is_cuda() && gwrite.is_contiguous() && gwrite.scalar_type() == at::kBFloat16
                  && gwrite.dim() == 2 && gwrite.size(0) == bh && gwrite.size(1) == L,
              "gwrite must be a contiguous CUDA bf16 [BH, L] tensor");
  TORCH_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == at::kBFloat16 && v.dim() == 4
                  && v.size(3) == kValueWidth,
              "v must be a contiguous CUDA bf16 [B, T, H, 64] tensor");
  const int64_t batch = v.size(0), tokens = v.size(1), heads = v.size(2);
  TORCH_CHECK(batch * heads == bh && tokens == L,
              "v's [B, T, H] must flatten to the plane's BH, L");
  TORCH_CHECK(L % kWindow == 0, "L must be a whole number of ", kWindow, "-token windows");
  TORCH_CHECK(o.is_cuda() && o.is_contiguous() && o.scalar_type() == at::kFloat && o.dim() == 3
                  && o.size(0) == bh && o.size(1) == L && o.size(2) == kValueWidth,
              "o must be a contiguous CUDA fp32 [BH, L, 64] tensor");
  TORCH_CHECK(den.is_cuda() && den.is_contiguous() && den.scalar_type() == at::kFloat
                  && den.dim() == 2 && den.size(0) == bh && den.size(1) == L,
              "den must be a contiguous CUDA fp32 [BH, L] tensor");
  TORCH_CHECK(level_modes >= 0 && level_modes < (1LL << (2 * levels)),
              "level_modes carries two bits per level and no more");

  const at::cuda::CUDAGuard guard(pread.device());
  detail::IntraParams p{
      reinterpret_cast<const __nv_bfloat16*>(pread.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(pwrite.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(gwrite.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
      o.data_ptr<float>(),
      den.data_ptr<float>(),
      (int)L,
      (int)width,
      (int)heads,
      (int)tokens,
      (uint32_t)level_modes,
  };
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch<2>(p, bh, L / kWindow, stream);
}

int64_t intra_build_stamp() {
  auto out = at::empty({1}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA));
  stamp_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(out.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out.cpu().item<int64_t>();
}

}  // namespace intra
}  // namespace rola
