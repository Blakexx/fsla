// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
//
// K31 R4: the intra term's reverse pass, its own translation unit.
//
// It REDUCES into the four gradient planes rather than returning them, exactly as the
// forward intra reduces into `o`/`den`: the carry's two scans own the cross-window half
// of the same planes, and the stream order is the only synchronization the pair needs.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "intra/intra_backward_kernel.cuh"

namespace rola {
namespace intra {

namespace bwdstamp {

__device__ int64_t g_bwd_stamp_probe;

//: THE HEAVIEST BUILT ARM's footprint, so the stamp moves whenever any arm's shape
//: does -- a fact only THIS binary can produce.
__global__ void bwd_stamp_kernel(int64_t* out) {
  *out = (int64_t)(sizeof(detail::IntraBwdSmem<2, kWidthFlat, kWindowWide>)) * 1000003
         + (int64_t)detail::threads_of(kWindowWide) + detail::kIntraBwdMagic;
  g_bwd_stamp_probe = *out;
}

}  // namespace bwdstamp

namespace {

//: THE REVERSE PASS IS BUILT AT DEPTH TWO ONLY. Its digit reduction's pass
//: geometry is derived per arm from the owner's bit fields, and the carry's two
//: scans refuse a third level for the same reason ...
// see docs/internals/intra/intra_backward.md#intra-bwd-arms-x
#define INTRA_BWD_ARMS_X(X_)             \
  X_(2, kWidthFlat, kWindowFlagship)     \
  X_(2, kWidthFlat, kWindowWide)         \
  X_(2, kWidthFlat, kWindowSmall)

template <int D, int B, int W>
void launch_bwd(const detail::IntraBwdParams& p, int64_t bh, int64_t windows, cudaStream_t stream) {
  constexpr int kSmem = (int)sizeof(detail::IntraBwdSmem<D, B, W>);
  const int optin = at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin;
  TORCH_CHECK(kSmem <= optin, "the intra backward arm's ", kSmem,
              "-byte block footprint exceeds this device's ", optin, "-byte opt-in limit");
  C10_CUDA_CHECK(cudaFuncSetAttribute(detail::intra_backward_kernel<D, B, W>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  const dim3 grid((unsigned)windows, (unsigned)bh, 1);
  detail::intra_backward_kernel<D, B, W><<<grid, detail::threads_of(W), kSmem, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void intra_backward(const at::Tensor& pread, const at::Tensor& pwrite, const at::Tensor& gwrite,
                    const at::Tensor& v, const at::Tensor& dnum, const at::Tensor& dden,
                    at::Tensor& d_pr, at::Tensor& d_pw, at::Tensor& d_gw, at::Tensor& d_v,
                    int64_t levels, int64_t level_width, int64_t level_modes, int64_t window) {
  TORCH_CHECK(pread.is_cuda() && pread.is_contiguous() && pread.scalar_type() == at::kBFloat16,
              "the operand planes are contiguous CUDA bf16");
  const int64_t bh = pread.size(0);
  const int64_t L = pread.size(1);
  TORCH_CHECK(L % window == 0, "L = ", L, " is not a whole number of ", window, "-token windows");
  TORCH_CHECK(dnum.scalar_type() == at::kBFloat16 && dnum.is_contiguous()
                  && dnum.sizes() == at::IntArrayRef({bh, L, (int64_t)kValueWidth}),
              "the seed's numerator is a contiguous [BH, L, d_v] bf16 plane");
  TORCH_CHECK(dden.scalar_type() == at::kFloat && dden.is_contiguous(),
              "the seed's denominator is a contiguous [BH, L] fp32 plane");

  detail::IntraBwdParams p{};
  p.pread = (const __nv_bfloat16*)pread.data_ptr();
  p.pwrite = (const __nv_bfloat16*)pwrite.data_ptr();
  p.gwrite = (const __nv_bfloat16*)gwrite.data_ptr();
  p.v = (const __nv_bfloat16*)v.data_ptr();
  p.dnum = (const __nv_bfloat16*)dnum.data_ptr();
  p.dden = dden.data_ptr<float>();
  p.d_pr = d_pr.data_ptr<float>();
  p.d_pw = d_pw.data_ptr<float>();
  p.d_gw = d_gw.data_ptr<float>();
  p.d_v = d_v.data_ptr<float>();
  p.L = (int)L;
  p.row_width = (int)pread.size(2);
  p.heads = (int)v.size(2);
  p.tokens = (int)v.size(1);
  p.modes = (uint32_t)level_modes;

  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t windows = L / window;
#define INTRA_BWD_DISPATCH(D_, B_, W_)                             \
  if (levels == D_ && level_width == B_ && window == W_) {          \
    launch_bwd<D_, B_, W_>(p, bh, windows, stream);                 \
    return;                                                         \
  }
  INTRA_BWD_ARMS_X(INTRA_BWD_DISPATCH)
#undef INTRA_BWD_DISPATCH
  TORCH_CHECK(false, "unbuilt intra backward arm: levels=", levels, " width=", level_width,
              " window=", window);
}

std::vector<std::vector<int64_t>> intra_backward_arms() {
  std::vector<std::vector<int64_t>> out;
#define INTRA_BWD_LIST(D_, B_, W_)                                                     \
  {                                                                                     \
    cudaFuncAttributes a{};                                                             \
    cudaFuncGetAttributes(&a, detail::intra_backward_kernel<D_, B_, W_>);                \
    out.push_back({D_, B_, W_, (int64_t)sizeof(detail::IntraBwdSmem<D_, B_, W_>),        \
                   a.numRegs, (int64_t)a.localSizeBytes, (int64_t)detail::threads_of(W_)}); \
  }
  INTRA_BWD_ARMS_X(INTRA_BWD_LIST)
#undef INTRA_BWD_LIST
  return out;
}

int64_t intra_backward_stamp() {
  auto out = at::zeros({1}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA));
  bwdstamp::bwd_stamp_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      out.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out.cpu().data_ptr<int64_t>()[0];
}

}  // namespace intra
}  // namespace rola
