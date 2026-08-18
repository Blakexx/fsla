// csrc/rola/src/chunk/ladder_r0.cu -- the backward ladder's rung-0 translation
// unit: the arm surface, the census query and the torch entries.  A TU OF ITS
// OWN, on `backward.cu`'s rule -- these instantiations must not enter a shard
// whose per-`cicc` footprint the build's job formula was measured on
// (docs/build.md).  docs/internals/chunk/ladder_r0.md
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <array>
#include <vector>

#include "ladder_r0.cuh"

namespace rola::chunk {

void check_arch_table();

template <int C, int BC>
static cudaError_t launch_rung0_pass1(const Rung0Params& p, cudaStream_t stream) {
  constexpr size_t sm = GeomR0<C, BC, false>::bytes;
  static bool configured = false;
  if (!configured) {
    const cudaError_t e = cudaFuncSetAttribute(
        rung0_pass1_kernel<C, BC>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    if (e != cudaSuccess) return e;
    configured = true;
  }
  const dim3 grid((unsigned)(p.N / BC), (unsigned)p.BH);
  rung0_pass1_kernel<C, BC><<<grid, r0_threads_for(BC), sm, stream>>>(
      p.w, p.v, p.dnum, p.dden, p.dr, p.L, p.N, p.v_row_bytes);
  return cudaGetLastError();
}

template <int C, int BC>
static cudaError_t launch_rung0_pass2(const Rung0Params& p, cudaStream_t stream) {
  constexpr size_t sm = GeomR0<C, BC, true>::bytes;
  static bool configured = false;
  if (!configured) {
    const cudaError_t e = cudaFuncSetAttribute(
        rung0_pass2_kernel<C, BC>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    if (e != cudaSuccess) return e;
    configured = true;
  }
  const dim3 grid((unsigned)(p.N / BC), (unsigned)p.BH);
  rung0_pass2_kernel<C, BC><<<grid, r0_threads_for(BC), sm, stream>>>(
      p.r, p.w, p.v, p.dnum, p.dden, p.dv, p.dw, p.L, p.N, p.v_row_bytes);
  return cudaGetLastError();
}

//: THE CENSUS ROW CARRIES BOTH HALVES OF THE FLOOR CLAIM -- the DECLARED live set
//: and the BUDGET the target admits -- exactly as the two scans' rows do.
template <int C, int BC, bool PASS2, typename K>
static std::array<int64_t, 9> rung0_census_arm(K kernel) {
  cudaFuncAttributes a{};
  if (cudaFuncGetAttributes(&a, kernel) != cudaSuccess) return {-1, -1, -1, -1, -1, -1, -1, -1, -1};
  return {a.numRegs, (int64_t)a.localSizeBytes, (int64_t)GeomR0<C, BC, PASS2>::bytes,
          (int64_t)GeomR0<C, BC, PASS2>::stages, (int64_t)kR0CtaTarget<C, BC, PASS2>,
          (int64_t)reg_floor_r0(),
          (int64_t)denseref::arch::reg_budget(denseref::arch::kCaps, r0_threads_for(BC),
                                              kR0CtaTarget<C, BC, PASS2>),
          (int64_t)GeomR0<C, BC, PASS2>::slot,
          //: THE TWELFTH COLUMN.  The CTA target alone stopped being the residency
          //: statement when the block gained groups: what the carve holds constant is
          //: WARPS PER SM, and at `BC = 128` that is one CTA of eight where `BC = 64`
          //: runs two of four -- docs/internals/chunk/ladder_r0.md#the-group-threshold
          (int64_t)r0_threads_for(BC)};
}

static void check_operands(const torch::Tensor& w, const torch::Tensor& v,
                           const torch::Tensor& dnum, const torch::Tensor& dden, int64_t chunk,
                           int64_t bc) {
  TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == torch::kBFloat16
                  && w.dim() == 3 && w.size(2) % bc == 0 && w.size(2) >= bc,
              "the ladder's coefficients are a contiguous bf16 [BH, L, N] plane whose "
              "slot space is a whole number of blocks: at this rung the slot panel is "
              "the identity and there are no digits");
  TORCH_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == torch::kBFloat16
                  && v.dim() == 4 && v.size(3) == kD,
              "V must be a contiguous bf16 [B, L, H, 64] tensor");
  TORCH_CHECK(dnum.is_cuda() && dnum.is_contiguous() && dnum.scalar_type() == torch::kBFloat16
                  && dnum.dim() == 3 && dnum.size(2) == kD,
              "dnum must be a contiguous bf16 [BH, L, 64] tensor");
  TORCH_CHECK(dden.is_cuda() && dden.is_contiguous() && dden.scalar_type() == torch::kBFloat16
                  && dden.dim() == 2,
              "dden must be a contiguous bf16 [BH, L] tensor: it is the readout's mass "
              "seed and the ones column of the transposed staircase");
  TORCH_CHECK(w.size(0) == v.size(0) * v.size(2) && w.size(0) == dnum.size(0)
                  && w.size(0) == dden.size(0) && w.size(1) == v.size(1)
                  && w.size(1) == dnum.size(1) && w.size(1) == dden.size(1),
              "the coefficient, value and seed planes must agree on (BH, L)");
  //: RAGGED T IS A LATER RUNG.  A partial final chunk is refused rather than
  //: masked: the mask is rung b6's, and a fallback here would be the special case
  //: that rung exists to delete.
  TORCH_CHECK(w.size(1) % chunk == 0,
              "the ladder's chunks are contiguous and whole: L must be a multiple of C "
              "(the ragged tail is rung b6)");
  //: THE GATHER BOUND, and it is the SLOT plane's now rather than the value plane's:
  //: every coefficient and per-slot-gradient address is a 32-bit displacement from a
  //: block's own base, and the widest row is the fp32 gradient's `N * 4` bytes.
  TORCH_CHECK(w.size(1) * w.size(2) * (int64_t)sizeof(float) < (int64_t)1 << 32,
              "L * N overruns the 32-bit gather displacement the addressing law is "
              "written on: L=",
              w.size(1), " N=", w.size(2));
}

std::vector<torch::Tensor> rung0_pass1(torch::Tensor w, torch::Tensor v, torch::Tensor dnum,
                                       torch::Tensor dden, int64_t chunk, int64_t bc) {
  check_arch_table();
  check_operands(w, v, dnum, dden, chunk, bc);
  const int BH = (int)w.size(0), L = (int)w.size(1), N = (int)w.size(2);
  //: `dR` is SLOT-DISJOINT over the block lattice -- a block owns its own columns and
  //: writes every one of them exactly once -- so it is allocated, never filled.
  auto dr = torch::empty({(long)BH, (long)L, (long)N}, w.options().dtype(torch::kFloat32));
#define R0_PASS1_X(C_, BC_)                                                                    \
  if (chunk == C_ && bc == BC_) {                                                              \
    Rung0Params p{nullptr,                                                                     \
                  reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),                        \
                  reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),                        \
                  reinterpret_cast<const __nv_bfloat16*>(dnum.data_ptr()),                     \
                  reinterpret_cast<const __nv_bfloat16*>(dden.data_ptr()),                     \
                  dr.data_ptr<float>(),                                                        \
                  nullptr,                                                                     \
                  nullptr,                                                                     \
                  BH,                                                                          \
                  L,                                                                           \
                  N,                                                                           \
                  (int)(v.size(2) * kD * (long)sizeof(__nv_bfloat16))};                        \
    C10_CUDA_CHECK((launch_rung0_pass1<C_, BC_>(p, at::cuda::getCurrentCUDAStream())));         \
    return {dr};                                                                                \
  }
  LADDER_R0_ARMS_X(R0_PASS1_X)
#undef R0_PASS1_X
  TORCH_CHECK(false, "the backward ladder is not built for chunk=", chunk, " BC=", bc);
}

std::vector<torch::Tensor> rung0_pass2(torch::Tensor r, torch::Tensor w, torch::Tensor v,
                                       torch::Tensor dnum, torch::Tensor dden, int64_t chunk,
                                       int64_t bc) {
  check_arch_table();
  check_operands(w, v, dnum, dden, chunk, bc);
  TORCH_CHECK(r.sizes() == w.sizes() && r.is_contiguous() && r.scalar_type() == torch::kBFloat16,
              "the read coefficients share the write side's plane shape");
  const int BH = (int)w.size(0), L = (int)w.size(1), N = (int)w.size(2);
  auto opts = w.options().dtype(torch::kFloat32);
  //: THE ONE FAN-IN TARGET, and the ONE fill.  `dv` contracts over the whole slot
  //: space, so every block holds a partial and the arena carries the identity BEFORE
  //: the first contribution -- one allocation-wide fill, never one per output.
  auto dv = torch::zeros({(long)BH, (long)L, (long)kD}, opts);
  auto dwo = torch::empty({(long)BH, (long)L, (long)N}, opts);
#define R0_PASS2_X(C_, BC_)                                                                    \
  if (chunk == C_ && bc == BC_) {                                                              \
    Rung0Params p{reinterpret_cast<const __nv_bfloat16*>(r.data_ptr()),                        \
                  reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),                        \
                  reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),                        \
                  reinterpret_cast<const __nv_bfloat16*>(dnum.data_ptr()),                     \
                  reinterpret_cast<const __nv_bfloat16*>(dden.data_ptr()),                     \
                  nullptr,                                                                     \
                  dv.data_ptr<float>(),                                                        \
                  dwo.data_ptr<float>(),                                                       \
                  BH,                                                                          \
                  L,                                                                           \
                  N,                                                                           \
                  (int)(v.size(2) * kD * (long)sizeof(__nv_bfloat16))};                        \
    C10_CUDA_CHECK((launch_rung0_pass2<C_, BC_>(p, at::cuda::getCurrentCUDAStream())));         \
    return {dv, dwo};                                                                           \
  }
  LADDER_R0_ARMS_X(R0_PASS2_X)
#undef R0_PASS2_X
  TORCH_CHECK(false, "the backward ladder is not built for chunk=", chunk, " BC=", bc);
}

std::vector<std::vector<int64_t>> rung0_census() {
  check_arch_table();
  std::vector<std::vector<int64_t>> out;
#define R0_CENSUS_X(C_, BC_)                                                              \
  {                                                                                       \
    std::vector<int64_t> p1{C_, BC_, 1};                                                  \
    for (auto x : rung0_census_arm<C_, BC_, false>(rung0_pass1_kernel<C_, BC_>))           \
      p1.push_back(x);                                                                    \
    out.push_back(p1);                                                                    \
    std::vector<int64_t> p2{C_, BC_, 2};                                                  \
    for (auto x : rung0_census_arm<C_, BC_, true>(rung0_pass2_kernel<C_, BC_>))            \
      p2.push_back(x);                                                                    \
    out.push_back(p2);                                                                    \
  }
  LADDER_R0_ARMS_X(R0_CENSUS_X)
#undef R0_CENSUS_X
  return out;
}

}  // namespace rola::chunk
