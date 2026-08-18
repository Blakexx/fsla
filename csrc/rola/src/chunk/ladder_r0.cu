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

template <int C, int BC, int D, int Bw>
static cudaError_t launch_rung0_pass1(const Rung0Params& p, cudaStream_t stream) {
  constexpr size_t sm = GeomR0<C, BC, false>::bytes;
  static bool configured = false;
  if (!configured) {
    const cudaError_t e = cudaFuncSetAttribute(
        rung0_pass1_kernel<C, BC, D, Bw>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    if (e != cudaSuccess) return e;
    configured = true;
  }
  //: THE LATTICE IS COMPILED at rung b3: `N = Bw^D` is the arm's own geometry and
  //: not a plane width the caller hands in.
  const dim3 grid((unsigned)(DigitR0<D, Bw, BC, GeomR0<C, BC, false>::BCg>::slots / BC),
                  (unsigned)p.BH);
  rung0_pass1_kernel<C, BC, D, Bw><<<grid, r0_threads_for(BC), sm, stream>>>(
      p.pr, p.pw, p.v, p.dnum, p.dden, p.dpr, p.L, p.v_row_bytes);
  return cudaGetLastError();
}

template <int C, int BC, int D, int Bw>
static cudaError_t launch_rung0_pass2(const Rung0Params& p, cudaStream_t stream) {
  constexpr size_t sm = GeomR0<C, BC, true>::bytes;
  static bool configured = false;
  if (!configured) {
    const cudaError_t e = cudaFuncSetAttribute(
        rung0_pass2_kernel<C, BC, D, Bw>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    if (e != cudaSuccess) return e;
    configured = true;
  }
  const dim3 grid((unsigned)(DigitR0<D, Bw, BC, GeomR0<C, BC, true>::BCg>::slots / BC),
                  (unsigned)p.BH);
  rung0_pass2_kernel<C, BC, D, Bw><<<grid, r0_threads_for(BC), sm, stream>>>(
      p.pr, p.pw, p.v, p.dnum, p.dden, p.dv, p.dpw, p.L, p.v_row_bytes);
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
          //: THE THREAD COUNT.  The CTA target alone stopped being the residency
          //: statement when the block gained groups: what the carve holds constant is
          //: WARPS PER SM, and at `BC = 128` that is one CTA of eight where `BC = 64`
          //: runs two of four -- docs/internals/chunk/ladder_r0.md#the-group-threshold
          (int64_t)r0_threads_for(BC)};
}

static void check_operands(const torch::Tensor& w, const torch::Tensor& v,
                           const torch::Tensor& dnum, const torch::Tensor& dden, int64_t chunk,
                           int64_t depth, int64_t width) {
  TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == torch::kBFloat16
                  && w.dim() == 3 && w.size(2) == depth * width,
              "the ladder's coefficients are PER LEVEL at rung b3: a contiguous bf16 "
              "[BH, L, D*B] amplitude plane, one packed block of B columns per level. "
              "Got a last dimension of ",
              w.size(2), " where D*B = ", depth * width);
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
  //: THE GATHER BOUND, on the AMPLITUDE plane now: every amplitude and digit-gradient
  //: address is a 32-bit displacement from a batch-head's own base, and the widest row
  //: is the fp32 digit gradient's `D * B * 4` bytes. The per-slot planes it used to be
  //: taken against are gone, which is why this bound got `N / (D*B)` times looser.
  TORCH_CHECK(w.size(1) * w.size(2) * (int64_t)sizeof(float) < (int64_t)1 << 32,
              "L * D * B overruns the 32-bit gather displacement the addressing law is "
              "written on: L=",
              w.size(1), " D*B=", w.size(2));
}

std::vector<torch::Tensor> rung0_pass1(torch::Tensor pr, torch::Tensor pw, torch::Tensor v,
                                       torch::Tensor dnum, torch::Tensor dden, int64_t chunk,
                                       int64_t bc, int64_t depth, int64_t width) {
  check_arch_table();
  check_operands(pw, v, dnum, dden, chunk, depth, width);
  TORCH_CHECK(
      pr.sizes() == pw.sizes() && pr.is_contiguous() && pr.scalar_type() == torch::kBFloat16,
      "the read amplitudes share the write side's plane shape");
  const int BH = (int)pw.size(0), L = (int)pw.size(1), P = (int)pw.size(2);
  //: THE READ SIDE'S DIGIT GRADIENT IS A FAN-IN NOW, and that is rung b3's own
  //: change: a DIGIT is shared across the block lattice where a SLOT was not, so
  //: pass 1 acquires its first `RED` and its arena needs the identity before the
  //: first contribution -- one allocation-wide fill, never one per contributing
  //: block -- docs/internals/chunk/ladder_r0.md#fan-in
  auto dpr = torch::zeros({(long)BH, (long)L, (long)P}, pw.options().dtype(torch::kFloat32));
#define R0_PASS1_X(C_, BC_, D_, B_)                                                            \
  if (chunk == C_ && bc == BC_ && depth == D_ && width == B_) {                                \
    Rung0Params p{reinterpret_cast<const __nv_bfloat16*>(pr.data_ptr()),                       \
                  reinterpret_cast<const __nv_bfloat16*>(pw.data_ptr()),                       \
                  reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),                        \
                  reinterpret_cast<const __nv_bfloat16*>(dnum.data_ptr()),                     \
                  reinterpret_cast<const __nv_bfloat16*>(dden.data_ptr()),                     \
                  dpr.data_ptr<float>(),                                                       \
                  nullptr,                                                                     \
                  nullptr,                                                                     \
                  BH,                                                                          \
                  L,                                                                           \
                  (int)(v.size(2) * kD * (long)sizeof(__nv_bfloat16))};                        \
    C10_CUDA_CHECK((launch_rung0_pass1<C_, BC_, D_, B_>(p, at::cuda::getCurrentCUDAStream()))); \
    return {dpr};                                                                              \
  }
  LADDER_R0_ARMS_X(R0_PASS1_X)
#undef R0_PASS1_X
  TORCH_CHECK(false, "the backward ladder is not built for chunk=", chunk, " BC=", bc, " D=", depth,
              " B=", width);
}

std::vector<torch::Tensor> rung0_pass2(torch::Tensor pr, torch::Tensor pw, torch::Tensor v,
                                       torch::Tensor dnum, torch::Tensor dden, int64_t chunk,
                                       int64_t bc, int64_t depth, int64_t width) {
  check_arch_table();
  check_operands(pw, v, dnum, dden, chunk, depth, width);
  TORCH_CHECK(
      pr.sizes() == pw.sizes() && pr.is_contiguous() && pr.scalar_type() == torch::kBFloat16,
      "the read amplitudes share the write side's plane shape");
  const int BH = (int)pw.size(0), L = (int)pw.size(1), P = (int)pw.size(2);
  auto opts = pw.options().dtype(torch::kFloat32);
  //: BOTH of pass 2's outputs fan in.  `dv` contracts over the whole slot space and
  //: `d pw` over the digits a block's slots share with every other block's, so each
  //: carries the identity BEFORE the first contribution.
  auto dv = torch::zeros({(long)BH, (long)L, (long)kD}, opts);
  auto dpw = torch::zeros({(long)BH, (long)L, (long)P}, opts);
#define R0_PASS2_X(C_, BC_, D_, B_)                                                            \
  if (chunk == C_ && bc == BC_ && depth == D_ && width == B_) {                                \
    Rung0Params p{reinterpret_cast<const __nv_bfloat16*>(pr.data_ptr()),                       \
                  reinterpret_cast<const __nv_bfloat16*>(pw.data_ptr()),                       \
                  reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),                        \
                  reinterpret_cast<const __nv_bfloat16*>(dnum.data_ptr()),                     \
                  reinterpret_cast<const __nv_bfloat16*>(dden.data_ptr()),                     \
                  nullptr,                                                                     \
                  dv.data_ptr<float>(),                                                        \
                  dpw.data_ptr<float>(),                                                       \
                  BH,                                                                          \
                  L,                                                                           \
                  (int)(v.size(2) * kD * (long)sizeof(__nv_bfloat16))};                        \
    C10_CUDA_CHECK((launch_rung0_pass2<C_, BC_, D_, B_>(p, at::cuda::getCurrentCUDAStream()))); \
    return {dv, dpw};                                                                          \
  }
  LADDER_R0_ARMS_X(R0_PASS2_X)
#undef R0_PASS2_X
  TORCH_CHECK(false, "the backward ladder is not built for chunk=", chunk, " BC=", bc, " D=", depth,
              " B=", width);
}

std::vector<std::vector<int64_t>> rung0_census() {
  check_arch_table();
  std::vector<std::vector<int64_t>> out;
  //: THE ROW NAMES THE WHOLE ARM, `(C, BC, D, B, pass)`: rung b3 made the topology a
  //: compiled axis, so two rows that agree on `(C, BC)` and differ on `(D, B)` are
  //: different kernels and a census keyed on the block width alone would collapse them.
#define R0_CENSUS_X(C_, BC_, D_, B_)                                                            \
  {                                                                                             \
    std::vector<int64_t> p1{C_, BC_, D_, B_, 1};                                                \
    for (auto x : rung0_census_arm<C_, BC_, false>(rung0_pass1_kernel<C_, BC_, D_, B_>))        \
      p1.push_back(x);                                                                          \
    out.push_back(p1);                                                                          \
    std::vector<int64_t> p2{C_, BC_, D_, B_, 2};                                                \
    for (auto x : rung0_census_arm<C_, BC_, true>(rung0_pass2_kernel<C_, BC_, D_, B_>))         \
      p2.push_back(x);                                                                          \
    out.push_back(p2);                                                                          \
  }
  LADDER_R0_ARMS_X(R0_CENSUS_X)
#undef R0_CENSUS_X
  return out;
}

}  // namespace rola::chunk
