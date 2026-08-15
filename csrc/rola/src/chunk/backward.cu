// csrc/rola/src/chunk/backward.cu -- SCAN B's translation unit: the arm
// surface, the census query and the torch entry.  A TU OF ITS OWN, on
// `factor.cu`'s rule: the backward's instantiations must not enter a shard whose
// per-`cicc` footprint the build's job formula was measured on
// (docs/build.md, tools/build_flags.py).  What Scan B computes at this stage and
// what it does not: docs/internals/chunk/backward.md
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <array>
#include <vector>

#include "backward.cuh"

namespace rola::chunk {

//: THE CLOSED-WORLD RUNTIME REFUSAL, defined once in `chunk.cu`: every entry
//: that reaches a kernel runs it first, and it has ONE definition so a second
//: translation unit cannot drift from the capability table it checks.
void check_arch_table();

//: THE ARMS SCAN B IS BUILT FOR.  One, deliberately: the pinned production arm
//: (`rola/ops/chunk.py`'s `_PIN_C`/`_PIN_BC` at depth two, width 64).  The
//: forward's arm cross is not owed here until the scan carries every term.
#define SCAN_B_ARMS_X(X) X(32, 128, 2, 64)

template <int C, int BC, int D, int B>
static cudaError_t launch_scan_b_arm(const ScanBParams& p, cudaStream_t stream) {
  constexpr size_t sm = GeomB<C, BC>::bytes;
  static bool configured = false;
  if (!configured) {
    const cudaError_t e = cudaFuncSetAttribute(
        scan_b_kernel<C, D, B, BC>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    if (e != cudaSuccess) return e;
    configured = true;
  }
  const dim3 grid((unsigned)LevelPlan<D, B, BC>::blocks, (unsigned)p.BH);
  scan_b_kernel<C, D, B, BC><<<grid, threads_for(BC), sm, stream>>>(
      p.pr, p.pw, p.gw, p.v, p.dnum, p.dden, p.tbl, p.dv, p.dgw, p.L, p.v_row_bytes);
  return cudaGetLastError();
}

template <int C, int BC, int D, int B>
static std::array<int64_t, 6> scan_b_census_arm() {
  cudaFuncAttributes a{};
  if (cudaFuncGetAttributes(&a, scan_b_kernel<C, D, B, BC>) != cudaSuccess)
    return {-1, -1, -1, -1, -1, -1};
  return {a.numRegs,
          (int64_t)a.localSizeBytes,
          (int64_t)GeomB<C, BC>::bytes,
          (int64_t)GeomB<C, BC>::stages,
          (int64_t)kScanBCtaTarget<C, BC>,
          (int64_t)reg_floor(C, BC)};
}

static bool is_arm(const std::vector<int64_t>& widths, int64_t D, int64_t B) {
  if ((int64_t)widths.size() != D) return false;
  for (auto w : widths)
    if (w != B) return false;
  return true;
}

std::vector<torch::Tensor> scan_b_probe(torch::Tensor pr, torch::Tensor pw, torch::Tensor gw,
                                        torch::Tensor v, torch::Tensor dnum, torch::Tensor dden,
                                        torch::Tensor tbl, int64_t chunk, int64_t bc,
                                        std::vector<int64_t> widths) {
  check_arch_table();
  TORCH_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == torch::kBFloat16
                  && v.dim() == 4 && v.size(3) == kD,
              "V must be a contiguous bf16 [B,L,H,64] tensor");
  TORCH_CHECK(pr.is_contiguous() && pw.is_contiguous() && gw.is_contiguous()
                  && pr.scalar_type() == torch::kBFloat16 && pw.scalar_type() == torch::kBFloat16
                  && gw.scalar_type() == torch::kBFloat16,
              "the amplitudes and the write gain are contiguous bf16");
  TORCH_CHECK(dnum.is_cuda() && dnum.is_contiguous() && dnum.scalar_type() == torch::kBFloat16
                  && dnum.dim() == 4 && dnum.size(3) == kD,
              "dnum must be a contiguous bf16 [B,H,L,64] tensor (the fold's A operand)");
  TORCH_CHECK(dden.is_cuda() && dden.is_contiguous() && dden.scalar_type() == torch::kBFloat16
                  && dden.dim() == 3,
              "dden must be a contiguous bf16 [B,H,L] tensor (the mass column's A row)");
  TORCH_CHECK(tbl.is_cuda() && tbl.is_contiguous() && tbl.scalar_type() == torch::kInt32,
              "the union table is a contiguous int32 tensor of packed bits");
  const int B_ = (int)v.size(0), L = (int)v.size(1), H = (int)v.size(2);
  TORCH_CHECK(L % chunk == 0, "L must be a multiple of the chunk");
  const int BH = B_ * H;
  auto opts = v.options().dtype(torch::kFloat32);
  //: BOTH OUTPUTS ARE FAN-IN TARGETS, so the identity of the sum has to be there
  //: before the first block contributes.
  auto dv = torch::zeros({B_, H, (long)L, (long)kD}, opts);
  auto dgw = torch::zeros({B_, H, (long)L}, opts);
#define SCAN_B_DISPATCH_X(C_, BC_, D_, B__)                                                    \
  if (chunk == C_ && bc == BC_ && is_arm(widths, D_, B__)) {                                   \
    const long want = (long)BH * 2 * LevelPlan<D_, B__, BC_>::entries * tok_words(L);          \
    TORCH_CHECK(tbl.numel() == want, "the union table was built for a different arm");         \
    ScanBParams p_{reinterpret_cast<const __nv_bfloat16*>(pr.data_ptr()),                      \
                   reinterpret_cast<const __nv_bfloat16*>(pw.data_ptr()),                      \
                   reinterpret_cast<const __nv_bfloat16*>(gw.data_ptr()),                      \
                   reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),                       \
                   reinterpret_cast<const __nv_bfloat16*>(dnum.data_ptr()),                    \
                   reinterpret_cast<const __nv_bfloat16*>(dden.data_ptr()),                    \
                   reinterpret_cast<const uint32_t*>(tbl.data_ptr()),                          \
                   dv.data_ptr<float>(),                                                       \
                   dgw.data_ptr<float>(),                                                      \
                   BH,                                                                         \
                   L,                                                                          \
                   H * kD * (int)sizeof(__nv_bfloat16)};                                       \
    C10_CUDA_CHECK((launch_scan_b_arm<C_, BC_, D_, B__>(p_, at::cuda::getCurrentCUDAStream()))); \
    return {dv, dgw};                                                                          \
  }
  SCAN_B_ARMS_X(SCAN_B_DISPATCH_X)
#undef SCAN_B_DISPATCH_X
  TORCH_CHECK(false, "Scan B is not built for arm chunk=", chunk, " BC=", bc,
              " depth=", widths.size(),
              ": the backward carries the pinned arm alone until the scan is whole.");
}

//: THE PARTITION PROBE'S ENTRY.  It takes the table and nothing else: the walk
//: reads no amplitude, so a probe that took one would be gating a wider surface
//: than the schedule.  Three int32 planes `[BH, blocks, L]` come back, `-1` where
//: no chunk gathered the token.
std::vector<torch::Tensor> walk_probe(torch::Tensor tbl, int64_t chunk, int64_t bc,
                                      std::vector<int64_t> widths, int64_t L) {
  check_arch_table();
  TORCH_CHECK(tbl.is_cuda() && tbl.is_contiguous() && tbl.scalar_type() == torch::kInt32,
              "the union table is a contiguous int32 tensor of packed bits");
  TORCH_CHECK(L > 0 && L % chunk == 0, "L must be a positive multiple of the chunk");
  auto opts = torch::TensorOptions().dtype(torch::kInt32).device(tbl.device());
#define WALK_PROBE_DISPATCH_X(C_, BC_, D_, B__)                                             \
  if (chunk == C_ && bc == BC_ && is_arm(widths, D_, B__)) {                                \
    using LP = LevelPlan<D_, B__, BC_>;                                                     \
    const long per = 2L * LP::entries * tok_words((int)L);                                  \
    TORCH_CHECK(tbl.numel() % per == 0, "the union table was built for a different arm");   \
    const long BH = tbl.numel() / per;                                                      \
    auto stampR = torch::full({BH, (long)LP::blocks, (long)L}, -1, opts);                   \
    auto stampW = torch::full({BH, (long)LP::blocks, (long)L}, -1, opts);                   \
    auto jout = torch::full({BH, (long)LP::blocks, (long)L}, -1, opts);                     \
    auto nchunk = torch::zeros({BH, (long)LP::blocks}, opts);                               \
    constexpr size_t sm = (size_t)GeomB<C_, BC_>::idx_slot                                  \
                          + 2 * kFormWords * sizeof(uint32_t) + kCookSlots * sizeof(int);   \
    const dim3 grid((unsigned)LP::blocks, (unsigned)BH);                                    \
    walk_probe_kernel<C_, D_, B__, BC_><<<grid, threads_for(BC_), sm,                       \
                                          at::cuda::getCurrentCUDAStream()>>>(              \
        reinterpret_cast<const uint32_t*>(tbl.data_ptr()), stampR.data_ptr<int>(),          \
        stampW.data_ptr<int>(), jout.data_ptr<int>(), nchunk.data_ptr<int>(), (int)L);      \
    C10_CUDA_CHECK(cudaGetLastError());                                                     \
    return {stampR, stampW, jout, nchunk};                                                  \
  }
  SCAN_B_ARMS_X(WALK_PROBE_DISPATCH_X)
#undef WALK_PROBE_DISPATCH_X
  TORCH_CHECK(false, "the descending walk is not built for arm chunk=", chunk, " BC=", bc,
              " depth=", widths.size());
}

std::vector<std::vector<int64_t>> scan_b_census() {
  check_arch_table();
  std::vector<std::vector<int64_t>> out;
#define SCAN_B_CENSUS_X(C_, BC_, D_, B__)                          \
  {                                                                \
    std::vector<int64_t> row{C_, BC_, D_, B__};                    \
    for (auto x : scan_b_census_arm<C_, BC_, D_, B__>()) row.push_back(x); \
    out.push_back(row);                                            \
  }
  SCAN_B_ARMS_X(SCAN_B_CENSUS_X)
#undef SCAN_B_CENSUS_X
  return out;
}

}  // namespace rola::chunk
