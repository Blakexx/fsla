// csrc/rola/src/chunk/backward.cu -- the reverse pass' translation unit: the arm
// surface, the census query and the torch entries.  A TU OF ITS OWN, on
// `factor.cu`'s rule -- these instantiations must not enter a shard whose per-`cicc`
// footprint the build's job formula was measured on (docs/build.md).
// docs/internals/chunk/backward.md
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <array>
#include <vector>

#include "../instantiations/scan_arms.inc"
#include "backward.cuh"

namespace rola::chunk {

//: THE CLOSED-WORLD RUNTIME REFUSAL, defined once in `chunk.cu`: every entry
//: that reaches a kernel runs it first, and it has ONE definition so a second
//: translation unit cannot drift from the capability table it checks.
void check_arch_table();

template <int C, int BC, int D, int B>
static cudaError_t launch_scan_b_arm(const ScanBParams& p, cudaStream_t stream) {
  constexpr size_t sm = GeomB<C, BC, D, B>::bytes;
  static bool configured = false;
  if (!configured) {
    const cudaError_t e = cudaFuncSetAttribute(
        scan_b_kernel<C, D, B, BC>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    if (e != cudaSuccess) return e;
    configured = true;
  }
  const dim3 grid((unsigned)LevelPlan<D, B, BC>::blocks, (unsigned)p.BH);
  scan_b_kernel<C, D, B, BC>
      <<<grid, threads_for(BC), sm, stream>>>(p.pr, p.pw, p.gw, p.v, p.dnum, p.dden, p.tbl, p.dv,
                                              p.dwsum, p.dgw, p.dpw, p.L, p.v_row_bytes);
  return cudaGetLastError();
}

//: THE CENSUS ROW CARRIES BOTH HALVES OF THE FLOOR CLAIM -- the DECLARED live set
//: and the BUDGET the target admits.  A row whose measured `numRegs` exceeds the
//: budget is a spill waiting -- docs/internals/chunk/backward.md#reg-floor-scan
template <int C, int BC, int D, int B>
static std::array<int64_t, 8> scan_b_census_arm() {
  cudaFuncAttributes a{};
  if (cudaFuncGetAttributes(&a, scan_b_kernel<C, D, B, BC>) != cudaSuccess)
    return {-1, -1, -1, -1, -1, -1, -1, -1};
  return {a.numRegs,
          (int64_t)a.localSizeBytes,
          (int64_t)GeomB<C, BC, D, B>::bytes,
          (int64_t)GeomB<C, BC, D, B>::stages,
          (int64_t)kScanBCtaTarget<C, BC, D, B>,
          (int64_t)reg_floor_scan(BC),
          (int64_t)denseref::arch::reg_budget(denseref::arch::kCaps, threads_for(BC),
                                              kScanBCtaTarget<C, BC, D, B>),
          (int64_t)LevelPlan<D, B, BC>::entries};
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
  const int BH = B_ * H;
  auto opts = v.options().dtype(torch::kFloat32);
  //: BOTH OUTPUTS ARE FAN-IN TARGETS, so the identity of the sum has to be there
  //: before the first block contributes.
  auto dv = torch::zeros({B_, H, (long)L, (long)kD}, opts);
  //: NOT `d gw`: this is `sum_s dW[j,s]`, unweighted, which has d gw's shape and
  //: not its value -- docs/internals/chunk/backward.md#dw-slotsum
  auto dwsum = torch::zeros({B_, H, (long)L}, opts);
  auto dgw = torch::zeros({B_, H, (long)L}, opts);
  //: the write-side DIGIT gradients, one column per level-digit of the packed
  //: amplitude row -- a fan-in destination, so it starts at the identity.
  auto dpw = torch::zeros({B_, H, (long)L, pw.size(-1)}, opts);
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
                   dwsum.data_ptr<float>(),                                                    \
                   dgw.data_ptr<float>(),                                                      \
                   dpw.data_ptr<float>(),                                                      \
                   BH,                                                                         \
                   L,                                                                          \
                   H * kD * (int)sizeof(__nv_bfloat16)};                                       \
    C10_CUDA_CHECK((launch_scan_b_arm<C_, BC_, D_, B__>(p_, at::cuda::getCurrentCUDAStream()))); \
    return {dv, dwsum, dgw, dpw};                                                                          \
  }
  SCAN_B_ARMS_X(SCAN_B_DISPATCH_X)
#undef SCAN_B_DISPATCH_X
  TORCH_CHECK(false, "Scan B is not built for arm chunk=", chunk, " BC=", bc,
              " depth=", widths.size(),
              ": no scan is built for it (rola/ops/chunk.py's scan_arms()).");
}

//: THE SEEDS, IN ONE ELEMENTWISE PASS -- the mirror of `divide_pass` and the ONE
//: place the eps floor is differentiated.  The three closed forms, the dtypes and
//: why `dm` is not zero: docs/internals/chunk/backward.md#seeds
__device__ __forceinline__ float as_float(const __nv_bfloat16& x) { return __bfloat162float(x); }
__device__ __forceinline__ float as_float(const float& x) { return x; }

template <typename InT>
__global__ void seed_kernel(const InT* __restrict__ dy, const InT* __restrict__ y,
                            const float* __restrict__ den, const float* __restrict__ mass,
                            __nv_bfloat16* __restrict__ dnum, __nv_bfloat16* __restrict__ dden,
                            float* __restrict__ dm, long tokens, int L, int H, float eps) {
  constexpr int kPerLane = kD / 32;
  const long tok = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (tok >= tokens) return;
  const int lane = threadIdx.x & 31;
  const long bh = tok / L, t = tok % L;
  //: `dy` and `y` arrive in the OPERAND layout `[B, T, H, d_v]` the forward emits;
  //: `dnum` leaves in the fold's `[BH, T, d_v]`, so the transpose is these indices.
  const long src = (((bh / H) * L + t) * H + (bh % H)) * kD;
  const float d = den[tok] + eps * (mass != nullptr ? mass[tok] : 1.0f);
  float acc = 0.f;
#pragma unroll
  for (int e = 0; e < kPerLane; ++e) {
    const int c = lane + e * 32;
    const float g = as_float(dy[src + c]);
    acc += g * as_float(y[src + c]);
    dnum[tok * kD + c] = __float2bfloat16_rn(__fdiv_rn(g, d));
  }
#pragma unroll
  for (int b = 1; b < 32; b <<= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, b);
  //: (A.2)'s STABLE FORM: `y` is the op's own output, so the denominator is
  //: divided by once instead of squared.
  const float dd = -__fdiv_rn(acc, d);
  if (lane == 0) {
    dden[tok] = __float2bfloat16_rn(dd);
    //: (A.3). Taken off the fp32 value rather than off the bf16 one it was just
    //: rounded to: `dm` is the floor's own gradient and inherits its dtype.
    if (dm != nullptr) dm[tok] = eps * dd;
  }
}

std::vector<torch::Tensor> backward_seeds(torch::Tensor dy, torch::Tensor y, torch::Tensor den,
                                          c10::optional<torch::Tensor> mass, double eps) {
  check_arch_table();
  TORCH_CHECK(dy.is_cuda() && dy.is_contiguous() && dy.dim() == 4 && dy.size(3) == kD,
              "the output cotangent is a contiguous [B, T, H, 64] tensor in y's layout");
  TORCH_CHECK(y.sizes() == dy.sizes() && y.scalar_type() == dy.scalar_type() && y.is_contiguous(),
              "y must be the op's own output: same layout, same dtype as its cotangent");
  const long B = dy.size(0), L = dy.size(1), H = dy.size(2), tokens = B * H * L;
  TORCH_CHECK(den.is_cuda() && den.is_contiguous() && den.scalar_type() == torch::kFloat32
                  && den.numel() == tokens,
              "the denominator is the contiguous fp32 [B, H, T] plane the forward emitted");
  const float* massPtr = nullptr;
  if (mass.has_value()) {
    TORCH_CHECK(mass->is_cuda() && mass->is_contiguous() && mass->scalar_type() == torch::kFloat32
                    && mass->numel() == tokens,
                "the read mass must be a contiguous fp32 [B, H, T] tensor");
    massPtr = mass->data_ptr<float>();
  }
  auto opts = dy.options();
  auto dnum = torch::empty({B, H, L, (long)kD}, opts.dtype(torch::kBFloat16));
  auto dden = torch::empty({B, H, L}, opts.dtype(torch::kBFloat16));
  //: THE FIFTH OUTPUT EXISTS ONLY WHEN THE MASS DOES. `m` is not an input of a
  //: declared-simplex bundle, and returning a zero for it would be a claim.
  auto dmass =
      massPtr != nullptr ? torch::empty({B, H, L}, opts.dtype(torch::kFloat32)) : torch::Tensor();
  constexpr int kThreads = 256;
  const int blocks = (int)((tokens + (kThreads / 32) - 1) / (kThreads / 32));
  const auto stream = at::cuda::getCurrentCUDAStream();
#define SEED_LAUNCH(T_, cast_)                                                        \
  seed_kernel<T_><<<blocks, kThreads, 0, stream>>>(                                   \
      cast_(dy.data_ptr()), cast_(y.data_ptr()), den.data_ptr<float>(), massPtr,      \
      reinterpret_cast<__nv_bfloat16*>(dnum.data_ptr()),                              \
      reinterpret_cast<__nv_bfloat16*>(dden.data_ptr()),                              \
      dmass.defined() ? dmass.data_ptr<float>() : nullptr, tokens, (int)L, (int)H,    \
      (float)eps)
  if (dy.scalar_type() == torch::kFloat32)
    SEED_LAUNCH(float, reinterpret_cast<const float*>);
  else if (dy.scalar_type() == torch::kBFloat16)
    SEED_LAUNCH(__nv_bfloat16, reinterpret_cast<const __nv_bfloat16*>);
  else
    TORCH_CHECK(false, "the cotangent arrives in the readout's dtype: fp32 or bf16");
#undef SEED_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (dmass.defined()) return {dnum, dden, dmass};
  return {dnum, dden};
}

//: SCAN A'S ENTRY.  It emits `d pr` alone: the read side's digit gradient is the
//: only thing the forward rebuild is for, and every write-side gradient is Scan B's.
std::vector<torch::Tensor> scan_a_probe(torch::Tensor pr, torch::Tensor pw, torch::Tensor gw,
                                        torch::Tensor v, torch::Tensor dnum, torch::Tensor dden,
                                        torch::Tensor tbl, int64_t chunk, int64_t bc,
                                        std::vector<int64_t> widths) {
  check_arch_table();
  TORCH_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == torch::kBFloat16
                  && v.dim() == 4 && v.size(3) == kD,
              "V must be a contiguous bf16 [B,L,H,64] tensor");
  TORCH_CHECK(dnum.is_cuda() && dnum.is_contiguous() && dnum.scalar_type() == torch::kBFloat16
                  && dnum.dim() == 4 && dnum.size(3) == kD,
              "dnum must be a contiguous bf16 [B,H,L,64] tensor");
  TORCH_CHECK(dden.is_cuda() && dden.is_contiguous() && dden.scalar_type() == torch::kBFloat16,
              "dden must be a contiguous bf16 [B,H,L] tensor");
  TORCH_CHECK(tbl.is_cuda() && tbl.is_contiguous() && tbl.scalar_type() == torch::kInt32,
              "the union table is a contiguous int32 tensor of packed bits");
  const int B_ = (int)v.size(0), L = (int)v.size(1), H = (int)v.size(2);
  const int BH = B_ * H;
  auto opts = v.options().dtype(torch::kFloat32);
  auto dpr = torch::zeros({B_, H, (long)L, pr.size(-1)}, opts);
#define SCAN_A_DISPATCH_X(C_, BC_, D_, B__)                                                       \
  if (chunk == C_ && bc == BC_ && is_arm(widths, D_, B__)) {                                      \
    const long want = (long)BH * 2 * LevelPlan<D_, B__, BC_>::entries * tok_words(L);             \
    TORCH_CHECK(tbl.numel() == want, "the union table was built for a different arm");            \
    constexpr size_t sm = GeomB<C_, BC_, D_, B__>::bytes;                                                  \
    static bool configured = false;                                                               \
    if (!configured) {                                                                            \
      C10_CUDA_CHECK(cudaFuncSetAttribute(scan_a_kernel<C_, D_, B__, BC_>,                        \
                                          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm)); \
      configured = true;                                                                          \
    }                                                                                             \
    const dim3 grid((unsigned)LevelPlan<D_, B__, BC_>::blocks, (unsigned)BH);                     \
    scan_a_kernel<C_, D_, B__, BC_><<<grid, threads_for(BC_), sm,                                 \
                                      at::cuda::getCurrentCUDAStream()>>>(                        \
        reinterpret_cast<const __nv_bfloat16*>(pr.data_ptr()),                                    \
        reinterpret_cast<const __nv_bfloat16*>(pw.data_ptr()),                                    \
        reinterpret_cast<const __nv_bfloat16*>(gw.data_ptr()),                                    \
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),                                     \
        reinterpret_cast<const __nv_bfloat16*>(dnum.data_ptr()),                                  \
        reinterpret_cast<const __nv_bfloat16*>(dden.data_ptr()),                                  \
        reinterpret_cast<const uint32_t*>(tbl.data_ptr()), dpr.data_ptr<float>(), L,              \
        H * kD * (int)sizeof(__nv_bfloat16));                                                     \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                                               \
    return {dpr};                                                                                 \
  }
  SCAN_B_ARMS_X(SCAN_A_DISPATCH_X)
#undef SCAN_A_DISPATCH_X
  TORCH_CHECK(false, "Scan A is not built for arm chunk=", chunk, " BC=", bc);
}

//: THE PARTITION PROBE'S ENTRY.  It takes the table and nothing else: the walk
//: reads no amplitude, so a probe that took one would gate a wider surface than
//: the schedule -- docs/internals/chunk/backward.md#partition-probe
std::vector<torch::Tensor> walk_probe(torch::Tensor tbl, int64_t chunk, int64_t bc,
                                      std::vector<int64_t> widths, int64_t L, int64_t BH) {
  check_arch_table();
  TORCH_CHECK(tbl.is_cuda() && tbl.is_contiguous() && tbl.scalar_type() == torch::kInt32,
              "the union table is a contiguous int32 tensor of packed bits");
  TORCH_CHECK(L > 0, "L is the caller's token count, and it is positive");
  //: THE BATCH COMES FROM THE CALLER, not from the table's extent: a plan whose
  //: blocks span the whole slot space builds NO table rows, so a zero-element
  //: table is legal and carries no batch to divide out.
  TORCH_CHECK(BH > 0, "the head-batch is the caller's, and it is positive");
  auto opts = torch::TensorOptions().dtype(torch::kInt32).device(tbl.device());
#define WALK_PROBE_DISPATCH_X(C_, BC_, D_, B__)                                             \
  if (chunk == C_ && bc == BC_ && is_arm(widths, D_, B__)) {                                \
    using LP = LevelPlan<D_, B__, BC_>;                                                     \
    const long per = 2L * LP::entries * tok_words((int)L);                                  \
    TORCH_CHECK(tbl.numel() == BH * per, "the union table was built for a different arm"); \
    auto stampR = torch::full({BH, (long)LP::blocks, (long)L}, -1, opts);                   \
    auto stampW = torch::full({BH, (long)LP::blocks, (long)L}, -1, opts);                   \
    auto jout = torch::full({BH, (long)LP::blocks, (long)L}, -1, opts);                     \
    auto nchunk = torch::zeros({BH, (long)LP::blocks}, opts);                               \
    constexpr size_t sm = (size_t)GeomB<C_, BC_, D_, B__>::idx_slot                                  \
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
