// csrc/rola/src/carry/backward.cu -- K31 R4: the reverse carry family's ONE translation unit.
//
// Dispatch, torch entries and the built arm list, mirroring `carry.cu`'s -- the two
// families answer to the same `(widths, k, m, W, C)` vocabulary because they walk the
// same window grid and rebuild the same schedule.
//
// THE TWO SCANS ARE LAUNCHED CONCURRENTLY, which is the whole content of the
// two-kernels resolution: no term couples the replayed state to the adjoint, so the
// two slabs are resident in the MACHINE at once and neither thread holds two.  Scan A
// takes a stream from the pool, Scan B keeps the caller's, and the two are joined by
// events before the entry returns.

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <array>

#include "backward_kernel.cuh"
#include "carry_api.cuh"
#include "facts/facts_kernel.cuh"

namespace rola::chunk {
template <int D, int B, int R0, int R1, int R2, int R3, int W0, int W1, int W2, int W3, int MASK>
cudaError_t launch_span_facts_arm(const __nv_bfloat16* pr, const __nv_bfloat16* pw, uint32_t* tbl,
                                  int BH, int L, cudaStream_t stream);
}  // namespace rola::chunk

namespace rola::carry {

//: THE BUILT SET, one macro so the dispatch, the census and the arm list cannot drift
//: apart.  The rows are `carry.cu`'s without the carve scalar: the backward's carve is
//: LEAF-split by derivation (the readouts contract over channels), not by argument.
#define BWD_ARMS_X(X_)       \
  X_(2, 256, 4, 8, 384, 32)  \
  X_(2, 256, 4, 8, 512, 32)  \
  X_(2, 64, 4, 8, 64, 32)

template <int D, int B, int K, int M>
using BwdOwnerRows =
    rola::chunk::SpanPlan<D, B, BoxPlan<D, B, K, M>::s(0), (D > 1 ? BoxPlan<D, B, K, M>::s(1) : B),
                          (D > 2 ? BoxPlan<D, B, K, M>::s(2) : B),
                          (D > 3 ? BoxPlan<D, B, K, M>::s(3) : B)>;

template <int D, int B, int K, int M, int W, int C>
static cudaError_t configure_smem() {
  static bool configured = false;
  if (configured) return cudaSuccess;
  using G = BwdGeom<D, B, K, M, W, C>;
  cudaError_t e =
      cudaFuncSetAttribute(&scan_a_kernel<D, B, K, M, W, C>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, (int)G::bytes_a);
  if (e != cudaSuccess) return e;
  e = cudaFuncSetAttribute(&scan_b_kernel<D, B, K, M, W, C>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, (int)G::bytes_b);
  if (e != cudaSuccess) return e;
  configured = true;
  return cudaSuccess;
}

template <int D, int B, int K, int M, int W, int C>
static cudaError_t launch_scans(const BwdParams& pa, const BwdParams& pb, int BH, cudaStream_t sa,
                                cudaStream_t sb) {
  using G = BwdGeom<D, B, K, M, W, C>;
  const cudaError_t cfg = configure_smem<D, B, K, M, W, C>();
  if (cfg != cudaSuccess) return cfg;
  const dim3 grid((unsigned)BoxPlan<D, B, K, M>::kOwners, (unsigned)BH);
  scan_a_kernel<D, B, K, M, W, C><<<grid, kThreads, G::bytes_a, sa>>>(pa);
  scan_b_kernel<D, B, K, M, W, C><<<grid, kThreads, G::bytes_b, sb>>>(pb);
  return cudaGetLastError();
}

template <int D, int B, int K, int M, int W, int C>
static std::array<int64_t, 12> census() {
  cudaFuncAttributes a{}, b{};
  if (cudaFuncGetAttributes(&a, scan_a_kernel<D, B, K, M, W, C>) != cudaSuccess
      || cudaFuncGetAttributes(&b, scan_b_kernel<D, B, K, M, W, C>) != cudaSuccess)
    return {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
  using G = BwdGeom<D, B, K, M, W, C>;
  return {a.numRegs,
          (int64_t)a.localSizeBytes,
          (int64_t)G::bytes_a,
          (int64_t)G::kStagesA,
          (int64_t)G::cta_target_a,
          (int64_t)G::kRegFloorA,
          b.numRegs,
          (int64_t)b.localSizeBytes,
          (int64_t)G::bytes_b,
          (int64_t)G::kStagesB,
          (int64_t)G::cta_target_b,
          (int64_t)G::kRegFloorB};
}

static bool is_bwd_arm(const std::vector<int64_t>& widths, int D, int B) {
  if ((int)widths.size() != D) return false;
  for (auto w : widths)
    if (w != B) return false;
  return true;
}

template <int D, int B, int K, int M>
static torch::Tensor bwd_owner_rows(const torch::Tensor& pr, const torch::Tensor& pw, int BH,
                                    int L) {
  using RP = BwdOwnerRows<D, B, K, M>;
  auto tbl = torch::empty({BH, 2, (long)RP::entries, rola::chunk::tok_words(L)},
                          pr.options().dtype(torch::kInt32));
  C10_CUDA_CHECK(
      (rola::chunk::launch_span_facts_arm<
          D, B, RP::span_at(0), RP::span_at(1), RP::span_at(2), RP::span_at(3), RP::span_at(0),
          RP::span_at(1), RP::span_at(2), RP::span_at(3), rola::chunk::kPrefillStateless>(
          (const __nv_bfloat16*)pr.data_ptr(), (const __nv_bfloat16*)pw.data_ptr(),
          reinterpret_cast<uint32_t*>(tbl.data_ptr()), BH, L, at::cuda::getCurrentCUDAStream())));
  return tbl;
}

std::vector<torch::Tensor> backward_inter(torch::Tensor pr, torch::Tensor pw, torch::Tensor gw,
                                          torch::Tensor vt, torch::Tensor dn,
                                          std::vector<int64_t> widths, int64_t k, int64_t m,
                                          int64_t window, int64_t chunk,
                                          c10::optional<torch::Tensor> s_in,
                                          c10::optional<torch::Tensor> ds_out, bool want_replay) {
  TORCH_CHECK(pr.is_cuda() && pw.is_cuda() && gw.is_cuda() && vt.is_cuda() && dn.is_cuda(),
              "cuda tensors only");
  TORCH_CHECK(pr.scalar_type() == torch::kBFloat16 && pw.scalar_type() == torch::kBFloat16
                  && gw.scalar_type() == torch::kBFloat16 && vt.scalar_type() == torch::kBFloat16
                  && dn.scalar_type() == torch::kBFloat16,
              "the operand planes are bf16");
  TORCH_CHECK(pr.is_contiguous() && pw.is_contiguous() && vt.is_contiguous() && dn.is_contiguous(),
              "contiguous planes");
  const int BH = (int)pr.size(0);
  const int L = (int)pr.size(1);
  TORCH_CHECK(vt.size(2) == kColsPad + kPadE && dn.sizes() == vt.sizes(),
              "the value and seed planes are pre-packed cols-wide and zero-padded to whole "
              "mma n-tiles; both have the same layout");

  int64_t N = 1;
  for (auto w : widths) N *= w;
  auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(pr.device());
  const int64_t wtot = (int64_t)widths.size() * (int64_t)widths[0];
  //: EVERY GRADIENT PLANE IS AT THE IDENTITY BEFORE THE FIRST CONTRIBUTION -- there is
  //: no output any owner writes WHOLE, so they are slices of one zeroed arena.
  auto d_pr = torch::zeros({BH, L, wtot}, opt);
  auto d_pw = torch::zeros({BH, L, wtot}, opt);
  auto d_gw = torch::zeros({BH, L}, opt);
  auto d_v = torch::zeros({BH, L, (int64_t)kDv}, opt);
  auto d_state_in = torch::zeros({BH, N / kAtomLeaves, (int64_t)kAtomLeaves, (int64_t)kCols}, opt);
  auto replay = want_replay
                    ? torch::zeros({BH, N / kAtomLeaves, (int64_t)kAtomLeaves, (int64_t)kCols}, opt)
                    : torch::zeros({0}, opt);

  BwdParams p{};
  p.pread = (const __nv_bfloat16*)pr.data_ptr();
  p.pwrite = (const __nv_bfloat16*)pw.data_ptr();
  p.gwrite = (const __nv_bfloat16*)gw.data_ptr();
  p.vt = (const __nv_bfloat16*)vt.data_ptr();
  p.dn = (const __nv_bfloat16*)dn.data_ptr();
  p.d_pr = d_pr.data_ptr<float>();
  p.d_pw = d_pw.data_ptr<float>();
  p.d_gw = d_gw.data_ptr<float>();
  p.d_v = d_v.data_ptr<float>();
  p.L = L;
  p.tok_words = rola::chunk::tok_words(L);

  auto main = at::cuda::getCurrentCUDAStream();
  torch::Tensor rows;
  cudaError_t err = cudaErrorInvalidValue;
#define BWD_ROWS(D_, B_, K_, M_, W_, C_)                                    \
  if (is_bwd_arm(widths, D_, B_) && k == K_ && m == M_ && window == W_ && chunk == C_) \
    rows = bwd_owner_rows<D_, B_, K_, M_>(pr, pw, BH, L);                   \
  else
  BWD_ARMS_X(BWD_ROWS) { TORCH_CHECK(false, "unbuilt backward arm"); }
#undef BWD_ROWS
  p.rows = reinterpret_cast<const uint32_t*>(rows.data_ptr());

  //: SCAN A TAKES A STREAM FROM THE POOL and Scan B keeps the caller's, so the two
  //: slabs are resident at once; every tensor Scan A touches is recorded against the
  //: side stream so the caching allocator cannot reuse it under the launch.
  auto side = at::cuda::getStreamFromPool();
  at::cuda::CUDAEvent fork, join;
  fork.record(main);
  fork.block(side);
  for (const auto& t : {pr, pw, gw, vt, dn, rows, d_pr, replay})
    c10::cuda::CUDACachingAllocator::recordStream(t.storage().data_ptr(), side);
  if (s_in.has_value())
    c10::cuda::CUDACachingAllocator::recordStream(s_in->storage().data_ptr(), side);

  BwdParams pa = p;
  pa.slab_in = s_in.has_value() ? s_in->data_ptr<float>() : nullptr;
  pa.slab_out = want_replay ? replay.data_ptr<float>() : nullptr;
  BwdParams pb = p;
  pb.slab_in = ds_out.has_value() ? ds_out->data_ptr<float>() : nullptr;
  pb.slab_out = d_state_in.data_ptr<float>();

#define BWD_DISPATCH(D_, B_, K_, M_, W_, C_)                                            \
  if (is_bwd_arm(widths, D_, B_) && k == K_ && m == M_ && window == W_ && chunk == C_)   \
    err = launch_scans<D_, B_, K_, M_, W_, C_>(pa, pb, BH, side, main);                  \
  else
  BWD_ARMS_X(BWD_DISPATCH) { TORCH_CHECK(false, "unbuilt backward arm"); }
#undef BWD_DISPATCH
  join.record(side);
  join.block(main);
  C10_CUDA_CHECK(err);
  return {d_pr, d_pw, d_gw, d_v, d_state_in, replay};
}

__global__ void bwd_stamp_kernel(int* out) {
  out[0] = kBwdBuildMagic;
  out[1] = kWarps * 1000 + kDv;
  out[2] = design::kCapsDigest;
}

std::vector<int64_t> backward_build_stamp(std::vector<int64_t> widths, int64_t k, int64_t m,
                                          int64_t window, int64_t chunk) {
  auto t = torch::zeros({3}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  bwd_stamp_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(t.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto h = t.cpu();
  std::vector<int64_t> r{h.data_ptr<int>()[0], h.data_ptr<int>()[1], h.data_ptr<int>()[2]};
#define BWD_CENSUS(D_, B_, K_, M_, W_, C_)                                            \
  if (is_bwd_arm(widths, D_, B_) && k == K_ && m == M_ && window == W_ && chunk == C_) { \
    for (auto x : census<D_, B_, K_, M_, W_, C_>()) r.push_back(x);                   \
    return r;                                                                          \
  }
  BWD_ARMS_X(BWD_CENSUS)
#undef BWD_CENSUS
  TORCH_CHECK(false, "unbuilt backward arm");
  return r;
}

}  // namespace rola::carry
