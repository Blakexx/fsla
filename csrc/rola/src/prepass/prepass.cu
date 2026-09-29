// csrc/rola/src/prepass/prepass.cu -- THE PRE-PASS DISPATCH: the boundary's refusals through the one
// derivation, the launch of the depth this line builds, the ledger binding and the plan report.
// See docs/internals/prepass/prepass.md
#include "prepass/prepass_api.cuh"

#include "common/torch_seam.cuh"

#include "common/arch_runtime.cuh"
#include "prepass/params.cuh"
#include "prepass/plan.cuh"
#include "prepass/prepass_host.cuh"
#include "prepass/prepass_kernel.cuh"

namespace rola::prepass {

namespace {

long long* g_ledger = nullptr;
int64_t g_ledger_ctas = 0;

template <int D, int Q>
cudaError_t launch(const PrepassParams& p, int owners, int bh, cudaStream_t stream) {
  using TP = TallPlan<D>;
  const auto fn = prepass_kernel<D, Q>;
  static bool configured = false;
  if (!configured) {
    const cudaError_t e =
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, TP::kSmemBytes);
    if (e != cudaSuccess) return e;
    configured = true;
  }
  fn<<<dim3((unsigned)owners, (unsigned)bh), (unsigned)TP::kThreads, TP::kSmemBytes, stream>>>(p);
  return cudaGetLastError();
}

}  // namespace

void prepass_forward(const Tensor& rows_read, const Tensor& rows_write, const Tensor& scale_read,
                     const Tensor& scale_write, const Tensor& gain, const Tensor& vscale,
                     Tensor& den, Tensor& scale, const std::vector<int64_t>& widths,
                     int64_t level_modes, const Tensor& liveness,
                     const std::optional<Tensor>& fold_dump) {
  PrepassCall call =
      derive_prepass_call(rows_read, rows_write, scale_read, scale_write, gain, vscale, den, scale,
                          widths, level_modes, liveness, fold_dump);
  PrepassParams& p = call.p;
  const int D = (int)widths.size();
  p.ledger = g_ledger;
  if (p.ledger != nullptr)
    STD_TORCH_CHECK(g_ledger_ctas >= (int64_t)p.g.owners * call.BH,
                    "the bound phase ledger covers ", g_ledger_ctas, " CTAs, the launch needs ",
                    (int64_t)p.g.owners * call.BH);
  STD_TORCH_CHECK(D == 2, "the pre-pass is built at depth two on this line; this call names depth ",
                  D);
  //: the fold's shape: the write-dense level's span in n-tiles of eight (a declared width, R9)
  const int wl = p.sparse_mask[1] & 1u ? 0 : (p.sparse_mask[1] & 2u ? 1 : -1);
  const int wd = wl == 0 ? 1 : 0;
  const int qbits = wl >= 0 ? p.g.span_bits[wd] - 3 : 0;
  STD_TORCH_CHECK(qbits >= 0 && qbits <= 4,
                  "the pre-pass has no fold shape for a write-dense span of ",
                  1 << p.g.span_bits[wd]);
  cudaError_t status = cudaSuccess;
  // R9-CASES: the fold's shape, the write-dense span in n-tiles of eight (refused above)
  switch (qbits) {
    case 0:
      status = launch<2, 1>(p, p.g.owners, (int)call.BH, current_stream());
      break;
    case 1:
      status = launch<2, 2>(p, p.g.owners, (int)call.BH, current_stream());
      break;
    case 2:
      status = launch<2, 4>(p, p.g.owners, (int)call.BH, current_stream());
      break;
    case 3:
      status = launch<2, 8>(p, p.g.owners, (int)call.BH, current_stream());
      break;
    case 4:
      status = launch<2, 16>(p, p.g.owners, (int)call.BH, current_stream());
      break;
  }
  STD_TORCH_CHECK(status == cudaSuccess,
                  "the pre-pass launch failed: ", cudaGetErrorString(status));
}

void prepass_ledger_bind(const std::optional<Tensor>& ledger) {
  if (!ledger) {
    g_ledger = nullptr;
    g_ledger_ctas = 0;
    return;
  }
  STD_TORCH_CHECK(ledger->is_cuda() && ledger->scalar_type() == Dtype::Long && ledger->dim() == 3
                      && ledger->size(1) == carry::warps_per_sm && ledger->size(2) == kPpPhases
                      && ledger->is_contiguous(),
                  "the phase ledger is a contiguous CUDA int64 tensor [ctas][", carry::warps_per_sm,
                  "][", kPpPhases, "]");
  g_ledger = reinterpret_cast<long long*>(ledger->mutable_data_ptr<int64_t>());
  g_ledger_ctas = ledger->size(0);
}

std::vector<int64_t> prepass_plan(int64_t depth) {
  STD_TORCH_CHECK(depth == 2, "the pre-pass is built at depth two on this line");
  using TP = TallPlan<2>;
  cudaFuncAttributes a{};
  cudaFuncGetAttributes(&a, prepass_kernel<2, 1>);
  return {TP::kThreads, TP::kSmemBytes, TP::kLeaves, TP::kLeavesPerWarp, TP::kPoolTok,
          TP::kChunks,  TP::kRange,     kPpPhases,   (int64_t)a.numRegs, (int64_t)a.localSizeBytes};
}

}  // namespace rola::prepass
