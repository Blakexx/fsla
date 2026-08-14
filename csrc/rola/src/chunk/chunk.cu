// csrc/rola/src/chunk/chunk.cu -- the chunk consumer's DISPATCH and torch
// entries (adopted from the ladder's rung 6G at P69).  This TU never touches
// a kernel symbol: arms arrive as extern declarations satisfied by the
// instantiation shards, exactly the consumer family's discipline.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <array>
#include <vector>

#include "chunk_kernel.cuh"

namespace rola::chunk {

//: the arm surface, DECLARED only -- definitions live in the instantiation
//: shards (chunk_s*.cu).  chunk.cu must never define one locally.
template <int C, bool SWZ, int BC, int D, int B>
cudaError_t launch_chunk_arm(const ChunkParams& p, cudaStream_t stream);
template <int D, int B, int BC>
cudaError_t launch_chunk_table_arm(const __nv_bfloat16* pr, const __nv_bfloat16* pw, uint32_t* tbl,
                                   int BH, int L, cudaStream_t stream);
template <int C, bool SWZ, int BC, int D, int B>
std::array<int64_t, 9> query_chunk_census_arm();
//: THE TABLE, AS A TENSOR: `[B*H, 2, entries, L/32]` of packed bits, sides apart
//: because the cook wants them apart.  It is a SEPARATE ENTRY POINT so that its
//: cost is reported on its own line and never folded into a kernel delta.
template <int D, int B, int BC>
static torch::Tensor build_table(const torch::Tensor& pr, const torch::Tensor& pw, int BH, int L) {
  constexpr int E = LevelPlan<D, B, BC>::entries;
  auto tbl = torch::empty({BH, 2, E, tok_words(L)}, pr.options().dtype(torch::kInt32));
  C10_CUDA_CHECK((launch_chunk_table_arm<D, B, BC>(
      reinterpret_cast<const __nv_bfloat16*>(pr.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(pw.data_ptr()),
      reinterpret_cast<uint32_t*>(tbl.data_ptr()), BH, L, at::cuda::getCurrentCUDAStream())));
  return tbl;
}

//: THE RATIO, IN ONE ELEMENTWISE PASS.  It cannot be in the kernel: `den[t]` is
//: a sum over the N/BC blocks and no block knows the total.  `eps` is a runtime
//: argument -- it is a constant of the DEFINITION and belongs to the definition's
//: file.  The divide is round-to-nearest by name.
__global__ void divide_kernel(const float* __restrict__ num, const float* __restrict__ den,
                              float* __restrict__ y, long vecs, float eps) {
  constexpr int kVec = 4;
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= vecs) return;
  const float d = den[i / (kD / kVec)] + eps;
  const float4 x = reinterpret_cast<const float4*>(num)[i];
  float4 r;
  r.x = __fdiv_rn(x.x, d);
  r.y = __fdiv_rn(x.y, d);
  r.z = __fdiv_rn(x.z, d);
  r.w = __fdiv_rn(x.w, d);
  reinterpret_cast<float4*>(y)[i] = r;
}

torch::Tensor divide_pass(torch::Tensor num, torch::Tensor den, double eps) {
  TORCH_CHECK(num.is_contiguous() && den.is_contiguous());
  TORCH_CHECK(num.scalar_type() == torch::kFloat32 && den.scalar_type() == torch::kFloat32);
  TORCH_CHECK(num.size(-1) == kD, "expect a numerator of d_v columns");
  auto y = torch::empty_like(num);
  const long vecs = num.numel() / 4;
  const int threads = 256;
  divide_kernel<<<(int)((vecs + threads - 1) / threads), threads, 0,
                  at::cuda::getCurrentCUDAStream()>>>(num.data_ptr<float>(), den.data_ptr<float>(),
                                                      y.data_ptr<float>(), vecs, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

//: THE ARGUMENT CHECKS OF A LAUNCH, SHARED BY THE TABLE AND THE KERNEL so that
//: the two can never be asked for different problems.
//: THE WIDTHS ARE A LIST -- the API speaks the widths policy's language even
//: though every built arm is uniform.  A caller with a jagged topology gets
//: the unbuilt-arm refusal, never a silent reshape.
static bool is_arm(const std::vector<int64_t>& widths, int64_t D, int64_t B) {
  if ((int64_t)widths.size() != D) return false;
  for (auto w : widths)
    if (w != B) return false;
  return true;
}

static void check_inputs(const torch::Tensor& pr, const torch::Tensor& pw, const torch::Tensor& v,
                         int64_t chunk, const std::vector<int64_t>& widths) {
  TORCH_CHECK(v.is_cuda() && v.scalar_type() == torch::kBFloat16 && v.is_contiguous());
  TORCH_CHECK(pr.is_contiguous() && pw.is_contiguous());
  TORCH_CHECK(pr.scalar_type() == torch::kBFloat16 && pw.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(v.dim() == 4 && v.size(3) == kD, "expect V [B,H,L,64]");
  int64_t wsum = 0;
  for (auto w : widths) wsum += w;
  TORCH_CHECK(pr.sizes() == pw.sizes() && pr.dim() == 4 && pr.size(3) == wsum,
              "expect packed amplitudes [B,H,L,sum(widths)]");
  const int B = v.size(0), H = v.size(1), L = v.size(2);
  TORCH_CHECK(pr.size(0) == B && pr.size(1) == H && pr.size(2) == L);
  TORCH_CHECK(L % chunk == 0, "L must be a multiple of the chunk");
  //: the widest gathered rows are the fp32 output (kD*4 B) and the packed
  //: amplitudes (sum(widths)*2 B); every gathered displacement must
  //: zero-extend from 32 bits.
  TORCH_CHECK((long)L <= kMaxGatherL && (long)L * wsum * 2 < (1L << 32),
              "a gathered displacement is 32 bits; this launch's would not "
              "fit (L=",
              L, ")");
}

torch::Tensor union_table(torch::Tensor pr, torch::Tensor pw, torch::Tensor v, int64_t chunk,
                          int64_t bc, std::vector<int64_t> widths) {
  check_inputs(pr, pw, v, chunk, widths);
  const int BH = v.size(0) * v.size(1), L = v.size(2);
#define RUNG5D_TABLE(C_, W_, BC_, D_, B_)                                     \
  if (chunk == C_ && bc == BC_ && is_arm(widths, D_, B_))                     \
    return build_table<D_, B_, BC_>(pr, pw, BH, L);
  RUNG5D_ARMS(RUNG5D_TABLE)
#undef RUNG5D_TABLE
  TORCH_CHECK(false, "unbuilt arm: chunk=", chunk, " BC=", bc, " depth=", widths.size());
}

std::vector<torch::Tensor> forward_raw(torch::Tensor pr, torch::Tensor pw, torch::Tensor gw,
                                       torch::Tensor v, torch::Tensor tbl, int64_t chunk,
                                       bool swizzle, int64_t bc, std::vector<int64_t> widths) {
  //: the write gain: per-token positive scalar, bf16, [B,H,L] -- spec 4.2.
  TORCH_CHECK(gw.is_cuda() && gw.is_contiguous() && gw.scalar_type() == torch::kBFloat16
                  && gw.dim() == 3 && gw.size(0) == v.size(0) && gw.size(1) == v.size(1)
                  && gw.size(2) == v.size(2),
              "g_write must be a contiguous bf16 [B,H,L] tensor");
  TORCH_CHECK(tbl.is_cuda() && tbl.is_contiguous() && tbl.scalar_type() == torch::kInt32,
              "the union table is a contiguous int32 tensor of packed bits");
  TORCH_CHECK(v.is_cuda() && v.scalar_type() == torch::kBFloat16 && v.is_contiguous());
  check_inputs(pr, pw, v, chunk, widths);
  const int B = v.size(0), H = v.size(1), L = v.size(2);
  //: The accumulator is zeroed, not merely allocated: the fan-in ADDS, so the
  //: identity of the sum has to be there before the first contribution.
  auto opts = v.options().dtype(torch::kFloat32);
  auto o = torch::zeros({B, H, L, (long)kD}, opts);
  auto den = torch::zeros({B, H, L}, opts);
  const int BH = B * H;
#define RUNG5D_DISPATCH(C_, W_, BC_, D_, B_)                                   \
  if (chunk == C_ && swizzle == W_ && bc == BC_ && is_arm(widths, D_, B_)) {   \
    const long want = (long)BH * 2 * LevelPlan<D_, B_, BC_>::entries *        \
                      tok_words(L);                                          \
    TORCH_CHECK(tbl.numel() == want,                                           \
                "the union table was built for a different arm");              \
    ChunkParams p_{reinterpret_cast<const __nv_bfloat16 *>(pr.data_ptr()),     \
                   reinterpret_cast<const __nv_bfloat16 *>(pw.data_ptr()),     \
                   reinterpret_cast<const __nv_bfloat16 *>(gw.data_ptr()),     \
                   reinterpret_cast<const __nv_bfloat16 *>(v.data_ptr()),      \
                   reinterpret_cast<const uint32_t *>(tbl.data_ptr()),         \
                   o.data_ptr<float>(), den.data_ptr<float>(), BH, L};         \
    C10_CUDA_CHECK((launch_chunk_arm<C_, W_, BC_, D_, B_>(                     \
        p_, at::cuda::getCurrentCUDAStream())));                               \
    return {o, den};                                                           \
  }
  RUNG5D_ARMS(RUNG5D_DISPATCH)
#undef RUNG5D_DISPATCH
  TORCH_CHECK(false, "unbuilt arm: chunk=", chunk, " swizzle=", swizzle, " BC=", bc,
              " depth=", widths.size());
}

torch::Tensor forward(torch::Tensor pr, torch::Tensor pw, torch::Tensor gw, torch::Tensor v,
                      int64_t chunk, bool swizzle, int64_t bc, std::vector<int64_t> widths,
                      double eps) {
  auto tbl = union_table(pr, pw, v, chunk, bc, widths);
  auto nd = forward_raw(pr, pw, gw, v, tbl, chunk, swizzle, bc, widths);
  return divide_pass(nd[0], nd[1], eps);
}

//: THE RING'S DEPTH IS PART OF THE BUILD'S IDENTITY, so a measurement can never
//: be attributed to a depth the running binary does not have.
__global__ void stamp_kernel(int* out) {
  out[0] = kBuildMagic + kFormWords;
  out[1] = (kGroupThreads / 32) * 1000 + kD;
  out[2] = design::kCapsDigest;
}

std::vector<int64_t> build_stamp(int64_t chunk, bool swizzle, int64_t bc,
                                 std::vector<int64_t> widths) {
  auto t = torch::zeros({3}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  stamp_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(t.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto h = t.cpu();
  std::vector<int64_t> r{h.data_ptr<int>()[0], h.data_ptr<int>()[1], h.data_ptr<int>()[2]};
#define RUNG5D_CENSUS(C_, W_, BC_, D_, B_)                                         \
  if (chunk == C_ && swizzle == W_ && bc == BC_ && is_arm(widths, D_, B_)) {       \
    const auto cs = query_chunk_census_arm<C_, W_, BC_, D_, B_>();                 \
    for (auto x : cs) r.push_back(x);                                              \
    return r;                                                                      \
  }
  RUNG5D_ARMS(RUNG5D_CENSUS)
#undef RUNG5D_CENSUS
  TORCH_CHECK(false, "unbuilt arm");
  return r;
}

//: The built set with every DERIVED quantity of an arm reported beside it: the
//: plan's depth and width, the block count, the block's inner/outer axes, the
//: staging depth and the CTA target.
std::vector<std::vector<int64_t>> arms() {
  std::vector<std::vector<int64_t>> out;
#define RUNG5D_LIST(C_, W_, BC_, D_, B_)                                      \
  out.push_back({C_, (int64_t)(W_), BC_, D_, B_,                              \
                 (int64_t)LevelPlan<D_, B_, BC_>::blocks,                     \
                 (int64_t)(BC_ / LevelPlan<D_, B_, BC_>::span_at(D_ - 1)),    \
                 (int64_t)LevelPlan<D_, B_, BC_>::span_at(D_ - 1),            \
                 (int64_t)Geom<C_, W_, BC_>::stages,                          \
                 (int64_t)Geom<C_, W_, BC_>::bytes,                           \
                 (int64_t)kCtaTarget<C_, W_, BC_>,                            \
                 (int64_t)LevelPlan<D_, B_, BC_>::entries});
  RUNG5D_ARMS(RUNG5D_LIST)
#undef RUNG5D_LIST
  return out;
}

}  // namespace rola::chunk
