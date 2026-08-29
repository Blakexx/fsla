// csrc/rola/src/carry/carry.cu -- the carry family's DISPATCH translation unit:
// dispatch and torch entries. THE ARM LIST LEFT THIS FILE AT K36 (tools/gen_shards.py
// + one TU per arm, KERNEL_STANDARDS section 7); this file instantiates no kernel,
// it matches a call against the generated rows. Full arm catalogue (12 arms, their
// (D,k,m,s,BC,W,C) and why each exists): see docs/internals/carry/carry.md#file-header-arms

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <array>

#include "carry_api.cuh"
#include "carry_arm_abi.cuh"
#include "carry_rows.cuh"
//: FOR THE BUILD STAMP ONLY -- the caps digest.  The kernel header is DELIBERATELY -- see docs/internals/carry/carry.md#near-line-81
#include "common/design.cuh"

namespace rola::carry {

//: the caps digest's namespace, aliased as the kernel header aliases it, so the -- see docs/internals/carry/carry.md#design
namespace design = denseref::design;

//: THE PER-LEVEL MODE DECLARATION (K33 law 3) is two bits per level with level -- see docs/internals/carry/carry.md#note-l92

//: THE BUILT SET, GENERATED (K36): the rows, and which of them this build -- see docs/internals/carry/carry.md#build-rows
#include "../instantiations/carry_arms.inc"

//: THE TABLE, AS A TENSOR.  `carry_rows.cuh` speaks pointers so that an arm TU never -- see docs/internals/carry/carry.md#rpr
template <class RPR, class RPW>
static torch::Tensor build_rows(const torch::Tensor& pr, const torch::Tensor& pw, int BH, int L) {
  auto tbl = torch::empty({BH, (long)rows_entries<RPR, RPW>(), rola::chunk::tok_words(L)},
                          pr.options().dtype(torch::kInt32));
  C10_CUDA_CHECK((launch_rows<RPR, RPW>(
      (const __nv_bfloat16*)pr.data_ptr(), (const __nv_bfloat16*)pw.data_ptr(),
      reinterpret_cast<uint32_t*>(tbl.data_ptr()), BH, L, at::cuda::getCurrentCUDAStream())));
  return tbl;
}

//: READING ONE ROW OF THE LIST. A row arrives with its INDEX first -- the index -- see docs/internals/carry/carry.md#carry-mo
#define CARRY_MO(D_, B_, K_, M_, DV_, W_, C_, CV_, MO_, ...) (MO_)
//: the seam's key, matched against the call's `(topology, k, m, d_v, W, C, -- see docs/internals/carry/carry.md#carry-match
#define CARRY_MATCH(D_, B_, K_, M_, DV_, W_, C_, CV_, ...)                                \
  (is_arm(widths, D_, B_) && k == K_ && m == M_ && d_v == DV_ && window == W_              \
   && chunk == C_ && carve == CV_)

//: `CV` is the SEAM's DECLARATION scalar, not a kernel axis: it names which -- see docs/internals/carry/carry.md#carry-cv-gate
#define CARRY_CV_GATE(I_, D_, B_, K_, M_, DV_, W_, C_, CV_, MO_, WPS_, NSW_, NSR_, PK_) \
  static_assert(((CV_) & 3) >= 1 && ((CV_) & 3) <= 3 && ((CV_) & ~7) == 0                    \
                    && (((CV_) >> 2) & 1) == (PK_) && (NSR_) == (NSW_) * (WPS_),             \
                "the seam's declaration scalar names a built (carve, executor map, park), "  \
                "its park bit IS the row's park declaration, and the executor map is the "   \
                "row's own arithmetic: one READ stream per warp over `WPS` warps a stream");
CARRY_ARMS_X(CARRY_CV_GATE)
#undef CARRY_CV_GATE

//: WHAT EACH BUILT ARM'S TRANSLATION UNIT EXPORTS. -- see docs/internals/carry/carry.md#carry-arms-x
CARRY_ARMS_X(CARRY_ARM_DECLARE)

static bool is_arm(const std::vector<int64_t>& widths, int D, int B) {
  if ((int)widths.size() != D) return false;
  for (auto w : widths)
    if (w != B) return false;
  return true;
}

std::vector<torch::Tensor> forward_inter(torch::Tensor pr, torch::Tensor pw, torch::Tensor gw,
                                         torch::Tensor v, std::vector<int64_t> widths, int64_t k,
                                         int64_t m, int64_t window, int64_t chunk, int64_t carve,
                                         int64_t level_modes, c10::optional<torch::Tensor> s_in,
                                         c10::optional<torch::Tensor> s_out,
                                         c10::optional<torch::Tensor> page_tbl,
                                         c10::optional<torch::Tensor> atom_bits) {
  TORCH_CHECK(pr.is_cuda() && pw.is_cuda() && gw.is_cuda() && v.is_cuda(), "cuda tensors only");
  TORCH_CHECK(pr.scalar_type() == torch::kBFloat16 && pw.scalar_type() == torch::kBFloat16
                  && gw.scalar_type() == torch::kBFloat16 && v.scalar_type() == torch::kBFloat16,
              "the operand planes are bf16");
  TORCH_CHECK(pr.is_contiguous() && pw.is_contiguous() && v.is_contiguous(), "contiguous planes");
  //: `pr`/`pw` are `[BH, L, sum_l B_l]`; `v` is `[B, T, H, d_v]`; `gw` is `[BH, L]`.
  const int BH = (int)pr.size(0);
  const int L = (int)pr.size(1);
  //: THE VALUE WIDTH IS READ OFF THE CALLER'S PLANE and is part of the arm key, so an -- see docs/internals/carry/carry.md#dv
  const int d_v = (int)v.size(-1);
  const int v_row_bytes = (int)(v.size(2) * d_v * v.element_size());

  auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(v.device());
  auto o = torch::zeros({BH, L, (int64_t)d_v}, opt);
  auto den = torch::zeros({BH, L}, opt);

  //: THE STATE PLANES ARE THE CALLER'S (docs/internals/state.md section 4). -- see docs/internals/carry/carry.md#torch-check
  for (const auto& t : {s_in, s_out})
    if (t.has_value())
      TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->scalar_type() == torch::kFloat32,
                  "a state plane is a contiguous fp32 cuda tensor");
  if (page_tbl.has_value()) {
    TORCH_CHECK(page_tbl->is_cuda() && page_tbl->is_contiguous()
                    && page_tbl->scalar_type() == torch::kInt32,
                "the page table is a contiguous int32 cuda tensor");
    TORCH_CHECK(s_in.has_value() || s_out.has_value(),
                "a page table addresses a state plane, and this call carries neither");
  }
  const float* sin_p = s_in.has_value() ? s_in->data_ptr<float>() : nullptr;
  float* sout_p = s_out.has_value() ? s_out->data_ptr<float>() : nullptr;
  const int32_t* ptbl_p =
      page_tbl.has_value() ? reinterpret_cast<const int32_t*>(page_tbl->data_ptr()) : nullptr;
  //: THE ACTIVITY FACT IS THE FACTS PASS' (P80b), and it is OPTIONAL at this seam -- see docs/internals/carry/carry.md#atom-bits
  if (atom_bits.has_value()) {
    long N = 1;
    for (auto w : widths) N *= w;
    TORCH_CHECK(atom_bits->is_cuda() && atom_bits->is_contiguous()
                    && atom_bits->scalar_type() == torch::kUInt8,
                "the atom activity bitmap is a contiguous uint8 cuda tensor");
    TORCH_CHECK(atom_bits->numel() == (long)BH * (N / kAtomLeaves),
                "the activity bitmap holds one byte per atom: BH * N / ", kAtomLeaves,
                " entries, got ", atom_bits->numel());
    TORCH_CHECK(s_in.has_value() || s_out.has_value(),
                "an activity bitmap gates a state sweep, and this call carries neither plane");
    TORCH_CHECK(!(s_in.has_value() && s_out.has_value()) || s_in->data_ptr() == s_out->data_ptr(),
                "a gated continuation ADVANCES IN PLACE (docs/internals/state.md): "
                "the exit sweep stores the atoms this call writes and no "
                "others, so an atom it skips must already hold what it carried in. Two "
                "distinct planes would silently drop every unwritten atom. Pass the same "
                "plane as both, or pass no activity bitmap.");
  }
  const uint8_t* abits_p =
      atom_bits.has_value() ? reinterpret_cast<const uint8_t*>(atom_bits->data_ptr()) : nullptr;
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaError_t err = cudaErrorInvalidValue;
  //: THE BALLOT ROWS COME FIRST, ON THE SAME STREAM (K31-i-d). One pass over the -- see docs/internals/carry/carry.md#tok-words
  torch::Tensor rows;
  const int tok_words = rola::chunk::tok_words(L);
  auto run = [&](int64_t mo) {
    bool matched = false;
#define CARRY_DISPATCH(I_, ...)                                                             \
  if (CARRY_MATCH(__VA_ARGS__) && mo == CARRY_MO(__VA_ARGS__)) {                            \
    matched = true;                                                                         \
    rows = torch::empty({BH, (long)carry_arm_rows_entries_##I_(), tok_words},               \
                        pr.options().dtype(torch::kInt32));                                 \
    err = carry_arm_rows_##I_((const __nv_bfloat16*)pr.data_ptr(),                          \
                              (const __nv_bfloat16*)pw.data_ptr(),                          \
                              reinterpret_cast<uint32_t*>(rows.data_ptr()), BH, L, stream);  \
    if (err == cudaSuccess)                                                                 \
      err = carry_arm_forward_##I_(                                                          \
          (const __nv_bfloat16*)pr.data_ptr(), (const __nv_bfloat16*)pw.data_ptr(),          \
          (const __nv_bfloat16*)gw.data_ptr(), (const __nv_bfloat16*)v.data_ptr(),           \
          reinterpret_cast<const uint32_t*>(rows.data_ptr()), o.data_ptr<float>(),           \
          den.data_ptr<float>(), BH, L, tok_words, v_row_bytes, sin_p, sout_p, ptbl_p,      \
          abits_p,                                                                           \
          stream);                                                                           \
  } else
    CARRY_ARMS_X(CARRY_DISPATCH) {}
#undef CARRY_DISPATCH
    return matched;
  };
  //: THE DECLARATION IS RESOLVED AT THIS SEAM, AND AN UNBUILT ONE IS REFUSED (P76 -- see docs/internals/carry/carry.md#torch-check-2
  TORCH_CHECK(run(level_modes), "unbuilt carry arm: widths=", widths, " k=", k, " m=", m,
              " d_v=", d_v, " window=", window, " chunk=", chunk, " carve=", carve,
              " level_modes=", level_modes);
  C10_CUDA_CHECK(err);
  return {o, den};
}

//: THE FILE'S OWN CONSTANTS, device-side, beside the magic: the channel-split body's -- see docs/internals/carry/carry.md#stampkernelentry
__global__ void stamp_kernel_entry(int* out) {
  out[0] = kCarryBuildMagic;
  out[1] = kWarps * 1000 + kAtomLeaves;
  out[2] = design::kCapsDigest;
}

std::vector<int64_t> build_stamp(std::vector<int64_t> widths, int64_t k, int64_t m, int64_t window,
                                 int64_t chunk, int64_t carve, int64_t level_modes, int64_t d_v) {
  auto t = torch::zeros({3}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  stamp_kernel_entry<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(t.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto h = t.cpu();
  std::vector<int64_t> r{h.data_ptr<int>()[0], h.data_ptr<int>()[1], h.data_ptr<int>()[2]};
#define CARRY_CENSUS(I_, ...)                                                               \
  if (CARRY_MATCH(__VA_ARGS__) && level_modes == CARRY_MO(__VA_ARGS__)) {                   \
    int64_t c[kCarryCensusFields];                                                          \
    carry_arm_census_##I_(c);                                                                \
    for (int i = 0; i < kCarryCensusFields; ++i) r.push_back(c[i]);                           \
    return r;                                                                               \
  }
  CARRY_ARMS_X(CARRY_CENSUS)
#undef CARRY_CENSUS
  TORCH_CHECK(false, "unbuilt carry arm");
  return r;
}

torch::Tensor owner_rows(torch::Tensor pr, torch::Tensor pw, std::vector<int64_t> widths, int64_t k,
                         int64_t m, bool carve) {
  TORCH_CHECK(pr.is_cuda() && pw.is_cuda() && pr.is_contiguous() && pw.is_contiguous(),
              "contiguous cuda planes");
  TORCH_CHECK(pr.scalar_type() == torch::kBFloat16 && pw.scalar_type() == torch::kBFloat16,
              "the operand planes are bf16");
  TORCH_CHECK(pr.sizes() == pw.sizes() && pr.dim() == 3, "expect packed amplitudes [BH, L, sumW]");
  const int BH = (int)pr.size(0), L = (int)pr.size(1);
#define CARRY_ROWS(I_, D_, B_, K_, M_, DV_, W_, C_, CV_, MO_, ...)                                \
  if constexpr ((MO_) == 0) {                                                                 \
    if (is_arm(widths, D_, B_) && k == K_ && m == M_)                                         \
      return carve                                                                            \
                 ? build_rows<SeamSplitPlan<D_, B_, K_, M_>, SeamSplitPlan<D_, B_, K_, M_>>(   \
                       pr, pw, BH, L)                                                         \
                 : build_rows<OwnerRows<D_, B_, K_, M_>, OwnerRows<D_, B_, K_, M_>>(pr, pw, BH,\
                                                                                    L);       \
  }
  CARRY_ARMS_X(CARRY_ROWS)
#undef CARRY_ROWS
  TORCH_CHECK(false, "unbuilt carry arm");
}

std::vector<std::vector<int64_t>> arms() {
  std::vector<std::vector<int64_t>> out;
#define CARRY_LIST(I_, ...)                                     \
  {                                                             \
    int64_t row[kCarryRowFields];                               \
    carry_arm_row_##I_(row);                                    \
    out.push_back(std::vector<int64_t>(row, row + kCarryRowFields)); \
  }
  CARRY_ARMS_X(CARRY_LIST)
#undef CARRY_LIST
  return out;
}

}  // namespace rola::carry
