// csrc/rola/src/carry/carry.cu -- the carry family's DISPATCH translation unit.
//
// Dispatch and torch entries.  THE ARM LIST LEFT THIS FILE AT K36: it lives in
// `tools/gen_shards.py`, which emits `../instantiations/carry_arms.inc` (the rows,
// and which of them this build carries) and ONE TRANSLATION UNIT PER ARM beside it
// (KERNEL_STANDARDS section 7).  This file therefore instantiates no kernel: it
// matches a call against the rows and calls the arm's four entry points, which is
// what makes an arm-subset build a FILE subset instead of a macro that still costs
// a parse of every row.  Each arm DECLARES its two carves
// -- the per-level span vector one STREAM owns on the write side and on the read
// side -- so the carve is a property of the arm and never a constant of a kernel,
// and the seam's `carve` scalar survives only to refuse a carve nothing builds.
//
// The arms (`s` = the owner box, `SW`/`SR` = the two stream boxes):
//   flagship   (256,256)  D=2  k=4 m=8   s=(8,16)     BC=128  W=384  C=32
//   flagship+  (256,256)  D=2  k=4 m=8   s=(8,16)     BC=128  W=512  C=32
//   small      ( 64, 64)  D=2  k=4 m=8   s=(8,16)     BC=128  W=64   C=32
//   deep-3     (16^3)     D=3  k=2 m=16  s=(2,4,16)   BC=128  W=64   C=32
//   deep-4     ( 8^4)     D=4  k=2 m=8   s=(2,2,4,8)  BC=128  W=64   C=32
//   carve-4    ( 4^4)     D=4  k=4 m=1   s=(4,4,4,4)  BC=256  W=64   C=32
//   flagship 1x8 (256,256) D=2  k=4 m=8   s=(8,16)     BC=128  W=384  C=32, twice
//   flagship 1x8 d_v=128   the SAME two, at the wide value plane
//   flagship 2x4 d_v=128 (256,256) D=2 k=4 m=4 s=(4,16) BC=64  W=384  C=32
//   + the PARK's twins (K35 S5): those three and the flagship d_v=64 ALT row again
//   flagship tied (256,256) D=2 k=4 m=8   s=(8,16)     BC=128  W=384  C=32, MO=3
// The first five hold ONE box of `BC = 128` -- the register budget's number, which
// does not move with depth -- so they differ only in how the same 128 leaves cut
// into digits, and `deep-4` is the topology whose atom rectangle cannot be a single
// level (`B = 8` is narrower than an atom).  `carve-4` is the MULTI-LEVEL CARVE arm
// and the one whose owner is the whole topology: its 256 leaves are cut into EIGHT
// streams of 32 by `SW = [1,4,2,4]` and `SR = [4,1,4,2]`, two carved levels per
// side, one warp per stream -- so it is also the arm that runs eight executors.
// `k = 4, m = 1` is the only capacity reaching the uniform box `(4,4,4,4)`.
// The LAST row is the TIED arm: the flagship cell declared BOTH-SPARSE at level zero,
// where the order-by-mode rule gives the two sides the SAME order and the derivation the
// SAME vector, so the exchange is the identity and is elided at compile time.
//
// `d_v` IS THE THIRD DIAL AND A KEY OF THE ARM (K35 S3b).  It is a template parameter
// of both bodies, so the `d_v = 128` rows are not a wider path but the same body at a
// wider declaration, and the seam tells them apart by the width the caller's value
// plane carries.  The card's CONTAINERS entry rules what the width costs: sixteen
// leaves per warp is REQUIRED at 128, and eight resident warps per SM is the target, so
// the two ways to spend them are 1x8 at `BC = 128` (rows 13/14, the two container
// declarations at the wide plane) and 2x4 at `BC = 64` (row 15, the CTA's box halved
// instead of its warps doubled -- four streams, one warp each, the ruled leaf split).
// The `d_v = 64` arms are UNTOUCHED by all of it, which is the point of making the
// width a parameter rather than a branch.
//
// THE EXECUTOR MAP IS THE SECOND DIAL (K35 S3a).  An arm declares `WPS`, the warps per
// WRITE stream, so the CTA's warp count is `streams x WPS` and the READ carve is
// declared at one stream per warp.  The two `flagship 1x8` rows are the SAME cell as
// `flagship ALT` reached with EIGHT executors instead of four, by the two available
// routes -- four write streams column-split two ways (`SW = [2,16]`, `WPS = 2`) and
// eight write streams one warp each (`SW = [1,16]`, `WPS = 1`) -- so the container
// question the card leaves to measurement has both of its answers built.  The existing
// four-warp `ALT` arms ARE the 2x4 container: two CTAs of four warps per SM at half the
// SM's box each, which is the same eight resident warps, so no arm is added for it.
// The window is the architecture doc's TWO-CONSTRAINT quantity (integration rung
// I1(b)): the C* cost basin is shallow, and the intra kernel -- which runs on the
// SAME window grid -- has ideal arithmetic intensity `W/4` flop/byte and needs
// `W >= ~380` to clear sm_86's compute/bandwidth balance.  The doc's admitted band
// is 384-512 and BOTH ends are built, because a third constraint decides between
// them: the intra refuses a window that does not divide `L`, and 384 does not
// divide the flagship length `L = N = 65536`.  512 does, at every power-of-two
// length, while clearing the intensity bar with room (128 flop/byte).  The
// `(64, 64)` arm is the conformance cell: the fp64 oracle materializes
// `[B,T,H,N]`, so `N = 4096` is where a whole cell is checkable, and its `W = 64`
// is held to keep that cell's window count meaningful at the lengths a reference
// can hold.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <array>

#include "carry_api.cuh"
#include "carry_arm_abi.cuh"
#include "carry_rows.cuh"
//: FOR THE BUILD STAMP ONLY -- the caps digest.  The kernel header is DELIBERATELY
//: absent: this TU instantiates no arm, and the two constants the stamp needs beside
//: the digest are in `box.cuh` with the family's other arm-independent geometry.
#include "common/design.cuh"

namespace rola::carry {

//: the caps digest's namespace, aliased as the kernel header aliases it, so the
//: stamp names the same constant by the same name it always did.
namespace design = denseref::design;

//: THE PER-LEVEL MODE DECLARATION (K33 law 3) is two bits per level with level 0 in
//: the low pair: bit 0 = the READ side is value-sparse there, bit 1 = the WRITE side.
//: `DENSE_BOTH` everywhere is always correct and only ever slower.  The encoding, the
//: private order it induces and the carve derived from that order are all in `box.cuh`
//: (`mode_declares`, `ModeOrder`, `StreamSpans`); this TU only passes the word through.

//: THE BUILT SET, GENERATED (K36): the rows, and which of them this build carries.
//: `tools/gen_shards.py` owns the list and emits one translation unit per arm
//: beside this include; `setup.py` selects the subset by compiling those files and
//: defining `ROLA_CARRY_BUILD_<i>` for each, so `CARRY_ARMS_X` expands to exactly
//: the arms the binary carries.  docs/build.md.
#include "../instantiations/carry_arms.inc"

//: THE TABLE, AS A TENSOR.  `carry_rows.cuh` speaks pointers so that an arm TU never
//: parses the tensor library (K36); the allocation it leaves to the caller is done
//: here, in the one TU that already pays for torch.
template <class RPR, class RPW>
static torch::Tensor build_rows(const torch::Tensor& pr, const torch::Tensor& pw, int BH, int L) {
  auto tbl = torch::empty({BH, (long)rows_entries<RPR, RPW>(), rola::chunk::tok_words(L)},
                          pr.options().dtype(torch::kInt32));
  C10_CUDA_CHECK((launch_rows<RPR, RPW>(
      (const __nv_bfloat16*)pr.data_ptr(), (const __nv_bfloat16*)pw.data_ptr(),
      reinterpret_cast<uint32_t*>(tbl.data_ptr()), BH, L, at::cuda::getCurrentCUDAStream())));
  return tbl;
}

//: READING ONE ROW OF THE LIST.  A row arrives with its INDEX first -- the index is
//: what names the arm's translation unit and its four entry points -- and the pieces a
//: consumer wants are taken from it HERE and nowhere else.  `CARRY_ARM_TYPE`, the arm
//: AS A TYPE, is not here: it needs the geometry, so it lives with the bodies in
//: `carry_arm.cuh` and this TU never names it.
#define CARRY_MO(D_, B_, K_, M_, DV_, W_, C_, CV_, MO_, ...) (MO_)
//: the seam's key, matched against the call's `(topology, k, m, d_v, W, C, carve)`.
//: `d_v` IS PART OF THE KEY (K35 S3b): the two widths of one declaration are two arms,
//: so the container rows 11/12 and 13/14 share their declaration scalar and are told
//: apart by the width the caller's value plane actually carries.
#define CARRY_MATCH(D_, B_, K_, M_, DV_, W_, C_, CV_, ...)                                \
  (is_arm(widths, D_, B_) && k == K_ && m == M_ && d_v == DV_ && window == W_              \
   && chunk == C_ && carve == CV_)

//: `CV` is the SEAM's DECLARATION scalar, not a kernel axis: it names which declared
//: (carve, executor map, PARK) an arm is, so a caller that asks for the deleted one --
//: or for an executor map nothing builds -- is refused rather than served something
//: else.  Its low two bits are the (carve, executor map): `1` the ruled leaf split, ONE
//: warp per write stream (`rola.ops.carry.LEAF_SPLIT`); `2` the 1x8 container as FOUR
//: write streams x TWO warps each (the column split); `3` as EIGHT streams x ONE warp;
//: and BIT 2 IS THE PARK (`+4`, `PARKED`), which changes nothing the caller can see;
//: the row's `PK_` must agree with it, asserted over the arms THIS BUILD carries.
#define CARRY_CV_GATE(I_, D_, B_, K_, M_, DV_, W_, C_, CV_, MO_, WPS_, NSW_, NSR_, PK_) \
  static_assert(((CV_) & 3) >= 1 && ((CV_) & 3) <= 3 && ((CV_) & ~7) == 0                    \
                    && (((CV_) >> 2) & 1) == (PK_) && (NSR_) == (NSW_) * (WPS_),             \
                "the seam's declaration scalar names a built (carve, executor map, park), "  \
                "its park bit IS the row's park declaration, and the executor map is the "   \
                "row's own arithmetic: one READ stream per warp over `WPS` warps a stream");
CARRY_ARMS_X(CARRY_CV_GATE)
#undef CARRY_CV_GATE

//: WHAT EACH BUILT ARM'S TRANSLATION UNIT EXPORTS.  Declared over `CARRY_ARMS_X` and
//: therefore over exactly the arms this build compiled: an arm outside the subset has
//: no declaration, no call site and no entry in `arms()`, so the seam's refusal below
//: is the only thing a caller asking for it can reach.
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
                                         c10::optional<torch::Tensor> page_tbl) {
  TORCH_CHECK(pr.is_cuda() && pw.is_cuda() && gw.is_cuda() && v.is_cuda(), "cuda tensors only");
  TORCH_CHECK(pr.scalar_type() == torch::kBFloat16 && pw.scalar_type() == torch::kBFloat16
                  && gw.scalar_type() == torch::kBFloat16 && v.scalar_type() == torch::kBFloat16,
              "the operand planes are bf16");
  TORCH_CHECK(pr.is_contiguous() && pw.is_contiguous() && v.is_contiguous(), "contiguous planes");
  //: `pr`/`pw` are `[BH, L, sum_l B_l]`; `v` is `[B, T, H, d_v]`; `gw` is `[BH, L]`.
  const int BH = (int)pr.size(0);
  const int L = (int)pr.size(1);
  //: THE VALUE WIDTH IS READ OFF THE CALLER'S PLANE and is part of the arm key, so an
  //: unbuilt width is refused by the dispatch exactly as an unbuilt topology is.
  const int d_v = (int)v.size(-1);
  const int v_row_bytes = (int)(v.size(2) * d_v * v.element_size());

  auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(v.device());
  auto o = torch::zeros({BH, L, (int64_t)d_v}, opt);
  auto den = torch::zeros({BH, L}, opt);

  //: THE STATE PLANES ARE THE CALLER'S (docs/internals/state.md section 4).  A
  //: continuation hands the SAME tensor as both, and advances it in place; a call that
  //: hands NEITHER is the NULL-STATE call: a runtime null check inside the one
  //: compiled body (K41; no STATE template axis) skips the sweep at that end, so
  //: nothing is read, written, or allocated there.  `page_tbl` null is the dense
  //: backing -- the slot is then the atom -- so the two backings are one kernel, one
  //: ABI and one accumulation order.
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
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaError_t err = cudaErrorInvalidValue;
  //: THE BALLOT ROWS COME FIRST, ON THE SAME STREAM (K31-i-d).  One pass over the
  //: amplitude planes votes every owner's rectangle test once; the kernel below then
  //: reads WORDS instead of re-reading those planes per (owner, window, token).  The
  //: table is a TRANSIENT of this call -- nothing else asks for it -- and it is held
  //: here rather than inside the dispatch so its lifetime outlives the launch.
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
          stream);                                                                           \
  } else
    CARRY_ARMS_X(CARRY_DISPATCH) {}
#undef CARRY_DISPATCH
    return matched;
  };
  //: THE DECLARATION IS RESOLVED AT THIS SEAM, AND AN UNBUILT ONE IS REFUSED (P76 R-B,
  //: overruling law 3).  The retry that forced `mo` to zero served the DENSE_BOTH body
  //: at the requested carve with no diagnostic -- a no-fallbacks violation that hid a
  //: 5x loss from a harness for nine stages (`## BISECT (corrected)`).  DENSE_BOTH is
  //: served when it is DECLARED and never as a substitute for something else, so the
  //: refusal names the whole key the caller asked for.
  TORCH_CHECK(run(level_modes), "unbuilt carry arm: widths=", widths, " k=", k, " m=", m,
              " d_v=", d_v, " window=", window, " chunk=", chunk, " carve=", carve,
              " level_modes=", level_modes);
  C10_CUDA_CHECK(err);
  return {o, den};
}

//: THE FILE'S OWN CONSTANTS, device-side, beside the magic: the channel-split body's
//: executor count and the page granule.  `d_v` LEFT THIS SET at K35 S3b -- it is an arm
//: parameter now, so it is reported by the per-arm CENSUS below and not by the stamp.
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
