// csrc/rola/src/carry/carry.cu -- K31 R1: the carry family's ONE translation unit.
//
// Dispatch, torch entries and the built arm list.  The arm list is small BY
// DESIGN: the carve question is SETTLED (the innermost level's leaf split, K33
// clean-sheet tier A), so the axis is gone from the kernel and survives only as the
// seam's `carve` scalar, which the dispatch admits at its one ruled value.
//
// The arms:
//   flagship   (256,256)  D=2  k=4 m=8   s=(8,16)     BC=128  W=384  C=32
//   flagship+  (256,256)  D=2  k=4 m=8   s=(8,16)     BC=128  W=512  C=32
//   small      ( 64, 64)  D=2  k=4 m=8   s=(8,16)     BC=128  W=64   C=32
//   deep-3     (16^3)     D=3  k=2 m=16  s=(2,4,16)   BC=128  W=64   C=32
//   deep-4     ( 8^4)     D=4  k=2 m=8   s=(2,2,4,8)  BC=128  W=64   C=32
// Every arm holds ONE box: `BC = 128` is the register budget's number and it does
// not move with depth, so the two deep arms differ from the flat ones only in how
// the same 128 leaves are cut into digits.  Both are conformance cells at `N =
// 4096`, and the deep-4 arm is the one topology whose atom rectangle cannot be a
// single level -- `B = 8` is narrower than an atom.
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
#include "carry_split_kernel.cuh"
#include "facts/facts_kernel.cuh"

//: THE OWNER BALLOT ROWS, DECLARED (K31-i-d).  The facts family owns the pass and
//: its instantiation shard owns the arms; this TU sees a DECLARATION, exactly as the
//: facts dispatch TU does, so nothing here can satisfy an arm locally.  The SPANS the
//: carry asks for are `BoxPlan::s(l)` -- the lattice's balanced owner runs, which is
//: the vocabulary the lex plan could not state.
namespace rola::chunk {
template <int D, int B, int S0, int S1, int S2, int S3, int MASK>
cudaError_t launch_span_facts_arm(const __nv_bfloat16* pr, const __nv_bfloat16* pw, uint32_t* tbl,
                                  int BH, int L, cudaStream_t stream);
}  // namespace rola::chunk

namespace rola::carry {

//: the plan the carry's rows are voted under, named ONCE: a `BoxPlan`'s spans as a
//: `SpanPlan`.  `D` is at most four and a level past `D` carries `B`.
template <int D, int B, int K, int M>
using OwnerRows =
    rola::chunk::SpanPlan<D, B, BoxPlan<D, B, K, M>::s(0), (D > 1 ? BoxPlan<D, B, K, M>::s(1) : B),
                          (D > 2 ? BoxPlan<D, B, K, M>::s(2) : B),
                          (D > 3 ? BoxPlan<D, B, K, M>::s(3) : B)>;

//: the plan the CARVE body's rows are voted under: the same table one level finer
//: at the carve level, so a warp reads its own group's row (`CarveRows`).
template <int D, int B, int K, int M, int CL>
using CarveRowPlan = rola::chunk::SpanPlan<D, B, CarveRows<D, B, K, M, kWarps, CL>::s(0),
                                           (D > 1 ? CarveRows<D, B, K, M, kWarps, CL>::s(1) : B),
                                           (D > 2 ? CarveRows<D, B, K, M, kWarps, CL>::s(2) : B),
                                           (D > 3 ? CarveRows<D, B, K, M, kWarps, CL>::s(3) : B)>;

//: THE PER-LEVEL MODE DECLARATION (K33 law 3), two bits per level with level 0 in
//: the low pair: bit 0 = the READ side is value-sparse there, bit 1 = the WRITE
//: side.  `DENSE_BOTH` everywhere is always correct and only ever slower.
constexpr uint32_t kModeReadSparse = 1u;
constexpr uint32_t kModeWriteSparse = 2u;
constexpr bool declares(uint32_t modes, int l, uint32_t side) {
  return ((modes >> (2 * l)) & side) != 0u;
}

//: THE BUILT SET, as one macro so the dispatch, the census and the arm list cannot
//: drift apart -- the shipped family's `CHUNK_ARMS_X` discipline.
//: MO_ is the per-level declaration this arm is BUILT for, and it chooses the
//: body.  `ALT` at depth `D` is write-sparse on the even levels and read-sparse on
//: the odd ones, i.e. `sum_l mode_l << 2l` = 6 / 38 / 102 at D = 2 / 3 / 4.
#define CARRY_ARMS_X(X_)             \
  X_(2, 256, 4, 8, 384, 32, 1, 0)    \
  X_(2, 256, 4, 8, 384, 32, 1, 6)    \
  X_(2, 256, 4, 8, 512, 32, 1, 0)    \
  X_(2, 256, 4, 8, 512, 32, 1, 6)    \
  X_(2, 64, 4, 8, 64, 32, 1, 0)      \
  X_(2, 64, 4, 8, 64, 32, 1, 6)      \
  X_(3, 16, 2, 16, 64, 32, 1, 0)     \
  X_(3, 16, 2, 16, 64, 32, 1, 38)    \
  X_(4, 8, 2, 8, 64, 32, 1, 0)       \
  X_(4, 8, 2, 8, 64, 32, 1, 102)

//: THE CARVE LEVEL, NAMED BY THE DECLARATION.  The read side's leaf carve keys on
//: the INNERMOST level the declaration marks read-sparse; a declaration that marks
//: none carves nothing and takes the channel-split body.  `CarveMap`'s private leaf
//: order is exact only while that level is the innermost one, so a declaration
//: naming an outer level is served channel-split too -- always correct, and stated
//: rather than silently carved at the wrong level.
template <int D, uint32_t MO>
constexpr int carve_level = declares(MO, D - 1, kModeReadSparse) ? D - 1 : -1;

//: `CV` is the SEAM's carve scalar, not a kernel axis: it names the one ruled
//: carve so a caller that asks for the deleted one is refused rather than served
//: something else.
template <int D, uint32_t MO>
static constexpr bool kCarves = carve_level<D, MO> >= 0;

//: THE WRITE CARVE LEVEL: the innermost level the declaration marks WRITE-sparse,
//: excluding the read carve's own.  A body with both carves evacuates its state
//: between the phases (K35-E); one with only the read carve keeps a single carve and
//: the write list stays shared, because every warp then owns every write entry.
constexpr int wcarve_of(uint32_t modes, int d, int cr) {
  for (int l = d - 1; l >= 0; --l)
    if (l != cr && declares(modes, l, kModeWriteSparse)) return l;
  return -1;
}
template <int D, uint32_t MO>
static constexpr int write_carve_level = wcarve_of(MO, D, carve_level<D, MO>);
template <int D, uint32_t MO>
static constexpr bool kTwoCarve = kCarves<D, MO> && (write_carve_level<D, MO> >= 0);

//: the plan a TWO-CARVE body's rows are voted under: the same table refined at BOTH
//: carved levels at once, so each warp reads its own group's row on EITHER side.
template <int D, int B, int K, int M, int CW, int CR>
using TwoCarveRowPlan =
    rola::chunk::SpanPlan<D, B, TwoCarveRows<D, B, K, M, kWarps, CW, CR>::s(0),
                          (D > 1 ? TwoCarveRows<D, B, K, M, kWarps, CW, CR>::s(1) : B),
                          (D > 2 ? TwoCarveRows<D, B, K, M, kWarps, CW, CR>::s(2) : B),
                          (D > 3 ? TwoCarveRows<D, B, K, M, kWarps, CW, CR>::s(3) : B)>;

//: THE ARM'S BODY, as one statement the launch, the census and the SMEM ledger all
//: read: only the selected one is instantiated.
template <int D, int B, int K, int M, int W, int C, uint32_t MO, bool STATE>
static constexpr auto arm_fn() {
  if constexpr (kCarves<D, MO>)
    return &carry_kernel<D, B, K, M, W, C, carve_level<D, MO>, write_carve_level<D, MO>, MO, STATE>;
  else
    return &carry_split_kernel<D, B, K, M, W, C, STATE>;
}

template <int D, int B, int K, int M, int W, int C, uint32_t MO>
static constexpr std::array<int64_t, 3> arm_geom() {
  if constexpr (kCarves<D, MO>) {
    using G = CarryGeom<D, B, K, M, W, C, carve_level<D, MO>>;
    return {(int64_t)G::bytes, (int64_t)G::kStages, (int64_t)G::cta_target};
  } else {
    using G = SplitGeom<D, B, K, M, W, C>;
    return {(int64_t)G::bytes, (int64_t)G::kStages, (int64_t)G::cta_target};
  }
}

template <int D, int B, int K, int M, int W, int C, int CV, uint32_t MO>
static cudaError_t launch(const __nv_bfloat16* pr, const __nv_bfloat16* pw, const __nv_bfloat16* gw,
                          const __nv_bfloat16* v, const uint32_t* rows, float* o, float* den,
                          int BH, int L, int tok_words, int v_row_bytes, const float* s_in,
                          float* s_out, cudaStream_t stream) {
  static_assert(CV == 1, "the innermost level's leaf split is the built carve");
  const auto fn = arm_fn<D, B, K, M, W, C, MO, true>();
  constexpr size_t sm = (size_t)arm_geom<D, B, K, M, W, C, MO>()[0];
  static bool configured = false;
  if (!configured) {
    const cudaError_t e =
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    if (e != cudaSuccess) return e;
    configured = true;
  }
  const dim3 grid((unsigned)BoxPlan<D, B, K, M>::kOwners, (unsigned)BH);
  fn<<<grid, kThreads, sm, stream>>>(pr, pw, gw, v, rows, o, den, L, tok_words, v_row_bytes, s_in,
                                     s_out);
  return cudaGetLastError();
}

template <int D, int B, int K, int M, int W, int C, int CV, uint32_t MO>
static std::array<int64_t, 11> census() {
  static_assert(CV == 1, "the innermost level's leaf split is the built carve");
  cudaFuncAttributes a{};
  if (cudaFuncGetAttributes(&a, arm_fn<D, B, K, M, W, C, MO, true>()) != cudaSuccess)
    return {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
  constexpr auto g = arm_geom<D, B, K, M, W, C, MO>();
  return {a.numRegs,
          (int64_t)a.sharedSizeBytes,
          a.binaryVersion,
          g[0],
          (int64_t)a.maxThreadsPerBlock,
          (int64_t)a.localSizeBytes,
          (int64_t)BoxPlan<D, B, K, M>::kBC,
          (int64_t)BoxPlan<D, B, K, M>::kOwners,
          g[1],
          g[2],
          (int64_t)carve_level<D, MO>};
}

static bool is_arm(const std::vector<int64_t>& widths, int D, int B) {
  if ((int)widths.size() != D) return false;
  for (auto w : widths)
    if (w != B) return false;
  return true;
}

//: THE TABLE, AS A TENSOR: `[BH, 2, entries, L/32]` packed bits, read rows then
//: write rows, one row per (level, owner-group).  Also a SEPARATE ENTRY POINT, so
//: the grain the kernel consumes and the grain the seam names can be compared bit
//: for bit (`tests/k31/test_owner_rows.py`).
template <class RP>
static torch::Tensor build_rows(const torch::Tensor& pr, const torch::Tensor& pw, int BH, int L) {
  auto tbl = torch::empty({BH, 2, (long)RP::entries, rola::chunk::tok_words(L)},
                          pr.options().dtype(torch::kInt32));
  C10_CUDA_CHECK((rola::chunk::launch_span_facts_arm<RP::depth, RP::kWidth, RP::span_at(0),
                                                     RP::span_at(1), RP::span_at(2), RP::span_at(3),
                                                     rola::chunk::kPrefillStateless>(
      (const __nv_bfloat16*)pr.data_ptr(), (const __nv_bfloat16*)pw.data_ptr(),
      reinterpret_cast<uint32_t*>(tbl.data_ptr()), BH, L, at::cuda::getCurrentCUDAStream())));
  return tbl;
}

//: the ROW PLAN the arm's body consumes: the carve body reads its own group's row
//: at the carve level, the channel-split body the owner's own row.
template <int D, int B, int K, int M, uint32_t MO>
static torch::Tensor build_arm_rows(const torch::Tensor& pr, const torch::Tensor& pw, int BH,
                                    int L) {
  if constexpr (kTwoCarve<D, MO>)
    return build_rows<TwoCarveRowPlan<D, B, K, M, write_carve_level<D, MO>, carve_level<D, MO>>>(
        pr, pw, BH, L);
  else if constexpr (kCarves<D, MO>)
    return build_rows<CarveRowPlan<D, B, K, M, carve_level<D, MO>>>(pr, pw, BH, L);
  else
    return build_rows<OwnerRows<D, B, K, M>>(pr, pw, BH, L);
}

std::vector<torch::Tensor> forward_inter(torch::Tensor pr, torch::Tensor pw, torch::Tensor gw,
                                         torch::Tensor v, std::vector<int64_t> widths, int64_t k,
                                         int64_t m, int64_t window, int64_t chunk, int64_t carve,
                                         int64_t level_modes, c10::optional<torch::Tensor> s_in) {
  TORCH_CHECK(pr.is_cuda() && pw.is_cuda() && gw.is_cuda() && v.is_cuda(), "cuda tensors only");
  TORCH_CHECK(pr.scalar_type() == torch::kBFloat16 && pw.scalar_type() == torch::kBFloat16
                  && gw.scalar_type() == torch::kBFloat16 && v.scalar_type() == torch::kBFloat16,
              "the operand planes are bf16");
  TORCH_CHECK(pr.is_contiguous() && pw.is_contiguous() && v.is_contiguous(), "contiguous planes");
  //: `pr`/`pw` are `[BH, L, sum_l B_l]`; `v` is `[B, T, H, d_v]`; `gw` is `[BH, L]`.
  const int BH = (int)pr.size(0);
  const int L = (int)pr.size(1);
  const int d_v = (int)v.size(-1);
  TORCH_CHECK(d_v == kDv, "R1 builds d_v = 64 only");
  const int v_row_bytes = (int)(v.size(2) * d_v * v.element_size());

  int64_t N = 1;
  for (auto w : widths) N *= w;
  auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(v.device());
  auto o = torch::zeros({BH, L, (int64_t)kDv}, opt);
  auto den = torch::zeros({BH, L}, opt);
  auto s_out = torch::zeros({BH, N / kAtomLeaves, (int64_t)kAtomLeaves, (int64_t)kCols}, opt);

  const float* sin_p = s_in.has_value() ? s_in->data_ptr<float>() : nullptr;
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
#define CARRY_DISPATCH(D_, B_, K_, M_, W_, C_, CV_, MO_)                                    \
  if (is_arm(widths, D_, B_) && k == K_ && m == M_ && window == W_ && chunk == C_           \
      && carve == CV_ && mo == MO_) {                                                       \
    matched = true;                                                                         \
    rows = build_arm_rows<D_, B_, K_, M_, (uint32_t)(MO_)>(pr, pw, BH, L);                  \
    err = launch<D_, B_, K_, M_, W_, C_, CV_, (uint32_t)(MO_)>(                             \
        (const __nv_bfloat16*)pr.data_ptr(), (const __nv_bfloat16*)pw.data_ptr(),           \
        (const __nv_bfloat16*)gw.data_ptr(), (const __nv_bfloat16*)v.data_ptr(),            \
        reinterpret_cast<const uint32_t*>(rows.data_ptr()), o.data_ptr<float>(),            \
        den.data_ptr<float>(), BH, L, tok_words, v_row_bytes, sin_p,                        \
        s_out.data_ptr<float>(), stream);                                                   \
  } else
    CARRY_ARMS_X(CARRY_DISPATCH) {}
#undef CARRY_DISPATCH
    return matched;
  };
  //: THE DECLARATION IS RESOLVED AT THIS SEAM (law 3).  A declaration this family
  //: has no body built for is served DENSE_BOTH, which is always correct and only
  //: ever slower; the TOPOLOGY is never degraded -- an unbuilt (D, B, k, m, W, C)
  //: is refused outright.
  if (!run(level_modes)) TORCH_CHECK(run(0), "unbuilt carry arm");
  C10_CUDA_CHECK(err);
  return {o, den, s_out};
}

__global__ void stamp_kernel_entry(int* out) {
  out[0] = kCarryBuildMagic;
  out[1] = kWarps * 1000 + kDv;
  out[2] = design::kCapsDigest;
}

std::vector<int64_t> build_stamp(std::vector<int64_t> widths, int64_t k, int64_t m, int64_t window,
                                 int64_t chunk, int64_t carve, int64_t level_modes) {
  auto t = torch::zeros({3}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  stamp_kernel_entry<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(t.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto h = t.cpu();
  std::vector<int64_t> r{h.data_ptr<int>()[0], h.data_ptr<int>()[1], h.data_ptr<int>()[2]};
#define CARRY_CENSUS(D_, B_, K_, M_, W_, C_, CV_, MO_)                                      \
  if (is_arm(widths, D_, B_) && k == K_ && m == M_ && window == W_ && chunk == C_           \
      && carve == CV_ && level_modes == MO_) {                                              \
    for (auto x : census<D_, B_, K_, M_, W_, C_, CV_, (uint32_t)(MO_)>()) r.push_back(x);   \
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
#define CARRY_ROWS(D_, B_, K_, M_, W_, C_, CV_, MO_)                              \
  if (is_arm(widths, D_, B_) && k == K_ && m == M_ && (MO_) == 0)                  \
    return carve ? build_rows<CarveRowPlan<D_, B_, K_, M_, D_ - 1>>(pr, pw, BH, L) \
                 : build_rows<OwnerRows<D_, B_, K_, M_>>(pr, pw, BH, L);
  CARRY_ARMS_X(CARRY_ROWS)
#undef CARRY_ROWS
  TORCH_CHECK(false, "unbuilt carry arm");
}

std::vector<std::vector<int64_t>> arms() {
  std::vector<std::vector<int64_t>> out;
#define CARRY_LIST(D_, B_, K_, M_, W_, C_, CV_, MO_)                                        \
  out.push_back({D_, B_, K_, M_, W_, C_, CV_, MO_, (int64_t)BoxPlan<D_, B_, K_, M_>::kBC,   \
                 (int64_t)BoxPlan<D_, B_, K_, M_>::kOwners,                                 \
                 arm_geom<D_, B_, K_, M_, W_, C_, (uint32_t)(MO_)>()[0],                    \
                 (int64_t)carve_level<D_, (uint32_t)(MO_)>});
  CARRY_ARMS_X(CARRY_LIST)
#undef CARRY_LIST
  return out;
}

}  // namespace rola::carry
