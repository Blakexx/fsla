// csrc/rola/src/carry/carry.cu -- K31 R1: the carry family's ONE translation unit.
//
// Dispatch, torch entries and the built arm list.  Each arm DECLARES its two carves
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
// The first five hold ONE box of `BC = 128` -- the register budget's number, which
// does not move with depth -- so they differ only in how the same 128 leaves cut
// into digits, and `deep-4` is the topology whose atom rectangle cannot be a single
// level (`B = 8` is narrower than an atom).  `carve-4` is the MULTI-LEVEL CARVE arm
// and the one whose owner is the whole topology: its 256 leaves are cut into EIGHT
// streams of 32 by `SW = [1,4,2,4]` and `SR = [4,1,4,2]`, two carved levels per
// side, one warp per stream -- so it is also the arm that runs eight executors.
// `k = 4, m = 1` is the only capacity reaching the uniform box `(4,4,4,4)`.
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

//: the plan the CARVED body's rows are voted under: the same table refined at every
//: level EITHER side carves, so each stream reads its own range's rows and a side
//: wanting the owner's run ORs the rows it already holds (`CarveRows`).
template <class SW, class SR>
using CarveRowPlan = rola::chunk::SpanPlan<CarveRows<SW, SR>::kLevels, CarveRows<SW, SR>::kWidth,
                                           CarveRows<SW, SR>::s(0), CarveRows<SW, SR>::s(1),
                                           CarveRows<SW, SR>::s(2), CarveRows<SW, SR>::s(3)>;

//: THE PER-LEVEL MODE DECLARATION (K33 law 3) is two bits per level with level 0 in
//: the low pair: bit 0 = the READ side is value-sparse there, bit 1 = the WRITE side.
//: `DENSE_BOTH` everywhere is always correct and only ever slower.  The kernel reads
//: it through `mode_declares` (carry_kernel.cuh); this TU only passes it through, so
//: the encoding is named there and nowhere else.

//: THE BUILT SET, as one macro so the dispatch, the census and the arm list cannot
//: drift apart -- the shipped family's `CHUNK_ARMS_X` discipline.
//: MO_ is the per-level declaration this arm is BUILT for.  `ALT` at depth `D` is
//: write-sparse on the even levels and read-sparse on the odd ones, i.e.
//: `sum_l mode_l << 2l` = 6 / 38 / 102 at D = 2 / 3 / 4.
//:
//: `SW*_` and `SR*_` ARE THE ARM'S TWO CARVES, DECLARED: the per-level digit span
//: one STREAM owns on the write side and on the read side, with a level past `D`
//: carrying `B`.  Full spans on both sides declare NO carve and take the
//: channel-split body; anything else is a stream grid, and the two sides' grids must
//: agree in size because this stage maps one stream to one warp.  A side carves
//: where the declaration marks it sparse, which is what the private order's rule
//: then reads off the vector (`SpanMap`).
#define CARRY_ARM_0(X_) X_(2, 256, 4, 8, 384, 32, 1, 0, 8, 16, 256, 256, 8, 16, 256, 256)
#define CARRY_ARM_1(X_) X_(2, 256, 4, 8, 384, 32, 1, 6, 2, 16, 256, 256, 8, 4, 256, 256)
#define CARRY_ARM_2(X_) X_(2, 256, 4, 8, 512, 32, 1, 0, 8, 16, 256, 256, 8, 16, 256, 256)
#define CARRY_ARM_3(X_) X_(2, 256, 4, 8, 512, 32, 1, 6, 2, 16, 256, 256, 8, 4, 256, 256)
#define CARRY_ARM_4(X_) X_(2, 64, 4, 8, 64, 32, 1, 0, 8, 16, 64, 64, 8, 16, 64, 64)
#define CARRY_ARM_5(X_) X_(2, 64, 4, 8, 64, 32, 1, 6, 2, 16, 64, 64, 8, 4, 64, 64)
#define CARRY_ARM_6(X_) X_(3, 16, 2, 16, 64, 32, 1, 0, 2, 4, 16, 16, 2, 4, 16, 16)
#define CARRY_ARM_7(X_) X_(3, 16, 2, 16, 64, 32, 1, 38, 2, 4, 16, 16, 2, 4, 16, 16)
#define CARRY_ARM_8(X_) X_(4, 8, 2, 8, 64, 32, 1, 0, 2, 2, 4, 8, 2, 2, 4, 8)
#define CARRY_ARM_9(X_) X_(4, 8, 2, 8, 64, 32, 1, 102, 2, 2, 1, 8, 2, 2, 4, 2)
#define CARRY_ARM_10(X_) X_(4, 4, 4, 1, 64, 32, 1, 102, 1, 4, 2, 4, 4, 1, 4, 2)

#define CARRY_ARMS_ALL(X_)                                                \
  CARRY_ARM_0(X_) CARRY_ARM_1(X_) CARRY_ARM_2(X_) CARRY_ARM_3(X_)         \
  CARRY_ARM_4(X_) CARRY_ARM_5(X_) CARRY_ARM_6(X_) CARRY_ARM_7(X_)         \
  CARRY_ARM_8(X_) CARRY_ARM_9(X_) CARRY_ARM_10(X_)

//: THE ITERATION SUBSET, A BUILD KNOB AND NOTHING MORE.  Compiling eleven arms to
//: test one is the difference between a minute and an hour, so a build may define
//: `ROLA_CARRY_ARMS(X_)` as any list of the rows above and get exactly those.  The
//: DEFAULT IS EVERY DECLARED ARM, so a plain build is the closed-world one; a subset
//: binary is SELF-IDENTIFYING, because `arms()` lists what was built and every entry
//: point refuses an arm it does not carry.  A gate build never sets it.
//: `ROLA_CARRY_ARMS=10 python -m pip install -e .` (setup.py, `docs/build.md`).
#ifndef ROLA_CARRY_ARMS
#define ROLA_CARRY_ARMS(X_) CARRY_ARMS_ALL(X_)
#endif
#define CARRY_ARMS_X(X_) ROLA_CARRY_ARMS(X_)

//: THE ARM'S SPAN VECTORS, named once: a side's carve is a per-level span vector,
//: and a level past `D` carries `B`, the plan's identity.
template <int D, int B, int K, int M, int S0, int S1, int S2, int S3>
using ArmSpans = StreamSpans<D, B, K, M, S0, S1, S2, S3>;

//: READING ONE ROW OF THE LIST.  The row is long -- a topology, a window shape, a
//: declaration and two span vectors -- so the pieces a consumer wants are taken from
//: it HERE and nowhere else, and no consumer re-spells the row's shape.
#define CARRY_ARM(D_, B_, K_, M_, W_, C_, CV_, MO_, SW0_, SW1_, SW2_, SW3_, SR0_, SR1_, SR2_, SR3_)                                                                       \
  Arm<D_, B_, K_, M_, W_, C_, SW0_, SW1_, SW2_, SW3_, SR0_, SR1_, SR2_, SR3_, (uint32_t)(MO_)>
#define CARRY_MO(D_, B_, K_, M_, W_, C_, CV_, MO_, ...) (MO_)
#define CARRY_ROW_HEAD(D_, B_, K_, M_, W_, C_, CV_, MO_, ...) D_, B_, K_, M_, W_, C_, CV_, MO_
//: the seam's key, matched against the call's `(topology, k, m, W, C, carve)`.
#define CARRY_MATCH(D_, B_, K_, M_, W_, C_, CV_, ...) \
  (is_arm(widths, D_, B_) && k == K_ && m == M_ && window == W_ && chunk == C_ && carve == CV_)

//: `CV` is the SEAM's carve scalar, not a kernel axis: it names the one ruled carve
//: so a caller that asks for the deleted one is refused rather than served something
//: else.  Asserted once over the whole list, where the list is.
#define CARRY_CV_GATE(D_, B_, K_, M_, W_, C_, CV_, ...) \
  static_assert((CV_) == 1, "the innermost level's leaf split is the built carve");
CARRY_ARMS_X(CARRY_CV_GATE)
#undef CARRY_CV_GATE

//: THE OWNER'S OWN SPANS as a vector, and the SINGLE-LEVEL LEAF SPLIT the seam's
//: `owner_rows(carve=True)` diagnostic names -- the innermost level's owner run over
//: `kSeamCarveStreams` streams, which is the grain that entry point documents.
static constexpr int kSeamCarveStreams = 4;
template <int D, int B, int K, int M>
constexpr int owner_span(int l) {
  return l < D ? BoxPlan<D, B, K, M>::s(l) : B;
}
template <int D, int B, int K, int M>
constexpr int split_span(int l) {
  return owner_span<D, B, K, M>(l) / (l == D - 1 ? kSeamCarveStreams : 1);
}
template <int D, int B, int K, int M>
using OwnerSpans = ArmSpans<D, B, K, M, owner_span<D, B, K, M>(0), owner_span<D, B, K, M>(1),
                            owner_span<D, B, K, M>(2), owner_span<D, B, K, M>(3)>;
template <int D, int B, int K, int M>
using LeafSplitSpans = ArmSpans<D, B, K, M, split_span<D, B, K, M>(0), split_span<D, B, K, M>(1),
                                split_span<D, B, K, M>(2), split_span<D, B, K, M>(3)>;

//: A CARVED ARM is one whose streams are finer than the owner's box on either side;
//: an arm that carves nothing takes the channel-split body, which holds the whole
//: box in every warp and splits the CHANNELS instead.
template <class SW, class SR>
static constexpr bool kCarves = (SW::kStreams > 1) || (SR::kStreams > 1);

//: ONE ARM, AS ONE TYPE: the topology, the window shape, the declaration and the two
//: carves.  The body, the geometry, the row plan and the census all read it, so an
//: arm cannot be described two ways.
template <int D, int B, int K, int M, int W, int C, int W0, int W1, int W2, int W3, int R0, int R1,
          int R2, int R3, uint32_t MO>
struct Arm {
  using BP = BoxPlan<D, B, K, M>;
  using SW = ArmSpans<D, B, K, M, W0, W1, W2, W3>;
  using SR = ArmSpans<D, B, K, M, R0, R1, R2, R3>;
  static constexpr bool carved = kCarves<SW, SR>;
  //: the READ side's innermost carved level, which the seam reports as the arm's
  //: carve level, and `-1` for a body that carves nothing.
  static constexpr int read_carve(int l) {
    return l < 0 ? -1 : (SR::carves(l) ? l : read_carve(l - 1));
  }
  static constexpr int kReadCarve = read_carve(D - 1);

  template <bool STATE>
  static constexpr auto fn() {
    if constexpr (carved)
      return &carry_kernel<D, B, K, M, W, C, W0, W1, W2, W3, R0, R1, R2, R3, MO, STATE>;
    else
      return &carry_split_kernel<D, B, K, M, W, C, STATE>;
  }
  //: `{smem bytes, ring depth, CTA target, THREADS}` -- the launch reads the last,
  //: because the executor count is the stream count and no longer a file constant.
  static constexpr std::array<int64_t, 4> geom() {
    if constexpr (carved) {
      using G = CarryGeom<D, B, K, M, W, C, R0, R1, R2, R3>;
      return {(int64_t)G::bytes, (int64_t)G::kStages, (int64_t)G::cta_target, (int64_t)G::kThreads};
    } else {
      using G = SplitGeom<D, B, K, M, W, C>;
      return {(int64_t)G::bytes, (int64_t)G::kStages, (int64_t)G::cta_target, (int64_t)kThreads};
    }
  }
};

template <class A>
static cudaError_t launch(const __nv_bfloat16* pr, const __nv_bfloat16* pw, const __nv_bfloat16* gw,
                          const __nv_bfloat16* v, const uint32_t* rows, float* o, float* den,
                          int BH, int L, int tok_words, int v_row_bytes, const float* s_in,
                          float* s_out, cudaStream_t stream) {
  const auto fn = A::template fn<true>();
  constexpr size_t sm = (size_t)A::geom()[0];
  static bool configured = false;
  if (!configured) {
    const cudaError_t e =
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    if (e != cudaSuccess) return e;
    configured = true;
  }
  const dim3 grid((unsigned)A::BP::kOwners, (unsigned)BH);
  fn<<<grid, (unsigned)A::geom()[3], sm, stream>>>(pr, pw, gw, v, rows, o, den, L, tok_words,
                                                   v_row_bytes, s_in, s_out);
  return cudaGetLastError();
}

template <class A>
static std::array<int64_t, 11> census() {
  cudaFuncAttributes a{};
  if (cudaFuncGetAttributes(&a, A::template fn<true>()) != cudaSuccess)
    return {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
  constexpr auto g = A::geom();
  return {a.numRegs,
          (int64_t)a.sharedSizeBytes,
          a.binaryVersion,
          g[0],
          (int64_t)a.maxThreadsPerBlock,
          (int64_t)a.localSizeBytes,
          (int64_t)A::BP::kBC,
          (int64_t)A::BP::kOwners,
          g[1],
          g[2],
          (int64_t)A::kReadCarve};
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

//: the ROW PLAN the arm's body consumes: the carved body reads its own stream's
//: range at every level either side carves, the channel-split body the owner's run.
//: An uncarved arm's table IS the owner's, because the span vectors are then the
//: owner's spans -- one statement, not two paths.
template <class A>
static torch::Tensor build_arm_rows(const torch::Tensor& pr, const torch::Tensor& pw, int BH,
                                    int L) {
  return build_rows<CarveRowPlan<typename A::SW, typename A::SR>>(pr, pw, BH, L);
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
#define CARRY_DISPATCH(...)                                                                 \
  if (CARRY_MATCH(__VA_ARGS__) && mo == CARRY_MO(__VA_ARGS__)) {                            \
    matched = true;                                                                         \
    rows = build_arm_rows<CARRY_ARM(__VA_ARGS__)>(pr, pw, BH, L);                           \
    err = launch<CARRY_ARM(__VA_ARGS__)>(                                                   \
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
#define CARRY_CENSUS(...)                                                                   \
  if (CARRY_MATCH(__VA_ARGS__) && level_modes == CARRY_MO(__VA_ARGS__)) {                   \
    for (auto x : census<CARRY_ARM(__VA_ARGS__)>()) r.push_back(x);                         \
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
#define CARRY_ROWS(D_, B_, K_, M_, W_, C_, CV_, MO_, ...)                                      \
  if constexpr ((MO_) == 0) {                                                                  \
    if (is_arm(widths, D_, B_) && k == K_ && m == M_)                                          \
      return carve ? build_rows<CarveRowPlan<OwnerSpans<D_, B_, K_, M_>,                       \
                                             LeafSplitSpans<D_, B_, K_, M_>>>(pr, pw, BH, L)   \
                   : build_rows<OwnerRows<D_, B_, K_, M_>>(pr, pw, BH, L);                     \
  }
  CARRY_ARMS_X(CARRY_ROWS)
#undef CARRY_ROWS
  TORCH_CHECK(false, "unbuilt carry arm");
}

std::vector<std::vector<int64_t>> arms() {
  std::vector<std::vector<int64_t>> out;
#define CARRY_LIST(...)                                                                     \
  out.push_back({CARRY_ROW_HEAD(__VA_ARGS__), CARRY_ARM(__VA_ARGS__)::BP::kBC,              \
                 CARRY_ARM(__VA_ARGS__)::BP::kOwners, CARRY_ARM(__VA_ARGS__)::geom()[0],    \
                 (int64_t)CARRY_ARM(__VA_ARGS__)::kReadCarve});
  CARRY_ARMS_X(CARRY_LIST)
#undef CARRY_LIST
  return out;
}

}  // namespace rola::carry
