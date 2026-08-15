// csrc/rola/src/chunk/chunk.cu -- the chunk consumer's DISPATCH and torch
// entries (adopted from the ladder's rung 6G).  This TU never touches
// a kernel symbol: arms arrive as extern declarations satisfied by the
// instantiation shards, which is what keeps one arm's codegen in one object.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <array>
#include <mutex>
#include <set>
#include <vector>

#include "chunk_kernel.cuh"

namespace rola::chunk {

//: THE RUNTIME HALF OF THE CLOSED-WORLD ARCH RULE (setup.py's R4).  Its
//: compile-time half is `arch_caps.cuh`'s `static_assert(tabulated(kArch))`,
//: which fixes the table this binary's shapes were DERIVED from; this is the
//: refusal at first launch on a device that table has no row for, or has a row
//: that the driver disagrees with.  Both halves must stay: a derivation is only
//: as good as the facts it was closed over, and only a fact read off the DEVICE
//: can establish that this is the device they were read for.
//: docs/internals/chunk/arch_caps.md
static void check_arch_table() {
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  //: ONCE PER PROCESS PER DEVICE, on the first launch rather than at import: a
  //: CUDA context as a side effect of `import` costs more than the check.  The
  //: device is recorded only after it passes, so a refusal is raised on every
  //: call rather than once.
  static std::mutex mu;
  static std::set<int> admitted;
  {
    const std::lock_guard<std::mutex> hold(mu);
    if (admitted.count(dev)) return;
  }
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  const int cc = prop.major * 100 + prop.minor * 10;
  TORCH_CHECK(denseref::arch::tabulated(cc),
              "the chunk consumer's capability table has no row for compute capability ",
              prop.major, ".", prop.minor,
              "; add it to caps_of() rather than letting an operation guess which "
              "implementation it has. This binary is closed-world: it runs the "
              "architectures it was measured on and refuses the rest.");
  const denseref::arch::Caps want = denseref::arch::caps_of(cc);
  TORCH_CHECK(denseref::arch::supported(want), "UNSUPPORTED ARCHITECTURE at compute capability ",
              prop.major, ".", prop.minor,
              ": a REQUIRED capability clause of the chunk baseline is false in its own "
              "row. There is no partial support and no slower route to take.");
  int smem_per_sm = 0, smem_per_cta_max = 0;
  C10_CUDA_CHECK(
      cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
  C10_CUDA_CHECK(
      cudaDeviceGetAttribute(&smem_per_cta_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  TORCH_CHECK(want.smem_per_sm == smem_per_sm && want.smem_per_cta_max == smem_per_cta_max
                  && want.regs_per_sm == prop.regsPerMultiprocessor
                  && want.max_threads_per_sm == prop.maxThreadsPerMultiProcessor,
              "the chunk consumer's capability table disagrees with the driver at compute "
              "capability ",
              prop.major, ".", prop.minor, ": table {smem/SM ", want.smem_per_sm, ", smem/CTA ",
              want.smem_per_cta_max, ", regs/SM ", want.regs_per_sm, ", threads/SM ",
              want.max_threads_per_sm, "} vs driver {smem/SM ", smem_per_sm, ", smem/CTA ",
              smem_per_cta_max, ", regs/SM ", prop.regsPerMultiprocessor, ", threads/SM ",
              prop.maxThreadsPerMultiProcessor,
              "}. Every arm's staging depth, CTA target and shared-memory fit compiled "
              "into this binary was derived from the table, so they are wrong for this "
              "device.");
  const std::lock_guard<std::mutex> hold(mu);
  admitted.insert(dev);
}

//: the arm surface, DECLARED only -- definitions live in the instantiation
//: shards (chunk_s*.cu).  chunk.cu must never define one locally.
template <int C, int BC, int D, int B, bool STATE>
cudaError_t launch_chunk_arm(const ChunkParams& p, cudaStream_t stream);
template <int D, int B, int BC>
cudaError_t launch_chunk_table_arm(const __nv_bfloat16* pr, const __nv_bfloat16* pw, uint32_t* tbl,
                                   int BH, int L, cudaStream_t stream);
template <int D, int B>
cudaError_t launch_chunk_atom_table_arm(const __nv_bfloat16* pw, uint32_t* tbl, int BH, int L,
                                        cudaStream_t stream);
template <int D, int B>
cudaError_t launch_chunk_atom_bitmap_arm(const uint32_t* tbl, uint8_t* atom_bits, int BH, int L,
                                         cudaStream_t stream);
template <int D, int B, int BC>
cudaError_t launch_chunk_block_bitmap_arm(const uint32_t* tbl, const int32_t* page_tbl,
                                          uint8_t* block_bits, int BH, int L, cudaStream_t stream);
template <int C, int BC, int D, int B, bool STATE>
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
  check_arch_table();
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
  check_arch_table();
  check_inputs(pr, pw, v, chunk, widths);
  const int BH = v.size(0) * v.size(1), L = v.size(2);
#define CHUNK_TABLE_X(C_, BC_, D_, B_)                                     \
  if (chunk == C_ && bc == BC_ && is_arm(widths, D_, B_))                     \
    return build_table<D_, B_, BC_>(pr, pw, BH, L);
  CHUNK_ARMS_X(CHUNK_TABLE_X)
#undef CHUNK_TABLE_X
  TORCH_CHECK(false, "unbuilt arm: chunk=", chunk, " BC=", bc, " depth=", widths.size());
}

//: THE STATS PASS: the EXACT set of atoms this call's routing writes, as
//: `[BH, N/16]` uint8 -- the argument the planner admits pages from.  Two
//: launches on one stream and no host synchronization; the atom-grain table is a
//: TRANSIENT, allocated and freed here, because nothing else asks for it.
//:
//: NO `C` AND NO `BC`.  The atom rectangle is the invariant quantum, so the
//: bitmap is BC-free and may be taken BEFORE the plan picks a `BC`.  Only the
//: WRITE side is built: reads of never-written atoms are inert
//: (docs/internals/paging/stats.md).
template <int D, int B>
static torch::Tensor build_atom_bits(const torch::Tensor& pw, int BH, int L) {
  using LP = LevelPlan<D, B, kAtomLeaves>;
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto tbl = torch::empty({BH, (long)LP::entries, tok_words(L)}, pw.options().dtype(torch::kInt32));
  auto bits = torch::empty({BH, (long)LP::blocks}, pw.options().dtype(torch::kUInt8));
  C10_CUDA_CHECK((launch_chunk_atom_table_arm<D, B>(
      reinterpret_cast<const __nv_bfloat16*>(pw.data_ptr()),
      reinterpret_cast<uint32_t*>(tbl.data_ptr()), BH, L, stream)));
  C10_CUDA_CHECK((launch_chunk_atom_bitmap_arm<D, B>(
      reinterpret_cast<const uint32_t*>(tbl.data_ptr()),
      reinterpret_cast<uint8_t*>(bits.data_ptr()), BH, L, stream)));
  return bits;
}

torch::Tensor stats(torch::Tensor pw, std::vector<int64_t> widths) {
  check_arch_table();
  TORCH_CHECK(pw.is_cuda() && pw.is_contiguous() && pw.scalar_type() == torch::kBFloat16,
              "the write amplitudes must be a contiguous bf16 CUDA tensor");
  int64_t wsum = 0;
  for (auto w : widths) wsum += w;
  TORCH_CHECK(pw.dim() == 4 && pw.size(3) == wsum,
              "expect packed write amplitudes [B,H,L,sum(widths)]");
  const int BH = (int)(pw.size(0) * pw.size(1)), L = (int)pw.size(2);
#define CHUNK_STATS_X(D_, B_)                                                      \
  if (is_arm(widths, D_, B_)) return build_atom_bits<D_, B_>(pw, BH, L);
  CHUNK_ATOM_ARMS_X(CHUNK_STATS_X)
#undef CHUNK_STATS_X
  TORCH_CHECK(false, "unbuilt stats arm: depth=", widths.size());
}

//: THE STATE PLANE, in either backing.  DENSE is `[BH, N/16, 16, d_v+1]` fp32
//: and contiguous, the same bytes as `[BH, N, d_v+1]` -- the atom-major view is
//: what the kernel addresses and the leaf-major one is what a reader reads.
//: PAGED is `[slots, 16, d_v+1]`: the same page, as many of them as the plan
//: committed, and the table says which atom is which.  It is ALLOCATED BY THE
//: CALLER: this entry launches, it does not own memory the caller carries.
static float* state_plane(const c10::optional<torch::Tensor>& s, const char* what, int BH, long N,
                          bool paged) {
  if (!s.has_value()) return nullptr;
  const torch::Tensor& t = *s;
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == torch::kFloat32, what,
              " must be a contiguous fp32 CUDA tensor");
  //: A SLOT'S PAGE IS A 32-BIT BYTE DISPLACEMENT from the plane's origin, which
  //: is what keeps the widening multiply out of the epilogue's loop bodies
  //: (chunk_kernel.cuh); the plane it displaces into is therefore bounded, and
  //: the bound is stated here rather than discovered as a wrapped address.  It
  //: binds the DENSE backing first -- a paged plane holds the atoms the plan
  //: realized, which is the thing that keeps this reachable at scale.
  TORCH_CHECK(t.numel() * (long)sizeof(float) < (1L << 32), what, " is ",
              t.numel() * (long)sizeof(float),
              " bytes; a page's displacement into it is 32 bits, so a plane must "
              "stay under 4 GiB (",
              (1L << 32) / ((long)kAtomLeaves * (kD + 1) * 4), " atoms)");
  if (paged) {
    TORCH_CHECK(t.dim() == 3 && t.size(1) == kAtomLeaves && t.size(2) == kD + 1 && t.size(0) > 0,
                what, " must be a [slots, ", kAtomLeaves, ", d_v + 1] paged plane; got ",
                t.sizes());
  } else {
    TORCH_CHECK(t.numel() == (long)BH * N * (kD + 1), what,
                " must hold BH * N * (d_v + 1) floats (the [BH, N/16, 16, d_v+1] "
                "dense state plane)");
  }
  return t.data_ptr<float>();
}

//: THE ATOM -> SLOT MAP, `[BH, N/16]` int32, `-1` where the atom has no page.
//: Its ABSENCE is the dense backing, where a slot is its atom -- one translation
//: for both (docs/internals/paging/paging.md).  The slots it names are NOT bounds
//: checked here: the check is a device-side read of the table, and the plan that
//: wrote it is the thing that owns the bound.
static const int32_t* page_table(const c10::optional<torch::Tensor>& t, int BH, long N) {
  if (!t.has_value()) return nullptr;
  TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->scalar_type() == torch::kInt32,
              "the page table must be a contiguous int32 CUDA tensor");
  TORCH_CHECK(t->numel() == (long)BH * (N / kAtomLeaves), "the page table must hold BH * N / ",
              kAtomLeaves, " entries (one per atom); got ", t->numel(), " for BH=", BH, " N=", N);
  return t->data_ptr<int32_t>();
}

//: THE LIVENESS EPILOGUE: `[BH, N/BC]` uint8, bits {READ, WRITE, STATE}, reduced
//: from the union table THIS launch already built.  A SEPARATE ENTRY POINT, on
//: the union table's own rule: a per-call reduction that the sparse grid pays
//: for must be reported on its own line and never folded into a kernel delta.
template <int D, int B, int BC>
static torch::Tensor build_block_bits(const torch::Tensor& tbl, const int32_t* pageTbl, int BH,
                                      int L) {
  auto bits =
      torch::empty({BH, (long)LevelPlan<D, B, BC>::blocks}, tbl.options().dtype(torch::kUInt8));
  C10_CUDA_CHECK((launch_chunk_block_bitmap_arm<D, B, BC>(
      reinterpret_cast<const uint32_t*>(tbl.data_ptr()), pageTbl,
      reinterpret_cast<uint8_t*>(bits.data_ptr()), BH, L, at::cuda::getCurrentCUDAStream())));
  return bits;
}

torch::Tensor block_bits(torch::Tensor tbl, int64_t bc, std::vector<int64_t> widths, int64_t L,
                         c10::optional<torch::Tensor> page_tbl) {
  check_arch_table();
  TORCH_CHECK(tbl.is_cuda() && tbl.is_contiguous() && tbl.scalar_type() == torch::kInt32
                  && tbl.dim() == 4 && tbl.size(1) == 2,
              "the union table is a contiguous int32 [BH, 2, entries, L/32] tensor");
  TORCH_CHECK(tbl.size(3) == tok_words((int)L), "the union table was built for a different L");
  const int BH = (int)tbl.size(0);
  long N = 1;
  for (auto w : widths) N *= w;
  const int32_t* const pageTbl = page_table(page_tbl, BH, N);
  //: NO `C` AXIS: the table this reduces is the (D, B, BC) one, so the arm walk
  //: matches on the block grain alone and an arm list repeating a grain at
  //: several chunk lengths resolves to the one instantiation.
#define CHUNK_BLOCK_BITS_X(C_, BC_, D_, B_)                                \
  if (bc == BC_ && is_arm(widths, D_, B_)) {                               \
    const long want = (long)LevelPlan<D_, B_, BC_>::entries;               \
    TORCH_CHECK(tbl.size(2) == want,                                       \
                "the union table was built for a different arm");          \
    return build_block_bits<D_, B_, BC_>(tbl, pageTbl, BH, (int)L);        \
  }
  CHUNK_ARMS_X(CHUNK_BLOCK_BITS_X)
#undef CHUNK_BLOCK_BITS_X
  TORCH_CHECK(false, "unbuilt arm: BC=", bc, " depth=", widths.size());
}

std::vector<torch::Tensor> forward_raw(torch::Tensor pr, torch::Tensor pw, torch::Tensor gw,
                                       torch::Tensor v, torch::Tensor tbl, int64_t chunk,
                                       int64_t bc, std::vector<int64_t> widths,
                                       c10::optional<torch::Tensor> s_in,
                                       c10::optional<torch::Tensor> s_out,
                                       c10::optional<torch::Tensor> page_tbl) {
  check_arch_table();
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
  long N = 1;
  for (auto w : widths) N *= w;
  const int32_t* const pageTbl = page_table(page_tbl, BH, N);
  const float* const sIn = state_plane(s_in, "the entry state", BH, N, pageTbl != nullptr);
  float* const sOut = state_plane(s_out, "the final state", BH, N, pageTbl != nullptr);
  TORCH_CHECK(pageTbl == nullptr || sIn != nullptr || sOut != nullptr,
              "a page table resolves the state plane's atoms; a call that "
              "carries no state has nothing for it to resolve");
  //: THE STATE VARIANT IS A STRUCTURAL CHOICE, taken here: a call that carries
  //: no state runs the kernel that has no state code in it at all
  //: (docs/internals/chunk/chunk_kernel.md).
#define CHUNK_DISPATCH_X(C_, BC_, D_, B_)                                   \
  if (chunk == C_ && bc == BC_ && is_arm(widths, D_, B_)) {                    \
    const long want = (long)BH * 2 * LevelPlan<D_, B_, BC_>::entries *        \
                      tok_words(L);                                          \
    TORCH_CHECK(tbl.numel() == want,                                           \
                "the union table was built for a different arm");              \
    ChunkParams p_{reinterpret_cast<const __nv_bfloat16 *>(pr.data_ptr()),     \
                   reinterpret_cast<const __nv_bfloat16 *>(pw.data_ptr()),     \
                   reinterpret_cast<const __nv_bfloat16 *>(gw.data_ptr()),     \
                   reinterpret_cast<const __nv_bfloat16 *>(v.data_ptr()),      \
                   reinterpret_cast<const uint32_t *>(tbl.data_ptr()),         \
                   o.data_ptr<float>(), den.data_ptr<float>(),                 \
                   sIn,             sOut,             pageTbl,                 \
                   BH,              L};                                        \
    const auto stream_ = at::cuda::getCurrentCUDAStream();                     \
    C10_CUDA_CHECK((sIn != nullptr || sOut != nullptr)                         \
                       ? (launch_chunk_arm<C_, BC_, D_, B_, true>(p_, stream_)) \
                       : (launch_chunk_arm<C_, BC_, D_, B_, false>(p_,          \
                                                                   stream_)));  \
    return {o, den};                                                           \
  }
  CHUNK_ARMS_X(CHUNK_DISPATCH_X)
#undef CHUNK_DISPATCH_X
  TORCH_CHECK(false, "unbuilt arm: chunk=", chunk, " BC=", bc, " depth=", widths.size());
}

torch::Tensor forward(torch::Tensor pr, torch::Tensor pw, torch::Tensor gw, torch::Tensor v,
                      int64_t chunk, int64_t bc, std::vector<int64_t> widths, double eps,
                      c10::optional<torch::Tensor> s_in, c10::optional<torch::Tensor> s_out,
                      c10::optional<torch::Tensor> page_tbl) {
  auto tbl = union_table(pr, pw, v, chunk, bc, widths);
  auto nd = forward_raw(pr, pw, gw, v, tbl, chunk, bc, widths, s_in, s_out, page_tbl);
  return divide_pass(nd[0], nd[1], eps);
}

//: THE RING'S DEPTH IS PART OF THE BUILD'S IDENTITY, so a measurement can never
//: be attributed to a depth the running binary does not have.
__global__ void stamp_kernel(int* out) {
  out[0] = kBuildMagic + kFormWords;
  out[1] = (kGroupThreads / 32) * 1000 + kD;
  out[2] = design::kCapsDigest;
}

std::vector<int64_t> build_stamp(int64_t chunk, int64_t bc, std::vector<int64_t> widths) {
  check_arch_table();
  auto t = torch::zeros({3}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  stamp_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(t.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto h = t.cpu();
  std::vector<int64_t> r{h.data_ptr<int>()[0], h.data_ptr<int>()[1], h.data_ptr<int>()[2]};
  //: BOTH STATE VARIANTS ARE REPORTED, the stateless one first: they are two
  //: kernels with two register censuses, and a stamp that named only one of them
  //: would attribute a measurement to a kernel the launch may not have run.
#define CHUNK_CENSUS_X(C_, BC_, D_, B_)                                         \
  if (chunk == C_ && bc == BC_ && is_arm(widths, D_, B_)) {                        \
    for (auto x : query_chunk_census_arm<C_, BC_, D_, B_, false>()) r.push_back(x); \
    for (auto x : query_chunk_census_arm<C_, BC_, D_, B_, true>()) r.push_back(x);  \
    return r;                                                                      \
  }
  CHUNK_ARMS_X(CHUNK_CENSUS_X)
#undef CHUNK_CENSUS_X
  TORCH_CHECK(false, "unbuilt arm");
  return r;
}

//: The built set with every DERIVED quantity of an arm reported beside it: the
//: plan's depth and width, the block count, the block's inner/outer axes, the
//: staging depth and the CTA target.
std::vector<std::vector<int64_t>> arms() {
  std::vector<std::vector<int64_t>> out;
#define CHUNK_LIST_X(C_, BC_, D_, B_)                                      \
  out.push_back({C_, BC_, D_, B_,                              \
                 (int64_t)LevelPlan<D_, B_, BC_>::blocks,                     \
                 (int64_t)(BC_ / LevelPlan<D_, B_, BC_>::span_at(D_ - 1)),    \
                 (int64_t)LevelPlan<D_, B_, BC_>::span_at(D_ - 1),            \
                 (int64_t)Geom<C_, BC_>::stages,                          \
                 (int64_t)Geom<C_, BC_>::bytes,                           \
                 (int64_t)kCtaTarget<C_, BC_>,                            \
                 (int64_t)LevelPlan<D_, B_, BC_>::entries});
  CHUNK_ARMS_X(CHUNK_LIST_X)
#undef CHUNK_LIST_X
  return out;
}

}  // namespace rola::chunk
