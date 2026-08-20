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
#include <map>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

#include "../dispatch_switch.cuh"
#include "chunk_kernel.cuh"

namespace rola::chunk {

//: THE RUNTIME HALF OF THE CLOSED-WORLD ARCH RULE (setup.py's R4).  Its
//: compile-time half is `arch_caps.cuh`'s `static_assert(tabulated(kArch))`,
//: which fixes the table this binary's shapes were DERIVED from; this is the
//: refusal at first launch on a device that table has no row for, or has a row
//: that the driver disagrees with.  Both halves must stay: a derivation is only
//: as good as the facts it was closed over, and only a fact read off the DEVICE
//: can establish that this is the device they were read for.
//: docs/internals/facts/arch_caps.md
//: NOT TU-LOCAL: the adjoint's translation unit launches too, and a second
//: copy of this refusal is a second thing that can drift from the table.
void check_arch_table() {
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
template <int D, int B, int BC, int MASK>
cudaError_t launch_chunk_facts_arm(const __nv_bfloat16* pr, const __nv_bfloat16* pw, uint32_t* tbl,
                                   uint32_t* atom_tbl, int BH, int L, cudaStream_t stream);
template <int D, int B>
cudaError_t launch_chunk_atom_bitmap_arm(const uint32_t* tbl, uint8_t* atom_bits, int BH, int L,
                                         cudaStream_t stream);
template <int D, int B, int BC>
cudaError_t launch_chunk_block_bitmap_arm(const uint32_t* tbl, const int32_t* page_tbl,
                                          uint8_t* block_bits, int BH, int L, cudaStream_t stream);
template <int C, int BC, int D, int B, bool STATE>
std::array<int64_t, 9> query_chunk_census_arm();
//: THE KERNEL AXIS, RESOLVED: the carried state's I/O is compiled in or out, and
//: which variant a call wants is read off its pointers.
template <int C, int BC, int D, int B>
static cudaError_t launch_variant(const ChunkParams& p, cudaStream_t stream) {
  const bool state = p.s_in != nullptr || p.s_out != nullptr;
  return state ? launch_chunk_arm<C, BC, D, B, true>(p, stream)
               : launch_chunk_arm<C, BC, D, B, false>(p, stream);
}

//: THE FACTS, AS TENSORS: the union table `[B*H, 2, entries, L/32]` of packed bits,
//: sides apart because the cook wants them apart, and -- when this call has
//: residency to plan -- the atom-grain write bitmap `[B*H, N/16]` the page plan
//: admits from.  ONE PASS over the amplitude planes emits both row sets; the
//: atom-grain TABLE is a transient, allocated and freed here, because nothing else
//: asks for it.  The bitmap's own launch stays keyed `(D, B)`, so the published fact
//: is BC-free however this pass was templated (docs/internals/paging/stats.md).
template <int D, int B, int BC>
static std::pair<torch::Tensor, c10::optional<torch::Tensor>> build_facts(const torch::Tensor& pr,
                                                                          const torch::Tensor& pw,
                                                                          int BH, int L, int mask,
                                                                          bool atoms) {
  using AP = LevelPlan<D, B, kAtomLeaves>;
  constexpr int E = LevelPlan<D, B, BC>::entries;
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto tbl = torch::empty({BH, 2, E, tok_words(L)}, pr.options().dtype(torch::kInt32));
  auto atom_tbl =
      atoms ? torch::empty({BH, (long)AP::entries, tok_words(L)}, pr.options().dtype(torch::kInt32))
            : torch::Tensor();
  int_switch<kPrefillStateless, kPrefillStateful>(
      mask, "the chunk facts pass' call class", [&](auto MASK) {
        C10_CUDA_CHECK((launch_chunk_facts_arm<D, B, BC, decltype(MASK)::value>(
            reinterpret_cast<const __nv_bfloat16*>(pr.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(pw.data_ptr()),
            reinterpret_cast<uint32_t*>(tbl.data_ptr()),
            atoms ? reinterpret_cast<uint32_t*>(atom_tbl.data_ptr()) : nullptr, BH, L, stream)));
      });
  if (!atoms) return {tbl, c10::nullopt};
  auto bits = torch::empty({BH, (long)AP::blocks}, pr.options().dtype(torch::kUInt8));
  C10_CUDA_CHECK((launch_chunk_atom_bitmap_arm<D, B>(
      reinterpret_cast<const uint32_t*>(atom_tbl.data_ptr()),
      reinterpret_cast<uint8_t*>(bits.data_ptr()), BH, L, stream)));
  return {tbl, bits};
}

//: THE RATIO, IN ONE ELEMENTWISE PASS.  It cannot be in the kernel: `den[t]` is
//: a sum over the N/BC blocks and no block knows the total.  `eps` is a runtime
//: argument -- it is a constant of the DEFINITION and belongs to the definition's
//: file.  The divide is round-to-nearest by name, and so is the narrowing store.
//:
//: `mass` MAY BE NULL, and null is 1: the read levels' per-token scale rides here
//: instead of being divided out of them, and the floor is scaled with the ratio so
//: nothing about the defined readout moves -- docs/internals/chunk/chunk.md#readout.
template <typename OutT>
__device__ __forceinline__ void store_quad(OutT* y, long vec, const float4& r);

template <>
__device__ __forceinline__ void store_quad<float>(float* y, long vec, const float4& r) {
  reinterpret_cast<float4*>(y)[vec] = r;
}

template <>
__device__ __forceinline__ void store_quad<__nv_bfloat16>(__nv_bfloat16* y, long vec,
                                                          const float4& r) {
  const __nv_bfloat162 lo = __floats2bfloat162_rn(r.x, r.y);
  const __nv_bfloat162 hi = __floats2bfloat162_rn(r.z, r.w);
  reinterpret_cast<uint2*>(y)[vec] =
      make_uint2(*reinterpret_cast<const uint32_t*>(&lo), *reinterpret_cast<const uint32_t*>(&hi));
}

template <typename OutT>
__global__ void divide_kernel(const float* __restrict__ num, const float* __restrict__ den,
                              const float* __restrict__ mass, OutT* __restrict__ y, long vecs,
                              int L, int H, float eps) {
  constexpr int kQuads = kD / 4;
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= vecs) return;
  const long tok = i / kQuads;
  const float d = den[tok] + eps * (mass != nullptr ? mass[tok] : 1.0f);
  const float4 x = reinterpret_cast<const float4*>(num)[i];
  float4 r;
  r.x = __fdiv_rn(x.x, d);
  r.y = __fdiv_rn(x.y, d);
  r.z = __fdiv_rn(x.z, d);
  r.w = __fdiv_rn(x.w, d);
  //: `[bh, t, d_v] -> [b, t, h, d_v]`: the operand layout `o_proj` reads, so the
  //: transpose and the narrowing are this store rather than two host passes.
  const long bh = tok / L, t = tok % L;
  store_quad<OutT>(y, ((bh / H * L + t) * H + bh % H) * kQuads + i % kQuads, r);
}

torch::Tensor divide_pass(torch::Tensor num, torch::Tensor den, c10::optional<torch::Tensor> mass,
                          double eps, at::ScalarType out) {
  check_arch_table();
  TORCH_CHECK(num.is_contiguous() && den.is_contiguous());
  TORCH_CHECK(num.scalar_type() == torch::kFloat32 && den.scalar_type() == torch::kFloat32);
  TORCH_CHECK(num.dim() == 4 && num.size(-1) == kD, "expect a numerator [B, H, T, d_v]");
  const long B = num.size(0), H = num.size(1), L = num.size(2);
  const float* massPtr = nullptr;
  if (mass.has_value()) {
    TORCH_CHECK(mass->is_cuda() && mass->is_contiguous() && mass->scalar_type() == torch::kFloat32
                    && mass->numel() == B * H * L,
                "the read mass must be a contiguous fp32 [B, H, T] tensor");
    massPtr = mass->data_ptr<float>();
  }
  auto y = torch::empty({B, L, H, (long)kD}, num.options().dtype(out));
  const long vecs = num.numel() / 4;
  const int threads = 256;
  const int blocks = (int)((vecs + threads - 1) / threads);
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (out == torch::kFloat32)
    divide_kernel<float>
        <<<blocks, threads, 0, stream>>>(num.data_ptr<float>(), den.data_ptr<float>(), massPtr,
                                         y.data_ptr<float>(), vecs, (int)L, (int)H, (float)eps);
  else if (out == torch::kBFloat16)
    divide_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        num.data_ptr<float>(), den.data_ptr<float>(), massPtr,
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), vecs, (int)L, (int)H, (float)eps);
  else
    TORCH_CHECK(false, "the readout is emitted in the operand's dtype: fp32 or bf16");
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

//: THE AMPLITUDE HALF, which the facts pass needs and the consumer needs too: the
//: planes carry the problem's `[B, H, L]` addressing, so nothing about the token
//: schedule has to be read off the value stream.
static void check_planes(const torch::Tensor& pr, const torch::Tensor& pw,
                         const std::vector<int64_t>& widths) {
  TORCH_CHECK(pr.is_cuda() && pr.is_contiguous() && pw.is_contiguous());
  TORCH_CHECK(pr.scalar_type() == torch::kBFloat16 && pw.scalar_type() == torch::kBFloat16);
  int64_t wsum = 0;
  for (auto w : widths) wsum += w;
  TORCH_CHECK(pr.sizes() == pw.sizes() && pr.dim() == 4 && pr.size(3) == wsum,
              "expect packed amplitudes [B,H,L,sum(widths)]");
  const int64_t L = pr.size(2);
  TORCH_CHECK(L <= kMaxGatherL && L * wsum * 2 < (1L << 32),
              "a gathered displacement is 32 bits; this launch's would not "
              "fit (L=",
              L, ")");
}

static void check_inputs(const torch::Tensor& pr, const torch::Tensor& pw, const torch::Tensor& v,
                         const std::vector<int64_t>& widths) {
  check_planes(pr, pw, widths);
  TORCH_CHECK(v.is_cuda() && v.scalar_type() == torch::kBFloat16 && v.is_contiguous());
  TORCH_CHECK(v.dim() == 4 && v.size(3) == kD, "expect V [B,L,H,64]");
  const int B = v.size(0), L = v.size(1), H = v.size(2);
  TORCH_CHECK(pr.size(0) == B && pr.size(1) == H && pr.size(2) == L);
  //: the widest gathered rows are the fp32 output (kD*4 B), the packed amplitudes
  //: (sum(widths)*2 B) and V's own token row (H*kD*2 B, because V is token-major);
  //: every gathered displacement must zero-extend from 32 bits.
  TORCH_CHECK((long)L * H * kD * 2 < (1L << 32),
              "a gathered displacement is 32 bits; this launch's would not "
              "fit (L=",
              L, ")");
}

std::pair<torch::Tensor, c10::optional<torch::Tensor>> facts(torch::Tensor pr, torch::Tensor pw,
                                                             int64_t chunk, int64_t bc,
                                                             std::vector<int64_t> widths,
                                                             int64_t mask, bool atoms) {
  check_arch_table();
  check_planes(pr, pw, widths);
  TORCH_CHECK(!atoms || mask == kPrefillStateful,
              "the atom-grain write bitmap is a stateful call's fact: it was asked for "
              "under call class ",
              mask);
  const int BH = (int)(pr.size(0) * pr.size(1)), L = (int)pr.size(2);
#define CHUNK_FACTS_X(C_, BC_, D_, B_)                    \
  if (chunk == C_ && bc == BC_ && is_arm(widths, D_, B_)) \
    return build_facts<D_, B_, BC_>(pr, pw, BH, L, (int)mask, atoms);
  CHUNK_ARMS_X(CHUNK_FACTS_X)
#undef CHUNK_FACTS_X
  TORCH_CHECK(false, "unbuilt arm: chunk=", chunk, " BC=", bc, " depth=", widths.size());
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
                                       int64_t bc, std::vector<int64_t> widths, int64_t run_table,
                                       c10::optional<torch::Tensor> s_in,
                                       c10::optional<torch::Tensor> s_out,
                                       c10::optional<torch::Tensor> page_tbl,
                                       c10::optional<torch::Tensor> block_bits) {
  check_arch_table();
  //: the write gain: per-token positive scalar, bf16, [B,H,L] -- spec 4.2.
  TORCH_CHECK(gw.is_cuda() && gw.is_contiguous() && gw.scalar_type() == torch::kBFloat16
                  && gw.dim() == 3 && gw.size(0) == v.size(0) && gw.size(1) == v.size(2)
                  && gw.size(2) == v.size(1),
              "g_write must be a contiguous bf16 [B,H,L] tensor");
  TORCH_CHECK(tbl.is_cuda() && tbl.is_contiguous() && tbl.scalar_type() == torch::kInt32,
              "the union table is a contiguous int32 tensor of packed bits");
  TORCH_CHECK(v.is_cuda() && v.scalar_type() == torch::kBFloat16 && v.is_contiguous());
  check_inputs(pr, pw, v, widths);
  const int B = v.size(0), L = v.size(1), H = v.size(2);
  //: The accumulator is zeroed, not merely allocated: the fan-in ADDS, so the
  //: identity of the sum has to be there before the first contribution.
  auto opts = v.options().dtype(torch::kFloat32);
  auto o = torch::zeros({B, H, L, (long)kD}, opts);
  auto den = torch::zeros({B, H, L}, opts);
  const int BH = B * H;
  long N = 1;
  for (auto w : widths) N *= w;
  const int32_t* const pageTbl = page_table(page_tbl, BH, N);
  //: THE SPARSE GRID IS THE CALLER'S CHOICE, taken by handing over a bitmap.
  //: There is no in-kernel density test to make it: what a skipped block would
  //: have cost is a property of the call class, and the caller is what knows
  //: whether this call's routing is worth reducing at all.
  const uint8_t* blockBits = nullptr;
  if (block_bits.has_value()) {
    const torch::Tensor& b = *block_bits;
    TORCH_CHECK(b.is_cuda() && b.is_contiguous() && b.scalar_type() == torch::kUInt8 && b.dim() == 2
                    && b.size(0) == BH,
                "the liveness bitmap is a contiguous uint8 [BH, N/BC] tensor");
    blockBits = reinterpret_cast<const uint8_t*>(b.data_ptr());
  }
  const float* const sIn = state_plane(s_in, "the entry state", BH, N, pageTbl != nullptr);
  float* const sOut = state_plane(s_out, "the final state", BH, N, pageTbl != nullptr);
  TORCH_CHECK(pageTbl == nullptr || sIn != nullptr || sOut != nullptr,
              "a page table resolves the state plane's atoms; a call that "
              "carries no state has nothing for it to resolve");
  //: THE STATE VARIANT IS A STRUCTURAL CHOICE, taken here: a call that carries
  //: no state runs the kernel that has no state code in it at all, and a call
  //: that hands over no liveness bitmap runs the one that has no work loop in it
  //: (docs/internals/chunk/chunk_kernel.md).
#define CHUNK_DISPATCH_X(C_, BC_, D_, B_)                                   \
  if (chunk == C_ && bc == BC_ && is_arm(widths, D_, B_)) {                    \
    const long rows = (long)LevelPlan<D_, B_, BC_>::blocks;                    \
    TORCH_CHECK(blockBits == nullptr || block_bits->size(1) == rows,           \
                "the liveness bitmap was built for a different arm");          \
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
                   blockBits,                                                  \
                   (uint32_t)run_table,                                        \
                   BH,              L,                                         \
                   H * kD * (int)sizeof(__nv_bfloat16)};                        \
    const auto stream_ = at::cuda::getCurrentCUDAStream();                     \
    C10_CUDA_CHECK((launch_variant<C_, BC_, D_, B_>(p_, stream_)));            \
    return {o, den};                                                           \
  }
  CHUNK_ARMS_X(CHUNK_DISPATCH_X)
#undef CHUNK_DISPATCH_X
  TORCH_CHECK(false, "unbuilt arm: chunk=", chunk, " BC=", bc, " depth=", widths.size());
}

torch::Tensor forward(torch::Tensor pr, torch::Tensor pw, torch::Tensor gw, torch::Tensor v,
                      int64_t chunk, int64_t bc, std::vector<int64_t> widths, double eps,
                      int64_t run_table, c10::optional<torch::Tensor> read_mass, at::ScalarType out,
                      c10::optional<torch::Tensor> s_in, c10::optional<torch::Tensor> s_out,
                      c10::optional<torch::Tensor> page_tbl,
                      c10::optional<torch::Tensor> block_bits) {
  //: the call class off the POINTERS, the same reading `launch_variant` takes: this
  //: composition asks for no residency facts, so the class governs nothing but which
  //: variant of the facts pass runs.
  const int64_t mask = (s_in || s_out || page_tbl) ? kPrefillStateful : kPrefillStateless;
  auto tbl = facts(pr, pw, chunk, bc, widths, mask, false).first;
  auto nd = forward_raw(pr, pw, gw, v, tbl, chunk, bc, widths, run_table, s_in, s_out, page_tbl,
                        block_bits);
  return divide_pass(nd[0], nd[1], read_mass, eps, out);
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
  //: BOTH BUILT VARIANTS ARE REPORTED, stateless first: they are two kernels with
  //: two register censuses, and a stamp that named one of them would attribute a
  //: measurement to a kernel the launch may not have run.
#define CHUNK_CENSUS_X(C_, BC_, D_, B_)                                             \
  if (chunk == C_ && bc == BC_ && is_arm(widths, D_, B_)) {                         \
    for (auto x : query_chunk_census_arm<C_, BC_, D_, B_, false>()) r.push_back(x); \
    for (auto x : query_chunk_census_arm<C_, BC_, D_, B_, true>()) r.push_back(x);  \
    return r;                                                                       \
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
