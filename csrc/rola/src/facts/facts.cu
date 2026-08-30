// csrc/rola/src/facts/facts.cu -- the facts layer's DISPATCH and torch
// entries, adopted verbatim from the deleted chunk consumer's dispatch TU
// (K31 batch; baseline = tag baseline/pre-k31).  This TU never touches a
// kernel symbol: the launcher arms arrive as extern declarations satisfied by
// the instantiation shard (facts_s0.cu), the consumer family's sharding rule.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <mutex>
#include <set>
#include <utility>
#include <vector>

#include "../dispatch_switch.cuh"
#include "common/arch_caps.cuh"
#include "facts_kernel.cuh"

namespace rola::chunk {

//: THE BOUND ON L, retained from the consumer era: the widest gathered row the -- see docs/internals/facts/facts.md#kd
static constexpr int kD = 64;
static constexpr long kMaxGatherL = (1L << 32) / (kD * (long)sizeof(float));

//: THE RUNTIME HALF OF THE CLOSED-WORLD ARCH RULE (setup.py's R4). -- see docs/internals/facts/facts.md#check-arch-table
void check_arch_table() {
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  //: ONCE PER PROCESS PER DEVICE, on the first launch rather than at import: a -- see docs/internals/facts/facts.md#check-arch-table-note-l41
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
              "the facts dispatch's capability table has no row for compute capability ",
              prop.major, ".", prop.minor,
              "; add it to caps_of() rather than letting an operation guess which "
              "implementation it has. This binary is closed-world: it runs the "
              "architectures it was measured on and refuses the rest.");
  const denseref::arch::Caps want = denseref::arch::caps_of(cc);
  TORCH_CHECK(denseref::arch::supported(want), "UNSUPPORTED ARCHITECTURE at compute capability ",
              prop.major, ".", prop.minor,
              ": a REQUIRED capability clause of the ratified baseline is false in its own "
              "row. There is no partial support and no slower route to take.");
  int smem_per_sm = 0, smem_per_cta_max = 0;
  C10_CUDA_CHECK(
      cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
  C10_CUDA_CHECK(
      cudaDeviceGetAttribute(&smem_per_cta_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  TORCH_CHECK(want.smem_per_sm == smem_per_sm && want.smem_per_cta_max == smem_per_cta_max
                  && want.regs_per_sm == prop.regsPerMultiprocessor
                  && want.max_threads_per_sm == prop.maxThreadsPerMultiProcessor,
              "the facts dispatch's capability table disagrees with the driver at compute "
              "capability ",
              prop.major, ".", prop.minor, ": table {smem/SM ", want.smem_per_sm, ", smem/CTA ",
              want.smem_per_cta_max, ", regs/SM ", want.regs_per_sm, ", threads/SM ",
              want.max_threads_per_sm, "} vs driver {smem/SM ", smem_per_sm, ", smem/CTA ",
              smem_per_cta_max, ", regs/SM ", prop.regsPerMultiprocessor, ", threads/SM ",
              prop.maxThreadsPerMultiProcessor,
              "}. Every launch shape compiled into this binary was derived from the "
              "table, so they are wrong for this device.");
  const std::lock_guard<std::mutex> hold(mu);
  admitted.insert(dev);
}

//: the arm surface, DECLARED only -- definitions live in the instantiation -- see docs/internals/facts/facts.md#near-line-79
template <int D, int B, int BC, int MASK>
cudaError_t launch_chunk_facts_arm(const __nv_bfloat16* pr, const __nv_bfloat16* pw, uint32_t* tbl,
                                   uint32_t* atom_tbl, int BH, int L, cudaStream_t stream);
template <int D, int B>
cudaError_t launch_chunk_atom_bitmap_arm(const uint32_t* tbl, uint8_t* atom_bits, int BH, int L,
                                         cudaStream_t stream);
template <int D, int B, int BC>
cudaError_t launch_chunk_block_bitmap_arm(const uint32_t* tbl, const int32_t* page_tbl,
                                          uint8_t* block_bits, int BH, int L, cudaStream_t stream);

//: THE FACTS, AS TENSORS: the union table `[B*H, 2, entries, L/32]` of packed -- see docs/internals/facts/facts.md#build-facts
template <int D, int B, int BC>
static std::pair<torch::Tensor, c10::optional<torch::Tensor>> build_facts(const torch::Tensor& pr,
                                                                          const torch::Tensor& pw,
                                                                          int BH, int L, int mask,
                                                                          bool atoms) {
  using AP = LevelPlan<D, B, kAtomLeaves>;
  constexpr int E = LevelPlan<D, B, BC>::entries;
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto tbl = torch::empty({BH, 2, E, tok_words(L)}, pr.options().dtype(torch::kInt32));
  auto atom_tbl = atoms ? torch::empty({BH, 2, (long)AP::entries, tok_words(L)},
                                       pr.options().dtype(torch::kInt32))
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

//: THE ARGUMENT CHECKS OF A LAUNCH, SHARED BY THE TABLE AND THE KERNEL so that -- see docs/internals/facts/facts.md#is-arm
static bool is_arm(const std::vector<int64_t>& widths, int64_t D, int64_t B) {
  if ((int64_t)widths.size() != D) return false;
  for (auto w : widths)
    if (w != B) return false;
  return true;
}

//: THE AMPLITUDE HALF, which the facts pass needs and the consumer needs too: the -- see docs/internals/facts/facts.md#checkplanes
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

std::pair<torch::Tensor, c10::optional<torch::Tensor>> facts(torch::Tensor pr, torch::Tensor pw,
                                                             int64_t chunk, int64_t bc,
                                                             std::vector<int64_t> widths,
                                                             int64_t mask, bool atoms) {
  check_arch_table();
  check_planes(pr, pw, widths);
  TORCH_CHECK(!atoms || mask == kPrefillStateful,
              "the atom-grain activity bitmap is a stateful call's fact: it was asked for "
              "under call class ",
              mask);
  const int BH = (int)(pr.size(0) * pr.size(1)), L = (int)pr.size(2);
#define CHUNK_FACTS_X(C_, BC_, D_, B_)                    \
  if (chunk == C_ && bc == BC_ && is_arm(widths, D_, B_)) \
    return build_facts<D_, B_, BC_>(pr, pw, BH, L, (int)mask, atoms);
  FACTS_ARMS_X(CHUNK_FACTS_X)
#undef CHUNK_FACTS_X
  TORCH_CHECK(false, "unbuilt arm: chunk=", chunk, " BC=", bc, " depth=", widths.size());
}

//: THE ATOM -> SLOT MAP, `[BH, N/16]` int32, `-1` where the atom has no page. -- see docs/internals/facts/facts.md#page-table
static const int32_t* page_table(const c10::optional<torch::Tensor>& t, int BH, long N) {
  if (!t.has_value()) return nullptr;
  TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->scalar_type() == torch::kInt32,
              "the page table must be a contiguous int32 CUDA tensor");
  TORCH_CHECK(t->numel() == (long)BH * (N / kAtomLeaves), "the page table must hold BH * N / ",
              kAtomLeaves, " entries (one per atom); got ", t->numel(), " for BH=", BH, " N=", N);
  return t->data_ptr<int32_t>();
}

//: THE LIVENESS EPILOGUE: `[BH, N/BC]` uint8, bits {READ, WRITE, STATE}, -- see docs/internals/facts/facts.md#build-block-bits
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
  //: NO `C` AXIS: the table this reduces is the (D, B, BC) one, so the arm walk -- see docs/internals/facts/facts.md#chunkblockbitsx
#define CHUNK_BLOCK_BITS_X(C_, BC_, D_, B_)                                \
  if (bc == BC_ && is_arm(widths, D_, B_)) {                               \
    const long want = (long)LevelPlan<D_, B_, BC_>::entries;               \
    TORCH_CHECK(tbl.size(2) == want,                                       \
                "the union table was built for a different arm");          \
    return build_block_bits<D_, B_, BC_>(tbl, pageTbl, BH, (int)L);        \
  }
  FACTS_ARMS_X(CHUNK_BLOCK_BITS_X)
#undef CHUNK_BLOCK_BITS_X
  TORCH_CHECK(false, "unbuilt arm: BC=", bc, " depth=", widths.size());
}

//: The built set with the plan quantities of an arm reported beside it: the -- see docs/internals/facts/facts.md#arms
std::vector<std::vector<int64_t>> arms() {
  std::vector<std::vector<int64_t>> out;
#define FACTS_LIST_X(C_, BC_, D_, B_)                                       \
  out.push_back({C_, BC_, D_, B_, (int64_t)LevelPlan<D_, B_, BC_>::blocks,     \
                 (int64_t)(BC_ / LevelPlan<D_, B_, BC_>::span_at(D_ - 1)),     \
                 (int64_t)LevelPlan<D_, B_, BC_>::span_at(D_ - 1),             \
                 (int64_t)LevelPlan<D_, B_, BC_>::entries});
  FACTS_ARMS_X(FACTS_LIST_X)
#undef FACTS_LIST_X
  return out;
}

}  // namespace rola::chunk
