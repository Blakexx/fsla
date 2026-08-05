// RoLA V3 consumer -- THE HOST DISPATCH TRANSLATION UNIT.
// ROLA_V3_SPEC.md §3 (structure), §4 (schedule), §5 (consumer), §6.4 (sequence
// parallelism), §7.1 (paging).
//
// WHAT IS IN THIS FILE, and nothing else is:
//   * the Torch-level entry `rola_consumer_forward` -- argument checking, the
//     `ConsumerParams` fill, and §6.4's three-pass sequencing;
//   * the launch dispatch `launch_consumer`, which resolves the plan's structural
//     axes to ONE template instantiation of `consumer::consumer_forward_kernel`;
//   * the two non-template helper kernels, `scan_kernel` (§6.4 pass 2's ordered
//     fold) and `divide_kernel` (`global`'s `num/den` epilogue);
//   * the driver-backed residency and footprint queries, and the arch-table check;
//   * the pybind surface.
//
// THE KERNEL ITSELF IS IN `consumer_kernel.cuh`, which is also where the §3/§5
// algorithm is documented. This file deliberately does NOT restate it: two
// descriptions of one kernel is one that can go stale.
//
// ONE CONSUMER: there is a single kernel family, so there is no second-kernel
// routing argument, no `intra_lowering` argument and no second occupancy oracle --
// what the planner prices and what the driver launches are the same object by
// construction. The three-pass §6.4 split below is DORMANT BY PLAN, not dead:
// implemented, oracle-gated, and not currently plan-selected.
//
// The dispatch matrix and its measured size, the residency query axes, the ordered
// fold, the entry-state stash: docs/internals/consumer.md

// The CUDA/ATen surface this TU launches and checks against.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "consumer_kernel.cuh"
#include "host_api.cuh"

namespace rola {

namespace {

// ---------------------------------------------------------------------------
// §6.4 pass 2: the ordered fold. One CTA per SPLIT owner, sequential over that
// owner's segments inside it, `E := A*E + B` ascending and no inverse of `A`
// anywhere (P5.21's obligation). docs/internals/consumer.md#ordered-fold
// ---------------------------------------------------------------------------

//: §4.5's page granule, mirrored from `rola.ops.paging.MMA_K_QUANTUM` and equal by
//: construction to `consumer_kernel.cuh`'s `kMmaKQuantum` -- one number, kept honest
//: by `tests/unit/test_paging_atoms.py` and the paged equivalence fixtures. Declared
//: here because `scan_kernel` must not include the consumer kernel's header.
constexpr int kScanAtomLeaves = 16;

__global__ void scan_kernel(ScanParams p) {
  extern __shared__ float E[];
  const int u = blockIdx.x;
  const int bh = p.owners[2 * u];
  const int owner = p.owners[2 * u + 1];
  const int nseg = p.nseg[u];
  const int slot0 = p.slot[u];
  const int BC = p.BC;
  const int cols = p.cols;
  const int n = BC * cols;

  // The owner's entry state, addressed exactly as the consumer addresses it, so a
  // paged and an unpaged fold see the same numbers.
  const int atoms_per_bh = static_cast<int>(p.N / kScanAtomLeaves);
  for (int e = threadIdx.x; e < n; e += blockDim.x) {
    const int i = e / cols;
    const int c = e - i * cols;
    const int q = i / kScanAtomLeaves;
    const int r = i - q * kScanAtomLeaves;
    int64_t row;
    if (p.page_table != nullptr) {
      const int physical = p.page_table[static_cast<int64_t>(bh) * atoms_per_bh
                                        + owner * (BC / kScanAtomLeaves) + q];
      row = physical >= 0 ? static_cast<int64_t>(physical) * kScanAtomLeaves + r
                          : static_cast<int64_t>(-1);
    } else {
      row = static_cast<int64_t>(bh) * p.N + static_cast<int64_t>(owner) * BC + i;
    }
    E[e] = row >= 0 ? p.state[row * cols + c] : 0.0f;
  }
  __syncthreads();

  // ---- the ENTRY-STATE STASH: what makes pass 3's reads disjoint from pass 3's
  // writes. docs/internals/consumer.md#entry-stash -----------------------------
  for (int e = threadIdx.x; e < n; e += blockDim.x) {
    p.ws_state[static_cast<int64_t>(slot0) * n + e] = E[e];
  }
  __syncthreads();

  for (int s = 0; s < nseg - 1; ++s) {
    const int64_t slot = static_cast<int64_t>(slot0) + 1 + s;
    const int K = p.ws_count[slot];
    //: HAZARD written-leaf-fold -- docs/internals/consumer.md#written-leaf-fold
    for (int e = threadIdx.x; e < K * cols; e += blockDim.x) {
      const int k = e / cols;
      const int c = e - k * cols;
      const int i = p.ws_leaf[slot * BC + k];
      E[i * cols + c] =
          fmaf(p.ws_scale[slot * BC + k], E[i * cols + c], p.ws_state[(slot * BC + k) * cols + c]);
    }
    __syncthreads();
    for (int e = threadIdx.x; e < n; e += blockDim.x) {
      p.ws_state[slot * n + e] = E[e];
    }
    __syncthreads();
  }
}

void scan_pass(const ScanParams& p) {
  if (p.n_split == 0) return;
  const size_t smem = static_cast<size_t>(p.BC) * p.cols * sizeof(float);
  scan_kernel<<<p.n_split, 256, smem, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void divide_kernel(const float* __restrict__ num, const float* __restrict__ den,
                              float* __restrict__ out, int64_t rows, int DV, float eps) {
  const int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (idx >= rows * DV) return;
  const int64_t row = idx / DV;
  out[idx] = num[idx] / (den[row] + eps);
}

}  // namespace

// ---------------------------------------------------------------------------
// Host entry
// ---------------------------------------------------------------------------

//: The SM's shared-memory BUDGET, device-queried. §10's ruling records this in the
//: PLAN rather than reading it at launch, so that a plan is reproducible off the
//: machine that made it: docs/internals/consumer.md#arch-table
int64_t rola_sm_shared_bytes() {
  int value = 0;
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(
      cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
  return value;
}

void divide_pass(const float* num, const float* den, float* out, int64_t rows, int d_v, float eps) {
  const int64_t total = rows * d_v;
  const int threads = 256;
  const int64_t blocks = (total + threads - 1) / threads;
  divide_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(num, den, out, rows, d_v,
                                                                          eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

//: VALIDATE THE COMPILE-TIME ARCH TABLE AGAINST THE DRIVER. `__launch_bounds__`
//: needs compile-time constants, so `sm_resources()` is a table, and a wrong table
//: costs residency SILENTLY. This does NOT make the table a planning authority --
//: spec §6.5 leaves that with the driver's occupancy API.
//: docs/internals/consumer.md#arch-table
void rola_consumer_check_arch_table() {
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  const int arch = prop.major * 100 + prop.minor * 10;
  TORCH_CHECK(sm_arch_tabulated(arch),
              "the new consumer's SM-resource table has no row for compute capability ", prop.major,
              ".", prop.minor,
              "; add it to sm_resources() rather than letting the launch bound guess");
  const SmResources want = sm_resources(arch);
  int smem_per_sm = 0;
  C10_CUDA_CHECK(
      cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
  TORCH_CHECK(
      want.smem_per_sm == smem_per_sm && want.max_threads_per_sm == prop.maxThreadsPerMultiProcessor
          && want.regs_per_sm == prop.regsPerMultiprocessor,
      "the new consumer's SM-resource table disagrees with the driver at "
      "compute capability ",
      prop.major, ".", prop.minor, ": table {smem ", want.smem_per_sm, ", threads ",
      want.max_threads_per_sm, ", regs ", want.regs_per_sm, "} vs driver {smem ", smem_per_sm,
      ", threads ", prop.maxThreadsPerMultiProcessor, ", regs ", prop.regsPerMultiprocessor,
      "}. The launch bounds compiled into this binary were derived from the "
      "table, so they are wrong for this device.");
}

// THE LAUNCH DISPATCH. Declared restrictions are hard checks, never coercions.
void launch_consumer(const ConsumerParams& p, int BT, int BC, int DV, bool global_norm, bool decay,
                     const std::string& mma_precision, bool compact) {
  //: Checked ONCE per process, on the first launch rather than at import (a CUDA
  //: context as a side effect of `import` is worse than the check is worth), and
  //: nothing can reach a launch bound without passing through here.
  static const bool arch_checked = [] {
    rola_consumer_check_arch_table();
    return true;
  }();
  (void)arch_checked;
  //: THE FULL PLAN SURFACE, and the single authority for it. These are the planner's
  //: own supported sets, so no shape a plan can select is refused and no shape it
  //: cannot select is compiled: docs/internals/consumer.md#plan-surface
  TORCH_CHECK((BT == 16 || BT == 32) && (BC == 16 || BC == 64) && (DV == 32 || DV == 64),
              "the consumer instantiates BT in {16, 32}, BC in {16, 64}, "
              "d_v in {32, 64}; got BT=",
              BT, " BC=", BC, " d_v=", DV);

  //: `mma_precision` selects BOTH GEMM sites together (RULING 1 revised): "bf16" is
  //: the production point, "tf32x3" the exactness-gate point. Both are instantiated.
  const bool exact = (mma_precision == "tf32x3");
  //: THE INSTANTIATION MATRIX. Every axis is a template because every axis is a
  //: STRUCTURAL plan fact with few legal values -- the standing "branch on structure,
  //: not thresholds" rule -- and there is deliberately NO runtime fallback on any of
  //: them: an out-of-range value is a hard TORCH_CHECK, never a slower path.
  //: The axes, why each one must be compile-time, the MEASURED instantiation count
  //: (the ratification manifest is the authority, not this comment) and the measured
  //: build cost: docs/internals/consumer.md#instantiation-matrix
#define ROLA_CONSUMER_ONE(BTV, DVV, BCVAL, DVAL, PRECV, DECAYV, GLOBALV, COMPACTV)          \
  {                                                                                    \
    constexpr size_t smem =                                                            \
        consumer::consumer_smem_bytes<BTV, BCVAL, DVV, DVAL, PRECV, PRECV, DECAYV, GLOBALV,  \
                                COMPACTV>();                                           \
    auto kernel = consumer::consumer_forward_kernel<BTV, BCVAL, DVV, DVAL, PRECV, PRECV,  \
                                                 DECAYV, GLOBALV, COMPACTV>;           \
    if (smem > 48 * 1024) {                                                            \
      C10_CUDA_CHECK(cudaFuncSetAttribute(                                             \
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));\
    }                                                                                  \
    if (p.n_owners == 0) return;                                                       \
    kernel<<<p.n_owners, consumer::kConsumerThreads, smem,                                   \
             at::cuda::getCurrentCUDAStream()>>>(p);                                   \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                                    \
    return;                                                                            \
  }
  //: HAZARD dispatch-braces -- docs/internals/consumer.md#dispatch-braces
#define ROLA_CONSUMER_CMP(BTV, DVV, BCVAL, DVAL, PRECV, DECAYV, GLOBALV)                    \
  {                                                                                    \
    if (compact) ROLA_CONSUMER_ONE(BTV, DVV, BCVAL, DVAL, PRECV, DECAYV, GLOBALV, true)      \
    ROLA_CONSUMER_ONE(BTV, DVV, BCVAL, DVAL, PRECV, DECAYV, GLOBALV, false)                  \
  }
#define ROLA_CONSUMER_PREC(BTV, DVV, BCVAL, DVAL, PRECV)                                    \
  if (decay && global_norm) ROLA_CONSUMER_CMP(BTV, DVV, BCVAL, DVAL, PRECV, true, true)     \
  if (decay) ROLA_CONSUMER_CMP(BTV, DVV, BCVAL, DVAL, PRECV, true, false)                   \
  if (global_norm) ROLA_CONSUMER_CMP(BTV, DVV, BCVAL, DVAL, PRECV, false, true)             \
  ROLA_CONSUMER_CMP(BTV, DVV, BCVAL, DVAL, PRECV, false, false)
#define ROLA_CONSUMER_LAUNCH(BTV, DVV, BCVAL, DVAL)                                         \
  case DVAL: {                                                                         \
    if (exact) { ROLA_CONSUMER_PREC(BTV, DVV, BCVAL, DVAL, MmaPrec::kTf32x3) }              \
    ROLA_CONSUMER_PREC(BTV, DVV, BCVAL, DVAL, MmaPrec::kBf16)                               \
  }
#define ROLA_CONSUMER_D(BTV, DVV, BCVAL)                                                    \
  switch (p.D) {                                                                       \
    ROLA_CONSUMER_LAUNCH(BTV, DVV, BCVAL, 1)                                                \
    ROLA_CONSUMER_LAUNCH(BTV, DVV, BCVAL, 2)                                                \
    ROLA_CONSUMER_LAUNCH(BTV, DVV, BCVAL, 3)                                                \
    ROLA_CONSUMER_LAUNCH(BTV, DVV, BCVAL, 4)                                                \
    default:                                                                           \
      TORCH_CHECK(false, "D must be in [1, 4] (spec §1), got ", p.D);                  \
  }
#define ROLA_CONSUMER_BC(BTV, DVV)                                                          \
  if (BC == 16) { ROLA_CONSUMER_D(BTV, DVV, 16) } else { ROLA_CONSUMER_D(BTV, DVV, 64) }
#define ROLA_CONSUMER_DV(BTV)                                                               \
  if (DV == 32) { ROLA_CONSUMER_BC(BTV, 32) } else { ROLA_CONSUMER_BC(BTV, 64) }
  if (BT == 16) {
    ROLA_CONSUMER_DV(16)
  } else {
    ROLA_CONSUMER_DV(32)
  }
#undef ROLA_CONSUMER_DV
#undef ROLA_CONSUMER_BC
#undef ROLA_CONSUMER_D
#undef ROLA_CONSUMER_LAUNCH
#undef ROLA_CONSUMER_PREC
#undef ROLA_CONSUMER_CMP
#undef ROLA_CONSUMER_ONE
}

//: The consumer's residency answer, from the DRIVER. It takes five axes because
//: RESIDENCY IS PER-INSTANTIATION and a single reported number would over-promise on
//: one arm and under-report another: docs/internals/consumer.md#occupancy-axes
int64_t rola_consumer_occupancy(bool decay, bool compact, int64_t BC, int64_t BT, int64_t DV) {
  //: Queried at `D = 2`, the at-scale cell's topology depth; `D` is not a query axis
  //: because it moves only the `[D][BC]` digit table.
  int blocks = 0;
#define ROLA_CONSUMER_OCC(BTV, DVV, BCV, DECAYV, COMPACTV)                                 \
  {                                                                                   \
    constexpr size_t smem =                                                           \
        consumer::consumer_smem_bytes<BTV, BCV, DVV, 2, consumer::kReadPrecDefault,             \
                                consumer::kFoldPrecDefault, DECAYV,                    \
                                true, COMPACTV>();                                    \
    auto kernel = consumer::consumer_forward_kernel<BTV, BCV, DVV, 2,                    \
                                                 consumer::kFoldPrecDefault,             \
                                                 consumer::kReadPrecDefault, DECAYV, true,\
                                                 COMPACTV>;                           \
    if (smem > 48 * 1024) {                                                           \
      C10_CUDA_CHECK(cudaFuncSetAttribute(                                            \
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));\
    }                                                                                 \
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(                     \
        &blocks, kernel, consumer::kConsumerThreads, smem));                                \
  }
#define ROLA_CONSUMER_OCC_BC(BTV, DVV, BCV)                                                \
  {                                                                                   \
    if (decay && compact) ROLA_CONSUMER_OCC(BTV, DVV, BCV, true, true)                     \
    else if (decay) ROLA_CONSUMER_OCC(BTV, DVV, BCV, true, false)                          \
    else if (compact) ROLA_CONSUMER_OCC(BTV, DVV, BCV, false, true)                        \
    else ROLA_CONSUMER_OCC(BTV, DVV, BCV, false, false)                                     \
  }
#define ROLA_CONSUMER_OCC_DV(BTV, DVV)                                                     \
  { if (BC == 16) ROLA_CONSUMER_OCC_BC(BTV, DVV, 16) else ROLA_CONSUMER_OCC_BC(BTV, DVV, 64) }
#define ROLA_CONSUMER_OCC_BT(BTV)                                                          \
  { if (DV == 32) ROLA_CONSUMER_OCC_DV(BTV, 32) else ROLA_CONSUMER_OCC_DV(BTV, 64) }
  TORCH_CHECK(BC == 16 || BC == 64, "BC must be 16 or 64, got ", BC);
  TORCH_CHECK(BT == 16 || BT == 32, "BT must be 16 or 32, got ", BT);
  TORCH_CHECK(DV == 32 || DV == 64, "d_v must be 32 or 64, got ", DV);
  if (BT == 16)
    ROLA_CONSUMER_OCC_BT(16)
  else
    ROLA_CONSUMER_OCC_BT(32)
#undef ROLA_CONSUMER_OCC_BT
#undef ROLA_CONSUMER_OCC_DV
#undef ROLA_CONSUMER_OCC_BC
#undef ROLA_CONSUMER_OCC
  return blocks;
}

int64_t rola_consumer_smem_bytes(bool decay, bool compact, int64_t BC, int64_t BT, int64_t DV) {
#define ROLA_CONSUMER_SMEM(BTV, DVV, BCV, DECAYV, COMPACTV)                                \
  return static_cast<int64_t>(                                                        \
      consumer::consumer_smem_bytes<BTV, BCV, DVV, 2, consumer::kReadPrecDefault,             \
                                    consumer::kFoldPrecDefault, DECAYV, true,           \
                              COMPACTV>());
#define ROLA_CONSUMER_SMEM_BC(BTV, DVV, BCV)                                               \
  {                                                                                   \
    if (decay && compact) ROLA_CONSUMER_SMEM(BTV, DVV, BCV, true, true)                    \
    if (decay) ROLA_CONSUMER_SMEM(BTV, DVV, BCV, true, false)                              \
    if (compact) ROLA_CONSUMER_SMEM(BTV, DVV, BCV, false, true)                            \
    ROLA_CONSUMER_SMEM(BTV, DVV, BCV, false, false)                                        \
  }
#define ROLA_CONSUMER_SMEM_DV(BTV, DVV)                                                    \
  { if (BC == 16) ROLA_CONSUMER_SMEM_BC(BTV, DVV, 16) ROLA_CONSUMER_SMEM_BC(BTV, DVV, 64) }
#define ROLA_CONSUMER_SMEM_BT(BTV)                                                         \
  { if (DV == 32) ROLA_CONSUMER_SMEM_DV(BTV, 32) ROLA_CONSUMER_SMEM_DV(BTV, 64) }
  TORCH_CHECK(BC == 16 || BC == 64, "BC must be 16 or 64, got ", BC);
  TORCH_CHECK(BT == 16 || BT == 32, "BT must be 16 or 32, got ", BT);
  TORCH_CHECK(DV == 32 || DV == 64, "d_v must be 32 or 64, got ", DV);
  if (BT == 16) ROLA_CONSUMER_SMEM_BT(16) ROLA_CONSUMER_SMEM_BT(32)
#undef ROLA_CONSUMER_SMEM_BT
#undef ROLA_CONSUMER_SMEM_DV
#undef ROLA_CONSUMER_SMEM_BC
#undef ROLA_CONSUMER_SMEM
}

// ---------------------------------------------------------------------------
// Torch-level entry. Consumes exactly the Stage 3 schedule
// (`rola/planner/schedule.py`) plus the value stream and the dense state. Every
// argument is checked loudly; there is no silent shape coercion and no CPU
// fallback. docs/internals/consumer.md#what-is-here
// ---------------------------------------------------------------------------

namespace {

const float* f32_ptr(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32, got ", t.scalar_type());
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  return t.data_ptr<float>();
}

const uint32_t* u32_ptr(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == at::kInt, name, " must be int32 bit patterns, got ",
              t.scalar_type());
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  return reinterpret_cast<const uint32_t*>(t.data_ptr<int32_t>());
}

}  // namespace

std::vector<at::Tensor> rola_consumer_forward(
    std::vector<at::Tensor> amp_read, std::vector<at::Tensor> amp_write,
    c10::optional<std::vector<at::Tensor>> amp_clock, c10::optional<at::Tensor> dials,
    at::Tensor read_bits, at::Tensor write_bits, at::Tensor read_tok_bits,
    at::Tensor write_tok_bits, at::Tensor read_union, at::Tensor write_union, at::Tensor owners,
    c10::optional<at::Tensor> g_read, at::Tensor g_write, at::Tensor v, at::Tensor state,
    c10::optional<at::Tensor> page_table, std::vector<int64_t> level_stride,
    std::vector<int64_t> level_modulus, std::vector<int64_t> level_row_offset,
    std::vector<int64_t> level_width, std::vector<int64_t> owner_stride,
    int64_t first_spanned_level, int64_t slice_digits, int64_t n_slices, int64_t owners_total,
    int64_t n_heads, int64_t BT, int64_t BC, bool global_norm, std::string mma_precision,
    bool compact, double eps,
    // ---- §6.4 sequence parallelism. ALL-OR-NOTHING (14 arguments or none; `P = 1`
    // is the primary configuration): docs/internals/consumer.md#all-or-nothing
    c10::optional<at::Tensor> work_seg,        // [n_owners] int32, keyed like `owners`
    c10::optional<at::Tensor> work_nseg,       // [n_owners] int32
    c10::optional<at::Tensor> work_slot,       // [n_owners] int32
    c10::optional<at::Tensor> summary_owners,  // [n_summary, 2] int32
    c10::optional<at::Tensor> summary_seg,     // [n_summary] int32
    c10::optional<at::Tensor> summary_nseg,    // [n_summary] int32
    c10::optional<at::Tensor> summary_slot,    // [n_summary] int32
    c10::optional<at::Tensor> scan_owners,     // [n_split, 2] int32
    c10::optional<at::Tensor> scan_nseg,       // [n_split] int32
    c10::optional<at::Tensor> scan_slot,       // [n_split] int32
    c10::optional<at::Tensor> ws_state,        // [n_slots, BC, cols] float32
    c10::optional<at::Tensor> ws_scale,        // [n_slots, BC] float32
    c10::optional<at::Tensor> ws_leaf,         // [n_slots, BC] int32
    c10::optional<at::Tensor> ws_count) {      // [n_slots] int32
  const int64_t D = static_cast<int64_t>(amp_read.size());
  TORCH_CHECK(D >= 1 && D <= kMaxLevels, "D must be in [1, 4] (spec §1), got ", D);
  TORCH_CHECK(static_cast<int64_t>(amp_write.size()) == D, "amp_write must have D entries");

  TORCH_CHECK(v.dim() == 3, "v must be [BH, T, d_v], got ", v.dim(), " dims");
  const int64_t BH = v.size(0);
  const int64_t T = v.size(1);
  const int64_t d_v = v.size(2);
  // §4.5: paged, `N` is not readable off the state, so it is derived from the owner
  // partition -- the definition. docs/internals/consumer.md#state-derivation
  const int64_t cols = d_v + (global_norm ? 1 : 0);
  int64_t N;
  if (page_table.has_value()) {
    TORCH_CHECK(BC % kScanAtomLeaves == 0,
                "BC must be a whole number of MMA_K_QUANTUM=", kScanAtomLeaves,
                "-leaf atoms under paging (spec §1b); got BC=", BC);
    TORCH_CHECK(state.dim() == 3 && state.size(1) == kScanAtomLeaves && state.size(2) == cols,
                "paged state must be [physical_slots, MMA_K_QUANTUM(", kScanAtomLeaves,
                "), d_v(+1 under global)] -- the page granule is the "
                "ATOM, not the owner rectangle (spec §4.5)");
    const auto& table = *page_table;
    const int64_t atoms_per_bh = owners_total * BC / kScanAtomLeaves;
    TORCH_CHECK(table.is_cuda() && table.is_contiguous() && table.scalar_type() == at::kInt
                    && table.dim() == 2 && table.size(0) == BH && table.size(1) == atoms_per_bh,
                "page_table must be contiguous int32 [BH, atoms_per_bh = N/", kScanAtomLeaves,
                "] on the state's device");
    N = owners_total * BC;
  } else {
    TORCH_CHECK(state.dim() == 3, "state must be [BH, N, d_v(+1)]");
    TORCH_CHECK(state.size(0) == BH, "state batch-head dim must match v");
    N = state.size(1);
    TORCH_CHECK(state.size(2) == cols, "state's last dim must be d_v(+1 under global), got ",
                state.size(2));
  }
  TORCH_CHECK(owners.dim() == 2 && owners.size(1) == 2, "owners must be [n_owners, 2]");
  TORCH_CHECK(owners.scalar_type() == at::kInt, "owners must be int32");
  TORCH_CHECK(read_bits.dim() == 3, "read_bits must be [BH, total_rows, n_words]");
  // §4's support word, TILE-MAJOR. Shape-checked against the tile bitmap it refines
  // and against the amplitude stream's tile axis, so a schedule built at a different
  // BT cannot reach the gather. docs/internals/consumer.md#support-shape-check
  TORCH_CHECK(read_tok_bits.dim() == 3 && write_tok_bits.sizes() == read_tok_bits.sizes(),
              "read/write_tok_bits must both be [BH, n_tiles, total_rows]");
  TORCH_CHECK(
      read_tok_bits.size(0) == read_bits.size(0) && read_tok_bits.size(2) == read_bits.size(1),
      "tok_bits must carry the same [BH, total_rows] level-digit rows as the tile bitmap");
  TORCH_CHECK(BT >= 1 && BT <= 32,
              "BT must be in [1, 32] (spec §6): the support word carries one bit per token "
              "of a tile, got BT=",
              BT);

  ConsumerParams p{};
  for (int64_t l = 0; l < D; ++l) {
    TORCH_CHECK(amp_read[l].dim() == 4, "amp_read[", l, "] must be [BH, n_tiles, BT, pitch]");
    p.amp_read[l] = f32_ptr(amp_read[l], "amp_read");
    p.amp_write[l] = f32_ptr(amp_write[l], "amp_write");
    p.level_pitch_read[l] = static_cast<int>(amp_read[l].size(3));
    p.level_pitch_write[l] = static_cast<int>(amp_write[l].size(3));
    if (amp_clock.has_value()) {
      TORCH_CHECK((*amp_clock)[l].dim() == 4, "amp_clock[", l,
                  "] must be [BH, n_tiles, BT, pitch]");
      p.amp_clock[l] = f32_ptr((*amp_clock)[l], "amp_clock");
      p.level_pitch_clock[l] = static_cast<int>((*amp_clock)[l].size(3));
    } else {
      p.amp_clock[l] = nullptr;
      p.level_pitch_clock[l] = 0;
    }
    p.level_stride[l] = static_cast<int>(level_stride[l]);
    p.level_modulus[l] = static_cast<int>(level_modulus[l]);
    p.level_row_offset[l] = static_cast<int>(level_row_offset[l]);
    p.level_width[l] = static_cast<int>(level_width[l]);
    p.owner_stride[l] = static_cast<int>(owner_stride[l]);
    TORCH_CHECK(p.level_stride[l] >= 1, "level_stride must be >= 1");
    TORCH_CHECK(p.level_modulus[l] >= 1, "level_modulus must be >= 1");
    TORCH_CHECK(p.owner_stride[l] >= 1, "owner_stride must be >= 1 (use 1 for spanned levels)");
  }

  const bool decay = dials.has_value();
  TORCH_CHECK(decay == amp_clock.has_value(),
              "LeafMassDecay needs BOTH the dials and the clock stream, or neither; the clock "
              "stream is built by build_clock_stream(), a separate builder from the schedule "
              "so that the STORED write tensor can never key a tile bitmap (P5.20.5 finding 2)");
  if (decay) {
    TORCH_CHECK(static_cast<int64_t>(amp_clock->size()) == D, "amp_clock must have D entries");
    TORCH_CHECK(dials->dim() == 2, "dials must be [H, total_rows], got ", dials->dim(), " dims");
    p.dials = f32_ptr(*dials, "dials");
  } else {
    p.dials = nullptr;
  }

  p.read_bits = u32_ptr(read_bits, "read_bits");
  p.read_tok_bits = u32_ptr(read_tok_bits, "read_tok_bits");
  p.write_tok_bits = u32_ptr(write_tok_bits, "write_tok_bits");
  p.write_bits = u32_ptr(write_bits, "write_bits");
  p.read_union = u32_ptr(read_union, "read_union");
  p.write_union = u32_ptr(write_union, "write_union");
  p.owners = owners.data_ptr<int32_t>();
  p.g_read = g_read.has_value() ? f32_ptr(*g_read, "g_read") : nullptr;
  p.g_write = f32_ptr(g_write, "g_write");
  p.v = f32_ptr(v, "v");
  TORCH_CHECK(state.scalar_type() == at::kFloat && state.is_contiguous(),
              "state must be contiguous float32");
  p.state = state.data_ptr<float>();
  p.page_table = page_table.has_value() ? page_table->data_ptr<int32_t>() : nullptr;

  TORCH_CHECK(!(global_norm && g_read.has_value()),
              "normalization='global' fixes g_read := 1 (spec §1); pass g_read=None");
  TORCH_CHECK(global_norm || g_read.has_value(), "normalization='raw' requires g_read");

  auto num = at::zeros({BH, T, d_v}, v.options());
  at::Tensor den;
  if (global_norm) {
    den = at::zeros({BH, T}, v.options());
    p.den = den.data_ptr<float>();
  } else {
    p.den = nullptr;
  }
  p.num = num.data_ptr<float>();

  p.N = N;
  p.BH = static_cast<int>(BH);
  p.T = static_cast<int>(T);
  p.n_tiles = static_cast<int>(amp_read[0].size(1));
  p.n_words = static_cast<int>(read_bits.size(2));
  p.total_rows = static_cast<int>(read_bits.size(1));
  TORCH_CHECK(read_tok_bits.size(1) == p.n_tiles, "tok_bits' tile axis (", read_tok_bits.size(1),
              ") must match the amplitude "
              "stream's (",
              p.n_tiles, ") -- a schedule built at a different BT");
  p.n_slices = static_cast<int>(n_slices);
  p.D = static_cast<int>(D);
  p.first_spanned_level = static_cast<int>(first_spanned_level);
  p.slice_digits = static_cast<int>(slice_digits);
  p.n_owners = static_cast<int>(owners.size(0));
  TORCH_CHECK(n_heads >= 1 && BH % n_heads == 0, "H=", n_heads, " must be >= 1 and divide BH=", BH);
  p.H = static_cast<int>(n_heads);
  if (decay) {
    TORCH_CHECK(dials->size(0) == n_heads && dials->size(1) == p.total_rows,
                "dials must be [H, total_rows] = [", n_heads, ", ", p.total_rows, "], got [",
                dials->size(0), ", ", dials->size(1), "]");
  }
  TORCH_CHECK(p.n_words <= kMaxWords, "n_words=", p.n_words,
              " exceeds the per-CTA SMEM candidate budget kMaxWords=", kMaxWords, " (T > ",
              static_cast<int64_t>(kMaxWords) * 32 * BT,
              " at this BT); raise kMaxWords or tile the token range");

  // Numeric FORMAT axis (findings D4.3), stamped by the plan, resolved once here.
  // Not a tolerance knob and not a density branch.
  TORCH_CHECK(mma_precision == "tf32x3" || mma_precision == "bf16",
              "mma_precision must be 'tf32x3' or 'bf16', got ", mma_precision);

  // ---- §6.4 sequence parallelism: three passes, two launch boundaries. Sequenced
  // HERE rather than by three Python calls because the three passes are ONE
  // CONTRACT: docs/internals/consumer.md#all-or-nothing
  const int64_t n_sp = static_cast<int64_t>(work_seg.has_value()) + work_nseg.has_value()
                       + work_slot.has_value() + summary_owners.has_value()
                       + summary_seg.has_value() + summary_nseg.has_value()
                       + summary_slot.has_value() + scan_owners.has_value() + scan_nseg.has_value()
                       + scan_slot.has_value() + ws_state.has_value() + ws_scale.has_value()
                       + ws_leaf.has_value() + ws_count.has_value();
  TORCH_CHECK(n_sp == 0 || n_sp == 14,
              "the §6.4 sequence-parallel arguments are all-or-nothing: got ", n_sp,
              " of 14. A partial split plan would run some passes and not others, "
              "which is not a degraded answer but a wrong one.");
  const bool split = n_sp == 14;

  const int BT_i = static_cast<int>(BT);
  const int BC_i = static_cast<int>(BC);
  const int DV_i = static_cast<int>(d_v);

  //: ONE dispatch point for every consumer launch this call makes, so a pass cannot
  //: pick a different kernel from the one its siblings picked. `P` is a WORK-LIST
  //: SHAPE, not a kernel.
  auto launch_pass = [&](ConsumerParams& pp) {
    launch_consumer(pp, BT_i, BC_i, DV_i, global_norm, decay, mma_precision, compact);
  };
  if (split) {
    auto i32 = [](const at::Tensor& t, const char* name, int64_t rows) {
      TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kInt, name,
                  " must be a contiguous int32 CUDA tensor");
      TORCH_CHECK(t.dim() >= 1 && t.size(0) == rows, name, " must have ", rows, " rows");
      return t.data_ptr<int32_t>();
    };
    const int64_t n_work = owners.size(0);
    const int64_t n_summary = summary_owners->size(0);
    const int64_t n_split = scan_owners->size(0);
    TORCH_CHECK(summary_owners->dim() == 2 && summary_owners->size(1) == 2,
                "summary_owners must be [n_summary, 2]");
    TORCH_CHECK(scan_owners->dim() == 2 && scan_owners->size(1) == 2,
                "scan_owners must be [n_split, 2]");
    TORCH_CHECK(ws_state->dim() == 3 && ws_state->size(1) == BC && ws_state->size(2) == cols,
                "ws_state must be [n_slots, BC, d_v(+1 under global)]");
    TORCH_CHECK(
        ws_state->is_cuda() && ws_state->is_contiguous() && ws_state->scalar_type() == at::kFloat,
        "ws_state must be contiguous float32 on the device");
    TORCH_CHECK(ws_scale->is_cuda() && ws_scale->is_contiguous()
                    && ws_scale->scalar_type() == at::kFloat && ws_scale->dim() == 2
                    && ws_scale->size(0) == ws_state->size(0) && ws_scale->size(1) == BC,
                "ws_scale must be contiguous float32 [n_slots, BC]");
    const int64_t n_slots = ws_state->size(0);
    p.ws_state = ws_state->data_ptr<float>();
    p.ws_scale = ws_scale->data_ptr<float>();
    p.ws_leaf = i32(*ws_leaf, "ws_leaf", n_slots);
    p.ws_count = i32(*ws_count, "ws_count", n_slots);

    // ---- pass 1: the per-segment summaries. The grid OMITS every owner's last
    // segment. docs/internals/consumer.md#all-or-nothing
    p.owners = i32(*summary_owners, "summary_owners", n_summary);
    p.n_owners = static_cast<int>(n_summary);
    p.work_seg = i32(*summary_seg, "summary_seg", n_summary);
    p.work_nseg = i32(*summary_nseg, "summary_nseg", n_summary);
    p.work_slot = i32(*summary_slot, "summary_slot", n_summary);
    p.pass_mode = kPassSummary;
    launch_pass(p);

    // ---- pass 2: the ordered fold ------------------------------------------
    ScanParams s{};
    s.owners = i32(*scan_owners, "scan_owners", n_split);
    s.nseg = i32(*scan_nseg, "scan_nseg", n_split);
    s.slot = i32(*scan_slot, "scan_slot", n_split);
    s.ws_state = p.ws_state;
    s.ws_scale = p.ws_scale;
    s.ws_leaf = p.ws_leaf;
    s.ws_count = p.ws_count;
    s.state = p.state;
    s.page_table = p.page_table;
    s.N = N;
    s.BC = BC_i;
    s.cols = static_cast<int>(cols);
    s.n_split = static_cast<int>(n_split);
    scan_pass(s);

    // ---- pass 3: the reads --------------------------------------------------
    p.owners = owners.data_ptr<int32_t>();
    p.n_owners = static_cast<int>(n_work);
    p.work_seg = i32(*work_seg, "work_seg", n_work);
    p.work_nseg = i32(*work_nseg, "work_nseg", n_work);
    p.work_slot = i32(*work_slot, "work_slot", n_work);
    p.pass_mode = kPassRead;
    launch_pass(p);
  } else {
    launch_pass(p);
  }

  at::Tensor y = num;
  if (global_norm) {
    y = at::empty_like(num);
    divide_pass(num.data_ptr<float>(), den.data_ptr<float>(), y.data_ptr<float>(), BH * T,
                static_cast<int>(d_v), static_cast<float>(eps));
  }
  return {y, state, num, global_norm ? den : at::Tensor()};
}

}  // namespace rola
