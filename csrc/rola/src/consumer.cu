// RoLA V3 consumer -- THE HOST DISPATCH TRANSLATION UNIT.
//
// ROLA_V3_SPEC.md §3 (topology-agnostic kernel structure), §4 (schedule), §5
// (consumer), §6.4 (sequence parallelism), §7.1 (paging).
//
// WHAT IS IN THIS FILE, and nothing else is:
//   * the Torch-level entry `rola_consumer_forward` -- argument checking, the
//     `ConsumerParams` fill, and §6.4's three-pass sequencing;
//   * the launch dispatch `launch_consumer`, which resolves the plan's structural axes
//     to ONE template instantiation of `consumer::consumer_forward_kernel`;
//   * the two non-template helper kernels, `scan_kernel` (§6.4 pass 2's ordered
//     fold) and `divide_kernel` (`global`'s `num/den` epilogue);
//   * the driver-backed residency and footprint queries, and the arch-table check;
//   * the pybind surface.
//
// THE KERNEL ITSELF IS IN `consumer_kernel.cuh`, which is also where the §3/§5
// algorithm and every structural decision behind it are documented. This file
// deliberately does NOT restate them: two descriptions of one kernel is one that can
// go stale.
//
// ONE CONSUMER (item 6b, 2026-08-01). The previous register-resident consumer
// (its kernel header -- also named `consumer_kernel.cuh`, readable at
// `c3c2f845`, and a DIFFERENT kernel from the one this file now includes --
// plus `intra_gram.cuh`, `consumer_matrix.def` and its eight shard TUs), its `intra_lowering` template axis, its analytic SMEM mirror and
// its intra-kernel tracer are DELETED. There is one kernel family, so there is no
// second-kernel routing argument, no `intra_lowering` argument, and no second occupancy
// oracle: what the planner prices and what the driver launches are the same thing by
// construction rather than by a gate that pins two accountings together.

// The CUDA/ATen surface this TU launches and checks against. It used to arrive
// transitively through the register-resident consumer's kernel header; with that
// header deleted the dependency is stated where it is used.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "consumer_kernel.cuh"
#include "host_api.cuh"

namespace rola {

// THE §10i.12 RESTRUCTURE, and the ONLY consumer. See `consumer_kernel.cuh`
// for the algorithm; the supported shape surface is declared and TORCH_CHECKed at
// `launch_consumer` below, which is its single authority.

namespace {

// ---------------------------------------------------------------------------
// §6.4 pass 2: the ordered fold
// ---------------------------------------------------------------------------
//
// One CTA per SPLIT owner; the composition is sequential over that owner's
// segments inside it. `GRADIENT_PROOFS_WOLFRAM.md` P5.21's obligation is that the
// fold is ORDERED and that nothing anywhere inverts `A` -- both hold literally
// here: the loop runs `s = 0 .. nseg-2` ascending and the only operation is
// `E := A*E + B`.
//
// The boundary is published DENSE, in place over the sparse summary the same slot
// just supplied. §6.4 states both halves: the summary is normatively sparse
// (traffic scales with realized writes, not with `P`), while the accumulated
// boundary pass 3 reads is dense because cumulative write coverage saturates. They
// are the same bytes at different points of the slot's life, which is what keeps the
// per-boundary cost one slot rather than two.
//
// The owner's slot run is `P` slots, not `P - 1` (fix pass R1, spec §6.4's bound
// amended to match): slot `slot0` is the ENTRY-STATE STASH this kernel publishes
// below, and slots `slot0 + 1 .. slot0 + P - 1` are the `P - 1` summaries/boundaries.
// The stash is what removes pass 3's intra-launch dependence on `p.state`.
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

  // The owner's entry state, addressed exactly as the consumer addresses it (§7.1:
  // paging changes which base the owner's `BC` contiguous rows start at, nothing
  // else), so a paged and an unpaged fold see the same numbers.
  bool resident = true;
  int64_t base;
  if (p.page_table != nullptr) {
    const int physical = p.page_table[static_cast<int64_t>(bh) * p.owners_total + owner];
    resident = physical >= 0;
    base = static_cast<int64_t>(resident ? physical : 0) * BC;
  } else {
    base = static_cast<int64_t>(bh) * p.N + static_cast<int64_t>(owner) * BC;
  }
  for (int e = threadIdx.x; e < n; e += blockDim.x) {
    E[e] = resident ? p.state[base * cols + e] : 0.0f;
  }
  __syncthreads();

  // ---- the ENTRY-STATE STASH (fix pass R1) --------------------------------
  // Slot `slot0` holds `M_0` itself, dense, published HERE -- in pass 2, a launch
  // that strictly precedes pass 3. Pass 3's segment 0 reads it instead of reading
  // `p.state`, which is what makes pass 3's reads disjoint from pass 3's writes
  // (the last segment of the same owner stores those exact `p.state` rows from
  // another CTA of that launch, with no ordering between them). The read this
  // replaces was not free before -- segment 0 read the same numbers out of
  // `p.state` -- so the added TRAFFIC is one dense store per split owner, and the
  // added SPACE is the one slot that takes §6.4's bound from `(P-1)` to `P` per
  // split owner. A non-resident owner stashes zeros, which is exactly what its
  // segment 0 read before.
  for (int e = threadIdx.x; e < n; e += blockDim.x) {
    p.ws_state[static_cast<int64_t>(slot0) * n + e] = E[e];
  }
  __syncthreads();

  for (int s = 0; s < nseg - 1; ++s) {
    const int64_t slot = static_cast<int64_t>(slot0) + 1 + s;
    const int K = p.ws_count[slot];
    // `E := A*E + B` ON THE WRITTEN-LEAF LIST ONLY. A leaf absent from the list is
    // not read, not written, and not arithmetically touched -- which is what makes
    // §6.4's unwritten-leaf carve-out BITWISE rather than merely equal, and is why
    // the carve-out gate is a genuine detector of fold bugs rather than a
    // concession. Emitting an identity row for it would be numerically exact too,
    // but it would make the invariant depend on `fmaf(1, x, 0) == x` instead of on
    // the leaf never being visited.
    for (int e = threadIdx.x; e < K * cols; e += blockDim.x) {
      const int k = e / cols;
      const int c = e - k * cols;
      const int i = p.ws_leaf[slot * BC + k];
      E[i * cols + c] = fmaf(p.ws_scale[slot * BC + k], E[i * cols + c],
                             p.ws_state[(slot * BC + k) * cols + c]);
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
//: PLAN rather than reading it at launch, so that a plan is reproducible and its
//: residency reasoning is auditable off the machine that made it. The planner's
//: `DEFAULT_SM_SHARED_BYTES` is the CPU-only fallback and is deliberately a plan
//: field, not a hidden default.
int64_t rola_sm_shared_bytes() {
  int value = 0;
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &value, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
  return value;
}

void divide_pass(const float* num, const float* den, float* out, int64_t rows, int d_v, float eps) {
  const int64_t total = rows * d_v;
  const int threads = 256;
  const int64_t blocks = (total + threads - 1) / threads;
  divide_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      num, den, out, rows, d_v, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

//: VALIDATE THE COMPILE-TIME ARCH TABLE AGAINST THE DRIVER.
//:
//: `__launch_bounds__` needs compile-time constants, so `sm_resources()` is a table
//: keyed on `__CUDA_ARCH__` rather than a driver query. A table can be stale or
//: mis-keyed in a way a query cannot, and the failure mode is SILENT: a bound derived
//: from the wrong resource numbers is either DISCARDED by `ptxas` (out of range) or
//: honoured at the wrong residency. So the table is CHECKED against `cudaDeviceProp`
//: for the device actually running, and a mismatch is a hard error -- a
//: compile-target/hardware mismatch must fail loudly rather than quietly cost
//: occupancy.
//:
//: **THIS DOES NOT MAKE THE TABLE A PLANNING AUTHORITY.** Spec §6.5: the driver's
//: occupancy API remains the sole authority on residency, and
//: `rola_consumer_occupancy` -- which calls it -- is what every residency claim in
//: this batch is read from. The table's only job is the launch bound.
void rola_consumer_check_arch_table() {
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  const int arch = prop.major * 100 + prop.minor * 10;
  TORCH_CHECK(sm_arch_tabulated(arch),
              "the new consumer's SM-resource table has no row for compute capability ",
              prop.major, ".", prop.minor,
              "; add it to sm_resources() rather than letting the launch bound guess");
  const SmResources want = sm_resources(arch);
  int smem_per_sm = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
  TORCH_CHECK(want.smem_per_sm == smem_per_sm &&
                  want.max_threads_per_sm == prop.maxThreadsPerMultiProcessor &&
                  want.regs_per_sm == prop.regsPerMultiprocessor,
              "the new consumer's SM-resource table disagrees with the driver at "
              "compute capability ", prop.major, ".", prop.minor, ": table {smem ",
              want.smem_per_sm, ", threads ", want.max_threads_per_sm, ", regs ",
              want.regs_per_sm, "} vs driver {smem ", smem_per_sm, ", threads ",
              prop.maxThreadsPerMultiProcessor, ", regs ", prop.regsPerMultiprocessor,
              "}. The launch bounds compiled into this binary were derived from the "
              "table, so they are wrong for this device.");
}

// THE LAUNCH DISPATCH. Declared restrictions are hard checks, never coercions.
void launch_consumer(const ConsumerParams& p, int BT, int BC, int DV, bool global_norm,
                  bool decay, const std::string& mma_precision, bool compact) {
  //: Checked ONCE per process, on the first launch rather than at import: a
  //: `cudaGetDeviceProperties` at module init would create a CUDA context as a side
  //: effect of `import`, which is a worse behaviour than the check is worth. By the
  //: first launch the context exists anyway, and nothing can reach a launch bound
  //: without passing through here.
  static const bool arch_checked = [] { rola_consumer_check_arch_table(); return true; }();
  (void)arch_checked;
  //: THE FULL PLAN SURFACE, as of item 6a: `BT in {16, 32}` (`_SUPPORTED_BT`),
  //: `BC in {16, 64}` (`_SUPPORTED_BC`), `d_v in {32, 64}` (`_SUPPORTED_DV`). These
  //: are the planner's own supported sets, so the kernel instantiates exactly the
  //: configurations a plan can name: no shape a plan can select is refused, and no
  //: shape it cannot select is compiled.
  TORCH_CHECK((BT == 16 || BT == 32) && (BC == 16 || BC == 64) &&
                  (DV == 32 || DV == 64),
              "the consumer instantiates BT in {16, 32}, BC in {16, 64}, "
              "d_v in {32, 64}; got BT=", BT, " BC=", BC, " d_v=", DV);

  //: `mma_precision` selects BOTH GEMM sites together (RULING 1 revised): "bf16" is
  //: the production point, "tf32x3" the exactness-gate point. Both are instantiated.
  const bool exact = (mma_precision == "tf32x3");
  //: THE INSTANTIATION MATRIX, AND WHY EVERY ONE OF ITS AXES IS A TEMPLATE.
  //:
  //:   `BT(2) x BC(2) x DV(2) x D(4) x PREC(2) x DECAY(2) x GLOBAL(2) x COMPACT(2)`
  //:   = **512** (item 6a; it was 128 before `BT` and `d_v` were lifted).
  //:
  //: Each is a STRUCTURAL plan fact, fixed before launch, with few legal values --
  //: the standing "branch on structure, not thresholds" rule, and the same status
  //: `BT` and `BC` have always had. `D` is the topology's level count (spec §1:
  //: `1 <= D <= 4`); `DECAY` changes which terms exist; `GLOBAL` changes the
  //: STATE'S SHAPE (the `+1` mass column) and so cannot be a runtime test without
  //: making `cols` a runtime value in every state address.
  //:
  //: **There is deliberately no runtime fallback on ANY of them.** An out-of-range
  //: value is a hard `TORCH_CHECK`, never a slower path: one comprehensive path
  //: ships. Compile time is WATCHED rather than assumed -- measured at 64 s for 32
  //: instantiations in this one translation unit. A `.def`-plus-shard-TU split (the
  //: mechanism the deleted consumer's 256-tuple matrix used, readable at `c3c2f845`)
  //: remains the answer if a later axis makes that false.
#define ROLA_CONSUMER_ONE(BTV, DVV, BCVAL, DVAL, PRECV, DECAYV, GLOBALV, COMPACTV)          \
  {                                                                                    \
    constexpr size_t smem =                                                            \
        consumer::consumer_smem_bytes<BTV, BCVAL, DVV, DVAL, PRECV, DECAYV, GLOBALV,         \
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
  //: **THE BRACES ARE LOAD-BEARING.** Every `ROLA_CONSUMER_ONE` is a braced block ending
  //: in `return`, so the enclosing dispatch is a chain of `if (cond) BLOCK`. This
  //: macro expands to TWO blocks, and without the outer braces the second one would
  //: escape its guard and launch UNCONDITIONALLY -- which is exactly what an earlier
  //: revision did: every call ran the `decay` instantiation, which dereferenced a
  //: null `p.dials`. Caught by the compact gate's own dense arm, i.e. by the
  //: instrument this axis was added with.
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

//: The new consumer's own residency answer, from the DRIVER, for the report's A/B
//: table -- and it takes `decay` because RESIDENCY IS PER-INSTANTIATION. The decay
//: arm carries three extra `[BC]` vectors, so a single reported number would either
//: over-promise on one arm or under-report the other; the whole point of deriving
//: `__launch_bounds__` from `consumer_smem_bytes` is that a decay instantiation which
//: regressed the budget would drop its OWN bound rather than silently over-promise,
//: and this is how that is checked from Python rather than asserted.
int64_t rola_consumer_occupancy(bool decay, bool compact, int64_t BC, int64_t BT,
                                int64_t DV) {
  //: Queried at `D = 2`, the at-scale cell's topology depth. `D` changes only the
  //: `[D][BC]` digit table (256 B per level), so the residency answer is reported for
  //: the shape actually measured rather than for a nominal worst case.
  //:
  //: `BC` IS A QUERY AXIS because it is the axis the launch bound's binding term
  //: changes on: at `BC = 64` shared memory binds (3 CTAs), at `BC = 16` the REGISTER
  //: FILE does (4). A residency claim about "the kernel" that quoted only `BC = 64`
  //: would be a claim about half the matrix.
  int blocks = 0;
#define ROLA_CONSUMER_OCC(BTV, DVV, BCV, DECAYV, COMPACTV)                                 \
  {                                                                                   \
    constexpr size_t smem =                                                           \
        consumer::consumer_smem_bytes<BTV, BCV, DVV, 2, consumer::kReadPrecDefault, DECAYV,    \
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
  //: `compact` joins `decay` as a QUERY AXIS for the same reason: the compact arm
  //: carries `c_idx`/`c_slot`/`c_count`, and `compact + decay` is the tightest
  //: residency cell in the kernel (~140 B of margin at 3 CTAs/SM). A single reported
  //: number would over-promise on it.
  //: `BT` AND `d_v` JOIN THE QUERY AXES (item 6a) for exactly the reason `BC` and
  //: `DECAY` are already on it: they move the two terms the bound is the minimum of.
  //: `BT = 32` doubles the y-fragment count (the REGISTER term) and grows `WT`/`Gm`
  //: (the SMEM term); `d_v = 32` halves the state block. A residency number that
  //: quoted only `BT = 16, d_v = 64` would be a claim about a quarter of the matrix.
  TORCH_CHECK(BC == 16 || BC == 64, "BC must be 16 or 64, got ", BC);
  TORCH_CHECK(BT == 16 || BT == 32, "BT must be 16 or 32, got ", BT);
  TORCH_CHECK(DV == 32 || DV == 64, "d_v must be 32 or 64, got ", DV);
  if (BT == 16) ROLA_CONSUMER_OCC_BT(16) else ROLA_CONSUMER_OCC_BT(32)
#undef ROLA_CONSUMER_OCC_BT
#undef ROLA_CONSUMER_OCC_DV
#undef ROLA_CONSUMER_OCC_BC
#undef ROLA_CONSUMER_OCC
  return blocks;
}

int64_t rola_consumer_smem_bytes(bool decay, bool compact, int64_t BC, int64_t BT,
                                 int64_t DV) {
#define ROLA_CONSUMER_SMEM(BTV, DVV, BCV, DECAYV, COMPACTV)                                \
  return static_cast<int64_t>(                                                        \
      consumer::consumer_smem_bytes<BTV, BCV, DVV, 2, consumer::kReadPrecDefault, DECAYV, true,\
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
// (`routing/planner/schedule.py`) plus the value stream and the dense state, and
// returns `(y, final_state)`. Every argument is checked loudly; there is no
// silent shape coercion and no CPU fallback.
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
  TORCH_CHECK(t.scalar_type() == at::kInt, name, " must be int32 bit patterns, got ", t.scalar_type());
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  return reinterpret_cast<const uint32_t*>(t.data_ptr<int32_t>());
}

}  // namespace

std::vector<at::Tensor> rola_consumer_forward(
    std::vector<at::Tensor> amp_read,
    std::vector<at::Tensor> amp_write,
    c10::optional<std::vector<at::Tensor>> amp_clock,
    c10::optional<at::Tensor> dials,
    at::Tensor read_bits,
    at::Tensor write_bits,
    at::Tensor read_tok_bits,
    at::Tensor write_tok_bits,
    at::Tensor read_union,
    at::Tensor write_union,
    at::Tensor owners,
    c10::optional<at::Tensor> admit,
    c10::optional<at::Tensor> g_read,
    at::Tensor g_write,
    at::Tensor v,
    at::Tensor state,
    c10::optional<at::Tensor> page_table,
    std::vector<int64_t> level_stride,
    std::vector<int64_t> level_modulus,
    std::vector<int64_t> level_row_offset,
    std::vector<int64_t> level_width,
    std::vector<int64_t> owner_stride,
    int64_t first_spanned_level,
    int64_t slice_digits,
    int64_t n_slices,
    int64_t owners_total,
    int64_t n_heads,
    int64_t BT,
    int64_t BC,
    bool global_norm,
    std::string mma_precision,
    bool compact,
    int64_t ring_slots,
    double eps,
    // ---- §6.4 sequence parallelism. All-or-nothing: either every one of these is
    // present (a split plan) or none is (`P = 1`, the primary configuration, which
    // takes the single-launch path below unchanged).
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
    c10::optional<at::Tensor> ws_count) {       // [n_slots] int32
  const int64_t D = static_cast<int64_t>(amp_read.size());
  TORCH_CHECK(D >= 1 && D <= kMaxLevels, "D must be in [1, 4] (spec §1), got ", D);
  TORCH_CHECK(static_cast<int64_t>(amp_write.size()) == D, "amp_write must have D entries");

  TORCH_CHECK(v.dim() == 3, "v must be [BH, T, d_v], got ", v.dim(), " dims");
  const int64_t BH = v.size(0);
  const int64_t T = v.size(1);
  const int64_t d_v = v.size(2);
  // §7.1: paging changes the state's SHAPE, not its addressing rule. Unpaged it is
  // `[BH, N, cols]` and an owner starts at `bh * N + owner * BC`; paged it is
  // `[physical_pages, BC, cols]` and an owner starts at `page_table[...] * BC`.
  // `N` is then not readable off the state, so it is derived from the owner
  // partition -- which is the definition, `N = owners_total * BC`.
  const int64_t cols = d_v + (global_norm ? 1 : 0);
  int64_t N;
  if (page_table.has_value()) {
    TORCH_CHECK(state.dim() == 3 && state.size(1) == BC && state.size(2) == cols,
                "paged state must be [physical_pages, BC, d_v(+1 under global)]");
    const auto& table = *page_table;
    TORCH_CHECK(table.is_cuda() && table.is_contiguous() &&
                    table.scalar_type() == at::kInt && table.dim() == 2 &&
                    table.size(0) == BH && table.size(1) == owners_total,
                "page_table must be contiguous int32 [BH, owners_total] on the state's device");
    N = owners_total * BC;
  } else {
    TORCH_CHECK(state.dim() == 3, "state must be [BH, N, d_v(+1)]");
    TORCH_CHECK(state.size(0) == BH, "state batch-head dim must match v");
    N = state.size(1);
    TORCH_CHECK(state.size(2) == cols,
                "state's last dim must be d_v(+1 under global), got ", state.size(2));
  }
  TORCH_CHECK(owners.dim() == 2 && owners.size(1) == 2, "owners must be [n_owners, 2]");
  TORCH_CHECK(owners.scalar_type() == at::kInt, "owners must be int32");
  TORCH_CHECK(read_bits.dim() == 3, "read_bits must be [BH, total_rows, n_words]");
  // §4's support word, TILE-MAJOR. Shape-checked against the tile bitmap it is the
  // refinement of (same `total_rows` level-digit rows) and, below, against the
  // amplitude stream's tile axis -- so a stale or mis-shaped schedule cannot reach
  // the gather.
  TORCH_CHECK(read_tok_bits.dim() == 3 && write_tok_bits.sizes() == read_tok_bits.sizes(),
              "read/write_tok_bits must both be [BH, n_tiles, total_rows]");
  TORCH_CHECK(read_tok_bits.size(0) == read_bits.size(0) &&
                  read_tok_bits.size(2) == read_bits.size(1),
              "tok_bits must carry the same [BH, total_rows] level-digit rows as the tile bitmap");
  TORCH_CHECK(ring_slots >= 1 && ring_slots <= kMaxRingSlots,
              "ring_slots must be in [1, ", kMaxRingSlots, "] (1 declines the staging "
              "ring); it is a PLAN value selected under §6's occupancy constraint, got ",
              ring_slots);
  TORCH_CHECK(BT >= 1 && BT <= 32,
              "BT must be in [1, 32] (spec §6): the support word carries one bit per token "
              "of a tile, got BT=", BT);

  ConsumerParams p{};
  for (int64_t l = 0; l < D; ++l) {
    TORCH_CHECK(amp_read[l].dim() == 4, "amp_read[", l, "] must be [BH, n_tiles, BT, pitch]");
    p.amp_read[l] = f32_ptr(amp_read[l], "amp_read");
    p.amp_write[l] = f32_ptr(amp_write[l], "amp_write");
    p.level_pitch_read[l] = static_cast<int>(amp_read[l].size(3));
    p.level_pitch_write[l] = static_cast<int>(amp_write[l].size(3));
    if (amp_clock.has_value()) {
      TORCH_CHECK((*amp_clock)[l].dim() == 4, "amp_clock[", l, "] must be [BH, n_tiles, BT, pitch]");
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

  p.ring_slots = static_cast<int>(ring_slots);
  p.read_bits = u32_ptr(read_bits, "read_bits");
  p.read_tok_bits = u32_ptr(read_tok_bits, "read_tok_bits");
  p.write_tok_bits = u32_ptr(write_tok_bits, "write_tok_bits");
  p.write_bits = u32_ptr(write_bits, "write_bits");
  p.read_union = u32_ptr(read_union, "read_union");
  p.write_union = u32_ptr(write_union, "write_union");
  p.owners = owners.data_ptr<int32_t>();
  p.admit = admit.has_value() ? u32_ptr(*admit, "admit") : nullptr;
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
  TORCH_CHECK(read_tok_bits.size(1) == p.n_tiles,
              "tok_bits' tile axis (", read_tok_bits.size(1), ") must match the amplitude "
              "stream's (", p.n_tiles, ") -- a schedule built at a different BT");
  p.n_slices = static_cast<int>(n_slices);
  p.D = static_cast<int>(D);
  p.first_spanned_level = static_cast<int>(first_spanned_level);
  p.slice_digits = static_cast<int>(slice_digits);
  p.owners_total = static_cast<int>(owners_total);
  p.n_owners = static_cast<int>(owners.size(0));
  TORCH_CHECK(n_heads >= 1 && BH % n_heads == 0,
              "H=", n_heads, " must be >= 1 and divide BH=", BH);
  p.H = static_cast<int>(n_heads);
  if (decay) {
    TORCH_CHECK(dials->size(0) == n_heads && dials->size(1) == p.total_rows,
                "dials must be [H, total_rows] = [", n_heads, ", ", p.total_rows, "], got [",
                dials->size(0), ", ", dials->size(1), "]");
  }
  // `gram` staging geometry: level `l >= j` contributes `modulus` columns to the
  // per-tile staged amplitude rows. Derived from the SAME per-level arrays the
  // digit lowering uses, so it cannot drift from the owner rectangle it describes.
  {
    int off = 0;
    for (int64_t l = 0; l < D; ++l) {
      p.level_gram_offset[l] = off;
      if (l >= first_spanned_level) off += p.level_modulus[l];
    }
    p.gram_cols = off;
  }

  TORCH_CHECK(p.n_words <= kMaxWords,
              "n_words=", p.n_words, " exceeds the per-CTA SMEM candidate budget kMaxWords=",
              kMaxWords, " (T > ", static_cast<int64_t>(kMaxWords) * 32 * BT,
              " at this BT); raise kMaxWords or tile the token range");

  // Numeric FORMAT axis (findings D4.3), stamped by the plan, resolved once here.
  // Not a tolerance knob and not a density branch: tf32x3 is what the exactness
  // gate against the fp64 oracle runs, bf16 is the performance configuration.
  TORCH_CHECK(mma_precision == "tf32x3" || mma_precision == "bf16",
              "mma_precision must be 'tf32x3' or 'bf16', got ", mma_precision);
  // Both decay arms honour it: a read-only tile under LeafMassDecay takes the §5
  // readout (its MIXED tiles take the token-sequential walk, which is fp32 and
  // has no operand format to choose), so no stamped field is ignored here.

  // ---- §6.4 sequence parallelism: three passes, two launch boundaries ------
  // Sequenced HERE rather than by three Python calls, because the three passes are
  // one contract: pass 3 is only correct against the boundary pass 2 published from
  // the summary pass 1 emitted, and interposing a Python frame would let a caller
  // run them out of order or skip one. They are ordinary stream-ordered launches --
  // no device synchronization is taken and none is needed.
  const int64_t n_sp = static_cast<int64_t>(work_seg.has_value()) +
                       work_nseg.has_value() + work_slot.has_value() +
                       summary_owners.has_value() + summary_seg.has_value() +
                       summary_nseg.has_value() + summary_slot.has_value() +
                       scan_owners.has_value() + scan_nseg.has_value() +
                       scan_slot.has_value() + ws_state.has_value() +
                       ws_scale.has_value() + ws_leaf.has_value() + ws_count.has_value();
  TORCH_CHECK(n_sp == 0 || n_sp == 14,
              "the §6.4 sequence-parallel arguments are all-or-nothing: got ", n_sp,
              " of 14. A partial split plan would run some passes and not others, "
              "which is not a degraded answer but a wrong one.");
  const bool split = n_sp == 14;

  const int BT_i = static_cast<int>(BT);
  const int BC_i = static_cast<int>(BC);
  const int DV_i = static_cast<int>(d_v);

  //: ONE dispatch point for every consumer launch this call makes -- the unsplit
  //: one, and §6.4's pass 1 and pass 3. Written once so a pass cannot pick a
  //: different kernel from the one its siblings picked. `P` is a WORK-LIST SHAPE,
  //: not a kernel: both consumers read `work_seg`/`work_nseg`/`work_slot`/
  //: `pass_mode` out of the same parameter block.
  auto launch_pass = [&](ConsumerParams& pp) {
    launch_consumer(pp, BT_i, BC_i, DV_i, global_norm, decay, mma_precision, compact);
  };
  if (split) {
    auto i32 = [](const at::Tensor& t, const char* name, int64_t rows) {
      TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kInt,
                  name, " must be a contiguous int32 CUDA tensor");
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
    TORCH_CHECK(ws_state->is_cuda() && ws_state->is_contiguous() &&
                    ws_state->scalar_type() == at::kFloat,
                "ws_state must be contiguous float32 on the device");
    TORCH_CHECK(ws_scale->is_cuda() && ws_scale->is_contiguous() &&
                    ws_scale->scalar_type() == at::kFloat && ws_scale->dim() == 2 &&
                    ws_scale->size(0) == ws_state->size(0) && ws_scale->size(1) == BC,
                "ws_scale must be contiguous float32 [n_slots, BC]");
    const int64_t n_slots = ws_state->size(0);
    p.ws_state = ws_state->data_ptr<float>();
    p.ws_scale = ws_scale->data_ptr<float>();
    p.ws_leaf = i32(*ws_leaf, "ws_leaf", n_slots);
    p.ws_count = i32(*ws_count, "ws_count", n_slots);

    // ---- pass 1: the per-segment summaries ---------------------------------
    // Grid is the summary work list, which omits every owner's LAST segment: that
    // segment's outcome is the owner's final state and pass 3 produces it directly,
    // so summarizing it would be `P`th of the work for nothing. The owner's slot run
    // is nonetheless `P` slots: `P - 1` boundaries plus pass 2's entry-state stash
    // (fix pass R1 -- it is what keeps pass 3's reads disjoint from pass 3's writes).
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
    s.owners_total = p.owners_total;
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
    divide_pass(num.data_ptr<float>(), den.data_ptr<float>(), y.data_ptr<float>(),
              BH * T, static_cast<int>(d_v), static_cast<float>(eps));
  }
  return {y, state, num, global_norm ? den : at::Tensor()};
}

}  // namespace rola
