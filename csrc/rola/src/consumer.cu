// RoLA V3 consumer -- THE HOST DISPATCH TRANSLATION UNIT.
// ROLA_V3_SPEC.md §3 (structure), §4 (schedule), §5 (consumer), §6.4 (sequence
// parallelism), §7.1 (paging).
//
// WHAT IS IN THIS FILE, and nothing else is:
//   * the Torch-level entry `rola_consumer_forward` -- argument checking, the
//     `ConsumerParams` fill, and §6.4's three-pass sequencing;
//   * the launch dispatch `launch_consumer`, which resolves the plan's structural
//     axes to ONE template instantiation of `consumer::consumer_forward_kernel`;
//   * the three non-template helper kernels, `scan_kernel` (§6.4 pass 2's ordered
//     fold), `emit_reduce_kernel` (the split-K emission's fixed-order fold) and
//     `divide_kernel` (`global`'s `num/den` epilogue);
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
//
// **THIS FILE DOES NOT INCLUDE `consumer_kernel.cuh` OR `consumer_launch.cuh`, AND
// THAT IS A GATE, NOT AN ACCIDENT.** Every `consumer_forward_kernel` instantiation
// lives in exactly one generated shard under `instantiations/`; this file reaches
// them through declared-not-defined host wrappers. Were the launcher's definition
// visible here, the compiler would instantiate all 512 kernels back into this
// translation unit -- the build would link, the census would look right, and nothing
// would be sharded. `tests/unit/test_shard_partition.py` asserts the two `#include`s
// stay absent, and `tools/fatbin_entries.py` proves the outcome from the built
// binary rather than from the source.

// The CUDA/ATen surface this TU launches and checks against.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "arch_table.cuh"
#include "consumer.cuh"
#include "host_api.cuh"

namespace rola {

//: THE GENERATED DISPATCH TABLE: every arm's launcher, residency query, footprint and
//: staleness key, one contiguous block per shard, plus the mixed-radix key over the
//: matrix axes and the index that turns a key into a row. Every resolution in this
//: file -- the launch and both device-backed queries -- goes through it:
//: csrc/rola/src/instantiations/README.md
#include "instantiations/dispatch_table.inc"

namespace {

//: THE ONE RESOLUTION IN THIS FILE. A key is a mixed-radix index over the axes' value
//: positions and the index is generated beside it, so there is no axis order written
//: down twice and no search: an unreachable choice keys to -1, an absent arm indexes
//: to `nullptr`, and everything else is a load. docs/internals/consumer.md#shard-rows
const ShardRow* dispatch_row(const DispatchChoice& choice) {
  const int key = dispatch_key(choice);
  return key < 0 ? nullptr : kRowByKey[key];
}

//: THE DEVICE'S SHARE OF THE ADMISSION TEST, read ONCE per process. It is the
//: per-block OPT-IN maximum and not the per-SM budget: the opt-in is what
//: `cudaFuncSetAttribute` will grant a single block, which is the number a row's
//: footprint has to fit inside. Cached because it cannot change under a process and
//: because a launch is not the place to ask the driver a question with a fixed answer.
DeviceFacts device_facts() {
  static const DeviceFacts facts = [] {
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    int optin = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    return DeviceFacts{static_cast<size_t>(optin)};
  }();
  return facts;
}

}  // namespace

namespace {

// ---------------------------------------------------------------------------
// §6.4 pass 2: the ordered fold. One CTA per SPLIT owner, sequential over that
// owner's segments inside it, `E := A*E + B` ascending and no inverse of `A`
// anywhere (P5.21's obligation). docs/internals/consumer.md#ordered-fold
// ---------------------------------------------------------------------------

//: §4.5's page granule IS `arch_table.cuh`'s `kMmaKQuantum` -- ONE definition, which is
//: what lets `scan_kernel` address the state exactly as the consumer does without
//: including the consumer kernel's header. Mirrored on the Python side by
//: `rola.ops.paging.MMA_K_QUANTUM` and kept honest by `tests/unit/test_paging_atoms.py`
//: and the paged equivalence fixtures. docs/internals/arch_table.md#mma-quantum

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
  const int atoms_per_bh = static_cast<int>(p.N / kMmaKQuantum);
  for (int e = threadIdx.x; e < n; e += blockDim.x) {
    const int i = e / cols;
    const int c = e - i * cols;
    const int q = i / kMmaKQuantum;
    const int r = i - q * kMmaKQuantum;
    int64_t row;
    if (p.page_table != nullptr) {
      const int physical =
          p.page_table[static_cast<int64_t>(bh) * atoms_per_bh + owner * (BC / kMmaKQuantum) + q];
      row = physical >= 0 ? static_cast<int64_t>(physical) * kMmaKQuantum + r
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

// ---------------------------------------------------------------------------
// The split-K emission reduce: ONE fixed-order fold of the consumer's
// single-writer workspace slots into `num`/`den`. This launch is what makes `y`
// bit-identical across runs -- the consumer stores plain contributions into its
// own slot and the order they are combined in is decided HERE, by the host's
// `(owner, seg)`-sorted work list, never by the scheduler.
// docs/internals/consumer.md#emit-workspace
// ---------------------------------------------------------------------------

struct EmitReduceParams {
  const float* ws_num;      // [n_work, T, DV]
  const float* ws_den;      // [n_work, T] or nullptr under `raw`
  const uint32_t* rowmask;  // [n_work, n_tiles]; bit t of (w, tile) = row emitted
  const int32_t* works;     // [n_work] work ids, grouped by bh, (owner, seg) ascending
  const int32_t* off;       // [BH + 1] group bounds into `works`
  float* num;               // [BH, T, DV] -- written FULLY (empty-allocated)
  float* den;               // [BH, T] or nullptr
  int BH;
  int T;
  int DV;
  int BT;
  int n_tiles;
};

//: One THREAD per output element `(bh, token, ch)`, walking its batch-head's
//: work items in the host-fixed order -- every output is one SEQUENTIAL fp32
//: sum over an order-invariant term list, which is the whole determinism
//: argument. A warp spans consecutive channels of one token, so the row-mask
//: word is a warp-broadcast load and the payload/store are fully coalesced.
//: The `ch == 0` lane folds the token's denominator into the same walk (the
//: mask word is already in hand).
constexpr int kEmitReduceThreads = 256;

__global__ void emit_reduce_kernel(EmitReduceParams p) {
  const int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<int64_t>(p.BH) * p.T * p.DV) return;
  const int ch = static_cast<int>(idx % p.DV);
  const int64_t row = idx / p.DV;  // bh * T + token
  const int bh = static_cast<int>(row / p.T);
  const int token = static_cast<int>(row - static_cast<int64_t>(bh) * p.T);
  const int tile = token / p.BT;
  const int t = token - tile * p.BT;
  const bool den_lane = p.ws_den != nullptr && ch == 0;
  //: One SEQUENTIAL fp32 sum in the host-fixed order. A 4-way interleaved
  //: accumulator variant was measured a wash on the worst (high-K, short-T)
  //: probe cell -- the walk is bound by the workspace read-back, not by the
  //: dependent-add chain -- so the simplest order ships.
  float acc = 0.0f;
  float dacc = 0.0f;
  for (int k = p.off[bh]; k < p.off[bh + 1]; ++k) {
    const int w = p.works[k];
    const uint32_t m = p.rowmask[static_cast<int64_t>(w) * p.n_tiles + tile];
    if (((m >> t) & 1u) == 0u) continue;
    acc += p.ws_num[(static_cast<int64_t>(w) * p.T + token) * p.DV + ch];
    if (den_lane) dacc += p.ws_den[static_cast<int64_t>(w) * p.T + token];
  }
  p.num[idx] = acc;
  if (den_lane) p.den[row] = dacc;
}

void emit_reduce_pass(const EmitReduceParams& p) {
  const int64_t total = static_cast<int64_t>(p.BH) * p.T * p.DV;
  const int64_t blocks = (total + kEmitReduceThreads - 1) / kEmitReduceThreads;
  emit_reduce_kernel<<<blocks, kEmitReduceThreads, 0, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

//: `OutT` is the P09 family's post-divide cast: the DIVIDE itself is fp32 over the
//: fp32 `num`/`den` accumulators (which never narrow), and only the stored quotient
//: takes the arm's output format (bf16 under `kBf16`, float under `kTf32x3`).
//: Overloads, not `static_cast` -- the build defines
//: `__CUDA_NO_BFLOAT16_CONVERSIONS__`, so the narrowing must be the intrinsic.
__device__ __forceinline__ void store_quotient(float* out, int64_t i, float q) { out[i] = q; }
__device__ __forceinline__ void store_quotient(__nv_bfloat16* out, int64_t i, float q) {
  out[i] = __float2bfloat16(q);
}

template <typename OutT>
__global__ void divide_kernel(const float* __restrict__ num, const float* __restrict__ den,
                              OutT* __restrict__ out, int64_t rows, int DV, float eps) {
  const int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (idx >= rows * DV) return;
  const int64_t row = idx / DV;
  store_quotient(out, idx, num[idx] / (den[row] + eps));
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

template <typename OutT>
void divide_pass(const float* num, const float* den, OutT* out, int64_t rows, int d_v, float eps) {
  const int64_t total = rows * d_v;
  const int threads = 256;
  const int64_t blocks = (total + threads - 1) / threads;
  divide_kernel<OutT>
      <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(num, den, out, rows, d_v, eps);
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
                     const std::string& mma_precision) {
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

  TORCH_CHECK(p.D >= 1 && p.D <= kMaxLevels, "D must be in [1, 4] (spec §1), got ", p.D);

  //: `mma_precision` selects BOTH GEMM sites together (RULING 1 revised): "bf16" is
  //: the production point, "tf32x3" the exactness-gate point. Both are instantiated.
  //: THE INSTANTIATION MATRIX. Every axis is a template because every axis is a
  //: STRUCTURAL plan fact with few legal values -- the standing "branch on structure,
  //: not thresholds" rule -- and there is deliberately NO runtime fallback on any of
  //: them: an out-of-range value is a hard refusal, never a slower path.
  //: The axes, why each one must be compile-time, the MEASURED instantiation count
  //: (the ratification manifest is the authority, not this comment) and the measured
  //: build cost: docs/internals/consumer.md#instantiation-matrix
  const DispatchChoice choice{
      BT,    BC,         DV, p.D, mma_precision == "tf32x3" ? MmaPrec::kTf32x3 : MmaPrec::kBf16,
      decay, global_norm};
  const ShardRow* row = dispatch_row(choice);
  //: A KEY THIS BINARY HAS NO ARM FOR RESOLVES TO NOTHING, and nothing is what is
  //: launched. The checks above already admit only the matrix's own closure, so a null
  //: here means the generator and the checks disagree -- a defect to fail on.
  TORCH_CHECK(row != nullptr, "no consumer instantiation for BT=", BT, " BC=", BC, " d_v=", DV,
              " D=", p.D, " precision=", mma_precision, " decay=", decay, " global=", global_norm);
  //: THE ROW'S ENABLE PREDICATE, ON A DEVICE INPUT. Its footprint is `constexpr`; what
  //: this device grants a block is not. Refusing here names both numbers, where the
  //: opt-in would otherwise fail inside the launcher and surface as a bare CUDA error.
  const DeviceFacts device = device_facts();
  TORCH_CHECK(row_admitted(*row, device), "the consumer arm BT=", BT, " BC=", BC, " d_v=", DV,
              " D=", p.D, " precision=", mma_precision, " decay=", decay, " global=", global_norm,
              " needs ", row->smem_bytes,
              " B of dynamic shared memory and this device grants a block at most ",
              device.max_dynamic_smem_bytes,
              " B. The arm is compiled but not runnable here; select a smaller tile.");
  row->launch(p, at::cuda::getCurrentCUDAStream());
  //: THE ONE ERROR SITE. The shard's launcher is torch-free and reports through the
  //: CUDA runtime's error state -- a refused shared-memory opt-in and a failed launch
  //: both land here: docs/internals/consumer.md#instantiation-matrix
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {

//: The residency and footprint queries answer at `D = 2`, the at-scale cell's
//: topology depth (`D` moves only the `[D][BC]` digit table), under the planner's
//: `global` normalization and the production precision.
constexpr int kQueryD = 2;
constexpr bool kQueryGlobal = true;
constexpr MmaPrec kQueryPrec = MmaPrec::kBf16;

const ShardRow& query_row(bool decay, int64_t BC, int64_t BT, int64_t DV) {
  const ShardRow* row =
      dispatch_row(DispatchChoice{static_cast<int>(BT), static_cast<int>(BC), static_cast<int>(DV),
                                  kQueryD, kQueryPrec, decay, kQueryGlobal});
  TORCH_CHECK(row != nullptr, "no consumer instantiation to query at BT=", BT, " BC=", BC,
              " d_v=", DV,
              ": the matrix carries BT in {16, 32}, BC in {16, 64} and "
              "d_v in {32, 64}");
  return *row;
}

}  // namespace

//: The consumer's residency answer, from the DRIVER. It takes four axes because
//: RESIDENCY IS PER-INSTANTIATION and a single reported number would over-promise on
//: one arm and under-report another: docs/internals/consumer.md#occupancy-axes
int64_t rola_consumer_occupancy(bool decay, int64_t BC, int64_t BT, int64_t DV) {
  const int blocks = query_row(decay, BC, BT, DV).occupancy();
  //: The shard's query is torch-free and reports through the CUDA runtime's error
  //: state; this is where it becomes a Python exception.
  C10_CUDA_CHECK(cudaGetLastError());
  return blocks;
}

//: THE TABLE, AS DATA. Every row this binary carries, in the generated order, so the
//: set the dispatch can reach is READABLE rather than inferred. It touches no device:
//: the fields are static data and the occupancy function pointer is deliberately NOT
//: called here, so the census can run before a CUDA context exists.
//: `tests/unit/test_dispatch_table.py` joins this against `cuobjdump`'s entry points,
//: which is what makes "the table and the binary carry the same arms" a checked
//: statement rather than a property of one generator.
//: docs/internals/consumer.md#table-census
std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t, std::string, bool, bool, int64_t,
                       std::string, std::string>>
rola_dispatch_table() {
  std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t, std::string, bool, bool, int64_t,
                         std::string, std::string>>
      out;
  for (const ShardRowBlock& block : kShardRowBlocks) {
    for (int i = 0; i < block.count; ++i) {
      const ShardRow& row = block.rows[i];
      out.emplace_back(row.BT, row.BC, row.DV, row.D,
                       row.prec == MmaPrec::kTf32x3 ? "tf32x3" : "bf16", row.decay, row.global_norm,
                       static_cast<int64_t>(row.smem_bytes), row.shard, row.src_key);
    }
  }
  return out;
}

//: The device fact the per-row enable predicate keys on, exposed so the same admission
//: the launch performs can be performed by a planner before it selects.
int64_t rola_max_dynamic_smem_bytes() {
  return static_cast<int64_t>(device_facts().max_dynamic_smem_bytes);
}

int64_t rola_consumer_smem_bytes(bool decay, int64_t BC, int64_t BT, int64_t DV) {
  return static_cast<int64_t>(query_row(decay, BC, BT, DV).smem_bytes);
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

//: The MMA precision's AMPLITUDE-CLASS storage dtype (spanned amplitude levels and
//: the value stream): bf16 under `bf16`, fp32 under `tf32x3`. A tensor bound
//: through here must ALREADY be in that dtype -- a mismatch is a hard error, never
//: a silent cast; the cast belongs to the schedule builder, below support
//: selection (I6-B §2.B4).
const void* amp_class_ptr(const at::Tensor& t, bool bf16_arm, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  const auto expected = bf16_arm ? at::kBFloat16 : at::kFloat;
  TORCH_CHECK(t.scalar_type() == expected, name, " must be ", expected,
              " (the dtype mma_precision implies for this stream), got ", t.scalar_type(),
              " -- the amplitude/value storage dtype is keyed to the MMA precision and a "
              "mismatch is never silently cast");
  return t.const_data_ptr();
}

}  // namespace

std::vector<at::Tensor> rola_consumer_forward(
    std::vector<at::Tensor> amp_read, std::vector<at::Tensor> amp_write,
    c10::optional<std::vector<at::Tensor>> amp_clock, c10::optional<at::Tensor> dials,
    at::Tensor read_bits, at::Tensor write_bits, at::Tensor read_tok_bits,
    at::Tensor write_tok_bits, at::Tensor read_union, at::Tensor write_union, at::Tensor owners,
    at::Tensor emit_works, at::Tensor emit_off, c10::optional<at::Tensor> g_read,
    at::Tensor g_write, at::Tensor v, at::Tensor state, c10::optional<at::Tensor> page_table,
    std::vector<int64_t> level_stride, std::vector<int64_t> level_modulus,
    std::vector<int64_t> level_row_offset, std::vector<int64_t> level_width,
    std::vector<int64_t> owner_stride, int64_t first_spanned_level, int64_t slice_digits,
    int64_t n_slices, int64_t owners_total, int64_t n_heads, int64_t BT, int64_t BC,
    bool global_norm, std::string mma_precision, double eps,
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
    TORCH_CHECK(BC % kMmaKQuantum == 0, "BC must be a whole number of MMA_K_QUANTUM=", kMmaKQuantum,
                "-leaf atoms under paging (spec §1b); got BC=", BC);
    TORCH_CHECK(state.dim() == 3 && state.size(1) == kMmaKQuantum && state.size(2) == cols,
                "paged state must be [physical_slots, MMA_K_QUANTUM(", kMmaKQuantum,
                "), d_v(+1 under global)] -- the page granule is the "
                "ATOM, not the owner rectangle (spec §4.5)");
    const auto& table = *page_table;
    const int64_t atoms_per_bh = owners_total * BC / kMmaKQuantum;
    TORCH_CHECK(table.is_cuda() && table.is_contiguous() && table.scalar_type() == at::kInt
                    && table.dim() == 2 && table.size(0) == BH && table.size(1) == atoms_per_bh,
                "page_table must be contiguous int32 [BH, atoms_per_bh = N/", kMmaKQuantum,
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

  // Numeric FORMAT axis (findings D4.3), stamped by the plan. Validated HERE --
  // before any dtype-keyed binding below reads it -- so a bad string fails as
  // itself rather than as a confusing amplitude-dtype mismatch.
  TORCH_CHECK(mma_precision == "tf32x3" || mma_precision == "bf16",
              "mma_precision must be 'tf32x3' or 'bf16', got ", mma_precision);
  const bool bf16_arm = mma_precision == "bf16";
  //: THE MEMORY-NARROWING SCOPE (P09 family): bf16 amplitude/value storage
  //: applies on the bf16 GLOBAL-NORM NON-DECAY arms. Both carves are structural
  //: template axes the dispatch already forks on, and both are MEASURED, not
  //: preferred (workflows/perf/p09_bf16_family.md): the DECAY arms' clock/panel
  //: registers sit at the allocator wall (the ratchet found in-loop spill on all
  //: 22 offenders, every one bf16+decay); the RAW arm has no `num/den` ratio to
  //: cancel the stacked roundings, and its worst tier-1 topology measured past
  //: `BF16_RTOL` on 2 of 6 seeds under narrowed storage (the global arm sits at
  //: ~60% of budget on the same cells).
  const bool narrow_mem = bf16_arm && global_norm && !dials.has_value();

  ConsumerParams p{};
  for (int64_t l = 0; l < D; ++l) {
    TORCH_CHECK(amp_read[l].dim() == 4, "amp_read[", l, "] must be [BH, n_tiles, BT, pitch]");
    // Per LEVEL ROLE (P09 family, Option B): SPANNED levels (`l >= j`) store the
    // precision-implied amplitude type, FIXED levels (`l < j`) stay fp32 -- they
    // feed the per-token row scales, whose error is uniform over the output row.
    const bool spanned = l >= first_spanned_level;
    p.amp_read[l] = amp_class_ptr(amp_read[l], narrow_mem && spanned, "amp_read");
    p.amp_write[l] = amp_class_ptr(amp_write[l], narrow_mem && spanned, "amp_write");
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
  p.v = amp_class_ptr(v, narrow_mem, "v");
  TORCH_CHECK(state.scalar_type() == at::kFloat && state.is_contiguous(),
              "state must be contiguous float32");
  p.state = state.data_ptr<float>();
  p.page_table = page_table.has_value() ? page_table->data_ptr<int32_t>() : nullptr;

  TORCH_CHECK(!(global_norm && g_read.has_value()),
              "normalization='global' fixes g_read := 1 (spec §1); pass g_read=None");
  TORCH_CHECK(global_norm || g_read.has_value(), "normalization='raw' requires g_read");

  // `num`/`den` are the readout accumulators and NEVER narrow (fp32-accumulator
  // ruling, same class as `state`): fp32 whatever `v`'s storage dtype is. Only
  // `y` -- the post-divide quotient -- takes the arm's output format, below.
  // Both are `empty`, not `zeros`: `emit_reduce_kernel` writes every element
  // (a token nothing contributes to gets the sum of zero terms, 0.0f).
  auto num = at::empty({BH, T, d_v}, v.options().dtype(at::kFloat));
  at::Tensor den;
  if (global_norm) den = at::empty({BH, T}, v.options().dtype(at::kFloat));

  // ---- the split-K emission workspace: one slot per (work item, token), a row
  // mask per (work item, tile), and the reduce's host-FIXED fold order --
  // `emit_works`/`emit_off`, BUILT UPSTREAM (once per schedule at `P = 1`, per
  // call from the split work list) so the per-step cost here is three
  // allocations and one memset. docs/internals/consumer.md#emit-workspace
  const int64_t n_work = owners.size(0);
  const int64_t n_tiles_l = amp_read[0].size(1);
  TORCH_CHECK(emit_works.is_cuda() && emit_works.is_contiguous()
                  && emit_works.scalar_type() == at::kInt && emit_works.dim() == 1
                  && emit_works.size(0) == n_work,
              "emit_works must be a contiguous int32 [n_work] CUDA tensor keyed to the "
              "SAME work list as `owners` -- a mismatched fold order is a wrong answer, "
              "not a slower one");
  TORCH_CHECK(emit_off.is_cuda() && emit_off.is_contiguous() && emit_off.scalar_type() == at::kInt
                  && emit_off.dim() == 1 && emit_off.size(0) == BH + 1,
              "emit_off must be a contiguous int32 [BH + 1] CUDA tensor");
  auto ws_num = at::empty({n_work, T, d_v}, v.options().dtype(at::kFloat));
  at::Tensor ws_den;
  if (global_norm) ws_den = at::empty({n_work, T}, v.options().dtype(at::kFloat));
  auto rowmask = at::zeros({n_work, n_tiles_l}, v.options().dtype(at::kInt));
  p.emit_num = ws_num.data_ptr<float>();
  p.emit_den = global_norm ? ws_den.data_ptr<float>() : nullptr;
  p.emit_rowmask = reinterpret_cast<uint32_t*>(rowmask.data_ptr<int32_t>());

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
    launch_consumer(pp, BT_i, BC_i, DV_i, global_norm, decay, mma_precision);
  };
  if (split) {
    auto i32 = [](const at::Tensor& t, const char* name, int64_t rows) {
      TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kInt, name,
                  " must be a contiguous int32 CUDA tensor");
      TORCH_CHECK(t.dim() >= 1 && t.size(0) == rows, name, " must have ", rows, " rows");
      return t.data_ptr<int32_t>();
    };
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

  // ---- the emission reduce: the ONE ordered fold of the split-K partials.
  // After pass 3 (the only emitting pass), before the divide.
  EmitReduceParams er{};
  er.ws_num = p.emit_num;
  er.ws_den = p.emit_den;
  er.rowmask = p.emit_rowmask;
  er.works = emit_works.data_ptr<int32_t>();
  er.off = emit_off.data_ptr<int32_t>();
  er.num = num.data_ptr<float>();
  er.den = global_norm ? den.data_ptr<float>() : nullptr;
  er.BH = p.BH;
  er.T = p.T;
  er.DV = static_cast<int>(d_v);
  er.BT = BT_i;
  er.n_tiles = p.n_tiles;
  emit_reduce_pass(er);

  // `y` takes the arm's output format under `global` (P09 family: the cast is the
  // divide epilogue's store -- the divide itself is fp32 over the fp32
  // accumulators). The `raw` arm returns `num` UNCAST: the ruling's surface is
  // the POST-DIVIDE quotient, and raw has no divide -- its `y` IS the
  // accumulator, and narrowing it was MEASURED to put the raw arm's stacked
  // family error at the BF16_RTOL line (2 of 6 seeds over on the worst tier-1
  // cell; the global arm's ratio cancels the common roundings and sits at ~60%
  // of budget). Evidence: workflows/perf/p09_bf16_family.md.
  at::Tensor y = num;
  if (global_norm) {
    y = at::empty({BH, T, d_v}, v.options());
    if (narrow_mem) {
      divide_pass(num.data_ptr<float>(), den.data_ptr<float>(),
                  reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()), BH * T,
                  static_cast<int>(d_v), static_cast<float>(eps));
    } else {
      divide_pass(num.data_ptr<float>(), den.data_ptr<float>(), y.data_ptr<float>(), BH * T,
                  static_cast<int>(d_v), static_cast<float>(eps));
    }
  }
  return {y, state, num, global_norm ? den : at::Tensor()};
}

}  // namespace rola
