#include "p3_page_plan_pybind.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/extension.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace fla_rola::p3 {
namespace {

constexpr int kPlanTensorCount = 19;
constexpr int kCacheTensorCount = 10;
constexpr int kPlanTouchedTags = 0;
constexpr int kPlanEarliest = 1;
constexpr int kPlanTouchedIds = 2;
constexpr int kPlanSortedIds = 3;
constexpr int kPlanTouchedCount = 4;
constexpr int kPlanNewFlags = 5;
constexpr int kPlanNewRanks = 6;
constexpr int kPlanNewLogicalIds = 7;
constexpr int kPlanNewEarliest = 8;
constexpr int kPlanNewCount = 9;
constexpr int kPlanRequiredSlots = 10;
constexpr int kPlanStatus = 11;
constexpr int kPlanGeneration = 12;
constexpr int kPlanEpoch = 13;
constexpr int kPlanLastEpoch = 14;
constexpr int kPlanSortTemp = 15;
//: §7.1 `CAPPED` admission-order workspace and its reported count.
constexpr int kPlanAdmitKeys = 16;
constexpr int kPlanAdmitSorted = 17;
constexpr int kPlanCappedPages = 18;

constexpr int kCachePageTable = 0;
constexpr int kCacheEarliest = 1;
constexpr int kCacheAllocatedIds = 2;
constexpr int kCacheAllocatedCount = 3;
constexpr int kCacheMappedCapacity = 4;
constexpr int kCacheNextCompact = 5;
constexpr int kCacheGeneration = 6;
constexpr int kCacheValueBase = 7;
constexpr int kCacheHitCount = 8;
constexpr int kCacheLogicalAllocationCount = 9;

int checked_int(int64_t value, const char* name) {
  TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(),
              name, " exceeds CUDA int range");
  return static_cast<int>(value);
}

int64_t checked_product(int64_t lhs, int64_t rhs, const char* name) {
  TORCH_CHECK(lhs >= 0 && rhs >= 0 &&
                  (lhs == 0 || rhs <= std::numeric_limits<int64_t>::max() / lhs),
              name, " overflows int64");
  return lhs * rhs;
}

void reject_cuda_graph_capture(c10::Device device, const char* operation) {
  auto stream = at::cuda::getCurrentCUDAStream(device.index());
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(), &status));
  TORCH_CHECK(status == cudaStreamCaptureStatusNone, operation,
              " is forbidden during CUDA-graph capture; planning, growth, "
              "publication, and reset must finish before capture");
}

void check_cuda(const torch::Tensor& tensor, c10::Device device,
                const char* name) {
  TORCH_CHECK(tensor.defined() && tensor.is_cuda() && tensor.device() == device,
              name, " must be a defined CUDA tensor on the route device");
}

void check_1d(const torch::Tensor& tensor, c10::ScalarType dtype,
              int64_t size, c10::Device device, const char* name) {
  check_cuda(tensor, device, name);
  TORCH_CHECK(tensor.scalar_type() == dtype && tensor.is_contiguous() &&
                  tensor.dim() == 1 && tensor.size(0) == size,
              name, " has the wrong contiguous dtype or shape");
}

torch::Tensor device_i32(const std::vector<int64_t>& values, c10::Device device,
                         const char* name) {
  auto cpu = torch::empty({static_cast<int64_t>(values.size())},
                          torch::TensorOptions().dtype(torch::kInt32));
  auto* dst = cpu.data_ptr<int>();
  for (size_t i = 0; i < values.size(); ++i) {
    TORCH_CHECK(values[i] >= -1 && values[i] <= std::numeric_limits<int>::max(),
                name, " contains a value outside the signed int32 metadata range");
    dst[i] = static_cast<int>(values[i]);
  }
  return cpu.to(device, torch::kInt32, false, true);
}

torch::Tensor device_i64(const std::vector<int64_t>& values, c10::Device device) {
  auto cpu = torch::empty({static_cast<int64_t>(values.size())},
                          torch::TensorOptions().dtype(torch::kInt64));
  std::copy(values.begin(), values.end(), cpu.data_ptr<int64_t>());
  return cpu.to(device, torch::kInt64, false, true);
}

torch::Tensor device_u8(const std::vector<int64_t>& values, c10::Device device) {
  auto cpu = torch::empty({static_cast<int64_t>(values.size())},
                          torch::TensorOptions().dtype(torch::kUInt8));
  auto* dst = cpu.data_ptr<uint8_t>();
  for (size_t i = 0; i < values.size(); ++i) {
    TORCH_CHECK(values[i] >= 0 && values[i] <= 1,
                "RouteDescriptor write support flags must be binary");
    dst[i] = static_cast<uint8_t>(values[i]);
  }
  return cpu.to(device, torch::kUInt8, false, true);
}

void check_tensor_list(const std::vector<torch::Tensor>& tensors, int count,
                       const char* name) {
  TORCH_CHECK(static_cast<int>(tensors.size()) == count,
              name, " must contain exactly ", count, " persistent tensors");
  for (int i = 0; i < count; ++i) {
    TORCH_CHECK(tensors[i].defined() && tensors[i].is_cuda() &&
                    tensors[i].is_contiguous(),
                name, "[", i, "] must be a defined contiguous CUDA tensor");
  }
}

void check_plan_tensors(const std::vector<torch::Tensor>& tensors,
                        c10::Device device, uint32_t touched_capacity,
                        uint32_t max_new_pages, uint64_t logical_pages,
                        size_t workspace_bytes) {
  check_tensor_list(tensors, kPlanTensorCount, "page plan tensors");
  const auto u64 = torch::kUInt64;
  const auto u32 = torch::kUInt32;
  check_1d(tensors[kPlanTouchedTags], u64, logical_pages, device, "plan.touched_tags");
  check_1d(tensors[kPlanEarliest], u64, logical_pages, device, "plan.earliest_epoch_tile");
  check_1d(tensors[kPlanTouchedIds], u64, touched_capacity, device, "plan.touched_ids");
  check_1d(tensors[kPlanSortedIds], u64, touched_capacity, device, "plan.sorted_ids");
  for (int index : {kPlanTouchedCount, kPlanNewCount, kPlanStatus}) {
    TORCH_CHECK(tensors[index].device() == device &&
                    tensors[index].scalar_type() == u32 && tensors[index].dim() == 0,
                "page plan scalar tensor has wrong dtype/shape at slot ", index);
  }
  check_1d(tensors[kPlanNewFlags], u32, touched_capacity, device, "plan.new_flags");
  check_1d(tensors[kPlanNewRanks], u32, touched_capacity, device, "plan.new_ranks");
  check_1d(tensors[kPlanNewLogicalIds], u64, max_new_pages, device, "plan.new_logical_ids");
  check_1d(tensors[kPlanNewEarliest], u64, max_new_pages, device, "plan.new_earliest_tiles");
  check_1d(tensors[kPlanAdmitKeys], u64, max_new_pages, device, "plan.admit_keys");
  check_1d(tensors[kPlanAdmitSorted], u64, max_new_pages, device, "plan.admit_sorted");
  TORCH_CHECK(tensors[kPlanCappedPages].device() == device &&
                  tensors[kPlanCappedPages].scalar_type() == u32 &&
                  tensors[kPlanCappedPages].dim() == 0,
              "plan.capped_pages must be a device uint32 scalar");
  for (int index : {kPlanGeneration, kPlanEpoch, kPlanLastEpoch, kPlanRequiredSlots}) {
    TORCH_CHECK(tensors[index].device() == device &&
                    tensors[index].scalar_type() == u64 && tensors[index].dim() == 0,
                "page plan scalar tensor has wrong dtype/shape at slot ", index);
  }
  TORCH_CHECK(tensors[kPlanSortTemp].device() == device &&
                  tensors[kPlanSortTemp].scalar_type() == torch::kUInt8 &&
                  tensors[kPlanSortTemp].numel() >= static_cast<int64_t>(workspace_bytes),
              "plan.sort_temp is smaller than the required CUB workspace");
}

void check_cache_metadata_tensors(
    const std::vector<torch::Tensor>& tensors, c10::Device device,
    int64_t logical_pages, int64_t physical_pages) {
  check_tensor_list(tensors, kCacheTensorCount, "page cache tensors");
  TORCH_CHECK(tensors[kCachePageTable].scalar_type() == torch::kInt32 &&
                  tensors[kCachePageTable].dim() == 2 &&
                  tensors[kCachePageTable].numel() == logical_pages,
              "cache.page_table must be contiguous int32 [BH,pages_per_bh]");
  check_1d(tensors[kCacheEarliest], torch::kUInt64, logical_pages, device,
           "cache.earliest_write_epoch_tile");
  check_1d(tensors[kCacheAllocatedIds], torch::kInt32, physical_pages, device,
           "cache.allocated_logical_page_ids");
  TORCH_CHECK(tensors[kCachePageTable].device() == device,
              "cache.page_table must be on the route device");
  for (int index : {kCacheAllocatedCount, kCacheMappedCapacity,
                    kCacheNextCompact, kCacheGeneration,
                    kCacheHitCount, kCacheLogicalAllocationCount}) {
    const auto dtype = index == kCacheAllocatedCount ? torch::kUInt32 : torch::kUInt64;
    TORCH_CHECK(tensors[index].device() == device &&
                    tensors[index].scalar_type() == dtype && tensors[index].dim() == 0,
                "cache scalar tensor has wrong dtype/shape at slot ", index);
  }
}

void check_plan_epoch_args(int64_t generation, int64_t epoch) {
  TORCH_CHECK(generation > 0 && epoch > 0 &&
                  generation <= std::numeric_limits<uint32_t>::max() &&
                  epoch <= std::numeric_limits<uint32_t>::max(),
              "page plan generation and epoch must be nonzero uint32 values");
}

PagePlanStorage make_plan(const std::vector<torch::Tensor>& t,
                          uint64_t logical_pages) {
  PagePlanStorage plan{
      t[kPlanTouchedTags].data_ptr<uint64_t>(), t[kPlanEarliest].data_ptr<uint64_t>(),
      t[kPlanTouchedIds].data_ptr<uint64_t>(), t[kPlanSortedIds].data_ptr<uint64_t>(),
      t[kPlanTouchedCount].data_ptr<uint32_t>(), t[kPlanNewFlags].data_ptr<uint32_t>(),
      t[kPlanNewRanks].data_ptr<uint32_t>(), t[kPlanNewLogicalIds].data_ptr<uint64_t>(),
      t[kPlanNewEarliest].data_ptr<uint64_t>(),
      t[kPlanAdmitKeys].data_ptr<uint64_t>(), t[kPlanAdmitSorted].data_ptr<uint64_t>(),
      t[kPlanCappedPages].data_ptr<uint32_t>(), t[kPlanNewCount].data_ptr<uint32_t>(),
      t[kPlanRequiredSlots].data_ptr<uint64_t>(), t[kPlanStatus].data_ptr<uint32_t>(),
      t[kPlanGeneration].data_ptr<uint64_t>(), t[kPlanEpoch].data_ptr<uint64_t>(),
      t[kPlanLastEpoch].data_ptr<uint64_t>(), t[kPlanSortTemp].data_ptr(),
      static_cast<size_t>(t[kPlanSortTemp].numel()),
      static_cast<uint32_t>(t[kPlanTouchedIds].numel()),
      static_cast<uint32_t>(t[kPlanNewLogicalIds].numel()), logical_pages};
  return plan;
}

PagePlanCacheView make_cache(const std::vector<torch::Tensor>& t,
                             const c10::optional<torch::Tensor>& mass_base,
                             uint64_t logical_pages, uint64_t max_physical_pages,
                             uint32_t page_states, uint32_t dqk, uint32_t dv,
                             bool identity) {
  PagePlanCacheView cache{
      t[kCachePageTable].data_ptr<int32_t>(),
      t[kCacheEarliest].data_ptr<uint64_t>(),
      t[kCacheAllocatedIds].data_ptr<int32_t>(),
      t[kCacheValueBase].data_ptr<float>(),
      mass_base.has_value() ? mass_base.value().data_ptr<float>() : nullptr,
      t[kCacheAllocatedCount].data_ptr<uint32_t>(),
      t[kCacheMappedCapacity].data_ptr<uint64_t>(),
      t[kCacheNextCompact].data_ptr<uint64_t>(),
      t[kCacheGeneration].data_ptr<uint64_t>(),
      t[kCacheHitCount].data_ptr<uint64_t>(),
      t[kCacheLogicalAllocationCount].data_ptr<uint64_t>(), logical_pages,
      max_physical_pages, page_states, dqk, dv, identity};
  return cache;
}

struct NativePagePlanContext {
  PagePlanStorage plan;
  PagePlanCacheView cache;
  PagePlanMarkArgs mark;
  PagePlanFinalizeArgs finalize;
  PagePlanPublishArgs publish;
  std::vector<torch::Tensor> metadata;
};

//: Build the mark/finalize/publish argument bundle from V3's §4 schedule.
//:
//: Re-key note (findings `paging.md` D1/D2). V2's version of this function took a
//: `RouteDescriptor` (compact bf16 factor arena + support bitset + mixed-radix
//: strides) and a `PrefixRunSchedule` (keyed run-record families). Neither exists
//: in V3. What replaces both is the §4 schedule's own write-amplitude stream --
//: one fp32 `[BH, n_tiles, BT, pitch_l]` tensor per routing level -- plus the §3
//: owner rectangle, which under §7.1's `page_states := BC` IS the page.
NativePagePlanContext make_context(
    std::vector<torch::Tensor> write_amps,
    std::vector<int64_t> level_widths,
    std::vector<int64_t> level_pitches,
    std::vector<int64_t> level_write_can_zero,
    std::vector<int64_t> owner_stride,
    int64_t token_count, int64_t token_tile,
    int64_t first_spanned_level, int64_t slice_digits, int64_t n_slices,
    std::vector<torch::Tensor> plan_tensors,
    std::vector<torch::Tensor> cache_tensors,
    c10::optional<torch::Tensor> mass_base,
    int64_t page_states64, int64_t pages_per_bh64,
    int64_t max_suffix_combinations64, int64_t generation64, int64_t epoch64,
    bool identity_physical_ids) {
  const int64_t levels = static_cast<int64_t>(write_amps.size());
  TORCH_CHECK(levels > 0 && levels <= kPagePlanMaxLevels &&
                  static_cast<int64_t>(level_widths.size()) == levels &&
                  static_cast<int64_t>(level_pitches.size()) == levels &&
                  static_cast<int64_t>(level_write_can_zero.size()) == levels &&
                  static_cast<int64_t>(owner_stride.size()) == levels,
              "page plan level metadata vectors must all have length D in [1, ",
              kPagePlanMaxLevels, "]");
  TORCH_CHECK(write_amps[0].defined() && write_amps[0].is_cuda(),
              "write_amps must be CUDA tensors");
  const auto device = write_amps[0].device();
  const int64_t batch_heads = write_amps[0].size(0);
  // Spec §7.1's `token_tile`, settled PARAMETRIC over the plan's `BT` (findings
  // `paging.md` D3). V2 hard-asserted 32 because its route arena was physically
  // packed in 32-token tiles; V3's stream is `BT`-shaped by construction, so the
  // constraint here is the schedule's own (`BT` is a compile-time constant in
  // {16, 32}, spec §6), not a layout accident.
  TORCH_CHECK(token_tile == 16 || token_tile == 32,
              "token_tile must be the plan's BT, a compile-time constant in {16, 32} (spec §6)");
  TORCH_CHECK(token_count > 0, "token_count must be positive");
  const int64_t n_tiles = (token_count + token_tile - 1) / token_tile;

  int64_t leaf_count = 1;
  std::vector<int64_t> amp_ptrs(levels);
  for (int64_t level = 0; level < levels; ++level) {
    const auto& amps = write_amps[level];
    check_cuda(amps, device, "write_amps");
    TORCH_CHECK(amps.scalar_type() == torch::kFloat32 && amps.is_contiguous() &&
                    amps.dim() == 4 && amps.size(0) == batch_heads &&
                    amps.size(1) == n_tiles && amps.size(2) == token_tile &&
                    amps.size(3) == level_pitches[level],
                "write_amps[", level,
                "] must be contiguous fp32 [BH, n_tiles, BT, pitch] matching the schedule");
    TORCH_CHECK(level_widths[level] >= 1 && level_widths[level] <= level_pitches[level],
                "level width must be positive and fit inside the staged pitch");
    TORCH_CHECK(level_write_can_zero[level] == 0 || level_write_can_zero[level] == 1,
                "level_write_can_zero must be 0/1");
    TORCH_CHECK(owner_stride[level] >= 1, "owner strides must be positive");
    leaf_count = checked_product(leaf_count, level_widths[level], "leaf count");
    amp_ptrs[level] = reinterpret_cast<int64_t>(amps.data_ptr<float>());
  }

  TORCH_CHECK(first_spanned_level >= 0 && first_spanned_level < levels &&
                  slice_digits > 0 && n_slices > 0 &&
                  slice_digits * n_slices == level_widths[first_spanned_level],
              "owner geometry (j, c, n_slices) is not the §3 rectangle for these widths");
  TORCH_CHECK(page_states64 > 0 && pages_per_bh64 > 0 &&
                  leaf_count == checked_product(pages_per_bh64, page_states64,
                                                "owner-partitioned leaf count"),
              "spec §7.1 requires page_states == BC and one page per owner: "
              "pages_per_bh * page_states must equal N exactly");
  const uint64_t logical_pages = checked_product(
      batch_heads, pages_per_bh64, "logical page count");
  TORCH_CHECK(logical_pages <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
              "logical page count exceeds the int32 page-table ABI");
  TORCH_CHECK(generation64 > 0 && epoch64 > 0 &&
                  generation64 <= std::numeric_limits<uint32_t>::max() &&
                  epoch64 <= std::numeric_limits<uint32_t>::max(),
              "page plan generation and epoch must be nonzero uint32 values");
  TORCH_CHECK(max_suffix_combinations64 > 0 &&
                  max_suffix_combinations64 <= std::numeric_limits<uint32_t>::max(),
              "max_suffix_combinations must fit uint32");

  check_tensor_list(cache_tensors, kCacheTensorCount, "page cache tensors");
  auto value_base = cache_tensors[kCacheValueBase];
  check_cuda(value_base, device, "cache.value_base");
  // §7.1: `dqk` disappears as an ABI degeneracy -- it is set to 1, not removed, so
  // `paged_value_offset` stays the SAME expression it was and collapses to
  // `(physical * BC + row) * cols + v`. `cols` is `d_v` plus the `global`
  // normalization's mass column, which V3 carries INSIDE the state row rather
  // than in V2's separate mass plane.
  TORCH_CHECK(value_base.scalar_type() == torch::kFloat32 && value_base.is_contiguous() &&
                  value_base.dim() == 4 && value_base.size(0) > 0 &&
                  value_base.size(0) <= std::numeric_limits<int32_t>::max() &&
                  value_base.size(1) == page_states64 && value_base.size(2) == 1 &&
                  value_base.size(3) > 0,
              "cache.value_base must be contiguous fp32 [physical_pages, BC, 1, cols]");
  const uint32_t page_states = static_cast<uint32_t>(page_states64);
  const uint32_t dqk = 1;
  const uint32_t dv = static_cast<uint32_t>(value_base.size(3));
  TORCH_CHECK(!mass_base.has_value(),
              "V3 carries the `global` mass as the last state column, not a separate "
              "mass plane; pass mass_base=None");
  check_cuda(cache_tensors[kCachePageTable], device, "cache.page_table");
  TORCH_CHECK(cache_tensors[kCachePageTable].scalar_type() == torch::kInt32 &&
                  cache_tensors[kCachePageTable].is_contiguous() &&
                  cache_tensors[kCachePageTable].dim() == 2 &&
                  cache_tensors[kCachePageTable].size(0) == batch_heads &&
                  cache_tensors[kCachePageTable].size(1) == pages_per_bh64,
              "cache.page_table must be contiguous int32 [BH, pages_per_bh]");
  check_1d(cache_tensors[kCacheEarliest], torch::kUInt64,
           logical_pages, device, "cache.earliest_write_epoch_tile");
  check_1d(cache_tensors[kCacheAllocatedIds], torch::kInt32,
           value_base.size(0), device, "cache.allocated_logical_page_ids");
  for (int index : {kCacheAllocatedCount, kCacheMappedCapacity,
                    kCacheNextCompact, kCacheGeneration}) {
    const auto dtype = index == kCacheAllocatedCount ? torch::kUInt32 : torch::kUInt64;
    TORCH_CHECK(cache_tensors[index].scalar_type() == dtype &&
                    cache_tensors[index].dim() == 0,
                "cache scalar tensor has wrong dtype/shape at slot ", index);
  }
  const uint32_t touched_capacity = static_cast<uint32_t>(
      plan_tensors.size() > kPlanTouchedIds ? plan_tensors[kPlanTouchedIds].numel() : 0);
  const uint32_t max_new_pages = static_cast<uint32_t>(
      plan_tensors.size() > kPlanNewLogicalIds ? plan_tensors[kPlanNewLogicalIds].numel() : 0);
  TORCH_CHECK(touched_capacity > 0 && max_new_pages > 0 &&
                  max_new_pages <= touched_capacity,
              "page plan tensor capacities are invalid");
  const size_t workspace_bytes = page_plan_workspace_bytes(touched_capacity);
  check_plan_tensors(plan_tensors, device, touched_capacity, max_new_pages,
                     logical_pages, workspace_bytes);
  TORCH_CHECK(plan_tensors[kPlanEarliest].data_ptr() !=
                  cache_tensors[kCacheEarliest].data_ptr(),
              "page-plan earliest scratch must be distinct from persistent cache earliest metadata");

  auto d_widths = device_i32(level_widths, device, "level widths");
  auto d_pitches = device_i32(level_pitches, device, "level pitches");
  auto d_owner_stride = device_i32(owner_stride, device, "owner strides");
  auto d_write_can_zero = device_u8(level_write_can_zero, device);
  auto d_amp_ptrs = device_i64(amp_ptrs, device);

  PagePlanStorage plan = make_plan(plan_tensors, logical_pages);
  PagePlanCacheView cache = make_cache(
      cache_tensors, c10::nullopt, logical_pages, value_base.size(0), page_states,
      dqk, dv, identity_physical_ids);
  PagePlanV3StreamView stream{
      reinterpret_cast<const float* const*>(d_amp_ptrs.data_ptr<int64_t>()),
      d_widths.data_ptr<int>(), d_pitches.data_ptr<int>(),
      d_write_can_zero.data_ptr<uint8_t>(),
      checked_int(levels, "level count"),
      checked_int(batch_heads, "batch heads"),
      checked_int(token_count, "token count"),
      checked_int(n_tiles, "tile count"),
      checked_int(token_tile, "token tile")};
  PagePlanOwnerGeometry owners{
      d_owner_stride.data_ptr<int>(),
      checked_int(first_spanned_level, "first spanned level"),
      checked_int(slice_digits, "slice digits"),
      checked_int(n_slices, "slice count"),
      checked_int(pages_per_bh64, "owners total")};
  PagePlanMarkArgs mark{plan, stream, owners, static_cast<uint64_t>(generation64),
                        static_cast<uint64_t>(epoch64), page_states,
                        static_cast<uint32_t>(pages_per_bh64),
                        static_cast<uint32_t>(max_suffix_combinations64)};
  PagePlanFinalizeArgs finalize{plan, cache,
                               static_cast<uint64_t>(generation64),
                               static_cast<uint64_t>(epoch64)};
  PagePlanPublishArgs publish{plan, cache, static_cast<uint64_t>(generation64),
                              static_cast<uint64_t>(epoch64)};
  NativePagePlanContext result{plan, cache, mark, finalize, publish,
                               {d_widths, d_pitches, d_owner_stride,
                                d_write_can_zero, d_amp_ptrs}};
  return result;
}

void record_tensor(
    const torch::Tensor& tensor, const c10::cuda::CUDAStream& stream) {
  if (tensor.defined() && tensor.numel() != 0) {
    c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
  }
}

std::vector<torch::Tensor> result_status(const std::vector<torch::Tensor>& plan_tensors) {
  // `capped_pages` rides the status tuple because §7.1 makes capping a REPORTED
  // outcome, never a silent one: every caller that reads the status reads the
  // denied-page count in the same hop, so there is no path that observes success
  // without observing the degradation.
  return {plan_tensors[kPlanStatus], plan_tensors[kPlanRequiredSlots],
          plan_tensors[kPlanNewCount], plan_tensors[kPlanCappedPages]};
}

}  // namespace

__global__ void paged_state_readiness_kernel(
    const int32_t* page_table, const uint64_t* earliest_epoch_tile,
    const uint64_t* generation_tensor, int64_t entries,
    int64_t physical_pages, uint64_t generation, uint32_t plan_epoch) {
  if (*generation_tensor != generation) {
    asm volatile("trap;");
  }
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < entries;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int32_t physical = page_table[index];
    if (physical < kAbsentPage || physical >= physical_pages) {
      asm volatile("trap;");
    }
    const uint64_t packed = earliest_epoch_tile[index];
    const uint32_t epoch = static_cast<uint32_t>(packed >> 32);
    if (physical != kAbsentPage && (epoch == 0 || epoch > plan_epoch)) {
      asm volatile("trap;");
    }
  }
}

void launch_paged_state_readiness_assert(
    torch::Tensor page_table, torch::Tensor earliest_epoch_tile,
    torch::Tensor generation_tensor,
    int64_t physical_pages, int64_t generation, int64_t plan_epoch,
    cudaStream_t stream) {
  TORCH_CHECK(page_table.is_cuda() && earliest_epoch_tile.is_cuda() &&
                  generation_tensor.is_cuda() &&
                  page_table.device() == earliest_epoch_tile.device() &&
                  page_table.device() == generation_tensor.device() &&
                  page_table.scalar_type() == torch::kInt32 &&
                  earliest_epoch_tile.scalar_type() == torch::kUInt64 &&
                  generation_tensor.scalar_type() == torch::kUInt64 &&
              page_table.is_contiguous() && earliest_epoch_tile.is_contiguous() &&
                  generation_tensor.is_contiguous() && generation_tensor.dim() == 0 &&
                  page_table.numel() == earliest_epoch_tile.numel(),
              "paged state readiness metadata has the wrong device/layout");
  TORCH_CHECK(physical_pages > 0 && generation > 0 &&
                  generation <= std::numeric_limits<uint32_t>::max() &&
                  plan_epoch > 0 && plan_epoch <= std::numeric_limits<uint32_t>::max(),
              "paged state readiness metadata is invalid");
  const int64_t entries = page_table.numel();
  const int blocks = static_cast<int>(std::min<int64_t>(
      4096, std::max<int64_t>(1, (entries + 255) / 256)));
  paged_state_readiness_kernel<<<blocks, 256, 0, stream>>>(
      page_table.data_ptr<int32_t>(), earliest_epoch_tile.data_ptr<uint64_t>(),
      generation_tensor.data_ptr<uint64_t>(), entries, physical_pages,
      static_cast<uint64_t>(generation), static_cast<uint32_t>(plan_epoch));
}

std::vector<torch::Tensor> rola_p3_page_plan_prepare_cuda(
    std::vector<torch::Tensor> write_amps, std::vector<int64_t> level_widths,
    std::vector<int64_t> level_pitches, std::vector<int64_t> level_write_can_zero,
    std::vector<int64_t> owner_stride, int64_t token_count, int64_t token_tile,
    int64_t first_spanned_level, int64_t slice_digits, int64_t n_slices,
    std::vector<torch::Tensor> plan_tensors, std::vector<torch::Tensor> cache_tensors,
    c10::optional<torch::Tensor> mass_base,
    int64_t page_states, int64_t pages_per_bh, int64_t max_suffix_combinations,
    int64_t generation, int64_t epoch, bool identity_physical_ids) {
  TORCH_CHECK(!write_amps.empty() && write_amps[0].defined(),
              "write_amps must contain one staged amplitude tensor per routing level");
  const auto device = write_amps[0].device();
  c10::cuda::CUDAGuard guard(device);
  reject_cuda_graph_capture(device, "sparse page-plan prepare");
  auto context = make_context(
      write_amps, std::move(level_widths), std::move(level_pitches),
      std::move(level_write_can_zero), std::move(owner_stride), token_count,
      token_tile, first_spanned_level, slice_digits, n_slices, plan_tensors,
      cache_tensors, mass_base, page_states, pages_per_bh,
      max_suffix_combinations, generation, epoch, identity_physical_ids);
  auto stream = at::cuda::getCurrentCUDAStream(write_amps[0].get_device());
  launch_page_plan_reset(context.plan, generation, epoch, stream.stream());
  launch_page_plan_mark(context.mark, stream.stream());
  launch_page_plan_finalize(context.finalize, stream.stream());
  for (const auto& tensor : context.metadata) record_tensor(tensor, stream);
  for (const auto& tensor : write_amps) record_tensor(tensor, stream);
  for (const auto& tensor : plan_tensors) record_tensor(tensor, stream);
  for (int index = 0; index < kCacheTensorCount; ++index) {
    if (index != kCacheValueBase) record_tensor(cache_tensors[index], stream);
  }
  return result_status(plan_tensors);
}

//: Emit spec §7.1's admit mask from the published page table. Separate from
//: `publish` because the consumer needs it as its own tensor and because it is
//: idempotent -- re-deriving it never mutates plan or cache state.
void rola_p3_page_admit_mask_cuda(
    torch::Tensor page_table, torch::Tensor admit) {
  TORCH_CHECK(page_table.defined() && page_table.is_cuda() &&
                  page_table.is_contiguous() && page_table.dim() == 2 &&
                  page_table.scalar_type() == torch::kInt32,
              "page_table must be contiguous int32 [BH, pages_per_bh]");
  const int64_t batch_heads = page_table.size(0);
  const int64_t pages_per_bh = page_table.size(1);
  const int64_t words_per_bh = (pages_per_bh + 31) / 32;
  TORCH_CHECK(admit.defined() && admit.is_cuda() && admit.is_contiguous() &&
                  admit.device() == page_table.device() &&
                  admit.scalar_type() == torch::kUInt32 && admit.dim() == 2 &&
                  admit.size(0) == batch_heads && admit.size(1) == words_per_bh,
              "admit must be contiguous uint32 [BH, ceil(pages_per_bh/32)] on the page-table device");
  c10::cuda::CUDAGuard guard(page_table.device());
  auto stream = at::cuda::getCurrentCUDAStream(page_table.get_device());
  launch_page_admit_mask(
      page_table.data_ptr<int32_t>(),
      reinterpret_cast<uint32_t*>(admit.data_ptr()),
      static_cast<uint64_t>(page_table.numel()),
      static_cast<uint32_t>(pages_per_bh),
      static_cast<uint32_t>(batch_heads), stream.stream());
  record_tensor(page_table, stream);
  record_tensor(admit, stream);
}

std::vector<torch::Tensor> rola_p3_page_plan_finalize_cuda(
    std::vector<torch::Tensor> plan_tensors, std::vector<torch::Tensor> cache_tensors,
    c10::optional<torch::Tensor> mass_base,
    int64_t generation, int64_t epoch, int64_t page_states, int64_t dqk, int64_t dv,
    bool identity_physical_ids) {
  TORCH_CHECK(plan_tensors.size() == kPlanTensorCount &&
                  cache_tensors.size() == kCacheTensorCount,
              "finalize requires the persistent plan/cache tensor lists");
  check_tensor_list(plan_tensors, kPlanTensorCount, "page plan tensors");
  check_tensor_list(cache_tensors, kCacheTensorCount, "page cache tensors");
  auto value_base = cache_tensors[kCacheValueBase];
  check_cuda(value_base, value_base.device(), "cache.value_base");
  const uint64_t logical_pages = static_cast<uint64_t>(cache_tensors[kCachePageTable].numel());
  TORCH_CHECK(logical_pages <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
              "persistent page cache logical page count exceeds int64");
  check_cache_metadata_tensors(cache_tensors, value_base.device(),
                               static_cast<int64_t>(logical_pages), value_base.size(0));
  TORCH_CHECK(cache_tensors[kCachePageTable].dim() == 2 &&
                  cache_tensors[kCacheEarliest].dim() == 1 &&
                  cache_tensors[kCacheEarliest].numel() == logical_pages &&
                  cache_tensors[kCacheAllocatedIds].dim() == 1 &&
                  cache_tensors[kCacheAllocatedIds].numel() == value_base.size(0),
              "persistent page cache metadata has the wrong shape");
  const uint32_t touched_capacity = static_cast<uint32_t>(plan_tensors[kPlanTouchedIds].numel());
  const uint32_t max_new_pages = static_cast<uint32_t>(plan_tensors[kPlanNewLogicalIds].numel());
  check_plan_epoch_args(generation, epoch);
  check_plan_tensors(plan_tensors, value_base.device(), touched_capacity, max_new_pages,
                     logical_pages, page_plan_workspace_bytes(touched_capacity));
  TORCH_CHECK(plan_tensors[kPlanEarliest].data_ptr() !=
                  cache_tensors[kCacheEarliest].data_ptr(),
              "page-plan earliest scratch must be distinct from persistent cache earliest metadata");
  TORCH_CHECK(value_base.scalar_type() == torch::kFloat32 && value_base.dim() == 4 &&
                  value_base.size(1) == page_states && value_base.size(2) == dqk &&
                  value_base.size(3) == dv,
              "value_base geometry does not match finalize metadata");
  TORCH_CHECK(!mass_base.has_value(),
              "V3 carries the `global` mass as the last state column (spec §7.1 dqk := 1); "
              "pass mass_base=None");
  auto cache = make_cache(cache_tensors, mass_base, logical_pages, value_base.size(0),
                          checked_int(page_states, "page_states"), checked_int(dqk, "dqk"),
                          checked_int(dv, "dv"), identity_physical_ids);
  auto plan = make_plan(plan_tensors, logical_pages);
  c10::cuda::CUDAGuard guard(value_base.device());
  reject_cuda_graph_capture(value_base.device(), "sparse page-plan finalize");
  auto stream = at::cuda::getCurrentCUDAStream(value_base.get_device());
  PagePlanFinalizeArgs args{
      plan, cache, static_cast<uint64_t>(generation), static_cast<uint64_t>(epoch)};
  launch_page_plan_finalize(args, stream.stream());
  for (const auto& tensor : plan_tensors) record_tensor(tensor, stream);
  for (int index = 0; index < kCacheTensorCount; ++index) {
    if (index != kCacheValueBase) record_tensor(cache_tensors[index], stream);
  }
  return result_status(plan_tensors);
}

std::vector<torch::Tensor> rola_p3_page_plan_publish_cuda(
    std::vector<torch::Tensor> plan_tensors, std::vector<torch::Tensor> cache_tensors,
    c10::optional<torch::Tensor> mass_base,
    int64_t generation, int64_t epoch, int64_t page_states, int64_t dqk, int64_t dv,
    bool identity_physical_ids) {
  TORCH_CHECK(plan_tensors.size() == kPlanTensorCount &&
                  cache_tensors.size() == kCacheTensorCount,
              "publish requires the persistent plan/cache tensor lists");
  check_tensor_list(plan_tensors, kPlanTensorCount, "page plan tensors");
  check_tensor_list(cache_tensors, kCacheTensorCount, "page cache tensors");
  auto value_base = cache_tensors[kCacheValueBase];
  check_cuda(value_base, value_base.device(), "cache.value_base");
  const uint64_t logical_pages = static_cast<uint64_t>(cache_tensors[kCachePageTable].numel());
  TORCH_CHECK(logical_pages <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
              "persistent page cache logical page count exceeds int64");
  check_cache_metadata_tensors(cache_tensors, value_base.device(),
                               static_cast<int64_t>(logical_pages), value_base.size(0));
  TORCH_CHECK(cache_tensors[kCachePageTable].dim() == 2 &&
                  cache_tensors[kCacheEarliest].dim() == 1 &&
                  cache_tensors[kCacheEarliest].numel() == logical_pages &&
                  cache_tensors[kCacheAllocatedIds].dim() == 1 &&
                  cache_tensors[kCacheAllocatedIds].numel() == value_base.size(0),
              "persistent page cache metadata has the wrong shape");
  const uint32_t touched_capacity = static_cast<uint32_t>(plan_tensors[kPlanTouchedIds].numel());
  const uint32_t max_new_pages = static_cast<uint32_t>(plan_tensors[kPlanNewLogicalIds].numel());
  check_plan_epoch_args(generation, epoch);
  check_plan_tensors(plan_tensors, value_base.device(), touched_capacity, max_new_pages,
                     logical_pages, page_plan_workspace_bytes(touched_capacity));
  TORCH_CHECK(plan_tensors[kPlanEarliest].data_ptr() !=
                  cache_tensors[kCacheEarliest].data_ptr(),
              "page-plan earliest scratch must be distinct from persistent cache earliest metadata");
  TORCH_CHECK(value_base.scalar_type() == torch::kFloat32 && value_base.is_contiguous() &&
                  value_base.dim() == 4 && value_base.size(1) == page_states &&
                  value_base.size(2) == dqk && value_base.size(3) == dv,
              "value_base geometry does not match publish metadata");
  TORCH_CHECK(!mass_base.has_value(),
              "V3 carries the `global` mass as the last state column (spec §7.1 dqk := 1); "
              "pass mass_base=None");
  auto cache = make_cache(cache_tensors, mass_base, logical_pages, value_base.size(0),
                          checked_int(page_states, "page_states"), checked_int(dqk, "dqk"),
                          checked_int(dv, "dv"), identity_physical_ids);
  auto plan = make_plan(plan_tensors, logical_pages);
  PagePlanPublishArgs args{plan, cache, static_cast<uint64_t>(generation),
                           static_cast<uint64_t>(epoch)};
  c10::cuda::CUDAGuard guard(value_base.device());
  reject_cuda_graph_capture(value_base.device(), "sparse page-plan publish");
  auto stream = at::cuda::getCurrentCUDAStream(value_base.get_device());
  launch_page_plan_publish(args, stream.stream());
  for (const auto& tensor : plan_tensors) record_tensor(tensor, stream);
  for (int index = 0; index < kCacheTensorCount; ++index) {
    if (index != kCacheValueBase) record_tensor(cache_tensors[index], stream);
  }
  return result_status(plan_tensors);
}

void rola_p3_page_cache_reset_cuda(
    std::vector<torch::Tensor> cache_tensors) {
  check_tensor_list(cache_tensors, kCacheTensorCount, "page cache tensors");
  auto page_table = cache_tensors[kCachePageTable];
  auto earliest = cache_tensors[kCacheEarliest];
  auto allocated_ids = cache_tensors[kCacheAllocatedIds];
  auto allocated_count = cache_tensors[kCacheAllocatedCount];
  auto next_compact = cache_tensors[kCacheNextCompact];
  auto hit_count = cache_tensors[kCacheHitCount];
  auto logical_allocation_count = cache_tensors[kCacheLogicalAllocationCount];
  const auto device = page_table.device();
  check_cuda(page_table, device, "cache.page_table");
  check_1d(earliest, torch::kUInt64, page_table.numel(), device,
           "cache.earliest_write_epoch_tile");
  check_1d(allocated_ids, torch::kInt32,
           cache_tensors[kCacheValueBase].size(0), device,
           "cache.allocated_logical_page_ids");
  TORCH_CHECK(page_table.scalar_type() == torch::kInt32 &&
                  page_table.is_contiguous() && page_table.dim() == 2,
              "cache.page_table must be contiguous int32 [BH,pages_per_bh]");
  TORCH_CHECK(allocated_count.scalar_type() == torch::kUInt32 &&
                  allocated_count.dim() == 0 &&
                  next_compact.scalar_type() == torch::kUInt64 &&
                  next_compact.dim() == 0 &&
                  hit_count.scalar_type() == torch::kUInt64 && hit_count.dim() == 0 &&
                  logical_allocation_count.scalar_type() == torch::kUInt64 &&
                  logical_allocation_count.dim() == 0,
              "page cache reset scalar metadata has the wrong dtype/shape");
  c10::cuda::CUDAGuard guard(device);
  reject_cuda_graph_capture(device, "sparse page-cache reset");
  auto stream = at::cuda::getCurrentCUDAStream(page_table.get_device());
  launch_page_cache_reset(
      page_table.data_ptr<int32_t>(), earliest.data_ptr<uint64_t>(),
      allocated_ids.data_ptr<int32_t>(), allocated_count.data_ptr<uint32_t>(),
      next_compact.data_ptr<uint64_t>(), hit_count.data_ptr<uint64_t>(),
      logical_allocation_count.data_ptr<uint64_t>(), page_table.numel(),
      allocated_ids.numel(), stream.stream());
  for (int index = 0; index < kCacheTensorCount; ++index) {
    if (index != kCacheValueBase) record_tensor(cache_tensors[index], stream);
  }
}

}  // namespace fla_rola::p3
