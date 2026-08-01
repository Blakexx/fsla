// Spec §7.1 page commitment, re-keyed from V2 onto V3's value panels.
//
// UNCHANGED FROM V2, deliberately (§7.1 "planner reuse inventory"): the
// three-phase host API (reset -> mark -> finalize -> publish), the CUB
// sort-dedupe-scan over the touched set, generation/epoch tagging, the growth
// transactionality, the readiness assert, and the CUDA-graph rule.
//
// CHANGED: `page_states := BC` (one page is one owner), `dqk := 1`, the level set
// is V3's routing levels, and the new `kPagePlanCapped` status with its admit
// mask. The mark phase is re-driven over V3's §4 amplitude stream because V2's
// PrefixRunSchedule families have no V3 counterpart (findings `paging.md` D1);
// the PREDICATE it evaluates -- "a zero complete write factor may neither deposit
// nor allocate" -- is carried over unchanged, which is what §7.1 requires.
#include "page_plan.cuh"

#include <cub/cub.cuh>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace rola::paging {
namespace {

constexpr int kThreads = 128;

__device__ __forceinline__ std::uint64_t pack_epoch_tile(
    std::uint64_t epoch, std::uint64_t tile) {
  return ((epoch & 0xffffffffull) << 32) | (tile & 0xffffffffull);
}

__device__ __forceinline__ void mark_page(
    const PagePlanMarkArgs& args,
    std::uint64_t logical_page,
    std::uint32_t tile) {
  if (logical_page >= args.plan.logical_page_count) return;
  const std::uint64_t tag =
      ((args.generation & 0xffffffffull) << 32) |
      (args.epoch & 0xffffffffull);
  auto* tag_address = reinterpret_cast<unsigned long long*>(
      args.plan.touched_tags + logical_page);
  const unsigned long long previous = atomicExch(
      tag_address, static_cast<unsigned long long>(tag));
  if (previous != tag) {
    const std::uint32_t slot = atomicAdd(args.plan.touched_count, 1u);
    if (slot < args.plan.touched_capacity) {
      args.plan.touched_ids[slot] = logical_page;
    } else {
      atomicExch(args.plan.status, kPagePlanOverflow);
    }
  }

  auto* earliest = reinterpret_cast<unsigned long long*>(
      args.plan.earliest_epoch_tile + logical_page);
  unsigned long long old = *earliest;
  const unsigned long long candidate = static_cast<unsigned long long>(
      pack_epoch_tile(args.epoch, tile));
  while (true) {
    const std::uint64_t old_epoch = old >> 32;
    const std::uint32_t old_tile = static_cast<std::uint32_t>(old);
    if (old_epoch == args.epoch && old_tile <= tile) return;
    const unsigned long long prior = atomicCAS(earliest, old, candidate);
    if (prior == old) return;
    old = prior;
  }
}

//: One entry of the §4 write-amplitude stream. Token-major:
//: `amps[l][((bh * n_tiles + tile) * BT + lane) * pitch_l + digit]`.
//: Liveness is EXACT nonzero-ness -- never a threshold (spec §1.2): entmax's exact
//: support zeros are the scheduling authority, and a tolerance here would either
//: invent a page or drop a deposit.
__device__ __forceinline__ float write_amp(
    const PagePlanStreamView& stream, int bh, int token, int level, int digit) {
  const int tile = token / stream.token_tile;
  const int lane = token % stream.token_tile;
  const int pitch = stream.level_pitch[level];
  const std::int64_t offset =
      ((static_cast<std::int64_t>(bh) * stream.n_tiles + tile) * stream.token_tile + lane) *
          pitch + digit;
  return stream.level_amps[level][offset];
}

//: §7.1's first-write trigger, third clause. A leaf needs EVERY level's digit
//: live, so a token with a wholly dead level writes nothing at all -- for any
//: owner. Hoisted out of the owner walk because it does not depend on the owner.
//: Evaluated over the levels BELOW the boundary (`l > j`), which the owner spans
//: in full; the boundary level's own liveness is decided per slice inside the walk.
__device__ __forceinline__ bool spanned_levels_live(
    const PagePlanStreamView& stream, int bh, int token, int j) {
  for (int level = j + 1; level < stream.level_count; ++level) {
    bool any = false;
    const int width = stream.level_width[level];
    for (int digit = 0; digit < width; ++digit) {
      if (write_amp(stream, bh, token, level, digit) != 0.0f) { any = true; break; }
    }
    if (!any) return false;
  }
  return true;
}

//: The owner-space counterpart of V2's `enumerate_suffix`. V2 walked the full
//: mixed radix over `N` and divided each emitted leaf by `page_states`; under
//: §7.1's `page_states := BC` that division is knowable BEFORE the walk, so the
//: walk collapses onto the `N / BC` owner rectangles. Exact, not a superset
//: (findings `paging.md` D2): the live leaf set is a Cartesian product, an owner
//: is a Cartesian box, and two such sets intersect iff they do so factorwise.
//:
//: Levels `0 .. j-1` contribute the owner's FIXED digits; level `j` contributes
//: its slice. Digit pruning is V2's, unchanged: a zero write factor terminates
//: the branch, and a level that cannot zero skips the test entirely.
__device__ __forceinline__ void mark_owners_for_token(
    const PagePlanMarkArgs& args, int bh, int token, std::uint32_t tile) {
  const PagePlanStreamView& stream = args.stream;
  const PagePlanOwnerGeometry& geom = args.owners;
  const int j = geom.first_spanned_level;
  if (!spanned_levels_live(stream, bh, token, j)) return;

  int next_digit[kPagePlanMaxLevels];
  int rest_acc[kPagePlanMaxLevels + 1];
  for (int level = 0; level < j; ++level) next_digit[level] = 0;
  rest_acc[0] = 0;
  int level = 0;
  std::uint32_t combinations = 0;

  while (level >= 0) {
    if (level == j) {
      // Boundary level: one owner per slice holding at least one live digit.
      const int width = stream.level_width[j];
      const int c = geom.slice_digits;
      const std::int64_t owner_base =
          static_cast<std::int64_t>(rest_acc[j]) * geom.n_slices;
      for (int slice = 0; slice < geom.n_slices; ++slice) {
        bool live = false;
        for (int d = slice * c; d < (slice + 1) * c && d < width; ++d) {
          if (write_amp(stream, bh, token, j, d) != 0.0f) { live = true; break; }
        }
        if (!live) continue;
        if (combinations >= args.max_suffix_combinations) {
          atomicExch(args.plan.status, kPagePlanOverflow);
          return;
        }
        ++combinations;
        const std::int64_t owner = owner_base + slice;
        if (owner < 0 || owner >= geom.owners_total) {
          atomicExch(args.plan.status, kPagePlanOverflow);
          return;
        }
        mark_page(args, static_cast<std::uint64_t>(
            static_cast<std::int64_t>(bh) * args.pages_per_bh + owner), tile);
      }
      --level;
      continue;
    }

    const int width = stream.level_width[level];
    int digit = -1;
    while (next_digit[level] < width) {
      const int candidate = next_digit[level]++;
      if (stream.level_write_can_zero[level] != 0 &&
          write_amp(stream, bh, token, level, candidate) == 0.0f) {
        continue;
      }
      digit = candidate;
      break;
    }
    if (digit < 0) {
      next_digit[level] = 0;
      --level;
      continue;
    }
    rest_acc[level + 1] = rest_acc[level] + digit * geom.owner_stride[level];
    ++level;
    if (level < j) next_digit[level] = 0;
  }
}

__global__ void reset_clear_kernel(PagePlanStorage plan) {
  const std::uint32_t old_count = min(*plan.touched_count, plan.touched_capacity);
  for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < old_count;
       index += gridDim.x * blockDim.x) {
    const std::uint64_t logical = plan.touched_ids[index];
    if (logical < plan.logical_page_count) {
      plan.touched_tags[logical] = 0;
      plan.earliest_epoch_tile[logical] = 0;
    }
    plan.touched_ids[index] = UINT64_MAX;
  }
}

__global__ void reset_state_kernel(
    PagePlanStorage plan, std::uint64_t generation, std::uint64_t epoch) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (*plan.plan_generation != generation) *plan.last_epoch = 0;
  if (epoch == 0 || epoch <= *plan.last_epoch || (epoch >> 32) != 0) {
    *plan.status = kPagePlanOverflow;
    return;
  }
  *plan.last_epoch = epoch;
  *plan.plan_generation = generation;
  *plan.plan_epoch = epoch;
  *plan.touched_count = 0;
  *plan.new_count = 0;
  *plan.required_slots = 0;
  *plan.status = kPagePlanReady;
}

__global__ void mark_kernel(PagePlanMarkArgs args) {
  const PagePlanStreamView& stream = args.stream;
  // §7.1's dense-write short-circuit, carried over verbatim in intent: when no
  // write level can zero (the softmax-write-level case), every owner is written by
  // the first token, and enumerating owners per token would be O(L * N / BC)
  // planner work for a constant answer. Mark the page domain directly instead.
  bool dense_write = true;
  for (int level = 0; level < stream.level_count && dense_write; ++level) {
    dense_write = stream.level_write_can_zero[level] == 0;
  }
  const std::int64_t total = dense_write
      ? static_cast<std::int64_t>(args.plan.logical_page_count)
      : static_cast<std::int64_t>(stream.batch_heads) * stream.token_count;

  for (std::int64_t linear = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       linear < total;
       linear += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    if (dense_write) {
      mark_page(args, static_cast<std::uint64_t>(linear), 0u);
      continue;
    }
    const int bh = static_cast<int>(linear / stream.token_count);
    const int token = static_cast<int>(linear % stream.token_count);
    mark_owners_for_token(
        args, bh, token,
        static_cast<std::uint32_t>(token / stream.token_tile));
  }
}

__global__ void new_flags_kernel(
    PagePlanStorage plan, PagePlanCacheView cache, std::uint32_t touched_count) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= touched_count) return;
  const std::uint64_t logical = plan.sorted_ids[index];
  const bool valid = logical != UINT64_MAX && logical < cache.logical_page_count;
  const bool previous_different = index == 0 || plan.sorted_ids[index - 1] != logical;
  plan.new_flags[index] = valid && previous_different &&
      cache.page_table[logical] == kAbsentPage;
}

__global__ void new_pages_kernel(PagePlanStorage plan, std::uint32_t touched_count) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= touched_count || plan.new_flags[index] == 0) return;
  const std::uint32_t rank = plan.new_ranks[index];
  if (rank >= plan.max_new_pages) {
    atomicExch(plan.status, kPagePlanOverflow);
    return;
  }
  const std::uint64_t logical = plan.sorted_ids[index];
  plan.new_logical_ids[rank] = logical;
  plan.new_earliest_tiles[rank] = plan.earliest_epoch_tile[logical];
}

//: §7.1 `CAPPED`, step 1 of 3 -- key the new pages by admission order.
//: `key = (first_write_tile << 32) | logical_id`. `first_write_tile` is the EXACT
//: per-page minimum the mark phase recorded (V2's atomicCAS-on-packed loop,
//: unchanged), so ascending key order is exactly §7.1's "increasing
//: `first_write_tile` order", with the logical id as a deterministic tie break.
//: Ranks beyond `new_count` are `UINT64_MAX` so the fixed-capacity sort keeps them
//: at the tail.
__global__ void admit_keys_kernel(PagePlanStorage plan) {
  const std::uint32_t rank = blockIdx.x * blockDim.x + threadIdx.x;
  if (rank >= plan.max_new_pages) return;
  if (rank >= *plan.new_count) {
    plan.admit_keys[rank] = UINT64_MAX;
    return;
  }
  const std::uint64_t logical = plan.new_logical_ids[rank];
  const std::uint64_t tile = plan.new_earliest_tiles[rank] & 0xffffffffull;
  plan.admit_keys[rank] = (tile << 32) | (logical & 0xffffffffull);
}

//: §7.1 `CAPPED`, step 3 of 3 -- rewrite the new-page list in admission order,
//: truncated to the admitted budget. Only runs when the cap actually bit
//: (`capped_pages > 0`), so the uncapped path publishes in exactly the order V2
//: published in and this increment adds no reordering to the common case.
__global__ void admit_apply_kernel(PagePlanStorage plan, std::uint64_t epoch) {
  if (*plan.capped_pages == 0) return;
  const std::uint32_t rank = blockIdx.x * blockDim.x + threadIdx.x;
  if (rank >= *plan.new_count) return;
  const std::uint64_t key = plan.admit_sorted[rank];
  plan.new_logical_ids[rank] = key & 0xffffffffull;
  plan.new_earliest_tiles[rank] =
      ((epoch & 0xffffffffull) << 32) | (key >> 32);
}

__global__ void finalize_status_kernel(
    PagePlanStorage plan, PagePlanCacheView cache,
    std::uint64_t generation, std::uint64_t epoch) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  *plan.capped_pages = 0;
  if (*plan.status == kPagePlanOverflow) return;
  if (!cache.identity_physical_ids &&
      *cache.next_compact_slot != static_cast<std::uint64_t>(*cache.allocated_count)) {
    *plan.status = kPagePlanOverflow;
    return;
  }
  if (generation != *cache.generation || generation != *plan.plan_generation ||
      epoch != *plan.plan_epoch) {
    *plan.status = kPagePlanOverflow;
    return;
  }
  const std::uint64_t base = cache.identity_physical_ids
      ? static_cast<std::uint64_t>(*cache.allocated_count)
      : *cache.next_compact_slot;
  std::uint32_t count = *plan.new_count;

  // §7.1's allocator hard cap. V2 treated `required > max_physical_pages` as an
  // ERROR; architecture §4.5 promises the cap degrades gracefully, so V3 admits
  // in `first_write_tile` order until the cap is reached and DENIES the rest.
  // The denied pages' write factors are annihilated by the admit mask -- the same
  // rule as a structurally zero write. Capping CHANGES THE FUNCTION COMPUTED, so
  // `capped_pages` is a reported quantity and the status is terminal-distinct
  // from `READY`; it is never silently folded into success.
  if (base + count > cache.max_physical_pages) {
    const std::uint64_t budget =
        base >= cache.max_physical_pages ? 0ull : cache.max_physical_pages - base;
    *plan.capped_pages = count - static_cast<std::uint32_t>(budget);
    count = static_cast<std::uint32_t>(budget);
    *plan.new_count = count;
  }
  const std::uint64_t required = base + count;
  *plan.required_slots = required;

  // Growth takes precedence over the cap: §7.1's lifecycle re-runs `finalize`
  // after a `NEED_GROWTH` host grow, and the mark set is unchanged across that
  // retry, so the cap decision is recomputed identically and becomes terminal on
  // the second pass. Reporting `CAPPED` before growth would strand pages the
  // arena is about to be able to hold.
  if (required > *cache.mapped_capacity_pages) {
    *plan.status = kPagePlanNeedGrowth;
  } else {
    __threadfence();
    atomicExch(plan.status,
               *plan.capped_pages != 0 ? kPagePlanCapped : kPagePlanReady);
  }
}

__global__ void count_new_kernel(PagePlanStorage plan, std::uint32_t touched_count) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (touched_count == 0) {
    *plan.new_count = 0;
    return;
  }
  const std::uint32_t last = touched_count - 1;
  const std::uint32_t count = plan.new_ranks[last] + plan.new_flags[last];
  *plan.new_count = min(count, plan.max_new_pages);
  if (count > plan.max_new_pages) atomicExch(plan.status, kPagePlanOverflow);
}

//: `READY` and `CAPPED` are both SUCCESSFUL terminal plan states: a capped plan is
//: fully published for the pages it admitted, and the denied ones are annihilated
//: at the write side, not left half-committed. Publication therefore accepts both,
//: and only `OVERFLOW`/`NEED_GROWTH` block it.
__device__ __forceinline__ bool plan_publishable(std::uint32_t status) {
  return status == kPagePlanReady || status == kPagePlanCapped;
}

__global__ void publish_precheck_kernel(
    PagePlanStorage plan, PagePlanCacheView cache,
    std::uint64_t generation, std::uint64_t epoch) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const std::uint32_t current_status = *plan.status;
  if (!plan_publishable(current_status) ||
      generation != *cache.generation || epoch != *plan.plan_epoch ||
      generation != *plan.plan_generation ||
      (!cache.identity_physical_ids &&
       *cache.next_compact_slot != static_cast<std::uint64_t>(*cache.allocated_count))) {
    atomicExch(plan.status, kPagePlanOverflow);
    return;
  }
  const std::uint32_t count = *plan.new_count;
  const std::uint64_t base = cache.identity_physical_ids
      ? 0
      : *cache.next_compact_slot;
  const std::uint64_t required = cache.identity_physical_ids
      ? static_cast<std::uint64_t>(*cache.allocated_count) + count
      : base + count;
  if (required > *cache.mapped_capacity_pages || required > cache.max_physical_pages) {
    atomicExch(plan.status, kPagePlanOverflow);
    return;
  }
  const std::uint32_t allocated_base = *cache.allocated_count;
  for (std::uint32_t rank = 0; rank < count; ++rank) {
    const std::uint64_t logical = plan.new_logical_ids[rank];
    const std::uint64_t physical_u64 = cache.identity_physical_ids
        ? logical : base + rank;
    if (logical >= cache.logical_page_count ||
        logical > static_cast<std::uint64_t>(INT32_MAX) ||
        physical_u64 >= cache.max_physical_pages ||
        physical_u64 > static_cast<std::uint64_t>(INT32_MAX) ||
        cache.page_table[logical] != kAbsentPage ||
        allocated_base + rank >= cache.max_physical_pages) {
      atomicExch(plan.status, kPagePlanOverflow);
      return;
    }
  }
}

__global__ void publish_kernel(
    PagePlanStorage plan, PagePlanCacheView cache,
    std::uint64_t generation, std::uint64_t epoch) {
  if (blockIdx.x != 0 || threadIdx.x != 0 ||
      !plan_publishable(*plan.status) ||
      generation != *cache.generation || epoch != *plan.plan_epoch ||
      generation != *plan.plan_generation) return;
  const std::uint32_t count = *plan.new_count;
  const std::uint64_t base = cache.identity_physical_ids
      ? 0 : *cache.next_compact_slot;
  const std::uint32_t allocated_base = *cache.allocated_count;
  for (std::uint32_t rank = 0; rank < count; ++rank) {
    const std::uint64_t logical = plan.new_logical_ids[rank];
    const std::uint64_t physical_u64 = cache.identity_physical_ids
        ? logical : base + rank;
    const std::int32_t physical = static_cast<std::int32_t>(physical_u64);
    cache.page_table[logical] = physical;
    cache.earliest_write_epoch_tile[logical] = plan.new_earliest_tiles[rank];
    cache.allocated_logical_page_ids[allocated_base + rank] =
        static_cast<std::int32_t>(logical);
  }
  if (!cache.identity_physical_ids) {
    *cache.next_compact_slot = base + count;
  }
  *cache.allocated_count = allocated_base + count;
  if (count == 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(cache.hit_count), 1ull);
  } else {
    atomicAdd(reinterpret_cast<unsigned long long*>(cache.logical_allocation_count),
              static_cast<unsigned long long>(count));
  }
  __threadfence();
  // Preserve `CAPPED` through publication: it is the reported outcome of the call,
  // not a transient condition that success overwrites (§7.1 "reported, never silent").
  atomicExch(plan.status,
             *plan.capped_pages != 0 ? kPagePlanCapped : kPagePlanReady);
}

__global__ void zero_new_pages_kernel(
    PagePlanStorage plan, PagePlanCacheView cache,
    std::uint64_t generation, std::uint64_t epoch) {
  const std::uint32_t count = *plan.new_count;
  const std::uint64_t value_page_elements =
      static_cast<std::uint64_t>(cache.page_states) * cache.dqk * cache.dv;
  const std::uint64_t mass_page_elements =
      static_cast<std::uint64_t>(cache.page_states) * cache.dqk;
  for (std::uint32_t rank = blockIdx.x;
       rank < count;
       rank += gridDim.x) {
    if (!plan_publishable(*plan.status)) return;
    if (generation != *cache.generation || generation != *plan.plan_generation ||
        epoch != *plan.plan_epoch) return;
    const std::uint64_t logical = plan.new_logical_ids[rank];
    const std::uint64_t base = cache.identity_physical_ids
        ? 0 : *cache.next_compact_slot;
    const std::uint64_t physical = cache.identity_physical_ids
        ? logical : base + rank;
    if (physical >= cache.max_physical_pages) {
      atomicExch(plan.status, kPagePlanOverflow);
      return;
    }
    float* value = cache.value_base + physical * value_page_elements;
    for (std::uint64_t index = threadIdx.x; index < value_page_elements;
         index += blockDim.x)
      value[index] = 0.0f;
    if (cache.mass_base != nullptr) {
      float* mass = cache.mass_base + physical * mass_page_elements;
      for (std::uint64_t index = threadIdx.x; index < mass_page_elements;
           index += blockDim.x)
        mass[index] = 0.0f;
    }
  }
}

__global__ void page_cache_reset_kernel(
    std::int32_t* page_table,
    std::uint64_t* earliest_write_epoch_tile,
    std::int32_t* allocated_logical_page_ids,
    std::uint32_t* allocated_count,
    std::uint64_t logical_page_count,
    std::uint64_t allocated_capacity) {
  const std::uint32_t count = min(
      *allocated_count, static_cast<std::uint32_t>(allocated_capacity));
  for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += gridDim.x * blockDim.x) {
    const std::int32_t logical = allocated_logical_page_ids[index];
    if (logical >= 0 && static_cast<std::uint64_t>(logical) < logical_page_count) {
      page_table[logical] = kAbsentPage;
      earliest_write_epoch_tile[logical] = 0;
    }
    allocated_logical_page_ids[index] = kAbsentPage;
  }
}

__global__ void page_cache_reset_metadata_kernel(
    std::uint32_t* allocated_count,
    std::uint64_t* next_compact_slot) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *allocated_count = 0;
    *next_compact_slot = 0;
  }
}

void check_pointer(const void* pointer, const char* name) {
  if (pointer == nullptr) throw std::invalid_argument(name);
}

void validate_plan(const PagePlanStorage& plan, bool require_sort_workspace) {
  if (plan.logical_page_count == 0) {
    throw std::invalid_argument("page plan logical_page_count must be nonzero");
  }
  if (plan.touched_capacity == 0 || plan.max_new_pages == 0 ||
      plan.max_new_pages > plan.touched_capacity) {
    throw std::invalid_argument("page plan capacities are invalid");
  }
  check_pointer(plan.touched_tags, "page plan touched tags");
  check_pointer(plan.earliest_epoch_tile, "page plan earliest epoch/tile");
  check_pointer(plan.touched_ids, "page plan touched ids");
  check_pointer(plan.sorted_ids, "page plan sorted ids");
  check_pointer(plan.touched_count, "page plan touched count");
  check_pointer(plan.new_flags, "page plan new flags");
  check_pointer(plan.new_ranks, "page plan new ranks");
  check_pointer(plan.new_logical_ids, "page plan new logical ids");
  check_pointer(plan.new_earliest_tiles, "page plan new earliest tiles");
  check_pointer(plan.admit_keys, "page plan admit keys");
  check_pointer(plan.admit_sorted, "page plan sorted admit keys");
  check_pointer(plan.capped_pages, "page plan capped page count");
  check_pointer(plan.new_count, "page plan new count");
  check_pointer(plan.required_slots, "page plan required slots");
  check_pointer(plan.status, "page plan status");
  check_pointer(plan.plan_generation, "page plan generation");
  check_pointer(plan.plan_epoch, "page plan epoch");
  check_pointer(plan.last_epoch, "page plan last epoch");
  if (require_sort_workspace) {
    check_pointer(plan.sort_temp, "page plan CUB workspace");
    if (plan.sort_temp_bytes == 0) {
      throw std::invalid_argument("page plan CUB workspace is empty");
    }
  }
}

void validate_cache(const PagePlanCacheView& cache) {
  check_pointer(cache.page_table, "page cache page table");
  check_pointer(cache.earliest_write_epoch_tile,
               "page cache earliest epoch/tile");
  check_pointer(cache.allocated_logical_page_ids,
               "page cache allocated logical ids");
  check_pointer(cache.value_base, "page cache value base");
  check_pointer(cache.allocated_count, "page cache allocated count");
  check_pointer(cache.mapped_capacity_pages, "page cache mapped capacity");
  check_pointer(cache.next_compact_slot, "page cache compact slot");
  check_pointer(cache.generation, "page cache generation");
  if (cache.logical_page_count == 0 || cache.max_physical_pages == 0 ||
      cache.page_states == 0) {
    throw std::invalid_argument("page cache geometry is invalid");
  }
}

void validate_mark_args(const PagePlanMarkArgs& args) {
  validate_plan(args.plan, false);
  const PagePlanStreamView& stream = args.stream;
  const PagePlanOwnerGeometry& geom = args.owners;
  if (stream.level_count <= 0 || stream.level_count > kPagePlanMaxLevels ||
      stream.batch_heads <= 0 || stream.token_count <= 0 || stream.n_tiles <= 0 ||
      args.pages_per_bh == 0 || args.page_states == 0 ||
      args.max_suffix_combinations == 0) {
    throw std::invalid_argument("page plan stream geometry is invalid");
  }
  // Spec §7.1's `token_tile` assert, settled PARAMETRIC over `BT in {16, 32}`
  // rather than V2's hard `== 32` (findings `paging.md` D3). V2's 32 was a
  // property of its route arena layout; V3's amplitude stream is `BT`-shaped by
  // construction (`[BH, n_tiles, BT, pitch]`), and pinning `BT := 32` under
  // paging would let a memory-management flag move the §6.2 compute plan -- the
  // shipping default selects `BT = 16` on the `decay = None` arm at `BC = 16`.
  // The bound is the schedule's own contract (`BT <= 32`, one token word), not 32
  // itself.
  if (stream.token_tile != 16 && stream.token_tile != 32) {
    throw std::invalid_argument(
        "page plan token_tile must be the plan's BT, a compile-time constant in {16, 32} "
        "(spec §6); V2's hard-asserted 32 is settled parametric here");
  }
  if (stream.n_tiles != (stream.token_count + stream.token_tile - 1) / stream.token_tile) {
    throw std::invalid_argument("page plan n_tiles disagrees with token_count / BT");
  }
  if (geom.first_spanned_level < 0 || geom.first_spanned_level >= stream.level_count ||
      geom.slice_digits <= 0 || geom.n_slices <= 0 || geom.owners_total <= 0) {
    throw std::invalid_argument("page plan owner geometry is invalid");
  }
  if (static_cast<std::uint32_t>(geom.owners_total) != args.pages_per_bh) {
    throw std::invalid_argument(
        "page plan pages_per_bh must equal owners_total (spec §7.1: one page is one owner)");
  }
  const std::uint64_t expected_pages =
      static_cast<std::uint64_t>(stream.batch_heads) * args.pages_per_bh;
  if (expected_pages != args.plan.logical_page_count) {
    throw std::invalid_argument("page plan pages_per_bh disagrees with logical pages");
  }
  check_pointer(stream.level_amps, "page plan level amplitude pointers");
  check_pointer(stream.level_width, "page plan level widths");
  check_pointer(stream.level_pitch, "page plan level pitches");
  check_pointer(stream.level_write_can_zero, "page plan write-zero metadata");
  check_pointer(geom.owner_stride, "page plan owner strides");
}

}  // namespace

void launch_page_plan_reset(
    const PagePlanStorage& plan,
    std::uint64_t generation,
    std::uint64_t epoch,
    cudaStream_t stream) {
  validate_plan(plan, false);
  check_pointer(plan.touched_count, "page plan touched count");
  check_pointer(plan.last_epoch, "page plan last epoch");
  if (generation == 0 || epoch == 0 || (generation >> 32) != 0 ||
      (epoch >> 32) != 0) {
    throw std::invalid_argument(
        "page plan generation/epoch must be nonzero and fit uint32 tags");
  }
  if (plan.touched_capacity != 0) {
    const int blocks = std::max<std::uint32_t>(
        1, std::min<std::uint32_t>(1024,
            (plan.touched_capacity + kThreads - 1) / kThreads));
    reset_clear_kernel<<<
        blocks, kThreads, 0, stream>>>(plan);
  }
  reset_state_kernel<<<1, 1, 0, stream>>>(plan, generation, epoch);
}

void launch_page_plan_mark(const PagePlanMarkArgs& args, cudaStream_t stream) {
  validate_mark_args(args);
  const int blocks = 1024;
  mark_kernel<<<blocks, kThreads, 0, stream>>>(args);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void admit_mask_kernel(
    const std::int32_t* page_table, std::uint32_t* admit,
    std::uint64_t logical_page_count, std::uint32_t pages_per_bh,
    std::uint32_t words_per_bh) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= logical_page_count) return;
  const std::uint32_t bh = static_cast<std::uint32_t>(index / pages_per_bh);
  const std::uint32_t owner = static_cast<std::uint32_t>(index % pages_per_bh);
  if (page_table[index] == kAbsentPage) return;
  atomicOr(&admit[static_cast<std::uint64_t>(bh) * words_per_bh + (owner >> 5)],
           std::uint32_t{1} << (owner & 31));
}

void launch_page_admit_mask(
    const std::int32_t* page_table,
    std::uint32_t* admit,
    std::uint64_t logical_page_count,
    std::uint32_t pages_per_bh,
    std::uint32_t batch_heads,
    cudaStream_t stream) {
  check_pointer(page_table, "admit mask page table");
  check_pointer(admit, "admit mask");
  if (logical_page_count == 0 || pages_per_bh == 0 || batch_heads == 0 ||
      logical_page_count !=
          static_cast<std::uint64_t>(batch_heads) * pages_per_bh) {
    throw std::invalid_argument("admit mask geometry is invalid");
  }
  const std::uint32_t words_per_bh = (pages_per_bh + 31) / 32;
  C10_CUDA_CHECK(cudaMemsetAsync(
      admit, 0,
      static_cast<std::size_t>(batch_heads) * words_per_bh * sizeof(std::uint32_t),
      stream));
  const std::uint64_t blocks = (logical_page_count + kThreads - 1) / kThreads;
  admit_mask_kernel<<<static_cast<unsigned>(std::min<std::uint64_t>(blocks, 2147483647u)),
                      kThreads, 0, stream>>>(
      page_table, admit, logical_page_count, pages_per_bh, words_per_bh);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_page_plan_finalize(
    const PagePlanFinalizeArgs& args,
    cudaStream_t stream) {
  validate_plan(args.plan, true);
  validate_cache(args.cache);
  // Untouched slots are generation-cleared UINT64_MAX sentinels. Sorting the
  // declared persistent capacity keeps planning stream-ordered and removes
  // the old host count read between mark and finalize.
  const auto touched_count = args.plan.touched_capacity;
  std::size_t sort_temp_bytes = args.plan.sort_temp_bytes;
  cub::DeviceRadixSort::SortKeys(
      args.plan.sort_temp, sort_temp_bytes,
      args.plan.touched_ids, args.plan.sorted_ids,
      touched_count, 0, 64, stream);
  new_flags_kernel<<<
      std::max<std::uint32_t>(1, (touched_count + kThreads - 1) / kThreads),
      kThreads, 0, stream>>>(args.plan, args.cache, touched_count);
  std::size_t scan_temp_bytes = args.plan.sort_temp_bytes;
  cub::DeviceScan::ExclusiveSum(
      args.plan.sort_temp, scan_temp_bytes,
      args.plan.new_flags, args.plan.new_ranks,
      touched_count, stream);
  count_new_kernel<<<1, 1, 0, stream>>>(args.plan, touched_count);
  new_pages_kernel<<<
      std::max<std::uint32_t>(1, (touched_count + kThreads - 1) / kThreads),
      kThreads, 0, stream>>>(args.plan, touched_count);
  // §7.1 `CAPPED` admission order, in three stream-ordered steps. The sort is over
  // the DECLARED new-page capacity, not a host-read count, for the same reason the
  // touched-set sort above is: keeping planning stream-ordered is what lets the
  // whole plane be graph-captured (§7.1's CUDA-graph rule).
  const std::uint32_t admit_blocks =
      std::max<std::uint32_t>(1, (args.plan.max_new_pages + kThreads - 1) / kThreads);
  admit_keys_kernel<<<admit_blocks, kThreads, 0, stream>>>(args.plan);
  std::size_t admit_temp_bytes = args.plan.sort_temp_bytes;
  cub::DeviceRadixSort::SortKeys(
      args.plan.sort_temp, admit_temp_bytes,
      args.plan.admit_keys, args.plan.admit_sorted,
      args.plan.max_new_pages, 0, 64, stream);
  finalize_status_kernel<<<1, 1, 0, stream>>>(
      args.plan, args.cache, args.generation, args.epoch);
  admit_apply_kernel<<<admit_blocks, kThreads, 0, stream>>>(args.plan, args.epoch);
}

void launch_page_plan_publish(
    const PagePlanPublishArgs& args,
    cudaStream_t stream) {
  validate_plan(args.plan, false);
  validate_cache(args.cache);
  publish_precheck_kernel<<<1, 1, 0, stream>>>(
      args.plan, args.cache, args.generation, args.epoch);
  const int blocks = std::max<int>(
      1, std::min<int64_t>(65535, (args.plan.max_new_pages + kThreads - 1) / kThreads));
  zero_new_pages_kernel<<<blocks, kThreads, 0, stream>>>(
      args.plan, args.cache, args.generation, args.epoch);
  publish_kernel<<<1, 1, 0, stream>>>(
      args.plan, args.cache, args.generation, args.epoch);
}

void launch_page_cache_reset(
    std::int32_t* page_table,
    std::uint64_t* earliest_write_epoch_tile,
    std::int32_t* allocated_logical_page_ids,
    std::uint32_t* allocated_count,
    std::uint64_t* next_compact_slot,
    std::uint64_t* hit_count,
    std::uint64_t* logical_allocation_count,
    std::uint64_t logical_page_count,
    std::uint64_t allocated_capacity,
    cudaStream_t stream) {
  check_pointer(page_table, "page cache reset page table");
  check_pointer(earliest_write_epoch_tile, "page cache reset earliest metadata");
  check_pointer(allocated_logical_page_ids, "page cache reset allocated ids");
  check_pointer(allocated_count, "page cache reset allocated count");
  check_pointer(next_compact_slot, "page cache reset compact slot");
  check_pointer(hit_count, "page cache reset hit count");
  check_pointer(logical_allocation_count, "page cache reset logical allocation count");
  if (logical_page_count == 0 || allocated_capacity == 0 ||
      allocated_capacity > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("page cache reset capacities are invalid");
  }
  const int blocks = std::max<int>(1, std::min<std::uint64_t>(
      256, (allocated_capacity + kThreads - 1) / kThreads));
  page_cache_reset_kernel<<<blocks, kThreads, 0, stream>>>(
      page_table, earliest_write_epoch_tile, allocated_logical_page_ids,
      allocated_count, logical_page_count,
      allocated_capacity);
  page_cache_reset_metadata_kernel<<<1, 1, 0, stream>>>(
      allocated_count, next_compact_slot);
  C10_CUDA_CHECK(cudaMemsetAsync(hit_count, 0, sizeof(std::uint64_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      logical_allocation_count, 0, sizeof(std::uint64_t), stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::size_t page_plan_workspace_bytes(std::uint32_t touched_capacity) {
  if (touched_capacity == 0) return 0;
  std::size_t sort_bytes = 0;
  std::size_t scan_bytes = 0;
  cub::DeviceRadixSort::SortKeys(
      nullptr, sort_bytes,
      static_cast<const std::uint64_t*>(nullptr),
      static_cast<std::uint64_t*>(nullptr),
      touched_capacity, 0, 64);
  cub::DeviceScan::ExclusiveSum(
      nullptr, scan_bytes,
      static_cast<const std::uint32_t*>(nullptr),
      static_cast<std::uint32_t*>(nullptr),
      touched_capacity);
  return std::max(sort_bytes, scan_bytes);
}

}  // namespace rola::paging
