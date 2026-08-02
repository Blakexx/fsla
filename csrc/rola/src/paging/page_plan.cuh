#pragma once

#include <cuda_runtime_api.h>
#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

namespace rola::paging {

constexpr std::uint32_t kPagePlanReady = 0;
constexpr std::uint32_t kPagePlanNeedGrowth = 1;
constexpr std::uint32_t kPagePlanOverflow = 2;
//: Spec §7.1's third TERMINAL status. `kPagePlanOverflow` stays what it was -- a
//: DEFECT (a capacity invariant of the plan workspace itself was violated, or a
//: generation/epoch mismatch). `kPagePlanCapped` is the graceful degradation
//: architecture §4.5 promises: the allocator's hard cap `max_physical_pages` was
//: reached, pages were admitted in increasing `first_write_tile` order until it
//: was, and the denied pages' writes are annihilated by the admit mask. It
//: CHANGES THE FUNCTION COMPUTED, so it is reported (`capped_pages`), never silent.
constexpr std::uint32_t kPagePlanCapped = 3;

constexpr std::int32_t kAbsentPage = -1;
constexpr int kPagePlanMaxLevels = 8;

//: The V3 write-amplitude stream (spec §4 / `planner/schedule.py:Schedule.write_amps`).
//:
//: Re-key note (V2 -> V3, findings `paging.md` D1). V2 fed the mark phase two
//: arenas -- a bf16 tile-major `write_factors` and a separate
//: `write_support_words` bitset -- because the V2 producer stored value and
//: support separately. V3 stores ONE fp32 tensor per level, `[BH, n_tiles, BT,
//: pitch_l]`, token-major, whose contract is "a nonzero entry IS a live token"
//: (§4). The exact `!= 0` test is the same entmax-support authority used
//: everywhere else (§1.2 -- never a threshold), so one arena replaces two and no
//: predicate changes: §7.1's "a zero complete write factor may neither deposit
//: nor allocate" is evaluated directly on the amplitude.
struct PagePlanStreamView {
  //: Device-resident array of `level_count` pointers, one per routing level.
  const float* const* level_amps;
  const int* level_width;   //: device [D]
  const int* level_pitch;   //: device [D] -- `Plan.amplitude_pitch(width_l)`
  //: device [D]; 0 iff the level's write factor is structurally nonzero (a
  //: softmax write level). All-zero enables §7.1's dense short-circuit.
  const std::uint8_t* level_write_can_zero;
  int level_count;
  int batch_heads;
  int token_count;
  int n_tiles;
  //: `BT`. Spec §7.1's `token_tile`, settled PARAMETRIC over `{16, 32}` rather
  //: than pinned at V2's hard-asserted 32 (findings `paging.md` D3). V2's 32 was
  //: a property of its route arena LAYOUT; V3's stream is `BT`-shaped by
  //: construction, and pinning would let a memory-management flag move the §6.2
  //: compute plan.
  int token_tile;
};

//: The §3 owner rectangle, as the mark phase needs it. Under §7.1's
//: `page_states := BC` one page IS one owner, so the mark phase enumerates
//: OWNERS (`N / BC` of them), not leaves (`N`). The projection is EXACT, not a
//: candidate superset: the live leaf set at a token is a Cartesian product of
//: per-level live digit sets, an owner rectangle is a Cartesian box, and two
//: such sets intersect iff they intersect factorwise (findings `paging.md` D2).
struct PagePlanOwnerGeometry {
  const int* owner_stride;  //: device [D]; `prod(width[l+1:j])` for `l < j`, else 1
  int first_spanned_level;  //: `j`
  int slice_digits;         //: `c`
  int n_slices;             //: `width_j / c`
  int owners_total;         //: `N / BC`, and equal to `pages_per_bh`
};

struct PagePlanStorage {
  // All arrays are persistent, predeclared workspace. touched_tags and
  // earliest_epoch_tile are indexed by flattened logical page id.
  std::uint64_t* touched_tags;
  std::uint64_t* earliest_epoch_tile;
  std::uint64_t* touched_ids;
  std::uint64_t* sorted_ids;
  std::uint32_t* touched_count;
  std::uint32_t* new_flags;
  std::uint32_t* new_ranks;
  std::uint64_t* new_logical_ids;
  std::uint64_t* new_earliest_tiles;
  //: §7.1 `CAPPED` admission ordering. `admit_keys[r] = (first_write_tile << 32) |
  //: logical_id` for new page `r`; sorting it ascending gives EARLIEST-WRITER-WINS
  //: with a deterministic tie break (the logical id), which is what makes the
  //: degraded function reproducible rather than launch-order dependent. The
  //: `first_write_tile` minimum is exact (V2's atomicCAS loop, reused verbatim),
  //: so the order is exact too.
  std::uint64_t* admit_keys;
  std::uint64_t* admit_sorted;
  //: Number of new pages DENIED by the cap this call. Nonzero <=> status is
  //: `kPagePlanCapped`. Telemetry, never a silent condition (§7.1).
  std::uint32_t* capped_pages;
  std::uint32_t* new_count;
  std::uint64_t* required_slots;
  std::uint32_t* status;
  std::uint64_t* plan_generation;
  std::uint64_t* plan_epoch;
  std::uint64_t* last_epoch;
  void* sort_temp;
  std::size_t sort_temp_bytes;
  std::uint32_t touched_capacity;
  std::uint32_t max_new_pages;
  std::uint64_t logical_page_count;
};

// Cache metadata is device-resident. VMM growth changes mapped_capacity_pages,
// but never changes page-table identity or the logical ids in a preserved plan.
struct PagePlanCacheView {
  std::int32_t* page_table;
  std::uint64_t* earliest_write_epoch_tile;
  std::int32_t* allocated_logical_page_ids;
  float* value_base;
  float* mass_base;
  std::uint32_t* allocated_count;
  std::uint64_t* mapped_capacity_pages;
  std::uint64_t* next_compact_slot;
  std::uint64_t* generation;
  std::uint64_t* hit_count;
  std::uint64_t* logical_allocation_count;
  std::uint64_t logical_page_count;
  std::uint64_t max_physical_pages;
  std::uint32_t page_states;
  std::uint32_t dqk;
  std::uint32_t dv;
  bool identity_physical_ids;
};

struct PagePlanMarkArgs {
  PagePlanStorage plan;
  PagePlanStreamView stream;
  PagePlanOwnerGeometry owners;
  std::uint64_t generation;
  std::uint64_t epoch;
  //: §7.1: `page_states := BC`. Retained as a field rather than hardcoded so the
  //: accessor arithmetic in `paged_state.cuh` stays the SAME expression it was
  //: (§7.1: "V2's existing arithmetic with a chosen constant, not a new keying").
  std::uint32_t page_states;
  std::uint32_t pages_per_bh;
  //: Guard on the per-token owner Cartesian walk, carried over from V2's
  //: `max_suffix_combinations`. Exceeding it is a DEFECT (`kPagePlanOverflow`),
  //: not a cap: the cap is on PAGES, this is on planner work.
  std::uint32_t max_suffix_combinations;
};

struct PagePlanFinalizeArgs {
  PagePlanStorage plan;
  PagePlanCacheView cache;
  std::uint64_t generation;
  std::uint64_t epoch;
};

struct PagePlanPublishArgs {
  PagePlanStorage plan;
  PagePlanCacheView cache;
  std::uint64_t generation;
  std::uint64_t epoch;
};

void launch_page_plan_reset(
    const PagePlanStorage& plan,
    std::uint64_t generation,
    std::uint64_t epoch,
    cudaStream_t stream);

void launch_page_plan_mark(
    const PagePlanMarkArgs& args,
    cudaStream_t stream);

void launch_page_plan_finalize(
    const PagePlanFinalizeArgs& args,
    cudaStream_t stream);

void launch_page_plan_publish(
    const PagePlanPublishArgs& args,
    cudaStream_t stream);

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
    cudaStream_t stream);

//: Emit spec §7.1's admit mask: one bit per OWNER, `[BH, ceil(owners_total/32)]`
//: `uint32`, bit `owner & 31` of word `owner >> 5`. The bit is set iff the owner's
//: logical page is PUBLISHED (`page_table[bh * pages_per_bh + owner] != kAbsentPage`).
//:
//: That single rule covers both roles the mask plays. Under `kPagePlanReady` every
//: write-candidate owner is published, so the mask is all-ones over the live set and
//: costs the consumer one word read at CTA init. Under `kPagePlanCapped` the denied
//: owners are exactly the unpublished ones, and the consumer's existing
//: `if (!admitted) wr = 0u` annihilates their write factor -- §7.1's "same rule as a
//: structurally zero write, no new semantics, no new code path in the deposit".
void launch_page_admit_mask(
    const std::int32_t* page_table,
    std::uint32_t* admit,
    std::uint64_t logical_page_count,
    std::uint32_t pages_per_bh,
    std::uint32_t batch_heads,
    cudaStream_t stream);

std::size_t page_plan_workspace_bytes(std::uint32_t touched_capacity);

}  // namespace rola::paging
