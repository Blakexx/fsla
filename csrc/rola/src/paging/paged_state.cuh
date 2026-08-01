#pragma once

#include <cstdint>

namespace fla_rola::p0::shared_gram {

constexpr int kAbsentStatePage = -1;

// The only native continuation-state view. Fully committed and progressively
// mapped backing differ in ownership, not in consumer addressing.
struct PagedStateView {
  const int* page_table;
  float* value_base;
  float* mass_base;
  const uint64_t* earliest_epoch_tile;
  int logical_pages_per_bh;
  int page_states;
  int nc;
  int dqk;
  int dv;
  uint32_t generation;
  uint32_t plan_epoch;
};

struct ResolvedStatePage {
  int physical_page;
  int row;
  int first_write_tile;
  bool present;
};

__device__ __forceinline__ int logical_state_page(
    const PagedStateView& state, int leaf_state) {
  return leaf_state / state.page_states;
}

__device__ __forceinline__ ResolvedStatePage resolve_state_page(
    const PagedStateView& state, int bh, int leaf_state) {
  if (state.page_table == nullptr || state.page_states <= 0) {
    return {kAbsentStatePage, 0, -1, false};
  }
  const int logical_page = logical_state_page(state, leaf_state);
  const int logical_id = bh * state.logical_pages_per_bh + logical_page;
  const int physical_page = state.page_table[logical_id];
  int first_write_tile = -1;
  if (state.earliest_epoch_tile != nullptr) {
    const uint64_t packed = state.earliest_epoch_tile[logical_id];
    if (static_cast<uint32_t>(packed >> 32) == state.plan_epoch) {
      first_write_tile = static_cast<int>(static_cast<uint32_t>(packed));
    }
  }
  return {
      physical_page,
      leaf_state - logical_page * state.page_states,
      first_write_tile,
      physical_page != kAbsentStatePage,
  };
}

__device__ __forceinline__ bool page_has_history(
    const ResolvedStatePage& page) {
  return page.present && page.first_write_tile < 0;
}

__device__ __forceinline__ int64_t paged_value_offset(
    const PagedStateView& state,
    const ResolvedStatePage& page,
    int feature,
    int value) {
  return (((static_cast<int64_t>(page.physical_page) * state.page_states + page.row) *
               state.dqk +
           feature) *
              state.dv +
          value);
}

__device__ __forceinline__ int64_t paged_mass_offset(
    const PagedStateView& state,
    const ResolvedStatePage& page,
    int feature) {
  return ((static_cast<int64_t>(page.physical_page) * state.page_states + page.row) *
              state.dqk +
          feature);
}

}  // namespace fla_rola::p0::shared_gram
