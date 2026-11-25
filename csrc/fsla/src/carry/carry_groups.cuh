// THE CARRY'S GROUPS, a call's operands and a group's words -- docs/internals/carry/carry_groups.md
#pragma once

#include <cstdint>

#include "plan/cta_carve.h"
#include "plan/mask_prepare_layout.h"

namespace fsla {
namespace carry {
namespace detail {

constexpr int kSides = plan::kLevelSides;
constexpr int kLevels = plan::kCarveMaxLevels;

//: THE CALL'S OPERANDS a role reads its groups from -- see docs/internals/carry/carry_groups.md#operands
struct Operands {
  const std::uint32_t* mask[kSides];
  const std::uint8_t* workspace;
  std::int64_t prepared[kSides];
  const std::uint8_t* density;
  const std::uint8_t* values;
  std::int64_t tokens;
  std::int64_t ctas;
  int line_bytes;
  int width_bits[kLevels];
  int levels;
  int words;
  int windows;
};

//: A SIDE'S FIRST LEVEL-SIDE, in the bundle's slot order: the read side's are the first `kLevels`, the write side's the next
constexpr int side_at(bool read) { return read ? 0 : kLevels; }

//: THE CARVE OF A CALL, from its densities as the mask-preparation pass reads them (a byte `b` is `b + 1` 256ths, the read side's row first) -- see docs/internals/carry/carry_groups.md#carve
__device__ __forceinline__ plan::PrefillCarve carve_of(const Operands& o) {
  int width[kLevels] = {}, write[kLevels] = {}, read[kLevels] = {};
#pragma unroll
  for (int l = 0; l < kLevels; ++l) {
    width[l] = o.width_bits[l];
    if (l < o.levels) {
      read[l] = static_cast<int>(o.density[l]) + 1;
      write[l] = static_cast<int>(o.density[o.levels + l]) + 1;
    }
  }
  return plan::prefill_carve(o.levels, width, write, read);
}

//: WHETHER A LEVEL-SIDE IS WHOLE: its group is the level's size, and that alone -- see docs/internals/carry/carry_groups.md#whole
__device__ __forceinline__ bool whole(const plan::PrefillCarve& c, bool read, int level) {
  const int group = read ? c.read_group_bits[level] : c.write_group_bits[level];
  return group == c.width_bits[level];
}

//: THE LEVELS A SIDE LISTS, a bit a level: those that are not whole -- see docs/internals/carry/carry_groups.md#whole
__device__ __forceinline__ int listed_levels(const plan::PrefillCarve& c, bool read) {
  int listed = 0;
#pragma unroll
  for (int l = 0; l < kLevels; ++l) listed |= l < c.levels && !whole(c, read, l) ? 1 << l : 0;
  return listed;
}

//: A CTA'S PLACE IN THE CALL: its head, its slice of the value channels and its index along each level -- see docs/internals/carry/carry_groups.md#place
struct Place {
  std::int64_t head;
  int slice;
  int cta[kLevels];
};

__device__ __forceinline__ Place place_of(const Operands& o, const plan::PrefillCarve& c,
                                          std::int64_t at, int slices) {
  Place p{};
  p.slice = static_cast<int>(at % slices);
  const std::int64_t leaf = at / slices;
  p.head = leaf / o.ctas;
  plan::prefill_cta_at(c, leaf % o.ctas, p.cta);
  return p;
}

//: A FOLD'S LEVELS, formed once: a set's group on level `l` is `base[l] + ((set >> shift[l]) & mask[l])` -- see docs/internals/carry/carry_groups.md#sets
__device__ __forceinline__ void fold_levels(const plan::PrefillCarve& c, const Place& p, int fold,
                                            int* base, int* shift, int* mask) {
#pragma unroll
  for (int l = 0; l < kLevels; ++l) {
    const int fold_bits = c.fold_bits[l], carve_bits = c.carve_bits[l];
    const int group_bits = c.write_group_bits[l];
    const int half = (fold << fold_bits) & (l == c.split_level ? ~0 : 0);
    const int over = fold_bits - group_bits;
    base[l] = ((p.cta[l] << carve_bits) + half) >> group_bits;
    mask[l] = (1 << (over > 0 ? over : 0)) - 1;
    shift[l] = 0;
  }
  //: the set's bits a level, the write side's densest level's lowest; each of the carve's arrays is read before the level is compared with anything
  int taken = 0;
#pragma unroll
  for (int i = kLevels - 1; i >= 0; --i) {
#pragma unroll
    for (int l = 0; l < kLevels; ++l) {
      const int order = c.write_order[i], over = c.fold_bits[l] - c.write_group_bits[l];
      const bool here = i < c.levels && order == l;
      shift[l] = here ? taken : shift[l];
      taken += here ? (over > 0 ? over : 0) : 0;
    }
  }
}

//: A LANE'S ROW of a level-side's words in a window: `row[group]` is its word of the group, the mask's own at a group of one digit and the prepared words otherwise -- see docs/internals/carry/carry_groups.md#word
template <int WORDS>
__device__ __forceinline__ const std::uint32_t* level_row(const Operands& o, int s, int group_bits,
                                                          std::int64_t head, int window, int word) {
  const int wb = o.width_bits[s % kLevels], groups = 1 << (wb - group_bits);
  const std::int64_t at = std::int64_t{window} * WORDS + word;
  //: a word past the mask's last reads the mask's first word; the call's own tokens then count it for nothing
  const std::uint32_t* const raw = o.mask[s] + ((head * o.words + (at < o.words ? at : 0)) << wb);
  const std::int64_t block = (head * o.windows + window) * plan::prepared_block_bytes(wb, WORDS);
  const std::uint8_t* const words =
      o.workspace + o.prepared[s] + block + plan::prepared_words_at(groups);
  const std::uint32_t* const made = reinterpret_cast<const std::uint32_t*>(words) + word * groups;
  return group_bits == 0 ? raw : made;
}

//: A LEVEL-SIDE'S ROW A WINDOW ON, in words from the window before's -- see docs/internals/carry/carry_groups.md#word
template <int WORDS>
__device__ __forceinline__ std::int64_t level_step(const Operands& o, int s, int group_bits) {
  const int wb = o.width_bits[s % kLevels];
  const std::int64_t raw = std::int64_t{WORDS} << wb;
  const std::int64_t made =
      plan::prepared_block_bytes(wb, WORDS) / static_cast<int>(sizeof(std::uint32_t));
  return group_bits == 0 ? raw : made;
}

}  // namespace detail
}  // namespace carry
}  // namespace fsla
