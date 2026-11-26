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

//: FIXED by the carve's own order (`plan::PrefillCarve` and the prepared layout: the read side first): the write side among its two
constexpr int kWriteSide = 1;

//: THE CALL'S OPERANDS a role reads its groups from -- see docs/internals/carry/carry_groups.md#operands
struct Operands {
  const std::uint8_t* workspace;
  std::int64_t prepared[plan::kCarveSides];
  const std::uint8_t* density;
  int masked[plan::kCarveSides];
  const std::uint8_t* values;
  const std::uint8_t* rows[kSides];
  std::int64_t tokens;
  std::int64_t ctas;
  int line_bytes;
  int width_bits[kLevels];
  int levels;
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
      //: a level-side with no mask is walked whole, whatever its density byte: the mask's absence is the dense flag
      read[l] =
          (o.masked[0] >> l & 1) != 0 ? static_cast<int>(o.density[l]) + 1 : plan::kWholeDensity;
      write[l] = (o.masked[kWriteSide] >> l & 1) != 0
                     ? static_cast<int>(o.density[o.levels + l]) + 1
                     : plan::kWholeDensity;
    }
  }
  return plan::prefill_carve(o.levels, width, write, read);
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
__host__ __device__ __forceinline__ void fold_levels(const plan::PrefillCarve& c, const Place& p,
                                                     int fold, int* base, int* shift, int* mask) {
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

//: A SIDE'S LEAF GROUPS' LINES of a batch-head's first window, as mask words: leaf group `leaf`'s line is `WORDS` words from `leaf * WORDS`, a word a mask word of the window -- see docs/internals/carry/carry_groups.md#lines
template <int WORDS>
__device__ __forceinline__ const std::uint32_t* leaf_lines(const Operands& o, int side, int bits,
                                                           std::int64_t head) {
  const std::int64_t block = head * o.windows * plan::leaf_block_bytes(bits, WORDS);
  const std::uint8_t* const lines =
      o.workspace + o.prepared[side] + block + plan::leaf_lines_at(bits, WORDS);
  return reinterpret_cast<const std::uint32_t*>(lines);
}

//: A SIDE'S LINES A WINDOW ON, in words from the window before's -- see docs/internals/carry/carry_groups.md#lines
template <int WORDS>
__device__ __forceinline__ std::int64_t leaf_step(int bits) {
  return plan::leaf_block_bytes(bits, WORDS) / static_cast<int>(sizeof(std::uint32_t));
}

}  // namespace detail
}  // namespace carry
}  // namespace fsla
