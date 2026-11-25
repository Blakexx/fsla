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
  int width_bits[kLevels];
  int levels;
  int words;
  int windows;
};

//: A SIDE'S FIRST LEVEL-SIDE: the write side's are the first `kLevels`, the read side's the next
constexpr int side_at(bool read) { return read ? kLevels : 0; }

//: THE CARVE OF A CALL, from its densities as the mask-preparation pass reads them (a byte `b` is `b + 1` 256ths) -- see docs/internals/carry/carry_groups.md#carve
__device__ __forceinline__ plan::PrefillCarve carve_of(const Operands& o) {
  int write[kLevels] = {}, read[kLevels] = {};
#pragma unroll
  for (int l = 0; l < kLevels; ++l) {
    if (l < o.levels) {
      write[l] = static_cast<int>(o.density[l]) + 1;
      read[l] = static_cast<int>(o.density[o.levels + l]) + 1;
    }
  }
  return plan::prefill_carve(o.levels, o.width_bits, write, read);
}

//: A LANE'S WORD of a group's window on one level-side -- see docs/internals/carry/carry_groups.md#word
template <int WORDS>
__device__ __forceinline__ std::uint32_t group_word(const Operands& o, int s, int group_bits,
                                                    std::int64_t head, int window, int word,
                                                    int group) {
  const int wb = o.width_bits[s % kLevels];
  //: a level-side with no mask is whole: every token of the call is live in its one group
  if (o.mask[s] == nullptr) return ~0u;
  if (window * WORDS + word >= o.words) return 0u;
  if (group_bits == 0) {
    const std::int64_t at = head * o.words + std::int64_t{window} * WORDS + word;
    return o.mask[s][(at << wb) + group];
  }
  const int groups = 1 << (wb - group_bits);
  const std::int64_t block = (head * o.windows + window) * plan::prepared_block_bytes(wb, WORDS);
  const std::uint8_t* const words =
      o.workspace + o.prepared[s] + block + plan::prepared_words_at(groups);
  return reinterpret_cast<const std::uint32_t*>(words)[word * groups + group];
}

}  // namespace detail
}  // namespace carry
}  // namespace fsla
