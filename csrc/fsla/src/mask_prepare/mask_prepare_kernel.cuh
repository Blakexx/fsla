// THE MASK-PREPARATION PASS, its device code -- docs/internals/mask_prepare/mask_prepare.md
// What the library's kernel and the lab's timed kernel both compile: the launch's facts, a level's group words formed in shared memory, and a CTA's leaf groups' lines.
#pragma once

#include <cstdint>

#include "common/atoms/warp.cuh"
#include "common/smem_layout.cuh"
#include "common/structure_switch.cuh"
#include "plan/leaf_groups.h"
#include "plan/levels.h"
#include "plan/mask_prepare_layout.h"

namespace fsla {
namespace mask_prepare {
namespace detail {

constexpr int kThreads = 1024;
constexpr int kSides = plan::kCarveSides;
constexpr int kLevels = plan::kCarveMaxLevels;
constexpr int kLevelSides = plan::kLevelSides;
//: FIXED by the walk's loads: a thread holds a group's digits, 16 at most, as log2
constexpr int kUnitBitsMost = 4;
//: A LEVEL'S THREADS: a quarter of the CTA's, a thread a digit of the widest level at a group of one digit
constexpr int kLevelThreads = kThreads / kLevels;
//: the groups a level's formed words hold a mask word: the widest level's digits
constexpr int kFormedGroups = 1 << plan::kCarveMaxWidthBits;
static_assert(kLevelThreads == kFormedGroups, "a thread a digit of the widest level");
//: the group sizes a level is formed at, by log2: those smaller than the widest level (a group of its level's size is the level whole, and nothing is formed for it)
constexpr int kPrepared = plan::kCarveMaxWidthBits;

//: THE SIDE'S LEAF GROUPS AS HANDED, a CTA's first warp to the rest: their count's log2, then a value a level of the side's group and of each field of a leaf group's index (`plan::LeafGroups`)
//: FIXED by this record's own order: the count first
constexpr int kHandedBits = 0;
//: FIXED by this record's own order: the groups after the count
constexpr int kHandedGroup = 1;
constexpr int kHandedInner = kHandedGroup + kLevels;
constexpr int kHandedInnerAt = kHandedInner + kLevels;
constexpr int kHandedOuter = kHandedInnerAt + kLevels;
constexpr int kHandedOuterAt = kHandedOuter + kLevels;
constexpr int kHandedCount = kHandedOuterAt + kLevels;

//: THE PASS'S SHARED MEMORY: the handed leaf groups, and each level's group words of the CTA's window, `formed[(level * words + word) * kFormedGroups + group]` -- see docs/internals/mask_prepare/mask_prepare.md#smem
struct Handed;
struct Formed;
template <class Bt>
constexpr int kWindowWords = plan::window_of_cc(Bt::build.cc) / 32;
template <class Bt>
using Smem = SmemLayout<Bt, 1, Fixed<Handed, int, kHandedCount>,
                        Fixed<Formed, std::uint32_t, kLevels * kFormedGroups * kWindowWords<Bt>>>;

//: THE LAUNCH'S FACTS: each level-side's mask (`side * 4 + level`, the read side first; null where it has none) and each side's masked levels (`masked`, a bit a level), each side's place in the workspace,
//: its first CTA (`first_cta[kSides]` the grid) and its batch-heads; every level's width, masked or not; the call's tokens a head, its mask words and its windows
struct Args {
  const std::uint32_t* mask[kLevelSides];
  int masked[kSides];
  std::int64_t out[kSides];
  std::int64_t tokens;
  int first_cta[kSides + 1];
  int heads[kSides];
  int width_bits[kLevels];
  int words;
  int windows;
  int levels;
};

//: A THREAD'S DIGITS OF ONE WORD AS LOADED: 16 bytes a load where it holds 4 digits or more
template <int U>
struct Raw {
  uint4 line[U / 4];
};

template <>
struct Raw<1> {
  std::uint32_t word;
};

template <>
struct Raw<2> {
  uint2 pair;
};

template <int U>
__device__ __forceinline__ void load(Raw<U>& raw, const std::uint32_t* at) {
  if constexpr (U == 1) {
    raw.word = at[0];
  } else if constexpr (U == 2) {
    raw.pair = *reinterpret_cast<const uint2*>(at);
  } else {
    constexpr int kLines = U / 4;
    const uint4* const lines = reinterpret_cast<const uint4*>(at);
#pragma unroll
    for (int i = 0; i < kLines; ++i) raw.line[i] = lines[i];
  }
}

//: a token's bit set where it is live on any of the thread's digits
template <int U>
__device__ __forceinline__ std::uint32_t word_of(const Raw<U>& raw) {
  if constexpr (U == 1) {
    return raw.word;
  } else if constexpr (U == 2) {
    return raw.pair.x | raw.pair.y;
  } else {
    constexpr int kLines = U / 4;
    std::uint32_t w = 0;
#pragma unroll
    for (int i = 0; i < kLines; ++i) {
      w |= raw.line[i].x | raw.line[i].y | raw.line[i].z | raw.line[i].w;
    }
    return w;
  }
}

//: A GROUP'S WORD FINISHED -- see docs/internals/mask_prepare/mask_prepare.md#finish
template <int U, int LANES>
__device__ __forceinline__ void finish(const Raw<U>& raw, std::uint32_t keep, bool store,
                                       std::uint32_t* word) {
  std::uint32_t w = word_of<U>(raw) & keep;
#pragma unroll
  for (int d = 1; d < LANES; d <<= 1) w |= __shfl_xor_sync(0xffffffffu, w, d);
  if (store) word[0] = w;
}

//: A LEVEL-SIDE'S GROUP WORDS OF ONE WINDOW, FORMED IN SHARED MEMORY at a group of `1 << GB` digits: a thread a unit of the level's digits, the warp's first `first` (a value every lane holds) -- see docs/internals/mask_prepare/mask_prepare.md#the-walk
template <int GB, int WORDS>
__device__ __forceinline__ void form(const Args& a, int s, std::int64_t head, int window, int first,
                                     std::uint32_t* formed) {
  constexpr int kUnitBits = GB < kUnitBitsMost ? GB : kUnitBitsMost, kUnit = 1 << kUnitBits;
  constexpr int kLanes = 1 << (GB - kUnitBits);
  //: FIXED by the card: a warp is 32 threads
  constexpr int kWarp = 32;
  const int wb = a.width_bits[s % kLevels], units = 1 << (wb - kUnitBits);
  //: a warp past the level's units forms nothing; a lane past them in the level's last warp walks the first unit and stores nothing; of a group's lanes the first stores
  if (first >= units) return;
  const int u = first + (static_cast<int>(threadIdx.x) & (kWarp - 1));
  const bool mine = u < units;
  const int me = mine ? u : 0, group = me >> (GB - kUnitBits);
  const bool store = mine && (me & (kLanes - 1)) == 0;
  const std::int64_t first_word = head * a.words + std::int64_t{window} * WORDS;
  const int have = a.words - window * WORDS < WORDS ? a.words - window * WORDS : WORDS;
  const std::int64_t stride = std::int64_t{1} << wb;
  const std::uint32_t* const at = a.mask[s] + (first_word << wb) + (me << kUnitBits);

  //: THE WALK'S LOADS HELD, issued before any is read: 8 words' at up to four digits a thread, 4 at eight, 2 at more
  constexpr int kHeld = kUnit < 8 ? 8 : kUnit < 16 ? 4 : 2;
  static_assert(WORDS > 0 && WORDS % kHeld == 0 && WORDS * 32 < 65536, "no window of trips");
  Raw<kUnit> held[kHeld];
  const std::uint32_t* row = at;
  std::uint32_t* word = formed + group;
#pragma unroll
  for (int i = 0; i < kHeld; ++i) load<kUnit>(held[i], i < have ? row + i * stride : at);
#pragma unroll 1
  for (int j = 0; j < WORDS; j += kHeld) {
#pragma unroll
    for (int i = 0; i < kHeld; ++i) {
      //: a word finished, and the load `kHeld` words on issued into the registers it left; a word past the mask's last reads as zero
      const std::uint32_t keep = j + i < have ? ~0u : 0u;
      finish<kUnit, kLanes>(held[i], keep, store, word + i * kFormedGroups);
      load<kUnit>(held[i], j + i + kHeld < have ? row + (i + kHeld) * stride : at);
    }
    row += kHeld * stride;
    word += kHeld * kFormedGroups;
  }
}

//: THE SIDE'S LEAF GROUPS, from the call's carve, handed in shared words; a density byte `b` is `b + 1` 256ths. ITS OWN FUNCTION, not inlined: its values are allocated apart from the walk's -- see docs/internals/mask_prepare/mask_prepare.md#the-kernel
__device__ __noinline__ void hand_leaf_groups(const Args& a, const std::uint8_t* density, int side,
                                              int* handed) {
  //: the call hands one read head's densities
  int width[kLevels] = {}, write[kLevels] = {}, read[1][kLevels] = {};
#pragma unroll
  for (int l = 0; l < kLevels; ++l) {
    width[l] = a.width_bits[l];
    if (l < a.levels) {
      //: A LEVEL-SIDE WITH NO MASK IS WALKED WHOLE, whatever its density byte: the mask's absence is the dense flag, and the workspace is sized by it
      read[0][l] = plan::carve_density(a.masked[0], l, density[l]);
      write[l] = plan::carve_density(a.masked[plan::kWriteSide], l, density[a.levels + l]);
    }
  }
  const plan::PrefillCarve c =
      plan::prefill_carve<plan::PrefillPolicy>(a.levels, width, write, read, 1);
  const plan::LeafGroups g = plan::leaf_groups(c, side == 0, 0);
  if (threadIdx.x != 0) return;
  handed[kHandedBits] = g.bits;
#pragma unroll
  for (int l = 0; l < kLevels; ++l) {
    handed[kHandedGroup + l] = side == 0 ? c.read_group_bits[0][l] : c.write_group_bits[l];
    handed[kHandedInner + l] = g.inner[l];
    handed[kHandedInnerAt + l] = g.inner_at[l];
    handed[kHandedOuter + l] = g.outer[l];
    handed[kHandedOuterAt + l] = g.outer_at[l];
  }
}

//: A LEAF GROUP'S FIELD OF ONE LISTED LEVEL, as a thread uses it: where the level's formed words start, and the two fields of a leaf group's index that are its group of the level
struct Used {
  const std::uint32_t* formed;
  int inner, inner_at, outer, outer_at;
};

//: THE TRIPS OF A SIDE THAT LISTS `USED` LEVELS: a thread a leaf group a trip, a shared load a listed level a mask word -- see docs/internals/mask_prepare/mask_prepare.md#lines
template <int WORDS, int USED>
__device__ __forceinline__ void lay_trips(const Used* used, int count, int warp, int cut,
                                          std::uint32_t last, std::uint16_t* counts, uint4* lines) {
  //: FIXED by the card: a warp is 32 threads
  constexpr int kWarp = 32;
  //: A WARP PAST THE SIDE'S LAST LEAF GROUP LAYS NOTHING: `warp`, its index, is a value every lane holds, so a trip is the warps that hold a leaf group and no others
#pragma unroll 1
  for (int base = 0; base + warp * kWarp < count; base += kThreads) {
    const int leaf = base + static_cast<int>(threadIdx.x);
    //: a lane past the side's last leaf group, in its last warp, forms the first again and stores nothing
    const bool live = leaf < count;
    const int of_leaf = live ? leaf : 0;
    const std::uint32_t* word[USED > 0 ? USED : 1];
#pragma unroll
    for (int k = 0; k < USED; ++k) {
      const int inside = (of_leaf >> used[k].inner_at) & ((1 << used[k].inner) - 1);
      const int above = (of_leaf >> used[k].outer_at) & ((1 << used[k].outer) - 1);
      word[k] = used[k].formed + ((above << used[k].inner) | inside);
    }
    uint4* const line = lines + std::int64_t{of_leaf} * (WORDS / 4);
    std::uint32_t total = 0;
#pragma unroll 1
    for (int j = 0; j < WORDS; j += 4) {
      std::uint32_t v[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        //: THE CALL'S OWN TOKENS of the word: every bit below the call's last word, its low bits there, none past it
        std::uint32_t w = j + k < cut ? ~0u : j + k == cut ? last : 0u;
#pragma unroll
        for (int u = 0; u < USED; ++u) w &= word[u][(j + k) * kFormedGroups];
        v[k] = w;
        total += static_cast<std::uint32_t>(__popc(w));
      }
      if (live) line[j / 4] = make_uint4(v[0], v[1], v[2], v[3]);
    }
    if (live) counts[leaf] = static_cast<std::uint16_t>(total);
  }
}

//: A CTA'S LEAF GROUPS' LINES AND COUNTS -- see docs/internals/mask_prepare/mask_prepare.md#lines
template <int WORDS>
__device__ __forceinline__ void lay_lines(const Args& a, std::uint8_t* workspace, int side,
                                          std::int64_t at, int window, int warp, const int* handed,
                                          const std::uint32_t* formed) {
  static_assert(WORDS % 4 == 0, "a line is 16-byte stores");
  //: FIXED by the card: a warp is 32 threads
  constexpr int kWarp = 32;
  const int bits = handed[kHandedBits], count = 1 << bits;
  if (warp * kWarp >= count) return;
  //: THE LEVELS THE SIDE LISTS, packed to the front: a thread loads a word of each and of no other
  Used used[kLevels] = {};
  int listed = 0;
#pragma unroll
  for (int l = 0; l < kLevels; ++l) {
    const bool lists = l < a.levels && handed[kHandedGroup + l] < a.width_bits[l];
    const int inner = handed[kHandedInner + l], inner_at = handed[kHandedInnerAt + l];
    const int outer = handed[kHandedOuter + l], outer_at = handed[kHandedOuterAt + l];
#pragma unroll
    for (int k = 0; k < kLevels; ++k) {
      const bool here = lists && listed == k;
      used[k].formed = here ? formed + l * (WORDS * kFormedGroups) : used[k].formed;
      used[k].inner = here ? inner : used[k].inner;
      used[k].inner_at = here ? inner_at : used[k].inner_at;
      used[k].outer = here ? outer : used[k].outer;
      used[k].outer_at = here ? outer_at : used[k].outer_at;
    }
    listed += lists ? 1 : 0;
  }
  std::uint8_t* const block = workspace + a.out[side] + at * plan::leaf_block_bytes(bits, WORDS);
  std::uint16_t* const counts = reinterpret_cast<std::uint16_t*>(block);
  uint4* const lines = reinterpret_cast<uint4*>(block + plan::leaf_lines_at(bits, WORDS));
  //: the window's word the call's last token is in, from the window's first: under 0 where the call ended before the window, `WORDS` where it ends after it
  const std::int64_t to_last = (a.tokens >> 5) - std::int64_t{window} * WORDS;
  const int cut = to_last < 0 ? -1 : to_last > WORDS ? WORDS : static_cast<int>(to_last);
  const std::uint32_t last = (1u << static_cast<int>(a.tokens & 31)) - 1u;
  //: the count of listed levels is a value every lane holds: the body compiled for it holds that many loads a word
  uniform_switch<kLevels + 1>(::denseref::ops::lane_broadcast(listed, 0), [&](auto member) {
    lay_trips<WORDS, decltype(member)::value>(used, count, warp, cut, last, counts, lines);
  });
}

//: A CTA OF THE GRID: a side's (batch-head, window), `cta` its index -- see docs/internals/mask_prepare/mask_prepare.md#the-kernel
template <int WORDS>
__device__ __forceinline__ void mask_prepare_cta(const Args& a, const std::uint8_t* density,
                                                 std::uint8_t* workspace, int cta, int* handed,
                                                 std::uint32_t* formed) {
  const int side = cta >= a.first_cta[1] ? 1 : 0;
  const std::int64_t at = cta - a.first_cta[side];
  const std::int64_t head = at / a.windows;
  const int window = static_cast<int>(at % a.windows);
  //: THE LEAF GROUPS ARE THE CARVE'S, derived once a CTA by its first warp and handed to the rest
  //: FIXED by the card: a warp is 32 threads
  constexpr unsigned kWarp = 32;
  if (threadIdx.x < kWarp) hand_leaf_groups(a, density, side, handed);
  __syncthreads();

  //: EACH LISTED LEVEL'S GROUP WORDS, a level its quarter of the CTA's threads; a level walked whole forms nothing.
  //: THE WARP'S PLACE AND ITS LEVEL'S GROUP ARE VALUES EVERY LANE HOLDS (lane 0's, by shuffle): the branches below stand on them, so the walk's shuffles are whole
  constexpr int kLevelWarps = kLevelThreads / static_cast<int>(kWarp);
  const int warp =
      ::denseref::ops::lane_broadcast(static_cast<int>(threadIdx.x) / static_cast<int>(kWarp), 0);
  const int level = warp / kLevelWarps, first = (warp % kLevelWarps) * static_cast<int>(kWarp);
  const int s = side * kLevels + level;
  const int group_bits = ::denseref::ops::lane_broadcast(handed[kHandedGroup + level], 0);
  if (level < a.levels && group_bits < a.width_bits[level]) {
    uniform_switch<kPrepared>(group_bits, [&](auto member) {
      constexpr int kBits = decltype(member)::value;
      form<kBits, WORDS>(a, s, head, window, first, formed + level * (WORDS * kFormedGroups));
    });
  }
  __syncthreads();
  lay_lines<WORDS>(a, workspace, side, at, window, warp, handed, formed);
}

//: ONE GRID OVER THE CALL'S SIDES' (BATCH-HEAD, WINDOW)S
template <class Bt>
__global__ __launch_bounds__(kThreads, 1) void mask_prepare_kernel(const __grid_constant__ Args a,
                                                                   const std::uint8_t* density,
                                                                   std::uint8_t* workspace) {
  //: the window is the card's: a constant of the build
  mask_prepare_cta<kWindowWords<Bt>>(a, density, workspace, static_cast<int>(blockIdx.x),
                                     Smem<Bt>::template get<Handed>(),
                                     Smem<Bt>::template get<Formed>());
}

//: the grid a side needs: a CTA a (batch-head, window)
inline std::int64_t ctas_of(std::int64_t heads, std::int64_t windows) { return heads * windows; }

}  // namespace detail
}  // namespace mask_prepare
}  // namespace fsla
