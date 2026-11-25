// THE MASK-PREPARATION PASS, its device code -- docs/internals/mask_prepare/mask_prepare.md
// What the library's kernel and the lab's timed kernel both compile: the launch's facts, the walk and a CTA's work.
#pragma once

#include <cstdint>

#include "common/structure_switch.cuh"
#include "plan/levels.h"
#include "plan/mask_prepare_layout.h"

namespace fsla {
namespace mask_prepare {
namespace detail {

constexpr int kThreads = 1024;
constexpr int kSides = plan::kLevelSides;
constexpr int kGroupSizes = plan::kGroupSizes;

//: THE LAUNCH'S FACTS of each level-side `s`: its mask (null where it has none), where its blocks start in the workspace, its first CTA (`first_cta[kSides]` the grid), its width and its batch-heads
struct Args {
  const std::uint32_t* mask[kSides];
  std::int64_t out[kSides];
  int first_cta[kSides + 1];
  int width_bits[kSides];
  int heads[kSides];
  int words;
  int windows;
  int levels;
};

//: the index of a group's bits among `plan::kGroupSizeBits`
__host__ __device__ constexpr int group_index(int bits) { return bits < 2 ? 0 : bits - 1; }

constexpr bool group_index_inverts_the_sizes() {
  for (int i = 0; i < kGroupSizes; ++i) {
    if (group_index(plan::group_size_bits(i)) != i) return false;
  }
  return true;
}

static_assert(group_index_inverts_the_sizes(), "group_index inverts group_size_bits");

//: A THREAD'S DIGITS OF ONE WORD AS LOADED: 16 bytes a load where it holds 4 digits or more
template <int U>
struct Raw {
  uint4 line[U / 4];
};

template <>
struct Raw<1> {
  std::uint32_t word;
};

template <int U>
__device__ __forceinline__ void load(Raw<U>& raw, const std::uint32_t* at) {
  if constexpr (U == 1) {
    raw.word = at[0];
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

//: A WORD FINISHED -- see docs/internals/mask_prepare/mask_prepare.md#finish
template <int U, int LANES>
__device__ __forceinline__ void finish(const Raw<U>& raw, std::uint32_t keep, bool store,
                                       std::uint32_t* word, std::uint32_t& live) {
  std::uint32_t w = word_of<U>(raw) & keep;
#pragma unroll
  for (int d = 1; d < LANES; d <<= 1) w |= __shfl_xor_sync(0xffffffffu, w, d);
  if constexpr (U * LANES > 1) {
    if (store) word[0] = w;
  }
  live += static_cast<std::uint32_t>(__popc(w));
}

//: A LEVEL-SIDE AT A GROUP OF `1 << GB` DIGITS -- see docs/internals/mask_prepare/mask_prepare.md#the-walk
template <int GB, int WORDS>
__device__ __forceinline__ void prepare(const Args& a, std::uint8_t* workspace, int s, int cta) {
  constexpr int kUnitBits = GB < 4 ? GB : 4, kUnit = 1 << kUnitBits, kLanes = 1 << (GB - kUnitBits);
  const int wb = a.width_bits[s], ub = wb - kUnitBits, gb = wb - GB;
  const std::int64_t windows = std::int64_t{a.heads[s]} * a.windows;
  const std::int64_t unit = std::int64_t{cta} * kThreads + threadIdx.x;
  if ((std::int64_t{cta} * kThreads) >> ub >= windows) return;

  //: a thread past the level-side's last window walks its first and stores nothing; of a group's lanes the first stores
  const std::int64_t at_window = (unit >> ub) < windows ? unit >> ub : 0;
  const int u = static_cast<int>(unit) & ((1 << ub) - 1), group = u >> (GB - kUnitBits);
  const bool store = (unit >> ub) < windows && (u & (kLanes - 1)) == 0;
  const int window = static_cast<int>(at_window % a.windows);
  const std::int64_t first_word = (at_window / a.windows) * a.words + std::int64_t{window} * WORDS;
  const int have = a.words - window * WORDS < WORDS ? a.words - window * WORDS : WORDS;
  const std::int64_t stride = std::int64_t{1} << wb;
  const std::uint32_t* const at = a.mask[s] + (first_word << wb) + (u << kUnitBits);
  const std::int64_t block_bytes = plan::prepared_block_bytes(wb, WORDS);
  std::uint8_t* const block = workspace + a.out[s] + at_window * block_bytes;
  std::uint16_t* const totals = reinterpret_cast<std::uint16_t*>(block);

  const int groups = 1 << gb;
  std::uint8_t* const words = block + plan::prepared_words_at(groups);
  std::uint32_t* word = reinterpret_cast<std::uint32_t*>(words) + group;

  //: THE WALK'S LOADS HELD, issued before any is read: 8 words' at one digit a thread or at four, 2 at more
  constexpr int kHeld = kUnit < 8 ? 8 : kUnit < 16 ? 4 : 2;
  static_assert(WORDS > 0 && WORDS % kHeld == 0 && WORDS * 32 <= 65536, "no window of trips");
  Raw<kUnit> held[kHeld];
  const std::uint32_t* row = at;
#pragma unroll
  for (int i = 0; i < kHeld; ++i) load<kUnit>(held[i], i < have ? row + i * stride : at);
  std::uint32_t live = 0;
#pragma unroll 1
  for (int j = 0; j < WORDS; j += kHeld) {
#pragma unroll
    for (int i = 0; i < kHeld; ++i) {
      //: a word finished, and the load `kHeld` words on issued into the registers it left; a word past the mask's last reads as zero
      const std::uint32_t keep = j + i < have ? ~0u : 0u;
      finish<kUnit, kLanes>(held[i], keep, store, word + i * groups, live);
      load<kUnit>(held[i], j + i + kHeld < have ? row + (i + kHeld) * stride : at);
    }
    row += kHeld * stride;
    word += kHeld * groups;
  }
  if (store) totals[group] = static_cast<std::uint16_t>(live);
}

//: A CTA OF THE GRID, `cta` its index -- see docs/internals/mask_prepare/mask_prepare.md#the-kernel
template <int WORDS>
__device__ __forceinline__ void mask_prepare_cta(const Args& a, const std::uint8_t* density,
                                                 std::uint8_t* workspace, int cta) {
  int s = 0;
#pragma unroll
  for (int i = 1; i < kSides; ++i) s += cta >= a.first_cta[i] ? 1 : 0;
  const int side = s / plan::kCarveMaxLevels, level = s % plan::kCarveMaxLevels;
  //: a byte `b` is `b + 1` 256ths; a group that is the level's size is every token: nothing is prepared, no reader asks
  const int dense = static_cast<int>(density[side * a.levels + level]) + 1;
  const int bits = plan::prefill_group_bits(dense, a.width_bits[s]);
  if (bits == a.width_bits[s]) return;
  uniform_switch<kGroupSizes>(group_index(bits), [&](auto member) {
    constexpr int kBits = plan::kGroupSizeBits[decltype(member)::value];
    prepare<kBits, WORDS>(a, workspace, s, cta - a.first_cta[s]);
  });
}

//: ONE GRID OVER THE CALL'S MASKED LEVEL-SIDES
template <class Bt>
__global__ __launch_bounds__(kThreads, 1) void mask_prepare_kernel(const __grid_constant__ Args a,
                                                                   const std::uint8_t* density,
                                                                   std::uint8_t* workspace) {
  //: the window is the card's: a constant of the build
  constexpr int kWords = plan::window_of_cc(Bt::build.cc) / 32;
  mask_prepare_cta<kWords>(a, density, workspace, static_cast<int>(blockIdx.x));
}

//: the grid a level-side needs at a group of one digit, which is what the host can know: a thread a (window, digit)
inline std::int64_t ctas_of(std::int64_t heads, std::int64_t windows, int width_bits) {
  return ((heads * windows << width_bits) + kThreads - 1) / kThreads;
}

}  // namespace detail
}  // namespace mask_prepare
}  // namespace fsla
