// THE MASK-PREPARATION PASS, its device code -- docs/internals/mask_prepare/mask_prepare.md
// What the library's kernel and the lab's timed kernel both compile: the launch's facts, the walk and a CTA's work.
#pragma once

#include <cstdint>

#include "common/smem_layout.cuh"
#include "common/structure_switch.cuh"
#include "plan/levels.h"
#include "plan/mask_prepare_layout.h"

namespace fsla {
namespace mask_prepare {
namespace detail {

constexpr int kThreads = 1024;
constexpr int kSides = plan::kLevelSides;
constexpr int kLevels = plan::kCarveMaxLevels;
//: FIXED by the walk's loads: a thread holds a group's digits, 16 at most, as log2
constexpr int kUnitBitsMost = 4;

//: THE PASS'S SHARED MEMORY: one word, the group a CTA's first warp hands to the rest -- see docs/internals/mask_prepare/mask_prepare.md#the-kernel
struct Handed;
template <class Bt>
using Smem = SmemLayout<Bt, 1, One<Handed, int>>;
//: the group sizes the pass prepares, by log2: those smaller than the widest level (a group of its level's size is the level whole, and nothing is prepared for it)
constexpr int kPrepared = plan::kCarveMaxWidthBits;

//: THE LAUNCH'S FACTS of each level-side `s`: its mask (null where it has none), where its blocks start in the workspace, its first CTA (`first_cta[kSides]` the grid), its width and its batch-heads; and every level's width, masked or not
struct Args {
  const std::uint32_t* mask[kSides];
  std::int64_t out[kSides];
  int first_cta[kSides + 1];
  int width_bits[kSides];
  int heads[kSides];
  int level_bits[plan::kCarveMaxLevels];
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

//: whether CTA `cta` of level-side `s` holds no window at a group of `1 << bits` digits: its first thread's window is past the last -- see docs/internals/mask_prepare/mask_prepare.md#the-kernel
__device__ __forceinline__ bool surplus(const Args& a, int s, int bits, int cta) {
  const int unit_bits = bits < kUnitBitsMost ? bits : kUnitBitsMost;
  const std::int64_t first = (std::int64_t{cta} * kThreads) >> (a.width_bits[s] - unit_bits);
  return first >= std::int64_t{a.heads[s]} * a.windows;
}

//: A LEVEL-SIDE AT A GROUP OF `1 << GB` DIGITS -- see docs/internals/mask_prepare/mask_prepare.md#the-walk
template <int GB, int WORDS>
__device__ __forceinline__ void prepare(const Args& a, std::uint8_t* workspace, int s, int cta) {
  constexpr int kUnitBits = GB < kUnitBitsMost ? GB : kUnitBitsMost, kUnit = 1 << kUnitBits;
  constexpr int kLanes = 1 << (GB - kUnitBits);
  const int wb = a.width_bits[s], ub = wb - kUnitBits, gb = wb - GB;
  const std::int64_t windows = std::int64_t{a.heads[s]} * a.windows;
  const std::int64_t unit = std::int64_t{cta} * kThreads + threadIdx.x;
  if (surplus(a, s, GB, cta)) return;

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

  //: THE WALK'S LOADS HELD, issued before any is read: 8 words' at up to four digits a thread, 4 at eight, 2 at more
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

//: A LEVEL-SIDE'S GROUP, as log2, from the call's carve; a density byte `b` is `b + 1` 256ths. ITS OWN FUNCTION, not inlined: its values are allocated apart from the walk's, which holds the kernel's 64 registers -- see docs/internals/mask_prepare/mask_prepare.md#the-kernel
__device__ __noinline__ int carved_group_bits(const Args& a, const std::uint8_t* density, int side,
                                              int level) {
  int width[kLevels] = {}, write[kLevels] = {}, read[kLevels] = {};
#pragma unroll
  for (int l = 0; l < kLevels; ++l) {
    width[l] = a.level_bits[l];
    if (l < a.levels) {
      read[l] = static_cast<int>(density[l]) + 1;
      write[l] = static_cast<int>(density[a.levels + l]) + 1;
    }
  }
  const plan::PrefillCarve c = plan::prefill_carve(a.levels, width, write, read);
  int bits = 0;
  //: each of the carve's arrays is read before the level is compared with anything
#pragma unroll
  for (int l = 0; l < kLevels; ++l) {
    const int of_read = c.read_group_bits[l], of_write = c.write_group_bits[l];
    bits = l == level ? (side == 0 ? of_read : of_write) : bits;
  }
  return bits;
}

//: A CTA OF THE GRID, `cta` its index, `handed` its one shared word -- see docs/internals/mask_prepare/mask_prepare.md#the-kernel
template <int WORDS>
__device__ __forceinline__ void mask_prepare_cta(const Args& a, const std::uint8_t* density,
                                                 std::uint8_t* workspace, int cta, int* handed) {
  int s = 0;
#pragma unroll
  for (int i = 1; i < kSides; ++i) s += cta >= a.first_cta[i] ? 1 : 0;
  const int side = s / kLevels, level = s % kLevels;
  //: THE SIZE FIRST, a byte (`b` is `b + 1` 256ths): a level-side whole at its size is whole, and a CTA surplus at its size is surplus at any group raised from it
  const int dense = static_cast<int>(density[side * a.levels + level]) + 1;
  const int size = plan::prefill_group_bits(dense, a.width_bits[s]);
  if (size == a.width_bits[s] || surplus(a, s, size, cta - a.first_cta[s])) return;
  //: THE GROUP IS THE CARVE'S, derived once a CTA by its first warp and handed to the rest
  //: FIXED by the card: a warp is 32 threads
  constexpr unsigned kWarp = 32;
  if (threadIdx.x < kWarp) {
    const int mine = carved_group_bits(a, density, side, level);
    if (threadIdx.x == 0) handed[0] = mine;
  }
  __syncthreads();
  const int bits = handed[0];
  //: a group that is the level's size is every token: nothing is prepared, no reader asks
  if (bits == a.width_bits[s]) return;
  uniform_switch<kPrepared>(bits, [&](auto member) {
    constexpr int kBits = decltype(member)::value;
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
  mask_prepare_cta<kWords>(a, density, workspace, static_cast<int>(blockIdx.x),
                           Smem<Bt>::template get<Handed>());
}

//: the grid a level-side needs at a group of one digit, which is what the host can know: a thread a (window, digit)
inline std::int64_t ctas_of(std::int64_t heads, std::int64_t windows, int width_bits) {
  return ((heads * windows << width_bits) + kThreads - 1) / kThreads;
}

}  // namespace detail
}  // namespace mask_prepare
}  // namespace fsla
