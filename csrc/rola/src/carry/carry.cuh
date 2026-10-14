// csrc/rola/src/carry/carry.cuh -- THE CARRY's constants and operand block: a CTA a box of leaves, window by window
// its state read out to the window's readers and the window's writes folded in. see docs/internals/carry/carry.md
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace rola::carry {

//: THE CTA, THE WINDOW AND THE LISTS. -- see docs/internals/carry/carry.md#constants
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kWindow = 512;
//: A LIST'S BOUNDARIES: its entries before every 64th union slot, the last its length.
constexpr int kListBound = 64;
constexpr int kListAt = kWindow / kListBound;
constexpr int kListCap = kWindow;
constexpr int kMaxLevels = 4;
//: THE BOX, every card's: 32 lists (the write side's list level) x 32 digits (the read side's) x 32 channels.
constexpr int kBoxLists = 32, kBoxDigits = 32, kBoxChannels = 32;
constexpr int kStateValues = kBoxLists * kBoxDigits * kBoxChannels;
//: A SLICE: a token's operand row over the box's 32 digits of one level, f16.
constexpr int kSliceBytes = 64;

//: THE LISTS' ADDRESSES, the facts pass's to emit and the carry's to read: a gap slot's token, the panel's rows past
//: their dummy rows, a pool row's swizzled chunk. -- see docs/internals/carry/carry.md#lists
constexpr uint16_t kGap = 0x8000;
constexpr int kPanelRow = kBoxChannels * 2;
constexpr int kPadBytes = 8 * kPanelRow;

__host__ __device__ __forceinline__ constexpr uint32_t panel_row(uint32_t s) {
  return 128u * (s >> 1) + 32u * ((2u * (s & 1u)) ^ ((s >> 1) & 3u));
}

__host__ __device__ __forceinline__ constexpr uint32_t pool_chunk(int row, uint32_t s, uint32_t k) {
  return row == 128 ? k ^ (s & 7u) : (k & ~3u) | ((k ^ (s >> 1)) & 3u);
}

//: A PLANE where the bundle holds it: a token's row at `base + b sb + h sh + t st` bytes.
struct Plane {
  const uint8_t* base;
  int64_t sb, st, sh;
};

//: ONE SIDE OF THE CARVE: its list level (-1 for none) and operand levels. -- see docs/internals/carry/carry.md#sides
struct Side {
  int list, operands;
  int operand[kMaxLevels], elem[kMaxLevels];
  Plane rows[kMaxLevels];
};

//: ONE SIDE'S LISTS, the facts pass's in their final form. -- see docs/internals/carry/carry.md#lists
struct SideLists {
  const uint16_t* union_tok;
  const int32_t* union_count;
  const uint2* entries;
  const uint16_t* at;
  int width, groups;
};

//: THE OPERAND BLOCK, a grid constant. -- see docs/internals/carry/carry.md#params
struct CarryParams {
  int D, H, L, windows, dv, channels;
  int widths[kMaxLevels], extents[kMaxLevels], stride[kMaxLevels];
  Side write, read;
  Plane values;
  SideLists write_lists, read_lists;
  const int16_t* write_exp;
  const int8_t* unit;
  int readout_f16;  // the readout's MMAs accumulate in fp16 (1) or fp32 (0): carry_api.md#carry-readout-accumulate
  int prefer;  // the readout candidate the plans take first where it fits, -1 none: carry_api.md#carry-forward
  uint32_t* num;
  float* check;
  float* probe;
  long long* ledger;
};

//: THE PHASE LEDGER's phases. -- see docs/internals/carry/carry_kernel.md#phases
enum CarryPhase : int {
  kPhaseLanded,
  kPhaseMaxima,
  kPhaseStore,
  kPhaseReadout,
  kPhaseDrain,
  kPhaseFill,
  kPhaseFold,
  kPhasePlan,
  kCarryPhases
};

}  // namespace rola::carry
