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
constexpr int kListAt = 8;
constexpr int kListCap = kWindow;
constexpr int kMaxLevels = 4;
//: THE BOX, every card's: 32 lists (the write side's list level) x 32 digits (the read side's) x 32 channels.
constexpr int kBoxLists = 32, kBoxDigits = 32, kBoxChannels = 32;
constexpr int kStateValues = kBoxLists * kBoxDigits * kBoxChannels;
//: A SLICE: a token's operand row over the box's 32 digits of one level, f16.
constexpr int kSliceBytes = 64;
//: A LIST'S ENTRIES STAGED a window ahead: its first 16; the rest are read in place.
constexpr int kHeadEntries = 16;

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
  const uint2* heads;
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
  kCarryPhases
};

}  // namespace rola::carry
