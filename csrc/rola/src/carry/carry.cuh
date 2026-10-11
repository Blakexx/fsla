// csrc/rola/src/carry/carry.cuh -- THE CARRY's constants and operand block: a CTA a box of leaves, window by window
// its state read out to the window's readers and the window's writes folded in. see docs/internals/carry/carry.md
#pragma once

#include <cstdint>

namespace rola::carry {

//: THE CTA, THE WINDOW, THE PASS, THE STATE AND A SLICE. -- see docs/internals/carry/carry.md#constants
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kWindow = 512;
constexpr int kPassSlots = 128;
constexpr int kListAt = 8;
constexpr int kListCap = kWindow;
constexpr int kMaxLevels = 4;
constexpr int kStateValues = kThreads * 128;
constexpr int kSliceBytes = 64;

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

//: ONE SIDE'S LISTS, the facts pass's at its list level. -- see docs/internals/carry/carry.md#lists
struct SideLists {
  const uint16_t* union_tok;
  const uint16_t* union_count;
  const uint8_t* entries;
  const uint16_t* at;
  int width, groups, entry_bytes;
};

//: THE OPERAND BLOCK, a grid constant. -- see docs/internals/carry/carry.md#params
struct CarryParams {
  int D, H, L, windows, dv, channels;
  int widths[kMaxLevels], extents[kMaxLevels], stride[kMaxLevels];
  Side write, read;
  Plane values;
  SideLists write_lists, read_lists;
  const int8_t* unit;
  uint32_t* num;
  float* check;
  long long* ledger;
};

//: THE PHASE LEDGER's phases. -- see docs/internals/carry/carry_kernel.md#phases
enum CarryPhase : int {
  kPhaseLists,
  kPhaseUnits,
  kPhasePublish,
  kPhaseReadout,
  kPhaseDrain,
  kPhaseFold,
  kCarryPhases
};

}  // namespace rola::carry
