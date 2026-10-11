// csrc/rola/src/carry/plan.cuh -- ONE ARM'S PLAN: the warp tiles an arm's key implies, its registers and its shared
// memory, time-shared by phase on sm_86. see docs/internals/carry/plan.md
#pragma once

#include "carry/carry.cuh"

namespace rola::carry {

//: A POOL ROW'S PITCH, an odd multiple of 32 bytes. -- see docs/internals/carry/plan.md#plan
constexpr int pitch_of(int bytes) { return 32 * (((bytes + 31) / 32) | 1); }

constexpr int max_of(int a, int b) { return a > b ? a : b; }

//: THE BLOCK'S SHARED MEMORY on sm_86, the bound every plan's region fits.
constexpr int kSmemCap = 99 * 1024;

//: THE FOLD'S UNIT, in facts passes: the largest of 4, 2, 1 whose two slots fit `room` bytes at `pitch`, the first
//: clear of the readout's `read` bytes. -- see docs/internals/carry/plan.md#units
constexpr bool unit_fits(int passes, int pitch, int read, int room) {
  return max_of(read, passes * kPassSlots * pitch) + passes * kPassSlots * pitch <= room;
}

constexpr int unit_passes(int pitch, int read, int room) {
  return unit_fits(4, pitch, read, room) ? 4 : unit_fits(2, pitch, read, room) ? 2 : 1;
}

//: THE ARM'S PLAN: each side's warp tile, the registers and the pool rows. -- see docs/internals/carry/plan.md#plan
template <class Arm>
struct Plan {
  static constexpr int kChannels = Arm::channels;
  static constexpr int kLeaves = kStateValues / kChannels;
  static constexpr int kWarpLeaves = kLeaves / kWarps;
  static constexpr int kFoldRows = Arm::fold_rows, kFoldLists = kWarpLeaves / kFoldRows;
  static constexpr int kReadRows = Arm::read_rows, kReadLists = kWarpLeaves / kReadRows;
  static constexpr int kFoldFactors = Arm::fold_factors, kReadFactors = Arm::read_factors;
  static constexpr bool kPublishRegs = Arm::publish_regs != 0;
  static constexpr int kStateRegs = kWarpLeaves * kChannels / 32;
  static constexpr int kReadRegs = kStateRegs / 2;
  static constexpr int kWriteLists = kWarps * kFoldLists, kReadBoxLists = kWarps * kReadLists;
  static_assert(kFoldLists * kFoldRows == kWarpLeaves && kReadLists * kReadRows == kWarpLeaves,
                "a warp's tile is its share of the box");
  static_assert(kFoldRows >= 16 && kReadRows >= 16,
                "a warp's operand rows are an MMA tile at least");

  static constexpr int kValuesAt = kSliceBytes * kFoldFactors;
  static constexpr int kWritePitch = pitch_of(kValuesAt + 2 * kChannels);
  static constexpr int kReadPitch = pitch_of(kSliceBytes * kReadFactors);
  static constexpr int kPanelWords = kChannels / 2;
  static constexpr int kWriteSlots = 2, kReadSlots = 2;
  static constexpr int kFoldMTiles = kFoldRows / 16;

  //: THE PERSISTENT TENANTS, then ONE PHASE REGION the publish, the readout and the fold take in turn.
  static constexpr int kUnion = 0;
  static constexpr int kCount = kUnion + 2 * kWindow * 2;
  static constexpr int kWriteAt = kCount + 16;
  static constexpr int kReadAt = kWriteAt + kWriteLists * kListAt * 2;
  static constexpr int kWriteExp = kReadAt + kReadBoxLists * kListAt * 2;
  static constexpr int kUnits = kWriteExp + kWriteLists * 4;
  static constexpr int kClock = kUnits + kWindow;
  static constexpr int kBook = kClock + kWarps * (kCarryPhases + 1) * 8;
  static constexpr int kBars = kBook + kWriteLists * 8;
  static constexpr int kPhase = (kBars + kWriteSlots * 8 + 127) / 128 * 128;

  //: THE PHASE REGION: the publish; the readout's pool and panel; the fold's two unit slots, slot 0 clear of the
  //: readout's bytes (a window's first unit fills under the readout), slot 1 over them.
  static constexpr int kPublishRounds = 2;
  static constexpr int kPublishBytes = kPublishRegs ? 0 : kLeaves * kChannels * 2 / kPublishRounds;
  static constexpr int kReadPool = 0;
  static constexpr int kPanel = kReadPool + kReadSlots * kPassSlots * kReadPitch;
  static constexpr int kReadBytes = kPanel + kPassSlots * kPanelWords * 4;
  static constexpr int kFoldUnit = unit_passes(kWritePitch, kReadBytes, kSmemCap - kPhase);
  static constexpr int kUnitSlots = kFoldUnit * kPassSlots;
  static constexpr int kUnitBytes = kUnitSlots * kWritePitch;
  static constexpr int kWriteSlot0 = max_of(kReadBytes, kUnitBytes);
  static constexpr int kWriteBytes = kWriteSlot0 + kUnitBytes;
  static constexpr int kSmemBytes = kPhase + max_of(kPublishBytes, max_of(kReadBytes, kWriteBytes));
  static_assert(kSmemBytes <= kSmemCap, "the block's shared memory fits sm_86's 99 KB");
  static_assert(kPublishBytes <= kWriteSlot0,
                "a publish round clears write slot 0, which fills under it");
};

}  // namespace rola::carry
