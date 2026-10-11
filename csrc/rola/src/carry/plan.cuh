// csrc/rola/src/carry/plan.cuh -- ONE ARM'S PLAN: the warp tiles an arm's key implies, its registers and its shared
// memory, time-shared by phase on sm_86. see docs/internals/carry/plan.md
#pragma once

#include "carry/carry.cuh"

namespace rola::carry {

//: A POOL ROW'S PITCH, an odd multiple of 32 bytes. -- see docs/internals/carry/plan.md#plan
constexpr int pitch_of(int bytes) { return 32 * (((bytes + 31) / 32) | 1); }

constexpr int max_of(int a, int b) { return a > b ? a : b; }

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
  static constexpr int kWriteSlots = 3, kReadSlots = 2;

  //: THE PERSISTENT TENANTS, then ONE PHASE REGION the publish, the readout and the fold take in turn.
  static constexpr int kUnion = 0;
  static constexpr int kCount = kUnion + 2 * kWindow * 2;
  static constexpr int kWriteAt = kCount + 16;
  static constexpr int kReadAt = kWriteAt + kWriteLists * kListAt * 2;
  static constexpr int kUnits = kReadAt + kReadBoxLists * kListAt * 2;
  static constexpr int kClock = kUnits + kWindow;
  static constexpr int kBars = kClock + kWarps * (kCarryPhases + 1) * 8;
  static constexpr int kPhase = (kBars + kWriteSlots * 8 + 127) / 128 * 128;

  static constexpr int kPublishBytes = kPublishRegs ? 0 : kLeaves * kChannels * 2;
  static constexpr int kReadPool = 0;
  static constexpr int kPanel = kReadPool + kReadSlots * kPassSlots * kReadPitch;
  static constexpr int kReadBytes = kPanel + kPassSlots * kPanelWords * 4;
  static constexpr int kWritePool = 0;
  static constexpr int kWriteBytes = kWritePool + kWriteSlots * kPassSlots * kWritePitch;
  static constexpr int kSmemBytes = kPhase + max_of(kPublishBytes, max_of(kReadBytes, kWriteBytes));
  static_assert(kSmemBytes <= 99 * 1024, "the block's shared memory fits sm_86's 99 KB");
};

}  // namespace rola::carry
