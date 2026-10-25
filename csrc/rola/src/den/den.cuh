// csrc/rola/src/den/den.cuh -- THE DEN PASS's constants, operand block and shared tenants: each token's window-aligned
// denominator and readout bound -- its read leaf product against the write mass of the windows before -- computed
// beside the carry by CTAs of 64 level-S rows x 64 level-R digits. see docs/internals/den/den.md
#pragma once

#include <cuda_fp16.h>

#include <cstdint>

#include "carry/carry.cuh"
#include "common/phase_clock.cuh"

namespace rola::den {

using carry::kGap;
using carry::kListAt;
using carry::kListBound;
using carry::kListCap;
using carry::kPreRows;
using carry::kWindow;
using carry::Plane;
using carry::SideLists;

//: THE CTA'S SHAPE and the passes' grains. -- see docs/internals/den/den.md#constants
constexpr int kWarps = 16, kThreads = kWarps * 32;
constexpr int kRows = 64, kSpan = 64, kBand = kRows / kWarps, kDigits = kSpan / kWarps;
constexpr int kChannels = 2, kSteps = kSpan / 16, kTileRows = 16, kRound = 128, kChunk = 64;
constexpr int kSlabBytes = kTileRows * kSpan * 2, kMaxHeads = 4;
constexpr int kCoefTop = 4, kUnitMin = kCoefTop - 127;

//: THE PHASE LEDGER's phases. -- see docs/internals/den/den_kernel.md#phases
enum DenPhase : int {
  kPhaseStage,
  kPhasePublish,
  kPhaseDen,
  kPhaseDrain,
  kPhaseWrites,
  kPhaseWait,
  kDenPhases
};

//: THE OPERAND BLOCK. -- see docs/internals/den/den.md#params
struct DenParams {
  Plane read_s, read_r, write_s, write_r;
  SideLists read_lists, write_lists;
  const int16_t* list_exp;
  const __half* gains16;
  const int16_t* gains_exp;
  float* den;
  float* num;
  long long* ledger;
  int L, windows, heads, write_heads, width_s, width_r, tiles_s, level_s;
};

//: THE SHARED TENANTS, byte offsets. -- see docs/internals/den/den.md#tenants
struct Tenants {
  static constexpr int kRegion = 0, kRegionBytes = kWarps * 2 * kSlabBytes;
  static constexpr int kMassBytes = kChannels * kRows * kSpan * 4;
  static constexpr int kChunkWs = kRegion + kMassBytes,
                       kChunkWr = kChunkWs + 2 * kChunk * kRows * 2;
  static constexpr int kScratch = kRegion + kRegionBytes;
  static constexpr int kReadsBytes = kWarps * kRound * 4;
  static constexpr int kSlotTokenBytes = kWarps * (kChunk + 16) * 2;
  static constexpr int kSlotCBytes = kWarps * kChannels * (kChunk + 16) * 4;
  static constexpr int kScratchBytes =
      kReadsBytes > kSlotTokenBytes + kSlotCBytes ? kReadsBytes : kSlotTokenBytes + kSlotCBytes;
  static constexpr int kAcc = kScratch + kScratchBytes,
                       kAccBytes = kChannels * kMaxHeads * kWindow * 4;
  static constexpr int kUnion = kAcc + kAccBytes, kUnionBytes = 2 * (kSpan / 32) * kWindow * 2;
  static constexpr int kExps = kUnion + kUnionBytes, kExpsBytes = 2 * kSpan * kChannels * 2;
  static constexpr int kGains = kExps + kExpsBytes, kGainsBytes = 2 * kChunk * kChannels * 2;
  static constexpr int kHeld = kGains + kGainsBytes, kHeldBytes = kSpan * kChannels * 2;
  static constexpr int kClock = kHeld + kHeldBytes,
                       kClockBytes = kWarps * PhaseClock<kDenPhases, true>::kRow * 8;
  static constexpr int kBytes = kClock + kClockBytes;
};

static_assert(Tenants::kMassBytes + kWarps * kSlabBytes <= Tenants::kRegionBytes,
              "the dense form's one slab a warp fits beside the mass");
static_assert(Tenants::kChunkWr + 2 * kChunk * kSpan * 2
                  <= Tenants::kRegion + Tenants::kRegionBytes,
              "the dense form's chunk buffers fit the region past the mass");
static_assert(Tenants::kBytes <= 101376, "the tenants fit sm_86's opt-in shared memory");

}  // namespace rola::den
