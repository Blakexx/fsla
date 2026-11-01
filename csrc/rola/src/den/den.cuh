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
using carry::kListBound;
using carry::kPreRows;
using carry::list_at;
using carry::Plane;
using carry::kListAt;
using carry::kListCap;
using carry::SideLists;

//: THE CTA'S SHAPE, its two roles and the passes' grains: 8 write warps of `kDigits` digits and 8 den warps of
//: `kBand`-row bands, each role's MMAs two n-tiles wide. -- see docs/internals/den/den.md#constants
constexpr int kWarps = 16, kThreads = kWarps * 32;
constexpr int kWriteWarps = 8, kDenWarps = kWarps - kWriteWarps, kRoleThreads = kWriteWarps * 32;
constexpr int kRows = 64, kSpan = 64, kBand = kRows / kDenWarps, kDigits = kSpan / kWriteWarps;
constexpr int kChannels = 2, kBandTiles = kBand * kChannels / 8,
              kDigitTiles = kDigits * kChannels / 8;
constexpr int kSteps = kSpan / 16, kTileRows = 16, kRound = 64, kChunk = 64;
//: THE DENSE FORM'S BATCH: four tiles (64 tokens) staged once for every band, their bands' partials reduced together.
constexpr int kBatchTiles = kChunk / kTileRows;
constexpr int kSlabBytes = kTileRows * kSpan * 2, kMaxHeads = 4;
constexpr int kCoefTop = 4, kUnitMin = kCoefTop - 127;
//: A DEN WORD: an entry's place (query head x W + token), its list in the band, its f16 coefficient.
constexpr int kPlaceBits = 13, kListBits = 3;
static_assert(kBand <= 1 << kListBits && kPlaceBits + kListBits <= 16,
              "a word's place and list share its low half");
//: THE NAMED BARRIERS past the CTA's: the publish handed to the den warps, and the den warps' own.
constexpr int kBarPublish = 1, kBarDen = 2, kBarWrite = 3;

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

//: THE SHARED TENANTS, byte offsets, at window `W`; device code's at the compiling card's. The roles' tenants are
//: each their own (the den and the writes of a window run at once). -- see docs/internals/den/den.md#tenants
template <int W>
struct TenantsAt {
  static constexpr int kMass = 0, kMassBytes = kChannels * kRows * kSpan * 4;
  static constexpr int kWriteRegion = kMass + kMassBytes,
                       kWriteRegionBytes = kWriteWarps * 2 * kSlabBytes;
  static constexpr int kChunkWs = kWriteRegion, kChunkWr = kChunkWs + 2 * kChunk * kRows * 2;
  static constexpr int kWords = kWriteRegion + kWriteRegionBytes,
                       kWordsBytes = kDenWarps * kRound * 4;
  static constexpr int kResults = kWords + kWordsBytes,
                       kResultsBytes = kDenWarps * kRound * kChannels * 4;
  static constexpr int kSlotToken = kResults + kResultsBytes,
                       kSlotTokenBytes = kWriteWarps * (kChunk + 16) * 2;
  static constexpr int kSlotC = kSlotToken + kSlotTokenBytes,
                       kSlotCBytes = kWriteWarps * kChannels * (kChunk + 16) * 4;
  static constexpr int kAcc = kSlotC + kSlotCBytes, kAccBytes = kChannels * kMaxHeads * W * 4;
  //: the dense form's bands' partials, two parities of a batch's `[tile][token x channel][band]` (in the list form's
  //: words, results and slots, which the dense form does not use)
  static constexpr int kPartials = kWords,
                       kPartialsBytes = 2 * kBatchTiles * kTileRows * kChannels * kDenWarps * 4;
  static constexpr int kUnion = kAcc + kAccBytes, kUnionBytes = 2 * (kSpan / 32) * W * 2;
  static constexpr int kExps = kUnion + kUnionBytes, kExpsBytes = 2 * kSpan * kChannels * 2;
  static constexpr int kHeld = kExps + kExpsBytes, kHeldBytes = kSpan * kChannels * 2;
  static constexpr int kClock = kHeld + kHeldBytes,
                       kClockBytes = kWarps * PhaseClock<kDenPhases, true>::kRow * 8;
  //: the dense form's window gains, two parities of `[W][2]` f16: past the partials in the list form's scratch where
  //: they fit (W 512), else their own
  static constexpr int kGainsWBytes = 2 * W * kChannels * 2;
  static constexpr bool kGainsOverlay = kPartialsBytes + kGainsWBytes <= kAcc - kWords;
  static constexpr int kGainsW = kGainsOverlay ? kWords + kPartialsBytes : kClock + kClockBytes;
  static constexpr int kBytes = kClock + kClockBytes + (kGainsOverlay ? 0 : kGainsWBytes);
};

static_assert(kDenWarps * 2 * kSlabBytes <= TenantsAt<512>::kMassBytes,
              "the den warps' two slabs a warp fit the mass's region once every band has its B");
static_assert(2 * kChunk * (kRows + kSpan) * 2 <= TenantsAt<512>::kWriteRegionBytes,
              "the dense form's two chunk parities fit the write warps' region");
static_assert(kMaxHeads * 1024 <= 1 << kPlaceBits,
              "a word's place holds every query head's window at W 1,024");
static_assert(TenantsAt<512>::kPartialsBytes <= TenantsAt<512>::kAcc - TenantsAt<512>::kWords,
              "the dense form's partials fit the list form's den and write scratch");
static_assert(2 * kChunk * (kSpan + kRows) * 2 <= TenantsAt<512>::kMassBytes,
              "the dense den's two batches of level-R and level-S read rows fit the mass's region");
static_assert(TenantsAt<512>::kBytes <= 101376, "the tenants fit sm_86's opt-in shared memory");

using Tenants = TenantsAt<kWindow>;

}  // namespace rola::den
