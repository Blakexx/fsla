// csrc/rola/src/carry/plan.cuh -- ONE ARM'S PLAN on the compiling card: the box's warp tiles, its registers, its pool
// rows, and its shared memory -- the persistent tenants, then the planned AREA each window's plan places its buffers
// in (`schedule.cuh`). see docs/internals/carry/plan.md
#pragma once

#include "carry/carry.cuh"
#include "common/arch_caps.cuh"

namespace rola::carry {

namespace arch = ::denseref::arch;

constexpr int align_of(int x, int q) { return (x + q - 1) / q * q; }

//: THE BARRIERS, every card's: the staging and the maxima a window, two image slots, two read buffers and two fold
//: buffers (each landed and freed; a read buffer also read), and the readout's release. -- see plan.md#barriers
enum CarryBarrier : int {
  kBarStaged,
  kBarPosted,
  kBarStored,
  kBarImageFree = kBarStored + 2,
  kBarReadLanded = kBarImageFree + 2,
  kBarReadDone = kBarReadLanded + 2,
  kBarReadFree = kBarReadDone + 2,
  kBarFoldLanded = kBarReadFree + 2,
  kBarFoldFree = kBarFoldLanded + 2,
  kBarReleased = kBarFoldFree + 2,
  kBarCount
};

constexpr int kPlanBytes = 48;  // a warp's copy of a window's plan: schedule.md#plan

//: THE PERSISTENT TENANTS' BYTES, every card's and arm's. -- see docs/internals/carry/plan.md#tenants
struct Tenants {
  static constexpr int kUnion = 0;
  static constexpr int kCount = kUnion + 2 * 2 * kWindow * 2;
  static constexpr int kAt = kCount + 16;
  static constexpr int kWriteExp = kAt + 2 * 2 * kBoxLists * kListAt * 2;
  static constexpr int kUnits = kWriteExp + 2 * kBoxLists * 4;
  static constexpr int kBook = kUnits + 2 * kWindow;
  static constexpr int kMaxima = kBook + kBoxLists * 8;
  static constexpr int kExps = kMaxima + kWarps * kBoxDigits * 4;
  static constexpr int kClock = kExps + kBoxDigits * 4;
  static constexpr int kBars = kClock + kWarps * (kCarryPhases + 1) * 8;
  static constexpr int kPlans = kBars + kBarCount * 8;
  static constexpr int kAddr = kPlans + kWarps * 2 * kPlanBytes;
  static constexpr int kAddrBytes = 256;
  static constexpr int kEnd = align_of(kAddr + kAddrBytes, 128);
};

constexpr int kPlaneBytes = kBoxLists * kBoxChannels * 2;
//: THE IMAGE: whole, or a ring of two 8-plane batches. -- see docs/internals/carry/plan.md#image
constexpr int kBatchPlanes = 8, kBatches = kBoxDigits / kBatchPlanes;
constexpr int kImageWhole = kBoxDigits * kPlaneBytes, kImageRing = 2 * kBatchPlanes * kPlaneBytes;

//: THE ARM'S PLAN on card `Cc`: the box's tiles, a side's pool row (its operand slices, then the write side's values),
//: and the area the card's shared memory leaves past the tenants. -- see docs/internals/carry/plan.md#plan
template <class Arm, int Cc = arch::kArch>
struct Plan : Tenants {
  static constexpr arch::Caps kCaps = arch::caps_of(Cc);
  static_assert(Arm::fold_rows == kBoxDigits && Arm::read_rows == kBoxLists
                    && Arm::channels == kBoxChannels,
                "the arm is the box's: a fold list 32 digits, a read digit 32 lists, 32 channels");
  static constexpr int kFoldFactors = Arm::fold_factors, kReadFactors = Arm::read_factors;
  //: a warp's tiles: lists `w + 8 j` to fold, digits `w + 8 i` to read out
  static constexpr int kFoldLists = kBoxLists / kWarps, kReadDigits = kBoxDigits / kWarps;
  static constexpr int kFoldMTiles = kBoxDigits / 16;
  static constexpr int kStateRegs = kFoldLists * kBoxDigits * kBoxChannels / 32;
  static_assert(kFoldMTiles * 2 == kBatches && kReadDigits == kBatches,
                "a batch is half an m-tile of every fold list and one digit of every reader");

  static constexpr int kReadRow = kSliceBytes * kReadFactors;
  static constexpr int kValuesAt = kSliceBytes * kFoldFactors,
                       kWriteRow = kValuesAt + kBoxChannels * 2;

  static constexpr int kSmemBytes = kCaps.smem_per_cta_max;
  static constexpr int kArea = kSmemBytes - kEnd;
};

}  // namespace rola::carry
