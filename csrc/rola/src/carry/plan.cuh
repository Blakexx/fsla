// csrc/rola/src/carry/plan.cuh -- ONE ARM'S PLAN on the compiling card: its tiles, registers, pool rows and shared
// memory (the tenants, the AREA, the read tables). see docs/internals/carry/plan.md
#pragma once

#include "carry/carry.cuh"
#include "common/arch_caps.cuh"

namespace rola::carry {

namespace arch = ::denseref::arch;

constexpr int align_of(int x, int q) { return (x + q - 1) / q * q; }

//: THE ASYNC COPIES' BARRIERS: the staging, each slot's landed. -- see docs/internals/carry/plan.md#barriers
enum CarryBarrier : int { kBarStaged, kBarSlot, kBarCount = kBarSlot + 2 };

//: THE FIXED TENANTS' BYTES, every card's, arm's and call's. -- see docs/internals/carry/plan.md#tenants
struct Tenants {
  static constexpr int kCount = 0;
  static constexpr int kUnionW = kCount + 16;
  static constexpr int kPreW = kUnionW + 2 * kWindow * 2;
  static constexpr int kWriteExp = kPreW + 2 * kPreRows * kListAt * 2;
  static constexpr int kBook = kWriteExp + 2 * kBoxLists * 4;
  static constexpr int kScale = kBook + kBoxLists * 8;
  static constexpr int kKeys = align_of(kScale + kBoxLists * 8, 16);
  static constexpr int kExps = kKeys + 2 * kBoxDigits * 4;
  static constexpr int kBits = kExps + kBoxDigits * 4;
  //: the window's box top and largest digit exponent, then a dummy byte a dead unit's store lands on; each leaf's
  //: exponent byte; each leaf's f16 scale
  static constexpr int kTops = kBits + 8;
  static constexpr int kDeadUnit = kTops + 8;
  static constexpr int kLeafExp = align_of(kDeadUnit + 8, 16);
  static constexpr int kLeafScale = kLeafExp + kBoxDigits * kBoxLists;
  static constexpr int kClock = align_of(kLeafScale + kBoxDigits * kBoxLists * 2, 8);
  static constexpr int kBars = kClock + kWarps * (kCarryPhases + 1) * 8;
  static constexpr int kAddr = align_of(kBars + kBarCount * 8, 16);
  static constexpr int kAddrBytes = 256;
  static constexpr int kEnd = align_of(kAddr + kAddrBytes, 16);
};

//: THE READ TABLES at `heads` query heads a write head, at the top of a CTA's `smem` bytes; the area below them.
//: -- see docs/internals/carry/plan.md#read-tables
__host__ __device__ constexpr ReadTables tables_of(int heads, int smem) {
  const int pre_bytes = align_of(2 * kPreRows * kListAt * heads * 2, 16);
  const int bytes = 2 * heads * kWindow * 2 + pre_bytes + 2 * heads * kWindow + heads * kWindow
                    + heads * kWindow * 4 + 2 * plan_words(heads) * 4;
  const int union_at = (smem - bytes) / 128 * 128, area_at = align_of(Tenants::kEnd, 128);
  const int pre_at = union_at + 2 * heads * kWindow * 2, scale_at = pre_at + pre_bytes;
  const int units_at = scale_at + 2 * heads * kWindow, bounds_at = units_at + heads * kWindow;
  return ReadTables{heads,    plan_words(heads), union_at,  pre_at,
                    scale_at, units_at,          bounds_at, bounds_at + heads * kWindow * 4,
                    area_at,  union_at - area_at};
}

constexpr int kPlaneBytes = kBoxLists * kBoxChannels * 2;
//: THE IMAGE: whole, or a 16-plane half. -- see docs/internals/carry/plan.md#image
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
  static constexpr int kAreaAt = tables_of(1, kSmemBytes).area_at;
  static constexpr int kArea = tables_of(1, kSmemBytes).area;
};

}  // namespace rola::carry
