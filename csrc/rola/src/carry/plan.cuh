// csrc/rola/src/carry/plan.cuh -- ONE ARM'S PLAN on the compiling card: its tiles, registers, pool rows and shared
// memory (the tenants, the AREA, the read tables). see docs/internals/carry/plan.md
#pragma once

#include "carry/carry.cuh"
#include "common/arch_caps.cuh"

namespace rola::carry {

namespace arch = ::denseref::arch;

constexpr int align_of(int x, int q) { return (x + q - 1) / q * q; }

//: THE ASYNC COPIES' BARRIERS: the staging, each slot's landed. -- see docs/internals/carry/plan.md#barriers

//: THE FIXED TENANTS' BYTES, every card's, arm's and call's. -- see docs/internals/carry/plan.md#tenants
template <int W>
struct Tenants {
  static constexpr int kCount = 0;
  static constexpr int kUnionW = kCount + 16;
  static constexpr int kPreW = kUnionW + 2 * W * 2;
  static constexpr int kWriteExp = kPreW + 2 * kPreRows * list_at(W) * 2;
  static constexpr int kBook = kWriteExp + 2 * kBoxLists * 4;
  static constexpr int kScale = kBook + kBoxLists * 8;
  static constexpr int kKeys = align_of(kScale + kBoxLists * 8, 16);
  static constexpr int kExps = kKeys + 2 * kBoxDigits * 4;
  static constexpr int kBits = kExps + kWarps * kBoxDigits * 4;
  //: the den image (the dense fold's scratch while it folds, `fold.md#dense`); each digit's den factor and largest
  static constexpr int kDenImage = align_of(kBits + 8, 16);
  static constexpr int kDenFactor = align_of(kDenImage + kBoxDigits * kBoxLists * 2, 16);
  static constexpr int kDenMax = kDenFactor + kBoxDigits * 4;
  static constexpr int kClock = align_of(kDenMax + kBoxDigits * 4, 8);
  static constexpr int kAddr = align_of(kClock + kWarps * (kCarryPhases + 1) * 8, 16);
  static constexpr int kAddrBytes = 256;
  static constexpr int kEnd = align_of(kAddr + kAddrBytes, 16);
};

//: THE READ TABLES at `heads` query heads, at the top of `smem` bytes. -- see docs/internals/carry/plan.md#read-tables
template <int W>
__host__ __device__ constexpr ReadTables tables_of(int heads, int smem) {
  const int pre_bytes = align_of(2 * kPreRows * list_at(W) * heads * 2, 16);
  const int bytes = 2 * heads * W * 2 + pre_bytes + 2 * heads * W + 2 * plan_words(heads, W) * 4;
  const int union_at = (smem - bytes) / 128 * 128, area_at = align_of(Tenants<W>::kEnd, 128);
  const int pre_at = union_at + 2 * heads * W * 2, scale_at = pre_at + pre_bytes;
  return ReadTables{heads,    plan_words(heads, W),     union_at, pre_at,
                    scale_at, scale_at + 2 * heads * W, area_at,  union_at - area_at};
}

constexpr int kPlaneBytes = kBoxLists * kBoxChannels * 2;
//: THE IMAGE: whole, or a 16-plane half. -- see docs/internals/carry/plan.md#image
constexpr int kBatchPlanes = 8, kBatches = kBoxDigits / kBatchPlanes;
constexpr int kImageWhole = kBoxDigits * kPlaneBytes, kImageRing = 2 * kBatchPlanes * kPlaneBytes;

//: THE ARM'S PLAN on card `Cc`: the box's tiles, a side's pool row (its operand slices, then the write side's values),
//: and the area the card's shared memory leaves past the tenants. -- see docs/internals/carry/plan.md#plan
template <class Arm, int Cc = arch::kArch>
struct Plan : Tenants<window_for(Cc)> {
  static constexpr arch::Caps kCaps = arch::caps_of(Cc);
  //: the card's window, and its list boundaries and cap
  static constexpr int kWindow = window_for(Cc), kListAt = list_at(kWindow), kListCap = kWindow;
  static_assert(Arm::fold_rows == kBoxDigits && Arm::read_rows == kBoxLists
                    && Arm::channels == kBoxChannels,
                "the arm is the box's: a fold list 32 digits, a read digit 32 lists, 32 channels");
  static_assert(Arm::read_factors == 1, "the value-carve readout serves a listed read side");
  static constexpr int kFoldFactors = Arm::fold_factors, kReadFactors = Arm::read_factors;
  //: THE HAND-OFF'S FORM: the fold's lists the readout's digits (both sides list one level) or its K
  static constexpr bool kSameSplit = Arm::same_split != 0;
  //: a warp's fold tile: lists `w + 8 j`, each the box's 32 rows in m-tiles of 16
  static constexpr int kFoldLists = kBoxLists / kWarps;
  static constexpr int kFoldMTiles = kBoxDigits / 16;
  static constexpr int kStateRegs = kFoldLists * kBoxDigits * kBoxChannels / 32;
  static_assert(
      kFoldMTiles * 2 == kBatches && kFoldLists == kBatches,
      "a batch (8 digits) is half an m-tile of every fold list, or one list of every warp's");

  static constexpr int kReadRow = kSliceBytes * kReadFactors;
  static constexpr int kValuesAt = kSliceBytes * kFoldFactors,
                       kWriteRow = kValuesAt + kBoxChannels * 2;

  static constexpr int kSmemBytes = kCaps.smem_per_cta_max;
  static constexpr int kAreaAt = tables_of<kWindow>(1, kSmemBytes).area_at;
  static constexpr int kArea = tables_of<kWindow>(1, kSmemBytes).area;
};

}  // namespace rola::carry
