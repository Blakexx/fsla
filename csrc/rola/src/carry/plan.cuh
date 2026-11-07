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
  static constexpr int kBook = kWriteExp + 2 * kMaxExtent * 4;
  static constexpr int kScale = kBook + kMaxExtent * 8;
  static constexpr int kKeys = align_of(kScale + kMaxExtent * 8, 16);
  static constexpr int kExps = kKeys + 2 * kMaxExtent * 4;
  static constexpr int kBits = kExps + kWarps * kMaxExtent * 4;
  //: the den image (the dense fold's scratch while it folds, `fold.md#dense`); each digit's den factor and largest
  static constexpr int kDenImage = align_of(kBits + 8, 16);
  static constexpr int kDenFactor = align_of(kDenImage + kMaxExtent * kMaxExtent * 2, 16);
  static constexpr int kDenMax = kDenFactor + kMaxExtent * 4;
  static constexpr int kClock = align_of(kDenMax + kMaxExtent * 4, 8);
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

//: A BATCH of the image's stores: 8 digits' planes. -- see docs/internals/carry/plan.md#image
constexpr int kBatchPlanes = 8;

//: THE ARM'S PLAN on card `Cc`: the box's tiles, a side's pool row (its operand slices, then the write side's values),
//: and the area the card's shared memory leaves past the tenants. -- see docs/internals/carry/plan.md#plan
template <class Arm, int Cc = arch::kArch>
struct Plan : Tenants<window_for(Cc)> {
  static constexpr arch::Caps kCaps = arch::caps_of(Cc);
  //: the card's window, and its list boundaries and cap
  static constexpr int kWindow = window_for(Cc), kListAt = list_at(kWindow), kListCap = kWindow;
  static_assert(Arm::read_factors == 1, "the value-carve readout serves a listed read side");
  static constexpr int kFoldFactors = Arm::fold_factors, kReadFactors = Arm::read_factors;
  //: THE HAND-OFF'S FORM: the fold's lists the readout's digits (both sides list one level) or its K
  static constexpr bool kSameSplit = Arm::same_split != 0;
  //: THE ARM'S BOX: the fold's rows, channels and lists; the readout's digits and K. -- see docs/internals/carry/plan.md#box
  static constexpr int kRows = Arm::fold_rows, kChannels = Arm::channels,
                       kLists = kStateValues / (kRows * kChannels);
  static constexpr int kDigits = kSameSplit ? kLists : kRows, kK = kSameSplit ? kRows : kLists;
  static_assert(kLists * kRows * kChannels == kStateValues && Arm::read_rows == kK,
                "the arm's box holds the state, its read rows the readout's K");
  static_assert((kLists == 16 || kLists == 32) && (kRows == 16 || kRows == 32)
                    && (kChannels == 32 || kChannels == 64),
                "the boxes the components serve: 16 or 32 lists and rows, 32 or 64 channels");
  static constexpr int kFoldLists = kLists / kWarps;
  static constexpr int kFoldMTiles = kRows / 16, kFoldNTiles = kChannels / 8;
  static constexpr int kStateRegs = kFoldLists * kRows * kChannels / 32;
  //: THE DEN'S REGISTERS: a lane's (m-tile, list) pairs, two rows each. -- see docs/internals/carry/fold.md#state
  static constexpr int kDenLists = kFoldLists * kFoldMTiles / 4, kDenRegs = 2 * kDenLists;
  static constexpr int kChannelGroups = kChannels / 8, kTokenGroups = kWarps / kChannelGroups;
  static constexpr int kDenWords = kChannelGroups, kDenRowBytes = kDenWords * 4;
  //: A SIDE'S LIST GROUP, as this arm reads its lists. -- see docs/internals/carry/plan.md#list-groups
  static constexpr int kReadListGroup = kDigits, kWriteListGroup = kLists;
  static_assert(kReadListGroup + 1 <= kPreRows && kWriteListGroup + 1 <= kPreRows,
                "a list group's prefix rows, one a list and its totals, inside the tenants' cap");
  //: THE IMAGE: a digit's plane, its slices, batches and halves. -- see docs/internals/carry/plan.md#image
  static constexpr int kPlaneBytes = kK * kChannels * 2, kSliceBlock = kK * 16;
  static constexpr int kBatches = kDigits / kBatchPlanes, kHalfPlanes = kDigits / 2;
  static constexpr int kImageWhole = kDigits * kPlaneBytes, kImageRing = kHalfPlanes * kPlaneBytes;
  static_assert(
      kSameSplit ? kFoldLists == kBatches : kFoldMTiles * 2 == kBatches,
      "a batch (8 digits) is half an m-tile of every fold list, or one list of every warp's");

  static_assert(kFoldFactors == 1 || kFoldFactors == 2,
                "a write side's one list level, or its two dense levels");
  static constexpr int kReadRow = 2 * kK;
  static constexpr int kValuesAt = 2 * kRows + (kFoldFactors == 2 ? 2 * kLists : 0),
                       kWriteRow = kValuesAt + kChannels * 2;

  static constexpr int kSmemBytes = kCaps.smem_per_cta_max;
  static constexpr int kAreaAt = tables_of<kWindow>(1, kSmemBytes).area_at;
  static constexpr int kArea = tables_of<kWindow>(1, kSmemBytes).area;
};

}  // namespace rola::carry
