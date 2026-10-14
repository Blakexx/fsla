// csrc/rola/src/carry/plan.cuh -- ONE ARM'S PLAN on the compiling card: the box's warp tiles, its registers, and its
// shared memory -- the persistent tenants, then the image, the panel and the fills' ring, whose sizes the card's
// capacity picks. see docs/internals/carry/plan.md
#pragma once

#include "carry/carry.cuh"
#include "common/arch_caps.cuh"

namespace rola::carry {

namespace arch = ::denseref::arch;

constexpr int max_of(int a, int b) { return a > b ? a : b; }

constexpr int align_of(int x, int q) { return (x + q - 1) / q * q; }

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
  static constexpr int kMaxBars = 10;
  static constexpr int kReadHeads = kBars + kMaxBars * 8;
  static constexpr int kHeadBytes = 4 * kHeadEntries * 8;
  static constexpr int kFoldHeads = kReadHeads + kWarps * kHeadBytes;
  static constexpr int kAddr = kFoldHeads + kWarps * kHeadBytes;
  static constexpr int kAddrBytes = 256;
  static constexpr int kEnd = align_of(kAddr + kAddrBytes, 128);
};

//: A FORM: the image in `parts`, a readout pass of `slots`, a fold unit of `unit` slots.
struct Form {
  int parts, slots, unit;
};

//: THE FORMS IN ORDER OF PREFERENCE: fewest image stores a dense window (parts x passes), then the larger unit.
constexpr Form kForms[] = {{1, 512, 128}, {1, 512, 64}, {1, 256, 128}, {1, 256, 64},
                           {2, 512, 128}, {2, 512, 64}, {2, 256, 128}, {2, 256, 64},
                           {1, 128, 128}, {1, 128, 64}, {2, 128, 128}, {2, 128, 64}};
constexpr int kFormCount = sizeof(kForms) / sizeof(kForms[0]);

constexpr int kPlaneBytes = kBoxLists * kBoxChannels * 2;

//: A FORM's PHASE REGION: its image, the pads and panel, the readout ring's two slots. -- see plan.md#forms
constexpr int ring_slot(const Form& f, int read_row, int write_row, int k) {
  return f.slots < kWindow ? max_of(f.slots * read_row, f.unit * write_row)
                           : (k == 0 ? f.slots * read_row : f.unit * write_row);
}

constexpr int form_bytes(const Form& f, int read_row, int write_row) {
  return kBoxDigits / f.parts * kPlaneBytes + kPadBytes + f.slots * kPanelRow
         + ring_slot(f, read_row, write_row, 0) + ring_slot(f, read_row, write_row, 1);
}

constexpr int form_of(int room, int read_row, int write_row) {
  for (int k = 0; k < kFormCount; ++k)
    if (form_bytes(kForms[k], read_row, write_row) <= room) return k;
  return kFormCount;
}

//: THE ARM'S PLAN on card `Cc`: the box's tiles, a side's pool row (its operand slices, then the write side's values),
//: and the first form the card's shared memory holds. -- see docs/internals/carry/plan.md#plan
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

  static constexpr int kReadRow = kSliceBytes * kReadFactors;
  static constexpr int kValuesAt = kSliceBytes * kFoldFactors,
                       kWriteRow = kValuesAt + kBoxChannels * 2;

  static constexpr int kForm = form_of(kCaps.smem_per_cta_max - kEnd, kReadRow, kWriteRow);
  static_assert(kForm < kFormCount, "a form fits the card's shared memory");
  static constexpr int kParts = kForms[kForm].parts, kPassSlots = kForms[kForm].slots;
  static constexpr int kUnitSlots = kForms[kForm].unit;
  static_assert(kFoldMTiles % kParts == 0 && kReadDigits % kParts == 0,
                "a part is whole m-tiles of every fold list and whole digits of every reader");

  //: THE PHASE REGION: the image, the pads, the panel, the readout ring's slots
  static constexpr int kPlanes = kBoxDigits / kParts;
  static constexpr int kImageBytes = kPlanes * kPlaneBytes;
  static constexpr int kPanelBytes = kPassSlots * kPanelRow;
  static constexpr int kReadBytes = kPassSlots * kReadRow, kUnitBytes = kUnitSlots * kWriteRow;
  static constexpr int kImage = kEnd;
  static constexpr int kPads = kImage + kImageBytes;
  static constexpr int kPanel = kPads + kPadBytes;
  static constexpr int kRing0 = kPanel + kPanelBytes;
  static constexpr int kRing1 = kRing0 + ring_slot(kForms[kForm], kReadRow, kWriteRow, 0);
  static constexpr int kSmemBytes = kEnd + form_bytes(kForms[kForm], kReadRow, kWriteRow);
  //: the fold's ring: units over the image, then the panel (free only once the drain has read it)
  static constexpr int kImageUnits = kImageBytes / kUnitBytes;
  static constexpr int kFoldRing = kImageUnits + kPanelBytes / kUnitBytes;
  static_assert(kFoldRing >= 1 && 4 + kFoldRing <= kMaxBars,
                "a barrier a ring slot and the two staging barriers");
};

}  // namespace rola::carry
