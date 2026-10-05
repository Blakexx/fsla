// csrc/rola/src/owner/owner.cuh -- THE OWNER CARRY's constants, carve and operand block: a CTA owns a box of leaves
// and carries its windows' writes into it and its state out to the readers. see docs/internals/owner/owner.md
#pragma once

#include <cstdint>

namespace rola::owner {

//: THE WINDOW IS THE INTRA'S AND THE FACTS PASS'S: sixteen 32-token words a row. -- see docs/internals/owner/owner.md#constants
constexpr int kWindow = 512;
constexpr int kWindowWords = kWindow / 32;
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kChannels = 32;
constexpr int kChannelGroups = 2;
constexpr int kListCapacity = kWindow;
constexpr int kWritePool = 128;
constexpr int kWritePass = kWritePool;
constexpr int kReadPool = 128;
constexpr int kReadPass = kReadPool - 1;

//: A CARVE: RS read-sparse by RD read-dense digits a CTA. -- see docs/internals/owner/owner.md#carve
template <int RS_, int RD_>
struct Carve {
  static constexpr int RS = RS_, RD = RD_;
  static constexpr int kFoldLists = RD / kWarps;
  static constexpr int kReadDigits = RS / kWarps;
  static constexpr int kWordRows = RD + RS;
  static_assert(RD % kWarps == 0 && RS % kWarps == 0,
                "a warp owns whole lists and whole read digits");
};

//: A STEP FORM. -- see docs/internals/owner/owner.md#step
enum Step : int { kStepChosen = 0, kStepPerList = 1, kStepMerged = 2 };

//: A LEVEL'S ROWS OR THE VALUE PLANE, where the bundle holds them, strides in bytes.
struct RowView {
  const void* base;
  int64_t sb, st, sh;
};

//: THE OPERAND BLOCK, a grid constant. -- see docs/internals/owner/owner.md#params
struct OwnerParams {
  RowView read[2], write[2], v;
  const float2* fm;
  const uint32_t* words;
  const uint32_t* list_exp;
  int64_t side_words, row_words;
  int level_s, level_r;
  int row_base[2], width[2], list_width;
  int sparse_write, sparse_read;
  int H, L, windows, step;
  uint32_t* dump;
  long long* ledger;
};

}  // namespace rola::owner
