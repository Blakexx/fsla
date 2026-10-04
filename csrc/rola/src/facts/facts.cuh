// csrc/rola/src/facts/facts.cuh -- THE FACTS PASS's constants and operand block: both sides' class-1 words
// (`liveness_contract.cuh`) and the write lists' exponents from the bundle's rows. see docs/internals/facts/facts.md
#pragma once

#include <cstdint>

#include "facts/liveness_contract.cuh"

namespace rola::facts {

//: THE GRID: two blocks of eight warps an SM walk items of `kItemUnits` units, `kParts` warps a unit.
//: -- see docs/internals/facts/facts.md#the-grid
constexpr int kFactsWarps = 8;
constexpr int kFactsThreads = kFactsWarps * 32;
constexpr int kFactsBlocksPerSm = 2;
constexpr int kItemUnits = 4;
constexpr int kParts = kFactsWarps / kItemUnits;

//: THE WINDOW, the carry's 512 tokens and the lists' grain, and a warp's part: its words and its rows' store pairs.
constexpr int kWindowWords = 16;
constexpr int kWindowTokens = kWindowWords * kTokBits;
constexpr int kPartWords = kWindowWords / kParts;
constexpr int kPartRowPairs = 32 / kParts / 2;
constexpr int kTileStride = kWindowWords + 1;

constexpr int kMaxWidth = 256;
constexpr int kMaxMaskWords = kMaxLevels * kMaxWidth / 32;
constexpr int kMaxUnits = kSides * kMaxLevels * kMaxWidth / 32;

//: THE LIST EXPONENTS' FLOOR, the smallest normal fp32: a dead list's exponent is -126.
//: -- see docs/internals/facts/facts.md#lists
constexpr float kListFloor = 1.17549435e-38f;

enum FactsPhase : int { kPhaseColumns, kPhaseVote, kPhaseStore, kFactsPhases };

//: A level's rows (digits contiguous) or an fp32 column, by its batch, token and head strides.
struct RowView {
  const void* base;
  int64_t sb, st, sh;
};

struct ColumnView {
  const float* base;
  int64_t sb, st, sh;
};

//: THE PASS'S OPERAND BLOCK, by value. -- see docs/internals/facts/facts.md#factsparams
struct FactsParams {
  RowView rows[kSides][kMaxLevels];
  ColumnView scale_write[kMaxLevels];
  ColumnView g_write, v_scale;
  uint32_t* words;
  uint32_t* unions;
  uint32_t* list_exp;  // [BH, windows, B_list], a digit's (K_m, K_v) int16 pair
  long long* ledger;
  int D, H, L, token_words, windows, items, rows_total, units, groups, tiles, list_level,
      list_items, window_items;
  int B[kMaxLevels];
  int row_base[kMaxLevels];
  int group_base[kMaxLevels];
  uint8_t unit[kMaxUnits];
  uint32_t dense[kSides];
  uint32_t mask[kSides][kMaxMaskWords];
};

constexpr int unit_code(int side, int level, int group) { return side << 5 | level << 3 | group; }

constexpr bool is_list_unit(const FactsParams& p, int code) {
  return (code >> 5) == kWrite && ((code >> 3) & 3) == p.list_level;
}

}  // namespace rola::facts
