// csrc/rola/src/facts/facts.cuh -- THE FACTS PASS's constants and operand block: both sides' class-1 words
// (`liveness_contract.cuh`) and the write lists' exponents from the bundle's rows. see docs/internals/facts/facts.md
#pragma once

#include <cstdint>

#include "facts/liveness_contract.cuh"

namespace rola::facts {

//: THE GRID: two blocks of eight warps an SM take list and vote items. -- see docs/internals/facts/facts.md#the-grid
constexpr int kFactsWarps = 8;
constexpr int kFactsThreads = kFactsWarps * 32;
constexpr int kFactsBlocksPerSm = 2;
constexpr int kItemUnits = 4;
constexpr int kParts = kFactsWarps / kItemUnits;
constexpr int kListItemUnits = 2;
constexpr int kListParts = kFactsWarps / kListItemUnits;

//: THE ROW ALIGNMENT LAW's grain (`rola.routing.quant.ROW_ALIGN`). -- see docs/internals/facts/liveness_contract.md#alignment
constexpr int kRowAlign = 32;

//: A WARP'S RING: `kStages` words of 32 tokens x 32 bytes, `kAhead` in flight -- or, for a dense list level's f16 rows,
//: `kDenseStages` words of 32 tokens x 64 bytes in the same bytes. -- see docs/internals/facts/facts.md#load
constexpr int kStages = 4;
constexpr int kAhead = kStages - 1;
constexpr int kWordBytes = 32 * 32;
constexpr int kDenseStages = 2;
constexpr int kDenseWordBytes = 32 * 64;
constexpr int kRingBytes = kFactsWarps * kStages * kWordBytes;
static_assert(kDenseStages * kDenseWordBytes == kStages * kWordBytes,
              "both rings fill a warp's bytes");

//: THE WINDOW, the carry's 512 tokens and the lists' grain, and a warp's part: its words and its rows' store pairs.
constexpr int kWindowWords = 16;
constexpr int kWindowTokens = kWindowWords * kTokBits;
constexpr int kPartWords = kWindowWords / kParts;
constexpr int kListPartWords = kWindowWords / kListParts;
constexpr int kTileStride = kWindowWords + 1;

//: THE STAGED WINDOW'S COLUMN SLOTS: its tokens with a pad after every 8, and past them the window's largest.
constexpr int kColumnSlots = kWindowTokens + kWindowTokens / 8;

constexpr int kMaxWidth = 256;
constexpr int kMaxMaskWords = kMaxLevels * kMaxWidth / 32;
constexpr int kMaxUnits = kSides * kMaxLevels * kMaxWidth / 32;

//: THE LIST EXPONENTS' FLOOR, the smallest normal fp32. -- see docs/internals/facts/facts.md#lists
constexpr float kListFloor = 1.17549435e-38f;

//: THE LISTS OUT: a group's union ranked into slots, a pass `kPassSlots` of them; a (window, digit) list its passes'
//: entries in order, each pass round-robin by slot & 7, compact (a step past its pass's end is the reader's to mask),
//: at most `kListCap`, its pass starts and end `kListAt` a list; a write entry one 8-byte record (slot, 0, c_v, c_m),
//: its coefficient pair at 2^(kCoefTop - E), E its list's exponent floored at the fold's smallest unit.
//: -- see docs/internals/facts/facts.md#lists-out
constexpr int kPassSlots = 128;
constexpr int kMaxPasses = kWindowTokens / kPassSlots;
constexpr int kListCap = kWindowTokens;
constexpr int kListAt = 8;
constexpr int kCoefTop = 4;
constexpr int kUnitMin = kCoefTop - 127;
static_assert(kMaxPasses + 1 <= kListAt, "a list's pass starts and its end fit its starts");
//: a listing warp's entry table, 16 bytes a (digit, pass): a list item's in its spare tiles, a vote item's in its ring
constexpr int kListTableKeys = 32 / kListParts * kMaxPasses;
static_assert(kFactsWarps * kListTableKeys * 16 <= (kItemUnits - kListItemUnits) * 32 * kTileStride * 4,
              "a list item's tables fit its spare tiles");
static_assert(32 / kParts * kMaxPasses * 16 <= kStages * kWordBytes, "a vote item's table fits its warp's ring");

enum FactsPhase : int { kPhaseColumns, kPhaseVote, kPhaseLists, kPhaseStore, kFactsPhases };

//: A SIDE'S LISTS OUT, its level's: entries and starts `[BH, windows, B, kListCap | kListAt]` -- a read entry its slot,
//: a write entry its record -- and the union's tokens and counts `[BH, windows, groups, kWindowTokens | 1]`
struct ListsOut {
  uint16_t* slot;
  uint2* record;
  uint16_t* at;
  uint16_t* union_tok;
  uint16_t* union_count;
};

//: A level's rows (under the row alignment law, with their width in digits) by batch, token, head strides in BYTES --
//: u8 digits, or a dense write level's f16 -- or an fp32 column by its strides in elements.
struct RowView {
  const void* base;
  int64_t sb, st, sh;
  int width;
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
  float2* fm;  // [BH, L], a token's (F, m): the fold's factor products (facts.md#factor-products)
  ListsOut lists[kSides];  // a side's lists, its listed level's (facts.md#lists-out)
  uint32_t* gains16;       // [BH, L], a token's (m, F) as an f16 pair at its window's scales (facts.md#gains)
  int16_t* gains_exp;      // [BH, windows, 2], those scales' exponents
  int* counters;  // [2]: the next item, the blocks finished; zero between calls
  long long* ledger;
  int D, H, L, token_words, windows, items, rows_total, units, groups, tiles, list_level, read_list_level,
      list_units, vote_units, list_item_units, list_items, vote_items, window_items;
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

//: a side's listed level: the write side's list level, the read side's read list level (-1 for none)
constexpr int listed_level(const FactsParams& p, int side) {
  return side == kWrite ? p.list_level : p.read_list_level;
}

}  // namespace rola::facts
