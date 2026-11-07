// csrc/rola/src/facts/facts.cuh -- THE FACTS PASS's constants and operand block: both sides' class-1 words
// (`liveness_contract.cuh`), the window's columns, and each side's lists in the carry's final form, a CTA a unit.
// see docs/internals/facts/facts.md
#pragma once

#include <cstdint>

#include "carry/carry.cuh"
#include "facts/liveness_contract.cuh"

namespace rola::facts {

//: THE CTA: eight warps a UNIT, four CTAs an SM at W = 512 and two at 1,024 (a list lane's twice the words and
//: boundaries in registers). -- see docs/internals/facts/facts.md#the-grid
constexpr int kFactsWarps = 8;
constexpr int kFactsThreads = kFactsWarps * 32;
constexpr int kFactsCtasMin = kWindow >= 1024 ? 2 : 4;

//: THE ROW ALIGNMENT LAW's grain (`rola.routing.quant.ROW_ALIGN`). -- see docs/internals/facts/liveness_contract.md#alignment
constexpr int kRowAlign = 32;

//: THE WINDOW, the compiling card's (common/window.cuh; host code takes its device's), and the counts the stages map
//: onto it: a warp's 32-token blocks, a thread's tokens, a list lane's words. -- see docs/internals/facts/facts.md#window
constexpr int kWindowTokens = kWindow;
constexpr int kWindowWords = kWindowTokens / kTokBits;
constexpr int kWindowTiles = kWindowTokens / kTileTokens;
constexpr int kWarpBlocks = kWindowWords / kFactsWarps;
constexpr int kThreadTokens = kWindowTokens / kFactsThreads;
constexpr int kLaneWords = kWindowWords / 8;

constexpr int kMaxWidth = 256;
constexpr int kMaxGroups = kMaxWidth / kGroupDigits;
constexpr int kMaxMaskWords = kMaxLevels * kMaxWidth / 32;
//: THE QUERY HEADS a write head's read side merges: the parameter block's bound, and a window's at W (every head's rows
//: at a 16-bit pool offset). -- see docs/internals/facts/facts.md#window
constexpr int kMaxGroup = 16;

__host__ __device__ constexpr int max_group(int w) { return 8192 / w; }

//: A UNIT, coded (kind, side, level, group, head in the group). -- see docs/internals/facts/facts.md#the-plan
enum UnitKind : int { kUnitVote = 0, kUnitList = 1, kUnitColumns = 2 };

constexpr int kMaxUnits = kMaxLevels * kMaxGroups * (kMaxGroup + 1) + 1;

__host__ __device__ constexpr uint32_t unit_code(int kind, int side, int level, int group,
                                                 int head) {
  return static_cast<uint32_t>(kind | side << 2 | level << 3 | group << 5 | head << 8);
}

__host__ __device__ constexpr int unit_kind(uint32_t c) { return c & 3; }

__host__ __device__ constexpr int unit_side(uint32_t c) { return (c >> 2) & 1; }

__host__ __device__ constexpr int unit_level(uint32_t c) { return (c >> 3) & 3; }

__host__ __device__ constexpr int unit_group(uint32_t c) { return (c >> 5) & 7; }

__host__ __device__ constexpr int unit_head(uint32_t c) { return (c >> 8) & 15; }

//: A BIT TILE: the group's 32 digits x the window's 16 words, row `d` word `j` at `16 d + (j ^ (d / 2 % 16))`.
//: -- see docs/internals/facts/facts.md#tiles
__host__ __device__ constexpr int tile_at(int d, int j) {
  return kWindowWords * d + (j ^ (kWindowWords >= 32 ? (d & 31) : ((d >> 1) & 15)));
}

//: THE LIST EXPONENTS' FLOOR, the smallest normal fp32. -- see docs/internals/facts/facts.md#lists
constexpr float kListFloor = 1.17549435e-38f;

//: THE LISTS OUT, the carry's final form (`carry/carry.cuh` owns its addresses), a write entry's coefficients at
//: 2^(kCoefTop - E) with E floored at `kUnitMin`. -- see docs/internals/facts/facts.md#lists-out
constexpr int kListCap = kWindowTokens;
constexpr int kListBound = carry::kListBound;
constexpr int kListAt = kWindowTokens / kListBound;
constexpr int kLaneBounds = kListAt / 8;
constexpr int kCoefTop = 4;
constexpr int kUnitMin = kCoefTop - 127;
static_assert(kFactsWarps == carry::kWarps && kGroupDigits == carry::kMaxExtent,
              "the lists are the carry's: its box lists, a boundary every 64 slots");
static_assert(kWindowTokens == kWindow,
              "the facts pass's window is the compiling card's (common/window.cuh)");

//: THE STEP: a union's extent is its size in whole steps of the carry's 8 entries.
constexpr int kStep = 8;

enum FactsPhase : int { kPhaseLoad, kPhaseVote, kPhaseRanks, kPhaseLists, kFactsPhases };

//: THE CTA'S SHARED MEMORY, by offset: the window, the bit tile, the columns, the union's words, each boundary's first
//: token, the warps' maxima, a head's lists' boundaries. -- see docs/internals/facts/facts.md#shared-memory
template <int W>
struct FactsSmemAt {
  static constexpr int kWin = 0;
  static constexpr int kTile = kWin + W * kGroupDigits;
  static constexpr int kCols = kTile + kGroupDigits * (W / kTokBits) * 4;
  static constexpr int kUword = kCols + W * 8;
  static constexpr int kBound = kUword + (W / kTokBits) * 4;
  static constexpr int kWmax = kBound + (W / kListBound) * 4;
  static constexpr int kTable = kWmax + kFactsWarps * 8;
  static constexpr int kBytes = kTable + kGroupDigits * (W / kListBound + 1) * 2;
};

using FactsSmem = FactsSmemAt<kWindowTokens>;

template <int W>
constexpr bool tenants_aligned() {
  using S = FactsSmemAt<W>;
  return S::kTile % 16 == 0 && S::kCols % 16 == 0 && S::kUword % 16 == 0 && S::kWmax % 8 == 0
         && S::kTable % 16 == 0;
}

static_assert(tenants_aligned<512>() && tenants_aligned<1024>(),
              "each tenant on its access's alignment");

//: A SIDE'S LISTS OUT at its list level: entries, prefix rows (mod 2^16), exact totals, the union by slot, its extent.
//: -- see docs/internals/facts/facts_api.md#rola-facts-forward
struct ListsOut {
  uint2* entries;
  uint16_t* pre;
  int32_t* totals;
  uint16_t* union_tok;
  int32_t* union_count;
};

//: A level's rows by batch, token, head strides in BYTES and width in digits; an fp32 column by element strides.
struct RowView {
  const uint8_t* base;
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
  uint32_t*
      gains16;  // [BH, L], a token's (m, F) as an f16 pair at its window's scales (facts.md#gains)
  int16_t* gains_exp;  // [BH, windows, 2], those scales' exponents
  ListsOut lists[kSides];
  long long* ledger;
  int D, H, g, L, token_words, windows, rows_total, groups, tiles, per_window, read_row;
  int B[kMaxLevels];
  int row_base[kMaxLevels];
  int group_base[kMaxLevels];
  int list_level[kSides];
  uint32_t unit[kMaxUnits];
  uint32_t mask[kSides][kMaxMaskWords];
};

}  // namespace rola::facts
