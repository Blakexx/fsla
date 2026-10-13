// csrc/rola/src/facts/facts.cuh -- THE FACTS PASS's constants and operand block: both sides' class-1 words
// (`liveness_contract.cuh`), the window's columns, and each side's lists in the carry's final form, a CTA a unit.
// see docs/internals/facts/facts.md
#pragma once

#include <cstdint>

#include "carry/carry.cuh"
#include "facts/liveness_contract.cuh"

namespace rola::facts {

//: THE CTA: eight warps a UNIT, three CTAs an SM. -- see docs/internals/facts/facts.md#the-grid
constexpr int kFactsWarps = 8;
constexpr int kFactsThreads = kFactsWarps * 32;
constexpr int kFactsCtasMin = 3;

//: THE ROW ALIGNMENT LAW's grain (`rola.routing.quant.ROW_ALIGN`). -- see docs/internals/facts/liveness_contract.md#alignment
constexpr int kRowAlign = 32;

//: THE WINDOW, the carry's 512 tokens and the lists' grain.
constexpr int kWindowWords = 16;
constexpr int kWindowTokens = kWindowWords * kTokBits;
constexpr int kWindowTiles = kWindowTokens / kTileTokens;

constexpr int kMaxWidth = 256;
constexpr int kMaxGroups = kMaxWidth / kGroupDigits;
constexpr int kMaxMaskWords = kMaxLevels * kMaxWidth / 32;

//: A UNIT, coded (kind, side, level, group). -- see docs/internals/facts/facts.md#the-plan
enum UnitKind : int { kUnitVote = 0, kUnitList = 1, kUnitColumns = 2 };

constexpr int kMaxUnits = kSides * kMaxLevels * kMaxGroups + 1;

__host__ __device__ constexpr uint32_t unit_code(int kind, int side, int level, int group) {
  return static_cast<uint32_t>(kind | side << 2 | level << 3 | group << 5);
}

__host__ __device__ constexpr int unit_kind(uint32_t c) { return c & 3; }

__host__ __device__ constexpr int unit_side(uint32_t c) { return (c >> 2) & 1; }

__host__ __device__ constexpr int unit_level(uint32_t c) { return (c >> 3) & 3; }

__host__ __device__ constexpr int unit_group(uint32_t c) { return (c >> 5) & 7; }

//: A BIT TILE: the group's 32 digits x the window's 16 words, row `d` word `j` at `16 d + (j ^ (d / 2 % 16))`.
//: -- see docs/internals/facts/facts.md#tiles
__host__ __device__ constexpr int tile_at(int d, int j) {
  return kWindowWords * d + (j ^ ((d >> 1) & 15));
}

//: THE COLORING: 8 colors, the union at most `kUnionCap` a color, the claim queue a ring twice the most it holds,
//: `kClaimRounds` rounds on every warp. -- see docs/internals/facts/facts.md#coloring
constexpr int kColors = 8;
constexpr int kUnionCap = kWindowTokens / kColors;
constexpr int kQueueCap = 2 * kWindowTokens;
constexpr int kClaimRounds = 4;

//: THE LIST EXPONENTS' FLOOR, the smallest normal fp32. -- see docs/internals/facts/facts.md#lists
constexpr float kListFloor = 1.17549435e-38f;

//: THE LISTS OUT, the carry's final form (`carry/carry.cuh` owns its addresses), a write entry's coefficients at
//: 2^(kCoefTop - E) with E floored at `kUnitMin`. -- see docs/internals/facts/facts.md#lists-out
constexpr int kListCap = carry::kListCap;
constexpr int kListAt = carry::kListAt;
constexpr int kHeadEntries = carry::kHeadEntries;
constexpr int kHeadLists = kGroupDigits / kFactsWarps;
constexpr int kCoefTop = 4;
constexpr int kUnitMin = kCoefTop - 127;
constexpr int kMaxPasses = 4;
static_assert(kWindowTokens == carry::kWindow && kFactsWarps == carry::kWarps
                  && kGroupDigits == carry::kBoxLists,
              "the lists are the carry's: its window, and a warp's heads its box lists w + 8 j");
constexpr int kMinPassLog = 7, kMaxPassLog = 9;
static_assert(kWindowTokens >> kMinPassLog == kMaxPasses && kMaxPasses + 1 <= kListAt,
              "a list's pass starts and its end fit its starts");

enum FactsPhase : int {
  kPhaseLoad,
  kPhaseVote,
  kPhaseInputs,
  kPhaseClaims,
  kPhaseSingles,
  kPhaseSlots,
  kPhaseLists,
  kFactsPhases
};

//: THE CTA'S SHARED MEMORY, by offset. -- see docs/internals/facts/facts.md#shared-memory
struct FactsSmem {
  static constexpr int kWin = 0;
  static constexpr int kTile = kWin + kWindowTokens * kGroupDigits;
  static constexpr int kSlot = kTile + kGroupDigits * kWindowWords * 4;
  static constexpr int kUnion = kSlot + kWindowTokens * 2;
  static constexpr int kCols = kUnion + kWindowTokens * 2;
  static constexpr int kScratch = kCols + kWindowTokens * 8;
  //: the coloring's: digits a token, the queue, counts and capacity a digit, the deal's index bases and ends
  static constexpr int kMasks = kScratch;
  static constexpr int kQueue = kMasks + kWindowTokens * 4;
  static constexpr int kCounts = kQueue + kQueueCap * 2;
  static constexpr int kCap = kCounts + kGroupDigits * 16;
  static constexpr int kPre = kCap + kGroupDigits * 4;
  static constexpr int kEnds = kPre + kGroupDigits * kWindowWords * 2;
  static constexpr int kColorEnd = kEnds + kGroupDigits * kColors * 2;
  //: the placement's, over the dead digits: a (list, pass)'s color counts, then its earlier passes', and its start
  static constexpr int kTab = kScratch;
  static constexpr int kStart = kTab + kGroupDigits * kMaxPasses * 8;
  static constexpr int kPlaceEnd = kStart + kGroupDigits * kMaxPasses * 2;
  static constexpr int kSmall = kColorEnd;
  //: the union's words, word and union color counts, the rounds' appends, the warps' maxima, extent and queue length
  static constexpr int kUword = kSmall;
  static constexpr int kMword = kUword + kWindowWords * 4;
  static constexpr int kTword = kMword + kWindowWords * 4;
  static constexpr int kWcount = kTword + kWindowWords * 4;
  static constexpr int kUcount = kWcount + kWindowWords * kColors;
  static constexpr int kAppends = kUcount + kColors * 4;
  static constexpr int kWmax = kAppends + kColors * 4;
  static constexpr int kMisc = kWmax + kFactsWarps * 8;
  static constexpr int kBytes = kMisc + 16;
};

static_assert(
    FactsSmem::kTile % 16 == 0 && FactsSmem::kCols % 16 == 0 && FactsSmem::kScratch % 16 == 0
        && FactsSmem::kCounts % 16 == 0 && FactsSmem::kEnds % 16 == 0 && FactsSmem::kSmall % 16 == 0
        && FactsSmem::kUcount % 16 == 0 && FactsSmem::kPlaceEnd <= FactsSmem::kQueue,
    "each tenant on its access's alignment, the placement's tables inside the dead masks");

//: A SIDE'S LISTS OUT at its list level: entries, starts, heads, the union by slot, its extent.
//: -- see docs/internals/facts/facts_api.md#rola-facts-forward
struct ListsOut {
  uint2* entries;
  uint16_t* at;
  uint2* heads;
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
  int D, H, L, token_words, windows, rows_total, groups, tiles, per_window, read_row;
  int B[kMaxLevels];
  int row_base[kMaxLevels];
  int group_base[kMaxLevels];
  int list_level[kSides];
  int pass_log[kSides];
  uint32_t unit[kMaxUnits];
  uint32_t mask[kSides][kMaxMaskWords];
};

}  // namespace rola::facts
