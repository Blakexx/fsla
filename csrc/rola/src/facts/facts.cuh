// csrc/rola/src/facts/facts.cuh -- THE FACTS PASS's constants and operand block: both sides' class-1 words
// (`liveness_contract.cuh`) and the write lists' exponents from the bundle's rows. see docs/internals/facts/facts.md
#pragma once

#include <cstdint>

#include "facts/liveness_contract.cuh"

namespace rola::facts {

//: THE GRID: one block of sixteen warps an SM, two TEAMS of eight over the block's UNITS (batch-head, window, span).
//: -- see docs/internals/facts/facts.md#the-grid
constexpr int kFactsWarps = 16;
constexpr int kFactsThreads = kFactsWarps * 32;
constexpr int kFactsBlocksPerSm = 1;
constexpr int kTeams = 2;
constexpr int kTeamWarps = kFactsWarps / kTeams;
constexpr int kTeamThreads = kTeamWarps * 32;

//: THE ROW ALIGNMENT LAW's grain (`rola.routing.quant.ROW_ALIGN`). -- see docs/internals/facts/liveness_contract.md#alignment
constexpr int kRowAlign = 32;

//: THE WINDOW, the carry's 512 tokens and the lists' grain.
constexpr int kWindowWords = 16;
constexpr int kWindowTokens = kWindowWords * kTokBits;

//: THE TWO STREAMS: RING items (a span of `kSpanDigits` digits) through a warp's ring of `kRingDepth` stages of
//: `kStageBytes`; the write list level's HALVES (`kHalfDigits`) held whole in the HELD WINDOW through their build.
//: -- see docs/internals/facts/facts.md#load
constexpr int kSpanDigits = 128;
constexpr int kHalfDigits = 64;
constexpr int kStageBytes = 1024;
constexpr int kRingDepth = 2;
constexpr int kStageTokens = kStageBytes / kSpanDigits;
constexpr int kHalfBytes = kWindowTokens * kHalfDigits;
constexpr int kHalfRegion = kTokBits * kHalfDigits;
constexpr int kBlocks = kWindowWords / kTeamWarps;
constexpr int kHalfQuanta = kHalfDigits / 16 * kTokBits / 32;
constexpr int kHalfGroups = kHalfDigits / kGroupDigits;
constexpr int kDenseStageTokens = kStageBytes / (2 * kSpanDigits);
constexpr int kRingStages = kTokBits / kStageTokens;
constexpr int kColumnPasses = kWindowTokens / kTeamThreads;

constexpr int kMaxWidth = 256;
constexpr int kMaxSpans = kMaxWidth / kSpanDigits;
constexpr int kMaxMaskWords = kMaxLevels * kMaxWidth / 32;

//: A BIT TILE: an item's digits x the window's 16 words, row `d` word `j` at `16 d + (j ^ (d / 2 % 16))`.
//: -- see docs/internals/facts/facts.md#tiles
constexpr int kTileWords = kWindowWords;

__host__ __device__ constexpr int tile_at(int d, int j) {
  return kTileWords * d + (j ^ ((d >> 1) & 15));
}

//: A UNIT'S PLAN, the same for every unit of a span: its items in order, each a coded (kind, side, level, group0,
//: groups, team). -- see docs/internals/facts/facts.md#the-plan
enum ItemKind : int { kItemHalf = 0, kItemVote = 1, kItemReadList = 2, kItemDenseMax = 3 };

constexpr int kMaxPlan = 2 + kSides * kMaxLevels;

__host__ __device__ constexpr uint32_t plan_code(int kind, int side, int level, int group0,
                                                 int groups, int team) {
  return static_cast<uint32_t>(kind | side << 2 | level << 3 | group0 << 5 | groups << 8
                               | team << 11);
}

__host__ __device__ constexpr int plan_kind(uint32_t c) { return c & 3; }

__host__ __device__ constexpr int plan_side(uint32_t c) { return (c >> 2) & 1; }

__host__ __device__ constexpr int plan_level(uint32_t c) { return (c >> 3) & 3; }

__host__ __device__ constexpr int plan_group0(uint32_t c) { return (c >> 5) & 7; }

__host__ __device__ constexpr int plan_groups(uint32_t c) { return (c >> 8) & 7; }

__host__ __device__ constexpr int plan_team(uint32_t c) { return (c >> 11) & 1; }

//: A BUILD: four warps a 32-digit group (each a quarter of its digits).
constexpr int kParts = 4;
constexpr int kListDigits = kGroupDigits / kParts;

//: THE LIST EXPONENTS' FLOOR, the smallest normal fp32. -- see docs/internals/facts/facts.md#lists
constexpr float kListFloor = 1.17549435e-38f;

//: THE LISTS OUT: passes of `kPassSlots` slots, at most `kListCap` entries a list, `kListAt` starts a list, a write
//: entry's coefficients at 2^(kCoefTop - E), E floored at `kUnitMin`. -- see docs/internals/facts/facts.md#lists-out
constexpr int kPassSlots = 128;
constexpr int kMaxPasses = kWindowTokens / kPassSlots;
constexpr int kListCap = kWindowTokens;
constexpr int kListAt = 8;
constexpr int kCoefTop = 4;
constexpr int kUnitMin = kCoefTop - 127;
static_assert(kMaxPasses + 1 <= kListAt, "a list's pass starts and its end fit its starts");
//: a building warp's entry table, 16 bytes a (digit, pass) of its quarter of a group
constexpr int kListTableKeys = kListDigits * kMaxPasses;
static_assert(kListTableKeys == 32, "a building warp's table is a key a lane");

enum FactsPhase : int { kPhaseColumns, kPhaseVote, kPhaseLists, kPhaseStore, kFactsPhases };

//: THE BLOCK'S SHARED MEMORY, by offset, the teams' apart. -- see docs/internals/facts/facts.md#shared-memory
struct FactsSmem {
  static constexpr int kHeld = 0;
  static constexpr int kDenseTop = kHeld + kTeamWarps * kRingDepth * kStageBytes;
  static constexpr int kCols = kHeld + kHalfBytes;
  static constexpr int kHalfTiles = kCols + kWindowTokens * 8;
  static constexpr int kHalfLive = kHalfTiles + 2 * kHalfDigits * kTileWords * 4;
  static constexpr int kTop = kHalfLive + kHalfDigits / kGroupDigits * kWindowWords * 4;
  static constexpr int kRing = kTop + 2 * kHalfDigits * 4;
  static constexpr int kSpanTiles = kRing + kTeamWarps * kRingDepth * kStageBytes;
  static constexpr int kSpanLive = kSpanTiles + 2 * kSpanDigits * kTileWords * 4;
  static constexpr int kTables = kSpanLive + 2 * kSpanDigits / kGroupDigits * kWindowWords * 4;
  static constexpr int kMarks = kTables + kFactsWarps * kListTableKeys * 16;
  static constexpr int kBytes = kMarks + kFactsWarps * 48;
};

static_assert(FactsSmem::kDenseTop + 2 * 2 * kSpanDigits * 4 <= FactsSmem::kCols,
              "a dense list level's maxima fit");

//: the kernel's static shared memory beside it (the host checks the compiled kernel's own)
constexpr int kFactsStaticBytes = 1088;
static_assert(FactsSmem::kBytes + kFactsStaticBytes <= 99 * 1024,
              "the block's shared memory fits sm_86's 99 KB");

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
  uint32_t*
      gains16;  // [BH, L], a token's (m, F) as an f16 pair at its window's scales (facts.md#gains)
  int16_t* gains_exp;  // [BH, windows, 2], those scales' exponents
  long long* ledger;
  int D, H, L, token_words, windows, units, spans, rows_total, groups, tiles, list_level,
      read_list_level;
  int B[kMaxLevels];
  int row_base[kMaxLevels];
  int group_base[kMaxLevels];
  int plan_items[kMaxSpans];
  uint32_t plan[kMaxSpans][kMaxPlan];
  uint32_t dense[kSides];
  uint32_t mask[kSides][kMaxMaskWords];
};

//: a side's listed level: the write side's list level, the read side's read list level (-1 for none)
constexpr int listed_level(const FactsParams& p, int side) {
  return side == kWrite ? p.list_level : p.read_list_level;
}

}  // namespace rola::facts
