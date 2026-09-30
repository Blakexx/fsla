// csrc/rola/src/intra/intra.cuh -- THE INTRA's constants and operand block: the within-window pairs, the Gram over the
// bundle's u8 rows, the value product over its s8 values, reduced into the output planes.
// see docs/internals/intra/intra.md
#pragma once

#include <cstdint>

namespace rola::intra {

//: THE WINDOW IS THE CARRY'S, so the two kernels run one grid; a window is eight 64-token tiles, walked in blocks of
//: two row tiles. -- see docs/internals/intra/intra.md#constants
constexpr int kWindow = 512;
constexpr int kTile = 64;
constexpr int kWindowTiles = kWindow / kTile;
constexpr int kRowTiles = 2;
constexpr int kBlockRows = kRowTiles * kTile;
constexpr int kBlocks = kWindowTiles / kRowTiles;
constexpr int kWarpRows = 16;
//: A PIECE: at most `kPieceTiles` of a block's writer tiles, one CTA's sweep, so no CTA carries a block's whole
//: diagonal; a block's pieces reduce their partial sums into the same rows. -- see docs/internals/intra/intra.md#pieces
constexpr int kPieceTiles = 4;

struct Piece {
  int block, t0, t1;
};

//: A WINDOW'S PIECES, longest first and the heavier block first among equals, so the long sweeps start first.
constexpr int piece_count() {
  int n = 0;
  for (int b = 0; b < kBlocks; ++b) n += ((b + 1) * kRowTiles + kPieceTiles - 1) / kPieceTiles;
  return n;
}

struct Pieces {
  Piece table[piece_count()]{};

  static constexpr int count() { return piece_count(); }

  constexpr Pieces() {
    int n = 0;
    for (int len = kPieceTiles; len > 0; --len)
      for (int b = kBlocks - 1; b >= 0; --b)
        for (int t0 = 0; t0 < (b + 1) * kRowTiles; t0 += kPieceTiles) {
          const int t1 =
              t0 + kPieceTiles < (b + 1) * kRowTiles ? t0 + kPieceTiles : (b + 1) * kRowTiles;
          if (t1 - t0 == len) table[n++] = Piece{b, t0, t1};
        }
  }
};

constexpr int kWarps = kBlockRows / kWarpRows;
constexpr int kThreads = kWarps * 32;
constexpr int kValueWidth = 64;
constexpr int kMaxLevels = 4;

//: A COPY GRANULE, an ldmatrix row, the column side's stage of digits, and a SLAB: one level's 32 digits, one IMMA
//: k-step, the slab gate's grain. -- see docs/internals/intra/intra.md#constants
constexpr int kChunk = 16;
constexpr int kStageDigits = 128;
constexpr int kSlabDigits = 32;
constexpr int kStageBuffers = 3;
constexpr int kCopyAhead = 1;

enum IntraPhase : int {
  kPhaseReader,
  kPhaseGate,
  kPhaseWait,
  kPhaseGram,
  kPhaseEpilogue,
  kPhaseValue,
  kPhaseDeposit,
  kIntraPhases
};

//: One level's rows or a value plane where the bundle holds them: the batch, token and head strides in elements.
struct RowView {
  const void* base;
  int64_t sb, st, sh;
};

struct ColumnView {
  const float* base;
  int64_t sb, st, sh;
};

//: THE OPERAND BLOCK, by value. -- see docs/internals/intra/intra.md#intraparams
struct IntraParams {
  RowView read[kMaxLevels];
  RowView write[kMaxLevels];
  ColumnView scale_write[kMaxLevels];
  ColumnView g_write, v_scale;
  RowView v_hat, v_lo;
  const uint32_t* unions;
  float* num;
  float* den;
  long long* ledger;
  int H, L, windows, weight_digits;
  uint32_t dense[2];
};

}  // namespace rola::intra
