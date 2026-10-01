#pragma once

// THE PRODUCER: one launch from a stretch of tokens to the bundle, every width and kind a runtime field of the job
// table the layer's declared routing compiles into. see docs/internals/produce/produce.md

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace rola {
namespace produce {

constexpr int kProduceThreads = 256;
constexpr int kProduceWarps = kProduceThreads / 32;
//: A block's tokens: four MMA token groups. -- see docs/internals/produce/produce.md#accumulate
constexpr int kTokenTile = 32;
constexpr int kTokenGroups = kTokenTile / 8;
//: Four lanes' 16-byte chunks: two MMA k-steps.
constexpr int kKBlock = 32;
//: A solve row's widest; a job holds two (a union's) or a value job's heads.
constexpr int kMaxWidth = 256;
constexpr int kMaxColumns = 2 * kMaxWidth;
//: Four words past the columns: an accumulator store's 32 lanes hit 32 banks.
constexpr int kLogitStride = kMaxColumns + 4;
constexpr int kLogitBytes = kTokenTile * kLogitStride * static_cast<int>(sizeof(float));
//: Where a sliced projection's partials sit in each token row: past the columns a sliced job has (at most 64).
constexpr int kPartialBase = 64;
//: The threshold's Newton steps at most: rows of every measured distribution settle in 8 or fewer.
constexpr int kNewtonSteps = 16;
//: A SPARSE LEVEL's term slot: a window of tokens, and the token tiles in it. -- see docs/internals/produce/produce.md#sparse
constexpr int kTermWindow = 512;
constexpr int kWindowTiles = kTermWindow / kTokenTile;

//: THE JOB TABLE, one row of int32 per job: its addressing. Its control rides the parameters as a shape word.
//: -- see docs/internals/produce/produce.md#jobs
enum JobField : int {
  kFieldHead,
  kFieldHeads,
  kFieldWidth,
  kFieldIn0,
  kFieldIn1,
  kFieldOutRead,
  kFieldOutWrite,
  kFieldLevel,
  kJobFields
};

enum JobKind : int { kJobRoute = 0, kJobValue = 1 };

//: A launch's jobs at most: one shape word each in the parameter space.
constexpr int kMaxJobs = 1024;

//: THE SOLVES a route job runs on its row: the closed registry's kerneled activations.
enum SolveKind : int { kSoftmax = 0, kEntmax15 = 1, kEntmax20 = 2, kUnion15 = 3, kUnion20 = 4 };

enum RowForm : int { kRowsU8 = 0, kRowsF16 = 1 };

//: THE PHASE LEDGER's phases, a warp's cycles each, booked by job over the tiles. -- see docs/internals/produce/produce.md#phase-ledger
enum ProducePhase : int {
  kPhaseJob,
  kPhaseProject,
  kPhaseBarrier,
  kPhaseRowLoad,
  kPhaseThreshold,
  kPhaseLevels,
  kPhaseBundle,
  kPhaseFinish,
  kPhaseValues,
  kProducePhases
};

//: A JOB: its control (the shape word's fields) and its addressing (the table's). -- see docs/internals/produce/produce.md#jobs
struct Job {
  int kind, solve, rows, reads, writes, head, heads, width, in0, in1, out_read, out_write, level,
      fold_read, fold_write, sparse_read, sparse_write;
};

//: A BLOCK'S TOKENS: a sequence's tile, or the flat batch's at one token. -- see docs/internals/produce/produce.md#jobs
struct Tile {
  int token0, live, b, window, slot_tile;
};

//: THE OPERANDS, as addresses: every tensor the host entry validated, with the strides it read off them.
struct ProduceParams {
  const __nv_bfloat16* x;        // [BT, hidden]
  const __nv_bfloat16* route_W;  // [H, hidden, C], (h, k, c) at h * w_head + c * w_col + k
  int64_t w_head, w_col;
  const float* route_bias;   // [H, C] or null
  const float* gain_bias;    // [H, G]
  const __nv_bfloat16* v_W;  // [H * d_v, hidden], (c, k) at c * v_row + k
  int64_t v_row;
  const int32_t* jobs;  // [n_jobs, kJobFields]

  uint8_t* amp_read;  // the dense levels [B, H, T, rows], head-major (produce.md#bundle)
  uint8_t* amp_write;
  int32_t* words_read;  // the sparse levels' words and terms (produce.md#sparse)
  int32_t* words_write;
  uint8_t* terms_read;
  uint8_t* terms_write;
  int32_t* flags;     // [jobs, 2, B, windows, 2], zero between calls
  float* scale_read;  // [B, H, T, D]
  float* scale_write;
  float* mass_read;  // [B, H, T, D], the finisher's operands
  float* mass_write;
  float* read_mass;    // [B, H, T]
  float* g_write;      // [B, H, T]
  int8_t* v_hat;       // [B, H, T, d_v]
  uint8_t* v_lo;       // [B, H, T, d_v] or null (one digit)
  float* v_scale;      // [B, H, T]
  int32_t* arrivals;   // [tiles, H], zero between calls: the finisher resets its own
  float* logit_probe;  // [jobs, BT, kMaxColumns] or null: every job's projection, for its gate
  float* level_probe;  // the same, a route job's solved levels: read at [0, w), write at [w, 2w)
  long long* ledger;   // [jobs, warps, phases] or null
  int16_t shapes[kMaxJobs];  // a job's shape word (produce.md#jobs), by block index

  int BT, T, hidden, H, C, G, rows_read, rows_write, D, d_v, row_form, route_jobs_per_head;
  int B, tiles_per_seq, nw, windows, slot, sparse_rows_read, sparse_rows_write;
};

}  // namespace produce
}  // namespace rola
