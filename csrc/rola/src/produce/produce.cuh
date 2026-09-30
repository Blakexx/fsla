#pragma once

// THE PRODUCER: one launch from a stretch of tokens to the bundle -- every routing level's projection, solve and
// unit-max rows, the write gain, and the value digits. The layer's declared routing compiles once into the job table
// this reads; every width and kind is a runtime field of it, so the family is one instantiation.
// see docs/internals/produce/produce.md

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace rola {
namespace produce {

constexpr int kProduceThreads = 256;
constexpr int kProduceWarps = kProduceThreads / 32;
//: The tokens a block carries through the projection: each weight element it streams feeds this many dot products.
constexpr int kTokenTile = 4;
//: A level's widest row; a union job reads two logit rows of it.
constexpr int kMaxWidth = 256;
//: A job's columns: one level's row, both of a union's, or a value job's heads.
constexpr int kMaxColumns = 2 * kMaxWidth;

//: THE JOB TABLE, one row of int32 per job -- see docs/internals/produce/produce.md#jobs
enum JobField : int {
  kFieldKind,
  kFieldHead,
  kFieldHeads,
  kFieldSolve,
  kFieldWidth,
  kFieldIn0,
  kFieldIn1,
  kFieldOutRead,
  kFieldOutWrite,
  kFieldLevel,
  kFieldFoldRead,
  kFieldFoldWrite,
  kJobFields
};

enum JobKind : int { kJobRoute = 0, kJobValue = 1 };

//: THE SOLVES a route job runs on its row: the closed registry's kerneled activations.
enum SolveKind : int { kSoftmax = 0, kEntmax15 = 1, kEntmax20 = 2, kUnion15 = 3, kUnion20 = 4 };

enum RowForm : int { kRowsU8 = 0, kRowsF16 = 1 };

struct Job {
  int kind, head, heads, solve, width, in0, in1, out_read, out_write, level, fold_read, fold_write;
};

//: THE OPERANDS, as addresses: every tensor the host entry validated, with the strides it read off them.
struct ProduceParams {
  const __nv_bfloat16* x;  // [BT, hidden]
  //: [H, hidden, C]: (h, k, c) at h * w_head + k * w_k + c -- the columns of one (head, k) are contiguous.
  const __nv_bfloat16* route_W;
  int64_t w_head, w_k;
  const float* route_bias;  // [H, C] or null
  const float* gain_bias;   // [H, G]
  //: [H * d_v, hidden] with unit stride on the channel axis: (c, k) at k * v_k + c.
  const __nv_bfloat16* v_W;
  int64_t v_k;
  const int32_t* jobs;  // [n_jobs, kJobFields]

  uint8_t* amp_read;  // [BT, H, rows], u8 or f16
  uint8_t* amp_write;
  float* scale_read;  // [BT, H, D]
  float* scale_write;
  float* mass_read;  // [BT, H, D], the finisher's operands
  float* mass_write;
  float* read_mass;   // [BT, H]
  float* g_write;     // [BT, H]
  int8_t* v_hat;      // [BT, H, d_v]
  uint8_t* v_lo;      // [BT, H, d_v] or null (one digit)
  float* v_scale;     // [BT, H]
  int32_t* arrivals;  // [tiles, H], zero between calls: the finisher resets its own

  int BT, hidden, H, C, G, rows, D, d_v, row_form, route_jobs_per_head;
};

}  // namespace produce
}  // namespace rola
