// csrc/rola/src/carry/carry_arm_abi.cuh -- the seam between the dispatch and an arm.
//
// K36 gave every carry arm its own translation unit (KERNEL_STANDARDS section 7), so
// the dispatch reaches an arm's body through NON-TEMPLATE functions it can declare and
// not define.  This header is that declaration, and it is deliberately the SMALLEST
// header in the family: `cuda_runtime`, `cuda_bf16`, `cstdint`, nothing else.
//
// WHY IT SPEAKS POINTERS AND NOT TENSORS.  An arm TU that parses the tensor library's
// extension header costs 33.7 s and 2.9 GB of `cicc` before it compiles one
// instruction of the arm; the arm's own codegen is 6 s (measured, K36 -- a bare CUDA
// TU is 0.9 s and 0.19 GB).  With one TU per arm that fixed cost is paid 22 times and
// it also sets the per-TU resident set, which is what caps `MAX_JOBS`.  So EVERY
// ALLOCATION IS THE DISPATCH'S: `carry.cu` sizes the ballot table from
// `carry_arm_rows_entries_<i>`, and the arm TU only launches into memory it was
// handed.
// docs/internals/carry/carry_arm.md.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace rola::carry {

//: the census's width, so the dispatch's buffer and the arm's writer agree by name.
static constexpr int kCarryCensusFields = 23;
//: the `arms()` row's width: the nine declaration fields plus BC, owners, smem and the
//: read side's carve level.
static constexpr int kCarryRowFields = 13;

//: THE STATE PLANES AND THE TABLE ARE THE CALLER'S, which is the shipped contract
//: (docs/internals/state.md section 4): a continuation hands the SAME plane as `s_in` and
//: `s_out` and advances it in place, and `page_tbl` null is the dense backing -- the slot
//: is then the atom.  BOTH PLANES NULL IS THE NULL-STATE CALL and reaches an arm variant
//: nothing is read, written, or allocated at either end (state.md section 1) -- a
//: RUNTIME null check inside the one compiled body (K41; no STATE template axis).
//:
//: WHAT ONE ARM'S TRANSLATION UNIT EXPORTS.  Five entry points: launch the body,
//: launch and size its ballot table, and state its census and its row.
//: `CARRY_ARM_DEFINE` (`carry_arm.cuh`) defines exactly these.  The SEGMENT LAYOUT
//: PROBE left with the per-stream segments and the exchange it gated (K35 M1P).
#define CARRY_ARM_DECLARE(I_, ...)                                                            \
  cudaError_t carry_arm_forward_##I_(                                                          \
      const __nv_bfloat16* pr, const __nv_bfloat16* pw, const __nv_bfloat16* gw,               \
      const __nv_bfloat16* v, const uint32_t* rows, float* o, float* den, int BH, int L,       \
      int tok_words, int v_row_bytes, const float* s_in, float* s_out, const int32_t* page_tbl, \
      cudaStream_t stream);                                                                     \
  cudaError_t carry_arm_rows_##I_(const __nv_bfloat16* pr, const __nv_bfloat16* pw,            \
                                  uint32_t* tbl, int BH, int L, cudaStream_t stream);          \
  int carry_arm_rows_entries_##I_();                                                           \
  void carry_arm_census_##I_(int64_t* out);                                                    \
  void carry_arm_row_##I_(int64_t* out);

}  // namespace rola::carry
