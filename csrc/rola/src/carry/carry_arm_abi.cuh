// csrc/rola/src/carry/carry_arm_abi.cuh -- the dispatch/arm seam: NON-TEMPLATE
// functions the dispatch declares and not defines, speaking POINTERS not
// `torch::Tensor` (every allocation is the dispatch's).
// See docs/internals/carry/carry_arm_abi.md
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace rola::carry {

static constexpr int kCarryCensusFields = 23;  // the dispatch buffer/arm writer's shared width
static constexpr int kCarryRowFields =
    13;  // see docs/internals/carry/carry_arm_abi.md#kcarryrowfields

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
