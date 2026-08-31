// csrc/rola/src/carry/params.cuh -- THE PARAMETER BLOCK: the descriptor's addressing, the
// operands and the per-launch dials, handed to the kernel as ONE value.
// Each field's class in the primitive ledger: see docs/internals/carry/carry_kernel.md#ledger
#pragma once

#include <cuda_bf16.h>

#include <cstdint>

#include "common/geom.cuh"

namespace rola::carry {

//: THE FORM a phase runs in. -- see docs/internals/carry/carry_kernel.md#forms
enum class Form : int { kUnion = 0, kSparse = 1 };
constexpr int kForms = 2;

struct CarryParams {
  //: the problem's addressing, derived once on the host from (descriptor, launch).
  CarryGeomRT g;

  const __nv_bfloat16* pread;
  const __nv_bfloat16* pwrite;
  const __nv_bfloat16* gwrite;
  const __nv_bfloat16* v;
  float* num;
  float* den;
  const float* state_in;
  float* state_out;
  const int32_t* page_table;
  const int32_t* liveness;
  const uint8_t* activity;

  int L;               //: tokens in the call
  int liveness_words;  //: ceil(L / 32), the class-1 block's word stride
  int readout_form;    //: Form, the read side's
  int fold_form;       //: Form, the write side's
  //: `G` per side, as log2 LEAVES. -- see docs/internals/carry/carry_kernel.md#streams
  int readout_grain_bits;
  int fold_grain_bits;
};

}  // namespace rola::carry
