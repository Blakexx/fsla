// THE PREFILL'S MASK-PREPARATION PASS, its entry points -- docs/internals/mask_prepare/mask_prepare.md
#pragma once

#include <cstdint>
#include <vector>

#include "common/torch_seam.cuh"

namespace fsla {
namespace mask_prepare {

//: the pass over a call's masks: `masks[i]` of level-side `slots[i]`, `density` the call's `[2, D]` uint8
void mask_prepare(const std::vector<Tensor>& masks, const std::vector<int64_t>& slots,
                  const Tensor& density, Tensor& workspace);

//: the bytes of the prepared masks' layout, from the call's shape alone
int64_t mask_prepare_bytes(const std::vector<int64_t>& width_bits,
                           const std::vector<int64_t>& heads, int64_t words);

}  // namespace mask_prepare
}  // namespace fsla
