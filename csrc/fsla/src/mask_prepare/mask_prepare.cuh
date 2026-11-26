// THE PREFILL'S MASK-PREPARATION PASS, its entry points -- docs/internals/mask_prepare/mask_prepare.md
#pragma once

#include <cstdint>
#include <vector>

#include "common/torch_seam.cuh"

namespace fsla {
namespace mask_prepare {

//: the pass over a call: each side's leaf groups' mask words and counts into the workspace -- see docs/internals/mask_prepare/mask_prepare.md#entry
void mask_prepare(const std::vector<Tensor>& masks, const std::vector<int64_t>& slots,
                  const std::vector<int64_t>& widths, const std::vector<int64_t>& heads,
                  int64_t tokens, const Tensor& density, Tensor& workspace);

//: the bytes of the leaf groups' masks' layout, from the call's shape alone: its levels' digits, each side's masked levels (a bit a level) and batch-heads, its tokens a head
int64_t mask_prepare_bytes(const std::vector<int64_t>& widths, const std::vector<int64_t>& masked,
                           const std::vector<int64_t>& heads, int64_t tokens, int64_t device);

}  // namespace mask_prepare
}  // namespace fsla
