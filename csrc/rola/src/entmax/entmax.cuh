// THE PRODUCER'S UNION ENTMAX SOLVE -- host declarations. The implementation, both
// kernels and the host dispatch, is `entmax.cu`.
//
// THE STRUCTURAL IDEA: ONE ROW PER (LOGICAL) WARP. Every collective in the solve runs
// along one token's row, so all of them are warp-scope and shuffle-based -- which is
// what makes register demand O(items-per-lane), the warp count a free occupancy knob,
// and the shared-memory need zero.
//
// THE SUPPORT WORD LAYOUT IS FROZEN (the per-leaf write bitmap has nine downstream
// consumers): one CTA owns exactly the 32 tokens of one support word.
//
// Why this is CUDA and not Triton, the measured occupancy history, and the whole
// solve: docs/internals/entmax/entmax.md
#pragma once

#include <cstdint>

#include <torch/types.h>

namespace rola {
namespace entmax {

//: One launch = one union routing LEVEL = one width class = one alpha; the caller's
//: per-level loop supplies the levels -- docs/internals/entmax/entmax.md#launch-shape
//:
//: `read_values`/`write_values` accept BOTH route-value layouts (rank-3 dense,
//: rank-4 tile-major) through ONE stride 5-tuple, so layout is data here and never a
//: codegen axis -- docs/internals/entmax/entmax.md#strides
void union_forward(const torch::Tensor& read_logits, const torch::Tensor& write_logits,
                   torch::Tensor& midpoint_values, torch::Tensor& support_words,
                   torch::Tensor& read_values, torch::Tensor& write_values, double alpha);

void union_backward(const torch::Tensor& read_logits, const torch::Tensor& write_logits,
                    const torch::Tensor& midpoint_values, const torch::Tensor& support_words,
                    const torch::Tensor& read_values, const torch::Tensor& write_values,
                    const torch::Tensor& d_read, const torch::Tensor& d_write,
                    torch::Tensor& d_read_logits, torch::Tensor& d_write_logits, double alpha);

}  // namespace entmax
}  // namespace rola
