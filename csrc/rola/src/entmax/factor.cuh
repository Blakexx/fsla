// THE PRODUCER'S INDEPENDENT (PER-SIDE) ROUTING SOLVES -- host declarations. The four
// kernels and the host dispatch are `factor.cu`; the warp-row vocabulary they share with
// the union solve is `entmax.cuh`.
//
// ONE ENTRY, MANY LEVELS. Each entry takes BASE tensors plus per-level element offsets,
// so a plan's levels that share a width class, an alpha and an output dtype run in ONE
// launch (`blockIdx.z` indexes the table) doing exactly the arithmetic the per-level
// loop did. Widths inside a launch may differ; their PADDED width may not.
//
// The solve, the batching rule and the mask contract: docs/internals/entmax/factor.md
#pragma once

#include <cstdint>
#include <vector>

#include <torch/types.h>

namespace rola {
namespace entmax {

//: `mask` is a detached boolean candidate mask laid out like `logits`; absent means every
//: element is a candidate. `values_second` absent means the level stores one side.
void factor_forward(const torch::Tensor& logits, const c10::optional<torch::Tensor>& mask,
                    torch::Tensor& values, const c10::optional<torch::Tensor>& values_second,
                    torch::Tensor& support_words, const std::vector<int64_t>& logit_offsets,
                    const std::vector<int64_t>& value_offsets,
                    const std::vector<int64_t>& support_offsets, const std::vector<int64_t>& widths,
                    double alpha);

//: `accumulate` adds into `d_logits` instead of storing -- what a TIED level's second
//: side needs, and the reason it is a parameter rather than a fact about the tensor.
void factor_backward(const torch::Tensor& values, const torch::Tensor& support_words,
                     const c10::optional<torch::Tensor>& mask, const torch::Tensor& d_values,
                     const c10::optional<torch::Tensor>& d_values_second, torch::Tensor& d_logits,
                     const std::vector<int64_t>& logit_offsets,
                     const std::vector<int64_t>& value_offsets,
                     const std::vector<int64_t>& support_offsets,
                     const std::vector<int64_t>& widths, bool accumulate, double alpha);

void softmax_forward(const torch::Tensor& logits, torch::Tensor& values,
                     const c10::optional<torch::Tensor>& values_second,
                     const std::vector<int64_t>& logit_offsets,
                     const std::vector<int64_t>& value_offsets, const std::vector<int64_t>& widths);

void softmax_backward(const torch::Tensor& values, const torch::Tensor& d_values,
                      const c10::optional<torch::Tensor>& d_values_second, torch::Tensor& d_logits,
                      const std::vector<int64_t>& logit_offsets,
                      const std::vector<int64_t>& value_offsets, const std::vector<int64_t>& widths,
                      bool accumulate);

}  // namespace entmax
}  // namespace rola
