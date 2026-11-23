// THE PRODUCER'S INDEPENDENT (PER-SIDE) ROUTING SOLVES -- host declarations; the four
// kernels and the host dispatch are `factor.cu`. ONE ENTRY, MANY LEVELS: a plan's
// levels sharing a width class, alpha and output dtype run in ONE launch
// (`blockIdx.z` indexes the table). Full solve, batching rule and mask contract:
// see docs/internals/entmax/factor.md
#pragma once

#include <cstdint>
#include <vector>

#include "common/torch_seam.cuh"

namespace fsla {
namespace entmax {

//: THE ARMS THIS BINARY CARRIES, a row each: `[lw, ipt, alpha15, logit, value]` (`[lw, ipt, logit, value]` for the softmax kernels), the dtypes as the contract's codes
std::vector<std::vector<int64_t>> factor_forward_arms();

std::vector<std::vector<int64_t>> factor_backward_arms();

std::vector<std::vector<int64_t>> softmax_forward_arms();

std::vector<std::vector<int64_t>> softmax_backward_arms();

//: `mask` is a detached boolean candidate mask laid out like `logits`; absent means every -- see docs/internals/entmax/factor.md#factorforward
void factor_forward(const Tensor& logits, const std::optional<Tensor>& mask, Tensor& values,
                    const std::optional<Tensor>& values_second, Tensor& support_words,
                    const std::vector<int64_t>& logit_offsets,
                    const std::vector<int64_t>& value_offsets,
                    const std::vector<int64_t>& support_offsets, const std::vector<int64_t>& widths,
                    int64_t heads, double alpha);

//: `accumulate` adds into `d_logits` instead of storing -- what a TIED level's second -- see docs/internals/entmax/factor.md#factorbackward
void factor_backward(const Tensor& values, const Tensor& support_words,
                     const std::optional<Tensor>& mask, const Tensor& d_values,
                     const std::optional<Tensor>& d_values_second, Tensor& d_logits,
                     const std::vector<int64_t>& logit_offsets,
                     const std::vector<int64_t>& value_offsets,
                     const std::vector<int64_t>& support_offsets,
                     const std::vector<int64_t>& widths, bool accumulate, int64_t heads,
                     double alpha);

void softmax_forward(const Tensor& logits, Tensor& values,
                     const std::optional<Tensor>& values_second,
                     const std::vector<int64_t>& logit_offsets,
                     const std::vector<int64_t>& value_offsets, const std::vector<int64_t>& widths,
                     int64_t heads);

void softmax_backward(const Tensor& values, const Tensor& d_values,
                      const std::optional<Tensor>& d_values_second, Tensor& d_logits,
                      const std::vector<int64_t>& logit_offsets,
                      const std::vector<int64_t>& value_offsets, const std::vector<int64_t>& widths,
                      bool accumulate, int64_t heads);

}  // namespace entmax
}  // namespace fsla
