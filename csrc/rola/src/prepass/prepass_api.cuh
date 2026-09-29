// csrc/rola/src/prepass/prepass_api.cuh -- the pre-pass's host entries, declared for the seam.
// See docs/internals/prepass/prepass.md#api
#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "common/torch_seam.cuh"

namespace rola::prepass {

//: prepass_api.md
void prepass_forward(const Tensor& rows_read, const Tensor& rows_write, const Tensor& scale_read,
                     const Tensor& scale_write, const Tensor& gain, const Tensor& vscale,
                     Tensor& den, Tensor& scale, const std::vector<int64_t>& widths,
                     int64_t level_modes, const Tensor& liveness,
                     const std::optional<Tensor>& fold_dump);

void prepass_ledger_bind(const std::optional<Tensor>& ledger);

std::vector<int64_t> prepass_plan(int64_t depth);

}  // namespace rola::prepass
