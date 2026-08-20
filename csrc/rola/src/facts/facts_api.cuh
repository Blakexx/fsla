// csrc/rola/src/facts/facts_api.cuh -- the facts layer's host entry
// declarations for the pybind TU (the host_api.cuh convention: declaration
// only, so the binding TU never sees a device body).
#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace rola::chunk {

std::pair<torch::Tensor, c10::optional<torch::Tensor>> facts(torch::Tensor pr, torch::Tensor pw,
                                                             int64_t chunk, int64_t bc,
                                                             std::vector<int64_t> widths,
                                                             int64_t mask, bool atoms);
torch::Tensor block_bits(torch::Tensor tbl, int64_t bc, std::vector<int64_t> widths, int64_t L,
                         c10::optional<torch::Tensor> page_tbl);
std::vector<std::vector<int64_t>> arms();

}  // namespace rola::chunk
