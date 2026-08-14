// csrc/rola/src/chunk/chunk_api.cuh -- the chunk consumer's host entry
// declarations for the pybind TU (the host_api.cuh convention: declaration
// only, so the binding TU never sees a device body).
#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace rola::chunk {

torch::Tensor union_table(torch::Tensor pr, torch::Tensor pw, torch::Tensor v, int64_t chunk,
                          int64_t bc, std::vector<int64_t> widths);
std::vector<torch::Tensor> forward_raw(torch::Tensor pr, torch::Tensor pw, torch::Tensor gw,
                                       torch::Tensor v, torch::Tensor tbl, int64_t chunk,
                                       bool swizzle, int64_t bc, std::vector<int64_t> widths);
torch::Tensor forward(torch::Tensor pr, torch::Tensor pw, torch::Tensor gw, torch::Tensor v,
                      int64_t chunk, bool swizzle, int64_t bc, std::vector<int64_t> widths,
                      double eps);
torch::Tensor divide_pass(torch::Tensor num, torch::Tensor den, double eps);
std::vector<int64_t> build_stamp(int64_t chunk, bool swizzle, int64_t bc,
                                 std::vector<int64_t> widths);
std::vector<std::vector<int64_t>> arms();

}  // namespace rola::chunk
