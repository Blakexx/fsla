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
                                       int64_t bc, std::vector<int64_t> widths,
                                       c10::optional<torch::Tensor> s_in,
                                       c10::optional<torch::Tensor> s_out,
                                       c10::optional<torch::Tensor> page_tbl,
                                       c10::optional<torch::Tensor> block_bits);
torch::Tensor forward(torch::Tensor pr, torch::Tensor pw, torch::Tensor gw, torch::Tensor v,
                      int64_t chunk, int64_t bc, std::vector<int64_t> widths, double eps,
                      c10::optional<torch::Tensor> read_mass, at::ScalarType out,
                      c10::optional<torch::Tensor> s_in, c10::optional<torch::Tensor> s_out,
                      c10::optional<torch::Tensor> page_tbl,
                      c10::optional<torch::Tensor> block_bits);
torch::Tensor stats(torch::Tensor pw, std::vector<int64_t> widths);
torch::Tensor block_bits(torch::Tensor tbl, int64_t bc, std::vector<int64_t> widths, int64_t L,
                         c10::optional<torch::Tensor> page_tbl);
torch::Tensor divide_pass(torch::Tensor num, torch::Tensor den, c10::optional<torch::Tensor> mass,
                          double eps, at::ScalarType out);
std::vector<int64_t> build_stamp(int64_t chunk, int64_t bc, std::vector<int64_t> widths);
std::vector<std::vector<int64_t>> arms();
//: SCAN B (csrc/rola/src/chunk/backward.cu), the reverse adjoint's stage-1
//: surface: the terms it carries and the ones it does not are the header's.
std::vector<torch::Tensor> scan_b_probe(torch::Tensor pr, torch::Tensor pw, torch::Tensor gw,
                                        torch::Tensor v, torch::Tensor dnum, torch::Tensor dden,
                                        torch::Tensor tbl, int64_t chunk, int64_t bc,
                                        std::vector<int64_t> widths);
std::vector<torch::Tensor> walk_probe(torch::Tensor tbl, int64_t chunk, int64_t bc,
                                      std::vector<int64_t> widths, int64_t L);
std::vector<std::vector<int64_t>> scan_b_census();

}  // namespace rola::chunk
