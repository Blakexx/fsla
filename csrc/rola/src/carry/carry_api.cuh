// csrc/rola/src/carry/carry_api.cuh -- the K31 carry family's host entry
// declarations for the pybind TU, DELIBERATELY OUTSIDE the shipped dispatch.
// See docs/internals/carry/carry_api.md
#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace rola::carry {

std::vector<torch::Tensor> forward_inter(torch::Tensor pr, torch::Tensor pw, torch::Tensor gw,
                                         torch::Tensor v, std::vector<int64_t> widths, int64_t k,
                                         int64_t m, int64_t window, int64_t chunk, int64_t carve,
                                         int64_t level_modes, c10::optional<torch::Tensor> s_in,
                                         c10::optional<torch::Tensor> s_out,
                                         c10::optional<torch::Tensor> page_tbl);
// the INTER term only, undivided -- see docs/internals/carry/carry_api.md#note-l14

torch::Tensor owner_rows(torch::Tensor pr, torch::Tensor pw, std::vector<int64_t> widths, int64_t k,
                         int64_t m,
                         bool carve);  // the ballot rows, packed bits; see carry_api.md#note-l35

std::vector<int64_t> build_stamp(
    std::vector<int64_t> widths, int64_t k, int64_t m, int64_t window, int64_t chunk, int64_t carve,
    int64_t level_modes,
    int64_t d_v);  // device build identity + census; see carry_api.md#note-l42

std::vector<torch::Tensor> backward_inter(torch::Tensor pr, torch::Tensor pw, torch::Tensor gw,
                                          torch::Tensor vt, torch::Tensor dn,
                                          std::vector<int64_t> widths, int64_t k, int64_t m,
                                          int64_t window, int64_t chunk,
                                          c10::optional<torch::Tensor> s_in,
                                          c10::optional<torch::Tensor> ds_out,
                                          c10::optional<torch::Tensor> page_tbl, bool want_replay);
// R4's two reverse scans -- see docs/internals/carry/carry_api.md#backwardinter

std::vector<int64_t> backward_build_stamp(
    std::vector<int64_t> widths, int64_t k, int64_t m, int64_t window,
    int64_t chunk);  // the reverse family's device stamp; see carry_api.md#backwardbuildstamp

std::vector<std::vector<int64_t>> arms();  // the built (widths, k, m, ...) rows

}  // namespace rola::carry
