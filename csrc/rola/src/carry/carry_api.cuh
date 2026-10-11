// csrc/rola/src/carry/carry_api.cuh -- the carry family's HOST entries for the pybind TU.
// see docs/internals/carry/carry_api.md
#pragma once

#include "common/torch_seam.cuh"

#include <cstdint>
#include <optional>
#include <vector>

namespace rola::carry {

//: ONE LAUNCH over a call's windows. -- see docs/internals/carry/carry_api.md#carry-forward
void carry_forward(std::vector<Tensor> read_rows, std::vector<Tensor> write_rows, Tensor values,
                   std::vector<Tensor> write_lists, Tensor write_exp,
                   std::vector<Tensor> read_lists, Tensor unit, Tensor num, Tensor check,
                   std::vector<int64_t> widths, std::vector<int64_t> extents, int64_t channels,
                   int64_t write_list, std::vector<int64_t> write_operand, int64_t read_list,
                   std::vector<int64_t> read_operand, std::vector<int64_t> key);

void carry_ledger_bind(const std::optional<Tensor>& ledger);
void carry_probe_bind(const std::optional<Tensor>& probe);
int64_t carry_build_stamp();
std::vector<std::vector<int64_t>> carry_census();
std::vector<std::vector<int64_t>> carry_arms();

}  // namespace rola::carry
