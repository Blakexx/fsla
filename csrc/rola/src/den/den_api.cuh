// csrc/rola/src/den/den_api.cuh -- the den pass's HOST entries for the pybind TU.
// see docs/internals/den/den_api.md
#pragma once

#include "common/torch_seam.cuh"

#include <cstdint>
#include <optional>
#include <vector>

namespace rola::den {

//: ONE LAUNCH over a call's windows. -- see docs/internals/den/den_api.md#den-forward
void den_forward(std::vector<Tensor> read_rows, std::vector<Tensor> write_rows,
                 std::vector<Tensor> read_lists, std::vector<Tensor> write_lists, Tensor list_exp,
                 Tensor gains16, Tensor gains_exp, Tensor den, Tensor num,
                 std::vector<int64_t> widths, int64_t level_s, int64_t read_list,
                 int64_t write_list, int64_t heads);

void den_ledger_bind(const std::optional<Tensor>& ledger);  // den_api.md#den-ledger-bind

}  // namespace rola::den
