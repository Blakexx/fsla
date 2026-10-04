// csrc/rola/src/facts/facts_api.cuh -- the facts pass's HOST entry for the pybind TU. See
// docs/internals/facts/facts_api.md
#pragma once

#include "common/torch_seam.cuh"

#include <cstdint>
#include <optional>
#include <vector>

namespace rola::facts {

void rola_facts_ledger_bind(std::optional<Tensor> ledger);  // facts_api.md#rola-facts-ledger-bind

void rola_facts_forward(std::vector<Tensor> read, std::vector<Tensor> write,
                        std::vector<Tensor> scale_write, Tensor g_write, Tensor v_scale,
                        Tensor words, Tensor unions, Tensor list_exp, std::vector<int64_t> widths,
                        int64_t dense_read, std::vector<int64_t> mask_read, int64_t dense_write,
                        std::vector<int64_t> mask_write, int64_t list_level,
                        int64_t window);  // facts_api.md#rola-facts-forward

}  // namespace rola::facts
