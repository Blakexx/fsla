// csrc/rola/src/intra/intra_api.cuh -- the intra's HOST entries for the pybind TU. See
// docs/internals/intra/intra_api.md
#pragma once

#include "common/torch_seam.cuh"

#include <cstdint>
#include <optional>
#include <vector>

namespace rola::intra {

void rola_intra_forward(std::vector<Tensor> read, std::vector<Tensor> write,
                        std::vector<Tensor> scale_write, Tensor g_write, Tensor v_hat,
                        std::optional<Tensor> v_lo, Tensor v_scale, Tensor unions, Tensor num,
                        Tensor den, std::vector<int64_t> widths, int64_t dense_read,
                        int64_t dense_write, int64_t weight_digits,
                        int64_t window);  // intra_api.md#rola-intra-forward

void rola_intra_ledger_bind(std::optional<Tensor> ledger);  // intra_api.md#rola-intra-ledger-bind

std::vector<std::vector<int64_t>> intra_arms();  // intra_api.md#intra-arms

int64_t intra_build_stamp();  // intra_api.md#intra-build-stamp

}  // namespace rola::intra
