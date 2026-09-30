// csrc/rola/src/produce/produce_api.cuh -- the producer's HOST entry for the pybind TU. See
// docs/internals/produce/produce_api.md
#pragma once

#include "common/torch_seam.cuh"

#include <cstdint>
#include <optional>

namespace rola::produce {

void rola_produce_ledger_bind(
    std::optional<Tensor> ledger);  // produce_api.md#rola-produce-ledger-bind

void rola_produce_forward(Tensor x, Tensor route_W, std::optional<Tensor> route_bias,
                          Tensor gain_bias, Tensor v_W, Tensor jobs, Tensor amp_read,
                          Tensor amp_write, Tensor scale_read, Tensor scale_write, Tensor mass_read,
                          Tensor mass_write, Tensor read_mass, Tensor g_write, Tensor v_hat,
                          std::optional<Tensor> v_lo, Tensor v_scale, Tensor arrivals,
                          std::optional<Tensor> logit_probe, std::optional<Tensor> level_probe,
                          Tensor shapes, int64_t levels,
                          int64_t route_jobs_per_head);  // produce_api.md#rola-produce-forward

}  // namespace rola::produce
