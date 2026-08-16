// THE FUSED PRODUCER -- host declarations. The kernels and their dispatch are
// `producer.cu`; the components they compose are `tile_gemm.cuh` and `solve_row.cuh`.
// Design of record: docs/internals/entmax/producer.md.
#pragma once

#include <cstdint>
#include <vector>

#include <torch/types.h>

namespace rola {
namespace entmax {

//: THE PROBE. Same tile-GEMM component as the shipping kernel, with the fp32 logit tile
//: emitted to a gate buffer instead of solved in place. Shipped and ratified so the
//: agreement it certifies is a CALL-GRAPH fact rather than a second implementation;
//: reached from tests only -- docs/internals/entmax/producer.md#probe
void producer_probe(const torch::Tensor& x, const torch::Tensor& route_W,
                    const c10::optional<torch::Tensor>& route_bias, torch::Tensor& logits);

}  // namespace entmax
}  // namespace rola
