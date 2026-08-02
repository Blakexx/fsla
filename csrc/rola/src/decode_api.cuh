#pragma once

// THE DECODE PATH'S HOST-FACING DECLARATIONS, and the ONLY decode header
// `csrc/rola/rola_api.cpp` includes.
//
// It is split from `decode.cuh` for a mechanical reason worth stating, because the
// split looks gratuitous until you hit it: `rola_api.cpp` is HOST-ONLY and is compiled
// by the host compiler, not by `nvcc`. `decode.cuh` carries `__device__` helpers (the
// bitmap funnel shift, the set-bit boundary cuts) that the host compiler cannot parse,
// so a binding TU that included it would fail to compile. Guarding those helpers with
// `#ifdef __CUDACC__` would work and would be worse: the header would then mean two
// different things to two compilers. One header per audience instead.

#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace rola {
namespace decode {

//: THE BOUND SURFACE. `rola_decode_capacity` exists so a Python test can assert the
//: C++ level-width capacity against `rola.routing.topology.MAX_BRANCH_WIDTH`: a
//: `static_assert` cannot read a Python constant, so the pair of them is what makes
//: the producer/decode capacity relation checkable rather than merely written down.
int64_t rola_decode_capacity();
int64_t rola_decode_producer_width_mirror();

at::Tensor rola_decode_forward(
    std::vector<at::Tensor> amp_read,
    std::vector<at::Tensor> amp_write,
    c10::optional<std::vector<at::Tensor>> amp_clock,
    c10::optional<at::Tensor> dials,
    c10::optional<at::Tensor> g_read,
    at::Tensor g_write,
    at::Tensor v,
    at::Tensor state,
    at::Tensor bits,
    at::Tensor pc_coarse,
    at::Tensor ws,
    at::Tensor ctr,
    std::vector<int64_t> level_widths,
    std::vector<int64_t> level_strides,
    std::vector<int64_t> level_spans,
    std::vector<int64_t> level_row_offsets,
    int64_t heads,
    int64_t n_split,
    double eps,
    bool global_norm);

}  // namespace decode
}  // namespace rola
