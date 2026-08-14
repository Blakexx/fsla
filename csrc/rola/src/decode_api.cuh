#pragma once

// THE DECODE PATH'S HOST-FACING DECLARATIONS, and the ONLY decode header
// `csrc/rola/rola_api.cpp` includes.
//
// The split from `decode.cuh` is mechanical and looks gratuitous until you hit it:
// `rola_api.cpp` is HOST-ONLY, and `decode.cuh` carries `__device__` helpers the host
// compiler cannot parse. ONE HEADER PER AUDIENCE, rather than an `#ifdef __CUDACC__`
// that would make one header mean two things to two compilers.
//
// The full argument, and what each bound symbol claims: docs/internals/decode_api.md

#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace rola {
namespace decode {

//: THE BOUND SURFACE. The two accessors exist so a Python test can assert the C++
//: level-width capacity against `rola.routing.topology.MAX_BRANCH_WIDTH` -- a
//: `static_assert` cannot read a Python constant:
//: docs/internals/decode_api.md#bound-surface
int64_t rola_decode_capacity();
int64_t rola_decode_producer_width_mirror();

at::Tensor rola_decode_forward(std::vector<at::Tensor> amp_read, std::vector<at::Tensor> amp_write,
                               c10::optional<std::vector<at::Tensor>> amp_clock,
                               c10::optional<at::Tensor> dials, at::Tensor g_write, at::Tensor v,
                               at::Tensor state, at::Tensor bits, at::Tensor pc_coarse,
                               at::Tensor ws, at::Tensor ctr, std::vector<int64_t> level_widths,
                               std::vector<int64_t> level_strides, std::vector<int64_t> level_spans,
                               std::vector<int64_t> level_row_offsets, int64_t heads,
                               int64_t n_split, double eps, c10::optional<at::Tensor> page_table);

}  // namespace decode
}  // namespace rola
