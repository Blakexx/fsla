#pragma once

// THE DECODE PATH'S HOST-FACING DECLARATIONS, and the ONLY decode header
// `csrc/rola/rola_api.cpp` includes.
// see docs/internals/decode/decode_api.md#note-l3

#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace rola {
namespace decode {

//: THE BOUND SURFACE. The two accessors exist so a Python test can assert the
//: C++ level-width capacity against `rola.routing.topology.MAX_BRANCH_WIDTH`
//: ...
// see docs/internals/decode/decode_api.md#rola-decode-capacity
int64_t rola_decode_capacity();
int64_t rola_decode_producer_width_mirror();

//: THE STEP'S DECLARED RESIDENCY in CTAs per SM, so the split policy sizes one wave from
//: the launch bound itself rather than from a python mirror of it:
//: docs/internals/decode/decode_api.md#residency
int64_t rola_decode_residency(bool decay, int64_t levels);

//: THE DEVICE-SIDE BUILD STAMP, a fact only the loaded binary can produce. A path check
//: or a hash of the sources cannot catch a stale `.so` at the right path:
//: docs/internals/decode/decode_api.md#build-stamp
int64_t rola_decode_build_stamp();

//: THE ARMS THIS BINARY CARRIES, `[d_v, D, decay]` per row. A full build lists the whole
//: declared matrix; an ITERATION build (`ROLA_DECODE_ARMS`) lists its subset and is not
//: shippable: docs/internals/decode/decode_api.md#arm-subset
std::vector<std::vector<int64_t>> rola_decode_arms();

//: THE WHOLE STEP IN ONE LAUNCH -- fold, factor, walk, AND ITS OWN ADMISSION
//: QUESTION -- over the caller's RAW `[B, 1, H, *]` views.
// see docs/internals/decode/decode_api.md#note-l43
at::Tensor rola_decode_forward(
    std::vector<at::Tensor> read, std::vector<at::Tensor> write, std::vector<int64_t> normalize,
    c10::optional<at::Tensor> dials, at::Tensor g_write, at::Tensor v, at::Tensor state,
    at::Tensor ws, at::Tensor ctr, at::Tensor growth, at::Tensor growth_any, at::Tensor growth_ctr,
    at::Tensor done, at::Tensor atom_bits, std::vector<int64_t> level_widths,
    std::vector<int64_t> level_row_offsets, int64_t lattice_k, int64_t lattice_m, int64_t n_split,
    double eps, c10::optional<at::Tensor> page_table, c10::optional<at::Tensor> pool_slots,
    c10::optional<at::Tensor> pool_cursor, c10::optional<at::Tensor> pool_map);

}  // namespace decode
}  // namespace rola
