#pragma once

// THE DECODE PATH'S HOST-FACING DECLARATIONS, and the ONLY decode header
// `csrc/rola/rola_api.cpp` includes.
//
// The split from `decode.cuh` is mechanical and looks gratuitous until you hit it:
// `rola_api.cpp` is HOST-ONLY, and `decode.cuh` carries `__device__` helpers the host
// compiler cannot parse. ONE HEADER PER AUDIENCE, rather than an `#ifdef __CUDACC__`
// that would make one header mean two things to two compilers.
//
// The full argument, and what each bound symbol claims: docs/internals/decode/decode_api.md

#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace rola {
namespace decode {

//: THE BOUND SURFACE. The two accessors exist so a Python test can assert the C++
//: level-width capacity against `rola.routing.topology.MAX_BRANCH_WIDTH` -- a
//: `static_assert` cannot read a Python constant:
//: docs/internals/decode/decode_api.md#bound-surface
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

//: THE WHOLE STEP IN ONE LAUNCH -- fold, factor, walk, AND ITS OWN ADMISSION QUESTION --
//: over the caller's RAW `[B, 1, H, *]` views. There is no operand plane between the
//: producer and the walk, and no probe launch in front of it: the write set the walk
//: enumerates is the set the residency test reads, so the test is a pass over shared
//: memory. `lattice_k`/`lattice_m` name the `(k, m)` box the state plane is written in --
//: the leaf order travels with the plane rather than being assumed.
//: The verdict is PER BATCH-HEAD (`growth`, `done`, and the `growth_any` reduction the
//: host reads) beside the write-atom set it was reached from (`atom_bits`, which is what
//: a host admission then plans from), and under a slack pool a growth step admits
//: ITSELF: docs/internals/decode/decode_api.md#the-step
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
