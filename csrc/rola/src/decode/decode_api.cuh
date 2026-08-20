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
int64_t rola_decode_residency(bool decay);
int64_t rola_decode_producer_width_mirror();

//: THE EXPANSION'S GLOBAL SCRATCH, in int32 PER CTA, and `0` at every shipped topology.
//: The host multiplies by `BH * n_split`. Asking the kernel's own sizing function is what
//: keeps the buffer from being sized by a host-side mirror that can drift:
//: docs/internals/decode/decode_api.md#work-words
int64_t rola_decode_work_words(std::vector<int64_t> level_widths, int64_t d_v, bool decay,
                               int64_t device_index);

//: THE WHOLE STEP IN ONE LAUNCH -- fold, expand, walk, AND ITS OWN ADMISSION QUESTION --
//: over the caller's RAW `[B, 1, H, *]` views. There is no operand plane between the
//: producer and the walk, and no probe launch in front of it: the write map the walk
//: needs is the map the residency test reads, so the test is a pass over shared memory.
//: The verdict is PER BATCH-HEAD (`growth`, `done`, and the `growth_any` reduction the
//: host reads) beside the write-atom set it was reached from (`atom_bits`, which is what
//: a host admission then plans from), and under a slack pool a growth step admits
//: ITSELF: docs/internals/decode/decode_api.md#the-step
at::Tensor rola_decode_forward(
    std::vector<at::Tensor> read, std::vector<at::Tensor> write, std::vector<int64_t> normalize,
    c10::optional<at::Tensor> dials, at::Tensor g_write, at::Tensor v, at::Tensor state,
    at::Tensor work, at::Tensor ws, at::Tensor ctr, at::Tensor growth, at::Tensor growth_any,
    at::Tensor growth_ctr, at::Tensor done, at::Tensor atom_bits, std::vector<int64_t> level_widths,
    std::vector<int64_t> level_strides, std::vector<int64_t> level_spans,
    std::vector<int64_t> level_row_offsets, int64_t n_split, double eps,
    c10::optional<at::Tensor> page_table, c10::optional<at::Tensor> pool_slots,
    c10::optional<at::Tensor> pool_cursor, c10::optional<at::Tensor> pool_map);

}  // namespace decode
}  // namespace rola
