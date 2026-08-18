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

//: THE EXPANSION'S GLOBAL SCRATCH, in int32 PER CTA, and `0` at every shipped topology.
//: The host multiplies by `BH * n_split`. Asking the kernel's own sizing function is what
//: keeps the buffer from being sized by a host-side mirror that can drift:
//: docs/internals/decode/decode_api.md#work-words
int64_t rola_decode_work_words(std::vector<int64_t> level_widths, int64_t d_v, bool decay,
                               int64_t device_index);

//: THE WHOLE STEP IN ONE LAUNCH -- fold, expand, walk -- over the caller's RAW
//: `[B, 1, H, *]` views. There is no operand plane between the producer and the walk:
//: docs/internals/decode/decode_api.md#the-step
at::Tensor rola_decode_forward(std::vector<at::Tensor> read, std::vector<at::Tensor> write,
                               std::vector<int64_t> normalize, c10::optional<at::Tensor> dials,
                               at::Tensor g_write, at::Tensor v, at::Tensor state, at::Tensor work,
                               at::Tensor ws, at::Tensor ctr, std::vector<int64_t> level_widths,
                               std::vector<int64_t> level_strides, std::vector<int64_t> level_spans,
                               std::vector<int64_t> level_row_offsets, int64_t n_split, double eps,
                               c10::optional<at::Tensor> page_table,
                               c10::optional<at::Tensor> gate);

//: THE GROWTH PROBE, bound separately because it is a different HOST ORDER and not a
//: mode of the step: it runs BEFORE the step so the host can learn whether this step's
//: write set is already resident WITHOUT deriving that set itself. It folds and expands
//: its own write side, and writes the verdict (`growth`, and its `growth_any` reduction --
//: the step's GATE) beside the write-atom set that verdict was reached from (`atom_bits`,
//: which is what a growth step then admits) -- no state, no `y`, no table:
//: docs/internals/decode/decode_api.md#growth-probe
void rola_decode_growth_probe(std::vector<at::Tensor> write, at::Tensor page_table, at::Tensor work,
                              at::Tensor probe_bits, at::Tensor growth, at::Tensor growth_any,
                              at::Tensor probe_ctr, at::Tensor atom_bits, int64_t heads,
                              int64_t n_split, std::vector<int64_t> level_widths,
                              std::vector<int64_t> level_strides, std::vector<int64_t> level_spans);

}  // namespace decode
}  // namespace rola
