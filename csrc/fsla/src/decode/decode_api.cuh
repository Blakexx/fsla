#pragma once

// THE DECODE PATH'S HOST-FACING DECLARATIONS -- the ONLY decode header
// `csrc/fsla/fsla_api.cpp` includes; see docs/internals/decode/decode_api.md#note-l3

#include "common/torch_seam.cuh"

#include <cstdint>
#include <vector>

namespace fsla {
namespace decode {

int64_t fsla_decode_capacity();  // the bound surface; see decode_api.md#fsla-decode-capacity
int64_t fsla_decode_producer_width_mirror();

int64_t
fsla_decode_build_stamp();  // device-side build fact; see decode_api.md#fsladecodebuildstamp

std::vector<std::vector<int64_t>> fsla_decode_arms();  // see decode_api.md#fsladecodearms

//: binds the phase ledger the next steps book into, or unbinds it; see decode_api.md#decode-ledger-bind
void fsla_decode_ledger_bind(const std::optional<Tensor>& ledger);

//: the phase ledger's warps at the current card's build, what a ledger is sized by; see decode_api.md#decode-ledger-warps
int64_t fsla_decode_ledger_warps();

//: THE LAUNCH A DESCRIPTOR MEANS on `device`: arm `d_v`, levels, decay; value padding; level widths
std::vector<int64_t> fsla_decode_launch(std::vector<int64_t> descriptor, int64_t device);

//: THE LATTICE the launch function solves for level widths, and the box of a `(k, m)` on them: `[k, m, BC, owners, spans.., grids..]`, no device
std::vector<int64_t> fsla_decode_lattice(std::vector<int64_t> widths);
std::vector<int64_t> fsla_decode_lattice_box(std::vector<int64_t> widths, int64_t k, int64_t m);

Tensor fsla_decode_forward(std::vector<Tensor> read, std::vector<Tensor> write,
                           std::vector<Tensor> read_scale, std::vector<Tensor> write_scale,
                           std::vector<Tensor> read_mask, std::vector<Tensor> write_mask,
                           std::optional<Tensor> dials, Tensor v, Tensor v_scale, Tensor state,
                           Tensor yacc, Tensor queue, int64_t ctas, double eps,
                           std::vector<int64_t> descriptor);

}  // namespace decode
}  // namespace fsla
