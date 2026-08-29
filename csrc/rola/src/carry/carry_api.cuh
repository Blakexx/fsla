// csrc/rola/src/carry/carry_api.cuh -- the K31 R1 carry family's host entry
// declarations for the pybind TU (declaration only, so the binding TU never sees
// a device body).  This surface is DELIBERATELY OUTSIDE the shipped dispatch:
// nothing in `rola::chunk` calls it and nothing in it calls `rola::chunk`.
#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace rola::carry {

//: the INTER term only: `(num, den)`, undivided.
// see docs/internals/carry/carry_api.md#note-l14
std::vector<torch::Tensor> forward_inter(torch::Tensor pr, torch::Tensor pw, torch::Tensor gw,
                                         torch::Tensor v, std::vector<int64_t> widths, int64_t k,
                                         int64_t m, int64_t window, int64_t chunk, int64_t carve,
                                         int64_t level_modes, c10::optional<torch::Tensor> s_in,
                                         c10::optional<torch::Tensor> s_out,
                                         c10::optional<torch::Tensor> page_tbl);

//: the BALLOT ROWS as a tensor: `[BH, 2, entries, L/32]` packed bits, read rows
//: then write rows, one row per (level, group) ...
// see docs/internals/carry/carry_api.md#note-l35
torch::Tensor owner_rows(torch::Tensor pr, torch::Tensor pw, std::vector<int64_t> widths, int64_t k,
                         int64_t m, bool carve);

//: device-side build identity plus the built arm's register/SMEM census, the
//: carve level the declaration named (-1 = the channel-split body), the WRITE
//: stream count and the arm's `d_v`.
// see docs/internals/carry/carry_api.md#note-l42
std::vector<int64_t> build_stamp(std::vector<int64_t> widths, int64_t k, int64_t m, int64_t window,
                                 int64_t chunk, int64_t carve, int64_t level_modes, int64_t d_v);

//: R4's TWO REVERSE SCANS, launched concurrently: `(d_pr, d_pw, d_gw, d_v,
//: d_state_in, replay)`.  `vt`/`dn` are the pre-packed `[BH, L, cols-padded]` bf16
//: value and seed planes; `replay` is Scan A's slab, empty unless asked for.
std::vector<torch::Tensor> backward_inter(torch::Tensor pr, torch::Tensor pw, torch::Tensor gw,
                                          torch::Tensor vt, torch::Tensor dn,
                                          std::vector<int64_t> widths, int64_t k, int64_t m,
                                          int64_t window, int64_t chunk,
                                          c10::optional<torch::Tensor> s_in,
                                          c10::optional<torch::Tensor> ds_out,
                                          c10::optional<torch::Tensor> page_tbl, bool want_replay);

//: the reverse family's own device stamp plus a PER-KERNEL census: registers, local
//: bytes, SMEM, ring depth, CTA target and the declared floor, Scan A then Scan B.
std::vector<int64_t> backward_build_stamp(std::vector<int64_t> widths, int64_t k, int64_t m,
                                          int64_t window, int64_t chunk);

//: the built (widths, k, m, d_v, W, C, carve, modes, BC, owners, smem, carve level) rows.
std::vector<std::vector<int64_t>> arms();

}  // namespace rola::carry
