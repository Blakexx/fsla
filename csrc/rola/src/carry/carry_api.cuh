// csrc/rola/src/carry/carry_api.cuh -- the K31 R1 carry family's host entry
// declarations for the pybind TU (declaration only, so the binding TU never sees
// a device body).  This surface is DELIBERATELY OUTSIDE the shipped dispatch:
// nothing in `rola::chunk` calls it and nothing in it calls `rola::chunk`.
#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace rola::carry {

//: the INTER term only: `(num, den)`, undivided, plus the final state plane in
//: LATTICE leaf order.  `carve` names the one ruled carve: 1 = the innermost
//: level's leaf split.  `level_modes` is law 3's per-level declaration, two bits
//: per level (bit 0 of a level's pair = the READ side is value-sparse there, bit 1
//: = the WRITE side), and it CHOOSES THE BODY; DENSE_BOTH is always correct.
std::vector<torch::Tensor> forward_inter(torch::Tensor pr, torch::Tensor pw, torch::Tensor gw,
                                         torch::Tensor v, std::vector<int64_t> widths, int64_t k,
                                         int64_t m, int64_t window, int64_t chunk, int64_t carve,
                                         int64_t level_modes, c10::optional<torch::Tensor> s_in);

//: the BALLOT ROWS as a tensor: `[BH, 2, entries, L/32]` packed bits, read rows
//: then write rows, one row per (level, group) -- at the lattice's owner spans
//: `s_l`, or at the CARVE grain the kernel consumes (`carve`), which refines the
//: innermost level into the four warps' groups.
torch::Tensor owner_rows(torch::Tensor pr, torch::Tensor pw, std::vector<int64_t> widths, int64_t k,
                         int64_t m, bool carve);

//: device-side build identity plus the built arm's register/SMEM census, and the
//: carve level the declaration named (-1 = the channel-split body).
std::vector<int64_t> build_stamp(std::vector<int64_t> widths, int64_t k, int64_t m, int64_t window,
                                 int64_t chunk, int64_t carve, int64_t level_modes);

//: R4's TWO REVERSE SCANS, launched concurrently: `(d_pr, d_pw, d_gw, d_v,
//: d_state_in, replay)`.  `vt`/`dn` are the pre-packed `[BH, L, cols-padded]` bf16
//: value and seed planes; `replay` is Scan A's slab, empty unless asked for.
std::vector<torch::Tensor> backward_inter(torch::Tensor pr, torch::Tensor pw, torch::Tensor gw,
                                          torch::Tensor vt, torch::Tensor dn,
                                          std::vector<int64_t> widths, int64_t k, int64_t m,
                                          int64_t window, int64_t chunk,
                                          c10::optional<torch::Tensor> s_in,
                                          c10::optional<torch::Tensor> ds_out, bool want_replay);

//: the reverse family's own device stamp plus a PER-KERNEL census: registers, local
//: bytes, SMEM, ring depth, CTA target and the declared floor, Scan A then Scan B.
std::vector<int64_t> backward_build_stamp(std::vector<int64_t> widths, int64_t k, int64_t m,
                                          int64_t window, int64_t chunk);

//: the built (widths, k, m, W, C, carve) tuples.
std::vector<std::vector<int64_t>> arms();

}  // namespace rola::carry
