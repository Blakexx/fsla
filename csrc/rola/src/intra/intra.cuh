// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/extension.h>

#include <cstdint>

namespace rola {
namespace intra {

//: Per-level sparse-side declaration, two bits per level, level 0 in the low
//: pair.  Bit 0: the READ side's rows are value-sparse.  Bit 1: the WRITE
//: side's.  Neither bit set is DENSE_BOTH; both set is SHARED_SPARSE.  The
//: kernel is pattern-generic: the slab mask machinery reads all-ones for an
//: undeclared side and therefore never gates.
constexpr uint32_t kModeReadSparse = 1u;
constexpr uint32_t kModeWriteSparse = 2u;

//: The window is the intra term's unit and the CTA's whole subject; the tile is
//: the [C x C] output block the causal staircase is expressed on; a warp owns
//: kWarpRows of a tile's rows and reduces nothing with its siblings.
//:
//: THE WINDOW IS AN INSTANTIATION AXIS, mirroring the carry's `W` template
//: argument (integration rung I1(b)): the two kernels share ONE global window
//: grid, so an arm is a (D, W) pair on both sides.  `kWindowFlagship` is the
//: architecture doc's two-constraint window -- the intra's ideal arithmetic
//: intensity is `W/4` flop/byte, which must clear sm_86's compute/bandwidth
//: balance (~380) -- and `kWindowSmall` is the conformance cell's.
constexpr int kWindowFlagship = 384;
//: the doc's band runs to 512, and a THIRD constraint picks it out: a window must
//: DIVIDE `L`, and 384 does not divide the flagship length 65536 while 512 divides
//: every power-of-two length.  Both ends are built and measured.
constexpr int kWindowWide = 512;
constexpr int kWindowSmall = 64;
constexpr int kTile = 64;
constexpr int kWarpRows = 16;
//: the MMA's contraction step, and therefore the QUANTUM a level's operand slab is
//: staged in.  A level narrower than one step is staged PADDED, its tail zeroed
//: once: there is no `k < 16` tensor op, so the digits below the step are the only
//: way to contract eight of them.
constexpr int kSlabDigits = 16;
constexpr int kValueWidth = 64;

//: THE LEVEL WIDTH IS AN INSTANTIATION AXIS, not a constant: an arm is a
//: `(D, B, W)` triple, matching the carry family's topology arms.
constexpr int kWidthFlat = 256;
constexpr int kWidthDeep3 = 16;
constexpr int kWidthDeep4 = 8;

//: `window` selects the built arm; `L % window` must be zero, refused per arm.
void intra_forward(const at::Tensor& pread, const at::Tensor& pwrite, const at::Tensor& gwrite,
                   const at::Tensor& v, at::Tensor& o, at::Tensor& den, int64_t levels,
                   int64_t width, int64_t level_modes, int64_t window);

//: R4's reverse pass: it REDUCES the within-window half of the four cotangents into
//: planes the carry's two scans also write.  ``dnum`` is ``[BH, L, d_v]`` bf16 and
//: ``dden`` is ``[BH, L]`` fp32 -- the seeds the op seam forms from the division.
void intra_backward(const at::Tensor& pread, const at::Tensor& pwrite, const at::Tensor& gwrite,
                    const at::Tensor& v, const at::Tensor& dnum, const at::Tensor& dden,
                    at::Tensor& d_pr, at::Tensor& d_pw, at::Tensor& d_gw, at::Tensor& d_v,
                    int64_t levels, int64_t width, int64_t level_modes, int64_t window);

//: the reverse arms: (levels, width, window, smem, regs, local, threads).
std::vector<std::vector<int64_t>> intra_backward_arms();

//: the reverse pass's own device stamp.
int64_t intra_backward_stamp();

//: the built (levels, width, window) arms, one row each.
std::vector<std::vector<int64_t>> intra_arms();

//: The device-side build stamp: a fact only THIS binary can produce, read back
//: by the test battery so that a green run against a stale extension is
//: impossible to mistake for a green run against the tree.
int64_t intra_build_stamp();

}  // namespace intra
}  // namespace rola
