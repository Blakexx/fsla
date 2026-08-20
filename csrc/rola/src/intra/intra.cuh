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
constexpr int kSlabDigits = 16;
constexpr int kLevelWidth = 256;
constexpr int kValueWidth = 64;

//: `window` selects the built arm; `L % window` must be zero, refused per arm.
void intra_forward(const at::Tensor& pread, const at::Tensor& pwrite, const at::Tensor& gwrite,
                   const at::Tensor& v, at::Tensor& o, at::Tensor& den, int64_t levels,
                   int64_t level_modes, int64_t window);

//: the built (levels, window) arms, one row each.
std::vector<std::vector<int64_t>> intra_arms();

//: The device-side build stamp: a fact only THIS binary can produce, read back
//: by the test battery so that a green run against a stale extension is
//: impossible to mistake for a green run against the tree.
int64_t intra_build_stamp();

}  // namespace intra
}  // namespace rola
