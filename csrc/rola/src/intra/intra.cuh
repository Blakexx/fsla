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
constexpr int kWindow = 256;
constexpr int kTile = 64;
constexpr int kWarpRows = 16;
constexpr int kSlabDigits = 16;
constexpr int kLevelWidth = 256;
constexpr int kValueWidth = 64;

void intra_forward(const at::Tensor& pread, const at::Tensor& pwrite, const at::Tensor& gwrite,
                   const at::Tensor& v, at::Tensor& o, at::Tensor& den, int64_t levels,
                   int64_t level_modes);

//: The device-side build stamp: a fact only THIS binary can produce, read back
//: by the test battery so that a green run against a stale extension is
//: impossible to mistake for a green run against the tree.
int64_t intra_build_stamp();

}  // namespace intra
}  // namespace rola
