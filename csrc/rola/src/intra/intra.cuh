// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/extension.h>

#include <cstdint>

namespace rola {
namespace intra {

constexpr uint32_t kModeReadSparse = 1u;  // see docs/internals/intra/intra.md#kmodereadsparse
constexpr uint32_t kModeWriteSparse = 2u;

constexpr int kWindowFlagship = 384;  // see docs/internals/intra/intra.md#kwindowflagship
constexpr int kWindowWide = 512;      // see docs/internals/intra/intra.md#kwindowwide
constexpr int kWindowSmall = 64;
constexpr int kTile = 64;
constexpr int kWarpRows = 16;
constexpr int kSlabDigits =
    16;  // the MMA contraction step; see docs/internals/intra/intra.md#kslabdigits
constexpr int kValueWidth = 64;

constexpr int kWidthFlat = 256;  // an instantiation axis, not a constant; see intra.md#kwidthflat
constexpr int kWidthDeep3 = 16;
constexpr int kWidthDeep4 = 8;

// `window` selects the built arm; `L % window` must be zero, refused per arm; see intra.md#intra-forward
void intra_forward(const at::Tensor& pread, const at::Tensor& pwrite, const at::Tensor& gwrite,
                   const at::Tensor& v, const at::Tensor& sread, const at::Tensor& swrite,
                   at::Tensor& o, at::Tensor& den, int64_t levels, int64_t width,
                   int64_t level_modes, int64_t window);

// R4's reverse pass; see docs/internals/intra/intra.md#intrabackward
void intra_backward(const at::Tensor& pread, const at::Tensor& pwrite, const at::Tensor& gwrite,
                    const at::Tensor& v, const at::Tensor& dnum, const at::Tensor& dden,
                    at::Tensor& d_pr, at::Tensor& d_pw, at::Tensor& d_gw, at::Tensor& d_v,
                    int64_t levels, int64_t width, int64_t level_modes, int64_t window);

std::vector<std::vector<int64_t>>
intra_backward_arms();  // (levels, width, window, smem, regs, ...)
int64_t intra_backward_stamp();
std::vector<std::vector<int64_t>> intra_arms();  // the built (levels, width, window) arms
int64_t intra_build_stamp();  // device-side build fact; see intra.md#intrabuildstamp

}  // namespace intra
}  // namespace rola
