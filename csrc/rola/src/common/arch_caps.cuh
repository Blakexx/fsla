#pragma once

// LAYER 1 -- THE ARCHITECTURE CAPABILITY FACTS: the rows `common/hardware.cuh` generates from
// common/hardware.json, and what is derived from them. Entry kinds (REQUIRED/SELECTING/RESIDENCY/FACT) and the
// no-fallback-chain rule: see docs/internals/common/arch_caps.md

#include <cuda_runtime.h>

#include "common/hardware.cuh"

namespace denseref {
namespace arch {

constexpr bool supported(const Caps& c) {  // conjunction of REQUIRED; see arch_caps.md#note-l163
  return c.async_copy && c.tile_load && c.mma_bf16_f32 && c.bf16_cvt_packed && c.reg_transpose_b16;
}

// DERIVATION -- the quantitative half; only consumers of the FACT rows, no arch branch.

constexpr int round_up(int x, int q) { return (x + q - 1) / q * q; }

//: Registers a CTA actually reserves.  Allocation is per warp in `reg_alloc_unit` -- see docs/internals/common/arch_caps.md#regspercta
constexpr int regs_per_cta(const Caps& c, int threads, int regs_per_lane) {
  return threads / 32 * round_up(regs_per_lane * 32, c.reg_alloc_unit);
}

constexpr int ctas_by_smem(const Caps& c, int smem_per_cta) {
  return c.smem_per_sm / (round_up(smem_per_cta, c.smem_granularity) + c.driver_smem_reserve);
}

constexpr int ctas_by_regs(const Caps& c, int threads, int regs_per_lane) {
  return c.regs_per_sm / regs_per_cta(c, threads, regs_per_lane);
}

constexpr int ctas_by_threads(const Caps& c, int threads) { return c.max_threads_per_sm / threads; }

//: Resident CTAs per SM: the binding constraint among shared memory, registers, -- see docs/internals/common/arch_caps.md#min4
constexpr int min4(int a, int b, int d, int e) {
  return a < b ? (a < d ? (a < e ? a : e) : (d < e ? d : e))
               : (b < d ? (b < e ? b : e) : (d < e ? d : e));
}

constexpr int residency(const Caps& c, int smem_per_cta, int threads, int regs_per_lane) {
  return min4(ctas_by_smem(c, smem_per_cta), ctas_by_regs(c, threads, regs_per_lane),
              ctas_by_threads(c, threads), c.max_ctas_per_sm);
}

//: THE PER-ARCH REGISTER BUDGET. The largest per-lane register count that still -- see docs/internals/common/arch_caps.md#note-l203
constexpr int reg_budget(const Caps& c, int threads, int target_ctas) {
  return (c.regs_per_sm / target_ctas / (threads / 32) / c.reg_alloc_unit * c.reg_alloc_unit) / 32
                 < c.regs_per_lane_max
             ? (c.regs_per_sm / target_ctas / (threads / 32) / c.reg_alloc_unit * c.reg_alloc_unit)
                   / 32
             : c.regs_per_lane_max;
}

// THE COMPILING TARGET; inert in the host pass, pinned to the smallest tabulated config.
#ifdef __CUDA_ARCH__
constexpr int kArch = __CUDA_ARCH__;
#else
constexpr int kArch = 860;
#endif

static_assert(tabulated(kArch),
              "untabulated compute capability: add its row to caps_of() rather "
              "than letting an operation guess which implementation it has");

constexpr Caps kCaps = caps_of(kArch);

static_assert(supported(kCaps),
              "UNSUPPORTED ARCHITECTURE: a required capability "
              "clause is false; see the Caps row");

// `mma_unit_threads` is NOT asserted here -- an operation fact, not an arch one; lives in ops.cuh.

constexpr int caps_digest(const Caps& c) {  // see docs/internals/common/arch_caps.md#capsdigest
  return c.cc * 1000 + (int)c.async_copy * 1 + (int)c.tile_load * 2 + (int)c.mma_bf16_f32 * 4
         + (int)c.mma_b_from_regs * 8 + (int)c.bf16_cvt_packed * 16 + (int)c.bf16_simd_arith * 32
         + (int)c.async_bulk_copy * 64 + (int)c.warpgroup_mma * 128 + (int)c.mma_a_from_regs * 256
         + (int)c.accum_in_tmem * 512 + (int)c.mma_a_from_tmem * 1024
         + (int)c.reg_transpose_b16 * 2048 + (int)c.bf16_simd_fma * 4096 + c.mma_unit_threads * 2;
}

}  // namespace arch
}  // namespace denseref
