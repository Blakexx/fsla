// csrc/fsla/src/produce/produce.cuh -- THE PRODUCER'S FUSED FORWARD, one launch from the hidden states to the op's
// operands: its SKELETON, each component a stub writing its outputs' zeros. see docs/internals/produce/produce.md
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "common/atoms/barrier.cuh"

namespace fsla {
namespace produce {
namespace detail {

namespace ops = ::denseref::ops;

//: a CTA's token tile, a math warpgroup's share of it, the hidden a ring stage carries, a mask word's tokens
constexpr int kTileTokens = 128, kGroupTokens = 64, kKStep = 64, kWordTokens = 32;
//: THE ROLES: two math warpgroups, then one loader warp -- see docs/internals/produce/produce.md#roles
constexpr int kGroupThreads = 128, kMathThreads = 2 * kGroupThreads, kMathWarps = kMathThreads / 32;
constexpr int kThreads = kMathThreads + 32;
//: THE RING: three stages of `x` [128 x 64] and weights [256 x 64], then FULL, EMPTY and the live counts -- see docs/internals/produce/produce.md#ring
constexpr int kStages = 3, kStageBytes = (kTileTokens + 256) * kKStep * 2,
              kBarsAt = kStages * kStageBytes;
constexpr int kLiveAt = kBarsAt + 2 * kStages * 8, kSmemBytes = kLiveAt + 8 * 4;
//: the plan's bounds: levels, read heads a write head, jobs a write head
constexpr int kMaxLevels = 4, kMaxGroup = 16, kMaxJobs = kMaxLevels * (1 + kMaxGroup) + 1;
//: the density's sides, in its rows' order (api.md 2.6)
constexpr int kRead = 0, kWrite = 1, kSides = 2;

enum Activation : uint8_t { kSoftmax = 0, kEntmax15, kEntmax2 };

enum JobKind : uint8_t { kWriteJob = 0, kReadJob, kValueJob };

//: ONE JOB of a write head: a row span of its columns -- see docs/internals/produce/produce.md#jobs
struct Job {
  uint8_t kind, level, head, activation;
  //: its digits (a level) or channels (the values)
  uint16_t width;
  //: a write job's tied read heads, a bit each: their planes take copies of its rows
  uint16_t tied;
};

//: THE LAUNCH'S PLAN, its one `__grid_constant__` parameter -- see docs/internals/produce/produce.md#plan
struct Plan {
  int batch, tokens, write_heads, group, levels, hidden, d_v, jobs;
  int widths[kMaxLevels];
  bool masked[kSides][kMaxLevels];
  Job job[kMaxJobs];
  //: a level-side's planes in api.md's layouts, H its side's heads
  uint8_t* rows[kSides][kMaxLevels];
  float* scales[kSides][kMaxLevels];
  int8_t* lo[kSides][kMaxLevels];
  uint32_t* masks[kSides][kMaxLevels];
  __nv_bfloat16* values;
  uint8_t* density;
  unsigned long long* counts;
  uint32_t* ticket;
};

//: ZEROS OVER `runs` RUNS of `bytes`, `stride` apart, by a warpgroup -- see docs/internals/produce/produce.md#stores
__device__ __forceinline__ void zero_runs(uint8_t* at, int runs, int bytes, long long stride,
                                          int mine) {
  if (stride == bytes) {
    bytes *= runs;
    runs = 1;
  }
  const int chunks = (bytes + 15) >> 4, n = runs * chunks;
#pragma unroll 1
  for (int i = mine; i < n; i += kGroupThreads) {
    const int r = i / chunks, c = i - r * chunks, left = min(16, bytes - (c << 4));
    uint8_t* const p = at + r * stride + (c << 4);
    if (left == 16 && (reinterpret_cast<uintptr_t>(p) & 15u) == 0u) {
      *reinterpret_cast<uint4*>(p) = make_uint4(0u, 0u, 0u, 0u);
    } else {
#pragma unroll 1
      for (int k = 0; k < left; ++k) p[k] = 0;
    }
  }
}

//: A WARPGROUP'S SHARE: sequence `b`'s tokens `[t0, t1)` of the CTA's tile, and write head `w`
struct Share {
  int b, t0, t1, w;
};

//: a token's index in a token-major plane of `heads` heads
__device__ __forceinline__ long long place(const Plan& p, const Share& s, int t, int head,
                                           int heads) {
  return (static_cast<long long>(s.b) * p.tokens + t) * heads + head;
}

//: A LEVEL-SIDE'S OUTPUTS FOR ONE HEAD (the format's and mask's stubs) -- see docs/internals/produce/produce.md#format
template <bool LO>
__device__ __forceinline__ void store_level(const Plan& p, const Share& s, int side, int level,
                                            int head, int mine) {
  const int heads = side == kWrite ? p.write_heads : p.write_heads * p.group,
            width = p.widths[level];
  const int n = s.t1 - s.t0;
  const long long first = place(p, s, s.t0, head, heads);
  zero_runs(p.rows[side][level] + first * width, n, width, static_cast<long long>(heads) * width,
            mine);
  zero_runs(reinterpret_cast<uint8_t*>(p.scales[side][level] + first), n, 4, heads * 4ll, mine);
  if constexpr (LO)
    zero_runs(reinterpret_cast<uint8_t*>(p.lo[side][level] + first * width), n, width,
              static_cast<long long>(heads) * width, mine);
  if (p.masked[side][level]) {
    const int words = (p.tokens + kWordTokens - 1) / kWordTokens, w0 = s.t0 / kWordTokens;
    const int w1 = (s.t1 + kWordTokens - 1) / kWordTokens;
    uint32_t* const at =
        p.masks[side][level] + ((static_cast<long long>(s.b) * heads + head) * words + w0) * width;
    zero_runs(reinterpret_cast<uint8_t*>(at), 1, (w1 - w0) * width * 4, 0, mine);
  }
}

//: A JOB'S EPILOGUE on a warpgroup's share -- see docs/internals/produce/produce.md#epilogue
template <bool LO>
__device__ __forceinline__ void epilogue(const Plan& p, const Job& job, const Share& s, int mine) {
  if (s.t1 <= s.t0) return;
  if (job.kind == kWriteJob) {
    store_level<LO>(p, s, kWrite, job.level, s.w, mine);
#pragma unroll 1
    for (int g = 0; g < p.group; ++g)
      if ((job.tied >> g) & 1u) store_level<LO>(p, s, kRead, job.level, s.w * p.group + g, mine);
  } else if (job.kind == kReadJob) {
    store_level<LO>(p, s, kRead, job.level, s.w * p.group + job.head, mine);
  } else {
    const long long first = place(p, s, s.t0, s.w, p.write_heads);
    zero_runs(reinterpret_cast<uint8_t*>(p.values + first * p.d_v), s.t1 - s.t0, p.d_v * 2,
              static_cast<long long>(p.write_heads) * p.d_v * 2, mine);
  }
}

//: THE LOADER: a stage a K-step of each job (the stub stages no bytes) -- see docs/internals/produce/produce.md#ring
__device__ __forceinline__ void load_role(const Plan& p, ops::SmemAddr full, ops::SmemAddr empty,
                                          int lane) {
  const int stages = p.jobs * ((p.hidden + kKStep - 1) / kKStep);
  if (lane != 0) return;
#pragma unroll 1
  for (int i = 0; i < stages; ++i) {
    const int slot = i % kStages;
    if (i >= kStages) ops::mbar_wait(empty + 8 * slot, ((i / kStages) - 1) & 1);
    ops::mbar_arrive(full + 8 * slot);
  }
}

//: A MATH WARPGROUP: each job's K-steps (the ring's edges) then its epilogue -- see docs/internals/produce/produce.md#math
template <bool LO>
__device__ __forceinline__ void math_role(const Plan& p, const Share& s, ops::SmemAddr full,
                                          ops::SmemAddr empty, int mine) {
  const int steps = (p.hidden + kKStep - 1) / kKStep;
#pragma unroll 1
  for (int j = 0; j < p.jobs; ++j) {
#pragma unroll 1
    for (int k = 0; k < steps; ++k) {
      const int i = j * steps + k, slot = i % kStages;
      ops::mbar_wait(full + 8 * slot, (i / kStages) & 1);
      ops::warp_sync();
      if ((mine & 31) == 0) ops::mbar_arrive(empty + 8 * slot);
    }
    epilogue<LO>(p, p.job[j], s, mine);
  }
}

//: THE DENSITY'S TICKET: the last CTA turns the counts into bytes and zeroes them -- see docs/internals/produce/produce.md#density
__device__ __forceinline__ void density_ticket(const Plan& p, const unsigned* live) {
  for (int side = 0; side < kSides; ++side)
#pragma unroll 1
    for (int l = 0; l < p.levels; ++l)
      if (p.masked[side][l])
        atomicAdd(&p.counts[side * kMaxLevels + l],
                  static_cast<unsigned long long>(live[side * kMaxLevels + l]));
  __threadfence();
  if (atomicAdd(p.ticket, 1u) != gridDim.x * gridDim.y - 1) return;
  __threadfence();
  for (int side = 0; side < kSides; ++side)
#pragma unroll 1
    for (int l = 0; l < p.levels; ++l) {
      const int heads = side == kWrite ? p.write_heads : p.write_heads * p.group;
      const unsigned long long total =
          static_cast<unsigned long long>(p.batch) * p.tokens * heads * p.widths[l];
      const unsigned long long n = atomicExch(&p.counts[side * kMaxLevels + l], 0ull);
      p.density[side * p.levels + l] =
          p.masked[side][l] ? static_cast<uint8_t>((2ull * 255ull * n + total) / (2ull * total))
                            : uint8_t{255};
    }
  *p.ticket = 0u;
}

//: ONE CTA: a sequence's token tile and write head `blockIdx.y` -- see docs/internals/produce/produce.md#cta
template <bool LO>
__device__ __forceinline__ void produce_cta(const Plan& p, uint8_t* smem) {
  const int tid = static_cast<int>(threadIdx.x);
  unsigned* const live = reinterpret_cast<unsigned*>(smem + kLiveAt);
  const ops::SmemAddr full = ops::smem_addr(smem + kBarsAt), empty = full + 8 * kStages;
  if (tid < kSides * kMaxLevels) live[tid] = 0u;
  if (tid == 0)
    for (int s = 0; s < kStages; ++s) {
      ops::mbar_init(full + 8 * s, 1);
      ops::mbar_init(empty + 8 * s, kMathWarps);
    }
  __syncthreads();
  if (tid < kMathThreads) {
    const int tiles = (p.tokens + kTileTokens - 1) / kTileTokens;
    const int b = static_cast<int>(blockIdx.x) / tiles;
    const int t0 =
        (static_cast<int>(blockIdx.x) % tiles) * kTileTokens + (tid / kGroupThreads) * kGroupTokens;
    const Share s{b, min(t0, p.tokens), min(t0 + kGroupTokens, p.tokens),
                  static_cast<int>(blockIdx.y)};
    math_role<LO>(p, s, full, empty, tid % kGroupThreads);
  } else {
    load_role(p, full, empty, tid & 31);
  }
  __syncthreads();
  if (tid == 0) density_ticket(p, live);
}

}  // namespace detail
}  // namespace produce
}  // namespace fsla
