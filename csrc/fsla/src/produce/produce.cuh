// csrc/fsla/src/produce/produce.cuh -- THE PRODUCER'S FUSED FORWARD, one launch from the hidden states to the op's
// operands: its SKELETON, each component a stub writing its outputs' zeros. see docs/internals/produce/produce.md
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "common/atoms/barrier.cuh"
#include "common/smem_layout.cuh"

namespace fsla {
namespace produce {
namespace detail {

namespace ops = ::denseref::ops;

//: FIXED by the roles: a CTA's token tile, two warpgroups' m64 steps
constexpr int kTileTokens = 128;
//: FIXED by the step: a warpgroup's tokens, `wgmma`'s m64, and a row tile of the format
constexpr int kGroupTokens = 64;
//: FIXED by the ring's form: the hidden a stage carries, a 128-byte swizzled line of bf16
constexpr int kKStep = 64;
//: FIXED by the format: a mask word is 32 tokens
constexpr int kWordTokens = 32;
//: FIXED by the format: a row block is a token's 16 bytes, the integer step's K
constexpr int kBlockDigits = 16;
//: FIXED by the roles: a warpgroup's threads -- see docs/internals/produce/produce.md#roles
constexpr int kGroupThreads = 128;
constexpr int kMathThreads = 2 * kGroupThreads;
constexpr int kMathWarps = kMathThreads / 32;
constexpr int kThreads = kMathThreads + 32;
//: FIXED by the plan's parameter: the levels a plan holds (its `__grid_constant__` arrays)
constexpr int kMaxLevels = 4;
//: FIXED by the plan's parameter: the read heads a write head a plan holds
constexpr int kMaxGroup = 16;
constexpr int kMaxRows = kMaxGroup + 1;
constexpr int kMaxJobs = kMaxLevels * kMaxRows + 1;
//: FIXED by the barriers' room: the most stages the ring holds, whatever the card leaves
constexpr int kMaxStages = 8;

//: a level's two sides, each one tensor over its heads (the read side's query head `w G + g`)
enum Side : int { kRead = 0, kWrite, kSides };

enum Activation : uint8_t { kSoftmax = 0, kEntmax15, kEntmax2 };

enum JobKind : uint8_t { kWriteJob = 0, kReadJob, kValueJob };

//: A RING STAGE: a tile's `x` [128 x 64] and a job's weights [256 x 64], bf16
struct Stage {
  uint8_t bytes[(kTileTokens + 256) * kKStep * 2];
};

struct Bars;
struct Live;
struct Ring;

//: THE CTA'S SHARED MEMORY on the build `Bt`, `CTAS` CTAs an SM: the ring takes what the card leaves -- see
//: docs/internals/produce/produce.md#ring
template <class Bt, int CTAS>
using Smem = SmemLayout<Bt, CTAS, Fixed<Bars, uint64_t, 2 * kMaxStages>,
                        Fixed<Live, unsigned, kMaxRows * kMaxLevels>, Rest<Ring, Stage>>;

//: the ring's stages on the build
template <class Bt, int CTAS>
constexpr int kStages = Smem<Bt, CTAS>::template count<Ring> < kMaxStages
                            ? static_cast<int>(Smem<Bt, CTAS>::template count<Ring>)
                            : kMaxStages;

//: ONE JOB of a write head: a row span of its columns -- see docs/internals/produce/produce.md#jobs
struct Job {
  uint8_t kind, level, head, activation;
  //: a write job's tied read heads, a bit each: their planes take copies of its rows
  uint16_t tied;
};

//: THE LAUNCH'S PLAN, its one `__grid_constant__` parameter -- see docs/internals/produce/produce.md#plan
struct Plan {
  int batch, tokens, write_heads, group, levels, hidden, d_v, jobs;
  int widths[kMaxLevels];
  bool masked[kMaxRows][kMaxLevels];
  Job job[kMaxJobs];
  //: a level-side's rows, scales and lo plane over its heads; its masks a row (read head `g`, the write side at `group`)
  uint8_t* rows[kSides][kMaxLevels];
  float* scales[kSides][kMaxLevels];
  uint8_t* lo[kSides][kMaxLevels];
  uint32_t* masks[kMaxRows][kMaxLevels];
  __nv_bfloat16* values;
  float* growth;
  float* growth_sums;
  uint32_t* sequence_tickets;
  uint8_t* density;
  unsigned long long* counts;
  uint32_t* ticket;
};

//: THE LIBRARY'S TRACER: nothing -- see docs/internals/produce/produce.md#trace
struct NoTrace {
  __device__ __forceinline__ void phase(int) {}

  __device__ __forceinline__ void wait_begin(int) {}

  __device__ __forceinline__ void wait_end(int) {}

  __device__ __forceinline__ void arrive(int) {}
};

//: a CTA's phases, as its tracer names them
enum Site : int { kSiteEntry = 0, kSiteSetup, kSiteWindow, kSiteEpilogue, kSiteTicket, kSiteExit };

//: a ring barrier's number, as its tracer names it: a stage's FULL `slot`, its EMPTY `kMaxStages + slot`
constexpr int kEmptyId = kMaxStages;

//: A CONTIGUOUS RUN'S ZEROS by a warpgroup (`mine` its thread): `bytes` and the start a multiple of 4; words to a 16-byte
//: boundary, 16 bytes a store, then the tail's words -- no division -- see docs/internals/produce/produce.md#stores
__device__ __forceinline__ void zero_run(uint8_t* at, int bytes, int mine) {
  const uintptr_t off = reinterpret_cast<uintptr_t>(at) & 15u;
  const int lead = min(bytes, static_cast<int>((16u - off) & 15u));
  if (mine < (lead >> 2)) reinterpret_cast<uint32_t*>(at)[mine] = 0u;
  uint8_t* const body = at + lead;
  const int chunks = (bytes - lead) >> 4;
#pragma unroll 1
  for (int c = mine; c < chunks; c += kGroupThreads)
    reinterpret_cast<uint4*>(body)[c] = make_uint4(0u, 0u, 0u, 0u);
  const int tail = (bytes - lead - (chunks << 4)) >> 2;
  if (mine < tail) reinterpret_cast<uint32_t*>(body + (chunks << 4))[mine] = 0u;
}

//: A SHARE'S TOKENS' ZEROS by a warpgroup: `tokens` (at most a row tile's 64) runs of `bytes`, `stride` apart, each a
//: multiple of 4: two threads a token, 16 bytes a store where the runs allow it, else a word -- no division -- see
//: docs/internals/produce/produce.md#stores
__device__ __forceinline__ void zero_tokens(uint8_t* at, int tokens, int bytes, long long stride,
                                            int mine) {
  const int token = mine & (kGroupTokens - 1), half = mine >> 6;
  if (token >= tokens) return;
  uint8_t* const run = at + token * stride;
  const long long where = static_cast<long long>(reinterpret_cast<uintptr_t>(at));
  const bool wide = ((bytes | stride | where) & 15) == 0;
  if (wide) {
#pragma unroll 1
    for (int c = half; c < (bytes >> 4); c += 2)
      reinterpret_cast<uint4*>(run)[c] = make_uint4(0u, 0u, 0u, 0u);
  } else {
#pragma unroll 1
    for (int w = half; w < (bytes >> 2); w += 2) reinterpret_cast<uint32_t*>(run)[w] = 0u;
  }
}

//: A WARPGROUP'S SHARE: sequence `b`'s row tile `tile` (tokens `[t0, t1)`) and write head `w`
struct Share {
  int b, tile, t0, t1, w;
};

//: the bytes of one (sequence, write head)'s row tile of a level: its 16-digit blocks of 64 tokens' 16 bytes
__device__ __forceinline__ long long tile_bytes(int width) {
  return static_cast<long long>((width + kBlockDigits - 1) / kBlockDigits) * kGroupTokens
         * kBlockDigits;
}

//: A LEVEL ROW'S OUTPUTS FOR ONE SHARE (the format's and mask's stubs) -- see
//: docs/internals/produce/produce.md#format
template <bool LO>
__device__ __forceinline__ void store_level(const Plan& p, const Share& s, int row, int level,
                                            bool own_mask, int mine) {
  const int width = p.widths[level], tiles = (p.tokens + kGroupTokens - 1) / kGroupTokens;
  const int side = row == p.group ? kWrite : kRead;
  const int heads = side == kWrite ? p.write_heads : p.write_heads * p.group;
  const int head = side == kWrite ? s.w : s.w * p.group + row;
  const long long stream = static_cast<long long>(s.b) * p.write_heads + s.w;
  const long long at =
      ((static_cast<long long>(s.b) * heads + head) * tiles + s.tile) * tile_bytes(width);
  zero_run(p.rows[side][level] + at, static_cast<int>(tile_bytes(width)), mine);
  if constexpr (LO) zero_run(p.lo[side][level] + at, static_cast<int>(tile_bytes(width)), mine);
  const long long first = (static_cast<long long>(s.b) * p.tokens + s.t0) * heads + head;
  zero_tokens(reinterpret_cast<uint8_t*>(p.scales[side][level] + first), s.t1 - s.t0, 4,
              heads * 4ll, mine);
  if (own_mask && p.masked[row][level]) {
    const int words = (p.tokens + kWordTokens - 1) / kWordTokens, w0 = s.t0 / kWordTokens;
    const int w1 = (s.t1 + kWordTokens - 1) / kWordTokens;
    uint32_t* const m = p.masks[row][level] + (stream * words + w0) * width;
    zero_run(reinterpret_cast<uint8_t*>(m), (w1 - w0) * width * 4, mine);
  }
}

//: THE VALUES' SHARE and its growth's part (the values' stub) -- see
//: docs/internals/produce/produce.md#values
__device__ __forceinline__ void store_values(const Plan& p, const Share& s, int mine) {
  const long long first = (static_cast<long long>(s.b) * p.tokens + s.t0) * p.write_heads + s.w;
  zero_tokens(reinterpret_cast<uint8_t*>(p.values + first * p.d_v), s.t1 - s.t0, p.d_v * 2,
              static_cast<long long>(p.write_heads) * p.d_v * 2, mine);
  float* const sums = p.growth_sums + (static_cast<long long>(s.b) * p.write_heads + s.w) * p.d_v;
#pragma unroll 1
  for (int ch = mine; ch < p.d_v; ch += kGroupThreads) atomicAdd(sums + ch, 0.0f);
}

//: A JOB'S EPILOGUE on a warpgroup's share -- see docs/internals/produce/produce.md#epilogue
template <bool LO>
__device__ __forceinline__ void epilogue(const Plan& p, const Job& job, const Share& s, int mine) {
  if (s.t1 <= s.t0) return;
  if (job.kind == kWriteJob) {
    store_level<LO>(p, s, p.group, job.level, true, mine);
#pragma unroll 1
    for (int g = 0; g < p.group; ++g)
      if ((job.tied >> g) & 1u) store_level<LO>(p, s, g, job.level, false, mine);
  } else if (job.kind == kReadJob) {
    store_level<LO>(p, s, job.head, job.level, true, mine);
  } else {
    store_values(p, s, mine);
  }
}

//: THE LOADER: a stage a K-step of each job (the stub stages no bytes) -- see
//: docs/internals/produce/produce.md#ring
template <int STAGES, class Tr>
__device__ __forceinline__ void load_role(const Plan& p, ops::SmemAddr full, ops::SmemAddr empty,
                                          int lane, Tr& tr) {
  const int stages = p.jobs * ((p.hidden + kKStep - 1) / kKStep);
  if (lane != 0) return;
#pragma unroll 1
  for (int i = 0; i < stages; ++i) {
    const int slot = i % STAGES;
    if (i >= STAGES) {
      tr.wait_begin(kEmptyId + slot);
      ops::mbar_wait(empty + 8 * slot, ((i / STAGES) - 1) & 1);
      tr.wait_end(kEmptyId + slot);
    }
    tr.arrive(slot);
    ops::mbar_arrive(full + 8 * slot);
  }
}

//: A MATH WARPGROUP: each job's K-steps (the ring's edges) then its epilogue -- see
//: docs/internals/produce/produce.md#math
template <int STAGES, bool LO, class Tr>
__device__ __forceinline__ void math_role(const Plan& p, const Share& s, ops::SmemAddr full,
                                          ops::SmemAddr empty, int mine, Tr& tr) {
  const int steps = (p.hidden + kKStep - 1) / kKStep;
#pragma unroll 1
  for (int j = 0; j < p.jobs; ++j) {
    tr.phase(kSiteWindow);
#pragma unroll 1
    for (int k = 0; k < steps; ++k) {
      const int i = j * steps + k, slot = i % STAGES;
      tr.wait_begin(slot);
      ops::mbar_wait(full + 8 * slot, (i / STAGES) & 1);
      tr.wait_end(slot);
      ops::warp_sync();
      if ((mine & 31) == 0) {
        tr.arrive(kEmptyId + slot);
        ops::mbar_arrive(empty + 8 * slot);
      }
    }
    tr.phase(kSiteEpilogue);
    epilogue<LO>(p, p.job[j], s, mine);
  }
}

//: THE TICKETS: a write head's sequence's growth by its last CTA, the density by the launch's last --
//: see docs/internals/produce/produce.md#density
__device__ __forceinline__ void tickets(const Plan& p, const unsigned* live, int b, int w,
                                        int tiles) {
#pragma unroll 1
  for (int row = 0; row <= p.group; ++row)
#pragma unroll 1
    for (int l = 0; l < p.levels; ++l)
      if (p.masked[row][l])
        atomicAdd(&p.counts[row * kMaxLevels + l],
                  static_cast<unsigned long long>(live[row * kMaxLevels + l]));
  __threadfence();
  const long long stream = static_cast<long long>(b) * p.write_heads + w;
  if (atomicAdd(p.sequence_tickets + stream, 1u) == static_cast<unsigned>(tiles - 1)) {
    __threadfence();
#pragma unroll 1
    for (int ch = 0; ch < p.d_v; ++ch)
      p.growth[stream * p.d_v + ch] = atomicExch(p.growth_sums + stream * p.d_v + ch, 0.0f);
    p.sequence_tickets[stream] = 0u;
  }
  if (atomicAdd(p.ticket, 1u) != gridDim.x - 1) return;
  __threadfence();
#pragma unroll 1
  for (int row = 0; row <= p.group; ++row)
#pragma unroll 1
    for (int l = 0; l < p.levels; ++l) {
      const unsigned long long total =
          static_cast<unsigned long long>(p.batch) * p.tokens * p.write_heads * p.widths[l];
      const unsigned long long n = atomicExch(&p.counts[row * kMaxLevels + l], 0ull);
      p.density[row * p.levels + l] =
          p.masked[row][l] ? static_cast<uint8_t>((2ull * 255ull * n + total) / (2ull * total))
                           : uint8_t{255};
    }
  *p.ticket = 0u;
}

//: ONE CTA: a sequence's token tile and a write head, tiles fastest -- see
//: docs/internals/produce/produce.md#cta
template <class Bt, int CTAS, bool LO, class Tr>
__device__ __forceinline__ void produce_cta(const Plan& p, Tr& tr) {
  using S = Smem<Bt, CTAS>;
  constexpr int kDepth = kStages<Bt, CTAS>;
  static_assert(kDepth >= 2, "the ring holds two stages on every card");
  tr.phase(kSiteEntry);
  const int tid = static_cast<int>(threadIdx.x);
  const int tiles = (p.tokens + kTileTokens - 1) / kTileTokens;
  const int item = static_cast<int>(blockIdx.x), sequences = p.batch * tiles;
  const int b = item % sequences / tiles, tile = item % tiles, w = item / sequences;
  unsigned* const live = S::template get<Live>();
  const ops::SmemAddr full = ops::smem_addr(S::template get<Bars>()), empty = full + 8 * kMaxStages;
  tr.phase(kSiteSetup);
  if (tid < kMaxRows * kMaxLevels) live[tid] = 0u;
  if (tid == 0)
#pragma unroll
    for (int s = 0; s < kDepth; ++s) {
      ops::mbar_init(full + 8 * s, 1);
      ops::mbar_init(empty + 8 * s, kMathWarps);
    }
  __syncthreads();
  if (tid < kMathThreads) {
    const int group = tid / kGroupThreads, t0 = tile * kTileTokens + group * kGroupTokens;
    const Share s{b, 2 * tile + group, min(t0, p.tokens), min(t0 + kGroupTokens, p.tokens), w};
    math_role<kDepth, LO>(p, s, full, empty, tid % kGroupThreads, tr);
  } else {
    load_role<kDepth>(p, full, empty, tid & 31, tr);
  }
  tr.phase(kSiteTicket);
  __syncthreads();
  if (tid == 0) tickets(p, live, b, w, tiles);
  tr.phase(kSiteExit);
}

}  // namespace detail
}  // namespace produce
}  // namespace fsla
