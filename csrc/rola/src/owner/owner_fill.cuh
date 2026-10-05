// csrc/rola/src/owner/owner_fill.cuh -- THE FILL: a pass of the window's live tokens into the write pool (a token's
// write digits at both levels, f16 on a dense level and u8 on a sparse one by its live sectors, its f16 values for the
// CTA's channels and its factor products) or the read pool (its read-dense slice and its read-sparse slice), from the
// bundle where it lies.
// see docs/internals/owner/owner_fill.md
#pragma once

#include <cstdint>

#include "common/ops.cuh"
#include "owner/owner.cuh"
#include "owner/owner_lists.cuh"

namespace rola::owner::detail {

//: A WRITE ROW for a write format (`kHalfS`, `kHalfR`: level S's and level R's write digits f16, a dense level's, or
//: u8), every copied run at its global sector's 32-byte phase: at 16 x 64 level R's run, the f16 values, level S's
//: slice and the token's `(F, m)`; at 32 x 32 level S's run then level R's, the values and `(F, m)`.
//: -- see docs/internals/owner/owner_fill.md#pools
template <class C_, bool kHalfS_, bool kHalfR_>
struct WriteRows {
  using C = C_;
  static constexpr bool kHalfS = kHalfS_, kHalfR = kHalfR_, kSplit = C::RS == 16;
  static constexpr int kSBytes = (kHalfS ? 2 : 1) * C::RS, kRBytes = (kHalfR ? 2 : 1) * C::RD;
  static constexpr int kRun = kSplit ? kRBytes : kSBytes + kRBytes;
  static constexpr int kR = kSplit ? 0 : kSBytes, kValues = kRun,
                       kS = kSplit ? kValues + 2 * kChannels : 0;
  static constexpr int kFm = kSBytes + kRBytes + 2 * kChannels;
  static constexpr int kRow = (kFm + 8 + 31) & ~31;
  static_assert(kValues % 32 == 0 && kS % 32 == 0 && kFm % 8 == 0,
                "every run at its sector's phase");
};

//: A READ ROW: a 64-byte run (the read-dense slice, or at 32 x 32 both slices) and at 16 x 64 the read-sparse chunk.
template <class C>
struct ReadRows {
  static constexpr int kRun = 64;
  static constexpr int kRow = C::RS == 16 ? 96 : 64;
  static_assert(C::RD + (C::RS == 32 ? C::RS : 0) == kRun, "the run is the read slices");
};

//: THE WRITE POOL, the fold's: a pass's rows.
template <class W>
struct alignas(128) WritePool {
  uint8_t write[kWritePool][W::kRow];
};

//: THE READ POOL, the readout's.
template <class C>
struct alignas(128) ReadPool {
  uint8_t read[kReadPool][ReadRows<C>::kRow];
};

//: A PLANE'S BASE FOR THE CTA'S (batch, head): a token's row is this plus its token stride.
struct PlaneBase {
  const uint8_t* at;
  int64_t st;
};

__device__ __forceinline__ PlaneBase plane(const RowView& v, int b, int h, int64_t offset) {
  return PlaneBase{static_cast<const uint8_t*>(v.base) + b * v.sb + h * v.sh + offset, v.st};
}

__device__ __forceinline__ const uint8_t* token_row(const PlaneBase& pb, int tg) {
  return pb.at + tg * pb.st;
}

//: A LANE'S WRITE PLANES: a quad fills one token, lane `q` taking the run's 16-byte quanta `q, q + 4, ...`, its value
//: quantum, at 16 x 64 level S's quantum `q` if the slice has one, and lane 3 the token's `(F, m)`; the bases computed
//: once. -- see docs/internals/owner/owner_fill.md#quads
template <class W>
struct WriteLanes {
  static constexpr int kRunQuanta = W::kRun / 16, kRunSlots = (kRunQuanta + 3) / 4;
  PlaneBase run[kRunSlots], values, slice, fm;
  int q;

  __device__ __forceinline__ WriteLanes(const OwnerParams& p, int b, int h, int cg, int s0, int r0,
                                        int tid)
      : q(tid & 3) {
    const int64_t s_at = (W::kHalfS ? 2 : 1) * s0, r_at = (W::kHalfR ? 2 : 1) * r0;
    values = plane(p.v, b, h, 2 * cg * kChannels + 16 * q);
    slice = plane(p.write[p.level_s], b, h, s_at + 16 * q);
    fm = PlaneBase{
        reinterpret_cast<const uint8_t*>(p.fm + (static_cast<int64_t>(b) * p.H + h) * p.L),
        static_cast<int64_t>(sizeof(float2))};
#pragma unroll
    for (int j = 0; j < kRunSlots; ++j) {
      const int at = 16 * (q + 4 * j);
      if constexpr (W::kSplit)
        run[j] = plane(p.write[p.level_r], b, h, r_at + at);
      else
        run[j] = at < W::kSBytes ? plane(p.write[p.level_s], b, h, s_at + at)
                                 : plane(p.write[p.level_r], b, h, r_at + at - W::kSBytes);
    }
  }
};

//: A LANE'S READ PLANES: the read run's quantum `q` and, at 16 x 64, the read-sparse chunk.
template <class C>
struct ReadLanes {
  PlaneBase run, tail;

  __device__ __forceinline__ ReadLanes(const OwnerParams& p, int b, int h, int s0, int r0,
                                       int tid) {
    const int q = tid & 3;
    if constexpr (C::RS == 16) {
      run = plane(p.read[p.level_r], b, h, r0 + 16 * q);
      tail = plane(p.read[p.level_s], b, h, s0);
    } else {
      run = q < 2 ? plane(p.read[p.level_r], b, h, r0 + 16 * q)
                  : plane(p.read[p.level_s], b, h, s0 + 16 * (q - 2));
      tail = run;
    }
  }
};

//: A WRITE PASS: slots `[first, first + n)` of the window's write union, a quad a token: the row's run (a sparse level
//: R's 64 bytes at 16 x 64 by its live sectors, a sector the token is not live in landing as zeros and reading
//: nothing), the f16 values' 64 bytes, level S's slice at 16 x 64, and the token's `(F, m)`.
//: -- see docs/internals/owner/owner_fill.md#write-pass
template <class W>
__device__ void fill_write(WritePool<W>& pools, const ListTenants<typename W::C>& sm,
                           const WriteLanes<W>& lanes, int win, int first, int n, int tid) {
  constexpr int kRunSlots = WriteLanes<W>::kRunSlots, kRunQuanta = WriteLanes<W>::kRunQuanta;
  const int q = lanes.q;
#pragma unroll 1
  for (int r = tid >> 2; r < n; r += kThreads / 4) {
    const int t = sm.token_of[0][first + r];
    const int tg = win * kWindow + t;
    uint8_t* const row = pools.write[r];
#pragma unroll
    for (int j = 0; j < kRunSlots; ++j) {
      const ops::SmemAddr dst = ops::smem_addr(row + 16 * (q + 4 * j));
      if constexpr (W::kSplit && !W::kHalfR) {
        const bool live = (sm.sector_uni[q >> 1][t >> 5] >> (t & 31)) & 1u;
        ops::stage_tile_if(dst, token_row(lanes.run[j], tg), live);
      } else if (4 * (j + 1) <= kRunQuanta) {
        ops::stage_tile(dst, token_row(lanes.run[j], tg));
      } else {
        ops::stage_run_lanes<16>(dst, token_row(lanes.run[j], tg), q + 4 * j < kRunQuanta);
      }
    }
    ops::stage_tile(ops::smem_addr(row + W::kValues + 16 * q), token_row(lanes.values, tg));
    if constexpr (W::kSplit)
      ops::stage_run_lanes<16>(ops::smem_addr(row + W::kS + 16 * q), token_row(lanes.slice, tg),
                               q < W::kSBytes / 16);
    ops::stage_run_lanes<8>(ops::smem_addr(row + W::kFm), token_row(lanes.fm, tg), q == 3);
  }
  ops::stage_commit();
}

//: A READ PASS: slots `[first, first + n)` of the window's read union, a quad a token: the read-dense slice's 64-byte
//: run and the read-sparse chunk by the first lane (16 x 64), or the two slices' 32-byte runs (32 x 32).
template <class C>
__device__ void fill_read(ReadPool<C>& pools, const ListTenants<C>& sm, const ReadLanes<C>& lanes,
                          int win, int first, int n, int tid) {
  const int q = tid & 3;
#pragma unroll 1
  for (int r = tid >> 2; r < n; r += kThreads / 4) {
    const int tg = win * kWindow + sm.token_of[1][first + r];
    uint8_t* const row = pools.read[r];
    ops::stage_tile(ops::smem_addr(row + 16 * q), token_row(lanes.run, tg));
    if constexpr (C::RS == 16)
      ops::stage_run_lanes<16>(ops::smem_addr(row + ReadRows<C>::kRun), token_row(lanes.tail, tg),
                               q == 0);
  }
  ops::stage_commit();
}

}  // namespace rola::owner::detail
