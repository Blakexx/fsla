// csrc/rola/src/owner/owner_fill.cuh -- THE FILL: a pass of the window's live tokens into the write pool (a token's
// write-dense slice, the live sectors of its write-sparse slice, its value digits for the CTA's channels and its
// factors) or the read pool (its read-dense slice and its read-sparse slice), from the bundle where it lies.
// see docs/internals/owner/owner_fill.md
#pragma once

#include <cstdint>

#include "common/ops.cuh"
#include "owner/owner.cuh"
#include "owner/owner_lists.cuh"

namespace rola::owner::detail {

//: THE POOLS: a write row is the dense slice, the sparse slice, two value planes of the channel group and four factors;
//: a read row the read-dense slice then the read-sparse slice. -- see docs/internals/owner/owner_fill.md#pools
template <class C>
struct Pools {
  static constexpr int kWriteRow = C::RS + C::RD + 2 * kChannels + 16;
  static constexpr int kReadRow = C::RD + C::RS;
  uint8_t write[kWritePool][kWriteRow];
  uint8_t read[kReadPool][kReadRow];
};

//: A PLANE'S BASE FOR THE CTA'S (batch, head): a token's row is this plus its token stride.
struct PlaneBase {
  const uint8_t* at;
  int64_t st;
};

__device__ __forceinline__ PlaneBase plane(const RowView& v, int b, int h, int64_t offset) {
  return PlaneBase{static_cast<const uint8_t*>(v.base) + b * v.sb + h * v.sh + offset, v.st};
}

__device__ __forceinline__ PlaneBase plane(const ColumnView& v, int b, int h) {
  return PlaneBase{reinterpret_cast<const uint8_t*>(v.base + b * v.sb + h * v.sh), v.st * 4};
}

__device__ __forceinline__ const uint8_t* token_row(const PlaneBase& pb, int tg) {
  return pb.at + tg * pb.st;
}

//: A LANE'S PLANES: a quad fills one token, lane `q` taking the same part of every token it fills, so its plane bases
//: are computed once. -- see docs/internals/owner/owner_fill.md#quads
template <class C>
struct FillLanes {
  PlaneBase run, values, factor, dense, read_run, read_tail;
  int q;

  __device__ __forceinline__ FillLanes(const OwnerParams& p, int b, int h, int cg, int s0, int r0,
                                       int tid)
      : q(tid & 3) {
    const RowView& v = q < 2 ? p.v_hat : p.v_lo;
    values = plane(v.base != nullptr ? v : p.v_hat, b, h, cg * kChannels + 16 * (q & 1));
    factor = plane(p.factor[q], b, h);
    if constexpr (C::RS == 16) {
      run = plane(p.write[p.level_r], b, h, r0 + 16 * q);
      dense = plane(p.write[p.level_s], b, h, s0);
      read_run = plane(p.read[p.level_r], b, h, r0 + 16 * q);
      read_tail = plane(p.read[p.level_s], b, h, s0);
    } else {
      run = q < 2 ? plane(p.write[p.level_s], b, h, s0 + 16 * q)
                  : plane(p.write[p.level_r], b, h, r0 + 16 * (q - 2));
      read_run = q < 2 ? plane(p.read[p.level_r], b, h, r0 + 16 * q)
                       : plane(p.read[p.level_s], b, h, s0 + 16 * (q - 2));
      dense = run;
      read_tail = read_run;
    }
  }
};

//: A WRITE PASS: slots `[first, first + n)` of the window's write union, a quad a token: the 64-byte run of the sparse
//: slice (16 x 64) or the dense and sparse slices' 32-byte runs (32 x 32), the two value planes' 32-byte runs, the four
//: factors, and at 16 x 64 the dense slice's chunk by the quad's first lane. A sparse sector the token is not live in
//: lands as zeros and reads nothing. -- see docs/internals/owner/owner_fill.md#write-pass
template <class C>
__device__ void fill_write(Pools<C>& pools, const ListTenants<C>& sm, const FillLanes<C>& lanes,
                           const OwnerParams& p, int win, int first, int n, int tid) {
  const int q = lanes.q;
  const bool values_live = q < 2 || p.v_lo.base != nullptr;
#pragma unroll 1
  for (int r = tid >> 2; r < n; r += kThreads / 4) {
    const int t = sm.token_of[0][first + r];
    const int tg = win * kWindow + t;
    uint8_t* const row = pools.write[r];
    if constexpr (C::RS == 16) {
      const bool live = (sm.sector_uni[q >> 1][t >> 5] >> (t & 31)) & 1u;
      ops::stage_tile_if(ops::smem_addr(row + C::RS + 16 * q), token_row(lanes.run, tg), live);
      ops::stage_run_lanes<16>(ops::smem_addr(row), token_row(lanes.dense, tg), q == 0);
    } else {
      ops::stage_tile(ops::smem_addr(row + 16 * q), token_row(lanes.run, tg));
    }
    ops::stage_tile_if(ops::smem_addr(row + C::RS + C::RD + 16 * q), token_row(lanes.values, tg),
                       values_live);
    ops::stage_run<4>(ops::smem_addr(row + C::RS + C::RD + 2 * kChannels + 4 * q),
                      token_row(lanes.factor, tg));
  }
  ops::stage_commit();
}

//: A READ PASS: slots `[first, first + n)` of the window's read union, a quad a token: the read-dense slice's 64-byte
//: run and the read-sparse chunk by the first lane (16 x 64), or the two slices' 32-byte runs (32 x 32).
template <class C>
__device__ void fill_read(Pools<C>& pools, const ListTenants<C>& sm, const FillLanes<C>& lanes,
                          int win, int first, int n, int tid) {
  const int q = lanes.q;
#pragma unroll 1
  for (int r = tid >> 2; r < n; r += kThreads / 4) {
    const int tg = win * kWindow + sm.token_of[1][first + r];
    uint8_t* const row = pools.read[r];
    ops::stage_tile(ops::smem_addr(row + 16 * q), token_row(lanes.read_run, tg));
    if constexpr (C::RS == 16)
      ops::stage_run_lanes<16>(ops::smem_addr(row + C::RD), token_row(lanes.read_tail, tg), q == 0);
  }
  ops::stage_commit();
}

}  // namespace rola::owner::detail
