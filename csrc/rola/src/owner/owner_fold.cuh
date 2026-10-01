// csrc/rola/src/owner/owner_fold.cuh -- THE FOLD, as the skeleton prices it: a warp's write lists over a pass of the
// write pool, per list or merged, each k-step 32 tokens gathered from their pool rows by `ldmatrix.trans` -- the dense
// slice as A (leaves on M), the two value digits as B (channels on N) -- into the resident s32 state. The factor and
// scale formation is not built (the fold's arithmetic line owns it); the counts and the gathers are real.
// see docs/internals/owner/owner_fold.md
#pragma once

#include <cstdint>

#include "common/ops.cuh"
#include "common/static_for.cuh"
#include "owner/owner.cuh"
#include "owner/owner_fill.cuh"
#include "owner/owner_lists.cuh"

namespace rola::owner::detail {

//: A WARP'S FOLD STATE: per write list, the box's leaves (M) by the channel group (N), s32.
template <class C>
struct FoldState {
  static constexpr int kMTiles = C::RS / 16;
  int32_t acc[C::kFoldLists][kMTiles][4][4];
};

//: THE ZERO ROW a k-step's idle lanes address: the pool's last row, which a pass of `kWritePass` slots never fills.
template <class C>
__device__ __forceinline__ const uint8_t* zero_row(const Pools<C>& pools) {
  return pools.write[kWritePass];
}

//: THE B OPERAND OF A K-STEP: the 32 tokens' two value digits for the channel group, by four `ldmatrix.x4.trans` and a
//: byte permute a register -- the even and odd channels of each 16-channel half. -- see docs/internals/owner/owner_fold.md#operands
template <class C>
__device__ __forceinline__ void load_values(const uint8_t* row, uint32_t (&b)[2][4][2]) {
#pragma unroll
  for (int digit = 0; digit < 2; ++digit)
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      uint32_t v[4];
      ops::load_frag_t(v, ops::smem_addr(row + Pools<C>::kValues + 32 * digit + 16 * half));
      b[digit][2 * half][0] = ops::prmt(v[0], v[1], 0x6420u);
      b[digit][2 * half][1] = ops::prmt(v[2], v[3], 0x6420u);
      b[digit][2 * half + 1][0] = ops::prmt(v[0], v[1], 0x7531u);
      b[digit][2 * half + 1][1] = ops::prmt(v[2], v[3], 0x7531u);
    }
}

//: THE A OPERAND OF A K-STEP for one list: the 32 tokens' dense slices, leaves on M, by `ldmatrix.x4.trans` and a byte
//: permute a register, an m-tile a load.
template <class C>
__device__ __forceinline__ void load_dense(const uint8_t* row,
                                           uint32_t (&a)[FoldState<C>::kMTiles][4]) {
  constexpr int kDense = C::RS == 16 ? Pools<C>::kTail : 0;
  constexpr int kMTiles = FoldState<C>::kMTiles;
#pragma unroll
  for (int m = 0; m < kMTiles; ++m) {
    uint32_t v[4];
    ops::load_frag_t(v, ops::smem_addr(row + kDense + 16 * m));
    a[m][0] = ops::prmt(v[0], v[1], 0x6420u);
    a[m][1] = ops::prmt(v[0], v[1], 0x7531u);
    a[m][2] = ops::prmt(v[2], v[3], 0x6420u);
    a[m][3] = ops::prmt(v[2], v[3], 0x7531u);
  }
}

//: A LIST'S IMMAS OF A K-STEP: m-tiles x four n-tiles x two value digits (v_hat signed, v_lo unsigned).
//: @burst
template <class C>
__device__ __forceinline__ void fold_immas(int32_t (&acc)[FoldState<C>::kMTiles][4][4],
                                           const uint32_t (&a)[FoldState<C>::kMTiles][4],
                                           const uint32_t (&b)[2][4][2]) {
  constexpr int kMTiles = FoldState<C>::kMTiles;
#pragma unroll
  for (int m = 0; m < kMTiles; ++m)
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      ops::imma_u8s8_k32(acc[m][n], a[m], b[0][n][0], b[0][n][1]);
      ops::imma_u8_k32(acc[m][n], a[m], b[1][n][0], b[1][n][1]);
    }
}

//: A LANE'S ROW FOR A K-STEP: the pool row of entry `e` of the warp's buffer when it lies in `[at, end)`, else the zero
//: row; `ldmatrix.x4` takes token `8 (lane / 8) + lane % 8` of the step from this lane's address.
template <class C>
__device__ __forceinline__ const uint8_t* step_row(const Pools<C>& pools, const ListTenants<C>& sm,
                                                   int warp, int at, int end, int first, int lane,
                                                   bool listed) {
  const int e = at + lane;
  if (e >= end) return zero_row(pools);
  const int slot = listed ? sm.entries[0][warp][e] : e;
  return pools.write[slot - first];
}

//: The end of a sorted run of entries in this pass: the first index from `at` whose slot is past `limit`
template <class C>
__device__ __forceinline__ int pass_end(const ListTenants<C>& sm, int warp, int at, int total,
                                        int limit, int lane) {
  int end = at;
#pragma unroll 1
  while (true) {
    const int e = end + lane;
    const bool in = e < total && sm.entries[0][warp][e] < limit;
    const uint32_t mask = __ballot_sync(0xffffffffu, in);
    end += __popc(mask);
    if (mask != 0xffffffffu) return end;
  }
}

//: A WINDOW'S CURSORS: each list's first entry (per list), or the union's (merged).
template <class C>
__device__ __forceinline__ void fold_cursors(const ListTenants<C>& sm, int (&cursor)[C::kFoldLists],
                                             int warp) {
  constexpr int kLists = C::kFoldLists;
  int off = 0;
#pragma unroll
  for (int j = 0; j < kLists; ++j) {
    cursor[j] = off;
    off += sm.count[warp * C::kFoldLists + j];
  }
}

//: A WARP'S FOLD OF ONE PASS (slots `[first, first + n)`), per its window form: per list, each list's entries in the
//: pass k-step by k-step; merged (or a dense side), the warp union's, the value operand once a step for every list.
//: `cursor` carries each list's (or the union's) next entry across passes. -- see docs/internals/owner/owner_fold.md#forms
template <class C>
__device__ void fold_pass(const Pools<C>& pools, const ListTenants<C>& sm, const OwnerParams& p,
                          FoldState<C>& st, int (&cursor)[C::kFoldLists], int first, int n,
                          int warp, int lane) {
  const int row0 = warp * C::kFoldLists;
  const int limit = first + n;
  uint32_t b[2][4][2];
  uint32_t a[FoldState<C>::kMTiles][4];
  if (p.sparse_write != 0 && sm.form[0][warp] == kStepPerList) {
    int off = 0;
    rola::static_for<C::kFoldLists>([&](auto J) {
      constexpr int kJ = decltype(J)::value;
      const int total = off + sm.count[row0 + kJ];
      const int end = pass_end(sm, warp, cursor[kJ], total, limit, lane);
#pragma unroll 1
      for (int at = cursor[kJ]; at < end; at += 32) {
        const uint8_t* row = step_row(pools, sm, warp, at, end, first, lane, true);
        load_values<C>(row, b);
        load_dense<C>(row, a);
        fold_immas<C>(st.acc[kJ], a, b);
      }
      cursor[kJ] = end;
      off = total;
    });
    return;
  }
  const bool listed = p.sparse_write != 0;
  const int at0 = listed ? cursor[0] : first;
  const int end = listed ? pass_end(sm, warp, at0, sm.total[0][warp], limit, lane) : limit;
#pragma unroll 1
  for (int at = at0; at < end; at += 32) {
    const uint8_t* row = step_row(pools, sm, warp, at, end, first, lane, listed);
    load_values<C>(row, b);
    rola::static_for<C::kFoldLists>([&](auto J) {
      load_dense<C>(row, a);
      fold_immas<C>(st.acc[decltype(J)::value], a, b);
    });
  }
  cursor[0] = end;
}

}  // namespace rola::owner::detail
