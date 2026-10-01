// csrc/rola/src/owner/owner_lists.cuh -- THE WINDOW'S WORDS AND LISTS: the CTA's write and read rows of the facts
// pass's words staged a window ahead, their unions and ranks (a live token's pool slot), each row's live count, each
// warp's step form chosen from those exact counts, and the warp's list entries for that form.
// see docs/internals/owner/owner_lists.md
#pragma once

#include <cstdint>

#include "common/ops.cuh"
#include "facts/liveness_contract.cuh"
#include "owner/owner.cuh"

namespace rola::owner::detail {

namespace ops = denseref::ops;

//: A STEP'S PRICE in issue slots, the one cost model both forms are compared by: a k-step's coefficient operand, its
//: token operand, its IMMAs, and a list entry. -- see docs/internals/owner/owner_lists.md#step-cost
constexpr int kCostOperandA = 4;
constexpr int kCostOperandB = 12;
constexpr int kCostImmas = 8 * 16;
constexpr int kCostEntry = 6;

//: THE TENANTS THE LISTS USE: the staged rows (write rows first, then read rows) in two buffers, the unions, their ranks,
//: each row's live count, each warp's forms and entries (pool slots, one buffer a side a warp).
template <class C>
struct ListTenants {
  uint32_t words[2][C::kWordRows][kWindowWords];
  uint32_t partial[2][16][kWindowWords];
  uint32_t uni[2][kWindowWords];
  uint16_t base[2][kWindowWords];
  uint16_t count[C::kWordRows];
  uint16_t entries[2][kWarps][kListCapacity];
  uint16_t total[2][kWarps];
  uint8_t form[2][kWarps];
};

//: A ROW'S PLACE: write rows `[0, RD)` are level `level_r`'s write digits, read rows `[RD, RD + RS)` level `level_s`'s
//: read digits.
template <class C>
__device__ __forceinline__ bool row_is_write(int row) {
  return row < C::RD;
}

template <class C>
__device__ __forceinline__ bool row_sparse(const OwnerParams& p, int row) {
  return row_is_write<C>(row) ? p.sparse_write != 0 : p.sparse_read != 0;
}

//: THE WORDS: window `win`'s rows into buffer `buf`, a 16-byte chunk a thread; a dense side's rows are not read.
//: -- see docs/internals/owner/owner_lists.md#words
template <class C>
__device__ __forceinline__ void stage_words(ListTenants<C>& sm, const OwnerParams& p, int bh,
                                            int s0, int r0, int win, int buf, int tid) {
#pragma unroll 1
  for (int c = tid; c < C::kWordRows * 4; c += kThreads) {
    const int row = c >> 2, part = c & 3;
    if (!row_sparse<C>(p, row)) continue;
    const bool write = row_is_write<C>(row);
    const int side = write ? facts::kWrite : facts::kRead;
    const int lrow =
        write ? p.row_base[p.level_r] + r0 + row : p.row_base[p.level_s] + s0 + row - C::RD;
    const uint32_t* src = p.words + (static_cast<int64_t>(bh) * facts::kSides + side) * p.side_words
                          + lrow * p.row_words + win * kWindowWords + part * 4;
    ops::stage_tile(ops::smem_addr(&sm.words[buf][row][part * 4]), src);
  }
  ops::stage_commit();
}

//: The 16 lanes' inclusive prefix of `x` within the half-warp
__device__ __forceinline__ int half_prefix(int x, int lane) {
#pragma unroll
  for (int d = 1; d < 16; d <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, d, 16);
    if ((lane & 15) >= d) x += y;
  }
  return x;
}

__device__ __forceinline__ int half_sum(int x) {
#pragma unroll
  for (int d = 1; d < 16; d <<= 1) x += __shfl_xor_sync(0xffffffffu, x, d, 16);
  return x;
}

//: A WORD'S ENTRIES: each set bit of `bits` (word `k`) as its union slot, written from `at` in bit order
template <class C>
__device__ __forceinline__ void word_entries(ListTenants<C>& sm, int side, int warp, int k,
                                             uint32_t bits, int at) {
  const uint32_t uni = sm.uni[side][k];
  const int base = sm.base[side][k];
#pragma unroll 1
  for (int i = 0; bits != 0u; ++i) {
    const int b = __ffs(bits) - 1;
    bits &= bits - 1u;
    sm.entries[side][warp][at + i] = static_cast<uint16_t>(base + __popc(uni & ((1u << b) - 1u)));
  }
}

//: A WARP'S SIDE: its rows' union, the two forms' costs over the exact counts, the chosen form's entries. A dense
//: side's form is merged over every token, with no entries (the slot is the token).
template <class C, int kSide>
__device__ __forceinline__ void warp_lists(ListTenants<C>& sm, const OwnerParams& p, int buf,
                                           int warp, int lane) {
  constexpr int kLists = kSide == 0 ? C::kFoldLists : C::kReadDigits;
  const int row0 = kSide == 0 ? warp * C::kFoldLists : C::RD + warp * C::kReadDigits;
  const int k = lane & 15, half = lane >> 4;
  if ((kSide == 0 ? p.sparse_write : p.sparse_read) == 0) {
    if (lane == 0) {
      sm.form[kSide][warp] = kStepMerged;
      sm.total[kSide][warp] = kWindow;
    }
    return;
  }

  uint32_t u = 0u;
#pragma unroll
  for (int jj = 0; jj < kLists; jj += 2) u |= sm.words[buf][row0 + jj + half][k];
  u |= __shfl_xor_sync(0xffffffffu, u, 16);
  const int merged_n = half_sum(__popc(u));
  int listed = 0, steps = 0;
#pragma unroll
  for (int j = 0; j < kLists; ++j) {
    listed += sm.count[row0 + j];
    steps += (sm.count[row0 + j] + 31) >> 5;
  }
  const int per_list_cost =
      steps * (kCostOperandA + kCostOperandB + kCostImmas) + listed * kCostEntry;
  const int merged_cost =
      ((merged_n + 31) >> 5) * (kCostOperandB + kLists * (kCostOperandA + kCostImmas))
      + merged_n * kCostEntry;
  int form = p.step != kStepChosen          ? p.step
             : per_list_cost <= merged_cost ? kStepPerList
                                            : kStepMerged;
  if (listed > kListCapacity) form = kStepMerged;

  if (form == kStepMerged) {
    const int c = __popc(u);
    const int at = half_prefix(c, lane) - c;
    if (half == 0) word_entries(sm, kSide, warp, k, u, at);
  } else {
#pragma unroll
    for (int jj = 0; jj < kLists; jj += 2) {
      const int j = jj + half;
      int off = 0;
#pragma unroll
      for (int i = 0; i < kLists; ++i)
        if (i < j) off += sm.count[row0 + i];
      const uint32_t w = sm.words[buf][row0 + j][k];
      const int c = __popc(w);
      word_entries(sm, kSide, warp, k, w, off + half_prefix(c, lane) - c);
    }
  }
  if (lane == 0) {
    sm.form[kSide][warp] = static_cast<uint8_t>(form);
    sm.total[kSide][warp] = static_cast<uint16_t>(form == kStepMerged ? merged_n : listed);
  }
}

//: THE LISTS: the unions, ranks and row counts by the whole CTA, then each warp's fold lists (side 0) and read digits
//: (side 1). -- see docs/internals/owner/owner_lists.md#lists
template <class C>
__device__ void window_lists(ListTenants<C>& sm, const OwnerParams& p, int buf, int tid) {
  static_assert(
      C::kWordRows % 16 == 0 && C::kFoldLists % 2 == 0 && C::kReadDigits % 2 == 0,
      "the half-warps take rows and lists in pairs, so every loop around a shuffle has one trip count");
  constexpr int kRowTrips = C::kWordRows / 16;
  const int warp = tid >> 5, lane = tid & 31, k = tid & 15, q = tid >> 4;
  uint32_t part[2] = {0u, 0u};
#pragma unroll
  for (int i = 0; i < kRowTrips; ++i) {
    const int row = q + 16 * i;
    part[row_is_write<C>(row) ? 0 : 1] |= sm.words[buf][row][k];
    const int n = half_sum(row_sparse<C>(p, row) ? __popc(sm.words[buf][row][k]) : 32);
    if (k == 0) sm.count[row] = static_cast<uint16_t>(n);
  }
  sm.partial[0][q][k] = part[0];
  sm.partial[1][q][k] = part[1];
  __syncthreads();

  if (warp == 0) {
    const int side = lane >> 4;
    uint32_t u = 0u;
#pragma unroll
    for (int i = 0; i < 16; ++i) u |= sm.partial[side][i][k];
    if ((side == 0 ? p.sparse_write : p.sparse_read) == 0) u = ~0u;
    const int c = __popc(u);
    sm.uni[side][k] = u;
    sm.base[side][k] = static_cast<uint16_t>(half_prefix(c, lane) - c);
  }
  __syncthreads();

  warp_lists<C, 0>(sm, p, buf, warp, lane);
  warp_lists<C, 1>(sm, p, buf, warp, lane);
  __syncthreads();
}

}  // namespace rola::owner::detail
