// csrc/rola/src/owner/owner_lists.cuh -- THE WINDOW'S WORDS AND LISTS: the CTA's write and read rows of the facts
// pass's words staged a window ahead, their unions and ranks (a live token's pool slot), each row's live count, each
// warp's union words, and the read side's step form chosen from those exact counts with each warp's read entries (the
// fold builds its rows a pass from the union words).
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

//: THE TENANTS THE LISTS USE: the staged rows (write rows first, then read rows) in two buffers, the warps' and the
//: CTA's unions, their ranks, each row's live count, and each warp's read form, entries (pool slots) and their total.
template <class C>
struct ListTenants {
  uint32_t words[2][C::kWordRows][kWindowWords];
  uint32_t partial[2][kWarps][kWindowWords];
  uint32_t uni[2][kWindowWords];
  uint32_t sector_uni[C::RD / 32][kWindowWords];
  uint16_t base[2][kWindowWords];
  uint16_t token_of[2][kWindow];
  uint16_t count[C::kWordRows];
  uint16_t read_entries[kWarps][kListCapacity];
  uint16_t read_total[kWarps];
  uint8_t read_form[kWarps];
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

//: A WARP'S SIDE IN REGISTERS: its `kLists` rows, a lane `kWords` consecutive words of one (row `lane / kLanes`, words
//: from `kWords * (lane % kLanes)`), and what a few shuffles make of them: the row's count, the row's live tokens before
//: the lane's words, the row's first entry (the rows before it), the warp union's words of the lane's group and the
//: union's tokens before them. -- see docs/internals/owner/owner_lists.md#lanes
template <class C, int kSide>
struct SideLanes {
  static constexpr int kLists = kSide == 0 ? C::kFoldLists : C::kReadDigits;
  static constexpr int kLanes = 32 / kLists;
  static constexpr int kWords = kWindowWords / kLanes;
  uint32_t w[kWords], u[kWords];
  int row0, row, group, count, before, offset, listed, steps, union_count, union_before;

  __device__ __forceinline__ SideLanes(const ListTenants<C>& sm, int buf, int warp, int lane)
      : row0(kSide == 0 ? warp * C::kFoldLists : C::RD + warp * C::kReadDigits),
        row(lane / kLanes),
        group(lane % kLanes) {
    int c = 0;
#pragma unroll
    for (int i = 0; i < kWords; ++i) {
      w[i] = sm.words[buf][row0 + row][kWords * group + i];
      c += __popc(w[i]);
    }
    const int incl = group_prefix(c);
    before = incl - c;
    count = __shfl_sync(0xffffffffu, incl, kLanes - 1, kLanes);
    int rows = count;
#pragma unroll
    for (int d = 1; d < kLists; d <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, rows, d * kLanes);
      if (row >= d) rows += y;
    }
    offset = rows - count;
    listed = __shfl_sync(0xffffffffu, rows, 31);
    int s = group == 0 ? (count + 31) >> 5 : 0;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) s += __shfl_xor_sync(0xffffffffu, s, d);
    steps = s;
    int uc = 0;
#pragma unroll
    for (int i = 0; i < kWords; ++i) {
      u[i] = w[i];
#pragma unroll
      for (int d = kLanes; d < 32; d <<= 1) u[i] |= __shfl_xor_sync(0xffffffffu, u[i], d);
      uc += __popc(u[i]);
    }
    const int uincl = group_prefix(uc);
    union_before = uincl - uc;
    union_count = __shfl_sync(0xffffffffu, uincl, kLanes - 1, kLanes);
  }

  //: the inclusive prefix of `x` over the lanes of the row
  static __device__ __forceinline__ int group_prefix(int x) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
#pragma unroll
    for (int d = 1; d < kLanes; d <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, x, d, kLanes);
      if (lane % kLanes >= d) x += y;
    }
    return x;
  }
};

//: A WARP'S FORM for a side from the exact counts: the cheaper of per list and merged by the step cost, merged when the
//: lists' entries pass the buffer. -- see docs/internals/owner/owner_lists.md#step-cost
template <class C, int kSide>
__device__ __forceinline__ int side_form(const SideLanes<C, kSide>& sl, int step) {
  using S = SideLanes<C, kSide>;
  constexpr int kWords = S::kWords;
  const int per_list_cost =
      sl.steps * (kCostOperandA + kCostOperandB + kCostImmas) + sl.listed * kCostEntry;
  const int merged_cost =
      ((sl.union_count + 31) >> 5) * (kCostOperandB + S::kLists * (kCostOperandA + kCostImmas))
      + sl.union_count * kCostEntry;
  const int form = step != kStepChosen            ? step
                   : per_list_cost <= merged_cost ? kStepPerList
                                                  : kStepMerged;
  return sl.listed > kListCapacity ? kStepMerged : form;
}

//: A WARP'S SIDE, FIRST HALF: the rows' counts and the warp union's words into shared memory; a dense side's counts
//: are the window and its union every token.
template <class C, int kSide>
__device__ __forceinline__ void side_counts(ListTenants<C>& sm, const SideLanes<C, kSide>& sl,
                                            bool sparse, int warp, int lane) {
  using S = SideLanes<C, kSide>;
  constexpr int kWords = S::kWords;
  if (sl.group == 0)
    sm.count[sl.row0 + sl.row] = static_cast<uint16_t>(sparse ? sl.count : kWindow);
  if (sl.row == 0) {
#pragma unroll
    for (int i = 0; i < kWords; ++i)
      sm.partial[kSide][warp][S::kWords * sl.group + i] = sparse ? sl.u[i] : ~0u;
  }
  (void)lane;
}

//: A WORD'S READ ENTRIES: each set bit of `bits` (word `k`) as its read union slot, written from `at` in bit order
template <class C>
__device__ __forceinline__ void word_entries(ListTenants<C>& sm, int warp, int k, uint32_t bits,
                                             int at) {
  const uint32_t uni = sm.uni[1][k];
  const int base = sm.base[1][k];
#pragma unroll 1
  for (int i = 0; bits != 0u; ++i) {
    const int b = __ffs(bits) - 1;
    bits &= bits - 1u;
    sm.read_entries[warp][at + i] = static_cast<uint16_t>(base + __popc(uni & ((1u << b) - 1u)));
  }
}

//: A WARP'S READ SIDE, SECOND HALF: the chosen form's entries -- per digit, each row's tokens from its offset; merged,
//: the union's from 0 -- and the form and entry total. A dense side writes no entries.
template <class C>
__device__ __forceinline__ void read_entries(ListTenants<C>& sm, const SideLanes<C, 1>& sl,
                                             int form, bool sparse, int warp, int lane) {
  using S = SideLanes<C, 1>;
  constexpr int kWords = S::kWords;
  if (lane == 0) {
    sm.read_form[warp] = static_cast<uint8_t>(sparse ? form : kStepMerged);
    sm.read_total[warp] = static_cast<uint16_t>(!sparse               ? kWindow
                                                : form == kStepMerged ? sl.union_count
                                                                      : sl.listed);
  }
  if (!sparse) return;
  const bool merged = form == kStepMerged;
  if (merged && sl.row != 0) return;
  int at = merged ? sl.union_before : sl.offset + sl.before;
#pragma unroll
  for (int i = 0; i < kWords; ++i) {
    const uint32_t bits = merged ? sl.u[i] : sl.w[i];
    word_entries(sm, warp, S::kWords * sl.group + i, bits, at);
    at += __popc(bits);
  }
}

//: THE LISTS: each warp its own rows (counts, offsets, its unions, its read form), the CTA's unions, sector unions and
//: ranks from the warps' unions, then every warp's share of the slot-to-token table and its read entries.
//: -- see docs/internals/owner/owner_lists.md#lists
template <class C>
__device__ void window_lists(ListTenants<C>& sm, const OwnerParams& p, int buf, int tid) {
  static_assert(C::kFoldLists <= 8 && C::kReadDigits <= 8 && kWarps == 8,
                "a side's rows fill a warp's lanes evenly");
  constexpr int kSectors = C::RD / 32, kSectorWarps = kWarps / kSectors;
  const int warp = tid >> 5, lane = tid & 31;
  const bool sparse_w = p.sparse_write != 0, sparse_r = p.sparse_read != 0;
  const SideLanes<C, 0> fold(sm, buf, warp, lane);
  const SideLanes<C, 1> read(sm, buf, warp, lane);
  const int read_form = side_form(read, p.step);
  side_counts(sm, fold, sparse_w, warp, lane);
  side_counts(sm, read, sparse_r, warp, lane);
  __syncthreads();

  if (warp == 0) {
    const int side = lane >> 4, k = lane & 15;
    uint32_t u = 0u;
    uint32_t sector[kSectors] = {};
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      const uint32_t x = sm.partial[side][w][k];
      u |= x;
      sector[w / kSectorWarps] |= x;
    }
    const int c = __popc(u);
    sm.uni[side][k] = u;
    sm.base[side][k] = static_cast<uint16_t>(half_prefix(c, lane) - c);
    if (side == 0) {
#pragma unroll
      for (int s = 0; s < kSectors; ++s) sm.sector_uni[s][k] = sector[s];
    }
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int index = 4 * warp + i, side = index >> 4, k = index & 15;
    const uint32_t word = sm.uni[side][k];
    if ((word >> lane) & 1u)
      sm.token_of[side][sm.base[side][k] + __popc(word & ((1u << lane) - 1u))] =
          static_cast<uint16_t>(32 * k + lane);
  }
  read_entries(sm, read, read_form, sparse_r, warp, lane);
  __syncthreads();
}

//: A WINDOW'S WRITE SIDE AS THE FILL AND THE FOLD READ IT: its slot-to-token table, its sector unions (`[RD / 32][16]`)
//: and the warps' union words (`[8][16]`).
struct WriteView {
  const uint16_t* token_of;
  const uint32_t* sector_uni;
  const uint32_t* partial;
};

template <class C>
__device__ __forceinline__ WriteView write_view(const ListTenants<C>& sm) {
  return WriteView{sm.token_of[0], &sm.sector_uni[0][0], &sm.partial[0][0][0]};
}

//: THE FOLD'S WRITE LISTS, a window ahead with no CTA barrier: the box's write rows in three buffers staged two windows
//: ahead, and by window parity the warps' union words, the CTA's union, ranks, sector unions, live count and
//: slot-to-token table; each stage an mbarrier every thread arrives on. -- see docs/internals/owner/owner_lists.md#write-lists
template <class C>
struct WriteLists {
  static constexpr int kSectors = C::RD / 32;
  uint32_t words[3][C::RD][kWindowWords];
  uint32_t partial[2][kWarps][kWindowWords];
  uint32_t uni[2][kWindowWords];
  uint32_t sector_uni[2][kSectors][kWindowWords];
  uint16_t base[2][kWindowWords];
  uint16_t token_of[2][kWindow];
  int live[2];
  uint64_t words_landed[3], partials[2], table[2];

  __device__ __forceinline__ WriteView view(int win) const {
    const int b = win & 1;
    return WriteView{token_of[b], &sector_uni[b][0][0], &partial[b][0][0]};
  }
};

//: The barriers' phase parity for window `win` on a stage of `n` buffers
__device__ __forceinline__ uint32_t stage_parity(int win, int n) {
  return static_cast<uint32_t>(win / n) & 1u;
}

//: A DENSE WRITE SIDE'S LISTS, once: every token live, the table the identity, in both buffers.
template <class C>
__device__ __forceinline__ void dense_write_lists(WriteLists<C>& wl, int tid) {
#pragma unroll 1
  for (int i = tid; i < 2 * kWindow; i += kThreads)
    (&wl.token_of[0][0])[i] = static_cast<uint16_t>(i % kWindow);
  if (tid < 2) wl.live[tid] = kWindow;
}

//: THE WRITE ROWS OF WINDOW `win` into buffer `win % 3`, a 16-byte chunk a thread, every thread arriving on its landing.
template <class C>
__device__ __forceinline__ void stage_write_words(WriteLists<C>& wl, const OwnerParams& p, int bh,
                                                  int r0, int win, int tid) {
  static_assert(C::RD * 4 == kThreads, "a thread a chunk");
  const int row = tid >> 2, part = tid & 3, buf = win % 3;
  const uint32_t* src =
      p.words + (static_cast<int64_t>(bh) * facts::kSides + facts::kWrite) * p.side_words
      + (p.row_base[p.level_r] + r0 + row) * p.row_words + win * kWindowWords + part * 4;
  ops::stage_tile(ops::smem_addr(&wl.words[buf][row][part * 4]), src);
  ops::mbar_arrive_when_landed(ops::smem_addr(&wl.words_landed[buf]));
}

//: A WARP'S UNION WORDS FOR WINDOW `win`: its eight write rows ORed, a lane a word of four rows and a shuffle, once the
//: rows landed; every lane arrives on `partials`.
template <class C>
__device__ __forceinline__ void write_partial(WriteLists<C>& wl, int win, int warp, int lane) {
  static_assert(C::kFoldLists == 8, "a warp's eight write rows");
  const int buf = win % 3, k = lane & 15, half = lane >> 4;
  ops::mbar_wait(ops::smem_addr(&wl.words_landed[buf]), stage_parity(win, 3));
  uint32_t u = 0u;
#pragma unroll
  for (int i = 0; i < 4; ++i) u |= wl.words[buf][warp * 8 + 4 * half + i][k];
  u |= __shfl_xor_sync(0xffffffffu, u, 16);
  if (half == 0) wl.partial[win & 1][warp][k] = u;
  ops::mbar_arrive(ops::smem_addr(&wl.partials[win & 1]));
}

//: WINDOW `win`'s UNION AND TABLE once every warp's union words are in: each warp forms the union, its ranks and the
//: sector unions in registers (a lane a word), warp 0 stores them and the live count, and each warp stores its two words'
//: share of the slot-to-token table; every lane arrives on `table`.
template <class C>
__device__ __forceinline__ void write_table(WriteLists<C>& wl, int win, int warp, int lane) {
  constexpr int kSectors = WriteLists<C>::kSectors, kSectorWarps = kWarps / kSectors;
  const int b = win & 1, k = lane & 15;
  ops::mbar_wait(ops::smem_addr(&wl.partials[b]), stage_parity(win, 2));
  uint32_t u = 0u, sector[kSectors] = {};
#pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    const uint32_t x = wl.partial[b][w][k];
    u |= x;
    sector[w / kSectorWarps] |= x;
  }
  const int c = __popc(u), incl = half_prefix(c, lane), base = incl - c;
  if (warp == 0 && lane < kWindowWords) {
    wl.uni[b][k] = u;
    wl.base[b][k] = static_cast<uint16_t>(base);
#pragma unroll
    for (int s = 0; s < kSectors; ++s) wl.sector_uni[b][s][k] = sector[s];
    if (lane == kWindowWords - 1) wl.live[b] = incl;
  }
  const uint32_t below = (1u << lane) - 1u;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int word = 2 * warp + i;
    const uint32_t uw = __shfl_sync(0xffffffffu, u, word);
    const int bw = __shfl_sync(0xffffffffu, base, word);
    if ((uw >> lane) & 1u)
      wl.token_of[b][bw + __popc(uw & below)] = static_cast<uint16_t>(32 * word + lane);
  }
  ops::mbar_arrive(ops::smem_addr(&wl.table[b]));
}

}  // namespace rola::owner::detail
