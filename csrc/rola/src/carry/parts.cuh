// csrc/rola/src/carry/parts.cuh -- THE CARRY'S COMPONENTS at their real signatures, tenants and copies: a window's
// lists and facts in, a chunk's fills (its rows, its entries, its zeroed panel), the hand-off (the digit maxima, a
// batch's stores), a chunk's readout and its drain. see docs/internals/carry/parts.md
#pragma once

#include <cuda_fp16.h>

#include <climits>
#include <cstdint>
#include <type_traits>

#include "carry/plan.cuh"
#include "common/ops.cuh"
#include "common/static_for.cuh"

namespace rola::carry {

namespace ops = ::denseref::ops;

//: THE UNITS' CONSTANTS: c's top 2^4, a state held under 2^100 of its unit, the lowest unit, a list never live, a
//: window's largest add in units, the facts pass's dead exponent. -- see docs/internals/carry/fold.md#units
constexpr int kCoefTop = 4;
constexpr int kUnitHeadroom = 100;
constexpr int kUnitMin = kCoefTop - 127;
constexpr int kUnitNone = -1000;
constexpr float kWindowAdd = kWindow * 16.0f;
constexpr int kExpDead = -126;

//: A LIST'S UNIT ACROSS WINDOWS, in shared memory: its exponent and its state's bound in units.
struct UnitBook {
  int unit;
  float bound;
};

__device__ __forceinline__ float pow2(int e) {
  return __uint_as_float(static_cast<uint32_t>(e + 127) << 23);
}

//: A POWER OF TWO over the whole float range: normal from 2^-126, a subnormal's bit down to 2^-149, zero below.
__device__ __forceinline__ float pow2_wide(int e) {
  return e >= -126 ? pow2(min(e, 127)) : e >= -149 ? __uint_as_float(1u << (e + 149)) : 0.0f;
}

//: the smallest `e` with `x <= 2^e`, `kUnitNone` for zero
__device__ __forceinline__ int ceil_exp(float x) {
  const uint32_t bits = __float_as_uint(x);
  return bits == 0u ? kUnitNone : static_cast<int>(bits >> 23) - 127 + ((bits & 0x7fffffu) != 0u);
}

//: A WARP'S STATE, the fold's accumulators: per list, its digits' m-tiles by the channels' four n-tiles, fp32 in the
//: list's unit; and the den's, the v+1 channel absolute, four words a lane (`den_place`). -- see
//: docs/internals/carry/fold.md#state
template <class P>
struct State {
  float acc[P::kFoldLists][P::kFoldMTiles][4][4];
  float den[4];

  //: the state's `k`-th register, `k` compile-time
  template <int K>
  __device__ __forceinline__ float at() const {
    return acc[K / (P::kFoldMTiles * 16)][K / 16 % P::kFoldMTiles][K / 4 % 4][K % 4];
  }
};

//: THE CTA'S PLACE: its batch-head and its box's first channel (a level's first digit is `origin_of`).
//: -- see docs/internals/carry/parts.md#place
struct Place {
  int bh, channel0;
};

__device__ __forceinline__ int origin_of(const CarryParams& p, int level) {
  return (int)(blockIdx.x / (unsigned)p.stride[level]) % (p.widths[level] / p.extents[level])
         * p.extents[level];
}

__device__ __forceinline__ Place place_of(const CarryParams& p) {
  const int bh = (int)blockIdx.y;
  return Place{bh, (int)(blockIdx.x % (unsigned)(p.dv / p.channels)) * p.channels};
}

//: A SIDE'S OPERAND SLICES for this CTA: a level's slice at the box's digits, `bytes` a token's (a power of two), at
//: `st` a token.
struct Slices {
  const uint8_t* at[kMaxLevels];
  int st[kMaxLevels], sh[kMaxLevels];
  int bytes[kMaxLevels];
};

//: A SIDE'S WINDOW STATICS, the same every window: its union's group, its box's first list digit and its lists, and
//: whether it lists.
struct Statics {
  int group, digit0, lists, listed;
};

//: THE CTA'S ADDRESSES, worked out once before the windows, so the window loop divides by nothing: each side's statics
//: and operand slices, the values at the box's channels. -- see docs/internals/carry/parts.md#place
struct Addr {
  Statics side[2];
  Slices slices[2];
  const uint8_t* values;
  int channel0;
};

static_assert(sizeof(Addr) <= Tenants<512>::kAddrBytes, "the addresses fit their tenant");

__device__ __forceinline__ void addr_in(Addr& out, const CarryParams& p, const Place& c) {
  //: the read side's rows at the write head's first query head, `h g`
  const int b = c.bh / p.H, h = c.bh - b * p.H;
#pragma unroll 1
  for (int k = 0; k < 2; ++k) {
    const Side& s = k == 0 ? p.read : p.write;
    Statics& st = out.side[k];
    st.listed = s.list >= 0;
    st.digit0 = st.listed ? origin_of(p, s.list) : 0;
    st.group = st.digit0 / 32;
    st.lists = st.listed ? p.extents[s.list] : kBoxLists;
#pragma unroll 1
    for (int i = 0; i < s.operands; ++i) {
      const int level = s.operand[i];
      const Plane& pl = s.rows[i];
      out.slices[k].at[i] = pl.base + b * pl.sb + h * (k == 0 ? p.tables.heads : 1) * pl.sh
                            + (int64_t)origin_of(p, level) * s.elem[i];
      out.slices[k].st[i] = static_cast<int>(pl.st);
      out.slices[k].sh[i] = static_cast<int>(pl.sh);
      out.slices[k].bytes[i] = p.extents[level] * s.elem[i];
    }
  }
  out.values = p.values.base + b * p.values.sb + h * p.values.sh + c.channel0 * 2;
  out.channel0 = c.channel0;
}

//: a window's parity, opaque (`ops::opaque`), so an offset by it is formed at each use rather than held across a
//: window's steps
__device__ __forceinline__ int parity_of(int w) {
  int parity;
  asm volatile("and.b32 %0, %1, 1;" : "=r"(parity) : "r"(w));
  return parity;
}

//: A SIDE'S WINDOW: its union (the list level's 32-digit group holding the box's digits) and its box lists' first
//: digit, or every window token a slot where the side has no list level.
struct Window {
  int tokens, t0, group, digit0, lists;
  bool listed;
};

template <class P>
__device__ __forceinline__ Window window_of(const CarryParams& p, const uint8_t* smem, int side,
                                            int w) {
  //: read at each use (volatile): held live across the window loop they would cost the state registers
  const volatile Statics& s = reinterpret_cast<const volatile Addr*>(smem + P::kAddr)->side[side];
  Window out;
  out.t0 = ops::opaque(w) * kWindow;
  out.tokens = min(kWindow, p.L - out.t0);
  out.listed = s.listed != 0;
  out.digit0 = s.digit0;
  out.group = s.group;
  out.lists = s.lists;
  return out;
}

__device__ __forceinline__ int64_t list_row(const SideLists& l, const Place& c, int windows, int w,
                                            int digit) {
  return ((int64_t)c.bh * windows + w) * l.width + digit;
}

//: THE SIDES, as the tenants index them: the read side 0, the write side 1.
enum SideIndex : int { kRead = 0, kWrite = 1 };

//: A SIDE'S TENANTS for window `w` (two windows' a side, by parity): its union tokens and its group's boundaries in
//: prefix form -- the read side's in the call's read tables, a row of them its query heads' -- and its columns a row.
//: -- see docs/internals/carry/plan.md#tenants
__device__ __forceinline__ int cols_of(const CarryParams& p, int side) {
  return kListAt * (side == kRead ? p.tables.heads : 1);
}

template <class P>
__device__ __forceinline__ const uint16_t* union_of(const CarryParams& p, const uint8_t* smem,
                                                    int side, int w) {
  return side == kRead
             ? reinterpret_cast<const uint16_t*>(smem + p.tables.union_at)
                   + parity_of(w) * p.tables.heads * kWindow
             : reinterpret_cast<const uint16_t*>(smem + P::kUnionW) + parity_of(w) * kWindow;
}

template <class P>
__device__ __forceinline__ const uint16_t* pre_of(const CarryParams& p, const uint8_t* smem,
                                                  int side, int w) {
  return reinterpret_cast<const uint16_t*>(smem + (side == kRead ? p.tables.pre_at : P::kPreW))
         + parity_of(w) * kPreRows * cols_of(p, side);
}

//: A CHUNK'S COLUMNS in the boundaries, union slots `first .. first + slots - 1` (`first` a multiple of 64, the end a
//: multiple of 64 or the union's `total`): the one before its first slot (-1 for none) and the one at its end, of a
//: row's `cols`.
__device__ __forceinline__ int2 chunk_cols(int first, int slots, int total, int cols) {
  return make_int2(first / kListBound - 1,
                   first + slots >= total ? cols - 1 : (first + slots) / kListBound - 1);
}

//: a side's slots in window `w`: its union's count, or every token where the side has no list level
template <class P>
__device__ __forceinline__ int slots_of(const uint8_t* smem, const Window& win, int side, int w) {
  return win.listed ? reinterpret_cast<const int*>(smem + P::kCount)[2 * side + (w & 1)]
                    : win.tokens;
}

//: LISTS IN, a window ahead: a side's union tokens and count and its group's boundaries in prefix form, a row its query
//: heads' columns. -- see docs/internals/carry/parts.md#lists-in
template <class P>
__device__ __forceinline__ void lists_in(const CarryParams& p, const SideLists& l,
                                         const Window& win, const Place& c, int w, uint8_t* smem,
                                         int side) {
  if (!win.listed) return;
  const int cols = cols_of(p, side), slots = kWindow * l.heads;
  const int64_t group = ((int64_t)c.bh * p.windows + w) * l.groups + win.group;
  const int union_copies = slots * 2 / ops::kStageBytes, row_copies = cols * 2 / ops::kStageBytes;
  const ops::SmemAddr unions = ops::smem_addr(union_of<P>(p, smem, side, w)),
                      pres = ops::smem_addr(pre_of<P>(p, smem, side, w));
  //: the thread opaque, so the copies' sources are formed at each staging, never held across the windows
#pragma unroll 1
  for (int k = ops::opaque(static_cast<int>(threadIdx.x)); k < union_copies + kPreRows * row_copies;
       k += kThreads) {
    if (k < union_copies) {
      ops::stage_tile(unions + 16 * k, l.union_tok + group * slots + 8 * k);
    } else {
      const int r = k - union_copies;
      ops::stage_tile(pres + 16 * r, l.pre + group * kPreRows * cols + 8 * r);
    }
  }
  if (threadIdx.x == kThreads - 1)
    ops::stage_run<4>(ops::smem_addr(smem + P::kCount) + 4 * (2 * side + (w & 1)),
                      l.union_count + group);
}

//: FACTS IN, a window ahead: the box's write lists' exponent pairs and each of the read side's query heads' tokens' scale
//: bytes, the den pass's (query head `j` of write head `bh` the call's `bh g + j`): the f16 sum's scale.
template <class P>
__device__ __forceinline__ void facts_in(const CarryParams& p, const Window& ww, const Place& c,
                                         int w, uint8_t* smem) {
  constexpr int kUnitCopies = kWindow / ops::kStageBytes,
                kExpCopies = kBoxLists * 4 / ops::kStageBytes;
  const int heads = p.tables.heads;
  const ops::SmemAddr scale = ops::smem_addr(smem + p.tables.scale_at) + (w & 1) * heads * kWindow;
#pragma unroll 1
  for (int k = ops::opaque(static_cast<int>(threadIdx.x)); k < heads * kUnitCopies + kExpCopies;
       k += kThreads) {
    if (k < heads * kUnitCopies) {
      const int j = k / kUnitCopies;
      ops::stage_tile(scale + 16 * k, p.scale
                                          + ((int64_t)(c.bh * heads + j) * p.windows + w) * kWindow
                                          + 16 * (k - j * kUnitCopies));
    } else if (ww.listed) {
      const int e = k - heads * kUnitCopies;
      ops::stage_tile(
          ops::smem_addr(smem + P::kWriteExp) + (w & 1) * kBoxLists * 4 + 16 * e,
          p.write_exp + list_row(p.write_lists, c, p.windows, w, ww.digit0) * 2 + 8 * e);
    } else if (k == heads * kUnitCopies) {
      //: a write side with no list level: the window's shifts of (m, F), in the exponents' place
      ops::stage_run<4>(ops::smem_addr(smem + P::kWriteExp) + (w & 1) * kBoxLists * 4,
                        p.factor_shift + (static_cast<int64_t>(c.bh) * p.windows + w) * 2);
    }
  }
}

//: A POOL ROW'S SWIZZLE at a compiled row (`pool_chunk`'s): the row's own start alternates halves of the bank line.
//: -- see docs/internals/carry/parts.md#swizzle
template <int Row>
__device__ __forceinline__ constexpr uint32_t chunk_at(uint32_t s, uint32_t k) {
  static_assert(Row % 64 == 0 && (Row == 128 || (Row / 64) % 2 == 1),
                "a row of 128 bytes or an odd number of 64");
  return pool_chunk(Row, s, k);
}

//: LOGICAL CHUNK `k`'s ADDRESS in a row whose chunk 0 sits at `row0` (its swizzle folded in): an XOR inside the
//: swizzle's group, the group's own offset added.
template <int Row>
__device__ __forceinline__ uint32_t chunk_addr(uint32_t row0, uint32_t k) {
  if constexpr (Row == 128) return row0 ^ (16u * k);
  return (row0 ^ (16u * (k & 3u))) + 16u * (k & ~3u);
}

//: ONE ASYNC RUN of 16, 8 or 4 bytes, the run a slice's width gives; zero-filled for a gap slot (`live` false).
__device__ __forceinline__ void stage_bytes(ops::SmemAddr dst, const void* src, int run,
                                            bool live) {
  // R9-CASES: a copy run, 16, 8 or 4 bytes (the host refuses anything under 4)
  if (run == ops::kStageBytes) {
    ops::stage_run_if<16>(dst, src, live);
  } else if (run == 8) {
    ops::stage_run_if<8>(dst, src, live);
  } else if (run == 4) {
    ops::stage_run_if<4>(dst, src, live);
  } else {
    __trap();
  }
}

//: AN ASYNC 16-BYTE COPY at `dst + At`, issued where `At` is under `room`, its first `size` bytes read.
template <uint32_t At>
__device__ __forceinline__ void copy16(ops::SmemAddr dst, const void* src, int room, int size) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.gt.s32 p, %2, %4;\n@p cp.async.cg.shared.global [%0+%4], [%1], 16, %3;\n}\n" ::
          "r"(dst),
      "l"(src), "r"(room), "r"(size), "n"(At));
}

//: A 16-BYTE FILL of word `v` at `dst + 16 At`, stored where `At` is under `room`.
template <int At>
__device__ __forceinline__ void fill16(ops::SmemAddr dst, int room, uint32_t v) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.gt.s32 p, %1, %2;\n@p st.shared.v4.u32 [%0+%3], {%4, %4, %4, %4};\n}\n" ::
          "r"(dst),
      "r"(room), "n"(At), "n"(16 * At), "r"(v)
      : "memory");
}

//: A CLEARED PANEL WORD: its two s16 halves zero, offset by 2^15 each (the drain's `pair_magics` reads a word as is).
//: -- see docs/internals/carry/parts.md#drain
constexpr uint32_t kPanelClear = 0x80008000u;

//: A ROW'S SOURCE: `base` plus two products, each one wide multiply-add (the operands unsigned, under 2^32).
__device__ __forceinline__ const uint8_t* row_at(const uint8_t* base, int a, int b, int c, int d) {
  const uint64_t x = reinterpret_cast<uint64_t>(base)
                     + static_cast<uint64_t>(static_cast<uint32_t>(a)) * static_cast<uint32_t>(b);
  return reinterpret_cast<const uint8_t*>(
      x + static_cast<uint64_t>(static_cast<uint32_t>(c)) * static_cast<uint32_t>(d));
}

//: A CHUNK'S ROWS: `slots` union slots from `first`, each its token's row (its slices, then `Values` its values),
//: swizzled, by `cp.async`: thread `4 q + r` chunk `r` of slots `q + 64 k` of each 256, a trip one block -- its tokens
//: (unclamped), every source, then the copies on the trip's room; a gap slot zero-filled. -- see
//: docs/internals/carry/parts.md#fill
template <int Factors, bool Values>
__device__ __forceinline__ void fill(const volatile Slices& sl, const uint8_t* values,
                                     int values_st, const uint16_t* tok, const Window& win,
                                     int first, int slots, ops::SmemAddr pool) {
  constexpr int kRow = kSliceBytes * Factors + (Values ? kBoxChannels * 2 : 0);
  constexpr int kQuads = kThreads / 4, kChunks = kSliceBytes / ops::kStageBytes, kAhead = 4;
  //: the thread opaque, so its quad lane and first slot are formed at each fill, never held across the windows
  const int tid = ops::opaque(static_cast<int>(threadIdx.x)), r = tid & 3, q = tid >> 2;
  const uint8_t* base[Factors];
  int st[Factors], sh[Factors], bytes[Factors], size[Factors];
  bool active[Factors], narrow = false;
  static_for<Factors>([&](auto i) {
    bytes[i] = sl.bytes[i];
    st[i] = sl.st[i];
    sh[i] = Values ? 0 : sl.sh[i];
    base[i] = sl.at[i] + static_cast<int64_t>(win.t0) * st[i] + ops::kStageBytes * r;
    active[i] = ops::kStageBytes * r < bytes[i];
    size[i] = active[i] ? ops::kStageBytes : 0;
    narrow |= bytes[i] < ops::kStageBytes;
  });
  const uint8_t* const vbase =
      Values ? values + static_cast<int64_t>(win.t0) * values_st + ops::kStageBytes * r : nullptr;

  const bool listed = win.listed;
  const int16_t* const toks = reinterpret_cast<const int16_t*>(tok) + first + q;
  //: the thread's rows' places at slot `q`: each slice's chunk and the values' (`chunk_at`)
  const ops::SmemAddr row = pool + static_cast<uint32_t>(q * kRow);
  uint32_t at[Factors + 1];
  static_for<Factors + 1>([&](auto i) {
    at[i] =
        row
        + 16u * chunk_at<kRow>(static_cast<uint32_t>(q), static_cast<uint32_t>(kChunks * i + r));
  });

  //: a narrow slice (8 or 4 bytes) its quad's first lane's whole run; a trip of 256 slots, the first straight-line
  const auto rows = [&](auto kNarrow) {
    const auto trip_at = [&](int s0) {
      //: the trip's four tokens, loaded on either side and selected (an unlisted side's slot is its token)
      int u[kAhead];
      static_for<kAhead>([&](auto k) {
        const int v = toks[s0 + kQuads * k];
        u[k] = listed ? v : first + q + s0 + kQuads * k;
      });
      const int room = (slots - q - s0) * kRow;
      const uint32_t trip = static_cast<uint32_t>(s0 * kRow);

      //: every source first: a slot's query head `j` (the read side's) and token `t`, its gap mask
      const uint8_t* src[kAhead][Factors + 1];
      int gap[kAhead];
      static_for<kAhead>([&](auto k) {
        const int t = u[k] & (kWindow - 1),
                  j = Values ? 0 : (u[k] >> window_bits(kWindow)) & (window_heads(kWindow) - 1);
        gap[k] = u[k] >> 31;
        static_for<Factors>([&](auto i) { src[k][i] = row_at(base[i], t, st[i], j, sh[i]); });
        if constexpr (Values) src[k][Factors] = row_at(vbase, t, values_st, 0, 0);
      });

      static_for<kAhead>([&](auto k) {
        constexpr uint32_t kAt = static_cast<uint32_t>(decltype(k)::value * kQuads * kRow);
        static_for<Factors>([&](auto i) {
          if constexpr (decltype(kNarrow)::value) {
            const ops::SmemAddr dst = at[i] + trip + kAt;
            const bool issue = static_cast<int>(kAt) < room, live = gap[k] == 0;
            if (issue && !active[i] && r == 0) stage_bytes(dst, src[k][i], bytes[i], live);
            if (issue && active[i]) ops::stage_run_if<16>(dst, src[k][i], live);
          } else {
            copy16<kAt>(at[i] + trip, src[k][i], room, size[i] & ~gap[k]);
          }
        });
        if constexpr (Values)
          copy16<kAt>(at[Factors] + trip, src[k][Factors], room, ops::kStageBytes & ~gap[k]);
      });
    };

    trip_at(0);
#pragma unroll 1
    for (int s0 = kAhead * kQuads; s0 < slots - q; s0 += kAhead * kQuads) trip_at(s0);
  };
  if (narrow) {
    rows(std::true_type{});
  } else {
    rows(std::false_type{});
  }
}

//: A CHUNK'S RUNS, lane `l` list `l`'s: its first entry of the chunk, its count, and its place in a buffer's entries
//: (the lists' runs in list order), from its two prefix rows modulo 2^16. -- see docs/internals/carry/parts.md#entries
struct Run {
  int at, n, to;
};

template <class P>
__device__ __forceinline__ Run chunk_runs(const CarryParams& p, const uint8_t* smem,
                                          const Window& win, int side, int w, int first, int slots,
                                          int total, int lane) {
  if (!win.listed || lane >= win.lists) return Run{0, 0, 0};
  const int cols = cols_of(p, side);
  const int2 col = chunk_cols(first, slots, total, cols);
  //: the lane's two rows (lists `l`, `l + 1`): the chunk's end in both, its start where it is not the first
  const uint16_t* const row =
      pre_of<P>(p, smem, side, w) + ((win.digit0 & (kBoxLists - 1)) + lane) * cols;
  const int a1 = row[col.y], b1 = row[cols + col.y];
  const bool start = col.x >= 0;
  const int a0 = start ? row[max(col.x, 0)] : 0, b0 = start ? row[cols + max(col.x, 0)] : 0;
  return Run{(b0 - a0) & 0xffff, ((b1 - a1) - (b0 - a0)) & 0xffff, (a1 - a0) & 0xffff};
}

//: A CHUNK'S ENTRIES, a warp's share: its own lists' runs (`warp + 8 j`) to their places at `dst`, eight lanes a run,
//: two copies a trip, the trips the warp's longest run's. -- see docs/internals/carry/parts.md#entries
template <class P>
__device__ __forceinline__ void entries_fill(const CarryParams& p, const SideLists& l,
                                             const Window& win, const Place& c, int w,
                                             const Run& run, ops::SmemAddr dst, int warp,
                                             int lane) {
  static_assert(kBoxLists == 4 * kWarps, "a warp's four lists, eight lanes each");
  constexpr int kLanes = 8;
  //: a list's row `kListCap` entries a query head on (opaque: formed at each fill, never held across windows)
  const int cap = kListCap * ops::opaque(l.heads);
  const int list = warp + kWarps * (lane / kLanes), i = lane % kLanes;
  const int at = __shfl_sync(0xffffffffu, run.at, list), n = __shfl_sync(0xffffffffu, run.n, list),
            to = __shfl_sync(0xffffffffu, run.to, list);
  const int trips =
      (static_cast<int>(__reduce_max_sync(0xffffffffu, static_cast<unsigned>(n))) + kLanes - 1)
      / kLanes;
  const uint2* src = l.entries + list_row(l, c, p.windows, ops::opaque(w), win.digit0) * cap
                     + (list * cap + at + i);
  ops::SmemAddr d = dst + 8u * static_cast<uint32_t>(to + i);
  const int left = n - i;
  //: the first two trips straight-line, any more in a loop
  const auto two = [&](int t) {
    ops::stage_run_lanes<8>(d, src, kLanes * t < left);
    ops::stage_run_lanes<8>(d + 8u * kLanes, src + kLanes, kLanes * (t + 1) < left);
    src += 2 * kLanes;
    d += 16u * kLanes;
  };
  two(0);
#pragma unroll 1
  for (int t = 2; t < trips; t += 2) two(t);
}

//: A CHUNK'S BUFFER, laid out at its own slots: a read buffer's rows, its dummy panel rows, its panel (a slot pair a
//: 128-byte line), its entries; a fold buffer's rows, its entries. -- see docs/internals/carry/schedule.md#buffers
template <class P>
__host__ __device__ constexpr int read_pads(int slots) {
  return align_of(slots * P::kReadRow, 128);
}

template <class P>
__host__ __device__ constexpr int read_ents(int slots) {
  return read_pads<P>(slots) + kPadBytes + align_of(slots * kPanelRow, 128);
}

template <class P>
__host__ __device__ constexpr int fold_ents(int slots) {
  return align_of(slots * P::kWriteRow, 128);
}

//: A READ CHUNK'S FILL, the copies issued: its rows at `buf`, its dummy panel rows and panel cleared, its entries (a
//: warp its own digits' runs, `run` the chunk's). -- see docs/internals/carry/parts.md#fill
template <class P>
__device__ __forceinline__ void read_fill(const CarryParams& p, const uint8_t* smem,
                                          const Window& rw, const Place& c, int w, const Run& run,
                                          int first, int slots, ops::SmemAddr buf, int warp,
                                          int lane) {
  const volatile Addr& addr = *reinterpret_cast<const volatile Addr*>(smem + P::kAddr);
  fill<P::kReadFactors, false>(addr.slices[kRead], nullptr, 0, union_of<P>(p, smem, kRead, w), rw,
                               first, slots, buf);
  //: the dummy panel rows and the panel cleared by stores (`kPanelClear`), four a thread a trip, the first straight-line
  constexpr int kAhead = 4;
  const int tid = ops::opaque(static_cast<int>(threadIdx.x));
  const auto clear = [&](int from, int to, uint32_t v) {
    const int quanta = (to - from) / 16 - tid;
    const ops::SmemAddr zero = buf + static_cast<uint32_t>(from) + 16u * static_cast<uint32_t>(tid);
    const auto four = [&](int k) {
      const ops::SmemAddr at = zero + 16u * static_cast<uint32_t>(k);
      static_for<kAhead>([&](auto j) { fill16<decltype(j)::value * kThreads>(at, quanta - k, v); });
    };
    four(0);
#pragma unroll 1
    for (int k = kAhead * kThreads; k < quanta; k += kAhead * kThreads) four(k);
  };
  clear(read_pads<P>(slots), read_ents<P>(slots), kPanelClear);
  if (rw.listed)
    entries_fill<P>(p, p.read_lists, rw, c, w, run,
                    buf + static_cast<uint32_t>(read_ents<P>(slots)), warp, lane);
}

//: A WARP'S LISTS' UNITS, list `j` the box's list `warp + 8 j`.
template <class P>
__device__ __forceinline__ void list_units(int (&u)[P::kFoldLists], const uint8_t* smem, int warp) {
  //: read at each use (volatile): merged across a window's stores they would hold registers across it
  const volatile UnitBook* const book = reinterpret_cast<const volatile UnitBook*>(smem + P::kBook);
  static_for<P::kFoldLists>([&](auto J) { u[J] = book[warp + kWarps * J].unit; });
}

//: THE BOX'S TOP UNIT, its 32 lists' largest, one reduction a warp (the book read at its use).
template <class P>
__device__ __forceinline__ int box_top(const uint8_t* smem, int lane) {
  static_assert(kBoxLists == 32, "a lane a list");
  const volatile UnitBook* const book = reinterpret_cast<const volatile UnitBook*>(smem + P::kBook);
  return __reduce_max_sync(0xffffffffu, book[lane].unit);
}

//: THE KEYS' SPAN: a list more than this many units under the box's top keys as if this far (a live unit lies in
//: `kUnitMin .. 128`, a span under it; a never-live list's state is zero).
constexpr int kKeySpan = 255;

//: A LEAF'S EXPONENT BYTE from its key: the smallest `e` with its largest |value| at most `2^(e + top)`, held to
//: [-127, 127]; -128 a zero leaf. -- see docs/internals/carry/parts.md#leaves
__device__ __forceinline__ int leaf_byte(int key) {
  return key == INT_MIN ? -128 : min(max(((key + 0x7fffff) >> 23) - 127, -127), 127);
}

//: A LEAF'S EXPONENT BYTE's place at `kLeafExp`: its readout group's 32 contiguous (a digit's, a same-split arm's list's)
template <class P>
__device__ __forceinline__ constexpr int leaf_at(int digit, int list) {
  return P::kPublishRegs ? list * kBoxDigits + digit : digit * kBoxLists + list;
}

//: THE DIGIT KEYS, the hand-off's first step: a list's largest |value| a digit row at its unit as an integer key --
//: its fp32 bits plus (its unit less the box's top) x 2^23, a zero `INT_MIN` -- the bits joined across the row's four
//: lanes first (a list's lanes share its unit), lane `t` keying its list `t`'s leaf (its exponent byte at `kLeafExp`,
//: digit-major); a digit's key their max, one shared atomic max a lane, at window `w`'s parity in lane order (digit
//: `16 M + 8 H + g` at `4 g + 2 M + H`); returns the box's top. -- see docs/internals/carry/parts.md#maxima
template <class P>
__device__ __forceinline__ int post_keys(const State<P>& st, uint8_t* smem, int w, int warp,
                                         int lane) {
  static_assert(P::kFoldMTiles * 2 == 4, "a lane's four digit rows, one a lane of its quad");
  static_assert(P::kFoldLists == 4, "a quad's lane `t` keys its list `t`'s leaves");
  const int t = lane & 3, g = lane >> 2;
  const int top = box_top<P>(smem, lane);
  const volatile UnitBook* const book = reinterpret_cast<const volatile UnitBook*>(smem + P::kBook);
  const int shift = max(book[warp + kWarps * t].unit - top, -kKeySpan) * (1 << 23);
  int8_t* const leaves = reinterpret_cast<int8_t*>(smem + P::kLeafExp);
  int key[2][2], leaf[2][2];
  static_for<2>([&](auto M) {
    static_for<2>([&](auto H) {
      int bits[P::kFoldLists];
      static_for<P::kFoldLists>([&](auto J) {
        float x = fmaxf(fabsf(st.acc[J][M][0][2 * H]), fabsf(st.acc[J][M][0][2 * H + 1]));
        static_for<3>([&](auto N) {
          x = fmaxf(
              x, fmaxf(fabsf(st.acc[J][M][N + 1][2 * H]), fabsf(st.acc[J][M][N + 1][2 * H + 1])));
        });
        bits[J] = __float_as_int(x);
      });
      //: the quad's reduce-scatter of the bits: lane `t` ends with list `t`'s (lists `2 b + 0, 1` kept across `xor 2`,
      //: then one of them across `xor 1`), keyed; the digit's key their max across the quad
      const bool hi = t & 2, odd = t & 1;
      int keep0 = hi ? bits[2] : bits[0], keep1 = hi ? bits[3] : bits[1];
      keep0 = max(keep0, __shfl_xor_sync(0xffffffffu, hi ? bits[0] : bits[2], 2));
      keep1 = max(keep1, __shfl_xor_sync(0xffffffffu, hi ? bits[1] : bits[3], 2));
      const int mine_bits =
          max(odd ? keep1 : keep0, __shfl_xor_sync(0xffffffffu, odd ? keep0 : keep1, 1));
      const int mine_leaf = mine_bits == 0 ? INT_MIN : mine_bits + shift;
      int k = max(mine_leaf, __shfl_xor_sync(0xffffffffu, mine_leaf, 1));
      key[M][H] = max(k, __shfl_xor_sync(0xffffffffu, k, 2));
      leaf[M][H] = mine_leaf;
      leaves[leaf_at<P>(16 * M + 8 * H + g, warp + kWarps * t)] =
          static_cast<int8_t>(leaf_byte(mine_leaf));
    });
  });

  int* const keys = reinterpret_cast<int*>(smem + P::kKeys) + (w & 1) * kBoxDigits;
  if constexpr (P::kPublishRegs) {
    //: a same-split arm's groups are the warp's lists: list `t`'s key its leaves' max, stored by its owner (no atomic)
    int mine = max(max(leaf[0][0], leaf[0][1]), max(leaf[1][0], leaf[1][1]));
    static_for<3>([&](auto S) { mine = max(mine, __shfl_xor_sync(0xffffffffu, mine, 4 << S)); });
    keys[warp + kWarps * t] = mine;
  } else {
    const int mine = t & 2 ? (t & 1 ? key[1][1] : key[1][0]) : (t & 1 ? key[0][1] : key[0][0]);
    atomicMax(keys + lane, mine);
  }
  return top;
}

//: A DIGIT'S SWIZZLE of its plane's units: its low bit moved to bit 2, so a quarter-warp's stores (two digits, four
//: chunks) meet eight bank groups.
__host__ __device__ __forceinline__ constexpr uint32_t digit_swizzle(int d) {
  return static_cast<uint32_t>(((d & 1) << 2) | ((d >> 1) & 3));
}

//: A 16-BYTE UNIT of digit `d`'s image plane for list `l`, chunk `k`: channels `8 n + 2 k + b` at position `2 n + b`,
//: a writer lane's (`k` its lane in the quad); the fold's 16-byte stores and the readout's `ldmatrix.trans` both
//: conflict-free. -- see docs/internals/carry/parts.md#image
__host__ __device__ __forceinline__ constexpr uint32_t unit_of(int l, int k, int d) {
  return static_cast<uint32_t>(((4 * l + (k ^ ((l >> 1) & 3))) ^ digit_swizzle(d))
                               & (kPlaneBytes / 16 - 1));
}

//: THE IMAGE'S TOP: a digit's largest value placed under 2^10, so a 32-list sum fits an fp16 accumulator.
//: -- see docs/internals/carry/parts.md#maxima
constexpr int kImageTop = 10;

//: THE DIGITS' EXPONENTS from the keys at window `w`'s parity: lane `i`'s digit's (lane order), `(key + 0x7fffff) >>
//: 23` the key's exponent field rounded up past a nonzero mantissa, plus the box's top less 127, at least `kImageTop -
//: 127` -- the smallest `e` with the digit's largest |state| at most `2^e` -- recorded at `kExps` for the stores and
//: the readers, the box's top and the largest exponent at `kTops` (every warp the same values); the other parity's
//: keys reset for the next window. -- see docs/internals/carry/parts.md#maxima
template <class P>
__device__ __forceinline__ void post_exps(uint8_t* smem, int w, int top, int lane) {
  static_assert(kBoxDigits == 32, "lane `i` digit `i`, a parity's keys one bit apart");
  int* const keys = reinterpret_cast<int*>(smem + P::kKeys);
  const int at = (w & 1) * kBoxDigits + lane;
  const int key = keys[at];
  keys[at ^ kBoxDigits] = INT_MIN;
  const int e = max(((key + 0x7fffff) >> 23) + top - 127, kImageTop - 127);
  reinterpret_cast<int*>(smem + P::kExps)[lane] = e;
  const int emax = __reduce_max_sync(0xffffffffu, e);
  if (lane == 0) *reinterpret_cast<int2*>(smem + P::kTops) = make_int2(top, emax);
  __syncwarp();
}

//: digit `d`'s place in `kExps` (lane order: `16 M + 8 H + g` at `4 g + 2 M + H`)
__host__ __device__ __forceinline__ constexpr int exp_slot(int d) { return 4 * (d & 7) + (d >> 3); }

//: readout group `d`'s place in `kExps`: a digit's (`exp_slot`), or a same-split arm's list at its own index
template <class P>
__host__ __device__ __forceinline__ constexpr int group_slot(int d) {
  return P::kPublishRegs ? d : exp_slot(d);
}

//: THE LEAVES' SCALES, once a window after the exponents: each leaf's `2^(e_leaf - e_digit)` as f16 at `kLeafScale`
//: (digit-major, a digit's 32 lists 64 bytes), at least 2^-14 (rounded up, so still a bound), 0 for a zero leaf -- a
//: thread four leaves. -- see docs/internals/carry/parts.md#leaves
template <class P>
__device__ __forceinline__ void leaf_scales(uint8_t* smem) {
  const int tid = ops::opaque(static_cast<int>(threadIdx.x)), d = tid >> 3;
  const int top = reinterpret_cast<const volatile int2*>(smem + P::kTops)->x;
  const int rel = reinterpret_cast<const int*>(smem + P::kExps)[group_slot<P>(d)] - top;
  const uint32_t b4 = reinterpret_cast<const uint32_t*>(smem + P::kLeafExp)[tid];
  uint32_t h[4];
  static_for<4>([&](auto I) {
    const int b = static_cast<int8_t>(b4 >> (8 * I));
    h[I] = b == -128 ? 0u : static_cast<uint32_t>(15 + max(min(b - rel, 0), -14)) << 10;
  });
  reinterpret_cast<uint2*>(smem + P::kLeafScale)[tid] =
      make_uint2(h[0] | h[1] << 16, h[2] | h[3] << 16);
}

//: THE DEN'S TABLES, a window's, from the den's registers (the state before the window's writes): each digit's largest
//: by an integer max (`den_tables_max`), then each digit's factor and its f16 den row (`den_tables_out`, past a barrier).
//: -- see docs/internals/carry/parts.md#den
//: A DIGIT'S DEN EXPONENT `dE` from its largest den (`kDenMax`), `kImageTop - 127` where it has none.
__device__ __forceinline__ int den_exp(float top) {
  return top > 0.0f ? ceil_exp(top) : kImageTop - 127;
}

//: A LANE'S DEN WORDS' PLACES, each of the box's 32 x 32 once across the CTA, the list owner's on either side (word `i`'s
//: list `w + 8 (2 (t & 1) + (i & 1))`, digit `g + 8 (i >> 1) + 16 (t >> 1)`).
struct DenPlace {
  int list0, digit0, list_step;

  __device__ __forceinline__ int list(int i) const { return list0 + list_step * (i & 1); }

  __device__ __forceinline__ int digit(int i) const { return digit0 + 8 * (i >> 1); }
};

template <class P>
__device__ __forceinline__ DenPlace den_place(int warp, int lane) {
  const int g = lane >> 2, t = lane & 3;
  return DenPlace{warp + 16 * (t & 1), g + 16 * (t >> 1), kWarps};
}

//: AN INTEGER MAX INTO A SHARED WORD, fire and forget (a non-negative float's bits order as integers).
__device__ __forceinline__ void red_max_u32(ops::SmemAddr at, uint32_t v) {
  asm volatile("red.shared.max.u32 [%0], %1;" ::"r"(at), "r"(v) : "memory");
}

//: a den word's readout group and its place on the group's row (a digit's over lists; a same-split arm's list's over digits)
template <class P>
__device__ __forceinline__ int den_group(const DenPlace& at, int i) {
  return P::kPublishRegs ? at.list(i) : at.digit(i);
}

template <class P>
__device__ __forceinline__ int den_k(const DenPlace& at, int i) {
  return P::kPublishRegs ? at.digit(i) : at.list(i);
}

template <class P>
__device__ __forceinline__ void den_tables_max(uint8_t* smem, const State<P>& st, int warp,
                                               int lane) {
  const DenPlace at = den_place<P>(warp, lane);
  const ops::SmemAddr tops = ops::smem_addr(smem + P::kDenMax);
  static_for<4>([&](auto I) {
    red_max_u32(tops + 4u * static_cast<uint32_t>(den_group<P>(at, I)), __float_as_uint(st.den[I]));
  });
}

template <class P>
__device__ __forceinline__ void den_tables_out(uint8_t* smem, const State<P>& st, int warp,
                                               int lane) {
  const DenPlace at = den_place<P>(warp, lane);
  const float* const tops = reinterpret_cast<const float*>(smem + P::kDenMax);
  //: a word's row (its group's) at the group's scale, its place a pair `p` at word `2 (p & 7) + (p >> 3)`; a lane's
  //: four words in two groups (`I >> 1` a digit's, `I & 1` a same-split arm's list's)
  float scale[2];
  static_for<2>([&](auto B) {
    scale[B] = pow2(kImageTop - den_exp(tops[den_group<P>(at, P::kPublishRegs ? B : 2 * B)]));
  });
  static_for<4>([&](auto I) {
    const int group = den_group<P>(at, I), k = den_k<P>(at, I), pair = k >> 1;
    *reinterpret_cast<__half*>(smem + P::kDenRows + 64 * group + 4 * (2 * (pair & 7) + (pair >> 3))
                               + 2 * (k & 1)) =
        __float2half_rn(st.den[I] * scale[P::kPublishRegs ? (I & 1) : (I >> 1)]);
  });
  ops::store_shared_f32_if(ops::smem_addr(smem + P::kDenFactor) + 4u * static_cast<uint32_t>(lane),
                           pow2_wide(den_exp(tops[lane]) - kImageTop), warp == 0);
}

//: THE BOUNDS TABLE cleared for a window's counts (its last window's drain factors read): a thread its share, 16
//: bytes a store.
template <class P>
__device__ __forceinline__ void bounds_clear(const CarryParams& p, uint8_t* smem) {
  static_assert(P::kWindow % 16 == 0,
                "the table 16-byte aligned and whole 16-byte units (tables_of)");
#pragma unroll 1
  for (int i = ops::opaque(static_cast<int>(threadIdx.x)); i < p.tables.heads * (P::kWindow / 4);
       i += kThreads)
    reinterpret_cast<uint4*>(smem + p.tables.bounds_at)[i] = make_uint4(0u, 0u, 0u, 0u);
}

//: A LIST'S SCALE for a digit, `2^(a - e)` held to a normal float's exponents: the difference and the float's bits each
//: one integer multiply-add (the FMA pipe; the bits' multiplier and bias `bits` loaded, so ptxas cannot fold them into
//: an ALU shift), the two clamps on the ALU pipe.
__device__ __forceinline__ float list_scale(int a, int e, uint2 bits) {
  int x;
  asm("mad.lo.s32 %0, %1, -1, %2;" : "=r"(x) : "r"(e), "r"(a));
  x = min(max(x, -126), 127);
  uint32_t f;
  asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(f) : "r"(x), "r"(bits.x), "r"(bits.y));
  return __uint_as_float(f);
}

//: A BATCH'S STORES, the hand-off: the warp's lists' rows of batches `lo .. hi - 1` (8 digits each) into the image at
//: `image` (digit `d`'s plane `d & Mask`), f16 at each digit's power of two, each list moved there from its unit by
//: `list_scale`; a lane's row of a list one 16-byte store at its held base and a compile-time offset.
//: -- see docs/internals/carry/parts.md#store
//: a lane's unit for list `w + 8 j` at residue `g` is its unit for list `w`, 32 units on: so a batch's stores take
//: one base a lane, a list an offset
constexpr bool units_step_by_list() {
  for (int w = 0; w < kWarps; ++w)
    for (int k = 0; k < 4; ++k)
      for (int g = 0; g < 8; ++g)
        for (int j = 0; j < kBoxLists / kWarps; ++j)
          if (unit_of(w + kWarps * j, k, g) != unit_of(w, k, g) + 32u * j) return false;
  return true;
}

static_assert(units_step_by_list(),
              "a list's image unit is its warp's first list's, 32 units a list on");

template <class P, uint32_t Mask>
__device__ __forceinline__ void store_batches(const State<P>& st, uint8_t* smem,
                                              ops::SmemAddr image, int lo, int hi, int warp,
                                              int lane) {
  const int g = lane >> 2, t = lane & 3;
  //: the lane's base, opaque, so its terms are formed at each call rather than hoisted across the windows
  uint32_t at;
  asm volatile("mov.u32 %0, %1;"
               : "=r"(at)
               : "r"(image + 16u * unit_of(warp, t, g) + static_cast<uint32_t>(g) * kPlaneBytes));
  int4 ex;
  asm volatile("ld.shared.v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(ex.x), "=r"(ex.y), "=r"(ex.z), "=r"(ex.w)
               : "r"(ops::smem_addr(smem + P::kExps) + 16u * static_cast<uint32_t>(g))
               : "memory");
  const volatile uint32_t* const bp = reinterpret_cast<const volatile uint32_t*>(smem + P::kBits);
  const uint2 bits = make_uint2(bp[0], bp[1]);
  const int es[2][2] = {{ex.x, ex.y}, {ex.z, ex.w}};
  int a[P::kFoldLists];
  list_units<P>(a, smem, warp);
  static_for<P::kFoldLists>([&](auto J) { a[J] += kImageTop; });
  static_for<2>([&](auto M) {
    static_for<2>([&](auto H) {
      constexpr int b = 2 * M + H;
      if (b >= lo && b < hi) {
        static_for<P::kFoldLists>([&](auto J) {
          const float scale = list_scale(a[J], es[M][H], bits);
          uint32_t v[4];
          static_for<4>([&](auto N) {
            const __half2 h = __floats2half2_rn(st.acc[J][M][N][2 * H] * scale,
                                                st.acc[J][M][N][2 * H + 1] * scale);
            v[N] = *reinterpret_cast<const uint32_t*>(&h);
          });
          const uint32_t to = at + ((16u * M + 8u * H) & Mask) * kPlaneBytes + 512u * J;
          asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(to), "r"(v[0]), "r"(v[1]),
                       "r"(v[2]), "r"(v[3])
                       : "memory");
        });
      }
    });
  });
}

//: A LANE'S IMAGE UNIT for a warp's digits' state fragments (the A operand: 32 channels x 32 lists, `ldmatrix.trans`),
//: in bytes: its first, at the warp's swizzle -- a warp reads digits `warp + kWarps i`, one swizzle -- its four this,
//: `+ 1024` and `^ 32` (`image_units_step`). -- see docs/internals/carry/parts.md#image
__host__ __device__ __forceinline__ constexpr uint32_t image_lane(int lane, int warp) {
  return 16u * unit_of(8 * (lane >> 4) + (lane & 7), (lane >> 3) & 1, warp);
}

constexpr bool image_units_step() {
  for (int lane = 0; lane < 32; ++lane)
    for (int d = 0; d < kBoxDigits; ++d)
      for (int k = 0; k < 4; ++k)
        if (16u
                * unit_of(8 * (lane >> 4) + (lane & 7) + 16 * (k & 1),
                          2 * (k >> 1) + ((lane >> 3) & 1), d)
            != ((image_lane(lane, d % kWarps) ^ (32u * (k >> 1))) + 1024u * (k & 1)))
          return false;
  return true;
}

static_assert(image_units_step(),
              "a lane's four image units one base, + 1024 and ^ 32, a warp's digits one swizzle");

//: A DIGIT'S STATE FRAGMENTS from its image plane at `plane`: `(m * 2 + kb) * 4 + i`, m the channel tile, kb the
//: list half; `at` the lane's unit (`image_lane`).
__device__ __forceinline__ void state_in(uint32_t (&f)[16], ops::SmemAddr plane, uint32_t at) {
  const ops::SmemAddr b0 = plane + at, b2 = plane + (at ^ 32u);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    uint32_t a[4];
    ops::load_frag_t(a, (k >> 1 ? b2 : b0) + 1024u * (k & 1));
#pragma unroll
    for (int i = 0; i < 4; ++i) f[k * 4 + i] = a[i];
  }
}

//: THE EXACT PAIR: two outputs times the coefficient to the integer hi x 2^16 + lo, exact under u32 adds while each
//: final fits s16. -- see docs/internals/carry/parts.md#pair
__device__ __forceinline__ uint32_t s16_pair(float lo, float hi, float cu) {
  const uint32_t l = __float_as_uint(fmaf(lo, cu, 12582912.0f));
  const uint32_t h = __float_as_uint(fmaf(hi, cu, 12563648.0f));
  return (h << 16) + l;
}

//: a 16-bit shared load, zero-extended
__device__ __forceinline__ uint32_t ld_u16(ops::SmemAddr at) {
  uint32_t v;
  asm volatile("ld.shared.u16 %0, [%1];" : "=r"(v) : "r"(at));
  return v;
}

__device__ __forceinline__ void panel_red(uint32_t at, uint32_t v) {
  asm volatile("red.shared.add.u32 [%0], %1;" ::"r"(at), "r"(v) : "memory");
}

//: A READOUT TILE, 8 tokens on N against a digit's state (the A operand), in three parts a caller can interleave over
//: two tiles: its operands (`tile_in`: the read rows, B, at the entries' 16-byte chunks by one `ldmatrix.x4`; lane `l`
//: holds entry `l & 7` and scales its coefficient to its token's unit by `2^(sx - u)`, `sx` the digit's exponent less
//: `kImageTop` and `kCoefTop`), its four MMAs (`tile_mma`, accumulating in fp16 (`F16`) or fp32) and its adds
//: (`tile_add`: each lane's two tokens' outputs as exact pairs at their panel rows).
//: -- see docs/internals/carry/parts.md#readout
struct TileIn {
  uint32_t b[4], e0, e1, ex;
  float c0, c1, cs;
};

//: A COEFFICIENT (an f16 value, exponent field 103..142) times 2^shift by its exponent field, the shift held to
//: [-100, 100]: a shift under -100 leaves a product that rounds to no integer at the unit (the image holds a digit under
//: 2^15), a shift over 100 is no valid unit's (a bound's shift is at most 24); without it the field wraps to inf or NaN
//: for a zero digit read beside a live one.
__device__ __forceinline__ float coef_at_unit(float c, int shift) {
  return __uint_as_float(__float_as_uint(c)
                         + (static_cast<uint32_t>(min(max(shift, -100), 100)) << 23));
}

template <class P>
__device__ __forceinline__ TileIn tile_in(ops::SmemAddr pool, ops::SmemAddr units, int sx, uint2 e,
                                          int lane) {
  const int t = lane & 3;
  TileIn in;
  int u;
  asm volatile("ld.shared.s8 %0, [%1];" : "=r"(u) : "r"(units + (e.x >> 16)));
  const float c = __half2float(__ushort_as_half(static_cast<unsigned short>(e.y)));
  const float cu = coef_at_unit(c, sx - u);
  in.e0 = __shfl_sync(0xffffffffu, e.y, 2 * t);
  in.e1 = __shfl_sync(0xffffffffu, e.y, 2 * t + 1);
  in.c0 = __shfl_sync(0xffffffffu, cu, 2 * t);
  in.c1 = __shfl_sync(0xffffffffu, cu, 2 * t + 1);
  in.ex = e.x;
  in.cs = c;
  ops::load_frag(in.b, chunk_addr<P::kReadRow>(pool + 16u * (e.x & 0xffffu),
                                               static_cast<uint32_t>(lane >> 3)));
  return in;
}

//: @burst
template <bool F16>
__device__ __forceinline__ void tile_mma(float (&acc)[2][4], const uint32_t (&b)[4],
                                         const uint32_t (&f)[16]) {
  static_for<2>([&](auto M) {
    constexpr int m = decltype(M)::value;
    uint32_t h[2];
    static_for<2>([&](auto KB) {
      constexpr int kb = decltype(KB)::value;
      const uint32_t a[4] = {f[(m * 2 + kb) * 4], f[(m * 2 + kb) * 4 + 1], f[(m * 2 + kb) * 4 + 2],
                             f[(m * 2 + kb) * 4 + 3]};
      //: the first list half into a fresh fragment, the second accumulating
      if constexpr (F16 && kb == 0) {
        ops::hmma_f16_fresh(h, a, b[0], b[1]);
      } else if constexpr (F16) {
        ops::hmma_f16(h, a, b[2], b[3]);
      } else if constexpr (kb == 0) {
        ops::hmma_fresh(acc[m], a, b[0], b[1]);
      } else {
        ops::hmma(acc[m], a, b[2], b[3]);
      }
    });
    if constexpr (F16) {
      const float2 g0 = __half22float2(*reinterpret_cast<const __half2*>(&h[0])),
                   g8 = __half22float2(*reinterpret_cast<const __half2*>(&h[1]));
      acc[m][0] = g0.x;
      acc[m][1] = g0.y;
      acc[m][2] = g8.x;
      acc[m][3] = g8.y;
    }
  });
}

//: `lane_pads` is the dummy panel rows' address plus the lane's word, `4 (lane >> 2)`: a panel row is 32-byte aligned and
//: the word under 32, so the second channel half's row is the first's XOR 32 with the word kept.
__device__ __forceinline__ void tile_add(ops::SmemAddr lane_pads, const TileIn& in,
                                         const float (&acc)[2][4]) {
  const uint32_t b0 = lane_pads + (in.e0 >> 16), b1 = lane_pads + (in.e1 >> 16);
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    panel_red(b0 ^ (32u * m), s16_pair(acc[m][0], acc[m][2], in.c0));
    panel_red(b1 ^ (32u * m), s16_pair(acc[m][1], acc[m][3], in.c1));
  }
}

//: A TILE OF M DIGITS (the unified readout's OUTPUT SCALING, m = M): 8 tokens' rows loaded once (`row0` the lane's row),
//: each lane's two tokens' panel rows and each digit's coefficient at the token's unit (`2^(sx - u)`); its exit each
//: digit's exact pair summed as u32 words before one atomic a token and channel pair -- 4 atomics a tile at every M. A
//: digit's pair is exact while its own products lie within 2^22 of the pair's offset; chained as one FFMA sum the digits'
//: products leave that range (their s16 bound is the final sum's). -- see docs/internals/carry/parts.md#readout
template <int M>
struct TileInM {
  uint32_t b[4], e0, e1;
  float c0[M], c1[M];
};

template <class P, int M>
__device__ __forceinline__ TileInM<M> tile_in_m(ops::SmemAddr row0, ops::SmemAddr units,
                                                const int (&sx)[M], uint2 e,
                                                const uint32_t (&coef)[M], int lane) {
  const int t = lane & 3;
  TileInM<M> in;
  int u;
  asm volatile("ld.shared.s8 %0, [%1];" : "=r"(u) : "r"(units + (e.x >> 16)));
  in.e0 = __shfl_sync(0xffffffffu, e.y, 2 * t);
  in.e1 = __shfl_sync(0xffffffffu, e.y, 2 * t + 1);
  static_for<M>([&](auto Q) {
    const float c = __half2float(__ushort_as_half(static_cast<unsigned short>(coef[Q])));
    const float cu = coef_at_unit(c, sx[Q] - u);
    in.c0[Q] = __shfl_sync(0xffffffffu, cu, 2 * t);
    in.c1[Q] = __shfl_sync(0xffffffffu, cu, 2 * t + 1);
  });
  ops::load_frag(in.b, chunk_addr<P::kReadRow>(row0, static_cast<uint32_t>(lane >> 3)));
  return in;
}

template <int M>
__device__ __forceinline__ void tile_add_m(ops::SmemAddr lane_pads, const TileInM<M>& in,
                                           const float (&acc)[M][2][4]) {
  const uint32_t b0 = lane_pads + (in.e0 >> 16), b1 = lane_pads + (in.e1 >> 16);
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    uint32_t w0 = 0u, w1 = 0u;
    static_for<M>([&](auto Q) {
      w0 += s16_pair(acc[Q][m][0], acc[Q][m][2], in.c0[Q]);
      w1 += s16_pair(acc[Q][m][1], acc[Q][m][3], in.c1[Q]);
    });
    panel_red(b0 ^ (32u * m), w0);
    panel_red(b1 ^ (32u * m), w1);
  }
}

//: a slot 8 on in a chunk: its row 8 rows on (both swizzles keep `s & 7` and `(s >> 1) & 3`) and its panel row 512 bytes
//: on -- the dense readout's tiles step by these
template <int Row>
__host__ __device__ constexpr bool slot_step_is_tile_rows() {
  for (uint32_t s = 0; s < 1024; ++s)
    if (pool_chunk(Row, s + 8, 0) != pool_chunk(Row, s, 0)
        || panel_row(s + 8) != panel_row(s) + 512u)
      return false;
  return true;
}

//: A MASKED LANE'S ENTRY in a chunk's last step: its place `c`'s row of the chunk, coefficient 0, a dummy panel row, so
//: its products add nothing -- the pool and panel offsets by the chunk's first slot `first`, as the entries' are.
template <class P>
__device__ __forceinline__ uint2 masked_entry(int first, uint32_t c) {
  const uint32_t row = static_cast<uint32_t>(first) * (P::kReadRow / 16) + c * (P::kReadRow / 16)
                       + chunk_at<P::kReadRow>(c, 0);
  return make_uint2(row, (64u * static_cast<uint32_t>(first) + panel_row(c)) << 16);
}

//: THE CROSS-CTA ADDS, unpredicated: f16 pairs (no flush to zero), one fp32.
__device__ __forceinline__ void red_add_h2(uint32_t* at, uint32_t v) {
  asm volatile("red.global.add.noftz.f16x2 [%0], %1;" ::"l"(at), "r"(v) : "memory");
}

__device__ __forceinline__ void red_add_f32(float* at, float v) {
  asm volatile("red.global.add.f32 [%0], %1;" ::"l"(at), "f"(v) : "memory");
}

//: A TOKEN'S ROW in `den` (and in `num`, in units of `dv`): its units index `u` (a query head's token in the window)
//: as the head's row and the window's token.
__device__ __forceinline__ uint32_t token_row(uint32_t u, uint32_t L) {
  return (u >> window_bits(kWindow) & static_cast<uint32_t>(window_heads(kWindow) - 1)) * L
         + (u & (kWindow - 1));
}

//: THE DEN'S ADDS, fire and forget: a lane one distinct term of a group, the group's 32 one unpredicated
//: `red.global.add.f32` between step segments (a lane with none an exact zero). -- see docs/internals/carry/parts.md#den
constexpr uint32_t kNoToken = 0xffffffffu;

struct DenAdd {
  float v = 0.0f;
  uint32_t idx = kNoToken;

  __device__ __forceinline__ void take(bool mine, float x, uint32_t u) {
    v = mine ? x : v;
    idx = mine ? u : idx;
  }

  __device__ __forceinline__ void add(const CarryParams& p, int t0, int lane) {
    const uint32_t L = static_cast<uint32_t>(p.L);
    const uint32_t row =
        idx == kNoToken ? static_cast<uint32_t>(min(lane, p.L - 1 - t0)) : token_row(idx, L);
    red_add_f32(p.den + static_cast<int64_t>(blockIdx.y) * p.tables.heads * p.L + t0 + row, v);
    v = 0.0f;
    idx = kNoToken;
  }
};

//: A DIGIT'S DEN B WORDS for a tile, the lane's two (`row` its own, 8 bytes on by `lane & 7`).
__device__ __forceinline__ uint2 den_b(ops::SmemAddr row) {
  uint2 w;
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(w.x), "=r"(w.y) : "r"(row));
  return w;
}

//: THE DEN'S PRODUCTS a tile at the channels' accumulation: the tile's read rows as A against the den row as B's
//: columns 0 and 1, two m16n8k8 MMAs; token `r`'s den `D[r][0] + D[r + 8][1]`, each lane its own entry's.
//: -- see docs/internals/carry/parts.md#den
template <bool F16>
__device__ __forceinline__ float den_mma(const uint32_t (&b)[4], uint2 w, int lane) {
  float x;
  if constexpr (F16) {
    uint32_t h[2] = {0u, 0u};
    ops::hmma_k8_f16(h, b[0], b[1], w.x);
    ops::hmma_k8_f16(h, b[2], b[3], w.y);
    x = __half2float(__low2half(*reinterpret_cast<const __half2*>(&h[0])))
        + __half2float(__high2half(*reinterpret_cast<const __half2*>(&h[1])));
  } else {
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    ops::hmma_k8(c, b[0], b[1], w.x);
    ops::hmma_k8(c, b[2], b[3], w.y);
    x = c[0] + c[3];
  }
  return __shfl_sync(0xffffffffu, x, 4 * (lane & 7));
}

//: A LIST'S STATE FRAGMENTS from the fold's accumulators (a same-split arm's readout), `(m * 2 + kb) * 4 + i` as
//: `state_in`'s with the image's channel order (an A block `b`'s rows channels `8 (r >> 1) + 2 b + (r & 1)`, the pairs a
//: fold lane holds): each digit block's four n-tile pairs packed f16 at `scale`, transposed across the quad (lane `t`
//: then holding lane `b`'s pair of n-tile `t`, two butterfly rounds) and each 8x8 transposed where it lies.
//: -- see docs/internals/carry/parts.md#same-split
template <class P, int J>
__device__ __forceinline__ void state_from(uint32_t (&f)[16], const State<P>& st, float scale,
                                           int lane) {
  const bool hi = (lane & 2) != 0, odd = (lane & 1) != 0;
  static_for<2>([&](auto Mt) {
    static_for<2>([&](auto Ht) {
      constexpr int m = decltype(Mt)::value, h = decltype(Ht)::value;
      uint32_t x[4];
      static_for<4>([&](auto N) {
        const __half2 v =
            __floats2half2_rn(st.acc[J][m][N][2 * h] * scale, st.acc[J][m][N][2 * h + 1] * scale);
        x[N] = *reinterpret_cast<const uint32_t*>(&v);
      });
      //: the quad's 4 x 4 transpose: the off-diagonal pair of n-tiles across `xor 2`, then the odd ones across `xor 1`
      const uint32_t r0 = __shfl_xor_sync(0xffffffffu, hi ? x[0] : x[2], 2),
                     r1 = __shfl_xor_sync(0xffffffffu, hi ? x[1] : x[3], 2);
      x[0] = hi ? r0 : x[0];
      x[1] = hi ? r1 : x[1];
      x[2] = hi ? x[2] : r0;
      x[3] = hi ? x[3] : r1;
      const uint32_t s0 = __shfl_xor_sync(0xffffffffu, odd ? x[0] : x[1], 1),
                     s2 = __shfl_xor_sync(0xffffffffu, odd ? x[2] : x[3], 1);
      x[0] = odd ? s0 : x[0];
      x[1] = odd ? x[1] : s0;
      x[2] = odd ? s2 : x[2];
      x[3] = odd ? x[3] : s2;
      static_for<4>([&](auto B) {
        constexpr int b = decltype(B)::value;
        f[(2 * (b >> 1) + m) * 4 + (b & 1) + 2 * h] = ops::transpose_frag_b16(x[b]);
      });
    });
  });
}

//: A CHUNK'S READOUT over batches `lo .. hi - 1`, a warp's: each of its digits `8 i + warp`, its state from its image
//: plane (`d & mask` at `image`) and its exponent, its tiles over the chunk -- a listed side's run of the digit (`run`,
//: the chunk's) in the buffer's entries at `ents`, a lane past it masked; else every slot of the chunk, its entry
//: implied. `Pairs`: the digits two at a time, their tiles stepped together (two independent chains, all loads before
//: the MMAs, the adds after). The rows are at `rows`, the dummy panel rows at `pads`, the panel after them.
//: -- see docs/internals/carry/parts.md#readout
template <class P, bool F16, bool Pairs>
__device__ __forceinline__ void read_chunk(const CarryParams& p, const uint8_t* smem,
                                           const State<P>& st, const Window& rw, const Run& run,
                                           int first, int slots, int lo, int hi,
                                           ops::SmemAddr image, uint32_t mask, ops::SmemAddr rows,
                                           ops::SmemAddr pads, ops::SmemAddr ents, int warp,
                                           int lane) {
  constexpr int kStep = Pairs ? 2 : 1;
  const uint32_t at = image_lane(lane, warp);
  //: the lane's B words of a digit's den row
  const ops::SmemAddr den_rows =
      ops::smem_addr(smem + P::kDenRows) + 8u * static_cast<uint32_t>(lane & 7);
  const ops::SmemAddr units = ops::smem_addr(smem + p.tables.units_at);
  //: the entries' row and panel offsets are the union's: the chunk's pool taken back by its first slot, and each
  //: entry's panel offset made the chunk's (modulo 2^16, as a grouped union's are written) before it is shuffled
  const ops::SmemAddr pool_u = rows - static_cast<uint32_t>(first * P::kReadRow);
  const uint32_t panel_back = static_cast<uint32_t>(64 * first) << 16;
  const uint2 masked = masked_entry<P>(first, static_cast<uint32_t>(lane & 7));
  const ops::SmemAddr lane_pads = pads + 4u * static_cast<uint32_t>(lane >> 2);
  //: a lane's entry `e` of a run (`n` at `to`), the masked entry past the run: loaded unclamped (a place past the run
  //: is the next run's or the shared memory after the buffer, and is not used), its panel offset made the chunk's
  //: modulo 2^16
  const auto entry_at = [&](int e, int n, int to) {
    uint2 v;
    asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];"
                 : "=r"(v.x), "=r"(v.y)
                 : "r"(ents + 8u * static_cast<uint32_t>(to + e)));
    v = e < n ? v : masked;
    v.y -= panel_back;
    return v;
  };
  //: a same-split arm's groups are the warp's lists: their units and the scales' bits for their fragments' scale
  int unit[P::kFoldLists] = {};
  uint2 bits = make_uint2(0u, 0u);
  if constexpr (P::kPublishRegs) {
    list_units<P>(unit, smem, warp);
    const volatile uint32_t* const bp = reinterpret_cast<const volatile uint32_t*>(smem + P::kBits);
    bits = make_uint2(bp[0], bp[1]);
  }
  //: A STEP of the warp's groups `i ..` (`kStep`): each one's state fragments (its image plane, or on a same-split arm
  //: its list's accumulators), exponent and den factor, then its tiles
  const auto step = [&](auto i) {
    int d[kStep], sx[kStep];
    uint32_t f[kStep][16];
    float fac[kStep];
    static_for<kStep>([&](auto Q) {
      d[Q] = warp + kWarps * (i + Q);
      const int e = reinterpret_cast<const int*>(smem + P::kExps)[group_slot<P>(d[Q])];
      sx[Q] = e - kImageTop - kCoefTop;
      if constexpr (P::kPublishRegs) {
        constexpr int j = decltype(i)::value + decltype(Q)::value;
        state_from<P, j>(f[Q], st, list_scale(unit[j] + kImageTop, e, bits), lane);
      } else {
        state_in(f[Q], image + (static_cast<uint32_t>(d[Q]) & mask) * kPlaneBytes, at);
      }
      fac[Q] = reinterpret_cast<const float*>(smem + P::kDenFactor)[d[Q]];
    });

    //: the arm's read rows say the side (one factor listed, two dense, as the units take them)
    if constexpr (P::kReadFactors == 1) {
      int n[kStep], to[kStep], most = 0;
      static_for<kStep>([&](auto Q) {
        n[Q] = __shfl_sync(0xffffffffu, run.n, d[Q]);
        to[Q] = __shfl_sync(0xffffffffu, run.to, d[Q]);
        most = max(most, n[Q]);
      });
      const int tiles = (most + 7) >> 3;

      //: THE DEN'S GROUP (`DenAdd`), the run's: `kGroup` tiles' terms, a lane one (tile `dk`, digit `dq`), `k` tiles in
      constexpr int kGroup = 32 / (8 * kStep);
      const int dq = (lane >> 3) % kStep, dk = (lane >> 3) / kStep;
      DenAdd dadd;
      int k = 0, s = 0;
      //: an image arm's chains load their next tile's entry a tile ahead (the last tile's next past the run: masked,
      //: unused); a same-split arm's registers, holding its lists' state, re-schedule worse for it (`#readout`)
      constexpr bool kAhead = !P::kPublishRegs;
      uint2 ahead[kStep];
      if constexpr (kAhead)
        static_for<kStep>([&](auto Q) { ahead[Q] = entry_at(lane & 7, n[Q], to[Q]); });

      //: the run's tiles in segments ending where the den's group fills, its adds between them (§29 item 7)
      for (int any = __any_sync(0xffffffffu, s < tiles) ? 1 : 0; any > 0; --any)
#pragma unroll 1
        do {
#pragma unroll 1
          do {
            TileIn in[kStep];
            uint2 dw[kStep];
            float acc[kStep][2][4], dn[kStep];
            static_for<kStep>([&](auto Q) {
              uint2 e;
              if constexpr (kAhead) {
                e = ahead[Q];
                ahead[Q] = entry_at(8 * s + 8 + (lane & 7), n[Q], to[Q]);
              } else {
                e = entry_at(8 * s + (lane & 7), n[Q], to[Q]);
              }
              in[Q] = tile_in<P>(pool_u, units, sx[Q], e, lane);
              dw[Q] = den_b(den_rows + 64u * static_cast<uint32_t>(d[Q]));
            });
            static_for<kStep>([&](auto Q) { tile_mma<F16>(acc[Q], in[Q].b, f[Q]); });
            static_for<kStep>([&](auto Q) { dn[Q] = den_mma<F16>(in[Q].b, dw[Q], lane); });
            static_for<kStep>([&](auto Q) { tile_add(lane_pads, in[Q], acc[Q]); });
            //: THE DEN (`#den`): each digit's entries' terms, absolute, a lane its group slot's
            static_for<kStep>([&](auto Q) {
              dadd.take(dq == decltype(Q)::value && dk == k && 8 * s + (lane & 7) < n[Q],
                        dn[Q] * (in[Q].cs * fac[Q]), in[Q].ex >> 16);
            });
            ++k;
          } while (__any_sync(0xffffffffu, (++s < tiles) & (k < kGroup)));
          for (int full = __any_sync(0xffffffffu, k == kGroup) ? 1 : 0; full > 0; --full) {
            dadd.add(p, rw.t0, lane);
            k = 0;
          }
        } while (__any_sync(0xffffffffu, s < tiles));
      for (int rest = __any_sync(0xffffffffu, k > 0) ? 1 : 0; rest > 0; --rest)
        dadd.add(p, rw.t0, lane);
    } else {
      //: every slot of the chunk, its entry implied: its row and token, each digit's coefficient, its panel row (a slot
      //: past the chunk's end its place's dummy row); the step's digits summed into one exit a tile (every token is in
      //: every digit's list, so their union is each digit's own)
      //: the lane's slot `8 s + (lane & 7)`, stepped by 8 a tile: its row 8 rows on (the swizzle keeps `s & 7`
      //: and `(s >> 1) & 3`), its panel row 512 bytes on (`panel_row`), its token 8 on; past the chunk its place's
      //: dummy row (`slot_step_is_tile_rows`)
      static_assert(slot_step_is_tile_rows<P::kReadRow>(),
                    "a slot 8 on: its row 8 rows on, its panel 512");
      const uint32_t l7 = static_cast<uint32_t>(lane & 7);
      uint32_t slot = l7,
               row = rows + 16u * (l7 * (P::kReadRow / 16) + chunk_at<P::kReadRow>(l7, 0)),
               live = (kPadBytes + panel_row(l7)) << 16, tok = (first + l7) << 16;
      const uint32_t dummy = panel_row(l7) << 16;
      const int tiles = (slots + 7) >> 3;

      int s = 0;

      //: the tiles, one block a step (the den is the bound pass's: `units_dense`)
      for (int any = __any_sync(0xffffffffu, slots > 0) ? 1 : 0; any > 0; --any)
#pragma unroll 1
        do {
          uint32_t coef[kStep];
          static_for<kStep>([&](auto Q) {
            coef[Q] = ld_u16(chunk_addr<P::kReadRow>(row, (kSliceBytes + 2 * d[Q]) >> 4)
                             + (2 * d[Q] & 15));
          });
          const TileInM<kStep> in = tile_in_m<P, kStep>(
              row, units, sx, make_uint2(tok, static_cast<int>(slot) < slots ? live : dummy), coef,
              lane);
          float acc[kStep][2][4];
          static_for<kStep>([&](auto Q) { tile_mma<F16>(acc[Q], in.b, f[Q]); });
          tile_add_m(lane_pads, in, acc);
          slot += 8u;
          row += 8u * P::kReadRow;
          live += 512u << 16;
          tok += 8u << 16;
        } while (__any_sync(0xffffffffu, ++s < tiles));
    }
  };
  if constexpr (P::kPublishRegs) {
    //: the warp's four lists, two a step (the batches all of them: no image to part)
    static_for<P::kFoldLists / kStep>(
        [&](auto S) { step(std::integral_constant<int, kStep* decltype(S)::value>{}); });
  } else {
    int i = lo;
    for (int go = __any_sync(0xffffffffu, i < hi) ? 1 : 0; go > 0; --go)
#pragma unroll 1
      do {
        step(i);
      } while (__any_sync(0xffffffffu, (i += kStep) < hi));
  }
}

//: THE DRAIN'S FACTOR a token, written beside its unit byte into the bounds table (free once its counts are read): f16
//: pairs `2^(unit - scale - 1)` (the den pass's scale byte), fp32 `2^unit`; exact over the float range.
//: -- see docs/internals/carry/parts.md#drain
template <class P>
__device__ __forceinline__ float drain_factor(const CarryParams& p, const uint8_t* smem, int w,
                                              uint32_t idx, int unit) {
  const int sb = reinterpret_cast<const int8_t*>(
      smem + p.tables.scale_at)[(w & 1) * p.tables.heads * kWindow + idx];
  return pow2_wide(p.reduce == kReduceF32 ? unit : unit - sb - 1);
}

//: THE f16 BOUND'S MARGIN: a listed bound's dot product runs as two packed f16 chains of `kBoundChain` fused products,
//: each product and each partial rounded to nearest by at most 2^-11 of itself -- 2 x kBoundChain roundings a chain --
//: so the sum is raised to stay a bound; the dense bound's MMAs round at most twice a tile. The assertion holds the
//: margin to the chain's length: a longer chain needs a larger margin.
constexpr int kBoundChain = 8;
constexpr float kF16BoundMargin = 1.0f + 1.0f / 64.0f;
static_assert((1.0 - 2.0 * kBoundChain / 2048.0) * kF16BoundMargin >= 1.0,
              "the f16 margin covers the bound chain's roundings");

//: THE CTA'S UNITS AND DEN for a read chunk with no list level: each token's rows against the leaves' scales and den
//: rows, times each group's coefficient and factor, summed; the bound into the bounds table, the den one global add a
//: token a CTA. -- see docs/internals/carry/parts.md#units
template <class P>
__device__ __forceinline__ void units_dense(const CarryParams& p, const uint8_t* smem, int first,
                                            int slots, ops::SmemAddr rows, int t0, int warp,
                                            int lane) {
  static_assert(P::kReadRow == 128 && P::kReadFactors == 2,
                "a dense read row: its list slice, then its coefficients");
  const int g = lane >> 2, t = lane & 3;
  const ops::SmemAddr dead = ops::smem_addr(smem + P::kDeadUnit),
                      bound_at = ops::smem_addr(smem + p.tables.bounds_at);
  const uint32_t* const tab = reinterpret_cast<const uint32_t*>(smem + P::kLeafScale);
  const volatile uint32_t* const dtab =
      reinterpret_cast<const volatile uint32_t*>(smem + P::kDenRows);
  const float* const dfac = reinterpret_cast<const float*>(smem + P::kDenFactor);
  uint32_t a[2][2][4], da[2][2][4];
  float sc[2][2], fac[2][2];

  //: the bound's A in the table's natural pairs; the den's from the den rows' B words (pair `q` at `2 (q & 7) + (q >> 3)`),
  //: a word a load, so no two K halves fuse into one load's register pair and each MMA's quad stays put (`#units`)
  static_for<2>([&](auto M) {
    static_for<2>([&](auto K) {
      const int r0 = (16 * M + g) * 16, r1 = r0 + 8 * 16, c = 8 * K + t, w = 2 * t + K;
      a[M][K][0] = tab[r0 + c];
      a[M][K][1] = tab[r1 + c];
      a[M][K][2] = tab[r0 + c + 4];
      a[M][K][3] = tab[r1 + c + 4];
      da[M][K][0] = dtab[r0 + w];
      da[M][K][1] = dtab[r1 + w];
      da[M][K][2] = dtab[r0 + w + 8];
      da[M][K][3] = dtab[r1 + w + 8];
    });
    static_for<2>([&](auto H) {
      sc[M][H] =
          kF16BoundMargin
          * pow2(reinterpret_cast<const int*>(smem + P::kExps)[group_slot<P>(16 * M + 8 * H + g)]
                 - kCoefTop);
      fac[M][H] = dfac[16 * M + 8 * H + g];
    });
  });

  const int tiles = (slots + 7) >> 3;
  constexpr int kInFlight = 2, kSegment = 4;
  int s = warp;
  //: THE DEN'S GROUP (`DenAdd`): lane `l` the token `l & 7` of the segment's tile `l >> 3`
  DenAdd dadd;
  int k = 0;

  for (int any = __any_sync(0xffffffffu, s < tiles) ? 1 : 0; any > 0; --any)
#pragma unroll 1
    do {
#pragma unroll 1
      do {
        uint32_t b[kInFlight][4], cw[kInFlight][4], h[kInFlight][2][2];
        float hd[kInFlight][2][4];
        static_for<kInFlight>([&](auto F) {
          const uint32_t slot = 8u * (s + kWarps * F) + (lane & 7);
          const uint32_t row = rows + 128u * slot + 16u * (slot & 7u);
          ops::load_frag(b[F], chunk_addr<P::kReadRow>(row, static_cast<uint32_t>(lane >> 3)));
          ops::load_frag_t(cw[F],
                           chunk_addr<P::kReadRow>(row, 4u + static_cast<uint32_t>(lane >> 3)));
        });

        static_for<kInFlight>([&](auto F) {
          static_for<2>([&](auto M) {
            h[F][M][0] = h[F][M][1] = 0u;
            static_for<2>(
                [&](auto K) { ops::hmma_f16(h[F][M], a[M][K], b[F][2 * K], b[F][2 * K + 1]); });
            ops::hmma_fresh(hd[F][M], da[M][0], b[F][0], b[F][1]);
            ops::hmma(hd[F][M], da[M][1], b[F][2], b[F][3]);
          });
        });
        float sum[kInFlight][2], dn[kInFlight][2];

        static_for<kInFlight>([&](auto F) {
          sum[F][0] = sum[F][1] = dn[F][0] = dn[F][1] = 0.0f;
          static_for<2>([&](auto M) {
            static_for<2>([&](auto H) {
              const float2 v = __half22float2(*reinterpret_cast<const __half2*>(&h[F][M][H]));
              const float2 cc =
                  __half22float2(*reinterpret_cast<const __half2*>(&cw[F][2 * M + H]));
              const float x[2] = {v.x, v.y}, cs[2] = {cc.x, cc.y};
              static_for<2>([&](auto C) {
                const float c = cs[C];
                sum[F][C] = fmaf(c * sc[M][H], x[C], sum[F][C]);
                dn[F][C] = fmaf(c * fac[M][H], hd[F][M][2 * H + C], dn[F][C]);
              });
            });
          });
        });

        //: THE SUMS over a tile's digits by a reduce-scatter, a lane keeping token `2 t + c` (`c` its `g`'s low bit)
        const int c = (lane >> 2) & 1;
        float bound[kInFlight], den[kInFlight];
        static_for<kInFlight>([&](auto F) {
          bound[F] = (c ? sum[F][1] : sum[F][0])
                     + __shfl_xor_sync(0xffffffffu, c ? sum[F][0] : sum[F][1], 4);
          den[F] =
              (c ? dn[F][1] : dn[F][0]) + __shfl_xor_sync(0xffffffffu, c ? dn[F][0] : dn[F][1], 4);
          static_for<2>([&](auto S) {
            bound[F] += __shfl_xor_sync(0xffffffffu, bound[F], 8 << S);
            den[F] += __shfl_xor_sync(0xffffffffu, den[F], 8 << S);
          });
        });

        static_for<kInFlight>([&](auto F) {
          const uint32_t tok = 8u * (s + kWarps * F) + 2u * t + static_cast<uint32_t>(c);
          const uint32_t at = static_cast<int>(tok) < slots ? bound_at + 4u * (first + tok) : dead;
          asm volatile("st.shared.f32 [%0], %1;" ::"r"(at), "f"(bound[F]) : "memory");
          const int r = lane & 7;
          const float d = __shfl_sync(0xffffffffu, den[F], ((r & 1) << 2) | (r >> 1));
          const int slot = 8 * (s + kWarps * F) + r;
          dadd.take((lane >> 3) == k + F && slot < slots, d, static_cast<uint32_t>(first + slot));
        });
        k += kInFlight;
      } while (__any_sync(0xffffffffu, ((s += kInFlight * kWarps) < tiles) & (k < kSegment)));
      for (int full = __any_sync(0xffffffffu, k == kSegment) ? 1 : 0; full > 0; --full) {
        dadd.add(p, t0, lane);
        k = 0;
      }
    } while (__any_sync(0xffffffffu, s < tiles));
  for (int rest = __any_sync(0xffffffffu, k > 0) ? 1 : 0; rest > 0; --rest) dadd.add(p, t0, lane);
}

//: THE LISTED BOUNDS' TERMS for a read chunk, a thread an entry across the CTA (the chunk's runs contiguous in digit
//: order, an entry's digit the last run starting at or before it: five shuffles): its row against its digit's scales
//: (packed f16 products in two chains, raised by the f16 margin), times its coefficient and `2^(e_d - kCoefTop)`,
//: quantized up at 2^-24 of the window's reference `2^(emax + 5)` and added into its token's counter by a shared add;
//: `units_close` turns the counters into bytes after a barrier. -- see docs/internals/carry/parts.md#units
template <class P>
__device__ __forceinline__ void units_listed_add(const CarryParams& p, const uint8_t* smem,
                                                 const Run& run, int first, ops::SmemAddr rows,
                                                 ops::SmemAddr ents, int warp, int lane) {
  const int entries = __shfl_sync(0xffffffffu, run.to + run.n, kBoxDigits - 1);
  const ops::SmemAddr pool_u = rows - static_cast<uint32_t>(first * P::kReadRow);
  const ops::SmemAddr bounds = ops::smem_addr(smem + p.tables.bounds_at);
  const int emax = reinterpret_cast<const volatile int2*>(smem + P::kTops)->y;
  const int tid = 32 * warp + lane;

#pragma unroll 1
  for (int base = 0; base < entries; base += kThreads) {
    const int i = base + tid;
    //: the entry's digit: the last run whose start is at or before it
    int d = 0;
    static_for<5>([&](auto B) {
      const int probe = d + (16 >> B);
      d = __shfl_sync(0xffffffffu, run.to, probe) <= i ? probe : d;
    });
    if (i < entries) {
      uint2 v;
      asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];"
                   : "=r"(v.x), "=r"(v.y)
                   : "r"(ents + 8u * static_cast<uint32_t>(i)));
      const ops::SmemAddr row = pool_u + 16u * (v.x & 0xffffu);
      const ops::SmemAddr srow =
          ops::smem_addr(smem + P::kLeafScale) + 64u * static_cast<uint32_t>(d);
      __half2 dot[2] = {__float2half2_rn(0.0f), __float2half2_rn(0.0f)};
      static_for<4>([&](auto C) {
        uint4 r, q;
        asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                     : "r"(chunk_addr<P::kReadRow>(row, C)));
        asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(q.x), "=r"(q.y), "=r"(q.z), "=r"(q.w)
                     : "r"(srow + 16u * C));
        const uint32_t rr[4] = {r.x, r.y, r.z, r.w}, qq[4] = {q.x, q.y, q.z, q.w};
        static_for<4>([&](auto J) {
          dot[J & 1] = __hfma2(*reinterpret_cast<const __half2*>(&rr[J]),
                               *reinterpret_cast<const __half2*>(&qq[J]), dot[J & 1]);
        });
      });
      const float2 d0 = __half22float2(dot[0]), d1 = __half22float2(dot[1]);
      const float k =
          kF16BoundMargin
          * pow2(max(reinterpret_cast<const int*>(smem + P::kExps)[group_slot<P>(d)] - emax + 15,
                     -126));
      const float c = __half2float(__ushort_as_half(static_cast<unsigned short>(v.y)));
      const uint32_t q = __float2uint_ru(c * ((d0.x + d0.y) + (d1.x + d1.y)) * k);
      asm volatile("red.shared.add.u32 [%0], %1;" ::"r"(bounds + 4u * (v.x >> 16)), "r"(q)
                   : "memory");
    }
  }
}

//: THE UNITS from the bounds table, after the bound's barrier: a thread a slot, its token's bound read (a listed side's
//: counter of quanta at `2^(emax - 19)`, a dense side's fp32 bound), its byte into the units table and its drain's
//: factor in the bound's place (a listed gap slot has none); the next window's `bounds_clear` resets the table.
//: -- see docs/internals/carry/parts.md#units
template <class P>
__device__ __forceinline__ void units_close(const CarryParams& p, uint8_t* smem, int w, int first,
                                            int slots) {
  const uint16_t* const toks = union_of<P>(p, smem, kRead, w) + first;
  uint32_t* const bounds = reinterpret_cast<uint32_t*>(smem + p.tables.bounds_at);
  int8_t* const units = reinterpret_cast<int8_t*>(smem + p.tables.units_at);
  const int emax = reinterpret_cast<const volatile int2*>(smem + P::kTops)->y;
#pragma unroll 1
  for (int s = ops::opaque(static_cast<int>(threadIdx.x)); s < slots; s += kThreads) {
    uint32_t u = static_cast<uint32_t>(first + s);
    int unit;
    if constexpr (P::kReadFactors == 1) {
      const uint32_t tok = toks[s];
      if (tok & kGap) continue;
      u = tok & 0x7fffu;
      unit = unit_byte(static_cast<float>(bounds[u]), emax - 19);
    } else {
      unit = unit_byte(__uint_as_float(bounds[u]), 0);
    }
    units[u] = static_cast<int8_t>(unit);
    bounds[u] = __float_as_uint(drain_factor<P>(p, smem, w, u, unit));
  }
}

//: A WORD `off` words past `base` (`off` under 2^32), its own add: never re-associated into the base's sum.
template <class T>
__device__ __forceinline__ T* word_at(T* base, uint32_t off) {
  static_assert(sizeof(T) == 4, "a 32-bit word");
  uint64_t a;
  asm("mad.wide.u32 %0, %1, 4, %2;" : "=l"(a) : "r"(off), "l"(reinterpret_cast<uint64_t>(base)));
  return reinterpret_cast<T*>(a);
}

//: A PANEL WORD's TWO HALVES as offset magics: the word accumulated from `kPanelClear` (each half biased by 2^15, the
//: low half's borrow taken back from the high half's bias), each half's 16 bits placed under 2^23 + 2^22 by one byte
//: permute -- a half's value is its float less 12615680. -- see docs/internals/carry/parts.md#drain
__device__ __forceinline__ float2 pair_magics(uint32_t t) {
  constexpr uint32_t top = 0x4B400000u;
  return make_float2(__uint_as_float(ops::prmt(t, top, 0x7610u)),
                     __uint_as_float(ops::prmt(t, top, 0x7632u)));
}

//: a warp's fp32 slots `w + kWarps j` lie 512 panel bytes apart, `panel_row(w)`'s swizzle bits kept
constexpr bool slot_step_is_panel_lines() {
  for (uint32_t w = 0; w < static_cast<uint32_t>(kWarps); ++w)
    for (uint32_t j = 0; j < 64; ++j)
      if (panel_row(w + static_cast<uint32_t>(kWarps) * j) != panel_row(w) + 512u * j) return false;
  return true;
}

//: THE DRAIN: a chunk's panel words (at `panel`) to `num` by the call's reduce dtype, each slot's at the CTA's unit
//: converted by its token's factor; a gap slot adds zero at row 0. A slot's facts (its row, a 32-bit offset, and its
//: factor) formed once a slot by a lane and shuffled to its slot's lanes; four slots a step, a step past the warp's last
//: not issued. F16 PAIRS: a thread one word of every 16th slot, a slot past the chunk in a live step adding zero at the
//: warp's first slot's row. FP32: a warp a slot in turn, a lane a channel. -- see docs/internals/carry/parts.md#drain
template <class P>
__device__ __forceinline__ void drain(const CarryParams& p, const uint8_t* smem, const Window& rw,
                                      const Place& c, int w, int first, int slots,
                                      ops::SmemAddr panel) {
  const ops::SmemAddr tok0 = ops::smem_addr(union_of<P>(p, smem, kRead, w) + first);
  const ops::SmemAddr factor_at = ops::smem_addr(smem + p.tables.bounds_at);
  const uint32_t L = static_cast<uint32_t>(p.L);
  const bool listed = rw.listed;
  const int channel0 = reinterpret_cast<const volatile Addr*>(smem + P::kAddr)->channel0;
  const int64_t head0 = (int64_t)c.bh * p.tables.heads * p.L + rw.t0;

  //: a slot's units index (a query head's token in the window), its row in `num` in units of `dv` channels, and
  //: whether it holds a token (a gap slot holds none)
  const auto place = [&](int s, uint32_t& idx, uint32_t& row) {
    uint32_t u = static_cast<uint32_t>(first + s);
    if (listed) {
      asm volatile("ld.shared.u16 %0, [%1];" : "=r"(u) : "r"(tok0 + 2u * static_cast<uint32_t>(s)));
      u &= 0x7fffu;
    }
    idx = u;
    row = token_row(u, L);
  };

  const auto factor_of = [&](uint32_t idx) {
    float f;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(f) : "r"(factor_at + 4u * idx));
    return f;
  };

  if (p.reduce == kReduceF32) {
    constexpr int kAhead = 4, kBatch = 32;
    static_assert(slot_step_is_panel_lines(),
                  "a warp's next slot 512 panel bytes on, its swizzle bits kept");
    //: the thread, opaque (as below); the warp's slots `w0 + kWarps j`
    const uint32_t ut = static_cast<uint32_t>(ops::opaque(static_cast<int>(threadIdx.x)));
    const int lane = static_cast<int>(ut % 32u);
    const uint32_t w0 = ut / 32u;

    //: lane `l` word `8 m + g` (m = l >> 4, g = l & 7) of a slot, its half `(l >> 3) & 1`: channel `c` or `c + 2`, `c = 8
    //: (g >> 1) + 4 m + (g & 1)` (the image's unit, `unpack_num`) -- a permutation of the box's 32 channels
    const int m = lane >> 4, g = lane & 7;
    const bool high = (lane >> 3) & 1;
    float* const out = reinterpret_cast<float*>(p.num) + head0 * p.dv + channel0 + 8 * (g >> 1)
                       + 4 * m + (g & 1) + 2 * static_cast<int>(high);
    const ops::SmemAddr cell0 =
        panel
        + (panel_row(w0) ^ ((32u * static_cast<uint32_t>(m)) | (4u * static_cast<uint32_t>(g))));
    //: the lane's half's bytes under the magic's (`pair_magics`' permute for the one half)
    const uint32_t half = high ? 0x7632u : 0x7610u, dv = static_cast<uint32_t>(p.dv);
    //: the warp's slots in the chunk, from lane 0 (warp-uniform, as below)
    const int steps = __shfl_sync(
        0xffffffffu, static_cast<int>((static_cast<uint32_t>(slots) + kWarps - 1u - w0) / kWarps),
        0);
#pragma unroll 1
    for (int b = 0; b < steps; b += kBatch) {
      //: THE SLOTS' FACTS once a slot: lane `j` holds step `b + j`'s -- its row offset in `num` and its factor (a slot
      //: past the chunk formed at the warp's first, never used)
      const int sf = static_cast<int>(w0) + (b + lane) * kWarps;
      uint32_t idx, row;
      place(sf < slots ? sf : static_cast<int>(w0), idx, row);
      const uint32_t rp = row * dv;
      const float fs = factor_of(idx);
      const int end = min(steps - b, kBatch);
      //: A GROUP of four slots from `i`, as below
      const auto group = [&](int i, auto tail) {
        uint32_t v[kAhead], r[kAhead];
        float sc[kAhead];
        static_for<kAhead>([&](auto k) {
          asm volatile("ld.shared.u32 %0, [%1];"
                       : "=r"(v[k])
                       : "r"(cell0 + 512u * static_cast<uint32_t>(b + i + k)));
          r[k] = __shfl_sync(0xffffffffu, rp, i + k);
          sc[k] = __shfl_sync(0xffffffffu, fs, i + k);
        });
        static_for<kAhead>([&](auto k) {
          if (!decltype(tail)::value || decltype(k)::value == 0 || i + decltype(k)::value < end) {
            //: the magic off before the scale: an fp32 factor reaches 2^127
            const float x = __uint_as_float(ops::prmt(v[k], 0x4B400000u, half));
            red_add_f32(word_at(out, r[k]), (x - 12615680.0f) * sc[k]);
          }
        });
      };
      int i = 0;
#pragma unroll 1
      for (; i + kAhead <= end; i += kAhead) group(i, std::false_type{});
      if (i < end) group(i, std::true_type{});
    }
    return;
  }

  constexpr int kWords = kPanelRow / 4, kStride = kThreads / kWords, kAhead = 4;
  constexpr int kBatch = kWords;
  constexpr uint32_t kCellStride = kStride * kPanelRow;
  static_assert(kStride % 8 == 0, "a stride of slots keeps a cell's swizzle bits");
  static_assert(2 * kWords == 32 && kBatch % kAhead == 0,
                "a warp two slots a step, a half's facts its 16 lanes, a batch whole groups");
  static_assert(2 * (kWarps - 1) < kStride,
                "a warp's first slot under a stride: its step count from a positive");

  //: the thread, opaque, so its cell is formed at each drain rather than hoisted across the windows; the warp's slots
  //: its halves' `w0 + h + kStride i`
  const uint32_t ut = static_cast<uint32_t>(ops::opaque(static_cast<int>(threadIdx.x)));
  const int lane = static_cast<int>(ut % 32u), word = static_cast<int>(ut % kWords);
  const uint32_t s0 = ut / kWords, w0 = ut / 32u * 2u;
  const ops::SmemAddr cell0 =
      panel + (panel_row(s0) ^ ((32u * static_cast<uint32_t>(word >> 3)) | (4u * (word & 7))));
  const uint32_t pair = static_cast<uint32_t>(p.dv / 2);

  //: the write head's first query head's rows; a slot's row its query head `j` (`L` rows on) and token
  uint32_t* const out = p.num + head0 * pair + static_cast<uint32_t>(channel0) / 2u + word;
  //: the warp's steps (its first half's), from lane 0: warp-uniform to ptxas -- see docs/internals/carry/parts.md#drain
  const int steps = __shfl_sync(
      0xffffffffu, static_cast<int>((static_cast<uint32_t>(slots) + kStride - 1u - w0) / kStride),
      0);
#pragma unroll 1
  for (int b = 0; b < steps; b += kBatch) {
    //: THE SLOTS' FACTS once a slot: lane `16 h + j` holds half `h`'s step `b + j` -- its word offset in `num` and its
    //: factor; a slot past the chunk the warp's first slot's row at factor 0, so it adds exact zeros
    const int sf = static_cast<int>(w0) + lane / kBatch + (b + lane % kBatch) * kStride;
    uint32_t idx, row;
    place(sf < slots ? sf : static_cast<int>(w0), idx, row);
    const uint32_t rp = row * pair;
    const float f = factor_of(idx), fs = sf < slots ? f : 0.0f;
    const int end = min(steps - b, kBatch);
    //: A GROUP of four steps from `i`: their words and facts, then their adds; in a chunk's last group (`tail`) a step
    //: past the warp's last is not issued (a warp-uniform branch: the requests are the drain's floor)
    const auto group = [&](int i, auto tail) {
      uint32_t v[kAhead], r[kAhead];
      float sc[kAhead];
      static_for<kAhead>([&](auto k) {
        asm volatile("ld.shared.u32 %0, [%1];"
                     : "=r"(v[k])
                     : "r"(cell0 + static_cast<uint32_t>((b + i + k) * kCellStride)));
        r[k] = __shfl_sync(0xffffffffu, rp, i + k, kBatch);
        sc[k] = __shfl_sync(0xffffffffu, fs, i + k, kBatch);
      });
      static_for<kAhead>([&](auto k) {
        if (!decltype(tail)::value || decltype(k)::value == 0 || i + decltype(k)::value < end) {
          //: the f16 factor is at most 1, so the scaled magic is finite and one FFMA a half exact
          const float2 x = pair_magics(v[k]);
          const float bias = -12615680.0f * sc[k];
          const __half2 h = __floats2half2_rn(fmaf(x.x, sc[k], bias), fmaf(x.y, sc[k], bias));
          red_add_h2(word_at(out, r[k]), *reinterpret_cast<const uint32_t*>(&h));
        }
      });
    };
    int i = 0;
#pragma unroll 1
    for (; i + kAhead <= end; i += kAhead) group(i, std::false_type{});
    if (i < end) group(i, std::true_type{});
  }
}

}  // namespace rola::carry
