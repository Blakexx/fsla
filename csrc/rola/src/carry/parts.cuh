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

//: A CHUNK'S TOKEN-GROUP SPLIT, its slots `0 .. slots - 1`: the readout's group 0 below it, group 1 from it, a multiple
//: of 64 so a list's run splits where the prefix rows hold a boundary. -- see docs/internals/carry/readout.md#warps
__host__ __device__ __forceinline__ constexpr int group_split(int slots) {
  return slots <= kListBound ? slots
                             : kListBound * ((slots + 2 * kListBound - 1) / (2 * kListBound));
}

//: A CHUNK'S RUNS, lane `l` list `l`'s: its first entry of the chunk, its count, its count below the chunk's token-group
//: split, and its place in a buffer's entries (the lists' runs in list order), from its prefix rows modulo 2^16.
//: -- see docs/internals/carry/parts.md#entries
struct Run {
  int at, n, n0, to;
};

template <class P>
__device__ __forceinline__ Run chunk_runs(const CarryParams& p, const uint8_t* smem,
                                          const Window& win, int side, int w, int first, int slots,
                                          int total, int lane) {
  if (!win.listed || lane >= win.lists) return Run{0, 0, 0, 0};
  const int cols = cols_of(p, side);
  const int2 col = chunk_cols(first, slots, total, cols);
  //: the lane's two rows (lists `l`, `l + 1`): the chunk's end in both, its start where it is not the first, the split
  //: where it falls inside the chunk
  const uint16_t* const row =
      pre_of<P>(p, smem, side, w) + ((win.digit0 & (kBoxLists - 1)) + lane) * cols;
  const int a1 = row[col.y], b1 = row[cols + col.y];
  const bool start = col.x >= 0;
  const int a0 = start ? row[max(col.x, 0)] : 0, b0 = start ? row[cols + max(col.x, 0)] : 0;
  const int split = group_split(slots), cm = (first + split) / kListBound - 1;
  const int n = ((b1 - a1) - (b0 - a0)) & 0xffff;
  const int n0 = split >= slots ? n : ((row[cols + cm] - row[cm]) - (b0 - a0)) & 0xffff;
  return Run{(b0 - a0) & 0xffff, n, n0, (a1 - a0) & 0xffff};
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

//: A CHUNK'S BUFFER, laid out at its own slots: a read buffer's rows, its num panel (a slot's channels at the reduce
//: dtype: `panel_row_bytes`), its den words (`kDenWords` fp32 a slot, one a readout channel group), its entries; a fold
//: buffer's rows, its entries. -- see docs/internals/carry/schedule.md#buffers
__host__ __device__ __forceinline__ constexpr int panel_row_bytes(int reduce) {
  return reduce == kReduceF32 ? kBoxChannels * 4 : kBoxChannels * 2;
}

constexpr int kDenWords = kBoxChannels / 8, kDenRowBytes = kDenWords * 4;

template <class P>
__host__ __device__ constexpr int read_panel(int slots) {
  return align_of(slots * P::kReadRow, 128);
}

template <class P>
__host__ __device__ constexpr int read_dens(int slots, int reduce) {
  return read_panel<P>(slots) + align_of(slots * panel_row_bytes(reduce), 128);
}

template <class P>
__host__ __device__ constexpr int read_ents(int slots, int reduce) {
  return read_dens<P>(slots, reduce) + align_of(slots * kDenRowBytes, 128);
}

template <class P>
__host__ __device__ constexpr int fold_ents(int slots) {
  return align_of(slots * P::kWriteRow, 128);
}

//: A READ CHUNK'S FILL, the copies issued: its rows at `buf`, its panel and den words cleared, its entries (a warp its
//: own digits' runs, `run` the chunk's). -- see docs/internals/carry/parts.md#fill
template <class P>
__device__ __forceinline__ void read_fill(const CarryParams& p, const uint8_t* smem,
                                          const Window& rw, const Place& c, int w, const Run& run,
                                          int first, int slots, ops::SmemAddr buf, int warp,
                                          int lane) {
  const volatile Addr& addr = *reinterpret_cast<const volatile Addr*>(smem + P::kAddr);
  fill<P::kReadFactors, false>(addr.slices[kRead], nullptr, 0, union_of<P>(p, smem, kRead, w), rw,
                               first, slots, buf);
  //: the panel and the den words cleared by stores, four a thread a trip, the first straight-line
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
  clear(read_panel<P>(slots), read_ents<P>(slots, p.reduce), 0u);
  if (rw.listed)
    entries_fill<P>(p, p.read_lists, rw, c, w, run,
                    buf + static_cast<uint32_t>(read_ents<P>(slots, p.reduce)), warp, lane);
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

//: digit `d`'s place in `kExps` (lane order: `16 M + 8 H + g` at `4 g + 2 M + H`)
__host__ __device__ __forceinline__ constexpr int exp_slot(int d) { return 4 * (d & 7) + (d >> 3); }

//: THE DIGIT KEYS, the hand-off's first step: a digit's largest |value| at its list's unit as an integer key -- its
//: fp32 bits plus (the unit less the box's top) x 2^23, a zero `INT_MIN` -- at window `w`'s parity at the digit's place
//: (`exp_slot`); `post_keys` returns the box's top. -- see docs/internals/carry/parts.md#maxima
//: THE ALTERNATING FORM: a digit is a fold row, across every warp's lists -- the bits joined across the row's four
//: lanes first (a list's lanes share its unit), lane `t` keying its list `t`'s leaf; a digit's key their max, one
//: shared atomic max a lane.
template <class P>
__device__ __forceinline__ void post_row_keys(const State<P>& st, uint8_t* smem, int w, int top,
                                              int warp, int lane) {
  static_assert(P::kFoldMTiles * 2 == 4, "a lane's four digit rows, one a lane of its quad");
  static_assert(P::kFoldLists == 4, "a quad's lane `t` keys its list `t`'s leaves");
  const int t = lane & 3;
  const volatile UnitBook* const book = reinterpret_cast<const volatile UnitBook*>(smem + P::kBook);
  const int shift = max(book[warp + kWarps * t].unit - top, -kKeySpan) * (1 << 23);
  int key[2][2];
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
    });
  });

  int* const keys = reinterpret_cast<int*>(smem + P::kKeys) + (w & 1) * kBoxDigits;
  const int mine = t & 2 ? (t & 1 ? key[1][1] : key[1][0]) : (t & 1 ? key[0][1] : key[0][0]);
  atomicMax(keys + lane, mine);
}

//: THE SAME SPLIT'S KEYS: a digit is a fold list, every row of it this warp's -- its key the list's largest |value|
//: over the warp at its unit, lane `j` storing list `warp + 8 j`'s at its place (`exp_slot`), the one writer.
template <class P>
__device__ __forceinline__ void post_list_keys(const State<P>& st, uint8_t* smem, int w, int top,
                                               int warp, int lane) {
  const volatile UnitBook* const book = reinterpret_cast<const volatile UnitBook*>(smem + P::kBook);
  int mine = INT_MIN;
  static_for<P::kFoldLists>([&](auto J) {
    float x = 0.0f;
    static_for<P::kFoldMTiles>([&](auto M) {
      static_for<4>(
          [&](auto N) { static_for<4>([&](auto E) { x = fmaxf(x, fabsf(st.acc[J][M][N][E])); }); });
    });
    const int bits = __reduce_max_sync(0xffffffffu, __float_as_int(x));
    const int shift = max(book[warp + kWarps * J].unit - top, -kKeySpan) * (1 << 23);
    mine = lane == J ? (bits == 0 ? INT_MIN : bits + shift) : mine;
  });
  const ops::SmemAddr keys =
      ops::smem_addr(smem + P::kKeys) + 4u * static_cast<uint32_t>((w & 1) * kBoxDigits);
  ops::store_shared_u32_if(keys + 4u * static_cast<uint32_t>(exp_slot(warp + kWarps * (lane & 3))),
                           static_cast<uint32_t>(mine), lane < P::kFoldLists);
}

template <class P>
__device__ __forceinline__ int post_keys(const State<P>& st, uint8_t* smem, int w, int warp,
                                         int lane) {
  const int top = box_top<P>(smem, lane);
  if constexpr (P::kSameSplit)
    post_list_keys<P>(st, smem, w, top, warp, lane);
  else
    post_row_keys<P>(st, smem, w, top, warp, lane);
  return top;
}

//: THE IMAGE'S TOP: a digit's largest value placed under 2^10, so a 32-list sum fits an fp16 accumulator.
//: -- see docs/internals/carry/parts.md#maxima
constexpr int kImageTop = 10;

//: THE IMAGE, the value-carve readout's: digit `d`'s plane a slice a readout channel group `cg` -- 32 lists x 8 channels
//: f16, list `l`'s row 16 bytes at `l ^ (d & 7)`, so the hand-off's 4-byte stores and the readout's `ldmatrix.trans`
//: both meet 32 banks. -- see docs/internals/carry/parts.md#image
constexpr int kSliceBlock = kBoxLists * 16;

__host__ __device__ __forceinline__ constexpr uint32_t slice_row(int cg, int l, int d) {
  return static_cast<uint32_t>(cg * kSliceBlock + 16 * (l ^ (d & 7)));
}

//: THE DEN IMAGE: digit `d`'s 8-digit block, 32 lists x 8 digits f16, list `l`'s row 16 bytes; the digit at `2 (d & 7)`
//: in it. -- see docs/internals/carry/parts.md#den
__host__ __device__ __forceinline__ constexpr uint32_t den_row(int l, int d) {
  return static_cast<uint32_t>((d >> 3) * kSliceBlock + 16 * l);
}

//: THE DIGITS' EXPONENTS from the keys at window `w`'s parity: lane `i`'s digit's (lane order, `exp_slot`), `(key +
//: 0x7fffff) >> 23` the key's exponent field rounded up past a nonzero mantissa, plus the box's top less 127, at least
//: `kImageTop - 127` -- the smallest `e` with the digit's largest |state| at most `2^e` -- recorded in the WARP'S OWN
//: ROW of `kExps` (`exps_of`) for its stores and its readout: every warp the same values, none reading another's.
//: Warp 0 resets the other parity's keys for the next window. -- see docs/internals/carry/parts.md#maxima
template <class P>
__device__ __forceinline__ ops::SmemAddr exps_of(const uint8_t* smem, int warp) {
  return ops::smem_addr(smem + P::kExps) + 4u * static_cast<uint32_t>(kBoxDigits * warp);
}

template <class P>
__device__ __forceinline__ void post_exps(uint8_t* smem, int w, int top, int warp, int lane) {
  static_assert(kBoxDigits == 32, "lane `i` digit `i`, a parity's keys one bit apart");
  int* const keys = reinterpret_cast<int*>(smem + P::kKeys);
  const int at = (w & 1) * kBoxDigits + lane;
  const int key = keys[at];
  if (warp == 0) keys[at ^ kBoxDigits] = INT_MIN;
  reinterpret_cast<int*>(smem + P::kExps)[kBoxDigits * warp + lane] =
      max(((key + 0x7fffff) >> 23) + top - 127, kImageTop - 127);
  __syncwarp();
}

//: THE DEN'S TABLES, a window's, from the den's registers (the state before the window's writes): each digit's largest
//: by an integer max (`den_tables_max`), then each digit's factor and the den image (`den_tables_out`, past a barrier).
//: -- see docs/internals/carry/parts.md#den
//: A DIGIT'S DEN EXPONENT `dE` from its largest den (`kDenMax`), `kImageTop - 127` where it has none.
__device__ __forceinline__ int den_exp(float top) {
  return top > 0.0f ? ceil_exp(top) : kImageTop - 127;
}

//: A LANE'S DEN WORDS' PLACES, each of the box's 32 x 32 once across the CTA, the list owner's on either side (word `i`'s
//: fold list `w + 8 (2 (t & 1) + (i & 1))`, its fold row `digit` `g + 8 (i >> 1) + 16 (t >> 1)`).
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

//: A DEN WORD'S READOUT DIGIT and its K, `I` compile-time: its fold row and list (the alternating form), its list and
//: row (the same split); `den_word_scale` the one of the lane's two digits word `I` is at.
template <class P, int I>
__device__ __forceinline__ int den_digit(const DenPlace& at) {
  return P::kSameSplit ? at.list(I) : at.digit(I);
}

template <class P, int I>
__device__ __forceinline__ int den_k(const DenPlace& at) {
  return P::kSameSplit ? at.digit(I) : at.list(I);
}

template <class P>
__host__ __device__ constexpr int den_word_scale(int i) {
  return P::kSameSplit ? (i & 1) : (i >> 1);
}

//: AN INTEGER MAX INTO A SHARED WORD, fire and forget (a non-negative float's bits order as integers).
__device__ __forceinline__ void red_max_u32(ops::SmemAddr at, uint32_t v) {
  asm volatile("red.shared.max.u32 [%0], %1;" ::"r"(at), "r"(v) : "memory");
}

template <class P>
__device__ __forceinline__ void den_tables_max(uint8_t* smem, const State<P>& st, int warp,
                                               int lane) {
  const DenPlace at = den_place<P>(warp, lane);
  const ops::SmemAddr tops = ops::smem_addr(smem + P::kDenMax);
  static_for<4>([&](auto I) {
    red_max_u32(tops + 4u * static_cast<uint32_t>(den_digit<P, decltype(I)::value>(at)),
                __float_as_uint(st.den[I]));
  });
}

template <class P>
__device__ __forceinline__ void den_tables_out(uint8_t* smem, const State<P>& st, int warp,
                                               int lane) {
  const DenPlace at = den_place<P>(warp, lane);
  const float* const tops = reinterpret_cast<const float*>(smem + P::kDenMax);
  //: a word at its digit's scale, at its place in the den image (`den_row`); a lane's four words two digits'
  float scale[2];
  static_for<2>([&](auto B) {
    constexpr int I = P::kSameSplit ? decltype(B)::value : 2 * decltype(B)::value;
    scale[B] = pow2(kImageTop - den_exp(tops[den_digit<P, I>(at)]));
  });
  static_for<4>([&](auto I) {
    constexpr int i = decltype(I)::value;
    const int d = den_digit<P, i>(at);
    *reinterpret_cast<__half*>(smem + P::kDenImage + den_row(den_k<P, i>(at), d) + 2 * (d & 7)) =
        __float2half_rn(st.den[I] * scale[den_word_scale<P>(I)]);
  });
  ops::store_shared_f32_if(ops::smem_addr(smem + P::kDenFactor) + 4u * static_cast<uint32_t>(lane),
                           pow2_wide(den_exp(tops[lane]) - kImageTop), warp == 0);
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

//: A LANE'S HAND-OFF OPERANDS: its base (opaque, so its terms are formed at each call rather than hoisted across the
//: windows), the four exponents at `16 i` of its warp's row (`exp_slot`'s lane order), the bits `list_scale` takes,
//: and its lists' units at the image's top.
struct StoreFrame {
  uint32_t at;
  int4 ex;
  uint2 bits;
};

template <class P>
__device__ __forceinline__ StoreFrame store_frame(const uint8_t* smem, uint32_t base, int i,
                                                  int (&a)[P::kFoldLists], int warp) {
  StoreFrame f;
  asm volatile("mov.u32 %0, %1;" : "=r"(f.at) : "r"(base));
  asm volatile("ld.shared.v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(f.ex.x), "=r"(f.ex.y), "=r"(f.ex.z), "=r"(f.ex.w)
               : "r"(exps_of<P>(smem, warp) + 16u * static_cast<uint32_t>(i))
               : "memory");
  const volatile uint32_t* const bp = reinterpret_cast<const volatile uint32_t*>(smem + P::kBits);
  f.bits = make_uint2(bp[0], bp[1]);
  list_units<P>(a, smem, warp);
  static_for<P::kFoldLists>([&](auto J) { a[J] += kImageTop; });
  return f;
}

//: A BATCH'S STORES, the hand-off: the warp's state of batches `lo .. hi - 1` (8 digits each) into the image at `image`
//: (digit `d`'s plane `d & Mask`), f16 at each digit's power of two, each list moved there from its unit by
//: `list_scale`; a lane's channel pair of a slice row one 4-byte store at its held base and a compile-time offset.
//: -- see docs/internals/carry/parts.md#store
//: THE ALTERNATING FORM: a digit is a fold row, batch `2 M + H` its rows `16 M + 8 H + g`, list `warp + 8 J` the
//: digit's K row `(warp ^ g) + 8 J` (digit `g`'s swizzle).
template <class P, uint32_t Mask>
__device__ __forceinline__ void store_row_batches(const State<P>& st, uint8_t* smem,
                                                  ops::SmemAddr image, int lo, int hi, int warp,
                                                  int lane) {
  const int g = lane >> 2, t = lane & 3;
  int a[P::kFoldLists];
  const StoreFrame fr = store_frame<P>(smem,
                                       image + static_cast<uint32_t>(g) * kPlaneBytes
                                           + slice_row(0, warp, g) + 4u * static_cast<uint32_t>(t),
                                       g, a, warp);
  const uint32_t at = fr.at;
  const uint2 bits = fr.bits;
  const int es[2][2] = {{fr.ex.x, fr.ex.y}, {fr.ex.z, fr.ex.w}};
  static_for<2>([&](auto M) {
    static_for<2>([&](auto H) {
      constexpr int b = 2 * M + H;
      if (b >= lo && b < hi) {
        static_for<P::kFoldLists>([&](auto J) {
          const float scale = list_scale(a[J], es[M][H], bits);
          static_for<4>([&](auto N) {
            const __half2 h = __floats2half2_rn(st.acc[J][M][N][2 * H] * scale,
                                                st.acc[J][M][N][2 * H + 1] * scale);
            const uint32_t to =
                at + ((16u * M + 8u * H) & Mask) * kPlaneBytes + kSliceBlock * N + 128u * J;
            ops::store_shared_u32(to, *reinterpret_cast<const uint32_t*>(&h));
          });
        });
      }
    });
  });
}

//: THE SAME SPLIT: a digit is a fold list, batch `J` list `warp + 8 J`, its rows the digit's K -- row `16 M + 8 H + g`
//: at `16 M + 8 H + (g ^ warp)` (the list's swizzle, `warp`).
template <class P, uint32_t Mask>
__device__ __forceinline__ void store_list_batches(const State<P>& st, uint8_t* smem,
                                                   ops::SmemAddr image, int lo, int hi, int warp,
                                                   int lane) {
  const int g = lane >> 2, t = lane & 3;
  int a[P::kFoldLists];
  const StoreFrame fr = store_frame<P>(smem,
                                       image + static_cast<uint32_t>(warp) * kPlaneBytes
                                           + slice_row(0, g, warp) + 4u * static_cast<uint32_t>(t),
                                       warp, a, warp);
  const int es[4] = {fr.ex.x, fr.ex.y, fr.ex.z, fr.ex.w};
  static_for<P::kFoldLists>([&](auto J) {
    if (J >= lo && J < hi) {
      const float scale = list_scale(a[J], es[J], fr.bits);
      static_for<P::kFoldMTiles>([&](auto M) {
        static_for<2>([&](auto H) {
          static_for<4>([&](auto N) {
            const __half2 h = __floats2half2_rn(st.acc[J][M][N][2 * H] * scale,
                                                st.acc[J][M][N][2 * H + 1] * scale);
            const uint32_t to = fr.at + ((8u * J) & Mask) * kPlaneBytes + kSliceBlock * N
                                + 16u * (16u * M + 8u * H);
            ops::store_shared_u32(to, *reinterpret_cast<const uint32_t*>(&h));
          });
        });
      });
    }
  });
}

template <class P, uint32_t Mask>
__device__ __forceinline__ void store_batches(const State<P>& st, uint8_t* smem,
                                              ops::SmemAddr image, int lo, int hi, int warp,
                                              int lane) {
  if constexpr (P::kSameSplit)
    store_list_batches<P, Mask>(st, smem, image, lo, hi, warp, lane);
  else
    store_row_batches<P, Mask>(st, smem, image, lo, hi, warp, lane);
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

//: A WORD `off` words past `base` (`off` under 2^32), its own add: never re-associated into the base's sum.
template <class T>
__device__ __forceinline__ T* word_at(T* base, uint32_t off) {
  static_assert(sizeof(T) == 4, "a 32-bit word");
  uint64_t a;
  asm("mad.wide.u32 %0, %1, 4, %2;" : "=l"(a) : "r"(off), "l"(reinterpret_cast<uint64_t>(base)));
  return reinterpret_cast<T*>(a);
}

}  // namespace rola::carry
