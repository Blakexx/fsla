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

//: the smallest `e` with `x <= 2^e`, `kUnitNone` for zero
__device__ __forceinline__ int ceil_exp(float x) {
  const uint32_t bits = __float_as_uint(x);
  return bits == 0u ? kUnitNone : static_cast<int>(bits >> 23) - 127 + ((bits & 0x7fffffu) != 0u);
}

//: A WARP'S STATE, the fold's accumulators: per list, its digits' m-tiles by the channels' four n-tiles, fp32 in the
//: list's unit. -- see docs/internals/carry/fold.md#state
template <class P>
struct State {
  float acc[P::kFoldLists][P::kFoldMTiles][4][4];

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

static_assert(sizeof(Addr) <= Tenants::kAddrBytes, "the addresses fit their tenant");

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

//: the prefix rows' value at row `l`, column `c` (column -1 none) of a row's `cols`, modulo 2^16
__device__ __forceinline__ int pre_at(const uint16_t* pre, int l, int c, int cols) {
  return c < 0 ? 0 : static_cast<int>(pre[l * cols + c]);
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

//: FACTS IN, a window ahead: the box's write lists' exponent pairs and each of the read side's query heads' tokens'
//: readout units (query head `j` of write head `bh` the call's `bh g + j`).
template <class P>
__device__ __forceinline__ void facts_in(const CarryParams& p, const Window& ww, const Place& c,
                                         int w, uint8_t* smem) {
  constexpr int kUnitCopies = kWindow / ops::kStageBytes,
                kExpCopies = kBoxLists * 4 / ops::kStageBytes;
  const int heads = p.tables.heads;
  const ops::SmemAddr units = ops::smem_addr(smem + p.tables.units_at) + (w & 1) * heads * kWindow;
#pragma unroll 1
  for (int k = ops::opaque(static_cast<int>(threadIdx.x)); k < heads * kUnitCopies + kExpCopies;
       k += kThreads) {
    if (k < heads * kUnitCopies) {
      const int j = k / kUnitCopies;
      ops::stage_tile(units + 16 * k, p.unit
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

//: A CHUNK'S ROWS: `slots` union slots from `first`, each its token's row -- its side's operand slices at 64 bytes
//: each, then (`Values`) its values at the box's channels -- swizzled, by `cp.async`: a quad of lanes a slot, lane `r`
//: of the quad chunk `r` of each slice and of the values, the slot's token, sources and row formed once for them; a
//: narrow slice (8 or 4 bytes) its quad's first lane's; a gap slot's row zero-filled.
//: -- see docs/internals/carry/parts.md#fill
template <int Factors, bool Values>
__device__ __forceinline__ void fill(const volatile Slices& sl, const uint8_t* values,
                                     int values_st, const uint16_t* tok, const Window& win,
                                     int first, int slots, ops::SmemAddr pool) {
  constexpr int kRow = kSliceBytes * Factors + (Values ? kBoxChannels * 2 : 0);
  constexpr int kQuads = kThreads / 4, kChunks = kSliceBytes / ops::kStageBytes;
  //: the thread opaque, so its quad lane and first slot are formed at each fill, never held across the windows
  const int tid = ops::opaque(static_cast<int>(threadIdx.x)), r = tid & 3;
  const uint8_t* base[Factors];
  int st[Factors], sh[Factors], bytes[Factors];
  bool active[Factors], narrow = false;
  static_for<Factors>([&](auto i) {
    bytes[i] = sl.bytes[i];
    st[i] = sl.st[i];
    sh[i] = Values ? 0 : sl.sh[i];
    base[i] = sl.at[i] + static_cast<int64_t>(win.t0) * st[i] + ops::kStageBytes * r;
    active[i] = ops::kStageBytes * r < bytes[i];
    narrow |= bytes[i] < ops::kStageBytes;
  });

  const uint8_t* const vbase =
      Values ? values + static_cast<int64_t>(win.t0) * values_st + ops::kStageBytes * r : nullptr;
  //: a slice narrower than a chunk (8 or 4 bytes) its quad's first lane's whole run; else each lane its chunk, a chunk
  //: past a slice's bytes zero-filled into the slice's unused space
  const auto rows = [&](auto kNarrow) {
#pragma unroll 2
    for (int s = tid >> 2; s < slots; s += kQuads) {
      //: a slot's row: its query head `j` (the read side's) and token `t`, bit 15 a gap
      const int u = win.listed ? tok[first + s] : first + s, t = u & (kWindow - 1),
                j = Values ? 0 : (u >> 9) & 0x3f;
      const bool live = (u & 0x8000) == 0;
      const ops::SmemAddr row = pool + static_cast<uint32_t>(s * kRow);
      static_for<Factors>([&](auto i) {
        const ops::SmemAddr dst = row
                                  + 16u
                                        * chunk_at<kRow>(static_cast<uint32_t>(s),
                                                         static_cast<uint32_t>(kChunks * i + r));
        const uint8_t* const src = base[i] + static_cast<uint32_t>(t * st[i] + j * sh[i]);
        if constexpr (decltype(kNarrow)::value) {
          if (!active[i] && r == 0) stage_bytes(dst, src, bytes[i], live);
          if (active[i]) ops::stage_run_if<16>(dst, src, live);
        } else {
          ops::stage_run_if<16>(dst, src, live && active[i]);
        }
      });
      if constexpr (Values)
        ops::stage_run_if<16>(
            row
                + 16u
                      * chunk_at<kRow>(static_cast<uint32_t>(s),
                                       static_cast<uint32_t>(kChunks * Factors + r)),
            vbase + static_cast<uint32_t>(t * values_st), live);
    }
  };
  if (narrow) {
    rows(std::true_type{});
  } else {
    rows(std::false_type{});
  }
}

//: A CHUNK'S RUNS, lane `l` list `l`'s: its first entry of the chunk, its count, and its place in a buffer's entries
//: (the lists' runs in list order), four loads from the prefix rows, each a difference within the chunk modulo 2^16.
//: -- see docs/internals/carry/parts.md#entries
struct Run {
  int at, n, to;
};

template <class P>
__device__ __forceinline__ Run chunk_runs(const CarryParams& p, const uint8_t* smem,
                                          const Window& win, int side, int w, int first, int slots,
                                          int total, int lane) {
  if (!win.listed || lane >= win.lists) return Run{0, 0, 0};
  const uint16_t* pre = pre_of<P>(p, smem, side, w);
  const int cols = cols_of(p, side);
  const int2 col = chunk_cols(first, slots, total, cols);
  const int l = (win.digit0 & (kBoxLists - 1)) + lane;
  const int a0 = pre_at(pre, l, col.x, cols), a1 = pre_at(pre, l, col.y, cols);
  const int b0 = pre_at(pre, l + 1, col.x, cols), b1 = pre_at(pre, l + 1, col.y, cols);
  return Run{(b0 - a0) & 0xffff, ((b1 - a1) - (b0 - a0)) & 0xffff, (a1 - a0) & 0xffff};
}

//: A CHUNK'S ENTRIES, a warp's share: its own lists' runs (`warp + 8 j`) to their places at `dst`, 8 bytes a lane, the
//: four runs one stream -- entry `e` in the last run whose first it passes, its source and place offsets from `e` --
//: every full trip of 32 unpredicated, the stream's tail one branch a warp. -- see docs/internals/carry/parts.md#entries
template <class P>
__device__ __forceinline__ void entries_fill(const CarryParams& p, const SideLists& l,
                                             const Window& win, const Place& c, int w,
                                             const Run& run, ops::SmemAddr dst, int warp,
                                             int lane) {
  constexpr int kRuns = kBoxLists / kWarps;
  //: the box's first list's row, once: a list's is `kListCap` entries a query head on (opaque, so the lists' sources
  //: are formed at each fill rather than held across the windows)
  const int cap = kListCap * ops::opaque(l.heads);
  const uint2* const lists =
      l.entries + list_row(l, c, p.windows, ops::opaque(w), win.digit0) * cap;
  int first[kRuns], src[kRuns], to[kRuns], total = 0;
  static_for<kRuns>([&](auto J) {
    const int list = warp + kWarps * J;
    const int at = __shfl_sync(0xffffffffu, run.at, list),
              n = __shfl_sync(0xffffffffu, run.n, list),
              place = __shfl_sync(0xffffffffu, run.to, list);
    first[J] = total;
    src[J] = list * cap + at - total;
    to[J] = place - total;
    total += n;
  });
  const auto copy = [&](int e) {
    int s = src[0], d = to[0];
    static_for<kRuns - 1>([&](auto J) {
      const bool past = e >= first[J + 1];
      s = past ? src[J + 1] : s;
      d = past ? to[J + 1] : d;
    });
    ops::stage_run<8>(dst + 8u * static_cast<uint32_t>(d + e), lists + (s + e));
  };
  int k = 0;
#pragma unroll 1
  for (; k + 32 <= total; k += 32) copy(k + lane);
  if (lane < total - k) copy(k + lane);
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

//: A READ CHUNK'S FILL, the copies issued: its rows at `buf`, its dummy panel rows and panel zeroed, its entries (a
//: warp its own digits' runs). -- see docs/internals/carry/parts.md#fill
template <class P>
__device__ __forceinline__ void read_fill(const CarryParams& p, const uint8_t* smem,
                                          const Window& rw, const Place& c, int w, int first,
                                          int slots, int total, ops::SmemAddr buf, int warp,
                                          int lane) {
  const volatile Addr& addr = *reinterpret_cast<const volatile Addr*>(smem + P::kAddr);
  fill<P::kReadFactors, false>(addr.slices[kRead], nullptr, 0, union_of<P>(p, smem, kRead, w), rw,
                               first, slots, buf);
  const ops::SmemAddr pads = buf + static_cast<uint32_t>(read_pads<P>(slots));
  //: a zero quantum reads nothing: its source is any global address
#pragma unroll 4
  for (int k = ops::opaque(static_cast<int>(threadIdx.x));
       k < (read_ents<P>(slots) - read_pads<P>(slots)) / 16; k += kThreads)
    ops::stage_tile_if(pads + 16u * static_cast<uint32_t>(k), p.num, false);
  if (rw.listed)
    entries_fill<P>(p, p.read_lists, rw, c, w,
                    chunk_runs<P>(p, smem, rw, kRead, w, first, slots, total, lane),
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

//: THE DIGIT KEYS, the hand-off's first step: a list's largest |value| a digit row at its unit as an integer key --
//: its fp32 bits plus (its unit less the box's top) x 2^23, a zero `INT_MIN` -- one integer max a list, the row's four
//: lanes joined by two shuffles, one shared atomic max a lane, at window `w`'s parity in lane order (digit `16 M + 8 H
//: + g` at `4 g + 2 M + H`); returns the box's top. -- see docs/internals/carry/parts.md#maxima
template <class P>
__device__ __forceinline__ int post_keys(const State<P>& st, uint8_t* smem, int w, int warp,
                                         int lane) {
  static_assert(P::kFoldMTiles * 2 == 4, "a lane's four digit rows, one a lane of its quad");
  int u[P::kFoldLists];
  list_units<P>(u, smem, warp);
  const int top = box_top<P>(smem, lane);
  int shift[P::kFoldLists];
  static_for<P::kFoldLists>([&](auto J) { shift[J] = max(u[J] - top, -kKeySpan) * (1 << 23); });
  int key[2][2];
  static_for<2>([&](auto M) {
    static_for<2>([&](auto H) {
      int k = INT_MIN;
      static_for<P::kFoldLists>([&](auto J) {
        float x = fmaxf(fabsf(st.acc[J][M][0][2 * H]), fabsf(st.acc[J][M][0][2 * H + 1]));
        static_for<3>([&](auto N) {
          x = fmaxf(
              x, fmaxf(fabsf(st.acc[J][M][N + 1][2 * H]), fabsf(st.acc[J][M][N + 1][2 * H + 1])));
        });
        const int bits = __float_as_int(x);
        k = max(k, bits == 0 ? INT_MIN : bits + shift[J]);
      });
      k = max(k, __shfl_xor_sync(0xffffffffu, k, 1));
      key[M][H] = max(k, __shfl_xor_sync(0xffffffffu, k, 2));
    });
  });
  const int t = lane & 3;
  const int mine = t & 2 ? (t & 1 ? key[1][1] : key[1][0]) : (t & 1 ? key[0][1] : key[0][0]);
  atomicMax(reinterpret_cast<int*>(smem + P::kKeys) + (w & 1) * kBoxDigits + lane, mine);
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
//: the readers; the other parity's keys reset for the next window. -- see docs/internals/carry/parts.md#maxima
template <class P>
__device__ __forceinline__ void post_exps(uint8_t* smem, int w, int top, int lane) {
  int* const keys = reinterpret_cast<int*>(smem + P::kKeys);
  const int key = keys[(w & 1) * kBoxDigits + lane];
  keys[((w + 1) & 1) * kBoxDigits + lane] = INT_MIN;
  reinterpret_cast<int*>(smem + P::kExps)[lane] =
      max(((key + 0x7fffff) >> 23) + top - 127, kImageTop - 127);
  __syncwarp();
}

//: digit `d`'s place in `kExps` (lane order: `16 M + 8 H + g` at `4 g + 2 M + H`)
__host__ __device__ __forceinline__ constexpr int exp_slot(int d) { return 4 * (d & 7) + (d >> 3); }

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

//: A LANE'S IMAGE ADDRESSES for a digit's state fragments (the A operand: 32 channels x 32 lists, `ldmatrix.trans`):
//: its units at digit 0, each XORed with a digit's.
struct ImageLane {
  uint32_t u[4];
};

__device__ __forceinline__ ImageLane image_lane(int lane) {
  ImageLane out;
#pragma unroll
  for (int k = 0; k < 4; ++k)
    out.u[k] =
        16u
        * unit_of(8 * (lane >> 4) + (lane & 7) + 16 * (k & 1), 2 * (k >> 1) + ((lane >> 3) & 1), 0);
  return out;
}

//: A DIGIT'S STATE FRAGMENTS from its image plane at `plane`: `(m * 2 + kb) * 4 + i`, m the channel tile, kb the
//: list half.
__device__ __forceinline__ void state_in(uint32_t (&f)[16], ops::SmemAddr plane,
                                         const ImageLane& il, int d) {
  const uint32_t x = 16u * digit_swizzle(d);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    uint32_t a[4];
    ops::load_frag_t(a, plane + (il.u[k] ^ x));
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
  uint32_t b[4], e0, e1;
  float c0, c1;
};

template <class P>
__device__ __forceinline__ TileIn tile_in(ops::SmemAddr pool, ops::SmemAddr units, int sx, uint2 e,
                                          int lane) {
  const int t = lane & 3;
  TileIn in;
  int u;
  asm volatile("ld.shared.s8 %0, [%1];" : "=r"(u) : "r"(units + (e.x >> 16)));
  const float c = __half2float(__ushort_as_half(static_cast<unsigned short>(e.y)));
  const float cu = __uint_as_float(__float_as_uint(c) + (static_cast<uint32_t>(sx - u) << 23));
  in.e0 = __shfl_sync(0xffffffffu, e.y, 2 * t);
  in.e1 = __shfl_sync(0xffffffffu, e.y, 2 * t + 1);
  in.c0 = __shfl_sync(0xffffffffu, cu, 2 * t);
  in.c1 = __shfl_sync(0xffffffffu, cu, 2 * t + 1);
  ops::load_frag(in.b, chunk_addr<P::kReadRow>(pool + 16u * (e.x & 0xffffu),
                                               static_cast<uint32_t>(lane >> 3)));
  return in;
}

//: @burst
template <bool F16>
__device__ __forceinline__ void tile_mma(float (&acc)[2][4], const uint32_t (&b)[4],
                                         const uint32_t (&f)[16]) {
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    uint32_t h[2] = {0u, 0u};
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[m][i] = 0.0f;
#pragma unroll
    for (int kb = 0; kb < 2; ++kb) {
      const uint32_t a[4] = {f[(m * 2 + kb) * 4], f[(m * 2 + kb) * 4 + 1], f[(m * 2 + kb) * 4 + 2],
                             f[(m * 2 + kb) * 4 + 3]};
      if constexpr (F16) {
        ops::hmma_f16(h, a, b[2 * kb], b[2 * kb + 1]);
      } else {
        ops::hmma(acc[m], a, b[2 * kb], b[2 * kb + 1]);
      }
    }
    if constexpr (F16) {
      const float2 g0 = __half22float2(*reinterpret_cast<const __half2*>(&h[0])),
                   g8 = __half22float2(*reinterpret_cast<const __half2*>(&h[1]));
      acc[m][0] = g0.x;
      acc[m][1] = g0.y;
      acc[m][2] = g8.x;
      acc[m][3] = g8.y;
    }
  }
}

__device__ __forceinline__ void tile_add(ops::SmemAddr pads, const TileIn& in,
                                         const float (&acc)[2][4], int lane) {
  const int g = lane >> 2;
  const uint32_t b0 = pads + (in.e0 >> 16), b1 = pads + (in.e1 >> 16);
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    panel_red((b0 ^ (32u * m)) + 4u * g, s16_pair(acc[m][0], acc[m][2], in.c0));
    panel_red((b1 ^ (32u * m)) + 4u * g, s16_pair(acc[m][1], acc[m][3], in.c1));
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
    const float cu = __uint_as_float(__float_as_uint(c) + (static_cast<uint32_t>(sx[Q] - u) << 23));
    in.c0[Q] = __shfl_sync(0xffffffffu, cu, 2 * t);
    in.c1[Q] = __shfl_sync(0xffffffffu, cu, 2 * t + 1);
  });
  ops::load_frag(in.b, chunk_addr<P::kReadRow>(row0, static_cast<uint32_t>(lane >> 3)));
  return in;
}

template <int M>
__device__ __forceinline__ void tile_add_m(ops::SmemAddr pads, const TileInM<M>& in,
                                           const float (&acc)[M][2][4], int lane) {
  const int g = lane >> 2;
  const uint32_t b0 = pads + (in.e0 >> 16), b1 = pads + (in.e1 >> 16);
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    uint32_t w0 = 0u, w1 = 0u;
    static_for<M>([&](auto Q) {
      w0 += s16_pair(acc[Q][m][0], acc[Q][m][2], in.c0[Q]);
      w1 += s16_pair(acc[Q][m][1], acc[Q][m][3], in.c1[Q]);
    });
    panel_red((b0 ^ (32u * m)) + 4u * g, w0);
    panel_red((b1 ^ (32u * m)) + 4u * g, w1);
  }
}

//: A MASKED LANE'S ENTRY in a chunk's last step: its place `c`'s row of the chunk, coefficient 0, a dummy panel row, so
//: its products add nothing -- the pool and panel offsets by the chunk's first slot `first`, as the entries' are.
template <class P>
__device__ __forceinline__ uint2 masked_entry(int first, uint32_t c) {
  const uint32_t row = static_cast<uint32_t>(first) * (P::kReadRow / 16) + c * (P::kReadRow / 16)
                       + chunk_at<P::kReadRow>(c, 0);
  return make_uint2(row, (64u * static_cast<uint32_t>(first) + panel_row(c)) << 16);
}

//: A CHUNK'S READOUT over batches `lo .. hi - 1`, a warp's: each of its digits `8 i + warp`, its state from its image
//: plane (`d & mask` at `image`) and its exponent, its tiles over the chunk -- a listed side's run of the digit in the
//: buffer's entries at `ents`, a lane past it masked; else every slot of the chunk, its entry implied. `Pairs`: the
//: digits two at a time, their tiles stepped together (two independent chains, all loads before the MMAs, the adds
//: after). The rows are at `rows`, the dummy panel rows at `pads`, the panel after them.
//: -- see docs/internals/carry/parts.md#readout
template <class P, bool F16, bool Pairs>
__device__ __forceinline__ void read_chunk(const CarryParams& p, const uint8_t* smem,
                                           const Window& rw, int w, int first, int slots, int total,
                                           int lo, int hi, ops::SmemAddr image, uint32_t mask,
                                           ops::SmemAddr rows, ops::SmemAddr pads,
                                           ops::SmemAddr ents, int warp, int lane) {
  constexpr int kStep = Pairs ? 2 : 1;
  const ImageLane il = image_lane(lane);
  const ops::SmemAddr units =
      ops::smem_addr(smem + p.tables.units_at) + (w & 1) * p.tables.heads * kWindow;
  const Run run = chunk_runs<P>(p, smem, rw, kRead, w, first, slots, total, lane);
  //: the entries' row and panel offsets are the union's: the chunk's pool taken back by its first slot, and each
  //: entry's panel offset made the chunk's (modulo 2^16, as a grouped union's are written) before it is shuffled
  const ops::SmemAddr pool_u = rows - static_cast<uint32_t>(first * P::kReadRow);
  const uint32_t panel_back = static_cast<uint32_t>(64 * first) << 16;
  const uint2 masked = masked_entry<P>(first, static_cast<uint32_t>(lane & 7));
  //: a lane's entry `e` of a run (`n` at `to`): loaded at a clamped place, the masked entry past the run
  const auto entry_at = [&](int e, int n, int to) {
    uint2 v;
    asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];"
                 : "=r"(v.x), "=r"(v.y)
                 : "r"(ents + 8u * static_cast<uint32_t>(n > 0 ? to + min(e, n - 1) : 0)));
    v = e < n ? v : masked;
    v.y -= panel_back;
    return v;
  };
  int i = lo;
  for (int go = __any_sync(0xffffffffu, i < hi) ? 1 : 0; go > 0; --go)
#pragma unroll 1
    do {
      int d[kStep], sx[kStep];
      uint32_t f[kStep][16];
      static_for<kStep>([&](auto Q) {
        d[Q] = warp + kWarps * (i + Q);
        sx[Q] =
            reinterpret_cast<const int*>(smem + P::kExps)[exp_slot(d[Q])] - kImageTop - kCoefTop;
        state_in(f[Q], image + (static_cast<uint32_t>(d[Q]) & mask) * kPlaneBytes, il, d[Q]);
      });
      if (rw.listed) {
        int n[kStep], to[kStep], tiles = 0;
        static_for<kStep>([&](auto Q) {
          n[Q] = __shfl_sync(0xffffffffu, run.n, d[Q]);
          to[Q] = __shfl_sync(0xffffffffu, run.to, d[Q]);
          tiles = max(tiles, (n[Q] + 7) >> 3);
        });
        int s = 0;
        for (int any = __any_sync(0xffffffffu, s < tiles) ? 1 : 0; any > 0; --any)
#pragma unroll 1
          do {
            TileIn in[kStep];
            float acc[kStep][2][4];
            static_for<kStep>([&](auto Q) {
              in[Q] =
                  tile_in<P>(pool_u, units, sx[Q], entry_at(8 * s + (lane & 7), n[Q], to[Q]), lane);
            });
            static_for<kStep>([&](auto Q) { tile_mma<F16>(acc[Q], in[Q].b, f[Q]); });
            static_for<kStep>([&](auto Q) { tile_add(pads, in[Q], acc[Q], lane); });
          } while (__any_sync(0xffffffffu, ++s < tiles));
      } else {
        //: every slot of the chunk, its entry implied: its row and token, each digit's coefficient, its panel row (a slot
        //: past the chunk's end its place's dummy row); the step's digits summed into one exit a tile (every token is in
        //: every digit's list, so their union is each digit's own)
        int s = 0;
        for (int any = __any_sync(0xffffffffu, slots > 0) ? 1 : 0; any > 0; --any)
#pragma unroll 1
          do {
            const uint32_t slot = 8u * s + (lane & 7);
            const uint32_t panel =
                static_cast<int>(slot) < slots ? kPadBytes + panel_row(slot) : panel_row(slot & 7);
            const uint32_t row = slot * (P::kReadRow / 16) + chunk_at<P::kReadRow>(slot, 0);
            uint32_t coef[kStep];
            static_for<kStep>([&](auto Q) {
              if constexpr (P::kReadFactors > 1) {
                coef[Q] =
                    ld_u16(chunk_addr<P::kReadRow>(rows + 16u * row, (kSliceBytes + 2 * d[Q]) >> 4)
                           + (2 * d[Q] & 15));
              } else {
                coef[Q] = 0x3c00u;
              }
            });
            const TileInM<kStep> in =
                tile_in_m<P, kStep>(rows + 16u * row, units, sx,
                                    make_uint2((first + slot) << 16, panel << 16), coef, lane);
            float acc[kStep][2][4];
            static_for<kStep>([&](auto Q) { tile_mma<F16>(acc[Q], in.b, f[Q]); });
            tile_add_m(pads, in, acc, lane);
          } while (__any_sync(0xffffffffu, ++s < (slots + 7) >> 3));
      }
    } while (__any_sync(0xffffffffu, (i += kStep) < hi));
}

//: A GLOBAL ADD, unpredicated: ptxas puts a predicated `red.global` in a branch and a convergence region (BSSY, BRA,
//: BSYNC around each add).
__device__ __forceinline__ void red_add(uint32_t* at, uint32_t v) {
  asm volatile("red.global.add.u32 [%0], %1;" ::"l"(at), "r"(v) : "memory");
}

//: THE DRAIN: a chunk's panel words (at `panel`) to `num`, two tokens a request, packed s16 pairs at each token's unit
//: added as u32; a thread keeps one word of every 16th slot (16 slots on, its panel cell 1 KB on), two slots a step
//: loaded before their adds, unguarded (the union and the panel lie in shared memory past the chunk's slots); a step's
//: second slot past the chunk adds zero at the thread's first slot's row. A slot's row is a 32-bit word offset (the
//: host holds a head's rows under 2^32 words). -- see docs/internals/carry/parts.md#drain
template <class P>
__device__ __forceinline__ void drain(const CarryParams& p, const uint8_t* smem, const Window& rw,
                                      const Place& c, int w, int first, int slots,
                                      ops::SmemAddr panel) {
  constexpr int kWords = kPanelRow / 4, kStride = kThreads / kWords, kAhead = 2;
  constexpr uint32_t kCellStride = kStride * kPanelRow;
  static_assert(kStride % 8 == 0, "a stride of slots keeps a cell's swizzle bits");
  //: the thread, opaque, so its cell is formed at each drain rather than hoisted across the windows
  const int tid = ops::opaque(static_cast<int>(threadIdx.x));
  const int word = tid % kWords, s0 = tid / kWords;
  const ops::SmemAddr cell0 =
      panel + (panel_row(static_cast<uint32_t>(s0)) ^ ((32u * (word >> 3)) | (4u * (word & 7))));
  const ops::SmemAddr tok0 = ops::smem_addr(union_of<P>(p, smem, kRead, w) + first + s0);
  const uint32_t L = static_cast<uint32_t>(p.L), pair = static_cast<uint32_t>(p.dv / 2);
  const bool listed = rw.listed;
  //: the write head's first query head's rows; a slot's row its query head `j` (`L` rows on) and token
  uint32_t* const out = p.num + ((int64_t)c.bh * p.tables.heads * p.L + rw.t0) * pair
                        + reinterpret_cast<const volatile Addr*>(smem + P::kAddr)->channel0 / 2
                        + word;
#pragma unroll 1
  for (int i = 0; s0 + i * kStride < slots; i += kAhead) {
    uint32_t v[kAhead], r[kAhead];
    static_for<kAhead>([&](auto k) {
      asm volatile("ld.shared.u32 %0, [%1];"
                   : "=r"(v[k])
                   : "r"(cell0 + static_cast<uint32_t>((i + k) * kCellStride)));
      uint32_t u;
      asm volatile("ld.shared.u16 %0, [%1];"
                   : "=r"(u)
                   : "r"(tok0 + 2u * static_cast<uint32_t>((i + k) * kStride)));
      u = listed ? u : static_cast<uint32_t>(first + s0 + (i + k) * kStride);
      r[k] = ((u >> 9 & 0x3fu) * L + (u & (kWindow - 1))) * pair;
    });
    static_for<kAhead>([&](auto k) {
      const bool live = k == 0 || s0 + (i + k) * kStride < slots;
      red_add(out + (live ? r[k] : r[0]), live ? v[k] : 0u);
    });
  }
}

}  // namespace rola::carry
