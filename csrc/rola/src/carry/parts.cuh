// csrc/rola/src/carry/parts.cuh -- THE CARRY'S COMPONENTS at their real signatures, tenants and copies: a window's
// lists and facts in, a chunk's fills (its rows, its entries, its zeroed panel), the hand-off (the digit maxima, a
// batch's stores), a chunk's readout and its drain. see docs/internals/carry/parts.md
#pragma once

#include <cuda_fp16.h>

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
  int st[kMaxLevels];
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
      out.slices[k].at[i] =
          pl.base + b * pl.sb + h * pl.sh + (int64_t)origin_of(p, level) * s.elem[i];
      out.slices[k].st[i] = static_cast<int>(pl.st);
      out.slices[k].bytes[i] = p.extents[level] * s.elem[i];
    }
  }
  out.values = p.values.base + b * p.values.sb + h * p.values.sh + c.channel0 * 2;
  out.channel0 = c.channel0;
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
  out.t0 = w * kWindow;
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

//: A SIDE'S TENANTS for window `w` (two windows' a side, by parity): its union tokens, its count's word pair, its
//: group's boundaries in prefix form. -- see docs/internals/carry/plan.md#tenants
template <class P>
__device__ __forceinline__ const uint16_t* union_of(const uint8_t* smem, int side, int w) {
  return reinterpret_cast<const uint16_t*>(smem + P::kUnion) + (2 * side + (w & 1)) * kWindow;
}

//: the parity opaque, so the offset is formed at each use rather than held across a window's items
template <class P>
__device__ __forceinline__ const uint16_t* pre_of(const uint8_t* smem, int side, int w) {
  int parity;
  asm volatile("and.b32 %0, %1, 1;" : "=r"(parity) : "r"(w));
  return reinterpret_cast<const uint16_t*>(smem + P::kPre)
         + (2 * side + parity) * kPreRows * kListAt;
}

//: A CHUNK'S COLUMNS in the boundaries, union slots `first .. first + slots - 1` (`first` a multiple of 64, the end a
//: multiple of 64 or the union's `total`): the one before its first slot (-1 for none) and the one at its end.
__device__ __forceinline__ int2 chunk_cols(int first, int slots, int total) {
  return make_int2(first / kListBound - 1,
                   first + slots >= total ? kListAt - 1 : (first + slots) / kListBound - 1);
}

//: the prefix rows' value at row `l`, column `c` (column -1 none)
__device__ __forceinline__ int pre_at(const uint16_t* pre, int l, int c) {
  return c < 0 ? 0 : static_cast<int>(pre[l * kListAt + c]);
}

//: a side's slots in window `w`: its union's count, or every token where the side has no list level
template <class P>
__device__ __forceinline__ int slots_of(const uint8_t* smem, const Window& win, int side, int w) {
  return win.listed ? reinterpret_cast<const int*>(smem + P::kCount)[2 * side + (w & 1)]
                    : win.tokens;
}

//: LISTS IN, a window ahead: a side's union tokens and count and its group's boundaries in prefix form.
//: -- see docs/internals/carry/parts.md#lists-in
template <class P>
__device__ __forceinline__ void lists_in(const CarryParams& p, const SideLists& l,
                                         const Window& win, const Place& c, int w, uint8_t* smem,
                                         int side) {
  if (!win.listed) return;
  const int t = static_cast<int>(threadIdx.x), slot = 2 * side + (w & 1);
  const int64_t group = ((int64_t)c.bh * p.windows + w) * l.groups + win.group;
  constexpr int kUnionCopies = kWindow * 2 / ops::kStageBytes;
  if (t < kUnionCopies)
    ops::stage_tile(ops::smem_addr(smem + P::kUnion + slot * kWindow * 2) + 16 * t,
                    l.union_tok + group * kWindow + 8 * t);
  if (t >= kUnionCopies && t < kUnionCopies + kPreRows)
    ops::stage_tile(ops::smem_addr(smem + P::kPre) + 16 * (slot * kPreRows + t - kUnionCopies),
                    l.pre + (group * kPreRows + t - kUnionCopies) * kListAt);
  if (t == kThreads - 1)
    ops::stage_run<4>(ops::smem_addr(smem + P::kCount) + 4 * slot, l.union_count + group);
}

//: FACTS IN, a window ahead: the box's write lists' exponent pairs and the window's tokens' readout units.
template <class P>
__device__ __forceinline__ void facts_in(const CarryParams& p, const Window& ww, const Place& c,
                                         int w, uint8_t* smem) {
  const int t = static_cast<int>(threadIdx.x);
  constexpr int kUnitCopies = kWindow / ops::kStageBytes,
                kExpCopies = kBoxLists * 4 / ops::kStageBytes;
  if (t < kUnitCopies)
    ops::stage_tile(ops::smem_addr(smem + P::kUnits) + (w & 1) * kWindow + 16 * t,
                    p.unit + ((int64_t)c.bh * p.windows + w) * kWindow + 16 * t);
  if (ww.listed && t >= kUnitCopies && t < kUnitCopies + kExpCopies)
    ops::stage_tile(
        ops::smem_addr(smem + P::kWriteExp) + (w & 1) * kBoxLists * 4 + 16 * (t - kUnitCopies),
        p.write_exp + list_row(p.write_lists, c, p.windows, w, ww.digit0) * 2
            + 8 * (t - kUnitCopies));
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
  const int r = static_cast<int>(threadIdx.x) & 3;
  const uint8_t* base[Factors];
  int st[Factors], bytes[Factors];
  bool active[Factors], narrow = false;
  static_for<Factors>([&](auto i) {
    bytes[i] = sl.bytes[i];
    st[i] = sl.st[i];
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
    for (int s = static_cast<int>(threadIdx.x) >> 2; s < slots; s += kQuads) {
      const int u = win.listed ? tok[first + s] : first + s, t = u & (kWindow - 1);
      const bool live = u == t;
      const ops::SmemAddr row = pool + static_cast<uint32_t>(s * kRow);
      static_for<Factors>([&](auto i) {
        const ops::SmemAddr dst = row
                                  + 16u
                                        * chunk_at<kRow>(static_cast<uint32_t>(s),
                                                         static_cast<uint32_t>(kChunks * i + r));
        const uint8_t* const src = base[i] + t * st[i];
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
            vbase + t * values_st, live);
    }
  };
  if (narrow) {
    rows(std::true_type{});
  } else {
    rows(std::false_type{});
  }
}

//: A CHUNK'S RUNS, lane `l` list `l`'s: its first entry of the chunk, its count, and its place in a buffer's entries
//: (the lists' runs in list order), four loads from the prefix rows. -- see docs/internals/carry/parts.md#entries
struct Run {
  int at, n, to;
};

template <class P>
__device__ __forceinline__ Run chunk_runs(const uint8_t* smem, const Window& win, int side, int w,
                                          int first, int slots, int total, int lane) {
  if (!win.listed || lane >= win.lists) return Run{0, 0, 0};
  const uint16_t* pre = pre_of<P>(smem, side, w);
  const int2 col = chunk_cols(first, slots, total);
  const int l = (win.digit0 & (kBoxLists - 1)) + lane;
  const int a0 = pre_at(pre, l, col.x), a1 = pre_at(pre, l, col.y);
  const int b0 = pre_at(pre, l + 1, col.x), b1 = pre_at(pre, l + 1, col.y);
  return Run{b0 - a0, (b1 - a1) - (b0 - a0), a1 - a0};
}

//: A CHUNK'S ENTRIES, a warp's share: its own lists' runs (`warp + 8 j`) to their places at `dst`, 8 bytes a lane.
template <class P>
__device__ __forceinline__ void entries_fill(const CarryParams& p, const SideLists& l,
                                             const Window& win, const Place& c, int w,
                                             const Run& run, ops::SmemAddr dst, int warp,
                                             int lane) {
  //: the box's first list's row, once: a list's is `kListCap` entries a list on
  const uint2* const lists = l.entries + list_row(l, c, p.windows, w, win.digit0) * kListCap;
  static_for<kBoxLists / kWarps>([&](auto J) {
    const int list = warp + kWarps * J;
    const int at = __shfl_sync(0xffffffffu, run.at, list),
              n = __shfl_sync(0xffffffffu, run.n, list),
              to = __shfl_sync(0xffffffffu, run.to, list);
    const uint2* const src = lists + list * kListCap + at + lane;
    const ops::SmemAddr place = dst + 8u * static_cast<uint32_t>(to + lane);
#pragma unroll 1
    for (int k = 0; k < n; k += 32) ops::stage_run_lanes<8>(place + 8u * k, src + k, k + lane < n);
  });
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
  fill<P::kReadFactors, false>(addr.slices[kRead], nullptr, 0, union_of<P>(smem, kRead, w), rw,
                               first, slots, buf);
  const ops::SmemAddr pads = buf + static_cast<uint32_t>(read_pads<P>(slots));
  //: a zero quantum reads nothing: its source is any global address
#pragma unroll 4
  for (int k = static_cast<int>(threadIdx.x); k < (read_ents<P>(slots) - read_pads<P>(slots)) / 16;
       k += kThreads)
    ops::stage_tile_if(pads + 16u * static_cast<uint32_t>(k), p.num, false);
  if (rw.listed)
    entries_fill<P>(p, p.read_lists, rw, c, w,
                    chunk_runs<P>(smem, rw, kRead, w, first, slots, total, lane),
                    buf + static_cast<uint32_t>(read_ents<P>(slots)), warp, lane);
}

//: A WARP'S LISTS' UNITS, list `j` the box's list `warp + 8 j`.
template <class P>
__device__ __forceinline__ void list_units(int (&u)[P::kFoldLists], const uint8_t* smem, int warp) {
  //: read at each use (volatile): merged across a window's stores they would hold registers across it
  const volatile UnitBook* const book = reinterpret_cast<const volatile UnitBook*>(smem + P::kBook);
  static_for<P::kFoldLists>([&](auto J) { u[J] = book[warp + kWarps * J].unit; });
}

//: THE DIGIT MAXIMA, the hand-off's first step: a warp's largest |state| a digit over its lists, each list's at its
//: unit, as an exponent -- `ceil_exp` of the list's largest |accumulator| plus its unit, the largest over the lists --
//: the four lanes of a digit row joined by two shuffles, posted for the CTA at window `w`'s parity.
//: -- see docs/internals/carry/parts.md#maxima
template <class P>
__device__ __forceinline__ void post_maxima(const State<P>& st, uint8_t* smem, int w, int warp,
                                            int lane) {
  int16_t* const posts =
      reinterpret_cast<int16_t*>(smem + P::kMaxima) + ((w & 1) * kWarps + warp) * kBoxDigits;
  int u[P::kFoldLists];
  list_units<P>(u, smem, warp);
  static_for<P::kFoldMTiles>([&](auto M) {
    static_for<2>([&](auto H) {
      int e = 2 * kUnitNone;
      static_for<P::kFoldLists>([&](auto J) {
        float x = 0.0f;
        static_for<4>([&](auto N) {
          x = fmaxf(x, fmaxf(fabsf(st.acc[J][M][N][2 * H]), fabsf(st.acc[J][M][N][2 * H + 1])));
        });
        e = max(e, ceil_exp(x) + u[J]);
      });
      e = max(e, __shfl_xor_sync(0xffffffffu, e, 1));
      e = max(e, __shfl_xor_sync(0xffffffffu, e, 2));
      if ((lane & 3) == 0) posts[16 * M + 8 * H + (lane >> 2)] = static_cast<int16_t>(e);
    });
  });
}

//: A 16-BYTE UNIT of digit `d`'s image plane for list `l`, chunk `k` (8 channels): the fold's stores and the readout's
//: `ldmatrix.trans` both conflict-free. -- see docs/internals/carry/parts.md#image
__host__ __device__ __forceinline__ constexpr uint32_t unit_of(int l, int k, int d) {
  return static_cast<uint32_t>(((4 * l + (k ^ ((l >> 1) & 3))) ^ (d & 7)) & (kPlaneBytes / 16 - 1));
}

//: THE IMAGE'S TOP: a digit's largest value placed under 2^10, so a 32-list sum fits an fp16 accumulator.
//: -- see docs/internals/carry/parts.md#maxima
constexpr int kImageTop = 10;

//: A DIGIT'S EXPONENT: the smallest `e` with the digit's largest |state| (its lists at their units) at most 2^e, over
//: the 8 warps' posts at window `w`'s parity, at least `kImageTop - 127`.
template <class P>
__device__ __forceinline__ int digit_exp(const uint8_t* smem, int w, int d) {
  const int16_t* posts =
      reinterpret_cast<const int16_t*>(smem + P::kMaxima) + (w & 1) * kWarps * kBoxDigits + d;
  int e = kImageTop - 127;
  static_for<kWarps>([&](auto W) { e = max(e, posts[W * kBoxDigits]); });
  return e;
}

//: A BATCH'S STORES, the hand-off: the warp's lists' rows of batches `lo .. hi - 1` (8 digits each) into the image at
//: `image` (digit `d`'s plane `d & mask`), f16 at each digit's power of two, each list moved there from its unit
//: (`2^(unit + kImageTop - e)`, held to a normal float's range: past it the f16 is zero or the list's value is not the
//: digit's); `dexp` lane `d` digit `d`'s exponent (`digit_exp`). -- see docs/internals/carry/parts.md#store
//: a lane's unit for list `w + 8 j` at residue `g` is its unit for list `w`, 32 units on: so a batch's stores take
//: four bases a lane, a list an offset
constexpr bool units_step_by_list() {
  for (int w = 0; w < kWarps; ++w)
    for (int g = 0; g < 8; ++g)
      for (int j = 0; j < kBoxLists / kWarps; ++j)
        if (unit_of(w + kWarps * j, 0, g) != unit_of(w, 0, g) + 32u * j) return false;
  return true;
}

static_assert(units_step_by_list(),
              "a list's image unit is its warp's first list's, 32 units a list on");

template <class P>
__device__ __forceinline__ void store_batches(const State<P>& st, uint8_t* smem,
                                              ops::SmemAddr image, int lo, int hi, uint32_t mask,
                                              int dexp, int warp, int lane) {
  const int g = lane >> 2, t = lane & 3;
  //: the lane's unit base, opaque, so its terms are formed at each call rather than hoisted across the windows
  uint32_t base;
  asm volatile("mov.u32 %0, %1;" : "=r"(base) : "r"(unit_of(warp, 0, g)));
  //: channel tile `n`'s chunk is the unit XORed by `n` (the image is 128-byte aligned, so that is its address's)
  const uint32_t lane0 = image + 4u * t;
  int u[P::kFoldLists];
  list_units<P>(u, smem, warp);
  static_for<P::kFoldMTiles>([&](auto M) {
    static_for<2>([&](auto H) {
      constexpr int b = 2 * M + H;
      const int d = 16 * M + 8 * H + g;
      const int e = __shfl_sync(0xffffffffu, dexp, d);
      if (b >= lo && b < hi) {
        const uint32_t plane = (static_cast<uint32_t>(d) & mask) * kPlaneBytes;
        static_for<P::kFoldLists>([&](auto J) {
          const float scale = pow2(min(max(u[J] + kImageTop - e, -126), 127));
          static_for<4>([&](auto N) {
            const __half2 h = __floats2half2_rn(st.acc[J][M][N][2 * H] * scale,
                                                st.acc[J][M][N][2 * H + 1] * scale);
            const uint32_t v = *reinterpret_cast<const uint32_t*>(&h);
            const uint32_t at = lane0 + 16u * (base ^ static_cast<uint32_t>(N)) + plane
                                + 512u * static_cast<uint32_t>(J);
            asm volatile("st.shared.u32 [%0], %1;" ::"r"(at), "r"(v) : "memory");
          });
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
  const uint32_t x = 16u * (d & 7);
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

__device__ __forceinline__ void panel_red(uint32_t at, uint32_t v) {
  asm volatile("red.shared.add.u32 [%0], %1;" ::"r"(at), "r"(v) : "memory");
}

//: A READOUT TILE, 8 tokens on N against a digit's state (the A operand): their read rows (B, at the entries' 16-byte
//: chunks) by one `ldmatrix.x4`, four MMAs accumulating in fp16 (`F16`) or fp32, each lane's two tokens' outputs as
//: exact pairs at their panel rows; lane `l` holds entry `l & 7` and scales its coefficient to its token's unit by
//: `2^(sx - u)` (`sx` the digit's exponent less `kImageTop` and `kCoefTop`). -- see docs/internals/carry/parts.md#readout
//: @burst
template <class P, bool F16>
__device__ __forceinline__ void tile(ops::SmemAddr pool, ops::SmemAddr pads, ops::SmemAddr units,
                                     int sx, uint2 e, const uint32_t (&f)[16], int lane) {
  const int g = lane >> 2, t = lane & 3;
  int u;
  asm volatile("ld.shared.s8 %0, [%1];" : "=r"(u) : "r"(units + (e.x >> 16)));
  const float c = __half2float(__ushort_as_half(static_cast<unsigned short>(e.y)));
  const float cu = __uint_as_float(__float_as_uint(c) + (static_cast<uint32_t>(sx - u) << 23));
  const uint32_t e0 = __shfl_sync(0xffffffffu, e.y, 2 * t),
                 e1 = __shfl_sync(0xffffffffu, e.y, 2 * t + 1);
  const float c0 = __shfl_sync(0xffffffffu, cu, 2 * t),
              c1 = __shfl_sync(0xffffffffu, cu, 2 * t + 1);
  uint32_t b[4];
  ops::load_frag(
      b, chunk_addr<P::kReadRow>(pool + 16u * (e.x & 0xffffu), static_cast<uint32_t>(lane >> 3)));
  float acc[2][4] = {};
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    uint32_t h[2] = {0u, 0u};
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

  const uint32_t b0 = pads + (e0 >> 16), b1 = pads + (e1 >> 16);
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    panel_red((b0 ^ (32u * m)) + 4u * g, s16_pair(acc[m][0], acc[m][2], c0));
    panel_red((b1 ^ (32u * m)) + 4u * g, s16_pair(acc[m][1], acc[m][3], c1));
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
//: buffer's entries at `ents`, a lane past it masked; else every slot of the chunk, its entry implied. The rows are at
//: `rows`, the dummy panel rows at `pads`, the panel after them. -- see docs/internals/carry/parts.md#readout
template <class P, bool F16>
__device__ __forceinline__ void read_chunk(const uint8_t* smem, const Window& rw, int w, int first,
                                           int slots, int total, int lo, int hi,
                                           ops::SmemAddr image, uint32_t mask, ops::SmemAddr rows,
                                           ops::SmemAddr pads, ops::SmemAddr ents, int warp,
                                           int lane) {
  const ImageLane il = image_lane(lane);
  const ops::SmemAddr units = ops::smem_addr(smem + P::kUnits) + (w & 1) * kWindow;
  const Run run = chunk_runs<P>(smem, rw, kRead, w, first, slots, total, lane);
  //: the entries' row and panel offsets are the union's: the chunk's pool and panel taken back by its first slot
  const ops::SmemAddr pool_u = rows - static_cast<uint32_t>(first * P::kReadRow),
                      pads_u = pads - static_cast<uint32_t>(64 * first);
  const uint2 masked = masked_entry<P>(first, static_cast<uint32_t>(lane & 7));
  int i = lo;
  for (int go = __any_sync(0xffffffffu, i < hi) ? 1 : 0; go > 0; --go)
#pragma unroll 1
    do {
      const int d = warp + kWarps * i;
      const int sx = reinterpret_cast<const int*>(smem + P::kExps)[d] - kImageTop - kCoefTop;
      uint32_t f[16];
      state_in(f, image + (static_cast<uint32_t>(d) & mask) * kPlaneBytes, il, d);
      if (rw.listed) {
        const int n = __shfl_sync(0xffffffffu, run.n, d), to = __shfl_sync(0xffffffffu, run.to, d);
        const int tiles = (n + 7) >> 3;
        int s = 0;
        for (int any = __any_sync(0xffffffffu, s < tiles) ? 1 : 0; any > 0; --any)
#pragma unroll 1
          do {
            const int e = 8 * s + (lane & 7);
            uint2 entry = masked;
            if (e < n)
              asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];"
                           : "=r"(entry.x), "=r"(entry.y)
                           : "r"(ents + 8u * static_cast<uint32_t>(to + e)));
            tile<P, F16>(pool_u, pads_u, units, sx, entry, f, lane);
          } while (__any_sync(0xffffffffu, ++s < tiles));
      } else {
        //: every slot of the chunk, its entry implied: its row and token, its coefficient, its panel row (a slot past
        //: the chunk's end its place's dummy row)
        int s = 0;
        for (int any = __any_sync(0xffffffffu, slots > 0) ? 1 : 0; any > 0; --any)
#pragma unroll 1
          do {
            const uint32_t slot = 8u * s + (lane & 7);
            const uint32_t panel =
                static_cast<int>(slot) < slots ? kPadBytes + panel_row(slot) : panel_row(slot & 7);
            const uint32_t row = slot * (P::kReadRow / 16) + chunk_at<P::kReadRow>(slot, 0);
            uint32_t coef = 0x3c00u;
            if constexpr (P::kReadFactors > 1)
              asm volatile(
                  "ld.shared.u16 %0, [%1];"
                  : "=r"(coef)
                  : "r"(chunk_addr<P::kReadRow>(rows + 16u * row, (kSliceBytes + 2 * d) >> 4)
                        + (2 * d & 15)));
            const uint2 e = make_uint2(row | (first + slot) << 16, coef | (panel << 16));
            tile<P, F16>(rows, pads, units, sx, e, f, lane);
          } while (__any_sync(0xffffffffu, ++s < (slots + 7) >> 3));
      }
    } while (__any_sync(0xffffffffu, ++i < hi));
}

//: A GLOBAL ADD of a nonzero word, predicated rather than branched; zero adds nothing.
__device__ __forceinline__ void red_nonzero(uint32_t* at, uint32_t v) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.u32 p, %1, 0;\n @p red.global.add.u32 [%0], %1;\n}\n" ::"l"(at),
      "r"(v));
}

//: THE DRAIN: a chunk's panel words (at `panel`) to `num`, two tokens a request, packed s16 pairs at each token's unit
//: added as u32; a thread keeps one word of every 16th slot (16 slots on, its panel cell 1 KB on), two slots a step
//: loaded before their adds. -- see docs/internals/carry/parts.md#drain
template <class P>
__device__ __forceinline__ void drain(const CarryParams& p, const uint8_t* smem, const Window& rw,
                                      const Place& c, int w, int first, int slots,
                                      ops::SmemAddr panel) {
  constexpr int kWords = kPanelRow / 4, kStride = kThreads / kWords, kAhead = 2;
  constexpr uint32_t kCellStride = kStride * kPanelRow;
  static_assert(kStride % 8 == 0, "a stride of slots keeps a cell's swizzle bits");
  const int word = static_cast<int>(threadIdx.x) % kWords,
            s0 = static_cast<int>(threadIdx.x) / kWords;
  const ops::SmemAddr cell0 =
      panel + (panel_row(static_cast<uint32_t>(s0)) ^ ((32u * (word >> 3)) | (4u * (word & 7))));
  const uint16_t* tok = union_of<P>(smem, kRead, w) + first;
  const int pair = p.dv / 2;
  uint32_t* const out = p.num + ((int64_t)c.bh * p.L + rw.t0) * pair
                        + reinterpret_cast<const volatile Addr*>(smem + P::kAddr)->channel0 / 2
                        + word;
#pragma unroll 1
  for (int i = 0; s0 + i * kStride < slots; i += kAhead) {
    uint32_t v[kAhead];
    int t[kAhead];
    static_for<kAhead>([&](auto k) {
      const int s = s0 + (i + k) * kStride;
      v[k] = 0u;
      if (s < slots)
        asm volatile("ld.shared.u32 %0, [%1];"
                     : "=r"(v[k])
                     : "r"(cell0 + static_cast<uint32_t>((i + k) * kCellStride)));
      t[k] = s < slots ? (rw.listed ? tok[s] : first + s) : 0;
    });
    static_for<kAhead>([&](auto k) { red_nonzero(out + t[k] * pair, v[k]); });
  }
}

}  // namespace rola::carry
