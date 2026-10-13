// csrc/rola/src/carry/fold.cuh -- THE FOLD: a warp's write lists over a window's units into the fp32 state -- each
// list in its running unit, steps of 16 tokens and an 8-token tail, a step's A the slots' dense slice times the
// entries' c, its B their values, fp32-accumulate MMAs. see docs/internals/carry/fold.md
#pragma once

#include <cuda_fp16.h>

#include <cstdint>

#include "carry/parts.cuh"

namespace rola::carry {

//: THE UNITS' CONSTANTS: c's top 2^4, a state held under 2^100 of its unit, the lowest unit, a list never live, a
//: window's largest add in units, the facts pass's dead exponent. -- see docs/internals/carry/fold.md#units
constexpr int kCoefTop = 4;
constexpr int kUnitHeadroom = 100;
constexpr int kUnitMin = kCoefTop - 127;
constexpr int kUnitNone = -1000;
constexpr float kWindowAdd = kWindow * 16.0f;
constexpr int kExpDead = -126;
//: A DENSE WRITE LEVEL'S BYTE SCALE: its f16 digits carry 1/255 of the byte scale c is formed on.
constexpr float kByteScale = 1.0f / 255.0f;

//: A WARP'S LIST `j`: the box's list `warp + 8 j`, as the heads and the image's stores lay the lists out.
__device__ __forceinline__ constexpr int fold_list(int warp, int j) { return warp + kWarps * j; }

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

//: 2^e as an f16 pair: normal from 2^-14, subnormal to 2^-24, zero below.
__device__ __forceinline__ uint32_t pow2_half2(int e) {
  const uint32_t h = e >= -14   ? static_cast<uint32_t>(e + 15) << 10
                     : e >= -24 ? 1u << (e + 24)
                                : 0u;
  return h | (h << 16);
}

//: A UNIT'S NEXT WINDOW: a live list's exponent, held within the bound's headroom; the state's rescale returned and
//: the bound moved (`next_units` in tests/oracle/fold_reference.py). -- see docs/internals/carry/fold.md#units
__device__ __forceinline__ float next_unit(int k, UnitBook& book) {
  const int held = book.unit + ceil_exp(book.bound) - kUnitHeadroom;
  const int next = max(k <= kExpDead ? book.unit : max(k, held), kUnitMin);
  const int d = book.unit - next;
  const float f = d < -126 ? 0.0f : pow2(d);
  book.bound = __fmaf_rn(book.bound, f, kWindowAdd);
  book.unit = next;
  return f;
}

//: A WINDOW'S UNITS, before the warp's steps: each list's exponent moves its book (lane group `j` list `j`); its state's
//: rescale `f` (applied beside the list's first MMAs) and its c factor `2^(E - unit)` taken. A side with no list
//: level keeps its unit. -- see docs/internals/carry/fold.md#units
template <class P>
__device__ __forceinline__ void fold_units(uint32_t (&cf)[P::kFoldLists], float (&f)[P::kFoldLists],
                                           uint8_t* smem, const Window& ww, int w, int warp,
                                           int lane) {
  constexpr int kLanes = 32 / P::kFoldLists;
  const int j = lane / kLanes;
  const int16_t* exps =
      reinterpret_cast<const int16_t*>(smem + P::kWriteExp) + (w & 1) * kBoxLists * 2;
  const int k = ww.listed ? exps[2 * fold_list(warp, j) + 1] : kExpDead;
  UnitBook* const book = reinterpret_cast<UnitBook*>(smem + P::kBook) + fold_list(warp, j);
  UnitBook b = *book;
  const float fl = next_unit(k, b);
  const int cexp = max(k, kUnitMin) - b.unit;
  __syncwarp();
  if (lane % kLanes == 0) *book = b;
  static_for<P::kFoldLists>([&](auto J) {
    f[J] = __shfl_sync(0xffffffffu, fl, J * kLanes);
    cf[J] = pow2_half2(__shfl_sync(0xffffffffu, cexp, J * kLanes));
  });
}

//: AN ENTRY'S LANE VALUES: its slot's row offset in the unit's pool (its swizzle folded in) and its c paired with the
//: next lane's, moved to the list's unit. A pad (past the list's count) is row 0 with c = 0.
template <class P>
__device__ __forceinline__ void entry_of(uint2 rec, bool live, int u, uint32_t cf, uint32_t& off,
                                         uint32_t& cpair) {
  const uint32_t r = (rec.x & 0xffffu) - static_cast<uint32_t>(u * P::kUnitSlots);
  off = live ? r * P::kWriteRow + 16u * chunk_at<P::kWriteRow>(r, 0) : 0u;
  const uint32_t mine = live ? rec.y & 0xffffu : 0u;
  cpair = ops::hmul2(mine | (__shfl_xor_sync(0xffffffffu, mine, 1) << 16), cf);
}

//: A 16-TOKEN STEP into list `J`'s accumulators: its slots' dense slices (A, `ldmatrix.trans`, times c) and values
//: (B), eight fp32-accumulate MMAs. Lanes `0..15` hold its entries. -- see docs/internals/carry/fold.md#step
//: @burst
template <class P, int J>
__device__ __forceinline__ void step16(State<P>& st, ops::SmemAddr pool, uint32_t off,
                                       uint32_t cpair, int lane) {
  const int r = lane & 7, q = lane >> 3, t = lane & 3;
  const uint32_t oa = __shfl_sync(0xffffffffu, off, 8 * (q >> 1) + r);
  const uint32_t ob = __shfl_sync(0xffffffffu, off, 8 * (q & 1) + r);
  const uint32_t c01 = __shfl_sync(0xffffffffu, cpair, 2 * t),
                 c89 = __shfl_sync(0xffffffffu, cpair, 2 * t + 8);
  uint32_t a[P::kFoldMTiles][4], b[4][2];
  static_for<P::kFoldMTiles>([&](auto M) {
    ops::load_frag_t(a[M], chunk_addr<P::kWriteRow>(pool + oa, 2 * M + (q & 1)));
    a[M][0] = ops::hmul2(a[M][0], c01);
    a[M][1] = ops::hmul2(a[M][1], c01);
    a[M][2] = ops::hmul2(a[M][2], c89);
    a[M][3] = ops::hmul2(a[M][3], c89);
  });
  static_for<2>([&](auto H) {
    uint32_t x[4];
    ops::load_frag_t(x, chunk_addr<P::kWriteRow>(pool + ob, P::kValuesAt / 16 + 2 * H + (q >> 1)));
    b[2 * H][0] = x[0];
    b[2 * H][1] = x[1];
    b[2 * H + 1][0] = x[2];
    b[2 * H + 1][1] = x[3];
  });
  static_for<P::kFoldMTiles>([&](auto M) {
    static_for<4>([&](auto N) { ops::hmma(st.acc[J][M][N], a[M], b[N][0], b[N][1]); });
  });
}

//: AN 8-TOKEN TAIL into list `J`'s accumulators: m16n8k8 MMAs, half a step's tensor work. Lanes `0..7` hold its
//: entries. -- see docs/internals/carry/fold.md#step
//: @burst
template <class P, int J>
__device__ __forceinline__ void step8(State<P>& st, ops::SmemAddr pool, uint32_t off,
                                      uint32_t cpair, int lane) {
  const int r = lane & 7, q = lane >> 3, t = lane & 3;
  const uint32_t o = __shfl_sync(0xffffffffu, off, r), c01 = __shfl_sync(0xffffffffu, cpair, 2 * t);
  uint32_t a[4], b[4];
  ops::load_frag_t(a, chunk_addr<P::kWriteRow>(pool + o, q));
  ops::load_frag_t(b, chunk_addr<P::kWriteRow>(pool + o, P::kValuesAt / 16 + q));
#pragma unroll
  for (int k = 0; k < 4; ++k) a[k] = ops::hmul2(a[k], c01);
  static_for<P::kFoldMTiles>([&](auto M) {
    static_for<4>([&](auto N) { ops::hmma_k8(st.acc[J][M][N], a[2 * M], a[2 * M + 1], b[N]); });
  });
}

//: A LIST'S STEPS over its entries `s .. n`: 16-token steps while more than 8 remain, then an 8-token tail, `fetch`
//: the entry at an index -- each loop's step one block. -- see docs/internals/carry/fold.md#step
template <class P, int J, class Fetch>
__device__ __forceinline__ void steps(State<P>& st, uint32_t cf, ops::SmemAddr pool, int u, int s,
                                      int n, int lane, Fetch fetch) {
  for (int k = __any_sync(0xffffffffu, n - s > 8) ? 1 : 0; k > 0; --k)
#pragma unroll 1
    do {
      const int e = s + (lane & 15);
      uint32_t off, cpair;
      entry_of<P>(fetch(min(e, n - 1)), e < n, u, cf, off, cpair);
      step16<P, J>(st, pool, off, cpair, lane);
      s += 16;
    } while (__any_sync(0xffffffffu, n - s > 8));
  for (int tail = __any_sync(0xffffffffu, n - s > 0) ? 1 : 0; tail > 0; --tail) {
    const int k = s + (lane & 7);
    uint32_t off, cpair;
    entry_of<P>(fetch(min(k, n - 1)), k < n, u, cf, off, cpair);
    step8<P, J>(st, pool, off, cpair, lane);
  }
}

//: A FOLD UNIT, a warp's: each of its lists' entries in unit `u` -- the first 16 of unit 0 from the warp's heads, the
//: rest in place -- in 16-token steps and an 8-token tail, the list's rescale applied before its first; a side with no
//: list level every slot of the unit, c from the slot's row. -- see docs/internals/carry/fold.md#unit
template <class P>
__device__ __forceinline__ void fold_unit(const CarryParams& p, State<P>& st,
                                          const uint32_t (&cf)[P::kFoldLists],
                                          float (&f)[P::kFoldLists], const uint8_t* smem,
                                          const Window& ww, const Place& c, int w, int u, int slots,
                                          ops::SmemAddr pool, int warp, int lane) {
  const ops::SmemAddr heads = ops::smem_addr(smem + P::kFoldHeads) + warp * P::kHeadBytes;
  static_for<P::kFoldLists>([&](auto J) {
    static_for<P::kFoldMTiles>([&](auto M) {
      static_for<4>([&](auto N) { static_for<4>([&](auto I) { st.acc[J][M][N][I] *= f[J]; }); });
    });
    f[J] = 1.0f;

    const int list = fold_list(warp, J);
    int first = 0, n = 0;
    if (ww.listed) {
      const uint16_t* at = at_of<P>(smem, kWrite, w, list);
      first = at[u];
      n = at[u + 1] - first;
    } else {
      n = max(0, min(slots - u * P::kUnitSlots, P::kUnitSlots));
    }
    const uint2* entries =
        ww.listed
            ? p.write_lists.entries
                  + list_row(p.write_lists, c, p.windows, w, ww.digit0 + list) * kListCap + first
            : nullptr;

    if (!ww.listed) {
      //: every slot of the unit, c the list's digit of the row's coefficient slice (the row's second)
      steps<P, J>(st, cf[J], pool, u, 0, n, lane, [&](int k) {
        const uint32_t row0 =
            static_cast<uint32_t>(k) * P::kWriteRow + 16u * chunk_at<P::kWriteRow>(k, 0);
        uint32_t cv;
        asm volatile("ld.shared.u16 %0, [%1];"
                     : "=r"(cv)
                     : "r"(chunk_addr<P::kWriteRow>(pool + row0, (kSliceBytes + 2 * list) >> 4)
                           + (2 * list & 15)));
        return make_uint2(static_cast<uint32_t>(u * P::kUnitSlots + k), cv);
      });
    } else {
      //: unit 0's first entries from the warp's heads, the rest in place
      const int staged = u == 0 ? min(n, kHeadEntries) : 0;
      steps<P, J>(st, cf[J], pool, u, 0, staged, lane, [&](int k) {
        uint2 e;
        asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];"
                     : "=r"(e.x), "=r"(e.y)
                     : "r"(heads + 8u * (kHeadEntries * J + k)));
        return e;
      });
      steps<P, J>(st, cf[J], pool, u, staged, n, lane, [&](int k) { return entries[k]; });
    }
  });
}

//: THE PROBE (an instrument; nothing unbound): the state at the call's end, absolute, fp32 `[BH, N, d_v]` by global
//: leaf in level order. -- see docs/internals/carry/fold.md#probe
__device__ __forceinline__ int64_t level_stride(const CarryParams& p, int level) {
  int64_t s = 1;
#pragma unroll 1
  for (int l = level + 1; l < p.D; ++l) s *= p.widths[l];
  return s;
}

template <class P>
__device__ __forceinline__ void fold_probe(const CarryParams& p, const State<P>& st,
                                           const uint8_t* smem, const Window& win, const Place& c,
                                           int warp, int lane) {
  if (p.probe == nullptr) return;
  const int64_t step = level_stride(p, -1);
  int64_t base = 0;
#pragma unroll 1
  for (int l = 0; l < p.D; ++l) base += origin_of(p, l) * level_stride(p, l);
  const int64_t list_stride = win.listed ? level_stride(p, p.write.list) : 0;
  const int64_t row_stride = level_stride(p, p.write.operand[0]);
  const int g = lane >> 2, tq = lane & 3;
  const UnitBook* book = reinterpret_cast<const UnitBook*>(smem + P::kBook);
  static_for<P::kFoldLists>([&](auto J) {
    const int list = fold_list(warp, J);
    const float scale = pow2(book[list].unit - kCoefTop) * kByteScale;
    const int64_t at_list = base + list * list_stride;
    static_for<P::kFoldMTiles>([&](auto M) {
      static_for<2>([&](auto Hi) {
        const int row = 16 * M + g + 8 * Hi;
        const int64_t leaf = at_list + row * row_stride;
        float* const out = p.probe + (static_cast<int64_t>(c.bh) * step + leaf) * p.dv + c.channel0;
        static_for<4>([&](auto N) {
          *reinterpret_cast<float2*>(out + 8 * N + 2 * tq) =
              make_float2(st.acc[J][M][N][2 * Hi] * scale, st.acc[J][M][N][2 * Hi + 1] * scale);
        });
      });
    });
  });
}

}  // namespace rola::carry
