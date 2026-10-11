// csrc/rola/src/carry/fold.cuh -- THE FOLD: a warp's write lists over a window's units into the fp32 state -- each
// list in its running unit, a step one f16 operand of the slot's dense slice times the record's c, eight fp32-
// accumulate MMAs. see docs/internals/carry/fold.md
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

//: A WINDOW'S UNITS, before the warp's passes: each list's exponent moves its book (lane group `j` list `j`); its state's
//: rescale `f` (applied beside the list's first MMAs) and its c factor `2^(E - unit)` taken.
//: -- see docs/internals/carry/fold.md#units
template <class P>
__device__ __forceinline__ void fold_units(uint32_t (&cf)[P::kFoldLists], float (&f)[P::kFoldLists],
                                           uint8_t* smem, const Window& win, int warp, int lane) {
  constexpr int kLanes = 32 / P::kFoldLists;
  const int j = lane / kLanes;
  const int k = reinterpret_cast<const int16_t*>(
      smem + P::kWriteExp)[2 * list_of<P::kFoldLists>(warp, j, win.lists) + 1];
  UnitBook* const book = reinterpret_cast<UnitBook*>(smem + P::kBook) + warp * P::kFoldLists + j;
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

//: THE WINDOW'S LIST EXPONENTS IN, with the lists: the box lists' `(K_m, K_v)` pairs into their tenant.
template <class P>
__device__ __forceinline__ void exps_in(const CarryParams& p, const Window& win, const Place& c,
                                        int w, uint8_t* smem) {
  if (!win.listed) return;
  const uint8_t* src = reinterpret_cast<const uint8_t*>(
      p.write_exp + list_row(p.write_lists, c, p.windows, w, win.digit0) * 2);
  const int bytes = win.lists * 4, run = min(bytes, ops::kStageBytes);
  const ops::SmemAddr dst = ops::smem_addr(smem + P::kWriteExp);
#pragma unroll 1
  for (int k = static_cast<int>(threadIdx.x); k < bytes / run; k += kThreads)
    stage_bytes(dst + k * run, src + k * run, run);
}

//: A UNIT'S FILL: its slots' write rows into slot `u & 1` (slot 0 clear of the readout's region), landed on the
//: slot's barrier. -- see docs/internals/carry/fold.md#unit
template <class P>
__device__ __forceinline__ void fill_unit(const CarryParams& p, const Place& c, const Window& ww,
                                          int write_slots, int w, int u, uint8_t* smem) {
  const int slot = u & 1, first = u * P::kUnitSlots;
  const uint8_t* values =
      p.values.base + c.bh / p.H * p.values.sb + c.bh % p.H * p.values.sh + c.channel0 * 2;
  fill<P::kFoldFactors, P::kWritePitch, true>(
      slices_of<P::kFoldFactors>(p, p.write, c), values, p.values.st, P::kChannels,
      reinterpret_cast<const uint16_t*>(smem + P::kUnion + kWindow * 2), ww, first,
      min(write_slots - first, P::kUnitSlots),
      ops::smem_addr(smem + P::kPhase + (slot ? 0 : P::kWriteSlot0)));
  if (ww.listed) {
    const ops::SmemAddr heads = ops::smem_addr(smem + P::kPhase + P::kHeads + slot * P::kHeadBytes);
    const uint16_t* at = reinterpret_cast<const uint16_t*>(smem + P::kWriteAt);
#pragma unroll 1
    for (int k = static_cast<int>(threadIdx.x); k < ww.lists * 16; k += kThreads) {
      const int list = k >> 4, e = k & 15;
      const int begin = at[list * kListAt + P::kFoldUnit * u];
      const int n = at[list * kListAt + P::kFoldUnit * (u + 1)] - begin;
      const uint2* recs = reinterpret_cast<const uint2*>(p.write_lists.entries)
                          + list_row(p.write_lists, c, p.windows, w, ww.digit0 + list) * kListCap;
      ops::stage_run_if<8>(heads + 8 * k, recs + begin + min(e, max(n - 1, 0)), e < n);
    }
  }
  ops::mbar_arrive_when_landed(ops::smem_addr(smem + P::kBars) + 8 * slot);
}

//: A SLOT'S ROW OFFSET in a unit's pool: its row by the pitch, the swizzle's chunk bit folded in.
template <class P>
__device__ __forceinline__ uint32_t row_offset(uint32_t slot) {
  const uint32_t r = slot & (P::kUnitSlots - 1);
  return r * P::kWritePitch + 16u * ((r >> 2) & 1u);
}

//: A STEP'S OPERANDS at fixed registers: A (the dense slice's m-tiles), B (the values' n-tiles), the c pairs.
template <class P>
struct Operand {
  uint32_t a[P::kFoldMTiles][4], b[4][2], c01, c89;
};

//: A STEP'S OPERANDS LOADED, its 16 entries held by lanes `16 h ..` as row offsets and c pairs: A (the dense slice)
//: and B (the values) by `ldmatrix.trans`, c moved to the list's unit. -- see docs/internals/carry/fold.md#step
template <class P>
__device__ __forceinline__ void load_step(Operand<P>& o, ops::SmemAddr pool, uint32_t off,
                                          uint32_t cpair, uint32_t cf, int h, int lane) {
  const int r = lane & 7, q = lane >> 3, tq = lane & 3;
  const uint32_t oa = __shfl_sync(0xffffffffu, off, 16 * h + 8 * (q >> 1) + r);
  const uint32_t ob = __shfl_sync(0xffffffffu, off, 16 * h + 8 * (q & 1) + r);
  o.c01 = ops::hmul2(__shfl_sync(0xffffffffu, cpair, 16 * h + 2 * tq), cf);
  o.c89 = ops::hmul2(__shfl_sync(0xffffffffu, cpair, 16 * h + 2 * tq + 8), cf);
  const ops::SmemAddr ea = pool + (oa ^ (16u * static_cast<uint32_t>(q & 1)));
  static_for<P::kFoldMTiles>([&](auto M) { ops::load_frag_t(o.a[M], ea + 32u * M); });
  const ops::SmemAddr eb = pool + P::kValuesAt + (ob ^ (16u * static_cast<uint32_t>(q >> 1)));
  static_for<2>([&](auto H) {
    uint32_t x[4];
    ops::load_frag_t(x, eb + 32u * H);
    o.b[2 * H][0] = x[0];
    o.b[2 * H][1] = x[1];
    o.b[2 * H + 1][0] = x[2];
    o.b[2 * H + 1][1] = x[3];
  });
}

//: A LIST'S FIRST STEP'S OPERANDS from its heads in shared memory (16 records; zero past its count, a pad's row 0
//: and c = 0): each lane's two row offsets and its four tokens' c read in place, no shuffles.
//: -- see docs/internals/carry/fold.md#step
template <class P>
__device__ __forceinline__ void load_head(Operand<P>& o, ops::SmemAddr pool, const uint8_t* head,
                                          uint32_t cf, int lane) {
  const int r = lane & 7, q = lane >> 3, tq = lane & 3;
  const uint16_t* h16 = reinterpret_cast<const uint16_t*>(head);
  const uint32_t oa = row_offset<P>(h16[4 * (8 * (q >> 1) + r)]);
  const uint32_t ob = row_offset<P>(h16[4 * (8 * (q & 1) + r)]);
  const uint4 p01 = *reinterpret_cast<const uint4*>(head + 16 * tq);
  const uint4 p89 = *reinterpret_cast<const uint4*>(head + 16 * tq + 64);
  o.c01 = ops::hmul2(ops::prmt(p01.y, p01.w, 0x5410u), cf);
  o.c89 = ops::hmul2(ops::prmt(p89.y, p89.w, 0x5410u), cf);
  const ops::SmemAddr ea = pool + (oa ^ (16u * static_cast<uint32_t>(q & 1)));
  static_for<P::kFoldMTiles>([&](auto M) { ops::load_frag_t(o.a[M], ea + 32u * M); });
  const ops::SmemAddr eb = pool + P::kValuesAt + (ob ^ (16u * static_cast<uint32_t>(q >> 1)));
  static_for<2>([&](auto H) {
    uint32_t x[4];
    ops::load_frag_t(x, eb + 32u * H);
    o.b[2 * H][0] = x[0];
    o.b[2 * H][1] = x[1];
    o.b[2 * H + 1][0] = x[2];
    o.b[2 * H + 1][1] = x[3];
  });
}

//: A STEP'S MMAS into list `J`'s accumulators: A times c, eight fp32-accumulate MMAs.
template <class P, int J>
__device__ __forceinline__ void mma_step(State<P>& st, const Operand<P>& o) {
  static_for<P::kFoldMTiles>([&](auto M) {
    const uint32_t aj[4] = {ops::hmul2(o.a[M][0], o.c01), ops::hmul2(o.a[M][1], o.c01),
                            ops::hmul2(o.a[M][2], o.c89), ops::hmul2(o.a[M][3], o.c89)};
    static_for<4>([&](auto N) { ops::hmma(st.acc[J][M][N], aj, o.b[N][0], o.b[N][1]); });
  });
}

//: a record pair's lane values: the row offset of lane `L`'s entry and its c paired with the next lane's
template <class P>
__device__ __forceinline__ void pair_of(uint2 rec, uint32_t& off, uint32_t& cpair) {
  off = row_offset<P>(rec.x & 0xffffu);
  const uint32_t mine = rec.y & 0xffffu;
  cpair = mine | (__shfl_xor_sync(0xffffffffu, mine, 1) << 16);
}

template <class P>
__device__ __forceinline__ const uint2* list_records(const CarryParams& p, const Window& win,
                                                     const Place& c, int w, int warp, int j) {
  return reinterpret_cast<const uint2*>(p.write_lists.entries)
         + list_row(p.write_lists, c, p.windows, w,
                    win.digit0 + list_of<P::kFoldLists>(warp, j, win.lists))
               * kListCap;
}

//: list `j`'s first entry and count in unit `u`, from its pass starts in shared memory
template <class P>
__device__ __forceinline__ int2 list_span(const uint8_t* smem, const Window& win, int u, int warp,
                                          int j) {
  const uint16_t* at = reinterpret_cast<const uint16_t*>(smem + P::kWriteAt)
                       + list_of<P::kFoldLists>(warp, j, win.lists) * kListAt;
  const int begin = at[P::kFoldUnit * u];
  return make_int2(begin, at[P::kFoldUnit * (u + 1)] - begin);
}

//: A FOLD UNIT, a warp's: every list's first step from its heads in shared memory, in one straight-line block (an empty
//: list's is pads, adding nothing), so the scheduler loads a list's operands under the MMAs before them; a list past
//: 16 entries continues from its records in global memory. -- see docs/internals/carry/fold.md#unit
template <class P>
__device__ __forceinline__ void fold_unit(const CarryParams& p, State<P>& st,
                                          const uint32_t (&cf)[P::kFoldLists],
                                          float (&f)[P::kFoldLists], const uint8_t* smem,
                                          const Window& win, const Place& c, int w, int u,
                                          ops::SmemAddr pool, int warp, int lane) {
  const uint8_t* heads = smem + P::kPhase + P::kHeads + (u & 1) * P::kHeadBytes;
  static_for<P::kFoldLists>([&](auto J) {
    Operand<P> o;
    load_head<P>(o, pool, heads + 128 * list_of<P::kFoldLists>(warp, J, win.lists), cf[J], lane);
    static_for<P::kFoldMTiles>([&](auto M) {
      static_for<4>([&](auto N) { static_for<4>([&](auto I) { st.acc[J][M][N][I] *= f[J]; }); });
    });
    f[J] = 1.0f;
    mma_step<P, J>(st, o);
  });
  int longest = 0;
  static_for<P::kFoldLists>(
      [&](auto J) { longest = max(longest, list_span<P>(smem, win, u, warp, J).y); });
  for (int k = ops::once(longest > 16); k > 0; --k) {
    static_for<P::kFoldLists>([&](auto J) {
      const int2 span = list_span<P>(smem, win, u, warp, J);
      const int n = span.y, steps = (n + 15) >> 4;
      const uint2* recs = list_records<P>(p, win, c, w, warp, J) + span.x;
#pragma unroll 1
      for (int s = 1; s < steps; ++s) {
        const uint2 v = recs[min(16 * s + (lane & 15), n - 1)];
        const bool live = (lane < 16) & (16 * s + lane < n);
        const uint2 rec = make_uint2(live ? v.x : 0u, live ? v.y : 0u);
        uint32_t off, cpair;
        pair_of<P>(rec, off, cpair);
        Operand<P> o;
        load_step<P>(o, pool, off, cpair, cf[J], 0, lane);
        mma_step<P, J>(st, o);
      }
    });
  }
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
  const UnitBook* book = reinterpret_cast<const UnitBook*>(smem + P::kBook) + warp * P::kFoldLists;
  static_for<P::kFoldLists>([&](auto J) {
    const int list = list_of<P::kFoldLists>(warp, J, win.lists);
    const float scale = pow2(book[J].unit - kCoefTop) * kByteScale;
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
