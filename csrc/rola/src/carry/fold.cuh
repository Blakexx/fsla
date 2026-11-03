// csrc/rola/src/carry/fold.cuh -- THE FOLD: a warp's write lists over a window's chunks into the fp32 state -- each
// list in its running unit, steps of 16 tokens and an 8-token tail, a step's A the slots' dense slice times the
// entries' c, its B their values, fp32-accumulate MMAs. see docs/internals/carry/fold.md
#pragma once

#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

#include "carry/parts.cuh"

namespace rola::carry {

//: A DENSE WRITE LEVEL'S BYTE SCALE: its f16 digits carry 1/255 of the byte scale c is formed on.
constexpr float kByteScale = 1.0f / 255.0f;

//: A WARP'S LIST `j`: the box's list `warp + 8 j`, as the image's stores lay the lists out.
__device__ __forceinline__ constexpr int fold_list(int warp, int j) { return warp + kWarps * j; }

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

//: A WRITE SIDE WITH NO LIST LEVEL'S WINDOW EXPONENT, every list's, from the window's shift of F (`gains_shift`: its
//: largest F in [2^-4, 2^-3) after it): F under 2^(-3 - shift) and a dense byte under 2^8 -- a bound, the dense level's
//: floor holding each list's largest within 2^13 of it. -- see docs/internals/carry/fold.md#units
__device__ __forceinline__ int dense_exp(int shift) { return 5 - shift; }

//: A WINDOW'S UNITS, before the warp's steps: each list's exponent moves its book (lane group `j` list `j`); its state's
//: rescale `f` (applied beside the list's first MMAs) and its c factor `2^(E - unit)` posted to the list's scale (the
//: `kScale` tenant, read by the window's fold chunks). A side with no list level keeps its unit.
//: -- see docs/internals/carry/fold.md#units
struct ListScale {
  float f;
  uint32_t cf;
};

template <class P>
__device__ __forceinline__ void fold_units(uint8_t* smem, const Window& ww, int w, int warp,
                                           int lane) {
  constexpr int kLanes = 32 / P::kFoldLists;
  const int j = lane / kLanes;
  const int16_t* exps =
      reinterpret_cast<const int16_t*>(smem + P::kWriteExp) + parity_of(w) * kBoxLists * 2;
  const int k = ww.listed ? exps[2 * fold_list(warp, j) + 1] : dense_exp(exps[1]);
  UnitBook* const book = reinterpret_cast<UnitBook*>(smem + P::kBook) + fold_list(warp, j);
  UnitBook b = *book;
  const float fl = next_unit(k, b);
  const int cexp = max(k, kUnitMin) - b.unit;
  __syncwarp();
  if (lane % kLanes == 0) {
    *book = b;
    reinterpret_cast<ListScale*>(smem + P::kScale)[fold_list(warp, j)] =
        ListScale{fl, pow2_half2(cexp)};
  }
  __syncwarp();
}

//: AN ENTRY'S LANE VALUES: its slot's row offset in the union (its swizzle folded in; the chunk's pool is taken back
//: by its first slot `first`) and its c paired with the next lane's, moved to the list's unit; where the list carries
//: the den (`Den`), its c_m (the record's high half) paired the same way -- one shuffle of the record's word for both.
//: A pad (past the run) is the chunk's first row with c = 0. -- see docs/internals/carry/fold.md#den
template <class P, bool Den>
__device__ __forceinline__ void entry_of(uint2 rec, bool live, int first, uint32_t cf,
                                         uint32_t& off, uint32_t& cpair, uint32_t& dpair) {
  const uint32_t s = live ? rec.x & 0xffffu : static_cast<uint32_t>(first);
  off = s * P::kWriteRow + 16u * chunk_at<P::kWriteRow>(s, 0);
  const uint32_t mine = live ? rec.y : 0u, next = __shfl_xor_sync(0xffffffffu, mine, 1);
  cpair = ops::hmul2(__byte_perm(mine, next, 0x5410), cf);
  if constexpr (Den) dpair = __byte_perm(mine, next, 0x7632);
}

//: A LIST'S DEN COLUMN over its steps: two m-tiles, the column in every one of B's (each lane its quad's rows).
//: -- see docs/internals/carry/fold.md#den
template <class P>
using DenColumn = float[P::kFoldMTiles][4];

//: A 16-TOKEN STEP into list `J`'s accumulators: its slots' dense slices (A, `ldmatrix.trans`, times c) and values
//: (B), eight fp32-accumulate MMAs; where the list carries the den (`Den`), its column's two more on the raw slices
//: (B the entries' c_m, every column). Lanes `0..15` hold its entries. -- see docs/internals/carry/fold.md#step
//: @burst
template <class P, int J, bool Den>
__device__ __forceinline__ void step16(State<P>& st, ops::SmemAddr pool, uint32_t off,
                                       uint32_t cpair, uint32_t dpair, DenColumn<P>& den,
                                       int lane) {
  const int r = lane & 7, q = lane >> 3, t = lane & 3;
  const uint32_t oa = __shfl_sync(0xffffffffu, off, 8 * (q >> 1) + r);
  const uint32_t ob = __shfl_sync(0xffffffffu, off, 8 * (q & 1) + r);
  const uint32_t c01 = __shfl_sync(0xffffffffu, cpair, 2 * t),
                 c89 = __shfl_sync(0xffffffffu, cpair, 2 * t + 8);
  uint32_t raw[P::kFoldMTiles][4], a[P::kFoldMTiles][4], b[4][2];
  static_for<P::kFoldMTiles>([&](auto M) {
    ops::load_frag_t(raw[M], chunk_addr<P::kWriteRow>(pool + oa, 2 * M + (q & 1)));
    a[M][0] = ops::hmul2(raw[M][0], c01);
    a[M][1] = ops::hmul2(raw[M][1], c01);
    a[M][2] = ops::hmul2(raw[M][2], c89);
    a[M][3] = ops::hmul2(raw[M][3], c89);
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
  if constexpr (Den) {
    const uint32_t d01 = __shfl_sync(0xffffffffu, dpair, 2 * t),
                   d89 = __shfl_sync(0xffffffffu, dpair, 2 * t + 8);
    static_for<P::kFoldMTiles>([&](auto M) { ops::hmma(den[M], raw[M], d01, d89); });
  }
}

//: AN 8-TOKEN TAIL into list `J`'s accumulators: m16n8k8 MMAs, half a step's tensor work; the den's column on the raw
//: slices where the list carries it. Lanes `0..7` hold its entries. -- see docs/internals/carry/fold.md#step
//: @burst
template <class P, int J, bool Den>
__device__ __forceinline__ void step8(State<P>& st, ops::SmemAddr pool, uint32_t off,
                                      uint32_t cpair, uint32_t dpair, DenColumn<P>& den, int lane) {
  const int r = lane & 7, q = lane >> 3, t = lane & 3;
  const uint32_t o = __shfl_sync(0xffffffffu, off, r), c01 = __shfl_sync(0xffffffffu, cpair, 2 * t);
  uint32_t raw[4], a[4], b[4];
  ops::load_frag_t(raw, chunk_addr<P::kWriteRow>(pool + o, q));
  ops::load_frag_t(b, chunk_addr<P::kWriteRow>(pool + o, P::kValuesAt / 16 + q));
#pragma unroll
  for (int k = 0; k < 4; ++k) a[k] = ops::hmul2(raw[k], c01);
  static_for<P::kFoldMTiles>([&](auto M) {
    static_for<4>([&](auto N) { ops::hmma_k8(st.acc[J][M][N], a[2 * M], a[2 * M + 1], b[N]); });
  });
  if constexpr (Den) {
    const uint32_t d01 = __shfl_sync(0xffffffffu, dpair, 2 * t);
    static_for<P::kFoldMTiles>(
        [&](auto M) { ops::hmma_k8(den[M], raw[2 * M], raw[2 * M + 1], d01); });
  }
}

//: A LIST'S STEPS over its `n` entries: 16-token steps while more than 8 remain, then an 8-token tail, `fetch` the
//: entry at an index; `off`, `cpair` and `dpair` arrive holding the first step's lanes (entry `lane & 15`), and each
//: step fetches the next's in its own block, so its loads issue under its MMAs. -- see docs/internals/carry/fold.md#step
template <class P, int J, bool Den, class Fetch>
__device__ __forceinline__ void steps(State<P>& st, uint32_t cf, ops::SmemAddr pool, int first,
                                      int n, int lane, uint32_t off, uint32_t cpair, uint32_t dpair,
                                      DenColumn<P>& den, Fetch fetch) {
  int s = 0;
  for (int k = __any_sync(0xffffffffu, n > 8) ? 1 : 0; k > 0; --k)
#pragma unroll 1
    do {
      step16<P, J, Den>(st, pool, off, cpair, dpair, den, lane);
      s += 16;
      const int e = s + (lane & 15);
      const uint2 rec = fetch(min(e, n - 1));
      entry_of<P, Den>(rec, e < n, first, cf, off, cpair, dpair);
    } while (__any_sync(0xffffffffu, n - s > 8));
  for (int tail = __any_sync(0xffffffffu, n - s > 0) ? 1 : 0; tail > 0; --tail)
    step8<P, J, Den>(st, pool, off, cpair, dpair, den, lane);
}

//: A FOLD CHUNK'S FILL, the copies issued: its rows (slices then values) at `buf`, its entries after them (a warp its
//: own lists' runs, `run` the chunk's). -- see docs/internals/carry/parts.md#fill
template <class P>
__device__ __forceinline__ void fold_fill(const CarryParams& p, const uint8_t* smem,
                                          const Window& ww, const Place& c, int w, const Run& run,
                                          int first, int slots, ops::SmemAddr buf, int warp,
                                          int lane) {
  const volatile Addr& addr = *reinterpret_cast<const volatile Addr*>(smem + P::kAddr);
  fill<P::kFoldFactors, true>(addr.slices[kWrite], addr.values, static_cast<int>(p.values.st),
                              union_of<P>(p, smem, kWrite, w), ww, first, slots, buf);
  const ops::SmemAddr ents = buf + static_cast<uint32_t>(fold_ents<P>(slots));
  if (ww.listed) {
    entries_fill<P>(p, p.write_lists, ww, c, w, run, ents, warp, lane);
  } else {
    //: a side with no list level: each slot's token's (F, m), 8 bytes, in the entries' place
    const float2* const src = p.factors + static_cast<int64_t>(c.bh) * p.L + ww.t0 + first;
#pragma unroll 1
    for (int k = ops::opaque(static_cast<int>(threadIdx.x)); k < slots; k += kThreads)
      ops::stage_run<8>(ents + 8u * static_cast<uint32_t>(k), src + k);
  }
}

//: A FOLD CHUNK, a warp's: each of its lists' run of the chunk (`run`, the chunk's) from the buffer's entries at
//: `ents`, in 16-token steps and an 8-token tail over the rows at `rows`, the list's scale read from its tenant and its
//: rescale applied before the window's first chunk (`rescale`) where its unit moved; a side with no list level every
//: slot of the chunk, c from the slot's row. -- see docs/internals/carry/fold.md#chunk
//: THE FOLD'S LAPS, an instrument: `laps(m)` at the chunk's marks (0 its lists' first steps' lanes read, `1 + 2 J` list
//: `J`'s rescale guarded, `2 + 2 J` its steps done); the kernel's are none. -- see docs/internals/carry/fold.md#laps
struct NoLaps {
  __device__ __forceinline__ void operator()(int) const {}
};

//: THE DEN LISTS a channel box carries, a mask over a warp's lists `j`: where `j boxes / 4` is the box.
//: -- see docs/internals/carry/fold.md#den
__device__ __forceinline__ uint32_t den_lists(int channel0, int dv) {
  const int box = channel0 / kBoxChannels, boxes = dv / kBoxChannels;
  uint32_t mask = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) mask |= (j * boxes / 4 == box ? 1u : 0u) << j;
  return mask;
}

//: A LIST'S DEN COLUMN INTO THE DEN STATE (`kDenState`), each row's value times `2^(K - kCoefTop)`: lane `4 g + t`
//: row `g + 8 t` (every lane holds its quad's four rows: the column is B's every column), one add and one integer max
//: into the digit's largest (`kDenMax`, a non-negative float's bits) a lane. -- see docs/internals/carry/fold.md#den
template <class P>
__device__ __forceinline__ void den_add(const uint8_t* smem, int list, const DenColumn<P>& den,
                                        int k, int lane) {
  static_assert(P::kFoldMTiles == 2, "a quad's lane `t` takes its row `g + 8 t`");
  const int t = lane & 3, digit = (lane >> 2) + 8 * t;
  const bool odd = (t & 1) != 0, high = (t & 2) != 0;
  const float x0 = odd ? den[0][2] : den[0][0], x1 = odd ? den[1][2] : den[1][0];
  const float x = high ? x1 : x0;
  const ops::SmemAddr a =
      ops::smem_addr(smem + P::kDenState) + 4u * static_cast<uint32_t>(den_state_word(list, digit));
  float v;
  asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(a));
  v = fmaf(x, pow2(k - kCoefTop), v);
  ops::store_shared_f32(a, v);
  asm volatile("red.shared.max.u32 [%0], %1;" ::"r"(ops::smem_addr(smem + P::kDenMax)
                                                    + 4u * static_cast<uint32_t>(digit)),
               "r"(__float_as_uint(v))
               : "memory");
}

//: A DENSE WRITE SIDE'S DEN a chunk: the box's den lists (`dens`, a contiguous run of a warp's lists `j`, so lists `8
//: j0 .. 8 j1`) against every digit at once -- a warp an m-tile of lists by every digit of `Den += Cm^T S`
//: over the chunk's slots, 16 a step: A the rows' coefficient slices (`ldmatrix.trans`) each made `c_m` as the facts
//: records are, B the slices; then into the den state and each digit's largest. -- see docs/internals/carry/fold.md#den
template <class P>
__device__ __forceinline__ void fold_den_dense(const uint8_t* smem, ops::SmemAddr rows,
                                               ops::SmemAddr ents, int slots, uint32_t dens,
                                               float p2m, int km, int warp, int lane) {
  const int j0 = __ffs(static_cast<int>(dens)) - 1, j1 = 32 - __clz(static_cast<int>(dens));
  const int m = (j0 >> 1) + (warp >> 1), n0 = 2 * (warp & 1);
  const int g = lane >> 2, t = lane & 3, r = lane & 7, q = lane >> 3;

  //: a slot's row, held to the chunk where `last` (a full step's slots all live; 16 on keeps a row's swizzle)
  const auto row_of = [&](int k, bool last) {
    const uint32_t s = static_cast<uint32_t>(last ? min(k, slots - 1) : k);
    return rows + s * P::kWriteRow + 16u * chunk_at<P::kWriteRow>(s, 0);
  };

  //: a token pair's `m` (its entries' second word), 0 past the chunk where `last`
  const auto m_pair = [&](int k, bool last) {
    float4 v;
    asm volatile(
        "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "r"(ents + 8u * static_cast<uint32_t>(last ? min(k, slots - 2 + (slots & 1)) : k)));
    return last ? make_float2(k < slots ? v.y : 0.0f, k + 1 < slots ? v.w : 0.0f)
                : make_float2(v.y, v.w);
  };

  //: the warp's tile, where the box carries den lists and the warp's m-tile is among theirs (warp-uniform: shuffled
  //: trips, so ptxas fences no collective after them)
  for (int live = __any_sync(0xffffffffu, dens != 0u && m < ((j1 + 1) >> 1)) ? 1 : 0; live > 0;
       --live) {
    float acc[2][4] = {};
    //: A STEP of 16 slots from `kk`: its rows' A (each made `c_m`) and B, two MMAs; `last` the chunk's partial step
    //: (its rows held to the chunk, a slot past it `m` 0), else every slot live and the addresses `16 kWriteRow` on
    const auto step = [&](int kk, bool last) {
      uint32_t a[4], b[4];
      ops::load_frag_t(a, chunk_addr<P::kWriteRow>(row_of(kk + r + 8 * (q >> 1), last),
                                                   kSliceBytes / 16 + 2 * m + (q & 1)));
      ops::load_frag_t(b,
                       chunk_addr<P::kWriteRow>(row_of(kk + r + 8 * (q & 1), last), n0 + (q >> 1)));
      const float2 mk[2] = {m_pair(kk + 2 * t, last), m_pair(kk + 8 + 2 * t, last)};
      static_for<4>([&](auto R) {
        const float2 bb = __half22float2(*reinterpret_cast<const __half2*>(&a[R]));
        const float2 mm = mk[R >> 1];
        const __half2 c =
            __floats2half2_rn(__fmul_rn(__fmul_rn(__fmul_rn(bb.x, 255.0f), mm.x), p2m),
                              __fmul_rn(__fmul_rn(__fmul_rn(bb.y, 255.0f), mm.y), p2m));
        a[R] = *reinterpret_cast<const uint32_t*>(&c);
      });
      ops::hmma(acc[0], a, b[0], b[1]);
      ops::hmma(acc[1], a, b[2], b[3]);
    };

    const int full = 16 * (slots >> 4);
    int kk = 0;
    for (int go = __any_sync(0xffffffffu, kk < full) ? 1 : 0; go > 0; --go)
#pragma unroll 1
      do {
        step(kk, false);
        kk += 16;
      } while (__any_sync(0xffffffffu, kk < full));
    for (int tail = __any_sync(0xffffffffu, kk < slots) ? 1 : 0; tail > 0; --tail) step(kk, true);

    //: rows `16 m + g` (and `+ 8`) in the box's lists, digits `8 n + 2 t` (and `+ 1`)
    const float f = pow2(km - kCoefTop);
    static_for<2>([&](auto N) {
      static_for<4>([&](auto I) {
        const int list = 16 * m + g + 8 * (I >> 1), digit = 8 * (n0 + N) + 2 * t + (I & 1);
        const bool mine = list >= 8 * j0 && list < 8 * j1;
        const ops::SmemAddr at = ops::smem_addr(smem + P::kDenState)
                                 + 4u * static_cast<uint32_t>(den_state_word(list, digit));
        float v;
        asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(at));
        v = fmaf(acc[N][I], f, v);
        ops::store_shared_f32_if(at, v, mine);
        asm volatile("red.shared.max.u32 [%0], %1;" ::"r"(ops::smem_addr(smem + P::kDenMax)
                                                          + 4u * static_cast<uint32_t>(digit)),
                     "r"(mine ? __float_as_uint(v) : 0u)
                     : "memory");
      });
    });
  }
}

template <class P, class Laps = NoLaps>
__device__ __forceinline__ void fold_chunk(State<P>& st, bool rescale, const uint8_t* smem,
                                           const Window& ww, int w, const Run& run, int first,
                                           int slots, ops::SmemAddr rows, ops::SmemAddr ents,
                                           int warp, int lane, uint32_t dens, Laps laps = {}) {
  const ops::SmemAddr pool = rows - static_cast<uint32_t>(first * P::kWriteRow);
  //: a side with no list level: 2^(4 - E) of the window's exponents of F and m (`dense_exp`), the facts records' scale
  const int16_t* const exps =
      reinterpret_cast<const int16_t*>(smem + P::kWriteExp) + parity_of(w) * kBoxLists * 2;
  const float p2 = ww.listed ? 1.0f : ldexpf(1.0f, 4 - dense_exp(exps[1])),
              p2m = ww.listed ? 1.0f : ldexpf(1.0f, 4 - dense_exp(exps[0]));
  //: THE WARP'S LISTS: every list's scale, count, base and first step's lanes before any list's MMAs (four independent
  //: chains), then each list's rescale and steps; `fetch(base, k)` a list's entry `k`
  //: `listed`: the lists' steps may carry the den (a side with no list level's is its own pass, `fold_den_dense`)
  const auto lists = [&](const int(&n)[P::kFoldLists], const int(&base)[P::kFoldLists], auto fetch,
                         auto listed) {
    constexpr int kLists = P::kFoldLists;
    ListScale sc[kLists];
    uint32_t off[kLists], cpair[kLists], dpair[kLists];
    static_for<kLists>([&](auto J) {
      sc[J] = reinterpret_cast<const ListScale*>(smem + P::kScale)[fold_list(warp, J)];
    });
    static_for<kLists>([&](auto J) {
      const int e = lane & 15;
      const uint2 rec = fetch(base[J], max(min(e, n[J] - 1), 0));
      entry_of<P, decltype(listed)::value>(rec, e < n[J], first, sc[J].cf, off[J], cpair[J],
                                           dpair[J]);
    });

    laps(0);
    static_for<kLists>([&](auto J) {
      const float f = rescale ? sc[J].f : 1.0f;
      //: a list whose unit stood multiplies nothing (one trip where it moved; `f` is the warp's)
      for (int k = __any_sync(0xffffffffu, f != 1.0f) ? 1 : 0; k > 0; --k)
        static_for<P::kFoldMTiles>([&](auto M) {
          static_for<4>([&](auto N) { static_for<4>([&](auto I) { st.acc[J][M][N][I] *= f; }); });
        });
      laps(1 + 2 * J);
      //: a list whose den the box carries (the CTA's, a uniform branch): its column, then into the den state
      const auto list_fetch = [&](int k) { return fetch(base[J], k); };
      if constexpr (decltype(listed)::value) {
        if (((dens >> J) & 1u) != 0u) {
          DenColumn<P> den{};
          steps<P, J, true>(st, sc[J].cf, pool, first, n[J], lane, off[J], cpair[J], dpair[J], den,
                            list_fetch);
          den_add<P>(smem, fold_list(warp, J), den,
                     max(static_cast<int>(exps[2 * fold_list(warp, J)]), kUnitMin), lane);
        } else {
          DenColumn<P> none;
          steps<P, J, false>(st, sc[J].cf, pool, first, n[J], lane, off[J], cpair[J], 0u, none,
                             list_fetch);
        }
      } else {
        DenColumn<P> none;
        steps<P, J, false>(st, sc[J].cf, pool, first, n[J], lane, off[J], cpair[J], 0u, none,
                           list_fetch);
      }
      laps(2 + 2 * J);
    });
  };

  int n[P::kFoldLists], base[P::kFoldLists];
  if (ww.listed) {
    //: a list's count and place, each under 2^16, one shuffle
    const uint32_t packed = static_cast<uint32_t>(run.n) << 16 | static_cast<uint32_t>(run.to);
    static_for<P::kFoldLists>([&](auto J) {
      const uint32_t v = __shfl_sync(0xffffffffu, packed, fold_list(warp, J));
      n[J] = static_cast<int>(v >> 16);
      base[J] = static_cast<int>(v & 0xffffu);
    });
    lists(
        n, base,
        [&](int to, int k) {
          uint2 e;
          asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];"
                       : "=r"(e.x), "=r"(e.y)
                       : "r"(ents + 8u * static_cast<uint32_t>(to + k)));
          return e;
        },
        std::true_type{});
  } else {
    //: every slot of the chunk, c = f16(b F 2^(4 - E)) rounded once as the facts pass's records are: b the list's
    //: digit of the row's coefficient slice (the row's second) on the byte scale, F the slot's factor at `ents`
    static_for<P::kFoldLists>([&](auto J) {
      n[J] = slots;
      base[J] = fold_list(warp, J);
    });
    lists(
        n, base,
        [&](int list, int k) {
          const uint32_t row0 = rows + static_cast<uint32_t>(k) * P::kWriteRow
                                + 16u * chunk_at<P::kWriteRow>(static_cast<uint32_t>(k), 0);
          uint32_t cv;
          asm volatile("ld.shared.u16 %0, [%1];"
                       : "=r"(cv)
                       : "r"(chunk_addr<P::kWriteRow>(row0, (kSliceBytes + 2 * list) >> 4)
                             + (2 * list & 15)));
          float fk, mk;
          asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];"
                       : "=f"(fk), "=f"(mk)
                       : "r"(ents + 8u * static_cast<uint32_t>(k)));
          const float b = __half2float(__ushort_as_half(static_cast<unsigned short>(cv)));
          //: the packed conversion (F2FP, the ALU pipe; the scalar one runs on the conversion unit), the same rounding:
          //: c and the den's c_m, a record's two halves as the facts pass writes a listed side's
          const float b255 = __fmul_rn(b, 255.0f);
          const __half2 c = __floats2half2_rn(__fmul_rn(__fmul_rn(b255, fk), p2),
                                              __fmul_rn(__fmul_rn(b255, mk), p2m));
          return make_uint2(static_cast<uint32_t>(first + k),
                            static_cast<uint32_t>(__half_as_ushort(__low2half(c)))
                                | static_cast<uint32_t>(__half_as_ushort(__high2half(c))) << 16);
        },
        std::false_type{});
    fold_den_dense<P>(smem, rows, ents, slots, dens, p2m, dense_exp(exps[0]), warp, lane);
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
  if (p.probe == nullptr && p.probe_den == nullptr) return;
  const int64_t step = level_stride(p, -1);
  int64_t base = 0;
#pragma unroll 1
  for (int l = 0; l < p.D; ++l) base += origin_of(p, l) * level_stride(p, l);
  const int64_t list_stride = level_stride(p, win.listed ? p.write.list : p.write.operand[1]);
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
        if (p.probe != nullptr) {
          float* const out =
              p.probe + (static_cast<int64_t>(c.bh) * step + leaf) * p.dv + c.channel0;
          static_for<4>([&](auto N) {
            *reinterpret_cast<float2*>(out + 8 * N + 2 * tq) =
                make_float2(st.acc[J][M][N][2 * Hi] * scale, st.acc[J][M][N][2 * Hi + 1] * scale);
          });
        }
        //: the den state, by the box carrying the list (`den_lists`), the row's lane `tq` 0
        if (p.probe_den != nullptr && tq == 0 && ((den_lists(c.channel0, p.dv) >> J) & 1u))
          p.probe_den[static_cast<int64_t>(c.bh) * step + leaf] =
              reinterpret_cast<const float*>(smem + P::kDenState)[den_state_word(list, row)]
              * kByteScale;
      });
    });
  });
}

}  // namespace rola::carry
