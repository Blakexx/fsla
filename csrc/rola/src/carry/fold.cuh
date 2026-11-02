// csrc/rola/src/carry/fold.cuh -- THE FOLD: a warp's write lists over a window's chunks into the fp32 state -- each
// list in its running unit, steps of 16 tokens and an 8-token tail, a step's A the slots' dense slice times the
// entries' c, its B their values, fp32-accumulate MMAs. see docs/internals/carry/fold.md
#pragma once

#include <cuda_fp16.h>

#include <cstdint>

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
//: by its first slot `first`) and its c paired with the next lane's, moved to the list's unit. A pad (past the run) is
//: the chunk's first row with c = 0.
template <class P>
__device__ __forceinline__ void entry_of(uint2 rec, bool live, int first, uint32_t cf,
                                         uint32_t& off, uint32_t& cpair) {
  const uint32_t s = live ? rec.x & 0xffffu : static_cast<uint32_t>(first);
  off = s * P::kWriteRow + 16u * chunk_at<P::kWriteRow>(s, 0);
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

//: A LIST'S STEPS over its `n` entries: 16-token steps while more than 8 remain, then an 8-token tail, `fetch` the
//: entry at an index; `off` and `cpair` arrive holding the first step's lanes (entry `lane & 15`), and each step fetches
//: the next's in its own block, so its loads issue under its MMAs. -- see docs/internals/carry/fold.md#step
template <class P, int J, class Fetch>
__device__ __forceinline__ void steps(State<P>& st, uint32_t cf, ops::SmemAddr pool, int first,
                                      int n, int lane, uint32_t off, uint32_t cpair, Fetch fetch) {
  int s = 0;
  for (int k = __any_sync(0xffffffffu, n > 8) ? 1 : 0; k > 0; --k)
#pragma unroll 1
    do {
      step16<P, J>(st, pool, off, cpair, lane);
      s += 16;
      const int e = s + (lane & 15);
      entry_of<P>(fetch(min(e, n - 1)), e < n, first, cf, off, cpair);
    } while (__any_sync(0xffffffffu, n - s > 8));
  for (int tail = __any_sync(0xffffffffu, n - s > 0) ? 1 : 0; tail > 0; --tail)
    step8<P, J>(st, pool, off, cpair, lane);
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
template <class P>
__device__ __forceinline__ void fold_chunk(State<P>& st, bool rescale, const uint8_t* smem,
                                           const Window& ww, int w, const Run& run, int first,
                                           int slots, ops::SmemAddr rows, ops::SmemAddr ents,
                                           int warp, int lane) {
  const ops::SmemAddr pool = rows - static_cast<uint32_t>(first * P::kWriteRow);
  //: a side with no list level: 2^(4 - E) of the window's exponent (`dense_exp`), the facts records' scale
  const float p2 =
      ww.listed ? 1.0f
                : ldexpf(1.0f, 4
                                   - dense_exp(reinterpret_cast<const int16_t*>(
                                       smem + P::kWriteExp)[parity_of(w) * kBoxLists * 2 + 1]));
  //: THE WARP'S LISTS: every list's scale, count, base and first step's lanes before any list's MMAs (four independent
  //: chains), then each list's rescale and steps; `fetch(base, k)` a list's entry `k`
  const auto lists = [&](const int(&n)[P::kFoldLists], const int(&base)[P::kFoldLists],
                         auto fetch) {
    constexpr int kLists = P::kFoldLists;
    ListScale sc[kLists];
    uint32_t off[kLists], cpair[kLists];
    static_for<kLists>([&](auto J) {
      sc[J] = reinterpret_cast<const ListScale*>(smem + P::kScale)[fold_list(warp, J)];
    });
    static_for<kLists>([&](auto J) {
      const int e = lane & 15;
      entry_of<P>(fetch(base[J], max(min(e, n[J] - 1), 0)), e < n[J], first, sc[J].cf, off[J],
                  cpair[J]);
    });

    static_for<kLists>([&](auto J) {
      const float f = rescale ? sc[J].f : 1.0f;
      //: a list whose unit stood multiplies nothing (one trip where it moved; `f` is the warp's)
      for (int k = __any_sync(0xffffffffu, f != 1.0f) ? 1 : 0; k > 0; --k)
        static_for<P::kFoldMTiles>([&](auto M) {
          static_for<4>([&](auto N) { static_for<4>([&](auto I) { st.acc[J][M][N][I] *= f; }); });
        });
      steps<P, J>(st, sc[J].cf, pool, first, n[J], lane, off[J], cpair[J],
                  [&](int k) { return fetch(base[J], k); });
    });
  };

  int n[P::kFoldLists], base[P::kFoldLists];
  if (ww.listed) {
    static_for<P::kFoldLists>([&](auto J) {
      n[J] = __shfl_sync(0xffffffffu, run.n, fold_list(warp, J));
      base[J] = __shfl_sync(0xffffffffu, run.to, fold_list(warp, J));
    });
    lists(n, base, [&](int to, int k) {
      uint2 e;
      asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];"
                   : "=r"(e.x), "=r"(e.y)
                   : "r"(ents + 8u * static_cast<uint32_t>(to + k)));
      return e;
    });
  } else {
    //: every slot of the chunk, c = f16(b F 2^(4 - E)) rounded once as the facts pass's records are: b the list's
    //: digit of the row's coefficient slice (the row's second) on the byte scale, F the slot's factor at `ents`
    static_for<P::kFoldLists>([&](auto J) {
      n[J] = slots;
      base[J] = fold_list(warp, J);
    });
    lists(n, base, [&](int list, int k) {
      const uint32_t row0 = rows + static_cast<uint32_t>(k) * P::kWriteRow
                            + 16u * chunk_at<P::kWriteRow>(static_cast<uint32_t>(k), 0);
      uint32_t cv;
      asm volatile(
          "ld.shared.u16 %0, [%1];"
          : "=r"(cv)
          : "r"(chunk_addr<P::kWriteRow>(row0, (kSliceBytes + 2 * list) >> 4) + (2 * list & 15)));
      float fk;
      asm volatile("ld.shared.f32 %0, [%1];"
                   : "=f"(fk)
                   : "r"(ents + 8u * static_cast<uint32_t>(k)));
      const float b = __half2float(__ushort_as_half(static_cast<unsigned short>(cv)));
      //: the packed conversion (F2FP, the ALU pipe; the scalar one runs on the conversion unit), the same rounding
      const __half2 c = __floats2half2_rn(__fmul_rn(__fmul_rn(__fmul_rn(b, 255.0f), fk), p2), 0.0f);
      return make_uint2(static_cast<uint32_t>(first + k),
                        static_cast<uint32_t>(__half_as_ushort(__low2half(c))));
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
