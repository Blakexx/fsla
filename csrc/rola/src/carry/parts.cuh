// csrc/rola/src/carry/parts.cuh -- THE CARRY'S COMPONENTS at their real signatures, tenants and copies: a window's
// lists, facts and heads in, the ring's fills, the hand-off (the digit maxima, a part's stores), the readout and the
// drain. The skeleton's parts (KERNEL_STANDARDS section 19) keep the work's counts and leave named math to their
// components. see docs/internals/carry/parts.md
#pragma once

#include <cuda_fp16.h>

#include <cstdint>

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

//: A SIDE'S OPERAND SLICES for this CTA: a level's slice at the box's digits, `bytes` a token's, at `st` a token.
template <int Factors>
struct Slices {
  const uint8_t* at[Factors];
  int64_t st[Factors];
  int bytes[Factors];
};

template <int Factors>
__device__ __forceinline__ Slices<Factors> slices_of(const CarryParams& p, const Side& s,
                                                     const Place& c) {
  Slices<Factors> out;
  static_for<Factors>([&](auto i) {
    const int level = s.operand[i];
    const Plane& pl = s.rows[i];
    out.at[i] = pl.base + c.bh / p.H * pl.sb + c.bh % p.H * pl.sh
                + (int64_t)origin_of(p, level) * s.elem[i];
    out.st[i] = pl.st;
    out.bytes[i] = p.extents[level] * s.elem[i];
  });
  return out;
}

//: A SIDE'S WINDOW: its union (the list level's 32-digit group holding the box's digits) and its box lists' first
//: digit, or every window token a slot where the side has no list level.
struct Window {
  int tokens, t0, group, digit0, lists;
  bool listed;
};

__device__ __forceinline__ Window window_of(const CarryParams& p, const Side& s, int w) {
  Window out;
  out.t0 = w * kWindow;
  out.tokens = min(kWindow, p.L - out.t0);
  out.listed = s.list >= 0;
  out.digit0 = out.listed ? origin_of(p, s.list) : 0;
  out.group = out.digit0 / 32;
  out.lists = out.listed ? p.extents[s.list] : kBoxLists;
  return out;
}

__device__ __forceinline__ int64_t list_row(const SideLists& l, const Place& c, int windows, int w,
                                            int digit) {
  return ((int64_t)c.bh * windows + w) * l.width + digit;
}

//: THE SIDES, as the tenants index them: the read side 0, the write side 1.
enum SideIndex : int { kRead = 0, kWrite = 1 };

//: A SIDE'S TENANTS for window `w` (two windows' a side, by parity): its union tokens, its count's word pair, a
//: list's pass starts. -- see docs/internals/carry/plan.md#tenants
template <class P>
__device__ __forceinline__ const uint16_t* union_of(const uint8_t* smem, int side, int w) {
  return reinterpret_cast<const uint16_t*>(smem + P::kUnion) + (2 * side + (w & 1)) * kWindow;
}

template <class P>
__device__ __forceinline__ const uint16_t* at_of(const uint8_t* smem, int side, int w, int list) {
  return reinterpret_cast<const uint16_t*>(smem + P::kAt)
         + ((2 * side + (w & 1)) * kBoxLists + list) * kListAt;
}

//: a side's slots in window `w`: its union's count, or every token where the side has no list level
template <class P>
__device__ __forceinline__ int slots_of(const uint8_t* smem, const Window& win, int side, int w) {
  return win.listed ? reinterpret_cast<const int*>(smem + P::kCount)[2 * side + (w & 1)]
                    : win.tokens;
}

//: LISTS IN, a window ahead: a side's union tokens and count and its box lists' pass starts.
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
  if (t >= kUnionCopies && t < kUnionCopies + win.lists)
    ops::stage_tile(ops::smem_addr(smem + P::kAt) + 16 * (slot * kBoxLists + t - kUnionCopies),
                    l.at + list_row(l, c, p.windows, w, win.digit0 + t - kUnionCopies) * kListAt);
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

//: HEADS IN, a window ahead and a warp's own: its 4 lists' (or digits') first 16 entries, 16 bytes a lane, copied
//: once the warp has read the last window's. -- see docs/internals/carry/parts.md#heads-in
__device__ __forceinline__ void heads_in(const SideLists& l, const Window& win, const Place& c,
                                         int windows, int w, ops::SmemAddr heads, int warp,
                                         int lane) {
  if (!win.listed) return;
  const int64_t block = (((int64_t)c.bh * windows + w) * l.groups + win.group) * kWarps + warp;
  ops::stage_tile(heads + 16 * lane,
                  reinterpret_cast<const uint8_t*>(l.heads + block * 4 * kHeadEntries) + 16 * lane);
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

//: A RING FILL: `slots` union slots from `first`, each its token's row -- its side's operand slices at 64 bytes each,
//: then (`Values`) its values at the box's channels -- swizzled, by `cp.async`, landing on the slot's barrier; a gap
//: slot's row zero-filled. -- see docs/internals/carry/parts.md#fill
template <int Factors, bool Values>
__device__ __forceinline__ void fill(const Slices<Factors>& sl, const uint8_t* values,
                                     int64_t values_st, const uint16_t* tok, const Window& win,
                                     int first, int slots, ops::SmemAddr pool, ops::SmemAddr bar) {
  constexpr int kRow = kSliceBytes * Factors + (Values ? kBoxChannels * 2 : 0);
  const auto at = [&](int s, int byte) {
    return pool + s * kRow + 16 * chunk_at<kRow>(static_cast<uint32_t>(s), byte >> 4) + (byte & 15);
  };
  static_for<Factors>([&](auto i) {
    const int bytes = sl.bytes[i], run = min(bytes, ops::kStageBytes), runs = bytes / run;
#pragma unroll 1
    for (int k = static_cast<int>(threadIdx.x); k < slots * runs; k += kThreads) {
      const int s = k / runs, r = k - s * runs;
      const int u = win.listed ? tok[first + s] : first + s, t = u & (kWindow - 1);
      stage_bytes(at(s, i * kSliceBytes + r * run), sl.at[i] + (win.t0 + t) * sl.st[i] + r * run,
                  run, u == t);
    }
  });
  if constexpr (Values) {
    constexpr int kRuns = kBoxChannels * 2 / ops::kStageBytes;
#pragma unroll 1
    for (int k = static_cast<int>(threadIdx.x); k < slots * kRuns; k += kThreads) {
      const int s = k / kRuns, r = k - s * kRuns;
      const int u = win.listed ? tok[first + s] : first + s, t = u & (kWindow - 1);
      ops::stage_run_if<16>(at(s, Factors * kSliceBytes + r * ops::kStageBytes),
                            values + (win.t0 + t) * values_st + r * ops::kStageBytes, u == t);
    }
  }
  ops::mbar_arrive_when_landed(bar);
}

//: A WARP'S LISTS' UNITS, list `j` the box's list `warp + 8 j`.
template <class P>
__device__ __forceinline__ void list_units(int (&u)[P::kFoldLists], const uint8_t* smem, int warp) {
  const UnitBook* const book = reinterpret_cast<const UnitBook*>(smem + P::kBook);
  static_for<P::kFoldLists>([&](auto J) { u[J] = book[warp + kWarps * J].unit; });
}

//: THE DIGIT MAXIMA, the hand-off's first step: a warp's largest |state| a digit over its lists, each list's at its
//: unit, as an exponent -- `ceil_exp` of the list's largest |accumulator| plus its unit, the largest over the lists --
//: the four lanes of a digit row joined by two shuffles, posted for the CTA. -- see docs/internals/carry/parts.md#maxima
template <class P>
__device__ __forceinline__ void post_maxima(const State<P>& st, uint8_t* smem, int warp, int lane) {
  int* const posts = reinterpret_cast<int*>(smem + P::kMaxima) + warp * kBoxDigits;
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
      if ((lane & 3) == 0) posts[16 * M + 8 * H + (lane >> 2)] = e;
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
//: the 8 warps' posts, at least `kImageTop - 127`.
template <class P>
__device__ __forceinline__ int digit_exp(const uint8_t* smem, int d) {
  const int* posts = reinterpret_cast<const int*>(smem + P::kMaxima) + d;
  int e = kImageTop - 127;
  static_for<kWarps>([&](auto W) { e = max(e, posts[W * kBoxDigits]); });
  return e;
}

//: A PART'S STORES, the hand-off: the warp's lists' rows of the part's digits into the image, f16 at each digit's
//: power of two, each list moved there from its unit (`2^(unit + kImageTop - e)`, held to a normal float's range: past
//: it the f16 is zero or the list's value is not the digit's); warp 0 records the part's exponents for the readers.
//: -- see docs/internals/carry/parts.md#store
template <class P, int Part>
__device__ __forceinline__ void store_part(const State<P>& st, uint8_t* smem, int warp, int lane) {
  constexpr int kTiles = P::kFoldMTiles / P::kParts;
  const int g = lane >> 2, t = lane & 3;
  const ops::SmemAddr image = ops::smem_addr(smem + P::kImage);
  int u[P::kFoldLists];
  list_units<P>(u, smem, warp);
  static_for<kTiles>([&](auto K) {
    constexpr int m = Part * kTiles + K;
    static_for<2>([&](auto H) {
      const int d = 16 * m + 8 * H + g;
      const int e = digit_exp<P>(smem, d);
      if (warp == 0 && t == 0) reinterpret_cast<int*>(smem + P::kExps)[d] = e;
      const ops::SmemAddr plane = image + (d % P::kPlanes) * kPlaneBytes + 4 * t;
      static_for<P::kFoldLists>([&](auto J) {
        const float scale = pow2(min(max(u[J] + kImageTop - e, -126), 127));
        static_for<4>([&](auto N) {
          const __half2 h =
              __floats2half2_rn(st.acc[J][m][N][2 * H] * scale, st.acc[J][m][N][2 * H + 1] * scale);
          const uint32_t at = (plane + 16 * unit_of(warp + kWarps * J, 0, g)) ^ (16u * N);
          const uint32_t v = *reinterpret_cast<const uint32_t*>(&h);
          asm volatile("st.shared.u32 [%0], %1;" ::"r"(at), "r"(v) : "memory");
        });
      });
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

//: A DIGIT'S STATE FRAGMENTS from the image: `(m * 2 + kb) * 4 + i`, m the channel tile, kb the list half.
template <class P>
__device__ __forceinline__ void state_in(uint32_t (&f)[16], ops::SmemAddr image,
                                         const ImageLane& il, int d) {
  const ops::SmemAddr plane = image + (d % P::kPlanes) * kPlaneBytes;
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

//: A READOUT TILE, 8 tokens on N against a digit's state (the A operand): their read rows (B) by one `ldmatrix.x4`,
//: four MMAs, each lane's two tokens' outputs as exact pairs at their panel rows; lane `l` holds entry `l & 7` and
//: scales its coefficient to its token's unit by `2^(sx - u)` (`sx` the digit's exponent less `kImageTop` and
//: `kCoefTop`). -- see docs/internals/carry/parts.md#readout
//: @burst
template <class P>
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
  ops::load_frag(b,
                 chunk_addr<P::kReadRow>(pool + (e.x & 0xffffu), static_cast<uint32_t>(lane >> 3)));
  float acc[2][4] = {};
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int kb = 0; kb < 2; ++kb) {
      const uint32_t a[4] = {f[(m * 2 + kb) * 4], f[(m * 2 + kb) * 4 + 1], f[(m * 2 + kb) * 4 + 2],
                             f[(m * 2 + kb) * 4 + 3]};
      ops::hmma(acc[m], a, b[2 * kb], b[2 * kb + 1]);
    }
  const uint32_t b0 = pads + (e0 >> 16), b1 = pads + (e1 >> 16);
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    panel_red((b0 ^ (32u * m)) + 4u * g, s16_pair(acc[m][0], acc[m][2], c0));
    panel_red((b1 ^ (32u * m)) + 4u * g, s16_pair(acc[m][1], acc[m][3], c1));
  }
}

//: THE READOUT OF A PART, a warp's: each of its digits `w + 8 i` in the part, its state from the image and its
//: exponent, its tiles over pass `q` -- a listed side's entries (the first 16 of pass 0 from the warp's heads, the rest
//: in place), else every slot of the pass, the coefficient the digit's of the slot's row's coefficient slice (its
//: second). -- see docs/internals/carry/parts.md#readout
template <class P>
__device__ __forceinline__ void readout_part(const CarryParams& p, const uint8_t* smem,
                                             const Window& rw, const Place& c, int w, int q,
                                             int slots, int part, ops::SmemAddr pool, int warp,
                                             int lane) {
  constexpr int kDigits = P::kReadDigits / P::kParts;
  const ImageLane il = image_lane(lane);
  const ops::SmemAddr image = ops::smem_addr(smem + P::kImage),
                      pads = ops::smem_addr(smem + P::kPads);
  const ops::SmemAddr heads = ops::smem_addr(smem + P::kReadHeads) + warp * P::kHeadBytes;
  const ops::SmemAddr units = ops::smem_addr(smem + P::kUnits) + (w & 1) * kWindow;
#pragma unroll 1
  for (int k = 0; k < kDigits; ++k) {
    const int i = part * kDigits + k, d = warp + kWarps * i;
    const int sx = reinterpret_cast<const int*>(smem + P::kExps)[d] - kImageTop - kCoefTop;
    uint32_t f[16];
    state_in<P>(f, image, il, d);
    if (rw.listed) {
      const uint16_t* at = at_of<P>(smem, kRead, w, d);
      const int first = at[q], tiles = (at[q + 1] - first + 7) >> 3;
      const uint2* entries =
          p.read_lists.entries + list_row(p.read_lists, c, p.windows, w, rw.digit0 + d) * kListCap;
      //: pass 0's first tiles from the warp's heads, the rest in place: each loop's step one block
      const int staged = q == 0 ? min(tiles, kHeadEntries / 8) : 0;
      int s = 0;
      for (int k = __any_sync(0xffffffffu, s < staged) ? 1 : 0; k > 0; --k)
#pragma unroll 1
        do {
          uint2 e;
          asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];"
                       : "=r"(e.x), "=r"(e.y)
                       : "r"(heads + 8u * (kHeadEntries * i + 8 * s + (lane & 7))));
          tile<P>(pool, pads, units, sx, e, f, lane);
        } while (__any_sync(0xffffffffu, ++s < staged));
      for (int k = __any_sync(0xffffffffu, s < tiles) ? 1 : 0; k > 0; --k)
#pragma unroll 1
        do tile<P>(pool, pads, units, sx, entries[first + 8 * s + (lane & 7)], f, lane);
        while (__any_sync(0xffffffffu, ++s < tiles));
    } else {
      //: every slot of the pass, its entry implied: its row and token, its coefficient, its panel row (a slot past the
      //: pass's end its place's dummy row)
      int s = 0;
      for (int k = __any_sync(0xffffffffu, slots > 0) ? 1 : 0; k > 0; --k)
#pragma unroll 1
        do {
          const uint32_t slot = 8u * s + (lane & 7);
          const uint32_t panel =
              static_cast<int>(slot) < slots ? kPadBytes + panel_row(slot) : panel_row(slot & 7);
          const uint32_t row0 = slot * P::kReadRow + 16u * chunk_at<P::kReadRow>(slot, 0);
          uint32_t coef = 0x3c00u;
          if constexpr (P::kReadFactors > 1)
            asm volatile("ld.shared.u16 %0, [%1];"
                         : "=r"(coef)
                         : "r"(chunk_addr<P::kReadRow>(pool + row0, (kSliceBytes + 2 * d) >> 4)
                               + (2 * d & 15)));
          const uint2 e = make_uint2(row0 | (q * P::kPassSlots + slot) << 16, coef | (panel << 16));
          tile<P>(pool, pads, units, sx, e, f, lane);
        } while (__any_sync(0xffffffffu, ++s < (slots + 7) >> 3));
    }
  }
}

//: THE DRAIN: a pass's panel words to `num`, two tokens a request, packed s16 pairs at each token's unit added as
//: u32, each word cleared. -- see docs/internals/carry/parts.md#drain
template <class P>
__device__ __forceinline__ void drain(const CarryParams& p, uint8_t* smem, const Window& rw,
                                      const Place& c, int w, int first, int slots) {
  constexpr int kWords = kPanelRow / 4;
  uint8_t* const panel = smem + P::kPanel;
  const uint16_t* tok = union_of<P>(smem, kRead, w);
#pragma unroll 1
  for (int k = static_cast<int>(threadIdx.x); k < slots * kWords; k += kThreads) {
    const int s = k / kWords, word = k - s * kWords;
    uint32_t* const cell = reinterpret_cast<uint32_t*>(
        panel + ((panel_row(static_cast<uint32_t>(s)) ^ (32u * (word >> 3))) + 4u * (word & 7)));
    const uint32_t v = *cell;
    const int t = rw.listed ? tok[first + s] : first + s;
    if (v != 0u)
      atomicAdd(p.num + ((int64_t)c.bh * p.L + rw.t0 + t) * (p.dv / 2) + c.channel0 / 2 + word, v);
    *cell = 0u;
  }
}

}  // namespace rola::carry
