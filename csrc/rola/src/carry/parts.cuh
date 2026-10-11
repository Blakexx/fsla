// csrc/rola/src/carry/parts.cuh -- THE CARRY'S COMPONENTS, each at its real signature, tenants and copies: lists in,
// units in, publish, the pools' fills, the readout and fold passes and the drain. The MMAs and conversions are stubs
// (KERNEL_STANDARDS section 19's skeleton) that read what the real step reads. see docs/internals/carry/parts.md
#pragma once

#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

#include "carry/plan.cuh"
#include "common/ops.cuh"
#include "common/static_for.cuh"

namespace rola::carry {

namespace ops = ::denseref::ops;

//: A WARP'S STATE, the fold's accumulators: per list, its rows' m-tiles by the channels' four n-tiles, fp32 in the
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
  out.lists = out.listed ? p.extents[s.list] : 1;
  return out;
}

__device__ __forceinline__ int64_t list_row(const SideLists& l, const Place& c, int windows, int w,
                                            int digit) {
  return ((int64_t)c.bh * windows + w) * l.width + digit;
}

//: LISTS IN: a side's union tokens and count, and its box lists' pass starts, into the persistent tenants.
//: -- see docs/internals/carry/parts.md#lists-in
template <class P>
__device__ __forceinline__ void lists_in(const CarryParams& p, const SideLists& l,
                                         const Window& win, const Place& c, int w, uint8_t* smem,
                                         int side, int at_offset) {
  if (!win.listed) return;
  const int64_t group = ((int64_t)c.bh * p.windows + w) * l.groups + win.group;
  const uint16_t* tok = l.union_tok + group * kWindow;
  const ops::SmemAddr dst = ops::smem_addr(smem + P::kUnion + side * kWindow * 2);
#pragma unroll 1
  for (int k = (int)threadIdx.x; k < kWindow * 2 / 16; k += kThreads)
    ops::stage_tile(dst + 16 * k, tok + 8 * k);
  const ops::SmemAddr at = ops::smem_addr(smem + at_offset);
#pragma unroll 1
  for (int k = (int)threadIdx.x; k < win.lists; k += kThreads)
    ops::stage_tile(at + 16 * k, l.at + list_row(l, c, p.windows, w, win.digit0 + k) * kListAt);
  if (threadIdx.x == 0) reinterpret_cast<uint16_t*>(smem + P::kCount)[side] = l.union_count[group];
}

__device__ __forceinline__ int slots_of(const uint8_t* smem, int count_at, const Window& win,
                                        int side) {
  return win.listed ? reinterpret_cast<const uint16_t*>(smem + count_at)[side] : win.tokens;
}

//: UNITS IN: the window's tokens' readout units, the den pass's exponent bytes.
template <class P>
__device__ __forceinline__ void units_in(const CarryParams& p, const Place& c, int t0, int tokens,
                                         uint8_t* smem) {
  const int8_t* src = p.unit + (int64_t)c.bh * p.L + t0;
#pragma unroll 1
  for (int k = (int)threadIdx.x; k < tokens; k += kThreads)
    reinterpret_cast<int8_t*>(smem + P::kUnits)[k] = src[k];
}

//: ONE ASYNC RUN of 16, 8 or 4 bytes, the run a slice's or a list row's width gives.
__device__ __forceinline__ void stage_bytes(ops::SmemAddr dst, const void* src, int run) {
  // R9-CASES: a copy run, 16, 8 or 4 bytes (the host refuses anything under 4)
  if (run == ops::kStageBytes) {
    ops::stage_run<16>(dst, src);
  } else if (run == 8) {
    ops::stage_run<8>(dst, src);
  } else if (run == 4) {
    ops::stage_run<4>(dst, src);
  } else {
    __trap();
  }
}

//: A POOL'S FILL: a pass's slots, each its token's operand slices (then, on the write side, its values at the CTA's
//: channels) into its row; `cp.async`, landed by the caller's wait. -- see docs/internals/carry/parts.md#fill
template <int Factors, int Pitch, bool Values>
__device__ __forceinline__ void fill(const Slices<Factors>& sl, const uint8_t* values,
                                     int64_t values_st, int channels, const uint16_t* tok,
                                     const Window& win, int first, int slots, ops::SmemAddr pool) {
  //: the write rows' swizzle: the 16-byte halves of each 32-byte pair swapped on rows with bit 2 set
  const auto swz = [](int s) { return Values ? 16 * ((s >> 2) & 1) : 0; };
  static_for<Factors>([&](auto i) {
    const int bytes = sl.bytes[i];
    const int run = min(bytes, ops::kStageBytes);
    const int runs = bytes / run;
#pragma unroll 1
    for (int k = (int)threadIdx.x; k < slots * runs; k += kThreads) {
      const int s = k / runs, r = k - s * runs;
      const int t = win.listed ? tok[first + s] : first + s;
      const uint8_t* src = sl.at[i] + (win.t0 + t) * sl.st[i] + r * run;
      const ops::SmemAddr dst = pool + s * Pitch + ((i * kSliceBytes + r * run) ^ swz(s));
      stage_bytes(dst, src, run);
    }
  });
  if constexpr (Values) {
    const int runs = channels * 2 / ops::kStageBytes;
#pragma unroll 1
    for (int k = (int)threadIdx.x; k < slots * runs; k += kThreads) {
      const int s = k / runs, r = k - s * runs;
      const int t = win.listed ? tok[first + s] : first + s;
      ops::stage_run<16>(
          pool + s * Pitch + ((Factors * kSliceBytes + r * ops::kStageBytes) ^ swz(s)),
          values + (win.t0 + t) * values_st + r * ops::kStageBytes);
    }
  }
}

//: A WARP'S LIST: the box list its tile slot `j` maps to, warps sharing a list where the box has fewer.
template <int TileLists>
__device__ __forceinline__ int list_of(int warp, int j, int lists) {
  return (warp * TileLists + j) * lists / (kWarps * TileLists);
}

//: THE PUBLISH: the fold's fp32 state to the readout's f16 operand, through shared memory where the two tiles'
//: partitions differ. -- see docs/internals/carry/parts.md#publish
template <class P>
__device__ __forceinline__ void publish(const State<P>& st, uint32_t (&read)[P::kReadRegs],
                                        uint8_t* smem, int warp, int lane) {
  const auto pack = [&](auto k) {
    const __half2 h = __floats2half2_rn(st.template at<2 * k>(), st.template at<2 * k + 1>());
    return *reinterpret_cast<const uint32_t*>(&h);
  };
  if constexpr (P::kPublishRegs) {
    static_for<P::kReadRegs>([&](auto k) { read[k] = pack(k); });
    ops::rendezvous();
  } else {
    //: each round converts its share just before storing it, so one round's registers are live at a time
    constexpr int kRound = P::kReadRegs / P::kPublishRounds / 4;
    static_for<P::kPublishRounds>([&](auto R) {
      uint4* const mine = reinterpret_cast<uint4*>(smem + P::kPhase) + warp * kRound * 32 + lane;
      static_for<kRound>([&](auto k) {
        constexpr int r = 4 * (R * kRound + k);
        mine[32 * k] = make_uint4(
            pack(std::integral_constant<int, r>{}), pack(std::integral_constant<int, r + 1>{}),
            pack(std::integral_constant<int, r + 2>{}), pack(std::integral_constant<int, r + 3>{}));
      });
      ops::rendezvous();
      const uint4* const theirs = reinterpret_cast<const uint4*>(smem + P::kPhase)
                                  + (warp + 1) % kWarps * kRound * 32 + lane;
      static_for<kRound>([&](auto k) {
        constexpr int r = 4 * (R * kRound + k);
        const uint4 v = theirs[32 * k];
        read[r] = v.x;
        read[r + 1] = v.y;
        read[r + 2] = v.z;
        read[r + 3] = v.w;
      });
      ops::rendezvous();
    });
  }
}

//: A READOUT PASS, a warp's: its lists' entries in the pass, each slot's read row against the f16 state, the
//: products to the slot's panel words. Stub: one word of the row against the state's folded word, one shared add
//: an entry (the real step's register plan is its own component's).
//: -- see docs/internals/carry/parts.md#readout-pass
template <class P>
__device__ __forceinline__ void readout_pass(const CarryParams& p, uint32_t weight,
                                             const uint8_t* smem, const Window& win, const Place& c,
                                             int w, int pass, int slots, ops::SmemAddr pool,
                                             ops::SmemAddr panel, int warp, int lane) {
  const uint16_t* at = reinterpret_cast<const uint16_t*>(smem + P::kReadAt);
#pragma unroll 1
  for (int j = 0; j < P::kReadLists; ++j) {
    const int list = list_of<P::kReadLists>(warp, j, win.lists);
    const int begin = win.listed ? at[list * kListAt + pass] : 0;
    const int end = win.listed ? at[list * kListAt + pass + 1] : slots;
    const uint16_t* entries =
        reinterpret_cast<const uint16_t*>(p.read_lists.entries)
        + list_row(p.read_lists, c, p.windows, w, win.digit0 + list) * kListCap;
#pragma unroll 1
    for (int e = begin + lane; e < end; e += 32) {
      const int slot = win.listed ? entries[e] - pass * kPassSlots : e;
      uint32_t row;
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(row) : "r"(pool + slot * P::kReadPitch));
      ops::atom_shared_add_u32(panel + 4 * (slot * P::kPanelWords + (e & (P::kPanelWords - 1))),
                               (row ^ weight) & 1);
    }
  }
}

//: THE DRAIN: a pass's panel words to `num`, packed s16 pairs at each token's unit added as u32, the panel cleared.
//: -- see docs/internals/carry/parts.md#drain
template <class P>
__device__ __forceinline__ void drain(const CarryParams& p, const uint8_t* smem, const Window& win,
                                      const Place& c, int first, int slots, uint32_t* panel) {
  const uint16_t* tok = reinterpret_cast<const uint16_t*>(smem + P::kUnion);
#pragma unroll 1
  for (int k = (int)threadIdx.x; k < slots * P::kPanelWords; k += kThreads) {
    const int s = k / P::kPanelWords, word = k - s * P::kPanelWords;
    const uint32_t v = panel[k];
    const int t = win.listed ? tok[first + s] : first + s;
    if (v != 0u)
      atomicAdd(p.num + ((int64_t)c.bh * p.L + win.t0 + t) * (p.dv / 2) + c.channel0 / 2 + word, v);
    panel[k] = 0u;
  }
}

//: A FOLD UNIT'S STUB, for the paths the fold step does not take yet (a side with no list level, or two operand
//: levels): each entry's record and one word of its row's values, summed into the state.
//: -- see docs/internals/carry/fold.md#stub
template <class P>
__device__ __forceinline__ void fold_stub(const CarryParams& p, State<P>& st, const uint8_t* smem,
                                          const Window& win, const Place& c, int w, int u,
                                          int slots, ops::SmemAddr pool, int warp, int lane) {
  const uint16_t* at = reinterpret_cast<const uint16_t*>(smem + P::kWriteAt);
  float sum = 0.f;
#pragma unroll 1
  for (int j = 0; j < P::kFoldLists; ++j) {
    const int list = list_of<P::kFoldLists>(warp, j, win.lists);
    const int begin = win.listed ? at[list * kListAt + P::kFoldUnit * u] : 0;
    const int end = win.listed ? at[list * kListAt + P::kFoldUnit * (u + 1)] : slots;
    const uint2* records = reinterpret_cast<const uint2*>(p.write_lists.entries)
                           + list_row(p.write_lists, c, p.windows, w, win.digit0 + list) * kListCap;
#pragma unroll 1
    for (int e = begin + lane; e < end; e += 32) {
      const uint2 rec = win.listed ? records[e] : make_uint2(static_cast<uint32_t>(e), 0x3c003c00u);
      const uint32_t r = (rec.x & 0xffffu) & (P::kUnitSlots - 1);
      uint32_t row;
      asm volatile("ld.shared.u32 %0, [%1];"
                   : "=r"(row)
                   : "r"(pool + r * P::kWritePitch + P::kValuesAt));
      const __half2 cv = *reinterpret_cast<const __half2*>(&rec.y);
      const __half2 v = *reinterpret_cast<const __half2*>(&row);
      sum += __low2float(cv) * __low2float(v);
    }
  }
  st.acc[0][0][0][0] += sum;
}

}  // namespace rola::carry
