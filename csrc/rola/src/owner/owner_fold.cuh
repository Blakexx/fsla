// csrc/rola/src/owner/owner_fold.cuh -- THE FOLD: a warp's write lists over the window's passes into the resident
// state and mass, in the declared fp16 arithmetic -- a token's list factor c = b F at its list's window unit, level
// S's digit times it once in f16, one HMMA a (16 tokens, 8 channels) -- fp32 accumulate straight into the state; the
// passes pipelined through three write slots, each read in place.
// see docs/internals/owner/owner_fold.md
#pragma once

#include <cstdint>

#include "common/ops.cuh"
#include "common/static_for.cuh"
#include "facts/liveness_contract.cuh"
#include "owner/owner.cuh"
#include "owner/owner_fill.cuh"
#include "owner/owner_lists.cuh"

namespace rola::owner::detail {

//: THE UNITS' CONSTANTS: c's top 2^4, a state held under 2^100 of its unit, the lowest unit (a normal prescale), a list
//: never live, a window's largest add in units, the facts pass's dead exponent. -- see docs/internals/owner/owner_fold.md#units
constexpr int kCoefTop = 4;
constexpr int kUnitHeadroom = 100;
constexpr int kUnitMin = kCoefTop - 127;
constexpr int kUnitNone = -1000;
constexpr float kWindowAdd = kWindow * 16.0f;
constexpr int kExpDead = -126;

//: A SPARSE LEVEL S's DIGIT AS f16: `(1024 + b) 2^-8 - 4 = b / 256` from the byte under a 0x64 high byte.
//: -- see docs/internals/owner/owner_fold.md#operands
constexpr uint32_t kHalfMagic = 0x64646464u, kHalfScale = 0x1c001c00u, kHalfBias = 0xc400c400u;

//: A WARP'S ACCUMULATORS: per list the 16 leaves by the channel group, fp32 in the list's unit. The mass lives in
//: shared memory (`FoldTenants::mass`). -- see docs/internals/owner/owner_fold.md#state
template <class C>
struct FoldAcc {
  static_assert(C::RS == 16 && C::kFoldLists == 8, "the fold is built for the ruled 16 x 64 carve");
  float acc[C::kFoldLists][4][4];
};

//: THE LEAF OF AN ACCUMULATOR ROW: lane group `g`'s rows `g` and `g + 8` (`hi`) are leaves `g`, `g + 8` when level S's
//: digits are f16, and `2 g`, `2 g + 1` when its bytes are converted in registers. -- see docs/internals/owner/owner_fold.md#operands
template <class W>
__device__ __forceinline__ int acc_leaf(int g, int hi) {
  return W::kHalfS ? g + 8 * hi : 2 * g + hi;
}

//: A LIST'S WINDOW PRESCALES `2^(4 - unit)` (state, mass), held by its four lanes (list `lane >> 2`).
struct ListUnits {
  float pu = 1.0f, pv = 1.0f;
};

//: A LIST'S UNITS ACROSS WINDOWS, in shared memory: the state's and the mass's exponents and bounds (a window's step
//: never reads them). -- see docs/internals/owner/owner_fold.md#units
struct UnitBook {
  int u = kUnitNone, v = kUnitNone;
  float bu = 0.0f, bv = 0.0f;
};

//: THE FOLD'S SHARED TENANTS: every list's units and the CTA's mass (16 leaves x its lists, fp32 in each list's mass
//: unit; a warp owns its lists' columns).
template <class C>
struct FoldTenants {
  UnitBook units[kWarps][C::kFoldLists];
  float mass[C::RS][C::RD];
};

//: THE PASS PIPELINE'S TENANTS: three write slots, each warp's rows for its pass, and each slot's barriers -- `landed`
//: (every thread's fill copies arrive as they land) and `drained` (every thread arrives after its pass's last loads).
//: -- see docs/internals/owner/owner_fold.md#pipeline
template <class W>
struct FoldPool {
  WriteSlot<W> slot[kWriteSlots];
  uint16_t rows[kWarps][W::kPassRows];
  uint64_t landed[kWriteSlots], drained[kWriteSlots];
};

//: 2^e for an exponent in the normal range
__device__ __forceinline__ float pow2(int e) {
  return __uint_as_float(static_cast<uint32_t>(e + 127) << 23);
}

//: the smallest `e` with `x <= 2^e`, `kUnitNone` for zero
__device__ __forceinline__ int ceil_exp(float x) {
  const uint32_t bits = __float_as_uint(x);
  return bits == 0u ? kUnitNone : static_cast<int>(bits >> 23) - 127 + ((bits & 0x7fffffu) != 0u);
}

//: A UNIT'S NEXT WINDOW: a live list's exponent, held within the bound's headroom; the state's rescale returned, the
//: bound and prescale moved. -- see docs/internals/owner/owner_fold.md#units
__device__ __forceinline__ float next_unit(int k, int& unit, float& bound, float& pre) {
  const int held = unit + ceil_exp(bound) - kUnitHeadroom;
  const int next = max(k <= kExpDead ? unit : max(k, held), kUnitMin);
  const int d = unit - next;
  const float f = d < -126 ? 0.0f : pow2(d);
  bound = __fmaf_rn(bound, f, kWindowAdd);
  unit = next;
  pre = pow2(kCoefTop - next);
  return f;
}

//: A WINDOW'S UNITS, before its passes: each list's `(K_m, K_v)` from the facts pass moves its book's units (lane
//: `4 j` writes list j's back), the state and mass are rescaled to them, and the lanes take the window's prescales.
template <class C>
__device__ __forceinline__ void fold_window(FoldAcc<C>& st, ListUnits& lu, FoldTenants<C>& ft,
                                            const OwnerParams& p, int bh, int r0, int win, int warp,
                                            int lane) {
  const uint32_t pair =
      __ldg(p.list_exp + (static_cast<int64_t>(bh) * p.windows + win) * p.list_width + r0
            + warp * C::kFoldLists + (lane >> 2));
  UnitBook book = ft.units[warp][lane >> 2];
  const float fu = next_unit(static_cast<int16_t>(pair >> 16), book.u, book.bu, lu.pu);
  const float fv = next_unit(static_cast<int16_t>(pair & 0xffffu), book.v, book.bv, lu.pv);
  __syncwarp();
  if ((lane & 3) == 0) ft.units[warp][lane >> 2] = book;
  rola::static_for<C::kFoldLists>([&](auto J) {
    constexpr int j = decltype(J)::value;
    const float f = __shfl_sync(0xffffffffu, fu, 4 * j);
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) st.acc[j][n][i] *= f;
  });
  const int g = lane >> 2, tq = lane & 3, list = warp * C::kFoldLists + 2 * tq;
  const float f0 = __shfl_sync(0xffffffffu, fv, 8 * tq);
  const float f1 = __shfl_sync(0xffffffffu, fv, 8 * tq + 4);
  ft.mass[g][list] *= f0;
  ft.mass[g][list + 1] *= f1;
  ft.mass[g + 8][list] *= f0;
  ft.mass[g + 8][list + 1] *= f1;
}

//: A WARP'S ROWS FOR A PASS (union slots `[first, first + n)`): those whose token is in its union, ranked by a ballot,
//: and their count; a dense level R's are the pass's own. -- see docs/internals/owner/owner_fold.md#rows
template <class W>
__device__ __forceinline__ int pass_rows(const ListTenants<typename W::C>& sm, uint16_t* rows,
                                         int warp, int lane, int first, int n) {
  if constexpr (W::kHalfR) {
    return n;
  } else {
    const uint32_t below = (1u << lane) - 1u;
    constexpr int kGroups = W::kPassRows / 32;
    int at = 0;
    __syncwarp();
#pragma unroll
    for (int k = 0; k < kGroups; ++k) {
      const int i = 32 * k + lane;
      bool mine = false;
      if (i < n) {
        const int t = sm.token_of[0][first + i];
        mine = ((sm.partial[0][warp][t >> 5] >> (t & 31)) & 1u) != 0u;
      }
      const uint32_t ballot = __ballot_sync(0xffffffffu, mine);
      if (mine) rows[at + __popc(ballot & below)] = static_cast<uint16_t>(i);
      at += __popc(ballot);
    }
    __syncwarp();
    return at;
  }
}

//: A STEP'S ROWS: the lane's `ldmatrix` tokens' and its four c tokens', each c token's membership; past the warp's
//: count row 0 with c = 0. -- see docs/internals/owner/owner_fold.md#rows
template <class W>
struct StepRows {
  int a, b, c[4];
  bool in[4];

  __device__ __forceinline__ StepRows(const uint16_t* rows, int count, int e0, int lane) {
    const int r = lane & 7, q = lane >> 3, tq = lane & 3;
    const auto row = [&](int e, bool& in) {
      in = e < count;
      return in ? (W::kHalfR ? e : static_cast<int>(rows[e])) : 0;
    };
    bool unused;
    a = row(e0 + 8 * (q >> 1) + r, unused);
    b = row(e0 + 8 * (q & 1) + r, unused);
#pragma unroll
    for (int k = 0; k < 4; ++k) c[k] = row(e0 + 2 * tq + (k & 1) + 8 * (k >> 1), in[k]);
  }
};

//: A STEP'S OPERANDS: A = level S's digits (f16, or bytes converted in registers) and B = the values, in place in the
//: slot; the list's c at the lane's four tokens, for the values and the mass. -- see docs/internals/owner/owner_fold.md#operands
struct StepOperands {
  uint32_t a[4], b[4][2], cv[2], cm[2];
};

template <class W>
__device__ __forceinline__ void step_digits(const WriteSlot<W>& slot, const StepRows<W>& rows,
                                            int lane, uint32_t (&a)[4], uint32_t (&b)[4][2]) {
  const int q = lane >> 3;
  if constexpr (W::kHalfS) {
    ops::load_frag_t(a, ops::smem_addr(slot.rows + W::at(rows.a, W::kS / 16 + (q & 1))));
  } else {
    uint32_t x[2];
    ops::load_frag_t2(x, ops::smem_addr(slot.rows + W::at(rows.b, W::kS / 16)));
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      a[2 * k] = ops::hfma2(ops::prmt(x[k], kHalfMagic, 0x4240u), kHalfScale, kHalfBias);
      a[2 * k + 1] = ops::hfma2(ops::prmt(x[k], kHalfMagic, 0x4341u), kHalfScale, kHalfBias);
    }
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    uint32_t t[4];
    ops::load_frag_t(t,
                     ops::smem_addr(slot.rows + W::at(rows.b, W::kValues / 16 + 2 * h + (q >> 1))));
    b[2 * h][0] = t[0];
    b[2 * h][1] = t[1];
    b[2 * h + 1][0] = t[2];
    b[2 * h + 1][1] = t[3];
  }
}

//: A LIST'S c AT A LANE'S FOUR TOKENS to f16 pairs: `fp32(b F) x 2^(4 - unit)` for the values, or `fp32(b m)` at the
//: mass's unit (`kMass`), `b` the list's write coefficient on the byte scale (a sparse level R's byte, a dense one's
//: f16 times 255, exact) at byte `at` of the row. -- see docs/internals/owner/owner_fold.md#coefficient
template <class W, bool kMass>
__device__ __forceinline__ void list_coef(const WriteSlot<W>& slot, const StepRows<W>& rows, int at,
                                          float pre, uint32_t (&c)[2]) {
  float v[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const uint8_t* const digit = slot.rows + W::at(rows.c[k], at >> 4) + (at & 15);
    const float b =
        W::kHalfR ? facts::kRowMaxDigit * __half2float(*reinterpret_cast<const __half*>(digit))
                  : static_cast<float>(*digit);
    const float2 f = slot.fm[rows.c[k]];
    v[k] = rows.in[k] ? __fmul_rn(__fmul_rn(b, kMass ? f.y : f.x), pre) : 0.0f;
  }
  c[0] = ops::pack_half2(v[0], v[1]);
  c[1] = ops::pack_half2(v[2], v[3]);
}

//: A STEP'S OPERANDS LOADED: entries `[e0, e0 + 16)` of the warp's pass rows; the coefficient byte is the warp's lane
//: group's list, `at` in the row.
template <class W>
__device__ __forceinline__ void load_step(StepOperands& o, const WriteSlot<W>& slot,
                                          const uint16_t* rows, int count, int e0, int at,
                                          const ListUnits& lu, int lane) {
  const StepRows<W> sr(rows, count, e0, lane);
  step_digits(slot, sr, lane, o.a, o.b);
  list_coef<W, false>(slot, sr, at, lu.pu, o.cv);
  list_coef<W, true>(slot, sr, at, lu.pv, o.cm);
}

//: A LIST'S A: the dense digits times its c, two f16 products a register.
__device__ __forceinline__ void list_a(const uint32_t (&a)[4], uint32_t c01, uint32_t c89,
                                       uint32_t (&aj)[4]) {
  aj[0] = ops::hmul2(a[0], c01);
  aj[1] = ops::hmul2(a[1], c01);
  aj[2] = ops::hmul2(a[2], c89);
  aj[3] = ops::hmul2(a[3], c89);
}

//: A MERGED STEP, every list of the warp over the step's 16 tokens, two shuffles a list carrying c to the A layout; the
//: mass by one more atom into a fresh fp32 fragment summed into `m`. -- see docs/internals/owner/owner_fold.md#forms
template <class C>
__device__ __forceinline__ void merged_step(FoldAcc<C>& st, float (&m)[4], const StepOperands& o,
                                            int lane) {
  const int tq = lane & 3;
  float d[4];
  ops::hmma_fresh(d, o.a, o.cm[0], o.cm[1]);
#pragma unroll
  for (int i = 0; i < 4; ++i) m[i] += d[i];
  rola::static_for<C::kFoldLists>([&](auto J) {
    constexpr int j = decltype(J)::value;
    uint32_t aj[4];
    list_a(o.a, __shfl_sync(0xffffffffu, o.cv[0], 4 * j + tq),
           __shfl_sync(0xffffffffu, o.cv[1], 4 * j + tq), aj);
#pragma unroll
    for (int n = 0; n < 4; ++n) ops::hmma(st.acc[j][n], aj, o.b[n][0], o.b[n][1]);
  });
}

//: A PASS'S MASS SUMS into the warp's mass columns: lane (g, tq) owns its two leaves (`acc_leaf`) of lists `2 tq`,
//: `2 tq + 1`.
template <class W>
__device__ __forceinline__ void add_mass(FoldTenants<typename W::C>& ft, const float (&m)[4],
                                         int warp, int lane) {
  const int g = lane >> 2, list = warp * W::C::kFoldLists + 2 * (lane & 3);
  const int lo = acc_leaf<W>(g, 0), hi = acc_leaf<W>(g, 1);
  ft.mass[lo][list] += m[0];
  ft.mass[lo][list + 1] += m[1];
  ft.mass[hi][list] += m[2];
  ft.mass[hi][list + 1] += m[3];
}

//: A WARP'S FOLD OF PASS `g` (union slots `[first, first + n)`, slot `g % 3`): chunks of two steps' operands, the slot
//: drained after the last loads, the mid-pass `middle` fill after its chunk's MMAs; `lap(k)` books wait, rows, loads,
//: MMAs, middle, tail. -- see docs/internals/owner/owner_fold.md#pipeline
template <class W, class Middle, class Lap>
__device__ void fold_pass(FoldAcc<typename W::C>& st, FoldTenants<typename W::C>& ft,
                          FoldPool<W>& pool, const ListTenants<typename W::C>& sm,
                          const ListUnits& lu, int g, int first, int n, int warp, int lane,
                          Middle&& middle, Lap&& lap) {
  using C = typename W::C;
  const int s = g % kWriteSlots;
  ops::mbar_wait(ops::smem_addr(&pool.landed[s]), static_cast<uint32_t>(g / kWriteSlots) & 1u);
  lap(0);
  const uint16_t* const rows = pool.rows[warp];
  const int count = pass_rows<W>(sm, pool.rows[warp], warp, lane, first, n);
  const int steps = (count + 15) >> 4, mid = max(min(3, steps - 1), 0);
  const int at = W::kR + (W::kHalfR ? 2 : 1) * (warp * C::kFoldLists + (lane >> 2));
  const WriteSlot<W>& slot = pool.slot[s];
  float m[4] = {};
  lap(1);
#pragma unroll 1
  for (int s0 = 0; s0 == 0 || s0 < steps; s0 += 2) {
    StepOperands o0, o1;
    if (s0 < steps) load_step(o0, slot, rows, count, 16 * s0, at, lu, lane);
    if (s0 + 1 < steps) load_step(o1, slot, rows, count, 16 * (s0 + 1), at, lu, lane);
    if (s0 + 2 >= steps) ops::mbar_arrive(ops::smem_addr(&pool.drained[s]));
    lap(2);
    if (s0 < steps) merged_step(st, m, o0, lane);
    if (s0 + 1 < steps) merged_step(st, m, o1, lane);
    lap(3);
    if (s0 <= mid && mid < s0 + 2) middle();
    lap(4);
  }
  add_mass<W>(ft, m, warp, lane);
  lap(5);
}

}  // namespace rola::owner::detail
