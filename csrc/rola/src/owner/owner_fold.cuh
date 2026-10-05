// csrc/rola/src/owner/owner_fold.cuh -- THE FOLD: a warp's write lists over a pass of the write pool into the resident
// state and mass, in the declared fp16 arithmetic -- a token's list factor c = b F at its list's window unit, the dense
// digit times it once in f16, one HMMA a (16 tokens, 8 channels) -- fp32 accumulate straight into the state.
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
//: -- see docs/internals/owner/owner_fold.md#stage
constexpr uint32_t kHalfMagic = 0x64646464u, kHalfScale = 0x1c001c00u, kHalfBias = 0xc400c400u;

//: A WARP'S ACCUMULATORS: per list the 16 leaves by the channel group, fp32 in the list's unit. The mass lives in
//: shared memory (`FoldTenants::mass`). -- see docs/internals/owner/owner_fold.md#state
template <class C>
struct FoldAcc {
  static_assert(C::RS == 16 && C::kFoldLists == 8, "the fold is built for the ruled 16 x 64 carve");
  float acc[C::kFoldLists][4][4];
};

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

//: THE FOLD'S SHARED TENANTS: a pass's rows restaged for the operands -- the dense digits in f16 and the values, each at
//: a pitch an 8-row `ldmatrix` reads without conflict -- each row's `F` and `m`, every list's units, and the CTA's mass
//: (16 leaves x its lists, fp32 in each list's mass unit; a warp owns its lists' columns).
template <class C>
struct FoldTenants {
  static constexpr int kDensePitch = 24, kValuePitch = 40;
  uint16_t dense[kWritePool][kDensePitch];
  uint16_t vals[kWritePool][kValuePitch];
  float2 fm[kWritePool];
  UnitBook units[kWarps][C::kFoldLists];
  float mass[C::RS][C::RD];
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

//: A PASS'S ROWS RESTAGED, two threads a row: level S's digits as f16 (a dense level's copied, a sparse level's bytes
//: converted to `b / 256`) with the token's `(F, m)`, or the values copied; the rows past the pass to a whole step
//: zeroed, so a dense side's last step reads nothing but zeros there.
template <class W>
__device__ void stage_pass(FoldTenants<typename W::C>& ft, const WritePool<W>& pools, int n,
                           int tid) {
  const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
  const int rows = (n + 15) & ~15;
#pragma unroll 1
  for (int i = tid; i < 2 * rows; i += kThreads) {
    const int r = i >> 1;
    const bool live = r < n;
    const uint8_t* const row = pools.write[r];
    if (i & 1) {
      const uint4* const src = reinterpret_cast<const uint4*>(row + W::kValues);
      uint4* const dst = reinterpret_cast<uint4*>(&ft.vals[r][0]);
#pragma unroll
      for (int k = 0; k < 4; ++k) dst[k] = live ? src[k] : zero;
      continue;
    }
    ft.fm[r] = live ? *reinterpret_cast<const float2*>(row + W::kFm) : make_float2(0.0f, 0.0f);
    const uint4* const src = reinterpret_cast<const uint4*>(row + W::kS);
    uint4* const dst = reinterpret_cast<uint4*>(&ft.dense[r][0]);
    if constexpr (W::kHalfS) {
      dst[0] = live ? src[0] : zero;
      dst[1] = live ? src[1] : zero;
    } else {
      const uint4 d = live ? src[0] : zero;
      const uint32_t w[4] = {d.x, d.y, d.z, d.w};
      uint32_t h[8];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        h[2 * k] = ops::hfma2(ops::prmt(w[k], kHalfMagic, 0x4140u), kHalfScale, kHalfBias);
        h[2 * k + 1] = ops::hfma2(ops::prmt(w[k], kHalfMagic, 0x4342u), kHalfScale, kHalfBias);
      }
      dst[0] = make_uint4(h[0], h[1], h[2], h[3]);
      dst[1] = make_uint4(h[4], h[5], h[6], h[7]);
    }
  }
}

//: ENTRY `e` OF A RUN as a pass row, and whether it is in the run; past the end row 0, a valid row whose c is zero.
template <class C>
__device__ __forceinline__ int run_row(const ListTenants<C>& sm, int warp, int e, int end,
                                       int first, bool& in) {
  in = e < end;
  return in ? sm.entries[0][warp][e] - first : 0;
}

//: The end of a sorted run of entries in this pass: the first index from `at` whose slot is past `limit`
template <class C>
__device__ __forceinline__ int pass_end(const ListTenants<C>& sm, int warp, int at, int total,
                                        int limit, int lane) {
  int end = at;
#pragma unroll 1
  while (true) {
    const int e = end + lane;
    const bool in = e < total && sm.entries[0][warp][e] < limit;
    const uint32_t mask = __ballot_sync(0xffffffffu, in);
    end += __popc(mask);
    if (mask != 0xffffffffu) return end;
  }
}

//: A STEP'S ROWS: those of the lane's A and B `ldmatrix` tokens (`8 (q >> 1) + r`, `8 (q & 1) + r`) and of its four c
//: tokens (2 tq, 2 tq + 1, 2 tq + 8, 2 tq + 9), each c token's membership: in its run (a listed side's entries) or in
//: the pass (a dense side's, the step's own rows, the pass zero-padded). -- see docs/internals/owner/owner_fold.md#operands
template <class C, bool kListed>
struct StepRows {
  int a, b, c[4];
  bool in[4];

  __device__ __forceinline__ StepRows(const ListTenants<C>& sm, int warp, int lane, int e0, int end,
                                      int first) {
    const int r = lane & 7, q = lane >> 3, tq = lane & 3;
    if constexpr (kListed) {
      bool unused;
      a = run_row(sm, warp, e0 + 8 * (q >> 1) + r, end, first, unused);
      b = run_row(sm, warp, e0 + 8 * (q & 1) + r, end, first, unused);
#pragma unroll
      for (int k = 0; k < 4; ++k)
        c[k] = run_row(sm, warp, e0 + 2 * tq + (k & 1) + 8 * (k >> 1), end, first, in[k]);
    } else {
      const int base = e0 - first;
      a = base + 8 * (q >> 1) + r;
      b = base + 8 * (q & 1) + r;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        c[k] = base + 2 * tq + (k & 1) + 8 * (k >> 1);
        in[k] = c[k] < end - first;
      }
    }
  }
};

//: A STEP'S SHARED OPERANDS for its 16 tokens: the dense digits as A (leaves g and g + 8) and the values as B (four
//: n-tiles), by `ldmatrix.trans` from the restaged rows.
template <class C>
__device__ __forceinline__ void step_operands(const FoldTenants<C>& ft, int ra, int rb, int lane,
                                              uint32_t (&a)[4], uint32_t (&b)[4][2]) {
  const int q = lane >> 3;
  ops::load_frag_t(a, ops::smem_addr(&ft.dense[ra][8 * (q & 1)]));
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    uint32_t t[4];
    ops::load_frag_t(t, ops::smem_addr(&ft.vals[rb][16 * h + 8 * (q >> 1)]));
    b[2 * h][0] = t[0];
    b[2 * h][1] = t[1];
    b[2 * h + 1][0] = t[2];
    b[2 * h + 1][1] = t[3];
  }
}

//: A LIST'S c AT A LANE'S FOUR TOKENS to f16 pairs: `fp32(b F) x 2^(4 - unit)` for the values, or `fp32(b m)` at the
//: mass's unit (`kMass`), `b` the list's write coefficient on the byte scale (a sparse level R's byte, a dense one's
//: f16 times 255, exact). -- see docs/internals/owner/owner_fold.md#coefficient
template <class W, bool kMass, bool kListed>
__device__ __forceinline__ void list_coef(const WritePool<W>& pools,
                                          const FoldTenants<typename W::C>& ft,
                                          const StepRows<typename W::C, kListed>& rows, int digit,
                                          float pre, uint32_t (&c)[2]) {
  float v[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const uint8_t* const run = pools.write[rows.c[k]] + W::kR;
    const float b =
        W::kHalfR ? facts::kRowMaxDigit * __half2float(reinterpret_cast<const __half*>(run)[digit])
                  : static_cast<float>(run[digit]);
    const float2 f = ft.fm[rows.c[k]];
    v[k] = rows.in[k] ? __fmul_rn(__fmul_rn(b, kMass ? f.y : f.x), pre) : 0.0f;
  }
  c[0] = ops::pack_half2(v[0], v[1]);
  c[1] = ops::pack_half2(v[2], v[3]);
}

//: A LIST'S A: the dense digits times its c, two f16 products a register.
__device__ __forceinline__ void list_a(const uint32_t (&a)[4], uint32_t c01, uint32_t c89,
                                       uint32_t (&aj)[4]) {
  aj[0] = ops::hmul2(a[0], c01);
  aj[1] = ops::hmul2(a[1], c01);
  aj[2] = ops::hmul2(a[2], c89);
  aj[3] = ops::hmul2(a[3], c89);
}

//: A MERGED STEP, every list of the warp over the step's 16 tokens: the operands once, lane (g, tq) forms list g's c,
//: two shuffles a list carry it to the A layout; the mass by one more atom (A the dense digits, B the lists' `c_m`, lane
//: (g, tq)'s own pair for list g) into a fresh fp32 fragment added by FADD into the pass's sums `m` -- no accumulation
//: chain, so the all-positive terms keep fp32's rounding. -- see docs/internals/owner/owner_fold.md#forms
template <class W, bool kListed>
__device__ __forceinline__ void merged_step(FoldAcc<typename W::C>& st, float (&m)[4],
                                            const FoldTenants<typename W::C>& ft,
                                            const WritePool<W>& pools,
                                            const ListTenants<typename W::C>& sm,
                                            const ListUnits& lu, int warp, int lane, int e0,
                                            int end, int first) {
  using C = typename W::C;
  const int tq = lane & 3, digit = warp * C::kFoldLists + (lane >> 2);
  const StepRows<C, kListed> rows(sm, warp, lane, e0, end, first);
  uint32_t a[4], b[4][2], cv[2], cm[2];
  step_operands(ft, rows.a, rows.b, lane, a, b);
  list_coef<W, false>(pools, ft, rows, digit, lu.pu, cv);
  list_coef<W, true>(pools, ft, rows, digit, lu.pv, cm);
  float d[4];
  ops::hmma_fresh(d, a, cm[0], cm[1]);
#pragma unroll
  for (int i = 0; i < 4; ++i) m[i] += d[i];
  rola::static_for<C::kFoldLists>([&](auto J) {
    constexpr int j = decltype(J)::value;
    uint32_t aj[4];
    list_a(a, __shfl_sync(0xffffffffu, cv[0], 4 * j + tq),
           __shfl_sync(0xffffffffu, cv[1], 4 * j + tq), aj);
#pragma unroll
    for (int n = 0; n < 4; ++n) ops::hmma(st.acc[j][n], aj, b[n][0], b[n][1]);
  });
}

//: A PASS'S MASS SUMS into the warp's mass columns: lane (g, tq) owns leaves g, g + 8 of lists `2 tq`, `2 tq + 1`.
template <class C>
__device__ __forceinline__ void add_mass(FoldTenants<C>& ft, const float (&m)[4], int warp,
                                         int lane) {
  const int g = lane >> 2, list = warp * C::kFoldLists + 2 * (lane & 3);
  ft.mass[g][list] += m[0];
  ft.mass[g][list + 1] += m[1];
  ft.mass[g + 8][list] += m[2];
  ft.mass[g + 8][list + 1] += m[3];
}

//: A WARP'S FOLD OF ONE PASS (rows `[first, first + n)` of the window's write union), merged: every list over every
//: row (a dense level R, its f16 rows) or over the warp union's entries in the pass, the cursor carrying the union's next entry across
//: the window's passes, the mass's sums added to the warp's columns at the end.
//: `lap(k)` books the pass's head (0), first step (1), other steps (2) and tail (3) on a part's ledger.
//: -- see docs/internals/owner/owner_fold.md#forms
template <class W, class Lap>
__device__ void fold_pass(FoldAcc<typename W::C>& st, FoldTenants<typename W::C>& ft,
                          const WritePool<W>& pools, const ListTenants<typename W::C>& sm,
                          const ListUnits& lu, int& cursor, int first, int n, int warp, int lane,
                          Lap&& lap) {
  const int limit = first + n;
  float m[4] = {};
  if constexpr (W::kHalfR) {
    lap(0);
    merged_step<W, false>(st, m, ft, pools, sm, lu, warp, lane, first, limit, first);
    lap(1);
#pragma unroll 1
    for (int e = first + 16; e < limit; e += 16)
      merged_step<W, false>(st, m, ft, pools, sm, lu, warp, lane, e, limit, first);
  } else {
    const int at0 = cursor;
    const int end = pass_end(sm, warp, at0, sm.total[0][warp], limit, lane);
    lap(0);
#pragma unroll 1
    for (int t = ops::once(at0 < end); t > 0; --t)
      merged_step<W, true>(st, m, ft, pools, sm, lu, warp, lane, at0, end, first);
    lap(1);
#pragma unroll 1
    for (int e = at0 + 16; e < end; e += 16)
      merged_step<W, true>(st, m, ft, pools, sm, lu, warp, lane, e, end, first);
    cursor = end;
  }
  lap(2);
  add_mass(ft, m, warp, lane);
  lap(3);
}

}  // namespace rola::owner::detail
