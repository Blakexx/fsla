// csrc/rola/src/owner/owner_fold.cuh -- THE FOLD: a warp's write lists over a pass of the write pool into the resident
// fp32 state, in the declared arithmetic -- each token's factor a 16-bit mantissa at a power-of-two exponent, a
// power-of-two scale a (list, leaf, pass) from the products' bit lengths, each coefficient rounded once into two u8
// digits, the digit products into three s32 sums by IMMA, and the pass's sums into the state.
// see docs/internals/owner/owner_fold.md
#pragma once

#include <cstdint>

#include "common/ops.cuh"
#include "common/static_for.cuh"
#include "owner/owner.cuh"
#include "owner/owner_fill.cuh"
#include "owner/owner_lists.cuh"

namespace rola::owner::detail {

//: A WARP'S FOLD STATE: per write list, the box's 16 leaves (M) by the channel group (N), fp32.
template <class C>
struct FoldState {
  static_assert(C::RS == 16, "the fold is built for the ruled 16 x 64 carve: one m-tile a list");
  float acc[C::kFoldLists][4][4];
};

//: THE FOLD'S SHARED TENANTS, a warp's own: each list's next entry across a window's passes, and every list's scale
//: bytes when the lists share a run (two lists a word, a lane's own).
template <class C>
struct FoldTenants {
  int cursor[kWarps][C::kFoldLists];
  uint32_t lens[kWarps][C::kFoldLists / 2][32];
};

//: THE FOLD'S PHASES on a phase clock: a list's head (cursors, runs), its scales, its first step's operands, its
//: bursts, its state add; a part or kernel numbers its own phases from `kFoldPhases`.
enum FoldPhase : int { kFoldHead = 0, kFoldScale, kFoldFirst, kFoldBursts, kFoldAdd, kFoldPhases };

//: THE ARITHMETIC'S CONSTANTS: the factor's range, the token factor's and the leaf scale's power-of-two offsets (their
//: product lands on fp32's denormal grid, so a rounded coefficient's bits are its integer), the scales' biases.
//: -- see docs/internals/owner/owner_fold.md#arithmetic
constexpr float kFactorFloor = 0x1p-64f;
constexpr int kFactorTop = 32;
constexpr int kTokenShift = -50, kLeafShift = -83;
constexpr int kExpoBias = 5;
constexpr uint32_t kBitsBias = 119, kTokenBias = 71, kLive = 128, kHalfFloor = 0x0400u;
//: 2^23 as fp32, a byte in its low mantissa bits read as 2^23 + u: the first `prmt` source, which is a register, so
//: the selector rides the immediate
constexpr uint32_t kMagic = 0x4b000000u;

//: a power of two as fp32, for an exponent in the normal range
__device__ __forceinline__ float pow2(int e) {
  return __uint_as_float(static_cast<uint32_t>(e + 127) << 23);
}

//: THE TOKEN FACTORS OF A PASS, a thread a pool row: `((g sw_S) sw_R) vs` as `m 2^e` (16-bit `m`, `e` in
//: `[-79, 32]`, below `2^-64` zero) stored back as `m 2^(e - 50)` in fp32, and the dense digits' bit lengths biased.
//: -- see docs/internals/owner/owner_fold.md#factor
template <class C>
__device__ void token_factors(WritePool<C>& pools, const OwnerParams& p, int n, int tid) {
#pragma unroll 1
  for (int r = tid; r < n; r += kThreads) {
    uint8_t* const row = pools.write[r];
    float* const f = reinterpret_cast<float*>(row + Rows<C>::kFactors);
    const float sw_s = p.level_s == 0 ? f[2] : f[3], sw_r = p.level_s == 0 ? f[3] : f[2];
    const float factor = __fmul_rn(__fmul_rn(__fmul_rn(f[0], sw_s), sw_r), f[1]);
    const uint32_t bits = __float_as_uint(factor);
    uint32_t m = (((bits & 0x7fffffu) | 0x800000u) + 128u) >> 8;
    int e = static_cast<int>(bits >> 23) - 142;
    if (m == 65536u) {
      m = 32768u;
      ++e;
    }
    if (e > kFactorTop) {
      m = 65535u;
      e = kFactorTop;
    }
    const uint32_t scaled =
        (static_cast<uint32_t>(e + 15 + kTokenShift + 127) << 23) | ((m & 0x7fffu) << 8);
    f[0] = factor >= kFactorFloor ? __uint_as_float(scaled) : 0.0f;
    const uint8_t* const dense = row + Rows<C>::kTail;
    uint32_t out[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      uint32_t o = 0u;
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const uint32_t u = dense[2 * i + k];
        o |= (u != 0u ? 32u - __clz(u) + kBitsBias : 0u) << (16 * k);
      }
      out[i] = o;
    }
    uint4* const dst = reinterpret_cast<uint4*>(pools.bits[r]);
    dst[0] = make_uint4(out[0], out[1], out[2], out[3]);
    dst[1] = make_uint4(out[4], out[5], out[6], out[7]);
  }
}

//: ONE VALUE DIGIT OF A K-STEP AS B: the 32 tokens' digit for the channel group, by two `ldmatrix.x4.trans` and a
//: byte permute a register -- `b[n]` is n-tile `n`, the even and odd channels of each 16-channel half. -- see
//: docs/internals/owner/owner_fold.md#operands
template <class C>
__device__ __forceinline__ void load_value_digit(const uint8_t* row, int digit,
                                                 uint32_t (&b)[4][2]) {
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    uint32_t v[4];
    ops::load_frag_t(v, ops::smem_addr(row + Rows<C>::kValues + 32 * digit + 16 * half));
    b[2 * half][0] = ops::prmt(v[0], v[1], 0x6420u);
    b[2 * half][1] = ops::prmt(v[2], v[3], 0x6420u);
    b[2 * half + 1][0] = ops::prmt(v[0], v[1], 0x7531u);
    b[2 * half + 1][1] = ops::prmt(v[2], v[3], 0x7531u);
  }
}

//: THE DENSE DIGITS OF A K-STEP in the A layout: register `2 h + l` holds leaf `2 g + l` at the tokens `a_token(tq,
//: h, b)`, by `ldmatrix.x4.trans` and a byte permute a register.
template <class C>
__device__ __forceinline__ void load_dense(const uint8_t* row, uint32_t (&a)[4]) {
  uint32_t v[4];
  ops::load_frag_t(v, ops::smem_addr(row + Rows<C>::kTail));
  a[0] = ops::prmt(v[0], v[1], 0x6420u);
  a[1] = ops::prmt(v[0], v[1], 0x7531u);
  a[2] = ops::prmt(v[2], v[3], 0x6420u);
  a[3] = ops::prmt(v[2], v[3], 0x7531u);
}

//: the step token byte `b` of A register `2 h + l` holds
__device__ __forceinline__ int a_token(int tq, int h, int b) {
  return 16 * h + 8 * (b >> 1) + 2 * tq + (b & 1);
}

//: A RUN'S STEP: entry `e` of the run as a pool row, and whether it is in the run; past the run's end, the run's last
//: row (its factor is taken as zero) -- branch-free, so a step's token loads schedule into the burst before it.
template <class C>
__device__ __forceinline__ int run_slot(const ListTenants<C>& sm, int warp, int e, int end,
                                        int first, bool listed, bool& in) {
  in = e < end;
  const int at = in ? e : end - 1;
  const int listed_slot = sm.entries[0][warp][at];
  return (listed ? listed_slot : at) - first;
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

//: A TOKEN'S LIST FACTOR: `u_ws m 2^(e - 50)` for list byte `s` of its row, exact in fp32 (zero past the run).
template <class C>
__device__ __forceinline__ float list_factor(const WritePool<C>& pools, int slot, int s, bool in) {
  const uint8_t* const row = pools.write[slot];
  const float f = *reinterpret_cast<const float*>(row + Rows<C>::kFactors);
  const float c = __fmul_rn(__uint2float_rn(row[s]), f);
  return in ? c : 0.0f;
}

//: A TOKEN PAIR'S BIASED BIT LENGTHS: `bitlen(c) + e + 71` of this lane's token (0 for a zero factor) and of the next
//: lane's, halfwords over the fp16 floor.
__device__ __forceinline__ uint32_t token_pair(float ct) {
  const uint32_t t =
      static_cast<uint32_t>(max(static_cast<int>(__float_as_uint(ct) >> 23) - kExpoBias, 0))
      + kHalfFloor;
  return t | (__shfl_down_sync(0xffffffffu, t, 1) << 16);
}

//: A STEP'S BIT LENGTHS: this lane's own token row's sixteen biased bit lengths, by `ldmatrix.trans` -- `r[0][j]` holds
//: leaf `g`, `r[1][j]` leaf `8 + g`, at tokens `8 j + {2 tq, 2 tq + 1}`.
template <class C>
__device__ __forceinline__ void load_bits(const WritePool<C>& pools, int slot,
                                          uint32_t (&r)[2][4]) {
  ops::load_frag_t(r[0], ops::smem_addr(&pools.bits[slot][0]));
  ops::load_frag_t(r[1], ops::smem_addr(&pools.bits[slot][8]));
}

//: A STEP INTO A LIST'S SCALES: the step's products' bit-length bounds, the running maxima over its token pairs.
__device__ __forceinline__ void scale_step(const uint32_t (&r)[2][4], uint32_t pair, int tq,
                                           uint32_t& lo, uint32_t& hi) {
  uint32_t a[4], b[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t w = __shfl_sync(0xffffffffu, pair, 8 * j + 2 * tq);
    a[j] = r[0][j] + w;
    b[j] = r[1][j] + w;
  }
  lo = ops::hmax2(lo, ops::hmax2(ops::hmax2(a[0], a[1]), ops::hmax2(a[2], a[3])));
  hi = ops::hmax2(hi, ops::hmax2(ops::hmax2(b[0], b[1]), ops::hmax2(b[2], b[3])));
}

//: A LIST'S SCALES, FINISHED: the running maxima reduced over the quad and the halves, the A layout's leaves `2 g` and
//: `2 g + 1` fetched, as bytes (`len(2 g) | len(2 g + 1) << 8`, the floor taken off).
__device__ __forceinline__ uint32_t finish_scales(uint32_t lo, uint32_t hi, int g) {
#pragma unroll
  for (int x = 1; x < 4; x <<= 1) {
    lo = ops::hmax2(lo, __shfl_xor_sync(0xffffffffu, lo, x));
    hi = ops::hmax2(hi, __shfl_xor_sync(0xffffffffu, hi, x));
  }
  lo = ops::hmax2(lo, ops::prmt(lo, lo, 0x1032u));
  hi = ops::hmax2(hi, ops::prmt(hi, hi, 0x1032u));
  const uint32_t mine = ops::prmt(lo, hi, 0x5410u);
  const uint32_t even = __shfl_sync(0xffffffffu, mine, 4 * ((2 * g) & 7));
  const uint32_t odd = __shfl_sync(0xffffffffu, mine, 4 * ((2 * g + 1) & 7));
  return ops::prmt(ops::prmt(even, odd, g < 4 ? 0x5410u : 0x7632u), 0u, 0x4420u);
}

//: ONE LIST'S SCALES OF A PASS (its own run): each leaf's largest `bitlen(u) + bitlen(c) + e` over its live products,
//: biased into halfwords on the fp16 order (`HMNMX2`). -- see docs/internals/owner/owner_fold.md#scale
template <class C>
__device__ __forceinline__ uint32_t list_scales(const WritePool<C>& pools, const ListTenants<C>& sm,
                                                int s, int warp, int lane, int at0, int end,
                                                int first, bool listed) {
  constexpr uint32_t kFloor = kHalfFloor * 0x10001u;
  uint32_t lo = kFloor, hi = kFloor;
#pragma unroll 1
  for (int at = at0; at < end; at += 32) {
    bool in;
    const int slot = run_slot(sm, warp, at + lane, end, first, listed, in);
    uint32_t r[2][4];
    load_bits(pools, slot, r);
    scale_step(r, token_pair(list_factor(pools, slot, s, in)), lane & 3, lo, hi);
  }
  return finish_scales(lo, hi, lane >> 2);
}

//: EVERY LIST'S SCALES OF A PASS over a run they share (merged, or a dense side): the step's rows, bit lengths and
//: factors read once for all of the warp's lists, eight independent chains a step; two lists' bytes a word.
template <class C>
__device__ __forceinline__ void warp_scales(const WritePool<C>& pools, const ListTenants<C>& sm,
                                            int warp, int lane, int at0, int end, int first,
                                            bool listed, uint32_t (&lens)[C::kFoldLists / 2]) {
  constexpr int kLists = C::kFoldLists;
  constexpr uint32_t kFloor = kHalfFloor * 0x10001u;
  uint32_t lo[kLists], hi[kLists];
#pragma unroll
  for (int j = 0; j < kLists; ++j) lo[j] = hi[j] = kFloor;
#pragma unroll 1
  for (int at = at0; at < end; at += 32) {
    bool in;
    const int slot = run_slot(sm, warp, at + lane, end, first, listed, in);
    const uint8_t* const row = pools.write[slot];
    const float f = in ? *reinterpret_cast<const float*>(row + Rows<C>::kFactors) : 0.0f;
    const float bias = -__fmul_rn(f, 0x1p23f);
    const uint2 bytes = *reinterpret_cast<const uint2*>(row + warp * kLists);
    uint32_t r[2][4];
    load_bits(pools, slot, r);
#pragma unroll
    for (int j = 0; j < kLists; ++j) {
      const float magic =
          __uint_as_float(ops::prmt(kMagic, j < 4 ? bytes.x : bytes.y, 0x3004u + (j & 3)));
      scale_step(r, token_pair(__fmaf_rn(magic, f, bias)), lane & 3, lo[j], hi[j]);
    }
  }
#pragma unroll
  for (int j = 0; j < kLists; j += 2)
    lens[j / 2] = finish_scales(lo[j], hi[j], lane >> 2)
                  | (finish_scales(lo[j + 1], hi[j + 1], lane >> 2) << 16);
}

//: HALF A STEP'S COEFFICIENTS (register `2 h + l`): its eight elements formed and packed into the two digits.
template <int H>
__device__ __forceinline__ void coefficient_half(const uint32_t (&u)[4], const float (&c)[4],
                                                 const float (&scale)[2], const float (&bias)[2],
                                                 uint32_t (&hi)[4], uint32_t (&lo)[4]) {
#pragma unroll
  for (int l = 0; l < 2; ++l) {
    int32_t a[4];
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const float magic = __uint_as_float(ops::prmt(kMagic, u[2 * H + l], 0x3004u + b));
      a[b] = __float_as_int(__fmul_rn(__fmaf_rn(magic, scale[l], bias[l]), c[b]));
    }
    const uint32_t w01 = ops::pack_sat_u16(a[1], a[0]), w23 = ops::pack_sat_u16(a[3], a[2]);
    lo[2 * H + l] = ops::prmt(w01, w23, 0x6420u);
    hi[2 * H + l] = ops::prmt(w01, w23, 0x7531u);
  }
}

//: THE LIST FACTORS OF A STEP'S TOKENS in the A layout: each lane's eight, by shuffle from their tokens' lanes.
__device__ __forceinline__ void layout_factors(float ct, int tq, float (&c)[2][4]) {
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int b = 0; b < 4; ++b) c[h][b] = __shfl_sync(0xffffffffu, ct, a_token(tq, h, b));
}

//: A STEP'S COEFFICIENT DIGITS, the set two steps alternate; the value digits are one buffer, each digit reloaded in
//: place once its last reader in the burst is done.
struct StepDigits {
  uint32_t hi[4], lo[4];
};

//: THE VALUE DIGITS a burst reads: `v_hat` as B, `v_lo` as the two raw `ldmatrix` results it permutes itself.
struct StepValues {
  uint32_t vhat[4][2];
  uint32_t vlo_raw[2][4];
};

//: THE RAW `v_lo` MATRICES of a step's tokens: two `ldmatrix.x4.trans`, permuted later by `vlo_layout`.
template <class C>
__device__ __forceinline__ void load_vlo_raw(const uint8_t* row, uint32_t (&raw)[2][4]) {
#pragma unroll
  for (int half = 0; half < 2; ++half)
    ops::load_frag_t(raw[half], ops::smem_addr(row + Rows<C>::kValues + 32 + 16 * half));
}

__device__ __forceinline__ void vlo_layout(const uint32_t (&raw)[2][4], uint32_t (&b)[4][2]) {
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    b[2 * half][0] = ops::prmt(raw[half][0], raw[half][1], 0x6420u);
    b[2 * half][1] = ops::prmt(raw[half][2], raw[half][3], 0x6420u);
    b[2 * half + 1][0] = ops::prmt(raw[half][0], raw[half][1], 0x7531u);
    b[2 * half + 1][1] = ops::prmt(raw[half][2], raw[half][3], 0x7531u);
  }
}

//: A STEP'S BURST: its sixteen digit products, the `v_hat` ones first, with (`kGather`) the next step's operands built
//: around them from its token (`slot`, `ct`, carried a step ahead); `kFresh` starts the sums.
//: -- see docs/internals/owner/owner_fold.md#burst
//: @burst
template <class C, bool kFresh, bool kGather>
__device__ __forceinline__ void step_burst(int32_t (&hh)[4][4], int32_t (&mid)[4][4],
                                           int32_t (&ll)[4][4], const StepDigits& cur,
                                           StepDigits& next, StepValues& val,
                                           const WritePool<C>& pools, const ListTenants<C>& sm,
                                           int s, int warp, int lane, int at_after, int end,
                                           int first, bool listed, const float (&scale)[2],
                                           const float (&bias)[2], int& slot, float& ct) {
  uint32_t u[4] = {};
  float c[2][4] = {};
  if constexpr (kGather) {
    load_dense<C>(pools.write[slot], u);
    layout_factors(ct, lane & 3, c);
  }
  uint32_t vlo[4][2];
  vlo_layout(val.vlo_raw, vlo);

#pragma unroll
  for (int n = 0; n < 4; ++n) {
    if constexpr (kFresh)
      ops::imma_u8s8_k32_fresh(hh[n], cur.hi, val.vhat[n][0], val.vhat[n][1]);
    else
      ops::imma_u8s8_k32(hh[n], cur.hi, val.vhat[n][0], val.vhat[n][1]);
  }
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    if constexpr (kFresh)
      ops::imma_u8s8_k32_fresh(mid[n], cur.lo, val.vhat[n][0], val.vhat[n][1]);
    else
      ops::imma_u8s8_k32(mid[n], cur.lo, val.vhat[n][0], val.vhat[n][1]);
  }

  if constexpr (kGather) {
    load_value_digit<C>(pools.write[slot], 0, val.vhat);
    coefficient_half<0>(u, c[0], scale, bias, next.hi, next.lo);
  }
#pragma unroll
  for (int n = 0; n < 4; ++n) ops::imma_u8_k32(mid[n], cur.hi, vlo[n][0], vlo[n][1]);
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    if constexpr (kFresh)
      ops::imma_u8_k32_fresh(ll[n], cur.lo, vlo[n][0], vlo[n][1]);
    else
      ops::imma_u8_k32(ll[n], cur.lo, vlo[n][0], vlo[n][1]);
  }
  if constexpr (kGather) {
    load_vlo_raw<C>(pools.write[slot], val.vlo_raw);
    coefficient_half<1>(u, c[1], scale, bias, next.hi, next.lo);
    bool in;
    slot = run_slot(sm, warp, at_after + lane, end, first, listed, in);
    ct = list_factor(pools, slot, s, in);
  }
}

//: THE COEFFICIENT'S SCALE AND BIAS for the leaves' scale bytes: `2^(-83 - K)` and `-2^23` of it.
__device__ __forceinline__ void leaf_scales(uint32_t lens, float (&scale)[2], float (&bias)[2]) {
#pragma unroll
  for (int l = 0; l < 2; ++l) {
    const int len = static_cast<int>((lens >> (8 * l)) & 0xffu);
    const int k =
        len >= static_cast<int>(kLive) ? len - static_cast<int>(kBitsBias + kTokenBias) : 0;
    scale[l] = __fmul_rn(pow2(kLeafShift + 64 - k), 0x1p-64f);
    bias[l] = -pow2(kLeafShift + 23 - k);
  }
}

//: A LIST'S SUMS OF A PASS, one body for every list: its first step's operands whole and the second step's token,
//: then the steps as bursts, each building the next step's operands, into `hh`, `mid`, `ll`. -- see
//: docs/internals/owner/owner_fold.md#list
template <class C, class Clock>
__device__ __forceinline__ void list_sums(const WritePool<C>& pools, const ListTenants<C>& sm,
                                          int s, int warp, int lane, int at0, int end, int first,
                                          bool listed, uint32_t lens, int32_t (&hh)[4][4],
                                          int32_t (&mid)[4][4], int32_t (&ll)[4][4], Clock& clk) {
  float scale[2], bias[2];
  leaf_scales(lens, scale, bias);

  StepDigits x, y;
  StepValues val;
  {
    bool in;
    const int slot0 = run_slot(sm, warp, at0 + lane, end, first, listed, in);
    uint32_t u[4];
    float c[2][4];
    load_dense<C>(pools.write[slot0], u);
    layout_factors(list_factor(pools, slot0, s, in), lane & 3, c);
    load_value_digit<C>(pools.write[slot0], 0, val.vhat);
    load_vlo_raw<C>(pools.write[slot0], val.vlo_raw);
    coefficient_half<0>(u, c[0], scale, bias, x.hi, x.lo);
    coefficient_half<1>(u, c[1], scale, bias, x.hi, x.lo);
  }

  bool in;
  int slot = run_slot(sm, warp, at0 + 32 + lane, end, first, listed, in);
  float ct = list_factor(pools, slot, s, in);
  clk.lap(kFoldFirst);
  const int steps = (end - at0 + 31) >> 5;
  const bool one = steps == 1;
#pragma unroll 1
  for (int t = ops::once(one); t > 0; --t)
    step_burst<C, true, false>(hh, mid, ll, x, y, val, pools, sm, s, warp, lane, 0, end, first,
                               listed, scale, bias, slot, ct);
#pragma unroll 1
  for (int t = ops::once(!one); t > 0; --t) {
    step_burst<C, true, true>(hh, mid, ll, x, y, val, pools, sm, s, warp, lane, at0 + 64, end,
                              first, listed, scale, bias, slot, ct);
    //: steps 1 .. steps - 1 remain, y holds step 1: pairs alternate the sets, the last burst gathers nothing
    int i = 1;
#pragma unroll 1
    for (; i + 2 < steps; i += 2) {
      step_burst<C, false, true>(hh, mid, ll, y, x, val, pools, sm, s, warp, lane,
                                 at0 + 32 * (i + 2), end, first, listed, scale, bias, slot, ct);
      step_burst<C, false, true>(hh, mid, ll, x, y, val, pools, sm, s, warp, lane,
                                 at0 + 32 * (i + 3), end, first, listed, scale, bias, slot, ct);
    }
    const bool odd = i + 1 >= steps;
#pragma unroll 1
    for (int r = ops::once(!odd); r > 0; --r) {
      step_burst<C, false, true>(hh, mid, ll, y, x, val, pools, sm, s, warp, lane,
                                 at0 + 32 * (i + 2), end, first, listed, scale, bias, slot, ct);
      step_burst<C, false, false>(hh, mid, ll, x, y, val, pools, sm, s, warp, lane, 0, end, first,
                                  listed, scale, bias, slot, ct);
    }
#pragma unroll 1
    for (int r = ops::once(odd); r > 0; --r)
      step_burst<C, false, false>(hh, mid, ll, y, x, val, pools, sm, s, warp, lane, 0, end, first,
                                  listed, scale, bias, slot, ct);
  }
  clk.lap(kFoldBursts);
}

//: THE STATE GAINS A LIST'S SUMS: `ll` at the scale's unit, then `256 hh + mid` (exact in s32) at 2^8 of it.
__device__ __forceinline__ void add_sums(float (&acc)[4][4], const int32_t (&hh)[4][4],
                                         const int32_t (&mid)[4][4], const int32_t (&ll)[4][4],
                                         uint32_t lens) {
  float low[2], high[2];
#pragma unroll
  for (int l = 0; l < 2; ++l) {
    const int len = static_cast<int>((lens >> (8 * l)) & 0xffu);
    const int k =
        len >= static_cast<int>(kLive) ? len - static_cast<int>(kBitsBias + kTokenBias) : 0;
    low[l] = pow2(k - 16);
    high[l] = pow2(k - 8);
  }
#pragma unroll
  for (int n = 0; n < 4; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int l = i >> 1;
      const float x = __fmaf_rn(__int2float_rn(ll[n][i]), low[l], acc[n][i]);
      acc[n][i] = __fmaf_rn(__int2float_rn(hh[n][i] * 256 + mid[n][i]), high[l], x);
    }
}

//: A WARP'S FOLD OF ONE PASS (slots `[first, first + n)`): each of its lists in turn over its run of the pass -- the
//: list's own entries (per list, its scales its own), or the warp union's (merged) or every slot (a dense side), where
//: a token outside the list has a zero byte and so a zero factor, the same sums, every list's scales read in one pass
//: over the shared run -- one body, the state's list chosen at the add. The warp's cursors carry each run's next entry
//: across the window's passes. -- see docs/internals/owner/owner_fold.md#forms
template <class C, class Clock>
__device__ void fold_pass(const WritePool<C>& pools, const ListTenants<C>& sm, FoldTenants<C>& ft,
                          const OwnerParams& p, FoldState<C>& st, int first, int n, int warp,
                          int lane, Clock& clk) {
  int* const cursor = ft.cursor[warp];
  const int row0 = warp * C::kFoldLists;
  const int limit = first + n;
  const bool listed = p.sparse_write != 0;
  const bool per_list = listed && sm.form[0][warp] == kStepPerList;
  const int union_at = listed ? cursor[0] : first;
  const int union_end =
      listed && !per_list ? pass_end(sm, warp, union_at, sm.total[0][warp], limit, lane) : limit;
  clk.lap(kFoldHead);

  if (!per_list && union_at < union_end) {
    constexpr int kLensWords = C::kFoldLists / 2;
    uint32_t lens[kLensWords];
    warp_scales(pools, sm, warp, lane, union_at, union_end, first, listed, lens);
#pragma unroll
    for (int i = 0; i < kLensWords; ++i) ft.lens[warp][i][lane] = lens[i];
    clk.lap(kFoldScale);
  }

  int off = 0;
#pragma unroll 1
  for (int j = 0; j < C::kFoldLists; ++j) {
    int at0 = union_at, end = union_end;
    if (per_list) {
      const int total = off + sm.count[row0 + j];
      at0 = cursor[j];
      end = pass_end(sm, warp, at0, total, limit, lane);
      cursor[j] = end;
      off = total;
    }
    clk.lap(kFoldHead);
    if (at0 == end) continue;
    const uint32_t lens =
        per_list ? list_scales(pools, sm, row0 + j, warp, lane, at0, end, first, listed)
                 : (ft.lens[warp][j >> 1][lane] >> (16 * (j & 1))) & 0xffffu;
    clk.lap(kFoldScale);
    int32_t hh[4][4], mid[4][4], ll[4][4];
    list_sums(pools, sm, row0 + j, warp, lane, at0, end, first, listed, lens, hh, mid, ll, clk);
    rola::static_for<C::kFoldLists>([&](auto J) {
      if (j == decltype(J)::value) add_sums(st.acc[decltype(J)::value], hh, mid, ll, lens);
    });
    clk.lap(kFoldAdd);
  }
  if (listed && !per_list) cursor[0] = union_end;
}

//: A WINDOW'S CURSORS: each list's first entry (per list), or the union's (merged).
template <class C>
__device__ __forceinline__ void fold_cursors(const ListTenants<C>& sm, FoldTenants<C>& ft,
                                             int warp) {
  constexpr int kLists = C::kFoldLists;
  int off = 0;
#pragma unroll
  for (int j = 0; j < kLists; ++j) {
    ft.cursor[warp][j] = sm.form[0][warp] == kStepPerList ? off : 0;
    off += sm.count[warp * kLists + j];
  }
}

}  // namespace rola::owner::detail
