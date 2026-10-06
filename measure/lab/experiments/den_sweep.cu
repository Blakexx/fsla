// measure/lab/experiments/den_sweep.cu -- QUESTION: the carry's window-aligned denominator depends only on routing
// (den(t) = the read leaf product against the write mass of the windows before t's -- tests/oracle/reference.py), so a
// pass running beside the carry could hand its readout a common unit. With lanes on each half's dense level and loops
// on its sparse level, what does a CTA of 64 level-S rows x 256 level-R digits (4 a head) cost a window, against the
// carry's ~20K? METHOD: writes (dense level S, sparse level R, fp32 accumulate): a warp owns 32 level-R digits, a lane
// 2 level-S rows, the mass in registers; the warp walks its lists (each digit's padded to 4 entries of token and u8
// value) 128 entries at a time through its shared slots (the token's coefficient byte offset, gain x value), the next
// 128 loading meanwhile; each digit the fill overlaps (a ballot) steps its entries four at a time, an entry one
// contiguous 4-byte read of the token's staged f16 coefficients a lane and two FFMAs. The den (sparse level S, dense
// level R, fp16 like the readout): a warp owns 8 level-S rows, a half-warp a token, a lane 16 level-R digits; the mass
// transposed once a window through shared memory (f16, swizzled) into registers, each row scaled by the power of two
// that keeps eight u8 x mass products under f16 overflow; the rows' read lists (the token's read-row byte offset | u8
// value << 24, each row's padded to 4) staged a window ahead, as a facts pass would emit them; each row (unrolled)
// walks its entries two a warp-iteration, the read rows loaded four entries ahead in a two-slot ring, 8 HFMA2s a lane,
// a reduction over the half-warp. One head's 4 CTAs over a 32-window pool streamed from DRAM, as a few pass CTAs would
// run beside a compute-bound carry; checked against a host reference first. The block's span a window; PART isolates
// the halves.
// see docs/internals/tools/lab.md#den-sweep
#include "lab.cuh"

namespace den_sweep {

constexpr int kThreads = 256, kWarps = 8, kWindow = 512, kRows = 64, kDigits = 256, kPool = 32;
constexpr int kSlices = kDigits / kRows, kBlock = kDigits / kWarps, kBand = kRows / kWarps,
              kOffsets = 264;
constexpr int kFill = 128, kListPad = 4, kReadCap = 2560;

//: THE SHARED TENANTS: one region holding the published mass (f16 `[64][256]`, a row's 16-byte chunks XORed with
//: row / 2) from the window's start until the den has it in registers, then the window's level-S write coefficients
//: (f16 `[512][64]`); the window's gains; a window ahead the CTA's rows' read lists and their offsets; each warp's
//: slots for 128 list entries and its digits' list bounds
struct alignas(16) Shared {
  uint8_t region[kWindow * kRows * 2];
  float m[kWindow];
  uint32_t reads[2][kReadCap];
  uint32_t read_at[2][kRows + 8];
  alignas(16) uint32_t slot_at[kWarps][kFill + 4];
  alignas(16) float slot_c[kWarps][kFill + 4];
  uint2 bounds[kWarps][kBlock];
};

//: THE POOL, a window each: level S's read lists digit-major (offsets; entries `token x 256 | value << 24`, a digit's
//: padded to 4 with value 0), level R's write lists digit-major (offsets, tokens, u8 values, a digit's padded to 4 with
//: value 0), per token level S's f16 write coefficients and level R's u8 read row, the gain
struct Pool {
  const uint32_t* read_at;  // [pool][264]
  const uint32_t* reads;    // [pool][512 READ_K + 1024]
  const uint16_t* offsets;  // [pool][264]
  const uint16_t* tokens;   // [pool][512 WRITE_K + 1024]
  const uint8_t* values;    // [pool][512 WRITE_K + 1024]
  const __half* ws;         // [pool x 512][256]
  const uint8_t* rr;        // [pool x 512][256]
  const float* m;           // [pool x 512]
};

__device__ __forceinline__ void red_f32(float* p, float v) {
  asm volatile("red.global.add.f32 [%0], %1;" ::"l"(p), "f"(v) : "memory");
}

//: two u8 of a word (`sel` 0x4140 the low pair, 0x4342 the high) as an exact f16 pair: 1024 + b by PRMT, less 1024
__device__ __forceinline__ uint32_t u8_pair(uint32_t w, uint32_t sel) {
  uint32_t x, d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(x) : "r"(w), "r"(0x64646464u), "r"(sel));
  asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(x), "r"(0x64006400u));
  return d;
}

__device__ __forceinline__ uint32_t hfma2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

__device__ __forceinline__ uint32_t hmax2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("max.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

__device__ __forceinline__ float2 h2f(uint32_t h) {
  return __half22float2(*reinterpret_cast<const __half2*>(&h));
}

__device__ __forceinline__ uint32_t lds32(const uint8_t* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}

//: the mass (a lane's rows `2 lane + i` of the warp's 32 digits) into the region as f16
__device__ __forceinline__ void publish(Shared& sh, const float (&mw)[2][kBlock], int warp,
                                        int lane) {
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int row = 2 * lane + i, chunk = (4 * warp + q) ^ lane;
      uint4 v;
      v.x = lab::pack_half2(mw[i][8 * q], mw[i][8 * q + 1]);
      v.y = lab::pack_half2(mw[i][8 * q + 2], mw[i][8 * q + 3]);
      v.z = lab::pack_half2(mw[i][8 * q + 4], mw[i][8 * q + 5]);
      v.w = lab::pack_half2(mw[i][8 * q + 6], mw[i][8 * q + 7]);
      *reinterpret_cast<uint4*>(sh.region + row * 2 * kDigits + 16 * chunk) = v;
    }
}

//: the den's mass: the warp's rows `8 warp + i`, the lane's digits `8 sub ..` and `128 + 8 sub ..` (f16 pairs), each row
//: scaled by 2^e so its largest value lies in [16, 32); `inv[i]` = 2^-e
__device__ __forceinline__ void load_band(const Shared& sh, uint32_t (&md)[kBand][8],
                                          float (&inv)[kBand], int warp, int sub) {
#pragma unroll
  for (int i = 0; i < kBand; ++i) {
    const int row = kBand * warp + i, x = (row >> 1) & 31;
    const uint4 lo =
        *reinterpret_cast<const uint4*>(sh.region + row * 2 * kDigits + 16 * (sub ^ x));
    const uint4 hi =
        *reinterpret_cast<const uint4*>(sh.region + row * 2 * kDigits + 16 * ((sub + 16) ^ x));
    const uint32_t v[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    uint32_t m2 = v[0];
#pragma unroll
    for (int k = 1; k < 8; ++k) m2 = hmax2(m2, v[k]);
    uint32_t top = max(m2 & 0xffffu, m2 >> 16);
#pragma unroll
    for (int d = 8; d >= 1; d >>= 1) top = max(top, __shfl_xor_sync(0xffffffffu, top, d));
    const int biased = top == 0u ? 19 : max(static_cast<int>(top >> 10), 1);
    const int e = 19 - biased, e1 = min(e, 15), e2 = e - e1;
    const uint32_t s1 = static_cast<uint32_t>(e1 + 15) << 10, s2 = static_cast<uint32_t>(e2 + 15)
                                                                   << 10;
#pragma unroll
    for (int k = 0; k < 8; ++k)
      md[i][k] = lab::hmul2(lab::hmul2(v[k], s1 | (s1 << 16)), s2 | (s2 << 16));
    inv[i] = __int_as_float((127 - e) << 23);
  }
}

//: window `win`'s read lists for the CTA's rows (entries `[lo, hi)`) and their offsets into buffer `win & 1`
__device__ __forceinline__ void stage_reads(Shared& sh, const Pool& g, int win, int lo, int hi,
                                            int s0, int tid) {
  const int b = win & 1, pw = win % kPool;
#pragma unroll 1
  for (int i = tid; i < hi - lo; i += kThreads)
    lab::cp_async4(lab::smem(&sh.reads[b][i]), g.reads + lo + i);
  if (tid <= kRows)
    lab::cp_async4(lab::smem(&sh.read_at[b][tid]), g.read_at + pw * kOffsets + s0 + tid);
}

__device__ __forceinline__ void stage_coefficients(Shared& sh, const Pool& g, int win, int s0,
                                                   int tid) {
  const int64_t t0 = static_cast<int64_t>(win % kPool) * kWindow;
#pragma unroll 1
  for (int i = tid; i < kWindow * 8; i += kThreads)
    lab::cp_async16(lab::smem(sh.region + 16 * i),
                    g.ws + (t0 + (i >> 3)) * kDigits + s0 + 8 * (i & 7));
  if (tid < kWindow / 4) lab::cp_async16(lab::smem(&sh.m[4 * tid]), g.m + t0 + 4 * tid);
}

//: A DEN SLOT: a half-warp's entry and the lane's 16 bytes of its token's read row (each row's entries in pairs of slots,
//: a step of four, reduced together after both slots' arithmetic)
struct Slot {
  uint32_t e;
  uint2 lo, hi;
};

//: a lane's share of an entry (one a half-warp) against the band's scaled row I: two chains of 4 HFMA2s
template <int I>
__device__ __forceinline__ float den_part(const uint32_t (&md)[kBand][8], const float (&inv)[kBand],
                                          const Slot& s) {
  const uint32_t w[4] = {s.lo.x, s.lo.y, s.hi.x, s.hi.y};
  uint32_t a0 = 0u, a1 = 0u;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    a0 = hfma2(u8_pair(w[k], 0x4140u), md[I][2 * k], a0);
    a1 = hfma2(u8_pair(w[k], 0x4342u), md[I][2 * k + 1], a1);
  }
  const float2 f0 = h2f(a0), f1 = h2f(a1);
  return ((f0.x + f0.y) + (f1.x + f1.y)) * (static_cast<float>(s.e >> 24) * inv[I]);
}

//: PART 0 both halves; 1 the den alone; 2 the writes alone; 3 neither (the staging, the publish, the barriers)
template <int WRITE_K, int READ_K, int PART>
__global__ void __launch_bounds__(kThreads, 1) sweep(int windows, Pool g, float* den, long long* t,
                                                     int* out) {
  extern __shared__ __align__(128) unsigned char raw[];
  Shared& sh = *reinterpret_cast<Shared*>(raw);
  const int tid = static_cast<int>(threadIdx.x), lane = lab::lane_id(), warp = lab::warp_id();
  const int s0 = static_cast<int>(blockIdx.x) * kRows;
  constexpr bool kDen = PART == 0 || PART == 1, kWrites = PART == 0 || PART == 2;
  constexpr int kWriteList = kWindow * WRITE_K + kListPad * kDigits;
  float mw[2][kBlock] = {};
  if (lane < 4) {
    sh.slot_at[warp][kFill + lane] = 0u;
    sh.slot_c[warp][kFill + lane] = 0.0f;
  }
  stage_reads(sh, g, 0, g.read_at[s0], g.read_at[s0 + kRows], s0, tid);
  lab::cp_commit();
  int next_lo = windows > 1 ? g.read_at[kOffsets + s0] : 0;
  int next_hi = windows > 1 ? g.read_at[kOffsets + s0 + kRows] : 0;
  lab::cp_wait<0>();
  float sink = 0.0f;
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int win = 0; win < windows; ++win) {
    const int b = win & 1, pw = win % kPool;
    const int64_t t0 = static_cast<int64_t>(pw) * kWindow;
    __syncthreads();
    //: the next window's read lists; the mass as of the windows before into the region
    if (win + 1 < windows) stage_reads(sh, g, win + 1, next_lo, next_hi, s0, tid);
    lab::cp_commit();
    if (win + 2 < windows) {
      next_lo = g.read_at[((win + 2) % kPool) * kOffsets + s0];
      next_hi = g.read_at[((win + 2) % kPool) * kOffsets + s0 + kRows];
    }
    //: the window's writes' digit bounds and first 128 entries, in flight through the den
    const uint16_t* const off = g.offsets + pw * kOffsets + kBlock * warp;
    const uint16_t* const tok = g.tokens + pw * kWriteList;
    const uint8_t* const val = g.values + pw * kWriteList;
    int o_l = 0, e_l = 0, tkn[4] = {}, first = 0, last = 0;
    uint32_t vn[4] = {};
    const auto fill_load = [&](int f0, int(&tk)[4], uint32_t(&v)[4]) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int idx = f0 + 32 * q + lane;
        tk[q] = idx < last ? tok[idx] : 0;
        v[q] = idx < last ? val[idx] : 0u;
      }
    };
    if constexpr (kWrites) {
      o_l = off[lane];
      e_l = off[lane + 1];
      first = __shfl_sync(0xffffffffu, o_l, 0);
      last = __shfl_sync(0xffffffffu, e_l, 31);
      fill_load(first, tkn, vn);
    }
    publish(sh, mw, warp, lane);
    __syncthreads();
    uint32_t md[kBand][8];
    float inv[kBand];
    if constexpr (kDen) load_band(sh, md, inv, warp, lane & 15);
    __syncthreads();
    //: the region free: the window's coefficients and gains
    stage_coefficients(sh, g, win, s0, tid);
    lab::cp_commit();
    if constexpr (kDen) {
      const int h = lane >> 4, sub = lane & 15;
      const uint32_t* const rl = sh.reads[b];
      const int base = static_cast<int>(sh.read_at[b][0]);
      const int w_lo = static_cast<int>(sh.read_at[b][kBand * warp]) - base;
      const int w_hi = static_cast<int>(sh.read_at[b][kBand * warp + kBand]) - base;
      const uint8_t* const rr = g.rr + t0 * kDigits + 8 * sub;
      float* const dw = den + t0;
      //: the half-warp's entry `idx + h` and its read row's bytes, nothing past the warp's entries
      const auto load = [&](Slot& s, int idx) {
        s.e = idx < w_hi ? rl[idx + h] : 0u;
        const uint8_t* const p = rr + (s.e & 0xffffffu);
        s.lo = *reinterpret_cast<const uint2*>(p);
        s.hi = *reinterpret_cast<const uint2*>(p + 128);
      };
      Slot sa, sb;
      load(sa, w_lo);
      load(sb, w_lo + 2);
      int e = w_lo;
      //: each row's entries four at a time, slot a the first two, slot b the next, each reloaded four ahead
      const auto run_row = [&](auto row) {
        constexpr int I = decltype(row)::value;
        const int end = static_cast<int>(sh.read_at[b][kBand * warp + I + 1]) - base;
#pragma unroll 1
        for (; e < end; e += 4) {
          float pa = den_part<I>(md, inv, sa), pb = den_part<I>(md, inv, sb);
          const uint32_t ea = sa.e, eb = sb.e;
          load(sa, e + 4);
          load(sb, e + 6);
#pragma unroll
          for (int d = 8; d >= 1; d >>= 1) {
            pa += __shfl_xor_sync(0xffffffffu, pa, d);
            pb += __shfl_xor_sync(0xffffffffu, pb, d);
          }
          if (sub == 0 && (ea >> 24) != 0u) red_f32(dw + ((ea & 0xffffffu) >> 8), pa);
          if (sub == 0 && (eb >> 24) != 0u) red_f32(dw + ((eb & 0xffffffu) >> 8), pb);
        }
      };
      run_row(std::integral_constant<int, 0>{});
      run_row(std::integral_constant<int, 1>{});
      run_row(std::integral_constant<int, 2>{});
      run_row(std::integral_constant<int, 3>{});
      run_row(std::integral_constant<int, 4>{});
      run_row(std::integral_constant<int, 5>{});
      run_row(std::integral_constant<int, 6>{});
      run_row(std::integral_constant<int, 7>{});
    }
    lab::cp_wait<0>();
    __syncthreads();
    //: the next window's read rows toward L2, its lists landed
    if (kDen && win + 1 < windows) {
      const uint32_t* const nl = sh.reads[b ^ 1];
      const int n = static_cast<int>(sh.read_at[b ^ 1][kRows] - sh.read_at[b ^ 1][0]);
      const uint8_t* const nr = g.rr + static_cast<int64_t>((win + 1) % kPool) * kWindow * kDigits;
#pragma unroll 1
      for (int i = tid; i < n; i += kThreads) {
        const uint8_t* const row = nr + (nl[i] & 0xffffffu);
        asm volatile("prefetch.global.L2 [%0];" ::"l"(row));
        asm volatile("prefetch.global.L2 [%0];" ::"l"(row + 128));
      }
    }
    //: the window's writes into the warp's 32 digits
    if constexpr (kWrites) {
      const uint8_t* const coef = sh.region + 4 * lane;
      sh.bounds[warp][lane] = make_uint2(static_cast<uint32_t>(o_l), static_cast<uint32_t>(e_l));
#pragma unroll 1
      for (int f0 = first; f0 < last; f0 += kFill) {
        const int tk[4] = {tkn[0], tkn[1], tkn[2], tkn[3]};
        const uint32_t v[4] = {vn[0], vn[1], vn[2], vn[3]};
        if (f0 + kFill < last) fill_load(f0 + kFill, tkn, vn);
        __syncwarp();
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          sh.slot_at[warp][32 * q + lane] = static_cast<uint32_t>(2 * kRows * tk[q]);
          sh.slot_c[warp][32 * q + lane] = sh.m[tk[q]] * static_cast<float>(v[q]);
        }
        __syncwarp();
        const uint32_t live = __ballot_sync(0xffffffffu, e_l > f0 && o_l < f0 + kFill);
        //: each digit the fill overlaps, its entries there four at a time (lists padded to 4, fills a multiple)
#pragma unroll
        for (int d = 0; d < kBlock; ++d) {
          if (((live >> d) & 1u) == 0u) continue;
          const uint2 bd = sh.bounds[warp][d];
          const int lo = max(static_cast<int>(bd.x), f0) - f0,
                    hi = min(static_cast<int>(bd.y), f0 + kFill) - f0;
          //: eight entries an iteration, the second four's coefficients zeroed past the digit
#pragma unroll 1
          for (int x = lo; x < hi; x += 8) {
            const uint4 a4 = *reinterpret_cast<const uint4*>(&sh.slot_at[warp][x]);
            const uint4 b4 = *reinterpret_cast<const uint4*>(&sh.slot_at[warp][x + 4]);
            const float4 c4 = *reinterpret_cast<const float4*>(&sh.slot_c[warp][x]);
            float4 d4 = *reinterpret_cast<const float4*>(&sh.slot_c[warp][x + 4]);
            if (x + 4 >= hi) d4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const float2 q0 = h2f(lds32(coef + a4.x)), q1 = h2f(lds32(coef + a4.y));
            const float2 q2 = h2f(lds32(coef + a4.z)), q3 = h2f(lds32(coef + a4.w));
            const float2 q4 = h2f(lds32(coef + b4.x)), q5 = h2f(lds32(coef + b4.y));
            const float2 q6 = h2f(lds32(coef + b4.z)), q7 = h2f(lds32(coef + b4.w));
            float u0 = c4.x * q0.x, u1 = c4.x * q0.y, w0 = d4.x * q4.x, w1 = d4.x * q4.y;
            u0 = fmaf(c4.y, q1.x, u0);
            u1 = fmaf(c4.y, q1.y, u1);
            w0 = fmaf(d4.y, q5.x, w0);
            w1 = fmaf(d4.y, q5.y, w1);
            u0 = fmaf(c4.z, q2.x, u0);
            u1 = fmaf(c4.z, q2.y, u1);
            w0 = fmaf(d4.z, q6.x, w0);
            w1 = fmaf(d4.z, q6.y, w1);
            u0 = fmaf(c4.w, q3.x, u0);
            u1 = fmaf(c4.w, q3.y, u1);
            w0 = fmaf(d4.w, q7.x, w0);
            w1 = fmaf(d4.w, q7.y, w1);
            mw[0][d] += u0 + w0;
            mw[1][d] += u1 + w1;
          }
        }
      }
    }
  }
  clock.stop(t);
#pragma unroll
  for (int d = 0; d < kBlock; ++d) sink += mw[0][d] + mw[1][d];
  lab::sink(out, lab::bits(sink));
}

uint32_t hash3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA77u ^ (c + 0x165667B1u) * 0xC2B2AE3Du;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  return h ^ (h >> 12);
}

//: a token's `k` live digits of 256, distinct (all of them at 256)
std::vector<int> live_digits(uint32_t seed, uint32_t token, int k) {
  std::vector<int> d;
  if (k >= kDigits) {
    for (int i = 0; i < kDigits; ++i) d.push_back(i);
    return d;
  }
  for (uint32_t j = 0; static_cast<int>(d.size()) < k; ++j) {
    const int x = static_cast<int>(hash3(seed, token, j) % kDigits);
    if (std::find(d.begin(), d.end(), x) == d.end()) d.push_back(x);
  }
  return d;
}

struct Host {
  std::vector<uint32_t> read_at, reads;
  std::vector<uint16_t> offsets, tokens;
  std::vector<uint8_t> values, wr, rr, rs;
  std::vector<__half> ws;
  std::vector<float> m;
};

Host make_host(int write_k, int read_k) {
  Host h;
  const size_t tokens = static_cast<size_t>(kPool) * kWindow;
  const size_t wlist = static_cast<size_t>(kWindow) * write_k + kListPad * kDigits;
  const size_t rlist = static_cast<size_t>(kWindow) * read_k + kListPad * kDigits;
  h.read_at.assign(static_cast<size_t>(kPool) * kOffsets, 0u);
  h.reads.assign(kPool * rlist, 0u);
  h.offsets.assign(static_cast<size_t>(kPool) * kOffsets, 0);
  h.tokens.assign(kPool * wlist, 0);
  h.values.assign(kPool * wlist, 0);
  h.ws.resize(tokens * kDigits);
  h.wr.assign(tokens * kDigits, 0);
  h.rr.resize(tokens * kDigits);
  h.rs.assign(tokens * kDigits, 0);
  h.m.resize(tokens);
  for (size_t t = 0; t < tokens; ++t) {
    for (int d : live_digits(1, static_cast<uint32_t>(t), write_k))
      h.wr[t * kDigits + d] = static_cast<uint8_t>(1 + hash3(2, t, d) % 255);
    for (int d : live_digits(3, static_cast<uint32_t>(t), read_k))
      h.rs[t * kDigits + d] = static_cast<uint8_t>(1 + hash3(4, t, d) % 255);
    for (int d = 0; d < kDigits; ++d) {
      h.ws[t * kDigits + d] = __float2half(1e-3f * (1 + (hash3(5, t, d) & 255)));
      h.rr[t * kDigits + d] = static_cast<uint8_t>(1 + hash3(6, t, d) % 255);
    }
    h.m[t] = 1e-4f * (1 + (t & 7));
  }
  //: both levels' lists digit-major, tokens in order within a digit, each digit's padded to 4 entries (value 0); the
  //: kernel's capacity: a CTA's rows' read lists fit its 2560 entries
  for (size_t win = 0; win < static_cast<size_t>(kPool); ++win) {
    size_t at = 0, rat = 0;
    for (int d = 0; d < kDigits; ++d) {
      h.offsets[win * kOffsets + d] = static_cast<uint16_t>(at);
      h.read_at[win * kOffsets + d] = static_cast<uint32_t>(win * rlist + rat);
      for (int tt = 0; tt < kWindow; ++tt) {
        const size_t t = win * kWindow + tt;
        if (const uint8_t v = h.wr[t * kDigits + d]) {
          h.tokens[win * wlist + at] = static_cast<uint16_t>(tt);
          h.values[win * wlist + at++] = v;
        }
        if (const uint8_t v = h.rs[t * kDigits + d])
          h.reads[win * rlist + rat++] =
              static_cast<uint32_t>(tt) * kDigits | static_cast<uint32_t>(v) << 24;
      }
      at = (at + kListPad - 1) / kListPad * kListPad;
      rat = (rat + kListPad - 1) / kListPad * kListPad;
    }
    h.offsets[win * kOffsets + kDigits] = static_cast<uint16_t>(at);
    h.read_at[win * kOffsets + kDigits] = static_cast<uint32_t>(win * rlist + rat);
    for (int s = 0; s < kDigits; s += kRows)
      if (h.read_at[win * kOffsets + s + kRows] - h.read_at[win * kOffsets + s] > kReadCap)
        std::exit(3);
  }
  return h;
}

//: the den over the first `windows` of the pool in order, in double
std::vector<double> reference(const Host& h, int windows) {
  std::vector<double> mass(static_cast<size_t>(kDigits) * kDigits, 0.0),
      den(static_cast<size_t>(windows) * kWindow);
  for (int w = 0; w < windows; ++w) {
    for (int tt = 0; tt < kWindow; ++tt) {
      const size_t t = static_cast<size_t>(w) * kWindow + tt;
      double s = 0;
      for (int a = 0; a < kDigits; ++a) {
        if (!h.rs[t * kDigits + a]) continue;
        double row = 0;
        for (int c = 0; c < kDigits; ++c) row += h.rr[t * kDigits + c] * mass[a * kDigits + c];
        s += h.rs[t * kDigits + a] * row;
      }
      den[t] = s;
    }
    for (int tt = 0; tt < kWindow; ++tt) {
      const size_t t = static_cast<size_t>(w) * kWindow + tt;
      for (int c = 0; c < kDigits; ++c) {
        if (!h.wr[t * kDigits + c]) continue;
        const double k = h.m[t] * h.wr[t * kDigits + c];
        for (int a = 0; a < kDigits; ++a)
          mass[a * kDigits + c] += k * __half2float(h.ws[t * kDigits + a]);
      }
    }
  }
  return den;
}

struct Device {
  Pool pool{};
  std::vector<void*> owned;

  template <class V>
  const typename V::value_type* up(const V& v) {
    void* p = nullptr;
    const size_t bytes = v.size() * sizeof(v[0]);
    LAB_CHECK(cudaMalloc(&p, bytes));
    LAB_CHECK(cudaMemcpy(p, v.data(), bytes, cudaMemcpyHostToDevice));
    owned.push_back(p);
    return static_cast<const typename V::value_type*>(p);
  }

  explicit Device(const Host& h) {
    pool.read_at = up(h.read_at);
    pool.reads = up(h.reads);
    pool.offsets = up(h.offsets);
    pool.tokens = up(h.tokens);
    pool.values = up(h.values);
    pool.ws = up(h.ws);
    pool.rr = up(h.rr);
    pool.m = up(h.m);
  }

  ~Device() {
    for (void* p : owned) cudaFree(p);
  }
};

constexpr int kWindows = 128, kChecked = 8;

template <int WRITE_K, int READ_K, int PART = 0>
void variant(const lab::Bench& b, float* den) {
  const std::string name = lab::kernel("sweep", {WRITE_K, READ_K, PART});
  if (PART <= 1) b.expect(name, "RED", ">=", 1);
  if (PART <= 1) b.expect(name, "HFMA2", ">=", 64);
  if (PART == 0 || PART == 2) b.expect(name, "FFMA", ">=", 64);
  if (b.expect_only) return;
  const Host h = make_host(WRITE_K, READ_K);
  const Device dv(h);
  const int bytes = static_cast<int>(sizeof(Shared));
  LAB_CHECK(cudaFuncSetAttribute(sweep<WRITE_K, READ_K, PART>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  if (PART == 0) {
    //: the first windows once, against the double reference: every den within 1%
    LAB_CHECK(cudaMemset(den, 0, sizeof(float) * kPool * kWindow));
    sweep<WRITE_K, READ_K, PART>
        <<<kSlices, kThreads, bytes>>>(kChecked, dv.pool, den, b.times, b.sinks);
    LAB_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(kChecked * kWindow);
    LAB_CHECK(cudaMemcpy(got.data(), den, sizeof(float) * got.size(), cudaMemcpyDeviceToHost));
    const std::vector<double> want = reference(h, kChecked);
    double worst = 0;
    for (size_t i = 0; i < got.size(); ++i)
      if (want[i] > 0) worst = std::max(worst, std::abs(got[i] - want[i]) / want[i]);
    if (worst > 1e-2) {
      std::fprintf(stderr, "den_sweep: den off the reference by %.3g (write k %d, read k %d)\n",
                   worst, WRITE_K, READ_K);
      std::exit(1);
    }
  }
  const lab::Samples s = b.variant(
      [&] {
        sweep<WRITE_K, READ_K, PART>
            <<<kSlices, kThreads, bytes>>>(kWindows, dv.pool, den, b.times, b.sinks);
      },
      [&] { return b.span(kWarps, kSlices) / kWindows; });
  const char* part[4] = {"", " / den alone", " / writes alone", " / neither"};
  b.record(std::to_string(WRITE_K) + " live level-R write digits, " + std::to_string(READ_K)
               + " live level-S read digits a token" + part[PART],
           "cycles a window", s);
}

}  // namespace den_sweep

int main(int argc, char** argv) {
  using namespace den_sweep;
  lab::Bench b(argc, argv);
  float* den = nullptr;
  if (!b.expect_only) LAB_CHECK(cudaMalloc(&den, sizeof(float) * kPool * kWindow));
  variant<4, 4>(b, den);
  variant<4, 4, 1>(b, den);
  variant<4, 4, 2>(b, den);
  variant<4, 4, 3>(b, den);
  variant<16, 16>(b, den);
  variant<16, 16, 1>(b, den);
  variant<16, 16, 2>(b, den);
}
