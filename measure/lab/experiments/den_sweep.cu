// measure/lab/experiments/den_sweep.cu -- QUESTION: the carry's window-aligned denominator depends only on routing
// (den(t) = the read leaf product against the write mass of the windows before t's -- tests/oracle/reference.py), so a
// pass running beside the carry could hand its readout a common unit. With lanes on each half's dense level and loops
// on its sparse level, what does a CTA of 64 level-S rows x 256 level-R digits (4 a head) cost a window, against the
// carry's ~20K, digit-major or token-major? METHOD: writes (dense level S, sparse level R, fp32 accumulate): a warp
// owns 32 level-R digits, a lane 2 level-S rows, the mass in registers. Digit-major walks the warp's lists (each
// digit's padded to 4 entries of token and u8 value) 128 entries at a time through a warp's shared slots, the next 128
// loading meanwhile; each digit the fill overlaps steps its entries four at a time, an entry one contiguous 4-byte read
// of the token's staged f16 coefficients a lane, two FFMAs. Token-major walks the window's tokens 32 at a time, each
// lane holding a token's entries; a token with entries in the warp's digits reads its coefficients once and dispatches
// each entry by a switch (nvcc compiles it to a compare tree). The den (sparse level S, dense level R, fp16 like the
// readout): a warp owns 8 level-S rows, a lane 8 level-R digits; the window's mass transposed through shared memory
// once (f16, swizzled) into registers, each row scaled by the power of two that puts its largest value in [32, 64) so
// four u8 x mass products sum in f16 without overflow. A token's read row (8 bytes a lane, PRMT + HSUB2 to exact f16)
// against a row is 4 HFMA2s; digit-major walks each row's read-live tokens, token-major the tokens read-live on any of
// the warp's rows (each live row by a switch), in batches of 4 loaded two batches ahead, reduced across lanes 4 at
// once. One head's 4 CTAs over a 32-window pool (17 MB, streamed from DRAM), as a few pass CTAs would run beside a
// compute-bound carry; checked against a host reference first. The block's span a window; PART isolates the halves.
// see docs/internals/tools/lab.md#den-sweep
#include "lab.cuh"

namespace den_sweep {

constexpr int kThreads = 256, kWarps = 8, kWindow = 512, kWords = 16, kRows = 64, kDigits = 256,
              kPool = 32;
constexpr int kSlices = kDigits / kRows, kBlock = kDigits / kWarps, kBand = kRows / kWarps,
              kOffsets = 264;
constexpr int kBatch = 4, kPairs = 512, kFill = 128, kListPad = 4;
constexpr uint16_t kPad = 0xffffu;

//: THE SHARED TENANTS: one region holding the published mass (f16 `[64][256]`, a row's 16-byte chunks XORed with
//: row / 2) from the window's start until the den has it in registers, then the window's level-S write coefficients
//: (f16 `[512][64]`); the window's gains; the CTA's level-S read rows' words a window ahead; each warp's den tokens
//: and (digit-major) each batch's row; each warp's slots for 128 list entries (token, coefficient)
struct alignas(16) Shared {
  uint8_t region[kWindow * kRows * 2];
  float m[kWindow];
  uint32_t words[2][kRows][kWords];
  uint16_t pairs[kWarps][kPairs];
  uint8_t batch_row[kWarps][kPairs / kBatch];
  alignas(16) uint16_t slot_token[kWarps][kFill + 8];
  alignas(16) float slot_c[kWarps][kFill + 8];
};

//: THE POOL, a window each: level S's read words, level R's write lists digit-major (offsets, tokens, u8 values; a
//: digit's list padded to 4 entries with value 0) and token-major (`digit | value << 8`), per token level S's f16 write
//: coefficients, level R's and level S's u8 read rows, the gain
struct Pool {
  const uint32_t* words;     // [pool][256][16]
  const uint16_t* offsets;   // [pool][264]
  const uint16_t* tokens;    // [pool][512 WRITE_K + 1024]
  const uint8_t* values;     // [pool][512 WRITE_K + 1024]
  const uint16_t* by_token;  // [pool][512][WRITE_K]
  const __half* ws;          // [pool x 512][256]
  const uint8_t* rr;         // [pool x 512][256]
  const uint8_t* rs;         // [pool x 512][256]
  const float* m;            // [pool x 512]
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

//: the den's mass: the warp's rows `8 warp + i` (the lane's digits `8 lane ..`, f16 pairs), each scaled by 2^e so its
//: largest value lies in [32, 64); `inv[i]` = 2^-e
__device__ __forceinline__ void load_band(const Shared& sh, uint32_t (&md)[kBand][4],
                                          float (&inv)[kBand], int warp, int lane) {
#pragma unroll
  for (int i = 0; i < kBand; ++i) {
    const int row = kBand * warp + i, chunk = lane ^ ((row >> 1) & 31);
    const uint4 v = *reinterpret_cast<const uint4*>(sh.region + row * 2 * kDigits + 16 * chunk);
    md[i][0] = v.x;
    md[i][1] = v.y;
    md[i][2] = v.z;
    md[i][3] = v.w;
    const uint32_t m2 = hmax2(hmax2(v.x, v.y), hmax2(v.z, v.w));
    uint32_t top = max(m2 & 0xffffu, m2 >> 16);
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) top = max(top, __shfl_xor_sync(0xffffffffu, top, d));
    const int biased = top == 0u ? 20 : max(static_cast<int>(top >> 10), 1);
    const int e = 20 - biased, e1 = min(e, 15), e2 = e - e1;
    const uint32_t s1 = static_cast<uint32_t>(e1 + 15) << 10, s2 = static_cast<uint32_t>(e2 + 15)
                                                                   << 10;
#pragma unroll
    for (int k = 0; k < 4; ++k)
      md[i][k] = lab::hmul2(lab::hmul2(md[i][k], s1 | (s1 << 16)), s2 | (s2 << 16));
    inv[i] = __int_as_float((127 - e) << 23);
  }
}

__device__ __forceinline__ void stage_words(Shared& sh, const Pool& g, int win, int s0, int tid) {
  const int row = tid >> 2, part = tid & 3;
  lab::cp_async16(lab::smem(&sh.words[win & 1][row][4 * part]),
                  g.words + ((win % kPool) * kDigits + s0 + row) * kWords + 4 * part);
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

//: A WARP'S DEN TOKENS: the set bits of `bits` (lane k and k + 16 holding word k) appended at `pos` in token order and
//: padded with kPad to a whole batch (the pool's lists fit, checked on the host); returns the padded count
__device__ __forceinline__ int append(uint16_t* list, uint32_t bits, int pos, int lane) {
  const int k = lane & 15, n = __popc(bits);
  int inc = n;
#pragma unroll
  for (int d = 1; d < 16; d <<= 1) {
    const int x = __shfl_up_sync(0xffffffffu, inc, d, 16);
    if (k >= d) inc += x;
  }
  const int total = __shfl_sync(0xffffffffu, inc, 15),
            padded = (total + kBatch - 1) & ~(kBatch - 1);
  if (lane < 16) {
    int at = pos + inc - n;
#pragma unroll 1
    while (bits != 0u) {
      list[at++] = static_cast<uint16_t>(32 * k + __ffs(bits) - 1);
      bits &= bits - 1u;
    }
  }
  if (lane < padded - total) list[pos + total + lane] = kPad;
  return padded;
}

//: A DEN BATCH: 4 tokens' read rows (the lane's 8 level-R bytes) and their level-S read coefficients (digit-major: the
//: batch row's byte; token-major: the warp's 8 rows' bytes)
struct Batch {
  uint2 r[kBatch];
  uint2 q[kBatch];
};

template <int ORDER>
__device__ __forceinline__ void load_batch(Batch& x, const uint16_t* list, const uint8_t* rr,
                                           const uint8_t* rs, int bb, int row, int lane) {
#pragma unroll
  for (int j = 0; j < kBatch; ++j) {
    const uint16_t t = list[kBatch * bb + j];
    const bool live = t != kPad;
    const int tt = live ? t : 0;
    x.r[j] = *reinterpret_cast<const uint2*>(rr + tt * kDigits + 8 * lane);
    if constexpr (ORDER == 0)
      x.q[j].x = live ? rs[tt * kDigits + row] : 0u;
    else
      x.q[j] = live ? *reinterpret_cast<const uint2*>(rs + tt * kDigits) : make_uint2(0u, 0u);
  }
}

//: a token's read row (the lane's 8 bytes) against the band's scaled row I: 4 HFMA2s, each half summing 4 products
template <int I>
__device__ __forceinline__ float dot(const uint32_t (&md)[kBand][4], const uint2& r) {
  uint32_t acc = 0u;
  acc = hfma2(u8_pair(r.x, 0x4140u), md[I][0], acc);
  acc = hfma2(u8_pair(r.x, 0x4342u), md[I][1], acc);
  acc = hfma2(u8_pair(r.y, 0x4140u), md[I][2], acc);
  acc = hfma2(u8_pair(r.y, 0x4342u), md[I][3], acc);
  const float2 f = h2f(acc);
  return f.x + f.y;
}

template <int I>
__device__ __forceinline__ void row_batch(const uint32_t (&md)[kBand][4], const float (&inv)[kBand],
                                          const Batch& x, float (&p)[kBatch]) {
#pragma unroll
  for (int j = 0; j < kBatch; ++j)
    p[j] = static_cast<float>(x.q[j].x) * inv[I] * dot<I>(md, x.r[j]);
}

__device__ __forceinline__ uint32_t row_byte(const uint2& q, int i) {
  return ((i < 4 ? q.x : q.y) >> (8 * (i & 3))) & 0xffu;
}

//: 4 per-lane partials reduced across the warp: lane `8 p` ends holding pair p's sum
__device__ __forceinline__ float reduce4(const float (&p)[kBatch], int lane) {
  const bool h4 = lane & 16, h3 = lane & 8;
  float q2[2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
    q2[i] = (h4 ? p[i + 2] : p[i]) + __shfl_xor_sync(0xffffffffu, h4 ? p[i] : p[i + 2], 16);
  float q1 = (h3 ? q2[1] : q2[0]) + __shfl_xor_sync(0xffffffffu, h3 ? q2[0] : q2[1], 8);
  q1 += __shfl_xor_sync(0xffffffffu, q1, 4);
  q1 += __shfl_xor_sync(0xffffffffu, q1, 2);
  return q1 + __shfl_xor_sync(0xffffffffu, q1, 1);
}

//: the bytes of `v` that are nonzero, as four bits
__device__ __forceinline__ uint32_t nonzero4(uint32_t v) {
  const uint32_t top = (((v & 0x7f7f7f7fu) + 0x7f7f7f7fu) | v) & 0x80808080u;
  return (top * 0x00204081u) >> 28;
}

template <int D>
__device__ __forceinline__ void add(float (&mw)[2][kBlock], float c, float2 f) {
  mw[0][D] += c * f.x;
  mw[1][D] += c * f.y;
}

//: PART 0 both halves; 1 the den alone; 2 the writes alone; 3 neither (the staging, the publish, the barriers)
template <int ORDER, int WRITE_K, int READ_K, int PART>
__global__ void __launch_bounds__(kThreads, 1) sweep(int windows, Pool g, float* den, long long* t,
                                                     int* out) {
  extern __shared__ __align__(128) unsigned char raw[];
  Shared& sh = *reinterpret_cast<Shared*>(raw);
  const int tid = static_cast<int>(threadIdx.x), lane = lab::lane_id(), warp = lab::warp_id();
  const int s0 = static_cast<int>(blockIdx.x) * kRows;
  constexpr bool kDen = PART == 0 || PART == 1, kWrites = PART == 0 || PART == 2;
  constexpr int kList = kWindow * WRITE_K + kListPad * kDigits;
  float mw[2][kBlock] = {};
  if (lane < 8) {
    sh.slot_token[warp][kFill + lane] = 0;
    sh.slot_c[warp][kFill + lane] = 0.0f;
  }
  stage_words(sh, g, 0, s0, tid);
  lab::cp_commit();
  lab::cp_wait<0>();
  float sink = 0.0f;
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int win = 0; win < windows; ++win) {
    const int b = win & 1, pw = win % kPool;
    const int64_t t0 = static_cast<int64_t>(pw) * kWindow;
    __syncthreads();
    //: the next window's words; the mass as of the windows before into the region
    if (win + 1 < windows) stage_words(sh, g, win + 1, s0, tid);
    lab::cp_commit();
    publish(sh, mw, warp, lane);
    __syncthreads();
    uint32_t md[kBand][4];
    float inv[kBand];
    int batches = 0;
    if constexpr (kDen) {
      load_band(sh, md, inv, warp, lane);
      uint16_t* const list = sh.pairs[warp];
      if constexpr (ORDER == 0) {
#pragma unroll 1
        for (int i = 0, pos = 0; i < kBand; ++i) {
          const int padded = append(list, sh.words[b][kBand * warp + i][lane & 15], pos, lane);
          for (int x = lane; x < padded / kBatch; x += 32)
            sh.batch_row[warp][pos / kBatch + x] = static_cast<uint8_t>(i);
          pos += padded;
          batches = pos / kBatch;
        }
      } else {
        uint32_t u = 0;
#pragma unroll
        for (int i = 0; i < kBand; ++i) u |= sh.words[b][kBand * warp + i][lane & 15];
        batches = append(list, u, 0, lane) / kBatch;
      }
      __syncwarp();
    }
    __syncthreads();
    //: the region free: the window's coefficients and gains
    stage_coefficients(sh, g, win, s0, tid);
    lab::cp_commit();
    if constexpr (kDen) {
      const uint16_t* const list = sh.pairs[warp];
      const uint8_t* const rr = g.rr + t0 * kDigits;
      const uint8_t* const rs = g.rs + t0 * kDigits + s0 + kBand * warp;
      float* const dw = den + t0;
      const auto row_of = [&](int bb) {
        return ORDER == 0 ? static_cast<int>(sh.batch_row[warp][bb]) : 0;
      };
      Batch cur, nx1, nx2;
      if (batches > 0) load_batch<ORDER>(cur, list, rr, rs, 0, row_of(0), lane);
      if (batches > 1) load_batch<ORDER>(nx1, list, rr, rs, 1, row_of(1), lane);
#pragma unroll 1
      for (int bb = 0; bb < batches; ++bb) {
        if (bb + 2 < batches) load_batch<ORDER>(nx2, list, rr, rs, bb + 2, row_of(bb + 2), lane);
        float p[kBatch];
        if constexpr (ORDER == 0) {
          switch (row_of(bb)) {
            case 0:
              row_batch<0>(md, inv, cur, p);
              break;
            case 1:
              row_batch<1>(md, inv, cur, p);
              break;
            case 2:
              row_batch<2>(md, inv, cur, p);
              break;
            case 3:
              row_batch<3>(md, inv, cur, p);
              break;
            case 4:
              row_batch<4>(md, inv, cur, p);
              break;
            case 5:
              row_batch<5>(md, inv, cur, p);
              break;
            case 6:
              row_batch<6>(md, inv, cur, p);
              break;
            case 7:
              row_batch<7>(md, inv, cur, p);
              break;
            default:
              __trap();
          }
        } else {
#pragma unroll
          for (int j = 0; j < kBatch; ++j) {
            const uint2 q = cur.q[j], x = cur.r[j];
            uint32_t live = nonzero4(q.x) | (nonzero4(q.y) << 4);
            float acc = 0.0f;
#pragma unroll 1
            while (live != 0u) {
              const int i = __ffs(live) - 1;
              live &= live - 1u;
              float a;
              switch (i) {
                case 0:
                  a = static_cast<float>(row_byte(q, 0)) * inv[0] * dot<0>(md, x);
                  break;
                case 1:
                  a = static_cast<float>(row_byte(q, 1)) * inv[1] * dot<1>(md, x);
                  break;
                case 2:
                  a = static_cast<float>(row_byte(q, 2)) * inv[2] * dot<2>(md, x);
                  break;
                case 3:
                  a = static_cast<float>(row_byte(q, 3)) * inv[3] * dot<3>(md, x);
                  break;
                case 4:
                  a = static_cast<float>(row_byte(q, 4)) * inv[4] * dot<4>(md, x);
                  break;
                case 5:
                  a = static_cast<float>(row_byte(q, 5)) * inv[5] * dot<5>(md, x);
                  break;
                case 6:
                  a = static_cast<float>(row_byte(q, 6)) * inv[6] * dot<6>(md, x);
                  break;
                case 7:
                  a = static_cast<float>(row_byte(q, 7)) * inv[7] * dot<7>(md, x);
                  break;
                default:
                  __trap();
              }
              acc += a;
            }
            p[j] = acc;
          }
        }
        const float sum = reduce4(p, lane);
        const uint16_t tk = list[kBatch * bb + (lane >> 3)];
        if ((lane & 7) == 0 && tk != kPad) red_f32(dw + tk, sum);
        cur = nx1;
        nx1 = nx2;
      }
    }
    lab::cp_wait<0>();
    __syncthreads();
    //: the window's writes into the warp's 32 digits
    if constexpr (kWrites) {
      const uint8_t* const coef = sh.region + 4 * lane;
      if constexpr (ORDER == 0) {
        const uint16_t* const off = g.offsets + pw * kOffsets + kBlock * warp;
        const uint16_t* const tok = g.tokens + pw * kList;
        const uint8_t* const val = g.values + pw * kList;
        const int o_l = off[lane], e_l = off[lane + 1];
        const int first = __shfl_sync(0xffffffffu, o_l, 0),
                  last = __shfl_sync(0xffffffffu, e_l, 31);
        //: the warp's entries from `f0`, 128 at a time, four a lane
        const auto load = [&](int f0, int(&tk)[4], uint32_t(&v)[4]) {
#pragma unroll
          for (int h = 0; h < 4; ++h) {
            const int idx = f0 + 32 * h + lane;
            tk[h] = idx < last ? tok[idx] : 0;
            v[h] = idx < last ? val[idx] : 0u;
          }
        };
        int tkn[4];
        uint32_t vn[4];
        load(first, tkn, vn);
#pragma unroll 1
        for (int f0 = first; f0 < last; f0 += kFill) {
          const int tk[4] = {tkn[0], tkn[1], tkn[2], tkn[3]};
          const uint32_t v[4] = {vn[0], vn[1], vn[2], vn[3]};
          if (f0 + kFill < last) load(f0 + kFill, tkn, vn);
          __syncwarp();
#pragma unroll
          for (int h = 0; h < 4; ++h) {
            sh.slot_token[warp][32 * h + lane] = static_cast<uint16_t>(tk[h]);
            sh.slot_c[warp][32 * h + lane] = sh.m[tk[h]] * static_cast<float>(v[h]);
          }
          __syncwarp();
          //: each digit the fill overlaps, its entries there four at a time (lists padded to 4, fills a multiple)
#pragma unroll
          for (int d = 0; d < kBlock; ++d) {
            const int lo = max(__shfl_sync(0xffffffffu, o_l, d), f0) - f0;
            const int hi = min(__shfl_sync(0xffffffffu, e_l, d), f0 + kFill) - f0;
#pragma unroll 1
            for (int e = lo; e < hi; e += 4) {
              const uint2 t4 = *reinterpret_cast<const uint2*>(&sh.slot_token[warp][e]);
              const float4 c4 = *reinterpret_cast<const float4*>(&sh.slot_c[warp][e]);
              const float2 f0v = h2f(lds32(coef + 2 * kRows * (t4.x & 0xffffu)));
              const float2 f1v = h2f(lds32(coef + 2 * kRows * (t4.x >> 16)));
              const float2 f2v = h2f(lds32(coef + 2 * kRows * (t4.y & 0xffffu)));
              const float2 f3v = h2f(lds32(coef + 2 * kRows * (t4.y >> 16)));
              mw[0][d] += c4.x * f0v.x + c4.y * f1v.x + c4.z * f2v.x + c4.w * f3v.x;
              mw[1][d] += c4.x * f0v.y + c4.y * f1v.y + c4.z * f2v.y + c4.w * f3v.y;
            }
          }
        }
      } else {
        const uint16_t* const bt = g.by_token + t0 * WRITE_K;
#pragma unroll 1
        for (int grp = 0; grp < kWindow / 32; ++grp) {
          const uint16_t* const mine = bt + (32 * grp + lane) * WRITE_K;
          uint32_t mask = 0;
#pragma unroll
          for (int e = 0; e < WRITE_K; ++e)
            if (((mine[e] & 0xffu) >> 5) == static_cast<uint32_t>(warp)) mask |= 1u << e;
          uint32_t live = __ballot_sync(0xffffffffu, mask != 0u);
#pragma unroll 1
          while (live != 0u) {
            const int j = __ffs(live) - 1;
            live &= live - 1u;
            const int tt = 32 * grp + j;
            uint32_t mj = __shfl_sync(0xffffffffu, mask, j);
            const float mt = sh.m[tt];
            const float2 f = h2f(lds32(coef + 2 * kRows * tt));
#pragma unroll 1
            while (mj != 0u) {
              const int e = __ffs(mj) - 1;
              mj &= mj - 1u;
              const uint32_t en = bt[tt * WRITE_K + e];
              const float c = mt * static_cast<float>(en >> 8);
              switch (en & 31u) {
                case 0:
                  add<0>(mw, c, f);
                  break;
                case 1:
                  add<1>(mw, c, f);
                  break;
                case 2:
                  add<2>(mw, c, f);
                  break;
                case 3:
                  add<3>(mw, c, f);
                  break;
                case 4:
                  add<4>(mw, c, f);
                  break;
                case 5:
                  add<5>(mw, c, f);
                  break;
                case 6:
                  add<6>(mw, c, f);
                  break;
                case 7:
                  add<7>(mw, c, f);
                  break;
                case 8:
                  add<8>(mw, c, f);
                  break;
                case 9:
                  add<9>(mw, c, f);
                  break;
                case 10:
                  add<10>(mw, c, f);
                  break;
                case 11:
                  add<11>(mw, c, f);
                  break;
                case 12:
                  add<12>(mw, c, f);
                  break;
                case 13:
                  add<13>(mw, c, f);
                  break;
                case 14:
                  add<14>(mw, c, f);
                  break;
                case 15:
                  add<15>(mw, c, f);
                  break;
                case 16:
                  add<16>(mw, c, f);
                  break;
                case 17:
                  add<17>(mw, c, f);
                  break;
                case 18:
                  add<18>(mw, c, f);
                  break;
                case 19:
                  add<19>(mw, c, f);
                  break;
                case 20:
                  add<20>(mw, c, f);
                  break;
                case 21:
                  add<21>(mw, c, f);
                  break;
                case 22:
                  add<22>(mw, c, f);
                  break;
                case 23:
                  add<23>(mw, c, f);
                  break;
                case 24:
                  add<24>(mw, c, f);
                  break;
                case 25:
                  add<25>(mw, c, f);
                  break;
                case 26:
                  add<26>(mw, c, f);
                  break;
                case 27:
                  add<27>(mw, c, f);
                  break;
                case 28:
                  add<28>(mw, c, f);
                  break;
                case 29:
                  add<29>(mw, c, f);
                  break;
                case 30:
                  add<30>(mw, c, f);
                  break;
                case 31:
                  add<31>(mw, c, f);
                  break;
                default:
                  __trap();
              }
            }
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
  std::vector<uint32_t> words;
  std::vector<uint16_t> offsets, tokens, by_token;
  std::vector<uint8_t> values, wr, rr, rs;
  std::vector<__half> ws;
  std::vector<float> m;
};

Host make_host(int write_k, int read_k) {
  Host h;
  const size_t tokens = static_cast<size_t>(kPool) * kWindow;
  h.words.assign(static_cast<size_t>(kPool) * kDigits * kWords, 0u);
  h.offsets.assign(static_cast<size_t>(kPool) * kOffsets, 0);
  const size_t list = static_cast<size_t>(kWindow) * write_k + kListPad * kDigits;
  h.tokens.assign(kPool * list, 0);
  h.values.assign(kPool * list, 0);
  h.by_token.assign(tokens * write_k, 0);
  h.ws.resize(tokens * kDigits);
  h.wr.assign(tokens * kDigits, 0);
  h.rr.resize(tokens * kDigits);
  h.rs.assign(tokens * kDigits, 0);
  h.m.resize(tokens);
  for (size_t t = 0; t < tokens; ++t) {
    const size_t win = t / kWindow;
    const int tt = static_cast<int>(t % kWindow);
    int e = 0;
    for (int d : live_digits(1, static_cast<uint32_t>(t), write_k)) {
      const uint8_t v = static_cast<uint8_t>(1 + hash3(2, t, d) % 255);
      h.wr[t * kDigits + d] = v;
      h.by_token[t * write_k + e++] = static_cast<uint16_t>(d | (v << 8));
    }
    for (int d : live_digits(3, static_cast<uint32_t>(t), read_k)) {
      h.rs[t * kDigits + d] = static_cast<uint8_t>(1 + hash3(4, t, d) % 255);
      h.words[(win * kDigits + d) * kWords + tt / 32] |= 1u << (tt % 32);
    }
    for (int d = 0; d < kDigits; ++d) {
      h.ws[t * kDigits + d] = __float2half(1e-3f * (1 + (hash3(5, t, d) & 255)));
      h.rr[t * kDigits + d] = static_cast<uint8_t>(1 + hash3(6, t, d) % 255);
    }
    h.m[t] = 1e-4f * (1 + (t & 7));
  }
  //: level R's write lists, digit-major, tokens in order within a digit, each padded to 4 entries (value 0)
  for (size_t win = 0; win < static_cast<size_t>(kPool); ++win) {
    size_t at = 0;
    for (int d = 0; d < kDigits; ++d) {
      h.offsets[win * kOffsets + d] = static_cast<uint16_t>(at);
      for (int tt = 0; tt < kWindow; ++tt) {
        const uint8_t v = h.wr[(win * kWindow + tt) * kDigits + d];
        if (!v) continue;
        h.tokens[win * list + at] = static_cast<uint16_t>(tt);
        h.values[win * list + at++] = v;
      }
      at = (at + kListPad - 1) / kListPad * kListPad;
    }
    h.offsets[win * kOffsets + kDigits] = static_cast<uint16_t>(at);
  }
  //: the kernel's capacity: a warp's padded den tokens fit its 512
  for (size_t win = 0; win < static_cast<size_t>(kPool); ++win) {
    for (int band = 0; band < kDigits / kBand; ++band) {
      int rows = 0;
      uint32_t any[kWords] = {};
      for (int i = 0; i < kBand; ++i) {
        int n = 0;
        for (int k = 0; k < kWords; ++k) {
          const uint32_t w = h.words[(win * kDigits + kBand * band + i) * kWords + k];
          n += __builtin_popcount(w);
          any[k] |= w;
        }
        rows += (n + kBatch - 1) / kBatch * kBatch;
      }
      int u = 0;
      for (uint32_t w : any) u += __builtin_popcount(w);
      if (rows > kPairs || u > kPairs) std::exit(3);
    }
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
    pool.words = up(h.words);
    pool.offsets = up(h.offsets);
    pool.tokens = up(h.tokens);
    pool.values = up(h.values);
    pool.by_token = up(h.by_token);
    pool.ws = up(h.ws);
    pool.rr = up(h.rr);
    pool.rs = up(h.rs);
    pool.m = up(h.m);
  }

  ~Device() {
    for (void* p : owned) cudaFree(p);
  }
};

constexpr int kWindows = 128, kChecked = 8;

template <int ORDER, int WRITE_K, int READ_K, int PART = 0>
void variant(const lab::Bench& b, float* den) {
  const std::string name = lab::kernel("sweep", {ORDER, WRITE_K, READ_K, PART});
  if (PART <= 1) b.expect(name, "RED", ">=", 1);
  if (PART == 0 || PART == 2) b.expect(name, "FFMA", ">=", 64);
  if (b.expect_only) return;
  const Host h = make_host(WRITE_K, READ_K);
  const Device dv(h);
  const int bytes = static_cast<int>(sizeof(Shared));
  LAB_CHECK(cudaFuncSetAttribute(sweep<ORDER, WRITE_K, READ_K, PART>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  if (PART == 0) {
    //: the first windows once, against the double reference: every den within 1%
    LAB_CHECK(cudaMemset(den, 0, sizeof(float) * kPool * kWindow));
    sweep<ORDER, WRITE_K, READ_K, PART>
        <<<kSlices, kThreads, bytes>>>(kChecked, dv.pool, den, b.times, b.sinks);
    LAB_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(kChecked * kWindow);
    LAB_CHECK(cudaMemcpy(got.data(), den, sizeof(float) * got.size(), cudaMemcpyDeviceToHost));
    const std::vector<double> want = reference(h, kChecked);
    double worst = 0;
    for (size_t i = 0; i < got.size(); ++i)
      if (want[i] > 0) worst = std::max(worst, std::abs(got[i] - want[i]) / want[i]);
    if (worst > 1e-2) {
      std::fprintf(stderr,
                   "den_sweep: den off the reference by %.3g (order %d, write k %d, read k %d)\n",
                   worst, ORDER, WRITE_K, READ_K);
      std::exit(1);
    }
  }
  const lab::Samples s = b.variant(
      [&] {
        sweep<ORDER, WRITE_K, READ_K, PART>
            <<<kSlices, kThreads, bytes>>>(kWindows, dv.pool, den, b.times, b.sinks);
      },
      [&] { return b.span(kWarps, kSlices) / kWindows; });
  const char* order[2] = {"digit-major", "token-major"};
  const char* part[4] = {"", " / den alone", " / writes alone", " / neither"};
  b.record(std::string(order[ORDER]) + ", " + std::to_string(WRITE_K)
               + " live level-R write digits, " + std::to_string(READ_K)
               + " live level-S read digits a token" + part[PART],
           "cycles a window", s);
}

}  // namespace den_sweep

int main(int argc, char** argv) {
  using namespace den_sweep;
  lab::Bench b(argc, argv);
  float* den = nullptr;
  if (!b.expect_only) LAB_CHECK(cudaMalloc(&den, sizeof(float) * kPool * kWindow));
  variant<0, 4, 4>(b, den);
  variant<0, 4, 4, 1>(b, den);
  variant<0, 4, 4, 2>(b, den);
  variant<0, 4, 4, 3>(b, den);
  variant<0, 16, 16>(b, den);
  variant<1, 4, 4>(b, den);
  variant<1, 4, 4, 1>(b, den);
  variant<1, 4, 4, 2>(b, den);
  variant<1, 16, 16>(b, den);
}
