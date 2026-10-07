// measure/lab/experiments/den_sweep.cu -- QUESTION: the carry's window-aligned denominator depends only on routing
// (den(t) = the read leaf product against the write mass of the windows before t's -- tests/oracle/reference.py), and
// the same product against the mass of F = m x the writing token's value scale is num(t), with num / den bounding every
// readout channel of t; a pass running beside the carry could hand its readout both, a common unit and each token's
// bound. At 8 CTAs a head (64 level-S rows x 128 level-R digits each, 16 warps), what do they cost a window on the
// cells' own routing, against the carry? METHOD: the cells' routing and gains as the facts pass hands them to the
// carry (`den_sweep.inputs.py`: the probe cells at the flagship widths, 32 windows); a cell whose level R writes and
// level S reads are sparse takes the list form, each half's MMA taking N from its sparse level as 4 of its digits x
// the two channels (m, F), an entry live in two of N's 8 columns. Writes (dense level S, sparse level R, f16 x f16 ->
// fp32): a warp owns 8 level-R digits, two MMAs' N; a 4-digit tile's list entries are K (arranged so each aligned 8
// holds distinct token & 7, as the facts pass could emit them), the 64 rows M, the mass the accumulators, held at the
// window's gain scales (powers of two the facts pass would emit, the gains pre-scaled, the accumulators rescaled when
// they change); each 64 entries fill a warp's slots (token; each channel's gain x value as f16 with the digit's
// column); A = the tokens' staged f16 coefficients by `ldmatrix.trans` (a token's 16-byte chunks XORed with token &
// 7), B = each entry's coefficients in its columns. The den (sparse level S, dense level R, u8 x u8 -> s32): a warp
// owns 4 level-S rows, N, and its band's tiles of 16 read-list entries (token, digit, u8 value), M; K = the CTA's 128
// digits; A = each entry's u8 read row as loaded (a lane's contiguous quarter, the K order the published mass
// follows), the first two tiles loaded before B is built; B = the mass as 16-bit fixed point (a column scaled by the
// power of two putting its largest value in [2^14, 2^15)) in high and low byte planes, two integer MMAs a 32 digits;
// an entry's results are its row's two columns (high x 256 + low) x 2^-e x its value, each added by the lane holding
// it. The mass crosses lanes once a window through shared memory (f16); the read lists stage once the den is done
// with the window's, the next window's read rows prefetching to L2. One head's CTAs; every den and numerator checked
// against a host reference first. The block's span a window; PART isolates the halves.
// see docs/internals/tools/lab.md#den-sweep
#include <fstream>

#include "lab.cuh"

namespace den_sweep {

constexpr int kThreads = 512, kWarps = 16, kWindow = 512, kRows = 64, kDigits = 256, kPool = 32;
constexpr int kBand = 4, kDenK = 128, kOffsets = 264, kChunk = 64, kReadCap = 4096, kSpan = 128;
//: the published mass: a row 16 bytes past its digits, a channel 8 bytes past its rows
constexpr int kPitch = 2 * kSpan + 16, kPlane = kRows * kPitch + 8;

//: THE SHARED TENANTS: one region holding the published mass (f16 `[channel][64][logical digit]`) from the window's
//: start until the den has its B fragments, then the window's level-S write coefficients (f16 `[512][64]`, a token's
//: 16-byte chunks XORed with token & 7); the window's gains (m and F, an f16 pair a token); the window's read lists for
//: the CTA's rows and their offsets; a window ahead the bounds of the window after's and the CTA's write-tile offsets;
//: each warp's slots for 64 write entries (token; each channel's coefficient with its digit's column)
struct alignas(16) Shared {
  uint8_t region[kWindow * kRows * 2];
  uint32_t mf[kWindow];
  uint32_t reads[kReadCap];
  uint32_t read_at[kRows + 8];
  uint32_t ahead[2][2];
  uint32_t tile_at[2][kSpan / 4 + 4];
  alignas(16) uint16_t slot_token[kWarps][kChunk + 16];
  alignas(16) uint32_t slot_c[kWarps][2][kChunk + 16];
};

//: THE POOL, a window each (offsets absolute): level S's read lists digit-major (entries `token << 8 | digit | value <<
//: 24`), level R's write lists by 4-digit tile (`token | (digit & 3) << 9` and u8 values, each aligned 8 of distinct
//: token & 7 where the residues allow), per token level S's f16 write coefficients and level R's u8 read row, the write
//: gains m and F at the window's scales, and the scales
struct Pool {
  const uint32_t* read_at;  // [pool][264]
  const uint32_t* reads;
  const uint32_t* write_at;  // [pool][264]
  const uint16_t* tokens;
  const uint8_t* values;
  const __half* ws;    // [pool x 512][256]
  const uint8_t* rr;   // [pool x 512][256]
  const uint32_t* mf;  // [pool x 512] (m, F) as f16
  const float2* sc;    // [pool] (m's, F's)
};

__device__ __forceinline__ void red_f32(float* p, float v) {
  asm volatile("red.global.add.f32 [%0], %1;" ::"l"(p), "f"(v) : "memory");
}

__device__ __forceinline__ uint32_t hmax2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("max.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

__device__ __forceinline__ float2 h2f(uint32_t h) {
  return __half22float2(*reinterpret_cast<const __half2*>(&h));
}

//: a level-R digit's column in the published mass (relative to the CTA's first): digit `32 tq + 8 kk + e` sits where
//: the integer K index puts lane tq's bytes `8 kk + e` of a read row (`4 tq + e` of k-step kk for e < 4, `16 + 4 tq +
//: e - 4` after)
__device__ __forceinline__ int logical(int d) {
  const int tq = d >> 5, r = d & 31, kk = r >> 3, e = r & 7;
  return 32 * kk + (e < 4 ? 4 * tq + e : 12 + 4 * tq + e);
}

//: the mass (accumulator fragments: rows `16 mt + g (+ 8)`, a lane's two columns two adjacent digits of one channel --
//: the list form's n-tile 4 digits x 2 channels, the dense form's a channel's 8 digits) at the held scales into the
//: region as f16 at its logical columns
template <bool LISTS>
__device__ __forceinline__ void publish(uint8_t* region, const float (&acc)[4][2][4], float2 held,
                                        int warp, int lane) {
  const int g = lane >> 2, tq = lane & 3;
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int ch = LISTS ? tq >> 1 : nt;
      const int d = 8 * warp + (LISTS ? 4 * nt + 2 * (tq & 1) : 2 * tq);
      const float s = ch ? held.y : held.x;
      uint8_t* const p = region + ch * kPlane + (16 * mt + g) * kPitch + 2 * logical(d);
      *reinterpret_cast<uint32_t*>(p) = lab::pack_half2(acc[mt][nt][0] * s, acc[mt][nt][1] * s);
      *reinterpret_cast<uint32_t*>(p + 8 * kPitch) =
          lab::pack_half2(acc[mt][nt][2] * s, acc[mt][nt][3] * s);
    }
}

//: a lane's den B: column g of rows `row` (its k positions), as 16-bit fixed point at the column's power of two, high
//: and low bytes apart; returns the column's 2^-e
__device__ __forceinline__ float fixed_b(const uint8_t* rowp, int tq, uint32_t (&bh)[kDenK / 32][2],
                                         uint32_t (&bl)[kDenK / 32][2]) {
  uint2 h[kDenK / 32][2];
  uint32_t m2 = 0u;
#pragma unroll
  for (int kk = 0; kk < kDenK / 32; ++kk)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      h[kk][j] = *reinterpret_cast<const uint2*>(rowp + 2 * (32 * kk + 16 * j + 4 * tq));
      m2 = hmax2(m2, hmax2(h[kk][j].x, h[kk][j].y));
    }
  uint32_t top = max(m2 & 0xffffu, m2 >> 16);
  top = max(top, __shfl_xor_sync(0xffffffffu, top, 1));
  top = max(top, __shfl_xor_sync(0xffffffffu, top, 2));
  const int e = top == 0u ? 0 : 29 - static_cast<int>(top >> 10);
  const float sc = __int_as_float((127 + e) << 23);
#pragma unroll
  for (int kk = 0; kk < kDenK / 32; ++kk)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const float2 v01 = h2f(h[kk][j].x), v23 = h2f(h[kk][j].y);
      const uint32_t u0 = __float_as_uint(fmaf(v01.x, sc, 8388608.0f));
      const uint32_t u1 = __float_as_uint(fmaf(v01.y, sc, 8388608.0f));
      const uint32_t u2 = __float_as_uint(fmaf(v23.x, sc, 8388608.0f));
      const uint32_t u3 = __float_as_uint(fmaf(v23.y, sc, 8388608.0f));
      const uint32_t t01 = __byte_perm(u0, u1, 0x5410), t23 = __byte_perm(u2, u3, 0x5410);
      bl[kk][j] = __byte_perm(t01, t23, 0x6420);
      bh[kk][j] = __byte_perm(t01, t23, 0x7531);
    }
  return __int_as_float((127 - e) << 23);
}

__device__ __forceinline__ float2 inverse(float2 s) { return make_float2(1.0f / s.x, 1.0f / s.y); }

//: window `win`'s read lists for the CTA's rows (entries `[lo, hi)`) and their offsets, the bounds of the window
//: after's into `ahead[win & 1]`, and the CTA's write-tile offsets into `tile_at[win & 1]`
__device__ __forceinline__ void stage_reads(Shared& sh, const Pool& g, int win, int lo, int hi,
                                            int s0, int r0, int tid) {
  const int b = win & 1, pw = win % kPool, pn = (win + 1) % kPool;
#pragma unroll 1
  for (int i = tid; i < hi - lo; i += kThreads)
    lab::cp_async4(lab::smem(&sh.reads[i]), g.reads + lo + i);
  if (tid <= kRows)
    lab::cp_async4(lab::smem(&sh.read_at[tid]), g.read_at + pw * kOffsets + s0 + tid);
  if (tid < 2)
    lab::cp_async4(lab::smem(&sh.ahead[b][tid]), g.read_at + pn * kOffsets + s0 + kRows * tid);
  if (tid >= kThreads - kSpan / 4 - 1) {
    const int i = tid - (kThreads - kSpan / 4 - 1);
    lab::cp_async4(lab::smem(&sh.tile_at[b][i]), g.write_at + pw * kOffsets + r0 + 4 * i);
  }
}

__device__ __forceinline__ void stage_coefficients(Shared& sh, const Pool& g, int win, int s0,
                                                   int tid) {
  const int64_t t0 = static_cast<int64_t>(win % kPool) * kWindow;
#pragma unroll 1
  for (int i = tid; i < kWindow * 8; i += kThreads) {
    const int tt = i >> 3, part = i & 7;
    lab::cp_async16(lab::smem(sh.region + 128 * tt + 16 * (part ^ (tt & 7))),
                    g.ws + (t0 + tt) * kDigits + s0 + 8 * part);
  }
  if (tid < kWindow / 4) lab::cp_async16(lab::smem(&sh.mf[4 * tid]), g.mf + t0 + 4 * tid);
}

//: A DEN TILE: entries `16 i + g` and `+ 8` of the band's, and the lane's quarter of each one's read row
struct Tile {
  uint32_t ea, eb;
  uint4 a[kDenK / 64], b[kDenK / 64];
};

__device__ __forceinline__ uint32_t word_of(const uint4 (&v)[kDenK / 64], int kk) {
  const uint4& q = v[kk >> 2];
  return (kk & 3) == 0 ? q.x : (kk & 3) == 1 ? q.y : (kk & 3) == 2 ? q.z : q.w;
}

//: PART 0 both halves; 1 the den alone; 2 the writes alone; 3 neither (the staging, the publish, the barriers). The
//: output `den` holds each token's (den, num)
template <int PART>
__global__ void __launch_bounds__(kThreads, 1) sweep(int windows, Pool g, float* den, long long* t,
                                                     int* out) {
  extern __shared__ __align__(128) unsigned char raw[];
  Shared& sh = *reinterpret_cast<Shared*>(raw);
  constexpr bool kDen = PART == 0 || PART == 1, kWrites = PART == 0 || PART == 2;
  constexpr int kSteps = kDenK / 32;
  const int tid = static_cast<int>(threadIdx.x), lane = lab::lane_id(), warp = lab::warp_id();
  const int g8 = lane >> 2, tq = lane & 3, band = warp;
  const int s0 = static_cast<int>(blockIdx.x) % (kDigits / kRows) * kRows;
  const int r0 = static_cast<int>(blockIdx.x) / (kDigits / kRows) * kSpan;
  //: n-tile nt's columns: channel `n >> 2` of digit `8 warp + 4 nt + (n & 3)` (writes), of row `4 band + (n & 3)`
  //: (den); a lane's accumulators channel `tq >> 1`
  float acc[4][2][4] = {};
  stage_reads(sh, g, 0, static_cast<int>(g.read_at[s0]), static_cast<int>(g.read_at[s0 + kRows]),
              s0, r0, tid);
  lab::cp_commit();
  float sink = 0.0f;
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int win = 0; win < windows; ++win) {
    const int b = win & 1, pw = win % kPool;
    const int64_t t0 = static_cast<int64_t>(pw) * kWindow;
    //: the mass accumulates at the window's gain scales; the scales it holds
    const float2 held = win > 0 ? inverse(g.sc[(win - 1) % kPool]) : make_float2(1.0f, 1.0f);
    lab::cp_wait<0>();
    __syncthreads();
    //: the mass as of the windows before into the region
    publish<true>(sh.region, acc, held, warp, lane);
    __syncthreads();
    //: the den's first two tiles in flight before its B is built (the read lists landed with the last window)
    [[maybe_unused]] Tile ta, tb;
    const uint32_t* const rl = sh.reads;
    const int base = static_cast<int>(sh.read_at[0]);
    const int w_lo = static_cast<int>(sh.read_at[kBand * band]) - base;
    const int w_hi = static_cast<int>(sh.read_at[kBand * band + kBand]) - base;
    const int tiles = (w_hi - w_lo + 15) >> 4;
    const uint8_t* const rr = g.rr + t0 * kDigits + r0 + kDenK / 4 * tq;
    float* const dw = den + 2 * t0;
    const auto load = [&](Tile& x, int i) {
      const int ia = w_lo + 16 * i + g8, ib = ia + 8;
      x.ea = ia < w_hi ? rl[ia] : 0u;
      x.eb = ib < w_hi ? rl[ib] : 0u;
      const uint4* const pa = reinterpret_cast<const uint4*>(rr + (x.ea & 0x1ff00u));
      const uint4* const pb = reinterpret_cast<const uint4*>(rr + (x.eb & 0x1ff00u));
#pragma unroll
      for (int j = 0; j < kDenK / 64; ++j) {
        x.a[j] = pa[j];
        x.b[j] = pb[j];
      }
    };
    if constexpr (kDen) {
      load(ta, 0);
      load(tb, 1);
    }
    uint32_t bh[kSteps][2], bl[kSteps][2];
    float inv_even = 1.0f, inv_odd = 1.0f;
    if constexpr (kDen) {
      const float inv =
          fixed_b(sh.region + (g8 >> 2) * kPlane + (kBand * band + (g8 & 3)) * kPitch, tq, bh, bl);
      inv_even = __shfl_sync(0xffffffffu, inv, 8 * tq);
      inv_odd = __shfl_sync(0xffffffffu, inv, 8 * tq + 4);
    }
    __syncthreads();
    //: the region free: the window's coefficients and gains
    stage_coefficients(sh, g, win, s0, tid);
    lab::cp_commit();
    if constexpr (kDen) {
      //: a tile's integer MMAs (the loaded words are A), then each entry's row's columns (high x 256 + low) x 2^-e x
      //: its value, the lanes holding them adding them
      const auto run = [&](const Tile& x) {
        int ch[4] = {}, cl[4] = {};
#pragma unroll
        for (int kk = 0; kk < kSteps; ++kk) {
          const uint32_t a[4] = {word_of(x.a, 2 * kk), word_of(x.b, 2 * kk),
                                 word_of(x.a, 2 * kk + 1), word_of(x.b, 2 * kk + 1)};
          lab::mma_u8u8(ch, a, bh[kk][0], bh[kk][1]);
          lab::mma_u8u8(cl, a, bl[kk][0], bl[kk][1]);
        }
        const int ra = static_cast<int>(x.ea & 0xffu) - s0 - kBand * band;
        const int rb = static_cast<int>(x.eb & 0xffu) - s0 - kBand * band;
        if ((x.ea >> 24) != 0u && (ra >> 1) == (tq & 1)) {
          const bool odd = ra & 1;
          const float v = static_cast<float>(odd ? ch[1] : ch[0]) * 256.0f
                          + static_cast<float>(odd ? cl[1] : cl[0]);
          red_f32(dw + 2 * ((x.ea >> 8) & 0x1ffu) + (tq >> 1),
                  v * (odd ? inv_odd : inv_even) * static_cast<float>(x.ea >> 24));
        }
        if ((x.eb >> 24) != 0u && (rb >> 1) == (tq & 1)) {
          const bool odd = rb & 1;
          const float v = static_cast<float>(odd ? ch[3] : ch[2]) * 256.0f
                          + static_cast<float>(odd ? cl[3] : cl[2]);
          red_f32(dw + 2 * ((x.eb >> 8) & 0x1ffu) + (tq >> 1),
                  v * (odd ? inv_odd : inv_even) * static_cast<float>(x.eb >> 24));
        }
      };
      //: two tiles in flight, each reloaded two ahead
#pragma unroll 1
      for (int i = 0; i < tiles; i += 2) {
        run(ta);
        load(ta, i + 2);
        if (i + 1 < tiles) run(tb);
        load(tb, i + 3);
      }
    }
    lab::cp_wait<0>();
    __syncthreads();
    //: the den done with this window's lists: the next window's in their place, its read rows toward L2
    if (win + 1 < windows) {
      const int lo = static_cast<int>(sh.ahead[b][0]), hi = static_cast<int>(sh.ahead[b][1]);
      stage_reads(sh, g, win + 1, lo, hi, s0, r0, tid);
      lab::cp_commit();
      if constexpr (kDen) {
        const uint8_t* const nr =
            g.rr + static_cast<int64_t>((win + 1) % kPool) * kWindow * kDigits + r0;
#pragma unroll 1
        for (int i = lo + tid; i < hi; i += kThreads)
          asm volatile("prefetch.global.L2 [%0];" ::"l"(nr + (g.reads[i] & 0x1ff00u)));
      }
    }
    //: the window's writes: each 4-digit tile's entries 64 at a time, 16 a k-step
    if constexpr (kWrites) {
      const uint16_t* const tok = g.tokens;
      const uint8_t* const val = g.values;
      uint16_t* const st = sh.slot_token[warp];
      uint32_t* const sm = sh.slot_c[warp][0];
      uint32_t* const sf = sh.slot_c[warp][1];
      const uint32_t* const mine = sh.slot_c[warp][g8 >> 2];
      const int bound_l = lane <= 2 ? static_cast<int>(sh.tile_at[b][2 * warp + lane]) : 0;
      //: the held mass to this window's gain scale
      const float2 sc = g.sc[pw];
      const float to = tq >> 1 ? sc.y * held.y : sc.x * held.x;
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[mt][nt][i] *= to;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int t_lo = __shfl_sync(0xffffffffu, bound_l, nt),
                  t_hi = __shfl_sync(0xffffffffu, bound_l, nt + 1);
#pragma unroll 1
        for (int c0 = t_lo; c0 < t_hi; c0 += kChunk) {
          __syncwarp();
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int idx = c0 + 32 * h + lane;
            const uint32_t tk = idx < t_hi ? tok[idx] : 0u, v = idx < t_hi ? val[idx] : 0u;
            const uint32_t c = lab::hmul2(sh.mf[tk & 0x1ffu], lab::pack_half2(v, v));
            st[32 * h + lane] = static_cast<uint16_t>(tk & 0x1ffu);
            sm[32 * h + lane] = (c & 0xffffu) | (tk >> 9) << 16;
            sf[32 * h + lane] = (c >> 16) | (tk >> 9) << 16;
          }
          __syncwarp();
          const int steps = (min(kChunk, t_hi - c0) + 15) >> 4;
#pragma unroll 1
          for (int ks = 0; ks < steps; ++ks) {
            //: A: the 16 entries' coefficient rows, four 16-row tiles; B: an entry's coefficient in its digit's
            //: column of the lane's channel
            const int q = lane >> 3, tk = st[16 * ks + 8 * (q >> 1) + (lane & 7)];
            const uint8_t* const row = sh.region + 128 * tk;
            const uint2 lo = *reinterpret_cast<const uint2*>(&mine[16 * ks + 2 * tq]);
            const uint2 hi = *reinterpret_cast<const uint2*>(&mine[16 * ks + 2 * tq + 8]);
            const auto pick = [&](uint32_t w) {
              return ((w >> 16) & 3u) == static_cast<uint32_t>(g8 & 3) ? w & 0xffffu : 0u;
            };
            const uint32_t b0 = pick(lo.x) | pick(lo.y) << 16, b1 = pick(hi.x) | pick(hi.y) << 16;
#pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
              uint32_t a[4];
              lab::ldsm_t(a, lab::smem(row + 16 * ((2 * mt + (q & 1)) ^ (tk & 7))));
              lab::mma_f16_f32(acc[mt][nt], a, b0, b1);
            }
          }
        }
      }
    }
  }
  clock.stop(t);
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) sink += acc[mt][nt][0] + acc[mt][nt][3];
  lab::sink(out, lab::bits(sink));
}

//: THE DENSE FORM: a cell whose levels are dense on both sides takes no lists, and each MMA one channel. Writes: K =
//: the window's tokens in chunks of 64 (level S's f16 coefficients and level R's f16 values staged double-buffered, a
//: token's 16-byte chunks XORed with token & 7), A = the coefficients by `ldmatrix.trans`, B = the values by
//: `ldmatrix.trans` times each token's gain (f16, at the window's scale); a warp owns 8 level-R digits, an n-tile a
//: channel. The den: every token against every row, the warp's 8 rows (an n-tile a channel, B the mass's high and low
//: planes) over its half of the window's 16-token tiles, each tile's integer results times the token's level-S read
//: coefficients, summed, added once a token a channel a warp.
struct alignas(16) Dense {
  uint8_t mass[2 * kPlane];
  __half ws[2][kChunk][kRows];
  __half wr[2][kChunk][kSpan];
  __half m[2][kChunk], f[2][kChunk];
};

struct DensePool {
  const __half* ws;  // [pool x 512][256] level S's write plane
  const __half* wr;  // [pool x 512][256] level R's write plane
  const uint8_t* rs;
  const uint8_t* rr;
  const __half* m;  // [pool x 512] the gains at the window's scales
  const __half* f;
  const float2* sc;  // [pool]
};

__device__ __forceinline__ void stage_chunk(Dense& sh, const DensePool& g, int64_t t0, int c,
                                            int s0, int r0, int tid) {
  const int b = c & 1;
  const int64_t base = t0 + kChunk * c;
  //: 64 tokens x 8 chunks of level S, 64 x 16 of level R, one copy a thread each and 3 more
#pragma unroll 1
  for (int i = tid; i < kChunk * 8; i += kThreads) {
    const int tt = i >> 3, part = i & 7;
    lab::cp_async16(lab::smem(&sh.ws[b][tt][8 * (part ^ (tt & 7))]),
                    g.ws + (base + tt) * kDigits + s0 + 8 * part);
  }
#pragma unroll 1
  for (int i = tid; i < kChunk * 16; i += kThreads) {
    const int tt = i >> 4, part = i & 15;
    lab::cp_async16(lab::smem(&sh.wr[b][tt][8 * (part ^ (tt & 7))]),
                    g.wr + (base + tt) * kDigits + r0 + 8 * part);
  }
  if (tid < kChunk / 8) lab::cp_async16(lab::smem(&sh.m[b][8 * tid]), g.m + base + 8 * tid);
  if (tid >= kThreads - kChunk / 8) {
    const int i = tid - (kThreads - kChunk / 8);
    lab::cp_async16(lab::smem(&sh.f[b][8 * i]), g.f + base + 8 * i);
  }
}

//: PART 0 both halves; 1 the den alone; 2 the writes alone
template <int PART>
__global__ void __launch_bounds__(kThreads, 1) sweep_dense(int windows, DensePool g, float* den,
                                                           long long* t, int* out) {
  extern __shared__ __align__(128) unsigned char raw[];
  Dense& sh = *reinterpret_cast<Dense*>(raw);
  constexpr bool kDen = PART == 0 || PART == 1, kWrites = PART == 0 || PART == 2;
  constexpr int kSteps = kDenK / 32, kTokenTiles = kWindow / 16, kChunks = kWindow / kChunk;
  const int tid = static_cast<int>(threadIdx.x), lane = lab::lane_id(), warp = lab::warp_id();
  const int g8 = lane >> 2, tq = lane & 3;
  const int s0 = static_cast<int>(blockIdx.x) % (kDigits / kRows) * kRows;
  const int r0 = static_cast<int>(blockIdx.x) / (kDigits / kRows) * kSpan;
  //: the den warp's rows (from `8 group`) and its half of the token tiles
  const int group = warp & 7, tile0 = warp >> 3;
  float acc[4][2][4] = {};
  float sink = 0.0f;
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int win = 0; win < windows; ++win) {
    const int pw = win % kPool;
    const int64_t t0 = static_cast<int64_t>(pw) * kWindow;
    const float2 held = win > 0 ? inverse(g.sc[(win - 1) % kPool]) : make_float2(1.0f, 1.0f);
    lab::cp_wait<0>();
    __syncthreads();
    if constexpr (kWrites) {
      stage_chunk(sh, g, t0, 0, s0, r0, tid);
      lab::cp_commit();
    }
    publish<false>(sh.mass, acc, held, warp, lane);
    __syncthreads();
    if constexpr (kDen) {
      uint32_t bh[2][kSteps][2], bl[2][kSteps][2];
      float inv[2][2];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const float iv =
            fixed_b(sh.mass + nt * kPlane + (8 * group + g8) * kPitch, tq, bh[nt], bl[nt]);
        inv[nt][0] = __shfl_sync(0xffffffffu, iv, 8 * tq);
        inv[nt][1] = __shfl_sync(0xffffffffu, iv, 8 * tq + 4);
      }
      //: the tiles: tokens g and g + 8 of each, the lane's quarter of their read rows; each token's level-S read
      //: coefficients for the lane's columns
      const uint8_t* const rr = g.rr + t0 * kDigits + r0 + kDenK / 4 * tq;
      const uint8_t* const rs = g.rs + t0 * kDigits + s0 + 8 * group + 2 * tq;
      float* const dw = den + 2 * t0;
      const auto load = [&](Tile& x, int i) {
        const int ta = 16 * i + g8;
        const uint4* const pa = reinterpret_cast<const uint4*>(rr + ta * kDigits);
        const uint4* const pb = reinterpret_cast<const uint4*>(rr + (ta + 8) * kDigits);
#pragma unroll
        for (int j = 0; j < kDenK / 64; ++j) {
          x.a[j] = pa[j];
          x.b[j] = pb[j];
        }
      };
      //: one tile in flight, reloaded once its MMAs have its words (two would spill)
      Tile x;
      load(x, tile0);
#pragma unroll 1
      for (int ti = tile0; ti < kTokenTiles; ti += 2) {
        int ch[2][4] = {}, cl[2][4] = {};
#pragma unroll
        for (int kk = 0; kk < kSteps; ++kk) {
          const uint32_t a[4] = {word_of(x.a, 2 * kk), word_of(x.b, 2 * kk),
                                 word_of(x.a, 2 * kk + 1), word_of(x.b, 2 * kk + 1)};
#pragma unroll
          for (int nt = 0; nt < 2; ++nt) {
            lab::mma_u8u8(ch[nt], a, bh[nt][kk][0], bh[nt][kk][1]);
            lab::mma_u8u8(cl[nt], a, bl[nt][kk][0], bl[nt][kk][1]);
          }
        }
        const int tk = 16 * ti + g8;
        if (ti + 2 < kTokenTiles) load(x, ti + 2);
        const uint32_t qa = *reinterpret_cast<const uint16_t*>(rs + tk * kDigits);
        const uint32_t qb = *reinterpret_cast<const uint16_t*>(rs + (tk + 8) * kDigits);
        float da[2] = {}, db[2] = {};
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int c = 0; c < 2; ++c) {
            const float va = static_cast<float>(ch[nt][c] * 256 + cl[nt][c]) * inv[nt][c];
            const float vb = static_cast<float>(ch[nt][2 + c] * 256 + cl[nt][2 + c]) * inv[nt][c];
            da[nt] += va * static_cast<float>((qa >> (8 * c)) & 0xffu);
            db[nt] += vb * static_cast<float>((qb >> (8 * c)) & 0xffu);
          }
        //: over the quad, the lane keeping channel `tq >> 1`
        const bool f = tq & 2;
        float ka = f ? da[1] : da[0], kb = f ? db[1] : db[0];
        ka += __shfl_xor_sync(0xffffffffu, f ? da[0] : da[1], 2);
        kb += __shfl_xor_sync(0xffffffffu, f ? db[0] : db[1], 2);
        ka += __shfl_xor_sync(0xffffffffu, ka, 1);
        kb += __shfl_xor_sync(0xffffffffu, kb, 1);
        if ((tq & 1) == 0) {
          red_f32(dw + 2 * tk + (tq >> 1), ka);
          red_f32(dw + 2 * (tk + 8) + (tq >> 1), kb);
        }
      }
      //: the next window's read rows toward L2
      if (win + 1 < windows) {
        const uint8_t* const nr =
            g.rr + static_cast<int64_t>((win + 1) % kPool) * kWindow * kDigits + r0;
        asm volatile("prefetch.global.L2 [%0];" ::"l"(nr + tid * kDigits));
      }
    }
    //: the window's writes: a warp's 8 digits over the window's tokens, 64 a chunk, two chunks in flight
    if constexpr (kWrites) {
      const float2 sc = g.sc[pw];
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          acc[mt][0][i] *= sc.x * held.x;
          acc[mt][1][i] *= sc.y * held.y;
        }
#pragma unroll 1
      for (int c = 0; c < kChunks; ++c) {
        lab::cp_wait<0>();
        __syncthreads();
        if (c + 1 < kChunks) {
          stage_chunk(sh, g, t0, c + 1, s0, r0, tid);
          lab::cp_commit();
        }
        const int b = c & 1, q = lane >> 3;
#pragma unroll
        for (int kp = 0; kp < kChunk / 32; ++kp) {
          //: B for k-steps 2 kp and 2 kp + 1: the warp's 8 digits over 32 tokens, times each token's gains
          const int tb_row = 32 * kp + lane;
          uint32_t bx[4];
          lab::ldsm_t(bx, lab::smem(&sh.wr[b][tb_row][8 * (warp ^ (tb_row & 7))]));
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int k0 = 32 * kp + 16 * h;
            const uint32_t m01 = *reinterpret_cast<const uint32_t*>(&sh.m[b][k0 + 2 * tq]);
            const uint32_t m89 = *reinterpret_cast<const uint32_t*>(&sh.m[b][k0 + 8 + 2 * tq]);
            const uint32_t f01 = *reinterpret_cast<const uint32_t*>(&sh.f[b][k0 + 2 * tq]);
            const uint32_t f89 = *reinterpret_cast<const uint32_t*>(&sh.f[b][k0 + 8 + 2 * tq]);
            const uint32_t bm0 = lab::hmul2(bx[2 * h], m01), bm1 = lab::hmul2(bx[2 * h + 1], m89);
            const uint32_t bf0 = lab::hmul2(bx[2 * h], f01), bf1 = lab::hmul2(bx[2 * h + 1], f89);
            const int tk = k0 + 8 * (q >> 1) + (lane & 7);
#pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
              uint32_t a[4];
              lab::ldsm_t(a, lab::smem(&sh.ws[b][tk][8 * ((2 * mt + (q & 1)) ^ (tk & 7))]));
              lab::mma_f16_f32(acc[mt][0], a, bm0, bm1);
              lab::mma_f16_f32(acc[mt][1], a, bf0, bf1);
            }
          }
        }
      }
    }
  }
  clock.stop(t);
#pragma unroll
  for (int mt = 0; mt < 4; ++mt) sink += acc[mt][0][0] + acc[mt][1][3];
  lab::sink(out, lab::bits(sink));
}

//: A CELL'S ROUTING as `den_sweep.inputs.py` writes it
struct Cell {
  std::string name;
  int tokens = 0, level_s = 0, sparse_write = 0, sparse_read = 0, window = 0;
  std::vector<__half> ws, wr;
  std::vector<uint8_t> rs, rr;
  std::vector<float> m, f;
};

Cell load(const std::string& dir, const std::string& name) {
  std::ifstream in(dir + "/" + name + ".den", std::ios::binary);
  int head[6] = {};
  in.read(reinterpret_cast<char*>(head), sizeof(head));
  if (!in || head[0] != 0x314E4544) {
    std::fprintf(stderr, "den_sweep: %s/%s.den is not a den input\n", dir.c_str(), name.c_str());
    std::exit(2);
  }
  Cell c;
  c.name = name;
  c.tokens = head[1];
  c.level_s = head[2];
  c.sparse_write = head[3];
  c.sparse_read = head[4];
  c.window = head[5];
  const size_t plane = static_cast<size_t>(c.tokens) * kDigits;
  c.ws.resize(plane);
  c.wr.resize(plane);
  c.rs.resize(plane);
  c.rr.resize(plane);
  c.m.resize(c.tokens);
  c.f.resize(c.tokens);
  in.read(reinterpret_cast<char*>(c.ws.data()), static_cast<std::streamsize>(plane * 2));
  in.read(reinterpret_cast<char*>(c.wr.data()), static_cast<std::streamsize>(plane * 2));
  in.read(reinterpret_cast<char*>(c.rs.data()), static_cast<std::streamsize>(plane));
  in.read(reinterpret_cast<char*>(c.rr.data()), static_cast<std::streamsize>(plane));
  in.read(reinterpret_cast<char*>(c.m.data()), static_cast<std::streamsize>(c.tokens) * 4);
  in.read(reinterpret_cast<char*>(c.f.data()), static_cast<std::streamsize>(c.tokens) * 4);
  if (!in || c.window != kWindow || c.tokens != kPool * kWindow) {
    std::fprintf(stderr, "den_sweep: %s is not %d windows of %d tokens\n", name.c_str(), kPool,
                 kWindow);
    std::exit(2);
  }
  return c;
}

//: a window's gain scale: the power of two putting the largest of `gains` in [2^-bits, 2^(1 - bits))
float scale_of(const std::vector<float>& gains, int win, int bits) {
  float top = 0.0f;
  for (int tt = 0; tt < kWindow; ++tt) top = std::max(top, gains[win * kWindow + tt]);
  return top > 0.0f ? std::ldexp(1.0f, -std::ilogb(top) - bits) : 1.0f;
}

//: THE POOL'S HOST SIDE: level S's read lists digit-major; level R's write lists by 4-digit tile, the tile's entries
//: (token order within a digit) taken one from each of the largest residue buckets (token & 7) an aligned 8, the
//: largest again once the residues run out; each channel's window gain scale (largest gain in [2^-4, 2^-3)), the gains
//: pre-scaled. The kernel's capacity: a CTA's rows' read lists fit its 4096 entries
struct Host {
  std::vector<uint32_t> read_at, reads, write_at, mf;
  std::vector<uint16_t> tokens;
  std::vector<uint8_t> values;
  std::vector<float2> sc;
};

Host lists(const Cell& c) {
  Host h;
  h.read_at.assign(static_cast<size_t>(kPool) * kOffsets, 0u);
  h.write_at.assign(static_cast<size_t>(kPool) * kOffsets, 0u);
  h.mf.resize(c.tokens);
  h.sc.resize(kPool);
  for (int win = 0; win < kPool; ++win) {
    h.sc[win] = make_float2(scale_of(c.m, win, 4), scale_of(c.f, win, 4));
    for (int tt = 0; tt < kWindow; ++tt) {
      const int t = win * kWindow + tt;
      h.mf[t] = static_cast<uint32_t>(__half_as_ushort(__float2half(c.m[t] * h.sc[win].x)))
                | static_cast<uint32_t>(__half_as_ushort(__float2half(c.f[t] * h.sc[win].y))) << 16;
    }
    for (int d = 0; d < kDigits; ++d) {
      h.read_at[win * kOffsets + d] = static_cast<uint32_t>(h.reads.size());
      for (int tt = 0; tt < kWindow; ++tt)
        if (const uint8_t v = c.rs[(static_cast<size_t>(win) * kWindow + tt) * kDigits + d])
          h.reads.push_back(static_cast<uint32_t>(tt) << 8 | static_cast<uint32_t>(d)
                            | static_cast<uint32_t>(v) << 24);
    }
    h.read_at[win * kOffsets + kDigits] = static_cast<uint32_t>(h.reads.size());
    for (int tile = 0; tile < kDigits / 4; ++tile) {
      std::vector<std::pair<uint16_t, uint8_t>> bucket[8];
      size_t left = 0;
      for (int d = 4 * tile; d < 4 * tile + 4; ++d) {
        h.write_at[win * kOffsets + d] = static_cast<uint32_t>(h.tokens.size());
        for (int tt = 0; tt < kWindow; ++tt) {
          const float v =
              __half2float(c.wr[(static_cast<size_t>(win) * kWindow + tt) * kDigits + d]);
          if (v != 0.0f) {
            bucket[tt & 7].push_back(
                {static_cast<uint16_t>(tt | (d & 3) << 9), static_cast<uint8_t>(v)});
            ++left;
          }
        }
      }
      const auto emit = [&](int r) {
        h.tokens.push_back(bucket[r].back().first);
        h.values.push_back(bucket[r].back().second);
        bucket[r].pop_back();
        --left;
      };
      while (left > 0) {
        int order[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        std::sort(order, order + 8,
                  [&](int x, int y) { return bucket[x].size() > bucket[y].size(); });
        int taken = 0;
        for (int i = 0; i < 8; ++i)
          if (!bucket[order[i]].empty()) {
            emit(order[i]);
            ++taken;
          }
        for (; taken < 8 && left > 0; ++taken)
          emit(static_cast<int>(
              std::max_element(bucket, bucket + 8,
                               [](const auto& x, const auto& y) { return x.size() < y.size(); })
              - bucket));
      }
    }
    h.write_at[win * kOffsets + kDigits] = static_cast<uint32_t>(h.tokens.size());
    for (int s = 0; s < kDigits; s += kRows)
      if (h.read_at[win * kOffsets + s + kRows] - h.read_at[win * kOffsets + s] > kReadCap) {
        std::fprintf(stderr, "den_sweep: %s window %d holds more read entries than %d\n",
                     c.name.c_str(), win, kReadCap);
        std::exit(3);
      }
  }
  return h;
}

//: each token's (den, num) over the first `windows` of the cell in order, in double (the gains unscaled)
std::vector<double> reference(const Cell& c, int windows) {
  const size_t square = static_cast<size_t>(kDigits) * kDigits;
  std::vector<double> mass(2 * square, 0.0), out(static_cast<size_t>(windows) * kWindow * 2);
  for (int w = 0; w < windows; ++w) {
    for (int tt = 0; tt < kWindow; ++tt) {
      const size_t t = static_cast<size_t>(w) * kWindow + tt;
      for (int ch = 0; ch < 2; ++ch) {
        double s = 0;
        for (int a = 0; a < kDigits; ++a) {
          if (!c.rs[t * kDigits + a]) continue;
          double row = 0;
          for (int r = 0; r < kDigits; ++r)
            row += c.rr[t * kDigits + r] * mass[ch * square + a * kDigits + r];
          s += c.rs[t * kDigits + a] * row;
        }
        out[2 * t + ch] = s;
      }
    }
    for (int tt = 0; tt < kWindow; ++tt) {
      const size_t t = static_cast<size_t>(w) * kWindow + tt;
      for (int r = 0; r < kDigits; ++r) {
        const double v = __half2float(c.wr[t * kDigits + r]);
        if (v == 0) continue;
        for (int a = 0; a < kDigits; ++a) {
          const double x = v * __half2float(c.ws[t * kDigits + a]);
          mass[a * kDigits + r] += c.m[t] * x;
          mass[square + a * kDigits + r] += c.f[t] * x;
        }
      }
    }
  }
  return out;
}

struct Device {
  Pool pool{};
  std::vector<void*> owned;

  template <class V>
  const typename V::value_type* up(const V& v) {
    void* p = nullptr;
    LAB_CHECK(cudaMalloc(&p, std::max<size_t>(v.size() * sizeof(v[0]), 16)));
    LAB_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(v[0]), cudaMemcpyHostToDevice));
    owned.push_back(p);
    return static_cast<const typename V::value_type*>(p);
  }

  Device() = default;

  Device(const Cell& c, const Host& h) {
    pool.read_at = up(h.read_at);
    pool.reads = up(h.reads);
    pool.write_at = up(h.write_at);
    pool.tokens = up(h.tokens);
    pool.values = up(h.values);
    pool.ws = up(c.ws);
    pool.rr = up(c.rr);
    pool.mf = up(h.mf);
    pool.sc = up(h.sc);
  }

  ~Device() {
    for (void* p : owned) cudaFree(p);
  }
};

//: THE DENSE FORM'S POOL: the planes as they are, each channel's gains at its window's scale (the power of two putting
//: the largest in [0.5, 1)) as f16
struct DenseHost {
  std::vector<__half> m, f;
  std::vector<float2> sc;
};

DenseHost dense_gains(const Cell& c) {
  DenseHost h;
  h.m.resize(c.tokens);
  h.f.resize(c.tokens);
  h.sc.resize(kPool);
  for (int win = 0; win < kPool; ++win) {
    h.sc[win] = make_float2(scale_of(c.m, win, 1), scale_of(c.f, win, 1));
    for (int tt = 0; tt < kWindow; ++tt) {
      const int t = win * kWindow + tt;
      h.m[t] = __float2half(c.m[t] * h.sc[win].x);
      h.f[t] = __float2half(c.f[t] * h.sc[win].y);
    }
  }
  return h;
}

constexpr int kWindows = 128, kChecked = 8;

//: each token's den and num over the first windows against the double reference, every one within 1%
template <class Launch>
void check(const Cell& c, float* den, Launch launch) {
  LAB_CHECK(cudaMemset(den, 0, sizeof(float) * kPool * kWindow * 2));
  launch();
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<float> got(kChecked * kWindow * 2);
  LAB_CHECK(cudaMemcpy(got.data(), den, sizeof(float) * got.size(), cudaMemcpyDeviceToHost));
  const std::vector<double> want = reference(c, kChecked);
  double worst[2] = {};
  for (size_t i = 0; i < got.size(); ++i)
    if (want[i] > 0) worst[i & 1] = std::max(worst[i & 1], std::abs(got[i] - want[i]) / want[i]);
  if (worst[0] > 1e-2 || worst[1] > 1e-2) {
    std::fprintf(stderr, "den_sweep: den off the reference by %.3g, num by %.3g on %s\n", worst[0],
                 worst[1], c.name.c_str());
    std::exit(1);
  }
}

template <int PART>
void dense_variant(const lab::Bench& b, float* den, const Cell* c) {
  const std::string name = lab::kernel("sweep_dense", {PART});
  if (PART <= 1) b.expect(name, "IMMA.16832.U8.U8", ">=", 16);
  if (PART == 0 || PART == 2) b.expect(name, "HMMA.16816.F32", ">=", 16);
  if (PART == 0 || PART == 2) b.expect(name, "LDSM", ">=", 5);
  if (b.expect_only) return;
  const DenseHost h = dense_gains(*c);
  Device dv;
  DensePool pool{dv.up(c->ws), dv.up(c->wr), dv.up(c->rs), dv.up(c->rr),
                 dv.up(h.m),   dv.up(h.f),   dv.up(h.sc)};
  const int bytes = static_cast<int>(sizeof(Dense)), grid = (kDigits / kRows) * (kDigits / kSpan);
  LAB_CHECK(
      cudaFuncSetAttribute(sweep_dense<PART>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  if (PART == 0)
    check(*c, den, [&] {
      sweep_dense<PART><<<grid, kThreads, bytes>>>(kChecked, pool, den, b.times, b.sinks);
    });
  const lab::Samples s = b.variant(
      [&] { sweep_dense<PART><<<grid, kThreads, bytes>>>(kWindows, pool, den, b.times, b.sinks); },
      [&] { return b.span(kWarps, grid) / kWindows; });
  const char* part[3] = {"", " / den alone", " / writes alone"};
  b.record(c->name + " (8 CTAs a head, dense form)" + part[PART], "cycles a window", s);
}

template <int PART>
void variant(const lab::Bench& b, float* den, const Cell* c, const Host* h) {
  const std::string name = lab::kernel("sweep", {PART});
  if (PART <= 1) b.expect(name, "IMMA.16832.U8.U8", ">=", kDenK / 16);
  if (PART == 0 || PART == 2) b.expect(name, "HMMA.16816.F32", ">=", 8);
  if (PART == 0 || PART == 2) b.expect(name, "LDSM", ">=", 8);
  if (b.expect_only) return;
  const Device dv(*c, *h);
  const int bytes = static_cast<int>(sizeof(Shared)), grid = (kDigits / kRows) * (kDigits / kSpan);
  LAB_CHECK(cudaFuncSetAttribute(sweep<PART>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  if (PART == 0)
    check(*c, den, [&] {
      sweep<PART><<<grid, kThreads, bytes>>>(kChecked, dv.pool, den, b.times, b.sinks);
    });
  const lab::Samples s = b.variant(
      [&] { sweep<PART><<<grid, kThreads, bytes>>>(kWindows, dv.pool, den, b.times, b.sinks); },
      [&] { return b.span(kWarps, grid) / kWindows; });
  const char* part[4] = {"", " / den alone", " / writes alone", " / neither"};
  b.record(c->name + " (8 CTAs a head)" + part[PART], "cycles a window", s);
}

}  // namespace den_sweep

int main(int argc, char** argv) {
  using namespace den_sweep;
  lab::Bench b(argc, argv);
  if (b.expect_only) {
    variant<0>(b, nullptr, nullptr, nullptr);
    variant<1>(b, nullptr, nullptr, nullptr);
    variant<2>(b, nullptr, nullptr, nullptr);
    variant<3>(b, nullptr, nullptr, nullptr);
    dense_variant<0>(b, nullptr, nullptr);
    dense_variant<1>(b, nullptr, nullptr);
    dense_variant<2>(b, nullptr, nullptr);
    return 0;
  }
  std::string dir;
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], "--inputs")) dir = argv[i + 1];
  if (dir.empty()) {
    std::fprintf(stderr, "den_sweep: --inputs <directory> (tools/lab.py writes it)\n");
    return 2;
  }
  float* den = nullptr;
  LAB_CHECK(cudaMalloc(&den, sizeof(float) * kPool * kWindow * 2));
  std::ifstream names(dir + "/cells.txt");
  bool first = true;
  for (std::string name; std::getline(names, name);) {
    if (name.empty()) continue;
    const Cell c = load(dir, name);
    if (!c.sparse_write && !c.sparse_read) {
      dense_variant<0>(b, den, &c);
      dense_variant<1>(b, den, &c);
      dense_variant<2>(b, den, &c);
      continue;
    }
    if (!c.sparse_write || !c.sparse_read) {
      std::fprintf(stderr,
                   "den_sweep: %s mixes dense and sparse sides, which neither form here takes\n",
                   name.c_str());
      return 2;
    }
    const Host h = lists(c);
    variant<0>(b, den, &c, &h);
    variant<1>(b, den, &c, &h);
    variant<2>(b, den, &c, &h);
    if (first) variant<3>(b, den, &c, &h);
    first = false;
  }
}
