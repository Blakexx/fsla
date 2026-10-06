// measure/lab/experiments/den_sweep.cu -- QUESTION: the carry's window-aligned denominator depends only on routing
// (den(t) = the read leaf product against the write mass of the windows before t's -- tests/oracle/reference.py), so a
// pass running beside the carry could hand its readout a common unit. On the tensor cores, what does a CTA of 64
// level-S rows x SPAN level-R digits (256: 4 a head; 128: 8 a head) cost a window, against the carry's ~20K? METHOD:
// the den is one channel, so each half's MMA takes N from its sparse level, an entry live in one of N's 8 columns.
// Writes (dense level S, sparse level R, fp32 accumulate): a warp owns SPAN / 8 digits, an 8-digit tile an MMA's N, the
// tile's list entries (padded to 4 a digit) its K, the 64 rows its M, the mass the accumulators; each 64 entries fill a
// warp's slots (token; gain x value as f16 with the digit's column), A = the tokens' staged f16 coefficients by
// `ldmatrix.trans` (a token's 16-byte chunks XORed with token & 7), B = each entry's coefficient in its column. The den
// (sparse level S, dense level R): a warp owns 8 level-S rows, N; its rows' read-list entries (token, digit, u8 value,
// each row's padded to 4, staged a window ahead as a facts pass would emit them) are M, 16 a tile; K = the CTA's
// digits, A = each entry's u8 read row (a lane's contiguous quarter, PRMT + HSUB2 to exact f16), B = the published f16
// mass by `ldmatrix`, fp32 accumulate; an entry's result is its row's column, times its value, added by the lane
// holding it. The mass crosses lanes once a window through shared memory, the next window's read rows prefetch to L2.
// One head's CTAs over a 32-window pool streamed from DRAM, as a few pass CTAs would run beside a compute-bound carry;
// checked against a host reference first. The block's span a window; PART isolates the halves.
// see docs/internals/tools/lab.md#den-sweep
#include "lab.cuh"

namespace den_sweep {

constexpr int kThreads = 256, kWarps = 8, kWindow = 512, kRows = 64, kDigits = 256, kPool = 32;
constexpr int kBand = kRows / kWarps, kOffsets = 264, kChunk = 64, kListPad = 4, kReadCap = 2560;

//: THE SHARED TENANTS: one region holding the published mass (f16 `[64][logical digit]`, a row 16 bytes past 2 SPAN)
//: from the window's start until the den has its B fragments, then the window's level-S write coefficients (f16
//: `[512][64]`, a token's 16-byte chunks XORed with token & 7); the window's gains; a window ahead the CTA's rows' read
//: lists and their offsets; each warp's slots for 64 write entries
struct alignas(16) Shared {
  uint8_t region[kWindow * kRows * 2];
  float m[kWindow];
  uint32_t reads[2][kReadCap];
  uint32_t read_at[2][kRows + 8];
  alignas(16) uint32_t slot_token[kWarps][kChunk + 16];
  alignas(16) uint32_t slot_cd[kWarps][kChunk + 16];
};

//: THE POOL, a window each: level S's read lists digit-major (offsets; entries `token << 8 | digit | value << 24`, a
//: digit's padded to 4 with value 0), level R's write lists digit-major (offsets, `token | (digit & 7) << 9`, u8
//: values, a digit's padded to 4 with value 0), per token level S's f16 write coefficients and level R's u8 read row,
//: the gain
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

//: a level-R digit's column in the published mass (relative to the CTA's first): digit `(SPAN / 4) tq + 4 kk + e` sits
//: where the den's K index puts lane tq's bytes `4 kk + e` of a read row (`2 tq + {0, 1, 8, 9}` of k-step kk)
template <int SPAN>
__device__ __forceinline__ int logical(int d) {
  const int tq = d / (SPAN / 4), kk = (d % (SPAN / 4)) >> 2, e = d & 3;
  return 16 * kk + 2 * tq + (e < 2 ? e : e + 6);
}

template <int SPAN>
__host__ __device__ constexpr int pitch() {
  return 2 * SPAN + 16;
}

//: the mass (accumulator fragments: rows `16 mt + g (+ 8)`, digits `SPAN / 8 warp + 8 nt + 2 tq (+ 1)`) into the region
//: as f16 at its logical columns
template <int SPAN>
__device__ __forceinline__ void publish(Shared& sh, const float (&acc)[4][SPAN / 64][4], int warp,
                                        int lane) {
  const int g = lane >> 2, tq = lane & 3;
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int nt = 0; nt < SPAN / 64; ++nt) {
      const int col = 2 * logical<SPAN>(SPAN / 8 * warp + 8 * nt + 2 * tq);
      *reinterpret_cast<uint32_t*>(sh.region + (16 * mt + g) * pitch<SPAN>() + col) =
          lab::pack_half2(acc[mt][nt][0], acc[mt][nt][1]);
      *reinterpret_cast<uint32_t*>(sh.region + (16 * mt + g + 8) * pitch<SPAN>() + col) =
          lab::pack_half2(acc[mt][nt][2], acc[mt][nt][3]);
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
  for (int i = tid; i < kWindow * 8; i += kThreads) {
    const int tt = i >> 3, part = i & 7;
    lab::cp_async16(lab::smem(sh.region + 128 * tt + 16 * (part ^ (tt & 7))),
                    g.ws + (t0 + tt) * kDigits + s0 + 8 * part);
  }
  if (tid < kWindow / 4) lab::cp_async16(lab::smem(&sh.m[4 * tid]), g.m + t0 + 4 * tid);
}

//: A DEN TILE: entries `16 i + g` and `+ 8` of the warp's, and the lane's quarter of each one's read row
template <int SPAN>
struct Tile {
  uint32_t ea, eb;
  uint4 a[SPAN / 64], b[SPAN / 64];
};

template <int SPAN>
__device__ __forceinline__ uint32_t word_of(const uint4 (&v)[SPAN / 64], int kk) {
  const uint4& q = v[kk >> 2];
  return (kk & 3) == 0 ? q.x : (kk & 3) == 1 ? q.y : (kk & 3) == 2 ? q.z : q.w;
}

//: PART 0 both halves; 1 the den alone; 2 the writes alone; 3 neither (the staging, the publish, the barriers)
template <int SPAN, int WRITE_K, int READ_K, int PART>
__global__ void __launch_bounds__(kThreads, 1) sweep(int windows, Pool g, float* den, long long* t,
                                                     int* out) {
  extern __shared__ __align__(128) unsigned char raw[];
  Shared& sh = *reinterpret_cast<Shared*>(raw);
  constexpr bool kDen = PART == 0 || PART == 1, kWrites = PART == 0 || PART == 2;
  constexpr int kWriteList = kWindow * WRITE_K + kListPad * kDigits, kTiles = SPAN / 64,
                kSteps = SPAN / 16;
  const int tid = static_cast<int>(threadIdx.x), lane = lab::lane_id(), warp = lab::warp_id();
  const int g8 = lane >> 2, tq = lane & 3;
  const int s0 = static_cast<int>(blockIdx.x) % (kDigits / kRows) * kRows;
  const int r0 = static_cast<int>(blockIdx.x) / (kDigits / kRows) * SPAN;
  float acc[4][kTiles][4] = {};
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
    //: the next window's read lists; the window's write-tile bounds; the mass as of the windows before into the region
    if (win + 1 < windows) stage_reads(sh, g, win + 1, next_lo, next_hi, s0, tid);
    lab::cp_commit();
    if (win + 2 < windows) {
      next_lo = g.read_at[((win + 2) % kPool) * kOffsets + s0];
      next_hi = g.read_at[((win + 2) % kPool) * kOffsets + s0 + kRows];
    }
    const int d0 = r0 + SPAN / 8 * warp;
    const int bound_l = lane <= kTiles ? g.offsets[pw * kOffsets + d0 + 8 * lane] : 0;
    publish<SPAN>(sh, acc, warp, lane);
    __syncthreads();
    //: the den's B: the warp's 8 rows over the CTA's logical digits, k-steps in pairs
    uint32_t bf[kSteps][2];
    if constexpr (kDen) {
#pragma unroll
      for (int kp = 0; kp < kSteps / 2; ++kp) {
        uint32_t x[4];
        lab::ldsm(x, lab::smem(sh.region + (kBand * warp + (lane & 7)) * pitch<SPAN>()
                               + 2 * (32 * kp + 8 * (lane >> 3))));
        bf[2 * kp][0] = x[0];
        bf[2 * kp][1] = x[1];
        bf[2 * kp + 1][0] = x[2];
        bf[2 * kp + 1][1] = x[3];
      }
    }
    __syncthreads();
    //: the region free: the window's coefficients and gains
    stage_coefficients(sh, g, win, s0, tid);
    lab::cp_commit();
    if constexpr (kDen) {
      const uint32_t* const rl = sh.reads[b];
      const int base = static_cast<int>(sh.read_at[b][0]);
      const int w_lo = static_cast<int>(sh.read_at[b][kBand * warp]) - base;
      const int w_hi = static_cast<int>(sh.read_at[b][kBand * warp + kBand]) - base;
      const int tiles = (w_hi - w_lo + 15) >> 4;
      const uint8_t* const rr = g.rr + t0 * kDigits + r0 + SPAN / 4 * tq;
      float* const dw = den + t0;
      const auto load = [&](Tile<SPAN>& x, int i) {
        const int ia = w_lo + 16 * i + g8, ib = ia + 8;
        x.ea = ia < w_hi ? rl[ia] : 0u;
        x.eb = ib < w_hi ? rl[ib] : 0u;
        const uint4* const pa = reinterpret_cast<const uint4*>(rr + (x.ea & 0x1ff00u));
        const uint4* const pb = reinterpret_cast<const uint4*>(rr + (x.eb & 0x1ff00u));
#pragma unroll
        for (int j = 0; j < SPAN / 64; ++j) {
          x.a[j] = pa[j];
          x.b[j] = pb[j];
        }
      };
      //: a tile's MMAs, then each entry's row column times its value, added by the lane holding that column
      const auto run = [&](const Tile<SPAN>& x) {
        float c[4] = {};
#pragma unroll
        for (int kk = 0; kk < kSteps; ++kk) {
          const uint32_t wa = word_of<SPAN>(x.a, kk), wb = word_of<SPAN>(x.b, kk);
          const uint32_t a[4] = {u8_pair(wa, 0x4140u), u8_pair(wb, 0x4140u), u8_pair(wa, 0x4342u),
                                 u8_pair(wb, 0x4342u)};
          lab::mma_f16_f32(c, a, bf[kk][0], bf[kk][1]);
        }
        const int ra = static_cast<int>(x.ea & 0xffu) - s0 - kBand * warp;
        const int rb = static_cast<int>(x.eb & 0xffu) - s0 - kBand * warp;
        if ((x.ea >> 24) != 0u && (ra >> 1) == tq)
          red_f32(dw + ((x.ea >> 8) & 0x1ffu),
                  (ra & 1 ? c[1] : c[0]) * static_cast<float>(x.ea >> 24));
        if ((x.eb >> 24) != 0u && (rb >> 1) == tq)
          red_f32(dw + ((x.eb >> 8) & 0x1ffu),
                  (rb & 1 ? c[3] : c[2]) * static_cast<float>(x.eb >> 24));
      };
      //: two tiles in flight, each reloaded two ahead
      Tile<SPAN> ta, tb;
      load(ta, 0);
      load(tb, 1);
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
    //: the next window's read rows toward L2, its lists landed
    if (kDen && win + 1 < windows) {
      const uint32_t* const nl = sh.reads[b ^ 1];
      const int n = static_cast<int>(sh.read_at[b ^ 1][kRows] - sh.read_at[b ^ 1][0]);
      const uint8_t* const nr =
          g.rr + static_cast<int64_t>((win + 1) % kPool) * kWindow * kDigits + r0;
#pragma unroll 1
      for (int i = tid; i < n; i += kThreads) {
        const uint8_t* const row = nr + (nl[i] & 0x1ff00u);
        asm volatile("prefetch.global.L2 [%0];" ::"l"(row));
        if constexpr (SPAN > 128) asm volatile("prefetch.global.L2 [%0];" ::"l"(row + 128));
      }
    }
    //: the window's writes: each 8-digit tile's entries 64 at a time, 16 a k-step
    if constexpr (kWrites) {
      const uint16_t* const tok = g.tokens + pw * kWriteList;
      const uint8_t* const val = g.values + pw * kWriteList;
      uint32_t* const st = sh.slot_token[warp];
      uint32_t* const scd = sh.slot_cd[warp];
#pragma unroll
      for (int nt = 0; nt < kTiles; ++nt) {
        const int t_lo = __shfl_sync(0xffffffffu, bound_l, nt),
                  t_hi = __shfl_sync(0xffffffffu, bound_l, nt + 1);
#pragma unroll 1
        for (int c0 = t_lo; c0 < t_hi; c0 += kChunk) {
          __syncwarp();
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int idx = c0 + 32 * h + lane;
            const uint32_t tk = idx < t_hi ? tok[idx] : 0u, v = idx < t_hi ? val[idx] : 0u;
            const float c = sh.m[tk & 0x1ffu] * static_cast<float>(v);
            st[32 * h + lane] = tk & 0x1ffu;
            scd[32 * h + lane] =
                static_cast<uint32_t>(__half_as_ushort(__float2half_rn(c))) | (tk >> 9) << 16;
          }
          __syncwarp();
          const int steps = (min(kChunk, t_hi - c0) + 15) >> 4;
#pragma unroll 1
          for (int ks = 0; ks < steps; ++ks) {
            //: A: the 16 entries' coefficient rows, four 16-row tiles; B: an entry's coefficient in its digit's column
            const int q = lane >> 3, tk = static_cast<int>(st[16 * ks + 8 * (q >> 1) + (lane & 7)]);
            const uint8_t* const row = sh.region + 128 * tk;
            const uint2 lo = *reinterpret_cast<const uint2*>(&scd[16 * ks + 2 * tq]);
            const uint2 hi = *reinterpret_cast<const uint2*>(&scd[16 * ks + 2 * tq + 8]);
            const auto pick = [&](uint32_t w) {
              return ((w >> 16) & 7u) == static_cast<uint32_t>(g8) ? w & 0xffffu : 0u;
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
    for (int nt = 0; nt < kTiles; ++nt) sink += acc[mt][nt][0] + acc[mt][nt][3];
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
          h.tokens[win * wlist + at] = static_cast<uint16_t>(tt | (d & 7) << 9);
          h.values[win * wlist + at++] = v;
        }
        if (const uint8_t v = h.rs[t * kDigits + d])
          h.reads[win * rlist + rat++] = static_cast<uint32_t>(tt) << 8 | static_cast<uint32_t>(d)
                                         | static_cast<uint32_t>(v) << 24;
      }
      for (; at % kListPad; ++at) h.tokens[win * wlist + at] = static_cast<uint16_t>((d & 7) << 9);
      for (; rat % kListPad; ++rat) h.reads[win * rlist + rat] = static_cast<uint32_t>(d);
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

template <int SPAN, int WRITE_K, int READ_K, int PART = 0>
void variant(const lab::Bench& b, float* den) {
  const std::string name = lab::kernel("sweep", {SPAN, WRITE_K, READ_K, PART});
  if (PART <= 1) b.expect(name, "RED", ">=", 2);
  if (PART <= 1) b.expect(name, "HMMA.16816.F32", ">=", SPAN / 16);
  if (PART == 0 || PART == 2) b.expect(name, "LDSM", ">=", 4);
  if (b.expect_only) return;
  const Host h = make_host(WRITE_K, READ_K);
  const Device dv(h);
  const int bytes = static_cast<int>(sizeof(Shared)), grid = (kDigits / kRows) * (kDigits / SPAN);
  LAB_CHECK(cudaFuncSetAttribute(sweep<SPAN, WRITE_K, READ_K, PART>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  if (PART == 0) {
    //: the first windows once, against the double reference: every den within 1%
    LAB_CHECK(cudaMemset(den, 0, sizeof(float) * kPool * kWindow));
    sweep<SPAN, WRITE_K, READ_K, PART>
        <<<grid, kThreads, bytes>>>(kChecked, dv.pool, den, b.times, b.sinks);
    LAB_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(kChecked * kWindow);
    LAB_CHECK(cudaMemcpy(got.data(), den, sizeof(float) * got.size(), cudaMemcpyDeviceToHost));
    const std::vector<double> want = reference(h, kChecked);
    double worst = 0;
    for (size_t i = 0; i < got.size(); ++i)
      if (want[i] > 0) worst = std::max(worst, std::abs(got[i] - want[i]) / want[i]);
    if (worst > 1e-2) {
      std::fprintf(stderr,
                   "den_sweep: den off the reference by %.3g (span %d, write k %d, read k %d)\n",
                   worst, SPAN, WRITE_K, READ_K);
      std::exit(1);
    }
  }
  const lab::Samples s = b.variant(
      [&] {
        sweep<SPAN, WRITE_K, READ_K, PART>
            <<<grid, kThreads, bytes>>>(kWindows, dv.pool, den, b.times, b.sinks);
      },
      [&] { return b.span(kWarps, grid) / kWindows; });
  const char* part[4] = {"", " / den alone", " / writes alone", " / neither"};
  b.record(std::to_string(kDigits / SPAN * 4) + " CTAs a head (64 x " + std::to_string(SPAN) + "), "
               + std::to_string(WRITE_K) + " live level-R write digits, " + std::to_string(READ_K)
               + " live level-S read digits a token" + part[PART],
           "cycles a window", s);
}

}  // namespace den_sweep

int main(int argc, char** argv) {
  using namespace den_sweep;
  lab::Bench b(argc, argv);
  float* den = nullptr;
  if (!b.expect_only) LAB_CHECK(cudaMalloc(&den, sizeof(float) * kPool * kWindow));
  variant<256, 4, 4>(b, den);
  variant<256, 4, 4, 1>(b, den);
  variant<256, 4, 4, 2>(b, den);
  variant<256, 4, 4, 3>(b, den);
  variant<256, 16, 16>(b, den);
  variant<128, 4, 4>(b, den);
  variant<128, 4, 4, 1>(b, den);
  variant<128, 4, 4, 2>(b, den);
  variant<128, 16, 16>(b, den);
}
