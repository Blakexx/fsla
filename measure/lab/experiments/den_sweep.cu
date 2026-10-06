// measure/lab/experiments/den_sweep.cu -- QUESTION: the carry's window-aligned denominator depends only on routing
// (den(t) = the read leaf product against the write mass of the windows before t's -- tests/oracle/reference.py), so a
// separate pass could compute it ahead of the carry and let the carry's readout add at a common unit. What does that
// pass cost a window at sparse routing, against the carry's CTA-windows? METHOD: a CTA owns 16 level-S rows x all 256
// level-R digits of a head's mass, a thread a level-R digit's 16 values in registers. Each window: the mass published
// as f16 (`[S][logical R]`, the R order the den's operand loads make contiguous); the den as MMAs over the window's
// tokens read-live on any of the CTA's rows (A = the token's u8 level-R read row from global, made f16 by PRMT; B = the
// published mass by `ldmatrix`; N = the 16 rows), each row's result times the token's level-S read coefficient, summed,
// reduced to global a token; the window's writes, each thread over its digit's list (token, u8 value) against the
// level-S write coefficients (f16, a token's two 16-byte halves swapped on its bit 2 so random tokens spread over the
// banks) and the gains, 16 FFMAs an entry, the next entry's coefficients and the one after's token in flight. Half the
// warps write first. Each warp finds its tiles' tokens from the union of the rows' facts words; lists, coefficients and
// gains by `cp.async` a window ahead, words two; a tile's read rows a tile ahead (a window's first from the window
// before). A routing pool per head (8 windows, 22 MB over five heads, so it streams from DRAM) at `WRITE_K` live
// level-R write digits and `READ_K` live level-S read digits a token. Checked against a host reference first. The
// block's span a window; PART isolates the halves.
// see docs/internals/tools/lab.md#den-sweep
#include "lab.cuh"

namespace den_sweep {

constexpr int kThreads = 256, kWarps = 8, kWindow = 512, kWords = 16, kRows = 16, kDigits = 256,
              kPool = 8;
constexpr int kPitch = 2 * kDigits + 16, kOffsets = 264;

//: THE SHARED TENANTS: the published mass; a window ahead level R's write lists, the CTA's level-S write coefficients
//: and the gains; two ahead its level-S read rows' facts words
template <int WRITE_K>
struct alignas(16) Shared {
  uint8_t mass[kRows][kPitch];
  __half ws[2][kWindow][kRows];
  float m[2][kWindow];
  uint32_t words[3][kRows][kWords];
  uint16_t tokens[2][kWindow * WRITE_K];
  uint8_t values[2][kWindow * WRITE_K];
};

//: THE POOL, per head and window: level S's read words, level R's write lists (digit-major offsets, then the entries'
//: tokens and u8 values), per token level S's f16 write coefficients, level R's and level S's u8 read rows, the gain
struct Pool {
  const uint32_t* words;    // [heads][pool][256][16]
  const uint16_t* offsets;  // [heads][pool][264]
  const uint16_t* tokens;   // [heads][pool][512 WRITE_K]
  const uint8_t* values;    // [heads][pool][512 WRITE_K]
  const __half* ws;         // [heads][pool x 512][256]
  const uint8_t* rr;        // [heads][pool x 512][256]
  const uint8_t* rs;        // [heads][pool x 512][256]
  const float* m;           // [heads][pool x 512]
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

//: a level-R digit's column in the published mass: digit `64 tq + 4 kk + e` sits where the MMA's K index puts lane tq's
//: bytes `4 kk + e` of a read row (`2 tq + {0, 1, 8, 9}` of k-step kk)
__device__ __forceinline__ int logical(int r) {
  const int tq = r >> 6, kk = (r >> 2) & 15, e = r & 3;
  return 16 * kk + 2 * tq + (e < 2 ? e : e + 6);
}

__device__ __forceinline__ int64_t window_token(int head, int win) {
  return (static_cast<int64_t>(head) * kPool + win % kPool) * kWindow;
}

//: window `win`'s level-R write lists, the CTA's level-S write coefficients (a token's two halves swapped on its bit 2)
//: and the gains into buffer `win & 1`; the CTA's rows' words into ring slot `win % 3`
template <int WRITE_K>
__device__ __forceinline__ void stage_lists(Shared<WRITE_K>& sh, const Pool& g, int head, int win,
                                            int s0, int tid) {
  const int b = win & 1;
  const int64_t t0 = window_token(head, win), e0 = t0 * WRITE_K;
#pragma unroll 1
  for (int i = tid; i < kWindow * 2; i += kThreads) {
    const int tt = i >> 1, h = (i & 1) ^ (tt & 4 ? 1 : 0);
    lab::cp_async16(lab::smem(&sh.ws[b][tt][8 * h]), g.ws + (t0 + tt) * kDigits + s0 + 8 * (i & 1));
  }
  if (tid < kWindow / 4) lab::cp_async16(lab::smem(&sh.m[b][4 * tid]), g.m + t0 + 4 * tid);
#pragma unroll 1
  for (int i = tid; i < kWindow * WRITE_K / 8; i += kThreads)
    lab::cp_async16(lab::smem(&sh.tokens[b][8 * i]), g.tokens + e0 + 8 * i);
#pragma unroll 1
  for (int i = tid; i < kWindow * WRITE_K / 16; i += kThreads)
    lab::cp_async16(lab::smem(&sh.values[b][16 * i]), g.values + e0 + 16 * i);
}

template <int WRITE_K>
__device__ __forceinline__ void stage_words(Shared<WRITE_K>& sh, const Pool& g, int head, int win,
                                            int s0, int tid) {
  if (tid < kRows * kWords / 4) {
    const int row = tid >> 2, part = tid & 3;
    lab::cp_async16(
        lab::smem(&sh.words[win % 3][row][4 * part]),
        g.words + ((static_cast<int64_t>(head) * kPool + win % kPool) * kDigits + s0 + row) * kWords
            + 4 * part);
  }
}

//: A WARP'S VIEW OF A WINDOW'S READ-LIVE TOKENS (on any of the CTA's rows): lane j (and j + 16) holds word j of their
//: union and its exclusive prefix count
struct Union {
  uint32_t word;
  int pre, count;
};

template <int WRITE_K>
__device__ __forceinline__ Union union_of(const Shared<WRITE_K>& sh, int slot, int lane) {
  const int j = lane & 15;
  uint32_t u = 0;
#pragma unroll
  for (int row = 0; row < kRows; ++row) u |= sh.words[slot][row][j];
  const int n = __popc(u);
  int inc = n;
#pragma unroll
  for (int d = 1; d < 16; d <<= 1) {
    const int x = __shfl_up_sync(0xffffffffu, inc, d, 16);
    if (j >= d) inc += x;
  }
  return {u, inc - n, __shfl_sync(0xffffffffu, inc, 15)};
}

//: the token at union position `p` (< count): the last word whose prefix is at most `p`, then its bit by halving
__device__ __forceinline__ int token_at(const Union& un, int p) {
  int w = 0;
#pragma unroll
  for (int step = 8; step >= 1; step >>= 1)
    if (__shfl_sync(0xffffffffu, un.pre, w + step) <= p) w += step;
  uint32_t bits = __shfl_sync(0xffffffffu, un.word, w);
  int k = p - __shfl_sync(0xffffffffu, un.pre, w), at = 0;
#pragma unroll
  for (int half = 16; half >= 1; half >>= 1) {
    const int c = __popc(bits & ((1u << half) - 1u));
    if (k >= c) {
      k -= c;
      bits >>= half;
      at += half;
    }
  }
  return 32 * w + at;
}

//: A DEN TILE: union positions `16 i + g` and `+ 8`, lane tq's 64 bytes of each token's level-R read row and its two
//: pairs of the CTA's level-S read coefficients
struct Tile {
  uint4 a[4], b[4];
  uint32_t qa[2], qb[2];
  int ta, tb;
};

__device__ __forceinline__ void load_tile(Tile& x, const Union& un, const uint8_t* rr,
                                          const uint8_t* rs, int i, int lane) {
  const int g = lane >> 2, tq = lane & 3;
  x.ta = token_at(un, min(16 * i + g, un.count - 1));
  x.tb = token_at(un, min(16 * i + g + 8, un.count - 1));
  const uint4* const pa = reinterpret_cast<const uint4*>(rr + x.ta * kDigits + 64 * tq);
  const uint4* const pb = reinterpret_cast<const uint4*>(rr + x.tb * kDigits + 64 * tq);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    x.a[j] = pa[j];
    x.b[j] = pb[j];
  }
#pragma unroll
  for (int nt = 0; nt < 2; ++nt) {
    x.qa[nt] = *reinterpret_cast<const uint16_t*>(rs + x.ta * kDigits + 8 * nt + 2 * tq);
    x.qb[nt] = *reinterpret_cast<const uint16_t*>(rs + x.tb * kDigits + 8 * nt + 2 * tq);
  }
}

__device__ __forceinline__ uint32_t word_of(const uint4 (&v)[4], int kk) {
  const uint4& q = v[kk >> 2];
  return (kk & 3) == 0 ? q.x : (kk & 3) == 1 ? q.y : (kk & 3) == 2 ? q.z : q.w;
}

//: PART 0 both halves; 1 the den alone; 2 the writes alone; 3 neither (the staging, the publish, the barriers)
template <int WRITE_K, int READ_K, int PART>
__global__ void __launch_bounds__(kThreads, 1) sweep(int windows, Pool g, float* den, long long* t,
                                                     int* out) {
  extern __shared__ __align__(128) unsigned char raw[];
  Shared<WRITE_K>& sh = *reinterpret_cast<Shared<WRITE_K>*>(raw);
  const int tid = static_cast<int>(threadIdx.x), lane = lab::lane_id(), warp = lab::warp_id();
  const int head = static_cast<int>(blockIdx.x) / (kDigits / kRows);
  const int s0 = (static_cast<int>(blockIdx.x) % (kDigits / kRows)) * kRows;
  const int g8 = lane >> 2, tq = lane & 3, r = tid;
  float* const den_h = den + static_cast<int64_t>(head) * kPool * kWindow;
  const uint8_t* const rr_h = g.rr + window_token(head, 0) * kDigits;
  const uint8_t* const rs_h = g.rs + window_token(head, 0) * kDigits + s0;
  const uint16_t* const off_h = g.offsets + static_cast<int64_t>(head) * kPool * kOffsets;
  float col[kRows] = {};
  stage_lists(sh, g, head, 0, s0, tid);
  stage_words(sh, g, head, 0, s0, tid);
  if (windows > 1) stage_words(sh, g, head, 1, s0, tid);
  lab::cp_commit();
  int off = off_h[r], cnt = off_h[r + 1] - off;
  lab::cp_wait<0>();
  __syncthreads();
  Union un{}, un_next{};
  Tile cur{}, nxt{}, ahead{};
  if constexpr (PART != 2 && PART != 3) {
    un = union_of(sh, 0, lane);
    if (warp < (un.count + 15) >> 4) load_tile(cur, un, rr_h, rs_h, warp, lane);
  }
  float sink = 0.0f;
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int win = 0; win < windows; ++win) {
    const int b = win & 1, pw = win % kPool;
    lab::cp_wait<0>();
    __syncthreads();
    //: the next window's lists, coefficients and gains, the words two ahead
    if (win + 1 < windows) stage_lists(sh, g, head, win + 1, s0, tid);
    if (win + 2 < windows) stage_words(sh, g, head, win + 2, s0, tid);
    lab::cp_commit();
    const int next_off = win + 1 < windows ? off_h[((win + 1) % kPool) * kOffsets + r] : 0;
    const int next_end = win + 1 < windows ? off_h[((win + 1) % kPool) * kOffsets + r + 1] : 0;
    //: the next window's tokens and first tile; its words landed with this window's barrier
    if (PART != 2 && PART != 3 && win + 1 < windows) {
      un_next = union_of(sh, (win + 1) % 3, lane);
      if (warp < (un_next.count + 15) >> 4)
        load_tile(ahead, un_next,
                  rr_h + static_cast<int64_t>((win + 1) % kPool) * kWindow * kDigits,
                  rs_h + static_cast<int64_t>((win + 1) % kPool) * kWindow * kDigits, warp, lane);
    }
    //: the mass as of the windows before, published at its logical columns
    {
      const int c = logical(r);
#pragma unroll
      for (int s = 0; s < kRows; ++s)
        *reinterpret_cast<__half*>(&sh.mass[s][2 * c]) = __float2half_rn(col[s]);
    }
    __syncthreads();
    const auto den_part = [&] {
      const int tiles = (un.count + 15) >> 4;
      const uint8_t* const rr = rr_h + static_cast<int64_t>(pw) * kWindow * kDigits;
      const uint8_t* const rs = rs_h + static_cast<int64_t>(pw) * kWindow * kDigits;
#pragma unroll 1
      for (int i = warp; i < tiles; i += kWarps) {
        if (i + kWarps < tiles) load_tile(nxt, un, rr, rs, i + kWarps, lane);
        float acc[2][4] = {};
#pragma unroll
        for (int kp = 0; kp < 8; ++kp) {
          //: B for k-steps 2 kp and 2 kp + 1: the 16 rows (two n-tiles) over 32 logical digits
          uint32_t bf[2][4];
#pragma unroll
          for (int nt = 0; nt < 2; ++nt)
            lab::ldsm(bf[nt],
                      lab::smem(&sh.mass[8 * nt + (lane & 7)][2 * (32 * kp + 8 * (lane >> 3))]));
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const uint32_t wa = word_of(cur.a, 2 * kp + h), wb = word_of(cur.b, 2 * kp + h);
            const uint32_t a[4] = {u8_pair(wa, 0x4140u), u8_pair(wb, 0x4140u), u8_pair(wa, 0x4342u),
                                   u8_pair(wb, 0x4342u)};
            lab::mma_f16_f32(acc[0], a, bf[0][2 * h], bf[0][2 * h + 1]);
            lab::mma_f16_f32(acc[1], a, bf[1][2 * h], bf[1][2 * h + 1]);
          }
        }
        float da = 0.0f, db = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          da += static_cast<float>(cur.qa[nt] & 0xffu) * acc[nt][0]
                + static_cast<float>(cur.qa[nt] >> 8) * acc[nt][1];
          db += static_cast<float>(cur.qb[nt] & 0xffu) * acc[nt][2]
                + static_cast<float>(cur.qb[nt] >> 8) * acc[nt][3];
        }
#pragma unroll
        for (int d = 1; d <= 2; d <<= 1) {
          da += __shfl_xor_sync(0xffffffffu, da, d);
          db += __shfl_xor_sync(0xffffffffu, db, d);
        }
        if (tq == 0 && 16 * i + g8 < un.count) red_f32(den_h + pw * kWindow + cur.ta, da);
        if (tq == 0 && 16 * i + g8 + 8 < un.count) red_f32(den_h + pw * kWindow + cur.tb, db);
        cur = nxt;
      }
    };
    //: the window's writes into the thread's digit: its list against the coefficients and gains, three entries in
    //: flight (one computing, the next's coefficients loading, the one after's token loading)
    const auto write_part = [&] {
      const uint16_t* const tok = sh.tokens[b] + off;
      const uint8_t* const val = sh.values[b] + off;
      const auto entry = [&](int i, int& tt, uint32_t& v) {
        tt = i < cnt ? tok[i] : 0;
        v = i < cnt ? val[i] : 0u;
      };
      const auto fetch = [&](int tt, uint32_t v, float& c, uint4(&w)[2]) {
        const bool sw = (tt & 4) != 0;
        c = sh.m[b][tt] * static_cast<float>(v);
        const uint4 x0 = *reinterpret_cast<const uint4*>(&sh.ws[b][tt][0]);
        const uint4 x1 = *reinterpret_cast<const uint4*>(&sh.ws[b][tt][8]);
        w[0] = sw ? x1 : x0;
        w[1] = sw ? x0 : x1;
      };
      int t1, t2;
      uint32_t v1, v2;
      entry(0, t1, v1);
      entry(1, t2, v2);
      float c;
      uint4 w[2];
      fetch(t1, v1, c, w);
#pragma unroll 1
      for (int i = 0; i < cnt; ++i) {
        int t3;
        uint32_t v3;
        entry(i + 2, t3, v3);
        float cn;
        uint4 wn[2];
        fetch(t2, v2, cn, wn);
        t2 = t3;
        v2 = v3;
        const uint32_t h[8] = {w[0].x, w[0].y, w[0].z, w[0].w, w[1].x, w[1].y, w[1].z, w[1].w};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h[j]));
          col[2 * j] += c * f.x;
          col[2 * j + 1] += c * f.y;
        }
        c = cn;
        w[0] = wn[0];
        w[1] = wn[1];
      }
    };
    if constexpr (PART == 1) den_part();
    if constexpr (PART == 2) write_part();
    if constexpr (PART == 0) {
      if (warp < kWarps / 2) {
        den_part();
        write_part();
      } else {
        write_part();
        den_part();
      }
    }
    un = un_next;
    cur = ahead;
    off = next_off;
    cnt = next_end - next_off;
  }
  clock.stop(t);
#pragma unroll
  for (int s = 0; s < kRows; ++s) sink += col[s];
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
  int heads = 0;
  std::vector<uint32_t> words;
  std::vector<uint16_t> offsets, tokens;
  std::vector<uint8_t> values;
  std::vector<__half> ws;
  std::vector<uint8_t> wr, rr, rs;
  std::vector<float> m;
};

Host make_host(int heads, int write_k, int read_k) {
  Host h;
  h.heads = heads;
  const size_t tokens = static_cast<size_t>(heads) * kPool * kWindow;
  h.words.assign(static_cast<size_t>(heads) * kPool * kDigits * kWords, 0u);
  h.offsets.assign(static_cast<size_t>(heads) * kPool * kOffsets, 0);
  h.tokens.assign(tokens * write_k, 0);
  h.values.assign(tokens * write_k, 0);
  h.ws.resize(tokens * kDigits);
  h.wr.assign(tokens * kDigits, 0);
  h.rr.resize(tokens * kDigits);
  h.rs.assign(tokens * kDigits, 0);
  h.m.resize(tokens);
  for (size_t t = 0; t < tokens; ++t) {
    const size_t win = t / kWindow;
    const int tt = static_cast<int>(t % kWindow);
    for (int d : live_digits(1, static_cast<uint32_t>(t), write_k))
      h.wr[t * kDigits + d] = static_cast<uint8_t>(1 + hash3(2, t, d) % 255);
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
  //: level R's write lists, digit-major, tokens in order within a digit
  for (size_t win = 0; win < static_cast<size_t>(heads) * kPool; ++win) {
    size_t at = 0;
    for (int d = 0; d < kDigits; ++d) {
      h.offsets[win * kOffsets + d] = static_cast<uint16_t>(at);
      for (int tt = 0; tt < kWindow; ++tt) {
        const uint8_t v = h.wr[(win * kWindow + tt) * kDigits + d];
        if (!v) continue;
        h.tokens[win * kWindow * write_k + at] = static_cast<uint16_t>(tt);
        h.values[win * kWindow * write_k + at++] = v;
      }
    }
    h.offsets[win * kOffsets + kDigits] = static_cast<uint16_t>(at);
  }
  return h;
}

//: head 0's den over the pool's windows in order, in double
std::vector<double> reference(const Host& h) {
  std::vector<double> mass(static_cast<size_t>(kDigits) * kDigits, 0.0), den(kPool * kWindow, 0.0);
  for (int w = 0; w < kPool; ++w) {
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
    pool.ws = up(h.ws);
    pool.rr = up(h.rr);
    pool.rs = up(h.rs);
    pool.m = up(h.m);
  }

  ~Device() {
    for (void* p : owned) cudaFree(p);
  }
};

constexpr int kWindows = 128;

template <int WRITE_K, int READ_K, int PART = 0>
void variant(const lab::Bench& b, float* den) {
  const std::string name = lab::kernel("sweep", {WRITE_K, READ_K, PART});
  if (PART <= 1) b.expect(name, "HMMA.16816.F32", ">=", 32);
  if (PART <= 1) b.expect(name, "LDSM", ">=", 2);
  if (PART <= 1) b.expect(name, "RED", ">=", 2);
  if (PART == 0 || PART == 2) b.expect(name, "FFMA", ">=", 16);
  if (PART == 3) b.expect(name, "BAR", ">=", 2);
  if (b.expect_only) return;
  const int heads = b.sms / (kDigits / kRows), grid = heads * (kDigits / kRows);
  const Host h = make_host(heads, WRITE_K, READ_K);
  const Device dv(h);
  const int bytes = static_cast<int>(sizeof(Shared<WRITE_K>));
  LAB_CHECK(cudaFuncSetAttribute(sweep<WRITE_K, READ_K, PART>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  const size_t den_bytes = sizeof(float) * heads * kPool * kWindow;
  if (PART == 0) {
    //: head 0 over the pool's windows once, against the double reference: every den within 1%
    LAB_CHECK(cudaMemset(den, 0, den_bytes));
    sweep<WRITE_K, READ_K, PART><<<grid, kThreads, bytes>>>(kPool, dv.pool, den, b.times, b.sinks);
    LAB_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(kPool * kWindow);
    LAB_CHECK(cudaMemcpy(got.data(), den, sizeof(float) * got.size(), cudaMemcpyDeviceToHost));
    const std::vector<double> want = reference(h);
    double worst = 0;
    for (size_t i = 0; i < got.size(); ++i)
      if (want[i] > 0) worst = std::max(worst, std::abs(got[i] - want[i]) / want[i]);
    if (worst > 1e-2) {
      std::fprintf(stderr, "den_sweep: den off the reference by %.3g at write k %d, read k %d\n",
                   worst, WRITE_K, READ_K);
      std::exit(1);
    }
  }
  const lab::Samples s = b.variant(
      [&] {
        sweep<WRITE_K, READ_K, PART>
            <<<grid, kThreads, bytes>>>(kWindows, dv.pool, den, b.times, b.sinks);
      },
      [&] { return b.span(kWarps) / kWindows; });
  const char* part[4] = {"", " / den alone", " / writes alone", " / neither"};
  const std::string label = std::to_string(WRITE_K) + " live level-R write digits, "
                            + std::to_string(READ_K) + " live level-S read digits a token"
                            + part[PART];
  b.record(label, "cycles a window", s);
}

}  // namespace den_sweep

int main(int argc, char** argv) {
  using namespace den_sweep;
  lab::Bench b(argc, argv);
  float* den = nullptr;
  if (!b.expect_only) LAB_CHECK(cudaMalloc(&den, sizeof(float) * b.sms * kPool * kWindow));
  variant<4, 4>(b, den);
  variant<4, 4, 1>(b, den);
  variant<4, 4, 2>(b, den);
  variant<4, 4, 3>(b, den);
  variant<16, 16>(b, den);
  variant<4, 256>(b, den);
}
