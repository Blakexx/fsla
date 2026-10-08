// measure/lab/experiments/gang_ring.cu -- QUESTION: in one persistent cooperative grid, can sidecar CTAs build each
// window's per-list slot lists from the bitlists into a ring of R windows fast enough for the carry CTAs that consume
// them, and what do the ring's waits cost the consumers against the same consumers given every list up front?
// METHOD: a CTA an SM, every CTA resident (`cudaLaunchCooperativeKernel`); the first `side` CTAs are sidecars, the rest
// carry CTAs, each owning a group of 32 level-R digits (CTA c the group c % 8). A synthetic head of 64 windows, each
// token live in k of 256 digits. A sidecar takes (window, group) items in window order and builds the group's 32 lists
// from the window's bitlists -- the union's ranks, each digit's tokens as their ranks' pool rows ordered round-robin by
// rank & 7, padded to steps of 16, each entry its pool row's byte offset and its f16 coefficient pair (the level-R byte
// times the token's (F, m)) -- into the ring slot, after every carry warp has staged window w - R. A carry warp stages
// its 4 lists a window ahead by cp.async, frees the slot, and steps them in list_fold form 5's shape (10 MMAs a step,
// runtime list lengths, a two-slot operand ring, the remainder step peeled) over a static pool. The carry CTAs' span a
// window against the busiest scheduler's MMAs at 32 cycles; the sidecars' alone. see docs/internals/tools/lab.md#gang-ring
#include "lab.cuh"

#include <cuda_fp16.h>

#include <random>
#include <utility>
#include <vector>

namespace gang_ring {

template <class F, int... I>
__device__ __forceinline__ void static_for_impl(F&& f, std::integer_sequence<int, I...>) {
  (f(std::integral_constant<int, I>{}), ...);
}

template <int N, class F>
__device__ __forceinline__ void static_for(F&& f) {
  static_for_impl(f, std::make_integer_sequence<int, N>{});
}

constexpr int kThreads = 256, kWarps = 8, kWindow = 512, kWords = 16, kDigits = 256, kGroup = 32;
constexpr int kGroups = kDigits / kGroup, kRows = 256, kPitch = 160, kWindows = 64;
//: a list's entries a window, padded to 16, at most; a warp's four
constexpr int kListCap = 80, kGroupCap = kGroup * kListCap, kWarpCap = 4 * kListCap;

//: A (window, group)'S LISTS: each list's start (entries, a multiple of 16; `at[32]` the end), and per entry its pool
//: row's byte offset (the swizzle folded in) and its coefficient pair
struct alignas(32) Lists {
  uint16_t at[48];
  uint16_t off[kGroupCap];
  uint16_t cv[kGroupCap];
  uint16_t cm[kGroupCap];
};

struct alignas(16) CarryShared {
  uint8_t rows[kRows * kPitch];
  float mass[kWarps][2][4][32];
  uint16_t buf[kWarps][2][3][kWarpCap];
};

struct alignas(16) SideShared {
  uint8_t bytes[2][kWindow][kGroup];
  float2 fm[2][kWindow];
  uint16_t lists[3][kGroupCap];
  uint32_t pcount[kWarps];
  uint32_t count[kWarps][kGroup][2];
};

constexpr size_t kShared = sizeof(CarryShared) > sizeof(SideShared) ? sizeof(CarryShared)
                                                                    : sizeof(SideShared);

__device__ __forceinline__ int load_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void wait_at_least(const int* p, int target) {
  while (load_acquire(p) < target) {
  }
}

__device__ __forceinline__ uint16_t load_cg(const uint16_t* p) {
  uint16_t v;
  asm volatile("ld.global.cg.u16 %0, [%1];" : "=h"(v) : "l"(p));
  return v;
}

__device__ __forceinline__ uint32_t row_at(uint32_t row) {
  return row * kPitch + 16u * ((row >> 2) & 1u);
}

struct Op {
  uint32_t a[2][4], b[4][2], c[4];
};

struct Acc {
  float acc[4][2][4][4];
};

//: a step's operands from its 16 entries in the warp's buffer (form 5's load: a lane's row an XOR)
__device__ __forceinline__ void load_step(Op& o, uint32_t pool, const uint16_t* off,
                                          const uint16_t* cv, const uint16_t* cm, int lane) {
  const int r = lane & 7, q = lane >> 3, tq = lane & 3;
  const uint32_t ea = pool + (off[8 * (q >> 1) + r] ^ (16u * static_cast<uint32_t>(q & 1)));
  const uint32_t eb = pool + 64u + (off[8 * (q & 1) + r] ^ (16u * static_cast<uint32_t>(q >> 1)));
  lab::ldsm_t(o.a[0], ea);
  lab::ldsm_t(o.a[1], ea + 32u);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    uint32_t x[4];
    lab::ldsm_t(x, eb + 32u * h);
    o.b[2 * h][0] = x[0];
    o.b[2 * h][1] = x[1];
    o.b[2 * h + 1][0] = x[2];
    o.b[2 * h + 1][1] = x[3];
  }
  o.c[0] = *reinterpret_cast<const uint32_t*>(cv + 2 * tq);
  o.c[1] = *reinterpret_cast<const uint32_t*>(cm + 2 * tq);
  o.c[2] = *reinterpret_cast<const uint32_t*>(cv + 2 * tq + 8);
  o.c[3] = *reinterpret_cast<const uint32_t*>(cm + 2 * tq + 8);
}

template <int J>
__device__ __forceinline__ void step(Acc& st, float (&m)[2][4], const Op& o) {
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    lab::mma_f16_f32(m[mt], o.a[mt], o.c[1], o.c[3]);
    const uint32_t aj[4] = {lab::hmul2(o.a[mt][0], o.c[0]), lab::hmul2(o.a[mt][1], o.c[0]),
                            lab::hmul2(o.a[mt][2], o.c[2]), lab::hmul2(o.a[mt][3], o.c[2])};
#pragma unroll
    for (int n = 0; n < 4; ++n) lab::mma_f16_f32(st.acc[J][mt][n], aj, o.b[n][0], o.b[n][1]);
  }
}

//: A LIST'S STEPS: `n` steps, two operand slots, the next step's loads before this step's MMAs, the odd step peeled
template <int J>
__device__ __forceinline__ void list_steps(Acc& st, CarryShared& sh, const uint16_t* off,
                                           const uint16_t* cv, const uint16_t* cm, int n, int w,
                                           int lane) {
  const uint32_t pool = lab::smem(sh.rows);
  float m[2][4] = {};
  Op o0, o1;
  load_step(o0, pool, off, cv, cm, lane);
  int s = 0;
#pragma unroll 1
  for (; s + 1 < n; s += 2) {
    load_step(o1, pool, off + 16 * (s + 1), cv + 16 * (s + 1), cm + 16 * (s + 1), lane);
    step<J>(st, m, o0);
    const int next = min(s + 2, n - 1);
    load_step(o0, pool, off + 16 * next, cv + 16 * next, cm + 16 * next, lane);
    step<J>(st, m, o1);
  }
  if (s < n) step<J>(st, m, o0);
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int i = 0; i < 4; ++i) sh.mass[w][mt][i][lane] += m[mt][i];
}

//: A WARP'S WINDOW STAGED into buffer `b`: its 4 lists' entries by cp.async from the window's lists; the lists' starts
//: within the buffer returned. Mode 0 first waits for the window's lists.
template <int MODE>
__device__ __forceinline__ void stage(int (&starts)[5], CarryShared& sh, const Lists* lists,
                                      const int* ready, int ready_at, int b, int w, int lane) {
  if constexpr (MODE == 0) {
    if (lane == 0) wait_at_least(ready + ready_at, 1);
    __syncwarp();
  }
#pragma unroll
  for (int j = 0; j < 5; ++j) starts[j] = load_cg(lists->at + 4 * w + j);
  const int lo = starts[0], chunks = (starts[4] - lo) / 8;
  const uint16_t* src[3] = {lists->off + lo, lists->cv + lo, lists->cm + lo};
#pragma unroll
  for (int a = 0; a < 3; ++a)
    for (int k = lane; k < chunks; k += 32)
      lab::cp_async16(lab::smem(&sh.buf[w][b][a][8 * k]), src[a] + 8 * k);
  lab::cp_commit();
#pragma unroll
  for (int j = 4; j >= 0; --j) starts[j] -= lo;
}

template <int MODE>
__device__ __forceinline__ void release_window(int* done, int win, int lane) {
  __syncwarp();
  if (lane == 0) {
    __threadfence();
    atomicAdd(done + win, 1);
  }
}

template <int MODE>
__device__ __forceinline__ void compute(Acc& st, CarryShared& sh, const int (&starts)[5], int b,
                                        int w, int lane) {
  static_for<4>([&](auto J) {
    constexpr int j = decltype(J)::value;
    const uint16_t* base = &sh.buf[w][b][0][starts[j]];
    list_steps<j>(st, sh, base, base + (&sh.buf[w][b][1][0] - &sh.buf[w][b][0][0]),
                  base + (&sh.buf[w][b][2][0] - &sh.buf[w][b][0][0]),
                  (starts[j + 1] - starts[j]) / 16, w, lane);
  });
}

template <int MODE>
__device__ void carry_role(CarryShared& sh, int windows, int side, int depth, const Lists* ring,
                           const Lists* all, const int* ready, int* done, long long* t, int* out) {
  const int lane = lab::lane_id(), w = lab::warp_id(), tid = static_cast<int>(threadIdx.x);
  const int g = (static_cast<int>(blockIdx.x) - side) % kGroups;
#pragma unroll 1
  for (int i = tid; i < kRows * kPitch / 2; i += kThreads)
    reinterpret_cast<uint16_t*>(sh.rows)[i] = static_cast<uint16_t>(0x2c00 + ((i * 7) & 255));
#pragma unroll 1
  for (int i = tid; i < kWarps * 2 * 4 * 32; i += kThreads) (&sh.mass[0][0][0][0])[i] = 0.0f;
  __syncthreads();
  const auto lists_of = [&](int win) {
    return MODE == 1 ? all + static_cast<size_t>(win) * kGroups + g
                     : ring + static_cast<size_t>(win % depth) * kGroups + g;
  };
  Acc st = {};
  lab::WarpSpan clock;
  clock.start();
  int s0[5], s1[5];
  stage<MODE>(s0, sh, lists_of(0), ready, g, 0, w, lane);
#pragma unroll 1
  for (int win = 0; win < windows; win += 2) {
    stage<MODE>(s1, sh, lists_of(win + 1), ready, (win + 1) * kGroups + g, 1, w, lane);
    lab::cp_wait<1>();
    release_window<MODE>(done, win, lane);
    compute<MODE>(st, sh, s0, 0, w, lane);
    const int ahead = min(win + 2, windows - 1);
    stage<MODE>(s0, sh, lists_of(ahead), ready, ahead * kGroups + g, 0, w, lane);
    lab::cp_wait<1>();
    release_window<MODE>(done, win + 1, lane);
    compute<MODE>(st, sh, s1, 1, w, lane);
  }
  lab::cp_wait_all();
  clock.stop(t);
  uint32_t fold = lab::bits(sh.mass[w][0][0][lane]);
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int n = 0; n < 4; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) fold += lab::bits(st.acc[j][mt][n][i]);
  lab::sink(out, fold);
}

//: AN ITEM'S INPUTS, one item ahead: the lane's digit's two words of the warp's pair (registers), the group's level-R
//: bytes and the window's (F, m) into stage `buf` (cp.async)
__device__ __forceinline__ uint2 side_load(SideShared& sh, int buf, int item, const uint32_t* words,
                                           const uint8_t* bytes, const float2* fm) {
  const int win = item / kGroups, g = item % kGroups, tid = static_cast<int>(threadIdx.x);
  for (int i = tid; i < 2 * kWindow; i += kThreads) {
    const int token = i >> 1, half = i & 1;
    lab::cp_async16(
        lab::smem(&sh.bytes[buf][token][16 * half]),
        bytes + (static_cast<size_t>(win) * kWindow + token) * kDigits + kGroup * g + 16 * half);
  }
  for (int i = tid; i < kWindow / 2; i += kThreads)
    lab::cp_async16(lab::smem(&sh.fm[buf][2 * i]), fm + static_cast<size_t>(win) * kWindow + 2 * i);
  lab::cp_commit();
  return *reinterpret_cast<const uint2*>(
      words + (static_cast<size_t>(win) * kDigits + kGroup * g + lab::lane_id()) * kWords
      + 2 * lab::warp_id());
}

//: per residue (packed 8 bits a residue, residues 0-3 in `lo`, 4-7 in `hi`), the entries of one word in slot order
__device__ __forceinline__ void count_word(uint32_t& lo, uint32_t& hi, uint32_t bits, uint32_t u,
                                           uint32_t base) {
  while (bits) {
    const int t = __ffs(bits) - 1;
    const uint32_t r = (base + __popc(u & ((1u << t) - 1u))) & 7u, inc = 1u << (8u * (r & 3u));
    lo += r < 4 ? inc : 0u;
    hi += r < 4 ? 0u : inc;
    bits &= bits - 1u;
  }
}

__device__ __forceinline__ uint32_t packed(uint32_t lo, uint32_t hi, uint32_t r) {
  return ((r < 4 ? lo : hi) >> (8u * (r & 3u))) & 255u;
}

//: A SIDECAR'S ITEM: one (window, group)'s 32 lists from its bitlists. Thread (warp w, lane d) holds digit d's words
//: 2w, 2w + 1: the group's union a warp-wide OR, a word's base the warps' popcounts before it, a digit's per-residue
//: counts the warps' packed counts, each entry's place round-robin by slot & 7; the lists assembled in shared memory
//: and copied out whole runs, after the slot frees (mode 0)
template <int MODE>
__device__ void side_build(SideShared& sh, int buf, uint2 wv, Lists* dst, int* ready, int item,
                           const int* done, int done_at, int target) {
  const int w = lab::warp_id(), d = lab::lane_id(), tid = static_cast<int>(threadIdx.x);
  const uint32_t u0 = __reduce_or_sync(0xffffffffu, wv.x), u1 = __reduce_or_sync(0xffffffffu, wv.y);
  if (d == 0) sh.pcount[w] = __popc(u0) + __popc(u1);
  __syncthreads();
  uint32_t base = 0;
#pragma unroll
  for (int ww = 0; ww < kWarps; ++ww) base += ww < w ? sh.pcount[ww] : 0u;
  const uint32_t b0 = base, b1 = base + __popc(u0);
  uint32_t lo = 0, hi = 0;
  count_word(lo, hi, wv.x, u0, b0);
  count_word(lo, hi, wv.y, u1, b1);
  sh.count[w][d][0] = lo;
  sh.count[w][d][1] = hi;
  __syncthreads();
  int size[8] = {}, before[8] = {};
#pragma unroll
  for (int ww = 0; ww < kWarps; ++ww) {
    const uint32_t l = sh.count[ww][d][0], h = sh.count[ww][d][1];
#pragma unroll
    for (int rr = 0; rr < 8; ++rr) {
      const int c = static_cast<int>(packed(l, h, static_cast<uint32_t>(rr)));
      size[rr] += c;
      before[rr] += ww < w ? c : 0;
    }
  }
  int n = 0;
#pragma unroll
  for (int rr = 0; rr < 8; ++rr) n += size[rr];
  const int padded = (n + 15) & ~15;
  int start = padded;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, start, o);
    start += d >= o ? v : 0;
  }
  const int total = __shfl_sync(0xffffffffu, start, 31);
  start -= padded;
  uint32_t slo = 0, shi = 0;
  const auto place = [&](uint32_t bits, uint32_t u, uint32_t b, int word) {
    while (bits) {
      const int t = __ffs(bits) - 1;
      const uint32_t slot = b + __popc(u & ((1u << t) - 1u)), r = slot & 7u;
      int j = static_cast<int>(packed(slo, shi, r)), pos = 0;
#pragma unroll
      for (int rr = 0; rr < 8; ++rr) j += rr == static_cast<int>(r) ? before[rr] : 0;
#pragma unroll
      for (int rr = 0; rr < 8; ++rr)
        pos += min(size[rr], j) + (rr < static_cast<int>(r) && size[rr] > j);
      const int e = start + pos, token = 32 * word + t;
      const float c = static_cast<float>(sh.bytes[buf][token][d]) * (1.0f / 256.0f);
      sh.lists[0][e] = static_cast<uint16_t>(row_at(slot & (kRows - 1)));
      sh.lists[1][e] = __half_as_ushort(__float2half(c * sh.fm[buf][token].x));
      sh.lists[2][e] = __half_as_ushort(__float2half(c * sh.fm[buf][token].y));
      const uint32_t inc = 1u << (8u * (r & 3u));
      slo += r < 4 ? inc : 0u;
      shi += r < 4 ? 0u : inc;
      bits &= bits - 1u;
    }
  };
  place(wv.x, u0, b0, 2 * w);
  place(wv.y, u1, b1, 2 * w + 1);
  if (w == 0)
    for (int p = n; p < padded; ++p)
      sh.lists[0][start + p] = sh.lists[1][start + p] = sh.lists[2][start + p] = 0;
  __syncthreads();
  if constexpr (MODE == 0) {
    if (tid == 0 && done_at >= 0) wait_at_least(done + done_at, target);
    __syncthreads();
  }
  if (w == 0) dst->at[d] = static_cast<uint16_t>(start);
  if (tid == 0) dst->at[kGroup] = static_cast<uint16_t>(total);
  const int chunks = total / 8;
  for (int i = tid; i < 3 * chunks; i += kThreads) {
    const int a = i / chunks, k = i - a * chunks;
    uint16_t* out = a == 0 ? dst->off : a == 1 ? dst->cv : dst->cm;
    *reinterpret_cast<uint4*>(out + 8 * k) = *reinterpret_cast<const uint4*>(&sh.lists[a][8 * k]);
  }
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    atomicExch(ready + item, 1);
  }
}

//: MODE 0 the ring (sidecars and carry CTAs together); 1 the carry CTAs on lists built beforehand for every window (no
//: waits); 2 the sidecars alone, building every window's lists
template <int MODE>
__global__ void __launch_bounds__(kThreads, 1) gang(int windows, int side, int depth,
                                                    const uint32_t* words, const uint8_t* bytes,
                                                    const float2* fm, Lists* ring, Lists* all,
                                                    int* ready, int* done, long long* t, int* out) {
  extern __shared__ __align__(128) unsigned char raw[];
  const int carries = static_cast<int>(gridDim.x) - side, items = windows * kGroups;
  if (static_cast<int>(blockIdx.x) >= side) {
    if constexpr (MODE != 2)
      carry_role<MODE>(*reinterpret_cast<CarryShared*>(raw), windows, side, depth, ring, all, ready,
                       done, t, out);
    return;
  }
  if constexpr (MODE == 1) return;
  SideShared& sh = *reinterpret_cast<SideShared*>(raw);
  lab::WarpSpan clock;
  clock.start();
  int item = static_cast<int>(blockIdx.x), buf = 0;
  uint2 wv = side_load(sh, 0, item, words, bytes, fm);
#pragma unroll 1
  for (; item < items; item += side, buf ^= 1) {
    lab::cp_wait_all();
    __syncthreads();
    const int next = item + side;
    const uint2 wn =
        next < items ? side_load(sh, buf ^ 1, next, words, bytes, fm) : make_uint2(0u, 0u);
    const int win = item / kGroups, g = item % kGroups;
    side_build<MODE>(sh, buf, wv,
                     MODE == 0 ? ring + static_cast<size_t>(win % depth) * kGroups + g : all + item,
                     ready, item, done, win - depth, carries * kWarps);
    wv = wn;
  }
  clock.stop(t);
}

struct Inputs {
  std::vector<uint32_t> words;
  std::vector<uint8_t> bytes;
  std::vector<float2> fm;
  std::vector<int> count;  // [window][digit]
};

//: every token live in k distinct digits of the 256, uniformly; a live byte in 1..255; (F, m) in [0.5, 2)
Inputs make_inputs(int k) {
  Inputs in;
  in.words.assign(static_cast<size_t>(kWindows) * kDigits * kWords, 0u);
  in.bytes.assign(static_cast<size_t>(kWindows) * kWindow * kDigits, 0);
  in.fm.resize(static_cast<size_t>(kWindows) * kWindow);
  in.count.assign(static_cast<size_t>(kWindows) * kDigits, 0);
  std::mt19937 rng(1234u + static_cast<uint32_t>(k));
  std::uniform_int_distribution<int> digit(0, kDigits - 1), byte(1, 255);
  std::uniform_real_distribution<float> gain(0.5f, 2.0f);
  for (int win = 0; win < kWindows; ++win)
    for (int tt = 0; tt < kWindow; ++tt) {
      const size_t token = static_cast<size_t>(win) * kWindow + tt;
      in.fm[token] = make_float2(gain(rng), gain(rng));
      for (int got = 0; got < k;) {
        const int d = digit(rng);
        uint8_t& b = in.bytes[token * kDigits + d];
        if (b) continue;
        b = static_cast<uint8_t>(byte(rng));
        in.words[(static_cast<size_t>(win) * kDigits + d) * kWords + tt / 32] |= 1u << (tt % 32);
        ++in.count[static_cast<size_t>(win) * kDigits + d];
        ++got;
      }
    }
  return in;
}

//: the busiest scheduler's paired warps' MMAs a window at 32 cycles, averaged over the windows
double floor_of(const Inputs& in) {
  double total = 0.0;
  for (int win = 0; win < kWindows; ++win) {
    long long worst = 0;
    for (int g = 0; g < kGroups; ++g)
      for (int sch = 0; sch < 4; ++sch) {
        long long mmas = 0;
        for (int w : {sch, sch + 4})
          for (int j = 0; j < 4; ++j)
            mmas += 10LL
                    * ((in.count[static_cast<size_t>(win) * kDigits + kGroup * g + 4 * w + j] + 15)
                       / 16);
        worst = std::max(worst, mmas);
      }
    total += 32.0 * static_cast<double>(worst);
  }
  return total / kWindows;
}

struct Device {
  uint32_t* words = nullptr;
  uint8_t* bytes = nullptr;
  float2* fm = nullptr;
  Lists *ring = nullptr, *all = nullptr;
  int *ready = nullptr, *done = nullptr;
};

template <int MODE>
void launch(const lab::Bench& b, const Device& dv, int side, int depth) {
  LAB_CHECK(cudaMemset(dv.ready, 0, sizeof(int) * kWindows * kGroups));
  LAB_CHECK(cudaMemset(dv.done, 0, sizeof(int) * kWindows));
  int windows = kWindows;
  void* args[] = {&windows,         &side,           &depth,          (void*)&dv.words,
                  (void*)&dv.bytes, (void*)&dv.fm,   (void*)&dv.ring, (void*)&dv.all,
                  (void*)&dv.ready, (void*)&dv.done, (void*)&b.times, (void*)&b.sinks};
  LAB_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(gang<MODE>), dim3(b.sms),
                                        dim3(kThreads), args, kShared, nullptr));
}

//: the slowest block's span a window among blocks [first, last): its latest warp's end less its earliest warp's start,
//: on its own SM's clock (SMs' clocks do not compare); a cooperative grid's blocks start together
double span_of(const lab::Bench& b, int first, int last) {
  std::vector<long long> h(static_cast<size_t>(b.sms) * kWarps * 2);
  LAB_CHECK(cudaMemcpy(h.data(), b.times, h.size() * sizeof(long long), cudaMemcpyDeviceToHost));
  long long worst = 0;
  for (int blk = first; blk < last; ++blk) {
    long long lo = h[2 * blk * kWarps], hi = h[2 * blk * kWarps + 1];
    for (int w = 1; w < kWarps; ++w) {
      lo = std::min(lo, h[2 * (blk * kWarps + w)]);
      hi = std::max(hi, h[2 * (blk * kWarps + w) + 1]);
    }
    worst = std::max(worst, hi - lo);
  }
  return static_cast<double>(worst) / kWindows;
}

template <int MODE>
void variant(const lab::Bench& b, const Device& dv, int k, int side, int depth, double floor) {
  const std::string name = lab::kernel("gang", {MODE});
  if (MODE != 2) b.expect(name, "HMMA.16816.F32", ">=", 10);
  if (MODE != 2) b.expect(name, "LDSM.16.MT88.4", ">=", 4);
  if (b.expect_only) return;
  const bool carry = MODE != 2;
  const lab::Samples s =
      b.variant([&] { launch<MODE>(b, dv, side, depth); },
                [&] { return carry ? span_of(b, side, b.sms) : span_of(b, 0, side); });
  const std::string label =
      "k " + std::to_string(k) + " / "
      + (MODE == 0 ? "ring of " + std::to_string(depth) + ", " + std::to_string(side) + " sidecars"
         : MODE == 1 ? "lists given, " + std::to_string(b.sms - side) + " carry CTAs"
                     : "build alone, " + std::to_string(side) + " sidecars");
  b.record(label, "cycles a window", s);
  if (carry) {
    lab::Samples over;
    for (double c : s.v) over.v.push_back(c / floor);
    b.record(label + " / floor", "x", over);
  }
}

}  // namespace gang_ring

int main(int argc, char** argv) {
  using namespace gang_ring;
  lab::Bench b(argc, argv);
  if (b.expect_only) {
    variant<0>(b, Device{}, 4, 4, 3, 1.0);
    variant<1>(b, Device{}, 4, 4, 3, 1.0);
    variant<2>(b, Device{}, 4, 4, 3, 1.0);
    return 0;
  }
  int coop = 0;
  LAB_CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0));
  if (!coop) {
    std::fprintf(stderr, "gang_ring: the device has no cooperative launch\n");
    return 2;
  }
  LAB_CHECK(cudaFuncSetAttribute(gang<0>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(kShared)));
  LAB_CHECK(cudaFuncSetAttribute(gang<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(kShared)));
  LAB_CHECK(cudaFuncSetAttribute(gang<2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(kShared)));
  int resident = 0;
  LAB_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, gang<0>, kThreads, kShared));
  if (resident < 1) {
    std::fprintf(stderr, "gang_ring: a CTA does not fit an SM\n");
    return 2;
  }
  Device dv;
  LAB_CHECK(cudaMalloc(&dv.words, sizeof(uint32_t) * kWindows * kDigits * kWords));
  LAB_CHECK(cudaMalloc(&dv.bytes, sizeof(uint8_t) * kWindows * kWindow * kDigits));
  LAB_CHECK(cudaMalloc(&dv.fm, sizeof(float2) * kWindows * kWindow));
  LAB_CHECK(cudaMalloc(&dv.ring, sizeof(Lists) * 4 * kGroups));
  LAB_CHECK(cudaMalloc(&dv.all, sizeof(Lists) * kWindows * kGroups));
  LAB_CHECK(cudaMalloc(&dv.ready, sizeof(int) * kWindows * kGroups));
  LAB_CHECK(cudaMalloc(&dv.done, sizeof(int) * kWindows));
  for (int k : {4, 16}) {
    const Inputs in = make_inputs(k);
    for (int c : in.count)
      if (c > kListCap) {
        std::fprintf(stderr, "gang_ring: a list of %d entries exceeds the cap %d\n", c, kListCap);
        return 2;
      }
    LAB_CHECK(cudaMemcpy(dv.words, in.words.data(), sizeof(uint32_t) * in.words.size(),
                         cudaMemcpyHostToDevice));
    LAB_CHECK(cudaMemcpy(dv.bytes, in.bytes.data(), in.bytes.size(), cudaMemcpyHostToDevice));
    LAB_CHECK(
        cudaMemcpy(dv.fm, in.fm.data(), sizeof(float2) * in.fm.size(), cudaMemcpyHostToDevice));
    const double floor = floor_of(in);
    for (int side : {1, 2, 4, 8}) variant<2>(b, dv, k, side, 3, floor);
    launch<2>(b, dv, 8, 3);
    LAB_CHECK(cudaDeviceSynchronize());
    variant<1>(b, dv, k, 4, 3, floor);
    for (int side : {2, 4, 8}) variant<0>(b, dv, k, side, 3, floor);
    for (int depth : {2, 4}) variant<0>(b, dv, k, 4, depth, floor);
  }
}
