// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// measure/lab/experiments/facts_color.cu -- THE COLORER WITHOUT THE WINDOW (option B, ruled 2026-10-13): a warp a list
// unit colors its union and writes its lists in the carry's final form from what the stream half hands it -- the
// unit's bit tile and its stash of live entries (`facts_color.inputs.py`) -- in small CTAs that hold no window and no
// CTA barrier.
// QUESTION: on the `facts_block` cell's 8,192 list units, does the colorer fit beside the stream half within the line
// (1.15x the real floor, 516 / 665 us at k 4 / 16), and what does a union cost by phase?
// see docs/internals/tools/lab.md#facts-color
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../lab.cuh"
#include "carry/carry.cuh"

namespace facts_color {

namespace carry = rola::carry;

//: THE CTA: four warps, a union each, four CTAs an SM
constexpr int kWarps = 4, kThreads = 32 * kWarps, kCtasMin = 4;
constexpr int kWords = 16, kTokens = 32 * kWords, kDigits = 32, kColors = 8, kUnionCap = kTokens / kColors;
constexpr int kQueueCap = kTokens, kListCap = kTokens, kListAt = 8, kHeadEntries = 16, kMaxPasses = 4;
constexpr uint32_t kAll = 0xffffffffu;
//: THE READ POOL'S ROW, the sm_86 even arm's 64 bytes
constexpr int kRow = 64;

//: A WARP'S SHARED MEMORY, by offset: each token's digits (then each (list, pass)'s color counts, then each (list, pass,
//: color)'s place base), the claim queue (then the union by slot, then each list's starts and color counts so far), each digit's biased
//: color counts and a zero record (then its singles' color ends), the union's color counts and words, each token's
//: color (then its slot)
struct Smem {
  static constexpr int kMasks = 0;
  static constexpr int kTab = 0;
  static constexpr int kBase = 0;
  static constexpr int kQueue = kMasks + kTokens * 4;
  static constexpr int kStart = kQueue;
  static constexpr int kRun = kStart + kDigits * kListAt * 2;
  static constexpr int kBySlot = kQueue;
  static constexpr int kCounts = kQueue + kQueueCap * 2;
  static constexpr int kEnds = kCounts;
  static constexpr int kUcount = kCounts + (kDigits + 1) * 16;
  static constexpr int kUword = kUcount + kColors * 4;
  static constexpr int kMword = kUword + kWords * 4;
  static constexpr int kTword = kMword + kWords * 4;
  static constexpr int kColor = kTword + kWords * 4;
  static constexpr int kBytes = kColor + kTokens * 2;
};
static_assert(Smem::kBase + kDigits * kMaxPasses * kColors * 2 <= Smem::kQueue &&
                  Smem::kRun + kDigits * kColors <= Smem::kCounts && Smem::kUcount % 16 == 0 && Smem::kColor % 16 == 0,
              "the lists' tables inside the dead masks and queue; each tenant on its access's alignment");

//: A DIGIT'S COLOR COUNT, BIASED: `64 - cap + count` a u16 field, so bit 6 is set where the digit is full and the
//: union's sums order as its counts do; the zero record past the digits for a lane with none left
constexpr uint32_t kBias = kUnionCap, kFullBit = 0x40u, kFullPair = 0x00400040u;
constexpr int kZero = kDigits;

//: THE OPERANDS: the cell's tiles and stash, one side's units, its pass and pool row; the lists out
struct Params {
  const uint32_t* tiles;
  const uint32_t* offsets;
  const uint16_t* tokens;
  const uint32_t* coefs;
  int unit0, units, pass_log, row;
  uint2* entries;
  uint16_t* at;
  uint2* heads;
  uint16_t* union_tok;
  int* union_count;
  long long* cycles;
  long long* phases;
};

//: THE PHASE CLOCK: a warp's cycles by phase (`kTimed` of them), lane 0's `clock64` at each boundary
constexpr int kTimed = 8;
struct PhaseClock {
  long long last, at[kTimed];
  __device__ __forceinline__ explicit PhaseClock(long long t) : last(t) {}
  __device__ __forceinline__ void mark(int ph) {
    const long long t = clock64();
    at[ph] = t - last;
    last = t;
  }
};

//: AN OCTET OF BYTES (0-3 in `lo`, 4-7 in `hi`): byte `c`, and `v` added there
__device__ __forceinline__ uint32_t byte_of(uint32_t lo, uint32_t hi, int c) {
  return ((c < 4 ? lo : hi) >> (8 * (c & 3))) & 0xffu;
}

__device__ __forceinline__ void add_byte(uint32_t& lo, uint32_t& hi, int c, uint32_t v) {
  const uint32_t x = v << (8 * (c & 3));
  lo += c < 4 ? x : 0u;
  hi += c < 4 ? 0u : x;
}

//: A 32 x 32 BIT TRANSPOSE across the warp, lane d's bit i to lane i's bit d (the facts pass's)
__device__ __forceinline__ uint32_t transpose(uint32_t x, int lane) {
  constexpr uint32_t kHalf[5] = {0x0000ffffu, 0x00ff00ffu, 0x0f0f0f0fu, 0x33333333u, 0x55555555u};
#pragma unroll
  for (int k = 0; k < 5; ++k) {
    const int s = 16 >> k;
    const uint32_t keep = (lane & s) ? ~kHalf[k] : kHalf[k];
    const uint32_t send = __funnelshift_l(x, x, (lane & s) ? s : 32 - s) & keep;
    x = (x & keep) | __shfl_xor_sync(kAll, send, s);
  }
  return x;
}

//: ONE BATCH OF CLAIMS: the `take` queued tokens from `head` (lane l the l-th) sum their digits' biased counts and
//: take the least-used color no full digit and the union's cap forbid (ties to the union's least-used, then
//: `(c - t) & 7`; a token no color allows is blocked and takes the union's least-used), claimed by shared atomics, a
//: claim over a bound withdrawn
struct Claim {
  uint32_t t, lost, kept;
  bool loser;
};

__device__ __forceinline__ Claim claim_batch(uint8_t* smem, uint32_t head, uint32_t take, int lane) {
  const uint32_t* const masks = reinterpret_cast<const uint32_t*>(smem + Smem::kMasks);
  const uint16_t* const queue = reinterpret_cast<const uint16_t*>(smem + Smem::kQueue);
  uint32_t* const counts = reinterpret_cast<uint32_t*>(smem + Smem::kCounts);
  uint32_t* const ucount = reinterpret_cast<uint32_t*>(smem + Smem::kUcount);
  uint16_t* const color = reinterpret_cast<uint16_t*>(smem + Smem::kColor);

  const bool live = static_cast<uint32_t>(lane) < take;
  const uint32_t t = live ? queue[(head + lane) & (kQueueCap - 1)] : 0u;
  const uint32_t m = live ? masks[t] : 0u;
  const int digits = static_cast<int>(__reduce_max_sync(kAll, __popc(m)));
  uint32_t s0 = 0u, s1 = 0u, s2 = 0u, s3 = 0u, f0 = 0u, f1 = 0u, f2 = 0u, f3 = 0u;
  uint32_t x = m;
#pragma unroll 2
  for (int i = 0; i < digits; ++i) {
    const int d = x != 0u ? __ffs(x) - 1 : kZero;
    const uint4 v = reinterpret_cast<const uint4*>(counts)[d];
    s0 += v.x;
    s1 += v.y;
    s2 += v.z;
    s3 += v.w;
    f0 |= v.x;
    f1 |= v.y;
    f2 |= v.z;
    f3 |= v.w;
    x &= x - 1u;
  }
  //: color c full at bit 6 + c / 2 (even c) or 22 + c / 2 (odd c)
  const uint32_t full = (f0 & kFullPair) | (f1 & kFullPair) << 1 | (f2 & kFullPair) << 2 | (f3 & kFullPair) << 3;
  const uint4 ua = reinterpret_cast<const uint4*>(ucount)[0], ub = reinterpret_cast<const uint4*>(ucount)[1];
  const uint32_t u[kColors] = {ua.x, ua.y, ua.z, ua.w, ub.x, ub.y, ub.z, ub.w};
  const uint32_t f[kColors] = {s0 & 0xffffu, s0 >> 16, s1 & 0xffffu, s1 >> 16,
                               s2 & 0xffffu, s2 >> 16, s3 & 0xffffu, s3 >> 16};
  const uint32_t nt = (0u - t) & 7u;
  uint32_t best = ~0u;
#pragma unroll
  for (int c = 0; c < kColors; ++c) {
    const bool ok = ((full >> ((c & 1) ? 22 + c / 2 : 6 + c / 2)) & 1u) == 0u && u[c] < kUnionCap;
    best = ok ? min(best, f[c] << 10 | u[c] << 3 | ((nt + c) & 7u)) : best;
  }
  bool blocked = live && best == ~0u;
  if (__any_sync(kAll, blocked)) {
    uint32_t b = ~0u;
#pragma unroll
    for (int c = 0; c < kColors; ++c) b = u[c] < kUnionCap ? min(b, u[c] << 3 | c) : b;
    best = blocked ? b : best;
  }
  const uint32_t c = blocked ? best & 7u : ((best & 7u) + t) & 7u;
  const uint32_t shift = 16u * (c & 1u), inc = 1u << shift;

  bool over = false;
  if (live) over = atomicAdd(ucount + c, 1u) >= static_cast<uint32_t>(kUnionCap);
  x = m;
#pragma unroll 2
  for (int i = 0; i < digits; ++i) {
    uint32_t old = 0u;
    if (x != 0u) old = atomicAdd(counts + 4 * (__ffs(x) - 1) + (c >> 1), inc);
    over |= !blocked && ((old >> shift) & kFullBit) != 0u;
    x &= x - 1u;
  }
  const bool loser = live && over;
  if (__any_sync(kAll, loser)) {
    if (loser) atomicSub(ucount + c, 1u);
    x = loser ? m : 0u;
#pragma unroll 2
    for (int i = 0; i < digits; ++i) {
      if (x != 0u) atomicSub(counts + 4 * (__ffs(x) - 1) + (c >> 1), inc);
      x &= x - 1u;
    }
  }
  const bool won = live && !over;
  if (won) color[t] = static_cast<uint16_t>(c);
  return Claim{t, __ballot_sync(kAll, loser), __ballot_sync(kAll, won), loser};
}

//: THE SINGLES, the facts pass's, lane `d` digit `d` (`k` its singles): the union filled to a level pass by pass, then
//: by quota alone at the union's cap; each digit's singles' color ends posted for the deal
__device__ __forceinline__ void singles(uint8_t* smem, int k, int lane) {
  const uint32_t* const ucount = reinterpret_cast<const uint32_t*>(smem + Smem::kUcount);
  const int d = lane;
  const uint4 cw = reinterpret_cast<const uint4*>(smem + Smem::kCounts)[d];
  const uint32_t cnt_lo = (cw.x & 0xffu) | (cw.x >> 16 & 0xffu) << 8 | (cw.y & 0xffu) << 16 | (cw.y >> 16 & 0xffu) << 24;
  const uint32_t cnt_hi = (cw.z & 0xffu) | (cw.z >> 16 & 0xffu) << 8 | (cw.w & 0xffu) << 16 | (cw.w >> 16 & 0xffu) << 24;
  const int room_cap = static_cast<int>(kBias);
  const uint4 ua = reinterpret_cast<const uint4*>(ucount)[0], ub = reinterpret_cast<const uint4*>(ucount)[1];
  uint32_t u_lo = ua.x | ua.y << 8 | ua.z << 16 | ua.w << 24;
  uint32_t u_hi = ub.x | ub.y << 8 | ub.z << 16 | ub.w << 24;

  const uint32_t total = __reduce_add_sync(kAll, static_cast<uint32_t>(k));
  int level;
  {
    int fa = 0, fb = 0;
#pragma unroll
    for (int c = 0; c < kColors; ++c) {
      fa += max(0, lane - static_cast<int>(byte_of(u_lo, u_hi, c)));
      fb += max(0, lane + 32 - static_cast<int>(byte_of(u_lo, u_hi, c)));
    }
    const uint32_t ba = __ballot_sync(kAll, static_cast<uint32_t>(fa) >= total),
                   bb = __ballot_sync(kAll, static_cast<uint32_t>(fb) >= total);
    level = ba != 0u ? __ffs(ba) - 1 : (bb != 0u ? 32 + __ffs(bb) - 1 : kUnionCap);
  }

  uint32_t take_lo = 0u, take_hi = 0u;
  int rem = k;
#pragma unroll 1
  for (int go = __any_sync(kAll, rem > 0) != 0; go > 0;
       go = static_cast<int>(__any_sync(kAll, rem > 0 && level <= kUnionCap) != 0)) {
    uint32_t w_lo = 0u, w_hi = 0u;
    int r = rem;
#pragma unroll
    for (int i = 0; i < kColors; ++i) {
      const int c = (d + i) & 7;
      const int q = level - static_cast<int>(byte_of(u_lo, u_hi, c));
      const int room =
          room_cap - static_cast<int>(byte_of(cnt_lo, cnt_hi, c)) - static_cast<int>(byte_of(take_lo, take_hi, c));
      const int w = q > 0 ? max(0, min(r, room)) : 0;
      r -= w;
      add_byte(w_lo, w_hi, c, static_cast<uint32_t>(w));
    }
    uint32_t want[4] = {(w_lo & 0xffu) | (w_lo >> 8 & 0xffu) << 16, (w_lo >> 16 & 0xffu) | (w_lo >> 24) << 16,
                        (w_hi & 0xffu) | (w_hi >> 8 & 0xffu) << 16, (w_hi >> 16 & 0xffu) | (w_hi >> 24) << 16};
    uint32_t incl[4] = {want[0], want[1], want[2], want[3]};
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const uint32_t y = __shfl_up_sync(kAll, incl[p], off);
        incl[p] += lane >= off ? y : 0u;
      }
    }
    uint32_t sum[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) sum[p] = __shfl_sync(kAll, incl[p], 31);
    bool granted = false;
#pragma unroll
    for (int c = 0; c < kColors; ++c) {
      const int sh = 16 * (c & 1);
      const int wc = static_cast<int>((want[c >> 1] >> sh) & 0xffffu);
      const int before = static_cast<int>(((incl[c >> 1] - want[c >> 1]) >> sh) & 0xffffu);
      const int q = max(0, level - static_cast<int>(byte_of(u_lo, u_hi, c)));
      const int g = max(0, min(wc, q - before));
      add_byte(take_lo, take_hi, c, static_cast<uint32_t>(g));
      rem -= g;
      const int all = min(static_cast<int>((sum[c >> 1] >> sh) & 0xffffu), q);
      add_byte(u_lo, u_hi, c, static_cast<uint32_t>(all));
      granted |= all > 0;
    }
    level += granted ? 0 : 1;
  }

  {
    int incl = rem;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const int y = __shfl_up_sync(kAll, incl, off);
      incl += lane >= off ? y : 0;
    }
    const int lo = incl - rem, hi = incl;
    int bound = 0;
#pragma unroll
    for (int c = 0; c < kColors; ++c) {
      const int q = kUnionCap - static_cast<int>(byte_of(u_lo, u_hi, c));
      add_byte(take_lo, take_hi, c, static_cast<uint32_t>(max(0, min(hi, bound + q) - max(lo, bound))));
      bound += q;
    }
  }
  uint32_t end[kColors];
  uint32_t e = 0u;
#pragma unroll
  for (int c = 0; c < kColors; ++c) {
    e += byte_of(take_lo, take_hi, c);
    end[c] = e;
  }
  reinterpret_cast<uint4*>(smem + Smem::kEnds)[d] =
      make_uint4(end[0] | end[1] << 16, end[2] | end[3] << 16, end[4] | end[5] << 16, end[6] | end[7] << 16);
}

//: THE COLORER: a warp a union, its tile in registers (lane d digit d's 16 words), every table in its own shared memory
template <bool kWrite, int kCell>
__global__ __launch_bounds__(kThreads, kCtasMin) void color(const Params p) {
  extern __shared__ __align__(16) uint8_t smem_all[];
  const int warp = lab::warp_id(), lane = lab::lane_id();
  const int u = p.unit0 + static_cast<int>(blockIdx.x) * kWarps + warp;
  uint8_t* const smem = smem_all + warp * Smem::kBytes;
  uint32_t* const masks = reinterpret_cast<uint32_t*>(smem + Smem::kMasks);
  uint32_t* const uword = reinterpret_cast<uint32_t*>(smem + Smem::kUword);
  uint32_t* const mword = reinterpret_cast<uint32_t*>(smem + Smem::kMword);
  uint32_t* const tword = reinterpret_cast<uint32_t*>(smem + Smem::kTword);
  uint16_t* const queue = reinterpret_cast<uint16_t*>(smem + Smem::kQueue);
  uint16_t* const slotv = reinterpret_cast<uint16_t*>(smem + Smem::kColor);
  uint16_t* const by_slot = reinterpret_cast<uint16_t*>(smem + Smem::kBySlot);
  const uint32_t below = lab::held((1u << lane) - 1u), le = lab::held((2u << lane) - 1u);
  const long long t0 = clock64();
  PhaseClock clk(t0);

  //: THE TILE: lane d its digit's 16 words, four 16-byte loads
  uint32_t w[kWords];
  {
    const uint4* src = reinterpret_cast<const uint4*>(p.tiles + (static_cast<int64_t>(u) * kDigits + lane) * kWords);
#pragma unroll
    for (int i = 0; i < kWords / 4; ++i) {
      const uint4 v = src[i];
      w[4 * i] = v.x;
      w[4 * i + 1] = v.y;
      w[4 * i + 2] = v.z;
      w[4 * i + 3] = v.w;
    }
  }

  //: THE STASH'S PREFETCH into L2, its tokens and coefficients a 128-byte line a lane
  const uint32_t e0 = p.offsets[u], e1 = p.offsets[u + 1];
  const uint16_t* const toks = p.tokens + e0;
  const uint32_t* const coefs = p.coefs + e0;
#pragma unroll 1
  for (uint32_t b = 128u * lane; b < 6u * (e1 - e0); b += 32u * 128u) {
    const char* line = b < 2u * (e1 - e0) ? reinterpret_cast<const char*>(toks) + b
                                         : reinterpret_cast<const char*>(coefs) + (b - 2u * (e1 - e0));
    asm volatile("prefetch.global.L2 [%0];" ::"l"(line));
  }

  clk.mark(0);
  //: THE INPUTS: each token's digits by a transpose of each word and its count's three ballots (the union's words); lane
  //: d its digit's singles and biased counts; the union's counts cleared
  int n = 0, k = 0;
  uint32_t threes = 0u;
#pragma unroll
  for (int j = 0; j < kWords; ++j) {
    n += __popc(w[j]);
    const uint32_t m = transpose(w[j], lane);
    masks[32 * j + lane] = m;
    const int c = __popc(m);
    const uint32_t uw = __ballot_sync(kAll, c >= 1), mw = __ballot_sync(kAll, c >= 2),
                   tw = __ballot_sync(kAll, c >= 3);
    k += __popc(w[j] & uw & ~mw);
    threes += __popc(tw);
    if (lane == j) {
      uword[j] = uw;
      mword[j] = mw;
      tword[j] = tw;
    }
  }
  {
    const uint32_t r = (kBias - static_cast<uint32_t>(n + kColors - 1) / kColors) * 0x00010001u;
    reinterpret_cast<uint4*>(smem + Smem::kCounts)[lane] = make_uint4(r, r, r, r);
    if (lane == 0) reinterpret_cast<uint4*>(smem + Smem::kCounts)[kZero] = make_uint4(0u, 0u, 0u, 0u);
    if (lane < kColors) reinterpret_cast<uint32_t*>(smem + Smem::kUcount)[lane] = 0u;
  }
  __syncwarp();

  clk.mark(1);
  //: THE QUEUE: the multi-digit tokens, three-or-more first, each kind in token order
  uint32_t r3 = 0u, r2 = threes;
#pragma unroll
  for (int j = 0; j < kWords; ++j) {
    const uint32_t tw = tword[j], mw = mword[j] & ~tw;
    if ((tw >> lane) & 1u) queue[r3 + __popc(tw & below)] = static_cast<uint16_t>(32 * j + lane);
    if ((mw >> lane) & 1u) queue[r2 + __popc(mw & below)] = static_cast<uint16_t>(32 * j + lane);
    r3 += __popc(tw);
    r2 += __popc(mw);
  }
  __syncwarp();

  clk.mark(2);
  //: THE CLAIMS: batches of 32 to the queue's end, losers requeued, a batch that keeps nothing followed by its first
  //: token alone (which always keeps)
  {
    bool solo = false;
    uint32_t head = 0u, tail = r2;
    for (int go = __any_sync(kAll, head < tail) != 0; go > 0; --go)
#pragma unroll 1
      do {
        const uint32_t take = solo ? 1u : min(32u, tail - head);
        const Claim cl = claim_batch(smem, head, take, lane);
        head += take;
        __syncwarp();
        if (cl.loser) queue[(tail + __popc(cl.lost & below)) & (kQueueCap - 1)] = static_cast<uint16_t>(cl.t);
        tail += __popc(cl.lost);
        solo = cl.kept == 0u;
        __syncwarp();
      } while (__any_sync(kAll, head < tail));
  }

  clk.mark(3);
  //: THE SINGLES
  singles(smem, k, lane);
  {
    const uint4 gap = make_uint4(0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u);
    reinterpret_cast<uint4*>(by_slot)[lane] = gap;
    reinterpret_cast<uint4*>(by_slot)[lane + 32] = gap;
  }
  __syncwarp();

  clk.mark(4);
  //: THE SLOTS, word by word: each token's color (claimed, or a single's dealt by its index among its digit's singles:
  //: the color ends at or below it, by a packed subtraction) and its slot, 8 x its color's tokens before it + its color;
  //: lane d keeps its singles before the word, lane c its color's tokens
  uint32_t run = 0u, base = 0u;
  const uint4* const ends = reinterpret_cast<const uint4*>(smem + Smem::kEnds);
  const uint32_t own0 = lab::held(0u - (lane & 1u)), own1 = lab::held(0u - ((lane >> 1) & 1u)),
                 own2 = lab::held(0u - ((lane >> 2) & 1u));
#pragma unroll
  for (int j = 0; j < kWords; ++j) {
    const uint32_t uw = uword[j], single = uw & ~mword[j];
    const bool in = (uw >> lane) & 1u, sg = (single >> lane) & 1u;
    const int t = 32 * j + lane;
    const int dd = sg ? __ffs(masks[t]) - 1 : lane;
    const uint32_t wd = __shfl_sync(kAll, w[j], dd), rd = __shfl_sync(kAll, run, dd);
    const uint4 e = ends[dd];
    const uint32_t k1 = (rd + __popc(wd & single & below) + 1u) * 0x00010001u;
    const uint32_t dealt = __popc(~((e.x | 0x80008000u) - k1) & 0x80008000u) +
                           __popc(~((e.y | 0x80008000u) - k1) & 0x80008000u) +
                           __popc(~((e.z | 0x80008000u) - k1) & 0x80008000u) +
                           __popc(~((e.w | 0x80008000u) - k1) & 0x00008000u);
    const uint32_t c = in ? (sg ? dealt : slotv[t] & 7u) : 0u;
    const uint32_t v0 = __ballot_sync(kAll, c & 1u), v1 = __ballot_sync(kAll, c & 2u), v2 = __ballot_sync(kAll, c & 4u);
    const uint32_t m0 = 0u - (c & 1u), m1 = 0u - ((c >> 1) & 1u), m2 = 0u - ((c >> 2) & 1u);
    const uint32_t same = uw & ~(v0 ^ m0) & ~(v1 ^ m1) & ~(v2 ^ m2);
    const uint32_t mine = uw & ~(v0 ^ own0) & ~(v1 ^ own1) & ~(v2 ^ own2);
    const uint32_t b = __shfl_sync(kAll, base, static_cast<int>(c));
    const uint32_t s = kColors * (b + __popc(same & below)) + c;
    if (in) {
      slotv[t] = static_cast<uint16_t>(s);
      by_slot[s] = static_cast<uint16_t>(t);
    }
    base += __popc(mine);
    run += __popc(w[j] & single);
  }
  const int extent = kColors * static_cast<int>(__reduce_max_sync(kAll, base));
  __syncwarp();

  clk.mark(5);
  //: THE UNION OUT: the union by slot and its extent
  {
    uint4* const dst = reinterpret_cast<uint4*>(p.union_tok + static_cast<int64_t>(u) * kTokens);
    dst[lane] = reinterpret_cast<const uint4*>(by_slot)[lane];
    dst[lane + 32] = reinterpret_cast<const uint4*>(by_slot)[lane + 32];
    if (lane == 0) p.union_count[u] = extent;
  }
  __syncwarp();

  //: THE COUNTS: each entry of the stash into its (list, pass)'s color count
  const int pl = p.pass_log;
  uint32_t* const tab = reinterpret_cast<uint32_t*>(smem + Smem::kTab);
  uint16_t* const baset = reinterpret_cast<uint16_t*>(smem + Smem::kBase);
  uint16_t* const startv = reinterpret_cast<uint16_t*>(smem + Smem::kStart);
  uint8_t* const runv = smem + Smem::kRun;
  reinterpret_cast<uint4*>(tab)[lane] = make_uint4(0u, 0u, 0u, 0u);
  reinterpret_cast<uint4*>(tab)[lane + 32] = make_uint4(0u, 0u, 0u, 0u);
  reinterpret_cast<uint2*>(runv)[lane] = make_uint2(0u, 0u);
  const int count = static_cast<int>(e1 - e0);
  const int chunks = __shfl_sync(kAll, (count + 31) >> 5, 0);
  __syncwarp();
#pragma unroll 1
  for (int i = 0; i < chunks; ++i) {
    const int e = 32 * i + lane;
    if (e < count) {
      const uint32_t tok = toks[e];
      const uint32_t s = slotv[tok & 511u];
      atomicAdd(tab + ((tok >> 9) * kMaxPasses + (s >> pl)) * 2 + ((s >> 2) & 1u), 1u << (8 * (s & 3u)));
    }
  }
  __syncwarp();

  //: THE STARTS, lane d list d's passes: each its steps (its largest color count) and start; its places' bases, `start -
  //: 8 x (the color's tokens in the earlier passes) + color` a (pass, color); the list's starts out
  {
    uint2 tq[kMaxPasses];
#pragma unroll
    for (int q = 0; q < kMaxPasses; ++q) tq[q] = reinterpret_cast<const uint2*>(tab)[lane * kMaxPasses + q];
    __syncwarp();
    uint32_t b_lo = 0u, b_hi = 0u;
    int start = 0;
    uint32_t at[kListAt];
#pragma unroll
    for (int q = 0; q < kMaxPasses; ++q) {
      const uint32_t m4 = __vmaxu4(tq[q].x, tq[q].y);
      const int steps = static_cast<int>(max(max(m4 & 0xffu, (m4 >> 8) & 0xffu), max((m4 >> 16) & 0xffu, m4 >> 24)));
      uint32_t bp[kColors / 2];
#pragma unroll
      for (int c = 0; c < kColors; c += 2)
        bp[c / 2] = static_cast<uint32_t>(start - kColors * static_cast<int>(byte_of(b_lo, b_hi, c)) + c) |
                    static_cast<uint32_t>(start - kColors * static_cast<int>(byte_of(b_lo, b_hi, c + 1)) + c + 1) << 16;
      reinterpret_cast<uint4*>(baset)[lane * kMaxPasses + q] = make_uint4(bp[0], bp[1], bp[2], bp[3]);
      at[q] = static_cast<uint32_t>(start);
      b_lo += tq[q].x;
      b_hi += tq[q].y;
      start += kColors * steps;
    }
#pragma unroll
    for (int q = kMaxPasses; q < kListAt; ++q) at[q] = static_cast<uint32_t>(start);
    reinterpret_cast<uint4*>(startv)[lane] =
        make_uint4(at[0] | at[1] << 16, at[2] | at[3] << 16, at[4] | at[5] << 16, at[6] | at[7] << 16);
    reinterpret_cast<uint4*>(p.at)[static_cast<int64_t>(u) * kDigits + lane] =
        make_uint4(at[0] | at[1] << 16, at[2] | at[3] << 16, at[4] | at[5] << 16, at[6] | at[7] << 16);
  }
  __syncwarp();

  //: THE PLACES, a lane an entry in the stash's digit order: its rank among its list's tokens of its color (its digit's
  //: run of the chunk with its color below it, plus the list's count so far), its place its pass's start + 8 x (rank -
  //: the color's tokens in the earlier passes) + its color. A list's pads are written as the walk reaches it, a lane a
  //: place (place p of pass q: step (p - start q) / 8 of color p % 8, a pad past the color's count)
  uint2* const ent = p.entries + static_cast<int64_t>(u) * kDigits * kListCap;
  uint2* const hd = p.heads + static_cast<int64_t>(u) * kDigits * kHeadEntries;
  const int cl = lane & 7;
  const uint32_t pad_lo = kWrite ? static_cast<uint32_t>(cl) : cl * kRow + 16u * carry::pool_chunk(kRow, cl, 0u);
  const uint32_t pad_hi = kWrite ? 0u : carry::panel_row(cl) << 16;
  int last = -1;
#pragma unroll 1
  for (int i = 0; i < chunks; ++i) {
    const int e = 32 * i + lane;
    const bool live = e < count;
    const uint32_t tok = live ? toks[e] : 0u;
    const uint32_t coef = live ? coefs[e] : 0u;
    const int t = static_cast<int>(tok & 511u), d = static_cast<int>(tok >> 9), key = live ? d : -1;
    const int prev = __shfl_up_sync(kAll, key, 1);
    const uint32_t firsts = __ballot_sync(kAll, lane == 0 || prev != key);
    //: the pads of each list whose first entry is in the chunk: its whole run, each place its pass's pad of its color
    //: (the entries land over theirs below)
    for (uint32_t fresh = __ballot_sync(kAll, live && (lane > 0 ? prev != key : key != last)); fresh != 0u;
         fresh &= fresh - 1u) {
      const int dl = __shfl_sync(kAll, d, __ffs(fresh) - 1);
      const uint4 sv = reinterpret_cast<const uint4*>(startv)[dl];
      const int s1 = static_cast<int>(sv.x >> 16), s2 = static_cast<int>(sv.y & 0xffffu),
                s3 = static_cast<int>(sv.y >> 16), len = static_cast<int>(sv.z & 0xffffu);
      uint2* const out = ent + dl * kListCap;
      if (lane < kHeadEntries && lane < len) {
        const uint32_t qp = (lane >= s1) + (lane >= s2) + (lane >= s3);
        const uint2 e2 = make_uint2(kWrite ? (qp << pl) + pad_lo : pad_lo, pad_hi);
        hd[((dl & 7) * 4 + (dl >> 3)) * kHeadEntries + lane] = e2;
      }
#pragma unroll 1
      for (int pp = lane; pp < len; pp += 32) {
        const uint32_t qp = (pp >= s1) + (pp >= s2) + (pp >= s3);
        out[pp] = make_uint2(kWrite ? (qp << pl) + pad_lo : pad_lo, pad_hi);
      }
    }
    __syncwarp();
    const uint32_t s = slotv[t];
    const int c = static_cast<int>(s & 7u), q = static_cast<int>(s >> pl);
    const uint32_t seg = 1u << (31 - __clz(firsts & le)), next = firsts & ~le;
    const uint32_t v0 = __ballot_sync(kAll, c & 1), v1 = __ballot_sync(kAll, c & 2), v2 = __ballot_sync(kAll, c & 4);
    const uint32_t m0 = 0u - (s & 1u), m1 = 0u - ((s >> 1) & 1u), m2 = 0u - ((s >> 2) & 1u);
    const uint32_t group = ((next & (0u - next)) - seg) & ~(v0 ^ m0) & ~(v1 ^ m1) & ~(v2 ^ m2);
    const uint32_t ran = runv[d * kColors + c];
    const uint32_t rank = ran + __popc(group & below);
    const int pos = baset[(d * kMaxPasses + q) * kColors + c] + kColors * static_cast<int>(rank);
    uint2 o2;
    if constexpr (kWrite) {
      o2 = make_uint2(s, coef);
    } else {
      const uint32_t loc = s - (static_cast<uint32_t>(q) << pl);
      o2 = make_uint2(loc * kRow + 16u * carry::pool_chunk(kRow, loc, 0u),
                      (coef & 0xffffu) | (carry::kPadBytes + carry::panel_row(loc)) << 16);
    }
    if (live) {
      ent[d * kListCap + pos] = o2;
      if (pos < kHeadEntries) hd[((d & 7) * 4 + (d >> 3)) * kHeadEntries + pos] = o2;
    }
    __syncwarp();
    if (live) runv[d * kColors + c] = static_cast<uint8_t>(ran + __popc(group));
    last = __shfl_sync(kAll, key, 31);
    __syncwarp();
  }
  clk.mark(7);
  if (lane == 0) {
    p.cycles[u] = clock64() - t0;
#pragma unroll
    for (int ph = 0; ph < kTimed; ++ph) p.phases[static_cast<int64_t>(u) * kTimed + ph] = clk.at[ph];
  }
}

//: THE STREAM HALF'S MODEL, the co-run's other kernel: `ctas` persistent CTAs of eight warps an SM, each taking units
//: (batch-head, window, then the window's 32: side, level, group) in turn, the next unit's window in flight -- a unit's
//: 512 tokens' 32-byte runs of its side's head-major plane `[H, T, 2 x 256]` by cp.async into a 16 KB window, each digit's words of live bits by a
//: ballot a 32-token block, the unit's 2 KB of words written, and a list unit's stash bytes (6 an entry, the cell's own
//: counts) written -- the facts pass less its coloring
constexpr int kStreamWarps = 8, kStreamThreads = 32 * kStreamWarps;
constexpr int kHeads = 4, kGroups = 8, kWindowsAll = 128, kTokensAll = kWindowsAll * kTokens, kPerWindow = 32;
constexpr int64_t kPlaneBytes = static_cast<int64_t>(kHeads) * kTokensAll * 512;
constexpr int64_t kRowBytes = 2 * kPlaneBytes;
constexpr int kStreamUnits = kHeads * kWindowsAll * kPerWindow;

__global__ __launch_bounds__(kStreamThreads) void stream(const uint8_t* __restrict__ rows, uint32_t* __restrict__ words,
                                                        uint8_t* __restrict__ stash,
                                                        const uint32_t* __restrict__ offsets, int write_units) {
  extern __shared__ __align__(16) uint8_t wins[];
  const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
  //: a unit's window into buffer `buf`: a thread four 16-byte copies, a token's run a lane pair
  const auto issue = [&](int unit, int buf) {
    if (unit < kStreamUnits) {
      const int x = unit % kPerWindow, w = unit / kPerWindow % kWindowsAll, h = unit / (kPerWindow * kWindowsAll);
      const uint8_t* base = rows + (x >> 4) * kPlaneBytes + (static_cast<int64_t>(h) * kTokensAll + kTokens * w) * 512 +
                            256 * ((x >> 3) & 1) + 32 * (x & 7);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int k = tid + kStreamThreads * j, tk = k >> 1, half = k & 1;
        lab::cp_async16(lab::smem(wins + buf * kTokens * 32 + 32 * tk + 16 * half), base + 512 * tk + 16 * half);
      }
    }
    lab::cp_commit();
  };
  const int stride = static_cast<int>(gridDim.x);
  int unit = static_cast<int>(blockIdx.x);
  issue(unit, 0);
#pragma unroll 1
  for (int it = 0; unit < kStreamUnits; ++it, unit += stride) {
    issue(unit + stride, (it + 1) & 1);
    lab::cp_wait<1>();
    __syncthreads();
    const uint8_t* const win = wins + (it & 1) * kTokens * 32;
    const int x = unit % kPerWindow, w = unit / kPerWindow % kWindowsAll, h = unit / (kPerWindow * kWindowsAll);
    const int side = x >> 4, lev = (x >> 3) & 1, grp = x & 7;
    //: THE VOTE: warp v tokens 64 v ..; lane l's token's 32 bytes to a mask of live digits, a ballot a digit
    uint32_t mine[2];
#pragma unroll
    for (int b = 0; b < 2; ++b) {
      const int tk = 64 * warp + 32 * b + lane;
      const uint4 xa = reinterpret_cast<const uint4*>(win + 32 * tk)[0], xb = reinterpret_cast<const uint4*>(win + 32 * tk)[1];
      const uint32_t q[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
      uint32_t m = 0u;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const uint32_t nz = (((q[i] & 0x7f7f7f7fu) + 0x7f7f7f7fu) | q[i]) & 0x80808080u;
        m |= (((nz >> 7) & 1u) | ((nz >> 14) & 2u) | ((nz >> 21) & 4u) | ((nz >> 28) & 8u)) << (4 * i);
      }
      mine[b] = 0u;
#pragma unroll
      for (int d = 0; d < 32; ++d) {
        const uint32_t v = __ballot_sync(kAll, (m >> d) & 1u);
        mine[b] = lane == d ? v : mine[b];
      }
    }
    __syncthreads();
    uint32_t* const out = words + static_cast<int64_t>(unit) * 512;
    out[lane * 16 + 2 * warp] = mine[0];
    out[lane * 16 + 2 * warp + 1] = mine[1];
    //: THE STASH: a list unit's (the write side's level 0, the read side's level 1) bytes, 16 a thread
    if (lev != side) {
      const int cu = (side == 1 ? 0 : write_units) + (h * kWindowsAll + w) * kGroups + grp;
      const uint32_t lo = 6u * offsets[cu], hi = 6u * offsets[cu + 1];
      for (uint32_t b = (lo & ~15u) + 16u * tid; b < hi; b += 16u * kStreamThreads)
        *reinterpret_cast<uint4*>(stash + b) = make_uint4(mine[0], mine[1], b, lo);
    }
  }
  lab::cp_wait<0>();
}

//: A CELL: its inputs (`facts_color.inputs.py`) on the host and the device, and the lists out
struct Cell {
  std::string name;
  int units = 0, write_units = 0, write_pass = 0, read_pass = 0, row = 0;
  std::vector<uint32_t> tiles, offsets, coefs;
  std::vector<uint16_t> tokens;
  uint32_t *d_tiles = nullptr, *d_offsets = nullptr, *d_coefs = nullptr;
  uint16_t* d_tokens = nullptr;
  uint2 *entries = nullptr, *heads = nullptr;
  uint16_t *at = nullptr, *union_tok = nullptr;
  int* union_count = nullptr;
  long long *cycles = nullptr, *phases = nullptr;
};

template <class T>
std::vector<T> read_file(const std::string& path, size_t n) {
  std::vector<T> out(n);
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f || std::fread(out.data(), sizeof(T), n, f) != n) {
    std::fprintf(stderr, "facts_color: %s is short or missing\n", path.c_str());
    std::exit(1);
  }
  std::fclose(f);
  return out;
}

template <class T>
T* upload(const std::vector<T>& h) {
  T* d = nullptr;
  LAB_CHECK(cudaMalloc(&d, h.size() * sizeof(T)));
  LAB_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

Cell load_cell(const std::string& dir, const std::string& name) {
  Cell c;
  c.name = name;
  const std::vector<int> meta = read_file<int>(dir + "/meta.bin", 6);
  if (meta[0] != 0x434F4C46) {
    std::fprintf(stderr, "facts_color: %s/meta.bin is not this lab's\n", dir.c_str());
    std::exit(1);
  }
  c.units = meta[1];
  c.write_units = meta[2];
  c.write_pass = meta[3];
  c.read_pass = meta[4];
  c.row = meta[5];
  c.tiles = read_file<uint32_t>(dir + "/tiles.bin", static_cast<size_t>(c.units) * kDigits * kWords);
  c.offsets = read_file<uint32_t>(dir + "/offsets.bin", static_cast<size_t>(c.units) + 1);
  c.tokens = read_file<uint16_t>(dir + "/tokens.bin", c.offsets.back());
  c.coefs = read_file<uint32_t>(dir + "/coefs.bin", c.offsets.back());
  c.d_tiles = upload(c.tiles);
  c.d_offsets = upload(c.offsets);
  c.d_tokens = upload(c.tokens);
  c.d_coefs = upload(c.coefs);
  LAB_CHECK(cudaMalloc(&c.entries, static_cast<size_t>(c.units) * kDigits * kListCap * sizeof(uint2)));
  LAB_CHECK(cudaMalloc(&c.heads, static_cast<size_t>(c.units) * kDigits * kHeadEntries * sizeof(uint2)));
  LAB_CHECK(cudaMalloc(&c.at, static_cast<size_t>(c.units) * kDigits * kListAt * 2));
  LAB_CHECK(cudaMalloc(&c.union_tok, static_cast<size_t>(c.units) * kTokens * 2));
  LAB_CHECK(cudaMalloc(&c.union_count, static_cast<size_t>(c.units) * 4));
  LAB_CHECK(cudaMalloc(&c.cycles, static_cast<size_t>(c.units) * 8));
  LAB_CHECK(cudaMalloc(&c.phases, static_cast<size_t>(c.units) * kTimed * 8));
  return c;
}

void free_cell(Cell& c) {
  for (void* ptr : {static_cast<void*>(c.d_tiles), static_cast<void*>(c.d_offsets), static_cast<void*>(c.d_tokens),
                    static_cast<void*>(c.d_coefs), static_cast<void*>(c.entries), static_cast<void*>(c.heads),
                    static_cast<void*>(c.at), static_cast<void*>(c.union_tok), static_cast<void*>(c.union_count),
                    static_cast<void*>(c.cycles), static_cast<void*>(c.phases)})
    LAB_CHECK(cudaFree(ptr));
}

int log2_of(int x) {
  int l = 0;
  while ((1 << l) < x) ++l;
  return l;
}

template <bool kWrite, int kCell>
void launch_side(const Cell& c, cudaStream_t st = nullptr) {
  Params p{};
  p.tiles = c.d_tiles;
  p.offsets = c.d_offsets;
  p.tokens = c.d_tokens;
  p.coefs = c.d_coefs;
  p.unit0 = kWrite ? 0 : c.write_units;
  p.units = kWrite ? c.write_units : c.units - c.write_units;
  p.pass_log = log2_of(kWrite ? c.write_pass : c.read_pass);
  p.row = c.row;
  p.entries = c.entries;
  p.at = c.at;
  p.heads = c.heads;
  p.union_tok = c.union_tok;
  p.union_count = c.union_count;
  p.cycles = c.cycles;
  p.phases = c.phases;
  color<kWrite, kCell><<<(p.units + kWarps - 1) / kWarps, kThreads, kWarps * Smem::kBytes, st>>>(p);
}

//: THE CHECK: each union's coloring against its invariants (every union token one slot, the residue layout, the
//: union's cap, the extent) and each list exactly against the mirror given that coloring (`facts.md#lists-out`'s
//: passes, round order, pads, starts and heads); the coloring's quality counted (tokens over a digit's capacity, gap
//: slots)
struct Verdict {
  size_t bad = 0, over = 0, gaps = 0, unions = 0, entries = 0;
  std::string first;
};

uint2 entry_of(bool write, int pl, int row, int s, int q, uint32_t coef) {
  if (write) return make_uint2(static_cast<uint32_t>(s), coef);
  const uint32_t loc = static_cast<uint32_t>(s - (q << pl));
  return make_uint2(loc * row + 16u * carry::pool_chunk(row, loc, 0u),
                    (coef & 0xffffu) | (carry::kPadBytes + carry::panel_row(loc)) << 16);
}

uint2 pad_of(bool write, int pl, int row, int q, int c) {
  if (write) return make_uint2((static_cast<uint32_t>(q) << pl) + c, 0u);
  return make_uint2(c * row + 16u * carry::pool_chunk(row, c, 0u), carry::panel_row(c) << 16);
}

Verdict check_cell(const Cell& c) {
  const size_t lists = static_cast<size_t>(c.units) * kDigits;
  std::vector<uint2> entries(lists * kListCap), heads(lists * kHeadEntries);
  std::vector<uint16_t> at(lists * kListAt), union_tok(static_cast<size_t>(c.units) * kTokens);
  std::vector<int> count(c.units);
  LAB_CHECK(cudaMemcpy(entries.data(), c.entries, entries.size() * sizeof(uint2), cudaMemcpyDeviceToHost));
  LAB_CHECK(cudaMemcpy(heads.data(), c.heads, heads.size() * sizeof(uint2), cudaMemcpyDeviceToHost));
  LAB_CHECK(cudaMemcpy(at.data(), c.at, at.size() * 2, cudaMemcpyDeviceToHost));
  LAB_CHECK(cudaMemcpy(union_tok.data(), c.union_tok, union_tok.size() * 2, cudaMemcpyDeviceToHost));
  LAB_CHECK(cudaMemcpy(count.data(), c.union_count, count.size() * 4, cudaMemcpyDeviceToHost));
  Verdict v;
  const auto fail = [&](int u, const std::string& what) {
    if (v.bad++ == 0) v.first = "unit " + std::to_string(u) + ": " + what;
  };
  for (int u = 0; u < c.units; ++u) {
    const bool write = u < c.write_units;
    const int pass = write ? c.write_pass : c.read_pass, pl = log2_of(pass);
    const uint32_t* tile = c.tiles.data() + static_cast<size_t>(u) * kDigits * kWords;
    uint32_t mask[kTokens];
    int n[kDigits] = {}, members = 0;
    for (int t = 0; t < kTokens; ++t) {
      mask[t] = 0;
      for (int d = 0; d < kDigits; ++d) mask[t] |= ((tile[d * kWords + t / 32] >> (t % 32)) & 1u) << d;
      members += mask[t] != 0;
      for (int d = 0; d < kDigits; ++d) n[d] += (mask[t] >> d) & 1;
    }
    int slot_of[kTokens], per[kColors] = {};
    std::fill(slot_of, slot_of + kTokens, -1);
    const uint16_t* by = union_tok.data() + static_cast<size_t>(u) * kTokens;
    for (int s = 0; s < kTokens; ++s) {
      if (by[s] == carry::kGap) continue;
      if (by[s] >= kTokens || mask[by[s]] == 0 || slot_of[by[s]] != -1) {
        fail(u, "slot " + std::to_string(s) + " holds no union token once");
        continue;
      }
      slot_of[by[s]] = s;
    }
    for (int col = 0; col < kColors; ++col) {
      int last = -1;
      bool gap = false;
      for (int r = 0; r < kUnionCap; ++r) {
        const uint16_t t = by[kColors * r + col];
        if (t == carry::kGap) {
          gap = true;
          continue;
        }
        if (gap || static_cast<int>(t) <= last) fail(u, "color " + std::to_string(col) + " not dense in token order");
        last = t;
        ++per[col];
      }
    }
    int widest = 0, placed = 0;
    for (int col = 0; col < kColors; ++col) {
      widest = std::max(widest, per[col]);
      placed += per[col];
    }
    if (placed != members) fail(u, "the union is not placed");
    if (count[u] != kColors * widest) fail(u, "the extent is not 8 x the widest color");
    v.gaps += static_cast<size_t>(kColors * widest - members);
    v.unions += static_cast<size_t>(members);
    const int passes = (count[u] + pass - 1) / pass;

    const uint32_t e0 = c.offsets[u], e1 = c.offsets[u + 1];
    std::vector<std::vector<std::pair<int, uint32_t>>> mine(kDigits);
    for (uint32_t e = e0; e < e1; ++e) {
      const int d = c.tokens[e] >> 9, t = c.tokens[e] & 511;
      if (!((mask[t] >> d) & 1u) || slot_of[t] < 0) fail(u, "a stash entry outside its digit");
      else mine[d].push_back({t, c.coefs[e]});
    }
    for (int d = 0; d < kDigits; ++d) {
      if (static_cast<int>(mine[d].size()) != n[d]) fail(u, "the stash disagrees with the tile");
      const int cap = (n[d] + kColors - 1) / kColors;
      int cols[kColors] = {};
      for (const auto& [t, coef] : mine[d]) ++cols[slot_of[t] & 7];
      for (int col = 0; col < kColors; ++col) v.over += static_cast<size_t>(std::max(0, cols[col] - cap));

      std::vector<uint2> want;
      uint16_t starts[kListAt];
      for (int q = 0; q < kMaxPasses; ++q) {
        starts[q] = static_cast<uint16_t>(want.size());
        if (q >= passes) continue;
        std::vector<std::pair<int, uint32_t>> by_color[kColors];
        for (const auto& [t, coef] : mine[d])
          if (slot_of[t] >> pl == q) by_color[slot_of[t] & 7].push_back({slot_of[t], coef});
        size_t steps = 0;
        for (auto& b : by_color) {
          std::sort(b.begin(), b.end());
          steps = std::max(steps, b.size());
        }
        for (size_t i = 0; i < steps; ++i)
          for (int col = 0; col < kColors; ++col)
            want.push_back(i < by_color[col].size()
                               ? entry_of(write, pl, c.row, by_color[col][i].first, q, by_color[col][i].second)
                               : pad_of(write, pl, c.row, q, col));
      }
      for (int q = kMaxPasses; q < kListAt; ++q) starts[q] = static_cast<uint16_t>(want.size());
      const size_t list = static_cast<size_t>(u) * kDigits + d;
      v.entries += want.size();
      for (int q = 0; q < kListAt; ++q)
        if (at[list * kListAt + q] != starts[q]) fail(u, "list " + std::to_string(d) + "'s starts");
      for (size_t i = 0; i < want.size(); ++i) {
        const uint2 got = entries[list * kListCap + i];
        if (got.x != want[i].x || got.y != want[i].y) fail(u, "list " + std::to_string(d) + "'s entry " + std::to_string(i));
      }
      const uint2* h = heads.data() + (static_cast<size_t>(u) * 8 + (d & 7)) * (4 * kHeadEntries) + (d >> 3) * kHeadEntries;
      for (size_t i = 0; i < std::min<size_t>(kHeadEntries, want.size()); ++i)
        if (h[i].x != want[i].x || h[i].y != want[i].y) fail(u, "list " + std::to_string(d) + "'s head " + std::to_string(i));
    }
  }
  return v;
}

//: THE BUDGET LINES a union (KERNEL_STANDARDS section 29), by class, stated before the code: the tile 4 loads; the
//: inputs 16 transposes (5 shuffles and ~20 integer each) and 48 ballots; the queue ~200; a claim batch ~60 + 26 a digit
//: of its widest token (k 4: ~2 batches of ~3 digits, k 16: ~11 of ~5); the singles ~900; the slots ~45 a word; the
//: counts ~10 and the places ~35 a chunk of 32 entries (k 4: 8 chunks, k 16: 32); the pads ~6 a pad (~25 / ~45 a
//: list, lane-serial)
struct Line {
  const char* cell;
  double shuffle, load, store, atomic, select, integer, control, global;
};

constexpr Line kLine[2] = {{"k4", 250, 200, 150, 20, 1150, 1650, 150, 70},
                           {"k16", 320, 460, 180, 110, 2200, 4400, 350, 220}};

template <bool kWrite, int kCell>
std::string declare(const lab::Bench& b, int units) {
  const std::string pattern = "facts_color::color<\\(bool\\)" + std::to_string(kWrite ? 1 : 0) + ", \\(int\\)" +
                              std::to_string(kCell) + ">\\(";
  b.expect(pattern, "REDUX", ">=", 2);
  b.expect(pattern, "ATOMS", ">=", 3);
  b.expect(pattern, "LDL", "==", 0);
  b.expect(pattern, "STL", "==", 0);
  //: the first form's divergent loops (the claims' digit walks, the pads, the lane-predicated stores), a debt item 9's
  //: source patterns pay off
  b.expect(pattern, "LOOP_CONVERGENCE", "<=", kWrite ? 278 : 342);
  const Line& l = kLine[kCell];
  b.units(pattern, "union", units);
  b.budget(pattern, "shuffle", l.shuffle);
  b.budget(pattern, "shared load", l.load);
  b.budget(pattern, "shared store", l.store);
  b.budget(pattern, "shared atomic", l.atomic);
  b.budget(pattern, "select/compare", l.select);
  b.budget(pattern, "integer/address", l.integer);
  b.budget(pattern, "control", l.control);
  b.budget(pattern, "global", l.global);
  return pattern;
}

template <int kCell>
void run_cell(const lab::Bench& b, const std::string& dir) {
  const std::string cell = kLine[kCell].cell;
  declare<true, kCell>(b, 4096);
  declare<false, kCell>(b, 4096);
  if (b.expect_only) return;
  Cell c = load_cell(dir + "/" + cell, cell);
  if (c.write_units % kWarps || (c.units - c.write_units) % kWarps || c.row != kRow) {
    std::fprintf(stderr, "facts_color: a side's units are not a whole number of CTAs, or the row is not %d\n", kRow);
    std::exit(1);
  }
  LAB_CHECK(cudaFuncSetAttribute(color<true, kCell>, cudaFuncAttributeMaxDynamicSharedMemorySize, kWarps * Smem::kBytes));
  LAB_CHECK(cudaFuncSetAttribute(color<false, kCell>, cudaFuncAttributeMaxDynamicSharedMemorySize, kWarps * Smem::kBytes));
  cudaEvent_t a, z;
  LAB_CHECK(cudaEventCreate(&a));
  LAB_CHECK(cudaEventCreate(&z));
  const auto elapsed = [&] {
    float ms = 0.0f;
    LAB_CHECK(cudaEventElapsedTime(&ms, a, z));
    return 1000.0 * ms;
  };
  const std::string units = std::to_string(c.units), side = std::to_string(c.write_units);
  const std::string both = "both sides' " + units + " unions / " + cell;
  const std::string wname = "the write side's " + side + " unions / " + cell;
  const std::string rname = "the read side's " + std::to_string(c.units - c.write_units) + " unions / " + cell;
  b.record(both, "us", b.variant([&] {
    LAB_CHECK(cudaEventRecord(a));
    launch_side<true, kCell>(c);
    launch_side<false, kCell>(c);
    LAB_CHECK(cudaEventRecord(z));
  }, elapsed));
  b.record(wname, "us", b.variant([&] {
    LAB_CHECK(cudaEventRecord(a));
    launch_side<true, kCell>(c);
    LAB_CHECK(cudaEventRecord(z));
  }, elapsed));
  b.record(rname, "us", b.variant([&] {
    LAB_CHECK(cudaEventRecord(a));
    launch_side<false, kCell>(c);
    LAB_CHECK(cudaEventRecord(z));
  }, elapsed));
  {
    uint8_t *rows = nullptr, *stash = nullptr;
    uint32_t* words = nullptr;
    LAB_CHECK(cudaMalloc(&rows, kRowBytes));
    LAB_CHECK(cudaMemset(rows, 1, kRowBytes));
    LAB_CHECK(cudaMalloc(&words, static_cast<size_t>(kStreamUnits) * 512 * 4));
    LAB_CHECK(cudaMalloc(&stash, 6ull * c.offsets.back() + 64));
    cudaStream_t s1, s2;
    LAB_CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    LAB_CHECK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
    cudaEvent_t joined;
    LAB_CHECK(cudaEventCreateWithFlags(&joined, cudaEventDisableTiming));
    LAB_CHECK(cudaFuncSetAttribute(stream, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * kTokens * 32));
    for (const int ctas : {1, 2}) {
      const auto run_stream = [&](cudaStream_t st) {
        stream<<<ctas * b.sms, kStreamThreads, 2 * kTokens * 32, st>>>(rows, words, stash, c.d_offsets, c.write_units);
      };
      const std::string sm = std::to_string(ctas) + " CTAs an SM";
      b.record("the stream half's model alone, " + sm + " / " + cell, "us", b.variant([&] {
        LAB_CHECK(cudaEventRecord(a, s1));
        run_stream(s1);
        LAB_CHECK(cudaEventRecord(z, s1));
      }, elapsed));
      b.unchecked("the stream half's model alone, " + sm + " / " + cell,
                  "a model of the stream half's traffic, its outputs a sink");
      b.record("the stream half's model, " + sm + ", and the colorer beside it / " + cell, "us", b.variant([&] {
        LAB_CHECK(cudaEventRecord(a, s1));
        LAB_CHECK(cudaStreamWaitEvent(s2, a, 0));
        run_stream(s1);
        launch_side<true, kCell>(c, s2);
        launch_side<false, kCell>(c, s2);
        LAB_CHECK(cudaEventRecord(joined, s2));
        LAB_CHECK(cudaStreamWaitEvent(s1, joined, 0));
        LAB_CHECK(cudaEventRecord(z, s1));
      }, elapsed));
      b.unchecked("the stream half's model, " + sm + ", and the colorer beside it / " + cell,
                  "the colorer's outputs are checked alone; this times the two co-resident");
    }
    LAB_CHECK(cudaStreamDestroy(s1));
    LAB_CHECK(cudaStreamDestroy(s2));
    LAB_CHECK(cudaEventDestroy(joined));
    LAB_CHECK(cudaFree(rows));
    LAB_CHECK(cudaFree(words));
    LAB_CHECK(cudaFree(stash));
  }
  std::vector<long long> cyc(c.units);
  LAB_CHECK(cudaMemcpy(cyc.data(), c.cycles, cyc.size() * 8, cudaMemcpyDeviceToHost));
  lab::Samples wc, rc;
  double ws = 0, rs = 0;
  for (int u = 0; u < c.units; ++u) (u < c.write_units ? ws : rs) += double(cyc[u]);
  wc.v.push_back(ws / c.write_units);
  rc.v.push_back(rs / (c.units - c.write_units));
  {
    static const char* kPhaseName[kTimed] = {"tile", "inputs", "queue", "claims", "singles", "slots", "union out",
                                              "lists"};
    std::vector<long long> ph(static_cast<size_t>(c.units) * kTimed);
    LAB_CHECK(cudaMemcpy(ph.data(), c.phases, ph.size() * 8, cudaMemcpyDeviceToHost));
    for (int side = 0; side < 2; ++side) {
      const int lo = side ? c.write_units : 0, hi = side ? c.units : c.write_units;
      std::fprintf(stderr, "facts_color %s %s side, a warp's cycles a union by phase:", cell.c_str(),
                   side ? "read" : "write");
      for (int k = 0; k < kTimed; ++k) {
        double sum = 0;
        for (int u2 = lo; u2 < hi; ++u2) sum += double(ph[static_cast<size_t>(u2) * kTimed + k]);
        std::fprintf(stderr, " %s %.0f", kPhaseName[k], sum / (hi - lo));
      }
      std::fprintf(stderr, "\n");
    }
  }
  b.record("a warp's cycles a union, write side / " + cell, "cycles", wc);
  b.record("a warp's cycles a union, read side / " + cell, "cycles", rc);

  LAB_CHECK(cudaMemset(c.entries, 0xAB, static_cast<size_t>(c.units) * kDigits * kListCap * sizeof(uint2)));
  LAB_CHECK(cudaMemset(c.heads, 0xAB, static_cast<size_t>(c.units) * kDigits * kHeadEntries * sizeof(uint2)));
  LAB_CHECK(cudaMemset(c.union_tok, 0xAB, static_cast<size_t>(c.units) * kTokens * 2));
  launch_side<true, kCell>(c);
  launch_side<false, kCell>(c);
  LAB_CHECK(cudaDeviceSynchronize());
  const Verdict v = check_cell(c);
  const std::string detail = units + " unions (" + std::to_string(v.unions) + " tokens), " + std::to_string(v.entries) +
                             " list entries against the mirror given the coloring: " + std::to_string(v.bad) +
                             " failures" + (v.bad ? " (first: " + v.first + ")" : std::string());
  for (const std::string& name : {both, wname, rname, std::string("a warp's cycles a union, write side / ") + cell,
                                  std::string("a warp's cycles a union, read side / ") + cell})
    b.check(name, v.bad == 0, detail);
  lab::Samples over, gaps;
  over.v.push_back(double(v.over) / c.units);
  gaps.v.push_back(double(v.gaps) / c.units);
  b.record("tokens over a digit's capacity a union / " + cell, "tokens", over);
  b.unchecked("tokens over a digit's capacity a union / " + cell, "a count of the checked outputs");
  b.record("gap slots a union / " + cell, "slots", gaps);
  b.unchecked("gap slots a union / " + cell, "a count of the checked outputs");
  free_cell(c);
}

}  // namespace facts_color

int main(int argc, char** argv) {
  using namespace facts_color;
  lab::Bench b(argc, argv);
  std::string dir;
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], "--inputs")) dir = argv[i + 1];
  if (!b.expect_only && dir.empty()) {
    std::fprintf(stderr, "facts_color: --inputs <directory> (tools/lab.py writes it)\n");
    return 1;
  }
  run_cell<0>(b, dir);
  run_cell<1>(b, dir);
}
