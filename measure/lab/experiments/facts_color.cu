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

//: THE CTA: four warps, a union each; four CTAs an SM on the write side, five on the read side (its two passes' tables
//: the smaller)
constexpr int kWarps = 4;
template <bool kWrite> constexpr int kCtasOf = kWrite ? 4 : 5;
constexpr int kWords = 16, kTokens = 32 * kWords, kDigits = 32, kColors = 8, kUnionCap = kTokens / kColors;
constexpr int kQueueCap = kTokens, kListCap = kTokens, kListAt = 8, kHeadEntries = 16, kMaxPasses = 4;
constexpr uint32_t kAll = 0xffffffffu;
//: THE READ POOL'S ROW, the sm_86 even arm's 64 bytes
constexpr int kRow = 64;

//: A WARP'S SHARED MEMORY, by offset: each token's digits (then each (list, pass)'s color counts, then each (list, pass,
//: color)'s place base), the claim queue a segment a pass (then a pass's singles' color ends and each digit's singles
//: dealt, then the union by slot, then each list's starts and color counts so far), each (pass, digit)'s biased color
//: counts and each pass's zero record, each pass's color counts, the union's words, each pass's tokens a word, each
//: word's tokens of its next pass and its first token's pass, each token's color (then its slot)
template <int kQ>
struct SmemOf {
  static constexpr int kMasks = 0;
  static constexpr int kTab = 0;
  static constexpr int kSlots = kMasks + kTokens * 2;
  static constexpr int kQueue = kMasks + kTokens * 4;
  static constexpr int kEnds = kQueue;
  static constexpr int kRuns = kEnds + kDigits * 16;
  static constexpr int kStart = kQueue;
  static constexpr int kRun = kStart + kDigits * kListAt * 2;
  static constexpr int kBySlot = kQueue;
  static constexpr int kRecs = kQueue + kQueueCap * 2;
  static constexpr int kUcount = kRecs + kQ * (kDigits + 1) * 8;
  static constexpr int kUword = kUcount + kQ * kColors * 4;
  static constexpr int kMword = kUword + kWords * 4;
  static constexpr int kTword = kMword + kWords * 4;
  static constexpr int kPmask = kTword + kWords * 4;
  static constexpr int kQlo = kPmask + kQ * kWords * 4;
  static constexpr int kColor = kQlo + kWords;
  static constexpr int kBytes = kColor + kTokens;
  static_assert(kTab + kDigits * kQ * 8 <= kSlots && kRun + kDigits * kColors <= kRecs &&
                    kRuns + kDigits * 2 <= kRecs && kUcount % 16 == 0 && kUword % 16 == 0 && kPmask % 16 == 0 &&
                    kColor % 16 == 0,
                "the lists' tables and the slots inside the dead masks, the deal's inside the dead queue; each tenant "
                "on its access's alignment");
};

//: A (PASS, DIGIT)'S COLOR COUNT, BIASED: `32 - cap + count` a byte, so bit 5 is set where the digit is full in the pass
//: and the claims' sums order as its counts do; each pass's zero record past its digits for a lane with none left
constexpr uint32_t kBias8 = 32u;
constexpr int kZero = kDigits, kRecStride = kDigits + 1;

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
  __device__ __forceinline__ explicit PhaseClock(long long t) : last(t), at{} {}
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

//: an octet's nonzero bytes, a bit a byte
__device__ __forceinline__ uint32_t nonzero8(uint32_t lo, uint32_t hi) {
  const uint32_t a = ((lo & 0x7f7f7f7fu) + 0x7f7f7f7fu | lo) & 0x80808080u;
  const uint32_t b = ((hi & 0x7f7f7f7fu) + 0x7f7f7f7fu | hi) & 0x80808080u;
  return ((a * 0x00204081u) >> 28 & 0xfu) | ((b * 0x00204081u) >> 24 & 0xf0u);
}

__device__ __forceinline__ uint32_t lanemask_lt() {
  uint32_t m;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
  return m;
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
//: take the least-used color no full digit and the pass's cap `cap` a color forbid (ties to the union's least-used, then
//: `(c - t) & 7`; a token no color allows is blocked and takes the union's least-used), claimed by shared atomics, a
//: claim over a bound withdrawn
struct Claim {
  uint32_t lost, kept;
  bool loser;
};

//: A RECORD'S BYTES TO u16 PAIRS: colors (0, 1) and (2, 3) of a word
__device__ __forceinline__ uint32_t pair_lo(uint32_t x) { return __byte_perm(x, 0u, 0x4140); }
__device__ __forceinline__ uint32_t pair_hi(uint32_t x) { return __byte_perm(x, 0u, 0x4342); }

template <int kQ>
__device__ __forceinline__ Claim claim_batch(uint8_t* smem, uint32_t t, uint32_t q, bool live, uint32_t cap, int lane) {
  using Smem = SmemOf<kQ>;
  const uint32_t* const masks = reinterpret_cast<const uint32_t*>(smem + Smem::kMasks);
  uint32_t* const recs = reinterpret_cast<uint32_t*>(smem + Smem::kRecs) + 2 * kRecStride * q;
  uint32_t* const ucount = reinterpret_cast<uint32_t*>(smem + Smem::kUcount) + kColors * q;
  uint8_t* const color = smem + Smem::kColor;

  const uint32_t m = live ? masks[t] : 0u;
  const int digits = static_cast<int>(__reduce_max_sync(kAll, __popc(m)));
  uint32_t s0 = 0u, s1 = 0u, s2 = 0u, s3 = 0u, f0 = 0u, f1 = 0u;
  uint32_t x = m;
#pragma unroll 2
  for (int i = 0; i < digits; ++i) {
    const int d = x != 0u ? __ffs(x) - 1 : kZero;
    const uint2 v = reinterpret_cast<const uint2*>(recs)[d];
    s0 += pair_lo(v.x);
    s1 += pair_hi(v.x);
    s2 += pair_lo(v.y);
    s3 += pair_hi(v.y);
    f0 |= v.x;
    f1 |= v.y;
    x &= x - 1u;
  }
  const uint4 ua = reinterpret_cast<const uint4*>(ucount)[0], ub = reinterpret_cast<const uint4*>(ucount)[1];
  const uint32_t u[kColors] = {ua.x, ua.y, ua.z, ua.w, ub.x, ub.y, ub.z, ub.w};
  const uint32_t f[kColors] = {s0 & 0xffffu, s0 >> 16, s1 & 0xffffu, s1 >> 16,
                               s2 & 0xffffu, s2 >> 16, s3 & 0xffffu, s3 >> 16};
  const uint32_t nt = (0u - t) & 7u;
  uint32_t best = ~0u;
#pragma unroll
  for (int c = 0; c < kColors; ++c) {
    const bool ok = (((c < 4 ? f0 : f1) >> (8 * (c & 3) + 5)) & 1u) == 0u && u[c] < cap;
    best = ok ? min(best, f[c] << 10 | u[c] << 3 | ((nt + c) & 7u)) : best;
  }
  const bool blocked = live && best == ~0u;
  if (__any_sync(kAll, blocked)) {
    uint32_t b = ~0u;
#pragma unroll
    for (int c = 0; c < kColors; ++c) b = u[c] < cap ? min(b, u[c] << 3 | c) : b;
    best = blocked ? b : best;
  }
  const uint32_t c = blocked ? best & 7u : ((best & 7u) + t) & 7u;
  const uint32_t shift = 8u * (c & 3u), inc = 1u << shift;

  bool over = false;
  if (live) over = atomicAdd(ucount + c, 1u) >= cap;
  x = m;
#pragma unroll 2
  for (int i = 0; i < digits; ++i) {
    uint32_t old = 0u;
    if (x != 0u) old = atomicAdd(recs + 2 * (__ffs(x) - 1) + (c >> 2), inc);
    over |= !blocked && ((old >> (shift + 5u)) & 1u) != 0u;
    x &= x - 1u;
  }
  const bool loser = live && over;
  if (__any_sync(kAll, loser)) {
    if (loser) atomicSub(ucount + c, 1u);
    x = loser ? m : 0u;
#pragma unroll 2
    for (int i = 0; i < digits; ++i) {
      if (x != 0u) atomicSub(recs + 2 * (__ffs(x) - 1) + (c >> 2), inc);
      x &= x - 1u;
    }
  }
  const bool won = live && !over;
  if (won) color[t] = static_cast<uint8_t>(c);
  return Claim{__ballot_sync(kAll, loser), __ballot_sync(kAll, won), loser};
}

//: THE SINGLES OF A PASS `q`, lane `d` digit `d`'s `k`, one a digit a round: each takes the first color in the rotation
//: from `d` with room under its pass capacity and in the pass (else the first with room in the pass), the lanes choosing
//: one color granted in digit order while the pass's room lasts; each digit's color ends posted for the deal
template <int kQ>
__device__ __forceinline__ void singles(uint8_t* smem, int k, uint32_t q, uint32_t cap, int lane) {
  using Smem = SmemOf<kQ>;
  const uint2 cw = reinterpret_cast<const uint2*>(smem + Smem::kRecs)[kRecStride * q + lane];
  const uint4 ua = reinterpret_cast<const uint4*>(smem + Smem::kUcount)[2 * q],
              ub = reinterpret_cast<const uint4*>(smem + Smem::kUcount)[2 * q + 1];
  const uint32_t u[kColors] = {ua.x, ua.y, ua.z, ua.w, ub.x, ub.y, ub.z, ub.w};
  //: the digit's room a color (32 - its biased count, at least 0) and the pass's, octets of bytes
  uint32_t d_lo = 0u, d_hi = 0u, r_lo = 0u, r_hi = 0u;
#pragma unroll
  for (int c = 0; c < kColors; ++c) {
    const uint32_t f = byte_of(cw.x, cw.y, c);
    add_byte(d_lo, d_hi, c, f < kBias8 ? kBias8 - f : 0u);
    add_byte(r_lo, r_hi, c, cap - u[c]);
  }
  const int rot = lane & 7;
  uint32_t t_lo = 0u, t_hi = 0u;
  int left = k;
  for (int go = __any_sync(kAll, left > 0) != 0; go > 0; --go)
#pragma unroll 1
    do {
      //: the colors with room, a bit a color, rotated to start at the lane's
      const uint32_t mine = nonzero8(d_lo, d_hi) & nonzero8(r_lo, r_hi), pass = nonzero8(r_lo, r_hi);
      const uint32_t pick = mine != 0u ? mine : pass;
      const uint32_t rotated = (pick >> rot | pick << (kColors - rot)) & 0xffu;
      const int c = left > 0 ? (__ffs(rotated) - 1 + rot) & 7 : kColors;
      const uint32_t grp = __match_any_sync(kAll, c);
      const bool got = c < kColors && static_cast<uint32_t>(__popc(grp & lanemask_lt())) < byte_of(r_lo, r_hi, c & 7);
      uint32_t g_lo = 0u, g_hi = 0u;
      if (got) {
        add_byte(t_lo, t_hi, c, 1u);
        if (mine != 0u) {
          d_lo -= c < 4 ? 1u << (8 * (c & 3)) : 0u;
          d_hi -= c < 4 ? 0u : 1u << (8 * (c & 3));
        }
        add_byte(g_lo, g_hi, c, 1u);
        --left;
      }
      r_lo -= __reduce_add_sync(kAll, g_lo);
      r_hi -= __reduce_add_sync(kAll, g_hi);
    } while (__any_sync(kAll, left > 0));
  uint32_t end[kColors];
  uint32_t e = 0u;
#pragma unroll
  for (int c = 0; c < kColors; ++c) {
    e += byte_of(t_lo, t_hi, c);
    end[c] = e;
  }
  reinterpret_cast<uint4*>(smem + Smem::kEnds)[lane] =
      make_uint4(end[0] | end[1] << 16, end[2] | end[3] << 16, end[4] | end[5] << 16, end[6] | end[7] << 16);
}

//: THE COLORER: a warp a union, its tile in registers (lane d digit d's 16 words), every table in its own shared memory
template <bool kWrite, int kCell>
__global__ __launch_bounds__(32 * kWarps, kCtasOf<kWrite>) void color(const Params p) {
  constexpr int kQ = kWrite ? 4 : 2;
  using Smem = SmemOf<kQ>;
  extern __shared__ __align__(16) uint8_t smem_all[];
  const int warp = lab::warp_id(), lane = lab::lane_id();
  const int local = static_cast<int>(blockIdx.x) * kWarps + warp;
  if (local >= p.units) return;
  const int u = p.unit0 + local;
  uint8_t* const smem = smem_all + warp * Smem::kBytes;
  uint32_t* const masks = reinterpret_cast<uint32_t*>(smem + Smem::kMasks);
  uint32_t* const uword = reinterpret_cast<uint32_t*>(smem + Smem::kUword);
  uint32_t* const mword = reinterpret_cast<uint32_t*>(smem + Smem::kMword);
  uint32_t* const tword = reinterpret_cast<uint32_t*>(smem + Smem::kTword);
  uint16_t* const queue = reinterpret_cast<uint16_t*>(smem + Smem::kQueue);
  uint8_t* const colv = smem + Smem::kColor;
  uint16_t* const slotv = reinterpret_cast<uint16_t*>(smem + Smem::kSlots);
  uint16_t* const by_slot = reinterpret_cast<uint16_t*>(smem + Smem::kBySlot);
  const uint32_t below = lab::held((1u << lane) - 1u), le = lab::held((2u << lane) - 1u);
  constexpr int kPl = kWrite ? 7 : 8, kPass = 1 << kPl;
  constexpr uint32_t kPassCap = kPass / kColors;
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
  //: THE INPUTS: each token's digits by a transpose of each word and its count's three ballots (the union's words);
  //: the passes -- the union's tokens in token order, `kPass` a pass -- as each word's first token's pass and the
  //: word's tokens of the next pass, and each pass's tokens a word
  uint32_t* const pmask = reinterpret_cast<uint32_t*>(smem + Smem::kPmask);
  uint8_t* const qlov = smem + Smem::kQlo;
  if (lane < kQ * kWords / 2) reinterpret_cast<uint2*>(pmask)[lane] = make_uint2(0u, 0u);
  uint32_t ur = 0u;
  __syncwarp();
#pragma unroll
  for (int j = 0; j < kWords; ++j) {
    const uint32_t m = transpose(w[j], lane);
    masks[32 * j + lane] = m;
    const int c = __popc(m);
    const uint32_t uw = __ballot_sync(kAll, c >= 1), mw = __ballot_sync(kAll, c >= 2),
                   tw = __ballot_sync(kAll, c >= 3);
    const uint32_t qlo = ur >> kPl;
    const uint32_t hi = uw & __ballot_sync(kAll, ((ur + __popc(uw & below)) >> kPl) != qlo);
    if (lane == j) {
      uword[j] = uw;
      mword[j] = mw;
      tword[j] = tw;
      qlov[j] = static_cast<uint8_t>(qlo);
      pmask[qlo * kWords + j] = uw & ~hi;
      if (hi != 0u) pmask[(qlo + 1) * kWords + j] = hi;
    }
    ur += __popc(uw);
  }
  const int passes = static_cast<int>((ur + kPass - 1) >> kPl);
  __syncwarp();

  const uint4* const ends = reinterpret_cast<const uint4*>(smem + Smem::kEnds);
  clk.mark(1);
  //: THE PASSES' INPUTS: lane d its digit's tokens and singles in each pass and its biased counts at the pass's
  //: capacity; each pass's queue, its multi-digit tokens three-or-more first, each kind in token order, at the pass's
  //: segment of the queue
  uint32_t sg01 = 0u, sg23 = 0u;
  {
    const uint4* const uw4 = reinterpret_cast<const uint4*>(uword);
    const uint4* const mw4 = reinterpret_cast<const uint4*>(mword);
#pragma unroll
    for (int q = 0; q < kQ; ++q) {
      int n = 0, k = 0;
      const uint4* const pm4 = reinterpret_cast<const uint4*>(pmask + q * kWords);
#pragma unroll
      for (int i = 0; i < kWords / 4; ++i) {
        const uint4 pm = pm4[i], uw = uw4[i], mw = mw4[i];
        const uint32_t pmv[4] = {pm.x, pm.y, pm.z, pm.w}, sgv[4] = {uw.x & ~mw.x, uw.y & ~mw.y, uw.z & ~mw.z, uw.w & ~mw.w};
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          n += __popc(w[4 * i + r] & pmv[r]);
          k += __popc(w[4 * i + r] & pmv[r] & sgv[r]);
        }
      }
      const uint32_t r = (kBias8 - static_cast<uint32_t>(n + kColors - 1) / kColors) * 0x01010101u;
      reinterpret_cast<uint2*>(smem + Smem::kRecs)[kRecStride * q + lane] = make_uint2(r, r);
      if (q < 2) sg01 |= static_cast<uint32_t>(k) << (16 * q);
      else sg23 |= static_cast<uint32_t>(k) << (16 * (q - 2));
    }
    if (lane < kQ) reinterpret_cast<uint2*>(smem + Smem::kRecs)[kRecStride * lane + kZero] = make_uint2(0u, 0u);
    if (lane < kQ * kColors) reinterpret_cast<uint32_t*>(smem + Smem::kUcount)[lane] = 0u;
  }
  uint32_t tl[kQ] = {};
#pragma unroll 1
  for (int q = 0; q < passes; ++q) {
    const int i = lane & 15;
    const uint32_t pm = pmask[q * kWords + i], tw = tword[i] & pm, mw = mword[i] & ~tword[i] & pm;
    const uint32_t cnt = __popc(tw) | __popc(mw) << 16;
    uint32_t incl = cnt;
#pragma unroll
    for (int off = 1; off < kWords; off <<= 1) {
      const uint32_t y = __shfl_up_sync(kAll, incl, off, kWords);
      incl += i >= off ? y : 0u;
    }
    const uint32_t all = __shfl_sync(kAll, incl, kWords - 1), threes = all & 0xffffu;
    const uint32_t ex = incl - cnt;
#pragma unroll
    for (int p = 0; p < kQ; ++p) tl[p] = p == q ? threes + (all >> 16) : tl[p];
    uint16_t* const seg = queue + (q << kPl);
    const uint32_t live = __ballot_sync(kAll, lane < kWords && pm != 0u);
    int j = __ffs(live) - 1;
    const int j1 = 31 - __clz(live | 1u);
    for (int go = __any_sync(kAll, j <= j1) != 0; go > 0; --go)
#pragma unroll 1
    do {
      const uint32_t before = __shfl_sync(kAll, ex, j);
      const uint32_t pmj = pmask[q * kWords + j], twj = tword[j] & pmj, mwj = mword[j] & ~tword[j] & pmj;
      if ((twj >> lane) & 1u) seg[(before & 0xffffu) + __popc(twj & below)] = static_cast<uint16_t>(32 * j + lane);
      if ((mwj >> lane) & 1u) seg[threes + (before >> 16) + __popc(mwj & below)] = static_cast<uint16_t>(32 * j + lane);
      ++j;
    } while (__any_sync(kAll, j <= j1));
  }
  __syncwarp();

  clk.mark(2);
  //: THE CLAIMS, every pass's at once: each batch shares its 32 lanes among the passes with tokens queued (a pass a run
  //: of lanes, at most its share), so a pass's tokens meet few of their own; losers requeued at their pass's tail (a ring
  //: of a pass); a batch that keeps nothing followed by one token alone (which always keeps)
  {
    constexpr uint32_t kRing = kPass - 1;
    uint32_t hd[kQ] = {};
    bool solo = false;
    const uint32_t ul = static_cast<uint32_t>(lane);
    uint32_t any = 0u;
#pragma unroll
    for (int p = 0; p < kQ; ++p) any |= tl[p];
    bool pending = true;
    for (int go = __any_sync(kAll, any) != 0; go > 0; --go)
#pragma unroll 1
      do {
        uint32_t active = 0u;
#pragma unroll
        for (int p = 0; p < kQ; ++p) active += tl[p] != hd[p] ? 1u : 0u;
        //: a pass's share of the lanes: 32, 16, 11 or 8 for 1-4 passes queued
        const uint32_t share = solo ? 1u : (0x080b1020u >> (8 * (active - 1u))) & 0xffu;
        uint32_t edge[kQ + 1], kq[kQ], taken = 0u;
        edge[0] = 0u;
#pragma unroll
        for (int p = 0; p < kQ; ++p) {
          kq[p] = solo && taken != 0u ? 0u : min(min(tl[p] - hd[p], share), 32u - taken);
          taken += kq[p];
          edge[p + 1] = taken;
        }
        uint32_t q = 0u;
#pragma unroll
        for (int p = 1; p < kQ; ++p) q += ul >= edge[p] ? 1u : 0u;
        const bool live = ul < edge[kQ];
        uint32_t start = edge[0], head = hd[0], tail = tl[0];
#pragma unroll
        for (int p = 1; p < kQ; ++p) {
          start = q == static_cast<uint32_t>(p) ? edge[p] : start;
          head = q == static_cast<uint32_t>(p) ? hd[p] : head;
          tail = q == static_cast<uint32_t>(p) ? tl[p] : tail;
        }
        const uint32_t t = live ? queue[(q << kPl) + ((head + ul - start) & kRing)] : 0u;
        const Claim cl = claim_batch<kQ>(smem, t, q, live, kPassCap, lane);
        __syncwarp();
        if (cl.loser) {
          const uint32_t mine = cl.lost & below & ~((1u << start) - 1u);
          queue[(q << kPl) + ((tail + __popc(mine)) & kRing)] = static_cast<uint16_t>(t);
        }
        const uint32_t lost = __reduce_add_sync(kAll, cl.loser ? 1u << (8 * q) : 0u);
#pragma unroll
        for (int p = 0; p < kQ; ++p) {
          tl[p] += (lost >> (8 * p)) & 0xffu;
          hd[p] += kq[p];
        }
        solo = cl.kept == 0u;
        __syncwarp();
        pending = false;
#pragma unroll
        for (int p = 0; p < kQ; ++p) pending |= tl[p] != hd[p];
      } while (__any_sync(kAll, pending));
  }

  clk.mark(3);
  //: THE SINGLES AND THE DEAL, pass by pass
#pragma unroll 1
  for (int q = 0; q < passes; ++q) {
    const int k = static_cast<int>(((q < 2 ? sg01 : sg23) >> (16 * (q & 1))) & 0xffffu);
    singles<kQ>(smem, k, static_cast<uint32_t>(q), kPassCap, lane);
    __syncwarp();
    //: the deal over the pass's words: a single's color by its index among its digit's singles of the pass (those
    //: before it in the word by a match on its digit, the earlier words' kept a digit) against the digit's color ends
    {
      uint16_t* const runs = reinterpret_cast<uint16_t*>(smem + Smem::kRuns);
      if (lane < kDigits / 2) reinterpret_cast<uint32_t*>(runs)[lane] = 0u;
      const int i = lane & 15;
      const uint32_t live = __ballot_sync(kAll, lane < kWords && pmask[q * kWords + i] != 0u);
      int j = __ffs(live) - 1;
      const int j1 = 31 - __clz(live | 1u);
      __syncwarp();
      for (int go = __any_sync(kAll, j <= j1) != 0; go > 0; --go)
#pragma unroll 1
      do {
        const uint32_t sgw = pmask[q * kWords + j] & uword[j] & ~mword[j];
        const bool sg = (sgw >> lane) & 1u;
        const int t = 32 * j + lane;
        const uint32_t m = sg ? masks[t] : 0u;
        const uint32_t grp = __match_any_sync(kAll, m);
        const int dd = sg ? __ffs(m) - 1 : 0;
        const uint32_t idx = runs[dd] + __popc(grp & below);
        const uint4 e = ends[dd];
        const uint32_t k1 = (idx + 1u) * 0x00010001u;
        const uint32_t dealt = __popc(~((e.x | 0x80008000u) - k1) & 0x80008000u) +
                               __popc(~((e.y | 0x80008000u) - k1) & 0x80008000u) +
                               __popc(~((e.z | 0x80008000u) - k1) & 0x80008000u) +
                               __popc(~((e.w | 0x80008000u) - k1) & 0x00008000u);
        __syncwarp();
        if (sg) {
          colv[t] = static_cast<uint8_t>(dealt);
          if ((grp & below) == 0u) runs[dd] = static_cast<uint16_t>(idx + __popc(grp));
        }
        __syncwarp();
        ++j;
      } while (__any_sync(kAll, j <= j1));
    }
    __syncwarp();
  }
  {
    const uint4 gap = make_uint4(0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u);
    reinterpret_cast<uint4*>(by_slot)[lane] = gap;
    reinterpret_cast<uint4*>(by_slot)[lane + 32] = gap;
  }
  __syncwarp();

  clk.mark(4);
  //: THE SLOTS, word by word: each token's slot, its pass's first slot + 8 x its color's tokens before it in the pass +
  //: its color; lane c keeps its color's tokens in the current pass
  uint32_t base = 0u, cur = 0u;
  const uint32_t own0 = lab::held(0u - (lane & 1u)), own1 = lab::held(0u - ((lane >> 1) & 1u)),
                 own2 = lab::held(0u - ((lane >> 2) & 1u));
#pragma unroll
  for (int j = 0; j < kWords; ++j) {
    const uint32_t uw = uword[j], qlo = qlov[j];
    const uint32_t hi = qlo + 1u < static_cast<uint32_t>(kQ) ? pmask[(qlo + 1u) * kWords + j] : 0u;
    const bool turn = uw != 0u && qlo != cur;
    base = turn ? 0u : base;
    cur = turn ? qlo : cur;
    const bool in = (uw >> lane) & 1u, up = (hi >> lane) & 1u;
    const int t = 32 * j + lane;
    const uint32_t c = in ? colv[t] & 7u : 0u;
    const uint32_t v0 = __ballot_sync(kAll, c & 1u), v1 = __ballot_sync(kAll, c & 2u), v2 = __ballot_sync(kAll, c & 4u);
    const uint32_t m0 = 0u - (c & 1u), m1 = 0u - ((c >> 1) & 1u), m2 = 0u - ((c >> 2) & 1u);
    const uint32_t same = uw & ~(v0 ^ m0) & ~(v1 ^ m1) & ~(v2 ^ m2);
    const uint32_t mine = uw & ~(v0 ^ own0) & ~(v1 ^ own1) & ~(v2 ^ own2);
    const uint32_t b = __shfl_sync(kAll, base, static_cast<int>(c));
    const uint32_t rank = up ? __popc(same & hi & below) : b + __popc(same & ~hi & below);
    const uint32_t s = ((qlo + (up ? 1u : 0u)) << kPl) + kColors * rank + c;
    if (in) {
      slotv[t] = static_cast<uint16_t>(s);
      by_slot[s] = static_cast<uint16_t>(t);
    }
    base = hi != 0u ? __popc(mine & hi) : base + __popc(mine);
    cur = hi != 0u ? qlo + 1u : cur;
  }
  const int extent =
      passes == 0 ? 0 : ((passes - 1) << kPl) + kColors * static_cast<int>(__reduce_max_sync(kAll, base));
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
  uint16_t* const startv = reinterpret_cast<uint16_t*>(smem + Smem::kStart);
  uint8_t* const runv = smem + Smem::kRun;
  reinterpret_cast<uint4*>(tab)[lane] = make_uint4(0u, 0u, 0u, 0u);
  reinterpret_cast<uint4*>(tab)[lane + 32] = make_uint4(0u, 0u, 0u, 0u);
  reinterpret_cast<uint2*>(runv)[lane] = make_uint2(0u, 0u);
  const int count = static_cast<int>(e1 - e0);
  const int chunks = __shfl_sync(kAll, (count + 31) >> 5, 0);
  __syncwarp();
  uint32_t ahead = lane < count ? toks[lane] : 0u;
#pragma unroll 1
  for (int i = 0; i < chunks; ++i) {
    const int e = 32 * i + lane;
    const uint32_t tok = ahead;
    ahead = e + 32 < count ? toks[e + 32] : 0u;
    if (e < count) {
      const uint32_t s = slotv[tok & 511u];
      atomicAdd(tab + ((tok >> 9) * kQ + (s >> pl)) * 2 + ((s >> 2) & 1u), 1u << (8 * (s & 3u)));
    }
  }
  __syncwarp();

  //: THE STARTS, lane d list d's passes: each its steps (its largest color count), its start and its colors' tokens in
  //: the earlier passes (over the counts); the list's starts out
  {
    uint2 tq[kQ];
#pragma unroll
    for (int q = 0; q < kQ; ++q) tq[q] = reinterpret_cast<const uint2*>(tab)[lane * kQ + q];
    __syncwarp();
    uint32_t b_lo = 0u, b_hi = 0u;
    int start = 0;
    uint32_t at[kListAt];
#pragma unroll
    for (int q = 0; q < kQ; ++q) {
      const uint32_t m4 = __vmaxu4(tq[q].x, tq[q].y);
      const int steps = static_cast<int>(max(max(m4 & 0xffu, (m4 >> 8) & 0xffu), max((m4 >> 16) & 0xffu, m4 >> 24)));
      reinterpret_cast<uint2*>(tab)[lane * kQ + q] = make_uint2(b_lo, b_hi);
      at[q] = static_cast<uint32_t>(start);
      b_lo += tq[q].x;
      b_hi += tq[q].y;
      start += kColors * steps;
    }
#pragma unroll
    for (int q = kQ; q < kListAt; ++q) at[q] = static_cast<uint32_t>(start);
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
  uint32_t tok_next = lane < count ? toks[lane] : 0u, coef_next = lane < count ? coefs[lane] : 0u;
#pragma unroll 1
  for (int i = 0; i < chunks; ++i) {
    const int e = 32 * i + lane;
    const bool live = e < count;
    const uint32_t tok = tok_next, coef = coef_next;
    tok_next = e + 32 < count ? toks[e + 32] : 0u;
    coef_next = e + 32 < count ? coefs[e + 32] : 0u;
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
    const uint32_t before = smem[Smem::kTab + (d * kQ + q) * 8 + c];
    const int pos = startv[d * kListAt + q] + kColors * static_cast<int>(rank - before) + c;
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
  color<kWrite, kCell><<<(p.units + kWarps - 1) / kWarps, 32 * kWarps, kWarps * SmemOf<kWrite ? 4 : 2>::kBytes, st>>>(p);
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
    int slot_of[kTokens];
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
    //: the passes: pass q the union's tokens of union rank q P .. q P + P - 1, each color's tokens in it dense from the
    //: pass's first slot in token order
    int rank_of[kTokens], placed = 0, last_widest = 0;
    for (int t = 0, r = 0; t < kTokens; ++t) rank_of[t] = mask[t] ? r++ : -1;
    const int npass = (members + pass - 1) / pass;
    for (int q = 0; q < npass; ++q) {
      int widest = 0;
      for (int col = 0; col < kColors; ++col) {
        int last = -1, per = 0;
        bool gap = false;
        for (int r = 0; r < pass / kColors; ++r) {
          const uint16_t t = by[q * pass + kColors * r + col];
          if (t == carry::kGap) {
            gap = true;
            continue;
          }
          if (gap || static_cast<int>(t) <= last) fail(u, "color " + std::to_string(col) + " not dense in token order");
          if (t < kTokens && rank_of[t] / pass != q) fail(u, "a token outside its pass");
          last = t;
          ++per;
        }
        widest = std::max(widest, per);
        placed += per;
      }
      last_widest = widest;
    }
    if (placed != members) fail(u, "the union is not placed");
    const int want_extent = npass == 0 ? 0 : (npass - 1) * pass + kColors * last_widest;
    if (count[u] != want_extent) fail(u, "the extent is not the last pass's first slot + 8 x its widest color");
    v.gaps += static_cast<size_t>(count[u] - members);
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
      for (int q = 0; q < npass; ++q) {
        int nq = 0, cols[kColors] = {};
        for (const auto& [t, coef] : mine[d])
          if (slot_of[t] / pass == q) {
            ++nq;
            ++cols[slot_of[t] & 7];
          }
        const int cap = (nq + kColors - 1) / kColors;
        for (int col = 0; col < kColors; ++col) v.over += static_cast<size_t>(std::max(0, cols[col] - cap));
      }

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
  if (c.row != kRow || c.write_pass != 128 ||
      c.read_pass != 256) {
    std::fprintf(stderr, "facts_color: the row is not %d or a pass not the sm_86 even arm's\n", kRow);
    std::exit(1);
  }
  LAB_CHECK(cudaFuncSetAttribute(color<true, kCell>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 kWarps * SmemOf<4>::kBytes));
  LAB_CHECK(cudaFuncSetAttribute(color<false, kCell>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 kWarps * SmemOf<2>::kBytes));
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
    static const char* kPhaseName[kTimed] = {"tile", "inputs", "-", "-", "passes", "slots", "union out",
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
  lab::Samples places;
  places.v.push_back(double(v.entries) / double(c.offsets.back()));
  b.record("list places, x the live entries / " + cell, "ratio", places);
  b.unchecked("list places, x the live entries / " + cell, "a count of the checked outputs");
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
