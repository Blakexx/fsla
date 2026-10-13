// THE FACTS PASS'S TRANSLATION UNIT -- the kernel that reads the bundle's rows once, a CTA a unit (a window's side,
// level and 32-digit group): both sides' class-1 words and tile unions, the window's columns, and each side's lists
// colored and placed in the carry's final form; and the host entry that validates the bundle's views, plans a
// window's units and launches them.
// see docs/internals/facts/facts.md

#include "common/torch_seam.cuh"

#include <algorithm>
#include <climits>
#include <optional>
#include <vector>

#include "common/arch_runtime.cuh"
#include "common/ops.cuh"
#include "common/phase_clock.cuh"
#include "facts/facts.cuh"
#include "facts/facts_api.cuh"

namespace rola {
namespace facts {
namespace detail {

namespace ops = ::denseref::ops;
using Clock = PhaseClock<kFactsPhases, true>;

//: A WINDOW BYTE: token `t` of its 32-token block at row `4 (t % 8) + t / 8` of the block's 1 KB, so the four tokens a
//: vote lane group reads together sit in four consecutive rows; the group's digit `d` its byte in the row.
//: -- see docs/internals/facts/facts.md#load
__device__ __forceinline__ int ring_row(int t) { return 4 * (t & 7) + (t >> 3); }

__device__ __forceinline__ int window_at(int t, int d) {
  return (t >> 5) * 1024 + ring_row(t & 31) * 32 + d;
}

//: A BYTE PERMUTE WITH ITS SELECTOR IMMEDIATE.
template <uint32_t Sel>
__device__ __forceinline__ uint32_t prmt_imm(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "n"(Sel));
  return d;
}

//: bit 7 of each byte set iff the byte is nonzero (three operations for four digits)
__device__ __forceinline__ uint32_t nonzero_bytes(uint32_t x) {
  return (((x & 0x7f7f7f7fu) + 0x7f7f7f7fu) | x) & 0x80808080u;
}

//: THE LIST EXPONENT: the smallest `K` with `x <= 2^K`, an `x` below the smallest normal at its exponent.
__device__ __forceinline__ int ceil_exponent(float x) {
  const uint32_t bits = __float_as_uint(fmaxf(x, kListFloor));
  return static_cast<int>(bits >> 23) - 127 + ((bits & 0x7fffffu) != 0u);
}

//: 2^e as fp32, `e` in [-126, 127]
__device__ __forceinline__ float pow2(int e) {
  return __uint_as_float(static_cast<uint32_t>(127 + e) << 23);
}

//: THE GAINS' POWER OF TWO for a window's largest `top`: `-4 - floor(log2 top)`, clamped to [-126, 127], 0 for a zero
//: largest. -- see docs/internals/facts/facts.md#gains
__device__ __forceinline__ int gains_shift(float top) {
  if (!(top > 0.0f)) return 0;
  return max(-126, min(127, -(static_cast<int>(__float_as_uint(top) >> 23) - 127) - 4));
}

//: BYTE `c` of a packed octet (bytes 0-3 in `lo`, 4-7 in `hi`), and the octet with `v` added there.
__device__ __forceinline__ uint32_t byte_of(uint32_t lo, uint32_t hi, int c) {
  return ((c < 4 ? lo : hi) >> (8 * (c & 3))) & 0xffu;
}

__device__ __forceinline__ void add_byte(uint32_t& lo, uint32_t& hi, int c, uint32_t v) {
  const uint32_t x = v << (8 * (c & 3));
  lo += c < 4 ? x : 0u;
  hi += c < 4 ? 0u : x;
}

//: A 32 x 32 BIT TRANSPOSE across the warp, lane d's bit i to lane i's bit d: a stage of stride s swaps the
//: off-diagonal blocks of lanes d and d ^ s, each lane keeping its half and sending the other rotated into place.
__device__ __forceinline__ uint32_t transpose(uint32_t x, int lane) {
  constexpr uint32_t kHalf[5] = {0x0000ffffu, 0x00ff00ffu, 0x0f0f0f0fu, 0x33333333u, 0x55555555u};
#pragma unroll
  for (int k = 0; k < 5; ++k) {
    const int s = 16 >> k;
    const uint32_t keep = (lane & s) ? ~kHalf[k] : kHalf[k];
    const uint32_t send = __funnelshift_l(x, x, (lane & s) ? s : 32 - s) & keep;
    x = (x & keep) | __shfl_xor_sync(0xffffffffu, send, s);
  }
  return x;
}

//: A UNIT'S PLACE: its code's fields and its (batch-head, window). -- see docs/internals/facts/facts.md#the-grid
struct Unit {
  int kind, side, level, group, b, h, bh, window;
};

__device__ __forceinline__ Unit unit_of(const FactsParams& p, int u) {
  const int r = u % p.per_window, rest = u / p.per_window;
  const uint32_t code = p.unit[r];
  Unit x;
  x.kind = unit_kind(code);
  x.side = unit_side(code);
  x.level = unit_level(code);
  x.group = unit_group(code);
  x.window = rest % p.windows;
  x.bh = rest / p.windows;
  x.b = x.bh / p.H;
  x.h = x.bh % p.H;
  return x;
}

__device__ __forceinline__ bool digit_in(const FactsParams& p, int side, int level, int digit) {
  const int row = p.row_base[level] + digit;
  return digit < p.B[level] && mask_bit(p.mask[side][row >> 5], row);
}

//: THE WINDOW'S COLUMN of token `i`: `m` = the gain times the write scales in level order and `F = m s`, a token past
//: `L` zero. -- see docs/internals/facts/facts.md#columns
__device__ __forceinline__ float2 load_column(const FactsParams& p, const Unit& x, int i) {
  const int t = x.window * kWindowTokens + i;
  if (t >= p.L) return make_float2(0.0f, 0.0f);
  float m = __ldg(p.g_write.base + x.b * p.g_write.sb + x.h * p.g_write.sh + t * p.g_write.st);
#pragma unroll
  for (int l = 0; l < kMaxLevels; ++l)
    if (l < p.D)
      m *= __ldg(p.scale_write[l].base + x.b * p.scale_write[l].sb + x.h * p.scale_write[l].sh
                 + t * p.scale_write[l].st);
  const float s =
      __ldg(p.v_scale.base + x.b * p.v_scale.sb + x.h * p.v_scale.sh + t * p.v_scale.st);
  return make_float2(m, m * s);
}

//: THE WINDOW'S COPIES: token `t`'s 32 bytes of the group, two 16-byte quanta, a lane pair a token, past the rows'
//: width or `L` zero-filled. -- see docs/internals/facts/facts.md#load
__device__ __forceinline__ void issue_window(const FactsParams& p, const Unit& x, ops::SmemAddr win,
                                             int tid) {
  const RowView v = p.rows[x.side][x.level];
  const int digit0 = kGroupDigits * x.group;
  const uint8_t* const rows = v.base + x.b * v.sb + x.h * v.sh + digit0;
  constexpr int kCopies = kWindowTokens * 2 / kFactsThreads;
#pragma unroll
  for (int k = 0; k < kCopies; ++k) {
    const int q = tid + kFactsThreads * k, i = q >> 1, part = q & 1;
    const int t = x.window * kWindowTokens + i;
    const uint32_t run =
        t < p.L ? static_cast<uint32_t>(max(0, min(16, v.width - digit0 - 16 * part))) : 0u;
    ops::stage_tile_n(win + window_at(i, 16 * part),
                      rows + static_cast<int64_t>(t) * v.st + 16 * part, run);
  }
}

//: THE COLUMNS OUT, the window's one unit's: each token's `(F, m)` and its gains pair at the window's powers of two,
//: from the warps' posted maxima. -- see docs/internals/facts/facts.md#gains
__device__ __forceinline__ void columns_out(const FactsParams& p, const Unit& x,
                                            const float2 (&col)[2], const float2* wmax, int tid) {
  float2 top = wmax[0];
#pragma unroll
  for (int w = 1; w < kFactsWarps; ++w)
    top = make_float2(fmaxf(top.x, wmax[w].x), fmaxf(top.y, wmax[w].y));
  const int sm = gains_shift(top.x), sf = gains_shift(top.y);
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int t = x.window * kWindowTokens + tid + kFactsThreads * k;
    if (t < p.L) {
      const int64_t at = static_cast<int64_t>(x.bh) * p.L + t;
      p.gains16[at] =
          static_cast<uint32_t>(__half_as_ushort(__float2half_rn(col[k].x * pow2(sm))))
          | static_cast<uint32_t>(__half_as_ushort(__float2half_rn(col[k].y * pow2(sf)))) << 16;
      p.fm[at] = make_float2(col[k].y, col[k].x);
    }
  }
  if (tid < 2)
    p.gains_exp[(static_cast<int64_t>(x.bh) * p.windows + x.window) * 2 + tid] =
        static_cast<int16_t>(tid ? sf : sm);
}

//: THE VOTE, a warp's two blocks: lane `(tk, dq)` token `8 tk + i`'s digits `4 dq ..` as a word, each byte's nonzero
//: bit at bit `i`; a 4 x 4 byte transpose across the token blocks leaves digit `4 dq + tk`'s 32 token bits, masked.
//: -- see docs/internals/facts/facts.md#vote
__device__ __forceinline__ void vote(const FactsParams& p, const Unit& x, const uint8_t* win,
                                     uint32_t* tile, int warp, int lane) {
  const int tk = lane >> 3, dq = lane & 7, d = 4 * dq + tk;
  const bool in = digit_in(p, x.side, x.level, kGroupDigits * x.group + d);
#pragma unroll
  for (int jj = 0; jj < 2; ++jj) {
    const int j = 2 * warp + jj;
    const uint8_t* const region = win + j * 1024;
    uint32_t flags = 0u;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      flags |=
          nonzero_bytes(*reinterpret_cast<const uint32_t*>(region + (4 * i + tk) * 32 + 4 * dq))
          >> (7 - i);
    uint32_t y = __shfl_xor_sync(0xffffffffu, flags, 16);
    flags = (tk & 2) ? prmt_imm<0x3276u>(flags, y) : prmt_imm<0x5410u>(flags, y);
    y = __shfl_xor_sync(0xffffffffu, flags, 8);
    flags = (tk & 1) ? prmt_imm<0x3715u>(flags, y) : prmt_imm<0x6240u>(flags, y);
    tile[tile_at(d, j)] = in ? flags : 0u;
  }
}

//: THE BITS OUT: each digit's window of words one 64-byte run (a half-warp a digit); warp `w` tile `w`'s union, a
//: digit a lane, one ballot. -- see docs/internals/facts/facts.md#tiles
__device__ __forceinline__ void bits_out(const FactsParams& p, const Unit& x, const uint32_t* tile,
                                         int tid, int warp, int lane) {
  const int word0 = x.window * kWindowWords, nwords = min(kWindowWords, p.token_words - word0);
  const int digit0 = kGroupDigits * x.group;
  uint32_t* const out = p.words
                        + ((static_cast<int64_t>(x.bh) * kSides + x.side) * p.rows_total
                           + p.row_base[x.level] + digit0)
                              * p.token_words
                        + word0;
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int i = tid + kFactsThreads * k, d = i >> 4, j = i & 15;
    if (digit0 + d < p.B[x.level] && j < nwords)
      out[static_cast<int64_t>(d) * p.token_words + j] = tile[tile_at(d, j)];
  }
  const uint32_t any = __ballot_sync(
      0xffffffffu, (tile[tile_at(lane, 2 * warp)] | tile[tile_at(lane, 2 * warp + 1)]) != 0u);
  const int tile_index = x.window * kWindowTiles + warp;
  if (lane == 0 && tile_index < p.tiles)
    p.unions[((static_cast<int64_t>(x.bh) * kSides + x.side) * p.groups + p.group_base[x.level]
              + x.group)
                 * p.tiles
             + tile_index] = any;
}

//: THE COLORING'S INPUTS, a warp's two words and four digits: each token's digits by a transpose and its count's three
//: ballots (the union's words: in one digit or more, two or more, three or more); each digit's capacity; the claim
//: state and the union by slot cleared. -- see docs/internals/facts/facts.md#coloring
__device__ __forceinline__ void prepare(uint8_t* smem, const uint32_t* tile, int tid, int warp,
                                        int lane) {
  uint32_t* const masks = reinterpret_cast<uint32_t*>(smem + FactsSmem::kMasks);
  uint32_t* const uword = reinterpret_cast<uint32_t*>(smem + FactsSmem::kUword);
  uint32_t* const mword = reinterpret_cast<uint32_t*>(smem + FactsSmem::kMword);
  uint32_t* const tword = reinterpret_cast<uint32_t*>(smem + FactsSmem::kTword);
  uint32_t* const cap = reinterpret_cast<uint32_t*>(smem + FactsSmem::kCap);
  uint4* const counts = reinterpret_cast<uint4*>(smem + FactsSmem::kCounts);
#pragma unroll
  for (int jj = 0; jj < 2; ++jj) {
    const int j = 2 * warp + jj;
    const uint32_t m = transpose(tile[tile_at(lane, j)], lane);
    masks[kTokBits * j + lane] = m;
    const int c = __popc(m);
    const uint32_t u = __ballot_sync(0xffffffffu, c >= 1), mw = __ballot_sync(0xffffffffu, c >= 2),
                   tw = __ballot_sync(0xffffffffu, c >= 3);
    if (lane == 0) {
      uword[j] = u;
      mword[j] = mw;
      tword[j] = tw;
    }
  }
  const int h = lane >> 4, jl = lane & 15;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int d = 4 * warp + 2 * r + h;
    int n = __popc(tile[tile_at(d, jl)]);
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) n += __shfl_xor_sync(0xffffffffu, n, off);
    if (jl == 0) {
      cap[d] = static_cast<uint32_t>(n + kColors - 1) / kColors;
      counts[d] = make_uint4(0u, 0u, 0u, 0u);
    }
  }
  if (tid < kColors) reinterpret_cast<uint32_t*>(smem + FactsSmem::kUcount)[tid] = 0u;
  if (tid < 2) reinterpret_cast<uint32_t*>(smem + FactsSmem::kAppends)[tid] = 0u;
  reinterpret_cast<uint32_t*>(smem + FactsSmem::kUnion)[tid] =
      static_cast<uint32_t>(carry::kGap) * 0x00010001u;
}

//: EACH DIGIT'S SINGLES BEFORE EACH WORD, warps 1-4 (8 digits a warp, 4 lanes a digit, 4 words a lane): the deal's
//: index base.
__device__ __forceinline__ void singles_before(uint8_t* smem, const uint32_t* tile, int warp,
                                               int lane) {
  const uint32_t* const uword = reinterpret_cast<const uint32_t*>(smem + FactsSmem::kUword);
  const uint32_t* const mword = reinterpret_cast<const uint32_t*>(smem + FactsSmem::kMword);
  uint16_t* const pre = reinterpret_cast<uint16_t*>(smem + FactsSmem::kPre);
  const int d = 8 * (warp - 1) + (lane >> 2), part = lane & 3;
  int n[4], sum = 0;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int j = 4 * part + q;
    n[q] = __popc(tile[tile_at(d, j)] & uword[j] & ~mword[j]);
    sum += n[q];
  }
  int incl = sum;
#pragma unroll
  for (int off = 1; off < 4; off <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, incl, off, 4);
    incl += part >= off ? y : 0;
  }
  int at = incl - sum;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    pre[d * kWindowWords + 4 * part + q] = static_cast<uint16_t>(at);
    at += n[q];
  }
}

//: THE QUEUE, a warp's two words: the multi-digit tokens, three-or-more first, each in token order -- a token's place
//: its kind's tokens in the earlier words plus its rank in its word; thread 0 the queue's length.
__device__ __forceinline__ void enqueue(uint8_t* smem, int tid, int warp, int lane) {
  const uint32_t* const mword = reinterpret_cast<const uint32_t*>(smem + FactsSmem::kMword);
  const uint32_t* const tword = reinterpret_cast<const uint32_t*>(smem + FactsSmem::kTword);
  uint16_t* const queue = reinterpret_cast<uint16_t*>(smem + FactsSmem::kQueue);
  const uint32_t below = (1u << lane) - 1u;
  int three = 0, two[2] = {0, 0}, three_at[2] = {0, 0};
#pragma unroll
  for (int i = 0; i < kWindowWords; ++i) {
    const int t3 = __popc(tword[i]), t2 = __popc(mword[i] & ~tword[i]);
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      three_at[jj] += i < 2 * warp + jj ? t3 : 0;
      two[jj] += i < 2 * warp + jj ? t2 : 0;
    }
    three += t3;
  }
#pragma unroll
  for (int jj = 0; jj < 2; ++jj) {
    const int j = 2 * warp + jj;
    const uint32_t tw = tword[j], mw = mword[j] & ~tw;
    if ((tw >> lane) & 1u)
      queue[three_at[jj] + __popc(tw & below)] = static_cast<uint16_t>(kTokBits * j + lane);
    if ((mw >> lane) & 1u)
      queue[three + two[jj] + __popc(mw & below)] = static_cast<uint16_t>(kTokBits * j + lane);
  }
  if (tid == 0) {
    int multi = 0;
#pragma unroll
    for (int i = 0; i < kWindowWords; ++i) multi += __popc(mword[i]);
    reinterpret_cast<int*>(smem + FactsSmem::kMisc)[1] = multi;
  }
}

//: ONE BATCH OF CLAIMS by a warp, the `take` queued tokens from `head` (lane l the l-th): each sums its digits' color
//: counts and takes the least-used color its digits' capacities (full where a count reached it) and the union's cap
//: allow (ties to the union's least-used, then `(c - t) & 7`; a token no color allows is blocked and takes the union's
//: least-used), claimed on the union and every digit by shared atomics, a claim over a bound withdrawn.
//: -- see docs/internals/facts/facts.md#coloring
struct Claim {
  uint32_t t, lost, kept;
  bool loser;
};

__device__ __forceinline__ Claim claim_batch(uint8_t* smem, uint32_t head, uint32_t take,
                                             int lane) {
  const uint32_t* const masks = reinterpret_cast<const uint32_t*>(smem + FactsSmem::kMasks);
  const uint16_t* const queue = reinterpret_cast<const uint16_t*>(smem + FactsSmem::kQueue);
  uint32_t* const counts = reinterpret_cast<uint32_t*>(smem + FactsSmem::kCounts);
  const uint32_t* const cap = reinterpret_cast<const uint32_t*>(smem + FactsSmem::kCap);
  uint32_t* const ucount = reinterpret_cast<uint32_t*>(smem + FactsSmem::kUcount);
  uint16_t* const color = reinterpret_cast<uint16_t*>(smem + FactsSmem::kSlot);

  const bool live = static_cast<uint32_t>(lane) < take;
  const uint32_t t = live ? queue[(head + lane) & (kQueueCap - 1)] : 0u;
  const uint32_t m = live ? masks[t] : 0u;
  const int digits = static_cast<int>(__reduce_max_sync(0xffffffffu, __popc(m)));
  uint32_t s0 = 0u, s1 = 0u, s2 = 0u, s3 = 0u, digits_full = 0u;
  uint32_t x = m;
#pragma unroll 2
  for (int i = 0; i < digits; ++i) {
    if (x != 0u) {
      const int d = __ffs(x) - 1;
      const uint4 v = reinterpret_cast<const uint4*>(counts)[d];
      const uint32_t cp = cap[d] * 0x00010001u;
      s0 += v.x;
      s1 += v.y;
      s2 += v.z;
      s3 += v.w;
      digits_full |= (__vcmpgeu2(v.x, cp) & 0x00010001u) | (__vcmpgeu2(v.y, cp) & 0x00010001u) << 2
                     | (__vcmpgeu2(v.z, cp) & 0x00010001u) << 4
                     | (__vcmpgeu2(v.w, cp) & 0x00010001u) << 6;
      x &= x - 1u;
    }
  }
  digits_full = (digits_full | digits_full >> 15) & 0xffu;

  const uint4 ua = reinterpret_cast<const uint4*>(ucount)[0],
              ub = reinterpret_cast<const uint4*>(ucount)[1];
  const uint32_t u[kColors] = {ua.x, ua.y, ua.z, ua.w, ub.x, ub.y, ub.z, ub.w};
  const uint32_t f[kColors] = {s0 & 0xffffu, s0 >> 16, s1 & 0xffffu, s1 >> 16,
                               s2 & 0xffffu, s2 >> 16, s3 & 0xffffu, s3 >> 16};
  uint32_t union_full = 0u;
#pragma unroll
  for (int c = 0; c < kColors; ++c) union_full |= (u[c] >= kUnionCap ? 1u : 0u) << c;
  const uint32_t allowed = ~(digits_full | union_full) & 0xffu;
  const bool blocked = allowed == 0u;
  const uint32_t pool = blocked ? ~union_full & 0xffu : allowed;
  uint32_t best = ~0u;
#pragma unroll
  for (int c = 0; c < kColors; ++c) {
    const uint32_t key = blocked ? (u[c] << 3 | static_cast<uint32_t>(c))
                                 : (f[c] << 10 | u[c] << 3 | ((c - t) & 7u));
    best = (pool >> c) & 1u ? min(best, key) : best;
  }
  const uint32_t c = blocked ? best & 7u : ((best & 7u) + t) & 7u;
  const uint32_t shift = 16u * (c & 1u), inc = 1u << shift;

  bool over = false;
  if (live) over = atomicAdd(ucount + c, 1u) >= static_cast<uint32_t>(kUnionCap);
  x = m;
#pragma unroll 2
  for (int i = 0; i < digits; ++i) {
    if (x != 0u) {
      const int d = __ffs(x) - 1;
      const uint32_t old = atomicAdd(counts + 4 * d + (c >> 1), inc);
      over |= !blocked && ((old >> shift) & 0xffffu) >= cap[d];
      x &= x - 1u;
    }
  }

  const bool loser = live && over;
  if (loser) atomicSub(ucount + c, 1u);
  x = loser ? m : 0u;
#pragma unroll 2
  for (int i = 0; i < digits; ++i) {
    if (x != 0u) {
      atomicSub(counts + 4 * (__ffs(x) - 1) + (c >> 1), inc);
      x &= x - 1u;
    }
  }
  const bool won = live && !over;
  if (won) color[t] = static_cast<uint16_t>(c);
  return Claim{t, __ballot_sync(0xffffffffu, loser), __ballot_sync(0xffffffffu, won), loser};
}

//: THE CLAIMS: up to `kClaimRounds` rounds on every warp, a batch each, while more than a batch is queued, losers
//: appended past the round's queue (a ring of `kQueueCap`: at most a window's tokens are queued, so a round's appends
//: never reach its reads); then warp 0's batches to the queue's end, losers requeued and a batch that keeps nothing
//: followed by its first token alone, which always keeps -- the rounds may keep nothing, warp 0's batches cannot.
//: -- see docs/internals/facts/facts.md#coloring
__device__ __forceinline__ void claim_all(uint8_t* smem, int warp, int lane) {
  uint16_t* const queue = reinterpret_cast<uint16_t*>(smem + FactsSmem::kQueue);
  uint32_t* const appends = reinterpret_cast<uint32_t*>(smem + FactsSmem::kAppends);
  const uint32_t below = (1u << lane) - 1u;
  uint32_t head = 0u,
           tail = static_cast<uint32_t>(reinterpret_cast<const int*>(smem + FactsSmem::kMisc)[1]);
#pragma unroll 1
  for (int r = 0; r < kClaimRounds && tail - head > 32u; ++r) {
    const uint32_t mine = head + 32u * static_cast<uint32_t>(warp);
    const Claim cl = claim_batch(smem, mine, mine < tail ? min(32u, tail - mine) : 0u, lane);
    uint32_t base = 0u;
    if (lane == 0 && cl.lost != 0u) base = atomicAdd(appends + (r & 1), __popc(cl.lost));
    base = __shfl_sync(0xffffffffu, base, 0);
    if (cl.loser)
      queue[(tail + base + __popc(cl.lost & below)) & (kQueueCap - 1)] =
          static_cast<uint16_t>(cl.t);
    __syncthreads();
    const uint32_t grown = appends[r & 1];
    if (warp == 0 && lane == 0) appends[(r + 1) & 1] = 0u;
    head = min(head + kFactsWarps * 32u, tail);
    tail += grown;
    __syncthreads();
  }
  if (warp != 0) return;
  bool solo = false;
  for (int k = __any_sync(0xffffffffu, head < tail) != 0; k > 0; --k)
#pragma unroll 1
    do {
      const uint32_t take = solo ? 1u : min(32u, tail - head);
      const Claim cl = claim_batch(smem, head, take, lane);
      head += take;
      if (cl.loser)
        queue[(tail + __popc(cl.lost & below)) & (kQueueCap - 1)] = static_cast<uint16_t>(cl.t);
      tail += __popc(cl.lost);
      solo = cl.kept == 0u;
      __syncwarp();
    } while (__any_sync(0xffffffffu, head < tail));
}

//: THE SINGLES, warp 0, lane `d` digit `d`: the union filled to a level, pass by pass -- each digit wanting its singles
//: at its room's colors (under its capacity) that have quota left, in the rotation `(d + i) & 7`; each color granting
//: in digit order while its quota (`level - n_c`) lasts; a pass that grants nothing raising the level -- and at the
//: union's cap the rest placed by quota alone; each digit's singles' color ends posted for the deal.
//: -- see docs/internals/facts/facts.md#coloring
__device__ void singles(uint8_t* smem, const uint32_t* tile, int lane) {
  const uint32_t* const uword = reinterpret_cast<const uint32_t*>(smem + FactsSmem::kUword);
  const uint32_t* const mword = reinterpret_cast<const uint32_t*>(smem + FactsSmem::kMword);
  const uint32_t* const ucount = reinterpret_cast<const uint32_t*>(smem + FactsSmem::kUcount);
  const int d = lane;
  const int k = reinterpret_cast<const uint16_t*>(smem + FactsSmem::kPre)[d * kWindowWords + 15]
                + __popc(tile[tile_at(d, 15)] & uword[15] & ~mword[15]);
  const uint4 cw = reinterpret_cast<const uint4*>(smem + FactsSmem::kCounts)[d];
  const uint32_t cnt_lo = (cw.x & 0xffu) | (cw.x >> 16 & 0xffu) << 8 | (cw.y & 0xffu) << 16
                          | (cw.y >> 16 & 0xffu) << 24;
  const uint32_t cnt_hi = (cw.z & 0xffu) | (cw.z >> 16 & 0xffu) << 8 | (cw.w & 0xffu) << 16
                          | (cw.w >> 16 & 0xffu) << 24;
  const int room_cap =
      static_cast<int>(reinterpret_cast<const uint32_t*>(smem + FactsSmem::kCap)[d]);
  const uint4 ua = reinterpret_cast<const uint4*>(ucount)[0],
              ub = reinterpret_cast<const uint4*>(ucount)[1];
  uint32_t u_lo = ua.x | ua.y << 8 | ua.z << 16 | ua.w << 24;
  uint32_t u_hi = ub.x | ub.y << 8 | ub.z << 16 | ub.w << 24;

  //: the level: the least whose quotas hold every single, lanes testing levels `lane` and `lane + 32`
  const uint32_t total = __reduce_add_sync(0xffffffffu, static_cast<uint32_t>(k));
  int level;
  {
    int fa = 0, fb = 0;
#pragma unroll
    for (int c = 0; c < kColors; ++c) {
      fa += max(0, lane - static_cast<int>(byte_of(u_lo, u_hi, c)));
      fb += max(0, lane + 32 - static_cast<int>(byte_of(u_lo, u_hi, c)));
    }
    const uint32_t ba = __ballot_sync(0xffffffffu, static_cast<uint32_t>(fa) >= total),
                   bb = __ballot_sync(0xffffffffu, static_cast<uint32_t>(fb) >= total);
    level = ba != 0u ? __ffs(ba) - 1 : (bb != 0u ? 32 + __ffs(bb) - 1 : kUnionCap);
  }

  //: the passes
  uint32_t take_lo = 0u, take_hi = 0u;
  int rem = k;
#pragma unroll 1
  for (int go = __any_sync(0xffffffffu, rem > 0) != 0; go > 0;
       go = static_cast<int>(__any_sync(0xffffffffu, rem > 0) != 0)
            & static_cast<int>(level <= kUnionCap)) {
    uint32_t w_lo = 0u, w_hi = 0u;
    int r = rem;
#pragma unroll
    for (int i = 0; i < kColors; ++i) {
      const int c = (d + i) & 7;
      const int q = level - static_cast<int>(byte_of(u_lo, u_hi, c));
      const int room = room_cap - static_cast<int>(byte_of(cnt_lo, cnt_hi, c))
                       - static_cast<int>(byte_of(take_lo, take_hi, c));
      const int w = q > 0 ? max(0, min(r, room)) : 0;
      r -= w;
      add_byte(w_lo, w_hi, c, static_cast<uint32_t>(w));
    }
    //: each color's wants before the lane's, its pairs as u16 halves
    uint32_t want[4] = {
        (w_lo & 0xffu) | (w_lo >> 8 & 0xffu) << 16, (w_lo >> 16 & 0xffu) | (w_lo >> 24) << 16,
        (w_hi & 0xffu) | (w_hi >> 8 & 0xffu) << 16, (w_hi >> 16 & 0xffu) | (w_hi >> 24) << 16};
    uint32_t incl[4] = {want[0], want[1], want[2], want[3]};
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const uint32_t y = __shfl_up_sync(0xffffffffu, incl[p], off);
        incl[p] += lane >= off ? y : 0u;
      }
    }
    uint32_t sum[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) sum[p] = __shfl_sync(0xffffffffu, incl[p], 31);
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

  //: the rest by quota alone, in digit order over the colors' room under the cap
  {
    int incl = rem;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, incl, off);
      incl += lane >= off ? y : 0;
    }
    const int lo = incl - rem, hi = incl;
    int bound = 0;
#pragma unroll
    for (int c = 0; c < kColors; ++c) {
      const int q = kUnionCap - static_cast<int>(byte_of(u_lo, u_hi, c));
      add_byte(take_lo, take_hi, c,
               static_cast<uint32_t>(max(0, min(hi, bound + q) - max(lo, bound))));
      bound += q;
    }
  }
  //: each color's end among the digit's singles, for the deal
  uint32_t end[kColors];
  uint32_t e = 0u;
#pragma unroll
  for (int c = 0; c < kColors; ++c) {
    e += byte_of(take_lo, take_hi, c);
    end[c] = e;
  }
  reinterpret_cast<uint4*>(smem + FactsSmem::kEnds)[d] = make_uint4(
      end[0] | end[1] << 16, end[2] | end[3] << 16, end[4] | end[5] << 16, end[6] | end[7] << 16);
}

//: THE SLOTS, a warp's two words: each token's color -- a multi-digit token's claimed, a single's dealt by its index
//: among its digit's singles against the digit's color ends -- and three ballots giving each color's tokens in the
//: word; after the words' color counts meet, each token's slot is `8 x (its color's tokens before it) + color`, into the
//: union by slot; warp 0 the extent. -- see docs/internals/facts/facts.md#coloring
__device__ __forceinline__ void slots(uint8_t* smem, const uint32_t* tile, int tid, int warp,
                                      int lane) {
  const uint32_t* const uword = reinterpret_cast<const uint32_t*>(smem + FactsSmem::kUword);
  const uint32_t* const mword = reinterpret_cast<const uint32_t*>(smem + FactsSmem::kMword);
  const uint32_t* const masks = reinterpret_cast<const uint32_t*>(smem + FactsSmem::kMasks);
  const uint16_t* const pre = reinterpret_cast<const uint16_t*>(smem + FactsSmem::kPre);
  const uint4* const ends = reinterpret_cast<const uint4*>(smem + FactsSmem::kEnds);
  uint16_t* const slot = reinterpret_cast<uint16_t*>(smem + FactsSmem::kSlot);
  uint16_t* const by_slot = reinterpret_cast<uint16_t*>(smem + FactsSmem::kUnion);
  uint8_t* const wcount = smem + FactsSmem::kWcount;
  const uint32_t below = (1u << lane) - 1u;
  constexpr int kEarlierWords = kWindowWords - 1;

  int rank[2], col[2];
  bool in[2];
#pragma unroll
  for (int jj = 0; jj < 2; ++jj) {
    const int j = 2 * warp + jj;
    const uint32_t u = uword[j], single = u & ~mword[j];
    in[jj] = (u >> lane) & 1u;
    uint32_t c = slot[kTokBits * j + lane] & 7u;
    if ((single >> lane) & 1u) {
      const int d = __ffs(masks[kTokBits * j + lane]) - 1;
      const uint32_t idx = pre[d * kWindowWords + j] + __popc(tile[tile_at(d, j)] & single & below);
      const uint4 e = ends[d];
      c = (idx >= (e.x & 0xffffu)) + (idx >= (e.x >> 16)) + (idx >= (e.y & 0xffffu))
          + (idx >= (e.y >> 16)) + (idx >= (e.z & 0xffffu)) + (idx >= (e.z >> 16))
          + (idx >= (e.w & 0xffffu));
    }
    c = in[jj] ? c : 0u;
    const uint32_t v0 = __ballot_sync(0xffffffffu, in[jj] && (c & 1u)),
                   v1 = __ballot_sync(0xffffffffu, in[jj] && (c & 2u)),
                   v2 = __ballot_sync(0xffffffffu, in[jj] && (c & 4u));
    const uint32_t same = u & ((c & 1u) ? v0 : ~v0) & ((c & 2u) ? v1 : ~v1) & ((c & 4u) ? v2 : ~v2);
    rank[jj] = __popc(same & below);
    col[jj] = static_cast<int>(c);
    const int cc = lane & 7;
    const uint32_t sel = u & ((cc & 1) ? v0 : ~v0) & ((cc & 2) ? v1 : ~v1) & ((cc & 4) ? v2 : ~v2);
    if (lane < kColors) wcount[j * kColors + lane] = static_cast<uint8_t>(__popc(sel));
  }

  __syncthreads();
  reinterpret_cast<uint32_t*>(smem + FactsSmem::kTab)[tid] = 0u;
#pragma unroll
  for (int jj = 0; jj < 2; ++jj) {
    const int j = 2 * warp + jj;
    int before = 0;
#pragma unroll
    for (int i = 0; i < kEarlierWords; ++i) before += i < j ? wcount[i * kColors + col[jj]] : 0;
    if (in[jj]) {
      const int s = kColors * (before + rank[jj]) + col[jj];
      slot[kTokBits * j + lane] = static_cast<uint16_t>(s);
      by_slot[s] = static_cast<uint16_t>(kTokBits * j + lane);
    }
  }

  if (warp == 0) {
    int n = 0;
#pragma unroll 1
    for (int i = 0; i < kWindowWords; ++i) n += wcount[i * kColors + (lane & 7)];
    const uint32_t widest = __reduce_max_sync(0xffffffffu, lane < kColors ? n : 0);
    if (lane == 0)
      reinterpret_cast<int*>(smem + FactsSmem::kMisc)[0] = kColors * static_cast<int>(widest);
  }
  __syncthreads();
}

//: THE LISTS, a warp's 4 lists `w + 8 j` (its own heads), 8 lanes a list, lane `s` its words `2s`, `2s + 1`: the first
//: walk counts each entry into its (list, pass)'s color count and takes the write maxima; lane `q` turns pass `q`'s
//: counts into its steps, start and earlier passes' counts; the second walk places and writes each entry, lane `q` its
//: pass's pads. -- see docs/internals/facts/facts.md#lists-out
template <bool kWriteSide>
__device__ void lists_out(const FactsParams& p, const Unit& x, uint8_t* smem, int tid, int warp,
                          int lane) {
  const uint8_t* const win = smem + FactsSmem::kWin;
  const uint32_t* const tile = reinterpret_cast<const uint32_t*>(smem + FactsSmem::kTile);
  const uint16_t* const slotv = reinterpret_cast<const uint16_t*>(smem + FactsSmem::kSlot);
  const float2* const cols = reinterpret_cast<const float2*>(smem + FactsSmem::kCols);
  uint32_t* const tab = reinterpret_cast<uint32_t*>(smem + FactsSmem::kTab);
  uint16_t* const startv = reinterpret_cast<uint16_t*>(smem + FactsSmem::kStart);
  const int dl = lane >> 3, s = lane & 7, d = warp + kFactsWarps * dl;
  const int width = p.B[x.level], digit = kGroupDigits * x.group + d;
  const bool listed = digit < width;
  const int pl = p.pass_log[x.side], ext = reinterpret_cast<const int*>(smem + FactsSmem::kMisc)[0];
  const int passes = (ext + (1 << pl) - 1) >> pl;
  const ListsOut out = p.lists[x.side];
  const int64_t win_index = static_cast<int64_t>(x.bh) * p.windows + x.window;
  const int64_t list = win_index * width + digit;
  const int64_t group_index = win_index * ((width + kGroupDigits - 1) / kGroupDigits) + x.group;
  uint2* const heads = out.heads + (group_index * kFactsWarps + warp) * (kHeadLists * kHeadEntries)
                       + dl * kHeadEntries;
  uint2* const entries = out.entries + list * kListCap;
  const uint32_t words[2] = {tile[tile_at(d, 2 * s)], tile[tile_at(d, 2 * s + 1)]};

  //: the first walk
  uint32_t run_lo = 0u, run_hi = 0u;
  float mx = 0.0f, mf = 0.0f;
#pragma unroll
  for (int jj = 0; jj < 2; ++jj) {
    uint32_t xw = words[jj];
    for (int k = __any_sync(0xffffffffu, xw != 0u) != 0; k > 0; --k)
#pragma unroll 1
      do {
        if (xw != 0u) {
          const int t = kTokBits * (2 * s + jj) + __ffs(xw) - 1;
          const int sl = slotv[t], c = sl & 7, q = sl >> pl;
          atomicAdd(tab + (d * kMaxPasses + q) * 2 + (c >> 2), 1u << (8 * (c & 3)));
          add_byte(run_lo, run_hi, c, 1u);
          if constexpr (kWriteSide) {
            const float b = static_cast<float>(win[window_at(t, d)]);
            const float2 col = cols[t];
            mx = fmaxf(mx, b * col.x);
            mf = fmaxf(mf, b * col.y);
          }
          xw &= xw - 1u;
        }
      } while (__any_sync(0xffffffffu, xw != 0u));
  }
  __syncwarp();
  //: the list's tokens of each color before the lane's words; on the write side its exponents and scales
  uint32_t pre_lo = run_lo, pre_hi = run_hi;
#pragma unroll
  for (int off = 1; off < 8; off <<= 1) {
    const uint32_t ylo = __shfl_up_sync(0xffffffffu, pre_lo, off, 8),
                   yhi = __shfl_up_sync(0xffffffffu, pre_hi, off, 8);
    pre_lo += s >= off ? ylo : 0u;
    pre_hi += s >= off ? yhi : 0u;
  }
  pre_lo -= run_lo;
  pre_hi -= run_hi;
  float sm = 1.0f, sv = 1.0f;
  if constexpr (kWriteSide) {
#pragma unroll
    for (int off = 1; off < 8; off <<= 1) {
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
      mf = fmaxf(mf, __shfl_xor_sync(0xffffffffu, mf, off));
    }
    const int km = ceil_exponent(mx), kv = ceil_exponent(mf);
    sm = pow2(kCoefTop - max(km, kUnitMin));
    sv = pow2(kCoefTop - max(kv, kUnitMin));
    if (listed && s == 0)
      p.list_exp[list] = static_cast<uint32_t>(static_cast<uint16_t>(km))
                         | static_cast<uint32_t>(static_cast<uint16_t>(kv)) << 16;
  }
  //: lane `s` pass `s`: its steps, its start and its colors' counts in the earlier passes
  uint32_t t_lo = 0u, t_hi = 0u;
  if (s < passes) {
    t_lo = tab[(d * kMaxPasses + s) * 2];
    t_hi = tab[(d * kMaxPasses + s) * 2 + 1];
  }
  const uint32_t m4 = __vmaxu4(t_lo, t_hi);
  const int steps =
      static_cast<int>(max(max(m4 & 0xffu, (m4 >> 8) & 0xffu), max((m4 >> 16) & 0xffu, m4 >> 24)));
  int start = kColors * steps;
  uint32_t b_lo = t_lo, b_hi = t_hi;
#pragma unroll
  for (int off = 1; off < 8; off <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, start, off, 8);
    const uint32_t ylo = __shfl_up_sync(0xffffffffu, b_lo, off, 8),
                   yhi = __shfl_up_sync(0xffffffffu, b_hi, off, 8);
    start += s >= off ? y : 0;
    b_lo += s >= off ? ylo : 0u;
    b_hi += s >= off ? yhi : 0u;
  }
  start -= kColors * steps;
  b_lo -= t_lo;
  b_hi -= t_hi;
  if (listed) out.at[list * kListAt + s] = static_cast<uint16_t>(start);
  if (s < kMaxPasses) {
    tab[(d * kMaxPasses + s) * 2] = b_lo;
    tab[(d * kMaxPasses + s) * 2 + 1] = b_hi;
    startv[d * kMaxPasses + s] = static_cast<uint16_t>(start);
  }
  __syncwarp();

  //: the second walk: each entry at its pass's start, 8 x its rank among its color there, plus its color
  run_lo = run_hi = 0u;
  const int row = p.read_row;
#pragma unroll
  for (int jj = 0; jj < 2; ++jj) {
    uint32_t xw = words[jj];
    for (int k = __any_sync(0xffffffffu, xw != 0u) != 0; k > 0; --k)
#pragma unroll 1
      do {
        if (xw != 0u) {
          const int t = kTokBits * (2 * s + jj) + __ffs(xw) - 1;
          xw &= xw - 1u;
          const uint32_t sl = slotv[t];
          const int c = static_cast<int>(sl & 7u), q = static_cast<int>(sl >> pl);
          const uint32_t rank = byte_of(pre_lo + run_lo, pre_hi + run_hi, c);
          add_byte(run_lo, run_hi, c, 1u);
          const uint32_t before =
              (tab[(d * kMaxPasses + q) * 2 + (c >> 2)] >> (8 * (c & 3))) & 0xffu;
          const int pos =
              startv[d * kMaxPasses + q] + kColors * static_cast<int>(rank - before) + c;
          const float b = static_cast<float>(win[window_at(t, d)]);
          uint2 e;
          if constexpr (kWriteSide) {
            const float2 col = cols[t];
            const uint32_t cv = __half_as_ushort(__float2half_rn(b * col.y * sv));
            const uint32_t cm = __half_as_ushort(__float2half_rn(b * col.x * sm));
            e = make_uint2(sl, cv | cm << 16);
          } else {
            const uint32_t local = sl - (static_cast<uint32_t>(q) << pl);
            const uint32_t coef = __half_as_ushort(__float2half_rn(b * (1.0f / 256.0f)));
            e = make_uint2(local * row + 16u * carry::pool_chunk(row, local, 0u),
                           coef | (carry::kPadBytes + carry::panel_row(local)) << 16);
          }
          if (listed) {
            entries[pos] = e;
            if (pos < kHeadEntries) heads[pos] = e;
          }
        }
      } while (__any_sync(0xffffffffu, xw != 0u));
  }
  //: lane `q`'s pads: each color's missing steps in pass `q`
  if (listed && s < passes) {
#pragma unroll
    for (int c = 0; c < kColors; ++c) {
      const uint2 e = kWriteSide ? make_uint2((static_cast<uint32_t>(s) << pl) + c, 0u)
                                 : make_uint2(c * row + 16u * carry::pool_chunk(row, c, 0u),
                                              carry::panel_row(c) << 16);
#pragma unroll 1
      for (int i = static_cast<int>(byte_of(t_lo, t_hi, c)); i < steps; ++i) {
        const int pos = start + kColors * i + c;
        entries[pos] = e;
        if (pos < kHeadEntries) heads[pos] = e;
      }
    }
  }
  //: the union by slot and its extent
  reinterpret_cast<uint32_t*>(out.union_tok + group_index * kWindowTokens)[tid] =
      reinterpret_cast<const uint32_t*>(smem + FactsSmem::kUnion)[tid];
  if (tid == 0) out.union_count[group_index] = ext;
}

//: THE GRID: a CTA a unit, a window's units innermost. Every unit loads its window (a columns unit nothing) and a write
//: list unit its columns; a vote or list unit votes and writes its bits; a list unit colors its union and writes its
//: lists. -- see docs/internals/facts/facts.md#the-grid
__global__ __launch_bounds__(kFactsThreads, kFactsCtasMin) void facts_pass(const FactsParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ long long s_clock[kFactsWarps][Clock::kRow];
  const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
  Clock clock(&s_clock[warp][0], lane, p.ledger != nullptr);
  const Unit x = unit_of(p, static_cast<int>(blockIdx.x));
  uint8_t* const win = smem + FactsSmem::kWin;
  uint32_t* const tile = reinterpret_cast<uint32_t*>(smem + FactsSmem::kTile);
  float2* const cols = reinterpret_cast<float2*>(smem + FactsSmem::kCols);
  float2* const wmax = reinterpret_cast<float2*>(smem + FactsSmem::kWmax);
  const bool listed = x.kind == kUnitList, writes = (x.side == kWrite) & listed;
  const bool columns = (x.kind == kUnitColumns) | writes;

  //: THE LOAD: the window's copies, a write unit's columns beside them, each warp's largest posted
  if (x.kind != kUnitColumns) issue_window(p, x, ops::smem_addr(win), tid);
  ops::stage_commit();
  float2 col[2] = {make_float2(0.0f, 0.0f), make_float2(0.0f, 0.0f)};
  if (columns) {
    float2 top = make_float2(0.0f, 0.0f);
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      col[k] = load_column(p, x, tid + kFactsThreads * k);
      cols[tid + kFactsThreads * k] = col[k];
      top = make_float2(fmaxf(top.x, col[k].x), fmaxf(top.y, col[k].y));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      top.x = fmaxf(top.x, __shfl_xor_sync(0xffffffffu, top.x, off));
      top.y = fmaxf(top.y, __shfl_xor_sync(0xffffffffu, top.y, off));
    }
    if (lane == 0) wmax[warp] = top;
  }
  ops::stage_wait<0>();
  __syncthreads();
  clock.lap(kPhaseLoad);
  if ((x.kind == kUnitColumns) | (writes & (x.group == 0))) columns_out(p, x, col, wmax, tid);
  if (x.kind == kUnitColumns) {
    clock.publish(p.ledger, 0, kFactsWarps, warp);
    return;
  }

  //: THE VOTE AND THE BITS OUT
  vote(p, x, win, tile, warp, lane);
  __syncthreads();
  bits_out(p, x, tile, tid, warp, lane);
  clock.lap(kPhaseVote);
  if (!listed) {
    clock.publish(p.ledger, 0, kFactsWarps, warp);
    return;
  }

  //: THE COLORING: its inputs; the queue beside the singles' index bases (warps 1-4); the claims; warp 0's singles;
  //: the slots
  prepare(smem, tile, tid, warp, lane);
  __syncthreads();
  enqueue(smem, tid, warp, lane);
  if (warp >= 1 && warp <= 4) singles_before(smem, tile, warp, lane);
  __syncthreads();
  clock.lap(kPhaseInputs);
  claim_all(smem, warp, lane);
  clock.lap(kPhaseClaims);
  if (warp == 0) singles(smem, tile, lane);
  __syncthreads();
  clock.lap(kPhaseSingles);
  slots(smem, tile, tid, warp, lane);
  clock.lap(kPhaseSlots);

  //: THE LISTS
  if (writes) {
    lists_out<true>(p, x, smem, tid, warp, lane);
  } else {
    lists_out<false>(p, x, smem, tid, warp, lane);
  }
  clock.lap(kPhaseLists);
  clock.publish(p.ledger, 0, kFactsWarps, warp);
}

//: A level's rows as the kernel reads them: `[B, T, H, width]` u8, digits contiguous; the view's strides in bytes.
RowView row_view(const Tensor& t, int64_t B, int64_t T, int64_t H, const char* name) {
  STD_TORCH_CHECK(t.is_cuda() && t.scalar_type() == Dtype::Byte, name, " is a CUDA u8 tensor, got ",
                  t.scalar_type());
  STD_TORCH_CHECK(t.dim() == 4 && t.size(0) == B && t.size(1) == T && t.size(2) == H, name,
                  " is [B, T, H, width] = [", B, ", ", T, ", H, width], got ", shape(t.sizes()));
  STD_TORCH_CHECK(t.stride(3) == 1 || t.size(3) == 1, name, "'s digits are contiguous");
  //: THE ROW ALIGNMENT LAW: the first row and every stride of a dimension longer than one on 32 bytes
  const bool aligned = reinterpret_cast<uintptr_t>(t.const_data_ptr()) % kRowAlign == 0
                       && (t.size(0) == 1 || t.stride(0) % kRowAlign == 0)
                       && (t.size(1) == 1 || t.stride(1) % kRowAlign == 0)
                       && (t.size(2) == 1 || t.stride(2) % kRowAlign == 0);
  STD_TORCH_CHECK(aligned, name,
                  " breaks the row alignment law: its rows start on and step by multiples of ",
                  kRowAlign, " bytes");
  return RowView{static_cast<const uint8_t*>(t.const_data_ptr()), t.stride(0), t.stride(1),
                 t.stride(2), static_cast<int>(t.size(3))};
}

ColumnView column_view(const Tensor& t, int64_t B, int64_t T, int64_t H, const char* name) {
  STD_TORCH_CHECK(t.is_cuda() && t.scalar_type() == Dtype::Float, name,
                  " is a CUDA fp32 tensor, got ", t.scalar_type());
  STD_TORCH_CHECK(t.dim() == 3 && t.size(0) == B && t.size(1) == T && t.size(2) == H, name,
                  " is [B, T, H] = [", B, ", ", T, ", ", H, "], got ", shape(t.sizes()));
  return ColumnView{t.const_data_ptr<float>(), t.stride(0), t.stride(1), t.stride(2)};
}

//: THE PLAN, a window's units: each side's sparse levels in order, a unit a group -- a list unit at the side's list
//: level, a vote unit elsewhere -- and the window's columns unit when the write side lists nothing.
//: -- see docs/internals/facts/facts.md#the-plan
void plan_units(FactsParams& p, const uint32_t (&dense)[kSides]) {
  int n = 0;
  for (int side = 0; side < kSides; ++side)
    for (int level = 0; level < p.D; ++level) {
      if (level_is_dense(dense[side], level)) continue;
      const int kind = level == p.list_level[side] ? kUnitList : kUnitVote;
      for (int g = 0; g < (p.B[level] + kGroupDigits - 1) / kGroupDigits; ++g)
        p.unit[n++] = unit_code(kind, side, level, g);
    }
  if (p.list_level[kWrite] < 0) p.unit[n++] = unit_code(kUnitColumns, kWrite, 0, 0);
  p.per_window = n;
}

//: THE BOUND PHASE LEDGER, or null: the next launches book into it.
long long* g_ledger = nullptr;

}  // namespace detail

//: -- see docs/internals/facts/facts_api.md#rola-facts-ledger-bind
void rola_facts_ledger_bind(std::optional<Tensor> ledger) {
  if (!ledger.has_value()) {
    detail::g_ledger = nullptr;
    return;
  }
  STD_TORCH_CHECK(ledger->is_cuda() && ledger->scalar_type() == Dtype::Long && ledger->dim() == 2
                      && ledger->size(0) == kFactsWarps && ledger->size(1) == kFactsPhases
                      && ledger->is_contiguous(),
                  "the facts pass's phase ledger is a contiguous CUDA int64 tensor [", kFactsWarps,
                  "][", kFactsPhases, "]");
  detail::g_ledger = reinterpret_cast<long long*>(ledger->mutable_data_ptr<int64_t>());
}

//: -- see docs/internals/facts/facts_api.md#rola-facts-forward
void rola_facts_forward(std::vector<Tensor> read, std::vector<Tensor> write,
                        std::vector<Tensor> scale_write, Tensor g_write, Tensor v_scale,
                        Tensor words, Tensor unions, Tensor list_exp, Tensor fm, Tensor gains16,
                        Tensor gains_exp, std::vector<Tensor> write_lists,
                        std::vector<Tensor> read_lists, std::vector<int64_t> widths,
                        int64_t dense_read, std::vector<int64_t> mask_read, int64_t dense_write,
                        std::vector<int64_t> mask_write, int64_t write_list_level,
                        int64_t read_list_level, int64_t window, int64_t read_pass,
                        int64_t write_pass, int64_t read_row) {
  check_arch_table();
  const int D = static_cast<int>(widths.size());
  STD_TORCH_CHECK(D >= 1 && D <= kMaxLevels, "the routing depth is one to four levels, got ", D);
  STD_TORCH_CHECK(static_cast<int>(read.size()) == D && static_cast<int>(write.size()) == D
                      && static_cast<int>(scale_write.size()) == D,
                  "each side holds one row set a level and the write side one scale a level: ", D);
  STD_TORCH_CHECK(window == kWindowTokens, "the facts pass's window is the carry's ", kWindowTokens,
                  " tokens, got ", window);
  const Tensor& first = read[0];
  STD_TORCH_CHECK(first.dim() == 4, "a level's rows are [B, T, H, width], got ",
                  shape(first.sizes()));
  const int64_t B = first.size(0), T = first.size(1), H = first.size(2);
  STD_TORCH_CHECK(T >= 1, "a call carries at least one token");

  FactsParams p{};
  p.D = D;
  p.H = static_cast<int>(H);
  p.L = static_cast<int>(T);
  p.token_words = token_words(p.L);
  p.windows = (p.token_words + kWindowWords - 1) / kWindowWords;
  const std::vector<Tensor>* sides[kSides] = {&read, &write};
  const int64_t dense_in[kSides] = {dense_read, dense_write};
  const std::vector<int64_t>* masks[kSides] = {&mask_read, &mask_write};
  uint32_t dense[kSides] = {};
  for (int level = 0; level < D; ++level) {
    STD_TORCH_CHECK(widths[level] >= 1 && widths[level] <= kMaxWidth, "level ", level, " is ",
                    widths[level], " digits wide; a level is 1 to ", kMaxWidth);
    p.B[level] = static_cast<int>(widths[level]);
    p.row_base[level] = p.rows_total;
    p.rows_total += p.B[level];
  }
  const int need = mask_words(p.rows_total);
  for (int side = 0; side < kSides; ++side) {
    STD_TORCH_CHECK(dense_in[side] >= 0 && dense_in[side] < (1 << D), "side ", side,
                    "'s dense-level bitmask ", dense_in[side], " names a level outside the depth ",
                    D);
    STD_TORCH_CHECK(static_cast<int>(masks[side]->size()) == need, "side ", side, " carries ",
                    masks[side]->size(), " digit-mask words; ", p.rows_total, " rows need ", need);
    dense[side] = static_cast<uint32_t>(dense_in[side]);
    for (int w = 0; w < need; ++w) p.mask[side][w] = static_cast<uint32_t>((*masks[side])[w]);
    for (int level = 0; level < D; ++level) {
      if (level_is_dense(dense[side], level)) continue;
      const Tensor& rows = (*sides[side])[level];
      p.rows[side][level] = detail::row_view(rows, B, T, H, side ? "write[l]" : "read[l]");
      //: a digit the mask names is read, so it lies inside the level's row
      const int64_t row_width = rows.size(3);
      for (int digit = 0; digit < p.B[level]; ++digit) {
        const int row = p.row_base[level] + digit;
        STD_TORCH_CHECK(digit < row_width || !mask_bit(p.mask[side][row >> 5], row), "side ", side,
                        " level ", level, "'s mask names digit ", digit, " past its ", row_width,
                        "-digit rows");
      }
    }
  }
  const int64_t list_levels[kSides] = {read_list_level, write_list_level};
  const int64_t passes[kSides] = {read_pass, write_pass};
  for (int side = 0; side < kSides; ++side) {
    const int64_t level = list_levels[side];
    STD_TORCH_CHECK(level >= -1 && level < D && (level < 0 || !level_is_dense(dense[side], level)),
                    "side ", side, "'s list level is a sparse level of the depth ", D,
                    " or -1 for none, got ", level);
    p.list_level[side] = static_cast<int>(level);
    int log = kMinPassLog;
    while (log < kMaxPassLog && (int64_t{1} << log) < passes[side]) ++log;
    STD_TORCH_CHECK(passes[side] == (int64_t{1} << log), "side ", side,
                    "'s pass is a power of two from ", 1 << kMinPassLog, " to ", 1 << kMaxPassLog,
                    " slots (a list's passes and its end fit its ", kListAt, " starts), got ",
                    passes[side]);
    p.pass_log[side] = log;
  }
  STD_TORCH_CHECK(read_row >= 64 && read_row % 64 == 0
                      && (read_row == 128 || read_row / 64 % 2 == 1)
                      && read_pass * read_row <= 65536,
                  "the read pool row is 128 bytes or an odd number of 64, its pass's pool inside a "
                  "16-bit offset; got ",
                  read_row, " bytes at a ", read_pass, "-slot pass");
  p.read_row = static_cast<int>(read_row);
  for (int level = 0; level < D; ++level)
    p.scale_write[level] = detail::column_view(scale_write[level], B, T, H, "scale_write[l]");
  p.g_write = detail::column_view(g_write, B, T, H, "g_write");
  p.v_scale = detail::column_view(v_scale, B, T, H, "v_scale");
  STD_TORCH_CHECK(words.is_cuda() && words.scalar_type() == Dtype::Int && words.is_contiguous()
                      && words.dim() == 4 && words.size(0) == B * H && words.size(1) == kSides
                      && words.size(2) == p.rows_total && words.size(3) == p.token_words,
                  "words is the contiguous int32 class-1 table [", B * H, ", 2, ", p.rows_total,
                  ", ", p.token_words, "], got ", shape(words.sizes()));
  LivenessLayout layout{D, {}, p.L};
  for (int level = 0; level < D; ++level) layout.B[level] = p.B[level];
  for (int level = 0; level < D; ++level) p.group_base[level] = layout.group_base(level);
  p.groups = layout.groups();
  p.tiles = layout.tiles();
  STD_TORCH_CHECK(unions.is_cuda() && unions.scalar_type() == Dtype::Int && unions.is_contiguous()
                      && unions.dim() == 4 && unions.size(0) == B * H && unions.size(1) == kSides
                      && unions.size(2) == p.groups && unions.size(3) == p.tiles,
                  "unions is the contiguous int32 tile-union table [", B * H, ", 2, ", p.groups,
                  ", ", p.tiles, "], got ", shape(unions.sizes()));
  p.words = reinterpret_cast<uint32_t*>(words.mutable_data_ptr<int32_t>());
  p.unions = reinterpret_cast<uint32_t*>(unions.mutable_data_ptr<int32_t>());
  const auto exact = [&](const Tensor& t, Dtype dtype, std::initializer_list<int64_t> want,
                         const char* name) {
    const std::vector<int64_t> dims(want);
    STD_TORCH_CHECK(
        t.is_cuda() && t.scalar_type() == dtype && t.is_contiguous() && t.sizes() == Shape(dims),
        name, " is a contiguous CUDA ", dtype, " ", shape(Shape(dims)), ", got ", t.scalar_type(),
        " ", shape(t.sizes()));
    return t.mutable_data_ptr();
  };
  if (p.list_level[kWrite] >= 0)
    p.list_exp = static_cast<uint32_t*>(exact(list_exp, Dtype::Short,
                                              {B * H, p.windows, p.B[p.list_level[kWrite]], 2},
                                              "list_exp (the write list level's exponents)"));
  p.fm = static_cast<float2*>(exact(fm, Dtype::Float, {B * H, T, 2}, "fm (the factor products)"));
  p.gains16 = static_cast<uint32_t*>(exact(gains16, Dtype::Half, {B * H, T, 2}, "gains16"));
  p.gains_exp =
      static_cast<int16_t*>(exact(gains_exp, Dtype::Short, {B * H, p.windows, 2}, "gains_exp"));
  //: THE LISTS OUT, a side's at its list level: entries, starts, heads, the union by slot, its extent
  const std::vector<Tensor>* lists[kSides] = {&read_lists, &write_lists};
  for (int side = 0; side < kSides; ++side) {
    STD_TORCH_CHECK(lists[side]->size() == 5, "side ", side,
                    "'s lists are (entries, at, heads, union, count), got ", lists[side]->size(),
                    " tensors");
    if (p.list_level[side] < 0) continue;
    const int64_t width = p.B[p.list_level[side]],
                  groups = (width + kGroupDigits - 1) / kGroupDigits;
    const std::vector<Tensor>& l = *lists[side];
    ListsOut& o = p.lists[side];
    o.entries = static_cast<uint2*>(
        exact(l[0], Dtype::Short, {B * H, p.windows, width, kListCap, 4}, "a side's entries"));
    o.at = static_cast<uint16_t*>(
        exact(l[1], Dtype::Short, {B * H, p.windows, width, kListAt}, "a side's starts"));
    o.heads = static_cast<uint2*>(exact(
        l[2], Dtype::Short, {B * H, p.windows, groups, kFactsWarps, kHeadLists * kHeadEntries, 4},
        "a side's heads"));
    o.union_tok = static_cast<uint16_t*>(
        exact(l[3], Dtype::Short, {B * H, p.windows, groups, kWindowTokens}, "a side's union"));
    o.union_count = static_cast<int32_t*>(
        exact(l[4], Dtype::Int, {B * H, p.windows, groups}, "a side's union extents"));
  }
  detail::plan_units(p, dense);
  p.ledger = detail::g_ledger;

  const DeviceGuard guard(first.get_device_index());
  const int64_t units = int64_t{B} * H * p.windows * p.per_window;
  STD_TORCH_CHECK(units <= INT32_MAX, "the call's ", units, " units exceed the int32 count");
  //: ONCE: the carveout at its most shared memory, so the SM holds as many units as it can
  static bool configured = false;
  if (!configured) {
    cudaFuncAttributes attributes{};
    ROLA_CUDA_CHECK(cudaFuncGetAttributes(&attributes, detail::facts_pass));
    int most = 0;
    ROLA_CUDA_CHECK(cudaDeviceGetAttribute(&most, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                           first.get_device_index()));
    STD_TORCH_CHECK(static_cast<int>(attributes.sharedSizeBytes) + FactsSmem::kBytes <= most,
                    "the facts pass's ", FactsSmem::kBytes, " bytes of dynamic and ",
                    attributes.sharedSizeBytes, " of static shared memory exceed this device's ",
                    most, " a block");
    ROLA_CUDA_CHECK(cudaFuncSetAttribute(
        detail::facts_pass, cudaFuncAttributeMaxDynamicSharedMemorySize, FactsSmem::kBytes));
    ROLA_CUDA_CHECK(cudaFuncSetAttribute(detail::facts_pass,
                                         cudaFuncAttributePreferredSharedMemoryCarveout,
                                         cudaSharedmemCarveoutMaxShared));
    configured = true;
  }
  detail::
      facts_pass<<<static_cast<int>(units), kFactsThreads, FactsSmem::kBytes, current_stream()>>>(
          p);
  ROLA_CUDA_LAUNCH_CHECK();
}

}  // namespace facts
}  // namespace rola
