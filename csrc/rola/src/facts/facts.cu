// THE FACTS PASS'S TRANSLATION UNIT -- the kernel that reads the bundle's rows once for both sides' class-1 words,
// their tile unions, the write list level's exponents and both sides' lists, and the host entry that validates the
// bundle's views, plans the units and launches it.
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
using Clock = PhaseClock<kFactsPhases>;

//: A HELD WINDOW'S TOKEN ROW: token `t` of a warp's 32 at row `4 (t % 8) + t / 8` of its 32-digit group's 1 KB, so
//: the four token blocks a lane group reads together sit in four consecutive rows.
//: -- see docs/internals/facts/facts.md#load
__device__ __forceinline__ int ring_row(int t) { return 4 * (t & 7) + (t >> 3); }

//: A BYTE PERMUTE WITH ITS SELECTOR IMMEDIATE, and one with it in a register.
template <uint32_t Sel>
__device__ __forceinline__ uint32_t prmt_imm(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "n"(Sel));
  return d;
}

__device__ __forceinline__ uint32_t prmt_reg(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
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

//: THE ROUND-ROBIN PLACE of rank `j` of residue `r` in a pass whose residues hold the byte counts `lo` (residues 0-3)
//: and `hi` (4-7), each at most 16, `j` below 16: every residue's entries in rounds below `j`, then round `j`'s
//: residues below `r`. Bytewise `c > j` is the carry of `c + 127 - j` into bit 7; the sum of `min(c, j)` is the sum of
//: the counts at or below `j` plus `j` for each count above. -- see docs/internals/facts/facts.md#lists-out
__device__ __forceinline__ int round_robin(uint32_t lo, uint32_t hi, int j, int r) {
  const uint32_t off = 0x7f7f7f7fu - 0x01010101u * static_cast<uint32_t>(j);
  const uint32_t gl = (lo + off) & 0x80808080u, gh = (hi + off) & 0x80808080u;
  const uint32_t ml = prmt_imm<0xba98u>(gl, 0u), mh = prmt_imm<0xba98u>(gh, 0u);
  const uint32_t le = __dp4a(lo & ~ml, 0x01010101u, __dp4a(hi & ~mh, 0x01010101u, 0u));
  const uint32_t below =
      r >= 4 ? gl | (gh & ((1u << (8 * (r - 4))) - 1u)) >> 4 : gl & ((1u << (8 * r)) - 1u);
  return static_cast<int>(le) + j * __popc(gl | gh >> 4) + __popc(below);
}

//: THE M-TH SET BIT of `w` (from 0), `w` holding more than `m`: a halving search on the low half's count.
__device__ __forceinline__ int nth_bit(uint32_t w, int m) {
  int pos = 0;
#pragma unroll
  for (int s = 16; s > 0; s >>= 1) {
    const int c = __popc(w & ((1u << s) - 1u));
    const bool up = m >= c;
    m = up ? m - c : m;
    pos = up ? pos + s : pos;
    w = up ? w >> s : w;
  }
  return pos;
}

//: A UNIT'S PLACE: the span outermost, then the batch, the window and the head innermost (a token's heads lie side by
//: side in its rows, a DRAM page's bytes together in time). -- see docs/internals/facts/facts.md#the-grid
struct Unit {
  int span, b, h, bh, window;
};

__device__ __forceinline__ Unit unit_of(const FactsParams& p, int u) {
  const int per = p.units / p.spans;
  Unit x;
  x.span = u / per;
  int r = u - x.span * per;
  x.h = r % p.H;
  r /= p.H;
  x.window = r % p.windows;
  x.b = r / p.windows;
  x.bh = x.b * p.H + x.h;
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

//: THE GAINS' POWER OF TWO for a window's largest `top`: `-4 - floor(log2 top)` (the largest in [2^-4, 2^-3)), clamped
//: to [-126, 127], 0 for a zero largest. -- see docs/internals/facts/facts.md#gains
__device__ __forceinline__ int gains_shift(float top) {
  if (!(top > 0.0f)) return 0;
  return max(-126, min(127, -(static_cast<int>(__float_as_uint(top) >> 23) - 127) - 4));
}

//: A TEAM WARP'S GAINS OUT, the window's largest from all its columns (no barrier past the staging one): its 64
//: tokens' `(m, F)` as f16 pairs at the window's powers of two, warp 0 the exponents.
__device__ void store_gains(const FactsParams& p, const Unit& x, const float2* cols, int tw,
                            int lane) {
  float2 top = make_float2(0.0f, 0.0f);
#pragma unroll
  for (int i = 0; i < kWindowWords; ++i) {
    const float2 c = cols[32 * i + lane];
    top = make_float2(fmaxf(top.x, c.x), fmaxf(top.y, c.y));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    top.x = fmaxf(top.x, __shfl_xor_sync(0xffffffffu, top.x, off));
    top.y = fmaxf(top.y, __shfl_xor_sync(0xffffffffu, top.y, off));
  }
  const int sm = gains_shift(top.x), sf = gains_shift(top.y);
#pragma unroll
  for (int k = 0; k < kBlocks; ++k) {
    const int i = kTokBits * (kWindowWords / kTeamWarps * tw + k) + lane,
              t = x.window * kWindowTokens + i;
    if (t < p.L)
      p.gains16[static_cast<int64_t>(x.bh) * p.L + t] =
          static_cast<uint32_t>(__half_as_ushort(__float2half_rn(cols[i].x * pow2(sm))))
          | static_cast<uint32_t>(__half_as_ushort(__float2half_rn(cols[i].y * pow2(sf)))) << 16;
  }
  if (tw == 0 && lane < 2)
    p.gains_exp[(static_cast<int64_t>(x.bh) * p.windows + x.window) * 2 + lane] =
        static_cast<int16_t>(lane ? sf : sm);
}

//: A TEAM WARP'S COPIES OF A WRITE HALF into the held window: its blocks' tokens' runs of the half's 64 digits, block
//: `j` at `j x 2 KB`, a run's 16-byte quanta at the token's row of their 32-digit group (`ring_row`), bytes past the
//: rows' width or a token past `L` zero-filled and unread. -- see docs/internals/facts/facts.md#load
__device__ __forceinline__ void issue_half(const FactsParams& p, const Unit& x, uint32_t code,
                                           uint8_t* held, int tw, int lane) {
  const RowView v = p.rows[kWrite][plan_level(code)];
  const int digit0 = kGroupDigits * plan_group0(code);
  const uint8_t* const rows = static_cast<const uint8_t*>(v.base) + x.b * v.sb + x.h * v.sh;
#pragma unroll
  for (int jj = 0; jj < kBlocks; ++jj) {
    const int j = kBlocks * tw + jj;
    const ops::SmemAddr dst = ops::smem_addr(held + j * kHalfRegion);
#pragma unroll
    for (int q = 0; q < kHalfQuanta; ++q) {
      const int quantum = 32 * q + lane, token = quantum / 4, part = quantum % 4;
      const int t = x.window * kWindowTokens + kTokBits * j + token;
      const uint32_t run =
          t < p.L ? static_cast<uint32_t>(max(0, min(16, v.width - digit0 - 16 * part))) : 0u;
      ops::stage_tile_n(dst + (part >> 1) * 1024 + ring_row(token) * 32 + 16 * (part & 1),
                        rows + static_cast<int64_t>(t) * v.st + digit0 + 16 * part, run);
    }
  }
}

__device__ __forceinline__ int stages_of(uint32_t code) {
  return (plan_kind(code) == kItemDenseMax ? 2 : 1) * kBlocks * kTokBits / kStageTokens;
}

//: A TEAM WARP'S PLACE IN ONE OF ITS STREAMS (its team's ring items, or the held window's halves): the block's
//: `n`-th unit `u`, its plan entry `entry` and that entry's `code`, a ring item's stage `stage`; past the block's last
//: unit `valid` is false. A ring item's copies are prepared once an item: the lane's first quantum's source and token
//: at the next stage, a stage's step (the second quantum half a stage on), the lane's run (bytes inside the width).
struct Stream {
  int n, u, entry, stage, t;
  bool valid;
  uint32_t code, run;
  const uint8_t* src;
  int64_t step;
};

__device__ __forceinline__ int quanta_of(uint32_t code) {
  return kSpanDigits * (plan_kind(code) == kItemDenseMax ? 2 : 1) / 16;
}

//: A TEAM WARP'S RING ITEM COPIES, prepared: its tokens' runs of the span, 8 tokens of u8 a stage (a run 128 bytes,
//: 8 lanes a run) or 4 of a dense list level's f16 (256, 16 lanes), zero-filled past the width and `L`.
__device__ __forceinline__ void prepare_ring(const FactsParams& p, Stream& c, int tw, int lane) {
  const Unit x = unit_of(p, c.u);
  const int quanta = quanta_of(c.code), bytes = quanta / 8;
  const RowView v = p.rows[plan_side(c.code)][plan_level(c.code)];
  const int digit0 = kGroupDigits * plan_group0(c.code), part = lane % quanta;
  c.t = x.window * kWindowTokens + kTokBits * kBlocks * tw + lane / quanta;
  c.src = static_cast<const uint8_t*>(v.base) + x.b * v.sb + x.h * v.sh
          + static_cast<int64_t>(c.t) * v.st + digit0 * bytes + 16 * part;
  c.step = kStageBytes / (kSpanDigits * bytes) * v.st;
  c.run = static_cast<uint32_t>(max(0, min(16, (v.width - digit0) * bytes - 16 * part)));
}

__device__ __forceinline__ bool unit_valid(const FactsParams& p, int n) {
  return static_cast<int>(blockIdx.x) + n * static_cast<int>(gridDim.x) < p.units;
}

//: the first entry at or after (`n`, `entry`) of team `team`'s stream of its kind (`kHalf` the write halves, else
//: the ring items)
template <bool kHalf>
__device__ __forceinline__ void seek(const FactsParams& p, Stream& c, int team, int tw, int lane) {
  c.stage = 0;
#pragma unroll 1
  while (c.valid) {
    const int span = c.u / (p.units / p.spans);
#pragma unroll 1
    for (const int items = p.plan_items[span]; c.entry < items; ++c.entry) {
      c.code = p.plan[span][c.entry];
      if ((plan_kind(c.code) == kItemHalf) == kHalf && plan_team(c.code) == team) {
        if constexpr (!kHalf) prepare_ring(p, c, tw, lane);
        return;
      }
    }
    ++c.n;
    c.entry = 0;
    c.u = static_cast<int>(blockIdx.x) + c.n * static_cast<int>(gridDim.x);
    c.valid = c.u < p.units;
  }
}

template <bool kHalf>
__device__ __forceinline__ Stream first_of(const FactsParams& p, int team, int tw, int lane) {
  Stream c{};
  c.u = static_cast<int>(blockIdx.x);
  c.valid = c.u < p.units;
  seek<kHalf>(p, c, team, tw, lane);
  return c;
}

//: A RING STAGE'S FLAGS: lane `(g, dq)` = (lane / 8, lane % 8) digits `4 dq ..` of group `g`, the stage's 8 tokens one
//: byte of flags a digit (bit `i` token `i`). -- see docs/internals/facts/facts.md#vote
__device__ __forceinline__ uint32_t ring_flags(const uint8_t* slot, int lane) {
  uint32_t acc = 0u;
#pragma unroll
  for (int i = 0; i < kStageTokens; ++i)
    acc |= nonzero_bytes(*reinterpret_cast<const uint32_t*>(slot + i * kSpanDigits + 4 * lane))
           >> (7 - i);
  return acc;
}

//: A DENSE LIST LEVEL'S STAGE: lane `l` digits `4 l ..` of the span as f16, each to the byte scale (x 255, exact), its
//: largest `b m` and `b F` over the stage's 4 tokens.
__device__ __forceinline__ void dense_stage(const uint8_t* slot, const float2* cols, int lane,
                                            float2 (&best)[4]) {
#pragma unroll
  for (int i = 0; i < kDenseStageTokens; ++i) {
    const uint2 w = *reinterpret_cast<const uint2*>(slot + i * 2 * kSpanDigits + 8 * lane);
    const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&w.x));
    const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&w.y));
    const float b[4] = {kRowMaxDigit * lo.x, kRowMaxDigit * lo.y, kRowMaxDigit * hi.x,
                        kRowMaxDigit * hi.y};
    const float2 c = cols[i];
#pragma unroll
    for (int k = 0; k < 4; ++k)
      best[k] = make_float2(fmaxf(best[k].x, b[k] * c.x), fmaxf(best[k].y, b[k] * c.y));
  }
}

//: A WRITE HALF'S VOTE by a team warp, its blocks from the held window a 32-digit group at a time: lane `(tk, dq)`
//: token `8 tk + i`'s digits `4 dq ..` as one word, each byte's nonzero bit at bit `i`, a 4 x 4 byte transpose across
//: the token blocks leaving lane `(tk, dq)` digit `4 dq + tk`'s 32 tokens, masked; into the half's tile and union
//: words.
__device__ __forceinline__ void half_vote(const FactsParams& p, uint32_t code, const uint8_t* held,
                                          uint32_t* tile, uint32_t* live, int tw, int lane) {
  const int tk = lane >> 3, dq = lane & 7, groups = plan_groups(code), level = plan_level(code);
  const int digit0 = kGroupDigits * plan_group0(code);
#pragma unroll
  for (int g = 0; g < kHalfGroups; ++g) {
    if (g >= groups) break;
    const int d = 4 * dq + tk;
    const bool in = digit_in(p, kWrite, level, digit0 + kGroupDigits * g + d);
#pragma unroll
    for (int jj = 0; jj < kBlocks; ++jj) {
      const int j = kBlocks * tw + jj;
      const uint8_t* const region = held + j * kHalfRegion + g * 1024;
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
      flags = in ? flags : 0u;
      tile[tile_at(kGroupDigits * g + d, j)] = flags;
      const uint32_t any = __reduce_or_sync(0xffffffffu, flags);
      if (lane == 0) live[g * kWindowWords + j] = any;
    }
  }
}

//: AN ITEM'S BITS OUT of its tile by a team: each digit's window of words one 64-byte run (two digits a store, a
//: half-warp each), then each (group, 64-token tile)'s union, a digit a lane, one ballot.
//: -- see docs/internals/facts/facts.md#tiles
__device__ void bits_out(const FactsParams& p, const Unit& x, uint32_t code, const uint32_t* tile,
                         int tw, int lane) {
  const int side = plan_side(code), level = plan_level(code), group0 = plan_group0(code),
            groups = plan_groups(code);
  const int word0 = x.window * kWindowWords, nwords = min(kWindowWords, p.token_words - word0),
            j = lane & 15;
  uint32_t* const out = p.words
                        + ((static_cast<int64_t>(x.bh) * kSides + side) * p.rows_total
                           + p.row_base[level] + kGroupDigits * group0)
                              * p.token_words
                        + word0 + j;
#pragma unroll 1
  for (int r = 2 * tw + (lane >> 4); r < kGroupDigits * groups; r += 2 * kTeamWarps)
    if (kGroupDigits * group0 + r < p.B[level] && j < nwords)
      out[static_cast<int64_t>(r) * p.token_words] = tile[tile_at(r, j)];
#pragma unroll 1
  for (int pi = tw; pi < groups * (kWindowWords / 2); pi += kTeamWarps) {
    const int g = pi / (kWindowWords / 2), t = pi % (kWindowWords / 2), r = kGroupDigits * g + lane;
    const uint32_t any =
        __ballot_sync(0xffffffffu, (tile[tile_at(r, 2 * t)] | tile[tile_at(r, 2 * t + 1)]) != 0u);
    const int tile_index = x.window * (kWindowWords / 2) + t;
    if (lane == 0 && tile_index < p.tiles)
      p.unions[((static_cast<int64_t>(x.bh) * kSides + side) * p.groups + p.group_base[level]
                + group0 + g)
                   * p.tiles
               + tile_index] = any;
  }
}

//: ONE 32-DIGIT GROUP'S LISTS by its four warps, `part` a quarter of its digits: the union's tokens, a walk totalling
//: each digit's passes by residue, a walk placing each entry round-robin and writing it (a write entry with its
//: coefficient pair). -- see docs/internals/facts/facts.md#lists-out
template <bool kCoef>
__device__ void build(const FactsParams& p, const Unit& x, int side, int level, int group, int part,
                      int lane, const uint32_t* tile, const uint32_t* live, const uint8_t* held,
                      const float2* cols, uint32_t* top, uint4* table, uint32_t* marks) {
  constexpr int kWords = kWindowWords / kParts, kLanesAWord = 32 / kWords;
  const int width = p.B[level], groups = (width + kGroupDigits - 1) / kGroupDigits;
  const ListsOut out = p.lists[side];
  const int64_t win = static_cast<int64_t>(x.bh) * p.windows + x.window;
  uint32_t* const cell_mask = marks;
  uint16_t* const bases = reinterpret_cast<uint16_t*>(marks + kWords);

  //: the union, lane `j` holding its word `un` and base `ub`; part `part` stores its words' tokens
  const uint32_t un = lane < kWindowWords ? live[lane] : 0u;
  uint32_t ub = __popc(un);
#pragma unroll
  for (int off = 1; off < kWindowWords; off <<= 1) {
    const uint32_t y = __shfl_up_sync(0xffffffffu, ub, off);
    ub += lane >= off ? y : 0u;
  }
  const uint32_t union_count = __shfl_sync(0xffffffffu, ub, kWindowWords - 1);
  ub -= __popc(un);
  if (lane < kWindowWords) bases[lane] = static_cast<uint16_t>(ub);
  uint16_t* const tokens = out.union_tok + (win * groups + group) * kWindowTokens;
#pragma unroll
  for (int k = 0; k < kWords; ++k) {
    const int j = part * kWords + k;
    const uint32_t u = __shfl_sync(0xffffffffu, un, j), b = __shfl_sync(0xffffffffu, ub, j);
    if ((u >> lane) & 1u)
      tokens[b + __popc(u & ((1u << lane) - 1u))] = static_cast<uint16_t>(kTokBits * j + lane);
  }
  if (part == 0 && lane == 0)
    out.union_count[win * groups + group] = static_cast<uint16_t>(union_count);

  //: the lane's cells: their counts' running sums a byte each (`pre`), which are nonempty (into the warp's cell mask)
  const int q = lane % kParts, dl = lane / kParts;
  uint32_t pre = 0u, nonempty = 0u;
  int n = 0;
#pragma unroll
  for (int i = 0; i < kWords; ++i) {
    pre |= static_cast<uint32_t>(n) << (8 * i);
    const int c = __popc(tile[tile_at(part * kListDigits + dl, q * kWords + i)]);
    n += c;
    nonempty |= c != 0 ? 1u << i : 0u;
  }
#pragma unroll
  for (int w = 0; w < kWords; ++w) {
    const uint32_t v = __reduce_or_sync(
        0xffffffffu, lane / kLanesAWord == w ? nonempty << (lane % kLanesAWord * kWords) : 0u);
    if (lane == 0) cell_mask[w] = v;
  }
  int first = n;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, first, off);
    first += lane >= off ? y : 0;
  }
  const int total = __shfl_sync(0xffffffffu, first, 31);
  first -= n;
  table[lane] = make_uint4(0u, 0u, 0u, 0u);
  __syncwarp();

  //: the lane's share and the cell and bit it starts at
  const int share = lane * total >> 5, count = ((lane + 1) * total >> 5) - share;
  int c0 = 0;
  uint32_t bits0 = 0u, slot0 = 0u;
  if (total > 0) {
    const int g = min(share, total - 1);
    int o = 0;
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
      const int y = __shfl_sync(0xffffffffu, first, o + s);
      o = y <= g ? o + s : o;
    }
    int m = g - __shfl_sync(0xffffffffu, first, o);
    const uint32_t sums = __shfl_sync(0xffffffffu, pre, o);
    const int w =
        __popc((((static_cast<uint32_t>(m) * 0x01010101u) | 0x80808080u) - sums) & 0x80808080u) - 1;
    m -= static_cast<int>((sums >> (8 * w)) & 0xffu);
    c0 = o * kWords + w;
    const uint32_t word = tile[tile_at(part * kListDigits + (c0 >> 4), c0 & 15)];
    const int t0 = nth_bit(word, m);
    bits0 = word & (0xffffffffu << t0);
    slot0 = bases[c0 & 15] + __popc(live[c0 & 15] & ((1u << t0) - 1u));
  }
  __syncwarp();
  const int key0 = (c0 >> 4) * kMaxPasses + static_cast<int>(slot0) / kPassSlots;
  //: the lane's `count` entries from its start, each `(cell, bit, slot)`
  const auto walk = [&](auto&& visit) {
    int c = c0;
    uint32_t bits = bits0, u = live[c0 & 15], b = bases[c0 & 15];
#pragma unroll 1
    for (int e = 0; e < count; ++e) {
      if (bits == 0u) {
        int y = c + 1;
        uint32_t mw = cell_mask[y >> 5] & (0xffffffffu << (y & 31));
        while (mw == 0u) {
          y = (y | 31) + 1;
          mw = cell_mask[y >> 5];
        }
        c = (y & ~31) + __ffs(mw) - 1;
        bits = tile[tile_at(part * kListDigits + (c >> 4), c & 15)];
        u = live[c & 15];
        b = bases[c & 15];
      }
      const int t = __ffs(bits) - 1;
      bits &= bits - 1u;
      visit(c, t, b + __popc(u & ((1u << t) - 1u)));
    }
  };

  //: the first walk: each entry counted into the table, the lane's last (digit, pass) run's counts left in `run`; on
  //: the write side each digit's maxima, a lane's run of one digit joined by one atomic a run
  int cur = key0, cur_dl = c0 >> 4;
  bool many = false;
  uint32_t run_lo = 0u, run_hi = 0u;
  float run_m = 0.0f, run_f = 0.0f;
  const auto flush = [&] {
    atomicMax(top + 2 * (part * kListDigits + cur_dl), __float_as_uint(run_m));
    atomicMax(top + 2 * (part * kListDigits + cur_dl) + 1, __float_as_uint(run_f));
  };
  walk([&](int c, int t, uint32_t slot) {
    if constexpr (kCoef) {
      const int d = c >> 4, j = c & 15;
      const float byte =
          static_cast<float>(held[j * kHalfRegion + ring_row(t) * 32 + part * kListDigits + d]);
      const float2 col = cols[kTokBits * j + t];
      if (d != cur_dl) {
        flush();
        run_m = run_f = 0.0f;
        cur_dl = d;
      }
      run_m = fmaxf(run_m, byte * col.x);
      run_f = fmaxf(run_f, byte * col.y);
    }
    const int key = (c >> 4) * kMaxPasses + static_cast<int>(slot) / kPassSlots,
              r = static_cast<int>(slot & 7u);
    const uint32_t inc = 1u << (8 * (r & 3));
    atomicAdd(reinterpret_cast<uint32_t*>(table + key) + (r >> 2), inc);
    const bool same = key == cur;
    many |= !same;
    cur = key;
    run_lo = (same ? run_lo : 0u) + (r < 4 ? inc : 0u);
    run_hi = (same ? run_hi : 0u) + (r < 4 ? 0u : inc);
  });
  if constexpr (kCoef) {
    if (count > 0) flush();
  }
  //: the counts before each lane's first run: a scan of the last runs' counts, segmented where the run changes
  const int cur_before = __shfl_up_sync(0xffffffffu, cur, 1);
  uint32_t v_lo = run_lo, v_hi = run_hi;
  int head = many || lane == 0 || cur_before != cur;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const uint32_t xl = __shfl_up_sync(0xffffffffu, v_lo, off),
                   xh = __shfl_up_sync(0xffffffffu, v_hi, off);
    const int xf = __shfl_up_sync(0xffffffffu, head, off);
    if (lane >= off) {
      v_lo += head ? 0u : xl;
      v_hi += head ? 0u : xh;
      head |= xf;
    }
  }
  const bool carried = lane > 0 && cur_before == key0;
  const uint32_t before_lo = __shfl_up_sync(0xffffffffu, v_lo, 1),
                 before_hi = __shfl_up_sync(0xffffffffu, v_hi, 1);
  __syncwarp();

  //: a lane a digit's pass: its start from the digit's totals, the list's starts and end out, on the write side the
  //: digit's exponents (pass 0's lane) and its coefficient scales
  {
    const int key = lane, pass = key % kMaxPasses, d = key / kMaxPasses;
    const uint4 c = table[key];
    const int s = static_cast<int>(__dp4a(c.x, 0x01010101u, __dp4a(c.y, 0x01010101u, 0u)));
    int incl = s;
#pragma unroll
    for (int off = 1; off < kMaxPasses; off <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, incl, off);
      incl += pass >= off ? y : 0;
    }
    const int end = __shfl_sync(0xffffffffu, incl, lane | (kMaxPasses - 1));
    const int digit = group * kGroupDigits + part * kListDigits + d;
    if (digit < width) {
      uint16_t* const at = out.at + (win * width + digit) * kListAt;
      at[pass] = static_cast<uint16_t>(incl - s);
      at[kMaxPasses + pass] = static_cast<uint16_t>(end);
    }
    uint32_t scales = 0u;
    if constexpr (kCoef) {
      const int km = ceil_exponent(__uint_as_float(top[2 * (part * kListDigits + d)]));
      const int kv = ceil_exponent(__uint_as_float(top[2 * (part * kListDigits + d) + 1]));
      if (pass == 0 && digit < width)
        p.list_exp[win * width + digit] = static_cast<uint32_t>(static_cast<uint16_t>(km))
                                          | static_cast<uint32_t>(static_cast<uint16_t>(kv)) << 16;
      scales = static_cast<uint32_t>(kCoefTop - max(km, kUnitMin) + 128)
               | static_cast<uint32_t>(kCoefTop - max(kv, kUnitMin) + 128) << 8;
    }
    table[key].z = static_cast<uint32_t>(incl - s);
    table[key].w = scales;
  }
  __syncwarp();

  //: the second walk: each entry's rank among its residue from the run's counts so far, its place, its record
  cur = key0;
  run_lo = carried ? before_lo : 0u;
  run_hi = carried ? before_hi : 0u;
  walk([&](int c, int t, uint32_t slot) {
    const int key = (c >> 4) * kMaxPasses + static_cast<int>(slot) / kPassSlots,
              r = static_cast<int>(slot & 7u);
    const uint4 tc = table[key];
    const bool same = key == cur;
    cur = key;
    run_lo = same ? run_lo : 0u;
    run_hi = same ? run_hi : 0u;
    const int rank = static_cast<int>(((r < 4 ? run_lo : run_hi) >> (8 * (r & 3))) & 0xffu);
    const uint32_t inc = 1u << (8 * (r & 3));
    run_lo += r < 4 ? inc : 0u;
    run_hi += r < 4 ? 0u : inc;
    const int d = c >> 4, j = c & 15;
    const int64_t at = (win * width + group * kGroupDigits + part * kListDigits + d) * kListCap
                       + static_cast<int>(tc.z) + round_robin(tc.x, tc.y, rank, r);
    if constexpr (kCoef) {
      const float byte =
          static_cast<float>(held[j * kHalfRegion + ring_row(t) * 32 + part * kListDigits + d]);
      const float2 col = cols[kTokBits * j + t];
      const float sm = pow2(static_cast<int>(tc.w & 0xffu) - 128),
                  sv = pow2(static_cast<int>(tc.w >> 8) - 128);
      const uint32_t cv = __half_as_ushort(__float2half_rn(byte * col.y * sv));
      const uint32_t cm = __half_as_ushort(__float2half_rn(byte * col.x * sm));
      out.record[at] = make_uint2(slot, cv | cm << 16);
    } else {
      out.slot[at] = static_cast<uint16_t>(slot);
    }
  });
}

//: THE GRID: a block of sixteen warps an SM, two teams of eight, each its own pipeline over the block's units by a
//: stride -- team 0 the write list level (its halves from the held window, or a dense list level's span for its
//: maxima) with the window's columns and gains, team 1 the other levels' ring items; a team's items in plan order, one
//: team barrier after each vote and, for a half, one after its build, which frees the held window for the next half.
//: The teams share nothing but the SM: one's builds run under the other's stream.
//: -- see docs/internals/facts/facts.md#the-grid
__global__ __launch_bounds__(kFactsThreads, kFactsBlocksPerSm) void facts_pass(
    const FactsParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* const held = smem + FactsSmem::kHeld;
  float2* const cols = reinterpret_cast<float2*>(smem + FactsSmem::kCols);
  uint32_t* const half_tiles = reinterpret_cast<uint32_t*>(smem + FactsSmem::kHalfTiles);
  uint32_t* const half_live = reinterpret_cast<uint32_t*>(smem + FactsSmem::kHalfLive);
  uint32_t* const top = reinterpret_cast<uint32_t*>(smem + FactsSmem::kTop);
  uint32_t* const span_tiles = reinterpret_cast<uint32_t*>(smem + FactsSmem::kSpanTiles);
  uint32_t* const span_live = reinterpret_cast<uint32_t*>(smem + FactsSmem::kSpanLive);
  uint32_t* const dense_top = reinterpret_cast<uint32_t*>(smem + FactsSmem::kDenseTop);
  __shared__ __align__(8) unsigned long long full_held, full_ring[kFactsWarps][kRingDepth];
  __shared__ long long s_clock[kFactsWarps][Clock::kRow];
  const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  const int team = warp / kTeamWarps, tw = warp % kTeamWarps,
            tthread = static_cast<int>(threadIdx.x) % kTeamThreads;
  //: a team's ring: team 0's in the held window (only a dense list level, which has no halves, streams there)
  uint8_t* const ring =
      (team == 0 ? held : smem + FactsSmem::kRing) + tw * kRingDepth * kStageBytes;
  uint4* const table = reinterpret_cast<uint4*>(smem + FactsSmem::kTables) + warp * kListTableKeys;
  uint32_t* const marks = reinterpret_cast<uint32_t*>(smem + FactsSmem::kMarks) + warp * 12;
  Clock clock(&s_clock[warp][0], lane);
  const auto team_sync = [&] { ops::rendezvous_group(1 + team, kTeamThreads); };
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < kRingDepth; ++r) ops::mbar_init(ops::smem_addr(&full_ring[warp][r]), 32);
  }
  if (threadIdx.x == 0) ops::mbar_init(ops::smem_addr(&full_held), kTeamThreads);
#pragma unroll 1
  for (int i = static_cast<int>(threadIdx.x); i < 2 * kHalfDigits; i += kFactsThreads) top[i] = 0u;
#pragma unroll 1
  for (int i = static_cast<int>(threadIdx.x); i < 4 * kSpanDigits; i += kFactsThreads)
    dense_top[i] = 0u;
  __syncthreads();

  //: THE STREAMS' FIRST COPIES: each warp's ring's first stages, team 0's first write half
  Stream fill_at = first_of<false>(p, team, tw, lane);
  const ops::SmemAddr ring_at = ops::smem_addr(ring);
  const auto fill_ring = [&](int slot) {
    if (fill_at.valid) {
      Stream& c = fill_at;
      const int tokens = kStageBytes / 16 / quanta_of(c.code);
      const ops::SmemAddr dst = ring_at + slot * kStageBytes + 16 * lane;
      ops::stage_tile_n(dst, c.src, c.t < p.L ? c.run : 0u);
      ops::stage_tile_n(dst + kStageBytes / 2, c.src + (c.step >> 1),
                        c.t + tokens / 2 < p.L ? c.run : 0u);
      c.src += c.step;
      c.t += tokens;
      if (++c.stage == stages_of(c.code)) {
        ++c.entry;
        seek<false>(p, c, team, tw, lane);
      }
    }
    ops::mbar_arrive_when_landed(ops::smem_addr(&full_ring[warp][slot]));
  };
  Stream half_at = first_of<true>(p, 0, tw, lane);
  const auto fill_held = [&] {
    if (half_at.valid) {
      issue_half(p, unit_of(p, half_at.u), half_at.code, held, tw, lane);
      ++half_at.entry;
      seek<true>(p, half_at, 0, tw, lane);
    }
    ops::mbar_arrive_when_landed(ops::smem_addr(&full_held));
  };
#pragma unroll
  for (int s = 0; s < kRingDepth; ++s) fill_ring(s);
  if (team == 0) fill_held();
  int n_ring = 0, n_held = 0, n_dense = 0, n_items = 0;
  const auto take_stage = [&] {
    const int slot = n_ring % kRingDepth;
    ops::mbar_wait_sleep(ops::smem_addr(&full_ring[warp][slot]),
                         static_cast<uint32_t>(n_ring / kRingDepth & 1));
    __syncwarp();
    return slot;
  };
  const auto release_stage = [&](int slot) {
    __syncwarp();
    fill_ring(slot);
    ++n_ring;
  };

#pragma unroll 1
  for (int n = 0; unit_valid(p, n); ++n) {
    const Unit x = unit_of(p, static_cast<int>(blockIdx.x) + n * static_cast<int>(gridDim.x));
    const int items = p.plan_items[x.span];
    bool mine = false;
#pragma unroll 1
    for (int i = 0; i < items; ++i) mine |= plan_team(p.plan[x.span][i]) == team;
    if (!mine) continue;
    const bool columns = team == 0 && p.list_level >= 0;
    if (columns) {
#pragma unroll
      for (int k = 0; k < kColumnPasses; ++k) {
        const int i = tthread + kTeamThreads * k, t = x.window * kWindowTokens + i;
        const float2 c = load_column(p, x, i);
        cols[i] = c;
        if (x.span == 0 && t < p.L)
          p.fm[static_cast<int64_t>(x.bh) * p.L + t] = make_float2(c.y, c.x);
      }
    }
    clock.lap(kPhaseColumns);
    bool first = true;
#pragma unroll 1
    for (int i = 0; i < items; ++i) {
      const uint32_t code = p.plan[x.span][i];
      if (plan_team(code) != team) continue;
      const int kind = plan_kind(code), side = plan_side(code), level = plan_level(code);
      const int group0 = plan_group0(code), groups = plan_groups(code);
      const bool gains = first && columns && x.span == 0;
      first = false;
      if (kind == kItemHalf) {
        uint32_t* const tile = half_tiles + (n_held & 1) * kHalfDigits * kTileWords;
        if (tthread < 2 * kHalfDigits) top[tthread] = 0u;
        ops::mbar_wait_sleep(ops::smem_addr(&full_held), static_cast<uint32_t>(n_held++ & 1));
        __syncwarp();
        half_vote(p, code, held, tile, half_live, tw, lane);
        clock.lap(kPhaseVote);
        team_sync();
        if (gains) store_gains(p, x, cols, tw, lane);
        const int g = tw / kParts, part = tw % kParts;
        if (g < groups)
          build<true>(p, x, kWrite, level, group0 + g, part, lane,
                      tile + kGroupDigits * g * kTileWords, half_live + g * kWindowWords,
                      held + g * 1024, cols, top + 2 * kGroupDigits * g, table, marks);
        clock.lap(kPhaseLists);
        //: the held window and the maxima free: the next half's copies, then this one's bits (its tile alternates)
        team_sync();
        fill_held();
        bits_out(p, x, code, tile, tw, lane);
        clock.lap(kPhaseStore);
      } else if (kind == kItemDenseMax) {
        float2 best[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) best[k] = make_float2(0.0f, 0.0f);
#pragma unroll 1
        for (int st = 0; st < stages_of(code); ++st) {
          const int slot = take_stage();
          dense_stage(ring + slot * kStageBytes,
                      cols + kTokBits * kBlocks * tw + kStageBytes / (2 * kSpanDigits) * st, lane,
                      best);
          release_stage(slot);
        }
        uint32_t* const maxima = dense_top + (n_dense & 1) * 2 * kSpanDigits;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          if (best[k].x > 0.0f) atomicMax(maxima + 2 * (4 * lane + k), __float_as_uint(best[k].x));
          if (best[k].y > 0.0f)
            atomicMax(maxima + 2 * (4 * lane + k) + 1, __float_as_uint(best[k].y));
        }
        clock.lap(kPhaseVote);
        team_sync();
        if (gains) store_gains(p, x, cols, tw, lane);
        if (tthread < kSpanDigits) {
          const int digit = kGroupDigits * group0 + tthread;
          const int km = ceil_exponent(__uint_as_float(maxima[2 * tthread]));
          const int kv = ceil_exponent(__uint_as_float(maxima[2 * tthread + 1]));
          maxima[2 * tthread] = maxima[2 * tthread + 1] = 0u;
          if (tthread < kGroupDigits * groups && digit < p.B[level])
            p.list_exp[(static_cast<int64_t>(x.bh) * p.windows + x.window) * p.B[level] + digit] =
                static_cast<uint32_t>(static_cast<uint16_t>(km))
                | static_cast<uint32_t>(static_cast<uint16_t>(kv)) << 16;
        }
        ++n_dense;
        clock.lap(kPhaseStore);
        clock.lap(kPhaseLists);
      } else {
        uint32_t* const tile = span_tiles + (n_items & 1) * kSpanDigits * kTileWords;
        uint32_t* const live =
            span_live + (n_items & 1) * kSpanDigits / kGroupDigits * kWindowWords;
        uint32_t v[kBlocks][kTokBits / kStageTokens];
#pragma unroll
        for (int jj = 0; jj < kBlocks; ++jj)
#pragma unroll
          for (int st = 0; st < kRingStages; ++st) {
            const int slot = take_stage();
            v[jj][st] = ring_flags(ring + slot * kStageBytes, lane);
            release_stage(slot);
          }
        const int g = lane >> 3, dq = lane & 7;
#pragma unroll
        for (int jj = 0; jj < kBlocks; ++jj) {
          uint32_t any = 0u;
#pragma unroll
          for (int dd = 0; dd < 4; ++dd) {
            const uint32_t sel = static_cast<uint32_t>(dd) | static_cast<uint32_t>(4 + dd) << 4;
            uint32_t f = prmt_imm<0x5410u>(prmt_reg(v[jj][0], v[jj][1], sel),
                                           prmt_reg(v[jj][2], v[jj][3], sel));
            const int d = kGroupDigits * g + 4 * dq + dd;
            f = digit_in(p, side, level, kGroupDigits * group0 + d) ? f : 0u;
            tile[tile_at(d, kBlocks * tw + jj)] = f;
            any |= f;
          }
          any |= __shfl_xor_sync(0xffffffffu, any, 1);
          any |= __shfl_xor_sync(0xffffffffu, any, 2);
          any |= __shfl_xor_sync(0xffffffffu, any, 4);
          if (kind == kItemReadList && dq == 0) live[g * kWindowWords + kBlocks * tw + jj] = any;
        }
        clock.lap(kPhaseVote);
        team_sync();
        bits_out(p, x, code, tile, tw, lane);
        clock.lap(kPhaseStore);
        if (kind == kItemReadList) {
#pragma unroll 1
          for (int k = 0; k < kSpanDigits / kGroupDigits * kParts / kTeamWarps; ++k) {
            const int gg = tw / kParts + kTeamWarps / kParts * k, part = tw % kParts;
            if (gg < groups)
              build<false>(p, x, kRead, level, group0 + gg, part, lane,
                           tile + kGroupDigits * gg * kTileWords, live + gg * kWindowWords, nullptr,
                           cols, top, table, marks);
          }
        }
        clock.lap(kPhaseLists);
        ++n_items;
      }
    }
    //: the next unit's columns wait for this one's readers
    if (columns) team_sync();
  }
  clock.publish(p.ledger, 0, kFactsWarps, warp);
}

//: A level's rows as the kernel reads them: `[B, T, H, width]`, digits contiguous -- u8, or f16 for a dense write
//: level (`half`); the view's strides in bytes.
RowView row_view(const Tensor& t, int64_t B, int64_t T, int64_t H, bool half, const char* name) {
  STD_TORCH_CHECK(t.is_cuda() && t.scalar_type() == (half ? Dtype::Half : Dtype::Byte), name,
                  " is a CUDA ", half ? "f16 (a dense write level)" : "u8", " tensor, got ",
                  t.scalar_type());
  const int64_t size = t.element_size();
  STD_TORCH_CHECK(t.dim() == 4 && t.size(0) == B && t.size(1) == T && t.size(2) == H, name,
                  " is [B, T, H, width] = [", B, ", ", T, ", H, width], got ", shape(t.sizes()));
  STD_TORCH_CHECK(t.stride(3) == 1 || t.size(3) == 1, name, "'s digits are contiguous");
  //: THE ROW ALIGNMENT LAW: the first row and every stride of a dimension longer than one on 32 bytes
  const bool aligned = reinterpret_cast<uintptr_t>(t.const_data_ptr()) % kRowAlign == 0
                       && (t.size(0) == 1 || t.stride(0) * size % kRowAlign == 0)
                       && (t.size(1) == 1 || t.stride(1) * size % kRowAlign == 0)
                       && (t.size(2) == 1 || t.stride(2) * size % kRowAlign == 0);
  STD_TORCH_CHECK(aligned, name,
                  " breaks the row alignment law: its rows start on and step by multiples of ",
                  kRowAlign, " bytes");
  return RowView{t.const_data_ptr(), t.stride(0) * size, t.stride(1) * size, t.stride(2) * size,
                 static_cast<int>(t.size(3))};
}

ColumnView column_view(const Tensor& t, int64_t B, int64_t T, int64_t H, const char* name) {
  STD_TORCH_CHECK(t.is_cuda() && t.scalar_type() == Dtype::Float, name,
                  " is a CUDA fp32 tensor, got ", t.scalar_type());
  STD_TORCH_CHECK(t.dim() == 3 && t.size(0) == B && t.size(1) == T && t.size(2) == H, name,
                  " is [B, T, H] = [", B, ", ", T, ", ", H, "], got ", shape(t.sizes()));
  return ColumnView{t.const_data_ptr<float>(), t.stride(0), t.stride(1), t.stride(2)};
}

//: THE PLAN, a span's items in order: the write list level's halves in the held window (a sparse list level) or its
//: span for its maxima (a dense one), team 0's; the read list level's span and every other sparse level's span of
//: either side, team 1's. -- see docs/internals/facts/facts.md#the-plan
void plan_units(FactsParams& p) {
  const auto dense = [&](int side, int level) { return level_is_dense(p.dense[side], level); };
  const bool halves = p.list_level >= 0 && !dense(kWrite, p.list_level);
  int widest = 0;
  for (int side = 0; side < kSides; ++side)
    for (int level = 0; level < p.D; ++level)
      if (!dense(side, level) || (side == kWrite && level == p.list_level))
        widest = std::max(widest, p.B[level]);
  p.spans = std::max(1, (widest + kSpanDigits - 1) / kSpanDigits);

  const auto groups_from = [](int width, int digit0, int most) {
    return std::min(most, (width - digit0 + kGroupDigits - 1) / kGroupDigits);
  };

  for (int s = 0; s < p.spans; ++s) {
    std::vector<uint32_t> half, ring;
    for (int h = 0; halves && h < kSpanDigits / kHalfDigits; ++h) {
      const int digit0 = s * kSpanDigits + h * kHalfDigits;
      if (digit0 < p.B[p.list_level])
        half.push_back(plan_code(kItemHalf, kWrite, p.list_level, digit0 / kGroupDigits,
                                 groups_from(p.B[p.list_level], digit0, kHalfDigits / kGroupDigits),
                                 0));
    }
    const auto span_item = [&](int kind, int side, int level) {
      const int digit0 = s * kSpanDigits;
      if (digit0 < p.B[level])
        ring.push_back(plan_code(kind, side, level, digit0 / kGroupDigits,
                                 groups_from(p.B[level], digit0, kSpanDigits / kGroupDigits),
                                 kind == kItemDenseMax ? 0 : 1));
    };
    if (p.read_list_level >= 0 && !dense(kRead, p.read_list_level))
      span_item(kItemReadList, kRead, p.read_list_level);
    if (p.list_level >= 0 && dense(kWrite, p.list_level))
      span_item(kItemDenseMax, kWrite, p.list_level);
    for (int side = 0; side < kSides; ++side)
      for (int level = 0; level < p.D; ++level)
        if (!dense(side, level) && !(side == kWrite && level == p.list_level)
            && !(side == kRead && level == p.read_list_level))
          span_item(kItemVote, side, level);

    std::vector<uint32_t> items;
    const size_t mid = (ring.size() + 1) / 2;
    if (!half.empty()) items.push_back(half[0]);
    items.insert(items.end(), ring.begin(), ring.begin() + static_cast<std::ptrdiff_t>(mid));
    if (half.size() > 1) items.push_back(half[1]);
    items.insert(items.end(), ring.begin() + static_cast<std::ptrdiff_t>(mid), ring.end());
    p.plan_items[s] = static_cast<int>(items.size());
    for (size_t i = 0; i < items.size(); ++i) p.plan[s][i] = items[i];
  }
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
                        Tensor words, Tensor unions, Tensor list_exp, Tensor fm,
                        std::vector<Tensor> lists, std::vector<int64_t> widths, int64_t dense_read,
                        std::vector<int64_t> mask_read, int64_t dense_write,
                        std::vector<int64_t> mask_write, int64_t list_level,
                        int64_t read_list_level, int64_t window) {
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
  const int64_t dense[kSides] = {dense_read, dense_write};
  const std::vector<int64_t>* masks[kSides] = {&mask_read, &mask_write};
  for (int level = 0; level < D; ++level) {
    STD_TORCH_CHECK(widths[level] >= 1 && widths[level] <= kMaxWidth, "level ", level, " is ",
                    widths[level], " digits wide; a level is 1 to ", kMaxWidth);
    p.B[level] = static_cast<int>(widths[level]);
    p.row_base[level] = p.rows_total;
    p.rows_total += p.B[level];
  }
  const int need = mask_words(p.rows_total);
  for (int side = 0; side < kSides; ++side) {
    STD_TORCH_CHECK(dense[side] >= 0 && dense[side] < (1 << D), "side ", side,
                    "'s dense-level bitmask ", dense[side], " names a level outside the depth ", D);
    STD_TORCH_CHECK(static_cast<int>(masks[side]->size()) == need, "side ", side, " carries ",
                    masks[side]->size(), " digit-mask words; ", p.rows_total, " rows need ", need);
    p.dense[side] = static_cast<uint32_t>(dense[side]);
    for (int w = 0; w < need; ++w) p.mask[side][w] = static_cast<uint32_t>((*masks[side])[w]);
    for (int level = 0; level < D; ++level) {
      const Tensor& rows = (*sides[side])[level];
      const bool half = side == kWrite && level_is_dense(p.dense[side], level);
      p.rows[side][level] = detail::row_view(rows, B, T, H, half, side ? "write[l]" : "read[l]");
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
  STD_TORCH_CHECK(list_level >= -1 && list_level < D, "the list level is a level of the depth ", D,
                  " or -1 for none, got ", list_level);
  p.list_level = static_cast<int>(list_level);
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
  if (p.list_level >= 0) {
    STD_TORCH_CHECK(list_exp.is_cuda() && list_exp.scalar_type() == Dtype::Short
                        && list_exp.is_contiguous()
                        && list_exp.numel() == B * H * p.windows * p.B[p.list_level] * 2,
                    "list_exp is the contiguous int16 [", B * H, ", ", p.windows, ", ",
                    p.B[p.list_level], ", 2] list exponents");
    p.list_exp = reinterpret_cast<uint32_t*>(list_exp.mutable_data_ptr<int16_t>());
    STD_TORCH_CHECK(fm.is_cuda() && fm.scalar_type() == Dtype::Float && fm.is_contiguous()
                        && fm.numel() == B * H * p.L * 2,
                    "fm is the contiguous fp32 [", B * H, ", ", p.L, ", 2] factor products");
    p.fm = reinterpret_cast<float2*>(fm.mutable_data_ptr<float>());
  }
  //: THE LISTS OUT, a side's at its listed level when that level is voted: [entries, at, union, count] the write side's
  //: then the read side's, then the gains' f16 pairs and their exponents (docs/internals/facts/facts_api.md)
  STD_TORCH_CHECK(read_list_level >= -1 && read_list_level < D,
                  "the read list level is a level of the depth ", D, " or -1 for none, got ",
                  read_list_level);
  p.read_list_level = static_cast<int>(read_list_level);
  detail::plan_units(p);
  STD_TORCH_CHECK(lists.size() == 10, "lists holds 10 tensors (facts_api.md), got ", lists.size());
  const auto u16_view = [&](const Tensor& t, std::initializer_list<int64_t> want,
                            const char* name) {
    std::vector<int64_t> dims(want);
    STD_TORCH_CHECK(t.is_cuda()
                        && (t.scalar_type() == Dtype::Short || t.scalar_type() == Dtype::Half)
                        && t.is_contiguous() && t.dim() == static_cast<int64_t>(dims.size()),
                    name, " is a contiguous CUDA 16-bit tensor of ", dims.size(), " dimensions");
    for (size_t i = 0; i < dims.size(); ++i)
      STD_TORCH_CHECK(t.size(static_cast<int64_t>(i)) == dims[i], name, " dimension ", i, " is ",
                      dims[i], ", got ", t.size(static_cast<int64_t>(i)));
    return reinterpret_cast<uint16_t*>(t.mutable_data_ptr());
  };
  for (int side = 0; side < kSides; ++side) {
    const int level = side == kWrite ? p.list_level : p.read_list_level;
    if (level < 0 || level_is_dense(p.dense[side], level)) continue;
    const int64_t width = p.B[level], groups = (width + 31) / 32;
    const int at = side == kWrite ? 0 : 4;
    ListsOut& o = p.lists[side];
    if (side == kWrite)
      o.record = reinterpret_cast<uint2*>(
          u16_view(lists[at], {B * H, p.windows, width, kListCap, 4}, "a write list's records"));
    else
      o.slot = u16_view(lists[at], {B * H, p.windows, width, kListCap}, "a read list's slots");
    o.at = u16_view(lists[at + 1], {B * H, p.windows, width, kListAt}, "a list's starts");
    o.union_tok =
        u16_view(lists[at + 2], {B * H, p.windows, groups, kWindowTokens}, "a union's tokens");
    o.union_count = u16_view(lists[at + 3], {B * H, p.windows, groups}, "a union's counts");
  }
  if (p.list_level >= 0) {
    p.gains16 =
        reinterpret_cast<uint32_t*>(u16_view(lists[8], {B * H, p.L, 2}, "the gains' f16 pairs"));
    STD_TORCH_CHECK(lists[9].is_cuda() && lists[9].scalar_type() == Dtype::Short
                        && lists[9].is_contiguous() && lists[9].numel() == B * H * p.windows * 2,
                    "the gains' exponents are a contiguous CUDA int16 [", B * H, ", ", p.windows,
                    ", 2]");
    p.gains_exp = lists[9].mutable_data_ptr<int16_t>();
  }
  p.ledger = detail::g_ledger;

  const DeviceGuard guard(first.get_device_index());
  int sms = 0;
  ROLA_CUDA_CHECK(
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, first.get_device_index()));
  const int64_t units = int64_t{B} * H * p.windows * p.spans;
  STD_TORCH_CHECK(units <= INT32_MAX, "the call's ", units, " units exceed the int32 count");
  p.units = static_cast<int>(units);
  if (units == 0) return;
  const int blocks = static_cast<int>(std::min<int64_t>(units, int64_t{kFactsBlocksPerSm} * sms));
  //: ONCE: the held window, the rings, the tiles and the tables are dynamic shared memory, nearly the SM's whole
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
  detail::facts_pass<<<blocks, kFactsThreads, FactsSmem::kBytes, current_stream()>>>(p);
  ROLA_CUDA_LAUNCH_CHECK();
}

}  // namespace facts
}  // namespace rola
