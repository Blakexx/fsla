// THE FACTS PASS'S TRANSLATION UNIT -- the kernel that reads the bundle's rows once, a CTA a unit (a window's side,
// level and 32-digit group): both sides' class-1 words and tile unions, the window's columns, and each side's lists
// ranked and placed in the carry's final form; and the host entry that validates the bundle's views, plans a
// window's units and launches them.
// see docs/internals/facts/facts.md

#include "common/torch_seam.cuh"

#include <algorithm>
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

//: A UNIT'S PLACE: its code's fields and its (write head's batch-head, window). -- see docs/internals/facts/facts.md#the-grid
struct Unit {
  int kind, side, level, group, head, b, h, bh, window;
};

__device__ __forceinline__ Unit unit_of(const FactsParams& p) {
  const uint32_t code = p.unit[blockIdx.x];
  Unit x;
  x.kind = unit_kind(code);
  x.side = unit_side(code);
  x.level = unit_level(code);
  x.group = unit_group(code);
  x.head = unit_head(code);
  x.window = static_cast<int>(blockIdx.y);
  x.bh = static_cast<int>(blockIdx.z);
  x.b = x.bh / p.H;
  x.h = x.bh - x.b * p.H;
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
//: width or `L` zero-filled; a read unit's query head `j` of its write head's group. -- see docs/internals/facts/facts.md#load
__device__ __forceinline__ void issue_window(const FactsParams& p, const Unit& x, ops::SmemAddr win,
                                             int tid, int j) {
  const RowView v = p.rows[x.side][x.level];
  const int head = x.side == kRead ? x.h * p.g + j : x.h;
  const int digit0 = kGroupDigits * x.group, part = tid & 1, i = tid >> 1;
  constexpr int kCopies = kWindowTokens * 2 / kFactsThreads, kStep = kFactsThreads / 2;
  const int t = x.window * kWindowTokens + i;
  const uint8_t* const src =
      v.base + x.b * v.sb + head * v.sh + static_cast<int64_t>(t) * v.st + digit0 + 16 * part;
  const int64_t step = kStep * v.st;
  const uint32_t run = static_cast<uint32_t>(max(0, min(16, v.width - digit0 - 16 * part)));
  const ops::SmemAddr dst = win + window_at(i, 16 * part);
#pragma unroll
  for (int k = 0; k < kCopies; ++k)
    ops::stage_tile_n(dst + window_at(kStep * k, 0), src + k * step,
                      t + kStep * k < p.L ? run : 0u);
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
//: bit at bit `i`; a 4 x 4 byte transpose across the token blocks leaves digit `4 dq + tk`'s 32 token bits, masked;
//: the warp's OR of each word its union word.
//: -- see docs/internals/facts/facts.md#vote
__device__ __forceinline__ void vote(const FactsParams& p, const Unit& x, const uint8_t* win,
                                     uint32_t* tile, uint32_t* uword, int warp, int lane) {
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
    const uint32_t u = __reduce_or_sync(0xffffffffu, in ? flags : 0u);
    if (lane == 0) uword[j] = u;
  }
}

//: THE BITS OUT at the words' `slot` (a query head's read side, or the write side's last): each digit's window of words
//: one 64-byte run (a half-warp a digit); warp `w` tile `w`'s union, a digit a lane, one ballot.
//: -- see docs/internals/facts/facts.md#tiles
__device__ __forceinline__ void bits_out(const FactsParams& p, const Unit& x, const uint32_t* tile,
                                         int slot, int tid, int warp, int lane) {
  const int word0 = x.window * kWindowWords, d = tid >> 4, j = tid & 15;
  const int digits = p.B[x.level] - kGroupDigits * x.group;
  const bool word_in = j < p.token_words - word0;
  uint32_t* const out = p.words
                        + ((static_cast<int64_t>(x.bh) * (p.g + 1) + slot) * p.rows_total
                           + p.row_base[x.level] + kGroupDigits * x.group + d)
                              * p.token_words
                        + word0 + j;
  constexpr int kHalf = kFactsThreads / kWindowWords;
#pragma unroll
  for (int k = 0; k < 2; ++k)
    if (word_in & (d + kHalf * k < digits))
      out[static_cast<int64_t>(kHalf * k) * p.token_words] = tile[tile_at(d + kHalf * k, j)];
  const uint32_t any = __ballot_sync(
      0xffffffffu, (tile[tile_at(lane, 2 * warp)] | tile[tile_at(lane, 2 * warp + 1)]) != 0u);
  const int tile_index = x.window * kWindowTiles + warp;
  if (lane == 0 && tile_index < p.tiles)
    p.unions[((static_cast<int64_t>(x.bh) * (p.g + 1) + slot) * p.groups + p.group_base[x.level]
              + x.group)
                 * p.tiles
             + tile_index] = any;
}

//: THE RANKS of query head `j`'s rows, a warp's two words: each union row at its SLOT, `at` past the heads before it;
//: the token at the slot of each of the head's eight boundaries from `k0` its bound. Each list lane gets its two words'
//: union words and the union's tokens before them; `n` the head's union. -- see docs/internals/facts/facts.md#ranks
struct Ranks {
  uint32_t u[2];
  int pre[2], n;
};

//: a side's rows a window: a query head's tokens for each of the group's heads on the read side
__device__ __forceinline__ int heads_of(const FactsParams& p, const Unit& x) {
  return x.side == kRead ? p.g : 1;
}

__device__ __forceinline__ Ranks ranks(const FactsParams& p, const Unit& x, uint8_t* smem, int tid,
                                       int warp, int lane, int at, int k0, int j) {
  const uint32_t* const uword = reinterpret_cast<const uint32_t*>(smem + FactsSmem::kUword);
  int* const bound = reinterpret_cast<int*>(smem + FactsSmem::kBound);
  const ListsOut out = p.lists[x.side];
  const int width = p.B[x.level];
  const int64_t group_index = (static_cast<int64_t>(x.bh) * p.windows + x.window)
                                  * ((width + kGroupDigits - 1) / kGroupDigits)
                              + x.group;
  uint16_t* const union_tok = out.union_tok + group_index * heads_of(p, x) * kWindowTokens + at;
  const int i = lane & 15;
  const uint32_t ui = uword[i];
  int incl = __popc(ui);
#pragma unroll
  for (int off = 1; off < kWindowWords; off <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, incl, off, kWindowWords);
    incl += i >= off ? y : 0;
  }
  const int excl = incl - __popc(ui), n = __shfl_sync(0xffffffffu, incl, kWindowWords - 1);
  const uint32_t below = (1u << lane) - 1u;
#pragma unroll
  for (int jj = 0; jj < 2; ++jj) {
    const int jw = 2 * warp + jj;
    const uint32_t u = __shfl_sync(0xffffffffu, ui, jw);
    const int pre = __shfl_sync(0xffffffffu, excl, jw);
    if ((u >> lane) & 1u) {
      const int slot = pre + __popc(u & below), t = kTokBits * jw + lane;
      union_tok[slot] = static_cast<uint16_t>(kWindowTokens * j + t);
      const int c = (at + slot) / kListBound - 1 - k0;
      if ((at + slot) % kListBound == 0 && c >= 0) bound[c] = t;
    }
  }
  if (tid < kListAt && kListBound * (k0 + tid + 1) - at >= n) bound[tid] = kWindowTokens;
  const int s = lane & 7;
  return Ranks{{__shfl_sync(0xffffffffu, ui, 2 * s), __shfl_sync(0xffffffffu, ui, 2 * s + 1)},
               {__shfl_sync(0xffffffffu, excl, 2 * s), __shfl_sync(0xffffffffu, excl, 2 * s + 1)},
               n};
}

//: THE UNION'S TAIL, after its heads: every slot past the union `kGap`, the union's extent its size in whole steps.
__device__ __forceinline__ void union_tail(const FactsParams& p, const Unit& x, int n, int tid) {
  const int width = p.B[x.level], slots = heads_of(p, x) * kWindowTokens;
  const int64_t group_index = (static_cast<int64_t>(x.bh) * p.windows + x.window)
                                  * ((width + kGroupDigits - 1) / kGroupDigits)
                              + x.group;
  const ListsOut out = p.lists[x.side];
#pragma unroll 1
  for (int k = n + tid; k < slots; k += kFactsThreads)
    out.union_tok[group_index * slots + k] = carry::kGap;
  if (tid == 0) out.union_count[group_index] = (n + kStep - 1) & -kStep;
}

//: A LIST LANE'S TOKENS BEFORE token `b`, lane `s`'s words `2s`, `2s + 1`.
__device__ __forceinline__ int tokens_before(const uint32_t (&w)[2], int s, int b) {
  int n = 0;
#pragma unroll
  for (int jj = 0; jj < 2; ++jj) {
    const int r = b - kTokBits * (2 * s + jj);
    n += __popc(w[jj] & (r <= 0 ? 0u : (r >= kTokBits ? ~0u : (1u << r) - 1u)));
  }
  return n;
}

//: THE LISTS of query head `j`'s rows, a warp's 4 lists `w + 8 j`, 8 lanes a list, lane `s` its words `2s`, `2s + 1`:
//: each lane's tokens before each of the head's boundaries' bounds, summed over the list's lanes into its boundaries
//: past `placed` (the list's entries in the heads before) and scanned into the lane's first place; on the write side a
//: walk takes the list's maxima; a walk places and writes each entry, its slot `at` past the heads before.
//: -- see docs/internals/facts/facts.md#lists-out
template <bool kWriteSide>
__device__ __forceinline__ void lists_out(const FactsParams& p, const Unit& x, uint8_t* smem,
                                          const Ranks& r, int warp, int lane, int at, int j,
                                          int& placed) {
  const uint8_t* const win = smem + FactsSmem::kWin;
  const uint32_t* const tile = reinterpret_cast<const uint32_t*>(smem + FactsSmem::kTile);
  const float2* const cols = reinterpret_cast<const float2*>(smem + FactsSmem::kCols);
  const int* const bound = reinterpret_cast<const int*>(smem + FactsSmem::kBound);
  const int dl = lane >> 3, s = lane & 7, d = warp + kFactsWarps * dl;
  const int width = p.B[x.level], digit = kGroupDigits * x.group + d;
  const bool listed = digit < width;
  const ListsOut out = p.lists[x.side];
  const int64_t list = (static_cast<int64_t>(x.bh) * p.windows + x.window) * width + digit;
  uint2* const entries = out.entries + list * heads_of(p, x) * kListCap;
  const uint32_t words[2] = {tile[tile_at(d, 2 * s)], tile[tile_at(d, 2 * s + 1)]};

  //: the lane's tokens before the head's eight boundaries as u16 pairs, then its total, scanned over the list's lanes
  constexpr int kBoundPairs = kListAt / 2, kPairs = kBoundPairs + 1;
  uint32_t own[kPairs], inc[kPairs];
#pragma unroll
  for (int k = 0; k < kBoundPairs; ++k) {
    own[k] = static_cast<uint32_t>(tokens_before(words, s, bound[2 * k]))
             | static_cast<uint32_t>(tokens_before(words, s, bound[2 * k + 1])) << 16;
    inc[k] = own[k];
  }
  own[kBoundPairs] = static_cast<uint32_t>(__popc(words[0]) + __popc(words[1]));
  inc[kBoundPairs] = own[kBoundPairs];
#pragma unroll
  for (int off = 1; off < 8; off <<= 1)
#pragma unroll
    for (int k = 0; k < kPairs; ++k) {
      const uint32_t y = __shfl_up_sync(0xffffffffu, inc[k], off, 8);
      inc[k] += s >= off ? y : 0u;
    }
  //: lane `s` writes boundary `s` from the list's last lane; its first place is its tokens before
  uint32_t mine = __shfl_sync(0xffffffffu, inc[0], 7, 8);
#pragma unroll
  for (int k = 1; k < kBoundPairs; ++k) {
    const uint32_t all = __shfl_sync(0xffffffffu, inc[k], 7, 8);
    mine = s >> 1 == k ? all : mine;
  }
  //: a head's boundary past the list's entries before it; column 8 the list's entries through the head
  const int head_total = static_cast<int>(__shfl_sync(0xffffffffu, inc[kBoundPairs], 7, 8));
  uint16_t* const table = reinterpret_cast<uint16_t*>(smem + FactsSmem::kTable) + d * (kListAt + 1);
  table[s] =
      listed ? static_cast<uint16_t>(placed + (s & 1 ? mine >> 16 : mine & 0xffffu)) : uint16_t{0};
  if (s == kListAt - 1)
    table[kListAt] = listed ? static_cast<uint16_t>(placed + head_total) : uint16_t{0};
  const int first = placed + static_cast<int>(inc[kBoundPairs] - own[kBoundPairs]);
  placed += head_total;

  //: the write side's maxima, its exponents and scales
  float sm = 1.0f, sv = 1.0f;
  if constexpr (kWriteSide) {
    float mx = 0.0f, mf = 0.0f;
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      uint32_t xw = words[jj];
      for (int k = __any_sync(0xffffffffu, xw != 0u) != 0; k > 0; --k)
#pragma unroll 1
        do {
          if (xw != 0u) {
            const int t = kTokBits * (2 * s + jj) + __ffs(xw) - 1;
            const float b = static_cast<float>(win[window_at(t, d)]);
            const float2 col = cols[t];
            mx = fmaxf(mx, b * col.x);
            mf = fmaxf(mf, b * col.y);
            xw &= xw - 1u;
          }
        } while (__any_sync(0xffffffffu, xw != 0u));
    }
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

  //: the entries: the lane's `i`-th at its first place plus `i`
  const int row = p.read_row;
  int pos = first;
#pragma unroll
  for (int jj = 0; jj < 2; ++jj) {
    uint32_t xw = words[jj];
    for (int k = __any_sync(0xffffffffu, xw != 0u) != 0; k > 0; --k)
#pragma unroll 1
      do {
        if (xw != 0u) {
          const int bit = __ffs(xw) - 1, t = kTokBits * (2 * s + jj) + bit;
          xw &= xw - 1u;
          const uint32_t sl =
              static_cast<uint32_t>(at + r.pre[jj] + __popc(r.u[jj] & ((1u << bit) - 1u)));
          const float b = static_cast<float>(win[window_at(t, d)]);
          uint2 e;
          if constexpr (kWriteSide) {
            const float2 col = cols[t];
            const uint32_t cv = __half_as_ushort(__float2half_rn(b * col.y * sv));
            const uint32_t cm = __half_as_ushort(__float2half_rn(b * col.x * sm));
            e = make_uint2(sl, cv | cm << 16);
          } else {
            const uint32_t coef = __half_as_ushort(__float2half_rn(b * (1.0f / 256.0f)));
            e = make_uint2((sl * (row / 16) + carry::pool_chunk(row, sl, 0u))
                               | static_cast<uint32_t>(kWindowTokens * j + t) << 16,
                           coef | ((carry::kPadBytes + carry::panel_row(sl)) & 0xffffu) << 16);
          }
          if (listed) entries[pos] = e;
          ++pos;
        }
      } while (__any_sync(0xffffffffu, xw != 0u));
  }
}

//: THE BOUNDARIES OUT, a head's, warp 0's: lane `l` list `l`'s at columns `k0 ..` (a later head's own overwrite the
//: ones past this head's rows), and each column summed over the lists into the group's totals; the last head's
//: entries through it every column past its eight. -- see docs/internals/facts/facts.md#lists-out
__device__ __forceinline__ void bounds_out(const FactsParams& p, const Unit& x, uint8_t* smem,
                                           int lane, int k0, bool last) {
  const uint16_t* const table = reinterpret_cast<const uint16_t*>(smem + FactsSmem::kTable);
  int32_t* const sums = reinterpret_cast<int32_t*>(smem + FactsSmem::kTotals);
  if (lane <= kListAt) {
    int sum = 0;
#pragma unroll 1
    for (int l = 0; l < kGroupDigits; ++l) sum += table[l * (kListAt + 1) + lane];
    sums[lane] = sum;
  }
  __syncwarp();
  const int width = p.B[x.level], columns = heads_of(p, x) * kListAt;
  const int64_t group_index = (static_cast<int64_t>(x.bh) * p.windows + x.window)
                                  * ((width + kGroupDigits - 1) / kGroupDigits)
                              + x.group;
  uint16_t* const bounds = p.lists[x.side].bounds + (group_index * kGroupDigits + lane) * columns;
  const int end = last ? columns : min(columns, k0 + kListAt);
#pragma unroll 1
  for (int k = k0; k < end; ++k) bounds[k] = table[lane * (kListAt + 1) + min(k - k0, kListAt)];
  int32_t* const totals = p.lists[x.side].totals + group_index * columns;
#pragma unroll 1
  for (int k = k0 + lane; k < end; k += 32) totals[k] = sums[min(k - k0, kListAt)];
}

//: THE GRID: a CTA a unit, a window's units innermost. Every unit loads its window (a columns unit nothing) and a write
//: list unit its columns; a vote or list unit votes and writes its bits; a list unit ranks its union and writes its
//: lists -- a read list unit its write head's query heads in turn, each past the ones before.
//: -- see docs/internals/facts/facts.md#the-grid
__global__ __launch_bounds__(kFactsThreads, kFactsCtasMin) void facts_pass(const FactsParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ long long s_clock[kFactsWarps][Clock::kRow];
  const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
  Clock clock(&s_clock[warp][0], lane, p.ledger != nullptr);
  const Unit x = unit_of(p);
  uint8_t* const win = smem + FactsSmem::kWin;
  uint32_t* const tile = reinterpret_cast<uint32_t*>(smem + FactsSmem::kTile);
  float2* const cols = reinterpret_cast<float2*>(smem + FactsSmem::kCols);
  float2* const wmax = reinterpret_cast<float2*>(smem + FactsSmem::kWmax);
  const bool listed = x.kind == kUnitList, writes = (x.side == kWrite) & listed;
  const bool columns = (x.kind == kUnitColumns) | writes;
  const int heads = listed ? heads_of(p, x) : 1;

  //: THE LOAD: the window's copies, a write unit's columns beside them, each warp's largest posted
  if (x.kind != kUnitColumns) issue_window(p, x, ops::smem_addr(win), tid, x.head);
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

  int at = 0, placed = 0;
#pragma unroll 1
  for (int j = 0; j < heads; ++j) {
    const int head = listed ? j : x.head;
    if (j > 0) {
      //: the next head's window, once every warp is done with this one's
      __syncthreads();
      issue_window(p, x, ops::smem_addr(win), tid, head);
      ops::stage_commit();
      ops::stage_wait<0>();
      __syncthreads();
      clock.lap(kPhaseLoad);
    }
    //: THE VOTE AND THE BITS OUT
    vote(p, x, win, tile, reinterpret_cast<uint32_t*>(smem + FactsSmem::kUword), warp, lane);
    __syncthreads();
    bits_out(p, x, tile, x.side == kRead ? head : p.g, tid, warp, lane);
    clock.lap(kPhaseVote);
    if (!listed) break;

    //: THE RANKS AND THE LISTS
    const int k0 = at / kListBound;
    const Ranks r = ranks(p, x, smem, tid, warp, lane, at, k0, head);
    __syncthreads();
    clock.lap(kPhaseRanks);
    if (writes) {
      lists_out<true>(p, x, smem, r, warp, lane, at, head, placed);
    } else {
      lists_out<false>(p, x, smem, r, warp, lane, at, head, placed);
    }
    __syncthreads();
    if (warp == 0) bounds_out(p, x, smem, lane, k0, j == heads - 1);
    at += r.n;
    clock.lap(kPhaseLists);
  }
  if (listed) union_tail(p, x, at, tid);
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
//: level (the read side's its write head's query heads in turn), a vote unit elsewhere (the read side's one a query
//: head) -- and the window's columns unit when the write side lists nothing.
//: -- see docs/internals/facts/facts.md#the-plan
void plan_units(FactsParams& p, const uint32_t (&dense)[kSides]) {
  int n = 0;
  for (int side = 0; side < kSides; ++side)
    for (int level = 0; level < p.D; ++level) {
      if (level_is_dense(dense[side], level)) continue;
      const bool list = level == p.list_level[side];
      const int heads = side == kRead && !list ? p.g : 1;
      for (int g = 0; g < (p.B[level] + kGroupDigits - 1) / kGroupDigits; ++g)
        for (int head = 0; head < heads; ++head)
          p.unit[n++] = unit_code(list ? kUnitList : kUnitVote, side, level, g, head);
    }
  if (p.list_level[kWrite] < 0) p.unit[n++] = unit_code(kUnitColumns, kWrite, 0, 0, 0);
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
                        int64_t read_list_level, int64_t window, int64_t read_row) {
  check_arch_table();
  const int D = static_cast<int>(widths.size());
  STD_TORCH_CHECK(D >= 1 && D <= kMaxLevels, "the routing depth is one to four levels, got ", D);
  STD_TORCH_CHECK(static_cast<int>(read.size()) == D && static_cast<int>(write.size()) == D
                      && static_cast<int>(scale_write.size()) == D,
                  "each side holds one row set a level and the write side one scale a level: ", D);
  STD_TORCH_CHECK(window == kWindowTokens, "the facts pass's window is the carry's ", kWindowTokens,
                  " tokens, got ", window);
  const Tensor& first = read[0];
  STD_TORCH_CHECK(first.dim() == 4 && write[0].dim() == 4,
                  "a level's rows are [B, T, H, width], got ", shape(first.sizes()));
  //: GROUPED QUERIES: the read side's H query heads over the write side's H_w write heads, g = H / H_w each
  const int64_t B = first.size(0), T = first.size(1), Hq = first.size(2), H = write[0].size(2);
  STD_TORCH_CHECK(T >= 1, "a call carries at least one token");
  STD_TORCH_CHECK(H >= 1 && Hq % H == 0 && Hq / H <= kMaxGroup, "the read side's ", Hq,
                  " query heads are a whole number, at most ", kMaxGroup, ", of the write side's ",
                  H, " write heads");

  FactsParams p{};
  p.D = D;
  p.H = static_cast<int>(H);
  p.g = static_cast<int>(Hq / H);
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
      p.rows[side][level] =
          detail::row_view(rows, B, T, side == kRead ? Hq : H, side ? "write[l]" : "read[l]");
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
  for (int side = 0; side < kSides; ++side) {
    const int64_t level = list_levels[side];
    STD_TORCH_CHECK(level >= -1 && level < D && (level < 0 || !level_is_dense(dense[side], level)),
                    "side ", side, "'s list level is a sparse level of the depth ", D,
                    " or -1 for none, got ", level);
    p.list_level[side] = static_cast<int>(level);
  }
  STD_TORCH_CHECK(
      read_row >= 64 && read_row % 64 == 0 && (read_row == 128 || read_row / 64 % 2 == 1)
          && p.g * kWindowTokens * read_row / 16 <= 65536,
      "the read pool row is 128 bytes or an odd number of 64, a window's g heads' pool inside a "
      "16-bit offset of 16-byte chunks; got ",
      read_row, " bytes at g ", p.g);
  p.read_row = static_cast<int>(read_row);
  for (int level = 0; level < D; ++level)
    p.scale_write[level] = detail::column_view(scale_write[level], B, T, H, "scale_write[l]");
  p.g_write = detail::column_view(g_write, B, T, H, "g_write");
  p.v_scale = detail::column_view(v_scale, B, T, H, "v_scale");
  //: the words and unions a write head's: its query heads' read slots, then its write side's
  STD_TORCH_CHECK(words.is_cuda() && words.scalar_type() == Dtype::Int && words.is_contiguous()
                      && words.dim() == 4 && words.size(0) == B * H && words.size(1) == p.g + 1
                      && words.size(2) == p.rows_total && words.size(3) == p.token_words,
                  "words is the contiguous int32 class-1 table [", B * H, ", ", p.g + 1, ", ",
                  p.rows_total, ", ", p.token_words, "], got ", shape(words.sizes()));
  LivenessLayout layout{D, {}, p.L};
  for (int level = 0; level < D; ++level) layout.B[level] = p.B[level];
  for (int level = 0; level < D; ++level) p.group_base[level] = layout.group_base(level);
  p.groups = layout.groups();
  p.tiles = layout.tiles();
  STD_TORCH_CHECK(unions.is_cuda() && unions.scalar_type() == Dtype::Int && unions.is_contiguous()
                      && unions.dim() == 4 && unions.size(0) == B * H && unions.size(1) == p.g + 1
                      && unions.size(2) == p.groups && unions.size(3) == p.tiles,
                  "unions is the contiguous int32 tile-union table [", B * H, ", ", p.g + 1, ", ",
                  p.groups, ", ", p.tiles, "], got ", shape(unions.sizes()));
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
  //: THE LISTS OUT, a side's at its list level: entries, boundaries, the union by slot, its extent
  const std::vector<Tensor>* lists[kSides] = {&read_lists, &write_lists};
  for (int side = 0; side < kSides; ++side) {
    STD_TORCH_CHECK(lists[side]->size() == 5, "side ", side,
                    "'s lists are (entries, bounds, totals, union, count), got ",
                    lists[side]->size(), " tensors");
    if (p.list_level[side] < 0) continue;
    const int64_t width = p.B[p.list_level[side]],
                  groups = (width + kGroupDigits - 1) / kGroupDigits,
                  heads = side == kRead ? p.g : 1;
    const std::vector<Tensor>& l = *lists[side];
    ListsOut& o = p.lists[side];
    o.entries = static_cast<uint2*>(exact(
        l[0], Dtype::Short, {B * H, p.windows, width, heads * kListCap, 4}, "a side's entries"));
    o.bounds = static_cast<uint16_t*>(
        exact(l[1], Dtype::Short, {B * H, p.windows, groups, kGroupDigits, heads * kListAt},
              "a side's lists' boundaries"));
    o.totals = static_cast<int32_t*>(exact(
        l[2], Dtype::Int, {B * H, p.windows, groups, heads * kListAt}, "a side's groups' totals"));
    o.union_tok = static_cast<uint16_t*>(exact(
        l[3], Dtype::Short, {B * H, p.windows, groups, heads * kWindowTokens}, "a side's union"));
    o.union_count = static_cast<int32_t*>(
        exact(l[4], Dtype::Int, {B * H, p.windows, groups}, "a side's union extents"));
  }
  detail::plan_units(p, dense);
  p.ledger = detail::g_ledger;

  const DeviceGuard guard(first.get_device_index());
  //: THE GRID: a window's units by x, the windows by y, the batch-heads by z
  STD_TORCH_CHECK(p.windows <= 65535 && B * H <= 65535,
                  "the grid is (units a window, windows, batch-heads): ", p.windows,
                  " windows and ", B * H, " batch-heads, each at most 65535");
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
  const dim3 grid(static_cast<unsigned>(p.per_window), static_cast<unsigned>(p.windows),
                  static_cast<unsigned>(B * H));
  detail::facts_pass<<<grid, kFactsThreads, FactsSmem::kBytes, current_stream()>>>(p);
  ROLA_CUDA_LAUNCH_CHECK();
}

}  // namespace facts
}  // namespace rola
