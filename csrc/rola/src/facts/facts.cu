// THE FACTS PASS'S TRANSLATION UNIT -- the kernel that reads the bundle's rows once for both sides' class-1 words and
// the write lists' exponents, and the host entry that validates the bundle's views and launches it.
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

//: A TOKEN'S COLUMN SLOT in the staged window: a pad after every 8, so the four token blocks a lane group reads together
//: meet distinct banks.
__device__ __forceinline__ int column_slot(int t) { return t + (t >> 3); }

//: A WORD'S PLACE IN ITS RING SLOT: token `t`'s 32-byte row at row `4 (t % 8) + t / 8`, so the four token blocks a lane
//: group reads together (`8 tk + i`, `tk` 0..3) sit in four consecutive rows and a warp's reads meet 32 banks; a whole
//: row moves, so a copy keeps its run's 32-byte phase. -- see docs/internals/facts/facts.md#load
__device__ __forceinline__ int ring_row(int t) { return 4 * (t & 7) + (t >> 3); }

//: A LANE'S COPIES FOR ITS PART: lanes `2i`, `2i + 1` one token's halves of a unit's 32 digits, a word's two token
//: blocks -- one 16-byte quantum a half of u8 digits, or two of a dense list level's f16 (`kDense`) -- its source a
//: pointer set once and stepped a word at a time, a quantum's bytes past the rows' width or a token past `L`
//: zero-filled and unread. -- see docs/internals/facts/facts.md#load
template <bool kDense>
struct LaneCopies {
  static constexpr int kDigitBytes = kDense ? 2 : 1, kRow = 32 * kDigitBytes, kQuanta = kDigitBytes;
  static constexpr int kSlot = kDense ? kDenseWordBytes : kWordBytes;
  const uint8_t* src;             // token `t0 + lane / 2` of the next word, at the lane's half
  int64_t step;                   // a word's 32 tokens
  int64_t second;                 // the word's second 16 tokens
  int token, last;                // the lane's next token, and the call's tokens
  uint32_t run[kQuanta];          // the lane's bytes of a token, each quantum 0 to 16
  ops::SmemAddr dst, dst_second;  // the lane's places in slot 0

  LaneCopies() = default;

  __device__ __forceinline__ LaneCopies(const FactsParams& p, const RowView& view,
                                        const uint8_t* rows, int first_digit, int t0, int lane,
                                        ops::SmemAddr ring) {
    const int digit = first_digit + 16 * (lane & 1);
    token = t0 + (lane >> 1);
    last = p.L;
#pragma unroll
    for (int q = 0; q < kQuanta; ++q)
      run[q] = static_cast<uint32_t>(max(0, min(16, (view.width - digit) * kDigitBytes - 16 * q)));
    step = 32 * view.st;
    second = 16 * view.st;
    src = rows + token * view.st + digit * kDigitBytes;
    dst = ring + ring_row(lane >> 1) * kRow + 16 * kQuanta * (lane & 1);
    dst_second = ring + ring_row(16 + (lane >> 1)) * kRow + 16 * kQuanta * (lane & 1);
  }

  __device__ __forceinline__ void issue(int stage) {
#pragma unroll
    for (int q = 0; q < kQuanta; ++q) {
      ops::stage_tile_n(dst + stage * kSlot + 16 * q, src + 16 * q, token < last ? run[q] : 0u);
      ops::stage_tile_n(dst_second + stage * kSlot + 16 * q, src + second + 16 * q,
                        token + 16 < last ? run[q] : 0u);
    }
    src += step;
    token += 32;
  }
};

//: A BYTE PERMUTE WITH ITS SELECTOR IMMEDIATE: the one PRMT, no selector register reloaded a use.
template <uint32_t Sel>
__device__ __forceinline__ uint32_t prmt_imm(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "n"(Sel));
  return d;
}

//: A WORD'S DIGIT LANE: lane `(tk, dq)` reads token `8 tk + i`'s digits `4 dq .. 4 dq + 3` as one word (`i` 0..7). Its
//: VOTE: bit 7 of each byte set iff the byte is (three operations), shifted to bit `i` of the byte, so byte `k` holds
//: digit `4 dq + k`'s 8 tokens; a 4 x 4 byte transpose across the four token blocks then leaves lane `(tk, dq)` digit
//: `4 dq + tk`'s 32 tokens. Its LISTS: each byte to fp32 through the mantissa, `b m` and `b m s` maxed into its four
//: digits' pairs. -- see docs/internals/facts/facts.md#vote
template <bool kVote, bool kList>
__device__ __forceinline__ uint32_t word_lanes(const uint8_t* slot, const float2* cols, int tk,
                                               int dq, float2 (&best)[4]) {
  uint32_t flags = 0u;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint32_t x = *reinterpret_cast<const uint32_t*>(slot + (4 * i + tk) * 32 + 4 * dq);
    if constexpr (kVote) {
      const uint32_t y = (((x & 0x7f7f7f7fu) + 0x7f7f7f7fu) | x) & 0x80808080u;
      flags |= y >> (7 - i);
    }
    if constexpr (kList) {
      const float2 c = cols[column_slot(8 * tk + i)];
      const uint32_t magic = 0x4b000000u;
      const float b0 = __uint_as_float(prmt_imm<0x7440u>(x, magic)) - 8388608.0f;
      const float b1 = __uint_as_float(prmt_imm<0x7441u>(x, magic)) - 8388608.0f;
      const float b2 = __uint_as_float(prmt_imm<0x7442u>(x, magic)) - 8388608.0f;
      const float b3 = __uint_as_float(prmt_imm<0x7443u>(x, magic)) - 8388608.0f;
      best[0] = make_float2(fmaxf(best[0].x, b0 * c.x), fmaxf(best[0].y, b0 * c.y));
      best[1] = make_float2(fmaxf(best[1].x, b1 * c.x), fmaxf(best[1].y, b1 * c.y));
      best[2] = make_float2(fmaxf(best[2].x, b2 * c.x), fmaxf(best[2].y, b2 * c.y));
      best[3] = make_float2(fmaxf(best[3].x, b3 * c.x), fmaxf(best[3].y, b3 * c.y));
    }
  }
  //: the 4 x 4 byte transpose: two exchanges, the lanes' token block bits 1 then 0
  uint32_t y = __shfl_xor_sync(0xffffffffu, flags, 16);
  flags = (tk & 2) ? prmt_imm<0x3276u>(flags, y) : prmt_imm<0x5410u>(flags, y);
  y = __shfl_xor_sync(0xffffffffu, flags, 8);
  flags = (tk & 1) ? prmt_imm<0x3715u>(flags, y) : prmt_imm<0x6240u>(flags, y);
  return flags;
}

//: A DENSE LIST LEVEL'S WORD: lane `(tk, dq)` reads token `8 tk + i`'s f16 digits `4 dq .. 4 dq + 3` as one 8-byte
//: load (`i` 0..7), each to the byte scale (x 255, exact), `b m` and `b m s` maxed into its four digits' pairs.
//: -- see docs/internals/facts/facts.md#lists
__device__ __forceinline__ void dense_list_lanes(const uint8_t* slot, const float2* cols, int tk,
                                                 int dq, float2 (&best)[4]) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint2 x = *reinterpret_cast<const uint2*>(slot + (4 * i + tk) * 64 + 8 * dq);
    const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&x.x));
    const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&x.y));
    const float b[4] = {kRowMaxDigit * lo.x, kRowMaxDigit * lo.y, kRowMaxDigit * hi.x,
                        kRowMaxDigit * hi.y};
    const float2 c = cols[column_slot(8 * tk + i)];
#pragma unroll
    for (int k = 0; k < 4; ++k)
      best[k] = make_float2(fmaxf(best[k].x, b[k] * c.x), fmaxf(best[k].y, b[k] * c.y));
  }
}

//: A THREAD'S COLUMN LOADS for the window's tokens `threadIdx.x + i kFactsThreads`: the gain, the value scale and the
//: write scales, issued before the ring's copies so they lead the item's loads.
struct ColumnLoads {
  static constexpr int kPerThread = kWindowTokens / kFactsThreads;
  float m[kPerThread], s[kPerThread], f[kPerThread][kMaxLevels];
};

__device__ __forceinline__ ColumnLoads load_columns(const FactsParams& p, int b, int h,
                                                    int window) {
  constexpr int kPerThread = ColumnLoads::kPerThread;
  //: each column's (batch, head) base once; a token is then one multiply-add from it
  const float* const gain = p.g_write.base + b * p.g_write.sb + h * p.g_write.sh;
  const float* const vs = p.v_scale.base + b * p.v_scale.sb + h * p.v_scale.sh;
  const float* sw[kMaxLevels];
#pragma unroll
  for (int l = 0; l < kMaxLevels; ++l)
    sw[l] = l < p.D ? p.scale_write[l].base + b * p.scale_write[l].sb + h * p.scale_write[l].sh
                    : nullptr;
  ColumnLoads c;
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
    const int t = window * kWindowTokens + i * kFactsThreads + static_cast<int>(threadIdx.x);
    const bool in = t < p.L;
    c.m[i] = in ? __ldg(gain + t * p.g_write.st) : 0.0f;
    c.s[i] = in ? __ldg(vs + t * p.v_scale.st) : 0.0f;
#pragma unroll
    for (int l = 0; l < kMaxLevels; ++l)
      c.f[i][l] = in && l < p.D ? __ldg(sw[l] + t * p.scale_write[l].st) : 1.0f;
  }
  return c;
}

//: THE WINDOW'S COLUMNS, staged for its list units: a token's `m = g x the write scales` (the gain first, then the
//: levels in order) and `m x s`, a dead or past-the-end token zeros, and past them the window's largest of each; with
//: `fm` (the window's first list item) each live token's `(F, m) = (m s, m)` stored, the fold's factor products.
//: -- see docs/internals/facts/facts.md#lists
__device__ void stage_columns(ColumnLoads c, int warp, int lane, float2* cols, float2* warp_top,
                              float2* fm, int t0, int L) {
  constexpr int kPerThread = ColumnLoads::kPerThread;
  float2 top = make_float2(0.0f, 0.0f);
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
#pragma unroll
    for (int l = 0; l < kMaxLevels; ++l) c.m[i] *= c.f[i][l];
    const int t = i * kFactsThreads + static_cast<int>(threadIdx.x);
    cols[column_slot(t)] = make_float2(c.m[i], c.m[i] * c.s[i]);
    if (fm != nullptr && t0 + t < L) fm[t0 + t] = make_float2(c.m[i] * c.s[i], c.m[i]);
    top = make_float2(fmaxf(top.x, c.m[i]), fmaxf(top.y, c.m[i] * c.s[i]));
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    top.x = fmaxf(top.x, __shfl_xor_sync(0xffffffffu, top.x, offset));
    top.y = fmaxf(top.y, __shfl_xor_sync(0xffffffffu, top.y, offset));
  }
  if (lane == 0) warp_top[warp] = top;
  __syncthreads();
  if (threadIdx.x == 0) {
    float2 m = warp_top[0];
#pragma unroll
    for (int w = 1; w < kFactsWarps; ++w)
      m = make_float2(fmaxf(m.x, warp_top[w].x), fmaxf(m.y, warp_top[w].y));
    cols[kColumnSlots] = m;
  }
}

//: A READ PART'S COPIES STARTED, `kWords` words of the window: the lane's copies set up and the ring's first stages
//: less one issued, before a list item's column staging so the two loads overlap.
template <int kWords, bool kDense>
__device__ __forceinline__ LaneCopies<kDense> start_part(const FactsParams& p, int code, int b,
                                                         int h, int window, int part, int lane,
                                                         uint8_t* ring) {
  constexpr int kAh = (kDense ? kDenseStages : kStages) - 1;
  const int side = code >> 5, level = (code >> 3) & 3, group = code & 7;
  const RowView view = p.rows[side][level];
  const uint8_t* const rows = static_cast<const uint8_t*>(view.base) + b * view.sb + h * view.sh;
  LaneCopies<kDense> copies(p, view, rows, group * 32,
                            (window * kWindowWords + part * kWords) * kTokBits, lane,
                            ops::smem_addr(ring));
#pragma unroll
  for (int s = 0; s < kAh; ++s) {
    copies.issue(s);
    ops::stage_commit();
  }
  return copies;
}

//: A WARP'S PART OF A UNIT through its ring, tile by tile: a VOTE unit lane `(tk, dq)` digit `4 dq + tk`'s voted bits
//: into its tile row and each tile's union, one reduction; a LIST unit each digit's largest `b` x column, a dense list
//: level's unit (its rows f16, `kDense`) nothing else. -- see docs/internals/facts/facts.md#vote
template <bool kVote, bool kList, int kWords, bool kDense>
__device__ void read_part(const FactsParams& p, int code, int bh, int window, int part, int lane,
                          uint32_t* tile, uint8_t* ring, const float2* cols, float2* top,
                          LaneCopies<kDense>& copies, uint32_t* live) {
  static_assert(!kDense || (kList && !kVote), "a dense level's rows are read for its lists alone");
  constexpr int kSt = kDense ? kDenseStages : kStages, kAh = kSt - 1;
  const int side = code >> 5, level = (code >> 3) & 3, group = code & 7;
  const int tk = lane >> 3, dq = lane & 7, d = 4 * dq + tk;
  const int width = p.B[level], base = p.row_base[level];
  const int digit = group * 32 + d;
  const bool in_mask = digit < width && mask_bit(p.mask[side][(base + digit) >> 5], base + digit);
  uint32_t* const unions =
      p.unions
      + ((static_cast<int64_t>(bh) * kSides + side) * p.groups + p.group_base[level] + group)
            * p.tiles;
  const int word0 = window * kWindowWords + part * kWords;
  const int words = max(0, min(kWords, p.token_words - word0));
  uint32_t* const row = kVote ? tile + d * kTileStride + part * kWords : nullptr;

  float2 best[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) best[k] = make_float2(0.0f, 0.0f);
  //: the trip count a constant, a tile (two words) a step: a word past the call's end is zero-filled and stores nothing
#pragma unroll 1
  for (int j = 0; j < kWords; j += 2) {
    uint32_t pair = 0u;
#pragma unroll
    for (int w = 0; w < 2; ++w) {
      const int jj = j + w;
      if (jj + kAh < kWords) copies.issue((jj + kAh) % kSt);
      ops::stage_commit();
      ops::stage_wait<kAh>();
      __syncwarp();
      const uint8_t* const slot = ring + (jj % kSt) * LaneCopies<kDense>::kSlot;
      const float2* const word_cols = cols + column_slot((part * kWords + jj) * kTokBits);
      uint32_t voted = 0u;
      if constexpr (kDense)
        dense_list_lanes(slot, word_cols, tk, dq, best);
      else
        voted = word_lanes<kVote, kList>(slot, word_cols, tk, dq, best);
      __syncwarp();
      if constexpr (kVote) {
        const uint32_t bits = in_mask ? voted : 0u;
        if (jj < words) row[jj] = bits;
        pair |= bits;
        if (live != nullptr) {
          const uint32_t any = __reduce_or_sync(0xffffffffu, bits);
          if (lane == 0) live[part * kWords + jj] = any;
        }
      }
    }
    if constexpr (kVote) {
      const uint32_t u = __reduce_or_sync(0xffffffffu, pair != 0u ? 1u << d : 0u);
      if (j < words && lane == 0) unions[(word0 + j) >> 1] = u;
    }
  }

  if constexpr (kList) {
    //: the four token blocks' maxima joined, lane `dq` (block 0) holding digits `4 dq .. 4 dq + 3`
#pragma unroll
    for (int k = 0; k < 4; ++k) {
#pragma unroll
      for (int offset = 8; offset < 32; offset <<= 1) {
        best[k].x = fmaxf(best[k].x, __shfl_xor_sync(0xffffffffu, best[k].x, offset));
        best[k].y = fmaxf(best[k].y, __shfl_xor_sync(0xffffffffu, best[k].y, offset));
      }
    }
    if (lane < 8) {
#pragma unroll
      for (int k = 0; k < 4; ++k) top[4 * lane + k] = best[k];
    }
    __syncwarp();
  }
}

//: A WARP'S PART OF A UNIT'S TILE OUT: its share of the 32 rows, a row's window of words one 64-byte run, two rows a
//: store.
template <int kPartsT>
__device__ __forceinline__ void store_part(const FactsParams& p, int code, int bh, int window,
                                           int part, int lane, const uint32_t* tile) {
  constexpr int kPairs = 32 / kPartsT / 2;
  const int side = code >> 5, level = (code >> 3) & 3, group = code & 7;
  const int width = p.B[level], base = p.row_base[level];
  const int word0 = window * kWindowWords;
  const int words = min(kWindowWords, p.token_words - word0);
  const int j = lane & (kWindowWords - 1);
  uint32_t* const out = p.words
                        + (static_cast<int64_t>(bh) * kSides + side) * p.rows_total * p.token_words
                        + word0 + j;
#pragma unroll
  for (int pair = 0; pair < kPairs; ++pair) {
    const int r = 2 * (part * kPairs + pair) + (lane >> 4);
    const int d = group * 32 + r;
    if (d < width && j < words)
      out[static_cast<int64_t>(base + d) * p.token_words] = tile[r * kTileStride + j];
  }
}

//: THE LIST EXPONENT: the smallest `K` with `x <= 2^K`, an `x` below the smallest normal at its exponent.
__device__ __forceinline__ uint32_t ceil_exponent(float x) {
  const uint32_t bits = __float_as_uint(fmaxf(x, kListFloor));
  return static_cast<uint16_t>(static_cast<int>(bits >> 23) - 127 + ((bits & 0x7fffffu) != 0u));
}

//: A LIST UNIT'S EXPONENTS OUT, by its first part's warp: each digit's `kPartsT` parts joined (part `q` the warp
//: `unit_slot + q kFactsWarps / kPartsT`), a lane a digit's `(K_m, K_v)` pair, the unit's 32 one 128-byte run.
//: -- see docs/internals/facts/facts.md#lists
template <int kPartsT>
__device__ __forceinline__ void store_lists(const FactsParams& p, int code, int bh, int window,
                                            int lane, const float2 (*top)[32], int unit_slot) {
  const int digit = (code & 7) * 32 + lane;
  float2 m = top[unit_slot][lane];
#pragma unroll
  for (int part = 1; part < kPartsT; ++part) {
    const float2 o = top[unit_slot + part * (kFactsWarps / kPartsT)][lane];
    m = make_float2(fmaxf(m.x, o.x), fmaxf(m.y, o.y));
  }
  const uint32_t pair = ceil_exponent(m.x) | ceil_exponent(m.y) << 16;
  if (digit < p.B[p.list_level])
    p.list_exp[(static_cast<int64_t>(bh) * p.windows + window) * p.B[p.list_level] + digit] = pair;
}

//: THE LIST EXPONENT, signed: the smallest `K` with `x <= 2^K`, an `x` below the smallest normal at its exponent.
__device__ __forceinline__ int ceil_exponent_signed(float x) {
  const uint32_t bits = __float_as_uint(fmaxf(x, kListFloor));
  return static_cast<int>(bits >> 23) - 127 + ((bits & 0x7fffffu) != 0u);
}

//: 2^e as fp32, `e` in [-126, 127]
__device__ __forceinline__ float pow2(int e) { return __uint_as_float(static_cast<uint32_t>(127 + e) << 23); }

//: THE ROUND-ROBIN PLACE of the `j`-th entry of residue `r` in a pass whose residues hold the byte counts `lo` (residues
//: 0-3) and `hi` (4-7): the entries of the rounds before `j`, then round `j`'s residues below `r`.
//: -- see docs/internals/facts/facts.md#lists-out
__device__ __forceinline__ int round_robin(uint32_t lo, uint32_t hi, int j, int r) {
  const uint32_t jj = 0x01010101u * static_cast<uint32_t>(j);
  const int before = static_cast<int>(__dp4a(__vminu4(lo, jj), 0x01010101u, __dp4a(__vminu4(hi, jj), 0x01010101u, 0u)));
  const uint32_t gl = __vcmpgtu4(lo, jj) & 0x01010101u, gh = __vcmpgtu4(hi, jj) & 0x01010101u;
  const uint32_t below_lo = r >= 4 ? 0xffffffffu : (1u << (8 * r)) - 1u;
  const uint32_t below_hi = r >= 4 ? (1u << (8 * (r - 4))) - 1u : 0u;
  return before + __popc(gl & below_lo) + __popc(gh & below_hi);
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

//: A LISTED UNIT'S LISTS, by all its parts' warps: part `part` of `P` takes `32 / P` of the group's digits, lane
//: `(dl, q)` holding digit `dl`'s words of part `q` -- the warp's CELLS, cell `16 dl + j` word `j` of digit `dl`. The
//: warp's entries, cell by cell, are dealt evenly: lane `L` walks entries `L total / 32` on, found from the lanes'
//: counts (the owner lane by a search, the word by its counts' running sums, the bit by its rank), then each next set
//: bit and nonempty cell. A first walk totals each digit's pass by residue in the warp's TABLE and leaves the counts of
//: the lane's last run of one (digit, pass); a segmented scan of those gives each lane the counts before its first;
//: the second walk ranks each entry among its residue, places it round-robin within its pass and writes it -- on the
//: write side (`kCoef`) with its coefficient pair from its byte and the token's columns.
//: -- see docs/internals/facts/facts.md#lists-out
template <bool kCoef, int P>
__device__ void build_lists(const FactsParams& p, int code, int bh, int window, int part, int lane,
                            const uint32_t* tile, const uint32_t* live, const uint8_t* rings, uint4* table,
                            uint16_t* marks, int unit_slot, int item_units, const float2* cols,
                            const float2 (*top)[32]) {
  constexpr int kWords = kWindowWords / P, kDigits = 32 / P, kGroups = kWords / 4;
  constexpr int kKeys = kDigits * kMaxPasses, kOwned = kKeys / 32, kLanesAWord = 32 / kWords;
  static_assert(kWords % 4 == 0 && kKeys % 32 == 0 && (kOwned == 1 || !kCoef), "a lane's words go 4 a byte group");
  static_assert(kListAt == 2 * kMaxPasses, "a list's starts are its passes' and as many ends");
  const int side = code >> 5, level = (code >> 3) & 3, group = code & 7;
  const int width = p.B[level], groups = (width + 31) / 32;
  const int nw = max(0, min(kWindowWords, p.token_words - window * kWindowWords));
  const ListsOut out = p.lists[side];
  const int64_t win_index = static_cast<int64_t>(bh) * p.windows + window;
  const uint32_t below = (1u << lane) - 1u;
  uint32_t* const cell_mask = reinterpret_cast<uint32_t*>(marks);
  uint16_t* const bases = marks + 2 * kWindowWords;

  //: the union, lane `j` holding its word `un` and base `ub`; part `part` stores its words' tokens
  const uint32_t un = lane < kWindowWords ? live[lane] : 0u;
  uint32_t ub = __popc(un);
#pragma unroll
  for (int off = 1; off < kWindowWords; off <<= 1) {
    const uint32_t x = __shfl_up_sync(0xffffffffu, ub, off);
    ub += lane >= off ? x : 0u;
  }
  const uint32_t union_count = __shfl_sync(0xffffffffu, ub, kWindowWords - 1);
  ub -= __popc(un);
  if (lane < kWindowWords) bases[lane] = static_cast<uint16_t>(ub);
  uint16_t* const tokens = out.union_tok + (win_index * groups + group) * kWindowTokens;
#pragma unroll
  for (int k = 0; k < kWords; ++k) {
    const int j = part * kWords + k;
    const uint32_t u = __shfl_sync(0xffffffffu, un, j), b = __shfl_sync(0xffffffffu, ub, j);
    if ((u >> lane) & 1u) tokens[b + __popc(u & below)] = static_cast<uint16_t>(32 * j + lane);
  }
  if (part == 0 && lane == 0) out.union_count[win_index * groups + group] = static_cast<uint16_t>(union_count);

  //: the lane's cells: their counts' running sums a byte each within groups of 4 (`pre`), the groups' starts, which
  //: are nonempty (into the warp's cell mask)
  const int q = lane % P, dl = lane / P;
  uint32_t pre[kGroups], nonempty = 0u;
  int gstart[kGroups], n = 0;
#pragma unroll
  for (int g = 0; g < kGroups; ++g) {
    gstart[g] = n;
    pre[g] = 0u;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int j = q * kWords + 4 * g + i;
      pre[g] |= static_cast<uint32_t>(n - gstart[g]) << (8 * i);
      const int c = j < nw ? __popc(tile[(part * kDigits + dl) * kTileStride + j]) : 0;
      n += c;
      nonempty |= c != 0 ? 1u << (4 * g + i) : 0u;
    }
  }
#pragma unroll
  for (int x = 0; x < kWords; ++x) {
    const uint32_t v = __reduce_or_sync(0xffffffffu, lane / kLanesAWord == x ? nonempty << (lane % kLanesAWord * kWords)
                                                                           : 0u);
    if (lane == 0) cell_mask[x] = v;
  }
  int first = n;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int x = __shfl_up_sync(0xffffffffu, first, off);
    first += lane >= off ? x : 0;
  }
  const int total = __shfl_sync(0xffffffffu, first, 31);
  first -= n;

  //: THE TABLE, a warp's: a digit's pass (`kMaxPasses dl + pass`) its residues' totals, a byte each, its start, its
  //: digit's coefficient scales' exponents
  for (int i = lane; i < kKeys; i += 32) table[i] = make_uint4(0u, 0u, 0u, 0u);
  __syncwarp();

  //: the lane's share and the cell and bit it starts at
  const int share = lane * total >> 5, count = ((lane + 1) * total >> 5) - share;
  int c0 = 0, t0 = 0;
  uint32_t bits0 = 0u, slot0 = 0u;
  if (total > 0) {
    const int g = min(share, total - 1);
    int o = 0;
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
      const int x = __shfl_sync(0xffffffffu, first, o + s);
      o = x <= g ? o + s : o;
    }
    int m = g - __shfl_sync(0xffffffffu, first, o), k = 0, at = 0;
    uint32_t sums = __shfl_sync(0xffffffffu, pre[0], o);
#pragma unroll
    for (int gg = 1; gg < kGroups; ++gg) {
      const int gs = __shfl_sync(0xffffffffu, gstart[gg], o);
      const uint32_t ps = __shfl_sync(0xffffffffu, pre[gg], o);
      sums = m >= gs ? ps : sums;
      k = m >= gs ? 4 * gg : k;
      at = m >= gs ? gs : at;
    }
    m -= at;
    const int w = __popc((((static_cast<uint32_t>(m) * 0x01010101u) | 0x80808080u) - sums) & 0x80808080u) - 1;
    m -= static_cast<int>((sums >> (8 * w)) & 0xffu);
    c0 = o * kWords + k + w;
    const uint32_t word = tile[(part * kDigits + (c0 >> 4)) * kTileStride + (c0 & 15)];
    t0 = nth_bit(word, m);
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
        int x = c + 1;
        uint32_t mw = cell_mask[x >> 5] & (0xffffffffu << (x & 31));
        while (mw == 0u) {
          x = (x | 31) + 1;
          mw = cell_mask[x >> 5];
        }
        c = (x & ~31) + __ffs(mw) - 1;
        bits = tile[(part * kDigits + (c >> 4)) * kTileStride + (c & 15)];
        u = live[c & 15];
        b = bases[c & 15];
      }
      const int t = __ffs(bits) - 1;
      bits &= bits - 1u;
      visit(c, t, b + __popc(u & ((1u << t) - 1u)));
    }
  };

  //: the first walk: each entry counted into the table, the lane's last (digit, pass) run's counts left in `run`
  int cur = key0;
  bool many = false;
  uint32_t run_lo = 0u, run_hi = 0u;
  walk([&](int c, int, uint32_t slot) {
    const int key = (c >> 4) * kMaxPasses + static_cast<int>(slot) / kPassSlots, r = static_cast<int>(slot & 7u);
    const uint32_t inc = 1u << (8 * (r & 3));
    atomicAdd(reinterpret_cast<uint32_t*>(table + key) + (r >> 2), inc);
    const bool same = key == cur;
    many |= !same;
    cur = key;
    run_lo = (same ? run_lo : 0u) + (r < 4 ? inc : 0u);
    run_hi = (same ? run_hi : 0u) + (r < 4 ? 0u : inc);
  });
  //: the counts before each lane's first run: a scan of the last runs' counts, segmented where the run changes
  const int cur_before = __shfl_up_sync(0xffffffffu, cur, 1);
  uint32_t v_lo = run_lo, v_hi = run_hi;
  int head = many || lane == 0 || cur_before != cur;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const uint32_t xl = __shfl_up_sync(0xffffffffu, v_lo, off), xh = __shfl_up_sync(0xffffffffu, v_hi, off);
    const int xf = __shfl_up_sync(0xffffffffu, head, off);
    if (lane >= off) {
      v_lo += head ? 0u : xl;
      v_hi += head ? 0u : xh;
      head |= xf;
    }
  }
  const bool carried = lane > 0 && cur_before == key0;
  const uint32_t before_lo = __shfl_up_sync(0xffffffffu, v_lo, 1), before_hi = __shfl_up_sync(0xffffffffu, v_hi, 1);
  __syncwarp();

  //: a lane a digit's pass: its start from the digit's totals, the list's starts and end out
#pragma unroll
  for (int h = 0; h < kOwned; ++h) {
    const int key = 32 * h + lane, pass = key % kMaxPasses;
    const uint4 c = table[key];
    const int s = static_cast<int>(__dp4a(c.x, 0x01010101u, __dp4a(c.y, 0x01010101u, 0u)));
    int incl = s;
#pragma unroll
    for (int off = 1; off < kMaxPasses; off <<= 1) {
      const int x = __shfl_up_sync(0xffffffffu, incl, off);
      incl += pass >= off ? x : 0;
    }
    const int start = incl - s;
    const int end = __shfl_sync(0xffffffffu, incl, lane | (kMaxPasses - 1));
    const int digit = group * 32 + part * kDigits + key / kMaxPasses;
    if (digit < width) {
      uint16_t* const at = out.at + (win_index * width + digit) * kListAt;
      at[pass] = static_cast<uint16_t>(start);
      at[kMaxPasses + pass] = static_cast<uint16_t>(end);
    }
    uint32_t scales = 0u;
    if constexpr (kCoef) {
      const int d = part * kDigits + key / kMaxPasses;
      float2 m = top[unit_slot][d];
#pragma unroll
      for (int pp = 1; pp < P; ++pp) {
        const float2 o = top[unit_slot + pp * item_units][d];
        m = make_float2(fmaxf(m.x, o.x), fmaxf(m.y, o.y));
      }
      const int em = kCoefTop - max(ceil_exponent_signed(m.x), kUnitMin);
      const int ev = kCoefTop - max(ceil_exponent_signed(m.y), kUnitMin);
      scales = static_cast<uint32_t>(em + 128) | static_cast<uint32_t>(ev + 128) << 8;
    }
    table[key].z = static_cast<uint32_t>(start);
    table[key].w = scales;
  }
  __syncwarp();

  //: the second walk: each entry's rank among its residue from the run's counts so far, its place, its record
  cur = key0;
  run_lo = carried ? before_lo : 0u;
  run_hi = carried ? before_hi : 0u;
  walk([&](int c, int t, uint32_t slot) {
    const int key = (c >> 4) * kMaxPasses + static_cast<int>(slot) / kPassSlots, r = static_cast<int>(slot & 7u);
    const uint4 tc = table[key];
    const bool same = key == cur;
    cur = key;
    run_lo = same ? run_lo : 0u;
    run_hi = same ? run_hi : 0u;
    const int rank = static_cast<int>(((r < 4 ? run_lo : run_hi) >> (8 * (r & 3))) & 0xffu);
    const uint32_t inc = 1u << (8 * (r & 3));
    run_lo += r < 4 ? inc : 0u;
    run_hi += r < 4 ? 0u : inc;
    const int d = part * kDigits + (c >> 4), j = c & 15;
    const int64_t slot_at =
        (win_index * width + group * 32 + d) * kListCap + static_cast<int>(tc.z) + round_robin(tc.x, tc.y, rank, r);
    if constexpr (kCoef) {
      const uint8_t* const ring = rings + (unit_slot + j / kWords * item_units) * (kStages * kWordBytes);
      const float byte = static_cast<float>(ring[j % kWords * kWordBytes + ring_row(t) * 32 + d]);
      const float2 col = cols[column_slot(32 * j + t)];
      const float xm = byte * col.x, xv = byte * col.y;
      const float sm = pow2(static_cast<int>(tc.w & 0xffu) - 128), sv = pow2(static_cast<int>(tc.w >> 8) - 128);
      out.record[slot_at] = make_uint2(slot, __half_as_ushort(__float2half_rn(xv * sv))
                                                 | static_cast<uint32_t>(__half_as_ushort(__float2half_rn(xm * sm)))
                                                       << 16);
    } else {
      out.slot[slot_at] = static_cast<uint16_t>(slot);
    }
  });
}

//: THE WINDOW'S GAINS as f16 pairs (the den pass's operand), by every thread after the columns: each live token's
//: `(m, F)` at its window's powers of two, the largest of each in [2^-4, 2^-3) (a zero largest scaled by 1).
//: -- see docs/internals/facts/facts.md#gains
__device__ __forceinline__ int gains_shift(float top) {
  if (!(top > 0.0f)) return 0;
  return max(-126, min(127, -(static_cast<int>(__float_as_uint(top) >> 23) - 127) - 4));
}

__device__ void store_gains(const FactsParams& p, int bh, int window, const float2* cols) {
  const float2 top = cols[kColumnSlots];
  const int sm = gains_shift(top.x), sf = gains_shift(top.y);
  const float km = pow2(sm), kf = pow2(sf);
  for (int i = static_cast<int>(threadIdx.x); i < kWindowTokens; i += kFactsThreads) {
    const int t = window * kWindowTokens + i;
    if (t >= p.L) break;
    const float2 c = cols[column_slot(i)];
    p.gains16[static_cast<int64_t>(bh) * p.L + t] =
        static_cast<uint32_t>(__half_as_ushort(__float2half_rn(c.x * km)))
        | static_cast<uint32_t>(__half_as_ushort(__float2half_rn(c.y * kf))) << 16;
  }
  if (threadIdx.x == 0) {
    p.gains_exp[(static_cast<int64_t>(bh) * p.windows + window) * 2] = static_cast<int16_t>(sm);
    p.gains_exp[(static_cast<int64_t>(bh) * p.windows + window) * 2 + 1] = static_cast<int16_t>(sf);
  }
}

//: THE GRID: `kFactsBlocksPerSm` blocks an SM walk the items -- per (batch, window, head), heads innermost, the list
//: units first. An item's warps split its units' windows into parts, each through its warp's ring of `kStages` words
//: (a voted unit's into its tile, then stored), meeting at a barrier between the two and between items; a list item
//: stages the window's columns first. -- see docs/internals/facts/facts.md#the-grid
__global__ __launch_bounds__(kFactsThreads, kFactsBlocksPerSm) void facts_pass(
    const FactsParams p) {
  extern __shared__ __align__(128) uint8_t s_ring[];
  __shared__ __align__(16) uint32_t s_tile[kItemUnits][32 * kTileStride];
  __shared__ uint32_t s_live[kItemUnits][kWindowWords];
  __shared__ uint16_t s_marks[kFactsWarps][64];
  __shared__ float2 s_cols[kColumnSlots + 1];
  __shared__ float2 s_warp_top[kFactsWarps];
  __shared__ float2 s_top[kFactsWarps][32];
  __shared__ long long s_clock[kFactsWarps][Clock::kRow];
  //: THE ITEM, decomposed once by thread 0: the item, its place in its window, the window, the batch-head, its batch and
  //: head -- every warp reads them, none divides
  __shared__ int s_work[6];
  const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  uint8_t* const ring = s_ring + warp * kStages * kWordBytes;
  Clock clock(&s_clock[warp][0], lane);

  //: THE ITEMS TAKEN IN ORDER from the call's counter: a list item costs ~3x a vote item, so a static deal of a few
  //: items a block leaves its blocks unevenly loaded; taken in order, a window's items still run side by side
#pragma unroll 1
  for (;;) {
    if (threadIdx.x == 0) {
      const int taken = atomicAdd(p.counters, 1);
      const int slot = taken / p.window_items;
      const int batch_window = slot / p.H;
      const int head = slot - batch_window * p.H;
      const int batch = batch_window / p.windows;
      s_work[0] = taken;
      s_work[1] = taken - slot * p.window_items;
      s_work[2] = batch_window - batch * p.windows;
      s_work[3] = batch * p.H + head;
      s_work[4] = batch;
      s_work[5] = head;
    }
    __syncthreads();
    const int item = s_work[0];
    if (item >= p.items) break;
    const int k = s_work[1], window = s_work[2], bh = s_work[3], b = s_work[4], h = s_work[5];
    const bool list = k < p.list_items;
    const int item_units = list ? p.list_item_units : kItemUnits;
    const int slot_unit = warp % item_units, part = warp / item_units;
    const int unit = list ? k * item_units + slot_unit
                          : p.list_units + (k - p.list_items) * kItemUnits + slot_unit;
    const bool live = unit < (list ? p.list_units : p.list_units + p.vote_units);
    const int code = live ? p.unit[unit] : 0;
    const bool voted = live && !level_is_dense(p.dense[code >> 5], (code >> 3) & 3);
    uint32_t* const tile = voted ? s_tile[slot_unit] : nullptr;
    const bool listed = voted && ((code >> 3) & 3) == listed_level(p, code >> 5);
    uint32_t* const live_words = listed ? s_live[slot_unit] : nullptr;
    if (list) {
      //: a voted list level's rows are u8; a dense one's are f16 (`rola.routing.quant.DENSE_DTYPE`)
      const ColumnLoads loads = load_columns(p, b, h, window);
      LaneCopies<false> vote_copies;
      LaneCopies<true> dense_copies;
      if (live && voted)
        vote_copies = start_part<kListPartWords, false>(p, code, b, h, window, part, lane, ring);
      else if (live)
        dense_copies = start_part<kWindowWords, true>(p, code, b, h, window, part, lane, ring);
      stage_columns(loads, warp, lane, s_cols, s_warp_top,
                    k == 0 ? p.fm + static_cast<int64_t>(bh) * p.L : nullptr,
                    window * kWindowTokens, p.L);
      __syncthreads();
      if (k == 0) store_gains(p, bh, window, s_cols);
      clock.lap(kPhaseColumns);
      if (live && voted)
        read_part<true, true, kListPartWords, false>(p, code, bh, window, part, lane, tile, ring,
                                                     s_cols, s_top[warp], vote_copies, live_words);
      else if (live)
        read_part<false, true, kWindowWords, true>(p, code, bh, window, part, lane, tile, ring,
                                                   s_cols, s_top[warp], dense_copies, nullptr);
    } else {
      clock.lap(kPhaseColumns);
      if (live) {
        LaneCopies<false> copies =
            start_part<kPartWords, false>(p, code, b, h, window, part, lane, ring);
        read_part<true, false, kPartWords, false>(p, code, bh, window, part, lane, tile, ring,
                                                  s_cols, s_top[warp], copies, live_words);
      }
    }
    clock.lap(kPhaseVote);
    __syncthreads();

    if (listed) {
      //: the warp's entry table in what its item leaves idle: a list item's spare tiles, a vote item's ring
      if (list)
        build_lists<true, kListParts>(p, code, bh, window, part, lane, tile, s_live[slot_unit], s_ring,
                                      reinterpret_cast<uint4*>(s_tile[kListItemUnits]) + warp * kListTableKeys,
                                      s_marks[warp], slot_unit, item_units, s_cols, s_top);
      else
        build_lists<false, kParts>(p, code, bh, window, part, lane, tile, s_live[slot_unit], s_ring,
                                   reinterpret_cast<uint4*>(ring), s_marks[warp], slot_unit, item_units, s_cols,
                                   s_top);
    }
    clock.lap(kPhaseLists);
    if (voted) {
      if (list)
        store_part<kListParts>(p, code, bh, window, part, lane, tile);
      else
        store_part<kParts>(p, code, bh, window, part, lane, tile);
    }
    if (live && list && part == 0) {
      if (voted)
        store_lists<kListParts>(p, code, bh, window, lane, s_top, slot_unit);
      else
        store_lists<1>(p, code, bh, window, lane, s_top, slot_unit);
    }
    clock.lap(kPhaseStore);
    __syncthreads();
  }
  clock.publish(p.ledger, 0, kFactsWarps, warp);
  //: the last block out resets the counters, so they are zero between calls (the next launch follows this one)
  if (threadIdx.x == 0) {
    if (atomicAdd(p.counters + 1, 1) == static_cast<int>(gridDim.x) - 1) {
      p.counters[0] = 0;
      p.counters[1] = 0;
    }
  }
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
                        std::vector<int64_t> mask_write, int64_t list_level, int64_t read_list_level,
                        Tensor counters, int64_t window) {
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
  //: THE UNITS, the list level's first (their items stage the window's columns), then the other sparse levels'; a
  //: dense level's rows are its static mask and are never written
  STD_TORCH_CHECK(list_level >= -1 && list_level < D, "the list level is a level of the depth ", D,
                  " or -1 for none, got ", list_level);
  p.list_level = static_cast<int>(list_level);
  for (int pass = 0; pass < 2; ++pass) {
    for (int side = 0; side < kSides; ++side) {
      for (int level = 0; level < D; ++level) {
        const bool list = side == kWrite && level == p.list_level;
        if ((pass == 0) != list || (pass == 1 && level_is_dense(p.dense[side], level))) continue;
        for (int group = 0; group * 32 < p.B[level]; ++group) {
          p.unit[p.units++] = static_cast<uint8_t>(unit_code(side, level, group));
          p.vote_units += pass;
        }
      }
    }
  }
  p.list_units = p.units - p.vote_units;
  //: a dense list level's units are a warp each over the whole window (nothing voted, no tile); a voted one's
  //: `kListItemUnits` an item, their windows split
  const bool dense_list = p.list_level >= 0 && level_is_dense(p.dense[kWrite], p.list_level);
  p.list_item_units = dense_list ? kFactsWarps : kListItemUnits;
  p.list_items = (p.list_units + p.list_item_units - 1) / p.list_item_units;
  p.vote_items = (p.vote_units + kItemUnits - 1) / kItemUnits;
  p.window_items = p.list_items + p.vote_items;
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
  STD_TORCH_CHECK(read_list_level >= -1 && read_list_level < D, "the read list level is a level of the depth ", D,
                  " or -1 for none, got ", read_list_level);
  p.read_list_level = static_cast<int>(read_list_level);
  STD_TORCH_CHECK(lists.size() == 10, "lists holds 10 tensors (facts_api.md), got ", lists.size());
  const auto u16_view = [&](const Tensor& t, std::initializer_list<int64_t> want, const char* name) {
    std::vector<int64_t> dims(want);
    STD_TORCH_CHECK(t.is_cuda() && (t.scalar_type() == Dtype::Short || t.scalar_type() == Dtype::Half)
                        && t.is_contiguous() && t.dim() == static_cast<int64_t>(dims.size()),
                    name, " is a contiguous CUDA 16-bit tensor of ", dims.size(), " dimensions");
    for (size_t i = 0; i < dims.size(); ++i)
      STD_TORCH_CHECK(t.size(static_cast<int64_t>(i)) == dims[i], name, " dimension ", i, " is ", dims[i], ", got ",
                      t.size(static_cast<int64_t>(i)));
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
    o.union_tok = u16_view(lists[at + 2], {B * H, p.windows, groups, kWindowTokens}, "a union's tokens");
    o.union_count = u16_view(lists[at + 3], {B * H, p.windows, groups}, "a union's counts");
  }
  if (p.list_level >= 0) {
    p.gains16 = reinterpret_cast<uint32_t*>(u16_view(lists[8], {B * H, p.L, 2}, "the gains' f16 pairs"));
    STD_TORCH_CHECK(lists[9].is_cuda() && lists[9].scalar_type() == Dtype::Short && lists[9].is_contiguous()
                        && lists[9].numel() == B * H * p.windows * 2,
                    "the gains' exponents are a contiguous CUDA int16 [", B * H, ", ", p.windows, ", 2]");
    p.gains_exp = lists[9].mutable_data_ptr<int16_t>();
  }
  p.ledger = detail::g_ledger;

  const DeviceGuard guard(first.get_device_index());
  int sms = 0;
  ROLA_CUDA_CHECK(
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, first.get_device_index()));
  const int64_t items = int64_t{B} * H * p.windows * p.window_items;
  STD_TORCH_CHECK(items <= INT32_MAX, "the call's ", items, " work items exceed the int32 count");
  p.items = static_cast<int>(items);
  STD_TORCH_CHECK(counters.is_cuda() && counters.scalar_type() == Dtype::Int
                      && counters.is_contiguous() && counters.numel() == 2,
                  "counters is the pass's contiguous int32 [2], zero between calls");
  p.counters = counters.mutable_data_ptr<int32_t>();
  if (items == 0) return;
  const int blocks = static_cast<int>(std::min<int64_t>(items, int64_t{kFactsBlocksPerSm} * sms));
  //: ONCE: the ring is dynamic shared memory, and two blocks an SM want the shared carveout's whole
  static bool configured = false;
  if (!configured) {
    ROLA_CUDA_CHECK(cudaFuncSetAttribute(detail::facts_pass,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, kRingBytes));
    ROLA_CUDA_CHECK(cudaFuncSetAttribute(detail::facts_pass,
                                         cudaFuncAttributePreferredSharedMemoryCarveout,
                                         cudaSharedmemCarveoutMaxShared));
    configured = true;
  }
  detail::facts_pass<<<blocks, kFactsThreads, kRingBytes, current_stream()>>>(p);
  ROLA_CUDA_LAUNCH_CHECK();
}

}  // namespace facts
}  // namespace rola
