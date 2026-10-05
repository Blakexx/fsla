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
                          LaneCopies<kDense>& copies) {
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

//: THE GRID: `kFactsBlocksPerSm` blocks an SM walk the items -- per (batch, window, head), heads innermost, the list
//: units first. An item's warps split its units' windows into parts, each through its warp's ring of `kStages` words
//: (a voted unit's into its tile, then stored), meeting at a barrier between the two and between items; a list item
//: stages the window's columns first. -- see docs/internals/facts/facts.md#the-grid
__global__ __launch_bounds__(kFactsThreads, kFactsBlocksPerSm) void facts_pass(
    const FactsParams p) {
  extern __shared__ __align__(128) uint8_t s_ring[];
  __shared__ uint32_t s_tile[kItemUnits][32 * kTileStride];
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
      clock.lap(kPhaseColumns);
      if (live && voted)
        read_part<true, true, kListPartWords, false>(p, code, bh, window, part, lane, tile, ring,
                                                     s_cols, s_top[warp], vote_copies);
      else if (live)
        read_part<false, true, kWindowWords, true>(p, code, bh, window, part, lane, tile, ring,
                                                   s_cols, s_top[warp], dense_copies);
    } else {
      clock.lap(kPhaseColumns);
      if (live) {
        LaneCopies<false> copies =
            start_part<kPartWords, false>(p, code, b, h, window, part, lane, ring);
        read_part<true, false, kPartWords, false>(p, code, bh, window, part, lane, tile, ring,
                                                  s_cols, s_top[warp], copies);
      }
    }
    clock.lap(kPhaseVote);
    __syncthreads();

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
                        std::vector<int64_t> widths, int64_t dense_read,
                        std::vector<int64_t> mask_read, int64_t dense_write,
                        std::vector<int64_t> mask_write, int64_t list_level, Tensor counters,
                        int64_t window) {
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
