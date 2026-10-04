// THE FACTS PASS'S TRANSLATION UNIT -- the kernel that reads the bundle's rows once for both sides' class-1 words and
// the write lists' exponents, and the host entry that validates the bundle's views and launches it.
// see docs/internals/facts/facts.md

#include "common/torch_seam.cuh"

#include <algorithm>
#include <climits>
#include <numeric>
#include <optional>
#include <vector>

#include "common/arch_runtime.cuh"
#include "common/phase_clock.cuh"
#include "facts/facts.cuh"
#include "facts/facts_api.cuh"

namespace rola {
namespace facts {
namespace detail {

using Clock = PhaseClock<kFactsPhases>;

//: A WORD'S DIGITS, one a token: the warp's lanes one contiguous run of a row's digits, the word's 32 tokens walked,
//: a token past `L` unread.
__device__ __forceinline__ void load_word(uint8_t (&digit)[kTokBits], const uint8_t* at, int64_t st,
                                          int live, bool reads) {
#pragma unroll
  for (int i = 0; i < kTokBits; ++i) {
    digit[i] = reads && i < live ? __ldg(at) : uint8_t{0};
    at += st;
  }
}

__device__ __forceinline__ uint32_t word_bits(const uint8_t (&digit)[kTokBits]) {
  uint32_t bits = 0;
#pragma unroll
  for (int i = 0; i < kTokBits; ++i) bits |= digit_live(digit[i]) ? 1u << i : 0u;
  return bits;
}

//: A BYTE AS AN fp32 VALUE, through the mantissa: two full-rate operations where a convert is a quarter-rate one.
__device__ __forceinline__ float byte_value(uint8_t b) {
  return __uint_as_float(0x4b000000u | b) - 8388608.0f;
}

//: A WARP'S PART OF A UNIT, a (side, level, 32-digit group): its `kPartWords` of the window's words, each lane its
//: digit's bits into its tile row -- a sparse level's voted from the rows, a dense level's the width mask over the live
//: tokens -- and each 64-token tile's union, the ballot of its words. A LIST unit (the write side's list level) also
//: reads its rows, dense or not, for each digit's largest `b` x column over its tokens. -- see
//: docs/internals/facts/facts.md#vote
__device__ void vote_part(const FactsParams& p, int code, int bh, int word0, int lane,
                          uint32_t* row, const float2* cols, float2* top) {
  const int side = code >> 5, level = (code >> 3) & 3, group = code & 7;
  const int width = p.B[level], base = p.row_base[level];
  const int digit = group * 32 + lane;
  const bool in_mask = digit < width && mask_bit(p.mask[side][(base + digit) >> 5], base + digit);
  const bool dense = level_is_dense(p.dense[side], level);
  const bool list = side == kWrite && level == p.list_level;
  const int b = bh / p.H, h = bh - b * p.H;
  const RowView view = p.rows[side][level];
  const uint8_t* const column =
      static_cast<const uint8_t*>(view.base) + b * view.sb + h * view.sh + digit;
  uint32_t* const unions =
      p.unions
      + ((static_cast<int64_t>(bh) * kSides + side) * p.groups + p.group_base[level] + group)
            * p.tiles;
  const int window_t0 = (word0 / kWindowWords) * kWindowTokens;

  float2 best = make_float2(0.0f, 0.0f);
  uint32_t tile = 0u;
#pragma unroll 1
  for (int j = 0; j < kPartWords; ++j) {
    const int t0 = (word0 + j) * kTokBits;
    const int live = min(kTokBits, p.L - t0);
    if (live <= 0) break;
    const uint32_t mask_bits = in_mask ? (live == kTokBits ? ~0u : (1u << live) - 1u) : 0u;
    uint32_t bits;
    if (dense && !list) {
      bits = mask_bits;
    } else {
      uint8_t digits[kTokBits];
      load_word(digits, column + t0 * view.st, view.st, live, in_mask);
      bits = dense ? mask_bits : word_bits(digits);
      if (list) {
        const float2* const c = cols + (t0 - window_t0);
#pragma unroll
        for (int i = 0; i < kTokBits; ++i) {
          const float x = byte_value(digits[i]);
          const float2 ci = c[i];
          best.x = fmaxf(best.x, x * ci.x);
          best.y = fmaxf(best.y, x * ci.y);
        }
      }
    }
    row[j] = bits;
    tile |= bits;
    //: a tile closes at its second word, or at the call's last
    if (((word0 + j) & 1) || t0 + kTokBits >= p.L) {
      const uint32_t u = __ballot_sync(0xffffffffu, tile != 0u);
      if (lane == 0) unions[(word0 + j) >> 1] = u;
      tile = 0u;
    }
  }

  //: a dense level's zero digit at the floor: the window's largest column times it bounds every digit in the mask
  if (list && dense && in_mask) {
    best.x = fmaxf(best.x, kDenseFloorDigit * cols[kWindowTokens].x);
    best.y = fmaxf(best.y, kDenseFloorDigit * cols[kWindowTokens].y);
  }
  if (list) top[lane] = best;
}

//: A WARP'S PART OF A UNIT'S TILE OUT: its share of the 32 rows, a row's window of words one 64-byte run, two rows a
//: store.
__device__ __forceinline__ void store_part(const FactsParams& p, int code, int bh, int window,
                                           int part, int lane, const uint32_t* tile) {
  const int side = code >> 5, level = (code >> 3) & 3, group = code & 7;
  const int width = p.B[level], base = p.row_base[level];
  const int word0 = window * kWindowWords;
  const int words = min(kWindowWords, p.token_words - word0);
  const int j = lane & (kWindowWords - 1);
  uint32_t* const out = p.words
                        + (static_cast<int64_t>(bh) * kSides + side) * p.rows_total * p.token_words
                        + word0 + j;
#pragma unroll
  for (int pair = 0; pair < kPartRowPairs; ++pair) {
    const int r = 2 * (part * kPartRowPairs + pair) + (lane >> 4);
    const int d = group * 32 + r;
    if (d < width && j < words)
      out[static_cast<int64_t>(base + d) * p.token_words] = tile[r * kTileStride + j];
  }
}

__device__ __forceinline__ float column_at(const ColumnView& c, int b, int t, int h) {
  return __ldg(c.base + b * c.sb + t * c.st + h * c.sh);
}

//: THE WINDOW'S COLUMNS, staged for its list units: a token's `m = g x the write scales` (the gain first, then the
//: levels in order) and `m x s`, a dead or past-the-end token zeros, and past them the window's largest of each.
//: -- see docs/internals/facts/facts.md#lists
__device__ void stage_columns(const FactsParams& p, int bh, int window, int warp, int lane,
                              float2* cols, float2* warp_top) {
  constexpr int kPerThread = kWindowTokens / kFactsThreads;
  const int b = bh / p.H, h = bh - b * p.H;
  float2 top = make_float2(0.0f, 0.0f);
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
    const int k = i * kFactsThreads + static_cast<int>(threadIdx.x);
    const int t = window * kWindowTokens + k;
    float m = 0.0f, s = 0.0f;
    if (t < p.L) {
      m = column_at(p.g_write, b, t, h);
      s = column_at(p.v_scale, b, t, h);
#pragma unroll
      for (int l = 0; l < kMaxLevels; ++l)
        if (l < p.D) m *= column_at(p.scale_write[l], b, t, h);
    }
    cols[k] = make_float2(m, m * s);
    top = make_float2(fmaxf(top.x, m), fmaxf(top.y, m * s));
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
    cols[kWindowTokens] = m;
  }
}

//: THE LIST EXPONENT: the smallest `K` with `x <= 2^K`, an `x` below the smallest normal at its exponent.
__device__ __forceinline__ uint32_t ceil_exponent(float x) {
  const uint32_t bits = __float_as_uint(fmaxf(x, kListFloor));
  return static_cast<uint16_t>(static_cast<int>(bits >> 23) - 127 + ((bits & 0x7fffffu) != 0u));
}

//: A LIST UNIT'S EXPONENTS OUT, by its first part's warp: each digit's parts joined, a lane a digit's `(K_m, K_v)`
//: pair, the unit's 32 one 128-byte run. -- see docs/internals/facts/facts.md#lists
__device__ __forceinline__ void store_lists(const FactsParams& p, int code, int bh, int window,
                                            int lane, const float2 (*top)[32]) {
  const int digit = (code & 7) * 32 + lane;
  float2 m = top[0][lane];
#pragma unroll
  for (int part = 1; part < kParts; ++part)
    m = make_float2(fmaxf(m.x, top[part][lane].x), fmaxf(m.y, top[part][lane].y));
  const uint32_t pair = ceil_exponent(m.x) | ceil_exponent(m.y) << 16;
  if (digit < p.B[p.list_level])
    p.list_exp[(static_cast<int64_t>(bh) * p.windows + window) * p.B[p.list_level] + digit] = pair;
}

//: THE GRID: `kFactsBlocksPerSm` blocks an SM walk the items -- per (batch-head, window), its units `kItemUnits` at a
//: time, the list units first. An item's warps split each unit's window `kParts` ways into the unit's tile, then store
//: it, meeting at a barrier between the two and between items; an item holding a list unit stages the window's
//: columns first.
//: -- see docs/internals/facts/facts.md#the-grid
__global__ __launch_bounds__(kFactsThreads, kFactsBlocksPerSm) void facts_pass(
    const FactsParams p) {
  __shared__ uint32_t s_tile[kItemUnits][32 * kTileStride];
  __shared__ float2 s_cols[kWindowTokens + 1];
  __shared__ float2 s_warp_top[kFactsWarps];
  __shared__ float2 s_top[kItemUnits][kParts][32];
  __shared__ long long s_clock[kFactsWarps][Clock::kRow];
  const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  Clock clock(&s_clock[warp][0], lane);
#pragma unroll 1
  for (int item = static_cast<int>(blockIdx.x); item < p.items;
       item += static_cast<int>(gridDim.x)) {
    const int slot = item / p.window_items;
    const int k = item - slot * p.window_items;
    const int bh = slot / p.windows, window = slot - bh * p.windows;
    const bool list = k < p.list_items;
    if (list) {
      stage_columns(p, bh, window, warp, lane, s_cols, s_warp_top);
      __syncthreads();
    }
    clock.lap(kPhaseColumns);
    const int slot_unit = warp % kItemUnits, part = warp / kItemUnits;
    const int unit = k * kItemUnits + slot_unit;
    const bool live = unit < p.units;
    uint32_t* const tile = s_tile[slot_unit];
    if (live) {
      const int word0 = window * kWindowWords + part * kPartWords;
      uint32_t* const row = tile + lane * kTileStride + part * kPartWords;
      vote_part(p, p.unit[unit], bh, word0, lane, row, s_cols, s_top[slot_unit][part]);
    }
    clock.lap(kPhaseVote);
    __syncthreads();
    if (live) store_part(p, p.unit[unit], bh, window, part, lane, tile);
    if (live && part == 0 && is_list_unit(p, p.unit[unit]))
      store_lists(p, p.unit[unit], bh, window, lane, s_top[slot_unit]);
    clock.lap(kPhaseStore);
    __syncthreads();
  }
  clock.publish(p.ledger, 0, kFactsWarps, warp);
}

//: A level's rows as the kernel reads them: `[B, T, H, width]` u8, digits contiguous.
RowView row_view(const Tensor& t, int64_t B, int64_t T, int64_t H, const char* name) {
  STD_TORCH_CHECK(t.is_cuda() && t.scalar_type() == Dtype::Byte, name, " is a CUDA u8 tensor, got ",
                  t.scalar_type());
  STD_TORCH_CHECK(t.dim() == 4 && t.size(0) == B && t.size(1) == T && t.size(2) == H, name,
                  " is [B, T, H, width] = [", B, ", ", T, ", H, width], got ", shape(t.sizes()));
  STD_TORCH_CHECK(t.stride(3) == 1 || t.size(3) == 1, name, "'s digits are contiguous");
  return RowView{t.const_data_ptr(), t.stride(0), t.stride(1), t.stride(2)};
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
                        Tensor words, Tensor unions, Tensor list_exp, std::vector<int64_t> widths,
                        int64_t dense_read, std::vector<int64_t> mask_read, int64_t dense_write,
                        std::vector<int64_t> mask_write, int64_t list_level, int64_t window) {
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
  //: THE UNITS, the list level's first (their items stage the window's columns), then the other sparse levels', then
  //: the dense levels' (their mask words, nothing read)
  STD_TORCH_CHECK(list_level >= -1 && list_level < D, "the list level is a level of the depth ", D,
                  " or -1 for none, got ", list_level);
  p.list_level = static_cast<int>(list_level);
  for (int pass = 0; pass < 3; ++pass) {
    for (int side = 0; side < kSides; ++side) {
      for (int level = 0; level < D; ++level) {
        const bool list = side == kWrite && level == p.list_level;
        const bool dense = level_is_dense(p.dense[side], level);
        if ((pass == 0) != list || (pass == 1 && dense) || (pass == 2 && !dense)) continue;
        for (int group = 0; group * 32 < p.B[level]; ++group)
          p.unit[p.units++] = static_cast<uint8_t>(unit_code(side, level, group));
      }
    }
  }
  p.list_items =
      p.list_level < 0 ? 0 : (p.B[p.list_level] + 32 * kItemUnits - 1) / (32 * kItemUnits);
  p.window_items = (p.units + kItemUnits - 1) / kItemUnits;
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
  }
  p.ledger = detail::g_ledger;

  const DeviceGuard guard(first.get_device_index());
  int sms = 0;
  ROLA_CUDA_CHECK(
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, first.get_device_index()));
  const int64_t items = int64_t{B} * H * p.windows * p.window_items;
  STD_TORCH_CHECK(items <= INT32_MAX, "the call's ", items, " work items exceed the int32 count");
  p.items = static_cast<int>(items);
  //: THE GRID: an item a block, or, when the blocks walk the items, a count coprime to a window's items -- a block's
  //: stride then deals it every item position (a list item heavy, a dense one light) while the blocks at work together
  //: stay on neighbouring items
  int blocks = static_cast<int>(std::min<int64_t>(items, int64_t{kFactsBlocksPerSm} * sms));
  while (blocks < items && std::gcd(blocks, p.window_items) != 1) --blocks;
  detail::facts_pass<<<blocks, kFactsThreads, 0, current_stream()>>>(p);
  ROLA_CUDA_LAUNCH_CHECK();
}

}  // namespace facts
}  // namespace rola
