// THE FACTS PASS'S TRANSLATION UNIT -- the kernel that reads the bundle's rows once for both sides' class-1 words and
// the fold's window scales, and the host entry that validates the bundle's views and launches it.
// see docs/internals/facts/facts.md

#include "common/torch_seam.cuh"

#include <algorithm>
#include <climits>
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
template <typename Digit>
__device__ __forceinline__ void load_word(Digit (&digit)[kTokBits], const Digit* at, int64_t st,
                                          int live, bool reads) {
#pragma unroll
  for (int i = 0; i < kTokBits; ++i) {
    digit[i] = reads && i < live ? __ldg(at) : Digit(0);
    at += st;
  }
}

template <typename Digit>
__device__ __forceinline__ uint32_t word_bits(const Digit (&digit)[kTokBits]) {
  uint32_t bits = 0;
#pragma unroll
  for (int i = 0; i < kTokBits; ++i) bits |= digit_live(digit[i]) ? 1u << i : 0u;
  return bits;
}

//: A WARP'S PART OF A UNIT, a (side, level, 32-digit group): its `kPartWords` of the window's words, each lane its
//: digit's bits into its tile row -- a sparse level's voted from the rows, a dense level's the width mask over the live
//: tokens. -- see docs/internals/facts/facts.md#vote
template <typename Digit>
__device__ void vote_part(const FactsParams& p, int code, int bh, int word0, int lane,
                          uint32_t* row) {
  const int side = code >> 5, level = (code >> 3) & 3, group = code & 7;
  const int width = p.B[level], base = p.row_base[level];
  const int digit = group * 32 + lane;
  const bool in_mask = digit < width && mask_bit(p.mask[side][(base + digit) >> 5], base + digit);
  const bool dense = level_is_dense(p.dense[side], level);
  const int b = bh / p.H, h = bh - b * p.H;
  const RowView view = p.rows[side][level];
  const Digit* const column =
      static_cast<const Digit*>(view.base) + b * view.sb + h * view.sh + digit;
#pragma unroll 1
  for (int j = 0; j < kPartWords; ++j) {
    const int t0 = (word0 + j) * kTokBits;
    const int live = min(kTokBits, p.L - t0);
    if (live <= 0) break;
    if (dense) {
      row[j] = in_mask ? (live == kTokBits ? ~0u : (1u << live) - 1u) : 0u;
    } else {
      Digit digits[kTokBits];
      load_word(digits, column + t0 * view.st, view.st, live, in_mask);
      row[j] = word_bits(digits);
    }
  }
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

//: ZETA, a block a window: the largest `g m_w` and `g m_w s_t` over its tokens, `m_w` the write levels' scales
//: multiplied in level order after the gain, floored at the smallest normal. -- see docs/internals/facts/facts.md#zeta
__device__ void window_zeta(const FactsParams& p, int bh, int window, int warp, int lane,
                            float (*s_max)[2]) {
  constexpr int kPerThread = kWindowTokens / kFactsThreads;
  const int b = bh / p.H, h = bh - b * p.H;
  float col = 0.0f, col_s = 0.0f;
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
    const int t = window * kWindowTokens + i * kFactsThreads + static_cast<int>(threadIdx.x);
    if (t < p.L) {
      float m = column_at(p.g_write, b, t, h);
      const float s = column_at(p.v_scale, b, t, h);
#pragma unroll
      for (int l = 0; l < kMaxLevels; ++l)
        if (l < p.D) m *= column_at(p.scale_write[l], b, t, h);
      col = fmaxf(col, m);
      col_s = fmaxf(col_s, m * s);
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    col = fmaxf(col, __shfl_xor_sync(0xffffffffu, col, offset));
    col_s = fmaxf(col_s, __shfl_xor_sync(0xffffffffu, col_s, offset));
  }
  if (lane == 0) {
    s_max[warp][0] = col;
    s_max[warp][1] = col_s;
  }
  __syncthreads();
  if (threadIdx.x < 2) {
    float m = 0.0f;
#pragma unroll
    for (int w = 0; w < kFactsWarps; ++w) m = fmaxf(m, s_max[w][threadIdx.x]);
    p.zeta[(static_cast<int64_t>(bh) * p.windows + window) * 2 + threadIdx.x] =
        fmaxf(m, kZetaFloor);
  }
}

//: THE GRID: `kFactsBlocksPerSm` blocks an SM walk the items -- per (batch-head, window), its units `kItemUnits` at a
//: time, then its zeta. An item's warps split each unit's window `kParts` ways into the unit's tile, then store it,
//: meeting at a barrier between the two and between items. -- see docs/internals/facts/facts.md#the-grid
__global__ __launch_bounds__(kFactsThreads, kFactsBlocksPerSm) void facts_pass(
    const FactsParams p) {
  __shared__ uint32_t s_tile[kItemUnits][32 * kTileStride];
  __shared__ float s_max[kFactsWarps][2];
  __shared__ long long s_clock[kFactsWarps][Clock::kRow];
  const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  const int slot_unit = warp % kItemUnits, part = warp / kItemUnits;
  Clock clock(&s_clock[warp][0], lane);
  const int groups = (p.units + kItemUnits - 1) / kItemUnits;
  const int per_window = groups + 1;
#pragma unroll 1
  for (int item = static_cast<int>(blockIdx.x); item < p.items;
       item += static_cast<int>(gridDim.x)) {
    const int slot = item / per_window;
    const int k = item - slot * per_window;
    const int bh = slot / p.windows, window = slot - bh * p.windows;
    if (k == groups) {
      window_zeta(p, bh, window, warp, lane, s_max);
      clock.lap(kPhaseZeta);
    } else {
      const int unit = k * kItemUnits + slot_unit;
      uint32_t* const tile = s_tile[slot_unit];
      if (unit < p.units) {
        const int word0 = window * kWindowWords + part * kPartWords;
        uint32_t* const row = tile + lane * kTileStride + part * kPartWords;
        if (p.digit_bytes == 1) {
          vote_part<uint8_t>(p, p.unit[unit], bh, word0, lane, row);
        } else {
          vote_part<uint16_t>(p, p.unit[unit], bh, word0, lane, row);
        }
      }
      clock.lap(kPhaseVote);
      __syncthreads();
      if (unit < p.units) store_part(p, p.unit[unit], bh, window, part, lane, tile);
      clock.lap(kPhaseStore);
    }
    __syncthreads();
  }
  clock.publish(p.ledger, 0, kFactsWarps, warp);
}

//: A level's rows as the kernel reads them: `[B, T, H, width]`, digits contiguous.
RowView row_view(const Tensor& t, int64_t B, int64_t T, int64_t H, Dtype dtype, const char* name) {
  STD_TORCH_CHECK(t.is_cuda() && t.scalar_type() == dtype, name, " is a CUDA ", dtype,
                  " tensor like the bundle's first row, got ", t.scalar_type());
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
                        Tensor words, Tensor zeta, std::vector<int64_t> widths, int64_t dense_read,
                        std::vector<int64_t> mask_read, int64_t dense_write,
                        std::vector<int64_t> mask_write, int64_t window) {
  check_arch_table();
  const int D = static_cast<int>(widths.size());
  STD_TORCH_CHECK(D >= 1 && D <= kMaxLevels, "the routing depth is one to four levels, got ", D);
  STD_TORCH_CHECK(static_cast<int>(read.size()) == D && static_cast<int>(write.size()) == D
                      && static_cast<int>(scale_write.size()) == D,
                  "each side holds one row set a level and the write side one scale a level: ", D);
  STD_TORCH_CHECK(window == kWindowTokens, "the facts pass's window is the carry's ", kWindowTokens,
                  " tokens, got ", window);
  const Tensor& first = read[0];
  const Dtype dtype = first.scalar_type();
  STD_TORCH_CHECK(dtype == Dtype::Byte || dtype == Dtype::Half,
                  "the bundle's rows are u8 or f16, got ", dtype);
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
  p.digit_bytes = dtype == Dtype::Byte ? 1 : 2;
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
      p.rows[side][level] = detail::row_view(rows, B, T, H, dtype, side ? "write[l]" : "read[l]");
      //: a digit the mask names is read, so it lies inside the level's row
      const int64_t row_width = rows.size(3);
      for (int digit = 0; digit < p.B[level]; ++digit) {
        const int row = p.row_base[level] + digit;
        STD_TORCH_CHECK(digit < row_width || !mask_bit(p.mask[side][row >> 5], row), "side ", side,
                        " level ", level, "'s mask names digit ", digit, " past its ", row_width,
                        "-digit rows");
      }
      if (level_is_dense(p.dense[side], level)) continue;
      for (int group = 0; group * 32 < p.B[level]; ++group)
        p.unit[p.units++] = static_cast<uint8_t>(unit_code(side, level, group));
    }
  }
  //: a dense level's units write its mask words, read nothing
  for (int side = 0; side < kSides; ++side) {
    for (int level = 0; level < D; ++level) {
      if (!level_is_dense(p.dense[side], level)) continue;
      for (int group = 0; group * 32 < p.B[level]; ++group)
        p.unit[p.units++] = static_cast<uint8_t>(unit_code(side, level, group));
    }
  }
  for (int level = 0; level < D; ++level)
    p.scale_write[level] = detail::column_view(scale_write[level], B, T, H, "scale_write[l]");
  p.g_write = detail::column_view(g_write, B, T, H, "g_write");
  p.v_scale = detail::column_view(v_scale, B, T, H, "v_scale");
  STD_TORCH_CHECK(words.is_cuda() && words.scalar_type() == Dtype::Int && words.is_contiguous()
                      && words.dim() == 4 && words.size(0) == B * H && words.size(1) == kSides
                      && words.size(2) == p.rows_total && words.size(3) == p.token_words,
                  "words is the contiguous int32 class-1 table [", B * H, ", 2, ", p.rows_total,
                  ", ", p.token_words, "], got ", shape(words.sizes()));
  STD_TORCH_CHECK(zeta.is_cuda() && zeta.scalar_type() == Dtype::Float && zeta.is_contiguous()
                      && zeta.numel() == B * H * p.windows * 2,
                  "zeta is the contiguous fp32 [", B * H, ", ", p.windows, ", 2] window scales");
  p.words = reinterpret_cast<uint32_t*>(words.mutable_data_ptr<int32_t>());
  p.zeta = zeta.mutable_data_ptr<float>();
  p.ledger = detail::g_ledger;

  const DeviceGuard guard(first.get_device_index());
  int sms = 0;
  ROLA_CUDA_CHECK(
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, first.get_device_index()));
  const int64_t items = B * H * p.windows * ((p.units + kItemUnits - 1) / kItemUnits + 1);
  STD_TORCH_CHECK(items <= INT32_MAX, "the call's ", items, " work items exceed the int32 count");
  p.items = static_cast<int>(items);
  const int blocks = static_cast<int>(std::min<int64_t>(items, int64_t{kFactsBlocksPerSm} * sms));
  detail::facts_pass<<<blocks, kFactsThreads, 0, current_stream()>>>(p);
  ROLA_CUDA_LAUNCH_CHECK();
}

}  // namespace facts
}  // namespace rola
