// csrc/rola/src/intra/intra_kernel.cuh -- THE INTRA KERNEL: a CTA a (query head, window, piece of a block's sweep),
// the block's reader rows resident in shared memory, the writer tiles streamed through three stage buffers, each refilled
// by its last reader, under the slab gate the facts pass's unions give; each level's Gram as f16 MMAs on the rows as
// stored (a u8 digit widened in registers), the epilogue into the value product's f16 weights, the value product on the
// f16 values, and the block's rows reduced into the output planes at the end.
// see docs/internals/intra/intra_kernel.md
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#include "common/ops.cuh"
#include "common/phase_clock.cuh"
#include "common/structure_switch.cuh"
#include "facts/liveness_contract.cuh"
#include "intra/intra.cuh"

namespace rola::intra::detail {

namespace ops = denseref::ops;

//: THE COMPILING CARD'S WALK: its window, its block of reader rows, a warp 16 of them.
constexpr int kBlockRows = block_rows_for(::denseref::arch::kArch);
static_assert(kBlockRows > 0,
              "an untabulated compute capability has no intra block: add it to block_rows_for");
constexpr int kWindowTiles = kWindow / kTile;
constexpr int kRowTiles = kBlockRows / kTile;
constexpr int kWarpCols = warp_cols_for(::denseref::arch::kArch);
constexpr int kColWarps = kTile / kWarpCols;
constexpr int kRowWarps = kBlockRows / kWarpRows;
constexpr int kWarps = kRowWarps * kColWarps;
constexpr int kThreads = kWarps * 32;
//: a warp's n-tiles (8 writers each), their pairs (an ldmatrix.x4's two), a tile's channel pairs and union quads
constexpr int kNT = kWarpCols / 8;
constexpr int kNTPairs = kNT / 2;
constexpr int kChannelPairs = kValueWidth / 16;
constexpr int kWindowQuads = kWindowTiles / 4;
static_assert(kWarps == warps_for(::denseref::arch::kArch) && kWarps <= kLedgerWarps,
              "the walk's warps are the card table's");
using Pieces = PiecesAt<kWindow, kBlockRows>;
using Clock = PhaseClock<kIntraPhases, true>;

//: THE WINDOW'S PIECES where a CTA reads its own: one uniform constant load
__constant__ Pieces kPieceTable = Pieces{};

//: THE SHARED TENANTS at (W, block rows): three writer stages, the tile's values and writer factors, the warps' row
//: sums, the slab gate's table and the pipeline's mbarriers; the block's reader rows follow, `pitch` bytes a row.
//: -- see docs/internals/intra/intra_kernel.md#smem
template <int W, int Rows, int Warps>
struct alignas(128) SmemAt {
  uint8_t stage[kStageBuffers][kTile][kStageBytes];
  uint8_t values[kTile][kValueBytes];
  float2 fm[kTile];
  float den[Warps][kWarpRows];
  uint32_t gate[W / kTile][Rows / kTile];
  uint64_t full[kStageBuffers];
  uint64_t reader_full[kMaxStages], values_full;
  int released[kStageBuffers];
  int values_released[2];
  long long clock[Warps][Clock::kRow];
};

using Smem = SmemAt<kWindow, kBlockRows, kWarps>;

__device__ __forceinline__ uint8_t* reader_rows(Smem& sm) {
  return reinterpret_cast<uint8_t*>(&sm + 1);
}

//: A ROW'S 16-BYTE CHUNK `c` lies at chunk `c ^ swizzle(row)`: eight rows at one chunk in eight bank groups (ldmatrix),
//: four rows' 32-byte pairs in four (the pair reads). -- see docs/internals/intra/intra_kernel.md#swizzle
__device__ __forceinline__ int swizzle(int row) { return ((row & 3) << 1) | ((row >> 2) & 1); }

__device__ __forceinline__ const uint8_t* row_at(const RowView& v, int b, int t, int h) {
  return static_cast<const uint8_t*>(v.base) + b * v.sb + h * v.sh + static_cast<int64_t>(t) * v.st;
}

//: `rows` token strides of `st` bytes: one unsigned 32 x 32 -> 64 multiply (the seam bounds a stride to 31 bits)
__device__ __forceinline__ uint64_t rows_of(int rows, int64_t st) {
  return static_cast<uint64_t>(static_cast<uint32_t>(rows)) * static_cast<uint32_t>(st);
}

//: A u8 WORD'S FOUR DIGITS as two f16 pairs, (0, 1) and (2, 3), exact: at fp16 accumulate `d / 16` (the byte under
//: 0x54 is `64 + d / 16`, less 64), at fp32 the f16 subnormal `d 2^-24` (the byte alone).
//: -- see docs/internals/intra/intra_kernel.md#widening
template <bool kF16>
__device__ __forceinline__ void widen(uint32_t x, uint32_t& lo, uint32_t& hi) {
  if constexpr (kF16) {
    constexpr uint32_t kExponent = 0x54545454u, kOne = 0x3C003C00u, kLess64 = 0xD400D400u;
    lo = ops::hfma2(ops::prmt(x, kExponent, 0x4140u), kOne, kLess64);
    hi = ops::hfma2(ops::prmt(x, kExponent, 0x4342u), kOne, kLess64);
  } else {
    lo = ops::prmt(x, 0u, 0x4140u);
    hi = ops::prmt(x, 0u, 0x4342u);
  }
}

//: THE WARP'S STATE ACROSS A PIECE: a level's Gram (f16 pairs or fp32), the running level product and the output
template <bool kF16>
struct Acc;

template <>
struct Acc<true> {
  uint32_t gram[kNT][2];
  float product[kNT][4];
  float out[8][4];
};

template <>
struct Acc<false> {
  float gram[kNT][4];
  float product[kNT][4];
  float out[8][4];
};

template <bool kF16>
__device__ __forceinline__ void gram_mma(Acc<kF16>& acc, int n, const uint32_t (&a)[4], uint32_t b0,
                                         uint32_t b1) {
  if constexpr (kF16)
    ops::hmma_f16(acc.gram[n], a, b0, b1);
  else
    ops::hmma(acc.gram[n], a, b0, b1);
}

//: an n-tile's Gram as fp32, rows g (0, 1) and g + 8 (2, 3)
template <bool kF16>
__device__ __forceinline__ void gram_values(const Acc<kF16>& acc, int n, float (&v)[4]) {
  if constexpr (kF16) {
    const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&acc.gram[n][0]));
    const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&acc.gram[n][1]));
    v[0] = lo.x, v[1] = lo.y, v[2] = hi.x, v[3] = hi.y;
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = acc.gram[n][i];
  }
}

template <bool kF16>
__device__ __forceinline__ void zero_gram(Acc<kF16>& acc) {
  constexpr int kWords = kF16 ? 2 : 4;
#pragma unroll
  for (int n = 0; n < kNT; ++n)
#pragma unroll
    for (int i = 0; i < kWords; ++i) acc.gram[n][i] = 0;
}

//: A LANE'S FIXED PLACES in the operand loads: its reader rows and the writer tokens it addresses, with their swizzles.
//: -- see docs/internals/intra/intra_kernel.md#gram
struct Lanes {
  uint32_t a_row, f_row, f_row8, b_row, g_row;
  int a_hi, a_swz, f_h, f_swz;
  uint32_t f_off, b_q16, g_q16;

  __device__ __forceinline__ Lanes(Smem& sm, const IntraParams& p, int group, int col0, int lane) {
    const int g = lane >> 2, tq = lane & 3;
    const uint32_t rows = ops::smem_addr(reader_rows(sm));
    a_row = rows + static_cast<uint32_t>((group * kWarpRows + (lane & 15)) * p.pitch);
    a_hi = lane >> 4;
    a_swz = swizzle(lane & 15);
    f_row = rows + static_cast<uint32_t>((group * kWarpRows + g) * p.pitch);
    f_row8 = f_row + static_cast<uint32_t>(8 * p.pitch);
    f_h = tq >> 1;
    f_swz = swizzle(g);
    f_off = static_cast<uint32_t>(8 * (tq & 1));
    const int brow = (lane & 7) + ((lane >> 4) << 3);
    b_row = static_cast<uint32_t>((col0 + brow) * kStageBytes);
    b_q16 = static_cast<uint32_t>((((lane >> 3) & 1) ^ swizzle(lane & 7)) << 4);
    g_row = static_cast<uint32_t>((col0 + g) * kStageBytes + 8 * (tq & 1));
    g_q16 = static_cast<uint32_t>(((tq >> 1) ^ swizzle(g)) << 4);
  }
};

//: A LEVEL'S K ORDER: f16 on both sides the MMA's own (ldmatrix both); with a u8 side the u8 fragment's, a lane's four
//: consecutive digits a row (slot 2t, 2t + 1 <- digits 4t, 4t + 1; slot 2t + 8, 2t + 9 <- 4t + 2, 4t + 3), the f16
//: side read in it. -- see docs/internals/intra/intra_kernel.md#gram
template <int F>
constexpr bool kNatural = form_read_f16(F) && form_write_f16(F);

//: ONE SLAB'S OPERANDS as loaded (two k-steps of 16 digits): the warp's 16 readers' A words and the stage's 64 writer
//: tokens' B words, a u8 operand still four digits a word.
template <int F>
struct SlabFrags {
  uint32_t a[form_read_f16(F) ? 8 : 4];
  uint32_t b[form_write_f16(F) ? 4 * kNT : 2 * kNT];
};

//: A SLAB'S LOADS, each landing in the registers its MMA reads: ldmatrix for a u8 operand and for both of a natural
//: level; for the f16 reader of a u8 writer four 32-bit reads a k-step, for the f16 writer of a u8 reader a 64-bit read
//: an n-tile
template <int F, int S>
__device__ __forceinline__ void load_slab(const Lanes& ln, int rc, uint32_t buf, SlabFrags<F>& fr) {
  if constexpr (!form_read_f16(F)) {
    ops::load_frag(*reinterpret_cast<uint32_t(*)[4]>(&fr.a[0]),
                   ln.a_row + static_cast<uint32_t>(((rc + 2 * S + ln.a_hi) ^ ln.a_swz) << 4));
  } else if constexpr (kNatural<F>) {
#pragma unroll
    for (int k = 0; k < 2; ++k)
      ops::load_frag(
          *reinterpret_cast<uint32_t(*)[4]>(&fr.a[4 * k]),
          ln.a_row + static_cast<uint32_t>(((rc + 2 * (2 * S + k) + ln.a_hi) ^ ln.a_swz) << 4));
  } else {
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const uint32_t at =
          static_cast<uint32_t>(((rc + 2 * (2 * S + k) + ln.f_h) ^ ln.f_swz) << 4) + ln.f_off;
      fr.a[4 * k] = ops::load_shared_u32(ln.f_row + at);
      fr.a[4 * k + 1] = ops::load_shared_u32(ln.f_row8 + at);
      fr.a[4 * k + 2] = ops::load_shared_u32(ln.f_row + at + 4);
      fr.a[4 * k + 3] = ops::load_shared_u32(ln.f_row8 + at + 4);
    }
  }

  if constexpr (!form_write_f16(F)) {
#pragma unroll
    for (int j = 0; j < kNTPairs; ++j)
      ops::load_frag(*reinterpret_cast<uint32_t(*)[4]>(&fr.b[4 * j]),
                     buf + ln.b_row + (ln.b_q16 ^ (S << 5)) + j * 16 * kStageBytes);
  } else if constexpr (kNatural<F>) {
#pragma unroll
    for (int k = 0; k < 2; ++k)
#pragma unroll
      for (int j = 0; j < kNTPairs; ++j)
        ops::load_frag(*reinterpret_cast<uint32_t(*)[4]>(&fr.b[2 * kNT * k + 4 * j]),
                       buf + ln.b_row + (ln.b_q16 ^ ((2 * S + k) << 5)) + j * 16 * kStageBytes);
  } else {
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const uint32_t at = buf + ln.g_row + (ln.g_q16 ^ ((2 * S + k) << 5));
#pragma unroll
      for (int n = 0; n < kNT; ++n) {
        const uint2 b = ops::load_shared_u64(at + n * 8 * kStageBytes);
        fr.b[2 * (kNT * k + n)] = b.x, fr.b[2 * (kNT * k + n) + 1] = b.y;
      }
    }
  }
}

//: A SLAB'S 16 MMAs, a k-step's eight n-tiles apart before the next k-step's, u8 words widened as they are used
template <int F, bool kF16>
__device__ __forceinline__ void mma_slab(const SlabFrags<F>& fr, Acc<kF16>& acc) {
  uint32_t a[2][4];
  if constexpr (!form_read_f16(F)) {
    widen<kF16>(fr.a[0], a[0][0], a[0][2]);
    widen<kF16>(fr.a[1], a[0][1], a[0][3]);
    widen<kF16>(fr.a[2], a[1][0], a[1][2]);
    widen<kF16>(fr.a[3], a[1][1], a[1][3]);
  } else {
#pragma unroll
    for (int k = 0; k < 2; ++k)
#pragma unroll
      for (int i = 0; i < 4; ++i) a[k][i] = fr.a[4 * k + i];
  }
#pragma unroll
  for (int k = 0; k < 2; ++k)
#pragma unroll
    for (int n = 0; n < kNT; ++n) {
      if constexpr (!form_write_f16(F)) {
        uint32_t lo, hi;
        widen<kF16>(fr.b[4 * (n >> 1) + 2 * (n & 1) + k], lo, hi);
        gram_mma(acc, n, a[k], lo, hi);
      } else if constexpr (kNatural<F>) {
        //: ldmatrix's quads: an n-tile pair's (k 0-7, k 8-15) words for its first then its second n-tile
        const int w = 2 * kNT * k + 4 * (n >> 1) + 2 * (n & 1);
        gram_mma(acc, n, a[k], fr.b[w], fr.b[w + 1]);
      } else {
        gram_mma(acc, n, a[k], fr.b[2 * (kNT * k + n)], fr.b[2 * (kNT * k + n) + 1]);
      }
    }
}

//: THE GRAM OF ONE STAGE under the warp's slab bits: when every slab is live one straight-line block, each slab's loads
//: issued before the slab ahead of it multiplies; otherwise a branch a slab.
template <int F, bool kF16>
__device__ __forceinline__ void gram_stage(const Lanes& ln, int rc, uint32_t buf, uint32_t slabs,
                                           Acc<kF16>& acc) {
  constexpr int kSlabs = form_stage_digits(F) / kSlabDigits;
  if (slabs == (1u << kSlabs) - 1u) {
    SlabFrags<F> fr[2];
    load_slab<F, 0>(ln, rc, buf, fr[0]);
    rola::static_for<kSlabs>([&](auto S) {
      constexpr int kS = decltype(S)::value;
      if constexpr (kS + 1 < kSlabs) load_slab<F, kS + 1>(ln, rc, buf, fr[(kS + 1) & 1]);
      mma_slab<F, kF16>(fr[kS & 1], acc);
    });
    return;
  }
  rola::static_for<kSlabs>([&](auto S) {
    constexpr int kS = decltype(S)::value;
    if ((slabs >> kS) & 1u) {
      SlabFrags<F> fr;
      load_slab<F, kS>(ln, rc, buf, fr);
      mma_slab<F, kF16>(fr, acc);
    }
  });
}

//: A LEVEL'S GRAM INTO THE RUNNING PRODUCT: the first level lifted to digit units (an exact power of two), each later
//: one multiplied in.
template <bool kF16>
__device__ __forceinline__ void fold_level(Acc<kF16>& acc, bool first, float lift) {
  if (first) {
#pragma unroll
    for (int n = 0; n < kNT; ++n) {
      float v[4];
      gram_values(acc, n, v);
#pragma unroll
      for (int i = 0; i < 4; ++i) acc.product[n][i] = v[i] * lift;
    }
  } else {
#pragma unroll
    for (int n = 0; n < kNT; ++n) {
      float v[4];
      gram_values(acc, n, v);
#pragma unroll
      for (int i = 0; i < 4; ++i) acc.product[n][i] *= v[i];
    }
  }
}

//: THE EPILOGUE: the level product times each writer's factors -- `m` into the row sums `den`, `m s` into the value
//: product's weights -- zero above the diagonal on a diagonal tile. -- see docs/internals/intra/intra_kernel.md#epilogue
template <bool kF16>
__device__ __forceinline__ void epilogue(Smem& sm, float unit, bool diagonal, int warp, int group,
                                         int col0, int lane, Acc<kF16>& acc) {
  const int g = lane >> 2, tq = lane & 3;
  const int row0 = (group % (kRowWarps / kRowTiles)) * kWarpRows + g;
  float den[2] = {0.0f, 0.0f};
#pragma unroll
  for (int n = 0; n < kNT; ++n) {
    const uint4 w = ops::load_shared_v4(ops::smem_addr(&sm.fm[col0 + 8 * n + 2 * tq]));
    const float fs[2] = {__uint_as_float(w.x), __uint_as_float(w.z)};
    const float m[2] = {__uint_as_float(w.y), __uint_as_float(w.w)};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int col = col0 + 8 * n + 2 * tq + (i & 1), row = row0 + 8 * (i >> 1);
      const float pr = diagonal && col > row ? 0.0f : acc.product[n][i];
      den[i >> 1] += pr * m[i & 1];
      acc.product[n][i] = pr * fs[i & 1];
    }
  }
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    den[r] += __shfl_xor_sync(0xffffffffu, den[r], 1);
    den[r] += __shfl_xor_sync(0xffffffffu, den[r], 2);
  }
  if (tq == 0) {
    sm.den[warp][g] += den[0] * unit;
    sm.den[warp][g + 8] += den[1] * unit;
  }
}

//: THE VALUE PRODUCT: each reader's weights at its own power of two (its largest in [1, 2)) as f16, packed from the
//: accumulator's registers into the A fragments; the values by ldmatrix.trans; a channel pair's sums back at the
//: reader's power of two into the output. -- see docs/internals/intra/intra_kernel.md#value-product
template <bool kF16>
__device__ __forceinline__ void value_product(const Smem& sm, float unit, int col0, int lane,
                                              Acc<kF16>& acc) {
  constexpr int kSteps = kNTPairs;
  float m[2] = {0.0f, 0.0f};
#pragma unroll
  for (int n = 0; n < kNT; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) m[i >> 1] = fmaxf(m[i >> 1], acc.product[n][i]);
  float scale[2], undo[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    m[r] = fmaxf(m[r], __shfl_xor_sync(0xffffffffu, m[r], 1));
    m[r] = fmaxf(m[r], __shfl_xor_sync(0xffffffffu, m[r], 2));
    const uint32_t e = max((__float_as_uint(m[r]) >> 23) & 0xffu, 1u);
    scale[r] = __uint_as_float((254u - e) << 23);
    undo[r] = __uint_as_float(e << 23) * unit;
  }

  uint32_t a[kSteps][4];
#pragma unroll
  for (int kk = 0; kk < kSteps; ++kk)
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const float(&c)[4] = acc.product[2 * kk + half];
      a[kk][2 * half] = ops::pack_half2(c[0] * scale[0], c[1] * scale[0]);
      a[kk][2 * half + 1] = ops::pack_half2(c[2] * scale[1], c[3] * scale[1]);
    }

  const uint32_t values = ops::smem_addr(&sm.values[0][0]);
  const int token = col0 + (lane & 15);
#pragma unroll
  for (int jc = 0; jc < kChannelPairs; ++jc) {
    std::conditional_t<kF16, uint32_t[2], float[4]> sums[2] = {};
#pragma unroll
    for (int kk = 0; kk < kSteps; ++kk) {
      const int t = 16 * kk + token;
      uint32_t v[4];
      ops::load_frag_t(v, values + t * kValueBytes + (((2 * jc + (lane >> 4)) ^ swizzle(t)) << 4));
      if constexpr (kF16) {
        ops::hmma_f16(sums[0], a[kk], v[0], v[1]);
        ops::hmma_f16(sums[1], a[kk], v[2], v[3]);
      } else {
        ops::hmma(sums[0], a[kk], v[0], v[1]);
        ops::hmma(sums[1], a[kk], v[2], v[3]);
      }
    }
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      float s[4];
      if constexpr (kF16) {
        const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&sums[e][0]));
        const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&sums[e][1]));
        s[0] = lo.x, s[1] = lo.y, s[2] = hi.x, s[3] = hi.y;
      } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i] = sums[e][i];
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) acc.out[2 * jc + e][i] += s[i] * undo[i >> 1];
    }
  }
}

//: A WARP'S COPIES: one warp issues a whole stage or a tile's values, a lane's place fixed (row `lane / 8 + 4 i`, chunk
//: `lane % 8`), the source a row's 128 contiguous bytes eight lanes at a time.
//: -- see docs/internals/intra/intra_kernel.md#copies
struct Copier {
  static constexpr int kRowsPerCopy = 4, kCopies = kTile / kRowsPerCopy;
  uint32_t dst;
  int j, k, lane, b, h, bhw, t_window;

  __device__ __forceinline__ Copier(int b_, int h_, int bhw_, int t_window_, int lane_)
      : j(lane_ >> 3), k(lane_ & 7), lane(lane_), b(b_), h(h_), bhw(bhw_), t_window(t_window_) {
    dst = static_cast<uint32_t>(j * kStageBytes + ((k ^ (j << 1)) << 4));
  }

  //: tile `tile`'s stage `s` into buffer `buf`, each lane arriving on the buffer's full mbarrier when its copies land;
  //: a stage no warp reads is not copied, and its arrivals are immediate. Row `j + 4 i`'s chunk is `k ^ 2 j ^ (i & 1)`.
  __device__ __forceinline__ void stage(const Smem& sm, const IntraParams& p, int buf, int tile,
                                        int s, bool issue) const {
    if (!issue) {
      ops::mbar_arrive(ops::smem_addr(&sm.full[buf]));
      return;
    }
    const Stage& g = p.stage[s];
    const int level = g.level;
    const int elem = form_write_f16(p.form[level]) ? 2 : 1;
    const int byte = g.digit0 * elem + k * kChunk;
    const bool live = byte < p.width[level] * elem;
    const RowView& v = p.write[level];
    const uint8_t* const row = row_at(v, b, t_window + tile * kTile + j, h) + (live ? byte : 0);
    const uint32_t base = ops::smem_addr(&sm.stage[buf][0][0]) + dst;
#pragma unroll
    for (int i = 0; i < kCopies; ++i)
      ops::stage_tile_if((base + i * kRowsPerCopy * kStageBytes) ^ ((i & 1) << 4),
                         row + rows_of(i * kRowsPerCopy, v.st), live);
    ops::mbar_arrive_when_landed(ops::smem_addr(&sm.full[buf]));
  }

  //: tile `tile`'s values and writer factors, or immediate arrivals when no warp's pair with it is live
  __device__ __forceinline__ void values(const Smem& sm, const IntraParams& p, int tile,
                                         bool issue) const {
    if (!issue) {
      ops::mbar_arrive(ops::smem_addr(&sm.values_full));
      return;
    }
    const uint8_t* const row = row_at(p.values, b, t_window + tile * kTile + j, h) + k * kChunk;
    const uint32_t base = ops::smem_addr(&sm.values[0][0]) + dst;
#pragma unroll
    for (int i = 0; i < kCopies; ++i)
      ops::stage_tile((base + i * kRowsPerCopy * kValueBytes) ^ ((i & 1) << 4),
                      row + rows_of(i * kRowsPerCopy, p.values.st));
    ops::stage_tile(ops::smem_addr(&sm.fm[2 * lane]),
                    p.fm + static_cast<int64_t>(bhw) * p.L + t_window + tile * kTile + 2 * lane);
    ops::mbar_arrive_when_landed(ops::smem_addr(&sm.values_full));
  }
};

//: THE BLOCK'S READER ROWS for stage `s`, once: the stage's digits of each reader row into the row's slot for it,
//: arriving on the stage's own mbarrier, so the first Gram waits for one part, not the block's rows.
__device__ __forceinline__ void load_reader_stage(Smem& sm, const IntraParams& p, int s, int b,
                                                  int h, int t_block, int tid) {
  const Stage& g = p.stage[s];
  const int level = g.level, form = p.form[level];
  const int elem = form_read_f16(form) ? 2 : 1;
  const int chunks = form_stage_digits(form) * elem / kChunk;
  const int rows_per = kThreads / chunks;
  const int r = tid / chunks, k = tid % chunks;
  const int byte = g.digit0 * elem + k * kChunk;
  const bool live = byte < p.width[level] * elem;
  const RowView& v = p.read[level];
  const uint8_t* const src = row_at(v, b, t_block + r, h) + (live ? byte : 0);
  uint8_t* const rows = reader_rows(sm);
#pragma unroll 1
  for (int row = r; row < kBlockRows; row += rows_per)
    ops::stage_tile_if(
        ops::smem_addr(rows + row * p.pitch + (((g.reader_chunk + k) ^ swizzle(row)) << 4)),
        src + rows_of(row - r, v.st), live);
  ops::mbar_arrive_when_landed(ops::smem_addr(&sm.reader_full[s]));
}

//: A TILE UNION's address in the facts pass's grouped table: a write head's `group + 1` slots, its query heads' read
//: sides then its write side.
__device__ __forceinline__ const uint32_t* union_at(const IntraParams& p, int bhw, int slot,
                                                    int group, int tile) {
  const int tiles = p.L / kTile;
  return p.unions + ((static_cast<int64_t>(bhw) * (p.group + 1) + slot) * p.groups + group) * tiles
         + tile;
}

//: THE GATE TABLE: per (window tile, row tile) the slabs whose reader and writer tile unions meet, a lane a slab,
//: cleared whole when a level has none (the pair's product is exactly zero) or the pair lies above the diagonal; a
//: dense level answers all-ones unread. One warp. -- see docs/internals/intra/intra_kernel.md#slab-gate
template <int D>
__device__ __forceinline__ void gate_table(Smem& sm, const IntraParams& p, int bhw, int slot,
                                           int window, int block, int lane) {
  int level = 0;
#pragma unroll
  for (int l = 1; l < D; ++l)
    if (lane >= p.slab0[l]) level = l;
  const int group = p.group0[level] + lane - p.slab0[level];
  const bool slab = lane < p.slabs;
  const uint32_t all = slab ? ~0u : 0u;
  const int tile0 = window * kWindowTiles;

  uint32_t rows[kRowTiles], columns[kWindowTiles];
#pragma unroll
  for (int rt = 0; rt < kRowTiles; ++rt) rows[rt] = all;
#pragma unroll
  for (int t = 0; t < kWindowTiles; ++t) columns[t] = all;
  if (slab && !facts::level_is_dense(p.dense[facts::kRead], level)) {
    const uint32_t* const r = union_at(p, bhw, slot, group, tile0 + block * kRowTiles);
#pragma unroll
    for (int rt = 0; rt < kRowTiles; ++rt) rows[rt] = r[rt];
  }
  if (slab && !facts::level_is_dense(p.dense[facts::kWrite], level)) {
    const uint32_t* const w = union_at(p, bhw, p.group, group, tile0);
#pragma unroll
    for (int q = 0; q < kWindowQuads; ++q) {
      const uint4 x = ops::load_vec16(w + 4 * q);
      columns[4 * q] = x.x, columns[4 * q + 1] = x.y, columns[4 * q + 2] = x.z,
                  columns[4 * q + 3] = x.w;
    }
  }
  uint32_t level_slabs[D];
#pragma unroll
  for (int l = 0; l < D; ++l) level_slabs[l] = p.level_slabs[l];
#pragma unroll
  for (int t = 0; t < kWindowTiles; ++t)
#pragma unroll
    for (int rt = 0; rt < kRowTiles; ++rt) {
      uint32_t mask = __ballot_sync(0xffffffffu, (rows[rt] & columns[t]) != 0u);
#pragma unroll
      for (int l = 0; l < D; ++l)
        if ((mask & level_slabs[l]) == 0u) mask = 0u;
      if (lane == 0) sm.gate[t][rt] = t <= block * kRowTiles + rt ? mask : 0u;
    }
}

//: a stage's slabs of a gate word, from the stage's first
__device__ __forceinline__ uint32_t stage_slabs(const Stage& g, uint32_t gate) {
  return (gate >> g.slab0) & g.mask;
}

//: any row tile's gate word for writer tile `tile`
__device__ __forceinline__ uint32_t tile_gate(const Smem& sm, int tile) {
  uint32_t any = 0u;
#pragma unroll
  for (int rt = 0; rt < kRowTiles; ++rt) any |= sm.gate[tile][rt];
  return any;
}

//: whether tile `tile`'s stage `s` is copied: when a row tile reads one of its slabs (the piece's first stages, copied
//: before the gate is known, always are)
__device__ __forceinline__ bool stage_issued(const Smem& sm, const IntraParams& p, int tile,
                                             int s) {
  return stage_slabs(p.stage[s], tile_gate(sm, tile)) != 0u;
}

//: THE DEPOSIT: the warp's 16 rows of partial sums -- its writer columns' -- reduced into the output planes through a
//: staging tile of its own in the dead reader rows, a row's channels two lines of consecutive reductions; its rows'
//: partial sums into `den`. -- see docs/internals/intra/intra_kernel.md#deposit
__device__ __forceinline__ void deposit(Smem& sm, const IntraParams& p, int bh, int t_row, int warp,
                                        int lane, const float (&out)[8][4]) {
  constexpr int kPitch = kValueWidth + 8;
  static_assert(kWarpRows * kPitch * 4 == kStagingBytes, "the staging tile is the card table's");
  //: the reader rows are dead once every warp's last Gram is done
  __syncthreads();
  float* const staging = reinterpret_cast<float*>(reader_rows(sm) + warp * kStagingBytes);
  const int g = lane >> 2, tq = lane & 3;
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int r = 0; r < 2; ++r)
      ops::store_shared_f32x2(ops::smem_addr(&staging[(g + 8 * r) * kPitch + 8 * n + 2 * tq]),
                              out[n][2 * r], out[n][2 * r + 1]);
  __syncwarp();
  float* const num = p.num + (static_cast<int64_t>(bh) * p.L + t_row) * kValueWidth;
#pragma unroll
  for (int r = 0; r < kWarpRows; ++r) {
    ops::fan_in_add(num + r * kValueWidth + lane, staging[r * kPitch + lane]);
    ops::fan_in_add(num + r * kValueWidth + 32 + lane, staging[r * kPitch + 32 + lane]);
  }
  if (lane < kWarpRows)
    ops::fan_in_add(p.den + static_cast<int64_t>(bh) * p.L + t_row + lane, sm.den[warp][lane]);
}

//: THE RING'S PLACE: the piece's stage index, its buffer and that buffer's use, stepped once a stage.
struct Ring {
  int it = 0, buf = 0, use = 0;

  __device__ __forceinline__ void step() {
    ++it;
    if (++buf == kStageBuffers) buf = 0, ++use;
  }
};

//: the piece's stage `ahead` stages past (tile, s): its tile and stage, uniform
__device__ __forceinline__ int2 stage_after(const IntraParams& p, int tile, int s, int ahead) {
  int ns = s + ahead, ntile = tile;
#pragma unroll 1
  while (ns >= p.stages) ns -= p.stages, ++ntile;
  return make_int2(ntile, ns);
}

//: ONE STAGE OF A TILE: awaited by every warp and, for a live pair (`kLive`), read where the warp's slabs meet it under
//: the level's form `F`; released, and the last warp to release it refills its buffer with the stage `kStageBuffers`
//: on, which leaves the refill two stages to land. -- see docs/internals/intra/intra_kernel.md#pipeline
template <int F, bool kF16, bool kLive>
__device__ __forceinline__ void tile_stage(Smem& sm, const IntraParams& p, const Copier& copy,
                                           const Lanes& ln, int s, int tile, int t0, int t1,
                                           uint32_t mine, int lane, Ring& ring, Acc<kF16>& acc,
                                           Clock& clock) {
  //: lane 0 releases and counts for its warp
  const bool leader = lane == 0;
  //: EVERY warp waits for every stage, read or not, so no warp's release counts toward a buffer's next use before that
  //: use has landed. A stage no warp reads was not copied, and its wait is immediate.
  if (tile == t0) ops::mbar_wait(ops::smem_addr(&sm.reader_full[s]), 0);
  ops::mbar_wait(ops::smem_addr(&sm.full[ring.buf]), ring.use & 1);
  clock.lap(kPhaseWait);
  if constexpr (kLive) {
    const Stage& g = p.stage[s];
    gram_stage<F, kF16>(ln, g.reader_chunk, ops::smem_addr(&sm.stage[ring.buf][0][0]),
                        stage_slabs(g, mine), acc);
    clock.lap(kPhaseGram);
  }
  __syncwarp();
  int last = 0;
  if (leader) {
    __threadfence_block();
    last = atomicAdd(&sm.released[ring.buf], 1) == (ring.use + 1) * kWarps - 1;
  }
  if (__shfl_sync(0xffffffffu, last, 0)) {
    const int2 next = stage_after(p, tile, s, kStageBuffers);
    if (next.x < t1) {
      __threadfence_block();
      copy.stage(sm, p, ring.buf, next.x, next.y, stage_issued(sm, p, next.x, next.y));
    }
  }
  ring.step();
}

//: A TILE'S STAGES: for a live pair each level's stages under its form -- one uniform switch a level, its Gram opened
//: before them and folded after -- and for a dead one every stage awaited and released.
template <int D, bool kF16>
__device__ __forceinline__ void tile_stages(Smem& sm, const IntraParams& p, const Copier& copy,
                                            const Lanes& ln, int tile, int t0, int t1, int row_tile,
                                            int lane, Ring& ring, Acc<kF16>& acc, Clock& clock) {
  const uint32_t mine = sm.gate[tile][row_tile % kRowTiles];
  if (mine == 0u) {
#pragma unroll 1
    for (int s = 0; s < p.stages; ++s)
      tile_stage<kFormU8U8, kF16, false>(sm, p, copy, ln, s, tile, t0, t1, mine, lane, ring, acc,
                                         clock);
    return;
  }
  rola::static_for<D>([&](auto Level) {
    constexpr int kL = decltype(Level)::value;
    zero_gram(acc);
    rola::uniform_switch<kForms>(p.form[kL], [&](auto F) {
#pragma unroll 1
      for (int s = p.level_stage0[kL]; s < p.level_stage0[kL] + p.level_stages[kL]; ++s)
        tile_stage<decltype(F)::value, kF16, true>(sm, p, copy, ln, s, tile, t0, t1, mine, lane,
                                                   ring, acc, clock);
    });
    fold_level(acc, kL == 0, p.lift);
  });
}

//: A TILE'S VALUES: awaited after its last stage; the epilogue and the value product for a live pair, the values
//: released and the next tile's copied by the last warp to release them.
template <bool kF16>
__device__ __forceinline__ void tile_values(Smem& sm, const IntraParams& p, const Copier& copy,
                                            int tile, int t0, int t1, int row_tile, int warp,
                                            int group, int col0, int lane, Acc<kF16>& acc,
                                            Clock& clock) {
  const bool leader = lane == 0;
  const int r = tile - t0;
  ops::mbar_wait(ops::smem_addr(&sm.values_full), r & 1);
  if (sm.gate[tile][row_tile % kRowTiles] != 0u) {
    epilogue(sm, p.unit, tile == row_tile, warp, group, col0, lane, acc);
    clock.lap(kPhaseEpilogue);
    value_product(sm, p.unit, col0, lane, acc);
    clock.lap(kPhaseValue);
  }
  //: the last warp to release the tile's values copies the next tile's. Every warp awaits a tile's values before it
  //: releases them, and tile `t + 2`'s land only after tile `t + 1`'s last release, so tiles `t` and `t + 2` share a
  //: counter by parity and never interleave on it
  __syncwarp();
  int last = 0;
  if (leader) {
    __threadfence_block();
    last = atomicAdd(&sm.values_released[r & 1], 1) == (r / 2 + 1) * kWarps - 1;
  }
  if (__shfl_sync(0xffffffffu, last, 0) && tile + 1 < t1) {
    __threadfence_block();
    copy.values(sm, p, tile + 1, tile_gate(sm, tile + 1) != 0u);
  }
}

//: THE GRID: a CTA a (window, piece) along x -- a window's longest piece first -- and a query head along y, an arm a
//: depth `D` and an accumulation. The block's readers land once, a part a stage, tile 0's first stage beside the first
//: part; the piece's writer tiles stream through the stage buffers; its partial sums reduce into the output planes.
//: -- see docs/internals/intra/intra_kernel.md#the-grid
template <int D, bool kF16>
__global__ __launch_bounds__(kThreads, 1) void intra_kernel(const __grid_constant__ IntraParams p) {
  extern __shared__ __align__(128) unsigned char raw[];
  Smem& sm = *reinterpret_cast<Smem*>(raw);
  if (blockDim.x != kThreads) __trap();
  const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
  const int bh = static_cast<int>(blockIdx.y);
  const int window = static_cast<int>(blockIdx.x) / Pieces::count();
  const Piece piece = kPieceTable.table[static_cast<int>(blockIdx.x) % Pieces::count()];
  const int block = piece.block;
  const int b = bh / p.H, h = bh - b * p.H;
  const int hw = h / p.group, slot = h - hw * p.group;
  const int bhw = b * (p.H / p.group) + hw;
  const int t_window = window * kWindow;
  const int t_block = t_window + block * kBlockRows;
  //: the warp's reader group and its writer columns in a tile
  const int group = warp / kColWarps, col0 = (warp % kColWarps) * kWarpCols;
  const int row_tile = block * kRowTiles + group / (kRowWarps / kRowTiles);
  Clock clock(&sm.clock[warp][0], lane, p.ledger != nullptr);
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < kStageBuffers; ++i) {
      ops::mbar_init(ops::smem_addr(&sm.full[i]), 32);
      sm.released[i] = 0;
    }
    sm.values_released[0] = sm.values_released[1] = 0;
#pragma unroll 1
    for (int s = 0; s < p.stages; ++s) ops::mbar_init(ops::smem_addr(&sm.reader_full[s]), kThreads);
    ops::mbar_init(ops::smem_addr(&sm.values_full), 32);
  }
  __syncthreads();

  Acc<kF16> acc;
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc.out[n][i] = 0.0f;
  if (lane < kWarpRows) sm.den[warp][lane] = 0.0f;

  const Copier copy(b, hw, bhw, t_window, lane);
  const Lanes ln(sm, p, group, col0, lane);
  //: the reader parts in their stages' order, the piece's first stages beside the first, copied before the gate exists
#pragma unroll 1
  for (int s = 0; s < p.stages; ++s) {
    load_reader_stage(sm, p, s, b, h, t_block, tid);
    if (s == 0 && warp == 0)
#pragma unroll 1
      for (int i = 0; i < kStageBuffers; ++i) {
        const int2 at = stage_after(p, piece.t0, 0, i);
        if (at.x < piece.t1) copy.stage(sm, p, i, at.x, at.y, true);
      }
  }
  if (warp == 0) copy.values(sm, p, piece.t0, true);
  clock.lap(kPhaseReader);
  if (warp == 0) gate_table<D>(sm, p, bhw, slot, window, block, lane);
  __syncthreads();
  clock.lap(kPhaseGate);

  Ring ring;
#pragma unroll 1
  for (int tile = piece.t0; tile < piece.t1; ++tile) {
    tile_stages<D>(sm, p, copy, ln, tile, piece.t0, piece.t1, row_tile, lane, ring, acc, clock);
    tile_values(sm, p, copy, tile, piece.t0, piece.t1, row_tile, warp, group, col0, lane, acc,
                clock);
  }
  deposit(sm, p, bh, t_block + group * kWarpRows, warp, lane, acc.out);
  clock.lap(kPhaseDeposit);
  clock.publish(p.ledger, 0, kWarps, warp);
}

}  // namespace rola::intra::detail
