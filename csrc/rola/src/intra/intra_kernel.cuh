// csrc/rola/src/intra/intra_kernel.cuh -- THE INTRA KERNEL: a CTA a (batch-head, window, block of 128 readers), the
// block's reader rows resident in shared memory, the writer tiles streamed through three stage buffers under full and
// empty mbarriers under the slab gate the facts pass's words give; a tile's Gram over each level's digits, the epilogue
// into the value product's weights, the value product, and the block's rows reduced into the output planes at the end.
// see docs/internals/intra/intra_kernel.md
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <utility>

#include "common/ops.cuh"
#include "common/phase_clock.cuh"
#include "common/structure_switch.cuh"
#include "facts/liveness_contract.cuh"
#include "intra/intra.cuh"

namespace rola::intra::detail {

namespace ops = denseref::ops;
using Clock = PhaseClock<kIntraPhases>;

//: AN ARM: `D` levels of `B` u8 digits a side. -- see docs/internals/intra/intra_kernel.md#arms
template <int D, int B>
struct Arm {
  static constexpr int kRowBytes = D * B;
  static constexpr int kRowChunks = kRowBytes / kChunk;
  static constexpr int kLevelChunks = B / kChunk;
  static constexpr int kStagesPerLevel = B / kStageDigits;
  static constexpr int kStages = D * kStagesPerLevel;
  static constexpr int kStageChunks = kStageDigits / kChunk;
  static constexpr int kStageCopies = kTile * kStageChunks / kThreads;
  static constexpr int kColumns = (2 + D) * kTile;
  static constexpr int kSlabs = D * B / kSlabDigits;
  static constexpr int kSlabsPerLevel = B / kSlabDigits;
  static constexpr int kSlabsPerStage = kStageDigits / kSlabDigits;
  //: A digit is its unit value times 255, so a level's integer Gram is 255^2 its unit Gram.
  static constexpr float kUnit = 1.0f
                                 / (D == 1   ? 65025.0f
                                    : D == 2 ? 65025.0f * 65025.0f
                                    : D == 3 ? 65025.0f * 65025.0f * 65025.0f
                                             : 65025.0f * 65025.0f * 65025.0f * 65025.0f);
  static_assert(B % kStageDigits == 0, "a stage lies inside one level");
  static_assert(kStages >= kStageBuffers, "a tile's values land while its first stages are read");
  static_assert(kColumns <= kThreads, "a thread copies at most one writer column entry a tile");
  static_assert(kSlabs <= 32, "a tile pair's slabs are one word");
};

//: THE SHARED TENANTS: the block's reader rows, three writer stages, the tile's value rows and writer columns (the
//: gain, the value scale, each write level's scale), the slab gate's table, and the pipeline's mbarriers.
//: -- see docs/internals/intra/intra_kernel.md#smem
template <int D, int B>
struct alignas(128) Smem {
  uint8_t reader[kBlockRows][Arm<D, B>::kRowBytes];
  uint8_t stage[kStageBuffers][kTile][kStageDigits];
  uint8_t v_hat[kTile][kValueWidth];
  uint8_t v_lo[kTile][kValueWidth];
  float column[2 + D][kTile];
  float den[kWarps][kWarpRows];
  uint32_t gate[kWindowTiles][kRowTiles];
  uint64_t full[kStageBuffers], empty[kStageBuffers];
  uint64_t reader_full[Arm<D, B>::kStages], values_full, values_empty;
  long long clock[kWarps][Clock::kRow];
};

//: A ROW'S 16-BYTE CHUNK, placed so eight consecutive rows at one chunk lie in eight bank groups; a 64-byte value row's
//: four chunks by row pairs. -- see docs/internals/intra/intra_kernel.md#swizzle
__device__ __forceinline__ int swizzle8(int row, int chunk) { return chunk ^ (row & 7); }

__device__ __forceinline__ int swizzle4(int row, int chunk) { return chunk ^ ((row >> 1) & 3); }

template <typename T>
__device__ __forceinline__ const T* at(const RowView& v, int b, int t, int h) {
  return static_cast<const T*>(v.base) + b * v.sb + t * v.st + h * v.sh;
}

//: `rows` token strides of `st` elements: one unsigned 32 x 32 -> 64 multiply (the seam bounds a stride to 31 bits)
__device__ __forceinline__ uint64_t rows_of(int rows, int st) {
  return static_cast<uint64_t>(static_cast<uint32_t>(rows)) * static_cast<uint32_t>(st);
}

//: A THREAD'S COPIES: its place in a copy is fixed -- a stage's row `tid / 8` and chunk `tid % 8`, a value row's
//: `tid / 4` and `tid % 4`, one column entry -- so each base is computed once, 64-bit, and a copy adds its tile's rows by
//: one wide multiply-add of a 32-bit token stride. -- see docs/internals/intra/intra_kernel.md#copies
template <int D, int B>
struct Copier {
  using A = Arm<D, B>;
  //: the rows a copy instruction's threads cover, and the bytes between two of them in a stage buffer
  static constexpr int kRowsPerCopy = kThreads / A::kStageChunks;
  const uint8_t* stage_row[D];
  const uint8_t* value_row;
  const uint8_t* lo_row;
  const float* column;
  int column_st;
  uint32_t stage_dst, value_dst;

  __device__ __forceinline__ Copier(const IntraParams& p, int b, int h, int t_window, int tid) {
    const int j = tid / A::kStageChunks, k = tid % A::kStageChunks;
#pragma unroll
    for (int l = 0; l < D; ++l)
      stage_row[l] = at<uint8_t>(p.write[l], b, t_window + j, h) + k * kChunk;
    stage_dst = static_cast<uint32_t>(j * kStageDigits + swizzle8(j, k) * kChunk);
    const int vj = tid >> 2, vk = tid & 3;
    value_row = at<uint8_t>(p.v_hat, b, t_window + vj, h) + vk * kChunk;
    lo_row =
        p.v_lo.base != nullptr ? at<uint8_t>(p.v_lo, b, t_window + vj, h) + vk * kChunk : nullptr;
    value_dst = static_cast<uint32_t>(vj * kValueWidth + swizzle4(vj, vk) * kChunk);
    column = nullptr;
    column_st = 0;
    if (tid < A::kColumns) {
      const int c = tid / kTile, t = tid % kTile;
      const ColumnView& v = c == 0 ? p.g_write : c == 1 ? p.v_scale : p.scale_write[c - 2];
      column = v.base + b * v.sb + (t_window + t) * v.st + h * v.sh;
      column_st = static_cast<int>(v.st);
    }
  }

  //: tile `tile`'s stage `S`, arriving on its buffer's full mbarrier when it lands; a stage no warp reads is not
  //: copied, and its arrival is immediate
  template <int S>
  __device__ __forceinline__ void stage(const Smem<D, B>& sm, const IntraParams& p, int buf,
                                        int tile, bool issue) const {
    constexpr int kLevel = S / A::kStagesPerLevel;
    constexpr int kDigit0 = (S % A::kStagesPerLevel) * kStageDigits;
    if (!issue) {
      ops::mbar_arrive(ops::smem_addr(&sm.full[buf]));
      return;
    }
    const int st = static_cast<int>(p.write[kLevel].st);
    const uint32_t dst = ops::smem_addr(&sm.stage[buf][0][0]) + stage_dst;
    constexpr int kCopies = A::kStageCopies;
#pragma unroll
    for (int i = 0; i < kCopies; ++i)
      ops::stage_tile(dst + i * kRowsPerCopy * kStageDigits,
                      stage_row[kLevel] + rows_of(tile * kTile + i * kRowsPerCopy, st) + kDigit0);
    ops::mbar_arrive_when_landed(ops::smem_addr(&sm.full[buf]));
  }

  //: tile `tile`'s value digits and writer columns, or an immediate arrival when no warp's pair with it is live
  __device__ __forceinline__ void values(const Smem<D, B>& sm, const IntraParams& p, int tile,
                                         bool issue) const {
    if (!issue) {
      ops::mbar_arrive(ops::smem_addr(&sm.values_full));
      return;
    }
    const uint64_t rows = rows_of(tile * kTile, static_cast<int>(p.v_hat.st));
    ops::stage_tile(ops::smem_addr(&sm.v_hat[0][0]) + value_dst, value_row + rows);
    if (lo_row != nullptr)
      ops::stage_tile(ops::smem_addr(&sm.v_lo[0][0]) + value_dst, lo_row + rows);
    if (column != nullptr)
      ops::stage_run<4>(ops::smem_addr(&sm.column[0][0]) + static_cast<uint32_t>(threadIdx.x) * 4,
                        column + rows_of(tile * kTile, column_st));
    ops::mbar_arrive_when_landed(ops::smem_addr(&sm.values_full));
  }
};

//: THE BLOCK'S READER ROWS, once, a stage's digits at a time: part `S` goes out beside tile 0's stage `S` and arrives
//: on its own mbarrier, so the first Gram waits for one part, not the block's rows; a chunk a copy, consecutive threads
//: along a token's run.
template <int D, int B, int S>
__device__ __forceinline__ void load_reader_part(Smem<D, B>& sm, const IntraParams& p, int b, int h,
                                                 int t0, int tid) {
  using A = Arm<D, B>;
  constexpr int kLevel = S / A::kStagesPerLevel;
  constexpr int kDigit0 = (S % A::kStagesPerLevel) * kStageDigits;
  constexpr int kCopies = kBlockRows * A::kStageChunks / kThreads;
  constexpr int kRowsPerCopy = kThreads / A::kStageChunks;
  const int r = tid / A::kStageChunks, k = tid % A::kStageChunks;
  const int st = static_cast<int>(p.read[kLevel].st);
  const uint8_t* const row = at<uint8_t>(p.read[kLevel], b, t0 + r, h) + kDigit0 + k * kChunk;
  const uint32_t dst = ops::smem_addr(&sm.reader[r][swizzle8(r, S * A::kStageChunks + k) * kChunk]);
#pragma unroll
  for (int i = 0; i < kCopies; ++i)
    ops::stage_tile(dst + i * kRowsPerCopy * A::kRowBytes, row + rows_of(i * kRowsPerCopy, st));
  ops::mbar_arrive_when_landed(ops::smem_addr(&sm.reader_full[S]));
}

//: THE GATE TABLE: per (window tile, row tile) the slabs whose reader and writer tile unions meet (the facts pass's), a
//: lane a slab, cleared whole when a level has none (the pair's product is exactly zero) or the pair lies above the
//: diagonal; a dense level answers all-ones unread. One warp. -- see docs/internals/intra/intra_kernel.md#slab-gate
template <int D, int B>
__device__ __forceinline__ void gate_table(Smem<D, B>& sm, const IntraParams& p, int bh, int window,
                                           int block, int lane) {
  using A = Arm<D, B>;
  constexpr uint32_t kLevelMask = (1u << A::kSlabsPerLevel) - 1u;
  const int tiles = p.L / facts::kTileTokens;
  static_assert(B % facts::kGroupDigits == 0,
                "a level is whole groups, so slab `s` is the side's group `s`");
  const int level = min(lane / A::kSlabsPerLevel, D - 1);
  const int group = lane;
  const bool slab = lane < A::kSlabs;
  const uint32_t all = slab ? ~0u : 0u;
  const int tile0 = window * kWindowTiles;
  uint2 reader = {all, all};
  uint4 writer[2] = {{all, all, all, all}, {all, all, all, all}};
  if (slab && !facts::level_is_dense(p.dense[facts::kRead], level))
    reader = *reinterpret_cast<const uint2*>(p.unions
                                             + facts::union_address(bh, facts::kRead, group,
                                                                    tile0 + block * kRowTiles,
                                                                    A::kSlabs, tiles));
  if (slab && !facts::level_is_dense(p.dense[facts::kWrite], level)) {
    const uint32_t* const w =
        p.unions + facts::union_address(bh, facts::kWrite, group, tile0, A::kSlabs, tiles);
    writer[0] = ops::load_vec16(w);
    writer[1] = ops::load_vec16(w + 4);
  }
  const uint32_t rows[kRowTiles] = {reader.x, reader.y};
  const uint32_t columns[kWindowTiles] = {writer[0].x, writer[0].y, writer[0].z, writer[0].w,
                                          writer[1].x, writer[1].y, writer[1].z, writer[1].w};
#pragma unroll
  for (int t = 0; t < kWindowTiles; ++t)
#pragma unroll
    for (int rt = 0; rt < kRowTiles; ++rt) {
      uint32_t mask = __ballot_sync(0xffffffffu, (rows[rt] & columns[t]) != 0u);
#pragma unroll
      for (int l = 0; l < D; ++l)
        if (((mask >> (l * A::kSlabsPerLevel)) & kLevelMask) == 0u) mask = 0u;
      if (lane == 0) sm.gate[t][rt] = t <= block * kRowTiles + rt ? mask : 0u;
    }
}

//: stage `S`'s slabs of a gate word
template <int D, int B, int S>
__device__ __forceinline__ uint32_t stage_slabs(uint32_t gate) {
  using A = Arm<D, B>;
  return (gate >> (S * A::kSlabsPerStage)) & ((1u << A::kSlabsPerStage) - 1u);
}

//: whether tile `tile`'s stage `S` is copied: tile 0's always (its copies leave before the gate is known), a later
//: tile's when either row tile reads one of its slabs
template <int D, int B, int S>
__device__ __forceinline__ bool stage_issued(const Smem<D, B>& sm, int tile) {
  return tile == 0 || stage_slabs<D, B, S>(sm.gate[tile][0] | sm.gate[tile][1]) != 0u;
}

//: THE WARP'S STATE ACROSS A BLOCK: the level's s32 Gram, the running level product and the output; the row sums
//: live in shared memory, a warp's 16 rows, so no register holds them across the loop.
struct Acc {
  int32_t gram[8][4];
  float product[8][4];
  float out[8][4];
};

//: ONE K-STEP OF A STAGE: the warp's 16 reader rows against the stage's 64 writer tokens over one slab's 32 digits,
//: eight IMMAs into the level's s32 accumulator (fresh at a level's first step); both operands by plain ldmatrix off
//: token-major rows. -- see docs/internals/intra/intra_kernel.md#gram
template <int D, int B, int S, int K>
__device__ __forceinline__ void gram_step(const Smem<D, B>& sm, int buf, int warp, int lane,
                                          Acc& acc) {
  using A = Arm<D, B>;
  constexpr int kChunk0 =
      (S / A::kStagesPerLevel) * A::kLevelChunks + (S % A::kStagesPerLevel) * A::kStageChunks;
  constexpr bool kFresh = S % A::kStagesPerLevel == 0 && K == 0;
  const int arow = warp * kWarpRows + (lane & 15);
  const int brow = (lane & 7) + ((lane >> 4) << 3);
  uint32_t a[4];
  ops::load_frag(
      a, ops::smem_addr(&sm.reader[arow][swizzle8(arow, kChunk0 + 2 * K + (lane >> 4)) * kChunk]));
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int token = 16 * j + brow;
    uint32_t b[4];
    ops::load_frag(
        b,
        ops::smem_addr(&sm.stage[buf][token][swizzle8(token, 2 * K + ((lane >> 3) & 1)) * kChunk]));
    if constexpr (kFresh) {
      ops::imma_u8_k32_fresh(acc.gram[2 * j], a, b[0], b[1]);
      ops::imma_u8_k32_fresh(acc.gram[2 * j + 1], a, b[2], b[3]);
    } else {
      ops::imma_u8_k32(acc.gram[2 * j], a, b[0], b[1]);
      ops::imma_u8_k32(acc.gram[2 * j + 1], a, b[2], b[3]);
    }
  }
}

//: THE GRAM OF ONE STAGE, its four k-steps under the warp's slab bits: one straight-line block when every slab is live,
//: a branch a step otherwise; a level's first step, gated off, zeroes the accumulator in its place.
template <int D, int B, int S>
__device__ __forceinline__ void gram_stage(const Smem<D, B>& sm, int buf, int warp, int lane,
                                           uint32_t slabs, Acc& acc) {
  using A = Arm<D, B>;
  if (slabs == (1u << A::kSlabsPerStage) - 1u) {
    rola::static_for<A::kSlabsPerStage>(
        [&](auto K) { gram_step<D, B, S, decltype(K)::value>(sm, buf, warp, lane, acc); });
    return;
  }
  rola::static_for<A::kSlabsPerStage>([&](auto K) {
    constexpr int kK = decltype(K)::value;
    if ((slabs >> kK) & 1u) {
      gram_step<D, B, S, kK>(sm, buf, warp, lane, acc);
    } else if constexpr (S % A::kStagesPerLevel == 0 && kK == 0) {
#pragma unroll
      for (int n = 0; n < 8; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc.gram[n][i] = 0;
    }
  });
}

//: A LEVEL'S GRAM INTO THE RUNNING PRODUCT: exact s32 to fp32 (a level's sum is under 2^24), multiplied level by level.
__device__ __forceinline__ void fold_level(bool first, Acc& acc) {
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float g = static_cast<float>(acc.gram[n][i]);
      acc.product[n][i] = first ? g : acc.product[n][i] * g;
    }
}

//: THE EPILOGUE: the level product times the writer factor (a lane's two columns' factors, shuffled to the lanes that
//: hold them), zero above the diagonal on a diagonal tile, the row sums into `den`, and the value scale folded in: the
//: value product's `P`. -- see docs/internals/intra/intra_kernel.md#epilogue
template <int D, int B>
__device__ __forceinline__ void epilogue(Smem<D, B>& sm, bool diagonal, int warp, int lane,
                                         Acc& acc) {
  using A = Arm<D, B>;
  float f[2], s[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    f[j] = sm.column[0][2 * lane + j] * A::kUnit;
    s[j] = sm.column[1][2 * lane + j];
#pragma unroll
    for (int l = 0; l < D; ++l) f[j] *= sm.column[2 + l][2 * lane + j];
  }
  const int g = lane >> 2, tq = lane & 3;
  const int row0 = (warp % (kWarps / kRowTiles)) * kWarpRows + g;
  float den[2] = {0.0f, 0.0f};
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const float fa = __shfl_sync(0xffffffffu, f[0], 4 * n + tq);
    const float fb = __shfl_sync(0xffffffffu, f[1], 4 * n + tq);
    const float sa = __shfl_sync(0xffffffffu, s[0], 4 * n + tq);
    const float sb = __shfl_sync(0xffffffffu, s[1], 4 * n + tq);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int col = 8 * n + 2 * tq + (i & 1), row = row0 + 8 * (i >> 1);
      const float p = diagonal && col > row ? 0.0f : acc.product[n][i] * ((i & 1) ? fb : fa);
      den[i >> 1] += p;
      acc.product[n][i] = p * ((i & 1) ? sb : sa);
    }
  }
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    den[r] += __shfl_xor_sync(0xffffffffu, den[r], 1);
    den[r] += __shfl_xor_sync(0xffffffffu, den[r], 2);
  }
  if (tq == 0) {
    sm.den[warp][g] += den[0];
    sm.den[warp][g + 8] += den[1];
  }
}

//: THE VALUE PRODUCT: `P` against its row maximum over the tile (the quad's two shuffles) as `kWeightDigits` u8 digits,
//: packed from the accumulator's own registers into the A fragment (the K order the V load matches); V by ldmatrix.trans
//: and one prmt a register; the digit products into s32 by their weight, one scaled add a channel group.
//: -- see docs/internals/intra/intra_kernel.md#value-product
template <int D, int B, int kWeightDigits, bool kValueLo>
__device__ __forceinline__ void value_product(const Smem<D, B>& sm, int lane, Acc& acc) {
  constexpr float kLevels = kWeightDigits == 2 ? 65535.0f : 255.0f;
  constexpr float kValueLevels = kValueLo ? 32767.0f : 127.0f;
  constexpr uint32_t kHiSel = kWeightDigits == 2 ? 0x0051u : 0x0040u, kLoSel = 0x0040u;
  //: the digit products by weight: hi x v_hat, then (hi x v_lo, lo x v_hat), then lo x v_lo. Below hi x v_hat's
  //: weight their sum is exact in s32 (at most 1.6e9 over a tile at two and two); at one digit either way all of it is
  constexpr bool kSplit = kWeightDigits == 2 && kValueLo;
  constexpr int kHigh = kWeightDigits == 2 || kValueLo ? 256 : 1;

  float m[2] = {0.0f, 0.0f};
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) m[i >> 1] = fmaxf(m[i >> 1], acc.product[n][i]);
  float inv[2], unit[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    m[r] = fmaxf(m[r], __shfl_xor_sync(0xffffffffu, m[r], 1));
    m[r] = fmaxf(m[r], __shfl_xor_sync(0xffffffffu, m[r], 2));
    inv[r] = m[r] > 0.0f ? kLevels / m[r] : 0.0f;
    unit[r] = m[r] / (kLevels * kValueLevels);
  }

  uint32_t hi[2][4], lo[2][4];
#pragma unroll
  for (int kk = 0; kk < 2; ++kk) {
    uint32_t q[4][4];
#pragma unroll
    for (int t = 0; t < 4; ++t)
#pragma unroll
      for (int i = 0; i < 4; ++i)
        q[t][i] = __float2uint_rn(acc.product[4 * kk + t][i] * inv[i >> 1]);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      hi[kk][2 * half] = ops::pack_byte4(q[2 * half][0], q[2 * half][1], q[2 * half + 1][0],
                                         q[2 * half + 1][1], kHiSel);
      hi[kk][2 * half + 1] = ops::pack_byte4(q[2 * half][2], q[2 * half][3], q[2 * half + 1][2],
                                             q[2 * half + 1][3], kHiSel);
      if constexpr (kWeightDigits == 2) {
        lo[kk][2 * half] = ops::pack_byte4(q[2 * half][0], q[2 * half][1], q[2 * half + 1][0],
                                           q[2 * half + 1][1], kLoSel);
        lo[kk][2 * half + 1] = ops::pack_byte4(q[2 * half][2], q[2 * half][3], q[2 * half + 1][2],
                                               q[2 * half + 1][3], kLoSel);
      }
    }
  }

  //: one channel group at a time, a real branch apart, so a group's loads are not hoisted over the others'
#pragma unroll 1
  for (int j = 0; j < kValueWidth / 16; ++j) {
    rola::uniform_switch<kValueWidth / 16>(j, [&](auto J) {
      constexpr int kJ = decltype(J)::value;
      int32_t hh[2][4] = {}, mid[2][4] = {}, ll[2][4] = {};
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const int token = 32 * kk + 8 * (lane >> 3) + (lane & 7);
        uint32_t v[4];
        ops::load_frag_t(v, ops::smem_addr(&sm.v_hat[token][swizzle4(token, kJ) * kChunk]));
        const uint32_t even[2] = {ops::prmt(v[0], v[1], 0x6420u), ops::prmt(v[2], v[3], 0x6420u)};
        const uint32_t odd[2] = {ops::prmt(v[0], v[1], 0x7531u), ops::prmt(v[2], v[3], 0x7531u)};
        ops::imma_u8s8_k32(hh[0], hi[kk], even[0], even[1]);
        ops::imma_u8s8_k32(hh[1], hi[kk], odd[0], odd[1]);
        if constexpr (kWeightDigits == 2) {
          ops::imma_u8s8_k32(mid[0], lo[kk], even[0], even[1]);
          ops::imma_u8s8_k32(mid[1], lo[kk], odd[0], odd[1]);
        }
        if constexpr (kValueLo) {
          ops::load_frag_t(v, ops::smem_addr(&sm.v_lo[token][swizzle4(token, kJ) * kChunk]));
          const uint32_t lo_even[2] = {ops::prmt(v[0], v[1], 0x6420u),
                                       ops::prmt(v[2], v[3], 0x6420u)};
          const uint32_t lo_odd[2] = {ops::prmt(v[0], v[1], 0x7531u),
                                      ops::prmt(v[2], v[3], 0x7531u)};
          ops::imma_u8_k32(mid[0], hi[kk], lo_even[0], lo_even[1]);
          ops::imma_u8_k32(mid[1], hi[kk], lo_odd[0], lo_odd[1]);
          if constexpr (kWeightDigits == 2) {
            ops::imma_u8_k32(ll[0], lo[kk], lo_even[0], lo_even[1]);
            ops::imma_u8_k32(ll[1], lo[kk], lo_odd[0], lo_odd[1]);
          }
        }
      }
#pragma unroll
      for (int e = 0; e < 2; ++e)
#pragma unroll
        for (int i = 0; i < 4; ++i)
          acc.out[2 * kJ + e][i] += (kSplit ? static_cast<float>(hh[e][i]) * 65536.0f
                                                  + static_cast<float>(mid[e][i] * 256 + ll[e][i])
                                            : static_cast<float>(hh[e][i] * kHigh + mid[e][i]))
                                    * unit[i >> 1];
    });
  }
}

//: THE DEPOSIT: the warp's 16 rows reduced into the output planes through a staging tile, a row's channels two lines
//: of consecutive reductions; a row's sum into `den` from its quad. -- see docs/internals/intra/intra_kernel.md#deposit
template <int D, int B>
__device__ void deposit(Smem<D, B>& sm, const IntraParams& p, int bh, int t_row, int warp, int lane,
                        Acc& acc) {
  constexpr int kPitch = kValueWidth + 4;
  static_assert(kWarpRows * kPitch * 4 <= kWarpRows * Arm<D, B>::kRowBytes,
                "a warp stages in its own reader rows");
  //: the warp's own reader rows, which no other warp reads: its last Gram is done, and no CTA barrier is needed
  __syncwarp();
  float* const staging = reinterpret_cast<float*>(&sm.reader[warp * kWarpRows][0]);
  const int g = lane >> 2, tq = lane & 3;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int channel = 16 * (n >> 1) + 4 * tq + (n & 1);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      staging[(g + 8 * (i >> 1)) * kPitch + channel + 2 * (i & 1)] = acc.out[n][i];
  }
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

//: ONE STAGE OF A TILE: the stage two ahead issued once its buffer is released, the tile's values at its first stage
//: once the last tile's are released, each only if the gate reads it; this stage awaited and read where the warp's
//: slabs meet it, released; at the tile's last stage its values awaited, the epilogue and the value product for a live
//: pair, the values released. -- see docs/internals/intra/intra_kernel.md#pipeline
template <int D, int B, int S>
__device__ __forceinline__ void tile_stage(Smem<D, B>& sm, const IntraParams& p,
                                           const Copier<D, B>& copy, int tile, int tiles,
                                           int row_tile, int warp, int lane, bool& values_out,
                                           Acc& acc, Clock& clock) {
  using A = Arm<D, B>;
  constexpr int kNextS = (S + kStageBuffers - 1) % A::kStages;
  constexpr int kNextTile = (S + kStageBuffers - 1) / A::kStages;
  const int it = tile * A::kStages + S;
  const int next = it + kStageBuffers - 1;
  if (tile + kNextTile < tiles) {
    const int nbuf = next % kStageBuffers, use = next / kStageBuffers;
    if (use > 0) ops::mbar_wait(ops::smem_addr(&sm.empty[nbuf]), (use - 1) & 1);
    copy.template stage<kNextS>(sm, p, nbuf, tile + kNextTile,
                                stage_issued<D, B, kNextS>(sm, tile + kNextTile));
  }
  //: the tile's values leave once every warp has released the last tile's: tested at the tile's first stages, awaited
  //: two stages before they are read
  if constexpr (S <= A::kStages - 2) {
    if (!values_out) {
      const uint32_t released = ops::smem_addr(&sm.values_empty);
      if constexpr (S == A::kStages - 2) ops::mbar_wait(released, (tile - 1) & 1);
      if (S == A::kStages - 2 || ops::mbar_test(released, (tile - 1) & 1)) {
        copy.values(sm, p, tile, (sm.gate[tile][0] | sm.gate[tile][1]) != 0u);
        values_out = true;
      }
    }
  }
  const int buf = it % kStageBuffers;
  //: the warp's slabs of this tile pair: none when the pair is dead or above the diagonal
  const uint32_t mine = sm.gate[tile][row_tile % kRowTiles];
  const uint32_t slabs = stage_slabs<D, B, S>(mine);
  const bool first = tile == 0;
  const bool live = mine != 0u;
  //: a copy that leaves is awaited by a warp that reads it -- tile 0's by every warp -- so none is in flight at release
  if (first || slabs != 0u) {
    if (first) ops::mbar_wait(ops::smem_addr(&sm.reader_full[S]), 0);
    ops::mbar_wait(ops::smem_addr(&sm.full[buf]), (it / kStageBuffers) & 1);
    clock.lap(kPhaseWait);
  }
  if (live) {
    gram_stage<D, B, S>(sm, buf, warp, lane, slabs, acc);
    if constexpr ((S + 1) % A::kStagesPerLevel == 0) fold_level(S < A::kStagesPerLevel, acc);
    clock.lap(kPhaseGram);
  }
  __syncwarp();
  if (lane == 0) ops::mbar_arrive(ops::smem_addr(&sm.empty[buf]));
  if constexpr (S == A::kStages - 1) {
    if (first || live) ops::mbar_wait(ops::smem_addr(&sm.values_full), tile & 1);
    if (live) {
      epilogue(sm, tile == row_tile, warp, lane, acc);
      clock.lap(kPhaseEpilogue);
      if (p.weight_digits == 2) {
        if (p.v_lo.base != nullptr)
          value_product<D, B, 2, true>(sm, lane, acc);
        else
          value_product<D, B, 2, false>(sm, lane, acc);
      } else {
        if (p.v_lo.base != nullptr)
          value_product<D, B, 1, true>(sm, lane, acc);
        else
          value_product<D, B, 1, false>(sm, lane, acc);
      }
      clock.lap(kPhaseValue);
    }
    __syncwarp();
    if (lane == 0) ops::mbar_arrive(ops::smem_addr(&sm.values_empty));
    values_out = false;
  }
}

//: THE PROLOGUE'S COPIES: the reader's parts in stage order, the first two writer stages interleaved after parts 0
//: and 1, so tile 0's first stage waits for one part and one stage.
template <int D, int B, int... S>
__device__ __forceinline__ void load_reader_parts(Smem<D, B>& sm, const IntraParams& p,
                                                  const Copier<D, B>& copy, int b, int h,
                                                  int t_block, int tid,
                                                  std::integer_sequence<int, S...>) {
  const auto part = [&](auto I) {
    constexpr int kS = decltype(I)::value;
    load_reader_part<D, B, kS>(sm, p, b, h, t_block, tid);
    if constexpr (kS < kStageBuffers - 1) copy.template stage<kS>(sm, p, kS, 0, true);
  };
  (part(std::integral_constant<int, S>{}), ...);
}

template <int D, int B, int... S>
__device__ __forceinline__ void tile_stages(Smem<D, B>& sm, const IntraParams& p,
                                            const Copier<D, B>& copy, int tile, int tiles,
                                            int row_tile, int warp, int lane, bool& values_out,
                                            Acc& acc, Clock& clock,
                                            std::integer_sequence<int, S...>) {
  (tile_stage<D, B, S>(sm, p, copy, tile, tiles, row_tile, warp, lane, values_out, acc, clock),
   ...);
}

//: THE GRID: a CTA a (window, block) along x -- a window's heaviest block first -- and a batch-head along y. The
//: block's readers land once; the writer tiles `0 .. 2 block + 1` stream through the stage buffers, each warp waiting
//: only on the data it reads and releasing it after. -- see docs/internals/intra/intra_kernel.md#the-grid
template <int D, int B>
__global__ __launch_bounds__(kThreads, 1) void intra_kernel(const IntraParams p) {
  using A = Arm<D, B>;
  extern __shared__ __align__(128) unsigned char raw[];
  Smem<D, B>& sm = *reinterpret_cast<Smem<D, B>*>(raw);
  const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
  const int bh = static_cast<int>(blockIdx.y);
  const int window = static_cast<int>(blockIdx.x) / kBlocks;
  const int block = kBlocks - 1 - static_cast<int>(blockIdx.x) % kBlocks;
  const int b = bh / p.H, h = bh - b * p.H;
  const int t_window = window * kWindow;
  const int t_block = t_window + block * kBlockRows;
  const int row_tile = block * kRowTiles + warp / (kWarps / kRowTiles);
  const int tiles = (block + 1) * kRowTiles;
  Clock clock(&sm.clock[warp][0], lane);
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < kStageBuffers; ++i) {
      ops::mbar_init(ops::smem_addr(&sm.full[i]), kThreads);
      ops::mbar_init(ops::smem_addr(&sm.empty[i]), kWarps);
    }
    constexpr int kParts = A::kStages;
#pragma unroll
    for (int s = 0; s < kParts; ++s) ops::mbar_init(ops::smem_addr(&sm.reader_full[s]), kThreads);
    ops::mbar_init(ops::smem_addr(&sm.values_full), kThreads);
    ops::mbar_init(ops::smem_addr(&sm.values_empty), kWarps);
  }
  __syncthreads();

  Acc acc;
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      acc.out[n][i] = 0.0f;
    }
  if (lane < kWarpRows) sm.den[warp][lane] = 0.0f;

  const Copier<D, B> copy(p, b, h, t_window, tid);
  load_reader_parts(sm, p, copy, b, h, t_block, tid, std::make_integer_sequence<int, A::kStages>{});
  copy.values(sm, p, 0, true);
  clock.lap(kPhaseReader);
  if (warp == 0) gate_table(sm, p, bh, window, block, lane);
  __syncthreads();
  clock.lap(kPhaseGate);

  //: tile 0's values left with the prologue
  bool values_out = true;
#pragma unroll 1
  for (int tile = 0; tile < tiles; ++tile)
    tile_stages(sm, p, copy, tile, tiles, row_tile, warp, lane, values_out, acc, clock,
                std::make_integer_sequence<int, A::kStages>{});
  deposit(sm, p, bh, t_block + warp * kWarpRows, warp, lane, acc);
  clock.lap(kPhaseDeposit);
  clock.publish(p.ledger, 0, kWarps, warp);
}

}  // namespace rola::intra::detail
