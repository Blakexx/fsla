// csrc/rola/src/intra/intra_kernel.cuh -- THE INTRA KERNEL: a CTA a (batch-head, window, block of 128 readers), the
// block's reader rows resident in shared memory, the writer tiles streamed through three stage buffers under full and
// empty mbarriers under the slab gate the facts pass's words give; a tile's Gram over each level's digits, the epilogue
// into the value product's weights, the value product, and the block's rows reduced into the output planes at the end.
// see docs/internals/intra/intra_kernel.md
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "common/ops.cuh"
#include "common/phase_clock.cuh"
#include "common/structure_switch.cuh"
#include "facts/liveness_contract.cuh"
#include "intra/intra.cuh"

namespace rola::intra::detail {

namespace ops = denseref::ops;

//: THE WINDOW'S PIECES where a CTA reads its own: one uniform constant load
__constant__ Pieces kPieceTable = Pieces{};
using Clock = PhaseClock<kIntraPhases, true>;

//: THE SHARED TENANTS: three writer stages, the tile's value rows and writer columns (the gain, the value scale, each
//: write level's scale), the slab gate's table and the pipeline's mbarriers; the block's reader rows follow them,
//: `pitch` bytes a row (`reader_rows`). -- see docs/internals/intra/intra_kernel.md#smem
struct alignas(128) Smem {
  uint8_t stage[kStageBuffers][kTile][kStageDigits];
  uint8_t v_hat[kTile][kValueWidth];
  uint8_t v_lo[kTile][kValueWidth];
  float column[kColumns][kTile];
  float den[kWarps][kWarpRows];
  uint32_t gate[kWindowTiles][kRowTiles];
  uint64_t full[kStageBuffers], empty[kStageBuffers];
  uint64_t reader_full[kMaxStages], values_full;
  int claim[kStageBuffers];
  int values_released[2];
  long long clock[kWarps][Clock::kRow];
};

//: THE READER ROWS: `kBlockRows` rows of `pitch` bytes past the tenants, stage `s`'s digits in the row's slot `s`.
__device__ __forceinline__ uint8_t* reader_rows(Smem& sm) {
  return reinterpret_cast<uint8_t*>(&sm + 1);
}

__device__ __forceinline__ const uint8_t* reader_rows(const Smem& sm) {
  return reinterpret_cast<const uint8_t*>(&sm + 1);
}

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

//: A DIGIT'S UNIT: a digit is its unit value times 255, so a depth-`D` integer Gram product is 255^(2 D) its unit one.
template <int D>
constexpr float kUnit = 1.0f / (D == 1 ? 65025.0f : 65025.0f * 65025.0f);

//: CHUNK `k` OF A ROW in a level's stage `I`: the copy reads it where the level holds it, and a chunk past the level's
//: `width` digits (its last slab's tail, a short stage's empty slots) reads nothing and lands as zeros.
struct StageChunk {
  const uint8_t* row;
  bool live;
};

template <int I>
__device__ __forceinline__ StageChunk stage_chunk(const RowView& v, int width, int b, int t, int h,
                                                  int k) {
  const int digit = I * kStageDigits + k * kChunk;
  const bool live = digit < width;
  return StageChunk{at<uint8_t>(v, b, t, h) + (live ? digit : 0), live};
}

//: A WARP'S COPIES: one warp issues a whole stage or a tile's values -- the first warp to need a stage's buffer, the
//: last to release a tile's values -- so no warp's data waits on another warp's progress. A lane's place in a copy is
//: fixed (a stage row `lane / 8 + 4 i` and chunk `lane % 8`, a value row `lane / 4 + 8 i` and chunk `lane % 4`), so each
//: base is computed once, 64-bit, and a copy adds its tile's rows by one 32 x 32 -> 64 multiply-add.
//: -- see docs/internals/intra/intra_kernel.md#copies
template <int D>
struct Copier {
  static constexpr int kStageChunks = kStageDigits / kChunk;
  static constexpr int kStageRows = 32 / kStageChunks;
  static constexpr int kStagePerLane = kTile / kStageRows;
  static constexpr int kValueRows = 32 / (kValueWidth / kChunk);
  static constexpr int kValuePerLane = kTile / kValueRows;
  static constexpr int kColumnPerLane = (2 + D) * kTile / 32;
  static_assert(kStageRows == 4 && kValueRows == 8,
                "a lane's swizzled chunk is one bit flip across its rows");
  const uint8_t* stage_row[D];
  const uint8_t* value_row;
  const uint8_t* lo_row;
  uint32_t stage_dst, value_dst;
  //: bit `kLevelStages L + I`: the lane's chunk of level `L`'s stage `I` lies inside the level's digits
  uint32_t stage_live;
  int b, h, t_window, lane;

  __device__ __forceinline__ Copier(const IntraParams& p, int b_, int h_, int t_window_, int lane_)
      : stage_live(0u), b(b_), h(h_), t_window(t_window_), lane(lane_) {
    const int j = lane / kStageChunks, k = lane % kStageChunks;
#pragma unroll
    for (int l = 0; l < D; ++l) {
      stage_row[l] = at<uint8_t>(p.write[l], b, t_window + j, h) + k * kChunk;
#pragma unroll
      for (int i = 0; i < kLevelStages; ++i)
        if (i * kStageDigits + k * kChunk < p.width[l]) stage_live |= 1u << (kLevelStages * l + i);
    }
    stage_dst = static_cast<uint32_t>(j * kStageDigits + swizzle8(j, k) * kChunk);
    const int vj = lane / (kValueWidth / kChunk), vk = lane % (kValueWidth / kChunk);
    value_row = at<uint8_t>(p.v_hat, b, t_window + vj, h) + vk * kChunk;
    lo_row =
        p.v_lo.base != nullptr ? at<uint8_t>(p.v_lo, b, t_window + vj, h) + vk * kChunk : nullptr;
    value_dst = static_cast<uint32_t>(vj * kValueWidth + swizzle4(vj, vk) * kChunk);
  }

  //: tile `tile`'s level-`L` stage `I` by this warp's lanes, each arriving on the buffer's full mbarrier when its copies
  //: land; a stage no warp reads is not copied, and its arrivals are immediate. Row `j + 4 i`'s chunk is
  //: `k ^ j ^ 4 (i & 1)`.
  template <int L, int I>
  __device__ __forceinline__ void stage(const Smem& sm, const IntraParams& p, int buf, int tile,
                                        bool issue) const {
    if (!issue) {
      ops::mbar_arrive(ops::smem_addr(&sm.full[buf]));
      return;
    }
    const bool live = (stage_live >> (kLevelStages * L + I)) & 1u;
    const uint8_t* const row = stage_row[L] + (live ? I * kStageDigits : 0);
    const int st = static_cast<int>(p.write[L].st);
    const uint32_t dst = ops::smem_addr(&sm.stage[buf][0][0]) + stage_dst;
#pragma unroll
    for (int i = 0; i < kStagePerLane; ++i)
      ops::stage_tile_if((dst + i * kStageRows * kStageDigits) ^ ((i & 1) * 4 * kChunk),
                         row + rows_of(tile * kTile + i * kStageRows, st), live);
    ops::mbar_arrive_when_landed(ops::smem_addr(&sm.full[buf]));
  }

  //: tile `tile`'s value digits and writer columns by this warp's lanes, or immediate arrivals when no warp's pair with
  //: it is live
  __device__ __forceinline__ void values(const Smem& sm, const IntraParams& p, int tile,
                                         bool issue) const {
    if (!issue) {
      ops::mbar_arrive(ops::smem_addr(&sm.values_full));
      return;
    }
    const int st = static_cast<int>(p.v_hat.st);
    const uint32_t hat = ops::smem_addr(&sm.v_hat[0][0]) + value_dst;
    const uint32_t lo = ops::smem_addr(&sm.v_lo[0][0]) + value_dst;
#pragma unroll
    for (int i = 0; i < kValuePerLane; ++i) {
      const uint64_t rows = rows_of(tile * kTile + i * kValueRows, st);
      ops::stage_tile(hat + i * kValueRows * kValueWidth, value_row + rows);
      if (lo_row != nullptr) ops::stage_tile(lo + i * kValueRows * kValueWidth, lo_row + rows);
    }
#pragma unroll
    for (int i = 0; i < kColumnPerLane; ++i) {
      const int c = i / 2, t = lane + (i % 2) * 32;
      const ColumnView& v = c == 0 ? p.g_write : c == 1 ? p.v_scale : p.scale_write[c - 2];
      ops::stage_run<4>(ops::smem_addr(&sm.column[c][t]),
                        v.base + b * v.sb + h * v.sh + (t_window + tile * kTile + t) * v.st);
    }
    ops::mbar_arrive_when_landed(ops::smem_addr(&sm.values_full));
  }
};

//: THE BLOCK'S READER ROWS, once, a stage's digits at a time: level `L`'s stage `I` goes out as its own part into the
//: row's slot for the stage and arrives on its own mbarrier, so the first Gram waits for one part, not the block's
//: rows; a chunk a copy, consecutive threads along a token's run.
template <int L, int I>
__device__ __forceinline__ void load_reader_part(Smem& sm, const IntraParams& p, int b, int h,
                                                 int t0, int tid) {
  constexpr int kChunks = kStageDigits / kChunk;
  constexpr int kCopies = kBlockRows * kChunks / kThreads;
  constexpr int kRowsPerCopy = kThreads / kChunks;
  const int r = tid / kChunks, k = tid % kChunks;
  const int slot = p.stage[L][I].index;
  const RowView& v = p.read[L];
  const StageChunk chunk = stage_chunk<I>(v, p.width[L], b, t0 + r, h, k);
  const int st = static_cast<int>(v.st);
  const uint32_t dst =
      ops::smem_addr(reader_rows(sm) + r * p.pitch + slot * kStageDigits + swizzle8(r, k) * kChunk);
#pragma unroll
  for (int i = 0; i < kCopies; ++i)
    ops::stage_tile_if(dst + i * kRowsPerCopy * p.pitch, chunk.row + rows_of(i * kRowsPerCopy, st),
                       chunk.live);
  ops::mbar_arrive_when_landed(ops::smem_addr(&sm.reader_full[slot]));
}

//: THE GATE TABLE: per (window tile, row tile) the slabs whose reader and writer tile unions meet (the facts pass's), a
//: lane a slab, cleared whole when a level has none (the pair's product is exactly zero) or the pair lies above the
//: diagonal; a dense level answers all-ones unread. One warp. -- see docs/internals/intra/intra_kernel.md#slab-gate
template <int D>
__device__ __forceinline__ void gate_table(Smem& sm, const IntraParams& p, int bh, int window,
                                           int block, int lane) {
  const int tiles = p.L / facts::kTileTokens;
  int level = 0;
#pragma unroll
  for (int l = 1; l < D; ++l)
    if (lane >= p.slab0[l]) level = l;
  const int group = p.group0[level] + lane - p.slab0[level];
  const bool slab = lane < p.slabs;
  const uint32_t all = slab ? ~0u : 0u;
  const int tile0 = window * kWindowTiles;
  uint2 reader = {all, all};
  uint4 writer[2] = {{all, all, all, all}, {all, all, all, all}};
  if (slab && !facts::level_is_dense(p.dense[facts::kRead], level))
    reader = *reinterpret_cast<const uint2*>(p.unions
                                             + facts::union_address(bh, facts::kRead, group,
                                                                    tile0 + block * kRowTiles,
                                                                    p.groups, tiles));
  if (slab && !facts::level_is_dense(p.dense[facts::kWrite], level)) {
    const uint32_t* const w =
        p.unions + facts::union_address(bh, facts::kWrite, group, tile0, p.groups, tiles);
    writer[0] = ops::load_vec16(w);
    writer[1] = ops::load_vec16(w + 4);
  }
  const uint32_t rows[kRowTiles] = {reader.x, reader.y};
  const uint32_t columns[kWindowTiles] = {writer[0].x, writer[0].y, writer[0].z, writer[0].w,
                                          writer[1].x, writer[1].y, writer[1].z, writer[1].w};
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

//: whether tile `tile`'s level-`L` stage `I` is copied: tile 0's always (its copies leave before the gate is known), a
//: later tile's when either row tile reads one of its slabs
template <int L, int I>
__device__ __forceinline__ bool stage_issued(const Smem& sm, const IntraParams& p, int tile,
                                             int t0) {
  return tile == t0 || stage_slabs(p.stage[L][I], sm.gate[tile][0] | sm.gate[tile][1]) != 0u;
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
//: token-major rows, the reader's from the stage's slot `slot` of its row. -- see docs/internals/intra/intra_kernel.md#gram
template <int K, bool kFresh>
__device__ __forceinline__ void gram_step(const Smem& sm, const uint8_t* slot, int buf, int warp,
                                          int lane, Acc& acc) {
  const int arow = warp * kWarpRows + (lane & 15);
  const int brow = (lane & 7) + ((lane >> 4) << 3);
  uint32_t a[4];
  ops::load_frag(a, ops::smem_addr(slot + swizzle8(arow, 2 * K + (lane >> 4)) * kChunk));
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

//: A WHOLE STAGE'S K-STEPS, straight-line, the first fresh when the stage opens its level
template <bool kFresh>
__device__ __forceinline__ void gram_block(const Smem& sm, const uint8_t* slot, int buf, int warp,
                                           int lane, Acc& acc) {
  rola::static_for<kSlabsPerStage>([&](auto K) {
    constexpr int kK = decltype(K)::value;
    gram_step<kK, kFresh && kK == 0>(sm, slot, buf, warp, lane, acc);
  });
}

//: THE GRAM OF ONE STAGE, its k-steps under the warp's slab bits (`slabs`, the stage's own, none past its last): one
//: straight-line block when a whole stage is live, a branch a step otherwise; a level's first step (`kFirst`), gated
//: off, zeroes the accumulator in its place.
template <bool kFirst>
__device__ __forceinline__ void gram_stage(const Smem& sm, const uint8_t* slot, int buf, int warp,
                                           int lane, uint32_t slabs, Acc& acc) {
  if (slabs == (1u << kSlabsPerStage) - 1u) {
    gram_block<kFirst>(sm, slot, buf, warp, lane, acc);
    return;
  }
  if (slabs & 1u) {
    gram_step<0, kFirst>(sm, slot, buf, warp, lane, acc);
  } else if constexpr (kFirst) {
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc.gram[n][i] = 0;
  }
  rola::static_for<kSlabsPerStage - 1>([&](auto K) {
    constexpr int kK = decltype(K)::value + 1;
    if ((slabs >> kK) & 1u) gram_step<kK, false>(sm, slot, buf, warp, lane, acc);
  });
}

//: A LEVEL'S GRAM INTO THE RUNNING PRODUCT: exact s32 to fp32 (a level's sum is under 2^24), multiplied level by level.
template <bool kFirst>
__device__ __forceinline__ void fold_level(Acc& acc) {
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float g = static_cast<float>(acc.gram[n][i]);
      acc.product[n][i] = kFirst ? g : acc.product[n][i] * g;
    }
}

//: THE EPILOGUE: the level product times the writer factor (a lane's two columns' factors, shuffled to the lanes that
//: hold them), zero above the diagonal on a diagonal tile, the row sums into `den`, and the value scale folded in: the
//: value product's `P`. -- see docs/internals/intra/intra_kernel.md#epilogue
template <int D>
__device__ __forceinline__ void epilogue(Smem& sm, bool diagonal, int warp, int lane, Acc& acc) {
  float f[2], s[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    f[j] = sm.column[0][2 * lane + j] * kUnit<D>;
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
template <int kWeightDigits, bool kValueLo>
__device__ __forceinline__ void value_product(const Smem& sm, int lane, Acc& acc) {
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
__device__ void deposit(Smem& sm, const IntraParams& p, int bh, int t_row, int warp, int lane,
                        Acc& acc) {
  constexpr int kPitch = kValueWidth + 4;
  static_assert(kWarpRows * kPitch * 4 <= kWarpRows * kMinPitchStages * kStageDigits,
                "a warp stages in its own reader rows");
  //: the warp's own reader rows, which no other warp reads: its last Gram is done, and no CTA barrier is needed
  __syncwarp();
  float* const staging = reinterpret_cast<float*>(reader_rows(sm) + warp * kWarpRows * p.pitch);
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

//: ONE STAGE OF A TILE, level `L`'s stage `I`, the level's last when `kLevelEnds`: the next stage's copy claimed by the
//: first warp here -- the level's second stage, the next level's first, or past the last level the next tile's first;
//: this stage awaited by every warp and read where the warp's slabs meet it, the level folded at its last stage, the
//: buffer released. -- see docs/internals/intra/intra_kernel.md#pipeline
template <int D, int L, int I, bool kLevelEnds>
__device__ __forceinline__ void tile_stage(Smem& sm, const IntraParams& p, const Copier<D>& copy,
                                           const uint8_t* reader_row, int tile, int t0, int t1,
                                           int row_tile, int warp, int lane, Acc& acc,
                                           Clock& clock) {
  //: lane 0 claims, releases and counts for its warp
  const bool leader = lane == 0;
  const Stage& g = p.stage[L][I];
  constexpr bool kWraps = L == D - 1 && kLevelEnds;
  //: the piece's own stage index: its copies, claims and phases count from the piece's first tile
  const int r = tile - t0, it = r * p.stages + g.index;
  const int next = it + 1;
  //: the first warp to reach a stage's refill claims it and alone waits for the buffer's release
  if (!kWraps || tile + 1 < t1) {
    const int nbuf = next % kStageBuffers, use = next / kStageBuffers;
    int claimed = 0;
    if (leader) claimed = atomicMax(&sm.claim[nbuf], next) < next;
    if (__shfl_sync(0xffffffffu, claimed, 0)) {
      if (use > 0) ops::mbar_wait(ops::smem_addr(&sm.empty[nbuf]), (use - 1) & 1);
      if constexpr (kWraps)
        copy.template stage<0, 0>(sm, p, nbuf, tile + 1, stage_issued<0, 0>(sm, p, tile + 1, t0));
      else if constexpr (!kLevelEnds)
        copy.template stage<L, I + 1>(sm, p, nbuf, tile, stage_issued<L, I + 1>(sm, p, tile, t0));
      else
        copy.template stage<L + 1, 0>(sm, p, nbuf, tile, stage_issued<L + 1, 0>(sm, p, tile, t0));
    }
  }
  const int buf = it % kStageBuffers;
  //: the warp's slabs of this tile pair: none when the pair is dead or above the diagonal
  const uint32_t mine = sm.gate[tile][row_tile % kRowTiles];
  const uint32_t slabs = stage_slabs(g, mine);
  const bool live = mine != 0u;
  //: EVERY warp waits for every stage, read or not: a barrier's wait tests one phase bit, so it is sound only for a
  //: waiter within a phase of the barrier, and a warp that skipped waits could run ahead, claim a refill and read a
  //: release two phases old as done. A stage no warp reads was not copied, and its wait is immediate.
  if (tile == t0) ops::mbar_wait(ops::smem_addr(&sm.reader_full[g.index]), 0);
  ops::mbar_wait(ops::smem_addr(&sm.full[buf]), (it / kStageBuffers) & 1);
  clock.lap(kPhaseWait);
  if (live) {
    gram_stage<I == 0>(sm, reader_row + g.index * kStageDigits, buf, warp, lane, slabs, acc);
    if constexpr (kLevelEnds) fold_level<L == 0>(acc);
    clock.lap(kPhaseGram);
  }
  __syncwarp();
  if (leader) ops::mbar_arrive(ops::smem_addr(&sm.empty[buf]));
}

//: A TILE'S VALUES: awaited after its last stage; the epilogue and the value product for a live pair, the values
//: released and the next tile's copied by the last warp to release them.
template <int D>
__device__ __forceinline__ void tile_values(Smem& sm, const IntraParams& p, const Copier<D>& copy,
                                            int tile, int t0, int t1, int row_tile, int warp,
                                            int lane, Acc& acc, Clock& clock) {
  const bool leader = lane == 0;
  const int r = tile - t0;
  ops::mbar_wait(ops::smem_addr(&sm.values_full), r & 1);
  if (sm.gate[tile][row_tile % kRowTiles] != 0u) {
    epilogue<D>(sm, tile == row_tile, warp, lane, acc);
    clock.lap(kPhaseEpilogue);
    if (p.weight_digits == 2) {
      if (p.v_lo.base != nullptr)
        value_product<2, true>(sm, lane, acc);
      else
        value_product<2, false>(sm, lane, acc);
    } else {
      if (p.v_lo.base != nullptr)
        value_product<1, true>(sm, lane, acc);
      else
        value_product<1, false>(sm, lane, acc);
    }
    clock.lap(kPhaseValue);
  }
  //: the last warp to release the tile's values copies the next tile's. A warp with no live pair releases without
  //: waiting, so releases are counted by the tile's parity: a claim's wait keeps every warp within a tile of the
  //: slowest, and tiles `t` and `t + 2` never interleave on one counter
  __syncwarp();
  int last = 0;
  if (leader) {
    __threadfence_block();
    last = atomicAdd(&sm.values_released[r & 1], 1) == (r / 2 + 1) * kWarps - 1;
  }
  if (__shfl_sync(0xffffffffu, last, 0) && tile + 1 < t1) {
    __threadfence_block();
    copy.values(sm, p, tile + 1, (sm.gate[tile + 1][0] | sm.gate[tile + 1][1]) != 0u);
  }
}

//: THE GRID: a CTA a (window, piece) along x -- a window's longest piece first -- and a batch-head along y, an arm a
//: depth `D`. The block's readers land once, a part a stage, tile 0's first stage beside the first part (warp 0's, as a
//: claimant's); the piece's writer tiles stream through the stage buffers, level by level, a level's one or two stages
//: each, and its partial sums reduce into the output planes. -- see docs/internals/intra/intra_kernel.md#the-grid
template <int D>
__global__ __launch_bounds__(kThreads, 1) void intra_kernel(const __grid_constant__ IntraParams p) {
  extern __shared__ __align__(128) unsigned char raw[];
  Smem& sm = *reinterpret_cast<Smem*>(raw);
  const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
  const int bh = static_cast<int>(blockIdx.y);
  const int window = static_cast<int>(blockIdx.x) / Pieces::count();
  const Piece piece = kPieceTable.table[static_cast<int>(blockIdx.x) % Pieces::count()];
  const int block = piece.block;
  const int b = bh / p.H, h = bh - b * p.H;
  const int t_window = window * kWindow;
  const int t_block = t_window + block * kBlockRows;
  const int row_tile = block * kRowTiles + warp / (kWarps / kRowTiles);
  Clock clock(&sm.clock[warp][0], lane, p.ledger != nullptr);
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < kStageBuffers; ++i) {
      ops::mbar_init(ops::smem_addr(&sm.full[i]), 32);
      ops::mbar_init(ops::smem_addr(&sm.empty[i]), kWarps);
      sm.claim[i] = i == 0 ? 0 : -1;
    }
    sm.values_released[0] = sm.values_released[1] = 0;
#pragma unroll 1
    for (int s = 0; s < p.stages; ++s) ops::mbar_init(ops::smem_addr(&sm.reader_full[s]), kThreads);
    ops::mbar_init(ops::smem_addr(&sm.values_full), 32);
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

  const Copier<D> copy(p, b, h, t_window, lane);
  //: the warp's reader row this lane's A fragments address
  const uint8_t* const reader_row = reader_rows(sm) + (warp * kWarpRows + (lane & 15)) * p.pitch;
  rola::static_for<D>([&](auto Level) {
    constexpr int kL = decltype(Level)::value;
    load_reader_part<kL, 0>(sm, p, b, h, t_block, tid);
    if constexpr (kL == 0)
      if (tid < 32) copy.template stage<0, 0>(sm, p, 0, piece.t0, true);
    if (p.level_stages[kL] == 2) load_reader_part<kL, 1>(sm, p, b, h, t_block, tid);
  });
  if (warp == 0) copy.values(sm, p, piece.t0, true);
  clock.lap(kPhaseReader);
  if (warp == 0) gate_table<D>(sm, p, bh, window, block, lane);
  __syncthreads();
  clock.lap(kPhaseGate);

#pragma unroll 1
  for (int tile = piece.t0; tile < piece.t1; ++tile) {
    rola::static_for<D>([&](auto Level) {
      constexpr int kL = decltype(Level)::value;
      if (p.level_stages[kL] == 2) {
        tile_stage<D, kL, 0, false>(sm, p, copy, reader_row, tile, piece.t0, piece.t1, row_tile,
                                    warp, lane, acc, clock);
        tile_stage<D, kL, 1, true>(sm, p, copy, reader_row, tile, piece.t0, piece.t1, row_tile,
                                   warp, lane, acc, clock);
      } else {
        tile_stage<D, kL, 0, true>(sm, p, copy, reader_row, tile, piece.t0, piece.t1, row_tile,
                                   warp, lane, acc, clock);
      }
    });
    tile_values(sm, p, copy, tile, piece.t0, piece.t1, row_tile, warp, lane, acc, clock);
  }
  deposit(sm, p, bh, t_block + warp * kWarpRows, warp, lane, acc);
  clock.lap(kPhaseDeposit);
  clock.publish(p.ledger, 0, kWarps, warp);
}

}  // namespace rola::intra::detail
