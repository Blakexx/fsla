// THE CARRY'S COEFFICIENT LINES, a round's expanded into its warp's tile -- docs/internals/carry/carry_coefs.md
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "carry/carry_groups.cuh"
#include "carry/carry_list.cuh"
#include "carry/carry_values.cuh"
#include "common/atoms/address.cuh"
#include "common/atoms/copy.cuh"
#include "common/atoms/math.cuh"
#include "common/atoms/memory.cuh"
#include "common/atoms/tensor.cuh"
#include "common/atoms/warp.cuh"

namespace fsla {
namespace carry {
namespace detail {

//: FIXED by a level's rows on the card: a tile of 64 tokens, its 16-digit blocks one after another, a token's 16 bytes a line -- see docs/internals/carry/carry_coefs.md#rows
constexpr int kRowTileBits = 6, kRowBlockBits = 4;
constexpr int kRowBlockBytes = (1 << kRowTileBits) << kRowBlockBits;
//: FIXED by the step's operand of coefficients: a line is one token's 8 columns, a brick the 16 tokens' lines of 8 columns
constexpr int kBrickColumns = 8;
constexpr int kBrickBytes = kStepTokens * 2 * kBrickColumns;
//: FIXED by the tensor step (R25): a step is never wider than 64 columns
constexpr int kStepColumnBits = 6;

//: A STEP'S SHAPE: the bits of its columns a POSITION, position 0 the columns' lowest -- see docs/internals/carry/carry_coefs.md#shape
template <int B0, int B1, int B2, int B3>
struct StepShape {
  static constexpr int kHighBits = B1 + B2 + B3;
  static constexpr int kColumnBits = B0 + kHighBits;
  static constexpr int kColumns = 1 << kColumnBits;
  static constexpr int kBricks = kColumns / kBrickColumns;
  static_assert(kColumnBits >= 4 && kColumnBits <= kStepColumnBits, "a step is 16 to 64 columns");
};

//: A ROUND'S TILE OF COEFFICIENTS: its two steps' 16 tokens each, a step's bricks one after another
template <class S>
constexpr int kCoefStepBytes = S::kBricks * kBrickBytes;

template <class S>
constexpr int kCoefRoundBytes = (ops::kBallotLanes / kStepTokens) * kCoefStepBytes<S>;

//: WHERE A STEP'S COEFFICIENTS ARE READ, a position: token 0's line of the block its first digit lies in, that digit's place in the block, and a tile of the level's 64 tokens in bytes -- see docs/internals/carry/carry_coefs.md#source
struct CoefSource {
  const std::uint8_t* block[kLevels];
  int inside[kLevels];
  int tile_bytes[kLevels];
};

//: THE BLOCKS A POSITION'S BYTES OF A TOKEN ARE STAGED AS: its 16-digit blocks whole, one where its digits are fewer
template <int B>
constexpr int kPositionBlocks = B > kRowBlockBits ? 1 << (B - kRowBlockBits) : 1;

//: A TOKEN'S STAGED BYTES, a chunk: the positions' blocks one after another, 16 bytes each; a token's slot a PITCH of lines that is odd, so that the lanes' lines spread over the banks -- see docs/internals/carry/carry_coefs.md#stage
template <int LEVELS, int B0, int B1, int B2, int B3>
struct StageShape {
  static constexpr int kAt[kLevels + 1] = {
      0, LEVELS > 0 ? kPositionBlocks<B0> : 0,
      (LEVELS > 0 ? kPositionBlocks<B0> : 0) + (LEVELS > 1 ? kPositionBlocks<B1> : 0),
      (LEVELS > 0 ? kPositionBlocks<B0> : 0) + (LEVELS > 1 ? kPositionBlocks<B1> : 0)
          + (LEVELS > 2 ? kPositionBlocks<B2> : 0),
      (LEVELS > 0 ? kPositionBlocks<B0> : 0) + (LEVELS > 1 ? kPositionBlocks<B1> : 0)
          + (LEVELS > 2 ? kPositionBlocks<B2> : 0) + (LEVELS > 3 ? kPositionBlocks<B3> : 0)};
  static constexpr int kLines = kAt[kLevels];
  static constexpr int kPitch = kLines | 1;
  static constexpr int kBytes = ops::kBallotLanes * kPitch * (1 << kRowBlockBits);
};

//: THE WIDEST STAGE of the classes built: 64 digits of a level and a block of another, a pitch of 5 lines
constexpr int kStageMost = ops::kBallotLanes * 5 * (1 << kRowBlockBits);

//: A FOLD'S CLASS: the call's levels, its step's shape and the bits of a set's CHUNKS, a set wider than a step being `1 << chunk_bits` steps' columns -- see docs/internals/carry/carry_coefs.md#class
struct FoldClass {
  int levels;
  int bits[kLevels];
  int chunk_bits;
};

//: THE CLASSES OF ONE LEVEL AND OF TWO: 16, 32 and 64 columns, then a set of two and of four chunks of 64 -- see docs/internals/carry/carry_coefs.md#class
constexpr FoldClass kFoldClasses[] = {
    {1, {4, 0, 0, 0}, 0}, {1, {5, 0, 0, 0}, 0}, {1, {6, 0, 0, 0}, 0}, {1, {6, 0, 0, 0}, 1},
    {1, {6, 0, 0, 0}, 2}, {2, {4, 0, 0, 0}, 0}, {2, {3, 1, 0, 0}, 0}, {2, {2, 2, 0, 0}, 0},
    {2, {5, 0, 0, 0}, 0}, {2, {4, 1, 0, 0}, 0}, {2, {3, 2, 0, 0}, 0}, {2, {6, 0, 0, 0}, 0},
    {2, {5, 1, 0, 0}, 0}, {2, {4, 2, 0, 0}, 0}, {2, {3, 3, 0, 0}, 0}, {2, {6, 0, 0, 0}, 1},
    {2, {5, 1, 0, 0}, 1}, {2, {4, 2, 0, 0}, 1}, {2, {3, 3, 0, 0}, 1}, {2, {6, 0, 0, 0}, 2},
    {2, {5, 1, 0, 0}, 2}, {2, {4, 2, 0, 0}, 2}, {2, {3, 3, 0, 0}, 2}, {2, {2, 4, 0, 0}, 2}};
constexpr int kFoldClassCount = static_cast<int>(sizeof(kFoldClasses) / sizeof(kFoldClasses[0]));

//: THE CLASS OF A CALL'S LEVELS, A STEP'S BITS A POSITION AND A SET'S CHUNKS, from class `I` on; `kFoldClassCount` where the table has none
template <int I>
__host__ __device__ __forceinline__ int fold_class_from(int levels, const int* bits,
                                                        int chunk_bits) {
  if constexpr (I < kFoldClassCount) {
    constexpr FoldClass k = kFoldClasses[I];
    const bool is = levels == k.levels && bits[0] == k.bits[0] && bits[1] == k.bits[1]
                    && bits[2] == k.bits[2] && bits[3] == k.bits[3] && chunk_bits == k.chunk_bits;
    const int later = fold_class_from<I + 1>(levels, bits, chunk_bits);
    return is ? I : later;
  } else {
    return kFoldClassCount;
  }
}

//: A FOLD'S POSITIONS, formed once a fold: its levels by their extent in a set, the widest first -- see docs/internals/carry/carry_coefs.md#positions
struct FoldPositions {
  const std::uint8_t* rows[kLevels];
  int tile_bytes[kLevels];
  int first[kLevels], shift[kLevels], mask[kLevels];
  int fold_class;
};

//: `shift` and `mask` are the fold's (`fold_levels`); the rows are the head's, at the call's first token
__host__ __device__ __forceinline__ FoldPositions fold_positions(const Operands& o,
                                                                 const plan::PrefillCarve& c,
                                                                 const Place& p, int fold,
                                                                 const int* shift,
                                                                 const int* mask) {
  //: a level's extent in a set, and its place among the levels: each of the carve's arrays is read before the level is compared with anything
  int extent[kLevels], rank[kLevels];
#pragma unroll
  for (int l = 0; l < kLevels; ++l) {
    const int group = c.write_group_bits[l], held = c.fold_bits[l];
    extent[l] = l < c.levels ? (group < held ? group : held) : -1;
  }
#pragma unroll
  for (int l = 0; l < kLevels; ++l) {
    int before = 0;
#pragma unroll
    for (int k = 0; k < kLevels; ++k) {
      before += (extent[k] > extent[l] || (extent[k] == extent[l] && k < l)) ? 1 : 0;
    }
    rank[l] = before;
  }

  FoldPositions f{};
  int bits[kLevels] = {};
#pragma unroll
  for (int l = 0; l < kLevels; ++l) {
    const int wb = o.width_bits[l], held = c.fold_bits[l], carve = c.carve_bits[l];
    const int half = (fold << held) & (l == c.split_level ? ~0 : 0);
    const int tile = (1 << kRowTileBits) << wb;
    const std::uint8_t* const rows =
        o.rows[side_at(false) + l] + p.head * (o.tokens >> kRowTileBits) * tile;
#pragma unroll
    for (int at = 0; at < kLevels; ++at) {
      const bool here = l < c.levels && rank[l] == at;
      f.rows[at] = here ? rows : f.rows[at];
      f.tile_bytes[at] = here ? tile : f.tile_bytes[at];
      f.first[at] = here ? (p.cta[l] << carve) + half : f.first[at];
      f.shift[at] = here ? shift[l] : f.shift[at];
      f.mask[at] = here ? mask[l] : f.mask[at];
      bits[at] = here ? extent[l] : bits[at];
    }
  }

  //: the chunks' bits come off position 0
  const int all = bits[0] + bits[1] + bits[2] + bits[3];
  const int chunk_bits = all > kStepColumnBits ? all - kStepColumnBits : 0;
  bits[0] -= chunk_bits;
  f.fold_class = fold_class_from<0>(c.levels, bits, chunk_bits);
  return f;
}

//: WHERE CHUNK `chunk` OF SET `set` IS READ: a position's first digit of it, in its block -- see docs/internals/carry/carry_coefs.md#source
template <int LEVELS, int B0, int B1, int B2, int B3, int CHUNK_BITS>
__host__ __device__ __forceinline__ CoefSource coef_source(const FoldPositions& f, int set,
                                                           int chunk) {
  constexpr int kExtent[kLevels] = {B0 + CHUNK_BITS, B1, B2, B3};
  CoefSource s{};
#pragma unroll
  for (int at = 0; at < LEVELS; ++at) {
    const int digit = f.first[at] + (((set >> f.shift[at]) & f.mask[at]) << kExtent[at])
                      + (at == 0 ? chunk << B0 : 0);
    const int block = digit >> kRowBlockBits, in_block = digit & ((1 << kRowBlockBits) - 1);
    s.block[at] = f.rows[at] + block * kRowBlockBytes;
    s.inside[at] = in_block;
    s.tile_bytes[at] = f.tile_bytes[at];
  }
  return s;
}

//: THE COPIES OF A POSITION'S BLOCKS of a lane's token, into its slot of the stage
template <int B>
__device__ __forceinline__ void stage_position(ops::SmemAddr to, const std::uint8_t* from,
                                               bool live) {
  constexpr int kBlocks = kPositionBlocks<B>;
#pragma unroll
  for (int k = 0; k < kBlocks; ++k) {
    ops::stage_tile_if(to + static_cast<ops::SmemAddr>(k << kRowBlockBits),
                       from + k * kRowBlockBytes, live);
  }
}

//: WHERE A LANE'S TOKEN'S BLOCKS ARE COPIED FROM AND TO, a chunk: formed apart from the copies, so that a stage's loads and arithmetic all stand before its copies -- see docs/internals/carry/carry_coefs.md#stage
struct StagePlan {
  const std::uint8_t* from[kLevels];
  ops::SmemAddr to;
  bool on;
};

//: lane `i` takes place `i` of `tokens`; nothing is copied where `live` is false or the place holds no token
template <int LEVELS, int B0, int B1, int B2, int B3>
__device__ __forceinline__ StagePlan stage_plan(const CoefSource& c, const std::uint16_t* tokens,
                                                ops::SmemAddr stage, bool live) {
  using G = StageShape<LEVELS, B0, B1, B2, B3>;
  const int lane = static_cast<int>(threadIdx.x) & (ops::kBallotLanes - 1);
  const int listed = tokens[lane];
  StagePlan plan{};
  plan.on = live && listed != kNoToken;
  const int token = plan.on ? listed : 0;
  const int line = (token & ((1 << kRowTileBits) - 1)) << kRowBlockBits;
  const int tile_of = token >> kRowTileBits;
  plan.to = stage + static_cast<ops::SmemAddr>(lane * (G::kPitch << kRowBlockBits));
#pragma unroll
  for (int p = 0; p < LEVELS; ++p) {
    plan.from[p] = c.block[p] + static_cast<std::uint32_t>(tile_of * c.tile_bytes[p] + line);
  }
  return plan;
}

//: THE COPIES OF A CHUNK'S BYTES, uncommitted: a lane's token's blocks a position into its slot of the stage -- see docs/internals/carry/carry_coefs.md#stage
template <int LEVELS, int B0, int B1, int B2, int B3>
__device__ __forceinline__ void stage_bytes(const StagePlan& plan) {
  using G = StageShape<LEVELS, B0, B1, B2, B3>;
  if constexpr (LEVELS > 0) {
    stage_position<B0>(plan.to + (G::kAt[0] << kRowBlockBits), plan.from[0], plan.on);
  }
  if constexpr (LEVELS > 1) {
    stage_position<B1>(plan.to + (G::kAt[1] << kRowBlockBits), plan.from[1], plan.on);
  }
  if constexpr (LEVELS > 2) {
    stage_position<B2>(plan.to + (G::kAt[2] << kRowBlockBits), plan.from[2], plan.on);
  }
  if constexpr (LEVELS > 3) {
    stage_position<B3>(plan.to + (G::kAt[3] << kRowBlockBits), plan.from[3], plan.on);
  }
}

//: A POSITION'S BYTES of a lane's token, from its blocks in the stage, as fp32 a digit: `inside` its first digit's place in its block -- see docs/internals/carry/carry_coefs.md#expand
template <int B>
__device__ __forceinline__ void position_floats(ops::SmemAddr blocks, int inside,
                                                float (&out)[1 << B]) {
  constexpr int kDigits = 1 << B;
  if constexpr (B >= kRowBlockBits) {
    constexpr int kBlocks = kPositionBlocks<B>;
#pragma unroll
    for (int k = 0; k < kBlocks; ++k) {
      const uint4 w = ops::load_shared_v4(blocks + static_cast<ops::SmemAddr>(k << kRowBlockBits));
      const std::uint32_t q[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        out[16 * k + i] = static_cast<float>((q[i / 4] >> (8 * (i % 4))) & 0xffu);
      }
    }
  } else if constexpr (B == 3) {
    const uint2 w = ops::load_shared_u64(blocks + static_cast<ops::SmemAddr>(inside));
    const std::uint32_t q[2] = {w.x, w.y};
#pragma unroll
    for (int i = 0; i < kDigits; ++i) {
      out[i] = static_cast<float>((q[i / 4] >> (8 * (i % 4))) & 0xffu);
    }
  } else {
    //: fewer than 8 digits lie in one word of the block, from a byte of it on
    const std::uint32_t word =
        ops::load_shared_u32(blocks + static_cast<ops::SmemAddr>(inside & ~3));
    const std::uint32_t w = B == 2 ? word : word >> (8 * (inside & 3));
#pragma unroll
    for (int i = 0; i < kDigits; ++i) out[i] = static_cast<float>((w >> (8 * i)) & 0xffu);
  }
}

//: A ROUND'S COEFFICIENT LINES, a chunk: lane `i` reads its token's staged bytes, multiplies them out over the step's columns and stores its line in `tile`; a place with no token stores zeros -- see docs/internals/carry/carry_coefs.md#expand
template <int LEVELS, int B0, int B1, int B2, int B3>
__device__ __forceinline__ void expand_round(const CoefSource& c, const std::uint16_t* tokens,
                                             ops::SmemAddr stage, ops::SmemAddr tile) {
  using S = StepShape<B0, B1, B2, B3>;
  using G = StageShape<LEVELS, B0, B1, B2, B3>;
  constexpr int kBits[kLevels] = {B0, B1, B2, B3};
  constexpr int kLow = 1 << B0, kHigh = 1 << S::kHighBits;
  const int lane = static_cast<int>(threadIdx.x) & (ops::kBallotLanes - 1);
  const bool live = tokens[lane] != kNoToken;
  const ops::SmemAddr from =
      stage + static_cast<ops::SmemAddr>(lane * (G::kPitch << kRowBlockBits));
  //: a position the call has no level for is 1
  float f0[1 << B0] = {1.0f}, f1[1 << B1] = {1.0f}, f2[1 << B2] = {1.0f}, f3[1 << B3] = {1.0f};
  if constexpr (LEVELS > 0) {
    position_floats<B0>(from + (G::kAt[0] << kRowBlockBits), c.inside[0], f0);
  }
  if constexpr (LEVELS > 1) {
    position_floats<B1>(from + (G::kAt[1] << kRowBlockBits), c.inside[1], f1);
  }
  if constexpr (LEVELS > 2) {
    position_floats<B2>(from + (G::kAt[2] << kRowBlockBits), c.inside[2], f2);
  }
  if constexpr (LEVELS > 3) {
    position_floats<B3>(from + (G::kAt[3] << kRowBlockBits), c.inside[3], f3);
  }

  //: a position of one digit is a factor of the whole line; a place with no token is 0, whatever its slot holds
  const float one[kLevels] = {f0[0], f1[0], f2[0], f3[0]};
  float scale = live ? 1.0f : 0.0f;
#pragma unroll
  for (int p = 0; p < LEVELS; ++p) {
    if (kBits[p] == 0) scale *= one[p];
  }
  //: THE COLUMNS' HIGH BITS, positions 1 on, multiplied once; then a column is one multiplication
  float high[kHigh];
#pragma unroll
  for (int h = 0; h < kHigh; ++h) {
    float v = scale;
    if (B1 > 0) v *= f1[h & ((1 << B1) - 1)];
    if (B2 > 0) v *= f2[(h >> B1) & ((1 << B2) - 1)];
    if (B3 > 0) v *= f3[(h >> (B1 + B2)) & ((1 << B3) - 1)];
    high[h] = v;
  }
  const ops::SmemAddr to =
      tile
      + static_cast<ops::SmemAddr>((lane / kStepTokens) * kCoefStepBytes<S>
                                   + (lane % kStepTokens) * (2 * kBrickColumns));
  constexpr int kBricks = S::kBricks, kPairs = kBrickColumns / 2;
#pragma unroll
  for (int j = 0; j < kBricks; ++j) {
    std::uint32_t pair[kPairs];
#pragma unroll
    for (int i = 0; i < kPairs; ++i) {
      const int n = kBrickColumns * j + 2 * i;
      const float lo = (B0 > 0 ? f0[n % kLow] : 1.0f) * high[n >> B0];
      const float hi = (B0 > 0 ? f0[(n + 1) % kLow] : 1.0f) * high[(n + 1) >> B0];
      pair[i] = ops::pack_bf16x2(lo, hi);
    }
    ops::store_shared_vec16(to + static_cast<ops::SmemAddr>(j * kBrickBytes),
                            make_uint4(pair[0], pair[1], pair[2], pair[3]));
  }
}

//: STEP `step` OF A ROUND'S COEFFICIENT TILE as the tensor step's operand: 16 tokens by the shape's columns, a line a token's 8 columns -- see docs/internals/carry/carry_coefs.md#tile
template <class S>
__device__ __forceinline__ std::uint64_t coef_step(ops::SmemAddr tile, int step) {
  //: FIXED by the layout: a brick's tokens 8 to 15 follow its tokens 0 to 7, and a brick is 16 lines
  constexpr std::uint32_t kLater = kStepTokens / 2, kBrickLines = kStepTokens;
  return ops::step_operand(tile + static_cast<ops::SmemAddr>(step * kCoefStepBytes<S>), kLater,
                           kBrickLines);
}

}  // namespace detail
}  // namespace carry
}  // namespace fsla
