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

//: WHERE A STEP'S COEFFICIENTS ARE READ, a position: token 0's line of the block its first digit lies in, on that digit; a tile of the level's 64 tokens in bytes -- see docs/internals/carry/carry_coefs.md#source
struct CoefSource {
  const std::uint8_t* at[kLevels];
  int tile_bytes[kLevels];
};

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
    s.at[at] = f.rows[at] + (block * kRowBlockBytes + in_block);
    s.tile_bytes[at] = f.tile_bytes[at];
  }
  return s;
}

//: THE WORDS A POSITION'S BYTES OF A TOKEN ARE LOADED AS: a block of 16 digits four, fewer digits one load
template <int B>
constexpr int kPositionWords = B >= kRowBlockBits ? 1 << (B - 2) : (B == 3 ? 2 : 1);

//: A POSITION'S BYTES of a token: `1 << B` digits from `p` on, a block of 16 a load -- see docs/internals/carry/carry_coefs.md#expand
template <int B>
__device__ __forceinline__ void position_words(const std::uint8_t* p,
                                               std::uint32_t (&out)[kPositionWords<B>]) {
  if constexpr (B >= kRowBlockBits) {
    constexpr int kBlocks = 1 << (B - kRowBlockBits);
#pragma unroll
    for (int k = 0; k < kBlocks; ++k) {
      const uint4 w = *reinterpret_cast<const uint4*>(p + k * kRowBlockBytes);
      out[4 * k] = w.x;
      out[4 * k + 1] = w.y;
      out[4 * k + 2] = w.z;
      out[4 * k + 3] = w.w;
    }
  } else if constexpr (B == 3) {
    const uint2 w = *reinterpret_cast<const uint2*>(p);
    out[0] = w.x;
    out[1] = w.y;
  } else if constexpr (B == 2) {
    out[0] = *reinterpret_cast<const std::uint32_t*>(p);
  } else if constexpr (B == 1) {
    out[0] = *reinterpret_cast<const std::uint16_t*>(p);
  } else {
    out[0] = *p;
  }
}

//: the same bytes as fp32, a digit each
template <int B>
__device__ __forceinline__ void position_floats(const std::uint32_t (&w)[kPositionWords<B>],
                                                float (&out)[1 << B]) {
  constexpr int kDigits = 1 << B;
#pragma unroll
  for (int i = 0; i < kDigits; ++i) {
    out[i] = static_cast<float>((w[i / 4] >> (8 * (i % 4))) & 0xffu);
  }
}

//: A ROUND'S BYTES AS A LANE LOADED THEM, a chunk: its token's words a position, and whether its place holds a token -- see docs/internals/carry/carry_coefs.md#expand
template <int B0, int B1, int B2, int B3>
struct CoefBytes {
  std::uint32_t w0[kPositionWords<B0>], w1[kPositionWords<B1>], w2[kPositionWords<B2>];
  std::uint32_t w3[kPositionWords<B3>];
  bool live;
};

//: THE LOADS OF A ROUND'S CHUNK: lane `i` takes place `i` of `tokens` and loads its token's bytes a position; `tiles` the window's tiles of rows, a power of two -- see docs/internals/carry/carry_coefs.md#expand
template <int LEVELS, int B0, int B1, int B2, int B3>
__device__ __forceinline__ CoefBytes<B0, B1, B2, B3> load_round(const CoefSource& c,
                                                                const std::uint16_t* tokens,
                                                                int tiles) {
  const int lane = static_cast<int>(threadIdx.x) & (ops::kBallotLanes - 1);
  const int listed = tokens[lane];
  CoefBytes<B0, B1, B2, B3> b{};
  b.live = listed != kNoToken;
  //: a place past a set's last round holds what an earlier set left: its token is kept inside the window, and nothing is made of it
  const int token = b.live ? listed : 0;
  const int line = (token & ((1 << kRowTileBits) - 1)) << kRowBlockBits;
  const int tile_of = (token >> kRowTileBits) & (tiles - 1);
  const std::uint8_t* from[kLevels] = {};
#pragma unroll
  for (int p = 0; p < LEVELS; ++p) {
    from[p] = c.at[p] + static_cast<std::uint32_t>(tile_of * c.tile_bytes[p] + line);
  }
  if constexpr (LEVELS > 0) position_words<B0>(from[0], b.w0);
  if constexpr (LEVELS > 1) position_words<B1>(from[1], b.w1);
  if constexpr (LEVELS > 2) position_words<B2>(from[2], b.w2);
  if constexpr (LEVELS > 3) position_words<B3>(from[3], b.w3);
  return b;
}

//: A ROUND'S COEFFICIENT LINES, a chunk: a lane multiplies its token's bytes out over the step's columns and stores its line; a place with no token stores zeros -- see docs/internals/carry/carry_coefs.md#expand
template <int LEVELS, int B0, int B1, int B2, int B3>
__device__ __forceinline__ void expand_round(const CoefBytes<B0, B1, B2, B3>& b,
                                             ops::SmemAddr tile) {
  using S = StepShape<B0, B1, B2, B3>;
  constexpr int kBits[kLevels] = {B0, B1, B2, B3};
  constexpr int kLow = 1 << B0, kHigh = 1 << S::kHighBits;
  const int lane = static_cast<int>(threadIdx.x) & (ops::kBallotLanes - 1);
  //: a position the call has no level for is 1
  float f0[1 << B0] = {1.0f}, f1[1 << B1] = {1.0f}, f2[1 << B2] = {1.0f}, f3[1 << B3] = {1.0f};
  if constexpr (LEVELS > 0) position_floats<B0>(b.w0, f0);
  if constexpr (LEVELS > 1) position_floats<B1>(b.w1, f1);
  if constexpr (LEVELS > 2) position_floats<B2>(b.w2, f2);
  if constexpr (LEVELS > 3) position_floats<B3>(b.w3, f3);

  //: a position of one digit is a factor of the whole line; a place with no token is 0
  const float one[kLevels] = {f0[0], f1[0], f2[0], f3[0]};
  float scale = b.live ? 1.0f : 0.0f;
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
