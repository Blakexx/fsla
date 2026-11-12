// measure/lab/experiments/fold_stream.cu -- QUESTION: a box every token writes sparsely (4 of its 256 leaves) is
// folded from one pass over the tokens with no list and no look back; what does a token cost a CTA, and how much
// of the producer's work hides under the fold's? METHOD: a CTA an SM: one producer warp and one fold warpgroup
// holding the box's state, 256 leaves x 64 channels, as 32 accumulators of 8 leaves (m64n8k16). The producer walks
// the tokens in order, 8 at a time into registers by one coalesced load a token (128 bytes), and for each of a
// token's 4 leaves stores the value into the leaf's group's tile at the group's fill count and the token's row of
// 8 coefficients beside it -- both operands M|N-major, so a token's value and its row are contiguous stores. A
// chunk of 128 tokens on, it publishes the fill counts; the warpgroup sweeps the groups, a step each with no
// branch -- on its tile if it filled, else on a tile of zeros -- one chunk behind the producer. `producer`: no
// step runs. `fold`: no token is produced, the steps run on the tiles as they are. `together`: both, the state
// checked exactly. The tokens come from a table the cache holds (65,536 tokens, 8 MB) or does not (1,048,576,
// 128 MB); every CTA reads the same table.
// see docs/internals/tools/lab.md#fold-stream
#include "lab.cuh"

namespace fold_stream {

constexpr int kLeaves = 256, kWide = 8, kGroups = kLeaves / kWide, kChannels = 64, kLive = 4, kChunk = 128,
              kBatch = 8, kStep = 16, kTiles = 2;
constexpr int kFoldThreads = 128, kThreads = kFoldThreads + 32;
//: A VALUE TILE, M|N-major: a token's 8 channels of a brick are one 16-byte line, a brick's 16 tokens 16 lines,
//: bricks 17 lines apart -- so a warp's 32 lanes, 4 a brick, store a token on 32 banks. A coefficient tile: a
//: token's 8 coefficients one line
constexpr int kBrickLines = 17, kBrick = kBrickLines * 16, kValueTile = (kChannels / 8) * kBrick,
              kCoefTile = kStep * 16, kTile = kValueTile + kCoefTile, kGroupBytes = kTiles * kTile;
//: after the groups' tiles: a tile of zeros (a group with no full tile runs its step on it), the fill counts, the
//: published counts of two chunks, the two flags
constexpr int kZeroAt = kGroups * kGroupBytes, kFillAt = kZeroAt + kTile, kPubAt = kFillAt + 4 * kGroups,
              kFlagsAt = kPubAt + 8 * kGroups, kSmemBytes = kFlagsAt + 16;
enum Form : int { kProducer = 0, kFold, kTogether, kForms };
const char* kFormName[kForms] = {"the producer alone", "the fold alone", "the producer and the fold together"};

//: token `t`'s `j`-th leaf (a chunk gives every group exactly 16 arrivals: a step), its coefficient, its value at
//: channel `c`
constexpr int leaf_of(int t, int j) { return (5 * t + 3 * j) % 64 + 64 * j; }
constexpr int coef_of(int t, int j) { return (t + j) % 3 + 1; }
constexpr int value_of(int t, int c) { return (t + c) % 5 + 1; }

__device__ __forceinline__ uint32_t bf16_of(int v) { return __float_as_uint(static_cast<float>(v)) >> 16; }

inline uint32_t bf16_host(int v) {
  const float f = static_cast<float>(v);
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u >> 16;
}

template <int FORM>
__global__ void __launch_bounds__(kThreads, 1)
    fold(long long* times, int* sinks, const uint32_t* values, float* out, int chunks) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t base = lab::smem(smem);
  uint32_t* const fill = reinterpret_cast<uint32_t*>(smem + kFillAt);
  uint32_t* const pub = reinterpret_cast<uint32_t*>(smem + kPubAt);
  volatile uint32_t* const produced = reinterpret_cast<volatile uint32_t*>(smem + kFlagsAt);
  volatile uint32_t* const consumed = produced + 1;
#pragma unroll 1
  for (int i = tid; i < kSmemBytes / 4; i += kThreads) reinterpret_cast<uint32_t*>(smem)[i] = 0u;
  lab::bulk_fence();
  __syncthreads();

  lab::WarpSpan clock;
  clock.start();
  if (warp == kFoldThreads / 32) {
    //: THE PRODUCER: one pass over the tokens, a chunk at most one ahead of the fold
    uint32_t seen = 0;
#pragma unroll 1
    for (int c = 0; c < chunks; ++c) {
#pragma unroll 1
      while (*consumed + 2u <= static_cast<uint32_t>(c)) {}
      if constexpr (FORM == kFold) {
        //: no token: the counts alone, a chunk's arrivals a group
        fill[lane] = static_cast<uint32_t>(c + 1) * (kChunk * kLive / kGroups);
      } else {
#pragma unroll 1
        for (int b = 0; b < kChunk / kBatch; ++b) {
          const int t0 = c * kChunk + b * kBatch;
          uint32_t v[kBatch];
#pragma unroll
          for (int i = 0; i < kBatch; ++i) v[i] = values[static_cast<long long>(t0 + i) * 32 + lane];
#pragma unroll
          for (int i = 0; i < kBatch; ++i) {
#pragma unroll
            for (int j = 0; j < kLive; ++j) {
              const int t = t0 + i, leaf = leaf_of(t, j), g = leaf / kWide, col = leaf % kWide;
              //: the group's next slot: its tile of the two, its token of the step
              const uint32_t k = fill[g];
              if (lane == 0) fill[g] = k + 1u;
              uint8_t* const tile = smem + g * kGroupBytes + ((k / kStep) % kTiles) * kTile;
              const uint32_t line = (k % kStep) * 16;
              *reinterpret_cast<uint32_t*>(tile + (lane >> 2) * kBrick + line + (lane & 3) * 4) = v[i];
              if (lane < 4)
                *reinterpret_cast<uint32_t*>(tile + kValueTile + line + lane * 4)
                    = lane == col / 2 ? bf16_of(coef_of(t, j)) << (16 * (col % 2)) : 0u;
            }
          }
        }
      }
      //: THE CHUNK, published: the counts, then the edge the fold's steps read the tiles behind
      pub[(c & 1) * kGroups + lane] = fill[lane];
      seen ^= fill[lane];
      lab::bulk_fence();
      if (lane == 0) *produced = static_cast<uint32_t>(c + 1);
    }
    clock.stop(times);
    lab::sink(sinks, seen);
  } else {
    //: THE FOLD: the state, 8 leaves x 64 channels an accumulator; the tokens each group has taken
    float d[kGroups][4] = {};
    uint32_t taken[kGroups] = {};
#pragma unroll 1
    for (int c = 0; c < chunks; ++c) {
#pragma unroll 1
      while (*produced <= static_cast<uint32_t>(c)) {}
      if constexpr (FORM != kProducer) {
        lab::wg_fence();
        //: THE SWEEP, every group a step and no branch: its tile that filled, or the tile of zeros
#pragma unroll
        for (int g = 0; g < kGroups; ++g) {
          const bool full = taken[g] + kStep <= pub[(c & 1) * kGroups + g];
          const uint32_t at = base + (full ? g * kGroupBytes + ((taken[g] / kStep) % kTiles) * kTile : kZeroAt);
          taken[g] += full ? kStep : 0;
          lab::wg_bf16_n8_ss_mn(d[g], lab::wg_desc(at, 8, kBrickLines), lab::wg_desc(at + kValueTile, 8, 16));
        }
        lab::wg_commit();
        lab::wg_wait<0>();
      }
      if (tid == 0) *consumed = static_cast<uint32_t>(c + 1);
    }
#pragma unroll
    for (int g = 0; g < kGroups; ++g)
#pragma unroll
      for (int i = 0; i < 4; ++i) lab::wg_hold(d[g][i]);
    clock.stop(times);
    float* mine = out + (static_cast<long long>(blockIdx.x) * kFoldThreads + tid) * (4 * kGroups);
#pragma unroll
    for (int g = 0; g < kGroups; ++g)
#pragma unroll
      for (int i = 0; i < 4; ++i) mine[4 * g + i] = d[g][i];
  }
}

}  // namespace fold_stream

template <int FORM>
void run(lab::Bench& b, const uint32_t* values, float* out, int tokens) {
  using namespace fold_stream;
  constexpr int kCheckTokens = 4096;
  const std::string name = std::string(kFormName[FORM])
                           + (FORM == kFold ? "" : ", a table of " + std::to_string(tokens) + " tokens");
  const auto kernel = fold<FORM>;
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  if (FORM == kTogether) {
    kernel<<<b.sms, kThreads, kSmemBytes>>>(b.times, b.sinks, values, out, kCheckTokens / kChunk);
    LAB_CHECK(cudaDeviceSynchronize());
    std::vector<double> want(static_cast<size_t>(kLeaves) * kChannels, 0.0);
    for (int t = 0; t < kCheckTokens; ++t)
      for (int j = 0; j < kLive; ++j)
        for (int c = 0; c < kChannels; ++c)
          want[static_cast<size_t>(leaf_of(t, j)) * kChannels + c] += double(value_of(t, c)) * coef_of(t, j);
    const size_t per = static_cast<size_t>(kFoldThreads) * 4 * kGroups;
    std::vector<float> got(static_cast<size_t>(b.sms) * per);
    LAB_CHECK(cudaMemcpy(got.data(), out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    long long wrong = 0;
    for (int blk = 0; blk < b.sms; ++blk)
      for (int tid = 0; tid < kFoldThreads; ++tid) {
        //: the accumulator's layout CUTLASS documents: a thread's register `i` of group `g`
        const int rows = 16 * (tid >> 5) + (tid & 31) / 4, pair = tid & 3;
        for (int g = 0; g < kGroups; ++g)
          for (int i = 0; i < 4; ++i) {
            const int m = rows + 8 * ((i >> 1) & 1), leaf = kWide * g + 2 * pair + (i & 1);
            wrong += got[blk * per + static_cast<size_t>(tid) * 4 * kGroups + 4 * g + i]
                     != static_cast<float>(want[static_cast<size_t>(leaf) * kChannels + m]);
          }
      }
    b.check(name, wrong == 0,
            std::to_string(wrong) + " of " + std::to_string(got.size())
                + " state values differ from the exact ones after " + std::to_string(kCheckTokens) + " tokens");
  } else {
    b.unchecked(name, FORM == kProducer ? "an isolation: the tiles are written and no step reads them"
                                        : "an isolation: the steps run on tiles no token was written to");
  }
  const lab::Samples cycles = b.variant(
      [&] { kernel<<<b.sms, kThreads, kSmemBytes>>>(b.times, b.sinks, values, out, tokens / kChunk); },
      [&] { return b.span(kThreads >> 5) / tokens; });
  b.record(name, "cycles a token a CTA", cycles);
}

int main(int argc, char** argv) {
  using namespace fold_stream;
  lab::Bench b(argc, argv);
  for (int f = 0; f < kForms; ++f) {
    const std::string k = lab::kernel("fold", {f});
    b.expect(k, "HGMMA", "==", f == kProducer ? 0 : kGroups);
    b.exempt(k, "a stream probe: the producer's stores beside 8-wide steps, no kernel's step");
  }
  if (b.expect_only) return 0;
  constexpr int kSmall = 65536, kLarge = 1048576;
  std::vector<uint32_t> table(static_cast<size_t>(kLarge) * 32);
  for (size_t w = 0; w < table.size(); ++w) {
    const int t = static_cast<int>(w / 32), c = 2 * static_cast<int>(w % 32);
    table[w] = bf16_host(value_of(t, c)) | bf16_host(value_of(t, c + 1)) << 16;
  }
  uint32_t* values = nullptr;
  float* out = nullptr;
  LAB_CHECK(cudaMalloc(&values, table.size() * sizeof(uint32_t)));
  LAB_CHECK(cudaMemcpy(values, table.data(), table.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMalloc(&out, sizeof(float) * b.sms * kFoldThreads * 4 * kGroups));
  run<kFold>(b, values, out, kSmall);
  for (int tokens : {kSmall, kLarge}) {
    run<kProducer>(b, values, out, tokens);
    run<kTogether>(b, values, out, tokens);
  }
  return 0;
}
