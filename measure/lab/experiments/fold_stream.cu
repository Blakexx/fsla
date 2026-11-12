// measure/lab/experiments/fold_stream.cu -- QUESTION: a box every token writes sparsely (4 of its 256 leaves) is
// folded from one pass over the tokens with no list and no look back; what does a token cost a CTA by the step's
// width -- 64 leaves a step in four groups, or all 256 in one -- and with one producer warp or four? METHOD: a CTA
// an SM: producer warps and one fold warpgroup holding the box's state, 256 leaves x 64 channels, as accumulators
// of WIDE leaves. Both operands are M|N-major, bricks 17 lines apart: a token's value is one 4-byte store a lane on
// 32 banks. WIDE 64, a producer a group: it walks every token and loads every value (128 bytes, coalesced), and
// where a token's leaves fall in its group stores the value at its own running count and the token's row of 64
// coefficients beside it, one store each. WIDE 256, one group: a token's slot is its index, so there is no count;
// a producer takes the tokens of its residue, stores the value, clears the 4 words the slot's last token set and
// sets its own 4. A producer holds two batches of 8 tokens' values, the next loaded before this one is stored. A
// chunk of 64 tokens on, the producers arrive at a barrier; the fold, a chunk behind, runs the
// chunk's 4 steps a group in one commit group and frees the tiles by another barrier. `producer`: no step runs.
// `fold`: no token is produced. `together`: both, the state checked exactly. The tokens come from a table of
// 65,536 (8 MB) or of 1,048,576 (128 MB) that every CTA reads in step.
// see docs/internals/tools/lab.md#fold-stream
#include "lab.cuh"

namespace fold_stream {

constexpr int kLeaves = 256, kChannels = 64, kLive = 4, kChunk = 64, kStep = 16, kRing = 2 * kChunk / kStep;
constexpr int kFoldThreads = 128;
//: A TILE, M|N-major: a token's 8 rows (or columns) of a brick are one 16-byte line, a brick's 16 tokens 16 lines,
//: bricks 17 lines apart
constexpr int kBrickLines = 17, kBrick = kBrickLines * 16, kValueTile = (kChannels / 8) * kBrick;
template <int WIDE>
constexpr int kGroups = kLeaves / WIDE;
template <int WIDE>
constexpr int kTile = kValueTile + (WIDE / 8) * kBrick;
template <int WIDE>
constexpr int kGroupBytes = kRing * kTile<WIDE>;
//: after the groups' rings: a scratch word a lane (a store with nothing to say lands there), four barriers
template <int WIDE>
constexpr int kScratchAt = kGroups<WIDE> * kGroupBytes<WIDE>;
template <int WIDE>
constexpr int kSmemBytes = kScratchAt<WIDE> + 128 + 32;
enum Form : int { kProducer = 0, kFold, kTogether, kForms };
const char* kFormName[kForms] = {"the producers alone", "the fold alone", "the producers and the fold together"};

//: token `t`'s `j`-th leaf (one in each quarter of the box: a group of 64 takes one a token), its coefficient, its
//: value at channel `c`
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

//: the word of a coefficient row (two columns) that holds `leaf`, and the coefficient in its half
__device__ __forceinline__ uint32_t coef_word(int leaf, int coef) { return bf16_of(coef) << (16 * (leaf & 1)); }

template <int FORM, int WIDE, int WARPS>
__global__ void __launch_bounds__(kFoldThreads + 32 * WARPS, 1)
    fold(long long* times, int* sinks, const uint32_t* values, float* out, int chunks) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int kGroups = fold_stream::kGroups<WIDE>, kTile = fold_stream::kTile<WIDE>;
  constexpr int kGroupBytes = fold_stream::kGroupBytes<WIDE>, kScratchAt = fold_stream::kScratchAt<WIDE>;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t base = lab::smem(smem);
  //: THE BARRIERS, a pair by the chunk's parity: `ready`, the producers' arrivals; `freed`, the fold's
  const uint32_t ready = base + kScratchAt + 128, freed = ready + 16;
#pragma unroll 1
  for (int i = tid; i < kScratchAt / 4 + 32; i += blockDim.x) reinterpret_cast<uint32_t*>(smem)[i] = 0u;
  if (tid == 0) {
    lab::mbar_init(ready, WARPS);
    lab::mbar_init(ready + 8, WARPS);
    lab::mbar_init(freed, 1);
    lab::mbar_init(freed + 8, 1);
  }
  lab::bulk_fence();
  __syncthreads();

  lab::WarpSpan clock;
  clock.start();
  if (warp >= kFoldThreads / 32) {
    //: A PRODUCER: one pass over the tokens, a chunk at most one ahead of the fold. Its tokens are every `kStride`-th
    //: from `mine` (all 256 leaves a step: its residue) or every one (a group a producer); it holds two batches of
    //: their values, the next loaded before this one is stored
    const int mine = warp - kFoldThreads / 32;
    constexpr int kStride = WIDE == 256 ? WARPS : 1, kBatch = 8, kBatches = kChunk / kStride / kBatch;
    const int first = WIDE == 256 ? mine : 0, last = chunks * kChunk - 1;
    uint32_t* const scratch = reinterpret_cast<uint32_t*>(smem + kScratchAt) + lane;
    uint32_t count = 0;
    uint32_t now[kBatch], next[kBatch];
#pragma unroll
    for (int i = 0; i < kBatch; ++i)
      next[i] = values[static_cast<long long>(min(first + kStride * i, last)) * 32 + lane];
#pragma unroll 1
    for (int c = 0; c < chunks; ++c) {
      if (c >= 2) lab::mbar_wait(freed + 8 * (c & 1), ((c >> 1) - 1) & 1);
      if constexpr (FORM != kFold) {
#pragma unroll 1
        for (int b = 0; b < kBatches; ++b) {
          const int t0 = first + kStride * kBatch * (c * kBatches + b);
#pragma unroll
          for (int i = 0; i < kBatch; ++i) {
            now[i] = next[i];
            next[i] = values[static_cast<long long>(min(t0 + kStride * (kBatch + i), last)) * 32 + lane];
          }
#pragma unroll
          for (int i = 0; i < kBatch; ++i) {
            const int t = t0 + kStride * i;
            if constexpr (WIDE == 256) {
              //: ONE GROUP: token `t`'s slot is `t`
              const uint32_t slot = static_cast<uint32_t>(t) % (kRing * kStep);
              uint8_t* const tile = smem + (slot / kStep) * kTile;
              const uint32_t line = (slot % kStep) * 16;
              *reinterpret_cast<uint32_t*>(tile + (lane >> 2) * kBrick + line + (lane & 3) * 4) = now[i];
              //: its row: lane `j` clears the word the slot's last token set for its leaf `j`, then sets its own
              const int j = lane & 3, is = leaf_of(t, j), was = leaf_of(max(t - kRing * kStep, 0), j);
              const bool sets = lane < kLive, clears = sets && t >= kRing * kStep;
              uint8_t* const row = tile + kValueTile + line;
              uint32_t* const clear = reinterpret_cast<uint32_t*>(row + (was >> 3) * kBrick + ((was & 7) >> 1) * 4);
              uint32_t* const set = reinterpret_cast<uint32_t*>(row + (is >> 3) * kBrick + ((is & 7) >> 1) * 4);
              *(clears ? clear : scratch) = 0u;
              *(sets ? set : scratch) = coef_word(is, coef_of(t, j));
            } else {
              //: A GROUP A PRODUCER: a token whose leaves fall here is taken at the running count
              uint32_t word = 0;
              bool hit = false;
#pragma unroll
              for (int j = 0; j < kLive; ++j) {
                const int leaf = leaf_of(t, j), col = leaf % WIDE;
                const bool here = leaf / WIDE == mine;
                hit |= here;
                word |= here && lane == (col >> 1) ? coef_word(leaf, coef_of(t, j)) : 0u;
              }
              if (hit) {
                const uint32_t slot = count % (kRing * kStep);
                ++count;
                uint8_t* const tile = smem + mine * kGroupBytes + (slot / kStep) * kTile;
                const uint32_t at = (lane >> 2) * kBrick + (slot % kStep) * 16 + (lane & 3) * 4;
                *reinterpret_cast<uint32_t*>(tile + at) = now[i];
                *reinterpret_cast<uint32_t*>(tile + kValueTile + at) = word;
              }
            }
          }
        }
      }
      //: THE CHUNK, handed over: every lane's stores behind the edge the steps read them past, then the arrival
      lab::bulk_fence();
      __syncwarp();
      if (lane == 0) lab::mbar_arrive(ready + 8 * (c & 1));
    }
    clock.stop(times);
    lab::sink(sinks, count);
  } else {
    //: THE FOLD: the state, WIDE leaves x 64 channels an accumulator
    float d[kGroups][WIDE / 2] = {};
#pragma unroll 1
    for (int c = 0; c < chunks; ++c) {
      lab::mbar_wait(ready + 8 * (c & 1), (c >> 1) & 1);
      if constexpr (FORM != kProducer) {
        lab::wg_fence();
#pragma unroll
        for (int g = 0; g < kGroups; ++g) {
#pragma unroll
          for (int s = 0; s < kChunk / kStep; ++s) {
            const uint32_t at = base + g * kGroupBytes + ((c & 1) * (kChunk / kStep) + s) * kTile;
            const uint64_t a = lab::wg_desc(at, 8, kBrickLines), b = lab::wg_desc(at + kValueTile, 8, kBrickLines);
            if constexpr (WIDE == 256) lab::wg_bf16_n256_ss_mn(d[g], a, b);
            else lab::wg_bf16_n64_ss_mn(d[g], a, b);
          }
        }
        lab::wg_commit();
        lab::wg_wait<0>();
      }
      if (tid == 0) lab::mbar_arrive(freed + 8 * (c & 1));
    }
#pragma unroll
    for (int g = 0; g < kGroups; ++g)
#pragma unroll
      for (int i = 0; i < WIDE / 2; ++i) lab::wg_hold(d[g][i]);
    clock.stop(times);
    float* mine = out + (static_cast<long long>(blockIdx.x) * kFoldThreads + tid) * (kLeaves / 2);
#pragma unroll
    for (int g = 0; g < kGroups; ++g)
#pragma unroll
      for (int i = 0; i < WIDE / 2; ++i) mine[(WIDE / 2) * g + i] = d[g][i];
  }
}

}  // namespace fold_stream

template <int FORM, int WIDE, int WARPS>
void run(lab::Bench& b, const uint32_t* values, float* out, int tokens) {
  using namespace fold_stream;
  constexpr int kCheckTokens = 4096, kThreads = kFoldThreads + 32 * WARPS, kBytes = kSmemBytes<WIDE>;
  const std::string name = std::string(kFormName[FORM]) + ", steps of " + std::to_string(WIDE) + " leaves, "
                           + std::to_string(WARPS) + (WARPS == 1 ? " producer" : " producers")
                           + (FORM == kFold ? "" : ", a table of " + std::to_string(tokens) + " tokens");
  const auto kernel = fold<FORM, WIDE, WARPS>;
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kBytes));
  if (FORM == kTogether) {
    kernel<<<b.sms, kThreads, kBytes>>>(b.times, b.sinks, values, out, kCheckTokens / kChunk);
    LAB_CHECK(cudaDeviceSynchronize());
    std::vector<double> want(static_cast<size_t>(kLeaves) * kChannels, 0.0);
    for (int t = 0; t < kCheckTokens; ++t)
      for (int j = 0; j < kLive; ++j)
        for (int c = 0; c < kChannels; ++c)
          want[static_cast<size_t>(leaf_of(t, j)) * kChannels + c] += double(value_of(t, c)) * coef_of(t, j);
    const size_t per = static_cast<size_t>(kFoldThreads) * (kLeaves / 2);
    std::vector<float> got(static_cast<size_t>(b.sms) * per);
    LAB_CHECK(cudaMemcpy(got.data(), out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    long long wrong = 0;
    for (int blk = 0; blk < b.sms; ++blk)
      for (int tid = 0; tid < kFoldThreads; ++tid) {
        //: the accumulator's layout CUTLASS documents: a thread's register `i` of group `g`
        const int rows = 16 * (tid >> 5) + (tid & 31) / 4, pair = tid & 3;
        for (int g = 0; g < kGroups<WIDE>; ++g)
          for (int i = 0; i < WIDE / 2; ++i) {
            const int m = rows + 8 * ((i >> 1) & 1), leaf = WIDE * g + 8 * (i >> 2) + 2 * pair + (i & 1);
            wrong += got[blk * per + static_cast<size_t>(tid) * (kLeaves / 2) + (WIDE / 2) * g + i]
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
      [&] { kernel<<<b.sms, kThreads, kBytes>>>(b.times, b.sinks, values, out, tokens / kChunk); },
      [&] { return b.span(kThreads >> 5) / tokens; });
  b.record(name, "cycles a token a CTA", cycles);
}

template <int WIDE, int WARPS>
void forms(lab::Bench& b, const uint32_t* values, float* out) {
  using namespace fold_stream;
  constexpr int kSmall = 65536, kLarge = 1048576;
  run<kFold, WIDE, WARPS>(b, values, out, kSmall);
  run<kProducer, WIDE, WARPS>(b, values, out, kSmall);
  run<kTogether, WIDE, WARPS>(b, values, out, kSmall);
  run<kTogether, WIDE, WARPS>(b, values, out, kLarge);
}

template <int WIDE, int WARPS>
void expect(lab::Bench& b) {
  using namespace fold_stream;
  for (int f = 0; f < kForms; ++f) {
    const std::string k = lab::kernel("fold", {f, WIDE, WARPS});
    b.expect(k, "HGMMA", "==", f == kProducer ? 0 : kGroups<WIDE> * (kChunk / kStep));
    b.exempt(k, "a stream probe: the producers' stores beside the fold's steps, no kernel's step");
  }
}

int main(int argc, char** argv) {
  using namespace fold_stream;
  lab::Bench b(argc, argv);
  expect<64, 4>(b);
  expect<256, 1>(b);
  expect<256, 4>(b);
  if (b.expect_only) return 0;
  std::vector<uint32_t> table(static_cast<size_t>(1048576) * 32);
  for (size_t w = 0; w < table.size(); ++w) {
    const int t = static_cast<int>(w / 32), c = 2 * static_cast<int>(w % 32);
    table[w] = bf16_host(value_of(t, c)) | bf16_host(value_of(t, c + 1)) << 16;
  }
  uint32_t* values = nullptr;
  float* out = nullptr;
  LAB_CHECK(cudaMalloc(&values, table.size() * sizeof(uint32_t)));
  LAB_CHECK(cudaMemcpy(values, table.data(), table.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMalloc(&out, sizeof(float) * b.sms * kFoldThreads * (kLeaves / 2)));
  forms<64, 4>(b, values, out);
  forms<256, 1>(b, values, out);
  forms<256, 4>(b, values, out);
  return 0;
}
