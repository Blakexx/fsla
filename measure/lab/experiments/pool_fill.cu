// measure/lab/experiments/pool_fill.cu -- QUESTION: a 16 x 16 box's fold takes, for each of its 16 lists, the
// listed tokens' values and their coefficients on the box's 16 dense leaves; what does an entry cost a CTA when
// producer warps fill those two pools from SCATTERED rows of global memory -- a value's 128 bytes and the token's
// 16 dense bytes, each by the warps' own load -- and the fold's steps consume them? METHOD: a CTA an SM: a fold
// warpgroup holding the box's state (16 accumulators of 16 leaves x 64 channels) and 1, 2 or 4 producer warps. A
// chunk is 4 lists of 16 entries. A producer takes the entries of its residue, FOUR an instruction, 8 lanes an
// entry (`load_rate`: a warp's load is bound by its latency, so by the rows in flight): a lane loads 16 bytes of
// its entry's value and stores them as one line of the list's value tile (M|N-major; a row's 8 lanes cover the 32
// banks once), and loads a word of its token's 16 dense bytes, multiplies by the entry's byte, rounds to bf16 and
// stores to the list's coefficient tile, an instruction's four entries on four bank groups. The fold, a chunk
// behind, runs the chunk's 4 steps in one commit group. `producers`: no step runs. `together`: both, the state
// checked exactly after one window. An entry's token is drawn; the list it comes from is not loaded.
// TIMELINE: a producer's trip is two chunks, a chunk its batches of 16 entries, batches alternating two register
// halves: a batch's multiplies and its stores (its words loaded two batches before), then those registers' loads
// for the batch two on (32 rows in flight a warp); a chunk on, its arrival. The fold's trip is a chunk: the wait on
// the producers, the fence, 4 steps, the commit, its wait, the arrival that frees the tiles. Live: a producer's
// 40 words; the fold's 128 accumulator registers.
// see docs/internals/tools/lab.md#pool-fill
#include "lab.cuh"

namespace pool_fill {

constexpr int kLists = 16, kLeaves = 16, kEntries = 16, kChunkLists = 4, kChunk = kChunkLists * kEntries, kBatch = 16;
constexpr int kFoldThreads = 128, kTableTokens = 65536, kRowWords = 32;
//: a tile, M|N-major: a token's 8 rows (or columns) of a brick one 16-byte line, bricks 17 lines apart
constexpr int kBrick = 17 * 16, kValueTile = 8 * kBrick, kCoefTile = 2 * kBrick, kTile = kValueTile + kCoefTile;
constexpr int kBarsAt = 2 * kChunkLists * kTile, kSmemBytes = kBarsAt + 32;
//: the entry of a batch of 16 that lane group `g` of instruction `q` takes: an instruction's four entries store
//: their coefficients to four different bank groups (a tile is 8 banks on from the one before, a token's line 4)
template <int WARPS>
__device__ __forceinline__ int quad_entry(int g, int q) {
  if constexpr (WARPS == 1) return 2 * g + (q & 1) + 8 * (q >> 1);
  if constexpr (WARPS == 2) return g + 4 * (q & 1) + 8 * (q >> 1);
  return 4 * ((g >> 1) + 2 * (q >> 1)) + (g & 1) + 2 * (q & 1);
}

//: the token of a (CTA, list, entry of the window), its value at a channel, its byte on the list's digit and on a
//: dense leaf: small integers, every sum exact
constexpr uint32_t token_of(uint32_t block, uint32_t entry) {
  return ((block * 4099u + entry) * 2654435761u >> 12) & (kTableTokens - 1);
}
constexpr int value_of(int t, int c) { return (t + c) % 5 + 1; }
constexpr int sparse_of(int t) { return t % 2 + 1; }
constexpr int dense_of(int t, int b) { return (t + b) % 3 + 1; }

template <int FOLDS, int WARPS>
__global__ void __launch_bounds__(kFoldThreads + 32 * WARPS, 1)
    fill(long long* times, const uint32_t* values, const uint32_t* dense, float* out, int windows) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t base = lab::smem(smem), ready = base + kBarsAt, freed = ready + 16;
#pragma unroll 1
  for (int i = tid; i < kBarsAt / 4; i += blockDim.x) reinterpret_cast<uint32_t*>(smem)[i] = 0u;
  if (tid == 0) {
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      lab::mbar_init(ready + 8 * p, WARPS);
      lab::mbar_init(freed + 8 * p, 1);
    }
  }
  lab::bulk_fence();
  __syncthreads();

  lab::WarpSpan clock;
  clock.start();
  if (warp >= kFoldThreads / 32) {
    //: A PRODUCER: the entries of its residue, a chunk at most one ahead of the fold. An instruction carries FOUR
    //: entries, 8 lanes each: a lane 16 bytes of its entry's value (a line of a brick) and a word of its dense bytes
    const int mine = warp - kFoldThreads / 32, group = lane >> 3;
    constexpr int kBatches = kChunk / WARPS / kBatch, kQuads = kBatch / 4;
    const uint4* const wide = reinterpret_cast<const uint4*>(values);
    uint4 value[2][kQuads];
    uint32_t bytes[2][kQuads];
    const auto load = [&](int h, int batch) {
#pragma unroll
      for (int q = 0; q < kQuads; ++q) {
        const uint32_t t = token_of(blockIdx.x, mine + WARPS * (batch * kBatch + quad_entry<WARPS>(group, q)));
        value[h][q] = wide[static_cast<long long>(t) * (kRowWords / 4) + (lane & 7)];
        bytes[h][q] = dense[static_cast<long long>(t) * (kLeaves / 4) + (lane >> 1 & 3)];
      }
    };
    load(0, 0);
    load(1, 1);
    const int chunks = windows * (kLists / kChunkLists);
#pragma unroll 1
    for (int c = 0; c < chunks; c += 2) {
#pragma unroll
      for (int slot = 0; slot < 2; ++slot) {
        if (c + slot >= 2) lab::mbar_wait(freed + 8 * slot, (((c + slot) >> 1) - 1) & 1);
#pragma unroll
        for (int b = 0; b < kBatches; ++b) {
          //: the batch's registers: batches alternate the two halves, whatever a chunk holds of them
          const int h = (slot * kBatches + b) & 1;
#pragma unroll
          for (int q = 0; q < kQuads; ++q) {
            //: entry `e` of the chunk: list `e / 16` of its four, token `e % 16` of the step
            const int e = mine + WARPS * (b * kBatch + quad_entry<WARPS>(group, q));
            const uint32_t t = token_of(blockIdx.x, (c + slot) * kChunk + e);
            uint8_t* const tile = smem + (slot * kChunkLists + e / kEntries) * kTile;
            const uint32_t line = (e % kEntries) * 16;
            *reinterpret_cast<uint4*>(tile + (lane & 7) * kBrick + line) = value[h][q];
            //: lane `l` of its entry's 8 holds leaves `2 l` and `2 l + 1`: its two bytes times the entry's, as bf16
            const uint32_t pair = bytes[h][q] >> (16 * (lane & 1));
            const float weight = static_cast<float>(sparse_of(t));
            const uint32_t coef = lab::bf16x2_round(weight * static_cast<float>(pair & 255u),
                                                    weight * static_cast<float>(pair >> 8 & 255u));
            *reinterpret_cast<uint32_t*>(tile + kValueTile + (lane >> 2 & 1) * kBrick + line + (lane & 3) * 4) = coef;
          }
          load(h, (c + slot) * kBatches + b + 2);
        }
        lab::bulk_fence();
        __syncwarp();
        if (lane == 0) lab::mbar_arrive(ready + 8 * slot);
      }
    }
    clock.stop(times);
  } else {
    //: THE FOLD: the box's state, a list an accumulator
    float d[kLists][8] = {};
#pragma unroll 1
    for (int w = 0; w < windows; ++w) {
#pragma unroll
      for (int q = 0; q < kLists / kChunkLists; ++q) {
        const int c = w * (kLists / kChunkLists) + q;
        lab::mbar_wait(ready + 8 * (c & 1), (c >> 1) & 1);
        if constexpr (FOLDS) {
          const uint32_t tiles = lab::held(base + (c & 1) * kChunkLists * kTile);
          lab::wg_fence();
#pragma unroll
          for (int j = 0; j < kChunkLists; ++j)
            lab::wg_bf16_n16_ss_mn(d[q * kChunkLists + j], lab::wg_desc(tiles + j * kTile, 8, 17),
                                   lab::wg_desc(tiles + j * kTile + kValueTile, 8, 17));
          lab::wg_commit();
          lab::wg_wait<0>();
        }
        if (tid == 0) lab::mbar_arrive(freed + 8 * (c & 1));
      }
    }
#pragma unroll
    for (int a = 0; a < kLists; ++a)
#pragma unroll
      for (int i = 0; i < 8; ++i) lab::wg_hold(d[a][i]);
    clock.stop(times);
    float* const mine = out + (static_cast<long long>(blockIdx.x) * kFoldThreads + tid) * (kLists * 8);
#pragma unroll
    for (int a = 0; a < kLists; ++a)
#pragma unroll
      for (int i = 0; i < 8; ++i) mine[8 * a + i] = d[a][i];
  }
}

}  // namespace pool_fill

template <int FOLDS, int WARPS>
void run(lab::Bench& b, const uint32_t* values, const uint32_t* dense, float* out) {
  using namespace pool_fill;
  constexpr int kWindows = 256, kThreads = kFoldThreads + 32 * WARPS;
  const std::string name = std::string(FOLDS ? "the producers and the fold" : "the producers alone") + ", "
                           + std::to_string(WARPS) + (WARPS == 1 ? " producer" : " producers");
  const auto kernel = fill<FOLDS, WARPS>;
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  if (FOLDS) {
    kernel<<<b.sms, kThreads, kSmemBytes>>>(b.times, values, dense, out, 1);
    LAB_CHECK(cudaDeviceSynchronize());
    const size_t per = static_cast<size_t>(kFoldThreads) * kLists * 8;
    std::vector<float> got(static_cast<size_t>(b.sms) * per);
    LAB_CHECK(cudaMemcpy(got.data(), out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    long long wrong = 0;
    for (int blk = 0; blk < b.sms; ++blk)
      for (int tid = 0; tid < kFoldThreads; ++tid) {
        const int rows = 16 * (tid >> 5) + (tid & 31) / 4, pair = tid & 3;
        for (int a = 0; a < kLists; ++a)
          for (int i = 0; i < 8; ++i) {
            const int m = rows + 8 * ((i >> 1) & 1), leaf = 8 * (i >> 2) + 2 * pair + (i & 1);
            long long want = 0;
            for (int k = 0; k < kEntries; ++k) {
              const int t = static_cast<int>(token_of(blk, a * kEntries + k));
              want += value_of(t, m) * sparse_of(t) * dense_of(t, leaf);
            }
            wrong += got[blk * per + (static_cast<size_t>(tid) * kLists + a) * 8 + i] != static_cast<float>(want);
          }
      }
    b.check(name, wrong == 0, std::to_string(wrong) + " of " + std::to_string(got.size()) + " state values differ");
  } else {
    b.unchecked(name, "an isolation: the pools are filled and no step reads them");
  }
  const lab::Samples cycles = b.variant(
      [&] { kernel<<<b.sms, kThreads, kSmemBytes>>>(b.times, values, dense, out, kWindows); },
      [&] { return b.span(kThreads >> 5) / (static_cast<double>(kWindows) * kLists * kEntries); });
  b.record(name, "cycles an entry a CTA", cycles);
}

inline uint32_t bf16_host(int v) {
  const float f = static_cast<float>(v);
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u >> 16;
}

int main(int argc, char** argv) {
  using namespace pool_fill;
  lab::Bench b(argc, argv);
  for (int warps : {1, 2, 4})
    for (int folds = 0; folds < 2; ++folds) {
      const std::string k = lab::kernel("fill", {folds, warps});
      b.expect(k, "HGMMA", "==", folds ? kLists : 0);
      b.expect(k, "LDG", ">=", kBatch);
      b.exempt(k, "a fill probe: the producers' loads and stores beside the fold's steps, no kernel's step");
      b.expect(k, "LOOP_CONVERGENCE", "<=", 8);
      b.expect(k, "BSSY", ">=", 0);
    }
  if (b.expect_only) return 0;
  std::vector<uint32_t> host_values(static_cast<size_t>(kTableTokens) * kRowWords), host_dense(kTableTokens * (kLeaves / 4));
  for (size_t w = 0; w < host_values.size(); ++w) {
    const int t = static_cast<int>(w / kRowWords), c = 2 * static_cast<int>(w % kRowWords);
    host_values[w] = bf16_host(value_of(t, c)) | bf16_host(value_of(t, c + 1)) << 16;
  }
  for (size_t w = 0; w < host_dense.size(); ++w) {
    const int t = static_cast<int>(w / (kLeaves / 4)), b0 = 4 * static_cast<int>(w % (kLeaves / 4));
    host_dense[w] = dense_of(t, b0) | dense_of(t, b0 + 1) << 8 | dense_of(t, b0 + 2) << 16 | dense_of(t, b0 + 3) << 24;
  }
  uint32_t *values = nullptr, *dense = nullptr;
  float* out = nullptr;
  LAB_CHECK(cudaMalloc(&values, host_values.size() * 4));
  LAB_CHECK(cudaMemcpy(values, host_values.data(), host_values.size() * 4, cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMalloc(&dense, host_dense.size() * 4));
  LAB_CHECK(cudaMemcpy(dense, host_dense.data(), host_dense.size() * 4, cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMalloc(&out, sizeof(float) * b.sms * kFoldThreads * kLists * 8));
  run<0, 1>(b, values, dense, out);
  run<1, 1>(b, values, dense, out);
  run<0, 2>(b, values, dense, out);
  run<1, 2>(b, values, dense, out);
  run<0, 4>(b, values, dense, out);
  run<1, 4>(b, values, dense, out);
  return 0;
}
