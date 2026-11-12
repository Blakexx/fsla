// measure/lab/experiments/pool_fill.cu -- QUESTION: a 16 x 16 box's fold takes, for each of its 16 lists, the
// listed tokens' values and their coefficients on the box's 16 dense leaves; what does an entry cost a CTA when
// producer warps fill those two pools from SCATTERED rows of global memory -- a value's 128 bytes and the token's
// 16 dense bytes, each by the warps' own load -- and the fold's steps consume them? METHOD: a CTA an SM: a fold
// warpgroup holding the box's state (16 accumulators of 16 leaves x 64 channels) and 2, 4 or 8 producer warps. A
// window is the box's 16 lists of 16 entries, in one of two slots of tiles. A warp's scattered load costs
// less the more rows it keeps in flight (`load_rate`), so a producer takes the entries of its residue in batches of
// 32: 8 value instructions of FOUR entries, 8 lanes an entry -- a lane loads 16 bytes of its entry's value and
// stores them as one line of the list's value tile (M|N-major; a row's 8 lanes cover the 32 banks once) -- and
// ONE instruction of all 32 entries' dense bytes, a lane an entry's 16, which it multiplies by the entry's byte,
// rounds to bf16 and stores as the two lines of the list's coefficient tile, each store covering the banks
// evenly. The fold, a window behind, runs the window's 16 steps in one commit group. `producers`: no step runs.
// `together`: both, the state checked exactly after two windows. An entry's token is drawn; the list it comes
// from is not loaded.
// TIMELINE: a producer's trip is two windows, a window its batches of 32 entries, batches alternating two
// register halves: a batch's stores and multiplies (its words loaded two batches before), then those registers'
// loads for the batch two on (64 rows in flight a warp); a window on, its fence and arrival. The fold's trip is a window:
// the wait on the producers, the fence, 16 steps, the commit, its wait, the arrival that frees the tiles. Live: a
// producer's 72 words; the fold's 128 accumulator registers.
// see docs/internals/tools/lab.md#pool-fill
#include "lab.cuh"

namespace pool_fill {

constexpr int kLists = 16, kLeaves = 16, kEntries = 16, kWindow = kLists * kEntries, kBatch = 32;
constexpr int kFoldThreads = 128, kTableTokens = 65536, kRowWords = 32;
//: a tile, M|N-major: a token's 8 rows (or columns) of a brick one 16-byte line, bricks 17 lines apart
constexpr int kBrick = 17 * 16, kValueTile = 8 * kBrick, kCoefTile = 2 * kBrick, kTile = kValueTile + kCoefTile;
constexpr int kBarsAt = 2 * kLists * kTile, kSmemBytes = kBarsAt + 32;
//: which of a coefficient's two bricks a lane stores FIRST, by its entry's list: a producer's lines of a list
//: fall on 32 / WARPS banks and a list is 8 banks on from the one before, so the lists that share banks take
//: opposite bricks (a brick is 4 banks on) and each store instruction covers the 32 banks evenly
template <int WARPS>
__device__ __forceinline__ int first_brick(int list) {
  return list >> (WARPS == 8 ? 2 : WARPS == 4 ? 1 : 0) & 1;
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
    //: A PRODUCER: the entries of its residue, a window at most one ahead of the fold. A batch is 32 entries: 8
    //: value instructions of FOUR entries (8 lanes an entry, a lane 16 bytes of its value: a line of a brick), and
    //: ONE instruction of all 32 entries' dense bytes (a lane an entry's 16)
    const int mine = warp - kFoldThreads / 32, group = lane >> 3;
    constexpr int kBatches = kWindow / WARPS / kBatch, kQuads = kBatch / 4;
    const uint4* const wide = reinterpret_cast<const uint4*>(values);
    const uint4* const boxes = reinterpret_cast<const uint4*>(dense);
    uint4 value[2][kQuads], bytes[2];
    const auto load = [&](int h, int batch) {
#pragma unroll
      for (int q = 0; q < kQuads; ++q) {
        const uint32_t t = token_of(blockIdx.x, mine + WARPS * (batch * kBatch + 4 * q + group));
        value[h][q] = wide[static_cast<long long>(t) * (kRowWords / 4) + (lane & 7)];
      }
      bytes[h] = boxes[token_of(blockIdx.x, mine + WARPS * (batch * kBatch + lane))];
    };
    load(0, 0);
    load(1, 1);
#pragma unroll 1
    for (int w = 0; w < windows; w += 2) {
#pragma unroll
      for (int slot = 0; slot < 2; ++slot) {
        if (w + slot >= 2) lab::mbar_wait(freed + 8 * slot, (((w + slot) >> 1) - 1) & 1);
        uint8_t* const tiles = smem + slot * kLists * kTile;
#pragma unroll
        for (int b = 0; b < kBatches; ++b) {
          //: the batch's registers: batches alternate the two halves, whatever a window holds of them
          const int h = (slot * kBatches + b) & 1;
#pragma unroll
          for (int q = 0; q < kQuads; ++q) {
            //: entry `e` of the window: list `e / 16`, token `e % 16` of its step
            const int e = mine + WARPS * (b * kBatch + 4 * q + group);
            *reinterpret_cast<uint4*>(tiles + (e / kEntries) * kTile + (lane & 7) * kBrick + (e % kEntries) * 16) =
                value[h][q];
          }
          //: the lane's entry: its 16 bytes times the entry's, 8 pairs of bf16, a brick's line each four
          const int e = mine + WARPS * (b * kBatch + lane);
          const float weight = static_cast<float>(sparse_of(token_of(blockIdx.x, (w + slot) * kWindow + e)));
          const uint32_t word[4] = {bytes[h].x, bytes[h].y, bytes[h].z, bytes[h].w};
          uint4 line[2];
#pragma unroll
          for (int k = 0; k < 2; ++k) {
            uint32_t pair[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const uint32_t two = word[2 * k + j / 2] >> (16 * (j & 1));
              pair[j] = lab::bf16x2_round(weight * static_cast<float>(two & 255u),
                                          weight * static_cast<float>(two >> 8 & 255u));
            }
            line[k] = make_uint4(pair[0], pair[1], pair[2], pair[3]);
          }
          const int first = first_brick<WARPS>(e / kEntries);
          uint8_t* const at = tiles + (e / kEntries) * kTile + kValueTile + (e % kEntries) * 16;
          *reinterpret_cast<uint4*>(at + first * kBrick) = first ? line[1] : line[0];
          *reinterpret_cast<uint4*>(at + (first ^ 1) * kBrick) = first ? line[0] : line[1];
          load(h, (w + slot) * kBatches + b + 2);
        }
        //: the writers' fence: the steps read these stores through the tensor pipe's own view of shared memory
        lab::bulk_fence();
        __syncwarp();
        if (lane == 0) lab::mbar_arrive(ready + 8 * slot);
      }
    }
    clock.stop(times);
  } else {
    //: THE FOLD: the box's state, a list an accumulator; a window's 16 steps one commit group
    float d[kLists][8] = {};
#pragma unroll 1
    for (int w = 0; w < windows; ++w) {
      lab::mbar_wait(ready + 8 * (w & 1), (w >> 1) & 1);
      if constexpr (FOLDS) {
        const uint32_t tiles = lab::held(base + (w & 1) * kLists * kTile);
        lab::wg_fence();
#pragma unroll
        for (int j = 0; j < kLists; ++j)
          lab::wg_bf16_n16_ss_mn(d[j], lab::wg_desc(tiles + j * kTile, 8, 17),
                                 lab::wg_desc(tiles + j * kTile + kValueTile, 8, 17));
        lab::wg_commit();
        lab::wg_wait<0>();
      }
      if (tid == 0) lab::mbar_arrive(freed + 8 * (w & 1));
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
  constexpr int kWindows = 64, kThreads = kFoldThreads + 32 * WARPS;
  const std::string name = std::string(FOLDS ? "the producers and the fold" : "the producers alone") + ", "
                           + std::to_string(WARPS) + " producers";
  const auto kernel = fill<FOLDS, WARPS>;
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  if (FOLDS) {
    kernel<<<b.sms, kThreads, kSmemBytes>>>(b.times, values, dense, out, 2);
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
            for (int w = 0; w < 2; ++w)
              for (int k = 0; k < kEntries; ++k) {
              const int t = static_cast<int>(token_of(blk, w * kWindow + a * kEntries + k));
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
      [&] { return b.span(kThreads >> 5) / (static_cast<double>(kWindows) * kWindow); });
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
  for (int warps : {2, 4, 8})
    for (int folds = 0; folds < 2; ++folds) {
      const std::string k = lab::kernel("fill", {folds, warps});
      b.expect(k, "HGMMA", "==", folds ? kLists : 0);
      b.expect(k, "LDG.E.128", ">=", 2 * (kBatch / 4 + 1));
      b.expect(k, "STS.128", ">=", 2 * (kBatch / 4 + 2));
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
  run<0, 2>(b, values, dense, out);
  run<1, 2>(b, values, dense, out);
  run<0, 4>(b, values, dense, out);
  run<1, 4>(b, values, dense, out);
  run<0, 8>(b, values, dense, out);
  run<1, 8>(b, values, dense, out);
  return 0;
}
