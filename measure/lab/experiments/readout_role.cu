// measure/lab/experiments/readout_role.cu -- QUESTION: a CTA's two boxes share their read digits and each has left
// its state as an image in shared memory; what does a token cost when ONE readout warpgroup reads both images into
// one accumulator, stores the token's partial to a panel, and drain warps issue it as one line of f16 pairs?
// METHOD: a CTA an SM: a readout warpgroup and 1, 2 or 4 drain warps; the two boxes' images (16 tiles each, M|N-
// major) and a coefficient tile a read digit are written once. A chunk is 4 read digits, 16 tokens each: 8
// m64n16k16 steps in one commit group -- a digit's step on box 0's tile and on box 1's into the same accumulator --
// then the 64 tokens' fp32 rows stored to the panel of the chunk's parity (rows 272 bytes apart: a warp's lanes on
// 32 banks). The drain warps, a chunk behind, take the tokens of their residue: a row loaded, rounded to f16 pairs
// and added as one line into the token's output row. Every output row is checked exactly.
// TIMELINE: the readout's trip is a chunk: the wait on the drain's barrier two chunks back, the fence, 8 steps on
// tiles nothing writes, the commit and its wait, 32 stores a thread, the arrival. A drain warp's trip is two
// batches of 8 tokens at their own registers: a batch's rounds and global adds, then those registers' loads for
// the batch two on (a chunk's first two as it opens; with four drain warps none runs ahead). Live: the readout's
// 32 accumulator registers; a drain warp's 32.
// see docs/internals/tools/lab.md#readout-role
#include "lab.cuh"

namespace readout_role {

constexpr int kDigits = 16, kChannels = 64, kReadThreads = 128, kChunkDigits = 4, kTokens = 16 * kChunkDigits,
              kBatch = 8;
constexpr int kBrickLines = 17, kTileBytes = 137 * 16, kBoxBytes = kDigits * kTileBytes, kCoefBytes = 512;
//: a panel row: a token's 64 fp32 and a pad of 16 bytes, so rows two apart sit 8 banks apart
constexpr int kRowBytes = 272, kPanelBytes = kTokens * kRowBytes;
constexpr int kCoefAt = 2 * kBoxBytes, kPanelAt = kCoefAt + kDigits * kCoefBytes, kBarsAt = kPanelAt + 2 * kPanelBytes,
              kSmemBytes = kBarsAt + 32;

constexpr int state_of(int box, int m, int a, int b) { return (m + 3 * a + 5 * b + box) % 7 + 1; }
constexpr int coef_of(int b, int a, int n) { return (a + 2 * n + 3 * b) % 3 + 1; }
constexpr int coef_at(int n, int k) { return 16 * ((n % 8) + 16 * (n / 8) + 8 * (k / 8)) + 2 * (k % 8); }

__device__ __forceinline__ uint16_t bf16_of(int v) {
  return static_cast<uint16_t>(__float_as_uint(static_cast<float>(v)) >> 16);
}

template <int DRAINS>
__global__ void __launch_bounds__(kReadThreads + 32 * DRAINS, 1)
    readout(long long* times, uint32_t* out, int chunks, int rows_apart) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t base = lab::smem(smem), ready = base + kBarsAt, freed = ready + 16;
  //: THE IMAGES AND THE COEFFICIENTS, written once: value (channel m, leaf a) of read digit b at tile b, line
  //: `(m / 8) 17 + a`, slot `m % 8`
#pragma unroll 1
  for (int i = tid; i < 2 * kDigits * kDigits * kChannels; i += blockDim.x) {
    const int box = i / (kDigits * kDigits * kChannels), b = (i / (kDigits * kChannels)) % kDigits,
              a = (i / kChannels) % kDigits, m = i % kChannels;
    uint8_t* const tile = smem + box * kBoxBytes + b * kTileBytes;
    *reinterpret_cast<uint16_t*>(tile + 16 * ((m / 8) * kBrickLines + a) + 2 * (m % 8)) = bf16_of(state_of(box, m, a, b));
  }
#pragma unroll 1
  for (int i = tid; i < kDigits * kDigits * kDigits; i += blockDim.x) {
    const int b = i / (kDigits * kDigits), a = (i / kDigits) % kDigits, n = i % kDigits;
    *reinterpret_cast<uint16_t*>(smem + kCoefAt + b * kCoefBytes + coef_at(n, a)) = bf16_of(coef_of(b, a, n));
  }
  if (tid == 0) {
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      lab::mbar_init(ready + 8 * p, kReadThreads / 32);
      lab::mbar_init(freed + 8 * p, DRAINS);
    }
  }
  lab::bulk_fence();
  __syncthreads();

  lab::WarpSpan clock;
  clock.start();
  if (warp < kReadThreads / 32) {
    //: THE READOUT: a thread's rows are `16 warp + q` and that + 8, its tokens of a digit `8 h + 2 t + c`
    const int q = lane >> 2, t = lane & 3;
    const uint32_t mine = lab::held(static_cast<uint32_t>(kPanelAt + 2 * t * kRowBytes + (16 * warp + q) * 4));
    //: a digit's first step writes its accumulator whole: nothing is cleared between chunks
    float d[kChunkDigits][8];
#pragma unroll 1
    for (int c = 0; c < chunks; ++c) {
      if (c >= 2) lab::mbar_wait(freed + 8 * (c & 1), ((c >> 1) - 1) & 1);
      const uint32_t digit = static_cast<uint32_t>(c % (kDigits / kChunkDigits)) * kChunkDigits;
      const uint32_t images = lab::held(base + digit * kTileBytes);
      const uint32_t coefs = lab::held(base + kCoefAt + digit * kCoefBytes);
      lab::wg_fence();
#pragma unroll
      for (int j = 0; j < kChunkDigits; ++j)
#pragma unroll
        for (int box = 0; box < 2; ++box)
          lab::wg_bf16_n16_ss_mna(d[j], lab::wg_desc(images + box * kBoxBytes + j * kTileBytes, 8, kBrickLines),
                                  lab::wg_desc(coefs + j * kCoefBytes, 8, 16), box != 0);
      lab::wg_commit();
      lab::wg_wait<0>();
      //: THE PANEL: value (channel, token) to the token's row, a store a register
      uint8_t* const row = smem + mine + (c & 1) * kPanelBytes;
#pragma unroll
      for (int j = 0; j < kChunkDigits; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i)
          *reinterpret_cast<float*>(row + (16 * j + 8 * (i >> 2) + (i & 1)) * kRowBytes + 32 * ((i >> 1) & 1))
              = d[j][i];
      __syncwarp();
      if (lane == 0) lab::mbar_arrive(ready + 8 * (c & 1));
    }
  } else {
    //: A DRAIN WARP: the tokens of its residue; a lane holds a token's two channels
    const int mine = warp - kReadThreads / 32;
    constexpr int kBatches = kTokens / DRAINS / kBatch;
    static_assert(kBatches % 2 == 0);
    float lo[2][kBatch], hi[2][kBatch];
    const auto load = [&](int h, int c, int batch) {
#pragma unroll
      for (int i = 0; i < kBatch; ++i) {
        const float* const row = reinterpret_cast<const float*>(
            smem + kPanelAt + (c & 1) * kPanelBytes + (mine + DRAINS * (batch * kBatch + i)) * kRowBytes) + 2 * lane;
        lo[h][i] = row[0];
        hi[h][i] = row[1];
      }
    };
#pragma unroll 1
    for (int c = 0; c < chunks; ++c) {
      lab::mbar_wait(ready + 8 * (c & 1), (c >> 1) & 1);
      load(0, c, 0);
      load(1, c, 1);
#pragma unroll 1
      for (int b = 0; b < kBatches; b += 2) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
          for (int i = 0; i < kBatch; ++i) {
            const int token = mine + DRAINS * ((b + h) * kBatch + i);
            const long long at = static_cast<long long>(blockIdx.x) * (kDigits * 16)
                                 + (c % rows_apart) * kTokens + token;
            lab::red_add_f16x2(out + at * (kChannels / 2) + lane, lab::pack_half2(lo[h][i], hi[h][i]));
          }
          load(h, c, min(b + 2 + h, kBatches - 1));
        }
      }
      __syncwarp();
      if (lane == 0) lab::mbar_arrive(freed + 8 * (c & 1));
    }
  }
  clock.stop(times);
}

}  // namespace readout_role

inline float half_to_float(uint16_t h) {
  const int e = (h >> 10) & 31, m = h & 1023;
  const float mag = e == 0 ? std::ldexp(static_cast<float>(m), -24) : std::ldexp(static_cast<float>(m | 1024), e - 25);
  return h & 0x8000 ? -mag : mag;
}

template <int DRAINS>
void run(lab::Bench& b, uint32_t* out) {
  using namespace readout_role;
  constexpr int kWindow = kDigits / kChunkDigits, kChunks = 1024, kThreads = kReadThreads + 32 * DRAINS;
  const std::string name = "one readout warpgroup over two images, " + std::to_string(DRAINS)
                           + (DRAINS == 1 ? " drain warp" : " drain warps");
  const auto kernel = readout<DRAINS>;
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  const size_t words = static_cast<size_t>(b.sms) * kDigits * 16 * (kChannels / 2);
  LAB_CHECK(cudaMemset(out, 0, words * 4));
  //: THE CHECK: one window, every token's row its own
  kernel<<<b.sms, kThreads, kSmemBytes>>>(b.times, out, kWindow, kWindow);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> got(words);
  LAB_CHECK(cudaMemcpy(got.data(), out, words * 4, cudaMemcpyDeviceToHost));
  long long wrong = 0;
  for (size_t w = 0; w < words; ++w)
    for (int half = 0; half < 2; ++half) {
      const int m = 2 * static_cast<int>(w % (kChannels / 2)) + half;
      const int token = static_cast<int>(w / (kChannels / 2)) % (kDigits * 16), bb = token / 16, n = token % 16;
      long long want = 0;
      for (int box = 0; box < 2; ++box)
        for (int a = 0; a < kDigits; ++a) want += state_of(box, m, a, bb) * coef_of(bb, a, n);
      wrong += half_to_float(static_cast<uint16_t>(got[w] >> (16 * half))) != static_cast<float>(want);
    }
  b.check(name, wrong == 0, std::to_string(wrong) + " of " + std::to_string(2 * words) + " output values differ");
  LAB_CHECK(cudaMemset(out, 0, words * 4));
  const lab::Samples cycles = b.variant([&] { kernel<<<b.sms, kThreads, kSmemBytes>>>(b.times, out, kChunks, 1); },
                                        [&] { return b.span(kThreads >> 5) / kChunks / kTokens; });
  b.record(name, "cycles a token a CTA", cycles);
}

int main(int argc, char** argv) {
  using namespace readout_role;
  lab::Bench b(argc, argv);
  for (int drains : {1, 2, 4}) {
    const std::string k = lab::kernel("readout", {drains});
    b.expect(k, "HGMMA", "==", 2 * kChunkDigits);
    b.expect(k, "RED", ">=", 2 * kBatch);
    b.exempt(k, "a role probe: the readout's steps beside its panel and drain, no kernel's step");
    //: once a chunk: a role's `__syncwarp` before its one-lane arrival is a convergence call
    b.expect(k, "LOOP_CONVERGENCE", "<=", 8);
    //: the chunk's loop holds its steps and its barriers' polled waits: the waits' branches are the step's
    b.expect(k, "BSSY", ">=", 0);
  }
  if (b.expect_only) return 0;
  uint32_t* out = nullptr;
  LAB_CHECK(cudaMalloc(&out, static_cast<size_t>(b.sms) * kDigits * 16 * (kChannels / 2) * 4));
  run<1>(b, out);
  run<2>(b, out);
  run<4>(b, out);
  return 0;
}
