// measure/lab/experiments/state_image.cu -- QUESTION: where a box's two sides are sparse on different levels, the
// fold keeps its state in registers along one axis (16 accumulators of 16 leaves) and the readout reads it along
// the other; what does it cost a window to store the state to shared memory as 16 tiles the readout's steps take
// as their A operand -- a value a 2-byte store, or a pair a 4-byte store after one shuffle -- and to read them?
// METHOD: a CTA an SM, two warpgroups, each a box of 16 x 16 leaves x 64 channels in registers (fp32, an m64n16
// accumulator a fold digit). A window: every thread stores its 128 values, truncated to bf16, into the tile of
// each value's read digit, M|N-major (a line is one leaf's 8 channels), tiles 137 lines apart and bricks 17 so a
// warp's lanes fall on distinct banks; `halves`: a value a 2-byte store; `pairs`: a lane takes its neighbour's
// value of the same line by one shuffle and stores 4 bytes. `store`: the image alone, by either. `store and
// read` (by halves; the image is the same): then 16 m64n16k16 steps, A a tile of the image, B a coefficient tile,
// into one accumulator, checked exactly.
// TIMELINE: no load from global memory. A warpgroup's trip is a window: its 128 stores a thread (each from a
// register the fold holds; a pair's shuffle the instruction before its store), the edge the steps read the image
// behind, then one commit group of 16 steps and its wait. Live: the state, 128 registers a thread, and the
// readout's 8.
// see docs/internals/tools/lab.md#state-image
#include "lab.cuh"

namespace state_image {

constexpr int kDigits = 16, kGroupThreads = 128, kThreads = 2 * kGroupThreads;
//: AN IMAGE TILE, M|N-major: a line is one leaf's 8 channels of a brick; a brick's 16 leaves 16 lines, bricks 17
//: lines apart, tiles 137: a warp's lanes -- 4 tiles, 2 bricks, 4 words a line -- on 32 banks
constexpr int kBrickLines = 17, kTileLines = 137, kTileBytes = kTileLines * 16, kBoxBytes = kDigits * kTileBytes;
//: a coefficient tile, K-major: 16 tokens' columns of 16 leaves, a brick 8 lines of 16 bytes, its second half 8 on
constexpr int kCoefBytes = 512, kCoefAt = 2 * kBoxBytes, kSmemBytes = kCoefAt + kDigits * kCoefBytes;
enum Bits : int { kPairs = 1, kRead = 2 };

//: a box's state at (channel, fold digit, read digit) and a coefficient at (read digit, leaf, token): small
//: integers, exact in bf16 and in every sum
constexpr int state_of(int box, int m, int a, int b) { return (m + 3 * a + 5 * b + box) % 7 + 1; }
constexpr int coef_of(int b, int a, int n) { return (a + 2 * n + 3 * b) % 3 + 1; }
constexpr int coef_at(int n, int k) { return 16 * ((n % 8) + 16 * (n / 8) + 8 * (k / 8)) + 2 * (k % 8); }

__device__ __forceinline__ uint16_t bf16_of(int v) {
  return static_cast<uint16_t>(__float_as_uint(static_cast<float>(v)) >> 16);
}

template <int FORM>
__global__ void __launch_bounds__(kThreads, 1) image(long long* times, float* out, int windows) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, box = tid >> 7, warp = (tid >> 5) & 3, lane = tid & 31;
  //: a thread's rows are `16 warp + q` and that + 8; its columns `8 h + 2 t + c`
  const int q = lane >> 2, t = lane & 3;
  const uint32_t base = lab::smem(smem);
#pragma unroll 1
  for (int i = tid; i < kDigits * kDigits * kDigits; i += kThreads) {
    const int b = i / (kDigits * kDigits), a = (i / kDigits) % kDigits, n = i % kDigits;
    *reinterpret_cast<uint16_t*>(smem + kCoefAt + b * kCoefBytes + coef_at(n, a)) = bf16_of(coef_of(b, a, n));
  }
  //: THE STATE, as the fold holds it: `s[a][c | r << 1 | h << 2]`
  float s[kDigits][8];
#pragma unroll
  for (int a = 0; a < kDigits; ++a)
#pragma unroll
    for (int i = 0; i < 8; ++i)
      s[a][i] = static_cast<float>(state_of(box, 16 * warp + q + 8 * ((i >> 1) & 1), a, 8 * (i >> 2) + 2 * t + (i & 1)));
  float d[8] = {};
  //: A THREAD'S BASE into the image, held: every store is this plus a constant of (a, h, c, r)
  const uint32_t part = FORM & kPairs ? 16 * ((2 * warp + (q & 1)) * kBrickLines) + 4 * (q >> 1)
                                      : 16 * (2 * warp * kBrickLines) + 2 * q;
  uint8_t* const mine = smem + lab::held(static_cast<uint32_t>(box * kBoxBytes + 2 * t * kTileBytes) + part);
  lab::bulk_fence();
  __syncthreads();

  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int w = 0; w < windows; ++w) {
    //: THE IMAGE: value (channel m, fold digit a) of read digit b to tile b, line `(m / 8) 17 + a`, slot `m % 8`
#pragma unroll
    for (int a = 0; a < kDigits; ++a) {
#pragma unroll
      for (int hc = 0; hc < 4; ++hc) {
        const int h = hc >> 1, c = hc & 1;
        uint8_t* const tile = mine + (8 * h + c) * kTileBytes + 16 * a;
        if constexpr (FORM & kPairs) {
          //: a lane of even `q` stores row 0's pair of slots (q, q + 1), one of odd `q` row 1's (q - 1, q): each
          //: gives its neighbour the row that neighbour stores
          const bool odd = q & 1;
          const float own = odd ? s[a][c | 2 | h << 2] : s[a][c | h << 2];
          const float got = __shfl_xor_sync(0xffffffffu, odd ? s[a][c | h << 2] : s[a][c | 2 | h << 2], 4);
          const uint32_t word = lab::bf16x2_trunc(odd ? got : own, odd ? own : got);
          *reinterpret_cast<uint32_t*>(tile) = word;
        } else {
#pragma unroll
          for (int r = 0; r < 2; ++r)
            *reinterpret_cast<uint16_t*>(tile + 16 * r * kBrickLines)
                = static_cast<uint16_t>(__float_as_uint(s[a][c | r << 1 | h << 2]) >> 16);
        }
      }
    }
    lab::bulk_fence();
    __syncthreads();
    if constexpr (FORM & kRead) {
      //: THE READOUT: a step a read digit, A its tile of the image
      //: the tiles' addresses derived from one held base a window: 32 descriptors are not kept in registers
      const uint32_t image_at = lab::held(base + box * kBoxBytes), coef_at_ = lab::held(base + kCoefAt);
      lab::wg_fence();
      //: a rolled loop: its two descriptors are the step's only registers beside the state's 128
#pragma unroll 1
      for (int b = 0; b < kDigits; ++b)
        lab::wg_bf16_n16_ss_mna(d, lab::wg_desc(image_at + b * kTileBytes, 8, kBrickLines),
                                lab::wg_desc(coef_at_ + b * kCoefBytes, 8, 16));
      lab::wg_commit();
      lab::wg_wait<0>();
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) lab::wg_hold(d[i]);
  clock.stop(times);
  float* const to = out + (static_cast<long long>(blockIdx.x) * kThreads + tid) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) to[i] = d[i];
  //: the state is live to the end
  float live = 0.0f;
#pragma unroll
  for (int a = 0; a < kDigits; ++a)
#pragma unroll
    for (int i = 0; i < 8; ++i) live += s[a][i];
  to[0] += 0.0f * live;
}

}  // namespace state_image

template <int FORM>
void run(lab::Bench& b, float* out) {
  using namespace state_image;
  constexpr int kCheckWindows = 4, kWindows = 512;
  const std::string name = std::string(FORM & kPairs ? "a pair a 4-byte store after a shuffle" : "a value a 2-byte store")
                           + (FORM & kRead ? ", and the 16 steps that read it" : ", the image alone");
  const auto kernel = image<FORM>;
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  if (FORM & kRead) {
    kernel<<<b.sms, kThreads, kSmemBytes>>>(b.times, out, kCheckWindows);
    LAB_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(static_cast<size_t>(b.sms) * kThreads * 8);
    LAB_CHECK(cudaMemcpy(got.data(), out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    long long wrong = 0;
    for (int blk = 0; blk < b.sms; ++blk)
      for (int tid = 0; tid < kThreads; ++tid) {
        const int box = tid >> 7, rows = 16 * ((tid >> 5) & 3) + (tid & 31) / 4, pair = tid & 3;
        for (int i = 0; i < 8; ++i) {
          const int m = rows + 8 * ((i >> 1) & 1), n = 8 * (i >> 2) + 2 * pair + (i & 1);
          long long want = 0;
          for (int bb = 0; bb < kDigits; ++bb)
            for (int a = 0; a < kDigits; ++a) want += state_of(box, m, a, bb) * coef_of(bb, a, n);
          wrong += got[(static_cast<size_t>(blk) * kThreads + tid) * 8 + i] != static_cast<float>(want * kCheckWindows);
        }
      }
    b.check(name, wrong == 0,
            std::to_string(wrong) + " of " + std::to_string(got.size()) + " readout values differ from the exact ones");
  } else {
    b.unchecked(name, "an isolation: the image is stored and no step reads it");
  }
  const lab::Samples cycles = b.variant([&] { kernel<<<b.sms, kThreads, kSmemBytes>>>(b.times, out, kWindows); },
                                        [&] { return b.span(kThreads >> 5) / kWindows; });
  b.record(name, "cycles a window a CTA of two boxes", cycles);
}

int main(int argc, char** argv) {
  using namespace state_image;
  lab::Bench b(argc, argv);
  for (int form = 0; form < 3; ++form) {
    const std::string k = lab::kernel("image", {form});
    b.expect(k, "HGMMA", form & kRead ? ">=" : "==", form & kRead ? 1 : 0);
    b.expect(k, "SHFL", "==", form & kPairs ? 64 : 0);
    b.exempt(k, "an image probe: the state's stores beside the readout's steps, no kernel's step");
  }
  if (b.expect_only) return 0;
  float* out = nullptr;
  LAB_CHECK(cudaMalloc(&out, sizeof(float) * b.sms * kThreads * 8));
  run<0>(b, out);
  run<kPairs>(b, out);
  run<kRead>(b, out);
  return 0;
}
