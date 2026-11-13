// measure/lab/experiments/intra_chain.cu -- QUESTION: what does the intra's chain cost a tile pair on hopper, alone
// -- 64 readers by 64 writers of one window: each level's pair products as INTEGER steps of the rows' bytes, the
// levels multiplied in fp32, the causal mask, the product rounded to bf16 in the registers it lies in as the A
// operand of the value steps -- and which of its parts is the cost? METHOD: one warpgroup a CTA, a CTA an SM. A
// trip is a tile pair: for each of two levels, S integer steps (`m64n64k32`, u8 x u8 into s32; S = a level's
// digits over 32) into the level's accumulator, one commit group; the two accumulators converted and multiplied,
// masked where the row asks, rounded to bf16 pairs -- 16 words a thread, a writer group of 16 its 4; then 4 value
// steps (`m64n64k16`, A those registers, B a tile of values) into the output's accumulator. Every operand tile is
// constant (bytes of 1, values of 1.0), so every accumulator is checked exactly. Rows take the chain apart: the
// integer steps alone, the arithmetic between alone, the value steps with the arithmetic and no integer step; and one row converts the products on
// the conversion unit where the others make them on the float pipe.
// TIMELINE: no load: the tiles are written once before the clock. A trip: the fence, 2 S integer steps on held
// descriptors, the commit, the wait; 64 products made floats (two adds and a multiply-add each), 32 multiplies,
// the mask's selects and 16 rounds a thread; the
// fence, 4 value steps, the commit, the wait. Live: two integer accumulators of 32 through the arithmetic, the 16
// words of A through the value steps' wait, the output's 32 for the call.
// see docs/internals/tools/lab.md#intra-chain
#include "lab.cuh"

namespace intra_chain {

constexpr int kThreads = 128;
//: the tiles: a level's readers' bytes and writers' bytes (its steps' operands, a level its own tiles), and the
//: values
constexpr int kBytesA = 0, kBytesB = 8192, kLevel = 4096, kValues = 16384, kSmem = kValues + 4096;
//: a row's parts
//: `kXu` keeps the conversion unit: a product converted by a cast (`I2FP.F32.S32`) where the others' is made on the float pipe
enum Parts : int { kGram = 1, kMath = 2, kValue = 4, kMask = 8, kXu = 16 };

//: a level's product as fp32, exact, with no conversion unit: an integer under 2^24 in two halves of 12 bits, each
//: the low bits of a float at 2^23, joined by one multiply-add
__device__ __forceinline__ float product_float(int x) {
  const uint32_t u = static_cast<uint32_t>(x);
  const float hi = __uint_as_float(u >> 12 | 0x4b000000u) - 8388608.f;
  const float lo = __uint_as_float((u & 0xfffu) | 0x4b000000u) - 8388608.f;
  return fmaf(hi, 4096.f, lo);
}

//: A TILE PAIR a trip: `S` integer steps a level, the arithmetic, the value steps
template <int PARTS, int S>
__global__ void __launch_bounds__(kThreads, 1) chain(long long* times, float* out, int* grams, int trips) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, q = (tid & 31) >> 2, t = tid & 3;
#pragma unroll 1
  for (int i = tid; i < kValues / 4; i += kThreads) reinterpret_cast<uint32_t*>(smem)[i] = 0x01010101u;
#pragma unroll 1
  for (int i = tid; i < (kSmem - kValues) / 2; i += kThreads) reinterpret_cast<uint16_t*>(smem + kValues)[i] = 0x3f80;
  lab::bulk_fence();
  __syncthreads();
  const uint32_t base = lab::smem(smem);
  const uint64_t ra[2] = {lab::held(lab::wg_desc(base + kBytesA, 8, 16)), lab::held(lab::wg_desc(base + kBytesA + kLevel, 8, 16))};
  const uint64_t rb[2] = {lab::held(lab::wg_desc(base + kBytesB, 8, 16)), lab::held(lab::wg_desc(base + kBytesB + kLevel, 8, 16))};
  const uint64_t values = lab::held(lab::wg_desc(base + kValues, 8, 16));
  //: THE MASK, a thread's: bit `4 j + 2 r + e` set where writer `8 j + 2 t + e` is before reader `16 warp + q + 8 r`
  uint32_t before = 0;
#pragma unroll
  for (int i = 0; i < 32; ++i)
    before |= static_cast<uint32_t>(8 * (i >> 2) + 2 * t + (i & 1) < 16 * warp + q + 8 * ((i >> 1) & 1)) << i;
  before = lab::held(before);
  int g[2][32];
  float o[32];
  if constexpr (!(PARTS & kGram)) {
    //: no integer step writes the levels' accumulators: a level's product is what `S` steps of ones give
#pragma unroll
    for (int l = 0; l < 2; ++l)
#pragma unroll
      for (int i = 0; i < 32; ++i) g[l][i] = static_cast<int>(lab::held(static_cast<uint32_t>(32 * S)));
  }
  if constexpr (!(PARTS & kValue)) {
#pragma unroll
    for (int i = 0; i < 32; ++i) o[i] = 0.f;
  }
  uint32_t kept = 0;

  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int trip = 0; trip < trips; ++trip) {
    if constexpr (PARTS & kGram) {
      //: THE LEVELS' PAIR PRODUCTS: a level's first step writes its accumulator whole
      lab::wg_fence();
#pragma unroll
      for (int l = 0; l < 2; ++l)
#pragma unroll
        for (int s = 0; s < S; ++s) lab::wg_u8_n64_ss(g[l], ra[l], rb[l], s != 0);
      lab::wg_commit();
      lab::wg_wait<0>();
    }
    if constexpr (PARTS & kMath) {
      //: THE PAIR'S WEIGHT: the levels' products multiplied in fp32, nothing where the writer is not before the
      //: reader, a bf16 pair a word in the registers a value step reads as A -- writer group `c`'s words `4 c` on
      uint32_t a[16];
#pragma unroll
      for (int w = 0; w < 16; ++w) {
        float pair[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int i = 2 * w + e;
          const float weight = PARTS & kXu ? static_cast<float>(g[0][i]) * static_cast<float>(g[1][i])
                                           : product_float(g[0][i]) * product_float(g[1][i]);
          if constexpr (PARTS & kMask) pair[e] = before >> i & 1u ? weight : 0.f;
          else pair[e] = weight;
        }
        a[w] = lab::held(lab::bf16x2_round(pair[0], pair[1]));
      }
      if constexpr (PARTS & kValue) {
        //: THE VALUE STEPS, a writer group of 16 each: the call's first writes the output's accumulator whole
        lab::wg_fence();
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const uint32_t rows[4] = {a[4 * c], a[4 * c + 1], a[4 * c + 2], a[4 * c + 3]};
          lab::wg_bf16_n64_rs(o, rows, values, trip != 0 || c != 0);
        }
        lab::wg_commit();
        lab::wg_wait<0>();
      } else {
#pragma unroll
        for (int w = 0; w < 16; ++w) kept ^= a[w];
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 32; ++i) lab::wg_hold(o[i]);
#pragma unroll
  for (int l = 0; l < 2; ++l)
#pragma unroll
    for (int i = 0; i < 32; ++i) lab::wg_hold(g[l][i]);
  clock.stop(times);
  const long long at = static_cast<long long>(blockIdx.x) * kThreads + tid;
#pragma unroll
  for (int i = 0; i < 32; ++i) out[at * 32 + i] = o[i];
  //: a thread's levels' products, their largest and smallest, and the words of A a row without value steps kept
  int lo = g[0][0], hi = g[0][0];
#pragma unroll
  for (int l = 0; l < 2; ++l)
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      lo = min(lo, g[l][i]);
      hi = max(hi, g[l][i]);
    }
  grams[at * 3] = lo, grams[at * 3 + 1] = hi, grams[at * 3 + 2] = static_cast<int>(kept);
}

}  // namespace intra_chain

template <int PARTS, int S>
void run(lab::Bench& b, float* out, int* grams) {
  using namespace intra_chain;
  constexpr int kCheckTrips = 16, kTrips = 4096;
  const std::string name =
      std::string(PARTS == (kGram | kMath | kValue | kMask | kXu) ? "the chain, the causal mask on, the products by the conversion unit"
                  : PARTS == (kGram | kMath | kValue | kMask) ? "the chain, the causal mask on"
                  : PARTS == (kGram | kMath | kValue) ? "the chain, no mask"
                  : PARTS == kGram ? "the integer steps alone"
                  : PARTS == (kMath | kMask) ? "the arithmetic alone, the mask on"
                  : "the value steps and the arithmetic, no integer step")
      + ", " + std::to_string(32 * S) + " digits a level";
  const auto kernel = chain<PARTS, S>;
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  kernel<<<b.sms, kThreads, kSmem>>>(b.times, out, grams, kCheckTrips);
  LAB_CHECK(cudaDeviceSynchronize());
  const size_t threads = static_cast<size_t>(b.sms) * kThreads;
  std::vector<float> got(threads * 32);
  std::vector<int> got_grams(threads * 3);
  LAB_CHECK(cudaMemcpy(got.data(), out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
  LAB_CHECK(cudaMemcpy(got_grams.data(), grams, got_grams.size() * sizeof(int), cudaMemcpyDeviceToHost));
  long long wrong = 0, wrong_grams = 0;
  //: a level's product is 32 a step; a pair's weight its square; a reader's sum the weight a kept writer, a trip
  const float weight = static_cast<float>(32 * S) * static_cast<float>(32 * S);
  for (size_t th = 0; th < threads; ++th) {
    const int tid = static_cast<int>(th % kThreads), warp = tid >> 5, q = (tid & 31) >> 2;
    wrong_grams += got_grams[th * 3] != 32 * S || got_grams[th * 3 + 1] != 32 * S;
    for (int i = 0; i < 32; ++i) {
      const int reader = 16 * warp + q + 8 * ((i >> 1) & 1);
      const float want = PARTS & kValue ? weight * static_cast<float>(PARTS & kMask ? reader : 64) * kCheckTrips : 0.f;
      wrong += got[th * 32 + i] != want;
    }
  }
  b.check(name, wrong == 0 && wrong_grams == 0,
          std::to_string(wrong) + " of " + std::to_string(got.size()) + " sums and " + std::to_string(wrong_grams)
              + " of " + std::to_string(threads) + " threads' products differ");
  const lab::Samples cycles = b.variant([&] { kernel<<<b.sms, kThreads, kSmem>>>(b.times, out, grams, kTrips); },
                                        [&] { return b.span(kThreads >> 5) / kTrips; });
  b.record(name, "cycles a tile pair", cycles);
}

int main(int argc, char** argv) {
  using namespace intra_chain;
  lab::Bench b(argc, argv);
  for (int s : {2, 8})
    for (int parts : {kGram | kMath | kValue | kMask, kGram | kMath | kValue, static_cast<int>(kGram), kMath | kMask,
                      kMath | kValue, kGram | kMath | kValue | kMask | kXu}) {
      const std::string k = lab::kernel("chain", {parts, s});
      //: the row that keeps the conversion unit in its trip, on purpose
      if (parts & kXu) b.expect(k, "STEP_XU", ">=", 0);
      b.expect(k, "IGMMA", "==", parts & kGram ? 2 * s : 0);
      b.expect(k, "HGMMA", "==", parts & kValue ? 4 : 0);
      b.exempt(k, "a chain probe: the intra's steps on constant tiles, no kernel's schedule");
    }
  if (b.expect_only) return 0;
  float* out = nullptr;
  int* grams = nullptr;
  LAB_CHECK(cudaMalloc(&out, sizeof(float) * b.sms * kThreads * 32));
  LAB_CHECK(cudaMalloc(&grams, sizeof(int) * b.sms * kThreads * 3));
  for (int pass = 0; pass < 2; ++pass) {
    if (pass == 0) {
      run<kGram | kMath | kValue | kMask, 8>(b, out, grams);
      run<kGram | kMath | kValue, 8>(b, out, grams);
      run<kGram, 8>(b, out, grams);
      run<kMath | kMask, 8>(b, out, grams);
      run<kMath | kValue, 8>(b, out, grams);
      run<kGram | kMath | kValue | kMask | kXu, 8>(b, out, grams);
    } else {
      run<kGram | kMath | kValue | kMask, 2>(b, out, grams);
      run<kGram | kMath | kValue, 2>(b, out, grams);
      run<kGram, 2>(b, out, grams);
      run<kMath | kMask, 2>(b, out, grams);
      run<kMath | kValue, 2>(b, out, grams);
      run<kGram | kMath | kValue | kMask | kXu, 2>(b, out, grams);
    }
  }
  return 0;
}
