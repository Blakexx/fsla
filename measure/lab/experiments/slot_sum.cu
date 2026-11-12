// measure/lab/experiments/slot_sum.cu -- QUESTION: an atomic add to global memory is held by the SM at one line
// about every 10 to 12 cycles (`add_rate`). Were every CTA to STORE a token's partial line into a slot of its own
// and a later pass to sum a token's slots, what does a line cost stored to a slot nothing has touched, and what
// does a token cost the pass that sums its 32 slots in fp32, divides and writes the output? METHOD: a CTA an SM.
// STORE: 1 or 4 warps write 16,384 lines of f16 pairs from registers into the CTA's own 64 MB of global memory,
// each line once -- packed, or a line every 4 KB, as a slot of a token-major layout sits. SUM: 4, 8 or 16 warps;
// a token is 32 slots of 128 bytes; 8 lanes take a token, a lane 16 bytes of a slot an instruction -- 8 f16 turned
// to fp32 and added into the lane's 8 sums -- then the sums times the token's reciprocal, rounded to bf16 and
// stored, a 128-byte line a token. PACKED: a token's slots lie together. GATHERED: slot `s` lies in stream `s` of
// the CTA's 32, at a row scattered over the stream, as a partial a CTA copied out among its own would. 540 MB of
// slots, a CTA its own 4 MB. Every
// stored word and every output value is checked exactly.
// TIMELINE: no load in STORE: a warp's trip is 16 stores from one register to addresses constant off its base.
// A SUM warp's trip is four tokens, 8 lanes each, in four batches of 8 slots alternating two register halves: a
// batch's 64 conversions and adds a lane (its words loaded two batches before), then those registers' loads for
// the batch two on; after the fourth, the multiply, 4 rounds and one store. Live: 64 words of slots and 8 sums.
// see docs/internals/tools/lab.md#slot-sum
#include "lab.cuh"

namespace slot_sum {

constexpr int kLineWords = 32, kStoreLines = 16384, kRegionWords = 16 << 20, kTrip = 16;
constexpr int kSlots = 32, kChannels = 64, kTokens = 1024, kBatch = 8;

//: a slot's value at (token of the call, slot, channel), and the token's divisor: small, every result exact
constexpr int value_of(int token, int slot, int c) { return (token + slot + c) % 3 + 1; }
constexpr int den_of(int token) { return token % 2 ? 2 : 4; }

//: where slot `s` of a token sits: packed with its token's (the token-major layout), or in stream `s` of the CTA's
//: 32, at a row scattered over the stream (a layout a CTA a stream: a token's partials gathered)
constexpr long long packed_at(long long token, int slot) { return token * kSlots + slot; }
constexpr long long gathered_at(long long block, int token, int slot) {
  //: an odd multiple of the token modulo the stream's rows: every token a row of its own
  return (block * kSlots + slot) * kTokens + ((static_cast<uint32_t>(token) * 613u + slot * 331u) & (kTokens - 1));
}

template <int WARPS, int STRIDE>
__global__ void __launch_bounds__(32 * WARPS, 1) store(long long* times, uint32_t* region) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint32_t* const own = region + static_cast<long long>(blockIdx.x) * kRegionWords + lane;
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int i = 0; i < kStoreLines / WARPS; i += kTrip) {
#pragma unroll
    for (int j = 0; j < kTrip; ++j) {
      //: the warp's line `i + j` is the CTA's line `warp + WARPS (i + j)`, `STRIDE` lines apart
      const uint32_t line = static_cast<uint32_t>(warp + WARPS * (i + j)) * STRIDE;
      own[static_cast<long long>(line) * kLineWords] = line * kLineWords + lane + 1u;
    }
  }
  clock.stop(times);
}

//: the slots and the divisors, written on the device: 540 MB the host does not hold
__global__ void fill(uint16_t* slots, float* inverse, int gather) {
  const long long token = static_cast<long long>(blockIdx.x) * kTokens + blockIdx.y;
  for (int i = threadIdx.x; i < kSlots * kChannels; i += blockDim.x) {
    const int slot = i / kChannels;
    const long long row = gather ? gathered_at(blockIdx.x, blockIdx.y, slot) : packed_at(token, slot);
    slots[row * kChannels + i % kChannels] =
        __half_as_ushort(__float2half(static_cast<float>(value_of(static_cast<int>(token), slot, i % kChannels))));
  }
  if (threadIdx.x == 0) inverse[token] = 1.0f / static_cast<float>(den_of(static_cast<int>(token)));
}

template <int WARPS, int GATHER>
__global__ void __launch_bounds__(32 * WARPS, 1)
    sum(long long* times, const uint4* slots, const float* inverse, uint4* out) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, group = lane >> 3, piece = lane & 7;
  const long long first = static_cast<long long>(blockIdx.x) * kTokens;
  constexpr int kQuads = kTokens / 4 / WARPS;
  uint4 held[2][kBatch];
  //: batch `n` of the warp's call: slots `8 (n % 4)` on of its token of quad `n / 4`
  const auto token_of = [&](int quad) { return first + (warp + WARPS * min(quad, kQuads - 1)) * 4 + group; };
  const auto load = [&](int h, int n) {
    const long long token = token_of(n / 4);
#pragma unroll
    for (int q = 0; q < kBatch; ++q) {
      const int slot = (n % 4) * kBatch + q;
      const long long row = GATHER ? gathered_at(blockIdx.x, static_cast<int>(token - first), slot)
                                   : packed_at(token, slot);
      held[h][q] = slots[row * 8 + piece];
    }
  };
  load(0, 0);
  load(1, 1);
  float scale = inverse[token_of(0)];
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int quad = 0; quad < kQuads; ++quad) {
    float acc[8] = {};
#pragma unroll
    for (int b = 0; b < 4; ++b) {
#pragma unroll
      for (int q = 0; q < kBatch; ++q) {
        const uint32_t w[4] = {held[b & 1][q].x, held[b & 1][q].y, held[b & 1][q].z, held[b & 1][q].w};
#pragma unroll
        for (int i = 0; i < 8; ++i)
          acc[i] += __half2float(__ushort_as_half(static_cast<unsigned short>(w[i / 2] >> (16 * (i & 1)))));
      }
      load(b & 1, 4 * quad + b + 2);
    }
    //: the token's 8 channels of this lane: the sums over the divisor, as bf16
    const uint4 line = make_uint4(lab::bf16x2_round(acc[0] * scale, acc[1] * scale),
                                  lab::bf16x2_round(acc[2] * scale, acc[3] * scale),
                                  lab::bf16x2_round(acc[4] * scale, acc[5] * scale),
                                  lab::bf16x2_round(acc[6] * scale, acc[7] * scale));
    out[token_of(quad) * 8 + piece] = line;
    scale = inverse[token_of(quad + 1)];
  }
  clock.stop(times);
}

}  // namespace slot_sum

template <int WARPS, int STRIDE>
void run_store(lab::Bench& b, uint32_t* region) {
  using namespace slot_sum;
  const std::string name = std::string(STRIDE == 1 ? "a line stored, lines packed" : "a line stored, one every 4 KB")
                           + ", " + std::to_string(WARPS) + (WARPS == 1 ? " warp" : " warps");
  const size_t words = static_cast<size_t>(b.sms) * kRegionWords;
  LAB_CHECK(cudaMemset(region, 0, words * 4));
  store<WARPS, STRIDE><<<b.sms, 32 * WARPS>>>(b.times, region);
  LAB_CHECK(cudaDeviceSynchronize());
  //: the first and the last CTA's regions: a written line holds its words, every other nothing
  std::vector<uint32_t> got(kRegionWords);
  long long wrong = 0;
  for (int blk : {0, b.sms - 1}) {
    LAB_CHECK(cudaMemcpy(got.data(), region + static_cast<size_t>(blk) * kRegionWords, got.size() * 4,
                         cudaMemcpyDeviceToHost));
    for (size_t w = 0; w < got.size(); ++w) {
      const size_t line = w / kLineWords;
      const bool written = line % STRIDE == 0 && line / STRIDE < kStoreLines;
      wrong += got[w] != (written ? static_cast<uint32_t>(w) + 1u : 0u);
    }
  }
  b.check(name, wrong == 0, std::to_string(wrong) + " of " + std::to_string(2 * got.size()) + " words differ");
  const lab::Samples cycles = b.variant(
      [&] {
        LAB_CHECK(cudaMemset(region, 0, words * 4));
        store<WARPS, STRIDE><<<b.sms, 32 * WARPS>>>(b.times, region);
      },
      [&] { return b.span(WARPS) / kStoreLines; });
  b.record(name, "cycles a line a CTA", cycles);
}

inline uint32_t bf16_host(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u >> 16;
}

template <int WARPS, int GATHER>
void run_sum(lab::Bench& b, const uint4* slots, const float* inverse, uint4* out) {
  using namespace slot_sum;
  const std::string name = std::string(GATHER ? "a token's 32 partials gathered from 32 streams"
                                              : "a token's 32 slots, packed,")
                           + " summed, divided and stored, " + std::to_string(WARPS) + " warps";
  const size_t values = static_cast<size_t>(b.sms) * kTokens * kChannels;
  LAB_CHECK(cudaMemset(out, 0, values * 2));
  sum<WARPS, GATHER><<<b.sms, 32 * WARPS>>>(b.times, slots, inverse, out);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> got(values);
  LAB_CHECK(cudaMemcpy(got.data(), out, values * 2, cudaMemcpyDeviceToHost));
  long long wrong = 0;
  for (size_t v = 0; v < values; ++v) {
    const int token = static_cast<int>(v / kChannels), c = static_cast<int>(v % kChannels);
    int total = 0;
    for (int s = 0; s < kSlots; ++s) total += value_of(token, s, c);
    //: a sum under 97 over 2 or 4: exact in bf16
    wrong += got[v] != bf16_host(static_cast<float>(total) / static_cast<float>(den_of(token)));
  }
  b.check(name, wrong == 0, std::to_string(wrong) + " of " + std::to_string(values) + " output values differ");
  const lab::Samples cycles = b.variant([&] { sum<WARPS, GATHER><<<b.sms, 32 * WARPS>>>(b.times, slots, inverse, out); },
                                        [&] { return b.span(WARPS) / kTokens; });
  b.record(name, "cycles a token a CTA", cycles);
}

int main(int argc, char** argv) {
  using namespace slot_sum;
  lab::Bench b(argc, argv);
  for (int warps : {1, 4})
    for (int stride : {1, 32}) b.expect(lab::kernel("store", {warps, stride}), "STG", ">=", kTrip);
  for (int gather = 0; gather < 2; ++gather)
    for (int warps : {4, 8, 16}) {
      b.expect(lab::kernel("sum", {warps, gather}), "LDG.E.128", ">=", 2 * kBatch);
      //: a trip is a quad's four batches, and a batch's loads are read two batches on: its first two batches'
      //: inside the trip that issued them, a stage ahead all the same
      b.expect(lab::kernel("sum", {warps, gather}), "LOAD_IN_TRIP", "<=", 2 * kBatch);
    }
  if (b.expect_only) return 0;
  uint32_t* region = nullptr;
  uint16_t* slots = nullptr;
  float* inverse = nullptr;
  uint4* out = nullptr;
  LAB_CHECK(cudaMalloc(&region, static_cast<size_t>(b.sms) * kRegionWords * 4));
  LAB_CHECK(cudaMalloc(&slots, static_cast<size_t>(b.sms) * kTokens * kSlots * kChannels * 2));
  LAB_CHECK(cudaMalloc(&inverse, static_cast<size_t>(b.sms) * kTokens * 4));
  LAB_CHECK(cudaMalloc(&out, static_cast<size_t>(b.sms) * kTokens * kChannels * 2));
  fill<<<dim3(b.sms, kTokens), 256>>>(slots, inverse, 0);
  LAB_CHECK(cudaDeviceSynchronize());
  run_store<1, 1>(b, region);
  run_store<4, 1>(b, region);
  run_store<1, 32>(b, region);
  run_store<4, 32>(b, region);
  const uint4* const wide = reinterpret_cast<const uint4*>(slots);
  run_sum<4, 0>(b, wide, inverse, out);
  run_sum<8, 0>(b, wide, inverse, out);
  run_sum<16, 0>(b, wide, inverse, out);
  //: the same slots laid a stream a slot, a token's row scattered in each
  fill<<<dim3(b.sms, kTokens), 256>>>(slots, inverse, 1);
  LAB_CHECK(cudaDeviceSynchronize());
  run_sum<4, 1>(b, wide, inverse, out);
  run_sum<8, 1>(b, wide, inverse, out);
  run_sum<16, 1>(b, wide, inverse, out);
  return 0;
}
