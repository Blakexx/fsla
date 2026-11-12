// measure/lab/experiments/panel_merge.cu -- QUESTION: a launch's fold groups each leave a panel of f16 pairs, a row
// a token; what does a token cost the card to MERGE them into the fp32 output -- a group's row widened, scaled by
// the group's power of two, the groups summed in fp32 and added into the token's output row -- by the groups a
// launch holds? METHOD: a CTA an SM, 4 warps, each CTA its own run of tokens and each warp every fourth of them; no
// atomics: a row has one writer. A token: G loads of a 128-byte panel row (a lane a pair) and one of its output
// row's 256 bytes (a lane two fp32), G multiply-adds a half, two stores. G = 2, 8 and 16 over 262,144 tokens: the
// panels 67, 268 and 537 MB, none held by the cache. Every output row is checked exactly.
// TIMELINE: a warp's trip is two tokens, each at its own registers: a token's sums and its two stores (its panel
// words and its output row loaded a trip before), then those registers' loads for the token two on. Live: two
// tokens' G words and 2 fp32 each.
// see docs/internals/tools/lab.md#panel-merge
#include "lab.cuh"

namespace panel_merge {

constexpr int kChannels = 64, kWarps = 4, kThreads = 32 * kWarps;

//: group `g`'s pair of channels (c, c + 1) of token `t`, small integers exact in f16; the group's scale
constexpr int value_of(int g, int t, int c) { return (t + g + c) % 5 + 1; }
constexpr float scale_of(int g) { return static_cast<float>(1 << (g % 3)); }

template <int G>
__global__ void __launch_bounds__(kThreads, 1)
    merge(long long* times, const uint32_t* panels, float* out, int tokens, int per) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int first = blockIdx.x * per + warp, count = per / kWarps;
  uint32_t word[2][G];
  float lo[2], hi[2];
  //: a token's panel words and its output row, into the registers of stage `h`
  const auto load = [&](int h, int k) {
    const long long t = first + kWarps * min(k, count - 1);
#pragma unroll
    for (int g = 0; g < G; ++g) word[h][g] = panels[(static_cast<long long>(g) * tokens + t) * (kChannels / 2) + lane];
    lo[h] = out[t * kChannels + 2 * lane];
    hi[h] = out[t * kChannels + 2 * lane + 1];
  };
  load(0, 0);
  load(1, 1);

  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int k = 0; k < count; k += 2) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const long long t = first + kWarps * (k + h);
      float a = lo[h], b = hi[h];
#pragma unroll
      for (int g = 0; g < G; ++g) {
        const float2 pair = __half22float2(*reinterpret_cast<const __half2*>(&word[h][g]));
        a += pair.x * scale_of(g);
        b += pair.y * scale_of(g);
      }
      out[t * kChannels + 2 * lane] = a;
      out[t * kChannels + 2 * lane + 1] = b;
      load(h, k + h + 2);
    }
  }
  clock.stop(times);
}

}  // namespace panel_merge

inline uint32_t half_bits(int v) {
  //: a small positive integer as f16: an exponent and 10 bits
  int e = 0;
  while ((v >> (e + 1)) != 0) ++e;
  return static_cast<uint32_t>((e + 15) << 10 | ((v << (10 - e)) & 1023));
}

template <int G>
void run(lab::Bench& b, uint32_t* panels, float* out, int tokens) {
  using namespace panel_merge;
  const int per = (tokens / b.sms) & ~7, merged = per * b.sms;
  const std::string name = std::to_string(G) + " groups' panels merged";
  std::vector<uint32_t> host(static_cast<size_t>(G) * tokens * (kChannels / 2));
  for (int g = 0; g < G; ++g)
    for (int t = 0; t < tokens; ++t)
      for (int w = 0; w < kChannels / 2; ++w)
        host[(static_cast<size_t>(g) * tokens + t) * (kChannels / 2) + w]
            = half_bits(value_of(g, t, 2 * w)) | half_bits(value_of(g, t, 2 * w + 1)) << 16;
  LAB_CHECK(cudaMemcpy(panels, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMemset(out, 0, static_cast<size_t>(tokens) * kChannels * sizeof(float)));
  merge<G><<<b.sms, kThreads>>>(b.times, panels, out, tokens, per);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<float> got(static_cast<size_t>(merged) * kChannels);
  LAB_CHECK(cudaMemcpy(got.data(), out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
  long long wrong = 0;
  for (int t = 0; t < merged; ++t)
    for (int c = 0; c < kChannels; ++c) {
      float want = 0.0f;
      for (int g = 0; g < G; ++g) want += static_cast<float>(value_of(g, t, c)) * scale_of(g);
      wrong += got[static_cast<size_t>(t) * kChannels + c] != want;
    }
  b.check(name, wrong == 0,
          std::to_string(wrong) + " of " + std::to_string(got.size()) + " output values differ from the exact ones");
  const lab::Samples cycles = b.variant([&] { merge<G><<<b.sms, kThreads>>>(b.times, panels, out, tokens, per); },
                                        [&] { return b.span(kWarps) / per; });
  b.record(name, "SM-cycles a token", cycles);
}

int main(int argc, char** argv) {
  using namespace panel_merge;
  lab::Bench b(argc, argv);
  for (int groups : {2, 8, 16}) b.expect(lab::kernel("merge", {groups}), "STG", ">=", 4);
  if (b.expect_only) return 0;
  constexpr int kTokens = 262144, kMost = 16;
  uint32_t* panels = nullptr;
  float* out = nullptr;
  LAB_CHECK(cudaMalloc(&panels, static_cast<size_t>(kMost) * kTokens * (kChannels / 2) * 4));
  LAB_CHECK(cudaMalloc(&out, static_cast<size_t>(kTokens) * kChannels * sizeof(float)));
  run<2>(b, panels, out, kTokens);
  run<8>(b, panels, out, kTokens);
  run<16>(b, panels, out, kTokens);
  return 0;
}
