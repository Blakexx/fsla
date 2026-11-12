// measure/lab/experiments/drain_rate.cu -- QUESTION: what is an SM's capacity for fp32 adds into global memory --
// does a token's partial cost less as more warps issue it, or is the 4-warp number the path's own rate? And what
// do the adds cost when the CTAs' rows are spread over a window, as a carry's are, beside every CTA its own rows
// and every CTA the same ones? METHOD: a CTA an SM of 4, 8 or 16 warps; a round is a partial of 64 tokens x 64
// channels from registers, one `red.global.add.f32` a value, a warp's 32 lanes one 128-byte line. `own`: each CTA
// its own panel of 64 rows. `one`: every CTA the same 64 rows. `spread`: a token's row is drawn, a (CTA, round,
// token), from a panel of 1,024 rows every CTA shares. Every sum is checked exactly.
// TIMELINE: no load: a thread's values are registers from before the clock. A trip is a round: the thread's adds,
// each to its own line, issued back to back with nothing waited on; live: the values (32, 16 or 8 a thread).
// see docs/internals/tools/lab.md#drain-rate
#include "lab.cuh"

namespace drain_rate {

constexpr int kTokens = 64, kChannels = 64, kPanel = kTokens * kChannels, kRows = 1024;
enum Form : int { kOwn = 0, kOne, kSpread, kForms };
const char* kFormName[kForms] = {"each CTA its own rows", "every CTA the same rows", "rows spread over 1,024"};

//: a channel's value, a small integer: every sum exact in fp32; the row a (CTA, round, token) adds into
constexpr int value_of(int channel) { return channel % 7 + 1; }
constexpr uint32_t row_of(uint32_t block, uint32_t round, uint32_t token) {
  return ((block * kTokens + token) * 2654435761u + round * 40503u) >> 22;
}

template <int WARPS, int FORM>
__global__ void __launch_bounds__(32 * WARPS, 1) drain(long long* times, float* out, int rounds) {
  constexpr int kThreads = 32 * WARPS, kMine = kPanel / kThreads;
  const int tid = threadIdx.x;
  float v[kMine];
#pragma unroll
  for (int i = 0; i < kMine; ++i) v[i] = static_cast<float>(value_of((i * kThreads + tid) % kChannels));
  float* const own = out + static_cast<long long>(blockIdx.x) * kPanel;

  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
#pragma unroll
    for (int i = 0; i < kMine; ++i) {
      const int e = i * kThreads + tid;
      float* dst = out + e;
      if constexpr (FORM == kOwn) dst = own + e;
      if constexpr (FORM == kSpread) dst = out + row_of(blockIdx.x, r, e / kChannels) * kChannels + e % kChannels;
      lab::red_add_f32(dst, v[i]);
    }
  }
  clock.stop(times);
}

}  // namespace drain_rate

template <int WARPS, int FORM>
void run(lab::Bench& b, float* out) {
  using namespace drain_rate;
  constexpr int kCheckRounds = 16, kRounds = 2048, kMine = kPanel / (32 * WARPS);
  static_assert(kMine >= 1);
  const std::string name = std::string(kFormName[FORM]) + ", " + std::to_string(WARPS) + " warps";
  const size_t words = FORM == kOwn ? static_cast<size_t>(b.sms) * kPanel
                       : FORM == kOne ? static_cast<size_t>(kPanel) : static_cast<size_t>(kRows) * kChannels;
  LAB_CHECK(cudaMemset(out, 0, words * sizeof(float)));
  drain<WARPS, FORM><<<b.sms, 32 * WARPS>>>(b.times, out, kCheckRounds);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<float> got(words);
  LAB_CHECK(cudaMemcpy(got.data(), out, words * sizeof(float), cudaMemcpyDeviceToHost));
  std::vector<long long> adds(words / kChannels,
                              FORM == kOwn ? kCheckRounds : static_cast<long long>(kCheckRounds) * b.sms);
  if (FORM == kSpread) {
    std::fill(adds.begin(), adds.end(), 0);
    for (int blk = 0; blk < b.sms; ++blk)
      for (int r = 0; r < kCheckRounds; ++r)
        for (int t = 0; t < kTokens; ++t) ++adds[row_of(blk, r, t)];
  }
  long long wrong = 0;
  for (size_t i = 0; i < words; ++i)
    wrong += got[i] != static_cast<float>(adds[i / kChannels] * value_of(static_cast<int>(i % kChannels)));
  b.check(name, wrong == 0,
          std::to_string(wrong) + " of " + std::to_string(words) + " sums differ from the exact ones at "
              + std::to_string(kCheckRounds) + " rounds");
  const lab::Samples cycles = b.variant([&] { drain<WARPS, FORM><<<b.sms, 32 * WARPS>>>(b.times, out, kRounds); },
                                        [&] { return b.span(WARPS) / kRounds / kTokens; });
  b.record(name, "cycles a token a CTA", cycles);
}

template <int WARPS>
void forms(lab::Bench& b, float* out) {
  using namespace drain_rate;
  run<WARPS, kOwn>(b, out);
  run<WARPS, kOne>(b, out);
  run<WARPS, kSpread>(b, out);
}

int main(int argc, char** argv) {
  using namespace drain_rate;
  lab::Bench b(argc, argv);
  for (int warps : {4, 8, 16})
    for (int f = 0; f < kForms; ++f) b.expect(lab::kernel("drain", {warps, f}), "REDG", "==", kPanel / (32 * warps));
  if (b.expect_only) return 0;
  float* out = nullptr;
  const size_t words = std::max(static_cast<size_t>(b.sms) * kPanel, static_cast<size_t>(kRows) * kChannels);
  LAB_CHECK(cudaMalloc(&out, sizeof(float) * words));
  forms<4>(b, out);
  forms<8>(b, out);
  forms<16>(b, out);
  return 0;
}
