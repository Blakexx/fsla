// measure/lab/experiments/step_rolled.cu -- QUESTION: what does a tensor step cost its warpgroup when it is issued
// from a rolled loop with a run-time trip count, against the same steps written as a straight block? METHOD: a CTA an
// SM, one warpgroup; a trip is 16 m64n16k16 steps into 16 accumulators, both operands one tile of ones in shared
// memory, their descriptors held in registers. STRAIGHT: the 16 steps one block behind one fence. FENCED: the same
// with a fence before every step. ROLLED: for each of the 16 accumulators a loop of a run-time count (one) around its
// step, a fence before every step -- the form of a list's steps issued by its count. Recorded: cycles a step. The
// sums are checked exactly: 16 a step in every value.
// TIMELINE: a trip: [the fence,] 16 steps -- each [its loop's test and back branch,] [its fence,] the step -- the
// commit, the wait. Live: the 16 accumulators (128 registers a thread) and two descriptors.
// see docs/lab/step_rolled.md
#include "lab.cuh"

namespace step_rolled {

enum Form : int { kStraight = 0, kFenced = 1, kRolled = 2 };
constexpr int kThreads = 128, kSteps = 16, kTileA = 0, kTileB = 2048, kSmem = 4096;

template <int FORM>
__global__ void __launch_bounds__(kThreads, 1) steps(long long* times, float* out, int trips, int count) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x;
  const uint32_t base = lab::smem(smem);
  //: the two tiles of ones
#pragma unroll 1
  for (int i = tid; i < kSmem / 2; i += kThreads) reinterpret_cast<uint16_t*>(smem)[i] = 0x3f80;
  lab::bulk_fence();
  __syncthreads();
  const uint64_t a = lab::held(lab::wg_desc(base + kTileA, 8, 16)), b = lab::held(lab::wg_desc(base + kTileB, 8, 16));
  float d[kSteps][8] = {};
  const long long t0 = clock64();
#pragma unroll 1
  for (int t = 0; t < trips; ++t) {
    if constexpr (FORM == kStraight) lab::wg_fence();
#pragma unroll
    for (int s = 0; s < kSteps; ++s) {
      if constexpr (FORM == kRolled) {
        //: a list's steps by its count: the count is the launch's, which the toolkit cannot know
#pragma unroll 1
        for (int r = 0; r < count; ++r) {
          lab::wg_fence();
          lab::wg_bf16_n16_ss_mn(d[s], a, b, true);
        }
      } else {
        if constexpr (FORM == kFenced) lab::wg_fence();
        lab::wg_bf16_n16_ss_mn(d[s], a, b, true);
      }
    }
    lab::wg_commit();
    lab::wg_wait<0>();
  }
#pragma unroll
  for (int s = 0; s < kSteps; ++s)
#pragma unroll
    for (int i = 0; i < 8; ++i) lab::wg_hold(d[s][i]);
  const long long t1 = clock64();
  if (tid == 0) {
    times[2 * blockIdx.x] = t0;
    times[2 * blockIdx.x + 1] = t1;
  }
  //: every value is 16 a step: its largest and smallest, a thread
  float lo = d[0][0], hi = d[0][0];
#pragma unroll
  for (int s = 0; s < kSteps; ++s)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      lo = fminf(lo, d[s][i]);
      hi = fmaxf(hi, d[s][i]);
    }
  out[2 * (blockIdx.x * kThreads + tid)] = lo;
  out[2 * (blockIdx.x * kThreads + tid) + 1] = hi;
}

}  // namespace step_rolled

//: A ROW: the form's checked call, then its timed launches
template <int FORM>
void run(lab::Bench& b, float* out) {
  using namespace step_rolled;
  constexpr int kCheckTrips = 16, kTrips = 4096;
  static const char* const forms[] = {"16 steps a straight block behind one fence", "16 steps a straight block, a fence a step",
                                      "16 steps each in a rolled loop of a run-time count of one, a fence a step"};
  const std::string name = forms[FORM];
  const auto kernel = steps<FORM>;
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  kernel<<<b.sms, kThreads, kSmem>>>(b.times, out, kCheckTrips, 1);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<float> got(static_cast<size_t>(b.sms) * kThreads * 2);
  LAB_CHECK(cudaMemcpy(got.data(), out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
  long long wrong = 0;
  for (float v : got) wrong += v != 16.0f * kCheckTrips;
  b.check(name, wrong == 0, std::to_string(wrong) + " of " + std::to_string(got.size()) + " extremes differ from the exact sum");
  const lab::Samples cycles = b.variant([&] { kernel<<<b.sms, kThreads, kSmem>>>(b.times, out, kTrips, 1); },
                                        [&] {
                                          std::vector<long long> h(static_cast<size_t>(b.sms) * 2);
                                          LAB_CHECK(cudaMemcpy(h.data(), b.times, h.size() * sizeof(long long), cudaMemcpyDeviceToHost));
                                          double sum = 0;
                                          for (int blk = 0; blk < b.sms; ++blk) sum += double(h[2 * blk + 1] - h[2 * blk]);
                                          return sum / (double(b.sms) * kTrips * kSteps);
                                        });
  b.record(name, "cycles a step", cycles);
}

int main(int argc, char** argv) {
  using namespace step_rolled;
  lab::Bench b(argc, argv);
  for (int form : {kStraight, kFenced, kRolled}) {
    const std::string k = lab::kernel("step_rolled::steps", {form});
    b.expect(k, "HGMMA", ">=", kSteps);
    b.exempt(k, "a rate probe: what a step costs its warpgroup in each form, no kernel's step");
    //: the row's question: a trip's steps waited out at its end, the wait directly behind them
    b.expect(k, "WAIT_AFTER_ISSUE", "<=", 1);
    //: the rolled form is the row's question: its 16 steps each in a loop with no wait in it
    if (form == kRolled) b.expect(k, "STEP_ROLLED", "<=", kSteps);
  }
  if (b.expect_only) return 0;
  float* out = nullptr;
  LAB_CHECK(cudaMalloc(&out, sizeof(float) * b.sms * kThreads * 2));
  run<kStraight>(b, out);
  run<kFenced>(b, out);
  run<kRolled>(b, out);
  return 0;
}
