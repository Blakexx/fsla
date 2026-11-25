// measure/lab/experiments/carry_skeleton.cu -- QUESTION: does the LIBRARY's carry skeleton -- its three roles at their
// registers, their barriers and their order of work, every component a stub -- run a call's windows to the end on
// every multiprocessor, and what does a window of nothing but its barriers cost?
// METHOD: this file compiles the library's device code (`carry/carry_kernel.cuh`) and puts a clock around `carry_cta`,
// a CTA an SM of 512 threads; a call is 256 windows. Every CTA writes its windows when its roles have returned: a CTA
// that hangs or leaves early fails the check.
// ROWS: the hand-offs as shared barriers whose waits suspend (the first row), and as named barriers of the two roles'
// threads.
// TIMELINE: no MMA and no memory traffic. A window is the folds' four arrivals behind the readout's reads of the
// window before, the readout's two passes behind them and each panel's turn behind the drain's, the drain's four
// turns behind the readout's: suspended waits and arrivals alone. Live: nothing.
// see docs/internals/tools/lab.md#carry-skeleton
#include "lab.cuh"
#include "carry/carry_kernel.cuh"

namespace carry_skeleton {

namespace carry = fsla::carry::detail;
using Bt = ::denseref::arch::BuildTag<::denseref::arch::Build{900}>;
//: the registers a fold thread and a readout thread take: the lab's split (docs/lab/carry_token.md)
constexpr int kFoldRegs = 176, kReadRegs = 104;

//: THE LIBRARY'S CTA: the clock is the lab's, the work the library's
template <int WAIT>
__global__ void __launch_bounds__(carry::kThreads, 1) timed(long long* times, int* done, int windows) {
  lab::WarpSpan clock;
  clock.start();
  const carry::Args a{windows};
  carry::carry_cta<Bt, kFoldRegs, kReadRegs, WAIT>(a);
  clock.stop(times);
  if ((threadIdx.x & 31) == 0) atomicAdd(done + blockIdx.x, windows);
}

}  // namespace carry_skeleton

//: a row: the skeleton with its hand-offs in the form `WAIT`
template <int WAIT>
void run(lab::Bench& b, int* done, const std::string& name) {
  using namespace carry_skeleton;
  constexpr int kWindows = 256, kWarps = carry::kThreads / 32;
  LAB_CHECK(cudaMemset(done, 0, static_cast<size_t>(b.sms) * sizeof(int)));
  timed<WAIT><<<b.sms, carry::kThreads>>>(b.times, done, kWindows);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<int> got(b.sms);
  LAB_CHECK(cudaMemcpy(got.data(), done, got.size() * sizeof(int), cudaMemcpyDeviceToHost));
  long long wrong = 0;
  for (int v : got) wrong += v != kWindows * kWarps;
  b.check(name, wrong == 0,
          std::to_string(wrong) + " of " + std::to_string(b.sms) + " CTAs did not return from every warp");
  const lab::Samples cycles = b.variant([&] { timed<WAIT><<<b.sms, carry::kThreads>>>(b.times, done, kWindows); },
                                        [&] { return b.span(kWarps) / kWindows; });
  b.record(name, "cycles a window a CTA", cycles);
}

int main(int argc, char** argv) {
  using namespace carry_skeleton;
  lab::Bench b(argc, argv);
  if (b.expect_only) return 0;
  int* done = nullptr;
  LAB_CHECK(cudaMalloc(&done, static_cast<size_t>(b.sms) * sizeof(int)));
  const std::string base = "the skeleton's roles, every component a stub";
  run<0>(b, done, base);
  run<1>(b, done, base + ", its hand-offs named barriers");
  return 0;
}
