// measure/lab/experiments/call_pair.cu -- QUESTION: what is a call's time with the carry AND the intra, and does
// launching the two kernels TOGETHER on two streams buy anything over one after the other? The carry's CTA takes
// its multiprocessor's whole register file for the call, so an intra CTA runs only on a multiprocessor the carry
// left free or has left. METHOD: the two labs' own kernels, unchanged -- `carry_window`'s winner's form with the
// den and no clock inside a role, on the cell's 128 CTAs of the card's 132 multiprocessors, and `intra_window`'s
// form with the levels tile-major and the values token-major, 16 CTAs a window -- each over 16 calls of 65,536
// tokens so a launch's own time is small beside them. Rows, by wall time a call: the carry alone, the intra alone,
// the two on one stream, the two on two streams. The kernels' outputs are their own labs' to check.
// TIMELINE: the two kernels', each stated in its experiment's header; this experiment adds no device code.
// see docs/internals/tools/lab.md#call-pair
#include <chrono>

#define main carry_window_main
#include "carry_window.cu"
#undef main
#define main intra_window_main
#include "intra_window.cu"
#undef main

namespace call_pair {

//: a call is 65,536 tokens: 64 of either kernel's windows; the rows time 16
constexpr int kCallWindows = 64, kCalls = 16;
constexpr int kCarry = carry_window::kFold | carry_window::kRead | carry_window::kSuspend | carry_window::kLean
                       | carry_window::kNoClock | carry_window::kDen;
constexpr int kCarryCtas = 128;

}  // namespace call_pair

int main(int argc, char** argv) {
  using namespace call_pair;
  lab::Bench b(argc, argv);
  if (b.expect_only) {
    //: the kernels this binary holds are the two labs': each states its own expectations, and this binary's compile adds
    //: findings of its own, each declared below with its count
    carry_window_main(argc, argv);
    const int intra_status = intra_window_main(argc, argv);
    //: the intra's window kernels wait on their group straight after its steps, twice a kernel: the intra lab's debt
    //: (`_PRE_WAIT`), whose rule this experiment's rows are held to; predates the rule
    b.expect("intra_window::window", "WAIT_AFTER_ISSUE", "<=", 2);
    for (long long roles : {70756767629603LL, 289356664086200611LL}) {
      const std::string k = lab::kernel("carry_window::window", {roles});
      //: two of the carry lab's rows (not the carry this experiment launches) spill 8 to 12 bytes at their 128 registers in
      //: this binary, which holds the intra's kernels beside the carry lab's (the carry lab's own binary spills three
      //: other rows, declared in `carry_window.cu`)
      b.expect(k, "REGISTER_SPILL", "<=", 1);
      //: the same two rows keep an 8-byte stack frame for those spills
      b.expect(k, "LOCAL_MEMORY_USED", "<=", 1);
    }
    return intra_status;
  }
  if (b.sms < kCarryCtas) {
    std::fprintf(stderr, "call_pair: the cell is %d carry CTAs; this card has %d multiprocessors\n", kCarryCtas, b.sms);
    return 2;
  }
  const Buffers carry = buffers(b);
  const size_t tokens = static_cast<size_t>(kCallWindows) * intra_window::kWindow;
  const Call intra = call_of(tokens);
  const int intra_ctas = kCallWindows * intra_window::kTiles;
  //: the intra's CTAs are more than the multiprocessors: their clocks have a buffer of their own
  long long* intra_times = nullptr;
  LAB_CHECK(cudaMalloc(&intra_times, sizeof(long long) * intra_ctas * (intra_window::kThreads >> 5) * 2));
  const auto carry_kernel = carry_window::window<kCarry>;
  const auto intra_kernel = intra_window::window<intra_window::kMixed>;
  LAB_CHECK(cudaFuncSetAttribute(carry_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, carry_window::kSmemBytes));
  LAB_CHECK(cudaFuncSetAttribute(intra_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, intra_window::kSmemBytes));
  cudaStream_t first, second;
  LAB_CHECK(cudaStreamCreate(&first));
  LAB_CHECK(cudaStreamCreate(&second));
  const auto launch_carry = [&](cudaStream_t s) {
    carry_kernel<<<kCarryCtas, carry_window::threads_of<kCarry>, carry_window::kSmemBytes, s>>>(
        b.times, carry.waits, b.sinks, carry.values, carry.dense, carry.state, carry.out, nullptr, kCalls * kCallWindows);
  };
  const auto launch_intra = [&](cudaStream_t s) {
    intra_kernel<<<intra_ctas, intra_window::kThreads, intra_window::kSmemBytes, s>>>(
        intra.maps[0], intra.maps[1], intra.maps[2], intra.tiled, intra_times, intra.out, intra.den, kCalls,
        intra_window::kTiles);
  };
  //: a row: its launches, timed wall to wall from a drained device, a call's share
  const auto row = [&](const std::string& name, auto launches) {
    lab::Samples ms;
    launches();
    LAB_CHECK(cudaDeviceSynchronize());
    for (int r = 0; r < b.reps; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      launches();
      LAB_CHECK(cudaDeviceSynchronize());
      ms.v.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1000.0 / kCalls);
    }
    b.unchecked(name, "a pair of kernels each checked in its own experiment; this row is their wall time");
    b.record(name, "milliseconds a call of 65,536 tokens", ms);
  };
  row("the carry alone", [&] { launch_carry(first); });
  row("the intra alone", [&] { launch_intra(first); });
  row("the carry, then the intra: one stream", [&] {
    launch_carry(first);
    launch_intra(first);
  });
  row("the carry and the intra together: two streams", [&] {
    launch_carry(first);
    launch_intra(second);
  });
  row("the intra and the carry together, the intra launched first", [&] {
    launch_intra(second);
    launch_carry(first);
  });
  return 0;
}
