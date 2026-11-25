// measure/lab/experiments/carry_skeleton.cu -- QUESTION: does the LIBRARY's carry skeleton -- its three roles at their
// registers, their hand-offs and their order of work -- run a call's windows to the end on every multiprocessor, and
// what does a window of nothing but its hand-offs cost? The call has no tokens and no levels: the folds' supply, the
// one component built, lists nothing.
// METHOD: this file compiles the library's device code (`carry/carry_kernel.cuh`) and puts a clock around `carry_cta`,
// a CTA an SM of 512 threads; a call is 256 windows. Every warp adds its windows to its CTA's word when its role has
// returned: a CTA that hangs or leaves early fails the check. ROWS: the kernel as the library compiles it (its tracer
// nothing), and with every warp's hand-offs and windows stamped (warpline's stamps through the kernel's tracer), whose
// timelines of two CTAs the record holds.
// TIMELINE: no MMA and no memory traffic. A fold warp's empty listing a window (its running sums and a vote); a window is the folds' four arrivals behind the readout's reads of the
// window before, the readout's two passes behind them and each panel's turn behind the drain's, the drain's four
// turns behind the readout's: named barriers alone, the producer arriving and the waiter syncing. Live: nothing.
// see docs/internals/tools/lab.md#carry-skeleton
#include "lab.cuh"
#include "carry/carry_kernel.cuh"

namespace carry_skeleton {

namespace carry = fsla::carry::detail;
using Bt = ::denseref::arch::BuildTag<::denseref::arch::Build{900}>;
//: the registers a fold thread and a readout thread take: the lab's split (docs/lab/carry_token.md)
constexpr int kFoldRegs = 176, kReadRegs = 104;
//: a phase's site in a warp's timeline: above every barrier's
constexpr uint32_t kPhases = 0xc000u;

//: THE KERNEL'S TRACER AS WARPLINE'S STAMPS: a barrier's site is its number, a phase's `kPhases` on
template <bool ON>
struct Stamps {
  lab::Trace<ON> row;
  __device__ __forceinline__ void wait_begin(int id) { row.stamp(warpline::kWaitBegin, static_cast<uint32_t>(id)); }
  __device__ __forceinline__ void wait_end(int id) { row.stamp(warpline::kWaitEnd, static_cast<uint32_t>(id)); }
  __device__ __forceinline__ void arrive(int id) { row.stamp(warpline::kArrive, static_cast<uint32_t>(id)); }
  __device__ __forceinline__ void phase(int site) { row.mark(warpline::kPhaseBegin, kPhases + static_cast<uint32_t>(site)); }
  __device__ __forceinline__ void round(int, int, int, int, int, uint32_t) {}
};

//: THE LIBRARY'S CTA: the clock is the lab's, the work the library's
template <int TRACE>
__global__ void __launch_bounds__(carry::kThreads, 1)
    timed(long long* times, int* done, unsigned long long* trace, int windows) {
  Stamps<TRACE != 0> tr;
  tr.row.bind(trace);
  lab::WarpSpan clock;
  clock.start();
  //: a call of no tokens and no levels: every warp's lists are empty, so a window is its hand-offs
  carry::Operands o{};
  o.windows = windows;
  o.ctas = 1;
  o.line_bytes = 2 * carry::kChannels;
  carry::carry_cta<Bt, kFoldRegs, kReadRegs, Stamps<TRACE != 0>>(o, tr);
  clock.stop(times);
  tr.row.close(0);
  if ((threadIdx.x & 31) == 0) atomicAdd(done + blockIdx.x, windows);
}

}  // namespace carry_skeleton

//: a row: the skeleton, its warps' timelines stamped where `TRACE`
template <int TRACE>
void run(lab::Bench& b, int* done, unsigned long long* rows, const std::string& name) {
  using namespace carry_skeleton;
  constexpr int kWindows = TRACE != 0 ? 64 : 256, kWarps = carry::kThreads / 32;
  LAB_CHECK(cudaMemset(done, 0, static_cast<size_t>(b.sms) * sizeof(int)));
  timed<TRACE><<<b.sms, carry::kThreads>>>(b.times, done, rows, kWindows);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<int> got(b.sms);
  LAB_CHECK(cudaMemcpy(got.data(), done, got.size() * sizeof(int), cudaMemcpyDeviceToHost));
  long long wrong = 0;
  for (int v : got) wrong += v != kWindows * kWarps;
  b.check(name, wrong == 0,
          std::to_string(wrong) + " of " + std::to_string(b.sms) + " CTAs did not return from every warp");
  const lab::Samples cycles =
      b.variant([&] { timed<TRACE><<<b.sms, carry::kThreads>>>(b.times, done, rows, kWindows); },
                [&] { return b.span(kWarps) / kWindows; });
  b.record(name, "cycles a window a CTA", cycles);
  if constexpr (TRACE != 0) {
    std::vector<std::string> roles;
    for (int w = 0; w < kWarps; ++w) {
      const int thread = 32 * w;
      roles.push_back(thread < carry::kReadAt ? "fold box " + std::to_string(thread / carry::kGroup)
                      : thread < carry::kDrainAt ? "readout" : "drain");
    }
    std::vector<std::pair<uint32_t, std::string>> legend;
    for (int box = 0; box < carry::kFolds; ++box)
      legend.push_back({static_cast<uint32_t>(carry::kFoldId + box), "fold box " + std::to_string(box) + "'s own barrier"});
    legend.push_back({static_cast<uint32_t>(carry::kReadId), "the readout's own barrier"});
    for (int k = 0; k < carry::kChunks; ++k) {
      legend.push_back({static_cast<uint32_t>(carry::kTilesReadyId + k), "tile group " + std::to_string(k) + " published"});
      legend.push_back({static_cast<uint32_t>(carry::kTilesReadId + k), "tile group " + std::to_string(k) + " read"});
    }
    for (int p = 0; p < carry::kPanels; ++p) {
      legend.push_back({static_cast<uint32_t>(carry::kPanelReadyId + p), "panel " + std::to_string(p) + " filled"});
      legend.push_back({static_cast<uint32_t>(carry::kPanelFreedId + p), "panel " + std::to_string(p) + " drained"});
    }
    legend.insert(legend.end(), {{kPhases + carry::kSiteEntry, "entry"}, {kPhases + carry::kSiteExit, "exit"},
                                 {kPhases + carry::kSiteWindow, "window"}});
    if (b.sms > 1) b.trace(name, rows, kWarps, {0, b.sms / 2}, roles, legend);
    else b.trace(name, rows, kWarps, {0}, roles, legend);
  }
}

int main(int argc, char** argv) {
  using namespace carry_skeleton;
  lab::Bench b(argc, argv);
  if (b.expect_only) return 0;
  constexpr int kWarps = carry::kThreads / 32;
  int* done = nullptr;
  unsigned long long* rows = nullptr;
  const size_t trace_words = static_cast<size_t>(b.sms) * kWarps * lab::Trace<true>::kRow;
  LAB_CHECK(cudaMalloc(&done, static_cast<size_t>(b.sms) * sizeof(int)));
  LAB_CHECK(cudaMalloc(&rows, trace_words * 8));
  LAB_CHECK(cudaMemset(rows, 0, trace_words * 8));
  const std::string base = "the carry on a call of no tokens, a window its hand-offs";
  run<0>(b, done, rows, base);
  run<1>(b, done, rows, base + ", every warp's timeline stamped");
  return 0;
}
