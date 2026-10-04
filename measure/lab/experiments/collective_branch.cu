// measure/lab/experiments/collective_branch.cu -- QUESTION: does a warp collective under a branch on the warp's id
// (warp-uniform, but derived from threadIdx) stay inline, and what does its other form cost? METHOD: 8 shuffle chains in
// a runtime-trip loop with no branch (every warp shuffles), under `if (threadIdx.x >> 5 >= 4)`, or with the role taken
// by the trip count; the SASS expectations state the compiled form (inline SHFL, or a CALL into a convergence
// subroutine -- both role forms, CUDA 13.0), the timing a shuffle's cycles in each. see docs/internals/tools/lab.md#collective-branch
#include "lab.cuh"

namespace collective_branch {

//: FORM 0 no branch; 1 under the warp-role branch; 2 the role taken by trip count (the other role runs 0 trips)
template <int FORM>
__global__ void __launch_bounds__(256, 1) shuffles(int iters, long long* t, int* out) {
  const int lane = lab::lane_id();
  const bool role = lab::warp_id() >= 4;
  uint32_t x[8];
  for (int j = 0; j < 8; ++j) x[j] = lane * 7u + j;
  const int trips = FORM == 2 && !role ? 0 : iters;
  lab::WarpClock clock;
  clock.start();
  if (FORM != 1 || role) {
#pragma unroll 1
    for (int it = 0; it < trips; ++it)
#pragma unroll
      for (int j = 0; j < 8; ++j) x[j] = __shfl_sync(0xffffffffu, x[j], x[(j + 1) & 7] & 31u);
  }
  clock.stop(t);
  uint32_t fold = 0;
  for (int j = 0; j < 8; ++j) fold ^= x[j];
  lab::sink(out, fold);
}

constexpr int kIters = 4000;
const char* kFormName[3] = {"no branch", "under the warp-role branch", "role by trip count"};

template <int FORM>
void form(const lab::Bench& b) {
  const std::string name = lab::kernel("shuffles", {FORM});
  b.expect(name, "SHFL", ">=", 8);
  b.expect(name, "CALL", FORM == 0 ? "==" : ">=", FORM == 0 ? 0 : 8);
  if (b.expect_only) return;
  //: one shuffling warp a scheduler in every form: the branch-free form launches the 4 shuffling warps alone
  const int warps = FORM == 0 ? 4 : 8, first = FORM == 0 ? 0 : 4;
  const lab::Samples s =
      b.variant([&] { shuffles<FORM><<<b.sms, 32 * warps>>>(kIters, b.times, b.sinks); },
                [&] { return lab::Bench::mean(b.cycles(warps), warps, first, 4) / (8.0 * kIters); });
  b.record(kFormName[FORM], "cycles a shuffle", s);
}

}  // namespace collective_branch

int main(int argc, char** argv) {
  lab::Bench b(argc, argv);
  collective_branch::form<0>(b);
  collective_branch::form<1>(b);
  collective_branch::form<2>(b);
}
