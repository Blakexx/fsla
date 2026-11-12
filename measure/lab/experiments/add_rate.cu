// measure/lab/experiments/add_rate.cu -- QUESTION: a warp adding f16 lines to global memory from registers, with
// nothing between its adds, gets one out about every 39 cycles (`drain_role`), and the rate scales with the warps
// up to four. What holds a warp to that: a limit a warp, the lanes an instruction carries, the lines it touches,
// or the add being atomic? METHOD: a CTA an SM of 1, 2, 4 or 8 warps; each warp issues 16 instructions a trip to
// its own rows of global memory, from registers that never change. ONE LINE: an instruction's lanes add f16 pairs
// into one 128-byte line, by 32, 16 or 8 lanes. SPREAD: its 32 lanes each add into a different line. STORE: the
// one-line form as a plain 4-byte store a lane, no add. Every row's sums are checked exactly.
// TIMELINE: no load. A warp's trip is 16 instructions, each from a register set before the clock and to an
// address that is a constant off the warp's base: nothing waits on anything. Live: one word a thread.
// see docs/internals/tools/lab.md#add-rate
#include "lab.cuh"

namespace add_rate {

constexpr int kTrip = 16, kLineWords = 32, kWarpLines = kTrip * 32;
enum Form : int { kOneLine = 0, kSpread, kStore };

__device__ __forceinline__ uint32_t f16_of(int v) { return __half_as_ushort(__float2half(static_cast<float>(v))); }

template <int WARPS, int LANES, int FORM>
__global__ void __launch_bounds__(32 * WARPS, 1) add(long long* times, uint32_t* out, int rounds) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  //: a pair of ones: a word's sum after `rounds` adds is exact in f16 to 2,048
  const uint32_t v = lab::held(f16_of(1) | f16_of(1) << 16);
  //: instruction `i` of a trip: line `i` of the warp's (one line), or line `32 i + lane` (spread)
  uint32_t* const own = out + (static_cast<long long>(blockIdx.x) * 8 + warp) * (kWarpLines * kLineWords)
                        + (FORM == kSpread ? lane * kLineWords : 0) + lane;
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
#pragma unroll
    for (int i = 0; i < kTrip; ++i) {
      uint32_t* const to = own + i * (FORM == kSpread ? 32 : 1) * kLineWords;
      if (lane < LANES) {
        if constexpr (FORM == kStore) *to = v + r;
        else lab::red_add_f16x2(to, v);
      }
    }
  }
  clock.stop(times);
}

}  // namespace add_rate

template <int WARPS, int LANES, int FORM>
void run(lab::Bench& b, uint32_t* out) {
  using namespace add_rate;
  constexpr int kCheckRounds = 64, kRounds = 2048;
  const std::string name = std::string(FORM == kOneLine ? "f16 adds into one line an instruction"
                                       : FORM == kSpread ? "f16 adds, a lane a line of its own"
                                       : "plain stores into one line an instruction")
                           + ", " + std::to_string(LANES) + " lanes, " + std::to_string(WARPS)
                           + (WARPS == 1 ? " warp" : " warps");
  const auto kernel = add<WARPS, LANES, FORM>;
  const size_t words = static_cast<size_t>(b.sms) * 8 * kWarpLines * kLineWords;
  LAB_CHECK(cudaMemset(out, 0, words * 4));
  kernel<<<b.sms, 32 * WARPS>>>(b.times, out, kCheckRounds);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> got(words);
  LAB_CHECK(cudaMemcpy(got.data(), out, words * 4, cudaMemcpyDeviceToHost));
  //: a written word holds the rounds' sum in both halves (a store: its last value); every other word nothing
  long long wrong = 0;
  for (size_t w = 0; w < words; ++w) {
    const int lane = static_cast<int>(w % kLineWords), line = static_cast<int>(w / kLineWords) % kWarpLines,
              warp = static_cast<int>(w / (static_cast<size_t>(kWarpLines) * kLineWords)) % 8;
    const bool written = warp < WARPS && lane < LANES
                         && (FORM == kSpread ? line % 32 == lane : line < kTrip);
    uint32_t want = 0;
    if (written) {
      if (FORM == kStore) {
        want = (0x3c00u | 0x3c00u << 16) + (kCheckRounds - 1);
      } else {
        //: 64 as an f16: 2^6, exponent field 21
        want = 0x5400u | 0x5400u << 16;
      }
    }
    wrong += got[w] != want;
  }
  b.check(name, wrong == 0, std::to_string(wrong) + " of " + std::to_string(words) + " words differ");
  const lab::Samples cycles = b.variant([&] { kernel<<<b.sms, 32 * WARPS>>>(b.times, out, kRounds); },
                                        [&] { return b.span(WARPS) / (static_cast<double>(kRounds) * kTrip); });
  b.record(name, "cycles an instruction of each warp, a CTA", cycles);
}

int main(int argc, char** argv) {
  using namespace add_rate;
  lab::Bench b(argc, argv);
  for (int warps : {1, 2, 4, 8}) b.expect(lab::kernel("add", {warps, 32, kOneLine}), "RED", ">=", kTrip);
  for (int lanes : {8, 16}) {
    b.expect(lab::kernel("add", {1, lanes, kOneLine}), "RED", ">=", kTrip);
    //: the lanes past the count issue nothing: a lane's own predicate on each add
    b.expect(lab::kernel("add", {1, lanes, kOneLine}), "LOOP_CONVERGENCE", "<=", 2 * kTrip + 2);
  }
  for (int warps : {1, 4}) {
    b.expect(lab::kernel("add", {warps, 32, kSpread}), "RED", ">=", kTrip);
    b.expect(lab::kernel("add", {warps, 32, kStore}), "RED", "==", 0);
    //: the timed stores are there: one an instruction of the trip
    b.expect(lab::kernel("add", {warps, 32, kStore}), "STG", ">=", kTrip);
  }
  if (b.expect_only) return 0;
  uint32_t* out = nullptr;
  LAB_CHECK(cudaMalloc(&out, static_cast<size_t>(b.sms) * 8 * kWarpLines * kLineWords * 4));
  run<1, 32, kOneLine>(b, out);
  run<2, 32, kOneLine>(b, out);
  run<4, 32, kOneLine>(b, out);
  run<8, 32, kOneLine>(b, out);
  run<1, 16, kOneLine>(b, out);
  run<1, 8, kOneLine>(b, out);
  run<1, 32, kSpread>(b, out);
  run<4, 32, kSpread>(b, out);
  run<1, 32, kStore>(b, out);
  run<4, 32, kStore>(b, out);
  return 0;
}
