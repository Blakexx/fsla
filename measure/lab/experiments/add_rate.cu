// measure/lab/experiments/add_rate.cu -- QUESTION: the carry's output is a line added to global memory a token a
// pair of boxes (`carry_window`). What holds an add to global memory: a limit a warp, the lanes an
// instruction carries, the lines it touches, the add's type, or its being atomic? METHOD: a CTA an SM of 1, 2, 4 or 8
// warps; each warp issues 16 instructions a trip to
// its own rows of global memory, from registers that never change. ONE LINE: an instruction's lanes add f16 pairs
// into one 128-byte line, by 32, 16 or 8 lanes. SPREAD: its 32 lanes each add into a different line. STORE: the
// one-line form as a plain 4-byte store a lane, no add. INTEGER AND fp32: the one-line form by a 32-bit integer
// add and by an fp32 add, and 64-bit integer adds, 8 bytes a lane, two lines an instruction. VECTOR: one add of
// four f16 pairs a lane, 16 bytes -- 32 lanes an instruction are FOUR lines, 8 lanes one -- which asks whether an
// add's price is its instruction's or its lines'. ON A FEW SMs: the one-line form with CTAs on 8 and on 32 of the
// card's SMs, which asks whether the limit is an SM's own or the card's shared among them. Every row's sums are
// checked exactly.
// TIMELINE: no load. A warp's trip is 16 instructions, each from a register set before the clock and to an
// address that is a constant off the warp's base: nothing waits on anything. Live: one word a thread, a vector
// add's four operands that word.
// see docs/internals/tools/lab.md#add-rate
#include "lab.cuh"

namespace add_rate {

constexpr int kTrip = 16, kLineWords = 32, kWarpLines = kTrip * 32;
enum Form : int { kOneLine = 0, kSpread, kStore, kU32, kF32, kU64, kVector };

__device__ __forceinline__ uint32_t f16_of(int v) { return __half_as_ushort(__float2half(static_cast<float>(v))); }

template <int WARPS, int LANES, int FORM>
__global__ void __launch_bounds__(32 * WARPS, 1) add(long long* times, uint32_t* out, int rounds) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  //: a pair of ones: a word's sum after `rounds` adds is exact in f16 to 2,048
  const uint32_t v = lab::held(f16_of(1) | f16_of(1) << 16);
  //: instruction `i` of a trip: line `i` of the warp's (one line), or line `32 i + lane` (spread), or -- a vector
  //: add's lane its own 16 bytes -- lines `4 i` to `4 i + 3`, 8 lanes a line
  uint32_t* const own = out + (static_cast<long long>(blockIdx.x) * 8 + warp) * (kWarpLines * kLineWords)
                        + (FORM == kSpread ? lane * kLineWords : 0) + (FORM == kVector ? 4 * lane : lane);
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
#pragma unroll
    for (int i = 0; i < kTrip; ++i) {
      uint32_t* const to = own + i * (FORM == kSpread ? 32 : FORM == kVector ? 4 : FORM == kU64 ? 2 : 1) * kLineWords;
      if (lane < LANES) {
        if constexpr (FORM == kStore) *to = v + r;
        else if constexpr (FORM == kU32) lab::red_add_u32(to, 1u);
        else if constexpr (FORM == kF32) lab::red_add_f32(reinterpret_cast<float*>(to), 1.0f);
        //: 8 bytes a lane: an instruction is two lines, its lanes' words `2 lane` and the next
        else if constexpr (FORM == kU64) lab::red_add_u64(reinterpret_cast<uint64_t*>(to + lane), 1ull);
        else if constexpr (FORM == kVector) lab::red_add_f16x2_v4(to, v, v, v, v);
        else lab::red_add_f16x2(to, v);
      }
    }
  }
  clock.stop(times);
}

}  // namespace add_rate

template <int WARPS, int LANES, int FORM>
void run(lab::Bench& b, uint32_t* out, int ctas = 0) {
  using namespace add_rate;
  constexpr int kCheckRounds = 64, kRounds = 2048;
  const std::string name = std::string(FORM == kOneLine ? "f16 adds into one line an instruction"
                                       : FORM == kSpread ? "f16 adds, a lane a line of its own"
                                       : FORM == kStore ? "plain stores into one line an instruction"
                                       : FORM == kU32 ? "32-bit integer adds into one line an instruction"
                                       : FORM == kF32 ? "fp32 adds into one line an instruction"
                                       : FORM == kVector ? "vector f16 adds, 16 bytes a lane, 8 lanes a line"
                                       : "64-bit integer adds, two lines an instruction")
                           + ", " + std::to_string(LANES) + " lanes, " + std::to_string(WARPS)
                           + (WARPS == 1 ? " warp" : " warps")
                           + (ctas > 0 ? ", on " + std::to_string(ctas) + " SMs" : "");
  const int grid = ctas > 0 ? ctas : b.sms;
  const auto kernel = add<WARPS, LANES, FORM>;
  const size_t words = static_cast<size_t>(b.sms) * 8 * kWarpLines * kLineWords;
  LAB_CHECK(cudaMemset(out, 0, words * 4));
  kernel<<<grid, 32 * WARPS>>>(b.times, out, kCheckRounds);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> got(words);
  LAB_CHECK(cudaMemcpy(got.data(), out, words * 4, cudaMemcpyDeviceToHost));
  //: a written word holds its form's value after the rounds -- the sum in both f16 halves, a store's last value,
  //: an integer's or an fp32's count, a 64-bit count's low word and nothing in its high; every other word nothing
  long long wrong = 0;
  for (size_t w = 0; w < words; ++w) {
    const int lane = static_cast<int>(w % kLineWords), line = static_cast<int>(w / kLineWords) % kWarpLines,
              warp = static_cast<int>(w / (static_cast<size_t>(kWarpLines) * kLineWords)) % 8,
              cta = static_cast<int>(w / (static_cast<size_t>(8) * kWarpLines * kLineWords));
    //: a vector add's word `lane` of line `l` is its lane `8 (l % 4) + lane / 4`'s
    const bool written = cta < grid
                         && (FORM == kVector ? warp < WARPS && line < 4 * kTrip && 8 * (line % 4) + lane / 4 < LANES
                                             : warp < WARPS && lane < LANES
                                                   && (FORM == kSpread ? line % 32 == lane : line < (FORM == kU64 ? 2 * kTrip : kTrip)));
    uint32_t want = 0;
    if (written) {
      if (FORM == kStore) {
        want = (0x3c00u | 0x3c00u << 16) + (kCheckRounds - 1);
      } else if (FORM == kU32) {
        want = kCheckRounds;
      } else if (FORM == kF32) {
        const float f = static_cast<float>(kCheckRounds);
        std::memcpy(&want, &f, 4);
      } else if (FORM == kU64) {
        //: a lane's count in its low word, nothing in its high
        want = lane % 2 == 0 ? kCheckRounds : 0;
      } else {
        //: 64 as an f16: 2^6, exponent field 21
        want = 0x5400u | 0x5400u << 16;
      }
    }
    wrong += got[w] != want;
  }
  b.check(name, wrong == 0, std::to_string(wrong) + " of " + std::to_string(words) + " words differ");
  const lab::Samples cycles = b.variant([&] { kernel<<<grid, 32 * WARPS>>>(b.times, out, kRounds); },
                                        [&] { return b.span(WARPS, grid) / (static_cast<double>(kRounds) * kTrip); });
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
  for (int form : {kU32, kF32, kU64})
    for (int warps : {1, 4}) b.expect(lab::kernel("add", {warps, 32, form}), "RED", ">=", kTrip);
  for (int warps : {1, 4, 8}) b.expect(lab::kernel("add", {warps, 32, kVector}), "RED", ">=", kTrip);
  for (int warps : {1, 4}) {
    b.expect(lab::kernel("add", {warps, 8, kVector}), "RED", ">=", kTrip);
    b.expect(lab::kernel("add", {warps, 8, kVector}), "LOOP_CONVERGENCE", "<=", 2 * kTrip + 2);
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
  run<1, 32, kU32>(b, out);
  run<4, 32, kU32>(b, out);
  run<1, 32, kF32>(b, out);
  run<4, 32, kF32>(b, out);
  run<1, 32, kU64>(b, out);
  run<4, 32, kU64>(b, out);
  run<1, 32, kVector>(b, out);
  run<4, 32, kVector>(b, out);
  run<8, 32, kVector>(b, out);
  run<1, 8, kVector>(b, out);
  run<4, 8, kVector>(b, out);
  run<4, 32, kOneLine>(b, out, 8);
  run<4, 32, kOneLine>(b, out, 32);
  run<8, 32, kOneLine>(b, out, 8);
  return 0;
}
