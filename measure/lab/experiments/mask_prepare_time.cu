// measure/lab/experiments/mask_prepare_time.cu -- QUESTION: what does the LIBRARY's mask-preparation pass cost a token on
// a mask it reads once, at each of the nine group sizes its density byte can select -- the three the lab's own kernel
// measured (`mask_prep`: 1, 4 and 16 digits), 2 and 8, and the four whose lanes OR by warp shuffle (32 to 256
// digits)? METHOD: this file compiles the library's device code (`mask_prepare/mask_prepare_kernel.cuh`) and puts a
// clock around `mask_prepare_cta`, the work of one CTA of the library's grid: 1,024 threads over one 256-digit
// level-side of 4 to 5 live digits a token, a window the hopper card's (1,024 tokens). A round is the CTA's own windows (4 at a group of
// one digit, 8 at 2, 16 at 4, 32 at 8, 64 at 16 and above); a call is the rounds that cover 256 windows a CTA, each round fresh windows.
// The density byte selects the group as it does in the library. Every word and total of the first round is
// checked against the host's, in the library's layout.
// TIMELINE: no MMA. The library's walk: a thread holds its loads ahead (8 words' at up to four digits, 4 at eight, 2 at more), a trip
// finishes the held words and issues the loads that many words on; at 32 digits and above a word's finish is a warp
// shuffle across the group's lanes. Live: the held loads (8 words, pairs or lines; 4 words of 2 lines; or 2 words of 4 lines) and the running total.
// see docs/internals/tools/lab.md#mask-prepare-time
#include "lab.cuh"
#include "mask_prepare/mask_prepare_kernel.cuh"

namespace mask_prepare_time {

namespace pass = fsla::mask_prepare::detail;
namespace plan = fsla::plan;

constexpr int kWindow = plan::window_of_cc(plan::kHopperCc), kWords = kWindow / 32, kWidthBits = 8, kDigits = 1 << kWidthBits;

//: token `x` has 4 or 5 live digits, a stride apart from a start its own
constexpr int live_of(uint32_t x) { return 4 + (x % 3 == 0); }
constexpr uint32_t digit_of(uint32_t x, int i) { return (x * 5u + (x >> 3) + static_cast<uint32_t>(i) * 37u) & 255u; }

//: THE LIBRARY'S CTA, `rounds` rounds of it, round `r` the CTA `stride r` on: the clock is the lab's, the work the library's
__global__ void __launch_bounds__(pass::kThreads, 1)
    timed(long long* times, const __grid_constant__ pass::Args a, const uint8_t* density, uint8_t* workspace, int rounds,
          int stride) {
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int round = 0; round < rounds; ++round) {
    pass::mask_prepare_cta<kWords>(a, density, workspace, static_cast<int>(blockIdx.x) + round * stride);
  }
  clock.stop(times);
}

}  // namespace mask_prepare_time

int main(int argc, char** argv) {
  using namespace mask_prepare_time;
  lab::Bench b(argc, argv);
  //: the library's CTA dispatches once -- its level-side, its switch on the group, a surplus CTA's return -- outside its own
  //: loops; here that dispatch sits inside the lab's loop over rounds, which the library's kernel does not have
  //: the switch has an arm a group size, nine of them
  b.expect("mask_prepare_time::timed", "LOOP_CONVERGENCE", "<=", 26);
  if (b.expect_only) return 0;
  //: every group covers the same windows a call, 256 a CTA
  constexpr int kCtaWindows = 256;
  const int ctas = b.sms;
  const size_t windows = static_cast<size_t>(ctas) * kCtaWindows, mask_words = windows * kWords * kDigits;
  const size_t block = static_cast<size_t>(plan::prepared_block_bytes(kWidthBits, kWords)), bytes = windows * block;
  uint32_t* dmask = nullptr;
  uint8_t *dwork = nullptr, *ddensity = nullptr;
  LAB_CHECK(cudaMalloc(&dmask, mask_words * 4));
  LAB_CHECK(cudaMalloc(&dwork, bytes));
  LAB_CHECK(cudaMalloc(&ddensity, 2));
  //: the mask as the prefill is handed it: bit `x % 32` of word `x / 32` at a digit's column
  std::vector<uint32_t> mask(mask_words, 0);
  for (size_t x = 0; x < windows * kWindow; ++x) {
    const int n = live_of(static_cast<uint32_t>(x));
    for (int i = 0; i < n; ++i) mask[(x / 32) * kDigits + digit_of(static_cast<uint32_t>(x), i)] |= 1u << (x % 32);
  }
  LAB_CHECK(cudaMemcpy(dmask, mask.data(), mask_words * 4, cudaMemcpyHostToDevice));
  //: one level-side, slot 0: its blocks from the workspace's start, the grid its own (no other slot holds a CTA)
  pass::Args args{};
  args.mask[0] = dmask;
  args.width_bits[0] = kWidthBits;
  args.heads[0] = 1;
  args.words = static_cast<int>(windows * kWords);
  args.windows = static_cast<int>(windows);
  args.levels = 1;
  for (int s = 1; s <= pass::kSides; ++s) args.first_cta[s] = INT32_MAX;
  for (int member = 0; member < plan::kGroupSizes; ++member) {
    const int gb = plan::group_size_bits(member), g = 1 << gb, groups = kDigits >> gb;
    //: a density byte whose group is `g`: `byte + 1` 256ths of 256 digits is `byte + 1` digits; 255 would be dense
    const uint8_t byte = static_cast<uint8_t>(g == kDigits ? 200 : g - 1);
    if (plan::prefill_group_bits(byte + 1, kWidthBits) != gb) {
      std::fprintf(stderr, "the density byte %d does not select a group of %d digits\n", byte, g);
      return 1;
    }
    const uint8_t density[2] = {byte, 255};
    LAB_CHECK(cudaMemcpy(ddensity, density, 2, cudaMemcpyHostToDevice));
    //: a thread's digits, so a CTA's windows a round and its rounds a call
    const int unit = g < 16 ? g : 16, per = pass::kThreads / (kDigits / unit), rounds = kCtaWindows / per;
    const std::string name = "groups of " + std::to_string(g) + (g == 1 ? " digit" : " digits");
    //: the first round's windows, checked in the library's layout: every group's words and the window's total
    const size_t checked = static_cast<size_t>(ctas) * per;
    LAB_CHECK(cudaMemset(dwork, 0xff, checked * block));
    timed<<<ctas, pass::kThreads>>>(b.times, args, ddensity, dwork, 1, 0);
    LAB_CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> got(checked * block);
    LAB_CHECK(cudaMemcpy(got.data(), dwork, checked * block, cudaMemcpyDeviceToHost));
    long long wrong_words = 0, wrong_totals = 0;
    for (size_t w = 0; w < checked; ++w) {
      const uint8_t* const at = got.data() + w * block;
      const uint16_t* const totals = reinterpret_cast<const uint16_t*>(at);
      const uint32_t* const words = reinterpret_cast<const uint32_t*>(at + plan::prepared_words_at(groups));
      for (int q = 0; q < groups; ++q) {
        uint32_t live = 0;
        for (int j = 0; j < kWords; ++j) {
          uint32_t want = 0;
          for (int d = 0; d < g; ++d) want |= mask[(w * kWords + j) * kDigits + q * g + d];
          wrong_words += g > 1 && words[j * groups + q] != want;
          live += static_cast<uint32_t>(__builtin_popcount(want));
        }
        wrong_totals += totals[q] != live;
      }
    }
    b.check(name, wrong_words == 0 && wrong_totals == 0,
            std::to_string(wrong_words) + " words and " + std::to_string(wrong_totals) + " totals differ over " + std::to_string(checked) + " windows of "
                + std::to_string(groups) + " groups");
    const lab::Samples cycles = b.variant(
        [&] { timed<<<ctas, pass::kThreads>>>(b.times, args, ddensity, dwork, rounds, ctas); },
        [&] { return b.span(pass::kThreads >> 5) / rounds / (per * kWindow); });
    const std::string row = name + ", the mask once-read";
    b.unchecked(row, "the row above checks this kernel's first round; this one is its time");
    b.record(row, "SM-cycles a token", cycles);
  }
  return 0;
}
