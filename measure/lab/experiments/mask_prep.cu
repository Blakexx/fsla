// measure/lab/experiments/mask_prep.cu -- QUESTION: what does it cost a token to turn a sparse level-side's mask, as
// the prefill is handed it -- a word of 32 tokens a digit -- into what the gather and the carry read blind: each
// GROUP's word (its digits' words ORed) and the group's live tokens of the window before that word; and does that
// cost move with the group's size, or with how dense the mask is? METHOD: a CTA of 1,024 threads, a thread a
// (window, group) pair of one 256-digit level-side; a window is 1,024 tokens, 32 words a digit. A thread walks its
// group's 32 words in order with a running count: a word's stores are the group's word (where a group is more than
// a digit: at one digit the word is the mask's own, and only the count is written) and the count before it; after
// the last word, the window's total. A group of 1, 4 or 16 digits, a kernel each; a mask of 4 to 5 live digits a
// token and one of 32. Each is timed with its windows in cache -- a call's rounds over the same windows -- and
// ONCE-READ, a round fresh windows. Every word, count and total of the first round is checked against the host's.
// TIMELINE: no MMA. A thread holds two words' loads at fixed registers: a pass of the walk finishes one word -- its
// stores and its popcount -- and issues the load two words on into the registers it left, then does the same for
// the other. A load is kept as it arrived (a group's lines are ORed when its word is finished), so nothing reads a
// load in the pass that issued it. Live: a group's lines of two words (2, 8 or 32 registers) and the count.
// see docs/internals/tools/lab.md#mask-prep
#include "lab.cuh"

namespace mask_prep {

constexpr int kWindow = 1024, kWords = kWindow / 32, kDigits = 256, kThreads = 1024;

//: A CELL: token `x` has `live_of` live digits, a stride apart from a start its own
constexpr int live_of(int cell, uint32_t x) { return cell == 0 ? 4 + (x % 3 == 0) : 32; }
constexpr uint32_t digit_of(int cell, uint32_t x, int i) {
  return (x * 5u + (x >> 3) + static_cast<uint32_t>(i) * (cell == 0 ? 37u : 7u)) & 255u;
}

//: A GROUP'S WORD AS LOADED: its digits' words, 16 bytes a load where the group is 4 digits or more
template <int G>
struct Raw {
  uint4 line[G / 4];
};

template <>
struct Raw<1> {
  uint32_t word;
};

template <int G>
__device__ __forceinline__ void load(Raw<G>& raw, const uint32_t* at) {
  if constexpr (G == 1) {
    raw.word = at[0];
  } else {
    const uint4* const lines = reinterpret_cast<const uint4*>(at);
#pragma unroll
    for (int i = 0; i < G / 4; ++i) raw.line[i] = lines[i];
  }
}

//: THE GROUP'S WORD: a token's bit set where it is live on any digit of the group
template <int G>
__device__ __forceinline__ uint32_t word_of(const Raw<G>& raw) {
  if constexpr (G == 1) {
    return raw.word;
  } else {
    uint32_t w = 0;
#pragma unroll
    for (int i = 0; i < G / 4; ++i) w |= raw.line[i].x | raw.line[i].y | raw.line[i].z | raw.line[i].w;
    return w;
  }
}

//: A CTA's WINDOWS a round: `kThreads` over the level's groups of them, a thread a (window, group); round `r` is
//: the windows `(block + r stride) per` on
template <int G>
__global__ void __launch_bounds__(kThreads, 1)
    prepare(long long* times, const uint32_t* masks, uint32_t* words, uint16_t* counts, uint16_t* totals, int rounds,
            int stride) {
  constexpr int kGroups = kDigits / G, kPer = kThreads / kGroups;
  const int tid = threadIdx.x, group = tid % kGroups, local = tid / kGroups;
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int round = 0; round < rounds; ++round) {
    const long long window =
        (static_cast<long long>(blockIdx.x) + static_cast<long long>(round) * stride) * kPer + local;
    const uint32_t* at = masks + window * (kWords * kDigits) + group * G;
    [[maybe_unused]] uint32_t* word = words + window * (kWords * kGroups) + group;
    uint16_t* count = counts + window * (kWords * kGroups) + group;
    //: TWO WORDS' LOADS HELD: the walk's first two, issued before any is read
    Raw<G> a, b;
    load<G>(a, at);
    load<G>(b, at + kDigits);
    uint32_t before = 0;
#pragma unroll 1
    for (int pair = 0; pair < kWords / 2; ++pair) {
      //: A WORD FINISHED -- its stores, its count -- and the load two words on issued into the registers it left
      const uint32_t first = word_of<G>(a);
      if constexpr (G > 1) word[0] = first;
      count[0] = static_cast<uint16_t>(before);
      before += static_cast<uint32_t>(__popc(first));
      if (pair + 1 < kWords / 2) load<G>(a, at + 2 * kDigits);
      const uint32_t second = word_of<G>(b);
      if constexpr (G > 1) word[kGroups] = second;
      count[kGroups] = static_cast<uint16_t>(before);
      before += static_cast<uint32_t>(__popc(second));
      if (pair + 1 < kWords / 2) load<G>(b, at + 3 * kDigits);
      at += 2 * kDigits;
      word += 2 * kGroups;
      count += 2 * kGroups;
    }
    totals[window * kGroups + group] = static_cast<uint16_t>(before);
  }
  clock.stop(times);
}

}  // namespace mask_prep

int main(int argc, char** argv) {
  using namespace mask_prep;
  lab::Bench b(argc, argv);
  if (b.expect_only) return 0;
  //: every kernel covers the same windows a call, 256 a CTA, in rounds of its own 4, 16 or 64 windows
  constexpr int kCtaWindows = 256;
  const int ctas = b.sms;
  const size_t windows = static_cast<size_t>(ctas) * kCtaWindows, mask_words = windows * kWords * kDigits;
  uint32_t *dmask = nullptr, *dwords = nullptr;
  uint16_t *dcounts = nullptr, *dtotals = nullptr;
  LAB_CHECK(cudaMalloc(&dmask, mask_words * 4));
  LAB_CHECK(cudaMalloc(&dwords, mask_words / 4 * 4));
  LAB_CHECK(cudaMalloc(&dcounts, mask_words * 2));
  LAB_CHECK(cudaMalloc(&dtotals, windows * kDigits * 2));
  for (int cell = 0; cell < 2; ++cell) {
    //: the mask as the prefill is handed it: bit `x % 32` of word `x / 32` at a digit's column
    std::vector<uint32_t> mask(mask_words, 0);
    for (size_t x = 0; x < windows * kWindow; ++x) {
      const int n = live_of(cell, static_cast<uint32_t>(x));
      for (int i = 0; i < n; ++i) mask[(x / 32) * kDigits + digit_of(cell, static_cast<uint32_t>(x), i)] |= 1u << (x % 32);
    }
    LAB_CHECK(cudaMemcpy(dmask, mask.data(), mask_words * 4, cudaMemcpyHostToDevice));
    const auto run = [&](auto kernel, int g) {
      const int groups = kDigits / g, per = kThreads / groups, rounds = kCtaWindows / per;
      const std::string name = std::string(cell == 0 ? "4 to 5 live digits a token" : "32 live digits a token") + ", groups of "
                               + std::to_string(g) + (g == 1 ? " digit, the counts alone" : " digits, the groups' words and counts");
      //: the first round's windows, checked: every group's words, the count before each and the window's total
      const size_t checked = static_cast<size_t>(ctas) * per, slots = checked * kWords * groups;
      LAB_CHECK(cudaMemset(dwords, 0xff, slots * 4));
      LAB_CHECK(cudaMemset(dcounts, 0xff, slots * 2));
      LAB_CHECK(cudaMemset(dtotals, 0xff, checked * groups * 2));
      kernel<<<ctas, kThreads>>>(b.times, dmask, dwords, dcounts, dtotals, 1, 0);
      LAB_CHECK(cudaDeviceSynchronize());
      std::vector<uint32_t> got_words(slots);
      std::vector<uint16_t> got_counts(slots), got_totals(checked * groups);
      LAB_CHECK(cudaMemcpy(got_words.data(), dwords, slots * 4, cudaMemcpyDeviceToHost));
      LAB_CHECK(cudaMemcpy(got_counts.data(), dcounts, slots * 2, cudaMemcpyDeviceToHost));
      LAB_CHECK(cudaMemcpy(got_totals.data(), dtotals, checked * groups * 2, cudaMemcpyDeviceToHost));
      long long wrong_words = 0, wrong_counts = 0, wrong_totals = 0;
      for (size_t w = 0; w < checked; ++w)
        for (int q = 0; q < groups; ++q) {
          uint32_t before = 0;
          for (int j = 0; j < kWords; ++j) {
            uint32_t want = 0;
            for (int d = 0; d < g; ++d) want |= mask[(w * kWords + j) * kDigits + q * g + d];
            const size_t slot = (w * kWords + j) * groups + q;
            wrong_words += g > 1 && got_words[slot] != want;
            wrong_counts += got_counts[slot] != before;
            before += static_cast<uint32_t>(__builtin_popcount(want));
          }
          wrong_totals += got_totals[w * groups + q] != before;
        }
      b.check(name, wrong_words == 0 && wrong_counts == 0 && wrong_totals == 0,
              std::to_string(wrong_words) + " of " + std::to_string(g > 1 ? slots : 0) + " words, " + std::to_string(wrong_counts)
                  + " of " + std::to_string(slots) + " counts and " + std::to_string(wrong_totals) + " of "
                  + std::to_string(checked * groups) + " totals differ");
      for (int fresh = 0; fresh < 2; ++fresh) {
        //: the windows in cache: a call's rounds are the same windows; ONCE-READ: a round fresh windows
        const lab::Samples cycles = b.variant(
            [&] { kernel<<<ctas, kThreads>>>(b.times, dmask, dwords, dcounts, dtotals, rounds, fresh ? ctas : 0); },
            [&] { return b.span(kThreads >> 5) / rounds / (per * kWindow); });
        const std::string timed = name + (fresh ? ", the mask once-read" : ", the mask in cache");
        b.unchecked(timed, "the row above checks this kernel's first round; this one is its time");
        b.record(timed, "SM-cycles a token", cycles);
      }
    };
    run(prepare<1>, 1);
    run(prepare<4>, 4);
    run(prepare<16>, 16);
  }
  return 0;
}
