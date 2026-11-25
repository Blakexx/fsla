// measure/lab/experiments/carry_list.cu -- QUESTION: what does the LIBRARY's listing cost -- a warp turning a group's
// window of mask words into its live tokens in order, 32 places a round, counting inline -- by how many tokens are
// live, and where the words are two levels' ANDed?
// METHOD: this file compiles the library's device code (`carry/carry_list.cuh`: the listing; the select is this
// file's own, the form its record measured -- the library's supply lays its tokens). A CTA an SM of 16 warps, each warp a
// group's windows: a lane loads its word of each of two tables a window ahead, ANDs them, and the warp lists the
// window a round at a time into its 32 consecutive words of shared memory. Rows: 64 live tokens a window from one
// table (the other all ones); every token live; 64 live in both of two tables of a quarter live. The first windows'
// places and totals are checked against the host's.
// TIMELINE: no MMA. A window: a popcount and a running sum across the lanes (5 shuffles), then a round a 32 places --
// 5 shuffles of the running sums, 2 of the word and its start, the bit's place in the word, one store. Two global
// loads a lane a window, a window ahead. Live: the next window's word, the listing's four words.
// see docs/internals/tools/lab.md#carry-list
#include "lab.cuh"
#include "carry/carry_list.cuh"
#include "carry_select.cuh"

namespace carry_list {

namespace carry = fsla::carry::detail;
namespace ops = ::denseref::ops;
using carry_select::listed_token;

constexpr int kWords = 32, kWarps = 16, kWindow = kWords * 32;

template <int CHECK>
__global__ void __launch_bounds__(32 * kWarps, 1)
    listed(long long* times, const uint32_t* first, const uint32_t* second, uint32_t* out, int* totals, int* sunk,
           int windows) {
  __shared__ alignas(128) uint32_t ring[kWarps * kWords];
  const int warp = lab::warp_id(), lane = lab::lane_id();
  uint32_t* const mine = ring + kWords * warp + lane;
  const size_t group = (static_cast<size_t>(blockIdx.x) * kWarps + warp) * static_cast<size_t>(windows);
  uint32_t next = first[group * kWords + lane] & second[group * kWords + lane];
  //: every token a lane listed, folded: the timed kernel's output, so the listing is not work the compiler may drop
  uint32_t folded = 0;
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int w = 0; w < windows; ++w) {
    const carry::Listing s = carry::listing_of<kWords>(next);
    const size_t ahead = (group + static_cast<size_t>(w + 1 < windows ? w + 1 : w)) * kWords + lane;
    next = first[ahead] & second[ahead];
    //: a window's rounds: at least one, and on while any lane votes a place under the total
    int round = 0;
#pragma unroll 1
    do {
      const int token = listed_token<kWords>(s, round);
      *mine = static_cast<uint32_t>(token);
      folded = folded * 31u + static_cast<uint32_t>(token);
      if constexpr (CHECK != 0) out[(group + w) * kWindow + kWords * round + lane] = static_cast<uint32_t>(token);
      ++round;
    } while (::denseref::ops::lane_ballot(kWords * round < s.total) != 0u);
    if constexpr (CHECK != 0) {
      if (lane == 0) totals[group + w] = s.total;
    }
  }
  clock.stop(times);
  lab::sink(sunk, folded);
}

//: bit `x` of a table: live with one chance in `one_in`, by a hash of the place and the table's seed
inline bool live_at(uint64_t x, uint32_t seed, uint32_t one_in) {
  uint64_t h = (x + seed) * 0x9e3779b97f4a7c15ull;
  h ^= h >> 29;
  h *= 0xbf58476d1ce4e5b9ull;
  h ^= h >> 32;
  return h % one_in == 0;
}

}  // namespace carry_list

void run(lab::Bench& b, const std::string& name, uint32_t one_in_first, uint32_t one_in_second) {
  using namespace carry_list;
  constexpr int kTimed = 256, kChecked = 4;
  const size_t groups = static_cast<size_t>(b.sms) * kWarps, words = groups * kTimed * kWords;
  std::vector<uint32_t> first(words), second(words);
  long long live = 0;
  for (size_t i = 0; i < words; ++i) {
    uint32_t a = 0, c = 0;
    for (int bit = 0; bit < 32; ++bit) {
      a |= static_cast<uint32_t>(live_at(i * 32 + bit, 17, one_in_first)) << bit;
      c |= static_cast<uint32_t>(live_at(i * 32 + bit, 4099, one_in_second)) << bit;
    }
    first[i] = a;
    second[i] = c;
    live += __builtin_popcount(a & c);
  }
  uint32_t *dfirst = nullptr, *dsecond = nullptr, *dout = nullptr;
  int *dtotals = nullptr, *dsunk = nullptr;
  LAB_CHECK(cudaMalloc(&dfirst, words * 4));
  LAB_CHECK(cudaMalloc(&dsecond, words * 4));
  LAB_CHECK(cudaMalloc(&dout, groups * kChecked * kWindow * 4));
  LAB_CHECK(cudaMalloc(&dtotals, groups * kChecked * sizeof(int)));
  LAB_CHECK(cudaMalloc(&dsunk, groups * 32 * sizeof(int)));
  LAB_CHECK(cudaMemcpy(dfirst, first.data(), words * 4, cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMemcpy(dsecond, second.data(), words * 4, cudaMemcpyHostToDevice));

  //: the first windows of every group, checked: a group's windows are consecutive in the tables at `kChecked` a group
  std::vector<uint32_t> cfirst(groups * kChecked * kWords), csecond(cfirst.size());
  for (size_t g = 0; g < groups; ++g)
    for (int i = 0; i < kChecked * kWords; ++i) {
      cfirst[g * kChecked * kWords + i] = first[g * kTimed * kWords + i];
      csecond[g * kChecked * kWords + i] = second[g * kTimed * kWords + i];
    }
  uint32_t *dcfirst = nullptr, *dcsecond = nullptr;
  LAB_CHECK(cudaMalloc(&dcfirst, cfirst.size() * 4));
  LAB_CHECK(cudaMalloc(&dcsecond, csecond.size() * 4));
  LAB_CHECK(cudaMemcpy(dcfirst, cfirst.data(), cfirst.size() * 4, cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMemcpy(dcsecond, csecond.data(), csecond.size() * 4, cudaMemcpyHostToDevice));
  listed<1><<<b.sms, 32 * kWarps>>>(b.times, dcfirst, dcsecond, dout, dtotals, dsunk, kChecked);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> got(groups * kChecked * kWindow);
  std::vector<int> totals(groups * kChecked);
  LAB_CHECK(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost));
  LAB_CHECK(cudaMemcpy(totals.data(), dtotals, totals.size() * sizeof(int), cudaMemcpyDeviceToHost));
  long long wrong_places = 0, wrong_totals = 0, places = 0;
  for (size_t gw = 0; gw < groups * kChecked; ++gw) {
    int place = 0;
    for (int token = 0; token < kWindow; ++token) {
      const size_t word = gw * kWords + token / 32;
      if (((cfirst[word] & csecond[word]) >> (token % 32) & 1u) == 0) continue;
      wrong_places += got[gw * kWindow + place] != static_cast<uint32_t>(token);
      ++place;
    }
    places += place;
    wrong_totals += totals[gw] != place;
  }
  b.check(name, wrong_places == 0 && wrong_totals == 0,
          std::to_string(wrong_places) + " of " + std::to_string(places) + " places and " + std::to_string(wrong_totals)
              + " of " + std::to_string(groups * kChecked) + " totals differ");
  const double tokens_a_cta = static_cast<double>(live) / b.sms;
  const lab::Samples cycles =
      b.variant([&] { listed<0><<<b.sms, 32 * kWarps>>>(b.times, dfirst, dsecond, dout, dtotals, dsunk, kTimed); },
                [&] { return b.span(kWarps) / tokens_a_cta; });
  b.record(name, "cycles a listed token a CTA", cycles);
  for (void* p : {static_cast<void*>(dfirst), static_cast<void*>(dsecond), static_cast<void*>(dout),
                  static_cast<void*>(dtotals), static_cast<void*>(dsunk), static_cast<void*>(dcfirst),
                  static_cast<void*>(dcsecond)})
    LAB_CHECK(cudaFree(p));
}

int main(int argc, char** argv) {
  lab::Bench b(argc, argv);
  if (b.expect_only) return 0;
  run(b, "a window of 64 live tokens, one level's words", 16, 1);
  run(b, "every token of a window live", 1, 1);
  run(b, "a window of 64 live tokens, two levels' words ANDed, each a quarter live", 4, 4);
  return 0;
}
