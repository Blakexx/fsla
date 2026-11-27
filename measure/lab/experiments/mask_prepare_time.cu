// measure/lab/experiments/mask_prepare_time.cu -- QUESTION: what does the LIBRARY's mask-preparation pass cost a token, now that it writes a side's LEAF GROUPS' mask
// words and counts -- one listed level of 256 digits at each of the eight group sizes it forms (256 down to 2 leaf groups), two listed levels at groups of 4 digits
// (4,096 leaf groups of 16 leaves, the most the shape allows), and a side that lists no level (one leaf group, the call's own tokens)?
// METHOD: this file compiles the library's device code (`mask_prepare/mask_prepare_kernel.cuh`) and puts a clock around `mask_prepare_cta`, the work of one CTA of
// the library's grid: 1,024 threads over one (batch-head, window) of a side, a window the hopper card's (1,024 tokens), each token on 4 to 5 digits of a listed level.
// A call is `rounds` rounds a CTA, each round a fresh window. The density bytes select the groups as they do in the library. Every line and count of a separate
// RAGGED call (it ends inside a word, three words short of its last window) is checked against the host's, in the library's layout.
// TIMELINE: no MMA. The first warp's carve and its hand; each listed level's walk on its quarter of the threads (loads held ahead, a group's lanes ORed by warp
// shuffle from 32 digits, a word stored to shared memory a mask word); a CTA barrier; then a thread a leaf group a trip: a shared load a listed level a word, the
// AND, a popcount, a 16-byte store every four words. Live: the walk's held loads; then the leaf group's fields and four words.
// see docs/internals/tools/lab.md#mask-prepare-time
#include "lab.cuh"
#include "mask_prepare/mask_prepare_kernel.cuh"

namespace mask_prepare_time {

namespace pass = fsla::mask_prepare::detail;
namespace plan = fsla::plan;

constexpr int kWindow = plan::window_of_cc(plan::kHopperCc), kWords = kWindow / 32, kWidthBits = 8, kDigits = 1 << kWidthBits;
constexpr int kLevels = 2;
//: a call's rounds a CTA at most
constexpr int kRoundsMost = 16;

//: token `x` has 4 or 5 live digits on level 0 and 4 on level 1, each a stride apart from a start its own
constexpr int live_of(int level, uint32_t x) { return level == 0 ? 4 + (x % 3 == 0) : 4; }
constexpr uint32_t digit_of(int level, uint32_t x, int i) {
  return level == 0 ? (x * 5u + (x >> 3) + static_cast<uint32_t>(i) * 37u) & 255u
                    : (x * 11u + (x >> 2) + static_cast<uint32_t>(i) * 53u) & 255u;
}

//: THE LIBRARY'S CTA, `rounds` rounds of it, round `r` the CTA `stride r` on: the clock is the lab's, the work the library's
__global__ void __launch_bounds__(pass::kThreads, 1)
    timed(long long* times, const __grid_constant__ pass::Args a, const uint8_t* density, uint8_t* workspace, int rounds,
          int stride) {
  //: the words the library's kernel takes from its layout: the handed leaf groups one set a round (no round's is written under another's readers), the formed words one tile
  __shared__ int handed[kRoundsMost * pass::kHandedCount];
  __shared__ __align__(16) uint32_t formed[pass::kLevels * pass::kFormedGroups * kWords];
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int round = 0; round < rounds; ++round) {
    pass::mask_prepare_cta<kWords>(a, density, workspace, static_cast<int>(blockIdx.x) + round * stride,
                                   handed + round * pass::kHandedCount, formed);
  }
  clock.stop(times);
}

//: A ROW'S CALL: the read side's density byte a level, which of its levels have a mask, and its rounds a call
struct Row {
  uint8_t byte[kLevels];
  bool masked[kLevels];
  int rounds;
};

//: the host's lines and counts of the read side of a call of `tokens` tokens, against `got` in the library's layout: the lines and the counts that differ
void compare(const Row& row, const std::vector<uint32_t>* mask, long long tokens, const std::vector<uint8_t>& got, long long* wrong) {
  const int widths[kLevels] = {kWidthBits, kWidthBits}, whole[kLevels] = {256, 256};
  const int read[1][plan::kCarveMaxLevels] = {{row.byte[0] + 1, row.byte[1] + 1}};
  const plan::PrefillCarve c = plan::prefill_carve<plan::PrefillPolicy>(kLevels, widths, whole, read, 1);
  const plan::LeafGroups g = plan::leaf_groups(c, true, 0);
  const long long words = (tokens + 31) / 32, windows = (words + kWords - 1) / kWords;
  const size_t block = static_cast<size_t>(plan::leaf_block_bytes(g.bits, kWords));
  wrong[0] = wrong[1] = 0;
  for (long long w = 0; w < windows; ++w) {
    const uint8_t* const at = got.data() + static_cast<size_t>(w) * block;
    const uint16_t* const counts = reinterpret_cast<const uint16_t*>(at);
    const uint32_t* const lines = reinterpret_cast<const uint32_t*>(at + plan::leaf_lines_at(g.bits, kWords));
    for (int leaf = 0; leaf < (1 << g.bits); ++leaf) {
      int group[plan::kCarveMaxLevels] = {};
      plan::leaf_group_levels(g, leaf, group);
      uint32_t live = 0;
      for (int j = 0; j < kWords; ++j) {
        const long long word = w * kWords + j;
        //: the call's own tokens of the word, then each listed level's group's digits ORed
        uint32_t want = word < tokens / 32 ? ~0u : word == tokens / 32 ? (1u << (tokens % 32)) - 1u : 0u;
        for (int l = 0; l < kLevels; ++l) {
          const int gb = c.read_group_bits[0][l];
          if (gb == kWidthBits || !row.masked[l]) continue;
          uint32_t any = 0;
          for (int d = 0; d < (1 << gb) && word < words; ++d) {
            any |= mask[l][static_cast<size_t>(word) * kDigits + (static_cast<size_t>(group[l]) << gb) + d];
          }
          want &= any;
        }
        wrong[0] += lines[static_cast<size_t>(leaf) * kWords + j] != want;
        live += static_cast<uint32_t>(__builtin_popcount(want));
      }
      wrong[1] += counts[leaf] != live;
    }
  }
}

}  // namespace mask_prepare_time

int main(int argc, char** argv) {
  using namespace mask_prepare_time;
  lab::Bench b(argc, argv);
  //: the library's CTA dispatches once -- its first warp's carve and hand, a level's quarter of the threads switching on its group -- outside its own loops; here that
  //: dispatch sits inside the lab's loop over rounds, which the library's kernel does not have
  //: the switch has an arm a group size the pass forms, eight of them; the first warp's branch and its join are two more
  b.expect("mask_prepare_time::timed", "LOOP_CONVERGENCE", "<=", 26);
  if (b.expect_only) return 0;
  const int ctas = b.sms;
  const size_t windows = static_cast<size_t>(ctas) * kRoundsMost, mask_words = windows * kWords * kDigits;
  //: the masks as the prefill is handed them: bit `x % 32` of word `x / 32` at a digit's column
  std::vector<uint32_t> mask[kLevels];
  uint32_t* dmask[kLevels] = {};
  for (int l = 0; l < kLevels; ++l) {
    mask[l].assign(mask_words, 0);
    for (size_t x = 0; x < windows * kWindow; ++x) {
      const int n = live_of(l, static_cast<uint32_t>(x));
      for (int i = 0; i < n; ++i) mask[l][(x / 32) * kDigits + digit_of(l, static_cast<uint32_t>(x), i)] |= 1u << (x % 32);
    }
    LAB_CHECK(cudaMalloc(&dmask[l], mask_words * 4));
    LAB_CHECK(cudaMemcpy(dmask[l], mask[l].data(), mask_words * 4, cudaMemcpyHostToDevice));
  }
  //: the groups of one listed level, by the density byte that selects each, and the two rows more
  std::vector<Row> rows;
  std::vector<std::string> names;
  for (int gb = 0; gb < kWidthBits; ++gb) {
    const int g = 1 << gb;
    //: a density byte whose group is `g`: `byte + 1` 256ths of 256 digits is `byte + 1` digits
    if (plan::prefill_group_bits(g, kWidthBits) != gb) {
      std::fprintf(stderr, "the density byte %d does not select a group of %d digits\n", g - 1, g);
      return 1;
    }
    names.push_back("one listed level, groups of " + std::to_string(g) + (g == 1 ? " digit" : " digits") + ", "
                    + std::to_string(kDigits >> gb) + " leaf groups");
    rows.push_back({{static_cast<uint8_t>(g - 1), 255}, {true, false}, kRoundsMost});
  }
  names.push_back("two listed levels, groups of 4 digits, 4096 leaf groups");
  rows.push_back({{3, 3}, {true, true}, 4});
  names.push_back("no listed level, one leaf group");
  rows.push_back({{255, 255}, {false, false}, kRoundsMost});

  //: the workspace at the most a row needs: every row's read side at its shape's most leaf groups, the write side one leaf group
  const int widths[kLevels] = {kWidthBits, kWidthBits};
  size_t bytes = 0;
  for (const Row& row : rows) {
    const int masked[2] = {(row.masked[0] ? 1 : 0) | (row.masked[1] ? 2 : 0), 0};
    const std::int64_t heads[2] = {1, 1};
    const size_t need = static_cast<size_t>(
        plan::prepared_layout(kLevels, widths, masked, heads, static_cast<std::int64_t>(ctas) * row.rounds, kWords).bytes);
    bytes = need > bytes ? need : bytes;
  }
  uint8_t *dwork = nullptr, *ddensity = nullptr;
  LAB_CHECK(cudaMalloc(&dwork, bytes));
  LAB_CHECK(cudaMalloc(&ddensity, 2 * kLevels));

  //: THE CHECKED CALL IS RAGGED: `ctas` windows less three words and five tokens, so its last window's last words are past the mask and its last word is cut
  const long long checked_tokens = static_cast<long long>(ctas) * kWindow - 3 * 32 - 5;
  for (size_t i = 0; i < rows.size(); ++i) {
    const Row& row = rows[i];
    const std::string& name = names[i];
    //: row 0 the read side's, the timed side; the write side whole on both levels
    const uint8_t density[2 * kLevels] = {row.byte[0], row.byte[1], 255, 255};
    LAB_CHECK(cudaMemcpy(ddensity, density, sizeof density, cudaMemcpyHostToDevice));
    //: the read side alone, slot `level`: its blocks from the workspace's start, the grid its own (the write side holds no CTA)
    pass::Args args{};
    for (int l = 0; l < kLevels; ++l) {
      args.mask[l] = row.masked[l] ? dmask[l] : nullptr;
      args.masked[0] |= row.masked[l] ? 1 << l : 0;
      args.width_bits[l] = kWidthBits;
    }
    args.heads[0] = args.heads[1] = 1;
    args.levels = kLevels;
    args.first_cta[1] = args.first_cta[2] = INT32_MAX;

    pass::Args ragged = args;
    ragged.tokens = checked_tokens;
    ragged.words = static_cast<int>((checked_tokens + 31) / 32);
    ragged.windows = ctas;
    const int widest[kLevels] = {256, 256}, read[1][plan::kCarveMaxLevels] = {{row.byte[0] + 1, row.byte[1] + 1}};
    const plan::LeafGroups g = plan::leaf_groups(plan::prefill_carve<plan::PrefillPolicy>(kLevels, widths, widest, read, 1), true, 0);
    const size_t held = static_cast<size_t>(ctas) * static_cast<size_t>(plan::leaf_block_bytes(g.bits, kWords));
    LAB_CHECK(cudaMemset(dwork, 0x5a, held));
    timed<<<ctas, pass::kThreads>>>(b.times, ragged, ddensity, dwork, 1, 0);
    LAB_CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> got(held);
    LAB_CHECK(cudaMemcpy(got.data(), dwork, held, cudaMemcpyDeviceToHost));
    long long wrong[2];
    compare(row, mask, checked_tokens, got, wrong);
    b.check(name, wrong[0] == 0 && wrong[1] == 0,
            std::to_string(wrong[0]) + " words and " + std::to_string(wrong[1]) + " counts differ over " + std::to_string(ctas) + " windows of "
                + std::to_string(1 << g.bits) + " leaf groups, the call ragged");

    args.tokens = static_cast<long long>(ctas) * row.rounds * kWindow;
    args.words = ctas * row.rounds * kWords;
    args.windows = ctas * row.rounds;
    const int rounds = row.rounds;
    const lab::Samples cycles = b.variant(
        [&] { timed<<<ctas, pass::kThreads>>>(b.times, args, ddensity, dwork, rounds, ctas); },
        [&] { return b.span(pass::kThreads >> 5) / rounds / kWindow; });
    const std::string timed_row = name + ", the masks once-read";
    b.unchecked(timed_row, "the row above checks this kernel on a ragged call; this one is its time on whole windows");
    b.record(timed_row, "SM-cycles a token", cycles);
  }
  return 0;
}
