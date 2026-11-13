// measure/lab/experiments/facts_lists.cu -- QUESTION: what does the facts pass cost a token on hopper when a sparse
// level's liveness is a MASK it is handed -- a bit a digit a token -- and the pass emits that mask TURNED AROUND, a
// window's tokens by digit, and nothing else; and what does a value test of the rows cost beside it? METHOD: a CTA
// a window of 1,024 tokens, a thread a token, one side's sparse level of 256 digits. The pass is a counting sort
// of the window's live (token, digit) pairs by digit, in token order: a thread ORs its lane into a word a (warp,
// digit) for each of its live digits; a digit's thread sums the 32 warps' counts before each warp; the digits'
// offsets are a scan by the 8 warps of digits and their totals; a thread then stores each of its live digits'
// ENTRY -- the token, no byte -- at its digit's offset, the count before its warp and the lanes before it in its
// warp's word. Two forms find a token's live digits: the MASK's 32 bytes as handed, or a VALUE TEST of the token's
// 256 bytes, read from the tile-major rows the bundle holds. A call is the window's counts alone (what sizes its
// lists) or the lists too. Each is timed with every table in cache -- a call's rounds over the same windows -- and
// ONCE-READ, a round a fresh window; at a cell of 4 to 5 live digits a token and at one of 32. Every count and
// entry of the first round is checked against the host's.
// TIMELINE: no MMA and no staged operand: the mask's two loads (or the rows' 16) are a thread's first act of a
// round, used in it. A thread's round: those loads; a shared-memory atomic a live digit; the barrier; a digit's 32
// words summed by its thread, the offsets' scan (5 shuffles a warp, 8 totals); the barrier; a store a live digit.
// Live: a thread's 8 mask words.
// see docs/internals/tools/lab.md#facts-lists
#include "lab.cuh"

namespace facts_lists {

constexpr int kWindow = 1024, kThreads = kWindow, kWarps = kThreads / 32, kDigits = 256, kTile = 64;
//: shared memory: a word a (warp, digit) -- its bit `l` is lane l's token live on the digit; the count before
//: each warp a digit; the digits' offsets inside their warp of digits; the totals of the 8 warps that scan them
constexpr int kWordsAt = 0, kBeforeAt = kWordsAt + kWarps * kDigits * 4, kOffsetsAt = kBeforeAt + kWarps * kDigits * 2,
              kTotalsAt = kOffsetsAt + kDigits * 4, kSmemBytes = kTotalsAt + 8 * 4;
enum Form : int { kMask = 0, kValues = 1 };
enum Call : int { kCounts = 0, kLists = 1 };

//: A CELL: token `x` has `live_of` live digits, a stride apart from a start its own
constexpr int live_of(int cell, uint32_t x) { return cell == 0 ? 4 + (x % 3 == 0) : 32; }
constexpr uint32_t digit_of(int cell, uint32_t x, int i) {
  return (x * 5u + (x >> 3) + static_cast<uint32_t>(i) * (cell == 0 ? 37u : 7u)) & 255u;
}

//: a word's bytes that are not zero, a bit each in its low 4
__device__ __forceinline__ uint32_t nonzero4(uint32_t w) {
  const uint32_t high = (w | ((w & 0x7f7f7f7fu) + 0x7f7f7f7fu)) & 0x80808080u;
  return ((high >> 7) * 0x00204081u >> 21) & 0xfu;
}

//: A WINDOW a round: its sparse level's lists by digit, or their counts; round `r` is window `block + r stride`
template <int FORM, int CALL>
__global__ void __launch_bounds__(kThreads, 1)
    window(long long* times, const uint4* masks, const uint4* rows, const uint32_t* bases, uint16_t* counts,
           uint16_t* entries, int rounds, int stride) {
  extern __shared__ __align__(16) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  uint32_t* const words = reinterpret_cast<uint32_t*>(smem + kWordsAt);
  uint16_t* const before = reinterpret_cast<uint16_t*>(smem + kBeforeAt);
  uint32_t* const offsets = reinterpret_cast<uint32_t*>(smem + kOffsetsAt);
  uint32_t* const totals = reinterpret_cast<uint32_t*>(smem + kTotalsAt);
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int round = 0; round < rounds; ++round) {
    const long long at = static_cast<long long>(blockIdx.x) + static_cast<long long>(round) * stride;
    const long long token = at * kWindow + tid;
    //: THE TOKEN'S LIVE DIGITS, 8 words of 32: the mask as handed, or a test of its 256 bytes -- a block of the
    //: tile-major rows a load
    uint32_t live[8];
    if constexpr (FORM == kMask) {
      const uint4 lo = masks[2 * token], hi = masks[2 * token + 1];
      live[0] = lo.x, live[1] = lo.y, live[2] = lo.z, live[3] = lo.w, live[4] = hi.x, live[5] = hi.y, live[6] = hi.z, live[7] = hi.w;
    } else {
      const uint4* const mine = rows + (token / kTile) * (kDigits * kTile / 16) + token % kTile;
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        const uint4 a = mine[(2 * w) * kTile], b = mine[(2 * w + 1) * kTile];
        live[w] = nonzero4(a.x) | nonzero4(a.y) << 4 | nonzero4(a.z) << 8 | nonzero4(a.w) << 12 | nonzero4(b.x) << 16
                  | nonzero4(b.y) << 20 | nonzero4(b.z) << 24 | nonzero4(b.w) << 28;
      }
    }
#pragma unroll 1
    for (int i = tid; i < kWarps * kDigits; i += kThreads) words[i] = 0u;
    __syncthreads();
    //: THE LANES' WORDS: a live digit's bit of the token's lane in its warp's word
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      uint32_t m = live[w];
#pragma unroll 1
      while (m) {
        const int d = 32 * w + __ffs(static_cast<int>(m)) - 1;
        m &= m - 1;
        atomicOr(words + warp * kDigits + d, 1u << lane);
      }
    }
    __syncthreads();
    //: A DIGIT'S COUNTS BEFORE EACH WARP, by the digit's thread, and THE DIGITS' OFFSETS: the 8 warps of digits
    //: each scan their 32 totals
    if (tid < kDigits) {
      uint32_t sum = 0;
#pragma unroll 1
      for (int w = 0; w < kWarps; ++w) {
        before[w * kDigits + tid] = static_cast<uint16_t>(sum);
        sum += static_cast<uint32_t>(__popc(words[w * kDigits + tid]));
      }
      counts[at * kDigits + tid] = static_cast<uint16_t>(sum);
      if constexpr (CALL == kLists) {
        uint32_t scan = sum;
#pragma unroll
        for (int step = 1; step < 32; step <<= 1) {
          const uint32_t up = __shfl_up_sync(0xffffffffu, scan, step);
          scan += lane >= step ? up : 0u;
        }
        offsets[tid] = scan - sum;
        if (lane == 31) totals[warp] = scan;
      }
    }
    __syncthreads();
    if constexpr (CALL == kLists) {
      //: AN ENTRY a live digit, the token: at its digit's offset -- its warp of digits' start and its place in it
      //: -- the count before its warp and the lanes before it
      uint32_t first[8];
      uint32_t sum = 0;
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        first[w] = sum;
        sum += totals[w];
      }
      const uint32_t base = bases[at];
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        uint32_t m = live[w];
#pragma unroll 1
        while (m) {
          const int d = 32 * w + __ffs(static_cast<int>(m)) - 1;
          m &= m - 1;
          const uint32_t place = first[w] + offsets[d] + before[warp * kDigits + d]
                                 + static_cast<uint32_t>(__popc(words[warp * kDigits + d] & ((1u << lane) - 1u)));
          entries[base + place] = static_cast<uint16_t>(tid);
        }
      }
      __syncthreads();
    }
  }
  clock.stop(times);
}

}  // namespace facts_lists

int main(int argc, char** argv) {
  using namespace facts_lists;
  lab::Bench b(argc, argv);
  for (int form : {static_cast<int>(kMask), static_cast<int>(kValues)})
    for (int call : {static_cast<int>(kCounts), static_cast<int>(kLists)}) {
      const std::string k = lab::kernel("facts_lists::window", {form, call});
      //: a token's live digits are a loop a lane its own count of trips
      b.expect(k, "LOOP_CONVERGENCE", "<=", 64);
    }
  if (b.expect_only) return 0;
  constexpr int kRounds = 16;
  const int windows = b.sms;
  const size_t all_windows = static_cast<size_t>(windows) * kRounds, tokens = all_windows * kWindow;
  uint4 *dmask = nullptr, *drows = nullptr;
  uint32_t* dbases = nullptr;
  uint16_t *dentries = nullptr, *dcounts = nullptr;
  LAB_CHECK(cudaMalloc(&dmask, tokens * 32));
  LAB_CHECK(cudaMalloc(&drows, tokens * kDigits));
  LAB_CHECK(cudaMalloc(&dbases, (all_windows + 1) * 4));
  LAB_CHECK(cudaMalloc(&dcounts, all_windows * kDigits * 2));
  LAB_CHECK(cudaMalloc(&dentries, tokens * 32 * 2));
  for (int cell = 0; cell < 2; ++cell) {
    //: the mask, 8 words a token, and the level's rows tile-major; the host's lists of the first round's windows
    std::vector<uint32_t> mask(tokens * 8, 0);
    std::vector<uint8_t> tiled(tokens * kDigits, 0);
    std::vector<uint32_t> bases(all_windows + 1, 0);
    for (size_t x = 0; x < tokens; ++x) {
      const int n = live_of(cell, static_cast<uint32_t>(x));
      for (int i = 0; i < n; ++i) {
        const uint32_t d = digit_of(cell, static_cast<uint32_t>(x), i);
        mask[x * 8 + d / 32] |= 1u << (d % 32);
        tiled[(x / kTile) * (kDigits * kTile) + (d / 16) * (kTile * 16) + (x % kTile) * 16 + d % 16] =
            static_cast<uint8_t>(1u + (x * 7u + static_cast<uint32_t>(i) * 13u) % 255u);
      }
      bases[x / kWindow + 1] += static_cast<uint32_t>(n);
    }
    for (size_t w = 0; w < all_windows; ++w) bases[w + 1] += bases[w];
    std::vector<std::vector<uint16_t>> want(static_cast<size_t>(windows) * kDigits);
    for (size_t x = 0; x < static_cast<size_t>(windows) * kWindow; ++x)
      for (uint32_t d = 0; d < kDigits; ++d)
        if (mask[x * 8 + d / 32] >> (d % 32) & 1u) want[(x / kWindow) * kDigits + d].push_back(static_cast<uint16_t>(x % kWindow));
    LAB_CHECK(cudaMemcpy(dmask, mask.data(), tokens * 32, cudaMemcpyHostToDevice));
    LAB_CHECK(cudaMemcpy(drows, tiled.data(), tokens * kDigits, cudaMemcpyHostToDevice));
    LAB_CHECK(cudaMemcpy(dbases, bases.data(), (all_windows + 1) * 4, cudaMemcpyHostToDevice));
    const size_t checked = bases[windows];
    const auto run = [&](auto kernel, int form, int call) {
      const std::string name = std::string(cell == 0 ? "4 to 5 live digits a token" : "32 live digits a token") + ", "
                               + (form == kMask ? "the mask as handed" : "a value test of the rows") + ", "
                               + (call == kCounts ? "a window's counts" : "a window's lists");
      LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
      LAB_CHECK(cudaMemset(dentries, 0xff, checked * 2));
      LAB_CHECK(cudaMemset(dcounts, 0xff, static_cast<size_t>(windows) * kDigits * 2));
      kernel<<<windows, kThreads, kSmemBytes>>>(b.times, dmask, drows, dbases, dcounts, dentries, 1, 0);
      LAB_CHECK(cudaDeviceSynchronize());
      std::vector<uint16_t> got_counts(static_cast<size_t>(windows) * kDigits), got(checked);
      LAB_CHECK(cudaMemcpy(got_counts.data(), dcounts, got_counts.size() * 2, cudaMemcpyDeviceToHost));
      LAB_CHECK(cudaMemcpy(got.data(), dentries, checked * 2, cudaMemcpyDeviceToHost));
      long long wrong_counts = 0, wrong = 0;
      size_t at = 0;
      for (size_t l = 0; l < want.size(); ++l) {
        wrong_counts += got_counts[l] != want[l].size();
        if (call == kLists)
          for (uint16_t e : want[l]) wrong += got[at++] != e;
      }
      b.check(name, wrong_counts == 0 && wrong == 0,
              std::to_string(wrong_counts) + " of " + std::to_string(got_counts.size()) + " counts"
                  + (call == kLists ? " and " + std::to_string(wrong) + " of " + std::to_string(checked) + " entries" : "")
                  + " differ");
      for (int fresh = 0; fresh < 2; ++fresh) {
        //: the tables in cache: a call's rounds are the same windows; ONCE-READ: a round a fresh window
        const lab::Samples cycles = b.variant(
            [&] {
              kernel<<<windows, kThreads, kSmemBytes>>>(b.times, dmask, drows, dbases, dcounts, dentries, kRounds,
                                                         fresh ? windows : 0);
            },
            [&] { return b.span(kThreads >> 5) / kRounds / kWindow; });
        const std::string timed = name + (fresh ? ", the tables once-read" : ", the tables in cache");
        b.unchecked(timed, "the row above checks this kernel's first round; this one is its time");
        b.record(timed, "SM-cycles a token", cycles);
      }
    };
    run(window<kMask, kCounts>, kMask, kCounts);
    run(window<kMask, kLists>, kMask, kLists);
    run(window<kValues, kCounts>, kValues, kCounts);
    run(window<kValues, kLists>, kValues, kLists);
  }
  return 0;
}
