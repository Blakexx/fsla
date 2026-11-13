// measure/lab/experiments/facts_lists.cu -- QUESTION: what does the facts pass cost a token on hopper when a sparse
// level's liveness is a MASK it is handed -- a bit a digit a token -- and not a test of the rows' values, and what
// does the value test it replaces cost beside it? METHOD: a CTA a window of 1,024 tokens, a thread a token, for
// one side's sparse level of 256 digits. The pass is a counting sort of the window's live (token, digit) pairs by
// digit, in token order: a thread ORs its lane into a word a (warp, digit) for each of its live digits; a thread
// a digit then sums the 32 warps' counts before each warp, and the digits' offsets follow; a thread then places
// each of its live digits -- its rank the offsets' and the lanes before it in its warp's word -- as an ENTRY, the
// token and its byte on the digit, the byte read from the level's tile-major row. Two forms find a token's live
// digits: the MASK's 32 bytes as handed, or a VALUE TEST of the token's 256-byte row. A call is the window's
// counts alone (what sizes its lists) or the lists too. Every count and entry is checked against the host's.
// TIMELINE: no MMA and no staged operand. A thread's trip is its window once: its mask's two loads (or its row's
// 16 and their tests); a shared-memory atomic a live digit; the barrier; a digit's 32 words summed by its thread
// and the 256 offsets by one warp's; the barrier; a byte's load and an entry's store a live digit. Live: a thread's
// 8 mask words.
// see docs/internals/tools/lab.md#facts-lists
#include "lab.cuh"

namespace facts_lists {

constexpr int kWindow = 1024, kThreads = kWindow, kWarps = kThreads / 32, kDigits = 256, kTile = 64;
//: shared memory: a word a (warp, digit) -- its bit `l` is lane l's token live on the digit; the count before
//: each warp a digit; the digits' offsets
constexpr int kWordsAt = 0, kBeforeAt = kWordsAt + kWarps * kDigits * 4, kOffsetsAt = kBeforeAt + kWarps * kDigits * 2,
              kSmemBytes = kOffsetsAt + (kDigits + 1) * 4;
enum Form : int { kMask = 0, kValues = 1 };
enum Call : int { kCounts = 0, kLists = 1 };

//: THE CELL: token `x` has 4 live digits, and a fifth where `x % 3 == 0`, each a byte from 1 to 255
constexpr int live_of(uint32_t x) { return 4 + (x % 3 == 0); }
constexpr uint32_t digit_of(uint32_t x, int i) { return (x * 5u + (x >> 3) + static_cast<uint32_t>(i) * 37u) & 255u; }
constexpr uint8_t byte_of(uint32_t x, int i) { return static_cast<uint8_t>(1u + (x * 7u + static_cast<uint32_t>(i) * 13u) % 255u); }

//: a word's bytes that are not zero, a bit each in its low 4
__device__ __forceinline__ uint32_t nonzero4(uint32_t w) {
  const uint32_t high = (w | ((w & 0x7f7f7f7fu) + 0x7f7f7f7fu)) & 0x80808080u;
  return ((high >> 7) * 0x00204081u >> 21) & 0xfu;
}

//: A WINDOW: its sparse level's lists by digit, or their counts
template <int FORM, int CALL>
__global__ void __launch_bounds__(kThreads, 1)
    window(long long* times, const uint32_t* masks, const uint4* token_rows, const uint8_t* tiled_rows,
           const uint32_t* bases, uint16_t* counts, uint32_t* entries, int rounds) {
  extern __shared__ __align__(16) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const long long token = static_cast<long long>(blockIdx.x) * kWindow + tid;
  uint32_t* const words = reinterpret_cast<uint32_t*>(smem + kWordsAt);
  uint16_t* const before = reinterpret_cast<uint16_t*>(smem + kBeforeAt);
  uint32_t* const offsets = reinterpret_cast<uint32_t*>(smem + kOffsetsAt);
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int round = 0; round < rounds; ++round) {
#pragma unroll 1
    for (int i = tid; i < kWarps * kDigits; i += kThreads) words[i] = 0u;
    __syncthreads();
    //: THE TOKEN'S LIVE DIGITS, 8 words of 32: the mask as handed, or a test of its row's 256 bytes
    uint32_t live[8];
    if constexpr (FORM == kMask) {
      const uint4 lo = reinterpret_cast<const uint4*>(masks)[2 * token], hi = reinterpret_cast<const uint4*>(masks)[2 * token + 1];
      live[0] = lo.x, live[1] = lo.y, live[2] = lo.z, live[3] = lo.w, live[4] = hi.x, live[5] = hi.y, live[6] = hi.z, live[7] = hi.w;
    } else {
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        const uint4 a = token_rows[16 * token + 2 * w], b = token_rows[16 * token + 2 * w + 1];
        live[w] = nonzero4(a.x) | nonzero4(a.y) << 4 | nonzero4(a.z) << 8 | nonzero4(a.w) << 12 | nonzero4(b.x) << 16
                  | nonzero4(b.y) << 20 | nonzero4(b.z) << 24 | nonzero4(b.w) << 28;
      }
    }
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
    //: A DIGIT'S COUNTS BEFORE EACH WARP, by the digit's thread; its total left in the offsets' place
    if (tid < kDigits) {
      uint32_t sum = 0;
#pragma unroll 1
      for (int w = 0; w < kWarps; ++w) {
        before[w * kDigits + tid] = static_cast<uint16_t>(sum);
        sum += static_cast<uint32_t>(__popc(words[w * kDigits + tid]));
      }
      offsets[tid + 1] = sum;
      if constexpr (CALL == kCounts) counts[static_cast<long long>(blockIdx.x) * kDigits + tid] = static_cast<uint16_t>(sum);
    }
    __syncthreads();
    if constexpr (CALL == kLists) {
      //: THE DIGITS' OFFSETS, one thread's 256 sums; then an ENTRY a live digit, at its digit's offset, the count
      //: before its warp and the lanes before it: the token and its byte, read from the tile-major row
      if (tid == 0) {
        uint32_t sum = 0;
        offsets[0] = 0;
#pragma unroll 1
        for (int d = 0; d < kDigits; ++d) {
          const uint32_t n = offsets[d + 1];
          offsets[d] = sum;
          sum += n;
        }
        offsets[kDigits] = sum;
      }
      __syncthreads();
      const uint32_t base = bases[blockIdx.x];
      const uint8_t* const row = tiled_rows + (token / kTile) * (kDigits * kTile) + (token % kTile) * 16;
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        uint32_t m = live[w];
#pragma unroll 1
        while (m) {
          const int d = 32 * w + __ffs(static_cast<int>(m)) - 1;
          m &= m - 1;
          const uint32_t at = offsets[d] + before[warp * kDigits + d]
                              + static_cast<uint32_t>(__popc(words[warp * kDigits + d] & ((1u << lane) - 1u)));
          entries[base + at] = static_cast<uint32_t>(tid) | static_cast<uint32_t>(row[(d >> 4) * (kTile * 16) + (d & 15)]) << 16
                               | static_cast<uint32_t>(d) << 24;
        }
      }
      if (tid < kDigits) counts[static_cast<long long>(blockIdx.x) * kDigits + tid] =
          static_cast<uint16_t>((tid + 1 < kDigits ? offsets[tid + 1] : offsets[kDigits]) - offsets[tid]);
    }
    __syncthreads();
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
      //: an entry's byte is loaded where its entry is stored, a live digit a trip: the pass's first form, unstaged
      if (call == kLists) b.expect(k, "LOAD_IN_TRIP", "<=", 8);
    }
  if (b.expect_only) return 0;
  const int windows = b.sms;
  const size_t tokens = static_cast<size_t>(windows) * kWindow;
  //: the mask, 8 words a token; the level's rows token-major (the value test's) and tile-major (the bytes')
  std::vector<uint32_t> mask(tokens * 8, 0);
  std::vector<uint8_t> token_major(tokens * kDigits, 0), tiled(tokens * kDigits, 0);
  //: the host's lists: a window's entries by digit, a digit's by token
  std::vector<std::vector<uint32_t>> want(static_cast<size_t>(windows) * kDigits);
  std::vector<uint32_t> bases(windows + 1, 0);
  for (size_t x = 0; x < tokens; ++x) {
    for (int i = 0; i < live_of(static_cast<uint32_t>(x)); ++i) {
      const uint32_t d = digit_of(static_cast<uint32_t>(x), i);
      const uint8_t v = byte_of(static_cast<uint32_t>(x), i);
      mask[x * 8 + d / 32] |= 1u << (d % 32);
      token_major[x * kDigits + d] = v;
      tiled[(x / kTile) * (kDigits * kTile) + (d / 16) * (kTile * 16) + (x % kTile) * 16 + d % 16] = v;
    }
  }
  for (size_t x = 0; x < tokens; ++x)
    for (uint32_t d = 0; d < kDigits; ++d)
      if (token_major[x * kDigits + d])
        want[(x / kWindow) * kDigits + d].push_back(static_cast<uint32_t>(x % kWindow)
                                                    | static_cast<uint32_t>(token_major[x * kDigits + d]) << 16 | d << 24);
  for (int w = 0; w < windows; ++w) {
    uint32_t n = 0;
    for (int d = 0; d < kDigits; ++d) n += static_cast<uint32_t>(want[static_cast<size_t>(w) * kDigits + d].size());
    bases[w + 1] = bases[w] + n;
  }
  const size_t total = bases[windows];
  uint32_t *dmask = nullptr, *dbases = nullptr, *dentries = nullptr;
  uint8_t *dtoken = nullptr, *dtiled = nullptr;
  uint16_t* dcounts = nullptr;
  LAB_CHECK(cudaMalloc(&dmask, mask.size() * 4));
  LAB_CHECK(cudaMemcpy(dmask, mask.data(), mask.size() * 4, cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMalloc(&dtoken, token_major.size()));
  LAB_CHECK(cudaMemcpy(dtoken, token_major.data(), token_major.size(), cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMalloc(&dtiled, tiled.size()));
  LAB_CHECK(cudaMemcpy(dtiled, tiled.data(), tiled.size(), cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMalloc(&dbases, bases.size() * 4));
  LAB_CHECK(cudaMemcpy(dbases, bases.data(), bases.size() * 4, cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMalloc(&dentries, total * 4));
  LAB_CHECK(cudaMalloc(&dcounts, static_cast<size_t>(windows) * kDigits * 2));
  const auto run = [&](auto kernel, int form, int call) {
    const std::string name = std::string(form == kMask ? "the mask as handed" : "a value test of the rows") + ", "
                             + (call == kCounts ? "a window's counts" : "a window's lists");
    LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    LAB_CHECK(cudaMemset(dentries, 0xff, total * 4));
    LAB_CHECK(cudaMemset(dcounts, 0xff, static_cast<size_t>(windows) * kDigits * 2));
    kernel<<<windows, kThreads, kSmemBytes>>>(b.times, dmask, reinterpret_cast<const uint4*>(dtoken), dtiled, dbases,
                                               dcounts, dentries, 1);
    LAB_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> got_counts(static_cast<size_t>(windows) * kDigits);
    std::vector<uint32_t> got(total);
    LAB_CHECK(cudaMemcpy(got_counts.data(), dcounts, got_counts.size() * 2, cudaMemcpyDeviceToHost));
    LAB_CHECK(cudaMemcpy(got.data(), dentries, total * 4, cudaMemcpyDeviceToHost));
    long long wrong_counts = 0, wrong = 0;
    for (int w = 0; w < windows; ++w) {
      size_t at = bases[w];
      for (int d = 0; d < kDigits; ++d) {
        const auto& list = want[static_cast<size_t>(w) * kDigits + d];
        wrong_counts += got_counts[static_cast<size_t>(w) * kDigits + d] != list.size();
        if (call == kLists)
          for (uint32_t e : list) wrong += got[at++] != e;
      }
    }
    b.check(name, wrong_counts == 0 && wrong == 0,
            std::to_string(wrong_counts) + " of " + std::to_string(got_counts.size()) + " counts"
                + (call == kLists ? " and " + std::to_string(wrong) + " of " + std::to_string(total) + " entries" : "")
                + " differ");
    constexpr int kRounds = 64;
    const lab::Samples cycles = b.variant(
        [&] {
          kernel<<<windows, kThreads, kSmemBytes>>>(b.times, dmask, reinterpret_cast<const uint4*>(dtoken), dtiled,
                                                     dbases, dcounts, dentries, kRounds);
        },
        [&] { return b.span(kThreads >> 5) / kRounds / kWindow; });
    b.record(name, "SM-cycles a token", cycles);
  };
  run(window<kMask, kCounts>, kMask, kCounts);
  run(window<kMask, kLists>, kMask, kLists);
  run(window<kValues, kCounts>, kValues, kCounts);
  run(window<kValues, kLists>, kValues, kLists);
  return 0;
}
