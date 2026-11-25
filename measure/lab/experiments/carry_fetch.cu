// measure/lab/experiments/carry_fetch.cu -- QUESTION: what does a carry warp's own supply of a group's value lines
// cost -- the LIBRARY's listing of a window's live tokens, and the copies of their 128-byte value lines by token index
// straight into the steps' tiles in shared memory -- by how many tokens are live?
// METHOD: this file compiles the library's device code (`carry/carry_list.cuh`, `carry/carry_values.cuh`: the listing and the
// rows; the select and the copies it feeds are this file's own, the form its record measured). A CTA an SM
// of 16 warps, each warp a group's windows: a lane loads its mask word a window ahead; the warp lists the window a
// round of 32 places at a time and issues the round's copies, 8 instructions of 4 tokens, into the round's 32 rows --
// each a token's 128 bytes whole, its pieces in the order the step's swizzle reads -- two rounds in flight; a warp's
// tiles start on a 1,024-byte boundary. The values are one table of 256 windows every CTA reads (a unit's CTAs ask
// for the same lines). Rows: 64 live tokens a window; every token live. A second kernel lands each of the first
// window's first four rounds of every group and copies its tiles out: checked byte for byte against the host's, a
// dead place's lines zero.
// TIMELINE: no MMA. A window: the listing's running sums (5 shuffles), then a round -- the select (7 shuffles), and 8
// asynchronous copies a lane, each behind a shuffle of the round's tokens; a commit; a wait for the round before.
// One global load a lane a window, a window ahead. Live: the next window's word, the listing's four, the round's token.
// see docs/internals/tools/lab.md#carry-fetch
#include "lab.cuh"
#include "carry/carry_list.cuh"
#include "carry/carry_values.cuh"

namespace carry_fetch {

namespace carry = fsla::carry::detail;
namespace ops = ::denseref::ops;

//: THE TOKEN AT A PLACE of the listing: lane `p` of round `r` takes place `WORDS r + p`; -1 past the total -- this experiment's own: the library's supply lays its tokens (`carry_supply`)
template <int WORDS>
__device__ __forceinline__ int listed_token(const carry::Listing& s, int round) {
  const int place = WORDS * round + static_cast<int>(threadIdx.x & (WORDS - 1));
  //: the lane whose tokens hold the place: the count of lanes that end at or before it
  int lane = 0;
#pragma unroll
  for (int step = WORDS / 2; step >= 1; step >>= 1) {
    const int ends = ops::lane_broadcast(s.through, lane + step - 1);
    lane += ends <= place ? step : 0;
  }
  const auto word = static_cast<std::uint32_t>(ops::lane_broadcast(static_cast<int>(s.word), lane));
  const int before = ops::lane_broadcast(s.before, lane);
  const int token = WORDS * lane + ops::nth_set_bit(word, place - before);
  return place < s.total ? token : -1;
}


//: THE COPIES OF A ROUND'S VALUE LINES into its rows at `tiles`, one committed group; a token's line is `stride` bytes on from the token before's -- this experiment's own, fed by the select
template <int CHANNELS>
__device__ __forceinline__ void fetch_round(ops::SmemAddr tiles, const std::uint8_t* values,
                                            std::int64_t stride, int token) {
  constexpr int kLanes = ops::kBallotLanes, kPieces = CHANNELS / carry::kPieceChannels;
  constexpr int kTokens = kLanes / kPieces;
  static_assert(carry::kRowBytes<CHANNELS> == 128, "a row is the 128 bytes the step's swizzle turns");
  const int lane = static_cast<int>(threadIdx.x) & (kLanes - 1);
  const int piece = lane % kPieces, which = lane / kPieces;
#pragma unroll
  for (int i = 0; i < kPieces; ++i) {
    const int place = kTokens * i + which;
    const int from = ops::lane_broadcast(token, place);
    const ops::SmemAddr row = tiles + static_cast<ops::SmemAddr>(place * carry::kRowBytes<CHANNELS>);
    const std::uint8_t* const at = values + static_cast<std::int64_t>(from) * stride;
    const ops::SmemAddr to = ops::step_row_piece(row, static_cast<std::uint32_t>(piece));
    ops::stage_tile_if(to, at + piece * carry::kPieceBytes, from >= 0);
  }
  ops::stage_commit();
}

constexpr int kWords = 32, kWarps = 16, kWindow = kWords * 32, kChannels = 64, kLine = 2 * kChannels;
constexpr int kRound = carry::kRoundBytes<kChannels>, kValueWindows = 256;
//: the rounds of a group's first window the check keeps
constexpr int kKept = 4;
//: a step's rows start on a 1,024-byte boundary: the bytes from `a` to the first one
constexpr uint32_t kBoundary = 1024;
__device__ __forceinline__ uint32_t to_boundary(uint32_t a) { return (kBoundary - (a & (kBoundary - 1))) & (kBoundary - 1); }

//: THE TIMED KERNEL: a group's windows, listed and fetched
__global__ void __launch_bounds__(32 * kWarps, 1)
    fetched(long long* times, const uint32_t* words, const uint8_t* values, int windows) {
  __shared__ alignas(128) uint8_t tiles[kWarps * 2 * kRound + kBoundary];
  const int warp = lab::warp_id(), lane = lab::lane_id();
  const ops::SmemAddr base = ops::smem_addr(tiles);
  const ops::SmemAddr mine = base + to_boundary(base) + static_cast<uint32_t>(warp * 2 * kRound);
  const size_t group = (static_cast<size_t>(blockIdx.x) * kWarps + warp) * static_cast<size_t>(windows);
  uint32_t next = words[group * kWords + lane];
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int w = 0; w < windows; ++w) {
    const carry::Listing s = carry::listing_of<kWords>(next);
    const size_t ahead = (group + static_cast<size_t>(w + 1 < windows ? w + 1 : w)) * kWords + lane;
    next = words[ahead];
    const uint8_t* const lines = values + static_cast<size_t>(w % kValueWindows) * kWindow * kLine;
    //: a window's rounds: at least one, and on while any lane votes a place under the total; two rounds in flight
    int round = 0;
#pragma unroll 1
    do {
      const int token = listed_token<kWords>(s, round);
      fetch_round<kChannels>(mine + static_cast<uint32_t>((round & 1) * kRound), lines, kLine, token);
      ops::stage_wait<1>();
      ++round;
    } while (ops::lane_ballot(kWords * round < s.total) != 0u);
  }
  ops::stage_wait<0>();
  clock.stop(times);
}

//: THE CHECKED KERNEL: a group's first window's first `kKept` rounds, each landed and its two tiles copied out -- a
//: lane its 4 bytes of each 128
__global__ void __launch_bounds__(32 * kWarps, 1)
    checked(const uint32_t* words, const uint8_t* values, uint8_t* out, int* off_at) {
  __shared__ alignas(128) uint8_t tiles[kWarps * kRound + kBoundary];
  const int warp = lab::warp_id(), lane = lab::lane_id();
  const size_t group = static_cast<size_t>(blockIdx.x) * kWarps + warp;
  const uint8_t* const from = tiles + to_boundary(ops::smem_addr(tiles)) + warp * kRound;
  //: whether the warp's tile is off the boundary a step's rows start on, for the host
  if (lane == 0) off_at[group] = static_cast<int>((ops::smem_addr(from) & (kBoundary - 1)) != 0u);
  const carry::Listing s = carry::listing_of<kWords>(words[group * kWords + lane]);
#pragma unroll 1
  for (int round = 0; round < kKept; ++round) {
    const int token = listed_token<kWords>(s, round);
    fetch_round<kChannels>(ops::smem_addr(from), values, kLine, token);
    ops::stage_wait<0>();
    uint8_t* const to = out + (group * kKept + round) * kRound;
#pragma unroll 1
    for (int k = 0; k < kRound / 128; ++k) {
      const int i = 128 * k + 4 * lane;
      *reinterpret_cast<uint32_t*>(to + i) = *reinterpret_cast<const uint32_t*>(from + i);
    }
    __syncwarp();
  }
}

inline bool live_at(uint64_t x, uint32_t one_in) {
  uint64_t h = (x + 17) * 0x9e3779b97f4a7c15ull;
  h ^= h >> 29;
  h *= 0xbf58476d1ce4e5b9ull;
  h ^= h >> 32;
  return h % one_in == 0;
}

inline uint8_t value_byte(size_t i) { return static_cast<uint8_t>((i * 2654435761ull >> 13) | 1u); }

}  // namespace carry_fetch

void run(lab::Bench& b, const uint8_t* dvalues, const std::vector<uint8_t>& values, const std::string& name,
         uint32_t one_in) {
  using namespace carry_fetch;
  constexpr int kTimed = 256;
  const size_t groups = static_cast<size_t>(b.sms) * kWarps, count = groups * kTimed * kWords;
  std::vector<uint32_t> words(count);
  long long live = 0;
  for (size_t i = 0; i < count; ++i) {
    uint32_t a = 0;
    for (int bit = 0; bit < 32; ++bit) a |= static_cast<uint32_t>(live_at(i * 32 + bit, one_in)) << bit;
    words[i] = a;
    live += __builtin_popcount(a);
  }
  //: the check's call is one window a group: its words are each group's first window's
  std::vector<uint32_t> first(groups * kWords);
  for (size_t g = 0; g < groups; ++g)
    for (int i = 0; i < kWords; ++i) first[g * kWords + i] = words[g * kTimed * kWords + i];
  uint32_t *dwords = nullptr, *dfirst = nullptr;
  uint8_t* dout = nullptr;
  int* doff_at = nullptr;
  const size_t out_bytes = groups * kKept * kRound;
  LAB_CHECK(cudaMalloc(&dwords, count * 4));
  LAB_CHECK(cudaMalloc(&dfirst, first.size() * 4));
  LAB_CHECK(cudaMalloc(&dout, out_bytes));
  LAB_CHECK(cudaMalloc(&doff_at, groups * sizeof(int)));
  LAB_CHECK(cudaMemcpy(dwords, words.data(), count * 4, cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMemcpy(dfirst, first.data(), first.size() * 4, cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMemset(dout, 0xff, out_bytes));
  checked<<<b.sms, 32 * kWarps>>>(dfirst, dvalues, dout, doff_at);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<uint8_t> got(out_bytes);
  LAB_CHECK(cudaMemcpy(got.data(), dout, out_bytes, cudaMemcpyDeviceToHost));
  std::vector<int> off_at(groups);
  LAB_CHECK(cudaMemcpy(off_at.data(), doff_at, groups * sizeof(int), cudaMemcpyDeviceToHost));
  long long wrong = 0, checked = 0;
  for (size_t g = 0; g < groups; ++g) {
    std::vector<int> list;
    for (int token = 0; token < kWindow; ++token)
      if ((first[g * kWords + token / 32] >> (token % 32) & 1u) != 0) list.push_back(token);
    for (int round = 0; round < kKept; ++round)
      for (int place = 0; place < kWords; ++place) {
        const int at = kWords * round + place;
        const bool here = at < static_cast<int>(list.size());
        for (int piece = 0; piece < kChannels / carry::kPieceChannels; ++piece) {
          //: the place's row, and the piece where the step's swizzle turns it by the row's three bits
          const int turned = piece ^ (place & 7);
          const size_t line = (g * kKept + round) * kRound + static_cast<size_t>(place) * kLine
                              + static_cast<size_t>(turned) * carry::kPieceBytes;
          for (int i = 0; i < carry::kPieceBytes; ++i) {
            const uint8_t want =
                here ? values[static_cast<size_t>(list[at]) * kLine + piece * carry::kPieceBytes + i] : 0;
            wrong += got[line + i] != want;
            ++checked;
          }
        }
      }
  }
  long long off_boundary = 0;
  for (const int v : off_at) off_boundary += v != 0;
  b.check(name, wrong == 0 && off_boundary == 0,
          std::to_string(wrong) + " of " + std::to_string(checked) + " tile bytes differ; "
              + std::to_string(off_boundary) + " tiles off a boundary");
  const double lines_a_cta = static_cast<double>(live) / b.sms;
  const lab::Samples cycles =
      b.variant([&] { fetched<<<b.sms, 32 * kWarps>>>(b.times, dwords, dvalues, kTimed); },
                [&] { return b.span(kWarps) / lines_a_cta; });
  b.record(name, "cycles a 128-byte row a CTA", cycles);
  for (void* p : {static_cast<void*>(dwords), static_cast<void*>(dfirst), static_cast<void*>(dout),
                  static_cast<void*>(doff_at)})
    LAB_CHECK(cudaFree(p));
}

int main(int argc, char** argv) {
  using namespace carry_fetch;
  lab::Bench b(argc, argv);
  if (b.expect_only) return 0;
  std::vector<uint8_t> values(static_cast<size_t>(kValueWindows) * kWindow * kLine);
  for (size_t i = 0; i < values.size(); ++i) values[i] = value_byte(i);
  uint8_t* dvalues = nullptr;
  LAB_CHECK(cudaMalloc(&dvalues, values.size()));
  LAB_CHECK(cudaMemcpy(dvalues, values.data(), values.size(), cudaMemcpyHostToDevice));
  run(b, dvalues, values, "a window of 64 live tokens: listed, and their value lines fetched", 16);
  run(b, dvalues, values, "every token of a window live: listed, and their value lines fetched", 1);
  return 0;
}
