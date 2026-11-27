// measure/lab/experiments/mask_pack.cu -- QUESTION: the producer writes a level-side's mask as 32-token words, a bit
// the u8 digit is nonzero (Blake's ruling of 2026-11-27), from the epilogue's registers: what does the packer cost a
// 64-token block of 256 digits laid in a `wgmma` m64n256 accumulator (a thread's `d[4 q + 2 j + i]` is row `16 w + g +
// 8 j`, column `8 q + 2 t + i`)? METHOD: a warpgroup holds its block's quantized values (`255 level / max`, fp32) in
// that layout, loaded before the clock; a digit is nonzero where its value rounds above 0 (`> 0.5`, the FFMA's
// rounding). THE PACKER: a ballot a register index (8 tokens x 4 columns a ballot); lane `q` keeps the four ballots of
// its column group `q` (a select a ballot); each of its 8 columns' 16 token bits are the ballots' every fourth bit,
// gathered by three shift-and-mask rounds; a warp's 16-bit halves go to shared memory as one 16-byte store a lane, a
// named barrier joins the warp pair of a 32-token half, and each thread of the pair writes 4 words (16 bytes) of the
// half's 256 words to global memory. Every word is checked against the host's bits; densities 10%, 50%, 100%.
// TIMELINE: no load in the clock. A trip: 128 compares and ballots with their keeps, the gathers, the halves' store,
// the pair's barrier, the words' load and store. Live: the 128 values, the four kept ballots, 8 halves.
// see docs/internals/tools/lab.md#mask-pack
#include "lab.cuh"

namespace mask_pack {

constexpr int kWidth = 256, kValues = 128, kThreads = 128;
//: shared memory: a warp's halves [16 bits a column] at `warp * 512`, so a pair's are 512 bytes apart
constexpr int kSmem = 4 * kWidth * 2;

//: the 8 bits at every fourth position from `t` of `b`, gathered
__device__ __forceinline__ uint32_t gather4(uint32_t b, int t) {
  uint32_t v = (b >> t) & 0x11111111u;
  v = (v | (v >> 3)) & 0x03030303u;
  v = (v | (v >> 6)) & 0x000f000fu;
  return (v | (v >> 12)) & 0xffu;
}

__global__ void __launch_bounds__(kThreads, 1)
    pack(long long* times, const float* digits, uint32_t* words, float threshold, int trips) {
  __shared__ __align__(16) uint16_t halves[4][kWidth];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const long long row0 = static_cast<long long>(blockIdx.x) * 64 + 16 * warp + g;
  float d[kValues];
#pragma unroll
  for (int j = 0; j < 2; ++j)
#pragma unroll
    for (int q = 0; q < 32; ++q) {
      const float2 v = *reinterpret_cast<const float2*>(digits + (row0 + 8 * j) * kWidth + 8 * q + 2 * t);
      d[4 * q + 2 * j] = v.x;
      d[4 * q + 2 * j + 1] = v.y;
    }
  //: the pair of warps of a 32-token half, and its named barrier
  const int pair = warp >> 1;
  uint32_t* const out = words + (static_cast<long long>(blockIdx.x) * 2 + pair) * kWidth;
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int trip = 0; trip < trips; ++trip) {
    const float above = __uint_as_float(lab::held(__float_as_uint(threshold)));
    //: THE BALLOTS: lane `q` keeps the four of its column group
    uint32_t kept[4] = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int q = 0; q < 32; ++q)
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const uint32_t b = __ballot_sync(0xffffffffu, d[4 * q + r] > above);
        kept[r] = lane == q ? b : kept[r];
      }
    //: THE GATHERS: column `8 q + 2 t + i`'s 16 bits, tokens `g` (from row half 0) then `8 + g` (half 1)
    uint32_t h[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int ct = c, lo0 = gather4(kept[0], ct) | (gather4(kept[2], ct) << 8);
      const int lo1 = gather4(kept[1], ct) | (gather4(kept[3], ct) << 8);
      h[c] = static_cast<uint32_t>(lo0) | (static_cast<uint32_t>(lo1) << 16);
    }
    *reinterpret_cast<uint4*>(&halves[warp][8 * lane]) = make_uint4(h[0], h[1], h[2], h[3]);
    lab::bar_sync(1 + pair, 64);
    //: THE WORDS: column `c`'s word is its low half (the pair's first warp) and high half (its second)
    const int c0 = 4 * (tid & 63);
    const uint2 lo = *reinterpret_cast<const uint2*>(&halves[2 * pair][c0]);
    const uint2 hi = *reinterpret_cast<const uint2*>(&halves[2 * pair + 1][c0]);
    *reinterpret_cast<uint4*>(out + c0) = make_uint4(__byte_perm(lo.x, hi.x, 0x5410), __byte_perm(lo.x, hi.x, 0x7632),
                                                     __byte_perm(lo.y, hi.y, 0x5410), __byte_perm(lo.y, hi.y, 0x7632));
    lab::bar_sync(1 + pair, 64);
  }
  clock.stop(times);
}

}  // namespace mask_pack

void run(lab::Bench& b, int percent, float* digits, uint32_t* words) {
  using namespace mask_pack;
  const long long rows = static_cast<long long>(b.sms) * 64;
  std::vector<float> host(rows * kWidth);
  uint32_t s = 12345u + percent;
  for (auto& v : host) {
    s = s * 1664525u + 1013904223u;
    const bool live = (s >> 8) % 100 < static_cast<uint32_t>(percent);
    s = s * 1664525u + 1013904223u;
    //: a live digit's value past 0.5 (1 to 255), a dead one's at or below it
    v = live ? 0.51f + static_cast<float>((s >> 8) % 25450) / 100.0f : static_cast<float>((s >> 8) % 51) / 100.0f;
  }
  LAB_CHECK(cudaMemcpy(digits, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
  const std::string name = "256 columns, " + std::to_string(percent) + "% of digits nonzero";
  pack<<<b.sms, kThreads>>>(b.times, digits, words, 0.5f, 1);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> got(static_cast<size_t>(b.sms) * 2 * kWidth);
  LAB_CHECK(cudaMemcpy(got.data(), words, got.size() * 4, cudaMemcpyDeviceToHost));
  long long wrong = 0;
  for (size_t group = 0; group < got.size() / kWidth; ++group)
    for (int c = 0; c < kWidth; ++c) {
      uint32_t want = 0;
      for (int k = 0; k < 32; ++k) want |= static_cast<uint32_t>(host[(group * 32 + k) * kWidth + c] > 0.5f) << k;
      wrong += got[group * kWidth + c] != want;
    }
  b.check(name, wrong == 0, std::to_string(wrong) + " of " + std::to_string(got.size()) + " mask words differ");
  constexpr int kTrips = 256;
  const lab::Samples cycles = b.variant([&] { pack<<<b.sms, kThreads>>>(b.times, digits, words, 0.5f, kTrips); },
                                        [&] { return b.span(kThreads >> 5) / kTrips; });
  b.record(name + ": a 64-token block", "cycles a block", cycles);
}

int main(int argc, char** argv) {
  using namespace mask_pack;
  lab::Bench b(argc, argv);
  b.expect("mask_pack::pack\\(", "VOTE", ">=", 128);
  if (b.expect_only) return 0;
  float* digits = nullptr;
  uint32_t* words = nullptr;
  LAB_CHECK(cudaMalloc(&digits, static_cast<size_t>(b.sms) * 64 * kWidth * 4));
  LAB_CHECK(cudaMalloc(&words, static_cast<size_t>(b.sms) * 2 * kWidth * 4));
  for (int percent : {10, 50, 100}) run(b, percent, digits, words);
  return 0;
}
