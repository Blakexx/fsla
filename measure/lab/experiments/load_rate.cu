// measure/lab/experiments/load_rate.cu -- QUESTION: a warp's own load of a scattered 128-byte row costs it about 33
// cycles (`value_gather`), and four warps a quarter of that. What is that cost a property of: the instruction, its
// bytes, the loads in flight, or the warps issuing? METHOD: a CTA an SM of 1, 4, 8 or 16 warps; each warp fetches
// its own scattered rows of a table of 65,536 (8 MB) and folds every word into a chain a lane (`3 s + w`, an order
// no schedule can move a word's use across), nothing stored. NARROW: a lane loads 4 bytes, an instruction a row.
// WIDE: a lane loads 16 bytes, an instruction FOUR rows (8 lanes a row). A ring of 4, 16 or 64 instructions is in
// flight. Every lane's chain is checked exactly.
// TIMELINE: a warp's trip is the ring's two halves, each at its own registers: a half's words folded in (loaded a
// trip before), then those registers' loads for the trip after. Live: the ring's words (16 a thread narrow at 16
// in flight, 64 wide) and the chain.
// see docs/internals/tools/lab.md#load-rate
#include "lab.cuh"

namespace load_rate {

constexpr int kRowWords = 32, kRows = 65536;

//: the table's word `w`, and the row a (CTA, warp, fetch) takes
constexpr uint32_t table_of(uint32_t w) { return w * 2246822519u + 1u; }
constexpr uint32_t row_of(uint32_t block, uint32_t warp, uint32_t k) {
  return (((block * 16 + warp) * 2654435761u + k * 40503u) * 2246822519u >> 8) & (kRows - 1);
}

template <int WARPS, int WIDE, int DEPTH>
__global__ void __launch_bounds__(32 * WARPS, 1)
    fetch(long long* times, const uint32_t* table, uint32_t* out, int trips) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int last = trips * DEPTH - 1;
  uint32_t sum = 0;
  lab::WarpSpan clock;
  if constexpr (WIDE) {
    //: instruction `k` takes rows `4 k` to `4 k + 3`: 8 lanes a row, a lane 16 of its bytes
    const uint4* const wide = reinterpret_cast<const uint4*>(table);
    uint4 held[DEPTH];
    const auto load = [&](int k) {
      const uint32_t row = row_of(blockIdx.x, warp, 4 * min(k, last) + (lane >> 3));
      return wide[static_cast<long long>(row) * (kRowWords / 4) + (lane & 7)];
    };
#pragma unroll
    for (int i = 0; i < DEPTH; ++i) held[i] = load(i);
    clock.start();
#pragma unroll 1
    for (int t = 1; t <= trips; ++t) {
#pragma unroll
      for (int h = 0; h < DEPTH; h += DEPTH / 2) {
#pragma unroll
        for (int i = h; i < h + DEPTH / 2; ++i) sum = sum * 3u + (held[i].x + held[i].y + held[i].z + held[i].w);
#pragma unroll
        for (int i = h; i < h + DEPTH / 2; ++i) held[i] = load(t * DEPTH + i);
      }
    }
  } else {
    constexpr int kHalf = DEPTH / 2;
    uint32_t held[2][kHalf];
    const auto load = [&](int h, int trip) {
#pragma unroll
      for (int i = 0; i < kHalf; ++i) {
        const uint32_t row = row_of(blockIdx.x, warp, min(trip * DEPTH + h * kHalf + i, last));
        held[h][i] = table[static_cast<long long>(row) * kRowWords + lane];
      }
    };
    load(0, 0);
    load(1, 0);
    clock.start();
#pragma unroll 1
    for (int t = 1; t <= trips; ++t) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int i = 0; i < kHalf; ++i) sum = sum * 3u + held[h][i];
        load(h, t);
      }
    }
  }
  clock.stop(times);
  out[static_cast<long long>(blockIdx.x) * 512 + threadIdx.x] = sum;
}

}  // namespace load_rate

template <int WARPS, int WIDE, int DEPTH>
void run(lab::Bench& b, const std::vector<uint32_t>& host, const uint32_t* table, uint32_t* out) {
  using namespace load_rate;
  constexpr int kPer = WIDE ? 4 : 1, kTrips = 8192 / DEPTH, kCheckTrips = 4;
  const std::string form = WIDE ? "16 bytes a lane, 4 rows an instruction" : "4 bytes a lane, a row an instruction";
  const std::string name = form + ", " + std::to_string(WARPS) + (WARPS == 1 ? " warp, " : " warps, ")
                           + std::to_string(DEPTH) + " in flight";
  const auto kernel = fetch<WARPS, WIDE, DEPTH>;
  kernel<<<b.sms, 32 * WARPS>>>(b.times, table, out, kCheckTrips);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> got(static_cast<size_t>(b.sms) * 512);
  LAB_CHECK(cudaMemcpy(got.data(), out, got.size() * 4, cudaMemcpyDeviceToHost));
  long long wrong = 0;
  for (int blk = 0; blk < b.sms; ++blk)
    for (int tid = 0; tid < 32 * WARPS; ++tid) {
      const int w = tid >> 5, lane = tid & 31;
      uint32_t want = 0;
      for (int k = 0; k < kCheckTrips * DEPTH; ++k) {
        if (WIDE) {
          const size_t at = static_cast<size_t>(row_of(blk, w, 4 * k + (lane >> 3))) * kRowWords + 4 * (lane & 7);
          want = want * 3u + (host[at] + host[at + 1] + host[at + 2] + host[at + 3]);
        } else {
          want = want * 3u + host[static_cast<size_t>(row_of(blk, w, k)) * kRowWords + lane];
        }
      }
      wrong += got[static_cast<size_t>(blk) * 512 + tid] != want;
    }
  b.check(name, wrong == 0, std::to_string(wrong) + " of " + std::to_string(b.sms * 32 * WARPS) + " lane sums differ");
  const lab::Samples cycles = b.variant(
      [&] { kernel<<<b.sms, 32 * WARPS>>>(b.times, table, out, kTrips); },
      [&] { return b.span(WARPS) / (static_cast<double>(kTrips) * DEPTH * kPer * WARPS); });
  b.record(name, "cycles a 128-byte row a CTA", cycles);
}

int main(int argc, char** argv) {
  using namespace load_rate;
  lab::Bench b(argc, argv);
  for (int wide = 0; wide < 2; ++wide)
    for (int warps : {1, 4, 8, 16}) {
      b.expect(lab::kernel("fetch", {warps, wide, 16}), "LDG", ">=", 16);
      //: the narrow ring: the toolkit copies two of a trip's loaded words at the trip's end, whatever the source's
      //: form, so those two rows' loads run half a trip ahead of their use, not a whole one
      if (!wide) b.expect(lab::kernel("fetch", {warps, 0, 16}), "LOAD_IN_TRIP", "<=", 2);
    }
  for (int depth : {4, 64}) {
    b.expect(lab::kernel("fetch", {4, 0, depth}), "LDG", ">=", depth);
    b.expect(lab::kernel("fetch", {4, 0, depth}), "LOAD_IN_TRIP", "<=", 2);
  }
  if (b.expect_only) return 0;
  std::vector<uint32_t> host(static_cast<size_t>(kRows) * kRowWords);
  for (size_t w = 0; w < host.size(); ++w) host[w] = table_of(static_cast<uint32_t>(w));
  uint32_t *table = nullptr, *out = nullptr;
  LAB_CHECK(cudaMalloc(&table, host.size() * 4));
  LAB_CHECK(cudaMemcpy(table, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMalloc(&out, static_cast<size_t>(b.sms) * 512 * 4));
  run<1, 0, 16>(b, host, table, out);
  run<4, 0, 16>(b, host, table, out);
  run<8, 0, 16>(b, host, table, out);
  run<16, 0, 16>(b, host, table, out);
  run<1, 1, 16>(b, host, table, out);
  run<4, 1, 16>(b, host, table, out);
  run<8, 1, 16>(b, host, table, out);
  run<16, 1, 16>(b, host, table, out);
  run<4, 0, 4>(b, host, table, out);
  run<4, 0, 64>(b, host, table, out);
  return 0;
}
