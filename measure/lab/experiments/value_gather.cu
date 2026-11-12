// measure/lab/experiments/value_gather.cu -- QUESTION: a token's 128-byte value fetched by a WARP'S OWN LOAD from a
// row scattered over a table, and stored into shared memory: what does a row cost a CTA, by the warps issuing and
// by whether the cache holds the table -- beside the bulk engine's 128-byte request (`state_move`: 32.8 cycles)?
// METHOD: a CTA an SM of 1 or 4 warps; each warp fetches its own scattered rows, a lane 4 bytes of a row, and
// stores each into its 16-row ring in shared memory. The table is 65,536 rows (8 MB) or 1,048,576 (128 MB). The
// ring's last rows are checked exactly.
// TIMELINE: a warp's trip is two batches of 8 rows, each at its own registers: a batch's 8 stores (its rows loaded
// a trip before), then those registers' 8 loads for the batch two on. Live: 16 words a thread.
// see docs/internals/tools/lab.md#value-gather
#include "lab.cuh"

namespace value_gather {

constexpr int kBatch = 8, kRing = 2 * kBatch, kRowWords = 32;

//: the table's word `w`, and the row a (CTA, warp, fetch) takes of `rows` (a power of two)
constexpr uint32_t table_of(uint32_t w) { return w * 2246822519u + 1u; }
constexpr uint32_t row_of(uint32_t block, uint32_t warp, uint32_t k, uint32_t rows) {
  return (((block * 4 + warp) * 2654435761u + k * 40503u) * 2246822519u >> 8) & (rows - 1);
}

template <int WARPS>
__global__ void __launch_bounds__(32 * WARPS, 1)
    gather(long long* times, const uint32_t* table, uint32_t* out, int rows, int fetches) {
  __shared__ alignas(128) uint32_t ring[WARPS * kRing * kRowWords];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint32_t* const mine = ring + warp * kRing * kRowWords + lane;
  uint32_t held[2][kBatch];
  const auto load = [&](int h, int batch) {
#pragma unroll
    for (int i = 0; i < kBatch; ++i) {
      const uint32_t row = row_of(blockIdx.x, warp, min(batch * kBatch + i, fetches - 1), rows);
      held[h][i] = table[static_cast<long long>(row) * kRowWords + lane];
    }
  };
  load(0, 0);
  load(1, 1);

  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int b = 0; b < fetches / kBatch; b += 2) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
      for (int i = 0; i < kBatch; ++i) mine[(h * kBatch + i) * kRowWords] = held[h][i];
      load(h, b + h + 2);
    }
  }
  clock.stop(times);
  __syncthreads();
  //: the ring, for the host
#pragma unroll 1
  for (int i = threadIdx.x; i < WARPS * kRing * kRowWords; i += 32 * WARPS)
    out[static_cast<long long>(blockIdx.x) * (4 * kRing * kRowWords) + i] = ring[i];
}

}  // namespace value_gather

template <int WARPS>
void run(lab::Bench& b, const uint32_t* table, uint32_t* out, int rows) {
  using namespace value_gather;
  constexpr int kFetches = 8192;
  const std::string name = std::to_string(WARPS) + (WARPS == 1 ? " warp" : " warps") + ", a table of "
                           + std::to_string(rows) + " rows";
  gather<WARPS><<<b.sms, 32 * WARPS>>>(b.times, table, out, rows, kFetches);
  LAB_CHECK(cudaDeviceSynchronize());
  const size_t per = 4 * kRing * kRowWords;
  std::vector<uint32_t> got(static_cast<size_t>(b.sms) * per);
  LAB_CHECK(cudaMemcpy(got.data(), out, got.size() * 4, cudaMemcpyDeviceToHost));
  long long wrong = 0;
  for (int blk = 0; blk < b.sms; ++blk)
    for (int w = 0; w < WARPS; ++w)
      for (int slot = 0; slot < kRing; ++slot) {
        //: the ring's slot holds the last batch pair's row of that slot
        const uint32_t row = row_of(blk, w, kFetches - kRing + slot, rows);
        for (int i = 0; i < kRowWords; ++i)
          wrong += got[blk * per + (static_cast<size_t>(w) * kRing + slot) * kRowWords + i]
                   != table_of(row * kRowWords + i);
      }
  b.check(name, wrong == 0,
          std::to_string(wrong) + " of " + std::to_string(b.sms * WARPS * kRing * kRowWords) + " ring words differ");
  const lab::Samples cycles = b.variant(
      [&] { gather<WARPS><<<b.sms, 32 * WARPS>>>(b.times, table, out, rows, kFetches); },
      [&] { return b.span(WARPS) / (static_cast<double>(kFetches) * WARPS); });
  b.record(name, "cycles a 128-byte row a CTA", cycles);
}

int main(int argc, char** argv) {
  using namespace value_gather;
  lab::Bench b(argc, argv);
  for (int warps : {1, 4}) b.expect(lab::kernel("gather", {warps}), "LDG", ">=", 2 * kBatch);
  if (b.expect_only) return 0;
  constexpr int kSmall = 65536, kLarge = 1048576;
  std::vector<uint32_t> host(static_cast<size_t>(kLarge) * kRowWords);
  for (size_t w = 0; w < host.size(); ++w) host[w] = table_of(static_cast<uint32_t>(w));
  uint32_t *table = nullptr, *out = nullptr;
  LAB_CHECK(cudaMalloc(&table, host.size() * 4));
  LAB_CHECK(cudaMemcpy(table, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMalloc(&out, static_cast<size_t>(b.sms) * 4 * kRing * kRowWords * 4));
  for (int rows : {kSmall, kLarge}) {
    run<1>(b, table, out, rows);
    run<4>(b, table, out, rows);
  }
  return 0;
}
