// measure/lab/experiments/value_fetch.cu -- QUESTION: a token's 128-byte value fetched by index straight into shared
// memory by the THREADS' OWN ASYNCHRONOUS COPIES (`cp.async.cg`, 16 bytes a thread an instruction, no register holds
// the data): what does a scattered row cost a CTA, by the warps issuing, by the table, and where a unit's 16 CTAs ask
// for the same rows -- beside a warp's own loads (`value_gather`: 8.5 cycles with four warps) and the bulk engine's
// request (`state_move`: 32.8)?
// METHOD: a CTA an SM of 4 or 16 warps; a warp's instruction copies 4 rows, 8 lanes a row, into the warp's 16-row
// ring in shared memory. The table is 1,024 rows (a window's values, 128 KB), 65,536 (8 MB) or 1,048,576 (128 MB); a
// CTA takes its own rows, or its unit's (16 CTAs the same rows). The ring's last rows are checked exactly. FORMS, at 16
// warps on the window's table: the copy's size a register (the form that zero-fills a dead copy); the ring a tensor
// step's tile, a row's 8 pieces in bricks 17 lines apart; both; and BY BRICK -- a lane a row, an instruction one piece
// of the warp's 32 rows into that piece's run of 32 consecutive lines.
// TIMELINE: a warp's trip is a batch of 8 rows: two copy instructions, a commit, and a wait for the batch before it
// -- two batches in flight. Live: nothing of the data.
// see docs/internals/tools/lab.md#value-fetch
#include "lab.cuh"

namespace value_fetch {

constexpr int kPer = 4, kBatch = 2 * kPer, kRing = 2 * kBatch, kRowWords = 32, kRowBytes = 4 * kRowWords;
constexpr int kUnit = 16;
//: THE FORMS of a row beside the plain one: the copy's size a register (the form that zero-fills a dead copy), and
//: the ring a tensor step's tile -- a row's 8 pieces in 8 bricks 17 lines apart
constexpr int kSized = 1, kBricks = 2;
constexpr int kTileLines = 8 * 17, kTileWords = kTileLines * 4;

//: the table's word `w`, and the row an (owner, warp, fetch) takes of `rows` (a power of two)
constexpr uint32_t table_of(uint32_t w) { return w * 2246822519u + 1u; }
constexpr uint32_t row_of(uint32_t owner, uint32_t warp, uint32_t k, uint32_t rows) {
  return (((owner * 16 + warp) * 2654435761u + k * 40503u) * 2246822519u >> 8) & (rows - 1);
}

template <int WARPS, int FORM>
__global__ void __launch_bounds__(32 * WARPS, 1)
    fetch(long long* times, const uint32_t* table, uint32_t* out, int rows, int fetches, int shared) {
  __shared__ alignas(128) uint32_t ring[WARPS * kTileWords];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint32_t owner = shared != 0 ? blockIdx.x / kUnit : blockIdx.x;
  //: a lane's 16 bytes of its row: the row of an instruction's four the lane is in, and its place in that row
  const uint32_t within = static_cast<uint32_t>(lane & 7) * 16u;
  const uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(ring + warp * kTileWords));
  const uint32_t size = lab::held(16u);
  const uint8_t* const bytes = reinterpret_cast<const uint8_t*>(table) + within;
  const auto issue = [&](int batch) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int slot = (batch & 1) * kBatch + i * kPer + (lane >> 3);
      const uint32_t row = row_of(owner, warp, batch * kBatch + i * kPer + (lane >> 3), rows);
      const uint8_t* const src = bytes + static_cast<long long>(row) * kRowBytes;
      const uint32_t piece = static_cast<uint32_t>(lane & 7);
      const uint32_t to = base + ((FORM & kBricks) != 0 ? (piece * 17u + slot) * 16u : slot * kRowBytes + within);
      if constexpr ((FORM & kSized) != 0) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(to), "l"(src), "r"(size) : "memory");
      } else {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(to), "l"(src) : "memory");
      }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  };
  issue(0);

  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int b = 1; b < fetches / kBatch; ++b) {
    issue(b);
    asm volatile("cp.async.wait_group 1;" ::: "memory");
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  clock.stop(times);
  __syncthreads();
  //: the ring, for the host
#pragma unroll 1
  for (int i = threadIdx.x; i < WARPS * kTileWords; i += 32 * WARPS)
    out[static_cast<long long>(blockIdx.x) * (16 * kTileWords) + i] = ring[i];
}

//: THE FORM BY BRICK: a lane a row, an instruction ONE PIECE of the warp's 32 rows into that piece's run -- the tile a
//: tensor step reads, brick `i`'s 32 rows consecutive lines, so an instruction's 32 writes are 512 contiguous bytes
//: where its 32 reads are 16 bytes of 32 rows. A batch is 32 rows, 8 instructions; two batches in flight
constexpr int kPieces = 8, kRunRows = 32, kRunWords = kPieces * kRunRows * 4;

template <int WARPS>
__global__ void __launch_bounds__(32 * WARPS, 1)
    fetch_runs(long long* times, const uint32_t* table, uint32_t* out, int rows, int fetches) {
  __shared__ alignas(128) uint32_t ring[WARPS * 2 * kRunWords];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(ring + warp * 2 * kRunWords));
  const uint8_t* const bytes = reinterpret_cast<const uint8_t*>(table);
  const auto issue = [&](int batch) {
    const uint32_t row = row_of(blockIdx.x, warp, batch * kRunRows + lane, rows);
    const uint8_t* const src = bytes + static_cast<long long>(row) * kRowBytes;
    const uint32_t to = base + static_cast<uint32_t>((batch & 1) * kRunWords * 4 + lane * 16);
#pragma unroll
    for (int i = 0; i < kPieces; ++i) {
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(to + i * kRunRows * 16), "l"(src + 16 * i) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  };
  issue(0);

  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int b = 1; b < fetches / kRunRows; ++b) {
    issue(b);
    asm volatile("cp.async.wait_group 1;" ::: "memory");
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  clock.stop(times);
  __syncthreads();
#pragma unroll 1
  for (int i = threadIdx.x; i < WARPS * 2 * kRunWords; i += 32 * WARPS)
    out[static_cast<long long>(blockIdx.x) * (16 * 2 * kRunWords) + i] = ring[i];
}

}  // namespace value_fetch

template <int WARPS, int FORM = 0>
void run(lab::Bench& b, const uint32_t* table, uint32_t* out, int rows, int shared) {
  using namespace value_fetch;
  constexpr int kFetches = 8192;
  const std::string name = std::to_string(WARPS) + " warps, a table of " + std::to_string(rows) + " rows, "
                           + (shared != 0 ? "a unit's 16 CTAs the same rows" : "a CTA its own rows")
                           + ((FORM & kSized) != 0 ? ", the copy's size a register" : "")
                           + ((FORM & kBricks) != 0 ? ", a row's pieces in bricks 17 lines apart" : "");
  fetch<WARPS, FORM><<<b.sms, 32 * WARPS>>>(b.times, table, out, rows, kFetches, shared);
  LAB_CHECK(cudaDeviceSynchronize());
  const size_t per = 16 * kTileWords;
  std::vector<uint32_t> got(static_cast<size_t>(b.sms) * per);
  LAB_CHECK(cudaMemcpy(got.data(), out, got.size() * 4, cudaMemcpyDeviceToHost));
  long long wrong = 0;
  for (int blk = 0; blk < b.sms; ++blk)
    for (int w = 0; w < WARPS; ++w)
      for (int slot = 0; slot < kRing; ++slot) {
        //: the ring's slot holds the last batch pair's row of that slot
        const uint32_t owner = shared != 0 ? blk / kUnit : blk;
        const uint32_t row = row_of(owner, w, kFetches - kRing + slot, rows);
        for (int i = 0; i < kRowWords; ++i) {
          //: word `i` of the row: in the plain ring at the row's place, in the tile in piece `i / 4`'s brick
          const size_t at = (FORM & kBricks) != 0 ? ((i / 4) * 17 + slot) * 4 + i % 4 : slot * kRowWords + i;
          wrong += got[blk * per + static_cast<size_t>(w) * kTileWords + at] != table_of(row * kRowWords + i);
        }
      }
  b.check(name, wrong == 0,
          std::to_string(wrong) + " of " + std::to_string(b.sms * WARPS * kRing * kRowWords) + " ring words differ");
  const lab::Samples cycles = b.variant(
      [&] { fetch<WARPS, FORM><<<b.sms, 32 * WARPS>>>(b.times, table, out, rows, kFetches, shared); },
      [&] { return b.span(WARPS) / (static_cast<double>(kFetches) * WARPS); });
  b.record(name, "cycles a 128-byte row a CTA", cycles);
}

template <int WARPS>
void run_runs(lab::Bench& b, const uint32_t* table, uint32_t* out, int rows) {
  using namespace value_fetch;
  constexpr int kFetches = 8192, kBatches = kFetches / kRunRows;
  const std::string name = std::to_string(WARPS) + " warps, a table of " + std::to_string(rows)
                           + " rows, a CTA its own rows, an instruction one piece of 32 rows into its brick's run";
  fetch_runs<WARPS><<<b.sms, 32 * WARPS>>>(b.times, table, out, rows, kFetches);
  LAB_CHECK(cudaDeviceSynchronize());
  const size_t per = 16 * 2 * kRunWords;
  std::vector<uint32_t> got(static_cast<size_t>(b.sms) * per);
  LAB_CHECK(cudaMemcpy(got.data(), out, got.size() * 4, cudaMemcpyDeviceToHost));
  long long wrong = 0;
  for (int blk = 0; blk < b.sms; ++blk)
    for (int w = 0; w < WARPS; ++w)
      for (int batch = kBatches - 2; batch < kBatches; ++batch)
        for (int lane = 0; lane < kRunRows; ++lane) {
          const uint32_t row = row_of(blk, w, batch * kRunRows + lane, rows);
          for (int i = 0; i < kRowWords; ++i) {
            const size_t at = static_cast<size_t>(batch & 1) * kRunWords + ((i / 4) * kRunRows + lane) * 4 + i % 4;
            wrong += got[blk * per + static_cast<size_t>(w) * 2 * kRunWords + at] != table_of(row * kRowWords + i);
          }
        }
  b.check(name, wrong == 0,
          std::to_string(wrong) + " of " + std::to_string(b.sms * WARPS * 2 * kRunWords) + " ring words differ");
  const lab::Samples cycles = b.variant(
      [&] { fetch_runs<WARPS><<<b.sms, 32 * WARPS>>>(b.times, table, out, rows, kFetches); },
      [&] { return b.span(WARPS) / (static_cast<double>(kFetches) * WARPS); });
  b.record(name, "cycles a 128-byte row a CTA", cycles);
}

int main(int argc, char** argv) {
  using namespace value_fetch;
  lab::Bench b(argc, argv);
  for (int warps : {4, 16}) b.expect(lab::kernel("fetch", {warps, 0}), "LDGSTS", ">=", 2);
  if (b.expect_only) return 0;
  constexpr int kWindow = 1024, kSmall = 65536, kLarge = 1048576;
  std::vector<uint32_t> host(static_cast<size_t>(kLarge) * kRowWords);
  for (size_t w = 0; w < host.size(); ++w) host[w] = table_of(static_cast<uint32_t>(w));
  uint32_t *table = nullptr, *out = nullptr;
  LAB_CHECK(cudaMalloc(&table, host.size() * 4));
  LAB_CHECK(cudaMemcpy(table, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMalloc(&out, static_cast<size_t>(b.sms) * 16 * 2 * kRunWords * 4));
  run<4>(b, table, out, kSmall, 0);
  run<16>(b, table, out, kSmall, 0);
  run<4>(b, table, out, kLarge, 0);
  run<16>(b, table, out, kLarge, 0);
  run<16>(b, table, out, kWindow, 0);
  run<16>(b, table, out, kWindow, 1);
  run<16, kSized>(b, table, out, kWindow, 0);
  run<16, kBricks>(b, table, out, kWindow, 0);
  run<16, kSized | kBricks>(b, table, out, kWindow, 0);
  run_runs<16>(b, table, out, kWindow);
  return 0;
}
