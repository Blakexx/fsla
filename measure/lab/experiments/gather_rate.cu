// measure/lab/experiments/gather_rate.cu -- QUESTION: what does the gather's own access pattern cost a row -- rows
// fetched scattered from a table, stored into shared memory in the order they will be read, and the image stored out
// to global memory by one bulk copy -- when the table is one the card's cache holds (8 MB: `load_rate`,
// `state_move`) and when it is not (2 GB), and when the image goes to a ring the cache holds or to fresh memory?
// METHOD: a CTA an SM of 16 warps. A round is 64 DEPTH rows of 128 bytes: a lane loads 16 bytes, an instruction four
// rows, DEPTH instructions a warp in flight (loaded a round before their use). LOAD folds every word into a chain a
// lane; STORE stores a round's rows into one of three images, a warp's lanes on 32 banks; OUT then stores the image
// to the CTA's span of global memory by one bulk copy, a round's copy waited two rounds on so one barrier a round
// orders the images' reuse. Every form is checked exactly.
// TIMELINE: a round is its stores (or its chain), the next round's loads issued, and for OUT the writers' fence, the
// CTA's barrier and thread 0's copy, commit and wait on the copy before. Live: DEPTH loads of 16 bytes a thread.
// see docs/internals/tools/lab.md#gather-rate
#include "lab.cuh"

namespace gather_rate {

constexpr int kWarps = 16, kThreads = 32 * kWarps, kRowWords = 32, kImages = 3;
enum Form { kLoad, kStore, kOut };
//: a round's rows, and an image's 16-byte lines and bytes
template <int DEPTH> constexpr int kRound = 4 * kWarps * DEPTH;
template <int DEPTH> constexpr int kLines = kRound<DEPTH> * 8;
template <int DEPTH> constexpr int kBytes = kLines<DEPTH> * 16;

//: the table's word `w`, and the row a (CTA, round, slot of the round) takes of a table of `mask + 1` rows
constexpr uint32_t table_of(uint32_t w) { return w * 2246822519u + 1u; }
constexpr uint32_t row_of(uint32_t block, uint32_t round, uint32_t slot, uint32_t mask) {
  return ((((block * 4099u + round) * 2654435761u + slot * 40503u) * 2246822519u) >> 7) & mask;
}

__global__ void fill(uint32_t* table, long long words) {
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long w = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; w < words; w += stride)
    table[w] = table_of(static_cast<uint32_t>(w));
}

//: `ring` is the rounds a CTA's span of `out` holds before it is written over
template <int FORM, int DEPTH>
__global__ void __launch_bounds__(kThreads, 1)
    fetch(long long* times, const uint4* table, uint4* out, uint32_t mask, int rounds, int ring) {
  extern __shared__ __align__(128) uint4 image[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t base = lab::smem(image);
  uint4 held[DEPTH];
  //: instruction `i` of a warp takes the round's slots `4 (16 i + warp)` to that and 3: 8 lanes a row
  const auto load = [&](int r) {
#pragma unroll
    for (int i = 0; i < DEPTH; ++i) {
      const uint32_t row = row_of(blockIdx.x, min(r, rounds - 1), 4 * (i * kWarps + warp) + (lane >> 3), mask);
      held[i] = table[static_cast<long long>(row) * 8 + (lane & 7)];
    }
  };
  uint32_t sum = 0;
  load(0);
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
    if constexpr (FORM == kLoad) {
#pragma unroll
      for (int i = 0; i < DEPTH; ++i) sum = sum * 3u + (held[i].x + held[i].y + held[i].z + held[i].w);
    } else {
      uint4* const to = image + (r % kImages) * kLines<DEPTH>;
#pragma unroll
      for (int i = 0; i < DEPTH; ++i) to[(4 * (i * kWarps + warp) + (lane >> 3)) * 8 + (lane & 7)] = held[i];
    }
    load(r + 1);
    if constexpr (FORM == kOut) {
      //: THE ROUND'S ONE BARRIER: its image whole behind it; and thread 0 passes the next only with the copy before
      //: this one done, so the image a round stores into was copied out two rounds back
      lab::bulk_fence();
      __syncthreads();
      if (tid == 0) {
        lab::bulk_store(out + (static_cast<long long>(blockIdx.x) * ring + r % ring) * kLines<DEPTH>,
                        base + (r % kImages) * kBytes<DEPTH>, kBytes<DEPTH>);
        lab::bulk_commit();
        lab::bulk_wait<1>();
      }
    }
  }
  if constexpr (FORM == kOut) {
    if (tid == 0) lab::bulk_wait<0>();
  }
  clock.stop(times);
  //: THE STATE OUT, for the check: the chain, or the last round's image by ordinary stores
  if constexpr (FORM == kLoad) out[static_cast<long long>(blockIdx.x) * kThreads + tid].x = sum;
  if constexpr (FORM == kStore) {
    __syncthreads();
    const uint4* const last = image + ((rounds - 1) % kImages) * kLines<DEPTH>;
#pragma unroll 1
    for (int j = tid; j < kLines<DEPTH>; j += kThreads) out[static_cast<long long>(blockIdx.x) * kLines<DEPTH> + j] = last[j];
  }
}

}  // namespace gather_rate

//: `ring_rounds` 0 is fresh memory: a span as long as the launch
template <int FORM, int DEPTH>
void run(lab::Bench& b, const uint4* table, uint4* out, uint32_t rows, int ring_rounds, const std::string& what) {
  using namespace gather_rate;
  constexpr int kMeasured = 512, kChecked = 4;
  const uint32_t mask = rows - 1;
  const std::string name = std::string(rows > 65536 ? "a 2 GB table" : "an 8 MB table") + ", " + std::to_string(kRound<DEPTH>)
                           + " rows in flight: " + what;
  const auto kernel = fetch<FORM, DEPTH>;
  const size_t shared = FORM == kLoad ? 0 : static_cast<size_t>(kImages) * kBytes<DEPTH>;
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared)));
  kernel<<<b.sms, kThreads, shared>>>(b.times, table, out, mask, kChecked, kChecked);
  LAB_CHECK(cudaDeviceSynchronize());
  const size_t lines = FORM == kLoad ? kThreads : FORM == kStore ? kLines<DEPTH> : static_cast<size_t>(kChecked) * kLines<DEPTH>;
  std::vector<uint32_t> got(static_cast<size_t>(b.sms) * lines * 4);
  LAB_CHECK(cudaMemcpy(got.data(), out, got.size() * 4, cudaMemcpyDeviceToHost));
  long long wrong = 0;
  //: word `w` of line `line` of round `r`'s image, as the table holds it
  const auto word = [&](int blk, int r, int line, int w) {
    return table_of(row_of(blk, r, line >> 3, mask) * kRowWords + 4 * (line & 7) + w);
  };
  for (int blk = 0; blk < b.sms; ++blk) {
    const uint32_t* const mine = got.data() + static_cast<size_t>(blk) * lines * 4;
    if (FORM == kLoad) {
      for (int tid = 0; tid < kThreads; ++tid) {
        uint32_t want = 0;
        for (int r = 0; r < kChecked; ++r)
          for (int i = 0; i < DEPTH; ++i) {
            const int line = (4 * (i * kWarps + (tid >> 5)) + ((tid & 31) >> 3)) * 8 + (tid & 7);
            want = want * 3u + (word(blk, r, line, 0) + word(blk, r, line, 1) + word(blk, r, line, 2) + word(blk, r, line, 3));
          }
        wrong += mine[4 * tid] != want;
      }
    } else {
      for (int r = FORM == kStore ? kChecked - 1 : 0; r < kChecked; ++r)
        for (int line = 0; line < kLines<DEPTH>; ++line)
          for (int w = 0; w < 4; ++w)
            wrong += mine[((FORM == kStore ? 0 : static_cast<size_t>(r) * kLines<DEPTH>) + line) * 4 + w] != word(blk, r, line, w);
    }
  }
  b.check(name, wrong == 0, std::to_string(wrong) + " words differ");
  const int ring = ring_rounds > 0 ? ring_rounds : kMeasured;
  const lab::Samples cycles = b.variant(
      [&] { kernel<<<b.sms, kThreads, shared>>>(b.times, table, out, mask, kMeasured, ring); },
      [&] { return b.span(kWarps) / (static_cast<double>(kMeasured) * kRound<DEPTH>); });
  b.record(name, "cycles a 128-byte row a CTA", cycles);
}

int main(int argc, char** argv) {
  using namespace gather_rate;
  lab::Bench b(argc, argv);
  for (int form : {kLoad, kStore, kOut}) b.expect(lab::kernel("fetch", {form, 8}), "LDG", ">=", 8);
  b.expect(lab::kernel("fetch", {kOut, 4}), "LDG", ">=", 4);
  //: the LOAD form's ring: the toolkit copies the fourth register of one of a round's 8 wide loads at the round's end, so that
  //: instruction's four rows are read in the round that issued it: one load in its trip, declared because the
  //: ring's form keeps the copy, which the row measures
  b.expect(lab::kernel("fetch", {kLoad, 8}), "LOAD_IN_TRIP", "<=", 1);
  //: one thread issues a round's copy and waits the copy before: that branch's pair, once a round
  for (int depth : {4, 8}) b.expect(lab::kernel("fetch", {kOut, depth}), "LOOP_CONVERGENCE", "<=", 2);
  if (b.expect_only) return 0;
  constexpr uint32_t kSmall = 1u << 16, kLarge = 1u << 24;
  constexpr int kMeasured = 512;
  uint32_t* table = nullptr;
  uint4* out = nullptr;
  const long long words = static_cast<long long>(kLarge) * kRowWords;
  LAB_CHECK(cudaMalloc(&table, static_cast<size_t>(words) * 4));
  fill<<<4096, 256>>>(table, words);
  LAB_CHECK(cudaDeviceSynchronize());
  LAB_CHECK(cudaMalloc(&out, static_cast<size_t>(b.sms) * kMeasured * kBytes<8>));
  const uint4* const rows = reinterpret_cast<const uint4*>(table);
  for (uint32_t size : {kSmall, kLarge}) {
    run<kLoad, 8>(b, rows, out, size, 0, "loaded and folded");
    run<kStore, 8>(b, rows, out, size, 0, "loaded and stored into shared memory");
    run<kOut, 8>(b, rows, out, size, 4, "loaded, stored and copied out to a ring of 256 KB a CTA");
    run<kOut, 8>(b, rows, out, size, 0, "loaded, stored and copied out to fresh memory");
    run<kOut, 4>(b, rows, out, size, 8, "loaded, stored and copied out to a ring of 256 KB a CTA");
  }
  return 0;
}
