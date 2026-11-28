// measure/lab/experiments/l2_policy.cu -- QUESTION: the producer streams what it reads once (its `x` tile, its outputs)
// beside what every CTA reuses (a write head's weights from L2). Does an L2 policy -- the stream's loads `evict_first`,
// the table's `evict_last` (`createpolicy` and the loads' `L2::cache_hint`) -- keep the table and save time, against
// plain loads? METHOD: a CTA an SM of 512 threads; a trip loads 8 units (8 KB a CTA each, 16 bytes a lane) of a 1 GiB
// stream, each CTA its own slice once, and 8 units of a shared table every CTA walks from its own offset, round and
// round; every word folded into a chain a lane, nothing stored. Tables of 16 MB and 40 MB (the L2 is 50 MB), with and
// without the hints, a kernel a table size (so the profiler reads each table's own hit rate). Time a trip; the table's
// hit rate is read by the lab's `ncu` (the L2's read hits over the table's sectors: the stream's lines are read once).
// Every chain is checked exactly.
// TIMELINE: a thread's trip is the ring's two halves at their own registers (the stream's 8 units, the table's 8): a
// half folded, then its loads for the trip after. Live: the ring's 64 words, the chain, the two policies.
// see docs/internals/tools/lab.md#l2-policy
#include "lab.cuh"

namespace l2_policy {

constexpr int kThreads = 512, kHalf = 8;
constexpr long long kUnitBytes = 16LL * kThreads, kStreamUnits = 1LL << 17;

__host__ __device__ constexpr uint32_t word_of(long long w) { return static_cast<uint32_t>(w) * 2654435761u + 1u; }

__global__ void __launch_bounds__(kThreads, 1) fill(uint4* table, long long count) {
  for (long long i = blockIdx.x * static_cast<long long>(kThreads) + threadIdx.x; i < count;
       i += static_cast<long long>(gridDim.x) * kThreads)
    table[i] = make_uint4(word_of(4 * i), word_of(4 * i + 1), word_of(4 * i + 2), word_of(4 * i + 3));
}

//: THE POLICIES: a fraction 1.0 of the lines a load touches, evicted first or last
__device__ __forceinline__ uint64_t evict_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint64_t evict_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}

//: a 16-byte load under a policy, or plain
template <bool HINT>
__device__ __forceinline__ uint4 load(const uint4* p, uint64_t policy) {
  uint4 v;
  if constexpr (HINT)
    asm volatile("ld.global.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p), "l"(policy));
  else
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

//: unit `n` of a CTA: the stream's own slice in order, the table's from the CTA's offset round its `units`
__host__ __device__ inline long long stream_unit(int block, long long per, long long n) { return block * per + n; }
__host__ __device__ inline long long table_unit(int block, long long units, long long n) {
  return (block * 97 + n) % units;
}

//: a kernel a table size (`MB`), so the profiler's read of each kernel is one table's
template <int HINT, int MB>
__global__ void __launch_bounds__(kThreads, 1)
    kept(long long* times, const uint4* stream, const uint4* table, uint32_t* sums, long long per, int trips) {
  constexpr long long units = MB * (1LL << 20) / kUnitBytes;
  const int tid = threadIdx.x, block = blockIdx.x;
  const uint64_t first = evict_first(), last = evict_last();
  uint32_t sum = 0;
  uint4 held[2][kHalf];
  //: the table's position walked by adds, `table_unit` without a division a load
  long long at = table_unit(block, units, 0);
  const auto issue = [&](long long trip) {
#pragma unroll
    for (int i = 0; i < kHalf; ++i)
      held[0][i] = load<HINT != 0>(stream + stream_unit(block, per, min(trip, trips - 1LL) * kHalf + i) * kThreads + tid, first);
#pragma unroll
    for (int i = 0; i < kHalf; ++i) {
      const long long u = at + i;
      held[1][i] = load<HINT != 0>(table + (u >= units ? u - units : u) * kThreads + tid, last);
    }
    at += kHalf;
    at = at >= units ? at - units : at;
  };
  issue(0);
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (long long t = 1; t <= trips; ++t) {
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int i = 0; i < kHalf; ++i) sum = sum * 3u + (held[h][i].x + held[h][i].y + held[h][i].z + held[h][i].w);
    issue(t);
  }
  clock.stop(times);
  sums[static_cast<long long>(block) * kThreads + tid] = sum;
}

}  // namespace l2_policy

template <int HINT, int MB>
void run(lab::Bench& b, const uint4* stream, const uint4* table, uint32_t* sums) {
  using namespace l2_policy;
  constexpr int table_mb = MB;
  const long long units = table_mb * (1LL << 20) / kUnitBytes, per = kStreamUnits / b.sms / kHalf * kHalf;
  const int trips = static_cast<int>(per / kHalf);
  const std::string name = std::to_string(table_mb) + " MB table beside a 1 GiB stream, "
                           + (HINT ? "the stream evict-first and the table evict-last" : "no hint");
  const auto kernel = kept<HINT, MB>;
  constexpr int kCheck = 4;
  kernel<<<b.sms, kThreads>>>(b.times, stream, table, sums, per, kCheck);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> got(static_cast<size_t>(b.sms) * kThreads);
  LAB_CHECK(cudaMemcpy(got.data(), sums, got.size() * 4, cudaMemcpyDeviceToHost));
  long long wrong = 0;
  const auto words = [](long long e) { return word_of(4 * e) + word_of(4 * e + 1) + word_of(4 * e + 2) + word_of(4 * e + 3); };
  for (int blk = 0; blk < b.sms; ++blk)
    for (int tid = 0; tid < kThreads; ++tid) {
      uint32_t want = 0;
      for (long long t = 0; t < kCheck; ++t) {
        for (int i = 0; i < kHalf; ++i) want = want * 3u + words(stream_unit(blk, per, t * kHalf + i) * kThreads + tid);
        //: the table's words: the same pattern, its own indices
        for (int i = 0; i < kHalf; ++i) want = want * 3u + words(table_unit(blk, units, t * kHalf + i) * kThreads + tid);
      }
      wrong += got[static_cast<size_t>(blk) * kThreads + tid] != want;
    }
  b.check(name, wrong == 0, std::to_string(wrong) + " of " + std::to_string(got.size()) + " chains differ");
  cudaEvent_t e0, e1;
  LAB_CHECK(cudaEventCreate(&e0));
  LAB_CHECK(cudaEventCreate(&e1));
  lab::Samples us;
  const lab::Samples cycles = b.variant(
      [&] {
        LAB_CHECK(cudaEventRecord(e0));
        kernel<<<b.sms, kThreads>>>(b.times, stream, table, sums, per, trips);
        LAB_CHECK(cudaEventRecord(e1));
      },
      [&] {
        float ms = 0;
        LAB_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        us.v.push_back(ms * 1000.0);
        return b.span(kThreads >> 5) / trips;
      });
  b.record(name + ": a trip", "cycles a trip a CTA (64 KB of the stream, 64 KB of the table)", cycles);
  b.record(name + ": the launch", "microseconds", us);
  LAB_CHECK(cudaEventDestroy(e0));
  LAB_CHECK(cudaEventDestroy(e1));
}

int main(int argc, char** argv) {
  using namespace l2_policy;
  lab::Bench b(argc, argv);
  for (int hint : {0, 1})
    for (int mb : {16, 40}) b.expect(lab::kernel("kept", {hint, mb}), "LDG", ">=", 2 * kHalf);
  //: the ring's last loaded register is read in its own trip, as `stream_rate`'s and `load_rate`'s
  for (int hint : {0, 1})
    for (int mb : {16, 40}) b.expect(lab::kernel("kept", {hint, mb}), "LOAD_IN_TRIP", "<=", 2);
  if (b.expect_only) return 0;
  //: the stream and the table are one pattern of words (a word its index's), the table the stream's first 40 MB
  uint4* stream = nullptr;
  uint32_t* sums = nullptr;
  LAB_CHECK(cudaMalloc(&stream, kStreamUnits * kUnitBytes));
  LAB_CHECK(cudaMalloc(&sums, static_cast<size_t>(b.sms) * kThreads * 4));
  fill<<<b.sms, kThreads>>>(stream, kStreamUnits * kUnitBytes / 16);
  LAB_CHECK(cudaDeviceSynchronize());
  uint4* table = nullptr;
  LAB_CHECK(cudaMalloc(&table, 40LL << 20));
  LAB_CHECK(cudaMemcpy(table, stream, 40LL << 20, cudaMemcpyDeviceToDevice));
  {
    run<0, 16>(b, stream, table, sums);
    run<1, 16>(b, stream, table, sums);
    run<0, 40>(b, stream, table, sums);
    run<1, 40>(b, stream, table, sums);
  }
  return 0;
}
