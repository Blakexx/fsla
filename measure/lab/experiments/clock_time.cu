// measure/lab/experiments/clock_time.cu -- QUESTION: is the SM's cycle counter a faithful clock -- does it advance at
// one rate against the GPU's nanosecond timer whatever a warp is waiting on -- or does it run slow where an SM's warps
// wait on memory? METHOD: one warpgroup a CTA, a CTA an SM; a body runs many steps between two readings of BOTH clocks
// (`%clock64` and `%globaltimer`) by each warp's first lane: independent integer operations (issue), a chain of shared
// loads (latency in the SM), a chain of scattered loads of a 64 MB table (waits on global memory), a store a step to
// global memory, and one m64n64k16 wgmma a step with its commit and wait (the tensor pipe). A body's rate is its
// cycles over its microseconds, a warp's mean; the rows are read against each other and, at one CTA and at the full
// grid, against the grid.
// TIMELINE: one warpgroup, no other role; the two clocks read once before the steps and once after. A step is the
// body's instructions alone: no load is a stage ahead of its use (the chains are their own latency, by design -- the
// warps are meant to wait). Live: a body's value or eight, the wgmma body's accumulator (32 registers a thread), A
// (4) and its descriptor.
// see docs/internals/tools/lab.md#clock-time
#include "lab.cuh"

namespace clock_time {

constexpr int kM = 64, kK = 16, kN = 64, kThreads = 128, kWarps = kThreads / 32;
constexpr int kTileA = 0, kTileB = 2048, kTable = 4096, kTableWords = 256, kSmem = 8192;
//: the global table: 16M words, 64 MB -- past the L2, so a scattered load of it waits on global memory
constexpr int kGlobalWords = 1 << 24;
//: a thread's row of global stores
constexpr int kRowWords = 1024;
constexpr int kWide = 8;
enum Body { kIssue = 0, kSharedChain = 1, kGlobalChain = 2, kGlobalStore = 3, kTensor = 4, kBodies = 5 };

constexpr int read_a(int m, int k) { return (m + k) % 7 - 3; }
constexpr int read_b(int k, int n) { return (k + 2 * n) % 3 - 1; }
constexpr int tile_at(int n, int k) { return 16 * ((n % 8) + 16 * (n / 8) + 8 * (k / 8)) + 2 * (k % 8); }
constexpr uint32_t next(uint32_t x, uint32_t loaded) { return x * 1664525u + 1013904223u + loaded; }

__device__ __forceinline__ uint16_t bf16_of(int v) {
  return static_cast<uint16_t>(__float_as_uint(static_cast<float>(v)) >> 16);
}
//: the two clocks, read together
__device__ __forceinline__ void clocks(unsigned long long& cycles, unsigned long long& ns) {
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(cycles));
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
}

template <int BODY>
__global__ void __launch_bounds__(kThreads, 1) rate(unsigned long long* marks, uint32_t* out, const uint32_t* table,
                                                    uint32_t* rows, int steps) {
  __shared__ alignas(128) uint8_t smem[kSmem];
  const int tid = threadIdx.x, lane = tid & 31;
  auto* tile = reinterpret_cast<uint16_t*>(smem);
  auto* shared = reinterpret_cast<uint32_t*>(smem + kTable);
#pragma unroll 1
  for (int i = tid; i < kM * kK; i += blockDim.x) tile[(kTileA + tile_at(i / kK, i % kK)) / 2] = bf16_of(read_a(i / kK, i % kK));
#pragma unroll 1
  for (int i = tid; i < kN * kK; i += blockDim.x) tile[(kTileB + tile_at(i / kK, i % kK)) / 2] = bf16_of(read_b(i % kK, i / kK));
#pragma unroll 1
  for (int i = tid; i < kTableWords; i += blockDim.x) shared[i] = static_cast<uint32_t>(i) * 2654435761u;
  //: the tiles were written by ordinary stores and are read by the wgmma through the async proxy: fence them first
  lab::bulk_fence();
  __syncthreads();
  const uint32_t base = lab::smem(smem);
  const uint64_t db = lab::held(lab::wg_desc(base + kTileB, 8, 16));
  uint32_t a[4];
  const int g = 16 * ((tid >> 5) & 3) + lane / 4, kt = lane & 3;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    const int m = g + 8 * (w & 1), k = 8 * (w >> 1) + 2 * kt;
    a[w] = lab::held(bf16_of(read_a(m, k)) | static_cast<uint32_t>(bf16_of(read_a(m, k + 1))) << 16);
  }
  float d[32] = {};
  uint32_t x = lab::held(static_cast<uint32_t>(blockIdx.x * blockDim.x + tid) * 2654435761u + 1u);
  uint32_t y[kWide];
#pragma unroll
  for (int j = 0; j < kWide; ++j) y[j] = lab::held(static_cast<uint32_t>(tid) + 1u + 257u * j);
  uint32_t* const row = rows + static_cast<long long>(blockIdx.x * blockDim.x + tid) * kRowWords;

  unsigned long long c0, n0, c1, n1;
  clocks(c0, n0);
#pragma unroll 1
  for (int it = 0; it < steps; ++it) {
    if constexpr (BODY == kIssue) {
#pragma unroll
      for (int j = 0; j < kWide; ++j) y[j] = next(y[j], static_cast<uint32_t>(j));
    }
    if constexpr (BODY == kSharedChain) {
#pragma unroll
      for (int k = 0; k < 4; ++k) x = next(x, shared[(x >> 10) & (kTableWords - 1)]);
    }
    if constexpr (BODY == kGlobalChain) {
#pragma unroll
      for (int k = 0; k < 4; ++k) x = next(x, table[(x >> 6) & (kGlobalWords - 1)]);
    }
    if constexpr (BODY == kGlobalStore) {
      x = next(x, 1u);
      row[it & (kRowWords - 1)] = x;
    }
    if constexpr (BODY == kTensor) {
      lab::wg_fence();
      lab::wg_bf16_n64_rs(d, a, db);
      lab::wg_commit();
      lab::wg_wait<0>();
    }
  }
#pragma unroll
  for (int i = 0; i < 32; ++i) lab::wg_hold(d[i]);
  clocks(c1, n1);
  if (lane == 0) {
    unsigned long long* const mine = marks + 4 * (static_cast<long long>(blockIdx.x) * kWarps + (tid >> 5));
    mine[0] = c0;
    mine[1] = c1;
    mine[2] = n0;
    mine[3] = n1;
  }
#pragma unroll
  for (int j = 0; j < kWide; ++j) x ^= y[j];
#pragma unroll
  for (int i = 0; i < 32; ++i) x ^= __float_as_uint(d[i]);
  lab::sink(reinterpret_cast<int*>(out), x);
}

}  // namespace clock_time

template <int BODY>
void run(lab::Bench& b, unsigned long long* marks, uint32_t* out, const uint32_t* table, uint32_t* rows) {
  using namespace clock_time;
  static const char* const kNames[] = {"eight independent integer operations a step", "a chain of four shared loads a step",
                                       "a chain of four scattered loads of a 64 MB table a step",
                                       "a store to global memory a step", "one wgmma a step, its commit and wait"};
  //: steps for a launch of milliseconds: the nanosecond timer's own grain is then far under the reading
  static const int kSteps[] = {200000, 40000, 6000, 200000, 60000};
  const std::string name = kNames[BODY];
  std::vector<unsigned long long> got(static_cast<size_t>(b.sms) * kWarps * 4);
  //: a launch's two clocks, each warp's: its cycles, its nanoseconds
  const auto read = [&](double& cycles, double& ns, long long& bad) {
    LAB_CHECK(cudaDeviceSynchronize());
    LAB_CHECK(cudaMemcpy(got.data(), marks, got.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    cycles = ns = 0.0;
    bad = 0;
    for (size_t w = 0; w < got.size() / 4; ++w) {
      const double c = static_cast<double>(got[4 * w + 1] - got[4 * w]), n = static_cast<double>(got[4 * w + 3] - got[4 * w + 2]);
      bad += !(c > 0.0 && n > 0.0);
      cycles += c;
      ns += n;
    }
  };
  rate<BODY><<<b.sms, kThreads>>>(marks, out, table, rows, kSteps[BODY]);
  double cycles = 0.0, ns = 0.0;
  long long bad = 0;
  read(cycles, ns, bad);
  const double per_us = ns > 0.0 ? 1000.0 * cycles / ns : 0.0;
  b.check(name, bad == 0 && per_us > 500.0 && per_us < 4000.0,
          std::to_string(bad) + " warps read a clock that did not advance; the launch's rate " + std::to_string(per_us)
              + " cycles a microsecond (a GPU clock is 500 to 4,000)");
  const auto launch = [&] { rate<BODY><<<b.sms, kThreads>>>(marks, out, table, rows, kSteps[BODY]); };
  b.record(name + ": the cycle counter's rate", "cycles a microsecond", b.variant(launch, [&] {
    read(cycles, ns, bad);
    return 1000.0 * cycles / ns;
  }));
  b.record(name + ": a step by the cycle counter", "cycles a step", b.variant(launch, [&] {
    read(cycles, ns, bad);
    return cycles / (static_cast<double>(b.sms) * kWarps * kSteps[BODY]);
  }));
  b.record(name + ": a step by the nanosecond timer", "nanoseconds a step", b.variant(launch, [&] {
    read(cycles, ns, bad);
    return ns / (static_cast<double>(b.sms) * kWarps * kSteps[BODY]);
  }));
}

int main(int argc, char** argv) {
  using namespace clock_time;
  lab::Bench b(argc, argv);
  for (int body = 0; body < kBodies; ++body) {
    const std::string k = lab::kernel("rate", {body});
    b.expect(k, "HGMMA", "==", body == kTensor ? 1 : 0);
    //: the global chain's loads are read in the trip that issued them by design: the warps are meant to wait on them
    if (body == kGlobalChain) b.expect(k, "LOAD_IN_TRIP", "==", 4);
    //: the store body's global store is its question, in a step that holds no MMA; the tensor body holds none
    b.exempt(k, "a probe of the clock: no kernel's step");
  }
  if (b.expect_only) return 0;
  unsigned long long* marks = nullptr;
  uint32_t *out = nullptr, *table = nullptr, *rows = nullptr;
  LAB_CHECK(cudaMalloc(&marks, sizeof(unsigned long long) * b.sms * kWarps * 4));
  LAB_CHECK(cudaMalloc(&out, sizeof(uint32_t) * b.sms * kThreads));
  LAB_CHECK(cudaMalloc(&table, sizeof(uint32_t) * kGlobalWords));
  LAB_CHECK(cudaMemset(table, 0, sizeof(uint32_t) * kGlobalWords));
  LAB_CHECK(cudaMalloc(&rows, sizeof(uint32_t) * b.sms * kThreads * kRowWords));
  run<kIssue>(b, marks, out, table, rows);
  run<kSharedChain>(b, marks, out, table, rows);
  run<kGlobalChain>(b, marks, out, table, rows);
  run<kGlobalStore>(b, marks, out, table, rows);
  run<kTensor>(b, marks, out, table, rows);
  return 0;
}
