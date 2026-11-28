// measure/lab/experiments/stream_rate.cu -- QUESTION: what does the H100 stream, a CTA an SM, from HBM and from L2: the
// read rate, the write rate and a copy's, in bytes a cycle an SM and in GB/s the device? The producer's floors (its
// outputs to HBM, its weights from L2) are priced on these and no lab row recorded either. METHOD: a CTA an SM of 512
// threads; a lane moves 16 bytes an instruction, so an instruction is a UNIT of 8 KB a CTA. READ: a ring of 16 units in
// flight a thread in two halves at their own registers, each word folded into a chain a lane (`3 s + w`), nothing
// stored. WRITE: 16 stores a trip of a word formed from its index. COPY: the read's ring, each unit stored where it was
// read (another table). The source: HBM, each CTA its own slice of a 1 GiB table once (the copy 512 MB into 512 MB); L2
// DISTINCT, each CTA its own 128 KB slice of a 16.5 MB table 64 times; L2 SHARED, every CTA the whole of one 4 MB table
// 4 times from its own offset; the two L2 walks again PAST L1 (`ld.global.cg`, L2 only). Every chain, every written
// word and every copied word is checked exactly on a short launch.
// TIMELINE: a thread's trip is the ring's two halves, each at its own registers: a half's units folded (or stored),
// then that half's loads for the trip after. Live: the ring's 64 words and the chain.
// see docs/internals/tools/lab.md#stream-rate
#include "lab.cuh"

namespace stream_rate {

constexpr int kThreads = 512, kHalf = 8, kTrip = 2 * kHalf;
//: a unit: one instruction of the CTA, 16 bytes a lane
constexpr long long kUnitBytes = 16LL * kThreads;
enum Mode : int { kRead = 0, kWrite, kCopy, kReadL2, kModes };

//: the table's word at 32-bit index `w`
__host__ __device__ constexpr uint32_t word_of(long long w) {
  return static_cast<uint32_t>(w) * 2654435761u + 1u;
}

//: A CTA'S WALK: its unit `n` is the table's unit `(start + (n & span) * stride) & (wrap - 1)`, `start` the CTA's own
//: (`stride` 1 a contiguous slice; wider, the slice's units interleaved with the other CTAs')
struct Walk {
  long long start_stride, span, wrap, stride = 1;
  __host__ __device__ long long unit(int block, long long n) const {
    return (block * start_stride + (n & span) * stride) & (wrap - 1);
  }
};

__global__ void __launch_bounds__(kThreads, 1) fill(uint4* table, long long count) {
  for (long long i = blockIdx.x * static_cast<long long>(kThreads) + threadIdx.x; i < count;
       i += static_cast<long long>(gridDim.x) * kThreads)
    table[i] = make_uint4(word_of(4 * i), word_of(4 * i + 1), word_of(4 * i + 2), word_of(4 * i + 3));
}

template <int MODE>
__global__ void __launch_bounds__(kThreads, 1)
    stream(long long* times, const uint4* src, uint4* dst, uint32_t* sums, Walk walk, int units) {
  const int tid = threadIdx.x;
  const int block = blockIdx.x;
  const long long last = units - 1;
  const auto at = [&](long long n) { return walk.unit(block, min(n, last)) * kThreads + tid; };
  const int trips = units / kTrip;
  uint32_t sum = 0;
  lab::WarpSpan clock;
  if constexpr (MODE == kWrite) {
    clock.start();
#pragma unroll 1
    for (int t = 0; t < trips; ++t) {
#pragma unroll
      for (int i = 0; i < kTrip; ++i) {
        const long long e = at(static_cast<long long>(t) * kTrip + i);
        dst[e] = make_uint4(word_of(4 * e), word_of(4 * e + 1), word_of(4 * e + 2), word_of(4 * e + 3));
      }
    }
  } else {
    uint4 held[2][kHalf];
    const auto load = [&](int h, long long trip) {
#pragma unroll
      for (int i = 0; i < kHalf; ++i)
        held[h][i] = MODE == kReadL2 ? __ldcg(src + at(trip * kTrip + h * kHalf + i)) : src[at(trip * kTrip + h * kHalf + i)];
    };
    load(0, 0);
    load(1, 0);
    clock.start();
#pragma unroll 1
    for (int t = 1; t <= trips; ++t) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        if constexpr (MODE == kRead || MODE == kReadL2) {
#pragma unroll
          for (int i = 0; i < kHalf; ++i) sum = sum * 3u + (held[h][i].x + held[h][i].y + held[h][i].z + held[h][i].w);
        } else {
#pragma unroll
          for (int i = 0; i < kHalf; ++i) dst[at(static_cast<long long>(t - 1) * kTrip + h * kHalf + i)] = held[h][i];
        }
        load(h, t);
      }
    }
  }
  clock.stop(times);
  sums[static_cast<long long>(block) * kThreads + tid] = sum;
}

//: A ROW: its mode, its walk, the units a CTA moves timed and checked, and the tables' bytes it touches
struct Row {
  const char* name;
  int mode;
  Walk walk;
  int units, check_units;
  //: the CTAs it launches, one an SM (0: every SM)
  int ctas = 0;
};

}  // namespace stream_rate

template <int MODE>
void run(lab::Bench& b, const stream_rate::Row& row, const uint4* src, uint4* dst, uint32_t* sums) {
  using namespace stream_rate;
  const auto kernel = stream<MODE>;
  const int grid = row.ctas > 0 ? row.ctas : b.sms;
  //: THE CHECK: a short launch, every chain, written word or copied word against the host
  LAB_CHECK(cudaMemset(dst, 0, static_cast<size_t>(row.walk.wrap) * kUnitBytes));
  kernel<<<grid, kThreads>>>(b.times, src, dst, sums, row.walk, row.check_units);
  LAB_CHECK(cudaDeviceSynchronize());
  long long wrong = 0, checked = 0;
  if (MODE == kRead || MODE == kReadL2) {
    std::vector<uint32_t> got(static_cast<size_t>(grid) * kThreads);
    LAB_CHECK(cudaMemcpy(got.data(), sums, got.size() * 4, cudaMemcpyDeviceToHost));
    for (int blk = 0; blk < grid; ++blk)
      for (int tid = 0; tid < kThreads; ++tid) {
        uint32_t want = 0;
        for (long long n = 0; n < row.check_units; ++n) {
          const long long e = row.walk.unit(blk, n) * kThreads + tid;
          want = want * 3u + (word_of(4 * e) + word_of(4 * e + 1) + word_of(4 * e + 2) + word_of(4 * e + 3));
        }
        wrong += got[static_cast<size_t>(blk) * kThreads + tid] != want;
        ++checked;
      }
  } else {
    //: every unit a CTA wrote: the word of its index (a write) or the source's word, which is the same (a copy)
    for (int blk = 0; blk < grid; ++blk)
      for (long long n = 0; n < row.check_units; ++n) {
        std::vector<uint32_t> got(kThreads * 4);
        LAB_CHECK(cudaMemcpy(got.data(), dst + row.walk.unit(blk, n) * kThreads, kUnitBytes, cudaMemcpyDeviceToHost));
        for (int w = 0; w < kThreads * 4; ++w) {
          wrong += got[w] != word_of(row.walk.unit(blk, n) * kThreads * 4 + w);
          ++checked;
        }
      }
  }
  b.check(row.name, wrong == 0, std::to_string(wrong) + " of " + std::to_string(checked) + " words differ");

  cudaEvent_t e0, e1;
  LAB_CHECK(cudaEventCreate(&e0));
  LAB_CHECK(cudaEventCreate(&e1));
  const double moved = static_cast<double>(row.units) * kUnitBytes * (MODE == kCopy ? 2 : 1);
  lab::Samples per_sm;
  const lab::Samples device = b.variant(
      [&] {
        LAB_CHECK(cudaEventRecord(e0));
        kernel<<<grid, kThreads>>>(b.times, src, dst, sums, row.walk, row.units);
        LAB_CHECK(cudaEventRecord(e1));
      },
      [&] {
        float ms = 0;
        LAB_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        per_sm.v.push_back(moved / b.span(kThreads >> 5, grid));
        return moved * grid / (ms * 1.0e6);
      });
  b.record(std::string(row.name) + ": the device", "GB/s", device);
  b.record(std::string(row.name) + ": a CTA", "bytes a cycle an SM", per_sm);
  LAB_CHECK(cudaEventDestroy(e0));
  LAB_CHECK(cudaEventDestroy(e1));
}

int main(int argc, char** argv) {
  using namespace stream_rate;
  lab::Bench b(argc, argv);
  for (int m = 0; m < kModes; ++m) {
    const std::string k = lab::kernel("stream", {m});
    if (m != kWrite) b.expect(k, "LDG", ">=", kTrip);
    //: the ring: the toolkit copies a loaded register at the trip's end (as `load_rate`'s ring), so one of a trip's 16
    //: loads is read in the trip that issued it: declared, the form being the stream's
    if (m != kWrite) b.expect(k, "LOAD_IN_TRIP", "<=", 2);
    if (m == kWrite || m == kCopy) b.expect(k, "STG", ">=", kTrip);
  }
  if (b.expect_only) return 0;
  //: the tables: 1 GiB each, a unit 8 KB, so 131,072 units
  constexpr long long kTableUnits = 1LL << 17, kTableBytes = kTableUnits * kUnitBytes;
  uint4 *src = nullptr, *dst = nullptr;
  uint32_t* sums = nullptr;
  LAB_CHECK(cudaMalloc(&src, kTableBytes));
  LAB_CHECK(cudaMalloc(&dst, kTableBytes));
  LAB_CHECK(cudaMalloc(&sums, static_cast<size_t>(b.sms) * kThreads * 4));
  fill<<<b.sms, kThreads>>>(src, kTableBytes / 16);
  LAB_CHECK(cudaDeviceSynchronize());
  //: HBM: each CTA its own slice of the table once, 16 units a trip (992 = 62 trips: 7.75 MB a CTA, 1 GiB at 132 CTAs)
  const int hbm = static_cast<int>(kTableUnits / b.sms) / kTrip * kTrip;
  const Row rows[] = {
      {"HBM read, a CTA its own slice of a 1 GiB table", kRead, {hbm, ~0LL, kTableUnits}, hbm, 32},
      {"HBM write, a CTA its own slice of a 1 GiB table", kWrite, {hbm, ~0LL, kTableUnits}, hbm, 32},
      {"HBM copy, a CTA its own slice of 512 MB into 512 MB", kCopy, {hbm / 2 / kTrip * kTrip, ~0LL, kTableUnits / 2},
       hbm / 2 / kTrip * kTrip, 32},
      //: L2: a CTA's own 16 units (128 KB; 16.5 MB at 132 CTAs) 64 times over
      {"L2 read, a CTA its own 128 KB slice 64 times", kRead, {16, 15, 4096}, 16 * 64, 32},
      //: L2: every CTA the whole of one 512-unit (4 MB) table 4 times, each from its own offset
      {"L2 read, every CTA one 4 MB table 4 times", kRead, {4, ~0LL, 512}, 512 * 4, 32},
      //: THE SAME TWO WALKS PAST L1 (`ld.global.cg`: cached in L2 only, as the copies the producer stages by --
      //: `cp.async.cg` and the tensor copy -- read it), so a CTA's own 128 KB slice cannot hit its SM's L1
      {"L2 read past L1 (ld.global.cg), a CTA its own 128 KB slice 64 times", kReadL2, {16, 15, 4096}, 16 * 64, 32},
      {"L2 read past L1 (ld.global.cg), every CTA one 4 MB table 4 times", kReadL2, {4, ~0LL, 512}, 512 * 4, 32},
      //: WHY A CTA'S OWN SLICE PAST L1 READS SLOWER THAN A SHARED TABLE: the own slice on half the SMs (an aggregate
      //: cap on distinct lines would lift it; a path an SM would not); its 16 units interleaved over the table (an
      //: address placement would move it); and every CTA one shared 128 KB slice (sharing alone would lift it)
      {"L2 read past L1 (ld.global.cg), a CTA its own 128 KB slice 64 times, 66 CTAs", kReadL2, {16, 15, 4096}, 16 * 64,
       32, 66},
      {"L2 read past L1 (ld.global.cg), a CTA its own 128 KB interleaved over the table 64 times", kReadL2,
       {1, 15, 4096, 256}, 16 * 64, 32},
      {"L2 read past L1 (ld.global.cg), every CTA one 128 KB slice 64 times", kReadL2, {0, 15, 4096}, 16 * 64, 32},
  };
  run<kRead>(b, rows[0], src, dst, sums);
  run<kWrite>(b, rows[1], src, dst, sums);
  run<kCopy>(b, rows[2], src, dst, sums);
  run<kRead>(b, rows[3], src, dst, sums);
  run<kRead>(b, rows[4], src, dst, sums);
  run<kReadL2>(b, rows[5], src, dst, sums);
  run<kReadL2>(b, rows[6], src, dst, sums);
  run<kReadL2>(b, rows[7], src, dst, sums);
  run<kReadL2>(b, rows[8], src, dst, sums);
  run<kReadL2>(b, rows[9], src, dst, sums);
  return 0;
}
