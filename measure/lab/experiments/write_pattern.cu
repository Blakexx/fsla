// measure/lab/experiments/write_pattern.cu -- QUESTION: at what rate does the H100 write the producer's outputs at its
// cell, and which part of their address pattern prices it? The producer's floor prices its 323.0 MB at HBM's write rate
// as `stream_rate` measured it (3,170 GB/s, each CTA one contiguous slice), and the library's stores alone at the cell
// (`produce_skeleton`) wrote them at ~2.0 TB/s. METHOD: one kernel, one store form, three address maps of the same
// bytes. The producer's grid (2,048 CTAs: a sequence's 128-token tile and a write head, tiles fastest) at its stores
// alone's occupancy (three CTAs an SM, held by reserved shared memory), two warpgroups a CTA, each its 64-token share,
// 16 bytes a store and 8 in flight a thread. A share's 78,848 bytes at the cell (`layer-alt-entmax1.5-256x256-T65536-h4`:
// B 1, T 65,536, H_w 4, G 1, d_v 64): four level-sides' 16 KB row tiles, two masked rows' 2 KB of words, four
// level-sides' 64 scales of 4 bytes, and 64 tokens' 128 bytes of values. ROWS: THE SLICE, a share's bytes one
// contiguous run (the floor's form at the producer's grid); WHOLE SECTORS, the producer's regions with each scale plane
// head-major (a share's 64 scales one 256-byte run, every 32-byte sector whole); THE PRODUCER'S OWN, its regions as
// `docs/api.md#tile-major` lays them out (the scales token-major over the heads: a share's scale a 4-byte word every 16
// bytes, each 32-byte sector shared by the four heads' CTAs). THE CHECK: every byte of every output and a guard past
// each filled with a sentinel, then the launch: every output byte zero, every guard byte the sentinel.
// TIMELINE: a warpgroup's runs back to back (a level-side's row tile, its scales, its mask where masked; then the
// values), no barrier between; a CTA's end its last store. Live: a share's addresses.
// see docs/internals/tools/lab.md#write-pattern
#include <cstdint>
#include <string>
#include <vector>

#include "lab.cuh"

namespace write_pattern {

//: FIXED by the producer's CTA: two warpgroups, each a 64-token share of the CTA's 128-token tile
constexpr int kGroup = 128, kThreads = 2 * kGroup, kShareTokens = 64;
//: FIXED by the producer's store component: the 16-byte stores a thread keeps in flight
constexpr int kInFlight = 8;
//: FIXED by the cell: T, H_w, the level-sides, the masked rows, a row tile's bytes, a mask share's, a token's values
constexpr int kTokens = 65536, kHeads = 4, kSides = 4, kMasks = 2, kTile = 16384, kMask = 2048, kValue = 128;
constexpr int kShares = kTokens / kShareTokens, kTiles = kShares / 2, kItems = kTiles * kHeads;
//: a share's bytes: its four row tiles, its two masks, its four level-sides' scales, its values
constexpr int kShareBytes = kSides * kTile + kMasks * kMask + kSides * kShareTokens * 4 + kShareTokens * kValue;
//: FIXED by the stores alone's occupancy: the shared memory a CTA reserves, so three CTAs hold an SM and four do not
constexpr int kReserved = 72 * 1024;
enum Way : int { kSlice = 0, kWhole, kOwn, kWays };

//: THE OUTPUTS: the slice's one table; the producer's regions (each level-side's rows and scales, each masked row's
//: words, the values)
struct Planes {
  uint8_t* slice;
  uint8_t* rows[kSides];
  uint8_t* masks[kMasks];
  uint8_t* scales[kSides];
  uint8_t* values;
};

//: A CONTIGUOUS RUN'S ZEROS by a warpgroup (`m` its thread): `BYTES` a multiple of 16, 16 bytes a store, `kInFlight` a
//: trip
template <int BYTES>
__device__ __forceinline__ void zero_run(uint8_t* at, int m) {
  constexpr int kChunks = BYTES / 16;
  uint4* const body = reinterpret_cast<uint4*>(at);
#pragma unroll 1
  for (int c = m; c < kChunks; c += kGroup * kInFlight) {
#pragma unroll
    for (int u = 0; u < kInFlight; ++u)
      if (c + u * kGroup < kChunks) body[c + u * kGroup] = make_uint4(0u, 0u, 0u, 0u);
  }
}

//: A SHARE'S VALUES by a warpgroup: 64 tokens' 128 bytes, `kHeads * kValue` apart, 8 threads a token, every token in
//: flight at once
__device__ __forceinline__ void zero_values(uint8_t* at, int m) {
  constexpr int kLanes = kValue / 16, kPer = kGroup / kLanes;
  const int lane = m % kLanes, token = m / kLanes;
#pragma unroll
  for (int u = 0; u < kShareTokens / kPer; ++u)
    reinterpret_cast<uint4*>(at + static_cast<long long>(token + u * kPer) * kHeads * kValue)[lane] =
        make_uint4(0u, 0u, 0u, 0u);
}

//: THE KERNEL: CTA `blockIdx.x`'s item (tile fastest, then write head), each warpgroup its share, in `WAY`'s map
template <int WAY>
__global__ void __launch_bounds__(kThreads) stores(const __grid_constant__ Planes p) {
  const int tid = static_cast<int>(threadIdx.x), m = tid % kGroup;
  const int tile = static_cast<int>(blockIdx.x) % kTiles, w = static_cast<int>(blockIdx.x) / kTiles;
  const int share = 2 * tile + tid / kGroup, t0 = share * kShareTokens;
  if constexpr (WAY == kSlice) {
    zero_run<kShareBytes>(p.slice + (static_cast<long long>(w) * kShares + share) * kShareBytes, m);
  } else {
    const long long stream = static_cast<long long>(w) * kShares + share;
#pragma unroll
    for (int q = 0; q < kSides; ++q) {
      zero_run<kTile>(p.rows[q] + stream * kTile, m);
      if constexpr (WAY == kWhole) {
        zero_run<kShareTokens * 4>(p.scales[q] + (static_cast<long long>(w) * kTokens + t0) * 4, m);
      } else {
        if (m < kShareTokens)
          reinterpret_cast<uint32_t*>(p.scales[q] + (static_cast<long long>(t0 + m) * kHeads + w) * 4)[0] = 0u;
      }
      //: the cell's masked rows: the write side's first level and the read side's second (level-sides 0 and 3)
      if (q == 0 || q == kSides - 1) zero_run<kMask>(p.masks[q == 0 ? 0 : 1] + stream * kMask, m);
    }
    zero_values(p.values + (static_cast<long long>(t0) * kHeads + w) * kValue, m);
  }
}

constexpr uint8_t kSentinel = 0xA5;
constexpr size_t kGuard = 256;

//: ONE BUFFER: its device bytes (and a guard past them)
struct Buffer {
  uint8_t* at = nullptr;
  size_t bytes = 0;
};

//: THE BUFFERS of the rows: the slice's table and the producer's regions, each with its guard
struct Buffers {
  std::vector<Buffer> all;
  Planes planes{};

  uint8_t* make(size_t bytes) {
    Buffer b;
    b.bytes = bytes;
    LAB_CHECK(cudaMalloc(&b.at, bytes + kGuard));
    all.push_back(b);
    return b.at;
  }

  Buffers() {
    planes.slice = make(static_cast<size_t>(kItems) * 2 * kShareBytes);
    for (int q = 0; q < kSides; ++q) planes.rows[q] = make(static_cast<size_t>(kHeads) * kShares * kTile);
    for (int r = 0; r < kMasks; ++r) planes.masks[r] = make(static_cast<size_t>(kHeads) * kShares * kMask);
    for (int q = 0; q < kSides; ++q) planes.scales[q] = make(static_cast<size_t>(kHeads) * kTokens * 4);
    planes.values = make(static_cast<size_t>(kHeads) * kTokens * kValue);
  }

  //: the buffers a way writes: the slice's table, or every region
  std::vector<Buffer> of(int way) const {
    return way == kSlice ? std::vector<Buffer>{all.front()} : std::vector<Buffer>(all.begin() + 1, all.end());
  }

  ~Buffers() {
    for (const Buffer& b : all) cudaFree(b.at);
  }
};

}  // namespace write_pattern

//: A ROW: its way's check (every byte it owns zero, every guard the sentinel) and its launch timed
template <int WAY>
void run(lab::Bench& b, write_pattern::Buffers& o, const std::string& name) {
  using namespace write_pattern;
  LAB_CHECK(cudaFuncSetAttribute(stores<WAY>, cudaFuncAttributeMaxDynamicSharedMemorySize, kReserved));
  const auto go = [&] { stores<WAY><<<kItems, kThreads, kReserved>>>(o.planes); };
  long long wrong = 0, bytes = 0;
  for (const Buffer& buf : o.of(WAY)) LAB_CHECK(cudaMemset(buf.at, kSentinel, buf.bytes + kGuard));
  go();
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<uint8_t> h;
  for (const Buffer& buf : o.of(WAY)) {
    h.resize(buf.bytes + kGuard);
    LAB_CHECK(cudaMemcpy(h.data(), buf.at, h.size(), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < h.size(); ++i) wrong += h[i] != (i < buf.bytes ? 0 : kSentinel);
    bytes += static_cast<long long>(buf.bytes);
  }
  b.check(name, wrong == 0 && bytes == static_cast<long long>(kItems) * 2 * kShareBytes,
          std::to_string(wrong) + " wrong bytes over " + std::to_string(bytes) + " written");
  cudaEvent_t e0, e1;
  LAB_CHECK(cudaEventCreate(&e0));
  LAB_CHECK(cudaEventCreate(&e1));
  b.record(name, "microseconds a launch", b.variant(go, [&] {
             LAB_CHECK(cudaEventRecord(e0));
             go();
             LAB_CHECK(cudaEventRecord(e1));
             LAB_CHECK(cudaEventSynchronize(e1));
             float ms = 0;
             LAB_CHECK(cudaEventElapsedTime(&ms, e0, e1));
             return 1000.0 * ms;
           }));
  LAB_CHECK(cudaEventDestroy(e0));
  LAB_CHECK(cudaEventDestroy(e1));
}

int main(int argc, char** argv) {
  using namespace write_pattern;
  lab::Bench b(argc, argv);
  for (int w = 0; w < kWays; ++w) b.expect(lab::kernel("stores", {w}), "STG", ">=", kInFlight);
  if (b.expect_only) return 0;
  Buffers o;
  const std::string cell = "the producer's 323.0 MB at the cell, three CTAs an SM";
  run<kSlice>(b, o, cell + ": a share's bytes one contiguous run");
  run<kWhole>(b, o, cell + ": its regions, every sector whole");
  run<kOwn>(b, o, cell + ": its regions as the producer lays them out");
  return 0;
}
