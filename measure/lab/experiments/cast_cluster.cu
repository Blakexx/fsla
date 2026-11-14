// measure/lab/experiments/cast_cluster.cu -- QUESTION: `value_cast` found a multicast copy 3.6 times a CTA's own, in
// clusters of 8 with a cluster-wide barrier every 8 KB. Does a multicast pay in a SMALL cluster, a slot-sized tile
// and a hand-off that stops no thread but the issuer's -- and on tiles the cache has not seen, where a cluster's
// CTAs each asking for the same tile ask global memory for it? METHOD: clusters of 2, 4 and 8 CTAs, 4 warps each;
// a chunk is a tile of 160 rows of 128 bytes (20 KB), TWO OR FOUR in flight, a slot each in turn. `each`:
// every CTA issues its own tensor copy of the cluster's tile. `cast`: a CTA's readers done, one of its threads
// states the bytes at its own barrier and ARRIVES AT A BARRIER OF RANK 0's, one a slot, a count of the cluster's
// CTAs; rank 0's issuing thread waits on that barrier and issues one multicast copy to all. No cluster barrier, and
// no thread of a mate waits on another CTA. Each is timed on a few tiles the cache holds and on a fresh tile a
// chunk a cluster. The last tiles are checked exactly in every CTA.
// TIMELINE: a CTA's trip is a chunk, its slot the chunk's turn of the `D` in flight: every thread waits on the
// tile `D` chunks back in that slot and reads a word of it; the CTA's barrier; one thread states the chunk's bytes
// at the slot's barrier and, in `cast`, arrives at rank 0's for the slot; the copy issued by that thread (`each`) or by rank 0's once the cluster's arrivals are in (`cast`). Live:
// nothing held.
// see docs/internals/tools/lab.md#cast-cluster
#include <cuda.h>
#include <dlfcn.h>

#include "lab.cuh"

namespace cast_cluster {

constexpr int kThreads = 128, kTokens = 160, kRowBytes = 128, kTileBytes = kTokens * kRowBytes;
//: `D` slots in flight, their landed barriers and, in rank 0, the barriers its mates arrive at
template <int D>
constexpr int kBarsAt = D * kTileBytes;
template <int D>
constexpr int kFreeAt = kBarsAt<D> + 8 * D;
template <int D>
constexpr int kSmemBytes = kFreeAt<D> + 8 * D;
constexpr int kWarmTiles = 64, kChunks = 256;
enum Form : int { kEach = 0, kCast = 1 };

constexpr uint8_t byte_of(uint32_t at) { return static_cast<uint8_t>(at * 2654435761u >> 24); }
//: chunk `c`'s tile for cluster `g`: one of a few the cache holds, or a fresh one a chunk
constexpr int tile_of(int cold, int g, int c) { return cold ? g * kChunks + c % kChunks : (c * 7) % kWarmTiles; }

template <int C, int FORM, int COLD, int D>
__global__ void __cluster_dims__(C, 1, 1) __launch_bounds__(kThreads, 1)
    cast(const __grid_constant__ CUtensorMap map, long long* times, int* sinks, uint8_t* out, int chunks) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, cluster = blockIdx.x / C;
  const uint32_t rank = lab::cluster_rank();
  const uint32_t base = lab::smem(smem), landed = base + kBarsAt<D>;
  //: rank 0's barriers, as this CTA reaches them
  const uint32_t freed = lab::cluster_addr(base + kFreeAt<D>, 0);
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < D; ++i) {
      lab::mbar_init(landed + 8 * i, 1);
      if (rank == 0) lab::mbar_init(base + kFreeAt<D> + 8 * i, C);
    }
    lab::cluster_fence_init();
  }
  __syncthreads();
  //: every CTA of the cluster has its barriers before a mate arrives at one or a copy completes one
  lab::cluster_sync();

  uint32_t seen = 0;
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int c = 0; c < chunks; ++c) {
    const uint32_t slot = c % D, bar = landed + 8 * slot;
    //: THE LANDED EDGE: the tile `D` chunks back is in the slot; a word of it a thread
    if (c >= D) {
      lab::mbar_wait_suspended(bar, (c / D - 1) & 1);
      seen ^= smem[slot * kTileBytes + tid * 157] + c;
    }
    //: THE ISSUE EDGE: the CTA's readers of the slot and its barrier's stated bytes before the copy that overwrites
    //: it -- its own copy at once, or the cluster's once every CTA has arrived at rank 0's barrier
    __syncthreads();
    if (tid == 0) {
      lab::mbar_expect(bar, kTileBytes);
      const int row = tile_of(COLD, cluster, c) * kTokens;
      if constexpr (FORM == kEach) {
        lab::tensor_load_2d(base + slot * kTileBytes, &map, 0, row, bar);
      } else {
        lab::cluster_arrive(freed + 8 * slot);
        if (rank == 0) {
          lab::cluster_wait(base + kFreeAt<D> + 8 * slot, (c / D) & 1);
          lab::tensor_cast_2d(base + slot * kTileBytes, &map, 0, row, bar, (1u << C) - 1);
        }
      }
    }
  }
#pragma unroll
  for (int p = 0; p < D; ++p)
    if (chunks > p) lab::mbar_wait_suspended(landed + 8 * ((chunks - 1 - p) % D), ((chunks - 1 - p) / D) & 1);
  clock.stop(times);
  lab::sink(sinks, seen);
#pragma unroll 1
  for (int i = tid; i < 2 * kTileBytes; i += kThreads)
    out[static_cast<long long>(blockIdx.x) * 2 * kTileBytes + i] = smem[(i / kTileBytes + D - 2) * kTileBytes + i % kTileBytes];
  lab::cluster_sync();
}

}  // namespace cast_cluster

template <int C, int FORM, int COLD, int D>
void run(lab::Bench& b, const CUtensorMap& map, uint8_t* out) {
  using namespace cast_cluster;
  constexpr int kCheckChunks = 8;
  const int blocks = b.sms / C * C;
  const std::string name = std::string("clusters of ") + std::to_string(C) + ", "
                           + (FORM == kEach ? "every CTA its own copy" : "one multicast copy a cluster") + ", "
                           + (COLD ? "a fresh tile a chunk" : "tiles the cache holds") + ", " + std::to_string(D) + " in flight";
  const auto kernel = cast<C, FORM, COLD, D>;
  constexpr int bytes = kSmemBytes<D>;
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  kernel<<<blocks, kThreads, bytes>>>(map, b.times, b.sinks, out, kCheckChunks);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<uint8_t> got(static_cast<size_t>(blocks) * 2 * kTileBytes);
  LAB_CHECK(cudaMemcpy(got.data(), out, got.size(), cudaMemcpyDeviceToHost));
  long long wrong = 0;
  for (int blk = 0; blk < blocks; ++blk)
    for (int slot = 0; slot < 2; ++slot) {
      //: the last two chunks' tiles: slots `D - 2` and `D - 1`, 8 chunks a multiple of either depth
      const int chunk = kCheckChunks - 2 + slot;
      const uint32_t first = static_cast<uint32_t>(tile_of(COLD, blk / C, chunk)) * kTileBytes;
      for (int i = 0; i < kTileBytes; ++i)
        wrong += got[(static_cast<size_t>(blk) * 2 + slot) * kTileBytes + i] != byte_of(first + i);
    }
  b.check(name, wrong == 0, std::to_string(wrong) + " of " + std::to_string(got.size()) + " landed bytes differ");
  const lab::Samples cycles = b.variant(
      [&] { kernel<<<blocks, kThreads, bytes>>>(map, b.times, b.sinks, out, kChunks); },
      [&] { return b.span(kThreads >> 5, blocks) / kChunks / (kTileBytes / 1024.0); });
  b.record(name, "cycles a kilobyte a CTA", cycles);
}

template <int C>
void sizes(lab::Bench& b, const CUtensorMap& map, uint8_t* out) {
  using namespace cast_cluster;
  run<C, kEach, 0, 2>(b, map, out);
  run<C, kCast, 0, 2>(b, map, out);
  run<C, kEach, 1, 2>(b, map, out);
  run<C, kCast, 1, 2>(b, map, out);
  run<C, kEach, 0, 4>(b, map, out);
  run<C, kCast, 0, 4>(b, map, out);
  run<C, kEach, 1, 4>(b, map, out);
  run<C, kCast, 1, 4>(b, map, out);
}

int main(int argc, char** argv) {
  using namespace cast_cluster;
  lab::Bench b(argc, argv);
  for (int c : {2, 4, 8})
    for (int f = 0; f < 2; ++f)
      for (int cold = 0; cold < 2; ++cold)
        for (int d : {2, 4}) {
          const std::string k = lab::kernel("cast_cluster::cast", {c, f, cold, d});
          b.expect(k, "UTMALDG", "==", 1);
          //: one thread of a CTA states its bytes and issues or arrives: that thread's branch is the chunk's
          b.expect(k, "LOOP_CONVERGENCE", "<=", 16);
        }
  if (b.expect_only) return 0;
  //: the table: a cluster's fresh tiles a chunk at the smallest cluster, the few warm ones among them
  const size_t tiles = static_cast<size_t>(b.sms / 2) * kChunks;
  std::vector<uint8_t> host(tiles * kTileBytes);
  for (size_t i = 0; i < host.size(); ++i) host[i] = byte_of(static_cast<uint32_t>(i));
  uint8_t *table = nullptr, *out = nullptr;
  LAB_CHECK(cudaMalloc(&table, host.size()));
  LAB_CHECK(cudaMemcpy(table, host.data(), host.size(), cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMalloc(&out, static_cast<size_t>(b.sms) * 2 * kTileBytes));
  //: THE TENSOR MAP, from the driver (its library is the running one's: no toolkit is linked for it)
  using Encode = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                              const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                              CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  void* const driver = dlopen("libcuda.so.1", RTLD_NOW);
  const auto encode = driver ? reinterpret_cast<Encode>(dlsym(driver, "cuTensorMapEncodeTiled")) : nullptr;
  if (!encode) {
    std::fprintf(stderr, "cast_cluster: the driver's cuTensorMapEncodeTiled is not reachable\n");
    return 2;
  }
  CUtensorMap map;
  const cuuint64_t dims[2] = {kRowBytes, tiles * kTokens}, strides[1] = {kRowBytes};
  const cuuint32_t box[2] = {kRowBytes, kTokens}, steps[2] = {1, 1};
  const CUresult made = encode(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, table, dims, strides, box, steps,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                               CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (made != CUDA_SUCCESS) {
    std::fprintf(stderr, "cast_cluster: cuTensorMapEncodeTiled refused the map (%d)\n", static_cast<int>(made));
    return 2;
  }
  sizes<2>(b, map, out);
  sizes<4>(b, map, out);
  sizes<8>(b, map, out);
  return 0;
}
