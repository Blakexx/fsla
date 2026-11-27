// measure/lab/experiments/value_cast.cu -- QUESTION: the CTAs of a cluster all want the same tile of values; what
// does a kilobyte cost a CTA when ONE multicast tensor copy lands it in all eight, beside every CTA issuing its own
// copy of it? METHOD: clusters of 8 CTAs, 4 warps each; a chunk is a tile of 64 tokens x 128 bytes (8 KB) of a
// table every cluster reads. `each`: every CTA issues its own tensor copy of the chunk's tile. `cast`: the CTA of
// rank 0 issues one multicast copy to all eight; every CTA states the bytes at its own barrier and waits on it.
// Two tiles are in flight, a slot each by the chunk's parity. A copy into a slot is issued only past its readers:
// the CTA's own barrier before its own copy, the CLUSTER's before a copy into every mate. The last tiles are
// checked exactly in every CTA.
// TIMELINE: a CTA's trip is a chunk: every thread waits on the tile two chunks back and reads a word of it; one
// thread states the chunk's bytes at the barrier of its parity; the readers' barrier (the CTA's, or the
// cluster's); the copy issued (in `cast`, by rank 0 alone). Live: nothing held.
// see docs/internals/tools/lab.md#value-cast

#include "lab.cuh"

namespace value_cast {

constexpr int kCtas = 8, kThreads = 128, kTokens = 64, kRowBytes = 128, kTileBytes = kTokens * kRowBytes,
              kTableRows = 65536;
constexpr int kBarsAt = 2 * kTileBytes, kSmemBytes = kBarsAt + 16;
enum Form : int { kEach = 0, kCast, kForms };
const char* kFormName[kForms] = {"every CTA its own copy", "one multicast copy a cluster"};

constexpr uint8_t byte_of(uint32_t at) { return static_cast<uint8_t>(at * 2654435761u >> 24); }
constexpr int tile_of(int chunk) { return (chunk * 7) % (kTableRows / kTokens); }

template <int FORM>
__global__ void __cluster_dims__(kCtas, 1, 1) __launch_bounds__(kThreads, 1)
    cast(const __grid_constant__ CUtensorMap map, long long* times, int* sinks, uint8_t* out, int chunks) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x;
  const uint32_t rank = lab::cluster_rank();
  const uint32_t base = lab::smem(smem), landed = base + kBarsAt;
  if (tid == 0) {
    lab::mbar_init(landed, 1);
    lab::mbar_init(landed + 8, 1);
    lab::cluster_fence_init();
  }
  __syncthreads();
  //: every CTA of the cluster has its barriers before a copy can complete one
  lab::cluster_sync();

  uint32_t seen = 0;
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int c = 0; c < chunks; ++c) {
    const uint32_t slot = c & 1, bar = landed + 8 * slot;
    //: THE LANDED EDGE: the tile two chunks back is in the slot; a word of it a thread
    if (c >= 2) {
      lab::mbar_wait(bar, ((c >> 1) - 1) & 1);
      seen ^= smem[slot * kTileBytes + tid * 61] + c;
    }
    if (tid == 0) lab::mbar_expect(bar, kTileBytes);
    //: THE ISSUE EDGE: every reader of the slot, and every barrier's stated bytes, before the copy that overwrites
    //: it -- the CTA's own barrier for its own copy, the cluster's for a copy into every mate
    if constexpr (FORM == kEach) __syncthreads();
    if constexpr (FORM == kCast) lab::cluster_sync();
    if (tid == 0) {
      if constexpr (FORM == kEach) lab::tensor_load_2d(base + slot * kTileBytes, &map, 0, tile_of(c) * kTokens, bar);
      if constexpr (FORM == kCast) {
        if (rank == 0)
          lab::tensor_cast_2d(base + slot * kTileBytes, &map, 0, tile_of(c) * kTokens, bar, (1u << kCtas) - 1);
      }
    }
  }
#pragma unroll
  for (int p = 0; p < 2; ++p)
    if (chunks > p) lab::mbar_wait(landed + 8 * ((chunks - 1 - p) & 1), ((chunks - 1 - p) >> 1) & 1);
  clock.stop(times);
  lab::sink(sinks, seen);
#pragma unroll 1
  for (int i = tid; i < 2 * kTileBytes; i += kThreads)
    out[static_cast<long long>(blockIdx.x) * 2 * kTileBytes + i] = smem[i];
  lab::cluster_sync();
}

}  // namespace value_cast

template <int FORM>
void run(lab::Bench& b, const CUtensorMap& map, uint8_t* out) {
  using namespace value_cast;
  constexpr int kCheckChunks = 8, kChunks = 2048;
  const int blocks = b.sms / kCtas * kCtas;
  const std::string name = kFormName[FORM];
  const auto kernel = cast<FORM>;
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  kernel<<<blocks, kThreads, kSmemBytes>>>(map, b.times, b.sinks, out, kCheckChunks);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<uint8_t> got(static_cast<size_t>(blocks) * 2 * kTileBytes);
  LAB_CHECK(cudaMemcpy(got.data(), out, got.size(), cudaMemcpyDeviceToHost));
  long long wrong = 0;
  for (int blk = 0; blk < blocks; ++blk)
    for (int slot = 0; slot < 2; ++slot) {
      const int chunk = kCheckChunks - 2 + slot;
      for (int i = 0; i < kTileBytes; ++i)
        wrong += got[(static_cast<size_t>(blk) * 2 + slot) * kTileBytes + i]
                 != byte_of(static_cast<uint32_t>(tile_of(chunk)) * kTileBytes + i);
    }
  b.check(name, wrong == 0, std::to_string(wrong) + " of " + std::to_string(got.size()) + " landed bytes differ");
  const lab::Samples cycles = b.variant(
      [&] { kernel<<<blocks, kThreads, kSmemBytes>>>(map, b.times, b.sinks, out, kChunks); },
      [&] { return b.span(kThreads >> 5, blocks) / kChunks / (kTileBytes / 1024.0); });
  b.record(name, "cycles a kilobyte a CTA", cycles);
}

int main(int argc, char** argv) {
  using namespace value_cast;
  lab::Bench b(argc, argv);
  for (int f = 0; f < kForms; ++f) {
    b.expect(lab::kernel("cast", {f}), "UTMALDG", "==", 1);
    //: one thread of a CTA issues its copy: that thread's branch is the chunk's
    b.expect(lab::kernel("cast", {f}), "LOOP_CONVERGENCE", "<=", 8);
  }
  if (b.expect_only) return 0;
  std::vector<uint8_t> host(static_cast<size_t>(kTableRows) * kRowBytes);
  for (size_t i = 0; i < host.size(); ++i) host[i] = byte_of(static_cast<uint32_t>(i));
  uint8_t *table = nullptr, *out = nullptr;
  LAB_CHECK(cudaMalloc(&table, host.size()));
  LAB_CHECK(cudaMemcpy(table, host.data(), host.size(), cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMalloc(&out, static_cast<size_t>(b.sms) * 2 * kTileBytes));
  //: THE TENSOR MAPS, by the driver's encoder (`lab.cuh`)
  const auto encode = lab::tensor_map_encoder("value_cast");
  CUtensorMap map;
  const cuuint64_t dims[2] = {kRowBytes, kTableRows}, strides[1] = {kRowBytes};
  const cuuint32_t box[2] = {kRowBytes, kTokens}, steps[2] = {1, 1};
  const CUresult made = encode(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, table, dims, strides, box, steps,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                               CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (made != CUDA_SUCCESS) {
    std::fprintf(stderr, "value_cast: cuTensorMapEncodeTiled refused the map (%d)\n", static_cast<int>(made));
    return 2;
  }
  run<kEach>(b, map, out);
  run<kCast>(b, map, out);
  return 0;
}
