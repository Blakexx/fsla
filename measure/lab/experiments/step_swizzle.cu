// measure/lab/experiments/step_swizzle.cu -- QUESTION: does the tensor step read its A operand from TOKEN-MAJOR ROWS --
// each of its 16 K entries a contiguous 128 bytes holding that entry's 64 M values, the row's eight 16-byte pieces
// turned by the row's place (the descriptor's 128-byte swizzle) -- and compute what it computes from the unswizzled
// brick layout the lab's kernels use?
// METHOD: one warpgroup. The host lays one A [64 x 16] of small integers in both layouts and one B [16 x 16]; the
// kernel copies the three tiles into shared memory (the swizzled tile at a 1,024-byte boundary), issues one step on
// each A against the same B, and writes both accumulators out. CHECKED: every thread's 8 sums are the same from both
// layouts, and their total over the warpgroup is the product's total, computed on the host.
// TIMELINE: two m64n16k16 steps in one commit group, their operands landed before the fence. Live: two accumulators.
// see docs/internals/tools/lab.md#step-swizzle
#include "lab.cuh"

namespace step_swizzle {

constexpr int kM = 64, kK = 16, kN = 16, kThreads = 128;
//: a tile's room: the swizzled A is two 1,024-byte halves; the plain A 8 bricks 16 lines apart; B two
constexpr int kTile = 2048, kSwizzle128 = 1;

constexpr int a_of(int m, int k) { return (m * 7 + k * 3) % 4 + 1; }
constexpr int b_of(int k, int n) { return (k * 5 + n) % 3 + 1; }

__global__ void __launch_bounds__(kThreads, 1)
    stepped(const uint32_t* plain, const uint32_t* turned, const uint32_t* other, float* out, int* at_boundary) {
  __shared__ alignas(128) uint32_t smem[(4 * kTile) / 4];
  const uint32_t base = lab::smem(smem);
  //: the swizzle reads an address's own bits: the turned tile starts at the first 1,024-byte boundary
  const uint32_t skip = (1024u - (base & 1023u)) & 1023u;
  uint32_t* const first = smem + skip / 4;
#pragma unroll 1
  for (int i = threadIdx.x; i < kTile / 4; i += kThreads) {
    first[i] = turned[i];
    first[kTile / 4 + i] = plain[i];
    first[2 * (kTile / 4) + i] = other[i];
  }
  lab::bulk_fence();
  __syncthreads();
  const uint32_t tile = base + skip;
  const uint64_t a_turned = lab::wg_desc(tile, 64, 64) | (static_cast<uint64_t>(kSwizzle128) << 62);
  const uint64_t a_plain = lab::wg_desc(tile + kTile, 8, 16);
  const uint64_t b = lab::wg_desc(tile + 2 * kTile, 8, 16);
  float from_plain[8], from_turned[8];
  lab::wg_fence();
  lab::wg_bf16_n16_ss_mna(from_plain, a_plain, b, false);
  lab::wg_bf16_n16_ss_mna(from_turned, a_turned, b, false);
  lab::wg_commit();
  lab::wg_wait<0>();
  float* const mine = out + (static_cast<size_t>(blockIdx.x) * kThreads + threadIdx.x) * 16;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    mine[i] = from_plain[i];
    mine[8 + i] = from_turned[i];
  }
  if (threadIdx.x == 0) at_boundary[blockIdx.x] = static_cast<int>((tile & 1023u) == 0u);
}

inline uint16_t bf16(int v) {
  const float f = static_cast<float>(v);
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return static_cast<uint16_t>(u >> 16);
}

}  // namespace step_swizzle

int main(int argc, char** argv) {
  using namespace step_swizzle;
  lab::Bench b(argc, argv);
  //: a layout probe: two steps on tiles copied in once, their sums compared; no kernel's schedule
  b.exempt("step_swizzle::stepped", "a layout probe: one step on each of two layouts of the same operand, no kernel's step");
  b.expect("step_swizzle::stepped", "WAIT_AFTER_ISSUE", "<=", 1);
  //: the tiles are copied in once, a load and a store a word: the probe times nothing
  b.expect("step_swizzle::stepped", "LOAD_IN_TRIP", "<=", 3);
  if (b.expect_only) return 0;
  std::vector<uint16_t> plain(kTile / 2, 0), turned(kTile / 2, 0), other(kTile / 2, 0);
  double total = 0;
  for (int m = 0; m < kM; ++m)
    for (int k = 0; k < kK; ++k) {
      const uint16_t v = bf16(a_of(m, k));
      //: the brick layout: a line one K's 8 M values of a brick, a brick's K 0 to 7 8 lines, K 8 to 15 8 lines on, bricks 16 apart
      plain[(((m / 8) * 16 + k % 8 + (k / 8) * 8) * 16 + (m % 8) * 2) / 2] = v;
      //: the token-major rows: a K entry's 64 M values 128 bytes, its piece `m / 8` at `(m / 8) ^ (row & 7)`; 8 rows a half
      turned[((k / 8) * 1024 + (k % 8) * 128 + (((m / 8) ^ (k % 8)) * 16) + (m % 8) * 2) / 2] = v;
      for (int n = 0; n < kN; ++n) total += static_cast<double>(a_of(m, k)) * b_of(k, n);
    }
  for (int k = 0; k < kK; ++k)
    for (int n = 0; n < kN; ++n) other[(((n / 8) * 16 + n % 8 + (k / 8) * 8) * 16 + (k % 8) * 2) / 2] = bf16(b_of(k, n));
  uint32_t *dplain = nullptr, *dturned = nullptr, *dother = nullptr;
  float* dout = nullptr;
  int* dat = nullptr;
  const size_t outs = static_cast<size_t>(b.sms) * kThreads * 16;
  LAB_CHECK(cudaMalloc(&dplain, kTile));
  LAB_CHECK(cudaMalloc(&dturned, kTile));
  LAB_CHECK(cudaMalloc(&dother, kTile));
  LAB_CHECK(cudaMalloc(&dout, outs * sizeof(float)));
  LAB_CHECK(cudaMalloc(&dat, static_cast<size_t>(b.sms) * sizeof(int)));
  LAB_CHECK(cudaMemcpy(dplain, plain.data(), kTile, cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMemcpy(dturned, turned.data(), kTile, cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMemcpy(dother, other.data(), kTile, cudaMemcpyHostToDevice));
  stepped<<<b.sms, kThreads>>>(dplain, dturned, dother, dout, dat);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<float> got(outs);
  std::vector<int> at(b.sms);
  LAB_CHECK(cudaMemcpy(got.data(), dout, outs * sizeof(float), cudaMemcpyDeviceToHost));
  LAB_CHECK(cudaMemcpy(at.data(), dat, at.size() * sizeof(int), cudaMemcpyDeviceToHost));
  long long differ = 0, off_total = 0, off_boundary = 0;
  for (int blk = 0; blk < b.sms; ++blk) {
    double sum = 0;
    for (int t = 0; t < kThreads; ++t)
      for (int i = 0; i < 8; ++i) {
        const float* const mine = got.data() + (static_cast<size_t>(blk) * kThreads + t) * 16;
        differ += mine[i] != mine[8 + i];
        sum += mine[i];
      }
    off_total += sum != total;
    off_boundary += at[blk] == 0;
  }
  b.check("a step's A read from token-major rows, the 128-byte swizzle, against the brick layout",
          differ == 0 && off_total == 0 && off_boundary == 0,
          std::to_string(differ) + " of " + std::to_string(static_cast<long long>(b.sms) * kThreads * 8)
              + " sums differ between the layouts; " + std::to_string(off_total) + " of " + std::to_string(b.sms)
              + " warpgroups' totals are not the product's; " + std::to_string(off_boundary) + " tiles off a boundary");
  return 0;
}
