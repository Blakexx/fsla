// measure/lab/experiments/smem_rates.cu -- QUESTION: what does an SM's shared-memory pipe sustain a wavefront (128
// bytes a cycle is the documented bank rate) for loads -- `ld.shared.v4` and `ldmatrix.x4`, 4 wavefronts a warp each --
// and for atomic adds -- `red.shared.add.u32` with a lane a bank (1 wavefront a warp), two lanes a bank (2), four (4)?
// METHOD: one CTA an SM, 8 or 16 warps (2 or 4 a scheduler), each issuing independent operations at addresses that
// change every iteration (volatile asm, so none is hoisted or merged); the block's span over the wavefronts the SM
// issued.
// see docs/internals/tools/lab.md#smem-rates
#include "lab.cuh"

namespace smem_rates {

constexpr int kIters = 2000, kUnroll = 8, kRowWords = 1024;

__device__ __forceinline__ uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(a));
  return v;
}

__device__ __forceinline__ void red_add(uint32_t a, uint32_t v) {
  asm volatile("red.shared.add.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}

//: FORM 0 `ld.shared.v4` (a warp's 512 contiguous bytes); 1 `ldmatrix.x4` (four 8 x 16-byte matrices, rows 16 bytes
//: apart); 2 `red.shared.add` a lane a bank; 3 `red.shared.add` two lanes a bank (lanes l and l + 16 on one bank, words
//: 32 apart); 4 four lanes a bank
template <int FORM>
__global__ void __launch_bounds__(512, 1) pipe(int iters, long long* t, int* out) {
  extern __shared__ __align__(128) uint32_t words[];
  const int lane = lab::lane_id(), warp = lab::warp_id();
  for (int i = static_cast<int>(threadIdx.x); i < kRowWords * 16; i += static_cast<int>(blockDim.x))
    words[i] = static_cast<uint32_t>(i);
  __syncthreads();
  const uint32_t row = lab::smem(words + kRowWords * warp);
  uint32_t fold = 0;
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const uint32_t at = row + 512u * static_cast<uint32_t>((it + u) & 7);
      if constexpr (FORM == 0) {
        const uint4 v = lds128(at + 16u * lane);
        fold ^= v.x ^ v.y ^ v.z ^ v.w;
      } else if constexpr (FORM == 1) {
        uint32_t r[4];
        lab::ldsm(r, at + 16u * lane);
        fold ^= r[0] ^ r[1] ^ r[2] ^ r[3];
      } else if constexpr (FORM == 2) {
        red_add(at + 4u * lane, static_cast<uint32_t>(it));
      } else if constexpr (FORM == 3) {
        red_add(at + 4u * (lane & 15) + 128u * (lane >> 4), static_cast<uint32_t>(it));
      } else {
        red_add(at + 4u * (lane & 7) + 128u * (lane >> 3), static_cast<uint32_t>(it));
      }
    }
  }
  clock.stop(t);
  __syncthreads();
  lab::sink(out, fold ^ words[threadIdx.x]);
}

const char* kFormName[5] = {"ld.shared.v4, 4 wavefronts a warp", "ldmatrix.x4, 4 wavefronts a warp",
                            "red.shared.add, a lane a bank, 1 wavefront a warp",
                            "red.shared.add, two lanes a bank, 2 wavefronts a warp",
                            "red.shared.add, four lanes a bank, 4 wavefronts a warp"};
constexpr int kWavefronts[5] = {4, 4, 1, 2, 4};

template <int FORM>
void form(const lab::Bench& b, int warps) {
  const std::string name = lab::kernel("pipe", {FORM});
  if (FORM == 0) b.expect(name, "LDS.128", ">=", kUnroll);
  if (FORM == 1) b.expect(name, "LDSM.16.M88.4", ">=", kUnroll);
  if (FORM >= 2) b.expect(name, "ATOMS.ADD", ">=", kUnroll);
  if (b.expect_only) return;
  const int bytes = kRowWords * 16 * 4;
  LAB_CHECK(cudaFuncSetAttribute(pipe<FORM>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  const lab::Samples s = b.variant(
      [&] { pipe<FORM><<<b.sms, 32 * warps, bytes>>>(kIters, b.times, b.sinks); },
      [&] { return b.span(warps) / (double(warps) * kIters * kUnroll * kWavefronts[FORM]); });
  b.record(std::string(kFormName[FORM]) + " / " + std::to_string(warps) + " warps",
           "cycles a wavefront", s);
}

}  // namespace smem_rates

int main(int argc, char** argv) {
  lab::Bench b(argc, argv);
  for (int warps : {8, 16}) {
    smem_rates::form<0>(b, warps);
    smem_rates::form<1>(b, warps);
    smem_rates::form<2>(b, warps);
    smem_rates::form<3>(b, warps);
    smem_rates::form<4>(b, warps);
    if (b.expect_only) break;
  }
}
