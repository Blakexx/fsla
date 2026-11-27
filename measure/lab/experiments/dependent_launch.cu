// measure/lab/experiments/dependent_launch.cu -- QUESTION: the producer's launch is followed by the op's, which reads
// what it wrote. What does PROGRAMMATIC DEPENDENT LAUNCH save a pair of dependent kernels -- the second launched with
// the attribute, its prologue (what it reads that the first does not write) run while the first drains, then
// `griddepcontrol.wait` before it reads the first's outputs; the first `griddepcontrol.launch_dependents` at its top --
// against the two launched plainly? METHOD: the FIRST kernel: a CTA an SM of 256 threads writes its own 64 KB slice of
// an 8.4 MB buffer (16 bytes a lane an instruction). The SECOND: the same grid, its prologue loads its own 32 KB slice
// of a weights table into shared memory and folds it, then waits, then reads the first's slice and folds it with the
// weights, one word a thread out. A launch of 64 pairs back to back in one stream, timed by events; a pair's time is
// the mean. The second's every word is checked: it saw the first's writes of the same pair.
// TIMELINE: the first: its trigger, 16 stores a thread. The second: 8 loads a thread into shared memory and their
// fold, the dependency's wait, 16 loads a thread folded with the weights, a store. Live: a chain and its loads.
// see docs/internals/tools/lab.md#dependent-launch
#include "lab.cuh"

namespace dependent_launch {

constexpr int kThreads = 256, kSlice = 64 * 1024 / 16, kWeights = 32 * 1024 / 16, kPairs = 64;

__device__ __forceinline__ void launch_dependents() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void wait_dependency() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

//: the first kernel: its slice of `buffer`, each word its index plus the pair's `round`
__global__ void __launch_bounds__(kThreads, 1) first(uint4* buffer, uint32_t round) {
  launch_dependents();
  uint4* const slice = buffer + static_cast<long long>(blockIdx.x) * kSlice;
#pragma unroll
  for (int i = 0; i < kSlice / kThreads; ++i) {
    const uint32_t e = (blockIdx.x * kSlice + i * kThreads + threadIdx.x) * 4u + round;
    slice[i * kThreads + threadIdx.x] = make_uint4(e, e + 1, e + 2, e + 3);
  }
}

//: the second: its weights staged and folded (no dependency), then the first's slice read
__global__ void __launch_bounds__(kThreads, 1) second(const uint4* buffer, const uint4* weights, uint32_t* out) {
  __shared__ uint4 staged[kWeights];
  const uint4* const mine = weights + static_cast<long long>(blockIdx.x) * kWeights;
#pragma unroll
  for (int i = 0; i < kWeights / kThreads; ++i) staged[i * kThreads + threadIdx.x] = mine[i * kThreads + threadIdx.x];
  __syncthreads();
  uint32_t w = 0;
#pragma unroll
  for (int i = 0; i < kWeights / kThreads; ++i) w += staged[((i + threadIdx.x) % (kWeights / kThreads)) * kThreads + threadIdx.x].x;
  wait_dependency();
  const uint4* const slice = buffer + static_cast<long long>(blockIdx.x) * kSlice;
  uint32_t sum = w;
#pragma unroll
  for (int i = 0; i < kSlice / kThreads; ++i) {
    const uint4 v = slice[i * kThreads + threadIdx.x];
    sum = sum * 3u + (v.x ^ v.w);
  }
  out[blockIdx.x * kThreads + threadIdx.x] = sum;
}

}  // namespace dependent_launch

int main(int argc, char** argv) {
  using namespace dependent_launch;
  lab::Bench b(argc, argv);
  b.expect("dependent_launch::first\\(", "STG", ">=", 16);
  b.expect("dependent_launch::second\\(", "LDG", ">=", 16);
  if (b.expect_only) return 0;
  uint4 *buffer = nullptr, *weights = nullptr;
  uint32_t* out = nullptr;
  LAB_CHECK(cudaMalloc(&buffer, static_cast<size_t>(b.sms) * kSlice * 16));
  LAB_CHECK(cudaMalloc(&weights, static_cast<size_t>(b.sms) * kWeights * 16));
  LAB_CHECK(cudaMalloc(&out, static_cast<size_t>(b.sms) * kThreads * 4));
  std::vector<uint32_t> hw(static_cast<size_t>(b.sms) * kWeights * 4);
  for (size_t i = 0; i < hw.size(); ++i) hw[i] = static_cast<uint32_t>(i * 2654435761u);
  LAB_CHECK(cudaMemcpy(weights, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice));
  cudaStream_t stream;
  LAB_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  //: a pair: the first plainly, the second plainly or with the programmatic serialization attribute
  const auto pair = [&](bool dependent, uint32_t round) {
    first<<<b.sms, kThreads, 0, stream>>>(buffer, round);
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(b.sms);
    config.blockDim = dim3(kThreads);
    config.stream = stream;
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;
    config.attrs = attribute;
    config.numAttrs = dependent ? 1 : 0;
    LAB_CHECK(cudaLaunchKernelEx(&config, second, static_cast<const uint4*>(buffer), static_cast<const uint4*>(weights), out));
  };
  //: the host's fold of CTA `blk`, thread `tid` after the pair of `round`
  const auto want = [&](int blk, int tid, uint32_t round) {
    uint32_t w = 0;
    for (int i = 0; i < kWeights / kThreads; ++i)
      w += hw[((static_cast<size_t>(blk) * kWeights) + ((i + tid) % (kWeights / kThreads)) * kThreads + tid) * 4];
    uint32_t sum = w;
    for (int i = 0; i < kSlice / kThreads; ++i) {
      const uint32_t e = (blk * kSlice + i * kThreads + tid) * 4u + round;
      sum = sum * 3u + (e ^ (e + 3));
    }
    return sum;
  };
  cudaEvent_t e0, e1;
  LAB_CHECK(cudaEventCreate(&e0));
  LAB_CHECK(cudaEventCreate(&e1));
  for (int dependent = 0; dependent < 2; ++dependent) {
    const std::string name = dependent ? "the second launched dependent (griddepcontrol)" : "two plain launches";
    //: THE CHECK: every pair's second read its own first's words: the last of a run of pairs, each its own round
    for (uint32_t r = 1; r <= 8; ++r) pair(dependent, r);
    LAB_CHECK(cudaStreamSynchronize(stream));
    std::vector<uint32_t> got(static_cast<size_t>(b.sms) * kThreads);
    LAB_CHECK(cudaMemcpy(got.data(), out, got.size() * 4, cudaMemcpyDeviceToHost));
    long long wrong = 0;
    for (int blk = 0; blk < b.sms; ++blk)
      for (int tid = 0; tid < kThreads; ++tid) wrong += got[static_cast<size_t>(blk) * kThreads + tid] != want(blk, tid, 8);
    b.check(name, wrong == 0, std::to_string(wrong) + " of " + std::to_string(got.size()) + " folds differ");
    const lab::Samples us = b.variant(
        [&] {
          LAB_CHECK(cudaEventRecord(e0, stream));
          for (int p = 0; p < kPairs; ++p) pair(dependent, static_cast<uint32_t>(p));
          LAB_CHECK(cudaEventRecord(e1, stream));
          LAB_CHECK(cudaStreamSynchronize(stream));
        },
        [&] {
          float ms = 0;
          LAB_CHECK(cudaEventElapsedTime(&ms, e0, e1));
          return ms * 1000.0 / kPairs;
        });
    b.record(name + ": a pair", "microseconds a pair", us);
  }
  return 0;
}
