// measure/lab/lab.cuh -- THE LAB'S KIT: what every experiment shares -- a warp's cycles over a region, a sink the
// compiler cannot drop, the atoms as PTX, and the host runner that launches a variant, repeats it, prints one JSON line
// a result and declares the SASS each kernel must contain (checked by `tools/lab.py` before any number is kept).
// see docs/internals/tools/lab.md
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace lab {

// ------------------------------------------------------------------------------------------------- device

__device__ __forceinline__ int warp_id() { return static_cast<int>(threadIdx.x) >> 5; }
__device__ __forceinline__ int lane_id() { return static_cast<int>(threadIdx.x) & 31; }

//: A WARP'S CYCLES over a region, lane 0's `clock64`, stored at `(block, warp)`.
struct WarpClock {
  long long t0 = 0;
  __device__ __forceinline__ void start() { t0 = clock64(); }
  __device__ __forceinline__ void stop(long long* out) {
    const long long dt = clock64() - t0;
    if (lane_id() == 0) out[blockIdx.x * (blockDim.x >> 5) + warp_id()] = dt;
  }
};

//: THE SINK: a fold of everything a thread computed, stored unconditionally. A result tested against a value the
//: compiler can fold lets ptxas delete the work behind it -- the failure the expectations exist to catch.
__device__ __forceinline__ void sink(int* out, uint32_t v) {
  out[blockIdx.x * blockDim.x + threadIdx.x] = static_cast<int>(v);
}
__device__ __forceinline__ uint32_t bits(float x) { return __float_as_uint(x); }
__device__ __forceinline__ uint32_t bits(uint32_t x) { return x; }
__device__ __forceinline__ uint32_t bits(int x) { return static_cast<uint32_t>(x); }

//: THE ATOMS, one PTX instruction each, so a variant names exactly what it issues.
__device__ __forceinline__ void mma_f16_f32(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                            uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16_f32_fresh(float (&d)[4], const uint32_t (&a)[4],
                                                  uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%10,%10,%10,%10};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.0f));
}
__device__ __forceinline__ void mma_f16_f16(uint32_t (&d)[2], const uint32_t (&a)[4], uint32_t b0,
                                            uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
      : "+r"(d[0]), "+r"(d[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_bf16_f32(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                             uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_u8s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// --------------------------------------------------------------------------------------------------- host

//: A KERNEL'S DEMANGLED NAME AS AN EXPECTATION'S PATTERN: `kernel("tax", {3, 17})` matches
//: `ns::tax<(int)3, (int)17>(...)`, the spelling `cu++filt` gives integer template arguments.
inline std::string kernel(const std::string& name, std::initializer_list<int> args) {
  std::string out = name + "<";
  int i = 0;
  for (int a : args) out += std::string(i++ ? ", " : "") + "\\(int\\)" + std::to_string(a);
  return out + ">\\(";
}

#define LAB_CHECK(call)                                                                       \
  do {                                                                                        \
    const cudaError_t e_ = (call);                                                            \
    if (e_ != cudaSuccess) {                                                                  \
      std::fprintf(stderr, "lab: %s failed: %s\n", #call, cudaGetErrorString(e_));            \
      std::exit(2);                                                                           \
    }                                                                                         \
  } while (0)

//: A RESULT'S SAMPLES, one a repetition.
struct Samples {
  std::vector<double> v;
  double median() const {
    std::vector<double> s = v;
    std::sort(s.begin(), s.end());
    return s.empty() ? 0.0 : s[s.size() / 2];
  }
};

//: THE RUNNER. `--expect` prints the declared SASS and runs nothing; otherwise each `variant` warms up once, repeats
//: `--reps` times, and `record` prints a result as one JSON line `tools/lab.py` reads.
class Bench {
 public:
  static constexpr int kMaxWarps = 32;
  int sms = 0, reps = 7;
  bool expect_only = false;
  long long* times = nullptr;
  int* sinks = nullptr;

  Bench(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      if (!std::strcmp(argv[i], "--expect")) expect_only = true;
      if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
    }
    if (expect_only) return;
    LAB_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    LAB_CHECK(cudaMalloc(&times, sizeof(long long) * sms * kMaxWarps));
    LAB_CHECK(cudaMalloc(&sinks, sizeof(int) * sms * kMaxWarps * 32));
    cudaDeviceProp prop{};
    LAB_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("{\"lab\": \"device\", \"name\": \"%s\", \"cc\": %d%d, \"sms\": %d}\n", prop.name, prop.major,
                prop.minor, sms);
  }

  //: a kernel whose demangled name matches `kernel` must hold `op` `count` times (`op` an opcode prefix, `cmp` one of
  //: ">=", "==", "<=")
  void expect(const std::string& kernel, const std::string& op, const char* cmp, int count) const {
    if (!expect_only) return;
    std::string escaped;
    for (char c : kernel) escaped += c == '\\' ? std::string("\\\\") : std::string(1, c);
    std::printf("{\"lab\": \"expect\", \"kernel\": \"%s\", \"op\": \"%s\", \"cmp\": \"%s\", \"count\": %d}\n",
                escaped.c_str(), op.c_str(), cmp, count);
  }

  //: the per-(block, warp) cycles of the last launch, `warps` a block
  std::vector<long long> cycles(int warps) const {
    std::vector<long long> h(static_cast<size_t>(sms) * warps);
    LAB_CHECK(cudaMemcpy(h.data(), times, h.size() * sizeof(long long), cudaMemcpyDeviceToHost));
    return h;
  }

  //: the mean over blocks of warps `[first, first + count)`'s cycles
  static double mean(const std::vector<long long>& t, int warps, int first, int count) {
    double s = 0;
    const size_t blocks = t.size() / warps;
    for (size_t b = 0; b < blocks; ++b)
      for (int w = first; w < first + count; ++w) s += double(t[b * warps + w]);
    return s / double(blocks * count);
  }

  //: one variant's samples: `launch()` runs it once, `measure()` reads the value from its cycles
  template <class Launch, class Measure>
  Samples variant(Launch launch, Measure measure) const {
    Samples out;
    launch();
    LAB_CHECK(cudaDeviceSynchronize());
    for (int r = 0; r < reps; ++r) {
      launch();
      LAB_CHECK(cudaDeviceSynchronize());
      out.v.push_back(measure());
    }
    return out;
  }

  //: a result, as one JSON line
  void record(const std::string& name, const char* unit, const Samples& s) const {
    std::string vals;
    for (size_t i = 0; i < s.v.size(); ++i) vals += (i ? ", " : "") + std::to_string(s.v[i]);
    std::printf("{\"lab\": \"result\", \"name\": \"%s\", \"unit\": \"%s\", \"samples\": [%s]}\n", name.c_str(), unit,
                vals.c_str());
  }
};

}  // namespace lab
