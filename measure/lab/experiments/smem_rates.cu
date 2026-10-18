// measure/lab/experiments/smem_rates.cu -- QUESTION: what does an SM's shared-memory pipe sustain a wavefront (128
// bytes a cycle is the documented bank rate) for loads -- `ld.shared.v4` and `ldmatrix.x4`, 4 wavefronts a warp each --
// and for atomic adds -- `red.shared.add.u32` with a lane a bank (1 wavefront a warp), two lanes a bank (2), four (4)
// -- and which global traffic takes its cycles: data a load returns to registers from L1 or from L2, allocated in L1
// or not, and a `cp.async` copy's write into shared memory, allocated in L1 or not?
// METHOD: one CTA an SM, 8 or 16 warps (2 or 4 a scheduler), each issuing independent operations at addresses that
// change every iteration (volatile asm, so none is hoisted or merged); the block's span over the wavefronts the SM
// issued. The sharing: 8 warps issue `ldmatrix.x4` as above while 8 stream global data in one form, then each alone;
// the ldmatrix warps' cycles a wavefront and the stream's bytes a cycle, the pipe cycles the stream took from them.
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

//: THE SHARING: 16 warps, 0-7 issuing `ldmatrix.x4` as form 1 does over their 4 KB, 8-15 streaming global data in
//: form `GF` (below), a batch 8 instructions of 512 bytes a warp; `ROLE` 1 the ldmatrix warps alone, 2 the stream alone
//: (`kBatches` a warp), 3 both, the stream running until every ldmatrix warp is done
constexpr int kShareWarps = 16, kHalf = 8, kBatch = 8, kRing = 8192, kBatches = 4000;
constexpr uint32_t kLarge = 4u << 20, kSmall = 16u << 10;
//: the shared layout: the ldmatrix warps' rows, each stream warp's two-batch copy ring, the count of finished warps
constexpr int kRingsAt = kHalf * kRowWords * 4, kDoneAt = kRingsAt + kHalf * kRing, kShareBytes = kDoneAt + 16;

//: FORM `GF`: 0 `ld.global.nc.v4` (`__ldg`) over a 16 KB table (L1 hits); 1 the same over a 4 MB table (L2 hits,
//: allocated in L1); 2 `ld.global.cg.v4` over it (not allocated); 3 `cp.async.cg` from it into the warp's ring; 4 `cp.async.ca`
const char* kGlobalName[5] = {"ld.global.nc.v4, L1 hits", "ld.global.nc.v4, L2 hits", "ld.global.cg.v4, L2 hits",
                              "cp.async.cg into shared memory, L2 hits", "cp.async.ca into shared memory, L2 hits"};

template <int GF>
__device__ __forceinline__ uint32_t global16(const uint8_t* src, uint32_t dst) {
  uint4 v{};
  if constexpr (GF <= 1) {
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(src));
  } else if constexpr (GF == 2) {
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(src));
  } else if constexpr (GF == 3) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
  } else {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
  }
  return v.x ^ v.y ^ v.z ^ v.w;
}

template <int GF, int ROLE>
__global__ void __launch_bounds__(512, 1) share(int iters, const uint8_t* table, long long* t, int* out, int* done_n) {
  extern __shared__ __align__(128) uint32_t words[];
  const int lane = lab::lane_id(), warp = lab::warp_id();
  for (int i = static_cast<int>(threadIdx.x); i < kHalf * kRowWords; i += static_cast<int>(blockDim.x))
    words[i] = static_cast<uint32_t>(i);
  volatile int* const done = reinterpret_cast<volatile int*>(reinterpret_cast<uint8_t*>(words) + kDoneAt);
  if (threadIdx.x == 0) *done = 0;
  __syncthreads();
  uint32_t fold = 0;
  lab::WarpSpan clock;
  clock.start();
  if (warp < kHalf) {
    if constexpr ((ROLE & 1) != 0) {
      const uint32_t row = lab::smem(words + kRowWords * warp);
#pragma unroll 1
      for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
          uint32_t r[4];
          lab::ldsm(r, row + 512u * static_cast<uint32_t>((it + u) & 7) + 16u * lane);
          fold ^= r[0] ^ r[1] ^ r[2] ^ r[3];
        }
      }
      if (lane == 0) atomicAdd(const_cast<int*>(done), 1);
    }
  } else if constexpr ((ROLE & 2) != 0) {
    const uint32_t g = static_cast<uint32_t>(warp - kHalf), span = GF == 0 ? kSmall : kLarge;
    const uint32_t ring = lab::smem(words) + kRingsAt + g * kRing;
    uint32_t at = ((blockIdx.x * kHalf + g) * 64u * 512u) & (span - 1);
    int n = 0;
#pragma unroll 1
    while (ROLE == 3 ? *done < kHalf : n < kBatches) {
#pragma unroll
      for (int u = 0; u < kBatch; ++u)
        fold ^= global16<GF>(table + ((at + 512u * u) & (span - 1)) + 16u * lane,
                             ring + 4096u * (n & 1) + 512u * u + 16u * lane);
      if constexpr (GF >= 3) asm volatile("cp.async.commit_group;\ncp.async.wait_group 1;" ::: "memory");
      at = (at + 512u * kBatch) & (span - 1);
      ++n;
    }
    if constexpr (GF >= 3) asm volatile("cp.async.wait_group 0;" ::: "memory");
    if (lane == 0) done_n[blockIdx.x * kShareWarps + warp] = n;
  }
  clock.stop(t);
  lab::sink(out, fold);
}

//: one sharing variant: the ldmatrix warps' cycles a wavefront (roles 1, 3) and the stream's bytes a cycle (2, 3), each
//: over its own warps' span, the mean over SMs
template <int GF, int ROLE>
void share_form(const lab::Bench& b, const uint8_t* table, int* done_n) {
  const std::string name = lab::kernel("share", {GF, ROLE});
  if (ROLE & 1) b.expect(name, "LDSM.16.M88.4", ">=", kUnroll);
  if (ROLE & 2) b.expect(name, GF >= 3 ? "LDGSTS" : "LDG.E", ">=", kBatch);
  if (b.expect_only) return;
  LAB_CHECK(cudaFuncSetAttribute(share<GF, ROLE>, cudaFuncAttributeMaxDynamicSharedMemorySize, kShareBytes));
  lab::Samples bytes;
  const lab::Samples ldsm = b.variant(
      [&] { share<GF, ROLE><<<b.sms, 32 * kShareWarps, kShareBytes>>>(kIters, table, b.times, b.sinks, done_n); },
      [&] {
        std::vector<long long> h(static_cast<size_t>(b.sms) * kShareWarps * 2);
        std::vector<int> n(static_cast<size_t>(b.sms) * kShareWarps);
        LAB_CHECK(cudaMemcpy(h.data(), b.times, h.size() * sizeof(long long), cudaMemcpyDeviceToHost));
        LAB_CHECK(cudaMemcpy(n.data(), done_n, n.size() * sizeof(int), cudaMemcpyDeviceToHost));
        const auto span = [&](int sm, int first) {
          long long lo = h[2 * (sm * kShareWarps + first)], hi = h[2 * (sm * kShareWarps + first) + 1];
          for (int w = first + 1; w < first + kHalf; ++w) {
            lo = std::min(lo, h[2 * (sm * kShareWarps + w)]);
            hi = std::max(hi, h[2 * (sm * kShareWarps + w) + 1]);
          }
          return double(hi - lo);
        };
        double cyc = 0, rate = 0;
        for (int sm = 0; sm < b.sms; ++sm) {
          cyc += span(sm, 0) / (double(kHalf) * kIters * kUnroll * 4);
          double moved = 0;
          for (int w = kHalf; w < kShareWarps; ++w) moved += double(n[sm * kShareWarps + w]) * kBatch * 512;
          rate += moved / span(sm, kHalf);
        }
        bytes.v.push_back(rate / b.sms);
        return cyc / b.sms;
      });
  const std::string beside = ROLE == 1 ? "ldmatrix.x4 alone, 8 of 16 warps" : "ldmatrix.x4 beside " +
                                                                                    std::string(kGlobalName[GF]);
  const std::string stream = std::string(kGlobalName[GF]) + (ROLE == 3 ? ", beside ldmatrix.x4" : ", alone");
  if (ROLE & 1) {
    b.unchecked(beside, "a rate probe: its loads fold into a sink");
    b.record(beside, "cycles a wavefront", ldsm);
  }
  if (ROLE & 2) {
    b.unchecked(stream, "a rate probe: its loads fold into a sink");
    b.record(stream, "bytes a cycle", bytes);
  }
}

template <int GF>
void shared_by(const lab::Bench& b, const uint8_t* table, int* done_n) {
  share_form<GF, 2>(b, table, done_n);
  share_form<GF, 3>(b, table, done_n);
}

}  // namespace smem_rates

int main(int argc, char** argv) {
  using namespace smem_rates;
  lab::Bench b(argc, argv);
  for (int warps : {8, 16}) {
    form<0>(b, warps);
    form<1>(b, warps);
    form<2>(b, warps);
    form<3>(b, warps);
    form<4>(b, warps);
    if (b.expect_only) break;
  }
  uint8_t* table = nullptr;
  int* done_n = nullptr;
  if (!b.expect_only) {
    LAB_CHECK(cudaMalloc(&table, kLarge));
    LAB_CHECK(cudaMemset(table, 0x5a, kLarge));
    LAB_CHECK(cudaMalloc(&done_n, sizeof(int) * b.sms * kShareWarps));
  }
  share_form<0, 1>(b, table, done_n);
  shared_by<0>(b, table, done_n);
  shared_by<1>(b, table, done_n);
  shared_by<2>(b, table, done_n);
  shared_by<3>(b, table, done_n);
  shared_by<4>(b, table, done_n);
}
