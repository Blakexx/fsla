// measure/lab/experiments/tma_box.cu -- QUESTION: the producer's loader stages a K-step's `x` [128 x 64] and weights
// [256 x 64] bf16 (48 KB) a stage; what does a stage cost a CTA by the tensor copy (TMA) against `cp.async` by one
// warp? METHOD: a CTA an SM, a ring of three 48 KB stages, a loader and a warpgroup that waits on each stage's FULL
// barrier, reads a word of it a thread and releases it (its EMPTY barrier); the operands L2-resident (each CTA its
// own 128-token tile of an 8.6 MB `x` of hidden 256, every CTA the same 256 x 256 weights), the K-steps of four jobs.
// THE FORMS: `cp.async` 16 bytes a lane by one warp into wgmma's unswizzled K-major bricks (the projection's
// `stage`); one tensor copy an operand of a 2-d box of 128-byte lines, 128-byte swizzled (the form a swizzled wgmma
// descriptor reads); one tensor copy an operand of a 3-d box of 16-byte lines (8 values, rows, the K chunks: the
// unswizzled bricks themselves). The last three stages landed are checked byte for byte in every CTA.
// TIMELINE: the loader's trip is a stage: its slot's EMPTY edge, the copies (one thread's two tensor copies after
// its stated bytes, or a warp's 96 `cp.async` and their landed arrivals), the FULL edge. The reader's trip: the FULL
// edge, a word a thread, the EMPTY arrival. Live: nothing held.
// see docs/internals/tools/lab.md#tma-box
#include <cuda.h>
#include <cuda_bf16.h>
#include <dlfcn.h>

#include "lab.cuh"

namespace tma_box {

constexpr int kTokens = 128, kCols = 256, kK = 64, kHidden = 256, kStages = 3, kSteps = kHidden / kK;
constexpr int kXBytes = kTokens * kK * 2, kWBytes = kCols * kK * 2, kStageBytes = kXBytes + kWBytes;
constexpr int kBarsAt = kStages * kStageBytes, kSmem = kBarsAt + 2 * kStages * 8 + 1024;
constexpr int kThreads = 128 + 32;
enum Form : int { kCpAsync = 0, kBox128, kBox16, kForms };
const char* kFormName[kForms] = {"cp.async, a warp, 16 bytes a lane", "a tensor copy, 128-byte lines swizzled",
                                 "a tensor copy, 16-byte lines (the unswizzled bricks)"};

//: an operand's stage by `cp.async`: R rows, two lanes a row's 32 bytes (the projection's `stage`)
template <int R>
__device__ __forceinline__ void stage(uint32_t dst, const __nv_bfloat16* src, int lane) {
  const int row = lane >> 1, half = lane & 1;
  const uint32_t to = dst + (row >> 3) * 256 + half * 128 + (row & 7) * 16;
  const __nv_bfloat16* from = src + static_cast<long long>(row) * kHidden + half * 8;
#pragma unroll
  for (int rb = 0; rb < R / 16; ++rb)
#pragma unroll
    for (int kp = 0; kp < 4; ++kp)
      lab::cp_async16(to + kp * (R * 32) + rb * 512, from + static_cast<long long>(rb) * 16 * kHidden + kp * 16);
}

template <int FORM>
__global__ void __launch_bounds__(kThreads, 1)
    copy(const __grid_constant__ CUtensorMap xmap, const __grid_constant__ CUtensorMap wmap, const __nv_bfloat16* x,
         const __nv_bfloat16* w, long long* times, int* sinks, uint8_t* out, int total) {
  extern __shared__ __align__(1024) uint8_t raw[];
  //: the 128-byte swizzle's atoms are 1,024 bytes: the ring starts on one
  uint8_t* const smem = raw + ((1024 - (lab::smem(raw) & 1023)) & 1023);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t base = lab::smem(smem), full = base + kBarsAt, empty = full + 8 * kStages;
  if (tid == 0) {
    for (int s = 0; s < kStages; ++s) {
      lab::mbar_init(full + 8 * s, FORM == kCpAsync ? 32 : 1);
      lab::mbar_init(empty + 8 * s, 4);
    }
  }
  __syncthreads();
  uint32_t seen = 0;
  lab::WarpSpan clock;
  clock.start();
  if (warp == 4) {
#pragma unroll 1
    for (int i = 0; i < total; ++i) {
      const int slot = i % kStages, k = (i % kSteps) * kK;
      if (i >= kStages) lab::mbar_wait(empty + 8 * slot, ((i / kStages) - 1) & 1);
      const uint32_t at = base + slot * kStageBytes, bar = full + 8 * slot;
      if constexpr (FORM == kCpAsync) {
        stage<kTokens>(at, x + static_cast<long long>(blockIdx.x) * kTokens * kHidden + k, lane);
        stage<kCols>(at + kXBytes, w + k, lane);
        lab::mbar_arrive_when_landed(bar);
      } else if (lane == 0) {
        lab::mbar_expect(bar, kStageBytes);
        if constexpr (FORM == kBox128) {
          lab::tensor_load_2d(at, &xmap, k, blockIdx.x * kTokens, bar);
          lab::tensor_load_2d(at + kXBytes, &wmap, k, 0, bar);
        } else {
          lab::tensor_load_3d(at, &xmap, 0, blockIdx.x * kTokens, k / 8, bar);
          lab::tensor_load_3d(at + kXBytes, &wmap, 0, 0, k / 8, bar);
        }
      }
    }
  } else {
#pragma unroll 1
    for (int i = 0; i < total; ++i) {
      const int slot = i % kStages;
      lab::mbar_wait(full + 8 * slot, (i / kStages) & 1);
      seen += reinterpret_cast<const uint32_t*>(smem + slot * kStageBytes)[tid * 97 % (kStageBytes / 4)];
      __syncwarp();
      if (lane == 0) lab::mbar_arrive(empty + 8 * slot);
    }
  }
  clock.stop(times);
  lab::sink(sinks, seen);
  __syncthreads();
  for (int i = tid; i < kBarsAt; i += kThreads) out[static_cast<long long>(blockIdx.x) * kBarsAt + i] = smem[i];
}

inline uint16_t x_of(long long r, int k) { return static_cast<uint16_t>(r * 131 + k * 7 + 1); }
inline uint16_t w_of(long long c, int k) { return static_cast<uint16_t>(c * 37 + k * 5 + 3); }

//: where a stage's operand of R rows lands element (row, k) of the K-step, by form
inline int landed(int form, int R, int row, int k) {
  if (form == kCpAsync) return (k / 16) * (R * 32) + (row / 8) * 256 + ((k % 16) / 8) * 128 + (row % 8) * 16 + (k % 8) * 2;
  if (form == kBox16) return (k / 8) * (R * 16) + row * 16 + (k % 8) * 2;
  //: the 128-byte swizzle: a line a row, its 16-byte chunk `c` at `c ^ (row % 8)`
  return row * 128 + (((k / 8) ^ (row % 8)) * 16) + (k % 8) * 2;
}

}  // namespace tma_box

template <int FORM>
void run(lab::Bench& b, const CUtensorMap& xmap, const CUtensorMap& wmap, const __nv_bfloat16* x,
         const __nv_bfloat16* w, uint8_t* out) {
  using namespace tma_box;
  const auto kernel = copy<FORM>;
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  const std::string name = kFormName[FORM];
  constexpr int kCheck = 4 * kSteps + 1, kTotal = 64 * kSteps;
  kernel<<<b.sms, kThreads, kSmem>>>(xmap, wmap, x, w, b.times, b.sinks, out, kCheck);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<uint8_t> got(static_cast<size_t>(b.sms) * kBarsAt);
  LAB_CHECK(cudaMemcpy(got.data(), out, got.size(), cudaMemcpyDeviceToHost));
  long long wrong = 0, checked = 0;
  for (int blk = 0; blk < b.sms; ++blk)
    for (int i = kCheck - kStages; i < kCheck; ++i) {
      const uint8_t* slot = got.data() + static_cast<size_t>(blk) * kBarsAt + (i % kStages) * kStageBytes;
      const int k0 = (i % kSteps) * kK;
      for (int r = 0; r < kTokens + kCols; ++r)
        for (int k = 0; k < kK; ++k) {
          const bool isx = r < kTokens;
          const int row = isx ? r : r - kTokens;
          const uint16_t want = isx ? x_of(static_cast<long long>(blk) * kTokens + row, k0 + k) : w_of(row, k0 + k);
          uint16_t have;
          std::memcpy(&have, slot + (isx ? 0 : kXBytes) + landed(FORM, isx ? kTokens : kCols, row, k), 2);
          wrong += have != want;
          ++checked;
        }
    }
  b.check(name, wrong == 0, std::to_string(wrong) + " of " + std::to_string(checked) + " landed values differ");
  const lab::Samples cycles = b.variant(
      [&] { kernel<<<b.sms, kThreads, kSmem>>>(xmap, wmap, x, w, b.times, b.sinks, out, kTotal); },
      [&] { return b.span(kThreads >> 5) / kTotal; });
  b.record(name + ": a 48 KB stage", "cycles a stage a CTA", cycles);
}

int main(int argc, char** argv) {
  using namespace tma_box;
  lab::Bench b(argc, argv);
  b.expect(lab::kernel("copy", {0}), "LDGSTS", ">=", 96);
  for (int f : {1, 2}) {
    b.expect(lab::kernel("copy", {f}), "UTMALDG", "==", 2);
    //: one thread of the loader issues a stage's tensor copies: that lane's branch is the stage's
    b.expect(lab::kernel("copy", {f}), "LOOP_CONVERGENCE", "<=", 2);
  }
  if (b.expect_only) return 0;
  const long long rows = static_cast<long long>(b.sms) * kTokens;
  std::vector<uint16_t> hx(rows * kHidden), hw(static_cast<size_t>(kCols) * kHidden);
  for (long long r = 0; r < rows; ++r)
    for (int k = 0; k < kHidden; ++k) hx[r * kHidden + k] = x_of(r, k);
  for (int c = 0; c < kCols; ++c)
    for (int k = 0; k < kHidden; ++k) hw[static_cast<size_t>(c) * kHidden + k] = w_of(c, k);
  __nv_bfloat16 *x = nullptr, *w = nullptr;
  uint8_t* out = nullptr;
  LAB_CHECK(cudaMalloc(&x, hx.size() * 2));
  LAB_CHECK(cudaMalloc(&w, hw.size() * 2));
  LAB_CHECK(cudaMalloc(&out, static_cast<size_t>(b.sms) * kBarsAt));
  LAB_CHECK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMemcpy(w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
  //: THE TENSOR MAPS, from the driver (its library is the running one's: no toolkit is linked for it)
  using Encode = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                              const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                              CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  void* const driver = dlopen("libcuda.so.1", RTLD_NOW);
  const auto encode = driver ? reinterpret_cast<Encode>(dlsym(driver, "cuTensorMapEncodeTiled")) : nullptr;
  if (!encode) {
    std::fprintf(stderr, "tma_box: the driver's cuTensorMapEncodeTiled is not reachable\n");
    return 2;
  }
  const auto map2 = [&](CUtensorMap& m, void* p, long long outer, int box_outer) {
    const cuuint64_t dims[2] = {kHidden, static_cast<cuuint64_t>(outer)}, strides[1] = {kHidden * 2};
    const cuuint32_t box[2] = {kK, static_cast<cuuint32_t>(box_outer)}, steps[2] = {1, 1};
    return encode(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, p, dims, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
                  CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  };
  //: the 3-d view (8 values, the rows, the K chunks): a box lands its chunks' rows of 16 bytes one after another
  const auto map3 = [&](CUtensorMap& m, void* p, long long outer, int box_outer) {
    const cuuint64_t dims[3] = {8, static_cast<cuuint64_t>(outer), kHidden / 8}, strides[2] = {kHidden * 2, 16};
    const cuuint32_t box[3] = {8, static_cast<cuuint32_t>(box_outer), kK / 8}, steps[3] = {1, 1, 1};
    return encode(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, p, dims, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
                  CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  };
  CUtensorMap x2, w2, x3, w3;
  if (map2(x2, x, rows, kTokens) != CUDA_SUCCESS || map2(w2, w, kCols, kCols) != CUDA_SUCCESS) {
    std::fprintf(stderr, "tma_box: cuTensorMapEncodeTiled refused a 2-d map\n");
    return 2;
  }
  const bool three = map3(x3, x, rows, kTokens) == CUDA_SUCCESS && map3(w3, w, kCols, kCols) == CUDA_SUCCESS;
  run<kCpAsync>(b, x2, w2, x, w, out);
  run<kBox128>(b, x2, w2, x, w, out);
  if (three) run<kBox16>(b, x3, w3, x, w, out);
  else std::fprintf(stderr, "tma_box: cuTensorMapEncodeTiled refused the 3-d map; its form is not measured\n");
  return 0;
}
