// measure/lab/experiments/wgmma_width.cu -- QUESTION: what is one wgmma's own time by its OUTPUT WIDTH -- 8, 16,
// 32, 64, 128 and 256 columns at 64 rows and K 16 -- apart from its commit group's round trip? Does a step narrower
// than 64 columns cost less? METHOD: one warpgroup a CTA, a CTA an SM; a trip issues K steps into ONE accumulator
// in one commit group and waits for it, K = 1, 2, 4 and 8 (and 32 at 8 columns: a box's 32 groups of 8 leaves in
// one group); both operands shared-memory tiles of ones, every descriptor held in a register. A step's own time is
// the increment from 4 steps a group to 8 -- a group's first steps cost less each -- and the rest of a trip its
// group's round trip. Every accumulator is checked: 16 a step in every value.
// TIMELINE: no load: the tiles are written once before the clock. A trip is a commit group: the fence, K steps
// back to back on held descriptors, the commit, the wait. Live: the accumulator, N / 2 registers a thread.
// see docs/internals/tools/lab.md#wgmma-width
#include "lab.cuh"

namespace wgmma_width {

constexpr int kThreads = 128, kTileA = 0, kTileB = 2048;
//: B's tile, K-major: 8 columns a brick, a brick 16 lines of 16 bytes
template <int N>
constexpr int kSmem = kTileB + (N / 8) * 256;

//: m64n8k16, both operands shared memory and K-major: `d` += A [64 x 16] B [16 x 8]
__device__ __forceinline__ void step(float (&d)[4], uint64_t a, uint64_t b) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %6, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, p, 1, 1, 0, 0; }"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "l"(a), "l"(b), "r"(1));
}

//: m64n16k16, both operands shared memory and K-major: `d` += A [64 x 16] B [16 x 16]
__device__ __forceinline__ void step(float (&d)[8], uint64_t a, uint64_t b) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %10, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, p, 1, 1, 0, 0; }"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
               : "l"(a), "l"(b), "r"(1));
}

//: m64n32k16, both operands shared memory and K-major: `d` += A [64 x 16] B [16 x 32]
__device__ __forceinline__ void step(float (&d)[16], uint64_t a, uint64_t b) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %18, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, p, 1, 1, 0, 0; }"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
               : "l"(a), "l"(b), "r"(1));
}

//: m64n64k16, both operands shared memory and K-major: `d` += A [64 x 16] B [16 x 64]
__device__ __forceinline__ void step(float (&d)[32], uint64_t a, uint64_t b) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1, 0, 0; }"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
               : "l"(a), "l"(b), "r"(1));
}

//: m64n128k16, both operands shared memory and K-major: `d` += A [64 x 16] B [16 x 128]
__device__ __forceinline__ void step(float (&d)[64], uint64_t a, uint64_t b) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %66, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1, 0, 0; }"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
               : "l"(a), "l"(b), "r"(1));
}

//: m64n256k16, both operands shared memory and K-major: `d` += A [64 x 16] B [16 x 256]
__device__ __forceinline__ void step(float (&d)[128], uint64_t a, uint64_t b) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %130, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, p, 1, 1, 0, 0; }"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
               : "l"(a), "l"(b), "r"(1));
}

template <int N, int STEPS>
__global__ void __launch_bounds__(kThreads, 1) width(long long* times, float* out, int trips) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x;
#pragma unroll 1
  for (int i = tid; i < kSmem<N> / 2; i += kThreads) reinterpret_cast<uint16_t*>(smem)[i] = 0x3f80;
  lab::bulk_fence();
  __syncthreads();
  const uint32_t base = lab::smem(smem);
  const uint64_t a = lab::held(lab::wg_desc(base + kTileA, 8, 16)), b = lab::held(lab::wg_desc(base + kTileB, 8, 16));
  float d[N / 2] = {};

  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int t = 0; t < trips; ++t) {
    lab::wg_fence();
#pragma unroll
    for (int s = 0; s < STEPS; ++s) step(d, a, b);
    lab::wg_commit();
    lab::wg_wait<0>();
  }
#pragma unroll
  for (int i = 0; i < N / 2; ++i) lab::wg_hold(d[i]);
  clock.stop(times);
  //: every value is 16 a step: their largest and smallest, a thread
  float lo = d[0], hi = d[0];
#pragma unroll
  for (int i = 1; i < N / 2; ++i) {
    lo = fminf(lo, d[i]);
    hi = fmaxf(hi, d[i]);
  }
  out[2 * (blockIdx.x * kThreads + tid)] = lo;
  out[2 * (blockIdx.x * kThreads + tid) + 1] = hi;
}

}  // namespace wgmma_width

template <int N, int STEPS>
void run(lab::Bench& b, float* out) {
  using namespace wgmma_width;
  constexpr int kCheckTrips = 16, kTrips = 4096;
  const std::string name = std::to_string(N) + " columns, " + std::to_string(STEPS)
                           + (STEPS == 1 ? " step" : " steps") + " a group";
  constexpr int bytes = kSmem<N>;
  const auto kernel = width<N, STEPS>;
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  kernel<<<b.sms, kThreads, bytes>>>(b.times, out, kCheckTrips);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<float> got(static_cast<size_t>(b.sms) * kThreads * 2);
  LAB_CHECK(cudaMemcpy(got.data(), out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
  long long wrong = 0;
  for (float v : got) wrong += v != 16.0f * STEPS * kCheckTrips;
  b.check(name, wrong == 0,
          std::to_string(wrong) + " of " + std::to_string(got.size()) + " extremes differ from the exact sum");
  const lab::Samples cycles = b.variant([&] { kernel<<<b.sms, kThreads, bytes>>>(b.times, out, kTrips); },
                                        [&] { return b.span(kThreads >> 5) / kTrips; });
  b.record(name, "cycles a group", cycles);
}

template <int N>
void widths(lab::Bench& b, float* out) {
  run<N, 1>(b, out);
  run<N, 2>(b, out);
  run<N, 4>(b, out);
  run<N, 8>(b, out);
}

int main(int argc, char** argv) {
  using namespace wgmma_width;
  lab::Bench b(argc, argv);
  for (int n : {8, 16, 32, 64, 128, 256})
    for (int steps : {1, 2, 4, 8, 32}) {
      if (steps == 32 && n != 8) continue;
      const std::string k = lab::kernel("width", {n, steps});
      b.expect(k, "HGMMA", "==", steps);
      //: a step's shared-memory operands, both descriptors: A 64x16 and B 16xN bf16 elements, 2,048 + 32 N bytes (`lab steps`
      //: reads the same for every N here: 2,304 at 8 to 10,240 at 256)
      b.expect(k, "STEP_PORT", "<=", 2048 + 32 * n);
      b.exempt(k, "a rate probe: a wgmma's own time by its width, no kernel's step");
    }
  if (b.expect_only) return 0;
  float* out = nullptr;
  LAB_CHECK(cudaMalloc(&out, sizeof(float) * b.sms * kThreads * 2));
  widths<8>(b, out);
  run<8, 32>(b, out);
  widths<16>(b, out);
  widths<32>(b, out);
  widths<64>(b, out);
  widths<128>(b, out);
  widths<256>(b, out);
  return 0;
}
