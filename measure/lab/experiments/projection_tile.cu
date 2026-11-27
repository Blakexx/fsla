// measure/lab/experiments/projection_tile.cu -- QUESTION: the producer's projection tile alone: what does a job cost a
// CTA -- 128 tokens x 256 columns over the hidden axis, two math warpgroups of 64 tokens each issuing `wgmma`
// m64n256k16 from shared memory, one loader warp staging a K-step (64 hidden) of `x` [128 x 64] and the job's weights
// [256 x 64] by `cp.async` into a ring of three stages -- at hidden 256, 512 and 2,048, against its tensor floor (two
// warpgroups' steps on one pipe, 128 cycles a step)? METHOD: a CTA an SM, each its own 128-token tile of `x`, every CTA
// the same four jobs' weights (a write head's, from L2); a job is the K-loop and an epilogue that folds the
// accumulators; the next job's K-loop re-stages its tile (a write head's jobs, each re-reading its tile). Operands are
// small integers in bf16, so every sum is exact: the last job's accumulators of two CTAs are checked against the host's
// products.
// TIMELINE: the loader's trip is a stage: its slot's EMPTY edge, 96 copies a lane, the FULL edge (each lane's copies
// landed). A warpgroup's trip is a stage: the FULL edge, the async-proxy fence, 4 steps, the commit, a wait leaving one
// group in flight, then the previous stage's EMPTY edge; a job's end waits for all, releases, folds and clears. Live:
// the 128 accumulators a thread, two descriptors, the ring's parities.
// see docs/internals/tools/lab.md#projection-tile
#include <cuda_bf16.h>

#include "lab.cuh"

namespace projection_tile {

constexpr int kTokens = 128, kCols = 256, kK = 64, kStages = 3, kJobs = 4;
constexpr int kMath = 256, kThreads = kMath + 32;
//: a stage: `x` [128 x 64] then the weights [256 x 64], each K-major in bricks of 8 lines of 16 bytes: a k16 slice of
//: R rows is R * 32 bytes, its brick of 8 rows 256 bytes, the brick's K 8-15 128 bytes on
constexpr int kXBytes = kTokens * kK * 2, kWBytes = kCols * kK * 2, kStageBytes = kXBytes + kWBytes;
constexpr int kBarsAt = kStages * kStageBytes, kSmem = kBarsAt + 2 * kStages * 8;

//: m64n256k16, both operands shared memory and K-major: `d` += A [64 x 16] B [16 x 256]
__device__ __forceinline__ void step(float (&d)[128], uint64_t a, uint64_t b) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %130, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, p, 1, 1, 0, 0; }"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
               : "l"(a), "l"(b), "r"(1));
}

template <int HIDDEN>
__global__ void __launch_bounds__(kThreads, 1)
    project(long long* times, const __nv_bfloat16* x, const __nv_bfloat16* w, float* out, int* sinks, int jobs) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int kSteps = HIDDEN / kK;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t base = lab::smem(smem), full = base + kBarsAt, empty = full + 8 * kStages;
  if (tid == 0) {
    for (int s = 0; s < kStages; ++s) {
      lab::mbar_init(full + 8 * s, 32);
      lab::mbar_init(empty + 8 * s, kMath);
    }
  }
  __syncthreads();
  const int total = jobs * kSteps;
  lab::WarpSpan clock;
  clock.start();
  if (warp == kMath / 32) {
    //: THE LOADER: stage `i` into slot `i % 3` once the slot's readers released it
    const __nv_bfloat16* tile = x + static_cast<long long>(blockIdx.x) * kTokens * HIDDEN;
#pragma unroll 1
    for (int i = 0; i < total; ++i) {
      const int slot = i % kStages, job = i / kSteps, k = (i % kSteps) * kK;
      if (i >= kStages) lab::mbar_wait(empty + 8 * slot, ((i / kStages) - 1) & 1);
      const uint32_t at = base + slot * kStageBytes;
      lab::cp_async_bricks<kTokens>(at, tile + k, HIDDEN, lane);
      lab::cp_async_bricks<kCols>(at + kXBytes, w + static_cast<long long>(job) * kCols * HIDDEN + k, HIDDEN, lane);
      lab::mbar_arrive_when_landed(full + 8 * slot);
    }
  } else {
    //: A MATH WARPGROUP: its 64 tokens of the tile against every column of the job
    const int wg = warp >> 2;
    float d[128];
#pragma unroll
    for (int i = 0; i < 128; ++i) d[i] = 0.0f;
    float fold = 0.0f;
    //: a stage's steps: its FULL edge, the async-proxy fence (the copies wrote through the generic proxy), 4 steps
    const auto issue = [&](int i) {
      const int slot = i % kStages;
      lab::mbar_wait(full + 8 * slot, (i / kStages) & 1);
      lab::bulk_fence();
      const uint32_t at = base + slot * kStageBytes;
      lab::wg_fence();
#pragma unroll
      for (int s = 0; s < 4; ++s)
        step(d, lab::wg_desc(at + s * (kTokens * 32) + wg * 2048, 8, 16),
             lab::wg_desc(at + kXBytes + s * (kCols * 32), 8, 16));
      lab::wg_commit();
    };
#pragma unroll 1
    for (int job = 0; job < jobs; ++job) {
      const int first = job * kSteps;
      issue(first);
#pragma unroll 1
      for (int i = first + 1; i < first + kSteps; ++i) {
        issue(i);
        //: the previous stage's group has landed: its slot released, every thread an arrival
        lab::wg_wait<1>();
        lab::mbar_arrive(empty + 8 * ((i - 1) % kStages));
      }
      //: THE JOB'S END: every step landed, its last slot released, the accumulators folded, then cleared but the last
      lab::wg_wait<0>();
      lab::mbar_arrive(empty + 8 * ((first + kSteps - 1) % kStages));
      if (job + 1 < jobs) {
#pragma unroll
        for (int v = 0; v < 128; ++v) {
          fold += d[v];
          d[v] = 0.0f;
        }
      }
    }
    clock.stop(times);
    lab::sink(sinks, __float_as_uint(fold));
#pragma unroll
    for (int v = 0; v < 128; ++v) out[(static_cast<long long>(blockIdx.x) * kMath + tid) * 128 + v] = d[v];
    return;
  }
  clock.stop(times);
}

//: the operands: small integers, so every product and sum is exact in fp32
inline float x_of(long long r, int k) { return static_cast<float>((r * 7 + k * 3) % 5 - 2); }
inline float w_of(long long c, int k) { return static_cast<float>((c * 5 + k * 11) % 3 - 1); }

}  // namespace projection_tile

template <int HIDDEN>
void run(lab::Bench& b, const __nv_bfloat16* x, const __nv_bfloat16* w, float* out) {
  using namespace projection_tile;
  const auto kernel = project<HIDDEN>;
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  const std::string name = "hidden " + std::to_string(HIDDEN) + ", a job 128 tokens x 256 columns, three stages";
  //: THE CHECK: two CTAs, two jobs; the second job's accumulators against the host's products
  constexpr int kCheckCtas = 2, kCheckJobs = 2;
  kernel<<<kCheckCtas, kThreads, kSmem>>>(b.times, x, w, out, b.sinks, kCheckJobs);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<float> got(static_cast<size_t>(kCheckCtas) * kMath * 128);
  LAB_CHECK(cudaMemcpy(got.data(), out, got.size() * 4, cudaMemcpyDeviceToHost));
  long long wrong = 0;
  for (int cta = 0; cta < kCheckCtas; ++cta)
    for (int tid = 0; tid < kMath; ++tid) {
      const int wg = tid >> 7, wq = (tid >> 5) & 3, g = (tid & 31) >> 2, t = tid & 3;
      for (int v = 0; v < 128; ++v) {
        const int q = v >> 2, j = (v >> 1) & 1, i = v & 1;
        const long long r = static_cast<long long>(cta) * kTokens + wg * 64 + 16 * wq + g + 8 * j;
        const long long c = static_cast<long long>(kCheckJobs - 1) * kCols + 8 * q + 2 * t + i;
        float want = 0;
        for (int k = 0; k < HIDDEN; ++k) want += x_of(r, k) * w_of(c, k);
        wrong += got[(static_cast<size_t>(cta) * kMath + tid) * 128 + v] != want;
      }
    }
  b.check(name, wrong == 0, std::to_string(wrong) + " of " + std::to_string(got.size()) + " accumulators differ");
  lab::Samples floor_share;
  const lab::Samples cycles = b.variant([&] { kernel<<<b.sms, kThreads, kSmem>>>(b.times, x, w, out, b.sinks, kJobs); },
                                        [&] {
                                          const double job = b.span(kThreads >> 5) / kJobs;
                                          floor_share.v.push_back(job / (16.0 * HIDDEN));
                                          return job;
                                        });
  b.record(name + ": a job", "cycles a job a CTA", cycles);
  b.record(name + ": against the tensor floor", "the job's cycles over 16 x hidden", floor_share);
}

int main(int argc, char** argv) {
  using namespace projection_tile;
  lab::Bench b(argc, argv);
  for (int h : {256, 512, 2048}) {
    const std::string k = lab::kernel("project", {h});
    //: a job's first stage is issued before its loop, the rest in it: two copies of a stage's 4 steps
    b.expect(k, "HGMMA", "==", 8);
    //: the barriers' polled waits (`mbar_wait`'s loop: the loader's EMPTY and the warpgroups' FULL edges) are loops in
    //: the trips, their joins the convergence instructions counted
    b.expect(k, "LOOP_CONVERGENCE", "<=", 6);
    //: the FULL edge's polled wait is a branch in the step trip, before its steps
    b.expect(k, "BSSY", ">=", 0);
    b.expect(k, "LDGSTS", ">=", 96);
    b.exempt(k, "an isolation: the projection tile's ring and steps, its epilogue a fold");
  }
  if (b.expect_only) return 0;
  constexpr int kMaxHidden = 2048;
  const long long rows = static_cast<long long>(b.sms) * kTokens, cols = static_cast<long long>(kJobs) * kCols;
  __nv_bfloat16 *x = nullptr, *w = nullptr;
  float* out = nullptr;
  LAB_CHECK(cudaMalloc(&x, rows * kMaxHidden * 2));
  LAB_CHECK(cudaMalloc(&w, cols * kMaxHidden * 2));
  LAB_CHECK(cudaMalloc(&out, static_cast<size_t>(b.sms) * kMath * 128 * 4));
  const auto upload = [&](int hidden) {
    std::vector<__nv_bfloat16> hx(rows * hidden), hw(cols * hidden);
    for (long long r = 0; r < rows; ++r)
      for (int k = 0; k < hidden; ++k) hx[r * hidden + k] = __float2bfloat16(x_of(r, k));
    for (long long c = 0; c < cols; ++c)
      for (int k = 0; k < hidden; ++k) hw[c * hidden + k] = __float2bfloat16(w_of(c, k));
    LAB_CHECK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    LAB_CHECK(cudaMemcpy(w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
  };
  upload(256);
  run<256>(b, x, w, out);
  upload(512);
  run<512>(b, x, w, out);
  upload(2048);
  run<2048>(b, x, w, out);
  return 0;
}
