// measure/lab/experiments/step_queue.cu -- QUESTION: the carry's fold issues its steps as a chain -- a step's two
// operands loaded from a shared table, moved to the uniform registers, the step -- and beside the read side's warps
// a step costs some 58 cycles where alone it costs 48 and held 15 (`carry_window`). How does a step's time grow with
// each kind of memory traffic other warps put on the SM beside it, and with its rate? METHOD: a CTA an SM; one fold
// warpgroup issues trips of 16 m64n16k16 steps into 16 accumulators, a step's operands one 16-byte line of a shared
// table (LOADED) or one line held in registers (HELD); beside it 0, 1, 2 or 4 co-runner warps issue one kind of
// traffic back to back until the fold is done: SHARED LOADS (16 bytes a lane), SHARED STORES (16 bytes a lane), GLOBAL
// ADDS (an f16 pair a lane, a line an instruction, as a drain warp adds) or GLOBAL LOADS (16 bytes a lane, of lines
// the cache holds). Recorded: the fold's cycles a step, and the co-runners' instructions a hundred cycles. The fold's
// sums are checked exactly: 16 a step in every value.
// TIMELINE: the fold: a trip is the fence, 16 steps -- each its line loaded at its own issue behind a compiler edge,
// as the carry's fold does, or its held line -- the commit and the wait; its tiles of ones written before the clock.
// A co-runner warp: 16 independent instructions a trip to its own addresses, its sink folded; every trip it reads the
// fold's done flag. Live: the fold's 16 accumulators (128 registers a thread) and a line; a co-runner's sink and a
// line.
// see docs/lab/step_queue.md
#include "lab.cuh"

namespace step_queue {

enum Kind : int { kNone = 0, kSharedLoads, kSharedStores, kGlobalAdds, kGlobalLoads };
constexpr int kFold = 128, kSteps = 16, kTileA = 0, kTileB = 2048, kTableAt = 2560, kFlagAt = kTableAt + kSteps * 16,
              kCoAt = kFlagAt + 128, kCoBytes = 4096, kSmem = kCoAt + 4 * kCoBytes;
//: a block's clock words: the fold's start and end, then a co-runner warp's instructions and cycles
constexpr int kWords = 16;

template <int HELD, int KIND, int WARPS>
__global__ void __launch_bounds__(kFold + 128, 1) queue(long long* times, float* out, uint32_t* lines, int trips) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31;
  const uint32_t base = lab::smem(smem);
#pragma unroll 1
  for (int i = tid; i < kTableAt / 2; i += blockDim.x) reinterpret_cast<uint16_t*>(smem)[i] = 0x3f80;
  //: the table: every step's line the same two operands, the tiles of ones
  if (tid < kSteps) {
    uint64_t* const line = reinterpret_cast<uint64_t*>(smem + kTableAt + 16 * tid);
    line[0] = lab::wg_desc(base + kTileA, 8, 16);
    line[1] = lab::wg_desc(base + kTileB, 8, 16);
  }
  if (tid == 0) *reinterpret_cast<volatile uint32_t*>(smem + kFlagAt) = 0u;
  lab::bulk_fence();
  __syncthreads();
  long long* const clock = times + static_cast<long long>(blockIdx.x) * kWords;

  if (tid < kFold) {
    //: THE FOLD
    float d[kSteps][8] = {};
    const uint4* const table = reinterpret_cast<const uint4*>(smem + kTableAt);
    const uint4 held = HELD ? table[0] : make_uint4(0, 0, 0, 0);
    const long long t0 = clock64();
#pragma unroll 1
    for (int t = 0; t < trips; ++t) {
      lab::wg_fence();
#pragma unroll
      for (int s = 0; s < kSteps; ++s) {
        asm volatile("" ::: "memory");
        const uint4 two = HELD ? held : table[s];
        lab::wg_bf16_n16_ss_mn(d[s], static_cast<uint64_t>(two.y) << 32 | two.x,
                               static_cast<uint64_t>(two.w) << 32 | two.z, true);
      }
      lab::wg_commit();
      lab::wg_wait<0>();
    }
#pragma unroll
    for (int s = 0; s < kSteps; ++s)
#pragma unroll
      for (int i = 0; i < 8; ++i) lab::wg_hold(d[s][i]);
    const long long t1 = clock64();
    __syncwarp();
    if (tid == 0) {
      clock[0] = t0;
      clock[1] = t1;
      *reinterpret_cast<volatile uint32_t*>(smem + kFlagAt) = 1u;
    }
    //: every value is 16 a step: their largest and smallest, a thread
    float lo = d[0][0], hi = d[0][0];
#pragma unroll
    for (int s = 0; s < kSteps; ++s)
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        lo = fminf(lo, d[s][i]);
        hi = fmaxf(hi, d[s][i]);
      }
    out[2 * (blockIdx.x * kFold + tid)] = lo;
    out[2 * (blockIdx.x * kFold + tid) + 1] = hi;
  } else if (KIND != kNone && tid < kFold + 32 * WARPS) {
    //: A CO-RUNNER WARP: 16 independent instructions a trip to its own addresses, until the fold is done
    const int warp = (tid - kFold) >> 5;
    uint8_t* const own = smem + kCoAt + warp * kCoBytes;
    uint32_t* const glob = lines + (static_cast<long long>(blockIdx.x) * 4 + warp) * (64 * 32);
    const uint32_t one = lab::held(0x3c003c00u);
    uint32_t sink = 0;
    long long issued = 0;
    const long long t0 = clock64();
#pragma unroll 1
    while (*reinterpret_cast<volatile const uint32_t*>(smem + kFlagAt) == 0u) {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        //: each one PTX instruction the toolkit can neither merge nor drop
        const uint32_t at = lab::smem(own) + ((i * 256 + lane * 16) & (kCoBytes - 1));
        uint32_t x = 0, y = 0, z = 0, v = 0;
        if constexpr (KIND == kSharedLoads) {
          asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(v) : "r"(at));
          sink ^= x ^ y ^ z ^ v;
        } else if constexpr (KIND == kSharedStores) {
          asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(at), "r"(one), "r"(sink), "r"(one), "r"(sink) : "memory");
        } else if constexpr (KIND == kGlobalAdds) {
          lab::red_add_f16x2(glob + (static_cast<int>(issued & 3) * 16 + i) * 32 + lane, one);
        } else {
          const uint32_t* const from = glob + (((static_cast<int>(issued & 3) * 16 + i) * 32 + lane * 4) & (64 * 32 - 4));
          asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(v) : "l"(from));
          sink ^= x ^ y ^ z ^ v;
        }
      }
      issued += 16;
    }
    const long long t1 = clock64();
    if (lane == 0) {
      clock[2 + 2 * warp] = issued;
      clock[3 + 2 * warp] = t1 - t0;
    }
    lab::sink(reinterpret_cast<int*>(lines) + static_cast<long long>(gridDim.x) * 4 * 64 * 32, sink);
  }
}

}  // namespace step_queue

template <int HELD, int KIND, int WARPS>
void run(lab::Bench& b, float* out, uint32_t* lines) {
  using namespace step_queue;
  constexpr int kCheckTrips = 16, kTrips = 2048;
  static const char* const kinds[] = {"nothing", "shared loads", "shared stores", "global adds", "global loads"};
  const std::string name = std::string(HELD ? "held operands" : "loaded operands") + ", beside "
                           + (KIND == kNone ? std::string("nothing")
                                            : std::to_string(WARPS) + (WARPS == 1 ? " warp" : " warps") + " of " + kinds[KIND]);
  const auto kernel = queue<HELD, KIND, WARPS>;
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  const int threads = kFold + 128;
  kernel<<<b.sms, threads, kSmem>>>(b.times, out, lines, kCheckTrips);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<float> got(static_cast<size_t>(b.sms) * kFold * 2);
  LAB_CHECK(cudaMemcpy(got.data(), out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
  long long wrong = 0;
  for (float v : got) wrong += v != 16.0f * kCheckTrips;
  b.check(name, wrong == 0, std::to_string(wrong) + " of " + std::to_string(got.size()) + " extremes differ from the exact sum");
  lab::Samples rate;
  const auto words = [&] {
    std::vector<long long> h(static_cast<size_t>(b.sms) * kWords);
    LAB_CHECK(cudaMemcpy(h.data(), b.times, h.size() * sizeof(long long), cudaMemcpyDeviceToHost));
    return h;
  };
  const lab::Samples cycles = b.variant([&] { kernel<<<b.sms, threads, kSmem>>>(b.times, out, lines, kTrips); },
                                        [&] {
                                          const std::vector<long long> h = words();
                                          double step = 0, per = 0;
                                          for (int blk = 0; blk < b.sms; ++blk) {
                                            step += double(h[blk * kWords + 1] - h[blk * kWords]);
                                            for (int w = 0; w < WARPS && KIND != kNone; ++w)
                                              per += 100.0 * double(h[blk * kWords + 2 + 2 * w]) / double(h[blk * kWords + 3 + 2 * w]);
                                          }
                                          rate.v.push_back(per / b.sms);
                                          return step / (double(b.sms) * kTrips * kSteps);
                                        });
  b.record(name, "cycles a step", cycles);
  if (KIND != kNone) {
    b.unchecked(name + ": the co-runners' rate", "the co-runners' instructions count themselves: the row checks the fold");
    b.record(name + ": the co-runners' rate", "instructions a hundred cycles a CTA", rate);
  }
}

template <int HELD>
void sweep(lab::Bench& b, float* out, uint32_t* lines) {
  using namespace step_queue;
  run<HELD, kNone, 1>(b, out, lines);
  run<HELD, kSharedLoads, 1>(b, out, lines);
  run<HELD, kSharedLoads, 2>(b, out, lines);
  run<HELD, kSharedLoads, 4>(b, out, lines);
  run<HELD, kSharedStores, 1>(b, out, lines);
  run<HELD, kSharedStores, 2>(b, out, lines);
  run<HELD, kSharedStores, 4>(b, out, lines);
  run<HELD, kGlobalAdds, 1>(b, out, lines);
  run<HELD, kGlobalAdds, 2>(b, out, lines);
  run<HELD, kGlobalAdds, 4>(b, out, lines);
  run<HELD, kGlobalLoads, 1>(b, out, lines);
  run<HELD, kGlobalLoads, 2>(b, out, lines);
  run<HELD, kGlobalLoads, 4>(b, out, lines);
}

int main(int argc, char** argv) {
  using namespace step_queue;
  lab::Bench b(argc, argv);
  for (int held : {0, 1})
    for (int kind : {kNone, kSharedLoads, kSharedStores, kGlobalAdds, kGlobalLoads})
      for (int warps : {1, 2, 4}) {
        if (kind == kNone && warps != 1) continue;
        const std::string k = lab::kernel("queue", {held, kind, warps});
        b.expect(k, "HGMMA", ">=", kSteps);
        b.exempt(k, "a rate probe: a fold step's time beside other warps' traffic, no kernel's step");
        if (kind == kGlobalAdds) b.expect(k, "RED", ">=", 16);
        //: the global-load co-runner folds each load's words in its own trip, by design: it is traffic, not a stream
        if (kind == kGlobalLoads) b.expect(k, "LOAD_IN_TRIP", "<=", 16);
      }
  if (b.expect_only) return 0;
  float* out = nullptr;
  uint32_t* lines = nullptr;
  LAB_CHECK(cudaMalloc(&out, sizeof(float) * b.sms * kFold * 2));
  LAB_CHECK(cudaMalloc(&lines, sizeof(uint32_t) * (static_cast<size_t>(b.sms) * 4 * 64 * 32 + static_cast<size_t>(b.sms) * (kFold + 128))));
  LAB_CHECK(cudaMemset(lines, 0, sizeof(uint32_t) * static_cast<size_t>(b.sms) * 4 * 64 * 32));
  sweep<0>(b, out, lines);
  sweep<1>(b, out, lines);
  return 0;
}
