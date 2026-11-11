// measure/lab/experiments/cluster_sum.cu -- QUESTION: a token's output is a sum of partials that the CTAs of a
// cluster hold; what does a token cost a CTA when the cluster sums them in one CTA's shared memory before one drain
// to global memory, by each of three forms, beside every CTA draining its own? And does one collector hold a
// cluster's partials, or must the collecting be spread? METHOD: clusters of 8 CTAs, 4 warps each; a round is a
// partial of 64 tokens x 64 channels a CTA, from registers. `direct`: every CTA adds its partial to global memory.
// `slots`: every CTA stores its partial into its own slot of the collector's shared memory; the collector adds a
// token's 8 slots in registers and adds the sum to global memory. `atomic`: every CTA's warps add fp32 into the
// collector's one panel. `s64`: the same with 64-bit integers, converted back by the collector. One collector (rank
// 0) or eight (a token's collector its index modulo 8). A barrier a round each way orders them, the cluster's own
// barrier its start and end; sums are exact.
// see docs/internals/tools/lab.md#cluster-sum
#include "lab.cuh"

namespace cluster_sum {

constexpr int kCtas = 8, kThreads = 128, kTokens = 64, kChannels = 64, kPanel = kTokens * kChannels,
              kMine = kPanel / kThreads;
enum Form : int { kDirect = 0, kSlots, kAtomic, kInteger, kForms };
const char* kFormName[kForms] = {"every CTA drains its own", "fp32 slots, the collector adds",
                                 "fp32 atomic adds into one panel", "64-bit integer adds into one panel"};
constexpr float kScale = 1048576.0f;

//: a CTA's value at panel element `e`, a small integer: every sum exact in fp32
constexpr int value_of(int rank, int e) { return e % 7 + 1 + rank; }

//: THE COLLECTING, spread or not: the CTA that collects element `e`, the element's index there, the elements a
//: collector holds
template <int SPREAD>
constexpr int collector_of(int e) { return SPREAD ? (e / kChannels) % kCtas : 0; }
template <int SPREAD>
constexpr int local_of(int e) { return SPREAD ? (e / kChannels / kCtas) * kChannels + e % kChannels : e; }
template <int SPREAD>
constexpr int kLocal = SPREAD ? kPanel / kCtas : kPanel;
template <int SPREAD>
constexpr int global_of(int rank, int l) {
  return SPREAD ? (kCtas * (l / kChannels) + rank) * kChannels + l % kChannels : l;
}

//: the shared memory: the collector's panel (a slot a CTA for `slots`; 8 bytes a value for `s64`), then two barriers
template <int FORM, int SPREAD>
constexpr int kPanelBytes = kLocal<SPREAD> * (FORM == kSlots ? 4 * kCtas : FORM == kInteger ? 8 : 4);
template <int FORM, int SPREAD>
constexpr int kSmemBytes = kPanelBytes<FORM, SPREAD> + 16;

template <int FORM, int SPREAD>
__global__ void __cluster_dims__(kCtas, 1, 1) sum(long long* times, float* out, int rounds) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x;
  const uint32_t rank = lab::cluster_rank();
  const uint32_t base = lab::smem(smem), full = base + kPanelBytes<FORM, SPREAD>, done = full + 8;
  float* const dst = out + static_cast<long long>(blockIdx.x / kCtas) * kPanel;
  constexpr int kCollectors = SPREAD ? kCtas : 1;
  const bool collects = SPREAD || rank == 0;

  //: THE PANEL, zero; the barriers: `full` an arrival a CTA, `done` an arrival a collector
#pragma unroll 1
  for (int i = tid; i < kPanelBytes<FORM, SPREAD> / 4; i += kThreads) reinterpret_cast<uint32_t*>(smem)[i] = 0u;
  if (tid == 0) {
    lab::mbar_init(full, kCtas);
    lab::mbar_init(done, kCollectors);
  }
  __syncthreads();
  //: every CTA of the cluster has its panel and barriers before any reaches into another's
  lab::cluster_sync();
  //: every collector's panel, as this CTA reaches it
  uint32_t panel[kCollectors];
#pragma unroll
  for (int c = 0; c < kCollectors; ++c) panel[c] = lab::cluster_addr(base, SPREAD ? c : 0);
  float v[kMine];
#pragma unroll
  for (int i = 0; i < kMine; ++i) v[i] = static_cast<float>(value_of(rank, i * kThreads + tid));

  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
    //: THE PARTIAL OUT: this thread's values, each to its collector
#pragma unroll
    for (int i = 0; i < kMine; ++i) {
      const int e = i * kThreads + tid;
      if constexpr (FORM == kDirect) {
        lab::red_add_f32(dst + e, v[i]);
      } else {
        const int c = collector_of<SPREAD>(e), l = local_of<SPREAD>(e);
        if constexpr (FORM == kSlots) lab::cluster_store_f32(panel[c] + 4 * (rank * kLocal<SPREAD> + l), v[i]);
        if constexpr (FORM == kAtomic) lab::cluster_add_f32(panel[c] + 4 * l, v[i]);
        if constexpr (FORM == kInteger) lab::cluster_add_s64(panel[c] + 8 * l, static_cast<long long>(v[i] * kScale));
      }
    }
    if constexpr (FORM != kDirect) {
      __syncthreads();
      if (tid == 0) {
#pragma unroll
        for (int c = 0; c < kCollectors; ++c) lab::cluster_arrive(lab::cluster_addr(full, SPREAD ? c : 0));
      }
      //: THE COLLECTOR: its elements summed, added to global memory, the panel cleared for the next round
      if (collects) {
        lab::mbar_wait(full, r & 1);
#pragma unroll 1
        for (int l = tid; l < kLocal<SPREAD>; l += kThreads) {
          float sum = 0.0f;
          if constexpr (FORM == kSlots) {
#pragma unroll
            for (int s = 0; s < kCtas; ++s) sum += reinterpret_cast<const float*>(smem)[s * kLocal<SPREAD> + l];
          }
          if constexpr (FORM == kAtomic) {
            sum = reinterpret_cast<const float*>(smem)[l];
            reinterpret_cast<float*>(smem)[l] = 0.0f;
          }
          if constexpr (FORM == kInteger) {
            sum = static_cast<float>(reinterpret_cast<const long long*>(smem)[l]) / kScale;
            reinterpret_cast<long long*>(smem)[l] = 0;
          }
          lab::red_add_f32(dst + global_of<SPREAD>(rank, l), sum);
        }
        __syncthreads();
        if (tid == 0) {
#pragma unroll
          for (int s = 0; s < kCtas; ++s) lab::cluster_arrive(lab::cluster_addr(done, s));
        }
      }
      lab::mbar_wait(done, r & 1);
    }
  }
  clock.stop(times);
  //: no CTA leaves while a mate may still arrive at its barrier
  lab::cluster_sync();
}

}  // namespace cluster_sum

template <int FORM, int SPREAD>
void run(lab::Bench& b, float* out) {
  using namespace cluster_sum;
  constexpr int kCheckRounds = 16, kRounds = 1024;
  const int clusters = b.sms / kCtas, blocks = clusters * kCtas;
  const std::string name = std::string(kFormName[FORM])
                           + (FORM == kDirect ? "" : SPREAD ? ", eight collectors" : ", one collector");
  constexpr int bytes = kSmemBytes<FORM, SPREAD>;
  LAB_CHECK(cudaFuncSetAttribute(sum<FORM, SPREAD>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  const size_t words = static_cast<size_t>(clusters) * kPanel;
  LAB_CHECK(cudaMemset(out, 0, words * sizeof(float)));
  sum<FORM, SPREAD><<<blocks, kThreads, bytes>>>(b.times, out, kCheckRounds);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<float> got(words);
  LAB_CHECK(cudaMemcpy(got.data(), out, words * sizeof(float), cudaMemcpyDeviceToHost));
  long long wrong = 0;
  for (size_t i = 0; i < words; ++i) {
    const int e = static_cast<int>(i % kPanel);
    int want = 0;
    for (int rank = 0; rank < kCtas; ++rank) want += value_of(rank, e);
    wrong += got[i] != static_cast<float>(want * kCheckRounds);
  }
  b.check(name, wrong == 0,
          std::to_string(wrong) + " of " + std::to_string(words) + " sums differ from the exact ones at "
              + std::to_string(kCheckRounds) + " rounds");
  const lab::Samples cycles = b.variant(
      [&] { sum<FORM, SPREAD><<<blocks, kThreads, bytes>>>(b.times, out, kRounds); },
      [&] { return b.span(kThreads >> 5, blocks) / kRounds / kTokens; });
  b.record(name, "cycles a token a CTA", cycles);
}

int main(int argc, char** argv) {
  using namespace cluster_sum;
  lab::Bench b(argc, argv);
  if (b.expect_only) return 0;
  float* out = nullptr;
  LAB_CHECK(cudaMalloc(&out, sizeof(float) * (b.sms / kCtas) * kPanel));
  run<kDirect, 0>(b, out);
  run<kSlots, 0>(b, out);
  run<kSlots, 1>(b, out);
  run<kAtomic, 0>(b, out);
  run<kAtomic, 1>(b, out);
  run<kInteger, 0>(b, out);
  run<kInteger, 1>(b, out);
  return 0;
}
