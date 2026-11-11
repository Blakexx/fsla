// measure/lab/experiments/cluster_sum.cu -- QUESTION: a token's output is a sum of partials that the CTAs of a
// cluster hold; what does a token cost a CTA when the cluster sums them in one CTA's shared memory before one drain
// to global memory, by each of four forms, beside every CTA draining its own? And does one collector hold a
// cluster's partials, or must the collecting be spread? METHOD: clusters of 8 CTAs, 4 warps each; a round is a
// partial of 64 tokens x 64 channels a CTA, from registers. `direct`: every CTA adds its partial to global memory.
// `slots`: every CTA's warps store its partial into its own slot of the collector's shared memory; the collector
// adds a token's 8 slots in registers and adds the sum to global memory. `atomic`: the warps add fp32 into the
// collector's one panel. `bulk slots`: the slot form with the partial staged in the CTA's own shared memory and
// sent by one bulk copy a collector. `bulk integers`: staged as 64-bit integers and sent by one bulk reduction into
// the collector's one panel. One collector (rank 0) or eight (a token's collector its index modulo 8). Barriers a
// round order them (the bulk forms' bytes complete the collector's), the cluster's own its start and end.
// see docs/internals/tools/lab.md#cluster-sum
#include "lab.cuh"

namespace cluster_sum {

constexpr int kCtas = 8, kThreads = 128, kTokens = 64, kChannels = 64, kPanel = kTokens * kChannels,
              kMine = kPanel / kThreads;
enum Form : int { kDirect = 0, kSlots, kAtomic, kBulkSlots, kBulkInteger, kForms };
const char* kFormName[kForms] = {"every CTA drains its own", "fp32 slots stored by the warps",
                                 "fp32 atomic adds into one panel", "fp32 slots by a bulk copy",
                                 "64-bit integers by a bulk reduction"};
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

//: the shared memory: the collector's panel (a slot a CTA for the slot forms; 8 bytes a value for the integers),
//: a bulk form's staged partial (a collector's elements together), then two barriers
template <int FORM>
constexpr bool kSlotted = FORM == kSlots || FORM == kBulkSlots;
template <int FORM>
constexpr bool kBulk = FORM == kBulkSlots || FORM == kBulkInteger;
template <int FORM>
constexpr int kElement = FORM == kBulkInteger ? 8 : 4;
template <int FORM, int SPREAD>
constexpr int kPanelBytes = kLocal<SPREAD> * kElement<FORM> * (kSlotted<FORM> ? kCtas : 1);
template <int FORM, int SPREAD>
constexpr int kStageBytes = kBulk<FORM> ? kPanel * kElement<FORM> : 0;
template <int FORM, int SPREAD>
constexpr int kSmemBytes = kPanelBytes<FORM, SPREAD> + kStageBytes<FORM, SPREAD> + 16;

template <int FORM, int SPREAD>
__global__ void __cluster_dims__(kCtas, 1, 1) sum(long long* times, float* out, int rounds) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x;
  const uint32_t rank = lab::cluster_rank();
  constexpr int kStageAt = kPanelBytes<FORM, SPREAD>, kBars = kStageAt + kStageBytes<FORM, SPREAD>;
  constexpr int kCollectors = SPREAD ? kCtas : 1, kShare = kLocal<SPREAD> * kElement<FORM>;
  const uint32_t base = lab::smem(smem), full = base + kBars, done = full + 8;
  float* const dst = out + static_cast<long long>(blockIdx.x / kCtas) * kPanel;
  const bool collects = SPREAD || rank == 0;

  //: THE PANEL, zero; the barriers: `full` an arrival a CTA (a bulk form: the collector's one, its bytes the
  //: copies'), `done` an arrival a collector
#pragma unroll 1
  for (int i = tid; i < kPanelBytes<FORM, SPREAD> / 4; i += kThreads) reinterpret_cast<uint32_t*>(smem)[i] = 0u;
  if (tid == 0) {
    lab::mbar_init(full, kBulk<FORM> ? 1 : kCtas);
    lab::mbar_init(done, kCollectors);
  }
  if constexpr (kBulk<FORM>) lab::bulk_fence();
  __syncthreads();
  //: a bulk form's collector states the round's bytes before any mate can send them
  if constexpr (kBulk<FORM>) {
    if (collects && tid == 0) lab::mbar_expect(full, kCtas * kShare);
  }
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
    //: THE PARTIAL OUT: this thread's values, each to its collector (a bulk form: to this CTA's stage)
#pragma unroll
    for (int i = 0; i < kMine; ++i) {
      const int e = i * kThreads + tid;
      if constexpr (FORM == kDirect) {
        lab::red_add_f32(dst + e, v[i]);
      } else {
        const int c = collector_of<SPREAD>(e), l = local_of<SPREAD>(e);
        if constexpr (FORM == kSlots) lab::cluster_store_f32(panel[c] + 4 * (rank * kLocal<SPREAD> + l), v[i]);
        if constexpr (FORM == kAtomic) lab::cluster_add_f32(panel[c] + 4 * l, v[i]);
        if constexpr (FORM == kBulkSlots) reinterpret_cast<float*>(smem + kStageAt)[c * kLocal<SPREAD> + l] = v[i];
        if constexpr (FORM == kBulkInteger)
          reinterpret_cast<long long*>(smem + kStageAt)[c * kLocal<SPREAD> + l] = static_cast<long long>(v[i] * kScale);
      }
    }
    if constexpr (FORM != kDirect) {
      if constexpr (kBulk<FORM>) lab::bulk_fence();
      __syncthreads();
      //: THE HAND-OVER: a thread a collector -- its share of the stage by one bulk copy, or an arrival
      if (tid < kCollectors) {
        //: the collector's addresses mapped here: a register array indexed by the thread would go to the stack
        const uint32_t to = SPREAD ? tid : 0, bar = lab::cluster_addr(full, to), there = lab::cluster_addr(base, to);
        if constexpr (FORM == kBulkSlots)
          lab::cluster_bulk_copy(there + rank * kShare, base + kStageAt + tid * kShare, kShare, bar);
        if constexpr (FORM == kBulkInteger)
          lab::cluster_bulk_add_u64(there, base + kStageAt + tid * kShare, kShare, bar);
        if constexpr (!kBulk<FORM>) lab::cluster_arrive(bar);
      }
      //: THE COLLECTOR: its elements summed, added to global memory, the panel cleared for the next round
      if (collects) {
        lab::mbar_wait(full, r & 1);
#pragma unroll 1
        for (int l = tid; l < kLocal<SPREAD>; l += kThreads) {
          float sum = 0.0f;
          if constexpr (kSlotted<FORM>) {
#pragma unroll
            for (int s = 0; s < kCtas; ++s) sum += reinterpret_cast<const float*>(smem)[s * kLocal<SPREAD> + l];
          }
          if constexpr (FORM == kAtomic) {
            sum = reinterpret_cast<const float*>(smem)[l];
            reinterpret_cast<float*>(smem)[l] = 0.0f;
          }
          if constexpr (FORM == kBulkInteger) {
            sum = static_cast<float>(reinterpret_cast<const long long*>(smem)[l]) / kScale;
            reinterpret_cast<long long*>(smem)[l] = 0;
          }
          lab::red_add_f32(dst + global_of<SPREAD>(rank, l), sum);
        }
        if constexpr (kBulk<FORM>) lab::bulk_fence();
        __syncthreads();
        if (tid == 0) {
          if constexpr (kBulk<FORM>) lab::mbar_expect(full, kCtas * kShare);
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
  run<kBulkSlots, 0>(b, out);
  run<kBulkSlots, 1>(b, out);
  run<kBulkInteger, 0>(b, out);
  run<kBulkInteger, 1>(b, out);
  return 0;
}
