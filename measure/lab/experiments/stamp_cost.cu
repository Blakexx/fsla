// measure/lab/experiments/stamp_cost.cu -- QUESTION: what does one timeline stamp (the SM clock's low word beside an
// event's id, written by a warp) cost the warp that takes it -- in cycles, in registers and in issued instructions --
// and which form of it costs least? METHOD: one warpgroup a CTA, a CTA an SM; a step is either one m64n64k16 wgmma
// with its commit and wait (a step that mostly waits), four dependent integer operations each with a shared load
// (a step bound by its chain's latency), or eight independent integer operations (a step bound by issue); the step
// runs plain, then with one or two stamps a step in four forms -- the lab's own
// `Trace` (its first lane's predicated store to the warp's row in global memory); a ring in shared memory written by one lane under a held
// predicate; the same with the predicate folded into the ring offset's low bit (one register); the one-lane stamp's
// store alone, on the integer steps; and the leanest stamp -- every lane storing the same words to ONE entry, no
// predicate and no wrap (a ring sized for its run). A stamp's cost is a form's cycles a step less the
// plain step's, over its stamps a step. Every kernel's work is checked against the host's, and every ring form's
// stamps are read back: their ids in order, their clocks not decreasing.
// TIMELINE: one warpgroup, no other role. A step: its first stamp (the clock read, then its store), the step's work
// -- the fence, one wgmma from held A and a held descriptor, the commit and the wait; or four dependent operations,
// each a shared load of the table at an address its predecessor computed -- then its second stamp. No load is a stage
// ahead of its use: the chained step is its own latency, and the independent step's eight values wait on nothing. Live: the accumulator (32 registers a thread), A (4),
// the descriptor, the chain's value, and the form's own -- a ring offset, a held predicate, or the trace's row, count and predicate.
// see docs/internals/tools/lab.md#stamp-cost
#include "lab.cuh"

namespace stamp_cost {

constexpr int kM = 64, kK = 16, kN = 64, kThreads = 128, kWarps = kThreads / 32;
constexpr int kTileA = 0, kTileB = 2048, kTable = 4096, kTableWords = 256;
//: the rings: a warp's is `kWarpRing` bytes at a multiple of it, so an offset wraps inside its own ring by a mask
constexpr int kRing = 8192, kWarpRing = 8192, kRingBytes = kWarps * kWarpRing, kSmem = kRing + kRingBytes;
constexpr uint32_t kMask = kWarpRing - 1;
//: an entry's bytes: one lane's (clock, id)
constexpr int kEntry = 8;
//: kStoreAlone is the one-lane stamp's store (a constant for the clock) with no read
enum Form { kPlain = 0, kTrace = 1, kHeldPredicate = 2, kFoldedPredicate = 3, kStoreAlone = 4, kLean = 5 };
//: the CS2R a plain kernel holds, by step: its span's two clock reads, and on the wgmma step 16 more -- the zeroing
//: of its 32 accumulator registers, a CS2R of SRZ a pair, before the loop. A form's kernel holds these and one a stamp site
constexpr int kPlainCs2r[3] = {18, 2, 2};

constexpr int read_a(int m, int k) { return (m + k) % 7 - 3; }
constexpr int read_b(int k, int n) { return (k + 2 * n) % 3 - 1; }
constexpr int tile_at(int n, int k) { return 16 * ((n % 8) + 16 * (n / 8) + 8 * (k / 8)) + 2 * (k % 8); }
constexpr uint32_t table_at(int i) { return static_cast<uint32_t>(i) * 2654435761u; }
constexpr uint32_t next(uint32_t x, uint32_t loaded) { return x * 1664525u + 1013904223u + loaded; }
//: the independent step's values a thread
constexpr int kWide = 8;

__device__ __forceinline__ uint16_t bf16_of(int v) {
  return static_cast<uint16_t>(__float_as_uint(static_cast<float>(v)) >> 16);
}

//: THE STAMP, in the form under test. `off` is the next entry's byte offset in the CTA's rings.
template <int FORM>
struct Stamps {
  lab::Trace<FORM == kTrace> trace;
  uint32_t off = 0, mine = 0;

  __device__ __forceinline__ void bind(unsigned long long* rows) {
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if constexpr (FORM == kTrace) trace.bind(rows);
    if constexpr (FORM == kHeldPredicate || FORM == kStoreAlone) {
      off = warp * kWarpRing;
      mine = lane == 0;
    }
    if constexpr (FORM == kFoldedPredicate) off = warp * kWarpRing | (lane == 0 ? 1u : 0u);
    if constexpr (FORM == kLean) off = warp * kWarpRing;
  }

  __device__ __forceinline__ void stamp(uint32_t ring_at, uint32_t id) {
    if constexpr (FORM == kTrace) trace.stamp(lab::kPhase, id);
    if constexpr (FORM == kLean) {
      uint32_t now;
      asm volatile("mov.u32 %0, %%clock;" : "=r"(now));
      asm volatile("st.shared.v2.u32 [%0], {%1, %2};" :: "r"(ring_at + off), "r"(now), "r"(id) : "memory");
      off += kEntry;
    }
    if constexpr (FORM == kHeldPredicate || FORM == kFoldedPredicate || FORM == kStoreAlone) {
      uint32_t now = 0;
      if constexpr (FORM != kStoreAlone) asm volatile("mov.u32 %0, %%clock;" : "=r"(now));
      const uint32_t on = FORM == kFoldedPredicate ? off & 1u : mine;
      const uint32_t at = ring_at + (off & ~7u);
      asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %3, 0;\n @p st.shared.v2.u32 [%0], {%1, %2};\n}"
                   :: "r"(at), "r"(now), "r"(id), "r"(on) : "memory");
      off = (off & ~kMask) | ((off + kEntry) & kMask);
    }
  }

  __device__ __forceinline__ void close(uint32_t base) {
    if constexpr (FORM == kTrace) trace.close(base);
  }
};

template <int FORM, int PER, int WORK>
__global__ void __launch_bounds__(kThreads, 1) cost(long long* times, uint32_t* out, unsigned long long* rows,
                                                    uint32_t* rings, int steps) {
  __shared__ alignas(128) uint8_t smem[kSmem];
  const int tid = threadIdx.x, lane = tid & 31;
  auto* tile = reinterpret_cast<uint16_t*>(smem);
  auto* table = reinterpret_cast<uint32_t*>(smem + kTable);
  auto* ring_words = reinterpret_cast<uint32_t*>(smem + kRing);
#pragma unroll 1
  for (int i = tid; i < kM * kK; i += blockDim.x) tile[(kTileA + tile_at(i / kK, i % kK)) / 2] = bf16_of(read_a(i / kK, i % kK));
#pragma unroll 1
  for (int i = tid; i < kN * kK; i += blockDim.x) tile[(kTileB + tile_at(i / kK, i % kK)) / 2] = bf16_of(read_b(i % kK, i / kK));
#pragma unroll 1
  for (int i = tid; i < kTableWords; i += blockDim.x) table[i] = table_at(i);
#pragma unroll 1
  for (int i = tid; i < kRingBytes / 4; i += blockDim.x) ring_words[i] = 0;
  __syncthreads();
  const uint32_t base = lab::smem(smem);
  const uint64_t db = lab::held(lab::wg_desc(base + kTileB, 8, 16));
  uint32_t a[4];
  const int g = 16 * ((tid >> 5) & 3) + lane / 4, kt = lane & 3;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    const int m = g + 8 * (w & 1), k = 8 * (w >> 1) + 2 * kt;
    a[w] = lab::held(bf16_of(read_a(m, k)) | static_cast<uint32_t>(bf16_of(read_a(m, k + 1))) << 16);
  }
  float d[32] = {};
  uint32_t x = lab::held(static_cast<uint32_t>(tid) + 1u);
  uint32_t y[kWide];
#pragma unroll
  for (int j = 0; j < kWide; ++j) y[j] = lab::held(static_cast<uint32_t>(tid) + 1u + 257u * j);
  Stamps<FORM> stamps;
  stamps.bind(rows);

  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int it = 0; it < steps; ++it) {
    if constexpr (PER >= 1) stamps.stamp(base + kRing, 1);
    if constexpr (WORK == 0) {
      lab::wg_fence();
      lab::wg_bf16_n64_rs(d, a, db);
      lab::wg_commit();
      lab::wg_wait<0>();
    } else if constexpr (WORK == 1) {
#pragma unroll
      for (int k = 0; k < 4; ++k) x = next(x, table[(x >> 10) & (kTableWords - 1)]);
    } else {
#pragma unroll
      for (int j = 0; j < kWide; ++j) y[j] = next(y[j], static_cast<uint32_t>(j));
    }
    if constexpr (PER >= 2) stamps.stamp(base + kRing, 2);
  }
#pragma unroll
  for (int i = 0; i < 32; ++i) lab::wg_hold(d[i]);
  clock.stop(times);
  stamps.close(base);
  uint32_t* mine = out + (static_cast<long long>(blockIdx.x) * blockDim.x + tid) * 33;
#pragma unroll
  for (int j = 0; j < kWide; ++j) x ^= WORK == 2 ? y[j] : 0u;
  mine[0] = x;
#pragma unroll
  for (int i = 0; i < 32; ++i) mine[1 + i] = __float_as_uint(d[i]);
  __syncthreads();
#pragma unroll 1
  for (int i = tid; i < kRingBytes / 4; i += blockDim.x)
    rings[static_cast<long long>(blockIdx.x) * (kRingBytes / 4) + i] = ring_words[i];
}

}  // namespace stamp_cost

template <int FORM, int PER, int WORK>
void run(lab::Bench& b, uint32_t* out, unsigned long long* rows, uint32_t* rings) {
  using namespace stamp_cost;
  //: the timed run's stamps fit a warp's ring, which the lean form never wraps: 2 stamps x 512 steps x 8 bytes
  constexpr int kCheckSteps = 8, kSteps = kWarpRing / (2 * kEntry);
  static const char* const kForms[] = {"plain", "the lab's trace", "a ring, one lane, its predicate held",
                                       "a ring, one lane, its predicate in the offset", "the ring store alone",
                                       "a ring, every lane to one entry, no wrap"};
  const std::string name = std::string(WORK == 0 ? "a wgmma step" : WORK == 1 ? "a chained step" : "an independent step") + ", " + kForms[FORM]
                           + (PER == 0 ? "" : PER == 1 ? ", one stamp a step" : ", two stamps a step");
  cost<FORM, PER, WORK><<<b.sms, kThreads>>>(b.times, out, rows, rings, kCheckSteps);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> got(static_cast<size_t>(b.sms) * kThreads * 33), ring(static_cast<size_t>(b.sms) * kRingBytes / 4);
  LAB_CHECK(cudaMemcpy(got.data(), out, got.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  LAB_CHECK(cudaMemcpy(ring.data(), rings, ring.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  std::vector<double> d(kM * kN, 0.0);
  for (int m = 0; m < kM; ++m)
    for (int n = 0; n < kN; ++n)
      for (int k = 0; k < kK; ++k) d[m * kN + n] += kCheckSteps * read_a(m, k) * read_b(k, n);
  long long wrong = 0, checked = 0, bad_stamps = 0, stamps = 0;
  for (int blk = 0; blk < b.sms; ++blk) {
    for (int tid = 0; tid < kThreads; ++tid) {
      const uint32_t* mine = got.data() + (static_cast<size_t>(blk) * kThreads + tid) * 33;
      uint32_t x = static_cast<uint32_t>(tid) + 1u;
      if (WORK == 1)
        for (int i = 0; i < 4 * kCheckSteps; ++i) x = next(x, table_at((x >> 10) & (kTableWords - 1)));
      if (WORK == 2)
        for (int j = 0; j < kWide; ++j) {
          uint32_t y = static_cast<uint32_t>(tid) + 1u + 257u * j;
          for (int i = 0; i < kCheckSteps; ++i) y = next(y, static_cast<uint32_t>(j));
          x ^= y;
        }
      wrong += mine[0] != x;
      ++checked;
      const int g = 16 * ((tid >> 5) & 3) + (tid & 31) / 4, t = tid & 3;
      for (int j = 0; j < 32; ++j, ++checked) {
        const int m = g + 8 * ((j >> 1) & 1), n = 8 * (j >> 2) + 2 * t + (j & 1);
        float f;
        std::memcpy(&f, &mine[1 + j], sizeof f);
        wrong += f != (WORK == 0 ? static_cast<float>(d[m * kN + n]) : 0.0f);
      }
    }
    //: a ring form's stamps of each warp (its lane 0's), in order: the ids as stamped, the clocks not decreasing
    if (FORM == kHeldPredicate || FORM == kFoldedPredicate || FORM == kStoreAlone || FORM == kLean)
      for (int warp = 0; warp < kWarps; ++warp) {
        uint32_t before = 0;
        for (int e = 0; e < PER * kCheckSteps; ++e, ++stamps) {
          const size_t at = static_cast<size_t>(blk) * (kRingBytes / 4)
                            + (warp * kWarpRing + e * kEntry) / 4;
          const uint32_t now = ring[at], id = ring[at + 1];
          bad_stamps += id != static_cast<uint32_t>(e % (PER > 0 ? PER : 1) + 1) || (e > 0 && static_cast<int32_t>(now - before) < 0);
          before = now;
        }
      }
  }
  b.check(name, wrong == 0 && bad_stamps == 0,
          std::to_string(wrong) + " of " + std::to_string(checked) + " values differ from the host's at "
              + std::to_string(kCheckSteps) + " steps; " + std::to_string(bad_stamps) + " of " + std::to_string(stamps)
              + " stamps read back out of order or misnamed");
  const lab::Samples cycles = b.variant([&] { cost<FORM, PER, WORK><<<b.sms, kThreads>>>(b.times, out, rows, rings, kSteps); },
                                        [&] { return b.span(kWarps) / kSteps; });
  b.record(name, "cycles a step", cycles);
}

template <int WORK>
void work(lab::Bench& b, uint32_t* out, unsigned long long* rows, uint32_t* rings) {
  using namespace stamp_cost;
  run<kPlain, 0, WORK>(b, out, rows, rings);
  run<kTrace, 1, WORK>(b, out, rows, rings);
  run<kTrace, 2, WORK>(b, out, rows, rings);
  run<kHeldPredicate, 1, WORK>(b, out, rows, rings);
  run<kHeldPredicate, 2, WORK>(b, out, rows, rings);
  run<kFoldedPredicate, 1, WORK>(b, out, rows, rings);
  run<kFoldedPredicate, 2, WORK>(b, out, rows, rings);
  run<kLean, 1, WORK>(b, out, rows, rings);
  run<kLean, 2, WORK>(b, out, rows, rings);
  //: the store alone runs on the two integer steps
  if constexpr (WORK != 0) {
    run<kStoreAlone, 1, WORK>(b, out, rows, rings);
    run<kStoreAlone, 2, WORK>(b, out, rows, rings);
  }
}

int main(int argc, char** argv) {
  using namespace stamp_cost;
  lab::Bench b(argc, argv);
  for (int work_kind = 0; work_kind < 3; ++work_kind)
    for (int form = kPlain; form <= kLean; ++form)
      for (int per = form == kPlain ? 0 : 1; per <= (form == kPlain ? 0 : 2); ++per) {
        if (form == kStoreAlone && work_kind == 0) continue;
        const std::string k = lab::kernel("cost", {form, per, work_kind});
        b.expect(k, "HGMMA", "==", work_kind == 0 ? 1 : 0);
        //: the trace's stamp is a global store inside the step: the form under test
        if (form == kTrace && work_kind == 0) b.expect(k, "STEP_GLOBAL", ">=", 0);
        //: a clock read a stamp site and no more: a loop the compiler unrolled or a read it dropped is refused
        b.expect(k, "CS2R", form == kTrace ? ">=" : "==", kPlainCs2r[work_kind] + (form == kPlain || form == kStoreAlone ? 0 : per));
        b.exempt(k, "a probe of the recorder's own cost: no kernel's step");
      }
  if (b.expect_only) return 0;
  uint32_t *out = nullptr, *rings = nullptr;
  unsigned long long* rows = nullptr;
  LAB_CHECK(cudaMalloc(&out, sizeof(uint32_t) * b.sms * kThreads * 33));
  LAB_CHECK(cudaMalloc(&rings, sizeof(uint32_t) * b.sms * (kRingBytes / 4)));
  LAB_CHECK(cudaMalloc(&rows, sizeof(unsigned long long) * b.sms * kWarps * lab::Trace<true>::kRow));
  work<0>(b, out, rows, rings);
  work<1>(b, out, rows, rings);
  work<2>(b, out, rows, rings);
  return 0;
}
