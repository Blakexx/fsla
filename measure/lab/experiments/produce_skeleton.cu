// measure/lab/experiments/produce_skeleton.cu -- QUESTION: does the LIBRARY's producer skeleton -- its grid, its two
// math warpgroups and loader warp, its ring's two edges, its job table and the density's ticket -- write every element
// of every output for every routing of the test matrix, and what does the empty pipeline cost at the design's cell?
// METHOD: this file compiles the library's device code (`produce/produce.cuh`) into one kernel a lo-plane variant. THE
// CHECK: every output and a guard past each filled with a sentinel, then the launch, twice (the ticket and its counts
// must leave themselves zero): every output byte zero, every guard byte the sentinel, each density byte 0 where its
// level-side is masked and 255 where not -- for the five routings of `tests/integration/test_production_levels.py` at
// both alphas, at two shapes (3 sequences of 40 tokens, 2 write heads, hidden 96; 2 of 300, 3 write heads, hidden 64),
// with and without the lo plane, and at the timed shape. ROWS: the timed shape's launch (B 1, T 65,536, H_w 4, G 1,
// hidden 256, widths 256 and 256, d_v 64, every level-side entmax 1.5 and untied), without and with the lo plane, in
// microseconds and over its outputs' bytes at HBM's write rate.
// TIMELINE: no MMA and no staged bytes. The loader's trip a stage: its slot's EMPTY edge, a FULL arrival; a math warp's
// trip a stage: the FULL edge, lane 0's EMPTY arrival; a job's end its epilogue's zero stores, 16 bytes a store where
// whole. Live: the ring's parities, a share's addresses.
// see docs/internals/tools/lab.md#produce-skeleton
#include <cuda_bf16.h>

#include <cstring>
#include <vector>

#include "lab.cuh"
#include "produce/produce.cuh"

namespace produce_skeleton {

namespace pk = fsla::produce::detail;

//: THE KERNEL: the library's CTA, its stages' shared memory reserved (one CTA an SM, as the built kernel holds)
template <int LO>
__global__ void __launch_bounds__(pk::kThreads, 1) skeleton(const __grid_constant__ pk::Plan p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  pk::produce_cta<LO != 0>(p, smem);
}

//: a read entry: tied to the write side's level, or its own activation
constexpr int kTied = -1;

//: A ROUTING of the test matrix: the levels' widths, the write side's activations, each read head's entries
struct Routing {
  const char* name;
  std::vector<int> widths, write;
  std::vector<std::vector<int>> read;
};

//: `_routings(alpha)` of tests/integration/test_production_levels.py, `e` the alpha's entmax
std::vector<Routing> routings(int e) {
  const int s = pk::kSoftmax;
  return {
      {"tied", {4, 5}, {e, s}, {{kTied, s}}},
      {"mixed_per_level", {3, 5}, {s, e}, {{e, s}}},
      {"dense_only", {4}, {s}, {{s}}},
      {"both_sparse", {4, 5}, {e, s}, {{e, s}}},
      {"grouped", {4, 5}, {e, s}, {{kTied, e}, {s, s}, {e, kTied}}},
  };
}

//: A SHAPE: sequences, tokens, write heads, hidden, value channels
struct Shape {
  int batch, tokens, write_heads, hidden, d_v;
};

constexpr uint8_t kSentinel = 0xA5;
constexpr size_t kGuard = 256;

//: ONE OUTPUT: its device bytes (and a guard past them), filled with the sentinel before every launch
struct Buffer {
  uint8_t* at = nullptr;
  size_t bytes = 0;
};

//: THE OUTPUTS of a plan and their checks
struct Outputs {
  std::vector<Buffer> planes;
  Buffer density;
  pk::Plan plan{};

  uint8_t* add(size_t bytes) {
    Buffer b;
    b.bytes = bytes;
    LAB_CHECK(cudaMalloc(&b.at, bytes + kGuard));
    planes.push_back(b);
    return b.at;
  }

  void fill() const {
    for (const Buffer& b : planes) LAB_CHECK(cudaMemset(b.at, kSentinel, b.bytes + kGuard));
    LAB_CHECK(cudaMemset(density.at, kSentinel, density.bytes + kGuard));
  }

  //: the bytes that are wrong: an output byte not zero, a guard byte not the sentinel, a density byte not its side's
  long long wrong() const {
    long long n = 0;
    std::vector<uint8_t> h;
    for (const Buffer& b : planes) {
      h.resize(b.bytes + kGuard);
      LAB_CHECK(cudaMemcpy(h.data(), b.at, h.size(), cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < h.size(); ++i) n += h[i] != (i < b.bytes ? 0 : kSentinel);
    }
    h.resize(density.bytes + kGuard);
    LAB_CHECK(cudaMemcpy(h.data(), density.at, h.size(), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < h.size(); ++i) {
      const int side = static_cast<int>(i) / plan.levels, level = static_cast<int>(i) % plan.levels;
      n += h[i] != (i < density.bytes ? (plan.masked[side][level] ? 0 : 255) : kSentinel);
    }
    unsigned long long counts[pk::kSides * pk::kMaxLevels];
    uint32_t ticket = 1;
    LAB_CHECK(cudaMemcpy(counts, plan.counts, sizeof(counts), cudaMemcpyDeviceToHost));
    LAB_CHECK(cudaMemcpy(&ticket, plan.ticket, 4, cudaMemcpyDeviceToHost));
    for (unsigned long long c : counts) n += c != 0;
    return n + (ticket != 0);
  }

  ~Outputs() {
    for (const Buffer& b : planes) cudaFree(b.at);
    cudaFree(density.at);
    cudaFree(plan.counts);
    cudaFree(plan.ticket);
  }
};

//: THE PLAN of a routing at a shape: the job table of one write head and every output allocated in api.md's layout
void plan_of(const Routing& r, const Shape& sh, bool lo, Outputs& o) {
  pk::Plan& p = o.plan;
  const int levels = static_cast<int>(r.widths.size()), group = static_cast<int>(r.read.size());
  p.batch = sh.batch;
  p.tokens = sh.tokens;
  p.write_heads = sh.write_heads;
  p.group = group;
  p.levels = levels;
  p.hidden = sh.hidden;
  p.d_v = sh.d_v;
  const auto zeros = [&](int a) { return a != pk::kSoftmax; };
  for (int l = 0; l < levels; ++l) {
    p.widths[l] = r.widths[l];
    p.masked[pk::kWrite][l] = zeros(r.write[l]);
    bool any = false;
    for (int g = 0; g < group; ++g) any = any || zeros(r.read[g][l] == kTied ? r.write[l] : r.read[g][l]);
    p.masked[pk::kRead][l] = any;
  }
  int jobs = 0;
  for (int l = 0; l < levels; ++l) {
    uint16_t tied = 0;
    for (int g = 0; g < group; ++g) tied |= static_cast<uint16_t>(r.read[g][l] == kTied) << g;
    p.job[jobs++] = {pk::kWriteJob, static_cast<uint8_t>(l), 0, static_cast<uint8_t>(r.write[l]),
                     static_cast<uint16_t>(r.widths[l]), tied};
  }
  for (int g = 0; g < group; ++g)
    for (int l = 0; l < levels; ++l)
      if (r.read[g][l] != kTied)
        p.job[jobs++] = {pk::kReadJob, static_cast<uint8_t>(l), static_cast<uint8_t>(g),
                         static_cast<uint8_t>(r.read[g][l]), static_cast<uint16_t>(r.widths[l]), 0};
  p.job[jobs++] = {pk::kValueJob, 0, 0, 0, static_cast<uint16_t>(sh.d_v), 0};
  p.jobs = jobs;
  const size_t tokens = static_cast<size_t>(sh.batch) * sh.tokens, words = (sh.tokens + 31) / 32;
  for (int side = 0; side < pk::kSides; ++side) {
    const size_t heads = side == pk::kWrite ? sh.write_heads : static_cast<size_t>(sh.write_heads) * group;
    for (int l = 0; l < levels; ++l) {
      const size_t width = r.widths[l];
      p.rows[side][l] = o.add(tokens * heads * width);
      p.scales[side][l] = reinterpret_cast<float*>(o.add(tokens * heads * 4));
      p.lo[side][l] = lo ? reinterpret_cast<int8_t*>(o.add(tokens * heads * width)) : nullptr;
      p.masks[side][l] =
          p.masked[side][l] ? reinterpret_cast<uint32_t*>(o.add(sh.batch * heads * words * width * 4)) : nullptr;
    }
  }
  p.values = reinterpret_cast<__nv_bfloat16*>(o.add(tokens * sh.write_heads * sh.d_v * 2));
  o.density.bytes = static_cast<size_t>(pk::kSides) * levels;
  LAB_CHECK(cudaMalloc(&o.density.at, o.density.bytes + kGuard));
  p.density = o.density.at;
  LAB_CHECK(cudaMalloc(&p.counts, sizeof(unsigned long long) * pk::kSides * pk::kMaxLevels));
  LAB_CHECK(cudaMemset(p.counts, 0, sizeof(unsigned long long) * pk::kSides * pk::kMaxLevels));
  LAB_CHECK(cudaMalloc(&p.ticket, 4));
  LAB_CHECK(cudaMemset(p.ticket, 0, 4));
}

//: the launch: grid `B ceil(T / 128) x H_w`
template <int LO>
void launch(const pk::Plan& p, cudaStream_t stream = nullptr) {
  const dim3 grid(p.batch * ((p.tokens + pk::kTileTokens - 1) / pk::kTileTokens), p.write_heads);
  skeleton<LO><<<grid, pk::kThreads, pk::kSmemBytes, stream>>>(p);
}

//: THE CHECK of one plan: two launches, each from the sentinels
template <int LO>
long long covered(Outputs& o) {
  long long n = 0;
  for (int rep = 0; rep < 2; ++rep) {
    o.fill();
    launch<LO>(o.plan);
    LAB_CHECK(cudaDeviceSynchronize());
    n += o.wrong();
  }
  return n;
}

//: the outputs' bytes of a plan (the density's aside)
size_t output_bytes(const Outputs& o) {
  size_t n = 0;
  for (const Buffer& b : o.planes) n += b.bytes;
  return n;
}

}  // namespace produce_skeleton

template <int LO>
void run(lab::Bench& b) {
  using namespace produce_skeleton;
  LAB_CHECK(cudaFuncSetAttribute(skeleton<LO>, cudaFuncAttributeMaxDynamicSharedMemorySize, pk::kSmemBytes));
  const std::string lo = LO ? ", the lo plane" : "";
  //: THE CHECK over the test matrix
  long long wrong = 0, plans = 0;
  std::string first;
  for (int e : {static_cast<int>(pk::kEntmax15), static_cast<int>(pk::kEntmax2)})
    for (const Routing& r : routings(e))
      for (const Shape& sh : {Shape{3, 40, 2, 96, 16}, Shape{2, 300, 3, 64, 24}}) {
        Outputs o;
        plan_of(r, sh, LO != 0, o);
        const long long n = covered<LO>(o);
        ++plans;
        if (n && first.empty()) first = std::string(r.name) + " at " + std::to_string(sh.tokens) + " tokens";
        wrong += n;
      }
  //: THE TIMED SHAPE, its check and its time
  const Routing cell{"the cell", {256, 256}, {pk::kEntmax15, pk::kEntmax15}, {{pk::kEntmax15, pk::kEntmax15}}};
  Outputs o;
  plan_of(cell, Shape{1, 65536, 4, 256, 64}, LO != 0, o);
  const long long n = covered<LO>(o);
  wrong += n;
  if (n && first.empty()) first = "the timed shape";
  b.check("the skeleton" + lo, wrong == 0,
          std::to_string(wrong) + " wrong bytes over " + std::to_string(plans + 1) + " plans" +
              (first.empty() ? "" : ", the first at " + first));
  cudaEvent_t e0, e1;
  LAB_CHECK(cudaEventCreate(&e0));
  LAB_CHECK(cudaEventCreate(&e1));
  //: HBM's write rate, stream_rate's record (measure/rates.json: 3,170 GB/s)
  constexpr double kWriteBytesAMicrosecond = 3.170e6;
  const double floor_us = static_cast<double>(output_bytes(o)) / kWriteBytesAMicrosecond;
  lab::Samples share;
  const lab::Samples us = b.variant([&] { launch<LO>(o.plan); },
                                    [&] {
                                      LAB_CHECK(cudaEventRecord(e0));
                                      launch<LO>(o.plan);
                                      LAB_CHECK(cudaEventRecord(e1));
                                      LAB_CHECK(cudaEventSynchronize(e1));
                                      float ms = 0;
                                      LAB_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                                      share.v.push_back(1000.0 * ms / floor_us);
                                      return 1000.0 * ms;
                                    });
  b.record("the empty pipeline at the cell" + lo, "microseconds a launch", us);
  b.record("the empty pipeline at the cell" + lo + ": against the write floor", "the launch over its outputs' bytes at "
           "HBM's write rate", share);
  LAB_CHECK(cudaEventDestroy(e0));
  LAB_CHECK(cudaEventDestroy(e1));
}

int main(int argc, char** argv) {
  using namespace produce_skeleton;
  lab::Bench b(argc, argv);
  for (int lo : {0, 1}) {
    const std::string k = lab::kernel("skeleton", {lo});
    b.expect(k, "HGMMA", "==", 0);
    //: the stubs' zero stores are loops of lane-dependent trips (a share's chunks over its 128 threads, a ragged chunk's
    //: bytes) and the ring's edges polled waits: their joins, inlined at each store's site; the components replace them
    b.expect(k, "LOOP_CONVERGENCE", "<=", lo ? 200 : 160);
    b.exempt(k, "a skeleton: its components are stubs that write zeros; each lands against tools/budgets/produce.json");
  }
  if (b.expect_only) return 0;
  run<0>(b);
  run<1>(b);
  return 0;
}
