// measure/lab/experiments/produce_skeleton.cu -- QUESTION: does the LIBRARY's producer skeleton -- its grid, its two
// math warpgroups and loader warp, its ring's two edges, its job table and its tickets -- write every element of every
// output for every routing of the test matrix, and where does the empty pipeline's time go at the design's cell?
// METHOD: this file compiles the library's device code (`produce/produce.cuh`) into one kernel a (lo plane, CTAs an SM,
// trace) variant: the library holds the CTA, the lab its launch. THE CHECK: every output and a guard past each filled with a sentinel, then the launch, twice (the
// tickets and their sums must leave themselves zero): every output byte zero, every guard byte the sentinel, each
// density byte 0 where its level row is masked and 255 where not, the growth's sums and the tickets zero -- for the five
// routings of `tests/integration/test_production_levels.py` at both alphas, at two shapes (3 sequences of 40 tokens, 2
// write heads, hidden 96; 2 of 300, 3 write heads, hidden 64), and at the cell. ROWS at the cell
// (`layer-alt-entmax1.5-256x256-T65536-h4`: B 1, T 65,536, H_w 4, G 1, hidden 256, d_v 64): the launch; the launch at
// hidden 0 (no ring stage); the launch at two CTAs an SM (the ring two stages); each beside its twin with every warp's
// phases stamped (two CTAs' timelines); and the launch with the lo plane.
// TIMELINE: no MMA and no staged bytes. The loader's trip a stage: its slot's EMPTY edge, a FULL arrival; a math warp's
// trip a stage: the FULL edge, lane 0's EMPTY arrival; a job's end its epilogue's zero stores, 16 bytes a store where
// whole; the CTA's end its tickets. Live: the ring's parities, a share's addresses.
// see docs/internals/tools/lab.md#produce-skeleton
#include <cuda_bf16.h>

#include <cstring>
#include <vector>

#include "lab.cuh"
#include "produce/produce.cuh"

namespace produce_skeleton {

namespace pk = fsla::produce::detail;
using Bt = ::denseref::arch::BuildTag<::denseref::arch::Build{900}>;
constexpr int kWarps = pk::kThreads / 32;
//: a phase's site in a warp's timeline: above every barrier's
constexpr uint32_t kPhases = 0xc000u;

//: THE KERNEL'S TRACER AS WARPLINE'S STAMPS: a ring barrier's site is its number, a phase's `kPhases` on
template <bool ON>
struct Stamps {
  lab::Trace<ON> row;
  __device__ __forceinline__ void wait_begin(int id) { row.stamp(warpline::kWaitBegin, static_cast<uint32_t>(id)); }
  __device__ __forceinline__ void wait_end(int id) { row.stamp(warpline::kWaitEnd, static_cast<uint32_t>(id)); }
  __device__ __forceinline__ void arrive(int id) { row.stamp(warpline::kArrive, static_cast<uint32_t>(id)); }
  __device__ __forceinline__ void phase(int site) { row.mark(warpline::kPhaseBegin, kPhases + static_cast<uint32_t>(site)); }
};

//: the bytes a launch reserves: the ring's stages on the card
template <int CTAS>
constexpr int kReserved = static_cast<int>(pk::Smem<Bt, CTAS>::template reserved_to<pk::Ring>(pk::kStages<Bt, CTAS>));

//: THE KERNEL: the library's CTA, `CTAS` an SM, through warpline's stamps where `TRACE` (the library holds the CTA, the
//: lab its launch, until the op launches it)
template <int LO, int CTAS, int TRACE>
__global__ void __launch_bounds__(pk::kThreads, CTAS) skeleton(const __grid_constant__ pk::Plan p, unsigned long long* rows) {
  Stamps<TRACE != 0> tr;
  tr.row.bind(rows);
  pk::produce_cta<Bt, CTAS, LO != 0>(p, tr);
  tr.row.close(0);
}

//: a read entry: tied to the write side's level, or its own activation
constexpr int kTied = -1;

//: A ROUTING: the levels' widths, the write side's activations, each read head's entries
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

//: THE CELL: `layer-alt-entmax1.5-256x256-T65536-h4` (@fsla_cells), d_v its hidden over its heads
const Routing kCell{"the cell", {256, 256}, {pk::kEntmax15, pk::kSoftmax}, {{pk::kSoftmax, pk::kEntmax15}}};
constexpr Shape kCellShape{1, 65536, 4, 256, 64};

constexpr uint8_t kSentinel = 0xA5;
constexpr size_t kGuard = 256;

//: ONE BUFFER: its device bytes (and a guard past them)
struct Buffer {
  uint8_t* at = nullptr;
  size_t bytes = 0;
};

//: THE OUTPUTS of a plan and their checks: the outputs filled with the sentinel before every launch, the workspace zero
struct Outputs {
  std::vector<Buffer> planes, workspace;
  Buffer density;
  pk::Plan plan{};

  static Buffer make(size_t bytes) {
    Buffer b;
    b.bytes = bytes;
    LAB_CHECK(cudaMalloc(&b.at, bytes + kGuard));
    return b;
  }

  uint8_t* output(size_t bytes) {
    planes.push_back(make(bytes));
    return planes.back().at;
  }

  uint8_t* zeroed(size_t bytes) {
    workspace.push_back(make(bytes));
    LAB_CHECK(cudaMemset(workspace.back().at, 0, bytes));
    return workspace.back().at;
  }

  void fill() const {
    for (const Buffer& b : planes) LAB_CHECK(cudaMemset(b.at, kSentinel, b.bytes + kGuard));
    LAB_CHECK(cudaMemset(density.at, kSentinel, density.bytes + kGuard));
  }

  //: the bytes that are wrong: an output byte not zero, a guard byte not the sentinel, a density byte not its row's, a
  //: workspace byte not zero
  long long wrong() const {
    long long n = 0;
    std::vector<uint8_t> h;
    for (const Buffer& b : planes) {
      h.resize(b.bytes + kGuard);
      LAB_CHECK(cudaMemcpy(h.data(), b.at, h.size(), cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < h.size(); ++i) n += h[i] != (i < b.bytes ? 0 : kSentinel);
    }
    for (const Buffer& b : workspace) {
      h.resize(b.bytes);
      LAB_CHECK(cudaMemcpy(h.data(), b.at, h.size(), cudaMemcpyDeviceToHost));
      for (uint8_t v : h) n += v != 0;
    }
    h.resize(density.bytes + kGuard);
    LAB_CHECK(cudaMemcpy(h.data(), density.at, h.size(), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < h.size(); ++i) {
      const int row = static_cast<int>(i) / plan.levels, level = static_cast<int>(i) % plan.levels;
      n += h[i] != (i < density.bytes ? (plan.masked[row][level] ? 0 : 255) : kSentinel);
    }
    return n;
  }

  ~Outputs() {
    for (const Buffer& b : planes) cudaFree(b.at);
    for (const Buffer& b : workspace) cudaFree(b.at);
    cudaFree(density.at);
  }
};

//: THE PLAN'S EXTENTS, as the kernel derives them from the plan alone, against what was allocated: a field the kernel
//: reads and the host did not set refuses the run before any launch
void refuse_unsized(const Outputs& o, bool lo) {
  const pk::Plan& p = o.plan;
  const size_t streams = static_cast<size_t>(p.batch) * p.write_heads, tiles = (p.tokens + 63) / 64;
  const size_t words = (p.tokens + 31) / 32;
  std::vector<size_t> want;
  for (int l = 0; l < p.levels; ++l) {
    const size_t tile = static_cast<size_t>((p.widths[l] + 15) / 16) * 1024;
    for (int side = 0; side < pk::kSides; ++side) {
      const size_t heads = side == pk::kWrite ? 1 : p.group;
      want.push_back(heads * streams * tiles * tile);
      want.push_back(heads * streams * p.tokens * 4);
      if (lo) want.push_back(heads * streams * tiles * tile);
    }
    for (int row = p.group; row >= 0; --row)
      if (p.masked[row][l] && (row == p.group || p.masks[row][l] != p.masks[p.group][l]))
        want.push_back(streams * words * p.widths[l] * 4);
  }
  want.push_back(streams * p.tokens * p.d_v * 2);
  want.push_back(streams * p.d_v * 4);
  bool same = want.size() == o.planes.size();
  for (size_t i = 0; same && i < want.size(); ++i) same = want[i] == o.planes[i].bytes && want[i] > 0;
  if (!same) {
    std::fprintf(stderr, "produce_skeleton: the plan's extents (the kernel's view) are not its outputs' allocations\n");
    std::exit(2);
  }
}

//: THE PLAN of a routing at a shape: one write head's job table and every output in its layout
//: (docs/internals/produce/produce.md#plan)
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
  const size_t streams = static_cast<size_t>(sh.batch) * sh.write_heads, tokens = streams * sh.tokens;
  const size_t row_tiles = (sh.tokens + 63) / 64, words = (sh.tokens + 31) / 32;
  int jobs = 0;
  for (int l = 0; l < levels; ++l) {
    p.widths[l] = r.widths[l];
    const size_t width = r.widths[l], blocks = (width + 15) / 16;
    //: the level-sides' tensors, then each row's mask (the write side's, then each read head's): a tied head reads the write side's
    p.masked[group][l] = zeros(r.write[l]);
    for (int side = 0; side < pk::kSides; ++side) {
      const size_t heads = side == pk::kWrite ? 1 : group;
      p.rows[side][l] = o.output(heads * streams * row_tiles * blocks * 1024);
      p.scales[side][l] = reinterpret_cast<float*>(o.output(heads * tokens * 4));
      p.lo[side][l] = lo ? o.output(heads * streams * row_tiles * blocks * 1024) : nullptr;
    }
    uint16_t tied = 0;
    for (int row = group; row >= 0; --row) {
      if (row < group && r.read[row][l] == kTied) {
        tied |= static_cast<uint16_t>(1u << row);
        p.masked[row][l] = p.masked[group][l];
        p.masks[row][l] = p.masks[group][l];
        continue;
      }
      if (row < group) p.masked[row][l] = zeros(r.read[row][l]);
      p.masks[row][l] = p.masked[row][l] ? reinterpret_cast<uint32_t*>(o.output(streams * words * width * 4)) : nullptr;
    }
    p.job[jobs++] = {pk::kWriteJob, static_cast<uint8_t>(l), 0, static_cast<uint8_t>(r.write[l]), tied};
  }
  for (int g = 0; g < group; ++g)
    for (int l = 0; l < levels; ++l)
      if (r.read[g][l] != kTied)
        p.job[jobs++] = {pk::kReadJob, static_cast<uint8_t>(l), static_cast<uint8_t>(g), static_cast<uint8_t>(r.read[g][l]), 0};
  p.job[jobs++] = {pk::kValueJob, 0, 0, 0, 0};
  p.jobs = jobs;
  p.values = reinterpret_cast<__nv_bfloat16*>(o.output(tokens * sh.d_v * 2));
  p.growth = reinterpret_cast<float*>(o.output(streams * sh.d_v * 4));
  p.growth_sums = reinterpret_cast<float*>(o.zeroed(streams * sh.d_v * 4));
  p.sequence_tickets = reinterpret_cast<uint32_t*>(o.zeroed(streams * 4));
  p.counts = reinterpret_cast<unsigned long long*>(o.zeroed(sizeof(unsigned long long) * pk::kMaxRows * pk::kMaxLevels));
  p.ticket = reinterpret_cast<uint32_t*>(o.zeroed(4));
  o.density = Outputs::make(static_cast<size_t>(group + 1) * levels);
  p.density = o.density.at;
  refuse_unsized(o, lo);
}

//: the grid: a CTA a sequence's 128-token tile and a write head
int grid_of(const pk::Plan& p) { return p.batch * ((p.tokens + pk::kTileTokens - 1) / pk::kTileTokens) * p.write_heads; }

template <int LO, int CTAS, int TRACE>
void launch(const pk::Plan& p, unsigned long long* rows) {
  constexpr int kBytes = kReserved<CTAS>;
  skeleton<LO, CTAS, TRACE><<<grid_of(p), pk::kThreads, kBytes>>>(p, rows);
}

//: THE CHECK of one plan: two launches, each from the sentinels
template <int LO, int CTAS>
long long covered(Outputs& o) {
  long long n = 0;
  for (int rep = 0; rep < 2; ++rep) {
    o.fill();
    launch<LO, CTAS, 0>(o.plan, nullptr);
    LAB_CHECK(cudaDeviceSynchronize());
    n += o.wrong();
  }
  return n;
}

}  // namespace produce_skeleton

//: A ROW: the variant's check over the test matrix and the cell, its launch timed, and (TRACE) two CTAs' timelines
template <int LO, int CTAS, int TRACE>
void run(lab::Bench& b, const std::string& name, int hidden, unsigned long long* rows) {
  using namespace produce_skeleton;
  if constexpr (TRACE != 0)
    LAB_CHECK(cudaFuncSetAttribute(skeleton<LO, CTAS, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, kReserved<CTAS>));
  LAB_CHECK(cudaFuncSetAttribute(skeleton<LO, CTAS, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, kReserved<CTAS>));
  long long wrong = 0, plans = 0;
  std::string first;
  for (int e : {static_cast<int>(pk::kEntmax15), static_cast<int>(pk::kEntmax2)})
    for (const Routing& r : routings(e))
      for (const Shape& sh : {Shape{3, 40, 2, 96, 16}, Shape{2, 300, 3, 64, 24}}) {
        Outputs o;
        plan_of(r, sh, LO != 0, o);
        const long long n = covered<LO, CTAS>(o);
        ++plans;
        if (n && first.empty()) first = std::string(r.name) + " at " + std::to_string(sh.tokens) + " tokens";
        wrong += n;
      }
  Outputs o;
  plan_of(kCell, Shape{kCellShape.batch, kCellShape.tokens, kCellShape.write_heads, hidden, kCellShape.d_v}, LO != 0, o);
  const long long n = covered<LO, CTAS>(o);
  wrong += n;
  if (n && first.empty()) first = "the cell";
  b.check(name, wrong == 0, std::to_string(wrong) + " wrong bytes over " + std::to_string(plans + 1) + " plans" +
                                (first.empty() ? "" : ", the first at " + first));
  cudaEvent_t e0, e1;
  LAB_CHECK(cudaEventCreate(&e0));
  LAB_CHECK(cudaEventCreate(&e1));
  const auto timed = [&](auto go) {
    return b.variant(go, [&] {
      LAB_CHECK(cudaEventRecord(e0));
      go();
      LAB_CHECK(cudaEventRecord(e1));
      LAB_CHECK(cudaEventSynchronize(e1));
      float ms = 0;
      LAB_CHECK(cudaEventElapsedTime(&ms, e0, e1));
      return 1000.0 * ms;
    });
  };
  b.record(name, "microseconds a launch", timed([&] { launch<LO, CTAS, 0>(o.plan, nullptr); }));
  if constexpr (TRACE != 0) {
    const std::string traced = name + ", every warp's phases stamped";
    b.record(traced, "microseconds a launch", timed([&] { launch<LO, CTAS, 1>(o.plan, rows); }));
    std::vector<std::string> roles;
    for (int w = 0; w < kWarps; ++w) roles.push_back(w < pk::kMathWarps ? "math " + std::to_string(w / 4) : "loader");
    std::vector<std::pair<uint32_t, std::string>> legend;
    for (int s = 0; s < pk::kStages<Bt, CTAS>; ++s) {
      legend.push_back({static_cast<uint32_t>(s), "stage " + std::to_string(s) + " loaded"});
      legend.push_back({static_cast<uint32_t>(pk::kEmptyId + s), "stage " + std::to_string(s) + " released"});
    }
    legend.insert(legend.end(), {{kPhases + pk::kSiteEntry, "entry"}, {kPhases + pk::kSiteSetup, "setup"},
                                 {kPhases + pk::kSiteWindow, "window"}, {kPhases + pk::kSiteEpilogue, "epilogue"},
                                 {kPhases + pk::kSiteTicket, "ticket"}, {kPhases + pk::kSiteExit, "exit"}});
    b.trace(traced, rows, kWarps, {0, grid_of(o.plan) / 2}, roles, legend);
  }
  LAB_CHECK(cudaEventDestroy(e0));
  LAB_CHECK(cudaEventDestroy(e1));
}

int main(int argc, char** argv) {
  using namespace produce_skeleton;
  lab::Bench b(argc, argv);
  std::vector<std::string> names;
  for (const auto& v : std::vector<std::vector<long long>>{{0, 1, 0}, {0, 1, 1}, {1, 1, 0}, {0, 2, 0}, {0, 2, 1}})
    names.push_back(lab::kernel("skeleton", {v[0], v[1], v[2]}));
  for (const std::string& k : names) {
    b.expect(k, "HGMMA", "==", 0);
    //: the stubs' zero stores are loops of lane-dependent trips (a share's chunks over its 128 threads, a ragged chunk's
    //: words or bytes) and the ring's edges polled waits: their joins, inlined at each store's site
    b.expect(k, "LOOP_CONVERGENCE", "<=", 220);
    b.exempt(k, "a skeleton: its components are stubs that write zeros; each lands against tools/budgets/produce.json");
  }
  if (b.expect_only) return 0;
  unsigned long long* rows = nullptr;
  const size_t words = static_cast<size_t>(grid_of([] {
                         pk::Plan p{};
                         p.batch = kCellShape.batch;
                         p.tokens = kCellShape.tokens;
                         p.write_heads = kCellShape.write_heads;
                         return p;
                       }())) * kWarps * lab::Trace<true>::kRow;
  LAB_CHECK(cudaMalloc(&rows, words * 8));
  LAB_CHECK(cudaMemset(rows, 0, words * 8));
  const std::string cell = "the empty pipeline at the cell";
  run<0, 1, 1>(b, cell, kCellShape.hidden, rows);
  run<0, 1, 1>(b, cell + ", hidden 0", 0, rows);
  run<0, 2, 1>(b, cell + ", two CTAs an SM", kCellShape.hidden, rows);
  run<1, 1, 0>(b, cell + ", the lo plane", kCellShape.hidden, rows);
  return 0;
}
