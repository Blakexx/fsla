// measure/lab/experiments/fold_part.cu -- QUESTION: where does the carry's composed fold phase spend its 1.54x of the
// structural floor at nl64k-alt-k4? METHOD: the production fold (csrc/rola/src/carry/fold.cuh, the even box's arm) over
// one head's real write lists (`fold_part.inputs.py`), the pool filled once, one CTA an SM each a 32-digit box: mode 0
// the composed shape (a window's units, each `fold_unit` then a CTA barrier), mode 1 the same units back
// to back with no barrier (the step's own cost, steady), mode 2 the MMAs alone (the same steps, fixed rows and c, no
// records) -- each against the structural floor (the busiest scheduler's two warps' steps, every list at least one a
// unit, x 8 MMAs x 32 cycles); then modes 0 and 1 on the same lists with each unit's entries ordered by slot residue
// (`ordered.bin`: a unit's first step dealt across both 8-entry halves on distinct bank groups, pads on free ones),
// the bank conflicts' share; modes 3 (steady) and 4 (composed) fold 8-token steps on m16n8k8 MMAs (half a k16 MMA's
// cycles) from `packed.bin`'s heads, a second step only for a list past 8 entries, against the same floor.
// see docs/internals/tools/lab.md#fold-part
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "carry/fold.cuh"
#include "common/arm.cuh"
#include "lab.cuh"

namespace fold_part {

namespace rc = rola::carry;
using Arm = rola::Arm<0, 32, 1, 32, 1, 32, 0>;
using P = rc::Plan<Arm>;
constexpr int kRepeat = 16;

//: one head's lists, as the inputs file holds them
struct Lists {
  const uint16_t* at;
  const uint16_t* count;
  int windows, width, groups;
  const uint2* heads;
};

__device__ __forceinline__ rc::Window box_window(int box, int w) {
  return rc::Window{rc::kWindow, w * rc::kWindow, box, box * 32, 32, true};
}

//: MODE 2's unit: every list's steps (at least one a unit), the step's operands from a fixed residue-ordered row and
//: c = 1, no records
__device__ __forceinline__ void mma_unit(rc::State<P>& st, const uint8_t* smem,
                                         const rc::Window& win, int u, rc::ops::SmemAddr pool,
                                         int warp, int lane) {
  rola::static_for<P::kFoldLists>([&](auto J) {
    const int n = rc::list_span<P>(smem, win, u, warp, J).y;
    const int steps = max(1, (n + 15) >> 4);
#pragma unroll 1
    for (int s = 0; s < steps; ++s) {
      rc::Operand<P> o;
      rc::load_step<P>(o, pool, rc::row_offset<P>(static_cast<uint32_t>(lane & 15)), 0x3c003c00u,
                       0x3c003c00u, 0, lane);
      rc::mma_step<P, J>(st, o);
    }
  });
}

__device__ __forceinline__ void hmma_k8(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(b));
}

//: MODES 3 AND 4's STEP: heads `8 s ..` as eight m16n8k8 MMAs -- the slice's two m-tiles (A) and the values' four
//: n-tiles (B) by one `ldmatrix.x4.trans` each, lane (q, r) addressing entry r's 16-byte chunk q
template <int J>
__device__ __forceinline__ void step8(rc::State<P>& st, rc::ops::SmemAddr pool, const uint8_t* head, int lane) {
  const int r = lane & 7, q = lane >> 3, tq = lane & 3;
  const uint16_t* h16 = reinterpret_cast<const uint16_t*>(head);
  const uint32_t off = (rc::row_offset<P>(h16[4 * r]) ^ (16u * static_cast<uint32_t>(q & 1))) + 32u * (q >> 1);
  const uint4 c = *reinterpret_cast<const uint4*>(head + 16 * tq);
  const uint32_t c01 = rc::ops::prmt(c.y, c.w, 0x5410u);
  uint32_t a[4], b[4];
  rc::ops::load_frag_t(a, pool + off);
  rc::ops::load_frag_t(b, pool + P::kValuesAt + off);
  rola::static_for<P::kFoldMTiles>([&](auto M) {
    const uint32_t a0 = rc::ops::hmul2(a[2 * M], c01), a1 = rc::ops::hmul2(a[2 * M + 1], c01);
    rola::static_for<4>([&](auto N) { hmma_k8(st.acc[J][M][N], a0, a1, b[N]); });
  });
}

//: MODES 3 AND 4's UNIT: every list's first 8-token step in one block, then a second for each list past 8 entries
__device__ __forceinline__ void unit8(rc::State<P>& st, const uint8_t* smem, const rc::Window& win, int u,
                                      rc::ops::SmemAddr pool, int warp, int lane) {
  const uint8_t* heads = smem + P::kPhase + P::kHeads + (u & 1) * P::kHeadBytes;
  rola::static_for<P::kFoldLists>([&](auto J) {
    step8<J>(st, pool, heads + 128 * rc::list_of<P::kFoldLists>(warp, J, win.lists), lane);
  });
  rola::static_for<P::kFoldLists>([&](auto J) {
    for (int k = rc::ops::once(rc::list_span<P>(smem, win, u, warp, J).y > 8); k > 0; --k)
      step8<J>(st, pool, heads + 128 * rc::list_of<P::kFoldLists>(warp, J, win.lists) + 64, lane);
  });
}

template <int MODE>
__global__ __launch_bounds__(rc::kThreads, 1) void fold(const __grid_constant__ rc::CarryParams p,
                                                        Lists ls, long long* times, int* sinks) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int warp = lab::warp_id(), lane = lab::lane_id();
  const int box = static_cast<int>(blockIdx.x) % ls.groups;
  const rc::Place c{0, 0};
  const rc::ops::SmemAddr pool = rc::ops::smem_addr(smem + P::kPhase + P::kWriteSlot0);
  uint4* rows = reinterpret_cast<uint4*>(smem + P::kPhase + P::kWriteSlot0);
  for (int k = static_cast<int>(threadIdx.x); k < P::kUnitBytes / 16; k += rc::kThreads)
    rows[k] = make_uint4(0x2c002c00u, 0x2c002c00u, 0x2c002c00u, 0x2c002c00u);
  rc::State<P> st;
  rola::static_for<P::kFoldLists>([&](auto J) {
    rola::static_for<P::kFoldMTiles>([&](auto M) {
      rola::static_for<4>(
          [&](auto N) { rola::static_for<4>([&](auto I) { st.acc[J][M][N][I] = 0.f; }); });
    });
  });
  const uint32_t cf[P::kFoldLists] = {0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u};
  float f[P::kFoldLists] = {1.0f, 1.0f, 1.0f, 1.0f};
  long long spent = 0;
#pragma unroll 1
  for (int w = 0; w < ls.windows; ++w) {
    uint16_t* at = reinterpret_cast<uint16_t*>(smem + P::kWriteAt);
    for (int k = static_cast<int>(threadIdx.x); k < 32 * rc::kListAt; k += rc::kThreads)
      at[k] = ls.at[(static_cast<int64_t>(w) * ls.width + box * 32) * rc::kListAt + k];
    __syncthreads();
    const int units = (ls.count[w * ls.groups + box] + P::kUnitSlots - 1) / P::kUnitSlots;
    const rc::Window win = box_window(box, w);
#pragma unroll 1
    for (int u = 0; u < units; ++u) {
      //: the unit's heads, as the fill lands them: each box list's first 16 records, zero past its count
      uint2* heads =
          reinterpret_cast<uint2*>(smem + P::kPhase + P::kHeads + (u & 1) * P::kHeadBytes);
      for (int k = static_cast<int>(threadIdx.x); k < 32 * 16; k += rc::kThreads) {
        const int list = k >> 4, e = k & 15;
        const int begin = at[list * rc::kListAt + P::kFoldUnit * u];
        const int n = at[list * rc::kListAt + P::kFoldUnit * (u + 1)] - begin;
        const uint2* recs = reinterpret_cast<const uint2*>(p.write_lists.entries)
                            + (static_cast<int64_t>(w) * ls.width + box * 32 + list) * rc::kListCap;
        const int64_t at_head = ((static_cast<int64_t>(w) * ls.width + box * 32 + list) * 2 + u) * 16 + e;
        heads[k] = ls.heads ? ls.heads[at_head] : e < n ? recs[begin + e] : make_uint2(0u, 0u);
      }
      __syncthreads();
      const long long t0 = clock64();
#pragma unroll 1
      for (int r = 0; r < (MODE == 1 || MODE == 3 ? kRepeat : 1); ++r) {
        if constexpr (MODE == 2) {
          mma_unit(st, smem, win, u, pool, warp, lane);
        } else if constexpr (MODE >= 3) {
          unit8(st, smem, win, u, pool, warp, lane);
        } else {
          rc::fold_unit<P>(p, st, cf, f, smem, win, c, w, u, pool, warp, lane);
        }
      }
      if constexpr (MODE != 1 && MODE != 3) __syncthreads();
      spent += clock64() - t0;
      __syncthreads();
    }
  }
  uint32_t fold = 0u;
  rola::static_for<P::kStateRegs>([&](auto k) { fold += lab::bits(st.template at<k>()); });
  lab::sink(sinks, fold);
  if (lane == 0) times[blockIdx.x * rc::kWarps + warp] = spent / (MODE == 1 || MODE == 3 ? kRepeat : 1);
}

//: THE LISTS FILE (`fold_part.inputs.py`)
struct HostLists {
  int windows = 0, width = 0, groups = 0;
  std::vector<uint8_t> records;
  std::vector<uint16_t> at, count;
};

HostLists read_lists(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  int head[4];
  f.read(reinterpret_cast<char*>(head), sizeof(head));
  if (!f || head[0] != 0x444C4F46) {
    std::fprintf(stderr, "fold_part: %s is not a lists file\n", path.c_str());
    std::exit(1);
  }
  HostLists h;
  h.windows = head[1];
  h.width = head[2];
  h.groups = head[3];
  h.records.resize(static_cast<size_t>(h.windows) * h.width * rc::kListCap * 8);
  h.at.resize(static_cast<size_t>(h.windows) * h.width * rc::kListAt);
  h.count.resize(static_cast<size_t>(h.windows) * h.groups);
  f.read(reinterpret_cast<char*>(h.records.data()), static_cast<std::streamsize>(h.records.size()));
  f.read(reinterpret_cast<char*>(h.at.data()), static_cast<std::streamsize>(h.at.size() * 2));
  f.read(reinterpret_cast<char*>(h.count.data()), static_cast<std::streamsize>(h.count.size() * 2));
  return h;
}

//: THE ORDERED LISTS FILE (`fold_part.inputs.py`): the same lists, each unit's entries ordered, and its heads
HostLists read_ordered(const std::string& path, std::vector<uint2>& heads) {
  std::ifstream f(path, std::ios::binary);
  int head[5];
  f.read(reinterpret_cast<char*>(head), sizeof(head));
  if (!f || head[0] != 0x4F4C4F46 || head[4] != P::kFoldUnit) {
    std::fprintf(stderr, "fold_part: %s is not an ordered lists file for a %d-pass unit\n", path.c_str(),
                 P::kFoldUnit);
    std::exit(1);
  }
  HostLists h;
  h.windows = head[1];
  h.width = head[2];
  h.groups = head[3];
  h.records.resize(static_cast<size_t>(h.windows) * h.width * rc::kListCap * 8);
  h.at.resize(static_cast<size_t>(h.windows) * h.width * rc::kListAt);
  h.count.resize(static_cast<size_t>(h.windows) * h.groups);
  heads.resize(static_cast<size_t>(h.windows) * h.width * 2 * 16);
  f.read(reinterpret_cast<char*>(h.records.data()), static_cast<std::streamsize>(h.records.size()));
  f.read(reinterpret_cast<char*>(h.at.data()), static_cast<std::streamsize>(h.at.size() * 2));
  f.read(reinterpret_cast<char*>(h.count.data()), static_cast<std::streamsize>(h.count.size() * 2));
  f.read(reinterpret_cast<char*>(heads.data()), static_cast<std::streamsize>(heads.size() * 8));
  return h;
}

//: THE STRUCTURAL FLOOR a CTA-window: the busiest scheduler's two warps' steps (every list at least one a unit) x 8
//: MMAs x 32 cycles, averaged over the boxes and windows
double floor_of(const HostLists& h) {
  double total = 0;
  int cells = 0;
  for (int w = 0; w < h.windows; ++w)
    for (int box = 0; box < h.groups; ++box) {
      const int units = (h.count[w * h.groups + box] + P::kUnitSlots - 1) / P::kUnitSlots;
      int steps[rc::kWarps] = {};
      for (int u = 0; u < units; ++u)
        for (int list = 0; list < 32; ++list) {
          const uint16_t* a =
              &h.at[(static_cast<size_t>(w) * h.width + box * 32 + list) * rc::kListAt];
          const int n = a[P::kFoldUnit * (u + 1)] - a[P::kFoldUnit * u];
          steps[list / P::kFoldLists] += std::max(1, (n + 15) / 16);
        }
      int busiest = 0;
      for (int s = 0; s < 4; ++s) busiest = std::max(busiest, steps[s] + steps[s + 4]);
      total += busiest * 8.0 * 32.0;
      ++cells;
    }
  return total / cells;
}

template <int MODE>
void variant(const lab::Bench& b, const rc::CarryParams& p, const Lists& ls, double floor,
             const char* label) {
  const std::string name = lab::kernel("fold_part::fold", {MODE});
  b.expect(name, MODE >= 3 ? "HMMA.1688.F32" : "HMMA.16816.F32", ">=", P::kFoldLists * P::kFoldMTiles * 4);
  b.expect(name, "LDSM", ">=", 4);
  if (b.expect_only) return;
  const int bytes = P::kSmemBytes;
  LAB_CHECK(cudaFuncSetAttribute(fold<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  const lab::Samples s = b.variant(
      [&] { fold<MODE><<<b.sms, rc::kThreads, bytes>>>(p, ls, b.times, b.sinks); },
      [&] {
        return lab::Bench::mean(b.cycles(rc::kWarps), rc::kWarps, 0, rc::kWarps) / ls.windows;
      });
  b.record(std::string(label), "cycles a CTA-window", s);
  lab::Samples over;
  for (double v : s.v) over.v.push_back(v / floor);
  b.record(std::string(label) + " / floor", "x", over);
}

}  // namespace fold_part

int main(int argc, char** argv) {
  using namespace fold_part;
  lab::Bench b(argc, argv);
  std::string dir;
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], "--inputs")) dir = argv[i + 1];
  rc::CarryParams p{}, po{};
  Lists ls{}, lo{}, lp{};
  double floor = 1.0;
  if (!b.expect_only) {
    if (dir.empty()) {
      std::fprintf(stderr, "fold_part: --inputs <directory> (tools/lab.py writes it)\n");
      return 1;
    }
    const HostLists h = read_lists(dir + "/lists.bin");
    uint8_t* records = nullptr;
    uint16_t *at = nullptr, *count = nullptr;
    LAB_CHECK(cudaMalloc(&records, h.records.size()));
    LAB_CHECK(cudaMalloc(&at, h.at.size() * 2));
    LAB_CHECK(cudaMalloc(&count, h.count.size() * 2));
    LAB_CHECK(cudaMemcpy(records, h.records.data(), h.records.size(), cudaMemcpyHostToDevice));
    LAB_CHECK(cudaMemcpy(at, h.at.data(), h.at.size() * 2, cudaMemcpyHostToDevice));
    LAB_CHECK(cudaMemcpy(count, h.count.data(), h.count.size() * 2, cudaMemcpyHostToDevice));
    p.windows = h.windows;
    p.write_lists.entries = records;
    p.write_lists.width = h.width;
    p.write_lists.groups = h.groups;
    p.write_lists.entry_bytes = 8;
    ls = Lists{at, count, h.windows, h.width, h.groups, nullptr};
    floor = floor_of(h);
    std::vector<uint2> host_heads;
    const HostLists o = read_ordered(dir + "/ordered.bin", host_heads);
    uint8_t* ordered = nullptr;
    uint2* heads = nullptr;
    LAB_CHECK(cudaMalloc(&ordered, o.records.size()));
    LAB_CHECK(cudaMalloc(&heads, host_heads.size() * 8));
    LAB_CHECK(cudaMemcpy(ordered, o.records.data(), o.records.size(), cudaMemcpyHostToDevice));
    LAB_CHECK(cudaMemcpy(heads, host_heads.data(), host_heads.size() * 8, cudaMemcpyHostToDevice));
    po = p;
    po.write_lists.entries = ordered;
    lo = Lists{at, count, h.windows, h.width, h.groups, heads};
    std::vector<uint2> host_packed(host_heads.size());
    std::ifstream fp(dir + "/packed.bin", std::ios::binary);
    fp.read(reinterpret_cast<char*>(host_packed.data()), static_cast<std::streamsize>(host_packed.size() * 8));
    if (!fp) {
      std::fprintf(stderr, "fold_part: %s/packed.bin is short\n", dir.c_str());
      return 1;
    }
    uint2* packed = nullptr;
    LAB_CHECK(cudaMalloc(&packed, host_packed.size() * 8));
    LAB_CHECK(cudaMemcpy(packed, host_packed.data(), host_packed.size() * 8, cudaMemcpyHostToDevice));
    lp = Lists{at, count, h.windows, h.width, h.groups, packed};
    std::printf("{\"lab\": \"note\", \"floor\": %.1f}\n", floor);
  }
  variant<0>(b, p, ls, floor, "composed: a window's units then a barrier");
  variant<1>(b, p, ls, floor, "steady: the same units back to back");
  variant<2>(b, p, ls, floor, "the MMAs alone: fixed rows and c");
  variant<0>(b, po, lo, floor, "residue-ordered entries: composed");
  variant<1>(b, po, lo, floor, "residue-ordered entries: steady");
  variant<4>(b, po, lp, floor, "8-token steps: composed");
  variant<3>(b, po, lp, floor, "8-token steps: steady");
}
