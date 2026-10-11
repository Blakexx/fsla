// measure/lab/experiments/fold_mchan.cu -- QUESTION: what does the fold step cost with the MMA's M the value channels
// (each warp 16 channels of every leaf in the box; 4 warps a CTA, 2 CTAs an SM) beside today's leaf carve, on
// nl64k-alt-k4's real write lists? METHOD: a CTA owns a box of 256 leaves x 64 channels -- LISTS list digits x 256 /
// LISTS operand digits -- its warps the transposed state S^T (channels x leaves, fp32 accumulators); a list's step is
// A = the step's values transposed (16 channels x 16 tokens, `ldmatrix.trans` from the pool's token-major rows) times
// each token's c by HMUL2, B = the raw operand slice (16 tokens x 8 digits an n-tile, `ldmatrix.trans`), one MMA a leaf
// n-tile; every warp walks every list. The lists, records and heads are the facts pass's (`fold_mchan.inputs.py`); the
// pool (128 rows a CTA, swizzled, filled once) holds a token's slice and its 64 channels. Mode 0 composed (a window's
// lists then a CTA barrier), 1 steady (back to back), 2 the MMAs alone, timed a burst; whole loops: 3 a barrier a
// window, 4 heads prefetched by `cp.async`, 5 lane-ready step operands (the facts pass's form: a lane's two slots and
// two c pairs, 16 bytes), 6 that free-running on landed and drained mbarriers -- against the structural floor (the
// SM's MMAs a window, every list at least one step, at 32 cycles a scheduler). see docs/internals/tools/lab.md#fold-mchan
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "lab.cuh"

namespace fold_mchan {

constexpr int kWarps = 4, kThreads = 32 * kWarps, kCtas = 2;
constexpr int kChannels = 64, kListCap = 512, kListAt = 8;
constexpr int kSlots = 128, kRepeat = 16;

//: a box of LISTS list digits x DIGITS operand digits = 256 leaves; a pool row = the token's slice, then its channels
template <int LISTS>
struct Shape {
  static constexpr int kDigits = 256 / LISTS, kTiles = kDigits / 8;
  static constexpr int kSliceBytes = 2 * kDigits, kValuesAt = kSliceBytes;
  static constexpr int kPitch = 32 * (((kSliceBytes + 2 * kChannels + 31) / 32) | 1);
  static constexpr int kPool = 0, kHeads = kPool + kSlots * kPitch, kScratch = kHeads + LISTS * 128;
  static constexpr int kAt = kScratch + kWarps * 128;
  //: mode 4's prefetch: a second heads buffer and three pass-start buffers (window w + 2's land during window w)
  static constexpr int kHeads2 = kAt + LISTS * kListAt * 2, kAts = kHeads2 + LISTS * 128;
  //: modes 5 and 6: three buffers past the scratch (window w + 2's land during window w), each the lists' lane-ready
  //: first steps (a list's 32 lanes' two slots and two c pairs, 16 bytes a lane) then their pass starts
  static constexpr int kReadyBytes = LISTS * 32 * 16, kBufBytes = kReadyBytes + LISTS * kListAt * 2;
  static constexpr int kBufs = kAt, kBars = kBufs + 3 * kBufBytes;
  static constexpr int kOld = kAts + 3 * LISTS * kListAt * 2, kNew = kBars + 6 * 8;
  static constexpr int kBytes = kOld > kNew ? kOld : kNew;
  static_assert(kBytes <= 49 * 1024, "two CTAs an SM (sm_86: 100 KB an SM less 1 KB a CTA)");
};

struct Lists {
  const uint2* records;
  const uint16_t* at;
  const uint4* ready;
  int windows, width, groups;
};

__device__ __forceinline__ uint32_t row_offset(uint32_t slot, int pitch) {
  const uint32_t r = slot & (kSlots - 1);
  return r * pitch + 16u * ((r >> 2) & 1u);
}

//: ONE STEP of list `J`: lanes' entries' slots and four tokens' c from the heads (or a record pair for a later step)
template <int LISTS, int J, int MODE>
__device__ __forceinline__ void step(float (&acc)[256 / 8][4], uint32_t pool, const uint8_t* head, int warp,
                                     int lane) {
  using S = Shape<LISTS>;
  const int r = lane & 7, q = lane >> 3, tq = lane & 3;
  uint32_t oa, ob, c01, c89;
  if constexpr (MODE >= 5) {
    const uint4 rd = reinterpret_cast<const uint4*>(head)[lane];
    oa = row_offset(rd.x, S::kPitch);
    ob = row_offset(rd.y, S::kPitch);
    c01 = rd.z;
    c89 = rd.w;
  } else if constexpr (MODE == 2) {
    oa = row_offset(8 * (q >> 1) + r, S::kPitch);
    ob = row_offset(8 * (q & 1) + r, S::kPitch);
    c01 = c89 = 0x3c003c00u;
  } else {
    const uint16_t* h16 = reinterpret_cast<const uint16_t*>(head);
    oa = row_offset(h16[4 * (8 * (q >> 1) + r)], S::kPitch);
    ob = row_offset(h16[4 * (8 * (q & 1) + r)], S::kPitch);
    const uint4 p01 = *reinterpret_cast<const uint4*>(head + 16 * tq);
    const uint4 p89 = *reinterpret_cast<const uint4*>(head + 16 * tq + 64);
    uint32_t d;
    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(d) : "r"(p01.y), "r"(p01.w));
    c01 = d;
    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(d) : "r"(p89.y), "r"(p89.w));
    c89 = d;
  }
  //: A = the values transposed: rows the warp's 16 channels, columns the step's 16 tokens
  uint32_t a[4];
  lab::ldsm_t(a, pool + ((oa + S::kValuesAt + 32u * warp) ^ (16u * static_cast<uint32_t>(q & 1))));
  const uint32_t aj[4] = {lab::hmul2(a[0], c01), lab::hmul2(a[1], c01), lab::hmul2(a[2], c89),
                          lab::hmul2(a[3], c89)};
#pragma unroll
  for (int h = 0; h < S::kTiles / 2; ++h) {
    uint32_t b[4];
    lab::ldsm_t(b, pool + ((ob + 32u * h) ^ (16u * static_cast<uint32_t>(q >> 1))));
    lab::mma_f16_f32(acc[J * S::kTiles + 2 * h], aj, b[0], b[1]);
    lab::mma_f16_f32(acc[J * S::kTiles + 2 * h + 1], aj, b[2], b[3]);
  }
}

template <int N, class F, int... I>
__device__ __forceinline__ void each(F&& f, std::integer_sequence<int, I...>) {
  (f(std::integral_constant<int, I>{}), ...);
}

template <int N, class F>
__device__ __forceinline__ void each(F&& f) {
  each<N>(f, std::make_integer_sequence<int, N>{});
}

//: a window's lists: every list's first step from its heads in one straight-line block, a longer list's later steps
//: from its records (lanes < 16 entry `16 s + lane`, the four tokens' c by shuffles)
template <int LISTS, int MODE>
__device__ __forceinline__ void window(float (&acc)[256 / 8][4], uint32_t pool, const uint8_t* smem, const Lists& ls,
                                       int box, int w, int warp, int lane, int heads_at = Shape<LISTS>::kHeads,
                                       int at_at = Shape<LISTS>::kAt) {
  using S = Shape<LISTS>;
  const uint16_t* at = reinterpret_cast<const uint16_t*>(smem + at_at);
  each<LISTS>([&](auto J) {
    step<LISTS, J, MODE>(acc, pool, smem + heads_at + (MODE >= 5 ? 512 : 128) * J, warp, lane);
  });
  int longest = 0;
#pragma unroll
  for (int j = 0; j < LISTS; ++j) longest = max(longest, at[j * kListAt + 4] - at[j * kListAt]);
  for (int k = longest > 16 ? 1 : 0; k > 0; --k) {
    each<LISTS>([&](auto J) {
      const int n = at[J * kListAt + 4] - at[J * kListAt];
      const uint2* recs = ls.records + (static_cast<int64_t>(w) * ls.width + box * LISTS + J) * kListCap
                          + at[J * kListAt];
#pragma unroll 1
      for (int s = 1; s < (n + 15) / 16; ++s) {
        const uint2 v = recs[min(16 * s + (lane & 15), n - 1)];
        const bool live = (lane < 16) & (16 * s + lane < n);
        uint8_t* scratch = const_cast<uint8_t*>(smem) + S::kScratch + 128 * warp;
        __syncwarp();
        if (lane < 16) reinterpret_cast<uint2*>(scratch)[lane] = live ? v : make_uint2(0u, 0u);
        __syncwarp();
        step<LISTS, J, (MODE >= 5 ? 0 : MODE)>(acc, pool, scratch, warp, lane);
      }
    });
  }
}

template <int LISTS, int MODE>
__global__ __launch_bounds__(kThreads, kCtas) void fold(Lists ls, long long* times, int* sinks) {
  using S = Shape<LISTS>;
  extern __shared__ __align__(128) uint8_t smem[];
  const int warp = lab::warp_id(), lane = lab::lane_id();
  const int boxes = 256 / LISTS, box = static_cast<int>(blockIdx.x) % boxes;
  const uint32_t pool = lab::smem(smem + S::kPool);
  uint4* rows = reinterpret_cast<uint4*>(smem + S::kPool);
  for (int k = static_cast<int>(threadIdx.x); k < kSlots * S::kPitch / 16; k += kThreads)
    rows[k] = make_uint4(0x2c002c00u, 0x2c002c00u, 0x2c002c00u, 0x2c002c00u);
  float acc[256 / 8][4] = {};
  long long spent = 0;
  //: mode 4, production's shape: the next window's heads and the window after next's pass starts prefetched by
  //: `cp.async` during the fold, the whole loop timed (both CTAs' bursts drift as their windows differ)
  if constexpr (MODE == 4) {
    const auto at_buf = [&](int w) { return S::kAts + (w % 3) * LISTS * kListAt * 2; };
    const auto heads_buf = [&](int w) { return (w & 1) ? S::kHeads2 : S::kHeads; };
    const auto at_in = [&](int w) {
      if (static_cast<int>(threadIdx.x) < LISTS * kListAt * 2 / 16)
        lab::cp_async16(lab::smem(smem + at_buf(w)) + 16 * threadIdx.x,
                        reinterpret_cast<const uint8_t*>(ls.at + (static_cast<int64_t>(w) * ls.width + box * LISTS)
                                                                     * kListAt)
                            + 16 * threadIdx.x);
    };
    const auto heads_in = [&](int w) {
      const uint16_t* at = reinterpret_cast<const uint16_t*>(smem + at_buf(w));
      for (int k = static_cast<int>(threadIdx.x); k < LISTS * 16; k += kThreads) {
        const int j = k >> 4, e = k & 15, n = at[j * kListAt + 4] - at[j * kListAt];
        const uint2* src = ls.records + (static_cast<int64_t>(w) * ls.width + box * LISTS + j) * kListCap
                           + at[j * kListAt] + min(e, max(n - 1, 0));
        const uint32_t dst = lab::smem(smem + heads_buf(w)) + 8 * k;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(dst), "l"(src), "r"(e < n ? 8 : 0));
      }
    };
    at_in(0);
    if (ls.windows > 1) at_in(1);
    lab::cp_commit();
    lab::cp_wait_all();
    __syncthreads();
    heads_in(0);
    lab::cp_commit();
    lab::cp_wait_all();
    __syncthreads();
    const long long t0 = clock64();
#pragma unroll 1
    for (int w = 0; w < ls.windows; ++w) {
      if (w + 2 < ls.windows) at_in(w + 2);
      if (w + 1 < ls.windows) heads_in(w + 1);
      lab::cp_commit();
      window<LISTS, 0>(acc, pool, smem, ls, box, w, warp, lane, heads_buf(w), at_buf(w));
      lab::cp_wait_all();
      __syncthreads();
    }
    spent = clock64() - t0;
  }
  //: modes 5 and 6: the facts pass's lane-ready first steps (16 bytes a lane a list) and pass starts copied two windows
  //: ahead into three buffers, each landing on an mbarrier (every thread's copies); mode 5 a CTA barrier a window, mode
  //: 6 none -- a buffer is refilled once every warp has arrived past its window on a second mbarrier, so warps free-run
  if constexpr (MODE >= 5) {
    const uint32_t bars = lab::smem(smem + S::kBars), landed = bars, drained = bars + 24;
    const auto buf = [&](int w) { return S::kBufs + (w % 3) * S::kBufBytes; };
    const auto fill = [&](int w) {
      const int64_t first = static_cast<int64_t>(w) * ls.width + box * LISTS;
      const uint8_t* ready = reinterpret_cast<const uint8_t*>(ls.ready + first * 32);
      const uint8_t* at = reinterpret_cast<const uint8_t*>(ls.at + first * kListAt);
      const uint32_t dst = lab::smem(smem + buf(w));
      for (int k = static_cast<int>(threadIdx.x); k < S::kBufBytes / 16; k += kThreads)
        lab::cp_async16(dst + 16 * k,
                        k < S::kReadyBytes / 16 ? ready + 16 * k : at + 16 * (k - S::kReadyBytes / 16));
      lab::mbar_arrive_when_landed(landed + 8 * (w % 3));
    };
    if (threadIdx.x == 0)
      for (int k = 0; k < 3; ++k) {
        lab::mbar_init(landed + 8 * k, kThreads);
        lab::mbar_init(drained + 8 * k, kWarps);
      }
    __syncthreads();
    fill(0);
    if (ls.windows > 1) fill(1);
    const long long t0 = clock64();
#pragma unroll 1
    for (int w = 0; w < ls.windows; ++w) {
      lab::mbar_wait(landed + 8 * (w % 3), (w / 3) & 1);
      window<LISTS, MODE>(acc, pool, smem, ls, box, w, warp, lane, buf(w), buf(w) + S::kReadyBytes);
      if constexpr (MODE == 5) __syncthreads();
      __syncwarp();
      if (lane == 0) lab::mbar_arrive(drained + 8 * (w % 3));
      for (int k = w + 2 < ls.windows ? 1 : 0; k > 0; --k) {
        if (w >= 1) lab::mbar_wait(drained + 8 * ((w - 1) % 3), ((w - 1) / 3) & 1);
        fill(w + 2);
      }
    }
    spent = clock64() - t0;
  }
  //: mode 3 copies window 0's heads once and times the whole loop: every window a fold burst then a barrier, both CTAs
  //: doing nothing else, so the SM's throughput with the second CTA filling the first's edges
  const long long whole = clock64();
#pragma unroll 1
  for (int w = 0; w < (MODE >= 4 ? 0 : ls.windows); ++w) {
    for (int k = MODE == 3 && w > 0 ? 1 : 0; k > 0; --k) {
      window<LISTS, 0>(acc, pool, smem, ls, box, 0, warp, lane);
      __syncthreads();
    }
    if (MODE == 3 && w > 0) continue;
    //: the window's heads and pass starts, as the fill lands them (first 16 records a list, zero past its count)
    uint16_t* at = reinterpret_cast<uint16_t*>(smem + S::kAt);
    for (int k = static_cast<int>(threadIdx.x); k < LISTS * kListAt; k += kThreads)
      at[k] = ls.at[(static_cast<int64_t>(w) * ls.width + box * LISTS) * kListAt + k];
    __syncthreads();
    uint2* heads = reinterpret_cast<uint2*>(smem + S::kHeads);
    for (int k = static_cast<int>(threadIdx.x); k < LISTS * 16; k += kThreads) {
      const int j = k >> 4, e = k & 15, n = at[j * kListAt + 4] - at[j * kListAt];
      heads[k] = e < n ? ls.records[(static_cast<int64_t>(w) * ls.width + box * LISTS + j) * kListCap
                                    + at[j * kListAt] + e]
                       : make_uint2(0u, 0u);
    }
    __syncthreads();
    const long long t0 = clock64();
#pragma unroll 1
    for (int rr = 0; rr < (MODE == 1 ? kRepeat : 1); ++rr) window<LISTS, MODE>(acc, pool, smem, ls, box, w, warp, lane);
    if (MODE != 1) __syncthreads();
    spent += clock64() - t0;
    __syncthreads();
  }
  uint32_t fold = 0u;
#pragma unroll
  for (int t = 0; t < 256 / 8; ++t)
#pragma unroll
    for (int i = 0; i < 4; ++i) fold += lab::bits(acc[t][i]);
  lab::sink(sinks, fold);
  if (MODE == 3) spent = (clock64() - whole) * ls.windows / max(ls.windows - 1, 1);
  (void)whole;
  if (lane == 0) times[blockIdx.x * kWarps + warp] = spent / (MODE == 1 ? kRepeat : 1);
}

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
    std::fprintf(stderr, "fold_mchan: %s is not a lists file\n", path.c_str());
    std::exit(1);
  }
  HostLists h;
  h.windows = head[1];
  h.width = head[2];
  h.groups = head[3];
  h.records.resize(static_cast<size_t>(h.windows) * h.width * kListCap * 8);
  h.at.resize(static_cast<size_t>(h.windows) * h.width * kListAt);
  h.count.resize(static_cast<size_t>(h.windows) * h.groups);
  f.read(reinterpret_cast<char*>(h.records.data()), static_cast<std::streamsize>(h.records.size()));
  f.read(reinterpret_cast<char*>(h.at.data()), static_cast<std::streamsize>(h.at.size() * 2));
  f.read(reinterpret_cast<char*>(h.count.data()), static_cast<std::streamsize>(h.count.size() * 2));
  return h;
}

//: THE FACTS PASS'S LANE-READY FIRST STEPS, a uint4 a lane a list a window: the lane's A and B token slots
//: (`ldmatrix.trans` row addresses) and its two c pairs (the A fragment's token columns), zero past the list's count
std::vector<uint32_t> ready_of(const HostLists& h) {
  std::vector<uint32_t> out(static_cast<size_t>(h.windows) * h.width * 32 * 4);
  const uint32_t* rec = reinterpret_cast<const uint32_t*>(h.records.data());
  for (size_t list = 0; list < static_cast<size_t>(h.windows) * h.width; ++list) {
    const uint16_t* a = &h.at[list * kListAt];
    const int n = a[4] - a[0];
    const uint32_t* base = rec + (list * kListCap + a[0]) * 2;
    const auto slot = [&](int e) { return e < n ? base[2 * e] & 0xffffu : 0u; };
    const auto cv = [&](int e) { return e < n ? base[2 * e + 1] & 0xffffu : 0u; };
    for (int lane = 0; lane < 32; ++lane) {
      const int r = lane & 7, q = lane >> 3, tq = lane & 3;
      uint32_t* o = &out[(list * 32 + lane) * 4];
      o[0] = slot(8 * (q >> 1) + r);
      o[1] = slot(8 * (q & 1) + r);
      o[2] = cv(2 * tq) | cv(2 * tq + 1) << 16;
      o[3] = cv(8 + 2 * tq) | cv(9 + 2 * tq) << 16;
    }
  }
  return out;
}

//: THE STRUCTURAL FLOOR an SM-window: each scheduler holds one warp of each CTA, every warp walks all its box's lists
//: (at least one step a list) at kTiles MMAs a step -- two warps' MMAs at 32 cycles, averaged over boxes and windows
template <int LISTS>
double floor_of(const HostLists& h, int only = -1) {
  double mmas = 0;
  int cells = 0;
  for (int w = only < 0 ? 0 : only; w < (only < 0 ? h.windows : only + 1); ++w)
    for (int box = 0; box < 256 / LISTS; ++box) {
      int steps = 0;
      for (int j = 0; j < LISTS; ++j) {
        const uint16_t* a = &h.at[(static_cast<size_t>(w) * h.width + box * LISTS + j) * kListAt];
        steps += std::max(1, (a[4] - a[0] + 15) / 16);
      }
      mmas += steps * Shape<LISTS>::kTiles;
      ++cells;
    }
  return 2.0 * mmas / cells * 32.0;
}

template <int LISTS, int MODE>
void variant(const lab::Bench& b, const Lists& ls, double floor, const char* label) {
  const std::string name = lab::kernel("fold_mchan::fold", {LISTS, MODE});
  b.expect(name, "HMMA.16816.F32", ">=", LISTS * Shape<LISTS>::kTiles);
  b.expect(name, "LDSM", ">=", 2 * LISTS);
  if (b.expect_only) return;
  const int bytes = Shape<LISTS>::kBytes;
  LAB_CHECK(cudaFuncSetAttribute(fold<LISTS, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  int per_sm = 0;
  LAB_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fold<LISTS, MODE>, kThreads, bytes));
  if (per_sm < kCtas) {
    std::fprintf(stderr, "fold_mchan: %s fits %d CTAs an SM, not %d\n", name.c_str(), per_sm, kCtas);
    std::exit(1);
  }
  const lab::Samples s = b.variant(
      [&] { fold<LISTS, MODE><<<kCtas * b.sms, kThreads, bytes>>>(ls, b.times, b.sinks); },
      [&] { return lab::Bench::mean(b.cycles(kCtas * kWarps), kCtas * kWarps, 0, kCtas * kWarps) / ls.windows; });
  b.record(std::string(label), "cycles an SM-window", s);
  lab::Samples over;
  for (double v : s.v) over.v.push_back(v / floor);
  b.record(std::string(label) + " / floor", "x", over);
}

}  // namespace fold_mchan

int main(int argc, char** argv) {
  using namespace fold_mchan;
  lab::Bench b(argc, argv);
  std::string dir;
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], "--inputs")) dir = argv[i + 1];
  Lists ls{};
  double f16 = 1.0, f8 = 1.0, w16 = 1.0, w8 = 1.0;
  if (!b.expect_only) {
    if (dir.empty()) {
      std::fprintf(stderr, "fold_mchan: --inputs <directory> (tools/lab.py writes it)\n");
      return 1;
    }
    const HostLists h = read_lists(dir + "/lists.bin");
    uint8_t* records = nullptr;
    uint16_t* at = nullptr;
    uint32_t* ready = nullptr;
    const std::vector<uint32_t> host_ready = ready_of(h);
    LAB_CHECK(cudaMalloc(&ready, host_ready.size() * 4));
    LAB_CHECK(cudaMemcpy(ready, host_ready.data(), host_ready.size() * 4, cudaMemcpyHostToDevice));
    LAB_CHECK(cudaMalloc(&records, h.records.size()));
    LAB_CHECK(cudaMalloc(&at, h.at.size() * 2));
    LAB_CHECK(cudaMemcpy(records, h.records.data(), h.records.size(), cudaMemcpyHostToDevice));
    LAB_CHECK(cudaMemcpy(at, h.at.data(), h.at.size() * 2, cudaMemcpyHostToDevice));
    ls = Lists{reinterpret_cast<const uint2*>(records), at, reinterpret_cast<const uint4*>(ready), h.windows, h.width,
               h.groups};
    f16 = floor_of<16>(h);
    f8 = floor_of<8>(h);
    w16 = floor_of<16>(h, 0);
    w8 = floor_of<8>(h, 0);
    std::printf("{\"lab\": \"note\", \"floor_16x16\": %.1f, \"floor_8x32\": %.1f}\n", f16, f8);
  }
  variant<16, 0>(b, ls, f16, "16 x 16 box: composed");
  variant<16, 1>(b, ls, f16, "16 x 16 box: steady");
  variant<16, 2>(b, ls, f16, "16 x 16 box: the MMAs alone");
  variant<16, 3>(b, ls, w16, "16 x 16 box: bursts and barriers, whole loop");
  variant<8, 3>(b, ls, w8, "8 x 32 box: bursts and barriers, whole loop");
  variant<16, 4>(b, ls, f16, "16 x 16 box: production's loop, heads prefetched, whole loop");
  variant<8, 4>(b, ls, f8, "8 x 32 box: production's loop, heads prefetched, whole loop");
  variant<16, 5>(b, ls, f16, "16 x 16 box: lane-ready steps, a barrier a window, whole loop");
  variant<16, 6>(b, ls, f16, "16 x 16 box: lane-ready steps, mbarriers (free-running), whole loop");
  variant<8, 5>(b, ls, f8, "8 x 32 box: lane-ready steps, a barrier a window, whole loop");
  variant<8, 6>(b, ls, f8, "8 x 32 box: lane-ready steps, mbarriers (free-running), whole loop");
  variant<8, 0>(b, ls, f8, "8 x 32 box: composed");
  variant<8, 1>(b, ls, f8, "8 x 32 box: steady");
  variant<8, 2>(b, ls, f8, "8 x 32 box: the MMAs alone");
}
