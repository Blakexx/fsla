// measure/lab/experiments/read_image.cu -- QUESTION: the carry's readout with the state from a bf16 image in shared
// memory: tokens on the MMA's M (16-token steps, the state the B operand) or on its N (8-token tiles, the state the A
// operand), what does each cost against its tensor floor at nl64k-alt-k4's read lists and at a dense cell, and what
// do the panel's adds and the pass's drain add? METHOD: 8 warps, one CTA an SM a box of 32 lists x 32 read digits x
// 32 channels; a warp reads digits w, w + 8, w + 16, w + 24, holding a digit's state fragments (16 registers, from the
// image by `ldmatrix.trans`) while its tokens stream; read rows (32 lists of bf16, 64 bytes, swizzled) gathered from a
// 256-slot pool by slot; bf16 MMAs with fp32 accumulate; outputs times each token's coefficient to s16 pairs added by
// `red.shared` into the token's panel row; the drain each pass's end (`red.global.add` two rows a request, each word
// cleared). The image holds 16 planes (digit d on plane d % 16: a plane's access pattern is all that matters). Modes:
// form 0 tokens on M, 1 tokens on N; out 0 the MMAs alone (outputs summed into a sink), 1 + panel adds, 2 + the drain;
// the state a digit at a time, or a ring of two digits (the next one's fragments loading under this one's steps); a
// warp's heads (its digits' pass starts and first 16 slots) staged a window ahead by `cp.async`, as the fill would.
// see docs/internals/tools/lab.md#read-image
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "lab.cuh"

namespace read_image {

constexpr int kWarps = 8, kThreads = 32 * kWarps;
constexpr int kLists = 32, kDigits = 32, kChannels = 32;
constexpr int kListCap = 512, kListAt = 8, kSlots = 256, kWords = 16, kPanelPitch = 20;
constexpr int kPlanes = 16, kPlaneBytes = kLists * kChannels * 2, kRowBytes = kLists * 2;
constexpr int kImage = 0, kPool = kImage + kPlanes * kPlaneBytes, kPanel = kPool + kSlots * kRowBytes;
constexpr int kDummy = kPanel + kSlots * kPanelPitch * 4, kHeads = kDummy + 32 * 4;
//: a warp's heads for a window, double-buffered: its 4 digits' pass starts (16 bytes each), then their first 16 slots
constexpr int kHeadBytes = 4 * 16 + 4 * 32, kBytes = kHeads + 2 * kWarps * kHeadBytes;
constexpr float kUnit = 256.0f;

struct Lists {
  const uint16_t* slots;
  const uint16_t* at;
  const uint16_t* count;
  int windows, width, groups;
  uint32_t* out;
};

//: a 16-byte chunk `c` of a 64-byte row `row` (an image plane's list or a pool slot), swizzled by the row's bits 1-2
__device__ __forceinline__ uint32_t chunk(uint32_t base, int row, int c) {
  return base + row * kRowBytes + 16 * (c ^ ((row >> 1) & 3));
}

//: entry `e` of a digit's pass as (slot | c << 16), c a bf16 (1 here: the coefficient comes with the token's row and
//: moves no cycles), zero past its count: the first 16 of pass 0 from the warp's heads in shared memory (`head`), the
//: rest from global memory; a dense cell's every token
template <int DENSE>
__device__ __forceinline__ uint32_t entry(const Lists& ls, const uint16_t* head, int64_t row, int begin, int n,
                                          int e) {
  if constexpr (DENSE) {
    return e < n ? static_cast<uint32_t>(e) | (0x3f80u << 16) : 0u;
  } else {
    const uint32_t slot = head ? head[e & 15] : ls.slots[row * kListCap + begin + min(e, max(n - 1, 0))];
    return e < n ? slot | (0x3f80u << 16) : 0u;
  }
}

__device__ __forceinline__ float coef(uint32_t packed) { return __uint_as_float(packed & 0xffff0000u) * kUnit; }

__device__ __forceinline__ uint32_t s16_pair(float lo, float hi) {
  int a, b;
  asm("cvt.rni.sat.s16.f32 %0, %1;" : "=r"(a) : "f"(lo));
  asm("cvt.rni.sat.s16.f32 %0, %1;" : "=r"(b) : "f"(hi));
  return (static_cast<uint32_t>(a) & 0xffffu) | (static_cast<uint32_t>(b) << 16);
}

//: a token's add into its panel row, branch-free: a pad (entry 0) adds nothing to the lane's own dummy word
__device__ __forceinline__ void panel_add(uint32_t sm, uint32_t packed, int word, uint32_t v, int lane) {
  const uint32_t at = packed != 0u ? sm + kPanel + ((packed & 0xffu) * kPanelPitch + word) * 4 : sm + kDummy + lane * 4;
  asm volatile("red.shared.add.u32 [%0], %1;" ::"r"(at), "r"(v) : "memory");
}

//: A DIGIT'S STATE FRAGMENTS from its image plane, 16 registers: form 0's B (K = lists, N = channels) at
//: `(kb * 4 + nt) * 2 + h`, form 1's A (M = channels, K = lists) at `(m * 2 + kb) * 4 + i`
template <int FORM>
__device__ __forceinline__ void state_in(uint32_t (&f)[16], uint32_t plane, int lane) {
  const int r = lane & 7, q = lane >> 3;
#pragma unroll
  for (int u = 0; u < 2; ++u)
#pragma unroll
    for (int v = 0; v < 2; ++v) {
      uint32_t x[4];
      if constexpr (FORM == 0) {
        lab::ldsm_t(x, chunk(plane, 16 * u + 8 * (q & 1) + r, 2 * v + (q >> 1)));
        f[(u * 4 + 2 * v) * 2] = x[0];
        f[(u * 4 + 2 * v) * 2 + 1] = x[1];
        f[(u * 4 + 2 * v + 1) * 2] = x[2];
        f[(u * 4 + 2 * v + 1) * 2 + 1] = x[3];
      } else {
        lab::ldsm_t(x, chunk(plane, 16 * v + 8 * (q >> 1) + r, 2 * u + (q & 1)));
#pragma unroll
        for (int i = 0; i < 4; ++i) f[(u * 2 + v) * 4 + i] = x[i];
      }
    }
}

//: FORM 0, TOKENS ON M: the digit's state as B, a 16-token step's read rows as A
template <int OUT, int DENSE>
__device__ __forceinline__ void step_m(uint32_t sm, const Lists& ls, const uint16_t* head, int64_t row, int begin,
                                       int n, const uint32_t (&f)[16], int lane, float& sink, int s) {
  const int r = lane & 7, q = lane >> 3, g = lane >> 2, t = lane & 3;
  {
    const uint32_t mine = entry<DENSE>(ls, s < 1 ? head : nullptr, row, begin, n, 16 * s + (lane & 15));
    const uint32_t ea = __shfl_sync(0xffffffffu, mine, 8 * (q & 1) + r);
    const uint32_t eg = __shfl_sync(0xffffffffu, mine, g), eg8 = __shfl_sync(0xffffffffu, mine, g + 8);
    float acc[4][4] = {};
#pragma unroll
    for (int kb = 0; kb < 2; ++kb) {
      uint32_t a[4];
      lab::ldsm(a, chunk(sm + kPool, ea & 0xffu, 2 * kb + (q >> 1)));
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) lab::mma_bf16_f32(acc[nt], a, f[(kb * 4 + nt) * 2], f[(kb * 4 + nt) * 2 + 1]);
    }
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      if constexpr (OUT == 0) {
        sink += acc[nt][0] + acc[nt][1] + acc[nt][2] + acc[nt][3];
      } else {
        const float cg = coef(eg), cg8 = coef(eg8);
        panel_add(sm, eg, 4 * nt + t, s16_pair(acc[nt][0] * cg, acc[nt][1] * cg), lane);
        panel_add(sm, eg8, 4 * nt + t, s16_pair(acc[nt][2] * cg8, acc[nt][3] * cg8), lane);
      }
    }
  }
}

template <int OUT, int DENSE>
__device__ __forceinline__ void digit_m(uint32_t sm, const Lists& ls, const uint16_t* head, int64_t row, int begin,
                                        int n, const uint32_t (&f)[16], int lane, float& sink) {
#pragma unroll 1
  for (int s = 0; s < (n + 15) / 16; ++s) step_m<OUT, DENSE>(sm, ls, head, row, begin, n, f, lane, sink, s);
}

//: FORM 1, TOKENS ON N: the digit's state as A, an 8-token tile's read rows as B
template <int OUT, int DENSE>
__device__ __forceinline__ void digit_n(uint32_t sm, const Lists& ls, const uint16_t* head, int64_t row, int begin,
                                        int n, const uint32_t (&f)[16], int lane, float& sink) {
  const int r = lane & 7, q = lane >> 3, g = lane >> 2, t = lane & 3;
#pragma unroll 1
  for (int s = 0; s < (n + 7) / 8; ++s) {
    const uint32_t mine = entry<DENSE>(ls, s < 2 ? head : nullptr, row, begin, n, 8 * s + (lane & 7));
    const uint32_t eb = __shfl_sync(0xffffffffu, mine, r);
    const uint32_t e0 = __shfl_sync(0xffffffffu, mine, 2 * t), e1 = __shfl_sync(0xffffffffu, mine, 2 * t + 1);
    uint32_t b[4];
    lab::ldsm(b, chunk(sm + kPool, eb & 0xffu, q));
    float acc[2][4] = {};
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int kb = 0; kb < 2; ++kb) {
        const uint32_t a[4] = {f[(m * 2 + kb) * 4], f[(m * 2 + kb) * 4 + 1], f[(m * 2 + kb) * 4 + 2],
                               f[(m * 2 + kb) * 4 + 3]};
        lab::mma_bf16_f32(acc[m], a, b[2 * kb], b[2 * kb + 1]);
      }
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      if constexpr (OUT == 0) {
        sink += acc[m][0] + acc[m][1] + acc[m][2] + acc[m][3];
      } else {
        const float c0 = coef(e0), c1 = coef(e1);
        panel_add(sm, e0, 8 * m + g, s16_pair(acc[m][0] * c0, acc[m][2] * c0), lane);
        panel_add(sm, e1, 8 * m + g, s16_pair(acc[m][1] * c1, acc[m][3] * c1), lane);
      }
    }
  }
}

template <int FORM, int OUT, int DENSE, int RING>
__global__ __launch_bounds__(kThreads, 1) void read(Lists ls, long long* times, int* sinks) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int warp = lab::warp_id(), lane = lab::lane_id();
  const int boxes = ls.width / kDigits, box = static_cast<int>(blockIdx.x) % boxes;
  const uint32_t sm = lab::smem(smem);
  uint32_t* words = reinterpret_cast<uint32_t*>(smem);
  for (int k = static_cast<int>(threadIdx.x); k < kPanel / 4; k += kThreads) words[k] = 0x3d803d80u;
  for (int k = kPanel / 4 + static_cast<int>(threadIdx.x); k < kBytes / 4; k += kThreads) words[k] = 0u;
  __syncthreads();
  float sink = 0.0f;
  //: the warp's heads for window `w` into buffer `w & 1` by `cp.async` (lanes 0-11: four pass-start rows, eight
  //: 16-byte halves of the first 16 slots), a window ahead of their use
  const auto heads_in = [&](int w) {
    const int i = lane & 3, part = lane >> 2;
    const int64_t row = static_cast<int64_t>(w) * ls.width + box * kDigits + warp + 8 * i;
    const uint32_t dst = sm + kHeads + ((w & 1) * kWarps + warp) * kHeadBytes;
    if (!DENSE && part == 0) lab::cp_async16(dst + 16 * i, ls.at + row * kListAt);
    if (!DENSE && (part == 1 || part == 2)) lab::cp_async16(dst + 64 + 32 * i + 16 * (part - 1), ls.slots + row * kListCap + 8 * (part - 1));
    lab::cp_commit();
  };
  heads_in(0);
  int next = DENSE ? 2 * kSlots : ls.count[box];
  const long long t0 = clock64();
#pragma unroll 1
  for (int w = 0; w < ls.windows; ++w) {
    const int count = next;
    for (int k = w + 1 < ls.windows ? 1 : 0; k > 0; --k) {
      heads_in(w + 1);
      next = DENSE ? 2 * kSlots : ls.count[(w + 1) * ls.groups + box];
    }
    asm volatile("cp.async.wait_group 1;" ::: "memory");
    __syncwarp();
    const uint8_t* mine = smem + kHeads + ((w & 1) * kWarps + warp) * kHeadBytes;
    const int passes = (count + kSlots - 1) / kSlots;
#pragma unroll 1
    for (int p = 0; p < passes; ++p) {
      const auto digit = [&](int i, const uint32_t(&f)[16]) {
        const int64_t row = static_cast<int64_t>(w) * ls.width + box * kDigits + warp + 8 * i;
        const uint16_t* at = reinterpret_cast<const uint16_t*>(mine) + 8 * i;
        const uint16_t* head = p == 0 ? reinterpret_cast<const uint16_t*>(mine + 64) + 16 * i : nullptr;
        const int begin = DENSE ? 0 : at[2 * p], n = DENSE ? kSlots : at[2 * p + 2] - begin;
        if constexpr (FORM == 0) digit_m<OUT, DENSE>(sm, ls, head, row, begin, n, f, lane, sink);
        if constexpr (FORM == 1) digit_n<OUT, DENSE>(sm, ls, head, row, begin, n, f, lane, sink);
      };
      const auto plane = [&](int i) { return sm + kImage + ((warp + 8 * i) % kPlanes) * kPlaneBytes; };
      //: RING 2 (form 0): every digit's first step in one straight-line block, as the fold's, so the scheduler loads a
      //: digit's operands under the MMAs before them; the rare digit past 16 tokens continues after
      if constexpr (RING == 2) {
        const auto args = [&](int i, int64_t& row, const uint16_t*& head, int& begin, int& n) {
          row = static_cast<int64_t>(w) * ls.width + box * kDigits + warp + 8 * i;
          const uint16_t* at = reinterpret_cast<const uint16_t*>(mine) + 8 * i;
          head = p == 0 ? reinterpret_cast<const uint16_t*>(mine + 64) + 16 * i : nullptr;
          begin = DENSE ? 0 : at[2 * p];
          n = DENSE ? kSlots : at[2 * p + 2] - begin;
        };
        uint32_t f[4][16];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          int64_t row;
          const uint16_t* head;
          int begin, n;
          args(i, row, head, begin, n);
          state_in<FORM>(f[i], plane(i), lane);
          step_m<OUT, DENSE>(sm, ls, head, row, begin, n, f[i], lane, sink, 0);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          int64_t row;
          const uint16_t* head;
          int begin, n;
          args(i, row, head, begin, n);
#pragma unroll 1
          for (int s = 1; s < (n + 15) / 16; ++s) step_m<OUT, DENSE>(sm, ls, head, row, begin, n, f[i], lane, sink, s);
        }
      } else if constexpr (RING == 1) {
        uint32_t f0[16], f1[16];
        state_in<FORM>(f0, plane(0), lane);
        state_in<FORM>(f1, plane(1), lane);
        digit(0, f0);
        state_in<FORM>(f0, plane(2), lane);
        digit(1, f1);
        state_in<FORM>(f1, plane(3), lane);
        digit(2, f0);
        digit(3, f1);
      } else {
#pragma unroll 1
        for (int i = 0; i < 4; ++i) {
          uint32_t f[16];
          state_in<FORM>(f, plane(i), lane);
          digit(i, f);
        }
      }
      __syncthreads();
      if constexpr (OUT == 2) {
        const int rows = min(kSlots, count - kSlots * p);
#pragma unroll 1
        for (int k = 0; k < (min(rows - kSlots / kWarps * warp, kSlots / kWarps) + 1) / 2; ++k) {
          const int slot = kSlots / kWarps * warp + 2 * k + (lane >> 4), word = lane & 15;
          uint32_t* cell = words + kPanel / 4 + slot * kPanelPitch + word;
          uint32_t* target = ls.out + ((static_cast<int64_t>(w) * 2 + p) * kSlots + slot) * kWords + word;
          asm volatile("red.global.add.u32 [%0], %1;" ::"l"(target), "r"(*cell) : "memory");
          *cell = 0u;
        }
        __syncthreads();
      }
    }
  }
  const long long spent = clock64() - t0;
  lab::sink(sinks, lab::bits(sink));
  if (lane == 0) times[blockIdx.x * kWarps + warp] = spent;
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
  if (!f || head[0] != 0x44414552) {
    std::fprintf(stderr, "read_image: %s is not a read lists file\n", path.c_str());
    std::exit(1);
  }
  HostLists h;
  h.windows = head[1];
  h.width = head[2];
  h.groups = head[3];
  h.records.resize(static_cast<size_t>(h.windows) * h.width * kListCap * 2);
  h.at.resize(static_cast<size_t>(h.windows) * h.width * kListAt);
  h.count.resize(static_cast<size_t>(h.windows) * h.groups);
  f.read(reinterpret_cast<char*>(h.records.data()), static_cast<std::streamsize>(h.records.size()));
  f.read(reinterpret_cast<char*>(h.at.data()), static_cast<std::streamsize>(h.at.size() * 2));
  f.read(reinterpret_cast<char*>(h.count.data()), static_cast<std::streamsize>(h.count.size() * 2));
  return h;
}

//: THE TENSOR FLOOR a CTA-window: the busiest scheduler's two warps' MMAs (form 0: 8 a 16-token step; form 1: 4 an
//: 8-token tile) at 32 cycles (bf16, fp32 accumulate, sm_86), averaged over the boxes and windows
double floor_of(const HostLists& h, int form, bool dense) {
  const int boxes = h.width / kDigits;
  double total = 0;
  for (int w = 0; w < h.windows; ++w)
    for (int box = 0; box < boxes; ++box) {
      const int passes = dense ? 2 : (h.count[w * h.groups + box] + kSlots - 1) / kSlots;
      long long mmas[kWarps] = {};
      for (int p = 0; p < passes; ++p)
        for (int d = 0; d < kDigits; ++d) {
          const uint16_t* a = &h.at[(static_cast<size_t>(w) * h.width + box * kDigits + d) * kListAt];
          const int n = dense ? kSlots : a[2 * p + 2] - a[2 * p];
          mmas[d % kWarps] += form == 0 ? (n + 15) / 16 * 8 : (n + 7) / 8 * 4;
        }
      long long busiest = 0;
      for (int s = 0; s < 4; ++s) busiest = std::max(busiest, mmas[s] + mmas[s + 4]);
      total += busiest * 32.0;
    }
  return total / (static_cast<double>(h.windows) * boxes);
}

template <int FORM, int OUT, int DENSE, int RING>
void variant(const lab::Bench& b, const Lists& ls, double floor, double floor_m) {
  const std::string name = lab::kernel("read_image::read", {FORM, OUT, DENSE, RING});
  b.expect(name, "HMMA.16816.F32.BF16", ">=", FORM == 0 ? 8 : 4);
  if (b.expect_only) return;
  LAB_CHECK(cudaFuncSetAttribute(read<FORM, OUT, DENSE, RING>, cudaFuncAttributeMaxDynamicSharedMemorySize, kBytes));
  const lab::Samples s = b.variant([&] { read<FORM, OUT, DENSE, RING><<<b.sms, kThreads, kBytes>>>(ls, b.times, b.sinks); },
                                   [&] { return lab::Bench::mean(b.cycles(kWarps), kWarps, 0, kWarps) / ls.windows; });
  const char* forms[2] = {"tokens on M", "tokens on N"};
  const char* outs[3] = {"the MMAs alone", "+ panel adds", "+ panel adds + drain"};
  const std::string label = std::string(DENSE ? "dense / " : "alt-k4 / ") + forms[FORM] + " / " + outs[OUT]
                            + (RING == 2 ? " / first steps straight-line" : RING ? " / state ring" : " / state a digit at a time");
  b.record(label, "cycles a CTA-window", s);
  lab::Samples own, m;
  for (double v : s.v) {
    own.v.push_back(v / floor);
    m.v.push_back(v / floor_m);
  }
  b.record(label + " / its floor", "x", own);
  b.record(label + " / tokens-on-M floor", "x", m);
}

}  // namespace read_image

int main(int argc, char** argv) {
  using namespace read_image;
  lab::Bench b(argc, argv);
  std::string dir;
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], "--inputs")) dir = argv[i + 1];
  Lists ls{};
  double fm = 1, fn = 1, dm = 1, dn = 1;
  if (!b.expect_only) {
    if (dir.empty()) {
      std::fprintf(stderr, "read_image: --inputs <directory> (tools/lab.py writes it)\n");
      return 1;
    }
    const HostLists h = read_lists(dir + "/read.bin");
    uint8_t* records = nullptr;
    uint16_t *at = nullptr, *count = nullptr;
    uint32_t* out = nullptr;
    LAB_CHECK(cudaMalloc(&records, h.records.size()));
    LAB_CHECK(cudaMalloc(&at, h.at.size() * 2));
    LAB_CHECK(cudaMalloc(&count, h.count.size() * 2));
    LAB_CHECK(cudaMalloc(&out, static_cast<size_t>(h.windows) * 2 * kSlots * kWords * 4));
    LAB_CHECK(cudaMemcpy(records, h.records.data(), h.records.size(), cudaMemcpyHostToDevice));
    LAB_CHECK(cudaMemcpy(at, h.at.data(), h.at.size() * 2, cudaMemcpyHostToDevice));
    LAB_CHECK(cudaMemcpy(count, h.count.data(), h.count.size() * 2, cudaMemcpyHostToDevice));
    ls = Lists{reinterpret_cast<const uint16_t*>(records), at, count, h.windows, h.width, h.groups, out};
    fm = floor_of(h, 0, false);
    fn = floor_of(h, 1, false);
    dm = floor_of(h, 0, true);
    dn = floor_of(h, 1, true);
    std::printf("{\"lab\": \"note\", \"floor_m\": %.1f, \"floor_n\": %.1f, \"dense_m\": %.1f, \"dense_n\": %.1f}\n", fm,
                fn, dm, dn);
  }
  variant<0, 1, 0, 0>(b, ls, fm, fm);
  variant<1, 1, 0, 0>(b, ls, fn, fm);
  variant<0, 0, 0, 1>(b, ls, fm, fm);
  variant<0, 1, 0, 1>(b, ls, fm, fm);
  variant<0, 2, 0, 1>(b, ls, fm, fm);
  variant<1, 0, 0, 1>(b, ls, fn, fm);
  variant<1, 1, 0, 1>(b, ls, fn, fm);
  variant<1, 2, 0, 1>(b, ls, fn, fm);
  variant<0, 0, 0, 2>(b, ls, fm, fm);
  variant<0, 1, 0, 2>(b, ls, fm, fm);
  variant<0, 2, 0, 2>(b, ls, fm, fm);
  variant<0, 2, 1, 1>(b, ls, dm, dm);
  variant<1, 2, 1, 1>(b, ls, dn, dm);
}
