// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// measure/lab/experiments/color_union.cu -- THE FACTS PASS'S COLORING, PRICED: a box-window's read union colored by
// residue mod 8 (its pool rows' bank groups) so each digit's tokens spread over the banks (KERNEL_STANDARDS section
// 30), one warp a box-window from its bit tile in shared memory to each union token's slot. The algorithm is
// `color_union.inputs.py`'s `color`, checked exactly. see docs/internals/tools/lab.md#color-union
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../lab.cuh"

namespace color_union {

//: THE GRID: a fixed 64 blocks of 16 warps (one an SM), each warp `kPerWarp` box-windows, so a launch's units are known
//: before it runs
constexpr int kWarps = 16, kThreads = 32 * kWarps, kBlocks = 64, kPerWarp = 4;
constexpr long long kUnits = 1LL * kBlocks * kWarps * kPerWarp;

//: A WARP'S SHARED MEMORY: two tiles (the next box's copy in flight), the single tokens' words, each digit's counts a
//: residue (8 u16), forced counts (8 u16) and ties word, each digit's take (8 u16), each token's residue; the
//: multi-digit tokens' list (u16) lives where the plans go, dead before they are written
constexpr int kTileBytes = 32 * 16 * 4;
constexpr int kSingles = 2 * kTileBytes;
constexpr int kCounts = kSingles + 64;
constexpr int kForced = kCounts + 32 * 16;
constexpr int kList = kForced;
constexpr int kTies = kForced + 32 * 16;
constexpr int kTake = kTies + 32 * 4;
constexpr int kRes = kTake + 32 * 16;
constexpr int kWarpBytes = kRes + 512;
constexpr int kBytes = kWarps * kWarpBytes;
static_assert(kRes - kList >= 512 * 2, "the list holds every token");
static_assert(kBytes <= 99 * 1024, "the block fits sm_86's 99 KB");

//: THE TILE, the facts pass's: digit `d`'s word `j` at `16 d + (j ^ (d / 2 % 16))`, so 32 lanes reading one word of
//: every digit, or one digit's word each of 16 lanes, meet 32 banks
__device__ __forceinline__ int tile_at(int d, int j) { return 16 * d + (j ^ ((d >> 1) & 15)); }

template <int CELL>
__global__ __launch_bounds__(kThreads, 1) void color(const uint32_t* __restrict__ tiles, int boxes,
                                                     uint16_t* __restrict__ slots, uint16_t* __restrict__ extents,
                                                     long long* times, int* sinks) {
  extern __shared__ __align__(16) uint8_t smem[];
  const int warp = lab::warp_id(), lane = lab::lane_id();
  uint8_t* const base = smem + warp * kWarpBytes;
  uint32_t* const singles = reinterpret_cast<uint32_t*>(base + kSingles);
  uint32_t* const row = reinterpret_cast<uint32_t*>(base + kCounts) + 4 * lane;
  uint16_t* const list = reinterpret_cast<uint16_t*>(base + kList);
  uint16_t* const forced = reinterpret_cast<uint16_t*>(base + kForced);
  uint32_t* const ties = reinterpret_cast<uint32_t*>(base + kTies);
  uint16_t* const take = reinterpret_cast<uint16_t*>(base + kTake);
  uint8_t* const res = base + kRes;
  const int first = static_cast<int>(blockIdx.x) * kWarps + warp, stride = static_cast<int>(gridDim.x) * kWarps;
  //: a box's tile into buffer `buf`: lane l copies words l + 32 i (digit (l + 32 i) / 16), coalesced and swizzled
  const auto fetch = [&](int box, int buf) {
    const uint32_t* src = tiles + static_cast<int64_t>(box) * 512;
    const uint32_t dst = lab::smem(base + buf * kTileBytes);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int k = lane + 32 * i;
      lab::cp_async4(dst + 4u * tile_at(k >> 4, k & 15), src + k);
    }
    lab::cp_commit();
  };
  const int r = lane & 7, j = lane & 15, h = lane >> 4;
  //: field r of a pair of count words into bytes 2-3 (`__byte_perm`), the pair (r < 4 ? 0-1 : 2-3) picked after
  const uint32_t field = (2u * (r & 3)) << 8 | (2u * (r & 3) + 1) << 12;
  const int step = stride % boxes;
  int box = first % boxes, next = box + step - (box + step >= boxes ? boxes : 0);
  uint32_t fold = 0;
  lab::WarpSpan span;
  fetch(box, 0);
  span.start();
#pragma unroll 1
  for (int it = 0; it < kPerWarp; ++it) {
    fetch(next, (it + 1) & 1);
    lab::cp_wait<1>();
    __syncwarp();
    const uint32_t* tile = reinterpret_cast<const uint32_t*>(base + (it & 1) * kTileBytes);

    //: THE UNION AND ITS MULTI-DIGIT TOKENS: lane (j, h) folds word j of digits 16 h .., half 1 in the order that keeps
    //: the halves on opposite banks, then the halves meet
    uint32_t once = 0, twice = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const uint32_t x = tile[tile_at(16 * h + (i ^ h), j)];
      twice |= once & x;
      once |= x;
    }
    const uint32_t o = __shfl_xor_sync(~0u, once, 16), w2 = __shfl_xor_sync(~0u, twice, 16);
    const uint32_t uword = once | o, mword = twice | w2 | (once & o);
    if (lane < 16) singles[j] = uword & ~mword;
    //: the multi-digit tokens listed in order: word j's at its prefix
    {
      uint32_t x = lane < 16 ? mword : 0u, at = __popc(x);
#pragma unroll
      for (int s = 1; s < 16; s <<= 1) {
        const uint32_t y = __shfl_up_sync(~0u, at, s, 16);
        at += j >= s ? y : 0u;
      }
      at -= __popc(x);
#pragma unroll 1
      while (x) {
        list[at++] = static_cast<uint16_t>(32 * j + __ffs(x) - 1);
        x &= x - 1;
      }
    }
    const uint32_t multi = __reduce_add_sync(~0u, lane < 16 ? __popc(mword) : 0u);
    __syncwarp();

    //: THE MULTI-DIGIT TOKENS IN ORDER: lane d's counts a residue in registers (two u16 a word); a token's digits sum
    //: theirs, and it takes the residue they use least, then the least loaded (`lr`, residue lane & 7's load x 8 +
    //: itself), then the lowest; the next token's membership loads a step ahead
    uint32_t lr = r, c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    uint32_t t = list[0] & 511u;
    uint32_t in = (tile[tile_at(lane, t >> 5)] >> (t & 31)) & 1u;
#pragma unroll 1
    for (uint32_t i = 0; i < multi; ++i) {
      const uint32_t tn = list[i + 1] & 511u;
      const uint32_t inn = (tile[tile_at(lane, tn >> 5)] >> (tn & 31)) & 1u;
      const uint32_t m = 0u - in;
      const uint32_t s0 = __reduce_add_sync(~0u, c0 & m), s1 = __reduce_add_sync(~0u, c1 & m);
      const uint32_t s2 = __reduce_add_sync(~0u, c2 & m), s3 = __reduce_add_sync(~0u, c3 & m);
      const uint32_t p = r < 4 ? __byte_perm(s0, s1, field) : __byte_perm(s2, s3, field);
      const uint32_t best = __reduce_min_sync(~0u, (p & 0xffff0000u) | lr) & 7u;
      const uint32_t inc = in << ((best & 1) << 4), pair = best >> 1;
      c0 += pair == 0 ? inc : 0u;
      c1 += pair == 1 ? inc : 0u;
      c2 += pair == 2 ? inc : 0u;
      c3 += pair == 3 ? inc : 0u;
      lr += static_cast<uint32_t>(r) == best ? 8u : 0u;
      res[t] = static_cast<uint8_t>(best);
      t = tn;
      in = inn;
    }
    *reinterpret_cast<uint4*>(row) = make_uint4(c0, c1, c2, c3);
    uint32_t load = lr >> 3;
    __syncwarp();

    //: EACH DIGIT'S PLAN (lane d): its single tokens' words and count, then the water-fill over its counts -- the level
    //: every residue is filled to (`forced`) and the singles left for that level's residues (`ties`)
    uint32_t sw[16], k = 0;
#pragma unroll
    for (int q = 0; q < 16; ++q) {
      sw[q] = tile[tile_at(lane, q)] & singles[q];
      k += __popc(sw[q]);
    }
    const uint4 cw = *reinterpret_cast<const uint4*>(row);
    const uint32_t cc[8] = {cw.x & 0xffffu, cw.x >> 16, cw.y & 0xffffu, cw.y >> 16,
                            cw.z & 0xffffu, cw.z >> 16, cw.w & 0xffffu, cw.w >> 16};
    uint32_t level = cc[0], placed = 0;
#pragma unroll
    for (int q = 1; q < 8; ++q) level = min(level, cc[q]);
#pragma unroll 1
    for (;;) {
      uint32_t a = 0, next = ~0u;
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        a += cc[q] <= level;
        next = cc[q] > level ? min(next, cc[q]) : next;
      }
      if (next != ~0u && placed + a * (next - level) <= k) {
        placed += a * (next - level);
        level = next;
        continue;
      }
      const uint32_t q = (k - placed) / a;
      level += q;
      placed += q * a;
      break;
    }
    uint32_t active = 0, fw[4];
#pragma unroll
    for (int q = 0; q < 8; q += 2) {
      active |= (cc[q] <= level ? 1u : 0u) << q | (cc[q + 1] <= level ? 1u : 0u) << (q + 1);
      fw[q / 2] = (level > cc[q] ? level - cc[q] : 0u) | (level > cc[q + 1] ? level - cc[q + 1] : 0u) << 16;
    }
    *reinterpret_cast<uint4*>(forced + 8 * lane) = make_uint4(fw[0], fw[1], fw[2], fw[3]);
    ties[lane] = k == 0 ? 0u : (k - placed) | active << 16 | 1u << 24;
    __syncwarp();

    //: THE TIES, digit by digit: a level's singles go to its residues the least loaded first (lanes as residues)
#pragma unroll 1
    for (int d = 0; d < 32; ++d) {
      const uint32_t tr = ties[d];
      if (tr == 0) continue;
      uint32_t n = forced[8 * d + r];
      const uint32_t rem = tr & 0xffffu;
      if (rem != 0) {
        const bool act = (tr >> (16 + r)) & 1u;
        const uint32_t key = act ? (load << 3 | r) : 0xffffu;
        uint32_t rank = 0;
#pragma unroll
        for (int i = 1; i < 8; ++i) rank += __shfl_sync(~0u, key, (lane & ~7) | ((lane + i) & 7)) < key;
        n += act && rank < rem;
      }
      load += n;
      if (lane < 8) take[8 * d + r] = static_cast<uint16_t>(n);
    }
    __syncwarp();

    //: EACH DIGIT'S SINGLES (lane d), in token order, dealt to its take's residues in residue order
    {
      const uint4 tk = *reinterpret_cast<const uint4*>(take + 8 * lane);
      const uint32_t b1 = tk.x & 0xffffu, b2 = b1 + (tk.x >> 16), b3 = b2 + (tk.y & 0xffffu), b4 = b3 + (tk.y >> 16);
      const uint32_t b5 = b4 + (tk.z & 0xffffu), b6 = b5 + (tk.z >> 16), b7 = b6 + (tk.w & 0xffffu);
      uint32_t idx = 0;
#pragma unroll
      for (int q = 0; q < 16; ++q) {
        uint32_t x = sw[q];
#pragma unroll 1
        while (x) {
          const int bit = __ffs(x) - 1;
          x &= x - 1;
          const uint32_t rr = (b1 <= idx) + (b2 <= idx) + (b3 <= idx) + (b4 <= idx) + (b5 <= idx) + (b6 <= idx) +
                              (b7 <= idx);
          res[32 * q + bit] = static_cast<uint8_t>(rr);
          ++idx;
        }
      }
    }
    __syncwarp();

    //: THE SLOTS: a token's residue plus 8 x its rank among the residue's tokens, 32 tokens a word; lane r keeps
    //: residue r's running count
    uint16_t* const out = slots + static_cast<int64_t>(box) * 512;
    uint32_t count = 0;
#pragma unroll 4
    for (int q = 0; q < 16; ++q) {
      const uint32_t u = __shfl_sync(~0u, uword, q);
      const bool in = (u >> lane) & 1u;
      const uint32_t rr = res[32 * q + lane] & 7u;
      const uint32_t v0 = __ballot_sync(~0u, in && (rr & 1u)), v1 = __ballot_sync(~0u, in && (rr & 2u));
      const uint32_t v2 = __ballot_sync(~0u, in && (rr & 4u));
      const uint32_t same = u & (rr & 1u ? v0 : ~v0) & (rr & 2u ? v1 : ~v1) & (rr & 4u ? v2 : ~v2);
      const uint32_t at = __shfl_sync(~0u, count, static_cast<int>(rr));
      out[32 * q + lane] = in ? static_cast<uint16_t>(rr + 8 * (at + __popc(same & ((1u << lane) - 1)))) : 0xffffu;
      count += __popc(u & (r & 1 ? v0 : ~v0) & (r & 2 ? v1 : ~v1) & (r & 4 ? v2 : ~v2));
    }
    const uint32_t widest = __reduce_max_sync(~0u, count);
    if (lane == 0) extents[box] = static_cast<uint16_t>(8 * widest);
    fold ^= widest + count + load;
    box = next;
    next += step - (next + step >= boxes ? boxes : 0);
    __syncwarp();
  }
  lab::cp_wait<0>();
  span.stop(times);
  lab::sink(sinks, fold);
}

//: a cell's tiles and expected slots, host and device
struct Cell {
  int boxes = 0;
  uint32_t* tiles = nullptr;
  uint16_t *slots = nullptr, *extents = nullptr;
  std::vector<uint16_t> expected;
};

Cell load_cell(const std::string& dir) {
  Cell c;
  FILE* f = std::fopen((dir + "/tiles.bin").c_str(), "rb");
  int head[4] = {};
  if (!f || std::fread(head, 4, 4, f) != 4 || head[0] != 0x524F4C43) {
    std::fprintf(stderr, "color_union: %s/tiles.bin is not this lab's\n", dir.c_str());
    std::exit(1);
  }
  c.boxes = head[1];
  std::vector<uint32_t> t(static_cast<size_t>(c.boxes) * 512);
  if (std::fread(t.data(), 4, t.size(), f) != t.size()) std::exit(1);
  std::fclose(f);
  c.expected.resize(static_cast<size_t>(c.boxes) * 513);
  f = std::fopen((dir + "/expected.bin").c_str(), "rb");
  if (!f || std::fread(c.expected.data(), 2, c.expected.size(), f) != c.expected.size()) std::exit(1);
  std::fclose(f);
  LAB_CHECK(cudaMalloc(&c.tiles, t.size() * 4));
  LAB_CHECK(cudaMemcpy(c.tiles, t.data(), t.size() * 4, cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMalloc(&c.slots, static_cast<size_t>(c.boxes) * 512 * 2));
  LAB_CHECK(cudaMalloc(&c.extents, static_cast<size_t>(c.boxes) * 2));
  return c;
}

//: THE BUDGET LINE a box-window (KERNEL_STANDARDS section 29), by class: stated before the code at ~45 instructions a
//: multi-digit token and ~1.15K fixed, RESTATED from the counts at ~64 and ~2.6K -- the misses itemized at
//: docs/internals/tools/lab.md#color-union; the cells' multi-digit tokens ~40, ~317, ~0.1
struct Line {
  const char* cell;
  double shuffle, load, store, select, integer, control, global;
};

constexpr Line kLines[3] = {{"alt-k4", 430, 203, 113, 1325, 3650, 680, 38},
                            {"alt-k16", 1815, 760, 405, 4370, 13420, 1525, 38},
                            {"thinned 1/16", 125, 102, 27, 500, 1390, 410, 38}};

template <int CELL>
void variant(const lab::Bench& b, const Cell& c) {
  const std::string name = lab::kernel("color_union::color", {CELL});
  b.expect(name, "REDUX", ">=", 6);
  b.expect(name, "LDGSTS", ">=", 16);
  b.units(name, "box-window", kUnits);
  const Line& l = kLines[CELL];
  b.budget(name, "shuffle", l.shuffle);
  b.budget(name, "shared load", l.load);
  b.budget(name, "shared store", l.store);
  b.budget(name, "select/compare", l.select);
  b.budget(name, "integer/address", l.integer);
  b.budget(name, "control", l.control);
  b.budget(name, "global", l.global);
  if (b.expect_only) return;
  LAB_CHECK(cudaFuncSetAttribute(color<CELL>, cudaFuncAttributeMaxDynamicSharedMemorySize, kBytes));
  const auto launch = [&] {
    color<CELL><<<kBlocks, kThreads, kBytes>>>(c.tiles, c.boxes, c.slots, c.extents, b.times, b.sinks);
  };
  const lab::Samples s = b.variant(launch, [&] { return b.span(kWarps, kBlocks) / (kWarps * kPerWarp); });
  b.record(std::string("an SM's cycles a box-window, 16 warps / ") + l.cell, "cycles", s);
  const lab::Samples w = b.variant(launch, [&] {
    std::vector<long long> t(2 * kBlocks * kWarps);
    LAB_CHECK(cudaMemcpy(t.data(), b.times, t.size() * sizeof(long long), cudaMemcpyDeviceToHost));
    double sum = 0;
    for (int i = 0; i < kBlocks * kWarps; ++i) sum += double(t[2 * i + 1] - t[2 * i]);
    return sum / (kBlocks * kWarps * kPerWarp);
  });
  b.record(std::string("a warp's cycles a box-window / ") + l.cell, "cycles", w);
  LAB_CHECK(cudaMemset(c.slots, 0xAB, static_cast<size_t>(c.boxes) * 512 * 2));
  LAB_CHECK(cudaMemset(c.extents, 0xAB, static_cast<size_t>(c.boxes) * 2));
  launch();
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> got(static_cast<size_t>(c.boxes) * 513);
  LAB_CHECK(cudaMemcpy(got.data(), c.slots, static_cast<size_t>(c.boxes) * 512 * 2, cudaMemcpyDeviceToHost));
  LAB_CHECK(cudaMemcpy(got.data() + static_cast<size_t>(c.boxes) * 512, c.extents, static_cast<size_t>(c.boxes) * 2,
                       cudaMemcpyDeviceToHost));
  size_t bad = 0;
  for (size_t i = 0; i < got.size(); ++i) bad += got[i] != c.expected[i];
  const std::string detail = std::to_string(c.boxes) + " box-windows' slots and extents, " + std::to_string(bad) +
                             " of " + std::to_string(got.size()) + " differ from the mirror";
  b.check(std::string("a warp's cycles a box-window / ") + l.cell, bad == 0, detail);
  b.check(std::string("an SM's cycles a box-window, 16 warps / ") + l.cell, bad == 0, detail);
}

}  // namespace color_union

int main(int argc, char** argv) {
  using namespace color_union;
  lab::Bench b(argc, argv);
  std::string dir;
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], "--inputs")) dir = argv[i + 1];
  Cell cells[3];
  if (!b.expect_only) {
    if (dir.empty()) {
      std::fprintf(stderr, "color_union: --inputs <directory> (tools/lab.py writes it)\n");
      return 1;
    }
    if (b.sms * lab::Bench::kMaxWarps < 2 * kBlocks * kWarps) {
      std::fprintf(stderr, "color_union: %d SMs hold too few timer slots for %d blocks\n", b.sms, kBlocks);
      return 1;
    }
    cells[0] = load_cell(dir + "/alt-k4");
    cells[1] = load_cell(dir + "/alt-k16");
    cells[2] = load_cell(dir + "/thin");
  }
  variant<0>(b, cells[0]);
  variant<1>(b, cells[1]);
  variant<2>(b, cells[2]);
}
