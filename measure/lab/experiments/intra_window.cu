// measure/lab/experiments/intra_window.cu -- QUESTION: what does the intra cost a window on hopper as a kernel of
// its own, on real rows -- every pair of a window's 1,024 tokens whose writer is not after its reader, exact against
// the host -- and so what is its share of a call beside the carry's? METHOD: a CTA a tile of 64 READERS, 16 CTAs a
// window, a CTA an SM: one compute warpgroup and one loader warp. The loader stages the readers' rows once and then
// a tile of 64 WRITERS a trip, two slots deep, by TENSOR COPIES -- a level's rows are token-major in global memory
// and a step reads 16 bytes a token a line, so a copy is the box of 64 tokens by one 16-digit block, a kilobyte;
// the values likewise, a box of 64 tokens by 8 channels. The compute warpgroup's trip is `intra_chain`'s tile pair
// on those tiles: each of two levels' pair products as 8 integer steps, the levels multiplied in fp32, the pairs
// whose writer is after its reader dropped (the diagonal tile alone), the weight rounded to bf16 in place as the A
// of 4 value steps 72 columns wide -- the 65th column the den's, a constant brick of ones. A CTA's last act is its
// 64 tokens' lines, f16 pairs added to the output, and their den sums. Reader tile `i` runs `i + 1` trips. TWO
// MORE FORMS hold the rows TILE-MAJOR in global memory -- a tile's blocks one after another, as a step reads them
// -- so a tile's level is one contiguous copy: every plane so, or the levels alone with the values token-major.
// TIMELINE: the loader's trip: the wait on the slot's release two trips back, the barrier's bytes stated, 40 copies
// issued in two instructions (a lane a copy) -- tile-major, 3 contiguous copies in one or two instructions, a lane a
// plane; mixed, 2 and the values' 8 boxes. The compute's trip: the wait on the slot's landing; the fence, 16
// integer steps on descriptors a constant apart, the commit, the wait; 64 conversions, 32 multiplies, the mask's
// selects on the diagonal, 16 rounds; the fence, 4 value steps, the commit, the wait; the slot released. Staged a
// trip ahead: the next writer tile lands while this one is computed. Live: two integer accumulators of 32 through
// the arithmetic, 16 words of A through the value steps' wait, the output's 36 for the CTA.
// see docs/internals/tools/lab.md#intra-window
#include <cuda.h>
#include <dlfcn.h>

#include "lab.cuh"

namespace intra_window {

constexpr int kGroup = 128, kThreads = kGroup + 32;
constexpr int kTile = 64, kWindow = 1024, kTiles = kWindow / kTile, kDigits = 256, kChannels = 64, kLevels = 2;
//: A BLOCK: a tensor copy's box, 64 tokens by 16 bytes, a kilobyte -- 16 digits of a level, or 8 channels' values
constexpr int kBlock = kTile * 16, kLevelBytes = (kDigits / 16) * kBlock, kValueBlocks = kChannels / 8;
//: shared memory: the readers' two levels; two slots of a writer tile -- its two levels, its values' 8 blocks and a
//: ninth of ones, the den's column -- and the barriers
constexpr int kSlotValues = kLevels * kLevelBytes, kSlotBytes = kSlotValues + (kValueBlocks + 1) * kBlock;
constexpr int kReadersAt = 0, kSlotsAt = kLevels * kLevelBytes, kBarsAt = kSlotsAt + 2 * kSlotBytes,
              kSmemBytes = kBarsAt + 48;
//: a landed slot's bytes, the copies' sum, and the barriers: the readers landed, a slot landed and released
constexpr int kSlotLanded = kSlotValues + kValueBlocks * kBlock, kReadersLanded = kLevels * kLevelBytes;
constexpr int kReadersReady = 0, kSlotReady = 8, kSlotFreed = 24;

//: THE CELL: a level's row is 8 live digits of byte 1, so a level's product is at most 8 and a pair's weight at
//: most 64 -- exact in bf16 and in every sum; a value is 1 or 2
constexpr uint32_t digit_of(int level, uint32_t token, int i) {
  return (token * 5u + (token >> 3) + static_cast<uint32_t>(i) * 17u + static_cast<uint32_t>(level) * 3u) & 255u;
}
constexpr int value_of(uint32_t token, int channel) { return static_cast<int>((token + channel) % 2 + 1); }

//: the rows' two layouts in global memory: TOKEN-MAJOR, the bundle's -- a tile lands as tensor copies, a box a
//: block -- and TILE-MAJOR -- a tile of 64 tokens its blocks one after another, each a token's 16 bytes a line, as a
//: step reads them: a tile's level is one contiguous copy
//: and MIXED -- the levels tile-major, the values token-major, as a reader of a token's values at a run would have
//: them
enum Form : int { kBoxes = 0, kTiled = 1, kMixed = 2 };
struct Tiled {
  const uint8_t* levels[kLevels];
  const uint8_t* values;
};

//: a tile's copies into `at`: boxes of the token-major rows a lane each, or the tile-major rows' three runs
template <int FORM>
__device__ __forceinline__ void land(uint32_t at, bool whole, const CUtensorMap* level0, const CUtensorMap* level1,
                                     const CUtensorMap* values, const Tiled& tiled, int tokens, uint32_t bar) {
  const int lane = threadIdx.x & 31;
  if constexpr (FORM != kBoxes) {
    const long long tile = tokens / kTile;
    //: a level a lane, its rows chosen by a select: an index into the argument is local memory
    const uint8_t* const rows = lane ? tiled.levels[1] : tiled.levels[0];
    if (lane < kLevels) lab::bulk_load(at + lane * kLevelBytes, rows + tile * kLevelBytes, kLevelBytes, bar);
    if constexpr (FORM == kTiled) {
      if (whole && lane == kLevels)
        lab::bulk_load(at + kSlotValues, tiled.values + tile * (kValueBlocks * kBlock), kValueBlocks * kBlock, bar);
    } else {
      if (whole && lane < kValueBlocks) lab::tensor_load_2d(at + kSlotValues + lane * kBlock, values, 8 * lane, tokens, bar);
    }
  } else {
    //: a lane a block: level `lane / 16`, digits `16 (lane % 16)` on; the values' 8 blocks a lane of the first 8
    lab::tensor_load_2d(at + lane * kBlock, lane & 16 ? level1 : level0, 16 * (lane & 15), tokens, bar);
    if (whole && lane < kValueBlocks) lab::tensor_load_2d(at + kSlotValues + lane * kBlock, values, 8 * lane, tokens, bar);
  }
}

//: THE LOADER WARP: the readers' rows once, then a writer tile a trip into the slot the compute released
template <int FORM>
__device__ __forceinline__ void load_role(uint32_t base, long long* times, const CUtensorMap* level0,
                                          const CUtensorMap* level1, const CUtensorMap* values, const Tiled& tiled,
                                          int rounds) {
  const int lane = threadIdx.x & 31, mine = blockIdx.x % kTiles;
  const int first = (blockIdx.x / kTiles) * kWindow;
  const uint32_t bars = base + kBarsAt;
  //: the CTA's span is its warps' earliest start to their latest stop: the loader's is inside the compute's
  lab::WarpSpan clock;
  clock.start();
  if (lane == 0) lab::mbar_expect(bars + kReadersReady, kReadersLanded);
  land<FORM>(base + kReadersAt, false, level0, level1, values, tiled, first + mine * kTile, bars + kReadersReady);
  const int trips = rounds * (mine + 1);
#pragma unroll 1
  for (int n = 0; n < trips; ++n) {
    const uint32_t slot = n & 1, at = base + kSlotsAt + slot * kSlotBytes, ready = bars + kSlotReady + 8 * slot;
    const int tokens = first + (n % (mine + 1)) * kTile;
    if (n >= 2) lab::mbar_wait_suspended(bars + kSlotFreed + 8 * slot, ((n >> 1) - 1) & 1);
    if (lane == 0) lab::mbar_expect(ready, kSlotLanded);
    land<FORM>(at, true, level0, level1, values, tiled, tokens, ready);
  }
  clock.stop(times);
}

//: THE COMPUTE WARPGROUP: a trip a writer tile -- the levels' products, the weights, the value steps
__device__ __forceinline__ void compute_role(uint32_t base, long long* times, uint32_t* out, float* den, int rounds) {
  const int tid = threadIdx.x, warp = tid >> 5, q = (tid & 31) >> 2, t = tid & 3;
  const int mine = blockIdx.x % kTiles;
  const long long first = static_cast<long long>(blockIdx.x / kTiles) * kWindow + mine * kTile;
  const uint32_t bars = base + kBarsAt;
  //: THE MASK of the diagonal tile, a thread's: bit `4 j + 2 r + e` set where writer `8 j + 2 t + e` is not after
  //: reader `16 warp + q + 8 r`
  uint32_t kept = 0;
#pragma unroll
  for (int i = 0; i < 32; ++i)
    kept |= static_cast<uint32_t>(8 * (i >> 2) + 2 * t + (i & 1) <= 16 * warp + q + 8 * ((i >> 1) & 1)) << i;
  kept = lab::held(kept);
  int g[kLevels][32];
  float o[36];
  lab::WarpSpan clock;
  clock.start();
  lab::mbar_wait_suspended(bars + kReadersReady, 0);
#pragma unroll 1
  for (int round = 0; round < rounds; ++round) {
#pragma unroll 1
  for (int writers = 0; writers <= mine; ++writers) {
    const int n = round * (mine + 1) + writers;
    const uint32_t slot = n & 1;
    lab::mbar_wait_suspended(bars + kSlotReady + 8 * slot, (n >> 1) & 1);
    const uint32_t tile = lab::held(base + kSlotsAt + slot * kSlotBytes);
    //: THE LEVELS' PAIR PRODUCTS: a row's or a writer's 32 digits two lines a block apart, 8 rows a brick; a level's
    //: first step writes its accumulator whole
    const uint64_t readers = lab::wg_desc(base + kReadersAt, kBlock / 16, 8), written = lab::wg_desc(tile, kBlock / 16, 8);
    lab::wg_fence();
#pragma unroll
    for (int l = 0; l < kLevels; ++l)
#pragma unroll
      for (int s = 0; s < kDigits / 32; ++s) {
        const uint64_t on = static_cast<uint64_t>((l * kLevelBytes + 2 * s * kBlock) / 16);
        lab::wg_u8_n64_ss(g[l], readers + on, written + on, s != 0);
      }
    lab::wg_commit();
    lab::wg_wait<0>();
    //: THE PAIR'S WEIGHT: the levels' products multiplied in fp32, nothing on the diagonal tile where the writer is
    //: after the reader, a bf16 pair a word in the registers a value step reads as A -- writer group `c`'s words
    //: `4 c` on. The conversions are the conversion unit's (docs/lab/intra_chain.md)
    const uint32_t keeps = writers == mine ? kept : 0xffffffffu;
    uint32_t a[16];
#pragma unroll
    for (int w = 0; w < 16; ++w) {
      float pair[2];
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int i = 2 * w + e;
        const float weight = static_cast<float>(g[0][i]) * static_cast<float>(g[1][i]);
        pair[e] = keeps >> i & 1u ? weight : 0.f;
      }
      a[w] = lab::held(lab::bf16x2_round(pair[0], pair[1]));
    }
    //: THE VALUE STEPS, a writer group of 16 each: B a line a writer's 8 channels, a block a brick, the ninth the
    //: ones; a round's first step writes the output's accumulator whole
    const uint64_t valued = lab::wg_desc(tile + kSlotValues, 8, kBlock / 16);
    lab::wg_fence();
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint32_t rows[4] = {a[4 * c], a[4 * c + 1], a[4 * c + 2], a[4 * c + 3]};
      lab::wg_bf16_n72_rs_mn(o, rows, valued + static_cast<uint64_t>(c * 16), writers != 0 || c != 0);
    }
    lab::wg_commit();
    lab::wg_wait<0>();
    if (tid == 0) lab::mbar_arrive(bars + kSlotFreed + 8 * slot);
  }
    //: THE TOKENS' LINES, a round's end: a row's channels `8 j + 2 t` and the next an f16 pair added to its output
    //: line, and the den's sum by the threads that hold column 64
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const long long token = first + 16 * warp + q + 8 * r;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        lab::red_add_f16x2(out + token * (kChannels / 2) + 4 * j + t, lab::pack_half2(o[4 * j + 2 * r], o[4 * j + 2 * r + 1]));
      if (t == 0) lab::red_add_f32(den + token, o[32 + 2 * r]);
    }
  }
  clock.stop(times);
}

template <int FORM>
__global__ void __launch_bounds__(kThreads, 1)
    window(const __grid_constant__ CUtensorMap level0, const __grid_constant__ CUtensorMap level1,
           const __grid_constant__ CUtensorMap values, Tiled tiled, long long* times, uint32_t* out, float* den,
           int rounds) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x;
  const uint32_t base = lab::smem(smem);
  //: the den's column: every line of each slot's ninth block bf16 1 in its first place, nothing after
#pragma unroll 1
  for (int i = tid; i < 2 * kBlock / 2; i += kThreads)
    *reinterpret_cast<uint16_t*>(smem + kSlotsAt + (i / (kBlock / 2)) * kSlotBytes + kSlotLanded + 2 * (i % (kBlock / 2))) =
        (i & 7) == 0 ? static_cast<uint16_t>(0x3f80) : static_cast<uint16_t>(0);
  if (tid == 0) {
    lab::mbar_init(base + kBarsAt + kReadersReady, 1);
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      lab::mbar_init(base + kBarsAt + kSlotReady + 8 * s, 1);
      lab::mbar_init(base + kBarsAt + kSlotFreed + 8 * s, 1);
    }
  }
  lab::bulk_fence();
  __syncthreads();
  //: EACH ROLE'S WHOLE BODY IN ITS OWN BRANCH
  if (tid < kGroup) {
    compute_role(base, times, out, den, rounds);
  } else {
    load_role<FORM>(base, times, &level0, &level1, &values, tiled, rounds);
  }
}

}  // namespace intra_window

int main(int argc, char** argv) {
  using namespace intra_window;
  lab::Bench b(argc, argv);
  for (int form : {static_cast<int>(kBoxes), static_cast<int>(kTiled), static_cast<int>(kMixed)}) {
    const std::string k = lab::kernel("window", {form});
    b.expect(k, "IGMMA", "==", kLevels * kDigits / 32);
    b.expect(k, "HGMMA", "==", 4);
    b.expect(k, form == kBoxes ? "UTMALDG" : "UBLKCP", ">=", form == kMixed ? 2 : 3);
    //: the conversion unit kept in the trip, as hopper's row of the standards allows (KERNEL_STANDARDS 29, item 6)
    b.expect(k, "STEP_XU", ">=", 0);
    //: a lane's own copy and a holding thread's den sum are lane-dependent branches
    b.expect(k, "LOOP_CONVERGENCE", "<=", 8);
    //: the slot's release is one thread's arrival, a branch in the trip
    b.expect(k, "BSSY", ">=", 0);
    //: the tokens' lines are a CTA's last act, outside its trips: the timed call's rounds repeat the CTA, lines and all
    b.expect(k, "STEP_GLOBAL", ">=", 0);
    b.exempt(k, "a kernel's first lab: its budget follows this record");
  }
  if (b.expect_only) return 0;
  const int windows = b.sms / kTiles, blocks = windows * kTiles;
  const size_t tokens = static_cast<size_t>(windows) * kWindow;
  //: THE ROWS, token-major as the bundle holds them: a level's 256 bytes a token, a token's 64 bf16 values
  std::vector<uint8_t> rows[kLevels];
  for (int l = 0; l < kLevels; ++l) {
    rows[l].assign(tokens * kDigits, 0);
    for (size_t x = 0; x < tokens; ++x)
      for (int i = 0; i < 8; ++i) rows[l][x * kDigits + digit_of(l, static_cast<uint32_t>(x), i)] = 1;
  }
  std::vector<uint16_t> vals(tokens * kChannels);
  for (size_t x = 0; x < tokens; ++x)
    for (int c = 0; c < kChannels; ++c) vals[x * kChannels + c] = value_of(static_cast<uint32_t>(x), c) == 1 ? 0x3f80 : 0x4000;
  uint8_t* drows[kLevels] = {nullptr, nullptr};
  uint16_t* dvals = nullptr;
  uint32_t* out = nullptr;
  float* den = nullptr;
  for (int l = 0; l < kLevels; ++l) {
    LAB_CHECK(cudaMalloc(&drows[l], rows[l].size()));
    LAB_CHECK(cudaMemcpy(drows[l], rows[l].data(), rows[l].size(), cudaMemcpyHostToDevice));
  }
  LAB_CHECK(cudaMalloc(&dvals, vals.size() * 2));
  LAB_CHECK(cudaMemcpy(dvals, vals.data(), vals.size() * 2, cudaMemcpyHostToDevice));
  //: the same rows TILE-MAJOR: a tile's block `b` holds each of its 64 tokens' 16 bytes of that block, a line a token
  Tiled tiled{};
  {
    std::vector<uint8_t> t(tokens * kDigits);
    for (int l = 0; l < kLevels; ++l) {
      for (size_t x = 0; x < tokens; ++x)
        for (int d = 0; d < kDigits; ++d)
          t[(x / kTile) * kLevelBytes + (d / 16) * kBlock + (x % kTile) * 16 + d % 16] = rows[l][x * kDigits + d];
      uint8_t* dev = nullptr;
      LAB_CHECK(cudaMalloc(&dev, t.size()));
      LAB_CHECK(cudaMemcpy(dev, t.data(), t.size(), cudaMemcpyHostToDevice));
      tiled.levels[l] = dev;
    }
    std::vector<uint16_t> tv(tokens * kChannels);
    for (size_t x = 0; x < tokens; ++x)
      for (int c = 0; c < kChannels; ++c)
        tv[(x / kTile) * (kValueBlocks * kBlock / 2) + (c / 8) * (kBlock / 2) + (x % kTile) * 8 + c % 8] = vals[x * kChannels + c];
    uint16_t* dev = nullptr;
    LAB_CHECK(cudaMalloc(&dev, tv.size() * 2));
    LAB_CHECK(cudaMemcpy(dev, tv.data(), tv.size() * 2, cudaMemcpyHostToDevice));
    tiled.values = reinterpret_cast<const uint8_t*>(dev);
  }
  const size_t words = tokens * (kChannels / 2);
  LAB_CHECK(cudaMalloc(&out, words * 4));
  LAB_CHECK(cudaMalloc(&den, tokens * 4));
  LAB_CHECK(cudaMemset(out, 0, words * 4));
  LAB_CHECK(cudaMemset(den, 0, tokens * 4));
  //: THE TENSOR MAPS, from the driver (its library is the running one's: no toolkit is linked for it)
  using Encode = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                              const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                              CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  void* const driver = dlopen("libcuda.so.1", RTLD_NOW);
  const auto encode = driver ? reinterpret_cast<Encode>(dlsym(driver, "cuTensorMapEncodeTiled")) : nullptr;
  if (!encode) {
    std::fprintf(stderr, "intra_window: the driver's cuTensorMapEncodeTiled is not reachable\n");
    return 2;
  }
  CUtensorMap maps[3];
  const cuuint32_t steps[2] = {1, 1};
  for (int m = 0; m < 3; ++m) {
    const bool level = m < kLevels;
    const cuuint64_t dims[2] = {static_cast<cuuint64_t>(level ? kDigits : kChannels), tokens},
                     strides[1] = {static_cast<cuuint64_t>(level ? kDigits : 2 * kChannels)};
    const cuuint32_t box[2] = {static_cast<cuuint32_t>(level ? 16 : 8), kTile};
    const CUresult made = encode(&maps[m], level ? CU_TENSOR_MAP_DATA_TYPE_UINT8 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
                                 level ? static_cast<void*>(drows[m]) : static_cast<void*>(dvals), dims, strides, box,
                                 steps, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                 CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (made != CUDA_SUCCESS) {
      std::fprintf(stderr, "intra_window: cuTensorMapEncodeTiled refused map %d (%d)\n", m, static_cast<int>(made));
      return 2;
    }
  }
  const auto run = [&](auto kernel, int form) {
    const std::string name = std::string("a window of 1,024 tokens, two levels of 256 digits, 64 channels and the den, ")
                             + (form == kBoxes ? "the rows token-major: tensor copies a block"
                               : form == kTiled ? "the rows tile-major: a tile's level one copy"
                                                : "the levels tile-major, the values token-major");
    LAB_CHECK(cudaMemset(out, 0, words * 4));
    LAB_CHECK(cudaMemset(den, 0, tokens * 4));
    LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    kernel<<<blocks, kThreads, kSmemBytes>>>(maps[0], maps[1], maps[2], tiled, b.times, out, den, 1);
    LAB_CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> got(words);
    std::vector<float> got_den(tokens);
    LAB_CHECK(cudaMemcpy(got.data(), out, words * 4, cudaMemcpyDeviceToHost));
    LAB_CHECK(cudaMemcpy(got_den.data(), den, tokens * 4, cudaMemcpyDeviceToHost));
    //: THE HOST'S: a pair's weight the product of its levels' common live digits, a reader's sums over the window's
    //: writers not after it; a line's value rounded to f16 as the device rounds it
    long long wrong = 0, wrong_den = 0;
    std::vector<uint64_t> live(tokens * kLevels * 4, 0);
    for (size_t x = 0; x < tokens; ++x)
      for (int l = 0; l < kLevels; ++l)
        for (int i = 0; i < 8; ++i) {
          const uint32_t d = digit_of(l, static_cast<uint32_t>(x), i);
          live[(x * kLevels + l) * 4 + d / 64] |= 1ull << (d % 64);
        }
    const auto common = [&](size_t t, size_t s, int l) {
      int n = 0;
      for (int w = 0; w < 4; ++w) n += __builtin_popcountll(live[(t * kLevels + l) * 4 + w] & live[(s * kLevels + l) * 4 + w]);
      return n;
    };
    for (int w = 0; w < windows; ++w)
      for (int t = 0; t < kWindow; ++t) {
        const size_t reader = static_cast<size_t>(w) * kWindow + t;
        long long sum[2] = {0, 0}, mass = 0;
        for (int s = 0; s <= t; ++s) {
          const size_t writer = static_cast<size_t>(w) * kWindow + s;
          const int first = common(reader, writer, 0);
          if (!first) continue;
          const long long weight = static_cast<long long>(first) * common(reader, writer, 1);
          mass += weight;
          //: a writer's values are 1 and 2 by the parity of token plus channel
          sum[writer & 1] += weight;
        }
        wrong_den += got_den[reader] != static_cast<float>(mass);
        for (int c = 0; c < kChannels; ++c) {
          //: channel c's value of a writer of parity p is `(p + c) % 2 + 1`
          const long long want = sum[0] * (c % 2 + 1) + sum[1] * ((c + 1) % 2 + 1);
          const uint32_t word = got[reader * (kChannels / 2) + c / 2];
          wrong += lab::half_to_float(static_cast<uint16_t>(word >> (16 * (c & 1)))) != static_cast<float>(lab::f16_of(want));
        }
      }
    b.check(name, wrong == 0 && wrong_den == 0,
            std::to_string(wrong) + " of " + std::to_string(tokens * kChannels) + " output values and "
                + std::to_string(wrong_den) + " of " + std::to_string(tokens) + " den sums differ");
    //: a CTA's span is its tile's `i + 1` trips; a window is its 16 CTAs' spans, 136 trips
    constexpr int kRounds = 32;
    const lab::Samples cycles = b.variant(
        [&] { kernel<<<blocks, kThreads, kSmemBytes>>>(maps[0], maps[1], maps[2], tiled, b.times, out, den, kRounds); },
        [&] { return b.span(kThreads >> 5, blocks) * kTiles / kRounds; });
    b.record(name, "SM-cycles a window", cycles);
    lab::Samples token = cycles;
    for (double& v : token.v) v /= kWindow;
    b.record(name + ": a token", "SM-cycles a token", token);
  };
  run(window<kBoxes>, kBoxes);
  run(window<kTiled>, kTiled);
  run(window<kMixed>, kMixed);
  return 0;
}
