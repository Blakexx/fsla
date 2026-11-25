// measure/lab/experiments/carry_supply.cu -- QUESTION: does the LIBRARY's carry CTA, its fold warpgroups' SUPPLY and
// COEFFICIENTS built and every other component a stub, list each of its sets' live tokens from a call's real operands
// -- the masks, the mask-preparation pass's prepared words, the densities the carve is derived from, the write side's
// rows -- land their value lines in its rows and expand their coefficients into its tiles, exactly; and what does a
// window cost a CTA with them running?
// METHOD: this file compiles the library's device code (`carry/carry_kernel.cuh`, and the mask-preparation pass's
// kernel, which it launches first). A call of one head: two levels of 256 digits, 64 value channels, the write side
// masked on level 0 at 4 to 5 live digits a token and whole on level 1 (THE CELL), or masked on both (level 1 a run of
// 40 digits a token), or whole on both (no mask of its own: every token live in a fold's one set), or one digit a
// token on each level (sets that are mostly empty), or masked on both with the read side's densities crossed with
// the write side's (the carve alone changes); the read side's densities the cell's mirror, its masks never read (its role is a stub). The grid
// is the head's CTAs. CHECKED through the kernel's tracer, which waits for a round's copies and copies its tokens and
// rows out: every round of every set of every fold is there once, its tokens the set's list in order, its rows their
// value lines in the step's swizzle order, a dead place's row zero; every coefficient of every kept round's chunks is
// its levels' bytes multiplied in fp32 and rounded to bf16 once, a place with no token zeros; a set's chunks' columns
// are its leaves, each once; and on the host, the sets' leaves tile the leaf space once. TIMED with the library's tracer (nothing): a call of 64 windows.
// TIMELINE: no MMA. A fold warp, a window: the folds' hand-offs, then each of its sets -- the set's word a lane (the
// levels' group words ANDed, loaded a set ahead), the listing's running sums, the set's tokens laid in the warp's list (a
// trip a token of the fullest word), and a round of 32 places at a time its 8 shared loads and 8 asynchronous copies a
// lane into the warp's rows; a chunk of a round is a STAGE, its copies one group -- the chunk's bytes a level, a
// lane a token, and with a round's first chunk its value lines -- and a stage issues the next one's copies, waits
// for its own, then expands: a lane its staged bytes' shared loads, their conversion, a multiplication a column, a
// packed rounding a pair and a 16-byte store a brick into the warp's coefficient tile. Two groups in flight. Live:
// the next set's word, the listing's four, a round's 8 tokens.
// see docs/internals/tools/lab.md#carry-supply
#include <algorithm>
#include <cstring>
#include <map>
#include <tuple>

#include "lab.cuh"
#include "carry/carry_kernel.cuh"
#include "mask_prepare/mask_prepare_kernel.cuh"

namespace carry_supply {

namespace carry = fsla::carry::detail;
namespace pass = fsla::mask_prepare::detail;
namespace plan = fsla::plan;
namespace ops = ::denseref::ops;
using Bt = ::denseref::arch::BuildTag<::denseref::arch::Build{900}>;
//: the registers a fold thread and a readout thread take: the lab's split (docs/lab/carry_token.md)
constexpr int kFoldRegs = 176, kReadRegs = 104;
constexpr int kWindow = plan::window_of_cc(plan::kHopperCc), kWords = kWindow / 32;
constexpr int kLevels = 2, kWidthBits = 8, kDigits = 1 << kWidthBits;
constexpr int kLine = 2 * carry::kChannels, kRound = carry::kRoundBytes<carry::kChannels>;
constexpr int kFoldWarps = carry::kFolds * carry::kGroupWarps;
//: a kept round: a head (its set, window, round, whether its rows are off their boundary; its 32 tokens), then its rows
constexpr int kHead = 256, kTokensAt = 64, kEntry = kHead + kRound, kKept = 24;
//: a kept round's coefficient tile, a chunk: the widest step's
constexpr int kCoefs = carry::kWarpCoefs;
//: the digits a token is live on, on the level masked second: a run from a start its own
constexpr int kRun = 40;

//: token `x` has 4 or 5 live digits, a stride apart from a start its own
constexpr int live_of(uint32_t x) { return 4 + (x % 3 == 0); }
constexpr uint32_t digit_of(uint32_t x, int i) { return (x * 5u + (x >> 3) + static_cast<uint32_t>(i) * 37u) & 255u; }
inline uint8_t value_byte(size_t i) { return static_cast<uint8_t>((i * 2654435761ull >> 13) | 1u); }
//: a live digit's byte of a level's row: never zero
constexpr uint8_t row_byte(size_t token, int digit, int level) {
  return static_cast<uint8_t>((token * 131u + static_cast<size_t>(digit) * 31u + static_cast<size_t>(level) * 17u) % 255u + 1u);
}
//: where token `x`'s byte on `digit` lies in a level's rows, tile-major: a tile of 64 tokens, its 16-digit blocks one after another, a token's 16 bytes a line
constexpr size_t row_at(size_t x, int digit) {
  return (x >> carry::kRowTileBits) * (static_cast<size_t>(kDigits) << carry::kRowTileBits)
         + static_cast<size_t>(digit >> carry::kRowBlockBits) * carry::kRowBlockBytes
         + ((x & ((1u << carry::kRowTileBits) - 1)) << carry::kRowBlockBits) + (digit & ((1 << carry::kRowBlockBits) - 1));
}

//: THE CHECK'S TRACER: a round's copies waited for, then its tokens and rows copied out, a lane its 4 bytes of a row
struct Keep {
  uint8_t* out;
  uint8_t* tiles;
  int chunks;
  int n;
  __device__ __forceinline__ void wait_begin(int) {}
  __device__ __forceinline__ void wait_end(int) {}
  __device__ __forceinline__ void arrive(int) {}
  __device__ __forceinline__ void phase(int) {}
  __device__ __forceinline__ void round(int, int set, int window, int round, const uint16_t* tokens, uint32_t rows) {
    ops::stage_wait<0>();
    //: every lane's copies have landed before any lane reads a row
    __syncwarp();
    const int lane = lab::lane_id(), warp = lab::warp_id();
    //: no branch on the count: a round past the kept ones is written over the last, and the count fails the check
    const int slot = n < kKept ? n : kKept - 1;
    uint8_t* const to = out + ((static_cast<size_t>(blockIdx.x) * kFoldWarps + warp) * kKept + slot) * kEntry;
    const uint8_t* const base = carry::Smem<Bt>::template get<carry::FoldRows>();
    const uint8_t* const from = base + (rows - ops::smem_addr(base));
    int* const head = reinterpret_cast<int*>(to);
    //: the head's four words, each written alike by the lanes of its turn
    const int which = lane & 3;
    head[which] = which == 0 ? set : which == 1 ? window : which == 2 ? round : static_cast<int>((rows & 1023u) != 0u);
    const int listed = tokens[lane];
    reinterpret_cast<int*>(to + kTokensAt)[lane] = listed == carry::kNoToken ? -1 : listed;
#pragma unroll 1
    for (int k = 0; k < kRound / kLine; ++k) {
      const int i = kLine * k + 4 * lane;
      *reinterpret_cast<uint32_t*>(to + kHead + i) = *reinterpret_cast<const uint32_t*>(from + i);
    }
    ++n;
    __syncwarp();
  }
  //: a chunk's coefficient tile copied out as its warp stored it, a lane its 4 bytes of a line of the banks
  __device__ __forceinline__ void coefs(int, int, int, int, int chunk, const uint16_t*, uint32_t tile, int bytes) {
    __syncwarp();
    const int lane = lab::lane_id(), warp = lab::warp_id();
    const int slot = n < kKept ? n : kKept - 1;
    const size_t entry = (static_cast<size_t>(blockIdx.x) * kFoldWarps + warp) * kKept + slot;
    uint8_t* const to = tiles + (entry * chunks + chunk) * kCoefs;
    const uint8_t* const base = carry::Smem<Bt>::template get<carry::FoldCoefs>();
    const uint8_t* const from = base + (tile - ops::smem_addr(base));
#pragma unroll 1
    for (int k = 0; k < bytes / 128; ++k) {
      const int i = 128 * k + 4 * lane;
      *reinterpret_cast<uint32_t*>(to + i) = *reinterpret_cast<const uint32_t*>(from + i);
    }
    __syncwarp();
  }
};

//: THE LIBRARY'S CTA, its rounds kept
__global__ void __launch_bounds__(carry::kThreads, 1)
    kept(const __grid_constant__ carry::Operands o, uint8_t* out, uint8_t* tiles, int chunks, int* count) {
  Keep tr{out, tiles, chunks, 0};
  carry::carry_cta<Bt, kFoldRegs, kReadRegs, Keep>(o, tr);
  if (lab::lane_id() == 0 && lab::warp_id() < kFoldWarps) count[blockIdx.x * kFoldWarps + lab::warp_id()] = tr.n;
}

//: a phase's site in a warp's timeline: above every barrier's
constexpr uint32_t kPhases = 0xc000u;

//: THE KERNEL'S TRACER AS WARPLINE'S STAMPS: a wait's site is its number, a phase's `kPhases` on
template <bool ON>
struct Stamps {
  lab::Trace<ON> row;
  __device__ __forceinline__ void wait_begin(int id) { row.stamp(warpline::kWaitBegin, static_cast<uint32_t>(id)); }
  __device__ __forceinline__ void wait_end(int id) { row.stamp(warpline::kWaitEnd, static_cast<uint32_t>(id)); }
  __device__ __forceinline__ void arrive(int id) { row.stamp(warpline::kArrive, static_cast<uint32_t>(id)); }
  __device__ __forceinline__ void phase(int site) { row.mark(warpline::kPhaseBegin, kPhases + static_cast<uint32_t>(site)); }
  __device__ __forceinline__ void round(int, int, int, int, const uint16_t*, uint32_t) {}
  __device__ __forceinline__ void coefs(int, int, int, int, int, const uint16_t*, uint32_t, int) {}
};

//: THE LIBRARY'S CTA: the clock is the lab's, the work the library's; its warps' timelines stamped where `TRACE`
template <int TRACE>
__global__ void __launch_bounds__(carry::kThreads, 1)
    timed(long long* times, unsigned long long* trace, const __grid_constant__ carry::Operands o) {
  Stamps<TRACE != 0> tr;
  tr.row.bind(trace);
  lab::WarpSpan clock;
  clock.start();
  carry::carry_cta<Bt, kFoldRegs, kReadRegs, Stamps<TRACE != 0>>(o, tr);
  clock.stop(times);
  tr.row.close(0);
}

//: A CALL on the host: the write side's masks (none where a level has none), the values, the density, the carve
struct Call {
  int windows = 0;
  size_t tokens = 0;
  std::vector<uint32_t> mask[kLevels];
  std::vector<uint8_t> values;
  //: the write side's rows a level
  std::vector<uint8_t> level[kLevels];
  uint8_t density[2 * kLevels] = {};
  plan::PrefillCarve carve{};
  int ctas = 1;
};

//: THE CALLS, by the write side's masks: the cell (level 0 at 4 to 5 live digits a token, level 1 whole); both levels (level 1
//: a run of `kRun` digits too); whole (no mask: the dense end); one digit a token on each level (the sparse end)
//: crossed: both levels' masks, the read side's densities the write side's crossed (level 0's the write side's of level 1), so that the levels weigh alike
enum class Shape { kCell, kBoth, kWhole, kSparse, kCrossed };

inline Call make(Shape shape, int windows) {
  Call c;
  c.windows = windows;
  c.tokens = static_cast<size_t>(windows) * kWindow;
  const size_t words = c.tokens / 32;
  //: the cell's level-0 mask, whose byte is the read side's density on level 1 in every call
  std::vector<uint32_t> cell(words * kDigits, 0);
  if (shape != Shape::kWhole) c.mask[0].assign(words * kDigits, 0);
  if (shape == Shape::kBoth || shape == Shape::kSparse || shape == Shape::kCrossed) c.mask[1].assign(words * kDigits, 0);
  const int run = shape == Shape::kSparse ? 1 : kRun;

  for (size_t x = 0; x < c.tokens; ++x) {
    const uint32_t t = static_cast<uint32_t>(x), bit = 1u << (x % 32);
    const size_t at = (x / 32) * kDigits;
    for (int i = 0; i < live_of(t); ++i) {
      cell[at + digit_of(t, i)] |= bit;
      if (!c.mask[0].empty() && (i == 0 || shape != Shape::kSparse)) c.mask[0][at + digit_of(t, i)] |= bit;
    }
    if (!c.mask[1].empty())
      for (int i = 0; i < run; ++i) c.mask[1][at + ((t * 7u + static_cast<uint32_t>(i)) & 255u)] |= bit;
  }
  //: a level-side's byte as the library's helpers count it: the live (token, digit) pairs over all, in 255ths, rounded
  const auto byte = [&](const std::vector<uint32_t>& m) {
    long long live = 0;
    for (const uint32_t w : m) live += __builtin_popcount(w);
    const long long total = static_cast<long long>(c.tokens) * kDigits;
    return static_cast<uint8_t>((2 * 255 * live + total) / (2 * total));
  };
  //: row 0 the read side's, the cell's mirror (whole on level 0, sparse on level 1); row 1 the write side's, a level with no mask whole
  c.density[0] = 255;
  c.density[1] = byte(cell);
  c.density[kLevels] = c.mask[0].empty() ? 255 : byte(c.mask[0]);
  c.density[kLevels + 1] = c.mask[1].empty() ? 255 : byte(c.mask[1]);
  if (shape == Shape::kCrossed) {
    c.density[0] = c.density[kLevels + 1];
    c.density[1] = c.density[kLevels];
  }
  const int width_bits[kLevels] = {kWidthBits, kWidthBits};
  const int read[kLevels] = {c.density[0] + 1, c.density[1] + 1};
  const int write[kLevels] = {c.density[kLevels] + 1, c.density[kLevels + 1] + 1};
  c.carve = plan::prefill_carve(kLevels, width_bits, write, read);
  for (int l = 0; l < kLevels; ++l) c.ctas <<= kWidthBits - c.carve.carve_bits[l];
  c.values.resize(c.tokens * kLine);
  for (size_t i = 0; i < c.values.size(); ++i) c.values[i] = value_byte(i);
  //: a level's rows: a digit the mask has live a byte that is not zero, any other zero; a level with no mask every digit
  for (int l = 0; l < kLevels; ++l) {
    c.level[l].assign(c.tokens * kDigits, 0);
    for (size_t x = 0; x < c.tokens; ++x)
      for (int digit = 0; digit < kDigits; ++digit) {
        const bool live = c.mask[l].empty() || (c.mask[l][(x / 32) * kDigits + digit] >> (x % 32) & 1u) != 0;
        if (live) c.level[l][row_at(x, digit)] = row_byte(x, digit, l);
      }
  }
  return c;
}

//: THE CALL ON THE DEVICE: its operands, the mask-preparation pass run on them
struct Device {
  carry::Operands o{};
  std::vector<void*> held;
};

inline Device up(const Call& c) {
  Device d;
  const auto put = [&](const void* from, size_t bytes) {
    void* p = nullptr;
    LAB_CHECK(cudaMalloc(&p, bytes < 16 ? 16 : bytes));
    if (from != nullptr) LAB_CHECK(cudaMemcpy(p, from, bytes, cudaMemcpyHostToDevice));
    d.held.push_back(p);
    return p;
  };
  pass::Args a{};
  int side_bits[pass::kSides] = {};
  std::int64_t heads[pass::kSides] = {};
  for (int l = 0; l < kLevels; ++l) {
    if (c.mask[l].empty()) continue;
    //: the write side's slots are the second four
    const int s = pass::kSides / 2 + l;
    a.mask[s] = static_cast<const uint32_t*>(put(c.mask[l].data(), c.mask[l].size() * 4));
    a.width_bits[s] = side_bits[s] = kWidthBits;
    a.heads[s] = 1;
    heads[s] = 1;
    d.o.mask[s] = a.mask[s];
  }
  a.words = static_cast<int>(c.tokens / 32);
  a.windows = c.windows;
  a.levels = kLevels;
  for (int l = 0; l < kLevels; ++l) a.level_bits[l] = kWidthBits;
  const plan::PreparedLayout layout = plan::prepared_layout(side_bits, heads, c.windows, kWords);
  std::int64_t ctas = 0;
  for (int s = 0; s < pass::kSides; ++s) {
    a.out[s] = d.o.prepared[s] = layout.offset[s];
    a.first_cta[s] = static_cast<int>(ctas);
    ctas += pass::ctas_of(heads[s], c.windows, side_bits[s]);
  }
  a.first_cta[pass::kSides] = static_cast<int>(ctas);
  uint8_t* const density = static_cast<uint8_t*>(put(c.density, sizeof(c.density)));
  uint8_t* const work = static_cast<uint8_t*>(put(nullptr, static_cast<size_t>(layout.bytes)));
  pass::mask_prepare_kernel<Bt><<<static_cast<unsigned>(ctas), pass::kThreads>>>(a, density, work);
  LAB_CHECK(cudaDeviceSynchronize());
  d.o.workspace = work;
  d.o.density = density;
  d.o.values = static_cast<const uint8_t*>(put(c.values.data(), c.values.size()));
  for (int l = 0; l < kLevels; ++l)
    d.o.rows[pass::kSides / 2 + l] = static_cast<const uint8_t*>(put(c.level[l].data(), c.level[l].size()));
  d.o.tokens = static_cast<std::int64_t>(c.tokens);
  d.o.ctas = c.ctas;
  d.o.line_bytes = kLine;
  for (int l = 0; l < kLevels; ++l) d.o.width_bits[l] = kWidthBits;
  d.o.levels = kLevels;
  d.o.words = static_cast<int>(c.tokens / 32);
  d.o.windows = c.windows;
  return d;
}

inline void down(Device& d) {
  for (void* p : d.held) LAB_CHECK(cudaFree(p));
  d.held.clear();
}

//: A FOLD'S SET on the host: a level's group (its tokens are live on any of its digits) and its digits inside the fold
struct Set {
  int group_first[kLevels], group_count[kLevels], first[kLevels], count[kLevels];
};

inline Set set_of(const plan::PrefillCarve& c, const int* at, int fold, int set) {
  Set b{};
  int part[kLevels] = {};
  //: the set's index, the write side's densest level varying fastest
  for (int i = c.levels - 1; i >= 0; --i) {
    const int l = c.write_order[i], bits = std::max(0, c.fold_bits[l] - c.write_group_bits[l]);
    part[l] = set & ((1 << bits) - 1);
    set >>= bits;
  }
  for (int l = 0; l < kLevels; ++l) {
    const int gb = c.write_group_bits[l];
    const int origin = (at[l] << c.carve_bits[l]) + (l == c.split_level ? fold << c.fold_bits[l] : 0);
    const int group = (origin >> gb) + part[l];
    b.group_first[l] = group << gb;
    b.group_count[l] = 1 << gb;
    b.first[l] = std::max(group << gb, origin);
    b.count[l] = std::min((group + 1) << gb, origin + (1 << c.fold_bits[l])) - b.first[l];
  }
  return b;
}

}  // namespace carry_supply

//: a call's rows: its supply checked over `checked` windows, then timed at 64, plain and with every warp's timeline stamped
void run(lab::Bench& b, unsigned long long* rows, const std::string& name, carry_supply::Shape shape, int checked) {
  using namespace carry_supply;
  constexpr int kTimed = 64, kWarps = carry::kThreads / 32;
  const Call c = make(shape, checked);
  const plan::PrefillCarve& cv = c.carve;
  if (c.ctas > b.sms) {
    std::fprintf(stderr, "the call's %d CTAs a head are more than the card's %d multiprocessors\n", c.ctas, b.sms);
    std::exit(1);
  }
  Device d = up(c);
  const size_t entries = static_cast<size_t>(c.ctas) * kFoldWarps * kKept;
  //: the call's class, as a fold thread derives it: its step's columns a position and its sets' chunks
  carry::Operands host{};
  for (int l = 0; l < kLevels; ++l) {
    host.rows[pass::kSides / 2 + l] = c.level[l].data();
    host.width_bits[l] = kWidthBits;
  }
  host.tokens = static_cast<std::int64_t>(c.tokens);
  int none[plan::kCarveMaxLevels] = {};
  const carry::Place origin{};
  const int fold_class = carry::fold_positions(host, cv, origin, 0, none, none).fold_class;
  if (fold_class >= carry::kFoldClassCount) {
    std::fprintf(stderr, "the call's fold has no class\n");
    std::exit(1);
  }
  const carry::FoldClass folded = carry::kFoldClasses[fold_class];
  const int chunks = 1 << folded.chunk_bits;
  int columns = 1, column_at[kLevels + 1] = {};
  for (int at = 0; at < kLevels; ++at) {
    column_at[at + 1] = column_at[at] + folded.bits[at];
    columns <<= folded.bits[at];
  }
  const int step_bytes = columns / carry::kBrickColumns * carry::kBrickBytes;
  uint8_t* dout = nullptr;
  uint8_t* dtiles = nullptr;
  int* dcount = nullptr;
  LAB_CHECK(cudaMalloc(&dout, entries * kEntry));
  LAB_CHECK(cudaMalloc(&dtiles, entries * chunks * kCoefs));
  LAB_CHECK(cudaMalloc(&dcount, static_cast<size_t>(c.ctas) * kFoldWarps * sizeof(int)));
  LAB_CHECK(cudaMemset(dout, 0xff, entries * kEntry));
  LAB_CHECK(cudaMemset(dtiles, 0xff, entries * chunks * kCoefs));
  kept<<<c.ctas, carry::kThreads>>>(d.o, dout, dtiles, chunks, dcount);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<uint8_t> got(entries * kEntry);
  std::vector<uint8_t> tiles(entries * chunks * kCoefs);
  LAB_CHECK(cudaMemcpy(tiles.data(), dtiles, tiles.size(), cudaMemcpyDeviceToHost));
  std::vector<int> count(static_cast<size_t>(c.ctas) * kFoldWarps);
  LAB_CHECK(cudaMemcpy(got.data(), dout, got.size(), cudaMemcpyDeviceToHost));
  LAB_CHECK(cudaMemcpy(count.data(), dcount, count.size() * sizeof(int), cudaMemcpyDeviceToHost));
  //: a level's groups' words on the host: the OR of a group's digits' words
  const size_t words = c.tokens / 32;
  std::vector<uint32_t> grouped[kLevels];
  for (int l = 0; l < kLevels; ++l) {
    if (c.mask[l].empty()) continue;
    const int gb = cv.write_group_bits[l], groups = kDigits >> gb;
    grouped[l].assign(words * groups, 0);
    for (size_t j = 0; j < words; ++j)
      for (int digit = 0; digit < kDigits; ++digit) grouped[l][j * groups + (digit >> gb)] |= c.mask[l][j * kDigits + digit];
  }
  std::vector<int> tiled(static_cast<size_t>(kDigits) * kDigits, 0);
  long long wrong_tokens = 0, wrong_bytes = 0, unknown = 0, twice = 0, missing = 0, off = 0, spilled = 0, rounds = 0;
  long long wrong_coefs = 0, misplaced = 0;
  for (int cta = 0; cta < c.ctas; ++cta) {
    int at[plan::kCarveMaxLevels] = {};
    plan::prefill_cta_at(cv, cta, at);
    for (int fold = 0; fold < carry::kFolds; ++fold) {
      const int sets = fold < cv.folds ? cv.fold_sets : 0;
      //: the fold's positions as its threads form them, and which level a position is
      carry::Place place{};
      for (int l = 0; l < kLevels; ++l) place.cta[l] = at[l];
      int base[plan::kCarveMaxLevels], shift[plan::kCarveMaxLevels], mask[plan::kCarveMaxLevels];
      carry::fold_levels(cv, place, fold, base, shift, mask);
      const carry::FoldPositions positions = carry::fold_positions(host, cv, place, fold, shift, mask);
      int level_of[kLevels];
      for (int i = 0; i < kLevels; ++i) level_of[i] = positions.rows[i] == c.level[0].data() ? 0 : 1;
      //: a chunk's first digit a position, and a set's chunks' columns its leaves, each once
      const auto digit_of_chunk = [&](int set, int chunk, int i) {
        const int extent = folded.bits[i] + (i == 0 ? folded.chunk_bits : 0);
        return positions.first[i] + (((set >> positions.shift[i]) & positions.mask[i]) << extent)
               + (i == 0 ? chunk << folded.bits[0] : 0);
      };
      for (int set = 0; set < sets; ++set) {
        const Set box = set_of(cv, at, fold, set);
        std::map<std::pair<int, int>, int> leaves;
        for (int chunk = 0; chunk < chunks; ++chunk)
          for (int n = 0; n < columns; ++n) {
            int digit[kLevels];
            for (int i = 0; i < kLevels; ++i)
              digit[level_of[i]] = digit_of_chunk(set, chunk, i) + ((n >> column_at[i]) & ((1 << folded.bits[i]) - 1));
            ++leaves[{digit[0], digit[1]}];
          }
        for (int x = 0; x < box.count[0]; ++x)
          for (int y = 0; y < box.count[1]; ++y) {
            const auto it = leaves.find({box.first[0] + x, box.first[1] + y});
            misplaced += it == leaves.end() || it->second != 1;
          }
        misplaced += static_cast<int>(leaves.size()) != box.count[0] * box.count[1];
      }
      //: each set's list a window, from the masks
      std::vector<std::vector<std::vector<int>>> lists(sets, std::vector<std::vector<int>>(c.windows));
      for (int set = 0; set < sets; ++set) {
        const Set box = set_of(cv, at, fold, set);
        for (int x = 0; x < box.count[0]; ++x)
          for (int y = 0; y < box.count[1]; ++y) ++tiled[static_cast<size_t>(box.first[0] + x) * kDigits + box.first[1] + y];
        for (size_t j = 0; j < words; ++j) {
          uint32_t live = ~0u;
          for (int l = 0; l < kLevels; ++l) {
            if (cv.write_group_bits[l] == cv.width_bits[l]) continue;
            const int gb = cv.write_group_bits[l];
            live &= grouped[l][j * (kDigits >> gb) + (box.group_first[l] >> gb)];
          }
          for (int bit = 0; bit < 32; ++bit)
            if ((live >> bit & 1u) != 0) lists[set][j / kWords].push_back(static_cast<int>(j % kWords) * 32 + bit);
        }
      }
      //: warps that share a set split a word's tokens by bit: a set's list a part, the part a warp's
      int set_bits = 0;
      while ((1 << set_bits) < sets) ++set_bits;
      const int shared_bits = std::min(set_bits, carry::kGroupWarpBits), parts = carry::kGroupWarps >> shared_bits;
      const auto own = [&](int set, int window, int part) {
        std::vector<int> list;
        for (const int token : lists[set][window])
          if (token % 32 / (32 / parts) == part) list.push_back(token);
        return list;
      };
      std::map<std::tuple<int, int, int>, int> seen;
      for (int warp = 0; warp < carry::kGroupWarps; ++warp) {
        const size_t mine = static_cast<size_t>(cta) * kFoldWarps + fold * carry::kGroupWarps + warp;
        spilled += count[mine] > kKept;
        for (int e = 0; e < std::min(count[mine], kKept); ++e) {
          const uint8_t* const entry = got.data() + (mine * kKept + e) * kEntry;
          const int* const head = reinterpret_cast<const int*>(entry);
          const int set = head[0], window = head[1], round = head[2];
          const int part = warp >> shared_bits;
          if (set < 0 || set >= sets || window < 0 || window >= c.windows || round < 0) {
            ++unknown;
            continue;
          }
          const std::vector<int> list = own(set, window, part);
          if (kWords * round >= static_cast<int>(list.size())) {
            ++unknown;
            continue;
          }
          ++seen[{set * parts + part, window, round}];
          off += head[3] != 0;
          const int* const tokens = reinterpret_cast<const int*>(entry + kTokensAt);
          for (int place = 0; place < kWords; ++place) {
            const int i = kWords * round + place;
            const int token = i < static_cast<int>(list.size()) ? list[i] : -1;
            wrong_tokens += tokens[place] != token;
            const size_t line = (static_cast<size_t>(window) * kWindow + static_cast<size_t>(token < 0 ? 0 : token)) * kLine;
            for (int piece = 0; piece < carry::kChannels / carry::kPieceChannels; ++piece) {
              //: the place's row, and the piece where the step's swizzle turns it by the row's three bits
              const int turned = piece ^ (place & 7);
              for (int k = 0; k < carry::kPieceBytes; ++k) {
                const uint8_t want = token < 0 ? 0 : c.values[line + piece * carry::kPieceBytes + k];
                wrong_bytes += entry[kHead + place * kLine + turned * carry::kPieceBytes + k] != want;
              }
            }
            //: the place's coefficient line a chunk: a column its levels' bytes multiplied in fp32, rounded to bf16 once; no token, zeros
            const size_t x = static_cast<size_t>(window) * kWindow + static_cast<size_t>(token < 0 ? 0 : token);
            for (int chunk = 0; chunk < chunks; ++chunk) {
              const uint8_t* const tile = tiles.data() + ((mine * kKept + e) * chunks + chunk) * kCoefs;
              for (int n = 0; n < columns; ++n) {
                float product = token < 0 ? 0.0f : 1.0f;
                for (int i = 0; i < kLevels; ++i) {
                  const int digit = digit_of_chunk(set, chunk, i) + ((n >> column_at[i]) & ((1 << folded.bits[i]) - 1));
                  product *= static_cast<float>(c.level[level_of[i]][row_at(x, digit)]);
                }
                uint32_t bits;
                std::memcpy(&bits, &product, sizeof(bits));
                const uint16_t want = static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
                const size_t byte = static_cast<size_t>(place / carry::kStepTokens) * step_bytes
                                    + static_cast<size_t>(n / carry::kBrickColumns) * carry::kBrickBytes
                                    + static_cast<size_t>(place % carry::kStepTokens) * 2 * carry::kBrickColumns
                                    + static_cast<size_t>(n % carry::kBrickColumns) * 2;
                uint16_t have;
                std::memcpy(&have, tile + byte, sizeof(have));
                wrong_coefs += have != want;
              }
            }
          }
        }
      }
      for (int set = 0; set < sets; ++set)
        for (int window = 0; window < c.windows; ++window)
          for (int part = 0; part < parts; ++part) {
            const int tokens = static_cast<int>(own(set, window, part).size());
            for (int round = 0; kWords * round < tokens; ++round) {
              const auto it = seen.find({set * parts + part, window, round});
              missing += it == seen.end();
              twice += it != seen.end() && it->second > 1;
              ++rounds;
            }
          }
    }
  }
  long long untiled = 0;
  for (const int v : tiled) untiled += v != 1;
  const bool ok = wrong_tokens == 0 && wrong_bytes == 0 && wrong_coefs == 0 && misplaced == 0 && unknown == 0 && twice == 0
                  && missing == 0 && off == 0 && spilled == 0 && untiled == 0 && rounds > 0;
  b.check(name, ok,
          std::to_string(rounds) + " rounds of " + std::to_string(c.ctas) + " CTAs: " + std::to_string(missing) + " missing, "
              + std::to_string(twice) + " twice, " + std::to_string(unknown) + " not a set's, " + std::to_string(wrong_tokens)
              + " tokens, " + std::to_string(wrong_bytes) + " row bytes and " + std::to_string(wrong_coefs)
              + " coefficients differ, " + std::to_string(misplaced) + " leaves not once in their set's columns, " + std::to_string(off)
              + " rows off a boundary, " + std::to_string(spilled) + " warps past the kept rounds; " + std::to_string(untiled)
              + " leaves not in one set; the carve " + std::to_string(1 << cv.carve_bits[0]) + " x "
              + std::to_string(1 << cv.carve_bits[1]) + ", write groups " + std::to_string(1 << cv.write_group_bits[0])
              + " and " + std::to_string(1 << cv.write_group_bits[1]) + ", " + std::to_string(cv.fold_sets) + " sets a fold");
  down(d);
  LAB_CHECK(cudaFree(dout));
  LAB_CHECK(cudaFree(dtiles));
  LAB_CHECK(cudaFree(dcount));
  const Call t = make(shape, kTimed);
  Device dt = up(t);
  const std::string plain = name + ", timed", stamped = name + ", every warp's timeline stamped";
  b.unchecked(plain, "the row above checks this kernel's supply through its tracer; this is its time with the library's tracer");
  const lab::Samples cycles = b.variant([&] { timed<0><<<t.ctas, carry::kThreads>>>(b.times, rows, dt.o); },
                                        [&] { return b.span(kWarps, t.ctas) / kTimed; });
  b.record(plain, "cycles a window a CTA", cycles);
  b.unchecked(stamped, "the first row checks this kernel's supply; this is its timeline, its waits and phases stamped");
  const lab::Samples traced = b.variant([&] { timed<1><<<t.ctas, carry::kThreads>>>(b.times, rows, dt.o); },
                                        [&] { return b.span(kWarps, t.ctas) / kTimed; });
  b.record(stamped, "cycles a window a CTA", traced);
  std::vector<std::string> roles;
  for (int w = 0; w < kWarps; ++w) {
    const int thread = 32 * w;
    roles.push_back(thread < carry::kReadAt ? "fold box " + std::to_string(thread / carry::kGroup)
                    : thread < carry::kDrainAt ? "readout" : "drain");
  }
  std::vector<std::pair<uint32_t, std::string>> legend;
  for (int box = 0; box < carry::kFolds; ++box)
    legend.push_back({static_cast<uint32_t>(carry::kFoldId + box), "fold box " + std::to_string(box) + "'s own barrier"});
  legend.push_back({static_cast<uint32_t>(carry::kReadId), "the readout's own barrier"});
  for (int k = 0; k < carry::kChunks; ++k) {
    legend.push_back({static_cast<uint32_t>(carry::kTilesReadyId + k), "tile group " + std::to_string(k) + " published"});
    legend.push_back({static_cast<uint32_t>(carry::kTilesReadId + k), "tile group " + std::to_string(k) + " read"});
  }
  for (int p = 0; p < carry::kPanels; ++p) {
    legend.push_back({static_cast<uint32_t>(carry::kPanelReadyId + p), "panel " + std::to_string(p) + " filled"});
    legend.push_back({static_cast<uint32_t>(carry::kPanelFreedId + p), "panel " + std::to_string(p) + " drained"});
  }
  legend.insert(legend.end(), {{kPhases + carry::kSiteEntry, "entry"}, {kPhases + carry::kSiteExit, "exit"},
                               {kPhases + carry::kSiteWindow, "window"}, {kPhases + carry::kSiteSet, "set"},
                               {kPhases + carry::kSiteListed, "listed"}, {kPhases + carry::kSiteWord, "word"},
                               {kPhases + carry::kSiteFetch, "fetch"}, {kPhases + carry::kSiteIssued, "issued"},
                               {kPhases + carry::kSiteLanded, "landed"},
                               {kPhases + carry::kSiteCoefs, "coefficients"}});
  b.trace(stamped, rows, kWarps, {0, t.ctas / 2}, roles, legend);
  down(dt);
}

int main(int argc, char** argv) {
  using namespace carry_supply;
  lab::Bench b(argc, argv);
  if (b.expect_only) return 0;
  unsigned long long* rows = nullptr;
  const size_t trace_words = static_cast<size_t>(b.sms) * (carry::kThreads / 32) * lab::Trace<true>::kRow;
  LAB_CHECK(cudaMalloc(&rows, trace_words * 8));
  LAB_CHECK(cudaMemset(rows, 0, trace_words * 8));
  //: the second call has 2 sets a warp a window, a set a round and seldom 2: its check keeps 2 windows, about 4 rounds a warp of the 24 kept
  //: the whole call has 8 full rounds a warp a window; the sparse call's sets are mostly empty
  using carry_supply::Shape;
  run(b, rows, "the cell, the write side masked on level 0: the folds' supply and coefficients, the rest stubs", Shape::kCell, 4);
  run(b, rows, "the write side masked on both levels: the folds' supply and coefficients, the rest stubs", Shape::kBoth, 2);
  run(b, rows, "the write side whole on both levels: the folds' supply and coefficients, the rest stubs", Shape::kWhole, 2);
  run(b, rows, "the write side one digit a token on each level: the folds' supply and coefficients, the rest stubs", Shape::kSparse, 4);
  run(b, rows, "the write side masked on both levels, the read side's densities crossed with it: the folds' supply and coefficients, the rest stubs",
      Shape::kCrossed, 4);
  return 0;
}
