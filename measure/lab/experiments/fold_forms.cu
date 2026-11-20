// measure/lab/experiments/fold_forms.cu -- QUESTION: the fold's step by where its coefficients are -- a TILE in
// shared memory, a step one sparse digit's 16 tokens (today's form, its absent steps not issued and its short steps
// padded by the layout), or the warpgroup's REGISTERS, a step 16 tokens by a GROUP of four sparse digits, a warp a
// digit, the values the step's one shared-memory operand -- what does a window cost a CTA of two boxes in each?
// METHOD: a CTA an SM of two fold warpgroups, a box each of 16 lists of 12 to 24 entries (256 a window), 16
// accumulator sets of 8 or 4 of 32 in registers. A box's two packets land once, before the clock, a bulk
// copy each into its own buffer, and its windows take them in turn (no copy is in a window). TILE: a thread expands its
// entries (a dense line and a sparse byte each, loaded from the packet's header) to two lines of the entry's step's
// tile; the fence, the warpgroup's barrier; a list's one or two steps in a rolled loop. REGISTER: a step's 16
// positions are four CLASSES of four (a thread holds two dense rows of positions 2t, 2t + 1, 8 + 2t, 9 + 2t), the
// layout gives a digit's tokens classes of their own; a position's dense line and a step's cells (a thread's four
// positions' sparse bytes for its warp's digit, the classes its warp has) land with the values; a thread reads its
// cell and its positions' two dense bytes a step ahead, multiplies, converts and packs four words, and the step is
// issued -- every class computed, or only those its warp has; a commit group is four steps, each its own words, and
// the wait on it opens the next. The packets land before the clock. Every accumulator is checked exactly.
// TIMELINE: a warpgroup's trip is a window: (TILE: its entries' loads, 16 products and two stores an entry, the
// fence, the barrier;) the wait on its values; its steps, each (REGISTER) its four words formed from the stage loaded
// before, the next stage's reads, the fence; the commit, the wait. Live: 128
// accumulators a thread; TILE three entries' lines and bytes; REGISTER a stage's cell and four dense pairs.
// see docs/internals/tools/lab.md#fold-forms
#include "lab.cuh"

namespace fold_forms {

constexpr int kLists = 16, kStep = 16, kChannels = 64, kGroup = 128, kThreads = 2 * kGroup, kRing = 2, kHeld = 4;
//: THE LANDING IS NOT IN THE WINDOW: a box's two packets land once, before the clock, and its windows take them in
//: turn -- a window's landed bytes at the bulk copy's rate (33 cycles a kilobyte: `state_move`) are longer than
//: either form's fold, and from 60 MB of packets they were the card's memory bandwidth. Every box takes one of 8
//: columns' two
constexpr int kColumns = 8;
enum Form { kTile, kRegister, kRegisterSkip, kTileTable };
//: a window's positions a box (a plane's lines), its value bytes, and a packet: the values and a header
constexpr int kRun = 368, kValueBytes = 8 * kRun * 16, kMaxSteps = 24;
//: WHAT LANDS BESIDE THE VALUES in the register form: a position's dense line (16 bytes, a digit's byte at `2g + j`
//: for row `g + 8j`), then a step's 16 cells of 8 bytes (warp, t): the thread's four positions' sparse bytes for the
//: warp's digit and the classes the warp has
constexpr int kDenseBytes = kRun * 16, kCellBytes = kMaxSteps * 128, kLandBytes = kValueBytes + kDenseBytes + kCellBytes;
//: A PACKET: what lands, then a header read from global memory. THE TILE FORM'S: a list's steps (a byte, 16), the
//: entries' sparse bytes and dense lines (384: a thread's three). THE REGISTER FORM'S: a group's steps (a byte, 4)
//: and their sum
constexpr int kHeaderBytes = 8192, kPacketBytes = kLandBytes + kHeaderBytes, kTileSparse = 32, kTileDense = kTileSparse + 384,
              kTileToken = kTileDense + 384 * 16;
static_assert(kTileToken + 384 * 2 <= kHeaderBytes);
//: THE LEVEL, for the form that fetches its dense lines itself: 65,536 tokens' rows as the bundle holds them (rola
//: docs/ARCHITECTURE.md section 1), TILE-MAJOR -- a tile of 64 tokens its 16 blocks one after another, a block a
//: token's 16 bytes a line: 16 MB
constexpr int kLevelTokens = 65536;
constexpr size_t level_line(uint32_t token, int block) { return (static_cast<size_t>(token >> 6) * 16 + block) * 64 + (token & 63); }
//: SHARED MEMORY: a box's two landing buffers (the tile form's the values alone), its 32 tiles of 512 bytes, the
//: buffers' barriers
template <int FORM> constexpr bool kTiled = FORM == kTile || FORM == kTileTable;
template <int FORM> constexpr int kLand = kTiled<FORM> ? kValueBytes : kLandBytes;
template <int FORM> constexpr int kTilesAt = 4 * kLand<FORM>;
constexpr int kTileBytes = 32 * 512;
template <int FORM> constexpr int kBarsAt = kTilesAt<FORM> + (kTiled<FORM> ? 2 * kTileBytes : 0);
template <int FORM> constexpr int kShared = kBarsAt<FORM> + 64;
static_assert(kShared<0> <= 232448 && kShared<1> <= 232448);

//: THE CELL: a list's entries (they sum to 256), turned by the (CTA, box, ring window)
constexpr int kLens[kLists] = {20, 13, 22, 13, 18, 13, 24, 13, 19, 13, 21, 13, 17, 12, 13, 12};
constexpr int len_of(int turn, int j) { return kLens[(j + turn) % kLists]; }
//: an entry's value at a channel, its dense digit's byte, its sparse byte: small integers, exact in bf16 and fp32
constexpr uint32_t value_of(uint32_t u, int ch) { return ((u * 7u + ch * 5u + 3u) * 2654435761u >> 13) & 3u; }
//: an entry's token (in its window's 1,024 of the level) and its box's dense block; a dense byte is the level's
constexpr uint32_t token_of(uint32_t u) { return ((u / (16 * 32)) % 64) * 1024 + (((u * 2654435761u) >> 12) & 1023u); }
constexpr int block_of(uint32_t u) { return static_cast<int>((u / (2 * 16 * 32)) % 16); }
constexpr uint32_t level_of(uint32_t token, int block, int digit) {
  return (((token * 16u + block) * 11u + digit * 3u + 1u) * 2246822519u >> 11) & 3u;
}
constexpr uint32_t dense_of(uint32_t u, int digit) { return level_of(token_of(u), block_of(u), digit); }
constexpr uint32_t sparse_of(uint32_t u) { return 1u + ((u * 13u + 5u) * 40503u >> 7) % 3u; }
constexpr uint32_t entry_of(int column, int ring, int j, int n) {
  return static_cast<uint32_t>(((column * kRing + ring) * kLists + j) * 32 + n);
}

template <int FORM>
__global__ void __launch_bounds__(kThreads, 1)
    fold(long long* times, const uint8_t* packets, const uint8_t* level, float* state, int windows) {
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t base = lab::smem(smem);
  const int tid = threadIdx.x, box = tid >> 7, in = tid & (kGroup - 1), warp = in >> 5, lane = tid & 31, g = lane >> 2,
            t = lane & 3;
  const uint32_t bars = base + kBarsAt<FORM> + 16 * box;
  if (tid == 0) {
#pragma unroll
    for (int p = 0; p < 4; ++p) lab::mbar_init(base + kBarsAt<FORM> + 8 * p, 1);
  }
  lab::bulk_fence();
  __syncthreads();
  const int column = blockIdx.x * 2 + box;
  const uint8_t* const mine = packets + static_cast<size_t>(column % kColumns) * kRing * kPacketBytes;
  //: window `w`'s values into the buffer of its parity, the warpgroup's first thread's one copy
  const auto issue = [&](int w) {
    if (in == 0) {
      lab::mbar_expect(bars + 8 * (w & 1), kLand<FORM>);
      lab::bulk_load(base + (2 * box + (w & 1)) * kLand<FORM>, mine + static_cast<size_t>(w % kRing) * kPacketBytes,
                     kLand<FORM>, bars + 8 * (w & 1));
    }
  };
  issue(0);
  if (windows > 1) issue(1);
  lab::WarpSpan clock;
  clock.start();
  if constexpr (kTiled<FORM>) {
    float d[kLists][8];
    const uint64_t tiles = lab::wg_desc(base + kTilesAt<FORM> + box * kTileBytes, 8, 16);
    constexpr int kMine = 3;
    uint4 dense[kMine];
    uint32_t sparse[kMine];
    uint32_t token[kMine];
    //: a window's tokens, for the form that fetches its own dense lines: loaded a window before its lines are asked
    const auto load_tokens = [&](int w) {
      const uint8_t* const header = mine + static_cast<size_t>(w % kRing) * kPacketBytes + kLandBytes;
#pragma unroll
      for (int n = 0; n < kMine; ++n) token[n] = reinterpret_cast<const uint16_t*>(header + kTileToken)[in + n * kGroup];
    };
    const auto load_entries = [&](int w) {
      const uint8_t* const header = mine + static_cast<size_t>(w % kRing) * kPacketBytes + kLandBytes;
#pragma unroll
      for (int n = 0; n < kMine; ++n) {
        if constexpr (FORM == kTileTable)
          dense[n] = reinterpret_cast<const uint4*>(level)[level_line(token[n], (column % kColumns) % 16)];
        else
          dense[n] = reinterpret_cast<const uint4*>(header + kTileDense)[in + n * kGroup];
        sparse[n] = header[kTileSparse + in + n * kGroup];
      }
    };
    if constexpr (FORM == kTileTable) load_tokens(0);
    load_entries(0);
    if constexpr (FORM == kTileTable) load_tokens(1);
#pragma unroll 1
    for (int w = 0; w < windows; ++w) {
      const uint8_t* const header = mine + static_cast<size_t>(w % kRing) * kPacketBytes + kLandBytes;
      const uint4 steps_of = *reinterpret_cast<const uint4*>(header);
      //: THE FILL: a thread's three entries (a window's 384 places, the last of them no step's), each its 16
      //: products as two lines of its step's tile at its place; their lines and bytes were loaded a window ahead
#pragma unroll
      for (int n = 0; n < kMine; ++n) {
        const int e = in + n * kGroup;
        const uint32_t word[4] = {dense[n].x, dense[n].y, dense[n].z, dense[n].w};
        uint32_t line[2][4];
#pragma unroll
        for (int b = 0; b < 2; ++b)
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const uint32_t pair = word[2 * b + k / 2] >> (16 * (k & 1));
            line[b][k] = lab::bf16x2_trunc(static_cast<float>((pair & 255u) * sparse[n]),
                                           static_cast<float>(((pair >> 8) & 255u) * sparse[n]));
          }
        uint8_t* const at = smem + kTilesAt<FORM> + box * kTileBytes + (e >> 4) * 512 + (e & 15) * 16;
        *reinterpret_cast<uint4*>(at) = make_uint4(line[0][0], line[0][1], line[0][2], line[0][3]);
        *reinterpret_cast<uint4*>(at + 256) = make_uint4(line[1][0], line[1][1], line[1][2], line[1][3]);
      }
      load_entries(w + 1);
      if constexpr (FORM == kTileTable) load_tokens(w + 2);
      lab::bulk_fence();
      lab::bar_sync(1 + box, kGroup);
      if (w < kRing) lab::mbar_wait(bars + 8 * (w & 1), 0);
      const uint64_t lines = lab::wg_desc(base + (2 * box + (w & 1)) * kLand<FORM>, 8, kRun);
      const uint32_t counts[4] = {steps_of.x, steps_of.y, steps_of.z, steps_of.w};
      int at = 0;
#pragma unroll
      for (int j = 0; j < kLists; ++j) {
        const int n = static_cast<int>((counts[j / 4] >> (8 * (j & 3))) & 255u);
        //: a list's one or two steps: its tokens' value lines where they landed, its tile
#pragma unroll 1
        for (int r = 0; r < n; ++r, ++at) {
          lab::wg_fence();
          lab::wg_bf16_n16_ss_mn(d[j], lines + static_cast<uint64_t>(16 * at), tiles + static_cast<uint64_t>(32 * at),
                                 w != 0 || r != 0);
        }
      }
      lab::wg_commit();
      lab::wg_wait<0>();
    }
    clock.stop(times);
#pragma unroll
    for (int j = 0; j < kLists; ++j)
#pragma unroll
      for (int i = 0; i < 8; ++i) state[(static_cast<size_t>(column) * kGroup + in) * 128 + 8 * j + i] = d[j][i];
  } else {
    float d[4][32];
    //: a step's stage from its landed buffer: the thread's cell (four sparse bytes, the warp's classes), and the two
    //: dense bytes of its two rows at each of its four positions -- or only where its warp has the class
    const auto stage_of = [&](const uint8_t* landed, int at, uint32_t (&cell)[2], uint32_t (&pair)[4]) {
      const uint2 loaded = *reinterpret_cast<const uint2*>(landed + kValueBytes + kDenseBytes + at * 128 + (4 * warp + t) * 8);
      cell[0] = loaded.x;
      cell[1] = loaded.y;
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const uint8_t* const from = landed + kValueBytes + (at * kStep + 8 * (p >> 1) + 2 * t + (p & 1)) * 16 + 2 * g;
        if constexpr (FORM == kRegisterSkip) {
          //: a class the warp has not keeps the pair before: its step's words skip it by the same bit
          if (((cell[1] >> p) & 1u) != 0) pair[p] = *reinterpret_cast<const uint16_t*>(from);
        } else {
          pair[p] = *reinterpret_cast<const uint16_t*>(from);
        }
      }
    };
#pragma unroll 1
    for (int w = 0; w < windows; ++w) {
      const uint8_t* const header = mine + static_cast<size_t>(w % kRing) * kPacketBytes + kLandBytes;
      const uint32_t steps_of = *reinterpret_cast<const uint32_t*>(header);
      const int last = static_cast<int>(*reinterpret_cast<const uint32_t*>(header + 4)) - 1;
      const uint8_t* const landed = smem + (2 * box + (w & 1)) * kLand<FORM>;
      if (w < kRing) lab::mbar_wait(bars + 8 * (w & 1), 0);
      const uint64_t lines = lab::wg_desc(base + (2 * box + (w & 1)) * kLand<FORM>, 8, kRun);
      //: A STAGE AHEAD: a step's cell and dense pairs are read under the step before
      uint32_t cell[2], pair[4] = {0, 0, 0, 0};
      stage_of(landed, 0, cell, pair);
      int at = 0;
#pragma unroll
      for (int group = 0; group < 4; ++group) {
        const int n = static_cast<int>((steps_of >> (8 * group)) & 255u);
        //: A COMMIT GROUP IS `kHeld` STEPS, EACH ITS OWN FOUR WORDS, and a group's last steps one each: a step's
        //: registers are the tensor unit's until its group's wait, so a word written again under a step in flight is
        //: read wrong. No step is under a branch: the groups are two rolled loops
        const auto step = [&](uint32_t (&a)[4], bool add) {
          //: THE STEP'S COEFFICIENTS: word `2q + j` is row `g + 8j` at positions `8q + 2t` and the next
#pragma unroll
          for (int q = 0; q < 2; ++q)
#pragma unroll
            for (int j = 0; j < 2; ++j) {
              float lo = 0.0f, hi = 0.0f;
              if (FORM != kRegisterSkip || ((cell[1] >> (2 * q)) & 1u) != 0)
                lo = static_cast<float>(((pair[2 * q] >> (8 * j)) & 255u) * ((cell[0] >> (16 * q)) & 255u));
              if (FORM != kRegisterSkip || ((cell[1] >> (2 * q + 1)) & 1u) != 0)
                hi = static_cast<float>(((pair[2 * q + 1] >> (8 * j)) & 255u) * ((cell[0] >> (16 * q + 8)) & 255u));
              a[2 * q + j] = lab::bf16x2_trunc(lo, hi);
            }
          stage_of(landed, min(at + 1, last), cell, pair);
          lab::wg_fence();
          lab::wg_bf16_n64_rs_mn(d[group], a, lines + static_cast<uint64_t>(16 * at), add);
          ++at;
        };
        int r = 0;
#pragma unroll 1
        for (; r + kHeld <= n; r += kHeld) {
          uint32_t a[kHeld][4];
#pragma unroll
          for (int k = 0; k < kHeld; ++k) step(a[k], w != 0 || r + k != 0);
          lab::wg_commit();
          lab::wg_wait<0>();
        }
#pragma unroll 1
        for (; r < n; ++r) {
          uint32_t a[4];
          step(a, w != 0 || r != 0);
          lab::wg_commit();
          lab::wg_wait<0>();
        }
      }
    }
    clock.stop(times);
#pragma unroll
    for (int group = 0; group < 4; ++group)
#pragma unroll
      for (int i = 0; i < 32; ++i)
        state[(static_cast<size_t>(column) * kGroup + in) * 128 + 32 * group + i] = d[group][i];
  }
}

}  // namespace fold_forms

namespace {

using namespace fold_forms;

uint16_t bf16_of(uint32_t v) {
  const float f = static_cast<float>(v);
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  return static_cast<uint16_t>(bits >> 16);
}

//: position `at`'s value lines of a packet: an entry's, or a pad's zeros
void lay_values(uint8_t* packet, int at, bool there, uint32_t u) {
  for (int ch = 0; ch < kChannels; ++ch) {
    const uint16_t v = there ? bf16_of(value_of(u, ch)) : 0;
    std::memcpy(packet + (static_cast<size_t>(ch / 8) * kRun + at) * 16 + 2 * (ch & 7), &v, 2);
  }
}

//: a (column, ring window)'s TILE packet: a list's steps one after another, a short step's places pads
void build_tile(uint8_t* packet, int column, int ring) {
  int at = 0;
  for (int j = 0; j < kLists; ++j) {
    const int n = len_of(column + ring, j), steps = (n + kStep - 1) / kStep;
    packet[kLandBytes + j] = static_cast<uint8_t>(steps);
    for (int e = 0; e < steps * kStep; ++e, ++at) {
      const uint32_t u = entry_of(column, ring, j, e);
      lay_values(packet, at, e < n, u);
      packet[kLandBytes + kTileSparse + at] = static_cast<uint8_t>(e < n ? sparse_of(u) : 0);
      const uint16_t token = static_cast<uint16_t>(token_of(u));
      std::memcpy(packet + kLandBytes + kTileToken + 2 * at, &token, 2);
      for (int digit = 0; digit < 16; ++digit)
        packet[kLandBytes + kTileDense + at * 16 + digit] = static_cast<uint8_t>(e < n ? dense_of(u, digit) : 0);
    }
  }
  const uint32_t steps = static_cast<uint32_t>(at / kStep);
  std::memcpy(packet + kLandBytes + 16, &steps, 4);
  if (at > kRun) { std::fprintf(stderr, "lab: a tile packet of %d positions\n", at); std::exit(3); }
}

//: a (column, ring window)'s REGISTER packet: a group's lists cut into classes of four tokens, four classes a step;
//: class `p = 2q + i` of a step is its positions `8q + 2t + i`
void build_register(uint8_t* packet, int column, int ring) {
  int step = 0;
  for (int group = 0; group < 4; ++group) {
    struct Class { int warp, first, count; };
    std::vector<Class> classes;
    for (int warp = 0; warp < 4; ++warp)
      for (int first = 0, n = len_of(column + ring, 4 * group + warp); first < n; first += 4)
        classes.push_back({warp, first, std::min(4, n - first)});
    const int steps = (static_cast<int>(classes.size()) + 3) / 4;
    packet[kLandBytes + group] = static_cast<uint8_t>(steps);
    if ((step + steps) * kStep > kRun || step + steps > kMaxSteps) { std::fprintf(stderr, "lab: a register packet over its room\n"); std::exit(3); }
    for (int c = 0; c < steps * 4; ++c) {
      const int s = step + c / 4, p = c % 4, q = p >> 1, i = p & 1;
      const bool has = c < static_cast<int>(classes.size());
      for (int t = 0; t < 4; ++t) {
        const int at = s * kStep + 8 * q + 2 * t + i;
        const bool there = has && t < classes[c].count;
        const uint32_t u = there ? entry_of(column, ring, 4 * group + classes[c].warp, classes[c].first + t) : 0;
        lay_values(packet, at, there, u);
        //: a dense digit's byte at row `g + 8j`'s place `2g + j`: the digit itself
        for (int digit = 0; digit < 16; ++digit)
          packet[kValueBytes + at * 16 + digit] = static_cast<uint8_t>(there ? dense_of(u, digit) : 0);
        for (int warp = 0; warp < 4; ++warp) {
          uint8_t* const cell = packet + kValueBytes + kDenseBytes + s * 128 + (4 * warp + t) * 8;
          cell[p] = static_cast<uint8_t>(there && classes[c].warp == warp ? sparse_of(u) : 0);
          if (has && classes[c].warp == warp) cell[4] |= static_cast<uint8_t>(1 << p);
        }
      }
    }
    step += steps;
  }
  const uint32_t steps = static_cast<uint32_t>(step);
  std::memcpy(packet + kLandBytes + 4, &steps, 4);
}

//: a (column)'s accumulators after `windows`: list `j`'s dense digit by channel
std::vector<float> expected(int column, int windows) {
  std::vector<float> sum(kLists * 16 * kChannels, 0.0f);
  for (int ring = 0; ring < kRing; ++ring) {
    const int times = windows / kRing + (ring < windows % kRing ? 1 : 0);
    for (int j = 0; j < kLists; ++j)
      for (int n = 0; n < len_of(column + ring, j); ++n) {
        const uint32_t u = entry_of(column, ring, j, n);
        for (int digit = 0; digit < 16; ++digit) {
          const float c = static_cast<float>(dense_of(u, digit) * sparse_of(u) * static_cast<uint32_t>(times));
          for (int ch = 0; ch < kChannels; ++ch) sum[(j * 16 + digit) * kChannels + ch] += c * static_cast<float>(value_of(u, ch));
        }
      }
  }
  return sum;
}

template <int FORM>
void run(lab::Bench& b, const uint8_t* packets, const uint8_t* level, float* state, const std::string& name) {
  constexpr int kMeasured = 64, kChecked = 6;
  const auto kernel = fold<FORM>;
  constexpr int shared = kShared<FORM>;
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
  kernel<<<b.sms, kThreads, shared>>>(b.times, packets, level, state, kChecked);
  LAB_CHECK(cudaDeviceSynchronize());
  std::vector<float> got(static_cast<size_t>(b.sms) * 2 * kGroup * 128);
  LAB_CHECK(cudaMemcpy(got.data(), state, got.size() * 4, cudaMemcpyDeviceToHost));
  long long wrong = 0;
  std::string first;
  for (int column = 0; column < b.sms * 2; ++column) {
    const std::vector<float> want = expected(column % kColumns, kChecked);
    for (int in = 0; in < kGroup; ++in) {
      const int warp = in >> 5, g = (in & 31) >> 2, t = in & 3;
      const float* const mine = got.data() + (static_cast<size_t>(column) * kGroup + in) * 128;
      for (int e = 0; e < 128; ++e) {
        //: THE REGISTER MAP (the lab's wgmma_regs record): element `4q + 2j + i` is row `16 warp + g + 8j`,
        //: column `8q + 2t + i`
        const bool tiled = kTiled<FORM>;
        const int set = tiled ? e / 8 : e / 32, x = tiled ? e % 8 : e % 32;
        const int row = 16 * warp + g + 8 * ((x >> 1) & 1), col = 8 * (x >> 2) + 2 * t + (x & 1);
        const int list = tiled ? set : 4 * set + row / 16;
        const int digit = tiled ? col : 2 * (row % 8) + (row % 16) / 8, ch = tiled ? row : col;
        if (mine[e] != want[(list * 16 + digit) * kChannels + ch] && wrong++ < 6)
          first += " [box " + std::to_string(column) + " list " + std::to_string(list) + " digit " + std::to_string(digit) + " ch "
                   + std::to_string(ch) + ": " + std::to_string(mine[e]) + " for " + std::to_string(want[(list * 16 + digit) * kChannels + ch]) + "]";
      }
    }
  }
  b.check(name, wrong == 0, std::to_string(wrong) + " accumulators differ" + first);
  const lab::Samples cycles = b.variant(
      [&] { kernel<<<b.sms, kThreads, shared>>>(b.times, packets, level, state, kMeasured); },
      [&] { return b.span(kThreads / 32) / kMeasured; });
  b.record(name, "cycles a window a CTA", cycles);
}

}  // namespace

int main(int argc, char** argv) {
  lab::Bench b(argc, argv);
  for (int form : {kTile, kRegister, kRegisterSkip, kTileTable}) {
    const std::string k = lab::kernel("fold_forms::fold", {form});
    const bool tiled = form == kTile || form == kTileTable;
    //: the tile form: a list's steps are one instruction in a rolled loop, and one more is the toolkit's empty group
    //: for the wait (no step). The register form: a group's four held steps and its one more
    b.expect(k, "HGMMA", "==", tiled ? kLists + 1 : 4 * (kHeld + 1));
    //: the warpgroup's first thread alone issues a copy; the skipped form's classes are a branch each
    b.expect(k, "LOOP_CONVERGENCE", "<=", form == kRegisterSkip ? 32 : 8);
    //: the tile form's window is one commit group and its wait follows its steps; the register form waits a group
    //: of four, or a step, for the words it holds: what the forms cost SO is the question
    b.expect(k, "WAIT_AFTER_ISSUE", "<=", tiled ? kLists + 1 : 4 * (kHeld + 1));
    //: the tile forms issue a list's steps in a rolled loop, a fence a step: steps that do not pipeline (H100, 2026-11-20),
    //: a known defect of these rows: their recorded figures are of steps that do not pipeline
    if (tiled) b.expect(k, "STEP_ROLLED", "<=", 16);
    //: the register form's coefficients are converted between its steps, and the skipped form's classes branch there
    if (!tiled) {
      b.expect(k, "STEP_XU", "<=", 64);
      b.expect(k, "BSSY", "<=", 32);
    }
    b.exempt(k, "a comparison of two forms of one step: no budget is claimed for either");
  }
  if (b.expect_only) return 0;
  const size_t columns = kColumns, boxes = static_cast<size_t>(b.sms) * 2;
  std::vector<uint8_t> tile(columns * kRing * kPacketBytes, 0), reg(columns * kRing * kPacketBytes, 0);
  for (size_t column = 0; column < columns; ++column)
    for (int ring = 0; ring < kRing; ++ring) {
      build_tile(tile.data() + (column * kRing + ring) * kPacketBytes, static_cast<int>(column), ring);
      build_register(reg.data() + (column * kRing + ring) * kPacketBytes, static_cast<int>(column), ring);
    }
  uint8_t *dtile = nullptr, *dreg = nullptr, *dlevel = nullptr;
  std::vector<uint8_t> level(static_cast<size_t>(kLevelTokens) * 256);
  for (uint32_t token = 0; token < kLevelTokens; ++token)
    for (int block = 0; block < 16; ++block)
      for (int digit = 0; digit < 16; ++digit) level[level_line(token, block) * 16 + digit] = static_cast<uint8_t>(level_of(token, block, digit));
  LAB_CHECK(cudaMalloc(&dlevel, level.size()));
  LAB_CHECK(cudaMemcpy(dlevel, level.data(), level.size(), cudaMemcpyHostToDevice));
  float* state = nullptr;
  LAB_CHECK(cudaMalloc(&dtile, tile.size()));
  LAB_CHECK(cudaMalloc(&dreg, reg.size()));
  LAB_CHECK(cudaMalloc(&state, boxes * kGroup * 128 * 4));
  LAB_CHECK(cudaMemcpy(dtile, tile.data(), tile.size(), cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMemcpy(dreg, reg.data(), reg.size(), cudaMemcpyHostToDevice));
  run<kTile>(b, dtile, dlevel, state, "the fold's coefficients a tile in shared memory, a step a list");
  run<kRegister>(b, dreg, dlevel, state, "the fold's coefficients in registers, a step four digits, every class computed");
  run<kRegisterSkip>(b, dreg, dlevel, state, "the fold's coefficients in registers, a step four digits, a warp's own classes computed");
  run<kTileTable>(b, dtile, dlevel, state, "the fold's coefficients a tile, its dense lines fetched from the level by token");
  return 0;
}
