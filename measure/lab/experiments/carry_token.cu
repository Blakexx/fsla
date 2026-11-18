// measure/lab/experiments/carry_token.cu -- QUESTION: the carry's token-stream form -- a window's entries laid by the
// fold in one pass from one view in token order, the readout mirrored beside it -- is one design among the 178 kernels
// of `carry_window`. What does a window cost a CTA in that form alone, which role waits on which, and what do its
// own parts cost, in a file that holds that one design and nothing else? METHOD: a CTA an SM of four warpgroups, 512
// threads, registers traded by role: TWO FOLD warpgroups at 168 registers, a 16 x 16 box each, its state 16
// accumulator sets of 8 in registers; ONE READOUT warpgroup at 104; ONE DRAIN warpgroup of four warps at 72. A window
// is a box's 256 entries (16 lists of 16 to 32, a list's entries 8 to 24) and 256 tokens read the pair. The fold's
// window is a TABLE of 256 entries -- an entry's word, place and sparse byte, and the 32 steps' operands -- landed by
// one bulk copy issued a window ahead; each fold thread loads two entries' dense and value lines from global memory
// through its registers, expands the dense line to its two coefficient lines by the sparse byte, and stores the
// value lines to its list's runs across the box's two pool slots; the 32 steps are one block. Each fold warpgroup
// PUBLISHES its state as a bf16 image, 16 tiles, a tile group of 4 behind the readout, a line a channel's 8 fold
// digits. The readout is MIRRORED: a thread expands one (token, box)'s 16 bytes, loaded a pass ahead, to two lines of
// its digit's coefficient tile in shared memory, the tokens on the step's columns; a pass is 8 digits, two tile
// groups, 16 steps of 64 channels by 16 tokens one commit group, and 32 sums a thread per group stored as halves
// of the panel's rows. The four drain warps add each token's row as one f16 line to global memory. The roles' waits
// are suspended (`try_wait`), the registers are traded as stated, and no role reads a clock inside it: the rows are
// the plain kernel, its launch's edges stamped (`TRACE` 1), every warp's phases stamped (2: the coarse row, which is
// also each of the four detail rows -- one barrier's waits and arrivals more, `DETAIL`), and every warp's timeline
// stamped (3). The state and every output row are checked exactly in every row.
// TIMELINE: a fold warpgroup's trip is a window: the image a tile group at a time, four -- the wait on the readout's
// read of that group a window back, its 4 stores a thread of 16 bytes (two lines of two halves), their fence, the
// group's arrival -- then the wait on its table; a thread's two entries in turn, each its dense line and its first
// value line loaded, the dense line expanded to its two coefficient lines, its other 7 value lines loaded two at a
// time and stored to its list's run; a list's last step's pad lines zeroed from the steps' table; a thread of 32 its
// step's operands; the fence, the warpgroup's barrier, the next table's copy; 32 steps as one block, the commit and
// its wait. The readout's trip is a pass: the waits on its two tile groups, a thread's 16 bytes expanded to two lines
// of each group's tile (the bytes loaded a pass before), the fence, the warpgroup's barrier, 16 steps in one commit
// group, the next pass's bytes issued under them, the wait, the groups released; then a group at a time the wait on
// the panel two groups back, its 32 sums rounded to f16 pairs and stored, the arrival. A drain warp's trip is a group
// of 64 tokens, two batches of 8 a warp: the wait on the panel, the chunk's 16 loads, its 16 adds, the arrival. Live:
// a fold thread's 128 accumulators beside an entry's two words, its line and products and two value lines' 8 words;
// the readout's 64 sums (a chunk of 8 digits) and the 8 words of its bytes; a drain's 16.
// see docs/internals/tools/lab.md#carry-token
#include "lab.cuh"

namespace carry_token {

constexpr int kLists = 16, kEntries = 16, kWindow = kLists * kEntries;
constexpr int kChannels = 64, kTableTokens = 65536, kRowWords = 32;
constexpr int kGroup = 128, kDrains = 4, kThreads = 4 * kGroup;
//: the roles' first threads: fold 0, fold 1, the readout, then the four drain warps
constexpr int kReadAt = 2 * kGroup, kDrainAt = 3 * kGroup;
//: THE REGISTERS A THREAD. A raise takes what another warpgroup released from the count the kernel is launched
//: at -- 128 a thread at 512 threads, the SM's whole file: the folds' 40 more each are the 24 fewer of the readout
//: and the 56 fewer of the drains
constexpr int kEntryRegs = 128, kFoldRegs = 168, kReadRegs = 104, kDrainRegs = 72;
static_assert(kThreads * kEntryRegs == 65536 && 2 * (kFoldRegs - kEntryRegs) == 2 * kEntryRegs - kReadRegs - kDrainRegs);

//: A POOL SLOT, in lines: a box's window lays its 16 lists' runs across two of them
constexpr int kBlockSlotLines = 1488;
constexpr int kPoolBytes = kBlockSlotLines * 16;
//: AN IMAGE TILE (the readout's operand a read digit), K-major: a line a channel's 8 fold digits -- one fold
//: thread's own values, a store -- a brick 8 channels' low digits then their high, bricks 17 lines, tiles 137 lines
//: apart; a box's image has room for 16 tiles 161 lines apart
constexpr int kBrickLines = 17, kImageTile = 137 * 16, kImageRoom = 161 * 16, kBoxBytes = kLists * kImageRoom;
//: a panel row: a token's 64 f16 as its line will be added, rows 272 bytes apart
constexpr int kChunkDigits = 4, kTokens = 16 * kChunkDigits, kChunks = kLists / kChunkDigits, kRowBytes = 272,
              kPanelBytes = kTokens * kRowBytes, kBatch = 8;
//: THE SHARED MEMORY: the four pool slots (box 0's, box 1's), the two boxes' images, the two panels, the barriers'
//: 240 bytes, and 3,520 bytes this row does not use -- left, so that every address is the one this row had when it
//: was measured among `carry_window`'s -- then the readout's coefficient tiles, a (digit, box) each: 16 tokens by 16
//: fold digits, K-major, a line a token's 8 digits, a brick 8 tokens' low digits then their high, bricks 17 lines;
//: the box's zeros, two bricks of zeros 16 lines apart (an absent step's B); the steps' operands a box, 16 steps'
//: A and B descriptors, a thread of 16 each step's; 32 bytes unused; and a box's window's TABLE, landed, and its 16
//: lists' second steps' operands (their first are at `kStepsAt`)
constexpr int kImageAt = 4 * kPoolBytes, kPanelAt = kImageAt + 2 * kBoxBytes, kBarsAt = kPanelAt + 2 * kPanelBytes;
constexpr int kCoefTile = 2 * 17 * 16, kCoefAt = kBarsAt + 240 + 3520, kZeroAt = kCoefAt + 2 * 2 * 4 * kCoefTile,
              kStepsAt = kZeroAt + 2 * 2 * 16 * 16;
constexpr int kOneAt = kStepsAt + 2 * 16 * 16 + 32, kOneBytes = 160 * 16, kOneStepsAt = kOneAt + 2 * kOneBytes,
              kSmemBytes = kOneStepsAt + 2 * 16 * 16;
static_assert(kSmemBytes <= 232448);
//: the barriers, 8 bytes each: an image's tile group published and read (a group is the 4 read digits `8 h + c + 2 j`
//: of a chunk `2 h + c`: every fold thread holds a quarter of it); a panel filled and drained; a box's table landed
//: (box, 1 -- the copy's barrier); and the five this row initialises and never waits on or arrives at -- `kPoolReady`,
//: `kPoolFreed`, `kDone`, `kMeet` and `kCoefLanded` -- as its binary has them
constexpr int kPoolReady = 0, kPoolFreed = 32, kTilesReady = 64, kTilesRead = 96, kPanelReady = 128, kPanelFreed = 144,
              kDone = 160, kPoolLanded = 176, kMeet = 208, kCoefLanded = 232;
//: the named barriers: a box's fold warpgroup together behind its stores, and the readout's
constexpr int kPoolId = 1, kReadId = 11;
constexpr int kOutWindows = 4;
//: a slot's steps are this many rounds of its lists; a list's run of lines padded to its steps
constexpr int kBlockRounds = 2;
constexpr int block_spread(int n) { return (n + 15) & ~15; }
//: A WINDOW IN GLOBAL MEMORY, in lines, entry `t` the window's `t`th in token order: its TABLE -- an entry's word,
//: 256; the entries' sparse bytes; an entry's place, 256; the 32 steps' four halves, a list's first step then its
//: second -- which lands; then the entries' dense lines and their value lines, a brick a run of 256, which threads load
constexpr int kOneSparse = 64, kOnePlaces = 80, kOneSteps = 144, kOneDense = 160, kOneValues = kOneDense + 2 * kGroup,
              kOneLines = kOneValues + 16 * kGroup;
static_assert(kOneDense * 16 == kOneBytes);

//: the token of a (CTA, box, entry of the call), its value at a channel, its byte on its list's digit and on a
//: dense leaf; a read coefficient at (read digit, fold digit, token): small integers
constexpr uint32_t token_of(uint32_t source, uint32_t entry) {
  return ((source * 4099u + entry) * 2654435761u >> 12) & (kTableTokens - 1);
}
//: a window's entries are its sparse column's lists' (16 columns, a source its own): the window's first token and
//: the place the list holds; a source's 64 windows of 256 entries
constexpr int kColumns = 16, kCallWindows = 64, kWindowTokens = 1024;
constexpr int kListEntries = kCallWindows * kWindow;
constexpr uint32_t listed_place(uint32_t source, uint32_t entry) {
  return ((source * 4099u + entry % kListEntries) * 2654435761u >> 12) & (kWindowTokens - 1);
}
constexpr uint32_t listed_token(uint32_t source, uint32_t entry) {
  return (entry / kWindow % kCallWindows) * kWindowTokens + listed_place(source, entry);
}
//: a list's entries in a window, 8 to 24 and a window's 256: lists `2 p` and `2 p + 1` are 16 more and less the same
constexpr int packed_count(uint32_t column, uint32_t window, int list) {
  const int by = static_cast<int>(((column * 131u + window * 31u + static_cast<uint32_t>(list >> 1)) * 2654435761u >> 20) % 9u);
  return list & 1 ? kEntries - by : kEntries + by;
}
constexpr int value_of(int t, int c) { return (t + c) % 2 + 1; }
constexpr int sparse_of(int t) { return t % 2 + 1; }
constexpr int dense_of(int t, int b) { return (t + b) % 2 + 1; }
//: a reader: the token of a (CTA, entry of the call) -- read digit `b`'s token `n` of window `w` is entry
//: `256 w + 16 b + n` -- whose bytes on box `box`'s 16 leaves are the dense row of the token `box` on from it
constexpr uint32_t reader_of(uint32_t block, uint32_t entry) { return token_of(0x10000u + block, entry); }
constexpr uint32_t reader_row(uint32_t reader, int box) { return (reader + box) & (kTableTokens - 1); }

//: A DETAIL GROUP: one barrier's waits and arrivals, stamped beside a coarse row's phases in a row of its own -- a
//: stamp budget spent on one question a run
enum Detail : int {
  //: the folds' waits on a tile group read, and the readout's arrivals at it
  kWaitTilesRead = 1,
  //: the readout's waits on a tile group published, and the folds' arrivals at it
  kWaitTilesReady = 2,
  //: the drain warps' waits on a panel filled, and the readout's arrivals at it
  kWaitPanelReady = 4,
  //: the readout's waits on a panel drained, and the drain warps' arrivals at it
  kWaitPanelFreed = 8
};
//: A ROLE'S TIMELINE (the traced rows'): `TRACE` 0 stamps nothing; 1 the launch's two edges alone; 2 a phase's start
//: in every role, and the edges; 3 every stamp: each wait's start and end, each arrival and each copy's issue too. A
//: coarse row keeps, beside its phases' starts, the stamps of the groups in `DETAIL`
template <int TRACE, int DETAIL>
struct Tracer : lab::Trace<TRACE != 0> {
  using Row = lab::Trace<TRACE != 0>;
  static_assert(DETAIL == 0 || TRACE == 2);
  //: the launch's edges are a coarse row's and an edges row's: a fine row's kernels are at their registers' limit
  //: and hold no entry beside their timeline (their windows' marks time them)
  static constexpr bool kEdged = TRACE == 1 || TRACE == 2;
  static constexpr uint32_t kEntered = kEdged ? lab::kMarkEntries : 0u;
  //: a stamp's kind and group are literals where it is placed, so the others compile to nothing. A phase's start is
  //: an ANCHOR, on both clocks (a mark); any other stamp a row keeps is the cycle counter's
  __device__ __forceinline__ void stamp(uint32_t kind, uint32_t site, int group = 0) {
    if constexpr ((TRACE & 2) != 0) {
      if (kind == warpline::kPhaseBegin) Row::mark(kind, site);
      else if ((TRACE & 1) != 0 || (DETAIL & group) != 0) Row::stamp(kind, site);
    }
  }
  //: THE LAUNCH'S EDGE, a coarse row's and an edges row's: a mark
  __device__ __forceinline__ void edge(uint32_t site) {
    if constexpr (kEdged) Row::mark(warpline::kPhaseBegin, site);
  }
  __device__ __forceinline__ void at(uint32_t kind, uint32_t bar, int group = 0) { stamp(kind, this->site(bar), group); }
};
//: a phase's or an edge's site, at the ids `carry_window`'s records name them by (a trace of this file reads beside one
//: of its)
enum Site : uint32_t { kMarkWindow = 0xc000u, kMarkSteps = 0xc002u, kMarkPanel = 0xc003u, kMarkAdds = 0xc004u,
                       kStepsRead = 0xc007u, kMarkEntry = 0xc008u, kMarkExit = 0xc009u };

//: A WAIT at a barrier, suspended (`try_wait`), stamped
template <int TRACE, int DETAIL>
__device__ __forceinline__ void await(uint32_t bar, uint32_t parity, Tracer<TRACE, DETAIL>& tr, int group = 0) {
  tr.at(warpline::kWaitBegin, bar, group);
  lab::mbar_wait_suspended(bar, parity);
  tr.at(warpline::kWaitEnd, bar, group);
}
//: an arrival at `bar`, a warp's (its lane 0's), stamped where the row is traced
template <int TRACE, int DETAIL>
__device__ __forceinline__ void arrive(uint32_t bar, Tracer<TRACE, DETAIL>& tr, int group = 0) {
  tr.at(warpline::kArrive, bar, group);
  if ((threadIdx.x & 31) == 0) lab::mbar_arrive(bar);
}
//: the warp's steps waited out, stamped at `site` where the row is traced
template <int TRACE, int DETAIL>
__device__ __forceinline__ void steps_done(Tracer<TRACE, DETAIL>& tr, uint32_t site) {
  tr.stamp(warpline::kWaitBegin, site);
  lab::wg_wait<0>();
  tr.stamp(warpline::kWaitEnd, site);
}
//: a warp's eleven count words at `kCounts (block, warp)`, written zero: no role reads a clock inside it, so no wait
//: or phase is counted (the stores remain the kernel's)
constexpr int kCounts = 11;
__device__ __forceinline__ void clear_counts(uint32_t* waits) {
  if ((threadIdx.x & 31) == 0) {
    uint32_t* const to = waits + kCounts * (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5));
    to[0] = 0u, to[1] = 0u, to[2] = 0u, to[3] = 0u, to[4] = 0u, to[5] = 0u, to[6] = 0u, to[7] = 0u, to[8] = 0u, to[9] = 0u, to[10] = 0u;
  }
}

//: A STEP'S TWO OPERANDS from its four halves -- A's line and pitch, B's (no B: the tile of zeros) -- into `to`
__device__ __forceinline__ void make_step(const uint16_t* step, uint32_t pool_at, uint32_t zeros_at, uint64_t* to) {
  to[0] = lab::wg_desc(pool_at + step[0] * 16u, 8, step[1]);
  to[1] = step[2] == 0xffffu ? lab::wg_desc(zeros_at, 8, 16) : lab::wg_desc(pool_at + step[2] * 16u, 8, step[3]);
}

//: A FOLD WARPGROUP: its box's state; a window is the image, then the table's wait, its entries laid, and 32 steps
template <int TRACE, int DETAIL>
__device__ __forceinline__ void fold_role(uint8_t* smem, uint32_t base, long long* times, uint32_t* waits,
                                          float* state, const uint32_t* values, unsigned long long* trace, int windows) {
  const int tid = threadIdx.x, box = tid >> 7, warp = (tid >> 5) & 3, q = (tid & 31) >> 2, t = (tid & 31) & 3;
  const uint32_t bars = base + kBarsAt;
  //: a list's first step writes its accumulator whole: nothing is cleared (a cleared one in a role's branch
  //: serializes the steps, ptxas C7520)
  float d[kLists][8];
  //: a thread's base into its box's image: every store is this plus a constant of (tile, row, half)
  uint8_t* const mine = smem + lab::held(static_cast<uint32_t>(
      kImageAt + box * kBoxBytes + 2 * t * kImageTile + 16 * (2 * warp * kBrickLines + q)));
  //: window `w`'s first line in the views' table, and its table's one copy by the warpgroup's first thread
  const auto one_line = [&](int w) {
    return ((blockIdx.x * 2 + (threadIdx.x >> 7)) % kColumns * kCallWindows + static_cast<uint32_t>(w) % kCallWindows) * static_cast<uint32_t>(kOneLines);
  };
  const auto issue_one = [&](int w) {
    if ((threadIdx.x & (kGroup - 1)) == 0) {
      const int box_of = threadIdx.x >> 7;
      const uint32_t landed = base + kBarsAt + kPoolLanded + 8 * (2 * box_of + 1);
      lab::mbar_expect(landed, kOneBytes);
      lab::bulk_load(base + kOneAt + box_of * kOneBytes, reinterpret_cast<const uint4*>(values) + one_line(w), kOneBytes, landed);
    }
  };
  Tracer<TRACE, DETAIL> tr;
  tr.bind(trace, Tracer<TRACE, DETAIL>::kEntered);
  //: a fold marks its clock in global memory: there its held start spills
  lab::WarpMark::start(times);
  //: the box's zeros, and the warpgroup behind them; the first window's table
  if ((tid & (kGroup - 1)) < 32) *reinterpret_cast<uint4*>(smem + kZeroAt + box * 512 + 16 * (tid & 31)) = make_uint4(0, 0, 0, 0);
  lab::bulk_fence();
  lab::bar_sync(kPoolId + box, kGroup);
  issue_one(0);
#pragma unroll 1
  for (int w = 0; w < windows; ++w) {
    tr.stamp(warpline::kPhaseBegin, kMarkWindow);
    //: THE IMAGE, a tile group at a time behind the readout: channel m's fold digits `8 h` to `8 h + 7` of read
    //: digit b truncated to one line of tile b, `(m / 8) 17 + m % 8 + 8 h`, once the readout has read that group
    //: of the window before: 4 stores a thread a group.
    //: The first window's image is the cleared shared memory: the state is zero
#pragma unroll
    for (int k = 0; k < kChunks; ++k) {
      if (w > 0) await(bars + kTilesRead + 8 * k, (w - 1) & 1, tr, kWaitTilesRead);
      if (w > 0) {
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int i = (k & 1) | r << 1 | (k >> 1) << 2;
            uint32_t pair[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) pair[j] = lab::bf16x2_trunc(d[8 * h + 2 * j][i], d[8 * h + 2 * j + 1][i]);
            *reinterpret_cast<uint4*>(mine + (8 * (k >> 1) + (k & 1)) * kImageTile + 16 * (r * kBrickLines + 8 * h)) =
                make_uint4(pair[0], pair[1], pair[2], pair[3]);
          }
      }
      lab::bulk_fence();
      __syncwarp();
      arrive(bars + kTilesReady + 8 * k, tr, kWaitTilesReady);
    }
    //: THE WINDOW: its table landed; a thread's two entries, each loaded and laid; the pads; the steps' operands;
    //: then the window's 32 steps
    lab::mbar_wait_suspended(bars + kPoolLanded + 8 * (2 * box + 1), w & 1);
    const uint32_t pool = 2 * box * kPoolBytes;
    const int g = tid & (kGroup - 1);
    uint8_t* const image = smem + pool;
    const uint8_t* const table = smem + kOneAt + box * kOneBytes;
    const uint16_t* const steps_of = reinterpret_cast<const uint16_t*>(table + kOneSteps * 16);
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const int t = g + q * kGroup;
      const uint32_t word = reinterpret_cast<const uint32_t*>(table)[t];
      const uint32_t place = reinterpret_cast<const uint32_t*>(table + kOnePlaces * 16)[t];
      const float weight = static_cast<float>(table[kOneSparse * 16 + t]);
      //: (the window's line in the views' table is held as 32 bits: no address of it outlives the entry)
      const uint4* const from = reinterpret_cast<const uint4*>(values) + lab::held(one_line(w)) + t;
      uint4* const to = reinterpret_cast<uint4*>(image) + (place & 0xffffu);
      const uint32_t pitch = place >> 16;
      const uint4 landed = from[kOneDense];
      const uint4 first = from[kOneValues];
      const uint32_t line[4] = {landed.x, landed.y, landed.z, landed.w};
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint32_t pair[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const uint32_t two = line[2 * h + c / 2] >> (16 * (c & 1));
          pair[c] = lab::bf16x2_round(weight * static_cast<float>(two & 255u), weight * static_cast<float>(two >> 8 & 255u));
        }
        *reinterpret_cast<uint4*>(image + (h == 0 ? word & 0xffffu : word >> 16) * 16) = make_uint4(pair[0], pair[1], pair[2], pair[3]);
      }
      to[0] = first;
#pragma unroll
      for (int part = 0; part < 3; ++part) {
        const uint4 a = from[kOneValues + (1 + 2 * part) * 2 * kGroup], b = from[kOneValues + (2 + 2 * part) * 2 * kGroup];
        to[(1 + 2 * part) * pitch] = a, to[(2 + 2 * part) * pitch] = b;
      }
      const uint4 last = from[kOneValues + 7 * 2 * kGroup];
      to[7 * pitch] = last;
    }
    //: THE PADS, a thread a place of two lists' last steps: where a list's entries end inside a step, the places
    //: behind them in its two coefficient runs
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const uint16_t* const list = steps_of + 4 * (8 * q + (g >> 4));
      const uint32_t n = list[1], tail = n & 15u, k = static_cast<uint32_t>(g) & 15u;
      if (tail != 0 && k >= tail) {
        const uint32_t at = list[2] + (n & ~15u) + k;
        *reinterpret_cast<uint4*>(image + at * 16) = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(image + (at + list[3]) * 16) = make_uint4(0, 0, 0, 0);
      }
    }
    //: and, a thread of the first 32, ITS STEP's operands: step `16 r + j` is list `j`'s `r`th
    if (g < 32) {
      make_step(steps_of + 4 * g, base + pool, base + kZeroAt + box * 512,
                reinterpret_cast<uint64_t*>(smem + (g < 16 ? kStepsAt : kOneStepsAt) + box * 256 + 16 * (g & 15)));
    }
    lab::bulk_fence();
    lab::bar_sync(kPoolId + box, kGroup);
    //: THE NEXT WINDOW'S TABLE, behind every thread's reads of this one and under this window's steps
    if (w + 1 < windows) issue_one(w + 1);
    lab::wg_fence();
#pragma unroll
    for (int r = 0; r < kBlockRounds; ++r) {
      const uint64_t* const steps = reinterpret_cast<const uint64_t*>(smem + (r == 0 ? kStepsAt : kOneStepsAt) + box * 256);
#pragma unroll
      for (int j = 0; j < kLists; ++j) {
        //: a step's operands are loaded at its own issue: the edge keeps the window's loads from one register set
        asm volatile("" ::: "memory");
        lab::wg_bf16_n16_ss_mn(d[j], steps[2 * j], steps[2 * j + 1], w != 0 || r != 0);
      }
    }
    lab::wg_commit();
    lab::wg_wait<0>();
  }
#pragma unroll
  for (int a = 0; a < kLists; ++a)
#pragma unroll
    for (int i = 0; i < 8; ++i) lab::wg_hold(d[a][i]);
  lab::WarpMark::stop(times);
  tr.edge(kMarkExit);
  tr.close(base);
  clear_counts(waits);
  //: THE STATE OUT, for the checked call alone: 128 stores a thread, each instruction's lanes on 32 lines, issued
  //: while the readout and the drains are still in their last windows -- in a timed call they would be clocked as
  //: the read side's
  if (windows <= kOutWindows) {
    float* const to = state + (static_cast<long long>(blockIdx.x) * 2 * kGroup + tid) * (kLists * 8);
#pragma unroll
    for (int a = 0; a < kLists; ++a)
#pragma unroll
      for (int i = 0; i < 8; ++i) to[8 * a + i] = d[a][i];
  }
}

//: THE MIRRORED READOUT's warpgroup: a pass is two tile groups -- both awaited, a (token, box) a thread a group
//: expanded to its tile, the warpgroup's barrier, 8 steps a group in one commit group, then a thread's 32 sums a
//: group rounded and stored a half each to that group's panel
template <int TRACE, int DETAIL>
__device__ __forceinline__ void read_role(uint8_t* smem, uint32_t base, long long* times, uint32_t* waits,
                                          const uint32_t* dense, unsigned long long* trace, int windows) {
  Tracer<TRACE, DETAIL> tr;
  tr.bind(trace, Tracer<TRACE, DETAIL>::kEntered);
  constexpr int kG = 2;
  const int tid = threadIdx.x - kReadAt, warp = tid >> 5, lane = tid & 31, q = lane >> 2, t = lane & 3;
  const uint32_t bars = base + kBarsAt;
  //: the lane's (token, box) of a group: token `lane % 16` of the warp's digit on box `lane / 16`
  const int n = lane & 15, box = lane >> 4;
  uint4 row[kG];
  uint32_t reader_at[kG];
  const auto load = [&](int c) {
#pragma unroll
    for (int g = 0; g < kG; ++g) {
      const int k = (c + g) % kChunks;
      reader_at[g] = reader_of(blockIdx.x, ((c + g) / kChunks) * kWindow + 16 * (8 * (k >> 1) + (k & 1) + 2 * warp) + n);
      row[g] = reinterpret_cast<const uint4*>(dense)[reader_row(reader_at[g], box)];
    }
  };
  load(0);
  float d[kG * kChunkDigits][8];
  //: the lane's two lines of its digit's tile on its box (a group's tiles 8 on), and its sums' rows of a panel
  uint8_t* const lines = smem + kCoefAt + (2 * warp + box) * kCoefTile + ((n >> 3) * 17 + (n & 7)) * 16;
  const uint32_t mine = lab::held(static_cast<uint32_t>(kPanelAt + 2 * t * kRowBytes + 2 * (16 * warp + q)));
  lab::WarpMark::start(times);
#pragma unroll 1
  for (int w = 0; w < windows; ++w) {
#pragma unroll 1
    for (int c4 = 0; c4 < kChunks; c4 += kG) {
      const int c = kChunks * w + c4;
#pragma unroll
      for (int g = 0; g < kG; ++g) await(bars + kTilesReady + 8 * (c4 + g), w & 1, tr, kWaitTilesReady);
      //: THE COEFFICIENTS, to shared memory: the reader's 16 bytes on the box by its byte on its digit
#pragma unroll
      for (int g = 0; g < kG; ++g) {
        const float weight = static_cast<float>(sparse_of(reader_at[g]));
        const uint32_t word[4] = {row[g].x, row[g].y, row[g].z, row[g].w};
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          uint32_t pair[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const uint32_t two = word[2 * h + i / 2] >> (16 * (i & 1));
            pair[i] = lab::bf16x2_round(weight * static_cast<float>(two & 255u), weight * static_cast<float>(two >> 8 & 255u));
          }
          *reinterpret_cast<uint4*>(lines + g * 8 * kCoefTile + 8 * 16 * h) = make_uint4(pair[0], pair[1], pair[2], pair[3]);
        }
      }
      lab::bulk_fence();
      lab::bar_sync(kReadId, kGroup);
      const uint32_t tiles = lab::held(base + kCoefAt);
      tr.stamp(warpline::kPhaseBegin, kMarkSteps);
      lab::wg_fence();
#pragma unroll
      for (int g = 0; g < kG; ++g) {
        const uint32_t first = static_cast<uint32_t>(8 * ((c4 + g) >> 1) + ((c4 + g) & 1));
        const uint32_t images = base + kImageAt + first * kImageTile;
#pragma unroll
        for (int j = 0; j < kChunkDigits; ++j)
#pragma unroll
          for (int bx = 0; bx < 2; ++bx)
            lab::wg_bf16_n16_ss(d[g * kChunkDigits + j],
                                lab::wg_desc(images + bx * kBoxBytes + 2 * j * kImageTile, 8, kBrickLines),
                                lab::wg_desc(tiles + (g * 8 + 2 * j + bx) * kCoefTile, 8, 17), bx != 0);
      }
      lab::wg_commit();
      //: the next pass's bytes, issued under the steps
      load(c + kG);
      steps_done(tr, kStepsRead);
#pragma unroll
      for (int g = 0; g < kG; ++g) arrive(bars + kTilesRead + 8 * (c4 + g), tr, kWaitTilesRead);
      //: THE PANEL a group: digit `j`'s token `8 i + 2 t + p` on channel `16 warp + q + 8 r`, a half of that
      //: token's line
#pragma unroll
      for (int g = 0; g < kG; ++g) {
        const int cg = c + g;
        if (cg >= 2) await(bars + kPanelFreed + 8 * (cg & 1), ((cg >> 1) - 1) & 1, tr, kWaitPanelFreed);
        tr.stamp(warpline::kPhaseBegin, kMarkPanel);
        uint8_t* const panel = smem + mine + (cg & 1) * kPanelBytes;
#pragma unroll
        for (int j = 0; j < kChunkDigits; ++j)
#pragma unroll
          for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int r = 0; r < 2; ++r) {
              const uint32_t halves = lab::pack_half2(d[g * kChunkDigits + j][4 * i + 2 * r], d[g * kChunkDigits + j][4 * i + 2 * r + 1]);
              uint8_t* const at = panel + (16 * j + 8 * i) * kRowBytes + 16 * r;
              *reinterpret_cast<uint16_t*>(at) = static_cast<uint16_t>(halves);
              *reinterpret_cast<uint16_t*>(at + kRowBytes) = static_cast<uint16_t>(halves >> 16);
            }
        __syncwarp();
        arrive(bars + kPanelReady + 8 * (cg & 1), tr, kWaitPanelReady);
      }
    }
  }
  lab::WarpMark::stop(times);
  tr.edge(kMarkExit);
  tr.close(base);
  clear_counts(waits);
}

//: A DRAIN WARP: a chunk's 16 tokens of its residue in two batches of 8; a lane holds a token's two channels and
//: adds one f16 pair
template <int TRACE, int DETAIL>
__device__ __forceinline__ void drain_role(uint8_t* smem, uint32_t base, long long* times, uint32_t* waits,
                                           uint32_t* out, unsigned long long* trace, int windows) {
  Tracer<TRACE, DETAIL> tr;
  tr.bind(trace, Tracer<TRACE, DETAIL>::kEntered);
  const int lane = threadIdx.x & 31, mine = (threadIdx.x - kDrainAt) >> 5;
  const uint32_t bars = base + kBarsAt;
  constexpr int kBatches = kTokens / kDrains / kBatch;
  uint32_t held[2][kBatch];
  //: batch `n` of chunk `c` into register half `h`: a lane's word of a token's row is its f16 pair
  const auto load = [&](int c, int n, int h) {
#pragma unroll
    for (int i = 0; i < kBatch; ++i) {
      const uint8_t* const row = smem + kPanelAt + (c & 1) * kPanelBytes + (mine + kDrains * (n * kBatch + i)) * kRowBytes;
      held[h][i] = reinterpret_cast<const uint32_t*>(row)[lane];
    }
  };
  lab::WarpMark::start(times);
#pragma unroll 1
  for (int c = 0; c < kChunks * windows; ++c) {
    await(bars + kPanelReady + 8 * (c & 1), (c >> 1) & 1, tr, kWaitPanelReady);
    tr.stamp(warpline::kPhaseBegin, kMarkAdds);
    load(c, 0, 0);
    load(c, 1, 1);
    //: the chunk's digit `j` is read digit `8 h + c + 2 j`: its 16 tokens' rows start at 16 of that
    const long long first = static_cast<long long>(blockIdx.x) * (kOutWindows * kWindow)
                            + ((c / kChunks) % kOutWindows) * kWindow
                            + 16 * (8 * ((c % kChunks) >> 1) + ((c % kChunks) & 1));
#pragma unroll
    for (int n = 0; n < kBatches; ++n) {
#pragma unroll
      for (int i = 0; i < kBatch; ++i) {
        const int token = mine + kDrains * (n * kBatch + i);
        lab::red_add_f16x2(out + (first + 32 * (token / 16) + token % 16) * (kChannels / 2) + lane, held[n & 1][i]);
      }
    }
    __syncwarp();
    arrive(bars + kPanelFreed + 8 * (c & 1), tr, kWaitPanelFreed);
  }
  tr.edge(kMarkExit);
  tr.close(base);
  lab::WarpMark::stop(times);
  clear_counts(waits);
}

//: THE KERNEL of a row: `TRACE` and `DETAIL` select its stamps and nothing else -- a detail row is its coarse row but
//: for those stamps. `sinks` is not read: the parameter list is `carry_window`'s, whose binary this one is held to
template <int TRACE, int DETAIL>
__global__ void __launch_bounds__(kThreads, 1)
    window(long long* times, uint32_t* waits, int* sinks, const uint32_t* values, const uint32_t* dense, float* state,
           uint32_t* out, unsigned long long* trace, int windows) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x;
  const uint32_t base = lab::smem(smem);
  //: THE LAUNCH'S TWO EDGES, on both clocks: a warp's entry here, before the kernel's setup, and its exit at its
  //: role's end; the role's tracer writes after this one's mark
  {
    Tracer<TRACE, DETAIL> in;
    in.bind(trace);
    in.edge(kMarkEntry);
  }
  //: the images cleared: the state is zero at the call's start
#pragma unroll 1
  for (int i = tid; i < 2 * kBoxBytes / 4; i += kThreads) *reinterpret_cast<uint32_t*>(smem + kImageAt + 4 * i) = 0u;
  if (tid == 0) {
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      lab::mbar_init(base + kBarsAt + kPoolReady + 8 * p, 1);
      lab::mbar_init(base + kBarsAt + kPoolFreed + 8 * p, 1);
      lab::mbar_init(base + kBarsAt + kPoolLanded + 8 * p, 1);
    }
#pragma unroll
    for (int k = 0; k < kChunks; ++k) {
      lab::mbar_init(base + kBarsAt + kTilesReady + 8 * k, 2 * kGroup / 32);
      lab::mbar_init(base + kBarsAt + kTilesRead + 8 * k, kGroup / 32);
    }
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      lab::mbar_init(base + kBarsAt + kPanelReady + 8 * p, kGroup / 32);
      lab::mbar_init(base + kBarsAt + kPanelFreed + 8 * p, kDrains);
    }
    lab::mbar_init(base + kBarsAt + kDone, kGroup / 32);
#pragma unroll
    for (int i = 0; i < 3; ++i) lab::mbar_init(base + kBarsAt + kMeet + 8 * i, kGroup / 32);
    lab::mbar_init(base + kBarsAt + kCoefLanded, 1);
  }
  lab::bulk_fence();
  __syncthreads();

  //: EACH ROLE'S WHOLE BODY IN ITS OWN BRANCH, a warpgroup's registers set at the branch's head: the folds' raise is
  //: the release of the readout and of the drains' warpgroup
  if (tid < kReadAt) {
    lab::regs_up<kFoldRegs>();
    fold_role<TRACE, DETAIL>(smem, base, times, waits, state, values, trace, windows);
  } else if (tid < kDrainAt) {
    lab::regs_down<kReadRegs>();
    read_role<TRACE, DETAIL>(smem, base, times, waits, dense, trace, windows);
  } else {
    lab::regs_down<kDrainRegs>();
    drain_role<TRACE, DETAIL>(smem, base, times, waits, out, trace, windows);
  }
}

}  // namespace carry_token

inline uint32_t bf16_host(int v) {
  const float f = static_cast<float>(v);
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u >> 16;
}

//: an integer under 2^24 as the image holds it (truncated to bf16)
inline long long image_host(long long v) {
  const float f = static_cast<float>(v);
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u &= 0xffff0000u;
  float g;
  std::memcpy(&g, &u, 4);
  return static_cast<long long>(g);
}

//: A CALL'S BUFFERS: the window views, the readers' dense table, the state, the output and the waits' counts, each
//: sized to the grid
struct Buffers {
  uint32_t *values, *dense, *out, *waits;
  float* state;
};
inline Buffers buffers(const lab::Bench& b) {
  using namespace carry_token;
  //: the 65,536 tokens' rows: a token's 32 words are its 64 bf16 values, its dense row 16 bytes
  std::vector<uint32_t> host_values(static_cast<size_t>(kTableTokens) * kRowWords),
      host_dense(static_cast<size_t>(kTableTokens) * 4);
  for (size_t w = 0; w < host_values.size(); ++w) {
    const int t = static_cast<int>(w / kRowWords), c = 2 * static_cast<int>(w % kRowWords);
    host_values[w] = bf16_host(value_of(t, c)) | bf16_host(value_of(t, c + 1)) << 16;
  }
  for (size_t w = 0; w < host_dense.size(); ++w) {
    const int t = static_cast<int>(w / 4), b0 = 4 * static_cast<int>(w % 4);
    host_dense[w] = dense_of(t, b0) | dense_of(t, b0 + 1) << 8 | dense_of(t, b0 + 2) << 16 | dense_of(t, b0 + 3) << 24;
  }
  Buffers held{};
  //: THE VIEWS: each (column, window)'s view -- the window's 256 entries in token order, their places across the
  //: box's two pool slots: every list's value runs, then its two coefficient runs
  const size_t room = static_cast<size_t>(kOneLines) * 16;
  std::vector<uint8_t> views(static_cast<size_t>(kColumns) * kCallWindows * room, 0);
  struct Entry { uint32_t t; int j, k; };
  for (int c = 0; c < kColumns; ++c)
    for (int w = 0; w < kCallWindows; ++w) {
      uint8_t* const view = views.data() + (static_cast<size_t>(c) * kCallWindows + w) * room;
      int n[kLists], values_at[kLists], dense_at[kLists], second_at[kLists], run = 0, e = 0;
      for (int j = 0; j < kLists; ++j) n[j] = packed_count(c, w, j), values_at[j] = run, run += 8 * n[j];
      for (int j = 0; j < kLists; ++j) dense_at[j] = run, run += block_spread(n[j]);
      for (int j = 0; j < kLists; ++j) second_at[j] = run, run += block_spread(n[j]);
      std::vector<Entry> entries;
      for (int j = 0; j < kLists; ++j) {
        for (int k = 0; k < n[j]; ++k)
          entries.push_back({listed_token(static_cast<uint32_t>(c), static_cast<uint32_t>(w * kWindow + e + k)), j, k});
        e += n[j];
      }
      std::stable_sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.t < b.t; });
      if (entries.size() != static_cast<size_t>(2 * kGroup) || run > 2 * kBlockSlotLines) {
        std::fprintf(stderr, "carry_token: a window of %zu entries and %d lines is not this lab's\n", entries.size(), run);
        std::exit(2);
      }
      uint16_t* const steps = reinterpret_cast<uint16_t*>(view + kOneSteps * 16);
      for (int j = 0; j < kLists; ++j)
        for (int r = 0; r < kBlockRounds; ++r) {
          const bool live = 16 * r < n[j];
          uint16_t* const step = steps + 4 * (kLists * r + j);
          step[0] = static_cast<uint16_t>(values_at[j] + (live ? 16 * r : 0)), step[1] = static_cast<uint16_t>(n[j]);
          step[2] = live ? static_cast<uint16_t>(dense_at[j] + 16 * r) : 0xffff;
          step[3] = static_cast<uint16_t>(second_at[j] - dense_at[j]);
        }
      for (int r = 0; r < 2 * kGroup; ++r) {
        const Entry& at = entries[static_cast<size_t>(r)];
        reinterpret_cast<uint32_t*>(view)[r] = static_cast<uint32_t>(dense_at[at.j] + at.k) | static_cast<uint32_t>(second_at[at.j] + at.k) << 16;
        reinterpret_cast<uint32_t*>(view + kOnePlaces * 16)[r] = static_cast<uint32_t>(values_at[at.j] + at.k) | static_cast<uint32_t>(n[at.j]) << 16;
        view[kOneSparse * 16 + r] = static_cast<uint8_t>(sparse_of(static_cast<int>(at.t)));
        for (int dd = 0; dd < 16; ++dd) view[(kOneDense + r) * 16 + dd] = static_cast<uint8_t>(dense_of(static_cast<int>(at.t), dd));
        for (int br = 0; br < 8; ++br)
          std::memcpy(view + (kOneValues + br * 2 * kGroup + r) * 16, &host_values[static_cast<size_t>(at.t) * kRowWords + 4 * br], 16);
      }
    }
  uint8_t* vdev = nullptr;
  LAB_CHECK(cudaMalloc(&vdev, views.size()));
  LAB_CHECK(cudaMemcpy(vdev, views.data(), views.size(), cudaMemcpyHostToDevice));
  held.values = reinterpret_cast<uint32_t*>(vdev);
  LAB_CHECK(cudaMalloc(&held.dense, host_dense.size() * 4));
  LAB_CHECK(cudaMemcpy(held.dense, host_dense.data(), host_dense.size() * 4, cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMalloc(&held.state, sizeof(float) * b.sms * 2 * kGroup * (kLists * 8)));
  LAB_CHECK(cudaMalloc(&held.out, static_cast<size_t>(b.sms) * kOutWindows * kWindow * (kChannels / 2) * 4));
  LAB_CHECK(cudaMalloc(&held.waits, static_cast<size_t>(b.sms) * (kThreads >> 5) * kCounts * 4));
  return held;
}

//: A ROW: the kernel `window<TRACE, DETAIL>`, its checked call and its timed launches
template <int TRACE, int DETAIL = 0>
void run(lab::Bench& b, const Buffers& held) {
  using namespace carry_token;
  constexpr int kTimed = 64;
  const std::string name = std::string("the token-stream carry: both sides")
                           + (TRACE & 2 ? (TRACE & 1 ? ", every warp's timeline stamped" : ", every warp's phases stamped")
                                        : TRACE & 1 ? ", the launch's two edges stamped" : "")
                           + (DETAIL & kWaitTilesRead ? ", the waits on the tile groups read stamped" : "")
                           + (DETAIL & kWaitTilesReady ? ", the waits on the tile groups published stamped" : "")
                           + (DETAIL & kWaitPanelReady ? ", the waits on the panels filled stamped" : "")
                           + (DETAIL & kWaitPanelFreed ? ", the waits on the panels drained stamped" : "");
  constexpr int kWarps = kThreads >> 5;
  auto kernel = window<TRACE, DETAIL>;
  uint32_t* const values = held.values;
  uint32_t* const dense = held.dense;
  uint32_t* const out = held.out;
  uint32_t* const waits = held.waits;
  float* const state = held.state;
  //: THE TRACED rows' timelines: a warp's row of stamps (lab::Trace), the timed launches' last left in them; cleared,
  //: so a warp that stamps nothing reads as no stamps, never as the memory's last contents
  unsigned long long* rows = nullptr;
  if constexpr (TRACE != 0) {
    LAB_CHECK(cudaMalloc(&rows, static_cast<size_t>(b.sms) * kWarps * lab::Trace<true>::kRow * 8));
    LAB_CHECK(cudaMemset(rows, 0, static_cast<size_t>(b.sms) * kWarps * lab::Trace<true>::kRow * 8));
  }
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  const size_t words = static_cast<size_t>(b.sms) * kOutWindows * kWindow * (kChannels / 2);
  LAB_CHECK(cudaMemset(out, 0, words * 4));
  kernel<<<b.sms, kThreads, kSmemBytes>>>(b.times, waits, b.sinks, values, dense, state, out, rows, kOutWindows);
  LAB_CHECK(cudaDeviceSynchronize());
  const size_t per = static_cast<size_t>(2 * kGroup) * (kLists * 8);
  std::vector<float> got_state(static_cast<size_t>(b.sms) * per);
  std::vector<uint32_t> got(words);
  LAB_CHECK(cudaMemcpy(got_state.data(), state, got_state.size() * sizeof(float), cudaMemcpyDeviceToHost));
  LAB_CHECK(cudaMemcpy(got.data(), out, words * 4, cudaMemcpyDeviceToHost));
  long long wrong_state = 0, wrong_out = 0;
  //: the host's: s[box][m][a][b], read at each window's start through its image, then the window folded in
  std::vector<long long> s(2 * kChannels * kLists * kLists);
  const auto at = [](int box, int m, int a, int bb) { return ((box * kChannels + m) * kLists + a) * kLists + bb; };
  for (int blk = 0; blk < b.sms; ++blk) {
    std::fill(s.begin(), s.end(), 0);
    for (int w = 0; w < kOutWindows; ++w) {
      for (int token = 0; token < kWindow; ++token)
        for (int m = 0; m < kChannels; ++m) {
          const int bb = token / 16;
          const uint32_t rt = reader_of(blk, w * kWindow + token);
          long long want = 0;
          for (int box = 0; box < 2; ++box)
            for (int a = 0; a < kLists; ++a)
              want += image_host(s[at(box, m, a, bb)]) * sparse_of(static_cast<int>(rt)) * dense_of(static_cast<int>(reader_row(rt, box)), a);
          const size_t row = (static_cast<size_t>(blk) * kOutWindows + w) * kWindow + token;
          const uint32_t word = got[row * (kChannels / 2) + m / 2];
          wrong_out += lab::half_to_float(static_cast<uint16_t>(word >> (16 * (m & 1)))) != static_cast<float>(lab::f16_of(want));
        }
      for (int box = 0; box < 2; ++box)
        for (int e = 0; e < kWindow; ++e) {
          const int t = static_cast<int>(listed_token((blk * 2 + box) % kColumns, w * kWindow + e));
          //: the entry's list: the list whose count's run it falls in
          int a = 0;
          for (int before = packed_count((blk * 2 + box) % kColumns, w % kCallWindows, 0); before <= e; )
            before += packed_count((blk * 2 + box) % kColumns, w % kCallWindows, ++a);
          for (int m = 0; m < kChannels; ++m)
            for (int bb = 0; bb < kLists; ++bb)
              s[at(box, m, a, bb)] += value_of(t, m) * sparse_of(t) * dense_of(t, bb);
        }
    }
    for (int tid = 0; tid < 2 * kGroup; ++tid) {
      const int box = tid >> 7, rows_of = 16 * ((tid >> 5) & 3) + (tid & 31) / 4, pair = tid & 3;
      for (int a = 0; a < kLists; ++a)
        for (int i = 0; i < 8; ++i) {
          const int m = rows_of + 8 * ((i >> 1) & 1), bb = 8 * (i >> 2) + 2 * pair + (i & 1);
          wrong_state += got_state[blk * per + static_cast<size_t>(tid) * (kLists * 8) + a * 8 + i]
                         != static_cast<float>(s[at(box, m, a, bb)]);
        }
    }
  }
  b.check(name, wrong_state == 0 && wrong_out == 0,
          std::to_string(wrong_state) + " of " + std::to_string(got_state.size()) + " state values and "
              + std::to_string(wrong_out) + " of " + std::to_string(2 * words) + " output values differ");
  const lab::Samples cycles = b.variant(
      [&] { kernel<<<b.sms, kThreads, kSmemBytes>>>(b.times, waits, b.sinks, values, dense, state, out, rows, kTimed); },
      [&] { return b.span(kWarps) / kTimed; });
  b.record(name, "cycles a window a CTA", cycles);
  if constexpr (TRACE != 0) {
    //: a warp's role, and the sites' names: a barrier's offset over 8, or a phase's
    std::vector<std::string> roles;
    for (int w = 0; w < kWarps; ++w)
      roles.push_back(32 * w < kReadAt ? "fold box " + std::to_string(32 * w / kGroup) : 32 * w < kDrainAt ? "readout" : "drain");
    std::vector<std::pair<uint32_t, std::string>> legend;
    const auto site = [&](int offset, const std::string& what) { legend.push_back({static_cast<uint32_t>(offset) / 8u, what}); };
    for (int k = 0; k < kChunks; ++k) {
      site(kBarsAt + kTilesReady + 8 * k, "tile group " + std::to_string(k) + " published");
      site(kBarsAt + kTilesRead + 8 * k, "tile group " + std::to_string(k) + " read");
    }
    for (int p = 0; p < 2; ++p) {
      site(kBarsAt + kPanelReady + 8 * p, "panel " + std::to_string(p) + " filled");
      site(kBarsAt + kPanelFreed + 8 * p, "panel " + std::to_string(p) + " drained");
    }
    legend.insert(legend.end(), {{kMarkWindow, "window"}, {kMarkSteps, "issuing steps"}, {kMarkPanel, "storing the panel"},
                                 {kMarkAdds, "adding"}, {kStepsRead, "readout steps"}, {kMarkEntry, "entry"},
                                 {kMarkExit, "exit"}});
    //: two CTAs of the grid, or its one
    if (b.sms > 1) b.trace(name, rows, kWarps, {0, b.sms / 2}, roles, legend);
    else b.trace(name, rows, kWarps, {0}, roles, legend);
    LAB_CHECK(cudaFree(rows));
  }
}

int main(int argc, char** argv) {
  using namespace carry_token;
  lab::Bench b(argc, argv);
  //: the rows: their kernels' selectors -- the plain kernel, its launch's edges, the coarse row, its four detail
  //: rows and every stamp
  struct Row { int trace, detail; };
  constexpr Row kRows[] = {{0, 0}, {1, 0}, {2, 0}, {2, kWaitTilesRead}, {2, kWaitTilesReady}, {2, kWaitPanelReady},
                           {2, kWaitPanelFreed}, {3, 0}};
  for (const Row& row : kRows) {
    const std::string k = lab::kernel("carry_token::window", {row.trace, row.detail});
    //: the fold's 32 steps, one block, and the readout's 16: two groups of 4 digits, each digit's two steps
    b.expect(k, "HGMMA", "==", kLists * kBlockRounds + 2 * 2 * kChunkDigits);
    //: the fold's conversions are in its own trip, beside its steps: what the row measures
    b.expect(k, "STEP_XU", "<=", 40);
    //: the readout's panel is read by the drain warps' own loads, not by a step: its arrival needs no fence, and a
    //: chunk of 8 digits arrives at both panels, each behind its stores
    b.expect(k, "STORE_UNFENCED", "<=", 2);
    //: the readout's trip holds its coefficients' rounds and their load beside its steps, by design: the tensor
    //: pipe is idle most of a window
    b.expect(k, "STEP_XU", ">=", 0);
    b.expect(k, "STEP_GLOBAL", ">=", 0);
    b.exempt(k, "a composition probe: the carry's roles in one CTA, no kernel's step");
    //: once a slot or a chunk a role: a `__syncwarp` before a one-lane arrival is a convergence call, and each
    //: barrier's wait a region -- three roles' in one kernel
    b.expect(k, "LOOP_CONVERGENCE", "<=", 64);
    //: a wait's and a readout phase's counts are sums the toolkit may keep in uniform registers
    b.expect(k, "CLOCK_UNIFORM", "<=", 24);
    //: the roles' loops hold their steps beside their barriers' waits: the waits' branches are the steps'
    b.expect(k, "BSSY", ">=", 0);
  }
  if (b.expect_only) return 0;
  const Buffers held = buffers(b);
  run<0>(b, held);
  run<1>(b, held);
  run<2>(b, held);
  run<2, kWaitTilesRead>(b, held);
  run<2, kWaitTilesReady>(b, held);
  run<2, kWaitPanelReady>(b, held);
  run<2, kWaitPanelFreed>(b, held);
  run<3>(b, held);
  return 0;
}
