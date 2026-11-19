// measure/lab/experiments/carry_token.cu -- QUESTION: the carry's token-stream form -- a window's value lines landed once
// as the stream holds them and read into the fold's steps by `ldmatrix`, the fold expanding only its coefficients, the
// readout mirrored beside it: what does a window cost a CTA, which role waits on which, and what do the fold's fill
// and its steps cost apart? And what is the order of a window's entries worth -- placed so the 8 tokens a lane group
// reads for a step are on 8 bank groups, against token order? METHOD: a CTA an SM of four warpgroups, 512 threads,
// registers traded by role: TWO FOLD warpgroups at 176 registers, a 16 x 16 box each, its state 16 accumulator sets
// of 8 in registers; ONE READOUT warpgroup at 96; ONE DRAIN warpgroup of four warps at 64. A window is a box's 256
// entries (16 lists of 8 to 24) and 256 tokens read the pair. The fold's window LANDS by two bulk copies behind one
// barrier, issued under the last steps of the window before: a TABLE -- an entry's word and sparse byte, a lane's row
// of the 32 steps' token positions, the lists' counts -- and the window's VALUE LINES, an entry's 8 at its position. A
// fold thread loads its two entries' dense lines from global memory and expands each by its sparse byte to the two
// lines of its step's coefficient tile; no value line is moved. A step's A is read by `ldmatrix`, transposed, from the
// landed lines into registers -- a lane names its token's line -- and its B is its tile, addressed by the step's
// index; a step is a commit group, `SETS` sets of A held in turn (2 or 4, rows of each: the fold's registers hold no more). Each fold warpgroup PUBLISHES its state as a bf16
// image, 16 tiles, a tile group of 4 behind the readout, a line a channel's 8 fold digits. The readout is MIRRORED: a
// thread expands one (token, box)'s 16 bytes, loaded a pass ahead, to two lines of its digit's coefficient tile in
// shared memory, the tokens on the step's columns; a pass is 8 digits, two tile groups, 16 steps of 64 channels by 16
// tokens one commit group, and 32 sums a thread per group stored as halves of the panel's rows. The four drain warps
// add each token's row as one f16 line to global memory. The roles' waits are suspended (`try_wait`) and no role
// reads a clock inside it: the rows are the plain kernel, its launch's edges stamped (`TRACE` 1), every warp's phases
// stamped (2: the coarse row, which is also each of the four detail rows -- one barrier's waits and arrivals more,
// `DETAIL`), every warp's timeline stamped (3), and the plain and coarse kernels again on a window whose entries are
// in token order. The state and every output row are checked exactly in every row.
// TIMELINE: a fold warpgroup's trip is a window: the image a tile group at a time, four -- the wait on the readout's
// read of that group a window back, its 4 stores a thread of 16 bytes (two lines of two halves), their fence, the
// group's arrival -- then the wait on its table and value lines; a thread's two entries in turn, each its dense line
// loaded and expanded to its two coefficient lines; a list's last step's pad lines zeroed by its count; a lane's row
// of positions, two lines, loaded and held; the fence, the warpgroup's barrier; 32 steps, each its lines read by
// `ldmatrix`, the fence, the step, its commit and the wait on the step before; the warpgroup's barrier, the next window's copies; the last
// wait. The readout's trip is a pass: the waits on its two tile groups, a thread's 16 bytes expanded to two lines of
// each group's tile (the bytes loaded a pass before), the fence, the warpgroup's barrier, 16 steps in one commit
// group, the next pass's bytes issued under them, the wait, the groups released; then a group at a time the wait on
// the panel two groups back, its 32 sums rounded to f16 pairs and stored, the arrival. A drain warp's trip is a group
// of 64 tokens, two batches of 8 a warp: the wait on the panel, the chunk's 16 loads, its 16 adds, the arrival. Live:
// a fold thread's 128 accumulators beside its positions' 8 words and two sets of A's 8, and in its fill an entry's
// word, line and products; the readout's 64 sums (a chunk of 8 digits) and the 8 words of its bytes; a drain's 16.
// see docs/internals/tools/lab.md#carry-token
#include "lab.cuh"

namespace carry_token {

constexpr int kLists = 16, kEntries = 16, kWindow = kLists * kEntries;
constexpr int kChannels = 64, kTableTokens = 65536, kRowWords = 32;
constexpr int kGroup = 128, kDrains = 4, kThreads = 4 * kGroup;
//: the roles' first threads: fold 0, fold 1, the readout, then the four drain warps
constexpr int kReadAt = 2 * kGroup, kDrainAt = 3 * kGroup;
//: THE REGISTERS A THREAD. A raise takes what another warpgroup released from the count the kernel is launched
//: at -- 128 a thread at 512 threads, the SM's whole file: the folds' 48 more each are the 32 fewer of the readout
//: and the 64 fewer of the drains
constexpr int kEntryRegs = 128, kFoldRegs = 176, kReadRegs = 96, kDrainRegs = 64;
//: an isolation row whose read side is idle: the folds take what the idle warpgroups do not need
constexpr int kAloneFoldRegs = 192, kAloneRestRegs = 64;
static_assert(2 * (kAloneFoldRegs - kEntryRegs) == 2 * kEntryRegs - 2 * kAloneRestRegs);
static_assert(kThreads * kEntryRegs == 65536 && 2 * (kFoldRegs - kEntryRegs) == 2 * kEntryRegs - kReadRegs - kDrainRegs);

//: A POOL SLOT, in lines: a box's pool is two
constexpr int kBlockSlotLines = 1536;
constexpr int kPoolBytes = kBlockSlotLines * 16;
//: AN IMAGE TILE (the readout's operand a read digit), K-major: a line a channel's 8 fold digits -- one fold
//: thread's own values, a store -- a brick 8 channels' low digits then their high, bricks 17 lines, tiles 137 lines
//: apart; a box's image has room for 16 tiles 161 lines apart
constexpr int kBrickLines = 17, kImageTile = 137 * 16, kImageRoom = 161 * 16, kBoxBytes = kLists * kImageRoom;
//: a panel row: a token's 64 f16 as its line will be added, rows 272 bytes apart
constexpr int kChunkDigits = 4, kTokens = 16 * kChunkDigits, kChunks = kLists / kChunkDigits, kRowBytes = 272,
              kPanelBytes = kTokens * kRowBytes, kBatch = 8;
//: THE SHARED MEMORY: the four pool slots (box 0's, box 1's), the two boxes' images, the two panels, the barriers;
//: then the readout's coefficient tiles, a (digit, box) each: 16 tokens by 16 fold digits, K-major, a line a token's
//: 8 digits, a brick 8 tokens' low digits then their high, bricks 17 lines;
//: the box's zeros, two bricks of zeros 16 lines apart (an absent step's B); and a box's window's TABLE, landed
constexpr int kImageAt = 4 * kPoolBytes, kPanelAt = kImageAt + 2 * kBoxBytes, kBarsAt = kPanelAt + 2 * kPanelBytes;
constexpr int kCoefTile = 2 * 17 * 16, kCoefAt = kBarsAt + 128, kZeroAt = kCoefAt + 2 * 2 * 4 * kCoefTile;
constexpr int kOneAt = kZeroAt + 2 * 2 * 16 * 16, kOneBytes = 160 * 16, kSmemBytes = kOneAt + 2 * kOneBytes;
//: A BOX'S POOL, in lines: the window's value lines as they land -- a brick a run of 256, entry `e`'s line `c` at
//: `256 c + e`, so the bank group of a line is its entry's position mod 8 -- then the 32 steps' coefficient tiles, a
//: step 16 lines of its entries' low digits and, `kTilePitch` on, 16 of their high
constexpr int kValueLines = 16 * kGroup, kValueBytes = kValueLines * 16, kTileAt = kValueLines, kTilePitch = 32 * 16;
static_assert(kTileAt + 2 * kTilePitch == 2 * kBlockSlotLines);
static_assert(kSmemBytes <= 232448);
//: the barriers, 8 bytes each: an image's tile group published and read (a group is the 4 read digits `8 h + c + 2 j`
//: of a chunk `2 h + c`: every fold thread holds a quarter of it); a panel filled and drained; a box's window landed
constexpr int kTilesReady = 0, kTilesRead = 32, kPanelReady = 64, kPanelFreed = 80, kPoolLanded = 96;
//: the named barriers: a box's fold warpgroup together behind its stores, and the readout's
constexpr int kPoolId = 1, kReadId = 11;
constexpr int kOutWindows = 4;
//: a window's steps are this many rounds of its 16 lists
constexpr int kBlockRounds = 2;
//: A WINDOW IN GLOBAL MEMORY, in lines, entry `e` the one at position `e`: its TABLE, which lands -- an entry's
//: word, 256; the entries' sparse bytes; a lane's row of the 32 steps' positions, 16 rows of two lines; the lists'
//: counts, a line; which lists have a second step, a word -- then the entries' dense lines, which threads load, and
//: their value lines, a brick a run of 256, which land in the box's pool
constexpr int kOneSparse = 64, kOnePos = 80, kOneCounts = 112, kOneLive = 113, kOneDense = 160, kOneValues = kOneDense + 2 * kGroup,
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
  kWaitPanelFreed = 8,
  //: the fold's steps by eighths, and what follows its last step: four marks more in its steps phase
  kSplitSteps = 16
};
//: AN ISOLATION ROW's change to the kernel, to find what a step costs beside what: the read side idle (the folds
//: publish their image and wait on nobody); a step's A read by the step itself from 16 consecutive landed positions
//: (not its tokens: the row's sums are not the kernel's and it is not checked); the 32 steps one commit group
enum Probe : int { kAlone = 1, kFixedA = 2, kOneGroup = 4 };
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
//: a phase's or an edge's site
enum Site : uint32_t { kMarkWindow = 0xc000u, kMarkExpand = 0xc001u, kMarkSteps = 0xc002u, kMarkPanel = 0xc003u, kMarkAdds = 0xc004u,
                       kStepsRead = 0xc007u, kMarkEntry = 0xc008u, kMarkExit = 0xc009u,
                       kMarkSteps8 = 0xc00au, kMarkSteps16 = 0xc00bu, kMarkSteps24 = 0xc00cu, kMarkTail = 0xc00du };

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
//: A FOLD WARPGROUP: its box's state; a window is the image, then the wait on its table and value lines, its entries'
//: coefficients expanded, and 32 steps whose A is `ldmatrix`'s from the landed lines, `SETS` of A in flight
template <int TRACE, int DETAIL, int SETS, int PROBE>
__device__ __forceinline__ void fold_role(uint8_t* smem, uint32_t base, long long* times,
                                          float* state, const uint32_t* values, unsigned long long* trace, int windows) {
  const int tid = threadIdx.x, box = tid >> 7, warp = (tid >> 5) & 3, lane = tid & 31, q = lane >> 2, t = lane & 3;
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
    //: the window's table and its value lines, two copies of the warpgroup's first thread behind one barrier
    if ((threadIdx.x & (kGroup - 1)) == 0) {
      const int box_of = threadIdx.x >> 7;
      const uint32_t landed = base + kBarsAt + kPoolLanded + 8 * box_of;
      const uint4* const from = reinterpret_cast<const uint4*>(values) + one_line(w);
      lab::mbar_expect(landed, kOneBytes + kValueBytes);
      lab::bulk_load(base + kOneAt + box_of * kOneBytes, from, kOneBytes, landed);
      lab::bulk_load(base + 2 * box_of * kPoolBytes, from + kOneValues, kValueBytes, landed);
    }
  };
  Tracer<TRACE, DETAIL> tr;
  //: the launch's edges, on both clocks: the warp's entry at its role's start, its exit at its end
  tr.bind(trace);
  tr.edge(kMarkEntry);
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
      if constexpr ((PROBE & kAlone) == 0)
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
      if constexpr ((PROBE & kAlone) == 0) arrive(bars + kTilesReady + 8 * k, tr, kWaitTilesReady);
    }
    //: THE WINDOW: its table and value lines landed; a thread's two entries' coefficients, each a dense line loaded
    //: and expanded by its sparse byte to the two lines of its step's tile its word names; the pads; then 32 steps
    lab::mbar_wait_suspended(bars + kPoolLanded + 8 * box, w & 1);
    tr.stamp(warpline::kPhaseBegin, kMarkExpand);
    const uint32_t pool = 2 * box * kPoolBytes;
    const int g = tid & (kGroup - 1);
    uint8_t* const image = smem + pool;
    const uint8_t* const table = smem + kOneAt + box * kOneBytes;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int e = g + half * kGroup;
      const uint32_t word = reinterpret_cast<const uint32_t*>(table)[e];
      const float weight = static_cast<float>(table[kOneSparse * 16 + e]);
      //: (the window's line in the views' table is held as 32 bits: no address of it outlives the entry)
      const uint4 landed = (reinterpret_cast<const uint4*>(values) + lab::held(one_line(w)) + e)[kOneDense];
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
    }
    //: THE PADS, a thread a place of two lists' last steps: where a list's entries end inside a step, the places
    //: behind them in the step's tile
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const uint32_t j = static_cast<uint32_t>(8 * half + (g >> 4)), n = table[kOneCounts * 16 + j], tail = n & 15u,
                     k = static_cast<uint32_t>(g) & 15u;
      if (tail != 0 && k >= tail) {
        const uint32_t at = kTileAt + (16 * (n >> 4) + j) * 16 + k;
        *reinterpret_cast<uint4*>(image + at * 16) = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(image + (at + kTilePitch) * 16) = make_uint4(0, 0, 0, 0);
      }
    }
    //: A LANE'S TOKENS, a byte a step: the position of its token of each of the window's 32 steps -- lane `8 m + x`
    //: reads token `8 (m / 2) + x` of a step -- two lines of the table, held; and the lists whose second step is there
    const uint4* const row = reinterpret_cast<const uint4*>(table) + kOnePos + 2 * ((lane >> 4) * 8 + (lane & 7));
    const uint4 lo = row[0], hi = row[1];
    const uint32_t pos[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    const uint32_t live = *reinterpret_cast<const uint32_t*>(table + kOneLive * 16);
    lab::bulk_fence();
    lab::bar_sync(kPoolId + box, kGroup);
    tr.stamp(warpline::kPhaseBegin, kMarkSteps);
    //: THE STEPS: A is the step's 16 tokens' value lines of the warp's 16 channels, read by `ldmatrix` transposed --
    //: a lane names its token's line of the channel octet `2 warp + (lane / 8) % 2` -- into the registers the step
    //: takes it from; B is the step's tile, its address the first tile's and the step's index (the zeros where a
    //: list has no second step). `SETS` sets of A are held in turn: a step waits on the one `SETS` before it alone
    const uint32_t rows_at = base + pool + static_cast<uint32_t>(2 * warp + ((lane >> 3) & 1)) * (kValueBytes / 8);
    const uint64_t tiles = lab::wg_desc(base + pool + kTileAt * 16, 8, kTilePitch), zeros = lab::wg_desc(base + kZeroAt + box * 512, 8, 16);
    uint32_t a[SETS][4];
    lab::wg_fence();
#pragma unroll
    for (int r = 0; r < kBlockRounds; ++r)
#pragma unroll
      for (int j = 0; j < kLists; ++j) {
        const int step = 16 * r + j;
        if constexpr ((DETAIL & kSplitSteps) != 0) {
          if (step == 8) tr.stamp(warpline::kPhaseBegin, kMarkSteps8);
          if (step == 16) tr.stamp(warpline::kPhaseBegin, kMarkSteps16);
          if (step == 24) tr.stamp(warpline::kPhaseBegin, kMarkSteps24);
        }
        //: a step's lines are read at its own issue: the edge keeps the window's reads from one register set
        if constexpr ((PROBE & kFixedA) == 0) asm volatile("" ::: "memory");
        const uint64_t b = r == 0 || ((live >> j) & 1u) != 0 ? tiles + static_cast<uint64_t>(16 * step) : zeros;
        if constexpr ((PROBE & kFixedA) != 0) {
          lab::wg_bf16_n16_ss_mn(d[j], lab::wg_desc(base + pool + 16u * 16u * static_cast<uint32_t>(step & 15), 8, 256), b, w != 0 || r != 0);
        } else {
          lab::ldsm_t(a[step % SETS], rows_at + ((pos[step >> 2] >> (8 * (step & 3))) & 255u) * 16u);
          lab::wg_fence();
          lab::wg_bf16_n16_rs(d[j], a[step % SETS], b, w != 0 || r != 0);
        }
        if constexpr ((PROBE & kOneGroup) == 0) {
          lab::wg_commit();
          lab::wg_wait<SETS - 1>();
        }
      }
    if constexpr ((PROBE & kOneGroup) != 0) lab::wg_commit();
    if constexpr ((DETAIL & kSplitSteps) != 0) tr.stamp(warpline::kPhaseBegin, kMarkTail);
    //: THE NEXT WINDOW'S COPIES, behind every warp's last read of this window's lines -- the warpgroup's barrier is
    //: that edge -- and under its last steps
    lab::bar_sync(kPoolId + box, kGroup);
    if (w + 1 < windows) issue_one(w + 1);
    lab::wg_wait<0>();
  }
#pragma unroll
  for (int a = 0; a < kLists; ++a)
#pragma unroll
    for (int i = 0; i < 8; ++i) lab::wg_hold(d[a][i]);
  lab::WarpMark::stop(times);
  tr.edge(kMarkExit);
  tr.close(base);
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
__device__ __forceinline__ void read_role(uint8_t* smem, uint32_t base, long long* times,
                                          const uint32_t* dense, unsigned long long* trace, int windows) {
  Tracer<TRACE, DETAIL> tr;
  //: the launch's edges, on both clocks: the warp's entry at its role's start, its exit at its end
  tr.bind(trace);
  tr.edge(kMarkEntry);
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
}

//: A DRAIN WARP: a chunk's 16 tokens of its residue in two batches of 8; a lane holds a token's two channels and
//: adds one f16 pair
template <int TRACE, int DETAIL>
__device__ __forceinline__ void drain_role(uint8_t* smem, uint32_t base, long long* times,
                                           uint32_t* out, unsigned long long* trace, int windows) {
  Tracer<TRACE, DETAIL> tr;
  //: the launch's edges, on both clocks: the warp's entry at its role's start, its exit at its end
  tr.bind(trace);
  tr.edge(kMarkEntry);
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
}

//: THE KERNEL of a row: `TRACE` and `DETAIL` select its stamps and nothing else -- a detail row is its coarse row but
//: for those stamps; `SETS` the sets of A the fold's steps hold in flight
template <int TRACE, int DETAIL, int SETS, int PROBE>
__global__ void __launch_bounds__(kThreads, 1)
    window(long long* times, const uint32_t* values, const uint32_t* dense, float* state,
           uint32_t* out, unsigned long long* trace, int windows) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x;
  const uint32_t base = lab::smem(smem);
  //: the images cleared: the state is zero at the call's start
#pragma unroll 1
  for (int i = tid; i < 2 * kBoxBytes / 4; i += kThreads) *reinterpret_cast<uint32_t*>(smem + kImageAt + 4 * i) = 0u;
  if (tid == 0) {
#pragma unroll
    for (int p = 0; p < 2; ++p) lab::mbar_init(base + kBarsAt + kPoolLanded + 8 * p, 1);
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
  }
  lab::bulk_fence();
  __syncthreads();

  //: EACH ROLE'S WHOLE BODY IN ITS OWN BRANCH, a warpgroup's registers set at the branch's head: the folds' raise is
  //: the release of the readout and of the drains' warpgroup
  if (tid < kReadAt) {
    lab::regs_up<(PROBE & kAlone) != 0 ? kAloneFoldRegs : kFoldRegs>();
    fold_role<TRACE, DETAIL, SETS, PROBE>(smem, base, times, state, values, trace, windows);
  } else if (tid < kDrainAt) {
    lab::regs_down<(PROBE & kAlone) != 0 ? kAloneRestRegs : kReadRegs>();
    if constexpr ((PROBE & kAlone) != 0) lab::WarpMark::start(times), lab::WarpMark::stop(times);
    else read_role<TRACE, DETAIL>(smem, base, times, dense, trace, windows);
  } else {
    lab::regs_down<(PROBE & kAlone) != 0 ? kAloneRestRegs : kDrainRegs>();
    if constexpr ((PROBE & kAlone) != 0) lab::WarpMark::start(times), lab::WarpMark::stop(times);
    else drain_role<TRACE, DETAIL>(smem, base, times, out, trace, windows);
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

//: A CALL'S BUFFERS: the window views, the readers' dense table, the state and the output, each sized to the grid
struct Buffers {
  //: the views with their entries placed for the banks, and in token order
  uint32_t *values[2], *dense, *out;
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
  //: THE VIEWS: each (column, window)'s view -- the window's 256 entries, an entry's word (its two lines of its
  //: step's tile), sparse byte, dense line and value lines at its POSITION; a lane's row of the steps' positions; the
  //: lists' counts and which have a second step. `placed`: the positions are chosen so the 8 tokens a lane group
  //: reads for a step are on 8 bank groups (distinct mod 8); else an entry's position is its rank in token order
  const size_t room = static_cast<size_t>(kOneLines) * 16;
  struct Entry { uint32_t t; int j, k; };
  const auto build = [&](bool placed) {
    std::vector<uint8_t> views(static_cast<size_t>(kColumns) * kCallWindows * room, 0);
    for (int c = 0; c < kColumns; ++c)
      for (int w = 0; w < kCallWindows; ++w) {
        uint8_t* const view = views.data() + (static_cast<size_t>(c) * kCallWindows + w) * room;
        int n[kLists], e = 0;
        std::vector<Entry> entries;
        for (int j = 0; j < kLists; ++j) {
          n[j] = packed_count(c, w, j);
          for (int k = 0; k < n[j]; ++k)
            entries.push_back({listed_token(static_cast<uint32_t>(c), static_cast<uint32_t>(w * kWindow + e + k)), j, k});
          e += n[j];
        }
        std::stable_sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.t < b.t; });
        if (entries.size() != static_cast<size_t>(2 * kGroup)) {
          std::fprintf(stderr, "carry_token: a window of %zu entries is not this lab's\n", entries.size());
          std::exit(2);
        }
        //: an entry's position: its rank, or one of a bank group no other entry of its step's octet has
        int pos[2 * kGroup], group_of[2 * kGroup], octet_of[2 * kGroup], room_of[8], next_of[8],
            used[kLists * kBlockRounds * 2] = {};
        for (int m = 0; m < 8; ++m) room_of[m] = 2 * kGroup / 8, next_of[m] = m;
        for (int r = 0; r < 2 * kGroup && placed; ++r) {
          const Entry& at = entries[static_cast<size_t>(r)];
          const int octet = octet_of[r] = (16 * (at.k >> 4) + at.j) * 2 + ((at.k >> 3) & 1);
          int best = -1;
          for (int m = 0; m < 8; ++m)
            if (!(used[octet] >> m & 1) && room_of[m] > 0 && (best < 0 || room_of[m] > room_of[best])) best = m;
          //: no group with room is free of this octet: an earlier entry of another octet gives up a full group this
          //: octet lacks for one with room that its own octet lacks
          for (int full = 0; full < 8 && best < 0; ++full) {
            if (used[octet] >> full & 1) continue;
            for (int open = 0; open < 8 && best < 0; ++open) {
              if (room_of[open] == 0) continue;
              for (int e2 = 0; e2 < r && best < 0; ++e2)
                if (group_of[e2] == full && !(used[octet_of[e2]] >> open & 1)) {
                  used[octet_of[e2]] ^= 1 << full | 1 << open, group_of[e2] = open, --room_of[open], ++room_of[full];
                  best = full;
                }
            }
          }
          if (best < 0) { std::fprintf(stderr, "carry_token: no bank group left for an entry\n"); std::exit(2); }
          used[octet] |= 1 << best, --room_of[best], group_of[r] = best;
        }
        for (int r = 0; r < 2 * kGroup; ++r)
          if (placed) pos[r] = next_of[group_of[r]], next_of[group_of[r]] += 8;
          else pos[r] = r;
        //: a step's 16 places: an entry's position, or where the list has none there any position (its bank group
        //: one the octet's entries leave, where placed); the table's row `k` holds place `k` of the 32 steps
        uint8_t place[kLists * kBlockRounds][16];
        bool has[kLists * kBlockRounds][16] = {};
        for (int r = 0; r < 2 * kGroup; ++r) {
          const Entry& at = entries[static_cast<size_t>(r)];
          const int step = 16 * (at.k >> 4) + at.j, k = at.k & 15;
          place[step][k] = static_cast<uint8_t>(pos[r]), has[step][k] = true;
          reinterpret_cast<uint32_t*>(view)[pos[r]] =
              static_cast<uint32_t>(kTileAt + 16 * step + k) | static_cast<uint32_t>(kTileAt + kTilePitch + 16 * step + k) << 16;
          view[kOneSparse * 16 + pos[r]] = static_cast<uint8_t>(sparse_of(static_cast<int>(at.t)));
          for (int dd = 0; dd < 16; ++dd) view[(kOneDense + pos[r]) * 16 + dd] = static_cast<uint8_t>(dense_of(static_cast<int>(at.t), dd));
          for (int br = 0; br < 8; ++br)
            std::memcpy(view + (kOneValues + br * 2 * kGroup + pos[r]) * 16, &host_values[static_cast<size_t>(at.t) * kRowWords + 4 * br], 16);
        }
        for (int step = 0; step < kLists * kBlockRounds; ++step)
          for (int k = 0; k < 16; ++k) {
            if (!has[step][k]) {
              int m = 0;
              if (placed) {
                while (used[2 * step + (k >> 3)] >> m & 1) ++m;
                used[2 * step + (k >> 3)] |= 1 << m;
              }
              place[step][k] = static_cast<uint8_t>(m);
            }
            view[(kOnePos + 2 * k) * 16 + step] = place[step][k];
          }
        uint32_t live = 0;
        for (int j = 0; j < kLists; ++j) view[kOneCounts * 16 + j] = static_cast<uint8_t>(n[j]), live |= static_cast<uint32_t>(n[j] > 16) << j;
        std::memcpy(view + kOneLive * 16, &live, 4);
      }
    uint8_t* vdev = nullptr;
    LAB_CHECK(cudaMalloc(&vdev, views.size()));
    LAB_CHECK(cudaMemcpy(vdev, views.data(), views.size(), cudaMemcpyHostToDevice));
    return reinterpret_cast<uint32_t*>(vdev);
  };
  held.values[0] = build(true);
  held.values[1] = build(false);
  LAB_CHECK(cudaMalloc(&held.dense, host_dense.size() * 4));
  LAB_CHECK(cudaMemcpy(held.dense, host_dense.data(), host_dense.size() * 4, cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMalloc(&held.state, sizeof(float) * b.sms * 2 * kGroup * (kLists * 8)));
  LAB_CHECK(cudaMalloc(&held.out, static_cast<size_t>(b.sms) * kOutWindows * kWindow * (kChannels / 2) * 4));
  return held;
}

//: A ROW: the kernel `window<TRACE, DETAIL>`, its checked call and its timed launches
template <int TRACE, int DETAIL = 0, int ORDER = 0, int SETS = 2, int PROBE = 0>
void run(lab::Bench& b, const Buffers& held) {
  using namespace carry_token;
  constexpr int kTimed = 64;
  const std::string name = std::string("the token-stream carry: both sides") + (ORDER ? ", its entries in token order" : "")
                           + (SETS != 2 ? ", " + std::to_string(SETS) + " sets of A in flight" : std::string())
                           + (PROBE & kAlone ? ", the read side idle" : "")
                           + (PROBE & kFixedA ? ", a step's A read from 16 landed positions" : "")
                           + (PROBE & kOneGroup ? ", the 32 steps one commit group" : "")
                           + (TRACE & 2 ? (TRACE & 1 ? ", every warp's timeline stamped" : ", every warp's phases stamped")
                                        : TRACE & 1 ? ", the launch's two edges stamped" : "")
                           + (DETAIL & kWaitTilesRead ? ", the waits on the tile groups read stamped" : "")
                           + (DETAIL & kWaitTilesReady ? ", the waits on the tile groups published stamped" : "")
                           + (DETAIL & kWaitPanelReady ? ", the waits on the panels filled stamped" : "")
                           + (DETAIL & kWaitPanelFreed ? ", the waits on the panels drained stamped" : "")
                           + (DETAIL & kSplitSteps ? ", the fold's steps by eighths stamped" : "");
  constexpr int kWarps = kThreads >> 5;
  auto kernel = window<TRACE, DETAIL, SETS, PROBE>;
  uint32_t* const values = held.values[ORDER];
  uint32_t* const dense = held.dense;
  uint32_t* const out = held.out;
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
  kernel<<<b.sms, kThreads, kSmemBytes>>>(b.times, values, dense, state, out, rows, kOutWindows);
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
  if constexpr (PROBE != 0) b.unchecked(name, "an isolation row: its kernel is not the carry's");
  else b.check(name, wrong_state == 0 && wrong_out == 0,
          std::to_string(wrong_state) + " of " + std::to_string(got_state.size()) + " state values and "
              + std::to_string(wrong_out) + " of " + std::to_string(2 * words) + " output values differ");
  const lab::Samples cycles = b.variant(
      [&] { kernel<<<b.sms, kThreads, kSmemBytes>>>(b.times, values, dense, state, out, rows, kTimed); },
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
    legend.insert(legend.end(), {{kMarkWindow, "window"}, {kMarkExpand, "expanding"}, {kMarkSteps, "issuing steps"}, {kMarkPanel, "storing the panel"},
                                 {kMarkAdds, "adding"}, {kStepsRead, "readout steps"}, {kMarkSteps8, "issuing steps 8 on"}, {kMarkSteps16, "issuing steps 16 on"},
                                 {kMarkSteps24, "issuing steps 24 on"}, {kMarkTail, "after the steps"}, {kMarkEntry, "entry"},
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
  struct Row { int trace, detail, sets, probe; };
  constexpr Row kRows[] = {{0, 0, 2, 0}, {1, 0, 2, 0}, {2, 0, 2, 0}, {2, kWaitTilesRead, 2, 0}, {2, kWaitTilesReady, 2, 0},
                           {2, kWaitPanelReady, 2, 0}, {2, kWaitPanelFreed, 2, 0}, {3, 0, 2, 0}, {0, 0, 4, 0}, {2, 0, 4, 0},
                           {2, kSplitSteps, 2, 0}, {2, 0, 2, kAlone}, {2, 0, 2, kFixedA}, {2, 0, 2, kFixedA | kOneGroup},
                           {2, 0, 2, kAlone | kFixedA | kOneGroup}};
  for (const Row& row : kRows) {
    const std::string k = lab::kernel("carry_token::window", {row.trace, row.detail, row.sets, row.probe});
    //: the fold's 32 steps, one block, and the readout's 16: two groups of 4 digits, each digit's two steps
    b.expect(k, "HGMMA", "==", kLists * kBlockRounds + (row.probe & kAlone ? 0 : 2 * 2 * kChunkDigits));
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
  //: the plain and the coarse kernels on a window whose entries are in token order
  run<0, 0, 1>(b, held);
  run<2, 0, 1>(b, held);
  //: and with more sets of A in flight
  run<0, 0, 0, 4>(b, held);
  run<2, 0, 0, 4>(b, held);
  //: the steps by eighths; and the isolation rows
  run<2, kSplitSteps>(b, held);
  run<2, 0, 0, 2, kAlone>(b, held);
  run<2, 0, 0, 2, kFixedA>(b, held);
  run<2, 0, 0, 2, kFixedA | kOneGroup>(b, held);
  run<2, 0, 0, 2, kAlone | kFixedA | kOneGroup>(b, held);
  return 0;
}
