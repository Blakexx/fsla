// measure/lab/experiments/carry_token.cu -- QUESTION: the carry's token-stream form as a PACKET CONTRACT -- the carry
// blindly consumes a window's packet, a block of value lines and a header, and the gather lays the packet out however
// is best -- what does a window cost a CTA, which role waits on which, what do the fold's fill and its steps cost
// apart, and what does the read side cost the fold? METHOD: a CTA an SM of four warpgroups, 512 threads, registers
// traded by role: TWO FOLD warpgroups at 168 registers (176 in the grouped and per-column kernels and the
// `kSplit` row, the readout then at 112 and the drain at 48), a 16 x 16 box each, its state 16 accumulator sets of 8 in
// registers; ONE READOUT warpgroup at 120; ONE DRAIN warpgroup of four warps at 56. A window is a box's 256 entries
// (16 lists of 8 to 24) and 256 tokens read the pair. A PACKET lands by two bulk copies behind one barrier, issued
// behind the warpgroup's barrier that follows the wait on the window before's steps: its HEADER -- an entry's word and sparse byte, where each of the 32
// steps' A tile starts among the value lines, a pad's word for each place of a step no entry holds, which steps are there --
// and its VALUE LINES, a brick a run of 256. A fold thread loads its two entries' dense lines from global memory and
// expands each by its sparse byte to the two lines of its step's coefficient tile; a place of a tile no entry fills
// is a PAD's, its two lines written zero by a thread's third word; no value line is moved. A step reads its A from the landed lines -- 16 consecutive positions are
// an M|N-major tile, its address the block's and the header's start -- and its B from its tile, addressed by the
// step's index: nothing is loaded a step, and the 32 steps are one commit group. THE ONE RULE on a packet: a step's
// 16 tokens are consecutive positions. The host plays the gather with two per-list layouts: every entry its own value lines;
// and lists `2 p`, `2 p + 1` sharing four tokens, written once, the two lists' positions overlapping on them. GROUPED
// PACKETS (the grouped fold's): a group of four digits holds a token once, a step 16 tokens' value lines against a tile
// of 16 tokens by the group's 64 columns -- a token on one digit of its group (two passes of eight steps, a pass's
// tiles cleared), on two (two steps a group, half of a tile's lines no entry's),
// on all four (four steps, every line of a tile an entry's: nothing cleared), and on all 16 digits of
// its box (the four groups' steps on the box's one run of value lines); and the four-digit tokens a list a digit. Each
// fold warpgroup PUBLISHES its state as a bf16 image, 16 tiles, a tile group of 4 behind the readout, a line a
// channel's 8 fold digits. The readout is MIRRORED: a thread expands one (token, box)'s 16 bytes, loaded two passes
// ahead, to two lines of its digit's coefficient tile in shared memory, the tokens on the step's columns; a pass is 8
// digits, two tile groups, 16 steps of 64 channels by 16 tokens one commit group, and 32 sums a thread per group
// stored by 4 transposed matrix stores to the panel's rows. The four drain warps add each token's row as one f16 line to global memory.
// The roles' waits are suspended (`try_wait`) and no role reads a clock inside it: the rows are the plain kernel, its
// launch's edges stamped (`TRACE` 1), every warp's phases stamped (2: the coarse row, which is also each of the seven
// detail rows -- one barrier's waits and arrivals more, the packet's waits and its copies' issues, the fold's fill by
// phases, or its steps by eighths, `DETAIL`), every warp's timeline stamped (3), the plain row's long call by wall
// time (the card's clock, and a window's span), the plain and coarse kernels on the overlapping layout, the grouped kernels on
// their packets and the per-list kernel on the four-digit tokens, the per-list fold at the grouped kernel's registers
// (`kSplit`), THE PER-COLUMN FORM (`kColumn`: one kernel holding the per-list fold and the grouped one, a box's width word
// read from global memory before its window loop choosing its fold, any other word a trap) on packets of each width,
// the four-digit grouped fold with its tiles' room cleared (`kCleared`: issue work that fold does not need, a pair of rows
// that prices it composed against alone) and with each group's steps a run-time count of the packet's (`kCounted`), on
// the four-digit tokens laid in a group's two tile places and on a token on two digits of its group, or a voted loop on
// that count (`kLooped`: any count, on packets of one step a group, of two, and of four, two, one and one),
// a fold whose entries' dense lines are its own scattered loads from the tokens' table (`kTabled`: the packet holds a
// token's index an entry, two bytes, where it held the line; `kSpread` the same rows 32 KB apart, a 2 GB table; `kAhead`
// a window's rows loaded a window ahead, behind the entries of the window before),
// a readout whose passes are its packet's (`kListed`: a pass's 8 slots each name their read digit, their 16 readers and
// those readers' output lines, and a half's passes are the packet's count -- on the fixed readout's own readers, and on a
// half in two passes, each four slots and four pads),
// and thirteen ISOLATION rows (`kAlone`): a
// coarse kernel with the read side idle, the folds waiting on nobody. The state and every output row are checked
// exactly in every row but the isolation rows, whose kernels are not the carry's.
// TIMELINE: a fold warpgroup's trip is a window: the image a tile group at a time, four -- the wait on the readout's
// read of that group a window back, its 4 stores a thread of 16 bytes (two lines of two halves), their fence, the
// group's arrival -- then the wait on its packet; a thread's two entries in turn, each its dense line (loaded at the
// window's start, before the image) expanded to its two coefficient lines; a thread's pad's two lines written zero; the
// header's two lines of tile starts loaded and held; the fence, the warpgroup's barrier; 32 steps issued, the commit,
// the wait, the warpgroup's barrier; the next window's packet's copies. THE GROUPED FOLD's trip differs
// after the wait on its packet: a pass is its tiles cleared (16 stores a thread of 8 bytes), the fence and the
// warpgroup's barrier, a thread's entry of the pass expanded, the fence and the barrier, the pass's 8 steps of 64 columns
// one commit group, the wait, the warpgroup's barrier -- two passes a window where a token is on one digit of its group;
// where it is on all four, no clear: a thread's two entries expanded, the fence and the barrier, 4 steps one commit
// group, the wait, the barrier -- under `kCleared` opening with the clear, its fence and the warpgroup's barrier, and under
// `kCounted` a group's steps one straight block of one or two, chosen by its count, a byte of the header; under `kLooped` a
// loop a group on that count, its test a warp vote, a step's tile and start the window's next. Under `kTabled` an entry's
// dense line is two chained global loads at the window's start (its token's index from the packet's header, then the
// token's row); under `kAhead` those loads follow the entries' expansion, before the fence and the barrier, for the
// window after, and the two lines stay live through the steps. THE PER-COLUMN kernel reads its box's width word once, before the first window, and runs
// one of the two trips whole. The readout's trip is a pass: the waits on its two tile groups, a
// thread's 16 bytes expanded to two lines of each group's tile (the bytes loaded two passes before), the fence, the
// warpgroup's barrier, 16 steps in one commit group, the bytes of the pass after next issued under them, the wait, the groups
// released; then a group at a time the wait on the panel two groups back, its 32 sums rounded to f16 pairs and
// stored, the arrival. THE LISTED READOUT's pass is that trip in a loop a half on the packet's count, its test a warp
// vote: the lane's entry and its reader's row loaded two passes ahead through the packet (two chained global loads), a
// step's A the tile of its slot's digit, the half's arrival at its groups read after its last pass's steps; its drain
// warp holds a group's 16 lines of the packet (two loads, issued a group ahead) and adds where they say; a window's
// counts are loaded a window ahead in both roles: neither waits on a load of the packet.
// A drain warp's trip is a group of 64 tokens, two batches of 8 a warp: the wait on the panel,
// the chunk's 16 loads, its 16 adds, the arrival. Live: a fold thread's 128 accumulators beside the tile starts' 8
// words and, through the image, its two entries' dense lines' 8, and in its fill an entry's word and products; the
// readout's 64 sums (a chunk of 8 digits) and two passes' bytes and reader indices, 20 words; a drain's 16. The grouped fold
// holds a pass's tile starts in 2 words (1 where a window is 4 steps) and its kernel is 176 registers a fold thread, 112 the
// readout's and 48 a drain's; so is a `kAhead` kernel's.
// see docs/internals/tools/lab.md#carry-token
#include <chrono>

#include "lab.cuh"

namespace carry_token {

constexpr int kLists = 16, kEntries = 16, kWindow = kLists * kEntries;
constexpr int kChannels = 64, kTableTokens = 65536, kRowWords = 32;
constexpr int kGroup = 128, kDrains = 4, kThreads = 4 * kGroup;
//: the roles' first threads: fold 0, fold 1, the readout, then the four drain warps
constexpr int kReadAt = 2 * kGroup, kDrainAt = 3 * kGroup;
//: THE REGISTERS A THREAD. A raise takes what another warpgroup released from the count the kernel is launched
//: at -- 128 a thread at 512 threads, the SM's whole file: the folds' 40 more each are the 8 fewer of the readout
//: and the 72 fewer of the drains
constexpr int kEntryRegs = 128, kFoldRegs = 168, kReadRegs = 120, kDrainRegs = 56;
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
//: A PACKET IN GLOBAL MEMORY, in lines: its HEADER, which lands -- an entry's word (the two lines of a step's tile
//: its coefficients go to), 384: the window's 256 and, for every place of a step that is there and holds none, a PAD
//: -- a thread's third word, whose two lines it writes zero (a word past the pads names a place of a step that is not
//: there); the
//: entries' sparse bytes; where each of the 32 steps' A tile starts among the value lines, a byte a step; which steps are
//: there, a word -- then the entries' dense lines, which threads load, and the value lines, a brick a run of 256,
//: which land
constexpr int kPadded = 3 * kGroup, kOneSparse = 96, kOneTiles = 120, kOneLive = 122, kOneWide = 123, kOneToken = 124, kOneDense = 160, kOneValues = kOneDense + 2 * kGroup,
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
//: THE READERS' PACKETS (`kListed`), a block's in global memory, in words: a word a window -- its two halves' passes, a
//: byte each, one or more -- then the block's PASS RECORDS in the order they run. A record: an entry a (slot, token) --
//: a slot the 16 readers of one read digit, 8 slots a pass; the reader's index and, 16 bits up, its byte on its digit
//: (zero: a PAD, a place no reader holds) -- then the tokens' output lines as the drain warps take them (group, warp,
//: the warp's 16), two bytes each; then the slots' read digits, a byte each. A half's passes name digits of that half
constexpr int kPassLines = 128, kPassDigits = 192, kPassWords = 196, kReadPasses = 4;
constexpr int kReadHead = kCallWindows, kReadBlock = kReadHead + (kCallWindows * kReadPasses + 2) * kPassWords;
static_assert(kPassWords % 4 == 0 && kReadHead % 4 == 0 && kPassLines % 4 == 0);
//: a list's entries in a window, 8 to 24 and a window's 256: lists `2 p` and `2 p + 1` are 16 more and less the same
//: (layouts 3 and 4: every list 16 entries, a token on ALL FOUR digits of its group -- a group's 16 tokens)
constexpr int packed_count(int layout, uint32_t column, uint32_t window, int list) {
  if (layout >= 3) return kEntries;
  const int by = static_cast<int>(((column * 131u + window * 31u + static_cast<uint32_t>(list >> 1)) * 2654435761u >> 20) % 9u);
  return list & 1 ? kEntries - by : kEntries + by;
}
//: THE ENTRY whose token entry `e` of a window holds: itself, or in the shared layout, for the first `kShared` entries
//: of an odd list, the last `kShared` of the list before it
constexpr int kShared = 4;
constexpr int source_entry(int layout, uint32_t column, uint32_t window, int e) {
  //: (layout 5: a token on all 16 digits of its box -- the box's 16 tokens)
  if (layout == 5) return e % 16;
  //: (layout 6: a token on two digits of its group -- a list's 16 entries are its group's two steps' tokens of its parity)
  if (layout == 6 || (layout == 8 && e / 64 == 1)) return 64 * (e / 64) + 16 * ((e % 16) / 8) + 2 * ((e % 16) % 8) + ((e / 16) & 1);
  //: (layout 8: its first group's lists are its four steps, a token on one digit)
  if (layout == 8 && e / 64 == 0) return e;
  if (layout >= 3) return 64 * (e / 64) + e % 16;
  int a = 0, before = 0;
  while (before + packed_count(layout, column, window % kCallWindows, a) <= e) before += packed_count(layout, column, window % kCallWindows, a++);
  return layout == 1 && (a & 1) && e - before < kShared ? before - kShared + (e - before) : e;
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
  kSplitSteps = 16,
  //: the fold's fill by phases: what a thread does -- its entries' coefficients stored, its pad written, the
  //: tile starts loaded -- and what it waits, at the fill's barrier: three marks more in its fill
  kSplitFill = 32,
  //: the folds' waits on their packets' landing, and the copies' issues (a copy's barrier: its issue is its release)
  kWaitPacket = 64
};
//: AN ISOLATION ROW's change to the kernel: the read side idle -- the folds publish their image and wait on nobody
//: `kWide` is not an isolation: the kernel whose fold steps groups of four digits (a width is a kernel's)
//: `kSplit`: the per-list fold at the wide kernel's registers (176 / 112 / 48), to part the split from the width
//: `kFull` (with `kWide`): a group is one step a window -- its 16 tokens on all four digits
//: `kColumn`: THE PER-COLUMN FORM -- the per-list fold and the group-of-four fold (`kFull`'s) in one kernel, a box's
//: width word choosing its fold before its window loop
//: `kCleared` (with `kFull`): the tiles' room cleared and its barrier before the entries -- issue work the four-digit fold does
//: not need, priced by this row against `kFull`'s, composed and alone; and what `kCounted` needs
//: `kCounted` (with `kFull | kCleared`): a group's steps a run-time count of the packet's, one or two: a choice a group
//: between two straight blocks
//: `kLooped` (with `kFull | kCleared`): a group's steps a loop on the packet's count, whatever it is; the loop's test a warp
//: vote, one answer a warp
//: `kTabled`: an entry's dense line is the fold's own load from the tokens' table by its token's index (two bytes an entry
//: in the packet's header), not a line of the packet; `kSpread` with it: the table's rows 32 KB apart, 2 GB
static_assert(kOneToken * 16 + 2 * 2 * kGroup <= kOneDense * 16);
enum Probe : int { kAlone = 1, kWide = 16, kSplit = 32, kFull = 64, kColumn = 128, kCleared = 256, kCounted = 512, kLooped = 1024,
                   kTabled = 2048, kSpread = 4096,
                   //: `kAhead` (with `kTabled`): a window's rows are loaded a window ahead, behind the window before's entries
                   //: (its kernel is 176 / 112 / 48: a fold holds the next window's two lines through its steps)
                   kAhead = 8192,
                   //: `kListed`: THE READOUT'S PASSES ARE ITS PACKET'S -- a slot's read digit, its readers, their bytes and
                   //: their output lines, and a half's count of passes (a voted loop), where the fixed readout computes each
                   kListed = 16384 };
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
      //: the packet's waits and issues are their detail row's alone: the every-stamp row keeps its stamps
      else if (group == kWaitPacket ? (DETAIL & group) != 0 : (TRACE & 1) != 0 || (DETAIL & group) != 0) Row::stamp(kind, site);
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
                       kMarkSteps8 = 0xc00au, kMarkSteps16 = 0xc00bu, kMarkSteps24 = 0xc00cu, kMarkTail = 0xc00du,
                       kMarkZero = 0xc00eu, kMarkStarts = 0xc00fu, kMarkFillBar = 0xc010u };

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
//: A FOLD WARPGROUP: its box's state; a window is the image, then the wait on its packet, its entries' coefficients
//: expanded, and 32 steps that read A from the packet's value lines where its header says
template <int TRACE, int DETAIL, int PROBE>
__device__ __forceinline__ void fold_role(uint8_t* smem, uint32_t base, long long* times,
                                          float* state, const uint32_t* values, const uint32_t* table, unsigned long long* trace,
                                          int windows) {
  const int tid = threadIdx.x, box = tid >> 7, warp = (tid >> 5) & 3, lane = tid & 31, q = lane >> 2, t = lane & 3;
  const uint32_t bars = base + kBarsAt;
  //: a list's first step writes its accumulator whole: nothing is cleared (a cleared one in a role's branch
  //: serializes the steps, ptxas C7520)
  float d[kLists][8];
  Tracer<TRACE, DETAIL> tr;
  //: window `w`'s first line in the views' table, and its table's one copy by the warpgroup's first thread
  const auto one_line = [&](int w) {
    return ((blockIdx.x * 2 + (threadIdx.x >> 7)) % kColumns * kCallWindows + static_cast<uint32_t>(w) % kCallWindows) * static_cast<uint32_t>(kOneLines);
  };
  const auto issue_one = [&](int w) {
    //: the window's packet -- its header and its value lines -- two copies of the warpgroup's first thread behind one barrier
    if ((threadIdx.x & (kGroup - 1)) == 0) {
      const int box_of = threadIdx.x >> 7;
      const uint32_t landed = base + kBarsAt + kPoolLanded + 8 * box_of;
      const uint4* const from = reinterpret_cast<const uint4*>(values) + one_line(w);
      tr.at(warpline::kIssue, landed, kWaitPacket);
      lab::mbar_expect(landed, kOneBytes + kValueBytes);
      lab::bulk_load(base + kOneAt + box_of * kOneBytes, from, kOneBytes, landed);
      lab::bulk_load(base + 2 * box_of * kPoolBytes, from + kOneValues, kValueBytes, landed);
    }
  };
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
  //: A THREAD'S TWO ENTRIES' DENSE LINES of window `w`: the packet's lines, or (`kTabled`) the fold's own loads from the tokens'
  //: table -- the entry's token's index, two bytes of the packet's header in global memory, then that token's row, scattered
  uint4 dense_line[2];
  const auto rows_of = [&](int w) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int e = (tid & (kGroup - 1)) + half * kGroup;
      if constexpr ((PROBE & kTabled) != 0) {
        const uint32_t token = reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint4*>(values) + lab::held(one_line(w)) + kOneToken)[e];
        dense_line[half] = reinterpret_cast<const uint4*>(table)[(PROBE & kSpread) != 0 ? token << 11 : token];
      } else {
        dense_line[half] = (reinterpret_cast<const uint4*>(values) + lab::held(one_line(w)) + e)[kOneDense];
      }
    }
  };
  //: (`kAhead`: the first window's, then each window's behind the entries of the window before)
  if constexpr ((PROBE & kAhead) != 0) rows_of(0);
#pragma unroll 1
  for (int w = 0; w < windows; ++w) {
    tr.stamp(warpline::kPhaseBegin, kMarkWindow);
    const int g = tid & (kGroup - 1);
    //: A THREAD'S TWO ENTRIES' DENSE LINES, loaded at the window's start: the image phase covers their latency (the
    //: window's line in the views' table is held as 32 bits: no address of it outlives the loads)
    if constexpr ((PROBE & kAhead) == 0) rows_of(w);
    //: THE IMAGE, a tile group at a time behind the readout: channel m's fold digits `8 h` to `8 h + 7` of read
    //: digit b truncated to one line of tile b, `(m / 8) 17 + m % 8 + 8 h`, once the readout has read that group
    //: of the window before: 4 stores a thread a group.
    //: The first window's image is the cleared shared memory: the state is zero
    //: a thread's base into its box's image, formed a window (held through the image alone: the fill's registers are
    //: the fold's most): every store is this plus a constant of (tile, row, half)
    uint8_t* const mine = smem + lab::held(static_cast<uint32_t>(
        kImageAt + box * kBoxBytes + 2 * t * kImageTile + 16 * (2 * warp * kBrickLines + q)));
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
    //: THE WINDOW: its packet landed; a thread's two entries' coefficients, each a dense line loaded and expanded by
    //: its sparse byte to the two lines of its step's tile its word names; its pad; then 32 steps
    await(bars + kPoolLanded + 8 * box, w & 1, tr, kWaitPacket);
    tr.stamp(warpline::kPhaseBegin, kMarkExpand);
    const uint32_t pool = 2 * box * kPoolBytes;
    uint8_t* const image = smem + pool;
    const uint8_t* const table = smem + kOneAt + box * kOneBytes;
    //: THE WINDOW'S GROUP WIDTH, the packet's: lists a digit each, or GROUPS OF FOUR DIGITS -- a group's tokens once,
    //: its step 16 tokens' value lines against a tile of 16 tokens by its four digits' 64 columns, into the four
    //: digits' accumulators end to end. One straight block a width
    if constexpr ((PROBE & kWide) != 0) {
      if constexpr ((PROBE & kFull) != 0) {
        //: ONE STEP A GROUP: a thread's two entries -- a (token, digit) each -- expanded, four steps of 64 columns one
        //: commit group. Every line of a group's tile is an entry's (16 tokens by four digits): nothing is cleared
        if constexpr ((PROBE & kCleared) != 0) {
          //: THE TILES' ROOM CLEARED and the warpgroup's barrier behind the clear (a warp's 32 lanes store 32 consecutive
          //: 8-byte words an instruction): what a fold needs whose tiles hold lines no entry writes; where every line is
          //: an entry's it is issue work added, the probe's
          {
            const uint32_t x = lab::held(static_cast<uint32_t>(g));
            unsigned long long* const zero = reinterpret_cast<unsigned long long*>(image + kTileAt * 16 + (x >> 5) * 4096u + (x & 31u) * 8u);
#pragma unroll
            for (int i = 0; i < 16; ++i) zero[32 * i] = 0ull;
          }
          lab::bulk_fence();
          lab::bar_sync(kPoolId + box, kGroup);
        }
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int e = g + half * kGroup;
          const uint32_t word = reinterpret_cast<const uint32_t*>(table)[e];
          const float weight = static_cast<float>(table[kOneSparse * 16 + e]);
          const uint32_t line[4] = {dense_line[half].x, dense_line[half].y, dense_line[half].z, dense_line[half].w};
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
        if constexpr ((PROBE & kAhead) != 0)
          if (w + 1 < windows) rows_of(w + 1);
        lab::bulk_fence();
        lab::bar_sync(kPoolId + box, kGroup);
        tr.stamp(warpline::kPhaseBegin, kMarkSteps);
        const uint64_t lines = lab::wg_desc(base + pool, 8, 256);
        if constexpr ((PROBE & kLooped) != 0) {
          //: A GROUP'S STEPS, A LOOP ON THE PACKET'S COUNT: the loop's test is a warp vote, one answer a warp; a step's tile
          //: and its A's start are the window's next in the order issued
          const unsigned long long starts = *reinterpret_cast<const unsigned long long*>(table + kOneTiles * 16);
          const uint32_t counts = *reinterpret_cast<const uint32_t*>(table + kOneLive * 16);
          uint32_t step = 0;
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const uint32_t n = (counts >> (8 * q)) & 255u;
#pragma unroll
            for (int a = 0; a < 4; ++a)
#pragma unroll
              for (int i = 0; i < 8; ++i) lab::wg_hold(d[4 * q + a][i]);
            lab::wg_fence();
#pragma unroll 1
            for (uint32_t s = 0; __all_sync(0xffffffffu, s < n) != 0; ++s, ++step) {
              const uint64_t a = lines + ((starts >> (8 * step)) & 255ull);
              const uint64_t b = lab::wg_desc(base + pool + (kTileAt + 128 * step) * 16, 8, 16);
              lab::wg_bf16_n64_ss_mn4(d[4 * q], d[4 * q + 1], d[4 * q + 2], d[4 * q + 3], a, b, w != 0 || s != 0);
            }
            lab::wg_commit();
          }
        } else if constexpr ((PROBE & kCounted) != 0) {
          //: A GROUP'S STEPS, A COUNT OF THE PACKET'S: a byte a group in the header
          const unsigned long long starts = *reinterpret_cast<const unsigned long long*>(table + kOneTiles * 16);
          const uint32_t counts = *reinterpret_cast<const uint32_t*>(table + kOneLive * 16);
          //: A COUNT IS A CHOICE BETWEEN STRAIGHT BLOCKS, made outside them: one step or two, each block its own fence and
          //: commit group, the group's accumulators held across the choice; any other count traps
          //: -- see docs/lab/carry_token.md (the form)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            //: a group's steps take its own two places of the tiles' room and of the header's starts: every operand a constant
            const int step = 2 * q;
            const uint32_t n = (counts >> (8 * q)) & 255u;
#pragma unroll
            for (int a = 0; a < 4; ++a)
#pragma unroll
              for (int i = 0; i < 8; ++i) lab::wg_hold(d[4 * q + a][i]);
            const uint64_t a0 = lines + ((starts >> (8 * step)) & 255ull);
            const uint64_t b0 = lab::wg_desc(base + pool + (kTileAt + 128 * step) * 16, 8, 16);
            if (n == 2) {
              const uint64_t a1 = lines + ((starts >> (8 * (step + 1))) & 255ull);
              const uint64_t b1 = lab::wg_desc(base + pool + (kTileAt + 128 * (step + 1)) * 16, 8, 16);
              lab::wg_fence();
              lab::wg_bf16_n64_ss_mn4(d[4 * q], d[4 * q + 1], d[4 * q + 2], d[4 * q + 3], a0, b0, w != 0);
              lab::wg_bf16_n64_ss_mn4(d[4 * q], d[4 * q + 1], d[4 * q + 2], d[4 * q + 3], a1, b1, true);
              lab::wg_commit();
            } else if (n == 1) {
              lab::wg_fence();
              lab::wg_bf16_n64_ss_mn4(d[4 * q], d[4 * q + 1], d[4 * q + 2], d[4 * q + 3], a0, b0, w != 0);
              lab::wg_commit();
            } else {
              __trap();
            }
          }
        } else {
          const uint32_t starts = *reinterpret_cast<const uint32_t*>(table + kOneTiles * 16);
          lab::wg_fence();
#pragma unroll
          for (int q = 0; q < 4; ++q)
            lab::wg_bf16_n64_ss_mn4(d[4 * q], d[4 * q + 1], d[4 * q + 2], d[4 * q + 3], lines + ((starts >> (8 * q)) & 255u),
                                    lab::wg_desc(base + pool + (kTileAt + 128 * q) * 16, 8, 16), w != 0);
        }
        if constexpr ((PROBE & (kCounted | kLooped)) == 0) lab::wg_commit();
        lab::wg_wait<0>();
        if constexpr ((PROBE & (kCounted | kLooped)) != 0) {
#pragma unroll
          for (int a = 0; a < kLists; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) lab::wg_hold(d[a][i]);
        }
        lab::bar_sync(kPoolId + box, kGroup);
        if (w + 1 < windows) issue_one(w + 1);
        continue;
      }
#pragma unroll
      for (int pass = 0; pass < 2; ++pass) {
        //: A PASS is half the window's steps, two a group: its 8 tiles cleared -- 8 lines a thread -- then, behind
        //: the warpgroup's barrier, a thread's entry of the pass expanded to its two lines of its step's tile
        //: (a warp's 32 lanes store 32 consecutive 8-byte words an instruction: every bank twice, no conflict;
        //: a warp's quarter of the tiles is 16 such instructions)
        {
          const uint32_t x = lab::held(static_cast<uint32_t>(g));
          unsigned long long* const zero = reinterpret_cast<unsigned long long*>(image + kTileAt * 16 + (x >> 5) * 4096u + (x & 31u) * 8u);
#pragma unroll
          for (int i = 0; i < 16; ++i) zero[32 * i] = 0ull;
        }
        lab::bulk_fence();
        lab::bar_sync(kPoolId + box, kGroup);
        const int e = g + pass * kGroup;
        const uint32_t word = reinterpret_cast<const uint32_t*>(table)[e];
        const float weight = static_cast<float>(table[kOneSparse * 16 + e]);
        const uint32_t line[4] = {dense_line[pass].x, dense_line[pass].y, dense_line[pass].z, dense_line[pass].w};
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
        lab::bulk_fence();
        lab::bar_sync(kPoolId + box, kGroup);
        if (pass == 0) tr.stamp(warpline::kPhaseBegin, kMarkSteps);
        //: the pass's 8 steps' first positions, a byte a step in the landed header, loaded behind the fill
        const uint64_t lines = lab::wg_desc(base + pool, 8, 256);
        const unsigned long long starts = reinterpret_cast<const unsigned long long*>(table + kOneTiles * 16)[pass];
        lab::wg_fence();
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            const int step = 2 * q + r;
            const uint64_t a = lines + ((starts >> (8 * step)) & 255ull);
            const uint64_t b = lab::wg_desc(base + pool + (kTileAt + 128 * step) * 16, 8, 16);
            lab::wg_bf16_n64_ss_mn4(d[4 * q], d[4 * q + 1], d[4 * q + 2], d[4 * q + 3], a, b, w != 0 || pass != 0 || r != 0);
          }
        lab::wg_commit();
        lab::wg_wait<0>();
        //: THE PASS'S TILES ARE FREE -- and after the last pass the packet -- behind every warp's wait on the pass's steps:
        //: the warpgroup's barrier is that edge
        lab::bar_sync(kPoolId + box, kGroup);
      }
      if (w + 1 < windows) issue_one(w + 1);
      continue;
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int e = g + half * kGroup;
      const uint32_t word = reinterpret_cast<const uint32_t*>(table)[e];
      const float weight = static_cast<float>(table[kOneSparse * 16 + e]);
      const uint32_t line[4] = {dense_line[half].x, dense_line[half].y, dense_line[half].z, dense_line[half].w};
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
    if constexpr ((PROBE & kAhead) != 0)
      if (w + 1 < windows) rows_of(w + 1);
    if constexpr ((DETAIL & kSplitFill) != 0) tr.stamp(warpline::kPhaseBegin, kMarkZero);
    //: A PAD's two lines are zero: a thread's third word, no line loaded and nothing multiplied
    {
      const uint32_t word = reinterpret_cast<const uint32_t*>(table)[g + 2 * kGroup];
      //: by 8-byte stores of zero: a 16-byte store's four zero registers do not fit beside the traced rows' state
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        unsigned long long* const line = reinterpret_cast<unsigned long long*>(image + ((word >> (16 * half)) & 0xffffu) * 16);
        line[0] = 0ull;
        line[1] = 0ull;
      }
    }
    const uint32_t live = *reinterpret_cast<const uint32_t*>(table + kOneLive * 16);
    //: THE STEPS' TILES, a byte a step in the landed header: the first of the 16 consecutive positions whose value
    //: lines are the step's A; the header's two lines of them loaded here and held through the steps
    if constexpr ((DETAIL & kSplitFill) != 0) tr.stamp(warpline::kPhaseBegin, kMarkStarts);
    const uint4* const starts = reinterpret_cast<const uint4*>(table) + kOneTiles;
    const uint4 lo = starts[0], hi = starts[1];
    const uint32_t tile_of[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    if constexpr ((DETAIL & kSplitFill) != 0) tr.stamp(warpline::kPhaseBegin, kMarkFillBar);
    lab::bulk_fence();
    lab::bar_sync(kPoolId + box, kGroup);
    tr.stamp(warpline::kPhaseBegin, kMarkSteps);
    //: THE STEPS, one commit group: A is 16 positions' value lines as they landed -- a brick a run of 256, so 16
    //: consecutive positions are an M|N-major tile of bricks 256 lines apart -- its address the packet's first line
    //: and the step's start; B is the step's tile. A step that is not there is issued on the box's zeros; nothing is loaded a step
    const uint64_t lines = lab::wg_desc(base + pool, 8, 256), tiles = lab::wg_desc(base + pool + kTileAt * 16, 8, kTilePitch);
    const uint64_t zeros = lab::wg_desc(base + kZeroAt + box * 512, 8, 16);
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
        const uint64_t a = lines + ((tile_of[step >> 2] >> (8 * (step & 3))) & 255u);
        const uint64_t b = ((live >> step) & 1u) != 0 ? tiles + static_cast<uint64_t>(16 * step) : zeros;
        lab::wg_bf16_n16_ss_mn(d[j], a, b, w != 0 || r != 0);
      }
    lab::wg_commit();
    if constexpr ((DETAIL & kSplitSteps) != 0) tr.stamp(warpline::kPhaseBegin, kMarkTail);
    lab::wg_wait<0>();
    //: THE NEXT WINDOW'S PACKET, behind every warp's wait on the steps -- the warpgroup's barrier is that edge: every
    //: step has read this packet's lines and tiles
    lab::bar_sync(kPoolId + box, kGroup);
    if (w + 1 < windows) issue_one(w + 1);
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
  const int tid = threadIdx.x - kReadAt, warp = tid >> 5, lane = tid & 31;
  const uint32_t bars = base + kBarsAt;
  //: the lane's (token, box) of a group: token `lane % 16` of the warp's digit on box `lane / 16`
  const int n = lane & 15, box = lane >> 4;
  //: a pass's bytes in set `s`, the pass's parity in its window: a set is loaded two passes ahead
  uint4 row[2][kG];
  uint32_t reader_at[2][kG];
  const auto load = [&](int c, int s) {
#pragma unroll
    for (int g = 0; g < kG; ++g) {
      const int k = (c + g) % kChunks;
      reader_at[s][g] = reader_of(blockIdx.x, ((c + g) / kChunks) * kWindow + 16 * (8 * (k >> 1) + (k & 1) + 2 * warp) + n);
      row[s][g] = reinterpret_cast<const uint4*>(dense)[reader_row(reader_at[s][g], box)];
    }
  };
  load(0, 0);
  load(kG, 1);
  float d[kG * kChunkDigits][8];
  //: the lane's two lines of its digit's tile on its box (a group's tiles 8 on), and its sums' rows of a panel
  uint8_t* const lines = smem + kCoefAt + (2 * warp + box) * kCoefTile + ((n >> 3) * 17 + (n & 7)) * 16;
  //: the lane's row of a panel's stored matrices: matrix `m = lane / 8` is channels `16 warp + 8 (m % 2)` on, tokens
  //: `8 (m / 2)` on, and the lane names its token `lane % 8`'s 16 bytes of them
  const uint32_t mine = lab::held(base + static_cast<uint32_t>(kPanelAt + (8 * (lane >> 4) + (lane & 7)) * kRowBytes
                                                               + 2 * (16 * warp + 8 * ((lane >> 3) & 1))));
  lab::WarpMark::start(times);
#pragma unroll 1
  for (int w = 0; w < windows; ++w) {
#pragma unroll
    for (int c4 = 0; c4 < kChunks; c4 += kG) {
      const int c = kChunks * w + c4, s = c4 / kG;
#pragma unroll
      for (int g = 0; g < kG; ++g) await(bars + kTilesReady + 8 * (c4 + g), w & 1, tr, kWaitTilesReady);
      //: THE COEFFICIENTS, to shared memory: the reader's 16 bytes on the box by its byte on its digit
#pragma unroll
      for (int g = 0; g < kG; ++g) {
        const float weight = static_cast<float>(sparse_of(reader_at[s][g]));
        const uint32_t word[4] = {row[s][g].x, row[s][g].y, row[s][g].z, row[s][g].w};
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
      //: the bytes of the pass after next, issued under the steps into the set this pass's coefficients freed
      load(c + 2 * kG, s);
      steps_done(tr, kStepsRead);
#pragma unroll
      for (int g = 0; g < kG; ++g) arrive(bars + kTilesRead + 8 * (c4 + g), tr, kWaitTilesRead);
      //: THE PANEL a group: digit `j`'s sums as four 8 x 8 matrices (channels `16 warp + 8 r` on, tokens `8 i` on),
      //: matrix `r + 2 i`, stored transposed: a token's 8 channels a 16-byte piece of its line
#pragma unroll
      for (int g = 0; g < kG; ++g) {
        const int cg = c + g;
        if (cg >= 2) await(bars + kPanelFreed + 8 * (cg & 1), ((cg >> 1) - 1) & 1, tr, kWaitPanelFreed);
        tr.stamp(warpline::kPhaseBegin, kMarkPanel);
        const uint32_t panel = mine + (cg & 1) * kPanelBytes;
#pragma unroll
        for (int j = 0; j < kChunkDigits; ++j) {
          const float* const sums = d[g * kChunkDigits + j];
          const uint32_t four[4] = {lab::pack_half2(sums[0], sums[1]), lab::pack_half2(sums[2], sums[3]),
                                    lab::pack_half2(sums[4], sums[5]), lab::pack_half2(sums[6], sums[7])};
          lab::stsm_t(panel + 16 * j * kRowBytes, four);
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
    //: the chunk's digit `j` is read digit `8 h + c + 2 j`: its 16 tokens' rows start at 16 of that. THE LANE'S WORD
    //: OF THE WARP'S FIRST ROW is formed once a chunk; an add's address is that and a constant -- the warp's token
    //: `m` of the chunk is row `32 (m / 4) + 4 (m % 4)` on -- so an add is its one instruction
    const long long first = static_cast<long long>(blockIdx.x) * (kOutWindows * kWindow)
                            + ((c / kChunks) % kOutWindows) * kWindow
                            + 16 * (8 * ((c % kChunks) >> 1) + ((c % kChunks) & 1));
    uint32_t* const at = out + (first + mine) * (kChannels / 2) + lane;
    static_assert(kDrains == 4);
#pragma unroll
    for (int n = 0; n < kBatches; ++n) {
#pragma unroll
      for (int i = 0; i < kBatch; ++i) {
        const int m = n * kBatch + i;
        lab::red_add_f16x2(at + (32 * (m / 4) + 4 * (m % 4)) * (kChannels / 2), held[n & 1][i]);
      }
    }
    __syncwarp();
    arrive(bars + kPanelFreed + 8 * (c & 1), tr, kWaitPanelFreed);
  }
  tr.edge(kMarkExit);
  tr.close(base);
  lab::WarpMark::stop(times);
}

//: THE LISTED READOUT's warpgroup (`kListed`): a pass is the mirrored readout's -- 8 slots, a panel's group of 4 twice --
//: with its slots' digits, its readers and their bytes the packet's, and a half's passes a voted loop on its count
template <int TRACE, int DETAIL>
__device__ __forceinline__ void read_listed(uint8_t* smem, uint32_t base, long long* times, const uint32_t* dense,
                                            const uint32_t* readers, unsigned long long* trace, int windows) {
  Tracer<TRACE, DETAIL> tr;
  tr.bind(trace);
  tr.edge(kMarkEntry);
  constexpr int kG = 2;
  const int tid = threadIdx.x - kReadAt, warp = tid >> 5, lane = tid & 31;
  const uint32_t bars = base + kBarsAt;
  //: the lane's (token, box) of a slot: token `lane % 16` of the warp's slot of a group on box `lane / 16`
  const int n = lane & 15, box = lane >> 4;
  const uint32_t* const block = readers + static_cast<size_t>(blockIdx.x) * kReadBlock;
  //: A PASS'S SET: a group's entry of the lane and its reader's row, and the group's slots' digits; loaded two passes ahead
  struct Set {
    uint4 row[kG];
    uint32_t entry[kG], digits[kG];
  };
  const auto load = [&](uint32_t pass) {
    Set set;
    const uint32_t* const record = block + kReadHead + pass * kPassWords;
#pragma unroll
    for (int g = 0; g < kG; ++g) {
      set.entry[g] = record[16 * (4 * g + warp) + n];
      set.digits[g] = record[kPassDigits + g];
      set.row[g] = reinterpret_cast<const uint4*>(dense)[reader_row(set.entry[g] & 0xffffu, box)];
    }
    return set;
  };
  Set now = load(0), next = load(1);
  float d[kG * kChunkDigits][8];
  uint8_t* const lines = smem + kCoefAt + (2 * warp + box) * kCoefTile + ((n >> 3) * 17 + (n & 7)) * 16;
  const uint32_t mine = lab::held(base + static_cast<uint32_t>(kPanelAt + (8 * (lane >> 4) + (lane & 7)) * kRowBytes
                                                               + 2 * (16 * warp + 8 * ((lane >> 3) & 1))));
  lab::WarpMark::start(times);
  //: the block's passes run so far: a pass's record, and its panels' parity
  uint32_t pass = 0, ahead = block[0];
#pragma unroll 1
  for (int w = 0; w < windows; ++w) {
    //: a window's counts, loaded a window ahead
    const uint32_t counts = ahead;
    ahead = block[(w + 1) % kCallWindows];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int g = 0; g < kG; ++g) await(bars + kTilesReady + 8 * (2 * half + g), w & 1, tr, kWaitTilesReady);
      const uint32_t count = (counts >> (8 * half)) & 255u;
#pragma unroll 1
      for (uint32_t p = 0; __all_sync(0xffffffffu, p < count) != 0; ++p, ++pass) {
        //: THE COEFFICIENTS, to shared memory: the reader's 16 bytes on the box by its byte on its digit
#pragma unroll
        for (int g = 0; g < kG; ++g) {
          const float weight = static_cast<float>(now.entry[g] >> 16 & 255u);
          const uint32_t word[4] = {now.row[g].x, now.row[g].y, now.row[g].z, now.row[g].w};
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
        for (int g = 0; g < kG; ++g)
#pragma unroll
          for (int j = 0; j < kChunkDigits; ++j) {
            //: the slot's digit's tile of each box's image
            const uint32_t image = base + kImageAt + (now.digits[g] >> (8 * j) & 255u) * kImageTile;
#pragma unroll
            for (int bx = 0; bx < 2; ++bx)
              lab::wg_bf16_n16_ss(d[g * kChunkDigits + j], lab::wg_desc(image + bx * kBoxBytes, 8, kBrickLines),
                                  lab::wg_desc(tiles + (g * 8 + 2 * j + bx) * kCoefTile, 8, 17), bx != 0);
          }
        lab::wg_commit();
        //: the set of the pass after next, issued under the steps
        now = next;
        next = load(pass + 2);
        steps_done(tr, kStepsRead);
        //: the half's tile groups are read once its last pass's steps are
        if (__all_sync(0xffffffffu, p + 1 == count) != 0) {
#pragma unroll
          for (int g = 0; g < kG; ++g) arrive(bars + kTilesRead + 8 * (2 * half + g), tr, kWaitTilesRead);
        }
        //: THE PANEL a group, the mirrored readout's: slot `j`'s sums, a token's 8 channels a 16-byte piece of its line
#pragma unroll
        for (int g = 0; g < kG; ++g) {
          if (pass >= 1) await(bars + kPanelFreed + 8 * g, (pass - 1) & 1, tr, kWaitPanelFreed);
          tr.stamp(warpline::kPhaseBegin, kMarkPanel);
          const uint32_t panel = mine + g * kPanelBytes;
#pragma unroll
          for (int j = 0; j < kChunkDigits; ++j) {
            const float* const sums = d[g * kChunkDigits + j];
            const uint32_t four[4] = {lab::pack_half2(sums[0], sums[1]), lab::pack_half2(sums[2], sums[3]),
                                      lab::pack_half2(sums[4], sums[5]), lab::pack_half2(sums[6], sums[7])};
            lab::stsm_t(panel + 16 * j * kRowBytes, four);
          }
          __syncwarp();
          arrive(bars + kPanelReady + 8 * g, tr, kWaitPanelReady);
        }
      }
    }
  }
  lab::WarpMark::stop(times);
  tr.edge(kMarkExit);
  tr.close(base);
}

//: THE LISTED READOUT's drain warp: a group's 16 tokens of its residue, each added at the line the packet names
template <int TRACE, int DETAIL>
__device__ __forceinline__ void drain_listed(uint8_t* smem, uint32_t base, long long* times, uint32_t* out,
                                             const uint32_t* readers, unsigned long long* trace, int windows) {
  Tracer<TRACE, DETAIL> tr;
  tr.bind(trace);
  tr.edge(kMarkEntry);
  const int lane = threadIdx.x & 31, mine = (threadIdx.x - kDrainAt) >> 5;
  const uint32_t bars = base + kBarsAt;
  constexpr int kBatches = kTokens / kDrains / kBatch;
  static_assert(kDrains == 4 && kBatches * kBatch == 16);
  const uint32_t* const block = readers + static_cast<size_t>(blockIdx.x) * kReadBlock;
  //: the lane's word of the block's first line
  uint32_t* const at = out + static_cast<size_t>(blockIdx.x) * (kOutWindows * kWindow) * (kChannels / 2) + lane;
  uint32_t held[2][kBatch];
  const auto load = [&](int g, int batch, int h) {
#pragma unroll
    for (int i = 0; i < kBatch; ++i) {
      const uint8_t* const row = smem + kPanelAt + g * kPanelBytes + (mine + kDrains * (batch * kBatch + i)) * kRowBytes;
      held[h][i] = reinterpret_cast<const uint32_t*>(row)[lane];
    }
  };
  //: the warp's 16 tokens' lines of group `g` of pass `p`, the packet's: two loads
  const auto lines_of = [&](uint32_t p, int g, uint4& low, uint4& high) {
    const uint4* const listed = reinterpret_cast<const uint4*>(block + kReadHead + p * kPassWords + kPassLines) + 2 * (4 * g + mine);
    low = listed[0], high = listed[1];
  };
  //: A GROUP'S LINES ARE LOADED A GROUP AHEAD AND A WINDOW'S COUNTS A WINDOW AHEAD: the drain is the read side's binding
  //: chain, and a load it waited on would be the chain's
  uint4 low, high;
  lines_of(0, 0, low, high);
  uint32_t ahead = block[0];
  lab::WarpMark::start(times);
  uint32_t pass = 0;
#pragma unroll 1
  for (int w = 0; w < windows; ++w) {
    const uint32_t total = (ahead & 255u) + (ahead >> 8 & 255u);
    ahead = block[(w + 1) % kCallWindows];
#pragma unroll 1
    for (uint32_t p = 0; __all_sync(0xffffffffu, p < total) != 0; ++p, ++pass) {
#pragma unroll
      for (int g = 0; g < 2; ++g) {
        const uint32_t word[8] = {low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w};
        //: the next group's lines, issued before this group's wait
        lines_of(pass + g, 1 - g, low, high);
        await(bars + kPanelReady + 8 * g, pass & 1, tr, kWaitPanelReady);
        tr.stamp(warpline::kPhaseBegin, kMarkAdds);
        load(g, 0, 0);
        load(g, 1, 1);
#pragma unroll
        for (int batch = 0; batch < kBatches; ++batch) {
#pragma unroll
          for (int i = 0; i < kBatch; ++i) {
            const int m = batch * kBatch + i;
            lab::red_add_f16x2(at + (word[m / 2] >> (16 * (m & 1)) & 0xffffu) * (kChannels / 2), held[batch & 1][i]);
          }
        }
        __syncwarp();
        arrive(bars + kPanelFreed + 8 * g, tr, kWaitPanelFreed);
      }
    }
  }
  tr.edge(kMarkExit);
  tr.close(base);
  lab::WarpMark::stop(times);
}

//: THE KERNEL of a row: `TRACE` and `DETAIL` select its stamps and nothing else -- a detail row is its coarse row but
//: for those stamps; `PROBE` an isolation row's change
template <int TRACE, int DETAIL, int PROBE>
__global__ void __launch_bounds__(kThreads, 1)
    window(long long* times, const uint32_t* values, const uint32_t* dense, const uint32_t* table, const uint32_t* readers,
           float* state, uint32_t* out, unsigned long long* trace, int windows) {
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
    lab::regs_up<(PROBE & kAlone) != 0 ? kAloneFoldRegs : (PROBE & (kWide | kSplit | kColumn | kAhead)) != 0 ? 176 : kFoldRegs>();
    if constexpr ((PROBE & kColumn) != 0) {
      //: THE BOX'S WIDTH, its word of the device plan (here its first packet's header, in global memory), read before
      //: the window loop: a fold of each width, each its own straight body
      const uint32_t width =
          values[((blockIdx.x * 2 + (threadIdx.x >> 7)) % kColumns * kCallWindows * static_cast<uint32_t>(kOneLines) + kOneWide) * 4];
      if (width == 4) fold_role<TRACE, DETAIL, (PROBE & kAlone) | kWide | kFull>(smem, base, times, state, values, table, trace, windows);
      else if (width == 1) fold_role<TRACE, DETAIL, PROBE & kAlone>(smem, base, times, state, values, table, trace, windows);
      else __trap();
    } else {
      fold_role<TRACE, DETAIL, PROBE>(smem, base, times, state, values, table, trace, windows);
    }
  } else if (tid < kDrainAt) {
    lab::regs_down<(PROBE & kAlone) != 0 ? kAloneRestRegs : (PROBE & (kWide | kSplit | kColumn | kAhead)) != 0 ? 112 : kReadRegs>();
    if constexpr ((PROBE & kAlone) != 0) lab::WarpMark::start(times), lab::WarpMark::stop(times);
    else if constexpr ((PROBE & kListed) != 0) read_listed<TRACE, DETAIL>(smem, base, times, dense, readers, trace, windows);
    else read_role<TRACE, DETAIL>(smem, base, times, dense, trace, windows);
  } else {
    lab::regs_down<(PROBE & kAlone) != 0 ? kAloneRestRegs : (PROBE & (kWide | kSplit | kColumn | kAhead)) != 0 ? 48 : kDrainRegs>();
    if constexpr ((PROBE & kAlone) != 0) lab::WarpMark::start(times), lab::WarpMark::stop(times);
    else if constexpr ((PROBE & kListed) != 0) drain_listed<TRACE, DETAIL>(smem, base, times, out, readers, trace, windows);
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
  //: the packets: every entry its own value lines, and lists overlapping on shared tokens
  uint32_t *values[9], *readers[2], *dense, *spread, *out;
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
  //: THE PACKETS: each (column, window)'s -- the gather's part, played by the host. A list's entries are consecutive
  //: POSITIONS, so a step's tile is 16 positions from its list's start (held inside the block where a short list ends
  //: it); `layout` 1 has lists `2 p` and `2 p + 1` share `kShared` tokens, written once, the two lists' positions
  //: overlapping on them
  const size_t room = static_cast<size_t>(kOneLines) * 16;
  const auto build = [&](int layout) {
    std::vector<uint8_t> packets(static_cast<size_t>(kColumns) * kCallWindows * room, 0);
    for (int c = 0; c < kColumns; ++c)
      for (int w = 0; w < kCallWindows; ++w) {
        uint8_t* const packet = packets.data() + (static_cast<size_t>(c) * kCallWindows + w) * room;
        int n[kLists], first[kLists], start[kLists], at = 0, e = 0;
        for (int j = 0; j < kLists; ++j) {
          n[j] = packed_count(layout, c, w, j), first[j] = e, e += n[j];
          if (layout == 1 && (j & 1)) at -= kShared;
          start[j] = at, at += n[j];
        }
        //: THE BOX'S WIDTH WORD, the device plan's, held in every packet's header: the digits a step of its fold spans
        const uint32_t width = layout == 2 || layout == 3 || layout >= 5 ? 4 : 1;
        std::memcpy(packet + kOneWide * 16, &width, 4);
        //: GROUPS OF FOUR DIGITS (`layout` 2): a group's 64 entries are consecutive positions, its four steps 16 each;
        //: a pass is each group's two steps, 128 entries a thread each; an entry's word names its two lines of its
        //: step's tile, a tile 128 lines, its digit's two bricks of the tile's eight
        //: A TOKEN ON ALL FOUR DIGITS OF ITS GROUP, the group's packet (`layout` 3): a group's 16 tokens are one step of 64
        //: columns, a token's value lines once; an entry is a (token, digit), its word its digit's two bricks of the tile
        //: A TOKEN ON ALL 16 DIGITS OF ITS BOX (`layout` 5): the box's 16 tokens' value lines once, every group's step on
        //: that one run -- the box as one group is four steps of 64 columns on one A
        if (layout == 3 || layout == 5 || layout == 7) {
          //: a step a group: the header's count of each group's steps
          const uint32_t counts = 0x01010101u;
          std::memcpy(packet + kOneLive * 16, &counts, 4);
          for (int q = 0; q < 4; ++q) {
            //: a group's tile place: its own (`q`), or the first of its two (`layout` 7, the counted fold's)
            const int place_of = layout == 7 ? 2 * q : q;
            packet[kOneTiles * 16 + place_of] = static_cast<uint8_t>(layout == 5 ? 0 : 16 * q);
            for (int k = 0; k < 16; ++k) {
              const uint32_t t = listed_token(static_cast<uint32_t>(c), static_cast<uint32_t>(w * kWindow + (layout == 5 ? k : 64 * q + k)));
              for (int br = 0; br < 8; ++br)
                std::memcpy(packet + (kOneValues + br * 2 * kGroup + (layout == 5 ? k : 16 * q + k)) * 16, &host_values[static_cast<size_t>(t) * kRowWords + 4 * br], 16);
              for (int digit = 0; digit < 4; ++digit) {
                const int entry = 64 * q + 16 * digit + k;
                reinterpret_cast<uint32_t*>(packet)[entry] = static_cast<uint32_t>(kTileAt + 128 * place_of + 32 * digit + k)
                                                             | static_cast<uint32_t>(kTileAt + 128 * place_of + 32 * digit + 16 + k) << 16;
                packet[kOneSparse * 16 + entry] = static_cast<uint8_t>(sparse_of(static_cast<int>(t)));
                reinterpret_cast<uint16_t*>(packet + kOneToken * 16)[entry] = static_cast<uint16_t>(t);
                for (int dd = 0; dd < 16; ++dd) packet[(kOneDense + entry) * 16 + dd] = static_cast<uint8_t>(dense_of(static_cast<int>(t), dd));
              }
            }
          }
          continue;
        }
        //: A TOKEN ON TWO DIGITS OF ITS GROUP (`layout` 6): a group's 32 tokens are two steps of 64 columns, a token's value
        //: lines once; an entry is a (token, digit) -- the even places on the group's digits 0 and 2, the odd on 1 and 3 --
        //: and half of a tile's lines are no entry's. The header counts each group's steps
        if (layout == 6) {
          const uint32_t counts = 0x02020202u;
          std::memcpy(packet + kOneLive * 16, &counts, 4);
          for (int q = 0; q < 4; ++q)
            for (int s = 0; s < 2; ++s) {
              const int step = 2 * q + s;
              packet[kOneTiles * 16 + step] = static_cast<uint8_t>(16 * step);
              for (int k = 0; k < 16; ++k) {
                const uint32_t t = listed_token(static_cast<uint32_t>(c), static_cast<uint32_t>(w * kWindow + 64 * q + 16 * s + k));
                for (int br = 0; br < 8; ++br)
                  std::memcpy(packet + (kOneValues + br * 2 * kGroup + 16 * step + k) * 16, &host_values[static_cast<size_t>(t) * kRowWords + 4 * br], 16);
                for (int j = 0; j < 2; ++j) {
                  const int digit = 2 * j + (k & 1), entry = 64 * q + 32 * s + 16 * j + k, line = kTileAt + 128 * step + 32 * digit + k;
                  reinterpret_cast<uint32_t*>(packet)[entry] = static_cast<uint32_t>(line) | static_cast<uint32_t>(line + 16) << 16;
                  packet[kOneSparse * 16 + entry] = static_cast<uint8_t>(sparse_of(static_cast<int>(t)));
                  reinterpret_cast<uint16_t*>(packet + kOneToken * 16)[entry] = static_cast<uint16_t>(t);
                  for (int dd = 0; dd < 16; ++dd) packet[(kOneDense + entry) * 16 + dd] = static_cast<uint8_t>(dense_of(static_cast<int>(t), dd));
                }
              }
            }
          continue;
        }
        //: GROUPS TAKING FOUR, TWO, ONE AND ONE STEPS (`layout` 8): a token on one digit of the first group (a step a digit),
        //: on two of the second, on all four of the third and of the fourth; a step's tile and start are the window's next
        //: in that order, and the header counts each group's steps
        if (layout == 8) {
          const uint32_t counts = 0x01010204u;
          std::memcpy(packet + kOneLive * 16, &counts, 4);
          const auto lay = [&](int source, int pos, int tile, int digit, int entry, int place) {
            const uint32_t t = listed_token(static_cast<uint32_t>(c), static_cast<uint32_t>(w * kWindow + source));
            const int line = kTileAt + 128 * tile + 32 * digit + place;
            for (int br = 0; br < 8; ++br)
              std::memcpy(packet + (kOneValues + br * 2 * kGroup + pos) * 16, &host_values[static_cast<size_t>(t) * kRowWords + 4 * br], 16);
            reinterpret_cast<uint32_t*>(packet)[entry] = static_cast<uint32_t>(line) | static_cast<uint32_t>(line + 16) << 16;
            packet[kOneSparse * 16 + entry] = static_cast<uint8_t>(sparse_of(static_cast<int>(t)));
            reinterpret_cast<uint16_t*>(packet + kOneToken * 16)[entry] = static_cast<uint16_t>(t);
            for (int dd = 0; dd < 16; ++dd) packet[(kOneDense + entry) * 16 + dd] = static_cast<uint8_t>(dense_of(static_cast<int>(t), dd));
          };
          for (int tile = 0; tile < 8; ++tile)
            packet[kOneTiles * 16 + tile] = static_cast<uint8_t>(16 * tile);
          for (int k = 0; k < 16; ++k) {
            for (int s = 0; s < 4; ++s) lay(16 * s + k, 16 * s + k, s, s, 16 * s + k, k);
            for (int s = 0; s < 2; ++s)
              for (int j = 0; j < 2; ++j) lay(64 + 16 * s + k, 64 + 16 * s + k, 4 + s, 2 * j + (k & 1), 64 + 32 * s + 16 * j + k, k);
            for (int q = 2; q < 4; ++q)
              for (int digit = 0; digit < 4; ++digit) lay(64 * q + k, 96 + 16 * (q - 2) + k, 4 + q, digit, 64 * q + 16 * digit + k, k);
          }
          continue;
        }
        if (layout == 2) {
          for (int j4 = 0; j4 < kLists; j4 += 4)
            if (first[j4] != 16 * j4) { std::fprintf(stderr, "lab: a group of four lists is not 64 entries\n"); std::exit(3); }
          for (int j = 0; j < kLists; ++j)
            for (int k = 0; k < n[j]; ++k) {
              const int pos = first[j] + k, q = pos / 64, round = (pos % 64) / 16, place = pos % 16, pass = round / 2;
              const int step = 2 * q + round % 2, entry = pass * kGroup + 16 * step + place, digit = j - 4 * q;
              const uint32_t t = listed_token(static_cast<uint32_t>(c), static_cast<uint32_t>(w * kWindow + pos));
              packet[kOneTiles * 16 + 8 * pass + step] = static_cast<uint8_t>(64 * q + 16 * round);
              reinterpret_cast<uint32_t*>(packet)[entry] = static_cast<uint32_t>(kTileAt + 128 * step + 32 * digit + place)
                                                           | static_cast<uint32_t>(kTileAt + 128 * step + 32 * digit + 16 + place) << 16;
              packet[kOneSparse * 16 + entry] = static_cast<uint8_t>(sparse_of(static_cast<int>(t)));
              reinterpret_cast<uint16_t*>(packet + kOneToken * 16)[entry] = static_cast<uint16_t>(t);
              for (int dd = 0; dd < 16; ++dd) packet[(kOneDense + entry) * 16 + dd] = static_cast<uint8_t>(dense_of(static_cast<int>(t), dd));
              for (int br = 0; br < 8; ++br)
                std::memcpy(packet + (kOneValues + br * 2 * kGroup + pos) * 16, &host_values[static_cast<size_t>(t) * kRowWords + 4 * br], 16);
            }
          continue;
        }
        uint32_t live = 0;
        int pads = kWindow;
        std::vector<int> absent;
        for (int j = 0; j < kLists; ++j)
          for (int r = 0; r < kBlockRounds; ++r) {
            const int step = 16 * r + j;
            if (16 * r >= n[j]) {
              absent.push_back(step);
              continue;
            }
            const int tile = std::min(start[j] + 16 * r, 2 * kGroup - 16);
            packet[kOneTiles * 16 + step] = static_cast<uint8_t>(tile), live |= 1u << step;
            bool held_place[16] = {};
            for (int k = 16 * r; k < std::min(n[j], 16 * r + 16); ++k) {
              const int slot = first[j] + k, pos = start[j] + k, place = pos - tile;
              const uint32_t t = listed_token(static_cast<uint32_t>(c), static_cast<uint32_t>(w * kWindow + source_entry(layout, c, w, slot)));
              held_place[place] = true;
              reinterpret_cast<uint32_t*>(packet)[slot] =
                  static_cast<uint32_t>(kTileAt + 16 * step + place) | static_cast<uint32_t>(kTileAt + kTilePitch + 16 * step + place) << 16;
              packet[kOneSparse * 16 + slot] = static_cast<uint8_t>(sparse_of(static_cast<int>(t)));
              reinterpret_cast<uint16_t*>(packet + kOneToken * 16)[slot] = static_cast<uint16_t>(t);
              for (int dd = 0; dd < 16; ++dd) packet[(kOneDense + slot) * 16 + dd] = static_cast<uint8_t>(dense_of(static_cast<int>(t), dd));
              for (int br = 0; br < 8; ++br)
                std::memcpy(packet + (kOneValues + br * 2 * kGroup + pos) * 16, &host_values[static_cast<size_t>(t) * kRowWords + 4 * br], 16);
            }
            //: A PAD a place of this step no entry holds: its tile's two lines, a sparse byte of zero
            for (int place = 0; place < 16; ++place)
              if (!held_place[place]) {
                if (pads >= kPadded) { std::fprintf(stderr, "lab: a window of over %d entries and pads\n", kPadded); std::exit(3); }
                reinterpret_cast<uint32_t*>(packet)[pads++] =
                    static_cast<uint32_t>(kTileAt + 16 * step + place) | static_cast<uint32_t>(kTileAt + kTilePitch + 16 * step + place) << 16;
              }
          }
        //: a thread's third word past the pads names a place of a step that is not there, each its own (no two
        //: threads store one line): its tile is no step's. The absent steps' places are never fewer than these words
        for (size_t spare = 0; pads < kPadded; ++pads, ++spare) {
          if (spare >= 16 * absent.size()) { std::fprintf(stderr, "lab: more spare third words than absent places\n"); std::exit(3); }
          const int line = 16 * absent[spare / 16] + static_cast<int>(spare % 16);
          reinterpret_cast<uint32_t*>(packet)[pads] =
              static_cast<uint32_t>(kTileAt + line) | static_cast<uint32_t>(kTileAt + kTilePitch + line) << 16;
        }
        std::memcpy(packet + kOneLive * 16, &live, 4);
      }
    uint8_t* vdev = nullptr;
    LAB_CHECK(cudaMalloc(&vdev, packets.size()));
    LAB_CHECK(cudaMemcpy(vdev, packets.data(), packets.size(), cudaMemcpyHostToDevice));
    return reinterpret_cast<uint32_t*>(vdev);
  };
  held.values[0] = build(0);
  held.values[1] = build(1);
  held.values[2] = build(2);
  held.values[3] = build(3);
  held.values[4] = build(4);
  held.values[5] = build(5);
  held.values[6] = build(6);
  held.values[7] = build(7);
  held.values[8] = build(8);
  //: THE READERS' PACKETS, the gather's part again: `listed` 0 is the fixed readout's own passes (a half one pass, a slot
  //: its group's digit, a token's line its place); `listed` 1 the first half in two passes, each four of its digits (in
  //: falling order, every other slot) and four pads, the second half as before
  const auto build_readers = [&](int listed) {
    std::vector<uint32_t> packets(static_cast<size_t>(b.sms) * kReadBlock, 0);
    for (int blk = 0; blk < b.sms; ++blk) {
      uint32_t* const block = packets.data() + static_cast<size_t>(blk) * kReadBlock;
      int pass = 0;
      for (int w = 0; w < kCallWindows; ++w) {
        block[w] = listed == 1 ? 0x0102u : 0x0101u;
        for (int half = 0; half < 2; ++half)
          for (int p = 0; p < (listed == 1 && half == 0 ? 2 : 1); ++p, ++pass) {
            uint32_t* const record = block + kReadHead + pass * kPassWords;
            for (int q = 0; q < 8; ++q) {
              const int g = q / 4, j = q % 4;
              const bool split = listed == 1 && half == 0, pad = split && (q + p) % 2 != 0;
              const int digit = pad ? 0 : split ? 7 - (4 * p + q / 2) : 8 * half + g + 2 * j;
              reinterpret_cast<uint8_t*>(record + kPassDigits)[q] = static_cast<uint8_t>(digit);
              for (int n = 0; n < 16; ++n) {
                const uint32_t reader = reader_of(static_cast<uint32_t>(blk), static_cast<uint32_t>(w * kWindow + 16 * digit + n));
                record[16 * q + n] = pad ? 0u : reader | static_cast<uint32_t>(sparse_of(static_cast<int>(reader))) << 16;
                //: the drain warp `n % 4`'s token `4 j + n / 4` of the group
                reinterpret_cast<uint16_t*>(record + kPassLines)[(4 * g + n % 4) * 16 + 4 * j + n / 4] =
                    static_cast<uint16_t>(pad ? 0 : (w % kOutWindows) * kWindow + 16 * digit + n);
              }
            }
          }
      }
    }
    uint32_t* dev = nullptr;
    LAB_CHECK(cudaMalloc(&dev, packets.size() * 4));
    LAB_CHECK(cudaMemcpy(dev, packets.data(), packets.size() * 4, cudaMemcpyHostToDevice));
    return dev;
  };
  held.readers[0] = build_readers(0);
  held.readers[1] = build_readers(1);
  LAB_CHECK(cudaMalloc(&held.dense, host_dense.size() * 4));
  LAB_CHECK(cudaMemcpy(held.dense, host_dense.data(), host_dense.size() * 4, cudaMemcpyHostToDevice));
  //: THE SPREAD TABLE: the dense table's rows 32 KB apart -- 2 GB, a scattered row a cold line
  static_assert(kTableTokens == 65536);
  const size_t spread_bytes = static_cast<size_t>(kTableTokens) << 15;
  LAB_CHECK(cudaMalloc(&held.spread, spread_bytes));
  LAB_CHECK(cudaMemset(held.spread, 0, spread_bytes));
  for (int t = 0; t < kTableTokens; ++t)
    LAB_CHECK(cudaMemcpy(reinterpret_cast<uint8_t*>(held.spread) + (static_cast<size_t>(t) << 15), &host_dense[static_cast<size_t>(t) * 4], 16, cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMalloc(&held.state, sizeof(float) * b.sms * 2 * kGroup * (kLists * 8)));
  LAB_CHECK(cudaMalloc(&held.out, static_cast<size_t>(b.sms) * kOutWindows * kWindow * (kChannels / 2) * 4));
  return held;
}

//: A ROW: the kernel `window<TRACE, DETAIL>`, its checked call and its timed launches
template <int TRACE, int DETAIL = 0, int LAYOUT = 0, int PROBE = 0, int READ = 0>
void run(lab::Bench& b, const Buffers& held) {
  using namespace carry_token;
  constexpr int kTimed = 64;
  const std::string name = std::string("the token-stream carry: both sides") + (LAYOUT == 1 ? ", lists overlapping on shared tokens" : LAYOUT == 2 ? ", groups of four digits" : LAYOUT == 3 ? ", a token on all four digits of its group, groups of four" : LAYOUT == 4 ? ", a token on all four digits of its group, a list a digit" : LAYOUT == 5 ? ", a token on all 16 digits of its box, groups of four on one run of value lines" : LAYOUT == 6 ? ", a token on two digits of its group, groups of four" : LAYOUT == 7 ? ", a token on all four digits of its group, groups of four in a group's two tile places" : LAYOUT == 8 ? ", groups of four taking four, two, one and one steps" : "")
                           + (PROBE & kColumn ? ", the box's width word choosing its fold" : "") + (PROBE & kCleared ? ", its tiles cleared" : "")
                           + (PROBE & kCounted ? ", a group's steps counted at run time" : "") + (PROBE & kLooped ? ", a group's steps a voted loop on its count" : "")
                           + (PROBE & kSpread ? ", an entry's dense line from a 2 GB table of the tokens" : PROBE & kTabled ? ", an entry's dense line from the tokens' table" : "") + (PROBE & kAhead ? ", loaded a window ahead" : "")
                           + (PROBE & kListed ? (READ == 1 ? ", the readout's passes its packet's, a half in two padded passes" : ", the readout's passes its packet's") : "")
                           + (PROBE & kAlone ? ", the read side idle" : "") + (PROBE & kSplit ? ", registers 176 / 112 / 48" : "")
                           + (TRACE & 2 ? (TRACE & 1 ? ", every warp's timeline stamped" : ", every warp's phases stamped")
                                        : TRACE & 1 ? ", the launch's two edges stamped" : "")
                           + (DETAIL & kWaitTilesRead ? ", the waits on the tile groups read stamped" : "")
                           + (DETAIL & kWaitTilesReady ? ", the waits on the tile groups published stamped" : "")
                           + (DETAIL & kWaitPanelReady ? ", the waits on the panels filled stamped" : "")
                           + (DETAIL & kWaitPanelFreed ? ", the waits on the panels drained stamped" : "")
                           + (DETAIL & kSplitSteps ? ", the fold's steps by eighths stamped" : "")
                           + (DETAIL & kSplitFill ? ", the fold's fill by phases stamped" : "")
                           + (DETAIL & kWaitPacket ? ", the waits on the packets landed stamped" : "");
  constexpr int kWarps = kThreads >> 5;
  auto kernel = window<TRACE, DETAIL, PROBE>;
  uint32_t* const values = held.values[LAYOUT];
  uint32_t* const dense = held.dense;
  const uint32_t* const table = (PROBE & kSpread) != 0 ? held.spread : held.dense;
  const uint32_t* const readers = held.readers[READ];
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
  kernel<<<b.sms, kThreads, kSmemBytes>>>(b.times, values, dense, table, readers, state, out, rows, kOutWindows);
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
          const int t = static_cast<int>(listed_token((blk * 2 + box) % kColumns, w * kWindow + source_entry(LAYOUT, (blk * 2 + box) % kColumns, w, e)));
          //: the entry's list: the list whose count's run it falls in
          int a = 0;
          for (int before = packed_count(LAYOUT, (blk * 2 + box) % kColumns, w % kCallWindows, 0); before <= e; )
            before += packed_count(LAYOUT, (blk * 2 + box) % kColumns, w % kCallWindows, ++a);
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
  if constexpr ((PROBE & ~(kWide | kSplit | kFull | kColumn | kCleared | kCounted | kLooped | kTabled | kSpread | kAhead | kListed)) != 0) b.unchecked(name, "an isolation row: its kernel is not the carry's");
  else b.check(name, wrong_state == 0 && wrong_out == 0,
          std::to_string(wrong_state) + " of " + std::to_string(got_state.size()) + " state values and "
              + std::to_string(wrong_out) + " of " + std::to_string(2 * words) + " output values differ");
  const lab::Samples cycles = b.variant(
      [&] { kernel<<<b.sms, kThreads, kSmemBytes>>>(b.times, values, dense, table, readers, state, out, rows, kTimed); },
      [&] { return b.span(kWarps) / kTimed; });
  b.record(name, "cycles a window a CTA", cycles);
  //: THE CARD'S CLOCK AND THE WALL TIME, the plain row's (the carry_window rows' form): a long call's span in SM cycles
  //: over its wall time, the launch and its wait included, and its seconds scaled to 65,536 tokens, a window counted
  //: as 8 tokens a CTA as carry_window's is, the card's CTAs one a multiprocessor. The kernel is the row's own
  if constexpr (TRACE == 0 && DETAIL == 0 && LAYOUT == 0 && PROBE == 0) {
    constexpr int kLong = 4096;
    lab::Samples clock_rate, wall;
    for (int r = 0; r < 5; ++r) {
      LAB_CHECK(cudaMemset(b.times, 0, sizeof(long long) * b.sms * lab::Bench::kMaxWarps));
      LAB_CHECK(cudaDeviceSynchronize());
      const auto t0 = std::chrono::steady_clock::now();
      kernel<<<b.sms, kThreads, kSmemBytes>>>(b.times, values, dense, table, readers, state, out, rows, kLong);
      LAB_CHECK(cudaDeviceSynchronize());
      const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      clock_rate.v.push_back(b.span(kWarps) / seconds / 1e9);
      wall.v.push_back(seconds * 1000.0 * (65536.0 / (static_cast<double>(kLong) * 8 * b.sms)));
    }
    b.unchecked(name + ": the card's clock", "a rate read off this row's long call, which the row checks");
    b.record(name + ": the card's clock, the long call's span over its wall time", "GHz", clock_rate);
    b.unchecked(name + ": by wall time, scaled to 65,536 tokens",
                "the same launches as the card's clock: the row checks the kernel");
    b.record(name + ": the long call by wall time, scaled to 65,536 tokens", "milliseconds", wall);
  }
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
    for (int box = 0; box < 2; ++box) site(kBarsAt + kPoolLanded + 8 * box, "box " + std::to_string(box) + " packet landed");
    legend.insert(legend.end(), {{kMarkWindow, "window"}, {kMarkExpand, "expanding"}, {kMarkSteps, "issuing steps"}, {kMarkPanel, "storing the panel"},
                                 {kMarkAdds, "adding"}, {kStepsRead, "readout steps"}, {kMarkSteps8, "issuing steps 8 on"}, {kMarkSteps16, "issuing steps 16 on"},
                                 {kMarkSteps24, "issuing steps 24 on"}, {kMarkTail, "after the steps"}, {kMarkZero, "writing its pad"},
                                 {kMarkStarts, "loading tile starts"}, {kMarkFillBar, "at the fill's barrier"}, {kMarkEntry, "entry"},
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
  //: the rows' kernels: the plain one, its launch's edges, the coarse row, its five detail rows, every stamp, and
  //: the coarse row with the read side idle
  struct Row { int trace, detail, probe; };
  constexpr Row kRows[] = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {2, kWaitTilesRead, 0}, {2, kWaitTilesReady, 0}, {2, kWaitPanelReady, 0},
                           {2, kWaitPanelFreed, 0}, {2, kSplitSteps, 0}, {2, kSplitFill, 0}, {2, kWaitPacket, 0}, {3, 0, 0},
                           {2, 0, kAlone}, {0, 0, kWide}, {2, 0, kWide | kAlone}, {0, 0, kSplit}, {0, 0, kWide | kFull}, {2, 0, kWide | kFull | kAlone}, {0, 0, kColumn},
                           {0, 0, kWide | kFull | kCleared}, {2, 0, kWide | kFull | kCleared | kAlone}, {0, 0, kWide | kFull | kCleared | kCounted},
                           {2, 0, kWide | kFull | kCleared | kCounted | kAlone}, {0, 0, kWide | kFull | kCleared | kLooped},
                           {2, 0, kWide | kFull | kCleared | kLooped | kAlone},
                           {0, 0, kWide | kFull | kTabled}, {2, 0, kWide | kFull | kTabled | kAlone}, {0, 0, kWide | kFull | kTabled | kSpread},
                           {2, 0, kWide | kFull | kTabled | kSpread | kAlone}, {0, 0, kTabled}, {0, 0, kTabled | kSpread},
                           {0, 0, kWide | kFull | kTabled | kAhead}, {2, 0, kWide | kFull | kTabled | kAhead | kAlone},
                           {0, 0, kWide | kFull | kTabled | kSpread | kAhead}, {2, 0, kWide | kFull | kTabled | kSpread | kAhead | kAlone},
                           {0, 0, kTabled | kAhead}, {0, 0, kTabled | kSpread | kAhead},
                           {0, 0, kWide | kFull | kListed}, {2, 0, kWide | kFull}, {3, 0, kWide | kFull}, {2, 0, kWide | kFull | kListed},
                           {3, 0, kWide | kFull | kListed}};
  for (const Row& row : kRows) {
    const std::string k = lab::kernel("carry_token::window", {row.trace, row.detail, row.probe});
    //: the fold's 32 steps, one straight block, and the readout's 32: a window's two passes
    //: unrolled, each two groups of 4 digits, each digit's two steps
    //: (the per-column kernel holds a fold of each width: 32 steps and 4)
    const int fold_steps = row.probe & kColumn ? kLists * kBlockRounds + 4 : row.probe & kLooped ? 8 : row.probe & kCounted ? 12 : row.probe & kFull ? 4 : row.probe & kWide ? 16 : kLists * kBlockRounds;
    b.expect(k, "HGMMA", "==", fold_steps + (row.probe & kAlone ? 0 : kChunks * 2 * kChunkDigits));
    //: a fold step's cost to its warpgroup (`lab steps`): the port bytes and instructions a step, bounded by the
    //: most of any row
    b.expect(k, "STEP_PORT", "<=", row.probe & (kFull | kColumn) ? 24000 : row.probe & kWide ? 9000 : 4568);
    b.expect(k, "STEP_WORK", "<=", row.probe & (kFull | kColumn) ? 200 : row.probe & kWide ? 60 : 50);
    //: the fold's conversions are in its own trip, beside its steps: what the row measures
    b.expect(k, "STEP_XU", "<=", 40);
    //: THE VOTED LOOP's steps are rolled on purpose: its rows price a step in a loop of a run-time count beside the straight
    //: blocks (a loop a group; docs/lab/step_rolled.md prices one alone)
    if (row.probe & kLooped) b.expect(k, "STEP_ROLLED", "<=", 4);
    //: THE LISTED READOUT's steps are rolled on purpose: a half's passes are a loop on the packet's count, a pass's 16 steps its body
    if (row.probe & kListed) b.expect(k, "STEP_ROLLED", "<=", kChunks * 2 * kChunkDigits);
    //: the readout's panel is read by the drain warps' own loads, not by a step: its arrival needs no fence, and a
    //: chunk of 8 digits arrives at both panels, each behind its stores, and a window's two passes are unrolled
    b.expect(k, "STORE_UNFENCED", "<=", kChunks);
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
    //: a DECLARED DEBT (KERNEL_STANDARDS 29.16): the fold's and the readout's waits (`wg_commit(); wg_wait<0>()`) follow
    //: their steps directly -- the form the climb found costly, the warps idle through the steps and the tensor pipe
    //: through the rest of the window; the next window's fill belongs between the issue and the wait
    b.expect(k, "WAIT_AFTER_ISSUE", "<=", (row.probe & (kWide | kFull)) == kWide || (row.probe & kColumn) ? 4 : 3);
    //: the roles' loops hold their steps beside their barriers' waits: the waits' branches are the steps'
    b.expect(k, "BSSY", ">=", 0);
    //: THE FOLD'S PADS' STORES: a thread's pad lines are named by its packet's words, so a warp's 32 pad stores (6 `STS.128` of
    //: the fold) land wherever the data puts them and meet on banks by chance -- 43.5 excess wavefronts a CTA-window in the timed
    //: launch's dump (367,276 over its 8,448 CTA-windows, 132 CTAs of 64 windows), declared 44; the gather could place the lines
    //: conflict-free (owed)
    b.expect(k, "BANK_CONFLICT", "<=", 44);
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
  run<2, kSplitSteps>(b, held);
  run<2, kSplitFill>(b, held);
  run<2, kWaitPacket>(b, held);
  run<3>(b, held);
  //: an isolation row: the folds with the read side idle
  run<2, 0, 0, kAlone>(b, held);
  //: the plain and the coarse kernels on packets whose lists overlap on shared tokens
  run<0, 0, 1>(b, held);
  run<2, 0, 1>(b, held);
  //: the plain and the coarse kernels, and the folds alone, on packets of groups of four digits
  run<0, 0, 2, kWide>(b, held);
  run<0, 0, 0, kSplit>(b, held);
  //: a token on all four digits of its group: the group's kernel, and the per-list kernel on the same tokens
  run<0, 0, 3, kWide | kFull>(b, held);
  run<2, 0, 3, kWide | kFull | kAlone>(b, held);
  run<0, 0, 4>(b, held);
  run<2, 0, 4, kAlone>(b, held);
  run<2, 0, 2, kWide | kAlone>(b, held);
  //: a token on all 16 digits of its box: the group-of-four kernel, every group's step on the box's one run of value lines
  run<0, 0, 5, kWide | kFull>(b, held);
  //: THE PER-COLUMN FORM: one kernel on packets of each width -- a list a digit, and groups of four
  run<0, 0, 4, kColumn>(b, held);
  run<0, 0, 3, kColumn>(b, held);
  //: A FOLD'S ISSUE WORK: the four-digit grouped fold with its tiles' room cleared, plain and its folds alone
  run<0, 0, 3, kWide | kFull | kCleared>(b, held);
  run<2, 0, 3, kWide | kFull | kCleared | kAlone>(b, held);
  //: A GROUP'S STEPS COUNTED AT RUN TIME: one step a group (the four-digit tokens in a group's two tile places) and two (a
  //: token on two digits)
  run<0, 0, 7, kWide | kFull | kCleared | kCounted>(b, held);
  run<2, 0, 7, kWide | kFull | kCleared | kCounted | kAlone>(b, held);
  run<0, 0, 6, kWide | kFull | kCleared | kCounted>(b, held);
  run<2, 0, 6, kWide | kFull | kCleared | kCounted | kAlone>(b, held);
  //: A GROUP'S STEPS A VOTED LOOP ON ITS COUNT: one step a group, two, and four, two, one and one
  run<0, 0, 3, kWide | kFull | kCleared | kLooped>(b, held);
  run<2, 0, 3, kWide | kFull | kCleared | kLooped | kAlone>(b, held);
  run<0, 0, 6, kWide | kFull | kCleared | kLooped>(b, held);
  run<0, 0, 8, kWide | kFull | kCleared | kLooped>(b, held);
  run<2, 0, 8, kWide | kFull | kCleared | kLooped | kAlone>(b, held);
  //: AN ENTRY'S DENSE LINE FROM THE TOKENS' TABLE, the fold's own scattered load: the four-digit grouped fold and the per-list
  //: fold, on the lab's table and on the same rows 32 KB apart
  run<0, 0, 3, kWide | kFull | kTabled>(b, held);
  run<2, 0, 3, kWide | kFull | kTabled | kAlone>(b, held);
  run<0, 0, 3, kWide | kFull | kTabled | kSpread>(b, held);
  run<2, 0, 3, kWide | kFull | kTabled | kSpread | kAlone>(b, held);
  run<0, 0, 0, kTabled>(b, held);
  run<0, 0, 0, kTabled | kSpread>(b, held);
  //: THE SAME ROWS LOADED A WINDOW AHEAD
  run<0, 0, 3, kWide | kFull | kTabled | kAhead>(b, held);
  run<2, 0, 3, kWide | kFull | kTabled | kAhead | kAlone>(b, held);
  run<0, 0, 3, kWide | kFull | kTabled | kSpread | kAhead>(b, held);
  run<2, 0, 3, kWide | kFull | kTabled | kSpread | kAhead | kAlone>(b, held);
  run<0, 0, 0, kTabled | kAhead>(b, held);
  run<0, 0, 0, kTabled | kSpread | kAhead>(b, held);
  //: THE READOUT'S PASSES ITS PACKET'S: the four-digit grouped fold's window with the listed readout, on the fixed readout's own
  //: readers and on a half in two padded passes
  run<0, 0, 3, kWide | kFull | kListed>(b, held);
  run<0, 0, 3, kWide | kFull | kListed, 1>(b, held);
  //: THE TWO READOUTS' TIMELINES on the same packets: the fixed readout's and the listed one's, coarse and every stamp
  run<2, 0, 3, kWide | kFull>(b, held);
  run<3, 0, 3, kWide | kFull>(b, held);
  run<2, 0, 3, kWide | kFull | kListed>(b, held);
  run<3, 0, 3, kWide | kFull | kListed>(b, held);
  return 0;
}
