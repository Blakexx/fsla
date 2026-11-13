// measure/lab/experiments/carry_window.cu -- QUESTION: the carry's roles were each measured apart (`pool_fill`,
// `state_image`, `readout_role`); what does a window cost a CTA when they run TOGETHER, under one register file,
// one tensor pipe and one shared memory, and which role waits on which? METHOD: a CTA an SM of four warpgroups,
// 512 threads, registers traded by role. TWO FOLD warpgroups, a 16 x 16 box each, its state 16 accumulators in
// registers. TWO PRODUCER warps, one a box, fill a slot's 8 lists of 16 entries from scattered rows (`pool_fill`'s
// form, one batch in registers) and the fold runs the slot's 8 steps. A window is a box's 256 entries, and 256
// tokens read the pair: each fold warpgroup PUBLISHES its state as a bf16 image, 16 tiles, a group of 4 behind the
// readout, a line a channel's 8 fold digits: 4 stores of 16 bytes a thread a group. ONE READOUT warpgroup takes a chunk of 4 read digits of 16 tokens, a digit a warp's rows: a warp
// expands its tokens' coefficients from their rows' bytes IN ITS REGISTERS, and they are the A operand of the
// digit's step on each box's image tile, its B; one accumulator holds the chunk's 64 tokens by 64 channels, and
// the readout stores each token's row to a panel as the f16 pairs its line holds. TWO DRAIN warps add each
// token's row as one f16 line. The fold's, the readout's and the drains' waits at their barriers are clocked,
// and the readout's steps. Rows: the fold's side alone, the read side alone, both, and both with one access taken
// out: the producers' global loads, the drains' global adds, the image's stores; and with the readout DRAINING
// ITS OWN rows -- a readout warp loads its 16 tokens' lines back from the panel and adds them, no drain warp and
// no panel barrier -- and all four warps of the fourth warpgroup producers, two a box; and with THE READOUT LEAN --
// a chunk's digits taking turns in two sets of A registers, a digit's two steps a commit group, the readout at 80
// registers a thread and the fourth warpgroup at 112, its producers holding two batches; and with FIVE WARPGROUPS,
// 640 threads -- a fourth of four producers and a fifth of four drain warps at 40 registers a thread, the lean
// readout at 80; and with THE LONG WAITS ASLEEP -- a fold on its producer and on the readout's read, the readout
// on the image and a drain warp on the panel at named barriers, where a waiting warp issues nothing; and each side alone with one more
// access out -- the producers' loads, their coefficient arithmetic, both, and the readout's loads of its readers'
// bytes; and with those bytes ONE WIDE LOAD a readout warp a chunk -- a lane a row's 16 bytes, relaid through the
// warp's scratch in shared memory; and the read side with the fold's image stores, its fence behind them, or both
// taken out, and the readout and drains with the fold warps present and idle -- parked at a named barrier, or
// spinning at a polled one; a row that restores the fold's state dump to every call (the A/B that prices it in
// one session); the suspended and sleeping waits combined with the lean readout and the wide byte load; and rows
// whose tables are 16 replicas (128 MB of values) the cache cannot hold: an access's row is the replica its own
// position picks -- a reader row one replica a chunk and box -- as a large table's rows are each read once; and
// with THE DEN -- folded by two more steps a slot, their A the slot's lists' coefficient bricks and their B a
// column of ones; published as a ninth brick of the image's tiles; read as column 64 of the readout's steps, 72
// columns wide; and an fp32 a token through the panel and one more add a drain warp's lane, checked exactly. Then
// what the readout's steps
// wait behind: the readout and the drains with NO fold warp running (the image as it lies); every barrier's wait
// by the instruction that may suspend its thread, where the others poll. With both sides and no access out, the
// state and every output row are checked exactly, and a long call's span over its wall time reads the card's clock.
// TIMELINE: a fold warpgroup's trip is a window: the image a tile group at a time, four -- the wait on the
// readout's read of that group a window back, 4 stores a thread and their fence, the group's arrival -- then a
// slot twice: the wait on its producer, the fence, 8 steps, the commit and its wait, the slot freed. A producer's
// trip is a batch of 32 entries, a slot four of them: the batch's stores and multiplies (its words loaded a trip
// before), then the same registers' loads for the next batch; a slot on, the fence and the arrival. The
// readout's trip is a chunk: the wait on its tile group, 8 rounds a thread of coefficients (their bytes loaded a
// chunk before), every step's A formed in its own registers, the fence, 8 steps of 64 columns, the commit, the next chunk's 8 loads, the wait, the group
// released; then the wait on the panel two chunks back, 16 rounds to f16 pairs and stores a thread, the arrival.
// A drain warp's trip is a chunk, four batches of 8 tokens alternating two register halves: a batch's adds, each
// a word as the panel holds it, then those registers' loads for the batch two on (the chunk's first two as it
// opens). Live: a fold thread's 128 accumulators; a producer's 36 words; the readout's 32 accumulators, its 8
// steps' 32 words of A through their wait, and 8
// words; a drain's 16. THE DEN'S ROWS add: to the fold's slot 2 steps in its group and to its tile group one
// 4-byte store a holding thread; to the readout 4 accumulators (its steps 72 columns) and one fp32 store a row's
// holding thread; to a drain warp one load as the chunk opens and one add behind the chunk's lines'. THE ALIGNED
// CELL's fold stores a tile group as 16 words a thread (and the den's 2 values a holding thread) where it stores 4
// lines; nothing else moves. Live: 4 more
// fold accumulators, 4 more readout accumulators, a drain's one word.
// see docs/internals/tools/lab.md#carry-window
#include <chrono>

#include "lab.cuh"

namespace carry_window {

constexpr int kLists = 16, kEntries = 16, kWindow = kLists * kEntries, kSlotLists = 8, kSlot = kSlotLists * kEntries;
constexpr int kChannels = 64, kTableTokens = 65536, kRowWords = 32, kFill = 32, kReplicas = 16;
constexpr int kGroup = 128, kDrains = 2, kThreads = 4 * kGroup;
//: the roles' first threads: fold 0, fold 1, the readout, then one warpgroup of two producers and two drain warps
constexpr int kReadAt = 2 * kGroup, kProduceAt = 3 * kGroup, kDrainAt = kProduceAt + 64;
//: THE REGISTERS A THREAD. A raise takes what another warpgroup released from the count the kernel is launched
//: at -- 128 a thread at 512 threads, the SM's whole file: the folds' 32 more each are the 32 fewer of the readout
//: and of the producers' and drains' warpgroup
constexpr int kEntryRegs = 128, kFoldRegs = 160, kReadRegs = 96, kServeRegs = 96;
static_assert(kThreads * kEntryRegs == 65536
              && 2 * (kFoldRegs - kEntryRegs) == 2 * kEntryRegs - kReadRegs - kServeRegs);

//: A POOL TILE (the fold's operands a list), M|N-major: 8 bricks of values and 2 of coefficients, a brick 17 lines
constexpr int kBrick = 17 * 16, kValueTile = 8 * kBrick, kTile = 10 * kBrick, kPoolBytes = kSlotLists * kTile;
//: AN IMAGE TILE (the readout's operand a read digit), K-major: a line a channel's 8 fold digits -- one fold
//: thread's own values, a store -- a brick 8 channels' low digits then their high, bricks 17 lines, tiles 137; a NINTH brick is the den's, its first channel's two lines a (list half) each and the
//: rest nothing, tiles 155. A box's image has room for tiles 161 lines apart (`tile_of`)
constexpr int kBrickLines = 17, kImageBricks = 9, kImageRoom = 161 * 16, kBoxBytes = kLists * kImageRoom;
//: a panel row: a token's 64 f16 as its line will be added, rows 272 bytes apart
constexpr int kChunkDigits = 4, kTokens = 16 * kChunkDigits, kChunks = kLists / kChunkDigits, kRowBytes = 272,
              kPanelBytes = kTokens * kRowBytes, kBatch = 8;
//: a token's den in its panel row: an fp32 behind the line, at the word `row / 8 % 8` -- the 32 rows a drain warp's
//: lanes load are then on 32 banks, as the 8 a readout warp's lanes store are
constexpr int den_word(int row) { return 128 + 4 * ((row >> 3) & 7); }
constexpr int kImageAt = 4 * kPoolBytes, kPanelAt = kImageAt + 2 * kBoxBytes,
              kBarsAt = kPanelAt + 2 * kPanelBytes, kScratchAt = kBarsAt + 176;
//: a readout warp's scratch: its chunk's 32 rows of 16 bytes, row `16 box + token`
constexpr int kScratchBytes = 32 * 16, kOnesAt = kScratchAt + 4 * kScratchBytes;
//: THE DEN'S B: a brick a den step, 16 lines, every line of brick c holding 1 in slot c and nothing else -- an
//: M|N-major tile of 16 by 8 whose column c is ones
constexpr int kDenSteps = 4, kOnesBrick = 16 * 16, kSmemBytes = kOnesAt + kDenSteps * kOnesBrick;
//: the barriers, 8 bytes each: a pool slot filled and freed (box, slot); an image's tile group published and read
//: (a group is the 4 read digits `8 h + c + 2 j` of a chunk `2 h + c`: every fold thread holds a quarter of it); a
//: panel filled and drained
constexpr int kPoolReady = 0, kPoolFreed = 32, kTilesReady = 64, kTilesRead = 96, kPanelReady = 128, kPanelFreed = 144, kDone = 160;
//: the named barrier the parked row's fold warps sleep at until the readout's end: theirs and the readout's threads
constexpr int kParkId = 15, kParkCount = 3 * kGroup;
constexpr int kOutWindows = 4;
//: a row's roles, and what a row takes OUT of them to find what the two sides contend for: the producers' global
//: loads (the words made in registers), the drains' global adds (the lines folded into a sink), the image's stores
//: and what a row changes to find what the readout's steps wait behind: every wait SUSPENDED (`try_wait`) where the
//: others poll; the image STILL -- no fold warp runs, the readout reads it as it lies
//: and the row whose readout DRAINS ITS OWN rows: a readout warp loads its 16 tokens' lines back from the panel and
//: adds them itself, no drain warp and no panel barrier, and the fourth warpgroup is four producers, two a box
enum Roles : int {
  kFold = 1, kRead = 2, kNoLoads = 4, kNoAdds = 8, kNoImage = 16, kSuspend = 32, kStill = 64, kSelf = 128, kLean = 256, kFive = 512, kSleep = 1024,
  //: two more accesses a row takes out: the producers' coefficient arithmetic (the bytes stored as they are) and
  //: the readout's loads of its readers' bytes (the words made in registers)
  kNoMath = 2048, kNoBytes = 4096,
  //: THE WIDE row: a readout warp fetches its chunk's readers' bytes as ONE load -- a lane a whole row's 16 bytes,
  //: its 16 tokens on the two boxes -- stores them to the warp's own scratch and each thread reads its 8 words
  //: back, where the others' threads load each word from global memory, a lane a different row (8 instructions)
  kWide = 8192,
  //: one more access a row takes out: the fold's fence behind its image's stores (an isolation: the readout's steps
  //: may then read a tile before its stores are theirs to read)
  kNoFence = 16384,
  //: two rows of the readout and the drains with the fold warps PRESENT AND IDLE for the whole call -- parked at a
  //: named barrier, or spinning at a polled one -- where the still row's fold warps have left: what a present warp
  //: costs the others by being there
  kPark = 32768, kSpin = 65536,
  //: the row that RESTORES the fold's state dump to every call, as every record before this form had it: the A/B
  //: that prices the dump in one session
  kDump = 131072,
  //: the row whose tables the cache cannot hold: every ACCESS reads its row at a replica of the table picked by
  //: the access's own position, so the distinct rows touched grow with the accesses -- a replica's row holds its
  //: token's values, so the host's check stands -- as a large table's rows are each read once; 128 MB of values
  kBig = 262144,
  //: THE DEN: two more fold steps a slot, whose A is 8 lists' coefficients on 8 digits and whose B is a column of
  //: ones -- the box's den state, a value a (list, digit), in one accumulator set of 4 -- its image a ninth brick
  //: of each tile, the readout's steps 72 columns wide and a token's den an fp32 behind its panel row
  kDenFold = 524288,
  //: THE DEN'S OTHER PARTS, each a row's own so a chain of rows prices them one at a time: the image's ninth brick
  //: and its store (without the fold's part it stores a word of the num's state, the same instructions); the
  //: register split (readout 88, fourth warpgroup 104); the readout's 72-column steps; the panel's word and the
  //: drain's add (without the 72 columns, a word of the num's accumulator)
  kDenImage = 8388608, kDenSplit = 16777216, kDenRead = 33554432, kDenDrain = 67108864,
  kDen = kDenFold | kDenImage | kDenSplit | kDenRead | kDenDrain,
  //: the row that moves the image's tiles apart with nothing else changed: 154 lines where the tile is 137, and
  //: 161 where the den's is 155 -- what the tiles' pitch costs the fold's stores (an even pitch was measured
  //: slower)
  kPitch = 1048576,
  //: THE ALIGNED CELL: both sides list the same level, so a reader reads ONE list on the operand level's 16 digits
  //: -- an image tile is a LIST and its K the digits. A line (a channel's 8 digits of a list) is then four fold
  //: threads' values, a 4-byte word each: 16 word stores a thread a tile group where the other cell's are 4 lines;
  //: the den's line is 8 threads' values, 2 bytes each. The readout and the drains are the same code
  kAligned = 2097152,
  //: the rows with NO CLOCK INSIDE A ROLE: a wait is its wait and a phase is not lapped, so the window's cycles are
  //: the roles' own -- a role's clock reads compile differently from row to row (into registers in one kernel,
  //: through uniform registers in another), and these rows price a row's work without them
  kNoClock = 4194304
};
//: an image tile's pitch in bytes: the stores of a warp's lanes are two tiles apart
template <int ROLES>
constexpr int tile_of = 16 * (ROLES & kDenImage ? (ROLES & kPitch ? 161 : 155) : ROLES & kPitch ? 154 : 137);
//: a fold thread's state values: its 16 accumulator sets, and the den's 4
template <int ROLES>
constexpr int state_words = kLists * 8 + (ROLES & kDenFold ? 4 : 0);
//: a row in the replicated tables: the token's own, or the replica of it the access's position picks
template <int ROLES>
__device__ __forceinline__ uint32_t row_of(uint32_t t, uint32_t at) {
  if constexpr (ROLES & kBig) return t | ((at * 2654435761u >> 12) & ((kReplicas - 1) * kTableTokens));
  else return t;
}
//: THE FIVE-WARPGROUP row: 640 threads, a fourth warpgroup of four producers and a fifth of four drain warps. A
//: kernel of 640 threads enters with 96 registers a thread (docs/lab/reg_five.md), and the roles' counts sum to
//: that: the folds' 160, the lean readout's 80, and 40 a producer and a drain warp -- a producer holds one value
//: instruction's four entries at a time and a drain warp one batch of 4 tokens (the shapes 40 registers hold: a
//: producer of two instructions' entries or a drain warp of 8 tokens spills), and neither holds a clock or a count
constexpr int kFiveEntry = 96, kFiveProduceRegs = 40, kFiveDrainRegs = 40, kFiveReadRegs = 80;
constexpr int kFiveParts = 8, kFiveHold = 4;
static_assert(2 * kFoldRegs + kFiveReadRegs + kFiveProduceRegs + kFiveDrainRegs == 5 * kFiveEntry);
template <int ROLES>
constexpr int threads_of = ROLES & kFive ? 5 * kGroup : kThreads;
template <int ROLES>
constexpr int drains_of = ROLES & kFive ? 4 : kDrains;
template <int ROLES>
constexpr int drain_at = ROLES & kFive ? 4 * kGroup : kDrainAt;
//: the producers a box
template <int ROLES>
constexpr int producers_of = ROLES & (kSelf | kFive) ? 2 : 1;
//: the parts a producer's batch of values is loaded and stored in: a part in registers at a time
template <int ROLES>
constexpr int value_parts = ROLES & kFive ? kFiveParts : 1;
//: THE LEAN READOUT's row: a chunk's digits take turns in TWO sets of A registers, a digit's two steps a commit
//: group, a set written again once its group is done -- 16 words of A where every step's own are 32 -- and the 16
//: registers a thread it gives up are the fourth warpgroup's, whose producers then hold two batches (at 72 the
//: toolkit serializes the steps, C7512)
//: the den's readout holds 4 more accumulators, column 64's set: at 80 the toolkit serializes its steps, and its 8
//: more a thread are the fourth warpgroup's, as the wide and the big rows' are
template <int ROLES>
constexpr int read_regs = ROLES & kFive ? kFiveReadRegs : (ROLES & kLean) && (ROLES & (kDenSplit | kWide)) ? 88 : ROLES & kLean ? 80 : kReadRegs;
template <int ROLES>
constexpr int serve_regs = (ROLES & kLean) && (ROLES & (kWide | kDenSplit)) ? 104 : ROLES & kLean ? 112 : kServeRegs;
static_assert(read_regs<kLean | kWide> + serve_regs<kLean | kWide> == kReadRegs + kServeRegs
              && read_regs<kLean | kBig> + serve_regs<kLean | kBig> == kReadRegs + kServeRegs);
static_assert(read_regs<kLean | kDen> + serve_regs<kLean | kDen> == kReadRegs + kServeRegs);
static_assert(read_regs<kLean> + serve_regs<kLean> == kReadRegs + kServeRegs);
template <int ROLES>
constexpr int batches_held = ROLES & kLean ? 2 : 1;

//: the token of a (CTA, box, entry of the call), its value at a channel, its byte on its list's digit and on a
//: dense leaf; a read coefficient at (read digit, fold digit, token): small integers
constexpr uint32_t token_of(uint32_t source, uint32_t entry) {
  return ((source * 4099u + entry) * 2654435761u >> 12) & (kTableTokens - 1);
}
constexpr int value_of(int t, int c) { return (t + c) % 2 + 1; }
constexpr int sparse_of(int t) { return t % 2 + 1; }
constexpr int dense_of(int t, int b) { return (t + b) % 2 + 1; }
//: a reader: the token of a (CTA, entry of the call) -- read digit `b`'s token `n` of window `w` is entry
//: `256 w + 16 b + n` -- whose bytes on box `box`'s 16 leaves are the dense row of the token `box` on from it
constexpr uint32_t reader_of(uint32_t block, uint32_t entry) { return token_of(0x10000u + block, entry); }
constexpr uint32_t reader_row(uint32_t reader, int box) { return (reader + box) & (kTableTokens - 1); }

//: THE SLEEPING row's four long waits -- a fold on its producer and on the readout's read of a tile group, the
//: readout on a tile group's publish, a drain warp on the panel -- are at NAMED barriers, where a waiting warp issues nothing (the polled waits issue a test
//: and a branch every trip). A named barrier counts threads: its arrivers' and its waiters' together complete it.
//: Each pair alternates -- an arriver's next arrival follows its waiter's release, behind the polled barrier that
//: frees the slot or the panel, or the named one the readout's read of the tiles arrives at -- so no arrival is
//: counted twice
constexpr int kPoolId = 1, kTilesId = 5, kPanelId = 9, kReadId = 11;
constexpr int kPoolCount = kGroup + 32, kTilesCount = 3 * kGroup, kPanelCount = kGroup + 32 * kDrains;
//: a sleeping wait, its cycles added to the warp's count
__device__ __forceinline__ void slept(int id, int count, uint32_t& clocked) {
  const uint32_t t0 = static_cast<uint32_t>(clock64());
  lab::bar_sync(id, count);
  clocked += static_cast<uint32_t>(clock64()) - t0;
}

//: A WAIT at a barrier, polled or suspended by the row
template <int ROLES>
__device__ __forceinline__ void await(uint32_t bar, uint32_t parity) {
  if constexpr (ROLES & kSuspend) lab::mbar_wait_suspended(bar, parity);
  else lab::mbar_wait(bar, parity);
}
//: the same, its cycles added to the warp's count for that barrier: a role's time is its span less its waits
template <int ROLES>
__device__ __forceinline__ void waited(uint32_t bar, uint32_t parity, uint32_t& count) {
  if constexpr (ROLES & kNoClock) {
    await<ROLES>(bar, parity);
  } else {
    const uint32_t t0 = static_cast<uint32_t>(clock64());
    await<ROLES>(bar, parity);
    count += static_cast<uint32_t>(clock64()) - t0;
  }
}
//: a warp's four counts, stored at `kCounts (block, warp)`: its two waits, and the readout's steps and their A
constexpr int kCounts = 4;
__device__ __forceinline__ void tell(uint32_t* waits, uint32_t first, uint32_t second, uint32_t third = 0,
                                     uint32_t fourth = 0) {
  if ((threadIdx.x & 31) == 0) {
    uint32_t* const to = waits + kCounts * (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5));
    to[0] = first, to[1] = second, to[2] = third, to[3] = fourth;
  }
}
//: the cycles since `since`, and `since` moved to now: a phase's clock
__device__ __forceinline__ uint32_t lap(uint32_t& since) {
  const uint32_t now = static_cast<uint32_t>(clock64()), took = now - since;
  since = now;
  return took;
}

//: A FOLD WARPGROUP: its box's state; a window is the image, then two slots of 8 steps
template <int ROLES>
__device__ __forceinline__ void fold_role(uint8_t* smem, uint32_t base, long long* times, uint32_t* waits,
                                          float* state, int windows) {
  const int tid = threadIdx.x, box = tid >> 7, warp = (tid >> 5) & 3, lane = tid & 31, q = lane >> 2, t = lane & 3;
  const uint32_t bars = base + kBarsAt;
  //: a list's first step writes its accumulator whole: nothing is cleared (a cleared one in a role's branch
  //: serializes the steps, ptxas C7520)
  float d[kLists][8];
  //: the den's accumulator, 64 rows by 8: column c holds 8 lists' den on 8 digits, a row a (list, digit)
  float den[4];
  //: a thread's base into its box's image: every store is this plus a constant of (tile, row, half); the aligned
  //: cell's is its word of its row's line in the box's first tile
  uint8_t* const mine = smem + lab::held(static_cast<uint32_t>(
      ROLES & kAligned ? kImageAt + box * kBoxBytes + 16 * (2 * warp * kBrickLines + q) + 4 * t
                       : kImageAt + box * kBoxBytes + 2 * t * tile_of<ROLES> + 16 * (2 * warp * kBrickLines + q)));
  //: THE DEN'S STORE, a held base a thread for each of a tile group's two kinds and an immediate past it: EVERY
  //: lane stores, the lanes that hold the group's den values to the den's line and the others to a line of the
  //: ninth brick no step's kept column reads (channels 65 to 71) -- no branch around the store
  uint32_t den_at[2] = {0u, 0u};
  if constexpr (ROLES & kDenImage) {
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      if constexpr (ROLES & kAligned) {
        //: kind `p` is the group's list half: the holders are the threads `t == p`, lists `8 t + 2 warp` on
        const bool holds = t == p;
        den_at[p] = lab::held(static_cast<uint32_t>(
            kImageAt + box * kBoxBytes + (8 * (holds ? t : p) + 2 * warp) * tile_of<ROLES>
            + 16 * ((kImageBricks - 1) * kBrickLines + (holds ? 0 : t < 2 ? 1 : t)) + 2 * q));
      } else {
        //: kind `p` is the group's digit parity: the holders are the threads `t < 2` whose `q` has it
        const bool holds = t < 2 && (q & 1) == p;
        den_at[p] = lab::held(static_cast<uint32_t>(
            kImageAt + box * kBoxBytes + ((q & 1) == p ? q : q ^ 1) * tile_of<ROLES>
            + 16 * ((kImageBricks - 1) * kBrickLines + (holds ? 8 * t : t < 2 ? 1 + 8 * t : t)) + 4 * warp));
      }
    }
  }
  if constexpr (!(ROLES & kFold)) {
    //: no fold writes the state: one group of steps gives every accumulator a value to publish
    lab::wg_fence();
#pragma unroll
    for (int a = 0; a < kLists; ++a)
      lab::wg_bf16_n16_ss_mn(d[a], lab::wg_desc(base + kImageAt, 8, 17), lab::wg_desc(base + kImageAt, 8, 17), false);
    lab::wg_commit();
    lab::wg_wait<0>();
  }
  uint32_t on_read = 0, on_fill = 0;
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int w = 0; w < windows; ++w) {
    if constexpr (ROLES & kRead) {
      //: THE IMAGE, a tile group at a time behind the readout: channel m's fold digits `8 h` to `8 h + 7` of read
      //: digit b truncated to one line of tile b, `(m / 8) 17 + m % 8 + 8 h`, once the readout has read that group
      //: of the window before: 4 stores a thread a group.
      //: The first window's image is the cleared shared memory: the state is zero
#pragma unroll
      for (int k = 0; k < kChunks; ++k) {
        if constexpr (ROLES & kSleep) {
          if (w > 0) slept(kReadId + k, kTilesCount, on_read);
        } else {
          if (w > 0) waited<ROLES>(bars + kTilesRead + 8 * k, (w - 1) & 1, on_read);
        }
        if (w > 0 && !(ROLES & kNoImage)) {
          if constexpr (ROLES & kAligned) {
            //: the group's 4 tiles are lists `8 (k / 2) + k % 2 + 2 j`: a thread's word of each row's two lines
#pragma unroll
            for (int j = 0; j < kChunkDigits; ++j)
#pragma unroll
              for (int r = 0; r < 2; ++r)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                  const int a = 8 * (k >> 1) + (k & 1) + 2 * j;
                  *reinterpret_cast<uint32_t*>(mine + a * tile_of<ROLES> + 16 * (r * kBrickLines + 8 * h)) =
                      lab::bf16x2_trunc(d[a][r << 1 | h << 2], d[a][1 | r << 1 | h << 2]);
                }
            if constexpr (ROLES & kDenImage) {
              //: THE DEN'S LINES: list `8 s + l`'s den on digit `8 hc + r` is row `8 l + r` of column `2 s + hc`
              //: -- a holding thread's value of its list `8 t + 2 warp + k % 2`, 2 bytes of the list's tile's ninth
              //: brick
#pragma unroll
              for (int hc = 0; hc < 2; ++hc) {
                const float value = ROLES & kDenFold ? den[hc | (k & 1) << 1] : d[0][hc | (k & 1) << 1];
                *reinterpret_cast<uint16_t*>(smem + den_at[k >> 1] + (k & 1) * tile_of<ROLES> + 16 * 8 * hc) =
                    static_cast<uint16_t>(__float_as_uint(value) >> 16);
              }
            }
          } else {
  #pragma unroll
            for (int r = 0; r < 2; ++r)
  #pragma unroll
              for (int h = 0; h < 2; ++h) {
                const int i = (k & 1) | r << 1 | (k >> 1) << 2;
                uint32_t pair[4];
  #pragma unroll
                for (int j = 0; j < 4; ++j) pair[j] = lab::bf16x2_trunc(d[8 * h + 2 * j][i], d[8 * h + 2 * j + 1][i]);
                *reinterpret_cast<uint4*>(mine + (8 * (k >> 1) + (k & 1)) * tile_of<ROLES> + 16 * (r * kBrickLines + 8 * h)) =
                    make_uint4(pair[0], pair[1], pair[2], pair[3]);
              }
            if constexpr (ROLES & kDenImage) {
              //: THE DEN'S LINES of the group: column `2 s + k / 2` of the accumulator is lists `8 s` on of digits
              //: `8 (k / 2)` on, a thread's two rows lists `2 warp` and the next on digit `q` -- one word of line `s` of
              //: tile `8 (k / 2) + q`'s ninth brick; the tile's layout fixes the holders' places, so they share banks
              const uint32_t word = ROLES & kDenFold ? lab::bf16x2_trunc(den[k >> 1], den[(k >> 1) | 2])
                                                     : lab::bf16x2_trunc(d[0][k >> 1], d[0][(k >> 1) | 2]);
              *reinterpret_cast<uint32_t*>(smem + den_at[k & 1] + 8 * (k >> 1) * tile_of<ROLES>) = word;
            }
          }
        }
        if constexpr (!(ROLES & kNoFence)) lab::bulk_fence();
        if constexpr (ROLES & kSleep) {
          lab::bar_arrive(kTilesId + k, kTilesCount);
        } else {
          __syncwarp();
          if (lane == 0) lab::mbar_arrive(bars + kTilesReady + 8 * k);
        }
      }
    }
    if constexpr (ROLES & kFold) {
#pragma unroll
      for (int slot = 0; slot < 2; ++slot) {
        if constexpr (ROLES & kSleep) slept(kPoolId + 2 * box + slot, kPoolCount, on_fill);
        else waited<ROLES>(bars + kPoolReady + 8 * (2 * box + slot), w & 1, on_fill);
        const uint32_t tiles = lab::held(base + (2 * box + slot) * kPoolBytes);
        lab::wg_fence();
#pragma unroll
        //: a slot's descriptors are ONE derived from its pool's address and a constant of lines added a list: the
        //: start field is the address in lines, linear in it
        const uint64_t first = lab::wg_desc(tiles, 8, 17);
#pragma unroll
        for (int j = 0; j < kSlotLists; ++j)
          lab::wg_bf16_n16_ss_mn(d[slot * kSlotLists + j], first + static_cast<uint64_t>(j * (kTile / 16)),
                                 first + static_cast<uint64_t>(j * (kTile / 16) + kValueTile / 16), w != 0);
        if constexpr (ROLES & kDenFold) {
          //: THE DEN'S STEPS, in the slot's group, two: A is the slot's 8 lists' coefficient bricks of digits `8 h`
          //: to `8 h + 7` -- a brick a list, the bricks a pool tile apart -- and B the ones of column `2 slot + h`,
          //: so row `8 l + r` of that column is list `8 slot + l`'s den on digit `8 h + r`; the call's first writes
          //: the accumulator whole
          const uint64_t ones = lab::wg_desc(lab::held(base + kOnesAt), 8, 16);
          const uint64_t bricks = lab::wg_desc(tiles + kValueTile, 8, kTile / 16);
#pragma unroll
          for (int h = 0; h < 2; ++h)
            lab::wg_bf16_n8_ss_mn(den, bricks + static_cast<uint64_t>(h * (kBrick / 16)),
                                  ones + static_cast<uint64_t>((2 * slot + h) * (kOnesBrick / 16)),
                                  w != 0 || slot != 0 || h != 0);
        }
        lab::wg_commit();
        lab::wg_wait<0>();
        if ((tid & (kGroup - 1)) == 0) lab::mbar_arrive(bars + kPoolFreed + 8 * (2 * box + slot));
      }
    }
  }
#pragma unroll
  for (int a = 0; a < kLists; ++a)
#pragma unroll
    for (int i = 0; i < 8; ++i) lab::wg_hold(d[a][i]);
  if constexpr (ROLES & kDenFold) {
#pragma unroll
    for (int i = 0; i < 4; ++i) lab::wg_hold(den[i]);
  }
  clock.stop(times);
  tell(waits, on_read, on_fill);
  //: THE STATE OUT, for the checked call alone: 128 stores a thread, each instruction's lanes on 32 lines, issued
  //: while the readout and the drains are still in their last windows -- in a timed call they would be clocked as
  //: the read side's
  if (windows <= kOutWindows || (ROLES & kDump)) {
    float* const to = state + (static_cast<long long>(blockIdx.x) * 2 * kGroup + tid) * state_words<ROLES>;
#pragma unroll
    for (int a = 0; a < kLists; ++a)
#pragma unroll
      for (int i = 0; i < 8; ++i) to[8 * a + i] = d[a][i];
    if constexpr (ROLES & kDenFold) {
#pragma unroll
      for (int i = 0; i < 4; ++i) to[8 * kLists + i] = den[i];
    }
  }
}

//: A PRODUCER WARP, one or two a box: its share of a slot's four batches of 32 entries, one batch in registers at a
//: time or, in the lean row, two
template <int ROLES>
__device__ __forceinline__ void produce_role(uint8_t* smem, uint32_t base, long long* times, const uint32_t* values,
                                             const uint32_t* dense, int windows) {
  constexpr int kPer = producers_of<ROLES>;
  const int lane = threadIdx.x & 31, mine = (threadIdx.x - kProduceAt) >> 5, box = mine / kPer, part = mine % kPer;
  const int group = lane >> 3;
  const uint32_t bars = base + kBarsAt, source = blockIdx.x * 2 + box;
  constexpr int kQuads = kFill / 4, kBatches = kSlot / kFill, kOwn = kBatches / kPer;
  //: the warp's batch `g` of the call: its `g % kOwn`-th of slot `g / kOwn`, the slot's batch `part + kPer` of that
  const auto batch_of = [&](int g) { return (g / kOwn) * kBatches + part + kPer * (g % kOwn); };
  const uint4* const wide = reinterpret_cast<const uint4*>(values);
  const uint4* const boxes = reinterpret_cast<const uint4*>(dense);
  //: `kHeld` batches in registers, a batch a half: 96 registers a thread hold one, the lean row's 112 two. A
  //: batch's values are `kParts` parts of `kPart` instructions, a part in registers at a time
  constexpr int kHeld = batches_held<ROLES>, kParts = value_parts<ROLES>, kPart = kQuads / kParts;
  static_assert(kOwn % kHeld == 0);
  uint4 value[kHeld][kPart], bytes[kHeld];
  //: part `part_at` of batch `n` of the call into register half `h`: value instructions of four entries (8 lanes
  //: an entry, a lane 16 bytes)
  const auto load_values = [&](int n, int part_at, int h) {
#pragma unroll
    for (int qd = 0; qd < kPart; ++qd) {
      const uint32_t tk = token_of(source, n * kFill + 4 * (part_at * kPart + qd) + group);
      if constexpr (ROLES & kNoLoads) value[h][qd] = make_uint4(tk, tk, tk, tk);
      else value[h][qd] = wide[static_cast<long long>(row_of<ROLES>(tk, static_cast<uint32_t>(n * kFill + 4 * qd + group))) * (kRowWords / 4) + (lane & 7)];
    }
  };
  //: batch `n`'s dense bytes, all 32 entries' in one instruction (a lane an entry's 16)
  const auto load_bytes = [&](int n, int h) {
    const uint32_t tb = token_of(source, n * kFill + lane);
    if constexpr (ROLES & kNoLoads) bytes[h] = make_uint4(tb, tb, tb, tb);
    else bytes[h] = boxes[row_of<ROLES>(tb, static_cast<uint32_t>(n * kFill + lane) * 2u + 1u)];
  };
#pragma unroll
  for (int h = 0; h < kHeld; ++h) {
    load_values(batch_of(h), 0, h);
    load_bytes(batch_of(h), h);
  }
  lab::WarpSpan clock;
  if constexpr (ROLES & kFive) lab::WarpMark::start(times);
  else clock.start();
#pragma unroll 1
  for (int w = 0; w < windows; ++w) {
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
      if (w > 0) await<ROLES>(bars + kPoolFreed + 8 * (2 * box + slot), (w - 1) & 1);
      uint8_t* const tiles = smem + (2 * box + slot) * kPoolBytes;
      //: a rolled loop: a batch's addresses are formed in its trip, none held for the slot
#pragma unroll 1
      for (int o0 = 0; o0 < kOwn; o0 += kHeld)
#pragma unroll
       for (int h = 0; h < kHeld; ++h) {
        const int o = o0 + h, b = part + kPer * o, g = (2 * w + slot) * kOwn + o;
        //: a part's stores, then the same registers' loads for the next part: the batch's own, or the first of
        //: the batch `kHeld` on
        const auto part_out = [&](int part_at) {
#pragma unroll
          for (int qd = 0; qd < kPart; ++qd) {
            //: entry `e` of the slot: list `e / 16`, token `e % 16` of its step
            const int e = b * kFill + 4 * (part_at * kPart + qd) + group;
            *reinterpret_cast<uint4*>(tiles + (e / kEntries) * kTile + (lane & 7) * kBrick + (e % kEntries) * 16) =
                value[h][qd];
          }
          const bool last = part_at + 1 == kParts;
          load_values(batch_of(last ? g + kHeld : g), last ? 0 : part_at + 1, h);
        };
        part_out(0);
        //: the lane's entry: its 16 bytes times the entry's, 8 pairs of bf16, a brick's line each four
        const int e = b * kFill + lane;
        const float weight = static_cast<float>(sparse_of(token_of(source, w * kWindow + slot * kSlot + e)));
        const uint32_t word[4] = {bytes[h].x, bytes[h].y, bytes[h].z, bytes[h].w};
        uint4 line[2];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          uint32_t pair[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const uint32_t two = word[2 * k + j / 2] >> (16 * (j & 1));
            if constexpr (ROLES & kNoMath) pair[j] = two;
            else pair[j] = lab::bf16x2_round(weight * static_cast<float>(two & 255u),
                                             weight * static_cast<float>(two >> 8 & 255u));
          }
          line[k] = make_uint4(pair[0], pair[1], pair[2], pair[3]);
        }
        //: a batch's two lists take opposite bricks first: each store covers the 32 banks evenly
        const int first = (e / kEntries) & 1;
        uint8_t* const at = tiles + (e / kEntries) * kTile + kValueTile + (e % kEntries) * 16;
        *reinterpret_cast<uint4*>(at + first * kBrick) = first ? line[1] : line[0];
        *reinterpret_cast<uint4*>(at + (first ^ 1) * kBrick) = first ? line[0] : line[1];
        load_bytes(batch_of(g + kHeld), h);
        //: a rolled loop: a part's addresses are formed in its trip
#pragma unroll 1
        for (int part_at = 1; part_at < kParts; ++part_at) part_out(part_at);
      }
      lab::bulk_fence();
      if constexpr (ROLES & kSleep) {
        lab::bar_arrive(kPoolId + 2 * box + slot, kPoolCount);
      } else {
        __syncwarp();
        if (lane == 0) lab::mbar_arrive(bars + kPoolReady + 8 * (2 * box + slot));
      }
    }
  }
  if constexpr (ROLES & kFive) lab::WarpMark::stop(times);
  else clock.stop(times);
}

//: the readout's step: 64 tokens by the tile's 64 channels, or by its 72 columns where the den's is the 65th
template <int ROLES, int N>
__device__ __forceinline__ void read_step(float (&d)[N], const uint32_t (&a)[4], uint64_t b, bool add) {
  if constexpr (ROLES & kDenRead) lab::wg_bf16_n72_rs(d, a, b, add);
  else lab::wg_bf16_n64_rs(d, a, b, add);
}
//: where a call's den sums are: an fp32 a token behind the output's lines
__device__ __forceinline__ float* den_out(uint32_t* out) {
  return reinterpret_cast<float*>(out + static_cast<long long>(gridDim.x) * (kOutWindows * kWindow) * (kChannels / 2));
}
//: THE READOUT WARPGROUP: a chunk is 4 read digits of 16 tokens, a digit a warp's 16 rows. A warp expands its
//: tokens' coefficients in its registers and they are the step's A operand; the image's tile is its B; the
//: accumulator is the chunk's 64 tokens by 64 channels, stored to the panel as the f16 pairs a line holds
template <int ROLES>
__device__ __forceinline__ void read_role(uint8_t* smem, uint32_t base, long long* times, uint32_t* waits,
                                          const uint32_t* dense, uint32_t* out, int windows) {
  const int tid = threadIdx.x - kReadAt, warp = tid >> 5, lane = tid & 31, q = lane >> 2, t = lane & 3;
  const uint32_t bars = base + kBarsAt;
  //: a thread's rows are the chunk's tokens `16 warp + q` and that + 8; its words of a token's line `4 j + t`
  const uint32_t mine = lab::held(static_cast<uint32_t>(kPanelAt + (16 * warp + q) * kRowBytes + 4 * t));
  //: the reader at the thread's row `r` in chunk `c` of the call: token `q + 8 r` of the warp's digit
  const auto reader = [&](int c, int r) {
    const int k = c % kChunks;
    return reader_of(blockIdx.x, (c / kChunks) * kWindow + 16 * (8 * (k >> 1) + (k & 1) + 2 * warp) + q + 8 * r);
  };
  //: A's share: fold digits `8 h + 2 t` and the next of a row -- the word `2 h + t / 2` of the reader's 16 bytes
  //: on a box's leaves, its low or high pair -- loaded a chunk ahead
  uint32_t bytes[2][4];
  //: the wide row's: the lane's whole row of the chunk -- token `lane % 16` of the warp's digit on box `lane / 16`
  uint4 wide_row;
  uint8_t* const scratch = smem + kScratchAt + warp * kScratchBytes;
  const auto load = [&](int c) {
    if constexpr (ROLES & kWide) {
      const int k = c % kChunks;
      const uint32_t rd = reader_of(blockIdx.x, (c / kChunks) * kWindow + 16 * (8 * (k >> 1) + (k & 1) + 2 * warp)
                                                    + (lane & 15));
      wide_row = reinterpret_cast<const uint4*>(dense)[row_of<ROLES>(reader_row(rd, lane >> 4), static_cast<uint32_t>(c) * 64u + lane)];
    } else {
#pragma unroll
      for (int box = 0; box < 2; ++box)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          if constexpr (ROLES & kNoBytes) bytes[box][j] = reader(c, j & 1) + static_cast<uint32_t>(box + j);
          else bytes[box][j] = dense[static_cast<long long>(row_of<ROLES>(reader_row(reader(c, j & 1), box), (static_cast<uint32_t>(c) * 2u + static_cast<uint32_t>(j & 1)) * 2u + static_cast<uint32_t>(box))) * 4 + 2 * (j >> 1) + (t >> 1)];
    }
  };
  load(0);
  uint32_t on_image = 0, on_drain = 0, in_steps = 0, in_form = 0;
  //: the chunk's first step writes the accumulator whole: nothing is cleared between chunks
  //: the den's rows read a ninth brick of 8 columns: column 64 is the token's den
  static_assert(!(ROLES & kDenRead) || ((ROLES & kDenSplit) && (ROLES & kDenImage)));
  float d[ROLES & kDenRead ? 36 : 32];
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int w = 0; w < windows; ++w) {
#pragma unroll 1
    for (int c4 = 0; c4 < kChunks; ++c4) {
      const int c = kChunks * w + c4;
      //: chunk `2 h + c`'s digits are `8 h + c + 2 j`: its tile group, published for this window
      if constexpr (ROLES & kSleep) slept(kTilesId + c4, kTilesCount, on_image);
      else if constexpr (!(ROLES & kStill)) waited<ROLES>(bars + kTilesReady + 8 * c4, w & 1, on_image);
      uint32_t since = ROLES & kNoClock ? 0u : static_cast<uint32_t>(clock64());
      //: THE COEFFICIENTS, in registers: a row's two bytes times its reader's byte on the digit, a bf16 pair a word
      //: -- word `r | h << 1` of A, as an m64n16 accumulator lays its values
      if constexpr (ROLES & kWide) {
        //: the warp's rows relaid through its scratch: a lane's row stored whole, a thread's 8 words read back
        *reinterpret_cast<uint4*>(scratch + 16 * lane) = wide_row;
        __syncwarp();
#pragma unroll
        for (int box = 0; box < 2; ++box)
#pragma unroll
          for (int j = 0; j < 4; ++j)
            bytes[box][j] = *reinterpret_cast<const uint32_t*>(scratch + 16 * (16 * box + q + 8 * (j & 1))
                                                               + 4 * (2 * (j >> 1) + (t >> 1)));
      }
      uint32_t a[2][4];
#pragma unroll
      for (int box = 0; box < 2; ++box)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float weight = static_cast<float>(sparse_of(reader(c, j & 1)));
          const uint32_t two = bytes[box][j] >> (16 * (t & 1));
          a[box][j] = lab::bf16x2_round(weight * static_cast<float>(two & 255u),
                                        weight * static_cast<float>(two >> 8 & 255u));
        }
      const uint32_t first = static_cast<uint32_t>(8 * (c4 >> 1) + (c4 & 1));
      const uint32_t images = lab::held(base + kImageAt + first * tile_of<ROLES>);
      //: digit `j`'s step on each box's tile: its tokens are warp `j`'s rows, every other warp's A nothing
      if constexpr (ROLES & (kLean | kFive)) {
        //: TWO sets of A, a digit's two steps a commit group: a set is written again once the group two back,
        //: which read it, is done
        uint32_t rows[2][2][4];
        lab::wg_fence();
#pragma unroll
        for (int j = 0; j < kChunkDigits; ++j) {
          if (j >= 2) lab::wg_wait<1>();
#pragma unroll
          for (int box = 0; box < 2; ++box)
#pragma unroll
            for (int i = 0; i < 4; ++i) rows[j & 1][box][i] = lab::held(warp == j ? a[box][i] : 0u);
#pragma unroll
          for (int box = 0; box < 2; ++box)
            read_step<ROLES>(d, rows[j & 1][box],
                             lab::wg_desc(images + box * kBoxBytes + 2 * j * tile_of<ROLES>, 8, kBrickLines),
                             j != 0 || box != 0);
          lab::wg_commit();
        }
      } else {
        //: EVERY step's A is in its own registers before the first is issued: a register a step would write between
        //: two steps makes the toolkit wait out the one before
        uint32_t rows[kChunkDigits][2][4];
#pragma unroll
        for (int j = 0; j < kChunkDigits; ++j)
#pragma unroll
          for (int box = 0; box < 2; ++box)
#pragma unroll
            for (int i = 0; i < 4; ++i) rows[j][box][i] = lab::held(warp == j ? a[box][i] : 0u);
        if constexpr (!(ROLES & kNoClock)) in_form += lap(since);
        lab::wg_fence();
#pragma unroll
        for (int j = 0; j < kChunkDigits; ++j)
#pragma unroll
          for (int box = 0; box < 2; ++box)
            read_step<ROLES>(d, rows[j][box],
                             lab::wg_desc(images + box * kBoxBytes + 2 * j * tile_of<ROLES>, 8, kBrickLines),
                             j != 0 || box != 0);
        lab::wg_commit();
      }
      //: the next chunk's bytes, issued under the steps
      load(c + 1);
      lab::wg_wait<0>();
      if constexpr (!(ROLES & kNoClock)) in_steps += lap(since);
      //: the group is read: its box may publish the next window's over it
      if constexpr (!(ROLES & kStill))
        if constexpr (ROLES & kSleep) lab::bar_arrive(kReadId + c4, kTilesCount);
        else if (lane == 0) lab::mbar_arrive(bars + kTilesRead + 8 * c4);
      //: THE PANEL: a row's channels `8 j + 2 t` and the next rounded to an f16 pair, the word of the line an add
      //: takes as it lies; a warp's lanes on 32 banks. Where drain warps add it, once they have left it
      if constexpr (!(ROLES & kSelf))
        if (c >= 2) waited<ROLES>(bars + kPanelFreed + 8 * (c & 1), ((c >> 1) - 1) & 1, on_drain);
      uint8_t* const row = smem + mine + (c & 1) * kPanelBytes;
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int r = 0; r < 2; ++r)
          *reinterpret_cast<uint32_t*>(row + 8 * r * kRowBytes + 16 * j) =
              lab::pack_half2(d[4 * j + 2 * r], d[4 * j + 2 * r + 1]);
      if constexpr (ROLES & kDenDrain) {
        //: THE DEN'S WORD of a row, as the step summed it: the threads that hold column 64
        if (t == 0) {
#pragma unroll
          for (int r = 0; r < 2; ++r)
            *reinterpret_cast<float*>(row + 8 * r * kRowBytes + den_word(16 * warp + 8 * r)) = d[(ROLES & kDenRead ? 32 : 28) + 2 * r];
        }
      }
      __syncwarp();
      if constexpr (ROLES & kSelf) {
        //: THE WARP DRAINS ITS OWN 16 TOKENS, the digit `8 h + c + 2 warp`'s: a lane loads its word of a token's
        //: line and adds it, two batches of 8 in the accumulator's registers' place; no other warp reads the rows
        since = static_cast<uint32_t>(clock64());
        const long long first = static_cast<long long>(blockIdx.x) * (kOutWindows * kWindow)
                                + ((c / kChunks) % kOutWindows) * kWindow
                                + 16 * (8 * (c4 >> 1) + (c4 & 1) + 2 * warp);
        const uint32_t* const lines = reinterpret_cast<const uint32_t*>(
            smem + kPanelAt + (c & 1) * kPanelBytes + 16 * warp * kRowBytes) + lane;
#pragma unroll 1
        for (int n = 0; n < 16 / kBatch; ++n) {
          uint32_t held[kBatch];
#pragma unroll
          for (int i = 0; i < kBatch; ++i) held[i] = lines[(n * kBatch + i) * (kRowBytes / 4)];
#pragma unroll
          for (int i = 0; i < kBatch; ++i)
            lab::red_add_f16x2(out + (first + n * kBatch + i) * (kChannels / 2) + lane, held[i]);
        }
        on_drain += lap(since);
      } else {
        if constexpr (ROLES & kSleep) lab::bar_arrive(kPanelId + (c & 1), kPanelCount);
        else if (lane == 0) lab::mbar_arrive(bars + kPanelReady + 8 * (c & 1));
      }
    }
  }
  clock.stop(times);
  tell(waits, on_image, on_drain, in_steps, in_form);
  if constexpr (ROLES & kPark) lab::bar_arrive(kParkId, kParkCount);
  if constexpr (ROLES & kSpin)
    if (lane == 0) lab::mbar_arrive(bars + kDone);
}

//: A DRAIN WARP: a chunk's 32 tokens of its residue in four batches of 8; a lane holds a token's two channels and
//: adds one f16 pair
template <int ROLES>
__device__ __forceinline__ void drain_role(const uint8_t* smem, uint32_t base, long long* times, uint32_t* waits,
                                           int* sinks,
                                           uint32_t* out, int windows) {
  constexpr int kDrainers = drains_of<ROLES>;
  const int lane = threadIdx.x & 31, mine = (threadIdx.x - drain_at<ROLES>) >> 5;
  const uint32_t bars = base + kBarsAt;
  constexpr int kHold = ROLES & kFive ? kFiveHold : kBatch, kBatches = kTokens / kDrainers / kHold;
  static_assert(kBatches % 2 == 0);
  uint32_t held[2][kHold];
  uint32_t kept = 0;
  //: batch `n` of chunk `c` into register half `h`: a lane's word of a token's row is its f16 pair
  const auto load = [&](int c, int n, int h) {
#pragma unroll
    for (int i = 0; i < kHold; ++i)
      held[h][i] = reinterpret_cast<const uint32_t*>(
          smem + kPanelAt + (c & 1) * kPanelBytes + (mine + kDrainers * (n * kHold + i)) * kRowBytes)[lane];
  };
  uint32_t on_panel = 0;
  lab::WarpSpan clock;
  if constexpr (ROLES & kFive) lab::WarpMark::start(times);
  else clock.start();
#pragma unroll 1
  for (int c = 0; c < kChunks * windows; ++c) {
    if constexpr (ROLES & kSleep) slept(kPanelId + (c & 1), kPanelCount, on_panel);
    else if constexpr (ROLES & kFive) await<ROLES>(bars + kPanelReady + 8 * (c & 1), (c >> 1) & 1);
    else waited<ROLES>(bars + kPanelReady + 8 * (c & 1), (c >> 1) & 1, on_panel);
    if constexpr (!(ROLES & kFive)) {
      load(c, 0, 0);
      load(c, 1, 1);
    }
    //: THE DEN'S WORD, loaded as the chunk opens: a lane one of the warp's 32 tokens
    float den_sum = 0.f;
    if constexpr (ROLES & kDenDrain)
      den_sum = *reinterpret_cast<const float*>(smem + kPanelAt + (c & 1) * kPanelBytes
                                                + (mine + kDrainers * lane) * kRowBytes + den_word(mine + kDrainers * lane));
    //: the chunk's digit `j` is read digit `8 h + c + 2 j`: its 16 tokens' rows start at 16 of that
    const long long first = static_cast<long long>(blockIdx.x) * (kOutWindows * kWindow)
                            + ((c / kChunks) % kOutWindows) * kWindow
                            + 16 * (8 * ((c % kChunks) >> 1) + ((c % kChunks) & 1));
    if constexpr (ROLES & kFive) {
      //: the five-warpgroup row's drain holds one batch, in a rolled loop: its loads, then its adds
#pragma unroll 1
      for (int n = 0; n < kBatches; ++n) {
        load(c, n, 0);
#pragma unroll
        for (int i = 0; i < kHold; ++i) {
          const int token = mine + kDrainers * (n * kHold + i);
          lab::red_add_f16x2(out + (first + 32 * (token / 16) + token % 16) * (kChannels / 2) + lane, held[0][i]);
        }
      }
    } else {
#pragma unroll
      for (int n = 0; n < kBatches; ++n) {
#pragma unroll
        for (int i = 0; i < kHold; ++i) {
          const int token = mine + kDrainers * (n * kHold + i);
          const uint32_t pair = held[n & 1][i];
          if constexpr (ROLES & kNoAdds) kept ^= pair + token;
          else lab::red_add_f16x2(out + (first + 32 * (token / 16) + token % 16) * (kChannels / 2) + lane, pair);
        }
        if (n + 2 < kBatches) load(c, n + 2, n & 1);
      }
      if constexpr (ROLES & kDenDrain) {
        //: THE DEN'S ADD, behind the chunk's lines': the lane's token's sum
        const int token = mine + kDrainers * lane;
        lab::red_add_f32(den_out(out) + first + 32 * (token / 16) + token % 16, den_sum);
      }
    }
    __syncwarp();
    if (lane == 0) lab::mbar_arrive(bars + kPanelFreed + 8 * (c & 1));
  }
  if constexpr (ROLES & kFive) {
    lab::WarpMark::stop(times);
  } else {
    clock.stop(times);
    tell(waits, on_panel, 0u);
  }
  if constexpr (ROLES & kNoAdds) lab::sink(sinks, kept);
}

//: A ROLE A ROW LEAVES OUT: its warps' spans are empty, at the roles' start, so the block's span is the roles' own
__device__ __forceinline__ void idle_role(long long* times, uint32_t* waits) {
  lab::WarpSpan clock;
  clock.start();
  clock.stop(times);
  tell(waits, 0u, 0u);
}

template <int ROLES>
__global__ void __launch_bounds__(threads_of<ROLES>, 1)
    window(long long* times, uint32_t* waits, int* sinks, const uint32_t* values, const uint32_t* dense, float* state,
           uint32_t* out, int windows) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x;
  const uint32_t base = lab::smem(smem);
  //: the images cleared: the state is zero at the call's start
#pragma unroll 1
  for (int i = tid; i < 2 * kBoxBytes / 4; i += threads_of<ROLES>) *reinterpret_cast<uint32_t*>(smem + kImageAt + 4 * i) = 0u;
  if constexpr (ROLES & kDenFold) {
    //: the den's ones: brick c's every line bf16 1 in slot c
#pragma unroll 1
    for (int i = tid; i < kDenSteps * kOnesBrick / 2; i += threads_of<ROLES>)
      *reinterpret_cast<uint16_t*>(smem + kOnesAt + 2 * i) =
          (i & 7) == i / (kOnesBrick / 2) ? static_cast<uint16_t>(0x3f80) : static_cast<uint16_t>(0);
  }
  if (tid == 0) {
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      lab::mbar_init(base + kBarsAt + kPoolReady + 8 * p, producers_of<ROLES>);
      lab::mbar_init(base + kBarsAt + kPoolFreed + 8 * p, 1);
    }
#pragma unroll
    for (int k = 0; k < kChunks; ++k) {
      lab::mbar_init(base + kBarsAt + kTilesReady + 8 * k, 2 * kGroup / 32);
      lab::mbar_init(base + kBarsAt + kTilesRead + 8 * k, kGroup / 32);
    }
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      lab::mbar_init(base + kBarsAt + kPanelReady + 8 * p, kGroup / 32);
      lab::mbar_init(base + kBarsAt + kPanelFreed + 8 * p, drains_of<ROLES>);
    }
    lab::mbar_init(base + kBarsAt + kDone, kGroup / 32);
  }
  lab::bulk_fence();
  __syncthreads();

  //: EACH ROLE'S WHOLE BODY IN ITS OWN BRANCH, a warpgroup's registers set at the branch's head in every row: the
  //: folds' raise is the release of the readout and of the producers' and drains' warpgroup
  //: (docs/lab/carry_window.md)
  if (tid < kReadAt) {
    lab::regs_up<kFoldRegs>();
    if constexpr (ROLES & kStill) {
      idle_role(times, waits);
      //: present and idle until the readout is done
      if constexpr (ROLES & kPark) lab::bar_sync(kParkId, kParkCount);
      if constexpr (ROLES & kSpin) lab::mbar_wait(base + kBarsAt + kDone, 0);
    }
    else fold_role<ROLES>(smem, base, times, waits, state, windows);
  } else if (tid < kProduceAt) {
    lab::regs_down<read_regs<ROLES>>();
    if constexpr (ROLES & kRead) read_role<ROLES>(smem, base, times, waits, dense, out, windows);
    else idle_role(times, waits);
  } else if constexpr (ROLES & kFive) {
    //: a warpgroup of four producers and one of four drain warps, each its own count
    if (tid < drain_at<ROLES>) {
      lab::regs_down<kFiveProduceRegs>();
      if constexpr (ROLES & kFold) produce_role<ROLES>(smem, base, times, values, dense, windows);
      else idle_role(times, waits);
    } else {
      lab::regs_down<kFiveDrainRegs>();
      if constexpr (ROLES & kRead) drain_role<ROLES>(smem, base, times, waits, sinks, out, windows);
      else idle_role(times, waits);
    }
  } else {
    //: the producers and the drain warps are one warpgroup: one register call, the four warps together
    lab::regs_down<serve_regs<ROLES>>();
    if (tid < (ROLES & kSelf ? kThreads : kDrainAt)) {
      if constexpr (ROLES & kFold) produce_role<ROLES>(smem, base, times, values, dense, windows);
      else idle_role(times, waits);
    } else {
      if constexpr (ROLES & kRead) drain_role<ROLES>(smem, base, times, waits, sinks, out, windows);
      else idle_role(times, waits);
    }
  }
}

}  // namespace carry_window

inline uint32_t bf16_host(int v) {
  const float f = static_cast<float>(v);
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u >> 16;
}

//: an integer under 2^24 as the image holds it (truncated to bf16) and as a drain's line holds it (rounded to f16)
inline long long image_host(long long v) {
  const float f = static_cast<float>(v);
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u &= 0xffff0000u;
  float g;
  std::memcpy(&g, &u, 4);
  return static_cast<long long>(g);
}
inline long long line_host(long long v) {
  if (v < 2048) return v;
  int e = 0;
  while ((v >> e) >= 2048) ++e;
  long long kept = v >> e;
  const long long rest = v & ((1ll << e) - 1), half = 1ll << (e - 1);
  kept += rest > half || (rest == half && (kept & 1));
  return kept << e;
}
inline float half_to_float(uint16_t h) {
  const int e = (h >> 10) & 31, m = h & 1023;
  const float mag = e == 0 ? std::ldexp(static_cast<float>(m), -24) : std::ldexp(static_cast<float>(m | 1024), e - 25);
  return h & 0x8000 ? -mag : mag;
}

template <int ROLES>
void run(lab::Bench& b, uint32_t* waits, const uint32_t* values, const uint32_t* dense, float* state, uint32_t* out) {
  using namespace carry_window;
  constexpr int kTimed = 64;
  const std::string side = (ROLES & 3) == kFold ? "the fold's side alone: producers and steps"
                           : (ROLES & 3) == kRead ? "the read side alone: the image, the readout and the drain"
                           : "the carry: both sides";
  const std::string name = side + (ROLES & kNoLoads ? ", the producers' global loads taken out" : "")
                           + (ROLES & kNoAdds ? ", the drains' global adds taken out" : "")
                           + (ROLES & kNoImage ? ", the image's stores taken out" : "")
                           + (ROLES & kStill ? ", no fold warp running" : "")
                           + (ROLES & kSuspend ? ", every wait suspended" : "")
                           + (ROLES & kSelf ? ", the readout draining its own rows and four producers" : "")
                           + (ROLES & kLean ? ", the readout's A in two sets and two batches a producer" : "")
                           + (ROLES & kFive ? ", five warpgroups: four producers and four drain warps" : "")
                           + (ROLES & kSleep ? ", the long waits asleep at named barriers" : "")
                           + (ROLES & kNoMath ? ", the producers' coefficient arithmetic taken out" : "")
                           + (ROLES & kNoBytes ? ", the readout's byte loads taken out" : "")
                           + (ROLES & kWide ? ", the readout's bytes one wide load a warp" : "")
                           + (ROLES & kNoFence ? ", the fold's fence behind its image taken out" : "")
                           + (ROLES & kPark ? ", the fold warps parked at a named barrier" : "")
                           + (ROLES & kSpin ? ", the fold warps spinning at a polled barrier" : "")
                           + (ROLES & kDump ? ", the fold's state stored out in every call" : "")
                           + (ROLES & kBig ? ", the tables 16 replicas the cache cannot hold" : "")
                           + (ROLES & kAligned ? ", the sides' carves equal: an image tile a list, its lines four threads' words" : "")
                           + (ROLES & kNoClock ? ", no clock inside a role" : "")
                           + (ROLES & kPitch ? (ROLES & kDenImage ? ", the image's tiles 161 lines apart" : ", the image's tiles 154 lines apart") : "")
                           + ((ROLES & kDen) == kDen ? ", the den: two more steps a slot, the readout's steps 72 columns and an fp32 a token"
                              : ROLES & kDen ? std::string(", the den's parts alone: its fold steps")
                                                   + (ROLES & kDenImage ? ", its image brick" : "")
                                                   + (ROLES & kDenSplit ? ", its register split" : "")
                                                   + (ROLES & kDenRead ? ", its 72-column steps" : "")
                                             : std::string());
  constexpr int kLaunch = threads_of<ROLES>, kWarps = kLaunch >> 5;
  const auto kernel = window<ROLES>;
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  const size_t words = static_cast<size_t>(b.sms) * kOutWindows * kWindow * (kChannels / 2);
  const size_t tokens = static_cast<size_t>(b.sms) * kOutWindows * kWindow;
  LAB_CHECK(cudaMemset(out, 0, (words + tokens) * 4));
  if ((ROLES & ~(kSuspend | kSelf | kLean | kFive | kSleep | kWide | kDump | kBig | kDen | kPitch | kAligned | kNoClock)) == (kFold | kRead)) {
    kernel<<<b.sms, kLaunch, kSmemBytes>>>(b.times, waits, b.sinks, values, dense, state, out, kOutWindows);
    LAB_CHECK(cudaDeviceSynchronize());
    const size_t per = static_cast<size_t>(2 * kGroup) * state_words<ROLES>;
    std::vector<float> got_state(static_cast<size_t>(b.sms) * per);
    std::vector<uint32_t> got(words);
    std::vector<float> got_den(tokens);
    if ((ROLES & kDen) == kDen) LAB_CHECK(cudaMemcpy(got_den.data(), out + words, tokens * 4, cudaMemcpyDeviceToHost));
    LAB_CHECK(cudaMemcpy(got_state.data(), state, got_state.size() * sizeof(float), cudaMemcpyDeviceToHost));
    LAB_CHECK(cudaMemcpy(got.data(), out, words * 4, cudaMemcpyDeviceToHost));
    long long wrong_state = 0, wrong_out = 0, wrong_den = 0;
    //: the host's: s[box][m][a][b], read at each window's start through its image, then the window folded in
    std::vector<long long> s(2 * kChannels * kLists * kLists);
    const auto at = [](int box, int m, int a, int bb) { return ((box * kChannels + m) * kLists + a) * kLists + bb; };
    //: the den's: a value a (box, list, digit), the lists' coefficients summed
    std::vector<long long> sden(2 * kLists * kLists);
    for (int blk = 0; blk < b.sms; ++blk) {
      std::fill(s.begin(), s.end(), 0);
      std::fill(sden.begin(), sden.end(), 0);
      for (int w = 0; w < kOutWindows; ++w) {
        if ((ROLES & kDen) == kDen)
          for (int token = 0; token < kWindow; ++token) {
            //: a token's den: the den's image by the token's coefficients, as its channels' are
            const int bb = token / 16;
            const uint32_t rt = reader_of(blk, w * kWindow + token);
            long long want = 0;
            for (int box = 0; box < 2; ++box)
              for (int a = 0; a < kLists; ++a)
                want += image_host(ROLES & kAligned ? sden[(box * kLists + bb) * kLists + a]
                                                    : sden[(box * kLists + a) * kLists + bb])
                        * sparse_of(static_cast<int>(rt)) * dense_of(static_cast<int>(reader_row(rt, box)), a);
            wrong_den += got_den[(static_cast<size_t>(blk) * kOutWindows + w) * kWindow + token] != static_cast<float>(want);
          }
        for (int token = 0; token < kWindow; ++token)
          for (int m = 0; m < kChannels; ++m) {
            const int bb = token / 16;
            const uint32_t rt = reader_of(blk, w * kWindow + token);
            long long want = 0;
            for (int box = 0; box < 2; ++box)
              for (int a = 0; a < kLists; ++a)
                want += image_host(ROLES & kAligned ? s[at(box, m, bb, a)] : s[at(box, m, a, bb)])
                        * sparse_of(static_cast<int>(rt)) * dense_of(static_cast<int>(reader_row(rt, box)), a);
            const size_t row = (static_cast<size_t>(blk) * kOutWindows + w) * kWindow + token;
            const uint32_t word = got[row * (kChannels / 2) + m / 2];
            wrong_out += half_to_float(static_cast<uint16_t>(word >> (16 * (m & 1)))) != static_cast<float>(line_host(want));
          }
        for (int box = 0; box < 2; ++box)
          for (int e = 0; e < kWindow; ++e) {
            const int t = static_cast<int>(token_of(blk * 2 + box, w * kWindow + e)), a = e / kEntries;
            for (int m = 0; m < kChannels; ++m)
              for (int bb = 0; bb < kLists; ++bb)
                s[at(box, m, a, bb)] += value_of(t, m) * sparse_of(t) * dense_of(t, bb);
            for (int bb = 0; bb < kLists; ++bb) sden[(box * kLists + a) * kLists + bb] += sparse_of(t) * dense_of(t, bb);
          }
      }
      for (int tid = 0; tid < 2 * kGroup; ++tid) {
        const int box = tid >> 7, rows = 16 * ((tid >> 5) & 3) + (tid & 31) / 4, pair = tid & 3;
        for (int a = 0; a < kLists; ++a)
          for (int i = 0; i < 8; ++i) {
            const int m = rows + 8 * ((i >> 1) & 1), bb = 8 * (i >> 2) + 2 * pair + (i & 1);
            wrong_state += got_state[blk * per + static_cast<size_t>(tid) * state_words<ROLES> + a * 8 + i]
                           != static_cast<float>(s[at(box, m, a, bb)]);
          }
        if (ROLES & kDenFold)
          for (int i = 0; i < 4; ++i) {
            //: the den's accumulator: column c's row m is list `8 (c / 2) + m / 8`'s den on digit `8 (c % 2) + m % 8`
            const int m = rows + 8 * ((i >> 1) & 1), c = 2 * pair + (i & 1);
            const long long want = c < kDenSteps ? sden[(box * kLists + 8 * (c >> 1) + (m >> 3)) * kLists + 8 * (c & 1) + (m & 7)] : 0;
            wrong_state += got_state[blk * per + static_cast<size_t>(tid) * state_words<ROLES> + kLists * 8 + i]
                           != static_cast<float>(want);
          }
      }
    }
    b.check(name, wrong_state == 0 && wrong_out == 0 && wrong_den == 0,
            std::to_string(wrong_state) + " of " + std::to_string(got_state.size()) + " state values and "
                + std::to_string(wrong_out) + " of " + std::to_string(2 * words) + " output values differ"
                + ((ROLES & kDen) == kDen ? ", " + std::to_string(wrong_den) + " of " + std::to_string(tokens) + " den sums" : ""));
  } else {
    b.unchecked(name, "an isolation: a side's roles or one of their accesses absent");
  }
  //: THE WAITS: a role's warps' mean cycles a window at each of its barriers (fold: on the readout's read of a
  //: tile group, on its producer; readout: on a tile group's publish, on the panel's drain; drain: on the
  //: panel), and the readout's cycles forming its steps' A and in its steps. A producer's are not clocked
  struct Site { int first, count, slot; const char* name; };
  constexpr int kSites = 7;
  const Site sites[kSites] = {{0, 8, 0, "a fold warp waits on the readout's read"},
                              {0, 8, 1, "a fold warp waits on its producer"},
                              {8, 4, 0, "a readout warp waits on the image"},
                              {8, 4, 1, "a readout warp waits on the drain"},
                              {8, 4, 3, "a readout warp forming its steps' A"},
                              {8, 4, 2, "a readout warp in its steps"},
                              {drain_at<ROLES> >> 5, drains_of<ROLES>, 0, "a drain warp waits on the panel"}};
  constexpr int kFoldSites = 2;
  lab::Samples waited_at[kSites];
  const lab::Samples cycles = b.variant(
      [&] { kernel<<<b.sms, kLaunch, kSmemBytes>>>(b.times, waits, b.sinks, values, dense, state, out, kTimed); },
      [&] {
        std::vector<uint32_t> h(static_cast<size_t>(b.sms) * kWarps * kCounts);
        LAB_CHECK(cudaMemcpy(h.data(), waits, h.size() * 4, cudaMemcpyDeviceToHost));
        for (int i = 0; i < kSites; ++i) {
          double sum = 0;
          for (int blk = 0; blk < b.sms; ++blk)
            for (int w = sites[i].first; w < sites[i].first + sites[i].count; ++w)
              sum += h[kCounts * (static_cast<size_t>(blk) * kWarps + w) + sites[i].slot];
          waited_at[i].v.push_back(sum / (static_cast<double>(b.sms) * sites[i].count * kTimed));
        }
        return b.span(kWarps) / kTimed;
      });
  b.record(name, "cycles a window a CTA", cycles);
  //: the composed row's sites, and the read side's own but the fold's wait on a producer, which does not run there
  //: the long call and its wall time also run for the winner's form and the big tables': the headline's rows
  if constexpr (ROLES == (kFold | kRead) || ROLES == (kFold | kRead | kSuspend | kLean)
                || ROLES == (kFold | kRead | kBig) || ROLES == (kFold | kRead | kSuspend | kLean | kBig)) {
    //: THE CARD'S CLOCK: a long call's span in SM cycles over its wall time, the launch and its wait included (a
    //: call of some 20 ms: they are under a percent of it)
    constexpr int kLong = 4096;
    lab::Samples clock_rate, wall;
    for (int r = 0; r < 5; ++r) {
      LAB_CHECK(cudaDeviceSynchronize());
      const auto t0 = std::chrono::steady_clock::now();
      kernel<<<b.sms, kLaunch, kSmemBytes>>>(b.times, waits, b.sinks, values, dense, state, out, kLong);
      LAB_CHECK(cudaDeviceSynchronize());
      const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      clock_rate.v.push_back(b.span(kWarps) / seconds / 1e9);
      wall.v.push_back(seconds * 1000.0 * (65536.0 / (static_cast<double>(kLong) * 4 * kWindow)));
    }
    b.unchecked(name + ": the card's clock", "a rate read off this row's long call, which the row checks");
    b.record(name + ": the card's clock, the long call's span over its wall time", "GHz", clock_rate);
    //: THE WALL TIME, the headline a drifting clock cannot touch: the long call's seconds scaled to 65,536 tokens
    //: (a window is 4 x kWindow = 1,024 tokens across the call's CTAs)
    b.unchecked(name + ": by wall time, scaled to 65,536 tokens",
                "the same launches as the card's clock: the row checks the kernel");
    b.record(name + ": the long call by wall time, scaled to 65,536 tokens", "milliseconds", wall);
  }
  if (!(ROLES & (kNoLoads | kNoAdds | kNoImage | kNoClock)) && ROLES & kRead)
    for (int i = 0; i < kSites; ++i)
      if (ROLES & (kLean | kFive) && i == kFoldSites + 2) continue;
      else if (ROLES & kFive && i == kSites - 1) continue;
      else if (ROLES & kSelf ? i != kSites - 1 && (ROLES & kFold || i != 1)
          : ROLES & kStill ? i > kFoldSites : ROLES & kFold || i != 1)
        b.record(name + ": " + (ROLES & kSelf && i == kFoldSites + 1 ? "a readout warp draining its own rows" : sites[i].name),
                 "cycles a window a warp", waited_at[i]);
}

int main(int argc, char** argv) {
  using namespace carry_window;
  lab::Bench b(argc, argv);
  for (int roles : {1, 2, 3, 3 | kNoLoads, 3 | kNoAdds, 3 | kNoImage, 2 | kStill, 2 | kSuspend, 3 | kSuspend,
                    3 | kSelf, 1 | kSelf, 2 | kSelf, 3 | kLean, 1 | kLean, 2 | kLean,
                    3 | kFive, 1 | kFive, 2 | kFive, 3 | kSleep, 1 | kSleep, 2 | kSleep, 1 | kNoLoads, 1 | kNoMath,
                    1 | kNoLoads | kNoMath, 2 | kNoBytes, 2 | kStill | kNoBytes, 3 | kNoBytes,
                    3 | kNoBytes | kNoAdds, 3 | kWide, 2 | kWide, 2 | kStill | kWide, 3 | kWide | kSleep,
                    2 | kNoImage, 2 | kNoFence, 2 | kNoImage | kNoFence, 3 | kNoFence,
                    2 | kStill | kPark, 2 | kStill | kSpin, 3 | kDump,
                    3 | kSuspend | kLean, 3 | kSuspend | kWide, 3 | kSuspend | kLean | kWide,
                    3 | kSleep | kLean | kWide, 3 | kBig, 3 | kSuspend | kLean | kBig, 1 | kBig, 1 | kDen, 3 | kSuspend | kLean | kDen, 3 | kSuspend | kLean | kBig | kDen, 3 | kSuspend | kLean | kPitch, 3 | kSuspend | kLean | kDen | kPitch, 3 | kSuspend | kLean | kAligned, 3 | kSuspend | kLean | kAligned | kDen, 3 | kSuspend | kLean | kNoClock, 3 | kSuspend | kLean | kDen | kNoClock,
                    3 | kSuspend | kLean | kAligned | kNoClock, 3 | kSuspend | kLean | kAligned | kDen | kNoClock,
                    3 | kSuspend | kLean | kNoClock | kDenFold, 3 | kSuspend | kLean | kNoClock | kDenFold | kDenImage,
                    3 | kSuspend | kLean | kNoClock | kDenFold | kDenImage | kDenSplit,
                    3 | kSuspend | kLean | kNoClock | kDenFold | kDenImage | kDenSplit | kDenRead}) {
    const std::string k = lab::kernel("window", {roles});
    //: with no fold, the state's one group of steps is a rolled loop: one instruction
    b.expect(k, "HGMMA", "==",
             (roles & kFold ? 2 * kSlotLists + (roles & kDenFold ? kDenSteps : 0) : roles & kStill ? 0 : 1)
                 + (roles & kRead ? 2 * kChunkDigits : 0));
    if (roles & kRead) {
      //: the readout's panel is read by the drain warps' own loads, not by a step: its arrival needs no fence
      //: nor does the wide row's scratch, which only its own warp's loads read
      //: the row that takes the fold's fence out leaves its four tile groups' arrivals unfenced: what it measures
      b.expect(k, "STORE_UNFENCED", "<=", roles & kNoFence ? 5 : roles & kWide ? 2 : 1);
      //: the readout's trip holds its coefficients' rounds and their load beside its steps, by design: the tensor
      //: pipe is idle most of a window
      b.expect(k, "STEP_XU", ">=", 0);
      b.expect(k, "STEP_GLOBAL", ">=", 0);
    }
    //: the five-warpgroup row's producer holds one value instruction's entries: a part's loads are read a trip of
    //: its rolled loop on, with nothing between
    if ((roles & kFive) && (roles & kFold)) b.expect(k, "LOAD_IN_TRIP", "<=", 14);
    //: the wide row relays a warp's rows through its scratch -- stored a lane a row, read back a thread's words: a
    //: rearrangement on the way (KERNEL_STANDARDS 33), its reloads the 8 words
    if (roles & kWide) b.expect(k, "STORE_RELOAD", "<=", 8);
    b.exempt(k, "a composition probe: the carry's roles in one CTA, no kernel's step");
    //: once a slot or a chunk a role: a `__syncwarp` before a one-lane arrival is a convergence call, and each
    //: barrier's polled wait a region -- five roles' in one kernel
    b.expect(k, "LOOP_CONVERGENCE", "<=", 64);
    //: the roles' loops hold their steps beside their barriers' polled waits: the waits' branches are the steps'
    b.expect(k, "BSSY", ">=", 0);
  }
  if (b.expect_only) return 0;
  //: the tables hold `kReplicas` copies of the 65,536 tokens' rows: a replica's row is its token's, so the big
  //: row's spread reads the same values and the host's mirror stands
  std::vector<uint32_t> host_values(static_cast<size_t>(kReplicas) * kTableTokens * kRowWords),
      host_dense(static_cast<size_t>(kReplicas) * kTableTokens * 4);
  for (size_t w = 0; w < host_values.size(); ++w) {
    const int t = static_cast<int>(w / kRowWords) & (kTableTokens - 1), c = 2 * static_cast<int>(w % kRowWords);
    host_values[w] = bf16_host(value_of(t, c)) | bf16_host(value_of(t, c + 1)) << 16;
  }
  for (size_t w = 0; w < host_dense.size(); ++w) {
    const int t = static_cast<int>(w / 4) & (kTableTokens - 1), b0 = 4 * static_cast<int>(w % 4);
    host_dense[w] = dense_of(t, b0) | dense_of(t, b0 + 1) << 8 | dense_of(t, b0 + 2) << 16 | dense_of(t, b0 + 3) << 24;
  }
  uint32_t *values = nullptr, *dense = nullptr, *out = nullptr;
  float* state = nullptr;
  LAB_CHECK(cudaMalloc(&values, host_values.size() * 4));
  LAB_CHECK(cudaMemcpy(values, host_values.data(), host_values.size() * 4, cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMalloc(&dense, host_dense.size() * 4));
  LAB_CHECK(cudaMemcpy(dense, host_dense.data(), host_dense.size() * 4, cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMalloc(&state, sizeof(float) * b.sms * 2 * kGroup * state_words<kDen>));
  LAB_CHECK(cudaMalloc(&out, static_cast<size_t>(b.sms) * kOutWindows * kWindow * (kChannels / 2 + 1) * 4));
  uint32_t* waits = nullptr;
  LAB_CHECK(cudaMalloc(&waits, static_cast<size_t>(b.sms) * (threads_of<kFive> >> 5) * kCounts * 4));
  run<kFold>(b, waits, values, dense, state, out);
  run<kRead>(b, waits, values, dense, state, out);
  run<kFold | kRead>(b, waits, values, dense, state, out);
  run<kFold | kRead | kNoLoads>(b, waits, values, dense, state, out);
  run<kFold | kRead | kNoAdds>(b, waits, values, dense, state, out);
  run<kFold | kRead | kNoImage>(b, waits, values, dense, state, out);
  run<kRead | kStill>(b, waits, values, dense, state, out);
  run<kRead | kSuspend>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSelf>(b, waits, values, dense, state, out);
  run<kFold | kSelf>(b, waits, values, dense, state, out);
  run<kRead | kSelf>(b, waits, values, dense, state, out);
  run<kFold | kRead | kLean>(b, waits, values, dense, state, out);
  run<kFold | kLean>(b, waits, values, dense, state, out);
  run<kRead | kLean>(b, waits, values, dense, state, out);
  run<kFold | kRead | kFive>(b, waits, values, dense, state, out);
  run<kFold | kFive>(b, waits, values, dense, state, out);
  run<kRead | kFive>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSleep>(b, waits, values, dense, state, out);
  run<kFold | kSleep>(b, waits, values, dense, state, out);
  run<kRead | kSleep>(b, waits, values, dense, state, out);
  run<kFold | kNoLoads>(b, waits, values, dense, state, out);
  run<kFold | kNoMath>(b, waits, values, dense, state, out);
  run<kFold | kNoLoads | kNoMath>(b, waits, values, dense, state, out);
  run<kRead | kNoBytes>(b, waits, values, dense, state, out);
  run<kRead | kStill | kNoBytes>(b, waits, values, dense, state, out);
  run<kFold | kRead | kNoBytes>(b, waits, values, dense, state, out);
  run<kFold | kRead | kNoBytes | kNoAdds>(b, waits, values, dense, state, out);
  run<kFold | kRead | kWide>(b, waits, values, dense, state, out);
  run<kRead | kWide>(b, waits, values, dense, state, out);
  run<kRead | kStill | kWide>(b, waits, values, dense, state, out);
  run<kFold | kRead | kWide | kSleep>(b, waits, values, dense, state, out);
  run<kRead | kNoImage>(b, waits, values, dense, state, out);
  run<kRead | kNoFence>(b, waits, values, dense, state, out);
  run<kRead | kNoImage | kNoFence>(b, waits, values, dense, state, out);
  run<kFold | kRead | kNoFence>(b, waits, values, dense, state, out);
  run<kRead | kStill | kPark>(b, waits, values, dense, state, out);
  run<kRead | kStill | kSpin>(b, waits, values, dense, state, out);
  run<kFold | kRead | kDump>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kWide>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kWide>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSleep | kLean | kWide>(b, waits, values, dense, state, out);
  run<kFold | kRead | kBig>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kBig>(b, waits, values, dense, state, out);
  run<kFold | kBig>(b, waits, values, dense, state, out);
  run<kFold | kDen>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kDen>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kBig | kDen>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kPitch>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kDen | kPitch>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kAligned>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kAligned | kDen>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kNoClock>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kDen | kNoClock>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kAligned | kNoClock>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kAligned | kDen | kNoClock>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kNoClock | kDenFold>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kNoClock | kDenFold | kDenImage>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kNoClock | kDenFold | kDenImage | kDenSplit>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kNoClock | kDenFold | kDenImage | kDenSplit | kDenRead>(b, waits, values, dense, state, out);
  return 0;
}
