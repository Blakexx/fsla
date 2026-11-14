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
// fold accumulators, 4 more readout accumulators, a drain's one word. The den's image store is issued by every
// lane, the lanes without a den value to one place a warp. A CHAIN of rows with no clock inside a role adds the
// den a part at a time (the fold's steps, the image's brick, the register split, the 72 columns, the whole), and
// one row moves the winner's tiles to the den's pitch. Rows with the VALUES TILE-MAJOR in global memory price the
// producers' loads of a token's 8 lines a kilobyte apart, the tables cached and once-read. THE LISTED rows' fold
// reads its entries FROM LISTS: a batch's 32 places loaded a batch before the loads they address, a window's tokens its own
// 1,024, an entry's sparse byte and dense bytes a block each of the levels' tile-major rows. THE EXPANDED rows land a list's values with one bulk
// copy from a buffer a pass before the carry is taken to have written, a column's lists contiguous a window; the
// fold waits on those bytes beside its producer's arrival. THE WHOLE-ENTRY rows' copy carries the coefficients'
// bytes too: a slot's 8 copies are issued when the slot is freed, its producer waits for their bytes, then expands
// each entry's landed line in place -- no global load in its trips. Those rows' 16 columns are 40 MB that 16 boxes
// each read; THE APART rows give each box its own expanded windows, 660 MB the cache does not hold; THE FRESH rows
// keep the 16 columns and run 128 windows of their own addresses, read by their 16 boxes at once and never before.
// THE PACKED rows are fed the stream as ruled: a column's entries laid as its lists' steps, the lists uneven (8 to 24
// entries), a list's last step stored truncated and used where it lands, a step one bulk copy of its own size.
// THE PACKED rows' TIMELINE: a producer loads a window's step records and slot headers A WINDOW AHEAD of their use
// (a lane a record, a lane a header: 22 words held). Its trip is a slot: the copies of the slot `slots_ahead` on,
// issued from those registers -- none ahead with two slots of 8 tiles a box, TWO SLOTS AHEAD with four of 4 -- its
// own slot's header stored, the wait on its bytes, a trip of 32 entries a pair of tiles (a landed line's load, its
// rounds, two stores), the arrival. A fold's slot is the wait, the header's three words, then each list's steps, a
// loop a list. With TWO PRODUCERS A BOX the call's even slots are one's trips and its odd ones the other's: the slot
// two on is each one's own next. That row is the fold's side alone and unchecked, as every fold-alone row is.
// THE FOLD-EXPANDS rows' TIMELINE: a producer's slot is its 8 copies, issued when the fold frees the slot; a fold's
// slot is the wait on those bytes, one landed line's load a thread, its rounds, two stores, the warpgroup's
// barrier, then its 8 steps as one block. Live in the expansion beside the fold's 128 accumulators: the landed
// line's 4 words, its weight and a line's 4 pairs. THE FOLD-ISSUES rows have no producer: a fold issues its slot's
// next copies -- the window after's -- as it leaves the slot, a window's steps ahead of their use; the serve
// warpgroup's other two warps idle, or with FOUR DRAIN WARPS take 16 of a chunk's 64 tokens each. THE SUM-A-BOX
// rows' readout holds two accumulators, 64 words beside its 16 of A: a chunk's steps are two chains of 4.
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
              kBarsAt = kPanelAt + 2 * kPanelBytes, kScratchAt = kBarsAt + 208;
//: a readout warp's scratch: its chunk's 32 rows of 16 bytes, row `16 box + token`
constexpr int kScratchBytes = 32 * 16, kOnesAt = kScratchAt + 4 * kScratchBytes;
//: THE DEN'S B: a brick a den step, 16 lines, every line of brick c holding 1 in slot c and nothing else -- an
//: M|N-major tile of 16 by 8 whose column c is ones
constexpr int kDenSteps = 4, kOnesBrick = 16 * 16, kHeadersAt = kOnesAt + kDenSteps * kOnesBrick;
//: THE PACKED row's: a slot's HEADER, a producer's to its fold before the slot is ready -- three words of 64 bits:
//: a nibble a list its steps in the slot; a nibble a list its last step's entries where that step is truncated;
//: and whether the slot is the window's last
constexpr int kHeader = 32, kFourBarsAt = kHeadersAt + 8 * kHeader;
//: the four-slot row's barriers: a slot's filled, freed and landed, a (box, slot) each
constexpr int kSmemBytes = kFourBarsAt + 3 * 8 * 8;
//: the barriers, 8 bytes each: a pool slot filled and freed (box, slot); an image's tile group published and read
//: (a group is the 4 read digits `8 h + c + 2 j` of a chunk `2 h + c`: every fold thread holds a quarter of it); a
//: panel filled and drained
constexpr int kPoolReady = 0, kPoolFreed = 32, kTilesReady = 64, kTilesRead = 96, kPanelReady = 128, kPanelFreed = 144, kDone = 160;
//: the expanded row's: a pool slot's values landed, the bulk copies' bytes complete (box, slot)
constexpr int kPoolLanded = 176;
//: the named barrier the parked row's fold warps sleep at until the readout's end: theirs and the readout's threads
constexpr int kParkId = 15, kParkCount = 3 * kGroup;
constexpr int kOutWindows = 4;
//: a row's roles, and what a row takes OUT of them to find what the two sides contend for: the producers' global
//: loads (the words made in registers), the drains' global adds (the lines folded into a sink), the image's stores
//: and what a row changes to find what the readout's steps wait behind: every wait SUSPENDED (`try_wait`) where the
//: others poll; the image STILL -- no fold warp runs, the readout reads it as it lies
//: and the row whose readout DRAINS ITS OWN rows: a readout warp loads its 16 tokens' lines back from the panel and
//: adds them itself, no drain warp and no panel barrier, and the fourth warpgroup is four producers, two a box
enum Roles : long long {
  kFold = 1, kRead = 2, kNoLoads = 4, kNoAdds = 8, kNoImage = 16, kSuspend = 32, kStill = 64, kSelf = 128, kLean = 256, kFive = 512, kSleep = 1024,
  //: two more accesses a row takes out: the producers' coefficient arithmetic (the bytes stored as they are) and
  //: the readout's loads of its readers' bytes (the words made in registers)
  //: (with the whole-entry row's bits `kNoMath` is no row of its own: it names THE PACKED row, which does its arithmetic)
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
  //: THE DEN'S OTHER PARTS, each a row's own so a chain of rows prices them one at a time, each needing the one
  //: before it: the image's ninth brick and its store; the register split (readout 88, fourth warpgroup 104); the
  //: readout's 72-column steps; the panel's word and the drain's add
  kDenImage = 8388608, kDenSplit = 16777216, kDenRead = 33554432, kDenDrain = 67108864,
  kDen = kDenFold | kDenImage | kDenSplit | kDenRead | kDenDrain,
  //: the row that moves the image's tiles apart with nothing else changed: 154 lines where the tile is 137, and
  //: 161 where the den's is 155 -- what the tiles' pitch costs the fold's stores (an even pitch was measured
  //: slower); with no clock inside a role, 155 where the tile is 137: the den's own pitch without the den
  kPitch = 1048576,
  //: THE ALIGNED CELL: both sides list the same level, so a reader reads ONE list on the operand level's 16 digits
  //: -- an image tile is a LIST and its K the digits. A line (a channel's 8 digits of a list) is then four fold
  //: threads' values, a 4-byte word each: 16 word stores a thread a tile group where the other cell's are 4 lines;
  //: the den's line is 8 threads' values, 2 bytes each. The readout and the drains are the same code
  kAligned = 2097152,
  //: the rows with NO CLOCK INSIDE A ROLE: a wait is its wait and a phase is not lapped, so the window's cycles are
  //: the roles' own -- a role's clock reads compile differently from row to row (into registers in one kernel,
  //: through uniform registers in another), and these rows price a row's work without them
  kNoClock = 4194304,
  //: the row whose VALUES ARE TILE-MAJOR in global memory -- a tile of 64 tokens its 8 blocks of 8 channels one
  //: after another, a block a token's 16 bytes a line, the order a step's operand lands in -- so a token's 8 lines
  //: are a kilobyte apart where the others' are one run of 128 bytes: what that order costs the producers' loads
  kTiledValues = 134217728,
  //: THE LISTED row: the fold's entries come FROM LISTS -- a producer loads a batch's 32 entries' places in their
  //: window from a list buffer (2 bytes a place, as the facts pass's lists hold them; these places are hashed, not
  //: a mask's), a batch ahead of the loads they address;
  //: a window's tokens are its 1,024, their rows one stretch of each table; an entry's byte on its list's digit and
  //: its 16 dense bytes are read from the levels' tile-major rows, a block each
  kListed = 268435456,
  //: the listed row with EVERY WINDOW THE SAME 1,024 TOKENS: its rows stay in cache, where the listed row's are
  //: new each window -- what of the listed row is its loads' code and what its rows' arrival
  kHot = 536870912,
  //: the listed row whose producers ASK FOR A WINDOW'S ROWS A WINDOW AHEAD: for the batch a window on, a prefetch
  //: of each entry's values (4 lines, 32 bytes apart), its dense bytes and its sparse word, from that batch's
  //: places, themselves loaded a batch before
  kAhead = 1073741824,
  //: THE EXPANDED row (the listed row with its values elsewhere): an entry's VALUES ARE NOT GATHERED BY ITS CTA.
  //: A pass before the carry has written each sparse digit's entries' values a window, contiguous, in the layout a
  //: step reads -- once a token and sparse digit, shared by the 16 boxes of that digit's column -- and a producer
  //: lands a list's value tile with ONE BULK COPY; it still makes the list's coefficients from the lists and the
  //: levels' rows
  kExpanded = kListed | kTiledValues,
  //: THE WHOLE-ENTRY row: the pass has written an entry's COEFFICIENT BYTES beside its values too -- a list's ninth
  //: brick, a line an entry's 16 dense bytes and the 17th its 16 entries' sparse bytes -- so a list is one bulk copy
  //: and its producer loads nothing from global memory: it expands the landed bytes where they lie
  kWhole = kExpanded | kAhead,
  //: the whole-entry row with EACH BOX ITS OWN expanded windows, 660 MB of them: every list a copy of bytes no
  //: other box read and the cache does not hold, where the whole-entry row's 16 columns are 40 MB read by 16 boxes
  //: each -- what of that row is its bytes' being in the cache
  kApart = kWhole | kHot,
  //: the whole-entry row with its 16 columns' windows NEVER READ BEFORE: a launch runs 128 windows of its own
  //: addresses, 80 MB, so a window's bytes are read by its 16 boxes at once and are not in the cache when the
  //: first asks -- what a stream written up front for a segment is
  kFresh = kWhole | kNoLoads,
  //: THE PACKED row: the whole-entry row fed THE STREAM AS RULED, its lists UNEVEN -- a column's entries of a window
  //: laid as its lists' steps, a list's full steps tiles as the whole-entry row's and its last, part-filled step stored
  //: TRUNCATED, each brick only the entries it has, then their dense lines and a line of their sparse bytes. A
  //: step is one bulk copy of its own size; a truncated step is used where it lands, its operand's bricks as far
  //: apart as it has entries, its pads' coefficient lines zero. A window's steps are taken a round at a time, 8 a
  //: slot, and the fold runs a slot's by its header: each list's steps in it
  kPacked = kWhole | kNoMath,
  //: the packed row with FOUR SLOTS OF 4 TILES a box where it has two of 8: the same pool, three slots in flight
  //: behind the one its fold runs where there is one
  kFour = 1ll << 31,
  //: the fold-expands row with THE FOLD ISSUING ITS OWN SLOT'S COPIES, the window after's as it leaves the slot: no
  //: producer warp runs, and no hand-off of a freed slot
  kFoldIssues = 1ll << 32,
  //: the serve warpgroup FOUR DRAIN WARPS (a row with no producer): 16 of a chunk's 64 tokens each
  kDrains4 = 1ll << 33,
  //: THE READOUT A SUM A BOX: a chunk's steps on box 0's tiles add into one accumulator and those on box 1's into
  //: another -- two chains of 4 steps where one accumulator makes a chain of 8 -- summed a thread before the row is
  //: rounded; 32 registers a readout thread more, the serve warpgroup's. Its A is ONE set: a digit's pair of steps
  //: is done before the next pair's is written -- a chain's steps follow one another whatever is in flight
  kTwoSums = 1ll << 34
};
template <long long ROLES>
constexpr bool apart = (ROLES & kApart) == kApart;
template <long long ROLES>
constexpr bool fresh = (ROLES & kFresh) == kFresh;
template <long long ROLES>
constexpr bool packed = (ROLES & kPacked) == kPacked;
template <long long ROLES>
constexpr bool four = packed<ROLES> && (ROLES & kFour) != 0;
//: THE FOLD EXPANDS (the whole-entry row with `kFour`'s bit, not the packed row): a producer only issues a slot's
//: copies; the slot's fold warpgroup waits on their bytes and expands them itself, a thread an entry -- its 128
//: threads the slot's 128 entries at once, where one producer warp takes four trips -- then runs its steps
template <long long ROLES>
constexpr bool fold_expands = (ROLES & kWhole) == kWhole && !packed<ROLES> && (ROLES & kFour) != 0;
template <long long ROLES>
constexpr bool fold_issues = fold_expands<ROLES> && (ROLES & kFoldIssues) != 0;
//: the packed rows' pool: a slot's tiles, and a box's slots
template <long long ROLES>
constexpr int slot_tiles = four<ROLES> ? 4 : 8;
template <long long ROLES>
constexpr int box_slots = 16 / slot_tiles<ROLES>;
//: the slots a packed producer issues copies ahead of the one it expands: with four slots a box, two -- the fold's,
//: the producer's and two landing; with two, none
template <long long ROLES>
constexpr int slots_ahead = box_slots<ROLES> - 2;
//: a packed slot's barrier of `kind` (0 filled, 1 freed, 2 landed), `bars` the barriers' base
template <long long ROLES>
__device__ __forceinline__ uint32_t slot_bar(uint32_t bars, int kind, int box, uint32_t q) {
  if constexpr (four<ROLES>) return bars - kBarsAt + kFourBarsAt + 8 * (8 * kind + 4 * box + q);
  else return bars + (kind == 0 ? kPoolReady : kind == 1 ? kPoolFreed : kPoolLanded) + 8 * (2 * box + q);
}
//: an image tile's pitch in bytes: the stores of a warp's lanes are two tiles apart
template <long long ROLES>
constexpr int tile_of = 16 * (ROLES & kDenImage ? (ROLES & kPitch ? 161 : 155) : ROLES & kPitch ? (ROLES & kNoClock ? 155 : 154) : 137);
//: a fold thread's state values: its 16 accumulator sets, and the den's 4
template <long long ROLES>
constexpr int state_words = kLists * 8 + (ROLES & kDenFold ? 4 : 0);
//: a row in the replicated tables: the token's own, or the replica of it the access's position picks
template <long long ROLES>
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
template <long long ROLES>
constexpr int threads_of = ROLES & kFive ? 5 * kGroup : kThreads;
template <long long ROLES>
constexpr int drains_of = ROLES & (kFive | kDrains4) ? 4 : kDrains;
template <long long ROLES>
constexpr int drain_at = ROLES & kFive ? 4 * kGroup : ROLES & kDrains4 ? kProduceAt : kDrainAt;
//: the producers a box
template <long long ROLES>
constexpr int producers_of = ROLES & (kSelf | kFive) ? 2 : 1;
//: the parts a producer's batch of values is loaded and stored in: a part in registers at a time
template <long long ROLES>
constexpr int value_parts = ROLES & kFive ? kFiveParts : 1;
//: THE LEAN READOUT's row: a chunk's digits take turns in TWO sets of A registers, a digit's two steps a commit
//: group, a set written again once its group is done -- 16 words of A where every step's own are 32 -- and the 16
//: registers a thread it gives up are the fourth warpgroup's, whose producers then hold two batches (at 72 the
//: toolkit serializes the steps, C7512)
//: the den's readout holds 4 more accumulators, column 64's set: at 80 the toolkit serializes its steps, and its 8
//: more a thread are the fourth warpgroup's, as the wide and the big rows' are
template <long long ROLES>
constexpr int read_regs = (ROLES & kFive ? kFiveReadRegs : (ROLES & kLean) && (ROLES & (kDenSplit | kWide)) ? 88 : ROLES & kLean ? 80 : kReadRegs)
                          + (ROLES & kTwoSums ? 32 : 0);
template <long long ROLES>
constexpr int serve_regs = ((ROLES & kLean) && (ROLES & (kWide | kDenSplit)) ? 104 : ROLES & kLean ? 112 : kServeRegs)
                           - (ROLES & kTwoSums ? 32 : 0)
                           - ((ROLES & kWhole) == kWhole && (ROLES & kNoMath) == 0 && (ROLES & kFour) && (ROLES & kFoldIssues) ? 16 : 0);
//: a fold that issues its own copies holds their addresses beside its state: 8 registers a thread more, the serve
//: warpgroup's, whose warps hold no entry
template <long long ROLES>
constexpr int fold_regs = (ROLES & kWhole) == kWhole && (ROLES & kNoMath) == 0 && (ROLES & kFour) && (ROLES & kFoldIssues) ? kFoldRegs + 8 : kFoldRegs;
static_assert(read_regs<kLean | kWide> + serve_regs<kLean | kWide> == kReadRegs + kServeRegs
              && read_regs<kLean | kBig> + serve_regs<kLean | kBig> == kReadRegs + kServeRegs);
static_assert(read_regs<kLean | kDen> + serve_regs<kLean | kDen> == kReadRegs + kServeRegs);
static_assert(read_regs<kLean> + serve_regs<kLean> == kReadRegs + kServeRegs);
template <long long ROLES>
//: the listed row's producers hold one batch: its places and its sparse words are 4 registers a thread two batches
//: do not leave
constexpr int batches_held = (ROLES & kLean) && !(ROLES & kListed) ? 2 : 1;

//: the token of a (CTA, box, entry of the call), its value at a channel, its byte on its list's digit and on a
//: dense leaf; a read coefficient at (read digit, fold digit, token): small integers
constexpr uint32_t token_of(uint32_t source, uint32_t entry) {
  return ((source * 4099u + entry) * 2654435761u >> 12) & (kTableTokens - 1);
}
//: THE LISTED row's tables behind the dense pointer: the readers' table as every row has it, then the dense
//: level's and the sparse level's rows, tile-major, 65,536 tokens each, then the lists -- a source's 64 windows of
//: 256 entries, 2 bytes an entry
constexpr int kLevelDigits = 256, kLevelTile = 64, kCallWindows = 64, kWindowTokens = 1024;
constexpr long long kLevelBytes = static_cast<long long>(kTableTokens) * kLevelDigits,
                    kLevelsAt = static_cast<long long>(kReplicas) * kTableTokens * 16, kListsAt = kLevelsAt + 2 * kLevelBytes;
constexpr int kListEntries = kCallWindows * kWindow;
//: a listed entry's token: its window's first and the place the list holds
constexpr uint32_t listed_place(uint32_t source, uint32_t entry) {
  return ((source * 4099u + entry % kListEntries) * 2654435761u >> 12) & (kWindowTokens - 1);
}
//: the expanded row's lists and values are a sparse column's, 16 columns, a source its own
constexpr int kColumns = 16;
//: the windows of its own addresses a column's expanded buffer holds
template <long long ROLES>
constexpr int expanded_windows = fresh<ROLES> ? 2 * kCallWindows : kCallWindows;
//: an expanded list: its value tile and a brick of its coefficients' bytes
constexpr int kExpandedList = kValueTile + kBrick;
constexpr long long kExpandedWindow = static_cast<long long>(kLists) * kExpandedList;
//: THE PACKED row's buffer: a head a (column, window) -- its slots' count, then a header a slot (8 at most: 32
//: steps in the four-slot row's slots of 4), then a record a step: the step's first line in the window's stream, and its entries -- and then
//: the streams, a (column, window) room for 32 full steps. A SLOT is the window's next 8 steps taken A ROUND AT A
//: TIME (every list's first step, then every list's second), laid in the order of their lists
constexpr int kPackedSteps = 2 * kLists, kPackedSlots = 8, kPackedHeadersAt = 16,
              kPackedRecordsAt = kPackedHeadersAt + kPackedSlots * kHeader, kPackedHead = kPackedRecordsAt + 8 * kPackedSteps,
              kPackedRoom = kPackedSteps * kExpandedList;
constexpr long long kPackedTable = static_cast<long long>(kColumns) * kCallWindows * kPackedHead;
//: a list's entries in a window, 8 to 24 and a window's 256: lists `2 p` and `2 p + 1` are 16 more and less the same
constexpr int packed_count(uint32_t column, uint32_t window, int list) {
  const int by = static_cast<int>(((column * 131u + window * 31u + static_cast<uint32_t>(list >> 1)) * 2654435761u >> 20) % 9u);
  return list & 1 ? kEntries - by : kEntries + by;
}
//: the lines a step of `n` entries is stored in: a whole-entry tile, or 8 bricks of `n` lines, `n` dense lines and
//: the sparse bytes' line
constexpr int packed_lines(int n) { return n == 0 ? 0 : n == kEntries ? kExpandedList / 16 : 9 * n + 1; }
constexpr uint32_t listed_token(uint32_t source, uint32_t entry, bool hot) {
  return (hot ? 0u : (entry / kWindow % kCallWindows) * kWindowTokens) + listed_place(source, entry);
}
//: a token's 16 bytes of block `block` of a tile-major level, in lines of 16 bytes
constexpr long long level_line(uint32_t token, int block) {
  return static_cast<long long>(token / kLevelTile) * (kLevelDigits * kLevelTile / 16) + block * kLevelTile + token % kLevelTile;
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
template <long long ROLES>
__device__ __forceinline__ void await(uint32_t bar, uint32_t parity) {
  if constexpr (ROLES & kSuspend) lab::mbar_wait_suspended(bar, parity);
  else lab::mbar_wait(bar, parity);
}
//: the same, its cycles added to the warp's count for that barrier: a role's time is its span less its waits
template <long long ROLES>
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
template <long long ROLES>
__device__ __forceinline__ void fold_role(uint8_t* smem, uint32_t base, long long* times, uint32_t* waits,
                                          float* state, const uint32_t* values, int windows) {
  static_assert(!(ROLES & kDrains4) || fold_issues<ROLES> || !(ROLES & kFold));
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
  //: lane stores, the lanes that hold the group's den values to the den's line and the others to ONE place a warp,
  //: line 1 of a ninth brick (docs/lab/carry_window.md, THE DEN'S PARTS)
  uint32_t den_at[2] = {0u, 0u};
  if constexpr (ROLES & kDenImage) {
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      if constexpr (ROLES & kAligned) {
        //: kind `p` is the group's list half: the holders are the threads `t == p`, lists `8 t + 2 warp` on
        const bool holds = t == p;
        den_at[p] = lab::held(static_cast<uint32_t>(
            kImageAt + box * kBoxBytes + (8 * (holds ? t : p) + 2 * warp) * tile_of<ROLES>
            + 16 * ((kImageBricks - 1) * kBrickLines + (holds ? 0 : 1)) + 2 * q));
      } else {
        //: kind `p` is the group's digit parity: the holders are the threads `t < 2` whose `q` has it
        const bool holds = t < 2 && (q & 1) == p;
        den_at[p] = lab::held(static_cast<uint32_t>(
            kImageAt + box * kBoxBytes + (holds ? q : p) * tile_of<ROLES>
            + 16 * ((kImageBricks - 1) * kBrickLines + (holds ? 8 * t : 1)) + 4 * warp));
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
  //: THE FOLD-ISSUES row's copies of slot `slot` of window `w`: the warpgroup's first 8 threads a tile each, its
  //: first the bytes' statement
  const auto issue = [&](int w, int slot) {
    const uint32_t source = blockIdx.x * 2 + (threadIdx.x >> 7), key = apart<ROLES> ? source : source % kColumns;
    const int g = threadIdx.x & (kGroup - 1), at = 2 * (threadIdx.x >> 7) + slot;
    const uint32_t landed = base + kBarsAt + kPoolLanded + 8 * at;
    const uint8_t* const from = reinterpret_cast<const uint8_t*>(values)
                                + (static_cast<long long>(key) * expanded_windows<ROLES> + w % expanded_windows<ROLES>) * kExpandedWindow
                                + (slot * kSlotLists + g) * kExpandedList;
    if (g == 0) lab::mbar_expect(landed, kSlotLists * kExpandedList);
    if (g < kSlotLists) lab::bulk_load(base + at * kPoolBytes + g * kTile, from, kExpandedList, landed);
  };
  uint32_t on_read = 0, on_fill = 0;
  //: the packed row's: the slots its steps have used, a slot and a phase each
  uint32_t uses = 0;
  lab::WarpSpan clock;
  //: where the readout keeps a sum a box every role marks its clock in global memory: a held start spills
  if constexpr (ROLES & kTwoSums) lab::WarpMark::start(times);
  else clock.start();
  if constexpr (fold_issues<ROLES>) {
    issue(0, 0);
    issue(0, 1);
  }
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
                const float value = den[hc | (k & 1) << 1];
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
              const uint32_t word = lab::bf16x2_trunc(den[k >> 1], den[(k >> 1) | 2]);
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
    if constexpr (packed<ROLES>) {
      //: THE PACKED row's slots, a window as many as its steps need: a slot's header says each list's steps in it
      //: (its tiles in the lists' order) and which list's last is truncated -- that step's bricks as far apart as
      //: it has entries. A list's steps are a loop of its own, bottom-tested on a vote; no slot ends inside one
      uint32_t seen = 0;
      bool more;
#pragma unroll 1
      do {
        const uint32_t q = uses % box_slots<ROLES>, use = uses / box_slots<ROLES>;
        waited<ROLES>(slot_bar<ROLES>(bars, 0, box, q), use & 1, on_fill);
        const uint64_t* const header = reinterpret_cast<const uint64_t*>(smem + kHeadersAt + (4 * box + q) * kHeader);
        const uint64_t steps = header[0], cut_at = header[1];
        more = header[2] == 0;
        uint32_t tile = lab::held(base + 2 * box * kPoolBytes + q * slot_tiles<ROLES> * kTile);
        lab::wg_fence();
#pragma unroll
        for (int a = 0; a < kLists; ++a) {
          int left = static_cast<int>(steps >> (4 * a) & 15u);
          const uint32_t cut = static_cast<uint32_t>(cut_at >> (4 * a) & 15u);
          if (__any_sync(0xffffffffu, left > 0)) {
#pragma unroll 1
            do {
              //: the list's last step of the slot is the truncated one, where the header names one
              const uint32_t apart = left == 1 && cut ? cut : 17u;
              lab::wg_bf16_n16_ss_mn(d[a], lab::wg_desc(tile, 8, apart), lab::wg_desc(tile + kValueTile, 8, 17),
                                     w != 0 || (seen >> a & 1u));
              seen |= 1u << a;
              tile += kTile;
              --left;
            } while (__any_sync(0xffffffffu, left > 0));
          }
        }
        lab::wg_commit();
        lab::wg_wait<0>();
        if ((tid & (kGroup - 1)) == 0) lab::mbar_arrive(slot_bar<ROLES>(bars, 1, box, q));
        ++uses;
      } while (__any_sync(0xffffffffu, more));
    } else if constexpr (ROLES & kFold) {
#pragma unroll
      for (int slot = 0; slot < 2; ++slot) {
        if constexpr (fold_expands<ROLES>) {
          //: THE SLOT'S BYTES, then ITS COEFFICIENTS by its own warpgroup: thread `16 j + k` entry `k` of tile `j` -- its
          //: landed dense line by its sparse byte, to its two coefficient lines -- and the warpgroup together before
          //: its steps read them
          waited<ROLES>(bars + kPoolLanded + 8 * (2 * box + slot), w & 1, on_fill);
          const int j = (tid & (kGroup - 1)) >> 4, k = tid & 15;
          uint8_t* const raw = smem + (2 * box + slot) * kPoolBytes + j * kTile + kValueTile;
          const uint4 landed = *reinterpret_cast<const uint4*>(raw + k * 16);
          const float weight = static_cast<float>(raw[16 * kEntries + k]);
          const uint32_t word[4] = {landed.x, landed.y, landed.z, landed.w};
          //: a line at a time, stored before the next is made: a fold thread has few registers past its state
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            uint32_t pair[4];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const uint32_t two = word[2 * h + c / 2] >> (16 * (c & 1));
              pair[c] = lab::bf16x2_round(weight * static_cast<float>(two & 255u), weight * static_cast<float>(two >> 8 & 255u));
            }
            *reinterpret_cast<uint4*>(raw + k * 16 + h * kBrick) = make_uint4(pair[0], pair[1], pair[2], pair[3]);
          }
          lab::bulk_fence();
          lab::bar_sync(kPoolId + box, kGroup);
        } else if constexpr (ROLES & kSleep) slept(kPoolId + 2 * box + slot, kPoolCount, on_fill);
        else waited<ROLES>(bars + kPoolReady + 8 * (2 * box + slot), w & 1, on_fill);
        //: the expanded row's values are the bulk copies': their bytes landed
        if constexpr ((ROLES & kWhole) == kExpanded) await<ROLES>(bars + kPoolLanded + 8 * (2 * box + slot), w & 1);
        const uint32_t tiles = lab::held(base + (2 * box + slot) * kPoolBytes);
        lab::wg_fence();
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
        if constexpr (fold_issues<ROLES>) {
          if (w + 1 < windows) issue(w + 1, slot);
        } else {
          if ((tid & (kGroup - 1)) == 0) lab::mbar_arrive(bars + kPoolFreed + 8 * (2 * box + slot));
        }
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
  if constexpr (ROLES & kTwoSums) lab::WarpMark::stop(times);
  else clock.stop(times);
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

//: THE PACKED row's PRODUCER WARP, one a box. A window's step records and slot headers are loaded A WINDOW AHEAD,
//: a lane a record. A slot's copies -- a lane a step, one bulk copy of its own size to its tile -- are issued
//: `slots_ahead` slots before the slot is expanded (`kOn`): the four-slot row's two ahead, so a slot's bytes land behind two
//: slots' expanding. A slot's trip: the copies of the slot that far on; its own header to the fold; the wait on its
//: bytes; its entries, a lane an entry -- its landed dense line by its sparse byte, to its two coefficient lines; a
//: pad's lines zero; and of a truncated step, every line its operand's bricks reach past its entries -- its landed
//: dense lines and sparse bytes, and what no copy of it wrote -- zeroed behind the read
template <long long ROLES>
__device__ __forceinline__ void produce_packed(uint8_t* smem, uint32_t base, long long* times, const uint32_t* values,
                                               int windows) {
  constexpr int kT = slot_tiles<ROLES>, kS = box_slots<ROLES>, kOn = slots_ahead<ROLES>;
  //: TWO PRODUCERS A BOX (the four-slot row with the serve warps all producing): the call's even slots are one's
  //: and its odd ones the other's, so the slot two on is each one's own next
  constexpr int kPer = producers_of<ROLES>;
  static_assert(kPer == 1 || kOn == kPer);
  const int lane = threadIdx.x & 31, mine = (threadIdx.x - kProduceAt) >> 5, box = mine / kPer, part = mine % kPer;
  const uint32_t bars = base + kBarsAt, key = (blockIdx.x * 2 + box) % kColumns;
  const uint8_t* const buffer = reinterpret_cast<const uint8_t*>(values);
  const auto head_of = [&](int w) { return buffer + (static_cast<long long>(key) * kCallWindows + w % kCallWindows) * kPackedHead; };
  const auto stream_of = [&](int w) {
    return buffer + kPackedTable + (static_cast<long long>(key) * kCallWindows + w % kCallWindows) * kPackedRoom;
  };
  //: a step's entries, the lane's of slot `j` of the window whose records are `of`: every lane takes the shuffle,
  //: and the lanes past the slot's tiles hold no step
  const auto entries_of = [&](const uint2& of, uint32_t j) {
    const uint32_t taken = __shfl_sync(0xffffffffu, of.y, (static_cast<uint32_t>(kT) * j + static_cast<uint32_t>(lane)) & 31u);
    return lane < kT ? taken : 0u;
  };
  //: THE COPIES of the call's slot `at`, slot `j` of the window whose records and stream these are: its pool slot
  //: freed by the fold's steps of the slot a ring before, a lane a step
  const auto issue = [&](const uint2& of, const uint8_t* stream, uint32_t j, uint32_t at) {
    const uint32_t q = at % kS, use = at / kS;
    if (use > 0) await<ROLES>(slot_bar<ROLES>(bars, 1, box, q), (use - 1) & 1);
    const uint32_t first = __shfl_sync(0xffffffffu, of.x, (static_cast<uint32_t>(kT) * j + static_cast<uint32_t>(lane)) & 31u);
    const uint32_t n = entries_of(of, j);
    const uint32_t bytes = static_cast<uint32_t>(packed_lines(static_cast<int>(n))) * 16u;
    const uint32_t all = __reduce_add_sync(0xffffffffu, bytes);
    const uint32_t landed = slot_bar<ROLES>(bars, 2, box, q);
    if (lane == 0) lab::mbar_expect(landed, all);
    if (n) lab::bulk_load(base + 2 * box * kPoolBytes + (q * kT + lane) * kTile, stream + static_cast<long long>(first) * 16, bytes, landed);
  };
  //: the window ahead's: its slots' count, a lane its step's record, a lane its slot's header
  uint32_t slots_next = *reinterpret_cast<const uint32_t*>(head_of(0));
  uint2 record_next = *reinterpret_cast<const uint2*>(head_of(0) + kPackedRecordsAt + 8 * lane);
  uint4 header_next[2];
#pragma unroll
  for (int h = 0; h < 2; ++h)
    header_next[h] = *reinterpret_cast<const uint4*>(head_of(0) + kPackedHeadersAt + (lane & (kPackedSlots - 1)) * kHeader + 16 * h);
  uint32_t uses = 0;
  lab::WarpSpan clock;
  clock.start();
  //: the call's first slots' copies, as far ahead as every later slot's
#pragma unroll
  for (int a = 0; a < kOn; ++a)
    if (__any_sync(0xffffffffu, a % kPer == part)) issue(record_next, stream_of(0), a, a);
#pragma unroll 1
  for (int w = 0; w < windows; ++w) {
    const uint32_t slots = slots_next;
    const uint2 record = record_next;
    const uint4 header[2] = {header_next[0], header_next[1]};
    slots_next = *reinterpret_cast<const uint32_t*>(head_of(w + 1));
    record_next = *reinterpret_cast<const uint2*>(head_of(w + 1) + kPackedRecordsAt + 8 * lane);
#pragma unroll
    for (int h = 0; h < 2; ++h)
      header_next[h] = *reinterpret_cast<const uint4*>(head_of(w + 1) + kPackedHeadersAt + (lane & (kPackedSlots - 1)) * kHeader + 16 * h);
    uint32_t i = 0;
#pragma unroll 1
    do {
      //: a slot is one producer's: the other's trip is its counts alone
      if (__any_sync(0xffffffffu, static_cast<int>(uses % kPer) == part)) {
      //: THE ISSUE EDGE: the slot `kOn` on -- this window's, or one of the next's first; none past the call. One
      //: body: whose records and stream it reads is a select
      const bool here = i + kOn < slots;
      uint2 of;
      of.x = here ? record.x : record_next.x, of.y = here ? record.y : record_next.y;
      if (__any_sync(0xffffffffu, here || w + 1 < windows))
        issue(of, stream_of(here ? w : w + 1), here ? i + kOn : i + kOn - slots, uses + kOn);
      const uint32_t q = uses % kS, use = uses / kS;
      uint8_t* const tiles = smem + 2 * box * kPoolBytes + q * kT * kTile;
      const uint32_t n = entries_of(record, i);
      //: THE SLOT'S HEADER, one lane's store of each half: the lane that holds the slot's
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint4 mine = make_uint4(__shfl_sync(0xffffffffu, header[h].x, i), __shfl_sync(0xffffffffu, header[h].y, i),
                                      __shfl_sync(0xffffffffu, header[h].z, i), __shfl_sync(0xffffffffu, header[h].w, i));
        if (lane == h) *reinterpret_cast<uint4*>(smem + kHeadersAt + (4 * box + q) * kHeader + 16 * h) = mine;
      }
      await<ROLES>(slot_bar<ROLES>(bars, 2, box, q), use & 1);
#pragma unroll
      for (int o = 0; o < kT / 2; ++o) {
        //: the lane's entry `k` of the slot's tile `j`, a step of `nj` entries
        const int j = 2 * o + (lane >> 4), k = lane & 15;
        const uint32_t nj = __shfl_sync(0xffffffffu, n, j);
        uint8_t* const tile = tiles + j * kTile;
        uint8_t* const raw = tile + (nj == kEntries ? kValueTile : 8 * nj * 16);
        uint4 bytes4 = make_uint4(0, 0, 0, 0);
        float weight = 0.0f;
        if (static_cast<uint32_t>(k) < nj) {
          bytes4 = *reinterpret_cast<const uint4*>(raw + k * 16);
          weight = static_cast<float>(raw[nj * 16 + k]);
        }
        const uint32_t word[4] = {bytes4.x, bytes4.y, bytes4.z, bytes4.w};
        uint4 line[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          uint32_t pair[4];
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const uint32_t two = word[2 * h + c / 2] >> (16 * (c & 1));
            pair[c] = lab::bf16x2_round(weight * static_cast<float>(two & 255u), weight * static_cast<float>(two >> 8 & 255u));
          }
          line[h] = make_uint4(pair[0], pair[1], pair[2], pair[3]);
        }
        //: the half-warp's reads of the tile's landed lines are done before any lane's store over them
        __syncwarp();
        if (nj) {
          //: a truncated step's lines past its values: brick 7's 16 lines end at `7 nj + 15`, its landed bytes at `9 nj`
          const uint32_t reach = nj > 15u - nj ? nj : 15u - nj;
          if (nj < kEntries && static_cast<uint32_t>(k) <= reach) *reinterpret_cast<uint4*>(raw + k * 16) = make_uint4(0, 0, 0, 0);
          //: a slot's two lists a trip take opposite bricks first: each store covers the 32 banks evenly
          const int lead = j & 1;
          uint8_t* const to = tile + kValueTile + k * 16;
          *reinterpret_cast<uint4*>(to + lead * kBrick) = lead ? line[1] : line[0];
          *reinterpret_cast<uint4*>(to + (lead ^ 1) * kBrick) = lead ? line[0] : line[1];
        }
      }
      lab::bulk_fence();
      __syncwarp();
      if (lane == 0) lab::mbar_arrive(slot_bar<ROLES>(bars, 0, box, q));
      }
      ++uses;
      ++i;
    } while (__any_sync(0xffffffffu, i < slots));
  }
  clock.stop(times);
}

//: A PRODUCER WARP, one or two a box: its share of a slot's four batches of 32 entries, one batch in registers at a
//: time or, in the lean row, two
template <long long ROLES>
__device__ __forceinline__ void produce_role(uint8_t* smem, uint32_t base, long long* times, const uint32_t* values,
                                             const uint32_t* dense, int windows) {
  if constexpr (packed<ROLES>) {
    produce_packed<ROLES>(smem, base, times, values, windows);
    return;
  }
  constexpr int kPer = producers_of<ROLES>;
  const int lane = threadIdx.x & 31, mine = (threadIdx.x - kProduceAt) >> 5, box = mine / kPer, part = mine % kPer;
  const int group = lane >> 3;
  const uint32_t bars = base + kBarsAt, source = blockIdx.x * 2 + box;
  constexpr int kQuads = kFill / 4, kBatches = kSlot / kFill, kOwn = kBatches / kPer;
  //: the warp's batch `g` of the call: its `g % kOwn`-th of slot `g / kOwn`, the slot's batch `part + kPer` of that
  const auto batch_of = [&](int g) { return (g / kOwn) * kBatches + part + kPer * (g % kOwn); };
  const uint4* const wide = reinterpret_cast<const uint4*>(values);
  //: the entries' dense bytes: the readers' table, or in the listed row the dense level's rows behind it
  const uint4* const boxes = reinterpret_cast<const uint4*>(reinterpret_cast<const uint8_t*>(dense)
                                                            + (ROLES & kListed ? kLevelsAt : 0));
  //: `kHeld` batches in registers, a batch a half: 96 registers a thread hold one, the lean row's 112 two. A
  //: batch's values are `kParts` parts of `kPart` instructions, a part in registers at a time
  constexpr int kHeld = batches_held<ROLES>, kParts = value_parts<ROLES>, kPart = kQuads / kParts;
  static_assert(kOwn % kHeld == 0);
  uint4 value[kHeld][kPart], bytes[kHeld];
  //: THE LISTED row's: the lane's entry's place in its window for the batch a half loads next, and the word of
  //: its sparse row that holds its list's byte
  uint32_t ahead[kHeld], sword[kHeld];
  //: the ahead row's: the lane's entry's place for the batch a window on
  uint32_t far[kHeld];
  static_assert(!(ROLES & kListed) || (kParts == 1 && !(ROLES & kBig) && (!(ROLES & kNoLoads) || fresh<ROLES>)));
  static_assert(!(ROLES & kTiledValues) || !(ROLES & kListed) || !(ROLES & kHot) || apart<ROLES>);
  constexpr bool kLanded = (ROLES & kExpanded) == kExpanded, kRaw = (ROLES & kWhole) == kWhole;
  //: the row that asks a window ahead, not the whole-entry row that shares its bit
  constexpr bool kAsks = (ROLES & kAhead) && !kLanded;
  //: the lists' and the expanded values' key: the source, or its sparse column
  const uint32_t key = kLanded && !apart<ROLES> ? source % kColumns : source;
  const uint16_t* const lists = reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(dense) + kListsAt)
                                + static_cast<long long>(key) * kListEntries;
  const auto load_places = [&](int n, int h) {
    ahead[h] = lists[(n * kFill + lane) % kListEntries];
  };
  //: THE AHEAD row's asks: batch `n`'s entries' rows, a lane an entry, from the places `far` holds
  const auto ask = [&](int n, int h) {
    const uint32_t tb = (ROLES & kHot ? 0u : static_cast<uint32_t>(n / (kWindow / kFill) % kCallWindows) * kWindowTokens) + far[h];
    const uint8_t* const row = reinterpret_cast<const uint8_t*>(values) + static_cast<long long>(tb) * (kRowWords * 4);
#pragma unroll
    for (int i = 0; i < 4; ++i) lab::touch(row + 32 * i);
    lab::touch(boxes + level_line(tb, static_cast<int>(source & 15)));
    lab::touch(reinterpret_cast<const uint32_t*>(boxes) + kLevelBytes / 4 + level_line(tb, static_cast<int>(source >> 4 & 15)) * 4);
  };
  //: batch `n`'s window's first token, and the lane's entry's list in its box
  const auto first_of = [&](int n) {
    return ROLES & kHot ? 0u : static_cast<uint32_t>(n / (kWindow / kFill) % kCallWindows) * kWindowTokens;
  };
  const auto list_of = [&](int n) { return (n / kBatches & 1) * kSlotLists + (n % kBatches * kFill + lane) / kEntries; };
  //: part `part_at` of batch `n` of the call into register half `h`: value instructions of four entries (8 lanes
  //: an entry, a lane 16 bytes)
  const auto load_values = [&](int n, int part_at, int h) {
    if constexpr (kLanded) return;
#pragma unroll
    for (int qd = 0; qd < kPart; ++qd) {
      uint32_t tk;
      //: the entry's place is its lane's: the lanes of a value instruction take their entry's by a shuffle
      if constexpr (ROLES & kListed) tk = first_of(n) + __shfl_sync(0xffffffffu, ahead[h], 4 * qd + group);
      else tk = token_of(source, n * kFill + 4 * (part_at * kPart + qd) + group);
      if constexpr (ROLES & kNoLoads) value[h][qd] = make_uint4(tk, tk, tk, tk);
      else {
        const uint32_t row = row_of<ROLES>(tk, static_cast<uint32_t>(n * kFill + 4 * qd + group));
        //: tile-major, line `p` of token `x` is 16 bytes at `512 (x / 64) + 64 p + x % 64`: the row and 448 a tile
        if constexpr (ROLES & kTiledValues) value[h][qd] = wide[row + (row >> 6) * 448u + static_cast<uint32_t>(lane & 7) * 64u];
        else value[h][qd] = wide[static_cast<long long>(row) * (kRowWords / 4) + (lane & 7)];
      }
    }
  };
  //: batch `n`'s dense bytes, all 32 entries' in one instruction (a lane an entry's 16)
  const auto load_bytes = [&](int n, int h) {
    if constexpr (ROLES & kListed) {
      //: the entry's 16 dense bytes, a block of the dense level's row, and the word of its sparse row's block that
      //: holds its list's byte
      const uint32_t tb = first_of(n) + ahead[h];
      bytes[h] = boxes[level_line(tb, static_cast<int>(source & 15))];
      sword[h] = (reinterpret_cast<const uint32_t*>(boxes) + kLevelBytes / 4)[
          level_line(tb, static_cast<int>(source >> 4 & 15)) * 4 + (list_of(n) >> 2)];
    } else {
      const uint32_t tb = token_of(source, n * kFill + lane);
      if constexpr (ROLES & kNoLoads) bytes[h] = make_uint4(tb, tb, tb, tb);
      else bytes[h] = boxes[row_of<ROLES>(tb, static_cast<uint32_t>(n * kFill + lane) * 2u + 1u)];
    }
  };
#pragma unroll
  for (int h = 0; h < kHeld; ++h) {
    if constexpr ((ROLES & kListed) && !kRaw) load_places(batch_of(h), h);
    load_values(batch_of(h), 0, h);
    if constexpr (!kRaw) load_bytes(batch_of(h), h);
    //: the places of the batch a half loads next, a batch ahead of the loads they address
    if constexpr ((ROLES & kListed) && !kRaw) load_places(batch_of(h + kHeld), h);
    if constexpr (kAsks) far[h] = lists[(batch_of(h + 2 * kOwn) * kFill + lane) % kListEntries];
  }
  lab::WarpSpan clock;
  //: a serve warp with few registers marks its clock in global memory: a held start is the value that spills
  if constexpr (ROLES & (kFive | kTwoSums)) lab::WarpMark::start(times);
  else clock.start();
#pragma unroll 1
  for (int w = 0; w < windows; ++w) {
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
      if (w > 0) await<ROLES>(bars + kPoolFreed + 8 * (2 * box + slot), (w - 1) & 1);
      uint8_t* const tiles = smem + (2 * box + slot) * kPoolBytes;
      if constexpr (kLanded) {
        //: THE SLOT'S VALUES, 8 lists' tiles, a bulk copy each from the column's expanded window: a lane a list
        const uint32_t landed = bars + kPoolLanded + 8 * (2 * box + slot);
        constexpr int kBytes = kRaw ? kExpandedList : kValueTile;
        if (lane == 0) lab::mbar_expect(landed, kSlotLists * kBytes);
        const uint8_t* const from = reinterpret_cast<const uint8_t*>(values)
                                    + (static_cast<long long>(key) * expanded_windows<ROLES> + w % expanded_windows<ROLES>) * kExpandedWindow
                                    + (slot * kSlotLists + lane) * kExpandedList;
        if (lane < kSlotLists) lab::bulk_load(base + (2 * box + slot) * kPoolBytes + lane * kTile, from, kBytes, landed);
        //: the whole-entry row's producer reads what landed: it waits for the bytes itself
        if constexpr (kRaw && !fold_expands<ROLES>) await<ROLES>(landed, w & 1);
      }
      //: where the fold expands, a producer's slot is its copies alone
      if constexpr (!fold_expands<ROLES>) {
      //: a rolled loop: a batch's addresses are formed in its trip, none held for the slot
#pragma unroll 1
      for (int o0 = 0; o0 < kOwn; o0 += kHeld)
#pragma unroll
       for (int h = 0; h < kHeld; ++h) {
        const int o = o0 + h, b = part + kPer * o, g = (2 * w + slot) * kOwn + o;
        //: a part's stores, then the same registers' loads for the next part: the batch's own, or the first of
        //: the batch `kHeld` on
        const auto part_out = [&](int part_at) {
          if constexpr (!kLanded) {
#pragma unroll
            for (int qd = 0; qd < kPart; ++qd) {
              //: entry `e` of the slot: list `e / 16`, token `e % 16` of its step
              const int e = b * kFill + 4 * (part_at * kPart + qd) + group;
              *reinterpret_cast<uint4*>(tiles + (e / kEntries) * kTile + (lane & 7) * kBrick + (e % kEntries) * 16) =
                  value[h][qd];
            }
          }
          const bool last = part_at + 1 == kParts;
          load_values(batch_of(last ? g + kHeld : g), last ? 0 : part_at + 1, h);
        };
        part_out(0);
        //: the lane's entry: its 16 bytes times the entry's, 8 pairs of bf16, a brick's line each four
        const int e = b * kFill + lane;
        //: the entry's byte on its list's digit: the listed row's from its row, a byte of the word loaded
        float weight;
        if constexpr (kRaw) {
          //: the entry's landed line and its sparse byte, read where the copy left them and overwritten below
          const uint8_t* const landed_at = tiles + (e / kEntries) * kTile + kValueTile;
          bytes[h] = *reinterpret_cast<const uint4*>(landed_at + (e % kEntries) * 16);
          weight = static_cast<float>(landed_at[16 * kEntries + e % kEntries]);
        } else if constexpr (ROLES & kListed) weight = static_cast<float>(sword[h] >> (8 * (((slot * kSlot + e) / kEntries) & 3)) & 255u);
        else weight = static_cast<float>(sparse_of(token_of(source, w * kWindow + slot * kSlot + e)));
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
        if constexpr (!kRaw) load_bytes(batch_of(g + kHeld), h);
        //: the places of the batch after that one, once this trip's loads have taken theirs
        if constexpr ((ROLES & kListed) && !kRaw) load_places(batch_of(g + 2 * kHeld), h);
        if constexpr (kAsks) {
          //: the batch a window on: its rows asked for with the places loaded a trip ago, then the next's places
          ask(batch_of(g + 2 * kOwn), h);
          far[h] = lists[(batch_of(g + 2 * kOwn + kHeld) * kFill + lane) % kListEntries];
        }
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
  }
  if constexpr (ROLES & (kFive | kTwoSums)) lab::WarpMark::stop(times);
  else clock.stop(times);
}

//: the readout's step: 64 tokens by the tile's 64 channels, or by its 72 columns where the den's is the 65th
template <long long ROLES, int N>
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
template <long long ROLES>
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
  static_assert(!(ROLES & kDenImage) || (ROLES & kDenFold));
  static_assert(!(ROLES & kDenDrain) || (ROLES & kDenRead));
  float d[ROLES & kDenRead ? 36 : 32];
  //: box 1's sum, where the row keeps a sum a box
  float d1[32];
  static_assert(!(ROLES & kTwoSums) || ((ROLES & kLean) && !(ROLES & (kDenRead | kFive))));
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
          if constexpr ((ROLES & kTwoSums) != 0) {
            if (j >= 1) lab::wg_wait<0>();
          } else {
            if (j >= 2) lab::wg_wait<1>();
          }
#pragma unroll
          for (int box = 0; box < 2; ++box)
#pragma unroll
            for (int i = 0; i < 4; ++i) rows[ROLES & kTwoSums ? 0 : j & 1][box][i] = lab::held(warp == j ? a[box][i] : 0u);
#pragma unroll
          for (int box = 0; box < 2; ++box)
            if constexpr ((ROLES & kTwoSums) != 0) {
              //: a box's own accumulator: its chain is its 4 digits' steps, the other box's beside it
              const uint64_t tile = lab::wg_desc(images + box * kBoxBytes + 2 * j * tile_of<ROLES>, 8, kBrickLines);
              if (box == 0) read_step<ROLES>(d, rows[0][0], tile, j != 0);
              else read_step<ROLES>(d1, rows[0][1], tile, j != 0);
            } else {
              read_step<ROLES>(d, rows[j & 1][box],
                               lab::wg_desc(images + box * kBoxBytes + 2 * j * tile_of<ROLES>, 8, kBrickLines),
                               j != 0 || box != 0);
            }
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
      //: the next chunk's bytes, issued under the steps -- or, where the readout keeps a sum a box, behind them: its
      //: two accumulators, its A and those bytes are more than a thread's registers at once
      if constexpr (!(ROLES & kTwoSums)) load(c + 1);
      lab::wg_wait<0>();
      if constexpr ((ROLES & kTwoSums) != 0) load(c + 1);
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
          if constexpr ((ROLES & kTwoSums) != 0)
            *reinterpret_cast<uint32_t*>(row + 8 * r * kRowBytes + 16 * j) =
                lab::pack_half2(d[4 * j + 2 * r] + d1[4 * j + 2 * r], d[4 * j + 2 * r + 1] + d1[4 * j + 2 * r + 1]);
          else
            *reinterpret_cast<uint32_t*>(row + 8 * r * kRowBytes + 16 * j) =
                lab::pack_half2(d[4 * j + 2 * r], d[4 * j + 2 * r + 1]);
      if constexpr (ROLES & kDenDrain) {
        //: THE DEN'S WORD of a row, as the step summed it: the threads that hold column 64
        if (t == 0) {
#pragma unroll
          for (int r = 0; r < 2; ++r)
            *reinterpret_cast<float*>(row + 8 * r * kRowBytes + den_word(16 * warp + 8 * r)) = d[32 + 2 * r];
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
template <long long ROLES>
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
  //: a serve warp with few registers marks its clock in global memory: a held start is the value that spills
  if constexpr (ROLES & (kFive | kTwoSums)) lab::WarpMark::start(times);
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
  } else if constexpr (ROLES & kTwoSums) {
    lab::WarpMark::stop(times);
    tell(waits, on_panel, 0u);
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

template <long long ROLES>
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
      lab::mbar_init(base + kBarsAt + kPoolLanded + 8 * p, 1);
    }
    if constexpr (four<ROLES>) {
#pragma unroll
      for (int i = 0; i < 3 * 8; ++i) lab::mbar_init(base + kFourBarsAt + 8 * i, 1);
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
    lab::regs_up<fold_regs<ROLES>>();
    if constexpr (ROLES & kStill) {
      idle_role(times, waits);
      //: present and idle until the readout is done
      if constexpr (ROLES & kPark) lab::bar_sync(kParkId, kParkCount);
      if constexpr (ROLES & kSpin) lab::mbar_wait(base + kBarsAt + kDone, 0);
    }
    else fold_role<ROLES>(smem, base, times, waits, state, values, windows);
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
    if (tid < (ROLES & kSelf ? kThreads : drain_at<ROLES>)) {
      //: where the fold issues its own copies no producer runs: its warps idle, or are drain warps
      if constexpr ((ROLES & kFold) && !fold_issues<ROLES>) produce_role<ROLES>(smem, base, times, values, dense, windows);
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

//: the value table laid tile-major, for the rows that read it so
static const uint32_t* tiled_values = nullptr;
//: the expanded row's values: a sparse column's lists' value tiles, a window after a window
static const uint32_t* expanded_values = nullptr;
//: the apart row's: a source its own expanded windows
static const uint32_t* apart_values = nullptr;
//: the fresh row's: the 16 columns' windows twice over, a launch's 128 its own addresses
static const uint32_t* fresh_values = nullptr;
//: the packed row's buffer: its heads, then its streams
static const uint32_t* packed_values = nullptr;
//: the four-slot row's: the same streams, their slots 4 steps
static const uint32_t* four_values = nullptr;
//: the listed row's tables: the two levels' tile-major rows and the lists
static const uint32_t* listed_tables = nullptr;

template <long long ROLES>
void run(lab::Bench& b, uint32_t* waits, const uint32_t* values, const uint32_t* dense, float* state, uint32_t* out) {
  using namespace carry_window;
  constexpr int kTimed = fresh<ROLES> ? 128 : 64;
  const std::string side = (ROLES & 3) == kFold ? "the fold's side alone: producers and steps"
                           : (ROLES & 3) == kRead ? "the read side alone: the image, the readout and the drain"
                           : "the carry: both sides";
  const std::string name = side + ((ROLES & kNoLoads) && !fresh<ROLES> ? ", the producers' global loads taken out" : "")
                           + (ROLES & kNoAdds ? ", the drains' global adds taken out" : "")
                           + (ROLES & kNoImage ? ", the image's stores taken out" : "")
                           + (ROLES & kStill ? ", no fold warp running" : "")
                           + (ROLES & kSuspend ? ", every wait suspended" : "")
                           + (ROLES & kSelf ? ", the readout draining its own rows and four producers" : "")
                           + (ROLES & kLean ? (ROLES & kListed ? ", the readout's A in two sets" : ", the readout's A in two sets and two batches a producer") : "")
                           + (ROLES & kFive ? ", five warpgroups: four producers and four drain warps" : "")
                           + (ROLES & kSleep ? ", the long waits asleep at named barriers" : "")
                           + ((ROLES & kNoMath) && !packed<ROLES> ? ", the producers' coefficient arithmetic taken out" : "")
                           + (ROLES & kNoBytes ? ", the readout's byte loads taken out" : "")
                           + (ROLES & kWide ? ", the readout's bytes one wide load a warp" : "")
                           + (ROLES & kNoFence ? ", the fold's fence behind its image taken out" : "")
                           + (ROLES & kPark ? ", the fold warps parked at a named barrier" : "")
                           + (ROLES & kSpin ? ", the fold warps spinning at a polled barrier" : "")
                           + (ROLES & kDump ? ", the fold's state stored out in every call" : "")
                           + (ROLES & kBig ? ", the tables 16 replicas the cache cannot hold" : "")
                           + (ROLES & kAligned ? ", the sides' carves equal: an image tile a list, its lines four threads' words" : "")
                           + (ROLES & kNoClock ? ", no clock inside a role" : "")
                           + ((ROLES & kExpanded) == kExpanded ? ", the values landed from a column's expanded lists" : ROLES & kTiledValues ? ", the values tile-major in global memory" : "")
                           + ((ROLES & kWhole) == kWhole ? ", the coefficients' bytes landed with them" : ROLES & kAhead ? ", a window's rows asked for a window ahead" : "")
                           + (apart<ROLES> ? ", each box its own expanded windows" : "")
                           + (fresh<ROLES> ? ", a column's windows never read before" : "")
                           + (packed<ROLES> ? ", a column's stream packed by digit, its lists uneven and their last steps truncated" : "")
                           + (four<ROLES> ? ", four slots of 4 tiles a box" : "")
                           + (fold_expands<ROLES> ? ", the fold warps expanding their own slot" : "")
                           + (fold_issues<ROLES> ? " and issuing its copies" : "")
                           + (ROLES & kDrains4 ? ", four drain warps" : "")
                           + (ROLES & kTwoSums ? ", the readout a sum a box" : "")
                           + (ROLES & kListed ? ((ROLES & kHot) && !apart<ROLES> ? ", the fold's entries from lists, every window the same tokens" : ", the fold's entries from lists, a window's own tokens") : "")
                           + (ROLES & kPitch ? (ROLES & kDenImage ? ", the image's tiles 161 lines apart" : ROLES & kNoClock ? ", the image's tiles 155 lines apart" : ", the image's tiles 154 lines apart") : "")
                           + ((ROLES & kDen) == kDen ? ", the den: two more steps a slot, the readout's steps 72 columns and an fp32 a token"
                              : ROLES & kDen ? std::string(", the den's parts alone: its fold steps")
                                                   + (ROLES & kDenImage ? ", its image brick" : "")
                                                   + (ROLES & kDenSplit ? ", its register split" : "")
                                                   + (ROLES & kDenRead ? ", its 72-column steps" : "")
                                             : std::string());
  constexpr int kLaunch = threads_of<ROLES>, kWarps = kLaunch >> 5;
  const auto kernel = window<ROLES>;
  //: the rows that read the values tile-major read the table laid so
  values = four<ROLES> ? four_values : packed<ROLES> ? packed_values : fresh<ROLES> ? fresh_values : apart<ROLES> ? apart_values : (ROLES & kExpanded) == kExpanded ? expanded_values : ROLES & kTiledValues ? tiled_values : values;
  dense = ROLES & kListed ? listed_tables : dense;
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  const size_t words = static_cast<size_t>(b.sms) * kOutWindows * kWindow * (kChannels / 2);
  const size_t tokens = static_cast<size_t>(b.sms) * kOutWindows * kWindow;
  LAB_CHECK(cudaMemset(out, 0, (words + tokens) * 4));
  if ((ROLES & ~(kSuspend | kSelf | kLean | kFive | kSleep | kWide | kDump | kBig | kDen | kPitch | kAligned | kNoClock | kTiledValues | kListed | kHot | kAhead | (fresh<ROLES> ? kNoLoads : 0) | (packed<ROLES> ? kNoMath : 0) | (four<ROLES> || fold_expands<ROLES> ? kFour | kFoldIssues | kDrains4 : 0) | kTwoSums)) == (kFold | kRead)) {
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
            const int t = static_cast<int>(ROLES & kListed ? listed_token((ROLES & kExpanded) == kExpanded && !apart<ROLES> ? (blk * 2 + box) % kColumns : blk * 2 + box,
                                                                          w * kWindow + e, (ROLES & kHot) && !apart<ROLES>)
                                                           : token_of(blk * 2 + box, w * kWindow + e));
            //: the entry's list: its sixteenth, or in the packed row the list whose count's run it falls in
            int a = e / kEntries;
            if (packed<ROLES>) {
              a = 0;
              for (int before = packed_count((blk * 2 + box) % kColumns, w % kCallWindows, 0); before <= e; )
                before += packed_count((blk * 2 + box) % kColumns, w % kCallWindows, ++a);
            }
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
  //: the whole-entry rows' fold alone: what its fold warps wait on their producer
  if ((ROLES & 3) == kFold && (ROLES & kWhole) == kWhole && !(ROLES & kNoClock))
    b.record(name + ": " + sites[1].name, "cycles a window a warp", waited_at[1]);
}

//: A CALL'S BUFFERS: the replicated tables, token-major and the values' tile-major twin, the state, the output and
//: the waits' counts
struct Buffers {
  uint32_t *values, *dense, *out, *waits;
  float* state;
};
inline Buffers buffers(const lab::Bench& b) {
  using namespace carry_window;
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
  {
    //: token `x`'s line `p` (channels `8 p` on) at tile `x / 64`, block `p`, line `x % 64`
    std::vector<uint32_t> tiled(host_values.size());
    for (size_t x = 0; x < host_values.size() / kRowWords; ++x)
      for (int w = 0; w < kRowWords; ++w)
        tiled[((x >> 6) * 512 + (w / 4) * 64 + (x & 63)) * 4 + w % 4] = host_values[x * kRowWords + w];
    uint32_t* dev = nullptr;
    LAB_CHECK(cudaMalloc(&dev, tiled.size() * 4));
    LAB_CHECK(cudaMemcpy(dev, tiled.data(), tiled.size() * 4, cudaMemcpyHostToDevice));
    tiled_values = dev;
  }
  {
    //: the listed row's: a level's byte of token `t` at digit `d` is `dense_of(t, d % 16)` in the dense level and
    //: `sparse_of(t)` in the sparse; a source's list entry its token's place in its window
    const size_t sources = static_cast<size_t>(b.sms) * 2;
    std::vector<uint8_t> blob(static_cast<size_t>(kListsAt) + sources * kListEntries * 2);
    std::memcpy(blob.data(), host_dense.data(), host_dense.size() * 4);
    for (uint32_t t = 0; t < static_cast<uint32_t>(kTableTokens); ++t)
      for (int d = 0; d < kLevelDigits; ++d) {
        const size_t at = static_cast<size_t>(kLevelsAt) + static_cast<size_t>(level_line(t, d / 16)) * 16 + d % 16;
        blob[at] = static_cast<uint8_t>(dense_of(static_cast<int>(t), d % 16));
        blob[static_cast<size_t>(kLevelBytes) + at] = static_cast<uint8_t>(sparse_of(static_cast<int>(t)));
      }
    uint16_t* const ids = reinterpret_cast<uint16_t*>(blob.data() + kListsAt);
    for (size_t src = 0; src < sources; ++src)
      for (int e = 0; e < kListEntries; ++e)
        ids[src * kListEntries + e] = static_cast<uint16_t>(listed_place(static_cast<uint32_t>(src), static_cast<uint32_t>(e)));
    uint8_t* dev = nullptr;
    LAB_CHECK(cudaMalloc(&dev, blob.size()));
    LAB_CHECK(cudaMemcpy(dev, blob.data(), blob.size(), cudaMemcpyHostToDevice));
    listed_tables = reinterpret_cast<const uint32_t*>(dev);
    //: the expanded values: column `c`'s window `w`'s list `a`'s tile -- brick `b` its 16 entries' channels `8 b`
    //: on, a line an entry, the 17th nothing -- from the entries' tokens as the lists place them
    //: `keys` sources' windows: the 16 columns' that the boxes share, or every source's own
    const auto expand = [&](int keys, int copies) {
      std::vector<uint32_t> wide(static_cast<size_t>(keys) * copies * kCallWindows * kExpandedWindow / 4, 0);
      for (int c = 0; c < keys; ++c)
        for (int w = 0; w < copies * kCallWindows; ++w)
          for (int e = 0; e < kWindow; ++e) {
            const uint32_t t = listed_token(static_cast<uint32_t>(c), static_cast<uint32_t>(w * kWindow + e), false);
            const size_t tile = ((static_cast<size_t>(c) * copies * kCallWindows + w) * kExpandedWindow + (e / kEntries) * kExpandedList) / 4;
            for (int br = 0; br < 8; ++br)
              for (int i = 0; i < 4; ++i)
                wide[tile + (br * kBrick + (e % kEntries) * 16) / 4 + i] = host_values[static_cast<size_t>(t) * kRowWords + 4 * br + i];
            //: the ninth brick: the entry's 16 dense bytes a line, and the list's 16 sparse bytes the 17th
            uint8_t* const raw = reinterpret_cast<uint8_t*>(wide.data()) + tile * 4 + kValueTile;
            for (int d = 0; d < 16; ++d) raw[(e % kEntries) * 16 + d] = static_cast<uint8_t>(dense_of(static_cast<int>(t), d));
            raw[16 * kEntries + e % kEntries] = static_cast<uint8_t>(sparse_of(static_cast<int>(t)));
          }
      uint32_t* wdev = nullptr;
      LAB_CHECK(cudaMalloc(&wdev, wide.size() * 4));
      LAB_CHECK(cudaMemcpy(wdev, wide.data(), wide.size() * 4, cudaMemcpyHostToDevice));
      return static_cast<const uint32_t*>(wdev);
    };
    expanded_values = expand(kColumns, 1);
    apart_values = expand(static_cast<int>(sources), 1);
    fresh_values = expand(kColumns, 2);
    {
    //: the packed rows' buffers, a slot `tiles` steps
    const auto pack = [&](int tiles) {
      //: THE PACKED row's: each (column, window)'s head and its stream. A window's steps -- a list's full ones and
      //: its truncated last -- are taken a round at a time, `tiles` a slot, a slot's laid in the order of their lists
      std::vector<uint8_t> buffer(static_cast<size_t>(kPackedTable) + static_cast<size_t>(kColumns) * kCallWindows * kPackedRoom, 0);
      struct Step { int list, round, first, entries; };
      for (int c = 0; c < kColumns; ++c)
        for (int w = 0; w < kCallWindows; ++w) {
          const size_t at = static_cast<size_t>(c) * kCallWindows + w;
          uint8_t* const head = buffer.data() + at * kPackedHead;
          uint8_t* const stream = buffer.data() + kPackedTable + at * kPackedRoom;
          std::vector<Step> steps;
          for (int a = 0, e = 0; a < kLists; ++a) {
            const int n = packed_count(c, w, a);
            for (int k0 = 0; k0 < n; k0 += kEntries) steps.push_back({a, k0 / kEntries, e + k0, std::min(kEntries, n - k0)});
            e += n;
          }
          std::stable_sort(steps.begin(), steps.end(), [](const Step& x, const Step& y) { return x.round < y.round; });
          const uint32_t total = static_cast<uint32_t>(steps.size()), slots = (total + tiles - 1) / tiles;
          //: and no window has fewer slots than a producer issues ahead (2 at most)
          if (total > static_cast<uint32_t>(kPackedSteps) || slots < 2) {
            std::fprintf(stderr, "carry_window: a packed window's %u steps do not fit its head\n", total);
            std::exit(2);
          }
          std::memcpy(head, &slots, 4);
          uint32_t line = 0;
          for (uint32_t i = 0; i < slots; ++i) {
            const auto from = steps.begin() + i * tiles, to = steps.begin() + std::min(total, (i + 1) * tiles);
            std::stable_sort(from, to, [](const Step& x, const Step& y) { return x.list < y.list; });
            uint64_t header[4] = {0, 0, i + 1 == slots, 0};
            for (auto st = from; st != to; ++st) {
              const int m = st->entries, pitch = m == kEntries ? kBrick / 16 : m;
              header[0] += 1ull << (4 * st->list);
              if (m < kEntries) header[1] |= static_cast<uint64_t>(m) << (4 * st->list);
              const uint32_t record[2] = {line, static_cast<uint32_t>(m)};
              std::memcpy(head + kPackedRecordsAt + 8 * (st - steps.begin()), record, 8);
              uint8_t* const tile = stream + static_cast<size_t>(line) * 16;
              uint8_t* const raw = tile + 8 * pitch * 16;
              for (int k = 0; k < m; ++k) {
                const uint32_t t = listed_token(static_cast<uint32_t>(c), static_cast<uint32_t>(w * kWindow + st->first + k), false);
                for (int br = 0; br < 8; ++br)
                  std::memcpy(tile + (br * pitch + k) * 16, &host_values[static_cast<size_t>(t) * kRowWords + 4 * br], 16);
                for (int dd = 0; dd < 16; ++dd) raw[k * 16 + dd] = static_cast<uint8_t>(dense_of(static_cast<int>(t), dd));
                raw[m * 16 + k] = static_cast<uint8_t>(sparse_of(static_cast<int>(t)));
              }
              line += static_cast<uint32_t>(packed_lines(m));
            }
            std::memcpy(head + kPackedHeadersAt + i * kHeader, header, kHeader);
          }
        }
      uint8_t* pdev = nullptr;
      LAB_CHECK(cudaMalloc(&pdev, buffer.size()));
      LAB_CHECK(cudaMemcpy(pdev, buffer.data(), buffer.size(), cudaMemcpyHostToDevice));
      return reinterpret_cast<const uint32_t*>(pdev);
    };
    packed_values = pack(8);
    four_values = pack(4);

    }
  }
  LAB_CHECK(cudaMalloc(&dense, host_dense.size() * 4));
  LAB_CHECK(cudaMemcpy(dense, host_dense.data(), host_dense.size() * 4, cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMalloc(&state, sizeof(float) * b.sms * 2 * kGroup * state_words<kDen>));
  LAB_CHECK(cudaMalloc(&out, static_cast<size_t>(b.sms) * kOutWindows * kWindow * (kChannels / 2 + 1) * 4));
  uint32_t* waits = nullptr;
  LAB_CHECK(cudaMalloc(&waits, static_cast<size_t>(b.sms) * (threads_of<kFive> >> 5) * kCounts * 4));
  return {values, dense, out, waits, state};
}

int main(int argc, char** argv) {
  using namespace carry_window;
  lab::Bench b(argc, argv);
  for (long long roles : std::initializer_list<long long>{1, 2, 3, 3 | kNoLoads, 3 | kNoAdds, 3 | kNoImage, 2 | kStill, 2 | kSuspend, 3 | kSuspend,
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
                    3 | kSuspend | kLean | kNoClock | kDenFold | kDenImage | kDenSplit | kDenRead,
                    3 | kSuspend | kLean | kNoClock | kPitch, 3 | kSuspend | kLean | kNoClock | kTiledValues,
                    3 | kSuspend | kLean | kNoClock | kBig, 3 | kSuspend | kLean | kNoClock | kBig | kTiledValues,
                    1 | kListed, 3 | kSuspend | kLean | kNoClock | kListed, 1 | kListed | kHot,
                    3 | kSuspend | kLean | kNoClock | kListed | kHot,
                    3 | kSuspend | kLean | kNoClock | kListed | kAhead, 1 | kExpanded,
                    3 | kSuspend | kLean | kNoClock | kExpanded, 1 | kWhole, 3 | kSuspend | kLean | kNoClock | kWhole,
                    1 | kApart, 3 | kSuspend | kLean | kNoClock | kApart, 1 | kFresh, 3 | kSuspend | kLean | kNoClock | kFresh, 1 | kPacked, 3 | kSuspend | kLean | kNoClock | kPacked, 1 | kPacked | kFour, 3 | kSuspend | kLean | kNoClock | kPacked | kFour, 1 | kPacked | kFour | kSelf, 1 | kWhole | kFour, 3 | kSuspend | kLean | kNoClock | kWhole | kFour, 3 | kSuspend | kLean | kWhole | kFour,
                    3 | kSuspend | kLean | kNoClock | kWhole | kFour | kFoldIssues,
                    3 | kSuspend | kLean | kWhole | kFour | kFoldIssues,
                    3 | kSuspend | kLean | kNoClock | kWhole | kFour | kFoldIssues | kDrains4,
                    3 | kSuspend | kLean | kWhole | kFour | kFoldIssues | kDrains4,
                    3 | kSuspend | kLean | kNoClock | kWhole | kFour | kTwoSums, 3 | kSuspend | kLean | kWhole | kFour | kTwoSums}) {
    const std::string k = lab::kernel("carry_window::window", {roles});
    //: with no fold, the state's one group of steps is a rolled loop: one instruction
    //: the ahead row's asks are in its binary: 6 a batch, at the producers' two sites
    if ((roles & kAhead) && (roles & kExpanded) != kExpanded) b.expect(k, "CCTL", ">=", 12);
    //: the packed row's fold steps are a list's rolled loop: the toolkit lays a list's first step and its later ones
    //: as two instructions, and one more at the loop's entry
    if ((roles & kPacked) == kPacked) {
      b.expect(k, "HGMMA", ">=", 2 * kSlotLists + (roles & kRead ? 2 * kChunkDigits : 0));
      b.expect(k, "HGMMA", "<=", 4 * kSlotLists + 2 + (roles & kRead ? 2 * kChunkDigits : 0));
    } else
      b.expect(k, "HGMMA", "==",
             (roles & kFold ? 2 * kSlotLists + (roles & kDenFold ? kDenSteps : 0) : roles & kStill ? 0 : 1)
                 + (roles & kRead ? 2 * kChunkDigits : 0));
    //: a packed producer's statement of a later slot's bytes follows its store of its own slot's header: the fold
    //: reads the header behind the slot's arrival, which the fence precedes
    if ((roles & kPacked) == kPacked && !(roles & kRead)) b.expect(k, "STORE_UNFENCED", "<=", 1);
    //: the fold-expands rows' conversions are in the fold's own trip, beside its steps: what the rows measure
    if ((roles & kWhole) == kWhole && (roles & kPacked) != kPacked && (roles & kFour)) b.expect(k, "STEP_XU", "<=", 40);
    if (roles & kRead) {
      //: the readout's panel is read by the drain warps' own loads, not by a step: its arrival needs no fence
      //: nor does the wide row's scratch, which only its own warp's loads read
      //: the row that takes the fold's fence out leaves its four tile groups' arrivals unfenced: what it measures
      b.expect(k, "STORE_UNFENCED", "<=", (roles & kNoFence ? 5 : roles & kWide ? 2 : 1) + ((roles & kPacked) == kPacked ? 1 : 0));
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
  const Buffers held = buffers(b);
  uint32_t *const values = held.values, *const dense = held.dense, *const out = held.out, *const waits = held.waits;
  float* const state = held.state;
  run<kFold>(b, waits, values, dense, state, out);
  run<kRead>(b, waits, values, dense, state, out);
  //: the fold-expands rows and the packed rows first: their waits are their own forms', and a hang shows at once
  run<kFold | kRead | kSuspend | kLean | kNoClock | kWhole | kFour | kTwoSums>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kWhole | kFour | kTwoSums>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kNoClock | kWhole | kFour | kFoldIssues>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kWhole | kFour | kFoldIssues>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kNoClock | kWhole | kFour | kFoldIssues | kDrains4>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kWhole | kFour | kFoldIssues | kDrains4>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kWhole | kFour>(b, waits, values, dense, state, out);
  run<kFold | kWhole | kFour>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kNoClock | kWhole | kFour>(b, waits, values, dense, state, out);
  run<kFold | kPacked | kFour | kSelf>(b, waits, values, dense, state, out);
  run<kFold | kPacked | kFour>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kNoClock | kPacked | kFour>(b, waits, values, dense, state, out);
  run<kFold | kPacked>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kNoClock | kPacked>(b, waits, values, dense, state, out);
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
  run<kFold | kRead | kSuspend | kLean | kNoClock | kPitch>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kNoClock | kTiledValues>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kNoClock | kBig>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kNoClock | kBig | kTiledValues>(b, waits, values, dense, state, out);
  run<kFold | kListed>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kNoClock | kListed>(b, waits, values, dense, state, out);
  run<kFold | kListed | kHot>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kNoClock | kListed | kHot>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kNoClock | kListed | kAhead>(b, waits, values, dense, state, out);
  run<kFold | kExpanded>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kNoClock | kExpanded>(b, waits, values, dense, state, out);
  run<kFold | kWhole>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kNoClock | kWhole>(b, waits, values, dense, state, out);
  run<kFold | kApart>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kNoClock | kApart>(b, waits, values, dense, state, out);
  run<kFold | kFresh>(b, waits, values, dense, state, out);
  run<kFold | kRead | kSuspend | kLean | kNoClock | kFresh>(b, waits, values, dense, state, out);
  return 0;
}
