// measure/lab/experiments/carry_window.cu -- QUESTION: the carry's roles were each measured apart (`pool_fill`,
// `state_image`, `readout_role`); what does a window cost a CTA when they run TOGETHER, under one register file,
// one tensor pipe and one shared memory, and which role waits on which? METHOD: a CTA an SM of four warpgroups,
// 512 threads, registers traded by role. TWO FOLD warpgroups, a 16 x 16 box each, its state 16 accumulators in
// registers. TWO PRODUCER warps, one a box, fill a slot's 8 lists of 16 entries from scattered rows (`pool_fill`'s
// form, one batch in registers) and the fold runs the slot's 8 steps. A window is a box's 256 entries, and 256
// tokens read the pair: each fold warpgroup PUBLISHES its state as a bf16 image, 16 tiles, a group of 4 behind the
// readout. ONE READOUT warpgroup takes a chunk of 4 read digits of 16 tokens, a digit a warp's rows: a warp
// expands its tokens' coefficients from their rows' bytes IN ITS REGISTERS, and they are the A operand of the
// digit's step on each box's image tile, its B; one accumulator holds the chunk's 64 tokens by 64 channels, and
// the readout stores each token's row to a panel as the f16 pairs its line holds. TWO DRAIN warps add each
// token's row as one f16 line. The fold's, the readout's and the drains' waits at their barriers are clocked,
// and the readout's steps. Rows: the fold's side alone, the read side alone, both, and both with one access taken
// out: the producers' global loads, the drains' global adds, the image's stores. Both: the state and every
// output row are checked exactly, and a long call's span over its wall time reads the card's clock.
// TIMELINE: a fold warpgroup's trip is a window: the image a tile group at a time, four -- the wait on the
// readout's read of that group a window back, 32 stores a thread and their fence, the group's arrival -- then a
// slot twice: the wait on its producer, the fence, 8 steps, the commit and its wait, the slot freed. A producer's
// trip is a batch of 32 entries, a slot four of them: the batch's stores and multiplies (its words loaded a trip
// before), then the same registers' loads for the next batch; a slot on, the fence and the arrival. The
// readout's trip is a chunk: the wait on its tile group, 8 rounds a thread of coefficients (their bytes loaded a
// chunk before), the fence, 8 steps of 64 columns, the commit, the next chunk's 8 loads, the wait, the group
// released; then the wait on the panel two chunks back, 16 rounds to f16 pairs and stores a thread, the arrival.
// A drain warp's trip is a chunk, four batches of 8 tokens alternating two register halves: a batch's adds, each
// a word as the panel holds it, then those registers' loads for the batch two on (the chunk's first two as it
// opens). Live: a fold thread's 128 accumulators; a producer's 36 words; the readout's 32 accumulators and 8
// words; a drain's 16.
// see docs/internals/tools/lab.md#carry-window
#include <chrono>

#include "lab.cuh"

namespace carry_window {

constexpr int kLists = 16, kEntries = 16, kWindow = kLists * kEntries, kSlotLists = 8, kSlot = kSlotLists * kEntries;
constexpr int kChannels = 64, kTableTokens = 65536, kRowWords = 32, kFill = 32;
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
//: AN IMAGE TILE (the readout's operand a read digit): a line a fold digit's 8 channels, bricks 17 lines, tiles 137
constexpr int kBrickLines = 17, kImageTile = 137 * 16, kBoxBytes = kLists * kImageTile;
//: a panel row: a token's 64 f16 as its line will be added, rows 272 bytes apart
constexpr int kChunkDigits = 4, kTokens = 16 * kChunkDigits, kChunks = kLists / kChunkDigits, kRowBytes = 272,
              kPanelBytes = kTokens * kRowBytes, kBatch = 8;
constexpr int kImageAt = 4 * kPoolBytes, kPanelAt = kImageAt + 2 * kBoxBytes,
              kBarsAt = kPanelAt + 2 * kPanelBytes, kSmemBytes = kBarsAt + 160;
//: the barriers, 8 bytes each: a pool slot filled and freed (box, slot); an image's tile group published and read
//: (a group is the 4 read digits `8 h + c + 2 j` of a chunk `2 h + c`: every fold thread holds a quarter of it); a
//: panel filled and drained
constexpr int kPoolReady = 0, kPoolFreed = 32, kTilesReady = 64, kTilesRead = 96, kPanelReady = 128, kPanelFreed = 144;
constexpr int kOutWindows = 4;
//: a row's roles, and what a row takes OUT of them to find what the two sides contend for: the producers' global
//: loads (the words made in registers), the drains' global adds (the lines folded into a sink), the image's stores
enum Roles : int { kFold = 1, kRead = 2, kNoLoads = 4, kNoAdds = 8, kNoImage = 16 };

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

//: A WAIT, its cycles added to the warp's count for that barrier: a role's time is its span less its waits
__device__ __forceinline__ void waited(uint32_t bar, uint32_t parity, uint32_t& count) {
  const uint32_t t0 = static_cast<uint32_t>(clock64());
  lab::mbar_wait(bar, parity);
  count += static_cast<uint32_t>(clock64()) - t0;
}
//: a warp's four counts, stored at `kCounts (block, warp)`: its two waits, and the readout's two clocked phases
constexpr int kCounts = 4;
__device__ __forceinline__ void tell(uint32_t* waits, uint32_t first, uint32_t second, uint32_t third = 0,
                                     uint32_t fourth = 0) {
  if ((threadIdx.x & 31) == 0) {
    uint32_t* const to = waits + kCounts * (blockIdx.x * (kThreads >> 5) + (threadIdx.x >> 5));
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
  //: a thread's base into its box's image: every store is this plus a constant of (fold digit, column, row)
  uint8_t* const mine = smem + lab::held(static_cast<uint32_t>(kImageAt + box * kBoxBytes + 2 * t * kImageTile
                                                               + 16 * (2 * warp * kBrickLines) + 2 * q));
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
      //: THE IMAGE, a tile group at a time behind the readout: value (channel m, fold digit a) of read digit b to
      //: tile b, line `(m / 8) 17 + a`, slot `m % 8`, once the readout has read that group of the window before.
      //: The first window's image is the cleared shared memory: the state is zero
#pragma unroll
      for (int k = 0; k < kChunks; ++k) {
        if (w > 0) waited(bars + kTilesRead + 8 * k, (w - 1) & 1, on_read);
        if (w > 0 && !(ROLES & kNoImage)) {
#pragma unroll
          for (int a = 0; a < kLists; ++a)
#pragma unroll
            for (int r = 0; r < 2; ++r)
              *reinterpret_cast<uint16_t*>(mine + (8 * (k >> 1) + (k & 1)) * kImageTile + 16 * a + 16 * r * kBrickLines)
                  = static_cast<uint16_t>(__float_as_uint(d[a][(k & 1) | r << 1 | (k >> 1) << 2]) >> 16);
        }
        lab::bulk_fence();
        __syncwarp();
        if (lane == 0) lab::mbar_arrive(bars + kTilesReady + 8 * k);
      }
    }
    if constexpr (ROLES & kFold) {
#pragma unroll
      for (int slot = 0; slot < 2; ++slot) {
        waited(bars + kPoolReady + 8 * (2 * box + slot), w & 1, on_fill);
        const uint32_t tiles = lab::held(base + (2 * box + slot) * kPoolBytes);
        lab::wg_fence();
#pragma unroll
        for (int j = 0; j < kSlotLists; ++j)
          lab::wg_bf16_n16_ss_mn(d[slot * kSlotLists + j], lab::wg_desc(tiles + j * kTile, 8, 17),
                                 lab::wg_desc(tiles + j * kTile + kValueTile, 8, 17), w != 0);
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
  clock.stop(times);
  tell(waits, on_read, on_fill);
  float* const to = state + (static_cast<long long>(blockIdx.x) * 2 * kGroup + tid) * (kLists * 8);
#pragma unroll
  for (int a = 0; a < kLists; ++a)
#pragma unroll
    for (int i = 0; i < 8; ++i) to[8 * a + i] = d[a][i];
}

//: A PRODUCER WARP, one a box: a slot's 128 entries in four batches of 32, batches alternating two register halves
template <int ROLES>
__device__ __forceinline__ void produce_role(uint8_t* smem, uint32_t base, long long* times, const uint32_t* values,
                                             const uint32_t* dense, int windows) {
  const int lane = threadIdx.x & 31, box = (threadIdx.x - kProduceAt) >> 5, group = lane >> 3;
  const uint32_t bars = base + kBarsAt, source = blockIdx.x * 2 + box;
  constexpr int kQuads = kFill / 4, kBatches = kSlot / kFill;
  const uint4* const wide = reinterpret_cast<const uint4*>(values);
  const uint4* const boxes = reinterpret_cast<const uint4*>(dense);
  //: ONE batch in registers: the warpgroup's 96 a thread hold no second
  uint4 value[kQuads], bytes;
  //: batch `n` of the call: 8 value instructions of four entries (8 lanes an entry, a lane 16 bytes), and one of
  //: all 32 entries' dense bytes (a lane an entry's 16)
  const auto load = [&](int n) {
#pragma unroll
    for (int qd = 0; qd < kQuads; ++qd) {
      const uint32_t tk = token_of(source, n * kFill + 4 * qd + group);
      if constexpr (ROLES & kNoLoads) value[qd] = make_uint4(tk, tk, tk, tk);
      else value[qd] = wide[static_cast<long long>(tk) * (kRowWords / 4) + (lane & 7)];
    }
    const uint32_t tb = token_of(source, n * kFill + lane);
    if constexpr (ROLES & kNoLoads) bytes = make_uint4(tb, tb, tb, tb);
    else bytes = boxes[tb];
  };
  load(0);
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int w = 0; w < windows; ++w) {
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
      if (w > 0) lab::mbar_wait(bars + kPoolFreed + 8 * (2 * box + slot), (w - 1) & 1);
      uint8_t* const tiles = smem + (2 * box + slot) * kPoolBytes;
      //: a rolled loop: a batch's addresses are formed in its trip, none held for the slot
#pragma unroll 1
      for (int b = 0; b < kBatches; ++b) {
#pragma unroll
        for (int qd = 0; qd < kQuads; ++qd) {
          //: entry `e` of the slot: list `e / 16`, token `e % 16` of its step
          const int e = b * kFill + 4 * qd + group;
          *reinterpret_cast<uint4*>(tiles + (e / kEntries) * kTile + (lane & 7) * kBrick + (e % kEntries) * 16) =
              value[qd];
        }
        //: the lane's entry: its 16 bytes times the entry's, 8 pairs of bf16, a brick's line each four
        const int e = b * kFill + lane;
        const float weight = static_cast<float>(sparse_of(token_of(source, w * kWindow + slot * kSlot + e)));
        const uint32_t word[4] = {bytes.x, bytes.y, bytes.z, bytes.w};
        uint4 line[2];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          uint32_t pair[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const uint32_t two = word[2 * k + j / 2] >> (16 * (j & 1));
            pair[j] = lab::bf16x2_round(weight * static_cast<float>(two & 255u),
                                        weight * static_cast<float>(two >> 8 & 255u));
          }
          line[k] = make_uint4(pair[0], pair[1], pair[2], pair[3]);
        }
        //: a batch's two lists take opposite bricks first: each store covers the 32 banks evenly
        const int first = (e / kEntries) & 1;
        uint8_t* const at = tiles + (e / kEntries) * kTile + kValueTile + (e % kEntries) * 16;
        *reinterpret_cast<uint4*>(at + first * kBrick) = first ? line[1] : line[0];
        *reinterpret_cast<uint4*>(at + (first ^ 1) * kBrick) = first ? line[0] : line[1];
        load((2 * w + slot) * kBatches + b + 1);
      }
      lab::bulk_fence();
      __syncwarp();
      if (lane == 0) lab::mbar_arrive(bars + kPoolReady + 8 * (2 * box + slot));
    }
  }
  clock.stop(times);
}

//: THE READOUT WARPGROUP: a chunk is 4 read digits of 16 tokens, a digit a warp's 16 rows. A warp expands its
//: tokens' coefficients in its registers and they are the step's A operand; the image's tile is its B; the
//: accumulator is the chunk's 64 tokens by 64 channels, stored to the panel as the f16 pairs a line holds
__device__ __forceinline__ void read_role(uint8_t* smem, uint32_t base, long long* times, uint32_t* waits,
                                          const uint32_t* dense, int windows) {
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
  const auto load = [&](int c) {
#pragma unroll
    for (int box = 0; box < 2; ++box)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        bytes[box][j] = dense[static_cast<long long>(reader_row(reader(c, j & 1), box)) * 4 + 2 * (j >> 1) + (t >> 1)];
  };
  load(0);
  uint32_t on_image = 0, on_drain = 0, in_steps = 0;
  //: the chunk's first step writes the accumulator whole: nothing is cleared between chunks
  float d[32];
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int w = 0; w < windows; ++w) {
#pragma unroll 1
    for (int c4 = 0; c4 < kChunks; ++c4) {
      const int c = kChunks * w + c4;
      //: chunk `2 h + c`'s digits are `8 h + c + 2 j`: its tile group, published for this window
      waited(bars + kTilesReady + 8 * c4, w & 1, on_image);
      //: THE COEFFICIENTS, in registers: a row's two bytes times its reader's byte on the digit, a bf16 pair a word
      //: -- word `r | h << 1` of A, as an m64n16 accumulator lays its values
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
      uint32_t since = static_cast<uint32_t>(clock64());
      const uint32_t first = static_cast<uint32_t>(8 * (c4 >> 1) + (c4 & 1));
      const uint32_t images = lab::held(base + kImageAt + first * kImageTile);
      lab::wg_fence();
      //: digit `j`'s step on each box's tile: its tokens are warp `j`'s rows, every other warp's A nothing
#pragma unroll
      for (int j = 0; j < kChunkDigits; ++j)
#pragma unroll
        for (int box = 0; box < 2; ++box) {
          const uint32_t rows[4] = {warp == j ? a[box][0] : 0u, warp == j ? a[box][1] : 0u,
                                    warp == j ? a[box][2] : 0u, warp == j ? a[box][3] : 0u};
          lab::wg_bf16_n64_rs_mn(d, rows, lab::wg_desc(images + box * kBoxBytes + 2 * j * kImageTile, 8, kBrickLines),
                                 j != 0 || box != 0);
        }
      lab::wg_commit();
      //: the next chunk's bytes, under the steps: the coefficients' registers are free once the steps are issued
      load(c + 1);
      lab::wg_wait<0>();
      in_steps += lap(since);
      //: the group is read: its box may publish the next window's over it
      if (lane == 0) lab::mbar_arrive(bars + kTilesRead + 8 * c4);
      //: THE PANEL, once the drain has left it: a row's channels `8 j + 2 t` and the next rounded to an f16 pair,
      //: the word of the line a drain warp adds as it lies; a warp's lanes on 32 banks
      if (c >= 2) waited(bars + kPanelFreed + 8 * (c & 1), ((c >> 1) - 1) & 1, on_drain);
      uint8_t* const row = smem + mine + (c & 1) * kPanelBytes;
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int r = 0; r < 2; ++r)
          *reinterpret_cast<uint32_t*>(row + 8 * r * kRowBytes + 16 * j) =
              lab::pack_half2(d[4 * j + 2 * r], d[4 * j + 2 * r + 1]);
      __syncwarp();
      if (lane == 0) lab::mbar_arrive(bars + kPanelReady + 8 * (c & 1));
    }
  }
  clock.stop(times);
  tell(waits, on_image, on_drain, in_steps);
}

//: A DRAIN WARP: a chunk's 32 tokens of its residue in four batches of 8; a lane holds a token's two channels and
//: adds one f16 pair
template <int ROLES>
__device__ __forceinline__ void drain_role(const uint8_t* smem, uint32_t base, long long* times, uint32_t* waits,
                                           int* sinks,
                                           uint32_t* out, int windows) {
  const int lane = threadIdx.x & 31, mine = (threadIdx.x - kDrainAt) >> 5;
  const uint32_t bars = base + kBarsAt;
  constexpr int kBatches = kTokens / kDrains / kBatch;
  static_assert(kBatches % 2 == 0);
  uint32_t held[2][kBatch];
  uint32_t kept = 0;
  //: batch `n` of chunk `c` into register half `n & 1`: a lane's word of a token's row is its f16 pair
  const auto load = [&](int c, int n) {
#pragma unroll
    for (int i = 0; i < kBatch; ++i)
      held[n & 1][i] = reinterpret_cast<const uint32_t*>(
          smem + kPanelAt + (c & 1) * kPanelBytes + (mine + kDrains * (n * kBatch + i)) * kRowBytes)[lane];
  };
  uint32_t on_panel = 0;
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int c = 0; c < kChunks * windows; ++c) {
    waited(bars + kPanelReady + 8 * (c & 1), (c >> 1) & 1, on_panel);
    load(c, 0);
    load(c, 1);
    //: the chunk's digit `j` is read digit `8 h + c + 2 j`: its 16 tokens' rows start at 16 of that
    const long long first = static_cast<long long>(blockIdx.x) * (kOutWindows * kWindow)
                            + ((c / kChunks) % kOutWindows) * kWindow
                            + 16 * (8 * ((c % kChunks) >> 1) + ((c % kChunks) & 1));
#pragma unroll
    for (int n = 0; n < kBatches; ++n) {
#pragma unroll
      for (int i = 0; i < kBatch; ++i) {
        const int token = mine + kDrains * (n * kBatch + i);
        const uint32_t pair = held[n & 1][i];
        if constexpr (ROLES & kNoAdds) kept ^= pair + token;
        else lab::red_add_f16x2(out + (first + 32 * (token / 16) + token % 16) * (kChannels / 2) + lane, pair);
      }
      if (n + 2 < kBatches) load(c, n + 2);
    }
    __syncwarp();
    if (lane == 0) lab::mbar_arrive(bars + kPanelFreed + 8 * (c & 1));
  }
  clock.stop(times);
  tell(waits, on_panel, 0u);
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
__global__ void __launch_bounds__(kThreads, 1)
    window(long long* times, uint32_t* waits, int* sinks, const uint32_t* values, const uint32_t* dense, float* state,
           uint32_t* out, int windows) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x;
  const uint32_t base = lab::smem(smem);
  //: the images cleared: the state is zero at the call's start
#pragma unroll 1
  for (int i = tid; i < 2 * kBoxBytes / 4; i += kThreads) *reinterpret_cast<uint32_t*>(smem + kImageAt + 4 * i) = 0u;
  if (tid == 0) {
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      lab::mbar_init(base + kBarsAt + kPoolReady + 8 * p, 1);
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
      lab::mbar_init(base + kBarsAt + kPanelFreed + 8 * p, kDrains);
    }
  }
  lab::bulk_fence();
  __syncthreads();

  //: EACH ROLE'S WHOLE BODY IN ITS OWN BRANCH, a warpgroup's registers set at the branch's head in every row: the
  //: folds' raise is the readout's release (docs/lab/carry_window.md)
  if (tid < kReadAt) {
    lab::regs_up<kFoldRegs>();
    fold_role<ROLES>(smem, base, times, waits, state, windows);
  } else if (tid < kProduceAt) {
    lab::regs_down<kReadRegs>();
    if constexpr (ROLES & kRead) read_role(smem, base, times, waits, dense, windows);
    else idle_role(times, waits);
  } else {
    //: the producers and the drain warps are one warpgroup: one register call, the four warps together
    lab::regs_down<kServeRegs>();
    if (tid < kDrainAt) {
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
                           ;
  const auto kernel = window<ROLES>;
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  const size_t words = static_cast<size_t>(b.sms) * kOutWindows * kWindow * (kChannels / 2);
  LAB_CHECK(cudaMemset(out, 0, words * 4));
  if (ROLES == (kFold | kRead)) {
    kernel<<<b.sms, kThreads, kSmemBytes>>>(b.times, waits, b.sinks, values, dense, state, out, kOutWindows);
    LAB_CHECK(cudaDeviceSynchronize());
    const size_t per = static_cast<size_t>(2 * kGroup) * kLists * 8;
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
                want += image_host(s[at(box, m, a, bb)]) * sparse_of(static_cast<int>(rt))
                        * dense_of(static_cast<int>(reader_row(rt, box)), a);
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
          }
      }
      for (int tid = 0; tid < 2 * kGroup; ++tid) {
        const int box = tid >> 7, rows = 16 * ((tid >> 5) & 3) + (tid & 31) / 4, pair = tid & 3;
        for (int a = 0; a < kLists; ++a)
          for (int i = 0; i < 8; ++i) {
            const int m = rows + 8 * ((i >> 1) & 1), bb = 8 * (i >> 2) + 2 * pair + (i & 1);
            wrong_state += got_state[blk * per + (static_cast<size_t>(tid) * kLists + a) * 8 + i]
                           != static_cast<float>(s[at(box, m, a, bb)]);
          }
      }
    }
    b.check(name, wrong_state == 0 && wrong_out == 0,
            std::to_string(wrong_state) + " of " + std::to_string(got_state.size()) + " state values and "
                + std::to_string(wrong_out) + " of " + std::to_string(2 * words) + " output values differ");
  } else {
    b.unchecked(name, "an isolation: a side's roles or one of their accesses absent");
  }
  //: THE WAITS: a role's warps' mean cycles a window at each of its barriers (fold: on the readout's read of a
  //: tile group, on its producer; readout: on a tile group's publish, on the panel's drain; drain: on the
  //: panel). A producer's are not clocked: its 128 registers hold no count
  struct Site { int first, count, slot; const char* name; };
  constexpr int kSites = 6;
  const Site sites[kSites] = {{0, 8, 0, "a fold warp waits on the readout's read"},
                              {0, 8, 1, "a fold warp waits on its producer"},
                              {8, 4, 0, "a readout warp waits on the image"},
                              {8, 4, 1, "a readout warp waits on the drain"},
                              {8, 4, 2, "a readout warp in its steps"},
                              {14, 2, 0, "a drain warp waits on the panel"}};
  lab::Samples waited_at[kSites];
  const lab::Samples cycles = b.variant(
      [&] { kernel<<<b.sms, kThreads, kSmemBytes>>>(b.times, waits, b.sinks, values, dense, state, out, kTimed); },
      [&] {
        std::vector<uint32_t> h(static_cast<size_t>(b.sms) * (kThreads >> 5) * kCounts);
        LAB_CHECK(cudaMemcpy(h.data(), waits, h.size() * 4, cudaMemcpyDeviceToHost));
        for (int i = 0; i < kSites; ++i) {
          double sum = 0;
          for (int blk = 0; blk < b.sms; ++blk)
            for (int w = sites[i].first; w < sites[i].first + sites[i].count; ++w)
              sum += h[kCounts * (static_cast<size_t>(blk) * (kThreads >> 5) + w) + sites[i].slot];
          waited_at[i].v.push_back(sum / (static_cast<double>(b.sms) * sites[i].count * kTimed));
        }
        return b.span(kThreads >> 5) / kTimed;
      });
  b.record(name, "cycles a window a CTA", cycles);
  //: the composed row's sites, and the read side's own but the fold's wait on a producer, which does not run there
  if (ROLES == (kFold | kRead)) {
    //: THE CARD'S CLOCK: a long call's span in SM cycles over its wall time, the launch and its wait included (a
    //: call of some 20 ms: they are under a percent of it)
    constexpr int kLong = 4096;
    lab::Samples clock_rate;
    for (int r = 0; r < 5; ++r) {
      LAB_CHECK(cudaDeviceSynchronize());
      const auto t0 = std::chrono::steady_clock::now();
      kernel<<<b.sms, kThreads, kSmemBytes>>>(b.times, waits, b.sinks, values, dense, state, out, kLong);
      LAB_CHECK(cudaDeviceSynchronize());
      const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      clock_rate.v.push_back(b.span(kThreads >> 5) / seconds / 1e9);
    }
    b.unchecked("the card's clock", "a rate read off the composed row's call, which that row checks");
    b.record("the card's clock: the composed call's span over its wall time", "GHz", clock_rate);
  }
  if (!(ROLES & (kNoLoads | kNoAdds | kNoImage)) && ROLES & kRead)
    for (int i = 0; i < kSites; ++i)
      if (ROLES & kFold || i != 1) b.record(name + ": " + sites[i].name, "cycles a window a warp", waited_at[i]);
}

int main(int argc, char** argv) {
  using namespace carry_window;
  lab::Bench b(argc, argv);
  for (int roles : {1, 2, 3, 3 | kNoLoads, 3 | kNoAdds, 3 | kNoImage}) {
    const std::string k = lab::kernel("window", {roles});
    //: with no fold, the state's one group of steps is a rolled loop: one instruction
    b.expect(k, "HGMMA", "==", (roles & kFold ? 2 * kSlotLists : 1) + (roles & kRead ? 2 * kChunkDigits : 0));
    if (roles & kRead) {
      //: the readout's panel is read by the drain warps' own loads, not by a step: its arrival needs no fence
      b.expect(k, "STORE_UNFENCED", "<=", 1);
      //: the readout's trip holds its coefficients' rounds and their load beside its steps, by design: the tensor
      //: pipe is idle most of a window
      b.expect(k, "STEP_XU", ">=", 0);
      b.expect(k, "STEP_GLOBAL", ">=", 0);
    }
    b.exempt(k, "a composition probe: the carry's roles in one CTA, no kernel's step");
    //: once a slot or a chunk a role: a `__syncwarp` before a one-lane arrival is a convergence call, and each
    //: barrier's polled wait a region -- five roles' in one kernel
    b.expect(k, "LOOP_CONVERGENCE", "<=", 64);
    //: the roles' loops hold their steps beside their barriers' polled waits: the waits' branches are the steps'
    b.expect(k, "BSSY", ">=", 0);
  }
  if (b.expect_only) return 0;
  std::vector<uint32_t> host_values(static_cast<size_t>(kTableTokens) * kRowWords), host_dense(kTableTokens * 4);
  for (size_t w = 0; w < host_values.size(); ++w) {
    const int t = static_cast<int>(w / kRowWords), c = 2 * static_cast<int>(w % kRowWords);
    host_values[w] = bf16_host(value_of(t, c)) | bf16_host(value_of(t, c + 1)) << 16;
  }
  for (size_t w = 0; w < host_dense.size(); ++w) {
    const int t = static_cast<int>(w / 4), b0 = 4 * static_cast<int>(w % 4);
    host_dense[w] = dense_of(t, b0) | dense_of(t, b0 + 1) << 8 | dense_of(t, b0 + 2) << 16 | dense_of(t, b0 + 3) << 24;
  }
  uint32_t *values = nullptr, *dense = nullptr, *out = nullptr;
  float* state = nullptr;
  LAB_CHECK(cudaMalloc(&values, host_values.size() * 4));
  LAB_CHECK(cudaMemcpy(values, host_values.data(), host_values.size() * 4, cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMalloc(&dense, host_dense.size() * 4));
  LAB_CHECK(cudaMemcpy(dense, host_dense.data(), host_dense.size() * 4, cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMalloc(&state, sizeof(float) * b.sms * 2 * kGroup * kLists * 8));
  LAB_CHECK(cudaMalloc(&out, static_cast<size_t>(b.sms) * kOutWindows * kWindow * (kChannels / 2) * 4));
  uint32_t* waits = nullptr;
  LAB_CHECK(cudaMalloc(&waits, static_cast<size_t>(b.sms) * (kThreads >> 5) * kCounts * 4));
  run<kFold>(b, waits, values, dense, state, out);
  run<kRead>(b, waits, values, dense, state, out);
  run<kFold | kRead>(b, waits, values, dense, state, out);
  run<kFold | kRead | kNoLoads>(b, waits, values, dense, state, out);
  run<kFold | kRead | kNoAdds>(b, waits, values, dense, state, out);
  run<kFold | kRead | kNoImage>(b, waits, values, dense, state, out);
  return 0;
}
