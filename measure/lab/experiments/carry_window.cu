// measure/lab/experiments/carry_window.cu -- QUESTION: the carry's roles were each measured apart (`pool_fill`,
// `state_image`, `readout_role`); what does a window cost a CTA when they run TOGETHER, under one register file,
// one tensor pipe and one shared memory? METHOD: a CTA an SM of four warpgroups, 512 threads, registers traded by
// role. TWO FOLD warpgroups, a 16 x 16 box each, its state 16 accumulators in registers. FOUR PRODUCER warps, two a
// box, fill a slot's 8 lists of 16 entries from scattered rows (`pool_fill`'s form) and the fold runs the slot's
// 8 steps. A window is a box's 256 entries, and 256 tokens read the pair: each fold warpgroup PUBLISHES its state
// as a bf16 image, 16 tiles, a group of 4 behind the readout; ONE READOUT warpgroup is the read side whole -- it
// expands a chunk's 64 readers' coefficients from their rows' bytes into its own tiles, runs a read digit's step
// on both boxes' image tiles into one accumulator, 4 digits a chunk, and adds each token's row to the output as
// f16 pairs from its registers. Rows: the fold's side alone, the read side alone, both, and both with one access
// taken out -- the producers' global loads, the readout's global adds, the image's stores. Both: the state and
// every output row are checked exactly.
// TIMELINE: a fold warpgroup's trip is a window: the image a tile group at a time, four -- the wait on the
// readout's read of that group a window back, 32 stores a thread and their fence, the group's arrival -- then a
// slot twice: the wait on its producers, the fence, 8 steps, the commit and its wait, the slot freed. A
// producer's trip is a slot, two batches of 32 entries at their own registers: a batch's stores and multiplies
// (its value words loaded two batches before, its dense bytes one), each register loaded again as it is stored;
// the fence and the arrival. The readout's trip is a chunk: the wait on its tile group, two stores a thread of
// coefficients (their
// bytes loaded a chunk before), the next chunk's load and their fence, 8 steps, the commit and its wait, the group
// released, then 16 lines a thread, each a shuffle, a round and an add. Live: a fold thread's 128 accumulators; a
// producer's 68 words; the readout's 32 and its 4 words of bytes.
// see docs/internals/tools/lab.md#carry-window
#include "lab.cuh"

namespace carry_window {

constexpr int kLists = 16, kEntries = 16, kWindow = kLists * kEntries, kSlotLists = 8, kSlot = kSlotLists * kEntries;
constexpr int kChannels = 64, kTableTokens = 65536, kRowWords = 32, kFill = 32;
constexpr int kGroup = 128, kThreads = 4 * kGroup;
//: the roles' first threads: fold 0, fold 1, the readout, the four producers
constexpr int kReadAt = 2 * kGroup, kProduceAt = 3 * kGroup;
//: THE REGISTERS A THREAD. A raise takes what another warpgroup released from the count the kernel is launched
//: at -- 128 a thread at 512 threads, the SM's whole file: the folds' 32 more each are the readout's 64 fewer
constexpr int kEntryRegs = 128, kFoldRegs = 160, kReadRegs = 64;
static_assert(kThreads * kEntryRegs == 65536 && 2 * (kFoldRegs - kEntryRegs) == kEntryRegs - kReadRegs);

//: A POOL TILE (the fold's operands a list), M|N-major: 8 bricks of values and 2 of coefficients, a brick 17 lines
constexpr int kBrick = 17 * 16, kValueTile = 8 * kBrick, kTile = 10 * kBrick, kPoolBytes = kSlotLists * kTile;
//: AN IMAGE TILE (the readout's operand a read digit): a line a fold digit's 8 channels, bricks 17 lines, tiles 137
constexpr int kBrickLines = 17, kImageTile = 137 * 16, kBoxBytes = kLists * kImageTile, kCoefBytes = 512;
//: a chunk is 4 read digits of 16 tokens; its coefficient tiles, one a (digit, box), are filled and read by the
//: readout itself
constexpr int kChunkDigits = 4, kChunks = kLists / kChunkDigits;
constexpr int kImageAt = 4 * kPoolBytes, kCoefAt = kImageAt + 2 * kBoxBytes,
              kBarsAt = kCoefAt + 2 * kChunkDigits * kCoefBytes, kSmemBytes = kBarsAt + 128;
//: the barriers, 8 bytes each: a pool slot filled and freed (box, slot); an image's tile group published and read
//: (a group is the 4 read digits `8 h + c + 2 j` of a chunk `2 h + c`: every fold thread holds a quarter of it)
constexpr int kPoolReady = 0, kPoolFreed = 32, kTilesReady = 64, kTilesRead = 96;
constexpr int kOutWindows = 4;
//: a row's roles, and what a row takes OUT of them to find what the two sides contend for: the producers' global
//: loads (the words made in registers), the readout's global adds (its lines folded into a sink), the image's stores
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

__device__ __forceinline__ uint16_t bf16_of(int v) {
  return static_cast<uint16_t>(__float_as_uint(static_cast<float>(v)) >> 16);
}

//: A FOLD WARPGROUP: its box's state; a window is the image, then two slots of 8 steps
template <int ROLES>
__device__ __forceinline__ void fold_role(uint8_t* smem, uint32_t base, long long* times, float* state, int windows) {
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
      lab::wg_bf16_n16_ss_mn(d[a], lab::wg_desc(base + kCoefAt, 8, 17), lab::wg_desc(base + kCoefAt, 8, 17), false);
    lab::wg_commit();
    lab::wg_wait<0>();
  }
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
        if (w > 0) lab::mbar_wait(bars + kTilesRead + 8 * k, (w - 1) & 1);
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
        lab::mbar_wait(bars + kPoolReady + 8 * (2 * box + slot), w & 1);
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
  float* const to = state + (static_cast<long long>(blockIdx.x) * 2 * kGroup + tid) * (kLists * 8);
#pragma unroll
  for (int a = 0; a < kLists; ++a)
#pragma unroll
    for (int i = 0; i < 8; ++i) to[8 * a + i] = d[a][i];
}

//: A PRODUCER WARP, one of a box's two: a slot's 64 entries of its residue in two batches of 32, a register half
//: each
template <int ROLES>
__device__ __forceinline__ void produce_role(uint8_t* smem, uint32_t base, long long* times, const uint32_t* values,
                                             const uint32_t* dense, int windows) {
  const int lane = threadIdx.x & 31, which = (threadIdx.x - kProduceAt) >> 5, box = which >> 1, mine = which & 1;
  const int group = lane >> 3;
  const uint32_t bars = base + kBarsAt, source = blockIdx.x * 2 + box;
  constexpr int kQuads = kFill / 4;
  static_assert(kSlot / 2 == 2 * kFill);
  const uint4* const wide = reinterpret_cast<const uint4*>(values);
  const uint4* const boxes = reinterpret_cast<const uint4*>(dense);
  uint4 value[2][kQuads], bytes;
  //: batch `n` of the call: 8 value instructions of four entries (8 lanes an entry, a lane 16 bytes) into a
  //: register half, two batches ahead; and one instruction of all 32 entries' dense bytes (a lane an entry's 16),
  //: one batch ahead in one set -- 128 registers a thread hold no second
  const auto load_bytes = [&](int n) {
    const uint32_t tb = token_of(source, mine + 2 * (n * kFill + lane));
    if constexpr (ROLES & kNoLoads) bytes = make_uint4(tb, tb, tb, tb);
    else bytes = boxes[tb];
  };
  const auto load = [&](int h, int qd, int n) {
    const uint32_t tk = token_of(source, mine + 2 * (n * kFill + 4 * qd + group));
    if constexpr (ROLES & kNoLoads) value[h][qd] = make_uint4(tk, tk, tk, tk);
    else value[h][qd] = wide[static_cast<long long>(tk) * (kRowWords / 4) + (lane & 7)];
  };
#pragma unroll
  for (int qd = 0; qd < kQuads; ++qd) {
    load(0, qd, 0);
    load(1, qd, 1);
  }
  load_bytes(0);
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int w = 0; w < windows; ++w) {
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
      if (w > 0) lab::mbar_wait(bars + kPoolFreed + 8 * (2 * box + slot), (w - 1) & 1);
      uint8_t* const tiles = smem + (2 * box + slot) * kPoolBytes;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int qd = 0; qd < kQuads; ++qd) {
          //: entry `e` of the slot: list `e / 16`, token `e % 16` of its step
          const int e = mine + 2 * (h * kFill + 4 * qd + group);
          *reinterpret_cast<uint4*>(tiles + (e / kEntries) * kTile + (lane & 7) * kBrick + (e % kEntries) * 16) =
              value[h][qd];
          //: the register's next load at once: an address is formed as its load is issued, none held for a batch
          load(h, qd, (2 * w + slot) * 2 + h + 2);
        }
        //: the lane's entry: its 16 bytes times the entry's, 8 pairs of bf16, a brick's line each four
        const int e = mine + 2 * (h * kFill + lane);
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
        //: lists that share banks take opposite bricks first: each store covers the 32 banks evenly
        const int first = (e / kEntries) & 1;
        uint8_t* const at = tiles + (e / kEntries) * kTile + kValueTile + (e % kEntries) * 16;
        *reinterpret_cast<uint4*>(at + first * kBrick) = first ? line[1] : line[0];
        *reinterpret_cast<uint4*>(at + (first ^ 1) * kBrick) = first ? line[0] : line[1];
        load_bytes((2 * w + slot) * 2 + h + 1);
      }
      lab::bulk_fence();
      __syncwarp();
      if (lane == 0) lab::mbar_arrive(bars + kPoolReady + 8 * (2 * box + slot));
    }
  }
  clock.stop(times);
}

//: THE READOUT WARPGROUP, the read side whole: a chunk is 4 read digits of 16 tokens -- their coefficients expanded
//: into its own tiles, a digit's step on each box's image tile into one accumulator, and the tokens' lines added
//: to the output from its registers
template <int ROLES>
__device__ __forceinline__ void read_role(uint8_t* smem, uint32_t base, long long* times, int* sinks,
                                          const uint32_t* dense, uint32_t* out, int windows) {
  const int tid = threadIdx.x - kReadAt, warp = tid >> 5, lane = tid & 31, q = lane >> 2, t = lane & 3;
  const uint32_t bars = base + kBarsAt;
  //: THE COEFFICIENTS' SHARE: a thread a (digit `j`, token `n`) of the chunk for one box
  const int pair = tid & 63, j_mine = pair >> 4, n_mine = pair & 15, box_mine = tid >> 6;
  const uint4* const boxes = reinterpret_cast<const uint4*>(dense);
  uint8_t* const tile =
      smem + kCoefAt + (2 * j_mine + box_mine) * kCoefBytes + 16 * ((n_mine % 8) + 16 * (n_mine / 8));
  //: the reader of the thread's (digit, token) in chunk `c` of the call, and its bytes on the thread's box
  const auto reader = [&](int c) {
    const int k = c % kChunks;
    return reader_of(blockIdx.x, (c / kChunks) * kWindow + 16 * (8 * (k >> 1) + (k & 1) + 2 * j_mine) + n_mine);
  };
  uint4 bytes = boxes[reader_row(reader(0), box_mine)];
  //: THE LINES' SHARE: a thread's rows are `16 warp + q` and that + 8; a lane of even `q` issues row 0's pair with
  //: its neighbour's, one of odd `q` row 1's
  const bool odd = q & 1;
  const uint32_t word = static_cast<uint32_t>(odd ? (16 * warp + q + 7) / 2 : (16 * warp + q) / 2);
  uint32_t kept = 0;
  //: a digit's first step writes its accumulator whole: nothing is cleared between chunks
  float d[kChunkDigits][8];
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int c = 0; c < kChunks * windows; ++c) {
    const int w = c / kChunks, c4 = c % kChunks;
    //: chunk `2 h + c`'s digits are `8 h + c + 2 j`: its tile group, published for this window
    lab::mbar_wait(bars + kTilesReady + 8 * c4, w & 1);
    //: THE COEFFICIENTS: the reader's 16 bytes (loaded a chunk before) times its byte on the digit, 8 pairs of
    //: bf16, a line each four; then the next chunk's bytes into the same registers
    const float weight = static_cast<float>(sparse_of(reader(c)));
    const uint32_t words[4] = {bytes.x, bytes.y, bytes.z, bytes.w};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      uint32_t two[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t both = words[2 * half + i / 2] >> (16 * (i & 1));
        two[i] = lab::bf16x2_round(weight * static_cast<float>(both & 255u),
                                   weight * static_cast<float>(both >> 8 & 255u));
      }
      *reinterpret_cast<uint4*>(tile + 16 * 8 * half) = make_uint4(two[0], two[1], two[2], two[3]);
    }
    bytes = boxes[reader_row(reader(c + 1), box_mine)];
    //: the writers' fence: the steps read these stores through the tensor pipe's own view of shared memory
    lab::bulk_fence();
    const uint32_t first = static_cast<uint32_t>(8 * (c4 >> 1) + (c4 & 1));
    const uint32_t images = lab::held(base + kImageAt + first * kImageTile);
    const uint32_t coefs = lab::held(base + kCoefAt);
    lab::wg_fence();
#pragma unroll
    for (int j = 0; j < kChunkDigits; ++j)
#pragma unroll
      for (int box = 0; box < 2; ++box)
        lab::wg_bf16_n16_ss_mna(d[j], lab::wg_desc(images + box * kBoxBytes + 2 * j * kImageTile, 8, kBrickLines),
                                lab::wg_desc(coefs + (2 * j + box) * kCoefBytes, 8, 16), box != 0);
    lab::wg_commit();
    lab::wg_wait<0>();
    //: the group is read: its box may publish the next window's over it
    if (lane == 0) lab::mbar_arrive(bars + kTilesRead + 8 * c4);
    //: THE LINES: digit `j`'s token `8 h + 2 t + cc` of read digit `first + 2 j`, its row 16 of that digit on
    const long long rows = static_cast<long long>(blockIdx.x) * (kOutWindows * kWindow) + (w % kOutWindows) * kWindow
                           + 16 * first + 2 * t;
#pragma unroll
    for (int j = 0; j < kChunkDigits; ++j)
#pragma unroll
      for (int hc = 0; hc < 4; ++hc) {
        const int h = hc >> 1, cc = hc & 1;
        const float own = odd ? d[j][cc | 2 | h << 2] : d[j][cc | h << 2];
        const float got = __shfl_xor_sync(0xffffffffu, odd ? d[j][cc | h << 2] : d[j][cc | 2 | h << 2], 4);
        const uint32_t line = lab::pack_half2(odd ? got : own, odd ? own : got);
        if constexpr (ROLES & kNoAdds) kept ^= line + hc;
        else lab::red_add_f16x2(out + (rows + 32 * j + 8 * h + cc) * (kChannels / 2) + word, line);
      }
  }
  clock.stop(times);
  if constexpr (ROLES & kNoAdds) lab::sink(sinks, kept);
}

//: A ROLE A ROW LEAVES OUT: its warps' spans are empty, at the roles' start, so the block's span is the roles' own
__device__ __forceinline__ void idle_role(long long* times) {
  lab::WarpSpan clock;
  clock.start();
  clock.stop(times);
}

template <int ROLES>
__global__ void __launch_bounds__(kThreads, 1)
    window(long long* times, int* sinks, const uint32_t* values, const uint32_t* dense, float* state, uint32_t* out,
           int windows) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x;
  const uint32_t base = lab::smem(smem);
  //: the images cleared: the state is zero at the call's start
#pragma unroll 1
  for (int i = tid; i < 2 * kBoxBytes / 4; i += kThreads) *reinterpret_cast<uint32_t*>(smem + kImageAt + 4 * i) = 0u;
  if (tid == 0) {
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      lab::mbar_init(base + kBarsAt + kPoolReady + 8 * p, 2);
      lab::mbar_init(base + kBarsAt + kPoolFreed + 8 * p, 1);
    }
#pragma unroll
    for (int k = 0; k < kChunks; ++k) {
      lab::mbar_init(base + kBarsAt + kTilesReady + 8 * k, 2 * kGroup / 32);
      lab::mbar_init(base + kBarsAt + kTilesRead + 8 * k, kGroup / 32);
    }
  }
  lab::bulk_fence();
  __syncthreads();

  //: EACH ROLE'S WHOLE BODY IN ITS OWN BRANCH, a warpgroup's registers set at the branch's head in every row: the
  //: folds' raise is the readout's release (docs/lab/carry_window.md)
  if (tid < kReadAt) {
    lab::regs_up<kFoldRegs>();
    fold_role<ROLES>(smem, base, times, state, windows);
  } else if (tid < kProduceAt) {
    lab::regs_down<kReadRegs>();
    if constexpr (ROLES & kRead) read_role<ROLES>(smem, base, times, sinks, dense, out, windows);
    else idle_role(times);
  } else {
    if constexpr (ROLES & kFold) produce_role<ROLES>(smem, base, times, values, dense, windows);
    else idle_role(times);
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
void run(lab::Bench& b, const uint32_t* values, const uint32_t* dense, float* state, uint32_t* out) {
  using namespace carry_window;
  constexpr int kTimed = 64;
  const std::string name = ROLES == kFold ? "the fold's side alone: producers and steps"
                           : ROLES == kRead ? "the read side alone: the image and the readout"
                           : ROLES == (kFold | kRead) ? "the carry: both sides"
                           : ROLES & kNoLoads ? "both sides, the producers' global loads taken out"
                           : ROLES & kNoAdds ? "both sides, the readout's global adds taken out"
                           : "both sides, the image's stores taken out";
  const auto kernel = window<ROLES>;
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  const size_t words = static_cast<size_t>(b.sms) * kOutWindows * kWindow * (kChannels / 2);
  LAB_CHECK(cudaMemset(out, 0, words * 4));
  if (ROLES == (kFold | kRead)) {
    kernel<<<b.sms, kThreads, kSmemBytes>>>(b.times, b.sinks, values, dense, state, out, kOutWindows);
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
  const lab::Samples cycles = b.variant(
      [&] { kernel<<<b.sms, kThreads, kSmemBytes>>>(b.times, b.sinks, values, dense, state, out, kTimed); },
      [&] { return b.span(kThreads >> 5) / kTimed; });
  b.record(name, "cycles a window a CTA", cycles);
}

int main(int argc, char** argv) {
  using namespace carry_window;
  lab::Bench b(argc, argv);
  for (int roles : {1, 2, 3, 3 | kNoLoads, 3 | kNoAdds, 3 | kNoImage}) {
    const std::string k = lab::kernel("window", {roles});
    //: with no fold, the state's one group of steps is a rolled loop: one instruction
    b.expect(k, "HGMMA", "==", (roles & kFold ? 2 * kSlotLists : 1) + (roles & kRead ? 2 * kChunkDigits : 0));
    b.exempt(k, "a composition probe: the carry's roles in one CTA, no kernel's step");
    //: once a slot or a chunk a role: a `__syncwarp` before a one-lane arrival is a convergence call, and each
    //: barrier's polled wait a region -- five roles' in one kernel
    b.expect(k, "LOOP_CONVERGENCE", "<=", 64);
    //: the roles' loops hold their steps beside their barriers' polled waits: the waits' branches are the steps'
    b.expect(k, "BSSY", ">=", 0);
    //: the readout is the read side whole: its trip holds its coefficients' rounds and load and its lines' rounds
    //: and adds beside its steps, by design -- the tensor pipe is idle most of a window
    if (roles & kRead) {
      b.expect(k, "STEP_XU", ">=", 0);
      b.expect(k, "STEP_GLOBAL", ">=", 0);
    }
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
  run<kFold>(b, values, dense, state, out);
  run<kRead>(b, values, dense, state, out);
  run<kFold | kRead>(b, values, dense, state, out);
  run<kFold | kRead | kNoLoads>(b, values, dense, state, out);
  run<kFold | kRead | kNoAdds>(b, values, dense, state, out);
  run<kFold | kRead | kNoImage>(b, values, dense, state, out);
  return 0;
}
