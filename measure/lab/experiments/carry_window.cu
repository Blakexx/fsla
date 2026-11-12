// measure/lab/experiments/carry_window.cu -- QUESTION: the carry's roles were each measured apart (`pool_fill`,
// `state_image`, `readout_role`); what does a window cost a CTA when they run TOGETHER, under one register file,
// one tensor pipe and one shared memory? METHOD: a CTA an SM of five warpgroups, registers traded by role. TWO
// FOLD warpgroups, a 16 x 16 box each, its state 16 accumulators in registers. FOUR PRODUCER warps, two a box,
// fill a slot's 8 lists of 16 entries from scattered rows (`pool_fill`'s form) and the fold runs the slot's 8
// steps. A window is a box's 256 entries, and 256 tokens read the pair: at its start each fold warpgroup
// PUBLISHES its state as a bf16 image, 16 tiles; ONE READOUT warpgroup runs a read digit's step on both boxes'
// tiles into one accumulator, 4 digits a chunk, and stores the 64 tokens' rows to a panel; FOUR DRAIN warps add
// each token's row as one f16 line. Rows: the fold's side alone, the read side alone, and both. Both: the state
// and every output row are checked exactly.
// TIMELINE: a fold warpgroup's trip is a window: the wait on the readout's last window, 128 stores a thread and
// their fence, the image's arrival; then a slot twice: the wait on its producers, the fence, 8 steps, the commit
// and its wait, the slot freed. A producer's trip is a batch of 32 entries, a slot two of them: the batch's
// stores and multiplies (its words loaded a trip before), then the same registers' loads for the next batch; a
// slot on, the fence and the arrival. The readout's trip is a chunk: the wait on the panel two chunks back, the
// fence, 8 steps, the commit and its wait, 32 stores a thread, the arrival; a window's image waited on at its
// first chunk and released after its last. A drain warp's trip is a chunk, its 16 tokens: the wait on the panel,
// 16 rows loaded, their rounds and adds, the arrival (its loads wait in the trip: `LOAD_IN_TRIP` is of global
// loads, and these are the panel's). Live: a fold thread's
// 128 accumulators; a producer's 36 words; the readout's 32; a drain's 32.
// see docs/internals/tools/lab.md#carry-window
#include "lab.cuh"

namespace carry_window {

constexpr int kLists = 16, kEntries = 16, kWindow = kLists * kEntries, kSlotLists = 8, kSlot = kSlotLists * kEntries;
constexpr int kChannels = 64, kTableTokens = 65536, kRowWords = 32, kFill = 32;
constexpr int kGroup = 128, kDrains = 4, kThreads = 5 * kGroup;
//: the roles' first threads: fold 0, fold 1, the readout, the producers, the drains
constexpr int kReadAt = 2 * kGroup, kProduceAt = 3 * kGroup, kDrainAt = 4 * kGroup;
//: THE REGISTERS A THREAD, by role, within the SM's file of 65,536
constexpr int kFoldRegs = 160, kReadRegs = 64, kProduceRegs = 80, kDrainRegs = 48;
static_assert(2 * kGroup * kFoldRegs + kGroup * (kReadRegs + kProduceRegs + kDrainRegs) == 65536);

//: A POOL TILE (the fold's operands a list), M|N-major: 8 bricks of values and 2 of coefficients, a brick 17 lines
constexpr int kBrick = 17 * 16, kValueTile = 8 * kBrick, kTile = 10 * kBrick, kPoolBytes = kSlotLists * kTile;
//: AN IMAGE TILE (the readout's operand a read digit): a line a fold digit's 8 channels, bricks 17 lines, tiles 137
constexpr int kBrickLines = 17, kImageTile = 137 * 16, kBoxBytes = kLists * kImageTile, kCoefBytes = 512;
//: a panel row: a token's 64 fp32 and a pad, rows 272 bytes apart
constexpr int kChunkDigits = 4, kTokens = 16 * kChunkDigits, kChunks = kLists / kChunkDigits, kRowBytes = 272,
              kPanelBytes = kTokens * kRowBytes, kBatch = 8;
constexpr int kImageAt = 4 * kPoolBytes, kCoefAt = kImageAt + 2 * kBoxBytes, kPanelAt = kCoefAt + kLists * kCoefBytes,
              kBarsAt = kPanelAt + 2 * kPanelBytes, kSmemBytes = kBarsAt + 112;
//: the barriers, 8 bytes each: a pool slot filled and freed (box, slot), the image published, the window read, a
//: panel filled and drained
constexpr int kPoolReady = 0, kPoolFreed = 32, kImaged = 64, kReadDone = 72, kPanelReady = 80, kPanelFreed = 96;
constexpr int kOutWindows = 4;
enum Roles : int { kFold = 1, kRead = 2 };

//: the token of a (CTA, box, entry of the call), its value at a channel, its byte on its list's digit and on a
//: dense leaf; a read coefficient at (read digit, fold digit, token): small integers
constexpr uint32_t token_of(uint32_t source, uint32_t entry) {
  return ((source * 4099u + entry) * 2654435761u >> 12) & (kTableTokens - 1);
}
constexpr int value_of(int t, int c) { return (t + c) % 2 + 1; }
constexpr int sparse_of(int t) { return t % 2 + 1; }
constexpr int dense_of(int t, int b) { return (t + b) % 2 + 1; }
constexpr int coef_of(int b, int a, int n) { return (a + 2 * n + 3 * b) % 3 + 1; }
constexpr int coef_at(int n, int k) { return 16 * ((n % 8) + 16 * (n / 8) + 8 * (k / 8)) + 2 * (k % 8); }

__device__ __forceinline__ uint16_t bf16_of(int v) {
  return static_cast<uint16_t>(__float_as_uint(static_cast<float>(v)) >> 16);
}

//: A FOLD WARPGROUP: its box's state; a window is the image, then two slots of 8 steps
template <int ROLES>
__device__ __forceinline__ void fold_role(uint8_t* smem, uint32_t base, long long* times, float* state, int windows) {
  lab::regs_up<kFoldRegs>();
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
      //: THE IMAGE: value (channel m, fold digit a) of read digit b to tile b, line `(m / 8) 17 + a`, slot `m % 8`,
      //: once the readout has left the window before
      //: the first window's image is the cleared shared memory: the state is zero
      if (w > 0) {
        lab::mbar_wait(bars + kReadDone, (w - 1) & 1);
#pragma unroll
        for (int a = 0; a < kLists; ++a)
#pragma unroll
          for (int hc = 0; hc < 4; ++hc)
#pragma unroll
            for (int r = 0; r < 2; ++r)
              *reinterpret_cast<uint16_t*>(mine + (8 * (hc >> 1) + (hc & 1)) * kImageTile + 16 * a + 16 * r * kBrickLines)
                  = static_cast<uint16_t>(__float_as_uint(d[a][(hc & 1) | r << 1 | (hc >> 1) << 2]) >> 16);
      }
      lab::bulk_fence();
      __syncwarp();
      if (lane == 0) lab::mbar_arrive(bars + kImaged);
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

//: A PRODUCER WARP: one of a box's two; a slot's 64 entries of its residue, two batches of 32
__device__ __forceinline__ void produce_role(uint8_t* smem, uint32_t base, long long* times, const uint32_t* values,
                                             const uint32_t* dense, int windows) {
  lab::regs_down<kProduceRegs>();
  const int lane = threadIdx.x & 31, which = (threadIdx.x - kProduceAt) >> 5, box = which >> 1, mine = which & 1;
  const int group = lane >> 3;
  const uint32_t bars = base + kBarsAt, source = blockIdx.x * 2 + box;
  constexpr int kQuads = kFill / 4;
  const uint4* const wide = reinterpret_cast<const uint4*>(values);
  const uint4* const boxes = reinterpret_cast<const uint4*>(dense);
  //: ONE batch in registers, the register file's share: 80 a thread hold no second
  uint4 value[kQuads], bytes;
  //: batch `n` of the call: 8 value instructions of four entries (8 lanes an entry, a lane 16 bytes), and one of
  //: all 32 entries' dense bytes (a lane an entry's 16)
  const auto load = [&](int n) {
#pragma unroll
    for (int qd = 0; qd < kQuads; ++qd) {
      const uint32_t tk = token_of(source, mine + 2 * (n * kFill + 4 * qd + group));
      value[qd] = wide[static_cast<long long>(tk) * (kRowWords / 4) + (lane & 7)];
    }
    bytes = boxes[token_of(source, mine + 2 * (n * kFill + lane))];
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
#pragma unroll
      for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int qd = 0; qd < kQuads; ++qd) {
          //: entry `e` of the slot: list `e / 16`, token `e % 16` of its step
          const int e = mine + 2 * (h * kFill + 4 * qd + group);
          *reinterpret_cast<uint4*>(tiles + (e / kEntries) * kTile + (lane & 7) * kBrick + (e % kEntries) * 16) =
              value[qd];
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
        load((2 * w + slot) * 2 + h + 1);
      }
      lab::bulk_fence();
      __syncwarp();
      if (lane == 0) lab::mbar_arrive(bars + kPoolReady + 8 * (2 * box + slot));
    }
  }
  clock.stop(times);
}

//: THE READOUT WARPGROUP: a chunk is 4 read digits, a digit's step on each box's tile into one accumulator
__device__ __forceinline__ void read_role(uint8_t* smem, uint32_t base, long long* times, int windows) {
  lab::regs_down<kReadRegs>();
  const int warp = (threadIdx.x >> 5) & 3, lane = threadIdx.x & 31, q = lane >> 2, t = lane & 3;
  const uint32_t bars = base + kBarsAt;
  const uint32_t mine = lab::held(static_cast<uint32_t>(kPanelAt + 2 * t * kRowBytes + (16 * warp + q) * 4));
  //: a digit's first step writes its accumulator whole: nothing is cleared between chunks
  float d[kChunkDigits][8];
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int w = 0; w < windows; ++w) {
    lab::mbar_wait(bars + kImaged, w & 1);
#pragma unroll 1
    for (int c4 = 0; c4 < kChunks; ++c4) {
      const int c = kChunks * w + c4;
      if (c >= 2) lab::mbar_wait(bars + kPanelFreed + 8 * (c & 1), ((c >> 1) - 1) & 1);
      const uint32_t images = lab::held(base + kImageAt + c4 * kChunkDigits * kImageTile);
      const uint32_t coefs = lab::held(base + kCoefAt + c4 * kChunkDigits * kCoefBytes);
      lab::wg_fence();
#pragma unroll
      for (int j = 0; j < kChunkDigits; ++j)
#pragma unroll
        for (int box = 0; box < 2; ++box)
          lab::wg_bf16_n16_ss_mna(d[j], lab::wg_desc(images + box * kBoxBytes + j * kImageTile, 8, kBrickLines),
                                  lab::wg_desc(coefs + j * kCoefBytes, 8, 16), box != 0);
      lab::wg_commit();
      lab::wg_wait<0>();
      //: THE PANEL: value (channel, token) to the token's row, a store a register
      uint8_t* const row = smem + mine + (c & 1) * kPanelBytes;
#pragma unroll
      for (int j = 0; j < kChunkDigits; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i)
          *reinterpret_cast<float*>(row + (16 * j + 8 * (i >> 2) + (i & 1)) * kRowBytes + 32 * ((i >> 1) & 1)) = d[j][i];
      __syncwarp();
      if (lane == 0) lab::mbar_arrive(bars + kPanelReady + 8 * (c & 1));
    }
    //: the window's image is read: its box may publish the next
    if (lane == 0) lab::mbar_arrive(bars + kReadDone);
  }
  clock.stop(times);
}

//: A DRAIN WARP: a chunk's tokens of its residue; a lane holds a token's two channels and adds one f16 pair
__device__ __forceinline__ void drain_role(const uint8_t* smem, uint32_t base, long long* times, uint32_t* out,
                                           int windows) {
  lab::regs_down<kDrainRegs>();
  const int lane = threadIdx.x & 31, mine = (threadIdx.x - kDrainAt) >> 5;
  const uint32_t bars = base + kBarsAt;
  static_assert(kTokens / kDrains == 2 * kBatch);
  float lo[2][kBatch], hi[2][kBatch];
  const auto load = [&](int h, int c, int batch) {
#pragma unroll
    for (int i = 0; i < kBatch; ++i) {
      const float* const row = reinterpret_cast<const float*>(
          smem + kPanelAt + (c & 1) * kPanelBytes + (mine + kDrains * (batch * kBatch + i)) * kRowBytes) + 2 * lane;
      lo[h][i] = row[0];
      hi[h][i] = row[1];
    }
  };
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int c = 0; c < kChunks * windows; ++c) {
    //: a chunk is this warp's two batches: both loaded as the chunk opens, nothing ahead of them
    lab::mbar_wait(bars + kPanelReady + 8 * (c & 1), (c >> 1) & 1);
    load(0, c, 0);
    load(1, c, 1);
    const long long first = static_cast<long long>(blockIdx.x) * (kOutWindows * kWindow)
                            + ((c / kChunks) % kOutWindows) * kWindow + (c % kChunks) * kTokens;
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int i = 0; i < kBatch; ++i) {
        const int token = mine + kDrains * (h * kBatch + i);
        lab::red_add_f16x2(out + (first + token) * (kChannels / 2) + lane, lab::pack_half2(lo[h][i], hi[h][i]));
      }
    __syncwarp();
    if (lane == 0) lab::mbar_arrive(bars + kPanelFreed + 8 * (c & 1));
  }
  clock.stop(times);
}

template <int ROLES>
__global__ void __launch_bounds__(kThreads, 1)
    window(long long* times, const uint32_t* values, const uint32_t* dense, float* state, uint32_t* out,
           int windows) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x;
  const uint32_t base = lab::smem(smem);
  //: the images cleared (the state is zero at the call's start), and the read coefficients, a tile a read digit
#pragma unroll 1
  for (int i = tid; i < 2 * kBoxBytes / 4; i += kThreads) *reinterpret_cast<uint32_t*>(smem + kImageAt + 4 * i) = 0u;
#pragma unroll 1
  for (int i = tid; i < kLists * kLists * kLists; i += kThreads) {
    const int b = i / (kLists * kLists), a = (i / kLists) % kLists, n = i % kLists;
    *reinterpret_cast<uint16_t*>(smem + kCoefAt + b * kCoefBytes + coef_at(n, a)) = bf16_of(coef_of(b, a, n));
  }
  if (tid == 0) {
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      lab::mbar_init(base + kBarsAt + kPoolReady + 8 * p, 2);
      lab::mbar_init(base + kBarsAt + kPoolFreed + 8 * p, 1);
    }
    lab::mbar_init(base + kBarsAt + kImaged, 2 * kGroup / 32);
    lab::mbar_init(base + kBarsAt + kReadDone, kGroup / 32);
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      lab::mbar_init(base + kBarsAt + kPanelReady + 8 * p, kGroup / 32);
      lab::mbar_init(base + kBarsAt + kPanelFreed + 8 * p, kDrains);
    }
  }
  lab::bulk_fence();
  __syncthreads();

  //: EACH ROLE'S WHOLE BODY IN ITS OWN BRANCH, its registers set at its head (docs/lab/reg_trade.md). A role a
  //: row leaves out still releases its registers: the fold's raise waits on every other warpgroup's
  if (tid < kReadAt) {
    fold_role<ROLES>(smem, base, times, state, windows);
  } else if (tid < kProduceAt) {
    if constexpr (ROLES & kRead) read_role(smem, base, times, windows);
    else lab::regs_down<kReadRegs>();
  } else if (tid < kDrainAt) {
    if constexpr (ROLES & kFold) produce_role(smem, base, times, values, dense, windows);
    else lab::regs_down<kProduceRegs>();
  } else {
    if constexpr (ROLES & kRead) drain_role(smem, base, times, out, windows);
    else lab::regs_down<kDrainRegs>();
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
                           : ROLES == kRead ? "the read side alone: the image, the readout and the drain"
                           : "the carry: both sides";
  const auto kernel = window<ROLES>;
  LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  const size_t words = static_cast<size_t>(b.sms) * kOutWindows * kWindow * (kChannels / 2);
  LAB_CHECK(cudaMemset(out, 0, words * 4));
  if (ROLES & kFold && ROLES & kRead) {
    kernel<<<b.sms, kThreads, kSmemBytes>>>(b.times, values, dense, state, out, kOutWindows);
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
            const int bb = token / 16, n = token % 16;
            long long want = 0;
            for (int box = 0; box < 2; ++box)
              for (int a = 0; a < kLists; ++a) want += image_host(s[at(box, m, a, bb)]) * coef_of(bb, a, n);
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
    b.unchecked(name, "an isolation: one side's roles with the other's absent");
  }
  const lab::Samples cycles = b.variant(
      [&] { kernel<<<b.sms, kThreads, kSmemBytes>>>(b.times, values, dense, state, out, kTimed); },
      [&] { return b.span(kThreads >> 5) / kTimed; });
  b.record(name, "cycles a window a CTA", cycles);
}

int main(int argc, char** argv) {
  using namespace carry_window;
  lab::Bench b(argc, argv);
  for (int roles : {1, 2, 3}) {
    const std::string k = lab::kernel("window", {roles});
    //: with no fold, the state's one group of steps is a rolled loop: one instruction
    b.expect(k, "HGMMA", "==", (roles & kFold ? 2 * kSlotLists : 1) + (roles & kRead ? 2 * kChunkDigits : 0));
    //: the readout's panel is read by the drain warps' own loads, not by a step: its arrival needs no fence
    if (roles & kRead) b.expect(k, "STORE_UNFENCED", "<=", 1);
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
  run<kFold>(b, values, dense, state, out);
  run<kRead>(b, values, dense, state, out);
  run<kFold | kRead>(b, values, dense, state, out);
  return 0;
}
