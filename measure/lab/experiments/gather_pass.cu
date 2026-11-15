// measure/lab/experiments/gather_pass.cu -- QUESTION: `facts_lists` moved a live (token, digit) pair's lines to its
// place in its digit's steps by its token's thread, a load and a store of global memory a line, and it cost twice
// the carry's window. What does the same re-arrangement cost when every byte crosses global memory in a BULK COPY
// and a line is moved in SHARED memory -- and what is the floor, the same copies with no line moved? METHOD: a
// window of 1,024 tokens, one side's sparse level of 256 digits; a window's STEP LISTS are handed (the lists pass's
// output padded to steps of 16: an entry its token and its digit's low 4 bits, a pad the token 1,024). The pass
// writes two planes. THE VALUES PLANE, a step a tile of 137 lines -- 8 bricks of 17 lines, line `k` of brick `b`
// entry k's channels `8 b` on, then a line of the 16 entries' bytes on the sparse digit: a CTA holds the window's
// values (128 KB) and, a sparse block of 16 digits at a time, that block's rows; a unit is 8 steps of one block.
// THE DENSE PLANE, block-major -- a dense block's 16 lines a step, the steps side by side: a CTA holds one dense
// block's rows of the window (16 KB) and walks every step, 192 a unit, 16 times. A pad's lines are zero: token
// 1,024 is a zero line after each table. Each plane is timed at 256, 512 and 1,024 threads, moved and as a flat
// stream, its rounds the same windows or a fresh window a round, at 4 to 5 live digits a token and at 32. Every
// line and pad of the first round is checked, each byte of the tables its own position's. TWO CONTROLS say what
// binds a flat stream that costs what the moved form does: the same kernels on A QUARTER of the multiprocessors
// (the card's memory, if a CTA's cycles fall), and the dense plane with SIX buffers out of half the size, four
// copies out in flight where three buffers allow one (a CTA's own queue, if they fall).
// THE STREAM, TOKEN-ORDERED, is the same two kernels fed lists BY SPARSE BLOCK, not by digit: a block's live tokens
// once each, in token order, padded to 16 only at the block's end. Its values kernel writes RECORDS -- a token's 8
// value lines and then its line of the block's 16 sparse bytes, 9 lines a token -- which the block's CTAs share;
// its dense plane is the block-major one, a CTA's dense block of its sparse block's tokens one run. No digit's
// step is padded in global memory: a CTA places a token on its digits' steps in shared memory by the mask. A carry's
// CTAs read ONE window's stream at a time, so the pass's planes can be A RING the cache holds: the stream's rows
// on 16 and on 8 multiprocessors, each CTA rewriting its region every round, and the same CTAs writing a new
// region a round -- the same reads, the planes past the cache.
// THE BLOCK STREAM, ITS LISTS THE PASS'S (`blocks`): the pass is handed the level's mask digit-major and no list; a
// block's 16 lists are made a warp a digit in 32 fixed trips, and a unit is a slot's image as the carry lands it --
// tables, then each list's value runs as long as the list -- with its plan. ITS TIMELINE: a window's values and its
// first block's mask and rows are asked before the first trip and waited on in it; a later block's are asked a
// unit ahead. A block's first unit is the lists (a barrier), the layout by nine threads (a barrier), the image and the
// writers' fence (the CTA's barrier); its second the last two. Thread 0 issues a unit's copy out and its plan at
// the next trip and keeps one copy in flight, three buffers in turn.
// THE COEFFICIENT RUNS (`coefs`): the block stream's coefficients made in the carry step's format -- a CTA a window.
// Its timeline: the window's lists first, a sparse block a trip, the block's mask and rows asked a trip ahead, an
// entry kept as its token and its sparse byte; every slot's layout once; then a dense block at a time, its rows
// asked a dense block ahead, a unit a slot's runs written a thread a (list, place, brick), fenced, and copied out
// at the next trip, three buffers in turn and one copy in flight.
// TIMELINE: no MMA. A trip is a unit. Thread 0 issues the copy out of the unit before (each writer fenced its own
// lines before the barrier), then the copies in of the unit AFTER this one -- its step list and, when it opens a
// block, the block's rows -- a unit ahead of their use; every thread waits on this unit's barriers, moves its
// lines shared to shared and fences them; thread 0 waits until all but the last `D - 2` copies out are done, and
// the CTA's barrier. `D` buffers out in turn (3; the dense plane's also 6): the one being filled, the one being
// copied, and those whose copies may be in flight. A window's values are asked at its first trip and used a trip
// later. Live: a unit's coordinates.
// see docs/internals/tools/lab.md#gather-pass
#include "lab.cuh"

namespace gather_pass {

constexpr int kWindow = 1024, kTile = 64, kTiles = kWindow / kTile, kDigits = 256, kBlocks = kDigits / 16;
//: a level's tile-major rows: a tile's block is its 64 tokens' lines of 16 digits
constexpr int kBlockBytes = kTile * 16, kTileBytes = kBlocks * kBlockBytes;
//: one block of a window in shared memory, a line a token, and the zero line a pad reads
constexpr int kLevelBytes = (kWindow + 1) * 16;
constexpr int kValueBytes = 128, kTileLines = 8 * 17 + 1, kSparseLine = kTileLines - 1;
constexpr uint32_t kPad = kWindow, kToken = 0x7ffu;
//: a window's steps have room for every digit's last, part-filled one
template <int LIVE>
constexpr int kMaxSteps = kWindow * LIVE / 16 + kDigits;
enum Move : int { kFlat = 0, kMoved = 1 };
//: what a step of the values kernel's plane is: a tile of 16 entries in bricks, or 16 tokens' records of 9 lines
enum Layout : int { kTiled = 0, kRecords = 1 };
template <int FORM>
constexpr int kLines = FORM == kRecords ? 16 * 9 : 8 * 17 + 1;

//: A UNIT: `steps` steps from `step` of the window's, of sparse block (values) or for dense block (dense) `block`
struct Unit {
  int block, step, steps;
};

// ------------------------------------------------------------------------------------------- the values plane
namespace values {
constexpr int kUnit = 8, kOutBytes = kUnit * kLines<kRecords> * 16;
constexpr int kValAt = 0, kSparseAt = kValAt + (kWindow + 1) * kValueBytes, kListAt = kSparseAt + 2 * kLevelBytes,
              kOutAt = kListAt + 2 * kUnit * 32, kFirstAt = kOutAt + 3 * kOutBytes, kBarsAt = kFirstAt + 80,
              kSmemBytes = kBarsAt + 5 * 8;
enum Bar : int { kValues = 0, kSparse = 8, kList = 24 };

//: the unit at step `s`, `g` a block not past it: the steps left of its sparse block, 8 at most
__device__ __forceinline__ Unit unit_at(const uint32_t* first, int g, int s) {
  if (s >= static_cast<int>(first[kBlocks])) return {kBlocks, s, 0};
#pragma unroll 1
  while (static_cast<int>(first[g + 1]) <= s) ++g;
  return {g, s, min(kUnit, static_cast<int>(first[g + 1]) - s)};
}

template <int THREADS, int MOVE, int FORM>
__global__ void __launch_bounds__(THREADS, 1)
    plane(long long* times, const uint4* vals, const uint4* rows, const uint4* lists, const uint32_t* firsts,
          uint4* out_plane, int rounds, int stride, int max_steps, int out_stride) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x;
  const uint32_t base = lab::smem(smem), bars = base + kBarsAt;
  uint4* const val = reinterpret_cast<uint4*>(smem + kValAt);
  uint32_t* const first = reinterpret_cast<uint32_t*>(smem + kFirstAt);
  if (tid < 8) val[kWindow * 8 + tid] = make_uint4(0, 0, 0, 0);
  if (tid < 2) reinterpret_cast<uint4*>(smem + kSparseAt + tid * kLevelBytes)[kWindow] = make_uint4(0, 0, 0, 0);
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < 5; ++i) lab::mbar_init(bars + 8 * i, 1);
    lab::bulk_fence();
  }
  __syncthreads();
  //: the loads so far, a slot and a phase each: of step lists, of sparse blocks, of windows' values
  uint32_t list_n = 0xffffffffu, sparse_n = 0xffffffffu, values_n = 0xffffffffu;
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int round = 0; round < rounds; ++round) {
    const long long at = static_cast<long long>(blockIdx.x) + static_cast<long long>(round) * stride;
    const uint4* const wvals = vals + at * (kWindow * kValueBytes / 16);
    const uint4* const wrows = rows + at * (kWindow * kDigits / 16);
    const uint4* const wlists = lists + at * max_steps * 2;
    //: a CTA's region of the plane: one it rewrites every round (a ring), or a new one a round
    uint4* const wplane = out_plane + (static_cast<long long>(blockIdx.x) + static_cast<long long>(round) * out_stride) * max_steps * kLines<FORM>;
    if (tid <= kBlocks) first[tid] = firsts[at * (kBlocks + 1) + tid];
    if (tid == 0) lab::bulk_wait<1>();
    __syncthreads();
    Unit prev{kBlocks, 0, 0}, cur{kBlocks, 0, 0}, next = unit_at(first, 0, 0);
    uint32_t prev_slot = 0;
    bool opens = false, heads = false;
    if (next.steps) {
      ++values_n;
      if (tid == 0) {
        lab::mbar_expect(bars + kValues, kWindow * kValueBytes);
#pragma unroll
        for (int t = 0; t < kTiles; ++t)
          lab::bulk_load(base + kValAt + t * (kTile * kValueBytes), wvals + t * (kTile * kValueBytes / 16),
                         kTile * kValueBytes, bars + kValues);
      }
    }
#pragma unroll 1
    for (;;) {
      //: THE ISSUE EDGE: the unit before's tiles out, the unit after's list and rows in
      if (tid == 0) {
        if (prev.steps) {
          lab::bulk_store(wplane + static_cast<long long>(prev.step) * kLines<FORM>, base + kOutAt + prev_slot * kOutBytes,
                          prev.steps * kLines<FORM> * 16);
          lab::bulk_commit();
        }
        if (next.steps) {
          const uint32_t slot = (list_n + 1) & 1u;
          lab::mbar_expect(bars + kList + 8 * slot, next.steps * 32);
          lab::bulk_load(base + kListAt + slot * kUnit * 32, wlists + next.step * 2, next.steps * 32, bars + kList + 8 * slot);
          if (next.block != cur.block) {
            const uint32_t level = (sparse_n + 1) & 1u;
            lab::mbar_expect(bars + kSparse + 8 * level, kTiles * kBlockBytes);
#pragma unroll
            for (int t = 0; t < kTiles; ++t)
              lab::bulk_load(base + kSparseAt + level * kLevelBytes + t * kBlockBytes,
                             wrows + t * (kTileBytes / 16) + next.block * (kBlockBytes / 16), kBlockBytes,
                             bars + kSparse + 8 * level);
          }
        }
      }
      if (!cur.steps && !next.steps) break;
      if (cur.steps) {
        //: THE LANDED EDGE: this unit's list; its block's rows and its window's values where it opens them
        lab::mbar_wait_suspended(bars + kList + 8 * (list_n & 1u), (list_n >> 1) & 1u);
        if (opens) lab::mbar_wait_suspended(bars + kSparse + 8 * (sparse_n & 1u), (sparse_n >> 1) & 1u);
        if (heads) lab::mbar_wait_suspended(bars + kValues, values_n & 1u);
        if constexpr (MOVE == kMoved) {
          const uint16_t* const list = reinterpret_cast<const uint16_t*>(smem + kListAt + (list_n & 1u) * kUnit * 32);
          const uint8_t* const sparse = smem + kSparseAt + (sparse_n & 1u) * kLevelBytes;
          uint4* const out = reinterpret_cast<uint4*>(smem + kOutAt + (list_n % 3u) * kOutBytes);
          if constexpr (FORM == kTiled) {
            //: A LINE a (step, brick, entry): the entry's token's value line of the brick; a brick-0 thread its byte too
  #pragma unroll
            for (int i = tid; i < kUnit * 128; i += THREADS) {
              const int j = i >> 7, brick = (i >> 4) & 7, k = i & 15;
              if (j < cur.steps) {
                const uint32_t e = list[j * 16 + k], token = e & kToken;
                out[j * kTileLines + brick * 17 + k] = val[token * 8 + brick];
                if (brick == 0)
                  reinterpret_cast<uint8_t*>(out + j * kTileLines + kSparseLine)[k] = sparse[token * 16 + (e >> 12)];
              }
            }
          } else {
            //: A LINE a (token, value line): 8 lanes a token's 128 bytes; its first lane the block's sparse line too
#pragma unroll
            for (int i = tid; i < kUnit * 128; i += THREADS) {
              const int entry = i >> 3, line = i & 7;
              if ((entry >> 4) < cur.steps) {
                const uint32_t token = list[entry] & kToken;
                out[entry * 9 + line] = val[token * 8 + line];
                if (line == 0) out[entry * 9 + 8] = reinterpret_cast<const uint4*>(sparse)[token];
              }
            }
          }
          //: THE WRITERS' EDGE: each thread's own lines, before the barrier the copy out follows
          lab::bulk_fence();
        }
      }
      if (tid == 0) lab::bulk_wait<1>();
      __syncthreads();
      opens = next.block != cur.block;
      heads = !cur.steps;
      prev = cur, prev_slot = list_n % 3u;
      cur = next;
      if (cur.steps) {
        ++list_n;
        sparse_n += opens;
        next = unit_at(first, cur.block, cur.step + cur.steps);
      }
    }
  }
  if (tid == 0) lab::bulk_wait<0>();
  __syncthreads();
  clock.stop(times);
}
}  // namespace values

// -------------------------------------------------------------------------------------------- the dense plane
namespace dense {
//: `D` buffers out share 576 steps of room: a unit 192 steps at 3, 96 at 6
template <int D>
constexpr int kUnit = 576 / D;
template <int D>
constexpr int kOutBytes = kUnit<D> * 256;
constexpr int kSrcAt = 0, kListAt = kSrcAt + 2 * kLevelBytes;
template <int D>
constexpr int kOutAt = kListAt + 2 * kUnit<D> * 32;
template <int D>
constexpr int kTotalAt = kOutAt<D> + D * kOutBytes<D>;
template <int D>
constexpr int kBarsAt = kTotalAt<D> + 16;
template <int D>
constexpr int kSmemBytes = kBarsAt<D> + 4 * 8;
enum Bar : int { kSource = 0, kList = 16 };

//: the unit at step `s` of dense block `b`: the window's steps left, `unit` at most, then the next block's first
__device__ __forceinline__ Unit unit_at(int total, int unit, int b, int s) {
  if (s >= total) ++b, s = 0;
  if (b >= kBlocks || total == 0) return {kBlocks, 0, 0};
  return {b, s, min(unit, total - s)};
}

template <int THREADS, int MOVE, int D>
__global__ void __launch_bounds__(THREADS, 1)
    plane(long long* times, const uint4* rows, const uint4* lists, const uint32_t* firsts, uint4* out_plane, int rounds,
          int stride, int max_steps, int out_stride) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x;
  const uint32_t base = lab::smem(smem), bars = base + kBarsAt<D>;
  uint32_t* const total_at = reinterpret_cast<uint32_t*>(smem + kTotalAt<D>);
  if (tid < 2) reinterpret_cast<uint4*>(smem + kSrcAt + tid * kLevelBytes)[kWindow] = make_uint4(0, 0, 0, 0);
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < 4; ++i) lab::mbar_init(bars + 8 * i, 1);
    lab::bulk_fence();
  }
  __syncthreads();
  uint32_t list_n = 0xffffffffu, source_n = 0xffffffffu;
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int round = 0; round < rounds; ++round) {
    const long long at = static_cast<long long>(blockIdx.x) + static_cast<long long>(round) * stride;
    const uint4* const wrows = rows + at * (kWindow * kDigits / 16);
    const uint4* const wlists = lists + at * max_steps * 2;
    uint4* const wplane = out_plane + (static_cast<long long>(blockIdx.x) + static_cast<long long>(round) * out_stride) * kBlocks * max_steps * 16;
    if (tid == 0) {
      *total_at = firsts[at * (kBlocks + 1) + kBlocks];
      lab::bulk_wait<D - 2>();
    }
    __syncthreads();
    const int total = static_cast<int>(*total_at);
    Unit prev{kBlocks, 0, 0}, cur{kBlocks, 0, 0}, next = unit_at(total, kUnit<D>, 0, 0);
    uint32_t prev_slot = 0;
    bool opens = false;
#pragma unroll 1
    for (;;) {
      //: THE ISSUE EDGE: the unit before's lines out, the unit after's list and its block's rows in
      if (tid == 0) {
        if (prev.steps) {
          lab::bulk_store(wplane + (static_cast<long long>(prev.block) * max_steps + prev.step) * 16,
                          base + kOutAt<D> + prev_slot * kOutBytes<D>, prev.steps * 256);
          lab::bulk_commit();
        }
        if (next.steps) {
          const uint32_t slot = (list_n + 1) & 1u;
          lab::mbar_expect(bars + kList + 8 * slot, next.steps * 32);
          lab::bulk_load(base + kListAt + slot * kUnit<D> * 32, wlists + next.step * 2, next.steps * 32, bars + kList + 8 * slot);
          if (next.block != cur.block) {
            const uint32_t level = (source_n + 1) & 1u;
            lab::mbar_expect(bars + kSource + 8 * level, kTiles * kBlockBytes);
#pragma unroll
            for (int t = 0; t < kTiles; ++t)
              lab::bulk_load(base + kSrcAt + level * kLevelBytes + t * kBlockBytes,
                             wrows + t * (kTileBytes / 16) + next.block * (kBlockBytes / 16), kBlockBytes,
                             bars + kSource + 8 * level);
          }
        }
      }
      if (!cur.steps && !next.steps) break;
      if (cur.steps) {
        lab::mbar_wait_suspended(bars + kList + 8 * (list_n & 1u), (list_n >> 1) & 1u);
        if (opens) lab::mbar_wait_suspended(bars + kSource + 8 * (source_n & 1u), (source_n >> 1) & 1u);
        if constexpr (MOVE == kMoved) {
          const uint16_t* const list = reinterpret_cast<const uint16_t*>(smem + kListAt + (list_n & 1u) * kUnit<D> * 32);
          const uint4* const src = reinterpret_cast<const uint4*>(smem + kSrcAt + (source_n & 1u) * kLevelBytes);
          uint4* const out = reinterpret_cast<uint4*>(smem + kOutAt<D> + (list_n % D) * kOutBytes<D>);
          //: A LINE an entry: its token's line of the block
#pragma unroll 1
          for (int i = tid; i < cur.steps * 16; i += THREADS) out[i] = src[list[i] & kToken];
          //: THE WRITERS' EDGE: each thread's own lines, before the barrier the copy out follows
          lab::bulk_fence();
        }
      }
      if (tid == 0) lab::bulk_wait<D - 2>();
      __syncthreads();
      opens = next.block != cur.block;
      prev = cur, prev_slot = list_n % D;
      cur = next;
      if (cur.steps) {
        ++list_n;
        source_n += opens;
        next = unit_at(total, kUnit<D>, cur.block, cur.step + cur.steps);
      }
    }
  }
  if (tid == 0) lab::bulk_wait<0>();
  __syncthreads();
  clock.stop(times);
}
}  // namespace dense

// ------------------------------------------------------------------------- the block stream, its lists the pass's
namespace blocks {
//: A UNIT is a SLOT: 8 digits' lists of a sparse block, its image the carry's -- the tables, then each list's 8 runs
//: of value lines, a brick a run, as long as the list. The lab's bounds: a list 32 entries, a slot 150
constexpr int kThreads = 512, kCap = 32, kUnits = 2 * kBlocks;
//: THE TABLES, in lines: a word an entry (its dense line and its second run's line, in its CTA's dense run), the
//: entries' sparse bytes, the second runs' pad lines, a step's four halves, the head (pads, entries, dense lines)
constexpr int kWordsAt = 0, kBytesAt = 40, kPadsAt = 50, kStepsAt = 66, kHeadAt = 74, kTables = 75;
constexpr int kRoomLines = 1280, kRoomBytes = kRoomLines * 16, kMaskBytes = 16 * 32 * 4;
constexpr int kValAt = 0, kSparseAt = kValAt + (kWindow + 1) * kValueBytes, kMaskAt = kSparseAt + 2 * kLevelBytes,
              kRecAt = kMaskAt + 2 * kMaskBytes, kCountAt = kRecAt + 16 * 64 * 2, kLayoutAt = kCountAt + 64,
              kTotalAt = kLayoutAt + 8 * 16, kSizeAt = kTotalAt + 16, kOutAt = kSizeAt + 16,
              kBarsAt = kOutAt + 3 * kRoomBytes, kSmemBytes = kBarsAt + 3 * 8;
enum Bar : int { kValues = 0, kLevel = 8 };
constexpr int spread(int n) { return (n + 15) & ~15; }

//: a warp's running sum of `v` over its lanes, each lane's own included: every lane takes every shuffle
__device__ __forceinline__ uint32_t scan(uint32_t v, int lane) {
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t before = __shfl_up_sync(0xffffffffu, v, o);
    v += lane >= o ? before : 0u;
  }
  return v;
}

//: `SCALED`: an entry's value lines multiplied by its byte on its sparse digit as they are written -- measured, and
//: not the design's form
template <int THREADS, int SCALED>
__global__ void __launch_bounds__(THREADS, 1)
    plane(long long* times, const uint4* vals, const uint4* rows, const uint32_t* masks, uint4* out_plane, uint32_t* plans,
          int rounds, int stride, int room, int ring) {
  static_assert(THREADS == kThreads);
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t base = lab::smem(smem), bars = base + kBarsAt;
  const uint4* const val = reinterpret_cast<const uint4*>(smem + kValAt);
  uint32_t* const count = reinterpret_cast<uint32_t*>(smem + kCountAt);
  uint4* const layout = reinterpret_cast<uint4*>(smem + kLayoutAt);
  uint32_t* const size = reinterpret_cast<uint32_t*>(smem + kSizeAt);
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < 3; ++i) lab::mbar_init(bars + 8 * i, 1);
    lab::bulk_fence();
  }
  __syncthreads();
  //: the units so far: a unit's image is buffer `units % 3`
  uint32_t units = 0;
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int round = 0; round < rounds; ++round) {
    const long long at = static_cast<long long>(blockIdx.x) + static_cast<long long>(round) * stride;
    const uint4* const wvals = vals + at * (kWindow * kValueBytes / 16);
    const uint4* const wrows = rows + at * (kWindow * kDigits / 16);
    const uint32_t* const wmasks = masks + at * (kDigits * 32);
    uint4* const wplane = out_plane + static_cast<long long>(blockIdx.x) * room;
    uint32_t* const wplan = plans + static_cast<long long>(blockIdx.x) * (2 * kUnits);
    //: a sparse block's mask and rows, to level `g % 2`, on its barrier
    const auto load_block = [&](int g) {
      const uint32_t level = static_cast<uint32_t>(g) & 1u;
      lab::mbar_expect(bars + kLevel + 8 * level, kMaskBytes + kTiles * kBlockBytes);
      lab::bulk_load(base + kMaskAt + level * kMaskBytes, wmasks + g * (16 * 32), kMaskBytes, bars + kLevel + 8 * level);
#pragma unroll
      for (int t = 0; t < kTiles; ++t)
        lab::bulk_load(base + kSparseAt + level * kLevelBytes + t * kBlockBytes,
                       wrows + t * (kTileBytes / 16) + g * (kBlockBytes / 16), kBlockBytes, bars + kLevel + 8 * level);
    };
    if (tid == 0) {
      lab::mbar_expect(bars + kValues, kWindow * kValueBytes);
#pragma unroll
      for (int t = 0; t < kTiles; ++t)
        lab::bulk_load(base + kValAt + t * (kTile * kValueBytes), wvals + t * (kTile * kValueBytes / 16),
                       kTile * kValueBytes, bars + kValues);
      load_block(0);
    }
#pragma unroll 1
    for (int u = 0; u <= kUnits; ++u) {
      //: THE ISSUE EDGE: the unit before's image out and its plan; the next block's mask and rows in, a unit ahead
      if (tid == 0) {
        if (u > 0) {
          const uint32_t slot = (units - 1) % 3u;
          //: a unit's room in the CTA's region, or in a ring of `ring` lines it rewrites as it goes
          const int at = (u - 1) * kRoomLines;
          lab::bulk_store(wplane + (ring ? at % ring : at), base + kOutAt + slot * kRoomBytes, size[slot]);
          lab::bulk_commit();
          wplan[2 * (u - 1)] = static_cast<uint32_t>((u - 1) * kRoomLines), wplan[2 * (u - 1) + 1] = size[slot];
        }
        if ((u & 1) && u + 1 < kUnits) load_block((u + 1) >> 1);
      }
      if (u == kUnits) break;
      const int g = u >> 1, h = u & 1, level = g & 1;
      uint16_t* const recs = reinterpret_cast<uint16_t*>(smem + kRecAt);
      if (h == 0) {
        //: THE LANDED EDGE, a block's: its mask and rows, and its window's values where it opens them
        lab::mbar_wait_suspended(bars + kLevel + 8 * level, (g >> 1) & 1u);
        if (u == 0) lab::mbar_wait_suspended(bars + kValues, round & 1u);
        //: THE LISTS, a block's 16: a warp a digit, a lane 32 tokens' bits of it. A lane's live tokens go to their
        //: ranks in the digit's list -- the lanes' counts before it and the bits under each -- and a token that is
        //: not live, or past the lab's bound, to the lane's own spare place
        const uint32_t bits = reinterpret_cast<const uint32_t*>(smem + kMaskAt + level * kMaskBytes)[tid];
        const uint32_t live = static_cast<uint32_t>(__popc(bits)), through = scan(live, lane), before = through - live;
        if (lane == 31) count[warp] = through;
#pragma unroll 1
        for (int i = 0; i < 32; ++i) {
          const uint32_t rank = before + static_cast<uint32_t>(__popc(bits & ((1u << i) - 1u)));
          recs[warp * 64 + ((bits >> i & 1u) && rank < kCap ? rank : 32u + lane)] = static_cast<uint16_t>(lane * 32 + i);
        }
        __syncthreads();
      }
      //: THE SLOT'S LAYOUT, once a CTA: its 8 lists' entries, runs' lines and pads before each -- nine threads, a
      //: list each and the totals, each summing the counts before it; published behind the unit's barrier
      if (tid <= 8) {
        uint32_t entries = 0, dense = 0, pads = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const uint32_t n = i < tid ? count[8 * h + i] : 0u, lines = static_cast<uint32_t>(spread(static_cast<int>(n)));
          entries += n, dense += lines, pads += lines - n;
        }
        if (tid < 8) layout[tid] = make_uint4(entries, kTables + 8 * entries, dense, pads);
        else *reinterpret_cast<uint4*>(smem + kTotalAt) = make_uint4(entries, dense, pads, 0);
      }
      __syncthreads();
      {
        //: THE IMAGE: a thread a (list, place) and four bricks -- an entry's value lines to its runs; and, the first
        //: half of the threads, its word, its sparse byte, or its pad's line; 16 threads a step's halves, one the head
        uint4* const out = reinterpret_cast<uint4*>(smem + kOutAt + (units % 3u) * kRoomBytes);
        const uint4 total = *reinterpret_cast<const uint4*>(smem + kTotalAt);
        const int cell = tid & 255, half = tid >> 8, jj = cell >> 5, k = cell & 31, j = 8 * h + jj;
        const int n = static_cast<int>(count[j]);
        const uint4 at = layout[jj];
        if (k < n) {
          const uint32_t token = recs[j * 64 + k];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            if constexpr (SCALED != 0) {
              const float weight = static_cast<float>(smem[kSparseAt + level * kLevelBytes + token * 16 + j]);
              const uint4 line = val[token * 8 + 4 * half + q];
              const uint32_t word[4] = {line.x, line.y, line.z, line.w};
              uint32_t scaled[4];
#pragma unroll
              for (int c = 0; c < 4; ++c)
                scaled[c] = lab::bf16x2_round(weight * __uint_as_float(word[c] << 16), weight * __uint_as_float(word[c] & 0xffff0000u));
              out[at.y + (4 * half + q) * n + k] = make_uint4(scaled[0], scaled[1], scaled[2], scaled[3]);
            } else {
              out[at.y + (4 * half + q) * n + k] = val[token * 8 + 4 * half + q];
            }
          }
          if (half == 0) {
            reinterpret_cast<uint32_t*>(out + kWordsAt)[at.x + k] = (at.z + k) | (total.y + at.z + k) << 16;
            reinterpret_cast<uint8_t*>(out + kBytesAt)[at.x + k] = smem[kSparseAt + level * kLevelBytes + token * 16 + j];
          }
        } else if (half == 0 && k < spread(n)) {
          reinterpret_cast<uint16_t*>(out + kPadsAt)[at.w + k - n] = static_cast<uint16_t>(total.y + at.z + k);
        }
        if (tid < 16) {
          const int r = tid >> 3, list = tid & 7, m = static_cast<int>(count[8 * h + list]);
          const uint4 of = layout[list];
          const bool has = 16 * r < m;
          uint16_t* const step = reinterpret_cast<uint16_t*>(out + kStepsAt) + 4 * tid;
          step[0] = static_cast<uint16_t>(of.y + (has ? 16 * r : 0)), step[1] = static_cast<uint16_t>(m);
          step[2] = has ? static_cast<uint16_t>(of.z + 16 * r) : static_cast<uint16_t>(0xffff), step[3] = static_cast<uint16_t>(total.y);
        }
        if (tid == 16) {
          out[kHeadAt] = make_uint4(total.z, total.x, total.y, 0);
          size[units % 3u] = (kTables + 8 * total.x) * 16;
        }
        //: THE WRITERS' EDGE: each thread's own lines, before the barrier the copy out follows
        lab::bulk_fence();
      }
      if (tid == 0) lab::bulk_wait<1>();
      __syncthreads();
      ++units;
    }
  }
  if (tid == 0) lab::bulk_wait<0>();
  __syncthreads();
  clock.stop(times);
}
}  // namespace blocks

// --------------------------------------------- the block stream's coefficients, made in the carry step's format
namespace coefs {
//: A CTA writes a window's COEFFICIENT RUNS: first THE WINDOW'S LISTS, once -- every sparse block's 16, an entry its
//: token and its byte on its sparse digit, and every slot's layout -- then a dense block at a time, a unit a slot's 8
//: lists for that dense block: each entry's 16 dense digits by its byte, bf16, a line a brick, the first bricks'
//: runs and then the second's, each list's padded with zero lines to its steps. The lab's bound: a list 32 entries
constexpr int kThreads = 512, kCap = 32, kUnits = 2 * kBlocks, kRoomLines = 512, kRoomBytes = kRoomLines * 16;
constexpr int kMaskBytes = 16 * 32 * 4;
constexpr int kDenseAt = 0, kSparseAt = kDenseAt + 2 * kLevelBytes, kMaskAt = kSparseAt + 2 * kLevelBytes,
              kRecAt = kMaskAt + 2 * kMaskBytes, kSpareAt = kRecAt + kDigits * kCap * 4, kCountAt = kSpareAt + kThreads * 4,
              kLayoutAt = kCountAt + kDigits * 4, kSizeAt = kLayoutAt + kUnits * 9 * 4, kOutAt = kSizeAt + 64;
//: `U` slots a trip: a buffer out is their rooms side by side
template <int U>
constexpr int kBarsAt = kOutAt + 3 * U * kRoomBytes;
template <int U>
constexpr int kSmemBytes = kBarsAt<U> + 4 * 8;
enum Bar : int { kDense = 0, kLevel = 16 };

//: `WIDEN`: a coefficient line is the entry's dense digits widened to bf16 alone, the entry's scalar on its values --
//: measured, and not the design's form, which multiplies them by its sparse byte. `U`: the slots a trip
//: `PART`: the whole kernel (0); its runs made and no copy out (1); its copies out and no run made (2) -- two
//: controls that part a trip's lines from its copy
template <int THREADS, int WIDEN, int U, int PART = 0>
__global__ void __launch_bounds__(THREADS, 1)
    plane(long long* times, const uint4* dense_rows, const uint4* rows, const uint32_t* masks, uint4* out_plane, int rounds,
          int stride, int room, int ring) {
  static_assert(THREADS == kThreads);
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t base = lab::smem(smem), bars = base + kBarsAt<U>;
  constexpr int kTrips = kUnits / U;
  uint32_t* const recs = reinterpret_cast<uint32_t*>(smem + kRecAt);
  uint32_t* const count = reinterpret_cast<uint32_t*>(smem + kCountAt);
  uint32_t* const layout = reinterpret_cast<uint32_t*>(smem + kLayoutAt);
  uint32_t* const size = reinterpret_cast<uint32_t*>(smem + kSizeAt);
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < 4; ++i) lab::mbar_init(bars + 8 * i, 1);
    lab::bulk_fence();
  }
  __syncthreads();
  //: the units so far, a buffer out each in turn
  uint32_t units = 0;
  lab::WarpSpan clock;
  clock.start();
#pragma unroll 1
  for (int round = 0; round < rounds; ++round) {
    const long long at = static_cast<long long>(blockIdx.x) + static_cast<long long>(round) * stride;
    const uint4* const wdense = dense_rows + at * (kWindow * kDigits / 16);
    const uint4* const wrows = rows + at * (kWindow * kDigits / 16);
    const uint32_t* const wmasks = masks + at * (kDigits * 32);
    uint4* const wplane = out_plane + static_cast<long long>(blockIdx.x) * room;
    //: a sparse block's mask and rows, and a dense block's rows, each to its buffer in turn on that buffer's barrier
    const auto load_block = [&](int g) {
      const uint32_t level = static_cast<uint32_t>(g) & 1u;
      lab::mbar_expect(bars + kLevel + 8 * level, kMaskBytes + kTiles * kBlockBytes);
      lab::bulk_load(base + kMaskAt + level * kMaskBytes, wmasks + g * (16 * 32), kMaskBytes, bars + kLevel + 8 * level);
#pragma unroll
      for (int t = 0; t < kTiles; ++t)
        lab::bulk_load(base + kSparseAt + level * kLevelBytes + t * kBlockBytes,
                       wrows + t * (kTileBytes / 16) + g * (kBlockBytes / 16), kBlockBytes, bars + kLevel + 8 * level);
    };
    const auto load_dense = [&](int db) {
      const uint32_t side = static_cast<uint32_t>(db) & 1u;
      lab::mbar_expect(bars + kDense + 8 * side, kTiles * kBlockBytes);
#pragma unroll
      for (int t = 0; t < kTiles; ++t)
        lab::bulk_load(base + kDenseAt + side * kLevelBytes + t * kBlockBytes,
                       wdense + t * (kTileBytes / 16) + db * (kBlockBytes / 16), kBlockBytes, bars + kDense + 8 * side);
    };
    if (tid == 0) {
      load_block(0);
      load_dense(0);
    }
    //: THE WINDOW'S LISTS, a sparse block a trip: a warp a digit, a lane 32 tokens' bits; a live token's entry -- the
    //: token and its byte on the digit -- goes to its rank in the digit's list, any other to the thread's spare
#pragma unroll 1
    for (int g = 0; g < kBlocks; ++g) {
      if (tid == 0 && g + 1 < kBlocks) load_block(g + 1);
      const uint32_t level = static_cast<uint32_t>(g) & 1u;
      lab::mbar_wait_suspended(bars + kLevel + 8 * level, (g >> 1) & 1u);
      const uint32_t bits = reinterpret_cast<const uint32_t*>(smem + kMaskAt + level * kMaskBytes)[tid];
      const uint32_t live = static_cast<uint32_t>(__popc(bits)), through = blocks::scan(live, lane), before = through - live;
      if (lane == 31) count[16 * g + warp] = through;
      const uint8_t* const bytes = smem + kSparseAt + level * kLevelBytes + warp;
#pragma unroll 1
      for (int i = 0; i < 32; ++i) {
        const uint32_t rank = before + static_cast<uint32_t>(__popc(bits & ((1u << i) - 1u))), token = static_cast<uint32_t>(lane * 32 + i);
        const bool kept = (bits >> i & 1u) && rank < kCap;
        (kept ? recs + (16 * g + warp) * kCap + rank : reinterpret_cast<uint32_t*>(smem + kSpareAt) + tid)[0] =
            token | static_cast<uint32_t>(bytes[token * 16]) << 16;
      }
      __syncthreads();
    }
    //: EVERY SLOT'S LAYOUT: its 8 lists' first runs' lines before each, and their sum -- a thread a (slot, list)
    if (tid < kUnits * 9) {
      const int unit = tid / 9, m = tid % 9;
      uint32_t lines = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) lines += i < m ? static_cast<uint32_t>(blocks::spread(static_cast<int>(count[8 * unit + i]))) : 0u;
      layout[tid] = lines;
    }
    __syncthreads();
#pragma unroll 1
    for (int trip = 0; trip <= kBlocks * kTrips; ++trip) {
      const int db = trip / kTrips, u0 = (trip % kTrips) * U;
      //: THE ISSUE EDGE: the unit before's runs out; the next dense block's rows, a dense block ahead
      if (tid == 0) {
        if (trip > 0 && PART != 1) {
          const uint32_t slot = (units - 1) % 3u;
#pragma unroll
          for (int q = 0; q < U; ++q)
            lab::bulk_store(wplane + (ring ? ((trip - 1) * U + q) * kRoomLines % ring : ((trip - 1) * U + q) * kRoomLines),
                            base + kOutAt + (slot * U + q) * kRoomBytes, size[slot * U + q]);
          lab::bulk_commit();
        }
        if (u0 == 0 && db + 1 < kBlocks) load_dense(db + 1);
      }
      if (trip == kBlocks * kTrips) break;
      if (u0 == 0) lab::mbar_wait_suspended(bars + kDense + 8 * (db & 1), (db >> 1) & 1u);
#pragma unroll
      for (int q = 0; q < U; ++q) {
        //: THE RUNS, a slot of the trip's: a thread a (list, place) and a brick -- an entry's 8 coefficients of the
        //: brick, rounded to bf16; a pad's line zero
        const int u = u0 + q;
        uint4* const out = reinterpret_cast<uint4*>(smem + kOutAt + ((units % 3u) * U + q) * kRoomBytes);
        const int cell = tid & 255, brick = tid >> 8, jj = cell >> 5, k = cell & 31, j = 8 * u + jj;
        const int n = static_cast<int>(count[j]);
        const uint32_t to = layout[9 * u + jj] + static_cast<uint32_t>(brick) * layout[9 * u + 8] + static_cast<uint32_t>(k);
        if (PART == 2) {
        } else if (k < n) {
          const uint32_t entry = recs[j * kCap + k];
          const float weight = WIDEN ? 1.0f : static_cast<float>(entry >> 16);
          const uint2 two = reinterpret_cast<const uint2*>(smem + kDenseAt + (db & 1) * kLevelBytes + (entry & 0xffffu) * 16)[brick];
          const uint32_t word[2] = {two.x, two.y};
          uint32_t pair[4];
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const uint32_t pairs = word[c >> 1] >> (16 * (c & 1));
            if constexpr (WIDEN != 0) pair[c] = lab::bf16x2_round(static_cast<float>(pairs & 255u), static_cast<float>(pairs >> 8 & 255u));
            else pair[c] = lab::bf16x2_round(weight * static_cast<float>(pairs & 255u), weight * static_cast<float>(pairs >> 8 & 255u));
          }
          out[to] = make_uint4(pair[0], pair[1], pair[2], pair[3]);
        } else if (k < blocks::spread(n)) {
          out[to] = make_uint4(0, 0, 0, 0);
        }
        if (tid == 0) size[(units % 3u) * U + q] = 2 * layout[9 * u + 8] * 16;
      }
      lab::bulk_fence();
      if (tid == 0 && PART != 1) lab::bulk_wait<1>();
      __syncthreads();
      ++units;
    }
  }
  if (tid == 0) lab::bulk_wait<0>();
  __syncthreads();
  clock.stop(times);
}
}  // namespace coefs

//: A CELL: token `x` has `live_of` live digits, a stride apart from a start its own (`facts_lists`'s cells)
constexpr int live_of(int cell, uint32_t x) { return cell == 0 ? 4 + (x % 3 == 0) : 32; }
constexpr uint32_t digit_of(int cell, uint32_t x, int i) {
  return (x * 5u + (x >> 3) + static_cast<uint32_t>(i) * (cell == 0 ? 37u : 7u)) & 255u;
}

}  // namespace gather_pass

int main(int argc, char** argv) {
  using namespace gather_pass;
  lab::Bench b(argc, argv);
  for (int threads : {256, 512, 1024})
    for (int move : {static_cast<int>(kFlat), static_cast<int>(kMoved)}) {
      //: a unit is a trip: one thread's issue, every thread's waits, and a window's units a count of its data's
      b.expect(lab::kernel("gather_pass::values::plane", {threads, move, kTiled}), "LOOP_CONVERGENCE", "<=", 64);
      if (threads == 512) b.expect(lab::kernel("gather_pass::values::plane", {threads, move, kRecords}), "LOOP_CONVERGENCE", "<=", 64);
      b.expect(lab::kernel("gather_pass::dense::plane", {threads, move, 3}), "LOOP_CONVERGENCE", "<=", 64);
      if (threads == 512) b.expect(lab::kernel("gather_pass::dense::plane", {threads, move, 6}), "LOOP_CONVERGENCE", "<=", 64);
    }
  //: the block stream's unit is a trip, and a block's lists' scan is every warp's, a digit each, in its trip
  for (int scaled : {0, 1}) b.expect(lab::kernel("gather_pass::blocks::plane", {blocks::kThreads, scaled}), "LOOP_CONVERGENCE", "<=", 64);
  for (auto form : {std::pair<int, int>{0, 1}, {0, 4}, {1, 4}})
    b.expect(lab::kernel("gather_pass::coefs::plane", {coefs::kThreads, form.first, form.second, 0}), "LOOP_CONVERGENCE", "<=", 64);
  for (int part : {1, 2}) b.expect(lab::kernel("gather_pass::coefs::plane", {coefs::kThreads, 0, 4, part}), "LOOP_CONVERGENCE", "<=", 64);
  if (b.expect_only) return 0;
  constexpr int kRounds = 8, kMax = kMaxSteps<32>;
  const int windows = b.sms;
  const size_t all_windows = static_cast<size_t>(windows) * kRounds, tokens = all_windows * kWindow;
  //: the values token-major and the dense level's rows tile-major, every byte its own position's
  constexpr uint32_t kValuesSalt = 0x9e3779b9u, kDenseSalt = 0x85ebca6bu;
  const auto byte_at = [](uint32_t salt, size_t i) { return static_cast<uint8_t>((static_cast<uint32_t>(i) ^ salt) * 2654435761u >> 24); };
  std::vector<uint8_t> hvalues(tokens * kValueBytes), hdense(tokens * kDigits);
  for (size_t i = 0; i < hvalues.size(); ++i) hvalues[i] = byte_at(kValuesSalt, i);
  for (size_t i = 0; i < hdense.size(); ++i) hdense[i] = byte_at(kDenseSalt, i);
  uint4 *dvalues = nullptr, *ddense = nullptr, *drows = nullptr, *dlists = nullptr, *dvplane = nullptr, *ddplane = nullptr;
  uint32_t *dfirsts = nullptr, *dmasks = nullptr, *dplans = nullptr;
  const size_t vplane_bytes = static_cast<size_t>(windows) * kMax * kLines<kRecords> * 16,
               dplane_bytes = static_cast<size_t>(windows) * kBlocks * kMax * 256;
  LAB_CHECK(cudaMalloc(&dvalues, hvalues.size()));
  LAB_CHECK(cudaMemcpy(dvalues, hvalues.data(), hvalues.size(), cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMalloc(&ddense, hdense.size()));
  LAB_CHECK(cudaMemcpy(ddense, hdense.data(), hdense.size(), cudaMemcpyHostToDevice));
  LAB_CHECK(cudaMalloc(&drows, tokens * kDigits));
  LAB_CHECK(cudaMalloc(&dlists, all_windows * kMax * 32));
  LAB_CHECK(cudaMalloc(&dfirsts, all_windows * (kBlocks + 1) * 4));
  LAB_CHECK(cudaMalloc(&dvplane, vplane_bytes));
  LAB_CHECK(cudaMalloc(&ddplane, dplane_bytes));
  LAB_CHECK(cudaMalloc(&dmasks, all_windows * kDigits * 32 * 4));
  LAB_CHECK(cudaMalloc(&dplans, static_cast<size_t>(windows) * blocks::kUnits * 2 * 4));
  for (int cell = 0; cell < 2; ++cell) {
    //: the sparse level's rows tile-major, and each window's step lists: a digit's tokens in order, 16 a step
    std::vector<uint8_t> tiled(tokens * kDigits, 0);
    //: [0] by digit (steps), [1] THE STREAM'S, by sparse block: a block's live tokens once each
    std::vector<uint16_t> all_lists[2];
    std::vector<uint32_t> all_firsts[2];
    for (int m = 0; m < 2; ++m) {
      all_lists[m].assign(all_windows * kMax * 16, static_cast<uint16_t>(kPad));
      all_firsts[m].assign(all_windows * (kBlocks + 1), 0);
    }
    std::vector<uint16_t>& lists = all_lists[0];
    std::vector<uint32_t>& firsts = all_firsts[0];
    std::vector<std::vector<uint16_t>> by_digit(kDigits), by_block(kBlocks);
    for (size_t w = 0; w < all_windows; ++w) {
      for (auto& l : by_digit) l.clear();
      for (auto& l : by_block) l.clear();
      for (size_t x = w * kWindow; x < (w + 1) * kWindow; ++x) {
        uint32_t blocks_live = 0;
        for (int i = 0, n = live_of(cell, static_cast<uint32_t>(x)); i < n; ++i) {
          const uint32_t d = digit_of(cell, static_cast<uint32_t>(x), i);
          uint8_t& byte = tiled[(x / kTile) * kTileBytes + (d / 16) * kBlockBytes + (x % kTile) * 16 + d % 16];
          if (!byte) by_digit[d].push_back(static_cast<uint16_t>(x % kWindow));
          byte = static_cast<uint8_t>(1u + (x * 7u + static_cast<uint32_t>(i) * 13u) % 255u);
          blocks_live |= 1u << (d / 16);
        }
        for (int g = 0; g < kBlocks; ++g)
          if (blocks_live >> g & 1u) by_block[g].push_back(static_cast<uint16_t>(x % kWindow));
      }
      size_t stream_step = 0;
      for (int g = 0; g < kBlocks; ++g) {
        all_firsts[1][w * (kBlocks + 1) + g] = static_cast<uint32_t>(stream_step);
        for (size_t e = 0; e < by_block[g].size(); ++e) all_lists[1][(w * kMax + stream_step + e / 16) * 16 + e % 16] = by_block[g][e];
        stream_step += (by_block[g].size() + 15) / 16;
      }
      all_firsts[1][w * (kBlocks + 1) + kBlocks] = static_cast<uint32_t>(stream_step);
      size_t step = 0;
      for (int d = 0; d < kDigits; ++d) {
        if (d % 16 == 0) firsts[w * (kBlocks + 1) + d / 16] = static_cast<uint32_t>(step);
        for (size_t e = 0; e < by_digit[d].size(); ++e)
          lists[(w * kMax + step + e / 16) * 16 + e % 16] = static_cast<uint16_t>(by_digit[d][e] | (d % 16) << 12);
        step += (by_digit[d].size() + 15) / 16;
      }
      firsts[w * (kBlocks + 1) + kBlocks] = static_cast<uint32_t>(step);
    }
    LAB_CHECK(cudaMemcpy(drows, tiled.data(), tiled.size(), cudaMemcpyHostToDevice));
    if (cell == 0) {
      //: THE BLOCK STREAM, its lists the pass's: the sparse level's MASK digit-major -- a digit's 1,024 bits of a
      //: window, 32 words -- is all the pass is handed beside the tables
      std::vector<uint32_t> mask(all_windows * kDigits * 32, 0);
      for (size_t x = 0; x < tokens; ++x)
        for (int d = 0; d < kDigits; ++d)
          if (tiled[(x / kTile) * kTileBytes + (d / 16) * kBlockBytes + (x % kTile) * 16 + d % 16])
            mask[((x / kWindow) * kDigits + d) * 32 + (x % kWindow) / 32] |= 1u << (x % 32);
      LAB_CHECK(cudaMemcpy(dmasks, mask.data(), mask.size() * 4, cudaMemcpyHostToDevice));
      using namespace blocks;
      const int room = kMax * kLines<kRecords>;
      if (kUnits * kRoomLines > room) std::exit(2);
      const auto kernel = blocks::plane<blocks::kThreads, 0>;
      LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, blocks::kSmemBytes));
      const auto launch = [&](int cta, int rounds, int stride, int ring = 0) {
        kernel<<<cta, blocks::kThreads, blocks::kSmemBytes>>>(b.times, dvalues, drows, dmasks, dvplane, dplans, rounds, stride, room, ring);
      };
      const std::string form = "4 to 5 live digits a token, the block stream's tables, values and sparse bytes, its lists made from the mask, 512 threads";
      //: the first round's images against the host's own lists, tables and runs, on every multiprocessor
      LAB_CHECK(cudaMemset(dvplane, 0xa5, vplane_bytes));
      launch(windows, 1, 0);
      LAB_CHECK(cudaDeviceSynchronize());
      std::vector<uint8_t> got(static_cast<size_t>(windows) * room * 16);
      std::vector<uint32_t> plan(static_cast<size_t>(windows) * kUnits * 2);
      LAB_CHECK(cudaMemcpy(got.data(), dvplane, got.size(), cudaMemcpyDeviceToHost));
      LAB_CHECK(cudaMemcpy(plan.data(), dplans, plan.size() * 4, cudaMemcpyDeviceToHost));
      long long wrong = 0, cells = 0;
      for (int w = 0; w < windows; ++w) {
        std::vector<std::vector<uint16_t>> by(kDigits);
        for (int d = 0; d < kDigits; ++d)
          for (int t = 0; t < kWindow; ++t)
            if (mask[(static_cast<size_t>(w) * kDigits + d) * 32 + t / 32] >> (t % 32) & 1u) by[d].push_back(static_cast<uint16_t>(t));
        for (int u = 0; u < kUnits; ++u) {
          const uint8_t* const image = got.data() + (static_cast<size_t>(w) * room + static_cast<size_t>(u) * kRoomLines) * 16;
          int n[8], entries = 0, lines = 0, pads = 0;
          for (int j = 0; j < 8; ++j) n[j] = static_cast<int>(by[8 * u + j].size()), entries += n[j], lines += blocks::spread(n[j]);
          for (int j = 0; j < 8; ++j)
            if (n[j] > kCap || entries > 150) {
              std::fprintf(stderr, "gather_pass: a list of %d entries in a slot of %d is not this lab's\n", n[j], entries);
              std::exit(2);
            }
          cells += 2, wrong += plan[(static_cast<size_t>(w) * kUnits + u) * 2] != static_cast<uint32_t>(u * kRoomLines);
          wrong += plan[(static_cast<size_t>(w) * kUnits + u) * 2 + 1] != static_cast<uint32_t>((kTables + 8 * entries) * 16);
          int e = 0, dense = 0;
          for (int j = 0; j < 8; ++j) {
            const int values_at = kTables + 8 * e;
            for (int k = 0; k < n[j]; ++k) {
              const size_t x = static_cast<size_t>(w) * kWindow + by[8 * u + j][k];
              uint32_t word;
              std::memcpy(&word, image + kWordsAt * 16 + (e + k) * 4, 4);
              cells += 2, wrong += word != (static_cast<uint32_t>(dense + k) | static_cast<uint32_t>(lines + dense + k) << 16);
              wrong += image[kBytesAt * 16 + e + k] != tiled[(x / kTile) * kTileBytes + (u / 2) * kBlockBytes + (x % kTile) * 16 + (8 * u + j) % 16];
              for (int br = 0; br < 8; ++br, ++cells)
                wrong += std::memcmp(image + (values_at + br * n[j] + k) * 16, hvalues.data() + x * kValueBytes + br * 16, 16) != 0;
            }
            for (int k = n[j]; k < blocks::spread(n[j]); ++k, ++pads, ++cells) {
              uint16_t pad;
              std::memcpy(&pad, image + kPadsAt * 16 + pads * 2, 2);
              wrong += pad != static_cast<uint16_t>(lines + dense + k);
            }
            for (int r = 0; r < 2; ++r, ++cells) {
              uint16_t step[4];
              std::memcpy(step, image + kStepsAt * 16 + (8 * r + j) * 8, 8);
              const bool has = 16 * r < n[j];
              wrong += step[0] != values_at + (has ? 16 * r : 0) || step[1] != n[j] || step[2] != (has ? dense + 16 * r : 0xffff) || step[3] != lines;
            }
            e += n[j], dense += blocks::spread(n[j]);
          }
          uint32_t head[3];
          std::memcpy(head, image + kHeadAt * 16, 12);
          ++cells, wrong += head[0] != static_cast<uint32_t>(pads) || head[1] != static_cast<uint32_t>(entries) || head[2] != static_cast<uint32_t>(lines);
        }
      }
      b.check(form, wrong == 0, std::to_string(wrong) + " of " + std::to_string(cells) + " words, bytes, lines, pads, steps and plans differ");
      {
        //: THE SAME KERNEL INTO A RING the cache holds: a CTA rewrites 8 slots' rooms, 164 KB, as it goes
        constexpr int kRing = 8 * kRoomLines;
        const lab::Samples cycles = b.variant([&] { launch(windows, kRounds, windows, kRing); },
                                              [&] { return b.span(blocks::kThreads >> 5, windows) / kRounds / kWindow; });
        const std::string timed = form + ", its plane a ring of 164 KB a CTA, a fresh window a round";
        b.unchecked(timed, "its kernel's first round is checked on every multiprocessor; a ring keeps no round to check");
        b.record(timed, "SM-cycles a token", cycles);
      }
      for (int quarter = 0; quarter < 2; ++quarter) {
        const int cta = quarter ? windows / 4 : windows;
        const lab::Samples cycles = b.variant([&] { launch(cta, kRounds, windows); },
                                              [&] { return b.span(blocks::kThreads >> 5, cta) / kRounds / kWindow; });
        const std::string timed = form + (quarter ? ", a quarter of the multiprocessors" : "") + ", a fresh window a round";
        b.unchecked(timed, "its kernel's first round is checked on every multiprocessor; this one is its time");
        b.record(timed, "SM-cycles a token", cycles);
      }
      {
        //: THE SCALED VALUES: the same kernel with an entry's value lines multiplied by its sparse byte; a value
        //: whose bits are no finite normal number is not compared
        const auto sk = blocks::plane<blocks::kThreads, 1>;
        LAB_CHECK(cudaFuncSetAttribute(sk, cudaFuncAttributeMaxDynamicSharedMemorySize, blocks::kSmemBytes));
        const auto slaunch = [&](int cta, int rounds, int stride, int ring = 0) {
          sk<<<cta, blocks::kThreads, blocks::kSmemBytes>>>(b.times, dvalues, drows, dmasks, dvplane, dplans, rounds, stride, room, ring);
        };
        const std::string sform = form + ", an entry's values multiplied by its sparse byte";
        LAB_CHECK(cudaMemset(dvplane, 0xa5, vplane_bytes));
        slaunch(windows, 1, 0);
        LAB_CHECK(cudaDeviceSynchronize());
        LAB_CHECK(cudaMemcpy(got.data(), dvplane, got.size(), cudaMemcpyDeviceToHost));
        const auto bf16 = [](float f) { uint32_t v; std::memcpy(&v, &f, 4); return static_cast<uint16_t>((v + 0x7fffu + (v >> 16 & 1u)) >> 16); };
        long long swrong = 0, svalues = 0;
        for (int w = 0; w < windows; ++w) {
          std::vector<std::vector<uint16_t>> by(kDigits);
          for (int d = 0; d < kDigits; ++d)
            for (int t = 0; t < kWindow; ++t)
              if (mask[(static_cast<size_t>(w) * kDigits + d) * 32 + t / 32] >> (t % 32) & 1u) by[d].push_back(static_cast<uint16_t>(t));
          for (int u = 0; u < kUnits; ++u) {
            const uint8_t* const image = got.data() + (static_cast<size_t>(w) * room + static_cast<size_t>(u) * kRoomLines) * 16;
            int e = 0;
            for (int j = 0; j < 8; ++j) {
              const int n = static_cast<int>(by[8 * u + j].size()), values_at = kTables + 8 * e;
              for (int k = 0; k < n; ++k) {
                const size_t x = static_cast<size_t>(w) * kWindow + by[8 * u + j][k];
                const float weight = static_cast<float>(tiled[(x / kTile) * kTileBytes + (u / 2) * kBlockBytes + (x % kTile) * 16 + (8 * u + j) % 16]);
                for (int v = 0; v < 64; ++v) {
                  uint16_t in, out;
                  std::memcpy(&in, hvalues.data() + x * kValueBytes + 2 * v, 2);
                  std::memcpy(&out, image + (values_at + (v / 8) * n + k) * 16 + 2 * (v % 8), 2);
                  const int exponent = in >> 7 & 0xff;
                  if (exponent == 0 || exponent == 0xff) continue;
                  uint32_t bits = static_cast<uint32_t>(in) << 16;
                  float value;
                  std::memcpy(&value, &bits, 4);
                  const float product = weight * value;
                  uint32_t pbits;
                  std::memcpy(&pbits, &product, 4);
                  if ((pbits >> 23 & 0xff) == 0xff) continue;
                  ++svalues, swrong += out != bf16(product);
                }
              }
              e += n;
            }
          }
        }
        b.check(sform, swrong == 0, std::to_string(swrong) + " of " + std::to_string(svalues) + " scaled values differ");
        const lab::Samples cycles = b.variant([&] { slaunch(windows, kRounds, windows); },
                                              [&] { return b.span(blocks::kThreads >> 5, windows) / kRounds / kWindow; });
        b.unchecked(sform + ", a fresh window a round", "its kernel's first round is checked on every multiprocessor; this one is its time");
        b.record(sform + ", a fresh window a round", "SM-cycles a token", cycles);
      }
      {
        //: THE COEFFICIENT RUNS: a dense block at a time, a slot's lists' entries' coefficients made in the step's
        //: format, checked line by line against the host's own product and rounding
        const int croom = kBlocks * coefs::kUnits * coefs::kRoomLines;
        if (static_cast<size_t>(windows) * croom * 16 > dplane_bytes) std::exit(2);
        for (auto pick : {std::pair<int, int>{0, 1}, {0, 4}, {1, 4}}) {
        const int widen = pick.first, slots = pick.second;
        const auto claunch = [&](int cta, int rounds, int stride, int ring = 0) {
          if (widen == 0 && slots == 1) {
            const auto ck = coefs::plane<coefs::kThreads, 0, 1>;
            LAB_CHECK(cudaFuncSetAttribute(ck, cudaFuncAttributeMaxDynamicSharedMemorySize, coefs::kSmemBytes<1>));
            ck<<<cta, coefs::kThreads, (coefs::kSmemBytes<1>)>>>(b.times, ddense, drows, dmasks, ddplane, rounds, stride, croom, ring);
          } else if (widen == 0) {
            const auto ck = coefs::plane<coefs::kThreads, 0, 4>;
            LAB_CHECK(cudaFuncSetAttribute(ck, cudaFuncAttributeMaxDynamicSharedMemorySize, coefs::kSmemBytes<4>));
            ck<<<cta, coefs::kThreads, (coefs::kSmemBytes<4>)>>>(b.times, ddense, drows, dmasks, ddplane, rounds, stride, croom, ring);
          } else {
            const auto ck = coefs::plane<coefs::kThreads, 1, 4>;
            LAB_CHECK(cudaFuncSetAttribute(ck, cudaFuncAttributeMaxDynamicSharedMemorySize, coefs::kSmemBytes<4>));
            ck<<<cta, coefs::kThreads, (coefs::kSmemBytes<4>)>>>(b.times, ddense, drows, dmasks, ddplane, rounds, stride, croom, ring);
          }
        };
        const std::string cform = std::string("4 to 5 live digits a token, the block stream's coefficients in the step's format, every dense block's, its lists made from the mask, 512 threads")
                                  + (widen ? ", the dense digits widened alone" : "") + (slots == 4 ? ", four slots a trip" : "");
        LAB_CHECK(cudaMemset(ddplane, 0xa5, dplane_bytes));
        claunch(windows, 1, 0);
        LAB_CHECK(cudaDeviceSynchronize());
        std::vector<uint8_t> cgot(static_cast<size_t>(windows) * croom * 16);
        LAB_CHECK(cudaMemcpy(cgot.data(), ddplane, cgot.size(), cudaMemcpyDeviceToHost));
        const auto bf16 = [](float f) { uint32_t v; std::memcpy(&v, &f, 4); return static_cast<uint16_t>((v + 0x7fffu + (v >> 16 & 1u)) >> 16); };
        long long cwrong = 0, clines = 0;
        for (int w = 0; w < windows; ++w) {
          std::vector<std::vector<uint16_t>> by(kDigits);
          for (int d = 0; d < kDigits; ++d)
            for (int t = 0; t < kWindow; ++t)
              if (mask[(static_cast<size_t>(w) * kDigits + d) * 32 + t / 32] >> (t % 32) & 1u) by[d].push_back(static_cast<uint16_t>(t));
          for (int db = 0; db < kBlocks; ++db)
            for (int u = 0; u < coefs::kUnits; ++u) {
              const uint8_t* const runs = cgot.data() + (static_cast<size_t>(w) * croom + (static_cast<size_t>(db) * coefs::kUnits + u) * coefs::kRoomLines) * 16;
              int spreads = 0, first = 0;
              for (int j = 0; j < 8; ++j) spreads += blocks::spread(static_cast<int>(by[8 * u + j].size()));
              for (int j = 0; j < 8; ++j) {
                const int n = static_cast<int>(by[8 * u + j].size());
                for (int k = 0; k < blocks::spread(n); ++k)
                  for (int br = 0; br < 2; ++br, ++clines) {
                    uint16_t want[8] = {};
                    if (k < n) {
                      const size_t x = static_cast<size_t>(w) * kWindow + by[8 * u + j][k];
                      const float weight = widen ? 1.0f : static_cast<float>(tiled[(x / kTile) * kTileBytes + (u / 2) * kBlockBytes + (x % kTile) * 16 + (8 * u + j) % 16]);
                      for (int c = 0; c < 8; ++c)
                        want[c] = bf16(weight * static_cast<float>(hdense[(x / kTile) * kTileBytes + db * kBlockBytes + (x % kTile) * 16 + 8 * br + c]));
                    }
                    cwrong += std::memcmp(runs + (first + br * spreads + k) * 16, want, 16) != 0;
                  }
                first += blocks::spread(n);
              }
            }
        }
        b.check(cform, cwrong == 0, std::to_string(cwrong) + " of " + std::to_string(clines) + " coefficient lines differ");
        for (int quarter = 0; quarter < 2; ++quarter) {
          const int cta = quarter ? windows / 4 : windows;
          const lab::Samples cycles = b.variant([&] { claunch(cta, kRounds, windows); },
                                                [&] { return b.span(coefs::kThreads >> 5, cta) / kRounds / kWindow; });
          const std::string timed = cform + (quarter ? ", a quarter of the multiprocessors" : "") + ", a fresh window a round";
          b.unchecked(timed, "its kernel's first round is checked on every multiprocessor; this one is its time");
          b.record(timed, "SM-cycles a token", cycles);
        }
        if (widen == 0 && slots == 4) {
          //: TWO CONTROLS part a trip: the runs made and no copy out, and the copies out and no run made
          for (int part : {1, 2}) {
            const lab::Samples cycles = b.variant(
                [&] {
                  if (part == 1) {
                    const auto ck = coefs::plane<coefs::kThreads, 0, 4, 1>;
                    LAB_CHECK(cudaFuncSetAttribute(ck, cudaFuncAttributeMaxDynamicSharedMemorySize, coefs::kSmemBytes<4>));
                    ck<<<windows, coefs::kThreads, (coefs::kSmemBytes<4>)>>>(b.times, ddense, drows, dmasks, ddplane, kRounds, windows, croom, 0);
                  } else {
                    const auto ck = coefs::plane<coefs::kThreads, 0, 4, 2>;
                    LAB_CHECK(cudaFuncSetAttribute(ck, cudaFuncAttributeMaxDynamicSharedMemorySize, coefs::kSmemBytes<4>));
                    ck<<<windows, coefs::kThreads, (coefs::kSmemBytes<4>)>>>(b.times, ddense, drows, dmasks, ddplane, kRounds, windows, croom, 0);
                  }
                },
                [&] { return b.span(coefs::kThreads >> 5, windows) / kRounds / kWindow; });
            const std::string timed = cform + (part == 1 ? ", its runs made and no copy out" : ", its copies out and no run made") + ", a fresh window a round";
            b.unchecked(timed, "a control: a part of the kernel's trip absent");
            b.record(timed, "SM-cycles a token", cycles);
          }
          //: THE SAME KERNEL INTO A RING the cache holds: a CTA rewrites one dense block's rooms, 256 KB, as it goes --
          //: every multiprocessor's ring together under the cache's size
          constexpr int kRing = coefs::kUnits * coefs::kRoomLines;
          const lab::Samples cycles = b.variant([&] { claunch(windows, kRounds, windows, kRing); },
                                                [&] { return b.span(coefs::kThreads >> 5, windows) / kRounds / kWindow; });
          const std::string timed = cform + ", its plane a ring of 256 KB a CTA, a fresh window a round";
          b.unchecked(timed, "its kernel's first round is checked on every multiprocessor; a ring keeps no round to check");
          b.record(timed, "SM-cycles a token", cycles);
        }
        }
      }
    }
    for (int stream = 0; stream < 2; ++stream) {
    const std::vector<uint16_t>& L = all_lists[stream];
    const std::vector<uint32_t>& F = all_firsts[stream];
    LAB_CHECK(cudaMemcpy(dlists, L.data(), L.size() * 2, cudaMemcpyHostToDevice));
    LAB_CHECK(cudaMemcpy(dfirsts, F.data(), F.size() * 4, cudaMemcpyHostToDevice));
    const std::string cell_name = cell == 0 ? "4 to 5 live digits a token" : "32 live digits a token";
    //: a first round's planes against the host's: a real entry's lines, a pad's zero byte and zero dense lines
    const auto check = [&](bool is_values) {
      const size_t bytes = is_values ? vplane_bytes : dplane_bytes;
      std::vector<uint8_t> out(bytes);
      LAB_CHECK(cudaMemcpy(out.data(), is_values ? dvplane : ddplane, bytes, cudaMemcpyDeviceToHost));
      long long wrong = 0, lines = 0;
      const uint8_t zero[16] = {};
      for (int w = 0; w < windows; ++w) {
        const size_t steps = F[static_cast<size_t>(w) * (kBlocks + 1) + kBlocks];
        for (size_t s = 0; s < steps; ++s)
          for (int k = 0; k < 16; ++k) {
            const uint16_t e = L[(static_cast<size_t>(w) * kMax + s) * 16 + k];
            const bool pad = (e & kToken) == kPad;
            const size_t x = static_cast<size_t>(w) * kWindow + (e & kToken);
            if (is_values) {
              const uint8_t* const tile = out.data() + (static_cast<size_t>(w) * kMax + s) * (stream ? kLines<kRecords> : kLines<kTiled>) * 16;
              //: the step's sparse block is the one its first step index falls in
              int g = 0;
              while (F[static_cast<size_t>(w) * (kBlocks + 1) + g + 1] <= s) ++g;
              const uint8_t* const row = tiled.data() + (x / kTile) * kTileBytes + g * kBlockBytes + (x % kTile) * 16;
              ++lines;
              if (stream)
                wrong += std::memcmp(tile + (k * 9 + 8) * 16, pad ? zero : row, 16) != 0;
              else
                wrong += tile[kSparseLine * 16 + k] != (pad ? 0 : row[e >> 12]);
              if (!pad)
                for (int br = 0; br < 8; ++br, ++lines)
                  wrong += std::memcmp(tile + (stream ? k * 9 + br : br * 17 + k) * 16, hvalues.data() + x * kValueBytes + br * 16, 16) != 0;
            } else {
              for (int bl = 0; bl < kBlocks; ++bl, ++lines) {
                const uint8_t* const got = out.data() + ((static_cast<size_t>(w) * kBlocks + bl) * kMax + s) * 256 + k * 16;
                wrong += std::memcmp(got, pad ? zero : hdense.data() + (x / kTile) * kTileBytes + bl * kBlockBytes + (x % kTile) * 16, 16) != 0;
              }
            }
          }
      }
      return std::to_string(wrong) + " of " + std::to_string(lines) + " arranged lines differ";
    };
    const auto launch_values = [&](auto kernel, int blocks, int threads, int, int rounds, int stride, int out_stride) {
      LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, values::kSmemBytes));
      kernel<<<blocks, threads, values::kSmemBytes>>>(b.times, dvalues, drows, dlists, dfirsts, dvplane, rounds, stride, kMax, out_stride);
    };
    const auto launch_dense = [&](auto kernel, int blocks, int threads, int depth, int rounds, int stride, int out_stride) {
      const int bytes = depth == 3 ? dense::kSmemBytes<3> : dense::kSmemBytes<6>;
      LAB_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
      kernel<<<blocks, threads, bytes>>>(b.times, ddense, dlists, dfirsts, ddplane, rounds, stride, kMax, out_stride);
    };
    //: `few`: 0 every multiprocessor's CTA; 1 a quarter of them, the rest idle; 8 or 16 that many CTAs, their planes
    //: A RING each rewrites every round -- or, `few` negative, the same CTAs writing A NEW REGION a round, eight
    //: times the footprint; `depth`: the buffers out
    const auto run = [&](auto launcher, auto kernel, bool is_values, int threads, int move, int few, int depth) {
      const bool quarter = few == 1;
      const int blocks = few == 0 ? windows : quarter ? windows / 4 : few < 0 ? -few : few;
      const int out_stride = few < 0 ? blocks : 0;
      //: a new region a round stays inside the planes: as many rounds as the card's multiprocessors hold regions
      const int row_rounds = out_stride ? std::min<int>(kRounds, windows / blocks) : kRounds;
      const std::string name = cell_name + ", "
                               + (stream ? (is_values ? "the stream's values and sparse lines" : "the stream's dense lines")
                                         : (is_values ? "the values plane" : "the dense plane")) + ", "
                               + std::to_string(threads) + " threads, " + (move ? "its lines moved" : "a flat stream of its copies")
                               + (depth == 6 ? ", six buffers out" : "") + (quarter ? ", a quarter of the multiprocessors" : "")
                               + (few > 1 ? ", " + std::to_string(blocks) + " multiprocessors, its plane a ring"
                                  : few < 0 ? ", " + std::to_string(blocks) + " multiprocessors, a new region a round" : "");
      const auto launch = [&](int rounds, int stride) { launcher(kernel, blocks, threads, depth, rounds, stride, out_stride); };
      if (move && few == 0) {
        LAB_CHECK(cudaMemset(is_values ? dvplane : ddplane, 0xa5, is_values ? vplane_bytes : dplane_bytes));
        launch(1, 0);
        LAB_CHECK(cudaDeviceSynchronize());
        const std::string detail = check(is_values);
        b.check(name, detail[0] == '0' && detail[1] == ' ', detail);
      }
      for (int fresh = 0; fresh < 2; ++fresh) {
        const lab::Samples cycles = b.variant([&] { launch(row_rounds, fresh ? windows : 0); },
                                              [&] { return b.span(threads >> 5, blocks) / row_rounds / kWindow; });
        const std::string timed = name + (fresh ? ", a fresh window a round" : ", its rounds the same windows");
        b.unchecked(timed, move ? "its kernel's first round is checked on every multiprocessor; this one is its time"
                                : "a flat stream moves no line: its copies in and out are the floor");
        b.record(timed, "SM-cycles a token", cycles);
      }
    };
    const auto both = [&](auto launcher, auto flat, auto moved, bool is_values, int threads, int few, int depth) {
      run(launcher, flat, is_values, threads, kFlat, few, depth);
      run(launcher, moved, is_values, threads, kMoved, few, depth);
    };
    if (stream) {
      both(launch_values, values::plane<512, kFlat, kRecords>, values::plane<512, kMoved, kRecords>, true, 512, 0, 3);
      both(launch_values, values::plane<512, kFlat, kRecords>, values::plane<512, kMoved, kRecords>, true, 512, 1, 3);
      both(launch_dense, dense::plane<512, kFlat, 3>, dense::plane<512, kMoved, 3>, false, 512, 0, 3);
      both(launch_dense, dense::plane<512, kFlat, 3>, dense::plane<512, kMoved, 3>, false, 512, 1, 3);
      for (int few : {16, -16, 8, -8}) {
        both(launch_values, values::plane<512, kFlat, kRecords>, values::plane<512, kMoved, kRecords>, true, 512, few, 3);
        both(launch_dense, dense::plane<512, kFlat, 3>, dense::plane<512, kMoved, 3>, false, 512, few, 3);
      }
      continue;
    }
    both(launch_values, values::plane<256, kFlat, kTiled>, values::plane<256, kMoved, kTiled>, true, 256, 0, 3);
    both(launch_values, values::plane<512, kFlat, kTiled>, values::plane<512, kMoved, kTiled>, true, 512, 0, 3);
    both(launch_values, values::plane<1024, kFlat, kTiled>, values::plane<1024, kMoved, kTiled>, true, 1024, 0, 3);
    both(launch_values, values::plane<512, kFlat, kTiled>, values::plane<512, kMoved, kTiled>, true, 512, 1, 3);
    both(launch_dense, dense::plane<256, kFlat, 3>, dense::plane<256, kMoved, 3>, false, 256, 0, 3);
    both(launch_dense, dense::plane<512, kFlat, 3>, dense::plane<512, kMoved, 3>, false, 512, 0, 3);
    both(launch_dense, dense::plane<1024, kFlat, 3>, dense::plane<1024, kMoved, 3>, false, 1024, 0, 3);
    both(launch_dense, dense::plane<512, kFlat, 3>, dense::plane<512, kMoved, 3>, false, 512, 1, 3);
    both(launch_dense, dense::plane<512, kFlat, 6>, dense::plane<512, kMoved, 6>, false, 512, 0, 6);
    both(launch_dense, dense::plane<512, kFlat, 6>, dense::plane<512, kMoved, 6>, false, 512, 1, 6);
    }
  }
  return 0;
}
