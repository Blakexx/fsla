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
// THE STREAM (the ruled form) is the same two kernels fed lists BY SPARSE BLOCK, not by digit: a block's live tokens
// once each, in token order, padded to 16 only at the block's end. Its values kernel writes RECORDS -- a token's 8
// value lines and then its line of the block's 16 sparse bytes, 9 lines a token -- which the block's CTAs share;
// its dense plane is the block-major one, a CTA's dense block of its sparse block's tokens one run. No digit's
// step is padded in global memory: a CTA places a token on its digits' steps in shared memory by the mask. A carry's
// CTAs read ONE window's stream at a time, so the pass's planes can be A RING the cache holds: the stream's rows
// on 16 and on 8 multiprocessors, each CTA rewriting its region every round, and the same CTAs writing a new
// region a round -- the same reads, the planes past the cache.
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
  uint32_t* dfirsts = nullptr;
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
    //: A RING each rewrites every round (in the cache: 16 regions are 19 MB at most) -- or, `few` negative, the same
    //: CTAs writing A NEW REGION a round, past the cache; `depth`: the buffers out
    const auto run = [&](auto launcher, auto kernel, bool is_values, int threads, int move, int few, int depth) {
      const bool quarter = few == 1;
      const int blocks = few == 0 ? windows : quarter ? windows / 4 : few < 0 ? -few : few;
      const int out_stride = few < 0 ? blocks : 0;
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
        const lab::Samples cycles = b.variant([&] { launch(kRounds, fresh ? windows : 0); },
                                              [&] { return b.span(threads >> 5, blocks) / kRounds / kWindow; });
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
