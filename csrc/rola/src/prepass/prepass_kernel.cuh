// csrc/rola/src/prepass/prepass_kernel.cuh -- THE PRE-PASS KERNEL: the int8 carry's skeleton at two
// columns a leaf. THE FRAME IS REAL AND THE COMPONENTS ARE STUBS WITH THEIR EDGES (KERNEL_STANDARDS
// section 19): the words staged a window ahead, the list counts, the pool filled a chunk ahead through
// two barriered slots, the fold over the warp's lists, the publish and rescale across the transfer, the
// token-major readout into the panel, the panel drained by RED.F32. Each component lands against its own
// budget line in the queue card; until it does, its stub touches its operands and produces zero.
// Every register array is indexed at compile time (section 20): no local memory.
// See docs/internals/prepass/prepass.md
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "common/geom.cuh"
#include "common/ops.cuh"
#include "common/phase_clock.cuh"
#include "common/static_for.cuh"
#include "facts/liveness_contract.cuh"
#include "prepass/params.cuh"
#include "prepass/plan.cuh"

namespace rola::prepass {

namespace ops = denseref::ops;
using Clock = rola::PhaseClock<kPpPhases>;

//: THE SHARED LEDGER'S ADDRESSES, from the plan's offsets. -- see docs/internals/prepass/prepass.md#smem
template <class TP>
struct Smem {
  char* base;

  __device__ __forceinline__ ops::SmemAddr at(int off) const { return ops::smem_addr(base + off); }

  __device__ __forceinline__ ops::SmemAddr words(int parity) const {
    return at(TP::kWordsOffset + parity * TP::kWordsBytes);
  }

  __device__ __forceinline__ ops::SmemAddr pool(int slot) const {
    return at(TP::kPoolOffset + slot * TP::kPoolSlotBytes);
  }

  __device__ __forceinline__ ops::SmemAddr ring(int warp, int slot) const {
    return at(TP::kRingOffset + (warp * TP::kRingSlots + slot) * TP::kRingSlotBytes);
  }

  __device__ __forceinline__ ops::SmemAddr xfer_values() const { return at(TP::kXferOffset); }

  __device__ __forceinline__ ops::SmemAddr xfer_scales() const {
    return at(TP::kXferOffset + TP::kXferValueBytes);
  }

  __device__ __forceinline__ ops::SmemAddr panel(int warp) const {
    return at(TP::kPanelOffset + warp * TP::kPanelRowBytes);
  }

  __device__ __forceinline__ ops::SmemAddr pool_full(int slot) const {
    return at(TP::kBarOffset + slot * 8);
  }

  __device__ __forceinline__ ops::SmemAddr pool_empty(int slot) const {
    return at(TP::kBarOffset + (TP::kPoolSlots + slot) * 8);
  }

  __device__ __forceinline__ ops::SmemAddr ring_bar(int warp, int slot) const {
    return at(TP::kBarOffset + (TP::kPoolBars + warp * TP::kRingSlots + slot) * 8);
  }

  __device__ __forceinline__ ops::SmemAddr words_bar(int parity) const {
    return at(TP::kBarOffset + (TP::kPoolBars + TP::kRingBars + parity) * 8);
  }

  __device__ __forceinline__ ops::SmemAddr pub_bar(int parity) const {
    return at(TP::kBarOffset + (TP::kPoolBars + TP::kRingBars + 2 + parity) * 8);
  }

  __device__ __forceinline__ ops::SmemAddr done_bar(int parity) const {
    return at(TP::kBarOffset + (TP::kPoolBars + TP::kRingBars + 4 + parity) * 8);
  }

  __device__ __forceinline__ long long* ledger(int warp) const {
    return reinterpret_cast<long long*>(base + TP::kLedgerOffset) + warp * kPpPhases;
  }

  __device__ __forceinline__ int* counts(int parity, int warp) const {
    return reinterpret_cast<int*>(base + TP::kCountOffset + parity * TP::kCountBytes)
           + warp * (TP::kLeavesPerWarp / 8 + TP::kBlocksPerWarp);
  }
};

//: THE FIRST DIGIT of level `l` in owner `O`'s region -- the runtime owner law, the tall set's.
__device__ __forceinline__ int region_digit0(const carry::CarryGeomRT& g, int O, int l) {
  return ((O >> g.owner_div_bits[l]) & (g.grid[l] - 1)) << g.span_bits[l];
}

//: A STREAM'S k-TH DIGIT of level `l` on side `s`, the region-relative digit the row map spells.
__device__ __forceinline__ int stream_digit(const carry::SideGeom& s, int l, int strm, int k) {
  return carry::geom_level_digit(s, l, strm, k);
}

//: THE REGION'S STAGED ROWS: the sparse levels' digit rows on a side, read side first. Every index is
//: compile-time, so the table is registers.
template <class TP>
struct StagedRows {
  int level_row[2]
               [carry::kGeomMaxLevels];  //: a sparse level's first staged row on that side, or -1
  int rl, wl;          //: the read side's and the write side's sparse level (the last)
  int rl_row, wl_row;  //: its first staged row
};

template <class TP>
__device__ __forceinline__ StagedRows<TP> staged_rows(const PrepassParams& p) {
  StagedRows<TP> r{};
  r.rl = r.wl = -1;
  r.rl_row = r.wl_row = 0;
  int at = 0;
  rola::static_for<2>([&](auto Sc) {
    constexpr int side = decltype(Sc)::value;
    rola::static_for<carry::kGeomMaxLevels>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      const bool sparse = l < p.g.D && ((p.sparse_mask[side] >> l) & 1u) != 0u;
      r.level_row[side][l] = sparse ? at : -1;
      if (sparse) {
        if (side == 0) {
          r.rl = l;
          r.rl_row = at;
        } else {
          r.wl = l;
          r.wl_row = at;
        }
        at += p.g.span[l];
      }
    });
  });
  return r;
}

// ------------------------------------------------------------------------ the words
//: COMPONENT `words`: the staged rows' words for window `win`, by cp.async, lane = word; landed at the
//: words barrier's next phase. One warp issues it. Budget: prepass.md#budgets.
template <class TP>
__device__ __forceinline__ void stage_words(const PrepassParams& p, const Smem<TP>& sm,
                                            const StagedRows<TP>& rows, int owner, int bh, int win,
                                            int parity, int lane) {
  const int t0 = win * TP::kWindow;
  const int left = (p.L - t0 + 31) / 32;
  const int nwords = left < TP::kRounds ? left : TP::kRounds;
  const int word = lane & 15, half = lane >> 4;
  rola::static_for<2>([&](auto Sc) {
    constexpr int side = decltype(Sc)::value;
    const int32_t* const table =
        p.liveness + ((long)bh * 2 + side) * ((long)p.g.wtot * p.liveness_words) + (t0 >> 5);
    rola::static_for<carry::kGeomMaxLevels>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      const int first = rows.level_row[side][l];
      const int d0 = region_digit0(p.g, owner, l);
      const int span = first < 0 ? 0 : p.g.span[l];
#pragma unroll 1
      for (int i = half; i < span; i += 2) {
        const int row = p.g.col_base[l] + d0 + i;
        const ops::SmemAddr dst =
            sm.words(parity) + (uint32_t)((first + i) * TP::kRounds + word) * 4u;
        ops::stage_run_if<4>(dst, table + (long)row * p.liveness_words + word, word < nwords);
      }
    });
  });
  ops::stage_commit();
  ops::mbar_arrive_when_landed(sm.words_bar(parity));
}

// ------------------------------------------------------------------------ the lists
//: COMPONENT `lists`: a warp's list counts from the staged words -- its 16 fold lists (one a write-sparse
//: digit) and its read side's union over its 4 sparse digits, by popcount. Lane = word of the window.
template <class TP>
__device__ __forceinline__ void list_counts(const PrepassParams& p, const Smem<TP>& sm,
                                            const StagedRows<TP>& rows, int parity, int warp,
                                            int lane) {
  int* const out = sm.counts(parity, warp);
  constexpr int nlists = TP::kLeavesPerWarp / 8;
  constexpr int kRounds = TP::kRounds, kBlocks = TP::kBlocksPerWarp;
  static_assert(nlists <= 32, "one lane a fold list");
  //: the fold lists, ONE LANE A LIST: lane k sums the popcounts of its digit's sixteen words.
  const int wl = rows.wl;
  const int wlists = wl >= 0 ? (1 << p.g.w.sub_bits[wl]) : 0;
  int count = 0;
  if (lane < wlists) {
    const int digit = stream_digit(p.g.w, wl, warp, lane);
    const ops::SmemAddr row =
        sm.words(parity) + (uint32_t)((rows.wl_row + digit) * TP::kRounds) * 4u;
#pragma unroll
    for (int w = 0; w < kRounds; ++w)
      count += ops::bit_count(ops::load_shared_u32(row + (uint32_t)w * 4u));
  }
  if (lane < nlists) out[lane] = count;
  //: the read union: the warp's sparse digits ORed a round (lane = round), the token-major tile count.
  const int rl = rows.rl;
  uint32_t any = 0u;
  if (rl >= 0 && lane < TP::kRounds) {
    const int rlists = 1 << p.g.r.sub_bits[rl];
#pragma unroll
    for (int k = 0; k < kBlocks; ++k) {
      if (k < rlists) {
        const int digit = stream_digit(p.g.r, rl, warp, k);
        any |= ops::load_shared_u32(sm.words(parity)
                                    + (uint32_t)((rows.rl_row + digit) * TP::kRounds + lane) * 4u);
      }
    }
  }
  int total = ops::bit_count(any);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) total += __shfl_xor_sync(0xFFFFFFFFu, total, o);
  if (lane == 0) out[nlists] = total;
}

// ------------------------------------------------------------------------- the pool
//: COMPONENT `fill`: chunk `chunk` of window `win` into slot `slot`: every token's write-sparse entries
//: over the region's digits, its write-dense entries, its two row scales, g and s_t, by cp.async; a slot
//: is full at `kThreads` landed arrivals. THE STUB stages the window's tokens in order (the compaction to
//: the union-live tokens is the component's own step).
template <class TP>
__device__ __forceinline__ void pool_fill(const PrepassParams& p, const Smem<TP>& sm,
                                          const StagedRows<TP>& rows, int owner, int bh, int win,
                                          int chunk, int slot, int tid) {
  const int t0 = win * TP::kWindow + chunk * TP::kPoolTok;
  const int tok = tid >> 2, part = tid & 3;  //: four threads a token
  const int t = t0 + tok;
  const bool live = t < p.L;
  const uint32_t row = (uint32_t)bh * (uint32_t)p.L + (uint32_t)(live ? t : 0);
  const ops::SmemAddr dst = sm.pool(slot) + (uint32_t)(tok * TP::kPoolRowBytes);
  const uint8_t* const wrow = p.rows_write + (long)row * p.g.wtot;
  //: the write-sparse level's entries: the region's digits, contiguous in the row.
  const int wl = rows.wl;
  if (wl >= 0) {
    const int col = p.g.col_base[wl] + region_digit0(p.g, owner, wl);
    const int span = p.g.span[wl];
#pragma unroll 1
    for (int c = part * 16; c < span; c += 64)
      ops::stage_run_if<16>(dst + (uint32_t)c, wrow + col + c, live);
  }
  //: the write-dense level's entries: the level that is not the sparse one, at depth two.
  const int wd = wl == 0 ? 1 : 0;
  if (part == 0 && wd < p.g.D) {
    const int col = p.g.col_base[wd] + region_digit0(p.g, owner, wd);
    ops::stage_run_if<8>(dst + (uint32_t)TP::kPoolSparseBytes, wrow + col, live);
  }
  if (part == 1)
    ops::stage_run_if<8>(dst + (uint32_t)(TP::kPoolSparseBytes + TP::kPoolDenseBytes),
                         p.scale_write + (long)row * p.g.D, live);
  if (part == 2) {
    ops::stage_run_if<4>(dst + (uint32_t)(TP::kPoolSparseBytes + TP::kPoolDenseBytes + 8),
                         p.gain + row, live);
    ops::stage_run_if<4>(dst + (uint32_t)(TP::kPoolSparseBytes + TP::kPoolDenseBytes + 12),
                         p.vscale + row, live);
  }
  ops::stage_commit();
  ops::mbar_arrive_when_landed(sm.pool_full(slot));
}

// ------------------------------------------------------------------------- the fold
//: THE WARP'S STATE: its 128 leaves' two columns, s32 -- the accumulator of the transposed fold's
//: m-tile, 16 lists x one n-tile x 4 registers (the m-tile's rows 4..15 are the pre-pass's padding).
template <class TP>
struct State {
  int32_t c[TP::kLeavesPerWarp / 8][4];
};

//: COMPONENT `fold`: the warp's lists over the chunk in the slot, one list a k16 IMMA step. THE STUB walks
//: each list's step count and touches the pool once a step; no MMA.
template <class TP>
__device__ __forceinline__ void fold(const Smem<TP>& sm, int slot, int parity, int warp, int lane,
                                     State<TP>& st) {
  const int* const counts = sm.counts(parity, warp);
  const ops::SmemAddr pool = sm.pool(slot);
  rola::static_for<TP::kLeavesPerWarp / 8>([&](auto Kc) {
    constexpr int k = decltype(Kc)::value;
    const int steps = (counts[k] / TP::kChunks + 15) / 16;  //: this chunk's share, in k16 steps
    int32_t touch = 0;
#pragma unroll 1
    for (int s = 0; s < steps; ++s) {
      const int tok = (s * 16 + (lane & 15)) & (TP::kPoolTok - 1);
      touch +=
          (int32_t)ops::load_shared_u8(pool + (uint32_t)(tok * TP::kPoolRowBytes + (lane >> 4)));
    }
    st.c[k][1] += touch;  //: the touched sum rides column 1, which the publish does not read
  });
}

// ---------------------------------------------------------------------- the transfer
//: COMPONENT `publish`: a fold warp's 128 leaves x 2 columns as s16 against each leaf's max, and the max
//: as a float, into the transfer in the READ layout. THE STUB writes the state's low halves and its maxima
//: from the s32 columns (zeros until the fold lands).
template <class TP>
__device__ __forceinline__ void publish(const Smem<TP>& sm, int warp, int lane,
                                        const State<TP>& st) {
  const uint32_t leaf0 = (uint32_t)(warp * TP::kLeavesPerWarp);
  uint32_t low = 0u;
  rola::static_for<TP::kLeavesPerWarp / 8>([&](auto Kc) {
    constexpr int k = decltype(Kc)::value;
    if ((lane >> 1) == k) low = (uint32_t)(st.c[k][0] & 0xFFFF);
  });
  const uint4 vals = make_uint4(low, 0u, 0u, 0u);
  //: 128 leaves x 2 columns x 2 B = 512 B a warp: one 16-byte store a lane; the maxima 1 KB: two.
  ops::store_shared_vec16(sm.xfer_values() + (leaf0 * TP::kCols * 2u) + (uint32_t)lane * 16u, vals);
  const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
  ops::store_shared_vec16(sm.xfer_scales() + (leaf0 * TP::kCols * 4u) + (uint32_t)lane * 16u, zero);
  ops::store_shared_vec16(sm.xfer_scales() + (leaf0 * TP::kCols * 4u) + 512u + (uint32_t)lane * 16u,
                          zero);
}

//: COMPONENT `rescale`: a read warp's 4 blocks from the transfer: each block's scale = the max of its 32
//: leaves' maxima (one a lane, five shuffles), its live min beside it, the columns requantized into the
//: block's unit; the hi bytes stay in registers. THE STUB performs the loads and the reductions.
template <class TP>
__device__ __forceinline__ void rescale(const Smem<TP>& sm, int warp, int lane,
                                        float (&sigma)[TP::kBlocksPerWarp],
                                        uint32_t (&hi)[TP::kBlocksPerWarp]) {
  rola::static_for<TP::kBlocksPerWarp>([&](auto Bc) {
    constexpr int b = decltype(Bc)::value;
    const uint32_t leaf = (uint32_t)((warp * TP::kBlocksPerWarp + b) * TP::kBlockLeaves + lane);
    const float m0 = ops::load_shared_f32(sm.xfer_scales() + leaf * (TP::kCols * 4u));
    const float m1 = ops::load_shared_f32(sm.xfer_scales() + leaf * (TP::kCols * 4u) + 4u);
    float mx = fmaxf(m0, m1);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, o));
    sigma[b] = mx;
    hi[b] = ops::load_shared_u32(sm.xfer_values() + leaf * (TP::kCols * 2u));
  });
}

// ---------------------------------------------------------------------- the readout
//: COMPONENT `readout`: the warp's union list, a 32-token tile at a time: the tile's read rows staged into
//: the ring (the dense sub-block's entries, the sparse entries, the scales), the per-block partials, the
//: two per-token sums into the panel. THE STUB stages the range's tokens in order, waits the ring, and
//: adds zeros into the panel.
//: THE TILE'S ROWS INTO THE RING, issued; the ring barrier arrives when they land.
template <class TP>
__device__ __forceinline__ void readout_issue(const PrepassParams& p, const Smem<TP>& sm,
                                              const StagedRows<TP>& rows, int owner, int bh, int t0,
                                              int tile, int rslot, int warp, int lane) {
  const int t = t0 + tile * TP::kTile + lane;
  const bool live = t < p.L;
  const uint32_t row = (uint32_t)bh * (uint32_t)p.L + (uint32_t)(live ? t : 0);
  const ops::SmemAddr dst = sm.ring(warp, rslot) + (uint32_t)(lane * TP::kRingRowBytes);
  const uint8_t* const rrow = p.rows_read + (long)row * p.g.wtot;
  const int rl = rows.rl;
  const int rd = rl == 0 ? 1 : 0;
  if (rd < p.g.D) {
    //: the warp's dense sub-block: its stream's first digit on the dense level.
    const int col =
        p.g.col_base[rd] + region_digit0(p.g, owner, rd) + stream_digit(p.g.r, rd, warp, 0);
    ops::stage_run_if<16>(dst, rrow + col, live);
    ops::stage_run_if<16>(dst + 16u, rrow + col + 16, live);
  }
  if (rl >= 0) {
    const int col =
        p.g.col_base[rl] + region_digit0(p.g, owner, rl) + stream_digit(p.g.r, rl, warp, 0);
    ops::stage_run_if<4>(dst + (uint32_t)TP::kBlockLeaves, rrow + col, live);
  }
  ops::stage_run_if<8>(dst + (uint32_t)TP::kBlockLeaves + 8u, p.scale_read + (long)row * p.g.D,
                       live);
  ops::stage_commit();
  ops::mbar_arrive_when_landed(sm.ring_bar(warp, rslot));
}

//: COMPONENT `readout`: a landed tile: the per-block partials and the two per-token sums into the panel.
//: THE STUB adds zeros; the hi bytes are the operand.
template <class TP>
__device__ __forceinline__ void readout_tile(const Smem<TP>& sm, int tile, int rslot,
                                             uint32_t rphase, int parity, int warp, int lane,
                                             const uint32_t (&hi)[TP::kBlocksPerWarp]) {
  ops::mbar_wait(sm.ring_bar(warp, rslot), rphase);
  const ops::SmemAddr dst = sm.ring(warp, rslot) + (uint32_t)(lane * TP::kRingRowBytes);
  const uint32_t a = ops::load_shared_u32(dst);
  const float partial = (float)((a & hi[0] & hi[1] & hi[2] & hi[3]) & 0u);
  //: the warp's own panel row: plain stores, summed across the warps at the drain.
  const uint32_t in_range = (uint32_t)((tile * TP::kTile + lane) & (TP::kRange - 1));
  ops::store_shared_f32x2(sm.panel(warp) + in_range * 8u, partial, partial);
  (void)parity;
}

//: COMPONENT `drain`: the range's eight warp rows summed a token, to the global sums by RED.F32 (256
//: threads: a token's column each); the rows are overwritten by the next range, never zeroed.
template <class TP>
__device__ __forceinline__ void drain(const PrepassParams& p, const Smem<TP>& sm, int bh, int t0,
                                      int range, int tid) {
  const int i = tid >> 1, col = tid & 1;  //: 128 tokens x 2 columns over 256 threads
  const int t = t0 + range * TP::kRange + i;
  constexpr int kWarps = TP::kWarps;
  float v = 0.0f;
#pragma unroll
  for (int w = 0; w < kWarps; ++w)
    v += ops::load_shared_f32(sm.panel(w) + (uint32_t)(i * 8 + col * 4));
  if (t < p.L) ops::red_global_add_f32((col == 0 ? p.den : p.scale) + (long)bh * p.L + t, v);
}

// ------------------------------------------------------------------------ the frame
//: THE WINDOW LOOP: a warp's program. Words a window ahead (one warp stages, all wait), the list counts,
//: the chunks filled a slot ahead and folded, the publish, the rescale after the publish barrier, the
//: ranges read out and drained. -- see docs/internals/prepass/prepass.md#frame
template <class TP>
__device__ __forceinline__ void window_loop(const PrepassParams& p, const Smem<TP>& sm, int owner,
                                            int bh, int tid, int warp, int lane, Clock& pc) {
  const StagedRows<TP> rows = staged_rows<TP>(p);
  const int nWindows = (p.L + TP::kWindow - 1) / TP::kWindow;
  const bool scanner = warp == TP::kWarps - 1;
  State<TP> st{};
  float sigma[TP::kBlocksPerWarp];
  uint32_t hi[TP::kBlocksPerWarp] = {0u, 0u, 0u, 0u};
  uint32_t fills0 = 0u, fills1 = 0u;
  uint32_t rings = 0u;

  if (tid < TP::kPoolSlots) {
    ops::mbar_init(sm.pool_full(tid), TP::kThreads);
    ops::mbar_init(sm.pool_empty(tid), TP::kWarps);
  }
  if (lane < TP::kRingSlots) ops::mbar_init(sm.ring_bar(warp, lane), 32u);
  if (tid < 2) {
    ops::mbar_init(sm.words_bar(tid), 32u);
    ops::mbar_init(sm.pub_bar(tid), TP::kThreads);
    ops::mbar_init(sm.done_bar(tid), TP::kThreads);
  }
  ops::rendezvous();

  if (scanner) stage_words<TP>(p, sm, rows, owner, bh, 0, 0, lane);
  pc.lap(kPpWords);

#pragma unroll 1
  for (int win = 0; win < nWindows; ++win) {
    const int parity = win & 1, t0 = win * TP::kWindow;
    const uint32_t phase = (uint32_t)(win >> 1) & 1u;
    ops::mbar_wait(sm.words_bar(parity), phase);
    pc.lap(kPpFrame);
    list_counts<TP>(p, sm, rows, parity, warp, lane);
    ops::rendezvous();
    pc.lap(kPpLists);
    if (scanner && win + 1 < nWindows) {
      stage_words<TP>(p, sm, rows, owner, bh, win + 1, parity ^ 1, lane);
      pc.lap(kPpWords);
    }
    //: the chunks, a slot ahead: fill c+1 while c is folded.
    pool_fill<TP>(p, sm, rows, owner, bh, win, 0, 0, tid);
    ++fills0;
    pc.lap(kPpFill);
#pragma unroll 1
    for (int c = 0; c < TP::kChunks; ++c) {
      const int slot = c & 1, nslot = slot ^ 1;
      if (c + 1 < TP::kChunks) {
        const uint32_t nfills = nslot ? fills1 : fills0;
        if (nfills > 0u) ops::mbar_wait(sm.pool_empty(nslot), (nfills - 1u) & 1u);
        pool_fill<TP>(p, sm, rows, owner, bh, win, c + 1, nslot, tid);
        if (nslot)
          ++fills1;
        else
          ++fills0;
        pc.lap(kPpFill);
      }
      ops::mbar_wait(sm.pool_full(slot), ((slot ? fills1 : fills0) - 1u) & 1u);
      pc.lap(kPpFrame);
      fold<TP>(sm, slot, parity, warp, lane, st);
      if (lane == 0)
        ops::mbar_arrive(sm.pool_empty(slot));  //: one arrival a warp: the barrier counts warps
      pc.lap(kPpFold);
    }
    publish<TP>(sm, warp, lane, st);
    ops::mbar_arrive(sm.pub_bar(parity));
    pc.lap(kPpPublish);
    ops::mbar_wait(sm.pub_bar(parity), phase);
    pc.lap(kPpFrame);
    rescale<TP>(sm, warp, lane, sigma, hi);
    pc.lap(kPpRescale);
    //: the ring a tile ahead: tile 0 issued, then each tile's successor before its own wait.
    readout_issue<TP>(p, sm, rows, owner, bh, t0, 0, (int)(rings & 1u), warp, lane);
#pragma unroll 1
    for (int range = 0; range < TP::kRanges; ++range) {
#pragma unroll 1
      for (int tile = 0; tile < TP::kRange / TP::kTile; ++tile) {
        const int abs_tile = range * (TP::kRange / TP::kTile) + tile;
        const int rslot = (int)(rings & 1u);
        if (abs_tile + 1 < TP::kTiles)
          readout_issue<TP>(p, sm, rows, owner, bh, t0, abs_tile + 1, rslot ^ 1, warp, lane);
        readout_tile<TP>(sm, abs_tile, rslot, (rings >> 1) & 1u, parity, warp, lane, hi);
        ++rings;
      }
      pc.lap(kPpReadout);
      ops::rendezvous();
      drain<TP>(p, sm, bh, t0, range, tid);
      ops::rendezvous();
      pc.lap(kPpDrain);
    }
    (void)sigma;
  }
}

// ----------------------------------------------------------------------- the kernel
template <int D>
__global__ __launch_bounds__(TallPlan<D>::kThreads, 1) void prepass_kernel(
    __grid_constant__ const PrepassParams p) {
  using TP = TallPlan<D>;
  extern __shared__ __align__(128) char smem_raw[];
  const Smem<TP> sm{smem_raw};
  const int owner = (int)blockIdx.x, bh = (int)blockIdx.y;
  const int tid = (int)threadIdx.x;
  const int warp = ops::uniform_warp<TP::kWarps>(), lane = tid & 31;
  Clock pc(sm.ledger(warp), lane);
  const long cta = (long)bh * gridDim.x + owner;
  if (p.trace != nullptr && cta < p.trace_ctas)
    pc.bind_trace(p.trace + (cta * TP::kWarps + warp) * p.trace_cap, p.trace_cap);
  window_loop<TP>(p, sm, owner, bh, tid, warp, lane, pc);
  pc.lap(kPpFrame);
  pc.publish(p.ledger, cta, TP::kWarps, warp);
}

}  // namespace rola::prepass
