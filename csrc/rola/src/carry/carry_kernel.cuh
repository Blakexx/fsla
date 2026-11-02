// csrc/rola/src/carry/carry_kernel.cuh -- THE CARRY KERNEL: one CTA a box, window by window in one fixed order: the
// first read chunk's fill issued and the next window's inputs staged under the maxima, then each read chunk (its image
// whole or in halves, its units, its tiles, its drain) and each fold chunk (its fill, the fold) from the window's plan
// (`schedule.cuh`), the fp32 state in registers throughout. see docs/internals/carry/carry_kernel.md
#pragma once

#include "carry/fold.cuh"
#include "carry/schedule.cuh"
#include "common/phase_clock.cuh"

namespace rola::carry {

//: THE WINDOW'S INPUTS, a window ahead, by every thread: both sides' lists, the write exponents and token units, and
//: the window's plan record for the CTA's box class. -- see docs/internals/carry/carry_kernel.md#staging
template <class P>
__device__ __forceinline__ void stage_window(const CarryParams& p, const Place& c, int w,
                                             uint8_t* smem) {
  const Window rw = window_of<P>(p, smem, kRead, w), ww = window_of<P>(p, smem, kWrite, w);
  lists_in<P>(p, p.read_lists, rw, c, w, smem, kRead);
  lists_in<P>(p, p.write_lists, ww, c, w, smem, kWrite);
  facts_in<P>(p, ww, c, w, smem);
  const int words = p.tables.plan_words, copies = words * 4 / ops::kStageBytes;
  const int t = ops::opaque(static_cast<int>(threadIdx.x)) - (kThreads - 1 - copies);
  if (t >= 0 && t < copies) {
    const int cls = rw.group * max(1, p.write_lists.groups) + ww.group;
    ops::stage_tile(
        ops::smem_addr(smem + p.tables.plans_at) + (w & 1) * words * 4 + 16 * t,
        p.plans + ((static_cast<int64_t>(c.bh) * p.windows + w) * box_classes(p) + cls) * words
            + 4 * t);
  }
}

//: THE KERNEL, a grid constant block in, one CTA an SM: grid `(boxes, batch-heads)`. A window: its first read chunk's
//: fill, the next window's staging, the maxima, its read chunks, its fold chunks -- a fixed order, the plan's chunks
//: and places read once a window. -- see docs/internals/carry/carry_kernel.md#kernel
template <class Arm>
__global__ __launch_bounds__(kThreads, 1) void carry_kernel(const __grid_constant__ CarryParams p) {
  using P = Plan<Arm>;
  extern __shared__ __align__(128) uint8_t smem[];
  //: the launch sized for this card's plan, or nothing runs (`carry_arm.md#plan`)
  uint32_t launched;
  asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(launched));
  if (launched != static_cast<uint32_t>(P::kSmemBytes)) __trap();
  const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
  //: the CTA's place where a part takes it: its batch-head re-read at each use (`Addr` holds its channel), so nothing
  //: holds a register across the windows for it
  const auto here = [&] { return Place{static_cast<int>(blockIdx.y), 0}; };
  PhaseClock<kCarryPhases, true> clock(
      reinterpret_cast<long long*>(smem + P::kClock) + warp * PhaseClock<kCarryPhases, true>::kRow,
      lane, p.ledger != nullptr);
  if (static_cast<int>(threadIdx.x) < kBoxLists)
    reinterpret_cast<UnitBook*>(smem + P::kBook)[threadIdx.x] = UnitBook{kUnitNone, 0.0f};
  if (static_cast<int>(threadIdx.x) < 2 * kBoxDigits)
    reinterpret_cast<int*>(smem + P::kKeys)[threadIdx.x] = INT_MIN;
  //: the listed bounds' counters, each cleared again by the chunk that reads it; the units defined before any read (a
  //: masked lane reads index 0)
#pragma unroll 1
  for (int i = static_cast<int>(threadIdx.x); i < p.tables.heads * kWindow; i += kThreads) {
    reinterpret_cast<uint32_t*>(smem + p.tables.bounds_at)[i] = 0u;
    reinterpret_cast<int8_t*>(smem + p.tables.units_at)[i] = static_cast<int8_t>(kUnitByteNone);
  }
  //: the scales' bits' multiplier and bias (`list_scale`), loaded at each use
  if (threadIdx.x == 0)
    *reinterpret_cast<uint2*>(smem + P::kBits) = make_uint2(1u << 23, 127u << 23);
  if (threadIdx.x == 0) addr_in(*reinterpret_cast<Addr*>(smem + P::kAddr), p, place_of(p));
  ops::rendezvous();

  //: the warp and the lane, opaque at each use, so each phase forms its own addresses where it runs rather than
  //: every phase's held across the window (KERNEL_STANDARDS §32.3)
  const auto wid = [] { return ops::opaque(static_cast<int>(threadIdx.x)) / 32; };
  const auto lid = [] { return ops::opaque(static_cast<int>(threadIdx.x)) % 32; };
  const ops::SmemAddr area = ops::smem_addr(smem + P::kAreaAt);
  const auto rw = [&](int w) { return window_of<P>(p, smem, kRead, w); };
  const auto ww = [&](int w) { return window_of<P>(p, smem, kWrite, w); };
  //: A CHUNK'S PLACE in the area, the plan's (`schedule.md#steps`)
  const auto place_at = [&](int w, int t) {
    return area + static_cast<uint32_t>(step_of<P>(p, smem, w, t).at());
  };
  //: A CHUNK'S RUNS, lane `l` list `l`'s, worked out once a chunk for its fill, its units and its tiles or fold
  const auto runs_of = [&](int side, int w, const Entry& e) {
    return chunk_runs<P>(p, smem, side == kRead ? rw(w) : ww(w), side, w, e.first(), e.slots(),
                         e.total(), lid());
  };
  //: A CHUNK'S COPIES issued into its place, one group
  const auto fill_read = [&](int w, const Entry& e, const Run& run, ops::SmemAddr buf) {
    read_fill<P>(p, smem, rw(w), here(), w, run, e.first(), e.slots(), buf, wid(), lid());
    ops::stage_commit();
  };
  const auto fill_fold = [&](int w, const Entry& e, const Run& run, ops::SmemAddr buf) {
    fold_fill<P>(p, smem, ww(w), here(), w, run, e.first(), e.slots(), buf, wid(), lid());
    ops::stage_commit();
  };
  //: THE COPIES LANDED: every group but the staging's where that was committed after the chunk's, else every group
  const auto landed = [&](bool staging_after) {
    if (staging_after) {
      ops::stage_wait<1>();
    } else {
      ops::stage_wait<0>();
    }
  };
  //: A READ CHUNK'S UNITS, the CTA's own, before its first adds (the arm's read rows say which side: one factor a
  //: listed side's row, two a dense side's row and coefficients): the terms or the bounds, a barrier, the bytes and the
  //: drain's factors, a barrier
  const auto chunk_units = [&](int w, const Entry& e, const Run& run, ops::SmemAddr buf) {
    if constexpr (P::kReadFactors == 1) {
      units_listed_add<P>(p, smem, run, e.first(), buf,
                          buf + static_cast<uint32_t>(read_ents<P>(e.slots())), wid(), lid());
    } else {
      units_dense<P>(p, smem, e.first(), e.slots(), buf, wid(), lid());
    }
    ops::rendezvous();
    units_close<P>(p, smem, w, e.first(), e.slots());
    ops::rendezvous();
  };
  //: A READ CHUNK'S TILES over batches `lo .. hi - 1` of the image at `area` (digit `d`'s plane `d & mask`)
  const auto read_out = [&](int w, const Entry& e, const Run& run, ops::SmemAddr buf, int lo,
                            int hi, uint32_t mask) {
    const ops::SmemAddr pads = buf + static_cast<uint32_t>(read_pads<P>(e.slots())),
                        ents = buf + static_cast<uint32_t>(read_ents<P>(e.slots()));
    if (p.readout_f16) {
      read_chunk<P, true, true>(p, smem, rw(w), run, e.first(), e.slots(), lo, hi, area, mask, buf,
                                pads, ents, wid(), lid());
    } else {
      read_chunk<P, false, true>(p, smem, rw(w), run, e.first(), e.slots(), lo, hi, area, mask, buf,
                                 pads, ents, wid(), lid());
    }
  };

  State<P> st;
  static_for<P::kFoldLists>([&](auto J) {
    static_for<P::kFoldMTiles>([&](auto M) {
      static_for<4>([&](auto N) { static_for<4>([&](auto I) { st.acc[J][M][N][I] = 0.f; }); });
    });
  });
  stage_window<P>(p, here(), 0, smem);
  ops::stage_commit();
  ops::stage_wait<0>();
  ops::rendezvous();

#pragma unroll 1
  for (int w = 0; w < p.windows; ++w) {
    const Stream s = stream_of<P>(p, smem, w);
    const bool more = w + 1 < p.windows;
    //: every warp done with the window before (its fold's rows may be where this window's first chunk lands)
    ops::rendezvous();
    clock.lap(kPhaseWait);
    //: THE KEYS before any copy of the window is issued (their shared loads ahead of the copies), then THE FIRST
    //: READ CHUNK'S FILL and the next window's staging under the keys' barrier, the exponents and the first store
    const int top = post_keys<P>(st, smem, w, wid(), lid());
    clock.lap(kPhaseMaxima);
    const Entry e0 = entry_at(s, 0);
    const Run run0 = runs_of(kRead, w, e0);
    const ops::SmemAddr buf0 = place_at(w, 0);
    if (s.nr > 0) fill_read(w, e0, run0, buf0);
    if (more) {
      stage_window<P>(p, here(), w + 1, smem);
      ops::stage_commit();
    }
    clock.lap(kPhaseFill);
    ops::rendezvous();
    clock.lap(kPhaseWait);
    post_exps<P>(smem, w, top, lid());
    leaf_scales<P>(smem);
    bounds_clear<P>(p, smem);
    clock.lap(kPhaseMaxima);

    //: THE READ CHUNKS in order: the chunk's fill (the first's issued above), its image -- whole, stored once before
    //: the first chunk; or in halves, each chunk storing both -- its copies landed, its units before its first add,
    //: its tiles, then every warp done and its panel drained (the next chunk may take its place)
#pragma unroll 1
    for (int i = 0; i < s.nr; ++i) {
      const Entry e = i == 0 ? e0 : entry_at(s, i);
      const Run run = i == 0 ? run0 : runs_of(kRead, w, e);
      const ops::SmemAddr buf = i == 0 ? buf0 : place_at(w, i);
      if (i > 0) {
        fill_read(w, e, run, buf);
        clock.lap(kPhaseFill);
      }
      // R9-CASES: ImageMode, the plan's image modes (schedule.cuh)
      if (s.mode == kModeWhole) {
        if (i == 0) store_batches<P, kBoxDigits - 1>(st, smem, area, 0, kBatches, wid(), lid());
        clock.lap(kPhaseStore);
        landed(i == 0 && more);
        clock.lap(kPhaseLanded);
        ops::rendezvous();
        clock.lap(kPhaseWait);
        chunk_units(w, e, run, buf);
        clock.lap(kPhaseUnits);
        read_out(w, e, run, buf, 0, kBatches, kBoxDigits - 1);
        clock.lap(kPhaseReadout);
      } else if (s.mode == kModeHalves) {
        //: the two halves as one block each (no branch on the half): the second's store after a rendezvous, the
        //: first's landing and units before its tiles
        static_for<2>([&](auto H) {
          constexpr int half = decltype(H)::value;
          if constexpr (half > 0) {
            ops::rendezvous();
            clock.lap(kPhaseWait);
          }
          store_batches<P, 2 * kBatchPlanes - 1>(st, smem, area, 2 * half, 2 * half + 2, wid(),
                                                 lid());
          clock.lap(kPhaseStore);
          if constexpr (half == 0) {
            landed(i == 0 && more);
            clock.lap(kPhaseLanded);
          }
          ops::rendezvous();
          clock.lap(kPhaseWait);
          if constexpr (half == 0) {
            chunk_units(w, e, run, buf);
            clock.lap(kPhaseUnits);
          }
          read_out(w, e, run, buf, 2 * half, 2 * half + 2, 2 * kBatchPlanes - 1);
          clock.lap(kPhaseReadout);
        });
      } else {
        __trap();
      }
      ops::rendezvous();
      clock.lap(kPhaseWait);
      drain<P>(p, smem, rw(w), here(), w, e.first(), e.slots(),
               buf + static_cast<uint32_t>(read_pads<P>(e.slots()) + kPadBytes));
      clock.lap(kPhaseDrain);
    }

    //: THE FOLD CHUNKS in order: every warp done draining (a fill may take the drained place) or folding the chunk
    //: before, the chunk's fill, its copies landed, the window's units at its first chunk, the fold; a window folding
    //: nothing steps its units at its end
    const int nf = s.n - s.nr;
    if (nf == 0) fold_units<P>(smem, ww(w), w, wid(), lid());
#pragma unroll 1
    for (int j = 0; j < nf; ++j) {
      const Entry e = entry_at(s, s.nr + j);
      const Run run = runs_of(kWrite, w, e);
      const ops::SmemAddr buf = place_at(w, s.nr + j);
      ops::rendezvous();
      clock.lap(kPhaseWait);
      fill_fold(w, e, run, buf);
      clock.lap(kPhaseFill);
      ops::stage_wait<0>();
      clock.lap(kPhaseLanded);
      ops::rendezvous();
      clock.lap(kPhaseWait);
      if (j == 0) fold_units<P>(smem, ww(w), w, wid(), lid());
      fold_chunk<P>(st, j == 0, smem, ww(w), w, run, e.first(), e.slots(), buf,
                    buf + static_cast<uint32_t>(fold_ents<P>(e.slots())), wid(), lid());
      clock.lap(kPhaseFold);
    }
  }

  //: THE CHECK, a warp's sum added once a warp -- see docs/internals/carry/carry_kernel.md#check
  float sum = 0.f;
  static_for<P::kStateRegs>([&](auto k) { sum += st.template at<k>(); });
  static_for<5>([&](auto o) { sum += __shfl_xor_sync(0xffffffffu, sum, 16 >> o); });
  if (lane == 0) atomicAdd(p.check, sum);
  fold_probe<P>(p, st, smem, window_of<P>(p, smem, kWrite, 0), place_of(p), warp, lane);
  clock.publish(p.ledger, (long)blockIdx.y * gridDim.x + blockIdx.x, kWarps, warp);
}

}  // namespace rola::carry
