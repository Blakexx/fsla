// csrc/rola/src/carry/carry_kernel.cuh -- THE CARRY KERNEL: one CTA a box, window by window under the window's plan
// (`schedule.cuh`): the state before the window's writes handed to the image and read out chunk by chunk, then the
// window's writes folded in chunk by chunk -- the fp32 state in registers throughout, every buffer's copies landing on
// a barrier, every reuse waiting for its last readers. see docs/internals/carry/carry_kernel.md
#pragma once

#include "carry/fold.cuh"
#include "carry/schedule.cuh"
#include "common/phase_clock.cuh"

namespace rola::carry {

//: THE WINDOW'S INPUTS, a window ahead, by every thread: both sides' lists, the write exponents and token units.
//: -- see docs/internals/carry/carry_kernel.md#staging
template <class P>
__device__ __forceinline__ void stage_window(const CarryParams& p, const Place& c, int w,
                                             uint8_t* smem) {
  const Window rw = window_of<P>(p, smem, kRead, w), ww = window_of<P>(p, smem, kWrite, w);
  lists_in<P>(p, p.read_lists, rw, c, w, smem, kRead);
  lists_in<P>(p, p.write_lists, ww, c, w, smem, kWrite);
  facts_in<P>(p, ww, c, w, smem);
}

//: A THREAD'S ARRIVALS: a bit a barrier, flipped at each of its arrivals. Every thread arrives at every phase of every
//: barrier, and a wait is for the phase of the thread's own last arrival, so the bit names its parity (a barrier never
//: arrived at passes: its phase before the first). -- see docs/internals/carry/carry_kernel.md#barriers
struct Arrivals {
  ops::SmemAddr bars;
  uint32_t bits;

  __device__ __forceinline__ ops::SmemAddr at(int b) const {
    return bars + 8u * static_cast<uint32_t>(b);
  }

  __device__ __forceinline__ void arrive(int b) {
    ops::mbar_arrive(at(b));
    bits ^= 1u << b;
  }

  __device__ __forceinline__ void landed(int b) {
    ops::mbar_arrive_when_landed(at(b));
    bits ^= 1u << b;
  }

  __device__ __forceinline__ void wait(int b) const {
    ops::mbar_wait(at(b), ((bits >> b) & 1u) ^ 1u);
  }

  //: the release's phase, which a window may leave unwaited, waited once every warp has posted the next window's
  //: maxima (it is complete), so a thread observes each phase before it arrives at the next; every other barrier is
  //: arrived at only where a later wait in its window takes it
  __device__ __forceinline__ void settle() const { wait(kBarReleased); }
};

//: THE KERNEL, a grid constant block in, one CTA an SM: grid `(boxes, batch-heads)`. A window: the maxima; the readout
//: in its plan's form; the next window's plan; the fold chunk by chunk. -- see docs/internals/carry/carry_kernel.md#kernel
template <class Arm>
__global__ __launch_bounds__(kThreads, 1) void carry_kernel(const __grid_constant__ CarryParams p) {
  using P = Plan<Arm>;
  extern __shared__ __align__(128) uint8_t smem[];
  //: the launch sized for this card's plan, or nothing runs (`carry_arm.md#plan`)
  uint32_t launched;
  asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(launched));
  if (launched != static_cast<uint32_t>(P::kSmemBytes)) __trap();
  const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
  const Place c = place_of(p);
  PhaseClock<kCarryPhases, true> clock(
      reinterpret_cast<long long*>(smem + P::kClock) + warp * PhaseClock<kCarryPhases, true>::kRow,
      lane, p.ledger != nullptr);
  Arrivals bar{ops::smem_addr(smem + P::kBars), 0u};
  if (threadIdx.x == 0) static_for<kBarCount>([&](auto b) { ops::mbar_init(bar.at(b), kThreads); });
  if (static_cast<int>(threadIdx.x) < kBoxLists)
    reinterpret_cast<UnitBook*>(smem + P::kBook)[threadIdx.x] = UnitBook{kUnitNone, 0.0f};
  if (threadIdx.x == 0) addr_in(*reinterpret_cast<Addr*>(smem + P::kAddr), p, c);
  ops::rendezvous();

  const ops::SmemAddr area = ops::smem_addr(smem + P::kEnd);
  const auto rw = [&](int w) { return window_of<P>(p, smem, kRead, w); };
  const auto ww = [&](int w) { return window_of<P>(p, smem, kWrite, w); };
  const auto at16 = [&](uint32_t k) { return area + 16u * k; };
  //: READ CHUNK `q` of window `w` into read buffer `q & 1` (of one, 0); FOLD CHUNK `k` into fold buffer `k & 1`
  const auto read_in = [&](int w, int q) {
    volatile WindowPlan& pl = plan_of<P>(smem, warp, w);
    const int s = q & (pl.rbufs - 1), first = q * pl.rchunk,
              slots = min(static_cast<int>(pl.rchunk), pl.runion - first);
    const ops::SmemAddr buf = at16(pl.rbuf[s]);
    read_fill<P>(p, smem, rw(w), c, w, first, slots, pl.runion, pl.rchunk, buf,
                 buf + 16u * pl.rents, warp, lane);
    bar.landed(kBarReadLanded + s);
  };
  const auto fold_in = [&](int w, int k) {
    volatile WindowPlan& pl = plan_of<P>(smem, warp, w);
    const int s = k & 1, first = k * pl.fchunk,
              slots = min(static_cast<int>(pl.fchunk), pl.funion - first);
    const ops::SmemAddr buf = at16(pl.fbuf[s]);
    fold_fill<P>(p, smem, ww(w), c, w, first, slots, pl.funion, buf, buf + 16u * pl.fents, warp,
                 lane);
    bar.landed(kBarFoldLanded + s);
  };
  //: A READ CHUNK'S TILES over batches `lo .. hi - 1` of the image at `image` (planes `d & mask`)
  const auto read_out = [&](int w, int q, int lo, int hi, ops::SmemAddr image, uint32_t mask) {
    volatile WindowPlan& pl = plan_of<P>(smem, warp, w);
    const int s = q & (pl.rbufs - 1), first = q * pl.rchunk,
              slots = min(static_cast<int>(pl.rchunk), pl.runion - first);
    const ops::SmemAddr buf = at16(pl.rbuf[s]),
                        pads = buf + static_cast<uint32_t>(align_of(pl.rchunk * P::kReadRow, 128));
    if (p.readout_f16) {
      read_chunk<P, true>(smem, rw(w), w, first, slots, pl.runion, lo, hi, image, mask, buf, pads,
                          buf + 16u * pl.rents, warp, lane);
    } else {
      read_chunk<P, false>(smem, rw(w), w, first, slots, pl.runion, lo, hi, image, mask, buf, pads,
                           buf + 16u * pl.rents, warp, lane);
    }
  };
  //: A READ CHUNK'S DRAIN, once every warp has read it; its buffer freed where a later fill waits for it
  const auto drain_out = [&](int w, int q, bool refill) {
    volatile WindowPlan& pl = plan_of<P>(smem, warp, w);
    const int s = q & (pl.rbufs - 1), first = q * pl.rchunk,
              slots = min(static_cast<int>(pl.rchunk), pl.runion - first);
    bar.wait(kBarReadDone + s);
    drain<P>(p, smem, rw(w), c, w, first, slots,
             at16(pl.rbuf[s])
                 + static_cast<uint32_t>(align_of(pl.rchunk * P::kReadRow, 128) + kPadBytes));
    if (refill) bar.arrive(kBarReadFree + s);
  };

  State<P> st;
  static_for<P::kFoldLists>([&](auto J) {
    static_for<P::kFoldMTiles>([&](auto M) {
      static_for<4>([&](auto N) { static_for<4>([&](auto I) { st.acc[J][M][N][I] = 0.f; }); });
    });
  });

  stage_window<P>(p, c, 0, smem);
  bar.landed(kBarStaged);
  bar.wait(kBarStaged);
  plan_window<P>(p, smem, 0, warp, lane);
  if (plan_of<P>(smem, warp, 0).runion > 0) read_in(0, 0);

#pragma unroll 1
  for (int w = 0; w < p.windows; ++w) {
    const bool next = w + 1 < p.windows;
    volatile WindowPlan& pl = plan_of<P>(smem, warp, w);
    const int rchunks = pl.rchunks, fchunks = pl.fchunks;

    //: THE DIGIT MAXIMA of the state before the window's writes, every warp's posted: so every warp is past the window
    //: before, and its buffers and the other parity's tenants are free
    post_maxima<P>(st, smem, warp, lane);
    bar.arrive(kBarPosted);
    bar.wait(kBarPosted);
    bar.settle();
    clock.lap(kPhaseMaxima);
    if (next) {
      stage_window<P>(p, c, w + 1, smem);
      bar.landed(kBarStaged);
    }
    if (fchunks > 0 && pl.beside) fold_in(w, 0);

    const int form = pl.form;
    if (form == kFormUnion && rchunks > 0) {
      //: THE UNION RESIDENT: the image a ring of batches, each stored, read over the union, its slot freed
      store_batches<P>(st, smem, area, 0, 1, kBatchPlanes - 1, warp, lane);
      bar.arrive(kBarStored);
      store_batches<P>(st, smem, area + kImageRing / 2, 1, 2, kBatchPlanes - 1, warp, lane);
      bar.arrive(kBarStored + 1);
      clock.lap(kPhaseStore);
      bar.wait(kBarReadLanded);
      clock.lap(kPhaseLanded);
      int b = 0;
#pragma unroll 1
      do {
        const int s = b & 1;
        const ops::SmemAddr slot = area + static_cast<uint32_t>(s * kImageRing / 2);
        bar.wait(kBarStored + s);
        clock.lap(kPhaseStore);
        read_out(w, 0, b, b + 1, slot, kBatchPlanes - 1);
        clock.lap(kPhaseReadout);
        if (b + 2 < kBatches) {
          bar.arrive(kBarImageFree + s);
          bar.wait(kBarImageFree + s);
          store_batches<P>(st, smem, slot, b + 2, b + 3, kBatchPlanes - 1, warp, lane);
          bar.arrive(kBarStored + s);
          clock.lap(kPhaseStore);
        }
      } while (__any_sync(0xffffffffu, ++b < kBatches));
      bar.arrive(kBarReadDone);
      drain_out(w, 0, false);
    } else if (form != kFormUnion) {
      //: THE IMAGE WHOLE, stored once; or (tiled) a ring of batches stored again for each chunk
      if (form == kFormImage) {
        store_batches<P>(st, smem, area, 0, kBatches, kBoxDigits - 1, warp, lane);
        bar.arrive(kBarStored);
        bar.wait(kBarStored);
        clock.lap(kPhaseStore);
      }
      //: two read buffers: the chunk before drained once every warp has read it, then the next chunk into its buffer;
      //: one: the chunk drained, then the next into it
      const bool two = pl.rbufs == 2;
      int q = 0;
      for (int go = __any_sync(0xffffffffu, rchunks > 0) ? 1 : 0; go > 0; --go)
#pragma unroll 1
        do {
          const int s = two ? q & 1 : 0;
          if (two && q >= 1) drain_out(w, q - 1, q + 1 < rchunks);
          clock.lap(kPhaseDrain);
          if (two && q + 1 < rchunks) {
            bar.wait(kBarReadFree + (s ^ 1));
            read_in(w, q + 1);
          }
          bar.wait(kBarReadLanded + s);
          clock.lap(kPhaseLanded);
          if (form == kFormImage) {
            read_out(w, q, 0, kBatches, area, kBoxDigits - 1);
            clock.lap(kPhaseReadout);
          } else {
            int b = 0;
            do {
              const int r = b & 1;
              const ops::SmemAddr slot = area + static_cast<uint32_t>(r * kImageRing / 2);
              if (b >= 2 || q >= 1) bar.wait(kBarImageFree + r);
              store_batches<P>(st, smem, slot, b, b + 1, kBatchPlanes - 1, warp, lane);
              bar.arrive(kBarStored + r);
              bar.wait(kBarStored + r);
              clock.lap(kPhaseStore);
              read_out(w, q, b, b + 1, slot, kBatchPlanes - 1);
              if (b + 2 < kBatches || q + 1 < rchunks) bar.arrive(kBarImageFree + r);
              clock.lap(kPhaseReadout);
            } while (__any_sync(0xffffffffu, ++b < kBatches));
          }
          bar.arrive(kBarReadDone + s);
          if (!two) {
            drain_out(w, q, q + 1 < rchunks);
            clock.lap(kPhaseDrain);
            if (q + 1 < rchunks) {
              bar.wait(kBarReadFree);
              read_in(w, q + 1);
            }
          }
        } while (__any_sync(0xffffffffu, ++q < rchunks));
      if (two && rchunks > 0) drain_out(w, rchunks - 1, false);
    }
    bar.arrive(kBarReleased);
    clock.lap(kPhaseDrain);

    //: THE NEXT WINDOW'S PLAN, and its first read chunk now where it leaves the fold's buffers alone
    bool early = false, late = false;
    if (next) {
      bar.wait(kBarStaged);
      plan_window<P>(p, smem, w + 1, warp, lane);
      volatile WindowPlan& nx = plan_of<P>(smem, warp, w + 1);
      early = nx.runion > 0 && (fchunks == 0 || nx.rbuf[1] <= pl.fzone);
      if (early) {
        bar.wait(kBarReleased);
        read_in(w + 1, 0);
      }
      late = !early && nx.runion > 0;
    }
    clock.lap(kPhasePlan);

    //: THE FOLD, chunk by chunk: a buffer's fill waits for its last readers, and over the readout's zone for its release
    uint32_t cf[P::kFoldLists];
    float fs[P::kFoldLists];
    fold_units<P>(cf, fs, smem, ww(w), w, warp, lane);
    int k = 0;
    for (int go = __any_sync(0xffffffffu, fchunks > 0) ? 1 : 0; go > 0; --go)
#pragma unroll 1
      do {
        const int s = k & 1;
        if (k == 0 && !pl.beside) {
          bar.wait(kBarReleased);
          fold_in(w, 0);
        }
        if (k + 1 < fchunks) {
          bar.wait(kBarFoldFree + (s ^ 1));
          if (pl.fbuf[s ^ 1] < pl.rzone) bar.wait(kBarReleased);
          fold_in(w, k + 1);
        }
        bar.wait(kBarFoldLanded + s);
        clock.lap(kPhaseFill);
        const int first = k * pl.fchunk,
                  slots = min(static_cast<int>(pl.fchunk), pl.funion - first);
        const ops::SmemAddr buf = at16(pl.fbuf[s]);
        fold_chunk<P>(st, cf, fs, smem, ww(w), w, first, slots, pl.funion, buf,
                      buf + 16u * pl.fents, warp, lane);
        //: freed where a later fill waits for it: the chunk two on, or the next window's first read chunk
        if (k + 2 < fchunks || late) bar.arrive(kBarFoldFree + s);
        clock.lap(kPhaseFold);
      } while (__any_sync(0xffffffffu, ++k < fchunks));
    if (late) {
      if (fchunks > 0) bar.wait(kBarFoldFree + ((fchunks - 1) & 1));
      if (fchunks > 1) bar.wait(kBarFoldFree + ((fchunks - 2) & 1));
      bar.wait(kBarReleased);
      read_in(w + 1, 0);
      clock.lap(kPhaseFill);
    }
  }

  float sum = 0.f;
  static_for<P::kStateRegs>([&](auto k) { sum += st.template at<k>(); });
  atomicAdd(p.check, sum);
  fold_probe<P>(p, st, smem, window_of<P>(p, smem, kWrite, 0), place_of(p), warp, lane);
  clock.publish(p.ledger, (long)blockIdx.y * gridDim.x + blockIdx.x, kWarps, warp);
}

}  // namespace rola::carry
