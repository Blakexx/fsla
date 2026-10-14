// csrc/rola/src/carry/carry_kernel.cuh -- THE CARRY KERNEL: one CTA a box, window by window the window's plan's items
// in order (`schedule.cuh`): the maxima, the store, each read chunk's readout and drain, each fold chunk -- the fp32
// state in registers throughout, each fill issued at its host once its region's last user is done, each item waiting
// for its buffer's copies and the item it depends on. see docs/internals/carry/carry_kernel.md
#pragma once

#include "carry/fold.cuh"
#include "carry/schedule.cuh"
#include "common/phase_clock.cuh"

namespace rola::carry {

//: THE WINDOW'S INPUTS, a window ahead, by every thread: both sides' lists, the write exponents and token units, and
//: the window's plan for the CTA's box class. -- see docs/internals/carry/carry_kernel.md#staging
template <class P>
__device__ __forceinline__ void stage_window(const CarryParams& p, const Place& c, int w,
                                             uint8_t* smem) {
  const Window rw = window_of<P>(p, smem, kRead, w), ww = window_of<P>(p, smem, kWrite, w);
  lists_in<P>(p, p.read_lists, rw, c, w, smem, kRead);
  lists_in<P>(p, p.write_lists, ww, c, w, smem, kWrite);
  facts_in<P>(p, ww, c, w, smem);
  constexpr int kPlanCopies = kPlanBytes / ops::kStageBytes,
                kPlanThread = kThreads - 1 - kPlanCopies;
  const int t = static_cast<int>(threadIdx.x) - kPlanThread;
  if (t >= 0 && t < kPlanCopies) {
    const int cls = rw.group * max(1, p.write_lists.groups) + ww.group;
    ops::stage_tile(
        ops::smem_addr(smem + P::kPlans) + (w & 1) * kPlanBytes + 16 * t,
        p.plans + ((static_cast<int64_t>(c.bh) * p.windows + w) * box_classes(p) + cls) * kPlanBytes
            + 16 * t);
  }
}

//: A THREAD'S ARRIVALS: a bit a barrier, flipped at each of its arrivals -- an item's done (barrier `kBarDone + k`) in
//: `done`, an item's landed (`kBarLanded + k`) in `land`'s low bits and the staging's and the ring's (`b <
//: kBarDone`) above them. Every thread arrives at every phase of a barrier, and a wait is for the phase of the thread's
//: own last arrival, so the bit names its parity (a barrier never arrived at passes: its phase before the first).
//: -- see docs/internals/carry/carry_kernel.md#barriers
struct Arrivals {
  static constexpr int kOwn = kPlanItems;
  static_assert(kOwn + kBarDone <= 32,
                "the items' landed bits and the staging's and the ring's share a word");
  ops::SmemAddr bars;
  uint32_t done, land;

  __device__ __forceinline__ ops::SmemAddr at(int b) const {
    return bars + 8u * static_cast<uint32_t>(b);
  }

  __device__ __forceinline__ void arrive(int b) {
    ops::mbar_arrive(at(b));
    land ^= 1u << (kOwn + b);
  }

  __device__ __forceinline__ void landed(int b) {
    ops::mbar_arrive_when_landed(at(b));
    land ^= 1u << (kOwn + b);
  }

  __device__ __forceinline__ void wait(int b) const {
    ops::mbar_wait(at(b), ((land >> (kOwn + b)) & 1u) ^ 1u);
  }

  __device__ __forceinline__ void item_done(uint32_t k) {
    ops::mbar_arrive(at(kBarDone + static_cast<int>(k)));
    done ^= 1u << k;
  }

  __device__ __forceinline__ void wait_done(uint32_t k) const {
    ops::mbar_wait(at(kBarDone + static_cast<int>(k)), ((done >> k) & 1u) ^ 1u);
  }

  __device__ __forceinline__ void item_landed(uint32_t k) {
    ops::mbar_arrive_when_landed(at(kBarLanded + static_cast<int>(k)));
    land ^= 1u << k;
  }

  __device__ __forceinline__ void wait_landed(uint32_t k) const {
    ops::mbar_wait(at(kBarLanded + static_cast<int>(k)), ((land >> k) & 1u) ^ 1u);
  }
};

//: THE KERNEL, a grid constant block in, one CTA an SM: grid `(boxes, batch-heads)`. A window: its plan's items in
//: order, each its hosted fills, its waits, its work, its done. -- see docs/internals/carry/carry_kernel.md#kernel
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
  Arrivals bar{ops::smem_addr(smem + P::kBars), 0u, 0u};
  if (threadIdx.x == 0) static_for<kBarCount>([&](auto b) { ops::mbar_init(bar.at(b), kThreads); });
  if (static_cast<int>(threadIdx.x) < kBoxLists)
    reinterpret_cast<UnitBook*>(smem + P::kBook)[threadIdx.x] = UnitBook{kUnitNone, 0.0f};
  if (threadIdx.x == 0) addr_in(*reinterpret_cast<Addr*>(smem + P::kAddr), p, place_of(p));
  ops::rendezvous();

  const ops::SmemAddr area = ops::smem_addr(smem + P::kEnd);
  const auto rw = [&](int w) { return window_of<P>(p, smem, kRead, w); };
  const auto ww = [&](int w) { return window_of<P>(p, smem, kWrite, w); };
  //: A FILL of item `k` of window `w`'s plan: its copies issued, arriving at its landed barrier as they land
  const auto fill_item = [&](int w, uint32_t k) {
    const Head h = plan_head<P>(smem, w);
    const Item it = plan_item<P>(smem, w, static_cast<int>(k));
    const ops::SmemAddr buf = area + 16u * it.buf();
    if (it.kind() == kItemRead) {
      read_fill<P>(p, smem, rw(w), here(), w, it.first(), it.slots(), h.runion(), buf, warp, lane);
    } else {
      fold_fill<P>(p, smem, ww(w), here(), w, it.first(), it.slots(), h.funion(), buf, warp, lane);
    }
    bar.item_landed(k);
  };
  //: A READ CHUNK'S TILES over batches `lo .. hi - 1` of the image at `image` (digit `d`'s plane `d & mask`)
  const auto read_out = [&](auto pairs, int w, const Head& h, const Item& it, int lo, int hi,
                            ops::SmemAddr image, uint32_t mask) {
    constexpr bool kPairs = decltype(pairs)::value;
    const ops::SmemAddr buf = area + 16u * it.buf(),
                        pads = buf + static_cast<uint32_t>(read_pads<P>(it.slots())),
                        ents = buf + static_cast<uint32_t>(read_ents<P>(it.slots()));
    if (p.readout_f16) {
      read_chunk<P, true, kPairs>(smem, rw(w), w, it.first(), it.slots(), h.runion(), lo, hi, image,
                                  mask, buf, pads, ents, warp, lane);
    } else {
      read_chunk<P, false, kPairs>(smem, rw(w), w, it.first(), it.slots(), h.runion(), lo, hi,
                                   image, mask, buf, pads, ents, warp, lane);
    }
  };

  State<P> st;
  static_for<P::kFoldLists>([&](auto J) {
    static_for<P::kFoldMTiles>([&](auto M) {
      static_for<4>([&](auto N) { static_for<4>([&](auto I) { st.acc[J][M][N][I] = 0.f; }); });
    });
  });
  int dexp = 0;

  stage_window<P>(p, here(), 0, smem);
  bar.landed(kBarStaged);
  bar.wait(kBarStaged);
  if (plan_head<P>(smem, 0).runion() > 0) fill_item(0, kFirstRead);

#pragma unroll 1
  for (int w = 0; w < p.windows; ++w) {
    if (w > 0) bar.wait(kBarStaged);
    int k = 0;
#pragma unroll 1
    do {
      const Item it = plan_item<P>(smem, w, k);
      const Head h = plan_head<P>(smem, w);
      //: THE FILLS it hosts, each once its region's last user is done; the next window's first read once its inputs
      //: have landed
#pragma unroll 1
      for (int f = 0; f < it.fills(); ++f) {
        const uint32_t target = it.fill(f), fw = it.fwait(f);
        if (target != kNoItem) {
          if (fw != kNoItem) bar.wait_done(fw);
          const bool next = target == kNextItem;
          if (next) bar.wait(kBarStaged);
          clock.lap(kPhaseWait);
          fill_item(next ? w + 1 : w, next ? static_cast<uint32_t>(kFirstRead) : target);
          clock.lap(kPhaseFill);
        }
      }
      const uint32_t kind = it.kind(), flags = it.flags();
      const uint32_t mode = h.mode();
      if ((flags & kFlagFilled) != 0u && (kind != kItemRead || mode == kModeWhole))
        bar.wait_landed(static_cast<uint32_t>(k));
      clock.lap(kPhaseLanded);
      if (it.dep() != kNoItem) bar.wait_done(it.dep());
      clock.lap(kPhaseWait);

      //: THE ITEM'S WORK, by its kind -- an exclusive chain (`carry_kernel.md#kernel`)
      // R9-CASES: ItemKind, the plan pass's item kinds (schedule.cuh)
      if (kind == kItemMaxima) {
        //: THE DIGIT MAXIMA of the state before the window's writes, every warp's posted, and each digit's exponent
        //: (lane `d` digit `d`'s), warp 0's recorded for the readers
        post_maxima<P>(st, smem, w, warp, lane);
        bar.item_done(0u);
        bar.wait_done(0u);
        dexp = digit_exp<P>(smem, w, lane);
        if (warp == 0) reinterpret_cast<int*>(smem + P::kExps)[lane] = dexp;
        clock.lap(kPhaseMaxima);
      } else if (kind == kItemStore) {
        //: THE STORE: the next window's inputs staged; the image stored whole in its mode
        if (w + 1 < p.windows) {
          stage_window<P>(p, here(), w + 1, smem);
          bar.landed(kBarStaged);
        }
        if (mode == kModeWhole)
          store_batches<P>(st, smem, area, 0, kBatches, kBoxDigits - 1, dexp, warp, lane);
        clock.lap(kPhaseStore);
      } else if (kind == kItemRead) {
        //: A READ CHUNK: the image whole; or a ring the chunk stores itself, a slot freed once every warp has read it
        //: (for the batch two on, or the next chunk's)
        const bool again = it.first() > 0, more = (flags & kFlagMoreReads) != 0u;
        // R9-CASES: ImageMode, the plan's image modes (schedule.cuh)
        if (mode == kModeWhole) {
          read_out(std::true_type{}, w, h, it, 0, kBatches, area, kBoxDigits - 1);
          clock.lap(kPhaseReadout);
        } else if (mode == kModeRing16) {
#pragma unroll 1
          for (int half = 0; half < 2; ++half) {
            const bool lead = half == 0;
            if (!lead || again) bar.wait(kBarImageFree);
            store_batches<P>(st, smem, area, 2 * half, 2 * half + 2, 2 * kBatchPlanes - 1, dexp,
                             warp, lane);
            bar.arrive(kBarStored);
            bar.wait(kBarStored);
            clock.lap(kPhaseStore);
            if (lead) bar.wait_landed(static_cast<uint32_t>(k));
            clock.lap(kPhaseLanded);
            read_out(std::true_type{}, w, h, it, 2 * half, 2 * half + 2, area,
                     2 * kBatchPlanes - 1);
            if (lead || more) bar.arrive(kBarImageFree);
            clock.lap(kPhaseReadout);
          }
        } else if (mode == kModeRing8) {
#pragma unroll 1
          for (int b = 0; b < 2; ++b) {
            if (again) bar.wait(kBarImageFree + b);
            store_batches<P>(st, smem, area + static_cast<uint32_t>(b * kImageRing / 2), b, b + 1,
                             kBatchPlanes - 1, dexp, warp, lane);
            bar.arrive(kBarStored + b);
          }
          clock.lap(kPhaseStore);
          bar.wait_landed(static_cast<uint32_t>(k));
          clock.lap(kPhaseLanded);
#pragma unroll 1
          for (int b = 0; b < kBatches; ++b) {
            const int r = b & 1;
            const ops::SmemAddr slot = area + static_cast<uint32_t>(r * kImageRing / 2);
            bar.wait(kBarStored + r);
            clock.lap(kPhaseStore);
            read_out(std::false_type{}, w, h, it, b, b + 1, slot, kBatchPlanes - 1);
            if (b + 2 < kBatches || more) bar.arrive(kBarImageFree + r);
            clock.lap(kPhaseReadout);
            if (b + 2 < kBatches) {
              bar.wait(kBarImageFree + r);
              store_batches<P>(st, smem, slot, b + 2, b + 3, kBatchPlanes - 1, dexp, warp, lane);
              bar.arrive(kBarStored + r);
              clock.lap(kPhaseStore);
            }
          }
        } else {
          __trap();
        }
      } else if (kind == kItemDrain) {
        //: A DRAIN: the read chunk's panel to `num`, once every warp has read it out
        const ops::SmemAddr buf = area + 16u * it.buf();
        drain<P>(p, smem, rw(w), here(), w, it.first(), it.slots(),
                 buf + static_cast<uint32_t>(read_pads<P>(it.slots()) + kPadBytes));
        clock.lap(kPhaseDrain);
      } else if (kind == kItemFold) {
        //: A FOLD CHUNK, the window's units stepped at its first
        if ((flags & kFlagUnits) != 0u) fold_units<P>(smem, ww(w), w, warp, lane);
        const ops::SmemAddr buf = area + 16u * it.buf();
        fold_chunk<P>(st, (flags & kFlagUnits) != 0u, smem, ww(w), w, it.first(), it.slots(),
                      h.funion(), buf, buf + static_cast<uint32_t>(fold_ents<P>(it.slots())), warp,
                      lane);
        clock.lap(kPhaseFold);
      } else if (kind == kItemBoundary) {
        //: THE BOUNDARY: a window with no fold steps its units all the same
        if ((flags & kFlagUnits) != 0u) fold_units<P>(smem, ww(w), w, warp, lane);
        clock.lap(kPhaseFold);
      } else {
        __trap();
      }
      if ((flags & kFlagDone) != 0u) bar.item_done(static_cast<uint32_t>(k));
    } while (__any_sync(0xffffffffu, ++k < plan_items<P>(smem, w)));
  }

  float sum = 0.f;
  static_for<P::kStateRegs>([&](auto k) { sum += st.template at<k>(); });
  atomicAdd(p.check, sum);
  fold_probe<P>(p, st, smem, window_of<P>(p, smem, kWrite, 0), place_of(p), warp, lane);
  clock.publish(p.ledger, (long)blockIdx.y * gridDim.x + blockIdx.x, kWarps, warp);
}

}  // namespace rola::carry
