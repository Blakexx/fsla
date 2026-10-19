// csrc/rola/src/carry/carry_kernel.cuh -- THE CARRY KERNEL: one CTA a box, window by window: the maxima, the store,
// then the window's entries (`schedule.cuh`) in order -- each read chunk's readout, drained as the next begins, each
// fold chunk -- the fp32 state in registers throughout, entry t + 1 filled into the other slot while t computes where
// the two fit. see docs/internals/carry/carry_kernel.md
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

//: A THREAD'S FLOW WORD: a bit a barrier, flipped at each of its arrivals -- a slot's landed at bit `slot`, the
//: staging's at bit 2 -- then the stream's cursor, its entry's slot at bit 3 and the window's step from bit 4 (the slot
//: flips at each entry's step, so it alternates across windows too). Every thread arrives at every phase of a barrier, and a wait
//: is for the phase of the thread's own last arrival, so a barrier's bit names its parity (one never arrived at passes:
//: its phase before the first) -- one phase in flight a barrier. Every other rendezvous is the hardware barrier
//: (`ops::rendezvous`). -- see docs/internals/carry/carry_kernel.md#barriers
struct Flow {
  static constexpr int kStagedBit = 2, kSlotBit = 3, kStepShift = 4;
  ops::SmemAddr bars;
  uint32_t v;

  __device__ __forceinline__ ops::SmemAddr at(int b) const {
    return bars + 8u * static_cast<uint32_t>(b);
  }

  __device__ __forceinline__ void staged() {
    ops::mbar_arrive_when_landed(at(kBarStaged));
    v ^= 1u << kStagedBit;
  }

  __device__ __forceinline__ void wait_staged() const {
    ops::mbar_wait(at(kBarStaged), ((v >> kStagedBit) & 1u) ^ 1u);
  }

  __device__ __forceinline__ void slot_landed(int slot) {
    ops::mbar_arrive_when_landed(at(kBarSlot + slot));
    v ^= 1u << slot;
  }

  __device__ __forceinline__ void wait_slot(int slot) const {
    ops::mbar_wait(at(kBarSlot + slot), ((v >> slot) & 1u) ^ 1u);
  }

  __device__ __forceinline__ int step() const { return static_cast<int>(v >> kStepShift); }

  __device__ __forceinline__ int slot() const { return static_cast<int>((v >> kSlotBit) & 1u); }

  __device__ __forceinline__ void step_on(bool entry) {
    v = (v + (1u << kStepShift)) ^ (entry ? 1u << kSlotBit : 0u);
  }

  __device__ __forceinline__ void open() { v &= (1u << kStepShift) - 1u; }
};

//: THE KERNEL, a grid constant block in, one CTA an SM: grid `(boxes, batch-heads)`. A window: the maxima, the store,
//: its entries step by step, its end. -- see docs/internals/carry/carry_kernel.md#kernel
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
  Flow flow{ops::smem_addr(smem + P::kBars), 0u};
  if (threadIdx.x == 0)
    static_for<kBarCount>([&](auto b) { ops::mbar_init(flow.at(b), kThreads); });
  if (static_cast<int>(threadIdx.x) < kBoxLists)
    reinterpret_cast<UnitBook*>(smem + P::kBook)[threadIdx.x] = UnitBook{kUnitNone, 0.0f};
  if (static_cast<int>(threadIdx.x) < 2 * kBoxDigits)
    reinterpret_cast<int*>(smem + P::kKeys)[threadIdx.x] = INT_MIN;
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
  //: AN ENTRY'S FILL of window `w`: its copies issued into its place, arriving at its slot's landed barrier as they
  //: land
  const auto fill_entry = [&](int w, const Entry& e, int slot, int at) {
    const ops::SmemAddr buf = area + static_cast<uint32_t>(at);
    if (e.side() == kRead) {
      read_fill<P>(p, smem, rw(w), here(), w, e.first(), e.slots(), e.total(), buf, wid(), lid());
    } else {
      fold_fill<P>(p, smem, ww(w), here(), w, e.first(), e.slots(), e.total(), buf, wid(), lid());
    }
    flow.slot_landed(slot);
  };
  //: A READ CHUNK'S TILES over batches `lo .. hi - 1` of the image at `image` (digit `d`'s plane `d & mask`)
  const auto read_out = [&](auto pairs, int w, const Entry& e, ops::SmemAddr buf, int lo, int hi,
                            ops::SmemAddr image, uint32_t mask) {
    constexpr bool kPairs = decltype(pairs)::value;
    const ops::SmemAddr pads = buf + static_cast<uint32_t>(read_pads<P>(e.slots())),
                        ents = buf + static_cast<uint32_t>(read_ents<P>(e.slots()));
    if (p.readout_f16) {
      read_chunk<P, true, kPairs>(p, smem, rw(w), w, e.first(), e.slots(), e.total(), lo, hi, image,
                                  mask, buf, pads, ents, wid(), lid());
    } else {
      read_chunk<P, false, kPairs>(p, smem, rw(w), w, e.first(), e.slots(), e.total(), lo, hi,
                                   image, mask, buf, pads, ents, wid(), lid());
    }
  };

  State<P> st;
  static_for<P::kFoldLists>([&](auto J) {
    static_for<P::kFoldMTiles>([&](auto M) {
      static_for<4>([&](auto N) { static_for<4>([&](auto I) { st.acc[J][M][N][I] = 0.f; }); });
    });
  });
  stage_window<P>(p, here(), 0, smem);
  flow.staged();
  flow.wait_staged();
  if (stream_of<P>(p, smem, 0).n > 0)
    fill_entry(0, entry_at(stream_of<P>(p, smem, 0), 0), kSlotTop, step_of<P>(p, smem, 0, 0).at());

#pragma unroll 1
  for (int w = 0; w < p.windows; ++w) {
    if (w > 0) flow.wait_staged();
    clock.lap(kPhaseWait);
    //: THE DIGIT MAXIMA of the state before the window's writes, every warp's keys posted, and each digit's exponent
    //: recorded by every warp (the same values) for its stores and the readers
    {
      const int top = post_keys<P>(st, smem, w, wid(), lid());
      ops::rendezvous();
      post_exps<P>(smem, w, top, lid());
    }
    clock.lap(kPhaseMaxima);
    //: THE STORE: the next window's inputs staged; a whole image stored
    if (w + 1 < p.windows) {
      stage_window<P>(p, here(), w + 1, smem);
      flow.staged();
    }
    if (stream_of<P>(p, smem, w).image == kImageWhole)
      store_batches<P, kBoxDigits - 1>(st, smem, area, 0, kBatches, wid(), lid());
    clock.lap(kPhaseStore);

    //: THE WINDOW'S STEPS: one an entry, then one that only retires the last; a window of no entries fills the next
    //: window's first in its one step
    flow.open();
#pragma unroll 1
    do {
      //: THE STEP, by its word: entry `t`'s place and whether filled ahead, the next filled now, the retired read's
      //: drain's rendezvous
      const int t = flow.step();
      const Stream s = stream_of<P>(p, smem, w);
      const bool work = t < s.n, cross = t + 1 >= s.n && (work || t == 0);
      const bool drained = t > 0 && t - 1 < s.nr;
      const StepWord sw = step_of<P>(p, smem, w, t);
      if (sw.next() && cross) flow.wait_staged();
      if (drained || (t > 0 && ((work && !sw.ahead()) || sw.next()))) ops::rendezvous();
      clock.lap(kPhaseWait);

      //: ENTRY `t - 1` RETIRED: every warp done with it; a read chunk's panel drained, and every warp done draining
      //: where a fill reuses its place
      if (drained) {
        const Entry d = entry_at(stream_of<P>(p, smem, w), t - 1);
        drain<P>(p, smem, rw(w), here(), w, d.first(), d.slots(),
                 area
                     + static_cast<uint32_t>(step_of<P>(p, smem, w, t - 1).at()
                                             + read_pads<P>(d.slots()) + kPadBytes));
        clock.lap(kPhaseDrain);
        if (sw.reuse()) ops::rendezvous();
        clock.lap(kPhaseWait);
      }
      //: THE FILLS: entry `t` unless filled ahead, then the next where planned
#pragma unroll 1
      for (int f = 0; f < 2; ++f) {
        const StepWord sf = step_of<P>(p, smem, w, t);
        if (f == 0 ? work && !sf.ahead() : sf.next()) {
          const bool other = f == 1 && cross;
          const int fw = other ? w + 1 : w, ft = f == 0 ? t : other ? 0 : t + 1;
          fill_entry(fw, entry_at(stream_of<P>(p, smem, fw), ft),
                     flow.slot() ^ (f == 1 && work ? 1 : 0), step_of<P>(p, smem, fw, ft).at());
        }
      }
      clock.lap(kPhaseFill);

      //: THE UNITS, stepped at the window's first fold chunk, or at its end where it folds nothing
      if (t == stream_of<P>(p, smem, w).nr) fold_units<P>(smem, ww(w), w, wid(), lid());

      //: THE ENTRY'S WORK: a read chunk from the whole image, or from the halves it stores itself; a fold chunk
      if (work) {
        const ops::SmemAddr buf =
            area + static_cast<uint32_t>(step_of<P>(p, smem, w, flow.step()).at());
        const Stream sw2 = stream_of<P>(p, smem, w);
        if (t < sw2.nr) {
          // R9-CASES: ImageMode, the plan's image modes (schedule.cuh)
          if (sw2.mode == kModeWhole) {
            flow.wait_slot(flow.slot());
            clock.lap(kPhaseLanded);
            if (t == 0) ops::rendezvous();
            clock.lap(kPhaseWait);
            read_out(std::true_type{}, w, entry_at(stream_of<P>(p, smem, w), flow.step()), buf, 0,
                     kBatches, area, kBoxDigits - 1);
            clock.lap(kPhaseReadout);
          } else if (sw2.mode == kModeHalves) {
#pragma unroll 1
            for (int half = 0; half < 2; ++half) {
              if (half > 0) ops::rendezvous();
              store_batches<P, 2 * kBatchPlanes - 1>(st, smem, area, 2 * half, 2 * half + 2, wid(),
                                                     lid());
              ops::rendezvous();
              clock.lap(kPhaseStore);
              if (half == 0) flow.wait_slot(flow.slot());
              clock.lap(kPhaseLanded);
              read_out(std::true_type{}, w, entry_at(stream_of<P>(p, smem, w), flow.step()), buf,
                       2 * half, 2 * half + 2, area, 2 * kBatchPlanes - 1);
              clock.lap(kPhaseReadout);
            }
          } else {
            __trap();
          }
        } else {
          flow.wait_slot(flow.slot());
          clock.lap(kPhaseLanded);
          const Entry e = entry_at(stream_of<P>(p, smem, w), flow.step());
          fold_chunk<P>(p, st, flow.step() == stream_of<P>(p, smem, w).nr, smem, ww(w), w,
                        e.first(), e.slots(), e.total(), buf,
                        buf + static_cast<uint32_t>(fold_ents<P>(e.slots())), wid(), lid());
          clock.lap(kPhaseFold);
        }
      }
      flow.step_on(flow.step() < stream_of<P>(p, smem, w).n);
    } while (__any_sync(0xffffffffu, flow.step() <= stream_of<P>(p, smem, w).n));
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
