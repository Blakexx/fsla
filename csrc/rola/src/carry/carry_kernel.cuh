// csrc/rola/src/carry/carry_kernel.cuh -- THE CARRY KERNEL: one CTA a box, readout-first windows -- the state before
// the window's writes handed to the image part by part and read out pass by pass, then the window's writes folded in
// -- the fp32 state in registers throughout, every async copy landing on an mbarrier. see
// docs/internals/carry/carry_kernel.md
#pragma once

#include "carry/fold.cuh"
#include "common/phase_clock.cuh"

namespace rola::carry {

//: THE WINDOW'S INPUTS, a window ahead, by every thread: both sides' lists, the write exponents and token units, the
//: warp's read heads; landing on the staging barrier. -- see docs/internals/carry/carry_kernel.md#staging
template <class P>
__device__ __forceinline__ void stage_window(const CarryParams& p, const Place& c, int w,
                                             uint8_t* smem, ops::SmemAddr bar, int warp, int lane) {
  const Window rw = window_of<P>(p, smem, kRead, w), ww = window_of<P>(p, smem, kWrite, w);
  lists_in<P>(p, p.read_lists, rw, c, w, smem, kRead);
  lists_in<P>(p, p.write_lists, ww, c, w, smem, kWrite);
  facts_in<P>(p, ww, c, w, smem);
  heads_in(p.read_lists, rw, c, p.windows, w,
           ops::smem_addr(smem + P::kReadHeads) + warp * P::kHeadBytes, warp, lane);
  ops::mbar_arrive_when_landed(bar);
}

//: THE KERNEL, a grid constant block in, one CTA an SM: grid `(boxes, batch-heads)`. The readout ring's two slots take
//: the window's passes then the fold's first unit; the fold's ring over the image and the panel takes the rest.
//: -- see docs/internals/carry/carry_kernel.md#kernel
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
  //: the barriers: the readout ring's two, the fold ring's, then the window's staging and the fold heads'
  const ops::SmemAddr bars = ops::smem_addr(smem + P::kBars);
  const ops::SmemAddr staged = bars + 8 * (2 + P::kFoldRing), heads_bar = staged + 8;
  if (threadIdx.x == 0)
    static_for<4 + P::kFoldRing>([&](auto b) { ops::mbar_init(bars + 8 * b, kThreads); });
  if (static_cast<int>(threadIdx.x) < kBoxLists)
    reinterpret_cast<UnitBook*>(smem + P::kBook)[threadIdx.x] = UnitBook{kUnitNone, 0.0f};
  if (threadIdx.x == 0) addr_in(*reinterpret_cast<Addr*>(smem + P::kAddr), p, c);
  //: the panel and its dummy rows start at zero
#pragma unroll 1
  for (int i = static_cast<int>(threadIdx.x); i < (kPadBytes + P::kPanelBytes) / 16; i += kThreads)
    reinterpret_cast<uint4*>(smem + P::kPads)[i] = make_uint4(0u, 0u, 0u, 0u);
  ops::rendezvous();

  //: the readout ring's slot `k`, an address computed rather than an array indexed at run time
  const auto ring = [&](int k) {
    return ops::smem_addr(smem + P::kRing0) + static_cast<uint32_t>(k * (P::kRing1 - P::kRing0));
  };
  const ops::SmemAddr fold_heads = ops::smem_addr(smem + P::kFoldHeads) + warp * P::kHeadBytes;
  //: a fold ring slot's pool: the image's units, then the panel's
  const auto fold_slot = [&](int k) {
    return k < P::kImageUnits
               ? ops::smem_addr(smem + P::kImage) + k * P::kUnitBytes
               : ops::smem_addr(smem + P::kPanel) + (k - P::kImageUnits) * P::kUnitBytes;
  };
  //: the fills: read pass `q` of window `w`, and fold unit `u`, into a pool landing on `bar`
  const volatile Addr& addr = *reinterpret_cast<const volatile Addr*>(smem + P::kAddr);
  const auto fill_read = [&](int w, int q, ops::SmemAddr pool, ops::SmemAddr bar) {
    const Window rw = window_of<P>(p, smem, kRead, w);
    const int first = q * P::kPassSlots;
    fill<P::kReadFactors, false>(
        addr.slices[kRead], nullptr, 0, union_of<P>(smem, kRead, w), rw, first,
        max(0, min(slots_of<P>(smem, rw, kRead, w) - first, P::kPassSlots)), pool, bar);
  };
  const auto fill_unit = [&](int w, int u, ops::SmemAddr pool, ops::SmemAddr bar) {
    const Window ww = window_of<P>(p, smem, kWrite, w);
    const int first = u * P::kUnitSlots;
    fill<P::kFoldFactors, true>(
        addr.slices[kWrite], addr.values, p.values.st, union_of<P>(smem, kWrite, w), ww, first,
        max(0, min(slots_of<P>(smem, ww, kWrite, w) - first, P::kUnitSlots)), pool, bar);
  };

  State<P> st;
  static_for<P::kFoldLists>([&](auto J) {
    static_for<P::kFoldMTiles>([&](auto M) {
      static_for<4>([&](auto N) { static_for<4>([&](auto I) { st.acc[J][M][N][I] = 0.f; }); });
    });
  });
  //: the readout ring's next fill to consume (slot `f & 1`), and each slot's barrier phase as a bit
  int f = 0;
  uint32_t ring_par = 0u, fold_par = 0u;

  stage_window<P>(p, c, 0, smem, staged, warp, lane);
  heads_in(p.write_lists, window_of<P>(p, smem, kWrite, 0), c, p.windows, 0, fold_heads, warp,
           lane);
  ops::mbar_arrive_when_landed(heads_bar);
  ops::mbar_wait(staged, 0u);
  fill_read(0, 0, ring(0), bars);

#pragma unroll 1
  for (int w = 0; w < p.windows; ++w) {
    const Window rw = window_of<P>(p, smem, kRead, w), ww = window_of<P>(p, smem, kWrite, w);
    const bool next = w + 1 < p.windows;
    const int read_slots = slots_of<P>(smem, rw, kRead, w),
              write_slots = slots_of<P>(smem, ww, kWrite, w);
    const int passes = max(1, (read_slots + P::kPassSlots - 1) / P::kPassSlots);
    const int units = max(1, (write_slots + P::kUnitSlots - 1) / P::kUnitSlots);

    //: THE DIGIT MAXIMA of the state before the window's writes
    post_maxima<P>(st, smem, warp, lane);
    clock.lap(kPhaseMaxima);

#pragma unroll 1
    for (int q = 0; __any_sync(0xffffffffu, q < passes); ++q) {
      //: THE NEXT FILL into the other slot: the next pass, or after the last the fold's first unit
      const int slot = f & 1;
      if (q + 1 < passes) {
        fill_read(w, q + 1, ring(slot ^ 1), bars + 8 * (slot ^ 1));
      } else {
        fill_unit(w, 0, ring(slot ^ 1), bars + 8 * (slot ^ 1));
      }
      ops::mbar_wait(bars + 8 * slot, (ring_par >> slot) & 1u);
      ring_par ^= 1u << slot;
      clock.lap(kPhaseLanded);
      const int first = q * P::kPassSlots, slots = min(read_slots - first, P::kPassSlots);
      //: THE HAND-OFF AND THE READOUT, part by part
      static_for<P::kParts>([&](auto Part) {
        ops::rendezvous();
        store_part<P, Part>(st, smem, warp, lane);
        ops::rendezvous();
        clock.lap(kPhaseStore);
        if (p.readout_f16) {
          readout_part<P, true>(p, smem, rw, c, w, q, slots, Part, ring(slot), warp, lane);
        } else {
          readout_part<P, false>(p, smem, rw, c, w, q, slots, Part, ring(slot), warp, lane);
        }
        clock.lap(kPhaseReadout);
      });
      ops::rendezvous();
      //: the image is free once the last pass is read: the fold ring's image units fill under the drain, and a
      //: resident fold's next window's inputs come in beside them
      if (q + 1 == passes) {
#pragma unroll 1
        for (int u = 1; u < min(units, 1 + P::kImageUnits); ++u)
          fill_unit(w, u, fold_slot(u - 1), bars + 8 * (1 + u));
        if constexpr (P::kResident)
          if (next) stage_window<P>(p, c, w + 1, smem, staged, warp, lane);
      }
      drain<P>(p, smem, rw, c, w, first, max(slots, 0));
      clock.lap(kPhaseDrain);
      ++f;
    }

    //: THE NEXT WINDOW'S INPUTS, the read heads once this warp's readout is done; resident, the panel's units and the
    //: next window's first pass into the last pass's slot, before the fold
    if constexpr (P::kResident) {
      //: the panel's units once every thread's drain has cleared it
      for (int k = ops::once(units > 1 + P::kImageUnits); k > 0; --k) {
        ops::rendezvous();
#pragma unroll 1
        for (int u = 1 + P::kImageUnits; u < units; ++u)
          fill_unit(w, u, fold_slot(u - 1), bars + 8 * (1 + u));
      }
      if (next) {
        ops::mbar_wait(staged, static_cast<uint32_t>((w + 1) & 1));
        fill_read(w + 1, 0, ring((f + 1) & 1), bars + 8 * ((f + 1) & 1));
      }
    } else if (next) {
      stage_window<P>(p, c, w + 1, smem, staged, warp, lane);
    }

    //: THE FOLD, unit by unit: unit 0 in the readout ring, the rest in the fold ring
    uint32_t cf[P::kFoldLists];
    float fs[P::kFoldLists];
    fold_units<P>(cf, fs, smem, ww, w, warp, lane);
    ops::mbar_wait(heads_bar, static_cast<uint32_t>(w & 1));
    const ops::SmemAddr image = ops::smem_addr(smem + P::kImage),
                        panel = ops::smem_addr(smem + P::kPanel);
    if constexpr (P::kResident) {
      //: THE RESIDENT FOLD: every unit landed, then each list once over them
      ops::mbar_wait(bars + 8 * (f & 1), (ring_par >> (f & 1)) & 1u);
      ring_par ^= 1u << (f & 1);
#pragma unroll 1
      for (int u = 1; u < units; ++u) {
        ops::mbar_wait(bars + 8 * (1 + u), (fold_par >> (u - 1)) & 1u);
        fold_par ^= 1u << (u - 1);
      }
      clock.lap(kPhaseFill);
      fold_unit<P>(p, st, cf, fs, smem, ww, c, w, 0, write_slots, ring(f & 1), image, panel, warp,
                   lane);
      ops::rendezvous();
      clock.lap(kPhaseFold);
      ++f;
    } else {
#pragma unroll 1
      for (int u = 0; __any_sync(0xffffffffu, u < units); ++u) {
        const int k = (u - 1) % P::kFoldRing;
        const ops::SmemAddr pool = u == 0 ? ring(f & 1) : fold_slot(k);
        if (u == 0) {
          ops::mbar_wait(bars + 8 * (f & 1), (ring_par >> (f & 1)) & 1u);
          ring_par ^= 1u << (f & 1);
        } else {
          ops::mbar_wait(bars + 8 * (2 + k), (fold_par >> k) & 1u);
          fold_par ^= 1u << k;
        }
        clock.lap(kPhaseFill);
        fold_unit<P>(p, st, cf, fs, smem, ww, c, w, u, write_slots, pool, image, panel, warp, lane);
        ops::rendezvous();
        clock.lap(kPhaseFold);
        //: a slot freed: the panel's units once the drain is done, the next window's first pass into unit 0's slot,
        //: a fold slot's next unit
        if (u == 0) {
          ++f;
#pragma unroll 1
          for (int v = 1 + P::kImageUnits; v < min(units, 1 + P::kFoldRing); ++v)
            fill_unit(w, v, fold_slot(v - 1), bars + 8 * (1 + v));
          if (next) {
            ops::mbar_wait(staged, static_cast<uint32_t>((w + 1) & 1));
            fill_read(w + 1, 0, ring(f & 1), bars + 8 * (f & 1));
          }
        } else if (u + P::kFoldRing < units) {
          fill_unit(w, u + P::kFoldRing, fold_slot(k), bars + 8 * (2 + k));
        }
      }
    }
    //: the panel zeroed where the fold's units lay in it, before the next window's adds
    for (int k = ops::once(units > 1 + P::kImageUnits); k > 0; --k)
#pragma unroll 1
      for (int i = static_cast<int>(threadIdx.x); i < P::kPanelBytes / 16; i += kThreads)
        reinterpret_cast<uint4*>(smem + P::kPanel)[i] = make_uint4(0u, 0u, 0u, 0u);
    //: the next window's fold heads, once this warp's fold is done
    if (next)
      heads_in(p.write_lists, window_of<P>(p, smem, kWrite, w + 1), c, p.windows, w + 1, fold_heads,
               warp, lane);
    ops::mbar_arrive_when_landed(heads_bar);
  }

  float sum = 0.f;
  static_for<P::kStateRegs>([&](auto k) { sum += st.template at<k>(); });
  atomicAdd(p.check, sum);
  fold_probe<P>(p, st, smem, window_of<P>(p, smem, kWrite, 0), place_of(p), warp, lane);
  clock.publish(p.ledger, (long)blockIdx.y * gridDim.x + blockIdx.x, kWarps, warp);
}

}  // namespace rola::carry
