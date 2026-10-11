// csrc/rola/src/carry/carry_kernel.cuh -- THE CARRY KERNEL: one CTA a box, the window loop over its components in
// sm_86's phase order (lists and units in, publish, readout passes each drained, fold passes), the fp32 state in
// registers throughout. see docs/internals/carry/carry_kernel.md
#pragma once

#include "carry/fold.cuh"
#include "common/phase_clock.cuh"

namespace rola::carry {

//: THE KERNEL, a grid constant block in, one CTA an SM: grid `(boxes, batch-heads)`. The readout passes double-buffer
//: their pool by `cp.async` groups; the fold passes ring three slots, each landed on its barrier.
//: -- see docs/internals/carry/carry_kernel.md#kernel
template <class Arm>
__global__ __launch_bounds__(kThreads, 1) void carry_kernel(const __grid_constant__ CarryParams p) {
  using P = Plan<Arm>;
  extern __shared__ __align__(128) uint8_t smem[];
  const int warp = (int)threadIdx.x / 32, lane = (int)threadIdx.x % 32;
  const Place c = place_of(p);
  PhaseClock<kCarryPhases, true> clock(
      reinterpret_cast<long long*>(smem + P::kClock) + warp * PhaseClock<kCarryPhases, true>::kRow,
      lane, p.ledger != nullptr);
  const ops::SmemAddr bars = ops::smem_addr(smem + P::kBars);
  if (threadIdx.x == 0)
    static_for<P::kWriteSlots>([&](auto s) { ops::mbar_init(bars + 8 * s, kThreads); });
  if (static_cast<int>(threadIdx.x) < P::kWriteLists)
    reinterpret_cast<UnitBook*>(smem + P::kBook)[threadIdx.x] = UnitBook{kUnitNone, 0.0f};
  ops::rendezvous();

  const uint16_t* write_tok = reinterpret_cast<const uint16_t*>(smem + P::kUnion + kWindow * 2);
  const uint16_t* read_tok = reinterpret_cast<const uint16_t*>(smem + P::kUnion);
  const ops::SmemAddr read_pool = ops::smem_addr(smem + P::kPhase + P::kReadPool);
  const ops::SmemAddr write_pool = ops::smem_addr(smem + P::kPhase);
  uint32_t* const panel = reinterpret_cast<uint32_t*>(smem + P::kPhase + P::kPanel);
  State<P> st;
  static_for<P::kFoldLists>([&](auto J) {
    static_for<P::kFoldMTiles>([&](auto M) {
      static_for<4>([&](auto N) { static_for<4>([&](auto I) { st.acc[J][M][N][I] = 0.f; }); });
    });
  });
  int uses0 = 0, uses1 = 0;

#pragma unroll 1
  for (int w = 0; w < p.windows; ++w) {
    lists_in<P>(p, p.write_lists, window_of(p, p.write, w), c, w, smem, 1, P::kWriteAt);
    lists_in<P>(p, p.read_lists, window_of(p, p.read, w), c, w, smem, 0, P::kReadAt);
    exps_in<P>(p, window_of(p, p.write, w), c, w, smem);
    ops::stage_commit();
    ops::stage_wait<0>();
    ops::rendezvous();
    clock.lap(kPhaseLists);
    units_in<P>(p, c, w * kWindow, min(kWindow, p.L - w * kWindow), smem);
    clock.lap(kPhaseUnits);

    {
      const Window ww = window_of(p, p.write, w);
      const int write_slots = slots_of(smem, P::kCount, ww, 1);
      for (int k = ops::once(write_slots > 0); k > 0; --k)
        fill_unit<P>(p, c, ww, write_slots, w, 0, smem);
    }

    uint32_t weight = 0u;
    {
      uint32_t read[P::kReadRegs];
      publish<P>(st, read, smem, warp, lane);
      static_for<P::kReadRegs>([&](auto k) { weight ^= read[k]; });
    }
    clock.lap(kPhasePublish);

    const int read_slots = slots_of(smem, P::kCount, window_of(p, p.read, w), 0);
    const int read_passes = (read_slots + kPassSlots - 1) / kPassSlots;
    fill<P::kReadFactors, P::kReadPitch, false>(slices_of<P::kReadFactors>(p, p.read, c), nullptr,
                                                0, 0, read_tok, window_of(p, p.read, w), 0,
                                                min(read_slots, kPassSlots), read_pool);
    ops::stage_commit();
#pragma unroll 1
    for (int q = 0; q < read_passes; ++q) {
      const Window rw = window_of(p, p.read, w);
      const int next = (q + 1) * kPassSlots;
      if (q + 1 < read_passes)
        fill<P::kReadFactors, P::kReadPitch, false>(
            slices_of<P::kReadFactors>(p, p.read, c), nullptr, 0, 0, read_tok, rw, next,
            min(read_slots - next, kPassSlots),
            read_pool + ((q + 1) & 1) * kPassSlots * P::kReadPitch);
      ops::stage_commit();
      ops::stage_wait<1>();
      ops::rendezvous();
      const int first = q * kPassSlots, slots = min(read_slots - first, kPassSlots);
      readout_pass<P>(p, weight, smem, rw, c, w, q, slots,
                      read_pool + (q & 1) * kPassSlots * P::kReadPitch, ops::smem_addr(panel), warp,
                      lane);
      ops::rendezvous();
      clock.lap(kPhaseReadout);
      drain<P>(p, smem, rw, c, first, slots, panel);
      ops::rendezvous();
      clock.lap(kPhaseDrain);
    }

    const Window ww = window_of(p, p.write, w);
    const int write_slots = slots_of(smem, P::kCount, ww, 1);
    const int units = (write_slots + P::kUnitSlots - 1) / P::kUnitSlots;
    const bool stepped = P::kFoldFactors == 1 && ww.listed;
    uint32_t cf[P::kFoldLists] = {};
    float f[P::kFoldLists] = {};
    for (int k = ops::once(stepped); k > 0; --k) fold_units<P>(cf, f, smem, ww, warp, lane);
#pragma unroll 1
    for (int u = 0; u < units; ++u) {
      for (int k = ops::once(u + 1 < units); k > 0; --k)
        fill_unit<P>(p, c, ww, write_slots, w, u + 1, smem);
      const int slot = u & 1;
      const ops::SmemAddr pool = write_pool + (slot ? 0 : P::kWriteSlot0);
      ops::mbar_wait(bars + 8 * slot, static_cast<uint32_t>(slot ? uses1 : uses0) & 1u);
      uses0 += 1 - slot;
      uses1 += slot;
      clock.lap(kPhaseFill);
      for (int k = ops::once(stepped); k > 0; --k)
        fold_unit<P>(p, st, cf, f, smem, ww, c, w, u, pool, warp, lane);
      for (int k = ops::once(!stepped); k > 0; --k)
        fold_stub<P>(p, st, smem, ww, c, w, u, min(write_slots - u * P::kUnitSlots, P::kUnitSlots),
                     pool, warp, lane);
      ops::rendezvous();
      clock.lap(kPhaseFold);
    }
  }

  float sum = 0.f;
  static_for<P::kStateRegs>([&](auto k) { sum += st.template at<k>(); });
  atomicAdd(p.check, sum);
  fold_probe<P>(p, st, smem, window_of(p, p.write, 0), c, warp, lane);
  clock.publish(p.ledger, (long)blockIdx.y * gridDim.x + blockIdx.x, kWarps, warp);
}

}  // namespace rola::carry
