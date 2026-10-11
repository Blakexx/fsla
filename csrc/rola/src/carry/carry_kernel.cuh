// csrc/rola/src/carry/carry_kernel.cuh -- THE CARRY KERNEL: one CTA a box, the window loop over its components in
// sm_86's phase order (lists and units in, publish, readout passes each drained, fold passes), the fp32 state in
// registers throughout. see docs/internals/carry/carry_kernel.md
#pragma once

#include "carry/parts.cuh"
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
  const Slices<P::kFoldFactors> ws = slices_of<P::kFoldFactors>(p, p.write, c);
  const Slices<P::kReadFactors> rs = slices_of<P::kReadFactors>(p, p.read, c);
  const uint8_t* values = p.values.base + c.b * p.values.sb + c.h * p.values.sh + c.channel0 * 2;
  PhaseClock<kCarryPhases, true> clock(
      reinterpret_cast<long long*>(smem + P::kClock) + warp * PhaseClock<kCarryPhases, true>::kRow,
      lane, p.ledger != nullptr);
  const ops::SmemAddr bars = ops::smem_addr(smem + P::kBars);
  if (threadIdx.x == 0)
    static_for<P::kWriteSlots>([&](auto s) { ops::mbar_init(bars + 8 * s, kThreads); });
  ops::rendezvous();

  const uint16_t* write_tok = reinterpret_cast<const uint16_t*>(smem + P::kUnion + kWindow * 2);
  const uint16_t* read_tok = reinterpret_cast<const uint16_t*>(smem + P::kUnion);
  const ops::SmemAddr read_pool = ops::smem_addr(smem + P::kPhase + P::kReadPool);
  const ops::SmemAddr write_pool = ops::smem_addr(smem + P::kPhase + P::kWritePool);
  uint32_t* const panel = reinterpret_cast<uint32_t*>(smem + P::kPhase + P::kPanel);
  float state[P::kStateRegs];
  static_for<P::kStateRegs>([&](auto k) { state[k] = 0.f; });
  uint32_t read[P::kReadRegs];
  int folded = 0;

#pragma unroll 1
  for (int w = 0; w < p.windows; ++w) {
    const Window ww = window_of(p, p.write, w), rw = window_of(p, p.read, w);
    lists_in<P>(p, p.write_lists, ww, c, w, smem, 1, P::kWriteAt);
    lists_in<P>(p, p.read_lists, rw, c, w, smem, 0, P::kReadAt);
    ops::stage_commit();
    ops::stage_wait<0>();
    ops::rendezvous();
    clock.lap(kPhaseLists);
    units_in<P>(p, c, rw.t0, rw.tokens, smem);
    clock.lap(kPhaseUnits);

    publish<P>(state, read, smem, warp, lane);
    clock.lap(kPhasePublish);

    const int read_slots = slots_of(smem, P::kCount, rw, 0);
    const int read_passes = (read_slots + kPassSlots - 1) / kPassSlots;
    fill<P::kReadFactors, P::kReadPitch, false>(rs, nullptr, 0, 0, read_tok, rw, 0,
                                                min(read_slots, kPassSlots), read_pool);
    ops::stage_commit();
#pragma unroll 1
    for (int q = 0; q < read_passes; ++q) {
      const int next = (q + 1) * kPassSlots;
      if (q + 1 < read_passes)
        fill<P::kReadFactors, P::kReadPitch, false>(
            rs, nullptr, 0, 0, read_tok, rw, next, min(read_slots - next, kPassSlots),
            read_pool + ((q + 1) & 1) * kPassSlots * P::kReadPitch);
      ops::stage_commit();
      ops::stage_wait<1>();
      ops::rendezvous();
      const int first = q * kPassSlots, slots = min(read_slots - first, kPassSlots);
      readout_pass<P>(p, read, smem, rw, c, w, q, slots,
                      read_pool + (q & 1) * kPassSlots * P::kReadPitch, ops::smem_addr(panel), warp,
                      lane);
      ops::rendezvous();
      clock.lap(kPhaseReadout);
      drain<P>(p, smem, rw, c, first, slots, panel);
      ops::rendezvous();
      clock.lap(kPhaseDrain);
    }

    const int write_slots = slots_of(smem, P::kCount, ww, 1);
    const int write_passes = (write_slots + kPassSlots - 1) / kPassSlots;
    auto fill_write = [&](int q) {
      const int slot = (folded + q) % P::kWriteSlots, first = q * kPassSlots;
      fill<P::kFoldFactors, P::kWritePitch, true>(ws, values, p.values.st, P::kChannels, write_tok,
                                                  ww, first, min(write_slots - first, kPassSlots),
                                                  write_pool + slot * kPassSlots * P::kWritePitch);
      ops::mbar_arrive_when_landed(bars + 8 * slot);
    };
#pragma unroll 1
    for (int q = 0; q < P::kWriteSlots - 1 && q < write_passes; ++q) fill_write(q);
#pragma unroll 1
    for (int q = 0; q < write_passes; ++q) {
      if (q + P::kWriteSlots - 1 < write_passes) fill_write(q + P::kWriteSlots - 1);
      const int g = folded + q, slot = g % P::kWriteSlots;
      ops::mbar_wait(bars + 8 * slot, (uint32_t)(g / P::kWriteSlots) & 1u);
      fold_pass<P>(p, state, smem, ww, c, w, q, min(write_slots - q * kPassSlots, kPassSlots),
                   write_pool + slot * kPassSlots * P::kWritePitch, warp, lane);
      ops::rendezvous();
      clock.lap(kPhaseFold);
    }
    folded += write_passes;
  }

  float sum = 0.f;
  static_for<P::kStateRegs>([&](auto k) { sum += state[k]; });
  atomicAdd(p.check, sum);
  clock.publish(p.ledger, (long)blockIdx.y * gridDim.x + blockIdx.x, kWarps, warp);
}

}  // namespace rola::carry
