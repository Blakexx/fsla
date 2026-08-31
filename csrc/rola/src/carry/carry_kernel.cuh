// csrc/rola/src/carry/carry_kernel.cuh -- THE CARRY BODY: the CTA box's prologue, its
// state slice, and the window loop whose every pipeline slot is a named stage.
//
// The kernel is ONE frame with four stage families: the two legs that exchange the
// accumulator between the sides' carves, PHASE 1's readout batches and PHASE 2's fold
// batches. Three split edges join them and nothing else in the loop is CTA-wide.
// The issue schedule, the accumulator's orientation, the primitive ledger and the
// per-moment shared-memory tenants: see docs/internals/carry/carry_kernel.md and
// docs/internals/carry/smem_ledger.md
#pragma once

#include <cuda_bf16.h>

#include "carry/box.cuh"
#include "carry/params.cuh"
#include "common/design.cuh"
#include "common/geom.cuh"
#include "common/ops.cuh"
#include "common/static_for.cuh"
#include "common/state_page.cuh"
#include "common/structure_switch.cuh"

namespace rola::carry {

namespace ops = denseref::ops;

//: THE SHARED BLOCK, as the ledger lays it out. -- see docs/internals/carry/smem_ledger.md
template <class BP>
struct Smem {
  char* base;

  __device__ __forceinline__ float* page_image(int i) const {
    return reinterpret_cast<float*>(base + BP::kRegionOffset + i * BP::kPageImageBytes);
  }

  __device__ __forceinline__ ops::SmemAddr v_tile(int slot) const {
    return ops::smem_addr(base + BP::kRegionOffset + slot * BP::kVTileBytes);
  }

  __device__ __forceinline__ char* column_block() const {
    return base + BP::kRegionOffset + BP::kColOffsetInRegion;
  }

  __device__ __forceinline__ char* raw_slot(int slot) const {
    return base + BP::kRawOffset + slot * BP::kRawSlotBytes;
  }

  __device__ __forceinline__ __nv_bfloat16* gain_slot(int slot) const {
    return reinterpret_cast<__nv_bfloat16*>(base + BP::kGainOffset + slot * BP::kGainSlotBytes);
  }

  __device__ __forceinline__ ops::SmemAddr stream_edge(StreamEdge e) const {
    return ops::smem_addr(base + BP::kStreamEdgeOffset + (int)e * (int)sizeof(ops::EdgeState));
  }

  __device__ __forceinline__ unsigned int* claim(Fill f) const {
    return reinterpret_cast<unsigned int*>(base + BP::kClaimOffset) + (int)f;
  }

  __device__ __forceinline__ ops::SmemAddr edge(Edge e) const {
    return ops::smem_addr(base + BP::kEdgeOffset + (int)e * (int)sizeof(ops::EdgeState));
  }

  __device__ __forceinline__ int32_t* page_slots() const {
    return reinterpret_cast<int32_t*>(base + BP::kPageSlotOffset);
  }
};

//: THE V TILE'S SWIZZLE: the 16-byte group index XORed with the token row, which is a
//: bijection of the tile and leaves `ldmatrix .trans` conflict free.
//: -- see docs/internals/carry/smem_ledger.md#v-tile
template <class BP>
__device__ __forceinline__ int v_group_offset(int token_row, int group) {
  constexpr int kGroups = BP::kDv / 8;
  return (token_row * BP::kDv + ((group ^ (token_row & (kGroups - 1))) * 8)) * 2;
}

//: ONE WARP'S SLOT, as the owner-local canonical index, with the member's spans as
//: immediates. -- see docs/internals/carry/carry_kernel.md#slot-map
template <int D, int BC, int NS, int A>
__device__ __forceinline__ int slot_local(const SideGeom& s, int strm, int P) {
  int local = 0;
  rola::static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    constexpr int sb = sb_span_bits(D, BC, NS, A, l);
    constexpr int ss = sb_shift_of(D, BC, NS, A, l);
    const int off = ((strm >> s.grid_div_bits[l]) & (s.grid[l] - 1)) << sb;
    const int dig = (P >> ss) & ((1 << sb) - 1);
    local += (off + dig) << owner_run_shift(D, BC, l);
  });
  return local;
}

//: THE ACCUMULATOR, C[m = channel][n = leaf], one warp's share.
//: -- see docs/internals/carry/carry_kernel.md#accumulator
template <class BP>
struct Acc {
  float c[BP::kMCh][BP::kNL][4];
};

// --------------------------------------------------------------- the frame's stages

//: PUBLISH: the accumulator's top halves into the transit region, in the write carve.
template <class BP, int RA, int WA>
__device__ __forceinline__ void out_leg_publish(const CarryParams&, const Smem<BP>&, Acc<BP>&) {}

//: COVER SLOT A: stage and expand read batch zero while the out leg's edge settles.
template <class BP, int RA>
__device__ __forceinline__ void cover_read_batch_zero(const CarryParams&, const Smem<BP>&, int,
                                                      int) {}

//: PULL: the region's rows back as the read side's A fragments.
template <class BP, int RA>
__device__ __forceinline__ void out_leg_pull(const CarryParams&, const Smem<BP>&) {}

//: STAGE: read batch b + 1's operands into the read stream's segment slot.
template <class BP, int RA>
__device__ __forceinline__ void readout_stage(const CarryParams&, const Smem<BP>&, int, int) {}

//: PRODUCE: read batch b + 1's coefficients, expanded from the staged factor rows.
template <class BP, int RA>
__device__ __forceinline__ void readout_produce(const CarryParams&, const Smem<BP>&, int, int) {}

//: MMA: read batch b against the pulled A fragments.
template <class BP, int RA>
__device__ __forceinline__ void readout_mma(const CarryParams&, const Smem<BP>&, int, int) {}

//: FAN-IN: read batch b - 1's panel reduced into num and den.
template <class BP, int RA>
__device__ __forceinline__ void readout_fan_in(const CarryParams&, const Smem<BP>&, int, int) {}

//: STORE-BACK: the region's rows written back as the accumulator's low halves.
template <class BP, int RA, int WA>
__device__ __forceinline__ void return_leg_store(const CarryParams&, const Smem<BP>&, Acc<BP>&) {}

//: REJOIN: the two halves back into the fp32 accumulator, exactly.
template <class BP, int WA>
__device__ __forceinline__ void return_leg_rejoin(const CarryParams&, const Smem<BP>&, Acc<BP>&) {}

// ------------------------------------------------------------------- PHASE 2's fold

//: ONE CLAIM on the stream's cursor. The cursor is MONOTONE and never reset: a warp
//: leaves a fill with exactly one failing claim, so every production point advances it by
//: its unit count plus one per warp and the next point knows its own base.
//: -- see docs/internals/carry/carry_kernel.md#fill
template <class BP>
__device__ __forceinline__ uint32_t claim_unit(const Smem<BP>& sm, Fill f, int lane) {
  unsigned int u = 0u;
  if (lane == 0) u = atomicAdd(sm.claim(f), 1u);
  return (uint32_t)ops::lane_broadcast((int)u, 0);
}

//: THE VALUE ROWS of one fill unit, into the stream's segment slot. A token past the
//: window's end zero fills and reads nothing.
template <class BP>
__device__ __forceinline__ void stage_unit_values(const CarryParams& p, const Smem<BP>& sm,
                                                  int slot, int lane, int bh, int seg0, int tok_end,
                                                  int unit) {
  constexpr int kGroups = BP::kDv / 8;
  constexpr int kQuanta = kUnitTokens * kGroups;
  const ops::SmemAddr tile = sm.v_tile(slot);
  const int row0 = unit * kUnitTokens;
  rola::static_for<(kQuanta + 31) / 32>([&](auto Ic) {
    const int idx = decltype(Ic)::value * 32 + lane;
    if (kQuanta % 32 != 0 && idx >= kQuanta) return;
    const int row = row0 + idx / kGroups, group = idx % kGroups;
    const int tok = seg0 + row;
    const uint32_t off =
        ((uint32_t)bh * (uint32_t)p.L + (uint32_t)(tok < tok_end ? tok : tok_end - 1))
            * (uint32_t)BP::kDv * 2u
        + (uint32_t)(group * 16);
    ops::stage_tile_if(tile + (uint32_t)v_group_offset<BP>(row, group), ops::gather_ptr(p.v, off),
                       tok < tok_end);
  });
}

//: THE FACTOR ROWS of one fill unit, staged AS THEY LIE: each token's D contiguous runs of
//: the box's own digits, at the widest quantum each run admits.
//: -- see docs/internals/carry/carry_kernel.md#factor-rows
template <class BP, int D>
__device__ __forceinline__ void stage_unit_factors(const CarryParams& p, const Smem<BP>& sm,
                                                   const int (&abase)[kGeomMaxLevels], int slot,
                                                   int lane, int bh, int seg0, int tok_end,
                                                   int unit) {
  const uint32_t plane = (uint32_t)bh * (uint32_t)p.L * (uint32_t)p.g.wtot * 2u;
  const ops::SmemAddr dst0 = ops::smem_addr(sm.raw_slot(slot));
  const int row0 = unit * kUnitTokens;
  rola::static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    constexpr int kSpan = owner_span(D, BP::kBC, l);
    constexpr int kBase = span_base(D, BP::kBC, l);
    constexpr int kQ = run_quantum(kSpan, kBase, BP::kRawStride);
    constexpr int kPerToken = kSpan * 2 / kQ;
    constexpr int kQuanta = kUnitTokens * kPerToken;
    rola::static_for<(kQuanta + 31) / 32>([&](auto Ic) {
      const int idx = decltype(Ic)::value * 32 + lane;
      if (kQuanta % 32 != 0 && idx >= kQuanta) return;
      const int row = row0 + idx / kPerToken, q = idx % kPerToken;
      const int tok = seg0 + row;
      const uint32_t off = plane
                           + (uint32_t)(tok < tok_end ? tok : tok_end - 1) * (uint32_t)p.g.wtot * 2u
                           + (uint32_t)(abase[l] * 2 + q * kQ);
      ops::stage_run_if<kQ>(dst0 + (uint32_t)((row * BP::kRawStride + kBase) * 2 + q * kQ),
                            ops::gather_ptr(p.pwrite, off), tok < tok_end);
    });
  });
}

//: PRODUCE ONE SEGMENT: every warp claims units until the segment is taken, so the fill
//: shares out by arrival rather than by assignment.
template <class BP, int D>
__device__ __forceinline__ void fold_produce(const CarryParams& p, const Smem<BP>& sm,
                                             const int (&abase)[kGeomMaxLevels], int lane, int bh,
                                             int seg0, int tok_end, int vslot, int fslot,
                                             bool values, bool factors, Fill point,
                                             uint32_t& base) {
  constexpr int kUnits = BP::kSeg / kUnitTokens;
  for (;;) {
    const uint32_t u = claim_unit<BP>(sm, point, lane);
    if (u >= base + kUnits) break;
    const int unit = (int)(u - base);
    if (values) stage_unit_values<BP>(p, sm, vslot, lane, bh, seg0, tok_end, unit);
    if (factors) stage_unit_factors<BP, D>(p, sm, abase, fslot, lane, bh, seg0, tok_end, unit);
  }
  base += (uint32_t)(kUnits + BP::kWarps);
  ops::stage_commit();
}

//: THE SEGMENT'S GAINS, read as one coalesced row and held in a register until the
//: iteration that covers the read has issued its deposits.
template <class BP>
__device__ __forceinline__ unsigned short read_gain(const CarryParams& p, int lane, int bh,
                                                    int seg0, int tok_end) {
  const int tok = seg0 + lane;
  if (lane >= BP::kSeg || tok >= tok_end) return 0u;
  return *reinterpret_cast<const unsigned short*>(
      ops::gather_ptr(p.gwrite, ((uint32_t)bh * (uint32_t)p.L + (uint32_t)tok) * 2u));
}

template <class BP>
__device__ __forceinline__ void write_gain(const Smem<BP>& sm, int slot, int lane,
                                           unsigned short g) {
  if (lane < BP::kSeg) reinterpret_cast<unsigned short*>(sm.gain_slot(slot))[lane] = g;
}

//: THE TRANSPOSE: the staged factor rows into the form the coefficient's PACKED multiply
//: reads. The MMA's B fragment pairs two ADJACENT TOKENS in one register, so a coefficient
//: pair is one column at two tokens -- which the packed plane stores a row apart and this
//: block stores adjacent. The outermost level's columns carry the gain, applied here once
//: per token pair instead of once per leaf.
//: -- see docs/internals/carry/carry_kernel.md#column-block
template <class BP, int D>
__device__ __forceinline__ void fold_transpose(const Smem<BP>& sm, int raw, int gain, int tid) {
  constexpr int kPairs = BP::kSpanTotal / 2;
  constexpr int kRows = BP::kSeg / 2;
  constexpr int kUnits = kPairs * kRows;
  constexpr int kGainPairs = owner_span(D, BP::kBC, 0) / 2;
  const uint32_t* const src = reinterpret_cast<const uint32_t*>(sm.raw_slot(raw));
  uint32_t* const dst = reinterpret_cast<uint32_t*>(sm.column_block());
  const unsigned short* const g = reinterpret_cast<const unsigned short*>(sm.gain_slot(gain));
  rola::static_for<(kUnits + BP::kThreads - 1) / BP::kThreads>([&](auto Ic) {
    const int idx = decltype(Ic)::value * BP::kThreads + tid;
    if (kUnits % BP::kThreads != 0 && idx >= kUnits) return;
    const int u = idx & (kRows - 1), pair = idx / kRows;
    const int row = 2 * u;
    const uint32_t w0 = src[row * (BP::kRawStride / 2) + pair];
    const uint32_t w1 = src[(row + 1) * (BP::kRawStride / 2) + pair];
    uint32_t lo = (w0 & 0xFFFFu) | (w1 << 16);
    uint32_t hi = (w0 >> 16) | (w1 & 0xFFFF0000u);
    if (pair < kGainPairs) {
      const uint32_t gp = (uint32_t)g[row] | ((uint32_t)g[row + 1] << 16);
      lo = ops::mul_bf16x2(lo, gp);
      hi = ops::mul_bf16x2(hi, gp);
    }
    dst[(2 * pair) * (BP::kTokStride / 2) + u] = lo;
    dst[(2 * pair + 1) * (BP::kTokStride / 2) + u] = hi;
  });
}

//: EXPAND: one segment's coefficients for this lane's leaves, in registers. Each level
//: contributes one PACKED multiply per (leaf tile, token pair); the slot map `scol` was
//: walked once at the prologue and is never walked again.
template <class BP, int D>
__device__ __forceinline__ void fold_expand(const Smem<BP>& sm, uint32_t (&bf)[BP::kNL][2],
                                            const int (&scol)[BP::kNL][kGeomMaxLevels], int lane) {
  const uint32_t* const col = reinterpret_cast<const uint32_t*>(sm.column_block());
  const int t2 = lane & 3;
  rola::static_for<BP::kNL>([&](auto Jc) {
    constexpr int j = decltype(Jc)::value;
    rola::static_for<2>([&](auto Hc) {
      constexpr int h = decltype(Hc)::value;
      const int word = t2 + 4 * h;
      uint32_t x = col[scol[j][0] * (BP::kTokStride / 2) + word];
      rola::static_for<D - 1>([&](auto Lc) {
        constexpr int l = decltype(Lc)::value + 1;
        x = ops::mul_bf16x2(x, col[scol[j][l] * (BP::kTokStride / 2) + word]);
      });
      bf[j][h] = x;
    });
  });
}

//: THE MASS CHANNEL'S A FRAGMENT: a constant. The deposit value is the token's values
//: followed by a one, so the accumulator's last m-tile has a single live row.
template <class BP>
__device__ __forceinline__ void mass_frag(uint32_t (&a)[4], int lane) {
  const uint32_t one = ((lane >> 2) == BP::kMassRow) ? 0x3F803F80u : 0u;
  a[0] = one;
  a[1] = 0u;
  a[2] = one;
  a[3] = 0u;
}

//: DRAIN: the segment's value tile into the A fragments, which ARE the second buffer --
//: the slot is free the moment the burst has issued.
template <class BP>
__device__ __forceinline__ void fold_drain(const Smem<BP>& sm, uint32_t (&vf)[BP::kMCh][4],
                                           int slot, int lane) {
  rola::static_for<BP::kMCh>([&](auto Mc) {
    constexpr int mc = decltype(Mc)::value;
    if constexpr (mc == BP::kMassTile) {
      mass_frag<BP>(vf[mc], lane);
    } else {
      //: THE TRANSPOSING TILE LOAD reads the segment as it is stored -- token major,
      //: channel minor -- and returns it as the A operand, channel major. The four
      //: 8x8 blocks are addressed by lanes in tile order, so the lane's token row
      //: carries the tile's k half and its channel group the tile's m half.
      const int row = (lane & 7) + ((lane >> 4) & 1) * 8;
      const int group = mc * 2 + ((lane >> 3) & 1);
      ops::load_frag_t(vf[mc], sm.v_tile(slot) + (uint32_t)v_group_offset<BP>(row, group));
    }
  });
}

//: COVER SLOT B: the write side's first segments' ROUTING data, walked and staged while
//: the return leg's edge settles. It reads routing only -- never the accumulator and never
//: the region -- which is what makes it independent of the phase it covers.
template <class BP, int D>
__device__ __forceinline__ void cover_write_batch_zero(const CarryParams& p, const Smem<BP>& sm,
                                                       const int (&abase)[kGeomMaxLevels], int warp,
                                                       int lane, int bh, int t0, int wLen,
                                                       uint32_t (&claim)[kFillPoints]) {
  const int nSeg = (wLen + BP::kSeg - 1) / BP::kSeg;
  const int tok_end = t0 + wLen;
  rola::static_for<BP::kSegmentSlots>([&](auto Ic) {
    constexpr int s = decltype(Ic)::value;
    if (s >= nSeg) return;
    const int seg0 = t0 + s * BP::kSeg;
    const unsigned short gain = warp == 0 ? read_gain<BP>(p, lane, bh, seg0, tok_end) : 0u;
    fold_produce<BP, D>(p, sm, abase, lane, bh, seg0, tok_end, 0, s, false, true,
                        Fill::kCoverFactors, claim[(int)Fill::kCoverFactors]);
    if (warp == 0) write_gain<BP>(sm, s, lane, gain);
  });
}

//: PHASE 2, one form. The fold is WINDOW-BLIND: a deposit never reads the accumulator, so
//: the segment MMAs of a whole window chain with no accumulator dependency and the
//: segment slot's refill is covered by the previous segment's chain.
//: -- see docs/internals/carry/carry_kernel.md#fold
template <class BP, int D, Form F>
__device__ __forceinline__ void fold_phase(const CarryParams& p, const Smem<BP>& sm, Acc<BP>& acc,
                                           const int (&scol)[BP::kNL][kGeomMaxLevels],
                                           const int (&abase)[kGeomMaxLevels], int tid, int warp,
                                           int lane, int bh, int t0, int wLen,
                                           uint32_t (&claim)[kFillPoints], uint32_t& phase) {
  constexpr int S = BP::kSegmentSlots;
  constexpr int Fs = BP::kFactorSlots;
  const int nSeg = (wLen + BP::kSeg - 1) / BP::kSeg;
  const int tok_end = t0 + wLen;

  //: the value rows of the first slots; their factor rows are already in the ring, staged
  //: by the cover slot the return leg's edge hid.
  rola::static_for<S>([&](auto Ic) {
    constexpr int s = decltype(Ic)::value;
    if (s < nSeg)
      fold_produce<BP, D>(p, sm, abase, lane, bh, t0 + s * BP::kSeg, tok_end, s, s, true, false,
                          Fill::kPrologueValues, claim[(int)Fill::kPrologueValues]);
  });

#pragma unroll 1
  for (int b = 0; b < nSeg; ++b) {
    const int ahead = b + S;
    const int seg0 = t0 + b * BP::kSeg;
    const unsigned short gain =
        warp == 0 ? read_gain<BP>(p, lane, bh, t0 + ahead * BP::kSeg, tok_end) : 0u;

    ops::stage_wait<S - 1>();
    ops::edge_arrive(sm.stream_edge(StreamEdge::kFilled));
    ops::edge_wait(sm.stream_edge(StreamEdge::kFilled), phase);

    uint32_t vf[BP::kMCh][4];
    fold_drain<BP>(sm, vf, b % S, lane);
    fold_transpose<BP, D>(sm, b % Fs, b % Fs, tid);

    ops::edge_arrive(sm.stream_edge(StreamEdge::kEmpty));
    ops::edge_wait(sm.stream_edge(StreamEdge::kEmpty), phase);

    if (ahead < nSeg)
      fold_produce<BP, D>(p, sm, abase, lane, bh, t0 + ahead * BP::kSeg, tok_end, ahead % S,
                          ahead % Fs, true, true, Fill::kSegment, claim[(int)Fill::kSegment]);

    uint32_t bf[BP::kNL][2];
    fold_expand<BP, D>(sm, bf, scol, lane);

    //: DEPOSIT: one MMA per (channel m-tile, leaf n-tile), the segment as the k step.
    rola::static_for<BP::kMCh>([&](auto Mc) {
      constexpr int mc = decltype(Mc)::value;
      rola::static_for<BP::kNL>([&](auto Jc) {
        constexpr int j = decltype(Jc)::value;
        ops::mma(acc.c[mc][j], vf[mc], bf[j]);
      });
    });

    if (warp == 0 && ahead < nSeg) write_gain<BP>(sm, ahead % Fs, lane, gain);
    phase ^= 1u;
  }
  ops::stage_wait<0>();
}

//: PHASE 1, one form: the four slots of a read batch, issued so each covers the next.
template <class BP, int RA, Form F>
__device__ __forceinline__ void readout_phase(const CarryParams& p, const Smem<BP>& sm, int wLen) {
  const int nSeg = (wLen + BP::kSeg - 1) / BP::kSeg;
#pragma unroll 1
  for (int b = 0; b < nSeg; ++b) {
    readout_stage<BP, RA>(p, sm, b + 1, nSeg);
    readout_produce<BP, RA>(p, sm, b + 1, nSeg);
    readout_mma<BP, RA>(p, sm, b, nSeg);
    readout_fan_in<BP, RA>(p, sm, b - 1, nSeg);
  }
}

// ------------------------------------------------------------------- the window loop

template <class BP, int D, int RA, int WA>
__device__ __forceinline__ void window_loop(const CarryParams& p, const Smem<BP>& sm, Acc<BP>& acc,
                                            const int (&scol)[BP::kNL][kGeomMaxLevels],
                                            const int (&abase)[kGeomMaxLevels], int tid, int warp,
                                            int lane, int bh) {
  const int nWindows = (p.L + kWindow - 1) / kWindow;
  uint32_t phase = 0u, seg_phase = 0u;
  uint32_t claim[kFillPoints] = {0u, 0u, 0u};

#pragma unroll 1
  for (int win = 0; win < nWindows; ++win) {
    const int t0 = win * kWindow;
    const int wLen = p.L - t0 < kWindow ? p.L - t0 : kWindow;

    out_leg_publish<BP, RA, WA>(p, sm, acc);
    ops::edge_arrive(sm.edge(Edge::kOutLeg));
    cover_read_batch_zero<BP, RA>(p, sm, t0, wLen);
    ops::edge_wait(sm.edge(Edge::kOutLeg), phase);
    out_leg_pull<BP, RA>(p, sm);

    rola::uniform_switch<kForms>(p.readout_form, [&](auto Rf) {
      readout_phase<BP, RA, (Form) decltype(Rf)::value>(p, sm, wLen);
    });

    return_leg_store<BP, RA, WA>(p, sm, acc);
    ops::edge_arrive(sm.edge(Edge::kReturnLeg));
    cover_write_batch_zero<BP, D>(p, sm, abase, warp, lane, bh, t0, wLen, claim);
    ops::edge_wait(sm.edge(Edge::kReturnLeg), phase);
    return_leg_rejoin<BP, WA>(p, sm, acc);

    rola::uniform_switch<kForms>(p.fold_form, [&](auto Ff) {
      fold_phase<BP, D, (Form) decltype(Ff)::value>(p, sm, acc, scol, abase, tid, warp, lane, bh,
                                                    t0, wLen, claim, seg_phase);
    });

    ops::edge_arrive(sm.edge(Edge::kWindowLine));
    ops::edge_wait(sm.edge(Edge::kWindowLine), phase);
    phase ^= 1u;
  }
}

// --------------------------------------------------------------------- the state slice

//: THE ENTRY SWEEP: the box's pages into the accumulator, hi alone for a page this call
//: only reads and hi with lo for one it will write.
//: -- see docs/internals/carry/carry_kernel.md#state-io
template <class BP, int D, int NS, int WA, bool Load>
__device__ __forceinline__ void state_sweep(const CarryParams& p, const Smem<BP>& sm, Acc<BP>& acc,
                                            const int32_t* slots, uint32_t act_read,
                                            uint32_t act_write, int warp, int lane, int tid,
                                            const void* plane_in, void* plane_out) {
#pragma unroll 1
  for (int r = 0; r < BP::kStateRounds; ++r) {
    const int a0 = r * BP::kStatePages;

    if constexpr (Load) {
      rola::static_for<BP::kStatePages>([&](auto Ic) {
        constexpr int i = decltype(Ic)::value;
        const int a = a0 + i;
        if (a >= BP::kAtoms || slots[a] < 0) return;
        if ((((act_read | act_write) >> a) & 1u) == 0u) return;
        rola::state_page::load<BP::kDv, BP::kLdO, BP::kThreads>(
            sm.page_image(i), plane_in, (uint32_t)slots[a] * BP::Page::kBytes, tid,
            ((act_write >> a) & 1u) != 0u);
      });
      ops::rendezvous();
    }

    rola::static_for<BP::kNL>([&](auto Jc) {
      constexpr int j = decltype(Jc)::value;
      rola::static_for<2>([&](auto Ec) {
        constexpr int e = decltype(Ec)::value;
        const int P = j * ops::kMmaN + 2 * (lane & 3) + e;
        const int local = slot_local<D, BP::kBC, NS, WA>(p.g.w, warp, P);
        const int a = local >> kAtomLeavesBits;
        const int row = local & (kAtomLeaves - 1);
        if (a < a0 || a >= a0 + BP::kStatePages) return;
        if (slots[a] < 0) return;
        const uint32_t live = Load ? (act_read | act_write) : act_write;
        if (((live >> a) & 1u) == 0u) return;
        float* const img = sm.page_image(a - a0) + row * BP::kLdO;
        rola::static_for<BP::kMCh>([&](auto Mc) {
          constexpr int mc = decltype(Mc)::value;
          rola::static_for<2>([&](auto Hc) {
            constexpr int h = decltype(Hc)::value;
            const int ch = mc * ops::kMmaM + (lane >> 2) + 8 * h;
            if (ch >= BP::kCols) return;
            if constexpr (Load) {
              acc.c[mc][j][e + 2 * h] = img[ch];
            } else {
              ops::store_f32(img + ch, acc.c[mc][j][e + 2 * h]);
            }
          });
        });
      });
    });

    if constexpr (!Load) {
      ops::rendezvous();
      rola::static_for<BP::kStatePages>([&](auto Ic) {
        constexpr int i = decltype(Ic)::value;
        const int a = a0 + i;
        if (a >= BP::kAtoms || slots[a] < 0) return;
        if (((act_write >> a) & 1u) == 0u) return;
        rola::state_page::store<BP::kDv, BP::kLdO, BP::kThreads>(
            sm.page_image(i), plane_out, (uint32_t)slots[a] * BP::Page::kBytes, tid);
      });
    }
    ops::rendezvous();
  }
}

// ------------------------------------------------------------------------- the kernel

template <int D, int DV, int WARPS>
__global__ __launch_bounds__(BoxPlan<D, DV, WARPS>::kThreads, 1) void carry_kernel(
    __grid_constant__ const CarryParams p) {
  using BP = BoxPlan<D, DV, WARPS>;
  constexpr int NS = BP::kStreams;
  constexpr int BC = BP::kBC;
  static_assert(owner_span_bits(D, BC, D - 1) >= kAtomLeavesBits,
                "the innermost level holds a page whole, so the owner-local index splits "
                "into a page and a row at the page rectangle");

  extern __shared__ __align__(16) char smem_raw[];
  const Smem<BP> sm{smem_raw};

  const int owner = (int)blockIdx.x;
  const int bh = (int)blockIdx.y;
  const int tid = (int)threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;

  //: THE BOX'S TWO CALL-LEVEL BITS, and the die-fast they gate -- before any shared touch.
  const uint32_t atom_base = (uint32_t)bh * (uint32_t)(p.g.leaves / kAtomLeaves);
  uint32_t act_read = 0u, act_write = 0u;
  rola::static_for<BP::kAtoms>([&](auto Ac) {
    constexpr int a = decltype(Ac)::value;
    const uint32_t canon =
        (uint32_t)(geom_canon_leaf_t<D, BC>(p.g, owner, a * kAtomLeaves)) >> kAtomLeavesBits;
    const uint8_t bits = ops::atom_activity(p.activity, atom_base + canon);
    act_read |= (uint32_t)((bits & ops::kAtomRead) != 0u) << a;
    act_write |= (uint32_t)((bits & ops::kAtomWritten) != 0u) << a;
  });
  if ((act_read | act_write) == 0u) return;

  //: THE THREE EDGES, and the box's page slots -- the only page-table reads in the launch.
  //: ONE thread initialises every edge: a barrier initialised inside a divergent region
  //: by the thread whose index names it is a hazard the sanitizer refuses by name.
  if (tid == 0) {
    rola::static_for<BP::kEdges>(
        [&](auto Ec) { ops::edge_init(sm.edge((Edge) decltype(Ec)::value), BP::kThreads); });
    rola::static_for<BP::kStreamEdges>([&](auto Ec) {
      ops::edge_init(sm.stream_edge((StreamEdge) decltype(Ec)::value), BP::kThreads);
    });
    rola::static_for<kFillPoints>([&](auto Fc) { *sm.claim((Fill) decltype(Fc)::value) = 0u; });
  }
  if (tid < BP::kAtoms) {
    const uint32_t canon =
        (uint32_t)(geom_canon_leaf_t<D, BC>(p.g, owner, tid * kAtomLeaves)) >> kAtomLeavesBits;
    sm.page_slots()[tid] = ops::page_slot(p.page_table, atom_base + canon);
  }
  ops::rendezvous();
  const int32_t* const slots = sm.page_slots();

  //: THE WRITE SIDE'S WARP SUB-BOX, selected ONCE from the generated member set.
  rola::uniform_switch<sub_boxes(D, BC, NS).count>(p.g.w.assign, [&](auto Wa) {
    constexpr int WA = decltype(Wa)::value;

    Acc<BP> acc;
    rola::static_for<BP::kMCh>([&](auto Mc) {
      constexpr int mc = decltype(Mc)::value;
      rola::static_for<BP::kNL>([&](auto Jc) {
        constexpr int j = decltype(Jc)::value;
        rola::static_for<4>([&](auto Kc) { acc.c[mc][j][decltype(Kc)::value] = 0.0f; });
      });
    });

    if (p.state_in != nullptr)
      state_sweep<BP, D, NS, WA, true>(p, sm, acc, slots, act_read, act_write, warp, lane, tid,
                                       p.state_in, nullptr);

    //: THE BOX'S FACTOR RUNS: where each level's staged run begins in the packed plane,
    //: and where this lane's leaves sit inside the staged row. Both are walked ONCE here
    //: and never again -- the fill reads the first, the expansion the second.
    int abase[kGeomMaxLevels];
    rola::static_for<D>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      abase[l] = p.g.col_base[l] + geom_digit<D, BC>(p.g, owner, 0, l);
    });
    rola::static_for<kGeomMaxLevels - D>([&](auto Lc) { abase[D + decltype(Lc)::value] = 0; });

    int scol[BP::kNL][kGeomMaxLevels];
    rola::static_for<BP::kNL>([&](auto Jc) {
      constexpr int j = decltype(Jc)::value;
      const int P = j * ops::kMmaN + (lane >> 2);
      const int local = slot_local<D, BC, NS, WA>(p.g.w, warp, P);
      rola::static_for<D>([&](auto Lc) {
        constexpr int l = decltype(Lc)::value;
        scol[j][l] = span_base(D, BC, l)
                     + ((local >> owner_run_shift(D, BC, l)) & (owner_span(D, BC, l) - 1));
      });
      rola::static_for<kGeomMaxLevels - D>([&](auto Lc) { scol[j][D + decltype(Lc)::value] = 0; });
    });

    //: THE READ SIDE'S WARP SUB-BOX, and the two phases' forms -- all outside the loop.
    rola::uniform_switch<sub_boxes(D, BC, NS).count>(p.g.r.assign, [&](auto Ra) {
      window_loop<BP, D, decltype(Ra)::value, WA>(p, sm, acc, scol, abase, tid, warp, lane, bh);
    });

    if (p.state_out != nullptr)
      state_sweep<BP, D, NS, WA, false>(p, sm, acc, slots, act_read, act_write, warp, lane, tid,
                                        nullptr, p.state_out);
  });
}

}  // namespace rola::carry
