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
#include "facts/liveness_contract.cuh"
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

  //: THE TRANSIT REGION, `[row = sb_swizzle(local)][col = channel]`, and PHASE 1's output
  //: blocks in the bytes the pull has finished with. -- see docs/internals/carry/smem_ledger.md#region
  __device__ __forceinline__ char* region() const { return base + BP::kRegionOffset; }

  __device__ __forceinline__ float* panel(int warp) const {
    return reinterpret_cast<float*>(base + BP::kRegionOffset + warp * BP::kPanelSliceBytes);
  }

  __device__ __forceinline__ ops::SmemAddr v_tile(int slot) const {
    return ops::smem_addr(base + BP::kVTileOffset + slot * BP::kVTileBytes);
  }

  __device__ __forceinline__ char* column_block(int slot) const {
    return base + BP::kColOffset + slot * (BP::kColBytes / BP::kColSlots);
  }

  __device__ __forceinline__ char* raw_slot(int slot) const {
    return base + BP::kRawOffset + slot * BP::kRawSlotBytes;
  }

  __device__ __forceinline__ char* read_raw_slot(int slot) const {
    return base + BP::kReadRawOffset + slot * BP::kRawSlotBytes;
  }

  __device__ __forceinline__ __nv_bfloat16* gain_slot(int slot) const {
    return reinterpret_cast<__nv_bfloat16*>(base + BP::kGainOffset + slot * BP::kGainSlotBytes);
  }

  __device__ __forceinline__ unsigned int* claim() const {
    return reinterpret_cast<unsigned int*>(base + BP::kClaimOffset);
  }

  __device__ __forceinline__ int32_t* page_slots() const {
    return reinterpret_cast<int32_t*>(base + BP::kPageSlotOffset);
  }

  //: A LIVE STREAM'S PRIVATE RING. The fold's carries a value tile, the factor rows it is
  //: folded against and the segment's gains; the readout's carries the factor rows and the
  //: WINDOW-RELATIVE token ids its deposits are addressed by. One slot each: the register
  //: fragments are the second buffer. -- see docs/internals/carry/smem_ledger.md#streams
  __device__ __forceinline__ char* fold_ring(int s) const {
    return base + BP::kSparseFoldOffset + s * BP::kFoldRingBytes;
  }

  __device__ __forceinline__ ops::SmemAddr fold_values(int s) const {
    return ops::smem_addr(fold_ring(s));
  }

  __device__ __forceinline__ char* fold_factors(int s) const {
    return fold_ring(s) + BP::kVTileBytes;
  }

  __device__ __forceinline__ unsigned short* fold_gains(int s) const {
    return reinterpret_cast<unsigned short*>(fold_ring(s) + BP::kVTileBytes + BP::kRawSlotBytes);
  }

  __device__ __forceinline__ char* read_ring(int s) const {
    return base + BP::kSparseReadOffset + s * BP::kReadRingBytes;
  }

  __device__ __forceinline__ unsigned short* read_tokens(int s) const {
    return reinterpret_cast<unsigned short*>(read_ring(s) + BP::kRawSlotBytes);
  }

  //: THE STREAM'S TWO COUNTERS, in TOKENS: `numProduced` claims a placement, `numConsumed`
  //: releases the slot. -- see docs/internals/carry/carry_kernel.md#counters
  __device__ __forceinline__ unsigned int* produced(int s) const {
    return reinterpret_cast<unsigned int*>(base + BP::kCounterOffset) + 2 * s;
  }

  __device__ __forceinline__ unsigned int* consumed(int s) const { return produced(s) + 1; }
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

// ------------------------------------------------------------------------ the fill

//: ONE CLAIM on the stream's cursor. The cursor is MONOTONE and never reset: a warp
//: leaves a fill with exactly one failing claim, so every production point advances it by
//: its unit count plus one per warp and the next point knows its own base.
//: -- see docs/internals/carry/carry_kernel.md#fill
template <class BP>
__device__ __forceinline__ uint32_t claim_unit(const Smem<BP>& sm, int lane) {
  unsigned int u = 0u;
  if (lane == 0) u = atomicAdd(sm.claim(), 1u);
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
//: the box's own digits, at the widest quantum each run admits. Both sides stage through
//: this: the runs a box reaches are the BOX's, so only the amplitude plane and the ring
//: differ. -- see docs/internals/carry/carry_kernel.md#factor-rows
template <class BP, int D>
__device__ __forceinline__ void stage_unit_factors(const CarryParams& p,
                                                   const __nv_bfloat16* amplitudes,
                                                   ops::SmemAddr dst0, const int (&abase)[D],
                                                   int lane, int bh, int seg0, int tok_end,
                                                   int unit) {
  const uint32_t plane = (uint32_t)bh * (uint32_t)p.L * (uint32_t)p.g.wtot * 2u;
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
                            ops::gather_ptr(amplitudes, off), tok < tok_end);
    });
  });
}

// ------------------------------------------------- the transit and PHASE 1's readout

//: THE STATE IN TRANSIT: the accumulator's halves while the read side owns them. `sa`/`ma`
//: are the readout's A operand -- the top halves, re-owned by the read carve -- and `lo`/
//: `mlo` are the bottom halves, which never leave the register file and are never
//: truncated. -- see docs/internals/carry/carry_kernel.md#transit
template <class BP>
struct Transit {
  uint32_t sa[BP::kDw][BP::kMassTile][4];
  uint32_t ma[BP::kDw][4];
  uint32_t lo[BP::kMassTile][BP::kNL][2];
  uint32_t mlo[BP::kNL];
};

//: A SIDE'S REGION ROW, as a shared address. The row IS the owner-local canonical index
//: under the region's swizzle -- one law, both sides.
//: -- see docs/internals/carry/carry_kernel.md#transit
template <class BP, int D, int BC>
__device__ __forceinline__ ops::SmemAddr transit_row(const Smem<BP>& sm, const SideGeom& s,
                                                     int strm, int P) {
  return ops::smem_addr(sm.region())
         + (uint32_t)(sb_swizzle(geom_local_row<D, BC>(s, strm, P)) * BP::kLdX * 2);
}

//: PUBLISH: the accumulator's top halves into the transit region, in the write carve's
//: order, and the bottom halves packed into the registers that carry them through PHASE 1.
//: The mass channel opens the last m-tile, so its row is written by the lane quad that
//: holds it and every other element of that tile is structurally zero.
template <class BP, int D, int BC>
__device__ __forceinline__ void out_leg_publish(const CarryParams& p, const Smem<BP>& sm,
                                                const Acc<BP>& acc, Transit<BP>& tr, int warp,
                                                int lane) {
  const uint32_t col = (uint32_t)((lane >> 2) * 2);
  rola::static_for<BP::kNL>([&](auto Jc) {
    constexpr int j = decltype(Jc)::value;
    rola::static_for<2>([&](auto Ec) {
      constexpr int e = decltype(Ec)::value;
      const ops::SmemAddr row =
          transit_row<BP, D, BC>(sm, p.g.w, warp, j * ops::kMmaN + 2 * (lane & 3) + e);
      rola::static_for<BP::kMassTile>([&](auto Mc) {
        constexpr int m = decltype(Mc)::value;
        rola::static_for<2>([&](auto Hc) {
          constexpr int h = decltype(Hc)::value;
          ops::store_shared_u16(row + col + (uint32_t)((m * ops::kMmaM + 8 * h) * 2),
                                ops::hi_bf16(acc.c[m][j][e + 2 * h]));
        });
      });
      if ((lane >> 2) == 0)
        ops::store_shared_u16(row + (uint32_t)(BP::kDv * 2),
                              ops::hi_bf16(acc.c[BP::kMassTile][j][e]));
    });
    rola::static_for<BP::kMassTile>([&](auto Mc) {
      constexpr int m = decltype(Mc)::value;
      tr.lo[m][j][0] = ops::pack_lo(acc.c[m][j][0], acc.c[m][j][1]);
      tr.lo[m][j][1] = ops::pack_lo(acc.c[m][j][2], acc.c[m][j][3]);
    });
    tr.mlo[j] = ops::pack_lo(acc.c[BP::kMassTile][j][0], acc.c[BP::kMassTile][j][1]);
  });
}

//: COVER SLOT A: the read side's first batches, claimed and staged while the out leg's edge
//: settles. It reads routing only -- never the accumulator and never the region -- which is
//: what makes it independent of the phase it covers. The EXPANSION cannot ride here: a
//: claimed fill's product is visible only after a CTA edge, and the only edge in reach is
//: the one this slot sits in front of.
template <class BP, int D, int Segs>
__device__ __forceinline__ void readout_stage(const CarryParams& p, const Smem<BP>& sm,
                                              const int (&abase)[D], int lane, int bh, int seg0,
                                              int tok_end, int slot, uint32_t& base) {
  constexpr int kPerSeg = BP::kSeg / kUnitTokens;
  constexpr int kUnits = Segs * kPerSeg;
  for (;;) {
    const uint32_t u = claim_unit<BP>(sm, lane);
    if (u >= base + kUnits) break;
    const int unit = (int)(u - base), k = unit / kPerSeg, su = unit % kPerSeg;
    stage_unit_factors<BP, D>(p, p.pread,
                              ops::smem_addr(sm.read_raw_slot((slot + k) % BP::kReadSlots)), abase,
                              lane, bh, seg0 + k * BP::kSeg, tok_end, su);
  }
  base += (uint32_t)(kUnits + BP::kWarps);
  ops::stage_commit();
}

template <class BP, int D, int BC>
__device__ __forceinline__ void cover_read_batch_zero(const CarryParams& p, const Smem<BP>& sm,
                                                      const int (&abase)[D], int lane, int bh,
                                                      int t0, int wLen, uint32_t& claim) {
  readout_stage<BP, D, BP::kReadSlots>(p, sm, abase, lane, bh, t0, t0 + wLen, 0, claim);
  ops::stage_wait<0>();
}

//: PULL: the region's rows back as the read side's A fragments. `ldmatrix.trans` reads the
//: region as it lies -- row = leaf, column = channel -- and returns `S^T`, m = channel and
//: k = leaf, which is the A operand the readout wants. The mass is scalar: an `x4` at its
//: m-tile would address channels the row does not have.
template <class BP, int D, int BC>
__device__ __forceinline__ void out_leg_pull(const CarryParams& p, const Smem<BP>& sm,
                                             Transit<BP>& tr, int warp, int lane) {
  const int rT = ((lane >> 4) & 1) * 8 + (lane & 7), cT = (lane >> 3) & 1;
  rola::static_for<BP::kDw>([&](auto Kc) {
    constexpr int kk = decltype(Kc)::value;
    const ops::SmemAddr row =
        transit_row<BP, D, BC>(sm, p.g.r, warp, kk * ops::kMmaK + rT) + (uint32_t)(cT * 8 * 2);
    rola::static_for<BP::kMassTile>([&](auto Mc) {
      constexpr int m = decltype(Mc)::value;
      ops::load_frag_t(tr.sa[kk][m], row + (uint32_t)(m * ops::kMmaM * 2));
    });
    rola::static_for<4>([&](auto Ic) { tr.ma[kk][decltype(Ic)::value] = 0u; });
    if ((lane >> 2) == 0) {
      rola::static_for<2>([&](auto Hc) {
        constexpr int h = decltype(Hc)::value;
        const int P = kk * ops::kMmaK + 8 * h + 2 * (lane & 3);
        const uint32_t c = (uint32_t)(BP::kDv * 2);
        tr.ma[kk][2 * h] =
            (uint32_t)ops::load_shared_u16(transit_row<BP, D, BC>(sm, p.g.r, warp, P) + c)
            | ((uint32_t)ops::load_shared_u16(transit_row<BP, D, BC>(sm, p.g.r, warp, P + 1) + c)
               << 16);
      });
    }
  });
}

//: PRODUCE: one read batch's coefficients, expanded from the staged factor rows AS THEY
//: LIE. The MMA's B fragment pairs two adjacent LEAVES in one register at a FIXED token,
//: and two adjacent leaves are two adjacent amplitude columns of that token's row -- so the
//: read side has no transpose and no column block. Every level above the innermost has a
//: sub-box shift of at least the page rectangle, so its column depends on the k-step alone
//: and is memoized there. -- see docs/internals/carry/carry_kernel.md#read-expansion
template <class BP, int D>
__device__ __forceinline__ void readout_produce(const Smem<BP>& sm, int slot, int lane,
                                                const int (&rcin)[BP::kDw][2],
                                                const int (&rcout)[D][BP::kDw],
                                                uint32_t (&rb)[BP::kDw][BP::kTokTiles][2]) {
  const ops::SmemAddr row0 =
      ops::smem_addr(sm.read_raw_slot(slot)) + (uint32_t)((lane >> 2) * BP::kRawStride * 2);
  rola::static_for<BP::kTokTiles>([&](auto Tc) {
    constexpr int t = decltype(Tc)::value;
    const ops::SmemAddr row = row0 + (uint32_t)(t * ops::kMmaN * BP::kRawStride * 2);
    rola::static_for<BP::kDw>([&](auto Kc) {
      constexpr int kk = decltype(Kc)::value;
      rola::static_for<2>([&](auto Hc) {
        constexpr int h = decltype(Hc)::value;
        uint32_t c = ops::load_shared_u32(row + (uint32_t)(rcin[kk][h] * 2));
        rola::static_for<D - 1>([&](auto Lc) {
          constexpr int l = decltype(Lc)::value;
          c = ops::mul_bf16x2(
              c, ops::splat_bf16x2_bits(ops::load_shared_u16(row + (uint32_t)(rcout[l][kk] * 2))));
        });
        rb[kk][t][h] = c;
      });
    });
  });
}

//: MMA: one read batch against the pulled A fragments. `kDw` k-steps by `kTokTiles` token
//: n-tiles by the channel m-tiles, and the panel is dead at the end of the batch.
template <class BP>
__device__ __forceinline__ void readout_mma(const Transit<BP>& tr,
                                            const uint32_t (&rb)[BP::kDw][BP::kTokTiles][2],
                                            float (&po)[BP::kMassTile][BP::kTokTiles][4],
                                            float (&pm)[BP::kTokTiles][4]) {
  rola::static_for<BP::kTokTiles>([&](auto Tc) {
    constexpr int t = decltype(Tc)::value;
    rola::static_for<4>([&](auto Ec) {
      constexpr int e = decltype(Ec)::value;
      rola::static_for<BP::kMassTile>([&](auto Mc) { po[decltype(Mc)::value][t][e] = 0.0f; });
      pm[t][e] = 0.0f;
    });
    rola::static_for<BP::kDw>([&](auto Kc) {
      constexpr int kk = decltype(Kc)::value;
      rola::static_for<BP::kMassTile>([&](auto Mc) {
        constexpr int m = decltype(Mc)::value;
        ops::mma(po[m][t], tr.sa[kk][m], rb[kk][t]);
      });
      ops::mma(pm[t], tr.ma[kk], rb[kk][t]);
    });
  });
}

//: DEPOSIT: one panel round of this warp's readout accumulator into the warp's OWN slice.
//: The warp writes its slice WHOLE, so there is nothing to clear and no atom in the path:
//: an accumulator element belongs to exactly one lane of one warp.
//: -- see docs/internals/carry/carry_kernel.md#fan-in
template <class BP>
__device__ __forceinline__ void readout_deposit(const Smem<BP>& sm, int warp, int lane, int round,
                                                const float (&po)[BP::kMassTile][BP::kTokTiles][4],
                                                const float (&pm)[BP::kTokTiles][4]) {
  float* const slice = sm.panel(warp);
  rola::static_for<BP::kPanelTiles>([&](auto Tc) {
    constexpr int t = decltype(Tc)::value;
    const int tile = round * BP::kPanelTiles + t;
    rola::static_for<4>([&](auto Ec) {
      constexpr int e = decltype(Ec)::value;
      float* const row = slice + (t * ops::kMmaN + 2 * (lane & 3) + (e & 1)) * BP::kLdO;
      rola::static_for<BP::kMassTile>([&](auto Mc) {
        constexpr int m = decltype(Mc)::value;
        ops::store_f32(row + m * ops::kMmaM + 8 * (e >> 1) + (lane >> 2), po[m][tile][e]);
      });
      if constexpr ((e >> 1) == 0)
        if ((lane >> 2) == 0) ops::store_f32(row + BP::kDv, pm[tile][e]);
    });
  });
}

//: FAN-IN: the CTA sums its warps' slices and reduces ONE contribution per owner box per
//: token into `num` and `den`. The plane's atom is native and the panel's is none: the
//: warps meet by writing disjoint cells, not by contending for one.
//: A token past the window's end carries a zero coefficient by construction but has no cell
//: in the planes, so this is the one place in the readout that tests a token against the
//: window's length. -- see docs/internals/carry/carry_kernel.md#fan-in
template <class BP>
__device__ __forceinline__ void readout_fan_in(const Smem<BP>& sm, float* num, float* den,
                                               int round, int tid, int tok0, int tok_end) {
  constexpr int kCells = BP::kPanelTokens * BP::kDv;
  const int base = tok0 + round * BP::kPanelTokens;
  rola::static_for<(kCells + BP::kThreads - 1) / BP::kThreads>([&](auto Ic) {
    const int idx = decltype(Ic)::value * BP::kThreads + tid;
    if (kCells % BP::kThreads != 0 && idx >= kCells) return;
    const int row = idx / BP::kDv, col = idx % BP::kDv;
    if (base + row >= tok_end) return;
    float sum = 0.0f;
    rola::static_for<BP::kWarps>(
        [&](auto Wc) { sum += sm.panel(decltype(Wc)::value)[row * BP::kLdO + col]; });
    ops::fan_in_add(
        reinterpret_cast<float*>(ops::gather_ptr(
            num, (uint32_t)(base + row) * (uint32_t)(BP::kDv * 4) + (uint32_t)(col * 4))),
        sum);
  });
  if (tid < BP::kPanelTokens && base + tid < tok_end) {
    float sum = 0.0f;
    rola::static_for<BP::kWarps>(
        [&](auto Wc) { sum += sm.panel(decltype(Wc)::value)[tid * BP::kLdO + BP::kDv]; });
    ops::fan_in_add(reinterpret_cast<float*>(ops::gather_ptr(den, (uint32_t)(base + tid) * 4u)),
                    sum);
  }
}

//: STORE-BACK: the pull run backwards. The readout never modified the state, so the halves
//: the reader holds are the ones the writer needs, written at the read carve's rows.
template <class BP, int D, int BC>
__device__ __forceinline__ void return_leg_store(const CarryParams& p, const Smem<BP>& sm,
                                                 const Transit<BP>& tr, int warp, int lane) {
  const uint32_t col = (uint32_t)((lane >> 2) * 2);
  rola::static_for<BP::kDw>([&](auto Kc) {
    constexpr int kk = decltype(Kc)::value;
    rola::static_for<2>([&](auto Hc) {
      constexpr int h = decltype(Hc)::value;
      const int P = kk * ops::kMmaK + 8 * h + 2 * (lane & 3);
      const ops::SmemAddr r0 = transit_row<BP, D, BC>(sm, p.g.r, warp, P);
      const ops::SmemAddr r1 = transit_row<BP, D, BC>(sm, p.g.r, warp, P + 1);
      rola::static_for<BP::kMassTile>([&](auto Mc) {
        constexpr int m = decltype(Mc)::value;
        rola::static_for<2>([&](auto Ic) {
          constexpr int i = decltype(Ic)::value;
          const uint32_t v = tr.sa[kk][m][2 * h + i];
          const uint32_t o = col + (uint32_t)((m * ops::kMmaM + 8 * i) * 2);
          ops::store_shared_u16(r0 + o, (uint16_t)(v & 0xFFFFu));
          ops::store_shared_u16(r1 + o, (uint16_t)(v >> 16));
        });
      });
      if ((lane >> 2) == 0) {
        const uint32_t v = tr.ma[kk][2 * h];
        ops::store_shared_u16(r0 + (uint32_t)(BP::kDv * 2), (uint16_t)(v & 0xFFFFu));
        ops::store_shared_u16(r1 + (uint32_t)(BP::kDv * 2), (uint16_t)(v >> 16));
      }
    });
  });
}

//: REJOIN: the two halves back into the fp32 accumulator, exactly. One `ldmatrix.trans`
//: lands on four accumulator elements of two n-tiles at once -- the A operand's map and the
//: accumulator's map agree on the channel and compose on the leaf -- and `join_hi_lo` puts
//: each bf16 pair back over the low halves the register file kept.
template <class BP, int D, int BC>
__device__ __forceinline__ void return_leg_rejoin(const CarryParams& p, const Smem<BP>& sm,
                                                  const Transit<BP>& tr, Acc<BP>& acc, int warp,
                                                  int lane) {
  const int rT = ((lane >> 4) & 1) * 8 + (lane & 7), cT = (lane >> 3) & 1;
  rola::static_for<BP::kNL / 2>([&](auto Jc) {
    constexpr int jp = decltype(Jc)::value;
    const ops::SmemAddr row =
        transit_row<BP, D, BC>(sm, p.g.w, warp, jp * ops::kMmaK + rT) + (uint32_t)(cT * 8 * 2);
    rola::static_for<BP::kMassTile>([&](auto Mc) {
      constexpr int m = decltype(Mc)::value;
      uint32_t hi[4];
      ops::load_frag_t(hi, row + (uint32_t)(m * ops::kMmaM * 2));
      rola::static_for<2>([&](auto Hc) {
        constexpr int h = decltype(Hc)::value;
        rola::static_for<2>([&](auto Ic) {
          constexpr int i = decltype(Ic)::value;
          constexpr int j = 2 * jp + h;
          acc.c[m][j][2 * i] = ops::join_hi_lo(hi[2 * h + i], tr.lo[m][j][i], 0);
          acc.c[m][j][2 * i + 1] = ops::join_hi_lo(hi[2 * h + i], tr.lo[m][j][i], 1);
        });
      });
    });
  });
  //: the mass m-tile holds ONE live channel, at the lane quad the accumulator's map gives
  //: it; every other element of the tile is structurally zero (`kMassRow == 0` and the
  //: deposit's A fragment has a single live row), so the rejoin states that rather than
  //: carrying sixteen zero registers through PHASE 1.
  rola::static_for<BP::kNL>([&](auto Jc) {
    constexpr int j = decltype(Jc)::value;
    rola::static_for<4>([&](auto Ec) { acc.c[BP::kMassTile][j][decltype(Ec)::value] = 0.0f; });
    if ((lane >> 2) == 0) {
      const int P = j * ops::kMmaN + 2 * (lane & 3);
      const uint32_t c = (uint32_t)(BP::kDv * 2);
      const uint32_t hi =
          (uint32_t)ops::load_shared_u16(transit_row<BP, D, BC>(sm, p.g.w, warp, P) + c)
          | ((uint32_t)ops::load_shared_u16(transit_row<BP, D, BC>(sm, p.g.w, warp, P + 1) + c)
             << 16);
      acc.c[BP::kMassTile][j][0] = ops::join_hi_lo(hi, tr.mlo[j], 0);
      acc.c[BP::kMassTile][j][1] = ops::join_hi_lo(hi, tr.mlo[j], 1);
    }
  });
}

//: PHASE 1, THE UNION FORM: the four slots of a read batch, issued so each covers the next.
//: -- see docs/internals/carry/carry_kernel.md#readout
template <class BP, int D>
__device__ __forceinline__ void readout_phase(const CarryParams& p, const Smem<BP>& sm,
                                              const Transit<BP>& tr, const int (&abase)[D],
                                              const int (&rcin)[BP::kDw][2],
                                              const int (&rcout)[D][BP::kDw], float* num,
                                              float* den, int tid, int warp, int lane, int bh,
                                              int t0, int wLen, uint32_t& claim) {
  const int nSeg = (wLen + BP::kSeg - 1) / BP::kSeg;
  const int tok_end = t0 + wLen;

  //: the region's bytes pass from the pull to the panels here, and the batches the cover
  //: staged become visible in the same crossing.
  ops::rendezvous_group((int)StreamEdge::kReadBatch, BP::kThreads);

  uint32_t rb[BP::kDw][BP::kTokTiles][2];
  readout_produce<BP, D>(sm, 0, lane, rcin, rcout, rb);

#pragma unroll 1
  for (int b = 0; b < nSeg; ++b) {
    readout_stage<BP, D, 1>(p, sm, abase, lane, bh, t0 + (b + BP::kReadSlots) * BP::kSeg, tok_end,
                            b % BP::kReadSlots, claim);

    float po[BP::kMassTile][BP::kTokTiles][4], pm[BP::kTokTiles][4];
    readout_mma<BP>(tr, rb, po, pm);

    uint32_t rn[BP::kDw][BP::kTokTiles][2];
    readout_produce<BP, D>(sm, (b + 1) % BP::kReadSlots, lane, rcin, rcout, rn);

    rola::static_for<BP::kPanelRounds>([&](auto Rc) {
      constexpr int r = decltype(Rc)::value;
      if constexpr (r > 0) ops::rendezvous_group((int)StreamEdge::kReadBatch, BP::kThreads);
      readout_deposit<BP>(sm, warp, lane, r, po, pm);
      ops::rendezvous_group((int)StreamEdge::kReadBatch, BP::kThreads);
      readout_fan_in<BP>(sm, num, den, r, tid, t0 + b * BP::kSeg, tok_end);
    });

    rola::static_for<BP::kDw>([&](auto Kc) {
      constexpr int kk = decltype(Kc)::value;
      rola::static_for<BP::kTokTiles>([&](auto Tc) {
        constexpr int t = decltype(Tc)::value;
        rola::static_for<2>([&](auto Hc) {
          constexpr int h = decltype(Hc)::value;
          rb[kk][t][h] = rn[kk][t][h];
        });
      });
    });

    ops::stage_wait<0>();
    ops::rendezvous_group((int)StreamEdge::kReadBatch, BP::kThreads);
  }
}

// ------------------------------------------------------------------- PHASE 2's fold

//: PRODUCE A RUN OF SEGMENTS: every warp claims units until the run is taken, so the fill
//: shares out by arrival rather than by assignment. The run is ONE production point --
//: two of them on one cursor would need an edge between, and a cover slot has none.
//: -- see docs/internals/carry/carry_kernel.md#fill
template <class BP, int D, int Segs>
__device__ __forceinline__ void fold_produce(const CarryParams& p, const Smem<BP>& sm,
                                             const int (&abase)[D], int lane, int bh, int seg_v,
                                             int seg_f, int tok_end, int vslot, int fslot,
                                             bool values, bool factors, uint32_t& base) {
  constexpr int kPerSeg = BP::kSeg / kUnitTokens;
  constexpr int kUnits = Segs * kPerSeg;
  for (;;) {
    const uint32_t u = claim_unit<BP>(sm, lane);
    if (u >= base + kUnits) break;
    const int unit = (int)(u - base), k = unit / kPerSeg, su = unit % kPerSeg;
    if (values)
      stage_unit_values<BP>(p, sm, (vslot + k) % BP::kSegmentSlots, lane, bh, seg_v + k * BP::kSeg,
                            tok_end, su);
    if (factors)
      stage_unit_factors<BP, D>(p, p.pwrite,
                                ops::smem_addr(sm.raw_slot((fslot + k) % BP::kFactorSlots)), abase,
                                lane, bh, seg_f + k * BP::kSeg, tok_end, su);
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
__device__ __forceinline__ void fold_transpose(const Smem<BP>& sm, int raw, int gain, int col,
                                               int tid) {
  constexpr int kPairs = BP::kSpanTotal / 2;
  constexpr int kRows = BP::kSeg / 2;
  constexpr int kUnits = kPairs * kRows;
  constexpr int kGainPairs = owner_span(D, BP::kBC, 0) / 2;
  const uint32_t* const src = reinterpret_cast<const uint32_t*>(sm.raw_slot(raw));
  uint32_t* const dst = reinterpret_cast<uint32_t*>(sm.column_block(col));
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

//: EXPAND: one segment's coefficients for this lane's leaves, in registers. A level's
//: column is loaded ONCE PER DISTINCT J-SLICE rather than once per tile -- at the flagship
//: member that is two columns a level instead of four -- and each level contributes one
//: packed multiply per (leaf tile, token pair).
//: -- see docs/internals/carry/carry_kernel.md#expansion
template <class BP, int D, int NS, int WA>
__device__ __forceinline__ void fold_expand(const Smem<BP>& sm, uint32_t (&bf)[BP::kNL][2],
                                            const int (&scol)[D][BP::kNL], int slot, int lane) {
  const uint32_t* const col = reinterpret_cast<const uint32_t*>(sm.column_block(slot));
  const int t2 = lane & 3;
  rola::static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    constexpr int kSh = slice_shift(sb_shift_of(D, BP::kBC, NS, WA, l));
    constexpr int kBits =
        slice_bits(sb_shift_of(D, BP::kBC, NS, WA, l), sb_span_bits(D, BP::kBC, NS, WA, l));
    uint32_t f[1 << kBits][2];
    rola::static_for<1 << kBits>([&](auto Vc) {
      constexpr int v = decltype(Vc)::value;
      rola::static_for<2>([&](auto Hc) {
        constexpr int h = decltype(Hc)::value;
        f[v][h] = col[scol[l][v] * (BP::kTokStride / 2) + t2 + 4 * h];
      });
    });
    rola::static_for<BP::kNL>([&](auto Jc) {
      constexpr int v = (decltype(Jc)::value >> kSh) & ((1 << kBits) - 1);
      rola::static_for<2>([&](auto Hc) {
        constexpr int j = decltype(Jc)::value, h = decltype(Hc)::value;
        bf[j][h] = l == 0 ? f[v][h] : ops::mul_bf16x2(bf[j][h], f[v][h]);
      });
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
__device__ __forceinline__ void fold_drain(ops::SmemAddr tile, uint32_t (&vf)[BP::kMCh][4],
                                           int lane) {
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
      ops::load_frag_t(vf[mc], tile + (uint32_t)v_group_offset<BP>(row, group));
    }
  });
}

//: THE SAME DRAIN WITHOUT THE MASS TILE, for a caller that builds the constant fragment at
//: its deposits rather than carrying it. -- see docs/internals/carry/carry_kernel.md#streams
template <class BP>
__device__ __forceinline__ void fold_drain_values(ops::SmemAddr tile,
                                                  uint32_t (&vf)[BP::kMassTile][4], int lane) {
  const int row = (lane & 7) + ((lane >> 4) & 1) * 8;
  rola::static_for<BP::kMassTile>([&](auto Mc) {
    constexpr int mc = decltype(Mc)::value;
    ops::load_frag_t(vf[mc], tile + (uint32_t)v_group_offset<BP>(row, mc * 2 + ((lane >> 3) & 1)));
  });
}

//: COVER SLOT B: the write side's first segments' ROUTING data, walked and staged while
//: the return leg's edge settles. It reads routing only -- never the accumulator and never
//: the region -- which is what makes it independent of the phase it covers.
template <class BP, int D>
__device__ __forceinline__ void cover_write_batch_zero(const CarryParams& p, const Smem<BP>& sm,
                                                       const int (&abase)[D], int warp, int lane,
                                                       int bh, int t0, int wLen, uint32_t& claim) {
  const int tok_end = t0 + wLen;
  fold_produce<BP, D, BP::kFactorSlots>(p, sm, abase, lane, bh, t0, t0, tok_end, 0, 0, false, true,
                                        claim);
  if (warp == 0)
    rola::static_for<BP::kFactorSlots>([&](auto Ic) {
      constexpr int s = decltype(Ic)::value;
      write_gain<BP>(sm, s, lane, read_gain<BP>(p, lane, bh, t0 + s * BP::kSeg, tok_end));
    });
}

//: PHASE 2, THE UNION FORM. The fold is WINDOW-BLIND: a deposit never reads the accumulator, so
//: the segment MMAs of a whole window chain with no accumulator dependency and the
//: segment slot's refill is covered by the previous segment's chain.
//: -- see docs/internals/carry/carry_kernel.md#fold
template <class BP, int D, int NS, int WA>
__device__ __forceinline__ void fold_phase(const CarryParams& p, const Smem<BP>& sm, Acc<BP>& acc,
                                           const int (&scol)[D][BP::kNL], const int (&abase)[D],
                                           int tid, int warp, int lane, int bh, int t0, int wLen,
                                           uint32_t& claim) {
  constexpr int S = BP::kSegmentSlots;
  constexpr int Fs = BP::kFactorSlots;
  constexpr int Cs = BP::kColSlots;
  const int nSeg = (wLen + BP::kSeg - 1) / BP::kSeg;
  const int tok_end = t0 + wLen;

  //: the value rows of the first slots; the factor rows they are folded against are
  //: already in the ring, staged by the cover slot the return leg's edge hid.
  fold_produce<BP, D, S>(p, sm, abase, lane, bh, t0, t0, tok_end, 0, 0, true, false, claim);

  ops::stage_wait<S - 1>();
  ops::rendezvous_group((int)StreamEdge::kFilled, BP::kThreads);
  fold_transpose<BP, D>(sm, 0, 0, 0, tid);
  ops::rendezvous_group((int)StreamEdge::kEmpty, BP::kThreads);

  uint32_t bf[BP::kNL][2];
  fold_expand<BP, D, NS, WA>(sm, bf, scol, 0, lane);

#pragma unroll 1
  for (int b = 0; b < nSeg; ++b) {
    const int ahead = b + S, next = b + 1;
    const unsigned short gain =
        warp == 0 ? read_gain<BP>(p, lane, bh, t0 + (ahead + 1) * BP::kSeg, tok_end) : 0u;

    uint32_t vf[BP::kMCh][4];
    fold_drain<BP>(sm.v_tile(b % S), vf, lane);
    if (next < nSeg) fold_transpose<BP, D>(sm, next % Fs, next % Fs, next % Cs, tid);

    ops::rendezvous_group((int)StreamEdge::kEmpty, BP::kThreads);

    //: the segment `S` ahead, and the factor rows one further still: the expansion runs a
    //: segment in front of the deposits, so its rows must land a segment earlier. A segment
    //: past the window's end stages zeros, which is what a zero coefficient already is.
    fold_produce<BP, D, 1>(p, sm, abase, lane, bh, t0 + ahead * BP::kSeg,
                           t0 + (ahead + 1) * BP::kSeg, tok_end, ahead % S, (ahead + 1) % Fs, true,
                           true, claim);

    uint32_t bnext[BP::kNL][2];
    if (next < nSeg) fold_expand<BP, D, NS, WA>(sm, bnext, scol, next % Cs, lane);

    //: DEPOSIT: one MMA per (channel m-tile, leaf n-tile), the segment as the k step.
    rola::static_for<BP::kMCh>([&](auto Mc) {
      constexpr int mc = decltype(Mc)::value;
      rola::static_for<BP::kNL>([&](auto Jc) {
        constexpr int j = decltype(Jc)::value;
        ops::mma(acc.c[mc][j], vf[mc], bf[j]);
      });
    });

    if (warp == 0) write_gain<BP>(sm, (ahead + 1) % Fs, lane, gain);
    rola::static_for<BP::kNL>([&](auto Jc) {
      constexpr int j = decltype(Jc)::value;
      rola::static_for<2>([&](auto Hc) {
        constexpr int h = decltype(Hc)::value;
        bf[j][h] = bnext[j][h];
      });
    });

    ops::stage_wait<S - 1>();
    ops::rendezvous_group((int)StreamEdge::kFilled, BP::kThreads);
  }
  ops::stage_wait<0>();
}

// ------------------------------------------------------- the streams, both phases

//: THE SENTINEL a slot the window did not fill carries: no token, so no deposit.
constexpr int kDeadToken = 0xFFFF;

//: THE CHANNEL M-TILES the sparse readout carries between deposits: the direct scatter frees
//: it from holding a whole [channels x tokens] partial.
constexpr int kScatterTiles = 1;

//: THE LIVE-TOKEN MASKS of one stream over one window, ONE PER LANE: the AND over levels of
//: the OR over the run of digits the stream's leaf range covers there.
//: -- see docs/internals/carry/carry_kernel.md#streams
//: `abase[l]` is the level's row base plus the OWNER's digit, so a stream's liveness row is
//: `abase[l]` plus the leaf range's own digit and the owner index never enters a window.
template <class BP, int D, int BC>
__device__ __forceinline__ uint32_t stream_masks(const CarryParams& p, const SideGeom& sg, int side,
                                                 const int (&abase)[D], int warp, int lane, int bh,
                                                 int P0, int gbits, int t0, int nWords) {
  const int local = geom_local_row<D, BC>(sg, warp, P0);
  const int32_t* const words =
      p.liveness + ((long)bh * rola::facts::kSides + side) * ((long)p.g.wtot * p.liveness_words)
      + (t0 >> 5) + lane;
  uint32_t m = lane < nWords ? 0xFFFFFFFFu : 0u;
  rola::static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    const int room = gbits - sg.sub_shift[l];
    const int count = 1 << (room <= 0 ? 0 : (room < sg.sub_bits[l] ? room : sg.sub_bits[l]));
    const int digit = (local >> owner_run_shift(D, BC, l)) & (owner_span(D, BC, l) - 1);
    const int32_t* row = words + (long)(abase[l] + digit) * p.liveness_words;
    uint32_t o = 0u;
    if (lane < nWords) {
#pragma unroll 1
      for (int i = 0; i < count; ++i, row += p.liveness_words) o |= (uint32_t)*row;
    }
    m &= o;
  });
  return m;
}

//: ONE LIVE PAIR'S ROWS INTO A PRIVATE RING, BRANCHED AND NEVER PREDICATED: a predicated-off
//: `cp.async` still zero-fills its destination, and under compaction that destination belongs
//: to a neighbour. -- see docs/internals/carry/carry_kernel.md#streams
template <class BP, int D>
__device__ __forceinline__ void stage_pair_fold(const CarryParams& p, const Smem<BP>& sm,
                                                const int (&abase)[D], int warp, int bh, int tok,
                                                int row) {
  constexpr int kGroups = BP::kDv / 8;
  const ops::SmemAddr tile = sm.fold_values(warp);
  const uint32_t voff = ((uint32_t)bh * (uint32_t)p.L + (uint32_t)tok) * (uint32_t)BP::kDv * 2u;
  rola::static_for<kGroups>([&](auto Gc) {
    constexpr int gp = decltype(Gc)::value;
    ops::stage_tile(tile + (uint32_t)v_group_offset<BP>(row, gp),
                    ops::gather_ptr(p.v, voff + (uint32_t)(gp * 16)));
  });
  const ops::SmemAddr raw = ops::smem_addr(sm.fold_factors(warp));
  const uint32_t plane = ((uint32_t)bh * (uint32_t)p.L + (uint32_t)tok) * (uint32_t)p.g.wtot * 2u;
  rola::static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    constexpr int kSpan = owner_span(D, BP::kBC, l);
    constexpr int kBase = span_base(D, BP::kBC, l);
    constexpr int kQ = run_quantum(kSpan, kBase, BP::kRawStride);
    rola::static_for<kSpan * 2 / kQ>([&](auto Qc) {
      constexpr int q = decltype(Qc)::value;
      ops::stage_run<kQ>(raw + (uint32_t)((row * BP::kRawStride + kBase) * 2 + q * kQ),
                         ops::gather_ptr(p.pwrite, plane + (uint32_t)(abase[l] * 2 + q * kQ)));
    });
  });
}

template <class BP, int D>
__device__ __forceinline__ void stage_pair_read(const CarryParams& p, const Smem<BP>& sm,
                                                const int (&abase)[D], int warp, int bh, int tok,
                                                int rel, int row) {
  const ops::SmemAddr raw = ops::smem_addr(sm.read_ring(warp));
  const uint32_t plane = ((uint32_t)bh * (uint32_t)p.L + (uint32_t)tok) * (uint32_t)p.g.wtot * 2u;
  rola::static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    constexpr int kSpan = owner_span(D, BP::kBC, l);
    constexpr int kBase = span_base(D, BP::kBC, l);
    constexpr int kQ = run_quantum(kSpan, kBase, BP::kRawStride);
    rola::static_for<kSpan * 2 / kQ>([&](auto Qc) {
      constexpr int q = decltype(Qc)::value;
      ops::stage_run<kQ>(raw + (uint32_t)((row * BP::kRawStride + kBase) * 2 + q * kQ),
                         ops::gather_ptr(p.pread, plane + (uint32_t)(abase[l] * 2 + q * kQ)));
    });
  });
  sm.read_tokens(warp)[row] = (unsigned short)rel;
}

//: THE UNFILLED ROWS of a segment the window ran out of tokens for, ZEROED rather than
//: predicated. -- see docs/internals/carry/carry_kernel.md#streams
template <class BP>
__device__ __forceinline__ void pad_fold_slot(const Smem<BP>& sm, int warp, int lane, int fill) {
  constexpr int kGroups = BP::kDv / 8;
  constexpr int kWords = BP::kRawStride / 2;
  const int rows = BP::kSeg - fill;
  const ops::SmemAddr tile = sm.fold_values(warp);
  const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll 1
  for (int i = lane; i < rows * kGroups; i += ops::kBallotLanes)
    ops::store_shared_vec16(tile + (uint32_t)v_group_offset<BP>(fill + i / kGroups, i % kGroups),
                            zero);
  const ops::SmemAddr raw = ops::smem_addr(sm.fold_factors(warp));
#pragma unroll 1
  for (int i = lane; i < rows * kWords; i += ops::kBallotLanes)
    ops::store_shared_u32(raw + (uint32_t)(((fill + i / kWords) * kWords + i % kWords) * 4), 0u);
#pragma unroll 1
  for (int i = lane; i < rows; i += ops::kBallotLanes) sm.fold_gains(warp)[fill + i] = 0u;
}

template <class BP>
__device__ __forceinline__ void pad_read_slot(const Smem<BP>& sm, int warp, int lane, int fill) {
  constexpr int kWords = BP::kRawStride / 2;
  const int rows = BP::kSeg - fill;
  const ops::SmemAddr raw = ops::smem_addr(sm.read_ring(warp));
#pragma unroll 1
  for (int i = lane; i < rows * kWords; i += ops::kBallotLanes)
    ops::store_shared_u32(raw + (uint32_t)(((fill + i / kWords) * kWords + i % kWords) * 4), 0u);
#pragma unroll 1
  for (int i = lane; i < rows; i += ops::kBallotLanes)
    sm.read_tokens(warp)[fill + i] = (unsigned short)kDeadToken;
}

//: EXPAND, FROM THE RAW ROWS: a private stream has ONE consumer, so the token pair is
//: assembled where it is used rather than by a cooperative transpose.
//: -- see docs/internals/carry/carry_kernel.md#fold-sparse
template <class BP, int D, int NS, int WA>
__device__ __forceinline__ void fold_expand_raw(const Smem<BP>& sm, uint32_t (&bf)[BP::kNL][2],
                                                const int (&scol)[D][BP::kNL], int warp, int lane,
                                                int tile0, int tiles) {
  const ops::SmemAddr raw = ops::smem_addr(sm.fold_factors(warp));
  const unsigned short* const gains = sm.fold_gains(warp);
  const int t2 = lane & 3;
  uint32_t gain[2];
  rola::static_for<2>([&](auto Hc) {
    constexpr int h = decltype(Hc)::value;
    const int u = t2 + 4 * h;
    gain[h] = (uint32_t)gains[2 * u] | ((uint32_t)gains[2 * u + 1] << 16);
  });
  rola::static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    constexpr int kSh = slice_shift(sb_shift_of(D, BP::kBC, NS, WA, l));
    constexpr int kBits =
        slice_bits(sb_shift_of(D, BP::kBC, NS, WA, l), sb_span_bits(D, BP::kBC, NS, WA, l));
    uint32_t f[1 << kBits][2];
    rola::static_for<1 << kBits>([&](auto Vc) {
      constexpr int v = decltype(Vc)::value;
      const uint32_t col = (uint32_t)(scol[l][v] * 2);
      rola::static_for<2>([&](auto Hc) {
        constexpr int h = decltype(Hc)::value;
        const ops::SmemAddr r0 = raw + (uint32_t)(2 * (t2 + 4 * h) * BP::kRawStride * 2) + col;
        const uint32_t pair =
            (uint32_t)ops::load_shared_u16(r0)
            | ((uint32_t)ops::load_shared_u16(r0 + (uint32_t)(BP::kRawStride * 2)) << 16);
        f[v][h] = l == 0 ? ops::mul_bf16x2(pair, gain[h]) : pair;
      });
    });
    rola::static_for<BP::kNL>([&](auto Jc) {
      constexpr int j = decltype(Jc)::value;
      constexpr int v = (j >> kSh) & ((1 << kBits) - 1);
      if (j < tile0 || j >= tile0 + tiles) return;
      rola::static_for<2>([&](auto Hc) {
        constexpr int h = decltype(Hc)::value;
        bf[j][h] = l == 0 ? f[v][h] : ops::mul_bf16x2(bf[j][h], f[v][h]);
      });
    });
  });
}

//: PRODUCE one read segment's coefficients for the k-steps a stream owns.
template <class BP, int D>
__device__ __forceinline__ void readout_produce_range(ops::SmemAddr slot, int lane,
                                                      const int (&rcin)[BP::kDw][2],
                                                      const int (&rcout)[D][BP::kDw],
                                                      uint32_t (&rb)[BP::kDw][BP::kTokTiles][2],
                                                      int k0, int ks) {
  const ops::SmemAddr row0 = slot + (uint32_t)((lane >> 2) * BP::kRawStride * 2);
  rola::static_for<BP::kTokTiles>([&](auto Tc) {
    constexpr int t = decltype(Tc)::value;
    const ops::SmemAddr row = row0 + (uint32_t)(t * ops::kMmaN * BP::kRawStride * 2);
    rola::static_for<BP::kDw>([&](auto Kc) {
      constexpr int kk = decltype(Kc)::value;
      if (kk < k0 || kk >= k0 + ks) return;
      rola::static_for<2>([&](auto Hc) {
        constexpr int h = decltype(Hc)::value;
        uint32_t c = ops::load_shared_u32(row + (uint32_t)(rcin[kk][h] * 2));
        rola::static_for<D - 1>([&](auto Lc) {
          constexpr int l = decltype(Lc)::value;
          c = ops::mul_bf16x2(
              c, ops::splat_bf16x2_bits(ops::load_shared_u16(row + (uint32_t)(rcout[l][kk] * 2))));
        });
        rb[kk][t][h] = c;
      });
    });
  });
}

//: SCATTER: the readout's partial straight into the planes. A compacted stream's token list
//: is its own, so the panel has no common row to sum and there is no panel.
//: -- see docs/internals/carry/carry_kernel.md#readout-sparse
template <class BP, int M0, int MN>
__device__ __forceinline__ void readout_scatter(const Smem<BP>& sm, float* num, int warp, int lane,
                                                int t0, const float (&po)[MN][BP::kTokTiles][4]) {
  const unsigned short* const ids = sm.read_tokens(warp);
  rola::static_for<BP::kTokTiles>([&](auto Tc) {
    constexpr int t = decltype(Tc)::value;
    rola::static_for<4>([&](auto Ec) {
      constexpr int e = decltype(Ec)::value;
      const int rel = (int)ids[t * ops::kMmaN + 2 * (lane & 3) + (e & 1)];
      if (rel == kDeadToken) return;
      //: ONE 64-BIT ADDRESS PER TOKEN, and the channel m-tiles at IMMEDIATE displacements off
      //: it: a lane's cells are a fixed sixteen floats apart.
      float* const row = reinterpret_cast<float*>(ops::gather_ptr(
          num, (uint32_t)(t0 + rel) * (uint32_t)(BP::kDv * 4)
                   + (uint32_t)((M0 * ops::kMmaM + 8 * (e >> 1) + (lane >> 2)) * 4)));
      rola::static_for<MN>([&](auto Mc) {
        constexpr int m = decltype(Mc)::value;
        ops::fan_in_add(row + m * ops::kMmaM, po[m][t][e]);
      });
    });
  });
}

//: THE MASS COLUMN's own deposit: one cell per token, on the lane quad that holds it.
template <class BP>
__device__ __forceinline__ void readout_scatter_mass(const Smem<BP>& sm, float* den, int warp,
                                                     int lane, int t0,
                                                     const float (&pm)[BP::kTokTiles][4]) {
  if ((lane >> 2) != 0) return;
  const unsigned short* const ids = sm.read_tokens(warp);
  rola::static_for<BP::kTokTiles>([&](auto Tc) {
    constexpr int t = decltype(Tc)::value;
    rola::static_for<2>([&](auto Ec) {
      constexpr int e = decltype(Ec)::value;
      const int rel = (int)ids[t * ops::kMmaN + 2 * (lane & 3) + e];
      if (rel == kDeadToken) return;
      ops::fan_in_add(reinterpret_cast<float*>(ops::gather_ptr(den, (uint32_t)(t0 + rel) * 4u)),
                      pm[t][e]);
    });
  });
}

//: PHASE 2, THE SPARSE FORM. One live stream per warp, the catalog walked in TURNS; the one
//: CTA crossing is the write-after-read against the rejoin, because the rings alias the
//: region. -- see docs/internals/carry/carry_kernel.md#fold-sparse
template <class BP, int D, int BC, int NS, int WA>
__device__ __forceinline__ void fold_phase_sparse(const CarryParams& p, const Smem<BP>& sm,
                                                  Acc<BP>& acc, const int (&scol)[D][BP::kNL],
                                                  const int (&abase)[D], int warp, int lane, int bh,
                                                  int t0, int wLen, uint32_t mask0) {
  const int gbits = p.fold_grain_bits;
  const int turns = BP::kLeavesPerWarp >> gbits;
  const int tiles = (1 << gbits) / ops::kMmaN;
  const int nWords = (wLen + ops::kBallotLanes - 1) / ops::kBallotLanes;
  unsigned int* const produced = sm.produced(warp);
  unsigned int* const consumed = sm.consumed(warp);

  ops::rendezvous_group((int)StreamEdge::kEmpty, BP::kThreads);

#pragma unroll 1
  for (int turn = 0; turn < turns; ++turn) {
    const int P0 = turn << gbits;
    const int tile0 = P0 / ops::kMmaN;
    const uint32_t mine = turn == 0
                              ? mask0
                              : stream_masks<BP, D, BC>(p, p.g.w, rola::facts::kWrite, abase, warp,
                                                        lane, bh, P0, gbits, t0, nWords);
    if (lane == 0) {
      *produced = 0u;
      *consumed = 0u;
    }
    __syncwarp();

    int word = 0;
    uint32_t live = (uint32_t)ops::lane_broadcast((int)mine, 0);

    //: ONE SEGMENT'S CLAIM: lanes place their live tokens at `numProduced + rank` until the
    //: slot's capacity is reached, and the slot's capacity IS one segment.
    auto claim = [&]() -> int {
      int n = 0;
#pragma unroll 1
      while (n < BP::kSeg) {
#pragma unroll 1
        while (live == 0u && word + 1 < nWords)
          live = (uint32_t)ops::lane_broadcast((int)mine, ++word);
        if (live == 0u) break;
        const int rank = __popc(live & ops::lanes_below());
        const bool take = ((live >> lane) & 1u) != 0u && rank < BP::kSeg - n;
        const uint32_t claimed = ops::lane_ballot(take);
        unsigned int base = 0u;
        if (lane == 0) base = atomicAdd(produced, (unsigned int)__popc(claimed));
        const int slot = ops::lane_broadcast((int)base, 0) + rank;
        const int tok = t0 + (word << 5) + lane;
        if (take) {
          stage_pair_fold<BP, D>(p, sm, abase, warp, bh, tok, slot & (BP::kSeg - 1));
          sm.fold_gains(warp)[slot & (BP::kSeg - 1)] = *(const unsigned short*)ops::gather_ptr(
              p.gwrite, ((uint32_t)bh * (uint32_t)p.L + (uint32_t)tok) * 2u);
        }
        n += __popc(claimed);
        live &= ~claimed;
      }
      ops::stage_commit();
      return n;
    };

    int fill = claim();
#pragma unroll 1
    while (fill > 0) {
      ops::stage_wait<0>();
      __syncwarp();
      if (fill < BP::kSeg) {
        pad_fold_slot<BP>(sm, warp, lane, fill);
        __syncwarp();
      }
      //: THE MASS TILE'S A FRAGMENT IS A CONSTANT, built at the deposit rather than drained.
      uint32_t vf[BP::kMassTile][4];
      fold_drain_values<BP>(sm.fold_values(warp), vf, lane);
      uint32_t bf[BP::kNL][2];
      fold_expand_raw<BP, D, NS, WA>(sm, bf, scol, warp, lane, tile0, tiles);
      if (lane == 0) atomicAdd(consumed, (unsigned int)fill);
      uint32_t mf[4];
      mass_frag<BP>(mf, lane);
      rola::static_for<BP::kMCh>([&](auto Mc) {
        constexpr int mc = decltype(Mc)::value;
        rola::static_for<BP::kNL>([&](auto Jc) {
          constexpr int j = decltype(Jc)::value;
          if (j < tile0 || j >= tile0 + tiles) return;
          if constexpr (mc == BP::kMassTile)
            ops::mma(acc.c[mc][j], mf, bf[j]);
          else
            ops::mma(acc.c[mc][j], vf[mc], bf[j]);
        });
      });
      //: THE COVER IS THE OTHER WARPS', NOT THIS ONE'S -- the sparse form crosses no CTA edge
      //: inside the phase. -- see docs/internals/carry/carry_kernel.md#fold-sparse
      fill = fill == BP::kSeg ? claim() : 0;
    }
  }
}

//: PHASE 1, THE SPARSE FORM: the same streams against the accumulator instead of into it, with
//: no value tile, no column block and no panel. The phase crosses the CTA zero times.
//: -- see docs/internals/carry/carry_kernel.md#readout-sparse
template <class BP, int D, int BC>
__device__ __forceinline__ void readout_phase_sparse(const CarryParams& p, const Smem<BP>& sm,
                                                     const Transit<BP>& tr, const int (&abase)[D],
                                                     const int (&rcin)[BP::kDw][2],
                                                     const int (&rcout)[D][BP::kDw], float* num,
                                                     float* den, int warp, int lane, int bh, int t0,
                                                     int wLen, uint32_t mask0) {
  const int gbits = p.readout_grain_bits;
  const int turns = BP::kLeavesPerWarp >> gbits;
  const int ks = (1 << gbits) / ops::kMmaK;
  const int nWords = (wLen + ops::kBallotLanes - 1) / ops::kBallotLanes;
  unsigned int* const produced = sm.produced(warp);
  unsigned int* const consumed = sm.consumed(warp);

#pragma unroll 1
  for (int turn = 0; turn < turns; ++turn) {
    const int P0 = turn << gbits;
    const int k0 = P0 / ops::kMmaK;
    const uint32_t mine = turn == 0
                              ? mask0
                              : stream_masks<BP, D, BC>(p, p.g.r, rola::facts::kRead, abase, warp,
                                                        lane, bh, P0, gbits, t0, nWords);
    if (lane == 0) {
      *produced = 0u;
      *consumed = 0u;
    }
    __syncwarp();

    int word = 0;
    uint32_t live = (uint32_t)ops::lane_broadcast((int)mine, 0);

    auto claim = [&]() -> int {
      int n = 0;
#pragma unroll 1
      while (n < BP::kSeg) {
#pragma unroll 1
        while (live == 0u && word + 1 < nWords)
          live = (uint32_t)ops::lane_broadcast((int)mine, ++word);
        if (live == 0u) break;
        const int rank = __popc(live & ops::lanes_below());
        const bool take = ((live >> lane) & 1u) != 0u && rank < BP::kSeg - n;
        const uint32_t claimed = ops::lane_ballot(take);
        unsigned int base = 0u;
        if (lane == 0) base = atomicAdd(produced, (unsigned int)__popc(claimed));
        const int slot = ops::lane_broadcast((int)base, 0) + rank;
        const int rel = (word << 5) + lane;
        if (take)
          stage_pair_read<BP, D>(p, sm, abase, warp, bh, t0 + rel, rel, slot & (BP::kSeg - 1));
        n += __popc(claimed);
        live &= ~claimed;
      }
      ops::stage_commit();
      return n;
    };

    int fill = claim();
#pragma unroll 1
    while (fill > 0) {
      ops::stage_wait<0>();
      __syncwarp();
      if (fill < BP::kSeg) {
        pad_read_slot<BP>(sm, warp, lane, fill);
        __syncwarp();
      }
      uint32_t rb[BP::kDw][BP::kTokTiles][2];
      readout_produce_range<BP, D>(ops::smem_addr(sm.read_ring(warp)), lane, rcin, rcout, rb, k0,
                                   ks);
      //: THE PARTIAL IS CARRIED A CHANNEL GROUP AT A TIME: a direct deposit can be issued
      //: before the next group's MMAs. -- see docs/internals/carry/carry_kernel.md#readout-sparse
      rola::static_for<BP::kMassTile / kScatterTiles>([&](auto Gc) {
        constexpr int g = decltype(Gc)::value;
        float po[kScatterTiles][BP::kTokTiles][4];
        rola::static_for<BP::kTokTiles>([&](auto Tc) {
          constexpr int t = decltype(Tc)::value;
          rola::static_for<4>([&](auto Ec) {
            constexpr int e = decltype(Ec)::value;
            rola::static_for<kScatterTiles>([&](auto Mc) { po[decltype(Mc)::value][t][e] = 0.0f; });
          });
          rola::static_for<BP::kDw>([&](auto Kc) {
            constexpr int kk = decltype(Kc)::value;
            if (kk < k0 || kk >= k0 + ks) return;
            rola::static_for<kScatterTiles>([&](auto Mc) {
              constexpr int m = decltype(Mc)::value;
              ops::mma(po[m][t], tr.sa[kk][g * kScatterTiles + m], rb[kk][t]);
            });
          });
        });
        readout_scatter<BP, g * kScatterTiles, kScatterTiles>(sm, num, warp, lane, t0, po);
      });
      {
        float pm[BP::kTokTiles][4];
        rola::static_for<BP::kTokTiles>([&](auto Tc) {
          constexpr int t = decltype(Tc)::value;
          rola::static_for<4>([&](auto Ec) { pm[t][decltype(Ec)::value] = 0.0f; });
          rola::static_for<BP::kDw>([&](auto Kc) {
            constexpr int kk = decltype(Kc)::value;
            if (kk < k0 || kk >= k0 + ks) return;
            ops::mma(pm[t], tr.ma[kk], rb[kk][t]);
          });
        });
        readout_scatter_mass<BP>(sm, den, warp, lane, t0, pm);
      }
      if (lane == 0) atomicAdd(consumed, (unsigned int)fill);
      fill = fill == BP::kSeg ? claim() : 0;
    }
  }
}

// ------------------------------------------------------------------- the window loop

template <class BP, int D, int BC, int NS, int WA>
__device__ __forceinline__ void window_loop(const CarryParams& p, const Smem<BP>& sm, Acc<BP>& acc,
                                            int owner, float* num, float* den, int tid, int warp,
                                            int lane, int bh) {
  //: THE BOX'S FACTOR RUNS, which are ALSO the class-1 table's row bases. Derived per PHASE
  //: rather than per launch, so the memo does not outlive the phase that reads it.
  //: -- see docs/internals/carry/carry_kernel.md#factor-rows
  const auto factor_runs = [&](int(&abase)[D]) {
    rola::static_for<D>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      abase[l] = p.g.col_base[l] + geom_digit<D, BC>(p.g, owner, 0, l);
    });
  };
  const int nWindows = (p.L + kWindow - 1) / kWindow;
  uint32_t claim = 0u;

#pragma unroll 1
  for (int win = 0; win < nWindows; ++win) {
    const int t0 = win * kWindow;
    const int wLen = p.L - t0 < kWindow ? p.L - t0 : kWindow;
    const int nWords = (wLen + ops::kBallotLanes - 1) / ops::kBallotLanes;
    Transit<BP> tr;

    //: COVER SLOT A: routing-only work in front of the out leg's edge. The union readout
    //: stages its shared ring here; the sparse readout folds its first stream's liveness,
    //: which reads the class-1 words and nothing else.
    out_leg_publish<BP, D, BC>(p, sm, acc, tr, warp, lane);
    int abase[D];
    factor_runs(abase);
    uint32_t read_mask = 0u;
    if (p.readout_form == (int)Form::kUnion)
      cover_read_batch_zero<BP, D, BC>(p, sm, abase, lane, bh, t0, wLen, claim);
    else
      read_mask = stream_masks<BP, D, BC>(p, p.g.r, rola::facts::kRead, abase, warp, lane, bh, 0,
                                          p.readout_grain_bits, t0, nWords);
    ops::rendezvous_group((int)Edge::kOutLeg, BP::kThreads);
    out_leg_pull<BP, D, BC>(p, sm, tr, warp, lane);

    //: THE READ SIDE'S COLUMN MEMO, derived in the PHASE THAT READS IT so it does not outlive
    //: it. The innermost level's run holds the leaf PAIR a B fragment register carries; every
    //: level above it depends on the k-step alone, and the read member is RUNTIME.
    //: -- see docs/internals/carry/carry_kernel.md#read-expansion
    int rcin[BP::kDw][2];
    int rcout[D][BP::kDw];
    rola::static_for<D>([&](auto Lc) {
      rola::static_for<BP::kDw>(
          [&](auto Kc) { rcout[decltype(Lc)::value][decltype(Kc)::value] = 0; });
    });
    rola::static_for<BP::kDw>([&](auto Kc) {
      constexpr int kk = decltype(Kc)::value;
      rola::static_for<2>([&](auto Hc) {
        constexpr int h = decltype(Hc)::value;
        const int local =
            geom_local_row<D, BC>(p.g.r, warp, kk * ops::kMmaK + 8 * h + 2 * (lane & 3));
        rcin[kk][h] = span_base(D, BC, D - 1) + (local & (owner_span(D, BC, D - 1) - 1));
      });
      const int head = geom_local_row<D, BC>(p.g.r, warp, kk * ops::kMmaK);
      rola::static_for<D - 1>([&](auto Lc) {
        constexpr int l = decltype(Lc)::value;
        rcout[l][kk] = span_base(D, BC, l)
                       + ((head >> owner_run_shift(D, BC, l)) & (owner_span(D, BC, l) - 1));
      });
    });

    rola::uniform_switch<kForms>(p.readout_form, [&](auto Rf) {
      if constexpr ((Form) decltype(Rf)::value == Form::kUnion) {
        readout_phase<BP, D>(p, sm, tr, abase, rcin, rcout, num, den, tid, warp, lane, bh, t0, wLen,
                             claim);
      } else {
        readout_phase_sparse<BP, D, BC>(p, sm, tr, abase, rcin, rcout, num, den, warp, lane, bh, t0,
                                        wLen, read_mask);
      }
    });

    //: COVER SLOT B, the same division on the write side.
    return_leg_store<BP, D, BC>(p, sm, tr, warp, lane);
    int wbase[D];
    factor_runs(wbase);
    uint32_t fold_mask = 0u;
    if (p.fold_form == (int)Form::kUnion)
      cover_write_batch_zero<BP, D>(p, sm, wbase, warp, lane, bh, t0, wLen, claim);
    else
      fold_mask = stream_masks<BP, D, BC>(p, p.g.w, rola::facts::kWrite, wbase, warp, lane, bh, 0,
                                          p.fold_grain_bits, t0, nWords);
    ops::rendezvous_group((int)Edge::kReturnLeg, BP::kThreads);
    return_leg_rejoin<BP, D, BC>(p, sm, tr, acc, warp, lane);

    int scol[D][BP::kNL];
    rola::static_for<D>([&](auto Lc) {
      rola::static_for<BP::kNL>(
          [&](auto Vc) { scol[decltype(Lc)::value][decltype(Vc)::value] = 0; });
    });
    rola::static_for<D>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      constexpr int kSh = slice_shift(sb_shift_of(D, BC, NS, WA, l));
      constexpr int kBits =
          slice_bits(sb_shift_of(D, BC, NS, WA, l), sb_span_bits(D, BC, NS, WA, l));
      rola::static_for<1 << kBits>([&](auto Vc) {
        constexpr int v = decltype(Vc)::value;
        const int P = (v << kSh) * ops::kMmaN + (lane >> 2);
        const int local = slot_local<D, BC, NS, WA>(p.g.w, warp, P);
        scol[l][v] = span_base(D, BC, l)
                     + ((local >> owner_run_shift(D, BC, l)) & (owner_span(D, BC, l) - 1));
      });
    });

    rola::uniform_switch<kForms>(p.fold_form, [&](auto Ff) {
      if constexpr ((Form) decltype(Ff)::value == Form::kUnion) {
        fold_phase<BP, D, NS, WA>(p, sm, acc, scol, wbase, tid, warp, lane, bh, t0, wLen, claim);
      } else {
        fold_phase_sparse<BP, D, BC, NS, WA>(p, sm, acc, scol, wbase, warp, lane, bh, t0, wLen,
                                             fold_mask);
      }
    });

    ops::rendezvous_group((int)Edge::kWindowLine, BP::kThreads);
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
  //: ONE 64-BIT BASE PER READOUT PLANE, and 32-bit byte displacements after it.
  float* const num = ops::pin_address(p.num + (long)bh * (long)p.L * (long)BP::kDv);
  float* const den = ops::pin_address(p.den + (long)bh * (long)p.L);
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

  //: THE FILL'S CURSORS, and the box's page slots -- the only page-table reads in the launch.
  //: The frame's edges and the streams' are NAMED BARRIERS and take neither bytes nor an
  //: initialiser. -- see docs/internals/carry/carry_kernel.md#edges
  if (tid == 0) *sm.claim() = 0u;
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

    //: THE TWO PHASES' FORMS are the only switches left inside the loop, one per phase.
    window_loop<BP, D, BC, NS, WA>(p, sm, acc, owner, num, den, tid, warp, lane, bh);

    if (p.state_out != nullptr)
      state_sweep<BP, D, NS, WA, false>(p, sm, acc, slots, act_read, act_write, warp, lane, tid,
                                        nullptr, p.state_out);
  });
}

}  // namespace rola::carry
