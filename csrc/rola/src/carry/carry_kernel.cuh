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

  __device__ __forceinline__ ops::SmemAddr pool_v() const {
    return ops::smem_addr(base + BP::kPoolVOffset);
  }

  //: A SIDE'S POOL: the write side's beside its value rows, the read side's alone.
  __device__ __forceinline__ ops::SmemAddr pool_amp(int side, int slot) const {
    const int at = side == rola::facts::kWrite ? BP::kPoolAmpOffset : BP::kReadAmpOffset;
    return ops::smem_addr(base + at + slot * BP::kAmpRowBytes);
  }

  __device__ __forceinline__ unsigned short* pool_list(int side, int stream) const {
    const int at = side == rola::facts::kWrite ? BP::kListOffset : BP::kReadListOffset;
    return reinterpret_cast<unsigned short*>(base + at + stream * BP::kListBytes);
  }

  __device__ __forceinline__ unsigned short* fill_scratch(int side, int warp) const {
    const int at = side == rola::facts::kWrite ? BP::kFillScratchOffset : BP::kReadScratchOffset;
    return reinterpret_cast<unsigned short*>(base + at + warp * BP::kFillScratchBytes);
  }

  __device__ __forceinline__ unsigned short* pool_token() const {
    return reinterpret_cast<unsigned short*>(base + BP::kPoolTokenOffset);
  }

  __device__ __forceinline__ ops::SmemAddr stage(int warp) const {
    return ops::smem_addr(base + BP::kStageOffset + warp * BP::kStageBytes);
  }

  __device__ __forceinline__ unsigned int* stream_words(int side, int stream) const {
    return reinterpret_cast<unsigned int*>(base + BP::kLiveOffset + side * BP::kLiveBytes
                                           + stream * BP::kRounds * 4);
  }

  __device__ __forceinline__ unsigned short* stream_prefix(int side, int stream) const {
    return reinterpret_cast<unsigned short*>(base + BP::kLiveOffset + side * BP::kLiveBytes
                                             + BP::kStreamWordBytes + stream * BP::kRounds * 2);
  }

  __device__ __forceinline__ char* read_raw_slot(int slot) const {
    return base + BP::kReadRawOffset + slot * BP::kRawSlotBytes;
  }

  __device__ __forceinline__ unsigned int* claim() const {
    return reinterpret_cast<unsigned int*>(base + BP::kClaimOffset);
  }

  __device__ __forceinline__ int32_t* page_slots() const {
    return reinterpret_cast<int32_t*>(base + BP::kPageSlotOffset);
  }
};

//: THE V POOL'S SWIZZLE: the 16-byte group index XORed with the row's residue, a bijection
//: of the row that leaves eight consecutive rows' `ldmatrix .trans` conflict free.
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

//: THE SENTINEL the read pool's dead row carries as its token: no token, so no deposit.
constexpr int kDeadToken = 0xFFFF;

//: THE POOL: this window's rows keyed by POSITION -- the value row and the box's amplitude
//: runs, the outermost carrying the gain. Filled BY POSITION (no cursor), gathered BY INDEX
//: (`ldmatrix` at per-lane row addresses). Row `kPoolPos` is the DEAD ROW a padded tile reads.
//: -- see docs/internals/carry/carry_kernel.md#pool
constexpr int kRoundTokens = 32;

//: THE DEAD ROW, zeroed once per window because the pool shares the transit's bytes.
template <class BP, int Side>
__device__ __forceinline__ void pool_zero_dead_row(const Smem<BP>& sm, int lane) {
  constexpr int kVBytes = Side == rola::facts::kWrite ? BP::kVRowBytes : 0;
  constexpr int kWords = (kVBytes + BP::kAmpRowBytes) / 4;
  for (int i = lane; i < kWords; i += ops::kBallotLanes) {
    const uint32_t off = (uint32_t)(i * 4);
    ops::store_shared_u32(off < kVBytes
                              ? sm.pool_v() + (uint32_t)(BP::kPoolPos * BP::kVRowBytes) + off
                              : sm.pool_amp(Side, BP::kPoolPos) + (off - kVBytes),
                          0u);
  }
  if constexpr (Side == rola::facts::kRead)
    if (lane == 0) sm.pool_token()[BP::kPoolPos] = (unsigned short)kDeadToken;
}

//: ONE POSITION'S ROWS into a slot, by lanes `l0 .. l0 + 7`: the value row's eight quanta
//: by `cp.async`; the OUTERMOST amplitude run through registers with the gain folded in
//: (`kWords0` words, one per lane); the runs below by `cp.async` at their quantum.
template <class BP, int D, int Side>
__device__ __forceinline__ void pool_copy_position(const CarryParams& p, const Smem<BP>& sm,
                                                   const int (&abase)[D], int sub, int bh, int tok,
                                                   int slot) {
  constexpr bool kW = Side == rola::facts::kWrite;
  constexpr int kGroups = BP::kDv / 8;
  static_assert(kGroups == 8, "a value row is eight sixteen-byte quanta, one per lane of eight");
  const __nv_bfloat16* const plane_p = kW ? p.pwrite : p.pread;
  const uint32_t plane = (uint32_t)bh * (uint32_t)p.L * (uint32_t)p.g.wtot * 2u;
  const uint32_t row = plane + (uint32_t)tok * (uint32_t)p.g.wtot * 2u;
  if constexpr (kW) {
    const uint32_t off = ((uint32_t)bh * (uint32_t)p.L + (uint32_t)tok) * (uint32_t)BP::kDv * 2u
                         + (uint32_t)(sub * 16);
    ops::stage_tile(sm.pool_v() + (uint32_t)v_group_offset<BP>(slot, sub),
                    ops::gather_ptr(p.v, off));
    constexpr int kWords0 = owner_span(D, BP::kBC, 0) / 2;
    constexpr int kBase0 = span_base(D, BP::kBC, 0);
    static_assert(kWords0 <= 8, "the outermost run is at most eight words per position");
    if (sub < kWords0) {
      const uint32_t a = *reinterpret_cast<const uint32_t*>(
          ops::gather_ptr(p.pwrite, row + (uint32_t)((abase[0] + 2 * sub) * 2)));
      const unsigned short g = *reinterpret_cast<const unsigned short*>(
          ops::gather_ptr(p.gwrite, ((uint32_t)bh * (uint32_t)p.L + (uint32_t)tok) * 2u));
      ops::store_shared_u32(sm.pool_amp(Side, slot) + (uint32_t)((kBase0 + 2 * sub) * 2),
                            ops::mul_bf16x2(a, ops::splat_bf16x2_bits(g)));
    }
  }
  rola::static_for<kW ? D - 1 : D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value + (kW ? 1 : 0);
    constexpr int kSpan = owner_span(D, BP::kBC, l);
    constexpr int kBase = span_base(D, BP::kBC, l);
    constexpr int kQ = run_quantum(kSpan, kBase, BP::kAmpStride);
    constexpr int kPerToken = kSpan * 2 / kQ;
    static_assert(kPerToken <= 8, "a level's run is at most eight quanta per position");
    if (sub < kPerToken)
      ops::stage_run<kQ>(sm.pool_amp(Side, slot) + (uint32_t)(kBase * 2 + sub * kQ),
                         ops::gather_ptr(plane_p, row + (uint32_t)(abase[l] * 2 + sub * kQ)));
  });
}

//: ONE ROUND'S FILL, by the round's OWNER warp: the round's live lanes rank themselves once
//: into the warp's scratch, four positions per step, eight lanes to a position, those whose
//: slot falls in the batch. THE SAME WALK MINTS THE LISTS: a lane whose position is in the
//: batch writes its slot into every stream that holds it, at the stream's own rank of it
//: less the stream's count before the batch. -- see docs/internals/carry/carry_kernel.md#fill
template <class BP, int D, int Side>
__device__ __forceinline__ void pool_fill_round(const CarryParams& p, const Smem<BP>& sm,
                                                const int (&abase)[D], int warp, int lane, int bh,
                                                int t0, int round, uint32_t live, int prefix,
                                                int base, int nStreams, int sbase) {
  unsigned short* const scratch = sm.fill_scratch(Side, warp);
  const int n = __popc(live);
  if ((live >> lane) & 1u) scratch[__popc(live & ops::lanes_below())] = (unsigned short)lane;
  ops::warp_sync();
  const int g = lane >> 3, sub = lane & 7;
#pragma unroll 1
  for (int s = 0; s < n; s += 4) {
    const int idx = s + g;
    if (idx < n) {
      const int slot = prefix + idx - base;
      if (slot >= 0 && slot < BP::kPoolPos)
        pool_copy_position<BP, D, Side>(p, sm, abase, sub, bh,
                                        t0 + round * kRoundTokens + (int)scratch[idx], slot);
    }
  }
  const uint32_t below = ops::lanes_below();
  const int rank = prefix + __popc(live & below) - base;
  const bool in = ((live >> lane) & 1u) && rank >= 0 && rank < BP::kPoolPos;
  if constexpr (Side == rola::facts::kRead)
    if (in) sm.pool_token()[rank] = (unsigned short)(round * kRoundTokens + lane);
#pragma unroll 1
  for (int s = 0; s < nStreams; ++s) {
    const uint32_t w = sm.stream_words(Side, s)[round];
    const int at =
        (int)sm.stream_prefix(Side, s)[round] + __popc(w & below) - ops::lane_broadcast(sbase, s);
    if (in && ((w >> lane) & 1u)) sm.pool_list(Side, s)[at] = (unsigned short)rank;
  }
  ops::warp_sync();
  ops::stage_commit();
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

//: ONE TILE'S FRAGMENTS: the value A fragments and the finished coefficient B fragments.
template <class BP>
struct Frags {
  uint32_t vf[BP::kMCh][4];
  uint32_t bf[BP::kNL][2];
};

//: GATHER ONE TILE: the list's next sixteen positions into fragments -- the value rows by
//: `ldmatrix.trans` (one per channel m-tile; the mass m-tile is the constant), the innermost
//: run's sixteen digits as the four 8x8 blocks (tokens 0-7 | 8-15) x (digits 0-7 | 8-15) so
//: n-tile j's inner half is j & 1, and the outer runs' product applied as one packed
//: multiply per fragment register. Issued a tile AHEAD of the deposits that cover it.
//: -- see docs/internals/carry/carry_kernel.md#pool-tile
template <class BP, int D, int NS, int WA>
__device__ __forceinline__ void pool_gather(const Smem<BP>& sm, Frags<BP>& f,
                                            const int (&scol)[D][BP::kNL], int stream, int lane,
                                            int taken) {
  constexpr int kSide = rola::facts::kWrite;
  const unsigned short* const list = sm.pool_list(kSide, stream);
  const int rowg = (lane & 7) + ((lane >> 4) & 1) * 8;
  const int colhalf = (lane >> 3) & 1;
  const int slot = list[(taken + rowg) & (BP::kListEntries - 1)];

  rola::static_for<BP::kMCh>([&](auto Mc) {
    constexpr int mc = decltype(Mc)::value;
    if constexpr (mc == BP::kMassTile) {
      mass_frag<BP>(f.vf[mc], lane);
    } else {
      ops::load_frag_t(f.vf[mc],
                       sm.pool_v() + (uint32_t)v_group_offset<BP>(slot, 2 * mc + colhalf));
    }
  });

  constexpr int kShI = slice_shift(sb_shift_of(D, BP::kBC, NS, WA, D - 1));
  constexpr int kBitsI =
      slice_bits(sb_shift_of(D, BP::kBC, NS, WA, D - 1), sb_span_bits(D, BP::kBC, NS, WA, D - 1));
  constexpr int kInnerTiles = 1 << kBitsI;
  static_assert(kShI == 0 && kInnerTiles % 2 == 0,
                "the innermost level is the fastest leaf digit and spans whole 16-column groups");
  uint32_t in[kInnerTiles / 2][4];
  rola::static_for<kInnerTiles / 2>([&](auto Gc) {
    constexpr int g = decltype(Gc)::value;
    //: `scol` is this lane's own digit column; the block's base is the tile's first.
    ops::load_frag_t(in[g], sm.pool_amp(kSide, slot)
                                + (uint32_t)((scol[D - 1][2 * g] - (lane >> 2) + 8 * colhalf) * 2));
  });
  rola::static_for<BP::kNL>([&](auto Jc) {
    constexpr int j = decltype(Jc)::value;
    constexpr int g = (j % kInnerTiles) / 2, c = j & 1;
    f.bf[j][0] = in[g][c];
    f.bf[j][1] = in[g][c + 2];
  });

  const int q = lane & 3;
  int r0[2], r1[2];
  rola::static_for<2>([&](auto Hc) {
    constexpr int h = decltype(Hc)::value;
    r0[h] = list[(taken + 2 * q + 8 * h) & (BP::kListEntries - 1)];
    r1[h] = list[(taken + 2 * q + 1 + 8 * h) & (BP::kListEntries - 1)];
  });
  rola::static_for<D - 1>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    constexpr int kSh = slice_shift(sb_shift_of(D, BP::kBC, NS, WA, l));
    constexpr int kBits =
        slice_bits(sb_shift_of(D, BP::kBC, NS, WA, l), sb_span_bits(D, BP::kBC, NS, WA, l));
    uint32_t head[1 << kBits][2];
    rola::static_for<1 << kBits>([&](auto Vc) {
      constexpr int v = decltype(Vc)::value;
      const uint32_t col = (uint32_t)(scol[l][v] * 2);
      rola::static_for<2>([&](auto Hc) {
        constexpr int h = decltype(Hc)::value;
        head[v][h] = (uint32_t)ops::load_shared_u16(sm.pool_amp(kSide, r0[h]) + col)
                     | ((uint32_t)ops::load_shared_u16(sm.pool_amp(kSide, r1[h]) + col) << 16);
      });
    });
    rola::static_for<BP::kNL>([&](auto Jc) {
      constexpr int j = decltype(Jc)::value;
      constexpr int v = (j >> kSh) & ((1 << kBits) - 1);
      rola::static_for<2>([&](auto Hc) {
        constexpr int h = decltype(Hc)::value;
        f.bf[j][h] = ops::mul_bf16x2(f.bf[j][h], head[v][h]);
      });
    });
  });
}

//: DEPOSIT ONE TILE into a PASS's n-tiles `[Tile0, Tile0 + Tiles)`, a compile-time range.
template <class BP, int Tile0, int Tiles>
__device__ __forceinline__ void pool_deposit(Acc<BP>& acc, const Frags<BP>& f) {
  rola::static_for<BP::kMCh>([&](auto Mc) {
    constexpr int mc = decltype(Mc)::value;
    rola::static_for<Tiles>([&](auto Jc) {
      constexpr int j = Tile0 + decltype(Jc)::value;
      ops::mma(acc.c[mc][j], f.vf[mc], f.bf[j]);
    });
  });
}

//: A PARTIAL TILE PADDED to the dead row, from the list's end to the tile's.
template <class BP>
__device__ __forceinline__ void pool_pad_tile(unsigned short* list, int count, int taken,
                                              int lane) {
  const int i = count + lane;
  if (i < taken + BP::kSeg) list[i & (BP::kListEntries - 1)] = (unsigned short)BP::kPoolPos;
}

//: ONE PASS of a batch: a stream's minted list, its whole tiles with each gather in front of
//: the previous deposits, its tail padded to the dead row.
template <class BP, int D, int NS, int WA, int Tile0, int Tiles>
__device__ __forceinline__ void pool_pass(const Smem<BP>& sm, Acc<BP>& acc,
                                          const int (&scol)[D][BP::kNL], int lane, int stream,
                                          int count) {
  unsigned short* const list = sm.pool_list(rola::facts::kWrite, stream);
  const int whole = count & ~(BP::kSeg - 1);
  if (count > whole) pool_pad_tile<BP>(list, count, whole, lane);
  const int tiles = (count + BP::kSeg - 1) / BP::kSeg;
  ops::warp_sync();
  if (tiles == 0) return;

  Frags<BP> f[2];
  pool_gather<BP, D, NS, WA>(sm, f[0], scol, stream, lane, 0);
#pragma unroll 1
  for (int t = 0; t + 1 < tiles; t += 2) {
    pool_gather<BP, D, NS, WA>(sm, f[1], scol, stream, lane, (t + 1) * BP::kSeg);
    pool_deposit<BP, Tile0, Tiles>(acc, f[0]);
    if (t + 2 < tiles) pool_gather<BP, D, NS, WA>(sm, f[0], scol, stream, lane, (t + 2) * BP::kSeg);
    pool_deposit<BP, Tile0, Tiles>(acc, f[1]);
  }
  if (tiles & 1) pool_deposit<BP, Tile0, Tiles>(acc, f[0]);
  ops::warp_sync();
}

//: A SIDE'S UNION WORD for round `lane`: the OR of its streams' words, read once per window.
template <class BP>
__device__ __forceinline__ uint32_t union_word(const Smem<BP>& sm, int side, int nStreams,
                                               int nRounds, int lane) {
  uint32_t u = 0u;
  if (lane < nRounds) {
#pragma unroll 1
    for (int s = 0; s < nStreams; ++s) u |= sm.stream_words(side, s)[lane];
  }
  return u;
}

//: A STREAM'S COUNT BEFORE A UNION RANK, in lane `stream` (every lane takes part: the round
//: is found by ballot): the round the rank falls in, the lane within it, the stream's prefix
//: there plus its bits below that lane.
template <class BP>
__device__ __forceinline__ int stream_count_before(const Smem<BP>& sm, int side, uint32_t cta,
                                                   int prefix, int total, int nRounds, int nStreams,
                                                   int lane, int rank) {
  const int s = lane < nStreams ? lane : nStreams - 1;
  if (rank >= total)
    return (int)sm.stream_prefix(side, s)[nRounds - 1]
           + __popc(sm.stream_words(side, s)[nRounds - 1]);
  const int r = __ffs((int)ops::lane_ballot(prefix + __popc(cta) > rank)) - 1;
  const int pr = ops::lane_broadcast(prefix, r);
  const uint32_t uw = (uint32_t)ops::lane_broadcast((int)cta, r);
  //: the lane of the rank's position: the set bit with `rank - pr` set bits below it.
  const uint32_t below = ops::lanes_below();
  const uint32_t at = ops::lane_ballot(((uw >> lane) & 1u) && __popc(uw & below) == rank - pr);
  return (int)sm.stream_prefix(side, s)[r] + __popc(sm.stream_words(side, s)[r] & (at - 1u));
}

//: PHASE 2, in BATCHES of the pool BY RANK: the CTA's live positions, in order, `kPoolPos`
//: at a time -- one batch when the window's live count fits. Fill the batch (each round by
//: its owner warp, minting every stream's list as it goes), one edge, one PASS per stream
//: the warp's box holds at the fold's grain -- its own list over the same rows into its own
//: n-tiles -- one edge. Nothing is reclaimed inside a batch, so nothing is tracked.
//: -- see docs/internals/carry/carry_kernel.md#fold
template <class BP, int D, int NS, int WA, int GB>
__device__ __forceinline__ void fold_phase(const CarryParams& p, const Smem<BP>& sm, Acc<BP>& acc,
                                           const int (&scol)[D][BP::kNL], const int (&abase)[D],
                                           int warp, int lane, int bh, int t0, int wLen) {
  const int nRounds = (wLen + kRoundTokens - 1) / kRoundTokens;
  constexpr int passes = BP::kLeavesPerWarp >> GB;
  constexpr int kTilesPerPass = (1 << GB) / ops::kMmaN;

  //: the union's words and their exclusive prefix, one lane per round.
  constexpr int kSide = rola::facts::kWrite;
  const int nStreams = BP::kWarps * passes;
  const uint32_t cta = union_word<BP>(sm, kSide, nStreams, nRounds, lane);
  int prefix = __popc(cta);
  rola::static_for<5>([&](auto Sc) {
    constexpr int d = 1 << decltype(Sc)::value;
    const int up = __shfl_up_sync(0xFFFFFFFFu, prefix, d);
    if (lane >= d) prefix += up;
  });
  const int total = ops::lane_broadcast(prefix, nRounds - 1);
  prefix -= __popc(cta);

  //: A WINDOW WITH NOTHING LIVE touches the pool not at all; the pool aliases the transit
  //: region, so no warp fills until every warp has rejoined its halves out of it.
  if (total == 0) return;
  ops::rendezvous_group((int)Edge::kFoldRound, BP::kThreads);
  if (warp == 0) pool_zero_dead_row<BP, kSide>(sm, lane);

#pragma unroll 1
  for (int base = 0; base < total; base += BP::kPoolPos) {
    const int sbase =
        stream_count_before<BP>(sm, kSide, cta, prefix, total, nRounds, nStreams, lane, base);
    const int send = stream_count_before<BP>(sm, kSide, cta, prefix, total, nRounds, nStreams, lane,
                                             base + BP::kPoolPos);
#pragma unroll 1
    for (int r = warp; r < nRounds; r += BP::kWarps)
      pool_fill_round<BP, D, kSide>(p, sm, abase, warp, lane, bh, t0, r,
                                    ops::lane_broadcast((int)cta, r),
                                    ops::lane_broadcast(prefix, r), base, nStreams, sbase);
    ops::stage_wait<0>();
    ops::rendezvous_group((int)Edge::kFoldRound, BP::kThreads);

    rola::static_for<passes>([&](auto Pc) {
      constexpr int pass = decltype(Pc)::value;
      const int s = warp * passes + pass;
      pool_pass<BP, D, NS, WA, pass * kTilesPerPass, kTilesPerPass>(
          sm, acc, scol, lane, s, ops::lane_broadcast(send - sbase, s));
    });
    ops::rendezvous_group((int)Edge::kFoldRound, BP::kThreads);
  }
}

// ------------------------------------------------------- the streams, both phases

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

template <class BP, int D, int K0, int KS>
__device__ __forceinline__ void readout_produce_range(const Smem<BP>& sm,
                                                      const unsigned short* list, int taken,
                                                      int lane, const int (&rcin)[BP::kDw][2],
                                                      const int (&rcout)[D][BP::kDw],
                                                      uint32_t (&rb)[BP::kDw][BP::kTokTiles][2]) {
  rola::static_for<BP::kTokTiles>([&](auto Tc) {
    constexpr int t = decltype(Tc)::value;
    const int slot = list[taken + t * ops::kMmaN + (lane >> 2)];
    const ops::SmemAddr row = sm.pool_amp(rola::facts::kRead, slot);
    rola::static_for<BP::kDw>([&](auto Kc) {
      constexpr int kk = decltype(Kc)::value;
      if constexpr (kk < K0 || kk >= K0 + KS) return;
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

//: ONE TILE'S DEPOSIT, COALESCED THROUGH THE WARP'S STAGING TILE: the partial is produced a
//: channel m-tile at a time and stored element-wise into `[token][channel]` rows; then each
//: token's 64 channels leave as two reductions of 32 consecutive floats (four sectors an
//: instruction, not thirty-two), the mass as one; a full tile deposits untested, the tail
//: tile skips its dead tokens. -- see docs/internals/carry/carry_kernel.md#readout-sparse
template <class BP, int K0, int KS>
__device__ __forceinline__ void readout_deposit_tile(
    const Smem<BP>& sm, const Transit<BP>& tr, const uint32_t (&rb)[BP::kDw][BP::kTokTiles][2],
    const unsigned short* list, int taken, float* num, float* den, int warp, int lane, int t0,
    bool full) {
  constexpr int kS = BP::kStageStride;
  const ops::SmemAddr stage = sm.stage(warp);
  const int q = lane & 3, r = lane >> 2;

  rola::static_for<BP::kMassTile>([&](auto Gc) {
    constexpr int g = decltype(Gc)::value;
    float po[BP::kTokTiles][4];
    rola::static_for<BP::kTokTiles>([&](auto Tc) {
      constexpr int t = decltype(Tc)::value;
      rola::static_for<4>([&](auto Ec) { po[t][decltype(Ec)::value] = 0.0f; });
      rola::static_for<BP::kDw>([&](auto Kc) {
        constexpr int kk = decltype(Kc)::value;
        if constexpr (kk < K0 || kk >= K0 + KS) return;
        ops::mma(po[t], tr.sa[kk][g], rb[kk][t]);
      });
      rola::static_for<4>([&](auto Ec) {
        constexpr int e = decltype(Ec)::value;
        const int c = g * ops::kMmaM + 8 * (e >> 1) + r;
        const int tk = t * ops::kMmaN + 2 * q + (e & 1);
        ops::store_shared_u32(stage + (uint32_t)((tk * kS + c) * 4), __float_as_uint(po[t][e]));
      });
    });
  });
  {
    float pm[BP::kTokTiles][4];
    rola::static_for<BP::kTokTiles>([&](auto Tc) {
      constexpr int t = decltype(Tc)::value;
      rola::static_for<4>([&](auto Ec) { pm[t][decltype(Ec)::value] = 0.0f; });
      rola::static_for<BP::kDw>([&](auto Kc) {
        constexpr int kk = decltype(Kc)::value;
        if constexpr (kk < K0 || kk >= K0 + KS) return;
        ops::mma(pm[t], tr.ma[kk], rb[kk][t]);
      });
      if (r == 0)
        rola::static_for<2>([&](auto Ec) {
          constexpr int e = decltype(Ec)::value;
          ops::store_shared_u32(
              stage + (uint32_t)(((t * ops::kMmaN + 2 * q + e) * kS + BP::kDv) * 4),
              __float_as_uint(pm[t][e]));
        });
    });
  }
  ops::warp_sync();

  //: lane `l & 15` holds token `l & 15`'s window-relative id. A FULL tile (every tile but
  //: a stream's last) deposits with no test at all; the tail tile skips its dead tokens by
  //: a warp-uniform test per token, so no zero is ever reduced anywhere.
  const int tok = (int)sm.pool_token()[list[taken + (lane & (BP::kSeg - 1))]];
  const uint32_t lanebase = (uint32_t)t0 * (uint32_t)(BP::kDv * 4) + (uint32_t)(lane * 4);
  const auto rows = [&](auto Full) {
    rola::static_for<BP::kSeg>([&](auto Tc) {
      constexpr int T = decltype(Tc)::value;
      const int rel = ops::lane_broadcast(tok, T);
      if constexpr (!decltype(Full)::value)
        if (rel == kDeadToken) return;
      float* const row = reinterpret_cast<float*>(
          ops::gather_ptr(num, (uint32_t)rel * (uint32_t)(BP::kDv * 4) + lanebase));
      rola::static_for<BP::kDv / 32>([&](auto Hc) {
        constexpr int h = decltype(Hc)::value;
        const float v =
            __uint_as_float(ops::load_shared_u32(stage + (uint32_t)((T * kS + lane + 32 * h) * 4)));
        ops::fan_in_add(row + 32 * h, v);
      });
    });
  };
  const float m = __uint_as_float(
      ops::load_shared_u32(stage + (uint32_t)(((lane & (BP::kSeg - 1)) * kS + BP::kDv) * 4)));
  float* const mrow = reinterpret_cast<float*>(ops::gather_ptr(den, (uint32_t)(t0 + tok) * 4u));
  if (full) {
    rows(std::integral_constant<bool, true>{});
    //: the mass by every lane, the upper half adding zeros beside the lower's cells.
    ops::fan_in_add(mrow, lane < BP::kSeg ? m : 0.0f);
  } else {
    rows(std::integral_constant<bool, false>{});
    if (lane < BP::kSeg && tok != kDeadToken) ops::fan_in_add(mrow, m);
  }
  ops::warp_sync();
}

//: ONE STREAM of a batch: its minted list, whole tiles, the tail padded to the dead row.
template <class BP, int D, int K0, int KS>
__device__ __forceinline__ void readout_stream(const Smem<BP>& sm, const Transit<BP>& tr,
                                               const int (&rcin)[BP::kDw][2],
                                               const int (&rcout)[D][BP::kDw], float* num,
                                               float* den, int warp, int lane, int stream,
                                               int count, int t0) {
  unsigned short* const list = sm.pool_list(rola::facts::kRead, stream);
  const int whole = count & ~(BP::kSeg - 1);
  if (count > whole) pool_pad_tile<BP>(list, count, whole, lane);
  const int tiles = (count + BP::kSeg - 1) / BP::kSeg;
  ops::warp_sync();
#pragma unroll 1
  for (int tile = 0; tile < tiles; ++tile) {
    uint32_t rb[BP::kDw][BP::kTokTiles][2];
    readout_produce_range<BP, D, K0, KS>(sm, list, tile * BP::kSeg, lane, rcin, rcout, rb);
    readout_deposit_tile<BP, K0, KS>(sm, tr, rb, list, tile * BP::kSeg, num, den, warp, lane, t0,
                                     (tile + 1) * BP::kSeg <= count);
  }
  ops::warp_sync();
}

//: PHASE 1, THE SPARSE FORM: the read pool -- the window's live positions' read rows, by
//: rank, in batches -- filled by the same walk as the fold's with every stream's list minted
//: as it goes; then each warp's streams at the readout's grain against the transit's A
//: fragments, scattering straight into the planes. The phase crosses the CTA at the batch
//: edges only. -- see docs/internals/carry/carry_kernel.md#readout-sparse
template <class BP, int D, int GB>
__device__ __forceinline__ void readout_phase_pool(const CarryParams& p, const Smem<BP>& sm,
                                                   const Transit<BP>& tr, const int (&abase)[D],
                                                   const int (&rcin)[BP::kDw][2],
                                                   const int (&rcout)[D][BP::kDw], float* num,
                                                   float* den, int warp, int lane, int bh, int t0,
                                                   int wLen) {
  constexpr int kSide = rola::facts::kRead;
  const int nRounds = (wLen + kRoundTokens - 1) / kRoundTokens;
  constexpr int passes = BP::kLeavesPerWarp >> GB;
  constexpr int kKs = (1 << GB) / ops::kMmaK;
  const int nStreams = BP::kWarps * passes;
  const uint32_t cta = union_word<BP>(sm, kSide, nStreams, nRounds, lane);
  int prefix = __popc(cta);
  rola::static_for<5>([&](auto Sc) {
    constexpr int d = 1 << decltype(Sc)::value;
    const int up = __shfl_up_sync(0xFFFFFFFFu, prefix, d);
    if (lane >= d) prefix += up;
  });
  const int total = ops::lane_broadcast(prefix, nRounds - 1);
  prefix -= __popc(cta);

  //: A WINDOW WITH NOTHING LIVE touches the pool not at all: the pool aliases the transit
  //: region, and with no batch there is no batch edge to order a dead-row write against the
  //: other warps' store-back. And no warp fills until every warp has pulled its transit rows.
  if (total == 0) return;
  ops::rendezvous_group((int)StreamEdge::kReadBatch, BP::kThreads);
  if (warp == 0) pool_zero_dead_row<BP, kSide>(sm, lane);

#pragma unroll 1
  for (int base = 0; base < total; base += BP::kPoolPos) {
    const int sbase =
        stream_count_before<BP>(sm, kSide, cta, prefix, total, nRounds, nStreams, lane, base);
    const int send = stream_count_before<BP>(sm, kSide, cta, prefix, total, nRounds, nStreams, lane,
                                             base + BP::kPoolPos);
#pragma unroll 1
    for (int r = warp; r < nRounds; r += BP::kWarps)
      pool_fill_round<BP, D, kSide>(p, sm, abase, warp, lane, bh, t0, r,
                                    ops::lane_broadcast((int)cta, r),
                                    ops::lane_broadcast(prefix, r), base, nStreams, sbase);
    ops::stage_wait<0>();
    ops::rendezvous_group((int)StreamEdge::kReadBatch, BP::kThreads);

    rola::static_for<passes>([&](auto Pc) {
      constexpr int pass = decltype(Pc)::value;
      const int s = warp * passes + pass;
      readout_stream<BP, D, pass * kKs, kKs>(sm, tr, rcin, rcout, num, den, warp, lane, s,
                                             ops::lane_broadcast(send - sbase, s), t0);
    });
    ops::rendezvous_group((int)StreamEdge::kReadBatch, BP::kThreads);
  }
}

//: A SIDE'S LIVE WORDS at a cover slot: each of the warp's streams at the side's grain,
//: its word per round (lane = round) and the word's exclusive prefix over the rounds.
template <class BP, int D, int BC>
__device__ __forceinline__ void cover_live_words(const CarryParams& p, const Smem<BP>& sm, int side,
                                                 const SideGeom& sg, const int (&abase)[D],
                                                 int gbits, int warp, int lane, int bh, int t0,
                                                 int nWords) {
  const int passes = BP::kLeavesPerWarp >> gbits;
#pragma unroll 1
  for (int pass = 0; pass < passes; ++pass) {
    const uint32_t m = stream_masks<BP, D, BC>(p, sg, side, abase, warp, lane, bh, pass << gbits,
                                               gbits, t0, nWords);
    int pre = __popc(m);
    rola::static_for<5>([&](auto Sc) {
      constexpr int d = 1 << decltype(Sc)::value;
      const int up = __shfl_up_sync(0xFFFFFFFFu, pre, d);
      if (lane >= d) pre += up;
    });
    const int s = warp * passes + pass;
    if (lane < nWords) {
      sm.stream_words(side, s)[lane] = m;
      sm.stream_prefix(side, s)[lane] = (unsigned short)(pre - __popc(m));
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
    //: stages its shared ring here; the sparse readout writes its streams' live words,
    //: which reads the class-1 words and nothing else.
    out_leg_publish<BP, D, BC>(p, sm, acc, tr, warp, lane);
    int abase[D];
    factor_runs(abase);
    if (p.readout_form == (int)Form::kUnion)
      cover_read_batch_zero<BP, D, BC>(p, sm, abase, lane, bh, t0, wLen, claim);
    else
      cover_live_words<BP, D, BC>(p, sm, rola::facts::kRead, p.g.r, abase, p.readout_grain_bits,
                                  warp, lane, bh, t0, nWords);
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
        rola::uniform_switch<BP::kGrainCases>(
            p.readout_grain_bits - BP::kGrainBitsMin, [&](auto Gc) {
              readout_phase_pool<BP, D, BP::kGrainBitsMin + decltype(Gc)::value>(
                  p, sm, tr, abase, rcin, rcout, num, den, warp, lane, bh, t0, wLen);
            });
      }
    });

    //: COVER SLOT B: the write side's liveness, folded over the warp's whole box -- routing
    //: words only, in front of the return leg's edge.
    return_leg_store<BP, D, BC>(p, sm, tr, warp, lane);
    int wbase[D];
    factor_runs(wbase);
    cover_live_words<BP, D, BC>(p, sm, rola::facts::kWrite, p.g.w, wbase, p.fold_grain_bits, warp,
                                lane, bh, t0, nWords);
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

    rola::uniform_switch<BP::kGrainCases>(p.fold_grain_bits - BP::kGrainBitsMin, [&](auto Gc) {
      fold_phase<BP, D, NS, WA, BP::kGrainBitsMin + decltype(Gc)::value>(p, sm, acc, scol, wbase,
                                                                         warp, lane, bh, t0, wLen);
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
