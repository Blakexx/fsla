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

  __device__ __forceinline__ ops::SmemAddr pool_v() const {
    return ops::smem_addr(base + BP::kPoolVOffset);
  }

  //: A SIDE'S POOL: the write side's beside its value rows, the read side's alone.
  __device__ __forceinline__ ops::SmemAddr pool_amp(int side, int slot) const {
    const int at = side == rola::facts::kWrite ? BP::kPoolAmpOffset : BP::kReadAmpOffset;
    return ops::smem_addr(base + at + slot * BP::kAmpRowBytes);
  }

  //: THE FOLD'S LISTS, one a stream; the readout's slices need none.
  __device__ __forceinline__ unsigned short* pool_list(int stream) const {
    return reinterpret_cast<unsigned short*>(base + BP::kListOffset + stream * BP::kListBytes);
  }

  __device__ __forceinline__ unsigned short* fill_scratch(int side, int warp) const {
    const int at = side == rola::facts::kWrite ? BP::kFillScratchOffset : BP::kReadScratchOffset;
    return reinterpret_cast<unsigned short*>(base + at + warp * BP::kFillScratchBytes);
  }

  __device__ __forceinline__ unsigned short* pool_gain() const {
    return reinterpret_cast<unsigned short*>(base + BP::kPoolGainOffset);
  }

  __device__ __forceinline__ unsigned short* gain_table() const {
    return reinterpret_cast<unsigned short*>(base + BP::kGainTableOffset);
  }

  //: THE UNIFORM TABLES of a side. -- see docs/internals/carry/carry_kernel.md#uniform
  __device__ __forceinline__ uint32_t* union_words(int side) const {
    return reinterpret_cast<uint32_t*>(base + BP::kUniformOffset + side * BP::kUniformWords * 4);
  }

  __device__ __forceinline__ int32_t* union_prefix(int side) const {
    return reinterpret_cast<int32_t*>(union_words(side) + BP::kRounds);
  }

  __device__ __forceinline__ int32_t* stream_base(int side) const {
    return reinterpret_cast<int32_t*>(union_words(side) + 2 * BP::kRounds);
  }

  __device__ __forceinline__ int32_t* stream_end(int side) const {
    return stream_base(side) + BP::kSideStreams;
  }

  __device__ __forceinline__ unsigned short* pool_token() const {
    return reinterpret_cast<unsigned short*>(base + BP::kPoolTokenOffset);
  }

  //: THE STAGING TILE, one a warp: its slice's rows in the row form the planes take.
  //: -- see docs/internals/carry/smem_ledger.md#pool
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

  __device__ __forceinline__ int32_t* page_slots() const {
    return reinterpret_cast<int32_t*>(base + BP::kPageSlotOffset);
  }

  //: THE READ MEMO's three tables, by k-step of the box. -- smem_ledger.md#memo
  __device__ __forceinline__ uint32_t* memo_rows(int K) const {
    return reinterpret_cast<uint32_t*>(base + BP::kMemoRowsOffset + K * 32 * 4);
  }

  __device__ __forceinline__ uint32_t* memo_mass(int K) const {
    return reinterpret_cast<uint32_t*>(base + BP::kMemoMassOffset + K * 4 * 4 * 4);
  }

  __device__ __forceinline__ unsigned short* memo_cols(int K) const {
    return reinterpret_cast<unsigned short*>(base + BP::kMemoColOffset
                                             + K * BP::kMemoColStride * 2);
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

// ------------------------------------------------- the transit and PHASE 1's readout

//: THE STATE IN TRANSIT: the accumulator's bottom halves while the read side owns the top
//: ones. The top halves live in the region for the whole of PHASE 1 -- the readout reads
//: them there as its B operand, stream by stream -- and the bottom halves never leave the
//: register file and are never truncated. -- see docs/internals/carry/carry_kernel.md#transit
template <class BP>
struct Transit {
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

//: THE READ MEMO, derived once per launch by warp `K % warps`: for k-step `K` of the box
//: (owner `K / kDw`, its k-step `K % kDw`) each lane's region row for the B fragment, the
//: four rows the mass lanes read, and the amplitude columns: the inner run's block for
//: each lane half and the outer runs' digit. -- see docs/internals/carry/carry_kernel.md#memo
template <class BP, int D, int BC>
__device__ __forceinline__ void derive_read_memo(const CarryParams& p, const Smem<BP>& sm, int warp,
                                                 int lane) {
  const int rT = ((lane >> 4) & 1) * 8 + (lane & 7), cT = (lane >> 3) & 1;
#pragma unroll 1
  for (int K = warp; K < BP::kKb; K += BP::kWarps) {
    const int ow = K / BP::kDw, kk = K % BP::kDw;
    sm.memo_rows(K)[lane] =
        transit_row<BP, D, BC>(sm, p.g.r, ow, kk * ops::kMmaK + rT) + (uint32_t)(cT * 8 * 2);
    if (lane < 16) {
      const int h = lane >> 3, i = (lane >> 2) & 1, q = lane & 3;
      const int P = kk * ops::kMmaK + 8 * h + 2 * q + i;
      sm.memo_mass(K)[q * 4 + 2 * h + i] =
          transit_row<BP, D, BC>(sm, p.g.r, ow, P) + (uint32_t)(BP::kDv * 2);
    }
    if (lane < 2) {
      const int local = geom_local_row<D, BC>(p.g.r, ow, kk * ops::kMmaK + 8 * lane);
      sm.memo_cols(K)[lane] =
          (unsigned short)(span_base(D, BC, D - 1) + (local & (owner_span(D, BC, D - 1) - 1)));
    }
    if (lane < D - 1) {
      const int head = geom_local_row<D, BC>(p.g.r, ow, kk * ops::kMmaK);
      int col = 0;
      rola::static_for<D - 1>([&](auto Lc) {
        constexpr int l = decltype(Lc)::value;
        if (l == lane)
          col = span_base(D, BC, l)
                + ((head >> owner_run_shift(D, BC, l)) & (owner_span(D, BC, l) - 1));
      });
      sm.memo_cols(K)[2 + lane] = (unsigned short)col;
    }
  }
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

// ------------------------------------------------------------------- PHASE 2's fold

//: THE SENTINEL the read pool's dead row carries as its token: no token, so no deposit.
constexpr int kDeadToken = 0xFFFF;

//: THE POOL: this window's rows keyed by POSITION -- the value row and the box's amplitude
//: runs, the outermost carrying the gain. Filled BY POSITION (no cursor), gathered BY INDEX
//: (`ldmatrix` at per-lane row addresses). Row `kPoolPos` is the DEAD ROW a padded tile reads.
//: -- see docs/internals/carry/carry_kernel.md#pool
constexpr int kRoundTokens = 32;

//: THE DEAD ROW, zeroed once per window because the pool shares the transit's bytes.
//: A SIDE'S BATCH, in positions. -- see docs/internals/carry/smem_ledger.md#pool
template <class BP, int Side>
constexpr int pool_positions_of() {
  return Side == rola::facts::kWrite ? BP::kPoolPos : BP::kReadPos;
}

template <class BP, int Side>
__device__ __forceinline__ void pool_zero_dead_row(const Smem<BP>& sm, int lane) {
  constexpr int kPos = pool_positions_of<BP, Side>();
  constexpr int kVBytes = Side == rola::facts::kWrite ? BP::kVRowBytes : 0;
  constexpr int kWords = (kVBytes + BP::kAmpRowBytes) / 4;
  for (int i = lane; i < kWords; i += ops::kBallotLanes) {
    const uint32_t off = (uint32_t)(i * 4);
    ops::store_shared_u32(off < kVBytes ? sm.pool_v() + (uint32_t)(kPos * BP::kVRowBytes) + off
                                        : sm.pool_amp(Side, kPos) + (off - kVBytes),
                          0u);
  }
  if (lane == 0) {
    if constexpr (Side == rola::facts::kRead)
      sm.pool_token()[kPos] = (unsigned short)kDeadToken;
    else
      sm.pool_gain()[kPos] = 0u;
  }
}

//: ONE POSITION'S ROWS into a slot, by lanes `l0 .. l0 + 7`: the value row's eight quanta
//: and every amplitude run by `cp.async` at its quantum. The gain is the gather's.
template <class BP, int D, int Side>
__device__ __forceinline__ void pool_copy_position(const CarryParams& p, const Smem<BP>& sm,
                                                   const int (&abase)[D], int sub, int bh, int tok,
                                                   int slot) {
  constexpr bool kW = Side == rola::facts::kWrite;
  constexpr int kGroups = BP::kDv / 8;
  static_assert(kGroups <= 8, "a value row is at most eight sixteen-byte quanta, one a lane");
  const __nv_bfloat16* const plane_p = kW ? p.pwrite : p.pread;
  const uint32_t plane = (uint32_t)bh * (uint32_t)p.L * (uint32_t)p.g.wtot * 2u;
  const uint32_t row = plane + (uint32_t)tok * (uint32_t)p.g.wtot * 2u;
  if constexpr (kW) {
    const uint32_t off = ((uint32_t)bh * (uint32_t)p.L + (uint32_t)tok) * (uint32_t)BP::kDv * 2u
                         + (uint32_t)(sub * 16);
    if (sub < kGroups)
      ops::stage_tile(sm.pool_v() + (uint32_t)v_group_offset<BP>(slot, sub),
                      ops::gather_ptr(p.v, off));
  }
  rola::static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
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
                                                int base, int nStreams) {
  unsigned short* const scratch = sm.fill_scratch(Side, warp);
  const int n = __popc(live);
  if ((live >> lane) & 1u) scratch[__popc(live & ops::lanes_below())] = (unsigned short)lane;
  warp_edge(warp);
  const int g = lane >> 3, sub = lane & 7;
  //: two steps an iteration, their address chains in flight together.
#pragma unroll 1
  for (int s = 0; s < n; s += 8) {
    rola::static_for<2>([&](auto Hc) {
      const int idx = s + decltype(Hc)::value * 4 + g;
      if (idx < n) {
        const int slot = prefix + idx - base;
        if (slot >= 0 && slot < pool_positions_of<BP, Side>())
          pool_copy_position<BP, D, Side>(p, sm, abase, sub, bh,
                                          t0 + round * kRoundTokens + (int)scratch[idx], slot);
      }
    });
  }
  const uint32_t below = ops::lanes_below();
  const int rank = prefix + __popc(live & below) - base;
  const bool in = ((live >> lane) & 1u) && rank >= 0 && rank < pool_positions_of<BP, Side>();
  if constexpr (Side == rola::facts::kRead) {
    if (in) sm.pool_token()[rank] = (unsigned short)(round * kRoundTokens + lane);
  } else {
    if (in) sm.pool_gain()[rank] = sm.gain_table()[round * kRoundTokens + lane];
  }
  if constexpr (Side == rola::facts::kWrite) {
#pragma unroll 4
    for (int s = 0; s < nStreams; ++s) {
      const uint32_t w = sm.stream_words(Side, s)[round];
      const int at =
          (int)sm.stream_prefix(Side, s)[round] + __popc(w & below) - sm.stream_base(Side)[s];
      if (in && ((w >> lane) & 1u)) sm.pool_list(s)[at] = (unsigned short)rank;
    }
  }
  warp_edge(warp);
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
  //: the tile's sixteen entries: a list is padded within its own bounds, so no wrap.
  const unsigned short* const tile = sm.pool_list(stream) + taken;
  const int rowg = (lane & 7) + ((lane >> 4) & 1) * 8;
  const int colhalf = (lane >> 3) & 1;
  const int slot = tile[rowg];
  constexpr int kGroups = BP::kDv / 8;
  const uint32_t vrow = (uint32_t)(slot * BP::kVRowBytes), s7 = (uint32_t)(slot & (kGroups - 1));

  rola::static_for<BP::kMCh>([&](auto Mc) {
    constexpr int mc = decltype(Mc)::value;
    if constexpr (mc == BP::kMassTile) {
      mass_frag<BP>(f.vf[mc], lane);
    } else {
      ops::load_frag_t(f.vf[mc],
                       sm.pool_v() + vrow + ((((uint32_t)(2 * mc + colhalf)) ^ s7) * 16u));
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
    r0[h] = tile[2 * q + 8 * h];
    r1[h] = tile[2 * q + 1 + 8 * h];
  });
  //: THE GAIN, one bf16 a position, as the pair of the two tokens a register holds: folded
  //: into the outermost run's digit, or into the fragment itself where there is no run above.
  uint32_t gain[2];
  rola::static_for<2>([&](auto Hc) {
    constexpr int h = decltype(Hc)::value;
    gain[h] = (uint32_t)sm.pool_gain()[r0[h]] | ((uint32_t)sm.pool_gain()[r1[h]] << 16);
  });
  if constexpr (D == 1) {
    rola::static_for<BP::kNL>([&](auto Jc) {
      constexpr int j = decltype(Jc)::value;
      f.bf[j][0] = ops::mul_bf16x2(f.bf[j][0], gain[0]);
      f.bf[j][1] = ops::mul_bf16x2(f.bf[j][1], gain[1]);
    });
  }
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
        if constexpr (l == 0) head[v][h] = ops::mul_bf16x2(head[v][h], gain[h]);
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
  if (i < taken + BP::kSeg) list[i] = (unsigned short)BP::kPoolPos;
}

//: ONE PASS of a batch: a stream's minted list, its whole tiles with each gather in front of
//: the previous deposits, its tail padded to the dead row.
template <class BP, int D, int NS, int WA, int Tile0, int Tiles>
__device__ __forceinline__ void pool_pass(const Smem<BP>& sm, Acc<BP>& acc,
                                          const int (&scol)[D][BP::kNL], int warp, int lane,
                                          int stream, int count) {
  unsigned short* const list = sm.pool_list(stream);
  const int whole = count & ~(BP::kSeg - 1);
  if (count > whole) pool_pad_tile<BP>(list, count, whole, lane);
  const int tiles = (count + BP::kSeg - 1) / BP::kSeg;
  warp_edge(warp);
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
  warp_edge(warp);
}

//: A SIDE'S UNION WORD for round `lane`: the OR of its streams' words, read once per window.
template <class BP>
__device__ __forceinline__ uint32_t union_word(const Smem<BP>& sm, int side, int nStreams,
                                               int nRounds, int lane) {
  //: the loop is warp-uniform; the lane's part in it is the predicate on the OR.
  uint32_t u = 0u;
  const int r = lane < nRounds ? lane : 0;
#pragma unroll 1
  for (int s = 0; s < nStreams; ++s) u |= lane < nRounds ? sm.stream_words(side, s)[r] : 0u;
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
  const int pr = sm.union_prefix(side)[r];
  const uint32_t uw = sm.union_words(side)[r];
  //: the lane of the rank's position: the set bit with `rank - pr` set bits below it.
  const uint32_t below = ops::lanes_below();
  const uint32_t at = ops::lane_ballot(((uw >> lane) & 1u) && __popc(uw & below) == rank - pr);
  return (int)sm.stream_prefix(side, s)[r] + __popc(sm.stream_words(side, s)[r] & (at - 1u));
}

//: THE UNION PUBLISHED: every warp writes the same words and prefix (lane = round), so a
//: warp reads its own after its edge and the total is one uniform load.
template <class BP>
__device__ __forceinline__ int publish_union(const Smem<BP>& sm, int side, uint32_t cta, int prefix,
                                             int nRounds, int warp, int lane) {
  if (lane < nRounds) {
    sm.union_words(side)[lane] = cta;
    sm.union_prefix(side)[lane] = prefix;
  }
  warp_edge(warp);
  return sm.union_prefix(side)[nRounds - 1] + __popc(sm.union_words(side)[nRounds - 1]);
}

//: A BATCH'S STREAM COUNTS PUBLISHED, lane = stream, the same by every warp.
template <class BP>
__device__ __forceinline__ void publish_stream_counts(const Smem<BP>& sm, int side, int sbase,
                                                      int send, int nStreams, int warp, int lane) {
  if (lane < nStreams) {
    sm.stream_base(side)[lane] = sbase;
    sm.stream_end(side)[lane] = send;
  }
  warp_edge(warp);
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
  prefix -= __popc(cta);
  const int total = publish_union<BP>(sm, kSide, cta, prefix, nRounds, warp, lane);

  //: A WINDOW WITH NOTHING LIVE touches the pool not at all; the pool aliases the transit
  //: region, so no warp fills until every warp has rejoined its halves out of it.
  if (total == 0) return;
  ops::stage_wait<0>();
  ops::rendezvous_group((int)Edge::kFoldRound, BP::kThreads);
  if (warp == 0) pool_zero_dead_row<BP, kSide>(sm, lane);

#pragma unroll 1
  for (int base = 0; base < total; base += BP::kPoolPos) {
    const int sbase =
        stream_count_before<BP>(sm, kSide, cta, prefix, total, nRounds, nStreams, lane, base);
    const int send = stream_count_before<BP>(sm, kSide, cta, prefix, total, nRounds, nStreams, lane,
                                             base + BP::kPoolPos);
    publish_stream_counts<BP>(sm, kSide, sbase, send, nStreams, warp, lane);
#pragma unroll 1
    for (int r = warp; r < nRounds; r += BP::kWarps)
      pool_fill_round<BP, D, kSide>(p, sm, abase, warp, lane, bh, t0, r, sm.union_words(kSide)[r],
                                    sm.union_prefix(kSide)[r], base, nStreams);
    ops::stage_wait<0>();
    ops::rendezvous_group((int)Edge::kFoldRound, BP::kThreads);

    rola::static_for<passes>([&](auto Pc) {
      constexpr int pass = decltype(Pc)::value;
      const int s = warp * passes + pass;
      pool_pass<BP, D, NS, WA, pass * kTilesPerPass, kTilesPerPass>(
          sm, acc, scol, warp, lane, s, sm.stream_end(kSide)[s] - sm.stream_base(kSide)[s]);
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
      + (t0 >> 5) + (lane < nWords ? lane : 0);
  //: EVERY LEVEL'S RUN IN ONE TRIP: each run's words are loaded sixteen at a time, in two
  //: eight-blocks under one uniform branch (a short run predicates its first block), and
  //: no run is OR-ed until every run's loads are out -- the walk waits once, not once a
  //: block. The rows are L2 hits at every length measured; the wait is their latency.
  int count[D];
  const int32_t* row[D];
  rola::static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    const int room = gbits - sg.sub_shift[l];
    count[l] = 1 << (room <= 0 ? 0 : (room < sg.sub_bits[l] ? room : sg.sub_bits[l]));
    const int digit = (local >> owner_run_shift(D, BC, l)) & (owner_span(D, BC, l) - 1);
    row[l] = words + (long)(abase[l] + digit) * p.liveness_words;
  });
  uint32_t o[D];
  rola::static_for<D>([&](auto Lc) { o[decltype(Lc)::value] = 0u; });

  int left[D];
  rola::static_for<D>([&](auto Lc) { left[decltype(Lc)::value] = count[decltype(Lc)::value]; });
  bool more = true;
  const long stride = p.liveness_words;
#pragma unroll 1
  while (more) {
    more = false;
    uint32_t v[D][16];
    rola::static_for<D>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      //: one pointer stepped a row at a time: one add a load, the stride never multiplied.
      const int32_t* q = row[l];
      rola::static_for<8>([&](auto Jc) {
        constexpr int j = decltype(Jc)::value;
        v[l][j] = j < left[l] ? (uint32_t)*q : 0u;
        q += stride;
      });
      if (left[l] > 8) {
        rola::static_for<8>([&](auto Jc) {
          v[l][8 + decltype(Jc)::value] = (uint32_t)*q;
          q += stride;
        });
      } else {
        rola::static_for<8>([&](auto Jc) { v[l][8 + decltype(Jc)::value] = 0u; });
      }
      row[l] = q;
      more |= left[l] > 16;
      left[l] -= 16;
    });

    rola::static_for<D>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      rola::static_for<16>([&](auto Jc) { o[l] |= v[l][decltype(Jc)::value]; });
    });
  }

  uint32_t m = lane < nWords ? 0xFFFFFFFFu : 0u;
  rola::static_for<D>([&](auto Lc) { m &= o[decltype(Lc)::value]; });
  return m;
}

//: PHASE 1 AT THE POSITION CARVE: a warp owns sixteen consecutive positions of the batch
//: and every channel of their rows, in its MMA accumulator across every stream -- each
//: stream's coefficients gathered once, its leaves read from the region as they lie -- and
//: the rows leave once a batch, drained under the MMAs that follow.
//: -- see docs/internals/carry/carry_kernel.md#readout

//: THE STAGING TILE: `[position][segment | mass a slice]` fp32 rows at a `32 + 8` stride (a
//: fragment's 64-bit pair stores bank free, a position's segment contiguous), holding one
//: 32-channel segment of one slice at a time, and a seventeenth row of the slices' token
//: ids by (slice, batch parity): a batch's own drains and the batch before's pending one
//: are in flight together.
template <class BP>
__device__ __forceinline__ constexpr uint32_t stage_ids(int slice, int parity) {
  return (uint32_t)(BP::kSeg * BP::kStageStride * 4 + (slice * 2 + parity) * BP::kSeg * 2);
}

//: A SLICE'S POSITION `i` for warp `w`: the slice's run of the batch, the warp's sixteen.
template <class BP>
__device__ __forceinline__ constexpr int slice_pos(int warp, int slice, int i) {
  return slice * BP::kWarps * BP::kSeg + warp * BP::kSeg + i;
}

//: ONE DRAIN GROUP: four positions of the staged segment leave as one reduction each of 32
//: consecutive floats -- one line an instruction, which is what the LSU's global queue
//: admits. The group's ids are one 8-byte load; its four values load before its four
//: reductions. A position past the batch's count carries the dead token and is skipped.
template <class BP, bool Full>
__device__ __forceinline__ void drain_group(ops::SmemAddr stage, float* numl, int slice, int parity,
                                            int seg, int g, int lane) {
  constexpr int kS = BP::kStageStride;
  const uint2 tw = ops::load_shared_u64(stage + stage_ids<BP>(slice, parity) + (uint32_t)(g * 8));
  int tok[4];
  float v[4];
  rola::static_for<4>([&](auto Tc) {
    constexpr int i = decltype(Tc)::value;
    tok[i] = (int)(((i < 2 ? tw.x : tw.y) >> (16 * (i & 1))) & 0xFFFFu);
    v[i] = __uint_as_float(ops::load_shared_u32(stage + (uint32_t)(((4 * g + i) * kS + lane) * 4)));
  });
  rola::static_for<4>([&](auto Tc) {
    constexpr int i = decltype(Tc)::value;
    if constexpr (!Full)
      if (tok[i] == kDeadToken) return;
    ops::fan_in_add(numl + (uint32_t)tok[i] * (uint32_t)BP::kDv + seg * 32, v[i]);
  });
}

template <class BP, bool Full>
__device__ __forceinline__ void drain_mass(ops::SmemAddr stage, float* den, int slice, int parity,
                                           int t0, int lane) {
  constexpr int kS = BP::kStageStride;
  const int tok =
      (int)ops::load_shared_u16(stage + stage_ids<BP>(slice, parity) + (uint32_t)((lane & 15) * 2));
  const float m = __uint_as_float(
      ops::load_shared_u32(stage + (uint32_t)(((lane & (BP::kSeg - 1)) * kS + 32 + slice) * 4)));
  float* const mrow = reinterpret_cast<float*>(ops::gather_ptr(den, (uint32_t)(t0 + tok) * 4u));
  if constexpr (Full) {
    ops::fan_in_add(mrow, lane < BP::kSeg ? m : 0.0f);
  } else {
    if (lane < BP::kSeg && tok != kDeadToken) ops::fan_in_add(mrow, m);
  }
}

//: A SEGMENT STAGED: four n-tiles' fragments to the tile's rows, eight 64-bit stores a lane.
template <class BP, int Seg>
__device__ __forceinline__ void stage_segment(ops::SmemAddr stage,
                                              const float (&c)[BP::kMassTile * 2][4], int lane) {
  constexpr int kS = BP::kStageStride;
  const int r = lane >> 2, q = lane & 3;
  rola::static_for<4>([&](auto Nc) {
    constexpr int j = Seg * 4 + decltype(Nc)::value;
    const uint32_t col = (uint32_t)((decltype(Nc)::value * ops::kMmaN + 2 * q) * 4);
    ops::store_shared_f32x2(stage + (uint32_t)(r * kS * 4) + col, c[j][0], c[j][1]);
    ops::store_shared_f32x2(stage + (uint32_t)((r + 8) * kS * 4) + col, c[j][2], c[j][3]);
  });
}

template <class BP>
__device__ __forceinline__ void stage_mass(ops::SmemAddr stage, const float (&cm)[4], int slice,
                                           int lane) {
  constexpr int kS = BP::kStageStride;
  if ((lane & 3) == 0) {
    ops::store_shared_u32(stage + (uint32_t)(((lane >> 2) * kS + 32 + slice) * 4),
                          __float_as_uint(cm[0]));
    ops::store_shared_u32(stage + (uint32_t)((((lane >> 2) + 8) * kS + 32 + slice) * 4),
                          __float_as_uint(cm[2]));
  }
}

//: THE PENDING SEGMENT AND THE MASSES of the batch before: the last slice's last segment
//: and every slice's mass, drained at the phase's end.
template <class BP, bool Full>
__device__ __forceinline__ void drain_pending(ops::SmemAddr stage, float* numl, float* den,
                                              uint32_t seen, int parity, int t0, int lane) {
  rola::static_for<BP::kSlices>([&](auto Sc) {
    constexpr int sl = decltype(Sc)::value;
    if ((seen >> sl) & 1u) drain_mass<BP, Full>(stage, den, sl, parity, t0, lane);
  });
  if ((seen >> (BP::kSlices - 1)) & 1u) {
    rola::static_for<BP::kSeg / 4>([&](auto Gc) {
      drain_group<BP, Full>(stage, numl, BP::kSlices - 1, parity, BP::kDv / 32 - 1,
                            decltype(Gc)::value, lane);
    });
  }
}

//: THE SLICE'S A FRAGMENTS for one stream: the inner run's sixteen digits by `ldmatrix`
//: once per distinct inner block -- a stream's k-steps walk the innermost digits fastest,
//: so `kKs` k-steps share `kBlocks` blocks -- and the outer runs' digit per k-step as one
//: packed multiply per fragment register. A position past the batch's count reads the
//: dead row.
template <class BP, int D, int BC, int kKs, int kBlocks>
__device__ __forceinline__ void slice_gather(const Smem<BP>& sm, int warp, int slice, int count,
                                             const int (&rcin)[kBlocks],
                                             const int (&rcout)[D - 1][kKs], int lane,
                                             uint32_t (&a)[kKs][4]) {
  constexpr int kSide = rola::facts::kRead;
  const int r = lane >> 2;
  const int iA = slice_pos<BP>(warp, slice, lane & 15), i0 = slice_pos<BP>(warp, slice, r),
            i1 = slice_pos<BP>(warp, slice, r + 8);
  const ops::SmemAddr rowA = sm.pool_amp(kSide, iA < count ? iA : BP::kReadPos);
  const ops::SmemAddr row0 = sm.pool_amp(kSide, i0 < count ? i0 : BP::kReadPos);
  const ops::SmemAddr row1 = sm.pool_amp(kSide, i1 < count ? i1 : BP::kReadPos);
  uint32_t in[kBlocks][4];
  rola::static_for<kBlocks>([&](auto Bc) {
    constexpr int b = decltype(Bc)::value;
    ops::load_frag(in[b], rowA + (uint32_t)(rcin[b] * 2));
  });
  rola::static_for<kKs>([&](auto Kc) {
    constexpr int i = decltype(Kc)::value;
    constexpr int b = i % kBlocks;
    if constexpr (D == 1) {
      rola::static_for<4>([&](auto Ic) { a[i][decltype(Ic)::value] = in[b][decltype(Ic)::value]; });
    } else {
      uint32_t h0 = 0u, h1 = 0u;
      rola::static_for<D - 1>([&](auto Lc) {
        constexpr int l = decltype(Lc)::value;
        const uint32_t d0 =
            ops::splat_bf16x2_bits(ops::load_shared_u16(row0 + (uint32_t)(rcout[l][i] * 2)));
        const uint32_t d1 =
            ops::splat_bf16x2_bits(ops::load_shared_u16(row1 + (uint32_t)(rcout[l][i] * 2)));
        h0 = l == 0 ? d0 : ops::mul_bf16x2(h0, d0);
        h1 = l == 0 ? d1 : ops::mul_bf16x2(h1, d1);
      });
      a[i][0] = ops::mul_bf16x2(in[b][0], h0);
      a[i][2] = ops::mul_bf16x2(in[b][2], h0);
      a[i][1] = ops::mul_bf16x2(in[b][1], h1);
      a[i][3] = ops::mul_bf16x2(in[b][3], h1);
    }
  });
}

//: THE STREAM'S OPERANDS off the memo: the columns, each lane's region row, the mass rows.
template <class BP, int D, int kKs, int kBlocks>
struct StreamOperands {
  int rcin[kBlocks];
  int rcout[D - 1][kKs];
  ops::SmemAddr rows[kKs];
  uint32_t bm[kKs][2];
};

template <class BP, int D, int kKs, int kBlocks>
__device__ __forceinline__ void stream_operands(const Smem<BP>& sm, int K0, int lane,
                                                StreamOperands<BP, D, kKs, kBlocks>& o) {
  rola::static_for<kKs>([&](auto Kc) {
    constexpr int i = decltype(Kc)::value;
    const unsigned short* const cols = sm.memo_cols(K0 + i);
    if constexpr (i < kBlocks) o.rcin[i] = cols[(lane >> 4) & 1];
    rola::static_for<D - 1>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      o.rcout[l][i] = cols[2 + l];
    });
    o.rows[i] = sm.memo_rows(K0 + i)[lane];
    o.bm[i][0] = o.bm[i][1] = 0u;
    if ((lane >> 2) == 0) {
      const uint4 mr = *reinterpret_cast<const uint4*>(sm.memo_mass(K0 + i) + (lane & 3) * 4);
      o.bm[i][0] =
          (uint32_t)ops::load_shared_u16(mr.x) | ((uint32_t)ops::load_shared_u16(mr.y) << 16);
      o.bm[i][1] =
          (uint32_t)ops::load_shared_u16(mr.z) | ((uint32_t)ops::load_shared_u16(mr.w) << 16);
    }
  });
}

//: ONE M-TILE of one slice: the slice's two n-tiles and, at m-tile 0, its mass.

template <class BP, int kKs, int M>
__device__ __forceinline__ void slice_mtile(const uint32_t (&a)[kKs][4],
                                            const uint32_t (&b4)[kKs][4],
                                            const uint32_t (&bm)[kKs][2],
                                            float (&c)[BP::kMassTile * 2][4], float (&cm)[4]) {
  rola::static_for<kKs>([&](auto Kc) {
    constexpr int i = decltype(Kc)::value;
    const uint32_t b0[2] = {b4[i][0], b4[i][2]};
    const uint32_t b1[2] = {b4[i][1], b4[i][3]};
    ops::mma(c[2 * M], a[i], b0);
    ops::mma(c[2 * M + 1], a[i], b1);
    if constexpr (M == 0) ops::mma(cm, a[i], bm[i]);
  });
}

//: ONE STREAM into every slice's accumulator, m-tile by m-tile over the region's rows for
//: the stream's k-steps: one load of the rows serves every slice. `pend`: the batch before's
//: pending segment and masses drain under this, the first live stream's, m-tiles.
template <class BP, int D, int BC, int GB, uint32_t Lives>
__device__ __forceinline__ void slice_stream(const Smem<BP>& sm, int s, int warp, int count,
                                             float (&c)[BP::kSlices][BP::kMassTile * 2][4],
                                             float (&cm)[BP::kSlices][4], int lane) {
  constexpr int kPasses = BP::kLeavesPerWarp >> GB;
  constexpr int kKs = (1 << GB) / ops::kMmaK;
  constexpr int kSpanIn = owner_span(D, BC, D - 1) / ops::kMmaK;
  constexpr int kBlocks = kKs < kSpanIn ? kKs : kSpanIn;
  static_assert(BP::kMassTile % 2 == 0, "a slice's row is whole segments of two m-tiles");
  const int K0 = (s / kPasses) * BP::kDw + (s % kPasses) * kKs;
  StreamOperands<BP, D, kKs, kBlocks> o;
  stream_operands<BP, D, kKs, kBlocks>(sm, K0, lane, o);
  uint32_t a[BP::kSlices][kKs][4];
  rola::static_for<BP::kSlices>([&](auto Sc) {
    constexpr int sl = decltype(Sc)::value;
    if constexpr ((Lives >> sl) & 1u)
      slice_gather<BP, D, BC, kKs, kBlocks>(sm, warp, sl, count, o.rcin, o.rcout, lane, a[sl]);
  });

  //: one load of the B fragments an m-tile, serving every slice; the pending groups spread
  //: over the m-tiles.
  rola::static_for<BP::kMassTile>([&](auto Mc) {
    constexpr int m = decltype(Mc)::value;
    uint32_t b4[kKs][4];
    rola::static_for<kKs>([&](auto Kc) {
      constexpr int i = decltype(Kc)::value;
      ops::load_frag_t(b4[i], o.rows[i] + (uint32_t)(m * ops::kMmaM * 2));
    });
    rola::static_for<BP::kSlices>([&](auto Sc) {
      constexpr int sl = decltype(Sc)::value;
      if constexpr ((Lives >> sl) & 1u) slice_mtile<BP, kKs, m>(a[sl], b4, o.bm, c[sl], cm[sl]);
    });
  });
}

//: THE LAST LIVE STREAM, slice by slice, so one staging buffer drains every segment
//: progressively: a slice's m-tiles run, each segment staged as its m-tile pair ends and
//: drained under the m-tiles after it -- the next slice's when its own are spent -- and
//: the last slice's last segment, with the masses, left pending for the next batch.
template <class BP, int D, int BC, int GB, int Slice, bool Full, bool Here>
__device__ __forceinline__ void last_stream_slice(
    const Smem<BP>& sm, int warp, int count, uint32_t seen, int parity,
    const StreamOperands<BP, D, (1 << GB) / ops::kMmaK,
                         ((1 << GB) / ops::kMmaK) < owner_span(D, BC, D - 1) / ops::kMmaK
                             ? (1 << GB) / ops::kMmaK
                             : owner_span(D, BC, D - 1) / ops::kMmaK>& o,
    float (&c)[BP::kMassTile * 2][4], float (&cm)[4], ops::SmemAddr stage, float* numl, int lane) {
  constexpr int kKs = (1 << GB) / ops::kMmaK;
  constexpr int kSpanIn = owner_span(D, BC, D - 1) / ops::kMmaK;
  constexpr int kBlocks = kKs < kSpanIn ? kKs : kSpanIn;
  constexpr int kSegs = BP::kDv / 32;
  const bool live = (seen >> Slice) & 1u;

  uint32_t a[kKs][4];
  if constexpr (Here)
    slice_gather<BP, D, BC, kKs, kBlocks>(sm, warp, Slice, count, o.rcin, o.rcout, lane, a);
  uint32_t b4[2][kKs][4];
  rola::static_for<kKs>(
      [&](auto Kc) { ops::load_frag_t(b4[0][decltype(Kc)::value], o.rows[decltype(Kc)::value]); });
  rola::static_for<BP::kMassTile>([&](auto Mc) {
    constexpr int m = decltype(Mc)::value;
    if constexpr (m + 1 < BP::kMassTile) {
      rola::static_for<kKs>([&](auto Kc) {
        constexpr int i = decltype(Kc)::value;
        ops::load_frag_t(b4[(m + 1) & 1][i], o.rows[i] + (uint32_t)((m + 1) * ops::kMmaM * 2));
      });
    }
    if constexpr (Here) slice_mtile<BP, kKs, m>(a, b4[m & 1], o.bm, c, cm);
    //: under m-tiles 0-1 of a later slice: the previous slice's last segment, staged at
    //: its end.
    if constexpr (Slice > 0 && m < 2) {
      if ((seen >> (Slice - 1)) & 1u) {
        rola::static_for<2>([&](auto Gc) {
          drain_group<BP, Full>(stage, numl, Slice - 1, parity, kSegs - 1,
                                m * 2 + decltype(Gc)::value, lane);
        });
      }
    }
    //: this slice's segments: staged as their m-tile pair ends, all but the last drained
    //: under the two m-tiles after; the last staged at the end, for the next slice or batch.
    if constexpr (m % 2 == 1 && m / 2 < kSegs - 1) {
      if (live) {
        warp_edge(warp);
        stage_segment<BP, m / 2>(stage, c, lane);
        warp_edge(warp);
      }
    }
    if constexpr (m >= 2) {
      constexpr int seg = m / 2 - 1;
      if (live) {
        drain_group<BP, Full>(stage, numl, Slice, parity, seg, 2 * (m & 1), lane);
        drain_group<BP, Full>(stage, numl, Slice, parity, seg, 2 * (m & 1) + 1, lane);
      }
    }
  });
  if (live) {
    warp_edge(warp);
    stage_segment<BP, kSegs - 1>(stage, c, lane);
    stage_mass<BP>(stage, cm, Slice, lane);
    warp_edge(warp);
  }
}

template <class BP, int D, int BC, int GB, bool Full0, bool Full1, uint32_t Lives>
__device__ __forceinline__ void last_stream(const Smem<BP>& sm, int s, int warp, int count,
                                            uint32_t seen, int parity,
                                            float (&c)[BP::kSlices][BP::kMassTile * 2][4],
                                            float (&cm)[BP::kSlices][4], ops::SmemAddr stage,
                                            float* numl, int lane) {
  constexpr int kPasses = BP::kLeavesPerWarp >> GB;
  constexpr int kKs = (1 << GB) / ops::kMmaK;
  constexpr int kSpanIn = owner_span(D, BC, D - 1) / ops::kMmaK;
  constexpr int kBlocks = kKs < kSpanIn ? kKs : kSpanIn;
  const int K0 = (s / kPasses) * BP::kDw + (s % kPasses) * kKs;
  StreamOperands<BP, D, kKs, kBlocks> o;
  stream_operands<BP, D, kKs, kBlocks>(sm, K0, lane, o);
  rola::static_for<BP::kSlices>([&](auto Sc) {
    constexpr int sl = decltype(Sc)::value;
    constexpr bool kFull = sl == 0 ? Full0 : Full1;
    last_stream_slice<BP, D, BC, GB, sl, kFull, ((Lives >> sl) & 1u) != 0u>(
        sm, warp, count, seen, parity, o, c[sl], cm[sl], stage, numl, lane);
  });
}

//: PHASE 1: the read pool -- the window's live positions' amplitude rows, by rank, in
//: batches of `kSlices` slices a warp -- filled by the same walk as the fold's; then each
//: warp its slices over every stream, against the region; then the slices' rows out. The
//: phase crosses the CTA at the batch edges only: after a fill, and before the next.
//: -- see docs/internals/carry/carry_kernel.md#readout
template <class BP, int D, int BC, int GB>
__device__ __forceinline__ void readout_phase_pool(const CarryParams& p, const Smem<BP>& sm,
                                                   const int (&abase)[D], float* num, float* den,
                                                   int warp, int lane, int bh, int t0, int wLen) {
  constexpr int kSide = rola::facts::kRead;
  constexpr int kNCh = BP::kMassTile * 2;
  const int nRounds = (wLen + kRoundTokens - 1) / kRoundTokens;
  constexpr int kPasses = BP::kLeavesPerWarp >> GB;
  constexpr int kStreams = BP::kWarps * kPasses;
  const uint32_t cta = union_word<BP>(sm, kSide, kStreams, nRounds, lane);
  int prefix = __popc(cta);
  rola::static_for<5>([&](auto Sc) {
    constexpr int d = 1 << decltype(Sc)::value;
    const int up = __shfl_up_sync(0xFFFFFFFFu, prefix, d);
    if (lane >= d) prefix += up;
  });
  prefix -= __popc(cta);
  const int total = publish_union<BP>(sm, kSide, cta, prefix, nRounds, warp, lane);

  //: A WINDOW WITH NOTHING LIVE touches the pool not at all. No warp fills until every warp
  //: has published its transit rows.
  if (total == 0) return;
  ops::rendezvous_group((int)StreamEdge::kReadBatch, BP::kThreads);
  if (warp == 0) pool_zero_dead_row<BP, kSide>(sm, lane);

  const ops::SmemAddr stage = sm.stage(warp);
  float* const numl = num + (size_t)t0 * BP::kDv + lane;
  uint32_t pend = 0u;
#pragma unroll 1
  for (int base = 0; base < total; base += BP::kReadPos) {
    const int count = total - base < BP::kReadPos ? total - base : BP::kReadPos;
#pragma unroll 1
    for (int r = warp; r < nRounds; r += BP::kWarps)
      pool_fill_round<BP, D, kSide>(p, sm, abase, warp, lane, bh, t0, r, sm.union_words(kSide)[r],
                                    sm.union_prefix(kSide)[r], base, kStreams);
    ops::stage_wait<0>();
    ops::rendezvous_group((int)StreamEdge::kReadBatch, BP::kThreads);

    //: the streams live in each slice: lane `s` answers for stream `s`; the ids per slice.
    const int parity = (base / BP::kReadPos) & 1;
    uint32_t live[BP::kSlices];
    bool full[BP::kSlices];
    static_assert(BP::kSlices == 1 || BP::kSlices == 2, "a warp owns one or two slices");
    rola::static_for<BP::kSlices>([&](auto Sc) {
      constexpr int sl = decltype(Sc)::value;
      const int c0 = stream_count_before<BP>(sm, kSide, cta, prefix, total, nRounds, kStreams, lane,
                                             base + slice_pos<BP>(warp, sl, 0));
      const int c1 = stream_count_before<BP>(sm, kSide, cta, prefix, total, nRounds, kStreams, lane,
                                             base + slice_pos<BP>(warp, sl, BP::kSeg));
      live[sl] = ops::lane_ballot(lane < kStreams && c1 > c0);
      full[sl] = slice_pos<BP>(warp, sl, BP::kSeg - 1) < count;
      if (lane < BP::kSeg) {
        const int i = slice_pos<BP>(warp, sl, lane);
        ops::store_shared_u16(stage + stage_ids<BP>(sl, parity) + (uint32_t)(lane * 2),
                              (uint16_t)sm.pool_token()[i < count ? i : BP::kReadPos]);
      }
    });
    warp_edge(warp);
    const uint32_t any = BP::kSlices == 2 ? live[0] | live[BP::kSlices - 1] : live[0];
    const uint32_t seen =
        (live[0] != 0u ? 1u : 0u) | (BP::kSlices == 2 && live[BP::kSlices - 1] != 0u ? 2u : 0u);
    //: the batch before's pending segment and masses, out before this batch's streams.
    if (pend != 0u) {
      drain_pending<BP, true>(stage, numl, den, pend, parity ^ 1, t0, lane);
      warp_edge(warp);
    }

    if (any == 0u) {
      pend = 0u;
    } else {
      const int first = __ffs((int)any) - 1, last = 31 - __clz((int)any);
      float c[BP::kSlices][kNCh][4], cm[BP::kSlices][4];
      rola::static_for<BP::kSlices>([&](auto Sc) {
        constexpr int sl = decltype(Sc)::value;
        rola::static_for<kNCh>([&](auto Jc) {
          rola::static_for<4>(
              [&](auto Ic) { c[sl][decltype(Jc)::value][decltype(Ic)::value] = 0.0f; });
        });
        rola::static_for<4>([&](auto Ic) { cm[sl][decltype(Ic)::value] = 0.0f; });
      });
#pragma unroll 1
      for (int s = first; s < last; ++s) {
        if (((any >> s) & 1u) == 0u) continue;
        const uint32_t lives = ((live[0] >> s) & 1u)
                               | (BP::kSlices == 2 ? ((live[BP::kSlices - 1] >> s) & 1u) << 1 : 0u);
        if (lives == 3u)
          slice_stream<BP, D, BC, GB, 3u>(sm, s, warp, count, c, cm, lane);
        else if (lives == 1u)
          slice_stream<BP, D, BC, GB, 1u>(sm, s, warp, count, c, cm, lane);
        else
          slice_stream<BP, D, BC, GB, 2u>(sm, s, warp, count, c, cm, lane);
      }
      {
        const uint32_t lives =
            ((live[0] >> last) & 1u)
            | (BP::kSlices == 2 ? ((live[BP::kSlices - 1] >> last) & 1u) << 1 : 0u);
        //: the last stream by its slices' fullness and liveness, each a compiled body.
        const auto run = [&](auto Fc) {
          constexpr int F = decltype(Fc)::value;
          constexpr bool F0 = F >= 1, F1 = F >= 2;
          if (lives == 3u)
            last_stream<BP, D, BC, GB, F0, F1, 3u>(sm, last, warp, count, seen, parity, c, cm,
                                                   stage, numl, lane);
          else if (lives == 1u)
            last_stream<BP, D, BC, GB, F0, F1, 1u>(sm, last, warp, count, seen, parity, c, cm,
                                                   stage, numl, lane);
          else
            last_stream<BP, D, BC, GB, F0, F1, 2u>(sm, last, warp, count, seen, parity, c, cm,
                                                   stage, numl, lane);
        };
        if (BP::kSlices == 2 && full[BP::kSlices - 1])
          run(std::integral_constant<int, 2>{});
        else if (full[0])
          run(std::integral_constant<int, 1>{});
        else
          run(std::integral_constant<int, 0>{});
      }
      pend = seen;
    }
    ops::rendezvous_group((int)StreamEdge::kReadBatch, BP::kThreads);
  }
  //: the window's last batch, drained with nothing to hide under.
  if (pend != 0u) {
    const int parity = ((total - 1) / BP::kReadPos) & 1;
    const int count = ((total - 1) % BP::kReadPos) + 1;
    if (slice_pos<BP>(warp, BP::kSlices - 1, BP::kSeg - 1) < count)
      drain_pending<BP, true>(stage, numl, den, pend, parity, t0, lane);
    else
      drain_pending<BP, false>(stage, numl, den, pend, parity, t0, lane);
    warp_edge(warp);
  }
}

//: THE WINDOW'S GAIN TABLE at cover slot B: the write side's gains for the window's tokens,
//: sixteen bytes a thread by `cp.async`, the tail past the last whole quantum by halfword.
//: Landed by the fold's first `stage_wait`, before its first fill.
template <class BP>
__device__ __forceinline__ void stage_gain_table(const CarryParams& p, const Smem<BP>& sm, int tid,
                                                 int bh, int t0, int wLen) {
  const unsigned short* const src =
      reinterpret_cast<const unsigned short*>(p.gwrite) + (size_t)bh * p.L + t0;
  const int whole = wLen / 8;
  if (tid < whole)
    ops::stage_tile(ops::smem_addr(sm.gain_table()) + (uint32_t)(tid * 16), src + tid * 8);
  if (tid < wLen - whole * 8) sm.gain_table()[whole * 8 + tid] = src[whole * 8 + tid];
  ops::stage_commit();
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

template <class BP, int D, int BC, int NS, int WA, int GR, int GW>
__device__ __forceinline__ void window_loop(const CarryParams& p, const Smem<BP>& sm, Acc<BP>& acc,
                                            int owner, float* num, float* den, int warp, int lane,
                                            int bh) {
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

#pragma unroll 1
  for (int win = 0; win < nWindows; ++win) {
    const int t0 = win * kWindow;
    const int wLen = p.L - t0 < kWindow ? p.L - t0 : kWindow;
    const int nWords = (wLen + ops::kBallotLanes - 1) / ops::kBallotLanes;
    Transit<BP> tr;

    //: COVER SLOT A: routing-only work in front of the out leg's edge -- the read streams'
    //: live words, which read the class-1 words and nothing else.
    out_leg_publish<BP, D, BC>(p, sm, acc, tr, warp, lane);
    int abase[D];
    factor_runs(abase);
    cover_live_words<BP, D, BC>(p, sm, rola::facts::kRead, p.g.r, abase, p.readout_grain_bits, warp,
                                lane, bh, t0, nWords);
    ops::rendezvous_group((int)Edge::kOutLeg, BP::kThreads);

    readout_phase_pool<BP, D, BC, GR>(p, sm, abase, num, den, warp, lane, bh, t0, wLen);

    //: COVER SLOT B: the fold streams' live words, in front of the return leg's edge.
    int wbase[D];
    factor_runs(wbase);
    cover_live_words<BP, D, BC>(p, sm, rola::facts::kWrite, p.g.w, wbase, p.fold_grain_bits, warp,
                                lane, bh, t0, nWords);
    stage_gain_table<BP>(p, sm, warp * 32 + lane, bh, t0, wLen);
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

    fold_phase<BP, D, NS, WA, GW>(p, sm, acc, scol, wbase, warp, lane, bh, t0, wLen);

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

//: THE GRAINS ARE TEMPLATE AXES of the kernel symbol, not a switch inside it: a symbol
//: holding two bodies a phase crosses ptxas's size cliff, past which every shuffle is
//: fenced and outlined into a subroutine whose convention holds forty registers.
//: -- see docs/internals/carry/carry_kernel.md#forms
template <int D, int DV, int WARPS, int GR, int GW>
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
  const int warp = ops::uniform_warp<BP::kWarps>(), lane = tid & 31;

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

  //: THE BOX'S PAGE SLOTS -- the only page-table reads in the launch. The frame's edges and
  //: the streams' are NAMED BARRIERS and take neither bytes nor an initialiser.
  //: -- see docs/internals/carry/carry_kernel.md#edges
  if (tid < BP::kAtoms) {
    const uint32_t canon =
        (uint32_t)(geom_canon_leaf_t<D, BC>(p.g, owner, tid * kAtomLeaves)) >> kAtomLeavesBits;
    sm.page_slots()[tid] = ops::page_slot(p.page_table, atom_base + canon);
  }
  derive_read_memo<BP, D, BC>(p, sm, warp, lane);
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

    //: THE TWO PHASES' GRAINS are the only switches left inside the loop, one per phase.
    window_loop<BP, D, BC, NS, WA, GR, GW>(p, sm, acc, owner, num, den, warp, lane, bh);

    if (p.state_out != nullptr)
      state_sweep<BP, D, NS, WA, false>(p, sm, acc, slots, act_read, act_write, warp, lane, tid,
                                        nullptr, p.state_out);
  });
}

}  // namespace rola::carry
