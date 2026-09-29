// csrc/rola/src/prepass/prepass_kernel.cuh -- THE PRE-PASS KERNEL: the int8 carry's skeleton at two
// columns a leaf. THE FRAME IS REAL AND THE COMPONENTS ARE STUBS WITH THEIR EDGES (KERNEL_STANDARDS
// section 19): the words staged a window ahead, the list counts, the pool filled a chunk ahead through
// two barriered slots, the fold over the warp's lists, the publish and rescale across the transfer, the
// token-major readout into the panel, the panel drained by RED.F32. Each component lands against its own
// budget line in the queue card; until it does, its stub touches its operands and produces zero.
// Every register array is indexed at compile time (section 20): no local memory.
// See docs/internals/prepass/prepass.md
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "common/geom.cuh"
#include "common/ops.cuh"
#include "common/phase_clock.cuh"
#include "common/static_for.cuh"
#include "facts/liveness_contract.cuh"
#include "prepass/params.cuh"
#include "prepass/plan.cuh"

namespace rola::prepass {

namespace ops = denseref::ops;
using Clock = rola::PhaseClock<kPpPhases>;

//: THE SHARED LEDGER'S ADDRESSES, from the plan's offsets. -- see docs/internals/prepass/prepass.md#smem
template <class TP>
struct Smem {
  char* base;

  __device__ __forceinline__ ops::SmemAddr at(int off) const { return ops::smem_addr(base + off); }

  __device__ __forceinline__ ops::SmemAddr words(int parity) const {
    return at(TP::kWordsOffset + parity * TP::kWordsBytes);
  }

  __device__ __forceinline__ ops::SmemAddr pool(int slot) const {
    return at(TP::kPoolOffset + slot * TP::kPoolSlotBytes);
  }

  __device__ __forceinline__ ops::SmemAddr ring(int warp, int slot) const {
    return at(TP::kRingOffset + (warp * TP::kRingSlots + slot) * TP::kRingSlotBytes);
  }

  __device__ __forceinline__ ops::SmemAddr xfer_values() const { return at(TP::kXferOffset); }

  __device__ __forceinline__ ops::SmemAddr xfer_scales() const {
    return at(TP::kXferOffset + TP::kXferValueBytes);
  }

  __device__ __forceinline__ ops::SmemAddr panel(int buf, int warp) const {
    return at(TP::kPanelOffset + (buf * TP::kWarps + warp) * TP::kPanelRowBytes);
  }

  __device__ __forceinline__ ops::SmemAddr state(int warp) const {
    return at(TP::kStateOffset + warp * TP::kStateBytes);
  }

  __device__ __forceinline__ ops::SmemAddr pool_full(int slot) const {
    return at(TP::kBarOffset + slot * 8);
  }

  __device__ __forceinline__ ops::SmemAddr pool_empty(int slot) const {
    return at(TP::kBarOffset + (TP::kPoolSlots + slot) * 8);
  }

  __device__ __forceinline__ ops::SmemAddr ring_bar(int warp, int slot) const {
    return at(TP::kBarOffset + (TP::kPoolBars + warp * TP::kRingSlots + slot) * 8);
  }

  __device__ __forceinline__ ops::SmemAddr words_bar(int parity) const {
    return at(TP::kBarOffset + (TP::kPoolBars + TP::kRingBars + parity) * 8);
  }

  __device__ __forceinline__ ops::SmemAddr pub_bar(int parity) const {
    return at(TP::kBarOffset + (TP::kPoolBars + TP::kRingBars + 2 + parity) * 8);
  }

  //: THE RANGE BARRIERS, a panel buffer each: its rows written (one arrival a warp), its drain done.
  __device__ __forceinline__ ops::SmemAddr rows_bar(int buf) const {
    return at(TP::kBarOffset + (TP::kPoolBars + TP::kRingBars + 4 + buf) * 8);
  }

  __device__ __forceinline__ ops::SmemAddr drained_bar(int buf) const {
    return at(TP::kBarOffset + (TP::kPoolBars + TP::kRingBars + 6 + buf) * 8);
  }

  __device__ __forceinline__ long long* ledger(int warp) const {
    return reinterpret_cast<long long*>(base + TP::kLedgerOffset) + warp * kPpPhases;
  }

  __device__ __forceinline__ int* counts(int parity, int warp) const {
    return reinterpret_cast<int*>(base + TP::kCountOffset + parity * TP::kCountBytes)
           + warp * (TP::kLeavesPerWarp / 8 + TP::kBlocksPerWarp);
  }
};

//: THE FIRST DIGIT of level `l` in owner `O`'s region -- the runtime owner law, the tall set's.
__device__ __forceinline__ int region_digit0(const carry::CarryGeomRT& g, int O, int l) {
  return ((O >> g.owner_div_bits[l]) & (g.grid[l] - 1)) << g.span_bits[l];
}

//: A STREAM'S k-TH DIGIT of level `l` on side `s`, the region-relative digit the row map spells.
__device__ __forceinline__ int stream_digit(const carry::SideGeom& s, int l, int strm, int k) {
  return carry::geom_level_digit(s, l, strm, k);
}

//: THE REGION'S STAGED ROWS: the sparse levels' digit rows on a side, read side first. Every index is
//: compile-time, so the table is registers.
template <class TP>
struct StagedRows {
  int level_row[2]
               [carry::kGeomMaxLevels];  //: a sparse level's first staged row on that side, or -1
  int rl, wl;          //: the read side's and the write side's sparse level (the last)
  int rl_row, wl_row;  //: its first staged row
};

template <class TP>
__device__ __forceinline__ StagedRows<TP> staged_rows(const PrepassParams& p) {
  StagedRows<TP> r{};
  r.rl = r.wl = -1;
  r.rl_row = r.wl_row = 0;
  int at = 0;
  rola::static_for<2>([&](auto Sc) {
    constexpr int side = decltype(Sc)::value;
    rola::static_for<carry::kGeomMaxLevels>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      const bool sparse = l < p.g.D && ((p.sparse_mask[side] >> l) & 1u) != 0u;
      r.level_row[side][l] = sparse ? at : -1;
      if (sparse) {
        if (side == 0) {
          r.rl = l;
          r.rl_row = at;
        } else {
          r.wl = l;
          r.wl_row = at;
        }
        at += p.g.span[l];
      }
    });
  });
  return r;
}

// ------------------------------------------------------------------------ the words
//: COMPONENT `words`: the staged rows' words for window `win`, by cp.async, lane = word; landed at the
//: words barrier's next phase. One warp issues it. Budget: prepass.md#budgets.
template <class TP>
__device__ __forceinline__ void stage_words(const PrepassParams& p, const Smem<TP>& sm,
                                            const StagedRows<TP>& rows, int owner, int bh, int win,
                                            int parity, int lane) {
  const int t0 = win * TP::kWindow;
  const int left = (p.L - t0 + 31) / 32;
  const int nwords = left < TP::kRounds ? left : TP::kRounds;
  //: a row's sixteen words are 64 contiguous, aligned bytes: four sixteen-byte copies, lane = (row, chunk).
  rola::static_for<2>([&](auto Sc) {
    constexpr int side = decltype(Sc)::value;
    const int32_t* const table =
        p.liveness + ((long)bh * 2 + side) * ((long)p.g.wtot * p.liveness_words) + (t0 >> 5);
    rola::static_for<carry::kGeomMaxLevels>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      const int first = rows.level_row[side][l];
      const int d0 = region_digit0(p.g, owner, l);
      const int span = first < 0 ? 0 : p.g.span[l];
#pragma unroll 1
      for (int i = lane; i < span * 4; i += 32) {
        const int r = i >> 2, chunk = i & 3;
        const int row = p.g.col_base[l] + d0 + r;
        const ops::SmemAddr dst =
            sm.words(parity) + (uint32_t)((first + r) * TP::kRounds + chunk * 4) * 4u;
        ops::stage_run_if<16>(dst, table + (long)row * p.liveness_words + chunk * 4,
                              chunk * 4 < nwords);
      }
    });
  });
  ops::stage_commit();
  ops::mbar_arrive_when_landed(sm.words_bar(parity));
}

// ------------------------------------------------------------------------ the lists
//: THE WARP'S ROWS, once a call: lane j's fold TILE -- its list (a write-sparse digit: the staged row and
//: the region-relative digit) and its dense n-tile's first digit -- and the warp's four read rows;
//: registers, compile-time indexed, never re-derived a window. -- prepass_kernel.md#tiles
template <class TP>
struct WarpRows {
  int wrow;                      //: this lane's tile's list row, or -1
  int wdig;                      //: that list's region-relative digit
  int nbase;                     //: the tile's first write-dense digit
  int rrow[TP::kBlocksPerWarp];  //: the warp's read lists' staged rows, or -1
};

template <class TP>
__device__ __forceinline__ WarpRows<TP> warp_rows(const PrepassParams& p,
                                                  const StagedRows<TP>& rows, int warp, int lane) {
  WarpRows<TP> r{};
  constexpr int kTilesBits = 4;
  static_assert(TP::kLeavesPerWarp / 8 == (1 << kTilesBits), "sixteen fold tiles a warp");
  //: a list's leaves are its write-dense span, 8 a tile: tile j = (list j >> qbits, n-tile j & (q - 1))
  const int wl = rows.wl, wd = wl == 0 ? 1 : 0;
  const int lists = wl >= 0 ? (1 << p.g.w.sub_bits[wl]) : 0;
  const int qbits = wl >= 0 && wd < p.g.D ? p.g.span_bits[wd] - 3 : 0;
  const int list = lane >> qbits;
  const bool tile = wl >= 0 && lane < (1 << kTilesBits) && list < lists;
  r.wdig = tile ? stream_digit(p.g.w, wl, warp, list) : 0;
  r.wrow = tile ? rows.wl_row + r.wdig : -1;
  r.nbase = tile ? (lane & ((1 << qbits) - 1)) * 8 : 0;
  const int rl = rows.rl;
  const int rlists = rl >= 0 ? (1 << p.g.r.sub_bits[rl]) : 0;
  rola::static_for<TP::kBlocksPerWarp>([&](auto Kc) {
    constexpr int k = decltype(Kc)::value;
    r.rrow[k] = k < rlists ? rows.rl_row + stream_digit(p.g.r, rl, warp, k) : -1;
  });
  return r;
}

//: COMPONENT `lists`: a warp's read side's union over its 4 sparse digits (lane = round), by popcount
//: over the window's live words: the readout's tile count. The fold reads its lists' words itself.
template <class TP>
__device__ __forceinline__ void list_counts(const Smem<TP>& sm, const WarpRows<TP>& wr, int nwords,
                                            int parity, int warp, int lane) {
  int* const out = sm.counts(parity, warp);
  constexpr int nlists = TP::kLeavesPerWarp / 8;
  constexpr int kBlocks = TP::kBlocksPerWarp;
  uint32_t any = 0u;
  if (lane < nwords) {
#pragma unroll
    for (int k = 0; k < kBlocks; ++k) {
      if (wr.rrow[k] >= 0)
        any |= ops::load_shared_u32(sm.words(parity)
                                    + (uint32_t)(wr.rrow[k] * TP::kRounds + lane) * 4u);
    }
  }
  int total = ops::bit_count(any);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) total += __shfl_xor_sync(0xFFFFFFFFu, total, o);
  if (lane == 0) out[nlists] = total;
}

// ------------------------------------------------------------------------- the pool
//: THE WINDOW'S FACTORS: kz_m = 2^23 / (255 zeta_m) for the two columns, zeta_m the window's largest
//: col_m(t) m_ws(t) m_wd(t); zero when the window has none. -- prepass_kernel.md#zeta
struct Zeta {
  float kz_g, kz_gs;
};

__device__ __forceinline__ float factor_of(float zeta) {
  return zeta > 0.0f ? 8388608.0f / (255.0f * zeta) : 0.0f;
}

//: the unit of a column's window sum: (acc_hi 256 + acc_lo) unit_m = the fp32 contribution.
__device__ __forceinline__ float unit_of(float kz) {
  return kz > 0.0f ? 256.0f / (65025.0f * kz) : 0.0f;
}

//: THE FILL'S REGISTERS: a lane's 16 bytes of one token's write-sparse row (four tokens a quad, a
//: 16-byte piece a quad) and up to two 16-byte pieces of write-dense rows, loaded a chunk ahead and
//: landed after the fold that hides their round trip. -- prepass_kernel.md#fill
struct FillRegs {
  uint4 sp;
  uint4 dn[2];
  float g, vs;  //: warp 0, lane = token: the gain, the value scale
  float2 m;     //: and the two write maxima -- the factors' inputs
};

//: THE 4x4 BYTE TRANSPOSE across a lane quad: lane i holds word W_i (a token's four digits); it leaves with
//: byte i of every lane's word (a digit's four tokens). Two shuffle rounds, two byte selects.
__device__ __forceinline__ uint32_t quad_transpose(uint32_t w, int lane) {
  const uint32_t r1 = (uint32_t)__shfl_xor_sync(0xFFFFFFFFu, (int)w, 1);
  const uint32_t x = ops::prmt(w, r1, (lane & 1) ? 0x3715u : 0x6240u);
  const uint32_t r2 = (uint32_t)__shfl_xor_sync(0xFFFFFFFFu, (int)x, 2);
  return ops::prmt(x, r2, (lane & 2) ? 0x3276u : 0x5410u);
}

__device__ __forceinline__ uint4 load_global_16(const uint8_t* src, bool live) {
  uint4 v = make_uint4(0u, 0u, 0u, 0u);
  if (live) v = *reinterpret_cast<const uint4*>(src);
  return v;
}

//: COMPONENT `fill`, the issue: the chunk's rows into registers. Sparse: lane = (token 4 warp + (lane & 3),
//: piece lane >> 2); dense: the span's 16-byte pieces over warps 4-7 (a span of 8 by 8-byte halves), a quad
//: four consecutive tokens at one piece.
template <class TP>
__device__ __forceinline__ FillRegs fill_issue(const PrepassParams& p, const StagedRows<TP>& rows,
                                               int owner, int bh, int win, int chunk, int warp,
                                               int lane) {
  FillRegs f{};
  f.dn[0] = f.dn[1] = make_uint4(0u, 0u, 0u, 0u);
  f.g = f.vs = 0.0f;
  f.m = make_float2(0.0f, 0.0f);
  const int t0 = win * TP::kWindow + chunk * TP::kPoolTok;
  const int wl = rows.wl, wd = wl == 0 ? 1 : 0;
  const bool factor_warp = warp == 0, dense_warps = warp >= 4, dense_first = warp == 4;

  if (factor_warp) {
    const int t = t0 + lane;
    if (t < p.L) {
      const long row = (long)bh * p.L + t;
      f.g = p.gain[row];
      f.vs = p.vscale[row];
      f.m = *reinterpret_cast<const float2*>(p.scale_write + row * 2);
    }
  }
  if (wl >= 0) {
    const int tok = 4 * warp + (lane & 3), piece = lane >> 2;
    const int t = t0 + tok;
    const bool live = t < p.L && piece * 16 < p.g.span[wl];
    const long row = (long)bh * p.L + (live ? t : 0);
    const int col = p.g.col_base[wl] + region_digit0(p.g, owner, wl);
    f.sp = load_global_16(p.rows_write + row * p.g.wtot + col + piece * 16, live);
  }

  if (dense_warps && wd < p.g.D) {
    const int col = p.g.col_base[wd] + region_digit0(p.g, owner, wd);
    if (p.g.span_bits[wd] == 3) {
      //: eight digits a token: a lane a token's 8 bytes, warp 4
      if (dense_first) {
        const int t = t0 + lane;
        const bool live = t < p.L;
        const long row = (long)bh * p.L + (live ? t : 0);
        if (live) {
          const uint2 v = *reinterpret_cast<const uint2*>(p.rows_write + row * p.g.wtot + col);
          f.dn[0] = make_uint4(v.x, v.y, 0u, 0u);
        }
      }
    } else {
      const int qb = p.g.span_bits[wd] - 4;  //: 16-byte pieces a token, log2
      rola::static_for<2>([&](auto Ic) {
        constexpr int i = decltype(Ic)::value;
        const int pc = (warp - 4) * 32 + lane + 128 * i;
        const int tok = 4 * (pc >> (2 + qb)) + (pc & 3), piece = (pc >> 2) & ((1 << qb) - 1);
        const int t = t0 + tok;
        const bool live = pc < (TP::kPoolTok << qb) && t < p.L;
        const long row = (long)bh * p.L + (live ? t : 0);
        f.dn[i] = load_global_16(p.rows_write + row * p.g.wtot + col + piece * 16, live);
      });
    }
  }
  return f;
}

//: THE LANDING of a lane's words: word j of the quad's four tokens is digits `d0 + 4 j .. + 3`; after the
//: transpose lane i holds digit `d0 + 4 j + i` for the four tokens, one word at that digit's row.
__device__ __forceinline__ void land_piece(ops::SmemAddr rows_base, int stride, int d0, int quad,
                                           const uint4& w, int lane, bool store) {
  const uint32_t t0 = quad_transpose(w.x, lane), t1 = quad_transpose(w.y, lane);
  const uint32_t t2 = quad_transpose(w.z, lane), t3 = quad_transpose(w.w, lane);
  if (!store) return;
  const ops::SmemAddr at = rows_base + (uint32_t)((d0 + (lane & 3)) * stride + quad * 4);
  ops::store_shared_u32(at, t0);
  ops::store_shared_u32(at + (uint32_t)(4 * stride), t1);
  ops::store_shared_u32(at + (uint32_t)(8 * stride), t2);
  ops::store_shared_u32(at + (uint32_t)(12 * stride), t3);
}

//: COMPONENT `fill`, the landing: the registers transposed into the slot's digit rows, the factors from
//: warp 0's registers, and one release arrival a thread on the slot's full barrier.
template <class TP>
__device__ __forceinline__ void fill_land(const PrepassParams& p, const Smem<TP>& sm,
                                          const StagedRows<TP>& rows, int slot, int warp, int lane,
                                          const FillRegs& f, const Zeta& z) {
  const ops::SmemAddr base = sm.pool(slot);
  const int wl = rows.wl, wd = wl == 0 ? 1 : 0;
  const bool factor_warp = warp == 0, dense_warps = warp >= 4, dense_first = warp == 4;
  //: every lane transposes (the shuffles are warp-wide); a lane whose piece is past the span stores nothing
  if (wl >= 0)
    land_piece(base, TP::kPoolSparseStride, (lane >> 2) * 16, warp, f.sp, lane,
               (lane >> 2) * 16 < p.g.span[wl]);
  if (dense_warps && wd < p.g.D) {
    if (p.g.span_bits[wd] == 3) {
      const ops::SmemAddr drows = base + (uint32_t)TP::kPoolDenseOff;
      const uint32_t t0w = quad_transpose(f.dn[0].x, lane), t1w = quad_transpose(f.dn[0].y, lane);
      if (dense_first) {
        const ops::SmemAddr at =
            drows + (uint32_t)((lane & 3) * TP::kPoolDenseStride + (lane >> 2) * 4);
        ops::store_shared_u32(at, t0w);
        ops::store_shared_u32(at + (uint32_t)(4 * TP::kPoolDenseStride), t1w);
      }
    } else {
      const int qb = p.g.span_bits[wd] - 4;
      rola::static_for<2>([&](auto Ic) {
        constexpr int i = decltype(Ic)::value;
        const int pc = (warp - 4) * 32 + lane + 128 * i;
        const int quad = pc >> (2 + qb), piece = (pc >> 2) & ((1 << qb) - 1);
        land_piece(base + (uint32_t)TP::kPoolDenseOff, TP::kPoolDenseStride, piece * 16, quad,
                   f.dn[i], lane, pc < (TP::kPoolTok << qb));
      });
    }
  }
  //: the factors P_m(t) = rint(col_m(t) m_ws m_wd kz_m), warp 0, a lane a token, into [column][token]
  if (factor_warp) {
    const float fm = f.g * f.m.x * f.m.y;  //: depth two: both write maxima
    ops::store_shared_u32(base + (uint32_t)(TP::kPoolFactorOff + lane * 4),
                          (uint32_t)__float2int_rn(fm * z.kz_g));
    ops::store_shared_u32(base + (uint32_t)(TP::kPoolFactorOff + TP::kPoolTok * 4 + lane * 4),
                          (uint32_t)__float2int_rn(fm * f.vs * z.kz_gs));
  }
  //: one release arrival a warp after its lanes' stores are ordered: the barrier counts warps
  __syncwarp();
  if (lane == 0) ops::mbar_arrive(sm.pool_full(slot));
}

// ------------------------------------------------------------------------- the fold
//: THE FOLD'S SHAPE at a write-dense span of 8 Q digits: 16 / Q lists a warp on the sixteen rows of ONE
//: m-tile (A = the lists' raw write-sparse words, no formation), Q n-tiles of eight write-dense digits,
//: each formed into FOUR B columns [g_hi, g_lo, gs_hi, gs_lo] from the token's factors: an accumulator an
//: (n-tile, column). -- prepass_kernel.md#fold
template <class TP, int Q>
struct FoldShape {
  static constexpr int kLists = TP::kTilesPerWarp / Q;
  static constexpr int kColumns = 2 * TP::kCols;  //: hi and lo of g and of g s_t
  static constexpr int kAccs = Q * kColumns;
  static_assert(kLists * Q == TP::kTilesPerWarp && kLists <= 16,
                "the tiles are whole lists on one m-tile");
};

//: THE WARP'S ACCUMULATORS at shape Q: s32, four a lane an (n-tile, column), live across a window's chunks,
//: folded into the fp32 state at its end.
template <class TP, int Q>
struct Acc {
  int32_t c[FoldShape<TP, Q>::kAccs][4];
};

//: COMPONENT `zeta`: a warp scans the window's tokens (a lane a token, sixteen rounds) for the two maxima;
//: every round's loads are issued before any is reduced, so the scan is one round trip.
template <class TP>
__device__ __forceinline__ Zeta window_zeta(const PrepassParams& p, int bh, int t0, int wl,
                                            int lane) {
  Zeta z{0.0f, 0.0f};
  if (wl < 0) return z;
  constexpr int kRounds = TP::kRounds;
  float g[kRounds], vs[kRounds];
  float2 m[kRounds];
#pragma unroll
  for (int r = 0; r < kRounds; ++r) {
    const int t = t0 + r * 32 + lane;
    const bool live = t < p.L;
    const long row = (long)bh * p.L + (live ? t : 0);
    g[r] = live ? p.gain[row] : 0.0f;
    vs[r] = live ? p.vscale[row] : 0.0f;
    m[r] =
        live ? *reinterpret_cast<const float2*>(p.scale_write + row * 2) : make_float2(0.0f, 0.0f);
  }
  float zg = 0.0f, zgs = 0.0f;
#pragma unroll
  for (int r = 0; r < kRounds; ++r) {
    const float f =
        g[r] * m[r].x * m[r].y;  //: at depth two the two write levels are the two maxima
    zg = fmaxf(zg, f);
    zgs = fmaxf(zgs, f * vs[r]);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    zg = fmaxf(zg, __shfl_xor_sync(0xFFFFFFFFu, zg, o));
    zgs = fmaxf(zgs, __shfl_xor_sync(0xFFFFFFFFu, zgs, o));
  }
  z.kz_g = factor_of(zg);
  z.kz_gs = factor_of(zgs);
  return z;
}

//: ONE k16 STEP of the chunk: the lane's four consecutive tokens are one word of every digit row. A is the
//: two lists' words as they are (rows `lane >> 2` and `+ 8`); the factors are two 16-byte loads (both
//: columns); for each n-tile the dense digit's word is unpacked once and formed into the four B columns
//: (`P a + 128`: hi = byte 2, lo = byte 1 of the product), one IMMA each.
template <class TP, int Q>
__device__ __forceinline__ void fold_step(ops::SmemAddr pool, int st, int kg, int n,
                                          uint32_t dig_lo, uint32_t dig_hi, Acc<TP, Q>& acc) {
  const uint32_t quad = (uint32_t)(4 * st + kg) * 4u;
  const uint32_t fac = (uint32_t)(TP::kPoolFactorOff + (16 * st + 4 * kg) * 4);
  const uint4 pg = ops::load_shared_vec16(pool + fac);
  const uint4 pgs = ops::load_shared_vec16(pool + fac + (uint32_t)(TP::kPoolTok * 4));
  const uint32_t p[2][4] = {{pg.x, pg.y, pg.z, pg.w}, {pgs.x, pgs.y, pgs.z, pgs.w}};
  const uint32_t a[2] = {ops::load_shared_u32(pool + dig_lo * TP::kPoolSparseStride + quad),
                         ops::load_shared_u32(pool + dig_hi * TP::kPoolSparseStride + quad)};
  rola::static_for<Q>([&](auto Nc) {
    constexpr int nt = decltype(Nc)::value;
    const uint32_t w = ops::load_shared_u32(
        pool + (uint32_t)(TP::kPoolDenseOff + (nt * 8 + n) * TP::kPoolDenseStride) + quad);
    uint32_t b[4];
    rola::static_for<4>([&](auto Ic) {
      constexpr int i = decltype(Ic)::value;
      b[i] = __byte_perm(w, 0u, 0x4440u + i);
    });
    rola::static_for<TP::kCols>([&](auto Cc) {
      constexpr int c = decltype(Cc)::value;
      uint32_t v[4];
      rola::static_for<4>([&](auto Ic) {
        constexpr int i = decltype(Ic)::value;
        v[i] = p[c][i] * b[i] + 128u;
      });
      ops::imma_u8(acc.c[nt * 4 + 2 * c], a, ops::pack_byte4(v[0], v[1], v[2], v[3], 0x0062u));
      ops::imma_u8(acc.c[nt * 4 + 2 * c + 1], a, ops::pack_byte4(v[0], v[1], v[2], v[3], 0x0051u));
    });
  });
}

//: COMPONENT `fold` at shape Q: the chunk's two k16 steps as one straight-line block. A lane's row is (list, digit row): lists
//: `4 mt + (lane >> 4)` and `+ 2` for its two A registers, digit row `(lane >> 2) & 3` -- g on rows 0-1,
//: g s_t on 2-3, hi on the even row. A list past the shape's count reads the row's zero bytes.
template <class TP, int Q>
__device__ __forceinline__ void fold_chunk(const Smem<TP>& sm, int slot, uint32_t union_word,
                                           int lane, uint32_t dig_lo, uint32_t dig_hi,
                                           Acc<TP, Q>& acc) {
  const ops::SmemAddr pool = sm.pool(slot);
  const int kg = lane & 3, n = lane >> 2;
  static_assert(TP::kCols == 2, "two factor columns: g and g s_t");
  //: both steps in one block, so their loads and formations interleave; the chunk is skipped whole when
  //: no list has a live token (`fold`), a step's own emptiness costs its loads and IMMAs of zeros
  (void)union_word;
  rola::static_for<TP::kPoolTok / 16>(
      [&](auto Sc) { fold_step<TP, Q>(pool, decltype(Sc)::value, kg, n, dig_lo, dig_hi, acc); });
}

//: THE LANE'S LIST DIGITS, once a launch: A's rows `lane >> 2` and `+ 8` are lists `lane >> 2` and
//: `+ 8`; a list past the shape's count reads the zero row.
template <class TP, int Q>
__device__ __forceinline__ void list_digits(int lane, int mydig, int lists, uint32_t& dig_lo,
                                            uint32_t& dig_hi) {
  const int llo = lane >> 2, lhi = llo + 8;
  const int dlo = __shfl_sync(0xFFFFFFFFu, mydig, llo),
            dhi = __shfl_sync(0xFFFFFFFFu, mydig, lhi & 15);
  dig_lo = (uint32_t)(llo < lists ? dlo : TP::kPoolZeroRow);
  dig_hi = (uint32_t)(lhi < lists ? dhi : TP::kPoolZeroRow);
}

//: A LANE'S LIST WORD for a chunk (lane = tile), loaded a chunk ahead of its fold.
template <class TP>
__device__ __forceinline__ uint32_t list_word(const Smem<TP>& sm, int chunk, int parity, int lane,
                                              const WarpRows<TP>& wr) {
  return (lane < TP::kTilesPerWarp && wr.wrow >= 0)
             ? ops::load_shared_u32(sm.words(parity)
                                    + (uint32_t)(wr.wrow * TP::kRounds + chunk) * 4u)
             : 0u;
}

//: COMPONENT `fold`: the chunk at the kernel's shape, skipped whole when the union of the warp's list
//: words (reduced a chunk ahead) has no live token.
template <class TP, int Q>
__device__ __forceinline__ void fold(const Smem<TP>& sm, int slot, uint32_t union_word, int lane,
                                     uint32_t dig_lo, uint32_t dig_hi, Acc<TP, Q>& acc) {
  if (union_word == 0u) return;
  fold_chunk<TP, Q>(sm, slot, union_word, lane, dig_lo, dig_hi, acc);
}

//: COMPONENT `close` at shape Q: a window's sums into the fp32 state -- a column's lo accumulator joins
//: its hi in the same lane, times the column's unit; rows `lane >> 2` and `+ 8` are lists, the
//: accumulator's columns leaves `2 kg + {0, 1}` of the n-tile -- and the accumulators cleared.
template <class TP, int Q>
__device__ __forceinline__ void window_close(const Smem<TP>& sm, int warp, int lane, const Zeta& z,
                                             int lists, Acc<TP, Q>& acc) {
  const ops::SmemAddr st = sm.state(warp);
  const int llo = lane >> 2, lhi = llo + 8, kg = lane & 3;
  const float unit[2] = {unit_of(z.kz_g), unit_of(z.kz_gs)};
  rola::static_for<Q>([&](auto Nc) {
    constexpr int nt = decltype(Nc)::value;
    rola::static_for<TP::kCols>([&](auto Cc) {
      constexpr int c = decltype(Cc)::value, hi = nt * 4 + 2 * c, lo = hi + 1;
      const int leaf_n = nt * 8 + 2 * kg;
      rola::static_for<2>([&](auto Hc) {
        constexpr int h = decltype(Hc)::value;  //: 0: rows lane >> 2 (c0, c1); 1: rows + 8 (c2, c3)
        const int list = h ? lhi : llo;
        if (list < lists) {
          const float v0 = fmaf((float)acc.c[hi][2 * h], 256.0f, (float)acc.c[lo][2 * h]) * unit[c];
          const float v1 =
              fmaf((float)acc.c[hi][2 * h + 1], 256.0f, (float)acc.c[lo][2 * h + 1]) * unit[c];
          const ops::SmemAddr a0 = st + (uint32_t)(((list * 8 * Q + leaf_n) * TP::kCols + c) * 4);
          const ops::SmemAddr a1 = a0 + (uint32_t)(TP::kCols * 4);
          ops::store_shared_f32(a0, ops::load_shared_f32(a0) + v0);
          ops::store_shared_f32(a1, ops::load_shared_f32(a1) + v1);
        }
      });
      acc.c[hi][0] = acc.c[hi][1] = acc.c[hi][2] = acc.c[hi][3] = 0;
      acc.c[lo][0] = acc.c[lo][1] = acc.c[lo][2] = acc.c[lo][3] = 0;
    });
  });
  __syncwarp();
}

//: THE STATE DUMP: a warp's fp32 state to the caller's plane by canonical leaf, when bound.
template <class TP>
__device__ __forceinline__ void dump_state(const PrepassParams& p, const Smem<TP>& sm,
                                           const StagedRows<TP>& rows, const WarpRows<TP>& wr,
                                           int owner, int bh, int warp, int lane) {
  if (p.fold_dump == nullptr || rows.wl < 0) return;
  const int wl = rows.wl, wd = wl == 0 ? 1 : 0;
  const ops::SmemAddr st = sm.state(warp);
  const long N = (long)p.g.width[0] * p.g.width[1];
  const int d0s = region_digit0(p.g, owner, wl), d0d = region_digit0(p.g, owner, wd);
#pragma unroll 1
  for (int i = lane; i < TP::kLeavesPerWarp; i += 32) {
    const int k = i >> 3, nn = i & 7;
    const int trow = __shfl_sync(0xFFFFFFFFu, wr.wrow, k);
    if (trow < 0) continue;
    const int ds = d0s + __shfl_sync(0xFFFFFFFFu, wr.wdig, k);
    const int dd = d0d + __shfl_sync(0xFFFFFFFFu, wr.nbase, k) + nn;
    const long leaf = wl == 0 ? (long)ds * p.g.width[1] + dd : (long)dd * p.g.width[1] + ds;
    float* const out = p.fold_dump + ((long)bh * N + leaf) * 2;
    out[0] = ops::load_shared_f32(st + (uint32_t)(i * TP::kCols * 4));
    out[1] = ops::load_shared_f32(st + (uint32_t)(i * TP::kCols * 4 + 4));
  }
}

// ---------------------------------------------------------------------- the transfer
//: COMPONENT `publish`: a fold warp's 128 leaves x 2 columns as s16 against each leaf's max, and the max
//: as a float, into the transfer in the READ layout. THE STUB reads the state's first float a lane and
//: writes zeros and its maxima (the real digits are the component's own step).
template <class TP>
__device__ __forceinline__ void publish(const Smem<TP>& sm, int warp, int lane) {
  const uint32_t leaf0 = (uint32_t)(warp * TP::kLeavesPerWarp);
  const uint32_t low =
      __float_as_uint(ops::load_shared_f32(sm.state(warp) + (uint32_t)(lane * 8))) & 0u;
  const uint4 vals = make_uint4(low, 0u, 0u, 0u);
  //: 128 leaves x 2 columns x 2 B = 512 B a warp: one 16-byte store a lane; the exponents 256 B: half.
  ops::store_shared_vec16(sm.xfer_values() + (leaf0 * TP::kCols * 2u) + (uint32_t)lane * 16u, vals);
  if (lane < 16)
    ops::store_shared_vec16(sm.xfer_scales() + (leaf0 * TP::kCols) + (uint32_t)lane * 16u,
                            make_uint4(0u, 0u, 0u, 0u));
}

//: COMPONENT `rescale`: a read warp's 4 blocks from the transfer: each block's scale = the max of its 32
//: leaves' maxima (one a lane, five shuffles), its live min beside it, the columns requantized into the
//: block's unit; the hi bytes stay in registers. THE STUB performs the loads and the reductions.
template <class TP>
__device__ __forceinline__ void rescale(const Smem<TP>& sm, int warp, int lane,
                                        float (&sigma)[TP::kBlocksPerWarp],
                                        uint32_t (&hi)[TP::kBlocksPerWarp]) {
  rola::static_for<TP::kBlocksPerWarp>([&](auto Bc) {
    constexpr int b = decltype(Bc)::value;
    const uint32_t leaf = (uint32_t)((warp * TP::kBlocksPerWarp + b) * TP::kBlockLeaves + lane);
    const float m0 = (float)(int)ops::load_shared_u8(sm.xfer_scales() + leaf * (uint32_t)TP::kCols);
    const float m1 =
        (float)(int)ops::load_shared_u8(sm.xfer_scales() + leaf * (uint32_t)TP::kCols + 1u);
    float mx = fmaxf(m0, m1);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, o));
    sigma[b] = mx;
    hi[b] = ops::load_shared_u32(sm.xfer_values() + leaf * (TP::kCols * 2u));
  });
}

// ---------------------------------------------------------------------- the readout
//: COMPONENT `readout`: the warp's union list, a 32-token tile at a time: the tile's read rows staged into
//: the ring (the dense sub-block's entries, the sparse entries, the scales), the per-block partials, the
//: two per-token sums into the panel. THE STUB stages the range's tokens in order, waits the ring, and
//: adds zeros into the panel.
//: THE TILE'S ROWS INTO THE RING, issued; the ring barrier arrives when they land.
template <class TP>
__device__ __forceinline__ void readout_issue(const PrepassParams& p, const Smem<TP>& sm,
                                              const StagedRows<TP>& rows, int owner, int bh, int t0,
                                              int tile, int rslot, int warp, int lane) {
  static_assert(TP::kBlockLeaves == 4 * 8, "a token's dense entries are one quad of 8-byte copies");
  const ops::SmemAddr ring = sm.ring(warp, rslot);
  const int tbase = t0 + tile * TP::kTile;
  const int rl = rows.rl;
  const int rd = rl == 0 ? 1 : 0;
  if (rd < p.g.D) {
    //: the warp's dense sub-block, its stream's first digit on the dense level: a quad a token, eight
    //: tokens an instruction (plan.md#ring).
    const int col =
        p.g.col_base[rd] + region_digit0(p.g, owner, rd) + stream_digit(p.g.r, rd, warp, 0);
    const int q = lane & 3;
    constexpr int kQuadRounds = TP::kTile / 8;
#pragma unroll
    for (int i = 0; i < kQuadRounds; ++i) {
      const int tok = 8 * i + (lane >> 2);
      const int t = tbase + tok;
      const bool live = t < p.L;
      const uint32_t row = (uint32_t)bh * (uint32_t)p.L + (uint32_t)(live ? t : 0);
      ops::stage_run_if<8>(ring + (uint32_t)(tok * TP::kRingRowBytes + q * 8),
                           p.rows_read + (long)row * p.g.wtot + col + q * 8, live);
    }
  }
  //: the sparse entries and the two read scales, a lane a token.
  const int t = tbase + lane;
  const bool live = t < p.L;
  const uint32_t row = (uint32_t)bh * (uint32_t)p.L + (uint32_t)(live ? t : 0);
  const ops::SmemAddr dst = ring + (uint32_t)(lane * TP::kRingRowBytes);
  if (rl >= 0) {
    //: the region's whole read-sparse row (8 or 16 digits; the warp's four are interleaved in it)
    const int col = p.g.col_base[rl] + region_digit0(p.g, owner, rl);
    const uint8_t* const src = p.rows_read + (long)row * p.g.wtot + col;
    ops::stage_run_if<8>(dst + (uint32_t)TP::kRingSparseOff, src, live);
    ops::stage_run_if<8>(dst + (uint32_t)TP::kRingSparseOff + 8u, src + 8,
                         live && p.g.span[rl] > 8);
  }
  ops::stage_run_if<8>(dst + (uint32_t)TP::kRingScaleOff, p.scale_read + (long)row * p.g.D, live);
  ops::stage_commit();
  ops::mbar_arrive_when_landed(sm.ring_bar(warp, rslot));
}

//: COMPONENT `readout`: a landed tile: the per-block partials and the two per-token sums into the panel.
//: THE STUB adds zeros; the hi bytes are the operand.
template <class TP>
__device__ __forceinline__ void readout_tile(const Smem<TP>& sm, int tile, int rslot,
                                             uint32_t rphase, int buf, int warp, int lane,
                                             const uint32_t (&hi)[TP::kBlocksPerWarp]) {
  ops::mbar_wait(sm.ring_bar(warp, rslot), rphase);
  const uint32_t a =
      ops::load_shared_u32(sm.ring(warp, rslot) + (uint32_t)(lane * TP::kRingRowBytes));
  const float partial = (float)((a & hi[0] & hi[1] & hi[2] & hi[3]) & 0u);
  //: the warp's own panel row: plain stores, summed across the warps at the drain.
  const uint32_t in_range = (uint32_t)((tile * TP::kTile + lane) & (TP::kRange - 1));
  ops::store_shared_f32x2(sm.panel(buf, warp) + in_range * 8u, partial, partial);
}

//: COMPONENT `drain`: the range's eight warp rows summed a token, to the global sums by RED.F32 (256
//: threads: a token's column each). Gated by the buffer's rows barrier, released by its drained barrier;
//: no CTA-wide barrier.
template <class TP>
__device__ __forceinline__ void drain(const PrepassParams& p, const Smem<TP>& sm, int bh, int t0,
                                      int range, int buf, int tid) {
  const int i = tid >> 1, col = tid & 1;  //: the range's tokens x 2 columns, a thread each
  const int t = t0 + range * TP::kRange + i;
  constexpr int kWarps = TP::kWarps;
  if (i >= TP::kRange) return;
  float v = 0.0f;
#pragma unroll
  for (int w = 0; w < kWarps; ++w)
    v += ops::load_shared_f32(sm.panel(buf, w) + (uint32_t)(i * 8 + col * 4));
  if (t < p.L) ops::red_global_add_f32((col == 0 ? p.den : p.scale) + (long)bh * p.L + t, v);
}

// ------------------------------------------------------------------------ the frame
//: THE WINDOW LOOP: a warp's program. Words a window ahead (one warp stages, all wait), the list counts,
//: the chunks filled a slot ahead and folded, the publish, the rescale after the publish barrier, the
//: ranges read out and drained. -- see docs/internals/prepass/prepass.md#frame
template <class TP, int Q>
__device__ __forceinline__ void window_loop(const PrepassParams& p, const Smem<TP>& sm, int owner,
                                            int bh, int tid, int warp, int lane, Clock& pc) {
  const StagedRows<TP> rows = staged_rows<TP>(p);
  const int nWindows = (p.L + TP::kWindow - 1) / TP::kWindow;
  const bool scanner = warp == TP::kWarps - 1;
  Acc<TP, Q> acc{};
  float sigma[TP::kBlocksPerWarp];
  uint32_t hi[TP::kBlocksPerWarp] = {0u, 0u, 0u, 0u};
  uint32_t fills0 = 0u, fills1 = 0u;
  uint32_t rings = 0u, ranges = 0u;
  const WarpRows<TP> wr = warp_rows<TP>(p, rows, warp, lane);
  //: the fold's shape: lists a warp and the write-dense span in n-tiles; lane = list for the digits
  const int wl = rows.wl, wd = wl == 0 ? 1 : 0;
  const int lists = wl >= 0 ? (1 << p.g.w.sub_bits[wl]) : 0;
  const int mydig = lane < lists ? stream_digit(p.g.w, wl, warp, lane) : 0;
  uint32_t dig_lo, dig_hi;
  list_digits<TP, Q>(lane, mydig, lists, dig_lo, dig_hi);
  (void)wd;

  if (tid < TP::kPoolSlots) {
    ops::mbar_init(sm.pool_full(tid), TP::kWarps);
    ops::mbar_init(sm.pool_empty(tid), TP::kWarps);
  }
  if (lane < TP::kRingSlots) ops::mbar_init(sm.ring_bar(warp, lane), 32u);
  if (tid < 2) {
    ops::mbar_init(sm.words_bar(tid), 32u);
    ops::mbar_init(sm.pub_bar(tid), TP::kWarps);
    ops::mbar_init(sm.rows_bar(tid), TP::kWarps);
    ops::mbar_init(sm.drained_bar(tid), TP::kWarps);
  }
  //: the sparse zero row (a padding list's operand) in both slots, and the fp32 state: zero once.
  if (tid < TP::kPoolSlots * (TP::kPoolSparseStride / 4))
    ops::store_shared_u32(sm.pool(tid / (TP::kPoolSparseStride / 4))
                              + (uint32_t)(TP::kPoolZeroRow * TP::kPoolSparseStride
                                           + (tid % (TP::kPoolSparseStride / 4)) * 4),
                          0u);
#pragma unroll 1
  for (int i = lane; i < TP::kLeavesPerWarp * TP::kCols; i += 32)
    ops::store_shared_f32(sm.state(warp) + (uint32_t)(i * 4), 0.0f);
  ops::rendezvous();

  if (scanner) stage_words<TP>(p, sm, rows, owner, bh, 0, 0, lane);
  pc.lap(kPpWords);

#pragma unroll 1
  for (int win = 0; win < nWindows; ++win) {
    const int parity = win & 1, t0 = win * TP::kWindow;
    const uint32_t phase = (uint32_t)(win >> 1) & 1u;
    const Zeta zeta = window_zeta<TP>(p, bh, t0, rows.wl, lane);
    pc.lap(kPpLists);
    ops::mbar_wait(sm.words_bar(parity), phase);
    pc.lap(kPpFrame);
    const int left = (p.L - t0 + 31) / 32;
    list_counts<TP>(sm, wr, left < TP::kRounds ? left : TP::kRounds, parity, warp, lane);
    ops::rendezvous();
    pc.lap(kPpLists);
    if (scanner && win + 1 < nWindows) {
      stage_words<TP>(p, sm, rows, owner, bh, win + 1, parity ^ 1, lane);
      pc.lap(kPpWords);
    }
    //: the chunks, a slot ahead and TWO CHUNKS OF LOADS IN FLIGHT: chunk c+2's rows are loaded before c
    //: is folded, chunk c+1's (loaded a chunk earlier) landed after it; the list words a chunk ahead.
    //: EACH OWNER WALKS THE CHUNKS FROM ITS OWN OFFSET, so the owners sharing a row slice read
    //: different lines at any moment (the sum is exact, so the order is free).
    const auto chunk_at = [&](int c) { return (c + owner) & (TP::kChunks - 1); };
    {
      FillRegs first = fill_issue<TP>(p, rows, owner, bh, win, chunk_at(0), warp, lane);
      fill_land<TP>(p, sm, rows, 0, warp, lane, first, zeta);
      ++fills0;
    }
    FillRegs held = fill_issue<TP>(p, rows, owner, bh, win, chunk_at(1), warp, lane);
    uint32_t union_word =
        __reduce_or_sync(0xFFFFFFFFu, list_word<TP>(sm, chunk_at(0), parity, lane, wr));
    pc.lap(kPpFill);
#pragma unroll 1
    for (int c = 0; c < TP::kChunks; ++c) {
      const int slot = c & 1, nslot = slot ^ 1;
      FillRegs next{};
      if (c + 2 < TP::kChunks)
        next = fill_issue<TP>(p, rows, owner, bh, win, chunk_at(c + 2), warp, lane);
      const uint32_t nextword =
          c + 1 < TP::kChunks ? list_word<TP>(sm, chunk_at(c + 1), parity, lane, wr) : 0u;
      ops::mbar_wait(sm.pool_full(slot), ((slot ? fills1 : fills0) - 1u) & 1u);
      pc.lap(kPpFrame);
      fold<TP, Q>(sm, slot, union_word, lane, dig_lo, dig_hi, acc);
      __syncwarp();
      if (lane == 0)
        ops::mbar_arrive(sm.pool_empty(slot));  //: one arrival a warp: the barrier counts warps
      union_word = __reduce_or_sync(0xFFFFFFFFu, nextword);
      pc.lap(kPpFold);
      if (c + 1 < TP::kChunks) {
        const uint32_t nfills = nslot ? fills1 : fills0;
        if (nfills > 0u) ops::mbar_wait(sm.pool_empty(nslot), (nfills - 1u) & 1u);
        fill_land<TP>(p, sm, rows, nslot, warp, lane, held, zeta);
        if (nslot)
          ++fills1;
        else
          ++fills0;
        pc.lap(kPpFill);
      }
      held = next;
    }
    window_close<TP, Q>(sm, warp, lane, zeta, lists, acc);
    publish<TP>(sm, warp, lane);
    __syncwarp();
    if (lane == 0) ops::mbar_arrive(sm.pub_bar(parity));
    pc.lap(kPpPublish);
    ops::mbar_wait(sm.pub_bar(parity), phase);
    pc.lap(kPpFrame);
    rescale<TP>(sm, warp, lane, sigma, hi);
    pc.lap(kPpRescale);
    //: the ring a tile ahead: tile 0 issued, then each tile's successor before its own wait.
    readout_issue<TP>(p, sm, rows, owner, bh, t0, 0, (int)(rings & 1u), warp, lane);
#pragma unroll 1
    for (int range = 0; range < TP::kRanges; ++range) {
      //: the panel buffer of this range: free once the range before last was drained.
      const int buf = (int)(ranges & 1u);
      if (ranges >= 2u) ops::mbar_wait(sm.drained_bar(buf), ((ranges >> 1) - 1u) & 1u);
#pragma unroll 1
      for (int tile = 0; tile < TP::kRange / TP::kTile; ++tile) {
        const int abs_tile = range * (TP::kRange / TP::kTile) + tile;
        const int rslot = (int)(rings & 1u);
        if (abs_tile + 1 < TP::kTiles)
          readout_issue<TP>(p, sm, rows, owner, bh, t0, abs_tile + 1, rslot ^ 1, warp, lane);
        readout_tile<TP>(sm, abs_tile, rslot, (rings >> 1) & 1u, buf, warp, lane, hi);
        ++rings;
      }
      if (lane == 0) ops::mbar_arrive(sm.rows_bar(buf));
      pc.lap(kPpReadout);
      ops::mbar_wait(sm.rows_bar(buf), (ranges >> 1) & 1u);
      drain<TP>(p, sm, bh, t0, range, buf, tid);
      __syncwarp();
      if (lane == 0) ops::mbar_arrive(sm.drained_bar(buf));
      ++ranges;
      pc.lap(kPpDrain);
    }
    (void)sigma;
  }
  dump_state<TP>(p, sm, rows, wr, owner, bh, warp, lane);
}

// ----------------------------------------------------------------------- the kernel
//: THE KERNEL at depth D and fold shape Q (the write-dense span in n-tiles of eight), both chosen on the
//: host from the declared widths. -- prepass_kernel.md#tiles
template <int D, int Q>
__global__ __launch_bounds__(TallPlan<D>::kThreads, 1) void prepass_kernel(
    __grid_constant__ const PrepassParams p) {
  using TP = TallPlan<D>;
  extern __shared__ __align__(128) char smem_raw[];
  const Smem<TP> sm{smem_raw};
  const int owner = (int)blockIdx.x, bh = (int)blockIdx.y;
  const int tid = (int)threadIdx.x;
  const int warp = ops::uniform_warp<TP::kWarps>(), lane = tid & 31;
  Clock pc(sm.ledger(warp), lane);
  const long cta = (long)bh * gridDim.x + owner;
  if (p.trace != nullptr && cta < p.trace_ctas)
    pc.bind_trace(p.trace + (cta * TP::kWarps + warp) * p.trace_cap, p.trace_cap);
  window_loop<TP, Q>(p, sm, owner, bh, tid, warp, lane, pc);
  pc.lap(kPpFrame);
  pc.publish(p.ledger, cta, TP::kWarps, warp);
}

}  // namespace rola::prepass
