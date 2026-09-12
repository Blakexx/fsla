// csrc/rola/src/carry/carry_kernel.cuh -- THE CARRY BODY: the CTA box's prologue, the
// owned boxes' state in registers, and the window loop: the head that SORTS the read side by
// first live box and mints the write side's per-warp and union words; the SNAPSHOT of the
// state's high halves the readout reads; the READOUT, position-carved over the sorted read
// tiles, each warp its own tiles; the FOLD, state-carved over a POOL of union-live tokens,
// each warp walking its own boxes' words and compacting its tokens into fragments.
//
// THE CARVE: a warp owns `kDealt` boxes (sixteen leaves each) and every channel of them,
// fp32 in registers for the call. The coefficient is a Kronecker product and is NEVER
// MATERIALIZED: the fold scales the inner tile (A) by each token's outer factor and gain and
// contracts the tokens against the V rows (B) for all channels; the readout scales the inner
// tile by the token's outer factor and contracts the box's positions against the snapshot.
// "Inner" and "outer" are SETS of levels (by where a level's digit falls against the
// sixteen-leaf position bits); a set of several levels is composed into one run a token.
// THE HAZARD RULE: the readout reads the snapshot and its own tiles; the fold reads the pool
// and writes registers; snapshot before either, fold complete before the next snapshot.
// COMPONENTS (KERNEL_STANDARDS §19): each function below is one, with its budget line in
// docs/internals/carry/carry_kernel.md#budgets and its part bench in
// benchmarks/unit/bench_carry_parts.py; the phase ledger times them in the composition.
#pragma once

#include <cuda_bf16.h>

#include <type_traits>

#include "carry/box.cuh"
#include "carry/params.cuh"
#include "common/design.cuh"
#include "common/geom.cuh"
#include "common/ops.cuh"
#include "common/state_page.cuh"
#include "common/static_for.cuh"
#include "facts/liveness_contract.cuh"

namespace rola::carry {

namespace ops = denseref::ops;

//: THE SHARED BLOCK, as the ledger lays it out. -- see docs/internals/carry/smem_ledger.md
template <class BP>
struct Smem {
  char* base;

  __device__ __forceinline__ ops::SmemAddr at(int off) const {
    return ops::smem_addr(base) + (uint32_t)off;
  }

  __device__ __forceinline__ const uint32_t* words(int side, int row) const {
    return reinterpret_cast<const uint32_t*>(base + BP::kWordsOffset + side * BP::kWordsBytes
                                             + row * BP::kRounds * 4);
  }

  __device__ __forceinline__ ops::SmemAddr words_addr(int side, int row) const {
    return at(BP::kWordsOffset + side * BP::kWordsBytes + row * BP::kRounds * 4);
  }

  //: THE HEAD'S PRODUCTS by window parity: the read order and tile masks; the write side's
  //: words a warp, the union words and the union's prefix by round.
  __device__ __forceinline__ uint16_t* order(int parity) const {
    return reinterpret_cast<uint16_t*>(base + BP::kOrderOffset + parity * BP::kOrderBytes);
  }

  __device__ __forceinline__ uint16_t* tilemask(int parity) const {
    return reinterpret_cast<uint16_t*>(base + BP::kTileMaskOffset + parity * BP::kTileMaskBytes);
  }

  __device__ __forceinline__ uint16_t* rcounts() const {
    return reinterpret_cast<uint16_t*>(base + BP::kRCountOffset);
  }

  __device__ __forceinline__ ops::SmemAddr rcounts_addr() const { return at(BP::kRCountOffset); }

  __device__ __forceinline__ uint16_t* masks() const {
    return reinterpret_cast<uint16_t*>(base + BP::kMaskOffset);
  }

  __device__ __forceinline__ uint32_t* warpwords(int parity, int warp) const {
    return reinterpret_cast<uint32_t*>(base + BP::kWarpWordsOffset + parity * BP::kWarpWordsBytes
                                       + warp * BP::kRounds * 4);
  }

  __device__ __forceinline__ uint32_t* unionwords(int parity) const {
    return reinterpret_cast<uint32_t*>(base + BP::kUnionOffset + parity * BP::kUnionBytes);
  }

  __device__ __forceinline__ uint16_t* prefix(int parity) const {
    return reinterpret_cast<uint16_t*>(base + BP::kPrefixOffset + parity * BP::kPrefixBytes);
  }

  __device__ __forceinline__ ops::SmemAddr prefix_addr(int parity) const {
    return at(BP::kPrefixOffset + parity * BP::kPrefixBytes);
  }

  __device__ __forceinline__ int32_t* live_count() const {
    return reinterpret_cast<int32_t*>(base + BP::kLiveOffset);
  }

  //: THE SNAPSHOT's row `row` (read order), 16-byte chunk `chunk`.
  __device__ __forceinline__ ops::SmemAddr snapshot() const { return at(BP::kSnapshotOffset); }

  __device__ __forceinline__ float* massrow() const {
    return reinterpret_cast<float*>(base + BP::kMassRowOffset);
  }

  __device__ __forceinline__ ops::SmemAddr massrow_addr() const { return at(BP::kMassRowOffset); }

  __device__ __forceinline__ uint16_t* rowmap() const {
    return reinterpret_cast<uint16_t*>(base + BP::kRowMapOffset);
  }

  __device__ __forceinline__ int32_t* page_slots() const {
    return reinterpret_cast<int32_t*>(base + BP::kSlotOffset);
  }

  __device__ __forceinline__ long long* ledger(int warp) const {
    return reinterpret_cast<long long*>(base + BP::kLedgerOffset) + warp * kPhases;
  }

  __device__ __forceinline__ int32_t* counter() const {
    return reinterpret_cast<int32_t*>(base + BP::kCounterOffset);
  }

  __device__ __forceinline__ uint32_t* walk(int warp) const {
    return reinterpret_cast<uint32_t*>(base + BP::kWalkOffset) + warp * BP::kWalkEntries;
  }

  //: THE POOL's slot `s`: its V rows, inner and outer run tiles, and its barriers.
  __device__ __forceinline__ ops::SmemAddr pool(int s) const {
    return at(BP::kRegionOffset + s * BP::kPoolSlotBytes);
  }

  __device__ __forceinline__ ops::SmemAddr full_bar(int s) const {
    return at(BP::kPoolBarOffset + s * 8);
  }

  __device__ __forceinline__ ops::SmemAddr empty_bar(int s) const {
    return at(BP::kPoolBarOffset + (BP::kPoolSlots + s) * 8);
  }

  //: THE READOUT's private block of warp `w`: ring slot `r`, and the drain stage.
  __device__ __forceinline__ ops::SmemAddr rring(int warp, int r) const {
    return at(BP::kReadOffset + warp * BP::kWarpReadBytes + r * BP::kRRSlotBytes);
  }

  __device__ __forceinline__ ops::SmemAddr drain(int warp) const {
    return at(BP::kReadOffset + warp * BP::kWarpReadBytes + BP::kRRSlots * BP::kRRSlotBytes);
  }
};

//: A WARP'S OWN EDGE, a named barrier: orders the warp's private rows across its lanes, and
//: makes its landed `cp.async` data visible. -- carry_kernel.md#edges
constexpr int kWarpEdge = 8;

__device__ __forceinline__ void warp_edge(int warp) { ops::rendezvous_group(kWarpEdge + warp, 32); }

//: THE PHASE LEDGER: a warp's cycles per phase, accumulated in its shared row by lane 0
//: and added to the launch's ledger at the kernel's end. -- #phase-ledger
struct PhaseClock {
  long long* acc;
  long long mark;
  bool lead;

  __device__ __forceinline__ PhaseClock(long long* row, int lane)
      : acc(row), mark(ops::cycles()), lead(lane == 0) {
    if (lead) {
#pragma unroll
      for (int ph = 0; ph < kPhases; ++ph) acc[ph] = 0;
    }
  }

  //: the cycles since the last mark go to phase `ph`.
  __device__ __forceinline__ void lap(int ph) {
    const long long now = ops::cycles();
    if (lead) acc[ph] += now - mark;
    mark = now;
  }
};

//: A 32-BYTE ROW TILE's chunk, `[row][16 halfwords]`, the second chunk swizzled by the
//: row's bit two, so eight consecutive rows' `ldmatrix` is bank free; and an element.
__device__ __forceinline__ uint32_t row32_off(int row, int chunk) {
  return (uint32_t)(row * 32 + ((chunk ^ ((row >> 2) & 1)) * 16));
}

__device__ __forceinline__ uint32_t row32_elem(int row, int col) {
  return row32_off(row, col >> 3) + (uint32_t)((col & 7) * 2);
}

//: A CHANNEL ROW (`[row][kDv]` bf16: V rows in the pool, the snapshot's leaf rows), the
//: 16-byte chunk XORed with the row so eight consecutive rows' `ldmatrix` at one chunk is
//: bank free at every row width.
template <class BP>
__device__ __forceinline__ uint32_t chan_row_off(int row, int chunk) {
  constexpr int kGroups = BP::kVChunks < 8 ? BP::kVChunks : 8;
  constexpr int kShift = BP::kVChunks < 8 ? (BP::kVChunks == 4 ? 1 : 0) : 0;
  return (uint32_t)(row * BP::kVRowBytes + ((chunk ^ ((row >> kShift) & (kGroups - 1))) * 16));
}

//: THE POOL ROW's parts, in slot `s`: the V row, the inner run tile, the outer run tile.
template <class BP>
__device__ __forceinline__ uint32_t pool_v_off(int row, int chunk) {
  return (uint32_t)BP::kPoolVOffset + chan_row_off<BP>(row, chunk);
}

template <class BP>
__device__ __forceinline__ uint32_t pool_inner_off(int row, int chunk) {
  return (uint32_t)BP::kPoolInnerOffset + row32_off(row, chunk);
}

template <class BP>
__device__ __forceinline__ uint32_t pool_outer_off(int row, int chunk) {
  return (uint32_t)BP::kPoolOuterOffset + row32_off(row, chunk);
}

template <class BP>
__device__ __forceinline__ uint32_t pool_gain_off(int row) {
  return (uint32_t)(BP::kPoolGainOffset + row * 4);
}

//: a packed bf16 pair's halves as floats, and their dot products.
__device__ __forceinline__ float lo_f(uint32_t x) { return __uint_as_float(x << 16); }

__device__ __forceinline__ float hi_f(uint32_t x) { return __uint_as_float(x & 0xFFFF0000u); }

__device__ __forceinline__ float dot2(uint32_t a, uint32_t b) {
  return lo_f(a) * lo_f(b) + hi_f(a) * hi_f(b);
}

__device__ __forceinline__ float dot2f(uint32_t a, float b0, float b1) {
  return lo_f(a) * b0 + hi_f(a) * b1;
}

__device__ __forceinline__ uint32_t bf16_bits(float v) {
  return (uint32_t)__bfloat16_as_ushort(__float2bfloat16(v));
}

//: the token's packed row of amplitudes on a side, and level `l`'s run in it.
template <int D>
__device__ __forceinline__ const char* row_ptr(const CarryParams& p, int side, int bh, int t0,
                                               int tok) {
  const __nv_bfloat16* const plane = side == rola::facts::kWrite ? p.pwrite : p.pread;
  const uint32_t row =
      ((uint32_t)bh * (uint32_t)p.L + (uint32_t)(t0 + tok)) * (uint32_t)p.g.wtot * 2u;
  return reinterpret_cast<const char*>(ops::gather_ptr(plane, row));
}

// ------------------------------------------------------------------- the box words

//: THE BOX WORDS STAGED, a window ahead: each of the box's digit rows, the window's words
//: by `cp.async`, by warp `row % warps`. Issued past the histogram edge (every walk of the
//: buffer is before it); landed by the next head's wait. -- carry_kernel.md#box-words
template <class BP, int D, int BC>
__device__ __forceinline__ void stage_box_words(const CarryParams& p, const Smem<BP>& sm, int side,
                                                const int (&abase)[D], int warp, int lane, int bh,
                                                int t0, int nWords) {
  const int32_t* const table =
      p.liveness + ((long)bh * rola::facts::kSides + side) * ((long)p.g.wtot * p.liveness_words)
      + (t0 >> 5);
#pragma unroll 1
  for (int r = warp; r < BP::kBoxRows; r += BP::kWarps) {
    int row = 0;
    rola::static_for<D>([&](auto Lc) {
      constexpr int l = decltype(Lc)::value;
      if (r >= span_base(D, BC, l) && r < span_base(D, BC, l) + owner_span(D, BC, l))
        row = abase[l] + r - span_base(D, BC, l);
    });
    if (lane < nWords)
      ops::stage_run<4>(sm.words_addr(side, r) + (uint32_t)(lane * 4),
                        table + (long)row * p.liveness_words + lane);
  }
  ops::stage_commit();
}

// -------------------------------------------------------------------------- the head

//: THE BOX WORDS OF A ROUND on a side: word `b` has bit `t` set when token `32 rho + t`
//: is live in box `b`. The plain layout is two runtime row bases and static offsets; a
//: composed or straddling one walks its level lists in uniform loops. No branch on a lane
//: value (KERNEL_STANDARDS §20). -- carry_kernel.md#box-words
template <class BP, int D, int BC, bool Composed, bool Straddle>
__device__ __forceinline__ void box_words(const Smem<BP>& sm, const SideLayout& lay, int side,
                                          int rho, uint32_t (&w)[BP::kBoxes]) {
  if constexpr (!Composed && !Straddle) {
    static_assert(kAtomLeaves * BP::kBoxes == BC,
                  "the plain layout is sixteen digits by a digit a box");
    const uint32_t* const irow = sm.words(side, lay.inner_row) + rho;
    const uint32_t* const orow = sm.words(side, lay.outer_row) + rho;
    uint32_t any = 0u;
    rola::static_for<kAtomLeaves>([&](auto Dc) { any |= irow[decltype(Dc)::value * BP::kRounds]; });
    rola::static_for<BP::kBoxes>([&](auto Bc) {
      constexpr int b = decltype(Bc)::value;
      w[b] = any & orow[b * BP::kRounds];
    });
    return;
  }
  uint32_t any = ~0u;
#pragma unroll 1
  for (int i = 0; i < lay.n_inner; ++i) {
    const uint32_t* const row = sm.words(side, lay.in_row[i]) + rho;
    const int span = lay.in_span[i];
    uint32_t a = 0u;
    rola::static_for<kAtomLeaves>([&](auto Dc) {
      constexpr int d = decltype(Dc)::value;
      if (d < span) a |= row[d * BP::kRounds];
    });
    any &= a;
  }

  rola::static_for<BP::kBoxes>([&](auto Bc) { w[decltype(Bc)::value] = any; });
#pragma unroll 1
  for (int i = 0; i < lay.n_outer; ++i) {
    const uint32_t* const row = sm.words(side, lay.out_row[i]) + rho;
    const int bsh = lay.out_bshift[i], msk = lay.out_mask[i];
    rola::static_for<BP::kBoxes>([&](auto Bc) {
      constexpr int b = decltype(Bc)::value;
      w[b] &= row[((b >> bsh) & msk) * BP::kRounds];
    });
  }
  if constexpr (Straddle) {
    const int per_class = 1 << (kAtomLeavesBits - lay.sshift);
    rola::static_for<BP::kBoxes>([&](auto Bc) {
      constexpr int b = decltype(Bc)::value;
      const uint32_t* const row =
          sm.words(side,
                   lay.srow + ((b & ((1 << lay.cbits) - 1)) << (kAtomLeavesBits - lay.sshift)))
          + rho;
      uint32_t a = 0u;
      rola::static_for<kAtomLeaves / 2>([&](auto Dc) {
        constexpr int d = decltype(Dc)::value;
        if (d < per_class) a |= row[d * BP::kRounds];
      });
      w[b] &= a;
    });
  }
}

//: THE HEAD OF A WINDOW (component `head`). Read side: the ORDER (rank -> token, sorted by
//: first live box, the dead last), the TILE MASKS, the live count. Write side: each warp's
//: word a round (tokens live in a box it owns), the UNION word, its prefix by round. A warp
//: takes round groups, lane = token. BUDGET: carry_kernel.md#budgets. -- carry_kernel.md#order
template <class BP, int D, int BC, bool RComposed, bool RStraddle, bool WComposed, bool WStraddle>
__device__ __forceinline__ void head(const CarryParams& p, const Smem<BP>& sm,
                                     const SideLayout (&lay)[2], int parity, int wLen, int warp,
                                     int lane, int (&live)[2]) {
  constexpr int kGroups = BP::kRounds / BP::kWarps;
  static_assert(kGroups * BP::kWarps == BP::kRounds, "the round groups deal evenly");

  const uint32_t lt = (1u << lane) - 1u;
  int key[kGroups], local[kGroups];
  uint32_t mask[kGroups];

  //: PASS ONE, a round group: the read side's box words, the mask, the bucket, the counts;

  //: the write side's box words, the warps' words, the union word.

  rola::static_for<kGroups>([&](auto Gc) {
    constexpr int g = decltype(Gc)::value;
    const int rho = g * BP::kWarps + warp;
    const int t = rho * 32 + lane;
    const bool in = t < wLen;
    const uint32_t inmask = wLen - rho * 32 >= 32
                                ? 0xFFFFFFFFu
                                : (wLen - rho * 32 <= 0 ? 0u : (1u << (wLen - rho * 32)) - 1u);
    uint32_t w[BP::kBoxes];
    box_words<BP, D, BC, RComposed, RStraddle>(sm, lay[rola::facts::kRead], rola::facts::kRead, rho,
                                               w);
    uint32_t m = 0u;
    rola::static_for<BP::kBoxes>([&](auto Bc) {
      constexpr int b = BP::kBoxes - 1 - decltype(Bc)::value;
      m = (m << 1) | ((w[b] >> lane) & 1u);
    });
    m = in ? m : 0u;
    //: the bucket: the first live box, or bucket 0 for every live token under the identity
    //: order (the sort's A/B: token order, every tile); the dead to the last bucket.
    const int k = m != 0u ? (p.order == kOrderIdentity ? 0 : __ffs((int)m) - 1) : BP::kBoxes;
    const uint32_t same = __match_any_sync(0xFFFFFFFFu, k);
    uint16_t* const counts = sm.rcounts();
    if (lane < BP::kBuckets) counts[lane * BP::kRounds + rho] = 0;
    __syncwarp();
    if ((same & lt) == 0u) counts[k * BP::kRounds + rho] = (uint16_t)__popc(same);
    key[g] = k;
    local[g] = __popc(same & lt);
    mask[g] = m;
    box_words<BP, D, BC, WComposed, WStraddle>(sm, lay[rola::facts::kWrite], rola::facts::kWrite,
                                               rho, w);
    uint32_t u = 0u;
    rola::static_for<BP::kWarps>([&](auto Wc) {
      constexpr int v = decltype(Wc)::value;
      uint32_t ww = 0u;
      rola::static_for<BP::kDealt>([&](auto Ic) { ww |= w[v + decltype(Ic)::value * BP::kWarps]; });
      ww &= inmask;
      u |= ww;
      if (lane == v) sm.warpwords(parity, v)[rho] = ww;
    });
    if (lane == BP::kWarps) sm.unionwords(parity)[rho] = u;
  });

  ops::rendezvous_group((int)Edge::kCounts, BP::kThreads);

  //: THE SCANS. Warp 0: the read side's count table, a lane a bucket -- the bucket's counts

  //: by round become its bases in place, the totals scan across lanes, the live count is the

  //: dead bucket's base. Warp 1: the union's prefix by round, a lane a round.

  if (warp == 0) {
    const ops::SmemAddr row = sm.rcounts_addr() + (uint32_t)(lane * BP::kRounds * 2);
    uint32_t c[BP::kRounds / 2];
    int total = 0;
    if (lane < BP::kBuckets) {
      rola::static_for<BP::kRounds / 8>([&](auto Vc) {
        const uint4 v = ops::load_shared_v4(row + (uint32_t)(decltype(Vc)::value * 16));
        c[4 * decltype(Vc)::value] = v.x;
        c[4 * decltype(Vc)::value + 1] = v.y;
        c[4 * decltype(Vc)::value + 2] = v.z;
        c[4 * decltype(Vc)::value + 3] = v.w;
      });
      rola::static_for<BP::kRounds / 2>([&](auto Ic) {
        total += (int)(c[decltype(Ic)::value] & 0xFFFFu) + (int)(c[decltype(Ic)::value] >> 16);
      });
    }
    int excl = total;
    rola::static_for<5>([&](auto Sc) {
      constexpr int d = 1 << decltype(Sc)::value;
      const int o = __shfl_up_sync(0xFFFFFFFFu, excl, d);
      if (lane >= d) excl += o;
    });
    excl -= total;
    if (lane == BP::kBoxes) sm.live_count()[rola::facts::kRead] = excl;
    if (lane < BP::kBuckets) {
      int run = excl;
      rola::static_for<BP::kRounds / 2>([&](auto Ic) {
        constexpr int i = decltype(Ic)::value;
        const int lo = run;
        run += (int)(c[i] & 0xFFFFu);
        const int hi = run;
        run += (int)(c[i] >> 16);
        c[i] = (uint32_t)lo | ((uint32_t)hi << 16);
      });
      rola::static_for<BP::kRounds / 8>([&](auto Vc) {
        constexpr int v = decltype(Vc)::value;
        ops::store_shared_vec16(row + (uint32_t)(v * 16),
                                make_uint4(c[4 * v], c[4 * v + 1], c[4 * v + 2], c[4 * v + 3]));
      });
    }
  } else if (warp == 1) {
    const int n = lane < BP::kRounds ? __popc(sm.unionwords(parity)[lane]) : 0;
    int incl = n;
    rola::static_for<4>([&](auto Sc) {
      constexpr int d = 1 << decltype(Sc)::value;
      const int o = __shfl_up_sync(0xFFFFFFFFu, incl, d);
      if (lane >= d) incl += o;
    });
    if (lane < BP::kRounds) sm.prefix(parity)[lane + 1] = (uint16_t)incl;
    if (lane == 0) sm.prefix(parity)[0] = 0;
    if (lane == BP::kRounds - 1) sm.live_count()[rola::facts::kWrite] = incl;
  }

  ops::rendezvous_group((int)Edge::kCounts, BP::kThreads);

  //: PASS TWO: the read side's scatter.

  rola::static_for<kGroups>([&](auto Gc) {
    constexpr int g = decltype(Gc)::value;
    const int rho = g * BP::kWarps + warp;
    const int t = rho * 32 + lane;
    const int rank = (int)sm.rcounts()[key[g] * BP::kRounds + rho] + local[g];
    sm.order(parity)[rank] = (uint16_t)t;
    sm.masks()[rank] = (uint16_t)mask[g];
  });

  ops::rendezvous_group((int)Edge::kOrder, BP::kThreads);

  //: THE TILE MASKS: two tiles a warp a pass, sixteen lanes a tile, the OR of its ranks'.
  static_assert(BP::kTiles % (2 * BP::kWarps) == 0, "the tile-mask passes deal evenly");
#pragma unroll 1
  for (int pass = 0; pass < BP::kTiles / (2 * BP::kWarps); ++pass) {
    const int tile = pass * 2 * BP::kWarps + warp * 2 + (lane >> 4);
    uint32_t m = sm.masks()[tile * BP::kTile + (lane & 15)];
    rola::static_for<4>([&](auto Sc) {
      m |= (uint32_t)__shfl_xor_sync(0xFFFFFFFFu, (int)m, 8 >> decltype(Sc)::value);
    });
    if ((lane & 15) == 0) sm.tilemask(parity)[tile] = (uint16_t)m;
  }
  live[rola::facts::kRead] = sm.live_count()[rola::facts::kRead];
  live[rola::facts::kWrite] = sm.live_count()[rola::facts::kWrite];
}

// ------------------------------------------------------------------------- the state

//: THE STATE: `S[leaf][channel]` for the warp's owned boxes, `[box][n-tile]` MMA tiles in
//: the C layout (lane (r, q): leaves `r`, `r + 8` of the box at channels `2q`, `2q + 1` of
//: the n-tile); and the boxes' MASSES, an MMA accumulator each whose column 0 is the mass
//: (lane (r, 0): leaves `r` in `[0]`, `r + 8` in `[2]`; the rest stays zero). -- #state
template <class BP>
struct State {
  float c[BP::kDealt][BP::kNT][4];
  float m[BP::kDealt][4];
};

__device__ __forceinline__ constexpr int dealt_box(int warp, int bi, int warps) {
  return warp + bi * warps;
}

//: THE ENTRY SWEEP: the owned boxes off the pages, hi and lo joined; a page this call only
//: reads gives hi alone; a page without a slot, or inactive, is zero. THE EXIT SWEEP the
//: mirror, for the pages this call writes; the exit's mass is read off the mass row past
//: its edge. -- see docs/internals/carry/carry_kernel.md#state-io
template <class BP, int D, int BC, bool Load>
__device__ __forceinline__ void state_sweep(const Smem<BP>& sm, const SideLayout& wo, State<BP>& st,
                                            const int32_t* slots, uint32_t act_read,
                                            uint32_t act_write, int warp, int lane,
                                            const void* plane_in, void* plane_out) {
  const int r = lane >> 2, q = lane & 3;

  rola::static_for<BP::kDealt>([&](auto Bi) {
    constexpr int bi = decltype(Bi)::value;
    const int b = dealt_box(warp, bi, BP::kWarps);
    rola::static_for<BP::kNT>([&](auto Jc) {
      constexpr int j = decltype(Jc)::value;
      rola::static_for<4>([&](auto Ec) {
        constexpr int e = decltype(Ec)::value;
        const int leaf = b * 16 + r + 8 * (e >> 1);
        const int ch = 8 * j + 2 * q + (e & 1);
        const int local = wo.local<D>(leaf);
        const int atom = local >> kAtomLeavesBits, row = local & (kAtomLeaves - 1);
        const int slot = slots[atom];
        const bool present = slot >= 0 && (((act_read | act_write) >> atom) & 1u) != 0u;
        const bool with_lo = ((act_write >> atom) & 1u) != 0u;
        const uint32_t page = (uint32_t)(slot < 0 ? 0 : slot) * BP::Page::kBytes;
        const uint32_t off = page + (uint32_t)((row * BP::kDv + ch) * 2);
        float& x = st.c[bi][j][e];
        if constexpr (Load) {
          uint32_t hi = 0u, lo = 0u;
          if (present) hi = *reinterpret_cast<const uint16_t*>(ops::gather_ptr(plane_in, off));
          if (present && with_lo)
            lo = *reinterpret_cast<const uint16_t*>(ops::gather_ptr(plane_in, off + BP::Page::kLo));
          x = __uint_as_float((hi << 16) | lo);
        } else if (slot >= 0 && with_lo) {
          const uint32_t w = __float_as_uint(x);
          *reinterpret_cast<uint16_t*>(ops::gather_ptr(plane_out, off)) = (uint16_t)(w >> 16);
          *reinterpret_cast<uint16_t*>(ops::gather_ptr(plane_out, off + BP::Page::kLo)) =
              (uint16_t)(w & 0xFFFFu);
        }
      });
    });

    //: the masses: lane (r, 0) holds leaves `r` in `[0]`, `r + 8` in `[2]`.
    rola::static_for<2>([&](auto Hc) {
      constexpr int h = decltype(Hc)::value;
      const int leaf = b * 16 + r + 8 * h;
      const int ml = wo.local<D>(leaf);
      const int matom = ml >> kAtomLeavesBits, mrow = ml & (kAtomLeaves - 1);
      const int mslot = slots[matom];
      const bool mpresent = mslot >= 0 && (((act_read | act_write) >> matom) & 1u) != 0u;
      const bool mlo = ((act_write >> matom) & 1u) != 0u;
      const uint32_t mo =
          (uint32_t)(mslot < 0 ? 0 : mslot) * BP::Page::kBytes + (uint32_t)(mrow * 2);
      if constexpr (Load) {
        uint32_t w = 0u;
        if (mpresent) {
          w = (uint32_t)*reinterpret_cast<const uint16_t*>(
                  ops::gather_ptr(plane_in, mo + BP::Page::kMassHi))
              << 16;
          if (mlo)
            w |= *reinterpret_cast<const uint16_t*>(
                ops::gather_ptr(plane_in, mo + BP::Page::kMassLo));
        }
        st.m[bi][2 * h] = q == 0 ? __uint_as_float(w) : 0.0f;
        st.m[bi][2 * h + 1] = 0.0f;
      } else if (q == 0 && mslot >= 0 && mlo) {
        const uint32_t w = __float_as_uint(sm.massrow()[wo.plane_row<D>(leaf)]);
        *reinterpret_cast<uint16_t*>(ops::gather_ptr(plane_out, mo + BP::Page::kMassHi)) =
            (uint16_t)(w >> 16);
        *reinterpret_cast<uint16_t*>(ops::gather_ptr(plane_out, mo + BP::Page::kMassLo)) =
            (uint16_t)(w & 0xFFFFu);
      }
    });
  });
}

//: THE SNAPSHOT (component `snapshot`), once a window before the readout: the owned boxes'
//: high halves into the snapshot as `[leaf in read order][channel]` pairs, the readout's B
//: by `ldmatrix`; and their masses into the mass row in read order. Shared: its edge is
//: the window's. BUDGET: carry_kernel.md#budgets. -- carry_kernel.md#snapshot
template <class BP, int D, int BC>
__device__ __forceinline__ void snapshot_publish(const Smem<BP>& sm, const State<BP>& st, int warp,
                                                 int lane) {
  const int r = lane >> 2, q = lane & 3;
  const ops::SmemAddr snap = sm.snapshot();
  const uint16_t* const rowmap = sm.rowmap();
  rola::static_for<BP::kDealt>([&](auto Bi) {
    constexpr int bi = decltype(Bi)::value;
    const int b = dealt_box(warp, bi, BP::kWarps);
    rola::static_for<2>([&](auto Hc) {
      constexpr int h = decltype(Hc)::value;
      const int ro = (int)rowmap[b * 16 + r + 8 * h];
      rola::static_for<BP::kNT>([&](auto Jc) {
        constexpr int j = decltype(Jc)::value;
        ops::store_shared_u32(snap + chan_row_off<BP>(ro, j) + (uint32_t)(4 * q),
                              ops::hi_bf16x2(st.c[bi][j][2 * h], st.c[bi][j][2 * h + 1]));
      });
    });
    if (q == 0) {
      sm.massrow()[(int)rowmap[b * 16 + r]] = st.m[bi][0];
      sm.massrow()[(int)rowmap[b * 16 + r + 8]] = st.m[bi][2];
    }
  });
}

// ------------------------------------------------------------------------- the pool

//: THE RUN BASES of a side's single inner and outer levels in the amplitude row, bytes:
//: `abase` selected by the layout's levels statically (a runtime index would put the
//: array in local memory).
template <int D>
__device__ __forceinline__ int2 run_bases(const SideLayout& lay, const int (&abase)[D]) {
  int inner = 0, outer = 0;
  rola::static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    if (lay.single_inner == l) inner = abase[l] * 2;
    if (lay.single_outer == l) outer = abase[l] * 2;
  });
  return make_int2(inner, outer);
}

//: THE POOL FILL (component `fill`): chunk `c` of the union into slot `s`: the chunk's rounds
//: by ballot over the prefix, dealt to warps; a round two passes of sixteen tokens, lanes
//: `j`, `j + 16` a token; V, the runs and the gain pair by `cp.async`, idle lanes to the
//: zero row; a slot is full at `kThreads` landed arrivals. Plain layouts only. BUDGET:
//: carry_kernel.md#budgets. -- carry_kernel.md#pool
template <class BP, int D, int BC>
__device__ __forceinline__ void pool_fill(const CarryParams& p, const Smem<BP>& sm, int2 bases,
                                          int parity, int chunk, int s, uint32_t rowbase, int warp,
                                          int lane) {
  static_assert(kAtomLeaves * BP::kBoxes == BC, "the fill copies plain sixteen-digit runs");
  static_assert(2 * BP::kWarps >= BP::kRounds, "a warp fills at most two rounds of a chunk");
  const int j = lane & 15, h = lane >> 4;
  const int lo = chunk * BP::kPoolTok, hi = lo + BP::kPoolTok;

  //: the chunk's rounds: the last round starting at or before `lo`, the last starting

  //: before `hi`.
  const int pr = lane < BP::kRounds ? (int)sm.prefix(parity)[lane] : 0x7FFFFFFF;
  const int r0 = ops::last_set(__ballot_sync(0xFFFFFFFFu, pr <= lo));
  const int r1 = ops::last_set(__ballot_sync(0xFFFFFFFFu, pr < hi));
  const ops::SmemAddr slot = sm.pool(s);

  rola::static_for<2>([&](auto Ic) {
    constexpr int i = decltype(Ic)::value;
    const int rr = r0 + warp + i * BP::kWarps;
    //: a warp without this round skips it: an opaque one-trip loop, not an `if` ptxas
    //: would if-convert into passes of zero-size copies (KERNEL_STANDARDS §20).
#pragma unroll 1
    for (int n = ops::once(rr <= r1); n > 0; --n) {
      const uint32_t u = sm.unionwords(parity)[rr];
      const int base = (int)sm.prefix(parity)[rr];
      rola::static_for<2>([&](auto Hc) {
        constexpr int half = decltype(Hc)::value;
        const int t = half * 16 + j;
        const int rank = base + __popc(u & ((1u << t) - 1u));
        const bool in = ((u >> t) & 1u) != 0u && rank >= lo && rank < hi;
        const int row = in ? rank - lo : BP::kPoolTok;
        const uint32_t trow = rowbase + (uint32_t)(in ? rr * 32 + t : 0);
        const char* const vrow =
            reinterpret_cast<const char*>(ops::gather_ptr(p.v, trow * (uint32_t)BP::kVRowBytes));
        rola::static_for<BP::kVChunks / 2>([&](auto Cc) {
          const int c = 2 * decltype(Cc)::value + h;
          ops::stage_run_if<16>(slot + pool_v_off<BP>(row, c), vrow + c * 16, in);
        });
        const char* const arow = reinterpret_cast<const char*>(
            ops::gather_ptr(p.pwrite, trow * (uint32_t)p.g.wtot * 2u));
        ops::stage_run_if<16>(slot + pool_inner_off<BP>(row, h), arow + bases.x + h * 16, in);
        ops::stage_run_if<16>(slot + pool_outer_off<BP>(row, h), arow + bases.y + h * 16, in);
        if (h == 0)
          ops::stage_run_if<4>(slot + pool_gain_off<BP>(row),
                               ops::gather_ptr(p.gwrite, (trow & ~1u) * 2u), in);
      });
    }
  });
  ops::stage_commit();

  ops::mbar_arrive_when_landed(sm.full_bar(s));
}

//: THE POOL CURSOR: a window's chunks take slots from 0 in turn, so chunk 0 always lands
//: in slot 0, the one free under the readout; each slot's use count (a byte a slot) gives
//: its barriers' parities: the n-th fill of a slot waits `empty` at parity `(n - 1) & 1`
//: (no wait for n = 0) and is taken at `full` parity `n & 1`.
template <class BP>
struct PoolCursor {
  uint32_t uses;

  __device__ __forceinline__ int n_of(int s) const { return (int)((uses >> (8 * s)) & 0xFFu); }

  __device__ __forceinline__ void wait_empty(const Smem<BP>& sm, int s) const {
    const int n = n_of(s);
    if (n > 0) ops::mbar_wait(sm.empty_bar(s), (uint32_t)(n - 1) & 1u);
  }

  __device__ __forceinline__ void wait_full(const Smem<BP>& sm, int s) const {
    ops::mbar_wait(sm.full_bar(s), (uint32_t)n_of(s) & 1u);
  }

  __device__ __forceinline__ void used(int s) { uses += 1u << (8 * s); }
};

// ----------------------------------------------------------------------- the fold

//: THE COMPOSED TILES of a slot, in the warp's scratch: the inner tile `[token][pos]` as
//: the product of the inner levels' runs, the outer table `[token][box]` as the product
//: of the outer levels' -- each only when more than one level makes it (else the raw
//: tile is it). Lane `2 i + h` writes row `i`'s positions / boxes `8 h ..`. Products in
//: fp32, rounded once.
template <class BP>
__device__ __forceinline__ float run_at(ops::SmemAddr raw, int l, int i, int digit) {
  return ops::bf16_bits_to_float(
      ops::load_shared_u16(raw + (uint32_t)(l * BP::kRunTileBytes) + row32_elem(i, digit)));
}

template <class BP, int D, int BC, bool Composed>
__device__ __forceinline__ void compose(const Smem<BP>& sm, const SideLayout& lay,
                                        ops::SmemAddr raw, int warp, int lane, ops::SmemAddr& inner,
                                        ops::SmemAddr& outer) {
  if constexpr (!Composed) {
    inner = raw + (uint32_t)(lay.single_inner * BP::kRunTileBytes);
    outer = raw + (uint32_t)(lay.single_outer * BP::kRunTileBytes);
    return;
  }
  const int i = lane >> 1, h = lane & 1;

  const auto run = [&](int l, int digit) -> float { return run_at<BP>(raw, l, i, digit); };

  if (lay.single_inner >= 0) {
    inner = raw + (uint32_t)(lay.single_inner * BP::kRunTileBytes);
  } else {
    uint32_t w[4];
    rola::static_for<4>([&](auto Jc) {
      constexpr int j2 = decltype(Jc)::value;
      uint32_t pair = 0u;
      rola::static_for<2>([&](auto Ec) {
        constexpr int e = decltype(Ec)::value;
        const int pos = 8 * h + 2 * j2 + e;
        float v = 1.0f;
        rola::static_for<D>([&](auto Lc) {
          constexpr int l = decltype(Lc)::value;
          if ((lay.inner >> l) & 1u) v *= run(l, lay.digit_of_pos(l, pos));
        });
        pair |= bf16_bits(v) << (16 * e);
      });
      w[j2] = pair;
    });
    inner = sm.scratch(warp, BP::kScratchInner);
    ops::store_shared_vec16(inner + row32_off(i, h), make_uint4(w[0], w[1], w[2], w[3]));
  }

  if (lay.single_outer >= 0) {
    outer = raw + (uint32_t)(lay.single_outer * BP::kRunTileBytes);
  } else {
    uint32_t w[4];
    rola::static_for<4>([&](auto Jc) {
      constexpr int j2 = decltype(Jc)::value;
      uint32_t pair = 0u;
      rola::static_for<2>([&](auto Ec) {
        constexpr int e = decltype(Ec)::value;
        const int box = 8 * h + 2 * j2 + e;
        float v = box < BP::kBoxes ? 1.0f : 0.0f;
        rola::static_for<D>([&](auto Lc) {
          constexpr int l = decltype(Lc)::value;
          if ((lay.outer >> l) & 1u) v *= run(l, lay.digit_of_box(l, box & (BP::kBoxes - 1)));
        });
        pair |= bf16_bits(v) << (16 * e);
      });
      w[j2] = pair;
    });
    outer = sm.scratch(warp, BP::kScratchOuter);
    ops::store_shared_vec16(outer + row32_off(i, h), make_uint4(w[0], w[1], w[2], w[3]));
  }
  warp_edge(warp);
}

//: THE STRADDLING LEVEL applied to a fragment for box `b`: a packed pair of its digits
//: for two (token, position) elements.
template <class BP, int D, int BC>
__device__ __forceinline__ uint32_t straddle_pair(const SideLayout& lay, ops::SmemAddr raw, int b,
                                                  int tok0, int pos0, int tok1, int pos1) {
  const ops::SmemAddr t = raw + (uint32_t)(lay.straddle * BP::kRunTileBytes);
  const uint32_t x0 = ops::load_shared_u16(t + row32_elem(tok0, lay.straddle_digit(b, pos0)));
  const uint32_t x1 = ops::load_shared_u16(t + row32_elem(tok1, lay.straddle_digit(b, pos1)));
  return x0 | (x1 << 16);
}

//: THE FRAGMENT LANE MAPS: the row lane `L` addresses for the run tiles (`ldmatrix.trans`
//: into the fold's A and the outer pairs) and for the V rows (B). -- carry_kernel.md#fold
__device__ __forceinline__ int run_lane_row(int lane) { return (lane & 7) + 8 * (lane >> 4); }

__device__ __forceinline__ int v_lane_row(int lane) { return (lane & 7) + 8 * ((lane >> 3) & 1); }

//: THE FOLD FRAGMENT: sixteen pool rows (lane `j < 16` holds entry `j`: row | gain << 16);
//: A, the outer pairs and V gathered once, the pairs scaled by the gains; then a box: its
//: pairs by shuffle, A scaled, `kNT` HMMAs into its state, one into its mass. -- #fold
template <class BP, int D, int BC>
__device__ __forceinline__ void fold_fragment(ops::SmemAddr slot, uint32_t entry, int warp,
                                              int lane, State<BP>& st) {
  const int q = lane & 3;

  const uint32_t re = (uint32_t)__shfl_sync(0xFFFFFFFFu, (int)entry, run_lane_row(lane));

  const uint32_t ve = (uint32_t)__shfl_sync(0xFFFFFFFFu, (int)entry, v_lane_row(lane));
  const int rrow = (int)(re & 0xFFu), vrow = (int)(ve & 0xFFu);
  const int rchunk = (lane >> 3) & 1;
  uint32_t a[4], sp[4], v[2 * BP::kNT];
  ops::load_frag_t(a, slot + pool_inner_off<BP>(rrow, rchunk));
  ops::load_frag_t(sp, slot + pool_outer_off<BP>(rrow, rchunk));

  rola::static_for<BP::kNT / 2>([&](auto Ic) {
    constexpr int i = decltype(Ic)::value;
    ops::load_frag_t(*reinterpret_cast<uint32_t(*)[4]>(v + 4 * i),
                     slot + pool_v_off<BP>(vrow, 2 * i + (lane >> 4)));
  });

  //: the gains of tokens 2q, 2q + 1 and 2q + 8, 2q + 9, off the entries' high halves.
  uint32_t gp[2];

  rola::static_for<2>([&](auto Hc) {
    constexpr int h = decltype(Hc)::value;
    const uint32_t e0 = (uint32_t)__shfl_sync(0xFFFFFFFFu, (int)entry, 2 * q + 8 * h);
    const uint32_t e1 = (uint32_t)__shfl_sync(0xFFFFFFFFu, (int)entry, 2 * q + 8 * h + 1);
    gp[h] = __byte_perm(e0, e1, 0x7632);
  });

  //: the transposed outer tile's pairs: `[0]` (box r, tokens 2q..), `[1]` (box r + 8,

  //: 2q..), `[2]` (box r, 2q + 8..), `[3]` (box r + 8, 2q + 8..).
  sp[0] = ops::mul_bf16x2(sp[0], gp[0]);
  sp[1] = ops::mul_bf16x2(sp[1], gp[0]);
  sp[2] = ops::mul_bf16x2(sp[2], gp[1]);
  sp[3] = ops::mul_bf16x2(sp[3], gp[1]);

  rola::static_for<BP::kDealt>([&](auto Bi) {
    constexpr int bi = decltype(Bi)::value;
    const int b = dealt_box(warp, bi, BP::kWarps);
    const int src = ((b & 7) << 2) | q;
    const uint32_t s0 = (uint32_t)__shfl_sync(0xFFFFFFFFu, (int)sp[b < 8 ? 0 : 1], src);
    const uint32_t s1 = (uint32_t)__shfl_sync(0xFFFFFFFFu, (int)sp[b < 8 ? 2 : 3], src);
    const uint32_t ab[4] = {ops::mul_bf16x2(a[0], s0), ops::mul_bf16x2(a[1], s0),
                            ops::mul_bf16x2(a[2], s1), ops::mul_bf16x2(a[3], s1)};
    rola::static_for<BP::kNT>([&](auto Jc) {
      constexpr int j = decltype(Jc)::value;
      ops::mma(st.c[bi][j], ab, v + 2 * j);
    });
    const uint32_t ones = (lane >> 2) == 0 ? 0x3F803F80u : 0u;
    const uint32_t bm[2] = {ones, ones};
    ops::mma(st.m[bi], ab, bm);
  });
}

//: THE FOLD (component `fold`): the write side over the pool. Per chunk the warp walks the
//: chunk's rounds, kept lanes queue their rows through the walk ring, every sixteen a
//: fragment; the tail pads from the zero row; the chunk two ahead is filled before the
//: walk; the slot is released after the last gather. BUDGET: carry_kernel.md#budgets. -- #fold
template <class BP, int D, int BC, bool Composed, bool Straddle>
__device__ __forceinline__ void fold(const CarryParams& p, const Smem<BP>& sm,
                                     const SideLayout& lay, int2 bases, int parity, int live,
                                     uint32_t rowbase, int warp, int lane, State<BP>& st,
                                     PoolCursor<BP>& cur, PhaseClock& pc) {
  static_assert(!Composed && !Straddle, "the fold is spelled for the plain layout");
  (void)lay;
  const int chunks = (live + BP::kPoolTok - 1) / BP::kPoolTok;
  //: chunk 0 landed under the readout; chunk 1 is issued here, chunk `c + 2` under chunk `c`.
  if (chunks > 1) {
    cur.wait_empty(sm, 1);
    pool_fill<BP, D, BC>(p, sm, bases, parity, 1, 1, rowbase, warp, lane);
  }
  uint32_t* const ring = sm.walk(warp);
  const uint32_t* const ww = sm.warpwords(parity, warp);
  const uint32_t* const uw = sm.unionwords(parity);
  const uint16_t* const prefix = sm.prefix(parity);
  const uint32_t lt = (1u << lane) - 1u;
#pragma unroll 1
  for (int c = 0; c < chunks; ++c) {
    const int s = c % BP::kPoolSlots;
    if (c + 2 < chunks) {
      const int s2 = (c + 2) % BP::kPoolSlots;
      cur.wait_empty(sm, s2);
      pool_fill<BP, D, BC>(p, sm, bases, parity, c + 2, s2, rowbase, warp, lane);
    }
    const ops::SmemAddr slot = sm.pool(s);
    cur.wait_full(sm, s);
    const int lo = c * BP::kPoolTok, hi = lo + BP::kPoolTok;
    const int pr = lane < BP::kRounds ? (int)prefix[lane] : 0x7FFFFFFF;
    const int r0 = ops::last_set(__ballot_sync(0xFFFFFFFFu, pr <= lo));
    const int r1 = ops::last_set(__ballot_sync(0xFFFFFFFFu, pr < hi));
    int queued = 0, taken = 0;
#pragma unroll 1
    for (int rr = r0; rr <= r1; ++rr) {
      const uint32_t u = uw[rr], w = ww[rr];
      const int rank = (int)prefix[rr] + __popc(u & lt);
      const bool in = ((w >> lane) & 1u) != 0u && rank >= lo && rank < hi;
      const uint32_t kept = __ballot_sync(0xFFFFFFFFu, in);
      if (in) {
        const int slot_ix = (queued + __popc(kept & lt)) & (BP::kWalkEntries - 1);
        //: the token's gain: the half of its pool row's gain pair its parity names.
        const uint32_t gw = ops::load_shared_u32(slot + pool_gain_off<BP>(rank - lo));
        const uint32_t g = (lane & 1) ? (gw >> 16) : (gw & 0xFFFFu);
        ring[slot_ix] = (uint32_t)(rank - lo) | (g << 16);
      }
      queued += __popc(kept);
      __syncwarp();
#pragma unroll 1
      while (queued - taken >= BP::kTile) {
        const uint32_t entry = ring[(taken + (lane & 15)) & (BP::kWalkEntries - 1)];
        fold_fragment<BP, D, BC>(slot, entry, warp, lane, st);
        taken += BP::kTile;
      }
    }
    //: the tail: the queued rows past the last fragment, the rest from the zero row.
#pragma unroll 1
    for (int n = ops::once(queued > taken); n > 0; --n) {
      const int j = lane & 15;
      const uint32_t entry =
          j < queued - taken ? ring[(taken + j) & (BP::kWalkEntries - 1)] : (uint32_t)BP::kPoolTok;
      fold_fragment<BP, D, BC>(slot, entry, warp, lane, st);
    }
    __syncwarp();
    if (lane == 0) ops::mbar_arrive(sm.empty_bar(s));
    cur.used(s);
  }
  pc.lap(kPhaseFold);
}

// -------------------------------------------------------------------- the readout

//: THE READOUT TILE's ring copies: the tile's sixteen tokens' inner and outer runs into
//: the warp's ring slot, lanes `l` and `l + 16` a row (alternate 16-byte halves), a rank
//: past the live count zero-filled; one copy group.
template <class BP, int D, int BC>
__device__ __forceinline__ void readout_issue(const CarryParams& p, const Smem<BP>& sm, int2 bases,
                                              const uint16_t* order, int k, int live,
                                              uint32_t rowbase, int warp, int slot_ix, int lane) {
  const int row = lane & 15, h = lane >> 4;
  const int rank = k * BP::kTile + row;
  const bool in = rank < live;
  const uint32_t tok = in ? order[rank] : 0u;
  const char* const arow = reinterpret_cast<const char*>(
      ops::gather_ptr(p.pread, (rowbase + tok) * (uint32_t)p.g.wtot * 2u));
  const ops::SmemAddr slot = sm.rring(warp, slot_ix);
  ops::stage_run_if<16>(slot + row32_off(row, h), arow + bases.x + h * 16, in);
  ops::stage_run_if<16>(slot + (uint32_t)BP::kRunTileBytes + row32_off(row, h),
                        arow + bases.y + h * 16, in);
  ops::stage_commit();
}

//: THE READOUT TILE: A (the inner tile, `ldmatrix`) once; per live box: the rows' outer
//: factors, A scaled, B the box off the snapshot (`ldmatrix.trans`, two n-tiles a load),
//: `kNT` HMMAs into y and one against the box's mass column into den; then the rows out:
//: y through the drain stage four rows at a time as whole lines by reduction, den by
//: reduction from the lanes holding it.
template <class BP, int D, int BC>
__device__ __forceinline__ void readout_tile(const Smem<BP>& sm, ops::SmemAddr slot, uint32_t mask,
                                             const uint16_t* order, int k, int live, float* num,
                                             float* den, int t0, int warp, int lane) {
  const int r = lane >> 2, q = lane & 3;
  uint32_t a[4];
  ops::load_frag(a, slot + row32_off((lane & 7) + 8 * ((lane >> 3) & 1), lane >> 4));
  const ops::SmemAddr outer = slot + (uint32_t)BP::kRunTileBytes;
  float y[BP::kNT][4], d[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  rola::static_for<BP::kNT>([&](auto Jc) {
    rola::static_for<4>([&](auto Ec) { y[decltype(Jc)::value][decltype(Ec)::value] = 0.0f; });
  });
  const int snap_row = (lane & 7) + 8 * ((lane >> 3) & 1);
  const float* const massrow = sm.massrow();
  uint32_t rest = mask;
#pragma unroll 1
  while (rest != 0u) {
    const int b = __ffs((int)rest) - 1;
    rest &= rest - 1u;
    const uint32_t o_r = ops::splat_u16(ops::load_shared_u16(outer + row32_elem(r, b)));
    const uint32_t o_r8 = ops::splat_u16(ops::load_shared_u16(outer + row32_elem(r + 8, b)));
    const uint32_t ab[4] = {ops::mul_bf16x2(a[0], o_r), ops::mul_bf16x2(a[1], o_r8),
                            ops::mul_bf16x2(a[2], o_r), ops::mul_bf16x2(a[3], o_r8)};
    const ops::SmemAddr brow = sm.snapshot() + chan_row_off<BP>(b * 16 + snap_row, 0);
    rola::static_for<BP::kNT / 2>([&](auto Ic) {
      constexpr int i = decltype(Ic)::value;
      uint32_t bf[4];
      //: the chunk index XORs with the row's swizzle: chunk `c` of the row is chunk 0's
      //: address XOR `16 c`.
      ops::load_frag_t(bf, brow ^ (uint32_t)((2 * i + (lane >> 4)) * 16));
      ops::mma(y[2 * i], ab, bf);
      ops::mma(y[2 * i + 1], ab, bf + 2);
    });
    //: den: the box's masses as a B column, lanes `r == 0`.
    uint32_t bm[2] = {0u, 0u};
    if (r == 0) {
      const float2 m0 = *reinterpret_cast<const float2*>(massrow + b * 16 + 2 * q);
      const float2 m1 = *reinterpret_cast<const float2*>(massrow + b * 16 + 2 * q + 8);
      bm[0] = ops::hi_bf16x2(m0.x, m0.y);
      bm[1] = ops::hi_bf16x2(m1.x, m1.y);
    }
    ops::mma(d, ab, bm);
  }

  //: THE ROWS OUT. Four rows a pass through the stage: lanes `r` in the pass's quarter

  //: write their row's pairs; then each lane reduces one float of each of the four rows'

  //: two lines. Dead ranks (past the live count) leave nothing.
  const ops::SmemAddr stage = sm.drain(warp);

  rola::static_for<4>([&](auto Pc) {
    constexpr int pass = decltype(Pc)::value;
    constexpr int rlo = (pass & 1) * 4, hsel = pass >> 1;  //: rows rlo..rlo+3 of half hsel
    __syncwarp();
    if ((r & 4) == (rlo & 4)) {
      const int srow = r & 3;
      rola::static_for<BP::kNT>([&](auto Jc) {
        constexpr int j = decltype(Jc)::value;
        ops::store_shared_u64(stage + (uint32_t)(srow * BP::kVRowBytes * 2 + (8 * j + 2 * q) * 4),
                              __float_as_uint(y[j][2 * hsel]), __float_as_uint(y[j][2 * hsel + 1]));
      });
    }
    __syncwarp();
    rola::static_for<4>([&](auto Rc) {
      constexpr int srow = decltype(Rc)::value;
      const int rank = k * BP::kTile + rlo + srow + 8 * hsel;
      if (rank < live) {
        const int tok = (int)order[rank];
        float* const row_out = num + (long)(t0 + tok) * BP::kDv;
        rola::static_for<BP::kDv / 32>([&](auto Lc) {
          constexpr int line = decltype(Lc)::value;
          const float v = ops::load_shared_f32(
              stage + (uint32_t)(srow * BP::kVRowBytes * 2 + (line * 32 + lane) * 4));
          ops::red_global_add_f32(row_out + line * 32 + lane, v);
        });
      }
    });
  });

  if (q == 0) {
    const int rank0 = k * BP::kTile + r, rank1 = rank0 + 8;
    if (rank0 < live) ops::red_global_add_f32(den + t0 + (int)order[rank0], d[0]);
    if (rank1 < live) ops::red_global_add_f32(den + t0 + (int)order[rank1], d[2]);
  }
}

//: THE READOUT (component `readout`): the read order's tiles, a warp its own (tile `k` to
//: warp `k % warps`), staged into the warp's ring a tile ahead. Per tile `readout_tile`.
//: BUDGET: carry_kernel.md#budgets. -- carry_kernel.md#readout
template <class BP, int D, int BC, bool Composed, bool Straddle>
__device__ __forceinline__ void readout(const CarryParams& p, const Smem<BP>& sm,
                                        const SideLayout& lay, int2 bases, int parity, int live,
                                        float* num, float* den, uint32_t rowbase, int t0, int warp,
                                        int lane, PhaseClock& pc) {
  static_assert(!Composed && !Straddle, "the readout is spelled for the plain layout");
  (void)lay;
  const uint16_t* const order = sm.order(parity);
  const uint16_t* const tmask = sm.tilemask(parity);
  const int tiles = (live + BP::kTile - 1) / BP::kTile;
  int n = 0;
  if (warp < tiles)
    readout_issue<BP, D, BC>(p, sm, bases, order, warp, live, rowbase, warp, 0, lane);
#pragma unroll 1
  for (int k = warp; k < tiles; k += BP::kWarps, ++n) {
    const int slot_ix = n & (BP::kRRSlots - 1);
    if (k + BP::kWarps < tiles)
      readout_issue<BP, D, BC>(p, sm, bases, order, k + BP::kWarps, live, rowbase, warp,
                               (n + 1) & (BP::kRRSlots - 1), lane);
    else
      ops::stage_commit();
    ops::stage_wait<1>();
    __syncwarp();
    readout_tile<BP, D, BC>(sm, sm.rring(warp, slot_ix), (uint32_t)tmask[k], order, k, live, num,
                            den, t0, warp, lane);
  }
  pc.lap(kPhaseReadout);
}

// ------------------------------------------------------------------- the window loop

template <class BP, int D, int BC, bool RComposed, bool RStraddle, bool WComposed, bool WStraddle>
__device__ __forceinline__ void window_loop(const CarryParams& p, const Smem<BP>& sm,
                                            const SideLayout (&lay)[2], State<BP>& st, int owner,
                                            float* num, float* den, int tid, int warp, int lane,
                                            int bh, PhaseClock& pc) {
  //: THE BOX'S FACTOR RUNS, which are ALSO the class-1 table's row bases. -- #factor-rows
  int abase[D];
  rola::static_for<D>([&](auto Lc) {
    constexpr int l = decltype(Lc)::value;
    abase[l] = p.g.col_base[l] + geom_digit<D, BC>(p.g, owner, 0, l);
  });
  const int2 wbases = run_bases<D>(lay[rola::facts::kWrite], abase);
  const int2 rbases = run_bases<D>(lay[rola::facts::kRead], abase);
  const int nWindows = (p.L + kWindow - 1) / kWindow;
  const auto len_of = [&](int win) {
    const int rest = p.L - win * kWindow;
    return rest < 0 ? 0 : (rest < kWindow ? rest : kWindow);
  };
  const auto words_of = [&](int win) { return (len_of(win) + 31) / 32; };
  const auto stage_words = [&](int win) {
    stage_box_words<BP, D, BC>(p, sm, rola::facts::kRead, abase, warp, lane, bh, win * kWindow,
                               words_of(win));
    stage_box_words<BP, D, BC>(p, sm, rola::facts::kWrite, abase, warp, lane, bh, win * kWindow,
                               words_of(win));
  };
  //: THE POOL'S BARRIERS: `kThreads` landed-copy arrivals fill a slot; `kWarps` releases
  //: empty it. THE ZERO ROW of every slot.
  if (tid < BP::kPoolSlots) {
    ops::mbar_init(sm.full_bar(tid), BP::kThreads);
    ops::mbar_init(sm.empty_bar(tid), BP::kWarps);
  }
  constexpr int kZeroParts = BP::kVChunks + 4;
#pragma unroll 1
  for (int i = tid; i < BP::kPoolSlots * kZeroParts; i += BP::kThreads) {
    const int s = i / kZeroParts, part = i % kZeroParts;
    if (part < BP::kVChunks)
      ops::store_shared_vec16(sm.pool(s) + pool_v_off<BP>(BP::kPoolTok, part),
                              make_uint4(0u, 0u, 0u, 0u));
    else if (part < BP::kVChunks + 2)
      ops::store_shared_vec16(sm.pool(s) + pool_inner_off<BP>(BP::kPoolTok, part - BP::kVChunks),
                              make_uint4(0u, 0u, 0u, 0u));
    else if (part < BP::kVChunks + 4)
      ops::store_shared_vec16(
          sm.pool(s) + pool_outer_off<BP>(BP::kPoolTok, part - BP::kVChunks - 2),
          make_uint4(0u, 0u, 0u, 0u));
    if (part == 0) ops::store_shared_u32(sm.pool(s) + pool_gain_off<BP>(BP::kPoolTok), 0u);
  }
  PoolCursor<BP> cur{0u};
  stage_words(0);
  ops::stage_wait<0>();
  ops::rendezvous();
  pc.lap(kPhaseEdges);
  //: THE HEAD of window `win`, its words landed; the next window's words issued after it;
  //: the window's first two pool chunks issued after it (into slots the last fold released).
  int live[2] = {0, 0};
  const auto run_head = [&](int win) {
    head<BP, D, BC, RComposed, RStraddle, WComposed, WStraddle>(p, sm, lay, win & 1, len_of(win),
                                                                warp, lane, live);
    if (win + 1 < nWindows) stage_words(win + 1);
    pc.lap(kPhaseHead);
  };
  //: THE WINDOW'S FIRST CHUNK, into slot 0 -- the slot free under the readout (the slot's
  //: previous reader finished in the last fold).
  const auto issue_first = [&](int win) {
    const uint32_t rowbase = (uint32_t)bh * (uint32_t)p.L + (uint32_t)(win * kWindow);
    if (live[rola::facts::kWrite] > 0) {
      cur.wait_empty(sm, 0);
      pool_fill<BP, D, BC>(p, sm, wbases, win & 1, 0, 0, rowbase, warp, lane);
    }
  };
  run_head(0);
  issue_first(0);

#pragma unroll 1
  for (int win = 0; win < nWindows; ++win) {
    const int t0 = win * kWindow, parity = win & 1;
    const uint32_t rowbase = (uint32_t)bh * (uint32_t)p.L + (uint32_t)t0;
    const int rlive = live[rola::facts::kRead], wlive = live[rola::facts::kWrite];
    //: THE WINDOW'S EDGE: the state as the readout reads it.
    if (rlive > 0) snapshot_publish<BP, D, BC>(sm, st, warp, lane);
    pc.lap(kPhaseSnapshot);
    ops::rendezvous();
    pc.lap(kPhaseEdges);
    readout<BP, D, BC, RComposed, RStraddle>(p, sm, lay[rola::facts::kRead], rbases, parity, rlive,
                                             num, den, rowbase, t0, warp, lane, pc);
    //: the next window's words, issued a window ago, landed before its head.
    ops::stage_wait<0>();
    ops::rendezvous();
    pc.lap(kPhaseEdges);
    const int wlive_now = wlive;
    if (win + 1 < nWindows) run_head(win + 1);
    fold<BP, D, BC, WComposed, WStraddle>(p, sm, lay[rola::facts::kWrite], wbases, parity,
                                          wlive_now, rowbase, warp, lane, st, cur, pc);
    if (win + 1 < nWindows) issue_first(win + 1);
  }
}

// ------------------------------------------------------------------------- the kernel

template <int D, int DV, int WARPS>
__global__ __launch_bounds__(BoxPlan<D, DV, WARPS>::kThreads, 1) void carry_kernel(
    __grid_constant__ const CarryParams p) {
  using BP = BoxPlan<D, DV, WARPS>;
  constexpr int BC = BP::kBC;

  extern __shared__ __align__(16) char smem_raw[];
  const Smem<BP> sm{smem_raw};

  const int owner = (int)blockIdx.x;
  const int bh = (int)blockIdx.y;
  //: ONE 64-BIT BASE PER READOUT PLANE, and 32-bit byte displacements after it.
  float* const num = ops::pin_address(p.num + (long)bh * (long)p.L * (long)BP::kDv);
  float* const den = ops::pin_address(p.den + (long)bh * (long)p.L);
  const int tid = (int)threadIdx.x;
  const int warp = ops::uniform_warp<BP::kWarps>(), lane = tid & 31;
  PhaseClock pc(sm.ledger(warp), lane);

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

  //: THE BOX'S PAGE SLOTS -- the only page-table reads in the launch -- and THE ROW MAP.
  if (tid < BP::kAtoms) {
    const uint32_t canon =
        (uint32_t)(geom_canon_leaf_t<D, BC>(p.g, owner, tid * kAtomLeaves)) >> kAtomLeavesBits;
    sm.page_slots()[tid] = ops::page_slot(p.page_table, atom_base + canon);
  }
  const SideLayout(&lay)[2] = p.lay;
#pragma unroll 1
  for (int v = tid; v < BC; v += BP::kThreads)
    sm.rowmap()[v] = (uint16_t)lay[rola::facts::kWrite].plane_row<D>(v);
  ops::rendezvous();
  const int32_t* const slots = sm.page_slots();

  State<BP> st;
  rola::static_for<BP::kDealt>([&](auto Bi) {
    rola::static_for<BP::kNT>([&](auto Jc) {
      rola::static_for<4>([&](auto Kc) {
        st.c[decltype(Bi)::value][decltype(Jc)::value][decltype(Kc)::value] = 0.0f;
      });
    });
    rola::static_for<4>([&](auto Kc) { st.m[decltype(Bi)::value][decltype(Kc)::value] = 0.0f; });
  });
  if (p.state_in != nullptr)
    state_sweep<BP, D, BC, true>(sm, lay[rola::facts::kWrite], st, slots, act_read, act_write, warp,
                                 lane, p.state_in, nullptr);
  pc.lap(kPhaseSweep);

  //: THE STREAMS' VARIANTS, chosen once by the sides' layouts among those the arm can have.
  const auto run = [&](auto RC, auto RS, auto WC, auto WS) {
    window_loop<BP, D, BC, decltype(RC)::value, decltype(RS)::value, decltype(WC)::value,
                decltype(WS)::value>(p, sm, lay, st, owner, num, den, tid, warp, lane, bh, pc);
  };
  using T = std::true_type;
  using F = std::false_type;
  if constexpr (!BP::kMayCompose) {
    run(F{}, F{}, F{}, F{});
  } else {
    const SideLayout& lr = lay[rola::facts::kRead];
    const SideLayout& lw = lay[rola::facts::kWrite];
    const bool rc = lr.single_inner < 0 || lr.single_outer < 0, rs = lr.straddle >= 0;
    const bool wc = lw.single_inner < 0 || lw.single_outer < 0, ws = lw.straddle >= 0;
    if (!rc && !wc)
      run(F{}, F{}, F{}, F{});
    else if constexpr (!BP::kMayStraddle) {
      if (rc && wc)
        run(T{}, F{}, T{}, F{});
      else if (rc)
        run(T{}, F{}, F{}, F{});
      else
        run(F{}, F{}, T{}, F{});
    } else {
      if (rs && ws)
        run(T{}, T{}, T{}, T{});
      else if (rs)
        run(T{}, T{}, wc ? T{} : F{}, F{});
      else if (ws)
        run(wc ? T{} : F{}, F{}, T{}, T{});
      else
        run(rc ? T{} : F{}, F{}, wc ? T{} : F{}, F{});
    }
  }

  if (p.state_out != nullptr) {
    snapshot_publish<BP, D, BC>(sm, st, warp, lane);
    ops::rendezvous();
    state_sweep<BP, D, BC, false>(sm, lay[rola::facts::kWrite], st, slots, act_read, act_write,
                                  warp, lane, nullptr, p.state_out);
  }
  pc.lap(kPhaseSweep);
  if (p.ledger != nullptr && lane == 0) {
    long long* const row =
        p.ledger + ((long)(bh * gridDim.x + owner) * BP::kWarps + warp) * kPhases;
#pragma unroll
    for (int ph = 0; ph < kPhases; ++ph)
      atomicAdd(reinterpret_cast<unsigned long long*>(row + ph), (unsigned long long)pc.acc[ph]);
  }
}

}  // namespace rola::carry
