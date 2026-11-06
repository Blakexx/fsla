// csrc/rola/src/carry/readout.cuh -- THE READOUT, the value carve over a listed read side: a warp its token group's
// slots and its channel group's 8 channels over every digit of the box. see docs/internals/carry/readout.md
#pragma once

#include "carry/handoff.cuh"
#include "carry/panel.cuh"

namespace rola::carry {

//: A WARP'S PLACE IN THE READOUT: its token group (a chunk's half of the slots, `group_split`) and its channel group
//: (8 channels). -- see docs/internals/carry/readout.md#warps
constexpr int kChannelGroups = kBoxChannels / 8, kTokenGroups = kWarps / kChannelGroups;
static_assert(
    kChannelGroups * kTokenGroups == kWarps && kTokenGroups == 2 && kDenWords == kChannelGroups,
    "the 32-channel box: two token groups of four channel groups, a den word a channel group");

//: AN f16 PAIR ADDED INTO A WORD, round to nearest.
__device__ __forceinline__ uint32_t add_h2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("add.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

//: A TILE'S TWO MMAs over the 32 lists into `acc`, accumulating in fp16 (`F16`) or fp32.
template <bool F16>
__device__ __forceinline__ void tile_mma(float (&acc)[4], const uint32_t (&a0)[4],
                                         const uint32_t (&a1)[4], const uint32_t (&b)[4]) {
  if constexpr (F16) {
    uint32_t h[2] = {0u, 0u};
    ops::hmma_f16(h, a0, b[0], b[1]);
    ops::hmma_f16(h, a1, b[2], b[3]);
    const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&h[0])),
                 hi = __half22float2(*reinterpret_cast<const __half2*>(&h[1]));
    acc[0] = lo.x;
    acc[1] = lo.y;
    acc[2] = hi.x;
    acc[3] = hi.y;
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] = 0.0f;
    ops::hmma(acc, a0, b[0], b[1]);
    ops::hmma(acc, a1, b[2], b[3]);
  }
}

//: A TILE ROW'S chunk slot (its row offset back by the chunk's first slot, modulo 2^16) and coefficient.
template <class P>
__device__ __forceinline__ uint32_t slot_of(uint32_t ex, int first) {
  constexpr uint32_t kChunks = P::kReadRow / 16;
  return (((ex & 0xffffu) - static_cast<uint32_t>(first) * kChunks) & 0xffffu) / kChunks;
}

__device__ __forceinline__ float coef_of(uint32_t ey) {
  return __half2float(__ushort_as_half(static_cast<unsigned short>(ey & 0xffffu)));
}

//: A TOKEN'S SCALE `2^(-scale - 1)` from its byte, its exponent field built directly (no branch in the step).
__device__ __forceinline__ float token_scale(int sb) {
  return __uint_as_float(static_cast<uint32_t>(min(max(-sb - 1, -126), 127) + 127) << 23);
}

//: A READ CHUNK'S ENTRY FACTORS, once a chunk before its readout (between its copies' landing and the rendezvous after
//: it): each entry of the warp's own digits (`warp + 8 j`, the runs its fill copied) rewritten in place to its row
//: offset and its coefficient (f16, the high half) and its num factor, fp32 -- `c 2^(e_d - kImageTop - kCoefTop)`, on
//: the f16 panel times its token's scale. -- see docs/internals/carry/readout.md#factors
template <class P>
__device__ __forceinline__ void read_factors(const CarryParams& p, const uint8_t* smem,
                                             const Run& run, int w, ops::SmemAddr ents, int warp,
                                             int lane) {
  __syncwarp();
  const ops::SmemAddr scale = ops::smem_addr(smem + p.tables.scale_at)
                              + static_cast<uint32_t>((w & 1) * p.tables.heads * kWindow);
  const ops::SmemAddr exps = exps_of<P>(smem, warp);
  const bool f32 = p.reduce == kReduceF32;
  static_for<kBoxDigits / kWarps>([&](auto J) {
    const int d = warp + kWarps * decltype(J)::value;
    const int n = __shfl_sync(0xffffffffu, run.n, d), to = __shfl_sync(0xffffffffu, run.to, d);
    const float dpow = pow2_wide(
        static_cast<int>(ops::load_shared_u32(exps + 4u * static_cast<uint32_t>(exp_slot(d))))
        - kImageTop - kCoefTop);
#pragma unroll 1
    for (int k = lane; k < n; k += 32) {
      const ops::SmemAddr at = ents + 8u * static_cast<uint32_t>(to + k);
      const uint2 e = ops::load_shared_u64(at);
      const float c = coef_of(e.y);
      const float ts = token_scale(static_cast<int8_t>(ops::load_shared_u8(scale + (e.x >> 16))));
      const float f = f32 ? c * dpow : c * dpow * ts;
      ops::store_shared_u64(at, (e.x & 0xffffu) | (e.y & 0xffffu) << 16, __float_as_uint(f));
    }
  });
}

//: THE LOWEST SET BIT's index (`x` nonzero), one `brev` and `bfind`, volatile so it is never moved into a branch.
__device__ __forceinline__ int lowest_bit(uint32_t x) {
  int i;
  asm volatile("{\n.reg .b32 r;\nbrev.b32 r, %1;\nbfind.shiftamt.u32 %0, r;\n}\n"
               : "=r"(i)
               : "r"(x));
  return i;
}

//: A TILE: a digit's 16 entries from `s` in a token group's run, its operands -- lane `l & 15`'s entry, the rows by
//: two `ldmatrix.x4`, the digit's slice (`Den`: the den image's block and the digit's den factor) -- loaded a tile ahead
//: of its MMAs. -- see docs/internals/carry/readout.md#chunk
struct Tile {
  uint32_t a0[4], a1[4], b[4];
  uint2 e;
  float k;
  int d, s, start, count;
};

//: THE PASS'S PLACE: the warp's channel group and lanes, its token group's run a lane a digit, the buffers.
struct ReadLanes {
  ops::SmemAddr pool_u, ents, image, den_image, den_fac, panel, dens;
  uint32_t mask, word, chan;
  int first, cnt, at, cg, lane, g, tq;
};

template <class P, bool Den>
__device__ __forceinline__ void tile_in(Tile& t, const ReadLanes& r, int d, int s) {
  t.d = d;
  t.s = s;
  t.count = __shfl_sync(0xffffffffu, r.cnt, d);
  t.start = __shfl_sync(0xffffffffu, r.at, d);
  const int mine = s + (r.lane & 15);
  t.e = ops::load_shared_u64(r.ents
                             + 8u * static_cast<uint32_t>(t.start + (mine < t.count ? mine : s)));
  const ops::SmemAddr row0 = r.pool_u + 16u * (t.e.x & 0xffffu);
  ops::load_frag(t.a0, chunk_addr<P::kReadRow>(row0, static_cast<uint32_t>(r.lane >> 4)));
  ops::load_frag(t.a1, chunk_addr<P::kReadRow>(row0, 2u + static_cast<uint32_t>(r.lane >> 4)));
  if constexpr (Den) {
    ops::load_frag_t(t.b, r.den_image + den_row(r.lane, d));
    t.k = ops::load_shared_f32(r.den_fac + 4u * static_cast<uint32_t>(d));
  } else {
    ops::load_frag_t(t.b, r.image + (static_cast<uint32_t>(d) & r.mask) * kPlaneBytes
                              + slice_row(r.cg, r.lane, d));
  }
}

//: A TILE'S MMAs and each of the lane's two tokens' add: the num pair at its factor into the panel, or (`Den`) the
//: digit's den column at `c` times its den factor into the slot's den word, by the lanes holding the column.
template <class P, bool F32, bool F16, bool Den>
__device__ __forceinline__ void tile_out(const Tile& t, const ReadLanes& r, bool valid) {
  float acc[4];
  tile_mma<F16>(acc, t.a0, t.a1, t.b);
  //: the lane's two tokens (rows `g`, `g + 8`): their entries, slots and predicates first, then both loads, both adds
  //: and both stores, so the two read-modify-writes overlap (each a distinct slot)
  uint32_t ex[2], ey[2], slot[2];
  bool live[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    ex[h] = __shfl_sync(0xffffffffu, t.e.x, r.g + 8 * h);
    ey[h] = __shfl_sync(0xffffffffu, t.e.y, r.g + 8 * h);
    live[h] = valid && t.s + r.g + 8 * h < t.count;
    slot[h] = slot_of<P>(ex[h], r.first);
  }

  if constexpr (Den) {
    //: the den column the digit's (`d & 1` of the lane's pair: a select, never a runtime index into `acc`), kept by the
    //: lanes holding it
    ops::SmemAddr at[2];
    float old[2];
    bool mine[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      mine[h] = live[h] && r.tq == ((t.d & 7) >> 1);
      at[h] = r.dens + kDenRowBytes * slot[h] + 4u * static_cast<uint32_t>(r.cg);
      old[h] = ops::load_shared_f32_if(at[h], mine[h]);
    }

#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float dn = (t.d & 1) ? acc[2 * h + 1] : acc[2 * h];
      ops::store_shared_f32_if(at[h], old[h] + dn * (coef_of(ex[h] >> 16) * t.k), mine[h]);
    }
  } else if constexpr (F32) {
    ops::SmemAddr at[2];
    uint2 old[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      at[h] = r.panel + panel_word<true>(slot[h], r.chan);
      old[h] = ops::load_shared_u64_if(at[h], live[h]);
    }

#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float f = __uint_as_float(ey[h]);
      ops::store_shared_f32x2_if(at[h], __uint_as_float(old[h].x) + acc[2 * h] * f,
                                 __uint_as_float(old[h].y) + acc[2 * h + 1] * f, live[h]);
    }
  } else {
    ops::SmemAddr at[2];
    uint32_t old[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      at[h] = r.panel + panel_word<false>(slot[h], r.word);
      old[h] = ops::load_shared_u32_if(at[h], live[h]);
    }

#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float f = __uint_as_float(ey[h]);
      const __half2 v = __floats2half2_rn(acc[2 * h] * f, acc[2 * h + 1] * f);
      ops::store_shared_u32_if(at[h], add_h2(old[h], *reinterpret_cast<const uint32_t*>(&v)),
                               live[h]);
    }
  }
}

//: A PASS over the tiles of the digits `live` names, in digit order, each tile's operands loaded while the one before it
//: computes: two tiles a trip (no operand moves between registers), the second's adds predicated off where it is past
//: the pass's last, the trip's one exit at its end (no branch among its MMAs, §20); a warp sync a tile orders one
//: tile's adds before the next's, whose tokens may be the same in other lanes. -- see docs/internals/carry/readout.md#chunk
template <class P, bool F32, bool F16, bool Den>
__device__ __forceinline__ void read_pass(const ReadLanes& r, uint32_t live) {
  if (live == 0u) return;
  //: the tile after `t`: its digit's next 16, else the next digit `live` names (its lowest bit above `t`'s, found on
  //: every tile so no branch forms around it); whether there is one
  const auto next = [&](const Tile& t, Tile& n) {
    const bool more = t.s + 16 < t.count;
    const uint32_t above = live & ~((2u << t.d) - 1u);
    const int nd = lowest_bit(above);
    tile_in<P, Den>(n, r, more || above == 0u ? t.d : nd, more ? t.s + 16 : 0);
    return more || above != 0u;
  };
  Tile a, b;
  tile_in<P, Den>(a, r, __ffs(live) - 1, 0);
  bool go = true;
#pragma unroll 1
  while (go) {
    const bool go_b = next(a, b);
    tile_out<P, F32, F16, Den>(a, r, true);
    __syncwarp();
    const bool go_a = next(b, a);
    tile_out<P, F32, F16, Den>(b, r, go_b);
    __syncwarp();
    go = go_b && go_a;
  }
}

//: THE READOUT of a chunk's slots over digits `lo .. hi - 1` of the image (plane `d & mask`) into the num panel and the
//: den words, reduce `F32`, accumulate `F16`: the num's pass over every digit with entries in the warp's token group,
//: then the den's over the digits its channel group names (`d mod 4`). -- see docs/internals/carry/readout.md#chunk
template <class P, bool F32, bool F16>
__device__ __forceinline__ void read_chunk(const uint8_t* smem, const Run& run, int first, int lo,
                                           int hi, ops::SmemAddr image, uint32_t mask,
                                           ops::SmemAddr rows, ops::SmemAddr panel,
                                           ops::SmemAddr dens, ops::SmemAddr ents, int warp,
                                           int lane) {
  static_assert(P::kReadFactors == 1, "the value-carve readout of a listed read side");
  const int tg = warp / kChannelGroups, cg = warp % kChannelGroups;
  const ReadLanes r{rows - static_cast<uint32_t>(first * P::kReadRow),
                    ents,
                    image,
                    ops::smem_addr(smem + P::kDenImage),
                    ops::smem_addr(smem + P::kDenFactor),
                    panel,
                    dens,
                    mask,
                    static_cast<uint32_t>(4 * cg + (lane & 3)),
                    static_cast<uint32_t>(8 * cg + 2 * (lane & 3)),
                    first,
                    tg ? run.n - run.n0 : run.n0,
                    run.to + (tg ? run.n0 : 0),
                    cg,
                    lane,
                    lane >> 2,
                    lane & 3};
  const uint32_t live = __ballot_sync(0xffffffffu, r.cnt > 0 && lane >= lo && lane < hi);
  read_pass<P, F32, F16, false>(r, live);
  read_pass<P, F32, F16, true>(r, live & (0x11111111u << cg));
}

}  // namespace rola::carry
