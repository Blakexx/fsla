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

__device__ __forceinline__ float coef_of(uint32_t ey) {
  return __half2float(__ushort_as_half(static_cast<unsigned short>(ey & 0xffffu)));
}

//: A TOKEN'S SCALE `2^(-scale - 1)` from its byte, its exponent field built directly (no branch in the step).
__device__ __forceinline__ float token_scale(int sb) {
  return __uint_as_float(static_cast<uint32_t>(min(max(-sb - 1, -126), 127) + 127) << 23);
}

//: A READ CHUNK'S ENTRY FACTORS, once a chunk before its readout (between its copies' landing and the rendezvous after
//: it): each entry of the warp's own digits (`warp + 8 j`, the runs its fill copied) rewritten in place to its row
//: key in the chunk (the 64-byte row's chunk 0 in 16-byte units) and its coefficient (f16, the high half) and its num
//: factor, fp32 -- `c 2^(e_d - kImageTop - kCoefTop)`, on
//: the f16 panel times its token's scale. -- see docs/internals/carry/readout.md#factors
template <class P>
__device__ __forceinline__ void read_factors(const CarryParams& p, const uint8_t* smem,
                                             const Run& run, int w, int first, ops::SmemAddr ents,
                                             int warp, int lane) {
  constexpr int kMine = kBoxDigits / kWarps;
  static_assert(P::kReadRow == 64,
                "a 64-byte pool row: its key the f16 panel's rotation (`panel_word`)");
  const uint32_t back = 4u * static_cast<uint32_t>(first);
  __syncwarp();
  const ops::SmemAddr scale = ops::smem_addr(smem + p.tables.scale_at)
                              + static_cast<uint32_t>((w & 1) * p.tables.heads * kWindow);
  const ops::SmemAddr exps = exps_of<P>(smem, warp);
  const bool f32 = p.reduce == kReduceF32;

  //: the warp's digits' runs and exponents, then their entries 32 a trip, the four digits' loads, products and stores
  //: each issued together so their chains overlap
  int n[kMine], to[kMine], most = 0;
  float dpow[kMine];
  static_for<kMine>([&](auto J) {
    const int d = warp + kWarps * decltype(J)::value;
    n[J] = __shfl_sync(0xffffffffu, run.n, d);
    to[J] = __shfl_sync(0xffffffffu, run.to, d);
    most = max(most, n[J]);
    dpow[J] = pow2_wide(
        static_cast<int>(ops::load_shared_u32(exps + 4u * static_cast<uint32_t>(exp_slot(d))))
        - kImageTop - kCoefTop);
  });
#pragma unroll 1
  for (int k = lane; k - lane < most; k += 32) {
    ops::SmemAddr at[kMine];
    uint2 e[kMine];
    bool live[kMine];
    static_for<kMine>([&](auto J) {
      live[J] = k < n[J];
      at[J] = ents + 8u * static_cast<uint32_t>(to[J] + k);
      e[J] = ops::load_shared_u64_if(at[J], live[J]);
    });
    static_for<kMine>([&](auto J) {
      //: a dead lane's place is 0, a byte of the table: read, never used
      const float c = coef_of(e[J].y);
      const float ts =
          token_scale(static_cast<int8_t>(ops::load_shared_u8(scale + (e[J].x >> 16))));
      const float f = f32 ? c * dpow[J] : c * dpow[J] * ts;
      ops::store_shared_u64_if(at[J], ((e[J].x - back) & 0xffffu) | (e[J].y & 0xffffu) << 16,
                               __float_as_uint(f), live[J]);
    });
  }
}

//: THE LOWEST SET BIT's index (`x` nonzero), one `brev` and `bfind`, volatile so it is never moved into a branch.
__device__ __forceinline__ int lowest_bit(uint32_t x) {
  int i;
  asm volatile("{\n.reg .b32 r;\nbrev.b32 r, %1;\nbfind.shiftamt.u32 %0, r;\n}\n"
               : "=r"(i)
               : "r"(x));
  return i;
}

//: A TILE: the first 16 entries of what remains of a digit's run in a token group (`cs`: its place, its count << 16),
//: its operands -- lane `l & 15`'s entry, the rows by two `ldmatrix.x4`, the digit's slice (`Den`: and the den image's
//: block and the digit's den factor) -- loaded a tile ahead of its MMAs. -- see docs/internals/carry/readout.md#chunk
struct Tile {
  uint32_t a0[4], a1[4], b[4], bd[4];
  uint2 e, t0, t1;
  uint32_t cs;
  float k;
  int d;
};

//: THE PASS'S PLACE: the warp's channel group and lanes, its token group's run a lane a digit (`cs`: its place, its
//: count << 16), the buffers.
struct ReadLanes {
  ops::SmemAddr rows, ents, image, den_image, den_fac, panel, dens;
  uint32_t mask, word4, chan4, cs;
  int cg, lane, g, tq;
};

template <class P, bool Den>
__device__ __forceinline__ void tile_in(Tile& t, const ReadLanes& r, int d, uint32_t cs) {
  t.d = d;
  t.cs = cs;
  const int mine = r.lane & 15;
  t.e = ops::load_shared_u64(
      r.ents + 8u * ((cs & 0xffffu) + (mine < static_cast<int>(cs >> 16) ? mine : 0)));
  const ops::SmemAddr row0 = r.rows + 16u * (t.e.x & 0xffffu);
  ops::load_frag(t.a0, chunk_addr<P::kReadRow>(row0, static_cast<uint32_t>(r.lane >> 4)));
  ops::load_frag(t.a1, chunk_addr<P::kReadRow>(row0, 2u + static_cast<uint32_t>(r.lane >> 4)));
  ops::load_frag_t(t.b, r.image + (static_cast<uint32_t>(d) & r.mask) * kPlaneBytes
                            + slice_row(r.cg, r.lane, d));
  if constexpr (Den) {
    ops::load_frag_t(t.bd, r.den_image + den_row(r.lane, d));
    t.k = ops::load_shared_f32(r.den_fac + 4u * static_cast<uint32_t>(d));
  }
}

//: A TILE'S MMAs and each of the lane's two tokens' adds: the num pair at its factor into the panel, and (`Den`, a
//: digit the warp's channel group names) the digit's den column at `c` times its den factor into the slot's den word,
//: by the lanes holding the column.
//: A TILE'S TOKENS: the lane's two (rows `g`, `g + 8`) entries, loaded before the next tile's operands and read after
//: this tile's MMAs (not shuffled from the row lanes; not a tile ahead, which holds them across the other tile's adds)
__device__ __forceinline__ void tile_tokens(Tile& t, const ReadLanes& r) {
  const int count = static_cast<int>(t.cs >> 16);
  const uint32_t start = t.cs & 0xffffu;
  t.t0 = ops::load_shared_u64(r.ents + 8u * (start + (r.g < count ? r.g : 0)));
  t.t1 = ops::load_shared_u64(r.ents + 8u * (start + (r.g + 8 < count ? r.g + 8 : 0)));
}

template <class P, bool F32, bool F16, bool Den>
__device__ __forceinline__ void tile_out(const Tile& t, const ReadLanes& r, bool valid) {
  float acc[4], dacc[4];
  tile_mma<F16>(acc, t.a0, t.a1, t.b);
  if constexpr (Den) tile_mma<F16>(dacc, t.a0, t.a1, t.bd);
  //: the lane's two tokens (rows `g`, `g + 8`): their row keys and predicates, then every load, then every add and
  //: store, so the read-modify-writes overlap (each a distinct slot and word)
  uint32_t key[2];
  float f[2];
  bool live[2], mine[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint2 e = h ? t.t1 : t.t0;
    key[h] = e.x & 0xffffu;
    f[h] = __uint_as_float(e.y);
    live[h] = valid && r.g + 8 * h < static_cast<int>(t.cs >> 16);
    //: the den column the digit's (`d & 1` of the lane's pair), kept by the lanes holding it
    mine[h] = Den && live[h] && r.tq == ((t.d & 7) >> 1);
  }

  ops::SmemAddr at[2], dat[2];
  uint2 old64[2];
  uint32_t old32[2];
  float dold[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    if constexpr (F32) {
      const uint32_t sl = key[h] >> 2;
      at[h] = r.panel + ((128u * sl + 16u * (sl & 7u)) ^ r.chan4);
      old64[h] = ops::load_shared_u64_if(at[h], live[h]);
    } else {
      at[h] = r.panel + ((key[h] << 4) ^ r.word4);
      old32[h] = ops::load_shared_u32_if(at[h], live[h]);
    }
    if constexpr (Den) {
      dat[h] = r.dens + ((key[h] << 2) & ~15u) + 4u * static_cast<uint32_t>(r.cg);
      dold[h] = ops::load_shared_f32_if(dat[h], mine[h]);
    }
  }

#pragma unroll
  for (int h = 0; h < 2; ++h) {
    if constexpr (F32) {
      ops::store_shared_f32x2_if(at[h], __uint_as_float(old64[h].x) + acc[2 * h] * f[h],
                                 __uint_as_float(old64[h].y) + acc[2 * h + 1] * f[h], live[h]);
    } else {
      const __half2 v = __floats2half2_rn(acc[2 * h] * f[h], acc[2 * h + 1] * f[h]);
      ops::store_shared_u32_if(at[h], add_h2(old32[h], *reinterpret_cast<const uint32_t*>(&v)),
                               live[h]);
    }
    if constexpr (Den) {
      const uint32_t ex = h ? t.t1.x : t.t0.x;
      const float dn = (t.d & 1) ? dacc[2 * h + 1] : dacc[2 * h];
      ops::store_shared_f32_if(dat[h], dold[h] + dn * (coef_of(ex >> 16) * t.k), mine[h]);
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
  //: lane `d`'s next digit `live` names above `d` (32 where none) and that digit's run, once a pass
  const uint32_t above_l = live & ~((2u << r.lane) - 1u);
  const int next_l = above_l != 0u ? lowest_bit(above_l) : 32;
  const uint32_t next_cs = __shfl_sync(0xffffffffu, r.cs, next_l & 31);
  //: the tile after `t`: its digit's next 16 (the run advanced by 16), else its digit's next and that one's run (two
  //: shuffles side by side); whether there is one
  const auto next = [&](const Tile& t, Tile& n) {
    const bool more = 16 < static_cast<int>(t.cs >> 16);
    const int nd = __shfl_sync(0xffffffffu, next_l, t.d);
    const uint32_t ncs = __shfl_sync(0xffffffffu, next_cs, t.d);
    const bool stay = more || nd == 32;
    tile_in<P, Den>(n, r, stay ? t.d : nd, more ? t.cs + 16u - (16u << 16) : stay ? t.cs : ncs);
    return more || nd < 32;
  };
  Tile a, b;
  const int d0 = lowest_bit(live);
  tile_in<P, Den>(a, r, d0, __shfl_sync(0xffffffffu, r.cs, d0));
  bool go = true;
#pragma unroll 1
  while (go) {
    tile_tokens(a, r);
    const bool go_b = next(a, b);
    tile_out<P, F32, F16, Den>(a, r, true);
    __syncwarp();
    tile_tokens(b, r);
    const bool go_a = next(b, a);
    tile_out<P, F32, F16, Den>(b, r, go_b);
    __syncwarp();
    //: warp-uniform by construction, shuffled from lane 0 so ptxas sees it so (no divergence checks on the shuffles)
    go = __shfl_sync(0xffffffffu, go_b && go_a, 0);
  }
}

//: THE READOUT of a chunk's slots over digits `lo .. hi - 1` of the image (plane `d & mask`) into the num panel and the
//: den words, reduce `F32`, accumulate `F16`: the num's pass over every digit with entries in the warp's token group,
//: then the den's over the digits its channel group names (`d mod 4`). -- see docs/internals/carry/readout.md#chunk
template <class P, bool F32, bool F16>
__device__ __forceinline__ void read_chunk(const uint8_t* smem, const Run& run, int lo, int hi,
                                           ops::SmemAddr image, uint32_t mask, ops::SmemAddr rows,
                                           ops::SmemAddr panel, ops::SmemAddr dens,
                                           ops::SmemAddr ents, int warp, int lane) {
  static_assert(P::kReadFactors == 1, "the value-carve readout of a listed read side");
  const int tg = warp / kChannelGroups, cg = warp % kChannelGroups, tq = lane & 3;
  const int count = tg ? run.n - run.n0 : run.n0, start = run.to + (tg ? run.n0 : 0);
  const ReadLanes r{rows,
                    ents,
                    image,
                    ops::smem_addr(smem + P::kDenImage),
                    ops::smem_addr(smem + P::kDenFactor),
                    panel,
                    dens,
                    mask,
                    static_cast<uint32_t>(4 * (4 * cg + tq)),
                    static_cast<uint32_t>(4 * (8 * cg + 2 * tq)),
                    static_cast<uint32_t>(start) | static_cast<uint32_t>(count) << 16,
                    cg,
                    lane,
                    lane >> 2,
                    tq};
  const uint32_t live = __ballot_sync(0xffffffffu, count > 0 && lane >= lo && lane < hi);
  //: the digits the warp's channel group names (`d mod 4`) with their den, then the others: each digit once
  const uint32_t own = live & (0x11111111u << cg);
  read_pass<P, F32, F16, true>(r, own);
  read_pass<P, F32, F16, false>(r, live & ~own);
}

}  // namespace rola::carry
