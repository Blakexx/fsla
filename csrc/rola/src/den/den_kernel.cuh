// csrc/rola/src/den/den_kernel.cuh -- THE DEN KERNEL: one CTA a tile of 64 level-S rows x 64 level-R digits of a write
// head, window by window: the mass of the windows before published from its fp32 fragments, each den warp's band of
// 4 rows against the window's read entries by f16 MMAs over rows staged a tile ahead, each token's den and num
// accumulated in shared memory and drained once a window, then the window's writes folded into the fragments; a cell
// whose read level S and write level R list takes the list form, a cell dense on both sides the dense form.
// see docs/internals/den/den_kernel.md
#pragma once

#include <cuda_fp16.h>

#include <cstdint>

#include "common/ops.cuh"
#include "common/static_for.cuh"
#include "den/den.cuh"

namespace rola::den {

namespace ops = ::denseref::ops;

//: THE MASS'S ADDRESS in the region: channel `ch`, row `a`, 16-byte chunk `c` of the row's 64 digits, the chunk XORed
//: by the row and channel so a quarter-warp's chunks of two rows x two channels meet distinct banks.
//: -- see docs/internals/den/den_kernel.md#region
__device__ __forceinline__ uint32_t mass_at(ops::SmemAddr region, int ch, int a, int c) {
  return region
         + static_cast<uint32_t>(ch * (kRows * kSpan * 4) + a * (kSpan * 4)
                                 + 16 * (c ^ ((a + 4 * ch) & 7)));
}

//: A SLAB ROW'S CHUNK: row `r` of a slab of 16 f16 rows (128 bytes each), its 16-byte chunk `c` XORed with the row's
//: residue, so the eight rows an `ldmatrix` matrix addresses meet distinct banks. -- see docs/internals/den/den_kernel.md#slabs
__device__ __forceinline__ uint32_t slab_at(ops::SmemAddr slab, int r, int c) {
  return slab + static_cast<uint32_t>(128 * r + 16 * (c ^ (r & 7)));
}

//: 2^e over fp32's normals, `e` held to [-126, 127].
__device__ __forceinline__ float pow2f(int e) {
  return __int_as_float((127 + min(max(e, -126), 127)) << 23);
}

//: a list's exponent as the facts pass's entries use it: at least `kUnitMin`.
__device__ __forceinline__ int list_e(int16_t k) { return max(static_cast<int>(k), kUnitMin); }

//: A WRITE HEAD'S PLANE ROW: head `h`'s token `t` (the call's) at digit `d0`.
__device__ __forceinline__ const uint8_t* plane_row(const Plane& pl, int b, int h, int t, int d0) {
  return pl.base + static_cast<int64_t>(b) * pl.sb + static_cast<int64_t>(t) * pl.st
         + static_cast<int64_t>(h) * pl.sh + 2 * d0;
}

//: A LANE'S TWO MASS COLUMNS of the fragments `acc[mt]` (rows `16 mt + g` and `+ 8`, columns `2 tq` and `2 tq + 1`):
//: channel `tq >> 1`, digits `4 warp + 2 (tq & 1)` and the next. -- see docs/internals/den/den_kernel.md#fragments
struct Columns {
  int ch, d0;
};

__device__ __forceinline__ Columns columns_of(int warp, int lane) {
  const int tq = lane & 3;
  return Columns{tq >> 1, kDigits * warp + 2 * (tq & 1)};
}

//: THE PUBLISH: the fragments at the windows before's scales (`f0`, `f1` the lane's two columns' powers of two) into the
//: region as fp32, one 8-byte store a row. -- see docs/internals/den/den_kernel.md#publish
__device__ __forceinline__ void publish(ops::SmemAddr region, const float (&acc)[4][4], float f0,
                                        float f1, int warp, int lane) {
  const Columns col = columns_of(warp, lane);
  const int g = lane >> 2, within = 8 * ((lane & 3) & 1);
  static_for<4>([&](auto M) {
    static_for<2>([&](auto H) {
      const uint32_t at = mass_at(region, col.ch, 16 * M + g + 8 * H, warp) + within;
      const float lo = acc[M][2 * H] * f0, hi = acc[M][2 * H + 1] * f1;
      asm volatile("st.shared.v2.f32 [%0], {%1, %2};" ::"r"(at), "f"(lo), "f"(hi) : "memory");
    });
  });
}

//: THE RESCALE of the fragments to the window's scales where a column's power of two moved (one trip on a vote).
__device__ __forceinline__ void rescale(float (&acc)[4][4], float f0, float f1) {
  for (int k = __any_sync(0xffffffffu, f0 != 1.0f || f1 != 1.0f) ? 1 : 0; k > 0; --k)
    static_for<4>([&](auto M) {
      acc[M][0] *= f0;
      acc[M][1] *= f1;
      acc[M][2] *= f0;
      acc[M][3] *= f1;
    });
}

//: THE DEN'S B: the band's four rows x two channels as the MMA's eight columns (lane `g`'s column: channel `g >> 2`,
//: row `4 band + (g & 3)`), the row's 64 digits as f16 at the power of two putting its largest under 2^15, four
//: k-steps of two registers (digits `16 kk + 2 tq ..` and `+ 8 ..`); `inv` the column's 2^-e.
//: -- see docs/internals/den/den_kernel.md#den-b
struct DenB {
  uint32_t b[kSteps][2];
  float inv;
};

__device__ __forceinline__ DenB den_b(ops::SmemAddr region, int band, int lane) {
  const int g = lane >> 2, tq = lane & 3;
  const int ch = g >> 2, a = kBand * band + (g & 3);
  float2 v[kSteps][2];
  float top = 0.0f;
  static_for<kSteps>([&](auto K) {
    static_for<2>([&](auto H) {
      asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];"
                   : "=f"(v[K][H].x), "=f"(v[K][H].y)
                   : "r"(mass_at(region, ch, a, 4 * K + 2 * H + (tq >> 1)) + 8 * (tq & 1)));
      top = fmaxf(top, fmaxf(fabsf(v[K][H].x), fabsf(v[K][H].y)));
    });
  });
  top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 1));
  top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 2));
  const int e = (__float_as_int(top) >> 23) - 127;
  const float s = top > 0.0f ? pow2f(14 - e) : 0.0f;
  DenB out;
  out.inv = top > 0.0f ? pow2f(e - 14) : 0.0f;
  static_for<kSteps>([&](auto K) {
    static_for<2>([&](auto H) { out.b[K][H] = ops::pack_half2(v[K][H].x * s, v[K][H].y * s); });
  });
  return out;
}

//: A SLAB'S A FRAGMENTS, row-major (an entry a row, the digits K): four k-steps by `ldmatrix`, lane `l` addressing row
//: `(l & 7) + 8 ((l >> 3) & 1)` at chunk `2 kk + (l >> 4)`, into a fresh fragment then accumulating.
//: -- see docs/internals/den/den_kernel.md#den-tile
//: @burst
__device__ __forceinline__ void den_mma(float (&c)[4], ops::SmemAddr slab, const DenB& bf,
                                        int lane) {
  const int r = (lane & 7) + 8 * ((lane >> 3) & 1), hi = lane >> 4;
  static_for<kSteps>([&](auto K) {
    uint32_t a[4];
    ops::load_frag(a, slab_at(slab, r, 2 * K + hi));
    if constexpr (K == 0) {
      ops::hmma_fresh(c, a, bf.b[K][0], bf.b[K][1]);
    } else {
      ops::hmma(c, a, bf.b[K][0], bf.b[K][1]);
    }
  });
}

//: A SLAB'S A FRAGMENTS, k-major (a token's coefficient row a slab row, the rows M): m-tile `mt` by `ldmatrix.trans`,
//: lane `l` addressing slab row `8 (l >> 4) + (l & 7)` at chunk `2 mt + ((l >> 3) & 1)`; four fp32-accumulate MMAs
//: into the fragments. -- see docs/internals/den/den_kernel.md#writes
//: @burst
__device__ __forceinline__ void write_mma(float (&acc)[4][4], ops::SmemAddr slab, uint32_t b0,
                                          uint32_t b1, int lane) {
  const int r = 8 * (lane >> 4) + (lane & 7), half = (lane >> 3) & 1;
  static_for<4>([&](auto M) {
    uint32_t a[4];
    ops::load_frag_t(a, slab_at(slab, r, 2 * M + half));
    ops::hmma(acc[M], a, b0, b1);
  });
}

//: A SLAB STAGED: lane `l` copies four 16-byte chunks (`4 (l & 1) ..`) of row `l >> 1` from `src`, the row's plane row
//: at the tile's first digit. -- see docs/internals/den/den_kernel.md#slabs
__device__ __forceinline__ void stage_slab(ops::SmemAddr slab, const uint8_t* src, int lane) {
  const int r = lane >> 1, c0 = 4 * (lane & 1);
  static_for<4>([&](auto C) { ops::stage_tile(slab_at(slab, r, c0 + C), src + 16 * (c0 + C)); });
}

//: AN ADD INTO THE ACCUMULATOR: channel `ch`'s row of place `place` (a query head's token), predicated on `live`.
__device__ __forceinline__ void acc_add(ops::SmemAddr acc, int heads, int ch, int place, float v,
                                        bool live) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %2, 0;\n@p red.shared.add.f32 [%0], %1;\n}\n" ::"r"(
          acc + 4u * static_cast<uint32_t>(ch * heads * kWindow + place)),
      "f"(v), "r"(static_cast<int>(live))
      : "memory");
}

//: THE DRAIN: the accumulator's rows to `den` and `num` (`red.global.add.f32`, a warp's lanes 32 consecutive tokens of
//: one channel), each row zeroed behind its read; every thread two rows a query head.
//: -- see docs/internals/den/den_kernel.md#drain
__device__ __forceinline__ void drain(const DenParams& p, uint8_t* smem, int bh, int w, int live) {
  float* const acc = reinterpret_cast<float*>(smem + Tenants::kAcc);
  const int n = kChannels * p.heads * live;
#pragma unroll 1
  for (int i = ops::opaque(static_cast<int>(threadIdx.x)); i < n; i += kThreads) {
    const int t = i % live, jj = i / live, ch = jj / p.heads, j = jj - ch * p.heads;
    const int at = (ch * p.heads + j) * kWindow + t;
    const float v = acc[at];
    acc[at] = 0.0f;
    float* const out = (ch ? p.num : p.den) + (static_cast<int64_t>(bh) * p.heads + j) * p.L
                       + static_cast<int64_t>(w) * kWindow + t;
    asm volatile("red.global.add.f32 [%0], %1;" ::"l"(out), "f"(v) : "memory");
  }
}

//: THE LIST FORM'S DEN over a band's four lists of window `w`: their entries one stream (a lane's entry's list by three
//: compares), staged a round of `kRound` at a time into the warp's row of packed words (`place | list << 14 | coef <<
//: 16`); a tile 16 entries, its rows a slab staged one tile ahead; an entry's results its row's two columns, the lane
//: holding one adding `coef x C x 2^-e` into its token's accumulator row. -- see docs/internals/den/den_kernel.md#den-list
__device__ __forceinline__ void den_lists(const DenParams& p, uint8_t* smem, const DenB& bf, int bh,
                                          int w, int s0, int r0, int band, int lane) {
  const int g = lane >> 2, tq = lane & 3, heads = p.heads;
  const int b = bh / p.write_heads, hw = bh % p.write_heads;
  const int groups = (p.width_s + 31) / 32, cols = kListAt * heads;
  const int s = s0 + kBand * band, gs = s >> 5, l0 = s & 31;
  const uint16_t* const pre =
      p.read_lists.pre + (static_cast<int64_t>(bh) * p.windows + w) * groups * kPreRows * cols
      + static_cast<int64_t>(gs) * kPreRows * cols + (cols - 1);
  int first[kBand + 1];
  first[0] = 0;
  static_for<kBand>([&](auto I) {
    first[I + 1] =
        first[I]
        + ((static_cast<int>(pre[(l0 + I + 1) * cols]) - static_cast<int>(pre[(l0 + I) * cols]))
           & 0xffff);
  });
  const int n = first[kBand];
  const uint2* const lists =
      p.read_lists.entries
      + (static_cast<int64_t>(bh) * p.windows + w) * p.width_s * (kListCap * heads)
      + static_cast<int64_t>(s) * (kListCap * heads);
  uint32_t* const words = reinterpret_cast<uint32_t*>(smem + Tenants::kScratch) + band * kRound;
  const ops::SmemAddr slabs = ops::smem_addr(smem + Tenants::kRegion) + band * (2 * kSlabBytes);
  const ops::SmemAddr acc = ops::smem_addr(smem + Tenants::kAcc);
  const float inv_lo = __shfl_sync(0xffffffffu, bf.inv, 4 * (2 * tq)),
              inv_hi = __shfl_sync(0xffffffffu, bf.inv, 4 * (2 * tq + 1));
  const auto row_src = [&](uint32_t wd) {
    const int place = wd & 0x3fff;
    return plane_row(p.read_r, b, hw * heads + (place >> window_bits(kWindow)),
                     w * kWindow + (place & (kWindow - 1)), r0);
  };

#pragma unroll 1
  for (int base = 0; base < n; base += kRound) {
    const int count = min(kRound, n - base);
    __syncwarp();
#pragma unroll 1
    for (int k = lane; k < count; k += 32) {
      const int e = base + k;
      int list = 0, off = e;
      static_for<kBand - 1>([&](auto I) {
        const bool past = e >= first[I + 1];
        list = past ? I + 1 : list;
        off = past ? e - first[I + 1] : off;
      });
      const uint2 v = lists[static_cast<int64_t>(list) * (kListCap * heads) + off];
      words[k] = (v.x >> 16) | static_cast<uint32_t>(list) << 14 | (v.y & 0xffffu) << 16;
    }
    __syncwarp();
    const auto word_at = [&](int i) { return words[min(i, count - 1)]; };
    const int tiles = (count + kTileRows - 1) / kTileRows;
    //: tile 0's rows issued; each tile issues the next's into the other slab before its own MMAs
    stage_slab(slabs, row_src(word_at(lane >> 1)), lane);
    ops::stage_commit();
#pragma unroll 1
    for (int i = 0; i < tiles; ++i) {
      const int next = min(i + 1, tiles - 1);
      stage_slab(slabs + ((i + 1) & 1) * kSlabBytes,
                 row_src(word_at(kTileRows * next + (lane >> 1))), lane);
      ops::stage_commit();
      ops::stage_wait<1>();
      __syncwarp();
      float c[4];
      den_mma(c, slabs + (i & 1) * kSlabBytes, bf, lane);
      static_for<2>([&](auto H) {
        const int e = kTileRows * i + g + 8 * H;
        const uint32_t wd = word_at(e);
        const int row = (wd >> 14) & 3;
        const float coef = __half2float(__ushort_as_half(static_cast<uint16_t>(wd >> 16)));
        const float v = coef * ((row & 1) ? c[2 * H + 1] * inv_hi : c[2 * H] * inv_lo);
        acc_add(acc, heads, tq >> 1, wd & 0x3fff, v, e < count && (row >> 1) == (tq & 1));
      });
    }
    ops::stage_wait<0>();
  }
}

//: THE DENSE FORM'S DEN over the window's tokens of every query head: a tile 16 tokens, their level-R rows one slab a
//: warp (staged, waited, no tile ahead: the dense form's chunk buffers hold the region's other half); the band's four
//: rows summed with the token's level-S coefficients (two a lane, the other two across `xor 1`), one add a lane a
//: tile: its even lanes the tile's first token, its odd the second. -- see docs/internals/den/den_kernel.md#den-dense
__device__ __forceinline__ void den_dense(const DenParams& p, uint8_t* smem, const DenB& bf, int bh,
                                          int w, int live, int s0, int r0, int band, int lane) {
  const int g = lane >> 2, tq = lane & 3, heads = p.heads;
  const int b = bh / p.write_heads, hw = bh % p.write_heads;
  const int tiles = heads * (live / kTileRows);
  const int a0 = s0 + kBand * band + 2 * (tq & 1);
  const ops::SmemAddr slab = ops::smem_addr(smem + Tenants::kRegion) + band * kSlabBytes;
  const ops::SmemAddr acc = ops::smem_addr(smem + Tenants::kAcc);
  const float inv_lo = __shfl_sync(0xffffffffu, bf.inv, 4 * (2 * tq)),
              inv_hi = __shfl_sync(0xffffffffu, bf.inv, 4 * (2 * tq + 1));
  //: a tile's places: its head's `live` tokens then the next head's, each a whole number of tiles
  const auto place_of = [&](int e) { return (e / live) * kWindow + e % live; };
  const auto token_row = [&](int idx, const Plane& pl, int d0) {
    return plane_row(pl, b, hw * heads + (idx >> window_bits(kWindow)),
                     w * kWindow + (idx & (kWindow - 1)), d0);
  };
#pragma unroll 1
  for (int i = 0; i < tiles; ++i) {
    stage_slab(slab, token_row(place_of(kTileRows * i + (lane >> 1)), p.read_r, r0), lane);
    ops::stage_commit();
    const int ia = place_of(kTileRows * i + g), ib = place_of(kTileRows * i + g + 8);
    const uint32_t ca = __ldg(reinterpret_cast<const uint32_t*>(token_row(ia, p.read_s, a0)));
    const uint32_t cb = __ldg(reinterpret_cast<const uint32_t*>(token_row(ib, p.read_s, a0)));
    ops::stage_wait<0>();
    __syncwarp();
    float c[4];
    den_mma(c, slab, bf, lane);
    const float2 xa = __half22float2(*reinterpret_cast<const __half2*>(&ca));
    const float2 xb = __half22float2(*reinterpret_cast<const __half2*>(&cb));
    float sa = xa.x * c[0] * inv_lo + xa.y * c[1] * inv_hi;
    float sb = xb.x * c[2] * inv_lo + xb.y * c[3] * inv_hi;
    sa += __shfl_xor_sync(0xffffffffu, sa, 1);
    sb += __shfl_xor_sync(0xffffffffu, sb, 1);
    acc_add(acc, heads, tq >> 1, (tq & 1) ? ib : ia, (tq & 1) ? sb : sa, true);
    __syncwarp();
  }
}

//: THE LIST FORM'S WRITES of window `w`: the warp's four lists at level R as one stream, 64 entries a chunk into the
//: warp's slots (the token from the group's union, each channel's coefficient with its list), then k-steps of 16: the
//: entries' tokens' level-S rows staged a step ahead into the warp's two slabs, a lane's B its column's channel of
//: the two entries at its k positions where their list is its digit, else zero. -- see docs/internals/den/den_kernel.md#writes
__device__ __forceinline__ void writes_lists(const DenParams& p, uint8_t* smem, float (&acc)[4][4],
                                             int bh, int w, int s0, int r0, int warp, int lane) {
  const int g = lane >> 2, tq = lane & 3;
  const int b = bh / p.write_heads, hw = bh % p.write_heads;
  const int groups = (p.width_r + 31) / 32;
  const int r = r0 + kDigits * warp, gr = r >> 5, l0 = r & 31;
  const uint16_t* const pre =
      p.write_lists.pre + (static_cast<int64_t>(bh) * p.windows + w) * groups * kPreRows * kListAt
      + static_cast<int64_t>(gr) * kPreRows * kListAt + (kListAt - 1);
  int first[kDigits + 1];
  first[0] = 0;
  static_for<kDigits>([&](auto I) {
    first[I + 1] = first[I]
                   + ((static_cast<int>(pre[(l0 + I + 1) * kListAt])
                       - static_cast<int>(pre[(l0 + I) * kListAt]))
                      & 0xffff);
  });
  const int n = first[kDigits];
  const uint2* const lists = p.write_lists.entries
                             + (static_cast<int64_t>(bh) * p.windows + w) * p.width_r * kListCap
                             + static_cast<int64_t>(r) * kListCap;
  const uint16_t* const toks = reinterpret_cast<const uint16_t*>(smem + Tenants::kUnion)
                               + ((w & 1) * (kSpan / 32) + (gr - (r0 >> 5))) * kWindow;
  uint16_t* const slot_token =
      reinterpret_cast<uint16_t*>(smem + Tenants::kScratch) + warp * (kChunk + 16);
  uint32_t* const slot_c =
      reinterpret_cast<uint32_t*>(smem + Tenants::kScratch + Tenants::kSlotTokenBytes)
      + warp * kChannels * (kChunk + 16);
  const uint32_t* const mine = slot_c + (g >> 2) * (kChunk + 16);
  const ops::SmemAddr slabs = ops::smem_addr(smem + Tenants::kRegion) + warp * (2 * kSlabBytes);
  const auto row_src = [&](int token) {
    return plane_row(p.write_s, b, hw, w * kWindow + token, s0);
  };

#pragma unroll 1
  for (int c0 = 0; c0 < n; c0 += kChunk) {
    __syncwarp();
    static_for<2>([&](auto H) {
      const int k = 32 * H + lane, e = c0 + k;
      int list = 0, off = e;
      static_for<kDigits - 1>([&](auto I) {
        const bool past = e >= first[I + 1];
        list = past ? I + 1 : list;
        off = past ? e - first[I + 1] : off;
      });
      const bool live = e < n;
      const uint2 v =
          live ? lists[static_cast<int64_t>(list) * kListCap + off] : make_uint2(0u, 0u);
      slot_token[k] = toks[v.x & 0xffffu] & 0x7fffu;
      slot_c[k] = (v.y >> 16) | static_cast<uint32_t>(list) << 16;
      slot_c[(kChunk + 16) + k] = (v.y & 0xffffu) | static_cast<uint32_t>(list) << 16;
    });
    __syncwarp();
    const int steps = (min(kChunk, n - c0) + 15) >> 4;
    stage_slab(slabs, row_src(slot_token[lane >> 1]), lane);
    ops::stage_commit();
#pragma unroll 1
    for (int ks = 0; ks < steps; ++ks) {
      const int next = min(ks + 1, steps - 1);
      stage_slab(slabs + ((ks + 1) & 1) * kSlabBytes, row_src(slot_token[16 * next + (lane >> 1)]),
                 lane);
      ops::stage_commit();
      const uint2 lo = *reinterpret_cast<const uint2*>(&mine[16 * ks + 2 * tq]);
      const uint2 hi = *reinterpret_cast<const uint2*>(&mine[16 * ks + 2 * tq + 8]);
      const auto pick = [&](uint32_t x) {
        return (x >> 16) == static_cast<uint32_t>(g & 3) ? x & 0xffffu : 0u;
      };
      ops::stage_wait<1>();
      __syncwarp();
      write_mma(acc, slabs + (ks & 1) * kSlabBytes, pick(lo.x) | pick(lo.y) << 16,
                pick(hi.x) | pick(hi.y) << 16, lane);
    }
    ops::stage_wait<0>();
  }
}

//: THE DENSE FORM'S CHUNK, staged: 64 tokens' level-S and level-R write rows at the tile's digits (16-byte chunks XORed
//: with the token's residue) and their gains, into parity `c & 1`: one copy of each a thread, the gains by sixteen.
//: -- see docs/internals/den/den_kernel.md#dense
__device__ __forceinline__ void stage_chunk(const DenParams& p, uint8_t* smem, int bh, int w, int c,
                                            int s0, int r0) {
  const int b = bh / p.write_heads, hw = bh % p.write_heads, par = c & 1;
  const int t0 = w * kWindow + kChunk * c;
  const ops::SmemAddr ws = ops::smem_addr(smem + Tenants::kChunkWs) + par * (kChunk * kRows * 2);
  const ops::SmemAddr wr = ops::smem_addr(smem + Tenants::kChunkWr) + par * (kChunk * kSpan * 2);
  const int tid = ops::opaque(static_cast<int>(threadIdx.x)), t = tid >> 3, part = tid & 7;
  static_assert(kChunk * 8 == kThreads, "a chunk's rows are one copy a thread");
  ops::stage_tile(slab_at(ws, t, part), plane_row(p.write_s, b, hw, t0 + t, s0 + 8 * part));
  ops::stage_tile(slab_at(wr, t, part), plane_row(p.write_r, b, hw, t0 + t, r0 + 8 * part));
  if (tid < kChunk / 4)
    ops::stage_tile(
        ops::smem_addr(smem + Tenants::kGains) + par * (kChunk * kChannels * 2) + 16 * tid,
        p.gains16 + (static_cast<int64_t>(bh) * p.L + t0) * kChannels + 8 * tid);
}

//: THE DENSE FORM'S WRITES of window `w`: eight chunks of 64 tokens, the next staged under this one's steps; a k-step's
//: B is the chunk's level-R rows by `ldmatrix.trans` (the pair of warps' eight digits, each lane taking its own
//: warp's digit across the pair by one shuffle) times the tokens' gains of its column's channel.
//: -- see docs/internals/den/den_kernel.md#dense
__device__ __forceinline__ void writes_dense(const DenParams& p, uint8_t* smem, float (&acc)[4][4],
                                             int bh, int w, int live, int s0, int r0, int warp,
                                             int lane) {
  const int g = lane >> 2, tq = lane & 3, chunks = live / kChunk;
  const int src = 4 * (4 * (warp & 1) + (g & 3)) + tq;
  const uint32_t sel = g >> 2 ? 0x7632u : 0x5410u;
#pragma unroll 1
  for (int c = 0; c < chunks; ++c) {
    stage_chunk(p, smem, bh, w, min(c + 1, chunks - 1), s0, r0);
    ops::stage_commit();
    ops::stage_wait<1>();
    ops::rendezvous();
    const int par = c & 1;
    const ops::SmemAddr ws = ops::smem_addr(smem + Tenants::kChunkWs) + par * (kChunk * kRows * 2);
    const ops::SmemAddr wr = ops::smem_addr(smem + Tenants::kChunkWr) + par * (kChunk * kSpan * 2);
    const uint32_t* const gains =
        reinterpret_cast<const uint32_t*>(smem + Tenants::kGains) + par * kChunk;
    static_for<kChunk / 32>([&](auto KP) {
      uint32_t bx[4];
      ops::load_frag_t(bx, slab_at(wr, 32 * KP + lane, warp >> 1));
      static_for<4>([&](auto I) { bx[I] = __shfl_sync(0xffffffffu, bx[I], src); });
      static_for<2>([&](auto H) {
        const int k0 = 32 * KP + 16 * H;
        const uint2 g01 = *reinterpret_cast<const uint2*>(&gains[k0 + 2 * tq]);
        const uint2 g89 = *reinterpret_cast<const uint2*>(&gains[k0 + 8 + 2 * tq]);
        const uint32_t b0 = ops::hmul2(bx[2 * H], ops::prmt(g01.x, g01.y, sel));
        const uint32_t b1 = ops::hmul2(bx[2 * H + 1], ops::prmt(g89.x, g89.y, sel));
        write_mma(acc, ws + static_cast<uint32_t>(128 * k0), b0, b1, lane);
      });
    });
  }
}

//: THE LIST FORM'S WINDOW INPUTS, issued: the span's two write unions and its list exponents into parity `w & 1` (the
//: other parity's union may still be read by a warp finishing the window before's writes).
__device__ __forceinline__ void stage_lists(const DenParams& p, uint8_t* smem, int bh, int w,
                                            int r0) {
  const int groups = (p.width_r + 31) / 32, tid = ops::opaque(static_cast<int>(threadIdx.x));
  constexpr int kUnionCopies = (kSpan / 32) * (kWindow / 8),
                kExpCopies = kSpan * kChannels * 2 / 16;
  if (tid < kUnionCopies) {
    const int gi = tid / (kWindow / 8), k = tid % (kWindow / 8);
    ops::stage_tile(
        ops::smem_addr(smem + Tenants::kUnion) + ((w & 1) * (kSpan / 32) + gi) * kWindow * 2
            + 16 * k,
        p.write_lists.union_tok
            + ((static_cast<int64_t>(bh) * p.windows + w) * groups + (r0 >> 5) + gi) * kWindow
            + 8 * k);
  } else if (tid < kUnionCopies + kExpCopies) {
    const int k = tid - kUnionCopies;
    ops::stage_tile(
        ops::smem_addr(smem + Tenants::kExps) + (w & 1) * (kSpan * kChannels * 2) + 16 * k,
        p.list_exp + ((static_cast<int64_t>(bh) * p.windows + w) * p.width_r + r0) * kChannels
            + 8 * k);
  }
}

//: A LANE'S COLUMN SCALES for window `w`: each column's exponent HELD (`kHeld`, the one its fragments sit at, moved
//: here to the window's where the list is live -- an unwritten list's -126 leaves the held one, so an empty window
//: never swings the fragments through the float range) -- the list form's staged list exponents, the dense form's
//: window gains' -- as the publish factor `2^(E_held - 4)` (dense `2^-shift_prev`) and the rescale `2^(E_held -
//: E_now)`. -- see docs/internals/den/den_kernel.md#fragments
struct Scales {
  float pub0, pub1, re0, re1;
};

__device__ __forceinline__ Scales scales_lists(uint8_t* smem, int w, int warp, int lane) {
  const Columns col = columns_of(warp, lane);
  const int16_t* const exps = reinterpret_cast<const int16_t*>(smem + Tenants::kExps);
  int16_t* const held =
      reinterpret_cast<int16_t*>(smem + Tenants::kHeld) + col.d0 * kChannels + col.ch;
  const auto raw = [&](int d) {
    return static_cast<int>(exps[((w & 1) * kSpan + d) * kChannels + col.ch]);
  };
  const int e0 = held[0], e1 = held[kChannels];
  const int k0 = raw(col.d0), k1 = raw(col.d0 + 1);
  const int n0 = k0 > -126 ? list_e(static_cast<int16_t>(k0)) : e0;
  const int n1 = k1 > -126 ? list_e(static_cast<int16_t>(k1)) : e1;
  held[0] = static_cast<int16_t>(n0);
  held[kChannels] = static_cast<int16_t>(n1);
  return Scales{pow2f(e0 - kCoefTop), pow2f(e1 - kCoefTop), pow2f(e0 - n0), pow2f(e1 - n1)};
}

__device__ __forceinline__ Scales scales_dense(const DenParams& p, int bh, int w, int lane) {
  const int ch = (lane & 3) >> 1;
  const int16_t* const e =
      p.gains_exp + (static_cast<int64_t>(bh) * p.windows + w) * kChannels + ch;
  const int now = *e, prev = w > 0 ? e[-kChannels] : now;
  const float pub = pow2f(-prev), re = pow2f(prev - now);
  return Scales{pub, pub, re, re};
}

//: THE KERNEL: grid `(tiles_s x tiles_r, BH)`, one CTA an SM. A window: the mass published at the windows before's
//: scales, a rendezvous, each band's B built, a rendezvous (the region free), the den over the window's read entries
//: (list) or tokens (dense) into the accumulator, a rendezvous, the drain, the fragments rescaled to the window, the
//: window's writes folded. -- see docs/internals/den/den_kernel.md#kernel
template <bool Dense>
__global__ __launch_bounds__(kThreads, 1) void den_kernel(const __grid_constant__ DenParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
  const int bh = static_cast<int>(blockIdx.y);
  const int s0 = static_cast<int>(blockIdx.x) % p.tiles_s * kRows;
  const int r0 = static_cast<int>(blockIdx.x) / p.tiles_s * kSpan;
  PhaseClock<kDenPhases, true> clock(reinterpret_cast<long long*>(smem + Tenants::kClock)
                                         + warp * PhaseClock<kDenPhases, true>::kRow,
                                     lane, p.ledger != nullptr);
  const ops::SmemAddr region = ops::smem_addr(smem + Tenants::kRegion);
  float acc[4][4];
  static_for<4>([&](auto M) { static_for<4>([&](auto I) { acc[M][I] = 0.0f; }); });
#pragma unroll 1
  for (int i = static_cast<int>(threadIdx.x); i < kSpan * kChannels; i += kThreads)
    reinterpret_cast<int16_t*>(smem + Tenants::kHeld)[i] = static_cast<int16_t>(kUnitMin);
#pragma unroll 1
  for (int i = static_cast<int>(threadIdx.x); i < Tenants::kAccBytes / 4; i += kThreads)
    reinterpret_cast<float*>(smem + Tenants::kAcc)[i] = 0.0f;

#pragma unroll 1
  for (int w = 0; w < p.windows; ++w) {
    const int live = min(kWindow, p.L - w * kWindow);
    //: the window's inputs issued and landed (the list form's exponents set the scales below), then every warp done
    //: with the window before (its slabs sit where the publish writes) and the inputs visible to all
    if constexpr (Dense) {
      stage_chunk(p, smem, bh, w, 0, s0, r0);
    } else {
      stage_lists(p, smem, bh, w, r0);
      ops::stage_commit();
      ops::stage_wait<0>();
    }
    ops::stage_commit();
    clock.lap(kPhaseStage);
    ops::rendezvous();
    clock.lap(kPhaseWait);
    const Scales sc = Dense ? scales_dense(p, bh, w, lane) : scales_lists(smem, w, warp, lane);
    publish(region, acc, sc.pub0, sc.pub1, warp, lane);
    ops::rendezvous();
    clock.lap(kPhasePublish);
    const DenB bf = den_b(region, warp, lane);
    ops::rendezvous();
    clock.lap(kPhaseWait);
    if constexpr (Dense) {
      den_dense(p, smem, bf, bh, w, live, s0, r0, warp, lane);
    } else {
      den_lists(p, smem, bf, bh, w, s0, r0, warp, lane);
    }
    clock.lap(kPhaseDen);
    ops::rendezvous();
    clock.lap(kPhaseWait);
    drain(p, smem, bh, w, live);
    clock.lap(kPhaseDrain);
    if constexpr (!Dense) ops::stage_wait<0>();
    rescale(acc, sc.re0, sc.re1);
    if constexpr (Dense) {
      writes_dense(p, smem, acc, bh, w, live, s0, r0, warp, lane);
    } else {
      writes_lists(p, smem, acc, bh, w, s0, r0, warp, lane);
    }
    clock.lap(kPhaseWrites);
  }
  clock.publish(p.ledger, static_cast<long>(blockIdx.y) * gridDim.x + blockIdx.x, kWarps, warp);
}

}  // namespace rola::den
