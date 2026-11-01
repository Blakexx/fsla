// csrc/rola/src/den/den_kernel.cuh -- THE DEN KERNEL: one CTA a tile of 64 level-S rows x 64 level-R digits of a write
// head, window by window, in two roles: 8 WRITE warps hold the mass of the windows before as fp32 fragments (8 digits
// a warp), publish it and fold the window's writes; 8 DEN warps take each band of 8 rows' B from the publish and run
// the window's read entries against it by f16 MMAs over rows staged a tile ahead, each token's den and num
// accumulated in shared memory and drained once a window. A window's den runs beside its writes. A cell whose read
// level S and write level R list takes the list form, a cell dense on both sides the dense form.
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

//: A WINDOW'S ROWS: a plane's batch, first head and the window's first token at a digit as one 64-bit base; a row is
//: the base plus two 32-bit products (the host holds a window's rows of a write head under 2^32 bytes).
//: -- see docs/internals/den/den_kernel.md#rows
struct WindowRows {
  const uint8_t* base;
  uint32_t st, sh;

  __device__ __forceinline__ const uint8_t* at(int token, int head) const {
    return base + (static_cast<uint32_t>(token) * st + static_cast<uint32_t>(head) * sh);
  }
};

__device__ __forceinline__ WindowRows window_rows(const Plane& pl, int b, int h0, int t0, int d0) {
  return WindowRows{plane_row(pl, b, h0, t0, d0), static_cast<uint32_t>(pl.st),
                    static_cast<uint32_t>(pl.sh)};
}

//: A WRITE LANE'S TWO MASS COLUMNS of n-tile `J` (rows `16 mt + g` and `+ 8`, columns `2 tq` and `2 tq + 1`): channel
//: `tq >> 1`, digits `8 ww + 4 J + 2 (tq & 1)` and the next. -- see docs/internals/den/den_kernel.md#fragments
struct Columns {
  int ch, d0;
};

__device__ __forceinline__ Columns columns_of(int ww, int J, int lane) {
  const int tq = lane & 3;
  return Columns{tq >> 1, kDigits * ww + 4 * J + 2 * (tq & 1)};
}

//: THE MASS a write warp holds: four m-tiles of the tile's 64 rows x its two n-tiles of four digits x two channels.
using Mass = float[4][kDigitTiles][4];

//: A WRITE LANE'S COLUMN SCALES for a window, each n-tile's two columns: the publish factor and the rescale.
struct Scales {
  float pub[kDigitTiles][2], re[kDigitTiles][2];
};

//: THE PUBLISH: the fragments at the windows before's scales into the region as fp32, one 8-byte store a row and
//: n-tile. -- see docs/internals/den/den_kernel.md#publish
__device__ __forceinline__ void publish(ops::SmemAddr region, const Mass& acc, const Scales& sc,
                                        int ww, int lane) {
  const int g = lane >> 2, within = 8 * ((lane & 3) & 1);
  static_for<kDigitTiles>([&](auto J) {
    const Columns col = columns_of(ww, J, lane);
    static_for<4>([&](auto M) {
      static_for<2>([&](auto H) {
        const uint32_t at = mass_at(region, col.ch, 16 * M + g + 8 * H, col.d0 >> 2) + within;
        const float lo = acc[M][J][2 * H] * sc.pub[J][0], hi = acc[M][J][2 * H + 1] * sc.pub[J][1];
        asm volatile("st.shared.v2.f32 [%0], {%1, %2};" ::"r"(at), "f"(lo), "f"(hi) : "memory");
      });
    });
  });
}

//: THE RESCALE of the fragments to the window's scales where a column's power of two moved (one trip on a vote).
__device__ __forceinline__ void rescale(Mass& acc, const Scales& sc) {
  bool moved = false;
  static_for<kDigitTiles>(
      [&](auto J) { moved = moved || sc.re[J][0] != 1.0f || sc.re[J][1] != 1.0f; });
  for (int k = __any_sync(0xffffffffu, moved) ? 1 : 0; k > 0; --k)
    static_for<kDigitTiles>([&](auto J) {
      static_for<4>([&](auto M) {
        acc[M][J][0] *= sc.re[J][0];
        acc[M][J][1] *= sc.re[J][1];
        acc[M][J][2] *= sc.re[J][0];
        acc[M][J][3] *= sc.re[J][1];
      });
    });
}

//: THE DEN'S B: the band's eight rows x two channels as two n-tiles of the MMA's eight columns (lane `g`'s column of
//: n-tile `J`: channel `g >> 2`, row `8 band + 4 J + (g & 3)`), the row's 64 digits as f16 at the power of two putting
//: its largest under 2^15, four k-steps of two registers; `inv` the column's 2^-e. -- see docs/internals/den/den_kernel.md#den-b
struct DenB {
  uint32_t b[kBandTiles][kSteps][2];
  float inv[kBandTiles];
};

__device__ __forceinline__ DenB den_b(ops::SmemAddr region, int band, int lane) {
  const int g = lane >> 2, tq = lane & 3;
  DenB out;
  static_for<kBandTiles>([&](auto J) {
    const int ch = g >> 2, a = kBand * band + 4 * J + (g & 3);
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
    out.inv[J] = top > 0.0f ? pow2f(e - 14) : 0.0f;
    static_for<kSteps>([&](auto K) {
      static_for<2>(
          [&](auto H) { out.b[J][K][H] = ops::pack_half2(v[K][H].x * s, v[K][H].y * s); });
    });
  });
  return out;
}

//: A DEN LANE'S COLUMN FACTORS: `inv` of its columns `2 tq` and `2 tq + 1` of each n-tile, from the lanes holding them.
struct DenInv {
  float f[kBandTiles][2];
};

__device__ __forceinline__ DenInv den_inv(const DenB& bf, int lane) {
  const int tq = lane & 3;
  DenInv out;
  static_for<kBandTiles>([&](auto J) {
    out.f[J][0] = __shfl_sync(0xffffffffu, bf.inv[J], 4 * (2 * tq));
    out.f[J][1] = __shfl_sync(0xffffffffu, bf.inv[J], 4 * (2 * tq + 1));
  });
  return out;
}

//: A SLAB'S A FRAGMENTS, row-major (an entry a row, the digits K): four k-steps by `ldmatrix`, lane `l` addressing row
//: `(l & 7) + 8 ((l >> 3) & 1)` at chunk `2 kk + (l >> 4)`, each against both n-tiles into fresh fragments.
//: -- see docs/internals/den/den_kernel.md#den-tile
//: @burst
__device__ __forceinline__ void den_mma(float (&c)[kBandTiles][4], ops::SmemAddr slab,
                                        const DenB& bf, int lane) {
  const int r = (lane & 7) + 8 * ((lane >> 3) & 1), hi = lane >> 4;
  static_for<kSteps>([&](auto K) {
    uint32_t a[4];
    ops::load_frag(a, slab_at(slab, r, 2 * K + hi));
    static_for<kBandTiles>([&](auto J) {
      if constexpr (K == 0) {
        ops::hmma_fresh(c[J], a, bf.b[J][K][0], bf.b[J][K][1]);
      } else {
        ops::hmma(c[J], a, bf.b[J][K][0], bf.b[J][K][1]);
      }
    });
  });
}

//: A SLAB'S A FRAGMENTS, k-major (a token's coefficient row a slab row, the rows M): m-tile `mt` by `ldmatrix.trans`,
//: lane `l` addressing slab row `8 (l >> 4) + (l & 7)` at chunk `2 mt + ((l >> 3) & 1)`; each against both n-tiles'
//: B, eight fp32-accumulate MMAs into the fragments. -- see docs/internals/den/den_kernel.md#writes
//: @burst
__device__ __forceinline__ void write_mma(Mass& acc, ops::SmemAddr slab,
                                          const uint32_t (&b)[kDigitTiles][2], int lane) {
  const int r = 8 * (lane >> 4) + (lane & 7), half = (lane >> 3) & 1;
  static_for<4>([&](auto M) {
    uint32_t a[4];
    ops::load_frag_t(a, slab_at(slab, r, 2 * M + half));
    static_for<kDigitTiles>([&](auto J) { ops::hmma(acc[M][J], a, b[J][0], b[J][1]); });
  });
}

//: A LANE'S SLAB COPIES: where its four 16-byte chunks of row `lane >> 1` -- the row's chunks `4 (lane & 1) ..` --
//: land in a slab, XORed with the row's residue, relative to the slab: fixed a lane, formed once.
//: -- see docs/internals/den/den_kernel.md#slabs
struct SlabLane {
  uint32_t rel[4];
};

__device__ __forceinline__ SlabLane slab_lane(int lane) {
  const int r = lane >> 1, c0 = 4 * (lane & 1);
  SlabLane out;
  static_for<4>(
      [&](auto C) { out.rel[C] = static_cast<uint32_t>(128 * r + 16 * ((c0 + C) ^ (r & 7))); });
  return out;
}

//: A ROW STAGED INTO A SLAB: the lane's 64 bytes of `row` (its first digit's address) by four copies at immediate
//: offsets, predicated on `live` (a step's last tile stages nothing, branch-free: a dead copy reads nothing and
//: zero-fills a slab already consumed). -- see docs/internals/den/den_kernel.md#slabs
__device__ __forceinline__ void stage_row(ops::SmemAddr slab, const SlabLane& sl,
                                          const uint8_t* row, int lane, bool live) {
  const uint8_t* const src = row + 64 * (lane & 1);
  static_for<4>([&](auto C) { ops::stage_tile_if(slab + sl.rel[C], src + 16 * C, live); });
}

//: AN ADD INTO THE ACCUMULATOR: channel `ch`'s row of place `place` (a query head's token). On sm_80/86 a shared fp32
//: add is a compare-and-swap loop, so no step issues one: the den's results are deposited after their round.
__device__ __forceinline__ void acc_add(ops::SmemAddr acc, int heads, int ch, int place, float v) {
  asm volatile("red.shared.add.f32 [%0], %1;" ::"r"(
                   acc + 4u * static_cast<uint32_t>(ch * heads * kWindow + place)),
               "f"(v)
               : "memory");
}

//: THE DRAIN, by the den warps: the accumulator's rows to `den` and `num` (`red.global.add.f32`, a warp's lanes 32
//: consecutive tokens of one channel and query head), each row zeroed behind its read.
//: -- see docs/internals/den/den_kernel.md#drain
__device__ __forceinline__ void drain(const DenParams& p, uint8_t* smem, int bh, int w, int live,
                                      int dt) {
  float* const acc = reinterpret_cast<float*>(smem + Tenants::kAcc);
#pragma unroll 1
  for (int jj = 0; jj < kChannels * p.heads; ++jj) {
    const int ch = jj / p.heads, j = jj - ch * p.heads;
    float* const out = (ch ? p.num : p.den) + (static_cast<int64_t>(bh) * p.heads + j) * p.L
                       + static_cast<int64_t>(w) * kWindow;
#pragma unroll 1
    for (int t = ops::opaque(dt); t < live; t += kRoleThreads) {
      const int at = (ch * p.heads + j) * kWindow + t;
      const float v = acc[at];
      acc[at] = 0.0f;
      asm volatile("red.global.add.f32 [%0], %1;" ::"l"(out + t), "f"(v) : "memory");
    }
  }
}

//: A LIST STREAM'S RUNS: `Lists` lists from `pre` (each list's prefix row's last column, `stride` apart): each list's
//: first entry in the stream, the stream's length last. -- see docs/internals/den/den_kernel.md#den-list
template <int Lists>
__device__ __forceinline__ void runs_of(int (&first)[Lists + 1], const uint16_t* pre, int stride) {
  first[0] = 0;
  static_for<Lists>([&](auto I) {
    first[I + 1] =
        first[I]
        + ((static_cast<int>(pre[(I + 1) * stride]) - static_cast<int>(pre[I * stride])) & 0xffff);
  });
}

//: An entry's list in its stream and its offset in the list, by `Lists - 1` compares against the runs.
template <int Lists>
__device__ __forceinline__ int2 list_of(const int (&first)[Lists + 1], int e) {
  int list = 0, off = e;
  static_for<Lists - 1>([&](auto I) {
    const bool past = e >= first[I + 1];
    list = past ? static_cast<int>(I) + 1 : list;
    off = past ? e - first[I + 1] : off;
  });
  return make_int2(list, off);
}

//: THE LIST FORM'S DEN over a band's eight lists of window `w`: their entries one stream, staged a round of `kRound`
//: at a time into the warp's row of words (`place | list << kPlaceBits | coef << 16`); a tile 16 entries, its rows a
//: slab staged one tile ahead from a word read a tile before; an entry's results its row's two columns, each stored
//: by the lane holding it as `coef x C x 2^-e` in the warp's results, and the round's results deposited into their
//: tokens' accumulator rows after its tiles. -- see docs/internals/den/den_kernel.md#den-list
__device__ __forceinline__ void den_lists(const DenParams& p, uint8_t* smem, const DenB& bf, int bh,
                                          int w, int s0, int r0, int band, int lane) {
  const int g = lane >> 2, tq = lane & 3, heads = p.heads;
  const int b = bh / p.write_heads, hw = bh % p.write_heads;
  const int groups = (p.width_s + 31) / 32, cols = kListAt * heads;
  const int s = s0 + kBand * band, gs = s >> 5, l0 = s & 31;
  int first[kBand + 1];
  runs_of<kBand>(first,
                 p.read_lists.pre
                     + (static_cast<int64_t>(bh) * p.windows + w) * groups * kPreRows * cols
                     + (static_cast<int64_t>(gs) * kPreRows + l0) * cols + (cols - 1),
                 cols);
  const int n = first[kBand];
  const uint2* const lists =
      p.read_lists.entries
      + (static_cast<int64_t>(bh) * p.windows + w) * p.width_s * (kListCap * heads)
      + static_cast<int64_t>(s) * (kListCap * heads);
  uint32_t* const words = reinterpret_cast<uint32_t*>(smem + Tenants::kWords) + band * kRound;
  //: each entry's two results; until its tile is staged, its first slot holds its row's offset in the window
  uint32_t* const slots =
      reinterpret_cast<uint32_t*>(smem + Tenants::kResults) + band * kRound * kChannels;
  const ops::SmemAddr results = ops::smem_addr(slots);
  const ops::SmemAddr slabs = ops::smem_addr(smem + Tenants::kMass) + band * (2 * kSlabBytes);
  const ops::SmemAddr acc = ops::smem_addr(smem + Tenants::kAcc);
  const DenInv inv = den_inv(bf, lane);
  const SlabLane sl = slab_lane(lane);
  const WindowRows rows = window_rows(p.read_r, b, hw * heads, w * kWindow, r0);
  constexpr uint32_t kPlaceMask = (1u << kPlaceBits) - 1;

#pragma unroll 1
  for (int base = 0; base < n; base += kRound) {
    //: THE ROUND: its entries' words and row offsets
    const int count = min(kRound, n - base), last = count - 1;
    ops::warp_sync();
#pragma unroll 1
    for (int k = lane; k < count; k += 32) {
      const int2 at = list_of<kBand>(first, base + k);
      const uint2 v = lists[static_cast<int64_t>(at.x) * (kListCap * heads) + at.y];
      const uint32_t place = v.x >> 16;
      words[k] = place | static_cast<uint32_t>(at.x) << kPlaceBits | (v.y & 0xffffu) << 16;
      slots[kChannels * k] =
          (place & (kWindow - 1)) * rows.st + (place >> window_bits(kWindow)) * rows.sh;
    }
    ops::warp_sync();

    //: THE TILES
    const auto offset = [&](int i) { return slots[kChannels * min(i, last)]; };
    const int tiles = (count + kTileRows - 1) / kTileRows;
    //: tile 0's rows issued; each tile issues the next's into the other slab before its own MMAs, from an offset read
    //: a tile before, and reads the next tile's words under its MMAs
    stage_row(slabs, sl, rows.base + offset(lane >> 1), lane, true);
    ops::stage_commit();
    uint32_t ahead = offset(kTileRows + (lane >> 1)), wd[2] = {words[g], words[g + 8]};
#pragma unroll 1
    for (int i = 0; i < tiles; ++i) {
      const uint32_t next = ahead, now[2] = {wd[0], wd[1]};
      ahead = offset(kTileRows * (i + 2) + (lane >> 1));
      wd[0] = words[(kTileRows * (i + 1) + g) & (kRound - 1)];
      wd[1] = words[(kTileRows * (i + 1) + g + 8) & (kRound - 1)];
      stage_row(slabs + ((i + 1) & 1) * kSlabBytes, sl, rows.base + next, lane, i + 1 < tiles);
      ops::stage_commit();
      ops::stage_wait<1>();
      ops::warp_sync();
      float c[kBandTiles][4];
      den_mma(c, slabs + (i & 1) * kSlabBytes, bf, lane);
      static_for<2>([&](auto H) {
        const int e = kTileRows * i + g + 8 * H;
        const int row = static_cast<int>(now[H] >> kPlaceBits) & ((1 << kListBits) - 1);
        const bool x = (row & 1) != 0;
        const float coef = __half2float(__ushort_as_half(static_cast<uint16_t>(now[H] >> 16)));
        float cv = (x ? c[0][2 * H + 1] : c[0][2 * H]) * (x ? inv.f[0][1] : inv.f[0][0]);
        static_for<kBandTiles - 1>([&](auto J1) {
          constexpr int J = J1 + 1;
          const float cj = (x ? c[J][2 * H + 1] : c[J][2 * H]) * (x ? inv.f[J][1] : inv.f[J][0]);
          cv = (row >> 2) == J ? cj : cv;
        });
        ops::store_shared_f32_if(results + 4u * static_cast<uint32_t>(kChannels * e + (tq >> 1)),
                                 coef * cv, e < count && ((row >> 1) & 1) == (tq & 1));
      });
    }
    ops::stage_wait<0>();

    //: THE DEPOSIT: the round's results into their tokens' accumulator rows, outside the steps
    ops::warp_sync();
#pragma unroll 1
    for (int k = lane; k < count; k += 32) {
      const int place = static_cast<int>(words[k] & kPlaceMask);
      const float2 r = *reinterpret_cast<const float2*>(&slots[kChannels * k]);
      acc_add(acc, heads, 0, place, r.x);
      acc_add(acc, heads, 1, place, r.y);
    }
  }
}

//: THE DENSE FORM'S DEN over the window's tokens of every query head, in BATCHES of four tiles (64 tokens of one query
//: head): a batch's level-R and level-S read rows staged ONCE into one of two parities by the den warps together; each
//: band's MMAs on the batch's tiles, its eight rows summed with the tokens' coefficients from the staged level-S rows,
//: each token's band sum stored in the batch's partials; the batch before's partials summed over the bands into the
//: accumulator, all under one den-role barrier a batch. No atomics: each token's row is written once a window.
//: -- see docs/internals/den/den_kernel.md#den-dense
__device__ __forceinline__ void den_dense(const DenParams& p, uint8_t* smem, const DenB& bf, int bh,
                                          int w, int live, int s0, int r0, int band, int lane) {
  const int g = lane >> 2, tq = lane & 3, heads = p.heads, per_head = live / kChunk;
  const int b = bh / p.write_heads, hw = bh % p.write_heads, batches = heads * per_head;
  const int dt = band * 32 + lane;
  constexpr int kRowsBytes = kChunk * 128, kBatchBytes = 2 * kRowsBytes;
  const ops::SmemAddr buf = ops::smem_addr(smem + Tenants::kMass);
  float* const partials = reinterpret_cast<float*>(smem + Tenants::kPartials);
  float* const acc = reinterpret_cast<float*>(smem + Tenants::kAcc);
  const DenInv inv = den_inv(bf, lane);
  const WindowRows rows_r = window_rows(p.read_r, b, hw * heads, w * kWindow, r0);
  const WindowRows rows_s = window_rows(p.read_s, b, hw * heads, w * kWindow, s0);

  //: a batch's copies into parity `par`: 64 tokens' 128-byte level-R then level-S rows, four copies a den thread
  const auto stage = [&](int head, int t0, int par, bool live_batch) {
    const ops::SmemAddr dst = buf + par * kBatchBytes;
    static_for<4>([&](auto K) {
      const int i = dt + kRoleThreads * (K & 1), tok = i >> 3, part = i & 7;
      const WindowRows& rows = K < 2 ? rows_r : rows_s;
      ops::stage_tile_if(slab_at(dst + (K >> 1) * kRowsBytes, tok, part),
                         rows.at(t0 + tok, head) + 16 * part, live_batch);
    });
  };

  //: a batch's partials summed over the bands: each value's eight band slots by two threads, four apiece
  const auto reduce = [&](int head, int t0, int par, bool live_batch) {
    const int v = dt >> 1, half = dt & 1, j = v >> 5, idx = v & 31;
    const float4 q = *reinterpret_cast<const float4*>(
        &partials[((par * kBatchTiles + j) * (kTileRows * kChannels) + idx) * kDenWarps
                  + 4 * half]);
    float sum = (q.x + q.y) + (q.z + q.w);
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    const int tok = idx >> 1, ch = idx & 1;
    if (live_batch && half == 0)
      acc[(ch * heads + head) * kWindow + t0 + kTileRows * j + tok] = sum;
  };
  static_assert(kBatchTiles * kTileRows * kChannels * kDenWarps == 4 * kRoleThreads,
                "a batch's partials are four a den thread");

  stage(0, 0, 0, true);
  ops::stage_commit();
  //: batch `k`'s query head and first token, the next's and the one before's, counted forward
  int head = 0, t0 = 0, phead = 0, pt0 = 0;
  int nhead = per_head > 1 ? 0 : 1, nt0 = per_head > 1 ? kChunk : 0;
#pragma unroll 1
  for (int k = 0; k < batches; ++k) {
    ops::stage_wait<0>();
    ops::rendezvous_group(kBarDen, kRoleThreads);
    stage(nhead, nt0, (k + 1) & 1, k + 1 < batches);
    ops::stage_commit();
    reduce(phead, pt0, (k + 1) & 1, k > 0);

    //: THE BATCH'S TILES
    const ops::SmemAddr rb = buf + (k & 1) * kBatchBytes, sbuf = rb + kRowsBytes;
    float* const mine = partials + (k & 1) * kBatchTiles * kTileRows * kChannels * kDenWarps;
    static_for<kBatchTiles>([&](auto T) {
      uint32_t ca[kBandTiles], cb[kBandTiles];
      static_for<kBandTiles>([&](auto J) {
        const uint32_t off = 8 * J + 4 * (tq & 1);
        ca[J] = ops::load_shared_u32(slab_at(sbuf, kTileRows * T + g, band) + off);
        cb[J] = ops::load_shared_u32(slab_at(sbuf, kTileRows * T + g + 8, band) + off);
      });
      float c[kBandTiles][4];
      den_mma(c, rb + T * kSlabBytes, bf, lane);
      float sa = 0.0f, sb = 0.0f;
      static_for<kBandTiles>([&](auto J) {
        const float2 xa = __half22float2(*reinterpret_cast<const __half2*>(&ca[J]));
        const float2 xb = __half22float2(*reinterpret_cast<const __half2*>(&cb[J]));
        sa += xa.x * c[J][0] * inv.f[J][0] + xa.y * c[J][1] * inv.f[J][1];
        sb += xb.x * c[J][2] * inv.f[J][0] + xb.y * c[J][3] * inv.f[J][1];
      });
      sa += __shfl_xor_sync(0xffffffffu, sa, 1);
      sb += __shfl_xor_sync(0xffffffffu, sb, 1);
      //: the lane's token (`g`, or `g + 8` on odd `tq`) and channel `tq >> 1`, its band's slot rotated by the value
      //: so a warp's 32 stores meet distinct banks
      const int idx = 2 * (g + 8 * (tq & 1)) + (tq >> 1);
      mine[(T * (kTileRows * kChannels) + idx) * kDenWarps
           + ((band + (idx >> 2)) & (kDenWarps - 1))] = (tq & 1) ? sb : sa;
    });
    phead = head;
    pt0 = t0;
    head = nhead;
    t0 = nt0;
    nt0 += kChunk;
    const bool wrap = nt0 == live;
    nhead = wrap ? nhead + 1 : nhead;
    nt0 = wrap ? 0 : nt0;
  }
  ops::rendezvous_group(kBarDen, kRoleThreads);
  reduce(phead, pt0, (batches + 1) & 1, true);
}

//: THE LIST FORM'S WRITES of window `w`: the warp's eight lists at level R as one stream, 64 entries a chunk into the
//: warp's slots (the token from the group's union, each channel's coefficient with its list), then k-steps of 16: the
//: entries' tokens' level-S rows staged a step ahead into the warp's two slabs, a lane's B its column's channel of
//: the two entries at its k positions where their list is its column's digit, else zero. -- see docs/internals/den/den_kernel.md#writes
__device__ __forceinline__ void writes_lists(const DenParams& p, uint8_t* smem, Mass& acc, int bh,
                                             int w, int s0, int r0, int ww, int lane) {
  const int g = lane >> 2, tq = lane & 3;
  const int b = bh / p.write_heads, hw = bh % p.write_heads;
  const int groups = (p.width_r + 31) / 32;
  const int r = r0 + kDigits * ww, gr = r >> 5, l0 = r & 31;
  int first[kDigits + 1];
  runs_of<kDigits>(first,
                   p.write_lists.pre
                       + (static_cast<int64_t>(bh) * p.windows + w) * groups * kPreRows * kListAt
                       + (static_cast<int64_t>(gr) * kPreRows + l0) * kListAt + (kListAt - 1),
                   kListAt);
  const int n = first[kDigits];
  const uint2* const lists = p.write_lists.entries
                             + (static_cast<int64_t>(bh) * p.windows + w) * p.width_r * kListCap
                             + static_cast<int64_t>(r) * kListCap;
  const uint16_t* const toks = reinterpret_cast<const uint16_t*>(smem + Tenants::kUnion)
                               + ((w & 1) * (kSpan / 32) + (gr - (r0 >> 5))) * kWindow;
  uint16_t* const slot_token =
      reinterpret_cast<uint16_t*>(smem + Tenants::kSlotToken) + ww * (kChunk + 16);
  uint32_t* const slot_c =
      reinterpret_cast<uint32_t*>(smem + Tenants::kSlotC) + ww * kChannels * (kChunk + 16);
  const uint32_t* const mine = slot_c + (g >> 2) * (kChunk + 16);
  const ops::SmemAddr slabs = ops::smem_addr(smem + Tenants::kWriteRegion) + ww * (2 * kSlabBytes);
  const SlabLane sl = slab_lane(lane);
  const WindowRows rows = window_rows(p.write_s, b, hw, w * kWindow, s0);

#pragma unroll 1
  for (int c0 = 0; c0 < n; c0 += kChunk) {
    ops::warp_sync();
    static_for<2>([&](auto H) {
      const int k = 32 * H + lane, e = c0 + k;
      const int2 at = list_of<kDigits>(first, e);
      const uint2 v =
          e < n ? lists[static_cast<int64_t>(at.x) * kListCap + at.y] : make_uint2(0u, 0u);
      slot_token[k] = toks[v.x & 0xffffu] & 0x7fffu;
      slot_c[k] = (v.y >> 16) | static_cast<uint32_t>(at.x) << 16;
      slot_c[(kChunk + 16) + k] = (v.y & 0xffffu) | static_cast<uint32_t>(at.x) << 16;
    });
    ops::warp_sync();
    const int steps = (min(kChunk, n - c0) + 15) >> 4;
    stage_row(slabs, sl, rows.at(slot_token[lane >> 1], 0), lane, true);
    ops::stage_commit();
    int ahead = slot_token[16 * min(1, steps - 1) + (lane >> 1)];
#pragma unroll 1
    for (int ks = 0; ks < steps; ++ks) {
      const int next = ahead;
      ahead = slot_token[16 * min(ks + 2, steps - 1) + (lane >> 1)];
      stage_row(slabs + ((ks + 1) & 1) * kSlabBytes, sl, rows.at(next, 0), lane, ks + 1 < steps);
      ops::stage_commit();
      const uint2 lo = *reinterpret_cast<const uint2*>(&mine[16 * ks + 2 * tq]);
      const uint2 hi = *reinterpret_cast<const uint2*>(&mine[16 * ks + 2 * tq + 8]);
      uint32_t bj[kDigitTiles][2];
      static_for<kDigitTiles>([&](auto J) {
        const uint32_t digit = 4 * J + (g & 3);
        const auto pick = [&](uint32_t x) { return (x >> 16) == digit ? x & 0xffffu : 0u; };
        bj[J][0] = pick(lo.x) | pick(lo.y) << 16;
        bj[J][1] = pick(hi.x) | pick(hi.y) << 16;
      });
      ops::stage_wait<1>();
      ops::warp_sync();
      write_mma(acc, slabs + (ks & 1) * kSlabBytes, bj, lane);
    }
    ops::stage_wait<0>();
  }
}

//: THE DENSE FORM'S CHUNK, staged by the write warps: 64 tokens' level-S and level-R write rows at the tile's digits
//: (16-byte chunks XORed with the token's residue) into parity `par`: two copies of each plane a write thread, every
//: copy predicated on `live` (a step stages nothing past the call's last chunk, branch-free).
//: -- see docs/internals/den/den_kernel.md#dense
__device__ __forceinline__ void stage_chunk(const DenParams& p, uint8_t* smem, int b, int hw, int w,
                                            int c, int par, int s0, int r0, int wt, bool live) {
  const int t0 = w * kWindow + kChunk * c;
  const ops::SmemAddr ws = ops::smem_addr(smem + Tenants::kChunkWs) + par * (kChunk * kRows * 2);
  const ops::SmemAddr wr = ops::smem_addr(smem + Tenants::kChunkWr) + par * (kChunk * kSpan * 2);
  const int t = ops::opaque(wt);
  static_assert(kChunk * 8 == 2 * kRoleThreads,
                "a chunk's rows of a plane are two copies a write thread");
  static_for<2>([&](auto K) {
    const int i = t + kRoleThreads * K, tok = i >> 3, part = i & 7;
    ops::stage_tile_if(slab_at(ws, tok, part), plane_row(p.write_s, b, hw, t0 + tok, s0 + 8 * part),
                       live);
    ops::stage_tile_if(slab_at(wr, tok, part), plane_row(p.write_r, b, hw, t0 + tok, r0 + 8 * part),
                       live);
  });
}

//: THE DENSE FORM'S WINDOW GAINS, staged by the write warps outside the steps: window `w`'s live tokens' `(m, F)`
//: f16 pairs into parity `w & 1`, 16 bytes a thread. -- see docs/internals/den/den_kernel.md#dense
__device__ __forceinline__ void stage_gains(const DenParams& p, uint8_t* smem, int bh, int w,
                                            int wt) {
  const int live = min(kWindow, p.L - w * kWindow);
  if (wt < live * kChannels * 2 / 16)
    ops::stage_tile(
        ops::smem_addr(smem + Tenants::kGainsW) + (w & 1) * (kWindow * kChannels * 2) + 16 * wt,
        p.gains16 + (static_cast<int64_t>(bh) * p.L + static_cast<int64_t>(w) * kWindow) * kChannels
            + 8 * wt);
}

//: THE DENSE FORM'S WRITES of window `w`: its chunks of 64 tokens, the `q`-th chunk of the call in parity `q & 1`:
//: landed and every write warp done with the chunk before (one write-role barrier), the next chunk -- the window's,
//: else the next window's first -- issued into the other parity, then the chunk's steps. A k-step's B is the chunk's
//: level-R rows by `ldmatrix.trans` over the warp's eight digits, each lane taking its column's digit by one shuffle
//: an n-tile, times the tokens' gains of its column's channel. Returns the next chunk's `q`.
//: -- see docs/internals/den/den_kernel.md#dense
__device__ __forceinline__ int writes_dense(const DenParams& p, uint8_t* smem, Mass& acc, int bh,
                                            int w, int live, int q, int s0, int r0, int ww,
                                            int lane, int wt) {
  const int g = lane >> 2, tq = lane & 3, chunks = live / kChunk;
  const int b = bh / p.write_heads, hw = bh % p.write_heads;
  const uint32_t sel = g >> 2 ? 0x7632u : 0x5410u;

#pragma unroll 1
  for (int c = 0; c < chunks; ++c, ++q) {
    ops::stage_wait<0>();
    ops::rendezvous_group(kBarWrite, kRoleThreads);
    const bool more = c + 1 < chunks, after = !more && w + 1 < p.windows;
    stage_chunk(p, smem, b, hw, after ? w + 1 : w,
                more    ? c + 1
                : after ? 0
                        : c,
                (q + 1) & 1, s0, r0, wt, more || after);
    ops::stage_commit();

    //: THE CHUNK'S STEPS
    const int par = q & 1;
    const ops::SmemAddr ws = ops::smem_addr(smem + Tenants::kChunkWs) + par * (kChunk * kRows * 2);
    const ops::SmemAddr wr = ops::smem_addr(smem + Tenants::kChunkWr) + par * (kChunk * kSpan * 2);
    const uint32_t* const gains =
        reinterpret_cast<const uint32_t*>(smem + Tenants::kGainsW) + (w & 1) * kWindow + kChunk * c;
    static_for<kChunk / 32>([&](auto KP) {
      uint32_t bx[4];
      ops::load_frag_t(bx, slab_at(wr, 32 * KP + lane, ww));
      uint32_t by[kDigitTiles][4];
      static_for<kDigitTiles>([&](auto J) {
        const int src = 4 * (4 * J + (g & 3)) + tq;
        static_for<4>([&](auto I) { by[J][I] = __shfl_sync(0xffffffffu, bx[I], src); });
      });
      static_for<2>([&](auto H) {
        const int k0 = 32 * KP + 16 * H;
        const uint2 g01 = *reinterpret_cast<const uint2*>(&gains[k0 + 2 * tq]);
        const uint2 g89 = *reinterpret_cast<const uint2*>(&gains[k0 + 8 + 2 * tq]);
        const uint32_t m01 = ops::prmt(g01.x, g01.y, sel), m89 = ops::prmt(g89.x, g89.y, sel);
        uint32_t bj[kDigitTiles][2];
        static_for<kDigitTiles>([&](auto J) {
          bj[J][0] = ops::hmul2(by[J][2 * H], m01);
          bj[J][1] = ops::hmul2(by[J][2 * H + 1], m89);
        });
        write_mma(acc, ws + static_cast<uint32_t>(128 * k0), bj, lane);
      });
    });
  }
  return q;
}

//: THE LIST FORM'S WINDOW INPUTS, issued by the write warps: the span's two write unions and its list exponents into
//: parity `w & 1` (the other parity is the window before's, read until its writes end).
__device__ __forceinline__ void stage_lists(const DenParams& p, uint8_t* smem, int bh, int w,
                                            int r0, int wt) {
  const int groups = (p.width_r + 31) / 32;
  constexpr int kUnionCopies = (kSpan / 32) * (kWindow / 8),
                kExpCopies = kSpan * kChannels * 2 / 16;
#pragma unroll 1
  for (int i = ops::opaque(wt); i < kUnionCopies + kExpCopies; i += kRoleThreads) {
    if (i < kUnionCopies) {
      const int gi = i / (kWindow / 8), k = i % (kWindow / 8);
      ops::stage_tile(
          ops::smem_addr(smem + Tenants::kUnion) + ((w & 1) * (kSpan / 32) + gi) * kWindow * 2
              + 16 * k,
          p.write_lists.union_tok
              + ((static_cast<int64_t>(bh) * p.windows + w) * groups + (r0 >> 5) + gi) * kWindow
              + 8 * k);
    } else {
      const int k = i - kUnionCopies;
      ops::stage_tile(
          ops::smem_addr(smem + Tenants::kExps) + (w & 1) * (kSpan * kChannels * 2) + 16 * k,
          p.list_exp + ((static_cast<int64_t>(bh) * p.windows + w) * p.width_r + r0) * kChannels
              + 8 * k);
    }
  }
}

//: A WRITE LANE'S COLUMN SCALES for window `w`: each column's exponent HELD (`kHeld`, the one its fragments sit at,
//: moved here to the window's where the list is live -- an unwritten list's -126 leaves the held one, so an empty
//: window never swings the fragments through the float range) -- the list form's staged list exponents, the dense
//: form's window gains' -- as the publish factor `2^(E_held - 4)` (dense `2^-shift_prev`) and the rescale `2^(E_held -
//: E_now)`. -- see docs/internals/den/den_kernel.md#fragments
__device__ __forceinline__ Scales scales_lists(uint8_t* smem, int w, int ww, int lane) {
  const int16_t* const exps = reinterpret_cast<const int16_t*>(smem + Tenants::kExps);
  Scales out;
  static_for<kDigitTiles>([&](auto J) {
    const Columns col = columns_of(ww, J, lane);
    int16_t* const held =
        reinterpret_cast<int16_t*>(smem + Tenants::kHeld) + col.d0 * kChannels + col.ch;
    static_for<2>([&](auto X) {
      const int e = held[X * kChannels];
      const int k = exps[((w & 1) * kSpan + col.d0 + X) * kChannels + col.ch];
      const int now = k > -126 ? list_e(static_cast<int16_t>(k)) : e;
      held[X * kChannels] = static_cast<int16_t>(now);
      out.pub[J][X] = pow2f(e - kCoefTop);
      out.re[J][X] = pow2f(e - now);
    });
  });
  return out;
}

__device__ __forceinline__ Scales scales_dense(int prev, int now) {
  Scales out;
  static_for<kDigitTiles>([&](auto J) {
    static_for<2>([&](auto X) {
      out.pub[J][X] = pow2f(-prev);
      out.re[J][X] = pow2f(prev - now);
    });
  });
  return out;
}

//: THE WRITE ROLE's windows: the publish handed to the den warps, the rescale, the window's writes with the next
//: window's inputs issued under them. -- see docs/internals/den/den_kernel.md#kernel
template <bool Dense>
__device__ __forceinline__ void write_role(const DenParams& p, uint8_t* smem,
                                           PhaseClock<kDenPhases, true>& clock, int bh, int s0,
                                           int r0, int ww, int lane) {
  const ops::SmemAddr region = ops::smem_addr(smem + Tenants::kMass);
  const int wt = ww * 32 + lane;
  Mass acc;
  static_for<4>([&](auto M) {
    static_for<kDigitTiles>([&](auto J) { static_for<4>([&](auto I) { acc[M][J][I] = 0.0f; }); });
  });
  int q = 0, last = 0, cur = 0;
  const int16_t* gexp = nullptr;
  if constexpr (Dense) {
    gexp = p.gains_exp + static_cast<int64_t>(bh) * p.windows * kChannels + ((lane & 3) >> 1);
    cur = *gexp;
    last = cur;
    stage_chunk(p, smem, bh / p.write_heads, bh % p.write_heads, 0, 0, 0, s0, r0, wt, true);
    stage_gains(p, smem, bh, 0, wt);
  } else {
    stage_lists(p, smem, bh, 0, r0, wt);
  }
  ops::stage_commit();
  if constexpr (!Dense) ops::stage_wait<0>();

#pragma unroll 1
  for (int w = 0; w < p.windows; ++w) {
    //: THE TOP AND THE PUBLISH
    const int live = min(kWindow, p.L - w * kWindow);
    clock.lap(kPhaseStage);
    ops::rendezvous();
    clock.lap(kPhaseWait);
    const Scales sc = Dense ? scales_dense(last, cur) : scales_lists(smem, w, ww, lane);
    publish(region, acc, sc, ww, lane);
    ops::rendezvous_arrive(kBarPublish, kThreads);
    clock.lap(kPhasePublish);

    //: THE WRITES, the next window's inputs issued under them
    rescale(acc, sc);
    if constexpr (Dense) {
      //: the next window's gains exponent, read under this window's writes
      int nxt = cur;
#pragma unroll 1
      for (int k = ops::once(w + 1 < p.windows); k > 0; --k) {
        nxt = gexp[static_cast<int64_t>(w + 1) * kChannels];
        stage_gains(p, smem, bh, w + 1, wt);
      }
      ops::stage_commit();
      q = writes_dense(p, smem, acc, bh, w, live, q, s0, r0, ww, lane, wt);
      last = cur;
      cur = nxt;
    } else {
#pragma unroll 1
      for (int k = ops::once(w + 1 < p.windows); k > 0; --k)
        stage_lists(p, smem, bh, w + 1, r0, wt);
      ops::stage_commit();
      writes_lists(p, smem, acc, bh, w, s0, r0, ww, lane);
      ops::stage_wait<0>();
    }
    clock.lap(kPhaseWrites);
  }
}

//: THE DEN ROLE's windows: B from the publish, the region freed among the den warps, the window's den into the
//: accumulator and its drain. -- see docs/internals/den/den_kernel.md#kernel
template <bool Dense>
__device__ __forceinline__ void den_role(const DenParams& p, uint8_t* smem,
                                         PhaseClock<kDenPhases, true>& clock, int bh, int s0,
                                         int r0, int band, int lane) {
  const ops::SmemAddr region = ops::smem_addr(smem + Tenants::kMass);
  const int dt = band * 32 + lane;
#pragma unroll 1
  for (int w = 0; w < p.windows; ++w) {
    const int live = min(kWindow, p.L - w * kWindow);
    ops::rendezvous();
    ops::rendezvous_group(kBarPublish, kThreads);
    clock.lap(kPhaseWait);
    const DenB bf = den_b(region, band, lane);
    ops::rendezvous_group(kBarDen, kRoleThreads);
    clock.lap(kPhasePublish);
    if constexpr (Dense) {
      den_dense(p, smem, bf, bh, w, live, s0, r0, band, lane);
    } else {
      den_lists(p, smem, bf, bh, w, s0, r0, band, lane);
    }
    clock.lap(kPhaseDen);
    ops::rendezvous_group(kBarDen, kRoleThreads);
    clock.lap(kPhaseWait);
    drain(p, smem, bh, w, live, dt);
    clock.lap(kPhaseDrain);
  }
}

//: THE KERNEL: grid `(tiles_s x tiles_r, BH)`, one CTA an SM, warps `0 .. 7` the write role and `8 .. 15` the den
//: role; every warp meets at each window's top barrier (the window before's slabs, adds and writes done).
//: -- see docs/internals/den/den_kernel.md#kernel
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
#pragma unroll 1
  for (int i = static_cast<int>(threadIdx.x); i < kSpan * kChannels; i += kThreads)
    reinterpret_cast<int16_t*>(smem + Tenants::kHeld)[i] = static_cast<int16_t>(kUnitMin);
#pragma unroll 1
  for (int i = static_cast<int>(threadIdx.x); i < Tenants::kAccBytes / 4; i += kThreads)
    reinterpret_cast<float*>(smem + Tenants::kAcc)[i] = 0.0f;
  if (warp < kWriteWarps) {
    write_role<Dense>(p, smem, clock, bh, s0, r0, warp, lane);
  } else {
    den_role<Dense>(p, smem, clock, bh, s0, r0, warp - kWriteWarps, lane);
  }
  clock.publish(p.ledger, static_cast<long>(blockIdx.y) * gridDim.x + blockIdx.x, kWarps, warp);
}

}  // namespace rola::den
