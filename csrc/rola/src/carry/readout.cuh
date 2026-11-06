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

//: THE READOUT of a chunk's slots over digits `lo .. hi - 1` of the image (plane `d & mask`, its exponent in the
//: warp's row) into the num panel and the den words, reduce `F32`, accumulate `F16`; a tile's rows past its run add
//: nothing (predicated stores). -- see docs/internals/carry/readout.md#chunk
template <class P, bool F32, bool F16>
__device__ __forceinline__ void read_chunk(const CarryParams& p, const uint8_t* smem,
                                           const Run& run, int w, int first, int lo, int hi,
                                           ops::SmemAddr image, uint32_t mask, ops::SmemAddr rows,
                                           ops::SmemAddr panel, ops::SmemAddr dens,
                                           ops::SmemAddr ents, int warp, int lane) {
  static_assert(P::kReadFactors == 1, "the value-carve readout of a listed read side");
  const int tg = warp / kChannelGroups, cg = warp % kChannelGroups, g = lane >> 2, tq = lane & 3;
  //: the entries' row offsets are the union's: the chunk's pool taken back by its first slot
  const ops::SmemAddr pool_u = rows - static_cast<uint32_t>(first * P::kReadRow);
  const ops::SmemAddr scale = ops::smem_addr(smem + p.tables.scale_at)
                              + static_cast<uint32_t>((w & 1) * p.tables.heads * kWindow);
  const ops::SmemAddr exps = exps_of<P>(smem, warp), den_fac = ops::smem_addr(smem + P::kDenFactor),
                      den_image = ops::smem_addr(smem + P::kDenImage);
  const uint32_t word = static_cast<uint32_t>(4 * cg + tq),
                 chan = static_cast<uint32_t>(8 * cg + 2 * tq);
#pragma unroll 1
  for (int d = lo; d < hi; ++d) {
    //: THE DIGIT: its run in this group's half, its slice, its exponent; the den's slice where `d mod 4` is this group's
    const int n = __shfl_sync(0xffffffffu, run.n, d), n0 = __shfl_sync(0xffffffffu, run.n0, d),
              to = __shfl_sync(0xffffffffu, run.to, d);
    const int start = to + (tg ? n0 : 0), count = tg ? n - n0 : n0;
    uint32_t bs[4], bd[4];
    ops::load_frag_t(
        bs, image + (static_cast<uint32_t>(d) & mask) * kPlaneBytes + slice_row(cg, lane, d));
    const bool den_here = (d & (kChannelGroups - 1)) == cg;
    ops::load_frag_t(bd, den_image + den_row(lane, d));
    //: the image's frame taken back: a digit at `2^(kImageTop - e)` of the fold's state, itself `2^kCoefTop` the state's
    const float dpow = pow2_wide(
        static_cast<int>(ops::load_shared_u32(exps + 4u * static_cast<uint32_t>(exp_slot(d))))
        - kImageTop - kCoefTop);
    const float dfac = ops::load_shared_f32(den_fac + 4u * static_cast<uint32_t>(d));
    const bool den_lane = den_here && tq == ((d & 7) >> 1);
#pragma unroll 1
    for (int s = 0; s < count; s += 16) {
      //: THE TILE: lane `l & 15`'s entry its row, the MMAs, then each of the lane's two tokens' panel add
      const int mine = s + (lane & 15);
      const uint2 e = ops::load_shared_u64(
          ents + 8u * static_cast<uint32_t>(start + (mine < count ? mine : s)));
      const ops::SmemAddr row0 = pool_u + 16u * (e.x & 0xffffu);
      uint32_t a0[4], a1[4];
      ops::load_frag(a0, chunk_addr<P::kReadRow>(row0, static_cast<uint32_t>(lane >> 4)));
      ops::load_frag(a1, chunk_addr<P::kReadRow>(row0, 2u + static_cast<uint32_t>(lane >> 4)));
      //: the den's MMAs on every warp (a warp-uniform `if` is a predicated HMMA, §20), the owner's stores keep them
      float acc[4], dacc[4];
      tile_mma<F16>(acc, a0, a1, bs);
      tile_mma<F16>(dacc, a0, a1, bd);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint32_t ex = __shfl_sync(0xffffffffu, e.x, g + 8 * h),
                       ey = __shfl_sync(0xffffffffu, e.y, g + 8 * h);
        const bool live = s + g + 8 * h < count;
        const uint32_t slot = slot_of<P>(ex, first);
        const float c = coef_of(ey);
        const float f =
            F32 ? c * dpow
                : c * dpow
                      * token_scale(static_cast<int8_t>(ops::load_shared_u8(scale + (ex >> 16))));
        if constexpr (F32) {
          const ops::SmemAddr at = panel + panel_word<true>(slot, chan);
          const uint2 old = ops::load_shared_u64(at);
          ops::store_shared_f32x2_if(at, __uint_as_float(old.x) + acc[2 * h] * f,
                                     __uint_as_float(old.y) + acc[2 * h + 1] * f, live);
        } else {
          const ops::SmemAddr at = panel + panel_word<false>(slot, word);
          const __half2 v = __floats2half2_rn(acc[2 * h] * f, acc[2 * h + 1] * f);
          ops::store_shared_u32_if(
              at, add_h2(ops::load_shared_u32(at), *reinterpret_cast<const uint32_t*>(&v)), live);
        }
        //: the den column the digit's (`d & 1` of the lane's pair: a select, never a runtime index into `dacc`)
        const float dn = (d & 1) ? dacc[2 * h + 1] : dacc[2 * h];
        const ops::SmemAddr dat = dens + kDenRowBytes * slot + 4u * static_cast<uint32_t>(cg);
        ops::store_shared_f32_if(dat, ops::load_shared_f32(dat) + dn * (c * dfac),
                                 live && den_lane);
      }
    }
  }
}

}  // namespace rola::carry
