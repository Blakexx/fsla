// csrc/rola/src/carry/panel.cuh -- THE PANEL, a read chunk's num and den words, and THE DRAIN of both to HBM.
// see docs/internals/carry/panel.md
#pragma once

#include "carry/parts.cuh"

namespace rola::carry {

namespace ops = ::denseref::ops;

//: SLOT `s`'s NUM WORD `w` in the panel, bytes, rotated by the slot. -- see docs/internals/carry/panel.md#words
template <bool F32>
__host__ __device__ __forceinline__ constexpr uint32_t panel_word(uint32_t s, uint32_t w) {
  return F32 ? 128u * s + 4u * ((w + 4u * (s & 7u)) & 31u)
             : 64u * s + 4u * ((w + 4u * ((s >> 1) & 3u)) & 15u);
}

//: THE DRAIN: a chunk's panel to `num` and its den words to `den`, after every warp's adds; warp `w` slots `w + 8 i`, a
//: gap slot adding nothing. -- see docs/internals/carry/panel.md#drain
template <class P, bool F32>
__device__ __forceinline__ void drain(const CarryParams& p, const uint16_t* union_tok, int first,
                                      int slots, ops::SmemAddr panel, ops::SmemAddr dens,
                                      int64_t head0, int channel0, int warp, int lane) {
  const uint32_t L = static_cast<uint32_t>(p.L), dv = static_cast<uint32_t>(p.dv);
  //: a slot's row (`token_row`: a word offset under 2^32, the host's bound), and whether it holds a token
  const auto row_of = [&](int s) { return token_row(union_tok[first + s] & 0x7fffu, L); };
  const auto live = [&](int s) { return (union_tok[first + s] & 0x8000u) == 0u; };
  if constexpr (F32) {
    float* const out = reinterpret_cast<float*>(p.num) + head0 * p.dv + channel0 + lane;
#pragma unroll 1
    for (int s = warp; s < slots; s += kWarps) {
      if (!live(s)) continue;
      red_add_f32(word_at(out, row_of(s) * dv),
                  ops::load_shared_f32(panel + panel_word<true>(static_cast<uint32_t>(s), lane)));
    }
  } else {
    const int half = lane >> 4, word = lane & 15;
    uint32_t* const out = p.num + head0 * (p.dv / 2) + channel0 / 2 + word;
#pragma unroll 1
    for (int pair = warp; 2 * pair < slots; pair += kWarps) {
      const int s = 2 * pair + half;
      if (s < slots && live(s))
        red_add_h2(word_at(out, row_of(s) * (dv / 2)),
                   ops::load_shared_u32(panel + panel_word<false>(static_cast<uint32_t>(s), word)));
    }
  }
  //: THE DEN: a thread a slot, its four words summed into one add
#pragma unroll 1
  for (int s = warp * 32 + lane; s < slots; s += kThreads) {
    if (!live(s)) continue;
    const uint4 d = ops::load_shared_vec16(dens + kDenRowBytes * static_cast<uint32_t>(s));
    red_add_f32(word_at(p.den + head0, row_of(s)),
                (__uint_as_float(d.x) + __uint_as_float(d.y))
                    + (__uint_as_float(d.z) + __uint_as_float(d.w)));
  }
}

}  // namespace rola::carry
