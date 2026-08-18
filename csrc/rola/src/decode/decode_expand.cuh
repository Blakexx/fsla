#pragma once

// THE T=1 SUPPORT EXPANSION -- `supp(R)` and `supp(W)` as EXACT leaf bitmaps, built by
// Kronecker broadcast from one side's staged amplitude rows.
//
// Three facts a reader must not miss:
//   1. A BITMAP, NOT A CANDIDATE LIST. Nothing here bounds a support it could have
//      built: there is no odometer, no candidate list, no per-row membership test.
//   2. Publishing an N-bit map costs ceil(N/32) words per side REGARDLESS OF SPARSITY,
//      because dead ranges must be written as zeros too. That is a FLOOR.
//   3. Membership is the EXACT-NONZERO test on the amplitude, never a threshold.
//
// The cost ledger, destination-word ownership, the ping-pong and the SMEM arm:
// docs/internals/decode/decode_expand.md

#include "decode.cuh"

namespace rola {
namespace decode {

//: THE EXPANSION, for `N_SIDES` sides AT ONCE. `amp0`/`amp1` are the sides' flat
//: `[sum_l b_l]` staged row sets; `buf` is `2 * N_SIDES * n_words` words, side `i`'s
//: ping-pong pair at `2 * i * n_words`. It returns the recursion's final PARITY, which is
//: the same for every side, so side `i`'s published map is `buf + 2*i*n_words + parity*
//: n_words` and no publishing pass exists. Parity rather than a pointer pair because an
//: indexable local array of pointers is a stack frame, and this kernel has none.
//:
//: THE RECURSION IS OVER LEVELS, INNERMOST FIRST: level `D-1` seeds one inner block, each
//: outer level broadcasts the block it has into every LIVE digit's leaf range and leaves
//: every dead digit's range zero. After level 0 the buffer holds `S_0 = N` bits.
//:
//: THE SIDES SHARE THE BARRIERS AND THE DIGIT ARITHMETIC, which is the whole reason they
//: are one loop: the destination-word decomposition depends on the LEVEL and the word,
//: never on the side, and a second pass over it would double both the barrier count and
//: the integer divisions for no new answer.
//: DESTINATION-WORD OWNERSHIP -- one thread owns one destination word and PULLS -- is
//: what makes this race-free without atomics.
//: docs/internals/decode/decode_expand.md#the-recursion
template <int N_SIDES>
__device__ __forceinline__ int expand_sides(const DecodeParams& p, const float* amp0,
                                            const float* amp1, uint32_t* buf) {
  const int tid = threadIdx.x;
  const int nw = p.n_words;
  const int D = p.D;
  const int total = N_SIDES * nw;
  int cur = 0;

  // ---- the seed: level D-1's own indicator, one inner block ------------------
  {
    const int last = D - 1;
    const int width = p.level_width[last];
    const int seed_words = ceil_div_int(width, 32);
    for (int j = tid; j < total; j += kDecodeThreads) {
      const int side = (N_SIDES == 1) ? 0 : j / nw;
      const int w = j - side * nw;
      uint32_t m = 0u;
      if (w < seed_words) {
        const float* row = (side == 0 ? amp0 : amp1) + p.level_amp_offset[last];
        const int g0 = w * 32;
        for (int b = 0; b < 32; ++b) {
          const int g = g0 + b;
          if (g >= width) break;
          //: MEMBERSHIP IS THE EXACT-NONZERO TEST -- the DEFINING EXPRESSION of
          //: `schedule.py`'s support bitmap, never a threshold:
          //: docs/internals/decode/decode_expand.md#the-recursion
          if (row[g] != 0.0f) m |= 1u << b;
        }
      }
      buf[side * 2 * nw + w] = m;
    }
  }
  __syncthreads();

  // ---- the broadcast levels, outward ----------------------------------------
  for (int l = D - 2; l >= 0; --l) {
    const int span_in = p.level_span[l + 1];
    const int span_out = p.level_span[l];
    const int in_words = ceil_div_int(span_in, 32);
    const int out_words = ceil_div_int(span_out, 32);
    for (int j = tid; j < total; j += kDecodeThreads) {
      const int side = (N_SIDES == 1) ? 0 : j / nw;
      const int dw = j - side * nw;
      const int pair = side * 2 * nw;
      //: HAZARD ping-pong -- docs/internals/decode/decode_expand.md#ping-pong
      const uint32_t* src = buf + pair + cur * nw;
      uint32_t* dst = buf + pair + (cur ^ 1) * nw;
      uint32_t m = 0u;
      if (dw < out_words) {
        const float* row = (side == 0 ? amp0 : amp1) + p.level_amp_offset[l];
        const int base = dw * 32;
        int q = 0;
        while (q < 32 && (base + q) < span_out) {
          const int pbit = base + q;
          const int g = pbit / span_in;
          const int i = pbit - g * span_in;
          int take = span_in - i;
          if (take > 32 - q) take = 32 - q;
          if (pbit + take > span_out) take = span_out - pbit;
          if (row[g] != 0.0f) m |= extract_bits(src, in_words, i, take) << q;
          q += take;
        }
      }
      dst[dw] = m;
    }
    __syncthreads();
    cur ^= 1;
  }

  return cur;
}

}  // namespace decode
}  // namespace rola
