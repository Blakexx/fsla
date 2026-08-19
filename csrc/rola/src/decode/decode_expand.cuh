#pragma once

// THE T=1 SUPPORT EXPANSION -- `supp(R)` and `supp(W)` as EXACT leaf bitmaps, built by
// Kronecker broadcast from one side's staged amplitude rows.
//
// Three facts a reader must not miss:
//   1. LEVEL 0 IS NEVER MATERIALIZED. What is built is the INNER BLOCK -- levels
//      `1..D-1`, `S_1 = N / b_0` bits. Level 0's broadcast is applied at READ time by
//      `LeafView`, so a split CTA's fixed cost is `O(S_1/32 + b_0)`, not `O(N/32)`.
//   2. A BITMAP, NOT A CANDIDATE LIST. Nothing here bounds a support it could have
//      built: there is no odometer, no candidate list, no per-row membership test.
//   3. Membership is the EXACT-NONZERO test on the amplitude, never a threshold.
//
// The cost ledger, destination-word ownership, the ping-pong and the SMEM arm:
// docs/internals/decode/decode_expand.md

#include "decode.cuh"

namespace rola {
namespace decode {

//: THE INNER BLOCK, for both sides AT ONCE: levels `1..D-1` expanded to `S_1` bits.
//: `amp_r`/`amp_w` are the sides' flat `[sum_l b_l]` staged row sets; `buf` is
//: `4 * inner_words` words, side `i`'s ping-pong pair at `2 * i * inner_words`. It
//: returns the recursion's final PARITY, which is the same for both sides, so side `i`'s
//: map is `buf + 2*i*inner_words + parity*inner_words` and no publishing pass exists.
//: Parity rather than a pointer pair because an indexable local array of pointers is a
//: stack frame, and this kernel has none.
//:
//: THE RECURSION IS OVER LEVELS, INNERMOST FIRST: level `D-1` seeds one inner block, each
//: outer level down to level 1 broadcasts the block it has into every LIVE digit's leaf
//: range and leaves every dead digit's range zero. `D == 1` has no inner level at all, so
//: `S_1 = 1` and the block is the single bit level 0 broadcasts into every leaf.
//:
//: THE SIDES SHARE THE BARRIERS AND THE DIGIT ARITHMETIC, which is the whole reason they
//: are one loop: the destination-word decomposition depends on the LEVEL and the word,
//: never on the side, and a second pass over it would double both the barrier count and
//: the integer divisions for no new answer.
//: DESTINATION-WORD OWNERSHIP -- one thread owns one destination word and PULLS -- is
//: what makes this race-free without atomics.
//: docs/internals/decode/decode_expand.md#the-recursion
template <int D>
__device__ __forceinline__ int expand_inner(const DecodeParams& p, const float* amp_r,
                                            const float* amp_w, uint32_t* buf) {
  const int tid = threadIdx.x;
  const int iw = p.inner_words;
  const int total = 2 * iw;
  int cur = 0;

  if (D == 1) {
    for (int j = tid; j < total; j += kDecodeThreads) {
      const int side = j / iw;
      const int w = j - side * iw;
      buf[side * 2 * iw + w] = (w == 0) ? 1u : 0u;
    }
    __syncthreads();
    return 0;
  }

  // ---- the seed: level D-1's own indicator, one inner block ------------------
  {
    const int last = D - 1;
    const int width = p.level_width[last];
    const int seed_words = ceil_div_int(width, 32);
    for (int j = tid; j < total; j += kDecodeThreads) {
      const int side = j / iw;
      const int w = j - side * iw;
      uint32_t m = 0u;
      if (w < seed_words) {
        const float* row = (side == 0 ? amp_r : amp_w) + p.level_amp_offset[last];
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
      buf[side * 2 * iw + w] = m;
    }
  }
  __syncthreads();

  // ---- the broadcast levels, outward, stopping ABOVE level 0 -----------------
  for (int l = D - 2; l >= 1; --l) {
    const int span_in = p.level_span[l + 1];
    const int span_out = p.level_span[l];
    const int in_words = ceil_div_int(span_in, 32);
    const int out_words = ceil_div_int(span_out, 32);
    for (int j = tid; j < total; j += kDecodeThreads) {
      const int side = j / iw;
      const int dw = j - side * iw;
      const int pair = side * 2 * iw;
      //: HAZARD ping-pong -- docs/internals/decode/decode_expand.md#ping-pong
      const uint32_t* src = buf + pair + cur * iw;
      uint32_t* dst = buf + pair + (cur ^ 1) * iw;
      uint32_t m = 0u;
      if (dw < out_words) {
        const float* row = (side == 0 ? amp_r : amp_w) + p.level_amp_offset[l];
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

//: LEVEL 0 AS A FUNCTION rather than an array: its Kronecker broadcast of the inner block,
//: applied at the word the caller asks for.
//:
//: NOTHING IS CAPTURED IN AN OBJECT, and that is a REGISTER BUDGET decision rather than
//: taste. Every operand the broadcast needs is one the walk already holds or the parameter
//: block already carries: `inner` is the READ side's block and the write side's sits a
//: fixed `2 * inner_words` past it; the amplitude rows are the fold's own staging; the
//: geometry (`S_1`, `inner_words`, `N`) is a parameter read, which costs no register. A
//: view struct carrying its own copies spills the global-scratch arm, where `inner` is a
//: 64-bit address that survives into the walk.

//: Bitmap word starting at leaf `base`, from a leaf position the caller ALREADY holds as
//: `(digit, offset)`. Every caller in the step walks leaves in order and has that pair in
//: hand -- the walk from its own digit decomposition, `locate_row` by carrying it across
//: contiguous words -- so level 0's broadcast costs no integer division at all.
//: docs/internals/decode/decode_expand.md#the-view
__device__ __forceinline__ uint32_t view_word_at(const DecodeParams& p, const uint32_t* inner,
                                                 const float* amp, int base, int g, int i) {
  const int span = p.inner_span;
  //: `level_span[0]` IS `N`, validated at the host entry, and it is the `int` mirror.
  const int n = p.level_span[0];
  uint32_t m = 0u;
  int q = 0;
  while (q < 32 && (base + q) < n) {
    int take = span - i;
    if (take > 32 - q) take = 32 - q;
    if (base + q + take > n) take = n - (base + q);
    if (amp[g] != 0.0f) m |= extract_bits(inner, p.inner_words, i, take) << q;
    q += take;
    i += take;
    if (i >= span) {
      i = 0;
      ++g;
    }
  }
  return m;
}

//: LEVEL 0'S ROW IS THE STAGED SET'S FIRST -- `level_amp_offset[0]` is zero by
//: construction, and reading it here would be an add and a parameter fetch inside the
//: walk for a constant.
__device__ __forceinline__ uint32_t view_r_at(const DecodeParams& p, const uint32_t* inner,
                                              const float* sa_r, int base, int g, int i) {
  return view_word_at(p, inner, sa_r, base, g, i);
}

__device__ __forceinline__ uint32_t view_w_at(const DecodeParams& p, const uint32_t* inner,
                                              const float* sa_w, int base, int g, int i) {
  return view_word_at(p, inner + 2 * p.inner_words, sa_w, base, g, i);
}

//: Bitmap word `w`, exactly as a materialized level-0 expansion would hold it: bits at or
//: beyond `N` are zero, which is what lets the walk skip a leaf bound test. For callers
//: with no leaf position in hand -- the atom tests, which jump.
__device__ __forceinline__ uint32_t view_w_word(const DecodeParams& p, const uint32_t* inner,
                                                const float* sa_w, int w) {
  const int base = w << 5;
  const int g = base / p.inner_span;
  return view_w_at(p, inner, sa_w, base, g, base - g * p.inner_span);
}

}  // namespace decode
}  // namespace rola
