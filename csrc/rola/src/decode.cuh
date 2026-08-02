#pragma once

// THE T=1 DECODE PATH -- parameter block and the objects both decode TUs share.
//
// THUNDER_RUN_SPEC.md §6b ruled that T = 1 HAS NO STAGE 2: every stage-2 decision is
// degenerate (BT: one token, no tiling; P: nothing to split; packing: K is one
// token's support and the masks subsume it; statistics: a population of one), so the
// decode path bypasses selection ARCHITECTURALLY and its config freezes at the
// prefill->decode boundary. What is left per step is data plane only, and its shape
// is the one §6b ratified: a TOKEN-STATIONARY, STATE-STREAMING gather-GEMV.
//
// **THE STRUCTURE, AND THE ONE FACT IT RESTS ON.** At T = 1 a leaf `s` is in the
// token's read support iff EVERY level's digit `d_l(s)` has a nonzero read
// amplitude, so
//
//     supp(R) = prod_l supp_l(R)                      (a Cartesian product)
//
// and the support's BITMAP is therefore a Kronecker product of the per-level
// indicator vectors. That makes the exact support an `N`-bit object that EXPANDS BY
// BROADCAST-COPY -- 8 KiB at `N = 65,536`, 128 KiB at `N = 1,048,576`, against a
// state stream of 130 MiB and 2 GiB. Materializing the truth is cheaper than
// bounding it, which is why there is no candidate list, no over-enumeration and no
// per-row membership test anywhere below: `rola_decode_expand` builds `bits_r` and
// `bits_w` exactly, and `rola_decode_gemv` `__ffs`-walks them, touching only rows
// that ARE in the support.
//
// **BC IS ABSENT AND THAT IS THE POINT (spec I1).** Owner grouping is a property of
// how the TILED consumer amortizes a `[BC][cols]` block across many tokens; with one
// token there is nothing to amortize. Nothing in this header, in `decode_expand.cu`
// or in `decode.cu` is an expression in `BC`: the support is leaf-granular, the work
// partition is over 32-leaf bitmap WORDS (a machine quantity, not a plan quantity),
// and the unpaged state address is `(bh * N + leaf) * cols + c`, which is leaf-major
// already -- owner `o`'s leaf `i` sits at row `o * BC + i = leaf` in the tiled
// kernel's own layout. `BC` is a launch-shape fact of that kernel and never a layout
// fact.
//
// **UNPAGED ONLY, DELIBERATELY.** The paged address under §4.5's re-key is
// `((slot << 4 | (leaf & 15)) * cols + c)` and is likewise BC-free, but TODAY
// `page_states := BC` (`rola/ops/paging.py`, and `consumer.cuh`'s "one page IS one
// owner"), so a paged decode address would have to know `BC`. The re-key is a
// merge-phase paired change; until it lands, decode REFUSES a page table rather than
// computing a BC-keyed address. That refusal is a hard `TORCH_CHECK` in
// `rola/ops/decode.py` and here, never a silent fallback.

#include <cstdint>

namespace rola {
namespace decode {

//: Spec §1: `1 <= D <= 4`. Same bound the consumer's `kMaxLevels` states, restated
//: here rather than included from `consumer.cuh` because the decode TUs share no
//: other object with the tiled consumer and a one-way include would imply they do.
constexpr int kMaxLevels = 4;

//: THE DECODE PATH'S LEVEL-WIDTH CAPACITY, and it is a GATE rather than a
//: convenience. `rola/routing/topology.py`'s `MAX_BRANCH_WIDTH = 256` is the widest
//: level the PRODUCER will emit; a decode path whose capacity were narrower would
//: accept a topology the producer can build and then read past the end of a staged
//: row. The `static_assert` below is the compile-time half of that gate; the runtime
//: half is `rola_decode_capacity()`, bound into the extension so
//: `tests/unit/test_decode_capacity.py` can assert the two constants agree -- a C++
//: assertion cannot read a Python constant, so the pair is what makes this checkable
//: rather than merely written down.
constexpr int kProducerMaxBranchWidth = 256;
constexpr int kDecodeMaxLevelWidth = 256;
static_assert(kDecodeMaxLevelWidth >= kProducerMaxBranchWidth,
              "the decode path's level-width capacity is below the producer's "
              "MAX_BRANCH_WIDTH: a topology the producer can emit would over-read "
              "the staged amplitude row");

//: Both decode kernels launch 256 threads. The number is not free-floating: it makes
//: `d_v = 64` exactly two columns per lane with 128 B coalesced row loads, it matches
//: the tiled consumer's `kConsumerThreads` so the two agree on a block size, and 6
//: CTAs of 256 fill sm_86's 1536-thread limit exactly.
constexpr int kDecodeThreads = 256;
constexpr int kDecodeWarps = kDecodeThreads / 32;

//: Bitmap words per COARSE GROUP. `pc_coarse` holds one exclusive-prefix entry per
//: group plus a total, so a CTA resolves its row range with one binary search over
//: `ceil(n_words / 32)` entries and one 32-word scan instead of a pass over the
//: bitmap. The same array is consulted DURING the walk to skip wholly-empty groups
//: (`pc_coarse[g + 1] == pc_coarse[g]`), which is what keeps the bitmap read from
//: dominating a high-batch high-sparsity step.
constexpr int kCoarseGroupWords = 32;

//: Where level `l`'s staged amplitude row starts inside the flat `sum_l b_l` SMEM
//: arrays. Host-computed, carried rather than recomputed, because the kernel needs it
//: on the innermost path.
struct DecodeParams {
  //: Per-level amplitude rows, `[BH, b_l]` CONTIGUOUS -- the `T = 1` degeneracy of
  //: the tiled consumer's `[BH, n_tiles, BT, pitch]` streams with `n_tiles = BT = 1`.
  //: There is no pitch: the decode op folds `[B, 1, H, b_l] -> [BH, b_l]` and the
  //: fold IS the contiguity.
  const float* amp_read[kMaxLevels];
  const float* amp_write[kMaxLevels];
  //: The DETACHED clock stream (`build_clock_stream`'s stored write amplitude);
  //: nullptr under `decay = None`. Its "stored != 0 => live != 0" guarantee is what
  //: makes §2.4's no-sweep claim exact: a leaf outside `supp(W)` has a zero factor at
  //: some level, so its clock is 0, so `keep^0 == 1` EXACTLY and its stored state is
  //: unchanged. Decay fires only on written rows, and written rows are streamed
  //: anyway -- there is no entry sweep and no exit sweep in this kernel.
  const float* amp_clock[kMaxLevels];

  int level_width[kMaxLevels];       // b_l
  int level_stride[kMaxLevels];      // leaf_stride[l] = prod(b_{l+1..D-1})
  int level_span[kMaxLevels];        // S_l = prod(b_{l..D-1});  S_0 = N
  int level_amp_offset[kMaxLevels];  // into the flat [sum_l b_l] SMEM staging
  int level_row_offset[kMaxLevels];  // into the flattened per-head dial array

  const float* dials;    // [H, total_rows] or nullptr
  const float* g_read;   // [BH] or nullptr (global fixes g_read := 1)
  const float* g_write;  // [BH]
  const float* v;        // [BH, d_v]

  //: `[BH, N, cols]`, `cols = d_v + is_global`. Read AND written in place: a row's
  //: read contribution and its write-back are performed by the SAME lane in program
  //: order, so the read-then-write ordering the tiled kernel spends a barrier on per
  //: tile costs one register here.
  float* state;
  float* y;  // [BH, d_v] -- the FUSED ratio readout, num/(den + eps)

  //: §1.1's exact support. `bits_r | bits_w` is formed word-by-word in a REGISTER at
  //: walk time and never stored: it is one instruction and one third of the storage.
  uint32_t* bits_r;    // [BH, n_words]
  uint32_t* bits_w;    // [BH, n_words]
  uint32_t* scratch;   // [BH, n_words] -- the expansion's ping-pong partner
  int32_t* pc_coarse;  // [BH, n_groups + 1] exclusive prefix, entry n_groups = M

  //: Split-K workspace. Slot-ordered and summed by the last-arriving CTA, which is
  //: what makes decode's `y` RUN-REPRODUCIBLE -- see `decode.cu`'s combine block for
  //: why that diverges from the tiled consumer on purpose.
  float* ws;      // [BH, n_split, cols]
  int32_t* ctr;   // [BH]

  int64_t N;
  int BH;
  int H;
  int D;
  int n_words;
  int n_groups;
  int n_split;
  int total_rows;  // sum_l b_l
  float eps;
};

//: `ceil(n / d)` for non-negative ints, used on host and device.
__host__ __device__ __forceinline__ int ceil_div_int(int n, int d) {
  return (n + d - 1) / d;
}

//: EXTRACT `count` bits of a packed bitmap starting at BIT index `start`, returned in
//: the low `count` bits (`count <= 32`). This is the shift-and-merge the Kronecker
//: copy is built from, and it is written once because it is the one place a funnel
//: shift can be off by one.
//:
//: The `(start & 31) == 0` fork is NOT an aligned fast path bolted onto a general
//: one -- it is the C++ shift operator's undefined behaviour at a shift width of 32
//: being avoided at the one site that can reach it. Both arms compute the same
//: function; the standing "no branch to dodge edge cases" rule is satisfied because
//: there is ONE strategy here (gather 32 bits from at most two source words) and the
//: fork is inside it.
__device__ __forceinline__ uint32_t extract_bits(const uint32_t* src, int src_words,
                                                 int start, int count) {
  const int w = start >> 5;
  const int sh = start & 31;
  const uint32_t lo = (w < src_words) ? src[w] : 0u;
  const uint32_t hi = ((w + 1) < src_words) ? src[w + 1] : 0u;
  const uint32_t v = (sh == 0) ? lo : ((lo >> sh) | (hi << (32 - sh)));
  return (count >= 32) ? v : (v & ((1u << count) - 1u));
}

//: Drop the LOWEST `k` set bits of `word` (`k <= 32`). The CTA's first bitmap word is
//: shared with the previous CTA's row range, and this is how the boundary is cut:
//: `word & (word - 1)` clears the lowest set bit, so `k` applications drop exactly
//: the `k` rows that belong to the CTA before this one. Exact by construction rather
//: than by an index comparison, which is what lets the row partition be defined in
//: ROW counts while the work is handed out in WORDS.
__device__ __forceinline__ uint32_t drop_low_set_bits(uint32_t word, int k) {
  for (int i = 0; i < k && word != 0u; ++i) word &= word - 1u;
  return word;
}

//: Keep only the LOWEST `k` set bits of `word`. The mirror of `drop_low_set_bits`, for
//: the CTA's last word. Applied BEFORE the drop, so that when a CTA's whole row range
//: lies inside one word the two together leave exactly bits `[drop, keep)`.
__device__ __forceinline__ uint32_t keep_low_set_bits(uint32_t word, int k) {
  uint32_t out = 0u;
  uint32_t rest = word;
  for (int i = 0; i < k && rest != 0u; ++i) {
    const uint32_t low = rest & (~rest + 1u);
    out |= low;
    rest &= rest - 1u;
  }
  return out;
}

// ---------------------------------------------------------------------------
// The two TUs' entry points. `decode_expand.cu` owns the first group,
// `decode.cu` the second; `rola_api.cpp` binds only the last two.
// ---------------------------------------------------------------------------

size_t decode_expand_smem_bytes(int n_words);
bool decode_expand_uses_smem(int n_words, int64_t smem_budget);
void launch_decode_expand(const DecodeParams& p, int64_t smem_budget);

size_t decode_gemv_smem_bytes(int total_rows, int d_v, int cols, bool decay);
void launch_decode_gemv(const DecodeParams& p, int d_v, bool global_norm, bool decay);

}  // namespace decode
}  // namespace rola
