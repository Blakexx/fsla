#pragma once

// THE T=1 DECODE PATH -- the parameter block and the objects both decode TUs share.
//
// THUNDER_RUN_SPEC.md §6b ruled that T = 1 HAS NO STAGE 2: every stage-2 decision is
// degenerate, so this path bypasses selection ARCHITECTURALLY and its config freezes at
// the prefill->decode boundary. What is left per step is data plane only, in the shape
// §6b ratified: a TOKEN-STATIONARY, STATE-STREAMING gather-GEMV.
//
// Three facts a reader must not miss:
//   1. supp(R) = prod_l supp_l(R) at T = 1, so the exact support is an N-bit Kronecker
//      object and it is MATERIALIZED, not bounded: no candidate list, no
//      over-enumeration, no per-row membership test anywhere on this path.
//   2. BC IS ABSENT AND THAT IS THE POINT (spec I1). Nothing here, in decode_expand.cu
//      or in decode.cu is an expression in BC; work is partitioned over bitmap WORDS.
//   3. PAGED AND DENSE ARE ONE WALK. The page granule is the ATOM (16 leaves), which is
//      what keeps the address BC-free; a null page table means the slot IS the atom and
//      the bytes are the dense path's, exactly.
//
// The §6b degeneracy table, the Kronecker size ledger, the BC argument and the paged
// base translation: docs/internals/decode/decode.md

#include <cstdint>

namespace rola {
namespace decode {

//: Spec §1: `1 <= D <= 4`. The same bound the consumer's `kMaxLevels` states, restated
//: rather than included -- the decode TUs share no other object with the tiled
//: consumer. docs/internals/decode/decode.md#max-levels
constexpr int kMaxLevels = 4;

//: THE LEVEL-WIDTH CAPACITY, a GATE and not a convenience: below the producer's
//: `MAX_BRANCH_WIDTH` it would accept a topology the producer can emit and then
//: over-read a staged row. Compile-time half here, runtime half in a bound accessor:
//: docs/internals/decode/decode.md#level-width-capacity
constexpr int kProducerMaxBranchWidth = 256;
constexpr int kDecodeMaxLevelWidth = 256;
static_assert(kDecodeMaxLevelWidth >= kProducerMaxBranchWidth,
              "the decode path's level-width capacity is below the producer's "
              "MAX_BRANCH_WIDTH: a topology the producer can emit would over-read "
              "the staged amplitude row");

//: Both decode kernels launch 256 threads. The number is pinned by three facts at once
//: (lane-to-column mapping, the tiled consumer's block size, sm_86 occupancy) and is
//: not free-floating: docs/internals/decode/decode.md#decode-threads
constexpr int kDecodeThreads = 256;
constexpr int kDecodeWarps = kDecodeThreads / 32;

//: Bitmap words per COARSE GROUP. `pc_coarse` is one exclusive-prefix entry per group
//: plus a total: a CTA resolves its row range without a pass over the bitmap, and the
//: walk skips empty groups with it. docs/internals/decode/decode.md#coarse-group
constexpr int kCoarseGroupWords = 32;

//: THE PAGE GRANULE, spec §4.5's `page_states := MMA_K_QUANTUM`, mirrored here because
//: the decode TUs share no header with the tiled consumer. It is geometry-owned and
//: fixed across every config forever, which is what makes the paged address BC-free.
//: docs/internals/decode/decode.md#the-atom
constexpr int kAtomLeaves = 16;
constexpr int kAtomShift = 4;
constexpr int kAtomMask = kAtomLeaves - 1;
static_assert(1 << kAtomShift == kAtomLeaves, "kAtomShift must be log2(kAtomLeaves)");

//: EVERY POINTER AND EXTENT ONE DECODE STEP NEEDS, and nothing the kernel could cheaply
//: recompute. Field by field: docs/internals/decode/decode.md#params-amps
struct DecodeParams {
  //: `[BH, b_l]` CONTIGUOUS -- the `T = 1` degeneracy of the tiled consumer's
  //: `[BH, n_tiles, BT, pitch]` streams. There is no pitch; the fold IS the contiguity.
  const float* amp_read[kMaxLevels];
  const float* amp_write[kMaxLevels];
  //: The DETACHED clock stream; nullptr under `decay = None`. Its
  //: "stored != 0 => live != 0" guarantee is why this kernel has NO entry sweep and NO
  //: exit sweep: docs/internals/decode/decode.md#detached-clock
  const float* amp_clock[kMaxLevels];

  int level_width[kMaxLevels];       // b_l
  int level_stride[kMaxLevels];      // leaf_stride[l] = prod(b_{l+1..D-1})
  int level_span[kMaxLevels];        // S_l = prod(b_{l..D-1});  S_0 = N
  int level_amp_offset[kMaxLevels];  // into the flat [sum_l b_l] SMEM staging
  int level_row_offset[kMaxLevels];  // into the flattened per-head dial array

  const float* dials;    // [H, total_rows] or nullptr
  const float* g_write;  // [BH]
  const float* v;        // [BH, d_v]

  //: DENSE `[BH, N, cols]` or PAGED `[slots, kAtomLeaves, cols]`, `cols = d_v + is_global`.
  //: Read AND written in place by the SAME lane in program order:
  //: docs/internals/decode/decode.md#state-in-place
  float* state;
  //: `[BH, atoms_per_bh]` int32, `-1` = ABSENT; `nullptr` = the dense backing, where the
  //: slot IS the atom. ONE base translation for both: docs/internals/decode/decode.md#paged-address
  const int32_t* page_tbl;
  //: `[BH]` int32, the GROWTH PROBE's verdict: `1` where this batch-head writes an
  //: atom the table does not map, `0` where it does not. Every CTA writes its own entry
  //: unconditionally, so the buffer needs no zero-fill and the probe needs no atomic:
  //: docs/internals/decode/decode_expand.md#growth-probe
  int32_t* growth;
  //: `[BH, atoms_per_bh]` bool, the GROWTH PROBE's second output: this step's own write
  //: set, condensed to atoms. It is what a growth step's `plan_exact` admits from, so the
  //: host never re-derives the set the probe has already expanded:
  //: docs/internals/decode/decode_expand.md#growth-probe
  bool* atom_bits;
  float* y;  // [BH, d_v] -- the FUSED ratio readout, num/(den + eps)

  //: §1.1's exact support. `bits_r | bits_w` is formed word-by-word in a REGISTER at
  //: walk time and never stored: docs/internals/decode/decode.md#bitmap-fields
  uint32_t* bits_r;    // [BH, n_words]
  uint32_t* bits_w;    // [BH, n_words]
  uint32_t* scratch;   // [BH, 2 * n_words] -- the expansion's ping-pong partner
  int32_t* pc_coarse;  // [BH, n_groups + 1] exclusive prefix, entry n_groups = M

  //: Split-K workspace. Slot-ordered and summed by the last-arriving CTA, which is what
  //: makes decode's `y` RUN-REPRODUCIBLE -- and that diverges from the tiled consumer
  //: on purpose: docs/internals/decode/decode.md#determinism
  float* ws;     // [BH, n_split, cols]
  int32_t* ctr;  // [BH]

  int64_t N;
  int atoms_per_bh;  // N / kAtomLeaves; the page table's row pitch. PAGED backing only.
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
__host__ __device__ __forceinline__ int ceil_div_int(int n, int d) { return (n + d - 1) / d; }

//: EXTRACT `count` bits of a packed bitmap starting at BIT index `start`, returned in
//: the low `count` bits (`count <= 32`) -- the shift-and-merge the Kronecker copy is
//: built from, written once because it is the one place a funnel shift can be off by
//: one. docs/internals/decode/decode.md#extract-bits
__device__ __forceinline__ uint32_t extract_bits(const uint32_t* src, int src_words, int start,
                                                 int count) {
  const int w = start >> 5;
  const int sh = start & 31;
  const uint32_t lo = (w < src_words) ? src[w] : 0u;
  const uint32_t hi = ((w + 1) < src_words) ? src[w + 1] : 0u;
  //: HAZARD shift-32-ub -- docs/internals/decode/decode.md#shift-32-ub
  const uint32_t v = (sh == 0) ? lo : ((lo >> sh) | (hi << (32 - sh)));
  return (count >= 32) ? v : (v & ((1u << count) - 1u));
}

//: Drop the LOWEST `k` set bits of `word` (`k <= 32`) -- how the CTA's FIRST bitmap word
//: is cut where it is shared with the previous CTA's range. Exact by construction, which
//: is what lets the partition be defined in ROWS while work is handed out in WORDS.
__device__ __forceinline__ uint32_t drop_low_set_bits(uint32_t word, int k) {
  for (int i = 0; i < k && word != 0u; ++i) word &= word - 1u;
  return word;
}

//: Keep only the LOWEST `k` set bits of `word` -- the mirror, for the CTA's LAST word.
//: The order in which the two are applied is load-bearing:
//: docs/internals/decode/decode.md#boundary-cut-order
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
void launch_decode_growth_probe(const DecodeParams& p, int64_t smem_budget);

size_t decode_gemv_smem_bytes(int total_rows, int d_v, int cols, bool decay);
void launch_decode_gemv(const DecodeParams& p, int d_v, bool decay);

}  // namespace decode
}  // namespace rola
