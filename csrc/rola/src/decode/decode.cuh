#pragma once

// THE T=1 DECODE PATH -- the parameter block and the objects both decode TUs share.
//
// T = 1 HAS NO STAGE 2: every stage-2 decision is degenerate, so this path bypasses
// selection ARCHITECTURALLY and its config freezes at the prefill->decode boundary.
// What is left per step is data plane only, in the ratified shape: a
// TOKEN-STATIONARY, STATE-STREAMING gather-GEMV.
//
// Three facts a reader must not miss:
//   1. THE LEAF ORDER IS THE LATTICE. A leaf is named by its owner, its sub-box and its
//      intra-sub-box digits, and the step reaches it by WALKING those factors -- there is
//      no leaf map, no bitmap over the leaf space, and no term in `N` in the step's
//      fixed cost.
//   2. supp(R) = prod_l supp_l(R) at T = 1, so the touched set is a PRODUCT at every
//      lattice granularity and the walk prunes at each one.
//   3. PAGED AND DENSE ARE ONE WALK. The page granule is the ATOM (16 leaves), which is
//      what keeps the address BC-free; a null page table means the slot IS the atom and
//      the bytes are the dense path's, exactly.
//
// The lattice, the walk's cost accounting and the paged base translation:
// docs/internals/decode/decode.md

#include <cstdint>

namespace rola {
namespace decode {

//: `1 <= D <= 4`. The same bound the consumer's `kMaxLevels` states, restated
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

//: The decode step launches 256 threads. The number is pinned by three facts at once
//: (lane-to-column mapping, the tiled consumer's block size, sm_86 occupancy) and is
//: not free-floating: docs/internals/decode/decode.md#decode-threads
constexpr int kDecodeThreads = 256;
constexpr int kDecodeWarps = kDecodeThreads / 32;

//: THE LATTICE'S DECLARED CAPACITIES, each refused loudly at the host entry: an owner's
//: per-level span and the capacity multiplier are read as SINGLE 32-bit slices, and the
//: intra-sub-box scan is a fixed cost per live sub-box.
//: docs/internals/decode/decode_lattice.md#capacities
constexpr int kDecodeMaxSpan = 32;
constexpr int kDecodeMaxCap = 32;
constexpr int kDecodeMaxSubBox = 64;

//: THE DECLARED RESIDENCY of the step kernel, in CTAs per SM, and it is a BUDGET rather
//: than a hope: `__launch_bounds__` turns it into a register cap the assembler must meet,
//: and the census is what proves it met it without spilling. The walk is a latency-bound
//: stream over state rows, so residency is the one thing it trades registers FOR. TWO
//: TEMPLATE AXES MOVE IT, each by one CTA: the DECAY arm carries a second per-leaf
//: product and a survival exponential through the same loop, and the DEEPEST arm carries
//: three more live registers per level (the staging offset and both sides' intra-sub-box
//: slices) through the same one. docs/internals/decode/decode.md#declared-residency
__host__ __device__ constexpr int decode_step_blocks_per_sm(bool decay, int levels) {
  return (decay || levels >= kMaxLevels) ? 4 : 5;
}

//: THE PAGE GRANULE, `page_states := MMA_K_QUANTUM`, mirrored here because
//: the decode TUs share no header with the tiled consumer. It is geometry-owned and
//: fixed across every config forever, which is what makes the paged address BC-free.
//: docs/internals/decode/decode.md#the-atom
constexpr int kAtomLeaves = 16;
constexpr int kAtomShift = 4;
constexpr int kAtomMask = kAtomLeaves - 1;
static_assert(1 << kAtomShift == kAtomLeaves, "kAtomShift must be log2(kAtomLeaves)");

//: ONE OPERAND'S ADDRESS, as the producer left it: the base and the three strides the
//: fold walks over a `[B, 1, H, W]` view. `sw` is `0` for the per-token scalars, which
//: have no last axis. docs/internals/decode/decode_fold.md#the-identity
struct Span {
  const void* base;
  int64_t sb;
  int64_t sh;
  int64_t sw;
};

//: EVERY POINTER AND EXTENT ONE DECODE STEP NEEDS, and nothing the kernel could cheaply
//: recompute. Field by field: docs/internals/decode/decode.md#params-amps
struct DecodeParams {
  //: THE RAW PER-TOKEN OPERANDS, ADDRESSED RATHER THAN COPIED: both sides' routing
  //: levels, the write gain and the token's value, each as the caller's own view. There
  //: is no `[BH, b_l]` plane between the producer and the walk -- the step folds these
  //: into shared memory itself: docs/internals/decode/decode_fold.md
  Span read[kMaxLevels];
  Span write[kMaxLevels];
  Span g_write;
  Span v;
  //: Per READ level: does it still carry its own mass, so the fold divides it by its row
  //: sum? The write side is never normalized.
  int normalize[kMaxLevels];
  //: The storage type of each operand family, `1` = bfloat16 and `0` = float. Branching
  //: on STORAGE selects a typed gather loop once per side, never a test per element:
  //: docs/internals/decode/decode_fold.md#the-dtype-arms
  int levels_bf16;
  int g_bf16;
  int v_bf16;

  int level_width[kMaxLevels];       // B_l
  int level_amp_offset[kMaxLevels];  // into the flat [sum_l b_l] SMEM staging
  int level_row_offset[kMaxLevels];  // into the flattened per-head dial array

  //: THE (k, m) LATTICE, the R0 spec's LATTICE-1 arithmetic in runtime scalars:
  //: `lat_s[l] = k * lat_m_l[l]` is an owner's span at level `l`, `lat_g[l] = B_l /
  //: lat_s[l]` its owner-grid extent, and a leaf's index is `(O*lat_m + U)*lat_sub + C`
  //: over the three mixed-radix numbers `O`, `U`, `C`. The suffix products are held
  //: rather than recomputed because the walk forms them once per unit.
  //: docs/internals/decode/decode_lattice.md
  int lat_k;
  int lat_m;
  int lat_msb;  // log2(lat_m)
  int lat_sub;  // k^D, the leaves one sub-box holds
  int lat_bc;   // lat_m * lat_sub, the leaves one owner holds
  int lat_m_l[kMaxLevels];
  int lat_s[kMaxLevels];
  int lat_g[kMaxLevels];
  int lat_gsuf[kMaxLevels];    // prod_{l' > l} lat_g[l']
  int lat_msuf[kMaxLevels];    // prod_{l' > l} lat_m_l[l']
  int lat_cshift[kMaxLevels];  // log2(k) * (D - 1 - l)
  //: The digit masks' word layout: level `l` owns `mask_words[l]` words at `mask_off[l]`
  //: of a `mask_total`-word plane, one plane per side.
  int mask_off[kMaxLevels];
  int mask_words[kMaxLevels];
  int mask_total;
  //: The union owner-coordinate lists' layout: `lat_g[l]` entries reserved per level.
  int olist_off[kMaxLevels];
  int olist_total;
  //: Atoms per sub-box, `lat_sub / kAtomLeaves`. PAGED BACKING ONLY -- the host refuses a
  //: page table under a lattice whose sub-box is thinner than an atom, because then an
  //: atom would straddle two sub-boxes and the claim's enumeration would not be an
  //: ascending partition: docs/internals/decode/decode.md#candidate-atoms
  int atoms_per_sub;

  const float* dials;  // [H, total_rows] or nullptr

  //: DENSE `[BH, N, cols]` or PAGED `[slots, kAtomLeaves, cols]`, `cols = d_v + 1`.
  //: Read AND written in place by the SAME lane in program order:
  //: docs/internals/decode/decode.md#state-in-place
  float* state;
  //: `[BH, atoms_per_bh]` int32, `-1` = ABSENT; `nullptr` = the dense backing, where the
  //: slot IS the atom. ONE base translation for both:
  //: docs/internals/decode/decode.md#paged-address
  //: MUTABLE because a pool claim folds into it, and only ever at the point where the
  //: batch-head's last CTA has arrived: docs/internals/decode/decode.md#the-claim
  int32_t* page_tbl;
  //: `[BH]` int32, THE PER-BATCH-HEAD VERDICT: `1` where this step writes an atom no
  //: table entry and no pool slot can address, `0` where it advanced. Written by the
  //: batch-head's last-arriving CTA: docs/internals/decode/decode.md#per-bh-verdict
  int32_t* growth;
  //: `[1]` int32, the OR of `growth` over batch-heads, and the `[1]` int32
  //: self-resetting arrival counter of the CTA that reduces it:
  //: docs/internals/decode/decode.md#growth-reduction
  int32_t* growth_any;
  int32_t* growth_ctr;
  //: `[BH]` int32, THE DONE FLAGS -- `1` where a batch-head has already deposited this
  //: step. A replay re-walks a done batch-head for its `y` and deposits NOTHING, which
  //: is what makes "exactly once per sequence" hold across the admission the replay
  //: exists for: docs/internals/decode/decode.md#done-flags
  int32_t* done;
  //: `[BH, atoms_per_bh]` bool, this step's own write set condensed to atoms. It is what
  //: a growth step's `plan_exact` admits from, so the host never re-derives the set the
  //: step has already expanded: docs/internals/decode/decode.md#atom-bits
  bool* atom_bits;

  //: THE SLACK POOL -- physical slots pre-committed and pre-zeroed by the host, from
  //: which a growth step admits ITSELF. `pool_slots` is `[BH, pool_cap]` slot ids,
  //: `pool_cursor` `[BH]` how many of each batch-head's are spent, `pool_map`
  //: `[BH, atoms_per_bh]` the claim the walk addresses through before the fold.
  //: `pool_cap = 0` (and three null pointers) is EXACT-COMMIT: every growth step goes to
  //: the host, which is the shipped default: docs/internals/decode/decode.md#the-claim
  const int32_t* pool_slots;
  int32_t* pool_cursor;
  int32_t* pool_map;
  int pool_cap;
  float* y;  // [BH, d_v] -- the FUSED ratio readout, num/(den + eps)

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
  int n_split;
  int total_rows;  // sum_l b_l
  float eps;
};

//: `ceil(n / d)` for non-negative ints, used on host and device.
__host__ __device__ __forceinline__ int ceil_div_int(int n, int d) { return (n + d - 1) / d; }

//: EXTRACT `count` bits of a packed bitmap starting at BIT index `start`, returned in
//: the low `count` bits (`count <= 32`) -- the shift-and-merge every lattice slice is
//: read through, written once because it is the one place a funnel shift can be off by
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

// ---------------------------------------------------------------------------
// The step's entry points. `decode.cu` owns the step and the host entry;
// `rola_api.cpp` binds only the host entry.
// ---------------------------------------------------------------------------

size_t decode_step_smem_bytes(int total_rows, int d_v, int cols, bool decay, int mask_total,
                              int olist_total);
void launch_decode_step(const DecodeParams& p, int d_v, bool decay, int64_t smem_budget);

}  // namespace decode
}  // namespace rola
