#pragma once

// THE T=1 DECODE PATH -- the parameter block and the objects both decode TUs share.
//
// T = 1 HAS NO STAGE 2: every stage-2 decision is degenerate, so this path bypasses
// selection ARCHITECTURALLY and its config freezes at the prefill->decode boundary.
// What is left per step is data plane only, in the ratified shape: a
// TOKEN-STATIONARY, STATE-STREAMING gather-GEMV.
//
// Three facts a reader must not miss:
//   1. THE LEAF ORDER IS THE LATTICE, in the ONE order `carry/box.cuh` defines: a leaf is
//      named by its owner and by its per-level RUN OFFSETS in mixed radix, and the step
//      reaches it by WALKING those factors -- there is no leaf map, no bitmap over the
//      leaf space, and no term in `N` in the step's fixed cost.
//   2. supp(R) = prod_l supp_l(R) at T = 1, so the touched set is a PRODUCT at every
//      lattice granularity and the walk prunes at each one.
//   3. PAGED AND DENSE ARE ONE WALK. The page granule is the ATOM (16 leaves), which is
//      what keeps the address BC-free; a null page table means the slot IS the atom and
//      the bytes are the dense path's, exactly.
//
// The lattice, the walk's cost accounting and the paged base translation:
// docs/internals/decode/decode.md

#include <cstdint>

#include "../carry/box.cuh"

namespace rola {
namespace decode {

//: THE PAGE GRANULE AND THE OWNER'S SHAPE COME FROM THE BOX, not from a mirror of it:
//: `carry/box.cuh` is where the lattice's addressing constants and its constexpr
//: `BoxPlan` arithmetic live, and the two families write and read ONE plane in ONE order.
//: What this path adds is the RUNTIME carrier below -- decode takes `(k, m)` and the
//: per-level widths as data rather than as template arms, so it holds `BoxPlan`'s fields
//: as scalars under `BoxPlan`'s own names and derives them by `BoxPlan`'s own formulas.
using rola::carry::kAtomLeaves;

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

//: THE LATTICE'S ONE DECLARED CAPACITY, refused loudly at the host entry: an owner's
//: per-level span is read as a SINGLE 32-bit slice, both when a level's owner grid is
//: built and when a unit's innermost run is resolved. The interleaved order's second and
//: third ceilings (a sub-box's leaves, the capacity multiplier) died with it -- under
//: mixed radix a unit is a contiguous run and there is no sub-box scan to bound.
//: docs/internals/decode/decode_lattice.md#capacities
constexpr int kDecodeMaxSpan = 32;

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

//: The atom as a SHIFT and a MASK. `kAtomLeaves` itself is the box's
//: (`rola::carry::kAtomLeaves`, imported above): it is geometry-owned and fixed across
//: every config forever, which is what makes the paged address BC-free.
//: docs/internals/decode/decode.md#the-atom
constexpr int kAtomShift = rola::carry::ilog2c(kAtomLeaves);
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

  //: THE (k, m) BOX, `BoxPlan`'s fields as runtime scalars under `BoxPlan`'s names:
  //: `lat_s[l] = k * m_l` is an owner's span at level `l`, `lat_g[l] = B_l / lat_s[l]`
  //: its owner-grid extent, and a leaf's lattice index is
  //: `lambda = O * BC + sum_l r_l << lat_run_shift[l]` -- the owner in mixed radix over
  //: `lat_g`, the OWNER-LOCAL index in mixed radix over the runs `lat_s`, most
  //: significant level first. `lat_run_shift[l] = log2 prod_{l' > l} s_l'` is
  //: `BoxPlan::run_shift`, and every suffix product is a SHIFT because every span is a
  //: power of two. docs/internals/decode/decode_lattice.md
  int lat_k;
  int lat_s[kMaxLevels];
  int lat_g[kMaxLevels];
  int lat_gsuf[kMaxLevels];       // prod_{l' > l} lat_g[l']
  int lat_run_shift[kMaxLevels];  // BoxPlan::run_shift(l)
  int lat_local_bits;             // BoxPlan::kLocalBits = log2 BC

  //: THE WALK'S UNIT, and the whole of it: a unit is `(owner, r_0 .. r_{lat_p - 1})` --
  //: an owner together with its OUTER run offsets -- so its leaves are the innermost
  //: levels' whole run, `1 << lat_unit_bits` of them, CONTIGUOUS and aligned in lattice
  //: order. `lat_p` is the outermost level whose suffix run reaches an atom, which is what
  //: makes a unit a whole number of atoms rather than a straddle: at the flagship spans
  //: `(8, 16)` it is `D - 1`, the unit is `(owner, r_0)` and its leaves are EXACTLY ONE
  //: atom at full occupancy. docs/internals/decode/decode_lattice.md#the-unit
  int lat_p;
  int lat_unit_bits;   // log2 of a unit's leaves = lat_run_shift[lat_p - 1], or kLocalBits
  int lat_outer_bits;  // lat_local_bits - lat_unit_bits: the unit's radix inside an owner
  int lat_inner_bits;  // log2 s_{D-1}: the innermost level's run, read as ONE slice
  int lat_hi_bits;     // lat_unit_bits - lat_inner_bits: the unit's mid-level radix
  //: `r_l`'s shift inside the unit index (`l < lat_p`) and inside the unit's mid-level
  //: index (`lat_p <= l < D - 1`). Both are `lat_run_shift[l]` re-based, so a unit and a
  //: slot each decompose by shifts alone and no device division exists on this path.
  int lat_oshift[kMaxLevels];
  int lat_hshift[kMaxLevels];
  //: The digit masks' word layout: level `l` owns `mask_words[l]` words at `mask_off[l]`
  //: of a `mask_total`-word plane, one plane per side.
  int mask_off[kMaxLevels];
  int mask_words[kMaxLevels];
  int mask_total;
  //: The union owner-coordinate lists' layout: `lat_g[l]` entries reserved per level.
  int olist_off[kMaxLevels];
  int olist_total;
  //: Atoms per unit as a shift, `lat_unit_bits - kAtomShift`. PAGED BACKING ONLY -- the
  //: host refuses a page table under a lattice whose owner is thinner than an atom,
  //: because then an atom would straddle two units and the claim's enumeration would not
  //: be a partition: docs/internals/decode/decode.md#candidate-atoms
  int atoms_per_unit_shift;

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
