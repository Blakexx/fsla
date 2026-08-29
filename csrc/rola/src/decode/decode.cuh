#pragma once

// THE T=1 DECODE PATH -- the parameter block and the objects both decode TUs
// share.
// see docs/internals/decode/decode.md#note-l3

#include <cstdint>

#include "../carry/box.cuh"

namespace rola {
namespace decode {

//: THE PAGE GRANULE AND THE OWNER'S SHAPE COME FROM THE BOX, not from a mirror
//: of it: `carry/box.cuh` is where the lattice's addressing constants and its
//: constexpr `BoxPlan` arithmetic live ...
// see docs/internals/decode/decode.md#kmaxlevels
using rola::carry::kAtomLeaves;

//: `1 <= D <= 4`. The same bound the consumer's `kMaxLevels` states, restated
//: rather than included -- the decode TUs share no other object with the tiled
//: consumer. docs/internals/decode/decode.md#max-levels
constexpr int kMaxLevels = 4;

//: THE LEVEL-WIDTH CAPACITY, a GATE and not a convenience: below the producer's
//: `MAX_BRANCH_WIDTH` it would accept a topology the producer can emit and then
//: over-read a staged row.
// see docs/internals/decode/decode.md#kproducermaxbranchwidth
constexpr int kProducerMaxBranchWidth = 256;
constexpr int kDecodeMaxLevelWidth = 256;
static_assert(kDecodeMaxLevelWidth >= kProducerMaxBranchWidth,
              "the decode path's level-width capacity is below the producer's "
              "MAX_BRANCH_WIDTH: a topology the producer can emit would over-read "
              "the staged amplitude row");

//: THE STEP LAUNCHES ONE FAT CTA PER SM, and that is a SHARING decision before
//: it is an occupancy one.
// see docs/internals/decode/decode.md#kdecodethreads
constexpr int kDecodeThreads = 1024;
constexpr int kDecodeWarps = kDecodeThreads / 32;

//: THE UNIT PARTITION'S GRANULARITY, AND IT IS NOT THE CTA'S WARP COUNT.
// see docs/internals/decode/decode.md#kunitblockwarps
constexpr int kUnitBlockWarps = 8;
constexpr int kUnitBlockWarpsBits = 3;
static_assert(1 << kUnitBlockWarpsBits == kUnitBlockWarps,
              "kUnitBlockWarpsBits must be log2(kUnitBlockWarps)");
static_assert(kUnitBlockWarps <= kDecodeWarps && kDecodeWarps % kUnitBlockWarps == 0,
              "a CTA's warps are a whole number of unit blocks");
constexpr int kUnitRepeats = kDecodeWarps / kUnitBlockWarps;

//: THE LATTICE'S ONE DECLARED CAPACITY, refused loudly at the host entry: an
//: owner's per-level span is read as a SINGLE 32-bit slice, both when a level's
//: owner grid is built and when a unit's innermost run is resolved.
// see docs/internals/decode/decode.md#kdecodemaxspan
constexpr int kDecodeMaxSpan = 32;

//: THE DECLARED RESIDENCY of the step kernel, in CTAs per SM, and it is a
//: BUDGET rather than a hope: `__launch_bounds__` turns it into a register cap
//: the assembler must meet, and the census is what proves it met it without ...
// see docs/internals/decode/decode.md#kdecoderesidentthreads
constexpr int kDecodeResidentThreads = 1024;
__host__ __device__ constexpr int decode_step_blocks_per_sm(bool decay, int levels) {
  (void)decay;
  (void)levels;
  return kDecodeResidentThreads / kDecodeThreads;
}

//: The atom as a SHIFT and a MASK. `kAtomLeaves` itself is the box's
//: (`rola::carry::kAtomLeaves`, imported above): it is geometry-owned and fixed
//: across every config forever ...
// see docs/internals/decode/decode.md#katomshift
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
  //: THE RAW PER-TOKEN OPERANDS, ADDRESSED RATHER THAN COPIED: both sides'
  //: routing levels, the write gain and the token's value, each as the caller's
  //: own view.
  // see docs/internals/decode/decode.md#decodeparams-note-l152
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

  //: THE (k, m) BOX, `BoxPlan`'s fields as runtime scalars under `BoxPlan`'s
  //: names: `lat_s[l] = k * m_l` is an owner's span at level `l`, `lat_g[l] = B_l
  //: / lat_s[l]` its owner-grid extent ...
  // see docs/internals/decode/decode.md#decodeparams-note-l174
  int lat_k;
  int lat_s[kMaxLevels];
  int lat_g[kMaxLevels];
  int lat_gsuf[kMaxLevels];       // prod_{l' > l} lat_g[l']
  int lat_sbits[kMaxLevels];      // log2 lat_s[l]: a digit splits into (owner, run) by shifts
  int lat_run_shift[kMaxLevels];  // BoxPlan::run_shift(l)
  int lat_local_bits;             // BoxPlan::kLocalBits = log2 BC

  //: THE WALK'S UNIT, and the whole of it: a unit is `(owner, r_0 .. r_{D-2})` --
  //: an owner together with every run offset but the innermost ...
  // see docs/internals/decode/decode.md#decodeparams-note-l190
  int lat_inner_bits;  // log2 s_{D-1}: the unit's leaves, and ONE slice of the digit mask
  int lat_outer_bits;  // lat_local_bits - lat_inner_bits: the unit's radix inside an owner
  //: `r_l`'s shift inside the unit index: `lat_run_shift[l]` re-based on the innermost run,
  //: so a unit decomposes by shifts alone and no device division exists on this path.
  int lat_oshift[kMaxLevels];
  //: The digit masks' word layout: level `l` owns `mask_words[l]` words at `mask_off[l]`
  //: of a `mask_total`-word plane, one plane per side.
  int mask_off[kMaxLevels];
  int mask_words[kMaxLevels];
  int mask_total;
  //: THE OWNER LIVENESS BITMASK'S word layout: one bit per OWNER COORDINATE, per
  //: level, per kind -- `omask_words[l] = ceil(lat_g[l] / 32)` words at
  //: `omask_off[l]` of an `omask_total`-word plane, one plane per kind.
  // see docs/internals/decode/decode.md#decodeparams-note-l209
  int omask_off[kMaxLevels];
  int omask_words[kMaxLevels];
  int omask_total;
  //: THE COMPACTED LISTS THE BITMASK PRODUCES, and they are read exactly as they
  //: were before D1: `olist[kind][level][i]` is the i-th live owner, one indexed
  //: load.
  // see docs/internals/decode/decode.md#decodeparams-note-l221
  int olist_off[kMaxLevels];
  int olist_total;
  //: THE WRITE SIDE'S LIVE-DIGIT LIST, OUTER LEVELS ONLY -- term 0's own grain, compacted
  //: from `mask_w` by the same prefix popcount:
  //: docs/internals/decode/decode_lattice.md#the-write-grain
  int dlist_off[kMaxLevels];
  int dlist_total;
  //: THE ATOM AND THE UNIT, RECONCILED, and PAGED BACKING ONLY. Exactly one of
  //: these is non-zero: a unit holds `1 << atoms_per_unit_shift` atoms, or an
  //: atom holds `1 << lat_units_per_atom_shift` consecutive units.
  // see docs/internals/decode/decode.md#decodeparams-note-l233
  int atoms_per_unit_shift;
  int lat_units_per_atom_shift;

  const float* dials;  // [H, total_rows] or nullptr

  //: DENSE `[BH, N, cols]` or PAGED `[slots, kAtomLeaves, cols]`, `cols = d_v + 1`.
  //: Read AND written in place by the SAME lane in program order:
  //: docs/internals/decode/decode.md#state-in-place
  float* state;
  //: `[BH, atoms_per_bh]` int32, `-1` = ABSENT; `nullptr` = the dense backing,
  //: where the slot IS the atom.
  // see docs/internals/decode/decode.md#decodeparams-note-l250
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
  //: `[BH]` int32, THE DONE FLAGS -- `1` where a batch-head has already deposited
  //: this step.
  // see docs/internals/decode/decode.md#decodeparams-note-l265
  int32_t* done;
  //: `[BH, atoms_per_bh]` bool, this step's own write set condensed to atoms. It is what
  //: a growth step's `plan_exact` admits from, so the host never re-derives the set the
  //: step has already expanded: docs/internals/decode/decode.md#atom-bits
  bool* atom_bits;

  //: THE SLACK POOL -- physical slots pre-committed and pre-zeroed by the host,
  //: from which a growth step admits ITSELF.
  // see docs/internals/decode/decode.md#decodeparams-note-l275
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

//: EXTRACT `count` bits of a packed bitmap starting at BIT index `start`,
//: returned in the low `count` bits (`count <= 32`) ...
// see docs/internals/decode/decode.md#extract-bits-2
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
                              int omask_total, int olist_total, int dlist_total);
void launch_decode_step(const DecodeParams& p, int d_v, bool decay, int64_t smem_budget);

}  // namespace decode
}  // namespace rola
