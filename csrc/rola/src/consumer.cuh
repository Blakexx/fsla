#pragma once

// RoLA V3 consumer -- shared parameter block and owner-geometry arithmetic.
//
// ROLA_V3_SPEC.md §3 (topology-agnostic kernel structure), §4 (schedule), §5
// (consumer). This header holds only what both the kernel and the host binding
// must agree on; the executable structure lives in `consumer.cu`.
//
// The kernel is TOPOLOGY-AGNOSTIC by construction: it never learns which levels
// are dense, sparse, inner or outer. Every level is described by the same four
// integers (`base`, `stride`, `modulus`, `row_offset`) and is walked by the same
// code. §3's "every apparent special case is a degeneracy of this path" is
// implemented literally -- a fixed level is `stride = BC, modulus = 1`, which
// makes its digit constant over the owner's leafs, and a fully spanned level is
// `base = 0, modulus = width`.

#include <cstdint>

// THE INTRA-KERNEL TRACER IS RETIRED (item 6b, 2026-08-01). `v3_trace.cuh`,
// `v3_trace_host.{cu,cuh}` and `development/measurement/rola_trace{,_capture}.py` were
// an instrument with 27 `V3T*` emission sites, ALL OF THEM in the REGISTER-RESIDENT
// consumer (whose header was also named `consumer_kernel.cuh`; readable at
// `c3c2f845`, and a different kernel from today's file of that name). It is deleted; the surviving consumer carries
// zero. An instrument pointed at no kernel is dead code by the standing rule, so it
// is retired WITH the kernel rather than left to look live. Re-instrumenting the
// surviving consumer is a new item with its own emission sites, not a revival.

namespace rola {

constexpr int kMaxLevels = 4;   // spec §1: 1 <= D <= 4
constexpr int kMaxWords = 128;  // candidate-tile words held in SMEM per CTA

//: The staging-ring depth is COMPILE-TIME: `consumer_ring_depth()` in
//: `consumer_kernel.cuh` resolves it per instantiation from `ROLA_C_RING_DEPTH`
//: (shipped 1; depth 2 DECLINED on measurement, session-9 R2(b)), and the
//: ratification manifest records the macro (`tools/ratify.py` `RATIFIED_DEFINES`).
//: The runtime `ConsumerParams::ring_slots` this file once carried was read by no
//: kernel line and was removed (coverage-closure H3); the agreement gate
//: `tests/unit/test_ring_depth_agreement.py` pins planner == compiled depth.

// One level's contribution to the leaf-digit decomposition of a local leaf index
// `i` in [0, BC):
//
//     digit(i) = base + (i / stride) % modulus
//
// with, per §3's owner rectangle:
//   fixed level  l < j : base = the owner's fixed digit, stride = BC, modulus = 1
//   boundary level j   : base = slice * c,               stride = inner, modulus = c
//   spanned level l > j: base = 0,                       stride = prod(widths[l+1:]),
//                                                        modulus = width_l
//
// `row_offset` is the level's first row in the packed tile-bitmap array.
// (Encoded per-level in `ConsumerParams`'s flat arrays below, not as a
// per-level struct -- there is no `LevelDesc`-shaped type in the kernel.)

struct ConsumerParams;

//: Level `level`'s DIGIT for the owner-local leaf `i`, `digit = base + (i /
//: stride) % modulus`. A fixed level has `stride = BC, modulus = 1`, so the term
//: vanishes and the digit is the owner's fixed digit -- the same expression, no
//: branch (§3). Declared here rather than in the `.cu`'s anonymous namespace
//: because `intra_panel.cuh`'s clock prefix needs it: the clock is a product over
//: ALL levels (it is an exponent, so a fixed level cannot hoist into the row
//: scale), which is the one operand assembly that walks the leaf digits outside
//: the kernel translation unit. Defined below the struct.
__device__ __forceinline__ int leaf_digit(const ConsumerParams& p, int level, int base, int i);

struct ConsumerParams {
  const float* amp_read[kMaxLevels];    // [BH, n_tiles, BT, pitch_read]
  const float* amp_write[kMaxLevels];   // [BH, n_tiles, BT, pitch_write]
  // LeafMassDecay clock stream, `c[t,s] = prod_l p_write_stored[t, d_l(s)]`
  // (spec §2). Staged by `build_clock_stream`, which is a separate builder from
  // the schedule so the STORED tensor can never key a tile bitmap (P5.20.5
  // finding 2). nullptr under `decay=None`. NOTE the product runs over ALL levels
  // including `l < j`: the clock is an exponent, so a fixed level's factor does
  // not hoist into the row scale the way the read/write factors do.
  const float* amp_clock[kMaxLevels];   // [BH, n_tiles, BT, pitch_clock] or nullptr
  int level_pitch_clock[kMaxLevels];
  int level_stride[kMaxLevels];         // per-level (i / stride) % modulus
  int level_modulus[kMaxLevels];
  int level_row_offset[kMaxLevels];
  int level_width[kMaxLevels];
  int level_pitch_read[kMaxLevels];
  int level_pitch_write[kMaxLevels];
  // prod(widths[l+1:j]) for the fixed levels l < j; the owner-id mixed radix.
  int owner_stride[kMaxLevels];
  // `gram` intra lowering only: where level `l >= j`'s `modulus` columns start in
  // the per-tile staged amplitude rows, and their total width. The factor Gram
  // reads each level's RAW amplitude row (not the assembled leaf panel), so those
  // rows are staged into SMEM once per tile -- `BT * gram_cols` floats, versus the
  // `BT^2 * gram_cols` global loads a direct read would issue. Zero when the
  // lowering is not selected.
  int level_gram_offset[kMaxLevels];
  int gram_cols;

  const int32_t* owners;        // [n_owners, 2]  (bh, owner)
  const uint32_t* read_bits;    // [BH, total_rows, n_words]
  const uint32_t* write_bits;   // [BH, total_rows, n_words]
  //: §4's SUPPORT WORD, materialized: `[BH, n_tiles, total_rows]`, bit `t` = token
  //: `tile * BT + t` (`BT <= 32`, spec §6). Same level-digit rows as `read_bits`, one
  //: granularity finer -- `read_bits` is literally its tile reduction, built from it
  //: -- but TILE-MAJOR, so one visit reads one contiguous slab.
  //: THIS IS THE SOLE SUPPORT AUTHORITY: §3 step 2's gather, step 1's FIXED-level
  //: token validity, and the staging predicate all read it, and no MEMBERSHIP
  //: decision is derived from an amplitude magnitude any more (harvest.md §23.2
  //: ruling 1; conformance audit item 3 / S1 -- the §3 step 2-3 inversion).
  //:
  //: PRECISELY, because an earlier wording overclaimed (Gate D): amplitude
  //: comparisons still exist in the kernel, at its `if (out != 0.0f)` split-K
  //: emission sites. Those are PERFORMANCE SKIPS on a VALUE
  //: -- adding `0.0f` is the identity, so including the term is a no-op and the test
  //: is over-inclusion-safe in the direction that matters. None of them decides
  //: membership, which is what this word is the authority for.
  const uint32_t* read_tok_bits;   // [BH, n_tiles, total_rows]
  const uint32_t* write_tok_bits;  // [BH, n_tiles, total_rows]
  const uint32_t* read_union;   // [BH, n_slices,  n_words]
  const uint32_t* write_union;  // [BH, n_slices,  n_words]

  // Flattened LeafMassDecay dials, [H, total_rows], level `l` at row_offset[l].
  // `rate[s] = prod_l delta_l[d_l(s)]` -- a plain product of DIALS (spec §2), not
  // V2's AND over per-branch retentions, so V2's pairwise log-union composition
  // has no V3 counterpart. Accumulated in fp32 and never bf16: `rate` near 1
  // makes `1 - rate` catastrophically cancelling, and `log1pf` cannot repair a
  // cancellation that already happened inside the product.
  const float* dials;           // [H, total_rows] or nullptr

  const float* g_read;          // [BH, T] or nullptr (global fixes g_read := 1)
  const float* g_write;         // [BH, T]
  const float* v;               // [BH, T, DV]

  float* num;                   // [BH, T, DV]   fp32, atomically accumulated
  float* den;                   // [BH, T]       fp32, or nullptr under `raw`
  //: Unpaged: `[BH, N, DV (+1 mass column under `global`)]`; ATOM `q` of owner `o`
  //: of batch-head `bh` starts at row `bh * N + o * BC + q * MMA_K_QUANTUM`.
  //: Paged (spec §4.5): `[physical_slots, MMA_K_QUANTUM, cols]`; that same atom
  //: starts at row `page_table[bh * atoms_per_bh + o * QPO + q] * MMA_K_QUANTUM`.
  //: A page is one `MMA_K_QUANTUM`-leaf ATOM and NOT one owner (`page_states := BC`
  //: is dead -- §1b makes `BC` a re-derivable COMPUTE tile and the page a
  //: MEMORY-RESIDENCY granule, so that a per-launch `BC` never rebuilds the arena).
  //: An owner is therefore `QPO = BC / MMA_K_QUANTUM` independently paged runs, and
  //: `atom_row_base` in `consumer_kernel.cuh` is the one place that arithmetic lives.
  float* state;
  //: `[BH, atoms_per_bh]` int32, or nullptr for the unpaged path. `kAbsentPage`
  //: (-1) means the atom has no committed slot, which can only happen for an atom
  //: with no write candidate at all -- its unpaged state is identically zero.
  //: Admission is TOTAL (the capped mode and its admit mask are deleted), so there
  //: is no second reason for an absent entry. An absent atom reads as zeros and is
  //: never stored back, which is what makes paging BITWISE INVISIBLE rather than
  //: approximately equivalent.
  const int32_t* page_table;

  int64_t N;
  int BH;
  int T;
  int n_tiles;
  int n_words;
  int total_rows;
  int n_slices;
  int D;
  int first_spanned_level;      // j
  int slice_digits;             // c
  int n_owners;
  int H;                        // heads; the dial row is `bh % H` (fold is b*H+h)

  // ---- §6.4 sequence parallelism (`P` per owner) --------------------------
  // The grid is a FLAT (owner, segment) work list: `owners` carries the
  // `(bh, owner)` pair as always, and the three arrays below carry the segment
  // coordinates, one entry per work item, in the same order.
  //
  // Segment `s` of an owner split `nseg` ways owns the contiguous TILE range
  // `[s*n_tiles/nseg, (s+1)*n_tiles/nseg)`. A `nullptr` here is `P = 1`
  // everywhere: `seg = 0`, `nseg = 1`, the range is the whole token axis, and
  // every expression below collapses to the pre-`P` kernel WITHOUT reading
  // memory -- which is what keeps `P = 1` bit-identical, and `P = 1` is the
  // primary benchmarked configuration (spec §6.4).
  const int32_t* work_seg;      // [n_owners] or nullptr
  const int32_t* work_nseg;     // [n_owners] or nullptr
  const int32_t* work_slot;     // [n_owners] first workspace slot OF THE OWNER

  // §6.4's partial-state workspace. LAUNCH-SCOPED and NOT the §7.1 arena: the
  // arena's contract is persistent cross-chunk state pages only, and mixing a
  // transient lifetime into it would put launch-scoped garbage under the
  // compactor. The framework's caching allocator owns this (a plain torch
  // tensor, sized by the plan); what is unified is the planner's LEDGER, not
  // the allocator.
  //
  // A split owner owns `nseg` slots starting at `slot[owner]` -- §6.4's amended
  // `P * BC * d_v` peak -- laid out so that pass 3's segment `p` reads exactly
  // `slot[owner] + p`:
  //
  //   slot[owner]              : the ENTRY-STATE STASH `M_0`, dense, written by
  //                              pass 2 and read by pass 3's segment 0.
  //   slot[owner] + 1 + s      : segment `s`'s boundary, `s in [0, nseg-1)`.
  //
  // **The stash is a correctness structure, not an optimization (fix pass R1).**
  // Without it pass 3's segment 0 read `p.state`, whose rows the SAME launch's
  // last segment writes from another CTA -- an unsynchronized intra-launch
  // dependence that only the work list's incidental ordering hid (a permitted
  // permutation of that list produced `y` with relative error 1.0). With it, every
  // pass-3 read is either a workspace slot written by an EARLIER LAUNCH or a
  // `p.state` row of an UNSPLIT owner, which no other CTA touches -- disjoint from
  // all same-launch writes by construction.
  //
  // A boundary slot carries TWO things at two different points of the three-pass
  // life, deliberately in the same bytes:
  //
  //   after pass 1 : the SPARSE written-leaf summary -- `ws_count[slot]` entries,
  //                  leaf ids in `ws_leaf`, per-leaf `A` in `ws_scale`, per-leaf
  //                  `B` in the first `ws_count` rows of `ws_state`.
  //   after pass 2 : the DENSE accumulated boundary state, leaf-indexed, written
  //                  in place over the summary the scan has just consumed.
  //
  // §6.4 states both: the summary is normatively sparse (its traffic scales with
  // realized writes), and the accumulated boundary pass 3 reads is dense. The stash
  // slot uses only `ws_state`; its `ws_scale`/`ws_leaf`/`ws_count` entries are
  // neither written nor read (the scan's fold indexes `slot[owner] + 1 + s`).
  float* ws_state;              // [n_slots, BC, cols]
  float* ws_scale;              // [n_slots, BC]
  int32_t* ws_leaf;             // [n_slots, BC]
  int32_t* ws_count;            // [n_slots]

  // Which of §6.4's three passes this launch is. LAUNCH-UNIFORM (every CTA in a
  // launch has the same value), so the branches it drives are uniform and are
  // structure, not a distributional property -- the same status as `is_write`.
  int pass_mode;
};

__device__ __forceinline__ int leaf_digit(const ConsumerParams& p, int level, int base, int i) {
  return base + (i / p.level_stride[level]) % p.level_modulus[level];
}

//: §6.4's three passes. Pass 2 is `scan_kernel`, a separate launch, so only
//: the two consumer passes appear here.
enum PassMode : int {
  kPassRead = 0,     //: pass 3 -- reads emitted, entry state from the scan, and the
                     //: LAST segment publishes the owner's final state. At `P = 1`
                     //: this is the whole kernel and is unchanged.
  kPassSummary = 1,  //: pass 1 -- entry state ZERO, no read emitted (a read-only
                     //: tile cannot move the state, and a read emitted here would be
                     //: double-counted by pass 3), publishes the sparse `(A, B)`.
};

//: Pass 2: compose the per-segment summaries into the accumulated boundary states,
//: in order. One CTA per SPLIT owner; the fold is sequential over segments inside
//: it, so there is no inversion anywhere (P5.21's obligation).
struct ScanParams {
  const int32_t* owners;        // [n_split, 2]  (bh, owner)
  const int32_t* nseg;          // [n_split]
  const int32_t* slot;          // [n_split]
  float* ws_state;
  const float* ws_scale;
  const int32_t* ws_leaf;
  const int32_t* ws_count;
  const float* state;
  const int32_t* page_table;
  int64_t N;
  int BC;
  int cols;
  int n_split;
};

}  // namespace rola
