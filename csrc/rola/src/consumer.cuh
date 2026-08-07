#pragma once

// RoLA V3 consumer -- shared parameter block and owner-geometry arithmetic.
// ROLA_V3_SPEC.md §3 (topology-agnostic structure), §4 (schedule), §5 (consumer),
// §6.4 (sequence parallelism). This header holds only what the kernel and the host
// binding must agree on; the executable structure lives in `consumer.cu` and the
// kernel body in `consumer_kernel.cuh`.
//
// The kernel is TOPOLOGY-AGNOSTIC by construction: it never learns which levels are
// dense, sparse, inner or outer. Every level is four integers (`base`, `stride`,
// `modulus`, `row_offset`) walked by the same code, so §3's "every apparent special
// case is a degeneracy of this path" is literal -- a fixed level is
// `stride = BC, modulus = 1`, a fully spanned one is `base = 0, modulus = width`.
// There is no branch on level kind and no `LevelDesc`-shaped type.
//
// The digit decomposition, the paged/unpaged state layouts, the §6.4 work list and
// its workspace, and why paging is BITWISE invisible: docs/internals/consumer.md

#include <cstdint>
#include <cuda_runtime.h>

//: For `clamp_bf16_finite` and the bf16 MMA primitives shared with the kernel. The
//: precision AXIS is gone (bf16 storage, fp32 accumulation, fp64 reference as
//: oracle -- the one precision story); what remains here is the format guard.
#include "readout_mma.cuh"

namespace rola {

constexpr int kMaxLevels = 4;  // spec §1: 1 <= D <= 4

struct ConsumerParams;

//: Level `level`'s DIGIT for the owner-local leaf `i`, `base + (i / stride) %
//: modulus` -- one expression for every level kind, no branch (§3). Declared here
//: rather than in the `.cu`'s anonymous namespace because `intra_panel.cuh`'s clock
//: prefix needs it. Defined below the struct.
//: docs/internals/consumer.md#leaf-digit
__device__ __forceinline__ int leaf_digit(const ConsumerParams& p, int level, int base, int i);

struct ConsumerParams {
  //: [BH, n_twords, 32, pitch] amplitude streams (token-major; the 32-token
  //: MACHINE WORD is the metadata granule, so the padding quantum is 32),
  //: TYPE-ERASED because the storage dtype is per LEVEL ROLE and per
  //: instantiation: SPANNED levels (`l >= j`) store the arm's amplitude type
  //: `AmpRT`/`AmpWT` (bf16 on the `kBf16` arms -- U07-d's fp16 decay
  //: application unified the decay carve), FIXED levels (`l < j`) always fp32
  //: (they feed the row scales). Each kernel use site types its own pointer
  //: from its loop's level role, so a missed site is a compile error; the host
  //: binding enforces the same rule per level with a hard dtype check.
  //: docs/internals/consumer.md#amp-dtype
  const void* amp_read[kMaxLevels];   // pitch_read elements per row
  const void* amp_write[kMaxLevels];  // pitch_write elements per row
  //: LeafMassDecay clock stream, `c[t,s] = prod_l p_write_stored[t, d_l(s)]` (spec
  //: §2); nullptr under `decay=None`. Built SEPARATELY from the schedule, and the
  //: product runs over ALL levels: docs/internals/consumer.md#clock-stream
  const float* amp_clock[kMaxLevels];  // [BH, n_twords, 32, pitch_clock] or nullptr
  int level_pitch_clock[kMaxLevels];
  int level_stride[kMaxLevels];  // per-level (i / stride) % modulus
  int level_modulus[kMaxLevels];
  int level_row_offset[kMaxLevels];
  int level_width[kMaxLevels];
  int level_pitch_read[kMaxLevels];
  int level_pitch_write[kMaxLevels];
  // prod(widths[l+1:j]) for the fixed levels l < j; the owner-id mixed radix.
  int owner_stride[kMaxLevels];
  const int32_t* owners;  // [n_owners, 2]  (bh, owner)
  //: §4's SUPPORT WORD, materialized: `[BH, n_twords, total_rows]`, bit `t` =
  //: token `word * 32 + t`. WORD-MAJOR, so the membership scan reads one
  //: contiguous slab per 32-token word.
  //: THIS IS THE SOLE SUPPORT AUTHORITY -- no membership decision is derived from an
  //: amplitude magnitude. docs/internals/consumer.md#support-word
  const uint32_t* read_tok_bits;   // [BH, n_twords, total_rows]
  const uint32_t* write_tok_bits;  // [BH, n_twords, total_rows]
  //: The TOKEN-GRANULAR slice union (schedule ABI v4; the card's TWO UNION
  //: FORMS, form (a)): bit `t` of word `[bh, slice, word]` = "some digit of the
  //: slice at the boundary level is read- OR write-live at token `word*32 + t`".
  //: The membership ballot's coarse term; the fine terms it ANDs under are the
  //: owner's fixed-level `read/write_tok_bits` rows. Derived from the same
  //: support authority, so over-approximation is impossible by construction.
  const uint32_t* slice_union_tok;  // [BH, n_slices, n_twords]
  //: The ATOM-MAJOR CONTIGUOUS UNION BITLISTS (TWO UNION FORMS, form (b);
  //: Blake's confirmed layout 2026-08-06): bit `t` of word `[bh, slice, q, w]`
  //: = "some leaf of atom `q` of this slice is live at token `w*32+t` on this
  //: side" (an atom = kMmaKQuantum consecutive leaves; QPO = BC/16 rows per
  //: slice). ATOM-MAJOR so one atom's whole token-liveness row is CONTIGUOUS
  //: -- the per-atom fetch era streams it without a layout migration. The
  //: ballot consumes it WORD-WISE: atom bit q of a visit's mask = (row q's
  //: word at the taken source word) AND taken != 0. Derived host-side with
  //: the token bits (one summaries pass, consistency by construction); the
  //: per-(owner,word) pre-OR'd form is FORBIDDEN (materialization rule) --
  //: the per-visit OR is the kernel's job.
  const uint32_t* read_atom_bits;   // [BH, n_slices, QPO, n_twords]
  const uint32_t* write_atom_bits;  // [BH, n_slices, QPO, n_twords]

  //: Flattened LeafMassDecay dials, [H, total_rows], level `l` at row_offset[l].
  //: `rate[s] = prod_l delta_l[d_l(s)]`, a plain product of DIALS (spec §2).
  //: HAZARD dials-fp32 -- docs/internals/consumer.md#dials-fp32
  const float* dials;  // [H, total_rows] or nullptr

  //: `g_read` is GONE: `global` fixes g_read := 1 and raw is dead.
  const float* g_write;  // [BH, T]
  //: [BH, T, DV] value stream, HBM storage bf16 (the one precision story); bound
  //: with a hard dtype check.
  const void* v;

  float* num;  // [BH, T, DV]   fp32, atomically accumulated
  float* den;  // [BH, T]       fp32, or nullptr under `raw`
  //: Unpaged: `[BH, N, DV (+1 mass column under `global`)]`. Paged (spec §4.5):
  //: `[physical_slots, MMA_K_QUANTUM, cols]`. A page is one `MMA_K_QUANTUM`-leaf
  //: ATOM and NOT one owner; `atom_row_base` in `consumer_kernel.cuh` is the one
  //: place that arithmetic lives. docs/internals/consumer.md#state-layout
  float* state;
  //: `[BH, atoms_per_bh]` int32, or nullptr for the unpaged path. `kAbsentPage`
  //: (-1) means the atom has no committed slot -- possible only for an atom with no
  //: write candidate at all, whose unpaged state is identically zero. It reads as
  //: zeros and is never stored back, which is what makes paging BITWISE INVISIBLE
  //: rather than approximately equivalent: docs/internals/consumer.md#page-table
  const int32_t* page_table;

  int64_t N;
  int BH;
  int T;
  int n_twords;  // ceil(T / 32): the scan stream's word count
  int total_rows;
  int n_slices;
  int D;
  int first_spanned_level;  // j
  int slice_digits;         // c
  int n_owners;
  int H;  // heads; the dial row is `bh % H` (fold is b*H+h)

  // ---- §6.4 sequence parallelism (`P` per owner) --------------------------
  //: A FLAT (owner, segment) work list; the three arrays carry the segment
  //: coordinates, one entry per work item, in `owners` order. A nullptr is `P = 1`
  //: everywhere and collapses every expression below to the pre-`P` kernel WITHOUT
  //: reading memory, which is what keeps `P = 1` BIT-IDENTICAL.
  //: docs/internals/consumer.md#work-list
  const int32_t* work_seg;   // [n_owners] or nullptr
  const int32_t* work_nseg;  // [n_owners] or nullptr
  const int32_t* work_slot;  // [n_owners] first workspace slot OF THE OWNER

  //: §6.4's partial-state workspace. LAUNCH-SCOPED and NOT the §7.1 arena; a split
  //: owner owns `nseg` slots from `slot[owner]`, of which slot 0 is the ENTRY-STATE
  //: STASH and `1 + s` is segment `s`'s boundary. The stash is a CORRECTNESS
  //: structure, not an optimization, and a boundary slot deliberately holds a sparse
  //: summary then a dense state in the same bytes:
  //: docs/internals/consumer.md#workspace, #boundary-slot
  float* ws_state;    // [n_slots, BC, cols]
  float* ws_scale;    // [n_slots, BC]
  int32_t* ws_leaf;   // [n_slots, BC]
  int32_t* ws_count;  // [n_slots]

  //: Which of §6.4's three passes this launch is. LAUNCH-UNIFORM, so the branches it
  //: drives are structure and not a distributional property -- the same status as
  //: `is_write`. docs/internals/consumer_kernel.md#pass-mode
  int pass_mode;
};

__device__ __forceinline__ int leaf_digit(const ConsumerParams& p, int level, int base, int i) {
  return base + (i / p.level_stride[level]) % p.level_modulus[level];
}

//: §6.4's three passes. Pass 2 is `scan_kernel`, a separate launch, so only the two
//: consumer passes appear here. docs/internals/consumer.md#pass-mode
enum PassMode : int {
  kPassRead = 0,     //: pass 3 -- reads emitted, entry state from the scan, LAST
                     //: segment publishes. At `P = 1` this is the whole kernel.
  kPassSummary = 1,  //: pass 1 -- entry state ZERO, no read emitted, publishes the
                     //: sparse `(A, B)`.
};

//: Pass 2: compose the per-segment summaries into the accumulated boundary states,
//: in order. One CTA per SPLIT owner, fold sequential over segments inside it.
struct ScanParams {
  const int32_t* owners;  // [n_split, 2]  (bh, owner)
  const int32_t* nseg;    // [n_split]
  const int32_t* slot;    // [n_split]
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

// ---------------------------------------------------------------------------
// THE PER-ARM ENTRY POINTS -- DECLARATIONS ONLY.
//
// One instantiation of each of these per template tuple lives in exactly one
// generated shard (`csrc/rola/src/instantiations/`), and their DEFINITIONS live in
// `consumer_launch.cuh`, which the dispatch translation unit must not include. That
// asymmetry is what makes the split REAL: with the definition out of scope, the
// compiler cannot satisfy a dispatch arm by re-instantiating it here. It does NOT
// make a missing member a link error -- this is a Python extension module, so the
// linker leaves the reference undefined for the loader and the failure would reach a
// SHIPPED artifact as an ImportError. The closure is therefore CHECKED, before the
// compile by the shard gate and after the link by `tools/fatbin_entries.py`:
// docs/internals/consumer.md#instantiation-matrix
// ---------------------------------------------------------------------------

template <int BC, int DV, int D, bool DECAY, int GT>
void launch_consumer_arm(const ConsumerParams& p, cudaStream_t stream);

template <int BC, int DV, int D, bool DECAY, int GT>
int query_consumer_occupancy_arm();

using LaunchFn = void (*)(const ConsumerParams&, cudaStream_t);
using OccFn = int (*)();

//: ONE ROW OF THE INSTANTIATION TABLE: an arm's structural coordinates and the three
//: things only its own shard can answer -- how to launch it, what the driver says
//: about its residency, and its `constexpr` shared-memory footprint. The residency
//: and footprint queries resolve a row rather than re-enumerating the matrix, which
//: is why the dispatch translation unit does not include the kernel header at all.
//: docs/internals/consumer.md#shard-rows
//: THE CHOICE A DISPATCH CONSUMES. One value per matrix axis and nothing else: no
//: statistics, no thresholds, no device facts. Something upstream -- the planner's
//: selector, a test, a debug pin -- DECIDES, and the table RESOLVES. The table never
//: computes a choice, which is what keeps selection replaceable without touching the
//: dispatch. docs/internals/consumer.md#dispatch-choice
struct DispatchChoice {
  int BC;
  int DV;
  int D;
  bool decay;
  //: THE WINDOW, in tokens. A structural axis like every other field here, and the
  //: one the PLANNER chooses from the realized routing rather than from the shape:
  //: a wide window is free at dense gathering, wastes tile at sparse gathering, and
  //: on the decay arm contracts the admitted routing domain in proportion to its
  //: width. docs/internals/consumer.md#dispatch-choice
  int GT;
};

//: THE MACHINE FACTS A ROW IS ADMITTED AGAINST. Filled from the driver once per
//: process. Held as a struct rather than passed as a scalar so that a second fact
//: (a register budget, a driver version) is a field and not a signature change.
//: docs/internals/consumer.md#row-admission
struct DeviceFacts {
  size_t max_dynamic_smem_bytes;
};

struct ShardRow {
  int BC;
  int DV;
  int D;
  bool decay;
  int GT;
  LaunchFn launch;
  OccFn occupancy;
  size_t smem_bytes;
  //: The generated translation unit that OWNS this row, and that unit's
  //: `members-sha256` -- the hash of the sorted tuple list it carries. The pair is
  //: the row's STALENESS KEY: it is what `tools/manifests/sm_*.json` records under
  //: `shards`, so a row can be joined against the ratification that measured it and
  //: against any record taken of this binary. docs/internals/consumer.md#row-staleness
  const char* shard;
  const char* src_key;
};

//: THE PER-ROW ENABLE PREDICATE, AND IT KEYS ON AN INPUT. A row is selectable iff its
//: `constexpr` footprint fits what THIS DEVICE will grant a block as dynamic shared
//: memory. The predicate is evaluated per row against a runtime fact, never on a
//: preprocessor symbol -- flash-attention's `#ifndef FLASHATTENTION_DISABLE_HDIM{N}`
//: is the shape being refused: it makes the arm set a property of the build rather
//: than of the machine, so the binary cannot say what it can run.
//: docs/internals/consumer.md#row-admission
inline bool row_admitted(const ShardRow& row, const DeviceFacts& device) {
  return row.smem_bytes <= device.max_dynamic_smem_bytes;
}

//: One shard's contiguous slice of the table. The blocks are concatenated, never
//: merged into one array: each shard OWNS its rows, so P06's generated dispatch
//: table is a concatenation rather than a rebuild.
struct ShardRowBlock {
  const ShardRow* rows;
  int count;
};

}  // namespace rola
