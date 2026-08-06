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

//: For `MmaPrec`, which is a template argument of every per-arm entry point declared
//: at the bottom of this file. The host binding and the kernel must agree on the
//: precision axis exactly as they must agree on `ConsumerParams`.
#include "readout_mma.cuh"

namespace rola {

constexpr int kMaxLevels = 4;   // spec §1: 1 <= D <= 4
constexpr int kMaxWords = 128;  // candidate-tile words held in SMEM per CTA

//: The staging-ring depth is COMPILE-TIME (`consumer_ring_depth()` resolves it per
//: instantiation from `ROLA_C_RING_DEPTH`, shipped 1) and the ratification manifest
//: records the macro: docs/internals/consumer.md#ring-depth

struct ConsumerParams;

//: Level `level`'s DIGIT for the owner-local leaf `i`, `base + (i / stride) %
//: modulus` -- one expression for every level kind, no branch (§3). Declared here
//: rather than in the `.cu`'s anonymous namespace because `intra_panel.cuh`'s clock
//: prefix needs it. Defined below the struct.
//: docs/internals/consumer.md#leaf-digit
__device__ __forceinline__ int leaf_digit(const ConsumerParams& p, int level, int base, int i);

struct ConsumerParams {
  //: [BH, n_tiles, BT, pitch] amplitude streams, TYPE-ERASED because the storage
  //: dtype is per LEVEL ROLE and per instantiation: SPANNED levels (`l >= j`)
  //: store the arm's amplitude type `AmpRT`/`AmpWT` (bf16 on the
  //: `kBf16 && GLOBAL && !DECAY` arms -- the decay and raw carves are measured,
  //: see the kernel's alias block -- fp32 everywhere else), FIXED levels
  //: (`l < j`) always fp32 (they feed the row scales). Each kernel use site
  //: types its own pointer from its loop's level role, so a missed site is a
  //: compile error; the host binding enforces the same rule per level with a
  //: hard dtype check. docs/internals/consumer.md#amp-dtype
  const void* amp_read[kMaxLevels];   // pitch_read elements per row
  const void* amp_write[kMaxLevels];  // pitch_write elements per row
  //: LeafMassDecay clock stream, `c[t,s] = prod_l p_write_stored[t, d_l(s)]` (spec
  //: §2); nullptr under `decay=None`. Built SEPARATELY from the schedule, and the
  //: product runs over ALL levels: docs/internals/consumer.md#clock-stream
  const float* amp_clock[kMaxLevels];  // [BH, n_tiles, BT, pitch_clock] or nullptr
  int level_pitch_clock[kMaxLevels];
  int level_stride[kMaxLevels];  // per-level (i / stride) % modulus
  int level_modulus[kMaxLevels];
  int level_row_offset[kMaxLevels];
  int level_width[kMaxLevels];
  int level_pitch_read[kMaxLevels];
  int level_pitch_write[kMaxLevels];
  // prod(widths[l+1:j]) for the fixed levels l < j; the owner-id mixed radix.
  int owner_stride[kMaxLevels];
  const int32_t* owners;       // [n_owners, 2]  (bh, owner)
  const uint32_t* read_bits;   // [BH, total_rows, n_words]
  const uint32_t* write_bits;  // [BH, total_rows, n_words]
  //: §4's SUPPORT WORD, materialized: `[BH, n_tiles, total_rows]`, bit `t` = token
  //: `tile * BT + t` (`BT <= 32`, spec §6). TILE-MAJOR, so one visit reads one
  //: contiguous slab; `read_bits` is literally its tile reduction.
  //: THIS IS THE SOLE SUPPORT AUTHORITY -- no membership decision is derived from an
  //: amplitude magnitude. docs/internals/consumer.md#support-word
  const uint32_t* read_tok_bits;   // [BH, n_tiles, total_rows]
  const uint32_t* write_tok_bits;  // [BH, n_tiles, total_rows]
  const uint32_t* read_union;      // [BH, n_slices,  n_words]
  const uint32_t* write_union;     // [BH, n_slices,  n_words]

  //: Flattened LeafMassDecay dials, [H, total_rows], level `l` at row_offset[l].
  //: `rate[s] = prod_l delta_l[d_l(s)]`, a plain product of DIALS (spec §2).
  //: HAZARD dials-fp32 -- docs/internals/consumer.md#dials-fp32
  const float* dials;  // [H, total_rows] or nullptr

  const float* g_read;   // [BH, T] or nullptr (global fixes g_read := 1)
  const float* g_write;  // [BH, T]
  //: [BH, T, DV] value stream, HBM storage in the arm's amplitude type `AmpWT`
  //: (same scope as the spanned amplitudes: bf16 on `kBf16 && GLOBAL && !DECAY`,
  //: fp32 elsewhere; it is staged into the bf16 MMA operand either way on that
  //: arm); typed in-kernel, bound with a hard per-precision dtype check.
  const void* v;

  //: THE SPLIT-K EMISSION WORKSPACE -- single-writer slots, never an atomic
  //: target. Work item `w`'s contribution to token `t` lands at
  //: `emit_num[(w*T + t)*DV + ch]` (and `emit_den[w*T + t]` under `global`);
  //: `emit_rowmask` bit `t` of entry `(w, tile)` -- host-zeroed -- says the row
  //: was emitted, so the reduce reads exactly the bytes the producer wrote.
  //: `num`/`den` themselves are written by `emit_reduce_kernel` in `consumer.cu`,
  //: which folds the contributions in one host-fixed order -- what makes `y`
  //: bit-identical across runs. docs/internals/consumer.md#emit-workspace
  float* emit_num;         // [n_work, T, DV] fp32
  float* emit_den;         // [n_work, T]     fp32, or nullptr under `raw`
  uint32_t* emit_rowmask;  // [n_work, n_tiles], zero-initialized by the host
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
  int n_tiles;
  int n_words;
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
  //: `is_write`. docs/internals/consumer.md#pass-mode
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

template <int BT, int BC, int DV, int D, MmaPrec PREC, bool DECAY, bool GLOBAL>
void launch_consumer_arm(const ConsumerParams& p, cudaStream_t stream);

template <int BT, int BC, int DV, int D, MmaPrec PREC, bool DECAY, bool GLOBAL>
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
  int BT;
  int BC;
  int DV;
  int D;
  MmaPrec prec;
  bool decay;
  bool global_norm;
};

//: THE MACHINE FACTS A ROW IS ADMITTED AGAINST. Filled from the driver once per
//: process. Held as a struct rather than passed as a scalar so that a second fact
//: (a register budget, a driver version) is a field and not a signature change.
//: docs/internals/consumer.md#row-admission
struct DeviceFacts {
  size_t max_dynamic_smem_bytes;
};

struct ShardRow {
  int BT;
  int BC;
  int DV;
  int D;
  MmaPrec prec;
  bool decay;
  bool global_norm;
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
