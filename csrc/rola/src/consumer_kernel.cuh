#pragma once

// THE RoLA V3 CONSUMER KERNEL -- THE ENDGAME EXECUTOR. It is the ONLY consumer, and
// this file is its BODY: the membership scan, the group assembler, the
// compact-into-registers atom walk, the four GEMMs in their phase order and the
// split-K epilogue. `ConsumerParams` and the host dispatch are `consumer.{cuh,cu}`;
// the supported shape surface is declared at `launch_consumer` and is deliberately
// NOT restated here.
//
// THE EXECUTOR'S SHAPE (P14 endgame form -- the schedule IS the tiling):
//
//   * MEMBERSHIP = STREAMED FINE SCAN. For each 32-token word of the segment's
//     range: AND the owner's fixed-level token-bit rows, AND the token-granular
//     slice-union bit row. There is no tile-candidacy question and no coarse
//     pre-filter (intersections do not survive coarsening -- AND-consumers read
//     fine); the scan streams `(j + 1)` words per 32 tokens, static addresses.
//   * GROUPS ARE ALWAYS FULL: members ballot into groups of `kGroupTokens == 16`
//     (the MMA quantum), stable ascending -- ORDER IS THE CAUSALITY REQUIREMENT.
//     The only ragged group is each segment's final flush. Visit counts are exact:
//     `ceil(gathered / 16)`. Dense inputs self-heal to IDENTITY GROUPING (all-ones
//     membership -> groups are consecutive 16-token blocks).
//   * THE ATOM WALK: the per-token atom-granule union bytes (schedule form (b))
//     are OR'd over the group's <= 16 members at ballot time into per-side ATOM
//     MASKS; every leaf-axis consumer (read-panel build, readout, Gram, mass
//     column, write-panel build, fold) walks only LIVE atoms. `mask == FULL`
//     selects the statically-unrolled dense arm -- dense pays no mask
//     instructions. Skipping is EXACT, not approximate: a dead atom's
//     coefficients are exact zeros (self-masking), and under mass decay a dead
//     atom's leaves have a zero clock prefix, hence survival factor s0 == 1 --
//     TICKS RIDE WRITES, which is the architecture fact that makes the endgame
//     legal under decay.
//   * WHOLE-COVERED-SPAN FETCH: a member token's covered amplitude span is read
//     by position (no support inspection anywhere below the ballot); zeros ride
//     the self-masking theorem.
//   * THE AMPLITUDE STREAMS ARE PREFETCHED. Both panels' coefficients come out of a
//     SHARED-MEMORY SLAB filled by `cp.async` from inside the PREVIOUS window's
//     visit, so the gather's DRAM latency leaves the visit's critical path entirely
//     and is not hidden by co-residency. A chunk no live atom reads is issued with
//     src-size zero -- the hardware zero-fills it, which is exactly the value a dead
//     atom's coefficient must take. Zero new barriers in the steady state: the issue
//     rides the staging barrier and the drain rides the end-of-window barrier.
//
// THE ONE PRECISION STORY: bf16 storage, fp32 accumulation, fp64 reference as
// oracle. The staged operands, amplitude streams and value stream are bf16; the
// state, accumulators, clock, row scales AND the decay panel application are
// fp32. (U07-d's fp16 application was built and CARVED on measurement: an
// admitted hot routing put the survival factor past fp16's 65504 range --
// measured NaN, not rounding. bf16 amplitude STORAGE on the decay arms
// survives the carve; the register cost is the ratchet's to adjudicate.)
//
// Three facts a reader must not miss:
//   1. THE STATE LIVES IN REGISTERS, TRANSPOSED, AS THE FOLD'S OWN D FRAGMENTS.
//      `mma.sync`'s D map and its A map are the same map, so holding the state
//      `[channel][leaf]` makes the fold's accumulator the readout's A operand
//      for free: the fold's `mma.sync` accumulates IN PLACE (`D == C == S^T`)
//      and the readout consumes the result with no load at all. A warp owns one
//      16-channel tile and a contiguous atom range of leaves, which is what makes
//      the writer and the reader of an element the same warp at the same lane.
//      Only the MASS COLUMN keeps a shared-memory home -- fp32, `[BC]` -- because
//      it is genuinely shared between warps. Accumulation is fp32 throughout; the
//      one narrowing is the state's bf16 A-operand fragment at the readout.
//   2. THE ONE-COPY INVARIANT IS A PROPERTY OF THE LAUNCH SHAPE: one CTA per
//      owner, run to completion, one HBM read and one HBM write of the owner's
//      state block. There is no wave machinery anywhere.
//   3. EVERY STRUCTURAL AXIS IS A TEMPLATE PARAMETER and no axis has a runtime
//      fallback -- one comprehensive path ships. The endgame walk IS the path;
//      there is no aligned walk (dense is its degeneracy, by data).
//
// The order of operations, the symbol glossary, the bank-conflict proofs, the
// deadlock argument and the hazard index: docs/internals/consumer_kernel.md

#include <cuda_bf16.h>

#include "arch_table.cuh"
#include "cp_async.cuh"
#include "intra_panel.cuh"
#include "consumer.cuh"
//: The two layers, in one include: `mma_tile.cuh` -> `smem_layout.cuh`.
//: docs/internals/mma_tile.md#cute-pin
#include "mma_tile.cuh"
#include "readout_mma.cuh"

namespace rola {
namespace consumer {

//: §12 rules 1 and 3's survival factors, taken from `intra_panel.cuh` rather than
//: reimplemented: it is the one clamp point and the one exponent sign convention.
using rola::panel_read_factor;
using rola::panel_write_factor;

//: §4.5's PAGED ADDRESS, RE-KEYED TO THE ATOM. A page is ONE `MMA_K_QUANTUM`-leaf
//: ATOM, not one owner, so an owner spans `QPO` INDEPENDENTLY PAGED atoms whose slots
//: are not contiguous in general and a CTA resolves `QPO` bases at startup. Returns
//: the first STATE ROW of atom `q` of `owner`, or `-1` when the atom has no committed
//: slot -- which under total admission means no write candidate at all, hence a dense
//: state that is identically zero. ONE EXPRESSION SERVES BOTH LAYOUTS: unpaged is the
//: identity slot map. docs/internals/consumer_kernel.md#atom-row-base
template <int BC>
__device__ __forceinline__ int64_t atom_row_base(const ConsumerParams& p, int bh, int owner,
                                                 int q) {
  constexpr int kAtomsPerOwner = BC / kMmaKQuantum;
  if (p.page_table != nullptr) {
    //: DERIVED rather than carried: the host sets `p.N = owners_total * BC` under
    //: paging, so there is no second place for the table's stride to be wrong.
    const int atoms_per_bh = static_cast<int>(p.N / kMmaKQuantum);
    const int64_t entry =
        static_cast<int64_t>(bh) * atoms_per_bh + static_cast<int64_t>(owner) * kAtomsPerOwner + q;
    const int physical = p.page_table[entry];
    return physical >= 0 ? static_cast<int64_t>(physical) * kMmaKQuantum : static_cast<int64_t>(-1);
  }
  return static_cast<int64_t>(bh) * p.N + static_cast<int64_t>(owner) * BC
         + static_cast<int64_t>(q) * kMmaKQuantum;
}

#define CONSUMER_D_ROWS(lane) const int _g = (lane) >> 2, _r = (lane) & 3

//: **THE FORCING-CENSUS INSTRUMENT, NEVER A SHIPPING KNOB.** When defined it
//: replaces the DERIVED launch bound with a fixed CTA count on every instantiation,
//: so a candidate register budget can be censused over the whole dispatch matrix in
//: one probe compile. `tools/ratify.py --census` is its only legitimate definer and
//: a manifest is only ever ratified off the UNFORCED compile:
//: docs/internals/consumer_kernel.md#force-ctas
#ifdef ROLA_FORCE_CTAS
#define ROLA_CONSUMER_CTAS_EXPR (ROLA_FORCE_CTAS)
#else
#define ROLA_CONSUMER_CTAS_EXPR                                       \
  ctas_for_residency(consumer_smem_bytes<BC, DV, D, DECAY, GT>(),     \
                     consumer_reg_budget(BC, DECAY, DV, D, GT))
#endif
template <int BC, int DV, int D, bool DECAY, int GT>
__global__ __launch_bounds__(kConsumerThreads, ROLA_CONSUMER_CTAS_EXPR) void
consumer_forward_kernel(ConsumerParams p) {
  //: THE WINDOW: one VISIT consumes `W` producer groups against ONE entry state.
  //: `kGT` is the window's token extent; `kGroupTokens` remains the group's, and
  //: every producer-side quantity below is still the group's.
  constexpr int W = window_groups(GT);
  constexpr int kGT = GT;
  static_assert(GT % kGroupTokens == 0 && W >= 1, "the window is a whole number of groups");
  constexpr int NT_GM = kGroupTokens / 8;  // n8 tiles of one Gram sub-block
  extern __shared__ char smem_raw[];

  // ---- THE SMEM CARVE. Every region, its pitch and its measured residency cost:
  // ---- docs/internals/smem_layout.md#smem-ledger
  constexpr int GMP = kGT + 8;  // Gm's row pitch (bf16) -- the WINDOW Gram
  constexpr int CKP = BC + 4;   // clk's row pitch, token-major over the window
  constexpr int kQPO = BC / kMmaKQuantum;
  constexpr int kWarps = kConsumerThreads / 32;
  static_assert(BC % kMmaKQuantum == 0, "BC must be a whole number of MMA k-quanta");
  //: The atom masks' FULL value: every atom of the owner's slice live this group.
  //: The CTA-uniform `mask == kFullMask` test is what selects the statically
  //: unrolled dense arm -- dense inputs pay zero mask instructions.
  constexpr unsigned kFullMask = (kQPO >= 32) ? 0xffffffffu : ((1u << kQPO) - 1u);
  //: **THE STATE'S RESIDENCY GEOMETRY.** `kMT` channel tiles (the MMA `m` axis),
  //: `kReps` leaf splits, `kAPW` atoms and `kStRegs` fp32 registers per owning warp
  //: per lane: docs/internals/smem_layout.md#state-residency
  constexpr int kMT = state_m_tiles(DV);
  constexpr int kReps = state_reps(BC, DV);
  constexpr int kAPW = state_atoms_per_warp(BC, DV);
  constexpr int kOwnWarps = state_owner_warps(BC, DV);
  constexpr int kStRegs = state_regs_per_lane(BC, DV);
  static_assert(kOwnWarps <= kWarps && kAPW * kReps == kQPO,
                "the state's ownership rule must tile the owner's atoms exactly");
  //: `Rp`, `Wp` and `Vp` are `W` SLOTS OF THE GROUP'S SHAPE. Their per-slot strides
  //: are what the window's GEMMs index by; every layout proof they carry is a
  //: statement about ONE slot and is therefore untouched by the window.
  constexpr int kWpPitch = leaf_pitch(BC);
  constexpr int kRpSlotElems = BC * kTokPitch;
  constexpr int kWpSlotElems = kGroupTokens * kWpPitch;
  constexpr int kVpSlotElems = DV * kTokPitch;
  constexpr int kSlots = sched_slots(GT);
  //: THE PREFETCH SLAB's row pitch and 16-byte chunk count, both compile-time worst
  //: cases over the owner rectangle's factorisations: docs/internals/smem_layout.md#prefetch-slab
  constexpr int kPfPitch = amp_slab_pitch(BC, D);
  constexpr int kPfChunks = amp_slab_chunks(BC, D);

  //: **THE STATE IS NOT IN THIS CARVE.** Its numerator lives in `St` below -- the
  //: fold's own D fragments, held in registers for the whole kernel. What remains here
  //: is the MASS COLUMN: fp32, `[BC]`, the denominator of a ratio whose consumers are
  //: broadcast matvecs, read by the readout's term 1 and written by the fold's term 3
  //: from different warps, so it is the one part of the state that is genuinely shared.
  //: docs/internals/consumer_kernel.md#state-residency
  float* const Smass = reinterpret_cast<float*>(smem_raw);
  constexpr size_t kMassBytes = static_cast<size_t>(BC) * sizeof(float);
  char* carve = smem_raw + kMassBytes;
  __nv_bfloat16* Rp = reinterpret_cast<__nv_bfloat16*>(carve);
  carve += static_cast<size_t>(W) * kRpSlotElems * sizeof(__nv_bfloat16);
  __nv_bfloat16* Wp = reinterpret_cast<__nv_bfloat16*>(carve);
  carve += static_cast<size_t>(W) * kWpSlotElems * sizeof(__nv_bfloat16);
  constexpr size_t kVwRegionBytes = vw_union_bytes<BC, DV, D, DECAY, GT>();
  //: `ldmatrix` TAKES A 16-BYTE-ALIGNED ROW ADDRESS, so every staged base and every
  //: staged ROW must be 16-byte aligned -- a requirement no pitch predicate implies.
  //: Stated, not trusted: docs/internals/consumer_kernel.md#ldsm-alignment
  constexpr size_t kRpOffsetBytes = kMassBytes;
  constexpr size_t kWpOffsetBytes =
      kRpOffsetBytes + static_cast<size_t>(W) * kRpSlotElems * sizeof(__nv_bfloat16);
  constexpr size_t kVpOffsetBytes =
      kWpOffsetBytes + static_cast<size_t>(W) * kWpSlotElems * sizeof(__nv_bfloat16);
  static_assert(kRpOffsetBytes % 16 == 0 && kWpOffsetBytes % 16 == 0 && kVpOffsetBytes % 16 == 0,
                "Rp, Wp and Vp must be 16-byte aligned for ldmatrix");
  static_assert((static_cast<size_t>(kRpSlotElems) * sizeof(__nv_bfloat16)) % 16 == 0
                    && (static_cast<size_t>(kWpSlotElems) * sizeof(__nv_bfloat16)) % 16 == 0
                    && (static_cast<size_t>(kVpSlotElems) * sizeof(__nv_bfloat16)) % 16 == 0,
                "every staged slot base must be 16-byte aligned for ldmatrix");
  __nv_bfloat16* Vp = reinterpret_cast<__nv_bfloat16*>(carve);
  //: `clk` ALIASES `Vp`'s bytes (dead before `Vp` is live; the reservation is the
  //: byte-max over the two tenants). Declared AT THE CARVE because the alias is the
  //: load-bearing fact. docs/internals/smem_layout.md#vw-clk-union
  float* clk = reinterpret_cast<float*>(carve);
  carve += kVwRegionBytes;
  int* dig = reinterpret_cast<int*>(carve);  // [D][BC]
  carve += static_cast<size_t>(D) * BC * sizeof(int);
  __nv_bfloat16* Gm = reinterpret_cast<__nv_bfloat16*>(carve);
  //: `Gm` IS AN `ldmatrix` SOURCE (the intra term's B and the intra mass term's), so
  //: its base carries the same 16-byte requirement the panels do, and for the same
  //: reason. It holds `G^T`: sub-block `(b, a)` starts at
  //: `Gm + b * kGroupTokens * GMP + a * kGroupTokens`, and
  //: `pitched_ldsm_banks_ok<16, 16, GMP>` below licenses that base at this pitch.
  constexpr size_t kGmOffsetBytes =
      kVpOffsetBytes + kVwRegionBytes + static_cast<size_t>(D) * BC * sizeof(int);
  static_assert(kGmOffsetBytes % 16 == 0, "Gm must be 16-byte aligned for ldmatrix");
  carve += static_cast<size_t>(kGT) * GMP * sizeof(__nv_bfloat16);
  float* log_keep = reinterpret_cast<float*>(carve);
  float* p_ref = log_keep + (DECAY ? BC : 0);
  float* s0 = p_ref + (DECAY ? BC : 0);
  //: THE COOKED SCHEDULE RING. `kSchedSlots` group descriptors
  //: (`kSchedSlotInts` ints each; the field offsets are
  //: `kSchedGN/RMask/WMask/Flags`), written by the PRODUCER WARP `kSchedDepth`
  //: groups ahead and published by the end-of-visit `__syncthreads()` of the
  //: visit that produced them. Every consumer warp READS the descriptor rather
  //: than deriving it. docs/internals/consumer_kernel.md#sched-ring
  int* const g_ring = reinterpret_cast<int*>(s0 + (DECAY ? BC : 0));
  //: THE PER-OWNER UPPER PRODUCTS, one pair of `kGroupTokens` floats per ring slot: the
  //: read row scale then the write row scale for each of the slot's member
  //: tokens. Cooked by the producer beside the member word (the stream-inventory
  //: amendment) and consumed as `sr` / `sw`.
  //: docs/internals/consumer_kernel.md#upper-product
  float* const g_up = reinterpret_cast<float*>(g_ring + kSlots * kSchedSlotInts);
  static_assert(!DECAY
                    || static_cast<size_t>(kGT) * CKP * sizeof(float)
                           <= vw_union_bytes<BC, DV, D, DECAY, GT>(),
                "clk aliases the Vp region and must fit IN BYTES");
  float* a_state = reinterpret_cast<float*>(Rp);         // [BC], the survival product A
  int* sum_slot = reinterpret_cast<int*>(Rp) + BC;       // [BC]
  int* sum_count = reinterpret_cast<int*>(Rp) + 2 * BC;  // [1]
  static_assert(static_cast<size_t>(kRpSlotElems) * sizeof(__nv_bfloat16)
                    >= (2 * static_cast<size_t>(BC) + 1) * sizeof(int),
                "pass 1's summary state must fit inside the read panel");

  // ---- EACH OPERAND'S PITCH PROVES ITS OWN CONFLICT-FREEDOM.
  // ---- docs/internals/consumer_kernel.md#pitches
  //: **THE PROOFS ARE ABOUT ONE SLOT / ONE SUB-BLOCK**, which is exactly why the window
  //: is `W` slots of the group's shape rather than one wider buffer: `Rp`, `Wp` and
  //: `Vp` keep their group-shaped predicates verbatim, and only `Gm` is asked a new
  //: question -- the same `ldmatrix` phase predicate at the WINDOW's pitch.
  static_assert(pitched_ldsm_banks_ok<BC, kGroupTokens, kTokPitch>(),
                "Rp: the ldmatrix.trans fetch conflicts or is misaligned");
  static_assert(pitched_ldsm_banks_ok<DV, kGroupTokens, kTokPitch>(),
                "Vp: the ldmatrix fetch conflicts or is misaligned");
  static_assert(pitched_ldsm_banks_ok<kGroupTokens, BC, kWpPitch>(),
                "Wp: the ldmatrix fetches conflict or are misaligned");
  static_assert(pitched_ldsm_banks_ok<kGroupTokens, kGroupTokens, GMP>(),
                "Gm: the intra term's ldmatrix.trans fetch conflicts or is misaligned");
  static_assert(pitched_store_banks_ok<BC, kGroupTokens, kTokPitch>(),
                "Rp: the staging store conflicts");
  static_assert(pitched_store_banks_ok<kGroupTokens, BC, kWpPitch>(),
                "Wp: the staging store conflicts");
  //: **`Vp`'s STAGING STORE IS THE ONE COLUMN-STRIDED WRITE IN THE KERNEL AND IT IS
  //: 4-WAY CONFLICTED, BY CONSTRUCTION RATHER THAN BY OVERSIGHT.** The value tile is
  //: read `m`-major (channel is the MMA's `m`) and written token-major (the global
  //: read must stay coalesced across channels), and no row pitch can serve both: an
  //: `ldmatrix` row demands `pitch % 8 == 0`, which forces `pitch / 2` even, which is
  //: exactly what makes a column walk alias banks. The trade is 4 conflicted SMEM
  //: stores per thread per group against a 2x sector count on the DRAM value stream.
  //: docs/internals/consumer_kernel.md#vp-store

  const int work = blockIdx.x;
  const int bh = p.owners[2 * work];
  const int owner = p.owners[2 * work + 1];
  const int seg = (p.work_seg != nullptr) ? p.work_seg[work] : 0;
  const int nseg = (p.work_nseg != nullptr) ? p.work_nseg[work] : 1;
  const int owner_slot0 = (p.work_slot != nullptr) ? p.work_slot[work] : 0;
  const bool summary = (p.pass_mode == kPassSummary);
  const bool last_seg = (seg == nseg - 1);
  //: §6.4's segment partition, on the SCAN STREAM's word axis (the 32-token machine
  //: word is the metadata granule; tile ranges died with the aligned walk).
  const int w_lo = static_cast<int>((static_cast<int64_t>(seg) * p.n_twords) / nseg);
  const int w_hi = static_cast<int>((static_cast<int64_t>(seg + 1) * p.n_twords) / nseg);
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  //: **THE WARP'S STATE OWNERSHIP** -- one CHANNEL TILE and one contiguous ATOM RANGE.
  //: Everything the register-resident state does is indexed off these three values.
  const int mch = warp % kMT;          // the warp's channel tile
  const int hrep = warp / kMT;         // its leaf split index
  const bool owns = warp < kOwnWarps;  // warps beyond the split own no state
  const int atom0 = owns ? hrep * kAPW : 0;
  const int m0 = mch * kMmaKQuantum;  // its first channel
  const unsigned own_mask = ((1u << kAPW) - 1u) << atom0;
  const int _sg = lane >> 2, _sr = lane & 3;

  // ---- owner decomposition (§3's rectangle) --------------------------------
  const int j = p.first_spanned_level;
  const int slice = owner % p.n_slices;
  const int rest = owner / p.n_slices;
  int base[D];
#pragma unroll
  for (int l = 0; l < D; ++l) {
    if (l < j) {
      base[l] = (rest / p.owner_stride[l]) % p.level_width[l];
    } else if (l == j) {
      base[l] = slice * p.slice_digits;
    } else {
      base[l] = 0;
    }
  }

  for (int idx = tid; idx < D * BC; idx += kConsumerThreads) {
    const int l = idx / BC;
    dig[idx] = leaf_digit(p, l, base[l], idx - l * BC);
  }

  // ---- §12 RULE 2: per-leaf log-retention, ONCE PER OWNER, IN LOG SPACE. The
  // ---- re-evaluation of `leaf_digit` (instead of reading `dig`) avoids a barrier:
  // ---- docs/internals/consumer_kernel.md#digit-hoist
  if (DECAY) {
    const int h = bh % p.H;
    for (int i = tid; i < BC; i += kConsumerThreads) {
      float rate = 1.0f;
#pragma unroll
      for (int l = 0; l < D; ++l) {
        rate *= p.dials[static_cast<int64_t>(h) * p.total_rows + p.level_row_offset[l]
                        + leaf_digit(p, l, base[l], i)];
      }
      log_keep[i] = log1pf(-rate);
    }
  }

  constexpr int cols = DV + 1;  // the mass column is unconditional (raw is dead)
  // ---- THE OWNER'S STATE BLOCK, READ FROM HBM ONCE (the one-copy invariant), at
  // ---- §4.5's ATOM granularity. docs/internals/consumer_kernel.md#entry-state
  const bool from_ws = (!summary) && (nseg > 1);
  const float* ws_entry =
      from_ws ? (p.ws_state + static_cast<int64_t>(owner_slot0 + seg) * BC * cols) : nullptr;
  //: **THE ENTRY STATE LANDS IN REGISTERS**, in the fold's own D-fragment map: lane
  //: `(g, r)` of the warp owning channel tile `mch` and atom `q` holds channels
  //: `m0 + g` and `m0 + 8 + g` at leaves `q * 16 + 8 * half + 2r` and `+ 1`. Eight
  //: floats per atom, and the 8 lanes sharing a leaf read 8 CONSECUTIVE channels, so
  //: the HBM read is one 32-byte segment per (leaf, half-warp).
  //: docs/internals/consumer_kernel.md#entry-state
  float St[kStRegs];
#pragma unroll
  for (int v = 0; v < kStRegs; ++v) St[v] = 0.0f;
  if (owns && !summary) {
#pragma unroll
    for (int ai = 0; ai < kAPW; ++ai) {
      const int q = atom0 + ai;
      const int64_t row0 =
          from_ws ? static_cast<int64_t>(q) * kMmaKQuantum : atom_row_base<BC>(p, bh, owner, q);
      if (row0 < 0) continue;
      const float* const src = from_ws ? ws_entry : p.state;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int64_t r0 = row0 + 8 * half + 2 * _sr;
        float* const st = St + ai * 8 + half * 4;
        st[0] = src[r0 * cols + m0 + _sg];
        st[1] = src[(r0 + 1) * cols + m0 + _sg];
        st[2] = src[r0 * cols + m0 + 8 + _sg];
        st[3] = src[(r0 + 1) * cols + m0 + 8 + _sg];
      }
    }
  }
  if (summary || from_ws) {
    for (int i = tid; i < BC; i += kConsumerThreads) {
      Smass[i] = summary ? 0.0f : ws_entry[i * cols + DV];
    }
  } else {
#pragma unroll
    for (int q = 0; q < kQPO; ++q) {
      const int64_t row0 = atom_row_base<BC>(p, bh, owner, q);
      for (int r = tid; r < kMmaKQuantum; r += kConsumerThreads) {
        Smass[q * kMmaKQuantum + r] = (row0 >= 0) ? p.state[(row0 + r) * cols + DV] : 0.0f;
      }
    }
  }
  if (summary) {
    for (int i = tid; i < BC; i += kConsumerThreads) {
      a_state[i] = 1.0f;  // `A`'s identity; `decay = None` never multiplies it
      sum_slot[i] = 0;    // the written-leaf FLAG until the publication rewrites it
    }
  }
  //: The token-major amplitude stream base for this `bh`: a member token's row is
  //: `tok_base + token` (whole-covered-span fetch: the span is read by position).
  const int64_t tok_base = static_cast<int64_t>(bh) * p.n_twords * 32;
  //: The slice's ATOM-MAJOR bitlist base: row `q`'s word for source word `w` is
  //: `ab_base + q * n_twords + w` -- one CONTIGUOUS token-bit row per atom.
  const int64_t ab_base = (static_cast<int64_t>(bh) * p.n_slices + slice) * kQPO * p.n_twords;

  // ---- THE PRODUCER. WARP 0 ALONE runs the membership scan and the ballot, and
  // ---- publishes COOKED GROUP DESCRIPTORS into `g_ring`. The scan's result is an
  // ---- OWNER-LEVEL fact, identical across the owner's atoms and across the CTA's
  // ---- warps, so it is cooked ONCE PER OWNER -- the multiplicity rule, and the
  // ---- issue cut this kernel's measured constraint asks for.
  // ---- docs/internals/consumer_kernel.md#producer
  static_assert(kQPO <= 32, "one producer lane per atom: the atom mask is ONE ballot");
  //: THE PRODUCER'S CURSOR STATE, RESTING IN SMEM BETWEEN COOKS. Scheduler state
  //: held in registers is live across the whole visit body and competes with the
  //: four GEMMs' working set -- measured as spills on the `BC = 16, decay = 0`
  //: key, the tightest register budget in the matrix, on both arches. The AND and
  //: the ballot still happen in registers; only the resting place is SMEM.
  //: docs/internals/consumer_kernel.md#producer-state
  int* p_st = reinterpret_cast<int*>(g_up + kSlots * 2 * kGroupTokens);
  //: [0] p_cur, [1] p_chunk_base, [2] p_live, [3] p_word, [4] p_rem, [5] p_gfr,
  //: [6] p_gfw, then the PER-LANE chunk (fr, fw, mem) and ingredient (ra, wa)
  //: planes, one word per lane each.
  int* p_chunk = p_st + 8;  // [5][32]: fr, fw, mem, ra, wa
  //: THE PREFETCH'S PER-OWNER CHUNK DESCRIPTORS. One entry per 16-byte chunk of a
  //: slab row: the SPANNED LEVEL it belongs to, its element offset inside that
  //: level's amplitude row (already aligned down to the 16-byte boundary), and the
  //: SET OF ATOMS whose digits it carries -- which is what makes the copy per-atom.
  //: Built once per owner because the owner's digit rectangle is fixed for the whole
  //: kernel; only the ATOM MASK the set is tested against varies per group.
  int* const ch_lvl = p_chunk + 5 * 32;
  int* const ch_src = ch_lvl + kPfChunks;
  int* const ch_atoms = ch_src + kPfChunks;
  //: THE TWO SLABS, read and write, `kGT` token rows each. They are the LAST regions
  //: of the carve so that `cp.async`'s 16-byte destination alignment is ONE align-up
  //: rather than a constraint every region before them would have to carry.
  constexpr size_t kPfBase = align16(consumer_smem_core_bytes<BC, DV, D, DECAY, GT>());
  __nv_bfloat16* const pfR = reinterpret_cast<__nv_bfloat16*>(smem_raw + kPfBase);
  __nv_bfloat16* const pfW = pfR + kGT * kPfPitch;
  static_assert(kPfBase % 16 == 0 && (kGT * kPfPitch * sizeof(__nv_bfloat16)) % 16 == 0,
                "both prefetch slab bases must be 16-byte aligned for cp.async");
  if (tid == 0) {
    p_st[0] = w_lo;
#pragma unroll
    for (int q = 1; q < 8; ++q) p_st[q] = 0;
  }
  if (warp == 0) {
#pragma unroll
    for (int q = 0; q < 5; ++q) p_chunk[q * 32 + lane] = 0;
  }

  // ---- THE PREFETCH SLAB'S OWNER-LEVEL GEOMETRY. A level's digits over the owner's
  // ---- `BC` leaves are the CONTIGUOUS range `[base_l, base_l + span_l)` -- the value
  // ---- set of `(i / stride) % modulus` over `i < BC` is either `0 .. (BC-1)/stride`
  // ---- or, on wrap, all of `[0, modulus)`. Each spanned level takes that range,
  // ---- aligned down to a 16-byte boundary and rounded up to one, as a sub-row of the
  // ---- slab; `pf_off[l]` is the one number a gather adds to its digit.
  // ---- **`pf_off` IS A REGISTER ARRAY, NOT A TABLE**, and that is the whole reason
  // ---- this gather is cheaper than the one it replaces: every gather's level loop is
  // ---- `#pragma unroll`, so `l` is a compile-time constant at every use and the
  // ---- offset never touches shared memory. `Plan.amplitude_pitch` is what licenses
  // ---- it -- the padded pitch makes a level's per-token lead a CONSTANT rather than
  // ---- a function of the token. docs/internals/consumer_kernel.md#prefetch
  int pf_off[D];
  int pf_nch = 0;
  {
    int at = 0;
#pragma unroll
    for (int l = 0; l < D; ++l) {
      pf_off[l] = 0;
      if (l < j) continue;
      const int stride_l = p.level_stride[l];
      const int mod_l = p.level_modulus[l];
      const int span_l = min(mod_l, (BC - 1) / stride_l + 1);
      const int lead_l = base[l] & 7;
      pf_off[l] = at + lead_l - base[l];
      at += ((lead_l + span_l + 7) / 8) * 8;
    }
    pf_nch = at >> 3;
  }
  //: THE CHUNK DESCRIPTORS, SERIAL ON ONE THREAD. `kPfChunks` is at most 12 and the
  //: work is once per owner; spreading it would cost a barrier to publish `at`.
  if (tid == 0) {
    int at = 0;
#pragma unroll
    for (int l = 0; l < D; ++l) {
      if (l < j) continue;
      const int stride_l = p.level_stride[l];
      const int mod_l = p.level_modulus[l];
      const int span_l = min(mod_l, (BC - 1) / stride_l + 1);
      const int lead_l = base[l] & 7;
      const int nc = (lead_l + span_l + 7) / 8;
      for (int c = 0; c < nc; ++c) {
        //: The chunk's digit range, in offsets from `base[l]`.
        const int c_lo = c * 8 - lead_l;
        unsigned bits = 0u;
        //: ATOM `q`'s digit range at this level. `(i / stride)` over the atom's 16
        //: leaves is the contiguous `[u0, u1]`; taken modulo the level's modulus it
        //: stays contiguous unless it wraps, and a wrap is answered with the whole
        //: range -- a SUPERSET only ever fetches a chunk that could have been
        //: zero-filled, which is safe in the one direction that matters.
#pragma unroll
        for (int q = 0; q < kQPO; ++q) {
          const int u0 = (q * kMmaKQuantum) / stride_l;
          const int u1 = (q * kMmaKQuantum + kMmaKQuantum - 1) / stride_l;
          int lo = 0, hi = mod_l - 1;
          if (u1 - u0 < mod_l - 1) {
            const int a0 = u0 % mod_l, a1 = u1 % mod_l;
            if (a0 <= a1) {
              lo = a0;
              hi = a1;
            }
          }
          if (hi > span_l - 1) hi = span_l - 1;
          if (!(hi < c_lo || lo >= c_lo + 8)) bits |= 1u << q;
        }
        ch_lvl[at + c] = l;
        ch_src[at + c] = (base[l] & ~7) + c * 8;
        ch_atoms[at + c] = static_cast<int>(bits);
      }
      at += nc;
    }
  }
  //: The scan's fixed-level row index is `p.level_row_offset[l] + base[l]`, formed
  //: AT THE POINT OF USE: `base[]` is already live across the whole kernel (the
  //: panels and the upper product all index it), so a hoisted copy would be `D`
  //: extra live registers on the tightest key.

  //: ONE COOKED GROUP, PUBLISHED TO `slot`. WARP 0 ONLY, and EVERY LANE ENTERS --
  //: the `__ballot_sync` / `__shfl_sync` below are full-warp collectives and the
  //: loop's control flow is warp-uniform by construction (`p_rem`, `p_live`,
  //: `p_cur`, `fill` are all warp-uniform; only the chunk and ingredient
  //: values are per-lane). The cursor is LOADED here and STORED at the end, so it
  //: is register-live only for the duration of the cook.
  auto produce_group = [&](int slot) {
    int* R = g_ring + slot * kSchedSlotInts;
    //: HAZARD producer-cursor, the LOAD's half -- docs/internals/consumer_kernel.md#producer-cursor
    __syncwarp();
    int p_cur = p_st[0];
    int p_chunk_base = p_st[1];
    unsigned p_live = static_cast<unsigned>(p_st[2]);
    int p_word = p_st[3];
    uint32_t p_rem = static_cast<uint32_t>(p_st[4]);
    //: THE COOK CARRIES NO PER-LANE REGISTER STATE -- only the uniform scalars
    //: above, which is what keeps the tightest keys inside their budget. The chunk
    //: planes are read ONE SCALAR AT A TIME at word selection, by the uniform index
    //: the ballot picked (the value a `__shfl` of a register plane would have
    //: produced); the selected word's fixed-level AND results are recovered the
    //: same way, its chunk lane being `p_word - p_chunk_base`. Each lane reads its
    //: OWN ingredient slot, so those carry no cross-lane traffic and need no
    //: reconvergence.
    int fill = 0;
    unsigned rmask = 0u, wmask = 0u;
    bool gr = false, gw = false;
    while (fill < kGroupTokens) {
      if (p_rem == 0u) {
        //: THE CHUNK, COOKED IN REGISTERS (contract item 2): one word per lane.
        //: Lane `w` ANDs the owner's `j` fixed-level token rows per side and the
        //: token-granular slice-union word -- the "k+1 per-digit bit-streams,
        //: identically indexed, one shared cursor structure" -- into its own
        //: `(fr, fw, member)`. The chunk's LIVE-WORD SET is then one ballot, and
        //: skipping dead words costs an `__ffs`. A refill needs no CTA barrier:
        //: the cook is warp-local and its collectives are warp-level.
        while (p_live == 0u && p_cur < w_hi) {
          const int w = p_cur + lane;
          uint32_t fr = 0xffffffffu, fw = 0xffffffffu, mem = 0u;
          if (w < w_hi) {
            const int64_t slab = (static_cast<int64_t>(bh) * p.n_twords + w) * p.total_rows;
#pragma unroll
            for (int l = 0; l < D; ++l) {
              if (l >= j) break;
              const int row_l = p.level_row_offset[l] + base[l];
              fr &= p.read_tok_bits[slab + row_l];
              fw &= p.write_tok_bits[slab + row_l];
            }
            const uint32_t un =
                p.slice_union_tok[(static_cast<int64_t>(bh) * p.n_slices + slice) * p.n_twords + w];
            //: The summary pass ballots the WRITE side alone (union-visit
            //: semantics on one side); `summary` is LAUNCH-UNIFORM.
            mem = (summary ? fw : (fr | fw)) & un;
          }
          //: The cooked chunk goes STRAIGHT to its planes. `fr`/`fw`/`mem` are
          //: transient here and die at the end of this iteration.
          p_chunk[0 * 32 + lane] = static_cast<int>(fr);
          p_chunk[1 * 32 + lane] = static_cast<int>(fw);
          p_chunk[2 * 32 + lane] = static_cast<int>(mem);
          p_chunk_base = p_cur;
          p_cur += kSchedChunkWords;
          p_live = __ballot_sync(0xffffffffu, mem != 0u);
          //: HAZARD producer-chunk -- docs/internals/consumer_kernel.md#producer-chunk
          __syncwarp();
        }
        if (p_live == 0u) break;  // the segment's stream end
        const int sel = __ffs(static_cast<int>(p_live)) - 1;
        p_live &= p_live - 1u;
        p_word = p_chunk_base + sel;
        p_rem = static_cast<uint32_t>(p_chunk[2 * 32 + sel]);
        //: Form (b)'s ingredient, ONE LANE PER ATOM -- which is what turns the
        //: base's `2 * QPO` per-warp tests into the two ballots below. Loaded
        //: ONCE PER SOURCE WORD, not once per ballot round.
        p_chunk[3 * 32 + lane] =
            (lane < kQPO)
                ? static_cast<int>(
                      p.read_atom_bits[ab_base + static_cast<int64_t>(lane) * p.n_twords + p_word])
                : 0;
        p_chunk[4 * 32 + lane] =
            (lane < kQPO)
                ? static_cast<int>(
                      p.write_atom_bits[ab_base + static_cast<int64_t>(lane) * p.n_twords + p_word])
                : 0;
      }
      const int room = kGroupTokens - fill;  // >= 1: a full group ran and reset `fill`
      uint32_t taken = p_rem;
      if (__popc(p_rem) > room) {
        //: the (room+1)-th set bit; bits below it are exactly the lowest `room`
        //: live tokens -- the remainder waits for the next group.
        const unsigned cut = __fns(p_rem, 0, room + 1);
        taken = p_rem & ((1u << cut) - 1u);
      }
      if (((taken >> lane) & 1u) != 0u) {
        R[fill + __popc(taken & ((1u << lane) - 1u))] = p_word * 32 + lane;
      }
      //: Form (b)'s consumption, word-wise: atom `q` is live for this group iff
      //: its bitlist row intersects the taken word. ONE BALLOT PER SIDE.
      rmask |= __ballot_sync(
          0xffffffffu,
          (lane < kQPO) && ((static_cast<uint32_t>(p_chunk[3 * 32 + lane]) & taken) != 0u));
      wmask |= __ballot_sync(
          0xffffffffu,
          (lane < kQPO) && ((static_cast<uint32_t>(p_chunk[4 * 32 + lane]) & taken) != 0u));
      const int p_sel = p_word - p_chunk_base;
      gr = gr || (static_cast<uint32_t>(p_chunk[0 * 32 + p_sel]) & taken) != 0u;
      gw = gw || (static_cast<uint32_t>(p_chunk[1 * 32 + p_sel]) & taken) != 0u;
      fill += __popc(taken);
      p_rem ^= taken;
    }
    // ---- THE PER-OWNER UPPER PRODUCT (the stream-inventory amendment). The
    // ---- FIXED levels' amplitudes are CONSTANT ALONG THE LEAF AXIS -- `j`
    // ---- scalars per member token per side, common to every atom's Kronecker
    // ---- coefficient -- so their per-token product is an OWNER-LEVEL fact and
    // ---- obeys the same multiplicity rule membership does: computed ONCE PER
    // ---- OWNER, never per atom. These are §3 step 3's ROW SCALES, and they ride
    // ---- the COOKED RING SLOT: the cook already owns the member token list, so
    // ---- they cost no extra stream and arrive PRE-MULTIPLIED a visit early,
    // ---- which lifts `2 * j` fp32 gathers off the visit's critical path.
    // ---- docs/internals/consumer_kernel.md#upper-product
    //: Predicated per side, so a side-dead group FETCHES nothing -- but it still
    //: PUBLISHES exact zeros. **THE PUBLICATION IS UNCONDITIONAL AND THAT IS A
    //: CORRECTNESS REQUIREMENT AT `W > 1`**: a window's side flags are the OR over
    //: its groups, so a side-dead group's `sw` row is read (as the broadcast B of
    //: the fold's mass term) even though its own `Wp` rows are exact zeros -- and
    //: `0 * NaN` is `NaN`, so a stale slot from a previous window is not inert.
    //: Zeroing here is what keeps the 0-times-dead discipline exact.
    {
      //: lanes below `fill` read back token indices OTHER lanes wrote above.
      __syncwarp();
      const bool need_r = !summary && gr && rmask != 0u;
      const bool need_w = gw && wmask != 0u;
      float* U = g_up + slot * 2 * kGroupTokens;
      const int tok = (lane < fill) ? R[lane] : 0;
      const int64_t tr = tok_base + tok;
      float rs = 0.0f, ws = 0.0f;
      //: A ragged group's padded slots take EXACT ZERO scales, which is what
      //: makes their coefficient rows inert through every GEMM and the mass
      //: fold's `sw[t]` term vanish.
      if (lane < fill) {
        rs = need_r ? 1.0f : 0.0f;
        ws = need_w ? p.g_write[static_cast<int64_t>(bh) * p.T + tok] : 0.0f;
#pragma unroll
        for (int l = 0; l < D; ++l) {
          if (l >= j) break;
          //: Fixed levels (`l < j`) are ALWAYS fp32 (Option B) -- the one place
          //: the amplitude dtype is not the arm's storage type.
          if (need_r) {
            rs *= static_cast<const float*>(p.amp_read[l])[tr * p.level_pitch_read[l] + base[l]];
          }
          if (need_w) {
            ws *= static_cast<const float*>(p.amp_write[l])[tr * p.level_pitch_write[l] + base[l]];
          }
        }
      }
      if (lane < kGroupTokens) {
        U[lane] = rs;
        U[kGroupTokens + lane] = ws;
      }
    }
    //: `fill == 0` is the stream-end sentinel every consumer breaks on; the
    //: final flush is the ragged group (`fill < kGroupTokens`).
    if (lane == 0) {
      R[kSchedGN] = fill;
      R[kSchedRMask] = static_cast<int>(rmask);
      R[kSchedWMask] = static_cast<int>(wmask);
      R[kSchedFlags] = (gr ? 1 : 0) | (gw ? 2 : 0);
    }
    //: THE CURSOR GOES BACK TO SLEEP IN SMEM -- nothing above stays live across
    //: the visit body.
    //: HAZARD producer-cursor, the STORE's half -- docs/internals/consumer_kernel.md#producer-cursor
    __syncwarp();
    if (lane == 0) {
      p_st[0] = p_cur;
      p_st[1] = p_chunk_base;
      p_st[2] = static_cast<int>(p_live);
      p_st[3] = p_word;
      p_st[4] = static_cast<int>(p_rem);
    }
  };

  // ---- THE PREFETCH ISSUE. Fills BOTH slabs for window `wnext` out of the cooked
  // ---- descriptors the producer published a window earlier. One `cp.async` per
  // ---- (window token, chunk, side); nothing here reads the amplitude data and
  // ---- nothing waits.
  // ----
  // ---- **THE DELEGATED CHOICE (P25 contract item 7), MADE HERE.** On sm_86 the
  // ---- `cp.async` commit/wait state is PER WARP, so the question was warp-
  // ---- partitioned streams against interleaved commit order. The answer is ONE
  // ---- COMMIT GROUP PER WARP PER VISIT, issued on a FLAT CTA-WIDE INDEX: the visit
  // ---- consumes the whole slab at a single point, so there is exactly one drain and
  // ---- a partial-fill ordering has no consumer that could use it -- interleaved
  // ---- groups would buy bookkeeping and nothing else. The flat stride is what makes
  // ---- a warp's 32 lanes take 32 CONSECUTIVE chunks, so each warp's group is a
  // ---- contiguous byte range, which is the only property the memory system rewards.
  // ----
  // ---- THE PREDICATE IS THE MECHANISM'S PER-ATOM HALF. A chunk is copied iff some
  // ---- atom LIVE ON THAT SIDE IN THAT GROUP holds a digit in it; otherwise it is
  // ---- issued with src-size zero and the hardware zero-fills it. That is EXACT: a
  // ---- leaf reading a zero-filled chunk belongs to an atom dead on that side, whose
  // ---- staged coefficient must be exactly zero, and zeroing one of its level
  // ---- factors delivers exactly that. The ragged tail rides the same predicate,
  // ---- with the token index CLAMPED so no out-of-range address is ever formed.
  // ---- docs/internals/consumer_kernel.md#prefetch
  auto pf_issue = [&](int wnext) {
    const int* const Rn = g_ring + ((wnext * W) % kSlots) * kSchedSlotInts;
    for (int idx = tid; idx < kGT * kPfChunks; idx += kConsumerThreads) {
      const int t = idx / kPfChunks;
      const int cg = idx - t * kPfChunks;
      if (cg >= pf_nch) continue;
      const int* const Ra = Rn + (t / kGroupTokens) * kSchedSlotInts;
      const int tt = t % kGroupTokens;
      const bool in_group = tt < Ra[kSchedGN];
      const int flags = Ra[kSchedFlags];
      //: The STAGED masks -- the side flag already folded in, exactly as the visit
      //: body's `rmr` / `wmr` fold it -- ANDed with the chunk's atom set.
      const unsigned atoms = static_cast<unsigned>(ch_atoms[cg]);
      const unsigned rm = (in_group && !summary && (flags & 1) != 0)
                              ? (atoms & static_cast<unsigned>(Ra[kSchedRMask]))
                              : 0u;
      const unsigned wm =
          (in_group && (flags & 2) != 0) ? (atoms & static_cast<unsigned>(Ra[kSchedWMask])) : 0u;
      const int l = ch_lvl[cg];
      const int64_t tr = tok_base + (in_group ? Ra[tt] : 0);
      const int so = ch_src[cg];
      const int dst = t * kPfPitch + cg * 8;
      cp_async_16(
          pfR + dst,
          static_cast<const __nv_bfloat16*>(p.amp_read[l]) + tr * p.level_pitch_read[l] + so,
          rm != 0u);
      cp_async_16(
          pfW + dst,
          static_cast<const __nv_bfloat16*>(p.amp_write[l]) + tr * p.level_pitch_write[l] + so,
          wm != 0u);
    }
    cp_async_commit();
  };

  //: THE RING'S COLD START -- the producer fills a WHOLE WINDOW of run-ahead before
  //: the first visit, so the steady state never waits on production. The residue the
  //: contract admits (item 6) is exactly this ramp plus the final drain.
  if (warp == 0) {
#pragma unroll 1
    for (int s = 0; s < W * kSchedDepth; ++s) produce_group(s);
  }
  //: THE ENTRY EPOCH, and the ONE barrier the prefetch adds anywhere -- once per
  //: kernel, never in the loop. It publishes the cold-start descriptors and the chunk
  //: descriptors to the threads that issue window 0's copies; the steady state's
  //: issue rides a barrier that already exists.
  __syncthreads();
  pf_issue(0);
  cp_async_wait_all();
  __syncthreads();

  // ---- THE WALK, OVER WINDOWS. The producer's COOKED RING feeds one visit body per
  // ---- WINDOW of `W` groups. Words ascend and each word is consumed LSB-first, so
  // ---- the window's tokens are GLOBALLY ascending ACROSS its groups as well as
  // ---- within one: the window Gram's causal structure is order-relative, so
  // ---- STABILITY IS THE CORRECTNESS REQUIREMENT. Consumers ASSEMBLE NOTHING -- they
  // ---- read the `W` descriptors warp 0 published a window ago, whose publication
  // ---- barrier is the end-of-window `__syncthreads()`.
  // ---- Group `a` of window `widx` is ring slot `widx * W + a` modulo `kSlots`, and
  // ---- `W` divides `kSlots` -- so a window's slots are CONTIGUOUS and a group index
  // ---- is a pure offset, with no per-group modulo and no array of pointers.
  // ---- docs/internals/consumer_kernel.md#group-assembler
  for (int widx = 0;; ++widx) {
    const int slot0 = (widx * W) % kSlots;
    const int* const Rw = g_ring + slot0 * kSchedSlotInts;
    const float* const Uw = g_up + slot0 * 2 * kGroupTokens;
    //: THE VISIT'S GROUP LIST: `tk_of(a)[q]` = the q-th gathered GLOBAL token index
    //: of group `a`, ascending. It is the cooked ring slot's first `kGroupTokens`
    //: ints -- addressed in place rather than copied.
    auto tk_of = [&](int a) { return Rw + a * kSchedSlotInts; };
    auto gN_of = [&](int a) { return Rw[a * kSchedSlotInts + kSchedGN]; };
    auto rm_of = [&](int a) { return static_cast<unsigned>(Rw[a * kSchedSlotInts + kSchedRMask]); };
    auto wm_of = [&](int a) { return static_cast<unsigned>(Rw[a * kSchedSlotInts + kSchedWMask]); };
    //: THE GROUP'S ROW SCALES, likewise addressed in place: `sr` (read) then `sw`
    //: (write), the producer's per-token upper products, read-only to every consumer.
    auto sr_of = [&](int a) { return Uw + a * 2 * kGroupTokens; };
    auto sw_of = [&](int a) { return Uw + a * 2 * kGroupTokens + kGroupTokens; };
    //: THE PER-GROUP SIDE FLAGS. A side whose atom mask is empty is a side with no
    //: live coefficients anywhere in the slice for that group -- skipping it is
    //: exact. They stay PER-GROUP because the GEMMs are: GEMM 1 walks group `a`'s
    //: read mask, a Gram sub-block walks `rm(a) & wm(b)`.
    //: **THE STAGED MASKS -- the ONE pair every consumer walks, held in registers
    //: below as `rmr` / `wmr`.** A group's atom mask and its side flag are
    //: INDEPENDENT bitmaps (the flag is the fixed levels' rows, the mask is the atom
    //: bitlists), so a group can carry live write ATOMS while being side-dead -- and
    //: a side-dead group's `Wp` and `Vp` slots are NOT STAGED. Walking the atom mask
    //: alone would send the fold into a slot holding a PREVIOUS WINDOW's panels,
    //: which at `W == 1` was impossible (the fold ran under the group's own
    //: `is_write`) and at `W > 1` is not. The masks below are therefore the side
    //: flag ALREADY FOLDED IN: `rmr[a] != 0` IS "group `a` staged read
    //: coefficients", and likewise `wmr[a]` for the write side.
    if (gN_of(0) == 0) break;  // the stream's end, as the producer saw it

    //: **THE DESCRIPTORS COME OFF SHARED MEMORY ONCE PER WINDOW AND LIVE IN
    //: REGISTERS FOR IT.** Every buffer in this kernel is carved out of ONE
    //: `extern __shared__` block, so the compiler must assume the fold's stores to
    //: `S` may alias the ring -- and a descriptor field read at its point of use is
    //: then a real SMEM load INSIDE the fold's item loop, re-issued per item and
    //: per slot, that no amount of `const` can hoist. Two masks per group is
    //: `2 * W` registers against that, which is the register policy's own test: a
    //: register that deletes instructions in the hot loop is bought, not leaked.
    //: THE MEMBER COUNT IS NOT AMONG THEM, and that is a measured line rather than
    //: an oversight: `gN` is read only by the STAGING loops, which run before the
    //: fold and are not the aliasing site, so hoisting it bought nothing -- and it
    //: was the register that put `<BC=16, DV=32, D<=3, nodecay>`, the tightest key
    //: in the matrix, over its 64-register budget and into an 8..20 B spill.
    unsigned rmr[W], wmr[W];
    bool w_read = false, w_write = false;
    unsigned w_wmask = 0u;
#pragma unroll
    for (int a = 0; a < W; ++a) {
      const int flags = Rw[a * kSchedSlotInts + kSchedFlags];
      rmr[a] = (!summary && (flags & 1) != 0) ? rm_of(a) : 0u;
      wmr[a] = ((flags & 2) != 0) ? wm_of(a) : 0u;
      if (rmr[a] != 0u) w_read = true;
      if (wmr[a] != 0u) {
        w_write = true;
        w_wmask |= wmr[a];
      }
    }
    const bool mixed = w_read && w_write;

    // ---- THE STAGING, AS A FUNCTION OF THE WINDOW: everything the visit's COMPUTE
    // ---- consumes but does not produce. Slots `[gN, kGroupTokens)` of a ragged
    // ---- group -- and every slot of a window's trailing dead groups -- stage
    // ---- EXACT-ZERO coefficient rows and zero row scales, so they are inert
    // ---- through every GEMM. STAGING IS BRANCHLESS WITHIN A LIVE GROUP (fetch
    // ---- consults nothing -- the frozen contract): dead atoms stage the exact
    // ---- zeros the tensors hold; the ATOM MASKS ARE COMPUTE-ONLY.
    // ---- docs/internals/consumer_kernel.md#staging

    // ---- DECAY, step (a): the clock prefix, `P_ref`, `s0` -- INTO `Vp`'s bytes,
    // ---- guarded by the WINDOW's write side STRUCTURALLY. **THE PREFIX RUNS ACROSS
    // ---- THE WINDOW, NOT ACROSS A GROUP**: that is what puts every one of the
    // ---- window's tokens in ONE reference frame, which is what makes the Gram's
    // ---- cross-group blocks legal under decay. Runs over ALL `BC` leaves
    // ---- (elementwise surplus-ALU work; keeps every leaf's clk/p_ref/s0 FINITE for
    // ---- the panels -- the 0*NaN discipline).
    // ---- docs/internals/consumer_kernel.md#decay-clock
    if (DECAY && w_write) {
      for (int i = tid; i < BC; i += kConsumerThreads) {
        float acc = 0.0f;
        for (int a = 0; a < W; ++a) {
          const int* T = tk_of(a);
          const int gna = gN_of(a);
          for (int t = 0; t < gna; ++t) {
            float cl = 1.0f;
#pragma unroll
            for (int l = 0; l < D; ++l) {
              cl *= p.amp_clock[l][(tok_base + T[t]) * p.level_pitch_clock[l] + dig[l * BC + i]];
            }
            //: HAZARD clock-prefix-fp32 -- docs/internals/consumer_kernel.md#clock-prefix-fp32
            acc += cl;
            clk[(a * kGroupTokens + t) * CKP + i] = acc;
          }
          //: A ragged or dead group's padded slots take a FINITE zero prefix so both
          //: panel factors stay finite (`keep^0 == 1`); their coefficient rows are
          //: exact zeros anyway.
          for (int t = gna; t < kGroupTokens; ++t) clk[(a * kGroupTokens + t) * CKP + i] = 0.0f;
        }
        const float pr = 0.5f * acc;
        p_ref[i] = pr;
        s0[i] = expf(pr * log_keep[i]);
      }
      __syncthreads();
      // ---- DECAY, step (c): the entry state enters the reference frame, ONCE PER
      // ---- WINDOW. `s0` multiplies the STATE -- in REGISTERS, two scalar `s0` loads
      // ---- and four multiplies per 8-leaf tile, where the shared-memory state paid a
      // ---- merge, a scale and a re-split at every element. The mass column rides the
      // ---- same sweep (same recurrence with `v := 1`). Dead atoms' leaves have a zero
      // ---- clock prefix, hence `s0 == 1`, so an unvisited atom is exactly unchanged.
      if (owns) {
#pragma unroll
        for (int ai = 0; ai < kAPW; ++ai) {
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            const int i0 = (atom0 + ai) * kMmaKQuantum + 8 * half + 2 * _sr;
            const float sa = s0[i0], sb = s0[i0 + 1];
            float* const st = St + ai * 8 + half * 4;
            st[0] *= sa;
            st[1] *= sb;
            st[2] *= sa;
            st[3] *= sb;
          }
        }
      }
      for (int i = tid; i < BC; i += kConsumerThreads) Smass[i] *= s0[i];
    }

    // ---- §3 STEP 3's ROW SCALES ARRIVE PRE-MULTIPLIED    }

    // ---- §3 STEP 3's ROW SCALES ARRIVE PRE-MULTIPLIED IN THE RING SLOT, cooked
    // ---- once per owner by the producer a window ago (the stream-inventory
    // ---- amendment). The staging computes and publishes no row scales of its own,
    // ---- so it needs no barrier of its own.
    // ---- docs/internals/consumer_kernel.md#upper-product
    for (int a = 0; a < W; ++a) {
      const int gN = gN_of(a);
      const bool is_read = rmr[a] != 0u;
      const bool is_write = wmr[a] != 0u;
      //: The window token index of this group's slot `t` -- the row `clk` is indexed
      //: by, and the only place the group's position in the window is visible to the
      //: panels at all.
      const int t0 = a * kGroupTokens;
      // ---- THE READ PANEL, `Rp[leaf][token]` (§3 steps 2-3). It is the READOUT's and
      // ---- the Gram's B operand, `k = leaf` on the row axis and `n = token` on the
      // ---- column axis, which is the form `ldmatrix.trans` reads.
      // ---- docs/internals/consumer_kernel.md#read-panel
      if (is_read) {
        __nv_bfloat16* const Rpa = Rp + a * kRpSlotElems;
        //: BRANCHLESS (fetch consults nothing -- the frozen contract): every atom's
        //: slots are staged by the pure affine walk; a dead atom's positions hold the
        //: exact zeros the amplitude tensor already holds (entmax), so the fetch
        //: layer cannot be wrong -- the masks are COMPUTE-ONLY (MMA skip). A padded
        //: token slot takes an exact zero, which is what makes its column inert.
        //: THE fp16 APPLICATION IS CARVED OUT (U07-d built, adjudicated 2026-08-06):
        //: an ADMITTED hot routing (panel bound 23.3 of 40) put the survival factor at
        //: ~e^15+, past fp16's 65504 max -- measured NaN, not a rounding. The panels
        //: apply in fp32 against the bf16 STORAGE (the memory win survives the carve).
        //: Evidence: workflows/perf/p26_evidence/m2_fp16_breach.txt.
        for (int idx = tid; idx < BC * kGroupTokens; idx += kConsumerThreads) {
          const int i = idx / kGroupTokens;
          const int t = idx - i * kGroupTokens;
          float ac = 0.0f;
          if (t < gN) {
            //: THE PREFETCHED ROW. `t0 + t` is the WINDOW token index the slab is keyed
            //: by; `pf_off[l]` is a register at a compile-time `l`, so the whole address
            //: is one 32-bit add over the digit the walk already reads -- where the
            //: direct form paid a 64-bit `tr * pitch` and a global load.
            const __nv_bfloat16* const pf = pfR + (t0 + t) * kPfPitch;
            ac = 1.0f;
#pragma unroll
            for (int l = 0; l < D; ++l) {
              if (l < j) continue;  // fixed levels hoist into the row scale (§3 step 3)
              ac *= __bfloat162float(pf[pf_off[l] + dig[l * BC + i]]);
            }
            //: THE WINDOW'S WRITE SIDE, NOT THE GROUP'S: a read-live group inside a
            //: write-live window sees state that the window's OTHER groups have
            //: already ticked, and `clk` is what carries that. At `W == 1` the two
            //: conditions coincide.
            if (DECAY && w_write) {
              ac *= panel_read_factor(clk[(t0 + t) * CKP + i], p_ref[i], log_keep[i]);
            }
          }
          stage_put(Rpa[rp_index(i, t)], ac);
        }
      }
      // ---- THE WRITE PANEL, `Wp[token][leaf]` -- stored TOKEN-MAJOR because it is the
      // ---- one operand read BOTH ways: the Gram's A (`m = token, k = leaf`) and the
      // ---- fold's B (`k = token, n = leaf`). One buffer, two `ldmatrix` atoms.
      // ---- docs/internals/consumer_kernel.md#write-panel
      __nv_bfloat16* const Wpa = Wp + a * kWpSlotElems;
      if (is_write) {
        //: BRANCHLESS, as the read panel: every atom staged, zeros ride self-masking,
        //: masks stay compute-only.
        for (int idx = tid; idx < kGroupTokens * BC; idx += kConsumerThreads) {
          const int t = idx / BC;
          const int i = idx - t * BC;
          float ac = 0.0f;
          if (t < gN) {
            const __nv_bfloat16* const pf = pfW + (t0 + t) * kPfPitch;
            ac = 1.0f;
#pragma unroll
            for (int l = 0; l < D; ++l) {
              if (l < j) continue;
              ac *= __bfloat162float(pf[pf_off[l] + dig[l * BC + i]]);
            }
            if (DECAY) {
              ac *= panel_write_factor(clk[(t0 + t) * CKP + i], p_ref[i], log_keep[i]);
            }
          }
          //: §6.4 PASS 1's per-leaf inclusion, COEFFICIENT HALF: a leaf is included iff
          //: some token's staged coefficient is nonzero. The store is IDEMPOTENT -- every
          //: writer writes the same `1` -- so the token axis needs no reduction.
          //: HAZARD summary-prequant -- docs/internals/consumer_kernel.md#summary-prequant
          if (summary && ac != 0.0f) sum_slot[i] = 1;
          stage_put(Wpa[wp_index(BC, t, i)], ac);
        }
      } else if (w_write) {
        //: **A SIDE-DEAD SLOT OF A WRITE-LIVE WINDOW STAGES EXACT ZEROS.** It is
        //: the same rule a ragged group's padded token slots already run on,
        //: extended from a slot's TAIL to a whole slot -- and it is what lets the
        //: fold consume all `W` slots with NO PER-SLOT PREDICATE: an unstaged slot
        //: would be a previous window's panel, a zero-staged one contributes an
        //: exactly-zero deposit. The branch is at the SLOT, never at the element,
        //: so the group quantum -- where a write-live window has no dead slot to
        //: reach this at all -- pays nothing for it.
        for (int e = tid; e < kWpSlotElems; e += kConsumerThreads) {
          Wpa[e] = __float2bfloat16(0.0f);
        }
      }
    }
    //: HAZARD clk-dead -- docs/internals/consumer_kernel.md#clk-dead
    if (DECAY && w_write) __syncthreads();  // <- clk is now DEAD; Vp may overwrite
    // ---- `Vp[ch, t] = scale_write[t] * v[token, ch]`, one slot per write-live group.
    // ---- CHANNEL-MAJOR, because the channel is the `m` axis of both GEMMs that read
    // ---- it. The write side folds the row scale into the VALUE tile because there `t`
    // ---- is the CONTRACTION index and the scale cannot hoist out of the sum; one
    // ---- staged object, two GEMMs. docs/internals/consumer_kernel.md#vp-store
    for (int a = 0; a < W; ++a) {
      if (!w_write) break;
      __nv_bfloat16* const Vpa = Vp + a * kVpSlotElems;
      if (wmr[a] == 0u) {
        //: The value tile of a side-dead slot, staged as the exact zeros that make
        //: the fold's slot loop predicate-free. The fill covers the row pad too,
        //: which no fetch reads.
        for (int e = tid; e < kVpSlotElems; e += kConsumerThreads) {
          Vpa[e] = __float2bfloat16(0.0f);
        }
        continue;
      }
      const int gN = gN_of(a);
      const int* const T = tk_of(a);
      const float* const swa = sw_of(a);
      for (int idx = tid; idx < kGroupTokens * DV; idx += kConsumerThreads) {
        const int t = idx / DV;
        const int ch = idx - t * DV;
        float vv = 0.0f;
        if (t < gN) {
          vv = __bfloat162float(static_cast<const __nv_bfloat16*>(
              p.v)[(static_cast<int64_t>(bh) * p.T + T[t]) * DV + ch]);
        }
        stage_put(Vpa[vp_index(ch, t)], swa[t] * vv);
      }
    }
    __syncthreads();

    //: **THE PREFETCH'S ISSUE POINT, AND IT COSTS NO BARRIER.** The barrier above is
    //: already there to publish the staged operands; it equally guarantees that every
    //: warp has finished READING the slab, which is the condition for refilling it.
    //: From here the copies for the next window run concurrently with this window's
    //: entire compute -- both Gram phases, the emission and the fold -- and are
    //: drained at the end of the visit body.
    pf_issue(widx + 1);

    // ---- PHASE A -- the two GEMMs that read only the ENTRY state and the staged
    // ---- panels: the readouts and the window Gram. They share a phase because
    // ---- neither writes what the other reads.

    // ---- GEMM 1: THE READOUTS, `y^T_a = S^T . Rp_a`, LIVE READ ATOMS ONLY. **THE A
    // ---- OPERAND IS THE WARP'S OWN STATE REGISTERS**, narrowed to bf16 words at the
    // ---- fragment and never loaded: the k-loop's only memory traffic is the B
    // ---- fragment. A warp holds one CHANNEL TILE and a LEAF RANGE, so the `kReps`
    // ---- warps of a channel tile each carry a PARTIAL over their own leaves -- and
    // ---- they need no reduction, because the emission below is already an
    // ---- `atomicAdd` split-K. docs/internals/consumer_kernel.md#gemm-readout
    float y[W][2][4];
    //: THE MASS COLUMN'S D FRAGMENTS. `dm` is to the denominator exactly what `y` is
    //: to the numerator: one accumulator per (GROUP, token tile) that both of the
    //: denominator's read-side terms land in. Its items are spread over ALL EIGHT
    //: warps -- the mass column is shared, so its terms carry no residency
    //: constraint. docs/internals/consumer_kernel.md#mass-column
    constexpr int kDmItems = (2 * W + kWarps - 1) / kWarps;
    float dm[kDmItems][4];
#pragma unroll
    for (int a = 0; a < W; ++a) {
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        y[a][nt][0] = 0.0f;
        y[a][nt][1] = 0.0f;
        y[a][nt][2] = 0.0f;
        y[a][nt][3] = 0.0f;
      }
      if (!owns || rmr[a] == 0u) continue;
      const __nv_bfloat16* const Rpa = Rp + a * kRpSlotElems;
      const unsigned rmask = rmr[a] & own_mask;
      if (rmask == own_mask) {
        mma_readout_tiles<kAPW>(St, Rpa, atom0 * kMmaKQuantum, lane, y[a]);
      } else {
        //: **THE SPARSE ARM WALKS THE WARP'S OWN ATOM SLOTS AT COMPILE TIME**, testing
        //: the mask bit, rather than walking the set bits: a runtime index into `St`
        //: would put the state in LOCAL MEMORY and undo the whole batch.
#pragma unroll
        for (int ai = 0; ai < kAPW; ++ai) {
          if (((rmask >> (atom0 + ai)) & 1u) == 0u) continue;
          mma_readout_tiles<1>(St + ai * 8, Rpa, (atom0 + ai) * kMmaKQuantum, lane, y[a]);
        }
      }
    }

    // ---- THE MASS COLUMN, TERM 1 -- `dm_a = broadcast(Smass) . Rp_a`, the SAME B
    // ---- OPERAND the readout uses with the mass column riding the A operand's `k`
    // ---- axis. Placement FORCED by phase B: the fold WRITES `Smass`, so term 1,
    // ---- which READS it, must already have run. Live read atoms only.
    // ---- docs/internals/consumer_kernel.md#mass-term1
    //: HAZARD mass-split -- docs/internals/consumer_kernel.md#mass-split
#pragma unroll
    for (int s = 0; s < kDmItems; ++s) {
      dm[s][0] = 0.0f;
      dm[s][1] = 0.0f;
      dm[s][2] = 0.0f;
      dm[s][3] = 0.0f;
      const int it = warp + s * kWarps;
      if (it >= 2 * W) continue;
      const int a = it >> 1, nt = it & 1;
      if (rmr[a] == 0u) continue;
      const unsigned rmask = rmr[a];
      const __nv_bfloat16* const Rpa = Rp + a * kRpSlotElems;
      if (rmask == kFullMask) {
        mma_mass_readout<kQPO>(Smass, Rpa, 0, nt * 8, lane, dm[s][0], dm[s][1], dm[s][2], dm[s][3]);
      } else {
        for (unsigned m = rmask; m; m &= m - 1) {
          const int q = __ffs(m) - 1;
          mma_mass_readout<1>(Smass, Rpa, q * kMmaKQuantum, nt * 8, lane, dm[s][0], dm[s][1],
                              dm[s][2], dm[s][3]);
        }
      }
    }

    // ---- GEMM 2: THE WINDOW GRAM, HELD TRANSPOSED -- `G^T[b][a] += Wp_b . Rp_a^T`
    // ---- over atoms live on BOTH sides, for every sub-block of the BLOCK LOWER
    // ---- TRIANGLE `b <= a`. THE SUB-CHUNKED DIAGONAL (FLA's `_intra_sub_inter` /
    // ---- `_intra_sub_intra` split, `fla/ops/gla/chunk.py:31-89,104-165`) is exactly
    // ---- this split: `b > a` is identically zero and is NEVER FORMED; `b < a` is a
    // ---- fully DENSE sub-block -- every token of `b` strictly precedes every token of
    // ---- `a` -- and stores UNMASKED; only the `W` DIAGONAL sub-blocks carry §5's
    // ---- INCLUSIVE causal mask. So the Gram is `O(W^2)` and the MASK is `O(W)`.
    // ---- We take their split and decline their scalar diagonal kernel: their gate
    // ---- varies along the contraction axis and cannot fold into one dot, ours is
    // ---- already folded into the panels at stage time, so our diagonal stays a
    // ---- masked GEMM. THE TRANSPOSE IS FREE -- the D fragment's rows become `token_b`
    // ---- and its columns `token_a`, which is the `k`-major form the intra term's B
    // ---- operand wants. docs/internals/consumer_kernel.md#gemm-gram
    if (mixed) {
      //: THE TRIANGLE IS ENUMERATED AT COMPILE TIME. `W` and `NT_GM` are template
      //: constants, so unrolling makes the item index -- and with it the
      //: item-to-warp assignment and the `a == b` diagonal test -- a CONSTANT per
      //: emitted block: no counter, no modulo, no branch on the mask's shape.
      int it = 0;
#pragma unroll
      for (int a = 0; a < W; ++a) {  // NOLINT -- the triangle, enumerated
#pragma unroll
        for (int b = 0; b <= a; ++b) {
#pragma unroll
          for (int nt = 0; nt < NT_GM; ++nt, ++it) {
            if (it % kWarps != warp) continue;
            //: **THE BLOCK IS SKIPPED ON THE SIDES, AND THE MMA ON THE ATOMS.**
            //: These are different conditions and the difference is load-bearing:
            //: GEMM 3 reads `G^T[b][a]` whenever both sides are live, so a block
            //: whose sides are live but whose atom masks are DISJOINT must still be
            //: STORED -- as the exact zeros it is -- or the intra term reads a
            //: previous window's Gram. (`BC == kMmaKQuantum` cannot express it: one
            //: atom per side makes "both live" and "masks intersect" the same
            //: statement, which is why only the `BC = 64` sparse cells saw it.)
            if (rmr[a] == 0u) continue;
            const unsigned gmask = rmr[a] & wmr[b];
            const __nv_bfloat16* const Rpa = Rp + a * kRpSlotElems;
            const __nv_bfloat16* const Wpb = Wp + b * kWpSlotElems;
            const int n0 = nt * 8;
            //: An empty `gmask` issues no `mma.sync` and the accumulators stay the
            //: exact zeros the block is; the stores below still run.
            float gd0 = 0.0f, gd1 = 0.0f, gd2 = 0.0f, gd3 = 0.0f;
            if (gmask == kFullMask) {
              mma_gram_atoms<kQPO>(Wpb, kWpPitch, Rpa, n0, lane, gd0, gd1, gd2, gd3);
            } else {
              for (unsigned m = gmask; m; m &= m - 1) {
                const int q = __ffs(m) - 1;
                mma_pitched_tile(Wpb, kWpPitch, 0, q * kMmaKQuantum, Rpa, kTokPitch,
                                 q * kMmaKQuantum, n0, lane, gd0, gd1, gd2, gd3);
              }
            }
            CONSUMER_D_ROWS(lane);
            const int c0 = n0 + 2 * _r, c1 = c0 + 1;
            __nv_bfloat16* const Gb = Gm + b * kGroupTokens * GMP + a * kGroupTokens;
            if (a == b) {
              //: The causal mask, in the transposed frame: the D fragment's ROW is
              //: `token_b` and its COLUMN is `token_a`, so §5's inclusive `u <= t`
              //: keeps the entries at or ABOVE the diagonal.
              stage_put(Gb[_g * GMP + c0], (c0 >= _g) ? gd0 : 0.0f);
              stage_put(Gb[_g * GMP + c1], (c1 >= _g) ? gd1 : 0.0f);
              stage_put(Gb[(_g + 8) * GMP + c0], (c0 >= _g + 8) ? gd2 : 0.0f);
              stage_put(Gb[(_g + 8) * GMP + c1], (c1 >= _g + 8) ? gd3 : 0.0f);
            } else {
              stage_put(Gb[_g * GMP + c0], gd0);
              stage_put(Gb[_g * GMP + c1], gd1);
              stage_put(Gb[(_g + 8) * GMP + c0], gd2);
              stage_put(Gb[(_g + 8) * GMP + c1], gd3);
            }
          }
        }
      }
    }

    // ---- THE VISIT'S ONE INTERNAL BARRIER, and it carries BOTH of the window's
    // ---- intra-CTA orderings at once: every readout's reads of `Smass` complete
    // ---- before the fold's writes of it, and the Gram's stores to `Gm` are published
    // ---- to phase B. **THE STATE'S NUMERATOR IS NO LONGER AMONG ITS EDGES** -- it is
    // ---- register-private, so the readout and the fold cannot race over it at all.
    // ---- It is a HARDWARE rendezvous because both edges are CTA-WIDE and
    // ---- CTA-UNIFORM. BOTH EDGES REQUIRE `mixed`, SO THE BARRIER DOES: a read-only
    // ---- window runs no fold and a write-only window reads no state, and neither
    // ---- has a Gram. A SINGLE-SIDED WINDOW PAYS NOTHING.
    // ---- **THE BARRIER COUNT IS PER WINDOW, NOT PER GROUP** -- three for `W`
    // ---- groups where the group-quantum visit paid three each.
    // ---- docs/internals/consumer_kernel.md#visit-barrier
    if (mixed) __syncthreads();

    // ---- PHASE B -- GEMM 3, the intra term `y^T_a += sum_{b <= a} Vp_b . G^T[b][a]`,
    // ---- accumulating into the SAME fragments the readout filled. The triangle is
    // ---- the Gram's; there is NO MASK anywhere here, because the zeros are already
    // ---- IN `Gm`. Token-axis contraction: no atom mask applies. An item belongs to
    // ---- ONE of the `kReps` warps holding that channel tile's `y` -- the others
    // ---- carry only their readout partial, and the emission sums the pieces.
    // ---- docs/internals/consumer_kernel.md#phase-c
    if (mixed && owns) {
#pragma unroll
      for (int a = 0; a < W; ++a) {
        if (rmr[a] == 0u) continue;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          if ((a * 2 + nt) % kReps != hrep) continue;
#pragma unroll
          for (int b = 0; b < W; ++b) {
            if (b > a) continue;
            mma_pitched_tile(Vp + b * kVpSlotElems, kTokPitch, m0, 0, Gm, GMP, b * kGroupTokens,
                             a * kGroupTokens + nt * 8, lane, y[a][nt][0], y[a][nt][1], y[a][nt][2],
                             y[a][nt][3]);
          }
        }
      }
    }
    if (mixed) {
      //: THE MASS COLUMN, TERM 2 -- the intra term's denominator,
      //: `dm_a += sum_{b <= a} broadcast(sw_b) . G^T[b][a]`. The causal mask is
      //: already IN `Gm` (the strictly-future entries of the diagonal block are
      //: stored zero and the strictly-future BLOCKS are never read), so the whole
      //: window contraction is the `u <= t` sum.
      //: HAZARD mass-split -- docs/internals/consumer_kernel.md#mass-split
#pragma unroll
      for (int s = 0; s < kDmItems; ++s) {
        const int it = warp + s * kWarps;
        if (it >= 2 * W) continue;
        const int a = it >> 1, nt = it & 1;
        if (rmr[a] == 0u) continue;
#pragma unroll
        for (int b = 0; b < W; ++b) {
          if (b > a) continue;
          mma_bcastA_tile(sw_of(b), Gm, GMP, b * kGroupTokens, a * kGroupTokens + nt * 8, lane,
                          dm[s][0], dm[s][1], dm[s][2], dm[s][3]);
        }
      }
    }

    // ---- SPLIT-K EMISSION of this window's partials, straight out of the D
    // ---- fragments, scattered through each group's token list (the P14 microbench
    // ---- priced exactly this pattern: banded, DRAM-byte ratio 1.0009x at the
    // ---- measured gap). The `out == 0` early-out runs FIRST so a ragged group's
    // ---- padded slots (`sr == 0`, coefficient rows exactly zero) never read the
    // ---- token list at all; members are `< T` by construction. The `!= 0` guard is
    // ---- a PERFORMANCE SKIP ON A VALUE, never a membership decision. IT RUNS BEFORE
    // ---- THE FOLD: the two are independent, and emitting first RETIRES every output
    // ---- fragment before the fold goes live, which is where the window's register
    // ---- high-water mark sits.
    // ---- **THE FRAGMENT IS TRANSPOSED AND THAT IMPROVES THE SCATTER**: the eight
    // ---- lanes sharing a `token` now hold eight CONSECUTIVE channels, so one
    // ---- `atomicAdd` instruction covers four 32-byte segments where the untransposed
    // ---- fragment covered eight. The `kReps` partials per output are what the split-K
    // ---- was already built for. docs/internals/consumer_kernel.md#split-k-emission
    if (owns) {
#pragma unroll
      for (int a = 0; a < W; ++a) {
        if (rmr[a] == 0u) continue;
        const int* const T = tk_of(a);
        const float* const sra = sr_of(a);
        CONSUMER_D_ROWS(lane);
        const int ch0 = m0 + _g, ch1 = ch0 + 8;
        auto emit = [&](int t, int ch, float val) {
          const float out = sra[t] * val;
          if (out == 0.0f) return;
          atomicAdd(&p.num[(static_cast<int64_t>(bh) * p.T + T[t]) * DV + ch], out);
        };
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          const int t0e = nt * 8 + 2 * _r, t1e = t0e + 1;
          emit(t0e, ch0, y[a][nt][0]);
          emit(t1e, ch0, y[a][nt][1]);
          emit(t0e, ch1, y[a][nt][2]);
          emit(t1e, ch1, y[a][nt][3]);
        }
      }
    }
    //: THE DENOMINATOR, OUT OF THE MASS FRAGMENTS. A broadcast A makes every output
    //: ROW of `dm` the same value, so row 0 -- the lanes with `g == 0` -- carries the
    //: tile's eight tokens. The `!= 0` guard retires a ragged group's padded slots on
    //: the same rule the numerator runs on: their `sr` is an exact zero, so the token
    //: list is never read for them.
#pragma unroll
    for (int s = 0; s < kDmItems; ++s) {
      const int it = warp + s * kWarps;
      if (it >= 2 * W) continue;
      const int a = it >> 1, nt = it & 1;
      if (rmr[a] == 0u) continue;
      const int* const T = tk_of(a);
      const float* const sra = sr_of(a);
      CONSUMER_D_ROWS(lane);
      if (_g == 0) {
        const int t0e = nt * 8 + 2 * _r, t1e = t0e + 1;
        const float da = sra[t0e] * dm[s][0];
        if (da != 0.0f) atomicAdd(&p.den[static_cast<int64_t>(bh) * p.T + T[t0e]], da);
        const float db = sra[t1e] * dm[s][1];
        if (db != 0.0f) atomicAdd(&p.den[static_cast<int64_t>(bh) * p.T + T[t1e]], db);
      }
    }

    // ---- GEMM 4, THE BOUNDARY FOLD `S^T += sum_b Vp_b . Wp_b`, AND THIS IS WHERE
    // ---- THE REGISTER-RESIDENT STATE PAYS: **the `mma.sync` ACCUMULATES DIRECTLY INTO
    // ---- THE STATE**, `D == C == St`. There is no read-modify-write, no format
    // ---- boundary and no shared-memory traffic of any kind on the state's numerator;
    // ---- the whole window's deposits land in the fragments that already hold it.
    // ---- The partition is the OWNERSHIP RULE itself -- one channel tile and one atom
    // ---- range per warp -- so it is static, uniform-cost and costs no atomic.
    // ---- docs/internals/consumer_kernel.md#gemm-fold
    if (w_write && owns) {
      //: **ONE A OPERAND FEEDS EVERY TILE THE WARP OWNS.** The value tile's fragment at
      //: the warp's channel rows is the same for all `2 * kAPW` state tiles, so it is
      //: fetched ONCE per staged group and the multiplies that follow are independent
      //: back-to-back issue into independent accumulators -- the deep unrolled form the
      //: register-resident state exists to make possible.
#pragma unroll
      for (int b = 0; b < W; ++b) {
        unsigned va0, va1, va2, va3;
        pitched_frag_a(Vp + b * kVpSlotElems, kTokPitch, m0, 0, lane, va0, va1, va2, va3);
        //: **THE WINDOW'S DEPOSITS ACCUMULATE WITH NO PREDICATE AT ALL** -- in ascending
        //: group order, over every staged slot. Every slot of a write-live window holds
        //: real coefficients or exact zeros (the staging above guarantees it) and a dead
        //: ATOM's rows are exact zeros by self-masking, so an unconditional `mma.sync` is
        //: EXACT and the hot loop carries no test.
#pragma unroll
        for (int ai = 0; ai < kAPW; ++ai) {
          //: A dead atom's rows are exact zeros and its `s0` is exactly 1, so skipping
          //: it is the identity rather than an approximation.
          if (((w_wmask >> (atom0 + ai)) & 1u) == 0u) continue;
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            float* const st = St + ai * 8 + half * 4;
            mma_fragA_pitchedB(va0, va1, va2, va3, Wp + b * kWpSlotElems, kWpPitch, 0,
                               (atom0 + ai) * kMmaKQuantum + 8 * half, lane, st[0], st[1], st[2],
                               st[3]);
          }
        }
      }
      // ---- DECAY's EXIT SCALE RIDES THE ACCUMULATOR, ONCE PER WINDOW: one multiply on
      // ---- a register value. It is once per window rather than once per group because
      // ---- every deposit is already in the window's ONE reference frame
      // ---- (`keep^{P_ref - P[u]}`), so the deposits simply add before the frame is
      // ---- left. docs/internals/consumer_kernel.md#fold-epilogue
      if (DECAY) {
#pragma unroll
        for (int ai = 0; ai < kAPW; ++ai) {
          if (((w_wmask >> (atom0 + ai)) & 1u) == 0u) continue;
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            const int i0 = (atom0 + ai) * kMmaKQuantum + 8 * half + 2 * _sr;
            const float sa = s0[i0], sb = s0[i0 + 1];
            float* const st = St + ai * 8 + half * 4;
            st[0] *= sa;
            st[1] *= sb;
            st[2] *= sa;
            st[3] *= sb;
          }
        }
      }
      // ---- THE MASS COLUMN, TERM 3 -- the fold's own term,
      // ---- `Smass += sum_b broadcast(sw_b) . Wp_b`, ON THE SAME TENSOR CORE AS THE
      // ---- FOLD ITSELF: one `mma.sync` per 8-leaf tile per staged group against a
      // ---- broadcast `sw`. It stays in SHARED MEMORY because the mass column is
      // ---- shared: `kMT` warps read it and one writes each tile.
      // ---- A dead atom's rows: acc == 0 and s0 == 1 -- the skip is the identity.
    }
    if (w_write) {
      for (int lt = warp; lt < BC / 8; lt += kWarps) {
        CONSUMER_D_ROWS(lane);
        if (((w_wmask >> (lt / 2)) & 1u) == 0u) continue;
        const int n0 = lt * 8;
        float m0v = 0.0f, m1v = 0.0f, m2v = 0.0f, m3v = 0.0f;
#pragma unroll
        for (int b = 0; b < W; ++b) {
          mma_bcastA_tile(sw_of(b), Wp + b * kWpSlotElems, kWpPitch, 0, n0, lane, m0v, m1v, m2v,
                          m3v);
        }
        (void)m2v;
        (void)m3v;
        //: Output ROW 0 of a broadcast-A tile carries the result, so only the four
        //: lanes with `g == 0` touch `Smass` -- a read the others would discard is
        //: still a shared-memory access to an address their own warp is about to write.
        if (_g == 0) {
          const int i0 = n0 + 2 * _r, i1 = i0 + 1;
          Smass[i0] = (DECAY ? s0[i0] : 1.0f) * (Smass[i0] + m0v);
          Smass[i1] = (DECAY ? s0[i1] : 1.0f) * (Smass[i1] + m1v);
        }
      }
    }

    // ---- §6.4 PASS 1's PER-LEAF EPILOGUE, once per WINDOW. The COEFFICIENT half of
    // ---- the inclusion test lives in the panel build; what is left here has no
    // ---- pre-quantization site: docs/internals/consumer_kernel.md#pass1-epilogue
    if (w_write && summary && DECAY) {
      for (int i = tid; i < BC; i += kConsumerThreads) {
        if (s0[i] != 1.0f) sum_slot[i] = 1;
        a_state[i] *= s0[i] * s0[i];
      }
    }
    //: THE RUN-AHEAD (contract items 2 and 6). The producer cooks the NEXT window's
    //: `W` groups into the slots the window `kSchedDepth` back vacated -- with
    //: `kSlots == W * (kSchedDepth + 1)` those are never the slots being read, which
    //: is the whole reason the ring is at least two windows deep. Production runs
    //: CONCURRENTLY with this window's compute and is published by the end-of-window
    //: barrier below, so NOTHING WAITS ON IT.
    if (warp == 0) {
#pragma unroll 1
      for (int a = 0; a < W; ++a) produce_group(((widx + kSchedDepth) * W + a) % kSlots);
    }
    //: THE DRAIN, IMMEDIATELY BEFORE A BARRIER THAT ALREADY EXISTS. `cp.async`
    //: completion is visible to the ISSUING THREAD only, so the cross-warp
    //: publication is the end-of-window rendezvous below -- which is why the wait
    //: goes here and not at the top of the next visit, where it would need a barrier
    //: of its own.
    cp_async_wait_all();
    __syncthreads();
  }

  if (summary) {
    // ---- §6.4 PASS 1's PUBLICATION: the SPARSE written-leaf summary. Slot
    // ---- `owner_slot0` is the scan's entry stash, so this lands one past it and
    // ---- pass 3's segment `q` reads `owner_slot0 + q` UNIFORMLY:
    // ---- docs/internals/consumer_kernel.md#pass1-publication
    const int64_t slot = static_cast<int64_t>(owner_slot0 + seg + 1);
    if (warp == 0) {
      int k = 0;
      for (int b0 = 0; b0 < BC; b0 += 32) {
        const int i = b0 + lane;
        const bool in_range = i < BC;
        const bool take = in_range && (sum_slot[i] != 0);
        const unsigned below = (1u << lane) - 1u;
        const unsigned mask = __ballot_sync(0xffffffffu, take);
        const int q = k + __popc(mask & below);
        if (in_range) sum_slot[i] = take ? q : -1;
        if (take) p.ws_leaf[slot * BC + q] = i;
        k += __popc(mask);
      }
      if (lane == 0) sum_count[0] = k;
    }
    __syncthreads();
    if (tid == 0) p.ws_count[slot] = sum_count[0];
    for (int i = tid; i < BC; i += kConsumerThreads) {
      const int q = sum_slot[i];
      if (q >= 0) p.ws_scale[slot * BC + q] = a_state[i];
    }
    //: **THE STATE LEAVES ITS REGISTERS AT THE PUBLICATION**, on the ownership map it
    //: entered on -- the compaction index `sum_slot[leaf]` is the only thing the write
    //: adds over the entry read's inverse.
    if (owns) {
#pragma unroll
      for (int ai = 0; ai < kAPW; ++ai) {
        const int q = atom0 + ai;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int i0 = q * kMmaKQuantum + 8 * half + 2 * _sr;
          const float* const st = St + ai * 8 + half * 4;
          const int q0 = sum_slot[i0], q1 = sum_slot[i0 + 1];
          if (q0 >= 0) {
            p.ws_state[(slot * BC + q0) * cols + m0 + _sg] = st[0];
            p.ws_state[(slot * BC + q0) * cols + m0 + 8 + _sg] = st[2];
          }
          if (q1 >= 0) {
            p.ws_state[(slot * BC + q1) * cols + m0 + _sg] = st[1];
            p.ws_state[(slot * BC + q1) * cols + m0 + 8 + _sg] = st[3];
          }
        }
      }
    }
    for (int i = tid; i < BC; i += kConsumerThreads) {
      const int q = sum_slot[i];
      if (q >= 0) p.ws_state[(slot * BC + q) * cols + DV] = Smass[i];
    }
    return;
  }
  // ---- THE OWNER'S STATE BLOCK, WRITTEN TO HBM ONCE. An unpaged atom is never
  // ---- stored back, PER ATOM. `last_seg` is §6.4's publication rule and is a
  // ---- DELIBERATE NUMERIC CHOICE, not the scan's job:
  // ---- docs/internals/consumer_kernel.md#state-store
  if (!last_seg) return;
  if (owns) {
#pragma unroll
    for (int ai = 0; ai < kAPW; ++ai) {
      const int q = atom0 + ai;
      const int64_t row0 = atom_row_base<BC>(p, bh, owner, q);
      if (row0 < 0) continue;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int64_t r0 = row0 + 8 * half + 2 * _sr;
        const float* const st = St + ai * 8 + half * 4;
        p.state[r0 * cols + m0 + _sg] = st[0];
        p.state[(r0 + 1) * cols + m0 + _sg] = st[1];
        p.state[r0 * cols + m0 + 8 + _sg] = st[2];
        p.state[(r0 + 1) * cols + m0 + 8 + _sg] = st[3];
      }
    }
  }
#pragma unroll
  for (int q = 0; q < kQPO; ++q) {
    const int64_t row0 = atom_row_base<BC>(p, bh, owner, q);
    if (row0 < 0) continue;
    for (int r = tid; r < kMmaKQuantum; r += kConsumerThreads) {
      p.state[(row0 + r) * cols + DV] = Smass[q * kMmaKQuantum + r];
    }
  }
}

#undef CONSUMER_D_ROWS

}  // namespace consumer
}  // namespace rola
