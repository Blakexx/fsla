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
#define ROLA_CONSUMER_CTAS_EXPR                                   \
  ctas_for_residency(consumer_smem_bytes<BC, DV, D, DECAY>(),     \
                     consumer_reg_budget(BC, DECAY, DV, D))
#endif
template <int BC, int DV, int D, bool DECAY>
__global__ __launch_bounds__(kConsumerThreads, ROLA_CONSUMER_CTAS_EXPR) void
consumer_forward_kernel(ConsumerParams p) {
  //: **THE VRAM -> SMEM PIPELINE'S DEPTH**, in staged visits, derived from this arm's
  //: own carve: docs/internals/smem_layout.md#pf-stages
  constexpr int S = pf_stages<BC, DV, D, DECAY>();
  constexpr int kSlots = sched_slots(S);
  constexpr int kSchedDepth = sched_depth(S);
  constexpr int NT = kGroupTokens / 8;  // n8 tiles of one Gram block
  extern __shared__ char smem_raw[];

  // ---- THE SMEM CARVE. Every region, its pitch and its measured residency cost:
  // ---- docs/internals/smem_layout.md#smem-ledger
  constexpr int CKP = BC + 4;  // clk's row pitch, token-major over the group
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
  constexpr int kWpPitch = leaf_pitch(BC);
  constexpr int kWpElems = kGroupTokens * kWpPitch;
  constexpr int kVpPitch = val_pitch(DV);
  //: THE STAGED REGIONS' row pitches and the amplitude slab's 16-byte chunk count,
  //: compile-time worst cases over the owner rectangle's factorisations:
  //: docs/internals/smem_layout.md#staged-regions
  constexpr int kPfPitch = amp_slab_pitch(BC, D);
  constexpr int kPfChunks = amp_slab_chunks(BC, D);
  constexpr int kScPitch = scale_slab_pitch(D);
  //: **THE PER-THREAD WALK EXTENTS, AS COMPILE-TIME COUNTS.** The two panel walks and
  //: the stage issue are flat `idx = tid; idx += kConsumerThreads` loops over
  //: compile-time extents, so how many elements a thread owns is a CONSTANT and the
  //: element's identity -- its leaf, its token, its chunk -- is FIXED FOR THE WHOLE
  //: OWNER. That is what licenses the per-owner register hoists below: a walk's
  //: per-element shared-memory constants are read once per owner into registers
  //: instead of once per visit out of `dig` / `log_keep` / the chunk descriptors.
  //: docs/internals/consumer_kernel.md#walk-hoists
  constexpr int kPanelIters = BC * kGroupTokens / kConsumerThreads;
  static_assert(kPanelIters * kConsumerThreads == BC * kGroupTokens,
                "a panel's elements must divide evenly among the CTA's threads");
  constexpr int kIssueIters = (kGroupTokens * kPfChunks + kConsumerThreads - 1) / kConsumerThreads;

  //: **THE STATE IS NOT IN THIS CARVE.** Its numerator lives in `St` below -- the
  //: fold's own D fragments, held in registers for the whole kernel. What remains here
  //: is the MASS COLUMN: fp32, `[BC]`, the denominator of a ratio whose consumers are
  //: broadcast matvecs, read by the readout's term 1 and written by the fold's term 3
  //: from different warps, so it is the one part of the state that is genuinely shared.
  //: **AND IT IS HELD TWICE.** Term 1 reads the visit's ENTRY column and the fold's
  //: term 3 writes its EXIT column; two buffers make those disjoint, which is what
  //: removes the read-before-write edge from the visit's barrier. The pair alternates
  //: per visit, so a visit's exit column is the next visit's entry column with no
  //: copy. docs/internals/consumer_kernel.md#mass-pingpong
  float* const Smass0 = reinterpret_cast<float*>(smem_raw);
  float* const Smass1 = Smass0 + BC;
  constexpr size_t kMassBytes = 2 * static_cast<size_t>(BC) * sizeof(float);
  char* carve = smem_raw + kMassBytes;
  //: `ldmatrix` TAKES A 16-BYTE-ALIGNED ROW ADDRESS, so every staged base and every
  //: staged ROW must be 16-byte aligned -- a requirement no pitch predicate implies.
  //: Stated, not trusted: docs/internals/consumer_kernel.md#ldsm-alignment
  __nv_bfloat16* const Rp = reinterpret_cast<__nv_bfloat16*>(carve);
  constexpr size_t kRpBytes = static_cast<size_t>(BC) * kTokPitch * sizeof(__nv_bfloat16);
  carve += kRpBytes;
  __nv_bfloat16* const Wp = reinterpret_cast<__nv_bfloat16*>(carve);
  constexpr size_t kWpBytes = static_cast<size_t>(kWpElems) * sizeof(__nv_bfloat16);
  carve += kWpBytes;
  //: THE CLOCK PREFIX, `[kGroupTokens][CKP]` fp32, with a home of its own: the value
  //: tile it used to alias is now filled `S` visits ahead, so the two are
  //: simultaneously live. docs/internals/consumer_kernel.md#decay-clock
  constexpr size_t kClkBytes =
      DECAY ? align16(static_cast<size_t>(kGroupTokens) * CKP * sizeof(float))
            : static_cast<size_t>(0);
  float* const clk = reinterpret_cast<float*>(carve);
  carve += kClkBytes;
  int* const dig = reinterpret_cast<int*>(carve);  // [D][BC]
  carve += static_cast<size_t>(D) * BC * sizeof(int);
  //: `Gm` IS AN `ldmatrix` SOURCE (the intra term's B and the intra mass term's), so
  //: its base carries the same 16-byte requirement the panels do, and for the same
  //: reason. It holds `G^T` at `kTokPitch`, and `pitched_ldsm_banks_ok<16, 16,
  //: kTokPitch>` below licenses that base at this pitch.
  __nv_bfloat16* const Gm = reinterpret_cast<__nv_bfloat16*>(carve);
  constexpr size_t kGmOffsetBytes =
      kMassBytes + kRpBytes + kWpBytes + kClkBytes + static_cast<size_t>(D) * BC * sizeof(int);
  static_assert(kMassBytes % 16 == 0 && (kMassBytes + kRpBytes) % 16 == 0
                    && (kMassBytes + kRpBytes + kWpBytes) % 16 == 0 && kGmOffsetBytes % 16 == 0,
                "Rp, Wp and Gm must be 16-byte aligned for ldmatrix");
  carve += static_cast<size_t>(kGroupTokens) * kTokPitch * sizeof(__nv_bfloat16);
  float* const log_keep = reinterpret_cast<float*>(carve);
  float* const p_ref = log_keep + (DECAY ? BC : 0);
  float* const s0 = p_ref + (DECAY ? BC : 0);
  //: THE COOKED SCHEDULE RING. `kSlots` group descriptors (`kSchedSlotInts` ints
  //: each; the field offsets are `kSchedGN/RMask/WMask/Flags`), written by the
  //: PRODUCER WARP `kSchedDepth` groups ahead and published by the end-of-visit
  //: `__syncthreads()` of the visit that produced them. Every consumer warp READS
  //: the descriptor rather than deriving it. docs/internals/consumer_kernel.md#sched-ring
  int* const g_ring = reinterpret_cast<int*>(s0 + (DECAY ? BC : 0));
  //: THE VISIT'S ROW SCALES: the read scale then the write scale for each of the
  //: group's member tokens, REDUCED FROM THE STAGED SCALARS at the top of the visit.
  //: They are a per-visit quantity rather than a ring field because the scalars they
  //: multiply ride the staged pipeline, not the cook.
  //: docs/internals/consumer_kernel.md#row-scales
  float* const sr = reinterpret_cast<float*>(g_ring + kSlots * kSchedSlotInts);
  float* const sw = sr + kGroupTokens;
  float* const a_state = reinterpret_cast<float*>(Rp);         // [BC], the survival product A
  int* const sum_slot = reinterpret_cast<int*>(Rp) + BC;       // [BC]
  int* const sum_count = reinterpret_cast<int*>(Rp) + 2 * BC;  // [1]
  static_assert(kRpBytes >= (2 * static_cast<size_t>(BC) + 1) * sizeof(int),
                "pass 1's summary state must fit inside the read panel");

  // ---- EACH OPERAND'S PITCH PROVES ITS OWN CONFLICT-FREEDOM.
  // ---- docs/internals/consumer_kernel.md#pitches
  static_assert(pitched_ldsm_banks_ok<BC, kGroupTokens, kTokPitch>(),
                "Rp: the ldmatrix.trans fetch conflicts or is misaligned");
  //: **THE VALUE TILE IS READ BY THE TRANSPOSING ATOM**, at its own token-major
  //: pitch: `k` is the TOKEN axis (the block's rows) and `n` the CHANNEL axis, and
  //: two such fetches at `n0` and `n0 + 8` reconstruct the fold's A operand.
  static_assert(pitched_ldsm_banks_ok<kGroupTokens, DV, kVpPitch>(),
                "Vp: the ldmatrix.trans fetch conflicts or is misaligned");
  static_assert(pitched_ldsm_banks_ok<kGroupTokens, BC, kWpPitch>(),
                "Wp: the ldmatrix fetches conflict or are misaligned");
  static_assert(pitched_ldsm_banks_ok<kGroupTokens, kGroupTokens, kTokPitch>(),
                "Gm: the intra term's ldmatrix.trans fetch conflicts or is misaligned");
  static_assert(pitched_store_banks_ok<BC, kGroupTokens, kTokPitch>(),
                "Rp: the staging store conflicts");
  static_assert(pitched_store_banks_ok<kGroupTokens, BC, kWpPitch>(),
                "Wp: the staging store conflicts");
  //: **THE VALUE TILE HAS NO STAGING STORE AT ALL.** It is a `cp.async` DESTINATION
  //: in the layout it is read in -- `DV` contiguous channels per token is exactly
  //: what HBM holds and exactly what an asynchronous copy can land -- so the
  //: column-strided, 4-way conflicted transpose the point-of-use gather used to pay
  //: has no site. The write gate's row scale is applied at the FRAGMENT instead, on
  //: the four registers the two `.trans` fetches deliver.
  //: docs/internals/consumer_kernel.md#value-stream

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
  //: The entry column lands in `Smass0`; the walk below alternates from there.
  if (summary || from_ws) {
    for (int i = tid; i < BC; i += kConsumerThreads) {
      Smass0[i] = summary ? 0.0f : ws_entry[i * cols + DV];
    }
  } else {
#pragma unroll
    for (int q = 0; q < kQPO; ++q) {
      const int64_t row0 = atom_row_base<BC>(p, bh, owner, q);
      for (int r = tid; r < kMmaKQuantum; r += kConsumerThreads) {
        Smass0[q * kMmaKQuantum + r] = (row0 >= 0) ? p.state[(row0 + r) * cols + DV] : 0.0f;
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
  int* const p_st = reinterpret_cast<int*>(sw + kGroupTokens);
  //: [0] p_cur, [1] p_chunk_base, [2] p_live, [3] p_word, [4] p_rem, [5] p_gfr,
  //: [6] p_gfw, then the PER-LANE chunk (fr, fw, mem) and ingredient (ra, wa)
  //: planes, one word per lane each.
  int* const p_chunk = p_st + 8;  // [5][32]: fr, fw, mem, ra, wa
  //: THE PREFETCH'S PER-OWNER CHUNK DESCRIPTORS. One entry per 16-byte chunk of a
  //: slab row: the SPANNED LEVEL it belongs to, its element offset inside that
  //: level's amplitude row (already aligned down to the 16-byte boundary), and the
  //: SET OF ATOMS whose digits it carries -- which is what makes the copy per-atom.
  //: Built once per owner because the owner's digit rectangle is fixed for the whole
  //: kernel; only the ATOM MASK the set is tested against varies per group.
  int* const ch_lvl = p_chunk + 5 * 32;
  int* const ch_src = ch_lvl + kPfChunks;
  int* const ch_atoms = ch_src + kPfChunks;
  //: THE GRAM'S PUBLICATION COUNTER. One int, monotone for the whole kernel: each
  //: producing warp adds one when its Gram sub-blocks are stored and fenced, and a
  //: consumer proceeds when the count reaches its visit's target. Monotone rather
  //: than reset-per-visit because a reset would itself need the rendezvous it
  //: replaces. docs/internals/consumer_kernel.md#gram-publication
  int* const gram_flag = ch_atoms + kPfChunks;
  //: **THE ISSUE'S PER-THREAD CHUNK, IN REGISTERS.** A thread's slot in the issue walk
  //: is `tid + k * kConsumerThreads`, so its (token, chunk) pair -- and therefore the
  //: chunk's LEVEL, its two source base pointers, their two row pitches, its atom set
  //: and its destination offset -- are FIXED FOR THE WHOLE OWNER. They are read out of
  //: the descriptors ONCE (below, at the entry epoch) and never again: the issue loop's
  //: only per-visit inputs are the group's masks and its token list.
  //: docs/internals/consumer_kernel.md#walk-hoists
  const __nv_bfloat16* ch_pr[kIssueIters];
  const __nv_bfloat16* ch_pw[kIssueIters];
  int ch_lpr[kIssueIters], ch_lpw[kIssueIters], ch_dst[kIssueIters], ch_tok[kIssueIters];
  unsigned ch_am[kIssueIters];
  //: A slot past the walk's extent or past this owner's live chunk count ISSUES
  //: NOTHING. It cannot ride the copy's own predicate: a false predicate ZERO-FILLS
  //: its destination, so a dead slot sharing a destination offset would erase a live
  //: chunk.
  bool ch_live[kIssueIters];
  //: **THE STAGED REGIONS, `S` STAGES DEEP, AND THEY ARE THE WHOLE VRAM -> SMEM
  //: LINK.** Four progressively-consumed streams land here and nowhere else: the two
  //: AMPLITUDE SLABS, the VALUE TILE (in the layout its `ldmatrix` reads, so the
  //: copy IS the staging) and the ROW-SCALE SCALARS (the write gate and the fixed
  //: levels' amplitudes, one row per token). They are the LAST regions of the carve
  //: so that `cp.async`'s 16-byte destination alignment is ONE align-up rather than
  //: a constraint every region before them would have to carry.
  //: docs/internals/consumer_kernel.md#staged-regions
  constexpr size_t kStBase = align16(consumer_smem_core_bytes<BC, DV, D, DECAY, S>());
  constexpr size_t kAmpBytes = static_cast<size_t>(kGroupTokens) * kPfPitch * sizeof(__nv_bfloat16);
  constexpr size_t kValBytes =
      align16(static_cast<size_t>(kGroupTokens) * kVpPitch * sizeof(__nv_bfloat16));
  constexpr size_t kScaleBytes =
      align16(static_cast<size_t>(kGroupTokens) * kScPitch * sizeof(float));
  constexpr size_t kStageBytes = 2 * kAmpBytes + kValBytes + kScaleBytes;
  static_assert(kStBase % 16 == 0 && kAmpBytes % 16 == 0,
                "every staged region base must be 16-byte aligned for cp.async and ldmatrix");
  static_assert(
      kStBase + static_cast<size_t>(S) * kStageBytes == consumer_smem_bytes<BC, DV, D, DECAY>(),
      "the kernel's carve and the launch's reservation must be one expression");
  char* const st_base = smem_raw + kStBase;
  //: One stage's four regions, by stage index. Addressed through lambdas because the
  //: issue and every consumer must not be able to disagree about the stride.
  //: ONE live pointer per visit, not four: the stage's base, with each region at a
  //: compile-time offset from it.
  auto stage_at = [&](int st) { return st_base + st * kStageBytes; };
  auto pfR_of = [](char* g) { return reinterpret_cast<__nv_bfloat16*>(g); };
  auto pfW_of = [](char* g) { return reinterpret_cast<__nv_bfloat16*>(g + kAmpBytes); };
  auto Vp_of = [](char* g) { return reinterpret_cast<__nv_bfloat16*>(g + 2 * kAmpBytes); };
  auto pfS_of = [](char* g) { return reinterpret_cast<float*>(g + 2 * kAmpBytes + kValBytes); };
  if (tid == 0) {
    p_st[0] = w_lo;
#pragma unroll
    for (int q = 1; q < 8; ++q) p_st[q] = 0;
    gram_flag[0] = 0;
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

  // ---- **THE STAGE ISSUE -- THE WHOLE VRAM -> SMEM LINK, IN ONE PLACE.** Fills stage
  // ---- `st` with everything visit `gnext` will progressively consume, out of the
  // ---- descriptor the producer cooked for it: both AMPLITUDE SLABS, the VALUE TILE
  // ---- and the ROW-SCALE SCALARS. Nothing here reads what it copies and nothing
  // ---- waits.
  // ----
  // ---- **THE DELEGATED CHOICE (P25 contract item 7), MADE HERE.** On sm_86 the
  // ---- `cp.async` commit/wait state is PER WARP, so the question was warp-
  // ---- partitioned streams against interleaved commit order. The answer is ONE
  // ---- COMMIT GROUP PER WARP PER VISIT: one stage is consumed at one point, so a
  // ---- partial-fill ordering has no consumer that could use it, and one group per
  // ---- stage is exactly what lets `cp.async.wait_group S - 1` pace the pipeline by
  // ---- STAGE. Each loop is issued on a FLAT CTA-WIDE INDEX so that a warp's 32 lanes
  // ---- take 32 CONSECUTIVE chunks -- each warp's share is a contiguous byte range,
  // ---- which is the only property the memory system rewards.
  // ----
  // ---- THE PREDICATE IS THE MECHANISM'S PER-ATOM HALF. A chunk is copied iff some
  // ---- atom LIVE ON THAT SIDE IN THAT GROUP holds a digit in it; otherwise it is
  // ---- issued with src-size zero and the hardware zero-fills it. That is EXACT: a
  // ---- leaf reading a zero-filled chunk belongs to an atom dead on that side, whose
  // ---- staged coefficient must be exactly zero, and zeroing one of its level
  // ---- factors delivers exactly that. The ragged tail rides the same predicate,
  // ---- with the token index CLAMPED so no out-of-range address is ever formed. The
  // ---- value tile and the scalars take the SAME rule: a dead or padded slot is
  // ---- zero-filled, which is the exact zero its consumers already require.
  // ---- docs/internals/consumer_kernel.md#stage-issue
  auto stage_issue = [&](int gnext, int st) {
    char* const dst_stage = stage_at(st);
    const int* const Ra = g_ring + (gnext % kSlots) * kSchedSlotInts;
    const int gN = Ra[kSchedGN];
    const int flags = Ra[kSchedFlags];
    //: The STAGED masks -- the side flag already folded in, exactly as the visit
    //: body's `rmr` / `wmr` fold it.
    const unsigned rmask =
        (!summary && (flags & 1) != 0) ? static_cast<unsigned>(Ra[kSchedRMask]) : 0u;
    const unsigned wmask = ((flags & 2) != 0) ? static_cast<unsigned>(Ra[kSchedWMask]) : 0u;
    __nv_bfloat16* const dR = pfR_of(dst_stage);
    __nv_bfloat16* const dW = pfW_of(dst_stage);
#pragma unroll
    for (int k = 0; k < kIssueIters; ++k) {
      if (!ch_live[k]) continue;
      const int t = ch_tok[k];
      const bool in_group = t < gN;
      const unsigned rm = in_group ? (ch_am[k] & rmask) : 0u;
      const unsigned wm = in_group ? (ch_am[k] & wmask) : 0u;
      const int64_t tr = tok_base + (in_group ? Ra[t] : 0);
      cp_async_16(dR + ch_dst[k], ch_pr[k] + tr * ch_lpr[k], rm != 0u);
      cp_async_16(dW + ch_dst[k], ch_pw[k] + tr * ch_lpw[k], wm != 0u);
    }
    //: **THE VALUE TILE, COPIED STRAIGHT INTO THE LAYOUT ITS `ldmatrix` READS.** `DV`
    //: contiguous channels per token is what HBM holds and what the token-major tile
    //: wants, so the asynchronous copy IS the staging -- there is no gather at the
    //: point of use and no transposing store. The write gate's row scale is applied
    //: later, at the fragment. docs/internals/consumer_kernel.md#value-stream
    constexpr int kValChunks = DV / 8;
    static_assert(DV % 8 == 0, "the value row must be a whole number of 16-byte chunks");
    __nv_bfloat16* const dV = Vp_of(dst_stage);
    for (int idx = tid; idx < kGroupTokens * kValChunks; idx += kConsumerThreads) {
      const int t = idx / kValChunks;
      const int c = idx - t * kValChunks;
      const bool live = (t < gN) && (wmask != 0u);
      const int64_t tr = static_cast<int64_t>(bh) * p.T + (live ? Ra[t] : 0);
      cp_async_16(dV + t * kVpPitch + c * 8,
                  static_cast<const __nv_bfloat16*>(p.v) + tr * DV + c * 8, live);
    }
    //: **THE ROW-SCALE SCALARS.** The write gate and the FIXED levels' amplitudes --
    //: `1 + 2j` fp32 per token, gathered by token, so 4 bytes is the copy's width.
    //: They are the last stream that was still fetched at the point of use, and it
    //: was the PRODUCER warp that paid for it: the cook's own dependent global chain.
    //: docs/internals/consumer_kernel.md#row-scales
    float* const dS = pfS_of(dst_stage);
    for (int idx = tid; idx < kGroupTokens * kScPitch; idx += kConsumerThreads) {
      const int t = idx / kScPitch;
      const int c = idx - t * kScPitch;
      const bool in_group = t < gN;
      const int64_t tr = tok_base + (in_group ? Ra[t] : 0);
      //: **THE LEVEL IS RECOVERED AT COMPILE TIME**, by an unrolled ladder over the
      //: slab's columns rather than by indexing `base[]` with `c`: a runtime index
      //: into the owner's per-level register arrays would put them in LOCAL MEMORY
      //: and undo the batch. docs/internals/consumer_kernel.md#row-scales
      const float* src = p.g_write;
      bool live = in_group && (wmask != 0u);
#pragma unroll
      for (int l = 0; l + 1 < D; ++l) {
        //: Fixed levels (`l < j`) are ALWAYS fp32 (Option B) -- the one place the
        //: amplitude dtype is not the arm's storage type.
        if (c == 1 + 2 * l) {
          src = static_cast<const float*>(p.amp_read[l]) + tr * p.level_pitch_read[l] + base[l];
          live = in_group && l < j && rmask != 0u;
        }
        if (c == 2 + 2 * l) {
          src = static_cast<const float*>(p.amp_write[l]) + tr * p.level_pitch_write[l] + base[l];
          live = in_group && l < j && wmask != 0u;
        }
      }
      if (c == 0) src = p.g_write + static_cast<int64_t>(bh) * p.T + (in_group ? Ra[t] : 0);
      cp_async_4(dS + idx, src, live);
    }
    cp_async_commit();
  };

  //: THE RING'S COLD START -- the producer cooks `kSchedDepth` groups of run-ahead
  //: before the first visit, so the steady state never waits on production, and the
  //: pipeline's `S` stages are filled behind it. The residue the contract admits
  //: (item 6) is exactly this ramp plus the final drain.
  if (warp == 0) {
#pragma unroll 1
    for (int g = 0; g < kSchedDepth; ++g) produce_group(g);
  }
  //: THE ENTRY EPOCH, and the ONE barrier the pipeline adds anywhere -- once per
  //: kernel, never in the loop. It publishes the cold-start descriptors and the chunk
  //: descriptors to the threads that issue the first stages' copies; in the steady
  //: state the issue rides a barrier that already exists.
  __syncthreads();

  // ---- **THE PER-OWNER REGISTER HOISTS.** Everything the three hot walks read out of
  // ---- shared memory that is CONSTANT FOR THE OWNER -- the chunk descriptors, the
  // ---- leaf digits both panels index their staged rows by, and the per-leaf log
  // ---- retention -- is read HERE, once, into registers, and never read again. Each
  // ---- one is written once per owner and read once per visit per element, so the
  // ---- reads-per-write ratio is the owner's visit count; the register policy spends
  // ---- registers exactly here. docs/internals/consumer_kernel.md#walk-hoists
#pragma unroll
  for (int k = 0; k < kIssueIters; ++k) {
    const int idx = tid + k * kConsumerThreads;
    const int t = idx / kPfChunks;
    const int cg = idx - t * kPfChunks;
    const bool live = (idx < kGroupTokens * kPfChunks) && (cg < pf_nch);
    ch_live[k] = live;
    const int l = live ? ch_lvl[cg] : 0;
    const int so = live ? ch_src[cg] : 0;
    ch_am[k] = live ? static_cast<unsigned>(ch_atoms[cg]) : 0u;
    ch_tok[k] = live ? t : 0;
    ch_dst[k] = live ? (t * kPfPitch + cg * 8) : 0;
    ch_pr[k] = static_cast<const __nv_bfloat16*>(p.amp_read[l]) + so;
    ch_pw[k] = static_cast<const __nv_bfloat16*>(p.amp_write[l]) + so;
    ch_lpr[k] = p.level_pitch_read[l];
    ch_lpw[k] = p.level_pitch_write[l];
  }
  //: THE READ PANEL'S WALK, PER THREAD: its `kPanelIters` elements' leaf, token,
  //: destination index, staged-row offsets (one per level, the digit ALREADY ADDED to
  //: the level's slab offset) and log retention.
  int pi_r[kPanelIters], pt_r[kPanelIters], px_r[kPanelIters], po_r[kPanelIters][D];
  int pi_w[kPanelIters], pt_w[kPanelIters], px_w[kPanelIters], po_w[kPanelIters][D];
  float lk_r[kPanelIters], lk_w[kPanelIters];
#pragma unroll
  for (int k = 0; k < kPanelIters; ++k) {
    const int idx = tid + k * kConsumerThreads;
    const int ir = idx / kGroupTokens;
    const int tr_ = idx - ir * kGroupTokens;
    pi_r[k] = ir;
    pt_r[k] = tr_;
    px_r[k] = rp_index(ir, tr_);
    const int tw = idx / BC;
    const int iw = idx - tw * BC;
    pi_w[k] = iw;
    pt_w[k] = tw;
    px_w[k] = wp_index(BC, tw, iw);
#pragma unroll
    for (int l = 0; l < D; ++l) {
      po_r[k][l] = pf_off[l] + dig[l * BC + ir];
      po_w[k][l] = pf_off[l] + dig[l * BC + iw];
    }
    lk_r[k] = DECAY ? log_keep[ir] : 0.0f;
    lk_w[k] = DECAY ? log_keep[iw] : 0.0f;
  }
  //: `#pragma unroll 1`: the ramp is once per kernel, and unrolling it would inline
  //: the issue `S` more times against a 64-register budget.
#pragma unroll 1
  for (int st = 0; st < S; ++st) stage_issue(st, st);
  cp_async_wait_group<S - 1>();
  __syncthreads();

  // ---- THE WALK, OVER GROUPS. The producer's COOKED RING feeds one visit body per
  // ---- group. Words ascend and each word is consumed LSB-first, so a group's tokens
  // ---- are GLOBALLY ascending: the Gram's causal structure is order-relative, so
  // ---- STABILITY IS THE CORRECTNESS REQUIREMENT. Consumers ASSEMBLE NOTHING -- they
  // ---- read the descriptor warp 0 published `kSchedDepth` groups ago, whose
  // ---- publication barrier is the end-of-visit `__syncthreads()`.
  // ---- Group `gidx` is ring slot `gidx % kSlots` and pipeline stage `gidx % S`.
  // ---- docs/internals/consumer_kernel.md#group-assembler
  //: THE MASS COLUMN'S LIVE HALF, alternating per visit: the buffer the visit READS.
  float* Smass = Smass0;
  //: THE GRAM PUBLICATION EPOCH -- CTA-uniform (it counts `mixed` visits, and `mixed`
  //: is a function of the descriptor every warp reads), so every warp computes the
  //: same target for `gram_flag` without communicating.
  int gram_epoch = 0;
  //: The Gram's producing warps: its `NT` items are dealt out `it % kWarps`, so warp
  //: `w` produces iff `w < min(NT, kWarps)`.
  constexpr int kGramProd = NT < kWarps ? NT : kWarps;
  for (int gidx = 0;; ++gidx) {
    const int st = (S == 1) ? 0 : (gidx % S);
    const int* const R = g_ring + (gidx % kSlots) * kSchedSlotInts;
    //: `gN == 0` is the stream's end, as the producer saw it; the final flush is the
    //: ragged group (`gN < kGroupTokens`).
    const int gN = R[kSchedGN];
    if (gN == 0) break;
    const int flags = R[kSchedFlags];
    //: **THE STAGED MASKS -- the ONE pair every consumer walks.** A group's atom mask
    //: and its side flag are INDEPENDENT bitmaps (the flag is the fixed levels' rows,
    //: the mask is the atom bitlists), so the masks below are the side flag ALREADY
    //: FOLDED IN: `rmr != 0` IS "this visit staged read coefficients", and likewise
    //: `wmr` for the write side.
    const unsigned rmr =
        (!summary && (flags & 1) != 0) ? static_cast<unsigned>(R[kSchedRMask]) : 0u;
    const unsigned wmr = ((flags & 2) != 0) ? static_cast<unsigned>(R[kSchedWMask]) : 0u;
    const bool is_read = rmr != 0u;
    const bool is_write = wmr != 0u;
    const bool mixed = is_read && is_write;
    //: THE VISIT'S TWO HALVES OF THE MASS COLUMN. `Smass` is the entry column every
    //: reader takes; `SmOut` is the exit column the fold's term 3 alone writes, and it
    //: becomes the next visit's entry at the bottom of the body. The end-of-visit
    //: `__syncthreads()` is what publishes it, so the alternation adds no barrier.
    float* const SmOut = (Smass == Smass0) ? Smass1 : Smass0;
    //: THIS VISIT'S STAGE. Four regions, filled `S` visits ago and waited on at the
    //: bottom of the previous visit.
    char* const stage = stage_at(st);

    //: **THE VALUE FRAGMENT IS TAKEN FIRST, AND THAT IS WHAT FREES ITS STAGE.** The
    //: tile's whole consumption is four registers per owning warp -- one A operand
    //: serves the fold's every state tile AND the intra term -- so lifting it here,
    //: before the staging barrier, retires the region for the rest of the visit and
    //: lets the same barrier that publishes the panels also license the refill. The
    //: ROW SCALE is applied to the fragment afterwards, in registers.
    //: docs/internals/consumer_kernel.md#value-stream
    unsigned va0 = 0u, va1 = 0u, va2 = 0u, va3 = 0u;
    if (owns && is_write) trans_frag_a(Vp_of(stage), kVpPitch, 0, m0, lane, va0, va1, va2, va3);

    // ---- THE STAGING, AS A FUNCTION OF THE VISIT: everything the compute consumes
    // ---- but does not produce. Slots `[gN, kGroupTokens)` of a ragged group stage
    // ---- EXACT-ZERO coefficient rows and zero row scales, so they are inert through
    // ---- every GEMM. STAGING IS BRANCHLESS WITHIN A LIVE GROUP (fetch consults
    // ---- nothing -- the frozen contract): dead atoms stage the exact zeros the
    // ---- tensors hold; the ATOM MASKS ARE COMPUTE-ONLY.
    // ---- docs/internals/consumer_kernel.md#staging

    // ---- DECAY, step (a): the clock prefix, `P_ref`, `s0`, guarded by the write side
    // ---- STRUCTURALLY. Runs over ALL `BC` leaves (elementwise surplus-ALU work; keeps
    // ---- every leaf's clk/p_ref/s0 FINITE for the panels -- the 0*NaN discipline).
    // ---- docs/internals/consumer_kernel.md#decay-clock
    if (DECAY && is_write) {
      for (int i = tid; i < BC; i += kConsumerThreads) {
        //: THE LEAF'S CLOCK ROWS, HOISTED OUT OF THE TOKEN LOOP: the digit is a leaf
        //: constant, so the level's clock row base is too -- one register per level
        //: against `gN` shared reads of `dig`. docs/internals/consumer_kernel.md#walk-hoists
        const float* ckb[D];
#pragma unroll
        for (int l = 0; l < D; ++l) ckb[l] = p.amp_clock[l] + dig[l * BC + i];
        float acc = 0.0f;
        for (int t = 0; t < gN; ++t) {
          float cl = 1.0f;
#pragma unroll
          for (int l = 0; l < D; ++l) {
            cl *= ckb[l][(tok_base + R[t]) * p.level_pitch_clock[l]];
          }
          //: HAZARD clock-prefix-fp32 -- docs/internals/consumer_kernel.md#clock-prefix-fp32
          acc += cl;
          clk[t * CKP + i] = acc;
        }
        //: A ragged group's padded slots take a FINITE zero prefix so both panel
        //: factors stay finite (`keep^0 == 1`); their coefficient rows are exact zeros
        //: anyway.
        for (int t = gN; t < kGroupTokens; ++t) clk[t * CKP + i] = 0.0f;
        const float pr = 0.5f * acc;
        p_ref[i] = pr;
        s0[i] = expf(pr * log_keep[i]);
      }
      __syncthreads();
      // ---- DECAY, step (c): the entry state enters the reference frame, ONCE PER
      // ---- VISIT. `s0` multiplies the STATE -- in REGISTERS, two scalar `s0` loads
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
            float* const stf = St + ai * 8 + half * 4;
            stf[0] *= sa;
            stf[1] *= sb;
            stf[2] *= sa;
            stf[3] *= sb;
          }
        }
      }
      for (int i = tid; i < BC; i += kConsumerThreads) Smass[i] *= s0[i];
    }

    // ---- §3 STEP 3's ROW SCALES, REDUCED FROM THE STAGED SCALARS. The FIXED levels'
    // ---- amplitudes are CONSTANT ALONG THE LEAF AXIS -- `j` scalars per member token
    // ---- per side, common to every atom's Kronecker coefficient -- so their per-token
    // ---- product is an OWNER-LEVEL fact and obeys the same multiplicity rule
    // ---- membership does: formed ONCE PER GROUP, never per atom. The scalars
    // ---- themselves rode the stage pipeline, so no thread gathers them here.
    // ---- A ragged group's padded slots take EXACT ZERO scales -- the copy zero-filled
    // ---- them -- which is what makes their coefficient rows inert through every GEMM
    // ---- and the mass fold's `sw[t]` term vanish.
    // ---- docs/internals/consumer_kernel.md#row-scales
    for (int t = tid; t < kGroupTokens; t += kConsumerThreads) {
      const float* const row = pfS_of(stage) + t * kScPitch;
      float rs = (is_read && t < gN) ? 1.0f : 0.0f;
      float ws = (is_write && t < gN) ? row[0] : 0.0f;
#pragma unroll
      for (int l = 0; l < D; ++l) {
        if (l >= j) break;
        rs *= row[1 + 2 * l];
        ws *= row[2 + 2 * l];
      }
      sr[t] = rs;
      sw[t] = ws;
    }

    // ---- THE READ PANEL, `Rp[leaf][token]` (§3 steps 2-3). It is the READOUT's and
    // ---- the Gram's B operand, `k = leaf` on the row axis and `n = token` on the
    // ---- column axis, which is the form `ldmatrix.trans` reads.
    // ---- docs/internals/consumer_kernel.md#read-panel
    if (is_read) {
      //: BRANCHLESS (fetch consults nothing -- the frozen contract): every atom's
      //: slots are staged by the pure affine walk; a dead atom's positions hold the
      //: exact zeros the amplitude tensor already holds (entmax), so the fetch layer
      //: cannot be wrong -- the masks are COMPUTE-ONLY (MMA skip). A padded token slot
      //: takes an exact zero, which is what makes its column inert.
      //: THE fp16 APPLICATION IS CARVED OUT (U07-d built, adjudicated 2026-08-06): an
      //: ADMITTED hot routing (panel bound 23.3 of 40) put the survival factor at
      //: ~e^15+, past fp16's 65504 max -- measured NaN, not a rounding. The panels
      //: apply in fp32 against the bf16 STORAGE (the memory win survives the carve).
      //: Evidence: workflows/perf/p26_evidence/m2_fp16_breach.txt.
#pragma unroll
      for (int k = 0; k < kPanelIters; ++k) {
        const int i = pi_r[k];
        const int t = pt_r[k];
        float ac = 0.0f;
        if (t < gN) {
          //: THE STAGED ROW, AT AN OFFSET THAT IS ALREADY A REGISTER: the level's slab
          //: offset and the leaf's digit are OWNER CONSTANTS, summed once at the entry
          //: epoch, so the walk's whole address is the staged row base plus one
          //: register -- no shared-memory read of `dig` and no add.
          const __nv_bfloat16* const pf = pfR_of(stage) + t * kPfPitch;
          ac = 1.0f;
#pragma unroll
          for (int l = 0; l < D; ++l) {
            if (l < j) continue;  // fixed levels hoist into the row scale (§3 step 3)
            ac *= __bfloat162float(pf[po_r[k][l]]);
          }
          if (DECAY && is_write) {
            ac *= panel_read_factor(clk[t * CKP + i], p_ref[i], lk_r[k]);
          }
        }
        stage_put(Rp[px_r[k]], ac);
      }
    }
    // ---- THE WRITE PANEL, `Wp[token][leaf]` -- stored TOKEN-MAJOR because it is the
    // ---- one operand read BOTH ways: the Gram's A (`m = token, k = leaf`) and the
    // ---- fold's B (`k = token, n = leaf`). One buffer, two `ldmatrix` atoms.
    // ---- docs/internals/consumer_kernel.md#write-panel
    if (is_write) {
      //: BRANCHLESS, as the read panel: every atom staged, zeros ride self-masking,
      //: masks stay compute-only.
#pragma unroll
      for (int k = 0; k < kPanelIters; ++k) {
        const int t = pt_w[k];
        const int i = pi_w[k];
        float ac = 0.0f;
        if (t < gN) {
          const __nv_bfloat16* const pf = pfW_of(stage) + t * kPfPitch;
          ac = 1.0f;
#pragma unroll
          for (int l = 0; l < D; ++l) {
            if (l < j) continue;
            ac *= __bfloat162float(pf[po_w[k][l]]);
          }
          if (DECAY) {
            ac *= panel_write_factor(clk[t * CKP + i], p_ref[i], lk_w[k]);
          }
        }
        //: §6.4 PASS 1's per-leaf inclusion, COEFFICIENT HALF: a leaf is included iff
        //: some token's staged coefficient is nonzero. The store is IDEMPOTENT -- every
        //: writer writes the same `1` -- so the token axis needs no reduction.
        //: HAZARD summary-prequant -- docs/internals/consumer_kernel.md#summary-prequant
        if (summary && ac != 0.0f) sum_slot[i] = 1;
        stage_put(Wp[px_w[k]], ac);
      }
    }
    __syncthreads();

    //: **THE ROW SCALE MEETS THE VALUE FRAGMENT, IN REGISTERS.** The fold contracts
    //: over TOKENS, so the write gate's per-token scale cannot hoist out of the sum
    //: and must ride the value operand -- but it need not ride its STORAGE. `a0`/`a1`
    //: carry tokens `2r, 2r + 1` and `a2`/`a3` tokens `2r + 8, 2r + 9`, the same
    //: rounding the staged product used to take.
    if (owns && is_write) {
      const float w0 = sw[2 * _sr], w1 = sw[2 * _sr + 1];
      const float w2 = sw[2 * _sr + 8], w3 = sw[2 * _sr + 9];
      scale_frag_pair(va0, w0, w1);
      scale_frag_pair(va1, w0, w1);
      scale_frag_pair(va2, w2, w3);
      scale_frag_pair(va3, w2, w3);
    }

    //: **THE NEXT STAGE'S ISSUE, AND IT COSTS NO BARRIER.** The barrier above is
    //: already there to publish the staged operands; it equally guarantees that every
    //: warp has finished READING this stage -- the panels above and the value fragment
    //: before them -- which is the condition for refilling it. From here the copies for
    //: group `gidx + S` run concurrently with this visit's entire compute AND with the
    //: `S - 1` visits after it; only the group about to be consumed is ever waited on.
    stage_issue(gidx + S, st);

    // ---- PHASE A -- the two GEMMs that read only the ENTRY state and the staged
    // ---- panels: the readouts and the Gram. They share a phase because neither
    // ---- writes what the other reads.

    // ---- GEMM 1: THE READOUT, `y^T = S^T . Rp`, LIVE READ ATOMS ONLY. **THE A
    // ---- OPERAND IS THE WARP'S OWN STATE REGISTERS**, narrowed to bf16 words at the
    // ---- fragment and never loaded: the k-loop's only memory traffic is the B
    // ---- fragment. A warp holds one CHANNEL TILE and a LEAF RANGE, so the `kReps`
    // ---- warps of a channel tile each carry a PARTIAL over their own leaves -- and
    // ---- they need no reduction, because the emission below is already an
    // ---- `atomicAdd` split-K. docs/internals/consumer_kernel.md#gemm-readout
    float y[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      y[nt][0] = 0.0f;
      y[nt][1] = 0.0f;
      y[nt][2] = 0.0f;
      y[nt][3] = 0.0f;
    }
    if (owns && is_read) {
      const unsigned rmask = rmr & own_mask;
      if (rmask == own_mask) {
        mma_readout_tiles<kAPW>(St, Rp, atom0 * kMmaKQuantum, lane, y);
      } else {
        //: **THE SPARSE ARM WALKS THE WARP'S OWN ATOM SLOTS AT COMPILE TIME**, testing
        //: the mask bit, rather than walking the set bits: a runtime index into `St`
        //: would put the state in LOCAL MEMORY and undo the whole batch.
#pragma unroll
        for (int ai = 0; ai < kAPW; ++ai) {
          if (((rmask >> (atom0 + ai)) & 1u) == 0u) continue;
          mma_readout_tiles<1>(St + ai * 8, Rp, (atom0 + ai) * kMmaKQuantum, lane, y);
        }
      }
    }

    // ---- THE MASS COLUMN, TERM 1 -- `dm = broadcast(Smass) . Rp`, the SAME B OPERAND
    // ---- the readout uses with the mass column riding the A operand's `k` axis. It
    // ---- reads the visit's ENTRY column, which the fold's term 3 does not write --
    // ---- term 3 writes the EXIT half -- so the two are unordered. Its items are
    // ---- spread over the warps: the mass column is shared, so its terms carry no
    // ---- residency constraint. Live read atoms only.
    // ---- docs/internals/consumer_kernel.md#mass-term1
    //: HAZARD mass-split -- docs/internals/consumer_kernel.md#mass-split
    float dm[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const bool dm_owner = (warp < 2) && is_read;
    const int dm_nt = warp;
    if (dm_owner) {
      if (rmr == kFullMask) {
        mma_mass_readout<kQPO>(Smass, Rp, 0, dm_nt * 8, lane, dm[0], dm[1], dm[2], dm[3]);
      } else {
        for (unsigned m = rmr; m; m &= m - 1) {
          const int q = __ffs(m) - 1;
          mma_mass_readout<1>(Smass, Rp, q * kMmaKQuantum, dm_nt * 8, lane, dm[0], dm[1], dm[2],
                              dm[3]);
        }
      }
    }

    // ---- GEMM 2: THE GRAM, HELD TRANSPOSED -- `G^T += Wp . Rp^T` over atoms live on
    // ---- BOTH sides, under §5's INCLUSIVE causal mask. THE TRANSPOSE IS FREE -- the D
    // ---- fragment's rows become `token_b` and its columns `token_a`, which is the
    // ---- `k`-major form the intra term's B operand wants.
    // ---- docs/internals/consumer_kernel.md#gemm-gram
    if (mixed) {
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        if (nt % kWarps != warp) continue;
        //: **THE BLOCK IS SKIPPED ON THE SIDES, AND THE MMA ON THE ATOMS.** These are
        //: different conditions and the difference is load-bearing: GEMM 3 reads `G^T`
        //: whenever both sides are live, so a block whose sides are live but whose atom
        //: masks are DISJOINT must still be STORED -- as the exact zeros it is -- or the
        //: intra term reads a previous visit's Gram.
        const unsigned gmask = rmr & wmr;
        const int n0 = nt * 8;
        //: An empty `gmask` issues no `mma.sync` and the accumulators stay the exact
        //: zeros the block is; the stores below still run.
        float gd0 = 0.0f, gd1 = 0.0f, gd2 = 0.0f, gd3 = 0.0f;
        if (gmask == kFullMask) {
          mma_gram_atoms<kQPO>(Wp, kWpPitch, Rp, n0, lane, gd0, gd1, gd2, gd3);
        } else {
          for (unsigned m = gmask; m; m &= m - 1) {
            const int q = __ffs(m) - 1;
            mma_pitched_tile(Wp, kWpPitch, 0, q * kMmaKQuantum, Rp, kTokPitch, q * kMmaKQuantum, n0,
                             lane, gd0, gd1, gd2, gd3);
          }
        }
        CONSUMER_D_ROWS(lane);
        const int c0 = n0 + 2 * _r, c1 = c0 + 1;
        //: The causal mask, in the transposed frame: the D fragment's ROW is `token_b`
        //: and its COLUMN is `token_a`, so §5's inclusive `u <= t` keeps the entries at
        //: or ABOVE the diagonal.
        stage_put(Gm[_g * kTokPitch + c0], (c0 >= _g) ? gd0 : 0.0f);
        stage_put(Gm[_g * kTokPitch + c1], (c1 >= _g) ? gd1 : 0.0f);
        stage_put(Gm[(_g + 8) * kTokPitch + c0], (c0 >= _g + 8) ? gd2 : 0.0f);
        stage_put(Gm[(_g + 8) * kTokPitch + c1], (c1 >= _g + 8) ? gd3 : 0.0f);
      }
      //: THE PUBLICATION HALF OF THE CURSOR. The fence orders this warp's `Gm` stores
      //: before the count that advertises them; one lane carries the arrival because
      //: the unit of publication is the WARP's whole share.
      if (warp < kGramProd) {
        __threadfence_block();
        if (lane == 0) atomicAdd(gram_flag, 1);
      }
    }

    // ---- GEMM 4, THE BOUNDARY FOLD `S^T += Vp . Wp`, AND THIS IS WHERE THE
    // ---- REGISTER-RESIDENT STATE PAYS: **the `mma.sync` ACCUMULATES DIRECTLY INTO THE
    // ---- STATE**, `D == C == St`. There is no read-modify-write, no format boundary
    // ---- and no shared-memory traffic of any kind on the state's numerator. The
    // ---- partition is the OWNERSHIP RULE itself -- one channel tile and one atom range
    // ---- per warp -- so it is static, uniform-cost and costs no atomic.
    // ----
    // ---- **IT RUNS BEFORE THE GRAM IS WAITED ON, AND THAT IS THE SCHEDULING POINT.**
    // ---- The fold reads only the staged panel and the value fragment and writes only
    // ---- register state and the exit mass column, so it is independent of `Gm` --
    // ---- which makes it the work every warp does while the Gram's producers finish.
    // ---- docs/internals/consumer_kernel.md#gemm-fold
    if (is_write && owns) {
      //: **ONE A OPERAND FEEDS EVERY TILE THE WARP OWNS**, and the B walk is the
      //: two-stage chain: `mma_fold_tiles` issues tile `s + 1`'s `ldmatrix` before tile
      //: `s`'s multiply. The ATOM one is the only test left -- a dead atom's rows are
      //: exact zeros and its `s0` is exactly 1, so the skip is the identity.
      mma_fold_tiles<kAPW>(va0, va1, va2, va3, Wp, kWpPitch, atom0, wmr, lane, St);
      // ---- DECAY's EXIT SCALE RIDES THE ACCUMULATOR, ONCE PER VISIT: one multiply on
      // ---- a register value. docs/internals/consumer_kernel.md#fold-epilogue
      if (DECAY) {
#pragma unroll
        for (int ai = 0; ai < kAPW; ++ai) {
          if (((wmr >> (atom0 + ai)) & 1u) == 0u) continue;
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            const int i0 = (atom0 + ai) * kMmaKQuantum + 8 * half + 2 * _sr;
            const float sa = s0[i0], sb = s0[i0 + 1];
            float* const stf = St + ai * 8 + half * 4;
            stf[0] *= sa;
            stf[1] *= sb;
            stf[2] *= sa;
            stf[3] *= sb;
          }
        }
      }
    }
    // ---- THE MASS COLUMN, TERM 3 -- the fold's own term,
    // ---- `SmOut = s0 * (Smass + broadcast(sw) . Wp)`, ON THE SAME TENSOR CORE AS THE
    // ---- FOLD ITSELF: one `mma.sync` per 8-leaf tile against a broadcast `sw`.
    // ---- **IT READS THE ENTRY COLUMN AND WRITES THE EXIT ONE**, which is what lets it
    // ---- run concurrently with the readout's term 1 instead of behind a barrier from
    // ---- it. A tile the visit does not write -- and every tile of a read-only visit --
    // ---- is CARRIED ACROSS, because the exit column is the next visit's entry.
    // ---- docs/internals/consumer_kernel.md#mass-pingpong
    for (int lt = warp; lt < BC / 8; lt += kWarps) {
      CONSUMER_D_ROWS(lane);
      const int n0 = lt * 8;
      const bool live = is_write && (((wmr >> (lt / 2)) & 1u) != 0u);
      float m0v = 0.0f, m1v = 0.0f, m2v = 0.0f, m3v = 0.0f;
      if (live) mma_bcastA_tile(sw, Wp, kWpPitch, 0, n0, lane, m0v, m1v, m2v, m3v);
      (void)m2v;
      (void)m3v;
      //: Output ROW 0 of a broadcast-A tile carries the result, so only the four lanes
      //: with `g == 0` touch the column.
      if (_g == 0) {
        const int i0 = n0 + 2 * _r, i1 = i0 + 1;
        SmOut[i0] = live ? (DECAY ? s0[i0] : 1.0f) * (Smass[i0] + m0v) : Smass[i0];
        SmOut[i1] = live ? (DECAY ? s0[i1] : 1.0f) * (Smass[i1] + m1v) : Smass[i1];
      }
    }

    // ---- **THE GRAM'S PUBLICATION, CURSOR-PACED RATHER THAN A PHASE BARRIER.** What
    // ---- is left of the visit's old rendezvous is a pure producer-consumer over `Gm`
    // ---- between a KNOWN, COMPILE-TIME set of producing warps and everyone else, so it
    // ---- is paced by a counter: producers fence their stores and add one, consumers
    // ---- spin to the visit's target. A warp that is not a producer never blocks a
    // ---- producer, and no warp waits for the CTA -- only for `kGramProd` arrivals. The
    // ---- target is MONOTONE over the kernel, so the counter is never reset and the
    // ---- reset never needs its own rendezvous. `Gm`'s reuse across visits is guarded
    // ---- by the end-of-visit `__syncthreads()`.
    // ---- BOTH SIDES REQUIRE `mixed`: a single-sided visit has no Gram and pays nothing
    // ---- at all here. docs/internals/consumer_kernel.md#gram-publication
    if (mixed) {
      ++gram_epoch;
      const int target = gram_epoch * kGramProd;
      volatile int* const gf = gram_flag;
      //: THE POLL IS BARE, AND THAT IS A MEASUREMENT rather than an oversight. A
      //: shared-memory poll is issue-ACTIVE, so a backoff was expected to matter most
      //: exactly here -- six of eight warps waiting on two producers is the largest
      //: waiter-to-producer ratio the matrix has. A `__nanosleep(64)` between polls
      //: measured 12.78 ms against the bare form's 12.81 (dense `N = L = 4096`,
      //: block-min-median of three), so the simpler form is the one that ships.
      //: docs/internals/consumer_kernel.md#gram-publication
      while (*gf < target) {
      }
      __threadfence_block();
    }

    // ---- PHASE B -- GEMM 3, the intra term `y^T += Vp . G^T`, accumulating into the
    // ---- SAME fragments the readout filled, against the SAME value fragment the fold
    // ---- used. There is NO MASK anywhere here, because the zeros are already IN `Gm`.
    // ---- Token-axis contraction: no atom mask applies. An item belongs to ONE of the
    // ---- `kReps` warps holding that channel tile's `y` -- the others carry only their
    // ---- readout partial, and the emission sums the pieces.
    // ---- docs/internals/consumer_kernel.md#phase-c
    if (mixed && owns) {
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        if (nt % kReps != hrep) continue;
        mma_fragA_pitchedB(va0, va1, va2, va3, Gm, kTokPitch, 0, nt * 8, lane, y[nt][0], y[nt][1],
                           y[nt][2], y[nt][3]);
      }
    }
    if (mixed && dm_owner) {
      //: THE MASS COLUMN, TERM 2 -- the intra term's denominator,
      //: `dm += broadcast(sw) . G^T`. The causal mask is already IN `Gm`, so the whole
      //: contraction is the `u <= t` sum.
      //: HAZARD mass-split -- docs/internals/consumer_kernel.md#mass-split
      mma_bcastA_tile(sw, Gm, kTokPitch, 0, dm_nt * 8, lane, dm[0], dm[1], dm[2], dm[3]);
    }

    // ---- SPLIT-K EMISSION of this visit's partials, straight out of the D fragments,
    // ---- scattered through the group's token list (the P14 microbench priced exactly
    // ---- this pattern: banded, DRAM-byte ratio 1.0009x at the measured gap). The
    // ---- `out == 0` early-out runs FIRST so a ragged group's padded slots (`sr == 0`,
    // ---- coefficient rows exactly zero) never read the token list at all; members are
    // ---- `< T` by construction. The `!= 0` guard is a PERFORMANCE SKIP ON A VALUE,
    // ---- never a membership decision.
    // ---- **THE FRAGMENT IS TRANSPOSED AND THAT IMPROVES THE SCATTER**: the eight lanes
    // ---- sharing a `token` now hold eight CONSECUTIVE channels, so one `atomicAdd`
    // ---- instruction covers four 32-byte segments where the untransposed fragment
    // ---- covered eight. The `kReps` partials per output are what the split-K was
    // ---- already built for. docs/internals/consumer_kernel.md#split-k-emission
    if (owns && is_read) {
      CONSUMER_D_ROWS(lane);
      const int ch0 = m0 + _g, ch1 = ch0 + 8;
      auto emit = [&](int t, int ch, float val) {
        const float out = sr[t] * val;
        if (out == 0.0f) return;
        atomicAdd(&p.num[(static_cast<int64_t>(bh) * p.T + R[t]) * DV + ch], out);
      };
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int t0e = nt * 8 + 2 * _r, t1e = t0e + 1;
        emit(t0e, ch0, y[nt][0]);
        emit(t1e, ch0, y[nt][1]);
        emit(t0e, ch1, y[nt][2]);
        emit(t1e, ch1, y[nt][3]);
      }
    }
    //: THE DENOMINATOR, OUT OF THE MASS FRAGMENTS. A broadcast A makes every output ROW
    //: of `dm` the same value, so row 0 -- the lanes with `g == 0` -- carries the tile's
    //: eight tokens. The `!= 0` guard retires a ragged group's padded slots on the same
    //: rule the numerator runs on: their `sr` is an exact zero, so the token list is
    //: never read for them.
    if (dm_owner) {
      CONSUMER_D_ROWS(lane);
      if (_g == 0) {
        const int t0e = dm_nt * 8 + 2 * _r, t1e = t0e + 1;
        const float da = sr[t0e] * dm[0];
        if (da != 0.0f) atomicAdd(&p.den[static_cast<int64_t>(bh) * p.T + R[t0e]], da);
        const float db = sr[t1e] * dm[1];
        if (db != 0.0f) atomicAdd(&p.den[static_cast<int64_t>(bh) * p.T + R[t1e]], db);
      }
    }

    // ---- §6.4 PASS 1's PER-LEAF EPILOGUE, once per VISIT. The COEFFICIENT half of the
    // ---- inclusion test lives in the panel build; what is left here has no
    // ---- pre-quantization site: docs/internals/consumer_kernel.md#pass1-epilogue
    if (is_write && summary && DECAY) {
      for (int i = tid; i < BC; i += kConsumerThreads) {
        if (s0[i] != 1.0f) sum_slot[i] = 1;
        a_state[i] *= s0[i] * s0[i];
      }
    }
    //: THE COOK-AHEAD. The producer cooks group `gidx + kSchedDepth` into the slot the
    //: visit `kSlots` back vacated. Production runs CONCURRENTLY with this visit's
    //: compute and is published by the end-of-visit barrier below, so NOTHING WAITS ON
    //: IT.
    if (warp == 0) produce_group((gidx + kSchedDepth) % kSlots);
    //: **THE PIPELINE'S ONLY WAIT, AND IT IS FOR ONE STAGE.** `cp.async` completion is
    //: visible to the ISSUING THREAD only, so the cross-warp publication is the
    //: end-of-visit rendezvous below -- which is why the wait goes here and not at the
    //: top of the next visit, where it would need a barrier of its own. `S - 1` groups
    //: stay in flight across it: the link runs ahead of the visit that consumes it by
    //: the full depth its carve admits.
    cp_async_wait_group<S - 1>();
    //: THE MASS COLUMN'S EXIT HALF BECOMES THE NEXT VISIT'S ENTRY HALF. The rotation is
    //: a pointer, published by the barrier below with everything else.
    Smass = SmOut;
    __syncthreads();
  }
  //: THE FINAL DRAIN: the stages issued past the stream's end have no consumer, but a
  //: kernel does not exit with copies outstanding into memory it is about to release.
  cp_async_wait_all();

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
