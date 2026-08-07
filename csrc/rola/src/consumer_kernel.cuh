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
//   1. THE STATE LIVES IN SMEM (`S[BC][SP]`), not in registers. That is what
//      decouples the thread count from the state layout, frees the CTA to 8
//      warps, and puts every per-visit term on the tensor cores as one of four
//      GEMMs (readout, Gram, intra, fold).
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
#include "intra_panel.cuh"
#include "consumer.cuh"
//: The three layers, in one include: `mma_tile.cuh` -> `smem_layout.cuh` ->
//: `pipeline.cuh`. docs/internals/mma_tile.md#cute-pin
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
#define ROLA_CONSUMER_CTAS_EXPR                                    \
  ctas_for_residency(consumer_smem_bytes<BC, DV, D, DECAY>(),      \
                     consumer_reg_budget(BC, DECAY, DV))
#endif
template <int BC, int DV, int D, bool DECAY>
__global__ __launch_bounds__(kConsumerThreads, ROLA_CONSUMER_CTAS_EXPR) void
consumer_forward_kernel(ConsumerParams p) {
  constexpr int kGT = kGroupTokens;  // 16: the group IS the MMA m-quantum
  //: `SP = DV + 4` IS DERIVED FROM THE bf16 MMA B MAP: `k = k0 + 2r` collides up to
  //: four ways at `DV + 1`; `DV + 4` makes the bank `(8r + g) % 32`, covering all 32
  //: exactly once. docs/internals/consumer_kernel.md#state-pitch
  constexpr int SP = DV + 4;
  constexpr int NT_DV = DV / 8;  // n8 column tiles of a DV-wide output
  extern __shared__ char smem_raw[];

  // ---- THE SMEM CARVE. Every region, its pitch and its measured residency cost:
  // ---- docs/internals/smem_layout.md#smem-ledger
  constexpr int BTP = kWtPitch;  // WT's row pitch, LEAF-MAJOR
  constexpr int GMP = kGT + 8;   // Gm's row pitch (bf16)
  constexpr int CKP = BC + 4;    // clk's row pitch, token-major
  constexpr int kQPO = BC / kMmaKQuantum;
  static_assert(BC % kMmaKQuantum == 0, "BC must be a whole number of MMA k-quanta");
  //: The atom masks' FULL value: every atom of the owner's slice live this visit.
  //: The CTA-uniform `mask == kFullMask` test is what selects the statically
  //: unrolled dense arm -- dense inputs pay zero mask instructions.
  constexpr unsigned kFullMask = (kQPO >= 32) ? 0xffffffffu : ((1u << kQPO) - 1u);
  constexpr int kRfWords = frag_a_words(BC);
  //: The staged-slot strides the atom-major fold walks (one slot today:
  //: `kFoldStagedGroups` is pinned 1 and these fold to zero offsets).
  constexpr int kWtSlotElems = BC * kWtPitch;
  constexpr int kVwSlotElems = vw_elems<DV>();

  float* S = reinterpret_cast<float*>(smem_raw);
  char* carve = smem_raw + align4(static_cast<size_t>(BC) * SP * sizeof(float));
  __nv_bfloat16* WT = reinterpret_cast<__nv_bfloat16*>(carve);
  carve += align4(static_cast<size_t>(BC) * BTP * sizeof(__nv_bfloat16));
  constexpr size_t kVwSlotBytes = vw_union_bytes<BC, DV, DECAY>();
  //: `ldmatrix` TAKES A 16-BYTE-ALIGNED ROW ADDRESS, so both staged bases must be
  //: 16-byte aligned -- a requirement `align4` does NOT imply. Stated, not trusted:
  //: docs/internals/consumer_kernel.md#ldsm-alignment
  constexpr size_t kWtOffsetBytes = align4(static_cast<size_t>(BC) * SP * sizeof(float));
  constexpr size_t kVwOffsetBytes =
      kWtOffsetBytes + align4(static_cast<size_t>(BC) * BTP * sizeof(__nv_bfloat16));
  static_assert(kVwOffsetBytes % 16 == 0 && kWtOffsetBytes % 16 == 0,
                "WT and Vw must be 16-byte aligned for ldmatrix");
  __nv_bfloat16* Vw = reinterpret_cast<__nv_bfloat16*>(carve);
  //: `clk` ALIASES `Vw`'s bytes (dead before `Vw` is live; the reservation is the
  //: byte-max over the two tenants). Declared AT THE CARVE because the alias is the
  //: load-bearing fact. docs/internals/smem_layout.md#vw-clk-union
  float* clk = reinterpret_cast<float*>(carve);
  carve += kVwSlotBytes;
  float* den = reinterpret_cast<float*>(carve);
  int* dig = reinterpret_cast<int*>(den + kGT);  // [D][BC]
  unsigned* Rf = reinterpret_cast<unsigned*>(dig + D * BC);
  __nv_bfloat16* Gm = reinterpret_cast<__nv_bfloat16*>(Rf + kRfWords);
  float* log_keep = reinterpret_cast<float*>(Gm + kGT * GMP);
  float* p_ref = log_keep + (DECAY ? BC : 0);
  float* s0 = p_ref + (DECAY ? BC : 0);
  int* ctl = reinterpret_cast<int*>(s0 + (DECAY ? BC : 0));
  //: THE COOKED SCHEDULE RING -- change 2's cooked stage. `kSchedSlots` group
  //: descriptors (`kSchedSlotInts` ints each; the field offsets are
  //: `kSchedGN/RMask/WMask/Flags`), written by the PRODUCER WARP `kSchedDepth`
  //: groups ahead and published by the end-of-visit `__syncthreads()` of the
  //: visit that produced them. Every consumer warp READS the descriptor instead
  //: of re-deriving it: the base's 8x-redundant assembler is what this deletes.
  //: docs/internals/consumer_kernel.md#sched-ring
  int* g_ring = ctl + ctl_words(BC);
  //: THE PER-OWNER UPPER PRODUCTS, one pair of `kGT` floats per ring slot: the
  //: read row scale then the write row scale for each of the slot's member
  //: tokens. Cooked by the producer beside the member word (the stream-inventory
  //: amendment) and consumed as `sr` / `sw`.
  //: docs/internals/consumer_kernel.md#upper-product
  float* g_up = reinterpret_cast<float*>(g_ring + kSchedSlots * kSchedSlotInts);
  static_assert(
      !DECAY || static_cast<size_t>(kGT) * CKP * sizeof(float) <= vw_union_bytes<BC, DV, DECAY>(),
      "clk aliases the Vw region and must fit IN BYTES");
  float* a_state = reinterpret_cast<float*>(Rf);         // [BC], the survival product A
  int* sum_slot = reinterpret_cast<int*>(Rf) + BC;       // [BC]
  int* sum_count = reinterpret_cast<int*>(Rf) + 2 * BC;  // [1]
  static_assert(frag_a_words(BC) >= 2 * BC + 1,
                "pass 1's summary state must fit inside the read-coefficient buffer");

  // ---- EACH OPERAND'S PITCH PROVES ITS OWN CONFLICT-FREEDOM.
  // ---- docs/internals/consumer_kernel.md#pitches
  static_assert(banks_distinct(1, 2 * SP), "S: GEMM1 B operand conflicts");
  static_assert(banks_distinct_u16(1, 2 * BTP), "WT: GEMM2 B operand conflicts");
  static_assert(banks_distinct(GMP / 2, 1), "Gm: GEMM3 A operand conflicts");
  static_assert(banks_distinct_u16(BTP, 2), "WT: GEMM4 A operand conflicts");
  static_assert(pitched_ldsm_banks_ok<kMmaKQuantum, kGT, kWtPitch>(),
                "WT: the ldmatrix fetch conflicts or is misaligned at kWtPitch");
  static_assert(stage_offset_matches_cute<kGT, DV>(),
                "Vw: the closed-form address is not CuTe's layout");
  static_assert(stage_layout_is_bijection<kGT, DV>(),
                "Vw: the swizzled layout does not tile [0, kGT*DV)");
  static_assert(stage_ldsm_banks_ok<kGT, DV>(),
                "Vw: the ldmatrix.x2.trans fetch conflicts or is misaligned");
  static_assert(stage_store_banks_ok<kGT, DV>(), "Vw: the staging store conflicts");

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
  constexpr int kWarps = kConsumerThreads / 32;
  //: The Gram's writer set: `kGT/8 = 2` warp tiles. The phase-C gate is narrowed to
  //: the writers (measured scope; free on bf16). docs/internals/consumer_kernel.md#gram-gate
  constexpr int kGramWarps = (kGT / 8 < kWarps) ? kGT / 8 : kWarps;

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
  if (summary || from_ws) {
    for (int idx = tid; idx < BC * cols; idx += kConsumerThreads) {
      const int i = idx / cols;
      const int c = idx - i * cols;
      S[i * SP + c] = summary ? 0.0f : ws_entry[i * cols + c];
    }
  } else {
#pragma unroll
    for (int q = 0; q < kQPO; ++q) {
      const int64_t row0 = atom_row_base<BC>(p, bh, owner, q);
      for (int idx = tid; idx < kMmaKQuantum * cols; idx += kConsumerThreads) {
        const int r = idx / cols;
        const int c = idx - r * cols;
        S[(q * kMmaKQuantum + r) * SP + c] = (row0 >= 0) ? p.state[(row0 + r) * cols + c] : 0.0f;
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

  // ---- THE PRODUCER (change 2, contract items 2/3/5). WARP 0 ALONE runs the
  // ---- membership scan and the ballot, and publishes COOKED GROUP DESCRIPTORS
  // ---- into `g_ring`. In the base every one of the 8 warps re-derived this
  // ---- identically -- including `2 * QPO` CTA-uniform GLOBAL loads of the atom
  // ---- bitlists PER BALLOT ROUND PER WARP. Cooking ONCE PER OWNER is the frozen
  // ---- contract's rule, and it is an ISSUE CUT -- the only kind of win
  // ---- available in a kernel measured ISSUE-BOUND (the P11 verdict: a prefetch
  // ---- that buys stall movement with instructions loses here).
  // ---- docs/internals/consumer_kernel.md#producer
  static_assert(kQPO <= 32, "one producer lane per atom: the atom mask is ONE ballot");
  //: The producer's cursor state, WARP-PRIVATE and therefore IN REGISTERS. The
  //: base needed an SMEM carry (`g_scan`) only because all 8 warps had to agree
  //: on it; with a single producer the carry is a register and the cross-visit
  //: register-liveness hazard it was placed in SMEM to dodge cannot arise.
  int p_cur = w_lo;      // next word to cook into the register chunk (warp-uniform)
  int p_chunk_base = 0;  // source-word index of chunk lane 0
  unsigned p_live = 0u;  // chunk lanes whose member word is still unconsumed
  //: PER-LANE chunk registers: lane `w` cooks source word `p_chunk_base + w`.
  uint32_t p_fr = 0u, p_fw = 0u, p_mem = 0u;
  //: PER-LANE ingredient: lane `q` holds atom `q`'s bitlist word for `p_word`.
  uint32_t p_ra = 0u, p_wa = 0u;
  int p_word = 0;                   // the word whose remainder is held (warp-uniform)
  uint32_t p_rem = 0u;              // that word's unconsumed member bits
  uint32_t p_gfr = 0u, p_gfw = 0u;  // that word's fixed-level AND results
  //: The scan's fixed-level row indices, hoisted OUT of the refill (registers
  //: now, not SMEM: the producer is one warp and owns them).
  int p_row[D];
#pragma unroll
  for (int l = 0; l < D; ++l) p_row[l] = p.level_row_offset[l] + base[l];

  //: ONE COOKED GROUP, PUBLISHED TO `slot`. WARP 0 ONLY, and EVERY LANE ENTERS --
  //: the `__ballot_sync` / `__shfl_sync` below are full-warp collectives and the
  //: loop's control flow is warp-uniform by construction (`p_rem`, `p_live`,
  //: `p_cur`, `fill` are all warp-uniform; only the chunk and ingredient
  //: registers are per-lane).
  auto produce_group = [&](int slot) {
    int* R = g_ring + slot * kSchedSlotInts;
    int fill = 0;
    unsigned rmask = 0u, wmask = 0u;
    bool gr = false, gw = false;
    while (fill < kGT) {
      if (p_rem == 0u) {
        //: THE CHUNK, COOKED IN REGISTERS (contract item 2): one word per lane.
        //: Lane `w` ANDs the owner's `j` fixed-level token rows per side and the
        //: token-granular slice-union word -- the "k+1 per-digit bit-streams,
        //: identically indexed, one shared cursor structure" -- into its own
        //: `(fr, fw, member)`. The chunk's LIVE-WORD SET is then one ballot, and
        //: skipping dead words costs an `__ffs`. No SMEM, no barrier: this is
        //: where the base's two `__syncthreads()` per refill went.
        while (p_live == 0u && p_cur < w_hi) {
          const int w = p_cur + lane;
          uint32_t fr = 0xffffffffu, fw = 0xffffffffu, mem = 0u;
          if (w < w_hi) {
            const int64_t slab = (static_cast<int64_t>(bh) * p.n_twords + w) * p.total_rows;
#pragma unroll
            for (int l = 0; l < D; ++l) {
              if (l >= j) break;
              fr &= p.read_tok_bits[slab + p_row[l]];
              fw &= p.write_tok_bits[slab + p_row[l]];
            }
            const uint32_t un =
                p.slice_union_tok[(static_cast<int64_t>(bh) * p.n_slices + slice) * p.n_twords + w];
            //: The summary pass ballots the WRITE side alone (union-visit
            //: semantics on one side); `summary` is LAUNCH-UNIFORM.
            mem = (summary ? fw : (fr | fw)) & un;
          }
          p_fr = fr;
          p_fw = fw;
          p_mem = mem;
          p_chunk_base = p_cur;
          p_cur += kSchedChunkWords;
          p_live = __ballot_sync(0xffffffffu, mem != 0u);
        }
        if (p_live == 0u) break;  // the segment's stream end
        const int sel = __ffs(static_cast<int>(p_live)) - 1;
        p_live &= p_live - 1u;
        p_word = p_chunk_base + sel;
        p_rem = __shfl_sync(0xffffffffu, p_mem, sel);
        p_gfr = __shfl_sync(0xffffffffu, p_fr, sel);
        p_gfw = __shfl_sync(0xffffffffu, p_fw, sel);
        //: Form (b)'s ingredient, ONE LANE PER ATOM -- which is what turns the
        //: base's `2 * QPO` per-warp tests into the two ballots below. Loaded
        //: ONCE PER SOURCE WORD, not once per ballot round.
        p_ra = (lane < kQPO)
                   ? p.read_atom_bits[ab_base + static_cast<int64_t>(lane) * p.n_twords + p_word]
                   : 0u;
        p_wa = (lane < kQPO)
                   ? p.write_atom_bits[ab_base + static_cast<int64_t>(lane) * p.n_twords + p_word]
                   : 0u;
      }
      const int room = kGT - fill;  // >= 1: a full group ran and reset `fill`
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
      rmask |= __ballot_sync(0xffffffffu, (lane < kQPO) && ((p_ra & taken) != 0u));
      wmask |= __ballot_sync(0xffffffffu, (lane < kQPO) && ((p_wa & taken) != 0u));
      gr = gr || (p_gfr & taken) != 0u;
      gw = gw || (p_gfw & taken) != 0u;
      fill += __popc(taken);
      p_rem ^= taken;
    }
    // ---- THE PER-OWNER UPPER PRODUCT (the stream-inventory amendment). The
    // ---- FIXED levels' amplitudes are CONSTANT ALONG THE LEAF AXIS -- `j`
    // ---- scalars per member token per side, common to every atom's Kronecker
    // ---- coefficient -- so their per-token product is an OWNER-LEVEL fact and
    // ---- obeys the same multiplicity rule membership does: computed ONCE PER
    // ---- OWNER, never per atom. These are §3 step 3's ROW SCALES, and the base
    // ---- already honoured the multiplicity (they were computed once per
    // ---- (visit, token) and held OUT of the coefficient blocks); what the
    // ---- amendment changes is their HOME.
    // ----
    // ---- HOME CHOSEN: THE COOKED-SCHEDULE RING (the amendment's suggested
    // ---- home), and the reason is that it is the only one that also BUYS
    // ---- something. The cook stage already owns the member token list, so the
    // ---- scales ride the slot at zero extra streams and arrive PRE-MULTIPLIED a
    // ---- visit early -- which lifts `2 * j` fp32 gathers off the visit's
    // ---- critical path into the run-ahead, and lets the `sw`-publication
    // ---- barrier in the staging go (it published exactly this). An
    // ---- owner-scoped mini-ring was rejected: it would carry its own cursor for
    // ---- data already keyed by precisely this ring's index. Per-atom
    // ---- duplication was rejected: it multiplies the one quantity the
    // ---- amendment exists to keep at multiplicity one.
    // ---- docs/internals/consumer_kernel.md#upper-product
    //: Predicated exactly as the base's two row-scale loops were, so a
    //: side-dead visit fetches nothing it did not fetch before.
    const bool need_r = !summary && gr && rmask != 0u;
    const bool need_w = gw && wmask != 0u;
    if (need_r || need_w) {
      //: lanes below `fill` read back token indices OTHER lanes wrote above.
      __syncwarp();
      float* U = g_up + slot * 2 * kGT;
      const int tok = (lane < fill) ? R[lane] : 0;
      const int64_t tr = tok_base + tok;
      float rs = 0.0f, ws = 0.0f;
      //: A ragged group's padded slots take EXACT ZERO scales, which is what
      //: makes their coefficient rows inert through every GEMM and the mass
      //: fold's `sw[t]` term vanish.
      if (lane < fill) {
        rs = 1.0f;
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
      if (lane < kGT) {
        if (need_r) U[lane] = rs;
        if (need_w) U[kGT + lane] = ws;
      }
    }
    //: `fill == 0` is the stream-end sentinel every consumer breaks on; the
    //: final flush is the ragged group (`fill < kGT`).
    if (lane == 0) {
      R[kSchedGN] = fill;
      R[kSchedRMask] = static_cast<int>(rmask);
      R[kSchedWMask] = static_cast<int>(wmask);
      R[kSchedFlags] = (gr ? 1 : 0) | (gw ? 2 : 0);
    }
  };

  //: THE RING'S COLD START -- the producer fills `kSchedDepth` slots before the
  //: first visit, so the steady state never waits on production. The residue the
  //: contract admits (item 6) is exactly this ramp plus the final drain.
  if (warp == 0) {
#pragma unroll 1
    for (int s = 0; s < kSchedDepth; ++s) produce_group(s);
  }
  __syncthreads();

  //: THE VISIT'S GROUP LIST: `t_idx[q]` = the q-th gathered GLOBAL token index,
  //: ascending (stability IS causality). It is the cooked ring slot's first
  //: `kGT` ints -- retargeted per visit rather than copied, so every staging
  //: loop reads the producer's published descriptor directly.
  const int* t_idx = g_ring;
  //: THE VISIT'S ROW SCALES, likewise retargeted per visit rather than copied:
  //: `sr[t]` / `sw[t]` are the producer's per-token upper products for the
  //: slot's tokens, read-only to every consumer.
  const float* sr = g_up;
  const float* sw = g_up + kGT;

  // ---- THE STAGING, AS A FUNCTION OF THE GROUP: everything the visit's COMPUTE
  // ---- consumes but does not produce. Tokens are `t_idx[0..gN)`, globally
  // ---- ascending; slots `[gN, kGT)` of a ragged group stage EXACT-ZERO
  // ---- coefficient rows and zero row scales, so they are inert through every GEMM.
  // ---- STAGING IS BRANCHLESS (fetch consults nothing -- the frozen contract):
  // ---- dead atoms stage the exact zeros the tensors hold; the ATOM MASKS ARE
  // ---- COMPUTE-ONLY -- the MMA walks and the fold's RMW skip dead atoms (the
  // ---- compact-into-registers walk), exactly and only as an efficiency.
  // ---- docs/internals/consumer_kernel.md#staging
  auto stage_group = [&](bool is_read, bool is_write, int gN, unsigned rmask, unsigned wmask) {
    // ---- DECAY, step (a): the clock prefix, `P_ref`, `s0` -- INTO `Vw`'s bytes,
    // ---- guarded by `is_write` STRUCTURALLY. Runs over ALL `BC` leaves (elementwise
    // ---- surplus-ALU work; keeps every leaf's clk/p_ref/s0 FINITE for the panels --
    // ---- the 0*NaN discipline). A dead write atom's leaves take acc == 0, hence
    // ---- s0 == 1: ticks ride writes, which is what makes the fold's atom skip
    // ---- EXACT under decay. docs/internals/consumer_kernel.md#decay-clock
    if (DECAY && is_write) {
      for (int i = tid; i < BC; i += kConsumerThreads) {
        float acc = 0.0f;
        for (int t = 0; t < gN; ++t) {
          float cl = 1.0f;
#pragma unroll
          for (int l = 0; l < D; ++l) {
            cl *= p.amp_clock[l][(tok_base + t_idx[t]) * p.level_pitch_clock[l] + dig[l * BC + i]];
          }
          //: HAZARD clock-prefix-fp32 -- docs/internals/consumer_kernel.md#clock-prefix-fp32
          acc += cl;
          clk[t * CKP + i] = acc;
        }
        //: A ragged group's padded slots take a FINITE zero prefix so both panel
        //: factors stay finite (`keep^0 == 1`); their coefficient rows are exact
        //: zeros anyway.
        for (int t = gN; t < kGT; ++t) clk[t * CKP + i] = 0.0f;
        const float pr = 0.5f * acc;
        p_ref[i] = pr;
        s0[i] = expf(pr * log_keep[i]);
      }
      __syncthreads();
      // ---- DECAY, step (c): the entry state enters the reference frame. `s0`
      // ---- multiplies the STATE; the mass column rides the same sweep (same
      // ---- recurrence with `v := 1`). Row-per-warp because `cols = DV + 1` is not
      // ---- a power of two. Dead atoms' rows multiply by exactly 1.
      for (int i = warp; i < BC; i += kWarps) {
        const float s = s0[i];
        for (int c = lane; c < cols; c += 32) S[i * SP + c] *= s;
      }
    }
    // ---- §3 STEP 3's ROW SCALES ARRIVE PRE-MULTIPLIED IN THE RING SLOT, cooked
    // ---- once per owner by the producer a visit ago (the stream-inventory
    // ---- amendment). Nothing is computed and nothing is published here, so the
    // ---- `sw`-publication `__syncthreads()` that used to stand at this point --
    // ---- its ONLY job was making the row scales visible to the `Vw` staging --
    // ---- is gone with them. docs/internals/consumer_kernel.md#upper-product
    // ---- THE READ PANEL (§3 steps 2-3). The loop walks FRAGMENT
    // ---- SLOTS (one k-step = one atom = 128 words), so the store is contiguous in
    // ---- `tid` and the buffer is atom-blocked by construction. The bf16 slot packs
    // ---- the leaf pair `(k, k+1)`.
    // ---- docs/internals/consumer_kernel.md#read-panel
    if (is_read) {
      //: One coefficient-pair accessor. THE fp16 APPLICATION IS CARVED OUT
      //: (U07-d built, adjudicated 2026-08-06): an ADMITTED hot routing (panel
      //: bound 23.3 of 40) put the survival factor at ~e^15+, past fp16's
      //: 65504 max -- measured NaN, not a rounding. The panels apply in fp32
      //: against the bf16 STORAGE (the memory win survives the carve); the
      //: register cost is the ratchet's to adjudicate.
      //: Evidence: workflows/perf/p26_evidence/m2_fp16_breach.txt.
      auto rf_pair = [&](int t, int k) -> unsigned {
        if (t >= gN) return 0u;  // padded slot: exact-zero pair
        const int64_t tr = tok_base + t_idx[t];
        float a0 = 1.0f, a1 = 1.0f;
#pragma unroll
        for (int l = 0; l < D; ++l) {
          if (l < j) continue;  // fixed levels hoist into the row scale (§3 step 3)
          const __nv_bfloat16* row =
              static_cast<const __nv_bfloat16*>(p.amp_read[l]) + tr * p.level_pitch_read[l];
          a0 *= __bfloat162float(row[dig[l * BC + k]]);
          a1 *= __bfloat162float(row[dig[l * BC + k + 1]]);
        }
        if (DECAY && is_write) {
          a0 *= panel_read_factor(clk[t * CKP + k], p_ref[k], log_keep[k]);
          a1 *= panel_read_factor(clk[t * CKP + k + 1], p_ref[k + 1], log_keep[k + 1]);
        }
        return pack_bf16x2(clamp_bf16_finite(a0), clamp_bf16_finite(a1));
      };
      //: BRANCHLESS (fetch consults nothing -- the frozen contract): every
      //: atom's slots are staged by the pure affine walk; a dead atom's
      //: positions hold the exact zeros the amplitude tensor already holds
      //: (entmax), so the fetch layer cannot be wrong -- the masks are
      //: COMPUTE-ONLY (MMA skip).
      for (int sl = tid; sl < kRfWords; sl += kConsumerThreads) {
        int t, k;
        frag_a_slot(sl, t, k);
        Rf[sl] = rf_pair(t, k);
      }
    }
    // ---- THE WRITE PANEL -- one leaf-PAIR per thread visit.
    // ---- The pass-1 inclusion ballot rides the
    // ---- same 16-lane token run per leaf. docs/internals/consumer_kernel.md#write-panel
    if (is_write) {
      auto wt_pairs = [&](int q) {
        // 8 leaf pairs x kGT tokens = 128 slots for atom q.
        for (int idx = tid; idx < 8 * kGT; idx += kConsumerThreads) {
          const int pr_i = idx / kGT;      // leaf pair within the atom
          const int t = idx - pr_i * kGT;  // token slot
          const int i0 = q * kMmaKQuantum + 2 * pr_i;
          float a0 = 0.0f, a1 = 0.0f;
          if (t < gN) {
            //: fp32 application against bf16 storage (the fp16 carve; see the
            //: read panel's note).
            const int64_t tr = tok_base + t_idx[t];
            a0 = 1.0f;
            a1 = 1.0f;
#pragma unroll
            for (int l = 0; l < D; ++l) {
              if (l < j) continue;
              const __nv_bfloat16* row =
                  static_cast<const __nv_bfloat16*>(p.amp_write[l]) + tr * p.level_pitch_write[l];
              a0 *= __bfloat162float(row[dig[l * BC + i0]]);
              a1 *= __bfloat162float(row[dig[l * BC + i0 + 1]]);
            }
            if (DECAY) {
              a0 *= panel_write_factor(clk[t * CKP + i0], p_ref[i0], log_keep[i0]);
              a1 *= panel_write_factor(clk[t * CKP + i0 + 1], p_ref[i0 + 1], log_keep[i0 + 1]);
            }
          }
          //: §6.4 PASS 1's per-leaf inclusion, COEFFICIENT HALF: a leaf is included
          //: iff some token's staged coefficient is nonzero. The kGT tokens of one
          //: leaf pair are one contiguous 16-lane run.
          //: HAZARD summary-prequant -- docs/internals/consumer_kernel.md#summary-prequant
          if (summary) {
            const unsigned run = 0xffffu << (lane & ~(kGT - 1));
            if (__any_sync(run, a0 != 0.0f) && (lane & (kGT - 1)) == 0) sum_slot[i0] = 1;
            if (__any_sync(run, a1 != 0.0f) && (lane & (kGT - 1)) == 0) sum_slot[i0 + 1] = 1;
          }
          stage_put(WT[wt_index(i0, t)], a0);
          stage_put(WT[wt_index(i0 + 1, t)], a1);
        }
      };
      //: BRANCHLESS, as the read panel: every atom staged, zeros ride
      //: self-masking, masks stay compute-only.
#pragma unroll
      for (int q = 0; q < kQPO; ++q) wt_pairs(q);
    }
    //: HAZARD clk-dead -- docs/internals/consumer_kernel.md#clk-dead
    if (DECAY && is_write) __syncthreads();  // <- clk is now DEAD; Vw may overwrite
    // ---- `Vw[t, ch] = scale_write[t] * v[token, ch]`. The write side folds the row
    // ---- scale into the VALUE tile because there `t` is the CONTRACTION index and
    // ---- the scale cannot hoist out of the sum; one staged object, two GEMMs.
    if (is_write) {
      for (int idx = tid; idx < kGT * DV; idx += kConsumerThreads) {
        const int t = idx / DV;
        const int ch = idx - t * DV;
        float vv = 0.0f;
        if (t < gN) {
          vv = __bfloat162float(static_cast<const __nv_bfloat16*>(
              p.v)[(static_cast<int64_t>(bh) * p.T + t_idx[t]) * DV + ch]);
        }
        stage_put(Vw[vw_index<DV>(t, ch)], sw[t] * vv);
      }
    }
  };

  // ---- THE WALK: the PRODUCER'S COOKED RING feeds ONE visit body. Words ascend
  // ---- and each word is consumed LSB-first, so `t_idx` is GLOBALLY ascending:
  // ---- the intra Gram's causal mask is order-relative, so STABILITY IS THE
  // ---- CORRECTNESS REQUIREMENT. Consumers ASSEMBLE NOTHING -- they read the
  // ---- descriptor warp 0 published `kSchedDepth` visits ago, whose publication
  // ---- barrier is the end-of-visit `__syncthreads()` that already existed. The
  // ---- base's per-warp-redundant assembler (8x the ballot ALU and 8x the
  // ---- atom-bitlist load issue) is what this deletes.
  // ---- docs/internals/consumer_kernel.md#group-assembler
  for (int gidx = 0;; ++gidx) {
    const int* R = g_ring + (gidx % kSchedSlots) * kSchedSlotInts;
    const int gN = R[kSchedGN];
    if (gN == 0) break;  // the stream's end, as the producer saw it
    //: The visit's per-side ATOM MASKS and side flags, cooked by the producer's
    //: two ballots. CTA-uniform by construction: one writer, one published word.
    const unsigned rmask = static_cast<unsigned>(R[kSchedRMask]);
    const unsigned wmask = static_cast<unsigned>(R[kSchedWMask]);
    const int gflags = R[kSchedFlags];
    t_idx = R;
    sr = g_up + (gidx % kSchedSlots) * 2 * kGT;
    sw = sr + kGT;

    // ---- THE VISIT'S CONTROL BLOCK, RESET. Behind it stands the previous visit's
    // ---- end-of-visit `__syncthreads()` (so no warp is still retiring into it);
    // ---- ahead of it stands the post-staging `__syncthreads()` below, which is
    // ---- what orders the reset before the visit's first `retire_item`. The base's
    // ---- separate group-ENTRY barrier existed to publish `t_idx`, and the ring
    // ---- publishes it a visit earlier -- so that barrier is gone.
    // ---- docs/internals/pipeline.md#edge-dag
    if (tid < ctl_words(BC)) ctl[tid] = 0;
    //: The visit's side flags. A side whose atom mask is empty is a side with
    //: no live coefficients anywhere in the slice this visit -- skipping it is
    //: exact (the whole-visit emptiness early-out the tile-level machinery
    //: structurally could not see).
    const bool is_read = !summary && ((gflags & 1) != 0) && rmask != 0u;
    const bool is_write = ((gflags & 2) != 0) && wmask != 0u;
    const unsigned gmask = rmask & wmask;

    stage_group(is_read, is_write, gN, rmask, wmask);

    __syncthreads();

    // ---- GEMM 1: THE READOUT, `y_group = R . S`, LIVE READ ATOMS ONLY. The B
    // ---- operand is packed out of `S` on the fly; STATE AND ACCUMULATION STAY
    // ---- fp32. `mask == FULL` is the statically-unrolled dense arm.
    // ---- docs/internals/consumer_kernel.md#gemm-readout
    float y0 = 0.0f, y1 = 0.0f, y2 = 0.0f, y3 = 0.0f;
    if (is_read && warp < NT_DV) {
      if (rmask == kFullMask) {
        mma_tile_fragA<BC, float, false>(Rf, S, SP, BC, warp * 8, lane, y0, y1, y2, y3);
      } else {
        for (unsigned m = rmask; m; m &= m - 1) {
          const int q = __ffs(m) - 1;
          mma_tile_fragA<kMmaKQuantum, float, false>(Rf + q * 128, S + q * kMmaKQuantum * SP, SP,
                                                     kMmaKQuantum, warp * 8, lane, y0, y1, y2, y3);
        }
      }
    }

    const bool mixed = is_read && is_write;

    // ---- THE MASS COLUMN, TERM 1 -- placement FORCED by phase B: the fold WRITES
    // ---- `S[i][DV]`, so term 1, which READS it, must already have run. Live read
    // ---- atoms only. docs/internals/consumer_kernel.md#mass-term1
    //: HAZARD mass-split -- docs/internals/consumer_kernel.md#mass-split
    if (is_read) {
      for (int t = tid; t < kGT; t += kConsumerThreads) {
        float acc = 0.0f;
        if (rmask == kFullMask) {
          for (int k = 0; k < BC; ++k) acc += frag_a_get(Rf, t, k) * S[k * SP + DV];
        } else {
          for (unsigned m = rmask; m; m &= m - 1) {
            const int q = __ffs(m) - 1;
            for (int kk = 0; kk < kMmaKQuantum; ++kk) {
              const int k = q * kMmaKQuantum + kk;
              acc += frag_a_get(Rf, t, k) * S[k * SP + DV];
            }
          }
        }
        den[t] = acc;
      }
    }

    // ---- B6: THE VISIT'S READ-THEN-WRITE ORDERING ON `S`, AS A COUNTER rather than
    // ---- a rendezvous, so only the FOLD waits and it waits behind real work. The
    // ---- retirement is UNCONDITIONAL: docs/internals/consumer_kernel.md#b6
    retire_item(&ctl[kCtlReadDone], lane);

    // ---- GEMM 2: the causal effective-score block, `G += R_q . W_q^T` over atoms
    // ---- live on BOTH sides (either side dead => the term is exactly zero, and the
    // ---- dead side's panel was never staged), then §5's INCLUSIVE causal mask
    // ---- applied on the completed Gram. docs/internals/consumer_kernel.md#gemm-gram
    float gd0 = 0.0f, gd1 = 0.0f, gd2 = 0.0f, gd3 = 0.0f;
    if (mixed && warp < kGT / 8) {
      if (gmask == kFullMask) {
#pragma unroll
        for (int q = 0; q < kQPO; ++q) {
          mma_tile_fragA_wtB(Rf + q * 128, WT + q * kMmaKQuantum * BTP, warp * 8, lane, gd0, gd1,
                             gd2, gd3);
        }
      } else {
        for (unsigned m = gmask; m; m &= m - 1) {
          const int q = __ffs(m) - 1;
          mma_tile_fragA_wtB(Rf + q * 128, WT + q * kMmaKQuantum * BTP, warp * 8, lane, gd0, gd1,
                             gd2, gd3);
        }
      }
      CONSUMER_D_ROWS(lane);
      const int n0 = warp * 8;
      const int c0 = n0 + 2 * _r, c1 = n0 + 2 * _r + 1;
      gm_put(Gm[_g * GMP + c0], (c0 <= _g) ? gd0 : 0.0f);
      gm_put(Gm[_g * GMP + c1], (c1 <= _g) ? gd1 : 0.0f);
      gm_put(Gm[(_g + 8) * GMP + c0], (c0 <= _g + 8) ? gd2 : 0.0f);
      gm_put(Gm[(_g + 8) * GMP + c1], (c1 <= _g + 8) ? gd3 : 0.0f);
    }
    // ---- B5, AS A COUNTER: `Gm` is published to phase C by the WRITERS ONLY,
    // ---- target `kGramWarps`, `!mixed` skipping it entirely.
    // ---- docs/internals/consumer_kernel.md#gram-gate
    //: HAZARD gram-gate-guard -- docs/internals/consumer_kernel.md#gram-gate-guard
    if (mixed && warp < kGramWarps) retire_item(&ctl[kCtlGramDone], lane);

    // ---- GEMM 4, THE BOUNDARY FOLD `S += W^T . Vw`, THE VISIT'S FLUID ITEM POOL,
    // ---- ATOM-MAJOR (Blake's confirmed loop-nest, 2026-08-06): an item claims one
    // ---- (live write atom, column tile); the atom's accumulator loads to
    // ---- registers, stays resident while the atom CONSUMES EVERY STAGED GROUP
    // ---- SLOT (`kFoldStagedGroups` -- 1 today; the per-atom fetch era deepens
    // ---- the staging and ONLY this constant and the staging producer move),
    // ---- then ONE writeback, release, next item. The readout stays group-major:
    // ---- its state reads are read-only. LIVE WRITE ATOMS ONLY -- skipping a
    // ---- dead atom's RMW is exact: its delta is zero and its exit scale is
    // ---- s0 == 1 (ticks ride writes).
    // ---- docs/internals/consumer_kernel.md#gemm-fold
    if (is_write) {
      await_count(&ctl[kCtlReadDone], kWarps);
      const int n_live = __popc(wmask);
      const int n_fold_items = n_live * NT_DV;
      for (;;) {
        //: HAZARD fold-claim-uniform -- docs/internals/consumer_kernel.md#fold-claim-uniform
        int c = 0;
        if (lane == 0) c = atomicAdd(&ctl[kCtlFold0], 1);
        c = __shfl_sync(0xffffffffu, c, 0);
        if (c >= n_fold_items) break;
        CONSUMER_D_ROWS(lane);
        const int mr = c / NT_DV;  // index into the LIVE atom list
        const int nt = c - mr * NT_DV;
        const int mt = (wmask == kFullMask) ? mr : static_cast<int>(__fns(wmask, 0, mr + 1));
        const int r0 = mt * 16 + _g, n0 = nt * 8 + 2 * _r;
        const int i0 = r0, i1 = r0 + 8;
        // ---- the atom's accumulator: RESIDENT IN REGISTERS from the load to
        // ---- the one writeback below.
        float a0 = S[i0 * SP + n0];
        float a1 = S[i0 * SP + n0 + 1];
        float a2 = S[i1 * SP + n0];
        float a3 = S[i1 * SP + n0 + 1];
#pragma unroll
        for (int gs = 0; gs < kFoldStagedGroups; ++gs) {
          float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
          mma_fold_tile<DV>(WT + gs * kWtSlotElems + mt * kMmaKQuantum * BTP,
                            Vw + gs * kVwSlotElems, nt * 8, lane, d0, d1, d2, d3);
          // ---- DECAY's EXIT SCALE RIDES THE ACCUMULATOR, per staged group:
          // ---- `acc <- s0 * (acc + d)`, one multiply on a register value.
          // ---- (At depth > 1 the per-group s0/window algebra is the per-atom
          // ---- fetch era's design item; depth is PINNED 1 until then.)
          // ---- docs/internals/consumer_kernel.md#fold-epilogue
          const float e0 = DECAY ? s0[i0] : 1.0f;
          const float e1 = DECAY ? s0[i1] : 1.0f;
          a0 = e0 * (a0 + d0);
          a1 = e0 * (a1 + d1);
          a2 = e1 * (a2 + d2);
          a3 = e1 * (a3 + d3);
        }
        // ---- the ONE writeback: the atom releases its accumulator.
        S[i0 * SP + n0] = a0;
        S[i0 * SP + n0 + 1] = a1;
        S[i1 * SP + n0] = a2;
        S[i1 * SP + n0 + 1] = a3;
      }
      // ---- the mass column's fold term, the same atom-major nest (a dead
      // ---- atom's rows: acc == 0 and s0 == 1 -- the skip is the identity).
      {
        const int n_live_rows = __popc(wmask) * kMmaKQuantum;
        for (int idx = tid; idx < n_live_rows; idx += kConsumerThreads) {
          const int q =
              (wmask == kFullMask) ? (idx >> 4) : static_cast<int>(__fns(wmask, 0, (idx >> 4) + 1));
          const int row = q * kMmaKQuantum + (idx & 15);
          float m = S[row * SP + DV];
#pragma unroll
          for (int gs = 0; gs < kFoldStagedGroups; ++gs) {
            float acc = 0.0f;
            for (int t = 0; t < kGT; ++t)
              acc += mma_src(WT[gs * kWtSlotElems + wt_index(row, t)]) * sw[t];
            m = (DECAY ? s0[row] : 1.0f) * (m + acc);
          }
          S[row * SP + DV] = m;
        }
      }
    }

    // ---- THE PHASE-C GATE, SKIPPED ON `!mixed` AND NARROWED TO THE WRITERS, and it
    // ---- is FREE -- `kCtlGramDone` is retired AHEAD of the fold. The wait is
    // ---- nonetheless real: docs/internals/consumer_kernel.md#gram-gate
    if (mixed) await_count(&ctl[kCtlGramDone], kGramWarps);

    // ---- PHASE C -- GEMM 3, the intra term `y_group += G . Vw`: the SAME warp
    // ---- tiling the readout used, so the two GEMMs accumulate into the same
    // ---- fragment. Token-axis contraction: no atom mask applies.
    // ---- docs/internals/consumer_kernel.md#phase-c
    if (mixed && warp < NT_DV) {
      mma_tile_vwB<__nv_bfloat16, DV>(Gm, GMP, Vw, 0, warp * 8, lane, y0, y1, y2, y3);
    }

    //: HAZARD mass-split -- docs/internals/consumer_kernel.md#mass-split
    if (mixed) {
      for (int t = tid; t < kGT; t += kConsumerThreads) {
        float acc = den[t];
        for (int u = 0; u <= t; ++u) acc += mma_src(Gm[t * GMP + u]) * sw[u];
        den[t] = acc;
      }
    }

    // ---- SPLIT-K EMISSION of this visit's partial, straight out of the D fragment,
    // ---- scattered through `t_idx` (the P14 microbench priced exactly this
    // ---- pattern: banded, DRAM-byte ratio 1.0009x at the measured gap). The
    // ---- `out == 0` early-out runs FIRST so a ragged group's padded slots
    // ---- (`sr == 0`, coefficient rows exactly zero) never read `t_idx` at all;
    // ---- members are `< T` by construction (the support words' padded tail
    // ---- carries no bit). The `!= 0` guard is a PERFORMANCE SKIP ON A VALUE,
    // ---- never a membership decision.
    // ---- docs/internals/consumer_kernel.md#split-k-emission
    if (is_read) {
      if (warp < NT_DV) {
        CONSUMER_D_ROWS(lane);
        const int n0 = warp * 8;
        const int c0 = n0 + 2 * _r, c1 = c0 + 1;
        auto emit = [&](int t, int ch, float val) {
          const float out = sr[t] * val;
          if (out == 0.0f) return;
          atomicAdd(&p.num[(static_cast<int64_t>(bh) * p.T + t_idx[t]) * DV + ch], out);
        };
        emit(_g, c0, y0);
        emit(_g, c1, y1);
        emit(_g + 8, c0, y2);
        emit(_g + 8, c1, y3);
      }
      for (int t = tid; t < gN; t += kConsumerThreads) {
        const float d = sr[t] * den[t];
        if (d != 0.0f) atomicAdd(&p.den[static_cast<int64_t>(bh) * p.T + t_idx[t]], d);
      }
    }
    // ---- §6.4 PASS 1's PER-LEAF EPILOGUE. The COEFFICIENT half of the inclusion
    // ---- test lives in the panel build; what is left here has no pre-quantization
    // ---- site: docs/internals/consumer_kernel.md#pass1-epilogue
    if (is_write && summary && DECAY) {
      for (int i = tid; i < BC; i += kConsumerThreads) {
        if (s0[i] != 1.0f) sum_slot[i] = 1;
        a_state[i] *= s0[i] * s0[i];
      }
    }
    //: THE RUN-AHEAD (contract items 2 and 6). The producer cooks group
    //: `gidx + kSchedDepth` into the slot group `gidx - 1` vacated -- with
    //: `kSchedSlots == kSchedDepth + 1` that slot is never the one being read,
    //: which is the whole reason the ring is at least 2 deep. Production runs
    //: CONCURRENTLY with this visit's compute and is published by the
    //: end-of-visit barrier below, so NOTHING WAITS ON IT: the two
    //: `__syncthreads()` the base paid per schedule refill are replaced by a
    //: slot that is already full when it is read. The steady state's only
    //: remaining schedule cost is the producer warp's own issue.
    if (warp == 0) produce_group((gidx + kSchedDepth) % kSchedSlots);
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
    for (int idx = tid; idx < BC * cols; idx += kConsumerThreads) {
      const int i = idx / cols;
      const int c = idx - i * cols;
      const int q = sum_slot[i];
      if (q >= 0) p.ws_state[(slot * BC + q) * cols + c] = S[i * SP + c];
    }
    return;
  }
  // ---- THE OWNER'S STATE BLOCK, WRITTEN TO HBM ONCE. An unpaged atom is never
  // ---- stored back, PER ATOM. `last_seg` is §6.4's publication rule and is a
  // ---- DELIBERATE NUMERIC CHOICE, not the scan's job:
  // ---- docs/internals/consumer_kernel.md#state-store
  if (!last_seg) return;
#pragma unroll
  for (int q = 0; q < kQPO; ++q) {
    const int64_t row0 = atom_row_base<BC>(p, bh, owner, q);
    if (row0 < 0) continue;
    for (int idx = tid; idx < kMmaKQuantum * cols; idx += kConsumerThreads) {
      const int r = idx / cols;
      const int c = idx - r * cols;
      p.state[(row0 + r) * cols + c] = S[(q * kMmaKQuantum + r) * SP + c];
    }
  }
}

#undef CONSUMER_D_ROWS

}  // namespace consumer
}  // namespace rola
