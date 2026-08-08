#pragma once

// THE CONSUMER'S SHARED-MEMORY LAYOUT LAYER -- what one CTA's shared memory looks
// like, and what it costs. Three kinds of thing live here and they are one concern:
//
//   1. THE ADDRESS FUNCTIONS -- the two derived pitches every staged panel is stored
//      at, the `rp_index` / `wp_index` / `vp_index` maps every access point goes
//      through so no two sites can disagree, and the STATE'S OWNERSHIP RULE (the
//      state itself is registers, not shared memory);
//   2. THE PROOFS -- bank-conflict freedom and `ldmatrix` alignment as `constexpr`
//      predicates, enumerated rather than asserted, so a future `BC`/`DV` fails
//      the BUILD instead of silently costing the 4.06x wavefront amplification the
//      unpadded build measured;
//   3. THE PRICE -- `consumer_smem_bytes` (the carve, as a byte count) and the
//      register budget and residency arithmetic the launch bound is derived from.
//
// THE GROUP IS THE TILE: the executor's work unit is a GROUP of `kMmaKQuantum`
// gathered tokens (the endgame form -- the BT axis is dead; groups are always full,
// `ceil(gathered/16)` of them). Every former `BT` term below is `kGroupTokens`.
//
// THE ONE PRECISION STORY: bf16 storage, fp32 accumulation, fp64 reference as
// oracle. The staged operands are bf16 UNCONDITIONALLY (there is no precision
// axis), and the SECTOR INVARIANT below is what the fetch/union/atom machinery may
// assume everywhere.
//
// The layout functions are the SPECIFICATION the kernel body's carve follows: the body
// computes its base pointers in the same order, and `consumer_smem_bytes` is what the
// launch reserves.
//
// The ledger cell by cell, the swizzle's three proofs, the pitch table and the
// measured register budget: docs/internals/smem_layout.md

#include <cuda_bf16.h>

#include "arch_table.cuh"
#include "readout_mma.cuh"

namespace rola {
namespace consumer {

constexpr int kConsumerThreads = 256;  // 8 warps: the CTA width an SMEM state frees

//: THE GROUP -- the PRODUCER's work unit: `kMmaKQuantum` gathered tokens, one MMA
//: m-quantum. It is deliberately THE SAME constant as the k-quantum: one multiply's
//: worth of tokens against one multiply's worth of state rows. The cook, the ballot
//: and the ring descriptor are all sized by it and none of them varies with the
//: window below.
constexpr int kGroupTokens = kMmaKQuantum;

//: **THE WINDOW, `GT` -- THE CONSUMER'S TOKEN QUANTUM, AND A TEMPLATE AXIS.** One
//: VISIT stages and consumes `W = GT / kGroupTokens` groups against ONE entry state
//: and folds them back in ONE writeback. `W > 1` is not a scheduling depth: pacing
//: the fold `W` groups behind means those groups all read the same entry state, so
//: their mutual deposits must arrive as Gram cross-terms and the Gram has to span
//: the window -- the enlarged-chunk formulation, whose cost is a block LOWER
//: TRIANGULAR `[GT x GT]` Gram (`W(W+1)/2` sub-blocks) and whose payment is that
//: every per-visit term -- the three barriers, the fold's read-modify-write of `S`,
//: the descriptor decode -- amortises `W`-fold.
//: `GT` is DISTRIBUTION-DEPENDENT and therefore a PLAN-SELECTED AXIS, never a
//: raised constant: a wide window is free at dense gathering and wastes tile at low
//: gathered-token counts. docs/internals/smem_layout.md#window
constexpr int window_groups(int GT) { return GT / kGroupTokens; }

//: THE SCHEDULE CHUNK: how many 32-token words the PRODUCER WARP cooks per
//: refill. It is exactly one word per lane, so a refill is one cooperative load
//: and the chunk's live-word set is one `__ballot_sync`. The cook is warp-local
//: throughout -- its collectives are warp-level -- so a refill costs no CTA
//: barrier. docs/internals/smem_layout.md#sched-chunk
constexpr int kSchedChunkWords = 32;
static_assert(kSchedChunkWords == 32, "the cooked chunk is one word per lane of the producer warp");

//: **THE RUN-AHEAD DEPTH.** How many WINDOWS' worth of groups the producer warp runs
//: ahead of the consumers. It is a `constexpr` (overridable at compile time for the
//: dial sweep), NEVER a template axis: the instantiation matrix does not gain a
//: coordinate. The ring holds `depth + 1` windows, so the producer never writes the
//: slots being read.
//: **TWO IS THE FLOOR, AND THE PREFETCH IS WHY.** The amplitude slab for window
//: `w + 1` is issued from inside window `w`'s visit -- that is the entire overlap --
//: so window `w + 1`'s TOKEN LIST must already be published when window `w` is
//: staging. At depth 1 it is not: the producer cooks `w + 1` during visit `w` and
//: publishes it at that visit's end. Depth 2 costs `W` descriptor slots (208 B at the
//: group quantum) and is the null configuration from here.
//: docs/internals/smem_layout.md#sched-ring
#ifndef ROLA_SCHED_DEPTH
#define ROLA_SCHED_DEPTH 2
#endif
constexpr int kSchedDepth = ROLA_SCHED_DEPTH;
static_assert(kSchedDepth >= 1, "the producer must run at least one group ahead");

//: The cooked ring holds a WHOLE WINDOW plus the run-ahead: the visit reads `W`
//: descriptors at once, so `W` slots must be simultaneously live and the producer
//: must still be cooking into slots nobody is reading.
constexpr int sched_slots(int GT) { return window_groups(GT) * (kSchedDepth + 1); }

//: **THE VALUES-SHARD DIAL (P25 contract item 4) IS STRUCTURALLY INERT AT EVERY
//: `GT`, AND IS THEREFORE NOT BUILT -- a derivation, not an omission.** The dial's
//: stated selection criterion is the VALUES-RING HEAD-OF-LINE residual: with one
//: shared value tile the slowest atom consumer paces the tile's recycling, and `k`
//: shards buy pacing slack by giving disjoint atom sets their own copies. That
//: residual is a function of how long an atom holds a tile RELATIVE TO ITS
//: NEIGHBOURS. The window widened the fold's accumulator residency from one staged
//: group to `W` -- but all `W` slots are staged in the SAME PHASE of the SAME
//: VISIT and released by the SAME end-of-window `__syncthreads()`, so every atom
//: holds every tile for exactly the window and THE CURSOR SPREAD IS STILL ZERO.
//: `k > 1` would be `k` byte-identical copies of tiles that no code path
//: distinguishes -- dead storage behind a dead branch. The dial becomes live only
//: if the window's staging stops being one barrier-delimited phase; until then `k`
//: is a PLAN-TIME quantity, pinned at 1 by the host.
//: docs/internals/smem_layout.md#values-shard

//: ONE COOKED GROUP DESCRIPTOR, in ints: the `kGroupTokens` gathered GLOBAL token
//: indices (ascending -- stability IS causality), then `gN` (the group's member
//: count; `0` is the stream-end sentinel), the read atom mask, the write atom
//: mask, and the side flags (bit 0 = the group has a read-live token, bit 1 =
//: write-live). Consumption multiplicity is `n_atoms + 1` -- which is what earns
//: it storage under the materialization rule, and is why the INGREDIENTS (which
//: only the producer reads) do not get a cooked buffer of their own.
constexpr int kSchedSlotInts = kGroupTokens + 4;
constexpr int kSchedGN = kGroupTokens;
constexpr int kSchedRMask = kGroupTokens + 1;
constexpr int kSchedWMask = kGroupTokens + 2;
constexpr int kSchedFlags = kGroupTokens + 3;

//: THE SECTOR INVARIANT, UNIVERSAL: one atom's worth of bf16 digits IS one memory
//: sector. `kMmaKQuantum` bf16 elements = 32 B = the fetch floor = the MMA k-quantum
//: = the union-bitmap granule. The fetch/union/atom code assumes this
//: unconditionally; there is no fp32-sector side path (fp32 STORAGE is dead).
static_assert(kMmaKQuantum * static_cast<int>(sizeof(__nv_bfloat16)) == kSectorBytes,
              "one atom of bf16 digits must be exactly one 32 B sector -- the "
              "fetch floor, the MMA k-quantum and the union granule coincide BY "
              "CONSTRUCTION, and the walk assumes it");

//: **THE OCCUPANCY TARGET, AS ONE MEASURED NUMBER, AND THE MEASUREMENT REFUSED THE
//: TLP->ILP SWITCH.** With the state register-resident the kernel's shared memory holds
//: only the staged panels -- 16.5 KB at the flagship where the SMEM state cost 31.6 --
//: so residency stopped being SMEM-bound and this budget became the ONLY thing setting
//: it. The expectation was FA's shape: few warps, many registers, deep unrolled
//: independent accumulators. The deep unroll was BUILT (one A operand feeds every state
//: tile of the fold; both token tiles of the readout share one narrowed state
//: fragment), and then the occupancy sweep went the other way. Dense `N = L = 4096`
//: (`p31_evidence/whole_layer.py`, block-min-median), one build per point, all with the
//: deep unroll in:
//:
//:     budget  CTAs/SM  warps/SM   wall
//:        128     2        16      15.05 ms
//:         85     3        24      12.63 ms
//:         64     4        32      12.42 ms
//:         51     5        40      13.80 ms
//:
//: `64` is the optimum and it is a 3.8% WIN over the SMEM-state baseline's 12.91 ms.
//: FEWER WARPS IS MONOTONICALLY WORSE HERE, by 21% at FA's own occupancy. The reading:
//: this kernel's explicit latency hiding (rings, prefetch, cook-ahead) covers the SMEM
//: pipeline, and the tensor pipe is not the constraint -- it runs at ~4% of peak. What
//: is left uncovered is DRAM: the value stream's gather and the emission's `atomicAdd`
//: traffic, whose latency only CTA-level parallelism hides. Registers bought back below
//: `64` start costing spill faster than the fifth CTA pays.
//: docs/internals/smem_layout.md#reg-budget
constexpr int consumer_reg_budget(int BC, bool decay, int DV, int D, int GT) {
  (void)BC;
  (void)decay;
  (void)DV;
  (void)D;
  (void)GT;
  return 64;
}

//: CTAs/SM to STATE IN THE LAUNCH BOUND -- the smallest of what the three hardware
//: resources admit, computed per instantiation rather than fixed. All three terms are
//: load-bearing: a missing thread term makes `ptxas` DISCARD an out-of-range bound
//: (not clamp it), and a missing register term buys the extra CTAs by spilling.
//: docs/internals/smem_layout.md#residency
constexpr int ctas_for_residency(size_t smem, int reg_budget) {
  const int by_smem = static_cast<int>(kSm.smem_per_sm / (smem + kDriverSmemReservePerCta));
  const int by_threads = kSm.max_threads_per_sm / kConsumerThreads;
  const int by_regs = kSm.regs_per_sm / (kConsumerThreads * reg_budget);
  const int m = by_smem < by_threads ? by_smem : by_threads;
  return m < by_regs ? m : by_regs;
}

//: **THE BANK-CONFLICT PROPERTY, AS A COMPILE-TIME PROOF RATHER THAN A COMMENT.**
//: Every operand this kernel stages is a PITCHED ROW-MAJOR bf16 block read by one of
//: the two `ldmatrix` atoms, so ONE predicate licenses every fetch: the instruction is
//: serviced 8 lanes at a time, each lane reading a 16-byte row, and the phase is
//: conflict-free exactly when its 8 rows cover all 32 banks. It additionally checks the
//: 16-byte ALIGNMENT `ldmatrix` requires, which no stride predicate could see. `ROWS`
//: and `COLS` are the extents actually addressed; `PITCH` is the row stride.
//: docs/internals/smem_layout.md#bank-predicates
template <int ROWS, int COLS, int PITCH>
constexpr bool pitched_ldsm_banks_ok() {
  for (int r0 = 0; r0 + 8 <= ROWS; r0 += 8) {
    for (int c0 = 0; c0 + 8 <= COLS; c0 += 8) {
      unsigned banks = 0;
      for (int l = 0; l < 8; ++l) {
        const size_t byte = static_cast<size_t>((r0 + l) * PITCH + c0) * 2;
        if (byte % 16 != 0) return false;
        for (int w = 0; w < 4; ++w) {
          const unsigned b = static_cast<unsigned>((byte / 4 + w) % 32);
          if (banks & (1u << b)) return false;
          banks |= 1u << b;
        }
      }
      if (banks != 0xffffffffu) return false;
    }
  }
  return true;
}

//: THE SAME QUESTION FOR THE STAGING STORE. Every panel is filled by a flat
//: `idx = row * COLS + col` loop at `kConsumerThreads` stride, so the 32 lanes of one
//: window must land on 32 banks -- with the 16-bit rule that two lanes in the SAME
//: 32-bit word do not conflict.
template <int ROWS, int COLS, int PITCH>
constexpr bool pitched_store_banks_ok() {
  for (int base = 0; base + 32 <= ROWS * COLS; base += 32) {
    int word[32] = {};
    int bank[32] = {};
    for (int l = 0; l < 32; ++l) {
      const int idx = base + l;
      const int e = (idx / COLS) * PITCH + idx % COLS;
      word[l] = e >> 1;
      bank[l] = (e >> 1) % 32;
    }
    for (int i = 0; i < 32; ++i)
      for (int k = i + 1; k < 32; ++k)
        if (bank[i] == bank[k] && word[i] != word[k]) return false;
  }
  return true;
}

//: **THE TOKEN-AXIS PITCH**, and the `+8` IS THE PROOF'S ANSWER: a `[* ][kGroupTokens]`
//: block serves an `ldmatrix.trans` whose `k` axis is the block's ROW axis, and
//: enumerated over `pad in {4, 8, 12, 16}`, `kGroupTokens + 8` is the unique pad that
//: passes. The READ PANEL (`[BC leaves][16 tokens]`), the VALUE TILE
//: (`[DV channels][16 tokens]`) and the WINDOW GRAM all take it.
//: docs/internals/smem_layout.md#pitches
constexpr int kTokPitch = kGroupTokens + 8;

//: **THE LEAF-AXIS PITCH**, the same `+8` answer for the WRITE PANEL, which is stored
//: `[16 tokens][BC leaves]` because it is read BOTH ways: as GEMM 2's A operand
//: (`m = token, k = leaf`, the non-transposing atom) and as the fold's B operand
//: (`k = token, n = leaf`, the transposing one). One buffer, two atoms, one pitch.
__host__ __device__ constexpr int leaf_pitch(int BC) { return BC + 8; }

//: The three panels' address functions. Functions rather than open-coded products
//: because the `ldmatrix` fetches and the scalar staging walks must not be able to
//: disagree about them.
__host__ __device__ __forceinline__ constexpr int rp_index(int leaf, int t) {
  return leaf * kTokPitch + t;
}

__host__ __device__ __forceinline__ constexpr int vp_index(int ch, int t) {
  return ch * kTokPitch + t;
}

__host__ __device__ __forceinline__ constexpr int wp_index(int BC, int t, int leaf) {
  return t * leaf_pitch(BC) + leaf;
}

//: **THE STATE'S OWNERSHIP RULE -- THE ONE FACT REGISTER RESIDENCY IS.** The state is
//: held TRANSPOSED, `[DV channels][BC leaves]`, in the D fragments of the fold; the
//: `m` axis is therefore the CHANNEL axis and its quantum is the MMA's `m16`. A warp
//: owns one CHANNEL TILE and a contiguous ATOM RANGE of leaves, so the warp that
//: WRITES an element is the warp that READS it, at the same lane -- which is the whole
//: requirement. Warps beyond `kMTiles * reps` own no state: `reps` is capped by the
//: atom count because a readout k-step is one whole atom.
//: docs/internals/smem_layout.md#state-residency
constexpr int state_m_tiles(int DV) { return DV / kMmaKQuantum; }

constexpr int state_reps(int BC, int DV) {
  const int by_warps = (kConsumerThreads / 32) / state_m_tiles(DV);
  const int by_atoms = BC / kMmaKQuantum;
  return by_warps < by_atoms ? by_warps : by_atoms;
}

constexpr int state_atoms_per_warp(int BC, int DV) {
  return (BC / kMmaKQuantum) / state_reps(BC, DV);
}

constexpr int state_owner_warps(int BC, int DV) { return state_m_tiles(DV) * state_reps(BC, DV); }

//: fp32 registers PER LANE the state costs: one D fragment (4 floats) per 8-leaf tile,
//: two tiles per atom. 16 at `BC = DV = 64`, inside an FA-like envelope by a wide
//: margin (FA spends 255 registers per thread).
constexpr int state_regs_per_lane(int BC, int DV) { return 8 * state_atoms_per_warp(BC, DV); }

//: The prefetch slab is a `cp.async` DESTINATION, and every `ldmatrix` source row is
//: 16-byte aligned, so every region base carries the 16-byte requirement.
constexpr size_t align16(size_t x) { return (x + 15) & ~static_cast<size_t>(15); }

//: **THE AMPLITUDE PREFETCH SLAB'S ROW PITCH, IN bf16 ELEMENTS.** One row per WINDOW
//: TOKEN holds every SPANNED level's digit sub-range, concatenated: level `l`
//: contributes `align8(lead_l + span_l)` elements, where `span_l` is the owner's
//: contiguous digit range at that level and `lead_l = base_l & 7` is the alignment
//: slack the 16-byte copy leaves in front of it. Both slabs -- read and write -- take
//: this layout, because the digit map is side-independent and both streams' pitches
//: obey the same alignment rule.
//: THE COMPILE-TIME WORST CASE IS THE DEGENERATE FACTORISATION. The spanned spans
//: multiply to `BC` (the owner rectangle's invariant), and `sum_l align8(7 + f_l)`
//: over `prod_l f_l == BC` is maximised by one span of `BC` and the rest at 1.
//: docs/internals/smem_layout.md#prefetch-slab
constexpr int amp_slab_pitch(int BC, int D) { return ((7 + BC + 7) / 8) * 8 + 8 * (D - 1); }

//: The slab row's 16-byte chunk count -- the extent of the per-owner chunk
//: descriptors (`ch_lvl` / `ch_src` / `ch_atoms`) and of the issue loop.
constexpr int amp_slab_chunks(int BC, int D) { return amp_slab_pitch(BC, D) / 8; }

//: THE `Vp` / `clk` UNION, AND WHY IT IS SIZED IN **BYTES** RATHER THAN ELEMENTS.
//: `clk` is DEAD before `Vp` is live, so it aliases `Vp`'s storage and the region is
//: sized as the MAX OVER ITS LIVE TENANTS -- `Vp` is bf16 and `clk` fp32, so a
//: pitch-max would under-allocate by 2x, SILENTLY. Only the RESERVATION is the max;
//: `Vp` is still ADDRESSED at its own pitch. docs/internals/smem_layout.md#vw-clk-union
//: The `Vp` REGION is `W` staged slots; `clk` is ONE buffer over the whole window
//: (the clock prefix runs across the window, not across a group), so the two tenants
//: scale differently and the max is taken over both at the window's width.
template <int BC, int DV, int D, bool DECAY, int GT>
constexpr size_t vw_union_bytes() {
  (void)D;
  const size_t vp = static_cast<size_t>(window_groups(GT)) * DV * kTokPitch * sizeof(__nv_bfloat16);
  const size_t clk =
      DECAY ? static_cast<size_t>(GT) * (BC + 4) * sizeof(float) : static_cast<size_t>(0);
  return align16(vp > clk ? vp : clk);
}

//: The carve UP TO the prefetch slab. Split out because the slab's base must be
//: 16-byte aligned and the kernel forms that base as `smem_raw + align16(core)`; one
//: expression serves the reservation and the carve.
//: docs/internals/smem_layout.md#smem-ledger
template <int BC, int DV, int D, bool DECAY, int GT>
constexpr size_t consumer_smem_core_bytes() {
  constexpr int W = window_groups(GT);
  //: **THE STATE IS NOT HERE.** Its numerator lives in the fold's D fragments --
  //: registers, one channel tile and one atom range per warp -- for the whole kernel;
  //: only the MASS COLUMN keeps a shared-memory home, fp32, `[BC]` floats, because it
  //: is the denominator of a ratio and its consumers are broadcast matvecs.
  //: **TWO OF THEM: THE MASS COLUMN PING-PONGS.** Term 1 of the denominator READS the
  //: entry column while the fold's term 3 WRITES the exit column; holding them in
  //: separate buffers is what deletes that read-before-write edge, which was one of
  //: the two the visit's phase barrier carried. `BC` floats is the whole price.
  //: docs/internals/smem_layout.md#mass-pingpong
  size_t b = 2 * static_cast<size_t>(BC) * sizeof(float);  // the mass column, both phases
  //: `Rp`, `Wp` and `Vp` are `W` SLOTS OF THE GROUP'S OWN SHAPE, never one wider
  //: buffer: that is what carries every one of their layout proofs across the
  //: window unchanged. Only `Gm` -- the window Gram -- takes a new shape.
  b += static_cast<size_t>(W) * BC * kTokPitch * sizeof(__nv_bfloat16);                 // Rp
  b += static_cast<size_t>(W) * kGroupTokens * leaf_pitch(BC) * sizeof(__nv_bfloat16);  // Wp
  b += vw_union_bytes<BC, DV, D, DECAY, GT>();     // Vp (+ clk)
  b += static_cast<size_t>(D) * BC * sizeof(int);  // dig
  //: THE WINDOW GRAM, `[GT][GT + 8]`, holding `G^T` -- block `(b, a)` at row block
  //: `b`, so the intra term reads it as a `k = token_b`-major B operand. Square,
  //: though only the block lower triangle is ever written or read: a packed
  //: triangular layout would make a block's base a non-affine function of `(a, b)`,
  //: and one closed-form address function per buffer is the standard every other
  //: operand here is held to.
  b += static_cast<size_t>(GT) * (GT + 8) * sizeof(__nv_bfloat16);  // Gm
  if (DECAY) b += 3 * static_cast<size_t>(BC) * sizeof(float);      // log_keep, p_ref, s0
  //: THE COOKED SCHEDULE RING: `sched_slots(GT)` group descriptors, written by the
  //: PRODUCER WARP a whole window ahead and read by every consumer warp. It is the
  //: only schedule storage the consumers touch.
  b += static_cast<size_t>(sched_slots(GT)) * kSchedSlotInts * sizeof(int);
  //: The slot's PER-OWNER UPPER PRODUCTS (`sr`, `sw`) -- §3 step 3's row scales,
  //: cooked in the ring beside the member word, one pair of `kGroupTokens` floats
  //: per slot.
  b += static_cast<size_t>(sched_slots(GT)) * 2 * kGroupTokens * sizeof(float);
  //: THE PRODUCER'S CURSOR STATE (8 scalars + five 32-lane planes: the cooked
  //: chunk's fr/fw/member and the current word's two atom-bitlist rows). It rests
  //: HERE rather than in the producer warp's registers because scheduler state
  //: held in registers is live across the visit body and spills the tightest key:
  //: docs/internals/smem_layout.md#reg-budget.
  b += (8 + 5 * 32) * sizeof(int);
  //: THE PER-OWNER CHUNK DESCRIPTORS: for each 16-byte chunk of a slab row, the level
  //: it belongs to, its element offset inside that level's amplitude row, and the SET
  //: OF ATOMS whose digits it carries. Built once per owner; read once per issued
  //: copy. docs/internals/smem_layout.md#prefetch-slab
  b += 3 * static_cast<size_t>(amp_slab_chunks(BC, D)) * sizeof(int);
  //: THE GRAM'S PUBLICATION COUNTER -- one int, incremented once per producing warp
  //: per mixed window, spun on by the consumers of `Gm`. It replaces the visit's
  //: phase barrier: docs/internals/consumer_kernel.md#gram-publication
  b += sizeof(int);
  return b;
}

//: The per-CTA shared-memory footprint, `constexpr` so the launch bound is DERIVED
//: from it rather than restated next to it. Every term, its measured residency
//: consequence and `DECAY`'s carve: docs/internals/smem_layout.md#smem-ledger
template <int BC, int DV, int D, bool DECAY, int GT>
constexpr size_t consumer_smem_bytes() {
  //: THE AMPLITUDE PREFETCH SLABS, read and write: `GT` token rows of
  //: `amp_slab_pitch` bf16 each. They are the LAST region so that the 16-byte
  //: alignment `cp.async` needs is one align-up rather than a constraint on every
  //: region before them. The CLOCK stream does not ride this slab -- it is fp32 and
  //: its consumer is a per-leaf column walk, and a window-deep fp32 slab is what
  //: would push the wide window past the per-CTA cap. THE VALUE STREAM DOES NOT RIDE
  //: THIS SLAB EITHER, and that is a CAPACITY refusal rather than a design one --
  //: `GT * DV` bf16 is affordable on every `GT = 16` arm and is what takes
  //: `<BC=16, DV=64, D=4, GT=64>` from two resident CTAs to one:
  //: docs/architecture/guarantees-choices-preferences.md#value-stream-staging
  return align16(consumer_smem_core_bytes<BC, DV, D, DECAY, GT>())
         + 2 * static_cast<size_t>(GT) * amp_slab_pitch(BC, D) * sizeof(__nv_bfloat16);
}

}  // namespace consumer
}  // namespace rola
