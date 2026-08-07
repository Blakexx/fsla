#pragma once

// THE CONSUMER'S SHARED-MEMORY LAYOUT LAYER -- what one CTA's shared memory looks
// like, and what it costs. Three kinds of thing live here and they are one concern:
//
//   1. THE ADDRESS FUNCTIONS -- the CuTe-owned swizzle the staged operands are stored
//      under, `WT`'s derived pitch, and the `vw_index` / `wt_index` / `stage_offset`
//      maps every access point goes through, so no two sites can disagree;
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

//: THE VENDORED CuTe PIN (`csrc/third_party/cutlass`, v3.5.1, closed-world rule 5),
//: a CODEGEN input rather than a packaging one. This layer uses it for ONE thing:
//: the staged operands' swizzled SMEM layout. The `ldmatrix` copy atoms that READ
//: that layout are `mma_tile.cuh`'s. docs/internals/smem_layout.md#cute-pin
#include <cute/tensor.hpp>

#include "arch_table.cuh"
#include "pipeline.cuh"
#include "readout_mma.cuh"

namespace rola {
namespace consumer {

constexpr int kConsumerThreads = 256;  // 8 warps: the CTA width an SMEM state frees

//: THE GROUP -- the executor's work unit: `kMmaKQuantum` gathered tokens, one MMA
//: m-quantum. It is deliberately THE SAME constant as the k-quantum: one multiply's
//: worth of tokens against one multiply's worth of state rows.
constexpr int kGroupTokens = kMmaKQuantum;

//: THE FOLD'S STAGED-GROUP DEPTH (Blake's confirmed atom-major nest,
//: 2026-08-06): how many groups' staged panels the fold consumes per atom
//: residency. PINNED 1 today -- the staging carve holds one group's set -- and
//: it is the ONE constant the per-atom fetch era deepens (together with the
//: staging producer); the fold's loop nest is already written over it, so
//: deepening is not a migration. The depth > 1 decay/window algebra is that
//: era's design item.
constexpr int kFoldStagedGroups = 1;

//: THE SCHEDULE CHUNK (the frozen contract's cook-once rule, item 2): how many
//: 32-token words the PRODUCER WARP cooks per refill. It is exactly one word
//: per lane, so the cooked chunk lives in REGISTERS -- the contract's "cooks the
//: per-token schedule ONE CHUNK AHEAD in registers" -- and the chunk's live-word
//: set is one `__ballot_sync`. A chunk in registers needs no barrier and no
//: storage, which is why the schedule-exhausted barrier dissolves (item 6).
//: docs/internals/smem_layout.md#sched-chunk
constexpr int kSchedChunkWords = 32;
static_assert(kSchedChunkWords == 32, "the cooked chunk is one word per lane of the producer warp");

//: **THE RUN-AHEAD DEPTH -- change-2's first dial.** How many groups the producer
//: warp runs ahead of the consumers. It is a `constexpr` (overridable at compile
//: time for the dial sweep), NEVER a template axis: the instantiation matrix does
//: not gain a coordinate. The ring holds `depth + 1` slots, so the ring is always
//: at least 2 (contract item 2: the producer must never write the slot being
//: read) and `depth = 1` is the CONFIGURATION-COLLAPSIBLE NULL -- minimal
//: machinery, one group of run-ahead, one descriptor of extra storage.
//: docs/internals/smem_layout.md#sched-ring
#ifndef ROLA_SCHED_DEPTH
#define ROLA_SCHED_DEPTH 1
#endif
constexpr int kSchedDepth = ROLA_SCHED_DEPTH;
static_assert(kSchedDepth >= 1, "the producer must run at least one group ahead");
constexpr int kSchedSlots = kSchedDepth + 1;

//: **THE VALUES-SHARD DIAL (contract item 4) IS STRUCTURALLY INERT AT
//: `kFoldStagedGroups == 1`, AND IS THEREFORE NOT BUILT -- a derivation, not an
//: omission.** The dial's stated selection criterion is the VALUES-RING
//: HEAD-OF-LINE residual: with one shared value tile the slowest atom consumer
//: paces the ring's recycling, and `k` shards buy pacing slack by giving disjoint
//: atom sets their own copies. That residual is a function of how long an atom
//: holds the tile. At `kFoldStagedGroups == 1` every atom consumes the staged
//: tile INSIDE ONE VISIT, between the same two barriers, so the tile's lifetime
//: is the visit and its recycling condition is discharged by the end-of-visit
//: `__syncthreads()` that already exists: THE CURSOR SPREAD IS ZERO AND THERE IS
//: NOTHING TO SHARD. `k > 1` would be `k` byte-identical copies of one tile that
//: no code path distinguishes -- dead storage behind a dead branch.
//: The dial becomes live exactly when the fold's staged depth deepens (an atom
//: then holds its accumulator across several groups' tiles and the spreads open
//: up), which is why this note lives next to the constant that gates it. Until
//: then `k` is a PLAN-TIME quantity only, pinned at 1 by the host.
static_assert(kFoldStagedGroups == 1,
              "the values-shard dial's premise changed: with the fold holding more "
              "than one staged group per atom residency the cursor spread is no "
              "longer zero, so `k` stops being inert and has to be built and swept");

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

//: **A MEASURED KERNEL FACT, NOT AN ARCHITECTURE FACT AND NOT A THRESHOLD.** The
//: register working set `ptxas` reports for this kernel's GEMM inner loop when given
//: slack, keyed on the axes the working set MEASURABLY varies with -- `BC` and
//: `DECAY` -- with every number read off `-Xptxas -v` and the MAXIMUM per key taken,
//: because a too-small budget is bought back by SPILLING. (The former `NM` key died
//: with the BT axis: every group is one m16 row block.)
//: docs/internals/smem_layout.md#reg-budget
constexpr int consumer_reg_budget(int BC, bool decay, int DV, int D) {
  (void)DV;
  //: `D` JOINED THE KEY when the producer landed, and it was added the way the
  //: rest of this function's numbers were: by census, not by taste. Level count
  //: is a structural template axis, and the cook's working set grows with it (one
  //: fixed-level row per level, per side, per refill). Measured over all 32 arms
  //: on both arches: at `D <= 3` the `BC = 16, decay = 0` arms fit 64 with zero
  //: spill; at `D = 4` they do not, and no amount of moving the cook's carried
  //: state into SMEM closes it (four successive diets took the matrix
  //: 120 B -> 28 B -> 16 B and stalled there, on sm_86 only).
  //:
  //: THE COST IS REAL AND IS RECORDED RATHER THAN HIDDEN: 80 puts `by_regs` at
  //: 65536 / (256 * 80) = 3, so these two arms run 3 CTAs/SM where 64 ran 4, and
  //: that is worth about -7.5% on the `topo-D4-w8x8x8x16-BC16` wall (measured,
  //: `workflows/perf/p25_change2.md`). It is spent to hold the ZERO-SPILL line;
  //: the measurement saying the 8 B spill was the cheaper of the two is on file
  //: for whoever revisits it.
  return (BC >= 64 || decay || D >= 4) ? 80 : 64;
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

//: **THE BANK-CONFLICT PROPERTY, AS A COMPILE-TIME PROOF RATHER THAN A COMMENT.** An
//: MMA operand load's 32 lanes address SMEM as an affine function of the fragment's
//: two lane coordinates, `g = lane >> 2` and `r = lane & 3`; the access is
//: conflict-free exactly when those 32 addresses land on 32 DISTINCT banks.
//: `stride_g` / `stride_r` are the address deltas in 4-BYTE BANKS per unit `g`/`r`.
//: docs/internals/smem_layout.md#bank-predicates
constexpr bool banks_distinct(int stride_g, int stride_r) {
  unsigned seen = 0;
  for (int g = 0; g < 8; ++g) {
    for (int r = 0; r < 4; ++r) {
      const int b = ((g * stride_g + r * stride_r) % 32 + 32) % 32;
      if (seen & (1u << b)) return false;
      seen |= 1u << b;
    }
  }
  return true;
}

//: **THE SAME PROOF FOR A 16-BIT OPERAND IS NOT THE SAME PREDICATE**: the bank of
//: element `e` is `(e >> 1) % 32`, and two lanes in the SAME 32-bit WORD do not
//: conflict -- so the condition is "any two lanes sharing a bank share a word",
//: which `banks_distinct` cannot express. `stride_g` / `stride_r` are the deltas in
//: 2-BYTE ELEMENTS. docs/internals/smem_layout.md#bank-predicates
constexpr bool banks_distinct_u16(int stride_g, int stride_r) {
  int word[32] = {};
  int bank[32] = {};
  int n = 0;
  for (int g = 0; g < 8; ++g) {
    for (int r = 0; r < 4; ++r) {
      const int e = g * stride_g + r * stride_r;
      word[n] = e >> 1;
      bank[n] = (e >> 1) % 32;
      ++n;
    }
  }
  for (int i = 0; i < 32; ++i) {
    for (int k = i + 1; k < 32; ++k) {
      if (bank[i] == bank[k] && word[i] != word[k]) return false;
    }
  }
  return true;
}

//: THE STAGED OPERANDS' SMEM LAYOUT, OWNED BY CuTe: both staged operands are stored
//: under the library's own `composition(Swizzle<B,3,3>, (ROWS,COLS):(COLS,1))` and
//: the PAD DISAPPEARS: the swizzle decorrelates banks by permuting 16-byte blocks
//: WITHIN the tile instead of by making it bigger, so the buffer is exactly
//: `ROWS * COLS` elements. `B` is chosen so the XOR field covers the whole 128-byte
//: bank cycle; `M = 3` keeps the 16-byte `ldmatrix` row intact; `S = 3` puts the XOR
//: source at the row index's low bits. Nothing here is asserted -- the three
//: properties are `static_assert`ed by enumerating the library's layout below:
//: docs/internals/smem_layout.md#stage-layout
template <int COLS>
constexpr int stage_swizzle_b() {
  return COLS >= 64 ? 3 : (COLS >= 32 ? 2 : 1);
}

template <int ROWS, int COLS>
using StageLayout =
    decltype(cute::composition(cute::Swizzle<stage_swizzle_b<COLS>(), 3, 3>{},
                               cute::Layout<cute::Shape<cute::Int<ROWS>, cute::Int<COLS>>,
                                            cute::Stride<cute::Int<COLS>, cute::_1>>{}));

//: THE ADDRESS FUNCTION IS A CLOSED FORM, AND CuTe IS ITS SPECIFICATION. Calling
//: `StageLayout{}(row, col)` at every access point measurably blocks the strength
//: reduction the linear map admits on the scalar row walks; this is that same map in
//: the two operations the swizzle actually is: `row * COLS + (col ^ mask(row))`, a
//: row-invariant mask the compiler CAN hoist. `stage_offset_matches_cute` enumerates
//: every `(row, col)` and requires the two to agree, which is what keeps the library
//: the authority: docs/internals/smem_layout.md#stage-offset
template <int COLS>
__host__ __device__ constexpr int stage_xor_mask(int row) {
  return static_cast<int>((static_cast<unsigned>(row * COLS) >> 6)
                          & ((1u << stage_swizzle_b<COLS>()) - 1u))
         << 3;
}

template <int COLS>
__host__ __device__ __forceinline__ constexpr int stage_offset(int row, int col) {
  return row * COLS + (col ^ stage_xor_mask<COLS>(row));
}

template <int ROWS, int COLS>
constexpr bool stage_offset_matches_cute() {
  static_assert(COLS >= 16, "the swizzle's XOR source must come from the row index");
  static_assert(COLS >= 8 << stage_swizzle_b<COLS>(),
                "the swizzle's XOR target must lie inside one row");
  for (int r = 0; r < ROWS; ++r)
    for (int c = 0; c < COLS; ++c)
      if (stage_offset<COLS>(r, c) != StageLayout<ROWS, COLS>{}(r, c)) return false;
  return true;
}

//: PROOF 1 -- the layout is a BIJECTION onto `[0, ROWS * COLS)`, which is what
//: licenses the reservation: a swizzle that overflowed its tile would alias the NEXT
//: buffer. docs/internals/smem_layout.md#stage-layout
template <int ROWS, int COLS>
constexpr bool stage_layout_is_bijection() {
  bool seen[ROWS * COLS] = {};
  for (int t = 0; t < ROWS; ++t) {
    for (int c = 0; c < COLS; ++c) {
      const int o = StageLayout<ROWS, COLS>{}(t, c);
      if (o < 0 || o >= ROWS * COLS || seen[o]) return false;
      seen[o] = true;
    }
  }
  return true;
}

//: PROOF 2 -- every `ldmatrix` FETCH is conflict-free: the instruction is serviced 8
//: lanes at a time, each lane reading a 16-byte row, so the phase is conflict-free
//: exactly when its 8 rows cover all 32 banks. It REPLACES `banks_distinct_u16` for
//: the staged operands and additionally checks the 16-byte ALIGNMENT `ldmatrix`
//: requires, which no pitch predicate could see.
template <int ROWS, int COLS>
constexpr bool stage_ldsm_banks_ok() {
  for (int r0 = 0; r0 + 8 <= ROWS; r0 += 8) {
    for (int c0 = 0; c0 + 8 <= COLS; c0 += 8) {
      unsigned banks = 0;
      for (int l = 0; l < 8; ++l) {
        const size_t byte = static_cast<size_t>(StageLayout<ROWS, COLS>{}(r0 + l, c0)) * 2;
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

//: THE SAME `ldmatrix` PHASE PREDICATE, ASKED OF A PITCHED (UNSWIZZLED) BUFFER. `WT`
//: is stored at a derived pad, so this LICENSES the atom at that pitch rather than
//: assuming a pad chosen for a scalar fragment map also serves a matrix load. `COLS`
//: is the column extent actually addressed (`kGroupTokens`), smaller than `PITCH`.
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

//: PROOF 3 -- the STAGING STORE stays conflict-free. Both staged buffers are filled by
//: a flat `idx = row * COLS + col` loop at `kConsumerThreads` stride, so the same-word
//: rule applies, asked of the swizzled addresses rather than of a stride.
template <int ROWS, int COLS>
constexpr bool stage_store_banks_ok() {
  for (int base = 0; base + 32 <= ROWS * COLS; base += 32) {
    int word[32] = {};
    int bank[32] = {};
    for (int l = 0; l < 32; ++l) {
      const int idx = base + l;
      const int e = StageLayout<ROWS, COLS>{}(idx / COLS, idx % COLS);
      word[l] = e >> 1;
      bank[l] = (e >> 1) % 32;
    }
    for (int i = 0; i < 32; ++i)
      for (int k = i + 1; k < 32; ++k)
        if (bank[i] == bank[k] && word[i] != word[k]) return false;
  }
  return true;
}

//: `WT`'s row pitch, and the `+8` IS THE PROOF'S ANSWER: `WT` serves GEMM 2's B map
//: and GEMM 4's A map, and enumerated over `pad in {4, 8, 12, 16}`,
//: `kGroupTokens + 8` is the UNIQUE pad that passes both. A swizzle here is DECLINED
//: on measured evidence (`WT` has six addressing sites, two of them scalar row
//: walks); the `ldmatrix` ATOMS stand, which this pitch admits.
//: docs/internals/smem_layout.md#stage-format
constexpr int kWtPitch = kGroupTokens + 8;

//: `Vw`'s ELEMENT COUNT and ADDRESS FUNCTION: stored under `StageLayout`, NO pad.
template <int DV>
constexpr int vw_elems() {
  return kGroupTokens * DV;
}

template <int DV>
__host__ __device__ __forceinline__ constexpr int vw_index(int t, int ch) {
  return stage_offset<DV>(t, ch);
}

//: `WT`'s ADDRESS FUNCTION -- row-major (leaf-major) at `kWtPitch`. A function rather
//: than an open-coded product because the `ldmatrix` fetches and the scalar walks
//: must not be able to disagree about it.
__host__ __device__ __forceinline__ constexpr int wt_index(int q, int t) {
  return q * kWtPitch + t;
}

constexpr size_t align4(size_t x) { return (x + 3) & ~static_cast<size_t>(3); }

//: FRAGMENT-ORDER OPERAND SYNTHESIS. We SYNTHESISE the read coefficients in SMEM
//: rather than receiving them, so we can store them in the order the MMA's A fragment
//: wants -- word `w` of lane `lane` at k-step `ks` at `[ks * 128 + lane * 4 + w]` --
//: and a warp loads a whole fragment with ONE 128-bit `uint4`: four full transactions,
//: no bank conflict and no address arithmetic at all. ONE k-step covers ONE ATOM
//: (`kMmaKQuantum` leaves), so the buffer is atom-blocked BY CONSTRUCTION -- which is
//: what lets the atom walk index a live atom's block affinely.
//: docs/internals/smem_layout.md#frag-order
constexpr int frag_a_words(int K) { return (K / kMmaKQuantum) * 128; }

//: THE `Vw` / `clk` UNION, AND WHY IT IS SIZED IN **BYTES** RATHER THAN ELEMENTS.
//: `clk` is DEAD before `Vw` is live, so it aliases `Vw`'s storage and the region is
//: sized as the MAX OVER ITS LIVE TENANTS -- `Vw` is bf16 and `clk` fp32, so a
//: pitch-max would under-allocate by 2x, SILENTLY. `Vw` is still ADDRESSED at its
//: own layout; only the RESERVATION is the max, and on the DECAY arm `clk` is the
//: larger tenant at every shipped shape:
//: docs/internals/smem_layout.md#vw-clk-union
template <int BC, int DV, bool DECAY>
constexpr size_t vw_union_bytes() {
  const size_t vw = static_cast<size_t>(vw_elems<DV>()) * sizeof(__nv_bfloat16);
  const size_t clk =
      DECAY ? static_cast<size_t>(kGroupTokens) * (BC + 4) * sizeof(float) : static_cast<size_t>(0);
  return align4(vw > clk ? vw : clk);
}

//: The per-CTA shared-memory footprint, `constexpr` so the launch bound is DERIVED
//: from it rather than restated next to it. Every term, its measured residency
//: consequence and `DECAY`'s carve: docs/internals/smem_layout.md#smem-ledger
template <int BC, int DV, int D, bool DECAY>
constexpr size_t consumer_smem_bytes() {
  size_t b = static_cast<size_t>(BC) * (DV + 4) * sizeof(float);            // S (+ mass column pad)
  b += static_cast<size_t>(kGroupTokens) * sizeof(float);                   // den
  b += static_cast<size_t>(D) * BC * sizeof(int);                           // dig
  b += align4(static_cast<size_t>(BC) * kWtPitch * sizeof(__nv_bfloat16));  // WT
  b += vw_union_bytes<BC, DV, DECAY>();                                     // Vw (+ clk)
  b += static_cast<size_t>(frag_a_words(BC)) * sizeof(unsigned);            // Rf
  b += static_cast<size_t>(kGroupTokens) * (kGroupTokens + 8) * sizeof(__nv_bfloat16);  // Gm
  if (DECAY) b += 3 * static_cast<size_t>(BC) * sizeof(float);  // log_keep, p_ref, s0
  b += static_cast<size_t>(ctl_words(BC)) * sizeof(int);        // ctl
  //: THE COOKED SCHEDULE RING (change 2): `kSchedSlots` group descriptors,
  //: written by the PRODUCER WARP a whole visit ahead and read by every
  //: consumer warp. It REPLACES the base's `t_idx` + `g_scan` cross-visit carry
  //: and its 3 x `kSchedChunkWords` scan batch: the assembler's cursor state is
  //: now warp-private (registers, no carry) and the cooked chunk is register-
  //: resident, so the ring is the ONLY scheduler storage that remains.
  b += static_cast<size_t>(kSchedSlots) * kSchedSlotInts * sizeof(int);
  //: The slot's PER-OWNER UPPER PRODUCTS (`sr`, `sw`) -- §3 step 3's row scales,
  //: cooked in the ring beside the member word (the stream-inventory amendment).
  //: They REPLACE the standalone `sr`/`sw` pair the carve used to hold, so the
  //: delta is `kSchedDepth` EXTRA copies of `2 * kGroupTokens` floats, not a new
  //: region.
  b += static_cast<size_t>(kSchedSlots) * 2 * kGroupTokens * sizeof(float);
  //: THE PRODUCER'S CURSOR STATE (8 scalars + five 32-lane planes: the cooked
  //: chunk's fr/fw/member and the current word's two atom-bitlist rows). It rests
  //: HERE rather than in the producer warp's registers because register-liveness
  //: across the visit body is the base's MEASURED §7.3 stateful-walk hazard --
  //: held in registers it spilled every `BC = 16, decay = 0` arm on both arches.
  b += (8 + 5 * 32) * sizeof(int);
  return b;
}

}  // namespace consumer
}  // namespace rola
