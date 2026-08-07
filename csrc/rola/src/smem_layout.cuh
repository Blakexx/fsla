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

//: THE MEMBERSHIP SCAN'S BATCH (the frozen contract's cook-once rule): how many
//: 32-token words one refill stages into SMEM. The scan words are CTA-uniform,
//: so a per-thread (per-warp-redundant) global walk pays 8x the sectors and
//: instructions the stream carries -- MEASURED as the whole of the endgame's
//: first-cut regression (+7.6M L1 sectors, +57M inst at atscale-L2048). One
//: thread loads each word's rows ONCE and PRECOMPUTES the member word into the
//: batch; the assembler consumes SMEM. Two uniform barriers per kScanBatch
//: words. 32 = one word per thread of one warp per refill, a power of two well
//: above the split-carry horizon.
constexpr int kScanBatch = 32;

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
constexpr int consumer_reg_budget(int BC, bool decay, int DV) {
  (void)DV;
  return (BC >= 64) ? 80 : (decay ? 80 : 64);
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
  b += 2 * static_cast<size_t>(kGroupTokens) * sizeof(float);               // sr, sw
  b += static_cast<size_t>(kGroupTokens) * sizeof(float);                   // den
  b += static_cast<size_t>(D) * BC * sizeof(int);                           // dig
  b += align4(static_cast<size_t>(BC) * kWtPitch * sizeof(__nv_bfloat16));  // WT
  b += vw_union_bytes<BC, DV, DECAY>();                                     // Vw (+ clk)
  b += static_cast<size_t>(frag_a_words(BC)) * sizeof(unsigned);            // Rf
  b += static_cast<size_t>(kGroupTokens) * (kGroupTokens + 8) * sizeof(__nv_bfloat16);  // Gm
  if (DECAY) b += 3 * static_cast<size_t>(BC) * sizeof(float);  // log_keep, p_ref, s0
  b += static_cast<size_t>(ctl_words(BC)) * sizeof(int);        // ctl
  //: The group scheduler's SMEM block: `t_idx` (`kGroupTokens` global token
  //: indices, warp 0 at ballot time) + `g_scan` (the assembler's cross-visit
  //: carry: fill, word cursor, fr, fw, member remainder, flags, plus the
  //: scan's `D` fixed-level row indices -- SMEM so no scheduler state is
  //: register-live across the visit body; the atom masks are CTA-uniform
  //: registers, derived word-wise from the bitlists at ballot).
  b += static_cast<size_t>(kGroupTokens + 6 + D) * sizeof(int);  // t_idx + g_scan
  b += 3 * static_cast<size_t>(kScanBatch) * sizeof(int);        // the scan batch (fr, fw, member)
  return b;
}

}  // namespace consumer
}  // namespace rola
