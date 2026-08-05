#pragma once

// THE CONSUMER'S SHARED-MEMORY LAYOUT LAYER -- what one CTA's shared memory looks
// like, and what it costs. Three kinds of thing live here and they are one concern:
//
//   1. THE ADDRESS FUNCTIONS -- the CuTe-owned swizzle the staged operands are stored
//      under, `WT`'s derived pitch, and the `vw_index` / `wt_index` / `stage_offset`
//      maps every access point goes through, so no two sites can disagree;
//   2. THE PROOFS -- bank-conflict freedom and `ldmatrix` alignment as `constexpr`
//      predicates, enumerated rather than asserted, so a future `BT`/`BC`/`DV` fails
//      the BUILD instead of silently costing the 4.06x wavefront amplification the
//      unpadded build measured;
//   3. THE PRICE -- `consumer_smem_bytes_at` (the carve, as a byte count), the register
//      budget and residency arithmetic the launch bound is derived from, and
//      `consumer_ring_depth`, which is a FOOTPRINT decision and therefore belongs with
//      the footprint rather than with the ring's mechanism.
//
// The layout functions are the SPECIFICATION the kernel body's carve follows: the body
// computes its base pointers in the same order, and the two are tied together by
// `consumer_smem_bytes_at` being what the launch actually reserves.
//
// The ledger cell by cell, the swizzle's three proofs, the pitch table, the measured
// register budget and the ring-depth rule: docs/internals/smem_layout.md

#include <cuda_bf16.h>

//: THE VENDORED CuTe PIN (`csrc/third_party/cutlass`, v3.5.1, closed-world rule 5),
//: a CODEGEN input rather than a packaging one. This layer uses it for ONE thing:
//: `Vw`'s swizzled SMEM layout. The `ldmatrix` copy atoms that READ that layout are
//: `mma_tile.cuh`'s. docs/internals/smem_layout.md#cute-pin
#include <cute/tensor.hpp>

#include "arch_table.cuh"
//: For `kMaxWords`, the candidate-tile word ceiling the carve reserves two arrays of.
//: A LAYER HEADER STATES EVERY NAME IT USES: a header that compiles only when some
//: other file included its dependency first is a latent break, not a saving, and this
//: one was exactly that until `tools/header_selfcheck.py` compiled it alone.
#include "consumer.cuh"
#include "pipeline.cuh"
#include "readout_mma.cuh"

namespace rola {
namespace consumer {

constexpr int kConsumerThreads = 256;  // 8 warps: the CTA width an SMEM state frees

//: §8 CALIBRATION PLACEHOLDER `C_WT_WINDOW` (I5): how many `kMmaKQuantum`-leaf ATOMS
//: of the write panel `WT` the tile's second phase holds at once. `0` means `QPO`,
//: which IS the plane, i.e. today's structure exactly. It is the ALLOCATION as well
//: as the phase, and it is PINNED to the plane on the `DECAY` arm as a correctness
//: matter: docs/internals/smem_layout.md#wt-window
#ifndef ROLA_C_WT_WINDOW
#define ROLA_C_WT_WINDOW 0
#endif

//: §8 CALIBRATION PLACEHOLDER `C_RING_DEPTH` (I5): how many TILES' worth of the
//: STAGING SET the CTA holds at once. `1` is today's structure exactly; `2` is the
//: structural ceiling (the run-ahead is one tile, so a third slot has no producer).
//: The depth is DERIVED per instantiation by `consumer_ring_depth` below; `DECAY` is
//: pinned to 1 for correctness, and the default is 1 against §8's tabled 2 on a
//: measured driver regression: docs/internals/smem_layout.md#ring-depth
#ifndef ROLA_C_RING_DEPTH
#define ROLA_C_RING_DEPTH 1
#endif

//: **A MEASURED KERNEL FACT, NOT AN ARCHITECTURE FACT AND NOT A THRESHOLD.** The
//: register working set `ptxas` reports for this kernel's GEMM inner loop when given
//: slack, keyed on the three axes the working set MEASURABLY varies with -- `BC`,
//: `DECAY` and `NM` -- with every number read off `-Xptxas -v` and the MAXIMUM per
//: key taken, because a too-small budget is bought back by SPILLING.
//: Every table behind it (the two `ncu` arms that make `DECAY` a key, the five-budget
//: `NM` sweep, the forcing census and the duration gate that keeps `BC = 64` at 96):
//: docs/internals/smem_layout.md#reg-budget
constexpr int consumer_reg_budget(int BT, int BC, bool decay, MmaPrec read_prec, MmaPrec fold_prec,
                                  int DV) {
  const int nm = BT / 16;                // the tile's m16 ROW BLOCKS
  const int warp_tiles = nm * (DV / 8);  // NM x NT_DV, what the warps split
  const int warps = kConsumerThreads / 32;
  //: The ONE structural class the forcing census clears at 64, with zero spill and
  //: unchanged in-loop local counts. `BC < 64` is a DURATION verdict (+2.42 %
  //: forced), not a census one:
  //: docs/internals/smem_layout.md#reg-budget-session9
  if (nm >= 2 && BC < 64 && !decay && read_prec == MmaPrec::kBf16 && fold_prec == MmaPrec::kBf16
      && warp_tiles <= warps) {
    return 64;
  }
  const int base = (BC >= 64) ? 80 : (decay ? 80 : 64);
  return (nm >= 2) ? 96 : base;
}

//: CTAs/SM to STATE IN THE LAUNCH BOUND -- the smallest of what the three hardware
//: resources admit, computed per instantiation rather than fixed. All three terms are
//: load-bearing: a missing thread term makes `ptxas` DISCARD an out-of-range bound
//: (not clamp it), and a missing register term buys the extra CTAs by spilling.
//: The per-bound spill measurements and why the bound is stamped rather than left to
//: the compiler: docs/internals/smem_layout.md#residency
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
//: conflict-free exactly when those 32 addresses land on 32 DISTINCT banks. Enumerated
//: rather than asserted as a modular formula, so each operand proves its own pitch and
//: a future `BT`/`BC`/`DV` fails the build instead of silently costing the 4.06x
//: wavefront amplification the unpadded build measured.
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

//: **THE SAME PROOF FOR A 16-BIT OPERAND IS NOT THE SAME PREDICATE**, and the
//: difference is not pedantry: `banks_distinct` would REJECT layouts that are
//: conflict-free and ACCEPT ones that are not, so a bf16 buffer proved with it is not
//: proved. The bank of element `e` is `(e >> 1) % 32`, and two lanes in the SAME
//: 32-bit WORD do not conflict -- so the condition is "any two lanes sharing a bank
//: share a word", which `banks_distinct` cannot express. `stride_g` / `stride_r` are
//: the deltas in 2-BYTE ELEMENTS. Why `e >> 1` is well defined at every call site:
//: docs/internals/smem_layout.md#bank-predicates
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

//: THE STAGED OPERANDS' SMEM LAYOUT, OWNED BY CuTe. On the
//: narrow arm both staged operands are stored under the library's own
//: `composition(Swizzle<B,3,3>, (ROWS,COLS):(COLS,1))` and the PAD DISAPPEARS: the
//: swizzle decorrelates banks by permuting 16-byte blocks WITHIN the tile instead of
//: by making it bigger, so the buffer is exactly `ROWS * COLS` elements and ONE
//: layout serves `WT`'s two operand maps by construction. `B` is chosen so the XOR
//: field covers the whole 128-byte bank cycle; `M = 3` keeps the 16-byte `ldmatrix`
//: row intact; `S = 3` puts the XOR source at the row index's low bits. Nothing here
//: is asserted -- the three properties are `static_assert`ed by enumerating the
//: library's layout below: docs/internals/smem_layout.md#stage-layout
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
//: `StageLayout{}(row, col)` at every access point is NOT free (measured: +15 to +18
//: registers on the `BT = 32` instantiations, a CTA lost on 7 of 32 shapes) because it
//: blocks the strength reduction the linear map admits on the scalar row walks. This
//: is that same map in the two operations the swizzle actually is:
//: `row * COLS + (col ^ mask(row))`, a row-invariant mask the compiler CAN hoist.
//: `stage_offset_matches_cute` enumerates every `(row, col)` and requires the two to
//: agree, which is what keeps the library the authority:
//: docs/internals/smem_layout.md#stage-offset
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
//: is the column extent actually addressed (`BT` tokens), smaller than `PITCH`.
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

//: THE STAGED-OPERAND STORAGE FORMAT. `WT` and `Vw` are stored in the MMA's own format
//: on the bf16 production arm and fp32 on the tf32x3 exactness arm. THE FULL-MANTISSA
//: REQUIREMENT IS THE tf32x3 ARM'S ALONE -- that is the arm that exists to be exact --
//: while on the bf16 arm every consumer narrows at the fragment anyway, so storing
//: narrow MOVES the narrowing site and changes no value. The axis is the CONJUNCTION of
//: both sites because both buffers feed both:
//: docs/internals/smem_layout.md#stage-format
template <MmaPrec READ_PREC, MmaPrec FOLD_PREC>
constexpr bool stage_is_narrow() {
  return READ_PREC == MmaPrec::kBf16 && FOLD_PREC == MmaPrec::kBf16;
}

//: `WT`'s row pitch, and the `+8` IS THE PROOF'S ANSWER rather than a copy of `Gm`'s:
//: `WT` serves GEMM 2's B map and GEMM 4's A map, and enumerated over
//: `pad in {4, 8, 12, 16}` at both `BT`, `BT + 8` is the UNIQUE pad that passes both.
//: A swizzle here is DECLINED on measured evidence (`WT` has six addressing sites, two
//: of them scalar row walks); the `ldmatrix` ATOMS stand, which this pitch admits.
//: docs/internals/smem_layout.md#stage-format
template <int BT, MmaPrec READ_PREC, MmaPrec FOLD_PREC>
constexpr int wt_pitch() {
  return stage_is_narrow<READ_PREC, FOLD_PREC>() ? BT + 8 : BT + 4;
}

//: `Vw`'s and `WT`'s ELEMENT COUNTS and ADDRESS FUNCTIONS -- one decision taken twice:
//: the narrow arm stores them under `StageLayout` and pays NO pad, the `tf32x3` arm
//: keeps padded row-major buffers. One `kNarrowStage` storage-format split, not a
//: second axis.
template <int BT, int DV, bool NARROW>
constexpr int vw_elems() {
  return NARROW ? BT * DV : BT * (DV + 8);
}

template <int BT, int DV, bool NARROW>
__host__ __device__ __forceinline__ constexpr int vw_index(int t, int ch) {
  if constexpr (NARROW) {
    return stage_offset<DV>(t, ch);
  } else {
    return t * (DV + 8) + ch;
  }
}

//: `WT`'s ADDRESS FUNCTION -- row-major at `wt_pitch`, on BOTH arms. A function rather
//: than an open-coded `q * BTP + t` because the `ldmatrix` fetches and the two scalar
//: walks must not be able to disagree about it.
template <int BT, bool NARROW>
__host__ __device__ __forceinline__ constexpr int wt_index(int q, int t) {
  return q * (NARROW ? BT + 8 : BT + 4) + t;
}

constexpr size_t align4(size_t x) { return (x + 3) & ~static_cast<size_t>(3); }

//: FRAGMENT-ORDER OPERAND SYNTHESIS. We SYNTHESISE the read coefficients in SMEM
//: rather than receiving them, so we can store them in the order the MMA's A fragment
//: wants -- word `w` of lane `lane` at k-step `ks` at `[ks * 128 + lane * 4 + w]` --
//: and a warp loads a whole fragment with ONE 128-bit `uint4`: four full transactions,
//: no bank conflict and no address arithmetic at all. BOTH precisions are laid out
//: (`READ_PREC` is a real axis), sharing one storage SHAPE and differing only in
//: the compile-time forward map. The two maps, and why fragment-ordering leaves the
//: buffer with no row pitch at all: docs/internals/smem_layout.md#frag-order

//: 32-bit words in a fragment-ordered A operand contracting over `K`, FOR ONE `m16`
//: ROW BLOCK. A `BT`-token tile holds `NM = BT / 16` of them, block-major at a FIXED
//: stride of `frag_a_words(BC)` -- fixed rather than packed to the compacted `Kpad`, so
//: a block's base is a compile-time multiple and the MMA's `ks` offset is unchanged.
template <MmaPrec PREC>
constexpr int frag_a_words(int K) {
  return (PREC == MmaPrec::kBf16 ? K / 16 : K / 8) * 128;
}

//: THE `WT` WINDOW'S ROW COUNT, RESOLVED ONCE (§8 placeholder `C_WT_WINDOW`).
//: `wt_window_rows` is THE definition -- `consumer_smem_bytes` sizes the buffer from it
//: and the kernel body indexes with it, so the allocation and the addressing cannot
//: disagree about how many rows exist. `0` (and any out-of-range value) means `QPO`,
//: the whole plane. `DECAY` PINS THE PLANE, and it is a correctness pin:
//: docs/internals/smem_layout.md#wt-window
constexpr int wt_window_atoms(int BC, bool DECAY) {
  const int qpo = BC / kMmaKQuantum;
  const int req = (ROLA_C_WT_WINDOW <= 0 || ROLA_C_WT_WINDOW > qpo) ? qpo : ROLA_C_WT_WINDOW;
  return DECAY ? qpo : req;
}
constexpr int wt_window_rows(int BC, bool DECAY) {
  return wt_window_atoms(BC, DECAY) * kMmaKQuantum;
}

//: THE `Vw` / `clk` UNION, AND WHY IT IS SIZED IN **BYTES** RATHER THAN ELEMENTS.
//: `clk` is DEAD before `Vw` is live, so it aliases `Vw`'s storage and the region is
//: sized as the MAX OVER ITS LIVE TENANTS. Taken over PITCHES that max is only sound
//: while the two tenants share an element size -- once `Vw` is bf16 and `clk` fp32 it
//: under-allocates by 2x, SILENTLY. `Vw` is still ADDRESSED at its own pitch; only the
//: RESERVATION is the max, and on the DECAY arm `clk` is the larger tenant at every
//: shipped shape: docs/internals/smem_layout.md#vw-clk-union
template <int BT, int BC, int DV, bool DECAY, MmaPrec READ_PREC, MmaPrec FOLD_PREC>
constexpr size_t vw_union_bytes() {
  constexpr bool kNarrow = stage_is_narrow<READ_PREC, FOLD_PREC>();
  const size_t vw = static_cast<size_t>(vw_elems<BT, DV, kNarrow>())
                    * (kNarrow ? sizeof(__nv_bfloat16) : sizeof(float));
  const size_t clk =
      DECAY ? static_cast<size_t>(BT) * (BC + 4) * sizeof(float) : static_cast<size_t>(0);
  return align4(vw > clk ? vw : clk);
}

//: The per-CTA shared-memory footprint, `constexpr` so the launch bound is DERIVED
//: from it rather than restated next to it. Every term, its measured residency
//: consequence and the three per-arm carves (`DECAY`'s three buffers rather than four,
//: `COMPACT`'s 520 B and the fourth buffer the design budgeted that this kernel does
//: not need, `ctl`'s 544 B of total slack on the tightest arm):
//: docs/internals/smem_layout.md#smem-ledger
template <int BT, int BC, int DV, int D, MmaPrec READ_PREC, MmaPrec FOLD_PREC, bool DECAY,
          bool GLOBAL, bool COMPACT, int RING>
constexpr size_t consumer_smem_bytes_at() {
  size_t b = static_cast<size_t>(BC) * (DV + 4) * sizeof(float);    // S
  b += RING * 2 * static_cast<size_t>(BT) * sizeof(float);          // sr, sw
  b += (GLOBAL ? 1 : 0) * static_cast<size_t>(BT) * sizeof(float);  // den
  b += 2 * static_cast<size_t>(kMaxWords) * sizeof(uint32_t);       // candidates
  b += static_cast<size_t>(D) * BC * sizeof(int);                   // dig
  //: `WT` IS ALLOCATED TO THE WINDOW, NOT TO THE PLANE -- at the shipped default the
  //: window IS the plane and this is `BC` rows exactly.
  b += align4(static_cast<size_t>(wt_window_rows(BC, DECAY)) * wt_pitch<BT, READ_PREC, FOLD_PREC>()
              * (stage_is_narrow<READ_PREC, FOLD_PREC>() ? sizeof(__nv_bfloat16) : sizeof(float)));
  b += RING * vw_union_bytes<BT, BC, DV, DECAY, READ_PREC, FOLD_PREC>();  // Vw (+ clk)
  b += RING * static_cast<size_t>(BT / 16) * frag_a_words<READ_PREC>(BC)
       * sizeof(unsigned);  // Rf, one A operand per m16 row block
  b += (READ_PREC == MmaPrec::kBf16) ? static_cast<size_t>(BT) * (BT + 8) * sizeof(__nv_bfloat16)
                                     : static_cast<size_t>(BT) * (BT + 4) * sizeof(float);  // Gm
  if (DECAY) b += 3 * static_cast<size_t>(BC) * sizeof(float);  // log_keep, p_ref, s0
  //: RINGED: the gathered list is a staging OUTPUT, so a run-ahead gather must not
  //: overwrite the live one.
  if (COMPACT) b += RING * (2 * static_cast<size_t>(BC) + 2) * sizeof(int);
  b += static_cast<size_t>(ctl_words(BC)) * sizeof(int);  // ctl
  return b;
}

//: THE RING DEPTH, DERIVED. One rule, applied per instantiation: take the deeper
//: staging set iff it costs NO CTA, off the same `ctas_for_residency`
//: `__launch_bounds__` is derived from, so the ring cannot silently disagree with the
//: bound the binary is compiled under. The `DECAY` pin is a CORRECTNESS term and is
//: applied FIRST, ahead of any footprint test (D2.3), and the shared-memory term is
//: checked SEPARATELY because a `by_regs` floor can mask an SMEM drop the driver then
//: sees: docs/internals/smem_layout.md#ring-depth
template <int BT, int BC, int DV, int D, MmaPrec READ_PREC, MmaPrec FOLD_PREC, bool DECAY,
          bool GLOBAL, bool COMPACT>
constexpr int consumer_ring_depth() {
  if (DECAY) return 1;  // D2.3, correctness
  constexpr int req = (ROLA_C_RING_DEPTH >= 2) ? 2 : 1;
  if (req == 1) return 1;
  constexpr int budget = consumer_reg_budget(BT, BC, DECAY, READ_PREC, FOLD_PREC, DV);
  constexpr size_t one =
      consumer_smem_bytes_at<BT, BC, DV, D, READ_PREC, FOLD_PREC, DECAY, GLOBAL, COMPACT, 1>();
  constexpr size_t two =
      consumer_smem_bytes_at<BT, BC, DV, D, READ_PREC, FOLD_PREC, DECAY, GLOBAL, COMPACT, 2>();
  if (ctas_for_residency(two, budget) != ctas_for_residency(one, budget)) return 1;
  //: AND THE SHARED-MEMORY TERM ON ITS OWN, which the line above does NOT imply.
  constexpr int threads = kSm.max_threads_per_sm / kConsumerThreads;
  constexpr int smem_one = static_cast<int>(kSm.smem_per_sm / (one + kDriverSmemReservePerCta));
  constexpr int smem_two = static_cast<int>(kSm.smem_per_sm / (two + kDriverSmemReservePerCta));
  const int cap_one = smem_one < threads ? smem_one : threads;
  const int cap_two = smem_two < threads ? smem_two : threads;
  return cap_two == cap_one ? 2 : 1;
}

template <int BT, int BC, int DV, int D, MmaPrec READ_PREC, MmaPrec FOLD_PREC, bool DECAY,
          bool GLOBAL, bool COMPACT>
constexpr size_t consumer_smem_bytes() {
  return consumer_smem_bytes_at<
      BT, BC, DV, D, READ_PREC, FOLD_PREC, DECAY, GLOBAL, COMPACT,
      consumer_ring_depth<BT, BC, DV, D, READ_PREC, FOLD_PREC, DECAY, GLOBAL, COMPACT>()>();
}

}  // namespace consumer
}  // namespace rola
