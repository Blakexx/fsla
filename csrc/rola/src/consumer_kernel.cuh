#pragma once

// THE RoLA V3 CONSUMER KERNEL. It is the ONLY consumer, and this file is its BODY:
// the §3/§5 tile algorithm, the compile-time budget, the
// SMEM carve, the MMA operand maps, the staging ring, the barrier/retirement
// protocol and the split-K epilogue. `ConsumerParams` and the host dispatch are
// `consumer.{cuh,cu}`; the supported shape surface is declared at
// `launch_consumer` and is deliberately NOT restated here.
//
// Three facts a reader must not miss:
//   1. THE STATE LIVES IN SMEM (`S[BC][SP]`), not in registers (§10i.13). That is
//      what decouples the thread count from the state layout, frees the CTA to 8
//      warps, and puts every per-tile term on the tensor cores as one of four
//      GEMMs (readout, Gram, intra, fold).
//   2. THE ONE-COPY INVARIANT IS A PROPERTY OF THE LAUNCH SHAPE, not of a policy:
//      one CTA per owner, run to completion, one HBM read and one HBM write of the
//      owner's state block. There is no wave machinery anywhere.
//   3. EVERY STRUCTURAL AXIS IS A TEMPLATE PARAMETER and no axis has a runtime
//      fallback -- one comprehensive path ships.
//
// The order of operations, the symbol glossary, the measured x2.32 at L=2048 against
// a register-resident consumer, every register/occupancy measurement behind the
// constants below, the bank-conflict proofs, the deadlock argument and the hazard
// index: docs/internals/consumer_kernel.md

#include <cuda/std/type_traits>
#include <cuda_bf16.h>

//: THE VENDORED CuTe PIN (`csrc/third_party/cutlass`, v3.5.1, closed-world rule 5),
//: a CODEGEN input rather than a packaging one. Used for exactly two things and both
//: are LAYOUT: `Vw`'s swizzled SMEM layout and the `ldmatrix` copy atoms. The
//: `mma.sync` calls stay hand-rolled PTX and the correspondence is `static_assert`ed
//: against the library's own maps: docs/internals/consumer_kernel.md#cute-pin
#include <cute/atom/copy_traits_sm75.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/tensor.hpp>

#include "arch_table.cuh"
#include "intra_panel.cuh"
#include "consumer.cuh"
#include "mma.cuh"
#include "readout_mma.cuh"
#include "segment_mask.cuh"

namespace rola {
namespace consumer {

using rola::mma::mma_m16n8k16;
using rola::mma::pack_bf16x2;
using rola::clamp_bf16_finite;
//: §12 rules 1 and 3's survival factors, taken from `intra_panel.cuh` rather than
//: reimplemented: it is the one clamp point and the one exponent sign convention.
using rola::panel_read_factor;
using rola::panel_write_factor;
//: §6.4's per-segment tile-range mask -- one named definition, because the host-side
//: segment partition at `P > 1` is pinned against exactly this arithmetic.
using rola::detail::segment_word_mask;

constexpr int kConsumerThreads = 256;  // 8 warps: the CTA width an SMEM state frees

//: `MMA_K_QUANTUM` -- the hardware `mma.sync` k-extent on the sm80-class bf16 path,
//: and THE named constant every quanta-shaped quantity in this kernel is an
//: expression in (spec §1 I1: no `BC`-derived literals). `QPO = BC / MMA_K_QUANTUM`
//: is "quanta per owner"; an ATOM is one `MMA_K_QUANTUM`-leaf block in canonical leaf
//: order. The planner mirrors this value at `rola.planner.schedule.MMA_K_QUANTUM`.
constexpr int kMmaKQuantum = 16;

//: §8 CALIBRATION PLACEHOLDER `C_WT_WINDOW` (I5): how many `MMA_K_QUANTUM`-leaf ATOMS
//: of the write panel `WT` the tile's second phase holds at once. `0` means `QPO`,
//: which IS the plane, i.e. today's structure exactly. It is the ALLOCATION as well
//: as the phase, and it is PINNED to the plane on the `DECAY` arm as a correctness
//: matter: docs/internals/consumer_kernel.md#wt-window
#ifndef ROLA_C_WT_WINDOW
#define ROLA_C_WT_WINDOW 0
#endif

//: §8 CALIBRATION PLACEHOLDER `C_RING_DEPTH` (I5): how many TILES' worth of the
//: STAGING SET the CTA holds at once. `1` is today's structure exactly; `2` is the
//: structural ceiling (the run-ahead is one tile, so a third slot has no producer).
//: The depth is DERIVED per instantiation by `consumer_ring_depth` below; `DECAY` is
//: pinned to 1 for correctness, and the default is 1 against §8's tabled 2 on a
//: measured driver regression: docs/internals/consumer_kernel.md#ring-depth
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
//: docs/internals/consumer_kernel.md#reg-budget
constexpr int consumer_reg_budget(int BT, int BC, bool decay, MmaPrec read_prec, MmaPrec fold_prec,
                                  int DV) {
  const int nm = BT / 16;                // the tile's m16 ROW BLOCKS
  const int warp_tiles = nm * (DV / 8);  // NM x NT_DV, what the warps split
  const int warps = kConsumerThreads / 32;
  //: The ONE structural class the forcing census clears at 64, with zero spill and
  //: unchanged in-loop local counts. `BC < 64` is a DURATION verdict (+2.42 %
  //: forced), not a census one:
  //: docs/internals/consumer_kernel.md#reg-budget-session9
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
//: the compiler: docs/internals/consumer_kernel.md#residency
constexpr int ctas_for_residency(size_t smem, int reg_budget) {
  const int by_smem = static_cast<int>(kSm.smem_per_sm / (smem + kDriverSmemReservePerCta));
  const int by_threads = kSm.max_threads_per_sm / kConsumerThreads;
  const int by_regs = kSm.regs_per_sm / (kConsumerThreads * reg_budget);
  const int m = by_smem < by_threads ? by_smem : by_threads;
  return m < by_regs ? m : by_regs;
}

//: MMA PRECISION IS ONE AXIS WITH TWO NAMED POINTS:
//: PRODUCTION = bf16 at BOTH sites, EXACTNESS = tf32x3 at BOTH sites. The split is
//: about what each arm is FOR -- the exactness arm exists so an oracle gate can see a
//: structural bug through the numerics, the production arm exists to ship. Why the
//: fold is bf16 in production, and the derived (not observed) state-deviation bound
//: it carries: docs/internals/consumer_kernel.md#mma-precision
constexpr MmaPrec kFoldPrecDefault = MmaPrec::kBf16;
constexpr MmaPrec kReadPrecDefault = MmaPrec::kBf16;

//: **THE BANK-CONFLICT PROPERTY, AS A COMPILE-TIME PROOF RATHER THAN A COMMENT.** An
//: MMA operand load's 32 lanes address SMEM as an affine function of the fragment's
//: two lane coordinates, `g = lane >> 2` and `r = lane & 3`; the access is
//: conflict-free exactly when those 32 addresses land on 32 DISTINCT banks. Enumerated
//: rather than asserted as a modular formula, so each operand proves its own pitch and
//: a future `BT`/`BC`/`DV` fails the build instead of silently costing the 4.06x
//: wavefront amplification the unpadded build measured.
//: `stride_g` / `stride_r` are the address deltas in 4-BYTE BANKS per unit `g`/`r`.
//: docs/internals/consumer_kernel.md#bank-predicates
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
//: docs/internals/consumer_kernel.md#bank-predicates
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
//: library's layout below: docs/internals/consumer_kernel.md#stage-layout
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
//: docs/internals/consumer_kernel.md#stage-offset
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
//: buffer. docs/internals/consumer_kernel.md#stage-layout
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
//: docs/internals/consumer_kernel.md#stage-format
template <MmaPrec READ_PREC, MmaPrec FOLD_PREC>
constexpr bool stage_is_narrow() {
  return READ_PREC == MmaPrec::kBf16 && FOLD_PREC == MmaPrec::kBf16;
}

//: `WT`'s row pitch, and the `+8` IS THE PROOF'S ANSWER rather than a copy of `Gm`'s:
//: `WT` serves GEMM 2's B map and GEMM 4's A map, and enumerated over
//: `pad in {4, 8, 12, 16}` at both `BT`, `BT + 8` is the UNIQUE pad that passes both.
//: A swizzle here is DECLINED on measured evidence (`WT` has six addressing sites, two
//: of them scalar row walks); the `ldmatrix` ATOMS stand, which this pitch admits.
//: docs/internals/consumer_kernel.md#stage-format
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
//: buffer with no row pitch at all: docs/internals/consumer_kernel.md#frag-order

//: 32-bit words in a fragment-ordered A operand contracting over `K`, FOR ONE `m16`
//: ROW BLOCK. A `BT`-token tile holds `NM = BT / 16` of them, block-major at a FIXED
//: stride of `frag_a_words(BC)` -- fixed rather than packed to the compacted `Kpad`, so
//: a block's base is a compile-time multiple and the MMA's `ks` offset is unchanged.
template <MmaPrec PREC>
constexpr int frag_a_words(int K) {
  return (PREC == MmaPrec::kBf16 ? K / 16 : K / 8) * 128;
}

//: The (m, k) that fragment slot `s` holds, and how many elements it packs.
template <MmaPrec PREC>
__device__ __forceinline__ void frag_a_slot(int s, int& m, int& k) {
  const int ks = s >> 7;
  const int rem = s & 127;
  const int lane = rem >> 2, w = rem & 3;
  m = (lane >> 2) + ((w & 1) << 3);
  const int r = lane & 3;
  k = (PREC == MmaPrec::kBf16) ? (16 * ks + ((w >> 1) << 3) + 2 * r)
                               : (8 * ks + ((w >> 1) << 2) + r);
}

//: THE `WT` WINDOW'S ROW COUNT, RESOLVED ONCE (§8 placeholder `C_WT_WINDOW`).
//: `wt_window_rows` is THE definition -- `consumer_smem_bytes` sizes the buffer from it
//: and the kernel body indexes with it, so the allocation and the addressing cannot
//: disagree about how many rows exist. `0` (and any out-of-range value) means `QPO`,
//: the whole plane. `DECAY` PINS THE PLANE, and it is a correctness pin:
//: docs/internals/consumer_kernel.md#wt-window
constexpr int wt_window_atoms(int BC, bool DECAY) {
  const int qpo = BC / kMmaKQuantum;
  const int req = (ROLA_C_WT_WINDOW <= 0 || ROLA_C_WT_WINDOW > qpo) ? qpo : ROLA_C_WT_WINDOW;
  return DECAY ? qpo : req;
}
constexpr int wt_window_rows(int BC, bool DECAY) {
  return wt_window_atoms(BC, DECAY) * kMmaKQuantum;
}

//: THE ITEM-STREAM CONTROL BLOCK -- named here because `consumer_smem_bytes` has to
//: account for it and the kernel body has to index it. `kCtlReadDone` gates the fold
//: (which WRITES `S`) on the readout (which READ it); `kCtlGramDone` gates phase C on
//: the Gram; `kCtlFold0 + w` is window `w`'s fold item-claim cursor, ONE PER WINDOW so
//: no cursor has to be reset mid-tile. What each counter means and the deadlock
//: argument that rests on them: docs/internals/consumer_kernel.md#item-stream
constexpr int kCtlReadDone = 0;
constexpr int kCtlGramDone = 1;
constexpr int kCtlFold0 = 2;
constexpr int ctl_words(int BC) { return kCtlFold0 + BC / kMmaKQuantum; }

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

//: THE `Vw` / `clk` UNION, AND WHY IT IS SIZED IN **BYTES** RATHER THAN ELEMENTS.
//: `clk` is DEAD before `Vw` is live, so it aliases `Vw`'s storage and the region is
//: sized as the MAX OVER ITS LIVE TENANTS. Taken over PITCHES that max is only sound
//: while the two tenants share an element size -- once `Vw` is bf16 and `clk` fp32 it
//: under-allocates by 2x, SILENTLY. `Vw` is still ADDRESSED at its own pitch; only the
//: RESERVATION is the max, and on the DECAY arm `clk` is the larger tenant at every
//: shipped shape: docs/internals/consumer_kernel.md#vw-clk-union
template <int BT, int BC, int DV, bool DECAY, MmaPrec READ_PREC, MmaPrec FOLD_PREC>
constexpr size_t vw_union_bytes() {
  constexpr bool kNarrow = stage_is_narrow<READ_PREC, FOLD_PREC>();
  const size_t vw = static_cast<size_t>(vw_elems<BT, DV, kNarrow>())
                    * (kNarrow ? sizeof(__nv_bfloat16) : sizeof(float));
  const size_t clk =
      DECAY ? static_cast<size_t>(BT) * (BC + 4) * sizeof(float) : static_cast<size_t>(0);
  return align4(vw > clk ? vw : clk);
}

//: THE STAGING SET: the SMEM objects the tile's STAGING phase PRODUCES and its COMPUTE
//: phase CONSUMES -- `Rf`, `Vw`, the two row scales, and under `COMPACT` the gathered
//: list. `Gm`, `WT` and `den` are deliberately NOT in it, and both exclusions are
//: structural rather than budgetary: docs/internals/consumer_kernel.md#staging-set
template <int BT, int BC, int DV, MmaPrec READ_PREC, MmaPrec FOLD_PREC, bool DECAY, bool COMPACT>
constexpr size_t staging_slot_bytes() {
  size_t b = 2 * static_cast<size_t>(BT) * sizeof(float);          // sr, sw
  b += vw_union_bytes<BT, BC, DV, DECAY, READ_PREC, FOLD_PREC>();  // Vw (+ clk)
  b += static_cast<size_t>(BT / 16) * frag_a_words<READ_PREC>(BC) * sizeof(unsigned);
  if (COMPACT) b += (2 * static_cast<size_t>(BC) + 2) * sizeof(int);
  return b;
}

//: The per-CTA shared-memory footprint, `constexpr` so the launch bound is DERIVED
//: from it rather than restated next to it. Every term, its measured residency
//: consequence and the three per-arm carves (`DECAY`'s three buffers rather than four,
//: `COMPACT`'s 520 B and the fourth buffer the design budgeted that this kernel does
//: not need, `ctl`'s 544 B of total slack on the tightest arm):
//: docs/internals/consumer_kernel.md#smem-ledger
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
//: sees: docs/internals/consumer_kernel.md#ring-depth
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

//: One `[16 x 8]` tile against a fragment-ordered A. `d += Af * B`.
template <int KMAX, typename TB, bool B_T, bool B_GATHER, MmaPrec PREC>
__device__ __forceinline__ void mma_tile_fragA(const unsigned* __restrict__ Af,
                                               const TB* __restrict__ B, int ldb,
                                               const int* __restrict__ b_row, int K, int n0,
                                               int lane, float& d0, float& d1, float& d2,
                                               float& d3);

__device__ __forceinline__ float mma_src(float x) { return x; }
//: Store a computed fp32 value into a buffer whose element type follows a precision
//: axis. Overloaded rather than `static_cast`, because the torch build defines
//: `__CUDA_NO_BFLOAT16_CONVERSIONS__` -- the narrowing must be the explicit intrinsic.
//: docs/internals/consumer_kernel.md#stage-format
__device__ __forceinline__ void gm_put(float& d, float x) { d = x; }
__device__ __forceinline__ void gm_put(__nv_bfloat16& d, float x) { d = __float2bfloat16(x); }
__device__ __forceinline__ float mma_src(__nv_bfloat16 x) { return __bfloat162float(x); }
//: Store a STAGED OPERAND. On the fp32 arm this is the assignment it always was; on the
//: narrow arm it is where §12 rule 3's format guard now lives, because it is where the
//: fp32 value last exists -- a MOVE of the clamp, not a second one, and what makes the
//: `ldmatrix` fetch bit-identical to the generic path.
__device__ __forceinline__ void stage_put(float& d, float x) { d = x; }
__device__ __forceinline__ void stage_put(__nv_bfloat16& d, float x) {
  d = __float2bfloat16(clamp_bf16_finite(x));
}

//: `D[16x8] += A[16 x K] * B[K x 8]`, fp32 accumulate, one warp. `TA`/`TB` are the
//: operands' SMEM STORAGE types and are independent of `PREC`; `A_T`/`B_T` select
//: their storage ORIENTATION, because the four GEMMs of a tile need three different
//: ones and a transposed COPY of a `[BT x BC]` panel would be pure SMEM traffic.
//: Fragment index maps are the PTX `mma.sync.aligned` layouts taken VERBATIM from
//: `readout_mma.cuh` rather than re-derived. §12 rule 3's clamp applies on the bf16 arm
//: only (tf32 carries fp32's 8-bit exponent, so a clamp there is dead code).
//: docs/internals/consumer_kernel.md#mma-tile
template <typename TA, bool A_T, typename TB, bool B_T, MmaPrec PREC>
__device__ __forceinline__ void mma_tile(const TA* __restrict__ A, int lda,
                                         const TB* __restrict__ B, int ldb, int K, int m0, int n0,
                                         int lane, float& d0, float& d1, float& d2, float& d3) {
  const int g = lane >> 2;
  const int r = lane & 3;
  auto a_at = [&](int m, int k) { return mma_src(A_T ? A[k * lda + m] : A[m * lda + k]); };
  auto b_at = [&](int k, int n) { return mma_src(B_T ? B[n * ldb + k] : B[k * ldb + n]); };
  const int ra = m0 + g;
  const int nb = n0 + g;
  if (PREC == MmaPrec::kTf32x3) {
    // m16n8k8. A: a0..a3 -> (g, r), (g+8, r), (g, r+4), (g+8, r+4).
    //          B: b0, b1 -> (row r, col g), (row r+4, col g).
    for (int k0 = 0; k0 < K; k0 += 8) {
      mma::mma_m16n8k8_tf32x3(d0, d1, d2, d3, a_at(ra, k0 + r), a_at(ra + 8, k0 + r),
                              a_at(ra, k0 + r + 4), a_at(ra + 8, k0 + r + 4), b_at(k0 + r, nb),
                              b_at(k0 + r + 4, nb), d0, d1, d2, d3);
    }
  } else {
    // m16n8k16. A: a0..a3 pack (g, 2r|2r+1), (g+8, 2r|2r+1),
    //                          (g, 2r+8|2r+9), (g+8, 2r+8|2r+9).
    //           B: b0, b1 pack (row 2r|2r+1, col g), (row 2r+8|2r+9, col g).
    auto cl = clamp_bf16_finite;
    for (int k0 = 0; k0 < K; k0 += 16) {
      const int ca = k0 + r * 2;
      const unsigned a0 = pack_bf16x2(cl(a_at(ra, ca)), cl(a_at(ra, ca + 1)));
      const unsigned a1 = pack_bf16x2(cl(a_at(ra + 8, ca)), cl(a_at(ra + 8, ca + 1)));
      const unsigned a2 = pack_bf16x2(cl(a_at(ra, ca + 8)), cl(a_at(ra, ca + 9)));
      const unsigned a3 = pack_bf16x2(cl(a_at(ra + 8, ca + 8)), cl(a_at(ra + 8, ca + 9)));
      const unsigned b0 = pack_bf16x2(cl(b_at(ca, nb)), cl(b_at(ca + 1, nb)));
      const unsigned b1 = pack_bf16x2(cl(b_at(ca + 8, nb)), cl(b_at(ca + 9, nb)));
      mma_m16n8k16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
    }
  }
}

//: THE SAME `[16 x 8]` TILE WITH `Vw` AS B, FETCHED BY ONE `ldmatrix`.
//: `Vw` is B for BOTH of its consumers (GEMM 3 and GEMM 4) and both want the same
//: 16(k) x 8(n) block per k-step; `.trans` is right because `Vw` is stored `k`-major
//: while the fragment wants `k`-adjacent pairs at a fixed `n`. It is BIT-IDENTICAL to
//: the generic path, not merely equivalent -- `ldmatrix` delivers the STORED patterns
//: and `stage_put`'s clamp already guarantees they are finite bf16, so the composition
//: is the identity and no summation is reordered. The fragment correspondence is
//: `static_assert`ed against CuTe's own maps below, never assumed.
//: docs/internals/consumer_kernel.md#ldsm-atoms
template <int BT, int DV>
__device__ __forceinline__ void vw_frag_b(const __nv_bfloat16* __restrict__ Vw, int k0, int n0,
                                          int lane, unsigned& b0, unsigned& b1) {
  const __nv_bfloat16* src = Vw + vw_index<BT, DV, true>(k0 + (lane & 15), n0);
  cute::SM75_U16x4_LDSM_T::copy(*reinterpret_cast<const cute::uint128_t*>(src), b0, b1);
}

//: `WT` AS GEMM 2's B. The SAME atom against the SAME algebra: `WT` is
//: stored leaf-major and the fragment wants `k`-adjacent pairs at a fixed `n`, which is
//: `.trans` again. The only difference from `Vw` is which axis is called what.
template <int ROWS, int BT>
__device__ __forceinline__ void wt_frag_b(const __nv_bfloat16* __restrict__ WT, int k0, int n0,
                                          int lane, unsigned& b0, unsigned& b1) {
  const __nv_bfloat16* src = WT + wt_index<BT, true>(k0 + (lane & 15), n0);
  cute::SM75_U16x4_LDSM_T::copy(*reinterpret_cast<const cute::uint128_t*>(src), b0, b1);
}

//: `WT` AS GEMM 4's A, the atom's OTHER form: here `WT` is `k`-ADJACENT,
//: which is what `ldmatrix` reads WITHOUT transposing, so the fold's A operand is one
//: `ldmatrix.x4` for the whole `[16 m x 16 k]` block. THE ADDRESSING IS THE ATOM'S OWN
//: -- `SM75_U32x4_LDSM_N`'s source tile is FOUR 8x8 MATRICES STACKED, not a 16x16 block
//: -- and the `(m, k)` reading of those four is the CALLER's, checked over all 256
//: slots by `ldsm_n_dst_map_matches`: docs/internals/consumer_kernel.md#ldsm-atoms
template <int ROWS, int BT>
__device__ __forceinline__ void wt_frag_a(const __nv_bfloat16* __restrict__ WT, int m0, int k0,
                                          int lane, unsigned& a0, unsigned& a1, unsigned& a2,
                                          unsigned& a3) {
  const int row = m0 + 8 * ((lane >> 3) & 1) + (lane & 7);
  const int col = k0 + 8 * (lane >> 4);
  const __nv_bfloat16* src = WT + wt_index<BT, true>(row, col);
  cute::SM75_U32x4_LDSM_N::copy(*reinterpret_cast<const cute::uint128_t*>(src), a0, a1, a2, a3);
}

//: GEMM 4, THE FOLD, WITH BOTH OPERANDS FETCHED BY `ldmatrix`:
//: `S += WT^T . Vw`, A = `WT` (`.N`), B = `Vw` (`.trans`). NARROW ARM ONLY.
template <int ROWS, int BT, int DV>
__device__ __forceinline__ void mma_fold_tile(const __nv_bfloat16* __restrict__ WT,
                                              const __nv_bfloat16* __restrict__ Vw, int m0, int n0,
                                              int lane, float& d0, float& d1, float& d2,
                                              float& d3) {
#pragma unroll
  for (int k0 = 0; k0 < BT; k0 += 16) {
    unsigned a0, a1, a2, a3, b0, b1;
    wt_frag_a<ROWS, BT>(WT, m0, k0, lane, a0, a1, a2, a3);
    vw_frag_b<BT, DV>(Vw, k0, n0, lane, b0, b1);
    mma_m16n8k16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
  }
}

//: `d += A . Vw` for a generic (indexed) A and `Vw` as B -- GEMM 3. NARROW ARM ONLY:
//: the 16-bit atoms have no fp32 form, and `kNarrowStage` is precisely the arm on which
//: `Vw` is 16-bit. `K` is `BT` at both call sites, trip count compile-time.
template <typename TA, int BT, int DV>
__device__ __forceinline__ void mma_tile_vwB(const TA* __restrict__ A, int lda,
                                             const __nv_bfloat16* __restrict__ Vw, int m0, int n0,
                                             int lane, float& d0, float& d1, float& d2, float& d3) {
  const int g = lane >> 2;
  const int r = lane & 3;
  const int ra = m0 + g;
  auto a_at = [&](int m, int k) { return mma_src(A[m * lda + k]); };
  auto cl = clamp_bf16_finite;
#pragma unroll
  for (int k0 = 0; k0 < BT; k0 += 16) {
    const int ca = k0 + r * 2;
    const unsigned a0 = pack_bf16x2(cl(a_at(ra, ca)), cl(a_at(ra, ca + 1)));
    const unsigned a1 = pack_bf16x2(cl(a_at(ra + 8, ca)), cl(a_at(ra + 8, ca + 1)));
    const unsigned a2 = pack_bf16x2(cl(a_at(ra, ca + 8)), cl(a_at(ra, ca + 9)));
    const unsigned a3 = pack_bf16x2(cl(a_at(ra + 8, ca + 8)), cl(a_at(ra + 8, ca + 9)));
    unsigned b0, b1;
    vw_frag_b<BT, DV>(Vw, k0, n0, lane, b0, b1);
    mma_m16n8k16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
  }
}

//: THE TWO LIBRARY MAPS THIS FILE DEPENDS ON, ENUMERATED AT COMPILE TIME. Both must
//: equal `k = 2r + (v & 1) + 8 * (v >> 1)`, `n = g` -- which is `mma_tile`'s `b0`/`b1`,
//: letter for letter, so a library bump that moved either fails the BUILD rather than
//: the oracle: docs/internals/consumer_kernel.md#cute-crosschecks
constexpr bool mma_b_map_matches() {
  using B = typename cute::MMA_Traits<cute::SM80_16x8x16_F32BF16BF16F32_TN>::BLayout;
  for (int lane = 0; lane < 32; ++lane) {
    for (int v = 0; v < 4; ++v) {
      const int off = B{}(lane, v);
      if (off % 8 != (lane >> 2)) return false;  // n = g
      if (off / 8 != 2 * (lane & 3) + (v & 1) + 8 * (v >> 1)) return false;
    }
  }
  return true;
}

constexpr bool ldsm_t_dst_map_matches() {
  using Dst = typename cute::Copy_Traits<cute::SM75_U16x4_LDSM_T>::DstLayout;
  for (int lane = 0; lane < 32; ++lane) {
    for (int v = 0; v < 4; ++v) {
      const int e = Dst{}(lane, v * 16) / 16;  // bit -> half
      if (e % 8 != (lane >> 2)) return false;  // n = g
      if (e / 8 != 2 * (lane & 3) + (v & 1) + 8 * (v >> 1)) return false;
    }
  }
  return true;
}

constexpr bool ldsm_t_src_map_matches() {
  using Src = typename cute::Copy_Traits<cute::SM75_U16x4_LDSM_T>::SrcLayout;
  for (int lane = 0; lane < 32; ++lane) {
    if (Src{}(lane, 0) / 128 != (lane & 15)) return false;  // row = lane & 15
  }
  return true;
}

//: THE SAME QUESTION FOR THE `.N` ATOM AND THE A FRAGMENT: under the
//: `(m, k)` reading `wt_frag_a` implements, the destination map must equal `mma_tile`'s
//: `a0..a3` and the source map must be `row = lane`.
constexpr bool ldsm_n_dst_map_matches() {
  using Dst = typename cute::Copy_Traits<cute::SM75_U32x4_LDSM_N>::DstLayout;
  for (int lane = 0; lane < 32; ++lane) {
    for (int v = 0; v < 8; ++v) {  // 8 halves
      const int e = Dst{}(lane, v * 16) / 16;
      const int srow = e / 8, scol = e % 8, i = srow / 8;
      const int m_cute = 8 * (i & 1) + srow % 8, k_cute = 8 * (i >> 1) + scol;
      if (m_cute != (lane >> 2) + 8 * ((v / 2) & 1)) return false;
      if (k_cute != 2 * (lane & 3) + v % 2 + 8 * ((v / 2) >> 1)) return false;
    }
  }
  return true;
}

constexpr bool ldsm_n_src_map_matches() {
  using Src = typename cute::Copy_Traits<cute::SM75_U32x4_LDSM_N>::SrcLayout;
  for (int lane = 0; lane < 32; ++lane) {
    if (Src{}(lane, 0) / 128 != lane) return false;  // row = lane
  }
  return true;
}

static_assert(mma_b_map_matches(), "CuTe's B fragment map is not mma_tile's");
static_assert(ldsm_t_dst_map_matches(), "ldmatrix.x2.trans does not deliver b0/b1");
static_assert(ldsm_t_src_map_matches(), "ldmatrix.x2.trans row addresses moved");
static_assert(ldsm_n_dst_map_matches(), "ldmatrix.x4 does not deliver a0..a3");
static_assert(ldsm_n_src_map_matches(), "ldmatrix.x4 row addresses moved");

//: The `[16 x 8]` D/C fragment's element map, shared by every writeback below:
//:   `d0, d1 -> (row g,     col 2r, 2r+1)`
//:   `d2, d3 -> (row g + 8, col 2r, 2r+1)`
//: `d += Af . B` with A FRAGMENT-ORDERED (one `uint4` per k-step, zero address
//: arithmetic); B keeps the generic indexed path because each warp wants a different
//: `n`-slice. `B_GATHER` is COMPACTION's readout indirection -- a TEMPLATE flag rather
//: than a `b_row != nullptr` test, so the non-compacted arm carries no load.
//: **THE K-LOOP KEEPS A COMPILE-TIME TRIP COUNT AND AN EARLY EXIT, never a runtime
//: bound**: docs/internals/consumer_kernel.md#mma-tile-fraga
template <int KMAX, typename TB, bool B_T, bool B_GATHER, MmaPrec PREC>
__device__ __forceinline__ void mma_tile_fragA(const unsigned* __restrict__ Af,
                                               const TB* __restrict__ B, int ldb,
                                               const int* __restrict__ b_row, int K, int n0,
                                               int lane, float& d0, float& d1, float& d2,
                                               float& d3) {
  const int g = lane >> 2;
  const int r = lane & 3;
  const int nb = n0 + g;
  auto b_at = [&](int k, int n) {
    return mma_src(B_GATHER ? B[b_row[k] * ldb + n] : (B_T ? B[n * ldb + k] : B[k * ldb + n]));
  };
  constexpr int kStep = (PREC == MmaPrec::kBf16) ? 16 : 8;
#pragma unroll
  for (int ks = 0; ks < KMAX / kStep; ++ks) {
    if (ks * kStep >= K) break;
    const uint4 a = *reinterpret_cast<const uint4*>(Af + ks * 128 + lane * 4);
    if (PREC == MmaPrec::kTf32x3) {
      mma::mma_m16n8k8_tf32x3(d0, d1, d2, d3, __uint_as_float(a.x), __uint_as_float(a.y),
                              __uint_as_float(a.z), __uint_as_float(a.w), b_at(ks * 8 + r, nb),
                              b_at(ks * 8 + r + 4, nb), d0, d1, d2, d3);
    } else {
      const int ca = ks * 16 + r * 2;
      const unsigned b0 =
          pack_bf16x2(clamp_bf16_finite(b_at(ca, nb)), clamp_bf16_finite(b_at(ca + 1, nb)));
      const unsigned b1 =
          pack_bf16x2(clamp_bf16_finite(b_at(ca + 8, nb)), clamp_bf16_finite(b_at(ca + 9, nb)));
      mma_m16n8k16(d0, d1, d2, d3, a.x, a.y, a.z, a.w, b0, b1, d0, d1, d2, d3);
    }
  }
}

//: GEMM 2 WITH A FRAGMENT-ORDERED A AND `WT` FETCHED BY `ldmatrix`.
//: `KMAX` is `WT`'s WINDOW row count and `K` the realized (causal, compacted) prefix,
//: so the early exit, its compile-time trip count and the k-order are the generic
//: path's, unchanged. NARROW ARM ONLY, for `vw_frag_b`'s reason. `KMAX % 16 == 0` is
//: what makes the fetch legal -- `ldmatrix` reads a WHOLE 16-row block.
template <int KMAX, int BT>
__device__ __forceinline__ void mma_tile_fragA_wtB(const unsigned* __restrict__ Af,
                                                   const __nv_bfloat16* __restrict__ WT, int K,
                                                   int n0, int lane, float& d0, float& d1,
                                                   float& d2, float& d3) {
  static_assert(KMAX % 16 == 0, "ldmatrix reads whole 16-row blocks of WT");
#pragma unroll
  for (int ks = 0; ks < KMAX / 16; ++ks) {
    if (ks * 16 >= K) break;
    const uint4 a = *reinterpret_cast<const uint4*>(Af + ks * 128 + lane * 4);
    unsigned b0, b1;
    wt_frag_b<KMAX, BT>(WT, ks * 16, n0, lane, b0, b1);
    mma_m16n8k16(d0, d1, d2, d3, a.x, a.y, a.z, a.w, b0, b1, d0, d1, d2, d3);
  }
}

//: Read element `(m, k)` back OUT of a fragment-ordered A operand. The mass column's
//: denominator is a `[BT x BC] x [BC x 1]` matvec, one column out of `DV + 1`, and an
//: `n8` MMA tile would waste 7/8 of a fragment -- so it stays a scalar reduction and
//: pays this inverse instead, ~10 integer ops per element once per tile.
template <MmaPrec PREC>
__device__ __forceinline__ float frag_a_get(const unsigned* Af, int m, int k) {
  //: HAZARD frag-a-get -- docs/internals/consumer_kernel.md#frag-a-get
  const int g = m & 7;
  const int w_lo = (m >> 3) & 1;
  if (PREC == MmaPrec::kBf16) {
    const int ks = k >> 4, kk = k & 15;
    const int lane = (g << 2) | ((kk & 7) >> 1);
    const int w = ((kk >= 8) << 1) | w_lo;
    const unsigned word = Af[ks * 128 + lane * 4 + w];
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&word);
    return __bfloat162float(h[kk & 1]);
  }
  const int ks = k >> 3, kk = k & 7;
  const int lane = (g << 2) | (kk & 3);
  const int w = ((kk >= 4) << 1) | w_lo;
  return __uint_as_float(Af[ks * 128 + lane * 4 + w]);
}

#define CONSUMER_D_ROWS(lane) const int _g = (lane) >> 2, _r = (lane) & 3

//: THE TILE'S INTRA-CTA PHASE ORDERING IS CARRIED BY COUNTERS RATHER THAN BY
//: `__syncthreads()`. The point is not to spell a barrier differently: a warp
//: that reaches a phase edge early goes and executes another WORK ITEM instead of
//: stalling on a rendezvous, which is what makes the fold a fluid pool. A WORK ITEM is
//: one `(GEMM, warp-tile)` pair with a WHOLE `k` extent, and full-`k` is a BIT-IDENTITY
//: requirement rather than a convenience. The item table is DERIVED, never stored.
//: The three counters, the release/acquire discipline and the deadlock argument that
//: rests on them: docs/internals/consumer_kernel.md#item-stream
__device__ __forceinline__ void retire_item(int* counter, int lane) {
  //: HAZARD release-acquire -- docs/internals/consumer_kernel.md#release-acquire
  __threadfence_block();
  if (lane == 0) atomicAdd(counter, 1);
}

__device__ __forceinline__ bool eligible(const volatile int* counter, int target) {
  //: HAZARD release-acquire -- docs/internals/consumer_kernel.md#release-acquire
  const bool ok = (*counter >= target);
  if (ok) __threadfence_block();
  return ok;
}

//: The spin, written in terms of `eligible` so the release/acquire discipline lives in
//: exactly TWO functions and can be audited there rather than at every phase edge.
//: `__nanosleep` backs the polling warp off the issue port.
__device__ __forceinline__ void await_count(const volatile int* counter, int target) {
  while (!eligible(counter, target)) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    __nanosleep(32);
#endif
  }
}

//: **THE FORCING-CENSUS INSTRUMENT, NEVER A SHIPPING KNOB.** When
//: defined it replaces the DERIVED launch bound with a fixed CTA count on every
//: instantiation, so a candidate register budget can be censused over the whole
//: dispatch matrix in one probe compile. `tools/ratify.py --census` is its only
//: legitimate definer and a manifest is only ever ratified off the UNFORCED compile;
//: forcing moves the REGISTER side only: docs/internals/consumer_kernel.md#force-ctas
#ifdef ROLA_FORCE_CTAS
#define ROLA_CONSUMER_CTAS_EXPR (ROLA_FORCE_CTAS)
#else
#define ROLA_CONSUMER_CTAS_EXPR                                                     \
  ctas_for_residency(consumer_smem_bytes<BT, BC, DV, D, READ_PREC, FOLD_PREC,       \
                                         DECAY, GLOBAL, COMPACT>(),                 \
                     consumer_reg_budget(BT, BC, DECAY, READ_PREC, FOLD_PREC, DV))
#endif
template <int BT, int BC, int DV, int D, MmaPrec FOLD_PREC, MmaPrec READ_PREC, bool DECAY,
          bool GLOBAL, bool COMPACT>
__global__ __launch_bounds__(kConsumerThreads, ROLA_CONSUMER_CTAS_EXPR) void
consumer_forward_kernel(ConsumerParams p) {
  //: `SP = DV + 4` IS DERIVED FROM THE bf16 MMA B MAP. The tf32 map (`k = k0 + r`)
  //: alone would ask for `DV + 1`, which COLLIDES up to four ways under bf16's
  //: `k = k0 + 2r` -- the production default. `DV + 4` makes the bank `(8r + g) % 32`,
  //: covering all 32 exactly once.
  //: docs/internals/consumer_kernel.md#state-pitch
  constexpr int SP = DV + 4;
  constexpr int NT_DV = DV / 8;  // n8 column tiles of a DV-wide output
  //: THE TILE'S OUTPUT IS `NM x NT_DV` WARP TILES, AND THAT PRODUCT -- not `DV` alone
  //: -- IS WHAT THE 8 WARPS PARTITION: docs/internals/consumer_kernel.md#warp-partition
  constexpr int NM = BT / 16;  // m16 row blocks of a BT-token tile
  extern __shared__ char smem_raw[];

  // ---- THE SMEM CARVE. Every region, its pitch, its measured residency cost and the
  // ---- 4.06x wavefront amplification the pads exist to prevent:
  // ---- docs/internals/consumer_kernel.md#smem-ledger
  //: THE STAGED OPERANDS' ELEMENT TYPE AND PITCHES. `StageT` is the format the MMA
  //: itself wants on the production arm, so the narrowing happens once at the store
  //: instead of once per fragment slot.
  constexpr bool kNarrowStage = stage_is_narrow<READ_PREC, FOLD_PREC>();
  using StageT = typename cuda::std::conditional<kNarrowStage, __nv_bfloat16, float>::type;
  //: `BTP` is `WT`'s ELEMENTS PER ROW -- the derived `BT + 8` pad on the narrow arm and
  //: `BT + 4` on the `tf32x3` one, each proven by the predicate its format makes correct.
  constexpr int BTP = wt_pitch<BT, READ_PREC, FOLD_PREC>();  // WT, LEAF-MAJOR
  //: `WT`'s ROW COUNT, from the same function the carve and `consumer_smem_bytes` take
  //: it from, so layout, allocation and addressing cannot disagree.
  constexpr int kWtRows = wt_window_rows(BC, DECAY);
  //: `Vw`'s ADDRESSING IS NOT A PITCH ON THE NARROW ARM -- it is `StageLayout`. `VWP`
  //: is the `tf32x3` arm's row-major pitch alone; every access goes through `vw_index`,
  //: so the two arms differ in ONE place.
  constexpr int VWP = DV + 8;  // the `tf32x3` arm's row pitch
  //: `Gm`'s STORAGE TYPE FOLLOWS `READ_PREC`, and that is a CORRECTNESS requirement
  //: rather than a nicety -- it is an INTERMEDIATE of the `y` path, so a bf16 `Gm` on
  //: the tf32x3 arm quantizes a value the exactness arm exists to keep exact (the
  //: oracle gate caught exactly that): docs/internals/consumer_kernel.md#gm-type
  using GmT =
      typename cuda::std::conditional<READ_PREC == MmaPrec::kBf16, __nv_bfloat16, float>::type;
  constexpr int GMP = (READ_PREC == MmaPrec::kBf16) ? BT + 8 : BT + 4;
  constexpr int kRing =
      consumer_ring_depth<BT, BC, DV, D, READ_PREC, FOLD_PREC, DECAY, GLOBAL, COMPACT>();
  //: D2.3, AS A CORRECTNESS ASSERT AND NOT AS A FOOTPRINT ONE: `clk` aliases `Vw`'s
  //: bytes, so a ringed `Vw` under `DECAY` would put slot `s + 1`'s clock prefix on top
  //: of slot `s`'s value tile while the fold is still reading it. The derivation
  //: returns 1 for exactly this reason; this states it so a future headroom change
  //: cannot quietly turn the ring on for the wrong arm.
  static_assert(!DECAY || kRing == 1,
                "DECAY excludes Vw -- and hence the whole staging set -- from the "
                "ring: clk aliases Vw's bytes and both panel builds read clk");
  static_assert(kRing == 1 || kRing == 2, "the run-ahead is one tile deep");
  float* S = reinterpret_cast<float*>(smem_raw);
  char* carve = smem_raw + align4(static_cast<size_t>(BC) * SP * sizeof(float));
  StageT* WT = reinterpret_cast<StageT*>(carve);
  carve += align4(static_cast<size_t>(wt_window_rows(BC, DECAY)) * BTP * sizeof(StageT));
  constexpr size_t kVwSlotBytes = vw_union_bytes<BT, BC, DV, DECAY, READ_PREC, FOLD_PREC>();
  constexpr int kVwSlot = static_cast<int>(kVwSlotBytes / sizeof(StageT));
  //: `ldmatrix` TAKES A 16-BYTE-ALIGNED ROW ADDRESS, so every slot base must be 16-byte
  //: aligned -- a requirement the `align4` carve does NOT imply and which
  //: `stage_ldsm_banks_ok` can only check RELATIVE to the base. Both terms are stated
  //: rather than trusted: docs/internals/consumer_kernel.md#ldsm-alignment
  constexpr size_t kWtOffsetBytes = align4(static_cast<size_t>(BC) * SP * sizeof(float));
  constexpr size_t kVwOffsetBytes =
      kWtOffsetBytes + align4(static_cast<size_t>(kWtRows) * BTP * sizeof(StageT));
  static_assert(!kNarrowStage || (kVwOffsetBytes % 16 == 0 && kVwSlotBytes % 16 == 0),
                "Vw must be 16-byte aligned at every ring slot for ldmatrix");
  static_assert(!kNarrowStage || kWtOffsetBytes % 16 == 0,
                "WT must be 16-byte aligned for ldmatrix");
  StageT* Vw_r0 = reinterpret_cast<StageT*>(carve);
  //: `clk` ALIASES `Vw`'s bytes. Declared AT THE CARVE rather than later as
  //: `clk = Vw`, because with two element types that assignment does not type-check
  //: and the alias is the load-bearing fact. It names SLOT 0 and needs no ring index:
  //: `DECAY` pins `kRing == 1`.
  float* clk = reinterpret_cast<float*>(carve);
  carve += kRing * kVwSlotBytes;
  float* sr_r0 = reinterpret_cast<float*>(carve);
  float* den = sr_r0 + kRing * 2 * BT;
  uint32_t* cand_read = reinterpret_cast<uint32_t*>(den + (GLOBAL ? BT : 0));
  uint32_t* cand_write = cand_read + kMaxWords;
  int* dig = reinterpret_cast<int*>(cand_write + kMaxWords);  // [D][BC]
  constexpr int kRfBlock = frag_a_words<READ_PREC>(BC);
  constexpr int kRfWords = NM * kRfBlock;
  unsigned* Rf_r0 = reinterpret_cast<unsigned*>(dig + D * BC);
  GmT* Gm = reinterpret_cast<GmT*>(Rf_r0 + kRing * kRfWords);
  float* log_keep =
      reinterpret_cast<float*>(Gm + (READ_PREC == MmaPrec::kBf16 ? BT * (BT + 8) : BT * (BT + 4)));
  float* p_ref = log_keep + (DECAY ? BC : 0);
  float* s0 = p_ref + (DECAY ? BC : 0);
  constexpr int CKP = BC + 4;  // clk's row pitch, token-major
  constexpr int kCSlot = COMPACT ? (2 * BC + 2) : 0;
  int* c_idx_r0 = reinterpret_cast<int*>(s0 + (DECAY ? BC : 0));
  int* ctl = c_idx_r0 + kRing * kCSlot;
  static_assert(!DECAY
                    || static_cast<size_t>(BT) * CKP * sizeof(float)
                           <= vw_union_bytes<BT, BC, DV, DECAY, READ_PREC, FOLD_PREC>(),
                "clk aliases the Vw region and must fit IN BYTES");
  float* a_state = reinterpret_cast<float*>(Rf_r0);         // [BC], the survival product A
  int* sum_slot = reinterpret_cast<int*>(Rf_r0) + BC;       // [BC]
  int* sum_count = reinterpret_cast<int*>(Rf_r0) + 2 * BC;  // [1]
  static_assert(frag_a_words<READ_PREC>(BC) >= 2 * BC + 1,
                "pass 1's summary state must fit inside the read-coefficient buffer");

  // ---- EACH OPERAND'S PITCH PROVES ITS OWN CONFLICT-FREEDOM. `stride_g`/`stride_r`
  // ---- are read straight off `mma_tile`'s fragment maps; the whole ledger, and which
  // ---- predicate each buffer's STORAGE FORMAT makes correct:
  // ---- docs/internals/consumer_kernel.md#pitches
  static_assert(banks_distinct(1, 2 * SP), "S: GEMM1 B operand conflicts");
  static_assert(kNarrowStage ? banks_distinct_u16(1, 2 * BTP) : banks_distinct(1, 2 * BTP),
                "WT: GEMM2 B operand conflicts");
  static_assert(banks_distinct(READ_PREC == MmaPrec::kBf16 ? GMP / 2 : GMP, 1),
                "Gm: GEMM3 A operand conflicts");
  static_assert(kNarrowStage ? banks_distinct_u16(BTP, 2) : banks_distinct(BTP, 1),
                "WT: GEMM4 A operand conflicts");
  static_assert(!kNarrowStage || pitched_ldsm_banks_ok<kWtRows, BT, BT + 8>(),
                "WT: the ldmatrix fetch conflicts or is misaligned at wt_pitch");
  static_assert(!kNarrowStage || stage_offset_matches_cute<BT, DV>(),
                "Vw: the closed-form address is not CuTe's layout");
  static_assert(!kNarrowStage || stage_layout_is_bijection<BT, DV>(),
                "Vw: the swizzled layout does not tile [0, BT*DV)");
  static_assert(!kNarrowStage || stage_ldsm_banks_ok<BT, DV>(),
                "Vw: the ldmatrix.x2.trans fetch conflicts or is misaligned");
  static_assert(!kNarrowStage || stage_store_banks_ok<BT, DV>(), "Vw: the staging store conflicts");
  static_assert(kNarrowStage || banks_distinct(1, VWP),
                "Vw: GEMM4 B operand conflicts (tf32x3 arm)");
  //: THE ONE OPERAND THAT IS NOT CONFLICT-FREE, AND THE TRADE IS EXPLICIT: on the
  //: `tf32x3` arm the fold and the intra GEMM read `Vw` under different `k` steps and
  //: NO single pitch satisfies both, so `Vw` is sized for the fold (~8x the operand
  //: traffic) and the intra term keeps a 2-way conflict. ON THE NARROW ARM THE TRADE
  //: DISAPPEARS -- both consumers issue the SAME `ldmatrix.x2.trans` against the SAME
  //: swizzled layout -- so this assert is the `tf32x3` arm's alone, and it fails the
  //: build if the note stops being true: docs/internals/consumer_kernel.md#vw-conflict-trade
  static_assert(kNarrowStage || !banks_distinct(1, 2 * VWP),
                "Vw/GEMM3: update this note if it changes");

  constexpr int kQPO = BC / kMmaKQuantum;
  static_assert(BC % kMmaKQuantum == 0, "BC must be a whole number of MMA k-quanta");
  constexpr int kWtWindow = wt_window_atoms(BC, DECAY);
  static_assert(kQPO % kWtWindow == 0,
                "C_WT_WINDOW must divide QPO -- the window loop's trip count is "
                "exact, never ragged (I1: an expression in BC and MMA_K_QUANTUM)");
  constexpr int kWinRows = wt_window_rows(BC, DECAY);
  constexpr int kNWin = kQPO / kWtWindow;
  constexpr bool kWtPlane = (kNWin == 1);
  static_assert(kWinRows == BC || !kWtPlane, "kWinRows must be BC exactly at the plane");

  const int work = blockIdx.x;
  const int bh = p.owners[2 * work];
  const int owner = p.owners[2 * work + 1];
  const int seg = (p.work_seg != nullptr) ? p.work_seg[work] : 0;
  const int nseg = (p.work_nseg != nullptr) ? p.work_nseg[work] : 1;
  const int owner_slot0 = (p.work_slot != nullptr) ? p.work_slot[work] : 0;
  const bool summary = (p.pass_mode == kPassSummary);
  const bool last_seg = (seg == nseg - 1);
  const int tile_lo = static_cast<int>((static_cast<int64_t>(seg) * p.n_tiles) / nseg);
  const int tile_hi = static_cast<int>((static_cast<int64_t>(seg + 1) * p.n_tiles) / nseg);
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  constexpr int kWarps = kConsumerThreads / 32;

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

  // ---- candidate tiles: the §4 word-AND chain, masked to this segment's tile range
  // ---- docs/internals/consumer_kernel.md#cta-entry
  for (int w = tid; w < p.n_words; w += kConsumerThreads) {
    const int64_t union_row = static_cast<int64_t>(bh) * p.n_slices + slice;
    uint32_t r = p.read_union[union_row * p.n_words + w];
    uint32_t wr = p.write_union[union_row * p.n_words + w];
    for (int l = 0; l < j; ++l) {
      const int64_t row = static_cast<int64_t>(bh) * p.total_rows + p.level_row_offset[l] + base[l];
      r &= p.read_bits[row * p.n_words + w];
      wr &= p.write_bits[row * p.n_words + w];
    }
    const uint32_t range = segment_word_mask(w, tile_lo, tile_hi);
    r &= range;
    wr &= range;
    cand_read[w] = summary ? 0u : r;
    cand_write[w] = wr;
  }

  for (int idx = tid; idx < D * BC; idx += kConsumerThreads) {
    const int l = idx / BC;
    dig[idx] = leaf_digit(p, l, base[l], idx - l * BC);
  }

  // ---- §12 RULE 2: per-leaf log-retention, ONCE PER OWNER, IN LOG SPACE. Both of the
  // ---- rule's prohibitions are STRUCTURAL here rather than checked, and the
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

  constexpr int cols = GLOBAL ? DV + 1 : DV;
  // ---- THE OWNER'S STATE BLOCK, READ FROM HBM ONCE (the one-copy invariant), at
  // ---- §4.5's ATOM granularity. Where this segment's entry state comes from, and why
  // ---- a SPLIT owner's pass-3 CTA never reads `p.state` (a correctness requirement,
  // ---- not a saving): docs/internals/consumer_kernel.md#entry-state
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
      sum_slot[i] = 0;    // the written-leaf FLAG until the compaction rewrites it
    }
  }
  __syncthreads();

  // ---- THE STAGING RING'S CANDIDATE WALK AND ITS TWO LAMBDAS ------------------
  // ---- docs/internals/consumer_kernel.md#candidate-walk
  auto next_cand = [&](int t) {
    for (; t < tile_hi; ++t) {
      const uint32_t cw = cand_read[t >> 5] | cand_write[t >> 5];
      if (((cw >> (t & 31)) & 1u) != 0u) break;
    }
    return t;
  };
  // ---- THE BUILD IS AN ATOM-RANGE FUNCTION, NOT A PLANE PASS: at the plane the
  // ---- range folds away and the emitted loop is the plane pass verbatim, which is what
  // ---- makes "the plane" a VALUE OF THE DIAL rather than a second code path (I4).
  //: HAZARD no-captured-arrays -- docs/internals/consumer_kernel.md#no-captured-arrays
  auto build_wt_rows = [&](int q_lo, int q_hi, auto&& amp_w_at, const int* cx, int Kv) {
    //: HAZARD wt-row-t-fastest -- docs/internals/consumer_kernel.md#wt-row-t-fastest
    //: HAZARD ballot-lane-run -- docs/internals/consumer_kernel.md#ballot-lane-run
    constexpr unsigned kRowLanes = (BT >= 32) ? 0xffffffffu : ((1u << BT) - 1u);
    static_assert(BT == 16 || BT == 32,
                  "the pass-1 emptiness ballot assumes the BT threads of one WT row "
                  "are a contiguous lane run inside one warp");
    for (int idx = tid; idx < BT * (q_hi - q_lo); idx += kConsumerThreads) {
      const int qr = idx / BT;
      const int q = q_lo + qr;
      const int t = idx - qr * BT;
      if (COMPACT && q >= Kv) {
        stage_put(WT[wt_index<BT, kNarrowStage>(qr, t)], 0.0f);
        continue;
      }
      const int i = COMPACT ? cx[q] : q;
      float a = 1.0f;
#pragma unroll
      for (int l = 0; l < D; ++l) {
        if (l < j) continue;
        a *= amp_w_at(l, t, i);
      }
      if (DECAY) a *= panel_write_factor(clk[t * CKP + i], p_ref[i], log_keep[i]);
      //: HAZARD summary-prequant -- docs/internals/consumer_kernel.md#summary-prequant
      if (summary && __any_sync(kRowLanes << (lane & ~(BT - 1)), a != 0.0f)
          && (lane & (BT - 1)) == 0) {
        sum_slot[i] = 1;
      }
      stage_put(WT[wt_index<BT, kNarrowStage>(qr, t)], a);
    }
  };

  // ---- THE STAGING, AS A FUNCTION OF (TILE, SLOT): everything the tile's COMPUTE
  // ---- consumes but does not produce. At `kRing == 1` it is called at the top of the
  // ---- tile body with `slot` identically 0; at `kRing == 2`, ONE VISIT EARLY, from
  // ---- inside the previous tile's compute.
  // ---- docs/internals/consumer_kernel.md#candidate-walk
  auto stage_tile = [&](int tile, int slot, bool is_read, bool is_write) {
    const int t0 = tile * BT;
    float* sr = sr_r0 + slot * 2 * BT;
    float* sw = sr + BT;
    unsigned* Rf = Rf_r0 + slot * kRfWords;
    StageT* Vw = Vw_r0 + slot * kVwSlot;
    int* c_idx = c_idx_r0 + slot * kCSlot;
    int* c_slot = c_idx + (COMPACT ? BC : 0);
    int* c_count = c_slot + (COMPACT ? BC : 0);

    const int64_t tile_row = (static_cast<int64_t>(bh) * p.n_tiles + tile) * BT;
    const float* amp_r[D];
    const float* amp_w[D];
#pragma unroll
    for (int l = 0; l < D; ++l) {
      amp_r[l] = p.amp_read[l] + tile_row * p.level_pitch_read[l];
      amp_w[l] = p.amp_write[l] + tile_row * p.level_pitch_write[l];
    }
    // ---- COMPACT: §3 step 2's SUPPORT-DRIVEN GATHER, three steps and none of them a
    // ---- branch on a density. THE SUPPORT WORDS NEVER REACH SHARED MEMORY -- warp 0
    // ---- computes them into REGISTERS and consumes them two instructions later, one
    // ---- fewer barrier than a staged support pair would need:
    // ---- docs/internals/consumer_kernel.md#compact-gather
    if (COMPACT && warp == 0) {
      const int64_t slab = (static_cast<int64_t>(bh) * p.n_tiles + tile) * p.total_rows;
      const int tile_tokens = p.T - t0;
      const uint32_t valid_tokens = (tile_tokens >= BT)
                                        ? ((BT == 32) ? 0xffffffffu : ((1u << BT) - 1u))
                                        : (tile_tokens <= 0 ? 0u : ((1u << tile_tokens) - 1u));
      uint32_t fr = is_read ? valid_tokens : 0u;
      uint32_t fw = is_write ? valid_tokens : 0u;
#pragma unroll
      for (int l = 0; l < D; ++l) {
        if (l >= j) break;
        const int64_t row = slab + p.level_row_offset[l] + base[l];
        if (is_read) fr &= p.read_tok_bits[row];
        if (is_write) fw &= p.write_tok_bits[row];
      }
      constexpr int kRounds = (BC + 31) / 32;
      uint32_t rwv[kRounds], wwv[kRounds];
#pragma unroll
      for (int b = 0; b < kRounds; ++b) {
        const int i = b * 32 + lane;
        uint32_t rw = (i < BC && is_read) ? fr : 0u;
        uint32_t ww = (i < BC && is_write) ? fw : 0u;
        if (i < BC) {
#pragma unroll
          for (int l = 0; l < D; ++l) {
            if (l < j) continue;  // the fixed levels are already in `fr`/`fw`
            const int64_t row = slab + p.level_row_offset[l] + dig[l * BC + i];
            if (is_read) rw &= p.read_tok_bits[row];
            if (is_write) ww &= p.write_tok_bits[row];
          }
        }
        rwv[b] = rw;
        wwv[b] = ww;
      }
      int k = 0;
#pragma unroll
      for (int pass = 0; pass < 2; ++pass) {
#pragma unroll
        for (int b = 0; b < kRounds; ++b) {
          const int i = b * 32 + lane;
          const bool in_range = i < BC;
          const uint32_t rw = rwv[b];
          const uint32_t ww = wwv[b];
          const bool active = (rw | ww) != 0u;
          const uint32_t first_write = ww & (0u - ww);
          const bool in_intra = (rw & (0u - first_write)) != 0u;
          const bool take = in_range && ((pass == 0) ? in_intra : (active && !in_intra));
          const unsigned mask = __ballot_sync(0xffffffffu, take);
          const int slot = k + __popc(mask & ((1u << lane) - 1u));
          if (take) {
            c_idx[slot] = i;
            c_slot[i] = slot;
          } else if (pass == 0 && in_range && !active) {
            c_slot[i] = -1;
          }
          k += __popc(mask);
        }
        if (pass == 0 && lane == 0) c_count[1] = k;  // KI
      }
      for (int q = k + lane; q < BC; q += 32) c_idx[q] = 0;
      if (lane == 0) c_count[0] = k;
    }
    if (COMPACT) __syncthreads();

    const int K = COMPACT ? c_count[0] : BC;
    constexpr int kUnit = (READ_PREC == MmaPrec::kTf32x3) ? 8 : 16;
    const int Kpad = ((K + kUnit - 1) / kUnit) * kUnit;  // readout k-extent
    const int KpadM = ((K + 15) / 16) * 16;              // fold m-extent
    const int wt_rows = COMPACT ? KpadM : BC;

    // ---- DECAY, step (a): the clock prefix, `P_ref`, `s0` -- INTO `Vw`'s bytes, guarded
    // ---- by `is_write` STRUCTURALLY. The clock product runs over ALL `D` levels where
    // ---- the panels' start at `j`: docs/internals/consumer_kernel.md#decay-clock
    if (DECAY && is_write) {
      const float* amp_c[D];
#pragma unroll
      for (int l = 0; l < D; ++l) {
        amp_c[l] = p.amp_clock[l] + tile_row * p.level_pitch_clock[l];
      }
      //: HAZARD compact-clock-list -- docs/internals/consumer_kernel.md#compact-clock-list
      for (int q = tid; q < (COMPACT ? K : BC); q += kConsumerThreads) {
        const int i = COMPACT ? c_idx[q] : q;
        float acc = 0.0f;
        for (int t = 0; t < BT; ++t) {
          float cl = 1.0f;
#pragma unroll
          for (int l = 0; l < D; ++l) {
            cl *= amp_c[l][t * p.level_pitch_clock[l] + dig[l * BC + i]];
          }
          //: HAZARD clock-prefix-fp32 -- docs/internals/consumer_kernel.md#clock-prefix-fp32
          acc += cl;
          clk[t * CKP + i] = acc;
        }
        const float pr = 0.5f * acc;
        p_ref[i] = pr;
        s0[i] = expf(pr * log_keep[i]);
      }
      __syncthreads();
      // ---- DECAY, step (c): the entry state enters the reference frame. `s0`
      // ---- multiplies the STATE, not the read panel; the mass column rides the same
      // ---- sweep because it is the same recurrence with `v := 1`. Row-per-warp rather
      // ---- than a flat `idx / cols` walk, because under `global` `cols = 65` is not a
      // ---- power of two: docs/internals/consumer_kernel.md#decay-clock
      for (int q = warp; q < (COMPACT ? K : BC); q += kWarps) {
        const int i = COMPACT ? c_idx[q] : q;
        const float s = s0[i];
        for (int c = lane; c < cols; c += 32) S[i * SP + c] *= s;
      }
    }
    // ---- §3 STEP 3's ROW SCALES, COMPUTED ABOVE THE PANEL BUILDS: constant
    // ---- along the leaf axis, so held OUT of the coefficient blocks, and the hoist
    // ---- collapses the pre-GEMM section from three phases to two:
    // ---- docs/internals/consumer_kernel.md#row-scales
    if (is_read) {
      for (int t = tid; t < BT; t += kConsumerThreads) {
        const int token = t0 + t;
        float s =
            (token < p.T)
                ? (p.g_read != nullptr ? p.g_read[static_cast<int64_t>(bh) * p.T + token] : 1.0f)
                : 0.0f;
#pragma unroll
        for (int l = 0; l < D; ++l) {
          if (l >= j) break;
          s *= amp_r[l][t * p.level_pitch_read[l] + base[l]];
        }
        sr[t] = s;
      }
    }
    if (is_write) {
      for (int t = tid; t < BT; t += kConsumerThreads) {
        const int token = t0 + t;
        float s = (token < p.T) ? p.g_write[static_cast<int64_t>(bh) * p.T + token] : 0.0f;
#pragma unroll
        for (int l = 0; l < D; ++l) {
          if (l >= j) break;
          s *= amp_w[l][t * p.level_pitch_write[l] + base[l]];
        }
        sw[t] = s;
      }
    }
    if (!DECAY && is_write) __syncthreads();
    // ---- THE READ PANEL (§3 steps 2-3). The loop walks FRAGMENT SLOTS, not
    // ---- `(token, leaf)` pairs, so the store is contiguous in `tid`; `COMPACT` builds
    // ---- it COMPACTED IN THE FIRST PLACE -- the §2.6 dividend a consumer that
    // ---- receives its panel dense pays a copy for:
    // ---- docs/internals/consumer_kernel.md#read-panel
    if (is_read) {
      auto amp_read_at = [&](int t, int k) {
        if (COMPACT && k >= K) return 0.0f;
        const int i = COMPACT ? c_idx[k] : k;
        float a = 1.0f;
#pragma unroll
        for (int l = 0; l < D; ++l) {
          if (l < j) continue;  // fixed levels hoist into the row scale (§3 step 3)
          a *= amp_r[l][t * p.level_pitch_read[l] + dig[l * BC + i]];
        }
        if (DECAY && is_write) {
          a *= panel_read_factor(clk[t * CKP + i], p_ref[i], log_keep[i]);
        }
        return a;
      };
      const int rf_words = COMPACT ? frag_a_words<READ_PREC>(Kpad) : kRfBlock;
#pragma unroll
      for (int mb = 0; mb < NM; ++mb) {
        unsigned* Rb = Rf + mb * kRfBlock;
        for (int sl = tid; sl < rf_words; sl += kConsumerThreads) {
          int tl, k;
          frag_a_slot<READ_PREC>(sl, tl, k);
          const int t = mb * 16 + tl;
          if (READ_PREC == MmaPrec::kBf16) {
            Rb[sl] = pack_bf16x2(clamp_bf16_finite(amp_read_at(t, k)),
                                 clamp_bf16_finite(amp_read_at(t, k + 1)));
          } else {
            Rb[sl] = __float_as_uint(amp_read_at(t, k));
          }
        }
      }
    }
    // ---- THE WRITE PANEL'S PLANE CALL SITE, hoisted OUT of the window loop when the
    // ---- window IS the plane -- which is what keeps the `DECAY` arm's build ABOVE the
    // ---- `clk -> Vw` lifetime barrier, where `clk` is still live.
    //: HAZARD wt-build-site -- docs/internals/consumer_kernel.md#wt-build-site
    if constexpr (kWtPlane && kRing == 1) {
      if (is_write) {
        build_wt_rows(
            0, wt_rows,
            [&](int l, int t, int i) {
              return amp_w[l][t * p.level_pitch_write[l] + dig[l * BC + i]];
            },
            c_idx, K);
      }
    }
    //: HAZARD clk-dead -- docs/internals/consumer_kernel.md#clk-dead
    if (DECAY && is_write) __syncthreads();  // <- clk is now DEAD
    // ---- `Vw[t, ch] = scale_write[t] * v[t0 + t, ch]`. The write side folds the row
    // ---- scale into the VALUE tile because there `t` is the CONTRACTION index and the
    // ---- scale cannot hoist out of the sum; one staged object, two GEMMs.
    if (is_write) {
      for (int idx = tid; idx < BT * DV; idx += kConsumerThreads) {
        const int t = idx / DV;
        const int ch = idx - t * DV;
        const int token = t0 + t;
        const float vv =
            (token < p.T) ? p.v[(static_cast<int64_t>(bh) * p.T + token) * DV + ch] : 0.0f;
        stage_put(Vw[vw_index<BT, DV, kNarrowStage>(t, ch)], sw[t] * vv);
      }
    }
  };

  // ---- THE CANDIDATE WALK, SOFTWARE-PIPELINED AT `kRing == 2`. `visit` -- not
  // ---- `tile` -- indexes the ring, because the slots alternate over CANDIDATE VISITS
  // ---- while the tile index skips.
  //: HAZARD loop-induction -- docs/internals/consumer_kernel.md#loop-induction
  if constexpr (kRing == 2) {
    const int first = next_cand(tile_lo);
    if (first < tile_hi) {
      stage_tile(first, 0, ((cand_read[first >> 5] >> (first & 31)) & 1u) != 0u,
                 ((cand_write[first >> 5] >> (first & 31)) & 1u) != 0u);
    }
  }
  for (int tile = tile_lo, visit = 0; tile < tile_hi; ++tile) {
    const bool is_read = ((cand_read[tile >> 5] >> (tile & 31)) & 1u) != 0u;
    const bool is_write = ((cand_write[tile >> 5] >> (tile & 31)) & 1u) != 0u;
    if (!is_read && !is_write) continue;

    const int t0 = tile * BT;
    const int slot = (kRing == 2) ? (visit & 1) : 0;
    float* sr = sr_r0 + slot * 2 * BT;
    float* sw = sr + BT;
    unsigned* Rf = Rf_r0 + slot * kRfWords;
    StageT* Vw = Vw_r0 + slot * kVwSlot;
    int* c_idx = c_idx_r0 + slot * kCSlot;
    int* c_count = c_idx + (COMPACT ? 2 * BC : 0);

    // ---- THE TILE'S CONTROL BLOCK, RESET. It needs no barrier of its own on
    // ---- EITHER side: behind it stands the previous tile's end-of-tile
    // ---- `__syncthreads()`, ahead of it the unconditional staging publication. The
    // ---- edge DAG this reset serves, and the two-part always-runnable claim that
    // ---- makes the tile deadlock-free:
    // ---- docs/internals/consumer_kernel.md#edge-dag
    if (tid < ctl_words(BC)) ctl[tid] = 0;
    if constexpr (kRing == 1) stage_tile(tile, 0, is_read, is_write);

    const int K = COMPACT ? c_count[0] : BC;
    const int KI = COMPACT ? c_count[1] : BC;
    constexpr int kUnit = (READ_PREC == MmaPrec::kTf32x3) ? 8 : 16;
    const int Kpad = ((K + kUnit - 1) / kUnit) * kUnit;    // readout k-extent
    const int KIpad = ((KI + kUnit - 1) / kUnit) * kUnit;  // intra Gram k-extent
    const int KpadM = ((K + 15) / 16) * 16;                // fold m-extent
    const int wt_rows = COMPACT ? KpadM : BC;
    const int64_t tile_row = (static_cast<int64_t>(bh) * p.n_tiles + tile) * BT;
    // ---- THE WRITE PANEL AT DEPTH 2, AND IT COSTS NO BARRIER: the previous tile's
    // ---- end-of-tile barrier stands between the last fold read of `WT` and this
    // ---- write, and the publication rendezvous below between this write and the
    // ---- Gram's read: docs/internals/consumer_kernel.md#wt-build-site
    if constexpr (kWtPlane && kRing == 2) {
      if (is_write) {
        const float* amp_w[D];
#pragma unroll
        for (int l = 0; l < D; ++l) {
          amp_w[l] = p.amp_write[l] + tile_row * p.level_pitch_write[l];
        }
        build_wt_rows(
            0, wt_rows,
            [&](int l, int t, int i) {
              return amp_w[l][t * p.level_pitch_write[l] + dig[l * BC + i]];
            },
            c_idx, K);
      }
    }

    __syncthreads();

    // ---- GEMM 1: THE READOUT, `y_tile = R . S`. The B operand is packed out of `S` on
    // ---- the fly; STATE AND ACCUMULATION STAY fp32. The y-tile accumulator is
    // ---- REGISTERS: docs/internals/consumer_kernel.md#gemm-readout
    constexpr int kYTiles = (NM * NT_DV + kWarps - 1) / kWarps;
    float y[kYTiles][4];
#pragma unroll
    for (int q = 0; q < kYTiles; ++q) {
      y[q][0] = 0.0f;
      y[q][1] = 0.0f;
      y[q][2] = 0.0f;
      y[q][3] = 0.0f;
    }
    //: HAZARD nm-fork -- docs/internals/consumer_kernel.md#nm-fork
    if constexpr (NM == 1) {
      if (is_read && warp < NT_DV) {
        mma_tile_fragA<BC, float, false, COMPACT, READ_PREC>(Rf, S, SP, c_idx, Kpad, warp * 8, lane,
                                                             y[0][0], y[0][1], y[0][2], y[0][3]);
      }
    } else if (is_read) {
#pragma unroll
      for (int q = 0; q < kYTiles; ++q) {
        const int wt = warp + q * kWarps;
        if (wt >= NM * NT_DV) break;
        const int mb = wt / NT_DV, nt = wt - mb * NT_DV;
        mma_tile_fragA<BC, float, false, COMPACT, READ_PREC>(Rf + mb * kRfBlock, S, SP, c_idx, Kpad,
                                                             nt * 8, lane, y[q][0], y[q][1],
                                                             y[q][2], y[q][3]);
      }
    }

    const bool mixed = is_read && is_write;

    // ---- THE MASS COLUMN, TERM 1 -- placement FORCED by the windowed phase B: the fold
    // ---- WRITES `S[i][DV]`, so term 1, which READS it, must already have run:
    // ---- docs/internals/consumer_kernel.md#mass-term1
    //: HAZARD mass-split -- docs/internals/consumer_kernel.md#mass-split
    if (GLOBAL && is_read) {
      for (int t = tid; t < BT; t += kConsumerThreads) {
        float acc = 0.0f;
        const unsigned* Rb = Rf + (NM == 1 ? 0 : (t >> 4) * kRfBlock);
        const int tl = (NM == 1) ? t : (t & 15);
        for (int k = 0; k < K; ++k) {
          acc += frag_a_get<READ_PREC>(Rb, tl, k) * S[(COMPACT ? c_idx[k] : k) * SP + DV];
        }
        den[t] = acc;
      }
    }

    // ---- B6: THE TILE'S READ-THEN-WRITE ORDERING ON `S`, AS A COUNTER rather than a
    // ---- rendezvous, so only the FOLD waits and it waits behind real work. The
    // ---- retirement is UNCONDITIONAL: docs/internals/consumer_kernel.md#b6
    retire_item(&ctl[kCtlReadDone], lane);

    // ---- PHASE B: the windowed {build, Gram, publish, fold} loop. The publication's
    // ---- move to the LAST WINDOW'S Gram is the one reordering. At `kNWin == 1` (the
    // ---- shipped default and the `DECAY` pin) the loop has ONE iteration:
    // ---- docs/internals/consumer_kernel.md#phase-b
    //: HAZARD gram-pinned -- docs/internals/consumer_kernel.md#gram-pinned
    constexpr int kGTiles = (NM * (BT / 8) + kWarps - 1) / kWarps;
    constexpr int kGramWarps = (NM * (BT / 8) < kWarps) ? NM * (BT / 8) : kWarps;
    //: WHETHER THE PHASE-C GATE IS NARROWED AT ALL, and this is a MEASURED SCOPE, not a
    //: preference: the narrowing is bit-identical everywhere and free on `bf16`, but on
    //: the `tf32x3` arms it re-schedules `ptxas` on already register-starved
    //: instantiations and the ratchet measured 12 acquiring IN-LOOP spill. The scope is
    //: therefore the PRECISION axis -- a FORMAT axis the dispatch already forks on, not
    //: a size test: docs/internals/consumer_kernel.md#gram-gate
    constexpr bool kNarrowGate = stage_is_narrow<READ_PREC, FOLD_PREC>();
    float gd[kGTiles][4];
#pragma unroll
    for (int q = 0; q < kGTiles; ++q) {
      gd[q][0] = 0.0f;
      gd[q][1] = 0.0f;
      gd[q][2] = 0.0f;
      gd[q][3] = 0.0f;
    }
    const int n_mt = COMPACT ? KpadM / 16 : BC / 16;
    const int qmax = COMPACT ? K : BC;
    for (int w = 0; w < kNWin; ++w) {
      if constexpr (!kWtPlane) {
        const int b_lo = w * kWinRows;
        const int b_hi = min(wt_rows, b_lo + kWinRows);
        if (is_write && b_lo < b_hi) {
          const float* amp_w[D];
#pragma unroll
          for (int l = 0; l < D; ++l) {
            amp_w[l] = p.amp_write[l] + tile_row * p.level_pitch_write[l];
          }
          build_wt_rows(
              b_lo, b_hi,
              [&](int l, int t, int i) {
                return amp_w[l][t * p.level_pitch_write[l] + dig[l * BC + i]];
              },
              c_idx, K);
        }
        __syncthreads();
      }
      // ---- GEMM 2: the causal effective-score block, `G += R_q . W_q^T`, then §5's
      // ---- INCLUSIVE causal mask applied AFTER the loop, on the completed Gram.
      // ---- `B_T = false` because the transpose `W^T` names is in the STORAGE -- which
      // ---- is the whole point of building `WT` leaf-major. `COMPACT` cuts this
      // ---- k-extent to §5's CAUSAL PREFIX with no gather at all:
      // ---- docs/internals/consumer_kernel.md#gemm-gram
      const int kw = min(kWinRows, KIpad - w * kWinRows);
      if constexpr (NM == 1) {
        if (mixed && warp < BT / 8 && kw > 0) {
          if constexpr (kNarrowStage) {
            mma_tile_fragA_wtB<kWinRows, BT>(Rf + w * (kWinRows / kUnit) * 128, WT, kw, warp * 8,
                                             lane, gd[0][0], gd[0][1], gd[0][2], gd[0][3]);
          } else {
            mma_tile_fragA<kWinRows, StageT, false, false, READ_PREC>(
                Rf + w * (kWinRows / kUnit) * 128, WT, BTP, nullptr, kw, warp * 8, lane, gd[0][0],
                gd[0][1], gd[0][2], gd[0][3]);
          }
        }
      } else if (mixed && kw > 0) {
#pragma unroll
        for (int q = 0; q < kGTiles; ++q) {
          const int gt = warp + q * kWarps;
          if (gt >= NM * (BT / 8)) break;
          const int gmb = gt / (BT / 8), gnt = gt - gmb * (BT / 8);
          if constexpr (kNarrowStage) {
            mma_tile_fragA_wtB<kWinRows, BT>(Rf + gmb * kRfBlock + w * (kWinRows / kUnit) * 128, WT,
                                             kw, gnt * 8, lane, gd[q][0], gd[q][1], gd[q][2],
                                             gd[q][3]);
          } else {
            mma_tile_fragA<kWinRows, StageT, false, false, READ_PREC>(
                Rf + gmb * kRfBlock + w * (kWinRows / kUnit) * 128, WT, BTP, nullptr, kw, gnt * 8,
                lane, gd[q][0], gd[q][1], gd[q][2], gd[q][3]);
          }
        }
      }

      if (w + 1 == kNWin) {
        if constexpr (NM == 1) {
          if (mixed && warp < BT / 8) {
            CONSUMER_D_ROWS(lane);
            const int n0 = warp * 8;
            const int c0 = n0 + 2 * _r, c1 = n0 + 2 * _r + 1;
            gm_put(Gm[_g * GMP + c0], (c0 <= _g) ? gd[0][0] : 0.0f);
            gm_put(Gm[_g * GMP + c1], (c1 <= _g) ? gd[0][1] : 0.0f);
            gm_put(Gm[(_g + 8) * GMP + c0], (c0 <= _g + 8) ? gd[0][2] : 0.0f);
            gm_put(Gm[(_g + 8) * GMP + c1], (c1 <= _g + 8) ? gd[0][3] : 0.0f);
          }
        } else if (mixed) {
          CONSUMER_D_ROWS(lane);
#pragma unroll
          for (int q = 0; q < kGTiles; ++q) {
            const int gt = warp + q * kWarps;
            if (gt >= NM * (BT / 8)) break;
            const int gmb = gt / (BT / 8), gnt = gt - gmb * (BT / 8);
            const int n0 = gnt * 8;
            const int m0 = gmb * 16;
            const int r0 = m0 + _g, r1 = m0 + _g + 8;
            const int c0 = n0 + 2 * _r, c1 = n0 + 2 * _r + 1;
            gm_put(Gm[r0 * GMP + c0], (c0 <= r0) ? gd[q][0] : 0.0f);
            gm_put(Gm[r0 * GMP + c1], (c1 <= r0) ? gd[q][1] : 0.0f);
            gm_put(Gm[r1 * GMP + c0], (c0 <= r1) ? gd[q][2] : 0.0f);
            gm_put(Gm[r1 * GMP + c1], (c1 <= r1) ? gd[q][3] : 0.0f);
          }
        }
        // ---- B5, AS A COUNTER: `Gm` is published to phase C. Narrow arm: retired by the
        // ---- WRITERS ONLY, target `kGramWarps`, `!mixed` skipping it entirely. Wide arm:
        // ---- every warp, target `kWarps`. Both satisfy the same release invariant:
        // ---- docs/internals/consumer_kernel.md#gram-gate
        //: HAZARD gram-gate-guard -- docs/internals/consumer_kernel.md#gram-gate-guard
        //: HAZARD gate-if-constexpr -- docs/internals/consumer_kernel.md#gate-if-constexpr
        if constexpr (kNarrowGate) {
          if (mixed && warp < kGramWarps) retire_item(&ctl[kCtlGramDone], lane);
        } else {
          retire_item(&ctl[kCtlGramDone], lane);
        }
      }

      // ---- PHASE B-2: GEMM 4, THE BOUNDARY FOLD `S += W^T . Vw`, AND IT IS THE TILE'S
      // ---- FLUID ITEM POOL -- the one GEMM whose accumulators are born and die inside
      // ---- one item. `W^T` is not a transpose at the fragment: `WT` IS the transpose:
      // ---- docs/internals/consumer_kernel.md#gemm-fold
      if (is_write) {
        if (w == 0) await_count(&ctl[kCtlReadDone], kWarps);
        {
          const int mt_lo = w * kWtWindow;
          const int mt_hi = min(n_mt, mt_lo + kWtWindow);
          const int n_fold_items = (mt_hi - mt_lo) * NT_DV;
          for (;;) {
            //: HAZARD fold-claim-uniform -- docs/internals/consumer_kernel.md#fold-claim-uniform
            int c = 0;
            if (lane == 0) c = atomicAdd(&ctl[kCtlFold0 + w], 1);
            c = __shfl_sync(0xffffffffu, c, 0);
            if (c >= n_fold_items) break;
            CONSUMER_D_ROWS(lane);
            const int mr = c / NT_DV;   // the WINDOW-LOCAL m-tile
            const int mt = mt_lo + mr;  // the tile-global one, for `S`
            const int nt = c - mr * NT_DV;
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
            if constexpr (kNarrowStage) {
              mma_fold_tile<kWtRows, BT, DV>(WT, Vw, mr * 16, nt * 8, lane, d0, d1, d2, d3);
            } else {
              mma_tile<StageT, false, StageT, false, FOLD_PREC>(WT, BTP, Vw, VWP, BT, mr * 16,
                                                                nt * 8, lane, d0, d1, d2, d3);
            }
            const int r0 = mt * 16 + _g, n0 = nt * 8 + 2 * _r;
            // ---- DECAY's EXIT SCALE RIDES THE FOLD'S EXISTING READ-MODIFY-WRITE: the
            // ---- inner `s0` was applied at entry, so what is left is
            // ---- `S <- s0 * (S + d)` -- ONE extra multiply on a value already in a
            // ---- register, replacing a separate `[BC x cols]` sweep that accounted for
            // ---- the bulk of decay's +219.6 % shared-store traffic:
            // ---- docs/internals/consumer_kernel.md#fold-epilogue
            //: HAZARD ragged-m-guard -- docs/internals/consumer_kernel.md#ragged-m-guard
            const int i0 = COMPACT ? c_idx[r0] : r0;
            const int i1 = COMPACT ? c_idx[r0 + 8] : r0 + 8;
            const float e0 = DECAY ? s0[i0] : 1.0f;
            const float e1 = DECAY ? s0[i1] : 1.0f;
            if (!COMPACT || r0 < K) {
              S[i0 * SP + n0] = e0 * (S[i0 * SP + n0] + d0);
              S[i0 * SP + n0 + 1] = e0 * (S[i0 * SP + n0 + 1] + d1);
            }
            if (!COMPACT || r0 + 8 < K) {
              S[i1 * SP + n0] = e1 * (S[i1 * SP + n0] + d2);
              S[i1 * SP + n0 + 1] = e1 * (S[i1 * SP + n0 + 1] + d3);
            }
          }
          if (GLOBAL) {
            const int q_lo = w * kWinRows;
            const int q_hi = min(qmax, q_lo + kWinRows);
            for (int q = q_lo + tid; q < q_hi; q += kConsumerThreads) {
              const int i = COMPACT ? c_idx[q] : q;
              float acc = 0.0f;
              for (int t = 0; t < BT; ++t)
                acc += mma_src(WT[wt_index<BT, kNarrowStage>(q - q_lo, t)]) * sw[t];
              S[i * SP + DV] = (DECAY ? s0[i] : 1.0f) * (S[i * SP + DV] + acc);
            }
          }
        }  // the window's item pool
      }  // if (is_write)
      if constexpr (!kWtPlane) {
        if (w + 1 < kNWin) __syncthreads();
      }
    }

    // ---- THE RUN-AHEAD, ISSUED AT THE ONE PROGRAM POINT where this tile has no
    // ---- compute left to protect. Four facts license the placement and all four are
    // ---- properties of that point: docs/internals/consumer_kernel.md#run-ahead
    if constexpr (kRing == 2) {
      const int nxt = next_cand(tile + 1);
      if (nxt < tile_hi) {
        stage_tile(nxt, slot ^ 1, ((cand_read[nxt >> 5] >> (nxt & 31)) & 1u) != 0u,
                   ((cand_write[nxt >> 5] >> (nxt & 31)) & 1u) != 0u);
      }
    }

    // ---- THE PHASE-C GATE, SKIPPED ON `!mixed` AND NARROWED TO THE WRITERS, and it is
    // ---- FREE -- `kCtlGramDone` is retired AHEAD of the fold. The wait is nonetheless
    // ---- real: docs/internals/consumer_kernel.md#gram-gate
    //: HAZARD gram-gate-guard -- docs/internals/consumer_kernel.md#gram-gate-guard
    if constexpr (kNarrowGate) {
      if (mixed) await_count(&ctl[kCtlGramDone], kGramWarps);
    } else {
      await_count(&ctl[kCtlGramDone], kWarps);
    }

    // ---- PHASE C -- GEMM 3, the intra term `y_tile += G . Vw`. The SAME tile
    // ---- decomposition the readout used, including the same `NM` fork, so the two
    // ---- GEMMs accumulate into the same warp's same fragment. POST-LOOP because it
    // ---- needs the completed Gram, which is what keeps `y` live across phase B:
    // ---- docs/internals/consumer_kernel.md#phase-c
    //: HAZARD nm-fork -- docs/internals/consumer_kernel.md#nm-fork
    if constexpr (NM == 1) {
      if (mixed && warp < NT_DV) {
        if constexpr (kNarrowStage) {
          mma_tile_vwB<GmT, BT, DV>(Gm, GMP, Vw, 0, warp * 8, lane, y[0][0], y[0][1], y[0][2],
                                    y[0][3]);
        } else {
          mma_tile<GmT, false, StageT, false, READ_PREC>(Gm, GMP, Vw, VWP, BT, 0, warp * 8, lane,
                                                         y[0][0], y[0][1], y[0][2], y[0][3]);
        }
      }
    } else if (mixed) {
#pragma unroll
      for (int q = 0; q < kYTiles; ++q) {
        const int wt = warp + q * kWarps;
        if (wt >= NM * NT_DV) break;
        const int mb = wt / NT_DV, nt = wt - mb * NT_DV;
        if constexpr (kNarrowStage) {
          mma_tile_vwB<GmT, BT, DV>(Gm, GMP, Vw, mb * 16, nt * 8, lane, y[q][0], y[q][1], y[q][2],
                                    y[q][3]);
        } else {
          mma_tile<GmT, false, StageT, false, READ_PREC>(Gm, GMP, Vw, VWP, BT, mb * 16, nt * 8,
                                                         lane, y[q][0], y[q][1], y[q][2], y[q][3]);
        }
      }
    }

    //: HAZARD mass-split -- docs/internals/consumer_kernel.md#mass-split
    if (GLOBAL && mixed) {
      for (int t = tid; t < BT; t += kConsumerThreads) {
        float acc = den[t];
        for (int u = 0; u <= t; ++u) acc += mma_src(Gm[t * GMP + u]) * sw[u];
        den[t] = acc;
      }
    }
    //: HAZARD emission-interval -- docs/internals/consumer_kernel.md#emission-interval

    // ---- SPLIT-K EMISSION of this tile's partial, straight out of the D fragment,
    // ---- with the §3 step-3 row scale applied here because it is constant along both
    // ---- `k` and `n`. The `!= 0.0f` guard is a PERFORMANCE SKIP ON A VALUE, never a
    // ---- membership decision: docs/internals/consumer_kernel.md#split-k-emission
    //: HAZARD emit-no-pointer-array -- docs/internals/consumer_kernel.md#emit-no-pointer-array
    if (is_read) {
      if constexpr (NM == 1) {
        if (warp < NT_DV) {
          CONSUMER_D_ROWS(lane);
          const int n0 = warp * 8;
          float* num_t0 = p.num + (static_cast<int64_t>(bh) * p.T + t0) * DV;
          auto emit = [&](int t, int ch, float val) {
            if (t0 + t >= p.T) return;
            const float out = sr[t] * val;
            if (out != 0.0f) atomicAdd(&num_t0[t * DV + ch], out);
          };
          const int c0 = n0 + 2 * _r, c1 = c0 + 1;
          emit(_g, c0, y[0][0]);
          emit(_g, c1, y[0][1]);
          emit(_g + 8, c0, y[0][2]);
          emit(_g + 8, c1, y[0][3]);
        }
      } else {
        CONSUMER_D_ROWS(lane);
        float* num_t0 = p.num + (static_cast<int64_t>(bh) * p.T + t0) * DV;
        auto emit = [&](int t, int ch, float val) {
          if (t0 + t >= p.T) return;
          const float out = sr[t] * val;
          if (out != 0.0f) atomicAdd(&num_t0[t * DV + ch], out);
        };
#pragma unroll
        for (int q = 0; q < kYTiles; ++q) {
          const int wt = warp + q * kWarps;
          if (wt >= NM * NT_DV) break;
          const int mb = wt / NT_DV, nt = wt - mb * NT_DV;
          const int n0 = nt * 8, m0 = mb * 16;
          const int c0 = n0 + 2 * _r, c1 = c0 + 1;
          emit(m0 + _g, c0, y[q][0]);
          emit(m0 + _g, c1, y[q][1]);
          emit(m0 + _g + 8, c0, y[q][2]);
          emit(m0 + _g + 8, c1, y[q][3]);
        }
      }
      if (GLOBAL) {
        for (int t = tid; t < BT; t += kConsumerThreads) {
          const int token = t0 + t;
          if (token >= p.T) continue;
          const float d = sr[t] * den[t];
          if (d != 0.0f) atomicAdd(&p.den[static_cast<int64_t>(bh) * p.T + token], d);
        }
      }
    }
    // ---- §6.4 PASS 1's PER-LEAF EPILOGUE, POST-LOOP -- per-leaf and window-independent.
    // ---- The COEFFICIENT half of the inclusion test lives in the panel build; what
    // ---- is left here has no pre-quantization site:
    // ---- docs/internals/consumer_kernel.md#pass1-epilogue
    if (is_write) {
      if (summary) {
        if (DECAY) {
          for (int q = tid; q < (COMPACT ? K : BC); q += kConsumerThreads) {
            const int i = COMPACT ? c_idx[q] : q;
            if (s0[i] != 1.0f) sum_slot[i] = 1;
          }
        }
        if (DECAY) {
          for (int q = tid; q < (COMPACT ? K : BC); q += kConsumerThreads) {
            const int i = COMPACT ? c_idx[q] : q;
            a_state[i] *= s0[i] * s0[i];
          }
        }
      }
    }
    __syncthreads();
    ++visit;
  }

  if (summary) {
    // ---- §6.4 PASS 1's PUBLICATION: the SPARSE written-leaf summary. Slot `owner_slot0`
    // ---- is the scan's entry stash, so this lands one past it and pass 3's segment `q`
    // ---- reads `owner_slot0 + q` UNIFORMLY:
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
  // ---- THE OWNER'S STATE BLOCK, WRITTEN TO HBM ONCE. An unpaged atom is never stored
  // ---- back, PER ATOM. `last_seg` is §6.4's publication rule and is a DELIBERATE
  // ---- NUMERIC CHOICE, not the scan's job:
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
