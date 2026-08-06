#pragma once

// THE RoLA V3 CONSUMER KERNEL. It is the ONLY consumer, and this file is its BODY:
// the §3/§5 tile algorithm, the launch bound, the staging ring's candidate walk, the
// four GEMMs in their phase order and the split-K epilogue. `ConsumerParams` and the
// host dispatch are `consumer.{cuh,cu}`; the supported shape surface is declared at
// `launch_consumer` and is deliberately NOT restated here.
//
// THE BODY COMPOSES THREE LAYERS and is the only file that knows the tile algorithm
// they serve:
//
//   `pipeline.cuh`    the item stream -- the control words and the three
//                     release/acquire primitives that order a tile's phases
//   `smem_layout.cuh` the carve -- the address functions, their compile-time
//                     conflict proofs, the register/residency budget and the ring
//                     depth: what the CTA's shared memory looks like and what it costs
//   `mma_tile.cuh`    the operand maps -- fragment order, the `ldmatrix` atoms and the
//                     four GEMMs' inner sequences
//
// Each includes the one above it, so including `mma_tile.cuh` brings all three. The
// pointer carve below is `smem_layout.cuh`'s byte ledger spelled as pointers, in the
// same order, and `consumer_smem_bytes_at` is what the launch reserves.
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

#include "arch_table.cuh"
#include "intra_panel.cuh"
#include "consumer.cuh"
//: The three layers, in one include: `mma_tile.cuh` -> `smem_layout.cuh` ->
//: `pipeline.cuh`. The CuTe pin they rest on is declared in the two headers that
//: actually use it. docs/internals/mma_tile.md#cute-pin
#include "mma_tile.cuh"
#include "readout_mma.cuh"
#include "segment_mask.cuh"

namespace rola {
namespace consumer {

//: §12 rules 1 and 3's survival factors, taken from `intra_panel.cuh` rather than
//: reimplemented: it is the one clamp point and the one exponent sign convention.
using rola::panel_read_factor;
using rola::panel_write_factor;
//: §6.4's per-segment tile-range mask -- one named definition, because the host-side
//: segment partition at `P > 1` is pinned against exactly this arithmetic.
using rola::detail::segment_word_mask;

//: `kMmaKQuantum` is `arch_table.cuh`'s, not this file's: it is one number for the
//: kernel, the §6.4 scan and the summary pass. The planner mirrors the same value at
//: `rola.planner.schedule.MMA_K_QUANTUM`. docs/internals/arch_table.md#mma-quantum

//: One widening for the amplitude-class storage types (`AmpRT`/`AmpWT`, P09
//: family): identity on fp32 arms, the explicit intrinsic on bf16 -- overloads
//: rather than a `static_cast` so the conversion survives
//: `__CUDA_NO_BFLOAT16_CONVERSIONS__` builds.
__device__ __forceinline__ float amp_to_float(float x) { return x; }
__device__ __forceinline__ float amp_to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

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
  ctas_for_residency(                                                               \
      consumer_smem_bytes<BT, BC, DV, D, READ_PREC, FOLD_PREC, DECAY, GLOBAL>(),    \
      consumer_reg_budget(BT, BC, DECAY, READ_PREC, FOLD_PREC, DV))
#endif
template <int BT, int BC, int DV, int D, MmaPrec FOLD_PREC, MmaPrec READ_PREC, bool DECAY,
          bool GLOBAL>
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
  // ---- docs/internals/smem_layout.md#smem-ledger
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
  //: THE AMPLITUDE-CLASS STORAGE TYPES (P09 family), derived from the EXISTING
  //: precision and decay axes the way `StageT`/`GmT` are -- zero new template
  //: axes. SPANNED amplitude levels (`l >= j`) and the value stream are stored in
  //: the arm's amplitude type (bf16 under `kBf16`: they feed operands the MMA
  //: narrows to bf16 anyway, so the fp32 store carried bytes, not precision).
  //: FIXED levels (`l < j`) stay `float` -- they feed the per-token ROW SCALES,
  //: whose error is uniform over the whole output row and never cancels
  //: (Option B). The read side keys off `READ_PREC`, the write/value side off
  //: `FOLD_PREC`, mirroring which GEMM consumes each stream. The clock stays
  //: `float` unconditionally (an exponent;
  //: docs/internals/consumer_kernel.md#clock-prefix-fp32). The DECAY and RAW
  //: arms keep `float` storage outright -- MEASURED carves on structural axes,
  //: not preferences (workflows/perf/p09_bf16_family.md): the ratchet found
  //: every in-loop-spill offender under narrowed storage on the bf16+DECAY arms
  //: (clock/panel registers at the allocator wall), and the RAW arm -- no
  //: `num/den` ratio to cancel stacked roundings -- measured past `BF16_RTOL`
  //: on its worst topology's tail seeds.
  using AmpRT = typename cuda::std::conditional<READ_PREC == MmaPrec::kBf16 && !DECAY && GLOBAL,
                                                __nv_bfloat16, float>::type;
  using AmpWT = typename cuda::std::conditional<FOLD_PREC == MmaPrec::kBf16 && !DECAY && GLOBAL,
                                                __nv_bfloat16, float>::type;
  constexpr int kRing = consumer_ring_depth<BT, BC, DV, D, READ_PREC, FOLD_PREC, DECAY, GLOBAL>();
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
  int* ctl = reinterpret_cast<int*>(s0 + (DECAY ? BC : 0));
  //: The token gather's group list: `t_idx[q]` = the q-th gathered GLOBAL token
  //: index, ascending (§8's stable gather -- ordering is the causality
  //: requirement, not a nicety). Written by warp 0 at ballot time, published by
  //: the group-entry `__syncthreads()`, read by every staging loop of a
  //: gathered visit. Dead bytes on an aligned walk.
  int* t_idx = ctl + ctl_words(BC);
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
      sum_slot[i] = 0;    // the written-leaf FLAG until the publication rewrites it
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
  auto build_wt_rows = [&](int q_lo, int q_hi, auto&& amp_w_at) {
    //: HAZARD wt-row-t-fastest -- docs/internals/consumer_kernel.md#wt-row-t-fastest
    //: HAZARD ballot-lane-run -- docs/internals/consumer_kernel.md#ballot-lane-run
    constexpr unsigned kRowLanes = (BT >= 32) ? 0xffffffffu : ((1u << BT) - 1u);
    static_assert(BT == 16 || BT == 32,
                  "the pass-1 emptiness ballot assumes the BT threads of one WT row "
                  "are a contiguous lane run inside one warp");
    for (int idx = tid; idx < BT * (q_hi - q_lo); idx += kConsumerThreads) {
      const int qr = idx / BT;
      const int i = q_lo + qr;
      const int t = idx - qr * BT;
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
  // ---- `tile < 0` is a GATHERED visit: the staged tokens are `t_idx[0..gN)`
  // ---- (globally ascending), `slot` is 0 and the ring is not used -- the gathered
  // ---- walk stages at the group top on every arm. Slots `[gN, BT)` of a ragged
  // ---- group stage EXACT-ZERO coefficient rows and zero row scales, so they are
  // ---- inert through every GEMM the way the aligned walk's `token >= T` tail is.
  // ---- docs/internals/consumer_kernel.md#candidate-walk, #token-gather
  auto stage_tile = [&](int tile, int slot, bool is_read, bool is_write, int gN) {
    const bool ga = tile < 0;
    const int t0 = ga ? 0 : tile * BT;
    float* sr = sr_r0 + slot * 2 * BT;
    float* sw = sr + BT;
    unsigned* Rf = Rf_r0 + slot * kRfWords;
    StageT* Vw = Vw_r0 + slot * kVwSlot;

    //: The amplitude-row base this visit's typed pointers advance by: the tile's
    //: first token row when aligned, the `bh` stream base when gathered (each
    //: gathered access then offsets by its own global token index).
    const int64_t tile_row = ga ? static_cast<int64_t>(bh) * p.n_tiles * BT
                                : (static_cast<int64_t>(bh) * p.n_tiles + tile) * BT;
    //: SPANNED-TYPED tile bases, in the baseline's exact shape (one typed
    //: pointer advance per level, no role branch -- a per-level role ternary
    //: here measurably spilled the D >= 3 decay arms). Entries `l < j` carry a
    //: WRONGLY-SCALED address when `AmpRT`/`AmpWT` is not `float` and are never
    //: dereferenced on that arm: the FIXED levels' consumers (the row-scale
    //: loops below) index their always-fp32 streams directly from the params on
    //: the split arm, outside any hot loop.
    const AmpRT* amp_r[D];
    const AmpWT* amp_w[D];
#pragma unroll
    for (int l = 0; l < D; ++l) {
      amp_r[l] = static_cast<const AmpRT*>(p.amp_read[l]) + tile_row * p.level_pitch_read[l];
      amp_w[l] = static_cast<const AmpWT*>(p.amp_write[l]) + tile_row * p.level_pitch_write[l];
    }
    //: Whether the amplitude storage is ROLE-SPLIT on this arm (P09 family:
    //: spanned narrow, fixed fp32). On the uniform-fp32 arms the row scales read
    //: `amp_r`/`amp_w` exactly as they always did.
    constexpr bool kAmpSplit = sizeof(AmpRT) == 2 || sizeof(AmpWT) == 2;

    // ---- DECAY, step (a): the clock prefix, `P_ref`, `s0` -- INTO `Vw`'s bytes, guarded
    // ---- by `is_write` STRUCTURALLY. The clock product runs over ALL `D` levels where
    // ---- the panels' start at `j`: docs/internals/consumer_kernel.md#decay-clock
    if (DECAY && is_write) {
      const float* amp_c[D];
#pragma unroll
      for (int l = 0; l < D; ++l) {
        amp_c[l] = p.amp_clock[l] + tile_row * p.level_pitch_clock[l];
      }
      //: ONE textual clock product for both walks (§3.4's restriction: the loop
      //: BOUNDS move, the product expression does not). `crow(t)` is the token's
      //: amplitude row under the base above: `t` aligned, `t_idx[t]` gathered.
      auto clock_prefix = [&](auto&& camp, int nT) {
        for (int i = tid; i < BC; i += kConsumerThreads) {
          float acc = 0.0f;
          for (int t = 0; t < nT; ++t) {
            float cl = 1.0f;
#pragma unroll
            for (int l = 0; l < D; ++l) {
              cl *= camp(l, t, i);
            }
            //: HAZARD clock-prefix-fp32 -- docs/internals/consumer_kernel.md#clock-prefix-fp32
            acc += cl;
            clk[t * CKP + i] = acc;
          }
          //: A ragged group's padded slots take a FINITE zero prefix so both
          //: panel factors stay finite (`keep^0 == 1`); their coefficient rows
          //: are exact zeros anyway. Aligned `nT == BT` makes this loop empty.
          for (int t = nT; t < BT; ++t) clk[t * CKP + i] = 0.0f;
          const float pr = 0.5f * acc;
          p_ref[i] = pr;
          s0[i] = expf(pr * log_keep[i]);
        }
      };
      //: The gathered accessor indexes the PARAMS directly (`tile_row` is the
      //: `bh` stream base there): capturing the per-level pointer ARRAY in a
      //: second lambda instantiation is the no-captured-arrays hazard, and the
      //: ratchet measured exactly that as D-scaled in-loop local growth.
      if (ga) {
        clock_prefix(
            [&](int l, int t, int i) {
              return p
                  .amp_clock[l][(tile_row + t_idx[t]) * p.level_pitch_clock[l] + dig[l * BC + i]];
            },
            gN);
      } else {
        clock_prefix([&](int l, int t,
                         int i) { return amp_c[l][t * p.level_pitch_clock[l] + dig[l * BC + i]]; },
                     BT);
      }
      __syncthreads();
      // ---- DECAY, step (c): the entry state enters the reference frame. `s0`
      // ---- multiplies the STATE, not the read panel; the mass column rides the same
      // ---- sweep because it is the same recurrence with `v := 1`. Row-per-warp rather
      // ---- than a flat `idx / cols` walk, because under `global` `cols = 65` is not a
      // ---- power of two: docs/internals/consumer_kernel.md#decay-clock
      for (int i = warp; i < BC; i += kWarps) {
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
        float s = 0.0f;
        if (!ga || t < gN) {
          const int token = ga ? t_idx[t] : t0 + t;
          //: The token's row under this visit's base: local `t` aligned, the
          //: global index gathered (`tile_row` is then the `bh` stream base).
          const int64_t tr = ga ? static_cast<int64_t>(token) : t;
          s = (token < p.T)
                  ? (p.g_read != nullptr ? p.g_read[static_cast<int64_t>(bh) * p.T + token] : 1.0f)
                  : 0.0f;
#pragma unroll
          for (int l = 0; l < D; ++l) {
            if (l >= j) break;
            //: FIXED levels are ALWAYS fp32 (Option B). On the split arm `amp_r`
            //: carries a spanned-scaled address for these slots, so the fp32
            //: stream is indexed directly from the params; this loop runs BT
            //: trips per tile, outside every hot loop.
            if constexpr (kAmpSplit) {
              s *= static_cast<const float*>(
                  p.amp_read[l])[(tile_row + tr) * p.level_pitch_read[l] + base[l]];
            } else {
              s *= amp_r[l][tr * p.level_pitch_read[l] + base[l]];
            }
          }
        }
        sr[t] = s;
      }
    }
    if (is_write) {
      for (int t = tid; t < BT; t += kConsumerThreads) {
        float s = 0.0f;
        if (!ga || t < gN) {
          const int token = ga ? t_idx[t] : t0 + t;
          const int64_t tr = ga ? static_cast<int64_t>(token) : t;
          s = (token < p.T) ? p.g_write[static_cast<int64_t>(bh) * p.T + token] : 0.0f;
#pragma unroll
          for (int l = 0; l < D; ++l) {
            if (l >= j) break;
            if constexpr (kAmpSplit) {
              s *= static_cast<const float*>(
                  p.amp_write[l])[(tile_row + tr) * p.level_pitch_write[l] + base[l]];
            } else {
              s *= amp_w[l][tr * p.level_pitch_write[l] + base[l]];
            }
          }
        }
        sw[t] = s;
      }
    }
    if (!DECAY && is_write) __syncthreads();
    // ---- THE READ PANEL (§3 steps 2-3). The loop walks FRAGMENT SLOTS, not
    // ---- `(token, leaf)` pairs, so the store is contiguous in `tid`:
    // ---- docs/internals/consumer_kernel.md#read-panel
    if (is_read) {
      //: ONE fragment-slot walk, two coefficient accessors -- the gathered one
      //: indexes the same product through `t_idx` and returns an exact zero for
      //: a ragged group's padded slots (its `clk` prefix is a finite zero, so
      //: the decay factor it skips is finite too).
      auto build_rf = [&](auto&& amp_read_at) {
#pragma unroll
        for (int mb = 0; mb < NM; ++mb) {
          unsigned* Rb = Rf + mb * kRfBlock;
          for (int sl = tid; sl < kRfBlock; sl += kConsumerThreads) {
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
      };
      if (ga) {
        //: Params-direct indexing, scalar captures only (no-captured-arrays):
        //: `tile_row` is the `bh` stream base on the gathered arm.
        build_rf([&](int t, int i) {
          if (t >= gN) return 0.0f;
          const int64_t tr = tile_row + t_idx[t];
          float a = 1.0f;
#pragma unroll
          for (int l = 0; l < D; ++l) {
            if (l < j) continue;  // fixed levels hoist into the row scale (§3 step 3)
            a *= amp_to_float(static_cast<const AmpRT*>(
                p.amp_read[l])[tr * p.level_pitch_read[l] + dig[l * BC + i]]);
          }
          if (DECAY && is_write) {
            a *= panel_read_factor(clk[t * CKP + i], p_ref[i], log_keep[i]);
          }
          return a;
        });
      } else {
        build_rf([&](int t, int i) {
          float a = 1.0f;
#pragma unroll
          for (int l = 0; l < D; ++l) {
            if (l < j) continue;  // fixed levels hoist into the row scale (§3 step 3)
            a *= amp_to_float(amp_r[l][t * p.level_pitch_read[l] + dig[l * BC + i]]);
          }
          if (DECAY && is_write) {
            a *= panel_read_factor(clk[t * CKP + i], p_ref[i], log_keep[i]);
          }
          return a;
        });
      }
    }
    // ---- THE WRITE PANEL'S PLANE CALL SITE, hoisted OUT of the window loop when the
    // ---- window IS the plane -- which is what keeps the `DECAY` arm's build ABOVE the
    // ---- `clk -> Vw` lifetime barrier, where `clk` is still live.
    //: HAZARD wt-build-site -- docs/internals/consumer_kernel.md#wt-build-site
    if constexpr (kWtPlane && kRing == 1) {
      if (is_write) {
        if (ga) {
          build_wt_rows(0, BC, [&](int l, int t, int i) {
            if (t >= gN) return 0.0f;
            return amp_to_float(static_cast<const AmpWT*>(
                p.amp_write[l])[(tile_row + t_idx[t]) * p.level_pitch_write[l] + dig[l * BC + i]]);
          });
        } else {
          build_wt_rows(0, BC, [&](int l, int t, int i) {
            return amp_to_float(amp_w[l][t * p.level_pitch_write[l] + dig[l * BC + i]]);
          });
        }
      }
    } else if constexpr (kWtPlane) {
      //: `kRing == 2` arms build the ALIGNED write panel at depth 2 in the walk;
      //: a GATHERED visit stages at the group top on every arm (it does not ride
      //: the ring), so its panel is built here, behind the same pre-GEMM barrier.
      if (ga && is_write) {
        build_wt_rows(0, BC, [&](int l, int t, int i) {
          if (t >= gN) return 0.0f;
          return amp_to_float(static_cast<const AmpWT*>(
              p.amp_write[l])[(tile_row + t_idx[t]) * p.level_pitch_write[l] + dig[l * BC + i]]);
        });
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
        float vv = 0.0f;
        if (!ga || t < gN) {
          const int token = ga ? t_idx[t] : t0 + t;
          vv = (token < p.T) ? amp_to_float(static_cast<const AmpWT*>(
                                   p.v)[(static_cast<int64_t>(bh) * p.T + token) * DV + ch])
                             : 0.0f;
        }
        stage_put(Vw[vw_index<BT, DV, kNarrowStage>(t, ch)], sw[t] * vv);
      }
    }
  };

  //: THE TOKEN-GATHER SCOPE, resolved once per CTA: launch-structural (both
  //: routing sides can zero -- `p.token_gather`, derived host-side from the
  //: activation registry) and read-pass only (the summary pass's per-leaf
  //: inclusion flags are tile-framed this increment). LAUNCH-UNIFORM, so the
  //: fork below is structure, not a distributional property.
  //: docs/internals/consumer_kernel.md#token-gather
  const bool tg = (p.token_gather != 0) && !summary;

  // ---- THE WALK: ONE visit body at ONE call site, TWO SCHEDULERS ahead of it.
  // ---- The aligned cursor and the gathered group assembler both advance to the
  // ---- next visit's coordinates `(tile, slot, is_read, is_write, gN)`; the body
  // ---- below is emitted ONCE, inline, so its accumulator arrays stay registers
  // ---- (a two-call-site body outlined and the ratchet refused it -- 444 arms of
  // ---- in-loop local growth). `tile >= 0` is an aligned tile (tokens
  // ---- `tile*BT + t`); `tile < 0` is a GATHERED group (tokens `t_idx[0..gN)`,
  // ---- slot 0, staged at the group top). The four GEMMs are token-oblivious;
  // ---- only the emission and the mass epilogue index tokens, through the same
  // ---- accessor the staging used. Gathered walk: docs/internals/consumer_kernel.md#token-gather
  //: HAZARD loop-induction -- docs/internals/consumer_kernel.md#loop-induction
  if constexpr (kRing == 2) {
    if (!tg) {
      const int first = next_cand(tile_lo);
      if (first < tile_hi) {
        stage_tile(first, 0, ((cand_read[first >> 5] >> (first & 31)) & 1u) != 0u,
                   ((cand_write[first >> 5] >> (first & 31)) & 1u) != 0u, BT);
      }
    }
  }
  int cur = tile_lo;  // both schedulers' source-tile cursor
  int visit = 0;      // aligned ring parity
  // The gathered assembler's carry: the word remainder of a split tile and the
  // group under fill. CTA-uniform values in registers -- every thread runs the
  // scan redundantly (all its loads are CTA-uniform addresses), so the group
  // boundaries are uniform with no broadcast and no per-tile barrier; only
  // warp 0 stores `t_idx`.
  int gtile = 0;
  uint32_t g_fr = 0u, g_fw = 0u, member = 0u;
  int fill = 0;
  bool gr = false, gw = false;
  //: The scan's fixed-level row indices, hoisted ONCE per CTA and STATICALLY
  //: indexed everywhere (unrolled-with-break, the row-scale loops' pattern), so
  //: the array stays in registers -- a runtime `l < j` loop reading `base[l]`
  //: per candidate tile put dynamic local traffic INSIDE the walk loop and the
  //: ratchet refused it.
  int srow[D];
#pragma unroll
  for (int l = 0; l < D; ++l) srow[l] = p.level_row_offset[l] + base[l];
  for (;;) {
    int tile, slot, gN;
    bool is_read, is_write;
    if (!tg) {
      const int nx = next_cand(cur);
      if (nx >= tile_hi) break;
      cur = nx + 1;
      tile = nx;
      is_read = ((cand_read[tile >> 5] >> (tile & 31)) & 1u) != 0u;
      is_write = ((cand_write[tile >> 5] >> (tile & 31)) & 1u) != 0u;
      slot = (kRing == 2) ? (visit & 1) : 0;
      gN = BT;
      ++visit;
    } else {
      // ---- THE GROUP ASSEMBLER (§8, incremental form). Membership = (AND of
      // ---- the owner's fixed-level token rows, per side, OR'd) AND the
      // ---- token-granular slice union -- ballot-time exclusion of slice-dead
      // ---- tokens is what keeps groups full. Source tiles ascend and each
      // ---- word is consumed LSB-first, so `t_idx` is GLOBALLY ascending: the
      // ---- intra Gram's causal mask is order-relative, so STABILITY IS THE
      // ---- CORRECTNESS REQUIREMENT. The word loop is bounded (a word splits
      // ---- across at most `ceil(32/BT) + 1` groups), never a per-token scan.
      while (fill < BT) {
        if (member == 0u) {
          const int nx = next_cand(cur);
          if (nx >= tile_hi) break;
          cur = nx + 1;
          gtile = nx;
          g_fr = 0xffffffffu;
          g_fw = 0xffffffffu;
          const int64_t slab = (static_cast<int64_t>(bh) * p.n_tiles + gtile) * p.total_rows;
#pragma unroll
          for (int l = 0; l < D; ++l) {
            if (l >= j) break;
            g_fr &= p.read_tok_bits[slab + srow[l]];
            g_fw &= p.write_tok_bits[slab + srow[l]];
          }
          member = (g_fr | g_fw)
                   & p.slice_union_tok[(static_cast<int64_t>(bh) * p.n_slices + slice) * p.n_tiles
                                       + gtile];
          continue;
        }
        const int room = BT - fill;  // >= 1: a full group ran and reset `fill`
        uint32_t taken = member;
        if (__popc(member) > room) {
          //: the (room+1)-th set bit; bits below it are exactly the lowest
          //: `room` live tokens -- the remainder waits for the next group.
          const unsigned cut = __fns(member, 0, room + 1);
          taken = member & ((1u << cut) - 1u);
        }
        if (warp == 0) {
          if (((taken >> lane) & 1u) != 0u) {
            t_idx[fill + __popc(taken & ((1u << lane) - 1u))] = gtile * BT + lane;
          }
        }
        gr = gr || (g_fr & taken) != 0u;
        gw = gw || (g_fw & taken) != 0u;
        fill += __popc(taken);
        member ^= taken;
      }
      //: The final flush is the ragged group (`fill < BT`); its padded slots
      //: stage exact zeros. `fill == 0` is the stream's end.
      if (fill == 0) break;
      tile = -1;
      slot = 0;
      is_read = gr;
      is_write = gw;
      gN = fill;
    }
    const bool ga = tile < 0;
    const int t0 = ga ? 0 : tile * BT;
    float* sr = sr_r0 + slot * 2 * BT;
    float* sw = sr + BT;
    unsigned* Rf = Rf_r0 + slot * kRfWords;
    StageT* Vw = Vw_r0 + slot * kVwSlot;

    // ---- THE TILE'S CONTROL BLOCK, RESET. It needs no barrier of its own on
    // ---- EITHER side: behind it stands the previous visit's end-of-visit
    // ---- `__syncthreads()`, ahead of it the unconditional staging publication.
    // ---- The edge DAG this reset serves, and the two-part always-runnable
    // ---- claim that makes the tile deadlock-free:
    // ---- docs/internals/pipeline.md#edge-dag
    if (tid < ctl_words(BC)) ctl[tid] = 0;
    if (ga) {
      __syncthreads();  // publish warp 0's `t_idx` writes to every staging loop
      stage_tile(-1, 0, is_read, is_write, gN);
    } else if constexpr (kRing == 1) {
      stage_tile(tile, 0, is_read, is_write, BT);
    }

    constexpr int kUnit = (READ_PREC == MmaPrec::kTf32x3) ? 8 : 16;
    // ---- THE WRITE PANEL AT DEPTH 2, AND IT COSTS NO BARRIER: the previous tile's
    // ---- end-of-tile barrier stands between the last fold read of `WT` and this
    // ---- write, and the publication rendezvous below between this write and the
    // ---- Gram's read. ALIGNED VISITS ONLY -- a gathered visit does not ride the
    // ---- ring and built its panel in `stage_tile`, behind the same barriers:
    // ---- docs/internals/consumer_kernel.md#wt-build-site
    if constexpr (kWtPlane && kRing == 2) {
      if (!ga && is_write) {
        const int64_t tile_row = (static_cast<int64_t>(bh) * p.n_tiles + tile) * BT;
        // `build_wt_rows` walks SPANNED levels only (`l < j` continues), so the
        // typed base is `AmpWT` for every level it touches.
        const AmpWT* amp_w[D];
#pragma unroll
        for (int l = 0; l < D; ++l) {
          amp_w[l] = static_cast<const AmpWT*>(p.amp_write[l]) + tile_row * p.level_pitch_write[l];
        }
        build_wt_rows(0, BC, [&](int l, int t, int i) {
          return amp_to_float(amp_w[l][t * p.level_pitch_write[l] + dig[l * BC + i]]);
        });
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
        mma_tile_fragA<BC, float, false, READ_PREC>(Rf, S, SP, BC, warp * 8, lane, y[0][0], y[0][1],
                                                    y[0][2], y[0][3]);
      }
    } else if (is_read) {
#pragma unroll
      for (int q = 0; q < kYTiles; ++q) {
        const int wt = warp + q * kWarps;
        if (wt >= NM * NT_DV) break;
        const int mb = wt / NT_DV, nt = wt - mb * NT_DV;
        mma_tile_fragA<BC, float, false, READ_PREC>(Rf + mb * kRfBlock, S, SP, BC, nt * 8, lane,
                                                    y[q][0], y[q][1], y[q][2], y[q][3]);
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
        for (int k = 0; k < BC; ++k) {
          acc += frag_a_get<READ_PREC>(Rb, tl, k) * S[k * SP + DV];
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
    constexpr int n_mt = BC / 16;
    for (int w = 0; w < kNWin; ++w) {
      if constexpr (!kWtPlane) {
        const int b_lo = w * kWinRows;
        const int b_hi = min(BC, b_lo + kWinRows);
        if (is_write && b_lo < b_hi) {
          // Spanned-only walk, as above: `AmpWT` for every level `build_wt_rows`
          // touches. The gathered accessor indexes the same product through
          // `t_idx` off the `bh` stream base, zeros on a ragged group's pad.
          const AmpWT* amp_w[D];
          const int64_t wrow = ga ? static_cast<int64_t>(bh) * p.n_tiles * BT
                                  : (static_cast<int64_t>(bh) * p.n_tiles + tile) * BT;
#pragma unroll
          for (int l = 0; l < D; ++l) {
            amp_w[l] = static_cast<const AmpWT*>(p.amp_write[l]) + wrow * p.level_pitch_write[l];
          }
          if (ga) {
            build_wt_rows(b_lo, b_hi, [&](int l, int t, int i) {
              if (t >= gN) return 0.0f;
              return amp_to_float(static_cast<const AmpWT*>(
                  p.amp_write[l])[(wrow + t_idx[t]) * p.level_pitch_write[l] + dig[l * BC + i]]);
            });
          } else {
            build_wt_rows(b_lo, b_hi, [&](int l, int t, int i) {
              return amp_to_float(amp_w[l][t * p.level_pitch_write[l] + dig[l * BC + i]]);
            });
          }
        }
        __syncthreads();
      }
      // ---- GEMM 2: the causal effective-score block, `G += R_q . W_q^T`, then §5's
      // ---- INCLUSIVE causal mask applied AFTER the loop, on the completed Gram.
      // ---- `B_T = false` because the transpose `W^T` names is in the STORAGE -- which
      // ---- is the whole point of building `WT` leaf-major:
      // ---- docs/internals/consumer_kernel.md#gemm-gram
      const int kw = min(kWinRows, BC - w * kWinRows);
      if constexpr (NM == 1) {
        if (mixed && warp < BT / 8 && kw > 0) {
          if constexpr (kNarrowStage) {
            mma_tile_fragA_wtB<kWinRows, BT>(Rf + w * (kWinRows / kUnit) * 128, WT, kw, warp * 8,
                                             lane, gd[0][0], gd[0][1], gd[0][2], gd[0][3]);
          } else {
            mma_tile_fragA<kWinRows, StageT, false, READ_PREC>(
                Rf + w * (kWinRows / kUnit) * 128, WT, BTP, kw, warp * 8, lane, gd[0][0], gd[0][1],
                gd[0][2], gd[0][3]);
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
            mma_tile_fragA<kWinRows, StageT, false, READ_PREC>(
                Rf + gmb * kRfBlock + w * (kWinRows / kUnit) * 128, WT, BTP, kw, gnt * 8, lane,
                gd[q][0], gd[q][1], gd[q][2], gd[q][3]);
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
            const int i0 = r0, i1 = r0 + 8;
            const float e0 = DECAY ? s0[i0] : 1.0f;
            const float e1 = DECAY ? s0[i1] : 1.0f;
            S[i0 * SP + n0] = e0 * (S[i0 * SP + n0] + d0);
            S[i0 * SP + n0 + 1] = e0 * (S[i0 * SP + n0 + 1] + d1);
            S[i1 * SP + n0] = e1 * (S[i1 * SP + n0] + d2);
            S[i1 * SP + n0 + 1] = e1 * (S[i1 * SP + n0 + 1] + d3);
          }
          if (GLOBAL) {
            const int q_lo = w * kWinRows;
            const int q_hi = min(BC, q_lo + kWinRows);
            for (int q = q_lo + tid; q < q_hi; q += kConsumerThreads) {
              float acc = 0.0f;
              for (int t = 0; t < BT; ++t)
                acc += mma_src(WT[wt_index<BT, kNarrowStage>(q - q_lo, t)]) * sw[t];
              S[q * SP + DV] = (DECAY ? s0[q] : 1.0f) * (S[q * SP + DV] + acc);
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
      if (!ga) {
        const int nxt = next_cand(tile + 1);
        if (nxt < tile_hi) {
          stage_tile(nxt, slot ^ 1, ((cand_read[nxt >> 5] >> (nxt & 31)) & 1u) != 0u,
                     ((cand_write[nxt >> 5] >> (nxt & 31)) & 1u) != 0u, BT);
        }
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
      //: A GATHERED visit's emission scatters through `t_idx` (the P14 microbench
      //: priced exactly this pattern: banded, DRAM-byte ratio 1.0009x at the
      //: measured gap). The `out == 0` early-out runs FIRST so a ragged group's
      //: padded slots (`sr == 0`, coefficient rows exactly zero) never read
      //: `t_idx` at all; gathered tokens are `< T` by construction (the support
      //: word's padded tail carries no bit).
      if constexpr (NM == 1) {
        if (warp < NT_DV) {
          CONSUMER_D_ROWS(lane);
          const int n0 = warp * 8;
          const int c0 = n0 + 2 * _r, c1 = c0 + 1;
          if (ga) {
            auto emit = [&](int t, int ch, float val) {
              const float out = sr[t] * val;
              if (out == 0.0f) return;
              atomicAdd(&p.num[(static_cast<int64_t>(bh) * p.T + t_idx[t]) * DV + ch], out);
            };
            emit(_g, c0, y[0][0]);
            emit(_g, c1, y[0][1]);
            emit(_g + 8, c0, y[0][2]);
            emit(_g + 8, c1, y[0][3]);
          } else {
            float* num_t0 = p.num + (static_cast<int64_t>(bh) * p.T + t0) * DV;
            auto emit = [&](int t, int ch, float val) {
              if (t0 + t >= p.T) return;
              const float out = sr[t] * val;
              if (out != 0.0f) atomicAdd(&num_t0[t * DV + ch], out);
            };
            emit(_g, c0, y[0][0]);
            emit(_g, c1, y[0][1]);
            emit(_g + 8, c0, y[0][2]);
            emit(_g + 8, c1, y[0][3]);
          }
        }
      } else if (ga) {
        CONSUMER_D_ROWS(lane);
        auto emit = [&](int t, int ch, float val) {
          const float out = sr[t] * val;
          if (out == 0.0f) return;
          atomicAdd(&p.num[(static_cast<int64_t>(bh) * p.T + t_idx[t]) * DV + ch], out);
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
          if (ga && t >= gN) continue;
          const int token = ga ? t_idx[t] : t0 + t;
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
          for (int i = tid; i < BC; i += kConsumerThreads) {
            if (s0[i] != 1.0f) sum_slot[i] = 1;
          }
        }
        if (DECAY) {
          for (int i = tid; i < BC; i += kConsumerThreads) {
            a_state[i] *= s0[i] * s0[i];
          }
        }
      }
    }
    __syncthreads();
    if (tg) {
      fill = 0;
      gr = false;
      gw = false;
    }
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
