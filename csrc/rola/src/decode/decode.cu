// THE T=1 DECODE STEP -- ONE KERNEL: fold, factor, walk. Token-stationary,
// state-streaming, lattice-walked. Owns the step kernel, its 16-way instantiation matrix
// and the decode path's host entry.
//
// The ratified shape, literally: the token's routing weights and value are RESIDENT
// (SMEM/registers), the STATE streams past, and the grid partitions the lattice's unit
// rank space.
//
// Three facts a reader must not miss:
//   1. There is one token, so the contraction index is the LEAF and the output index is
//      the COLUMN. Lanes map to OUTPUT COLUMNS: no MMA, and no warp reduction either --
//      the dot never crosses a lane.
//   2. DETERMINISM IS A SHIPPED PROPERTY HERE and it diverges from the prefill arm
//      ON PURPOSE: `y` is combined in SLOT ORDER, never by an `atomicAdd` race, because
//      decode's `y` picks the next token and a last-bit flip compounds
//      autoregressively. The state needs no argument at all -- the unit partition is a
//      PARTITION over owners and no atomic touches `state`.
//   3. THE PROLOGUE IS PER-CTA AND THAT IS THE DESIGN, AND IT IS ALSO THE ONLY PLACE A
//      BARRIER LIVES. Every split-K CTA folds and factors its own batch-head, so the
//      factor tables never leave shared memory and the split partition costs no launch;
//      below the prologue's last `__syncthreads` there is not one -- the walk never needed
//      one and the combine is an ARRIVAL COUNT:
//      docs/internals/decode/decode.md#per-cta-prologue,
//      docs/internals/decode/decode.md#the-warp-arrival
//
// The GEMV/GEMM argument, the walk's cost accounting against `N`, the measured
// reproducibility divergence, the barrier ledger, and the derivation that collapses the
// causal intra Gram into UPDATE-THEN-READ: docs/internals/decode/decode.md

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include "decode.cuh"
#include "decode_api.cuh"
#include "decode_fold.cuh"
#include "decode_lattice.cuh"
#include "dispatch_switch.cuh"

namespace rola {
namespace decode {
namespace detail {

//: `expf(88) = 1.65e38 < FLT_MAX`. THE SAME VALUE AND THE SAME CONVENTION as
//: `intra_panel.cuh`'s `kPanelMaxExponent`, restated because decode shares none of that
//: header's panel algebra. On this path the clamp is STRUCTURALLY INERT -- the exponent
//: cannot go positive -- and is written anyway so ONE convention governs every survival
//: exponential in the repository: docs/internals/decode/decode.md#decode-max-exponent
constexpr float kDecodeMaxExponent = 88.0f;

//: NOT an anonymous namespace: nvcc mangles anonymous-namespace kernels with a hash that
//: depends on the BUILD PATH, so a manifest ratified in one worktree certifies a name no
//: other checkout's build produces. A ratifiable kernel needs a named symbol.
namespace stampdet {

__device__ int64_t g_decode_stamp_probe;

//: THE STEP'S OWN FOOTPRINT, recompiled with this translation unit, so a test that reads
//: it back is reading a fact only the loaded binary can produce:
//: docs/internals/decode/decode_api.md#build-stamp
__global__ void decode_stamp_kernel(int64_t* out) {
  //: THE ENUMERATION'S SHAPE AND THE LAUNCH'S WIDTH ARE PART OF THE FOOTPRINT, alongside
  //: the two struct sizes and the span ceiling. The kind and term counts are what a walk's
  //: unit index MEANS, and the warp count is HOW MANY CTAs derive the step -- and a binary
  //: that disagrees about either is a different step however its structs measure. Both
  //: lessons were paid for: the union-of-products enumeration moved every walk in the
  //: extension without moving one byte of either struct, and the fat-CTA launch shape moved
  //: the derivation count from 256 to 80 without moving the enumeration. IT IS THE MOST
  //: SIGNIFICANT FIELD of the same positional scheme, so the three older fields still
  //: divide out unchanged and the inversion's plausibility bound still holds
  //: (`kKinds * kMaxTerms * kDecodeWarps` is 192 at eight warps, 768 at thirty-two).
  //: THE PARTITION'S GRANULARITY RIDES THE SAME FIELD, additively, because a binary that
  //: cuts the unit space at a different block width walks a different step however its
  //: width and its structs measure -- the round-4 lesson applied to the second of the two
  //: knobs the launch shape now carries:
  //: docs/internals/decode/decode_api.md#build-stamp
  *out = (static_cast<int64_t>(kKinds * kMaxTerms * kDecodeWarps + kUnitBlockWarps + kFieldWidthMax)
              * 4096
          + static_cast<int64_t>(sizeof(DecodeParams)))
             * 1000003
         + static_cast<int64_t>(sizeof(Unit<4>)) * 101 + static_cast<int64_t>(kDecodeMaxSpan);
  g_decode_stamp_probe = *out;
}

}  // namespace stampdet

template <int DV, int D, bool DECAY>
__global__ __launch_bounds__(kDecodeThreads, decode_step_blocks_per_sm(DECAY, D)) void
decode_step_kernel(DecodeParams p) {
  //: The mass column is UNCONDITIONAL (raw is dead), which is why the shape is a
  //: template axis and not a runtime test: docs/internals/decode/decode.md#template-axes
  constexpr int cols = DV + 1;
  //: Columns per lane: `NC` fully coalesced 128 B transactions, and the lane's
  //: accumulators are its own output columns.
  constexpr int NC = DV / 32;

  extern __shared__ char smem_raw[];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int seg = blockIdx.x;
  const int bh = blockIdx.y;
  const int b = bh / p.H;
  const int h = bh - b * p.H;
  const int total_rows = p.total_rows;

  // ---- SMEM ledger ----------------------------------------------------------
  //   sa_r, sa_w   [total_rows] fp32   both sides' amplitude rows, UNCOMPACTED and
  //                                    indexed by DIGIT: the digit IS the index.
  //   sd           [total_rows] fp32   this head's decay dials; DECAY only
  //   v_tok        [DV]         fp32   the token's value, resident
  //   fold  [warps][cols+3]     fp32   the cross-warp partials, bank-padded
  //   mask_r,mask_w [mask_total] u32   one bit per digit, per side
  //   omask  [4][omask_total]   u32    the four kinds' owner LIVENESS BITS, per level
  //   olist  [4][olist_total]   i32    those bits COMPACTED: the i-th live owner
  //   dlist  [dlist_total]      i32    the WRITE side's live digits, OUTER levels only
  float* sa_r = reinterpret_cast<float*>(smem_raw);
  float* sa_w = sa_r + total_rows;
  float* sd = sa_w + total_rows;
  float* v_tok = sd + (DECAY ? total_rows : 0);
  float* fold = v_tok + DV;
  uint32_t* mask_r = reinterpret_cast<uint32_t*>(fold + kDecodeWarps * (cols + 3));
  uint32_t* mask_w = mask_r + p.mask_total;
  uint32_t* omask = mask_w + p.mask_total;
  int32_t* olist = reinterpret_cast<int32_t*>(omask + kKinds * p.omask_total);
  int32_t* dlist = olist + kKinds * p.olist_total;

  __shared__ float s_sum[kMaxLevels];
  __shared__ typename LatticeScan::TempStorage scan_temp;
  __shared__ int s_running;
  __shared__ int s_n_o[kKinds * kMaxLevels];
  __shared__ int s_n_d[kMaxLevels];
  //: The retirement bound, folded by `publish_term_layout`'s own `atomicMax` -- which is why
  //: it is zeroed HERE, behind the fold's barrier: docs/internals/decode/decode.md#retirement
  __shared__ int s_umax;
  //: THE TERM TABLE, ONE PACKED WORD PER (term, level): the field's position, its width and
  //: the level's realized count in one read where three arrays used to be read separately.
  //: `D + 2` rows -- the walk's `D + 1` terms and the admission's candidate slot:
  //: docs/internals/decode/decode_lattice.md#the-terms
  __shared__ int s_fld[kMaxTerms * kMaxLevels];
  //: One unit count per TERM. The walk's outer loop is the term, so each term's rank space
  //: is partitioned across the grid on its own and the counts never need summing.
  __shared__ int s_units[kMaxTerms];
  //: THE CTA'S WARP ARRIVAL COUNT, and the whole of the combine's synchronisation. It is
  //: zeroed in the PROLOGUE, behind the prologue's own last barrier, which is what lets the
  //: combine below need no barrier at all: docs/internals/decode/decode.md#the-warp-arrival
  __shared__ int s_arrive;

  // ---- THE FOLD: every per-token operand, widened and staged, by this CTA ------
  //: TWO BARRIERS FOR THE WHOLE PROLOGUE. The read side's row sums are the only thing
  //: anything downstream waits on, so they take the first; everything else -- both sides'
  //: rows, the value, the dials -- is independent and shares the second.
  //: docs/internals/decode/decode_fold.md
  if (p.levels_bf16) {
    fold_row_sums<__nv_bfloat16>(p, b, h, p.read, p.normalize, s_sum);
  } else {
    fold_row_sums<float>(p, b, h, p.read, p.normalize, s_sum);
  }
  __syncthreads();
  if (p.levels_bf16) {
    fold_rows<__nv_bfloat16>(p, b, h, p.read, p.normalize, s_sum, sa_r);
    fold_rows<__nv_bfloat16>(p, b, h, p.write, nullptr, s_sum, sa_w);
  } else {
    fold_rows<float>(p, b, h, p.read, p.normalize, s_sum, sa_r);
    fold_rows<float>(p, b, h, p.write, nullptr, s_sum, sa_w);
  }
  if (p.v_bf16) {
    fold_value<__nv_bfloat16>(p, b, h, DV, v_tok);
  } else {
    fold_value<float>(p, b, h, DV, v_tok);
  }
  if (DECAY) {
    for (int l = 0; l < D; ++l) {
      const int width = p.level_width[l];
      const int off = p.level_amp_offset[l];
      const float* rd = p.dials + static_cast<int64_t>(h) * total_rows + p.level_row_offset[l];
      for (int g = tid; g < width; g += kDecodeThreads) sd[off + g] = rd[g];
    }
  }

  //: The two per-token row scales. At `T = 1` there is no owner rectangle, so what
  //: remains here is exactly the per-TOKEN part -- the side gains; the per-LEVEL factors
  //: the tiled kernel hoists stay in the per-LEAF product.
  //: docs/internals/decode/decode.md#row-scales
  const float sr = 1.0f;  // global fixes g_read := 1 (raw is dead)
  const float sw = p.g_bf16 ? fold_gain<__nv_bfloat16>(p, b, h) : fold_gain<float>(p, b, h);
  __syncthreads();

  // ---- THE FACTOR TABLES ------------------------------------------------------
  //: `sum_l B_l` bits and at most `sum_l g_l` entries: nothing in this prologue is sized
  //: in the leaf space: docs/internals/decode/decode_lattice.md
  build_digit_mask<D>(p, sa_r, mask_r);
  build_digit_mask<D>(p, sa_w, mask_w);
  if (tid < kKinds * kMaxLevels) s_n_o[tid] = 0;
  if (tid < kMaxLevels) s_n_d[tid] = 0;
  if (tid == 0) s_umax = 0;
  __syncthreads();
  build_owner_mask<D>(p, mask_r, mask_w, omask, s_n_o, s_n_d);
  __syncthreads();
  compact_owner_lists<D>(p, omask, mask_w, olist, dlist);
  __syncthreads();
  publish_term_layout<D>(p, s_n_o, s_n_d, s_fld, s_units, &s_umax);

  // ---- RETIREMENT: the split's ceiling is captured, the realized work is not ----
  //: `n_split` is FROZEN for the sequence and baked into the captured launch
  //: ([`decode.md#the-split-policy`](#the-split-policy)), and it is sized from the DENSE
  //: unit ceiling because that is the only count a host can know. The count this step
  //: ACTUALLY enumerates is a property of ITS OWN SUPPORT and exists only here, once the
  //: factor tables do. So the ceiling is what the host captures and the DEVICE is what
  //: spends it: a segment with no unit in any term retires, and it retires COMPLETELY --
  //: no walk, no partial, no arrival, and it is not summed.
  //: THE LIVE SEGMENTS ARE A PREFIX. A segment owns the `kUnitBlockWarps`-wide block at
  //: `seg * kUnitBlockWarps` of every period, so it has an iteration exactly when
  //: `seg * kUnitBlockWarps < max_t units[t]`,
  //: and `n_live` is that bound clamped into `[1, n_split]`. Every CTA of the batch-head
  //: derives it from the same published counts, so the arrival target and the summed slot
  //: range agree by construction and no CTA has to be told.
  //: IT IS NOT A THRESHOLD -- it is the walk's own loop bound, asked before the loop.
  //: AT LEAST ONE, ALWAYS: a batch-head whose support is empty still owes a `y`, and `y`
  //: is the ratio `0 / (0 + eps)` that segment 0 writes.
  //: IT RETIRES BEFORE THE ADMISSION, and that ordering is load-bearing rather than
  //: incidental: the table test and the pool claim read a page table the last-arriving
  //: LIVE segment folds into, so a retired segment must not be inside that window at all.
  //: docs/internals/decode/decode.md#retirement
  //: ONE SHARED READ AND ONE COMPARE. `build_term_layout` folds the maximum by `atomicMax`
  //: inside its own barrier, so learning it costs neither a second block synchronisation
  //: (which the dense cell, where nothing retires, measured at ~1 %) nor `D + 1` redundant
  //: reads per thread. The test is written as the COMPARE rather than as `seg >= n_live`
  //: because they are the same question -- `seg >= clamp(ceil(umax / U), 1, n_split)` is
  //: `seg > 0 && seg * U >= umax` at `U = kUnitBlockWarps`, since `seg < n_split` always --
  //: and this form keeps the
  //: division and the two clamps off the path every CTA walks, on the cell where nothing
  //: retires. `n_live` itself is only needed by the epilogue.
  const int umax = s_umax;
  if (seg > 0 && seg * kUnitBlockWarps >= umax) return;

  const int inner_bits = p.lat_inner_bits;

  //: THE CANDIDATE ATOMS, enumerated from the same lattice product the walk uses, and a
  //: PARTITION of the write side's atoms with no straddle case at all -- which is what the
  //: pool claim's canonical ranks need. A unit's leaves are a CONTIGUOUS ALIGNED run, so an
  //: atom and a unit nest one way or the other and the index says which: the low
  //: `atoms_per_unit_shift` bits pick an atom INSIDE a unit, and `lat_units_per_atom_shift`
  //: consecutive units make ONE atom. Exactly one of the two is non-zero.
  //: THE WINDOW IS THE UNIT'S RUN, CLIPPED TO AN ATOM. At the flagship spans `(8, 16)` a
  //: unit IS an atom: `n_units` and the atom index are both one, and membership is the
  //: unit's write slice being non-zero.
  //: IT NEEDS NO SECOND ENUMERATION AT ALL, because TERM 0 IS THE WRITE PRODUCT: the walk's
  //: own first term is the write side, in its own order, with `live_w`/`cw` the
  //: conditioning that term carries. docs/internals/decode/decode.md#candidate-atoms
  const int win_bits = inner_bits < kAtomShift ? inner_bits : kAtomShift;
  const uint32_t win_mask = (1u << (1 << win_bits)) - 1u;
  const int n_units = 1 << p.lat_units_per_atom_shift;
  //: THE CANDIDATE SLOT'S ROW of the same packed term table -- the ADMISSION's rank space,
  //: which is the write product in the OLD owner-and-run order because only that order is
  //: monotone in atom id: docs/internals/decode/decode_lattice.md#the-candidate-slot
  const int32_t* cand_fld = s_fld + (D + 1) * kMaxLevels;
  auto candidate_atom = [&](int i) -> int {
    const int a = i & ((1 << p.atoms_per_unit_shift) - 1);
    const int unit0 = (i >> p.atoms_per_unit_shift) << p.lat_units_per_atom_shift;
    int atom = -1;
    for (int q = 0; q < n_units; ++q) {
      const CandidateUnit u = resolve_candidate_unit<D>(p, mask_w, olist, cand_fld, unit0 + q);
      if (!u.valid) return -1;
      if (atom < 0) atom = (u.base + (a << kAtomShift)) >> kAtomShift;
      if (u.live_w && ((u.cw >> ((a << kAtomShift) & ((1 << inner_bits) - 1))) & win_mask) != 0u) {
        return atom;
      }
    }
    return -1;
  };

  //: FROM THE CANDIDATE SLOT, not from the walk's write term: the two enumerate the same
  //: set and only this one is monotone in atom id, which is what the claim's canonical
  //: ranks are defined in: docs/internals/decode/decode_lattice.md#the-write-grain
  const int n_candidates = (s_units[D + 1] >> p.lat_units_per_atom_shift) << p.atoms_per_unit_shift;

  // ---- THE ADMISSION QUESTION, asked of the map this CTA already holds ---------
  //: Three outcomes PER BATCH-HEAD: advance, no-op, or re-read without depositing:
  //: docs/internals/decode/decode.md#absorbed-verdict
  //: NO 64-BIT ADDRESS MAY SURVIVE INTO THE WALK -- each place that needs this
  //: batch-head's table row forms it again, at two registers of the declared residency:
  //: docs/internals/decode/decode.md#declared-residency
  const bool paged = p.page_tbl != nullptr;
  const bool done = paged && p.done[bh] != 0;
  const bool deposit = !done;
  bool claimed = false;
  bool needy = false;
  if (paged && !done) {
    const int64_t row_base = static_cast<int64_t>(bh) * p.atoms_per_bh;
    const int32_t* page_row = p.page_tbl + row_base;
    //: THE TABLE TEST: how many atoms this batch-head's write side reaches that the page
    //: table does not map. Every CTA of the batch-head computes it independently from the
    //: same two objects -- its own factor tables and a table no launch mutates until
    //: every CTA of that batch-head has arrived -- so the answer is a RECOMPUTE and never
    //: a communication. docs/internals/decode/decode.md#the-table-test
    int found = 0;
    for (int i = tid; i < n_candidates; i += kDecodeThreads) {
      const int a = candidate_atom(i);
      if (a >= 0 && page_row[a] < 0) ++found;
    }
    int excl = 0;
    int demand = 0;
    LatticeScan(scan_temp).ExclusiveSum(found, excl, demand);
    __syncthreads();
    if (demand > 0) {
      const int cursor = (p.pool_cap > 0) ? p.pool_cursor[bh] : 0;
      claimed = demand <= p.pool_cap - cursor;
      needy = !claimed;
      if (claimed) {
        //: THE CLAIM, replicated and canonical: the batch-head's touched-unmapped atoms
        //: take pool slots in ASCENDING ATOM ID from its own cursor, so every CTA of the
        //: batch-head derives the identical mapping with no atomic and no arrival.
        //:
        //: IT WRITES THE CLAIMS AND NOTHING ELSE. Not the page table, because the table is
        //: the claim's own predicate and a prologue write would move a sibling CTA's ranks;
        //: and not a whole shadow ROW either, which is the defect this deletes.
        //: THE ROW-COPY WAS A CROSS-CTA RACE. Every CTA of the batch-head copied the table
        //: into this row -- writing `-1` for every atom the table does not map yet -- while
        //: its siblings were already writing their claimed SLOTS into the same addresses,
        //: with no ordering between CTAs at all. The barrier below orders one CTA's copy
        //: against its own claim writes and says nothing about anyone else's: a sibling's
        //: `-1` could land on top of a claimed slot, and a third CTA reading that address
        //: in the window took the ABSENT path and skipped its deposit. The final value was
        //: always right, which is what the old comment's "identical mapping" was true of;
        //: the TRANSIENT was not, and correctness that survives only because one wave of
        //: CTAs happens to march in step is not correctness.
        //: SO THE ROW IS CLAIMS-ONLY, every write is the same value from every CTA, and the
        //: walk reads TWO SOURCES instead of one (`decode.md#the-claim`). It also deletes an
        //: `atoms_per_bh`-element bulk copy per CTA per claiming step.
        int32_t* claim_row = p.pool_map + row_base;
        const int32_t* slots = p.pool_slots + static_cast<int64_t>(bh) * p.pool_cap;
        if (tid == 0) s_running = 0;
        __syncthreads();
        for (int base = 0; base < n_candidates; base += kDecodeThreads) {
          const int i = base + tid;
          const int a = (i < n_candidates) ? candidate_atom(i) : -1;
          const bool want = a >= 0 && page_row[a] < 0;
          int rank = 0;
          int aggregate = 0;
          LatticeScan(scan_temp).ExclusiveSum(want ? 1 : 0, rank, aggregate);
          if (want) claim_row[a] = slots[cursor + s_running + rank];
          __syncthreads();
          if (tid == 0) s_running += aggregate;
          __syncthreads();
        }
      }
    }
  }
  //: THE CONDENSATION, published UNCONDITIONALLY by one CTA per batch-head: a dead
  //: atom's `false` is as much of the set as a live one's `true`, and publishing only
  //: the growing batch-heads would leave the buffer carrying an older step's answer for
  //: the others -- which is the set a host admission then plans from.
  if (paged && seg == 0) {
    bool* atom_row = p.atom_bits + static_cast<int64_t>(bh) * p.atoms_per_bh;
    for (int a = tid; a < p.atoms_per_bh; a += kDecodeThreads) atom_row[a] = false;
    __syncthreads();
    for (int i = tid; i < n_candidates; i += kDecodeThreads) {
      const int a = candidate_atom(i);
      if (a >= 0) atom_row[a] = true;
    }
  }
  //: THE PROLOGUE ENDS ON THIS BARRIER AND THERE IS NO OTHER BELOW IT. It publishes the
  //: claim row to the walk and the zeroed warp arrival count to the combine -- shared memory
  //: is not zeroed at CTA start, and this is the one place a zero can be published for free.
  //: docs/internals/decode/decode.md#no-barrier
  if (tid == 0) s_arrive = 0;
  __syncthreads();

  //: A NEEDY BATCH-HEAD RUNS NOTHING: no walk, no `y`, no state, no counter beyond its
  //: arrival -- which is what lets the step be enqueued before the admission it needs is
  //: known, and what makes the caller's replay THE SAME STEP rather than a second one.
  //: docs/internals/decode/decode.md#per-bh-verdict
  //: `n_live` -- the same bound the retirement test above asked as a COMPARE, formed here
  //: because the combine and the arrival need the NUMBER rather than the answer.
  const int want_live = ceil_div_int(umax, kUnitBlockWarps);
  const int n_live = want_live < 1 ? 1 : (want_live > p.n_split ? p.n_split : want_live);

  //: THE WARP'S OWN PARTIAL, and it exists whether or not the warp walks anything: the
  //: combine is an ARRIVAL COUNT, so every warp of a live CTA deposits and arrives, and a
  //: warp whose units are all beyond this term's count deposits exact zeros. Retirement at
  //: warp grain is therefore the walk it does not run, never an early `return`:
  //: docs/internals/decode/decode.md#the-warp-arrival
  float acc[NC];
#pragma unroll
  for (int q = 0; q < NC; ++q) acc[q] = 0.0f;
  float acc_m = 0.0f;

  if (!needy) {
    //: THE PAGED BASE TRANSLATION, resolved ONCE PER TOUCHED ATOM. The table row is
    //: pinned here (a plain origin, no per-atom widening multiply); the cache is keyed on
    //: the atom VALUE, so it is correct whatever order the walk reaches leaves in.
    //: `atom_base` is the atom's first row as an ELEMENT offset from `p.state`: dense is
    //: `bh * N + (atom << 4)` (which is why a topology whose `N` is not a whole number of
    //: atoms still addresses correctly -- the two halves recombine to `bh * N + leaf`),
    //: paged is `slot << 4`. docs/internals/decode/decode.md#paged-address
    //: A CLAIMING STEP READS TWO SOURCES: the page table first, and this step's CLAIM ROW
    //: only where the table says absent. That is the whole of the pool inside the walk, and
    //: it is the shape the race-free claim row forces -- the row no longer carries the
    //: table's own entries, so it cannot answer for an atom it did not claim.
    //: THE TABLE COMES FIRST BECAUSE IT ANSWERS ALMOST EVERY ATOM: a claiming step's newly
    //: claimed atoms are by construction the ones the table does NOT map, so the fallback is
    //: taken only on them and every resident atom costs one load, as before.
    //: THE TWO CAN NEVER DISAGREE. `_admit_pages` writes only atoms the table has as absent
    //: and there is no eviction, so a slot never moves once assigned; a claim row entry from
    //: an earlier step therefore equals the table's. The one path that could break that is
    //: `PageArena.reset`, which clears the table and rewinds the allocator -- so it now
    //: clears this row too (`rola/ops/paging.py`). docs/internals/decode/decode.md#the-claim
    const int32_t* page_row =
        !paged ? nullptr : p.page_tbl + static_cast<int64_t>(bh) * p.atoms_per_bh;
    const int32_t* claim_row =
        (paged && claimed) ? p.pool_map + static_cast<int64_t>(bh) * p.atoms_per_bh : nullptr;
    const int64_t dense_bh_base = static_cast<int64_t>(bh) * p.N;
    int cached_atom = -1;
    int64_t atom_base = 0;
    bool atom_present = false;

    // ---- THE WALK ---------------------------------------------------------------
    //: THE UNIT IS THE (owner, outer-run) PAIR and the grid partitions ITS OWN index
    //: space -- not the owner's -- so every touched leaf has exactly one owning CTA, warp
    //: and lane and the state needs no atomic:
    //: docs/internals/decode/decode.md#the-unit-partition
    //: A WARP TAKES A WHOLE UNIT, and a unit is the DRAM granule itself: under the
    //: mixed-radix order a unit's leaves are the innermost level's contiguous run, which at
    //: the flagship spans is one 16-leaf ATOM. Owner-contiguity across units buys nothing
    //: on top of that -- the sectors are already whole -- while partitioning at owner grain
    //: would cap the walk's parallelism at the LIVE OWNER COUNT and give each warp the
    //: whole `BC`-leaf box to walk serially. The CTA's warps take CONSECUTIVE units, so a
    //: CTA still covers one contiguous stretch of the plane per pass.
    //: EVERY LEVEL BUT THE INNERMOST LEAVES THE SLOT LOOP AND THAT IS THE UNIT'S POINT:
    //: a unit fixes their digits, so their amplitude product is formed ONCE, in
    //: `resolve_unit`, and every slot below costs ONE innermost amplitude rather than `D`.
    //: THE OUTER LOOP IS THE TERM, and it is a LATENCY decision: every level's kind is
    //: loop-invariant inside it, so the list bases, the counts, the field widths and the
    //: membership predicates all hoist and the per-unit dependent chain is the union walk's.
    //: The alternative -- one index space, term named per unit -- was BUILT AND MEASURED at
    //: `+2.3 ns` per unit, which cost the scattered cell more than the decomposition saves.
    //: docs/internals/decode/decode.md#the-term-loop
    for (int term = 0; term <= D; ++term) {
      //: THE TERM'S ROW OF THE PACKED TABLE, hoisted once per term: every level's kind,
      //: field width, field position and realized count is loop-invariant inside the term,
      //: so the resolver below reads ONE word per level instead of three:
      //: docs/internals/decode/decode_lattice.md#the-terms
      const int units = s_units[term];
      const int32_t* fld = s_fld + term * kMaxLevels;
      //: THE BLOCK, THE REPEAT AND THE PERIOD. A warp's low `kUnitBlockWarpsBits` name its
      //: unit inside the segment's block; the rest name WHICH of the CTA's `kUnitRepeats`
      //: blocks it serves, one period apart. `seg * kUnitBlockWarps + ub` sweeps one whole
      //: period bijectively, so the partition is the same one-owner-per-row partition it
      //: was, cut at a granularity the CTA's width no longer sets.
      const int period = p.n_split * kUnitBlockWarps;
      const int ub = warp & (kUnitBlockWarps - 1);
      const int ur = warp >> kUnitBlockWarpsBits;
      for (int unit = seg * kUnitBlockWarps + ub + ur * period; unit < units;
           unit += period * kUnitRepeats) {
        //: TWO RESOLVERS, ONE WALK. The write product is enumerated from live DIGITS and
        //: every other term from live owners crossed with the run radix; both hand back the
        //: identical `Unit<D>`, and the selector is the term loop's own invariant.
        //: docs/internals/decode/decode_lattice.md#the-write-grain
        const Unit<D> u = (term == 0) ? resolve_write_unit<D, DECAY>(p, mask_r, mask_w, sa_r, sa_w,
                                                                     sd, olist, dlist, fld, unit)
                                      : resolve_unit<D, DECAY>(p, mask_r, mask_w, sa_r, sa_w, sd,
                                                               olist, fld, term, unit);
        const uint32_t br = u.live_r ? u.cr : 0u;
        const uint32_t bw = u.live_w ? u.cw : 0u;
        //: THE ENUMERATED RUN IS THIS TERM'S. Term 0 walks the write run; every other term
        //: is disjoint from `supp(W)` by construction and walks the read remainder it names.
        //: A UNION `br | bw` WOULD DOUBLE-COUNT: a write unit whose innermost read digits
        //: leave `W` reaches leaves that the innermost split term also enumerates.
        //: `valid` needs no test here -- `resolve_unit` has already zeroed the slices of a
        //: unit its term does not realize. docs/internals/decode/decode_lattice.md#the-unit
        uint32_t bits = (term == 0) ? bw : br;

        //: THE SLOT LOOP IS THE INNERMOST LEVEL'S RUN, in ASCENDING lattice order -- the
        //: unit's own contiguous run -- iterated over the TERM's set bits, so a dead
        //: innermost digit costs one `__ffs` step and no address at all. Ascending is what
        //: makes `cached_atom` the walk's only reuse and enough of it:
        //: docs/internals/decode/decode.md#aligned-subranges
        //: NO SPECULATIVE UNROLLING: the trip count is a runtime popcount, and an unrolled
        //: body duplicates every live value of the walk's tightest scope.
#pragma unroll 1
        while (bits != 0u) {
          const int t = __ffs(static_cast<int>(bits)) - 1;
          bits &= bits - 1u;
          const bool in_r = ((br >> t) & 1u) != 0u;
          const bool in_w = ((bw >> t) & 1u) != 0u;

          const int off_last = u.dlast + t;
          const float ar = u.ar * sa_r[off_last];
          const float aw = u.aw * sa_w[off_last];
          const float rate = DECAY ? u.rate * sd[off_last] : 1.0f;

          const int leaf = u.base + t;
          const int atom = leaf >> kAtomShift;
          if (atom != cached_atom) {
            cached_atom = atom;
            if (page_row != nullptr) {
              int slot = page_row[atom];
              if (slot < 0 && claim_row != nullptr) slot = claim_row[atom];
              atom_present = slot >= 0;
              atom_base = static_cast<int64_t>(slot << kAtomShift) * cols;
            } else {
              atom_present = true;
              atom_base = (dense_bh_base + (atom << kAtomShift)) * cols;
            }
          }

          //: AN ABSENT ATOM READS AS ZEROS AND IS NOT WRITTEN. Exact, not defensive: the
          //: plan admits the step's write set exactly, so an atom with no slot is outside
          //: it, and a never-deposited leaf holds zeros in the dense backing too. The
          //: self-normalizing readout then contributes exactly nothing through it.
          //: docs/internals/decode/decode.md#absent-atom
          float* row = atom_present ? (p.state + atom_base + (leaf & kAtomMask) * cols) : nullptr;
          float sv[NC];
#pragma unroll
          for (int q = 0; q < NC; ++q) sv[q] = 0.0f;
          float smass = 0.0f;
          if (atom_present) {
#pragma unroll
            for (int q = 0; q < NC; ++q) sv[q] = row[lane + 32 * q];
            if (lane == 0) smass = row[DV];
          }

          //: UPDATE FIRST, THEN READ THE UPDATED ROW -- AND THAT ORDER IS THE SEMANTICS,
          //: not an optimization: survival is INCLUSIVE at `j = t`, so the whole causal
          //: intra Gram collapses into reading the row after it is written. A row in
          //: `supp(R) \ supp(W)` has `keep == 1` and no deposit, so the same expression is
          //: its readout too -- one path, no arm.
          //: A DONE BATCH-HEAD TAKES THE READ AND NOT THE DEPOSIT, and reproduces its `y`
          //: BIT FOR BIT while doing so: docs/internals/decode/decode.md#done-flags
          //: HAZARD update-then-read -- docs/internals/decode/decode.md#update-then-read
          if (in_w && atom_present && deposit) {
            //: DECAY AT READ, VIA THE DETACHED CLOCK, WITH NO SWEEP AND NO EXPONENT
            //: SPLIT: at `T = 1` the tiled kernel's rule-1 half-mass shift is exactly the
            //: identity, so ONE `expf` replaces its three. `log1pf` IS the composition
            //: (never a rounded `1 - rate`) and there is no epsilon floor -- both of that
            //: rule's rule 2's prohibitions: docs/internals/decode/decode.md#decay-at-read
            float keep = 1.0f;
            if (DECAY) {
              //: THE CLOCK IS THE STORED WRITE ALLOCATION ITSELF -- no side gain, which is
              //: what makes "stored != 0 => live != 0" an identity:
              //: docs/internals/decode/decode.md#detached-clock
              keep = expf(fminf(aw * log1pf(-rate), kDecodeMaxExponent));
            }
            const float wv = aw * sw;
#pragma unroll
            for (int q = 0; q < NC; ++q) {
              sv[q] = keep * sv[q] + wv * v_tok[lane + 32 * q];
              row[lane + 32 * q] = sv[q];
            }
            //: The mass column is the `v := 1` column of the identical recurrence.
            if (lane == 0) {
              smass = keep * smass + wv;
              row[DV] = smass;
            }
          }
          if (in_r) {
#pragma unroll
            for (int q = 0; q < NC; ++q) acc[q] += ar * sv[q];
            if (lane == 0) acc_m += ar * smass;
          }
        }
      }
    }
  }

  // ---- THE COMBINE: one deposit, one arrival, one summing warp ----------------
  //: ZERO `__syncthreads` BELOW THE PROLOGUE, and this is the mechanism that removes the
  //: last of them. Each warp deposits its `cols` partials into ITS OWN shared row, fences,
  //: and bumps the CTA's arrival count with one atomic; the LAST ARRIVER -- and no barrier
  //: -- is what makes "every row is written" true, and it alone sums them.
  //: DETERMINISM SURVIVES BECAUSE THE ORDER IS FIXED, not because the arrivals are: the sum
  //: runs over warps `0 .. kDecodeWarps - 1` per column, exactly the order the barrier-stepped
  //: tree it replaces used, so `y` is byte-reproducible at a fixed `n_split` as before.
  //: EVERY LANE FENCES ITS OWN STORES before the one-lane arrival, and every lane fences
  //: again after the broadcast before it reads a sibling's row (KERNEL_STANDARDS 12: a fence
  //: orders only the fencing thread).
  //: HAZARD fold-bank-padding -- docs/internals/decode/decode.md#fold-bank-padding
  const int fold_pitch = cols + 3;
  float* my_row = fold + warp * fold_pitch;
#pragma unroll
  for (int q = 0; q < NC; ++q) my_row[lane + 32 * q] = acc[q];
  if (lane == 0) my_row[DV] = acc_m;
  __threadfence_block();
  __syncwarp();
  int arrived = 0;
  if (lane == 0) arrived = atomicAdd(&s_arrive, 1);
  arrived = __shfl_sync(0xffffffffu, arrived, 0);
  if (arrived != kDecodeWarps - 1) return;
  __threadfence_block();

  //: THE MASS COLUMN IS READ BY EVERY LANE, not by lane 0 alone: the value is the same
  //: broadcast load for all of them and it is the denominator every lane divides by, so
  //: carrying it in all 32 registers costs one load and saves the shuffle.
  float part[NC];
#pragma unroll
  for (int q = 0; q < NC; ++q) part[q] = 0.0f;
  float part_m = 0.0f;
#pragma unroll
  for (int wq = 0; wq < kDecodeWarps; ++wq) {
    const float* row = fold + wq * fold_pitch;
#pragma unroll
    for (int q = 0; q < NC; ++q) part[q] += row[lane + 32 * q];
    part_m += row[DV];
  }
#pragma unroll
  for (int q = 0; q < NC; ++q) part[q] *= sr;  // sr == 1 under global; the one scale site
  part_m *= sr;

  // ---- the deterministic split-K combine, and the fused divide ---------------
  //: OVER THE LIVE SEGMENTS, NOT OVER THE CEILING -- and this is where retirement is
  //: actually paid for. The per-segment cost a large split carries is the ARRIVAL AND
  //: THE COMBINE (`n_split` atomics on one counter, then `n_split` `__ldcg` loads per
  //: column on ONE warp), not the replicated prologue: a variant that retired the walk
  //: but still wrote exact zeros and arrived was BUILT AND MEASURED AT ZERO. The order
  //: is still a FIXED ASCENDING SLOT ORDER over a range every CTA of the batch-head
  //: derives identically from the published counts, which is the whole of G1; `y` moves
  //: across `n_split` exactly as it always did (fp32 addition is not associative) and is
  //: byte-identical at a fixed one. docs/internals/decode/decode.md#slot-order
  //: A NEEDY BATCH-HEAD'S CTAs ARRIVE ANYWAY and deposit nothing. The arrival is what
  //: elects one CTA to publish the verdict below, and electing it from the same counter the
  //: combine uses is what makes "no CTA of this batch-head is still reading" the SAME
  //: argument in both cases: docs/internals/decode/decode.md#the-last-arriver
  if (n_live > 1) {
    if (!needy) {
      float* slot = p.ws + (static_cast<int64_t>(bh) * p.n_split + seg) * cols;
#pragma unroll
      for (int q = 0; q < NC; ++q) slot[lane + 32 * q] = part[q];
      if (lane == 0) slot[DV] = part_m;
    }
    __threadfence();
    __syncwarp();
    int last_cta = 0;
    if (lane == 0) last_cta = (atomicAdd(&p.ctr[bh], 1) == n_live - 1) ? 1 : 0;
    last_cta = __shfl_sync(0xffffffffu, last_cta, 0);
    if (last_cta == 0) return;
    __threadfence();

    //: SLOT ORDER, always -- the whole determinism argument. `__ldcg` bypasses L1 so
    //: the reader sees the stores the `__threadfence` above made visible.
    //: HAZARD slot-order -- docs/internals/decode/decode.md#slot-order
    if (!needy) {
      const float* base = p.ws + static_cast<int64_t>(bh) * p.n_split * cols;
#pragma unroll
      for (int q = 0; q < NC; ++q) part[q] = 0.0f;
      part_m = 0.0f;
      //: UNROLLED FOR MEMORY-LEVEL PARALLELISM, and the factor is a MEASURED one. The trip
      //: count is dynamic, so KERNEL_STANDARDS 6 asks for the pragma to be explicit -- and
      //: at `1` it is a correctness defect on this loop, not a style choice: the `n_live`
      //: loads are INDEPENDENT (only the adds chain), and forbidding the unroll leaves one
      //: load in flight, which measured 26.6 us of a 93 us dense step on ONE warp of ONE CTA
      //: while 79 SMs idled -- 50 % of the scattered step.
      //: docs/internals/decode/decode.md#slot-order
#pragma unroll 8
      for (int sl = 0; sl < n_live; ++sl) {
        const float* row = base + static_cast<int64_t>(sl) * cols;
#pragma unroll
        for (int q = 0; q < NC; ++q) part[q] += __ldcg(row + lane + 32 * q);
        //: THE MASS COLUMN IS ONE LANE'S HERE, not every lane's -- the opposite of the
        //: CTA fold above, and for the opposite reason: this loop's length IS the step's
        //: tail, so a broadcast load every lane repeats is a third of the only critical
        //: path left in the kernel. It is broadcast once, at the divide.
        if (lane == 0) part_m += __ldcg(row + DV);
      }
    }
    //: Reset for the next step: no `memset` launch between two decode steps, and the
    //: counter is safe to capture in a CUDA graph.
    if (lane == 0) p.ctr[bh] = 0;
  }
  if (!needy) {
    const float den = __shfl_sync(0xffffffffu, part_m, 0) + p.eps;
#pragma unroll
    for (int q = 0; q < NC; ++q) {
      p.y[static_cast<int64_t>(bh) * DV + lane + 32 * q] = part[q] / den;
    }
  }

  if (!paged) return;

  // ---- the per-batch-head publication, by its LAST-ARRIVING WARP --------------
  //: EVERY MUTATION A SIBLING CTA COULD HAVE READ HAPPENS HERE AND NOWHERE ELSE, which
  //: is why it is safe: docs/internals/decode/decode.md#the-last-arriver
  //: IT IS ONE WARP'S WORK NOW, not the CTA's, and that is the barrier deletion's price
  //: paid where it is cheapest: the CTA's other warps have already retired at the arrival
  //: count, so the fold is lane-strided over the page table row and its size is folded by a
  //: warp shuffle instead of a block scan. It runs only on a CLAIMING step.
  if (claimed) {
    //: THE CLAIM'S SIZE FALLS OUT OF THE FOLD ITSELF -- an atom the shadow row maps and
    //: the table does not IS a claimed atom -- so it is neither carried across the walk
    //: nor re-derived from the write-support enumeration. Both of those keep values alive
    //: through the loop the register budget is tight in, and both were measured to spill:
    //: docs/internals/decode/decode.md#declared-residency
    const int64_t row_base = static_cast<int64_t>(bh) * p.atoms_per_bh;
    int32_t* table_row = p.page_tbl + row_base;
    const int32_t* pool_row = p.pool_map + row_base;
    int folded = 0;
    for (int a = lane; a < p.atoms_per_bh; a += 32) {
      const int slot = pool_row[a];
      if (slot >= 0 && table_row[a] < 0) {
        table_row[a] = slot;
        ++folded;
      }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) folded += __shfl_down_sync(0xffffffffu, folded, off);
    if (lane == 0) p.pool_cursor[bh] += folded;
  }
  if (lane == 0) {
    p.growth[bh] = needy ? 1 : 0;
    if (!needy) p.done[bh] = 1;
  }
  __threadfence();
  __syncwarp();
  int last_bh = 0;
  if (lane == 0) last_bh = (atomicAdd(p.growth_ctr, 1) == p.BH - 1) ? 1 : 0;
  last_bh = __shfl_sync(0xffffffffu, last_bh, 0);
  if (last_bh == 0) return;
  __threadfence();

  //: THE REDUCTION THE HOST READS, and the done flags' self-reset. Nothing needy means
  //: every batch-head is done, so the two answers are ONE test and no launch stands
  //: between two steps: docs/internals/decode/decode.md#growth-reduction
  int seen = 0;
  for (int i = lane; i < p.BH; i += 32) seen |= __ldcg(p.growth + i);
  const bool any_needy = __any_sync(0xffffffffu, seen != 0);
  if (!any_needy) {
    for (int i = lane; i < p.BH; i += 32) p.done[i] = 0;
  }
  if (lane == 0) {
    *p.growth_any = any_needy ? 1 : 0;
    *p.growth_ctr = 0;
  }
}

}  // namespace detail

size_t decode_step_smem_bytes(int total_rows, int d_v, int cols, bool decay, int mask_total,
                              int omask_total, int olist_total, int dlist_total) {
  const size_t amps = static_cast<size_t>(total_rows) * (decay ? 3 : 2);
  const size_t fold = static_cast<size_t>(kDecodeWarps) * (cols + 3);
  const size_t tables =
      2 * static_cast<size_t>(mask_total) + kKinds * static_cast<size_t>(omask_total)
      + kKinds * static_cast<size_t>(olist_total) + static_cast<size_t>(dlist_total);
  return (amps + static_cast<size_t>(d_v) + fold + tables) * sizeof(float);
}

//: THE DECLARED ARM LIST, AS DATA. The decode family's instantiation matrix is the
//: product `d_v(2) x D(4) x decay(2)`, and it is written out one row per arm so a
//: BUILD can name a subset of it by index -- `ROLA_DECODE_ARMS=5` compiles arm 5 alone
//: and nothing else. That is an ITERATION knob and only that: the default below is
//: every declared arm, so a plain build and every gate build are the closed-world one,
//: and a subset binary is refused a shape it does not carry rather than mis-launching
//: it. docs/build.md, docs/internals/decode/decode_api.md#arm-subset
//: THE LAUNCH IS STILL WRITTEN ONCE. `dispatch_switch.cuh` refuses the MACRO LADDER --
//: a `#define` per axis whose BODY restates the `<<<...>>>` -- and this is not one: the
//: macro carries the arm's three CONSTANTS and its comparison, the launch itself is the
//: single generic lambda below, and no `<<<...>>>` is spelled twice.
#define DECODE_ARM_0(X_) X_(32, 1, 0)
#define DECODE_ARM_1(X_) X_(32, 1, 1)
#define DECODE_ARM_2(X_) X_(32, 2, 0)
#define DECODE_ARM_3(X_) X_(32, 2, 1)
#define DECODE_ARM_4(X_) X_(32, 3, 0)
#define DECODE_ARM_5(X_) X_(32, 3, 1)
#define DECODE_ARM_6(X_) X_(32, 4, 0)
#define DECODE_ARM_7(X_) X_(32, 4, 1)
#define DECODE_ARM_8(X_) X_(64, 1, 0)
#define DECODE_ARM_9(X_) X_(64, 1, 1)
#define DECODE_ARM_10(X_) X_(64, 2, 0)
#define DECODE_ARM_11(X_) X_(64, 2, 1)
#define DECODE_ARM_12(X_) X_(64, 3, 0)
#define DECODE_ARM_13(X_) X_(64, 3, 1)
#define DECODE_ARM_14(X_) X_(64, 4, 0)
#define DECODE_ARM_15(X_) X_(64, 4, 1)

#ifndef ROLA_DECODE_ARMS
#define ROLA_DECODE_ARMS(X_) \
  DECODE_ARM_0(X_) DECODE_ARM_1(X_) DECODE_ARM_2(X_) DECODE_ARM_3(X_)  \
  DECODE_ARM_4(X_) DECODE_ARM_5(X_) DECODE_ARM_6(X_) DECODE_ARM_7(X_)  \
  DECODE_ARM_8(X_) DECODE_ARM_9(X_) DECODE_ARM_10(X_) DECODE_ARM_11(X_)  \
  DECODE_ARM_12(X_) DECODE_ARM_13(X_) DECODE_ARM_14(X_) DECODE_ARM_15(X_)
#endif

//: THE BUILT SET, READ BACK OFF THE SAME LIST: `[d_v, D, decay]` per arm, in the order
//: the binary carries them, so a subset build is SELF-IDENTIFYING rather than merely
//: smaller: docs/internals/decode/decode_api.md#arm-subset
std::vector<std::vector<int64_t>> rola_decode_arms() {
  std::vector<std::vector<int64_t>> out;
#define ROLA_DECODE_ROW(DV_, D_, DECAY_) out.push_back({DV_, D_, DECAY_});
  ROLA_DECODE_ARMS(ROLA_DECODE_ROW)
#undef ROLA_DECODE_ROW
  return out;
}

void launch_decode_step(const DecodeParams& p, int d_v, bool decay, int64_t smem_budget) {
  //: The instantiation matrix is `DV(2) x D(4) x DECAY(2) = 16`. What is NOT an axis --
  //: `BT`, `BC`, the lattice's `(k, m)` and every operand's storage type -- is the
  //: interesting part: docs/internals/decode/decode.md#instantiation-matrix
  TORCH_CHECK(d_v == 32 || d_v == 64, "decode instantiates d_v in {32, 64}, got ", d_v);
  const int cols = d_v + 1;
  const size_t smem = decode_step_smem_bytes(p.total_rows, d_v, cols, decay, p.mask_total,
                                             p.omask_total, p.olist_total, p.dlist_total);
  TORCH_CHECK(static_cast<int64_t>(smem) <= smem_budget, "the decode step's staged operands need ",
              smem, " B of shared memory against a per-block budget of ", smem_budget,
              " B; sum_l width_l = ", p.total_rows,
              ". This is a declared capacity, refused loudly rather than taken as a "
              "slower path.");
  auto stream = at::cuda::getCurrentCUDAStream();
  //: THE LAUNCH IS WRITTEN ONCE. Each axis walker hands its value back as an
  //: `integral_constant`, so the three `::value`s below are template arguments and
  //: the `<<<...>>>` configuration cannot drift between arms:
  //: docs/internals/dispatch_switch.md#one-launch
  //: ONE LAUNCH SITE. Each axis arrives as an `integral_constant`, so the three
  //: `::value`s are template arguments and the `<<<...>>>` configuration cannot drift
  //: between arms: docs/internals/dispatch_switch.md#one-launch
  auto launch = [&](auto DV, auto LEVELS, auto DECAY) {
    detail::decode_step_kernel<decltype(DV)::value, decltype(LEVELS)::value, decltype(DECAY)::value>
        <<<dim3(p.n_split, p.BH), kDecodeThreads, smem, stream>>>(p);
  };
  bool matched = false;
#define ROLA_DECODE_TRY(DV_, D_, DECAY_)                                                  \
  if (!matched && d_v == (DV_) && p.D == (D_) && decay == ((DECAY_) != 0)) {               \
    matched = true;                                                                        \
    launch(std::integral_constant<int, (DV_)>{}, std::integral_constant<int, (D_)>{},      \
           std::integral_constant<bool, (DECAY_) != 0>{});                                 \
  }
  ROLA_DECODE_ARMS(ROLA_DECODE_TRY)
#undef ROLA_DECODE_TRY
  TORCH_CHECK(matched, "this binary does not carry the decode arm (d_v = ", d_v, ", D = ", p.D,
              ", decay = ", decay,
              "). A full build carries every declared arm; an ITERATION build "
              "(ROLA_DECODE_ARMS) carries a subset and is not shippable -- "
              "rola.ops.decode.arms() lists what this one has.");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Host entry
// ---------------------------------------------------------------------------

namespace detail {

void check_f32(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32, got ", t.scalar_type());
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_i32(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == at::kInt, name, " must be int32, got ", t.scalar_type());
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

//: ONE FLOATING AXIS OVER THE CLOSED SET THE PRODUCER AND THE VALUE PROJECTION EMIT: an
//: unmatched dtype is a REFUSAL rather than a silent widening on the host.
//: docs/internals/decode/decode_fold.md#the-dtype-arms
int is_bf16(at::ScalarType dtype, const char* what) {
  TORCH_CHECK(dtype == at::kFloat || dtype == at::kBFloat16, what,
              " must be float32 or bfloat16, got ", dtype);
  return dtype == at::kBFloat16 ? 1 : 0;
}

//: The `[B, 1, H, W] -> [B * H, W]` fold as an ADDRESS. `T != 1` is refused -- the fold
//: is defined on one token and a second one would be silently dropped:
//: docs/internals/decode/decode_fold.md#the-identity
Span span_of(const at::Tensor& t, int64_t BH, int64_t width, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.dim() == 4 && t.size(1) == 1, name, " must be [B, 1, H, W], got ", t.sizes());
  TORCH_CHECK(t.size(0) * t.size(2) == BH, name, " folds to ", t.size(0) * t.size(2),
              " batch-heads against ", BH);
  TORCH_CHECK(width == 0 || t.size(3) == width, name, " is ", t.size(3), " wide against ", width);
  return Span{t.data_ptr(), t.stride(0), t.stride(2), t.stride(3)};
}

Span scalar_span_of(const at::Tensor& t, int64_t BH, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.dim() == 3 && t.size(1) == 1, name, " must be [B, 1, H], got ", t.sizes());
  TORCH_CHECK(t.size(0) * t.size(2) == BH, name, " folds to ", t.size(0) * t.size(2),
              " batch-heads against ", BH);
  return Span{t.data_ptr(), t.stride(0), t.stride(2), 0};
}

int ilog2_exact(int64_t x, const char* what) {
  TORCH_CHECK(x >= 1, what, " must be positive, got ", x);
  int e = 0;
  while ((int64_t{1} << e) < x) ++e;
  TORCH_CHECK((int64_t{1} << e) == x, what, " must be a power of two, got ", x);
  return e;
}

//: THE (k, m) BOX, FILLED AND REFUSED HERE. The derivation is `carry/box.cuh`'s
//: `BoxPlan`, evaluated on runtime widths instead of template arguments -- decode takes
//: its lattice as data, because one built binary serves every topology the producer can
//: emit. Every constraint is that header's `static_assert` restated as a runtime check.
//: docs/internals/decode/decode_lattice.md#the-refusals
void fill_lattice(DecodeParams& p, int64_t k, int64_t m, int64_t N) {
  const int kappa = ilog2_exact(k, "the lattice span k");
  const int j = ilog2_exact(m, "the lattice capacity m");
  TORCH_CHECK(k <= kDecodeMaxSpan, "the lattice span k = ", k, " exceeds the decode path's ",
              kDecodeMaxSpan);
  p.lat_k = static_cast<int>(k);

  const int D = p.D;
  //: `BoxPlan`'s section 2.5 as the atom-rectangle law generalizes it: the INNERMOST
  //: level takes as much capacity as it can carry -- never less than its balanced share,
  //: never less than an atom asks, never more than its own width admits (K6) -- and the
  //: rest of `j` is spread balanced over the outer levels, innermost first.  The atom
  //: floor is a PREFERENCE and the width ceiling is the law: a level narrower than an
  //: atom spills its capacity outward and the atom then spans more than one level.
  const int floor_inner = kAtomShift > kappa ? kAtomShift - kappa : 0;
  int ceil_inner = 0;
  while ((p.lat_k << (ceil_inner + 1)) <= p.level_width[D - 1]
         && p.level_width[D - 1] % (p.lat_k << (ceil_inner + 1)) == 0)
    ++ceil_inner;
  const int share = j / D + ((j % D) ? 1 : 0);
  int a_inner = share > floor_inner ? share : floor_inner;
  if (a_inner > ceil_inner) a_inner = ceil_inner;
  if (a_inner > j) a_inner = j;
  const int j_outer = j - a_inner;

  int64_t bc = 1;
  int64_t owners = 1;
  int local_bits = 0;
  for (int l = 0; l < D; ++l) {
    const int a = (l == D - 1 || D == 1)
                      ? a_inner
                      : j_outer / (D - 1) + ((l >= (D - 1) - (j_outer % (D - 1))) ? 1 : 0);
    p.lat_s[l] = p.lat_k << a;
    TORCH_CHECK(p.lat_s[l] <= kDecodeMaxSpan, "level ", l, "'s owner span k*m_l = ", p.lat_s[l],
                " exceeds the decode path's ", kDecodeMaxSpan);
    TORCH_CHECK(p.level_width[l] % p.lat_s[l] == 0, "level ", l, " of width ", p.level_width[l],
                " is not a whole number of owner spans k*m_l = ", p.lat_s[l],
                " (R0 spec K6: an owner sits inside a level)");
    p.lat_g[l] = p.level_width[l] / p.lat_s[l];
    bc *= p.lat_s[l];
    owners *= p.lat_g[l];
    local_bits += kappa + a;
  }
  TORCH_CHECK(owners * bc == N, "the lattice partitions ", owners * bc, " leaves against ", N);
  p.lat_local_bits = local_bits;

  //: `BoxPlan::run_shift`: level `l`'s run offset sits at `log2 prod_{l\' > l} s_l\'` in the
  //: owner-local index, so the whole mixed radix is shifts and masks.
  int gsuf = 1;
  int run_shift = 0;
  for (int l = D - 1; l >= 0; --l) {
    p.lat_gsuf[l] = gsuf;
    p.lat_run_shift[l] = run_shift;
    gsuf *= p.lat_g[l];
    run_shift += ilog2_exact(p.lat_s[l], "the level's owner span");
  }

  //: THE UNIT, and the atom it is reconciled with. The unit is the innermost level's run
  //: -- ALWAYS, at every topology -- and the two shifts below are how an atom is named in
  //: terms of it: a run at or above `kAtomLeaves` holds whole atoms, a run below it means
  //: an atom is that many consecutive units. Exactly one of the two is non-zero, and only
  //: the CANDIDATE enumeration reads them; the walk is the same three loops either way.
  p.lat_inner_bits = ilog2_exact(p.lat_s[D - 1], "the innermost level's owner span");
  p.lat_outer_bits = local_bits - p.lat_inner_bits;
  for (int l = 0; l < D; ++l) {
    p.lat_oshift[l] = l < D - 1 ? p.lat_run_shift[l] - p.lat_inner_bits : 0;
  }
  p.atoms_per_unit_shift = p.lat_inner_bits > kAtomShift ? p.lat_inner_bits - kAtomShift : 0;
  p.lat_units_per_atom_shift = p.lat_inner_bits < kAtomShift ? kAtomShift - p.lat_inner_bits : 0;

  int mask_total = 0;
  int omask_total = 0;
  int olist_total = 0;
  int dlist_total = 0;
  for (int l = 0; l < D; ++l) {
    p.lat_sbits[l] = ilog2_exact(p.lat_s[l], "the level's owner span");
    p.mask_off[l] = mask_total;
    p.mask_words[l] = ceil_div_int(p.level_width[l], 32);
    mask_total += p.mask_words[l];
    p.omask_off[l] = omask_total;
    p.omask_words[l] = ceil_div_int(p.lat_g[l], 32);
    omask_total += p.omask_words[l];
    p.olist_off[l] = olist_total;
    olist_total += p.lat_g[l];
    p.dlist_off[l] = dlist_total;
    if (l < D - 1) dlist_total += p.level_width[l];
    TORCH_CHECK(p.lat_g[l] <= kDecodeThreads, "level ", l, "'s owner grid is ", p.lat_g[l],
                " wide against the ", kDecodeThreads,
                " threads that ballot its liveness in one block pass");
  }
  p.mask_total = mask_total;
  p.omask_total = omask_total;
  p.olist_total = olist_total;
  p.dlist_total = dlist_total;
}

//: The per-level geometry every decode entry validates the same way, filled into `p` and
//: reported back as `total_rows`.
int fill_levels(DecodeParams& p, const std::vector<at::Tensor>& levels,
                const std::vector<int64_t>& level_widths, int64_t BH, int64_t N) {
  int total_rows = 0;
  int64_t expect_n = 1;
  for (int l = 0; l < p.D; ++l) {
    const int width = static_cast<int>(level_widths[l]);
    TORCH_CHECK(width >= 1 && width <= kDecodeMaxLevelWidth, "level ", l, " width ", width,
                " exceeds the decode path's capacity ", kDecodeMaxLevelWidth,
                " (see kProducerMaxBranchWidth)");
    p.write[l] = span_of(levels[l], BH, width, "write[l]");
    p.level_width[l] = width;
    p.level_amp_offset[l] = total_rows;
    total_rows += width;
    expect_n *= width;
  }
  TORCH_CHECK(expect_n == N, "prod(level_widths) = ", expect_n,
              " disagrees with the state's leaf axis ", N);
  return total_rows;
}

int64_t smem_budget_of(int64_t device_index) {
  int smem_per_block = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&smem_per_block, cudaDevAttrMaxSharedMemoryPerBlock,
                                        static_cast<int>(device_index)));
  return smem_per_block;
}

}  // namespace detail

int64_t rola_decode_capacity() { return kDecodeMaxLevelWidth; }

//: The MIRRORED copy of the producer's `MAX_BRANCH_WIDTH`, bound separately from the
//: capacity because the two are DIFFERENT CLAIMS -- the second is what stops the
//: `static_assert` being satisfied by a stale mirror:
//: docs/internals/decode/decode.md#producer-width-mirror
int64_t rola_decode_producer_width_mirror() { return kProducerMaxBranchWidth; }

//: THE STEP'S DECLARED RESIDENCY, in CTAs per SM, read off the same `constexpr` the
//: `__launch_bounds__` are built from. The split policy needs it to size one wave, and a
//: python mirror of a launch bound is exactly the constant that drifts:
//: docs/internals/decode/decode_api.md#residency
int64_t rola_decode_residency(bool decay, int64_t levels) {
  TORCH_CHECK(levels >= 1 && levels <= kMaxLevels, "D must be in [1, 4], got ", levels);
  return decode_step_blocks_per_sm(decay, static_cast<int>(levels));
}

//: THE BUILD STAMP: a path or hash check cannot catch a stale `.so` at the right path,
//: and only a device-side fact can: docs/internals/decode/decode_api.md#build-stamp
int64_t rola_decode_build_stamp() {
  auto out = at::empty({1}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA));
  detail::stampdet::decode_stamp_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      out.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out.cpu().item<int64_t>();
}

at::Tensor rola_decode_forward(
    std::vector<at::Tensor> read, std::vector<at::Tensor> write, std::vector<int64_t> normalize,
    c10::optional<at::Tensor> dials, at::Tensor g_write, at::Tensor v, at::Tensor state,
    at::Tensor ws, at::Tensor ctr, at::Tensor growth, at::Tensor growth_any, at::Tensor growth_ctr,
    at::Tensor done, at::Tensor atom_bits, std::vector<int64_t> level_widths,
    std::vector<int64_t> level_row_offsets, int64_t lattice_k, int64_t lattice_m, int64_t n_split,
    double eps, c10::optional<at::Tensor> page_table, c10::optional<at::Tensor> pool_slots,
    c10::optional<at::Tensor> pool_cursor, c10::optional<at::Tensor> pool_map) {
  const int D = static_cast<int>(read.size());
  TORCH_CHECK(D >= 1 && D <= kMaxLevels, "D must be in [1, 4], got ", D);
  TORCH_CHECK(static_cast<int>(write.size()) == D && static_cast<int>(normalize.size()) == D
                  && static_cast<int>(level_widths.size()) == D
                  && static_cast<int>(level_row_offsets.size()) == D,
              "read, write, normalize and the two per-level integer vectors must each "
              "have D entries");

  TORCH_CHECK(v.dim() == 4 && v.size(1) == 1, "v must be [B, 1, H, d_v], got ", v.sizes());
  const int BH = static_cast<int>(v.size(0) * v.size(2));
  const int H = static_cast<int>(v.size(2));
  const int d_v = static_cast<int>(v.size(3));
  const int cols = d_v + 1;  // the mass column is unconditional (raw is dead)

  detail::check_f32(state, "state");
  detail::check_f32(ws, "ws");
  detail::check_i32(ctr, "ctr");

  //: THE TWO BACKINGS, validated where they differ and nowhere else. A page table names
  //: the leaf space (`[BH, N / kAtomLeaves]`) and the plane holds only the slots a plan
  //: committed; without one the plane IS the leaf space. Slot VALUES are not
  //: bounds-checked -- that would be a device-side read of the table, and the plan owns
  //: the bound. docs/internals/decode/decode_api.md#paged-state
  int64_t N = 0;
  int atoms_per_bh = 0;
  int32_t* page_ptr = nullptr;
  if (page_table.has_value()) {
    at::Tensor& tbl = *page_table;
    detail::check_i32(tbl, "page_table");
    TORCH_CHECK(tbl.dim() == 2 && tbl.size(0) == BH, "page_table must be [BH, N / ", kAtomLeaves,
                "] int32");
    atoms_per_bh = static_cast<int>(tbl.size(1));
    N = static_cast<int64_t>(atoms_per_bh) * kAtomLeaves;
    TORCH_CHECK(state.dim() == 3 && state.size(1) == kAtomLeaves && state.size(2) == cols,
                "under a page table the state is the arena's plane, [slots, ", kAtomLeaves,
                ", d_v + 1]");
    page_ptr = tbl.mutable_data_ptr<int32_t>();
  } else {
    TORCH_CHECK(state.dim() == 3 && state.size(0) == BH && state.size(2) == cols,
                "state must be [BH, N, d_v + 1]");
    N = state.size(1);
  }

  DecodeParams p{};
  p.D = D;
  const int total_rows = detail::fill_levels(p, write, level_widths, BH, N);
  detail::fill_lattice(p, lattice_k, lattice_m, N);
  const at::ScalarType level_dtype = read[0].scalar_type();
  for (int l = 0; l < D; ++l) {
    p.read[l] = detail::span_of(read[l], BH, p.level_width[l], "read[l]");
    TORCH_CHECK(read[l].scalar_type() == level_dtype && write[l].scalar_type() == level_dtype,
                "every routing level shares one dtype; level ", l, " disagrees");
    p.normalize[l] = normalize[l] != 0 ? 1 : 0;
    p.level_row_offset[l] = static_cast<int>(level_row_offsets[l]);
  }
  p.g_write = detail::scalar_span_of(g_write, BH, "g_write");
  p.v = detail::span_of(v, BH, d_v, "v");
  p.levels_bf16 = detail::is_bf16(level_dtype, "the routing levels");
  p.g_bf16 = detail::is_bf16(g_write.scalar_type(), "g_write");
  p.v_bf16 = detail::is_bf16(v.scalar_type(), "v");

  TORCH_CHECK(n_split >= 1, "n_split must be >= 1, got ", n_split);
  TORCH_CHECK(ws.dim() == 3 && ws.size(0) == BH && ws.size(1) == n_split && ws.size(2) == cols,
              "ws must be [BH, n_split, cols]");
  TORCH_CHECK(ctr.dim() == 1 && ctr.size(0) == BH, "ctr must be [BH] int32");

  //: THE ADMISSION BUFFERS, checked whichever backing this is: the step's own verdict
  //: lands in them under a page table and they are untouched without one, and validating
  //: them either way is what keeps the two backings ONE ABI:
  //: docs/internals/decode/decode_api.md#the-step
  detail::check_i32(growth, "growth");
  detail::check_i32(growth_any, "growth_any");
  detail::check_i32(growth_ctr, "growth_ctr");
  detail::check_i32(done, "done");
  TORCH_CHECK(growth.dim() == 1 && growth.size(0) == BH, "growth must be [BH] int32");
  TORCH_CHECK(done.dim() == 1 && done.size(0) == BH, "done must be [BH] int32");
  TORCH_CHECK(growth_any.dim() == 1 && growth_any.size(0) == 1, "growth_any must be [1] int32");
  TORCH_CHECK(growth_ctr.dim() == 1 && growth_ctr.size(0) == 1, "growth_ctr must be [1] int32");
  TORCH_CHECK(
      atom_bits.is_cuda() && atom_bits.scalar_type() == at::kBool && atom_bits.is_contiguous(),
      "atom_bits must be a contiguous CUDA bool tensor, got ", atom_bits.scalar_type());
  TORCH_CHECK(
      atom_bits.dim() == 2 && atom_bits.size(0) == BH && atom_bits.size(1) == N / kAtomLeaves,
      "atom_bits must be [BH, N / ", kAtomLeaves, "] bool, got ", atom_bits.sizes());

  //: THE ATOM'S PLACE IN THE LATTICE, required of a PAGED backing only, and now ONE
  //: condition where the interleaved order needed two: a unit is a contiguous aligned run,
  //: so an atom is a whole number of units or a unit is a whole number of atoms, and only
  //: the second makes the candidate enumeration a partition. The interleave's `k^D >= 16`
  //: and `log2(k) | 4` are discharged by the mixed-radix order itself (`box.cuh`, K3/K4).
  //: A dense backing never asks an atom a liveness question and carries no constraint:
  //: docs/internals/decode/decode.md#candidate-atoms
  TORCH_CHECK(page_ptr == nullptr || p.lat_local_bits >= kAtomShift,
              "a paged decode needs one walk unit's leaves to hold a whole number of ", kAtomLeaves,
              "-leaf atoms, and this (k, m) gives an owner only ", (1 << p.lat_local_bits),
              " (K13: the innermost span must cover an atom)");

  const bool decay = dials.has_value();
  const int64_t budget = detail::smem_budget_of(v.device().index());

  if (decay) {
    detail::check_f32(*dials, "dials");
    TORCH_CHECK(dials->dim() == 2 && dials->size(0) == H && dials->size(1) == total_rows,
                "dials must be [H, sum_l width_l]");
    p.dials = dials->data_ptr<float>();
  }

  //: `empty`, not `zeros`: every element is written, so a zero-fill is a launch the
  //: launch-latency-bound small-batch regime pays for nothing:
  //: docs/internals/decode/decode.md#empty-not-zeros
  auto y = at::empty({BH, d_v}, v.options().dtype(at::kFloat));
  p.state = state.data_ptr<float>();
  p.page_tbl = page_ptr;
  p.atoms_per_bh = atoms_per_bh;
  p.y = y.data_ptr<float>();
  p.ws = ws.data_ptr<float>();
  p.ctr = ctr.data_ptr<int32_t>();
  p.growth = growth.data_ptr<int32_t>();
  p.growth_any = growth_any.data_ptr<int32_t>();
  p.growth_ctr = growth_ctr.data_ptr<int32_t>();
  p.done = done.data_ptr<int32_t>();
  p.atom_bits = atom_bits.data_ptr<bool>();

  //: THE SLACK POOL, present or absent as ONE decision -- three buffers and a capacity,
  //: or four zeros and exact-commit. There is no half-configured pool:
  //: docs/internals/paging/paging.md#the-slack-pool
  const bool pooled = pool_slots.has_value();
  TORCH_CHECK(pooled == pool_cursor.has_value() && pooled == pool_map.has_value(),
              "the slack pool is its three buffers together: pass pool_slots, pool_cursor "
              "and pool_map, or none of them");
  if (pooled) {
    TORCH_CHECK(page_table.has_value(),
                "a slack pool admits into a page table and this state is dense-backed");
    detail::check_i32(*pool_slots, "pool_slots");
    detail::check_i32(*pool_cursor, "pool_cursor");
    detail::check_i32(*pool_map, "pool_map");
    TORCH_CHECK(pool_slots->dim() == 2 && pool_slots->size(0) == BH,
                "pool_slots must be [BH, pool_cap] int32, got ", pool_slots->sizes());
    TORCH_CHECK(pool_cursor->dim() == 1 && pool_cursor->size(0) == BH,
                "pool_cursor must be [BH] int32");
    TORCH_CHECK(
        pool_map->dim() == 2 && pool_map->size(0) == BH && pool_map->size(1) == atoms_per_bh,
        "pool_map must be the page table's shape [BH, N / ", kAtomLeaves, "] int32");
    p.pool_slots = pool_slots->data_ptr<int32_t>();
    p.pool_cursor = pool_cursor->data_ptr<int32_t>();
    p.pool_map = pool_map->data_ptr<int32_t>();
    p.pool_cap = static_cast<int>(pool_slots->size(1));
  }
  p.N = N;
  p.BH = BH;
  p.H = H;
  p.n_split = static_cast<int>(n_split);
  p.total_rows = total_rows;
  p.eps = static_cast<float>(eps);

  launch_decode_step(p, d_v, decay, budget);
  return y;
}

}  // namespace decode
}  // namespace rola
