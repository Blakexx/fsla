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
//   3. THE PROLOGUE IS PER-CTA AND THAT IS THE DESIGN. Every split-K CTA folds and
//      factors its own batch-head, so the factor tables never leave shared memory and
//      the split partition costs no launch:
//      docs/internals/decode/decode.md#per-cta-prologue
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
  //: THE ENUMERATION'S SHAPE IS PART OF THE FOOTPRINT, alongside the two struct sizes and
  //: the span ceiling: the kind and term counts are what a walk's unit index MEANS, and a
  //: binary that disagrees about them is a different step however its structs measure --
  //: which the union-of-products enumeration proved, since it moved every walk in the
  //: extension without moving one byte of either struct. IT IS THE MOST SIGNIFICANT FIELD
  //: of the same positional scheme, so the three older fields still divide out unchanged:
  //: docs/internals/decode/decode_api.md#build-stamp
  *out =
      (static_cast<int64_t>(kKinds * kMaxTerms) * 4096 + static_cast<int64_t>(sizeof(DecodeParams)))
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
  //   olist  [4][olist_total]   i32    the four kinds' owner coordinates, per level
  float* sa_r = reinterpret_cast<float*>(smem_raw);
  float* sa_w = sa_r + total_rows;
  float* sd = sa_w + total_rows;
  float* v_tok = sd + (DECAY ? total_rows : 0);
  float* fold = v_tok + DV;
  uint32_t* mask_r = reinterpret_cast<uint32_t*>(fold + kDecodeWarps * (cols + 3));
  uint32_t* mask_w = mask_r + p.mask_total;
  int32_t* olist = reinterpret_cast<int32_t*>(mask_w + p.mask_total);

  __shared__ float part[cols + 3];
  __shared__ float s_sum[kMaxLevels];
  __shared__ typename LatticeScan::TempStorage scan_temp;
  __shared__ int s_running;
  __shared__ int s_n_o[kKinds * kMaxLevels];
  __shared__ int s_bits[kMaxTerms * kMaxLevels];
  __shared__ int s_shift[kMaxTerms * kMaxLevels];
  //: One unit count per TERM. The walk's outer loop is the term, so each term's rank space
  //: is partitioned across the grid on its own and the counts never need summing.
  __shared__ int s_units[kMaxTerms];
  __shared__ int s_last_cta;

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
  __syncthreads();
  build_owner_lists<D>(p, mask_r, mask_w, olist, s_n_o, scan_temp);
  build_term_layout<D>(p, s_n_o, s_bits, s_shift, s_units);

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
  auto candidate_atom = [&](int i) -> int {
    const int a = i & ((1 << p.atoms_per_unit_shift) - 1);
    const int unit0 = (i >> p.atoms_per_unit_shift) << p.lat_units_per_atom_shift;
    int atom = -1;
    for (int q = 0; q < n_units; ++q) {
      const Unit<D> u = resolve_unit<D, false>(p, mask_r, mask_w, sa_r, sa_w, sd, olist, s_n_o,
                                               s_bits, s_shift, 0, unit0 + q);
      if (!u.valid) return -1;
      if (atom < 0) atom = (u.base + (a << kAtomShift)) >> kAtomShift;
      if (u.live_w && ((u.cw >> ((a << kAtomShift) & ((1 << inner_bits) - 1))) & win_mask) != 0u) {
        return atom;
      }
    }
    return -1;
  };

  const int n_candidates = (s_units[0] >> p.lat_units_per_atom_shift) << p.atoms_per_unit_shift;

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
        //: IT BUILDS A SHADOW TABLE ROW rather than writing the page table, and both
        //: halves of that are load-bearing. Not the table, because the table is the
        //: claim's own predicate and a prologue write would move a sibling CTA's ranks.
        //: A WHOLE ROW -- every atom, claimed or resident or absent -- because then the
        //: walk addresses through this row INSTEAD of the table, by one pointer, and the
        //: walk's own body knows nothing about pools.
        //: docs/internals/decode/decode.md#the-claim
        int32_t* shadow_row = p.pool_map + row_base;
        const int32_t* slots = p.pool_slots + static_cast<int64_t>(bh) * p.pool_cap;
        for (int a = tid; a < p.atoms_per_bh; a += kDecodeThreads) shadow_row[a] = page_row[a];
        if (tid == 0) s_running = 0;
        __syncthreads();
        for (int base = 0; base < n_candidates; base += kDecodeThreads) {
          const int i = base + tid;
          const int a = (i < n_candidates) ? candidate_atom(i) : -1;
          const bool want = a >= 0 && page_row[a] < 0;
          int rank = 0;
          int aggregate = 0;
          LatticeScan(scan_temp).ExclusiveSum(want ? 1 : 0, rank, aggregate);
          if (want) shadow_row[a] = slots[cursor + s_running + rank];
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
  __syncthreads();

  //: A NEEDY BATCH-HEAD RUNS NOTHING: no walk, no `y`, no state, no counter beyond its
  //: arrival -- which is what lets the step be enqueued before the admission it needs is
  //: known, and what makes the caller's replay THE SAME STEP rather than a second one.
  //: docs/internals/decode/decode.md#per-bh-verdict
  if (!needy) {
    float acc[NC];
#pragma unroll
    for (int q = 0; q < NC; ++q) acc[q] = 0.0f;
    float acc_m = 0.0f;

    //: THE PAGED BASE TRANSLATION, resolved ONCE PER TOUCHED ATOM. The table row is
    //: pinned here (a plain origin, no per-atom widening multiply); the cache is keyed on
    //: the atom VALUE, so it is correct whatever order the walk reaches leaves in.
    //: `atom_base` is the atom's first row as an ELEMENT offset from `p.state`: dense is
    //: `bh * N + (atom << 4)` (which is why a topology whose `N` is not a whole number of
    //: atoms still addresses correctly -- the two halves recombine to `bh * N + leaf`),
    //: paged is `slot << 4`. docs/internals/decode/decode.md#paged-address
    //: A CLAIMING STEP WALKS THE SHADOW ROW, and that is the whole of the pool inside
    //: the walk: one pointer, chosen once: docs/internals/decode/decode.md#the-claim
    const int32_t* page_row =
        !paged ? nullptr
               : (claimed ? p.pool_map : p.page_tbl) + static_cast<int64_t>(bh) * p.atoms_per_bh;
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
      const int units = s_units[term];
      for (int unit = seg * kDecodeWarps + warp; unit < units; unit += p.n_split * kDecodeWarps) {
        const Unit<D> u = resolve_unit<D, DECAY>(p, mask_r, mask_w, sa_r, sa_w, sd, olist, s_n_o,
                                                 s_bits, s_shift, term, unit);
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
              const int slot = page_row[atom];
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

    // ---- the 8-way cross-warp fold, in FIXED warp order ------------------------
    //: HAZARD fold-bank-padding -- docs/internals/decode/decode.md#fold-bank-padding
    const int fold_pitch = cols + 3;
#pragma unroll
    for (int q = 0; q < NC; ++q) fold[warp * fold_pitch + lane + 32 * q] = acc[q];
    if (lane == 0) fold[warp * fold_pitch + DV] = acc_m;
    __syncthreads();
    if (tid < cols) {
      float t = 0.0f;
#pragma unroll
      for (int wq = 0; wq < kDecodeWarps; ++wq) t += fold[wq * fold_pitch + tid];
      part[tid] = sr * t;  // sr == 1 under global; kept as the one scale site
    }
    __syncthreads();

    // ---- the deterministic split-K combine, and the fused divide ---------------
    if (p.n_split > 1) {
      float* slot = p.ws + (static_cast<int64_t>(bh) * p.n_split + seg) * cols;
      if (tid < cols) slot[tid] = part[tid];
      __threadfence();
      __syncthreads();
      if (tid == 0) {
        s_last_cta = (atomicAdd(&p.ctr[bh], 1) == p.n_split - 1) ? 1 : 0;
      }
      __syncthreads();
      if (s_last_cta == 0) return;

      //: SLOT ORDER, always -- the whole determinism argument. `__ldcg` bypasses L1 so
      //: the reader sees the stores the `__threadfence` above made visible.
      //: HAZARD slot-order -- docs/internals/decode/decode.md#slot-order
      if (tid < cols) {
        const float* base = p.ws + static_cast<int64_t>(bh) * p.n_split * cols;
        float t = 0.0f;
        for (int s = 0; s < p.n_split; ++s)
          t += __ldcg(base + static_cast<int64_t>(s) * cols + tid);
        part[tid] = t;
      }
      __syncthreads();
      //: Reset for the next step: no `memset` launch between two decode steps, and the
      //: counter is safe to capture in a CUDA graph.
      if (tid == 0) p.ctr[bh] = 0;
    }
    if (tid < DV) {
      const float den = part[DV];
      p.y[static_cast<int64_t>(bh) * DV + tid] = part[tid] / (den + p.eps);
    }
  } else if (p.n_split > 1) {
    //: A needy batch-head's CTAs ARRIVE ANYWAY. The arrival is what elects one CTA to
    //: publish the verdict below, and electing it from the same counter the combine uses
    //: is what makes "no CTA of this batch-head is still reading" the SAME argument in
    //: both branches: docs/internals/decode/decode.md#the-last-arriver
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      s_last_cta = (atomicAdd(&p.ctr[bh], 1) == p.n_split - 1) ? 1 : 0;
    }
    __syncthreads();
    if (s_last_cta == 0) return;
    if (tid == 0) p.ctr[bh] = 0;
  }

  if (!paged) return;

  // ---- the per-batch-head publication, by its LAST-ARRIVING CTA ---------------
  //: EVERY MUTATION A SIBLING CTA COULD HAVE READ HAPPENS HERE AND NOWHERE ELSE, which
  //: is why it is safe: docs/internals/decode/decode.md#the-last-arriver
  if (claimed) {
    //: THE CLAIM'S SIZE FALLS OUT OF THE FOLD ITSELF -- an atom the shadow row maps and
    //: the table does not IS a claimed atom -- so it is neither carried across the walk
    //: nor re-derived from the write-support enumeration. Both of those keep values alive
    //: through the loop the register budget is tight in, and both were measured to spill:
    //: docs/internals/decode/decode.md#declared-residency
    const int64_t row_base = static_cast<int64_t>(bh) * p.atoms_per_bh;
    int32_t* page_row = p.page_tbl + row_base;
    const int32_t* pool_row = p.pool_map + row_base;
    int folded = 0;
    for (int a = tid; a < p.atoms_per_bh; a += kDecodeThreads) {
      const int slot = pool_row[a];
      if (slot >= 0 && page_row[a] < 0) {
        page_row[a] = slot;
        ++folded;
      }
    }
    int excl = 0;
    int taken = 0;
    LatticeScan(scan_temp).ExclusiveSum(folded, excl, taken);
    if (tid == 0) p.pool_cursor[bh] += taken;
  }
  if (tid == 0) {
    p.growth[bh] = needy ? 1 : 0;
    if (!needy) p.done[bh] = 1;
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last_cta = (atomicAdd(p.growth_ctr, 1) == p.BH - 1) ? 1 : 0;
  __syncthreads();
  if (s_last_cta == 0) return;

  //: THE REDUCTION THE HOST READS, and the done flags' self-reset. Nothing needy means
  //: every batch-head is done, so the two answers are ONE test and no launch stands
  //: between two steps: docs/internals/decode/decode.md#growth-reduction
  __shared__ int s_any;
  if (tid == 0) s_any = 0;
  __syncthreads();
  int seen = 0;
  for (int i = tid; i < p.BH; i += kDecodeThreads) seen |= __ldcg(p.growth + i);
  if (seen) s_any = 1;
  __syncthreads();
  if (s_any == 0) {
    for (int i = tid; i < p.BH; i += kDecodeThreads) p.done[i] = 0;
  }
  if (tid == 0) {
    *p.growth_any = s_any;
    *p.growth_ctr = 0;
  }
}

}  // namespace detail

size_t decode_step_smem_bytes(int total_rows, int d_v, int cols, bool decay, int mask_total,
                              int olist_total) {
  const size_t amps = static_cast<size_t>(total_rows) * (decay ? 3 : 2);
  const size_t fold = static_cast<size_t>(kDecodeWarps) * (cols + 3);
  const size_t tables =
      2 * static_cast<size_t>(mask_total) + kKinds * static_cast<size_t>(olist_total);
  return (amps + static_cast<size_t>(d_v) + fold + tables) * sizeof(float);
}

void launch_decode_step(const DecodeParams& p, int d_v, bool decay, int64_t smem_budget) {
  //: The instantiation matrix is `DV(2) x D(4) x DECAY(2) = 16`. What is NOT an axis --
  //: `BT`, `BC`, the lattice's `(k, m)` and every operand's storage type -- is the
  //: interesting part: docs/internals/decode/decode.md#instantiation-matrix
  TORCH_CHECK(d_v == 32 || d_v == 64, "decode instantiates d_v in {32, 64}, got ", d_v);
  const int cols = d_v + 1;
  const size_t smem =
      decode_step_smem_bytes(p.total_rows, d_v, cols, decay, p.mask_total, p.olist_total);
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
  int_switch<32, 64>(d_v, "decode d_v", [&](auto DV) {
    bool_switch(decay, [&](auto DECAY) {
      int_switch<1, 2, 3, 4>(p.D, "D", [&](auto LEVELS) {
        detail::decode_step_kernel<decltype(DV)::value, decltype(LEVELS)::value,
                                   decltype(DECAY)::value>
            <<<dim3(p.n_split, p.BH), kDecodeThreads, smem, stream>>>(p);
      });
    });
  });
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
  int olist_total = 0;
  for (int l = 0; l < D; ++l) {
    p.mask_off[l] = mask_total;
    p.mask_words[l] = ceil_div_int(p.level_width[l], 32);
    mask_total += p.mask_words[l];
    p.olist_off[l] = olist_total;
    olist_total += p.lat_g[l];
    TORCH_CHECK(p.lat_g[l] <= kDecodeThreads, "level ", l, "'s owner grid is ", p.lat_g[l],
                " wide against the ", kDecodeThreads,
                " threads that build its list in one block pass");
  }
  p.mask_total = mask_total;
  p.olist_total = olist_total;
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
