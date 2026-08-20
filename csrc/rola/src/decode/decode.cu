// THE T=1 DECODE STEP -- ONE KERNEL: fold, expand, walk. Token-stationary,
// state-streaming, bitmap-walked. Owns the step kernel, its 32-way instantiation matrix
// and the decode path's host entry.
//
// The ratified shape, literally: the token's routing weights and value are RESIDENT
// (SMEM/registers), the STATE streams past, and the grid partitions the support.
//
// Three facts a reader must not miss:
//   1. There is one token, so the contraction index is the LEAF and the output index is
//      the COLUMN. Lanes map to OUTPUT COLUMNS: no MMA, and no warp reduction either --
//      the dot never crosses a lane.
//   2. DETERMINISM IS A SHIPPED PROPERTY HERE and it diverges from the prefill arm
//      ON PURPOSE: `y` is combined in SLOT ORDER, never by an `atomicAdd` race, because
//      decode's `y` picks the next token and a last-bit flip compounds
//      autoregressively. The state needs no argument at all -- the row partition is a
//      PARTITION and no atomic touches `state`.
//   3. THE PROLOGUE IS PER-CTA AND THAT IS THE DESIGN. Every split-K CTA folds and
//      expands its own batch-head, so the support map never leaves shared memory and the
//      split partition costs no launch: docs/internals/decode/decode.md#per-cta-prologue
//
// The GEMV/GEMM argument, the measured reproducibility divergence, the barrier ledger,
// and the derivation that collapses the causal intra Gram into UPDATE-THEN-READ:
// docs/internals/decode/decode.md

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cub/block/block_scan.cuh>

#include "decode.cuh"
#include "decode_api.cuh"
#include "decode_expand.cuh"
#include "decode_fold.cuh"
#include "dispatch_switch.cuh"

namespace rola {
namespace decode {
namespace detail {

using BlockScanT = cub::BlockScan<int, kDecodeThreads>;

//: `expf(88) = 1.65e38 < FLT_MAX`. THE SAME VALUE AND THE SAME CONVENTION as
//: `intra_panel.cuh`'s `kPanelMaxExponent`, restated because decode shares none of that
//: header's panel algebra. On this path the clamp is STRUCTURALLY INERT -- the exponent
//: cannot go positive -- and is written anyway so ONE convention governs every survival
//: exponential in the repository: docs/internals/decode/decode.md#decode-max-exponent
constexpr float kDecodeMaxExponent = 88.0f;

//: THE FACTORIZED INDEX, in three objects and no pass over the leaf space.
//:
//: `digit_class(d)` names which sides carry amplitude at outer digit `d`; class `0` is a
//: DEAD digit, whose whole `S_1`-leaf range is empty on both sides. Every count below is
//: a lookup keyed on it, because level 0 broadcasts ONE inner block into every live
//: digit and nothing else: docs/internals/decode/decode.md#the-factorized-index
__device__ __forceinline__ int digit_class(const float* ar, const float* aw, int d) {
  return (ar[d] != 0.0f ? 1 : 0) | (aw[d] != 0.0f ? 2 : 0);
}

__device__ __forceinline__ uint32_t class_word(int c, int k, const uint32_t* ir,
                                               const uint32_t* iw) {
  if (c == 0) return 0u;
  if (c == 1) return ir[k];
  if (c == 2) return iw[k];
  return ir[k] | iw[k];
}

//: Set bits of class `c`'s inner block in words strictly before `upto`. A SERIAL loop,
//: not a prefix table: the callers are one thread per outer digit and one per coarse
//: group, so the block covers `inner_words` work per thread in parallel -- which is never
//: more than the bitmap-wide popcount this index replaces, and at every shipped topology
//: is a handful of words.
__device__ __forceinline__ int class_popc(int c, const uint32_t* ir, const uint32_t* iw, int upto) {
  int acc = 0;
  for (int k = 0; k < upto; ++k) acc += __popc(class_word(c, k, ir, iw));
  return acc;
}

//: `cum[d]` = union rows in outer digits strictly before `d`, so `cum[b_0]` is `M`, the
//: exact touched-row count; `lw` is the ascending list of digits the WRITE side reaches,
//: which is what the atom enumeration walks instead of the atom space.
//: ONE SCAN EACH, and that is structural: a level is at most `kDecodeMaxLevelWidth` wide
//: and the block is exactly that many threads.
__device__ __forceinline__ void digit_index(const DecodeParams& p, const float* ar, const float* aw,
                                            const uint32_t* ir, const uint32_t* iw, int32_t* cum,
                                            int32_t* lw, int& n_live,
                                            BlockScanT::TempStorage& scan_temp) {
  static_assert(kDecodeMaxLevelWidth <= kDecodeThreads,
                "the digit index scans one level in a single block pass");
  const int tid = threadIdx.x;
  const int b0 = p.level_width[0];
  const int c = (tid < b0) ? digit_class(ar, aw, tid) : 0;
  int excl = 0;
  int total = 0;
  BlockScanT(scan_temp).ExclusiveSum((tid < b0) ? class_popc(c, ir, iw, p.inner_words) : 0, excl,
                                     total);
  if (tid < b0) cum[tid] = excl;
  if (tid == 0) cum[b0] = total;
  __syncthreads();
  const int live = (tid < b0 && (c & 2) != 0) ? 1 : 0;
  int lex = 0;
  int ltot = 0;
  BlockScanT(scan_temp).ExclusiveSum(live, lex, ltot);
  if (live) lw[lex] = tid;
  if (tid == 0) n_live = ltot;
  __syncthreads();
}

//: Union rows before leaf `X`: whole outer digits from `cum`, then the partial digit's
//: own words. The two sides' popcounts do not sum to `popcount(r | w)`, which is why the
//: CLASS -- not the sides -- is what is counted.
__device__ __forceinline__ int rows_before(const DecodeParams& p, const float* ar, const float* aw,
                                           const uint32_t* ir, const uint32_t* iw,
                                           const int32_t* cum, int X) {
  const int span = p.inner_span;
  const int d = X / span;
  const int r = X - d * span;
  int acc = cum[d];
  if (r != 0) {
    const int c = digit_class(ar, aw, d);
    const int k = r >> 5;
    acc += class_popc(c, ir, iw, k);
    const int rem = r & 31;
    if (rem != 0) acc += __popc(class_word(c, k, ir, iw) & ((1u << rem) - 1u));
  }
  return acc;
}

//: `pc[g]` = union rows in coarse groups strictly before `g`; `pc[n_groups]` = `M`. The
//: values are the ones a full bitmap popcount would produce, at one `rows_before` each:
//: docs/internals/decode/decode.md#coarse-index
__device__ __forceinline__ void coarse_index(const DecodeParams& p, const float* ar,
                                             const float* aw, const uint32_t* ir,
                                             const uint32_t* iw, const int32_t* cum, int32_t* pc) {
  const int ng = p.n_groups;
  const int n = static_cast<int>(p.N);
  for (int g = threadIdx.x; g <= ng; g += kDecodeThreads) {
    const int leaf = g * (kCoarseGroupWords * 32);
    pc[g] = rows_before(p, ar, aw, ir, iw, cum, leaf < n ? leaf : n);
  }
  __syncthreads();
}

//: Locate the bitmap word holding global set-bit index `row`, and how many set bits of
//: that word precede it. Two steps, NEITHER a scan of the bitmap: a binary search over
//: the coarse index, then ONE WARP-WIDE popcount of the group it lands in.
//:
//: ONE LANE PER GROUP WORD, which is exact because the coarse group IS a warp's width;
//: a serial walk of the group's words was, measured, the step's single largest barrier
//: stall -- one thread chaining dependent shared loads while the block waited.
//: docs/internals/decode/decode.md#locate-row
__device__ __forceinline__ void locate_row(const DecodeParams& p, const int32_t* pc,
                                           const uint32_t* inner, const float* sa_r,
                                           const float* sa_w, int n_words, int n_groups, int row,
                                           int& out_word, int& out_skip) {
  static_assert(kCoarseGroupWords == 32, "locate_row resolves a coarse group in one warp");
  const int lane = threadIdx.x & 31;
  int lo = 0;
  int hi = n_groups;
  while (lo < hi - 1) {
    const int mid = (lo + hi) >> 1;
    if (pc[mid] <= row)
      lo = mid;
    else
      hi = mid;
  }
  const int w = lo * kCoarseGroupWords + lane;
  const int base = w << 5;
  const int g = base / p.inner_span;
  const int i = base - g * p.inner_span;
  const int c =
      (w < n_words)
          ? __popc(view_r_at(p, inner, sa_r, base, g, i) | view_w_at(p, inner, sa_w, base, g, i))
          : 0;
  int incl = c;
#pragma unroll
  for (int s = 1; s < 32; s <<= 1) {
    const int t = __shfl_up_sync(0xffffffffu, incl, s);
    if (lane >= s) incl += t;
  }
  const int excl = incl - c;
  const int target = row - pc[lo];
  const unsigned hit = __ballot_sync(0xffffffffu, c != 0 && excl <= target && target < incl);
  const int sel = __ffs(static_cast<int>(hit)) - 1;
  out_word = __shfl_sync(0xffffffffu, w, sel);
  out_skip = target - __shfl_sync(0xffffffffu, excl, sel);
}

//: Is atom `a` in the write map? A bitmap WORD is exactly TWO atoms, because the atom is
//: 16 leaves and leaf order is the bitmap's bit order, so the membership of an atom is a
//: HALF-WORD TEST and no loop over leaves exists anywhere on this path.
//: docs/internals/decode/decode.md#atom-bits
__device__ __forceinline__ bool atom_written(uint32_t word, int a) {
  return (word & (0x0000ffffu << ((a & 1) << 4))) != 0u;
}

//: THE WRITE SIDE'S CANDIDATE ATOMS, enumerated through the outer digits that carry
//: write amplitude rather than through the atom space: a dead digit's whole `S_1`-leaf
//: range is dead, so this rectangle is the WRITE SUPPORT's size, not `N / 16`.
//: An atom straddling two live digits is taken by the LOWER one, which keeps the
//: enumeration a PARTITION in ascending atom id -- the pool claim's canonical ranks
//: depend on exactly that. docs/internals/decode/decode.md#candidate-atoms
__device__ __forceinline__ int atoms_per_digit(int span) {
  return ceil_div_int(span, kAtomLeaves) + 1;
}

__device__ __forceinline__ int candidate_atom(const DecodeParams& p, const int32_t* lw, int idx,
                                              int per) {
  const int span = p.inner_span;
  const int k = idx / per;
  const int d = lw[k];
  const int a = (d * span) / kAtomLeaves + (idx - k * per);
  if (a >= p.atoms_per_bh || a > ((d + 1) * span - 1) / kAtomLeaves) return -1;
  if (k > 0 && a <= ((lw[k - 1] + 1) * span - 1) / kAtomLeaves) return -1;
  return a;
}

//: THE TABLE TEST: how many atoms this batch-head's write side reaches that the page
//: table does not map. Every CTA of the batch-head computes it independently from the
//: same two objects -- its own expansion and a table no launch mutates until every CTA
//: of that batch-head has arrived -- so the answer is a RECOMPUTE and never a
//: communication. docs/internals/decode/decode.md#the-table-test
__device__ __forceinline__ int unmapped_write_atoms(const DecodeParams& p, const uint32_t* inner,
                                                    const float* sa_w, const int32_t* page_row,
                                                    const int32_t* lw, int n_live, int per,
                                                    BlockScanT::TempStorage& scan_temp) {
  int found = 0;
  for (int i = threadIdx.x; i < n_live * per; i += kDecodeThreads) {
    const int a = candidate_atom(p, lw, i, per);
    if (a >= 0 && atom_written(view_w_word(p, inner, sa_w, a >> 1), a) && page_row[a] < 0) ++found;
  }
  int excl = 0;
  int total = 0;
  BlockScanT(scan_temp).ExclusiveSum(found, excl, total);
  return total;
}

//: THE CLAIM, replicated and canonical: the batch-head's touched-unmapped atoms take
//: pool slots in ASCENDING ATOM ID from its own cursor, so every CTA of the batch-head
//: derives the identical mapping with no atomic and no arrival.
//:
//: IT BUILDS A SHADOW TABLE ROW rather than writing the page table, and both halves of
//: that are load-bearing. Not the table, because the table is the claim's own predicate
//: and a prologue write would move a sibling CTA's ranks. A WHOLE ROW -- every atom,
//: claimed or resident or absent -- because then the walk addresses through this row
//: INSTEAD of the table, by one pointer, and the walk's own body knows nothing about
//: pools. The fold back into the table happens once, later, where it is safe.
//: docs/internals/decode/decode.md#the-claim
__device__ __forceinline__ void claim_pool(const DecodeParams& p, const uint32_t* inner,
                                           const float* sa_w, const int32_t* page_row,
                                           const int32_t* slots, int32_t* shadow_row,
                                           const int32_t* lw, int n_live, int per, int cursor,
                                           BlockScanT::TempStorage& scan_temp, int& running) {
  const int tid = threadIdx.x;
  for (int a = tid; a < p.atoms_per_bh; a += kDecodeThreads) shadow_row[a] = page_row[a];
  if (tid == 0) running = 0;
  __syncthreads();
  const int total = n_live * per;
  for (int base = 0; base < total; base += kDecodeThreads) {
    const int i = base + tid;
    const int a = (i < total) ? candidate_atom(p, lw, i, per) : -1;
    const bool want =
        a >= 0 && atom_written(view_w_word(p, inner, sa_w, a >> 1), a) && page_row[a] < 0;
    int excl = 0;
    int aggregate = 0;
    BlockScanT(scan_temp).ExclusiveSum(want ? 1 : 0, excl, aggregate);
    if (want) shadow_row[a] = slots[cursor + running + excl];
    __syncthreads();
    if (tid == 0) running += aggregate;
    __syncthreads();
  }
}

template <int DV, int D, bool DECAY, bool IN_SMEM>
__global__ __launch_bounds__(kDecodeThreads, decode_step_blocks_per_sm(DECAY, IN_SMEM)) void
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
  const int nw = p.n_words;
  const int ng = p.n_groups;
  const int total_rows = p.total_rows;

  // ---- SMEM ledger ----------------------------------------------------------
  //   sa_r, sa_w   [total_rows] fp32   both sides' amplitude rows, UNCOMPACTED and
  //                                    indexed by DIGIT: the bitmap carries membership,
  //                                    so the digit IS the index.
  //   sd           [total_rows] fp32   this head's decay dials; DECAY only
  //   v_tok        [DV]         fp32   the token's value, resident
  //   fold  [warps][cols+3]     fp32   the cross-warp partials, bank-padded (see below)
  //   pc           [ng + 1]     i32    the coarse popcount index
  //   cum          [b0 + 1]     i32    union rows before each outer digit
  //   lw           [b0]         i32    the write side's live outer digits, ascending
  //   bits     [4 * iwords]     u32    the two sides' inner blocks and the expansion's
  //                                    ping-pong, on the IN_SMEM arm
  const int b0 = p.level_width[0];
  const int iwords = p.inner_words;
  float* sa_r = reinterpret_cast<float*>(smem_raw);
  float* sa_w = sa_r + total_rows;
  float* sd = sa_w + total_rows;
  float* v_tok = sd + (DECAY ? total_rows : 0);
  float* fold = v_tok + DV;
  int32_t* pc = reinterpret_cast<int32_t*>(fold + kDecodeWarps * (cols + 3));
  int32_t* cum = pc + ng + 1;
  int32_t* lw = cum + b0 + 1;
  uint32_t* smem_bits = reinterpret_cast<uint32_t*>(lw + b0);
  uint32_t* global_bits =
      p.work + (static_cast<int64_t>(bh) * p.n_split + seg) * kExpandBufWords * iwords;
  uint32_t* bits = IN_SMEM ? smem_bits : global_bits;

  __shared__ float part[cols + 3];
  __shared__ float s_sum[kMaxLevels];
  __shared__ typename BlockScanT::TempStorage scan_temp;
  __shared__ int s_running;
  __shared__ int s_n_live;
  __shared__ int s_w_start, s_w_last, s_drop, s_keep;
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

  // ---- THE EXPANSION: the inner block, and the index built on top of it -------
  //: LEVEL 0 IS NOT MATERIALIZED. `view_*_at` apply its broadcast where a word is asked
  //: for, and the index below is counted from the inner block and the digit classes, so
  //: nothing in this prologue is a pass over the leaf space:
  //: docs/internals/decode/decode.md#the-factorized-index
  const int parity = expand_inner<D>(p, sa_r, sa_w, bits);
  const uint32_t* inner_r = bits + parity * iwords;
  const uint32_t* inner_w = bits + 2 * iwords + parity * iwords;
  digit_index(p, sa_r, sa_w, inner_r, inner_w, cum, lw, s_n_live, scan_temp);
  coarse_index(p, sa_r, sa_w, inner_r, inner_w, cum, pc);
  const int n_live = s_n_live;
  const int per_digit = atoms_per_digit(p.inner_span);

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
    __syncthreads();
    const int demand = unmapped_write_atoms(p, inner_r, sa_w, p.page_tbl + row_base, lw, n_live,
                                            per_digit, scan_temp);
    __syncthreads();
    if (demand > 0) {
      const int cursor = (p.pool_cap > 0) ? p.pool_cursor[bh] : 0;
      claimed = demand <= p.pool_cap - cursor;
      needy = !claimed;
      if (claimed) {
        claim_pool(p, inner_r, sa_w, p.page_tbl + row_base,
                   p.pool_slots + static_cast<int64_t>(bh) * p.pool_cap, p.pool_map + row_base, lw,
                   n_live, per_digit, cursor, scan_temp, s_running);
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
    for (int i = tid; i < n_live * per_digit; i += kDecodeThreads) {
      const int a = candidate_atom(p, lw, i, per_digit);
      if (a >= 0 && atom_written(view_w_word(p, inner_r, sa_w, a >> 1), a)) atom_row[a] = true;
    }
  }
  __syncthreads();

  //: A NEEDY BATCH-HEAD RUNS NOTHING: no walk, no `y`, no state, no counter beyond its
  //: arrival -- which is what lets the step be enqueued before the admission it needs is
  //: known, and what makes the caller's replay THE SAME STEP rather than a second one.
  //: docs/internals/decode/decode.md#per-bh-verdict
  if (!needy) {
    // ---- resolve this CTA's ROW range into a WORD range -------------------------
    // Defined in ROWS (popcount-balanced by construction), handed out in WORDS; the two
    // boundary words are cut by dropping / keeping set bits, exact rather than by an index
    // comparison. docs/internals/decode/decode.md#row-range-to-words
    const int M = pc[ng];
    const int row_lo = static_cast<int>(static_cast<int64_t>(seg) * M / p.n_split);
    const int row_hi = static_cast<int>(static_cast<int64_t>(seg + 1) * M / p.n_split);
    if (row_hi <= row_lo) {
      if (tid == 0) {
        s_w_start = 1;
        s_w_last = 0;  // empty range: the walk below runs zero iterations
        s_drop = 0;
        s_keep = 0;
      }
      //: THE TWO BOUNDARIES ARE ONE WARP EACH, resolved side by side.
    } else if (warp < 2) {
      int w, skip;
      locate_row(p, pc, inner_r, sa_r, sa_w, nw, ng, warp == 0 ? row_lo : row_hi - 1, w, skip);
      if (lane == 0) {
        if (warp == 0) {
          s_w_start = w;
          s_drop = skip;
        } else {
          s_w_last = w;
          s_keep = skip + 1;
        }
      }
    }
    __syncthreads();

    const int w_start = s_w_start;
    const int w_last = s_w_last;
    const int drop_first = s_drop;
    const int keep_last = s_keep;

    float acc[NC];
#pragma unroll
    for (int k = 0; k < NC; ++k) acc[k] = 0.0f;
    float acc_m = 0.0f;

    // ---- THE WALK --------------------------------------------------------------
    int dig[D];
    float o_r = 1.0f, o_w = 1.0f, o_rate = 1.0f;
    auto recompute_outer = [&]() {
      o_r = 1.0f;
      o_w = 1.0f;
      o_rate = 1.0f;
#pragma unroll
      for (int l = 0; l < D - 1; ++l) {
        const int off = p.level_amp_offset[l] + dig[l];
        o_r *= sa_r[off];
        o_w *= sa_w[off];
        if (DECAY) o_rate *= sd[off];
      }
    };

    const int b_last = p.level_width[D - 1];
    const int off_last_base = p.level_amp_offset[D - 1];

    //: THE PAGED BASE TRANSLATION, resolved ONCE PER TOUCHED ATOM. The table row is pinned
    //: here (a plain origin, no per-atom widening multiply); the cache is keyed on the atom
    //: VALUE, so it is correct whatever order the walk reaches leaves in and costs at most
    //: two int32 loads per 32-leaf bitmap word. `atom_base` is the atom's first row as an
    //: ELEMENT offset from `p.state`: dense is `bh * N + (atom << 4)` (which is why a
    //: topology whose `N` is not a whole number of atoms still addresses correctly -- the
    //: two halves recombine to `bh * N + leaf`), paged is `slot << 4`.
    //: docs/internals/decode/decode.md#paged-address
    //: A CLAIMING STEP WALKS THE SHADOW ROW, and that is the whole of the pool inside
    //: the walk: one pointer, chosen once: docs/internals/decode/decode.md#the-claim
    const int32_t* page_row =
        !paged ? nullptr
               : (claimed ? p.pool_map : p.page_tbl) + static_cast<int64_t>(bh) * p.atoms_per_bh;
    const int64_t dense_bh_base = static_cast<int64_t>(bh) * p.N;
    int cached_atom = -1;
    int64_t atom_base = 0;
    bool atom_present = false;

    int w = w_start + warp;
    int g_cached = -1;
    bool g_empty = false;
    while (w <= w_last) {
      //: THE COARSE-INDEX CONSULT AT WALK TIME: a group whose prefix does not advance
      //: holds no touched row, so its 32 words are never read. Once per GROUP, not once
      //: per word: docs/internals/decode/decode.md#coarse-consult
      const int grp = w >> 5;
      if (grp != g_cached) {
        g_cached = grp;
        g_empty = (pc[grp + 1] == pc[grp]);
      }
      if (g_empty) {
        const int next = (grp + 1) << 5;
        w += ceil_div_int(next - w, kDecodeWarps) * kDecodeWarps;
        continue;
      }

      const int leaf_base = w * 32;
      //: The `D` integer divisions are paid ONCE PER WORD, not once per row -- a per-row
      //: decomposition would make the A100 dense arm instruction-bound:
      //: docs/internals/decode/decode.md#divisions-per-word
      //: IT COMES BEFORE THE WORDS because level 0's broadcast needs exactly this leaf
      //: position, so the two sides' words cost no division of their own.
      int inner_off = 0;
      {
        int rem = leaf_base;
#pragma unroll
        for (int l = 0; l < D; ++l) {
          const int s = p.level_stride[l];
          dig[l] = rem / s;
          rem -= dig[l] * s;
          if (l == 0) inner_off = rem;
        }
      }
      const uint32_t rw_word = view_r_at(p, inner_r, sa_r, leaf_base, dig[0], inner_off);
      const uint32_t ww_word = view_w_at(p, inner_r, sa_w, leaf_base, dig[0], inner_off);
      //: THE UNION, formed in a REGISTER and never stored.
      uint32_t u = rw_word | ww_word;
      //: HAZARD boundary-cut-order -- docs/internals/decode/decode.md#boundary-cut-order
      if (w == w_last) u = keep_low_set_bits(u, keep_last);
      if (w == w_start) u = drop_low_set_bits(u, drop_first);
      if (u == 0u) {
        w += kDecodeWarps;
        continue;
      }
      recompute_outer();

      int q = 0;
      int cur = dig[D - 1];
      //: The word is walked in ALIGNED SUB-RANGES, each inside ONE inner block, so the
      //: innermost digit needs no division and the outer digits are constant. ONE uniform
      //: loop that degenerates, not a fast path with a fallback:
      //: docs/internals/decode/decode.md#aligned-subranges
      while (q < 32) {
        int take = b_last - cur;
        if (take > 32 - q) take = 32 - q;
        uint32_t sub = (take >= 32) ? u : ((u >> q) & ((1u << take) - 1u));
        while (sub != 0u) {
          const int t = __ffs(static_cast<int>(sub)) - 1;
          sub &= sub - 1u;
          const int bit = q + t;
          //: Bits at or beyond `N` are never set (the expansion's loop stops at
          //: `span_out`), so the walk needs no leaf bound test.
          const int leaf = leaf_base + bit;
          const int off_last = off_last_base + cur + t;
          const bool in_r = ((rw_word >> bit) & 1u) != 0u;
          const bool in_w = ((ww_word >> bit) & 1u) != 0u;

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
          for (int k = 0; k < NC; ++k) sv[k] = 0.0f;
          float smass = 0.0f;
          if (atom_present) {
#pragma unroll
            for (int k = 0; k < NC; ++k) sv[k] = row[lane + 32 * k];
            if (lane == 0) smass = row[DV];
          }

          //: UPDATE FIRST, THEN READ THE UPDATED ROW -- AND THAT ORDER IS THE SEMANTICS,
          //: not an optimization: survival is INCLUSIVE at `j = t`, so the whole
          //: causal intra Gram collapses into reading the row after it is written. A row
          //: in `supp(R) \ supp(W)` has `keep == 1` and no deposit, so the same expression
          //: is its readout too -- one path, no arm.
          //: A DONE BATCH-HEAD TAKES THE READ AND NOT THE DEPOSIT, and reproduces its `y`
          //: BIT FOR BIT while doing so: the rows it would deposit into already hold the
          //: deposit, and update-then-read is exactly what makes re-reading them the same
          //: expression: docs/internals/decode/decode.md#done-flags
          //: HAZARD update-then-read -- docs/internals/decode/decode.md#update-then-read
          if (in_w && atom_present && deposit) {
            //: DECAY AT READ, VIA THE DETACHED CLOCK, WITH NO SWEEP AND NO EXPONENT
            //: SPLIT: at `T = 1` the tiled kernel's rule-1 half-mass shift is exactly
            //: the identity, so ONE `expf` replaces its three. `log1pf` IS the composition
            //: (never a rounded `1 - rate`) and there is no epsilon floor -- both of that
            //: rule's rule 2's prohibitions: docs/internals/decode/decode.md#decay-at-read
            float keep = 1.0f;
            const float wa = o_w * sa_w[off_last];
            if (DECAY) {
              const float rate = o_rate * sd[off_last];
              //: THE CLOCK IS THE STORED WRITE ALLOCATION ITSELF -- no side gain, which is
              //: what makes "stored != 0 => live != 0" an identity:
              //: docs/internals/decode/decode.md#detached-clock
              keep = expf(fminf(wa * log1pf(-rate), kDecodeMaxExponent));
            }
            const float wv = wa * sw;
#pragma unroll
            for (int k = 0; k < NC; ++k) {
              sv[k] = keep * sv[k] + wv * v_tok[lane + 32 * k];
              row[lane + 32 * k] = sv[k];
            }
            //: The mass column is the `v := 1` column of the identical recurrence.
            if (lane == 0) {
              smass = keep * smass + wv;
              row[DV] = smass;
            }
          }
          if (in_r) {
            const float a = o_r * sa_r[off_last];
#pragma unroll
            for (int k = 0; k < NC; ++k) acc[k] += a * sv[k];
            if (lane == 0) acc_m += a * smass;
          }
        }
        q += take;
        cur += take;
        if (cur >= b_last) {
          cur = 0;
#pragma unroll
          for (int l = D - 2; l >= 0; --l) {
            if (++dig[l] < p.level_width[l]) break;
            dig[l] = 0;
          }
          recompute_outer();
        }
      }
      w += kDecodeWarps;
    }

    // ---- the 8-way cross-warp fold, in FIXED warp order ------------------------
    //: HAZARD fold-bank-padding -- docs/internals/decode/decode.md#fold-bank-padding
    const int fold_pitch = cols + 3;
#pragma unroll
    for (int k = 0; k < NC; ++k) fold[warp * fold_pitch + lane + 32 * k] = acc[k];
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
    BlockScanT(scan_temp).ExclusiveSum(folded, excl, taken);
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

size_t decode_step_smem_bytes(int total_rows, int d_v, int cols, bool decay, int n_groups,
                              int outer_width, int inner_words, bool in_smem) {
  const size_t amps = static_cast<size_t>(total_rows) * (decay ? 3 : 2);
  const size_t fold = static_cast<size_t>(kDecodeWarps) * (cols + 3);
  const size_t index = static_cast<size_t>(n_groups) + 1 + 2 * static_cast<size_t>(outer_width) + 1;
  const size_t bits = in_smem ? static_cast<size_t>(kExpandBufWords) * inner_words : 0;
  return (amps + static_cast<size_t>(d_v) + fold + index + bits) * sizeof(float);
}

int64_t decode_work_words_per_cta(int total_rows, int d_v, int cols, bool decay, int n_groups,
                                  int outer_width, int inner_words, int64_t smem_budget) {
  const bool step_fits =
      static_cast<int64_t>(decode_step_smem_bytes(total_rows, d_v, cols, decay, n_groups,
                                                  outer_width, inner_words, true))
      <= smem_budget;
  return step_fits ? 0 : static_cast<int64_t>(kExpandBufWords) * inner_words;
}

void launch_decode_step(const DecodeParams& p, int d_v, bool decay, int64_t smem_budget) {
  //: The instantiation matrix is `DV(2) x D(4) x DECAY(2) x IN_SMEM(2) = 32` (the GLOBAL
  //: axis died with raw). What is NOT an axis -- `BT`, `BC`, and every operand's storage
  //: type -- is the interesting part: docs/internals/decode/decode.md#instantiation-matrix
  TORCH_CHECK(d_v == 32 || d_v == 64, "decode instantiates d_v in {32, 64}, got ", d_v);
  const int cols = d_v + 1;
  const bool in_smem =
      static_cast<int64_t>(decode_step_smem_bytes(p.total_rows, d_v, cols, decay, p.n_groups,
                                                  p.level_width[0], p.inner_words, true))
      <= smem_budget;
  const size_t smem = decode_step_smem_bytes(p.total_rows, d_v, cols, decay, p.n_groups,
                                             p.level_width[0], p.inner_words, in_smem);
  TORCH_CHECK(static_cast<int64_t>(smem) <= smem_budget, "the decode step's staged operands need ",
              smem, " B of shared memory against a per-block budget of ", smem_budget,
              " B; sum_l width_l = ", p.total_rows,
              ". This is a declared capacity, refused loudly rather than taken as a "
              "slower path.");
  auto stream = at::cuda::getCurrentCUDAStream();
  //: THE LAUNCH IS WRITTEN ONCE. Each axis walker hands its value back as an
  //: `integral_constant`, so the four `::value`s below are template arguments and
  //: the `<<<...>>>` configuration cannot drift between arms:
  //: docs/internals/dispatch_switch.md#one-launch
  int_switch<32, 64>(d_v, "decode d_v", [&](auto DV) {
    bool_switch(decay, [&](auto DECAY) {
      bool_switch(in_smem, [&](auto IN_SMEM) {
        int_switch<1, 2, 3, 4>(p.D, "D", [&](auto LEVELS) {
          detail::decode_step_kernel<decltype(DV)::value, decltype(LEVELS)::value,
                                     decltype(DECAY)::value, decltype(IN_SMEM)::value>
              <<<dim3(p.n_split, p.BH), kDecodeThreads, smem, stream>>>(p);
        });
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

//: The per-level geometry every decode entry validates the same way, filled into `p` and
//: reported back as `total_rows`.
int fill_levels(DecodeParams& p, const std::vector<at::Tensor>& levels,
                const std::vector<int64_t>& level_widths, const std::vector<int64_t>& level_strides,
                const std::vector<int64_t>& level_spans, int64_t BH, int64_t N) {
  int total_rows = 0;
  int64_t expect_n = 1;
  for (int l = 0; l < p.D; ++l) {
    const int width = static_cast<int>(level_widths[l]);
    TORCH_CHECK(width >= 1 && width <= kDecodeMaxLevelWidth, "level ", l, " width ", width,
                " exceeds the decode path's capacity ", kDecodeMaxLevelWidth,
                " (see kProducerMaxBranchWidth)");
    p.write[l] = span_of(levels[l], BH, width, "write[l]");
    p.level_width[l] = width;
    p.level_stride[l] = static_cast<int>(level_strides[l]);
    p.level_span[l] = static_cast<int>(level_spans[l]);
    p.level_amp_offset[l] = total_rows;
    total_rows += width;
    expect_n *= width;
  }
  TORCH_CHECK(expect_n == N, "prod(level_widths) = ", expect_n,
              " disagrees with the state's leaf axis ", N);
  TORCH_CHECK(p.level_span[0] == static_cast<int>(N),
              "level_spans[0] must be N (the whole leaf space)");
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

//: THE STEP'S DECLARED RESIDENCY, in CTAs per SM, read off the same `constexpr` the
//: `__launch_bounds__` are built from -- a python mirror of a launch bound is exactly
//: the constant that drifts. The split policy sizes ONE WAVE from it; the minimum over
//: the IN_SMEM axis is taken because the plan freezes the split before the arm, and a
//: conservative wave never over-launches: docs/internals/decode/decode_api.md
int64_t rola_decode_residency(bool decay) {
  const int a = decode_step_blocks_per_sm(decay, false);
  const int b = decode_step_blocks_per_sm(decay, true);
  return a < b ? a : b;
}

//: The MIRRORED copy of the producer's `MAX_BRANCH_WIDTH`, bound separately from the
//: capacity because the two are DIFFERENT CLAIMS -- the second is what stops the
//: `static_assert` being satisfied by a stale mirror:
//: docs/internals/decode/decode.md#producer-width-mirror
int64_t rola_decode_producer_width_mirror() { return kProducerMaxBranchWidth; }

//: THE ONE AUTHORITY over "does the expansion fit in shared memory here?". The host
//: allocates `BH * n_split *` this many int32, and `0` -- the answer at every shipped
//: topology -- means the bitmap never reaches DRAM at all. Asking the kernel's own sizing
//: function is what keeps the buffer from being sized by a Python mirror that can drift:
//: docs/internals/decode/decode_expand.md#smem-arm
int64_t rola_decode_work_words(std::vector<int64_t> level_widths, int64_t d_v, bool decay,
                               int64_t device_index) {
  int total_rows = 0;
  int64_t N = 1;
  for (int64_t width : level_widths) {
    total_rows += static_cast<int>(width);
    N *= width;
  }
  const int n_words = static_cast<int>((N + 31) / 32);
  const int n_groups = (n_words + kCoarseGroupWords - 1) / kCoarseGroupWords;
  const int outer_width = static_cast<int>(level_widths.at(0));
  const int inner_span = static_cast<int>(N / outer_width);
  return decode_work_words_per_cta(total_rows, static_cast<int>(d_v), static_cast<int>(d_v) + 1,
                                   decay, n_groups, outer_width, ceil_div_int(inner_span, 32),
                                   detail::smem_budget_of(device_index));
}

at::Tensor rola_decode_forward(
    std::vector<at::Tensor> read, std::vector<at::Tensor> write, std::vector<int64_t> normalize,
    c10::optional<at::Tensor> dials, at::Tensor g_write, at::Tensor v, at::Tensor state,
    at::Tensor work, at::Tensor ws, at::Tensor ctr, at::Tensor growth, at::Tensor growth_any,
    at::Tensor growth_ctr, at::Tensor done, at::Tensor atom_bits, std::vector<int64_t> level_widths,
    std::vector<int64_t> level_strides, std::vector<int64_t> level_spans,
    std::vector<int64_t> level_row_offsets, int64_t n_split, double eps,
    c10::optional<at::Tensor> page_table, c10::optional<at::Tensor> pool_slots,
    c10::optional<at::Tensor> pool_cursor, c10::optional<at::Tensor> pool_map) {
  const int D = static_cast<int>(read.size());
  TORCH_CHECK(D >= 1 && D <= kMaxLevels, "D must be in [1, 4], got ", D);
  TORCH_CHECK(static_cast<int>(write.size()) == D && static_cast<int>(normalize.size()) == D
                  && static_cast<int>(level_widths.size()) == D
                  && static_cast<int>(level_strides.size()) == D
                  && static_cast<int>(level_spans.size()) == D
                  && static_cast<int>(level_row_offsets.size()) == D,
              "read, write, normalize and the four per-level integer vectors must each "
              "have D entries");

  TORCH_CHECK(v.dim() == 4 && v.size(1) == 1, "v must be [B, 1, H, d_v], got ", v.sizes());
  const int BH = static_cast<int>(v.size(0) * v.size(2));
  const int H = static_cast<int>(v.size(2));
  const int d_v = static_cast<int>(v.size(3));
  const int cols = d_v + 1;  // the mass column is unconditional (raw is dead)

  detail::check_f32(state, "state");
  detail::check_f32(ws, "ws");
  detail::check_i32(ctr, "ctr");
  detail::check_i32(work, "work");

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
  const int total_rows =
      detail::fill_levels(p, write, level_widths, level_strides, level_spans, BH, N);
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

  TORCH_CHECK(p.level_stride[D - 1] == 1,
              "level_strides[D-1] must be 1 (the innermost level is contiguous in leaf order)");

  const int n_words = static_cast<int>((N + 31) / 32);
  const int n_groups = (n_words + kCoarseGroupWords - 1) / kCoarseGroupWords;
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

  //: `S_1 = N / b_0` -- the INNER span, which at `D == 1` is the empty product `1` and
  //: comes out of the same expression because `b_0` is then `N` itself.
  p.inner_span = static_cast<int>(N / p.level_width[0]);
  p.inner_words = ceil_div_int(p.inner_span, 32);

  const bool decay = dials.has_value();
  const int64_t budget = detail::smem_budget_of(v.device().index());
  const int64_t per_cta = decode_work_words_per_cta(total_rows, d_v, cols, decay, n_groups,
                                                    p.level_width[0], p.inner_words, budget);
  const int64_t want_work = per_cta * BH * n_split;
  TORCH_CHECK(work.dim() == 1 && work.size(0) == want_work,
              "the expansion's global scratch must be [", want_work,
              "] int32 for this geometry (0 = it fits in shared memory), got ", work.sizes());

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
  p.work = want_work == 0 ? nullptr : reinterpret_cast<uint32_t*>(work.data_ptr<int32_t>());
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
  p.n_words = n_words;
  p.n_groups = n_groups;
  p.n_split = static_cast<int>(n_split);
  p.total_rows = total_rows;
  p.eps = static_cast<float>(eps);

  launch_decode_step(p, d_v, decay, budget);
  return y;
}

}  // namespace decode
}  // namespace rola
