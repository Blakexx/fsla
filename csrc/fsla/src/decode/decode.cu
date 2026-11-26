// THE T=1 DECODE STEP -- ONE KERNEL: fold, factor, walk. Token-stationary,
// state-streaming, lattice-walked. Owns the step kernel, its 16-way instantiation matrix
// and the decode path's host entry.
//
// The ratified shape, literally: the token's routing weights and value are RESIDENT
// (SMEM/registers), the STATE streams past, and a work queue hands the lattice's unit
// rank space to resident CTAs.
//
// Three facts a reader must not miss:
//   1. There is one token, so the contraction index is the LEAF and the output index is
//      the COLUMN: a lane holds eight channels of a leaf and no MMA exists here.
//   2. THE CTAs VISIT HEADS. A head's work is known only after its prologue, so the grid is
//      one wave of resident CTAs that claim a head's positions, add their partials into its
//      sums, and steal the head with the most work left; the state stays byte-reproducible
//      (its positions partition the written leaves) and `y` is reproducible to its sums'
//      order: docs/internals/decode/decode.md#queue
//   3. THE PROLOGUE IS PER VISIT AND THAT IS THE DESIGN: the factor tables never leave
//      shared memory, and the walk takes no barrier:
//      docs/internals/decode/decode.md#per-cta-prologue,
//      docs/internals/decode/decode.md#no-barrier
//
// The GEMV/GEMM argument, the walk's cost accounting against `N`, the barrier ledger, and
// the derivation that collapses the causal intra Gram into UPDATE-THEN-READ:
// docs/internals/decode/decode.md

#include "common/torch_seam.cuh"
#include <cuda_bf16.h>

#include <algorithm>
#include <mutex>
#include <set>

#include "common/arch_runtime.cuh"
#include "common/select.cuh"
#include "common/smem_layout.cuh"
#include "common/phase_clock.cuh"
#include "common/state_page.cuh"
#include "decode.cuh"
#include "decode_selection.inc"
#include "plan/decode/contract.h"
#include "plan/decode_launch.h"
#include "plan/environment.h"
#include "common/arm_switch.cuh"
#include "decode_api.cuh"
#include "decode_fold.cuh"
#include "decode_lattice.cuh"
#include "decode_leaf.cuh"

namespace fsla {
namespace decode {
namespace detail {

//: NOT an anonymous namespace: nvcc mangles anonymous-namespace kernels with a hash that -- see docs/internals/decode/decode.md#near-line-51
namespace stampdet {

__device__ int64_t g_decode_stamp_probe;

//: THE STEP'S OWN FOOTPRINT, recompiled with this translation unit, so a test that reads -- see docs/internals/decode/decode.md#decodestampkernel
template <class Bt>
__global__ void decode_stamp_kernel(int64_t* out) {
  //: THE ENUMERATION'S SHAPE AND THE LAUNCH'S WIDTH ARE PART OF THE FOOTPRINT, -- see docs/internals/decode/decode.md#decode-stamp-kernel-note-l63
  *out = (static_cast<int64_t>(kKinds * kMaxTerms * DecodePlan<Bt::build>::kWarps + kUnitBlockWarps
                               + kFieldWidthMax)
              * 4096
          + static_cast<int64_t>(sizeof(DecodeParams)))
             * 1000003
         + static_cast<int64_t>(sizeof(Unit<4>)) * 101 + static_cast<int64_t>(kDecodeMaxSpan);
  g_decode_stamp_probe = *out;
}

}  // namespace stampdet

//: A VISIT'S PARTIAL INTO ITS HEAD, by warp 0: the CTA's fold summed a column a lane and added once, the -- see docs/internals/decode/decode.md#queue-finish
template <class Dp, int DV>
__device__ __forceinline__ void add_visit(const DecodeParams& p, const float* fold, int bh,
                                          int blocks, int* finished, int lane) {
  constexpr int cols = DV + 1;
  int positions_walked = 0;
#pragma unroll 1
  for (int wq = 0; wq < Dp::kWarps; ++wq) {
    positions_walked += __float_as_int(fold[wq * (cols + 3) + DV + 1]);
  }
  if (positions_walked == 0) return;
  float* const sums = p.yacc + static_cast<int64_t>(bh) * cols;
#pragma unroll 1
  for (int c = lane; c <= DV; c += 32) {
    float part = 0.0f;
#pragma unroll 1
    for (int wq = 0; wq < Dp::kWarps; ++wq) part += fold[wq * (cols + 3) + c];
    atomicAdd(sums + c, part);
  }
  __threadfence();
  __syncwarp();
  int last = 0;
  if (lane == 0) last = atomicAdd(finished, positions_walked) + positions_walked == blocks ? 1 : 0;
  last = __shfl_sync(0xffffffffu, last, 0);
  if (last == 0) return;
  //: THE HEAD'S LAST POSITIONS FINISH IT: y from the sums, which the finisher zeroes for the next step.
  __threadfence();
  const float den = __ldcg(sums + DV) + p.eps;
#pragma unroll 1
  for (int c = lane; c < DV; c += 32) {
    p.y[static_cast<int64_t>(bh) * DV + c] = __ldcg(sums + c) / den;
    sums[c] = 0.0f;
  }
  if (lane == 0) sums[DV] = 0.0f;
}

//: THE STEP at build `Bt::build` (a type: docs/internals/common/hardware.md#build-tag), at its plan's threads.
struct RowSums;
struct ListCounts;
struct DenseCounts;
struct TermFields;
struct TermUnits;
struct Blocks;
struct Next;
struct Last;
struct TermFirst;
struct ClockRows;
struct ValueRow;
struct FoldRows;
struct StepWords;

//: THE STEP'S SHARED MEMORY: the control state a field a tile, the clock's rows, the value and a fold row a warp, and what is left for the call's staged rows and tables -- see docs/internals/decode/decode.md#smem
template <class Bt, int DV>
struct DecodeTiles {
  using Dp = DecodePlan<Bt::build>;
  static constexpr int kClock = Dp::kWarps * PhaseClock<kDecodePhases>::kRow;
  static constexpr int kFold = Dp::kWarps * (DV + 4);
  static constexpr int kFields = kMaxTerms * kMaxLevels;
  static constexpr int kLists = kKinds * kMaxLevels;
};

template <class Bt, int DV, class Tl = DecodeTiles<Bt, DV>>
using DecodeSmem = SmemLayout<
    Bt, Tl::Dp::kBlocksPerSm, Fixed<RowSums, float, kMaxLevels>, Fixed<ListCounts, int, Tl::kLists>,
    Fixed<DenseCounts, int, kMaxLevels>, Fixed<TermFields, int, Tl::kFields>,
    Fixed<TermUnits, int, kMaxTerms>, One<Blocks, int>, One<Next, int>, One<Last, int>,
    Fixed<TermFirst, int, kMaxTerms + 1>, Fixed<ClockRows, long long, Tl::kClock>,
    Fixed<ValueRow, float, DV>, Fixed<FoldRows, float, Tl::kFold>, Rest<StepWords, std::uint32_t>>;

template <class Bt, int DV, int D, bool DECAY>
__global__
__launch_bounds__(DecodePlan<Bt::build>::kThreads, DecodePlan<Bt::build>::kBlocksPerSm) void
decode_step_kernel(DecodeParams p) {
  using Dp = DecodePlan<Bt::build>;
  //: The mass column is UNCONDITIONAL (raw is dead), which is why the shape is a -- see docs/internals/decode/decode.md#cols
  constexpr int cols = DV + 1;
  //: THE STATE PAGE: s24 planes at their group exponents, and the mass.
  using Page = fsla::state_page::Blocks<DV>;
  using Lanes = LeafLanes<DV>;

  using Smem = DecodeSmem<Bt, DV>;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int total_rows = p.total_rows;

  // ---- SMEM ledger ----------------------------------------------------------
  // sa_r, sa_w [total_rows] fp32 both sides' amplitude rows ...
  // see docs/internals/decode/decode.md#decode-step-kernel-note-l95
  float* sa_r = reinterpret_cast<float*>(Smem::template get<StepWords>());
  float* sa_w = sa_r + total_rows;
  float* sd = sa_w + total_rows;
  float* v_tok = Smem::template get<ValueRow>();
  float* fold = Smem::template get<FoldRows>();
  uint32_t* mask_r = reinterpret_cast<uint32_t*>(sd + (DECAY ? total_rows : 0));
  uint32_t* mask_w = mask_r + p.mask_total;
  int32_t* olist = reinterpret_cast<int32_t*>(mask_w + p.mask_total);
  int32_t* dlist = olist + kKinds * p.olist_total;

  float* const s_sum = Smem::template get<RowSums>();
  int* const s_n_o = Smem::template get<ListCounts>();
  int* const s_n_d = Smem::template get<DenseCounts>();
  //: THE TERM TABLE, ONE PACKED WORD PER (term, level): the field's position, its -- see docs/internals/decode/decode.md#decode-step-kernel-note-l137
  int* const s_fld = Smem::template get<TermFields>();
  //: One unit count per TERM. The walk's outer loop is the term, so each term's rank space -- see docs/internals/decode/decode.md#near-line-122
  int* const s_units = Smem::template get<TermUnits>();
  //: THE HEAD'S POSITION COUNT as this CTA holds it. -- see docs/internals/decode/decode.md#queue-positions
  int& s_blocks = *Smem::template get<Blocks>();
  //: The head the CTA visits next, as warp 0 chose, and at the exit whether this CTA is the last out. -- see docs/internals/decode/decode.md#queue-next
  int& s_next = *Smem::template get<Next>();
  int& s_last = *Smem::template get<Last>();
  //: A HEAD'S TERMS' FIRST POSITIONS, and past the last its position count. -- see docs/internals/decode/decode.md#queue-positions
  int* const s_first = Smem::template get<TermFirst>();
  using ClockRow = long long[PhaseClock<kDecodePhases>::kRow];
  ClockRow* const s_clock = reinterpret_cast<ClockRow*>(Smem::template get<ClockRows>());

  PhaseClock<kDecodePhases> clock(&s_clock[warp][0], lane);

  //: THE QUEUE, per batch-head: positions claimed, positions finished, the position count (-1 until known) -- see docs/internals/decode/decode.md#queue
  auto q_claim = [&](int i) -> int& { return p.queue[i]; };
  auto q_finished = [&](int i) -> int& { return p.queue[p.BH + i]; };
  auto q_blocks = [&](int i) -> int& { return p.queue[2 * p.BH + i]; };

  int* const q_cursor = p.queue + 3 * p.BH;
  int* const q_exit = q_cursor + 1;

  //: THE LANE'S GROUP AND CHANNELS: group `grp` walks its own unit; the lane holds channels -- see docs/internals/decode/decode.md#leaf-lanes
  const int grp = lane / Lanes::kLanes;
  const int ch0 = 8 * (lane % Lanes::kLanes);
  const bool row_lead = ch0 == 0;
  const bool exp_lead = (ch0 & 31) == 0;
  //: A CTA'S FIRST HEAD is its index modulo the heads; a head past the grid comes from the cursor. -- see docs/internals/decode/decode.md#queue-next
  int bh = static_cast<int>(blockIdx.x) % p.BH;

#pragma unroll 1
  while (bh >= 0) {
    const int b = bh / p.H;
    const int h = bh - b * p.H;

    // ---- THE PROLOGUE: three phases, three barriers ------------------------------
    //: PHASE A, EVERY OPERAND'S LOAD AT ONCE: a value channel, the dials and the digits of both sides -- see docs/internals/decode/decode.md#prologue
    const ValueUnit v_c =
        tid < DV ? load_value(p, b, h, tid) : ValueUnit{__float2bfloat16(0.0f), 0.0f};
    if (DECAY) {
#pragma unroll 1
      for (int i = tid; i < total_rows; i += Dp::kThreads) {
        int l = 0;
#pragma unroll
        for (int q = 1; q < D; ++q) {
          if (i >= p.level_amp_offset[q]) l = q;
        }
        sd[i] = p.dials[static_cast<int64_t>(h) * total_rows + p.level_row_offset[l]
                        + (i - p.level_amp_offset[l])];
      }
    }
    fold_digits<Dp, D>(p, b, h, sa_r, sa_w, mask_r, mask_w);
    if (tid < DV) v_tok[tid] = value_of(v_c);
    __syncthreads();
    clock.lap(kPhFold);

    //: PHASE B, FROM THE DIGIT MASKS, one warp a job: the owner and digit lists with their counts, and a decay step's write row sums -- see docs/internals/decode/decode.md#prologue
    build_lists<D>(p, mask_r, mask_w, olist, dlist, s_n_o, s_n_d);
    if (DECAY) sum_staged_rows(p, sa_w, s_sum, 2 * D - 1);
    __syncthreads();
    //: PHASE C: the decay's share scale and the term layout (whose barrier ends it).
    const float share = DECAY ? share_scale(p, s_sum) : 0.0f;
    publish_term_layout<D>(p, s_n_o, s_n_d, s_fld, s_units);
    clock.lap(kPhFactor);

    //: THE HEAD'S POSITIONS: a warp's share of each term in turn -- a unit of a whole-atom write term, a batch -- see docs/internals/decode/decode.md#queue-positions
    if (tid == 0) {
      int at = 0;
#pragma unroll
      for (int term = 0; term <= D; ++term) {
        s_first[term] = at;
        at +=
            (term == 0 && p.write_atoms) ? s_units[0] : ceil_div_int(s_units[term], Lanes::kGroups);
      }
      s_first[D + 1] = at;
    }

    //: THE HEAD'S POSITION COUNT, published for the stealers: every visitor derives the same one (an empty head has one position, which walks nothing).
    if (tid == 0) {
      s_blocks = s_first[D + 1] > 0 ? s_first[D + 1] : 1;
      q_blocks(bh) = s_blocks;
    }
    __syncthreads();
    const int blocks = s_blocks;

    //: THE WARP'S OWN PARTIAL for this head, over every position it claims.
    float acc[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) acc[c] = 0.0f;
    float acc_m = 0.0f;
    int walked = 0;

    //: THE BATCH-HEAD'S PLANE: an atom's page is its own place in it, plain arithmetic at each use.
    uint8_t* const plane =
        p.state + static_cast<int64_t>(bh) * p.atoms_per_bh * static_cast<int64_t>(Page::kBytes);

    // ---- THE WALK: a warp claims a position, walks it, claims again -------------------
    //: @barrier-free the walk: a lane group's read and write-back are one statement in program order; see docs/internals/decode/decode.md#no-barrier
    //: THE NEXT CLAIM IS ISSUED BEFORE THE CURRENT POSITION IS WALKED, so its round trip hides behind the walk. -- see docs/internals/decode/decode.md#queue-claim
    int claim = 0;
    if (lane == 0) claim = atomicAdd(&q_claim(bh), 1);
#pragma unroll 1
    while (true) {
      const int pos = __shfl_sync(0xffffffffu, claim, 0);
      if (pos >= blocks) break;
      if (lane == 0) claim = atomicAdd(&q_claim(bh), 1);
      ++walked;
      //: THE POSITION'S TERM: the last term whose first position it reaches (an empty head's one position walks nothing).
      int term = -1;
#pragma unroll
      for (int t = 0; t <= D; ++t) {
        if (pos >= s_first[t] && pos < s_first[t + 1]) term = t;
      }
      if (term < 0) continue;
      const int local = pos - s_first[term];
      //: THE TERM'S ROW OF THE PACKED TABLE: every level's -- see docs/internals/decode/decode.md#units-2
      const int units = s_units[term];
      const int32_t* fld = s_fld + term * kMaxLevels;
      //: TERM 0 IS THE WRITE PRODUCT: every leaf it walks is written, and no other term writes. -- see docs/internals/decode/decode_lattice.md#the-terms
      const bool writes = term == 0;

      if (writes && p.write_atoms) {
        //: THE WRITE TERM BY WHOLE ATOMS: a warp a unit, its lane groups a quarter of the atom's leaves each, every -- see docs/internals/decode/decode.md#whole-atoms
        constexpr int kRows = kAtomLeaves / Lanes::kGroups;
        const Unit<D> u = resolve_write_unit<D, DECAY>(p, mask_r, mask_w, sa_r, sa_w, sd, olist,
                                                       dlist, fld, local);
        if (u.cw == 0u) continue;
        const uint32_t br = u.live_r ? u.cr : 0u;
#pragma unroll 1
        for (int a = 0; a < (1 << p.atoms_per_unit_shift); ++a) {
          const int atom = (u.base >> kAtomShift) + a;
          uint8_t* const page = plane + atom * static_cast<int64_t>(Page::kBytes);
          //: ONE ADDRESS AN ATOM IS HELD: the group's first leaf's; a later leaf's is a constant past it
          const LeafAddr first = leaf_addr<DV>(page, grp, ch0);
          LeafSlice slice[kRows];
          //: EVERY LEAF'S LOADS ARE ISSUED HERE, before any leaf's arithmetic: each sits under a predicate only the launch knows (a plane has pages), so the assembler keeps it in place -- see docs/internals/decode/decode.md#atom-loads
          const bool resident = p.atoms_per_bh != 0;
#pragma unroll
          for (int k = 0; k < kRows; ++k) {
            slice[k] =
                load_leaf(leaf_after<DV>(first, k * Lanes::kGroups), resident, true, row_lead);
          }
#pragma unroll
          for (int k = 0; k < kRows; ++k) {
            const int t = (a << kAtomShift) + k * Lanes::kGroups + grp;
            //: ONLY THE WRITE SUPPORT IS THIS TERM'S: a leaf outside it belongs to a read term, which reads it.
            const bool mine = ((u.cw >> t) & 1u) != 0u;
            const bool in_r = ((br >> t) & 1u) != 0u;
            const float ar = (mine && in_r) ? u.ar * sa_r[u.dlast + t] : 0.0f;
            const float aw = mine ? u.aw * sa_w[u.dlast + t] : 0.0f;
            const float rate = (DECAY && mine) ? u.rate * sd[u.dlast + t] : 0.0f;
            deposit_leaf<DECAY>(slice[k], leaf_after<DV>(first, k * Lanes::kGroups), ar, aw, rate,
                                share, v_tok, ch0, mine, row_lead, exp_lead, acc, acc_m);
          }
        }
        continue;
      }

      const int unit = local * Lanes::kGroups + grp;
      const bool has_unit = unit < units;
      //: TWO RESOLVERS, ONE WALK. -- see docs/internals/decode/decode.md#u
      const int named = has_unit ? unit : 0;
      const Unit<D> u = writes ? resolve_write_unit<D, DECAY>(p, mask_r, mask_w, sa_r, sa_w, sd,
                                                              olist, dlist, fld, named)
                               : resolve_unit<D, DECAY>(p, mask_r, mask_w, sa_r, sa_w, sd, olist,
                                                        fld, term, named);
      const uint32_t br = (has_unit && u.live_r) ? u.cr : 0u;
      const uint32_t bw = (has_unit && u.live_w) ? u.cw : 0u;
      //: THE ENUMERATED RUN IS THIS TERM'S. Term 0 walks the write run; every other -- see docs/internals/decode/decode.md#decode-step-kernel-note-l474
      uint32_t bits = writes ? bw : br;

      //: THE SLOT LOOP IS THE INNERMOST LEVEL'S RUN, in ASCENDING lattice order, one leaf a group -- see docs/internals/decode/decode.md#t
#pragma unroll 1
      while (__any_sync(0xffffffffu, bits != 0u)) {
        const bool live = bits != 0u;
        const int t = live ? __ffs(static_cast<int>(bits)) - 1 : 0;
        bits &= bits - 1u;
        const int off_last = u.dlast + t;
        const bool in_r = ((br >> t) & 1u) != 0u;

        const int leaf = u.base + t;
        const int atom = leaf >> kAtomShift;
        const bool present = live;
        const float ar = (present && in_r) ? u.ar * sa_r[off_last] : 0.0f;
        //: A LEAF THIS STEP ONLY READS PULLS THE hi PLANE ALONE; a WRITTEN one both planes -- see docs/internals/decode/decode.md#int8-state
        uint8_t* const page = plane + atom * static_cast<int64_t>(Page::kBytes);
        const LeafAddr addr = leaf_addr<DV>(page, leaf & kAtomMask, ch0);
        const LeafSlice slice = load_leaf(addr, present, writes, row_lead);
        if (!writes) {
          read_hi(slice, ar, row_lead, acc, acc_m);
        } else {
          //: UPDATE FIRST, THEN READ THE STORED ROW -- AND THAT ORDER IS THE SEMANTICS, -- see docs/internals/decode/decode.md#decode-step-kernel-note-l532
          const float aw = present ? u.aw * sa_w[off_last] : 0.0f;
          const float rate = DECAY ? u.rate * sd[off_last] : 0.0f;
          deposit_leaf<DECAY>(slice, addr, ar, aw, rate, share, v_tok, ch0, present, row_lead,
                              exp_lead, acc, acc_m);
        }
      }
    }
    //: @barrier-free-end
    clock.lap(kPhWalk);

    // ---- THE READOUT: the CTA folds its warps' partials, adds them into the head's sums once, and the ------
    // ---- visit that finishes the head's positions finishes the head ------------------------------------------
    //: THE GROUPS' PARTIALS, summed across the warp by a fixed butterfly: every lane of a channel slice ends -- see docs/internals/decode/decode.md#leaf-lanes
#pragma unroll
    for (int off = Lanes::kLanes; off < 32; off <<= 1) {
#pragma unroll
      for (int c = 0; c < 8; ++c) acc[c] += __shfl_xor_sync(0xffffffffu, acc[c], off);
      acc_m += __shfl_xor_sync(0xffffffffu, acc_m, off);
    }
    //: THE CTA'S FOLD: a warp's row of partials and its position count, summed by warp 0 -- one set of -- see docs/internals/decode/decode.md#queue-finish
    float* const my_row = fold + warp * (cols + 3);
    if (grp == 0) {
#pragma unroll
      for (int c = 0; c < 8; ++c) my_row[ch0 + c] = acc[c];
    }
    if (lane == 0) {
      my_row[DV] = acc_m;
      my_row[DV + 1] = __int_as_float(walked);
    }
    __syncthreads();
    if (warp == 0) {
      add_visit<Dp, DV>(p, fold, bh, blocks, &q_finished(bh), lane);
    }
    clock.lap(kPhCombine);

    // ---- THE NEXT HEAD: an unstarted one, else the one with the most unclaimed positions --------------
    //: every warp is done with this head's tables before the next head's prologue overwrites them.
    __syncthreads();
    if (warp == 0) {
      int next = -1;
      const int started = p.ctas < p.BH ? p.ctas : p.BH;
      if (lane == 0 && started < p.BH) {
        const int c = atomicAdd(q_cursor, 1) + started;
        if (c < p.BH) next = c;
      }
      next = __shfl_sync(0xffffffffu, next, 0);
      if (next < 0) {
        //: STEALING: the head with the most positions left, among those whose count is published. -- see docs/internals/decode/decode.md#queue-next
        int best = -1;
        int most = 0;
#pragma unroll 1
        for (int base = 0; base < p.BH; base += 32) {
          const int cand = base + lane;
          int left = 0;
          if (cand < p.BH) {
            const int count = __ldcg(&q_blocks(cand));
            if (count > 0) left = count - __ldcg(&q_claim(cand));
          }
          int top = left;
          int who = cand;
#pragma unroll
          for (int off = 16; off > 0; off >>= 1) {
            const int t2 = __shfl_xor_sync(0xffffffffu, top, off);
            const int w2 = __shfl_xor_sync(0xffffffffu, who, off);
            if (t2 > top || (t2 == top && w2 < who)) {
              top = t2;
              who = w2;
            }
          }
          if (top > most) {
            most = top;
            best = who;
          }
        }
        //: A STEAL PAYS A PROLOGUE, so it takes a head only with a position left for every warp.
        if (most >= Dp::kWarps) {
          next = best;
        }
      }
      if (lane == 0) s_next = next;
    }
    __syncthreads();
    bh = s_next;
  }

  // ---- THE EXIT: the last CTA out resets the queue for the next step ----------------------
  //: No memset launch stands between two steps, so the step stays capturable. -- see docs/internals/decode/decode.md#queue
  clock.publish(p.ledger, static_cast<long>(blockIdx.x), Dp::kWarps, warp);
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = atomicAdd(q_exit, 1) == p.ctas - 1 ? 1 : 0;
  __syncthreads();
  if (s_last != 0) {
    __threadfence();
#pragma unroll 1
    for (int i = tid; i < p.BH; i += Dp::kThreads) {
      q_claim(i) = 0;
      q_finished(i) = 0;
      q_blocks(i) = -1;
    }
    if (tid == 0) {
      *q_cursor = 0;
      *q_exit = 0;
    }
  }
}

}  // namespace detail

//: THE DECODE ARM SET, defined over the generated selection header: the key `(d_v, levels, decay)` -- see docs/internals/common/arm_switch.md#arm-set
FSLA_DEFINE_ARM_SET(DecodeArmSet, DECODE);

//: THE BUILT SET, READ BACK OFF THE SAME SET: `[d_v, D, decay]` per arm, in the order -- see docs/internals/decode/decode.md#fsladecodearms
std::vector<std::vector<int64_t>> fsla_decode_arms() { return DecodeArmSet::rows(); }

//: THE BOUND PHASE LEDGER, or null: the next steps book into it; see decode.md#phase-ledger
long long* g_phase_ledger = nullptr;
int64_t g_phase_ledger_ctas = 0;

void fsla_decode_ledger_bind(const std::optional<Tensor>& ledger) {
  if (!ledger) {
    g_phase_ledger = nullptr;
    g_phase_ledger_ctas = 0;
    return;
  }
  const int warps = detail::step_warps(card_of(ledger->get_device_index()));
  STD_TORCH_CHECK(ledger->is_cuda() && ledger->scalar_type() == Dtype::Long && ledger->dim() == 3
                      && ledger->size(1) == warps && ledger->size(2) == kDecodePhases
                      && ledger->is_contiguous(),
                  "the decode phase ledger is a contiguous CUDA int64 tensor [ctas][", warps, "][",
                  kDecodePhases, "]");
  g_phase_ledger = reinterpret_cast<long long*>(ledger->mutable_data_ptr<int64_t>());
  g_phase_ledger_ctas = ledger->size(0);
}

void launch_decode_step(const Card& card, const plan::DecodeLaunch& launch, const DecodeParams& p) {
  const plan::DecodeArm& arm = launch.arm;
  const plan::DecodeArgs& args = launch.args;
  auto stream = current_stream();

  //: ONE LAUNCH SITE. Each axis arrives as an `integral_constant`, so the three -- see docs/internals/decode/decode.md#launch
  auto enqueue = [&](auto DV, auto LEVELS, auto DECAY) {
    constexpr int kDv = decltype(DV)::value, kLevels = decltype(LEVELS)::value;
    constexpr bool kDecay = decltype(DECAY)::value;
    build_switch(card, [&](auto b) {
      constexpr ::denseref::arch::Build kB = decltype(b)::build;
      if constexpr (in_image(kB)) {
        STD_TORCH_CHECK(args.threads == DecodePlan<kB>::kThreads, "the descriptor's CTA is ",
                        args.threads, " threads, the step's compiled plan ",
                        DecodePlan<kB>::kThreads);
        auto kernel =
            &detail::decode_step_kernel<::denseref::arch::BuildTag<kB>, kDv, kLevels, kDecay>;
        //: the step's shared memory is the kernel's layout on this card; the launch reserves it up to the call's last word -- see docs/internals/decode/decode.md#smem
        using Smem = detail::DecodeSmem<::denseref::arch::BuildTag<kB>, kDv>;
        constexpr std::int64_t kStep = Smem::template count<detail::StepWords>;
        const std::int64_t words = launch.layout.step_words;
        STD_TORCH_CHECK(words <= kStep, plan::decode_smem_refusal(launch, kStep));
        const int smem = static_cast<int>(Smem::template reserved_to<detail::StepWords>(words));

        //: the kernel's ceiling is the layout's bytes, settled once a device; a launch reserves `smem` of it -- see docs/internals/common/smem_layout.md#reserve
        static std::mutex mu;
        static std::set<int> settled;
        {
          const std::lock_guard<std::mutex> hold(mu);
          if (settled.count(card.device) == 0) {
            constexpr int kMost = static_cast<int>(Smem::bytes);
            FSLA_CUDA_CHECK(
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kMost));
            settled.insert(card.device);
          }
        }
        kernel<<<dim3(p.ctas), args.threads, smem, stream>>>(p);
      }
    });
  };
  arm_switch<DecodeArmSet>({{arm.d_v, arm.levels, arm.decay}}, [&](auto built) {
    constexpr int kDvArm = decltype(built)::values[0], kLevelsArm = decltype(built)::values[1];
    constexpr bool kDecayArm = decltype(built)::values[2] != 0;
    enqueue(std::integral_constant<int, kDvArm>{}, std::integral_constant<int, kLevelsArm>{},
            std::integral_constant<bool, kDecayArm>{});
  });
  FSLA_CUDA_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Host entry
// ---------------------------------------------------------------------------

namespace detail {

void check_f32(const Tensor& t, const char* name) {
  STD_TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  STD_TORCH_CHECK(t.scalar_type() == Dtype::Float, name, " must be float32, got ", t.scalar_type());
  STD_TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

//: an fp32 operand the step reads by address, strided as the producer left it.
void check_f32_span(const Tensor& t, const char* name) {
  STD_TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  STD_TORCH_CHECK(t.scalar_type() == Dtype::Float, name, " must be float32, got ", t.scalar_type());
}

void check_i32(const Tensor& t, const char* name) {
  STD_TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  STD_TORCH_CHECK(t.scalar_type() == Dtype::Int, name, " must be int32, got ", t.scalar_type());
  STD_TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

//: The `[B, 1, H, W] -> [B * H, W]` fold as an ADDRESS. `T != 1` is refused -- the fold -- see docs/internals/decode/decode.md#spanof
Span span_of(const Tensor& t, int64_t B, int64_t H, int64_t width, const char* name) {
  STD_TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  STD_TORCH_CHECK(t.dim() == 4 && t.size(1) == 1, name, " must be [B, 1, H, W], got ",
                  shape(t.sizes()));
  STD_TORCH_CHECK(t.size(0) == B && t.size(2) == H, name, " is ", shape(t.sizes()),
                  " against the value's B = ", B, ", H = ", H);
  STD_TORCH_CHECK(width == 0 || t.size(3) == width, name, " is ", t.size(3), " wide against ",
                  width);
  return Span{t.data_ptr(), t.stride(0), t.stride(2), t.stride(3)};
}

Span scalar_span_of(const Tensor& t, int64_t B, int64_t H, const char* name) {
  STD_TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  STD_TORCH_CHECK(t.dim() == 3 && t.size(1) == 1, name, " must be [B, 1, H], got ",
                  shape(t.sizes()));
  STD_TORCH_CHECK(t.size(0) == B && t.size(2) == H, name, " is ", shape(t.sizes()),
                  " against the value's B = ", B, ", H = ", H);
  return Span{t.data_ptr(), t.stride(0), t.stride(2), 0};
}

//: THE STEP'S WARPS at `card`'s build, for the host.
int step_warps(const Card& card) {
  int warps = 0;
  build_switch(card, [&](auto b) { warps = DecodePlan<decltype(b)::build>::kWarps; });
  return warps;
}

//: THE LAYOUT INTO THE PARAMS: the launch's layout (`plan::DecodeLayout`) copied field for field; nothing is derived here -- see docs/internals/decode/decode.md#fill-lattice
void fill_lattice(DecodeParams& p, const plan::DecodeLayout& layout) {
  p.lat_k = layout.lat_k;
  for (int l = 0; l < kMaxLevels; ++l) {
    p.lat_s[l] = layout.lat_s[l];
    p.lat_g[l] = layout.lat_g[l];
    p.lat_gsuf[l] = layout.lat_gsuf[l];
    p.lat_sbits[l] = layout.lat_sbits[l];
    p.lat_run_shift[l] = layout.lat_run_shift[l];
    p.lat_oshift[l] = layout.lat_oshift[l];
    p.mask_off[l] = layout.mask_off[l];
    p.mask_words[l] = layout.mask_words[l];
    p.olist_off[l] = layout.olist_off[l];
    p.dlist_off[l] = layout.dlist_off[l];
  }
  p.lat_local_bits = layout.lat_local_bits;
  p.lat_inner_bits = layout.lat_inner_bits;
  p.lat_outer_bits = layout.lat_outer_bits;
  p.mask_total = layout.mask_total;
  p.olist_total = layout.olist_total;
  p.dlist_total = layout.dlist_total;
  p.atoms_per_unit_shift = layout.atoms_per_unit_shift;
  p.write_atoms = layout.write_atoms;
}

//: The per-level geometry every decode entry validates the same way, filled into `p` and -- see docs/internals/decode/decode.md#filllevels
int fill_levels(DecodeParams& p, const std::vector<Tensor>& levels,
                const std::vector<int64_t>& level_widths, int64_t B, int64_t H, int64_t N) {
  int total_rows = 0;
  int64_t expect_n = 1;
  for (int l = 0; l < p.D; ++l) {
    const int width = static_cast<int>(level_widths[l]);
    p.write[l] = span_of(levels[l], B, H, width, "write[l]");
    p.level_width[l] = width;
    p.level_amp_offset[l] = total_rows;
    total_rows += width;
    expect_n *= width;
  }
  STD_TORCH_CHECK(expect_n == N, "prod(level_widths) = ", expect_n,
                  " disagrees with the state's leaf axis ", N);
  return total_rows;
}

}  // namespace detail

static_assert(plan::kLatticeAtomLeaves == kAtomLeaves && plan::kLatticeAtomShift == kAtomShift,
              "the launch's lattice law and the shared geometry state the atom's leaves alike");

int64_t fsla_decode_capacity() { return kDecodeMaxLevelWidth; }

int64_t fsla_decode_ledger_warps() {
  check_arch_table();
  return detail::step_warps(current_card());
}

//: The MIRRORED copy of the producer's `MAX_BRANCH_WIDTH`, bound separately -- see docs/internals/decode/decode.md#fsla-decode-producer-width-mirror
int64_t fsla_decode_producer_width_mirror() { return kProducerMaxBranchWidth; }

//: THE BUILD STAMP: a path or hash check cannot catch a stale `.so` at the right path, -- see docs/internals/decode/decode.md#fsladecodebuildstamp
int64_t fsla_decode_build_stamp() {
  auto out = empty_cuda({1}, Dtype::Long);
  build_switch(current_card(), [&](auto b) {
    constexpr ::denseref::arch::Build kB = decltype(b)::build;
    if constexpr (in_image(kB))
      detail::stampdet::decode_stamp_kernel<::denseref::arch::BuildTag<kB>>
          <<<1, 1, 0, current_stream()>>>(out.mutable_data_ptr<int64_t>());
  });
  FSLA_CUDA_LAUNCH_CHECK();
  return item<int64_t>(out);
}

Tensor fsla_decode_forward(std::vector<Tensor> read, std::vector<Tensor> write,
                           std::vector<Tensor> read_scale, std::vector<Tensor> write_scale,
                           std::vector<Tensor> read_mask, std::vector<Tensor> write_mask,
                           std::optional<Tensor> dials, Tensor v, Tensor v_scale, Tensor state,
                           Tensor yacc, Tensor queue, int64_t ctas, double eps,
                           std::vector<int64_t> descriptor) {
  check_arch_table();
  const int D = static_cast<int>(read.size());
  STD_TORCH_CHECK(D >= 1 && D <= kMaxLevels, "D must be in [1, 4], got ", D);
  STD_TORCH_CHECK(static_cast<int>(write.size()) == D, "read and write must each have D entries");

  STD_TORCH_CHECK(v.dim() == 4 && v.size(1) == 1, "v must be [B, 1, H, d_v], got ",
                  shape(v.sizes()));
  const int B = static_cast<int>(v.size(0));
  const int H = static_cast<int>(v.size(2));
  const int BH = B * H;
  const int d_v = static_cast<int>(v.size(3));
  const int cols = d_v + 1;  // the mass column is unconditional (raw is dead)

  //: THE DESCRIPTOR'S LAUNCH: the arm and its arguments from the call's facts, the one function; the call's operands must be the shape it says
  const Card& card = card_of(v.get_device_index());
  const plan::DecodeLaunch launch = plan::decode_launch(
      plan::checked_descriptor<plan::decode::Kernel>(descriptor, v.get_device_index()));
  STD_TORCH_CHECK(d_v == launch.arm.d_v, "v is ", d_v,
                  " channels wide; the descriptor's arm serves ", launch.arm.d_v,
                  " (the call's value width padded by ", launch.args.pad, ")");
  STD_TORCH_CHECK(D == launch.arm.levels, "the call has ", D,
                  " levels; the descriptor's arm serves ", launch.arm.levels);
  const std::vector<int64_t> level_widths(launch.args.widths, launch.args.widths + D);
  const std::vector<int64_t> level_row_offsets(launch.args.row_offsets,
                                               launch.args.row_offsets + D);

  STD_TORCH_CHECK(state.is_cuda() && state.scalar_type() == Dtype::Byte && state.is_contiguous(),
                  "state is the int8 state format's pages, a contiguous CUDA uint8 tensor; got ",
                  state.scalar_type());
  const int64_t page_bytes =
      d_v == 32 ? state_page::Blocks<32>::kBytes : state_page::Blocks<64>::kBytes;
  detail::check_f32(yacc, "yacc");
  detail::check_i32(queue, "queue");

  //: THE STATE'S PLANE: a page an atom, a batch-head's pages in atom order. -- see docs/internals/decode/decode.md#state
  STD_TORCH_CHECK(state.dim() == 3 && state.size(0) == BH && state.size(2) == page_bytes,
                  "the state is [BH, N / ", kAtomLeaves, ", ", page_bytes, "] bytes");
  int64_t N = 1;
  for (auto w : level_widths) N *= w;
  const int atoms_per_bh = static_cast<int>(N / kAtomLeaves);
  STD_TORCH_CHECK(state.size(1) == atoms_per_bh, "the state holds N / ", kAtomLeaves, " = ",
                  atoms_per_bh, " pages a batch-head; got ", state.size(1));

  DecodeParams p{};
  p.D = D;
  const int total_rows = detail::fill_levels(p, write, level_widths, B, H, N);
  detail::fill_lattice(p, launch.layout);
  //: THE COEFFICIENT ROWS: u8 at every level. -- see docs/internals/decode/decode.md#coefficient-rows
  STD_TORCH_CHECK(
      read_scale.size() == static_cast<size_t>(D) && write_scale.size() == static_cast<size_t>(D),
      "one row scale a level a side");
  for (int l = 0; l < D; ++l) {
    p.read[l] = detail::span_of(read[l], B, H, p.level_width[l], "read[l]");
    STD_TORCH_CHECK(read[l].scalar_type() == Dtype::Byte && write[l].scalar_type() == Dtype::Byte,
                    "the coefficient rows are u8; level ", l, " is ", read[l].scalar_type(),
                    " and ", write[l].scalar_type());
    detail::check_f32_span(read_scale[l], "read_scale[l]");
    detail::check_f32_span(write_scale[l], "write_scale[l]");
    p.read_scale[l] = detail::scalar_span_of(read_scale[l], B, H, "read_scale[l]");
    p.write_scale[l] = detail::scalar_span_of(write_scale[l], B, H, "write_scale[l]");
    p.level_row_offset[l] = static_cast<int>(level_row_offsets[l]);
  }
  //: THE MASKS: int32 words, DIGIT-packed, one token, `[B, H, ceil(width_l / 32)]` a side a level -- bit `d % 32` of word
  //: `d // 32` is digit `d` live, contiguous: `fold_digits` reads them; a digit whose bit is off contributes nothing.
  STD_TORCH_CHECK(
      read_mask.size() == static_cast<size_t>(D) && write_mask.size() == static_cast<size_t>(D),
      "one digit mask a level a side");
  for (int l = 0; l < D; ++l) {
    const int64_t words = (level_widths[l] + 31) / 32;
    for (const Tensor* m : {&read_mask[l], &write_mask[l]}) {
      detail::check_i32(*m, "mask[l]");
      STD_TORCH_CHECK(m->dim() == 3 && m->size(0) == B && m->size(1) == H && m->size(2) == words,
                      "mask[l] must be [B, H, ceil(width_l / 32)] int32, got ", shape(m->sizes()));
    }
    p.read_mask[l] = read_mask[l].mutable_data_ptr<int32_t>();
    p.write_mask[l] = write_mask[l].mutable_data_ptr<int32_t>();
  }
  //: THE VALUE: bf16 against the token's largest channel, the fp32 token scale.
  STD_TORCH_CHECK(v.scalar_type() == Dtype::BFloat16, "v is bf16, got ", v.scalar_type());
  p.v = detail::span_of(v, B, H, d_v, "v");
  detail::check_f32_span(v_scale, "v_scale");
  p.v_scale = detail::scalar_span_of(v_scale, B, H, "v_scale");

  STD_TORCH_CHECK(ctas == -1 || ctas >= 1,
                  "ctas is -1 (the card's grid) or a gate's forced count of at least 1, got ",
                  ctas);
  STD_TORCH_CHECK(yacc.dim() == 2 && yacc.size(0) == BH && yacc.size(1) == cols,
                  "yacc must be [BH, cols]");
  STD_TORCH_CHECK(queue.dim() == 1 && queue.size(0) == 3 * static_cast<int64_t>(BH) + 2,
                  "queue must be [3 BH + 2] int32");

  const bool decay = dials.has_value();
  STD_TORCH_CHECK(decay == (launch.arm.decay != 0), "the call's dials are ",
                  decay ? "present" : "absent", "; the descriptor's arm is the ",
                  launch.arm.decay != 0 ? "decaying" : "non-decaying", " one");

  if (decay) {
    detail::check_f32(*dials, "dials");
    STD_TORCH_CHECK(dials->dim() == 2 && dials->size(0) == H && dials->size(1) == total_rows,
                    "dials must be [H, sum_l width_l]");
    p.dials = dials->mutable_data_ptr<float>();
  }

  //: `empty`, not `zeros`: every element is written, so a zero-fill is a launch the -- see docs/internals/decode/decode.md#y
  auto y = empty_cuda({BH, d_v}, Dtype::Float, v.get_device_index());
  p.state = state.mutable_data_ptr<uint8_t>();
  p.atoms_per_bh = atoms_per_bh;
  p.y = y.mutable_data_ptr<float>();
  p.yacc = yacc.mutable_data_ptr<float>();
  p.queue = queue.mutable_data_ptr<int32_t>();
  p.N = N;
  p.BH = BH;
  p.H = H;
  //: the step's grid: the card's (a unit a run of the innermost level's span), or a gate's forced count
  p.ctas = plan::decode_grid(launch.args, launch.layout.unit_ceiling, BH, ctas);
  p.total_rows = total_rows;
  p.eps = static_cast<float>(eps);
  p.ledger = g_phase_ledger;
  if (p.ledger != nullptr)
    STD_TORCH_CHECK(g_phase_ledger_ctas >= p.ctas, "the bound decode phase ledger covers ",
                    g_phase_ledger_ctas, " CTAs; this step launches ", p.ctas);

  launch_decode_step(card, launch, p);
  return y;
}

//: THE LAUNCH A DESCRIPTOR MEANS, for the host: the arm's `d_v`, levels and decay, the padding of the call's
//: value width, then the level widths (`kDecodeMaxLevels` of them, 0 beyond the call's levels)
std::vector<int64_t> fsla_decode_launch(std::vector<int64_t> descriptor, int64_t device) {
  const plan::DecodeDescriptor d =
      plan::checked_descriptor<plan::decode::Kernel>(descriptor, static_cast<int>(device));
  const auto launch = plan::decode_launch(d);
  const auto& arm = launch.arm;
  std::vector<int64_t> out = {arm.d_v, arm.levels, arm.decay, launch.args.pad};
  out.insert(out.end(), launch.args.widths, launch.args.widths + plan::kDecodeMaxLevels);
  return out;
}

namespace detail {

//: the level widths of a lattice call, each held to the decode path's capacity, as the law's `int` array
int lattice_levels(const std::vector<int64_t>& widths, int out[kMaxLevels]) {
  const int levels = static_cast<int>(widths.size());
  STD_TORCH_CHECK(levels >= 1 && levels <= kMaxLevels, "a lattice has 1 to ", kMaxLevels,
                  " levels, got ", levels);
  for (int l = 0; l < levels; ++l) {
    STD_TORCH_CHECK(widths[l] >= 1 && widths[l] <= kDecodeMaxLevelWidth, "level ", l, " of width ",
                    widths[l], " is outside 1 to the decode path's ", kDecodeMaxLevelWidth);
    out[l] = static_cast<int>(widths[l]);
  }
  return levels;
}

//: `[k, m, BC, owners, s_0 .. s_{D-1}, g_0 .. g_{D-1}]` of a box
std::vector<int64_t> lattice_words(const plan::LatticeBox<kMaxLevels>& box) {
  std::vector<int64_t> out = {box.k, box.m, box.bc, box.owners};
  out.insert(out.end(), box.s, box.s + box.levels);
  out.insert(out.end(), box.g, box.g + box.levels);
  return out;
}

}  // namespace detail

//: THE LATTICE the launch function solves for level widths, for the host: no device, a pure function of the widths
std::vector<int64_t> fsla_decode_lattice(std::vector<int64_t> widths) {
  int w[kMaxLevels] = {};
  const int levels = detail::lattice_levels(widths, w);
  return detail::lattice_words(plan::lattice_solve<kMaxLevels>(w, levels));
}

//: THE BOX of `(k, m)` on level widths, for the host: the law's box, refused in its words when none fits
std::vector<int64_t> fsla_decode_lattice_box(std::vector<int64_t> widths, int64_t k, int64_t m) {
  int w[kMaxLevels] = {};
  const int levels = detail::lattice_levels(widths, w);
  const auto box = plan::lattice_box<kMaxLevels>(w, levels, static_cast<int>(k), m);
  if (box.refusal != plan::BoxRefusal::kNone) STD_TORCH_CHECK(false, plan::lattice_refusal(box, w));
  return detail::lattice_words(box);
}

}  // namespace decode
}  // namespace fsla
